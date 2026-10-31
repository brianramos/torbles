/* PROV-001 on a CUDA device: a provenance DAG (an evaluation plan's cone) under many valuations at once, in the
 * Boolean semiring B or the counting semiring N, with exactly the outputs of tr_prov_evaluate (math_prov.h).
 * SPDX-License-Identifier: Apache-2.0
 *
 * One work item per valuation (a source deletion, a set of present sources, multiplicities): it evaluates the cone in
 * increasing node order with the shared tr_prov_eval_count / tr_prov_eval_bool into its own scratch row (one value
 * per cone node) and writes its own outputs and status, so nothing depends on the order of the threads. Valuations
 * run in chunks whose scratch stays within TR_PROV_DEVICE_SCRATCH bytes (one launch per chunk). The host checks the
 * plan view, the valuations and the budget (output = valuations x outputs, bytes = every device copy and the scratch
 * chunk) before allocating anything on the device. */
#include "math_cuda.h"
#include "math_prov.h"
#include "math_handles.h"

TR_MGPU_ABI(tr_prov_node, TR_PROV_ABI);           /* structs passed to the CUDA library (math_handles.h) */
TR_MGPU_ABI(tr_prov_valuations, TR_PROV_ABI);
TR_MGPU_ABI(tr_prov_plan_view, TR_PROV_ABI);
TR_MGPU_ABI(tr_prov_eval_info, TR_PROV_ABI);

TR_KERNEL void tr_prov_count_kernel(tr_prov_plan_view p, tr_prov_valuations v, uint64_t first, uint64_t count,
                                    unsigned __int128 *scratch, uint64_t *out, int32_t *status) {
    TR_GRID_LOOP(i, count) {
        uint64_t w = first + i;
        status[w] = tr_prov_eval_count(&p, &v, w, scratch + i * p.nodes, out + 2 * w * p.noutputs);
    }
}

TR_KERNEL void tr_prov_bool_kernel(tr_prov_plan_view p, tr_prov_valuations v, uint64_t first, uint64_t count,
                                   uint8_t *scratch, uint8_t *out, int32_t *status) {
    TR_GRID_LOOP(i, count) {
        uint64_t w = first + i;
        tr_prov_eval_bool(&p, &v, w, scratch + i * p.nodes, out + w * p.noutputs);
        status[w] = 0;
    }
}

/* tr_budget_check, refusing a size past 64 bits even under an unlimited budget */
static int tr_prov_gpu_check(const tr_math_budget *b, const tr_math_need *n, tr_math_refusal *r) {
    int rc = tr_budget_check(b, n, r);
    if (rc) return rc;
    for (int i = 0; i < TR_BUDGET_RESOURCES; i++)
        if (n->need[i] == TR_BUDGET_UNKNOWN) {
            if (r) { r->resource = i; r->refused = 1; r->need = n->need[i]; r->limit = b->limit[i]; }
            return TR_MATH_BUDGET_CODE;
        }
    return 0;
}

#ifdef __cplusplus
extern "C" {
#endif
int tr_mgpu_prov_evaluate(const tr_prov_plan_view *p, int32_t semiring, const tr_prov_valuations *v, void *out,
                          int32_t *status, tr_prov_eval_info *info, const tr_math_budget *budget,
                          tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!p || !v || (semiring != TR_PROV_BOOL && semiring != TR_PROV_COUNT) || tr_prov_plan_view_check(p) ||
        tr_prov_valuations_check(v, p->nvars) || (v->count && p->noutputs && !out) || (v->count && !status))
        return -1;                                                                       /* TR_MATH_INVALID */
    const uint64_t chunk = tr_prov_device_chunk(p, v->count, semiring), width = tr_prov_scratch_width(semiring);
    tr_math_need need;
    for (int i = 0; i < TR_BUDGET_RESOURCES; i++) need.need[i] = 0;
    need.need[TR_BUDGET_OUTPUT] = tr_budget_mul(v->count, p->noutputs);
    need.need[TR_BUDGET_BYTES] = tr_prov_device_bytes(p, v, semiring, chunk);
    tr_math_budget dflt;
    if (!budget) { tr_budget_default(&dflt); budget = &dflt; }
    int rc = tr_prov_gpu_check(budget, &need, refusal);
    if (rc) return rc;
    if (info) {
        info->count = v->count; info->noutputs = p->noutputs; info->overflowed = 0; info->chunk = chunk;
        info->semiring = semiring; info->reserved = 0;
    }
    if (!v->count) return 0;
    const uint64_t overrides = v->offset[v->count], outbytes = v->count * p->noutputs * (semiring == TR_PROV_COUNT ? 16 : 1);
    tr_prov_node *dnode = NULL;
    uint32_t *dchild = NULL, *doutput = NULL, *dvar = NULL;
    uint64_t *doffset = NULL, *dvalue = NULL, *dfallback = NULL;
    void *dout = NULL, *dscratch = NULL;
    int32_t *dstatus = NULL;
    rc = tr_cuda_alloc((void **)&dnode, (size_t)p->nodes * sizeof(tr_prov_node));
    if (!rc) rc = tr_cuda_alloc((void **)&dchild, (size_t)p->nchild * 4);
    if (!rc) rc = tr_cuda_alloc((void **)&doutput, (size_t)p->noutputs * 4);
    if (!rc) rc = tr_cuda_alloc((void **)&doffset, (size_t)(v->count + 1) * 8);
    if (!rc) rc = tr_cuda_alloc((void **)&dvar, (size_t)overrides * 4);
    if (!rc) rc = tr_cuda_alloc((void **)&dvalue, (size_t)overrides * 8);
    if (!rc && v->fallback) rc = tr_cuda_alloc((void **)&dfallback, (size_t)v->count * 8);
    if (!rc) rc = tr_cuda_alloc(&dout, (size_t)outbytes);
    if (!rc) rc = tr_cuda_alloc((void **)&dstatus, (size_t)v->count * 4);
    if (!rc) rc = tr_cuda_alloc(&dscratch, (size_t)(chunk * p->nodes * width));
    if (!rc) rc = tr_cuda_copy_in(dnode, p->node, (size_t)p->nodes * sizeof(tr_prov_node));
    if (!rc) rc = tr_cuda_copy_in(dchild, p->child, (size_t)p->nchild * 4);
    if (!rc) rc = tr_cuda_copy_in(doutput, p->output, (size_t)p->noutputs * 4);
    if (!rc) rc = tr_cuda_copy_in(doffset, v->offset, (size_t)(v->count + 1) * 8);
    if (!rc) rc = tr_cuda_copy_in(dvar, v->var, (size_t)overrides * 4);
    if (!rc) rc = tr_cuda_copy_in(dvalue, v->value, (size_t)overrides * 8);
    if (!rc && v->fallback) rc = tr_cuda_copy_in(dfallback, v->fallback, (size_t)v->count * 8);
    tr_prov_plan_view dp = *p;
    dp.node = dnode; dp.child = dchild; dp.output = doutput;
    tr_prov_valuations dv = *v;
    dv.offset = doffset; dv.var = dvar; dv.value = dvalue; dv.fallback = v->fallback ? dfallback : NULL;
    for (uint64_t first = 0; first < v->count && !rc; first += chunk) {
        uint64_t n = v->count - first < chunk ? v->count - first : chunk;
        if (semiring == TR_PROV_COUNT)
            TR_LAUNCH(tr_prov_count_kernel, n, dp, dv, first, n, (unsigned __int128 *)dscratch, (uint64_t *)dout, dstatus);
        else
            TR_LAUNCH(tr_prov_bool_kernel, n, dp, dv, first, n, (uint8_t *)dscratch, (uint8_t *)dout, dstatus);
        rc = tr_cuda_sync();
    }
    if (!rc && outbytes) rc = tr_cuda_copy_out(out, dout, (size_t)outbytes);
    if (!rc) rc = tr_cuda_copy_out(status, dstatus, (size_t)v->count * 4);
    tr_cuda_free(dnode); tr_cuda_free(dchild); tr_cuda_free(doutput); tr_cuda_free(doffset); tr_cuda_free(dvar);
    tr_cuda_free(dvalue); tr_cuda_free(dfallback); tr_cuda_free(dout); tr_cuda_free(dstatus); tr_cuda_free(dscratch);
    if (rc) return rc;
    if (info) for (uint64_t w = 0; w < v->count; w++) info->overflowed += status[w] != 0;
    return 0;
}
#ifdef __cplusplus
}
#endif
