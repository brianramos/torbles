/* PROV-002 on a CUDA device: batched deletion propagation over a retained provenance DAG, with exactly the outputs of
 * tr_expl_propagate (math_explain.h).
 * SPDX-License-Identifier: Apache-2.0
 *
 * One work item per change set (a deletion of one or many source tuples, or any re-valuation): it runs the shared
 * tr_expl_set with its own scratch row (the nodes above its changed variables, in increasing order through a binary
 * heap, with the PLUS / TIMES / counting update laws and the early cutoff), writes its own rows of answers, status,
 * affected and changed counts, and restores the scratch entries it touched, so a row is clean for the next chunk and
 * nothing depends on the order of the threads. Sets run in chunks whose scratch stays within TR_EXPL_DEVICE_SCRATCH
 * bytes (one launch per chunk). The host checks the view, the change sets and the budget (output = sets x answers,
 * bytes = every device copy, the outputs and the scratch chunk) before allocating anything on the device. */
#include "math_cuda.h"
#include "math_explain.h"
#include "math_handles.h"

TR_MGPU_ABI(tr_expl_view, TR_EXPL_ABI);           /* structs passed to the CUDA library (math_handles.h) */
TR_MGPU_ABI(tr_expl_set_result, TR_EXPL_ABI);

TR_KERNEL void tr_expl_propagate_kernel(tr_expl_view s, tr_prov_valuations v, uint64_t first, uint64_t count,
                                        tr_expl_scratch all, uint64_t *out_n, uint8_t *out_b, tr_expl_set_result *res) {
    TR_GRID_LOOP(i, count) {
        const uint64_t w = first + i;
        const tr_expl_scratch x = tr_expl_scratch_row(&all, s.nodes, i);
        uint64_t affected = 0, changed = 0;
        int32_t status = tr_expl_set(&s, &v, w, &x, out_n + 2 * w * s.noutputs, out_b + w * s.noutputs, &affected,
                                     &changed);
        res[w].status = status; res[w].reserved = 0; res[w].affected = affected; res[w].changed = changed;
    }
}

/* tr_budget_check, refusing a size past 64 bits even under an unlimited budget */
static int tr_expl_gpu_check(const tr_math_budget *b, const tr_math_need *n, tr_math_refusal *r) {
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
int tr_mgpu_explain_propagate(const tr_expl_view *s, const tr_prov_valuations *v, uint64_t *out_n, uint8_t *out_b,
                              tr_expl_set_result *res, const tr_math_budget *budget, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!s || tr_expl_view_check(s) || tr_expl_sets_check(v, s->nvars) || (v->count && s->noutputs && (!out_n || !out_b)) ||
        (v->count && !res)) return -1;                                                    /* TR_MATH_INVALID */
    const uint64_t chunk = tr_expl_device_chunk(s, v->count), n = s->nodes ? s->nodes : 1;
    tr_math_need need;
    for (int i = 0; i < TR_BUDGET_RESOURCES; i++) need.need[i] = 0;
    need.need[TR_BUDGET_OUTPUT] = tr_budget_mul(v->count, s->noutputs);
    need.need[TR_BUDGET_BYTES] = tr_expl_device_bytes(s, v, chunk);
    tr_math_budget dflt;
    if (!budget) { tr_budget_default(&dflt); budget = &dflt; }
    int rc = tr_expl_gpu_check(budget, &need, refusal);
    if (rc) return rc;
    if (!v->count) return 0;
    const uint64_t overrides = v->offset[v->count], no = s->noutputs;
    tr_prov_node *dnode = NULL;
    uint32_t *dchild = NULL, *doutput = NULL, *drev = NULL, *douts = NULL, *dvarn = NULL, *dbase_c = NULL, *dvar = NULL;
    uint64_t *drev_at = NULL, *dout_at = NULL, *dbase_n = NULL, *doffset = NULL, *dvalue = NULL, *dn = NULL;
    uint8_t *dbase_b = NULL, *db = NULL;
    tr_expl_set_result *dres = NULL;
    tr_expl_scratch all;
    memset(&all, 0, sizeof all);
    rc = tr_cuda_alloc((void **)&dnode, (size_t)n * sizeof(tr_prov_node));
    if (!rc) rc = tr_cuda_alloc((void **)&dchild, (size_t)(s->nchild ? s->nchild : 1) * 4);
    if (!rc) rc = tr_cuda_alloc((void **)&drev, (size_t)(s->nchild ? s->nchild : 1) * 4);
    if (!rc) rc = tr_cuda_alloc((void **)&doutput, (size_t)(no ? no : 1) * 4);
    if (!rc) rc = tr_cuda_alloc((void **)&douts, (size_t)(no ? no : 1) * 4);
    if (!rc) rc = tr_cuda_alloc((void **)&dvarn, (size_t)(s->nvars ? s->nvars : 1) * 4);
    if (!rc) rc = tr_cuda_alloc((void **)&dbase_c, (size_t)n * 4);
    if (!rc) rc = tr_cuda_alloc((void **)&drev_at, (size_t)(s->nodes + 1) * 8);
    if (!rc) rc = tr_cuda_alloc((void **)&dout_at, (size_t)(s->nodes + 1) * 8);
    if (!rc) rc = tr_cuda_alloc((void **)&dbase_n, (size_t)n * 16);
    if (!rc) rc = tr_cuda_alloc((void **)&dbase_b, (size_t)n);
    if (!rc) rc = tr_cuda_alloc((void **)&doffset, (size_t)(v->count + 1) * 8);
    if (!rc) rc = tr_cuda_alloc((void **)&dvar, (size_t)(overrides ? overrides : 1) * 4);
    if (!rc) rc = tr_cuda_alloc((void **)&dvalue, (size_t)(overrides ? overrides : 1) * 8);
    if (!rc) rc = tr_cuda_alloc((void **)&dn, (size_t)(v->count * (no ? no : 1)) * 16);
    if (!rc) rc = tr_cuda_alloc((void **)&db, (size_t)(v->count * (no ? no : 1)));
    if (!rc) rc = tr_cuda_alloc((void **)&dres, (size_t)v->count * sizeof(tr_expl_set_result));
    if (!rc) rc = tr_cuda_alloc((void **)&all.newn, (size_t)(chunk * n) * 16);          /* zeroed: clean rows */
    if (!rc) rc = tr_cuda_alloc((void **)&all.sub, (size_t)(chunk * n) * 16);
    if (!rc) rc = tr_cuda_alloc((void **)&all.add, (size_t)(chunk * n) * 16);
    if (!rc) rc = tr_cuda_alloc((void **)&all.dcnt, (size_t)(chunk * n) * 8);
    if (!rc) rc = tr_cuda_alloc((void **)&all.heap, (size_t)(chunk * n) * 4);
    if (!rc) rc = tr_cuda_alloc((void **)&all.touched, (size_t)(chunk * n) * 4);
    if (!rc) rc = tr_cuda_alloc((void **)&all.flag, (size_t)(chunk * n));
    if (!rc && s->nodes) rc = tr_cuda_copy_in(dnode, s->node, (size_t)s->nodes * sizeof(tr_prov_node));
    if (!rc && s->nchild) rc = tr_cuda_copy_in(dchild, s->child, (size_t)s->nchild * 4);
    if (!rc && s->nchild) rc = tr_cuda_copy_in(drev, s->rev, (size_t)s->nchild * 4);
    if (!rc && no) rc = tr_cuda_copy_in(doutput, s->output, (size_t)no * 4);
    if (!rc && no) rc = tr_cuda_copy_in(douts, s->outs, (size_t)no * 4);
    if (!rc && s->nvars) rc = tr_cuda_copy_in(dvarn, s->var_node, (size_t)s->nvars * 4);
    if (!rc && s->nodes) rc = tr_cuda_copy_in(dbase_c, s->base_c, (size_t)s->nodes * 4);
    if (!rc) rc = tr_cuda_copy_in(drev_at, s->rev_at, (size_t)(s->nodes + 1) * 8);
    if (!rc) rc = tr_cuda_copy_in(dout_at, s->out_at, (size_t)(s->nodes + 1) * 8);
    if (!rc && s->nodes) rc = tr_cuda_copy_in(dbase_n, s->base_n, (size_t)s->nodes * 16);
    if (!rc && s->nodes) rc = tr_cuda_copy_in(dbase_b, s->base_b, (size_t)s->nodes);
    if (!rc) rc = tr_cuda_copy_in(doffset, v->offset, (size_t)(v->count + 1) * 8);
    if (!rc && overrides) rc = tr_cuda_copy_in(dvar, v->var, (size_t)overrides * 4);
    if (!rc && overrides) rc = tr_cuda_copy_in(dvalue, v->value, (size_t)overrides * 8);
    tr_expl_view ds = *s;
    ds.node = dnode; ds.child = dchild; ds.output = doutput; ds.rev_at = drev_at; ds.rev = drev; ds.out_at = dout_at;
    ds.outs = douts; ds.var_node = dvarn; ds.base_n = dbase_n; ds.base_b = dbase_b; ds.base_c = dbase_c;
    tr_prov_valuations dv = *v;
    dv.offset = doffset; dv.var = dvar; dv.value = dvalue; dv.fallback = NULL;
    for (uint64_t first = 0; first < v->count && !rc; first += chunk) {
        const uint64_t m = v->count - first < chunk ? v->count - first : chunk;
        TR_LAUNCH(tr_expl_propagate_kernel, m, ds, dv, first, m, all, dn, db, dres);
        rc = tr_cuda_sync();
    }
    if (!rc && no) rc = tr_cuda_copy_out(out_n, dn, (size_t)(v->count * no) * 16);
    if (!rc && no) rc = tr_cuda_copy_out(out_b, db, (size_t)(v->count * no));
    if (!rc) rc = tr_cuda_copy_out(res, dres, (size_t)v->count * sizeof(tr_expl_set_result));
    tr_cuda_free(dnode); tr_cuda_free(dchild); tr_cuda_free(drev); tr_cuda_free(doutput); tr_cuda_free(douts);
    tr_cuda_free(dvarn); tr_cuda_free(dbase_c); tr_cuda_free(drev_at); tr_cuda_free(dout_at); tr_cuda_free(dbase_n);
    tr_cuda_free(dbase_b); tr_cuda_free(doffset); tr_cuda_free(dvar); tr_cuda_free(dvalue); tr_cuda_free(dn);
    tr_cuda_free(db); tr_cuda_free(dres);
    tr_cuda_free(all.newn); tr_cuda_free(all.sub); tr_cuda_free(all.add); tr_cuda_free(all.dcnt); tr_cuda_free(all.heap);
    tr_cuda_free(all.touched); tr_cuda_free(all.flag);
    return rc;
}
#ifdef __cplusplus
}
#endif
