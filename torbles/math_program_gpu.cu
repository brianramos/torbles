/* META-002 on a CUDA device: the per-output finalization of a query program, with exactly the values of the CPU
 * (math_program.h, tr_prog_eval_slot), over states that are already on the device (the shared FND-006 fused pass kept
 * there by tr_mgpu_reduce_fused_device: tr_mgpu_program_run) or copied from host arrays (tr_mgpu_program_finalize).
 * SPDX-License-Identifier: Apache-2.0
 *
 * One kernel: a work item per value slot (measure, group) evaluates the shared inline tr_prog_eval_slot and writes its
 * own output slot (no atomics, no order dependence): COUNT(*) and COUNT read a count, SUM and AVG contract the
 * measure's coefficients with the shared exact 128-bit sums into an exact 256-bit integer. The host checks the
 * budget of every device copy (and, for tr_mgpu_program_run, of the fused pass as well) before allocating. */
#include "math_cuda.h"
#include "math_program.h"
#include "math_handles.h"

TR_MGPU_ABI(tr_prog_final_view, 1);      /* structs passed to the CUDA library (math_handles.h) */
TR_MGPU_ABI(tr_prog_fmeasure, 1);
TR_MGPU_ABI(tr_prog_fterm, 1);
TR_MGPU_ABI(tr_prog_fout, 1);
TR_MGPU_ABI(tr_prog_value, 1);

TR_KERNEL void tr_prog_final_kernel(const tr_prog_final_view *v, tr_prog_value *out) {
    TR_GRID_LOOP(i, v->nslots) tr_prog_eval_slot(v, i, &out[i]);
}

static int tr_prog_view_ok(const tr_prog_final_view *fv) {
    return fv && fv->nmeasures >= 0 && fv->nterms >= 0 && fv->noutputs >= 0 && (!fv->nmeasures || fv->measures) &&
           (!fv->nterms || fv->terms) && (!fv->noutputs || fv->outs) && (!fv->nslots || fv->nmeasures > 0);
}
/* copies the tables of fv with outs replaced by douts (device pointers) and launches; values into out */
static int tr_prog_launch(const tr_prog_final_view *fv, const tr_prog_fout *douts_host, tr_prog_value *out,
                          void **dmem, int *nmem) {
    int rc = 0;
    tr_prog_final_view d = *fv;
    void *p = NULL;
#define TR_PROG_DEV(src, bytes) do { if (!rc) { p = NULL; rc = tr_cuda_alloc(&p, (size_t)(bytes)); \
        if (!rc) { dmem[(*nmem)++] = p; if (src) rc = tr_cuda_copy_in(p, (src), (size_t)(bytes)); } } } while (0)
    TR_PROG_DEV(fv->measures, (uint64_t)fv->nmeasures * sizeof(tr_prog_fmeasure)); d.measures = (const tr_prog_fmeasure *)p;
    TR_PROG_DEV(fv->terms, (uint64_t)fv->nterms * sizeof(tr_prog_fterm)); d.terms = (const tr_prog_fterm *)p;
    TR_PROG_DEV(douts_host, (uint64_t)fv->noutputs * sizeof(tr_prog_fout)); d.outs = (const tr_prog_fout *)p;
    TR_PROG_DEV((const void *)NULL, fv->nslots * sizeof(tr_prog_value));
    tr_prog_value *dvals = (tr_prog_value *)p;
    TR_PROG_DEV(&d, sizeof d);
    const tr_prog_final_view *dv = (const tr_prog_final_view *)p;
#undef TR_PROG_DEV
    if (!rc && fv->nslots) {
        TR_LAUNCH(tr_prog_final_kernel, fv->nslots, dv, dvals);
        rc = tr_cuda_sync();
    }
    if (!rc) rc = tr_cuda_copy_out(out, dvals, (size_t)fv->nslots * sizeof(tr_prog_value));
    return rc;
}

#ifdef __cplusplus
extern "C" {
#endif
int tr_mgpu_program_finalize(const tr_prog_final_view *fv, const tr_math_budget *budget, tr_prog_value *out,
                             tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!tr_prog_view_ok(fv) || (fv->nslots && !out)) return -1;                          /* TR_MATH_INVALID */
    tr_math_budget dflt;
    if (!budget) { tr_budget_default(&dflt); budget = &dflt; }
    tr_math_need need;
    for (int i = 0; i < TR_BUDGET_RESOURCES; i++) need.need[i] = 0;
    need.need[TR_BUDGET_OUTPUT] = fv->nslots;
    need.need[TR_BUDGET_BYTES] = tr_prog_final_bytes(fv, 1);
    int rc = tr_budget_check(budget, &need, refusal);
    if (rc) return rc;
    tr_prog_fout *douts = (tr_prog_fout *)calloc((size_t)(fv->noutputs > 0 ? fv->noutputs : 1), sizeof(tr_prog_fout));
    void **dmem = (void **)calloc((size_t)(3 * fv->noutputs + 8), sizeof(void *));
    if (!douts || !dmem) { free(douts); free(dmem); return -2; }                           /* TR_MATH_MEMORY */
    int nmem = 0;
    for (int32_t o = 0; o < fv->noutputs && !rc; o++) {          /* each bank output's states, copied in */
        const tr_prog_fout *x = &fv->outs[o];
        douts[o] = *x;
        if (!x->rows) continue;
        uint64_t gm = x->ngroups * (uint64_t)x->nterms;
        void *p = NULL;
        rc = tr_cuda_alloc(&p, (size_t)x->ngroups * 8);
        if (!rc) { dmem[nmem++] = p; rc = tr_cuda_copy_in(p, x->rows, (size_t)x->ngroups * 8); douts[o].rows = (const uint64_t *)p; }
        if (!rc) { rc = tr_cuda_alloc(&p, (size_t)gm * 8); }
        if (!rc) { dmem[nmem++] = p; rc = tr_cuda_copy_in(p, x->count, (size_t)gm * 8); douts[o].count = (const uint64_t *)p; }
        if (!rc) { rc = tr_cuda_alloc(&p, (size_t)gm * 16); }
        if (!rc) { dmem[nmem++] = p; rc = tr_cuda_copy_in(p, x->sum, (size_t)gm * 16); douts[o].sum = (const tr_math_i128 *)p; }
    }
    if (!rc) rc = tr_prog_launch(fv, douts, out, dmem, &nmem);
    for (int i = 0; i < nmem; i++) tr_cuda_free(dmem[i]);
    free(douts); free(dmem);
    return rc;
}

int tr_mgpu_program_run(const tr_reduce_view *view, const tr_math_need *base, const tr_prog_final_view *fv,
                        const tr_math_budget *budget, tr_reduce_arena *arena, tr_prog_value *out,
                        tr_reduce_stats *stats, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!view || !base || !tr_prog_view_ok(fv) || fv->noutputs != view->noutputs || !arena || !arena->a64 ||
        !arena->a128 || arena->n64 != view->arena64 || arena->n128 != view->arena128 || (fv->nslots && !out)) return -1;
    tr_math_budget dflt;
    if (!budget) { tr_budget_default(&dflt); budget = &dflt; }
    tr_math_need need;                                /* the fused pass and the finalization, before anything runs */
    tr_reduce_device_need(view, base, &need);
    if (fv->nslots > need.need[TR_BUDGET_OUTPUT]) need.need[TR_BUDGET_OUTPUT] = fv->nslots;
    need.need[TR_BUDGET_BYTES] = tr_budget_add(need.need[TR_BUDGET_BYTES], tr_prog_final_bytes(fv, 0));
    int rc = tr_budget_check(budget, &need, refusal);
    if (rc) return rc;
    int err = 0;
    tr_reduce_stats st;
    memset(&st, 0, sizeof st);
    tr_mgpu_reduce_states *s = tr_mgpu_reduce_fused_device(view, base, budget, &st, refusal, &err);
    if (!s) return err ? err : -2;
    tr_reduce_arena dev;
    rc = tr_mgpu_reduce_states_arena(s, &dev);
    tr_prog_fout *douts = (tr_prog_fout *)calloc((size_t)(fv->noutputs > 0 ? fv->noutputs : 1), sizeof(tr_prog_fout));
    void *dmem[8];
    int nmem = 0;
    if (!douts) rc = -2;
    for (int32_t o = 0; !rc && o < view->noutputs; o++) {        /* the states where the fused pass left them */
        const tr_reduce_out *x = &view->outs[o];
        if (x->kind != TR_REDUCE_BANK) continue;
        douts[o].rows = dev.a64 + x->rows_at;
        douts[o].count = dev.a64 + x->count_at;
        douts[o].sum = dev.a128 + x->sum_at;
        douts[o].ngroups = x->ngroups; douts[o].nterms = x->nterms;
    }
    if (!rc) rc = tr_prog_launch(fv, douts, out, dmem, &nmem);
    if (!rc) rc = tr_mgpu_reduce_states_copy(s, arena);           /* the states, for adoption and histograms */
    for (int i = 0; i < nmem; i++) tr_cuda_free(dmem[i]);
    tr_mgpu_reduce_states_free(s);
    free(douts);
    st.kernels += 1;
    st.transfer_bytes += arena->n64 * 8 + arena->n128 * 16 + fv->nslots * sizeof(tr_prog_value);
    st.workspace_bytes = need.need[TR_BUDGET_BYTES];
    if (stats) *stats = st;
    return rc;
}
#ifdef __cplusplus
}
#endif
