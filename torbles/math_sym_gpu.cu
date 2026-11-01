/* SYM-001 / SYM-002 on a CUDA device: batched basis-action requests over retained moment states, and the batched
 * verification of declared symmetries over a support, with exactly the results of tr_sym_query_view_eval and
 * tr_sym_verify_view (math_sym.h).
 * SPDX-License-Identifier: Apache-2.0
 *
 * Requests: one work item per request runs the shared evaluator tr_sym_eval_request (exact 512-bit products and sums
 * of the action's int128 numerators and the int128 moments, or outward-rounded interval products and sums in the same
 * order) and writes its own output slot. Verification: a work item is (generator, tuple slice): it maps each tuple of
 * its slice by the generator (tr_sym_image, exact int128 with every coordinate checked into int64), finds the image
 * among the sorted support tuples by binary search (exact comparisons, no hashing), compares the multiplicities
 * (tr_sym_check_tuple), counts locally and adds one atomic per counter; the smallest violating row is an atomic
 * minimum. Neither result depends on the thread order. The host finishes the evidence with tr_sym_evidence_finish. */
#include "math_cuda.h"
#include "math_sym.h"
#include "math_handles.h"

TR_MGPU_ABI(tr_sym_query_view, TR_SYM_ABI);     /* structs passed to the CUDA library (math_handles.h) */
TR_MGPU_ABI(tr_sym_action_view, TR_SYM_ABI);
TR_MGPU_ABI(tr_sym_request, TR_SYM_ABI);
TR_MGPU_ABI(tr_sym_result, TR_SYM_ABI);
TR_MGPU_ABI(tr_sym_wide, TR_SYM_ABI);
TR_MGPU_ABI(tr_sym_support_view, TR_SYM_ABI);
TR_MGPU_ABI(tr_sym_group_view, TR_SYM_ABI);
TR_MGPU_ABI(tr_sym_gen, TR_SYM_ABI);
TR_MGPU_ABI(tr_sym_evidence, TR_SYM_ABI);

#define TR_SYM_SLICE_TUPLES 32           /* at least this many tuples per verification work item */
#define TR_SYM_SLICES (UINT64_C(1) << 16) /* at most this many work items per generator */
#define TR_SYM_ACC 6                      /* rows, tuples, violations, missing, broken, witness */

TR_KERNEL void tr_sym_query_kernel(tr_sym_query_view v, const tr_sym_request *req, uint64_t n, tr_sym_result *out) {
    TR_GRID_LOOP(i, n) tr_sym_eval_request(&v, &req[i], &out[i]);
}

TR_KERNEL void tr_sym_verify_kernel(tr_sym_support_view s, tr_sym_group_view g, uint64_t slices, uint64_t *acc) {
    TR_GRID_LOOP(w, (uint64_t)g.ngen * slices) {
        const int32_t j = (int32_t)(w / slices);
        const uint64_t first = w % slices;
        uint64_t rows = 0, tuples = 0, viol = 0, miss = 0, brok = 0, wit = TR_SYM_NONE;
        int64_t y[TR_SYM_MAX_COORDS];
        for (uint64_t t = first; t < s.distinct; t += slices) {
            uint64_t im = 0;
            int k = tr_sym_check_tuple(&s, &g, j, t, y, &im);
            rows += s.mult[t]; tuples++;
            if (k == TR_SYM_HOLDS) continue;
            viol += s.mult[t];
            if (k == TR_SYM_MISSING) miss++; else brok++;
            if (s.first[t] < wit) wit = s.first[t];
        }
        uint64_t *a = acc + (uint64_t)j * TR_SYM_ACC;
        if (rows) tr_atomic_add_u64(&a[0], rows);
        if (tuples) tr_atomic_add_u64(&a[1], tuples);
        if (viol) tr_atomic_add_u64(&a[2], viol);
        if (miss) tr_atomic_add_u64(&a[3], miss);
        if (brok) tr_atomic_add_u64(&a[4], brok);
        if (wit != TR_SYM_NONE) tr_atomic_min_u64(&a[5], wit);
    }
}

#ifdef __cplusplus
extern "C" {
#endif
int tr_mgpu_sym_query(const tr_sym_query_view *view, const tr_sym_request *requests, uint64_t n, tr_sym_result *out,
                      const tr_math_budget *budget, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (tr_sym_query_view_check(view) || (n && (!requests || !out))) return -1;
    tr_math_need need;
    tr_sym_query_needs(view, n, TR_SYM_NEED_DEVICE, &need);
    tr_math_budget dflt;
    if (!budget) { tr_budget_default(&dflt); budget = &dflt; }
    int rc = tr_budget_check(budget, &need, refusal);
    if (rc || !n) return rc;
    const uint64_t nb = (uint64_t)view->nb, na = (uint64_t)view->nactions, cells = nb * nb;
    void **mem = (void **)calloc(2 * na + 6, sizeof(void *));
    tr_sym_action_view *hv = (tr_sym_action_view *)calloc(na ? na : 1, sizeof(tr_sym_action_view));
    if (!mem || !hv) { free(mem); free(hv); return -2; }
    int nmem = 0;
#define TR_SYM_DEV(ptr, src, bytes) do { if (!rc) { void *p_ = NULL; rc = tr_cuda_alloc(&p_, (size_t)(bytes)); \
        if (!rc) { mem[nmem++] = p_; if (src) rc = tr_cuda_copy_in(p_, (src), (size_t)(bytes)); } (ptr) = p_; } } while (0)
    tr_sym_query_view d = *view;
    void *p = NULL;
    TR_SYM_DEV(p, view->moments, view->ngroups * nb * 16);
    d.moments = (const tr_math_i128 *)p;
    for (uint64_t i = 0; i < na && !rc; i++) {
        hv[i] = view->actions[i];
        if (view->actions[i].carrier == TR_SYM_EXACT) { TR_SYM_DEV(p, view->actions[i].num, cells * 16); hv[i].num = (const tr_math_i128 *)p; }
        else { TR_SYM_DEV(p, view->actions[i].iv, cells * 16); hv[i].iv = (const tr_math_interval *)p; }
        if (view->actions[i].permutation) { TR_SYM_DEV(p, view->actions[i].index, nb * 4); hv[i].index = (const int32_t *)p; }
    }
    if (na) { TR_SYM_DEV(p, hv, na * sizeof(tr_sym_action_view)); d.actions = (const tr_sym_action_view *)p; }
    if (view->nfunc) { TR_SYM_DEV(p, view->functionals, (uint64_t)view->nfunc * nb * 8); d.functionals = (const int64_t *)p; }
    tr_sym_request *dreq = NULL;
    tr_sym_result *dout = NULL;
    TR_SYM_DEV(p, requests, n * sizeof(tr_sym_request));
    dreq = (tr_sym_request *)p;
    TR_SYM_DEV(p, (const void *)NULL, n * sizeof(tr_sym_result));
    dout = (tr_sym_result *)p;
#undef TR_SYM_DEV
    if (!rc) {
        TR_LAUNCH(tr_sym_query_kernel, n, d, dreq, n, dout);
        rc = tr_cuda_sync();
    }
    if (!rc) rc = tr_cuda_copy_out(out, dout, (size_t)(n * sizeof(tr_sym_result)));
    for (int i = 0; i < nmem; i++) tr_cuda_free(mem[i]);
    free(mem); free(hv);
    return rc;
}

int tr_mgpu_sym_verify(const tr_sym_support_view *support, const tr_sym_group_view *group, uint64_t ncoef,
                       uint64_t nvalues, const tr_math_budget *budget, tr_sym_evidence *out, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (tr_sym_views_check(support, group, ncoef, nvalues) || !out) return -1;
    tr_math_need need;
    tr_sym_verify_needs(support, group, ncoef, nvalues, TR_SYM_NEED_DEVICE, &need);
    tr_math_budget dflt;
    if (!budget) { tr_budget_default(&dflt); budget = &dflt; }
    int rc = tr_budget_check(budget, &need, refusal);
    if (rc) return rc;
    const uint64_t D = support->distinct, ng = (uint64_t)group->ngen, n = (uint64_t)support->n;
    uint64_t acc_host[TR_SYM_MAX_GENERATORS * TR_SYM_ACC];
    for (uint64_t j = 0; j < ng; j++) {
        for (int k = 0; k < TR_SYM_ACC; k++) acc_host[j * TR_SYM_ACC + k] = 0;
        acc_host[j * TR_SYM_ACC + 5] = TR_SYM_NONE;
    }
    void *mem[8];
    int nmem = 0;
#define TR_SYM_DEV(ptr, src, bytes) do { if (!rc) { void *p_ = NULL; rc = tr_cuda_alloc(&p_, (size_t)(bytes)); \
        if (!rc) { mem[nmem++] = p_; if (src) rc = tr_cuda_copy_in(p_, (src), (size_t)(bytes)); } (ptr) = p_; } } while (0)
    tr_sym_support_view s = *support;
    tr_sym_group_view g = *group;
    void *p = NULL;
    TR_SYM_DEV(p, support->tuples, D * n * 8); s.tuples = (const int64_t *)p;
    TR_SYM_DEV(p, support->mult, D * 8); s.mult = (const uint64_t *)p;
    TR_SYM_DEV(p, support->first, D * 8); s.first = (const uint64_t *)p;
    TR_SYM_DEV(p, group->gens, ng * sizeof(tr_sym_gen)); g.gens = (const tr_sym_gen *)p;
    TR_SYM_DEV(p, ncoef ? (const void *)group->coef : (const void *)NULL, ncoef * 8); g.coef = (const int64_t *)p;
    TR_SYM_DEV(p, nvalues ? (const void *)group->keys : (const void *)NULL, nvalues * 8); g.keys = (const int64_t *)p;
    TR_SYM_DEV(p, nvalues ? (const void *)group->targets : (const void *)NULL, nvalues * 8); g.targets = (const int64_t *)p;
    uint64_t *dacc = NULL;
    TR_SYM_DEV(p, acc_host, ng * TR_SYM_ACC * 8); dacc = (uint64_t *)p;
#undef TR_SYM_DEV
    if (!rc) {
        uint64_t slices = (D + TR_SYM_SLICE_TUPLES - 1) / TR_SYM_SLICE_TUPLES;
        if (slices > TR_SYM_SLICES) slices = TR_SYM_SLICES;
        if (!slices) slices = 1;
        TR_LAUNCH(tr_sym_verify_kernel, ng * slices, s, g, slices, dacc);
        rc = tr_cuda_sync();
    }
    if (!rc) rc = tr_cuda_copy_out(acc_host, dacc, (size_t)(ng * TR_SYM_ACC * 8));
    for (int i = 0; i < nmem; i++) tr_cuda_free(mem[i]);
    if (rc) return rc;
    for (uint64_t j = 0; j < ng; j++) {
        tr_sym_evidence *e = &out[j];
        memset(e, 0, sizeof *e);
        const uint64_t *a = acc_host + j * TR_SYM_ACC;
        e->rows = a[0]; e->tuples = a[1]; e->violations = a[2]; e->missing = a[3]; e->broken = a[4]; e->witness = a[5];
        rc = tr_sym_evidence_finish(support, group, (int32_t)j, e);   /* host arrays: the caller's views */
        if (rc) return rc;
    }
    return 0;
}
#ifdef __cplusplus
}
#endif
