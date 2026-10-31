/* BOOL-001 on a CUDA device: the condition words of a scope (tr_bool_masks_build) and the shared-mask evaluation of
 * cofactors and differences (tr_bool_differences_view), with exactly the CPU's counts (math_bool.h).
 * SPDX-License-Identifier: Apache-2.0
 *
 * Masks: one work item per (condition, word) evaluates the selected rows of its 64-row word with the shared
 * tr_bool_condition_word and writes its own two output words.
 * Differences: work items (query q, word slice s): slice s of S takes the words s, s + S, s + 2S, ... A work item owns
 * 16 bytes of node words per program node (its slot of the scratch), evaluates each of its words like the CPU (the base
 * schedule, then the cone of the query once per assignment, all with bitwise operations, a valid word beside each
 * value word for UNKNOWN), counts with popcounts into local totals, and adds the histogram bins and cofactor counts of
 * each word with integer atomics (only nonzero addends) and its totals with one atomic per counter at the end (the
 * exact 128-bit sum with tr_atomic_add_wide). Integer atomics commute, so the counts do not depend on thread order. */
#include "math_cuda.h"
#include "math_bool.h"
#include "math_handles.h"

TR_MGPU_ABI(tr_bool_condition, TR_BOOL_ABI);      /* structs passed to the CUDA library (math_handles.h) */
TR_MGPU_ABI(tr_bool_node, TR_BOOL_ABI);
TR_MGPU_ABI(tr_bool_program_view, TR_BOOL_ABI);
TR_MGPU_ABI(tr_bool_domain, TR_BOOL_ABI);
TR_MGPU_ABI(tr_bool_query, TR_BOOL_ABI);
TR_MGPU_ABI(tr_bool_result, TR_BOOL_ABI);

TR_KERNEL void tr_bool_masks_kernel(tr_scope_view v, const tr_bool_condition *conds, uint64_t items, uint64_t nw,
                                    uint64_t *value, uint64_t *valid) {
    TR_GRID_LOOP(item, items) {
        tr_bool_condition c = conds[item / nw];
        uint64_t a = 0, b = 0;
        tr_bool_condition_word(&v, &c, item % nw, &a, &b);
        value[item] = a; valid[item] = b;
    }
}

/* per query: acc[8 q + 0..7] = f_true, f_false, f_unknown, known, unknown, xor_true, xor_false, nonzero;
 * sums[2 q], sums[2 q + 1] = the low and high words of the 128-bit sum */
TR_KERNEL void tr_bool_diff_kernel(tr_bool_domain d, tr_bool_program_view p, const tr_bool_plan *plans,
                                   const uint32_t *cones, uint64_t items, uint64_t slices, uint64_t *scratch,
                                   uint64_t *acc, int64_t *sums, uint64_t *hist, uint64_t *cof) {
    TR_GRID_LOOP(item, items) {
        const uint64_t j = item / slices, s = item % slices, root = (uint64_t)p.root;
        const tr_bool_plan P = plans[j];
        uint64_t *val = scratch + item * 2 * (uint64_t)p.nnodes;
        uint64_t ft = 0, ff = 0, fu = 0, kn = 0, un = 0, xt = 0, xf = 0, nz = 0;
        __int128 sum = 0;
        int32_t values[64];
        uint32_t counts[64];
        const uint64_t nt = (uint64_t)1 << P.order;
        for (uint64_t w = s; w < d.nw; w += slices) {
            uint64_t sel = tr_bool_selection_word(&d, w);
            if (!sel) continue;
            tr_bool_eval_base(&p, &d, w, val);
            uint64_t fv = val[2 * root] & sel, fk = val[2 * root + 1] & sel;
            ft += (uint64_t)tr_bool_popc(fv); ff += (uint64_t)tr_bool_popc(fk & ~fv); fu += (uint64_t)tr_bool_popc(sel & ~fk);
            tr_bool_acc A;
            tr_bool_acc_begin(&A, P.order, sel);
            for (uint64_t t = 0; t < nt; t++) {
                if (P.ncone) tr_bool_eval_cone(&p, &d, cones + P.cone, P.ncone, P.vars, t, 0, w, val);
                uint64_t rv = val[2 * root] & sel, rk = val[2 * root + 1] & sel;
                tr_bool_acc_add(&A, P.order, t, rv, rk);
                if (P.cof != TR_BOOL_NONE) {
                    uint64_t *c = cof + P.cof + 3 * t;
                    uint64_t c0 = (uint64_t)tr_bool_popc(rv), c1 = (uint64_t)tr_bool_popc(rk & ~rv),
                             c2 = (uint64_t)tr_bool_popc(sel & ~rk);
                    if (c0) tr_atomic_add_u64(&c[0], c0);
                    if (c1) tr_atomic_add_u64(&c[1], c1);
                    if (c2) tr_atomic_add_u64(&c[2], c2);
                }
            }
            uint64_t known = A.known;
            kn += (uint64_t)tr_bool_popc(known);
            un += (uint64_t)tr_bool_popc(sel & ~known);
            xt += (uint64_t)tr_bool_popc(A.parity & known);
            xf += (uint64_t)tr_bool_popc(~A.parity & known);
            int n = tr_bool_acc_leaves(&A, P.order, values, counts);
            for (int l = 0; l < n; l++) {
                int64_t b = (int64_t)values[l] - P.lo;
                if (b < 0 || b >= P.nbins) continue;             /* never: |DS f| <= 2^(m-1) */
                tr_atomic_add_u64(&hist[P.hist + (uint64_t)b], (uint64_t)counts[l]);
                if (values[l]) nz += counts[l];
                sum += (__int128)values[l] * (__int128)counts[l];
            }
        }
        uint64_t local[8] = {ft, ff, fu, kn, un, xt, xf, nz};
        for (int c = 0; c < 8; c++) if (local[c]) tr_atomic_add_u64(&acc[8 * j + (uint64_t)c], local[c]);
        if (sum) tr_atomic_add_wide(&sums[2 * j], &sums[2 * j + 1], sum);
    }
}

#ifdef __cplusplus
extern "C" {
#endif
int tr_mgpu_bool_masks(const tr_scope_view *view, const tr_bool_condition *conds, int32_t k,
                       const tr_math_budget *budget, uint64_t *value, uint64_t *valid, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!view || k < 0 || (k && !conds) || view->ncols < 0 || view->ncols > TR_SCOPE_MAX_COLUMNS ||
        (view->rows && !view->selection)) return -1;                                          /* TR_MATH_INVALID */
    for (int32_t i = 0; i < k; i++) if (tr_bool_condition_check(view, &conds[i])) return -1;
    const uint64_t nw = (view->rows + 63) / 64, items = (uint64_t)k * nw;
    if (items && (!value || !valid)) return -1;
    tr_math_budget dflt;
    if (!budget) { tr_budget_default(&dflt); budget = &dflt; }
    tr_math_need need;
    for (int i = 0; i < TR_BUDGET_RESOURCES; i++) need.need[i] = 0;
    need.need[TR_BUDGET_VARIABLES] = (uint64_t)k;
    need.need[TR_BUDGET_BYTES] = tr_bool_masks_device_bytes(view, conds, k);
    int rc = tr_budget_check(budget, &need, refusal);
    if (rc) return rc;
    if (k > TR_BOOL_MAX_CONDITIONS) return TR_MATH_UNSUPPORTED;
    if (!items) return 0;
    uint64_t used = 0;
    for (int32_t i = 0; i < k; i++) {
        used |= (uint64_t)1 << conds[i].col;
        if (conds[i].op >= TR_BOOL_COL_EQ) used |= (uint64_t)1 << conds[i].col2;
    }
    tr_scope_view d = *view;
    d.groups = NULL;                                          /* never read */
    void *mem[2 * TR_SCOPE_MAX_COLUMNS + 4];
    int nmem = 0;
    uint64_t *dsel = NULL, *dvalue = NULL, *dvalid = NULL;
    tr_bool_condition *dconds = NULL;
#define TR_BOOL_DEV(ptr, bytes) do { if (!rc) { rc = tr_cuda_alloc((void **)&(ptr), (size_t)(bytes)); if (!rc) mem[nmem++] = (void *)(ptr); } } while (0)
    TR_BOOL_DEV(dsel, nw * 8);
    if (!rc) rc = tr_cuda_copy_in(dsel, view->selection, (size_t)(nw * 8));
    d.selection = dsel;
    for (int32_t c = 0; c < view->ncols; c++) {
        tr_scope_view_column *col = &d.col[c];
        if (!((used >> c) & 1)) { col->codes = NULL; col->values = NULL; col->valid = NULL; continue; }
        if (view->col[c].values) {
            int64_t *x = NULL;
            TR_BOOL_DEV(x, view->rows * 8);
            if (!rc) rc = tr_cuda_copy_in(x, view->col[c].values, (size_t)(view->rows * 8));
            col->values = x;
        } else {
            uint32_t *x = NULL;
            TR_BOOL_DEV(x, view->rows * 4);
            if (!rc) rc = tr_cuda_copy_in(x, view->col[c].codes, (size_t)(view->rows * 4));
            col->codes = x;
        }
        if (view->col[c].valid) {
            uint8_t *x = NULL;
            TR_BOOL_DEV(x, view->rows);
            if (!rc) rc = tr_cuda_copy_in(x, view->col[c].valid, (size_t)view->rows);
            col->valid = x;
        }
    }
    TR_BOOL_DEV(dconds, (uint64_t)k * sizeof(tr_bool_condition));
    if (!rc) rc = tr_cuda_copy_in(dconds, conds, (size_t)k * sizeof(tr_bool_condition));
    TR_BOOL_DEV(dvalue, items * 8);
    TR_BOOL_DEV(dvalid, items * 8);
#undef TR_BOOL_DEV
    if (!rc) { TR_LAUNCH(tr_bool_masks_kernel, items, d, dconds, items, nw, dvalue, dvalid); rc = tr_cuda_sync(); }
    if (!rc) rc = tr_cuda_copy_out(value, dvalue, (size_t)(items * 8));
    if (!rc) rc = tr_cuda_copy_out(valid, dvalid, (size_t)(items * 8));
    for (int i = 0; i < nmem; i++) tr_cuda_free(mem[i]);
    return rc;
}

int tr_mgpu_bool_differences(const tr_bool_domain *dom, const tr_bool_program_view *prog, const tr_bool_query *q,
                             int32_t nq, const tr_math_budget *budget, tr_bool_result *results, uint64_t *hist,
                             uint64_t *cof, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    int rc = tr_bool_domain_check(dom);
    if (!rc) rc = tr_bool_program_check(prog);
    if (rc) return rc;
    if (dom->k != prog->k) return -6;                                                         /* TR_MATH_DOMAIN_ERROR */
    uint64_t ncones = 0, hist_total = 0, cof_total = 0, tensor = 0;
    int32_t top = 0;
    rc = tr_bool_plan_queries(prog, q, nq, NULL, NULL, &ncones, &hist_total, &cof_total, &tensor, &top);
    if (rc) return rc;
    if ((nq && !results) || (hist_total && !hist) || (cof_total && !cof)) return -1;
    tr_math_budget dflt;
    if (!budget) { tr_budget_default(&dflt); budget = &dflt; }
    tr_math_need need;
    tr_bool_differences_base_need(prog, dom, nq, ncones, hist_total, cof_total, tensor, top, &need);
    need.need[TR_BUDGET_BYTES] = tr_bool_device_bytes(prog, dom, nq, ncones, hist_total, cof_total);
    rc = tr_budget_check(budget, &need, refusal);
    if (rc) return rc;
    tr_bool_plan *plans = (tr_bool_plan *)calloc((size_t)(nq > 0 ? nq : 1), sizeof(tr_bool_plan));
    uint32_t *cones = (uint32_t *)calloc((size_t)(ncones ? ncones : 1), 4);
    if (!plans || !cones) { free(plans); free(cones); return -2; }                         /* TR_MATH_MEMORY */
    tr_bool_plan_queries(prog, q, nq, plans, cones, NULL, NULL, NULL, NULL, NULL);
    for (int32_t j = 0; j < nq; j++) {
        tr_bool_result *R = &results[j];
        memset(R, 0, sizeof *R);
        R->vars = plans[j].vars; R->order = plans[j].order; R->nbins = plans[j].nbins; R->lo = plans[j].lo;
        R->points = dom->selected; R->hist = plans[j].hist; R->cof = plans[j].cof;
    }
    if (hist_total) memset(hist, 0, (size_t)(hist_total * 8));
    if (cof_total) memset(cof, 0, (size_t)(cof_total * 8));
    if (!nq || !dom->selected) { free(plans); free(cones); return 0; }   /* nothing counts: every count is zero */
    const uint64_t nw = dom->nw, slices = tr_bool_slices(nw, nq, prog->nnodes), items = (uint64_t)nq * slices;
    tr_bool_domain d = *dom;
    tr_bool_program_view p = *prog;
    p.deps = NULL;                                            /* never read by the kernels (the cones are planned) */
    void *mem[16];
    int nmem = 0;
    uint64_t *dsel = NULL, *dvalue = NULL, *dvalid = NULL, *dscratch = NULL, *dacc = NULL, *dhist = NULL, *dcof = NULL;
    int64_t *dsums = NULL;
    tr_bool_node *dnodes = NULL;
    uint32_t *dsched = NULL, *dcones = NULL;
    tr_bool_plan *dplans = NULL;
#define TR_BOOL_DEV(ptr, bytes) do { if (!rc) { rc = tr_cuda_alloc((void **)&(ptr), (size_t)(bytes)); if (!rc) mem[nmem++] = (void *)(ptr); } } while (0)
    if (dom->kind == TR_BOOL_ROWS) {
        uint64_t words = (uint64_t)dom->k * nw;
        TR_BOOL_DEV(dsel, nw * 8);
        if (!rc) rc = tr_cuda_copy_in(dsel, dom->selection, (size_t)(nw * 8));
        TR_BOOL_DEV(dvalue, words * 8);
        if (!rc) rc = tr_cuda_copy_in(dvalue, dom->value, (size_t)(words * 8));
        if (dom->valid) {
            TR_BOOL_DEV(dvalid, words * 8);
            if (!rc) rc = tr_cuda_copy_in(dvalid, dom->valid, (size_t)(words * 8));
        }
        d.selection = dsel; d.value = dvalue; d.valid = dvalid;
    }
    TR_BOOL_DEV(dnodes, (uint64_t)prog->nnodes * sizeof(tr_bool_node));
    if (!rc) rc = tr_cuda_copy_in(dnodes, prog->nodes, (size_t)prog->nnodes * sizeof(tr_bool_node));
    TR_BOOL_DEV(dsched, (uint64_t)prog->nsched * 4);
    if (!rc) rc = tr_cuda_copy_in(dsched, prog->schedule, (size_t)prog->nsched * 4);
    p.nodes = dnodes; p.schedule = dsched;
    TR_BOOL_DEV(dplans, (uint64_t)nq * sizeof(tr_bool_plan));
    if (!rc) rc = tr_cuda_copy_in(dplans, plans, (size_t)nq * sizeof(tr_bool_plan));
    TR_BOOL_DEV(dcones, (ncones ? ncones : 1) * 4);
    if (!rc && ncones) rc = tr_cuda_copy_in(dcones, cones, (size_t)(ncones * 4));
    TR_BOOL_DEV(dscratch, items * 16 * (uint64_t)prog->nnodes);
    TR_BOOL_DEV(dacc, (uint64_t)nq * 64);
    TR_BOOL_DEV(dsums, (uint64_t)nq * 16);
    TR_BOOL_DEV(dhist, (hist_total ? hist_total : 1) * 8);
    TR_BOOL_DEV(dcof, (cof_total ? cof_total : 1) * 8);
#undef TR_BOOL_DEV
    if (!rc) {
        TR_LAUNCH(tr_bool_diff_kernel, items, d, p, dplans, dcones, items, slices, dscratch, dacc, dsums, dhist, dcof);
        rc = tr_cuda_sync();
    }
    uint64_t *acc = (uint64_t *)calloc((size_t)nq, 64);
    int64_t *sums = (int64_t *)calloc((size_t)nq, 16);
    if (!rc && (!acc || !sums)) rc = -2;
    if (!rc) rc = tr_cuda_copy_out(acc, dacc, (size_t)nq * 64);
    if (!rc) rc = tr_cuda_copy_out(sums, dsums, (size_t)nq * 16);
    if (!rc && hist_total) rc = tr_cuda_copy_out(hist, dhist, (size_t)(hist_total * 8));
    if (!rc && cof_total) rc = tr_cuda_copy_out(cof, dcof, (size_t)(cof_total * 8));
    if (!rc)
        for (int32_t j = 0; j < nq; j++) {
            tr_bool_result *R = &results[j];
            const uint64_t *a = acc + 8 * (uint64_t)j;
            R->f_true = a[0]; R->f_false = a[1]; R->f_unknown = a[2]; R->known = a[3]; R->unknown = a[4];
            R->xor_true = a[5]; R->xor_false = a[6]; R->nonzero = a[7];
            R->sum.lo = sums[2 * j]; R->sum.hi = sums[2 * j + 1];
        }
    for (int i = 0; i < nmem; i++) tr_cuda_free(mem[i]);
    free(plans); free(cones); free(acc); free(sums);
    return rc;
}
#ifdef __cplusplus
}
#endif
