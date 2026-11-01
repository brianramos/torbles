/* UNC-002 on a CUDA device: the batched evaluation of an approximation over the selected rows of a scope column with
 * its error accounting, the semantics of tr_approx_eval_view_rows (math_approx.h) and the same results bit for bit.
 * SPDX-License-Identifier: Apache-2.0
 *
 * tr_mgpu_approx_eval: one work item per 64-row word evaluates each selected row with the inline row evaluator
 * (tr_approx_eval_row: the enclosure of x, interval Clenshaw / Horner with the coefficient enclosures, the certified
 * error, the clamp to the certified range; or the binary64 Horner value with its bound) and writes the row's own
 * output slots (lo, hi, status). The group sums are exact: each work item adds floor(lo 2^F) and ceil(hi 2^F) of its
 * rows into a local 128-bit accumulator while the group stays the same and flushes it with the two-word 128-bit
 * atomic add, the counts with 64-bit atomic adds, so the totals do not depend on the order of the threads. No barrier,
 * no shared memory. The host validates the view, proves |sums| < 2^126 (tr_approx_sums_fit) and checks the budget
 * before copying. */
#include "math_cuda.h"
#include "math_approx.h"

TR_MGPU_ABI(tr_approx_eval_view, TR_APPROX_ABI);     /* structs passed to the CUDA library (math_handles.h) */
TR_MGPU_ABI(tr_approx_group, TR_APPROX_ABI);
TR_MGPU_ABI(tr_approx_eval_stats, TR_APPROX_ABI);

TR_DEVICE void tr_approx_flush(tr_approx_group *groups, uint32_t g, __int128 lo, __int128 hi, uint64_t rows, uint64_t ok,
                               uint64_t nulls, uint64_t refused) {
    tr_approx_group *G = &groups[g];
    if (lo) tr_atomic_add_wide(&G->lo[0], &G->lo[1], lo);
    if (hi) tr_atomic_add_wide(&G->hi[0], &G->hi[1], hi);
    if (rows) tr_atomic_add_u64(&G->rows, rows);
    if (ok) tr_atomic_add_u64(&G->ok, ok);
    if (nulls) tr_atomic_add_u64(&G->nulls, nulls);
    if (refused) tr_atomic_add_u64(&G->refused, refused);
}

TR_KERNEL void tr_approx_eval_kernel(tr_approx_eval_view ev, tr_scope_view v, int32_t col, int32_t fbits, uint64_t nw,
                                     double *lo, double *hi, int8_t *status, tr_approx_group *groups, uint64_t *counts) {
    uint64_t c_ok = 0, c_null = 0, c_dom = 0, c_ovf = 0;
    TR_GRID_LOOP(w, nw) {
        uint64_t first = w * 64, last = first + 64 < v.rows ? first + 64 : v.rows;
        uint32_t cur = 0;
        int have = 0;
        __int128 slo = 0, shi = 0;
        uint64_t rows = 0, ok = 0, nulls = 0, refused = 0;
        for (uint64_t row = first; row < last; row++) {
            if (!((v.selection[w] >> (row & 63)) & 1)) {
                lo[row] = 0.0; hi[row] = 0.0; status[row] = TR_APPROX_ROW_UNSELECTED;
                continue;
            }
            tr_math_interval F;
            double val = 0.0;
            int st = tr_approx_eval_row(&ev, &v, col, row, &F, &val);
            if (st != TR_APPROX_ROW_OK) { F.lo = 0.0; F.hi = 0.0; }
            lo[row] = F.lo; hi[row] = F.hi; status[row] = (int8_t)st;
            if (st == TR_APPROX_ROW_OK) c_ok++;
            else if (st == TR_APPROX_ROW_NULL) c_null++;
            else if (st == TR_APPROX_ROW_DOMAIN) c_dom++;
            else c_ovf++;
            if (!groups) continue;
            uint32_t g = tr_scope_group(&v, row);
            if (have && g != cur) {
                tr_approx_flush(groups, cur, slo, shi, rows, ok, nulls, refused);
                slo = 0; shi = 0; rows = ok = nulls = refused = 0;
            }
            cur = g; have = 1;
            rows++;
            if (st == TR_APPROX_ROW_OK) {
                __int128 a, b;
                tr_approx_fixed(F, fbits, &a, &b);
                slo += a; shi += b; ok++;
            } else if (st == TR_APPROX_ROW_NULL) nulls++;
            else refused++;
        }
        if (groups && have) tr_approx_flush(groups, cur, slo, shi, rows, ok, nulls, refused);
    }
    if (c_ok) tr_atomic_add_u64(&counts[0], c_ok);
    if (c_null) tr_atomic_add_u64(&counts[1], c_null);
    if (c_dom) tr_atomic_add_u64(&counts[2], c_dom);
    if (c_ovf) tr_atomic_add_u64(&counts[3], c_ovf);
}

#ifdef __cplusplus
extern "C" {
#endif
int tr_mgpu_approx_eval(const tr_scope_view *view, int32_t col, const tr_approx_eval_view *ev, int32_t fbits,
                        double *lo, double *hi, int8_t *status, tr_approx_group *groups, tr_approx_eval_stats *stats,
                        const tr_math_budget *budget, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!view || !ev || !stats || col < 0 || col >= view->ncols || view->ncols > TR_SCOPE_MAX_COLUMNS ||
        (view->rows && !view->selection) || view->ngroups < 1 || (view->ngroups > 1 && view->rows && !view->groups) ||
        fbits < 0 || fbits > TR_APPROX_MAX_FBITS || (ev->mode != TR_APPROX_EVAL_SERIES && ev->mode != TR_APPROX_EVAL_FLOAT) ||
        ev->ncoef < 1 || ev->ncoef > TR_APPROX_MAX_DEGREE + 1 || !ev->coef || !(ev->rad.lo > 0) ||
        (view->rows && !view->col[col].codes && !view->col[col].values)) return TR_MATH_INVALID;
    if (groups && !tr_approx_sums_fit(ev, fbits, view->selected)) return TR_MATH_OVERFLOW;
    tr_math_budget dflt;
    if (!budget) { tr_budget_default(&dflt); budget = &dflt; }
    tr_math_need need;
    for (int i = 0; i < TR_BUDGET_RESOURCES; i++) need.need[i] = 0;
    need.need[TR_BUDGET_OUTPUT] = view->rows;
    need.need[TR_BUDGET_BYTES] = tr_approx_eval_device_bytes(view, col, ev->ncoef);
    int rc = tr_budget_check(budget, &need, refusal);
    if (rc) return rc;
    memset(stats, 0, sizeof *stats);
    stats->rows = view->rows; stats->selected = view->selected; stats->fbits = fbits; stats->groups = (int32_t)view->ngroups;
    if (groups) memset(groups, 0, (size_t)view->ngroups * sizeof *groups);
    const uint64_t rows = view->rows, nw = (rows + 63) / 64;
    if (!nw) return 0;
    void *dmem[12];
    int nmem = 0;
#define TR_APPROX_DEV(ptr, bytes) do { if (!rc) { rc = tr_cuda_alloc((void **)&(ptr), (size_t)(bytes)); if (!rc) dmem[nmem++] = (void *)(ptr); } } while (0)
#define TR_APPROX_IN(dst, src, bytes) do { TR_APPROX_DEV(dst, bytes); if (!rc && (uint64_t)(bytes) != 0) rc = tr_cuda_copy_in(dst, src, (size_t)(bytes)); } while (0)
    tr_scope_view d = *view;
    uint64_t *dsel = NULL, *dcounts = NULL;
    uint32_t *dgroups = NULL;
    double *dlo = NULL, *dhi = NULL;
    int8_t *dstatus = NULL;
    tr_approx_group *dg = NULL;
    tr_math_interval *dcoef = NULL;
    TR_APPROX_IN(dsel, view->selection, nw * 8);
    d.selection = dsel;
    d.groups = NULL;
    if (view->groups && view->ngroups > 1) { TR_APPROX_IN(dgroups, view->groups, rows * 4); d.groups = dgroups; }
    for (int32_t c = 0; c < view->ncols; c++) {
        tr_scope_view_column *k = &d.col[c];
        if (c != col) { k->codes = NULL; k->values = NULL; k->valid = NULL; continue; }
        if (view->col[c].values) { int64_t *p = NULL; TR_APPROX_IN(p, view->col[c].values, rows * 8); k->values = p; }
        else { uint32_t *p = NULL; TR_APPROX_IN(p, view->col[c].codes, rows * 4); k->codes = p; }
        if (view->col[c].valid) { uint8_t *p = NULL; TR_APPROX_IN(p, view->col[c].valid, rows); k->valid = p; }
    }
    TR_APPROX_IN(dcoef, ev->coef, (uint64_t)ev->ncoef * sizeof(tr_math_interval));
    TR_APPROX_DEV(dlo, rows * 8);
    TR_APPROX_DEV(dhi, rows * 8);
    TR_APPROX_DEV(dstatus, rows);
    if (groups) TR_APPROX_DEV(dg, view->ngroups * sizeof(tr_approx_group));
    TR_APPROX_DEV(dcounts, 4 * 8);
#undef TR_APPROX_IN
#undef TR_APPROX_DEV
    tr_approx_eval_view e = *ev;
    e.coef = dcoef;
    if (!rc) { TR_LAUNCH(tr_approx_eval_kernel, nw, e, d, col, fbits, nw, dlo, dhi, dstatus, dg, dcounts); rc = tr_cuda_sync(); }
    uint64_t counts[4] = {0, 0, 0, 0};
    if (!rc && lo) rc = tr_cuda_copy_out(lo, dlo, (size_t)rows * 8);
    if (!rc && hi) rc = tr_cuda_copy_out(hi, dhi, (size_t)rows * 8);
    if (!rc && status) rc = tr_cuda_copy_out(status, dstatus, (size_t)rows);
    if (!rc && groups) rc = tr_cuda_copy_out(groups, dg, (size_t)view->ngroups * sizeof(tr_approx_group));
    if (!rc) rc = tr_cuda_copy_out(counts, dcounts, sizeof counts);
    for (int i = 0; i < nmem; i++) tr_cuda_free(dmem[i]);
    if (rc) return rc;
    stats->ok = counts[0]; stats->nulls = counts[1]; stats->domain = counts[2]; stats->overflow = counts[3];
    return 0;
}
#ifdef __cplusplus
}
#endif
