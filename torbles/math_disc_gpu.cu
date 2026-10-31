/* DISC-001 on a CUDA device: exhaustive verification of one affine / sum / product hypothesis over every used row,
 * with the semantics of tr_disc_verify (math_disc.h). Each thread accumulates its rows locally and adds its counts,
 * extremes and smallest violating row with atomics, so the evidence does not depend on thread order.
 * SPDX-License-Identifier: Apache-2.0 */
#include "math_cuda.h"
#include "math_disc.h"

TR_KERNEL void tr_disc_verify_kernel(tr_disc_view v, int64_t scale, int64_t bias, uint64_t *acc, int64_t *ext) {
    uint64_t checked = 0, excluded = 0, violations = 0, witness = UINT64_MAX, overflow = 0;
    int64_t lo_l = INT64_MAX, hi_l = INT64_MIN, lo_r = INT64_MAX, hi_r = INT64_MIN;
    TR_GRID_LOOP(row, v.rows) {
        int64_t x = 0, y = 0;
        int s = tr_disc_row(&v, row, scale, bias, &x, &y);
        if (s == 0) continue;
        if (s == 1) { excluded++; continue; }
        if (s < 0) { overflow = 1; continue; }
        checked++;
        if (x < lo_l) lo_l = x;
        if (x > hi_l) hi_l = x;
        if (y < lo_r) lo_r = y;
        if (y > hi_r) hi_r = y;
        if (s == 3) { violations++; if (row < witness) witness = row; }
    }
    if (checked) tr_atomic_add_u64(&acc[0], checked);
    if (excluded) tr_atomic_add_u64(&acc[1], excluded);
    if (violations) { tr_atomic_add_u64(&acc[2], violations); tr_atomic_min_u64(&acc[3], witness); }
    if (overflow) tr_atomic_or_u64(&acc[4], 1);
    if (checked) {
        tr_atomic_min_i64(&ext[0], lo_l); tr_atomic_max_i64(&ext[1], hi_l);
        tr_atomic_min_i64(&ext[2], lo_r); tr_atomic_max_i64(&ext[3], hi_r);
    }
}

#ifdef __cplusplus
extern "C" {
#endif
/* budget: NULL = the defaults of math_budgets.h; the device copies (tr_disc_bytes) are checked against its bytes */
int tr_mgpu_disc_verify(const tr_disc_view *view, int64_t scale, int64_t bias, const tr_math_budget *budget,
                        tr_disc_evidence *e, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!view || !e || (view->arity != 1 && view->arity != 2) ||
        (view->rows && (!view->used || !view->codes[0] || !view->codes[1] || (view->arity == 2 && !view->codes[2]))))
        return -1;
    tr_math_budget dflt;
    if (!budget) { tr_budget_default(&dflt); budget = &dflt; }
    int rc = tr_budget_check_one(budget, TR_BUDGET_BYTES, tr_disc_bytes(view), refusal);
    if (rc) return rc;
    tr_disc_init(e);
    tr_disc_view d = *view;
    uint64_t words = (view->rows + 63) / 64, *dused = NULL, *dacc = NULL;
    uint64_t acc[5] = {0, 0, 0, UINT64_MAX, 0};
    int64_t *dext = NULL, ext[4] = {INT64_MAX, INT64_MIN, INT64_MAX, INT64_MIN};
    uint32_t *dcol[6] = {NULL, NULL, NULL, NULL, NULL, NULL};
    rc = tr_cuda_alloc((void **)&dacc, sizeof acc);
    if (!rc) rc = tr_cuda_alloc((void **)&dext, sizeof ext);
    if (!rc) rc = tr_cuda_copy_in(dacc, acc, sizeof acc);
    if (!rc) rc = tr_cuda_copy_in(dext, ext, sizeof ext);
    if (!rc && view->rows) {
        rc = tr_cuda_alloc((void **)&dused, (size_t)words * 8);
        if (!rc) rc = tr_cuda_copy_in(dused, view->used, (size_t)words * 8);
        d.used = dused;
        for (int i = 0; !rc && i <= view->arity; i++) {
            rc = tr_cuda_alloc((void **)&dcol[i], (size_t)view->rows * 4);
            if (!rc) rc = tr_cuda_copy_in(dcol[i], view->codes[i], (size_t)view->rows * 4);
            d.codes[i] = dcol[i];
            if (!rc && view->validity[i]) {
                rc = tr_cuda_alloc((void **)&dcol[3 + i], (size_t)view->rows * 4);
                if (!rc) rc = tr_cuda_copy_in(dcol[3 + i], view->validity[i], (size_t)view->rows * 4);
                d.validity[i] = dcol[3 + i];
            }
        }
        if (!rc) { TR_LAUNCH(tr_disc_verify_kernel, view->rows, d, scale, bias, dacc, dext); rc = tr_cuda_sync(); }
    }
    if (!rc) rc = tr_cuda_copy_out(acc, dacc, sizeof acc);
    if (!rc) rc = tr_cuda_copy_out(ext, dext, sizeof ext);
    tr_cuda_free(dacc); tr_cuda_free(dext); tr_cuda_free(dused);
    for (int i = 0; i < 6; i++) tr_cuda_free(dcol[i]);
    if (rc) return rc;
    e->checked = acc[0]; e->excluded = acc[1]; e->violations = acc[2]; e->witness = acc[3]; e->overflow = acc[4] != 0;
    e->min_left = ext[0]; e->max_left = ext[1]; e->min_right = ext[2]; e->max_right = ext[3];
    e->usable = tr_disc_decide(view, scale, e);
    return e->overflow ? -3 : 0;                                 /* TR_MATH_OVERFLOW */
}
#ifdef __cplusplus
}
#endif
