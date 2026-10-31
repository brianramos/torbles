/* CUDA pre-pass of the output budgets (FND-010): the exact number of distinct occupied cells, counted on the device
 * before any output is built. Same contract as tr_math_budget_count_cells (math_budgets.h); the budget also covers
 * the device copies of the inputs. SPDX-License-Identifier: Apache-2.0 */
#include "math_cuda.h"
#include "math_budgets.h"

/* one bit per occupied cell (atomic or: the result does not depend on thread order) */
TR_KERNEL void tr_budget_mark_kernel(const uint32_t *cells, const uint64_t *rows, uint64_t n, uint64_t ncells,
                                     uint64_t *bitmap, uint64_t *invalid) {
    TR_GRID_LOOP(i, n) {
        if (rows && !((rows[i >> 6] >> (i & 63)) & 1)) continue;
        uint32_t c = cells[i];
        if ((uint64_t)c >= ncells) { tr_atomic_or_u64(invalid, 1); continue; }
        tr_atomic_or_u64(&bitmap[c >> 6], (uint64_t)1 << (c & 63));
    }
}

TR_KERNEL void tr_budget_popcount_kernel(const uint64_t *bitmap, uint64_t words, uint64_t *count) {
    TR_GRID_LOOP(w, words) {
        uint64_t v = bitmap[w];
        if (v) tr_atomic_add_u64(count, (uint64_t)tr_popc64(v));
    }
}

#ifdef __cplusplus
extern "C" {
#endif
int tr_mgpu_budgets_count_cells(const tr_math_budget *b, const uint32_t *cells, const uint64_t *rows, uint64_t n,
                                uint64_t ncells, uint64_t *distinct, tr_math_refusal *r) {
    tr_budget_none(r);
    if (!b || !distinct || (n && !cells)) return -1;
    tr_math_need need;
    tr_budget_count_cells_need(n, ncells, rows != NULL, 1, &need);
    int rc = tr_budget_check(b, &need, r);
    if (rc) return rc;
    uint64_t words = (ncells + 63) / 64, *dbits = NULL, *dacc = NULL, *drows = NULL, acc[2] = {0, 0};
    uint32_t *dcells = NULL;
    rc = tr_cuda_alloc((void **)&dbits, (size_t)(words ? words : 1) * 8);
    if (!rc) rc = tr_cuda_alloc((void **)&dacc, sizeof acc);
    if (!rc && n) rc = tr_cuda_alloc((void **)&dcells, (size_t)n * 4);
    if (!rc && n && rows) rc = tr_cuda_alloc((void **)&drows, (size_t)((n + 63) / 64) * 8);
    if (!rc && n) rc = tr_cuda_copy_in(dcells, cells, (size_t)n * 4);
    if (!rc && n && rows) rc = tr_cuda_copy_in(drows, rows, (size_t)((n + 63) / 64) * 8);
    if (!rc && n) { TR_LAUNCH(tr_budget_mark_kernel, n, dcells, drows, n, ncells, dbits, dacc + 1); rc = tr_cuda_sync(); }
    if (!rc && words) { TR_LAUNCH(tr_budget_popcount_kernel, words, dbits, words, dacc); rc = tr_cuda_sync(); }
    if (!rc) rc = tr_cuda_copy_out(acc, dacc, sizeof acc);
    tr_cuda_free(dbits); tr_cuda_free(dacc); tr_cuda_free(dcells); tr_cuda_free(drows);
    if (rc) return rc;
    if (acc[1]) return -1;                                   /* TR_MATH_INVALID: a cell id >= ncells */
    rc = tr_budget_check_one(b, TR_BUDGET_OUTPUT, acc[0], r);
    if (rc) return rc;
    *distinct = acc[0];
    return 0;
}
#ifdef __cplusplus
}
#endif
