/* GEOM-002 on a CUDA device: the batched reconstruction (and form) checks of tr_elim_check_view (math_elim.h) with
 * the same results bit for bit.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Work items (pair chunk, row slice): each selected row of the slice is checked for every pair of the chunk by the
 * shared inline tr_elim_pair_row (the target's and the form's NULL domains, their exact W-word integer forms under the
 * bound the host proved, the form's value against the target column's range); counted locally, then one atomic add
 * per nonzero counter, an atomic minimum for each first mismatching row (so the first row does not depend on the
 * order threads run in) and atomic maxima for the bit lengths. No barrier, no shared memory. The host validates the
 * view (tr_elim_pair_view_check: structure, and the view's extremes within the magnitudes the bounds were proven for)
 * and budgets the device copies (tr_elim_check_device_need) before anything is copied. */
#include "math_cuda.h"
#include "math_elim.h"

TR_MGPU_ABI(tr_elim_pair_view, TR_ELIM_ABI);       /* structs passed to the CUDA library (math_handles.h) */
TR_MGPU_ABI(tr_elim_check, TR_ELIM_ABI);

TR_KERNEL void tr_elim_check_kernel(tr_scope_view v, tr_elim_pair_view p, uint64_t slices, uint64_t per, uint64_t nchunks,
                                    uint64_t *acc) {
    const uint64_t nw = (v.rows + 63) / 64;
    TR_GRID_LOOP(i, nchunks * slices) {
        const uint64_t chunk = i / slices, s = i % slices;
        const uint64_t w0 = s * per, w1 = w0 + per < nw ? w0 + per : nw;
        const int32_t j0 = (int32_t)(chunk * TR_ELIM_PAIR_CHUNK);
        const int32_t j1 = j0 + TR_ELIM_PAIR_CHUNK < p.npairs ? j0 + TR_ELIM_PAIR_CHUNK : p.npairs;
        uint64_t local[TR_ELIM_PAIR_CHUNK * TR_ELIM_ACC];
        for (int32_t q = 0; q < TR_ELIM_PAIR_CHUNK; q++) tr_elim_acc_init(local + (uint64_t)q * TR_ELIM_ACC);
        for (uint64_t w = w0; w < w1; w++) {
            uint64_t bits = v.selection[w];
            while (bits) {
                const uint64_t r = w * 64 + (uint64_t)tr_ipoly_ctz(bits);
                bits &= bits - 1;
                for (int32_t j = j0; j < j1; j++) tr_elim_pair_row(&v, &p, j, r, local + (uint64_t)(j - j0) * TR_ELIM_ACC);
            }
        }
        for (int32_t j = j0; j < j1; j++) {
            const uint64_t *l = local + (uint64_t)(j - j0) * TR_ELIM_ACC;
            uint64_t *a = acc + (uint64_t)j * TR_ELIM_ACC;
            for (int k = 0; k < 10; k++) if (l[k]) tr_atomic_add_u64(&a[k], l[k]);
            for (int k = 10; k < 14; k++) if (l[k] != UINT64_MAX) tr_atomic_min_u64(&a[k], l[k]);
            for (int k = 14; k < 16; k++) if (l[k]) tr_atomic_max_u64(&a[k], l[k]);
        }
    }
}

/* copies the columns the kernel reads (codes or values, validity) and points the view at them */
static int tr_elim_gpu_columns(const tr_scope_view *view, tr_scope_view *d, uint32_t cols, void **mem, int *nmem) {
    int rc = 0;
    const uint64_t rows = view->rows;
    for (int32_t c = 0; c < view->ncols; c++) {
        tr_scope_view_column *k = &d->col[c];
        if (!((cols >> c) & 1)) { k->codes = NULL; k->values = NULL; k->valid = NULL; continue; }
        if (view->col[c].values) {
            int64_t *ptr = NULL;
            rc = tr_cuda_alloc((void **)&ptr, (size_t)rows * 8);
            if (rc) return rc;
            mem[(*nmem)++] = ptr;
            rc = tr_cuda_copy_in(ptr, view->col[c].values, (size_t)rows * 8);
            k->values = ptr;
        } else {
            uint32_t *ptr = NULL;
            rc = tr_cuda_alloc((void **)&ptr, (size_t)rows * 4);
            if (rc) return rc;
            mem[(*nmem)++] = ptr;
            rc = tr_cuda_copy_in(ptr, view->col[c].codes, (size_t)rows * 4);
            k->codes = ptr;
        }
        if (!rc && view->col[c].valid) {
            uint8_t *ptr = NULL;
            rc = tr_cuda_alloc((void **)&ptr, (size_t)rows);
            if (rc) return rc;
            mem[(*nmem)++] = ptr;
            rc = tr_cuda_copy_in(ptr, view->col[c].valid, (size_t)rows);
            k->valid = ptr;
        }
        if (rc) return rc;
    }
    return 0;
}

#ifdef __cplusplus
extern "C" {
#endif
int tr_mgpu_elim_check(const tr_scope_view *view, const tr_elim_pair_view *pairs, const tr_math_budget *budget,
                       tr_elim_check *out, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    int rc = tr_elim_pair_view_check(view, pairs);
    if (rc) return rc;
    if (pairs->npairs && !out) return -1;                                       /* TR_MATH_INVALID */
    tr_math_budget dflt;
    if (!budget) { tr_budget_default(&dflt); budget = &dflt; }
    tr_math_need need;
    tr_elim_check_device_need(view, pairs, &need);
    rc = tr_budget_check(budget, &need, refusal);
    if (rc) return rc;
    const int32_t n = pairs->npairs;
    if (!n) return 0;
    uint64_t *acc = (uint64_t *)calloc((size_t)n * TR_ELIM_ACC, 8);
    if (!acc) return -2;                                                        /* TR_MATH_MEMORY */
    for (int32_t j = 0; j < n; j++) tr_elim_acc_init(acc + (uint64_t)j * TR_ELIM_ACC);
    if (view->selected) {
        const tr_ident_cand_view *cand = &pairs->cand;
        void *mem[3 * TR_SCOPE_MAX_COLUMNS + 16];
        int nmem = 0;
        tr_scope_view d = *view;
        tr_elim_pair_view dp = *pairs;
        d.groups = NULL;
        const uint64_t nw = (view->rows + 63) / 64, ncand = (uint64_t)cand->ncand;
        uint64_t *dsel = NULL, *doff = NULL, *dacc = NULL;
        uint32_t *dsup = NULL;
        int32_t *dwords = NULL, *dtarget = NULL;
        uint8_t *dexp = NULL;
        int64_t *dcoef = NULL, *dden = NULL;
#define TR_ELIM_DEV(ptr, bytes) do { if (!rc) { rc = tr_cuda_alloc((void **)&(ptr), (size_t)(bytes)); if (!rc) mem[nmem++] = (void *)(ptr); } } while (0)
#define TR_ELIM_IN(dst, src, bytes) do { TR_ELIM_DEV(dst, bytes); if (!rc && (uint64_t)(bytes) != 0) rc = tr_cuda_copy_in(dst, src, (size_t)(bytes)); } while (0)
        TR_ELIM_IN(dsel, view->selection, nw * 8);
        d.selection = dsel;
        uint32_t cols = 0;
        for (int32_t i = 0; i < cand->nvars; i++) cols |= (uint32_t)1 << cand->var[i];
        if (!rc) rc = tr_elim_gpu_columns(view, &d, cols, mem, &nmem);
        TR_ELIM_IN(doff, cand->offset, (ncand + 1) * 8);
        TR_ELIM_IN(dsup, cand->support, ncand * 4);
        TR_ELIM_IN(dwords, cand->words, ncand * 4);
        TR_ELIM_IN(dexp, cand->exponents, cand->nterms * (uint64_t)cand->nvars);
        TR_ELIM_IN(dcoef, cand->coefficients, cand->nterms * 8);
        TR_ELIM_IN(dtarget, pairs->target, (uint64_t)n * 4);
        TR_ELIM_IN(dden, pairs->den, (uint64_t)n * 8);
        TR_ELIM_IN(dacc, acc, (uint64_t)n * TR_ELIM_ACC * 8);
#undef TR_ELIM_IN
#undef TR_ELIM_DEV
        dp.cand.offset = doff; dp.cand.support = dsup; dp.cand.words = dwords; dp.cand.exponents = dexp;
        dp.cand.coefficients = dcoef; dp.target = dtarget; dp.den = dden;
        uint64_t slices = (nw + TR_ELIM_SLICE_WORDS - 1) / TR_ELIM_SLICE_WORDS, per;
        if (slices > TR_ELIM_MAX_SLICES) slices = TR_ELIM_MAX_SLICES;
        if (!slices) slices = 1;
        per = (nw + slices - 1) / slices;
        if (!per) per = 1;
        const uint64_t nchunks = ((uint64_t)n + TR_ELIM_PAIR_CHUNK - 1) / TR_ELIM_PAIR_CHUNK;
        if (!rc) { TR_LAUNCH(tr_elim_check_kernel, nchunks * slices, d, dp, slices, per, nchunks, dacc); rc = tr_cuda_sync(); }
        if (!rc) rc = tr_cuda_copy_out(acc, dacc, (size_t)n * TR_ELIM_ACC * 8);
        for (int i = 0; i < nmem; i++) tr_cuda_free(mem[i]);
    }
    if (!rc) for (int32_t j = 0; j < n; j++) tr_elim_check_finish(view, pairs, j, acc + (uint64_t)j * TR_ELIM_ACC, &out[j]);
    free(acc);
    return rc;
}
#ifdef __cplusplus
}
#endif
