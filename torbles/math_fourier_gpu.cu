/* SPEC-001 on a CUDA device: the feature sums of a fixed spectral basis over the rows of a scope, with exactly the
 * integers of tr_fourier_sums (math_fourier.c): the same inline functions of math_fourier.h reduce every row's angle
 * exactly, enclose its sine and cosine (CALC-003's certified fixed-point series) and turn the weighted enclosures
 * outward into integers floor(lo 2^s), ceil(hi 2^s); the sums are exact 128-bit additions, so the device sums equal
 * the CPU's bit for bit whatever the order of the threads.
 *
 *   counts     work item = a slice of rows (rows s, s + S, s + 2S, ...): selected rows, counted rows and the exact
 *              weight sum per group, local while the group stays the same, then one atomic add per counter (the
 *              weight sum with the two-word 128-bit atomic tr_atomic_add_wide)
 *   features   work item = (frequency f, slice of rows): the four outward integers of every counted row (w cos and
 *              w sin, lower and upper), accumulated locally while the group stays the same and added to the group's
 *              128-bit sums with tr_atomic_add_wide
 *
 * The host checks the views and budgets every device copy before allocating (tr_fourier_device_bytes, resource
 * bytes); the plan carries the proof that no partial or total sum leaves 2^126 (math_fourier.h), so the two-word
 * atomics never wrap. A row whose bounds could not be represented (impossible under that proof) sets a flag and the
 * entry point returns TR_MATH_OVERFLOW instead of a sum; a group id out of range returns TR_MATH_INVALID.
 * SPDX-License-Identifier: Apache-2.0 */
#include "math_cuda.h"
#include "math_fourier.h"
#include "math_handles.h"

TR_MGPU_ABI(tr_fourier_plan_view, TR_FOURIER_ABI);   /* structs passed to the CUDA library (math_handles.h) */

#define TR_FOURIER_WORK (UINT64_C(1) << 19)      /* about this many work items per launch */
#define TR_FOURIER_MIN_ROWS 32                    /* and at least this many rows per work item */

TR_DEVICE void tr_fourier_flush(int64_t *sums, uint64_t cell, const __int128 *acc) {
    for (uint64_t c = 0; c < 4; c++)
        if (acc[c]) tr_atomic_add_wide(&sums[2 * (4 * cell + c)], &sums[2 * (4 * cell + c) + 1], acc[c]);
}

TR_KERNEL void tr_fourier_count_kernel(tr_fourier_plan_view pl, tr_scope_view v, uint64_t slices, uint64_t *rows,
                                       uint64_t *valid, int64_t *wsum, uint64_t *flags) {
    TR_GRID_LOOP(s, slices) {
        uint32_t cur = 0;
        uint64_t cr = 0, cv = 0;
        __int128 cw = 0;
        for (uint64_t r = s; r < v.rows; r += slices) {
            if (!tr_scope_is_selected(&v, r)) continue;
            uint32_t g = tr_scope_group(&v, r);
            if ((uint64_t)g >= pl.ngroups) { tr_atomic_or_u64(flags, 2); continue; }
            if (cr && g != cur) {
                tr_atomic_add_u64(&rows[cur], cr);
                if (cv) tr_atomic_add_u64(&valid[cur], cv);
                if (cw) tr_atomic_add_wide(&wsum[2 * (uint64_t)cur], &wsum[2 * (uint64_t)cur + 1], cw);
                cr = cv = 0;
                cw = 0;
            }
            cur = g; cr++;
            if (tr_fourier_row_ok(&pl, &v, r)) { cv++; cw += tr_fourier_row_weight(&pl, &v, r); }
        }
        if (cr) {
            tr_atomic_add_u64(&rows[cur], cr);
            if (cv) tr_atomic_add_u64(&valid[cur], cv);
            if (cw) tr_atomic_add_wide(&wsum[2 * (uint64_t)cur], &wsum[2 * (uint64_t)cur + 1], cw);
        }
    }
}

TR_KERNEL void tr_fourier_feature_kernel(tr_fourier_plan_view pl, tr_scope_view v, uint64_t slices, int64_t *sums,
                                         uint64_t *flags) {
    TR_GRID_LOOP(i, pl.nfreq * slices) {
        uint64_t f = i / slices, s = i % slices;
        uint32_t cur = 0;
        int have = 0;
        __int128 acc[4] = {0, 0, 0, 0};
        for (uint64_t r = s; r < v.rows; r += slices) {
            if (!tr_fourier_row_ok(&pl, &v, r)) continue;
            uint32_t g = tr_scope_group(&v, r);
            if ((uint64_t)g >= pl.ngroups) { tr_atomic_or_u64(flags, 2); continue; }
            if (have && g != cur) {
                tr_fourier_flush(sums, (uint64_t)cur * pl.nfreq + f, acc);
                acc[0] = acc[1] = acc[2] = acc[3] = 0;
            }
            cur = g; have = 1;
            __int128 t[4];
            if (!tr_fourier_row_terms(&pl, &v, f, r, t)) { tr_atomic_or_u64(flags, 1); continue; }
            for (int c = 0; c < 4; c++) acc[c] += t[c];
        }
        if (have) tr_fourier_flush(sums, (uint64_t)cur * pl.nfreq + f, acc);
    }
}

static uint64_t tr_fourier_slices(uint64_t rows, uint64_t work) {
    uint64_t s = TR_FOURIER_WORK / (work ? work : 1), cap = (rows + TR_FOURIER_MIN_ROWS - 1) / TR_FOURIER_MIN_ROWS;
    if (s > cap) s = cap;
    return s ? s : 1;
}

#ifdef __cplusplus
extern "C" {
#endif
int tr_mgpu_fourier_sums(const tr_fourier_plan_view *pl, const tr_scope_view *view, tr_math_i128 *sums, uint64_t *rows,
                         uint64_t *valid, tr_math_i128 *wsum, const tr_math_budget *budget, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!tr_fourier_plan_view_valid(pl, view) || !sums || !rows || !valid || !wsum) return TR_MATH_INVALID;
    tr_math_budget dflt;
    if (!budget) { tr_budget_default(&dflt); budget = &dflt; }
    const uint64_t G = pl->ngroups, m = pl->nfreq, cells = G * m, n = view->rows, md = m * (uint64_t)pl->ndims;
    tr_math_need need;
    memset(&need, 0, sizeof need);
    need.need[TR_BUDGET_STATES] = cells;
    need.need[TR_BUDGET_BYTES] = tr_fourier_device_bytes(pl, view);
    int rc = tr_budget_check(budget, &need, refusal);
    if (rc) return rc;
    memset(sums, 0, (size_t)cells * 4 * sizeof(tr_math_i128));
    memset(rows, 0, (size_t)G * 8);
    memset(valid, 0, (size_t)G * 8);
    memset(wsum, 0, (size_t)G * sizeof(tr_math_i128));
    if (!view->selected) return TR_MATH_OK;                      /* nothing selected: every sum is 0 */
    tr_scope_view d = *view;
    tr_fourier_plan_view p = *pl;
    void *mem[3 * TR_SCOPE_MAX_COLUMNS + 12];
    int nmem = 0;
    uint64_t *dsel = NULL, *drows = NULL, *dvalid = NULL, *dflags = NULL, *dk = NULL, flags = 0;
    uint32_t *dgroups = NULL;
    int64_t *dsums = NULL, *dwsum = NULL;
    int32_t *dksign = NULL;
    uint8_t *dkexact = NULL;
#define TR_FOURIER_DEV(ptr, bytes) do { if (!rc) { rc = tr_cuda_alloc((void **)&(ptr), (size_t)(bytes)); if (!rc) mem[nmem++] = (void *)(ptr); } } while (0)
    TR_FOURIER_DEV(dsel, ((n + 63) / 64) * 8);
    if (!rc) rc = tr_cuda_copy_in(dsel, view->selection, (size_t)(((n + 63) / 64) * 8));
    d.selection = dsel;
    if (view->groups) {
        TR_FOURIER_DEV(dgroups, n * 4);
        if (!rc) rc = tr_cuda_copy_in(dgroups, view->groups, (size_t)n * 4);
        d.groups = dgroups;
    }
    uint32_t used = 0;
    for (int32_t i = 0; i < pl->ndims; i++) used |= (uint32_t)1 << pl->column[i];
    if (pl->weight >= 0) used |= (uint32_t)1 << pl->weight;
    for (int32_t c = 0; c < view->ncols; c++) {
        tr_scope_view_column *k = &d.col[c];
        if (!((used >> c) & 1)) { k->codes = NULL; k->values = NULL; k->valid = NULL; continue; }   /* never read */
        if (view->col[c].values) {
            int64_t *q = NULL;
            TR_FOURIER_DEV(q, n * 8);
            if (!rc) rc = tr_cuda_copy_in(q, view->col[c].values, (size_t)n * 8);
            k->values = q;
        } else {
            uint32_t *q = NULL;
            TR_FOURIER_DEV(q, n * 4);
            if (!rc) rc = tr_cuda_copy_in(q, view->col[c].codes, (size_t)n * 4);
            k->codes = q;
        }
        if (view->col[c].valid) {
            uint8_t *q = NULL;
            TR_FOURIER_DEV(q, n);
            if (!rc) rc = tr_cuda_copy_in(q, view->col[c].valid, (size_t)n);
            k->valid = q;
        }
    }
    TR_FOURIER_DEV(dk, md * 32);
    if (!rc) rc = tr_cuda_copy_in(dk, pl->k, (size_t)md * 32);
    TR_FOURIER_DEV(dksign, md * 4);
    if (!rc) rc = tr_cuda_copy_in(dksign, pl->ksign, (size_t)md * 4);
    TR_FOURIER_DEV(dkexact, md);
    if (!rc) rc = tr_cuda_copy_in(dkexact, pl->kexact, (size_t)md);
    p.k = dk; p.ksign = dksign; p.kexact = dkexact;
    TR_FOURIER_DEV(dsums, cells * 64);
    TR_FOURIER_DEV(drows, G * 8);
    TR_FOURIER_DEV(dvalid, G * 8);
    TR_FOURIER_DEV(dwsum, G * 16);
    TR_FOURIER_DEV(dflags, 8);
#undef TR_FOURIER_DEV
    if (!rc) {
        uint64_t slices = tr_fourier_slices(n, 1);
        TR_LAUNCH(tr_fourier_count_kernel, slices, p, d, slices, drows, dvalid, dwsum, dflags);
        rc = tr_cuda_sync();
    }
    if (!rc) {
        uint64_t slices = tr_fourier_slices(n, m);
        TR_LAUNCH(tr_fourier_feature_kernel, m * slices, p, d, slices, dsums, dflags);
        rc = tr_cuda_sync();
    }
    if (!rc) rc = tr_cuda_copy_out(&flags, dflags, 8);
    if (!rc && !flags) rc = tr_cuda_copy_out(sums, dsums, (size_t)cells * 64);
    if (!rc && !flags) rc = tr_cuda_copy_out(rows, drows, (size_t)G * 8);
    if (!rc && !flags) rc = tr_cuda_copy_out(valid, dvalid, (size_t)G * 8);
    if (!rc && !flags) rc = tr_cuda_copy_out(wsum, dwsum, (size_t)G * 16);
    for (int i = 0; i < nmem; i++) tr_cuda_free(mem[i]);
    if (rc) return rc;
    if (flags & 2) return TR_MATH_INVALID;
    return flags ? (int)TR_MATH_OVERFLOW : (int)TR_MATH_OK;
}
#ifdef __cplusplus
}
#endif
