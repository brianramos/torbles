/* CALC-003 on a CUDA device: the coefficient sums of a separable family over the rows of a scope, with exactly the
 * integers of tr_calc_sums (math_calc.c): the same inline functions of math_calc.h compute every row's enclosure and
 * its outward integers floor(lo 2^s), ceil(hi 2^s), and the sums are exact 128-bit additions, so the device sums
 * equal the CPU's bit for bit whatever the order of the threads.
 *
 *   counts    work item = a slice of rows: selected rows and counted rows per group (local while the group stays the
 *             same, then one atomic add)
 *   PHASE     kernel 1: one work item per row: whether it counts and its 2k factor enclosures (sin and cos of
 *             theta_ri, 120-bit midpoint-radius) in a device array; kernel 2: work item = (coefficient S, slice of
 *             rows): the product of S's factors times the weight, accumulated locally per group and added to the
 *             group's 128-bit sums with exact two-word atomics (tr_calc_atomic_add_wide)
 *   EXP       one kernel, work item = a slice of rows: the exponential of each counted row, accumulated locally
 *
 * The host checks the views and budgets every device copy before allocating (tr_calc_device_bytes, resource bytes),
 * and the plan carries the proof that no sum leaves int128 (math_calc.h); a row whose bounds cannot be represented
 * (impossible under that proof) sets a flag and the entry point returns TR_MATH_OVERFLOW instead of a sum.
 * SPDX-License-Identifier: Apache-2.0 */
#include "math_cuda.h"
#include "math_calc.h"
#include "math_handles.h"

TR_MGPU_ABI(tr_calc_plan_view, TR_CALC_ABI);   /* structs passed to the CUDA library (math_handles.h) */
TR_MGPU_ABI(tr_scope_view, 1);
TR_MGPU_ABI(tr_scope_view_column, 1);
TR_MGPU_ABI(tr_math_i128, 1);

#define TR_CALC_WORK (UINT64_C(1) << 19)      /* about this many work items per launch */
#define TR_CALC_MIN_ROWS 32                    /* and at least this many rows per work item */

/* exact signed 128-bit accumulation in two words: lo += low word, hi += high word + the carry of that add */
TR_DEVICE void tr_calc_atomic_add_wide(int64_t *lo, int64_t *hi, __int128 x) {
    uint64_t xl = (uint64_t)x, xh = (uint64_t)((unsigned __int128)x >> 64);
    uint64_t old = tr_atomic_add_u64((uint64_t *)lo, xl);
    uint64_t carry = (old + xl) < old;
    if (xh + carry) tr_atomic_add_u64((uint64_t *)hi, xh + carry);
}
TR_DEVICE void tr_calc_flush(int64_t *sums, uint64_t cell, __int128 L, __int128 H) {
    if (L) tr_calc_atomic_add_wide(&sums[4 * cell], &sums[4 * cell + 1], L);
    if (H) tr_calc_atomic_add_wide(&sums[4 * cell + 2], &sums[4 * cell + 3], H);
}

TR_KERNEL void tr_calc_count_kernel(tr_calc_plan_view pl, tr_scope_view v, uint64_t slices, uint64_t *rows,
                                    uint64_t *valid) {
    TR_GRID_LOOP(s, slices) {
        uint32_t cur = 0;
        uint64_t cr = 0, cv = 0;
        for (uint64_t r = s; r < v.rows; r += slices) {
            if (!tr_scope_is_selected(&v, r)) continue;
            uint32_t g = tr_scope_group(&v, r);
            if (cr && g != cur) {
                tr_atomic_add_u64(&rows[cur], cr);
                if (cv) tr_atomic_add_u64(&valid[cur], cv);
                cr = cv = 0;
            }
            cur = g; cr++;
            if (tr_calc_row_ok(&pl, &v, r)) cv++;
        }
        if (cr) { tr_atomic_add_u64(&rows[cur], cr); if (cv) tr_atomic_add_u64(&valid[cur], cv); }
    }
}

TR_KERNEL void tr_calc_factor_kernel(tr_calc_plan_view pl, tr_scope_view v, tr_calc_fx_words *fac, uint8_t *ok) {
    TR_GRID_LOOP(r, v.rows) {
        int good = tr_calc_row_ok(&pl, &v, r);
        ok[r] = (uint8_t)good;
        if (!good) continue;
        for (int32_t i = 0; i < pl.nfactors; i++) {
            tr_calc_fx sn, cs;
            tr_calc_sincos_reduce(pl.k[i], pl.ksign[i], tr_scope_value(&v, pl.column[i], r), 0, &sn, &cs);
            tr_calc_fx_store(&fac[(r * (uint64_t)pl.nfactors + (uint64_t)i) * 2], sn);
            tr_calc_fx_store(&fac[(r * (uint64_t)pl.nfactors + (uint64_t)i) * 2 + 1], cs);
        }
    }
}

TR_KERNEL void tr_calc_phase_kernel(tr_calc_plan_view pl, tr_scope_view v, const tr_calc_fx_words *fac,
                                    const uint8_t *ok, uint64_t slices, int64_t *sums, uint64_t *flags) {
    const uint64_t k = (uint64_t)pl.nfactors;
    TR_GRID_LOOP(i, pl.nsubsets * slices) {
        uint64_t S = i / slices, s = i % slices;
        uint32_t cur = 0;
        int have = 0;
        __int128 accL = 0, accH = 0;
        for (uint64_t r = s; r < v.rows; r += slices) {
            if (!ok[r]) continue;
            uint32_t g = tr_scope_group(&v, r);
            if (have && g != cur) { tr_calc_flush(sums, (uint64_t)cur * pl.nsubsets + S, accL, accH); accL = accH = 0; }
            cur = g; have = 1;
            const tr_calc_fx_words *f = fac + r * k * 2;
            tr_calc_fx p = tr_calc_fx_load(&f[S & 1]);              /* the order of tr_calc_subset */
            for (uint64_t j = 1; j < k; j++) p = tr_calc_fx_mul(p, tr_calc_fx_load(&f[2 * j + ((S >> j) & 1)]));
            __int128 L, H;
            if (!tr_calc_scaled(tr_calc_row_weight(&pl, &v, r), p, pl.scale, &L, &H)) { tr_atomic_or_u64(flags, 1); continue; }
            accL += L; accH += H;
        }
        if (have) tr_calc_flush(sums, (uint64_t)cur * pl.nsubsets + S, accL, accH);
    }
}

TR_KERNEL void tr_calc_exp_kernel(tr_calc_plan_view pl, tr_scope_view v, uint64_t slices, int64_t *sums,
                                  uint64_t *flags) {
    TR_GRID_LOOP(s, slices) {
        uint32_t cur = 0;
        int have = 0;
        __int128 accL = 0, accH = 0;
        for (uint64_t r = s; r < v.rows; r += slices) {
            if (!tr_calc_row_ok(&pl, &v, r)) continue;
            uint32_t g = tr_scope_group(&v, r);
            if (have && g != cur) { tr_calc_flush(sums, (uint64_t)cur, accL, accH); accL = accH = 0; }
            cur = g; have = 1;
            __int128 L, H;
            if (!tr_calc_exp_row(&pl, &v, r, &L, &H)) { tr_atomic_or_u64(flags, 1); continue; }
            accL += L; accH += H;
        }
        if (have) tr_calc_flush(sums, (uint64_t)cur, accL, accH);
    }
}

static uint64_t tr_calc_slices(uint64_t rows, uint64_t work) {
    uint64_t s = TR_CALC_WORK / (work ? work : 1), cap = (rows + TR_CALC_MIN_ROWS - 1) / TR_CALC_MIN_ROWS;
    if (s > cap) s = cap;
    return s ? s : 1;
}

#ifdef __cplusplus
extern "C" {
#endif
int tr_mgpu_calc_sums(const tr_calc_plan_view *pl, const tr_scope_view *view, tr_math_i128 *sums, uint64_t *rows,
                      uint64_t *valid, const tr_math_budget *budget, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!tr_calc_plan_view_valid(pl, view) || !sums || !rows || !valid) return TR_MATH_INVALID;
    tr_math_budget dflt;
    if (!budget) { tr_budget_default(&dflt); budget = &dflt; }
    uint64_t G = pl->ngroups, m = pl->nsubsets, cells = G * m, n = view->rows;
    tr_math_need need;
    memset(&need, 0, sizeof need);
    need.need[TR_BUDGET_STATES] = cells;
    need.need[TR_BUDGET_BYTES] = tr_calc_device_bytes(pl, view);
    int rc = tr_budget_check(budget, &need, refusal);
    if (rc) return rc;
    memset(sums, 0, (size_t)cells * 2 * sizeof(tr_math_i128));
    memset(rows, 0, (size_t)G * 8);
    memset(valid, 0, (size_t)G * 8);
    if (!view->selected) return TR_MATH_OK;                    /* nothing selected: every sum is 0 */
    tr_scope_view d = *view;
    void *mem[3 * TR_SCOPE_MAX_COLUMNS + 10];
    int nmem = 0;
    uint64_t *dsel = NULL, *drows = NULL, *dvalid = NULL, *dflags = NULL, flags = 0;
    uint32_t *dgroups = NULL;
    int64_t *dsums = NULL;
    tr_calc_fx_words *dfac = NULL;
    uint8_t *dok = NULL;
#define TR_CALC_DEV(ptr, bytes) do { if (!rc) { rc = tr_cuda_alloc((void **)&(ptr), (size_t)(bytes)); if (!rc) mem[nmem++] = (void *)(ptr); } } while (0)
    TR_CALC_DEV(dsel, ((n + 63) / 64) * 8);
    if (!rc) rc = tr_cuda_copy_in(dsel, view->selection, (size_t)(((n + 63) / 64) * 8));
    d.selection = dsel;
    if (view->groups) {
        TR_CALC_DEV(dgroups, n * 4);
        if (!rc) rc = tr_cuda_copy_in(dgroups, view->groups, (size_t)n * 4);
        d.groups = dgroups;
    }
    uint32_t used = 0;
    for (int32_t i = 0; i < pl->nfactors; i++) used |= (uint32_t)1 << pl->column[i];
    if (pl->weight >= 0) used |= (uint32_t)1 << pl->weight;
    for (int32_t c = 0; c < view->ncols; c++) {
        tr_scope_view_column *k = &d.col[c];
        if (!((used >> c) & 1)) { k->codes = NULL; k->values = NULL; k->valid = NULL; continue; }   /* never read */
        if (view->col[c].values) {
            int64_t *p = NULL;
            TR_CALC_DEV(p, n * 8);
            if (!rc) rc = tr_cuda_copy_in(p, view->col[c].values, (size_t)n * 8);
            k->values = p;
        } else {
            uint32_t *p = NULL;
            TR_CALC_DEV(p, n * 4);
            if (!rc) rc = tr_cuda_copy_in(p, view->col[c].codes, (size_t)n * 4);
            k->codes = p;
        }
        if (view->col[c].valid) {
            uint8_t *p = NULL;
            TR_CALC_DEV(p, n);
            if (!rc) rc = tr_cuda_copy_in(p, view->col[c].valid, (size_t)n);
            k->valid = p;
        }
    }
    TR_CALC_DEV(dsums, cells * 32);
    TR_CALC_DEV(drows, G * 8);
    TR_CALC_DEV(dvalid, G * 8);
    TR_CALC_DEV(dflags, 8);
    if (pl->family == TR_CALC_PHASE) {
        TR_CALC_DEV(dfac, n * (uint64_t)pl->nfactors * 2 * sizeof(tr_calc_fx_words));
        TR_CALC_DEV(dok, n);
    }
#undef TR_CALC_DEV
    tr_calc_plan_view p = *pl;
    if (!rc) {
        uint64_t slices = tr_calc_slices(n, 1);
        TR_LAUNCH(tr_calc_count_kernel, slices, p, d, slices, drows, dvalid);
        rc = tr_cuda_sync();
    }
    if (!rc && pl->family == TR_CALC_PHASE) {
        TR_LAUNCH(tr_calc_factor_kernel, n, p, d, dfac, dok);
        rc = tr_cuda_sync();
        if (!rc) {
            uint64_t slices = tr_calc_slices(n, m);
            TR_LAUNCH(tr_calc_phase_kernel, m * slices, p, d, dfac, dok, slices, dsums, dflags);
            rc = tr_cuda_sync();
        }
    } else if (!rc) {
        uint64_t slices = tr_calc_slices(n, 1);
        TR_LAUNCH(tr_calc_exp_kernel, slices, p, d, slices, dsums, dflags);
        rc = tr_cuda_sync();
    }
    if (!rc) rc = tr_cuda_copy_out(&flags, dflags, 8);
    if (!rc && !flags) rc = tr_cuda_copy_out(sums, dsums, (size_t)cells * 32);
    if (!rc && !flags) rc = tr_cuda_copy_out(rows, drows, (size_t)G * 8);
    if (!rc && !flags) rc = tr_cuda_copy_out(valid, dvalid, (size_t)G * 8);
    for (int i = 0; i < nmem; i++) tr_cuda_free(mem[i]);
    if (rc) return rc;
    return flags ? (int)TR_MATH_OVERFLOW : (int)TR_MATH_OK;
}
#ifdef __cplusplus
}
#endif
