/* FND-004 on a CUDA device: the grouped exact moment bank of a scope and its rollup, with exactly the states of
 * tr_bank_build / tr_bank_rollup (math_bank.h).
 * SPDX-License-Identifier: Apache-2.0
 *
 * Work is split into work items (term j, slice s): slice s of S takes rows s, s + S, s + 2S, ... (consecutive threads
 * read consecutive rows). A work item accumulates its rows locally (a count and an exact 128-bit sum) while the group
 * id stays the same and adds them to the group's state with atomics when it changes and at its end: one group (or
 * rows clustered by group) costs a few atomics per work item, scattered groups one per row. Integer atomics commute,
 * so the states do not depend on thread order. Per-thread work is bounded: one term over ceil(rows / S) rows.
 * The bag multiplicity has its own kernel (one work item per slice); TR_BANK_JOINT first marks the jointly valid
 * selected rows (one flag per row, one writer each).
 *
 * Exact 128-bit sums: a product and a local sum may exceed int64, so they are added with tr_bank_atomic_add_wide, the
 * two-word accumulator of tr_atomic_add_i128 (math_cuda.h) generalized to a 128-bit addend (a local equivalent: lo +=
 * low word, hi += high word + the carry of that add). The final words are the exact sum modulo 2^128, which is the
 * exact sum because the host proved |sum| < 2^127 (tr_bank_prove) before launching. */
#include "math_cuda.h"
#include "math_bank.h"
#include "math_handles.h"

TR_MGPU_ABI(tr_scope_view, 1);          /* structs passed to the CUDA library (math_handles.h) */
TR_MGPU_ABI(tr_scope_view_column, 1);
TR_MGPU_ABI(tr_bank_states, 1);

#define TR_BANK_WORK (UINT64_C(1) << 19)      /* about this many work items per launch */
#define TR_BANK_MIN_ROWS 32                    /* and at least this many rows per work item */

TR_DEVICE void tr_bank_atomic_add_wide(int64_t *lo, int64_t *hi, __int128 x) {
    uint64_t xl = (uint64_t)x, xh = (uint64_t)((unsigned __int128)x >> 64);
    uint64_t old = tr_atomic_add_u64((uint64_t *)lo, xl);
    uint64_t carry = (old + xl) < old;
    if (xh + carry) tr_atomic_add_u64((uint64_t *)hi, xh + carry);
}

TR_KERNEL void tr_bank_rows_kernel(tr_scope_view v, uint64_t slices, uint64_t *rows) {
    TR_GRID_LOOP(s, slices) {
        uint32_t cur = 0;
        uint64_t c = 0;
        for (uint64_t r = s; r < v.rows; r += slices) {
            if (!tr_scope_is_selected(&v, r)) continue;
            uint32_t g = tr_scope_group(&v, r);
            if (c && g != cur) { tr_atomic_add_u64(&rows[cur], c); c = 0; }
            cur = g; c++;
        }
        if (c) tr_atomic_add_u64(&rows[cur], c);
    }
}

TR_KERNEL void tr_bank_joint_kernel(tr_scope_view v, uint32_t joint, uint8_t *ok) {
    TR_GRID_LOOP(r, v.rows) ok[r] = (uint8_t)(tr_scope_is_selected(&v, r) && tr_bank_joint_valid(&v, joint, r));
}

TR_KERNEL void tr_bank_terms_kernel(tr_scope_view v, const tr_bank_term *terms, int32_t nterms, const uint8_t *joint_ok,
                                    uint64_t slices, uint64_t *count, int64_t *sum) {
    TR_GRID_LOOP(i, (uint64_t)nterms * slices) {
        uint64_t j = i / slices, s = i % slices;
        tr_bank_term t = terms[j];
        uint32_t cur = 0;
        uint64_t c = 0;
        __int128 acc = 0;
        for (uint64_t r = s; r < v.rows; r += slices) {
            if (joint_ok ? !joint_ok[r] : (!tr_scope_is_selected(&v, r) || !tr_bank_term_valid(&v, &t, r))) continue;
            uint32_t g = tr_scope_group(&v, r);
            if (c && g != cur) {
                uint64_t k = (uint64_t)cur * (uint64_t)nterms + j;
                tr_atomic_add_u64(&count[k], c);
                if (acc) tr_bank_atomic_add_wide(&sum[2 * k], &sum[2 * k + 1], acc);
                c = 0; acc = 0;
            }
            cur = g; c++;
            acc += tr_bank_term_value(&v, &t, r);
        }
        if (c) {
            uint64_t k = (uint64_t)cur * (uint64_t)nterms + j;
            tr_atomic_add_u64(&count[k], c);
            if (acc) tr_bank_atomic_add_wide(&sum[2 * k], &sum[2 * k + 1], acc);
        }
    }
}

TR_KERNEL void tr_bank_rollup_kernel(const uint64_t *frows, const uint64_t *fcount, const int64_t *fsum, uint64_t ngroups,
                                     int32_t nterms, const uint32_t *map, uint64_t *rows, uint64_t *count, int64_t *sum) {
    uint64_t width = (uint64_t)nterms + 1;
    TR_GRID_LOOP(i, ngroups * width) {
        uint64_t g = i / width, j = i % width;
        if (!frows[g]) continue;                                 /* an absent fine group holds nothing */
        uint64_t q = map[g];
        if (j == (uint64_t)nterms) { tr_atomic_add_u64(&rows[q], frows[g]); continue; }
        uint64_t f = g * (uint64_t)nterms + j, k = q * (uint64_t)nterms + j;
        if (fcount[f]) tr_atomic_add_u64(&count[k], fcount[f]);
        tr_math_i128 w;
        w.lo = fsum[2 * f]; w.hi = fsum[2 * f + 1];
        __int128 x = tr_i128_get(w);
        if (x) tr_bank_atomic_add_wide(&sum[2 * k], &sum[2 * k + 1], x);
    }
}

/* S slices for `work` work items per slice: about TR_BANK_WORK work items, at least TR_BANK_MIN_ROWS rows each */
static uint64_t tr_bank_slices(uint64_t rows, uint64_t work) {
    uint64_t s = TR_BANK_WORK / (work ? work : 1), cap = (rows + TR_BANK_MIN_ROWS - 1) / TR_BANK_MIN_ROWS;
    if (s > cap) s = cap;
    return s ? s : 1;
}

#ifdef __cplusplus
extern "C" {
#endif
int tr_mgpu_bank_build(const tr_scope_view *view, const uint8_t *exps, int32_t nterms, int32_t mode,
                       const tr_math_budget *budget, tr_bank_states *out, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!view || !out || (mode != TR_BANK_SQL && mode != TR_BANK_JOINT) || view->ngroups < 1 ||
        out->ngroups != view->ngroups || out->nterms != nterms || !out->rows || (nterms > 0 && (!out->count || !out->sum)) ||
        (view->rows && !view->selection) || (view->rows && view->ngroups > 1 && !view->groups) ||
        view->ncols < 0 || view->ncols > TR_SCOPE_MAX_COLUMNS) return -1;                     /* TR_MATH_INVALID */
    if (nterms > TR_BANK_MAX_TERMS) return TR_MATH_UNSUPPORTED;
    tr_bank_term *terms = (tr_bank_term *)calloc((size_t)(nterms > 0 ? nterms : 1), sizeof(tr_bank_term));
    if (!terms) return -2;                                                                      /* TR_MATH_MEMORY */
    uint32_t joint = 0;
    int32_t degree = 0;
    int rc = tr_bank_compile(exps, nterms, view->ncols, terms, &joint, &degree);
    tr_math_need need;
    tr_math_budget dflt;
    if (!budget) { tr_budget_default(&dflt); budget = &dflt; }
    if (!rc) { tr_bank_device_need(view, nterms, mode, joint, degree, &need); rc = tr_budget_check(budget, &need, refusal); }
    if (!rc) rc = tr_bank_prove(view, terms, nterms, mode, joint);
    uint64_t n = view->rows, G = view->ngroups, gm = G * (uint64_t)(nterms > 0 ? nterms : 0);
    if (rc || !view->selected) {                       /* nothing selected: every state is zero, nothing to launch */
        if (!rc) {
            memset(out->rows, 0, (size_t)G * 8);
            if (gm) { memset(out->count, 0, (size_t)gm * 8); memset(out->sum, 0, (size_t)gm * 16); }
        }
        free(terms);
        return rc;
    }
    tr_scope_view d = *view;
    void *dmem[3 * TR_SCOPE_MAX_COLUMNS + 8];
    int nmem = 0;
    uint64_t *dsel = NULL, *drows = NULL, *dcount = NULL;
    uint32_t *dgroups = NULL;
    int64_t *dsum = NULL;
    uint8_t *djoint = NULL;
    tr_bank_term *dterms = NULL;
#define TR_BANK_DEV(ptr, bytes) do { if (!rc) { rc = tr_cuda_alloc((void **)&(ptr), (size_t)(bytes)); if (!rc) dmem[nmem++] = (void *)(ptr); } } while (0)
    TR_BANK_DEV(dsel, ((n + 63) / 64) * 8);
    if (!rc) rc = tr_cuda_copy_in(dsel, view->selection, (size_t)(((n + 63) / 64) * 8));
    d.selection = dsel;
    if (view->groups) {
        TR_BANK_DEV(dgroups, n * 4);
        if (!rc) rc = tr_cuda_copy_in(dgroups, view->groups, (size_t)n * 4);
        d.groups = dgroups;
    }
    for (int32_t c = 0; c < view->ncols; c++) {
        tr_scope_view_column *k = &d.col[c];
        if (!((joint >> c) & 1)) { k->codes = NULL; k->values = NULL; k->valid = NULL; continue; }   /* never read */
        if (view->col[c].values) {
            int64_t *p = NULL;
            TR_BANK_DEV(p, n * 8);
            if (!rc) rc = tr_cuda_copy_in(p, view->col[c].values, (size_t)n * 8);
            k->values = p;
        } else {
            uint32_t *p = NULL;
            TR_BANK_DEV(p, n * 4);
            if (!rc) rc = tr_cuda_copy_in(p, view->col[c].codes, (size_t)n * 4);
            k->codes = p;
        }
        if (view->col[c].valid) {
            uint8_t *p = NULL;
            TR_BANK_DEV(p, n);
            if (!rc) rc = tr_cuda_copy_in(p, view->col[c].valid, (size_t)n);
            k->valid = p;
        }
    }
    if (mode == TR_BANK_JOINT) TR_BANK_DEV(djoint, n);
    if (nterms > 0) {
        TR_BANK_DEV(dterms, (uint64_t)nterms * sizeof(tr_bank_term));
        if (!rc) rc = tr_cuda_copy_in(dterms, terms, (size_t)nterms * sizeof(tr_bank_term));
        TR_BANK_DEV(dcount, gm * 8);
        TR_BANK_DEV(dsum, gm * 16);
    }
    TR_BANK_DEV(drows, G * 8);
#undef TR_BANK_DEV
    if (!rc) {
        uint64_t s = tr_bank_slices(n, 1);
        TR_LAUNCH(tr_bank_rows_kernel, s, d, s, drows);
        if (djoint) TR_LAUNCH(tr_bank_joint_kernel, n, d, joint, djoint);
        if (nterms > 0) {
            uint64_t t = tr_bank_slices(n, (uint64_t)nterms);
            TR_LAUNCH(tr_bank_terms_kernel, (uint64_t)nterms * t, d, dterms, nterms, djoint, t, dcount, dsum);
        }
        rc = tr_cuda_sync();
    }
    if (!rc) rc = tr_cuda_copy_out(out->rows, drows, (size_t)G * 8);
    if (!rc && nterms > 0) rc = tr_cuda_copy_out(out->count, dcount, (size_t)gm * 8);
    if (!rc && nterms > 0) rc = tr_cuda_copy_out(out->sum, dsum, (size_t)gm * 16);
    for (int i = 0; i < nmem; i++) tr_cuda_free(dmem[i]);
    free(terms);
    return rc;
}

int tr_mgpu_bank_rollup(const tr_bank_states *fine, const uint32_t *map, uint64_t coarse_groups,
                        const tr_math_budget *budget, tr_bank_states *out, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!fine || !map || !out || fine->ngroups < 1 || fine->nterms < 0 || fine->nterms > TR_BANK_MAX_TERMS ||
        !fine->rows || (fine->nterms && (!fine->count || !fine->sum)) || coarse_groups < 1 ||
        coarse_groups > TR_SCOPE_MAX_GROUPS || out->ngroups != coarse_groups || out->nterms != fine->nterms ||
        !out->rows || (out->nterms && (!out->count || !out->sum))) return -1;                   /* TR_MATH_INVALID */
    tr_math_need need;
    tr_math_budget dflt;
    if (!budget) { tr_budget_default(&dflt); budget = &dflt; }
    const uint64_t G = fine->ngroups, m = (uint64_t)fine->nterms, fgm = G * m, cgm = coarse_groups * m;
    tr_bank_rollup_device_need(G, fine->nterms, coarse_groups, &need);
    int rc = tr_budget_check(budget, &need, refusal);
    if (rc) return rc;
    for (uint64_t g = 0; g < G; g++) if (map[g] >= coarse_groups) return -1;
    if (m) {                                           /* the same per-cell bound as the CPU rollup */
        unsigned __int128 *scratch = (unsigned __int128 *)calloc((size_t)coarse_groups, 16);
        if (!scratch) return -2;                                                                /* TR_MATH_MEMORY */
        rc = tr_bank_rollup_bound(fine, map, coarse_groups, scratch);
        free(scratch);
        if (rc) return rc;                                                                      /* TR_MATH_OVERFLOW */
    }
    memset(out->rows, 0, (size_t)coarse_groups * 8);
    if (cgm) { memset(out->count, 0, (size_t)cgm * 8); memset(out->sum, 0, (size_t)cgm * 16); }
    uint64_t *dfrows = NULL, *dfcount = NULL, *drows = NULL, *dcount = NULL;
    int64_t *dfsum = NULL, *dsum = NULL;
    uint32_t *dmap = NULL;
    rc = tr_cuda_alloc((void **)&dfrows, (size_t)G * 8);
    if (!rc) rc = tr_cuda_alloc((void **)&dmap, (size_t)G * 4);
    if (!rc) rc = tr_cuda_alloc((void **)&drows, (size_t)coarse_groups * 8);
    if (!rc && m) rc = tr_cuda_alloc((void **)&dfcount, (size_t)fgm * 8);
    if (!rc && m) rc = tr_cuda_alloc((void **)&dfsum, (size_t)fgm * 16);
    if (!rc && m) rc = tr_cuda_alloc((void **)&dcount, (size_t)cgm * 8);
    if (!rc && m) rc = tr_cuda_alloc((void **)&dsum, (size_t)cgm * 16);
    if (!rc) rc = tr_cuda_copy_in(dfrows, fine->rows, (size_t)G * 8);
    if (!rc) rc = tr_cuda_copy_in(dmap, map, (size_t)G * 4);
    if (!rc && m) rc = tr_cuda_copy_in(dfcount, fine->count, (size_t)fgm * 8);
    if (!rc && m) rc = tr_cuda_copy_in(dfsum, fine->sum, (size_t)fgm * 16);
    if (!rc) {
        TR_LAUNCH(tr_bank_rollup_kernel, G * (m + 1), dfrows, dfcount, dfsum, G, fine->nterms, dmap, drows, dcount, dsum);
        rc = tr_cuda_sync();
    }
    if (!rc) rc = tr_cuda_copy_out(out->rows, drows, (size_t)coarse_groups * 8);
    if (!rc && m) rc = tr_cuda_copy_out(out->count, dcount, (size_t)cgm * 8);
    if (!rc && m) rc = tr_cuda_copy_out(out->sum, dsum, (size_t)cgm * 16);
    tr_cuda_free(dfrows); tr_cuda_free(dmap); tr_cuda_free(drows); tr_cuda_free(dfcount); tr_cuda_free(dfsum);
    tr_cuda_free(dcount); tr_cuda_free(dsum);
    return rc;
}
#ifdef __cplusplus
}
#endif
