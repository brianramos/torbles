/* LIN-001 / LIN-002 on a CUDA device: the fused grouped Gram scan of a scope (exactly the states of
 * tr_lin_moments_build) and batched Gauss-Jordan elimination modulo primes (exactly tr_lin_batch_residues).
 * SPDX-License-Identifier: Apache-2.0
 *
 * Gram scan. Three kernels: (1) one work item per row writes the row's mask of present columns (bit 0: selected);
 * (2) the bag multiplicity, one work item per slice of rows, flushed with atomics when the group changes; (3) the
 * slots: one work item per (slot, slice), slice s taking rows s, s + S, s + 2S, ... (consecutive threads read
 * consecutive rows). With at most TR_LIN_LOCAL_GROUPS groups a work item keeps a local count and sum for every group
 * and flushes them once at its end (no atomics per row even for scattered groups); with more groups it accumulates
 * while the group stays the same and flushes when it changes. Sums are accumulated exactly in 128 bits when the host
 * proved every slot fits (tr_lin_needs_wide), else in 192 bits, and flushed with multiword integer atomics (each word
 * added with its carry into the next): integer additions commute, so the states do not depend on thread order.
 *
 * Modular elimination. One work item per (system, prime): it reduces the system's entries modulo the prime into its
 * own output block and runs tr_lin_modular_rref there (one output slot per work item). The host part of LIN-002
 * (Chinese remaindering, reconstruction, exact verification) is tr_lin_batch_finish in the engine library. */
#include "math_cuda.h"
#include "math_lin.h"
#include "math_handles.h"

TR_MGPU_ABI(tr_lin_spec, 1);             /* structs passed to the CUDA library (math_handles.h) */
TR_MGPU_ABI(tr_lin_states, 1);
TR_MGPU_ABI(tr_lin_batch_view, 1);

#define TR_LIN_WORK (UINT64_C(1) << 19)        /* about this many work items per launch */
#define TR_LIN_MIN_ROWS 32                      /* and at least this many rows per work item */

typedef struct {
    int32_t k, nslots, words, listwise;
    uint64_t ngroups;
    uint32_t full, reserved;
    int32_t col[TR_LIN_MAX_COLS];
} tr_lin_kargs;                                 /* a kernel argument (never crosses ctypes) */

/* w += x (mod 2^128, two words): exact when the final sum fits, which the host proved */
TR_DEVICE void tr_lin_atomic_add2(uint64_t *w, __int128 x) {
    uint64_t xl = (uint64_t)x, xh = (uint64_t)((unsigned __int128)x >> 64);
    uint64_t old = tr_atomic_add_u64(&w[0], xl);
    uint64_t carry = (old + xl) < old;
    if (xh + carry) tr_atomic_add_u64(&w[1], xh + carry);
}
/* w += x (mod 2^192, three words): every word is added with the carry out of the word below */
TR_DEVICE void tr_lin_atomic_add3(uint64_t *w, const uint64_t *x) {
    uint64_t old0 = tr_atomic_add_u64(&w[0], x[0]);
    uint64_t c0 = (old0 + x[0]) < old0;
    uint64_t y1 = x[1] + c0, c1a = y1 < c0;
    uint64_t old1 = y1 ? tr_atomic_add_u64(&w[1], y1) : 0;
    uint64_t c1 = y1 ? (old1 + y1) < old1 : 0;
    uint64_t y2 = x[2] + c1 + c1a;
    if (y2) tr_atomic_add_u64(&w[2], y2);
}
TR_DEVICE void tr_lin_flush(uint64_t *count, uint64_t *sum, uint64_t idx, int32_t words, int32_t listwise, uint64_t c,
                            __int128 a2, const uint64_t *a3) {
    if (!listwise) tr_atomic_add_u64(&count[idx], c);
    if (words == 2) { if (a2) tr_lin_atomic_add2(sum + 2 * idx, a2); }
    else if (a3[0] | a3[1] | a3[2]) tr_lin_atomic_add3(sum + 3 * idx, a3);
}

TR_KERNEL void tr_lin_mask_kernel(tr_scope_view v, tr_lin_kargs ka, uint32_t *mask) {
    TR_GRID_LOOP(r, v.rows) mask[r] = tr_lin_row_mask(&v, ka.col, ka.k, r);
}

TR_KERNEL void tr_lin_rows_kernel(tr_scope_view v, const uint32_t *mask, uint64_t slices, uint64_t *rows) {
    TR_GRID_LOOP(s, slices) {
        uint32_t cur = 0;
        uint64_t c = 0;
        for (uint64_t r = s; r < v.rows; r += slices) {
            if (!mask[r]) continue;
            uint32_t g = tr_scope_group(&v, r);
            if (c && g != cur) { tr_atomic_add_u64(&rows[cur], c); c = 0; }
            cur = g; c++;
        }
        if (c) tr_atomic_add_u64(&rows[cur], c);
    }
}

TR_KERNEL void tr_lin_slots_kernel(tr_scope_view v, tr_lin_kargs ka, const tr_lin_slot *slots, const uint32_t *mask,
                                   uint64_t slices, uint64_t *count, uint64_t *sum) {
    TR_GRID_LOOP(i, (uint64_t)ka.nslots * slices) {
        uint64_t j = i / slices, s = i % slices;
        tr_lin_slot sl = slots[j];
        uint32_t need = ka.listwise ? ka.full : (sl.need | 1u);
        const uint64_t S = (uint64_t)ka.nslots;
        if (ka.ngroups <= TR_LIN_LOCAL_GROUPS) {
            uint64_t cnt[TR_LIN_LOCAL_GROUPS], a3[3 * TR_LIN_LOCAL_GROUPS];
            __int128 a2[TR_LIN_LOCAL_GROUPS];
            for (uint64_t g = 0; g < ka.ngroups; g++) { cnt[g] = 0; a2[g] = 0; a3[3 * g] = a3[3 * g + 1] = a3[3 * g + 2] = 0; }
            for (uint64_t r = s; r < v.rows; r += slices) {
                if ((mask[r] & need) != need) continue;
                uint32_t g = tr_scope_group(&v, r);
                __int128 p = tr_lin_product(&v, ka.col, &sl, r);
                cnt[g]++;
                if (ka.words == 2) a2[g] += p; else tr_lin_w3_add(a3 + 3 * g, p);
            }
            for (uint64_t g = 0; g < ka.ngroups; g++)
                if (cnt[g]) tr_lin_flush(count, sum, g * S + j, ka.words, ka.listwise, cnt[g], a2[g], a3 + 3 * g);
        } else {
            uint32_t cur = 0;
            uint64_t c = 0, a3[3] = {0, 0, 0};
            __int128 a2 = 0;
            for (uint64_t r = s; r < v.rows; r += slices) {
                if ((mask[r] & need) != need) continue;
                uint32_t g = tr_scope_group(&v, r);
                if (c && g != cur) {
                    tr_lin_flush(count, sum, (uint64_t)cur * S + j, ka.words, ka.listwise, c, a2, a3);
                    c = 0; a2 = 0; a3[0] = a3[1] = a3[2] = 0;
                }
                cur = g; c++;
                __int128 p = tr_lin_product(&v, ka.col, &sl, r);
                if (ka.words == 2) a2 += p; else tr_lin_w3_add(a3, p);
            }
            if (c) tr_lin_flush(count, sum, (uint64_t)cur * S + j, ka.words, ka.listwise, c, a2, a3);
        }
    }
}

TR_KERNEL void tr_lin_rref_kernel(tr_lin_batch_view v, const tr_gfp *primes, uint64_t nprimes, const uint64_t *roff,
                                  uint64_t *res) {
    TR_GRID_LOOP(i, v.nsys * nprimes) {
        uint64_t s = i / nprimes, q = i % nprimes;
        int32_t m = v.rows[s], c = v.cols[s];
        uint64_t mc = (uint64_t)m * (uint64_t)c;
        uint64_t *blk = res + roff[s] + q * (4 + mc);
        tr_gfp f = primes[q];
        const uint64_t *src = v.data + v.offset[s] * v.words;
        for (uint64_t e = 0; e < mc; e++) blk[4 + e] = tr_lin_reduce_words(&f, src + e * v.words, v.words);
        tr_lin_modular_rref(&f, blk + 4, m, c, blk);
    }
}

/* S slices for `work` work items per slice: about TR_LIN_WORK work items, at least TR_LIN_MIN_ROWS rows each */
static uint64_t tr_lin_slices(uint64_t rows, uint64_t work) {
    uint64_t s = TR_LIN_WORK / (work ? work : 1), cap = (rows + TR_LIN_MIN_ROWS - 1) / TR_LIN_MIN_ROWS;
    if (s > cap) s = cap;
    return s ? s : 1;
}

#ifdef __cplusplus
extern "C" {
#endif
int tr_mgpu_lin_moments(const tr_scope_view *view, const tr_lin_spec *spec, int32_t marginals, const tr_math_budget *budget,
                        tr_lin_states *out, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!view || !spec || !out || view->ngroups < 1 || view->ncols < 0 || view->ncols > TR_SCOPE_MAX_COLUMNS ||
        (view->rows && !view->selection) || (view->rows && view->ngroups > 1 && !view->groups)) return -1;
    int32_t col[TR_LIN_MAX_COLS], k = tr_lin_columns(spec, view->ncols, col);
    if (k < 0) return k;
    marginals = spec->policy == TR_LIN_PAIRWISE && marginals;
    const int32_t S = tr_lin_slot_count(k, spec->policy, marginals);
    const uint64_t n = view->rows, G = view->ngroups, GS = G * (uint64_t)S;
    if (out->ngroups != G || out->nslots != S || out->words != TR_LIN_WORDS || !out->rows || !out->count || !out->sum) return -1;
    tr_lin_slot *slots = (tr_lin_slot *)calloc((size_t)S, sizeof(tr_lin_slot));
    if (!slots) return -2;                                                                       /* TR_MATH_MEMORY */
    tr_lin_layout(k, spec->policy, marginals, slots);
    int32_t words = tr_lin_words(spec->width, tr_lin_needs_wide(view, col, k, slots, S));
    tr_math_need need;
    tr_math_budget dflt;
    if (!budget) { tr_budget_default(&dflt); budget = &dflt; }
    tr_lin_device_need(view, col, k, S, words < 0 ? 3 : words, &need);
    int rc = tr_budget_check(budget, &need, refusal);
    if (!rc && words < 0) rc = words;                                                            /* TR_MATH_OVERFLOW */
    if (rc || !view->selected) {
        if (!rc) { memset(out->rows, 0, (size_t)G * 8); memset(out->count, 0, (size_t)GS * 8); memset(out->sum, 0, (size_t)GS * 24); }
        free(slots);
        return rc;
    }
    const int32_t listwise = spec->policy == TR_LIN_LISTWISE;
    tr_lin_kargs ka;
    memset(&ka, 0, sizeof ka);
    ka.k = k; ka.nslots = S; ka.words = words; ka.listwise = listwise; ka.ngroups = G; ka.full = 1;
    for (int32_t i = 0; i < TR_LIN_MAX_COLS; i++) ka.col[i] = i < k ? col[i] : -1;
    for (int32_t i = 1; i < k; i++) ka.full |= (uint32_t)1 << i;
    tr_scope_view d = *view;
    void *dmem[2 * TR_SCOPE_MAX_COLUMNS + 10];
    int nmem = 0;
    uint64_t *dsel = NULL, *drows = NULL, *dcount = NULL, *dsum = NULL;
    uint32_t *dgroups = NULL, *dmask = NULL;
    tr_lin_slot *dslots = NULL;
#define TR_LIN_DEV(ptr, bytes) do { if (!rc) { rc = tr_cuda_alloc((void **)&(ptr), (size_t)(bytes)); if (!rc) dmem[nmem++] = (void *)(ptr); } } while (0)
    TR_LIN_DEV(dsel, ((n + 63) / 64) * 8);
    if (!rc) rc = tr_cuda_copy_in(dsel, view->selection, (size_t)(((n + 63) / 64) * 8));
    d.selection = dsel;
    if (view->groups) {
        TR_LIN_DEV(dgroups, n * 4);
        if (!rc) rc = tr_cuda_copy_in(dgroups, view->groups, (size_t)n * 4);
        d.groups = dgroups;
    }
    uint64_t usedcols = 0;
    for (int32_t i = 1; i < k; i++) usedcols |= (uint64_t)1 << col[i];
    for (int32_t c = 0; c < view->ncols; c++) {
        tr_scope_view_column *kc = &d.col[c];
        if (!((usedcols >> c) & 1)) { kc->codes = NULL; kc->values = NULL; kc->valid = NULL; continue; }   /* never read */
        if (view->col[c].values) {
            int64_t *p = NULL;
            TR_LIN_DEV(p, n * 8);
            if (!rc) rc = tr_cuda_copy_in(p, view->col[c].values, (size_t)n * 8);
            kc->values = p;
        } else {
            uint32_t *p = NULL;
            TR_LIN_DEV(p, n * 4);
            if (!rc) rc = tr_cuda_copy_in(p, view->col[c].codes, (size_t)n * 4);
            kc->codes = p;
        }
        if (view->col[c].valid) {
            uint8_t *p = NULL;
            TR_LIN_DEV(p, n);
            if (!rc) rc = tr_cuda_copy_in(p, view->col[c].valid, (size_t)n);
            kc->valid = p;
        }
    }
    TR_LIN_DEV(dmask, n * 4);
    TR_LIN_DEV(dslots, (uint64_t)S * sizeof(tr_lin_slot));
    if (!rc) rc = tr_cuda_copy_in(dslots, slots, (size_t)S * sizeof(tr_lin_slot));
    TR_LIN_DEV(drows, G * 8);
    if (!listwise) TR_LIN_DEV(dcount, GS * 8);
    TR_LIN_DEV(dsum, GS * 8 * (uint64_t)words);
#undef TR_LIN_DEV
    if (!rc) {
        TR_LAUNCH(tr_lin_mask_kernel, n, d, ka, dmask);
        uint64_t s1 = tr_lin_slices(n, 1);
        TR_LAUNCH(tr_lin_rows_kernel, s1, d, dmask, s1, drows);
        uint64_t s2 = tr_lin_slices(n, (uint64_t)S);
        TR_LAUNCH(tr_lin_slots_kernel, (uint64_t)S * s2, d, ka, dslots, dmask, s2, dcount, dsum);
        rc = tr_cuda_sync();
    }
    if (!rc) rc = tr_cuda_copy_out(out->rows, drows, (size_t)G * 8);
    if (!rc && !listwise) rc = tr_cuda_copy_out(out->count, dcount, (size_t)GS * 8);
    if (!rc) rc = tr_cuda_copy_out(out->sum, dsum, (size_t)(GS * 8 * (uint64_t)words));
    for (int i = 0; i < nmem; i++) tr_cuda_free(dmem[i]);
    free(slots);
    if (rc) return rc;
    if (words == 2)                                   /* widen in place to three words, from the end */
        for (uint64_t i = GS; i-- > 0;) {
            uint64_t lo = out->sum[2 * i], hi = out->sum[2 * i + 1];
            out->sum[3 * i] = lo; out->sum[3 * i + 1] = hi; out->sum[3 * i + 2] = (hi >> 63) ? ~(uint64_t)0 : 0;
        }
    if (listwise)                                     /* every slot counts the complete rows: the constant slot's sum */
        for (uint64_t g = 0; g < G; g++)
            for (uint64_t j = 0; j < (uint64_t)S; j++) out->count[g * S + j] = out->sum[3 * (g * S)];
    return 0;
}

int tr_mgpu_lin_eliminate(const tr_lin_batch_view *view, const tr_gfp *primes, uint64_t nprimes, const tr_math_budget *budget,
                          uint64_t *residues, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!view || !primes || !residues || !view->nsys || !view->words || !view->data || !view->offset || !view->rows ||
        !view->cols || !nprimes || nprimes > 4096) return -1;
    uint64_t entries = 0;
    for (uint64_t s = 0; s < view->nsys; s++) {
        if (view->rows[s] < 1 || view->rows[s] > TR_LIN_MAX_ROWS || view->cols[s] < 2 || view->cols[s] > TR_LIN_MAX_UNKNOWNS + 1 ||
            view->offset[s] != entries) return -1;
        entries += (uint64_t)view->rows[s] * (uint64_t)view->cols[s];
    }
    if (entries != view->entries) return -1;
    for (uint64_t q = 0; q < nprimes; q++) if (!tr_gfp_consistent(&primes[q])) return -1;
    tr_math_budget dflt;
    if (!budget) { tr_budget_default(&dflt); budget = &dflt; }
    int rc = tr_budget_check_one(budget, TR_BUDGET_BYTES, tr_lin_batch_device_bytes(view, nprimes), refusal);
    if (rc) return rc;
    uint64_t *roff = (uint64_t *)malloc((size_t)view->nsys * 8), total = 0;
    if (!roff) return -2;
    for (uint64_t s = 0; s < view->nsys; s++) { roff[s] = total; total += nprimes * (4 + (uint64_t)view->rows[s] * (uint64_t)view->cols[s]); }
    tr_lin_batch_view d = *view;
    uint64_t *ddata = NULL, *doff = NULL, *droff = NULL, *dres = NULL;
    int32_t *drows = NULL, *dcols = NULL;
    tr_gfp *dprimes = NULL;
    rc = tr_cuda_alloc((void **)&ddata, (size_t)(view->entries * view->words * 8));
    if (!rc) rc = tr_cuda_alloc((void **)&doff, (size_t)view->nsys * 8);
    if (!rc) rc = tr_cuda_alloc((void **)&droff, (size_t)view->nsys * 8);
    if (!rc) rc = tr_cuda_alloc((void **)&drows, (size_t)view->nsys * 4);
    if (!rc) rc = tr_cuda_alloc((void **)&dcols, (size_t)view->nsys * 4);
    if (!rc) rc = tr_cuda_alloc((void **)&dprimes, (size_t)nprimes * sizeof(tr_gfp));
    if (!rc) rc = tr_cuda_alloc((void **)&dres, (size_t)total * 8);
    if (!rc) rc = tr_cuda_copy_in(ddata, view->data, (size_t)(view->entries * view->words * 8));
    if (!rc) rc = tr_cuda_copy_in(doff, view->offset, (size_t)view->nsys * 8);
    if (!rc) rc = tr_cuda_copy_in(droff, roff, (size_t)view->nsys * 8);
    if (!rc) rc = tr_cuda_copy_in(drows, view->rows, (size_t)view->nsys * 4);
    if (!rc) rc = tr_cuda_copy_in(dcols, view->cols, (size_t)view->nsys * 4);
    if (!rc) rc = tr_cuda_copy_in(dprimes, primes, (size_t)nprimes * sizeof(tr_gfp));
    d.data = ddata; d.offset = doff; d.rows = drows; d.cols = dcols;
    if (!rc) {
        TR_LAUNCH(tr_lin_rref_kernel, view->nsys * nprimes, d, dprimes, nprimes, droff, dres);
        rc = tr_cuda_sync();
    }
    if (!rc) rc = tr_cuda_copy_out(residues, dres, (size_t)total * 8);
    tr_cuda_free(ddata); tr_cuda_free(doff); tr_cuda_free(droff); tr_cuda_free(drows); tr_cuda_free(dcols);
    tr_cuda_free(dprimes); tr_cuda_free(dres);
    free(roff);
    return rc;
}
#ifdef __cplusplus
}
#endif
