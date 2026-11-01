/* PROB-001 on a CUDA device: the certified sums of many parameter requests over a declared support, with exactly the
 * integers of tr_prob_sums (math_prob.c): the same inline per-point terms of math_prob.h on the CPU and here.
 * SPDX-License-Identifier: Apache-2.0
 *
 * One work item per (request r, TR_PROB_CHUNK consecutive support points):
 *   EXPFAM  launch 1: n_i + bitlen(H_i) of each point with H_i > 0 (the integer part of CALC-003's reduction,
 *           tr_prob_exp_ref), a local maximum, then one atomic maximum per request (n_ref); an exponent past CALC-003's
 *           bound sets TR_PROB_EXP_OVERFLOW by atomic OR (the point is skipped, as on the CPU).
 *           launch 2: each point's exponential (CALC-003's tr_calc_exp2_reduce, 120-bit midpoint-radius), the
 *           outward integers floor / ceil of every wanted sum term at its proven scale (tr_prob_exp_point), local
 *           128-bit sums, then one exact two-word atomic addition per sum bound (tr_atomic_add_wide)
 *   CATEGORICAL  one launch: the exact sums W, W N_j, W N_j N_l and the certified bounds of n ln W (DIST-003's
 *           logarithm, tr_prob_cat_point), local sums, then exact two-word atomics; a zero weight with mass sets
 *           TR_PROB_NEG_INF by atomic OR
 * Every term is an exact integer and every partial sum is proven below 2^126 in magnitude in any order (the scales of
 * math_prob.h), so the 128-bit totals modulo 2^128 are the exact totals: the device sums, the reference exponents and
 * the flags equal the CPU's bit for bit whatever the order of the threads. No barrier, no shared memory. The host
 * checks the view and budgets the device copies (tr_prob_device_bytes, resource bytes; states = n x sums) before
 * allocating; tr_prob_finish (CPU) checks the readback against the proofs and finishes it exactly as it finishes
 * the CPU sums. */
#include "math_cuda.h"
#include "math_prob.h"
#include "math_handles.h"

TR_MGPU_ABI(tr_prob_plan_view, TR_PROB_ABI);   /* structs passed to the CUDA library (math_handles.h) */

/* the support points [b, e) of work item w (request *r) */
TR_DEVICE void tr_prob_item(const tr_prob_plan_view *v, uint64_t chunks, uint64_t w, uint64_t *r, uint64_t *b,
                            uint64_t *e) {
    *r = w / chunks;
    *b = (w % chunks) * TR_PROB_CHUNK;
    *e = v->d - *b < TR_PROB_CHUNK ? v->d : *b + TR_PROB_CHUNK;
}

TR_KERNEL void tr_prob_ref_kernel(tr_prob_plan_view v, uint64_t chunks, int64_t *nref, uint64_t *flags) {
    TR_GRID_LOOP(w, v.n * chunks) {
        uint64_t r, b, e;
        tr_prob_item(&v, chunks, w, &r, &b, &e);
        if (v.status[r]) continue;
        int64_t best = INT64_MIN;
        int over = 0;
        for (uint64_t i = b; i < e; i++) {
            if (!v.base[i]) continue;
            int64_t x;
            if (!tr_prob_exp_ref(&v, r, i, &x)) { over = 1; continue; }
            if (x > best) best = x;
        }
        if (best != INT64_MIN) tr_atomic_max_i64(&nref[r], best);
        if (over) tr_atomic_or_u64(&flags[r], (uint64_t)TR_PROB_EXP_OVERFLOW);
    }
}

TR_DEVICE void tr_prob_flush(tr_math_i128 *sums, uint64_t base, const __int128 *acc, uint64_t n) {
    for (uint64_t q = 0; q < n; q++)
        if (acc[q]) tr_atomic_add_wide(&sums[base + q].lo, &sums[base + q].hi, acc[q]);
}

TR_KERNEL void tr_prob_exp_kernel(tr_prob_plan_view v, uint64_t chunks, const int64_t *nref, uint64_t *flags,
                                  tr_math_i128 *sums) {
    TR_GRID_LOOP(w, v.n * chunks) {
        uint64_t r, b, e;
        tr_prob_item(&v, chunks, w, &r, &b, &e);
        if (v.status[r] || (flags[r] & TR_PROB_EXP_OVERFLOW)) continue;   /* flags of launch 1: complete */
        const int64_t ref = nref[r];
        __int128 acc[2 * TR_PROB_MAX_SUMS];
        for (uint64_t q = 0; q < v.nsums; q++) acc[q] = 0;
        uint64_t bad = 0;
        for (uint64_t i = b; i < e; i++) {
            if (!v.base[i]) continue;
            int ok = tr_prob_exp_point(&v, r, i, ref, acc);
            if (ok < 0) bad |= TR_PROB_EXP_OVERFLOW;
            else if (!ok) bad |= TR_PROB_INCONSISTENT;
        }
        tr_prob_flush(sums, r * v.nsums, acc, v.nsums);
        if (bad) tr_atomic_or_u64(&flags[r], bad);
    }
}

TR_KERNEL void tr_prob_cat_kernel(tr_prob_plan_view v, uint64_t chunks, uint64_t *flags, tr_math_i128 *sums) {
    TR_GRID_LOOP(w, v.n * chunks) {
        uint64_t r, b, e;
        tr_prob_item(&v, chunks, w, &r, &b, &e);
        if (v.status[r]) continue;
        const uint64_t nq = tr_prob_nq(&v);
        __int128 acc[TR_PROB_MAX_SUMS];
        for (uint64_t q = 0; q < v.nsums; q++) acc[q] = 0;
        int neg = 0;
        for (uint64_t i = b; i < e; i++)
            if (!tr_prob_cat_point(&v, r, i, acc, acc + nq)) neg = 1;
        tr_prob_flush(sums, r * v.nsums, acc, v.nsums);
        if (neg) tr_atomic_or_u64(&flags[r], (uint64_t)TR_PROB_NEG_INF);
    }
}

#ifdef __cplusplus
extern "C" {
#endif
int tr_mgpu_prob_sums(const tr_prob_plan_view *view, const tr_math_budget *budget, tr_math_i128 *sums, int64_t *nref,
                      uint64_t *flags, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!tr_prob_plan_view_valid(view) || (view->n && (!sums || !nref || !flags))) return -1;   /* TR_MATH_INVALID */
    const tr_prob_plan_view *v = view;
    for (uint64_t r = 0; r < v->n; r++) if (!v->status[r] && v->group[r] >= v->ngroups) return -1;
    tr_math_need need;
    for (int i = 0; i < TR_BUDGET_RESOURCES; i++) need.need[i] = 0;
    need.need[TR_BUDGET_STATES] = tr_budget_mul(v->n, v->nsums);
    need.need[TR_BUDGET_BYTES] = tr_prob_device_bytes(v);
    tr_math_budget dflt;
    if (!budget) { tr_budget_default(&dflt); budget = &dflt; }
    int rc = tr_budget_check(budget, &need, refusal);
    for (int i = 0; !rc && i < TR_BUDGET_RESOURCES; i++)
        if (need.need[i] == TR_BUDGET_UNKNOWN) {
            if (refusal) { refusal->resource = i; refusal->refused = 1; refusal->need = need.need[i];
                           refusal->limit = budget->limit[i]; }
            rc = TR_MATH_BUDGET_CODE;
        }
    if (rc) return rc;
    if (!v->n) return 0;
    const uint64_t n = v->n, d = v->d, m = (uint64_t)v->m, ns = v->nsums;
    tr_prob_plan_view dv = *v;
    void *mem[12];
    int nmem = 0;
#define TR_PROB_DEV(field, type, src, count) do { if (!rc) { type *p_ = NULL;                                     \
        rc = tr_cuda_alloc((void **)&p_, (size_t)(count) * sizeof(type));                                          \
        if (!rc) { mem[nmem++] = (void *)p_; rc = tr_cuda_copy_in(p_, (src), (size_t)(count) * sizeof(type)); }   \
        dv.field = p_; } } while (0)
    if (m) TR_PROB_DEV(feat, int64_t, v->feat, d * m);
    TR_PROB_DEV(group, uint64_t, v->group, n);
    TR_PROB_DEV(status, int32_t, v->status, n);
    if (v->kind == TR_PROB_EXPFAM) {
        TR_PROB_DEV(base, int64_t, v->base, d);
        TR_PROB_DEV(hbits, int32_t, v->hbits, d);
        TR_PROB_DEV(exp, tr_calc_plan_view, v->exp, n);
    } else {
        TR_PROB_DEV(counts, uint64_t, v->counts, v->ngroups * d);
        TR_PROB_DEV(weight, int64_t, v->weight, n * d);
        TR_PROB_DEV(lscale, int32_t, v->lscale, n);
    }
#undef TR_PROB_DEV
    tr_math_i128 *dsums = NULL;
    int64_t *dnref = NULL;
    uint64_t *dflags = NULL;
    if (!rc) { rc = tr_cuda_alloc((void **)&dsums, (size_t)(n * ns) * sizeof(tr_math_i128)); if (!rc) mem[nmem++] = dsums; }
    if (!rc) { rc = tr_cuda_alloc((void **)&dflags, (size_t)n * 8); if (!rc) mem[nmem++] = dflags; }
    if (!rc) { rc = tr_cuda_alloc((void **)&dnref, (size_t)n * 8); if (!rc) mem[nmem++] = dnref; }
    if (!rc) {                                                          /* every reference starts at INT64_MIN */
        for (uint64_t r = 0; r < n; r++) nref[r] = INT64_MIN;
        rc = tr_cuda_copy_in(dnref, nref, (size_t)n * 8);
    }
    if (!rc) {
        const uint64_t chunks = (d + TR_PROB_CHUNK - 1) / TR_PROB_CHUNK;
        if (v->kind == TR_PROB_EXPFAM) {
            TR_LAUNCH(tr_prob_ref_kernel, n * chunks, dv, chunks, dnref, dflags);
            rc = tr_cuda_sync();
            if (!rc) TR_LAUNCH(tr_prob_exp_kernel, n * chunks, dv, chunks, (const int64_t *)dnref, dflags, dsums);
        } else {
            TR_LAUNCH(tr_prob_cat_kernel, n * chunks, dv, chunks, dflags, dsums);
        }
        if (!rc) rc = tr_cuda_sync();
    }
    if (!rc) rc = tr_cuda_copy_out(sums, dsums, (size_t)(n * ns) * sizeof(tr_math_i128));
    if (!rc) rc = tr_cuda_copy_out(nref, dnref, (size_t)n * 8);
    if (!rc) rc = tr_cuda_copy_out(flags, dflags, (size_t)n * 8);
    for (int i = 0; i < nmem; i++) tr_cuda_free(mem[i]);
    return rc;
}
#ifdef __cplusplus
}
#endif
