/* KERNEL-002 on a CUDA device: the feature accumulation over the rows of a scope, with exactly the integers of
 * tr_kapprox_sums (math_kapprox.c): the same inline functions of math_kapprox.h compute every row's Gaussian prefactor
 * (CALC-003's certified fixed-point exp) and monomials, turn each weighted enclosure outward into integers and add
 * them exactly in 128 bits, so the device sums equal the CPU's bit for bit whatever the order of the threads.
 * SPDX-License-Identifier: Apache-2.0
 *
 *   counts     work item = a slice of rows (rows s, s + S, ...): selected and counted rows, W = sum w and A = sum |w|
 *              (two-word 128-bit atomics), Q = sum w^2 (a three-word carry chain of 64-bit atomic adds: each word
 *              adds its own word plus the carry of the word below, so the words hold the exact sum, below 2^192 by the
 *              host's proof), local while the group stays the same; the largest sum_j X_j^2 of the slice in its own
 *              output slot (the host takes the maximum of the slots)
 *   prefactor  TAYLOR, one work item per row: E = exp(-gamma |x|^2) = 2^e m (tr_kapprox_prefactor) into the row's slot
 *   features   TAYLOR, work item = (item t, slice of rows): w E u^a 2^S of every counted row (the monomial recomputed in
 *              the CPU's multiplication order, tr_kapprox_monomial), outward integers accumulated locally while the
 *              group stays the same, added with tr_atomic_add_wide
 *   RFF        SPEC-001's own device sums (tr_mgpu_fourier_sums: certified cos / sin of omega.x per row summed as exact
 *              scaled integers), after the counts kernel
 * The host validates the views and budgets every device copy before allocating (tr_kapprox_device_bytes); a row whose
 * bounds could not be represented (impossible under the plan's proof) sets a flag and the entry point returns
 * TR_MATH_OVERFLOW; a group id out of range returns TR_MATH_INVALID. */
#include "math_cuda.h"
#include "math_kapprox.h"
#include "math_handles.h"

TR_MGPU_ABI(tr_kapprox_plan_view, TR_KAPPROX_ABI);   /* structs passed to the CUDA library (math_handles.h) */
TR_MGPU_ABI(tr_kapprox_group, TR_KAPPROX_ABI);

TR_DEVICE void tr_kapprox_atomic_add192(uint64_t *dst, const uint64_t *x) {
    uint64_t carry = 0;
    for (int i = 0; i < 3; i++) {
        uint64_t v = x[i] + carry, c1 = v < carry;                  /* x + carry wrapped: only MAX + 1 */
        if (v) { uint64_t old = tr_atomic_add_u64(&dst[i], v); carry = c1 | ((old + v) < old); }
        else carry = c1;
    }
}
TR_DEVICE void tr_kapprox_flush(uint32_t g, uint64_t nr, uint64_t nv, __int128 w, __int128 a, const uint64_t *q,
                                uint64_t *rows, uint64_t *valid, int64_t *W, int64_t *A, uint64_t *Q) {
    if (nr) tr_atomic_add_u64(&rows[g], nr);
    if (nv) tr_atomic_add_u64(&valid[g], nv);
    if (w) tr_atomic_add_wide(&W[2 * (uint64_t)g], &W[2 * (uint64_t)g + 1], w);
    if (a) tr_atomic_add_wide(&A[2 * (uint64_t)g], &A[2 * (uint64_t)g + 1], a);
    if (q[0] | q[1] | q[2]) tr_kapprox_atomic_add192(&Q[3 * (uint64_t)g], q);
}

TR_KERNEL void tr_kapprox_count_kernel(tr_kapprox_plan_view pv, tr_scope_view v, uint64_t slices, uint64_t *rows,
                                       uint64_t *valid, int64_t *W, int64_t *A, uint64_t *Q, uint64_t *slot,
                                       uint64_t *flags) {
    TR_GRID_LOOP(s, slices) {
        uint32_t cur = 0;
        uint64_t nr = 0, nv = 0, q[3] = {0, 0, 0}, mx[3] = {0, 0, 0};
        __int128 w = 0, a = 0;
        int64_t X[TR_KAPPROX_MAX_DIMS];
        for (uint64_t r = s; r < v.rows; r += slices) {
            if (!tr_scope_is_selected(&v, r)) continue;
            uint32_t g = tr_scope_group(&v, r);
            if ((uint64_t)g >= pv.ngroups) { tr_atomic_or_u64(flags, 2); continue; }
            if (nr && g != cur) {
                tr_kapprox_flush(cur, nr, nv, w, a, q, rows, valid, W, A, Q);
                nr = nv = 0; w = a = 0; q[0] = q[1] = q[2] = 0;
            }
            cur = g; nr++;
            if (!tr_kapprox_row_ok(&pv, &v, r)) continue;
            int64_t wt = tr_kapprox_row_weight(&pv, &v, r);
            uint64_t aw = tr_abs_u64(wt);
            nv++; w += wt; a += (__int128)aw;
            tr_kapprox_add192(q, (unsigned __int128)aw * aw);
            for (int32_t j = 0; j < pv.ndims; j++) X[j] = tr_scope_value(&v, pv.column[j], r);
            uint64_t sq[3];
            tr_kapprox_sumsq(X, pv.ndims, sq);
            if (tr_kapprox_cmp192(sq, mx) > 0) { mx[0] = sq[0]; mx[1] = sq[1]; mx[2] = sq[2]; }
        }
        if (nr) tr_kapprox_flush(cur, nr, nv, w, a, q, rows, valid, W, A, Q);
        slot[3 * s] = mx[0]; slot[3 * s + 1] = mx[1]; slot[3 * s + 2] = mx[2];   /* one slot per work item */
    }
}

TR_KERNEL void tr_kapprox_prefactor_kernel(tr_kapprox_plan_view pv, tr_scope_view v, uint64_t *pref, uint64_t *flags) {
    TR_GRID_LOOP(r, v.rows) {
        uint64_t *o = pref + 4 * r;
        if (!tr_kapprox_row_ok(&pv, &v, r)) { o[0] = (uint64_t)TR_KAPPROX_NOROW; continue; }
        int64_t X[TR_KAPPROX_MAX_DIMS], e;
        for (int32_t j = 0; j < pv.ndims; j++) X[j] = tr_scope_value(&v, pv.column[j], r);
        tr_calc_fx m;
        if (!tr_kapprox_prefactor(&pv, X, &e, &m)) { tr_atomic_or_u64(flags, 1); o[0] = (uint64_t)TR_KAPPROX_NOROW; continue; }
        o[0] = (uint64_t)e;
        o[1] = (uint64_t)(unsigned __int128)m.c; o[2] = (uint64_t)((unsigned __int128)m.c >> 64); o[3] = m.r;
    }
}

TR_KERNEL void tr_kapprox_taylor_kernel(tr_kapprox_plan_view pv, tr_scope_view v, const uint64_t *pref, uint64_t slices,
                                        int64_t *sums, uint64_t *flags) {
    TR_GRID_LOOP(i, pv.nitems * slices) {
        uint64_t t = i / slices, s = i % slices;
        uint32_t cur = 0;
        int have = 0;
        __int128 lo = 0, hi = 0;
        int64_t X[TR_KAPPROX_MAX_DIMS];
        for (uint64_t r = s; r < v.rows; r += slices) {
            const uint64_t *o = pref + 4 * r;
            if ((int64_t)o[0] == TR_KAPPROX_NOROW) continue;          /* not counted (or refused: flagged) */
            uint32_t g = tr_scope_group(&v, r);
            if ((uint64_t)g >= pv.ngroups) { tr_atomic_or_u64(flags, 2); continue; }
            if (have && g != cur) {
                uint64_t cell = 2 * ((uint64_t)cur * pv.nitems + t);
                if (lo) tr_atomic_add_wide(&sums[2 * cell], &sums[2 * cell + 1], lo);
                if (hi) tr_atomic_add_wide(&sums[2 * cell + 2], &sums[2 * cell + 3], hi);
                lo = hi = 0;
            }
            cur = g; have = 1;
            for (int32_t j = 0; j < pv.ndims; j++) X[j] = tr_scope_value(&v, pv.column[j], r);
            tr_calc_fx m;
            m.c = (__int128)(((unsigned __int128)o[2] << 64) | o[1]); m.r = o[3];
            tr_calc_fx val = tr_kapprox_monomial(&pv, X, m, t);
            __int128 L, H;
            if (!tr_kapprox_term(tr_kapprox_row_weight(&pv, &v, r), (int64_t)o[0], val, pv.scale, &L, &H)) {
                tr_atomic_or_u64(flags, 1); continue;
            }
            lo += L; hi += H;
        }
        if (have) {
            uint64_t cell = 2 * ((uint64_t)cur * pv.nitems + t);
            if (lo) tr_atomic_add_wide(&sums[2 * cell], &sums[2 * cell + 1], lo);
            if (hi) tr_atomic_add_wide(&sums[2 * cell + 2], &sums[2 * cell + 3], hi);
        }
    }
}

#ifdef __cplusplus
extern "C" {
#endif
int tr_mgpu_kapprox_sums(const tr_kapprox_plan_view *pv, const tr_scope_view *view, tr_kapprox_group *groups,
                         uint64_t *maxsq, tr_math_i128 *sums, uint64_t *frows, uint64_t *fvalid, tr_math_i128 *fwsum,
                         const tr_math_budget *budget, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!tr_kapprox_plan_view_valid(pv, view) || !groups || !maxsq || !sums) return TR_MATH_INVALID;
    const int taylor = pv->method == TR_KAPPROX_TAYLOR;
    if (!taylor && (!frows || !fvalid || !fwsum)) return TR_MATH_INVALID;
    tr_math_budget dflt;
    if (!budget) { tr_budget_default(&dflt); budget = &dflt; }
    const uint64_t G = pv->ngroups, n = view->rows, items = pv->nitems;
    const uint64_t slices = tr_kapprox_slices(n, 1);
    tr_math_need need;
    memset(&need, 0, sizeof need);
    need.need[TR_BUDGET_STATES] = tr_budget_mul(G, items);
    need.need[TR_BUDGET_BYTES] = tr_kapprox_device_bytes(pv, view, slices);
    int rc = tr_budget_check(budget, &need, refusal);
    if (rc) return rc;
    memset(groups, 0, (size_t)G * sizeof *groups);
    maxsq[0] = maxsq[1] = maxsq[2] = 0;
    if (taylor) memset(sums, 0, (size_t)(G * items) * 2 * sizeof(tr_math_i128));
    if (view->selected) {
        tr_scope_view d = *view;
        tr_kapprox_plan_view p = *pv;
        void *mem[3 * TR_SCOPE_MAX_COLUMNS + 16];
        int nmem = 0;
        uint64_t *dsel = NULL, *drows = NULL, *dvalid = NULL, *dQ = NULL, *dslot = NULL, *dflags = NULL, *dpref = NULL;
        uint64_t flags = 0;
        uint32_t *dgroups = NULL;
        int64_t *dW = NULL, *dA = NULL, *dsums = NULL;
        uint8_t *dexps = NULL;
#define TR_KAPPROX_DEV(ptr, bytes) do { if (!rc) { rc = tr_cuda_alloc((void **)&(ptr), (size_t)(bytes)); if (!rc) mem[nmem++] = (void *)(ptr); } } while (0)
        TR_KAPPROX_DEV(dsel, ((n + 63) / 64) * 8);
        if (!rc) rc = tr_cuda_copy_in(dsel, view->selection, (size_t)(((n + 63) / 64) * 8));
        d.selection = dsel;
        if (view->groups) {
            TR_KAPPROX_DEV(dgroups, n * 4);
            if (!rc) rc = tr_cuda_copy_in(dgroups, view->groups, (size_t)n * 4);
            d.groups = dgroups;
        }
        uint32_t used = 0;
        for (int32_t i = 0; i < pv->ndims; i++) used |= (uint32_t)1 << pv->column[i];
        if (pv->weight >= 0) used |= (uint32_t)1 << pv->weight;
        for (int32_t c = 0; c < view->ncols; c++) {
            tr_scope_view_column *k = &d.col[c];
            if (!((used >> c) & 1)) { k->codes = NULL; k->values = NULL; k->valid = NULL; continue; }   /* never read */
            if (view->col[c].values) {
                int64_t *q = NULL;
                TR_KAPPROX_DEV(q, n * 8);
                if (!rc) rc = tr_cuda_copy_in(q, view->col[c].values, (size_t)n * 8);
                k->values = q;
            } else {
                uint32_t *q = NULL;
                TR_KAPPROX_DEV(q, n * 4);
                if (!rc) rc = tr_cuda_copy_in(q, view->col[c].codes, (size_t)n * 4);
                k->codes = q;
            }
            if (view->col[c].valid) {
                uint8_t *q = NULL;
                TR_KAPPROX_DEV(q, n);
                if (!rc) rc = tr_cuda_copy_in(q, view->col[c].valid, (size_t)n);
                k->valid = q;
            }
        }
        TR_KAPPROX_DEV(drows, G * 8);
        TR_KAPPROX_DEV(dvalid, G * 8);
        TR_KAPPROX_DEV(dW, G * 16);
        TR_KAPPROX_DEV(dA, G * 16);
        TR_KAPPROX_DEV(dQ, G * 24);
        TR_KAPPROX_DEV(dslot, slices * 24);
        TR_KAPPROX_DEV(dflags, 8);
        p.parent = NULL; p.axis = NULL;                                /* host only */
        if (taylor) {
            TR_KAPPROX_DEV(dexps, items * (uint64_t)pv->ndims);
            if (!rc) rc = tr_cuda_copy_in(dexps, pv->exps, (size_t)(items * (uint64_t)pv->ndims));
            p.exps = dexps;
            TR_KAPPROX_DEV(dpref, n * 32);
            TR_KAPPROX_DEV(dsums, G * items * 32);
        }
#undef TR_KAPPROX_DEV
        if (!rc) {
            TR_LAUNCH(tr_kapprox_count_kernel, slices, p, d, slices, drows, dvalid, dW, dA, dQ, dslot, dflags);
            rc = tr_cuda_sync();
        }
        if (!rc && taylor) {
            TR_LAUNCH(tr_kapprox_prefactor_kernel, n, p, d, dpref, dflags);
            rc = tr_cuda_sync();
        }
        if (!rc && taylor) {
            uint64_t fs = tr_kapprox_slices(n, items);
            TR_LAUNCH(tr_kapprox_taylor_kernel, items * fs, p, d, dpref, fs, dsums, dflags);
            rc = tr_cuda_sync();
        }
        uint64_t *hrows = NULL, *hvalid = NULL, *hQ = NULL, *hslot = NULL;
        int64_t *hW = NULL, *hA = NULL;
        if (!rc) {
            hrows = (uint64_t *)calloc((size_t)G, 8); hvalid = (uint64_t *)calloc((size_t)G, 8);
            hW = (int64_t *)calloc((size_t)G, 16); hA = (int64_t *)calloc((size_t)G, 16);
            hQ = (uint64_t *)calloc((size_t)G, 24); hslot = (uint64_t *)calloc((size_t)slices, 24);
            if (!hrows || !hvalid || !hW || !hA || !hQ || !hslot) rc = TR_MATH_MEMORY;
        }
        if (!rc) rc = tr_cuda_copy_out(&flags, dflags, 8);
        if (!rc && !flags) rc = tr_cuda_copy_out(hrows, drows, (size_t)G * 8);
        if (!rc && !flags) rc = tr_cuda_copy_out(hvalid, dvalid, (size_t)G * 8);
        if (!rc && !flags) rc = tr_cuda_copy_out(hW, dW, (size_t)G * 16);
        if (!rc && !flags) rc = tr_cuda_copy_out(hA, dA, (size_t)G * 16);
        if (!rc && !flags) rc = tr_cuda_copy_out(hQ, dQ, (size_t)G * 24);
        if (!rc && !flags) rc = tr_cuda_copy_out(hslot, dslot, (size_t)slices * 24);
        if (!rc && !flags && taylor) rc = tr_cuda_copy_out(sums, dsums, (size_t)(G * items) * 32);
        if (!rc && !flags) {
            for (uint64_t g = 0; g < G; g++) {
                tr_kapprox_group *x = &groups[g];
                x->rows = hrows[g]; x->valid = hvalid[g];
                x->W.lo = hW[2 * g]; x->W.hi = hW[2 * g + 1]; x->A.lo = hA[2 * g]; x->A.hi = hA[2 * g + 1];
                x->Q[0] = hQ[3 * g]; x->Q[1] = hQ[3 * g + 1]; x->Q[2] = hQ[3 * g + 2]; x->reserved = 0;
            }
            for (uint64_t s = 0; s < slices; s++)
                if (tr_kapprox_cmp192(hslot + 3 * s, maxsq) > 0) { maxsq[0] = hslot[3 * s]; maxsq[1] = hslot[3 * s + 1]; maxsq[2] = hslot[3 * s + 2]; }
        }
        free(hrows); free(hvalid); free(hW); free(hA); free(hQ); free(hslot);
        for (int i = 0; i < nmem; i++) tr_cuda_free(mem[i]);
        if (rc) return rc;
        if (flags & 2) return TR_MATH_INVALID;
        if (flags) return TR_MATH_OVERFLOW;
    }
    if (!taylor)                                                       /* SPEC-001's own device sums */
        return tr_mgpu_fourier_sums(&pv->fourier, view, sums, frows, fvalid, fwsum, budget, refusal);
    return TR_MATH_OK;
}
#ifdef __cplusplus
}
#endif
