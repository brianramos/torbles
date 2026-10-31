/* COUNT-002 on a CUDA device: the data-parallel parts of symbolic counting, with exactly the raw values of the CPU
 * functions of math_symcount.c (the answers are adopted natively: tr_symc_dd_adopt, tr_symc_brute_adopt).
 * SPDX-License-Identifier: Apache-2.0
 *
 * tr_mgpu_symcount_levels  the weighted count of a compiled decision diagram: one launch per level from the deepest,
 *                          one work item per node of the level, each computing its exact L-limb value from its
 *                          children's (finished by the earlier launches) with the shared tr_symc_node_count into its
 *                          own slot; then one work item per root (tr_symc_root_count). No atomics: one writer per slot.
 * tr_mgpu_symcount_brute   every assignment of the program's k variables: work items are contiguous slices of 64-point
 *                          words; each evaluates its words with BOOL-001's word evaluator (tr_symc_brute_classes),
 *                          counts the TRUE / FALSE / UNKNOWN assignments with popcounts and sums the weights of the
 *                          TRUE and FALSE ones (tr_symc_brute_weigh) locally, then adds each count with one 64-bit
 *                          atomic and each L-limb sum with a multi-word atomic add (each word's carry from the value
 *                          its own atomic returned: exact modulo 2^(64 L), exact because the host proved the total
 *                          below 2^(64 L)). Sums are order independent.
 * The host checks the views, the scaled weights (each below 2^126) and the budget (bytes = every device copy and the
 * scratch) before allocating anything on the device. */
#include "math_cuda.h"
#include "math_symcount.h"
#include "math_handles.h"

TR_MGPU_ABI(tr_symc_node, TR_SYMC_ABI);           /* structs passed to the CUDA library (math_handles.h) */
TR_MGPU_ABI(tr_symc_dd_view, TR_SYMC_ABI);

/* acc[0..L) += x[0..L) with one atomic per nonzero word */
TR_DEVICE void tr_symc_atomic_add_words(uint64_t *acc, const uint64_t *x, uint32_t L) {
    uint64_t carry = 0;
    for (uint32_t i = 0; i < L; i++) {
        uint64_t y = x[i] + carry, c1 = y < carry, c2 = 0;
        if (y) {
            uint64_t old = tr_atomic_add_u64(&acc[i], y);
            c2 = (uint64_t)((old + y) < old);
        }
        carry = c1 | c2;
    }
}

TR_KERNEL void tr_symc_level_kernel(tr_symc_dd_view d, const uint64_t *w, uint32_t L, uint64_t *values, uint64_t first,
                                    uint64_t count, uint64_t *tmp, uint64_t *over) {
    TR_GRID_LOOP(i, count) {
        if (tr_symc_node_count(&d, w, L, values, first + i, tmp + i * 2 * (uint64_t)L)) tr_atomic_or_u64(over, 1);
    }
}

TR_KERNEL void tr_symc_root_kernel(tr_symc_dd_view d, const uint64_t *w, uint32_t L, const uint64_t *values,
                                   uint64_t *out, uint64_t *tmp, uint64_t *over) {
    TR_GRID_LOOP(r, d.nroots) {
        if (tr_symc_root_count(&d, w, L, values, (uint32_t)r, out + r * (uint64_t)L, tmp + r * 2 * (uint64_t)L))
            tr_atomic_or_u64(over, 1);
    }
}

TR_KERNEL void tr_symc_brute_kernel(tr_bool_program_view p, tr_bool_domain d, const uint64_t *w, uint32_t L,
                                    uint64_t items, uint64_t *vals, uint64_t *sums, uint64_t *counts, uint64_t *wsum,
                                    uint64_t *over) {
    TR_GRID_LOOP(i, items) {
        uint64_t lo = (uint64_t)(((unsigned __int128)d.nw * i) / items);
        uint64_t hi = (uint64_t)(((unsigned __int128)d.nw * (i + 1)) / items);
        uint64_t *val = vals + i * 2 * (uint64_t)p.nnodes;
        uint64_t *st = w ? sums + i * 5 * (uint64_t)L : (uint64_t *)0;
        uint64_t c0 = 0, c1 = 0, c2 = 0;
        int ov = 0;
        if (w) tr_symc_w_zero(st, 2 * L);
        for (uint64_t word = lo; word < hi; word++) {
            uint64_t t, f, u;
            tr_symc_brute_classes(&p, &d, word, val, &t, &f, &u);
            c0 += (uint64_t)tr_popc64(t); c1 += (uint64_t)tr_popc64(f); c2 += (uint64_t)tr_popc64(u);
            if (w) {
                ov |= tr_symc_brute_weigh(w, p.k, word, t, L, st, st + 2 * L);
                ov |= tr_symc_brute_weigh(w, p.k, word, f, L, st + L, st + 2 * L);
            }
        }
        if (c0) tr_atomic_add_u64(&counts[0], c0);
        if (c1) tr_atomic_add_u64(&counts[1], c1);
        if (c2) tr_atomic_add_u64(&counts[2], c2);
        if (w) { tr_symc_atomic_add_words(wsum, st, L); tr_symc_atomic_add_words(wsum + L, st + L, L); }
        if (ov) tr_atomic_or_u64(over, 1);
    }
}

/* tr_budget_check, refusing a size past 64 bits even under an unlimited budget */
static int tr_symc_gpu_check(const tr_math_budget *b, const tr_math_need *n, tr_math_refusal *r) {
    int rc = tr_budget_check(b, n, r);
    if (rc) return rc;
    for (int i = 0; i < TR_BUDGET_RESOURCES; i++)
        if (n->need[i] == TR_BUDGET_UNKNOWN) {
            if (r) { r->resource = i; r->refused = 1; r->need = n->need[i]; r->limit = b->limit[i]; }
            return TR_MATH_BUDGET_CODE;
        }
    return 0;
}

#ifdef __cplusplus
extern "C" {
#endif
int tr_mgpu_symcount_levels(const tr_symc_dd_view *d, const uint64_t *scaled, uint32_t L, uint64_t *roots_out,
                            const tr_math_budget *budget, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!d || tr_symc_dd_view_check(d) || L < 1 || L > TR_SYMC_MAX_LIMBS || (d->nroots && !roots_out) ||
        tr_symc_scaled_check(scaled, d->nlevels)) return -1;                              /* TR_MATH_INVALID */
    tr_math_need need;
    for (int i = 0; i < TR_BUDGET_RESOURCES; i++) need.need[i] = 0;
    need.need[TR_BUDGET_CIRCUIT] = d->nnodes;
    need.need[TR_BUDGET_OUTPUT] = d->nroots;
    need.need[TR_BUDGET_BYTES] = tr_symc_levels_device_bytes(d, L);
    tr_math_budget dflt;
    if (!budget) { tr_budget_default(&dflt); budget = &dflt; }
    int rc = tr_symc_gpu_check(budget, &need, refusal);
    if (rc) return rc;
    uint64_t wide = d->nroots;
    for (uint32_t l = 0; l < d->nlevels; l++) if (d->count[l] > wide) wide = d->count[l];
    const uint64_t vbytes = d->nnodes * L * 8, rbytes = (uint64_t)d->nroots * L * 8;
    tr_symc_node *dnode = NULL;
    uint32_t *droots = NULL;
    uint64_t *dw = NULL, *dvalues = NULL, *dout = NULL, *dtmp = NULL, *dover = NULL;
    uint64_t *term = (uint64_t *)calloc(2 * (size_t)L, 8);
    if (!term) return -2;                                                                 /* TR_MATH_MEMORY */
    tr_symc_terminals(term, L);
    rc = tr_cuda_alloc((void **)&dnode, (size_t)d->nnodes * sizeof(tr_symc_node));
    if (!rc) rc = tr_cuda_alloc((void **)&droots, (size_t)(d->nroots ? d->nroots : 1) * 4);
    if (!rc) rc = tr_cuda_alloc((void **)&dw, (size_t)(d->nlevels ? d->nlevels : 1) * 32);
    if (!rc) rc = tr_cuda_alloc((void **)&dvalues, (size_t)vbytes);
    if (!rc) rc = tr_cuda_alloc((void **)&dout, (size_t)(rbytes ? rbytes : 8));
    if (!rc) rc = tr_cuda_alloc((void **)&dtmp, (size_t)(wide ? wide : 1) * 2 * L * 8);
    if (!rc) rc = tr_cuda_alloc((void **)&dover, 8);
    if (!rc) rc = tr_cuda_copy_in(dnode, d->node, (size_t)d->nnodes * sizeof(tr_symc_node));
    if (!rc && d->nroots) rc = tr_cuda_copy_in(droots, d->roots, (size_t)d->nroots * 4);
    if (!rc && d->nlevels) rc = tr_cuda_copy_in(dw, scaled, (size_t)d->nlevels * 32);
    if (!rc) rc = tr_cuda_copy_in(dvalues, term, 2 * (size_t)L * 8);
    tr_symc_dd_view dv = *d;
    dv.node = dnode; dv.roots = droots; dv.order = NULL; dv.first = NULL; dv.count = NULL;
    for (uint32_t l = d->nlevels; l-- > 0 && !rc;) {                    /* the deepest level first */
        if (!d->count[l]) continue;
        TR_LAUNCH(tr_symc_level_kernel, d->count[l], dv, (const uint64_t *)dw, L, dvalues, d->first[l], d->count[l],
                  dtmp, dover);
        rc = tr_cuda_sync();
    }
    if (!rc && d->nroots) {
        TR_LAUNCH(tr_symc_root_kernel, d->nroots, dv, (const uint64_t *)dw, L, (const uint64_t *)dvalues, dout, dtmp, dover);
        rc = tr_cuda_sync();
    }
    uint64_t over = 0;
    if (!rc && rbytes) rc = tr_cuda_copy_out(roots_out, dout, (size_t)rbytes);
    if (!rc) rc = tr_cuda_copy_out(&over, dover, 8);
    tr_cuda_free(dnode); tr_cuda_free(droots); tr_cuda_free(dw); tr_cuda_free(dvalues); tr_cuda_free(dout);
    tr_cuda_free(dtmp); tr_cuda_free(dover);
    free(term);
    if (rc) return rc;
    return over ? -3 : 0;                                                                 /* TR_MATH_OVERFLOW */
}

int tr_mgpu_symcount_brute(const tr_bool_program_view *p, const uint64_t *scaled, uint32_t L, uint64_t *counts,
                           uint64_t *wsum, const tr_math_budget *budget, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!p || tr_bool_program_check(p) || !counts || L < 1 || L > TR_SYMC_MAX_LIMBS) return -1;
    if (p->k > TR_SYMC_MAX_BRUTE_VARS) return -12;                                       /* TR_MATH_UNSUPPORTED */
    if (scaled && (!wsum || tr_symc_scaled_check(scaled, (uint64_t)p->k))) return -1;
    const int weighted = scaled != NULL;
    tr_math_need need;
    for (int i = 0; i < TR_BUDGET_RESOURCES; i++) need.need[i] = 0;
    need.need[TR_BUDGET_VARIABLES] = (uint64_t)p->k;
    need.need[TR_BUDGET_SUPPORT] = (uint64_t)1 << p->k;
    need.need[TR_BUDGET_CIRCUIT] = (uint64_t)p->nnodes;
    need.need[TR_BUDGET_BYTES] = tr_symc_brute_device_bytes(p, L, weighted);
    tr_math_budget dflt;
    if (!budget) { tr_budget_default(&dflt); budget = &dflt; }
    int rc = tr_symc_gpu_check(budget, &need, refusal);
    if (rc) return rc;
    const tr_bool_domain dom = tr_symc_assignments(p->k);
    const uint64_t items = tr_symc_brute_items(dom.nw);
    tr_bool_node *dnodes = NULL;
    uint64_t *ddeps = NULL, *dw = NULL, *dval = NULL, *dsum = NULL, *dcounts = NULL, *dwsum = NULL, *dover = NULL;
    uint32_t *dsched = NULL;
    rc = tr_cuda_alloc((void **)&dnodes, (size_t)p->nnodes * sizeof(tr_bool_node));
    if (!rc) rc = tr_cuda_alloc((void **)&ddeps, (size_t)p->nnodes * 8);
    if (!rc) rc = tr_cuda_alloc((void **)&dsched, (size_t)p->nsched * 4);
    if (!rc && weighted) rc = tr_cuda_alloc((void **)&dw, (size_t)(p->k ? p->k : 1) * 32);
    if (!rc) rc = tr_cuda_alloc((void **)&dval, (size_t)(items * 2 * (uint64_t)p->nnodes * 8));
    if (!rc && weighted) rc = tr_cuda_alloc((void **)&dsum, (size_t)(items * 5 * (uint64_t)L * 8));
    if (!rc) rc = tr_cuda_alloc((void **)&dcounts, 24);
    if (!rc) rc = tr_cuda_alloc((void **)&dwsum, 2 * (size_t)L * 8);
    if (!rc) rc = tr_cuda_alloc((void **)&dover, 8);
    if (!rc) rc = tr_cuda_copy_in(dnodes, p->nodes, (size_t)p->nnodes * sizeof(tr_bool_node));
    if (!rc) rc = tr_cuda_copy_in(ddeps, p->deps, (size_t)p->nnodes * 8);
    if (!rc) rc = tr_cuda_copy_in(dsched, p->schedule, (size_t)p->nsched * 4);
    if (!rc && weighted && p->k) rc = tr_cuda_copy_in(dw, scaled, (size_t)p->k * 32);
    tr_bool_program_view dp = *p;
    dp.nodes = dnodes; dp.deps = ddeps; dp.schedule = dsched;
    if (!rc) {
        TR_LAUNCH(tr_symc_brute_kernel, items, dp, dom, (const uint64_t *)(weighted ? dw : NULL), L, items, dval, dsum,
                  dcounts, dwsum, dover);
        rc = tr_cuda_sync();
    }
    uint64_t over = 0;
    if (!rc) rc = tr_cuda_copy_out(counts, dcounts, 24);
    if (!rc && weighted) rc = tr_cuda_copy_out(wsum, dwsum, 2 * (size_t)L * 8);
    if (!rc) rc = tr_cuda_copy_out(&over, dover, 8);
    tr_cuda_free(dnodes); tr_cuda_free(ddeps); tr_cuda_free(dsched); tr_cuda_free(dw); tr_cuda_free(dval);
    tr_cuda_free(dsum); tr_cuda_free(dcounts); tr_cuda_free(dwsum); tr_cuda_free(dover);
    if (rc) return rc;
    return over ? -3 : 0;
}
#ifdef __cplusplus
}
#endif
