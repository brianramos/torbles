/* MULTI-001 on a CUDA device: the exhaustive certification of a refinement map over every selected row, and batched
 * exact contrasts over many groups, with exactly the results of tr_multi_map_certify / tr_multi_contrasts
 * (math_multi.h).
 * SPDX-License-Identifier: Apache-2.0
 *
 * Certification, two passes over the rows (grid-stride, any thread order):
 *   1. per fine group g: first[g] = the smallest selected row, lo[g] / hi[g] = the smallest / largest coarse group of
 *      its rows (64-bit atomic min / max): the map is functional iff lo = hi for every group with rows;
 *   2. the witness: the smallest selected row r whose coarse group differs from the coarse group of first[fine(r)]
 *      (per-thread minimum, then one atomic min), the row a sequential scan in row order stops at.
 * Atomic min / max commute, so first, lo, hi and the witness do not depend on thread order. The host adopts them with
 * tr_multi_map_from_cert, which checks them against the scopes before the map exists.
 * Contrasts: one work item per (child group, term) writes its own output cell with tr_multi_contrast_at (the inline
 * function of the CPU loop); no atomics. */
#include "math_cuda.h"
#include "math_multi.h"
#include "math_handles.h"

TR_MGPU_ABI(tr_multi_q, 1);              /* structs passed to the CUDA library (math_handles.h) */
TR_MGPU_ABI(tr_multi_cert, 1);

/* the group id of row r in an id array (NULL: one group) */
TR_DEVICE uint32_t tr_multi_gid(const uint32_t *ids, uint64_t r) { return ids ? ids[r] : 0; }
TR_DEVICE int tr_multi_selected(const uint64_t *sel, uint64_t r) { return (int)((sel[r >> 6] >> (r & 63)) & 1); }

TR_KERNEL void tr_multi_first_kernel(const uint64_t *sel, const uint32_t *fg, const uint32_t *cg, uint64_t n,
                                     uint64_t *first, uint64_t *lo, uint64_t *hi) {
    TR_GRID_LOOP(r, n) {
        if (!tr_multi_selected(sel, r)) continue;
        uint32_t g = tr_multi_gid(fg, r);
        uint64_t q = tr_multi_gid(cg, r);
        tr_atomic_min_u64(&first[g], r);
        tr_atomic_min_u64(&lo[g], q);
        tr_atomic_max_u64(&hi[g], q);
    }
}

TR_KERNEL void tr_multi_witness_kernel(const uint64_t *sel, const uint32_t *fg, const uint32_t *cg, uint64_t n,
                                       const uint64_t *first, uint64_t *acc) {
    uint64_t witness = TR_MULTI_NONE, checked = 0;
    TR_GRID_LOOP(r, n) {
        if (!tr_multi_selected(sel, r)) continue;
        checked++;
        uint32_t g = tr_multi_gid(fg, r);
        if (tr_multi_gid(cg, r) != tr_multi_gid(cg, first[g]) && r < witness) witness = r;
    }
    if (witness != TR_MULTI_NONE) tr_atomic_min_u64(&acc[0], witness);
    if (checked) tr_atomic_add_u64(&acc[1], checked);
}

TR_KERNEL void tr_multi_contrast_kernel(tr_bank_states child, tr_bank_states parent, const uint32_t *map,
                                        const uint32_t *next, int32_t kind, const int32_t *terms, int32_t nsel,
                                        uint64_t first, uint64_t cells, tr_multi_q *out) {
    TR_GRID_LOOP(i, cells) tr_multi_contrast_at(&child, &parent, map, next, kind, terms, nsel, first, i, &out[i]);
}

#ifdef __cplusplus
extern "C" {
#endif
int tr_mgpu_multi_certify(const tr_scope_view *fine, const tr_scope_view *coarse, const tr_math_budget *budget,
                          tr_multi_cert *out, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!fine || !coarse || !out || fine->rows != coarse->rows || fine->ngroups < 1 || coarse->ngroups < 1 ||
        out->nfine != fine->ngroups || !out->first || !out->lo || !out->hi ||
        (fine->rows && (!fine->selection || !coarse->selection)) || (fine->rows && fine->ngroups > 1 && !fine->groups) ||
        (coarse->rows && coarse->ngroups > 1 && !coarse->groups)) return -1;                   /* TR_MATH_INVALID */
    tr_math_need need;
    tr_math_budget dflt;
    if (!budget) { tr_budget_default(&dflt); budget = &dflt; }
    tr_multi_certify_device_need(fine, coarse, &need);
    int rc = tr_budget_check(budget, &need, refusal);
    if (rc) return rc;
    const uint64_t n = fine->rows, G = fine->ngroups, nw = (n + 63) / 64;
    for (uint64_t g = 0; g < G; g++) { out->first[g] = TR_MULTI_NONE; out->lo[g] = UINT32_MAX; out->hi[g] = 0; }
    out->witness = TR_MULTI_NONE; out->checked = 0;
    if (!fine->selected) return 0;                    /* nothing selected: nothing to launch */
    uint64_t *init = (uint64_t *)malloc((size_t)G * 8);
    if (!init) return -2;                                                                      /* TR_MATH_MEMORY */
    for (uint64_t g = 0; g < G; g++) init[g] = UINT64_MAX;
    uint64_t acc[2] = {TR_MULTI_NONE, 0};
    uint64_t *dsel = NULL, *dfirst = NULL, *dlo = NULL, *dhi = NULL, *dacc = NULL;
    uint32_t *dfg = NULL, *dcg = NULL;
    rc = tr_cuda_alloc((void **)&dsel, (size_t)nw * 8);
    if (!rc) rc = tr_cuda_copy_in(dsel, fine->selection, (size_t)nw * 8);
    if (!rc && fine->groups) rc = tr_cuda_alloc((void **)&dfg, (size_t)n * 4);
    if (!rc && fine->groups) rc = tr_cuda_copy_in(dfg, fine->groups, (size_t)n * 4);
    if (!rc && coarse->groups) rc = tr_cuda_alloc((void **)&dcg, (size_t)n * 4);
    if (!rc && coarse->groups) rc = tr_cuda_copy_in(dcg, coarse->groups, (size_t)n * 4);
    if (!rc) rc = tr_cuda_alloc((void **)&dfirst, (size_t)G * 8);
    if (!rc) rc = tr_cuda_alloc((void **)&dlo, (size_t)G * 8);
    if (!rc) rc = tr_cuda_alloc((void **)&dhi, (size_t)G * 8);
    if (!rc) rc = tr_cuda_alloc((void **)&dacc, sizeof acc);
    if (!rc) rc = tr_cuda_copy_in(dfirst, init, (size_t)G * 8);
    if (!rc) rc = tr_cuda_copy_in(dlo, init, (size_t)G * 8);
    if (!rc) rc = tr_cuda_copy_in(dacc, acc, sizeof acc);
    free(init);
    if (!rc) {
        TR_LAUNCH(tr_multi_first_kernel, n, dsel, dfg, dcg, n, dfirst, dlo, dhi);
        rc = tr_cuda_sync();
    }
    if (!rc) {
        TR_LAUNCH(tr_multi_witness_kernel, n, dsel, dfg, dcg, n, dfirst, dacc);
        rc = tr_cuda_sync();
    }
    uint64_t *lo64 = NULL, *hi64 = NULL;
    if (!rc) {
        lo64 = (uint64_t *)malloc((size_t)G * 8);
        hi64 = (uint64_t *)malloc((size_t)G * 8);
        if (!lo64 || !hi64) rc = -2;
    }
    if (!rc) rc = tr_cuda_copy_out(out->first, dfirst, (size_t)G * 8);
    if (!rc) rc = tr_cuda_copy_out(lo64, dlo, (size_t)G * 8);
    if (!rc) rc = tr_cuda_copy_out(hi64, dhi, (size_t)G * 8);
    if (!rc) rc = tr_cuda_copy_out(acc, dacc, sizeof acc);
    if (!rc) {
        for (uint64_t g = 0; g < G; g++) {
            out->lo[g] = lo64[g] == UINT64_MAX ? UINT32_MAX : (uint32_t)lo64[g];
            out->hi[g] = (uint32_t)hi64[g];
        }
        out->witness = acc[0]; out->checked = acc[1];
    }
    free(lo64); free(hi64);
    tr_cuda_free(dsel); tr_cuda_free(dfg); tr_cuda_free(dcg); tr_cuda_free(dfirst); tr_cuda_free(dlo);
    tr_cuda_free(dhi); tr_cuda_free(dacc);
    return rc;
}

int tr_mgpu_multi_contrasts(const tr_bank_states *child, const tr_bank_states *parent, const uint32_t *map,
                            const uint32_t *next, int32_t kind, const int32_t *terms, int32_t nsel, uint64_t first,
                            uint64_t count, const tr_math_budget *budget, tr_multi_q *out, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!child || !parent || !map || !next || !terms || !tr_multi_kind_ok(kind) || child->nterms != parent->nterms ||
        child->nterms < 1 || nsel < 1 || nsel > child->nterms || child->ngroups < 1 || parent->ngroups < 1 ||
        !child->rows || !parent->rows || !child->count || !child->sum || !parent->count || !parent->sum ||
        first > child->ngroups || count > child->ngroups - first || (count && !out)) return -1;  /* TR_MATH_INVALID */
    for (int32_t a = 0; a < nsel; a++) {
        if (terms[a] < 0 || terms[a] >= child->nterms) return -1;
        for (int32_t b = 0; b < a; b++) if (terms[a] == terms[b]) return -1;
    }
    for (uint64_t g = 0; g < child->ngroups; g++)
        if ((map[g] != TR_MULTI_UNMAPPED && (uint64_t)map[g] >= parent->ngroups) ||
            (next[g] != TR_MULTI_UNMAPPED && (uint64_t)next[g] >= child->ngroups)) return -1;
    tr_math_need need;
    tr_math_budget dflt;
    if (!budget) { tr_budget_default(&dflt); budget = &dflt; }
    tr_multi_contrasts_device_need(child->ngroups, parent->ngroups, child->nterms, nsel, count, &need);
    int rc = tr_budget_check(budget, &need, refusal);
    if (rc || !count) return rc;
    const uint64_t m = (uint64_t)child->nterms, Gc = child->ngroups, Gp = parent->ngroups, cells = count * (uint64_t)nsel;
    tr_bank_states dc = *child, dp = *parent;
    uint64_t *crows = NULL, *ccount = NULL, *prows = NULL, *pcount = NULL;
    tr_math_i128 *csum = NULL, *psum = NULL;
    uint32_t *dmap = NULL, *dnext = NULL;
    int32_t *dterms = NULL;
    tr_multi_q *dout = NULL;
    rc = tr_cuda_alloc((void **)&crows, (size_t)Gc * 8);
    if (!rc) rc = tr_cuda_alloc((void **)&ccount, (size_t)(Gc * m) * 8);
    if (!rc) rc = tr_cuda_alloc((void **)&csum, (size_t)(Gc * m) * 16);
    if (!rc) rc = tr_cuda_alloc((void **)&prows, (size_t)Gp * 8);
    if (!rc) rc = tr_cuda_alloc((void **)&pcount, (size_t)(Gp * m) * 8);
    if (!rc) rc = tr_cuda_alloc((void **)&psum, (size_t)(Gp * m) * 16);
    if (!rc) rc = tr_cuda_alloc((void **)&dmap, (size_t)Gc * 4);
    if (!rc) rc = tr_cuda_alloc((void **)&dnext, (size_t)Gc * 4);
    if (!rc) rc = tr_cuda_alloc((void **)&dterms, (size_t)nsel * 4);
    if (!rc) rc = tr_cuda_alloc((void **)&dout, (size_t)cells * sizeof(tr_multi_q));
    if (!rc) rc = tr_cuda_copy_in(crows, child->rows, (size_t)Gc * 8);
    if (!rc) rc = tr_cuda_copy_in(ccount, child->count, (size_t)(Gc * m) * 8);
    if (!rc) rc = tr_cuda_copy_in(csum, child->sum, (size_t)(Gc * m) * 16);
    if (!rc) rc = tr_cuda_copy_in(prows, parent->rows, (size_t)Gp * 8);
    if (!rc) rc = tr_cuda_copy_in(pcount, parent->count, (size_t)(Gp * m) * 8);
    if (!rc) rc = tr_cuda_copy_in(psum, parent->sum, (size_t)(Gp * m) * 16);
    if (!rc) rc = tr_cuda_copy_in(dmap, map, (size_t)Gc * 4);
    if (!rc) rc = tr_cuda_copy_in(dnext, next, (size_t)Gc * 4);
    if (!rc) rc = tr_cuda_copy_in(dterms, terms, (size_t)nsel * 4);
    dc.rows = crows; dc.count = ccount; dc.sum = csum;
    dp.rows = prows; dp.count = pcount; dp.sum = psum;
    if (!rc) {
        TR_LAUNCH(tr_multi_contrast_kernel, cells, dc, dp, dmap, dnext, kind, dterms, nsel, first, cells, dout);
        rc = tr_cuda_sync();
    }
    if (!rc) rc = tr_cuda_copy_out(out, dout, (size_t)cells * sizeof(tr_multi_q));
    tr_cuda_free(crows); tr_cuda_free(ccount); tr_cuda_free(csum); tr_cuda_free(prows); tr_cuda_free(pcount);
    tr_cuda_free(psum); tr_cuda_free(dmap); tr_cuda_free(dnext); tr_cuda_free(dterms); tr_cuda_free(dout);
    return rc;
}
#ifdef __cplusplus
}
#endif
