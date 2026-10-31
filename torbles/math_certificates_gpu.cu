/* FND-007 on a CUDA device: exhaustive re-verification of one or many certificate claims over the selected rows of a
 * scope view, with the semantics of tr_cert_verify_view (math_certificates.h) and the same evidence bit for bit.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Work items are (claim j, slice s): slice s is a contiguous run of selection words. A work item evaluates the shared
 * row function tr_cert_row on its selected rows, counts checked / excluded / violations / overflow locally and adds
 * them with one atomic each, and keeps its smallest violating row, merged with an atomic minimum, so the evidence
 * does not depend on thread order. Two claim kinds need a second launch:
 *   UNIQUE    the first launch keeps, per value of the column's [min, max] domain, the smallest row holding it (atomic
 *             minimum); the second counts the rows that are not the first occurrence of their value (the smallest such
 *             row is the witness). The domain array is budgeted (8 bytes per value) before anything is allocated.
 *   COVERAGE  one launch marks the present groups (a group with a selected row; atomic OR into a bitmap); the row
 *             launch ORs each row's required-category bit into its group's mask (pending bits of one group are kept
 *             locally and flushed when the group changes); a launch over (claim, slice of groups) counts the present
 *             groups and those missing a required category (the smallest is the witness).
 * No barrier, no shared memory. The host checks the budget (tr_cert_device_need) and the claims' structure before
 * anything is copied; the claims must be checked by tr_cert_claim_check (libalg) beforehand. */
#include "math_cuda.h"
#include "math_certificates.h"

TR_MGPU_ABI(tr_cert_claim, TR_CERT_ABI);          /* structs passed to the CUDA library (math_handles.h) */
TR_MGPU_ABI(tr_cert_evidence, TR_CERT_ABI);
TR_MGPU_ABI(tr_cert_bit, TR_CERT_ABI);

#define TR_CERT_ACC 8                     /* per claim: checked, excluded, violations, witness, overflow, (unused) */
#define TR_CERT_SLICE_WORDS 64            /* selection words per work item (4096 rows), at least */
#define TR_CERT_MAX_SLICES 4096

TR_KERNEL void tr_cert_fill_kernel(uint64_t *p, uint64_t n, uint64_t value) {
    TR_GRID_LOOP(i, n) p[i] = value;
}

TR_KERNEL void tr_cert_presence_kernel(tr_scope_view v, uint64_t slices, uint64_t per, uint64_t *present) {
    uint64_t nw = (v.rows + 63) / 64;
    TR_GRID_LOOP(s, slices) {
        uint64_t w0 = s * per, w1 = w0 + per < nw ? w0 + per : nw;
        uint32_t cur = UINT32_MAX;
        for (uint64_t w = w0; w < w1; w++) {
            uint64_t bits = v.selection[w];
            while (bits) {
                uint64_t r = w * 64 + (uint64_t)tr_ipoly_ctz(bits);
                bits &= bits - 1;
                uint32_t g = tr_scope_group(&v, r);
                if (g != cur) { tr_atomic_or_u64(&present[g >> 6], (uint64_t)1 << (g & 63)); cur = g; }
            }
        }
    }
}

TR_KERNEL void tr_cert_rows_kernel(tr_scope_view v, const tr_cert_claim *claims, int32_t n, uint64_t slices,
                                   uint64_t per, uint64_t *acc, uint64_t *masks, const uint64_t *mask_off,
                                   uint64_t *first, const uint64_t *first_off) {
    uint64_t nw = (v.rows + 63) / 64;
    TR_GRID_LOOP(i, (uint64_t)n * slices) {
        uint64_t j = i / slices, s = i % slices;
        const tr_cert_claim *c = &claims[j];
        uint64_t w0 = s * per, w1 = w0 + per < nw ? w0 + per : nw;
        uint64_t checked = 0, excluded = 0, violations = 0, witness = UINT64_MAX, overflow = 0;
        uint32_t cur = UINT32_MAX;
        uint64_t pending = 0;
        int64_t lo = c->kind == TR_CLAIM_UNIQUE ? v.col[c->col[0]].min : 0;
        for (uint64_t w = w0; w < w1; w++) {
            uint64_t bits = v.selection[w];
            while (bits) {
                uint64_t r = w * 64 + (uint64_t)tr_ipoly_ctz(bits), bit = 0;
                bits &= bits - 1;
                int st = tr_cert_row(&v, c, r, &bit);
                if (c->kind == TR_CLAIM_COVERAGE) {
                    if (st == TR_CERT_ROW_EXCLUDED) { excluded++; continue; }
                    uint32_t g = tr_scope_group(&v, r);
                    if (g != cur) {
                        if (pending) tr_atomic_or_u64(&masks[mask_off[j] + cur], pending);
                        cur = g; pending = 0;
                    }
                    pending |= bit;
                    continue;
                }
                if (st == TR_CERT_ROW_EXCLUDED) { excluded++; continue; }
                if (st == TR_CERT_ROW_OVERFLOW) { overflow++; continue; }
                checked++;
                if (st == TR_CERT_ROW_VIOLATES) { violations++; if (r < witness) witness = r; continue; }
                if (c->kind == TR_CLAIM_UNIQUE) {
                    uint64_t d = (uint64_t)tr_scope_value(&v, c->col[0], r) - (uint64_t)lo;
                    tr_atomic_min_u64(&first[first_off[j] + d], r);
                }
            }
        }
        if (pending) tr_atomic_or_u64(&masks[mask_off[j] + cur], pending);
        uint64_t *a = acc + j * TR_CERT_ACC;
        if (checked) tr_atomic_add_u64(&a[0], checked);
        if (excluded) tr_atomic_add_u64(&a[1], excluded);
        if (violations) { tr_atomic_add_u64(&a[2], violations); tr_atomic_min_u64(&a[3], witness); }
        if (overflow) tr_atomic_add_u64(&a[4], overflow);
    }
}

/* UNIQUE, second pass: rows that are not the first occurrence of their value */
TR_KERNEL void tr_cert_unique_kernel(tr_scope_view v, const tr_cert_claim *claims, const int32_t *which, int32_t nu,
                                     uint64_t slices, uint64_t per, uint64_t *acc, const uint64_t *first,
                                     const uint64_t *first_off) {
    uint64_t nw = (v.rows + 63) / 64;
    TR_GRID_LOOP(i, (uint64_t)nu * slices) {
        int32_t j = which[i / slices];
        uint64_t s = i % slices;
        const tr_cert_claim *c = &claims[j];
        uint64_t w0 = s * per, w1 = w0 + per < nw ? w0 + per : nw;
        uint64_t violations = 0, witness = UINT64_MAX;
        int64_t lo = v.col[c->col[0]].min;
        for (uint64_t w = w0; w < w1; w++) {
            uint64_t bits = v.selection[w];
            while (bits) {
                uint64_t r = w * 64 + (uint64_t)tr_ipoly_ctz(bits), bit = 0;
                bits &= bits - 1;
                if (tr_cert_row(&v, c, r, &bit) != TR_CERT_ROW_HOLDS) continue;
                uint64_t d = (uint64_t)tr_scope_value(&v, c->col[0], r) - (uint64_t)lo;
                if (first[first_off[j] + d] != r) { violations++; if (r < witness) witness = r; }
            }
        }
        uint64_t *a = acc + (uint64_t)j * TR_CERT_ACC;
        if (violations) { tr_atomic_add_u64(&a[2], violations); tr_atomic_min_u64(&a[3], witness); }
    }
}

/* COVERAGE: the present groups, and those missing a required category */
TR_KERNEL void tr_cert_cover_kernel(const tr_cert_claim *claims, const int32_t *which, int32_t nc, uint64_t ngroups,
                                    uint64_t gslices, uint64_t gper, const uint64_t *present, const uint64_t *masks,
                                    const uint64_t *mask_off, uint64_t *acc) {
    TR_GRID_LOOP(i, (uint64_t)nc * gslices) {
        int32_t j = which[i / gslices];
        uint64_t s = i % gslices, g0 = s * gper, g1 = g0 + gper < ngroups ? g0 + gper : ngroups;
        uint64_t full = tr_cert_full_mask(claims[j].nvalues), checked = 0, violations = 0, witness = UINT64_MAX;
        for (uint64_t g = g0; g < g1; g++) {
            if (!((present[g >> 6] >> (g & 63)) & 1)) continue;
            checked++;
            if ((masks[mask_off[j] + g] & full) != full) { violations++; if (g < witness) witness = g; }
        }
        uint64_t *a = acc + (uint64_t)j * TR_CERT_ACC;
        if (checked) tr_atomic_add_u64(&a[0], checked);
        if (violations) { tr_atomic_add_u64(&a[2], violations); tr_atomic_min_u64(&a[3], witness); }
    }
}

/* the structure the kernels rely on (the full check is tr_cert_claim_check's, in libalg) */
static int tr_cert_gpu_claim_ok(const tr_scope_view *v, const tr_cert_claim *c) {
    if (c->kind < 1 || c->kind >= TR_CLAIM_KINDS || c->ncols < 1 || c->ncols > TR_CERT_MAX_COLS ||
        c->policy < TR_CERT_NULL_FORBID || c->policy > TR_CERT_NULL_PROPAGATE ||
        (c->carrier != TR_CARRIER_I128 && c->carrier != TR_CARRIER_GFP) || (c->carrier == TR_CARRIER_GFP && c->gf.p < 2))
        return 0;
    for (int32_t i = 0; i < c->ncols; i++) {
        int32_t k = c->col[i];
        if (k < 0 || k >= v->ncols || (v->rows && !v->col[k].codes && !v->col[k].values)) return 0;
    }
    switch (c->kind) {
    case TR_CLAIM_AFFINE: return c->ncols == 2;
    case TR_CLAIM_SUM: case TR_CLAIM_PRODUCT: return c->ncols == 3;
    case TR_CLAIM_REFINEMENT: return c->ncols == 2 && (!c->nvalues || (c->keys && c->targets));
    case TR_CLAIM_COVERAGE: return c->ncols == 1 && c->nvalues <= TR_CERT_MAX_REQUIRED && (!c->nvalues || c->keys) &&
                                   c->policy == TR_CERT_NULL_SKIP;
    case TR_CLAIM_PARITY:
        if (c->nbits < 1 || c->nbits > TR_CERT_MAX_BITS || !c->bits) return 0;
        for (int32_t j = 0; j < c->nbits; j++)
            if (c->bits[j].col < 0 || c->bits[j].col >= c->ncols || c->bits[j].bit < 0 || c->bits[j].bit > 63) return 0;
        return 1;
    case TR_CLAIM_POLYNOMIAL: return c->nterms <= TR_CERT_MAX_TERMS && (!c->nterms || (c->exponents && c->coefficients));
    case TR_CLAIM_REFERENCE: return c->ncols == 1 && (!c->nvalues || c->used);
    default: return c->ncols == 1;
    }
}

#ifdef __cplusplus
extern "C" {
#endif
int tr_mgpu_cert_verify(const tr_scope_view *view, const tr_cert_claim *claims, int32_t n, const tr_math_budget *budget,
                        tr_cert_evidence *out, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!view || n < 0 || (n && (!claims || !out)) || view->ngroups < 1 || (view->rows && !view->selection) ||
        (view->rows && view->ngroups > 1 && !view->groups) || view->ncols < 0 || view->ncols > TR_SCOPE_MAX_COLUMNS)
        return -1;                                                                              /* TR_MATH_INVALID */
    for (int32_t j = 0; j < n; j++) if (!tr_cert_gpu_claim_ok(view, &claims[j])) return -1;
    tr_math_budget dflt;
    if (!budget) { tr_budget_default(&dflt); budget = &dflt; }
    tr_math_need need;
    tr_cert_device_need(view, claims, n, &need);
    int rc = tr_budget_check(budget, &need, refusal);
    if (rc) return rc;
    for (int32_t j = 0; j < n; j++) {
        memset(&out[j], 0, sizeof out[j]);
        out[j].rows = view->rows; out[j].selected = view->selected; out[j].witness = TR_CERT_NONE;
        out[j].unit = claims[j].kind == TR_CLAIM_COVERAGE ? TR_CERT_UNIT_GROUPS : TR_CERT_UNIT_ROWS;
        out[j].holds = 1;
    }
    if (!n || !view->selected) return 0;                    /* nothing selected: vacuous evidence, nothing to launch */
    const uint64_t rows = view->rows, nw = (rows + 63) / 64, G = view->ngroups;
    uint32_t cols = 0;
    int32_t ncover = 0, nuniq = 0;
    for (int32_t j = 0; j < n; j++) {
        for (int32_t i = 0; i < claims[j].ncols; i++) cols |= (uint32_t)1 << claims[j].col[i];
        ncover += claims[j].kind == TR_CLAIM_COVERAGE;
        nuniq += claims[j].kind == TR_CLAIM_UNIQUE;
    }
    tr_cert_claim *dc = (tr_cert_claim *)calloc((size_t)n, sizeof *dc);
    uint64_t *acc = (uint64_t *)calloc((size_t)n * TR_CERT_ACC, 8);
    uint64_t *moff = (uint64_t *)calloc((size_t)n, 8), *foff = (uint64_t *)calloc((size_t)n, 8);
    int32_t *cover = (int32_t *)calloc((size_t)(ncover ? ncover : 1), 4), *uniq = (int32_t *)calloc((size_t)(nuniq ? nuniq : 1), 4);
    void **dmem = (void **)calloc((size_t)(3 * TR_SCOPE_MAX_COLUMNS + 16 + 6 * n), sizeof(void *));
    int nmem = 0;
    if (!dc || !acc || !moff || !foff || !cover || !uniq || !dmem) {
        free(dc); free(acc); free(moff); free(foff); free(cover); free(uniq); free(dmem); return -2;   /* TR_MATH_MEMORY */
    }
#define TR_CERT_DEV(ptr, bytes) do { if (!rc) { rc = tr_cuda_alloc((void **)&(ptr), (size_t)(bytes)); if (!rc) dmem[nmem++] = (void *)(ptr); } } while (0)
#define TR_CERT_IN(dst, src, bytes) do { TR_CERT_DEV(dst, bytes); if (!rc && (uint64_t)(bytes) != 0) rc = tr_cuda_copy_in(dst, src, (size_t)(bytes)); } while (0)
    tr_scope_view d = *view;
    uint64_t *dsel = NULL, *dacc = NULL, *dmasks = NULL, *dfirst = NULL, *dpresent = NULL, *dmoff = NULL, *dfoff = NULL;
    uint32_t *dgroups = NULL;
    int32_t *dcover = NULL, *duniq = NULL;
    tr_cert_claim *dclaims = NULL;
    TR_CERT_IN(dsel, view->selection, nw * 8);
    d.selection = dsel;
    d.groups = NULL;
    if (ncover && view->groups) { TR_CERT_IN(dgroups, view->groups, rows * 4); d.groups = dgroups; }
    for (int32_t c = 0; c < view->ncols; c++) {
        tr_scope_view_column *k = &d.col[c];
        if (!((cols >> c) & 1)) { k->codes = NULL; k->values = NULL; k->valid = NULL; continue; }
        if (view->col[c].values) { int64_t *p = NULL; TR_CERT_IN(p, view->col[c].values, rows * 8); k->values = p; }
        else { uint32_t *p = NULL; TR_CERT_IN(p, view->col[c].codes, rows * 4); k->codes = p; }
        if (view->col[c].valid) { uint8_t *p = NULL; TR_CERT_IN(p, view->col[c].valid, rows); k->valid = p; }
    }
    uint64_t mtotal = 0, ftotal = 0;
    int32_t ic = 0, iu = 0;
    for (int32_t j = 0; j < n && !rc; j++) {
        const tr_cert_claim *c = &claims[j];
        dc[j] = *c;
        if (c->kind == TR_CLAIM_PARITY) {
            tr_cert_bit *p = NULL; TR_CERT_IN(p, c->bits, (uint64_t)c->nbits * sizeof(tr_cert_bit)); dc[j].bits = p;
        }
        if (c->kind == TR_CLAIM_POLYNOMIAL && c->nterms) {
            uint8_t *e = NULL; int64_t *k = NULL;
            TR_CERT_IN(e, c->exponents, c->nterms * (uint64_t)c->ncols); TR_CERT_IN(k, c->coefficients, c->nterms * 8);
            dc[j].exponents = e; dc[j].coefficients = k;
        }
        if ((c->kind == TR_CLAIM_REFINEMENT || c->kind == TR_CLAIM_COVERAGE) && c->nvalues) {
            int64_t *k = NULL; TR_CERT_IN(k, c->keys, c->nvalues * 8); dc[j].keys = k;
        }
        if (c->kind == TR_CLAIM_REFINEMENT && c->nvalues) {
            int64_t *t = NULL; TR_CERT_IN(t, c->targets, c->nvalues * 8); dc[j].targets = t;
        }
        if (c->kind == TR_CLAIM_REFERENCE && c->nvalues) {
            uint64_t *u = NULL; TR_CERT_IN(u, c->used, ((c->nvalues + 63) / 64) * 8); dc[j].used = u;
        }
        if (c->kind == TR_CLAIM_COVERAGE) { moff[j] = mtotal; mtotal += G; cover[ic++] = j; }
        if (c->kind == TR_CLAIM_UNIQUE) { foff[j] = ftotal; ftotal += tr_cert_unique_domain(view, c); uniq[iu++] = j; }
        acc[(uint64_t)j * TR_CERT_ACC + 3] = UINT64_MAX;
    }
    TR_CERT_IN(dclaims, dc, (uint64_t)n * sizeof *dc);
    TR_CERT_IN(dacc, acc, (uint64_t)n * TR_CERT_ACC * 8);
    TR_CERT_IN(dmoff, moff, (uint64_t)n * 8);
    TR_CERT_IN(dfoff, foff, (uint64_t)n * 8);
    if (ncover) {
        TR_CERT_DEV(dmasks, mtotal * 8);
        TR_CERT_DEV(dpresent, ((G + 63) / 64) * 8);
        TR_CERT_IN(dcover, cover, (uint64_t)ncover * 4);
    }
    if (nuniq) { TR_CERT_DEV(dfirst, (ftotal ? ftotal : 1) * 8); TR_CERT_IN(duniq, uniq, (uint64_t)nuniq * 4); }
#undef TR_CERT_IN
#undef TR_CERT_DEV
    uint64_t slices = (nw + TR_CERT_SLICE_WORDS - 1) / TR_CERT_SLICE_WORDS;
    if (slices > TR_CERT_MAX_SLICES) slices = TR_CERT_MAX_SLICES;
    if (!slices) slices = 1;
    uint64_t per = (nw + slices - 1) / slices;
    if (!rc && nuniq && ftotal) { TR_LAUNCH(tr_cert_fill_kernel, ftotal, dfirst, ftotal, UINT64_MAX); rc = tr_cuda_sync(); }
    if (!rc && ncover) { TR_LAUNCH(tr_cert_presence_kernel, slices, d, slices, per, dpresent); rc = tr_cuda_sync(); }
    if (!rc) {
        TR_LAUNCH(tr_cert_rows_kernel, (uint64_t)n * slices, d, dclaims, n, slices, per, dacc, dmasks, dmoff, dfirst, dfoff);
        rc = tr_cuda_sync();
    }
    if (!rc && nuniq) {
        TR_LAUNCH(tr_cert_unique_kernel, (uint64_t)nuniq * slices, d, dclaims, duniq, nuniq, slices, per, dacc, dfirst, dfoff);
        rc = tr_cuda_sync();
    }
    if (!rc && ncover) {
        uint64_t gslices = (G + 1023) / 1024;
        if (gslices > TR_CERT_MAX_SLICES) gslices = TR_CERT_MAX_SLICES;
        uint64_t gper = (G + gslices - 1) / gslices;
        TR_LAUNCH(tr_cert_cover_kernel, (uint64_t)ncover * gslices, dclaims, dcover, ncover, G, gslices, gper, dpresent,
                  dmasks, dmoff, dacc);
        rc = tr_cuda_sync();
    }
    if (!rc) rc = tr_cuda_copy_out(acc, dacc, (size_t)n * TR_CERT_ACC * 8);
    for (int i = 0; i < nmem; i++) tr_cuda_free(dmem[i]);
    free(dmem);
    if (!rc)
        for (int32_t j = 0; j < n; j++) {
            const uint64_t *a = acc + (uint64_t)j * TR_CERT_ACC;
            out[j].checked = a[0]; out[j].excluded = a[1]; out[j].violations = a[2]; out[j].witness = a[3];
            out[j].overflow = a[4];
            out[j].holds = !a[4] && !a[2];
        }
    free(dc); free(acc); free(moff); free(foff); free(cover); free(uniq);
    return rc;
}
#ifdef __cplusplus
}
#endif
