/* GEOM-001 on a CUDA device: the agreement of pairs of representations on every selected row and the batched exact
 * check of reduction certificates, with the semantics of tr_geom_agree_view / tr_geom_checks_run (math_geom.h) and the
 * same results bit for bit.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Agreement: work items (pair chunk, row slice): each row of the slice evaluates P_j = D (f_j - g_j) exactly (DISC-002's
 * W-word evaluation with proven bounds, tr_ident_cand_row) for every pair of the chunk and, on a checked row, the bit
 * lengths of F_j = D_f f_j and G_j = D_g g_j; counted locally, then one atomic add per counter, an atomic minimum for
 * the smallest disagreeing row and atomic maxima for the bit lengths.
 * Checks: one work item per product (every term of P, every pair of terms q_a g_b of each cofactor and premise,
 * tr_geom_work) adds its signed 128-bit value into its certificate's dense slot with the two-word atomic
 * (tr_atomic_add_wide; the host proved every partial sum stays within 2^127 - 1, so the two-word sum is exact whatever
 * the order); a second pass counts the nonzero slots per certificate and keeps the smallest (atomic add, minimum).
 * A work item whose rank falls outside its certificate's slots (a malformed view) raises a flag: TR_MATH_INVALID.
 * No barrier, no shared memory: results do not depend on thread order. The host checks the budget (the device copies:
 * tr_geom_agree_device_need, tr_geom_check_device_need) and the views' structure before anything is copied. */
#include "math_cuda.h"
#include "math_geom.h"

TR_MGPU_ABI(tr_geom_pair_view, TR_GEOM_ABI);       /* structs passed to the CUDA library (math_handles.h) */
TR_MGPU_ABI(tr_geom_agreement, TR_GEOM_ABI);
TR_MGPU_ABI(tr_geom_check_view, TR_GEOM_ABI);
TR_MGPU_ABI(tr_geom_check_result, TR_GEOM_ABI);

#define TR_GEOM_PAIR_CHUNK 8                 /* pairs per work item */
#define TR_GEOM_ACC 8                        /* per pair: checked, excluded, disagreements, witness, undecided, bits */

TR_KERNEL void tr_geom_agree_kernel(tr_scope_view v, tr_ident_cand_view c, int32_t npairs, uint64_t slices, uint64_t per,
                                    uint64_t nchunks, uint64_t *acc) {
    const uint64_t nw = (v.rows + 63) / 64;
    TR_GRID_LOOP(i, nchunks * slices) {
        const uint64_t chunk = i / slices, s = i % slices;
        const uint64_t w0 = s * per, w1 = w0 + per < nw ? w0 + per : nw;
        const int32_t j0 = (int32_t)(chunk * TR_GEOM_PAIR_CHUNK);
        const int32_t j1 = j0 + TR_GEOM_PAIR_CHUNK < npairs ? j0 + TR_GEOM_PAIR_CHUNK : npairs;
        uint64_t checked[TR_GEOM_PAIR_CHUNK], excluded[TR_GEOM_PAIR_CHUNK], dis[TR_GEOM_PAIR_CHUNK];
        uint64_t witness[TR_GEOM_PAIR_CHUNK], undecided[TR_GEOM_PAIR_CHUNK], bf[TR_GEOM_PAIR_CHUNK], bg[TR_GEOM_PAIR_CHUNK];
        for (int32_t q = 0; q < TR_GEOM_PAIR_CHUNK; q++) {
            checked[q] = 0; excluded[q] = 0; dis[q] = 0; witness[q] = UINT64_MAX; undecided[q] = 0; bf[q] = 0; bg[q] = 0;
        }
        uint64_t val[TR_IDENT_MAX_WORDS];
        for (uint64_t w = w0; w < w1; w++) {
            uint64_t bits = v.selection[w];
            while (bits) {
                uint64_t r = w * 64 + (uint64_t)tr_ipoly_ctz(bits);
                bits &= bits - 1;
                for (int32_t j = j0; j < j1; j++) {
                    const int32_t q = j - j0;
                    int st = tr_ident_cand_row(&v, &c, 3 * j + 2, r);
                    if (st == TR_CERT_ROW_EXCLUDED) { excluded[q]++; continue; }
                    if (st == TR_CERT_ROW_OVERFLOW) { undecided[q]++; continue; }
                    checked[q]++;
                    if (st == TR_CERT_ROW_VIOLATES) { dis[q]++; if (r < witness[q]) witness[q] = r; }
                    tr_ident_cand_value(&v, &c, 3 * j, r, c.words[3 * j], val);
                    uint32_t b = tr_geom_bits(val, c.words[3 * j]);
                    if (b > bf[q]) bf[q] = b;
                    tr_ident_cand_value(&v, &c, 3 * j + 1, r, c.words[3 * j + 1], val);
                    b = tr_geom_bits(val, c.words[3 * j + 1]);
                    if (b > bg[q]) bg[q] = b;
                }
            }
        }
        for (int32_t j = j0; j < j1; j++) {
            const int32_t q = j - j0;
            uint64_t *a = acc + (uint64_t)j * TR_GEOM_ACC;
            if (checked[q]) tr_atomic_add_u64(&a[0], checked[q]);
            if (excluded[q]) tr_atomic_add_u64(&a[1], excluded[q]);
            if (dis[q]) { tr_atomic_add_u64(&a[2], dis[q]); tr_atomic_min_u64(&a[3], witness[q]); }
            if (undecided[q]) tr_atomic_add_u64(&a[4], undecided[q]);
            if (bf[q]) tr_atomic_max_u64(&a[5], bf[q]);
            if (bg[q]) tr_atomic_max_u64(&a[6], bg[q]);
        }
    }
}

/* the certificate owning a flattened index (bounds[0..n], nondecreasing; bounds[c] <= x < bounds[c + 1]) */
TR_HD int32_t tr_geom_owner(const uint64_t *bounds, int32_t n, uint64_t x) {
    int32_t lo = 0, hi = n - 1;
    while (lo < hi) {
        int32_t mid = lo + (hi - lo + 1) / 2;
        if (bounds[mid] <= x) lo = mid; else hi = mid - 1;
    }
    return lo;
}

TR_KERNEL void tr_geom_expand_kernel(tr_geom_check_view k, int64_t *slots, uint64_t *bad) {
    TR_GRID_LOOP(i, k.nwork) {
        const int32_t c = tr_geom_owner(k.work0, k.ncert, i);
        uint64_t rank;
        __int128 value;
        tr_geom_work(&k, c, i - k.work0[c], &rank, &value);
        const uint64_t base = k.slot0[c], n = k.slot0[c + 1] - base;
        if (rank >= n) { tr_atomic_or_u64(bad, 1); continue; }
        if (value) tr_atomic_add_wide(&slots[2 * (base + rank)], &slots[2 * (base + rank) + 1], value);
    }
}

TR_KERNEL void tr_geom_scan_kernel(tr_geom_check_view k, const int64_t *slots, uint64_t *acc) {
    TR_GRID_LOOP(i, k.nslots) {
        if (!slots[2 * i] && !slots[2 * i + 1]) continue;
        const int32_t c = tr_geom_owner(k.slot0, k.ncert, i);
        tr_atomic_add_u64(&acc[2 * (uint64_t)c], 1);
        tr_atomic_min_u64(&acc[2 * (uint64_t)c + 1], i - k.slot0[c]);
    }
}

/* the structure the kernels rely on (the full checks are libalg's: tr_geom_pairs_make, tr_geom_checks_make) */
static int tr_geom_gpu_view_ok(const tr_scope_view *v) {
    return v && v->ngroups >= 1 && (!v->rows || v->selection) && v->ncols >= 0 && v->ncols <= TR_SCOPE_MAX_COLUMNS;
}
static int tr_geom_gpu_pairs_ok(const tr_scope_view *v, const tr_geom_pair_view *p) {
    const tr_ident_cand_view *c = p ? &p->cand : (const tr_ident_cand_view *)0;
    if (!p || p->npairs < 0 || p->npairs > TR_GEOM_MAX_PAIRS || c->ncand != 3 * p->npairs || c->nvars < 1 ||
        c->nvars > TR_GEOM_MAX_VARS) return 0;
    if (p->npairs && (!c->offset || !c->support || !c->words || (c->nterms && (!c->exponents || !c->coefficients)))) return 0;
    for (int32_t i = 0; i < c->nvars; i++) {
        int32_t k = c->var[i];
        if (k < 0 || k >= v->ncols || (v->rows && !v->col[k].codes && !v->col[k].values)) return 0;
    }
    if (p->npairs && (c->offset[0] || c->offset[c->ncand] != c->nterms)) return 0;
    for (int32_t j = 0; j < c->ncand; j++)
        if (c->offset[j] > c->offset[j + 1] || !c->support[j] || (c->nvars < 32 && (c->support[j] >> c->nvars)) ||
            c->words[j] < 2 || c->words[j] > TR_IDENT_MAX_WORDS) return 0;
    return 1;
}
static uint64_t tr_geom_gpu_work_of(const tr_geom_check_view *k, int32_t c) {
    const uint64_t p0 = k->poly0[c];
    uint64_t w = k->term0[p0 + 1] - k->term0[p0];
    for (int32_t i = 0; i < k->npremises[c]; i++) {
        uint64_t q = p0 + 1 + 2 * (uint64_t)i;
        w += (k->term0[q + 1] - k->term0[q]) * (k->term0[q + 2] - k->term0[q + 1]);
    }
    return w;
}
static int tr_geom_gpu_checks_ok(const tr_geom_check_view *k) {
    if (!k || k->ncert < 0 || k->ncert > TR_GEOM_MAX_CHECKS) return 0;
    if (!k->ncert) return 1;
    if (!k->nvars || !k->npremises || !k->maxdeg || !k->overflow || !k->poly0 || !k->slot0 || !k->work0 || !k->term0 ||
        (k->nterms && (!k->exponents || !k->coefficients))) return 0;
    uint64_t polys = 0;
    for (int32_t c = 0; c < k->ncert; c++) {
        if (k->nvars[c] < 1 || k->nvars[c] > TR_GEOM_MAX_VARS || k->npremises[c] < 1 ||
            k->npremises[c] > TR_CERT_MAX_PREMISES || k->maxdeg[c] < 0 || k->maxdeg[c] > 2040 || k->poly0[c] != polys) return 0;
        polys += 1 + 2 * (uint64_t)k->npremises[c];
    }
    if (polys != k->npolys || k->term0[0] != 0 || k->term0[k->npolys] != k->nterms) return 0;
    for (uint64_t p = 0; p < k->npolys; p++) if (k->term0[p] > k->term0[p + 1]) return 0;
    if (k->slot0[0] || k->work0[0]) return 0;
    for (int32_t c = 0; c < k->ncert; c++) {
        uint64_t ns = k->overflow[c] ? 0 : tr_geom_slots(k->nvars[c], (uint64_t)k->maxdeg[c]);
        uint64_t nw = k->overflow[c] ? 0 : tr_geom_gpu_work_of(k, c);
        if (ns > TR_GEOM_MAX_SLOTS || k->slot0[c + 1] - k->slot0[c] != ns || k->work0[c + 1] - k->work0[c] != nw ||
            k->slot0[c + 1] < k->slot0[c] || k->work0[c + 1] < k->work0[c]) return 0;
    }
    return k->slot0[k->ncert] == k->nslots && k->work0[k->ncert] == k->nwork;
}
/* copies the columns the kernels read (codes or values, validity) and points the view at them */
static int tr_geom_gpu_columns(const tr_scope_view *view, tr_scope_view *d, uint32_t cols, void **mem, int *nmem) {
    int rc = 0;
    const uint64_t rows = view->rows;
    for (int32_t c = 0; c < view->ncols; c++) {
        tr_scope_view_column *k = &d->col[c];
        if (!((cols >> c) & 1)) { k->codes = NULL; k->values = NULL; k->valid = NULL; continue; }
        if (view->col[c].values) {
            int64_t *p = NULL;
            rc = tr_cuda_alloc((void **)&p, (size_t)rows * 8);
            if (rc) return rc;
            mem[(*nmem)++] = p;
            rc = tr_cuda_copy_in(p, view->col[c].values, (size_t)rows * 8);
            k->values = p;
        } else {
            uint32_t *p = NULL;
            rc = tr_cuda_alloc((void **)&p, (size_t)rows * 4);
            if (rc) return rc;
            mem[(*nmem)++] = p;
            rc = tr_cuda_copy_in(p, view->col[c].codes, (size_t)rows * 4);
            k->codes = p;
        }
        if (!rc && view->col[c].valid) {
            uint8_t *p = NULL;
            rc = tr_cuda_alloc((void **)&p, (size_t)rows);
            if (rc) return rc;
            mem[(*nmem)++] = p;
            rc = tr_cuda_copy_in(p, view->col[c].valid, (size_t)rows);
            k->valid = p;
        }
        if (rc) return rc;
    }
    return 0;
}

#ifdef __cplusplus
extern "C" {
#endif
int tr_mgpu_geom_agree(const tr_scope_view *view, const tr_geom_pair_view *pairs, const tr_math_budget *budget,
                       tr_geom_agreement *out, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!tr_geom_gpu_view_ok(view) || !tr_geom_gpu_pairs_ok(view, pairs) || (pairs->npairs && !out)) return -1;   /* INVALID */
    tr_math_budget dflt;
    if (!budget) { tr_budget_default(&dflt); budget = &dflt; }
    tr_math_need need;
    tr_geom_agree_device_need(view, pairs, &need);
    int rc = tr_budget_check(budget, &need, refusal);
    if (rc) return rc;
    const int32_t n = pairs->npairs;
    for (int32_t j = 0; j < n; j++) {
        memset(&out[j], 0, sizeof out[j]);
        out[j].rows = view->rows; out[j].selected = view->selected; out[j].witness = UINT64_MAX;
    }
    if (!n || !view->selected) return 0;
    uint64_t *acc = (uint64_t *)calloc((size_t)n * TR_GEOM_ACC, 8);
    if (!acc) return -2;                                                        /* TR_MATH_MEMORY */
    for (int32_t j = 0; j < n; j++) acc[(uint64_t)j * TR_GEOM_ACC + 3] = UINT64_MAX;
    const tr_ident_cand_view *cand = &pairs->cand;
    void *mem[3 * TR_SCOPE_MAX_COLUMNS + 16];
    int nmem = 0;
    tr_scope_view d = *view;
    tr_ident_cand_view dc = *cand;
    d.groups = NULL;
    const uint64_t nw = (view->rows + 63) / 64, ncand = (uint64_t)cand->ncand;
    uint64_t *dsel = NULL, *doff = NULL, *dacc = NULL;
    uint32_t *dsup = NULL;
    int32_t *dwords = NULL;
    uint8_t *dexp = NULL;
    int64_t *dcoef = NULL;
#define TR_GEOM_DEV(ptr, bytes) do { if (!rc) { rc = tr_cuda_alloc((void **)&(ptr), (size_t)(bytes)); if (!rc) mem[nmem++] = (void *)(ptr); } } while (0)
#define TR_GEOM_IN(dst, src, bytes) do { TR_GEOM_DEV(dst, bytes); if (!rc && (uint64_t)(bytes) != 0) rc = tr_cuda_copy_in(dst, src, (size_t)(bytes)); } while (0)
    TR_GEOM_IN(dsel, view->selection, nw * 8);
    d.selection = dsel;
    uint32_t cols = 0;
    for (int32_t i = 0; i < cand->nvars; i++) cols |= (uint32_t)1 << cand->var[i];
    if (!rc) rc = tr_geom_gpu_columns(view, &d, cols, mem, &nmem);
    TR_GEOM_IN(doff, cand->offset, (ncand + 1) * 8);
    TR_GEOM_IN(dsup, cand->support, ncand * 4);
    TR_GEOM_IN(dwords, cand->words, ncand * 4);
    TR_GEOM_IN(dexp, cand->exponents, cand->nterms * (uint64_t)cand->nvars);
    TR_GEOM_IN(dcoef, cand->coefficients, cand->nterms * 8);
    TR_GEOM_IN(dacc, acc, (uint64_t)n * TR_GEOM_ACC * 8);
    dc.offset = doff; dc.support = dsup; dc.words = dwords; dc.exponents = dexp; dc.coefficients = dcoef;
    uint64_t slices = (nw + TR_GEOM_SLICE_WORDS - 1) / TR_GEOM_SLICE_WORDS, per;
    if (slices > TR_GEOM_MAX_SLICES) slices = TR_GEOM_MAX_SLICES;
    if (!slices) slices = 1;
    per = (nw + slices - 1) / slices;
    if (!per) per = 1;
    const uint64_t nchunks = ((uint64_t)n + TR_GEOM_PAIR_CHUNK - 1) / TR_GEOM_PAIR_CHUNK;
    if (!rc) { TR_LAUNCH(tr_geom_agree_kernel, nchunks * slices, d, dc, n, slices, per, nchunks, dacc); rc = tr_cuda_sync(); }
    if (!rc) rc = tr_cuda_copy_out(acc, dacc, (size_t)n * TR_GEOM_ACC * 8);
    for (int i = 0; i < nmem; i++) tr_cuda_free(mem[i]);
    if (!rc)
        for (int32_t j = 0; j < n; j++) {
            const uint64_t *a = acc + (uint64_t)j * TR_GEOM_ACC;
            out[j].checked = a[0]; out[j].excluded = a[1]; out[j].disagreements = a[2]; out[j].witness = a[3];
            out[j].undecided = a[4]; out[j].bits_f = (uint32_t)a[5]; out[j].bits_g = (uint32_t)a[6];
        }
    free(acc);
    return rc;
}

int tr_mgpu_geom_check(const tr_geom_check_view *view, const tr_math_budget *budget, tr_geom_check_result *out,
                       tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!tr_geom_gpu_checks_ok(view) || (view->ncert && !out)) return -1;                     /* TR_MATH_INVALID */
    tr_math_budget dflt;
    if (!budget) { tr_budget_default(&dflt); budget = &dflt; }
    tr_math_need need;
    tr_geom_check_device_need(view, &need);
    int rc = tr_budget_check(budget, &need, refusal);
    if (rc) return rc;
    const int32_t n = view->ncert;
    for (int32_t c = 0; c < n; c++) {
        memset(&out[c], 0, sizeof out[c]);
        out[c].slots = view->slot0[c + 1] - view->slot0[c]; out[c].work = view->work0[c + 1] - view->work0[c];
        out[c].first = UINT64_MAX; out[c].overflow = view->overflow[c];
    }
    if (!n) return 0;
    uint64_t *acc = (uint64_t *)calloc((size_t)n * 2, 8);
    if (!acc) return -2;                                                        /* TR_MATH_MEMORY */
    for (int32_t c = 0; c < n; c++) acc[2 * (uint64_t)c + 1] = UINT64_MAX;
    void *mem[16];
    int nmem = 0;
    tr_geom_check_view dk = *view;
    int32_t *dnv = NULL, *dnp = NULL, *dmd = NULL, *dov = NULL;
    uint64_t *dp0 = NULL, *ds0 = NULL, *dw0 = NULL, *dt0 = NULL, *dacc = NULL, *dbad = NULL;
    uint8_t *dexp = NULL;
    int64_t *dcoef = NULL, *dslots = NULL;
    uint64_t bad = 0;
    TR_GEOM_IN(dnv, view->nvars, (uint64_t)n * 4);
    TR_GEOM_IN(dnp, view->npremises, (uint64_t)n * 4);
    TR_GEOM_IN(dmd, view->maxdeg, (uint64_t)n * 4);
    TR_GEOM_IN(dov, view->overflow, (uint64_t)n * 4);
    TR_GEOM_IN(dp0, view->poly0, (uint64_t)n * 8);
    TR_GEOM_IN(ds0, view->slot0, ((uint64_t)n + 1) * 8);
    TR_GEOM_IN(dw0, view->work0, ((uint64_t)n + 1) * 8);
    TR_GEOM_IN(dt0, view->term0, (view->npolys + 1) * 8);
    TR_GEOM_IN(dexp, view->exponents, view->nterms * TR_GEOM_MAX_VARS);
    TR_GEOM_IN(dcoef, view->coefficients, view->nterms * 8);
    TR_GEOM_IN(dacc, acc, (uint64_t)n * 16);
    TR_GEOM_IN(dbad, &bad, 8);
    TR_GEOM_DEV(dslots, view->nslots * 16);
    if (!rc && view->nslots) rc = tr_cuda_zero(dslots, (size_t)(view->nslots * 16));
#undef TR_GEOM_IN
#undef TR_GEOM_DEV
    dk.nvars = dnv; dk.npremises = dnp; dk.maxdeg = dmd; dk.overflow = dov; dk.poly0 = dp0; dk.slot0 = ds0;
    dk.work0 = dw0; dk.term0 = dt0; dk.exponents = dexp; dk.coefficients = dcoef;
    if (!rc && view->nwork) { TR_LAUNCH(tr_geom_expand_kernel, view->nwork, dk, dslots, dbad); rc = tr_cuda_sync(); }
    if (!rc && view->nslots) { TR_LAUNCH(tr_geom_scan_kernel, view->nslots, dk, dslots, dacc); rc = tr_cuda_sync(); }
    if (!rc) rc = tr_cuda_copy_out(acc, dacc, (size_t)n * 16);
    if (!rc) rc = tr_cuda_copy_out(&bad, dbad, 8);
    for (int i = 0; i < nmem; i++) tr_cuda_free(mem[i]);
    if (!rc && bad) rc = -1;                                                    /* a rank outside its slots */
    if (!rc)
        for (int32_t c = 0; c < n; c++) {
            if (out[c].overflow) continue;
            out[c].nonzero = acc[2 * (uint64_t)c]; out[c].first = acc[2 * (uint64_t)c + 1];
            out[c].holds = out[c].nonzero == 0;
        }
    free(acc);
    return rc;
}
#ifdef __cplusplus
}
#endif
