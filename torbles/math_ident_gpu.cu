/* DISC-002 on a CUDA device: the Gram moments of a plan (exact 128-bit sums, or the sums of their residues modulo each
 * recorded prime) and the exhaustive verification of many candidate identities over every selected row, with the
 * semantics of tr_ident_gram_build / tr_ident_verify_view (math_ident.h) and the same results bit for bit.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Gram: work items (moment chunk, slot, row slice): a slice is a contiguous run of selection words; the work item
 * evaluates the basis at each of its Gram rows (exactly, or modulo its slot's prime), accumulates the moments of its
 * chunk locally (signed 128-bit sums proven by the plan, or sums of residues < 2^62 each) and adds them with the
 * two-word atomic (tr_atomic_add_wide); the work items of chunk 0 and slot 0 count the Gram rows by the sign of their
 * weight and keep the weight extremes (atomic add, min, max). The host reduces the residue sums modulo their primes
 * when it adopts the states (tr_ident_gram_from_states).
 * Verification: work items (candidate chunk, row slice): each row of the slice is evaluated for every candidate of
 * the chunk (exact W-word evaluation, tr_ident_cand_row), counted locally, then added with one atomic per counter and
 * the smallest violating row by an atomic minimum.
 * No barrier, no shared memory: results do not depend on thread order. The host checks the budget (the device copies:
 * tr_ident_gram_device_need, tr_ident_verify_device_need) and the plan's or candidates' structure before anything is
 * copied. */
#include "math_cuda.h"
#include "math_ident.h"

TR_MGPU_ABI(tr_ident_moment_view, TR_IDENT_ABI);    /* structs passed to the CUDA library (math_handles.h) */
TR_MGPU_ABI(tr_ident_gram_states, TR_IDENT_ABI);
TR_MGPU_ABI(tr_ident_cand_view, TR_IDENT_ABI);

#define TR_IDENT_ACC 8                       /* per candidate: checked, excluded, violations, witness, overflow */

TR_KERNEL void tr_ident_gram_kernel(tr_scope_view v, tr_ident_moment_view m, uint64_t slices, uint64_t per,
                                    uint64_t nchunks, int64_t *acc, uint64_t *cnt, int64_t *ext) {
    const uint64_t nw = (v.rows + 63) / 64, nslots = m.repr == TR_IDENT_MODULAR ? (uint64_t)m.nprimes : 1;
    TR_GRID_LOOP(i, nchunks * nslots * slices) {
        const uint64_t chunk = i / (nslots * slices), slot = (i / slices) % nslots, s = i % slices;
        const uint64_t w0 = s * per, w1 = w0 + per < nw ? w0 + per : nw;
        const int32_t m0 = (int32_t)(chunk * TR_IDENT_CHUNK);
        const int32_t m1 = m0 + TR_IDENT_CHUNK < m.nmoments ? m0 + TR_IDENT_CHUNK : m.nmoments;
        const int counts = chunk == 0 && slot == 0;
        __int128 ex[TR_IDENT_CHUNK];
        unsigned __int128 rs[TR_IDENT_CHUNK];
        __int128 val[TR_IDENT_MAX_BASIS];
        uint64_t mval[TR_IDENT_MAX_BASIS];
        for (int32_t q = 0; q < TR_IDENT_CHUNK; q++) { ex[q] = 0; rs[q] = 0; }
        uint64_t domain = 0, pos = 0, zer = 0, neg = 0;
        int64_t wmin = INT64_MAX, wmax = INT64_MIN;
        const tr_gfp *f = m.repr == TR_IDENT_MODULAR ? &m.primes[slot] : (const tr_gfp *)0;
        for (uint64_t w = w0; w < w1; w++) {
            uint64_t bits = v.selection[w];
            while (bits) {
                uint64_t r = w * 64 + (uint64_t)tr_ipoly_ctz(bits);
                bits &= bits - 1;
                int64_t wt;
                if (!tr_ident_gram_row(&v, &m, r, &wt)) continue;
                if (counts) {
                    domain++;
                    if (wt > 0) pos++; else if (wt < 0) neg++; else zer++;
                    if (wt < wmin) wmin = wt;
                    if (wt > wmax) wmax = wt;
                }
                if (!wt) continue;
                if (m.repr == TR_IDENT_EXACT) {
                    tr_ident_basis_exact(&v, &m, r, val);
                    for (int32_t q = m0; q < m1; q++) ex[q - m0] += (__int128)wt * (val[m.mleft[q]] * val[m.mright[q]]);
                } else {
                    tr_ident_basis_mod(&v, &m, f, r, mval);
                    uint64_t wm = tr_gfp_reduce_i64_hd(f, wt);
                    for (int32_t q = m0; q < m1; q++)
                        rs[q - m0] += tr_gfp_mul_hd(f, wm, tr_gfp_mul_hd(f, mval[m.mleft[q]], mval[m.mright[q]]));
                }
            }
        }
        for (int32_t q = m0; q < m1; q++) {
            uint64_t at = 2 * ((uint64_t)q * nslots + slot);
            if (m.repr == TR_IDENT_EXACT) { if (ex[q - m0]) tr_atomic_add_wide(&acc[at], &acc[at + 1], ex[q - m0]); }
            else if (rs[q - m0]) tr_atomic_add_wide(&acc[at], &acc[at + 1], (__int128)rs[q - m0]);
        }
        if (counts && domain) {
            tr_atomic_add_u64(&cnt[0], domain);
            if (pos) tr_atomic_add_u64(&cnt[1], pos);
            if (zer) tr_atomic_add_u64(&cnt[2], zer);
            if (neg) tr_atomic_add_u64(&cnt[3], neg);
            tr_atomic_min_i64(&ext[0], wmin);
            tr_atomic_max_i64(&ext[1], wmax);
        }
    }
}

TR_KERNEL void tr_ident_verify_kernel(tr_scope_view v, tr_ident_cand_view c, uint64_t slices, uint64_t per,
                                      uint64_t nchunks, uint64_t *acc) {
    const uint64_t nw = (v.rows + 63) / 64;
    TR_GRID_LOOP(i, nchunks * slices) {
        const uint64_t chunk = i / slices, s = i % slices;
        const uint64_t w0 = s * per, w1 = w0 + per < nw ? w0 + per : nw;
        const int32_t j0 = (int32_t)(chunk * TR_IDENT_CAND_CHUNK);
        const int32_t j1 = j0 + TR_IDENT_CAND_CHUNK < c.ncand ? j0 + TR_IDENT_CAND_CHUNK : c.ncand;
        uint64_t checked[TR_IDENT_CAND_CHUNK], excluded[TR_IDENT_CAND_CHUNK], violations[TR_IDENT_CAND_CHUNK];
        uint64_t witness[TR_IDENT_CAND_CHUNK], overflow[TR_IDENT_CAND_CHUNK];
        for (int32_t q = 0; q < TR_IDENT_CAND_CHUNK; q++) {
            checked[q] = 0; excluded[q] = 0; violations[q] = 0; witness[q] = UINT64_MAX; overflow[q] = 0;
        }
        for (uint64_t w = w0; w < w1; w++) {
            uint64_t bits = v.selection[w];
            while (bits) {
                uint64_t r = w * 64 + (uint64_t)tr_ipoly_ctz(bits);
                bits &= bits - 1;
                for (int32_t j = j0; j < j1; j++) {
                    int st = tr_ident_cand_row(&v, &c, j, r);
                    int32_t q = j - j0;
                    if (st == TR_CERT_ROW_EXCLUDED) { excluded[q]++; continue; }
                    if (st == TR_CERT_ROW_OVERFLOW) { overflow[q]++; continue; }
                    checked[q]++;
                    if (st == TR_CERT_ROW_VIOLATES) { violations[q]++; if (r < witness[q]) witness[q] = r; }
                }
            }
        }
        for (int32_t j = j0; j < j1; j++) {
            int32_t q = j - j0;
            uint64_t *a = acc + (uint64_t)j * TR_IDENT_ACC;
            if (checked[q]) tr_atomic_add_u64(&a[0], checked[q]);
            if (excluded[q]) tr_atomic_add_u64(&a[1], excluded[q]);
            if (violations[q]) { tr_atomic_add_u64(&a[2], violations[q]); tr_atomic_min_u64(&a[3], witness[q]); }
            if (overflow[q]) tr_atomic_add_u64(&a[4], overflow[q]);
        }
    }
}

/* the structure the kernels rely on (the full checks are libalg's: tr_ident_plan_make, tr_ident_verify_view) */
static int tr_ident_gpu_view_ok(const tr_scope_view *v) {
    return v && v->ngroups >= 1 && (!v->rows || v->selection) && v->ncols >= 0 && v->ncols <= TR_SCOPE_MAX_COLUMNS;
}
static int tr_ident_gpu_col_ok(const tr_scope_view *v, int32_t c) {
    return c >= 0 && c < v->ncols && (!v->rows || v->col[c].codes || v->col[c].values);
}
static int tr_ident_gpu_plan_ok(const tr_scope_view *v, const tr_ident_moment_view *m) {
    if (!m || m->nvars < 1 || m->nvars > TR_IDENT_MAX_VARS || m->nbasis < 1 || m->nbasis > TR_IDENT_MAX_BASIS ||
        m->nmoments < 1 || m->nmoments > TR_IDENT_MAX_MOMENTS || !m->basis || !m->bparent || !m->bvar || !m->mleft ||
        !m->mright || (m->repr != TR_IDENT_EXACT && m->repr != TR_IDENT_MODULAR) ||
        (m->repr == TR_IDENT_MODULAR && (m->nprimes < 1 || m->nprimes > TR_IDENT_MAX_PRIMES || !m->primes))) return 0;
    for (int32_t i = 0; i < m->nvars; i++) if (!tr_ident_gpu_col_ok(v, m->var[i])) return 0;
    if (m->weight != -1 && !tr_ident_gpu_col_ok(v, m->weight)) return 0;
    for (int32_t b = 0; b < m->nbasis; b++)
        if (m->bparent[b] >= b || m->bparent[b] < -1 || m->bvar[b] < 0 || m->bvar[b] >= m->nvars) return 0;
    for (int32_t q = 0; q < m->nmoments; q++)
        if (m->mleft[q] < 0 || m->mleft[q] >= m->nbasis || m->mright[q] < 0 || m->mright[q] >= m->nbasis) return 0;
    for (int32_t q = 0; m->repr == TR_IDENT_MODULAR && q < m->nprimes; q++)
        if (!tr_gfp_consistent(&m->primes[q])) return 0;
    return 1;
}
static int tr_ident_gpu_cand_ok(const tr_scope_view *v, const tr_ident_cand_view *c) {
    if (!c || c->ncand < 0 || c->nvars < 1 || c->nvars > TR_IDENT_MAX_VARS || (c->ncand && (!c->offset || !c->support || !c->words)) ||
        (c->nterms && (!c->exponents || !c->coefficients))) return 0;
    for (int32_t i = 0; i < c->nvars; i++) if (!tr_ident_gpu_col_ok(v, c->var[i])) return 0;
    if (c->ncand && (c->offset[0] != 0 || c->offset[c->ncand] != c->nterms)) return 0;
    for (int32_t j = 0; j < c->ncand; j++)
        if (c->offset[j] > c->offset[j + 1] || (c->nvars < 32 && (c->support[j] >> c->nvars)) || !c->support[j] ||
            (c->words[j] != 0 && (c->words[j] < 2 || c->words[j] > TR_IDENT_MAX_WORDS))) return 0;
    return 1;
}
/* copies the columns the kernels read (codes or values, validity) and points the view at them */
static int tr_ident_gpu_columns(const tr_scope_view *view, tr_scope_view *d, uint32_t cols, void **mem, int *nmem) {
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
static void tr_ident_gpu_slices(uint64_t rows, uint64_t *slices, uint64_t *per) {
    uint64_t nw = (rows + 63) / 64, s = (nw + TR_IDENT_SLICE_WORDS - 1) / TR_IDENT_SLICE_WORDS;
    if (s > TR_IDENT_MAX_SLICES) s = TR_IDENT_MAX_SLICES;
    if (!s) s = 1;
    *slices = s;
    *per = (nw + s - 1) / s;
    if (!*per) *per = 1;
}

#ifdef __cplusplus
extern "C" {
#endif
int tr_mgpu_ident_gram(const tr_scope_view *view, const tr_ident_moment_view *plan, const tr_math_budget *budget,
                       tr_ident_gram_states *out, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!tr_ident_gpu_view_ok(view) || !out || !tr_ident_gpu_plan_ok(view, plan)) return -1;   /* TR_MATH_INVALID */
    const uint64_t nslots = plan->repr == TR_IDENT_MODULAR ? (uint64_t)plan->nprimes : 1;
    if (out->nmoments != plan->nmoments || (uint64_t)out->nslots != nslots || !out->acc) return -1;
    tr_math_budget dflt;
    if (!budget) { tr_budget_default(&dflt); budget = &dflt; }
    tr_math_need need;
    tr_ident_gram_device_need(view, plan, &need);
    int rc = tr_budget_check(budget, &need, refusal);
    if (rc) return rc;
    const uint64_t words = (uint64_t)plan->nmoments * nslots * 2;
    for (uint64_t i = 0; i < words; i++) out->acc[i] = 0;
    uint64_t cnt[4] = {0, 0, 0, 0};
    int64_t ext[2] = {INT64_MAX, INT64_MIN};
    if (view->selected) {
        void *mem[3 * TR_SCOPE_MAX_COLUMNS + 16];
        int nmem = 0;
        tr_scope_view d = *view;
        tr_ident_moment_view dm = *plan;
        d.groups = NULL;
        uint64_t nw = (view->rows + 63) / 64;
        uint64_t *dsel = NULL, *dcnt = NULL;
        int64_t *dacc = NULL, *dext = NULL;
        uint8_t *dbasis = NULL;
        int32_t *dpar = NULL, *dvar = NULL, *dl = NULL, *dr = NULL;
        tr_gfp *dprimes = NULL;
#define TR_IDENT_DEV(ptr, bytes) do { if (!rc) { rc = tr_cuda_alloc((void **)&(ptr), (size_t)(bytes)); if (!rc) mem[nmem++] = (void *)(ptr); } } while (0)
#define TR_IDENT_IN(dst, src, bytes) do { TR_IDENT_DEV(dst, bytes); if (!rc && (uint64_t)(bytes) != 0) rc = tr_cuda_copy_in(dst, src, (size_t)(bytes)); } while (0)
        TR_IDENT_IN(dsel, view->selection, nw * 8);
        d.selection = dsel;
        uint32_t cols = 0;
        for (int32_t i = 0; i < plan->nvars; i++) cols |= (uint32_t)1 << plan->var[i];
        if (plan->weight >= 0) cols |= (uint32_t)1 << plan->weight;
        if (!rc) rc = tr_ident_gpu_columns(view, &d, cols, mem, &nmem);
        TR_IDENT_IN(dbasis, plan->basis, (uint64_t)plan->nbasis * (uint64_t)plan->nvars);
        TR_IDENT_IN(dpar, plan->bparent, (uint64_t)plan->nbasis * 4);
        TR_IDENT_IN(dvar, plan->bvar, (uint64_t)plan->nbasis * 4);
        TR_IDENT_IN(dl, plan->mleft, (uint64_t)plan->nmoments * 4);
        TR_IDENT_IN(dr, plan->mright, (uint64_t)plan->nmoments * 4);
        if (plan->repr == TR_IDENT_MODULAR) TR_IDENT_IN(dprimes, plan->primes, nslots * sizeof(tr_gfp));
        TR_IDENT_IN(dacc, out->acc, words * 8);
        TR_IDENT_IN(dcnt, cnt, sizeof cnt);
        TR_IDENT_IN(dext, ext, sizeof ext);
#undef TR_IDENT_IN
#undef TR_IDENT_DEV
        dm.basis = dbasis; dm.bparent = dpar; dm.bvar = dvar; dm.mleft = dl; dm.mright = dr; dm.primes = dprimes;
        uint64_t slices, per;
        tr_ident_gpu_slices(view->rows, &slices, &per);
        const uint64_t nchunks = ((uint64_t)plan->nmoments + TR_IDENT_CHUNK - 1) / TR_IDENT_CHUNK;
        if (!rc) { TR_LAUNCH(tr_ident_gram_kernel, nchunks * nslots * slices, d, dm, slices, per, nchunks, dacc, dcnt, dext); rc = tr_cuda_sync(); }
        if (!rc) rc = tr_cuda_copy_out(out->acc, dacc, words * 8);
        if (!rc) rc = tr_cuda_copy_out(cnt, dcnt, sizeof cnt);
        if (!rc) rc = tr_cuda_copy_out(ext, dext, sizeof ext);
        for (int i = 0; i < nmem; i++) tr_cuda_free(mem[i]);
        if (rc) return rc;
    }
    out->domain = cnt[0]; out->positive = cnt[1]; out->zero = cnt[2]; out->negative = cnt[3];
    if (!cnt[0]) { out->wmin = plan->weight >= 0 ? 0 : 1; out->wmax = out->wmin; }
    else if (plan->weight < 0) { out->wmin = 1; out->wmax = 1; }
    else { out->wmin = ext[0]; out->wmax = ext[1]; }
    return 0;
}

int tr_mgpu_ident_verify(const tr_scope_view *view, const tr_ident_cand_view *cand, const tr_math_budget *budget,
                         tr_cert_evidence *out, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!tr_ident_gpu_view_ok(view) || !tr_ident_gpu_cand_ok(view, cand) || (cand->ncand && !out)) return -1;
    tr_math_budget dflt;
    if (!budget) { tr_budget_default(&dflt); budget = &dflt; }
    tr_math_need need;
    tr_ident_verify_device_need(view, cand, &need);
    int rc = tr_budget_check(budget, &need, refusal);
    if (rc) return rc;
    const int32_t n = cand->ncand;
    for (int32_t j = 0; j < n; j++) {
        memset(&out[j], 0, sizeof out[j]);
        out[j].rows = view->rows; out[j].selected = view->selected; out[j].witness = TR_CERT_NONE;
        out[j].unit = TR_CERT_UNIT_ROWS; out[j].holds = 1;
    }
    if (!n || !view->selected) return 0;
    uint64_t *acc = (uint64_t *)calloc((size_t)n * TR_IDENT_ACC, 8);
    if (!acc) return -2;                                                        /* TR_MATH_MEMORY */
    for (int32_t j = 0; j < n; j++) acc[(uint64_t)j * TR_IDENT_ACC + 3] = UINT64_MAX;
    void *mem[3 * TR_SCOPE_MAX_COLUMNS + 16];
    int nmem = 0;
    tr_scope_view d = *view;
    tr_ident_cand_view dc = *cand;
    d.groups = NULL;
    uint64_t nw = (view->rows + 63) / 64;
    uint64_t *dsel = NULL, *doff = NULL, *dacc = NULL;
    uint32_t *dsup = NULL;
    int32_t *dwords = NULL;
    uint8_t *dexp = NULL;
    int64_t *dcoef = NULL;
#define TR_IDENT_DEV(ptr, bytes) do { if (!rc) { rc = tr_cuda_alloc((void **)&(ptr), (size_t)(bytes)); if (!rc) mem[nmem++] = (void *)(ptr); } } while (0)
#define TR_IDENT_IN(dst, src, bytes) do { TR_IDENT_DEV(dst, bytes); if (!rc && (uint64_t)(bytes) != 0) rc = tr_cuda_copy_in(dst, src, (size_t)(bytes)); } while (0)
    TR_IDENT_IN(dsel, view->selection, nw * 8);
    d.selection = dsel;
    uint32_t cols = 0;
    for (int32_t i = 0; i < cand->nvars; i++) cols |= (uint32_t)1 << cand->var[i];
    if (!rc) rc = tr_ident_gpu_columns(view, &d, cols, mem, &nmem);
    TR_IDENT_IN(doff, cand->offset, ((uint64_t)n + 1) * 8);
    TR_IDENT_IN(dsup, cand->support, (uint64_t)n * 4);
    TR_IDENT_IN(dwords, cand->words, (uint64_t)n * 4);
    TR_IDENT_IN(dexp, cand->exponents, cand->nterms * (uint64_t)cand->nvars);
    TR_IDENT_IN(dcoef, cand->coefficients, cand->nterms * 8);
    TR_IDENT_IN(dacc, acc, (uint64_t)n * TR_IDENT_ACC * 8);
#undef TR_IDENT_IN
#undef TR_IDENT_DEV
    dc.offset = doff; dc.support = dsup; dc.words = dwords; dc.exponents = dexp; dc.coefficients = dcoef;
    uint64_t slices, per;
    tr_ident_gpu_slices(view->rows, &slices, &per);
    const uint64_t nchunks = ((uint64_t)n + TR_IDENT_CAND_CHUNK - 1) / TR_IDENT_CAND_CHUNK;
    if (!rc) { TR_LAUNCH(tr_ident_verify_kernel, nchunks * slices, d, dc, slices, per, nchunks, dacc); rc = tr_cuda_sync(); }
    if (!rc) rc = tr_cuda_copy_out(acc, dacc, (size_t)n * TR_IDENT_ACC * 8);
    for (int i = 0; i < nmem; i++) tr_cuda_free(mem[i]);
    if (!rc)
        for (int32_t j = 0; j < n; j++) {
            const uint64_t *a = acc + (uint64_t)j * TR_IDENT_ACC;
            out[j].checked = a[0]; out[j].excluded = a[1]; out[j].violations = a[2]; out[j].witness = a[3];
            out[j].overflow = a[4];
            out[j].holds = !a[4] && !a[2];
        }
    free(acc);
    return rc;
}
#ifdef __cplusplus
}
#endif
