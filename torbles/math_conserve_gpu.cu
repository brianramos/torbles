/* DISC-003 on a CUDA device: the incremental step of a tracker over the rows of one DELTA-001 delta, with the
 * semantics of tr_cons_job_run (math_conserve.h) and the same results bit for bit.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Gram increments: work items (moment chunk, row slice) evaluate the plan's basis exactly at each delta row of their
 * slice where every plan variable is present (DISC-002's inline tr_ident_basis_exact), add w * m_l m_r for the moments
 * of their chunk into local signed 128-bit sums (the host proved |G_u| + sum |w| B_u <= 2^127 - 1, so no partial sum
 * leaves 128 bits) and add each with one two-word atomic (tr_atomic_add_wide); the items of chunk 0 also add the
 * weights of the Gram rows by sign.
 * Incremental verification: work items (identity chunk, row slice) evaluate every identity of the chunk at each row of
 * the slice with DISC-002's exact W-word evaluation (tr_cons_eval_row -> tr_ident_cand_row; W proven from the delta's
 * extremes by the host), keep local counters of |weight| (checked, excluded, violating, undecided; inserted and deleted
 * apart) and the smallest violating inserted and deleted rows, then add one atomic per nonzero counter and take the
 * witnesses with atomic minima.
 * No barrier, no shared memory: the results do not depend on thread order. The host validates the job's views and
 * checks the budget of the device copies (tr_cons_delta_device_need) before anything is copied. */
#include "math_cuda.h"
#include "math_conserve.h"

TR_MGPU_ABI(tr_cons_job_view, TR_CONS_ABI);         /* structs passed to the CUDA library (math_handles.h) */
TR_MGPU_ABI(tr_cons_delta_out, TR_CONS_ABI);
TR_MGPU_ABI(tr_cons_delta_evidence, TR_CONS_ABI);

#define TR_CONS_EV 10                       /* per identity: the ten words of tr_cons_delta_evidence */

TR_KERNEL void tr_cons_gram_kernel(tr_scope_view v, tr_ident_moment_view m, const int64_t *weights, uint64_t slices,
                                   uint64_t per, uint64_t nchunks, int64_t *acc, uint64_t *cnt) {
    const uint64_t rows = v.rows;
    TR_GRID_LOOP(i, nchunks * slices) {
        const uint64_t chunk = i / slices, s = i % slices;
        const uint64_t r0 = s * per, r1 = r0 + per < rows ? r0 + per : rows;
        const int32_t m0 = (int32_t)(chunk * TR_CONS_CHUNK);
        const int32_t m1 = m0 + TR_CONS_CHUNK < m.nmoments ? m0 + TR_CONS_CHUNK : m.nmoments;
        const int counts = chunk == 0;
        __int128 ex[TR_CONS_CHUNK];
        __int128 val[TR_IDENT_MAX_BASIS];
        for (int32_t q = 0; q < TR_CONS_CHUNK; q++) ex[q] = 0;
        uint64_t gi = 0, gd = 0;
        for (uint64_t r = r0; r < r1; r++) {
            const int64_t w = weights[r];
            if (!w || !tr_cons_gram_row(&v, &m, r)) continue;
            if (counts) { if (w > 0) gi += (uint64_t)w; else gd += tr_abs_u64(w); }
            tr_ident_basis_exact(&v, &m, r, val);
            for (int32_t q = m0; q < m1; q++) ex[q - m0] += (__int128)w * (val[m.mleft[q]] * val[m.mright[q]]);
        }
        for (int32_t q = m0; q < m1; q++)
            if (ex[q - m0]) tr_atomic_add_wide(&acc[2 * (uint64_t)q], &acc[2 * (uint64_t)q + 1], ex[q - m0]);
        if (counts) {
            if (gi) tr_atomic_add_u64(&cnt[0], gi);
            if (gd) tr_atomic_add_u64(&cnt[1], gd);
        }
    }
}

TR_KERNEL void tr_cons_verify_kernel(tr_scope_view v, tr_ident_cand_view c, const int64_t *weights, uint64_t slices,
                                     uint64_t per, uint64_t nchunks, uint64_t *ev) {
    const uint64_t rows = v.rows;
    TR_GRID_LOOP(i, nchunks * slices) {
        const uint64_t chunk = i / slices, s = i % slices;
        const uint64_t r0 = s * per, r1 = r0 + per < rows ? r0 + per : rows;
        const int32_t j0 = (int32_t)(chunk * TR_CONS_CAND_CHUNK);
        const int32_t j1 = j0 + TR_CONS_CAND_CHUNK < c.ncand ? j0 + TR_CONS_CAND_CHUNK : c.ncand;
        tr_cons_delta_evidence loc[TR_CONS_CAND_CHUNK];
        for (int32_t q = 0; q < TR_CONS_CAND_CHUNK; q++) tr_cons_evidence_init(&loc[q]);
        for (uint64_t r = r0; r < r1; r++) {
            const int64_t w = weights[r];
            if (!w) continue;
            for (int32_t j = j0; j < j1; j++) tr_cons_eval_row(&v, &c, j, r, w, &loc[j - j0]);
        }
        for (int32_t j = j0; j < j1; j++) {
            const tr_cons_delta_evidence *e = &loc[j - j0];
            uint64_t *a = ev + (uint64_t)j * TR_CONS_EV;
            if (e->checked_ins) tr_atomic_add_u64(&a[0], e->checked_ins);
            if (e->checked_del) tr_atomic_add_u64(&a[1], e->checked_del);
            if (e->excluded_ins) tr_atomic_add_u64(&a[2], e->excluded_ins);
            if (e->excluded_del) tr_atomic_add_u64(&a[3], e->excluded_del);
            if (e->violations_ins) tr_atomic_add_u64(&a[4], e->violations_ins);
            if (e->violations_del) tr_atomic_add_u64(&a[5], e->violations_del);
            if (e->overflow_ins) tr_atomic_add_u64(&a[6], e->overflow_ins);
            if (e->overflow_del) tr_atomic_add_u64(&a[7], e->overflow_del);
            if (e->witness_ins != TR_CONS_NONE) tr_atomic_min_u64(&a[8], e->witness_ins);
            if (e->witness_del != TR_CONS_NONE) tr_atomic_min_u64(&a[9], e->witness_del);
        }
    }
}

/* the structure the kernels rely on (the full checks are libalg's: tr_cons_job_make) */
static int tr_cons_gpu_col_ok(const tr_scope_view *v, int32_t c) {
    return c >= 0 && c < v->ncols && (!v->rows || v->col[c].codes || v->col[c].values);
}
static int tr_cons_gpu_ok(const tr_cons_job_view *j) {
    const tr_scope_view *v = &j->rows;
    const tr_ident_moment_view *m = &j->plan;
    const tr_ident_cand_view *c = &j->cand;
    if (v->ncols < 0 || v->ncols > TR_SCOPE_MAX_COLUMNS || (v->rows && !j->weights)) return 0;
    if (m->nvars < 1 || m->nvars > TR_IDENT_MAX_VARS || m->nbasis < 1 || m->nbasis > TR_IDENT_MAX_BASIS ||
        m->nmoments < 1 || m->nmoments > TR_IDENT_MAX_MOMENTS || !m->basis || !m->bparent || !m->bvar || !m->mleft ||
        !m->mright || m->repr != TR_IDENT_EXACT || m->weight != -1) return 0;
    for (int32_t i = 0; i < m->nvars; i++) if (!tr_cons_gpu_col_ok(v, m->var[i])) return 0;
    for (int32_t b = 0; b < m->nbasis; b++)
        if (m->bparent[b] >= b || m->bparent[b] < -1 || m->bvar[b] < 0 || m->bvar[b] >= m->nvars) return 0;
    for (int32_t q = 0; q < m->nmoments; q++)
        if (m->mleft[q] < 0 || m->mleft[q] >= m->nbasis || m->mright[q] < 0 || m->mright[q] >= m->nbasis) return 0;
    if (c->ncand < 0 || c->ncand > TR_CONS_MAX_IDENTITIES || c->nvars < 1 || c->nvars > TR_IDENT_MAX_VARS ||
        (c->ncand && (!c->offset || !c->support || !c->words)) || (c->nterms && (!c->exponents || !c->coefficients)))
        return 0;
    for (int32_t i = 0; i < c->nvars; i++) if (!tr_cons_gpu_col_ok(v, c->var[i])) return 0;
    if (c->ncand && (c->offset[0] != 0 || c->offset[c->ncand] != c->nterms)) return 0;
    for (int32_t j = 0; j < c->ncand; j++)
        if (c->offset[j] > c->offset[j + 1] || (c->nvars < 32 && (c->support[j] >> c->nvars)) || !c->support[j] ||
            (c->words[j] != 0 && (c->words[j] < 2 || c->words[j] > TR_IDENT_MAX_WORDS))) return 0;
    return 1;
}
/* copies the delta columns the kernels read (values, validity) and points the view at them */
static int tr_cons_gpu_columns(const tr_scope_view *view, tr_scope_view *d, uint32_t cols, void **mem, int *nmem) {
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
            k->values = p; k->codes = NULL;
        } else {
            uint32_t *p = NULL;
            rc = tr_cuda_alloc((void **)&p, (size_t)rows * 4);
            if (rc) return rc;
            mem[(*nmem)++] = p;
            rc = tr_cuda_copy_in(p, view->col[c].codes, (size_t)rows * 4);
            k->codes = p; k->values = NULL;
        }
        if (!rc && view->col[c].valid) {
            uint8_t *p = NULL;
            rc = tr_cuda_alloc((void **)&p, (size_t)rows);
            if (rc) return rc;
            mem[(*nmem)++] = p;
            rc = tr_cuda_copy_in(p, view->col[c].valid, (size_t)rows);
            k->valid = p;
        } else k->valid = NULL;
        if (rc) return rc;
    }
    return 0;
}
static void tr_cons_gpu_slices(uint64_t rows, uint64_t *slices, uint64_t *per) {
    uint64_t s = (rows + TR_CONS_SLICE_ROWS - 1) / TR_CONS_SLICE_ROWS;
    if (s > TR_CONS_MAX_SLICES) s = TR_CONS_MAX_SLICES;
    if (!s) s = 1;
    *slices = s;
    *per = (rows + s - 1) / s;
    if (!*per) *per = 1;
}

#ifdef __cplusplus
extern "C" {
#endif
int tr_mgpu_cons_delta(const tr_cons_job_view *job, const tr_math_budget *budget, tr_cons_delta_out *out,
                       tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!job || !out || !tr_cons_gpu_ok(job)) return -1;                       /* TR_MATH_INVALID */
    const int32_t nm = job->plan.nmoments, n = job->cand.ncand;
    if (out->nmoments != nm || out->nident != n || !out->acc || (n && !out->ev)) return -1;
    tr_math_budget dflt;
    if (!budget) { tr_budget_default(&dflt); budget = &dflt; }
    tr_math_need need;
    tr_cons_delta_device_need(job, &need);
    int rc = tr_budget_check(budget, &need, refusal);
    if (rc) return rc;
    for (int32_t q = 0; q < 2 * nm; q++) out->acc[q] = 0;
    for (int32_t j = 0; j < n; j++) tr_cons_evidence_init(&out->ev[j]);
    out->gram_ins = out->gram_del = 0;
    const uint64_t rows = job->rows.rows;
    if (!rows) return 0;
    uint64_t *ev = (uint64_t *)calloc((size_t)(n ? n : 1) * TR_CONS_EV, 8);
    if (!ev) return -2;                                                         /* TR_MATH_MEMORY */
    for (int32_t j = 0; j < n; j++) { ev[(uint64_t)j * TR_CONS_EV + 8] = TR_CONS_NONE; ev[(uint64_t)j * TR_CONS_EV + 9] = TR_CONS_NONE; }
    uint64_t cnt[2] = {0, 0};
    void *mem[3 * TR_SCOPE_MAX_COLUMNS + 16];
    int nmem = 0;
    tr_scope_view d = job->rows;
    tr_ident_moment_view dm = job->plan;
    tr_ident_cand_view dc = job->cand;
    d.groups = NULL; d.selection = NULL;
    int64_t *dw = NULL, *dacc = NULL;
    uint64_t *dcnt = NULL, *dev = NULL, *doff = NULL;
    uint8_t *dbasis = NULL, *dexp = NULL;
    int32_t *dpar = NULL, *dvar = NULL, *dl = NULL, *dr = NULL, *dwords = NULL;
    uint32_t *dsup = NULL;
    int64_t *dcoef = NULL;
#define TR_CONS_DEV(ptr, bytes) do { if (!rc) { rc = tr_cuda_alloc((void **)&(ptr), (size_t)(bytes)); if (!rc) mem[nmem++] = (void *)(ptr); } } while (0)
#define TR_CONS_IN(dst, src, bytes) do { TR_CONS_DEV(dst, bytes); if (!rc && (uint64_t)(bytes) != 0) rc = tr_cuda_copy_in(dst, src, (size_t)(bytes)); } while (0)
    uint32_t cols = 0;
    for (int32_t i = 0; i < job->plan.nvars; i++) cols |= (uint32_t)1 << job->plan.var[i];
    for (int32_t i = 0; i < job->cand.nvars; i++) cols |= (uint32_t)1 << job->cand.var[i];
    rc = tr_cons_gpu_columns(&job->rows, &d, cols, mem, &nmem);
    TR_CONS_IN(dw, job->weights, rows * 8);
    TR_CONS_IN(dbasis, job->plan.basis, (uint64_t)job->plan.nbasis * (uint64_t)job->plan.nvars);
    TR_CONS_IN(dpar, job->plan.bparent, (uint64_t)job->plan.nbasis * 4);
    TR_CONS_IN(dvar, job->plan.bvar, (uint64_t)job->plan.nbasis * 4);
    TR_CONS_IN(dl, job->plan.mleft, (uint64_t)nm * 4);
    TR_CONS_IN(dr, job->plan.mright, (uint64_t)nm * 4);
    TR_CONS_IN(dacc, out->acc, (uint64_t)nm * 16);
    TR_CONS_IN(dcnt, cnt, sizeof cnt);
    if (n) {
        TR_CONS_IN(doff, job->cand.offset, ((uint64_t)n + 1) * 8);
        TR_CONS_IN(dsup, job->cand.support, (uint64_t)n * 4);
        TR_CONS_IN(dwords, job->cand.words, (uint64_t)n * 4);
        TR_CONS_IN(dexp, job->cand.exponents, job->cand.nterms * (uint64_t)job->cand.nvars);
        TR_CONS_IN(dcoef, job->cand.coefficients, job->cand.nterms * 8);
        TR_CONS_IN(dev, ev, (uint64_t)n * TR_CONS_EV * 8);
    }
#undef TR_CONS_IN
#undef TR_CONS_DEV
    dm.basis = dbasis; dm.bparent = dpar; dm.bvar = dvar; dm.mleft = dl; dm.mright = dr; dm.primes = NULL;
    dc.offset = doff; dc.support = dsup; dc.words = dwords; dc.exponents = dexp; dc.coefficients = dcoef;
    uint64_t slices, per;
    tr_cons_gpu_slices(rows, &slices, &per);
    const uint64_t mchunks = ((uint64_t)nm + TR_CONS_CHUNK - 1) / TR_CONS_CHUNK;
    if (!rc) { TR_LAUNCH(tr_cons_gram_kernel, mchunks * slices, d, dm, dw, slices, per, mchunks, dacc, dcnt); rc = tr_cuda_sync(); }
    if (!rc && n) {
        const uint64_t cchunks = ((uint64_t)n + TR_CONS_CAND_CHUNK - 1) / TR_CONS_CAND_CHUNK;
        TR_LAUNCH(tr_cons_verify_kernel, cchunks * slices, d, dc, dw, slices, per, cchunks, dev);
        rc = tr_cuda_sync();
    }
    if (!rc) rc = tr_cuda_copy_out(out->acc, dacc, (size_t)nm * 16);
    if (!rc) rc = tr_cuda_copy_out(cnt, dcnt, sizeof cnt);
    if (!rc && n) rc = tr_cuda_copy_out(ev, dev, (size_t)n * TR_CONS_EV * 8);
    for (int i = 0; i < nmem; i++) tr_cuda_free(mem[i]);
    if (!rc) {
        out->gram_ins = cnt[0]; out->gram_del = cnt[1];
        for (int32_t j = 0; j < n; j++) {
            const uint64_t *a = ev + (uint64_t)j * TR_CONS_EV;
            tr_cons_delta_evidence *e = &out->ev[j];
            e->checked_ins = a[0]; e->checked_del = a[1]; e->excluded_ins = a[2]; e->excluded_del = a[3];
            e->violations_ins = a[4]; e->violations_del = a[5]; e->overflow_ins = a[6]; e->overflow_del = a[7];
            e->witness_ins = a[8]; e->witness_del = a[9];
        }
    }
    free(ev);
    return rc;
}
#ifdef __cplusplus
}
#endif
