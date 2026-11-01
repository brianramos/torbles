/* META-001 on a CUDA device: batched snapshot checks of query parts with the semantics of tr_equiv_check_run
 * (math_equiv.h) and the same evidence bit for bit.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Work items (question, word block): each evaluates its question's cone on the BOOL-001 condition words of its block
 * (tr_equiv_word, the same inline code as the CPU loop: Kleene's connectives on 64 rows at a time) and, for measures,
 * GEOM-001's exact pair candidate P = D (f - g) at the rows both sides read (DISC-002's W-word evaluation, the width
 * proven by the host from the scope's extremes), keeps local counters (checked, excluded, differing, undecided, NULL
 * mismatches, NULL rows, the nine truth cells) and its smallest differing row, then adds one atomic per nonzero
 * counter and takes the witness with an atomic minimum: the first witness is the minimum row id in any thread order.
 * No barrier, no shared memory. The host validates the view and checks the budget of the device copies
 * (tr_equiv_device_need) before anything is copied. */
#include "math_cuda.h"
#include "math_equiv.h"

TR_MGPU_ABI(tr_equiv_check_view, TR_EQUIV_ABI);     /* structs passed to the CUDA library (math_handles.h) */
TR_MGPU_ABI(tr_equiv_evidence, TR_EQUIV_ABI);
TR_MGPU_ABI(tr_equiv_plan, TR_EQUIV_ABI);
TR_MGPU_ABI(tr_equiv_entry, TR_EQUIV_ABI);

#define TR_EQUIV_EV 16                      /* per question: checked .. witness (7 words), then the 9 cells */

TR_KERNEL void tr_equiv_check_kernel(tr_equiv_check_view v, uint64_t slices, uint64_t per, uint64_t *ev) {
    const uint64_t nw = (v.rows.rows + 63) / 64;
    TR_GRID_LOOP(i, (uint64_t)v.nq * slices) {
        const int32_t q = (int32_t)(i / slices);
        const uint64_t s = i % slices;
        const uint64_t w0 = s * per, w1 = w0 + per < nw ? w0 + per : nw;
        uint64_t val[2 * TR_EQUIV_MAX_CONE];
        tr_equiv_acc a;
        tr_equiv_acc_init(&a);
        for (uint64_t w = w0; w < w1; w++) tr_equiv_word(&v, q, w, val, &a);
        uint64_t *e = ev + (uint64_t)q * TR_EQUIV_EV;
        if (a.checked) tr_atomic_add_u64(&e[0], a.checked);
        if (a.excluded) tr_atomic_add_u64(&e[1], a.excluded);
        if (a.differ) tr_atomic_add_u64(&e[2], a.differ);
        if (a.undecided) tr_atomic_add_u64(&e[3], a.undecided);
        if (a.null_mismatch) tr_atomic_add_u64(&e[4], a.null_mismatch);
        if (a.nulls) tr_atomic_add_u64(&e[5], a.nulls);
        if (a.witness != TR_EQUIV_NONE) tr_atomic_min_u64(&e[6], a.witness);
        for (int k = 0; k < 9; k++) if (a.cells[k]) tr_atomic_add_u64(&e[7 + k], a.cells[k]);
    }
}

/* a column the view can read */
static int tr_equiv_gpu_col_ok(const tr_scope_view *v, int32_t c) {
    return c >= 0 && c < v->ncols && (!v->rows || v->col[c].codes || v->col[c].values);
}
/* the whole view is checked on the host: every index the kernel follows is in range */
static int tr_equiv_gpu_ok(const tr_equiv_check_view *v) {
    if (v->nq < 0 || v->nq > TR_EQUIV_MAX_QUESTIONS || (v->nq && !v->plans) || (v->nentries && !v->entries)) return 0;
    if (v->rows.ncols < 0 || v->rows.ncols > TR_SCOPE_MAX_COLUMNS || (v->rows.rows && !v->rows.selection)) return 0;
    const tr_bool_domain *d = &v->masks;
    if (tr_bool_domain_check(d) || d->kind != TR_BOOL_ROWS || d->points != v->rows.rows || d->k > TR_EQUIV_MAX_CONDITIONS) return 0;
    const tr_ident_cand_view *c = &v->cand;
    if (c->ncand < 0 || c->ncand % 3 || c->nvars < 0 || c->nvars > TR_IDENT_MAX_VARS) return 0;
    if (c->ncand && (!c->offset || !c->support || !c->words || (c->nterms && (!c->exponents || !c->coefficients)))) return 0;
    for (int32_t i = 0; i < c->nvars; i++) if (!tr_equiv_gpu_col_ok(&v->rows, c->var[i])) return 0;
    if (c->ncand && (c->offset[0] != 0 || c->offset[c->ncand] != c->nterms)) return 0;
    for (int32_t j = 0; j < c->ncand; j++)
        if (c->offset[j] > c->offset[j + 1] || (c->nvars < 32 && (c->support[j] >> c->nvars))) return 0;
    for (int32_t q = 0; q < v->nq; q++) {
        const tr_equiv_plan *p = &v->plans[q];
        if (p->kind < TR_EQUIV_PREDICATE || p->kind > TR_EQUIV_QUERY) return 0;
        if ((uint64_t)p->cone + p->ncone > v->nentries || p->ncone > TR_EQUIV_MAX_CONE) return 0;
        if (p->ra < -1 || p->ra >= (int32_t)p->ncone || p->rb < -1 || p->rb >= (int32_t)p->ncone) return 0;
        if (p->pair < -1 || 3 * (int64_t)p->pair + 2 >= c->ncand) return 0;
        for (int32_t k = 0; k < 32; k++)
            if ((((p->supa | p->supb) >> k) & 1) && !tr_equiv_gpu_col_ok(&v->rows, k)) return 0;
        for (uint32_t k = 0; k < p->ncone; k++) {
            const tr_equiv_entry *e = &v->entries[p->cone + k];
            if (e->op == TR_BOOL_CONST) { if (e->a < 0 || e->a > 2) return 0; }
            else if (e->op == TR_BOOL_VAR) { if (e->a < 0 || e->a >= d->k) return 0; }
            else if (e->op == TR_BOOL_NOT) { if (e->a < 0 || e->a >= (int32_t)k) return 0; }
            else if (e->op >= TR_BOOL_AND && e->op <= TR_BOOL_XOR) {
                if (e->a < 0 || e->a >= (int32_t)k || e->b < 0 || e->b >= (int32_t)k) return 0;
            } else return 0;
        }
    }
    return 1;
}
/* copies the scope columns in cols (values or codes, validity) and points the device view at them */
static int tr_equiv_gpu_columns(const tr_scope_view *view, tr_scope_view *d, uint64_t cols, void **mem, int *nmem) {
    int rc = 0;
    const uint64_t rows = view->rows;
    for (int32_t c = 0; c < view->ncols; c++) {
        tr_scope_view_column *k = &d->col[c];
        if (!((cols >> c) & 1) || !rows) { k->codes = NULL; k->values = NULL; k->valid = NULL; continue; }
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

#ifdef __cplusplus
extern "C" {
#endif
int tr_mgpu_equiv_check(const tr_equiv_check_view *view, const tr_math_budget *budget, tr_equiv_evidence *out,
                        tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!view || (view->nq && !out) || !tr_equiv_gpu_ok(view)) return -1;          /* TR_MATH_INVALID */
    tr_math_budget dflt;
    if (!budget) { tr_budget_default(&dflt); budget = &dflt; }
    tr_math_need need;
    tr_equiv_device_need(view, &need);
    int rc = tr_budget_check(budget, &need, refusal);
    if (rc) return rc;
    const int32_t nq = view->nq;
    const uint64_t rows = view->rows.rows, nw = (rows + 63) / 64;
    for (int32_t q = 0; q < nq; q++) {
        memset(&out[q], 0, sizeof out[q]);
        out[q].rows = rows; out[q].selected = view->rows.selected; out[q].witness = TR_EQUIV_NONE;
    }
    if (!nq || !nw) return 0;
    uint64_t *ev = (uint64_t *)calloc((size_t)nq * TR_EQUIV_EV, 8);
    if (!ev) return -2;                                                             /* TR_MATH_MEMORY */
    for (int32_t q = 0; q < nq; q++) ev[(uint64_t)q * TR_EQUIV_EV + 6] = TR_EQUIV_NONE;
    void *mem[3 * TR_SCOPE_MAX_COLUMNS + 16];
    int nmem = 0;
    tr_equiv_check_view d = *view;
    const tr_ident_cand_view *c = &view->cand;
    uint64_t cols = 0;
    for (int32_t i = 0; i < c->nvars; i++) cols |= (uint64_t)1 << c->var[i];
    for (int32_t q = 0; q < nq; q++) cols |= (uint64_t)(view->plans[q].supa | view->plans[q].supb);
    d.rows.groups = NULL;
    uint64_t *dsel = NULL, *dval = NULL, *dvalid = NULL, *doff = NULL, *dev = NULL;
    uint32_t *dsup = NULL;
    int32_t *dwords = NULL;
    uint8_t *dexp = NULL;
    int64_t *dcoef = NULL;
    tr_equiv_plan *dplans = NULL;
    tr_equiv_entry *dent = NULL;
    const uint64_t kw = (uint64_t)view->masks.k * nw;
#define TR_EQUIV_DEV(ptr, bytes) do { if (!rc) { rc = tr_cuda_alloc((void **)&(ptr), (size_t)(bytes)); if (!rc) mem[nmem++] = (void *)(ptr); } } while (0)
#define TR_EQUIV_IN(dst, src, bytes) do { TR_EQUIV_DEV(dst, bytes); if (!rc && (uint64_t)(bytes) != 0) rc = tr_cuda_copy_in(dst, src, (size_t)(bytes)); } while (0)
    rc = tr_equiv_gpu_columns(&view->rows, &d.rows, cols, mem, &nmem);
    TR_EQUIV_IN(dsel, view->rows.selection, nw * 8);
    if (kw) TR_EQUIV_IN(dval, view->masks.value, kw * 8);
    if (kw && view->masks.valid) TR_EQUIV_IN(dvalid, view->masks.valid, kw * 8);
    if (c->ncand) {
        TR_EQUIV_IN(doff, c->offset, ((uint64_t)c->ncand + 1) * 8);
        TR_EQUIV_IN(dsup, c->support, (uint64_t)c->ncand * 4);
        TR_EQUIV_IN(dwords, c->words, (uint64_t)c->ncand * 4);
        if (c->nterms) {
            TR_EQUIV_IN(dexp, c->exponents, c->nterms * (uint64_t)(c->nvars ? c->nvars : 1));
            TR_EQUIV_IN(dcoef, c->coefficients, c->nterms * 8);
        }
    }
    TR_EQUIV_IN(dplans, view->plans, (uint64_t)nq * sizeof(tr_equiv_plan));
    if (view->nentries) TR_EQUIV_IN(dent, view->entries, view->nentries * sizeof(tr_equiv_entry));
    TR_EQUIV_IN(dev, ev, (uint64_t)nq * TR_EQUIV_EV * 8);
#undef TR_EQUIV_IN
#undef TR_EQUIV_DEV
    d.rows.selection = dsel;
    d.masks.selection = dsel; d.masks.value = dval; d.masks.valid = view->masks.valid ? dvalid : NULL;
    d.cand.offset = doff; d.cand.support = dsup; d.cand.words = dwords; d.cand.exponents = dexp; d.cand.coefficients = dcoef;
    d.plans = dplans; d.entries = dent;
    uint64_t slices = (nw + TR_EQUIV_SLICE_WORDS - 1) / TR_EQUIV_SLICE_WORDS;
    if (slices > TR_EQUIV_MAX_SLICES) slices = TR_EQUIV_MAX_SLICES;
    if (!slices) slices = 1;
    uint64_t per = (nw + slices - 1) / slices;
    if (!per) per = 1;
    if (!rc) { TR_LAUNCH(tr_equiv_check_kernel, (uint64_t)nq * slices, d, slices, per, dev); rc = tr_cuda_sync(); }
    if (!rc) rc = tr_cuda_copy_out(ev, dev, (size_t)nq * TR_EQUIV_EV * 8);
    for (int i = 0; i < nmem; i++) tr_cuda_free(mem[i]);
    if (!rc)
        for (int32_t q = 0; q < nq; q++) {
            const uint64_t *a = ev + (uint64_t)q * TR_EQUIV_EV;
            tr_equiv_evidence *e = &out[q];
            e->checked = a[0]; e->excluded = a[1]; e->differ = a[2]; e->undecided = a[3];
            e->null_mismatch = a[4]; e->nulls = a[5]; e->witness = a[6];
            for (int k = 0; k < 9; k++) e->cells[k] = a[7 + k];
        }
    free(ev);
    return rc;
}
#ifdef __cplusplus
}
#endif
