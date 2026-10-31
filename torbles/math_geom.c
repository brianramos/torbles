/* GEOM-001: certified sparse ideal reduction; see math_geom.h.
 * SPDX-License-Identifier: Apache-2.0 */
#include "math_geom.h"
#include "mathematics.h"
#include "math_domain.h"
#include "math_handles.h"
#include <stdlib.h>
#include <string.h>

TR_MATH_ABI(tr_geom_order, TR_GEOM_ABI);   /* the ctypes boundary of GEOM-001 (math_handles.h) */
TR_MATH_ABI(tr_geom_poly, TR_GEOM_ABI);
TR_MATH_ABI(tr_geom_ideal_info, TR_GEOM_ABI);
TR_MATH_ABI(tr_geom_reduction_info, TR_GEOM_ABI);
TR_MATH_ABI(tr_geom_cost, TR_GEOM_ABI);
TR_MATH_ABI(tr_geom_pair_view, TR_GEOM_ABI);
TR_MATH_ABI(tr_geom_agreement, TR_GEOM_ABI);
TR_MATH_ABI(tr_geom_dep, TR_GEOM_ABI);
TR_MATH_ABI(tr_geom_check_view, TR_GEOM_ABI);
TR_MATH_ABI(tr_geom_check_result, TR_GEOM_ABI);

_Static_assert(sizeof(tr_geom_order) == 40, "tr_geom_order ABI");
_Static_assert(sizeof(tr_geom_poly) == 72, "tr_geom_poly ABI");
_Static_assert(sizeof(tr_geom_ideal_info) == 112, "tr_geom_ideal_info ABI");
_Static_assert(sizeof(tr_geom_reduction_info) == 88, "tr_geom_reduction_info ABI");
_Static_assert(sizeof(tr_geom_cost) == 64, "tr_geom_cost ABI");
_Static_assert(sizeof(tr_geom_pair_view) == 96, "tr_geom_pair_view ABI");
_Static_assert(sizeof(tr_geom_agreement) == 64, "tr_geom_agreement ABI");
_Static_assert(sizeof(tr_geom_dep) == 80, "tr_geom_dep ABI");
_Static_assert(sizeof(tr_geom_check_view) == 120, "tr_geom_check_view ABI");
_Static_assert(sizeof(tr_geom_check_result) == 48, "tr_geom_check_result ABI");

#define IDEAL_MAGIC UINT32_C(0x4c444947)
#define RED_MAGIC UINT32_C(0x44455247)
#define POL_MAGIC UINT32_C(0x4c4f5047)
#define PAIRS_MAGIC UINT32_C(0x52494147)
#define CHECKS_MAGIC UINT32_C(0x4b484347)
#define I128_LIMIT ((((unsigned __int128)1) << 127) - 1)
#define NV TR_GEOM_MAX_VARS

static void set_error(int32_t *error, int32_t code) { if (error) *error = code; }
static void *zalloc(uint64_t bytes) {
    if (bytes > (uint64_t)SIZE_MAX) return NULL;
    return calloc(1, bytes ? (size_t)bytes : 1);
}
static const tr_math_budget *budget_or(const tr_math_budget *b, tr_math_budget *store) {
    if (b) return b;
    tr_budget_default(store);
    return store;
}
static int32_t check_one(const tr_math_budget *b, int32_t res, uint64_t need, tr_math_refusal *r) {
    tr_math_budget dflt;
    return tr_budget_check_one(budget_or(b, &dflt), res, need, r);
}
static int view_ok(const tr_scope_view *v) {
    return v && v->ngroups >= 1 && (!v->rows || v->selection) && v->ncols >= 0 && v->ncols <= TR_SCOPE_MAX_COLUMNS;
}
static uint64_t mag_of(const tr_scope_view *v, int32_t c) {
    const tr_scope_view_column *k = &v->col[c];
    uint64_t m = k->nvalid ? (tr_abs_u64(k->min) > tr_abs_u64(k->max) ? tr_abs_u64(k->min) : tr_abs_u64(k->max)) : 0;
    return m < 1 ? 1 : m;
}

/* ================================================================ exact polynomial arithmetic */
typedef struct {
    const tr_math_budget *b; tr_math_refusal *r; int32_t rc;
    uint64_t max_terms, max_degree, max_iter, iter, peak, peak_degree;
} gctx;
static void gfail(gctx *x, int32_t rc) { if (!x->rc) x->rc = rc ? rc : TR_MATH_MEMORY; }
static void gbudget(gctx *x, int32_t resource, uint64_t need, uint64_t limit) {
    if (x->rc) return;
    x->rc = TR_MATH_BUDGET;
    if (x->r) { x->r->resource = resource; x->r->refused = 1; x->r->need = need; x->r->limit = limit; }
}
static void gctx_init(gctx *x, const tr_math_budget *b, tr_math_refusal *r, tr_math_budget *store) {
    memset(x, 0, sizeof *x);
    x->b = budget_or(b, store); x->r = r;
    tr_budget_none(r);
    x->max_terms = x->b->limit[TR_BUDGET_SUPPORT];
    x->max_degree = x->b->limit[TR_BUDGET_DEGREE];
    x->max_iter = x->b->limit[TR_BUDGET_ITERATIONS];
}
static int gstep(gctx *x) {                                    /* one iteration (an S-pair or a division step) */
    if (x->rc) return 0;
    if (++x->iter > x->max_iter) { gbudget(x, TR_BUDGET_ITERATIONS, x->iter, x->max_iter); return 0; }
    return 1;
}
static tr_num_rational Qi(int64_t n) { tr_num_rational q; q.numerator = n; q.denominator = 1; return q; }
static tr_num_rational qop(gctx *x, int op, tr_num_rational a, tr_num_rational b) {
    tr_num_rational r = Qi(0);
    if (x->rc) return r;
    int ok = op == 0 ? tr_q_add(a, b, &r) : op == 1 ? tr_q_sub(a, b, &r) : op == 2 ? tr_q_mul(a, b, &r) : tr_q_div(a, b, &r);
    if (!ok) { gfail(x, TR_MATH_OVERFLOW); r = Qi(0); }
    return r;
}
#define QADD(x, a, b) qop(x, 0, a, b)
#define QSUB(x, a, b) qop(x, 1, a, b)
#define QMUL(x, a, b) qop(x, 2, a, b)
#define QDIV(x, a, b) qop(x, 3, a, b)

typedef struct { uint8_t e[NV]; tr_num_rational c; } gterm;    /* 24 bytes */
typedef struct { gterm *t; uint64_t n, cap; } gpoly;           /* terms in decreasing order, nonzero coefficients */
typedef struct {
    int32_t kind, nv;
    int32_t col[NV];                 /* scope column of each exponent position (increasing) */
    int32_t sig[NV];                 /* positions by significance, most significant first */
} gorder;

static int edeg(const uint8_t *e) { int d = 0; for (int i = 0; i < NV; i++) d += e[i]; return d; }
static int mcmp(const gorder *o, const uint8_t *a, const uint8_t *b) {
    if (o->kind != TR_GEOM_LEX) {
        int da = edeg(a), db = edeg(b);
        if (da != db) return da < db ? -1 : 1;
        if (o->kind == TR_GEOM_DEGREVLEX) {
            for (int32_t i = o->nv - 1; i >= 0; i--) {
                int p = o->sig[i];
                if (a[p] != b[p]) return a[p] < b[p] ? 1 : -1;
            }
            return 0;
        }
    }
    for (int32_t i = 0; i < o->nv; i++) {
        int p = o->sig[i];
        if (a[p] != b[p]) return a[p] > b[p] ? 1 : -1;
    }
    return 0;
}
static int edivides(const uint8_t *a, const uint8_t *b) {     /* monomial a divides b */
    for (int i = 0; i < NV; i++) if (a[i] > b[i]) return 0;
    return 1;
}
static void gp_free(gpoly *p) { free(p->t); p->t = NULL; p->n = p->cap = 0; }
static int gp_reserve(gctx *x, gpoly *p, uint64_t n) {
    if (x->rc) return 0;
    if (n > x->max_terms) { gbudget(x, TR_BUDGET_SUPPORT, n, x->max_terms); return 0; }
    if (n <= p->cap) return 1;
    uint64_t cap = p->cap ? p->cap * 2 : 8;
    if (cap < n) cap = n;
    gterm *t = (gterm *)realloc(p->t, (size_t)cap * sizeof *t);
    if (!t) { gfail(x, TR_MATH_MEMORY); return 0; }
    p->t = t; p->cap = cap;
    return 1;
}
static int gp_push(gctx *x, gpoly *p, const uint8_t *e, tr_num_rational c) {
    if (!gp_reserve(x, p, p->n + 1)) return 0;
    memcpy(p->t[p->n].e, e, NV);
    p->t[p->n].c = c;
    p->n++;
    if (p->n > x->peak) x->peak = p->n;
    return 1;
}
static int gp_copy(gctx *x, const gpoly *a, gpoly *out) {
    out->n = 0;
    if (!gp_reserve(x, out, a->n ? a->n : 1)) return 0;
    if (a->n) memcpy(out->t, a->t, (size_t)a->n * sizeof *a->t);
    out->n = a->n;
    return 1;
}
/* e = a + m with the exponent and degree bounds checked */
static int eshift(gctx *x, const uint8_t *a, const uint8_t *m, uint8_t *e) {
    int d = 0;
    for (int i = 0; i < NV; i++) {
        int s = a[i] + m[i];
        if (s > 255) { gfail(x, TR_MATH_UNSUPPORTED); return 0; }
        e[i] = (uint8_t)s;
        d += s;
    }
    if ((uint64_t)d > x->max_degree) { gbudget(x, TR_BUDGET_DEGREE, (uint64_t)d, x->max_degree); return 0; }
    if ((uint64_t)d > x->peak_degree) x->peak_degree = (uint64_t)d;
    return 1;
}
/* out = A + c m B (out must not alias A or B) */
static int gp_axpy(gctx *x, const gorder *o, const gpoly *A, tr_num_rational c, const uint8_t *m, const gpoly *B, gpoly *out) {
    out->n = 0;
    uint64_t i = 0, j = 0;
    uint8_t e[NV];
    int have = 0;
    while ((i < A->n || j < B->n) && !x->rc) {
        if (j < B->n && !have) { if (!eshift(x, B->t[j].e, m, e)) break; have = 1; }
        int cmp = i >= A->n ? -1 : j >= B->n ? 1 : mcmp(o, A->t[i].e, e);
        if (cmp > 0) { gp_push(x, out, A->t[i].e, A->t[i].c); i++; }
        else if (cmp < 0) { gp_push(x, out, e, QMUL(x, c, B->t[j].c)); j++; have = 0; }
        else {
            tr_num_rational s = QADD(x, A->t[i].c, QMUL(x, c, B->t[j].c));
            if (s.numerator) gp_push(x, out, e, s);
            i++; j++; have = 0;
        }
    }
    return !x->rc;
}
/* a += c m B in place (through a scratch polynomial) */
static int gp_addmul(gctx *x, const gorder *o, gpoly *a, tr_num_rational c, const uint8_t *m, const gpoly *B) {
    if (!B->n || !c.numerator) return !x->rc;
    gpoly t = {NULL, 0, 0};
    if (!gp_axpy(x, o, a, c, m, B, &t)) { gp_free(&t); return 0; }
    gp_free(a);
    *a = t;
    return 1;
}
static int gp_mul(gctx *x, const gorder *o, const gpoly *A, const gpoly *B, gpoly *out) {
    gpoly acc = {NULL, 0, 0};
    for (uint64_t i = 0; i < A->n && !x->rc; i++) gp_addmul(x, o, &acc, A->t[i].c, A->t[i].e, B);
    if (x->rc) { gp_free(&acc); return 0; }
    gp_free(out);
    *out = acc;
    return 1;
}
static void gp_scale(gctx *x, gpoly *p, tr_num_rational c) {
    for (uint64_t i = 0; i < p->n; i++) p->t[i].c = QMUL(x, p->t[i].c, c);
}
/* sort decreasing (merge sort with the order as context), merge equal monomials, drop zeros */
static void gsort(const gorder *o, gterm *a, gterm *tmp, uint64_t n) {
    if (n < 2) return;
    uint64_t h = n / 2;
    gsort(o, a, tmp, h);
    gsort(o, a + h, tmp, n - h);
    uint64_t i = 0, j = h, k = 0;
    while (i < h && j < n) tmp[k++] = mcmp(o, a[i].e, a[j].e) >= 0 ? a[i++] : a[j++];
    while (i < h) tmp[k++] = a[i++];
    while (j < n) tmp[k++] = a[j++];
    memcpy(a, tmp, (size_t)n * sizeof *a);
}
static int gp_normalize(gctx *x, const gorder *o, gpoly *p) {
    if (x->rc) return 0;
    if (p->n > 1) {
        gterm *tmp = (gterm *)malloc((size_t)p->n * sizeof *tmp);
        if (!tmp) { gfail(x, TR_MATH_MEMORY); return 0; }
        gsort(o, p->t, tmp, p->n);
        free(tmp);
    }
    uint64_t m = 0;
    for (uint64_t i = 0; i < p->n;) {
        tr_num_rational s = p->t[i].c;
        uint64_t j = i + 1;
        while (j < p->n && !memcmp(p->t[j].e, p->t[i].e, NV)) { s = QADD(x, s, p->t[j].c); j++; }
        if (s.numerator) { p->t[m] = p->t[i]; p->t[m].c = s; m++; }
        i = j;
    }
    p->n = m;
    return !x->rc;
}
/* p = sum_j q_j G_j + r, every term of r irreducible (q: nb quotients, or NULL) */
static int gp_reduce(gctx *x, const gorder *o, const gpoly *p, gpoly *const *G, int32_t nb, gpoly *q, gpoly *r) {
    gpoly w = {NULL, 0, 0};
    if (!gp_copy(x, p, &w)) { gp_free(&w); return 0; }
    r->n = 0;
    if (q) for (int32_t j = 0; j < nb; j++) q[j].n = 0;
    uint64_t head = 0;
    while (head < w.n && !x->rc) {
        const gterm lt = w.t[head];
        int32_t j = -1;
        for (int32_t k = 0; k < nb && j < 0; k++) if (G[k]->n && edivides(G[k]->t[0].e, lt.e)) j = k;
        if (j < 0) { gp_push(x, r, lt.e, lt.c); head++; continue; }
        if (!gstep(x)) break;
        tr_num_rational coef = QDIV(x, lt.c, G[j]->t[0].c);
        uint8_t mono[NV];
        for (int i = 0; i < NV; i++) mono[i] = (uint8_t)(lt.e[i] - G[j]->t[0].e[i]);
        if (q) gp_push(x, &q[j], mono, coef);
        gpoly rest = {w.t + head, w.n - head, w.n - head}, t = {NULL, 0, 0};   /* the unreduced part */
        tr_num_rational neg = QSUB(x, Qi(0), coef);
        if (gp_axpy(x, o, &rest, neg, mono, G[j], &t)) { gp_free(&w); w = t; head = 0; }
        else gp_free(&t);
    }
    gp_free(&w);
    return !x->rc;
}

/* ------------------------------------------------------------------ orders and claims */
static int col_pos(const gorder *o, int32_t col) {
    for (int32_t i = 0; i < o->nv; i++) if (o->col[i] == col) return i;
    return -1;
}
/* the order over the variable set cols[n] (increasing scope columns): the spec's columns first, then the rest */
static int32_t order_make(const tr_geom_order *spec, const int32_t *cols, int32_t n, gorder *o) {
    memset(o, 0, sizeof *o);
    o->kind = spec ? spec->kind : TR_GEOM_DEGLEX;
    if (o->kind != TR_GEOM_LEX && o->kind != TR_GEOM_DEGLEX && o->kind != TR_GEOM_DEGREVLEX) return TR_MATH_INVALID;
    if (spec && (spec->nvars < 0 || spec->nvars > NV)) return TR_MATH_INVALID;
    o->nv = n;
    for (int32_t i = 0; i < n; i++) o->col[i] = cols[i];
    int used[NV] = {0};
    int32_t k = 0;
    for (int32_t i = 0; spec && i < spec->nvars; i++) {
        for (int32_t j = 0; j < i; j++) if (spec->col[j] == spec->col[i]) return TR_MATH_INVALID;
        int p = col_pos(o, spec->col[i]);
        if (p >= 0 && !used[p]) { o->sig[k++] = p; used[p] = 1; }
    }
    for (int32_t p = 0; p < n; p++) if (!used[p]) o->sig[k++] = p;
    return TR_MATH_OK;
}
static int32_t add_cols(int32_t *cols, int32_t *n, const int32_t *more, int32_t m) {
    for (int32_t i = 0; i < m; i++) {
        int seen = 0;
        for (int32_t j = 0; j < *n; j++) seen |= cols[j] == more[i];
        if (seen) continue;
        if (*n >= NV) return TR_MATH_UNSUPPORTED;
        cols[(*n)++] = more[i];
    }
    for (int32_t i = 1; i < *n; i++)                             /* increasing scope columns */
        for (int32_t j = i; j > 0 && cols[j - 1] > cols[j]; j--) { int32_t t = cols[j]; cols[j] = cols[j - 1]; cols[j - 1] = t; }
    return TR_MATH_OK;
}
/* an FND-007 claim as a polynomial over the order's variables (FND-007's reading of affine / sum / product) */
static int claim_poly(gctx *x, const gorder *o, const tr_cert_claim *k, gpoly *out) {
    int32_t at[TR_CERT_MAX_COLS];
    for (int32_t i = 0; i < k->ncols; i++) { at[i] = col_pos(o, k->col[i]); if (at[i] < 0) { gfail(x, TR_MATH_INVALID); return 0; } }
    uint8_t e[NV];
    tr_num_rational c;
    out->n = 0;
    if ((k->kind == TR_CLAIM_AFFINE && k->ncols != 2) ||
        ((k->kind == TR_CLAIM_SUM || k->kind == TR_CLAIM_PRODUCT) && k->ncols != 3)) { gfail(x, TR_MATH_INVALID); return 0; }
#define MONO0() memset(e, 0, sizeof e)
    switch (k->kind) {
    case TR_CLAIM_AFFINE:
        MONO0(); e[at[0]] = 1; gp_push(x, out, e, Qi(1));
        if (k->scale) { MONO0(); e[at[1]] = 1; if (!tr_q_make128(-(__int128)k->scale, 1, &c)) gfail(x, TR_MATH_OVERFLOW); gp_push(x, out, e, c); }
        if (k->bias) { MONO0(); if (!tr_q_make128(-(__int128)k->bias, 1, &c)) gfail(x, TR_MATH_OVERFLOW); gp_push(x, out, e, c); }
        break;
    case TR_CLAIM_SUM:
        MONO0(); e[at[0]] = 1; gp_push(x, out, e, Qi(1));
        MONO0(); e[at[1]]++; gp_push(x, out, e, Qi(-1));
        MONO0(); e[at[2]]++; gp_push(x, out, e, Qi(-1));
        break;
    case TR_CLAIM_PRODUCT:
        MONO0(); e[at[0]] = 1; gp_push(x, out, e, Qi(1));
        MONO0(); e[at[1]]++; e[at[2]]++; gp_push(x, out, e, Qi(-1));
        break;
    case TR_CLAIM_POLYNOMIAL:
        for (uint64_t t = 0; t < k->nterms && !x->rc; t++) {
            MONO0();
            for (int32_t i = 0; i < k->ncols; i++) e[at[i]] = k->exponents[t * (uint64_t)k->ncols + (uint64_t)i];
            if (!eshift(x, e, (const uint8_t *)"\0\0\0\0\0\0\0\0", e)) break;
            gp_push(x, out, e, Qi(k->coefficients[t]));
        }
        break;
    default:
        gfail(x, TR_MATH_INVALID);
    }
#undef MONO0
    return gp_normalize(x, o, out);
}
/* a tr_geom_poly as a polynomial over the order's variables */
static int poly_in(gctx *x, const gorder *o, const tr_geom_poly *f, gpoly *out) {
    out->n = 0;
    int32_t at[NV];
    for (int32_t i = 0; i < f->ncols; i++) { at[i] = col_pos(o, f->col[i]); if (at[i] < 0) { gfail(x, TR_MATH_INVALID); return 0; } }
    for (uint64_t t = 0; t < f->nterms && !x->rc; t++) {
        uint8_t e[NV] = {0};
        for (int32_t i = 0; i < f->ncols; i++) e[at[i]] = f->exponents[t * (uint64_t)f->ncols + (uint64_t)i];
        if (!eshift(x, e, (const uint8_t *)"\0\0\0\0\0\0\0\0", e)) break;
        tr_num_rational c;
        int64_t den = f->den ? f->den[t] : 1;
        if (den <= 0 || !tr_q_make(f->num[t], den, &c)) { gfail(x, den <= 0 ? TR_MATH_INVALID : TR_MATH_OVERFLOW); break; }
        gp_push(x, out, e, c);
    }
    return gp_normalize(x, o, out);
}
static int32_t poly_check(const tr_geom_poly *f) {
    if (!f || f->ncols < 0 || f->reserved || (f->nterms && (!f->exponents || !f->num))) return TR_MATH_INVALID;
    if (f->ncols > NV) return TR_MATH_UNSUPPORTED;
    for (int32_t i = 0; i < f->ncols; i++) {
        if (f->col[i] < 0 || f->col[i] >= TR_SCOPE_MAX_COLUMNS) return TR_MATH_INVALID;
        for (int32_t j = 0; j < i; j++) if (f->col[j] == f->col[i]) return TR_MATH_INVALID;
    }
    return TR_MATH_OK;
}
/* a polynomial over order o moved to order p (p's variables a superset of o's): positions remapped, re-sorted */
static int gp_remap(gctx *x, const gorder *o, const gpoly *a, const gorder *p, gpoly *out) {
    int map[NV];
    for (int32_t i = 0; i < o->nv; i++) { map[i] = col_pos(p, o->col[i]); if (map[i] < 0) { gfail(x, TR_MATH_INVALID); return 0; } }
    out->n = 0;
    for (uint64_t t = 0; t < a->n && !x->rc; t++) {
        uint8_t e[NV] = {0};
        for (int32_t i = 0; i < o->nv; i++) e[map[i]] = a->t[t].e[i];
        gp_push(x, out, e, a->t[t].c);
    }
    return gp_normalize(x, p, out);
}
/* lcm of the denominators (int64, 0 on overflow) */
static int64_t lcm_den(int64_t l, int64_t d) {
    if (!l) return 0;
    uint64_t g = tr_gcd_u64((uint64_t)l, (uint64_t)d);
    __int128 r = (__int128)(l / (int64_t)g) * d;
    return r > INT64_MAX ? 0 : (int64_t)r;
}
static int64_t gp_den(const gpoly *p, int64_t l) {
    for (uint64_t t = 0; t < p->n && l; t++) l = lcm_den(l, p->t[t].c.denominator);
    return l;
}
/* D p as int64 coefficients (0 on overflow) */
static int gp_integer(const gpoly *p, int64_t D, int64_t *out) {
    for (uint64_t t = 0; t < p->n; t++) {
        __int128 v = (__int128)p->t[t].c.numerator * (D / p->t[t].c.denominator);
        if (v < INT64_MIN || v > INT64_MAX) return 0;
        out[t] = (int64_t)v;
    }
    return 1;
}

/* ================================================================ owned polynomials */
struct tr_geom_pol_s {
    uint32_t magic;
    int32_t ncols;
    int32_t col[NV];
    uint64_t n;
    uint8_t *e;                      /* n x ncols */
    int64_t *num, *den;
};
static int pol_valid(const tr_geom_pol *p) { return p && p->magic == POL_MAGIC; }
void tr_geom_pol_free(tr_geom_pol *p) {
    if (!pol_valid(p)) return;
    free(p->e); free(p->num); free(p->den);
    p->magic = 0;
    free(p);
}
static tr_geom_pol *pol_from(const gorder *o, const gpoly *a, int32_t *error) {
    tr_geom_pol *p = (tr_geom_pol *)zalloc(sizeof *p);
    if (!p) { set_error(error, TR_MATH_MEMORY); return NULL; }
    p->magic = POL_MAGIC; p->ncols = o->nv; p->n = a->n;
    for (int32_t i = 0; i < NV; i++) p->col[i] = i < o->nv ? o->col[i] : -1;
    p->e = (uint8_t *)zalloc((a->n ? a->n : 1) * (uint64_t)(o->nv ? o->nv : 1));
    p->num = (int64_t *)zalloc((a->n ? a->n : 1) * 8); p->den = (int64_t *)zalloc((a->n ? a->n : 1) * 8);
    if (!p->e || !p->num || !p->den) { tr_geom_pol_free(p); set_error(error, TR_MATH_MEMORY); return NULL; }
    for (uint64_t t = 0; t < a->n; t++) {
        memcpy(p->e + t * (uint64_t)o->nv, a->t[t].e, (size_t)o->nv);
        p->num[t] = a->t[t].c.numerator; p->den[t] = a->t[t].c.denominator;
    }
    set_error(error, TR_MATH_OK);
    return p;
}
int32_t tr_geom_pol_get(const tr_geom_pol *p, int32_t *ncols, int32_t *cols, uint64_t *nterms) {
    if (!pol_valid(p) || !ncols || !nterms) return TR_MATH_INVALID;
    *ncols = p->ncols; *nterms = p->n;
    if (cols) for (int32_t i = 0; i < NV; i++) cols[i] = p->col[i];
    return TR_MATH_OK;
}
int32_t tr_geom_pol_export(const tr_geom_pol *p, uint8_t *e, int64_t *num, int64_t *den, uint64_t capacity) {
    if (!pol_valid(p) || capacity < p->n || (p->n && (!e || !num || !den))) return TR_MATH_INVALID;
    if (p->n) {
        memcpy(e, p->e, (size_t)(p->n * (uint64_t)p->ncols));
        memcpy(num, p->num, (size_t)p->n * 8); memcpy(den, p->den, (size_t)p->n * 8);
    }
    return TR_MATH_OK;
}

/* ================================================================ ideals */
struct tr_geom_ideal_s {
    uint32_t magic;
    int32_t nrel, nb, completed, policy, weakest;
    gorder order;
    tr_cert **certs;                 /* nrel owned clones */
    gpoly *rel;                      /* nrel relations */
    gpoly *basis;                    /* nb */
    gpoly *cof;                      /* nb x nrel: basis[j] = sum_i cof[j nrel + i] rel[i] */
    uint64_t pairs, steps, max_terms, max_degree, cofactor_terms, bytes;
};
static int ideal_valid(const tr_geom_ideal *I) { return I && I->magic == IDEAL_MAGIC; }
static void gp_free_n(gpoly *p, uint64_t n) { if (!p) return; for (uint64_t i = 0; i < n; i++) gp_free(&p[i]); free(p); }
void tr_geom_ideal_free(tr_geom_ideal *I) {
    if (!ideal_valid(I)) return;
    for (int32_t i = 0; i < I->nrel; i++) tr_cert_free(I->certs ? I->certs[i] : NULL);
    free(I->certs);
    gp_free_n(I->rel, (uint64_t)I->nrel);
    gp_free_n(I->basis, (uint64_t)I->nb);
    gp_free_n(I->cof, (uint64_t)I->nb * (uint64_t)I->nrel);
    I->magic = 0;
    free(I);
}
/* appends a basis element with its cofactors (both taken) */
static int basis_push(gctx *x, tr_geom_ideal *I, int32_t *cap, gpoly *g, gpoly *A) {
    if (x->rc) return 0;
    if (I->nb >= TR_GEOM_MAX_BASIS) { gfail(x, TR_MATH_UNSUPPORTED); return 0; }
    if ((uint64_t)I->nb + 1 > x->b->limit[TR_BUDGET_OUTPUT]) { gbudget(x, TR_BUDGET_OUTPUT, (uint64_t)I->nb + 1, x->b->limit[TR_BUDGET_OUTPUT]); return 0; }
    if (I->nb >= *cap) {
        int32_t nc = *cap ? *cap * 2 : 16;
        gpoly *b = (gpoly *)realloc(I->basis, (size_t)nc * sizeof *b);
        if (b) I->basis = b;
        gpoly *c = b ? (gpoly *)realloc(I->cof, (size_t)nc * (size_t)I->nrel * sizeof *c) : NULL;
        if (c) I->cof = c;
        if (!b || !c) { gfail(x, TR_MATH_MEMORY); return 0; }
        *cap = nc;
    }
    I->basis[I->nb] = *g;
    for (int32_t i = 0; i < I->nrel; i++) I->cof[(uint64_t)I->nb * I->nrel + i] = A[i];
    I->nb++;
    return 1;
}
static void monic(gctx *x, gpoly *g, gpoly *A, int32_t nrel) {
    if (!g->n) return;
    tr_num_rational inv = QDIV(x, Qi(1), g->t[0].c);
    gp_scale(x, g, inv);
    for (int32_t i = 0; i < nrel; i++) gp_scale(x, &A[i], inv);
}
typedef struct { int32_t a, b; } gpair;
static void elcm(const uint8_t *a, const uint8_t *b, uint8_t *l) { for (int i = 0; i < NV; i++) l[i] = a[i] > b[i] ? a[i] : b[i]; }
/* A_out = A_in - sum_k q_k A_k (the cofactors of a remainder) */
static int cof_subtract(gctx *x, const gorder *o, const tr_geom_ideal *I, gpoly *A, const gpoly *q, int32_t nb) {
    gpoly prod = {NULL, 0, 0};
    for (int32_t k = 0; k < nb && !x->rc; k++) {
        if (!q[k].n) continue;
        for (int32_t i = 0; i < I->nrel && !x->rc; i++) {
            const gpoly *Ak = &I->cof[(uint64_t)k * I->nrel + i];
            if (!Ak->n) continue;
            if (gp_mul(x, o, &q[k], Ak, &prod)) {
                const uint8_t zero[NV] = {0};
                gp_addmul(x, o, &A[i], Qi(-1), zero, &prod);
            }
        }
    }
    gp_free(&prod);
    return !x->rc;
}
static int buchberger(gctx *x, tr_geom_ideal *I, int32_t *cap) {
    const gorder *o = &I->order;
    const int32_t nrel = I->nrel;
    gpair *pairs = NULL;
    uint64_t np = 0, pcap = 0;
#define ADDPAIR(A, B) do { if (np >= pcap) { uint64_t nc = pcap ? 2 * pcap : 64; gpair *t = (gpair *)realloc(pairs, (size_t)nc * sizeof *t); \
        if (!t) { gfail(x, TR_MATH_MEMORY); break; } pairs = t; pcap = nc; } pairs[np].a = (A); pairs[np].b = (B); np++; } while (0)
    for (int32_t b = 1; b < I->nb && !x->rc; b++) for (int32_t a = 0; a < b && !x->rc; a++) ADDPAIR(a, b);
    gpoly *q = NULL, *A = NULL;
    gpoly S = {NULL, 0, 0}, tmp = {NULL, 0, 0}, r = {NULL, 0, 0};
    while (np && !x->rc && !q) {
        uint64_t best = 0;                                     /* the normal strategy: the smallest lcm degree */
        int bd = 1 << 30;
        for (uint64_t i = 0; i < np; i++) {
            uint8_t l[NV];
            elcm(I->basis[pairs[i].a].t[0].e, I->basis[pairs[i].b].t[0].e, l);
            int d = edeg(l);
            if (d < bd) { bd = d; best = i; }
        }
        gpair pr = pairs[best];
        memmove(pairs + best, pairs + best + 1, (size_t)(np - best - 1) * sizeof *pairs);
        np--;
        if (!gstep(x)) break;
        I->pairs++;
        const gpoly *ga = &I->basis[pr.a], *gb = &I->basis[pr.b];
        uint8_t L[NV], ma[NV], mb[NV];
        elcm(ga->t[0].e, gb->t[0].e, L);
        int coprime = 1;
        for (int i = 0; i < NV; i++) {
            ma[i] = (uint8_t)(L[i] - ga->t[0].e[i]); mb[i] = (uint8_t)(L[i] - gb->t[0].e[i]);
            if (ga->t[0].e[i] && gb->t[0].e[i]) coprime = 0;
        }
        if (coprime) continue;                                 /* Buchberger's product criterion */
        /* S = ma ga - mb gb (monic elements), with its cofactors */
        gpoly empty = {NULL, 0, 0};
        if (!gp_axpy(x, o, &empty, Qi(1), ma, ga, &tmp) || !gp_axpy(x, o, &tmp, Qi(-1), mb, gb, &S)) break;
        const int32_t nq = I->nb;
        q = (gpoly *)zalloc((uint64_t)nq * sizeof *q);
        A = (gpoly *)zalloc((uint64_t)nrel * sizeof *A);
        if (!q || !A) { gfail(x, TR_MATH_MEMORY); break; }
        for (int32_t i = 0; i < nrel && !x->rc; i++) {
            gpoly t1 = {NULL, 0, 0};
            gp_axpy(x, o, &empty, Qi(1), ma, &I->cof[(uint64_t)pr.a * nrel + i], &t1);
            gp_axpy(x, o, &t1, Qi(-1), mb, &I->cof[(uint64_t)pr.b * nrel + i], &A[i]);
            gp_free(&t1);
        }
        gpoly *G[TR_GEOM_MAX_BASIS];
        for (int32_t k = 0; k < nq; k++) G[k] = &I->basis[k];
        if (!x->rc) gp_reduce(x, o, &S, G, nq, q, &r);
        if (!x->rc && r.n) {
            cof_subtract(x, o, I, A, q, nq);
            gpoly g = r;
            r.t = NULL; r.n = r.cap = 0;
            monic(x, &g, A, nrel);
            if (basis_push(x, I, cap, &g, A)) {
                free(A); A = NULL;                             /* its polynomials now belong to the basis */
                for (int32_t k = 0; k < nq && !x->rc; k++) ADDPAIR(k, nq);
            } else gp_free(&g);
        }
        gp_free_n(q, (uint64_t)nq);
        q = NULL;
        if (A) { gp_free_n(A, (uint64_t)nrel); A = NULL; }
    }
#undef ADDPAIR
    if (A) gp_free_n(A, (uint64_t)nrel);
    gp_free(&S); gp_free(&tmp); gp_free(&r);
    free(pairs);
    return !x->rc;
}
/* the reduced basis: drop elements whose leading monomial another's divides, reduce every tail, sort increasing */
static int interreduce(gctx *x, tr_geom_ideal *I) {
    const gorder *o = &I->order;
    const int32_t nrel = I->nrel;
    /* a minimal basis: g_j goes when another leading monomial divides its own (equal ones: the earliest stays); by
     * transitivity a kept element divides every dropped one, so the marks are taken on the unmodified array */
    int drops[TR_GEOM_MAX_BASIS] = {0};
    for (int32_t j = 0; j < I->nb; j++)
        for (int32_t k = 0; k < I->nb && !drops[j]; k++) {
            if (k == j) continue;
            const uint8_t *lk = I->basis[k].t[0].e, *lj = I->basis[j].t[0].e;
            if (edivides(lk, lj) && (memcmp(lk, lj, NV) || k < j)) drops[j] = 1;
        }
    int32_t keep = 0;
    for (int32_t j = 0; j < I->nb; j++) {
        if (drops[j]) {
            gp_free(&I->basis[j]);
            for (int32_t i = 0; i < nrel; i++) gp_free(&I->cof[(uint64_t)j * nrel + i]);
            continue;
        }
        if (keep != j) {
            I->basis[keep] = I->basis[j];
            for (int32_t i = 0; i < nrel; i++) I->cof[(uint64_t)keep * nrel + i] = I->cof[(uint64_t)j * nrel + i];
        }
        keep++;
    }
    I->nb = keep;
    gpoly *q = (gpoly *)zalloc((uint64_t)(keep ? keep : 1) * sizeof *q);
    gpoly tail = {NULL, 0, 0}, r = {NULL, 0, 0};
    if (!q) { gfail(x, TR_MATH_MEMORY); return 0; }
    for (int32_t j = 0; j < keep && !x->rc; j++) {
        gpoly *G[TR_GEOM_MAX_BASIS];
        int32_t idx[TR_GEOM_MAX_BASIS], m = 0;
        for (int32_t k = 0; k < keep; k++) if (k != j) { G[m] = &I->basis[k]; idx[m] = k; m++; }
        gpoly *g = &I->basis[j];
        tail.n = 0;
        for (uint64_t t = 1; t < g->n && !x->rc; t++) gp_push(x, &tail, g->t[t].e, g->t[t].c);
        if (!gp_reduce(x, o, &tail, G, m, q, &r)) break;
        /* cofactors: A_j -= sum q A_k (in the original indexing) */
        gpoly qq[TR_GEOM_MAX_BASIS];
        memset(qq, 0, sizeof qq);
        for (int32_t k = 0; k < m; k++) qq[idx[k]] = q[k];
        cof_subtract(x, o, I, &I->cof[(uint64_t)j * nrel], qq, keep);
        gpoly ng = {NULL, 0, 0};
        gp_push(x, &ng, g->t[0].e, g->t[0].c);
        for (uint64_t t = 0; t < r.n && !x->rc; t++) gp_push(x, &ng, r.t[t].e, r.t[t].c);
        gp_free(g);
        *g = ng;
        for (int32_t k = 0; k < m; k++) q[k].n = 0;
    }
    for (int32_t k = 0; k < keep; k++) gp_free(&q[k]);
    free(q);
    gp_free(&tail); gp_free(&r);
    /* increasing leading monomial (insertion sort, few elements) */
    for (int32_t a = 1; a < I->nb && !x->rc; a++)
        for (int32_t b = a; b > 0 && mcmp(o, I->basis[b - 1].t[0].e, I->basis[b].t[0].e) > 0; b--) {
            gpoly t = I->basis[b]; I->basis[b] = I->basis[b - 1]; I->basis[b - 1] = t;
            for (int32_t i = 0; i < nrel; i++) {
                gpoly c = I->cof[(uint64_t)b * nrel + i];
                I->cof[(uint64_t)b * nrel + i] = I->cof[(uint64_t)(b - 1) * nrel + i];
                I->cof[(uint64_t)(b - 1) * nrel + i] = c;
            }
        }
    return !x->rc;
}
/* the relations' recorded sources (FND-007 records: snapshots and hypotheses) are one table (or one array scope) seen
 * through one column layout: the same identity, and each recorded scope column the same stored column both ways (two
 * scopes over a table can number its columns differently); a derived relation keeps no record (its premises do, and
 * FND-007 authorizes it only where every premise authorizes) */
static int32_t sources_agree(const tr_cert *const *rel, int32_t n) {
    tr_cert_source *first = (tr_cert_source *)malloc(sizeof *first), *cur = (tr_cert_source *)malloc(sizeof *cur);
    int32_t pairs[TR_GEOM_MAX_RELATIONS * TR_CERT_MAX_COLS][2], np = 0, rc = TR_MATH_OK, have = 0;
    if (!first || !cur) rc = TR_MATH_MEMORY;
    for (int32_t i = 0; i < n && !rc; i++) {
        const tr_cert_record *rec = tr_cert_get_record(rel[i]);
        if (!rec) continue;
        tr_cert_source *s = have ? cur : first;
        if (tr_cert_record_get(rec, s)) { rc = TR_MATH_INVALID; break; }
        if (have && (memcmp(first->identity, s->identity, sizeof first->identity) || first->source != s->source ||
                     first->data[0] != s->data[0] || first->data[1] != s->data[1])) { rc = TR_MATH_DOMAIN_ERROR; break; }
        have = 1;
        if (s->source != TR_SCOPE_TABLE) continue;               /* an array scope: its descriptor digest decides */
        for (int32_t k = 0; k < s->ncols && k < TR_SCOPE_MAX_COLUMNS && !rc; k++) {
            int32_t idx = s->col[k].index, col = s->col[k].col, seen = 0;
            for (int32_t q = 0; q < np && !rc; q++) {
                if ((pairs[q][0] == idx) != (pairs[q][1] == col)) rc = TR_MATH_DOMAIN_ERROR;   /* a different layout */
                seen |= pairs[q][0] == idx;
            }
            if (!rc && !seen && np < TR_GEOM_MAX_RELATIONS * TR_CERT_MAX_COLS) { pairs[np][0] = idx; pairs[np][1] = col; np++; }
        }
    }
    free(first); free(cur);
    return rc;
}
tr_geom_ideal *tr_geom_ideal_make(const tr_cert *const *relations, int32_t n, const tr_geom_order *spec,
                                  int32_t complete, const tr_math_budget *budget, tr_math_refusal *refusal,
                                  int32_t *error) {
    tr_math_budget store;
    gctx x;
    gctx_init(&x, budget, refusal, &store);
    if (!relations || n < 1 || (complete != 0 && complete != 1)) { set_error(error, TR_MATH_INVALID); return NULL; }
    if (n > TR_GEOM_MAX_RELATIONS) { set_error(error, TR_MATH_UNSUPPORTED); return NULL; }
    int32_t cols[NV], nc = 0, policy = 0, weakest = 0;
    for (int32_t i = 0; i < n; i++) {
        tr_cert_info info;
        tr_cert_claim k;
        if (!relations[i] || tr_cert_get_info(relations[i], &info) || tr_cert_get_claim(relations[i], &k)) {
            set_error(error, TR_MATH_INVALID); return NULL;
        }
        if (!(k.kind == TR_CLAIM_POLYNOMIAL || k.kind == TR_CLAIM_AFFINE || k.kind == TR_CLAIM_SUM ||
              k.kind == TR_CLAIM_PRODUCT) || k.field || k.tolerance || k.carrier != TR_CARRIER_I128 ||
            info.kind == TR_CERT_LAW || info.kind == TR_CERT_SCHEMA) { set_error(error, TR_MATH_DOMAIN_ERROR); return NULL; }
        if (i && k.policy != policy) { set_error(error, TR_MATH_DOMAIN_ERROR); return NULL; }   /* one NULL policy */
        policy = k.policy;
        if (info.kind > weakest) weakest = info.kind;
        int32_t rc = add_cols(cols, &nc, k.col, k.ncols);
        if (rc) { set_error(error, rc); return NULL; }
    }
    int32_t rs = sources_agree(relations, n);
    if (rs) { set_error(error, rs); return NULL; }
    tr_geom_ideal *I = (tr_geom_ideal *)zalloc(sizeof *I);
    if (!I) { set_error(error, TR_MATH_MEMORY); return NULL; }
    I->magic = IDEAL_MAGIC; I->nrel = n; I->policy = policy; I->weakest = weakest; I->completed = complete;
    int32_t rc = order_make(spec, cols, nc, &I->order);
    if (rc) { tr_geom_ideal_free(I); set_error(error, rc); return NULL; }
    I->certs = (tr_cert **)zalloc((uint64_t)n * sizeof *I->certs);
    I->rel = (gpoly *)zalloc((uint64_t)n * sizeof *I->rel);
    if (!I->certs || !I->rel) { tr_geom_ideal_free(I); set_error(error, TR_MATH_MEMORY); return NULL; }
    int32_t cap = 0;
    for (int32_t i = 0; i < n && !x.rc; i++) {
        int32_t e = 0;
        I->certs[i] = tr_cert_clone(relations[i], &e);
        if (!I->certs[i]) { gfail(&x, e); break; }
        tr_cert_claim k;
        tr_cert_get_claim(I->certs[i], &k);
        if (!claim_poly(&x, &I->order, &k, &I->rel[i])) break;
        if (!I->rel[i].n) continue;                            /* the zero polynomial generates nothing */
        gpoly g = {NULL, 0, 0};
        gpoly *A = (gpoly *)zalloc((uint64_t)n * sizeof *A);
        if (!A) { gfail(&x, TR_MATH_MEMORY); break; }
        gp_copy(&x, &I->rel[i], &g);
        const uint8_t zero[NV] = {0};
        gp_push(&x, &A[i], zero, Qi(1));
        monic(&x, &g, A, n);
        if (basis_push(&x, I, &cap, &g, A)) free(A);
        else { gp_free(&g); gp_free_n(A, (uint64_t)n); }
    }
    if (!x.rc && complete) { buchberger(&x, I, &cap); if (!x.rc) interreduce(&x, I); }
    if (x.rc) { tr_geom_ideal_free(I); set_error(error, x.rc); return NULL; }
    I->steps = x.iter - I->pairs;
    I->max_terms = x.peak; I->max_degree = x.peak_degree;
    uint64_t bytes = sizeof *I;
    for (int32_t j = 0; j < I->nb; j++) {
        bytes += I->basis[j].n * sizeof(gterm);
        for (int32_t i = 0; i < n; i++) { I->cofactor_terms += I->cof[(uint64_t)j * n + i].n; bytes += I->cof[(uint64_t)j * n + i].n * sizeof(gterm); }
    }
    I->bytes = bytes;
    set_error(error, TR_MATH_OK);
    return I;
}
int32_t tr_geom_ideal_get_info(const tr_geom_ideal *I, tr_geom_ideal_info *out) {
    if (!ideal_valid(I) || !out) return TR_MATH_INVALID;
    memset(out, 0, sizeof *out);
    out->nrelations = I->nrel; out->nvars = I->order.nv; out->kind = I->order.kind; out->completed = I->completed;
    for (int32_t i = 0; i < NV; i++) out->col[i] = i < I->order.nv ? I->order.col[I->order.sig[i]] : -1;
    out->nbasis = I->nb; out->policy = I->policy; out->weakest = I->weakest;
    out->pairs = I->pairs; out->reductions = I->steps; out->max_terms = I->max_terms; out->max_degree = I->max_degree;
    out->cofactor_terms = I->cofactor_terms; out->bytes = I->bytes;
    return TR_MATH_OK;
}
tr_geom_pol *tr_geom_ideal_element(const tr_geom_ideal *I, int32_t i, int32_t cofactor, int32_t *error) {
    if (!ideal_valid(I) || i < 0 || i >= I->nb || cofactor < 0 || cofactor > I->nrel) { set_error(error, TR_MATH_INVALID); return NULL; }
    const gpoly *p = cofactor ? &I->cof[(uint64_t)i * I->nrel + (cofactor - 1)] : &I->basis[i];
    return pol_from(&I->order, p, error);
}

/* ================================================================ reductions */
struct tr_geom_reduction_s {
    uint32_t magic;
    int32_t nrel, policy, claimable, code;
    gorder order;                    /* over the claim columns (increasing scope columns) */
    gpoly f, g, *h;
    tr_cert **certs;                 /* the relations' certificates (owned clones) */
    int64_t D;
    uint8_t *pe; int64_t *pc; uint64_t np;      /* P = D (f - g), canonical over the claim columns */
    uint8_t *qe; int64_t *qc; uint64_t *qoff;   /* the cofactor claims D h_i, concatenated (qoff: nrel + 1) */
    uint64_t steps;
};
static int red_valid(const tr_geom_reduction *r) { return r && r->magic == RED_MAGIC; }
void tr_geom_reduction_free(tr_geom_reduction *r) {
    if (!red_valid(r)) return;
    gp_free(&r->f); gp_free(&r->g);
    gp_free_n(r->h, (uint64_t)r->nrel);
    for (int32_t i = 0; r->certs && i < r->nrel; i++) tr_cert_free(r->certs[i]);
    free(r->certs);
    free(r->pe); free(r->pc); free(r->qe); free(r->qc); free(r->qoff);
    r->magic = 0;
    free(r);
}
/* the certificate's integer parts (claimable, or the code that refuses them) */
static void red_claims(gctx *x, tr_geom_reduction *R) {
    const int32_t nv = R->order.nv;
    int64_t D = gp_den(&R->f, 1);
    D = gp_den(&R->g, D);
    for (int32_t i = 0; i < R->nrel; i++) D = gp_den(&R->h[i], D);
    R->D = D;
    if (!D) { R->code = TR_MATH_OVERFLOW; return; }
    /* P = D (f - g) */
    gpoly diff = {NULL, 0, 0};
    const uint8_t zero[NV] = {0};
    if (!gp_axpy(x, &R->order, &R->f, Qi(-1), zero, &R->g, &diff)) { gp_free(&diff); return; }
    uint64_t total = 0;
    for (int32_t i = 0; i < R->nrel; i++) total += R->h[i].n;
    if (diff.n > TR_CERT_MAX_TERMS) { R->code = TR_MATH_UNSUPPORTED; gp_free(&diff); return; }
    for (int32_t i = 0; i < R->nrel; i++) if (R->h[i].n > TR_CERT_MAX_TERMS) { R->code = TR_MATH_UNSUPPORTED; gp_free(&diff); return; }
    int64_t *coef = (int64_t *)zalloc((diff.n ? diff.n : 1) * 8);
    uint8_t *ex = (uint8_t *)zalloc((diff.n ? diff.n : 1) * (uint64_t)nv);
    R->pe = (uint8_t *)zalloc((diff.n ? diff.n : 1) * (uint64_t)nv); R->pc = (int64_t *)zalloc((diff.n ? diff.n : 1) * 8);
    R->qe = (uint8_t *)zalloc((total ? total : 1) * (uint64_t)nv); R->qc = (int64_t *)zalloc((total ? total : 1) * 8);
    R->qoff = (uint64_t *)zalloc(((uint64_t)R->nrel + 1) * 8);
    if (!coef || !ex || !R->pe || !R->pc || !R->qe || !R->qc || !R->qoff) { free(coef); free(ex); gp_free(&diff); gfail(x, TR_MATH_MEMORY); return; }
    if (!gp_integer(&diff, D, coef)) { R->code = TR_MATH_OVERFLOW; free(coef); free(ex); gp_free(&diff); return; }
    for (uint64_t t = 0; t < diff.n; t++) memcpy(ex + t * nv, diff.t[t].e, (size_t)nv);
    uint64_t m = 0;
    int32_t rc = tr_cert_poly_canonical(nv, diff.n, ex, coef, 0, 0, R->pe, R->pc, &m);
    free(coef); free(ex);
    gp_free(&diff);
    if (rc) { R->code = rc; return; }
    R->np = m;
    uint64_t at = 0;
    for (int32_t i = 0; i < R->nrel; i++) {
        R->qoff[i] = at;
        if (!gp_integer(&R->h[i], D, R->qc + at)) { R->code = TR_MATH_OVERFLOW; return; }
        for (uint64_t t = 0; t < R->h[i].n; t++) memcpy(R->qe + (at + t) * nv, R->h[i].t[t].e, (size_t)nv);
        at += R->h[i].n;
    }
    R->qoff[R->nrel] = at;
    R->claimable = 1;
}
tr_geom_reduction *tr_geom_reduce(const tr_geom_ideal *I, const tr_geom_poly *f, const tr_math_budget *budget,
                                  tr_math_refusal *refusal, int32_t *error) {
    tr_math_budget store;
    gctx x;
    gctx_init(&x, budget, refusal, &store);
    if (!ideal_valid(I)) { set_error(error, TR_MATH_INVALID); return NULL; }
    int32_t rc = poly_check(f);
    if (rc) { set_error(error, rc); return NULL; }
    if (f->nterms > x.max_terms) { gbudget(&x, TR_BUDGET_SUPPORT, f->nterms, x.max_terms); set_error(error, x.rc); return NULL; }
    int32_t cols[NV], nc = 0;
    for (int32_t i = 0; i < I->order.nv; i++) cols[nc++] = I->order.col[i];
    rc = add_cols(cols, &nc, f->col, f->ncols);
    if (rc) { set_error(error, rc); return NULL; }
    tr_geom_reduction *R = (tr_geom_reduction *)zalloc(sizeof *R);
    if (!R) { set_error(error, TR_MATH_MEMORY); return NULL; }
    R->magic = RED_MAGIC; R->nrel = I->nrel; R->policy = I->policy;
    tr_geom_order spec;
    memset(&spec, 0, sizeof spec);
    spec.kind = I->order.kind; spec.nvars = I->order.nv;
    for (int32_t i = 0; i < I->order.nv; i++) spec.col[i] = I->order.col[I->order.sig[i]];
    order_make(&spec, cols, nc, &R->order);
    R->h = (gpoly *)zalloc((uint64_t)I->nrel * sizeof *R->h);
    R->certs = (tr_cert **)zalloc((uint64_t)I->nrel * sizeof *R->certs);
    gpoly *B = (gpoly *)zalloc((uint64_t)(I->nb ? I->nb : 1) * sizeof *B);
    gpoly *q = (gpoly *)zalloc((uint64_t)(I->nb ? I->nb : 1) * sizeof *q);
    gpoly prod = {NULL, 0, 0}, A = {NULL, 0, 0};
    if (!R->h || !R->certs || !B || !q) gfail(&x, TR_MATH_MEMORY);
    for (int32_t i = 0; i < I->nrel && !x.rc; i++) {
        int32_t e = 0;
        R->certs[i] = tr_cert_clone(I->certs[i], &e);
        if (!R->certs[i]) gfail(&x, e);
    }
    poly_in(&x, &R->order, f, &R->f);
    gpoly *G[TR_GEOM_MAX_BASIS];
    for (int32_t j = 0; j < I->nb && !x.rc; j++) { gp_remap(&x, &I->order, &I->basis[j], &R->order, &B[j]); G[j] = &B[j]; }
    if (!x.rc) gp_reduce(&x, &R->order, &R->f, G, I->nb, q, &R->g);
    /* h_i = sum_j q_j A_ji */
    for (int32_t j = 0; j < I->nb && !x.rc; j++) {
        if (!q[j].n) continue;
        for (int32_t i = 0; i < I->nrel && !x.rc; i++) {
            const gpoly *Aji = &I->cof[(uint64_t)j * I->nrel + i];
            if (!Aji->n) continue;
            if (gp_remap(&x, &I->order, Aji, &R->order, &A) && gp_mul(&x, &R->order, &q[j], &A, &prod)) {
                const uint8_t zero[NV] = {0};
                gp_addmul(&x, &R->order, &R->h[i], Qi(1), zero, &prod);
            }
        }
    }
    gp_free(&prod); gp_free(&A);
    gp_free_n(B, (uint64_t)(I->nb ? I->nb : 1));
    gp_free_n(q, (uint64_t)(I->nb ? I->nb : 1));
    if (!x.rc) red_claims(&x, R);
    if (x.rc) { tr_geom_reduction_free(R); set_error(error, x.rc); return NULL; }
    R->steps = x.iter;
    set_error(error, TR_MATH_OK);
    return R;
}
int32_t tr_geom_reduction_get_info(const tr_geom_reduction *R, tr_geom_reduction_info *out) {
    if (!red_valid(R) || !out) return TR_MATH_INVALID;
    memset(out, 0, sizeof *out);
    out->nvars = R->order.nv; out->nrelations = R->nrel;
    for (int32_t i = 0; i < NV; i++) out->col[i] = i < R->order.nv ? R->order.col[i] : -1;
    out->f_terms = R->f.n; out->g_terms = R->g.n;
    for (int32_t i = 0; i < R->nrel; i++) out->cofactor_terms += R->h[i].n;
    out->steps = R->steps; out->denominator = R->D; out->claimable = R->claimable; out->code = R->code;
    return TR_MATH_OK;
}
tr_geom_pol *tr_geom_reduction_part(const tr_geom_reduction *R, int32_t part, int32_t *error) {
    if (!red_valid(R) || part < 0 || part >= TR_GEOM_PART_COFACTOR + R->nrel) { set_error(error, TR_MATH_INVALID); return NULL; }
    const gpoly *p = part == TR_GEOM_PART_F ? &R->f : part == TR_GEOM_PART_G ? &R->g : &R->h[part - TR_GEOM_PART_COFACTOR];
    return pol_from(&R->order, p, error);
}
static void claim_set(tr_cert_claim *c, const gorder *o, int32_t policy, uint64_t n, const uint8_t *e, const int64_t *k) {
    memset(c, 0, sizeof *c);
    c->kind = TR_CLAIM_POLYNOMIAL; c->policy = policy; c->ncols = o->nv; c->carrier = TR_CARRIER_I128;
    for (int32_t i = 0; i < TR_CERT_MAX_COLS; i++) c->col[i] = i < o->nv ? o->col[i] : -1;
    c->nterms = n; c->exponents = n ? e : NULL; c->coefficients = n ? k : NULL;
}
int32_t tr_geom_reduction_claims(const tr_geom_reduction *R, tr_cert_claim *claim, tr_cert_claim *cofactors, int32_t capacity) {
    if (!red_valid(R) || !claim || (R->nrel && (!cofactors || capacity < R->nrel))) return TR_MATH_INVALID;
    if (!R->claimable) return R->code ? R->code : TR_MATH_INVALID;
    claim_set(claim, &R->order, R->policy, R->np, R->pe, R->pc);
    for (int32_t i = 0; i < R->nrel; i++)
        claim_set(&cofactors[i], &R->order, R->policy, R->qoff[i + 1] - R->qoff[i], R->qe + R->qoff[i] * (uint64_t)R->order.nv,
                  R->qc + R->qoff[i]);
    return TR_MATH_OK;
}
tr_cert *tr_geom_certify(const tr_geom_reduction *R, const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    if (!red_valid(R)) { set_error(error, TR_MATH_INVALID); return NULL; }
    tr_cert_claim claim, cof[TR_GEOM_MAX_RELATIONS];
    int32_t rc = tr_geom_reduction_claims(R, &claim, cof, TR_GEOM_MAX_RELATIONS);
    if (rc) { set_error(error, rc); return NULL; }
    return tr_cert_derive(&claim, (const tr_cert *const *)R->certs, cof, R->nrel, budget, refusal, error);
}

/* ================================================================ costs */
/* bits of sum_t |c_t| prod_v M_v^e (multiword, saturating at 640 bits) */
static uint64_t bound_bits(const tr_scope_view *v, int32_t ncols, const int32_t *cols, const uint8_t *e,
                           const int64_t *c, uint64_t nt, int *saturated) {
    enum { WW = 10 };
    uint64_t total[WW] = {0}, term[WW];
    int sat = 0;
    for (uint64_t t = 0; t < nt && !sat; t++) {
        memset(term, 0, sizeof term);
        term[0] = tr_abs_u64(c[t]);
        for (int32_t i = 0; i < ncols && !sat; i++) {
            uint64_t m = mag_of(v, cols[i]);
            for (uint8_t q = 0; q < e[t * (uint64_t)ncols + i] && !sat; q++) {
                unsigned __int128 carry = 0;
                for (int32_t w = 0; w < WW; w++) { unsigned __int128 y = (unsigned __int128)term[w] * m + carry; term[w] = (uint64_t)y; carry = y >> 64; }
                if (carry || term[WW - 1]) sat = 1;
            }
        }
        uint64_t cy = 0;
        for (int32_t w = 0; w < WW && !sat; w++) {
            uint64_t s = total[w] + term[w], c1 = s < term[w], s2 = s + cy, c2 = s2 < cy;
            total[w] = s2; cy = c1 | c2;
        }
        if (cy || total[WW - 1]) sat = 1;
    }
    *saturated = sat;
    if (sat) return (uint64_t)64 * WW;
    for (int32_t w = WW; w-- > 0;) if (total[w]) return (uint64_t)w * 64 + (uint64_t)(64 - __builtin_clzll(total[w]));
    return 0;
}
/* the integer form D p of a tr_geom_poly (owned arrays) */
static int32_t integer_form(const tr_geom_poly *f, int64_t *D, int64_t **coef) {
    int64_t l = 1;
    for (uint64_t t = 0; t < f->nterms && l; t++) {
        int64_t d = f->den ? f->den[t] : 1;
        if (d <= 0) return TR_MATH_INVALID;
        tr_num_rational q;
        if (!tr_q_make(f->num[t], d, &q)) return TR_MATH_OVERFLOW;
        l = lcm_den(l, q.denominator);
    }
    if (!l) return TR_MATH_OVERFLOW;
    int64_t *c = (int64_t *)zalloc((f->nterms ? f->nterms : 1) * 8);
    if (!c) return TR_MATH_MEMORY;
    for (uint64_t t = 0; t < f->nterms; t++) {
        int64_t d = f->den ? f->den[t] : 1;
        tr_num_rational q;
        if (!tr_q_make(f->num[t], d, &q)) { free(c); return TR_MATH_OVERFLOW; }
        __int128 v = (__int128)q.numerator * (l / q.denominator);
        if (v < INT64_MIN || v > INT64_MAX) { free(c); return TR_MATH_OVERFLOW; }
        c[t] = (int64_t)v;
    }
    *D = l; *coef = c;
    return TR_MATH_OK;
}
int32_t tr_geom_cost_of(const tr_scope *scope, const tr_geom_poly *f, tr_geom_cost *out) {
    tr_scope_view v;
    if (!scope || tr_scope_get_view(scope, &v) || !out) return TR_MATH_INVALID;
    int32_t rc = poly_check(f);
    if (rc) return rc;
    for (int32_t i = 0; i < f->ncols; i++) if (f->col[i] >= v.ncols) return TR_MATH_INVALID;
    memset(out, 0, sizeof *out);
    int64_t D = 1, *c = NULL;
    rc = integer_form(f, &D, &c);
    if (rc) return rc;
    uint32_t used = 0;
    for (uint64_t t = 0; t < f->nterms; t++) {
        if (!c[t]) continue;
        uint64_t d = 0;
        for (int32_t i = 0; i < f->ncols; i++) { uint8_t e = f->exponents[t * (uint64_t)f->ncols + i]; if (e) used |= UINT32_C(1) << i; d += e; }
        out->terms++;
        if (d > out->degree) out->degree = d;
        out->multiplications += (d > 1 ? d - 1 : 0) + (tr_abs_u64(c[t]) != 1 && d ? 1 : 0);
    }
    for (int32_t i = 0; i < f->ncols; i++) out->columns += (used >> i) & 1;
    int sat = 0;
    out->bound_bits = bound_bits(&v, f->ncols, f->col, f->exponents, c, f->nterms, &sat);
    out->denominator = D;
    free(c);
    return TR_MATH_OK;
}
int32_t tr_geom_choose(const tr_geom_cost *costs, int32_t n, int32_t priority, int32_t *chosen) {
    if (!costs || n < 1 || !chosen || priority < TR_GEOM_PRIORITY_COLUMNS || priority > TR_GEOM_PRIORITY_TERMS) return TR_MATH_INVALID;
    int32_t best = 0;
    for (int32_t i = 1; i < n; i++) {
        const tr_geom_cost *a = &costs[i], *b = &costs[best];
        uint64_t ka[3], kb[3];
        uint64_t ba = a->measured ? a->actual_bits : a->bound_bits, bb = b->measured ? b->actual_bits : b->bound_bits;
        uint64_t va[3] = {a->columns, ba, a->terms}, vb[3] = {b->columns, bb, b->terms};
        int first = priority - 1;
        ka[0] = va[first]; kb[0] = vb[first];
        int m = 1;
        for (int j = 0; j < 3; j++) if (j != first) { ka[m] = va[j]; kb[m] = vb[j]; m++; }
        int less = 0;
        for (int j = 0; j < 3; j++) if (ka[j] != kb[j]) { less = ka[j] < kb[j]; break; }
        if (less) best = i;
    }
    *chosen = best;
    return TR_MATH_OK;
}

/* ================================================================ agreement on rows */
struct tr_geom_pairs_s {
    uint32_t magic;
    int32_t npairs;
    tr_geom_pair_view view;
    uint64_t *offset; uint32_t *support; int32_t *words; uint8_t *e; int64_t *c;
};
static int pairs_valid(const tr_geom_pairs *p) { return p && p->magic == PAIRS_MAGIC; }
void tr_geom_pairs_free(tr_geom_pairs *p) {
    if (!pairs_valid(p)) return;
    free(p->offset); free(p->support); free(p->words); free(p->e); free(p->c);
    p->magic = 0;
    free(p);
}
static int32_t words_of(uint64_t bits) {
    for (int32_t W = 2; W <= TR_IDENT_MAX_WORDS; W++) if (bits <= (uint64_t)(64 * W - 1)) return W;
    return 0;
}
tr_geom_pairs *tr_geom_pairs_make(const tr_scope *scope, const tr_geom_poly *f, const tr_geom_poly *g, int32_t n,
                                  const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    tr_scope_view v;
    if (!scope || tr_scope_get_view(scope, &v) || n < 0 || (n && (!f || !g))) { set_error(error, TR_MATH_INVALID); return NULL; }
    if (n > TR_GEOM_MAX_PAIRS) { set_error(error, TR_MATH_UNSUPPORTED); return NULL; }
    int32_t cols[NV], nc = 0, rc = TR_MATH_OK;
    for (int32_t j = 0; j < n && !rc; j++) {
        rc = poly_check(&f[j]);
        if (!rc) rc = poly_check(&g[j]);
        if (!rc) rc = add_cols(cols, &nc, f[j].col, f[j].ncols);
        if (!rc) rc = add_cols(cols, &nc, g[j].col, g[j].ncols);
    }
    for (int32_t i = 0; !rc && i < nc; i++)
        if (cols[i] >= v.ncols || (v.rows && !v.col[cols[i]].codes && !v.col[cols[i]].values)) rc = TR_MATH_INVALID;
    if (!rc && !nc && n) rc = TR_MATH_INVALID;
    if (rc) { set_error(error, rc); return NULL; }
    tr_math_budget store;
    gctx x;
    gctx_init(&x, budget, refusal, &store);
    gorder o;
    tr_geom_order spec;
    memset(&spec, 0, sizeof spec);
    spec.kind = TR_GEOM_DEGLEX;
    order_make(&spec, cols, nc, &o);
    gpoly *F = (gpoly *)zalloc((uint64_t)(n ? 3 * n : 1) * sizeof *F);
    uint32_t *sup = (uint32_t *)zalloc((uint64_t)(n ? n : 1) * 4);
    int64_t (*Ds)[3] = (int64_t (*)[3])zalloc((uint64_t)(n ? n : 1) * sizeof *Ds);
    if (!F || !sup || !Ds) gfail(&x, TR_MATH_MEMORY);
    uint64_t total = 0;
    for (int32_t j = 0; j < n && !x.rc; j++) {
        gpoly a = {NULL, 0, 0}, b = {NULL, 0, 0};
        poly_in(&x, &o, &f[j], &a);
        poly_in(&x, &o, &g[j], &b);
        int64_t Df = gp_den(&a, 1), Dg = gp_den(&b, 1), D = lcm_den(Df, Dg);
        if (!Df || !Dg || !D) gfail(&x, TR_MATH_OVERFLOW);
        const uint8_t zero[NV] = {0};
        gpoly p = {NULL, 0, 0};
        gp_axpy(&x, &o, &a, Qi(-1), zero, &b, &p);
        F[3 * j] = a; F[3 * j + 1] = b; F[3 * j + 2] = p;
        Ds[j][0] = Df; Ds[j][1] = Dg; Ds[j][2] = D;
        for (int k = 0; k < 2; k++) {
            const tr_geom_poly *s = k ? &g[j] : &f[j];
            for (int32_t i = 0; i < s->ncols; i++) {
                int used = 0;
                for (uint64_t t = 0; t < s->nterms; t++) used |= s->exponents[t * (uint64_t)s->ncols + i] != 0;
                if (used) sup[j] |= UINT32_C(1) << col_pos(&o, s->col[i]);
            }
        }
        if (!sup[j]) sup[j] = nc >= 32 ? UINT32_MAX : (uint32_t)((UINT64_C(1) << nc) - 1);
        total += a.n + b.n + p.n;
    }
    tr_geom_pairs *P = NULL;
    if (!x.rc) {
        uint64_t bytes = tr_budget_add(sizeof *P, tr_budget_mul(total, (uint64_t)nc + 8));
        int32_t c2 = check_one(budget, TR_BUDGET_BYTES, bytes, refusal);
        if (c2) x.rc = c2;
    }
    if (!x.rc) {
        P = (tr_geom_pairs *)zalloc(sizeof *P);
        if (P) {
            P->magic = PAIRS_MAGIC; P->npairs = n;
            uint64_t nq = n ? (uint64_t)3 * (uint64_t)n : 1;
            P->offset = (uint64_t *)zalloc(((uint64_t)3 * n + 1) * 8); P->support = (uint32_t *)zalloc(nq * 4);
            P->words = (int32_t *)zalloc(nq * 4);
            P->e = (uint8_t *)zalloc((total ? total : 1) * (uint64_t)nc); P->c = (int64_t *)zalloc((total ? total : 1) * 8);
        }
        if (!P || !P->offset || !P->support || !P->words || !P->e || !P->c) gfail(&x, TR_MATH_MEMORY);
    }
    uint64_t at = 0;
    for (int32_t q = 0; q < 3 * n && !x.rc; q++) {
        const gpoly *s = &F[q];
        P->offset[q] = at;
        if (!gp_integer(s, Ds[q / 3][q % 3], P->c + at)) { gfail(&x, TR_MATH_OVERFLOW); break; }
        for (uint64_t t = 0; t < s->n; t++) memcpy(P->e + (at + t) * nc, s->t[t].e, (size_t)nc);
        int sat = 0;
        uint64_t bits = bound_bits(&v, nc, cols, P->e + at * nc, P->c + at, s->n, &sat);
        P->words[q] = sat ? 0 : words_of(bits);
        if (!P->words[q]) gfail(&x, TR_MATH_UNSUPPORTED);         /* past 512 bits: not evaluated here */
        P->support[q] = sup[q / 3];
        at += s->n;
    }
    if (P && P->offset) P->offset[3 * n] = at;
    for (int32_t q = 0; F && q < 3 * n; q++) gp_free(&F[q]);
    free(F); free(sup); free(Ds);
    if (x.rc) { tr_geom_pairs_free(P); set_error(error, x.rc); return NULL; }
    tr_ident_cand_view *cv = &P->view.cand;
    P->view.npairs = n;
    cv->ncand = 3 * n; cv->nvars = nc;
    for (int32_t i = 0; i < TR_IDENT_MAX_VARS; i++) cv->var[i] = i < nc ? cols[i] : -1;
    cv->nterms = at; cv->offset = P->offset; cv->support = P->support; cv->words = P->words;
    cv->exponents = P->e; cv->coefficients = P->c;
    set_error(error, TR_MATH_OK);
    return P;
}
int32_t tr_geom_pairs_get_view(const tr_geom_pairs *p, tr_geom_pair_view *out) {
    if (!pairs_valid(p) || !out) return TR_MATH_INVALID;
    *out = p->view;
    return TR_MATH_OK;
}
static int pair_view_ok(const tr_scope_view *v, const tr_geom_pair_view *p) {
    if (!p || p->npairs < 0 || p->cand.ncand != 3 * p->npairs || p->cand.nvars < 1 || p->cand.nvars > NV) return 0;
    for (int32_t i = 0; i < p->cand.nvars; i++) {
        int32_t c = p->cand.var[i];
        if (c < 0 || c >= v->ncols || (v->rows && !v->col[c].codes && !v->col[c].values)) return 0;
    }
    if (p->npairs && (!p->cand.offset || !p->cand.support || !p->cand.words || (p->cand.nterms && (!p->cand.exponents || !p->cand.coefficients))))
        return 0;
    if (p->npairs && (p->cand.offset[0] || p->cand.offset[p->cand.ncand] != p->cand.nterms)) return 0;
    for (int32_t j = 0; j < p->cand.ncand; j++)
        if (p->cand.offset[j] > p->cand.offset[j + 1] || !p->cand.support[j] || (p->cand.nvars < 32 && (p->cand.support[j] >> p->cand.nvars)) ||
            p->cand.words[j] < 2 || p->cand.words[j] > TR_IDENT_MAX_WORDS) return 0;
    return 1;
}
int32_t tr_geom_agree_need(const tr_scope_view *v, const tr_geom_pair_view *p, int32_t target, tr_math_need *need) {
    if (!view_ok(v) || !need || !pair_view_ok(v, p)) return TR_MATH_INVALID;
    if (target == TR_IDENT_NEED_DEVICE) { tr_geom_agree_device_need(v, p, need); return TR_MATH_OK; }
    if (target != TR_IDENT_NEED_CPU) return TR_MATH_INVALID;
    for (int i = 0; i < TR_BUDGET_RESOURCES; i++) need->need[i] = 0;
    need->need[TR_BUDGET_VARIABLES] = (uint64_t)p->cand.nvars;
    need->need[TR_BUDGET_SUPPORT] = p->cand.nterms;
    need->need[TR_BUDGET_OUTPUT] = (uint64_t)p->npairs;
    return TR_MATH_OK;
}
int32_t tr_geom_agree_view(const tr_scope_view *v, const tr_geom_pair_view *p, const tr_math_budget *budget,
                           tr_geom_agreement *out, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!view_ok(v) || !pair_view_ok(v, p) || (p->npairs && !out)) return TR_MATH_INVALID;
    tr_math_need need;
    tr_geom_agree_need(v, p, TR_IDENT_NEED_CPU, &need);
    tr_math_budget store;
    int32_t rc = tr_budget_check(budget_or(budget, &store), &need, refusal);
    if (rc) return rc;
    for (int32_t j = 0; j < p->npairs; j++) {
        memset(&out[j], 0, sizeof out[j]);
        out[j].rows = v->rows; out[j].selected = v->selected; out[j].witness = UINT64_MAX;
    }
    const uint64_t nw = (v->rows + 63) / 64;
    uint64_t val[TR_IDENT_MAX_WORDS];
    for (uint64_t w = 0; w < nw; w++) {
        uint64_t bits = v->selection[w];
        while (bits) {
            uint64_t r = w * 64 + (uint64_t)__builtin_ctzll(bits);
            bits &= bits - 1;
            for (int32_t j = 0; j < p->npairs; j++) {
                tr_geom_agreement *a = &out[j];
                int st = tr_ident_cand_row(v, &p->cand, 3 * j + 2, r);   /* P = D (f - g) */
                if (st == TR_CERT_ROW_EXCLUDED) { a->excluded++; continue; }
                if (st == TR_CERT_ROW_OVERFLOW) { a->undecided++; continue; }
                a->checked++;
                if (st == TR_CERT_ROW_VIOLATES) { a->disagreements++; if (r < a->witness) a->witness = r; }
                tr_ident_cand_value(v, &p->cand, 3 * j, r, p->cand.words[3 * j], val);
                uint32_t bf = tr_geom_bits(val, p->cand.words[3 * j]);
                tr_ident_cand_value(v, &p->cand, 3 * j + 1, r, p->cand.words[3 * j + 1], val);
                uint32_t bg = tr_geom_bits(val, p->cand.words[3 * j + 1]);
                if (bf > a->bits_f) a->bits_f = bf;
                if (bg > a->bits_g) a->bits_g = bg;
            }
        }
    }
    return TR_MATH_OK;
}

/* ================================================================ batched certificate checks */
struct tr_geom_checks_s {
    uint32_t magic;
    int32_t ncert;
    tr_geom_check_view view;
    int32_t *nvars, *npremises, *maxdeg, *overflow, *cols;     /* cols: ncert x NV */
    uint64_t *poly0, *slot0, *work0, *term0;
    uint8_t *e; int64_t *c;
};
static int checks_valid(const tr_geom_checks *k) { return k && k->magic == CHECKS_MAGIC; }
void tr_geom_checks_free(tr_geom_checks *k) {
    if (!checks_valid(k)) return;
    free(k->nvars); free(k->npremises); free(k->maxdeg); free(k->overflow); free(k->cols);
    free(k->poly0); free(k->slot0); free(k->work0); free(k->term0); free(k->e); free(k->c);
    k->magic = 0;
    free(k);
}
/* a claim's terms over the certificate's columns (FND-007's reading), appended; *ok = 0 for an int64 overflow */
static int32_t check_terms(const tr_cert_claim *k, const int32_t *cols, int32_t nv, uint8_t *e, int64_t *c, uint64_t cap,
                           uint64_t *n, int *ok) {
    int32_t at[TR_CERT_MAX_COLS];
    if (k->ncols < 1 || k->ncols > TR_CERT_MAX_COLS) return TR_MATH_INVALID;
    if ((k->kind == TR_CLAIM_AFFINE && k->ncols != 2) ||
        ((k->kind == TR_CLAIM_SUM || k->kind == TR_CLAIM_PRODUCT) && k->ncols != 3)) return TR_MATH_INVALID;
    for (int32_t i = 0; i < k->ncols; i++) {
        at[i] = -1;
        for (int32_t j = 0; j < nv; j++) if (cols[j] == k->col[i]) at[i] = j;
        if (at[i] < 0) return TR_MATH_INVALID;                   /* a column outside the claim's */
    }
#define CPUSH(val) do { if (*n >= cap) return TR_MATH_INVALID; memset(e + *n * NV, 0, NV); c[*n] = (val); } while (0)
    switch (k->kind) {
    case TR_CLAIM_AFFINE:
        CPUSH(1); e[*n * NV + at[0]] = 1; (*n)++;
        if (k->scale) { if (k->scale == INT64_MIN) *ok = 0; CPUSH(k->scale == INT64_MIN ? 0 : -k->scale); e[*n * NV + at[1]] = 1; (*n)++; }
        if (k->bias) { if (k->bias == INT64_MIN) *ok = 0; CPUSH(k->bias == INT64_MIN ? 0 : -k->bias); (*n)++; }
        return TR_MATH_OK;
    case TR_CLAIM_SUM:
        CPUSH(1); e[*n * NV + at[0]] = 1; (*n)++;
        CPUSH(-1); e[*n * NV + at[1]] += 1; (*n)++;
        CPUSH(-1); e[*n * NV + at[2]] += 1; (*n)++;
        return TR_MATH_OK;
    case TR_CLAIM_PRODUCT:
        CPUSH(1); e[*n * NV + at[0]] = 1; (*n)++;
        CPUSH(-1); e[*n * NV + at[1]] += 1; e[*n * NV + at[2]] += 1; (*n)++;
        return TR_MATH_OK;
    case TR_CLAIM_POLYNOMIAL:
        if (k->nterms && (!k->exponents || !k->coefficients)) return TR_MATH_INVALID;
        for (uint64_t t = 0; t < k->nterms; t++) {
            CPUSH(k->coefficients[t]);
            for (int32_t i = 0; i < k->ncols; i++) e[*n * NV + at[i]] = k->exponents[t * (uint64_t)k->ncols + (uint64_t)i];
            (*n)++;
        }
        return TR_MATH_OK;
    default:
        return TR_MATH_INVALID;
    }
#undef CPUSH
}
static uint64_t claim_terms(const tr_cert_claim *k) {
    return k->kind == TR_CLAIM_POLYNOMIAL ? k->nterms : k->kind == TR_CLAIM_AFFINE ? 3 : k->kind == TR_CLAIM_SUM ? 3 : 2;
}
tr_geom_checks *tr_geom_checks_make(int32_t ncert, const tr_cert_claim *claims, const int32_t *npremises,
                                    const tr_cert_claim *premises, const tr_cert_claim *cofactors,
                                    const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    if (ncert < 0 || (ncert && (!claims || !npremises))) { set_error(error, TR_MATH_INVALID); return NULL; }
    if (ncert > TR_GEOM_MAX_CHECKS) { set_error(error, TR_MATH_UNSUPPORTED); return NULL; }
    uint64_t nprem = 0, nterms = 0, npolys = 0;
    for (int32_t c = 0; c < ncert; c++) {
        if (npremises[c] < 1 || npremises[c] > TR_CERT_MAX_PREMISES) { set_error(error, TR_MATH_INVALID); return NULL; }
        nprem += (uint64_t)npremises[c];
    }
    if (nprem && (!premises || !cofactors)) { set_error(error, TR_MATH_INVALID); return NULL; }
    nprem = 0;
    for (int32_t c = 0; c < ncert; c++) {
        if (claims[c].kind != TR_CLAIM_POLYNOMIAL || claims[c].ncols < 1 || claims[c].ncols > NV ||
            claims[c].nterms > TR_CERT_MAX_TERMS || (claims[c].nterms && (!claims[c].exponents || !claims[c].coefficients))) {
            set_error(error, claims[c].ncols > NV || claims[c].nterms > TR_CERT_MAX_TERMS ? TR_MATH_UNSUPPORTED : TR_MATH_INVALID);
            return NULL;
        }
        nterms += claims[c].nterms;
        for (int32_t i = 0; i < npremises[c]; i++) {
            const tr_cert_claim *q = &cofactors[nprem + i], *g = &premises[nprem + i];
            if (q->kind != TR_CLAIM_POLYNOMIAL || q->nterms > TR_CERT_MAX_TERMS) { set_error(error, TR_MATH_INVALID); return NULL; }
            nterms += q->nterms + claim_terms(g);
        }
        nprem += (uint64_t)npremises[c];
        npolys += 1 + 2 * (uint64_t)npremises[c];
    }
    uint64_t bytes = tr_budget_add(tr_budget_mul(nterms, NV + 8), tr_budget_mul(npolys + (uint64_t)ncert + 2, 48));
    int32_t rc = check_one(budget, TR_BUDGET_BYTES, bytes, refusal);
    if (!rc) rc = check_one(budget, TR_BUDGET_OUTPUT, (uint64_t)ncert, refusal);
    if (rc) { set_error(error, rc); return NULL; }
    tr_geom_checks *K = (tr_geom_checks *)zalloc(sizeof *K);
    if (K) {
        K->magic = CHECKS_MAGIC; K->ncert = ncert;
        uint64_t n1 = (uint64_t)(ncert ? ncert : 1);
        K->nvars = (int32_t *)zalloc(n1 * 4); K->npremises = (int32_t *)zalloc(n1 * 4); K->maxdeg = (int32_t *)zalloc(n1 * 4);
        K->overflow = (int32_t *)zalloc(n1 * 4); K->cols = (int32_t *)zalloc(n1 * NV * 4);
        K->poly0 = (uint64_t *)zalloc(n1 * 8); K->slot0 = (uint64_t *)zalloc((n1 + 1) * 8); K->work0 = (uint64_t *)zalloc((n1 + 1) * 8);
        K->term0 = (uint64_t *)zalloc((npolys + 1) * 8);
        K->e = (uint8_t *)zalloc((nterms ? nterms : 1) * NV); K->c = (int64_t *)zalloc((nterms ? nterms : 1) * 8);
    }
    if (!K || !K->nvars || !K->npremises || !K->maxdeg || !K->overflow || !K->cols || !K->poly0 || !K->slot0 || !K->work0 ||
        !K->term0 || !K->e || !K->c) { tr_geom_checks_free(K); set_error(error, TR_MATH_MEMORY); return NULL; }
    uint64_t t = 0, pidx = 0, base = 0, slots = 0, work = 0;
    for (int32_t c = 0; c < ncert && !rc; c++) {
        const tr_cert_claim *P = &claims[c];
        int32_t nv = P->ncols, *cols = K->cols + (uint64_t)c * NV;
        for (int32_t i = 0; i < NV; i++) cols[i] = i < nv ? P->col[i] : -1;
        for (int32_t i = 0; i < nv && !rc; i++) for (int32_t j = 0; j < i; j++) if (cols[j] == cols[i]) rc = TR_MATH_INVALID;
        K->nvars[c] = nv; K->npremises[c] = npremises[c]; K->poly0[c] = pidx;
        int ok = 1;
        uint64_t start = t, deg = 0;
        unsigned __int128 mass = 0;
        K->term0[pidx++] = t;
        if (!rc) rc = check_terms(P, cols, nv, K->e, K->c, nterms, &t, &ok);
        for (uint64_t s = start; s < t; s++) {
            uint64_t d = 0;
            for (int i = 0; i < NV; i++) d += K->e[s * NV + i];
            if (d > deg) deg = d;
            mass += tr_abs_u64(K->c[s]);
        }
        uint64_t w = t - start;
        for (int32_t i = 0; i < npremises[c] && !rc; i++) {
            uint64_t q0 = t;
            K->term0[pidx++] = t;
            rc = check_terms(&cofactors[base + i], cols, nv, K->e, K->c, nterms, &t, &ok);
            if (rc) break;
            uint64_t g0 = t;
            K->term0[pidx++] = t;
            rc = check_terms(&premises[base + i], cols, nv, K->e, K->c, nterms, &t, &ok);
            if (rc) break;
            uint64_t dq = 0, dg = 0;
            unsigned __int128 mq = 0, mg = 0;
            for (uint64_t s = q0; s < g0; s++) { uint64_t d = 0; for (int k = 0; k < NV; k++) d += K->e[s * NV + k]; if (d > dq) dq = d; mq += tr_abs_u64(K->c[s]); }
            for (uint64_t s = g0; s < t; s++) { uint64_t d = 0; for (int k = 0; k < NV; k++) d += K->e[s * NV + k]; if (d > dg) dg = d; mg += tr_abs_u64(K->c[s]); }
            if (g0 > q0 && t > g0 && dq + dg > deg) deg = dq + dg;
            w += (g0 - q0) * (t - g0);
            /* (sum |q_a|) (sum |g_b|) bounds the products' magnitudes; an exponent sum past 255 cannot be ranked */
            unsigned __int128 prod = mq && mg ? (mq > I128_LIMIT / mg ? I128_LIMIT + 1 : mq * mg) : 0;
            if (prod > I128_LIMIT || mass > I128_LIMIT - prod) mass = I128_LIMIT + 1; else mass += prod;
            for (uint64_t a = q0; a < g0 && !rc; a++)
                for (uint64_t b = g0; b < t && !rc; b++)
                    for (int k = 0; k < NV; k++) if (K->e[a * NV + k] + K->e[b * NV + k] > 255) rc = TR_MATH_UNSUPPORTED;
        }
        base += (uint64_t)npremises[c];
        if (rc) break;
        if (deg > (uint64_t)INT32_MAX) deg = (uint64_t)INT32_MAX;
        K->maxdeg[c] = (int32_t)deg;
        uint64_t sc = deg + (uint64_t)nv > 4096 ? TR_BUDGET_UNKNOWN : tr_geom_slots(nv, deg);
        if (sc > TR_GEOM_MAX_SLOTS || deg > 2040) { rc = TR_MATH_UNSUPPORTED; break; }
        if (!ok || mass > I128_LIMIT) { K->overflow[c] = 1; w = 0; sc = 0; }   /* undecided: no expansion */
        K->slot0[c] = slots; K->work0[c] = work;
        slots = tr_budget_add(slots, sc); work = tr_budget_add(work, w);
    }
    K->term0[pidx] = t;
    K->slot0[ncert] = slots; K->work0[ncert] = work;
    if (!rc) rc = check_one(budget, TR_BUDGET_STATES, slots, refusal);
    if (rc) { tr_geom_checks_free(K); set_error(error, rc); return NULL; }
    tr_geom_check_view *v = &K->view;
    v->ncert = ncert; v->npolys = npolys; v->nterms = t; v->nslots = slots; v->nwork = work;
    v->nvars = K->nvars; v->npremises = K->npremises; v->maxdeg = K->maxdeg; v->overflow = K->overflow;
    v->poly0 = K->poly0; v->slot0 = K->slot0;
    v->work0 = K->work0; v->term0 = K->term0; v->exponents = K->e; v->coefficients = K->c;
    set_error(error, TR_MATH_OK);
    return K;
}
int32_t tr_geom_checks_get_view(const tr_geom_checks *K, tr_geom_check_view *out) {
    if (!checks_valid(K) || !out) return TR_MATH_INVALID;
    *out = K->view;
    return TR_MATH_OK;
}
int32_t tr_geom_checks_columns(const tr_geom_checks *K, int32_t c, int32_t *cols, int32_t capacity) {
    if (!checks_valid(K) || c < 0 || c >= K->ncert || !cols || capacity < K->nvars[c]) return TR_MATH_INVALID;
    for (int32_t i = 0; i < K->nvars[c]; i++) cols[i] = K->cols[(uint64_t)c * NV + i];
    return K->nvars[c];
}
static uint64_t checks_most(const tr_geom_checks *K) {
    const tr_geom_check_view *v = &K->view;
    uint64_t most = 0;
    for (int32_t c = 0; c < K->ncert; c++) if (v->slot0[c + 1] - v->slot0[c] > most) most = v->slot0[c + 1] - v->slot0[c];
    return most;
}
int32_t tr_geom_checks_need(const tr_geom_checks *K, int32_t target, tr_math_need *need) {
    if (!checks_valid(K) || !need) return TR_MATH_INVALID;
    if (target == TR_IDENT_NEED_DEVICE) { tr_geom_check_device_need(&K->view, need); return TR_MATH_OK; }
    if (target != TR_IDENT_NEED_CPU) return TR_MATH_INVALID;
    for (int i = 0; i < TR_BUDGET_RESOURCES; i++) need->need[i] = 0;
    const uint64_t most = checks_most(K);
    need->need[TR_BUDGET_SUPPORT] = K->view.nterms;
    need->need[TR_BUDGET_OUTPUT] = (uint64_t)K->ncert;
    need->need[TR_BUDGET_STATES] = most;
    need->need[TR_BUDGET_BYTES] = tr_budget_mul(most, 16);
    return TR_MATH_OK;
}
int32_t tr_geom_checks_run(const tr_geom_checks *K, const tr_math_budget *budget, tr_geom_check_result *out,
                           tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!checks_valid(K) || (K->ncert && !out)) return TR_MATH_INVALID;
    const tr_geom_check_view *v = &K->view;
    tr_math_need need;
    tr_geom_checks_need(K, TR_IDENT_NEED_CPU, &need);
    tr_math_budget store;
    int32_t rc = tr_budget_check(budget_or(budget, &store), &need, refusal);
    if (rc) return rc;
    const uint64_t most = need.need[TR_BUDGET_STATES];
    __int128 *slot = (__int128 *)zalloc((most ? most : 1) * sizeof(__int128));
    if (!slot) return TR_MATH_MEMORY;
    for (int32_t c = 0; c < K->ncert; c++) {
        tr_geom_check_result *r = &out[c];
        memset(r, 0, sizeof *r);
        r->slots = v->slot0[c + 1] - v->slot0[c]; r->work = v->work0[c + 1] - v->work0[c]; r->first = UINT64_MAX;
        r->overflow = K->overflow[c];
        if (r->overflow) continue;
        memset(slot, 0, (size_t)r->slots * sizeof(__int128));
        for (uint64_t w = 0; w < r->work; w++) {
            uint64_t rank;
            __int128 value;
            tr_geom_work(v, c, w, &rank, &value);
            slot[rank] += value;                                   /* proven: |partial sums| <= 2^127 - 1 */
        }
        for (uint64_t s = 0; s < r->slots; s++) if (slot[s]) { r->nonzero++; if (s < r->first) r->first = s; }
        r->holds = r->nonzero == 0;
    }
    free(slot);
    return TR_MATH_OK;
}
int32_t tr_geom_unrank(uint64_t rank, int32_t nv, uint8_t *e) {
    if (nv < 1 || nv > NV || !e) return TR_MATH_INVALID;
    uint64_t t = 0;
    while (tr_geom_slots(nv, t) <= rank) { t++; if (t > 2040) return TR_MATH_INVALID; }
    uint64_t within = rank - (t ? tr_geom_slots(nv, t - 1) : 0);
    uint64_t s[NV];
    for (int32_t i = nv - 2; i >= 0; i--) {                    /* the combinatorial number system, greedily */
        uint64_t x = (uint64_t)i;
        while (tr_geom_binom(x + 1, (uint32_t)i + 1) <= within) x++;
        s[i] = x;
        within -= tr_geom_binom(x, (uint32_t)i + 1);
    }
    for (int32_t i = 0; i < NV; i++) e[i] = 0;
    if (nv == 1) { e[0] = (uint8_t)t; return TR_MATH_OK; }
    e[0] = (uint8_t)s[0];
    for (int32_t i = 1; i <= nv - 2; i++) e[i] = (uint8_t)(s[i] - s[i - 1] - 1);
    e[nv - 1] = (uint8_t)(t - (s[nv - 2] - (uint64_t)(nv - 2)));
    return TR_MATH_OK;
}

/* ================================================================ derivatives and dependencies */
/* d p / d x_pos (formal) */
static int gp_partial(gctx *x, const gorder *o, const gpoly *p, int pos, gpoly *out) {
    out->n = 0;
    for (uint64_t t = 0; t < p->n && !x->rc; t++) {
        if (!p->t[t].e[pos]) continue;
        uint8_t e[NV];
        memcpy(e, p->t[t].e, NV);
        tr_num_rational c = QMUL(x, p->t[t].c, Qi(e[pos]));
        e[pos]--;
        gp_push(x, out, e, c);
    }
    return gp_normalize(x, o, out);
}
tr_geom_pol *tr_geom_derivative(const tr_geom_poly *f, int32_t column, int32_t through, const tr_geom_dep *deps,
                                int32_t ndeps, const tr_math_budget *budget, int32_t *reason, tr_math_refusal *refusal,
                                int32_t *error) {
    tr_math_budget store;
    gctx x;
    gctx_init(&x, budget, refusal, &store);
    int32_t why = TR_GEOM_DERIV_OK;
    if (reason) *reason = why;
    int32_t rc = poly_check(f);
    if (!rc && (column < 0 || column >= TR_SCOPE_MAX_COLUMNS || (through != 0 && through != 1) || ndeps < 0 ||
                (ndeps && !deps))) rc = TR_MATH_INVALID;
    if (!rc && !through && ndeps) rc = TR_MATH_INVALID;           /* a formal derivative takes no declarations */
    if (!rc && ndeps > TR_GEOM_MAX_DEPS) rc = TR_MATH_UNSUPPORTED;
    for (int32_t i = 0; i < ndeps && !rc; i++) {
        rc = poly_check(&deps[i].phi);
        if (!rc && (deps[i].column < 0 || deps[i].column >= TR_SCOPE_MAX_COLUMNS || deps[i].reserved)) rc = TR_MATH_INVALID;
        for (int32_t j = 0; j < i && !rc; j++) if (deps[j].column == deps[i].column) rc = TR_MATH_INVALID;
    }
    if (rc) { set_error(error, rc); return NULL; }
    if (through && !ndeps) { if (reason) *reason = TR_GEOM_DERIV_UNDECLARED; set_error(error, TR_MATH_DOMAIN_ERROR); return NULL; }
    int32_t cols[NV], nc = 0;
    rc = add_cols(cols, &nc, f->col, f->ncols);
    if (!rc) rc = add_cols(cols, &nc, &column, 1);
    for (int32_t i = 0; through && i < ndeps && !rc; i++) {
        rc = add_cols(cols, &nc, &deps[i].column, 1);
        if (!rc) rc = add_cols(cols, &nc, deps[i].phi.col, deps[i].phi.ncols);
    }
    if (rc) { set_error(error, rc); return NULL; }
    gorder o;
    tr_geom_order spec;
    memset(&spec, 0, sizeof spec);
    spec.kind = TR_GEOM_DEGLEX;
    order_make(&spec, cols, nc, &o);
    gpoly F = {NULL, 0, 0}, out = {NULL, 0, 0};
    poly_in(&x, &o, f, &F);
    const int u = col_pos(&o, column);
    if (!through) {
        gp_partial(&x, &o, &F, u, &out);
    } else {
        /* the declared graph: v -> the dependent columns its phi reads; acyclic (topological order) */
        int32_t pos[TR_GEOM_MAX_DEPS], order[TR_GEOM_MAX_DEPS], done[TR_GEOM_MAX_DEPS] = {0}, n_done = 0;
        gpoly phi[TR_GEOM_MAX_DEPS], dphi[TR_GEOM_MAX_DEPS];
        memset(phi, 0, sizeof phi); memset(dphi, 0, sizeof dphi);
        for (int32_t i = 0; i < ndeps && !x.rc; i++) {
            pos[i] = col_pos(&o, deps[i].column);
            if (deps[i].column == column) why = TR_GEOM_DERIV_DEPENDENT;
            poly_in(&x, &o, &deps[i].phi, &phi[i]);
            for (int32_t c = 0; c < deps[i].phi.ncols; c++) if (deps[i].phi.col[c] == deps[i].column) why = TR_GEOM_DERIV_CYCLE;
        }
        while (!x.rc && why == TR_GEOM_DERIV_OK && n_done < ndeps) {
            int progressed = 0;
            for (int32_t i = 0; i < ndeps; i++) {
                if (done[i]) continue;
                int ready = 1;
                for (int32_t j = 0; j < ndeps && ready; j++) {
                    if (done[j] || j == i) continue;
                    for (uint64_t t = 0; t < phi[i].n && ready; t++) if (phi[i].t[t].e[pos[j]]) ready = 0;
                }
                if (ready) { order[n_done++] = i; done[i] = 1; progressed = 1; }
            }
            if (!progressed) why = TR_GEOM_DERIV_CYCLE;
        }
        if (why != TR_GEOM_DERIV_OK) {
            for (int32_t i = 0; i < ndeps; i++) gp_free(&phi[i]);
            gp_free(&F);
            if (reason) *reason = why;
            set_error(error, x.rc ? x.rc : TR_MATH_DOMAIN_ERROR);
            return NULL;
        }
        /* d v_i / d u in dependency order: d phi_i/du + sum_j dphi_i/dv_j dv_j/du */
        gpoly tmp = {NULL, 0, 0}, part = {NULL, 0, 0};
        const uint8_t zero[NV] = {0};
        for (int32_t s = 0; s < ndeps && !x.rc; s++) {
            int32_t i = order[s];
            gp_partial(&x, &o, &phi[i], u, &dphi[i]);
            for (int32_t r = 0; r < s && !x.rc; r++) {
                int32_t j = order[r];
                if (gp_partial(&x, &o, &phi[i], pos[j], &part) && part.n && dphi[j].n && gp_mul(&x, &o, &part, &dphi[j], &tmp))
                    gp_addmul(&x, &o, &dphi[i], Qi(1), zero, &tmp);
            }
        }
        /* df/du + sum_i df/dv_i dv_i/du */
        gp_partial(&x, &o, &F, u, &out);
        for (int32_t i = 0; i < ndeps && !x.rc; i++)
            if (gp_partial(&x, &o, &F, pos[i], &part) && part.n && dphi[i].n && gp_mul(&x, &o, &part, &dphi[i], &tmp))
                gp_addmul(&x, &o, &out, Qi(1), zero, &tmp);
        gp_free(&tmp); gp_free(&part);
        for (int32_t i = 0; i < ndeps; i++) { gp_free(&phi[i]); gp_free(&dphi[i]); }
    }
    gp_free(&F);
    if (x.rc) { gp_free(&out); set_error(error, x.rc); return NULL; }
    tr_geom_pol *p = pol_from(&o, &out, error);
    gp_free(&out);
    return p;
}
int32_t tr_geom_dependency_check(const tr_scope *scope, const tr_geom_dep *deps, int32_t ndeps, const tr_math_budget *budget,
                                 tr_cert_evidence *out, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    tr_scope_view v;
    if (!scope || tr_scope_get_view(scope, &v) || ndeps < 0 || (ndeps && (!deps || !out))) return TR_MATH_INVALID;
    if (ndeps > TR_GEOM_MAX_DEPS) return TR_MATH_UNSUPPORTED;
    for (int32_t i = 0; i < ndeps; i++) {
        const tr_geom_dep *d = &deps[i];
        int32_t rc = poly_check(&d->phi);
        if (rc) return rc;
        int32_t cols[NV], nc = 0;
        rc = add_cols(cols, &nc, &d->column, 1);
        if (!rc) rc = add_cols(cols, &nc, d->phi.col, d->phi.ncols);
        if (rc) return rc;
        /* D v - D phi as an int64 polynomial over cols */
        int64_t D = 1, *pc = NULL;
        rc = integer_form(&d->phi, &D, &pc);
        if (rc) return rc;
        uint64_t n = d->phi.nterms + 1;
        uint8_t *e = (uint8_t *)zalloc(n * (uint64_t)nc);
        int64_t *c = (int64_t *)zalloc(n * 8);
        if (!e || !c) { free(pc); free(e); free(c); return TR_MATH_MEMORY; }
        for (int32_t k = 0; k < nc; k++) if (cols[k] == d->column) e[k] = 1;
        c[0] = D;
        for (uint64_t t = 0; t < d->phi.nterms; t++) {
            for (int32_t k = 0; k < d->phi.ncols; k++)
                for (int32_t q = 0; q < nc; q++) if (cols[q] == d->phi.col[k]) e[(t + 1) * nc + q] = d->phi.exponents[t * (uint64_t)d->phi.ncols + k];
            if (pc[t] == INT64_MIN) { free(pc); free(e); free(c); return TR_MATH_OVERFLOW; }
            c[t + 1] = -pc[t];
        }
        free(pc);
        tr_ident_poly ip;
        ip.nterms = n; ip.exponents = e; ip.coefficients = c;
        int32_t err = 0;
        tr_ident_set *s = tr_ident_candidates(scope, nc, cols, &ip, 1, NULL, budget, refusal, &err);
        free(e); free(c);
        if (!s) return err ? err : TR_MATH_MEMORY;
        tr_ident_cand_view cv;
        tr_ident_set_get_view(s, &cv);
        tr_ident_item it;
        tr_ident_set_items(s, &it, 1);
        rc = it.words ? tr_ident_verify_view(&v, &cv, budget, &out[i], refusal) : TR_MATH_OVERFLOW;
        tr_ident_set_free(s);
        if (rc) return rc;
    }
    return TR_MATH_OK;
}
