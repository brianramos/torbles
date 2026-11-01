/* GEOM-002: bounded elimination and generator selection; see math_elim.h.
 * SPDX-License-Identifier: Apache-2.0 */
#include "math_elim.h"
#include "math_handles.h"
#include <stdlib.h>
#include <string.h>

TR_MATH_ABI(tr_elim_spec, TR_ELIM_ABI);     /* the ctypes boundary of GEOM-002 (math_handles.h) */
TR_MATH_ABI(tr_elim_cost, TR_ELIM_ABI);
TR_MATH_ABI(tr_elim_check, TR_ELIM_ABI);
TR_MATH_ABI(tr_elim_pair_view, TR_ELIM_ABI);
TR_MATH_ABI(tr_elim_candidate, TR_ELIM_ABI);
TR_MATH_ABI(tr_elim_column, TR_ELIM_ABI);
TR_MATH_ABI(tr_elim_info, TR_ELIM_ABI);
TR_MATH_ABI(tr_elim_form_spec, TR_ELIM_ABI);
TR_MATH_ABI(tr_elim_form, TR_ELIM_ABI);
TR_MATH_ABI(tr_elim_forms_info, TR_ELIM_ABI);

_Static_assert(sizeof(tr_elim_spec) == 96, "tr_elim_spec ABI");
_Static_assert(sizeof(tr_elim_cost) == 112, "tr_elim_cost ABI");
_Static_assert(sizeof(tr_elim_check) == 144, "tr_elim_check ABI");
_Static_assert(sizeof(tr_elim_pair_view) == 176, "tr_elim_pair_view ABI");
_Static_assert(sizeof(tr_elim_candidate) == 120, "tr_elim_candidate ABI");
_Static_assert(sizeof(tr_elim_column) == 296, "tr_elim_column ABI");
_Static_assert(sizeof(tr_elim_info) == 136, "tr_elim_info ABI");
_Static_assert(sizeof(tr_elim_form_spec) == 16, "tr_elim_form_spec ABI");
_Static_assert(sizeof(tr_elim_form) == 312, "tr_elim_form ABI");
_Static_assert(sizeof(tr_elim_forms_info) == 48, "tr_elim_forms_info ABI");

#define SEL_MAGIC UINT32_C(0x4c454553)
#define PAIRS_MAGIC UINT32_C(0x5052454c)
#define FORMS_MAGIC UINT32_C(0x4d524f46)
#define NV TR_ELIM_MAX_VARS
#define INNER_ITERATIONS ((uint64_t)1 << 20)   /* GEOM-001's default iteration limit of one solve */

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
static void refuse(tr_math_refusal *r, int32_t resource, uint64_t need, uint64_t limit) {
    if (r) { r->resource = resource; r->refused = 1; r->need = need; r->limit = limit; }
}
static uint64_t mag_of(const tr_scope_view *v, int32_t c) {
    const tr_scope_view_column *k = &v->col[c];
    uint64_t m = k->nvalid ? (tr_abs_u64(k->min) > tr_abs_u64(k->max) ? tr_abs_u64(k->min) : tr_abs_u64(k->max)) : 0;
    return m < 1 ? 1 : m;
}
static uint64_t bitlen(uint64_t x) { return x ? (uint64_t)(64 - __builtin_clzll(x)) : 0; }
/* a column's width: the bits of max - min of its selected present values (at least 1) and a validity plane */
static uint64_t width_of(const tr_scope_view *v, int32_t c) {
    const tr_scope_view_column *k = &v->col[c];
    uint64_t b = k->nvalid ? bitlen((uint64_t)k->max - (uint64_t)k->min) : 0;
    return (b < 1 ? 1 : b) + (k->valid ? 1 : 0);
}
static uint64_t sat_add(uint64_t a, uint64_t b) { return tr_budget_add(a, b); }
static uint64_t sat_mul(uint64_t a, uint64_t b) { return tr_budget_mul(a, b); }

/* ================================================================ polynomials over scope columns */
typedef struct {
    int32_t ncols;
    int32_t col[NV];                 /* scope columns (distinct) */
    uint64_t n;
    uint8_t *e;                      /* n x ncols */
    int64_t *num, *den;              /* canonical rationals */
} epoly;

static void ep_free(epoly *p) { free(p->e); free(p->num); free(p->den); memset(p, 0, sizeof *p); }
static int32_t ep_alloc(epoly *p, int32_t ncols, uint64_t n) {
    p->ncols = ncols; p->n = n;
    p->e = (uint8_t *)zalloc(n * (uint64_t)(ncols ? ncols : 1));
    p->num = (int64_t *)zalloc(n * 8); p->den = (int64_t *)zalloc(n * 8);
    if (!p->e || !p->num || !p->den) { ep_free(p); return TR_MATH_MEMORY; }
    return TR_MATH_OK;
}
/* a GEOM-001 polynomial exported (the handle is not freed here) */
static int32_t ep_from_pol(const tr_geom_pol *g, epoly *out) {
    memset(out, 0, sizeof *out);
    int32_t ncols = 0, cols[NV];
    uint64_t n = 0;
    int32_t rc = tr_geom_pol_get(g, &ncols, cols, &n);
    if (rc) return rc;
    if (ncols < 0 || ncols > NV) return TR_MATH_UNSUPPORTED;
    rc = ep_alloc(out, ncols, n);
    if (rc) return rc;
    for (int32_t i = 0; i < NV; i++) out->col[i] = i < ncols ? cols[i] : -1;
    rc = tr_geom_pol_export(g, out->e, out->num, out->den, n);
    if (rc) ep_free(out);
    return rc;
}
/* a caller's tr_geom_poly copied with canonical coefficients */
static int32_t ep_from_poly(const tr_geom_poly *f, int32_t ncols_scope, epoly *out) {
    memset(out, 0, sizeof *out);
    if (!f || f->ncols < 0 || f->reserved || (f->nterms && (!f->exponents || !f->num))) return TR_MATH_INVALID;
    if (f->ncols > NV) return TR_MATH_UNSUPPORTED;
    for (int32_t i = 0; i < f->ncols; i++) {
        if (f->col[i] < 0 || f->col[i] >= ncols_scope) return TR_MATH_INVALID;
        for (int32_t j = 0; j < i; j++) if (f->col[j] == f->col[i]) return TR_MATH_INVALID;
    }
    int32_t rc = ep_alloc(out, f->ncols, f->nterms);
    if (rc) return rc;
    for (int32_t i = 0; i < NV; i++) out->col[i] = i < f->ncols ? f->col[i] : -1;
    if (f->nterms && f->ncols) memcpy(out->e, f->exponents, (size_t)(f->nterms * (uint64_t)f->ncols));
    for (uint64_t t = 0; t < f->nterms; t++) {
        int64_t d = f->den ? f->den[t] : 1;
        tr_num_rational q;
        if (d <= 0) { ep_free(out); return TR_MATH_INVALID; }
        if (!tr_q_make(f->num[t], d, &q)) { ep_free(out); return TR_MATH_OVERFLOW; }
        out->num[t] = q.numerator; out->den[t] = q.denominator;
    }
    return TR_MATH_OK;
}
static void ep_view(const epoly *p, tr_geom_poly *out) {
    memset(out, 0, sizeof *out);
    out->ncols = p->ncols;
    for (int32_t i = 0; i < NV; i++) out->col[i] = i < p->ncols ? p->col[i] : -1;
    out->nterms = p->n;
    out->exponents = p->n ? p->e : NULL; out->num = p->n ? p->num : NULL; out->den = p->n ? p->den : NULL;
}
/* the scope columns with a nonzero exponent in a nonzero term (mask over scope columns) */
static uint32_t ep_columns(const epoly *p) {
    uint32_t m = 0;
    for (uint64_t t = 0; t < p->n; t++) {
        if (!p->num[t]) continue;
        for (int32_t i = 0; i < p->ncols; i++) if (p->e[t * (uint64_t)p->ncols + i]) m |= UINT32_C(1) << p->col[i];
    }
    return m;
}
/* x alone (coefficient 1) */
static int ep_is_variable(const epoly *p, int32_t col) {
    uint64_t live = 0;
    int ok = 1;
    for (uint64_t t = 0; t < p->n; t++) {
        if (!p->num[t]) continue;
        live++;
        if (p->num[t] != 1 || p->den[t] != 1) ok = 0;
        for (int32_t i = 0; i < p->ncols; i++) {
            uint8_t e = p->e[t * (uint64_t)p->ncols + i];
            if (p->col[i] == col ? e != 1 : e != 0) ok = 0;
        }
    }
    return ok && live == 1;
}
static uint64_t ep_degree(const epoly *p) {
    uint64_t d = 0;
    for (uint64_t t = 0; t < p->n; t++) {
        uint64_t s = 0;
        for (int32_t i = 0; i < p->ncols; i++) s += p->e[t * (uint64_t)p->ncols + i];
        if (p->num[t] && s > d) d = s;
    }
    return d;
}
static uint64_t ep_terms(const epoly *p) { uint64_t k = 0; for (uint64_t t = 0; t < p->n; t++) k += p->num[t] != 0; return k; }
static int64_t lcm_den(int64_t l, int64_t d) {
    if (!l) return 0;
    uint64_t g = tr_gcd_u64((uint64_t)l, (uint64_t)d);
    __int128 r = (__int128)(l / (int64_t)g) * d;
    return r > INT64_MAX ? 0 : (int64_t)r;
}
static int64_t ep_den(const epoly *p, int64_t l) {
    for (uint64_t t = 0; t < p->n && l; t++) if (p->num[t]) l = lcm_den(l, p->den[t]);
    return l;
}
/* D p as int64 coefficients (0: an overflow) */
static int ep_scaled(const epoly *p, int64_t D, uint64_t t, int64_t *out) {
    __int128 v = (__int128)p->num[t] * (D / p->den[t]);
    if (v < INT64_MIN || v > INT64_MAX) return 0;
    *out = (int64_t)v;
    return 1;
}

/* bits of sum_t |c_t| prod_v M_v^e (multiword, saturating at 640 bits); M per variable position */
static uint64_t bound_bits(const uint64_t *mag, int32_t nv, const uint8_t *e, const int64_t *c, uint64_t nt) {
    enum { WW = 10 };
    uint64_t total[WW] = {0}, term[WW];
    for (uint64_t t = 0; t < nt; t++) {
        memset(term, 0, sizeof term);
        term[0] = tr_abs_u64(c[t]);
        for (int32_t i = 0; i < nv; i++)
            for (uint8_t q = 0; q < e[t * (uint64_t)nv + i]; q++) {
                unsigned __int128 carry = 0;
                for (int32_t w = 0; w < WW; w++) { unsigned __int128 y = (unsigned __int128)term[w] * mag[i] + carry; term[w] = (uint64_t)y; carry = y >> 64; }
                if (carry || term[WW - 1]) return (uint64_t)64 * WW;
            }
        uint64_t cy = 0;
        for (int32_t w = 0; w < WW; w++) {
            uint64_t s = total[w] + term[w], c1 = s < term[w], s2 = s + cy, c2 = s2 < cy;
            total[w] = s2; cy = c1 | c2;
        }
        if (cy || total[WW - 1]) return (uint64_t)64 * WW;
    }
    for (int32_t w = WW; w-- > 0;) if (total[w]) return (uint64_t)w * 64 + bitlen(total[w]);
    return 0;
}
static int32_t words_for(uint64_t bits) {
    for (int32_t W = 2; W <= TR_IDENT_MAX_WORDS; W++) if (bits <= (uint64_t)(64 * W - 1)) return W;
    return 0;
}

/* ================================================================ pairs */
struct tr_elim_pairs_s {
    uint32_t magic;
    int32_t npairs;
    tr_elim_pair_view view;
    uint64_t *offset, *bbits;
    uint32_t *support;
    int32_t *words, *target;
    uint8_t *e;
    int64_t *c, *den;
};
static int pairs_valid(const tr_elim_pairs *p) { return p && p->magic == PAIRS_MAGIC; }
void tr_elim_pairs_free(tr_elim_pairs *p) {
    if (!pairs_valid(p)) return;
    free(p->offset); free(p->bbits); free(p->support); free(p->words); free(p->target); free(p->e); free(p->c);
    free(p->den);
    p->magic = 0;
    free(p);
}
static int32_t var_index(const int32_t *var, int32_t nv, int32_t col) {
    for (int32_t i = 0; i < nv; i++) if (var[i] == col) return i;
    return -1;
}
/* pairs of epolys (targets, forms) on a view: one variable list (the columns either reads, increasing) */
static tr_elim_pairs *pairs_build(const tr_scope_view *v, const epoly *T, const epoly *F, const int32_t *ranged, int32_t n,
                                  const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error) {
    tr_math_budget store;
    const tr_math_budget *b = budget_or(budget, &store);
    tr_budget_none(refusal);
    if (n < 0 || n > TR_ELIM_MAX_PAIRS) { set_error(error, n < 0 ? TR_MATH_INVALID : TR_MATH_UNSUPPORTED); return NULL; }
    uint32_t cols = 0;
    uint64_t total = 0;
    for (int32_t j = 0; j < n; j++) {
        cols |= ep_columns(&T[j]) | ep_columns(&F[j]);
        if (ranged && ranged[j] >= 0) {
            if (ranged[j] >= v->ncols || !ep_is_variable(&T[j], ranged[j])) { set_error(error, TR_MATH_INVALID); return NULL; }
            cols |= UINT32_C(1) << ranged[j];
        }
        total += T[j].n + F[j].n;
    }
    int32_t var[NV], nv = 0;
    for (int32_t c = 0; c < TR_SCOPE_MAX_COLUMNS; c++) {
        if (!((cols >> c) & 1)) continue;
        if (nv >= NV) { set_error(error, TR_MATH_UNSUPPORTED); return NULL; }
        if (c >= v->ncols || (v->rows && !v->col[c].codes && !v->col[c].values)) { set_error(error, TR_MATH_INVALID); return NULL; }
        var[nv++] = c;
    }
    if (!nv) var[nv++] = 0 < v->ncols ? 0 : -1;               /* constants only: one (unread) variable slot */
    if (var[0] < 0) { set_error(error, TR_MATH_INVALID); return NULL; }
    uint64_t bytes = sat_add(sizeof(tr_elim_pairs), sat_mul(total, (uint64_t)nv + 8));
    bytes = sat_add(bytes, sat_mul((uint64_t)n, 8 * 3 + 4 * 4 + 8));
    int32_t rc = tr_budget_check_one(b, TR_BUDGET_BYTES, bytes, refusal);
    if (!rc) rc = tr_budget_check_one(b, TR_BUDGET_OUTPUT, (uint64_t)n, refusal);
    if (rc) { set_error(error, rc); return NULL; }
    tr_elim_pairs *P = (tr_elim_pairs *)zalloc(sizeof *P);
    if (!P) { set_error(error, TR_MATH_MEMORY); return NULL; }
    P->magic = PAIRS_MAGIC; P->npairs = n;
    uint64_t n1 = n ? (uint64_t)n : 1;
    P->offset = (uint64_t *)zalloc((2 * n1 + 1) * 8); P->bbits = (uint64_t *)zalloc(n1 * 8);
    P->support = (uint32_t *)zalloc(2 * n1 * 4); P->words = (int32_t *)zalloc(2 * n1 * 4);
    P->target = (int32_t *)zalloc(n1 * 4); P->den = (int64_t *)zalloc(n1 * 8);
    P->e = (uint8_t *)zalloc((total ? total : 1) * (uint64_t)nv); P->c = (int64_t *)zalloc((total ? total : 1) * 8);
    if (!P->offset || !P->bbits || !P->support || !P->words || !P->target || !P->den || !P->e || !P->c) {
        tr_elim_pairs_free(P); set_error(error, TR_MATH_MEMORY); return NULL;
    }
    uint64_t mag[NV];
    for (int32_t i = 0; i < nv; i++) mag[i] = mag_of(v, var[i]);
    uint64_t at = 0;
    rc = TR_MATH_OK;
    for (int32_t j = 0; j < n && !rc; j++) {
        int64_t D = ep_den(&T[j], ep_den(&F[j], 1));
        if (!D) { rc = TR_MATH_OVERFLOW; break; }
        P->den[j] = D;
        P->target[j] = ranged ? ranged[j] : -1;
        uint64_t bits[2] = {0, 0};
        for (int k = 0; k < 2 && !rc; k++) {
            const epoly *s = k ? &F[j] : &T[j];
            P->offset[2 * j + k] = at;
            uint32_t sup = 0;
            uint64_t start = at;
            for (uint64_t t = 0; t < s->n && !rc; t++) {
                if (!s->num[t]) continue;
                if (!ep_scaled(s, D, t, &P->c[at])) { rc = TR_MATH_OVERFLOW; break; }
                uint8_t *row = P->e + at * (uint64_t)nv;
                memset(row, 0, (size_t)nv);
                for (int32_t i = 0; i < s->ncols; i++) {
                    uint8_t ex = s->e[t * (uint64_t)s->ncols + i];
                    if (!ex) continue;
                    int32_t q = var_index(var, nv, s->col[i]);
                    row[q] = ex;
                    sup |= UINT32_C(1) << q;
                }
                at++;
            }
            P->support[2 * j + k] = sup;
            bits[k] = bound_bits(mag, nv, P->e + start * (uint64_t)nv, P->c + start, at - start);
        }
        P->bbits[j] = bits[1];
        uint64_t need = bits[0] > bits[1] ? bits[0] : bits[1];
        if (P->target[j] >= 0) {
            uint64_t r = bitlen((uint64_t)D) + bitlen(mag[var_index(var, nv, P->target[j])]);
            if (r > need) need = r;
        }
        int32_t W = words_for(need + 2);
        P->words[2 * j] = P->words[2 * j + 1] = W;
    }
    P->offset[2 * n] = at;
    if (rc) { tr_elim_pairs_free(P); set_error(error, rc); return NULL; }
    tr_elim_pair_view *pv = &P->view;
    pv->npairs = n;
    for (int32_t i = 0; i < NV; i++) pv->mag[i] = i < nv ? mag[i] : 0;
    tr_ident_cand_view *cv = &pv->cand;
    cv->ncand = 2 * n; cv->nvars = nv;
    for (int32_t i = 0; i < TR_IDENT_MAX_VARS; i++) cv->var[i] = i < nv ? var[i] : -1;
    cv->nterms = at; cv->offset = P->offset; cv->support = P->support; cv->words = P->words;
    cv->exponents = P->e; cv->coefficients = P->c;
    pv->target = P->target; pv->den = P->den;
    set_error(error, TR_MATH_OK);
    return P;
}
tr_elim_pairs *tr_elim_pairs_make(const tr_scope *scope, const tr_geom_poly *targets, const tr_geom_poly *forms,
                                  const int32_t *ranged, int32_t n, const tr_math_budget *budget, tr_math_refusal *refusal,
                                  int32_t *error) {
    tr_budget_none(refusal);
    tr_scope_view v;
    if (!scope || tr_scope_get_view(scope, &v) || n < 0 || (n && (!targets || !forms))) { set_error(error, TR_MATH_INVALID); return NULL; }
    if (n > TR_ELIM_MAX_PAIRS) { set_error(error, TR_MATH_UNSUPPORTED); return NULL; }
    epoly *T = (epoly *)zalloc((uint64_t)(n ? n : 1) * sizeof *T), *F = (epoly *)zalloc((uint64_t)(n ? n : 1) * sizeof *F);
    int32_t rc = (!T || !F) ? TR_MATH_MEMORY : TR_MATH_OK;
    for (int32_t j = 0; j < n && !rc; j++) {
        rc = ep_from_poly(&targets[j], v.ncols, &T[j]);
        if (!rc) rc = ep_from_poly(&forms[j], v.ncols, &F[j]);
    }
    tr_elim_pairs *P = NULL;
    if (!rc) P = pairs_build(&v, T, F, ranged, n, budget, refusal, &rc);
    for (int32_t j = 0; T && F && j < n; j++) { ep_free(&T[j]); ep_free(&F[j]); }
    free(T); free(F);
    set_error(error, rc);
    return rc ? NULL : P;
}
int32_t tr_elim_pairs_get_view(const tr_elim_pairs *p, tr_elim_pair_view *out) {
    if (!pairs_valid(p) || !out) return TR_MATH_INVALID;
    *out = p->view;
    return TR_MATH_OK;
}
uint64_t tr_elim_pairs_bound_bits(const tr_elim_pairs *p, int32_t j) {
    return pairs_valid(p) && j >= 0 && j < p->npairs ? p->bbits[j] : 0;
}
static int32_t pair_view_check(const tr_scope_view *v, const tr_elim_pair_view *p) { return tr_elim_pair_view_check(v, p); }
int32_t tr_elim_check_need(const tr_scope_view *v, const tr_elim_pair_view *p, int32_t target, tr_math_need *need) {
    int32_t rc = pair_view_check(v, p);
    if (rc) return rc;
    if (!need) return TR_MATH_INVALID;
    if (target == TR_ELIM_NEED_DEVICE) { tr_elim_check_device_need(v, p, need); return TR_MATH_OK; }
    if (target != TR_ELIM_NEED_CPU) return TR_MATH_INVALID;
    for (int i = 0; i < TR_BUDGET_RESOURCES; i++) need->need[i] = 0;
    need->need[TR_BUDGET_VARIABLES] = (uint64_t)p->cand.nvars;
    need->need[TR_BUDGET_SUPPORT] = p->cand.nterms;
    need->need[TR_BUDGET_OUTPUT] = (uint64_t)p->npairs;
    need->need[TR_BUDGET_BYTES] = sat_mul((uint64_t)p->npairs, 8 * TR_ELIM_ACC);
    return TR_MATH_OK;
}
int32_t tr_elim_check_view(const tr_scope_view *v, const tr_elim_pair_view *p, const tr_math_budget *budget,
                           tr_elim_check *out, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    int32_t rc = pair_view_check(v, p);
    if (rc) return rc;
    if (p->npairs && !out) return TR_MATH_INVALID;
    tr_math_need need;
    tr_elim_check_need(v, p, TR_ELIM_NEED_CPU, &need);
    tr_math_budget store;
    rc = tr_budget_check(budget_or(budget, &store), &need, refusal);
    if (rc) return rc;
    uint64_t *acc = (uint64_t *)zalloc((uint64_t)(p->npairs ? p->npairs : 1) * 8 * TR_ELIM_ACC);
    if (!acc) return TR_MATH_MEMORY;
    for (int32_t j = 0; j < p->npairs; j++) tr_elim_acc_init(acc + (uint64_t)j * TR_ELIM_ACC);
    const uint64_t nw = (v->rows + 63) / 64;
    for (uint64_t w = 0; w < nw; w++) {
        uint64_t bits = v->selection[w];
        while (bits) {
            uint64_t r = w * 64 + (uint64_t)__builtin_ctzll(bits);
            bits &= bits - 1;
            for (int32_t j = 0; j < p->npairs; j++) tr_elim_pair_row(v, p, j, r, acc + (uint64_t)j * TR_ELIM_ACC);
        }
    }
    for (int32_t j = 0; j < p->npairs; j++) tr_elim_check_finish(v, p, j, acc + (uint64_t)j * TR_ELIM_ACC, &out[j]);
    free(acc);
    return TR_MATH_OK;
}

/* ================================================================ costs */
/* the cost of D p on a view (chk: a check whose bits_form is |D p|'s, or NULL: the bound's words) */
static void cost_fill(const tr_scope_view *v, const epoly *p, int64_t D, uint64_t bbits, const tr_elim_check *chk,
                      tr_elim_cost *out) {
    memset(out, 0, sizeof *out);
    const uint64_t N = v->selected, words64 = (N + 63) / 64;
    out->rows = N; out->denominator = D; out->bound_bits = bbits;
    uint32_t cols = 0;
    for (uint64_t t = 0; t < p->n; t++) {
        if (!p->num[t]) continue;
        int64_t c = 0;
        if (!ep_scaled(p, D, t, &c)) c = 2;                      /* not reached: the pair was built */
        uint64_t d = 0;
        for (int32_t i = 0; i < p->ncols; i++) {
            uint8_t e = p->e[t * (uint64_t)p->ncols + i];
            if (e) cols |= UINT32_C(1) << p->col[i];
            d += e;
        }
        out->terms++;
        if (d > out->degree) out->degree = d;
        out->multiplications += (d > 1 ? d - 1 : 0) + (tr_abs_u64(c) != 1 && d ? 1 : 0);
    }
    for (int32_t c = 0; c < TR_SCOPE_MAX_COLUMNS; c++)
        if ((cols >> c) & 1) { out->columns++; out->scan_bits += width_of(v, c); }
    out->measured = chk != NULL;
    out->actual_bits = chk ? chk->bits_form : 0;
    uint64_t bits = chk ? out->actual_bits : bbits;
    out->words = (bits + 1 + 63) / 64;
    if (out->words < 1) out->words = 1;
    out->scan = sat_mul(words64, out->scan_bits);
    out->work = sat_mul(sat_mul(N, out->terms + out->multiplications), out->words);
    out->total = sat_add(out->scan, out->work);
}
static int cost_less(const tr_elim_cost *a, const tr_elim_cost *b, int32_t priority) {
    uint64_t ba = a->measured ? a->actual_bits : a->bound_bits, bb = b->measured ? b->actual_bits : b->bound_bits;
    uint64_t ka[3], kb[3];
    switch (priority) {
    case TR_ELIM_PRIORITY_SCAN: ka[0] = a->scan; ka[1] = a->total; ka[2] = a->work; kb[0] = b->scan; kb[1] = b->total; kb[2] = b->work; break;
    case TR_ELIM_PRIORITY_WORK: ka[0] = a->work; ka[1] = a->total; ka[2] = a->scan; kb[0] = b->work; kb[1] = b->total; kb[2] = b->scan; break;
    case TR_ELIM_PRIORITY_BITS: ka[0] = ba; ka[1] = a->total; ka[2] = a->scan; kb[0] = bb; kb[1] = b->total; kb[2] = b->scan; break;
    default: ka[0] = a->total; ka[1] = a->scan; ka[2] = a->work; kb[0] = b->total; kb[1] = b->scan; kb[2] = b->work; break;
    }
    for (int i = 0; i < 3; i++) if (ka[i] != kb[i]) return ka[i] < kb[i];
    return 0;
}
int32_t tr_elim_choose(const tr_elim_cost *costs, int32_t n, int32_t priority, int32_t *chosen) {
    if (!costs || n < 1 || !chosen || priority < 0 || priority > TR_ELIM_PRIORITY_BITS) return TR_MATH_INVALID;
    int32_t best = 0;
    for (int32_t i = 1; i < n; i++) if (cost_less(&costs[i], &costs[best], priority)) best = i;
    *chosen = best;
    return TR_MATH_OK;
}
/* checks of pairs on a view, in batches of TR_ELIM_MAX_PAIRS: the measuring checks of a selection or a form search
 * run under the request's bytes and variables limits only (its degree, support and iterations limits bound the
 * expansions and the solves, not the rows read to measure them) */
static int32_t check_batch(const tr_scope_view *v, const epoly *T, const epoly *F, const int32_t *ranged, int32_t n,
                           const tr_math_budget *outer, tr_elim_check *out, uint64_t *bbits, tr_math_refusal *refusal) {
    tr_math_budget mb;
    tr_budget_default(&mb);
    mb.limit[TR_BUDGET_BYTES] = outer->limit[TR_BUDGET_BYTES];
    mb.limit[TR_BUDGET_VARIABLES] = outer->limit[TR_BUDGET_VARIABLES];
    const tr_math_budget *b = &mb;
    for (int32_t s = 0; s < n; s += TR_ELIM_MAX_PAIRS) {
        int32_t k = n - s < TR_ELIM_MAX_PAIRS ? n - s : TR_ELIM_MAX_PAIRS, rc = 0;
        tr_elim_pairs *P = pairs_build(v, T + s, F + s, ranged ? ranged + s : NULL, k, b, refusal, &rc);
        if (!P) return rc ? rc : TR_MATH_MEMORY;
        rc = tr_elim_check_view(v, &P->view, b, out + s, refusal);
        if (!rc && bbits) for (int32_t j = 0; j < k; j++) bbits[s + j] = P->bbits[j];
        tr_elim_pairs_free(P);
        if (rc) return rc;
    }
    return TR_MATH_OK;
}
int32_t tr_elim_cost_of(const tr_scope *scope, const tr_geom_poly *f, const tr_math_budget *budget, tr_elim_cost *out,
                        tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    tr_scope_view v;
    if (!scope || tr_scope_get_view(scope, &v) || !out) return TR_MATH_INVALID;
    epoly p;
    int32_t rc = ep_from_poly(f, v.ncols, &p);
    if (rc) return rc;
    tr_elim_check chk;
    uint64_t bb = 0;
    tr_math_budget store;
    rc = check_batch(&v, &p, &p, NULL, 1, budget_or(budget, &store), &chk, &bb, refusal);
    if (!rc) {
        int64_t D = ep_den(&p, 1);
        if (!D) rc = TR_MATH_OVERFLOW;
        else cost_fill(&v, &p, D, bb, &chk, out);
    }
    ep_free(&p);
    return rc;
}

/* ================================================================ generator selection */
typedef struct {                     /* one candidate's solve */
    tr_geom_ideal *ideal;
    tr_geom_reduction *red[NV];
    epoly form[NV];
    tr_elim_check check[NV];
    tr_elim_cost cost[NV];
} solve_t;
static void solve_free(solve_t *s) {
    tr_geom_ideal_free(s->ideal);
    for (int32_t i = 0; i < NV; i++) { tr_geom_reduction_free(s->red[i]); ep_free(&s->form[i]); }
    memset(s, 0, sizeof *s);
}
struct tr_elim_selection_s {
    uint32_t magic;
    int32_t nvars, mode, nrel, priority, minimal, minimum, chosen, ncand, undecided, undecided_size;
    int32_t col[NV];
    uint32_t gen, elim;
    uint64_t solves, max_degree, max_terms, rows, direct_storage, bytes;
    tr_cert **certs;                 /* nrel owned clones */
    solve_t best;                    /* the chosen set's ideal, reductions, forms, checks and costs */
    tr_elim_column column[NV];
    tr_elim_candidate *cand;
};
static int sel_valid(const tr_elim_selection *s) { return s && s->magic == SEL_MAGIC; }
void tr_elim_selection_free(tr_elim_selection *s) {
    if (!sel_valid(s)) return;
    for (int32_t i = 0; s->certs && i < s->nrel; i++) tr_cert_free(s->certs[i]);
    free(s->certs);
    solve_free(&s->best);
    free(s->cand);
    s->magic = 0;
    free(s);
}
static int32_t first_failure(const tr_elim_check *c) {
    if (c->null_lost || c->null_extra) return TR_ELIM_R_NULL_MISMATCH;
    if (c->values) return TR_ELIM_R_VALUE_MISMATCH;
    if (c->undecided) return TR_ELIM_R_UNDECIDED;
    return TR_ELIM_R_RANGE;
}
typedef struct {
    const tr_scope_view *v;
    const tr_cert *const *rels;
    int32_t nrel, nvars;
    const int32_t *col;              /* the variables */
    uint32_t rmask;                  /* variables in a relation */
    const int32_t *extra; int32_t nextra;   /* relation columns that are not variables (eliminated first) */
    const tr_math_budget *outer, *inner;
    tr_math_refusal *refusal;
    uint64_t solves, max_degree, max_terms;
} sctx;
/* one solve: the completion under lex (extra, E in order, G) and the reductions of E; cand->status: ADMISSIBLE or
 * REJECTED with its reason (strict: a normal form reading a column outside gen); returns a native error that refuses
 * the whole selection (budget, memory, ...) */
static int32_t solve_one(sctx *x, uint32_t gen, int strict, const int32_t *eorder, int32_t ne, solve_t *s,
                         tr_elim_candidate *cand) {
    memset(s, 0, sizeof *s);
    if (++x->solves > x->outer->limit[TR_BUDGET_ITERATIONS]) {
        refuse(x->refusal, TR_BUDGET_ITERATIONS, x->solves, x->outer->limit[TR_BUDGET_ITERATIONS]);
        return TR_MATH_BUDGET;
    }
    cand->solve = (int32_t)(x->solves - 1);
    tr_geom_order o;
    memset(&o, 0, sizeof o);
    o.kind = TR_GEOM_LEX;
    for (int32_t i = 0; i < NV; i++) o.col[i] = -1;
    for (int32_t i = 0; i < x->nextra && o.nvars < NV; i++) o.col[o.nvars++] = x->extra[i];
    for (int32_t i = 0; i < ne && o.nvars < NV; i++) o.col[o.nvars++] = x->col[eorder[i]];
    for (int32_t i = 0; i < x->nvars && o.nvars < NV; i++) {
        int32_t seen = 0;
        for (int32_t k = 0; k < o.nvars; k++) seen |= o.col[k] == x->col[i];
        if (!seen && ((x->rmask >> i) & 1)) o.col[o.nvars++] = x->col[i];
    }
    int32_t rc = 0;
    s->ideal = tr_geom_ideal_make(x->rels, x->nrel, &o, 1, x->inner, x->refusal, &rc);
    if (!s->ideal) return rc ? rc : TR_MATH_MEMORY;
    tr_geom_ideal_info ii;
    tr_geom_ideal_get_info(s->ideal, &ii);
    if (ii.max_degree > x->max_degree) x->max_degree = ii.max_degree;
    if (ii.max_terms > x->max_terms) x->max_terms = ii.max_terms;
    cand->status = TR_ELIM_ADMISSIBLE; cand->failed = -1;
    for (int32_t k = 0; k < ne; k++) {
        const int32_t i = eorder[k];
        tr_geom_poly f;
        memset(&f, 0, sizeof f);
        const uint8_t one = 1;
        const int64_t num = 1;
        f.ncols = 1; f.col[0] = x->col[i];
        for (int32_t q = 1; q < NV; q++) f.col[q] = -1;
        f.nterms = 1; f.exponents = &one; f.num = &num; f.den = NULL;
        s->red[i] = tr_geom_reduce(s->ideal, &f, x->inner, x->refusal, &rc);
        if (!s->red[i]) return rc ? rc : TR_MATH_MEMORY;
        tr_geom_pol *g = tr_geom_reduction_part(s->red[i], TR_GEOM_PART_G, &rc);
        if (!g) return rc ? rc : TR_MATH_MEMORY;
        rc = ep_from_pol(g, &s->form[i]);
        tr_geom_pol_free(g);
        if (rc) return rc;
        uint64_t d = ep_degree(&s->form[i]), t = ep_terms(&s->form[i]);
        if (d > x->max_degree) x->max_degree = d;
        if (t > x->max_terms) x->max_terms = t;
        uint32_t used = ep_columns(&s->form[i]), allowed = 0;
        for (int32_t q = 0; q < x->nvars; q++) if ((gen >> q) & 1) allowed |= UINT32_C(1) << x->col[q];
        if (strict && (used & ~allowed)) {
            cand->status = TR_ELIM_REJECTED; cand->reason = TR_ELIM_R_NOT_RECONSTRUCTIBLE; cand->failed = i;
            return TR_MATH_OK;
        }
    }
    return TR_MATH_OK;
}
/* 1 when a search set's solve failed for want of exact room, not of input: GEOM-001 refused it for the degree or support
 * budget or its per-solve iteration limit (not the solves: refused before any work), or exact values passed int64 or a
 * representation limit. Such a set is undecided (its refusal recorded) and the search goes on. */
static int undecidable(const sctx *x, int32_t rc) {
    if (rc == TR_MATH_OVERFLOW || rc == TR_MATH_UNSUPPORTED) return 1;
    if (rc != TR_MATH_BUDGET) return 0;
    const int32_t r = x->refusal->resource;
    return r == TR_BUDGET_DEGREE || r == TR_BUDGET_SUPPORT ||
           (r == TR_BUDGET_ITERATIONS && x->solves <= x->outer->limit[TR_BUDGET_ITERATIONS]);
}
static int undecided_reason(int32_t reason) {
    return reason == TR_ELIM_R_BUDGET || reason == TR_ELIM_R_UNCLAIMABLE || reason == TR_ELIM_R_UNDECIDED;
}
/* the row checks and costs of a solve's reconstructions (eliminated in emask) */
static int32_t solve_check(sctx *x, uint32_t emask, solve_t *s, tr_elim_candidate *cand, uint32_t *failed_mask) {
    epoly T[NV], F[NV];
    int32_t ranged[NV], idx[NV], n = 0;
    memset(T, 0, sizeof T);
    *failed_mask = 0;
    for (int32_t i = 0; i < x->nvars; i++) {
        if (!((emask >> i) & 1)) continue;
        int32_t rc = ep_alloc(&T[n], 1, 1);
        if (rc) { for (int32_t k = 0; k < n; k++) ep_free(&T[k]); return rc; }
        T[n].col[0] = x->col[i];
        for (int32_t q = 1; q < NV; q++) T[n].col[q] = -1;
        T[n].e[0] = 1; T[n].num[0] = 1; T[n].den[0] = 1;
        F[n] = s->form[i];
        ranged[n] = x->col[i]; idx[n] = i; n++;
    }
    tr_elim_check chk[NV];
    uint64_t bb[NV];
    int32_t rc = n ? check_batch(x->v, T, F, ranged, n, x->outer, chk, bb, x->refusal) : TR_MATH_OK;
    for (int32_t k = 0; k < n; k++) ep_free(&T[k]);
    if (rc == TR_MATH_OVERFLOW) {                               /* a reconstruction past int64: not evaluated */
        cand->status = TR_ELIM_REJECTED; cand->reason = TR_ELIM_R_UNCLAIMABLE; cand->failed = n ? idx[0] : -1;
        tr_budget_none(x->refusal);
        return TR_MATH_OK;
    }
    if (rc) return rc;
    for (int32_t k = 0; k < n; k++) {
        const int32_t i = idx[k];
        s->check[i] = chk[k];
        int64_t D = ep_den(&s->form[i], 1);
        cost_fill(x->v, &s->form[i], D ? D : 1, bb[k], &chk[k], &s->cost[i]);
        if (!chk[k].holds) {
            *failed_mask |= UINT32_C(1) << i;
            if (cand->status != TR_ELIM_REJECTED) {
                cand->status = TR_ELIM_REJECTED; cand->reason = first_failure(&chk[k]); cand->failed = i;
            }
        }
    }
    return TR_MATH_OK;
}
static void candidate_costs(const sctx *x, const uint64_t *width, uint32_t gen, uint32_t elim, const solve_t *s,
                            tr_elim_candidate *c) {
    const uint64_t words64 = (x->v->selected + 63) / 64;
    c->generators = gen; c->eliminated = elim;
    c->size = __builtin_popcount(gen);
    c->stored_bits = 0;
    for (int32_t i = 0; i < x->nvars; i++) if ((gen >> i) & 1) c->stored_bits += width[i];
    c->storage = sat_mul(words64, c->stored_bits);
    c->recon_scan = c->recon_work = c->terms = c->degree = c->bits = 0;
    for (int32_t i = 0; s && i < x->nvars; i++) {
        if (!((elim >> i) & 1)) continue;
        const tr_elim_cost *k = &s->cost[i];
        c->recon_scan = sat_add(c->recon_scan, k->scan);
        c->recon_work = sat_add(c->recon_work, k->work);
        c->terms += k->terms;
        if (k->degree > c->degree) c->degree = k->degree;
        if (k->actual_bits > c->bits) c->bits = k->actual_bits;
    }
    c->total = sat_add(c->storage, sat_add(c->recon_scan, c->recon_work));
}
static int cand_less(const tr_elim_candidate *a, const tr_elim_candidate *b, int32_t priority) {
    uint64_t ka[3], kb[3];
    switch (priority) {
    case TR_ELIM_PRIORITY_SCAN: ka[0] = a->storage; ka[1] = a->total; ka[2] = a->recon_work;
                                kb[0] = b->storage; kb[1] = b->total; kb[2] = b->recon_work; break;
    case TR_ELIM_PRIORITY_WORK: ka[0] = a->recon_work; ka[1] = a->total; ka[2] = a->storage;
                                kb[0] = b->recon_work; kb[1] = b->total; kb[2] = b->storage; break;
    case TR_ELIM_PRIORITY_BITS: ka[0] = a->bits; ka[1] = a->total; ka[2] = a->storage;
                                kb[0] = b->bits; kb[1] = b->total; kb[2] = b->storage; break;
    default: ka[0] = a->total; ka[1] = a->storage; ka[2] = a->recon_work;
             kb[0] = b->total; kb[1] = b->storage; kb[2] = b->recon_work; break;
    }
    for (int i = 0; i < 3; i++) if (ka[i] != kb[i]) return ka[i] < kb[i];
    return 0;
}
static int32_t spec_check(const tr_elim_spec *sp, const tr_scope_view *v) {
    if (!sp || (sp->mode != TR_ELIM_SEARCH && sp->mode != TR_ELIM_ORDER) || sp->nvars < 0 || sp->norder < 0 ||
        sp->priority < 0 || sp->priority > TR_ELIM_PRIORITY_BITS || sp->max_size < -1 || (sp->rank_all != 0 && sp->rank_all != 1) ||
        sp->reserved[0] || sp->reserved[1]) return TR_MATH_INVALID;
    if (sp->nvars > NV || sp->norder > NV) return TR_MATH_UNSUPPORTED;
    for (int32_t i = 0; i < sp->nvars; i++) {
        if (sp->col[i] < 0 || sp->col[i] >= v->ncols) return TR_MATH_INVALID;
        for (int32_t j = 0; j < i; j++) if (sp->col[j] == sp->col[i]) return TR_MATH_INVALID;
    }
    for (int32_t i = 0; i < sp->norder; i++) {
        if (sp->order[i] < 0 || sp->order[i] >= v->ncols) return TR_MATH_INVALID;
        for (int32_t j = 0; j < i; j++) if (sp->order[j] == sp->order[i]) return TR_MATH_INVALID;
    }
    return TR_MATH_OK;
}
tr_elim_selection *tr_elim_select(const tr_scope *scope, const tr_cert *const *relations, int32_t n, const tr_elim_spec *spec,
                                  const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error) {
    tr_math_refusal own;
    if (!refusal) refusal = &own;                                /* the search reads the resource of a refusal */
    tr_budget_none(refusal);
    tr_scope_view v;
    if (!scope || tr_scope_get_view(scope, &v) || !relations || n < 1) { set_error(error, TR_MATH_INVALID); return NULL; }
    if (n > TR_ELIM_MAX_RELATIONS) { set_error(error, TR_MATH_UNSUPPORTED); return NULL; }
    int32_t rc = spec_check(spec, &v);
    if (rc) { set_error(error, rc); return NULL; }
    tr_math_budget store, inner;
    const tr_math_budget *b = budget_or(budget, &store);
    inner = *b;
    inner.limit[TR_BUDGET_ITERATIONS] = INNER_ITERATIONS;
    /* the relations' columns (scope columns of this layout) */
    uint32_t rcols = 0;
    for (int32_t i = 0; i < n; i++) {
        tr_cert_claim k;
        if (!relations[i] || tr_cert_get_claim(relations[i], &k)) { set_error(error, TR_MATH_INVALID); return NULL; }
        for (int32_t c = 0; c < k.ncols; c++) {
            if (k.col[c] < 0 || k.col[c] >= v.ncols) { set_error(error, TR_MATH_INVALID); return NULL; }
            rcols |= UINT32_C(1) << k.col[c];
        }
    }
    int32_t col[NV], nv = 0;
    if (spec->nvars) for (int32_t i = 0; i < spec->nvars; i++) col[nv++] = spec->col[i];
    else
        for (int32_t c = 0; c < TR_SCOPE_MAX_COLUMNS; c++) {
            if (!((rcols >> c) & 1)) continue;
            if (nv >= NV) { set_error(error, TR_MATH_UNSUPPORTED); return NULL; }
            col[nv++] = c;
        }
    rc = tr_budget_check_one(b, TR_BUDGET_VARIABLES, (uint64_t)nv, refusal);
    if (rc) { set_error(error, rc); return NULL; }
    for (int32_t i = 0; i < nv; i++)
        if (v.rows && !v.col[col[i]].codes && !v.col[col[i]].values) { set_error(error, TR_MATH_INVALID); return NULL; }
    for (int32_t i = 0; i < spec->norder; i++)
        if (var_index(col, nv, spec->order[i]) < 0) { set_error(error, TR_MATH_INVALID); return NULL; }
    uint32_t rmask = 0, vcols = 0;
    for (int32_t i = 0; i < nv; i++) { if ((rcols >> col[i]) & 1) rmask |= UINT32_C(1) << i; vcols |= UINT32_C(1) << col[i]; }
    int32_t extra[NV], nextra = 0;
    for (int32_t c = 0; c < TR_SCOPE_MAX_COLUMNS; c++)
        if (((rcols >> c) & 1) && !((vcols >> c) & 1)) {
            if (nextra + nv >= 2 * NV || nextra >= NV) { set_error(error, TR_MATH_UNSUPPORTED); return NULL; }
            extra[nextra++] = c;
        }
    if (nextra + __builtin_popcount(rmask) > NV) { set_error(error, TR_MATH_UNSUPPORTED); return NULL; }
    uint64_t bytes = sat_add(sizeof(tr_elim_selection), sat_mul(TR_ELIM_MAX_CANDIDATES + 1, sizeof(tr_elim_candidate)));
    rc = tr_budget_check_one(b, TR_BUDGET_BYTES, bytes, refusal);
    if (rc) { set_error(error, rc); return NULL; }
    tr_elim_selection *S = (tr_elim_selection *)zalloc(sizeof *S);
    if (S) S->cand = (tr_elim_candidate *)zalloc((TR_ELIM_MAX_CANDIDATES + 1) * sizeof *S->cand);
    if (S) S->certs = (tr_cert **)zalloc((uint64_t)n * sizeof *S->certs);
    if (!S || !S->cand || !S->certs) { if (S) { free(S->cand); free(S->certs); free(S); } set_error(error, TR_MATH_MEMORY); return NULL; }
    S->magic = SEL_MAGIC; S->nvars = nv; S->mode = spec->mode; S->nrel = n; S->undecided_size = -1;
    S->priority = spec->priority ? spec->priority : TR_ELIM_PRIORITY_TOTAL;
    for (int32_t i = 0; i < NV; i++) S->col[i] = i < nv ? col[i] : -1;
    for (int32_t i = 0; i < n && !rc; i++) {
        S->certs[i] = tr_cert_clone(relations[i], &rc);
        if (!S->certs[i] && !rc) rc = TR_MATH_MEMORY;
    }
    if (rc) { tr_elim_selection_free(S); set_error(error, rc); return NULL; }
    uint64_t width[NV];
    S->rows = v.selected;
    const uint64_t words64 = (v.selected + 63) / 64;
    for (int32_t i = 0; i < nv; i++) { width[i] = width_of(&v, col[i]); S->direct_storage = sat_add(S->direct_storage, sat_mul(words64, width[i])); }
    sctx x;
    memset(&x, 0, sizeof x);
    x.v = &v; x.rels = (const tr_cert *const *)S->certs; x.nrel = n; x.nvars = nv; x.col = col; x.rmask = rmask;
    x.extra = extra; x.nextra = nextra; x.outer = b; x.inner = &inner; x.refusal = refusal;
    const uint32_t all = nv >= 32 ? UINT32_MAX : (uint32_t)((UINT64_C(1) << nv) - 1);
    int32_t best = -1;
    solve_t cur;
    memset(&cur, 0, sizeof cur);
    if (spec->mode == TR_ELIM_ORDER) {
        /* one solve: the variables in relations, in the given order first, then V's order */
        int32_t eorder[NV] = {0}, ne = 0;
        for (int32_t i = 0; i < spec->norder; i++) { int32_t q = var_index(col, nv, spec->order[i]); if ((rmask >> q) & 1) eorder[ne++] = q; }
        for (int32_t i = 0; i < nv; i++) {
            int32_t seen = 0;
            for (int32_t k = 0; k < ne; k++) seen |= eorder[k] == i;
            if (!seen && ((rmask >> i) & 1)) eorder[ne++] = i;
        }
        tr_elim_candidate *c = &S->cand[0];
        memset(c, 0, sizeof *c);
        tr_budget_none(&c->refusal);
        rc = solve_one(&x, 0, 0, eorder, ne, &cur, c);           /* every normal form computed, decided below */
        uint32_t elim = 0;
        for (int32_t k = 0; !rc && k < ne; k++) {
            const int32_t i = eorder[k];
            if (!ep_is_variable(&cur.form[i], col[i])) elim |= UINT32_C(1) << i;
        }
        /* E's normal forms read only the variables outside E (the reduced basis); keep those that check */
        uint32_t failed = 0, keep = elim;
        if (!rc) {
            uint32_t allowed = 0;
            for (int32_t q = 0; q < nv; q++) if (!((elim >> q) & 1)) allowed |= UINT32_C(1) << col[q];
            for (int32_t q = 0; q < nv; q++) if (((elim >> q) & 1) && (ep_columns(&cur.form[q]) & ~allowed)) keep &= ~(UINT32_C(1) << q);
            c->status = TR_ELIM_ADMISSIBLE;
            rc = solve_check(&x, keep, &cur, c, &failed);
        }
        if (!rc) {
            keep &= ~failed;
            for (int32_t q = 0; q < nv; q++) if (!((keep >> q) & 1)) { tr_geom_reduction_free(cur.red[q]); cur.red[q] = NULL; ep_free(&cur.form[q]); }
            c->status = TR_ELIM_CHOSEN; c->reason = TR_ELIM_R_CHEAPEST; c->failed = failed ? __builtin_ctz(failed) : -1;
            candidate_costs(&x, width, all & ~keep, keep, &cur, c);
            S->ncand = 1; best = 0;
            S->best = cur; memset(&cur, 0, sizeof cur);
            S->minimal = 0;
        }
    } else {
        int32_t rpos[NV], nr = 0;
        for (int32_t i = 0; i < nv; i++) if ((rmask >> i) & 1) rpos[nr++] = i;
        const uint32_t forced = all & ~rmask;                    /* variables in no relation: always generators */
        const int32_t nforced = __builtin_popcount(forced);
        int32_t maxk = spec->max_size < 0 ? nv - 1 : (spec->max_size < nv - 1 ? spec->max_size : nv - 1);
        int found = 0, searched_all = 1;
        for (int32_t k = nforced; k <= maxk && !rc && !found; k++) {
            const int32_t kr = k - nforced;                      /* generators chosen among the relation variables */
            if (kr < 0 || kr >= nr) continue;
            int32_t comb[NV];
            for (int32_t i = 0; i < kr; i++) comb[i] = i;
            for (;;) {
                if (S->ncand >= TR_ELIM_MAX_CANDIDATES) { rc = TR_MATH_UNSUPPORTED; break; }
                uint32_t gen = forced;
                for (int32_t i = 0; i < kr; i++) gen |= UINT32_C(1) << rpos[comb[i]];
                int32_t eorder[NV], ne = 0;
                for (int32_t i = 0; i < nv; i++) if (((rmask >> i) & 1) && !((gen >> i) & 1)) eorder[ne++] = i;
                tr_elim_candidate *c = &S->cand[S->ncand];
                memset(c, 0, sizeof *c);
                tr_budget_none(&c->refusal);
                rc = solve_one(&x, gen, 1, eorder, ne, &cur, c);
                if (rc && undecidable(&x, rc)) {                 /* not decided within the budget: recorded, go on */
                    c->status = TR_ELIM_REJECTED; c->failed = -1;
                    c->reason = rc == TR_MATH_BUDGET ? TR_ELIM_R_BUDGET : TR_ELIM_R_UNCLAIMABLE;
                    if (rc == TR_MATH_BUDGET) c->refusal = *refusal;
                    tr_budget_none(refusal);
                    rc = TR_MATH_OK;
                }
                uint32_t failed = 0, elim = all & ~gen;
                if (!rc && c->status == TR_ELIM_ADMISSIBLE) rc = solve_check(&x, elim, &cur, c, &failed);
                if (rc) { solve_free(&cur); break; }
                candidate_costs(&x, width, gen, elim, c->status == TR_ELIM_ADMISSIBLE ? &cur : NULL, c);
                S->ncand++;
                if (c->status == TR_ELIM_REJECTED && undecided_reason(c->reason)) {
                    S->undecided++;
                    if (S->undecided_size < 0) S->undecided_size = c->size;     /* sizes increase: the smallest */
                }
                if (c->status == TR_ELIM_ADMISSIBLE) {
                    found = 1;
                    if (best < 0 || cand_less(c, &S->cand[best], S->priority)) {
                        if (best >= 0) S->cand[best].status = TR_ELIM_ADMISSIBLE;
                        best = S->ncand - 1;
                        solve_free(&S->best);
                        S->best = cur; memset(&cur, 0, sizeof cur);
                    } else solve_free(&cur);
                    if (!spec->rank_all) break;
                } else solve_free(&cur);
                /* the next combination in lexicographic order */
                int32_t p = kr - 1;
                while (p >= 0 && comb[p] == nr - kr + p) p--;
                if (p < 0) break;
                comb[p]++;
                for (int32_t i = p + 1; i < kr; i++) comb[i] = comb[i - 1] + 1;
            }
        }
        if (!rc && !found && maxk < nv - 1) searched_all = 0;
        if (!rc) {
            if (!found) {                                         /* nothing eliminated: G = V */
                tr_elim_candidate *c = &S->cand[S->ncand];
                memset(c, 0, sizeof *c);
                tr_budget_none(&c->refusal);
                c->solve = -1; c->failed = -1; c->reason = TR_ELIM_R_DIRECT;
                candidate_costs(&x, width, all, 0, NULL, c);
                best = S->ncand++;
                S->minimal = searched_all && !S->undecided;
            } else S->minimal = S->undecided_size < 0 || S->undecided_size >= S->cand[best].size;
            S->cand[best].status = TR_ELIM_CHOSEN;
            if (S->cand[best].reason != TR_ELIM_R_DIRECT) S->cand[best].reason = TR_ELIM_R_CHEAPEST;
            for (int32_t i = 0; i < S->ncand; i++)
                if (S->cand[i].status == TR_ELIM_ADMISSIBLE) S->cand[i].reason = TR_ELIM_R_COSTLIER;
        }
    }
    solve_free(&cur);
    if (rc) { tr_elim_selection_free(S); set_error(error, rc); return NULL; }
    S->chosen = best;
    S->gen = S->cand[best].generators; S->elim = S->cand[best].eliminated;
    S->minimum = __builtin_popcount(S->gen);
    S->solves = x.solves; S->max_degree = x.max_degree; S->max_terms = x.max_terms;
    for (int32_t i = 0; i < nv; i++) {
        tr_elim_column *k = &S->column[i];
        memset(k, 0, sizeof *k);
        k->column = col[i]; k->width = width[i]; k->direct = sat_mul(words64, width[i]);
        k->role = ((S->elim >> i) & 1) ? TR_ELIM_ELIMINATED : TR_ELIM_GENERATOR;
        k->read = TR_ELIM_READ_DIRECT;
        if (k->role != TR_ELIM_ELIMINATED) continue;
        k->reconstruction = S->best.cost[i]; k->check = S->best.check[i];
        uint32_t used = ep_columns(&S->best.form[i]);
        for (int32_t q = 0; q < nv; q++) if ((used >> col[q]) & 1) k->support |= UINT32_C(1) << q;
        tr_geom_reduction_info ri;
        k->claimable = S->best.red[i] && !tr_geom_reduction_get_info(S->best.red[i], &ri) && ri.claimable;
        if (k->reconstruction.total < k->direct) k->read = TR_ELIM_READ_RECONSTRUCT;
    }
    S->bytes = sizeof *S + (uint64_t)S->ncand * sizeof(tr_elim_candidate);
    for (int32_t i = 0; i < nv; i++) S->bytes += S->best.form[i].n * ((uint64_t)S->best.form[i].ncols + 16);
    set_error(error, TR_MATH_OK);
    return S;
}
int32_t tr_elim_selection_get_info(const tr_elim_selection *S, tr_elim_info *out) {
    if (!sel_valid(S) || !out) return TR_MATH_INVALID;
    memset(out, 0, sizeof *out);
    out->nvars = S->nvars; out->mode = S->mode;
    out->ngenerators = __builtin_popcount(S->gen); out->neliminated = __builtin_popcount(S->elim);
    for (int32_t i = 0; i < NV; i++) out->col[i] = S->col[i];
    out->generators = S->gen; out->eliminated = S->elim; out->minimum = S->minimum; out->minimal = S->minimal;
    out->undecided = S->undecided; out->undecided_size = S->undecided_size;
    out->ncandidates = S->ncand; out->chosen = S->chosen; out->nrelations = S->nrel; out->priority = S->priority;
    out->solves = S->solves; out->max_degree = S->max_degree; out->max_terms = S->max_terms; out->rows = S->rows;
    out->direct_storage = S->direct_storage; out->bytes = S->bytes;
    return TR_MATH_OK;
}
int32_t tr_elim_selection_candidates(const tr_elim_selection *S, tr_elim_candidate *out, int32_t capacity) {
    if (!sel_valid(S) || (S->ncand && !out) || capacity < S->ncand) return TR_MATH_INVALID;
    if (S->ncand) memcpy(out, S->cand, (size_t)S->ncand * sizeof *out);
    return TR_MATH_OK;
}
int32_t tr_elim_selection_column(const tr_elim_selection *S, int32_t i, tr_elim_column *out) {
    if (!sel_valid(S) || i < 0 || i >= S->nvars || !out) return TR_MATH_INVALID;
    *out = S->column[i];
    return TR_MATH_OK;
}
tr_geom_pol *tr_elim_selection_form(const tr_elim_selection *S, int32_t i, int32_t *error) {
    if (!sel_valid(S) || i < 0 || i >= S->nvars || !((S->elim >> i) & 1) || !S->best.red[i]) { set_error(error, TR_MATH_INVALID); return NULL; }
    return tr_geom_reduction_part(S->best.red[i], TR_GEOM_PART_G, error);
}
tr_cert *tr_elim_selection_certify(const tr_elim_selection *S, int32_t i, const tr_math_budget *budget,
                                   tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    if (!sel_valid(S) || i < 0 || i >= S->nvars || !((S->elim >> i) & 1) || !S->best.red[i]) { set_error(error, TR_MATH_INVALID); return NULL; }
    return tr_geom_certify(S->best.red[i], budget, refusal, error);
}
tr_elim_pairs *tr_elim_selection_pairs(const tr_elim_selection *S, const tr_scope *scope, const tr_math_budget *budget,
                                       tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    tr_scope_view v;
    if (!sel_valid(S) || !scope || tr_scope_get_view(scope, &v)) { set_error(error, TR_MATH_INVALID); return NULL; }
    epoly T[NV], F[NV];
    int32_t ranged[NV], n = 0, rc = TR_MATH_OK;
    memset(T, 0, sizeof T);
    for (int32_t i = 0; i < S->nvars && !rc; i++) {
        if (!((S->elim >> i) & 1)) continue;
        if (S->col[i] >= v.ncols) { rc = TR_MATH_INVALID; break; }
        rc = ep_alloc(&T[n], 1, 1);
        if (rc) break;
        T[n].col[0] = S->col[i];
        for (int32_t q = 1; q < NV; q++) T[n].col[q] = -1;
        T[n].e[0] = 1; T[n].num[0] = 1; T[n].den[0] = 1;
        F[n] = S->best.form[i]; ranged[n] = S->col[i]; n++;
    }
    tr_elim_pairs *P = rc ? NULL : pairs_build(&v, T, F, ranged, n, budget, refusal, &rc);
    for (int32_t k = 0; k < NV; k++) ep_free(&T[k]);
    set_error(error, rc);
    return rc ? NULL : P;
}

/* ================================================================ local elimination forms */
struct tr_elim_forms_s {
    uint32_t magic;
    int32_t ncand, chosen, nvars, priority;
    uint32_t reads;
    uint64_t solves, rows, bytes;
    tr_elim_form *info;
    epoly *poly;
    tr_geom_reduction **red;
};
static int forms_valid(const tr_elim_forms *f) { return f && f->magic == FORMS_MAGIC; }
void tr_elim_forms_free(tr_elim_forms *f) {
    if (!forms_valid(f)) return;
    for (int32_t i = 0; i < f->ncand; i++) {
        if (f->poly) ep_free(&f->poly[i]);
        if (f->red) tr_geom_reduction_free(f->red[i]);
    }
    free(f->info); free(f->poly); free(f->red);
    f->magic = 0;
    free(f);
}
/* 1 when T and F have int64 integer forms under their common denominator */
static int scalable(const epoly *T, const epoly *F) {
    int64_t D = ep_den(T, ep_den(F, 1)), c;
    if (!D) return 0;
    for (uint64_t t = 0; t < T->n; t++) if (T->num[t] && !ep_scaled(T, D, t, &c)) return 0;
    for (uint64_t t = 0; t < F->n; t++) if (F->num[t] && !ep_scaled(F, D, t, &c)) return 0;
    return 1;
}
tr_elim_forms *tr_elim_forms_make(const tr_scope *scope, const tr_elim_selection *S, const tr_geom_poly *q,
                                  const tr_elim_form_spec *fs, const tr_math_budget *budget, tr_math_refusal *refusal,
                                  int32_t *error) {
    tr_budget_none(refusal);
    tr_scope_view v;
    if (!scope || tr_scope_get_view(scope, &v) || !sel_valid(S) || !q || !fs || fs->max_eliminate < -1 ||
        (fs->normal_form != 0 && fs->normal_form != 1) || fs->priority < 0 || fs->priority > TR_ELIM_PRIORITY_BITS ||
        fs->reserved) { set_error(error, TR_MATH_INVALID); return NULL; }
    for (int32_t i = 0; i < S->nvars; i++) if (S->col[i] >= v.ncols) { set_error(error, TR_MATH_INVALID); return NULL; }
    tr_math_budget store, inner;
    const tr_math_budget *b = budget_or(budget, &store);
    inner = *b;
    inner.limit[TR_BUDGET_ITERATIONS] = INNER_ITERATIONS;
    epoly Q;
    int32_t rc = ep_from_poly(q, v.ncols, &Q);
    if (rc) { set_error(error, rc); return NULL; }
    /* the eliminated variables q reads, and the candidate sets S in increasing size then lexicographic order */
    const uint32_t qcols = ep_columns(&Q);
    int32_t pos[NV], m = 0;
    for (int32_t i = 0; i < S->nvars; i++) if (((S->elim >> i) & 1) && ((qcols >> S->col[i]) & 1)) pos[m++] = i;
    const int32_t maxe = fs->max_eliminate < 0 || fs->max_eliminate > m ? m : fs->max_eliminate;
    uint32_t sets[TR_ELIM_MAX_FORMS];
    int32_t nsets = 0;
    for (int32_t k = 1; k <= maxe; k++) {
        int32_t comb[NV];
        for (int32_t i = 0; i < k; i++) comb[i] = i;
        for (;;) {
            uint32_t s = 0;
            for (int32_t i = 0; i < k; i++) s |= UINT32_C(1) << pos[comb[i]];
            sets[nsets++] = s;
            int32_t p = k - 1;
            while (p >= 0 && comb[p] == m - k + p) p--;
            if (p < 0) break;
            comb[p]++;
            for (int32_t i = p + 1; i < k; i++) comb[i] = comb[i - 1] + 1;
        }
    }
    const int nf = fs->normal_form && S->best.ideal && S->elim;
    const int32_t ncand = 1 + nsets + nf;
    const uint64_t alternatives = (uint64_t)(nsets + nf);
    rc = tr_budget_check_one(b, TR_BUDGET_ITERATIONS, alternatives, refusal);    /* the solves, before any solve */
    if (!rc) rc = tr_budget_check_one(b, TR_BUDGET_BYTES, sat_add(sizeof(tr_elim_forms), sat_mul((uint64_t)ncand,
                                       sizeof(tr_elim_form) + sizeof(epoly) + 8)), refusal);
    if (rc) { ep_free(&Q); set_error(error, rc); return NULL; }
    tr_elim_forms *R = (tr_elim_forms *)zalloc(sizeof *R);
    if (R) {
        R->info = (tr_elim_form *)zalloc((uint64_t)ncand * sizeof *R->info);
        R->poly = (epoly *)zalloc((uint64_t)ncand * sizeof *R->poly);
        R->red = (tr_geom_reduction **)zalloc((uint64_t)ncand * sizeof *R->red);
    }
    if (!R || !R->info || !R->poly || !R->red) {
        if (R) { free(R->info); free(R->poly); free(R->red); free(R); }
        ep_free(&Q); set_error(error, TR_MATH_MEMORY); return NULL;
    }
    R->magic = FORMS_MAGIC; R->ncand = ncand; R->nvars = S->nvars; R->rows = v.selected;
    R->priority = fs->priority ? fs->priority : TR_ELIM_PRIORITY_TOTAL;
    for (int32_t i = 0; i < m; i++) R->reads |= UINT32_C(1) << pos[i];
    R->poly[0] = Q;                                               /* the direct candidate owns q's copy */
    R->info[0].kind = TR_ELIM_FORM_DIRECT;
    /* the reconstruction certificates of the variables q reads */
    tr_cert *rec[NV];
    memset(rec, 0, sizeof rec);
    tr_math_budget cb;                                            /* the reconstructions' certificates: not expansions */
    tr_budget_default(&cb);
    cb.limit[TR_BUDGET_BYTES] = b->limit[TR_BUDGET_BYTES];
    for (int32_t i = 0; i < m && !rc; i++) {
        tr_math_refusal r2;
        rec[pos[i]] = tr_geom_certify(S->best.red[pos[i]], &cb, &r2, &rc);
        if (!rec[pos[i]]) { if (rc == TR_MATH_OVERFLOW || rc == TR_MATH_UNSUPPORTED) rc = TR_MATH_OK; else if (!rc) rc = TR_MATH_MEMORY; }
    }
    for (int32_t c = 1; c < ncand && !rc; c++) {
        tr_elim_form *F = &R->info[c];
        tr_budget_none(&F->refusal);
        tr_geom_ideal *I = NULL;
        int32_t e = 0;
        if (c <= nsets) {
            const uint32_t s = sets[c - 1];
            F->kind = TR_ELIM_FORM_LOCAL; F->eliminated = s;
            tr_cert *prem[NV];
            int32_t np = 0, missing = 0;
            tr_geom_order o;
            memset(&o, 0, sizeof o);
            o.kind = TR_GEOM_LEX;
            for (int32_t i = 0; i < NV; i++) o.col[i] = -1;
            for (int32_t i = 0; i < S->nvars; i++) {
                if (!((s >> i) & 1)) continue;
                if (!rec[i]) missing = 1;
                prem[np++] = rec[i];
                o.col[o.nvars++] = S->col[i];
            }
            if (missing) { F->status = TR_ELIM_REJECTED; F->reason = TR_ELIM_R_UNCLAIMABLE; continue; }
            R->solves++;
            I = tr_geom_ideal_make((const tr_cert *const *)prem, np, &o, 0, &inner, &F->refusal, &e);
        } else {
            F->kind = TR_ELIM_FORM_NORMAL; F->eliminated = S->elim & R->reads;
            R->solves++;
        }
        tr_geom_reduction *red = NULL;
        if (F->kind == TR_ELIM_FORM_NORMAL || I) {
            tr_geom_poly qv;
            ep_view(&Q, &qv);
            red = tr_geom_reduce(F->kind == TR_ELIM_FORM_NORMAL ? S->best.ideal : I, &qv, &inner, &F->refusal, &e);
        }
        tr_geom_ideal_free(I);
        if (!red) {
            if (e == TR_MATH_BUDGET) { F->status = TR_ELIM_REJECTED; F->reason = TR_ELIM_R_BUDGET; continue; }
            if (e == TR_MATH_OVERFLOW || e == TR_MATH_UNSUPPORTED) { F->status = TR_ELIM_REJECTED; F->reason = TR_ELIM_R_UNCLAIMABLE; tr_budget_none(&F->refusal); continue; }
            rc = e ? e : TR_MATH_MEMORY;
            break;
        }
        tr_budget_none(&F->refusal);
        R->red[c] = red;
        tr_geom_reduction_info ri;
        tr_geom_reduction_get_info(red, &ri);
        F->claimable = ri.claimable;
        F->depth = F->kind == TR_ELIM_FORM_LOCAL ? 2 : 1;
        tr_geom_pol *g = tr_geom_reduction_part(red, TR_GEOM_PART_G, &e);
        if (!g) { rc = e ? e : TR_MATH_MEMORY; break; }
        rc = ep_from_pol(g, &R->poly[c]);
        tr_geom_pol_free(g);
        if (rc) break;
        if (!ri.claimable) { F->status = TR_ELIM_REJECTED; F->reason = TR_ELIM_R_UNCLAIMABLE; }
    }
    for (int32_t i = 0; i < NV; i++) tr_cert_free(rec[i]);
    /* the checks against q and the costs (each candidate measured on its own integer form) */
    if (!rc) {
        epoly *T = (epoly *)zalloc((uint64_t)ncand * sizeof *T);
        tr_elim_check *chk = (tr_elim_check *)zalloc((uint64_t)ncand * sizeof *chk);
        tr_elim_check *self = (tr_elim_check *)zalloc((uint64_t)ncand * sizeof *self);
        uint64_t *bb = (uint64_t *)zalloc((uint64_t)ncand * 8);
        int32_t *idx = (int32_t *)zalloc((uint64_t)ncand * 4);
        epoly *P = (epoly *)zalloc((uint64_t)ncand * sizeof *P);
        int32_t k = 0;
        if (!T || !chk || !self || !bb || !idx || !P) rc = TR_MATH_MEMORY;
        for (int32_t c = 0; c < ncand && !rc; c++) {
            if (c && !R->red[c]) continue;
            if (!scalable(&Q, &R->poly[c]) || !scalable(&R->poly[c], &R->poly[c])) {
                if (!c) { rc = TR_MATH_OVERFLOW; break; }          /* the direct expression itself past int64 */
                R->info[c].status = TR_ELIM_REJECTED; R->info[c].reason = TR_ELIM_R_UNCLAIMABLE;
                continue;
            }
            T[k] = Q; P[k] = R->poly[c]; idx[k++] = c;
        }
        if (!rc) rc = check_batch(&v, T, P, NULL, k, b, chk, NULL, refusal);
        if (!rc) rc = check_batch(&v, P, P, NULL, k, b, self, bb, refusal);
        for (int32_t j = 0; !rc && j < k; j++) {
            const int32_t c = idx[j];
            tr_elim_form *F = &R->info[c];
            F->check = chk[j];
            int64_t D = ep_den(&R->poly[c], 1);
            cost_fill(&v, &R->poly[c], D ? D : 1, bb[j], &self[j], &F->cost);
            F->columns = ep_columns(&R->poly[c]);
            if (F->status != TR_ELIM_REJECTED && !chk[j].holds) {
                F->status = TR_ELIM_REJECTED;
                F->reason = chk[j].null_lost || chk[j].null_extra ? TR_ELIM_R_NULL_MISMATCH :
                            chk[j].values ? TR_ELIM_R_VALUE_MISMATCH : TR_ELIM_R_UNDECIDED;
            }
        }
        free(T); free(chk); free(self); free(bb); free(idx); free(P);
    }
    if (rc) { tr_elim_forms_free(R); set_error(error, rc); return NULL; }
    /* the choice: the cheapest admissible by the priority, ties to the earliest (the direct candidate first) */
    int32_t best = 0;
    for (int32_t c = 1; c < ncand; c++)
        if (R->info[c].status != TR_ELIM_REJECTED && cost_less(&R->info[c].cost, &R->info[best].cost, R->priority)) best = c;
    for (int32_t c = 0; c < ncand; c++) {
        tr_elim_form *F = &R->info[c];
        if (F->status == TR_ELIM_REJECTED) continue;
        F->status = c == best ? TR_ELIM_CHOSEN : TR_ELIM_ADMISSIBLE;
        F->reason = c == best ? TR_ELIM_R_CHEAPEST : TR_ELIM_R_COSTLIER;
    }
    R->chosen = best;
    R->bytes = sizeof *R + (uint64_t)ncand * (sizeof(tr_elim_form) + sizeof(epoly) + 8);
    for (int32_t c = 0; c < ncand; c++) R->bytes += R->poly[c].n * ((uint64_t)R->poly[c].ncols + 16);
    set_error(error, TR_MATH_OK);
    return R;
}
int32_t tr_elim_forms_get_info(const tr_elim_forms *R, tr_elim_forms_info *out) {
    if (!forms_valid(R) || !out) return TR_MATH_INVALID;
    memset(out, 0, sizeof *out);
    out->ncandidates = R->ncand; out->chosen = R->chosen; out->nvars = R->nvars; out->priority = R->priority;
    out->reads = R->reads; out->solves = R->solves; out->rows = R->rows; out->bytes = R->bytes;
    return TR_MATH_OK;
}
int32_t tr_elim_forms_candidate(const tr_elim_forms *R, int32_t i, tr_elim_form *out) {
    if (!forms_valid(R) || i < 0 || i >= R->ncand || !out) return TR_MATH_INVALID;
    *out = R->info[i];
    return TR_MATH_OK;
}
tr_geom_pol *tr_elim_forms_poly(const tr_elim_forms *R, int32_t i, int32_t *error) {
    if (!forms_valid(R) || i < 1 || i >= R->ncand || !R->red[i]) { set_error(error, TR_MATH_INVALID); return NULL; }
    return tr_geom_reduction_part(R->red[i], TR_GEOM_PART_G, error);
}
tr_cert *tr_elim_forms_certify(const tr_elim_forms *R, int32_t i, const tr_math_budget *budget, tr_math_refusal *refusal,
                               int32_t *error) {
    tr_budget_none(refusal);
    if (!forms_valid(R) || i < 1 || i >= R->ncand || !R->red[i]) { set_error(error, TR_MATH_INVALID); return NULL; }
    return tr_geom_certify(R->red[i], budget, refusal, error);
}
