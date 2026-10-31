/* GEN-001: generating polynomials of finite-support distributions; see math_gen.h.
 * SPDX-License-Identifier: Apache-2.0 */
#include "math_gen.h"
#include "mathematics.h"
#include "math_domain.h"
#include "math_carriers.h"
#include "math_handles.h"
#include <stdlib.h>
#include <string.h>

TR_MATH_ABI(tr_gen_info, TR_GEN_ABI);              /* the ctypes boundary of GEN-001 (math_handles.h) */
TR_MATH_ABI(tr_gen_point, TR_GEN_ABI);
TR_MATH_ABI(tr_gen_value, TR_GEN_ABI);

_Static_assert(sizeof(tr_gen_info) == 152, "tr_gen_info ABI");
_Static_assert(sizeof(tr_gen_point) == 16, "tr_gen_point ABI");
_Static_assert(sizeof(tr_gen_value) == 40, "tr_gen_value ABI");

#define POLY_MAGIC UINT32_C(0x594c4f50)

struct tr_gen_poly_s {
    uint32_t magic;
    int32_t nvars, repr, mass, scale[2], wscale, nonneg;
    int64_t shift[2];
    uint64_t span[2], box, nterms, nonzero, rows, nulls, bytes, maxbits, maxlimbs;
    uint64_t *exps;            /* sparse: [nterms nvars] */
    tr_bigint **coef;          /* [nterms]; dense: NULL = 0 */
};

/* a term: exponents and a coefficient (owned by the builder that receives it, or borrowed in a view) */
typedef struct { uint64_t e[2]; tr_bigint *c; } gterm;

/* ------------------------------------------------------------------ helpers */
static void set_error(int32_t *error, int32_t code) { if (error) *error = code; }
static void *zalloc(uint64_t bytes) {
    if (bytes >= (uint64_t)SIZE_MAX) return NULL;
    return calloc(1, bytes ? (size_t)bytes : 1);
}
static const tr_math_budget *budget_of(const tr_math_budget *b, tr_math_budget *dflt) {
    if (b) return b;
    tr_budget_default(dflt);
    return dflt;
}
static void need_zero(tr_math_need *n) { for (int i = 0; i < TR_BUDGET_RESOURCES; i++) n->need[i] = 0; }
static int32_t check_need(const tr_math_budget *b, const tr_math_need *n, tr_math_refusal *r) {
    int32_t rc = tr_budget_check(b, n, r);
    if (rc) return rc;
    for (int32_t i = 0; i < TR_BUDGET_RESOURCES; i++)
        if (n->need[i] == TR_BUDGET_UNKNOWN) {
            if (r) { r->resource = i; r->refused = 1; r->need = n->need[i]; r->limit = b->limit[i]; }
            return TR_MATH_BUDGET;
        }
    return TR_MATH_OK;
}
static int poly_valid(const tr_gen_poly *p) { return p && p->magic == POLY_MAGIC; }
static void free_terms(gterm *t, uint64_t n) {
    if (!t) return;
    for (uint64_t i = 0; i < n; i++) tr_bigint_free(t[i].c);
    free(t);
}
static tr_bigint *big_i128(__int128 v, const tr_math_budget *b, tr_math_refusal *r, int32_t *e) {
    return tr_bigint_from_i128(tr_i128_set(v), b, r, e);
}
static uint64_t limbs_of(const tr_bigint *x) { return x ? tr_bigint_limb_count(x) : 0; }
static int is_zero(const tr_bigint *x) { return !x || tr_bigint_sign(x) == 0; }
static int term_cmp(const gterm *a, const gterm *b) {
    if (a->e[0] != b->e[0]) return a->e[0] < b->e[0] ? -1 : 1;
    if (a->e[1] != b->e[1]) return a->e[1] < b->e[1] ? -1 : 1;
    return 0;
}
static void term_sort(gterm *t, gterm *tmp, uint64_t n) {          /* stable merge sort */
    for (uint64_t w = 1; w < n; w *= 2) {
        for (uint64_t s = 0; s < n; s += 2 * w) {
            uint64_t m = s + w < n ? s + w : n, e = s + 2 * w < n ? s + 2 * w : n, i = s, j = m, o = s;
            while (i < m && j < e) tmp[o++] = term_cmp(&t[j], &t[i]) < 0 ? t[j++] : t[i++];
            while (i < m) tmp[o++] = t[i++];
            while (j < e) tmp[o++] = t[j++];
        }
        memcpy(t, tmp, (size_t)n * sizeof *t);
    }
}
/* the bytes of a state with `stored` terms of coefficients up to `limbs` limbs */
static uint64_t poly_bytes(int nvars, int repr, uint64_t stored, uint64_t nonzero, uint64_t limbs) {
    uint64_t b = tr_budget_add(sizeof(tr_gen_poly), tr_budget_mul(stored, 8));
    if (repr == TR_GEN_SPARSE) b = tr_budget_add(b, tr_budget_mul(stored, 8 * (uint64_t)nvars));
    return tr_budget_add(b, tr_budget_mul(nonzero, TR_GEN_BIG_BYTES(limbs)));
}
static uint64_t box_of(int nvars, const uint64_t *span) {
    return nvars == 1 ? span[0] : tr_budget_mul(span[0], span[1]);
}

/* ------------------------------------------------------------------ building a state */
static void poly_release(tr_gen_poly *p) {
    if (!p) return;
    if (p->coef) for (uint64_t i = 0; i < p->nterms; i++) tr_bigint_free(p->coef[i]);
    free(p->coef); free(p->exps);
    p->magic = 0;
    free(p);
}
void tr_gen_free(tr_gen_poly *p) { if (poly_valid(p)) poly_release(p); }

/* the state of terms t[0..n) in strict lexicographic order (zeros allowed): consumes every coefficient (also on
 * failure). span: the dense box (NULL: the hull of the nonzero terms). Budget: support = stored terms, bytes = the
 * arrays and the coefficients. */
static tr_gen_poly *poly_make(int nvars, const int64_t *shift, gterm *t, uint64_t n, int repr, const uint64_t *span,
                              const tr_math_budget *b, tr_math_refusal *r, int32_t *err) {
    uint64_t nonzero = 0, hull[2] = {0, 0}, maxbits = 0, maxlimbs = 0, coefbytes = 0;
    int nonneg = 1;
    for (uint64_t i = 0; i < n; i++) {
        if (is_zero(t[i].c)) continue;
        nonzero++;
        for (int c = 0; c < nvars; c++) if (t[i].e[c] + 1 > hull[c]) hull[c] = t[i].e[c] + 1;
        uint64_t bits = tr_bigint_bit_length(t[i].c), l = limbs_of(t[i].c);
        if (bits > maxbits) maxbits = bits;
        if (l > maxlimbs) maxlimbs = l;
        coefbytes = tr_budget_add(coefbytes, TR_GEN_BIG_BYTES(l));
        nonneg &= tr_bigint_sign(t[i].c) > 0;
    }
    uint64_t box[2] = {span ? span[0] : hull[0], span ? (nvars > 1 ? span[1] : 1) : (nvars > 1 ? hull[1] : 1)};
    if (nvars == 1) box[1] = 1;
    for (uint64_t i = 0; span && i < n; i++)
        for (int c = 0; c < nvars; c++)
            if (!is_zero(t[i].c) && t[i].e[c] >= box[c]) { free_terms(t, n); set_error(err, TR_MATH_INVALID); return NULL; }
    uint64_t cells = box_of(nvars, box), stored = repr == TR_GEN_DENSE ? cells : nonzero;
    tr_math_need need;
    need_zero(&need);
    need.need[TR_BUDGET_VARIABLES] = (uint64_t)nvars;
    need.need[TR_BUDGET_SUPPORT] = stored;
    need.need[TR_BUDGET_BYTES] = tr_budget_add(poly_bytes(nvars, repr, stored, 0, 0), coefbytes);
    int32_t rc = check_need(b, &need, r);
    tr_gen_poly *p = rc ? NULL : zalloc(sizeof *p);
    if (p) {
        p->coef = zalloc(tr_budget_mul(stored, sizeof(tr_bigint *)));
        if (repr == TR_GEN_SPARSE) p->exps = zalloc(tr_budget_mul(stored, 8 * (uint64_t)nvars));
        if (!p->coef || (repr == TR_GEN_SPARSE && !p->exps)) { poly_release(p); p = NULL; }
    }
    if (!p) { free_terms(t, n); set_error(err, rc ? rc : TR_MATH_MEMORY); return NULL; }
    p->nvars = nvars; p->repr = repr; p->nonneg = nonneg;
    for (int c = 0; c < 2; c++) p->shift[c] = c < nvars ? shift[c] : 0;
    p->span[0] = box[0]; p->span[1] = box[1]; p->box = cells; p->nterms = stored; p->nonzero = nonzero;
    p->maxbits = maxbits; p->maxlimbs = maxlimbs;
    p->bytes = need.need[TR_BUDGET_BYTES];
    uint64_t o = 0;
    for (uint64_t i = 0; i < n; i++) {
        if (is_zero(t[i].c)) { tr_bigint_free(t[i].c); continue; }
        if (repr == TR_GEN_DENSE) {
            uint64_t idx = nvars == 1 ? t[i].e[0] : t[i].e[0] * box[1] + t[i].e[1];
            p->coef[idx] = t[i].c;
        } else {
            for (int c = 0; c < nvars; c++) p->exps[o * (uint64_t)nvars + (uint64_t)c] = t[i].e[c];
            p->coef[o++] = t[i].c;
        }
    }
    free(t);
    p->magic = POLY_MAGIC;
    set_error(err, TR_MATH_OK);
    return p;
}
/* the nonzero terms of a state in lexicographic order (borrowed coefficients); *n their count */
static gterm *poly_view(const tr_gen_poly *p, uint64_t *n) {
    gterm *t = zalloc(tr_budget_mul(p->nonzero, sizeof(gterm)));
    if (!t) return NULL;
    uint64_t o = 0;
    for (uint64_t i = 0; i < p->nterms; i++) {
        if (is_zero(p->coef[i])) continue;
        if (p->repr == TR_GEN_DENSE) {
            if (p->nvars == 1) { t[o].e[0] = i; t[o].e[1] = 0; }
            else { t[o].e[0] = i / p->span[1]; t[o].e[1] = i % p->span[1]; }
        } else {
            t[o].e[0] = p->exps[i * (uint64_t)p->nvars];
            t[o].e[1] = p->nvars > 1 ? p->exps[i * (uint64_t)p->nvars + 1] : 0;
        }
        t[o++].c = p->coef[i];
    }
    *n = o;
    return t;
}
/* value v = e + shift as an int64: 1, or 0 when it does not fit */
static int value_of(uint64_t e, int64_t shift, int64_t *v) {
    __int128 x = (__int128)e + shift;
    if (x < INT64_MIN || x > INT64_MAX) return 0;
    *v = (int64_t)x;
    return 1;
}
static void copy_meta(tr_gen_poly *d, const tr_gen_poly *s) {
    d->mass = s->mass; d->wscale = s->wscale; d->rows = s->rows; d->nulls = s->nulls;
    d->scale[0] = s->scale[0]; d->scale[1] = s->scale[1];
}
/* copies of the coefficients of a view (owned terms) */
static gterm *own_terms(const gterm *v, uint64_t n, const tr_math_budget *b, tr_math_refusal *r, int32_t *err) {
    gterm *t = zalloc(tr_budget_mul(n, sizeof(gterm)));
    if (!t) { set_error(err, TR_MATH_MEMORY); return NULL; }
    int32_t e = TR_MATH_OK;
    for (uint64_t i = 0; i < n && !e; i++) {
        t[i] = v[i];
        t[i].c = tr_bigint_copy(v[i].c, b, r, &e);
    }
    if (e) { free_terms(t, n); set_error(err, e); return NULL; }
    return t;
}

/* ------------------------------------------------------------------ constructors */
static int32_t shift_check(int nvars, const int64_t *given, const int64_t *vmin, int any, int64_t *shift) {
    for (int c = 0; c < nvars; c++) {
        if (!given) shift[c] = any ? vmin[c] : 0;
        else {
            if (any && given[c] > vmin[c]) return TR_MATH_DOMAIN_ERROR;   /* a value below the declared shift */
            shift[c] = given[c];
        }
    }
    if (nvars == 1) shift[1] = 0;
    return TR_MATH_OK;
}

tr_gen_poly *tr_gen_from_hist(const tr_dist_hist *h, uint64_t g, int32_t mass, int32_t repr, const int64_t *shift,
                              const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    tr_dist_states st;
    tr_dist_hist_info hi;
    if (tr_dist_hist_get_states(h, &st) || tr_dist_hist_get_info(h, &hi) || g >= st.ngroups ||
        (repr != TR_GEN_DENSE && repr != TR_GEN_SPARSE) || (mass != TR_DIST_MASS_COUNT && mass != TR_DIST_MASS_WEIGHT) ||
        (mass == TR_DIST_MASS_WEIGHT && !hi.weighted)) { set_error(error, TR_MATH_INVALID); return NULL; }
    if (hi.k > TR_GEN_MAX_VARS) { set_error(error, TR_MATH_UNSUPPORTED); return NULL; }
    const int nv = hi.k;
    tr_dist_plan plan;
    if (st.repr == TR_DIST_DENSE) tr_dist_states_plan(&st, &plan);
    uint64_t b, e;
    if (st.repr == TR_DIST_DENSE) { b = g * st.cells; e = b + st.cells; } else { b = st.offset[g]; e = st.offset[g + 1]; }
    /* pass 1: the terms, the NULL rows, the smallest and largest values */
    uint64_t n = 0, nulls = 0;
    int64_t vmin[2] = {0, 0}, vmax[2] = {0, 0}, val[2];
    for (uint64_t i = b; i < e; i++) {
        if (!st.count[i]) continue;
        int null = 0;
        for (int c = 0; c < nv; c++) {
            int nl;
            if (st.repr == TR_DIST_DENSE) val[c] = tr_dist_plan_coord(&plan, i - b, c, &nl);
            else { nl = (st.nulls[i] >> c) & 1; val[c] = st.value[c][i]; }
            null |= nl;
        }
        if (null) { nulls += st.count[i]; continue; }
        if (mass == TR_DIST_MASS_WEIGHT && tr_i128_get(st.wsum[i]) == 0) continue;
        for (int c = 0; c < nv; c++) {
            if (!n || val[c] < vmin[c]) vmin[c] = val[c];
            if (!n || val[c] > vmax[c]) vmax[c] = val[c];
        }
        n++;
    }
    int64_t s[2];
    int32_t rc = shift_check(nv, shift, vmin, n > 0, s);
    if (rc) { set_error(error, rc); return NULL; }
    uint64_t span[2] = {1, 1};
    for (int c = 0; c < nv; c++) span[c] = n ? (uint64_t)vmax[c] - (uint64_t)s[c] + 1 : 0;
    if (n) for (int c = 0; c < nv; c++) if (span[c] == 0) { set_error(error, TR_MATH_OVERFLOW); return NULL; }  /* 2^64 */
    uint64_t stored = repr == TR_GEN_DENSE ? box_of(nv, span) : n;
    tr_math_need need;
    need_zero(&need);
    need.need[TR_BUDGET_VARIABLES] = (uint64_t)nv;
    need.need[TR_BUDGET_SUPPORT] = stored;
    need.need[TR_BUDGET_BYTES] = tr_budget_add(poly_bytes(nv, repr, stored, n, mass == TR_DIST_MASS_COUNT ? 1 : 2),
                                               tr_budget_mul(n, sizeof(gterm)));
    tr_math_budget dflt;
    const tr_math_budget *bud = budget_of(budget, &dflt);
    rc = check_need(bud, &need, refusal);
    if (rc) { set_error(error, rc); return NULL; }
    gterm *t = zalloc(tr_budget_mul(n, sizeof(gterm)));
    if (!t) { set_error(error, TR_MATH_MEMORY); return NULL; }
    uint64_t o = 0;
    for (uint64_t i = b; i < e && !rc; i++) {
        if (!st.count[i]) continue;
        int null = 0;
        for (int c = 0; c < nv; c++) {
            int nl;
            if (st.repr == TR_DIST_DENSE) val[c] = tr_dist_plan_coord(&plan, i - b, c, &nl);
            else { nl = (st.nulls[i] >> c) & 1; val[c] = st.value[c][i]; }
            null |= nl;
        }
        if (null || (mass == TR_DIST_MASS_WEIGHT && tr_i128_get(st.wsum[i]) == 0)) continue;
        for (int c = 0; c < 2; c++) t[o].e[c] = c < nv ? (uint64_t)val[c] - (uint64_t)s[c] : 0;
        t[o].c = mass == TR_DIST_MASS_COUNT ? tr_bigint_from_limbs(1, &st.count[i], 1, bud, refusal, &rc)
                                            : tr_bigint_from_i128(st.wsum[i], bud, refusal, &rc);
        o++;
    }
    if (rc) { free_terms(t, o); set_error(error, rc); return NULL; }
    tr_gen_poly *p = poly_make(nv, s, t, n, repr, repr == TR_GEN_DENSE ? span : NULL, bud, refusal, error);
    if (!p) return NULL;
    p->mass = mass; p->rows = st.rows[g]; p->nulls = nulls; p->wscale = mass == TR_DIST_MASS_WEIGHT ? hi.wscale : 0;
    for (int c = 0; c < nv; c++) p->scale[c] = hi.scale[c];
    return p;
}

tr_gen_poly *tr_gen_from_terms(int32_t nvars, uint64_t n, const int64_t *values, const tr_math_i128 *coef, int32_t repr,
                               const int64_t *shift, const tr_math_budget *budget, tr_math_refusal *refusal,
                               int32_t *error) {
    tr_budget_none(refusal);
    if (nvars < 1 || nvars > TR_GEN_MAX_VARS || (n && (!values || !coef)) || (repr != TR_GEN_DENSE && repr != TR_GEN_SPARSE)) {
        set_error(error, TR_MATH_INVALID); return NULL;
    }
    /* the merged terms: n coefficients of at most 2 limbs, a sum of n of them at most 3 */
    tr_math_need need;
    need_zero(&need);
    need.need[TR_BUDGET_VARIABLES] = (uint64_t)nvars;
    need.need[TR_BUDGET_SUPPORT] = n;
    need.need[TR_BUDGET_BYTES] = tr_budget_add(tr_budget_mul(n, 2 * sizeof(gterm)), poly_bytes(nvars, TR_GEN_SPARSE, n, n, 3));
    tr_math_budget dflt;
    const tr_math_budget *bud = budget_of(budget, &dflt);
    int32_t rc = check_need(bud, &need, refusal);
    if (rc) { set_error(error, rc); return NULL; }
    /* sort by value (as offsets from INT64_MIN: order-preserving unsigned exponents), merge duplicates */
    gterm *t = zalloc(tr_budget_mul(n, sizeof(gterm))), *tmp = zalloc(tr_budget_mul(n, sizeof(gterm)));
    if (!t || !tmp) { free(t); free(tmp); set_error(error, TR_MATH_MEMORY); return NULL; }
    for (uint64_t i = 0; i < n && !rc; i++) {
        for (int c = 0; c < 2; c++) t[i].e[c] = c < nvars ? (uint64_t)values[i * (uint64_t)nvars + (uint64_t)c] ^ (UINT64_C(1) << 63) : 0;
        t[i].c = tr_bigint_from_i128(coef[i], bud, refusal, &rc);
    }
    if (rc) { free(tmp); free_terms(t, n); set_error(error, rc); return NULL; }
    term_sort(t, tmp, n);
    free(tmp);
    uint64_t m = 0;                                     /* every coefficient lives in exactly one slot */
    for (uint64_t i = 0; i < n && !rc; i++) {
        if (m && !term_cmp(&t[m - 1], &t[i])) {
            tr_bigint *s = tr_bigint_add(t[m - 1].c, t[i].c, bud, refusal, &rc);
            tr_bigint_free(t[i].c); t[i].c = NULL;
            if (s) { tr_bigint_free(t[m - 1].c); t[m - 1].c = s; }
        } else { gterm x = t[i]; t[i].c = NULL; t[m++] = x; }
    }
    if (rc) { free_terms(t, n); set_error(error, rc); return NULL; }
    int64_t vmin[2] = {0, 0}, s[2];
    int any = 0;
    for (uint64_t i = 0; i < m; i++) {
        if (is_zero(t[i].c)) continue;
        for (int c = 0; c < nvars; c++) {
            int64_t v = (int64_t)(t[i].e[c] ^ (UINT64_C(1) << 63));
            if (!any || v < vmin[c]) vmin[c] = v;
        }
        any = 1;
    }
    rc = shift_check(nvars, shift, vmin, any, s);
    if (rc) { free_terms(t, m); set_error(error, rc); return NULL; }
    for (uint64_t i = 0; i < m; i++)
        for (int c = 0; c < nvars; c++) {
            int64_t v = (int64_t)(t[i].e[c] ^ (UINT64_C(1) << 63));
            t[i].e[c] = (uint64_t)v - (uint64_t)s[c];               /* v >= s for every nonzero term */
        }
    /* zero terms may lie below the shift: drop them before the order is used (they carry no value) */
    uint64_t k = 0;
    for (uint64_t i = 0; i < m; i++) { if (is_zero(t[i].c)) tr_bigint_free(t[i].c); else t[k++] = t[i]; }
    return poly_make(nvars, s, t, k, repr, NULL, bud, refusal, error);
}

tr_gen_poly *tr_gen_from_dense_i128(int64_t shift, uint64_t n, const tr_math_i128 *coef, int32_t repr,
                                    const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    int64_t v;
    if ((n && !coef) || (repr != TR_GEN_DENSE && repr != TR_GEN_SPARSE) || (n && !value_of(n - 1, shift, &v))) {
        set_error(error, n && coef && (repr == TR_GEN_DENSE || repr == TR_GEN_SPARSE) ? TR_MATH_OVERFLOW : TR_MATH_INVALID);
        return NULL;
    }
    tr_math_need need;
    need_zero(&need);
    need.need[TR_BUDGET_VARIABLES] = 1;
    need.need[TR_BUDGET_SUPPORT] = n;
    need.need[TR_BUDGET_BYTES] = tr_budget_add(tr_budget_mul(n, sizeof(gterm)), poly_bytes(1, repr, n, n, 2));
    tr_math_budget dflt;
    const tr_math_budget *bud = budget_of(budget, &dflt);
    int32_t rc = check_need(bud, &need, refusal);
    if (rc) { set_error(error, rc); return NULL; }
    gterm *t = zalloc(tr_budget_mul(n, sizeof(gterm)));
    if (!t) { set_error(error, TR_MATH_MEMORY); return NULL; }
    for (uint64_t i = 0; i < n && !rc; i++) { t[i].e[0] = i; t[i].c = tr_bigint_from_i128(coef[i], bud, refusal, &rc); }
    if (rc) { free_terms(t, n); set_error(error, rc); return NULL; }
    int64_t s[2] = {shift, 0};
    uint64_t span[2] = {n, 1};
    return poly_make(1, s, t, n, repr, repr == TR_GEN_DENSE ? span : NULL, bud, refusal, error);
}

/* ------------------------------------------------------------------ reading */
int32_t tr_gen_get_info(const tr_gen_poly *p, tr_gen_info *out) {
    if (!poly_valid(p) || !out) return TR_MATH_INVALID;
    memset(out, 0, sizeof *out);
    out->nterms = p->nterms; out->nonzero = p->nonzero; out->box = p->box;
    out->span[0] = p->span[0]; out->span[1] = p->span[1];
    out->shift[0] = p->shift[0]; out->shift[1] = p->shift[1];
    out->rows = p->rows; out->nulls = p->nulls; out->maxbits = p->maxbits; out->bytes = p->bytes;
    out->nvars = p->nvars; out->repr = p->repr; out->mass = p->mass; out->nonneg = p->nonneg;
    out->scale[0] = p->scale[0]; out->scale[1] = p->scale[1]; out->wscale = p->wscale;
    for (int c = 0; c < 2; c++) { out->vmin[c] = 0; out->vmax[c] = -1; }
    int any = 0;
    for (uint64_t i = 0; i < p->nterms; i++) {
        if (is_zero(p->coef[i])) continue;
        uint64_t e[2];
        if (p->repr == TR_GEN_DENSE) { e[0] = p->nvars == 1 ? i : i / p->span[1]; e[1] = p->nvars == 1 ? 0 : i % p->span[1]; }
        else { e[0] = p->exps[i * (uint64_t)p->nvars]; e[1] = p->nvars > 1 ? p->exps[i * (uint64_t)p->nvars + 1] : 0; }
        for (int c = 0; c < p->nvars; c++) {
            int64_t v = (int64_t)((uint64_t)p->shift[c] + e[c]);
            if (!any || v < out->vmin[c]) out->vmin[c] = v;
            if (!any || v > out->vmax[c]) out->vmax[c] = v;
        }
        any = 1;
    }
    return TR_MATH_OK;
}

int32_t tr_gen_export_size(const tr_gen_poly *p, uint64_t *nonzero, uint64_t *limbs) {
    if (!poly_valid(p) || !nonzero || !limbs) return TR_MATH_INVALID;
    uint64_t l = 0;
    for (uint64_t i = 0; i < p->nterms; i++) if (!is_zero(p->coef[i])) l += limbs_of(p->coef[i]);
    *nonzero = p->nonzero; *limbs = l;
    return TR_MATH_OK;
}

int32_t tr_gen_export(const tr_gen_poly *p, int64_t *values, int32_t *signs, uint64_t *loff, uint64_t *limbs,
                      uint64_t nonzero_capacity, uint64_t limb_capacity) {
    uint64_t nz, nl;
    if (tr_gen_export_size(p, &nz, &nl) || nonzero_capacity < nz || limb_capacity < nl || (nz && (!values || !signs)) ||
        !loff || (nl && !limbs)) return TR_MATH_INVALID;
    uint64_t n = 0;
    gterm *t = poly_view(p, &n);
    if (!t && p->nonzero) return TR_MATH_MEMORY;
    uint64_t at = 0;
    loff[0] = 0;
    for (uint64_t i = 0; i < n; i++) {
        for (int c = 0; c < p->nvars; c++) values[i * (uint64_t)p->nvars + (uint64_t)c] = (int64_t)((uint64_t)p->shift[c] + t[i].e[c]);
        int32_t sign;
        uint64_t count;
        tr_bigint_export(t[i].c, &sign, limbs + at, limb_capacity - at, &count);
        signs[i] = sign;
        at += count;
        loff[i + 1] = at;
    }
    free(t);
    return TR_MATH_OK;
}

/* the stored coefficient at exponents e (NULL when absent or zero) */
static const tr_bigint *coeff_at(const tr_gen_poly *p, const uint64_t *e) {
    for (int c = 0; c < p->nvars; c++) if (e[c] >= p->span[c]) return NULL;
    if (p->repr == TR_GEN_DENSE) return p->coef[p->nvars == 1 ? e[0] : e[0] * p->span[1] + e[1]];
    uint64_t lo = 0, hi = p->nterms;
    gterm key = {{e[0], p->nvars > 1 ? e[1] : 0}, NULL};
    while (lo < hi) {
        uint64_t m = lo + (hi - lo) / 2;
        gterm x = {{p->exps[m * (uint64_t)p->nvars], p->nvars > 1 ? p->exps[m * (uint64_t)p->nvars + 1] : 0}, NULL};
        int s = term_cmp(&x, &key);
        if (!s) return p->coef[m];
        if (s < 0) lo = m + 1; else hi = m;
    }
    return NULL;
}

int32_t tr_gen_coeff(const tr_gen_poly *p, const int64_t *value, const tr_math_budget *budget, tr_math_refusal *refusal,
                     tr_bigint **out) {
    tr_budget_none(refusal);
    if (!poly_valid(p) || !value || !out) return TR_MATH_INVALID;
    *out = NULL;
    uint64_t e[2] = {0, 0};
    int inside = 1;
    for (int c = 0; c < p->nvars; c++) {
        if (value[c] < p->shift[c]) inside = 0;
        else e[c] = (uint64_t)value[c] - (uint64_t)p->shift[c];
    }
    const tr_bigint *x = inside ? coeff_at(p, e) : NULL;
    int32_t rc = TR_MATH_OK;
    *out = x ? tr_bigint_copy(x, budget, refusal, &rc) : tr_bigint_from_i64(0, budget, refusal, &rc);
    return rc;
}

int32_t tr_gen_equal(const tr_gen_poly *a, const tr_gen_poly *b) {
    if (!poly_valid(a) || !poly_valid(b)) return TR_MATH_INVALID;
    if (a->nvars != b->nvars || a->nonzero != b->nonzero) return 0;
    uint64_t na, nb;
    gterm *ta = poly_view(a, &na), *tb = poly_view(b, &nb);
    int eq = ta && tb && na == nb;
    for (uint64_t i = 0; eq && i < na; i++) {
        for (int c = 0; c < a->nvars; c++)
            eq &= (__int128)ta[i].e[c] + a->shift[c] == (__int128)tb[i].e[c] + b->shift[c];
        eq = eq && tr_bigint_cmp(ta[i].c, tb[i].c) == 0;
    }
    if (!ta || !tb) eq = na == 0 && nb == 0 ? 1 : TR_MATH_MEMORY;
    free(ta); free(tb);
    return eq;
}

int32_t tr_gen_dense_i64(const tr_gen_poly *p, int64_t *out, uint64_t capacity) {
    if (!poly_valid(p) || p->nvars != 1 || !out || capacity < p->span[0]) return TR_MATH_INVALID;
    for (uint64_t e = 0; e < p->span[0]; e++) out[e] = 0;
    for (uint64_t i = 0; i < p->nterms; i++) {
        if (is_zero(p->coef[i])) continue;
        int64_t v;
        if (tr_bigint_to_i64(p->coef[i], &v)) return TR_MATH_OVERFLOW;
        out[p->repr == TR_GEN_DENSE ? i : p->exps[i]] = v;
    }
    return TR_MATH_OK;
}

/* ------------------------------------------------------------------ evaluation */
static int32_t step(tr_bigint **acc, const tr_bigint *x, int op, const tr_math_budget *b, tr_math_refusal *r) {
    int32_t e = TR_MATH_OK;
    tr_bigint *t = op == 1 ? tr_bigint_mul(*acc, x, b, r, &e) : tr_bigint_add(*acc, x, b, r, &e);
    if (!t) return e ? e : TR_MATH_MEMORY;
    tr_bigint_free(*acc);
    *acc = t;
    return TR_MATH_OK;
}
static uint64_t bits_u64(uint64_t v) { uint64_t n = 0; while (v) { n++; v >>= 1; } return n; }

/* N = sum x_i p^(e_i) q^(D - e_i) over the terms t[0..n) (ascending in e[c], distinct) of one variable c, x_i = row[i]
 * or the coefficient: the homogeneous Horner from the top, skipping the gaps between exponents (p^gap, q^gap), so the
 * work is the number of terms: S = x_top q^(D - e_top), S = S p^gap + x q^(D - e) downwards, N = S p^(e_lowest) */
static int32_t pow_step(tr_bigint **acc, const tr_bigint *base, uint64_t e, const tr_math_budget *b, tr_math_refusal *r) {
    if (!e) return TR_MATH_OK;
    int32_t rc = TR_MATH_OK;
    tr_bigint *f = tr_bigint_pow(base, e, b, r, &rc);
    if (f) { rc = step(acc, f, 1, b, r); tr_bigint_free(f); }
    return rc;
}
static int32_t horner(const gterm *t, uint64_t n, int c, uint64_t D, const tr_bigint *P, const tr_bigint *Q, int qone,
                      tr_bigint *const *row, tr_bigint **out, const tr_math_budget *b, tr_math_refusal *r) {
    int32_t rc = TR_MATH_OK;
    if (!n) { *out = tr_bigint_from_i64(0, b, r, &rc); return rc; }
    tr_bigint *qp = tr_bigint_from_i64(1, b, r, &rc), *acc = NULL;
    if (!rc && !qone) rc = pow_step(&qp, Q, D - t[n - 1].e[c], b, r);
    for (uint64_t i = n; i-- > 0 && !rc;) {
        if (i + 1 < n) {
            uint64_t gap = t[i + 1].e[c] - t[i].e[c];
            rc = pow_step(&acc, P, gap, b, r);
            if (!rc && !qone) rc = pow_step(&qp, Q, gap, b, r);
        }
        const tr_bigint *x = row ? row[i] : t[i].c;
        tr_bigint *tq = rc ? NULL : (qone ? tr_bigint_copy(x, b, r, &rc) : tr_bigint_mul(x, qp, b, r, &rc));
        if (!tq) break;
        if (!acc) acc = tq;
        else { rc = step(&acc, tq, 0, b, r); tr_bigint_free(tq); }
    }
    if (!rc) rc = pow_step(&acc, P, t[0].e[c], b, r);
    tr_bigint_free(qp);
    if (rc) { tr_bigint_free(acc); return rc; }
    *out = acc;
    return TR_MATH_OK;
}

int32_t tr_gen_eval(const tr_gen_poly *p, const int64_t *num, const int64_t *den, const tr_math_budget *budget,
                    tr_math_refusal *refusal, tr_bigint **onum, tr_bigint **oden) {
    tr_budget_none(refusal);
    if (!poly_valid(p) || !num || !den || !onum || !oden) return TR_MATH_INVALID;
    *onum = *oden = NULL;
    const int nv = p->nvars;
    for (int c = 0; c < nv; c++) if (den[c] < 1) return TR_MATH_INVALID;
    uint64_t n = 0;
    gterm *t = poly_view(p, &n);
    if (!t && p->nonzero) return TR_MATH_MEMORY;
    /* a variable at 0: only the terms of value 0 there survive; a nonzero term of negative value is a pole */
    int64_t s[2] = {p->shift[0], p->shift[1]};
    uint64_t m = n;
    for (int c = 0; c < nv; c++) {
        if (num[c] != 0) continue;
        uint64_t k = 0;
        for (uint64_t i = 0; i < m; i++) {
            __int128 v = (__int128)t[i].e[c] + s[c];
            if (v < 0) { free(t); return TR_MATH_NO_SOLUTION; }
            if (v == 0) { t[k] = t[i]; t[k].e[c] = 0; k++; }
        }
        m = k; s[c] = 0;
    }
    uint64_t D[2] = {0, 0};
    for (uint64_t i = 0; i < m; i++) for (int c = 0; c < nv; c++) if (t[i].e[c] > D[c]) D[c] = t[i].e[c];
    /* the size of every number before any is made: bits <= maxbits + sum_c (D_c + |s_c|) bits(max(|p_c|, q_c)) + log n */
    uint64_t bits = tr_budget_add(p->maxbits, bits_u64(m) + 2);
    for (int c = 0; c < nv; c++) {                                  /* ceil(log2 max(|p|, q)) bits per power */
        uint64_t mag = tr_abs_u64(num[c]) > (uint64_t)den[c] ? tr_abs_u64(num[c]) : (uint64_t)den[c];
        uint64_t as = s[c] < 0 ? (uint64_t)0 - (uint64_t)s[c] : (uint64_t)s[c];
        bits = tr_budget_add(bits, tr_budget_add(tr_budget_mul(tr_budget_add(D[c], as), bits_u64(mag - 1)), 2));
    }
    tr_math_need need;
    need_zero(&need);
    need.need[TR_BUDGET_VARIABLES] = (uint64_t)nv;
    need.need[TR_BUDGET_SUPPORT] = m;
    need.need[TR_BUDGET_BYTES] = tr_budget_mul(TR_GEN_BIG_BYTES(bits / 64 + 2), 8);
    tr_math_budget dflt;
    const tr_math_budget *bud = budget_of(budget, &dflt);
    int32_t rc = check_need(bud, &need, refusal);
    if (rc) { free(t); return rc; }
    tr_bigint *P[2] = {NULL, NULL}, *Q[2] = {NULL, NULL}, *N = NULL, *Dn = NULL, **rows = NULL;
    gterm *rt = NULL;
    for (int c = 0; c < nv && !rc; c++) {
        P[c] = tr_bigint_from_i64(num[c], bud, refusal, &rc);
        if (!rc) Q[c] = tr_bigint_from_i64(den[c], bud, refusal, &rc);
    }
    if (!rc && nv == 1) rc = horner(t, m, 0, D[0], P[0], Q[0], den[0] == 1, NULL, &N, bud, refusal);
    else if (!rc) {
        /* the row values R(e_0) = sum_e1 c p_1^e1 q_1^(D_1 - e1), then the Horner in e_0 over them */
        rows = calloc((size_t)(m ? m : 1), sizeof *rows);
        rt = zalloc(tr_budget_mul(m, sizeof(gterm)));
        if (!rows || !rt) rc = TR_MATH_MEMORY;
        uint64_t nr = 0;
        for (uint64_t i = 0; i < m && !rc;) {
            uint64_t j = i;
            while (j < m && t[j].e[0] == t[i].e[0]) j++;
            rc = horner(t + i, j - i, 1, D[1], P[1], Q[1], den[1] == 1, NULL, &rows[nr], bud, refusal);
            rt[nr].e[0] = t[i].e[0]; rt[nr].c = NULL;
            nr++;
            i = j;
        }
        if (!rc) rc = horner(rt, nr, 0, D[0], P[0], Q[0], den[0] == 1, rows, &N, bud, refusal);
        for (uint64_t i = 0; i < nr; i++) tr_bigint_free(rows[i]);
    }
    /* the denominator q^D, then the Laurent factors p^s / q^s or q^-s / p^-s */
    if (!rc) Dn = tr_bigint_from_i64(1, bud, refusal, &rc);
    for (int c = 0; c < nv && !rc; c++) {
        tr_bigint *f = NULL;
        if (D[c] && den[c] != 1) {
            f = tr_bigint_pow(Q[c], D[c], bud, refusal, &rc);
            if (f) { rc = step(&Dn, f, 1, bud, refusal); tr_bigint_free(f); f = NULL; }
        }
        if (rc || !s[c]) continue;
        uint64_t as = s[c] < 0 ? (uint64_t)0 - (uint64_t)s[c] : (uint64_t)s[c];
        tr_bigint *up = s[c] > 0 ? P[c] : Q[c], *down = s[c] > 0 ? Q[c] : P[c];
        f = tr_bigint_pow(up, as, bud, refusal, &rc);
        if (f) { rc = step(&N, f, 1, bud, refusal); tr_bigint_free(f); f = NULL; }
        if (!rc) f = tr_bigint_pow(down, as, bud, refusal, &rc);
        if (f) { rc = step(&Dn, f, 1, bud, refusal); tr_bigint_free(f); }
    }
    tr_bigq *q = rc ? NULL : tr_bigq_from_bigints(N, Dn, bud, refusal, &rc);
    if (q) {
        *onum = tr_bigint_copy(tr_bigq_numerator(q), bud, refusal, &rc);
        if (!rc) *oden = tr_bigint_copy(tr_bigq_denominator(q), bud, refusal, &rc);
        if (rc) { tr_bigint_free(*onum); tr_bigint_free(*oden); *onum = *oden = NULL; }
        tr_bigq_free(q);
    }
    for (int c = 0; c < 2; c++) { tr_bigint_free(P[c]); tr_bigint_free(Q[c]); }
    tr_bigint_free(N); tr_bigint_free(Dn);
    free(rows); free(rt); free(t);
    return rc;
}

/* ------------------------------------------------------------------ moments and operators */
/* x *= v^k (raw) or x *= (v)_k (falling factorial) */
static int32_t times_power(tr_bigint **x, int64_t v, uint32_t k, int falling, const tr_math_budget *b, tr_math_refusal *r) {
    int32_t rc = TR_MATH_OK;
    if (!k) return rc;
    if (!falling) {
        tr_bigint *bv = tr_bigint_from_i64(v, b, r, &rc), *pw = bv ? tr_bigint_pow(bv, k, b, r, &rc) : NULL;
        if (pw) rc = step(x, pw, 1, b, r);
        tr_bigint_free(bv); tr_bigint_free(pw);
        return rc;
    }
    for (uint32_t i = 0; i < k && !rc; i++) {
        tr_bigint *f = big_i128((__int128)v - i, b, r, &rc);
        if (f) rc = step(x, f, 1, b, r);
        tr_bigint_free(f);
    }
    return rc;
}
static uint64_t factor_limbs(uint32_t k) { return ((uint64_t)k * 65 + 63) / 64; }

int32_t tr_gen_moment(const tr_gen_poly *p, const uint32_t *k, int32_t falling, const tr_math_budget *budget,
                      tr_math_refusal *refusal, tr_bigint **out) {
    tr_budget_none(refusal);
    if (!poly_valid(p) || !k || !out || (p->nvars == 1 && k[1])) return TR_MATH_INVALID;
    *out = NULL;
    uint32_t k1 = k[0], k2 = p->nvars > 1 ? k[1] : 0;
    tr_math_need need;
    need_zero(&need);
    need.need[TR_BUDGET_VARIABLES] = (uint64_t)p->nvars;
    need.need[TR_BUDGET_DEGREE] = (uint64_t)k1 + k2;
    uint64_t limbs = tr_budget_add(tr_budget_add(p->maxlimbs, factor_limbs(k1 + k2)), bits_u64(p->nonzero) / 64 + 2);
    need.need[TR_BUDGET_BYTES] = tr_budget_mul(TR_GEN_BIG_BYTES(limbs), 4);
    tr_math_budget dflt;
    const tr_math_budget *bud = budget_of(budget, &dflt);
    int32_t rc = check_need(bud, &need, refusal);
    if (rc) return rc;
    uint64_t n = 0;
    gterm *t = poly_view(p, &n);
    if (!t && p->nonzero) return TR_MATH_MEMORY;
    tr_bigint *sum = tr_bigint_from_i64(0, bud, refusal, &rc);
    for (uint64_t i = 0; i < n && !rc; i++) {
        tr_bigint *x = tr_bigint_copy(t[i].c, bud, refusal, &rc);
        int64_t v0 = (int64_t)((uint64_t)p->shift[0] + t[i].e[0]), v1 = (int64_t)((uint64_t)p->shift[1] + t[i].e[1]);
        if (!rc) rc = times_power(&x, v0, k1, falling, bud, refusal);
        if (!rc && p->nvars > 1) rc = times_power(&x, v1, k2, falling, bud, refusal);
        if (!rc) rc = step(&sum, x, 0, bud, refusal);
        tr_bigint_free(x);
    }
    free(t);
    if (rc) { tr_bigint_free(sum); return rc; }
    *out = sum;
    return TR_MATH_OK;
}

/* coefficients c v_c^k (euler) or c (v_c)_k (derivative, shift - k) */
static tr_gen_poly *operator_apply(const tr_gen_poly *p, int32_t coord, uint32_t k, int falling, const tr_math_budget *budget,
                                   tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    if (!poly_valid(p) || coord < 0 || coord >= p->nvars) { set_error(error, TR_MATH_INVALID); return NULL; }
    int64_t s[2] = {p->shift[0], p->shift[1]};
    if (falling) {
        __int128 ns = (__int128)s[coord] - k;
        if (ns < INT64_MIN) { set_error(error, TR_MATH_OVERFLOW); return NULL; }
        s[coord] = (int64_t)ns;
    }
    uint64_t limbs = tr_budget_add(p->maxlimbs, factor_limbs(k));
    tr_math_need need;
    need_zero(&need);
    need.need[TR_BUDGET_VARIABLES] = (uint64_t)p->nvars;
    need.need[TR_BUDGET_DEGREE] = k;
    need.need[TR_BUDGET_SUPPORT] = p->nterms;
    need.need[TR_BUDGET_BYTES] = tr_budget_add(tr_budget_mul(p->nonzero, sizeof(gterm)),
                                               poly_bytes(p->nvars, p->repr, p->nterms, p->nonzero, limbs));
    tr_math_budget dflt;
    const tr_math_budget *bud = budget_of(budget, &dflt);
    int32_t rc = check_need(bud, &need, refusal);
    if (rc) { set_error(error, rc); return NULL; }
    uint64_t n = 0;
    gterm *v = poly_view(p, &n), *t = v ? own_terms(v, n, bud, refusal, &rc) : NULL;
    free(v);
    if (!t) { set_error(error, rc ? rc : TR_MATH_MEMORY); return NULL; }
    for (uint64_t i = 0; i < n && !rc; i++) {
        int64_t val = (int64_t)((uint64_t)p->shift[coord] + t[i].e[coord]);
        rc = times_power(&t[i].c, val, k, falling, bud, refusal);
    }
    if (rc) { free_terms(t, n); set_error(error, rc); return NULL; }
    tr_gen_poly *r = poly_make(p->nvars, s, t, n, p->repr, p->repr == TR_GEN_DENSE ? p->span : NULL, bud, refusal, error);
    if (r) copy_meta(r, p);
    return r;
}
tr_gen_poly *tr_gen_euler(const tr_gen_poly *p, int32_t coord, uint32_t k, const tr_math_budget *budget,
                          tr_math_refusal *refusal, int32_t *error) {
    return operator_apply(p, coord, k, 0, budget, refusal, error);
}
tr_gen_poly *tr_gen_derivative(const tr_gen_poly *p, int32_t coord, uint32_t k, const tr_math_budget *budget,
                               tr_math_refusal *refusal, int32_t *error) {
    return operator_apply(p, coord, k, 1, budget, refusal, error);
}

/* ------------------------------------------------------------------ products */
/* the sizes of one product: terms n, coefficient limbs L, hull (largest exponent + 1 per variable) */
typedef struct { uint64_t n, L, hull[2]; int dense; } gsize;
typedef struct { uint64_t pairs, box, terms, bytes, worst, span[2]; int use_box, out_dense; } gplan;

static void plan_product(int nv, gsize a, gsize b, uint64_t order, int fast, const tr_math_budget *bud, gplan *pl) {
    memset(pl, 0, sizeof *pl);
    pl->out_dense = a.dense && b.dense;
    for (int c = 0; c < 2; c++) {
        pl->span[c] = c < nv ? (a.hull[c] && b.hull[c] ? tr_budget_add(a.hull[c], b.hull[c] - 1) : 0) : 1;
    }
    if (order && pl->span[0] > order) pl->span[0] = order;
    pl->box = box_of(nv, pl->span);
    pl->pairs = tr_budget_mul(a.n, b.n);
    uint64_t lim = tr_budget_add(tr_budget_mul(pl->pairs, TR_GEN_BOX_FACTOR), TR_GEN_BOX_SLACK);
    pl->use_box = pl->out_dense || (pl->box <= bud->limit[TR_BUDGET_SUPPORT] && pl->box <= lim);
    pl->terms = pl->out_dense ? pl->box : (pl->box < pl->pairs ? pl->box : pl->pairs);
    uint64_t L = tr_budget_add(tr_budget_add(a.L, b.L), 1);
    /* the scratch: a box of 16-byte accumulators (128-bit sums or big-integer pointers), or the sorted pairs */
    uint64_t sbox = tr_budget_mul(pl->box, 16), ssort = tr_budget_mul(pl->pairs, 2 * sizeof(gterm) + 16);
    uint64_t views = tr_budget_mul(tr_budget_add(a.n, b.n), sizeof(gterm) + 8);
    uint64_t out = tr_budget_add(tr_budget_mul(pl->terms, sizeof(gterm)),
                                 poly_bytes(nv, pl->out_dense ? TR_GEN_DENSE : TR_GEN_SPARSE, pl->terms, pl->terms, L));
    pl->bytes = tr_budget_add(tr_budget_add(pl->use_box ? sbox : ssort, views), out);
    /* a step with smaller actual sizes may pick the other strategy: when the box is chosen here, a sorting step has
     * 64 P' < 16 B' <= 16 B; when sorting is chosen, a box step has 16 B' <= 64 P' + 16 x slack <= 64 P + 16 x slack */
    pl->worst = pl->use_box ? pl->bytes : tr_budget_add(pl->bytes, 16 * (uint64_t)TR_GEN_BOX_SLACK);
    (void)fast;
}
static void plan_need(int nv, const gplan *pl, tr_math_need *need) {
    need_zero(need);
    need->need[TR_BUDGET_VARIABLES] = (uint64_t)nv;
    need->need[TR_BUDGET_TENSOR] = pl->pairs;
    need->need[TR_BUDGET_SUPPORT] = pl->terms;
    need->need[TR_BUDGET_BYTES] = pl->bytes;
}
static gsize size_of(const tr_gen_poly *p, uint64_t order) {
    gsize s;
    s.n = p->nonzero; s.L = p->maxlimbs; s.dense = p->repr == TR_GEN_DENSE;
    for (int c = 0; c < 2; c++) s.hull[c] = c < p->nvars ? p->span[c] : 1;
    if (order && s.hull[0] > order) s.hull[0] = order;
    return s;
}
/* every coefficient fits int64: *maxabs their largest magnitude */
static int fits_i64(const gterm *t, uint64_t n, int64_t *vals, uint64_t *maxabs) {
    uint64_t m = 0;
    for (uint64_t i = 0; i < n; i++) {
        int64_t v;
        if (tr_bigint_to_i64(t[i].c, &v)) return 0;
        if (vals) vals[i] = v;
        if (tr_abs_u64(v) > m) m = tr_abs_u64(v);
    }
    *maxabs = m;
    return 1;
}
static int fast_ok(uint64_t na, uint64_t ma, uint64_t nb, uint64_t mb) {
    tr_dist_i256 p = tr_dist_w_mul_u128((unsigned __int128)ma * mb, (unsigned __int128)(na < nb ? na : nb));
    return !p.w[2] && !p.w[3] && !(p.w[1] >> 63);
}
static int pair_cmp(const gterm *a, const gterm *b) { return term_cmp(a, b); }

static tr_gen_poly *product(const tr_gen_poly *a, const tr_gen_poly *b, uint64_t order, const tr_math_budget *bud,
                            tr_math_refusal *refusal, int32_t *error) {
    const int nv = a->nvars;
    int64_t s[2] = {0, 0};
    for (int c = 0; c < nv; c++) {
        __int128 x = (__int128)a->shift[c] + b->shift[c];
        if (x < INT64_MIN || x > INT64_MAX) { set_error(error, TR_MATH_OVERFLOW); return NULL; }
        s[c] = (int64_t)x;
    }
    uint64_t na = 0, nb = 0;
    gterm *ta = poly_view(a, &na), *tb = poly_view(b, &nb);
    if ((!ta && a->nonzero) || (!tb && b->nonzero)) { free(ta); free(tb); set_error(error, TR_MATH_MEMORY); return NULL; }
    /* every value of the product must fit int64 */
    for (int c = 0; c < nv && na && nb; c++) {
        __int128 hi = (__int128)(a->span[c] - 1) + (b->span[c] - 1) + s[c];
        if (hi > INT64_MAX) { free(ta); free(tb); set_error(error, TR_MATH_OVERFLOW); return NULL; }
    }
    int64_t *va = zalloc(tr_budget_mul(na, 8)), *vb = zalloc(tr_budget_mul(nb, 8));
    uint64_t ma = 0, mb = 0;
    int fast = va && vb && fits_i64(ta, na, va, &ma) && fits_i64(tb, nb, vb, &mb) && fast_ok(na, ma, nb, mb);
    gsize sa = size_of(a, order), sb = size_of(b, order);
    sa.n = na; sb.n = nb;
    gplan pl;
    plan_product(nv, sa, sb, order, fast, bud, &pl);
    tr_math_need need;
    plan_need(nv, &pl, &need);
    int32_t rc = (!va || !vb) ? TR_MATH_MEMORY : check_need(bud, &need, refusal);
    gterm *out = NULL;
    uint64_t nout = 0;
    if (!rc && pl.use_box) {
        const uint64_t B = pl.box, w1 = nv > 1 ? pl.span[1] : 1;
        __int128 *acc = fast ? zalloc(tr_budget_mul(B, 16)) : NULL;
        tr_bigint **bacc = fast ? NULL : zalloc(tr_budget_mul(B, sizeof(tr_bigint *)));
        if ((fast && !acc) || (!fast && !bacc)) rc = TR_MATH_MEMORY;
        for (uint64_t i = 0; i < na && !rc; i++)
            for (uint64_t j = 0; j < nb && !rc; j++) {
                uint64_t e0 = ta[i].e[0] + tb[j].e[0], e1 = ta[i].e[1] + tb[j].e[1];
                if (order && e0 >= order) continue;
                uint64_t idx = nv > 1 ? e0 * w1 + e1 : e0;
                if (fast) { acc[idx] += (__int128)va[i] * vb[j]; continue; }
                tr_bigint *pr = tr_bigint_mul(ta[i].c, tb[j].c, bud, refusal, &rc);
                if (!pr) break;
                if (!bacc[idx]) bacc[idx] = pr;
                else { rc = step(&bacc[idx], pr, 0, bud, refusal); tr_bigint_free(pr); }
            }
        out = rc ? NULL : zalloc(tr_budget_mul(pl.out_dense ? B : pl.terms, sizeof(gterm)));
        if (!rc && !out) rc = TR_MATH_MEMORY;
        for (uint64_t idx = 0; idx < B && !rc; idx++) {
            int nz = fast ? acc[idx] != 0 : bacc[idx] != NULL;
            if (!nz && !pl.out_dense) continue;
            out[nout].e[0] = nv > 1 ? idx / w1 : idx; out[nout].e[1] = nv > 1 ? idx % w1 : 0;
            if (fast) out[nout].c = nz ? big_i128(acc[idx], bud, refusal, &rc) : NULL;
            else { out[nout].c = bacc[idx]; bacc[idx] = NULL; }
            nout++;
        }
        if (bacc) { for (uint64_t idx = 0; idx < B; idx++) tr_bigint_free(bacc[idx]); free(bacc); }
        free(acc);
    } else if (!rc) {
        /* sort the pairs by their exponent sums, then sum each run */
        gterm *pr = zalloc(tr_budget_mul(pl.pairs, sizeof(gterm))), *tmp = zalloc(tr_budget_mul(pl.pairs, sizeof(gterm)));
        if (!pr || !tmp) rc = TR_MATH_MEMORY;
        uint64_t np = 0;
        for (uint64_t i = 0; i < na && !rc; i++)
            for (uint64_t j = 0; j < nb; j++) {
                uint64_t e0 = ta[i].e[0] + tb[j].e[0], e1 = ta[i].e[1] + tb[j].e[1];
                if (order && e0 >= order) continue;
                pr[np].e[0] = e0; pr[np].e[1] = e1;
                pr[np].c = (tr_bigint *)(uintptr_t)(i * nb + j);           /* the pair's index, not a coefficient */
                np++;
            }
        if (!rc) {
            for (uint64_t w = 1; w < np; w *= 2) {                          /* stable merge sort on the sums */
                for (uint64_t st = 0; st < np; st += 2 * w) {
                    uint64_t m = st + w < np ? st + w : np, e = st + 2 * w < np ? st + 2 * w : np, i = st, j = m, o = st;
                    while (i < m && j < e) tmp[o++] = pair_cmp(&pr[j], &pr[i]) < 0 ? pr[j++] : pr[i++];
                    while (i < m) tmp[o++] = pr[i++];
                    while (j < e) tmp[o++] = pr[j++];
                }
                memcpy(pr, tmp, (size_t)np * sizeof *pr);
            }
            out = zalloc(tr_budget_mul(np, sizeof(gterm)));
            if (!out) rc = TR_MATH_MEMORY;
        }
        for (uint64_t i = 0; i < np && !rc;) {
            uint64_t j = i;
            __int128 acc = 0;
            tr_bigint *bacc = NULL;
            while (j < np && !term_cmp(&pr[j], &pr[i]) && !rc) {
                uint64_t id = (uint64_t)(uintptr_t)pr[j].c, x = id / nb, y = id % nb;
                if (fast) acc += (__int128)va[x] * vb[y];
                else {
                    tr_bigint *q = tr_bigint_mul(ta[x].c, tb[y].c, bud, refusal, &rc);
                    if (q && !bacc) bacc = q;
                    else if (q) { rc = step(&bacc, q, 0, bud, refusal); tr_bigint_free(q); }
                }
                j++;
            }
            if (!rc) {
                out[nout].e[0] = pr[i].e[0]; out[nout].e[1] = pr[i].e[1];
                out[nout].c = fast ? big_i128(acc, bud, refusal, &rc) : bacc;
                nout++;
            } else tr_bigint_free(bacc);
            i = j;
        }
        free(pr); free(tmp);
    }
    free(ta); free(tb); free(va); free(vb);
    if (rc) { free_terms(out, nout); set_error(error, rc); return NULL; }
    uint64_t span[2] = {pl.span[0], pl.span[1]};
    tr_gen_poly *r = poly_make(nv, s, out, nout, pl.out_dense ? TR_GEN_DENSE : TR_GEN_SPARSE,
                               pl.out_dense ? span : NULL, bud, refusal, error);
    if (r) { r->mass = a->mass == b->mass ? a->mass : 0; r->scale[0] = a->scale[0]; r->scale[1] = a->scale[1]; }
    return r;
}

static int32_t mul_check(const tr_gen_poly *a, const tr_gen_poly *b, uint64_t order) {
    if (!poly_valid(a) || !poly_valid(b) || a->nvars != b->nvars) return TR_MATH_INVALID;
    if (order && a->nvars != 1) return TR_MATH_UNSUPPORTED;
    if (a->nvars > 1 ? (a->scale[0] != b->scale[0] || a->scale[1] != b->scale[1]) : a->scale[0] != b->scale[0])
        return TR_MATH_DOMAIN_ERROR;                                  /* values of different decimal scales */
    return TR_MATH_OK;
}

tr_gen_poly *tr_gen_mul(const tr_gen_poly *a, const tr_gen_poly *b, uint64_t order, const tr_math_budget *budget,
                        tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    int32_t rc = mul_check(a, b, order);
    if (rc) { set_error(error, rc); return NULL; }
    tr_math_budget dflt;
    return product(a, b, order, budget_of(budget, &dflt), refusal, error);
}

int32_t tr_gen_mul_need(const tr_gen_poly *a, const tr_gen_poly *b, uint64_t order, const tr_math_budget *budget,
                        tr_math_need *need, uint64_t *terms) {
    int32_t rc = mul_check(a, b, order);
    if (rc || !need) return rc ? rc : TR_MATH_INVALID;
    tr_math_budget dflt;
    const tr_math_budget *bud = budget_of(budget, &dflt);
    gplan pl;
    plan_product(a->nvars, size_of(a, order), size_of(b, order), order, 0, bud, &pl);
    plan_need(a->nvars, &pl, need);
    if (terms) *terms = pl.terms;
    return TR_MATH_OK;
}

/* the chain of a power: step j multiplies G^(j-1) by G; the need is the largest step (with G^(j-1) held) */
static int32_t pow_plan(const tr_gen_poly *a, uint32_t k, uint64_t order, const tr_math_budget *bud, tr_math_need *need,
                        uint64_t *terms) {
    need_zero(need);
    need->need[TR_BUDGET_VARIABLES] = (uint64_t)a->nvars;
    gsize g1 = size_of(a, order), cur = g1;
    uint64_t t = g1.n;
    for (uint32_t j = 2; j <= k; j++) {
        gplan pl;
        plan_product(a->nvars, cur, g1, order, 0, bud, &pl);
        tr_math_need step_need;
        plan_need(a->nvars, &pl, &step_need);
        /* the actual step (smaller sizes) may pick the other strategy: bound the scratch of both; G^(j-1) is held */
        step_need.need[TR_BUDGET_BYTES] = tr_budget_add(pl.worst,
                                                        poly_bytes(a->nvars, cur.dense ? TR_GEN_DENSE : TR_GEN_SPARSE,
                                                                   cur.n, cur.n, cur.L));
        if (pl.use_box) step_need.need[TR_BUDGET_SUPPORT] = pl.terms;
        for (int i = 0; i < TR_BUDGET_RESOURCES; i++)
            if (step_need.need[i] > need->need[i]) need->need[i] = step_need.need[i];
        cur.n = pl.terms; cur.L = tr_budget_add(tr_budget_add(cur.L, g1.L), 1); cur.dense = pl.out_dense;
        cur.hull[0] = pl.span[0]; cur.hull[1] = pl.span[1];
        t = pl.terms;
        if (need->need[TR_BUDGET_BYTES] == TR_BUDGET_UNKNOWN) break;
    }
    if (terms) *terms = k == 0 ? 1 : t;
    return TR_MATH_OK;
}

int32_t tr_gen_pow_need(const tr_gen_poly *a, uint32_t k, uint64_t order, const tr_math_budget *budget, tr_math_need *need,
                        uint64_t *terms) {
    if (!poly_valid(a) || !need) return TR_MATH_INVALID;
    if (order && a->nvars != 1) return TR_MATH_UNSUPPORTED;
    tr_math_budget dflt;
    return pow_plan(a, k, order, budget_of(budget, &dflt), need, terms);
}

tr_gen_poly *tr_gen_truncate(const tr_gen_poly *p, uint64_t order, const tr_math_budget *budget, tr_math_refusal *refusal,
                             int32_t *error) {
    tr_budget_none(refusal);
    if (!poly_valid(p) || !order) { set_error(error, TR_MATH_INVALID); return NULL; }
    if (p->nvars != 1) { set_error(error, TR_MATH_UNSUPPORTED); return NULL; }
    tr_math_budget dflt;
    const tr_math_budget *bud = budget_of(budget, &dflt);
    uint64_t n = 0, k = 0;
    gterm *v = poly_view(p, &n);
    if (!v && p->nonzero) { set_error(error, TR_MATH_MEMORY); return NULL; }
    while (k < n && v[k].e[0] < order) k++;
    int32_t rc = TR_MATH_OK;
    gterm *t = own_terms(v, k, bud, refusal, &rc);
    free(v);
    if (!t) { set_error(error, rc ? rc : TR_MATH_MEMORY); return NULL; }
    uint64_t span[2] = {p->span[0] < order ? p->span[0] : order, 1};
    tr_gen_poly *r = poly_make(1, p->shift, t, k, p->repr, p->repr == TR_GEN_DENSE ? span : NULL, bud, refusal, error);
    if (r) copy_meta(r, p);
    return r;
}

tr_gen_poly *tr_gen_pow(const tr_gen_poly *a, uint32_t k, uint64_t order, const tr_math_budget *budget,
                        tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    if (!poly_valid(a)) { set_error(error, TR_MATH_INVALID); return NULL; }
    if (order && a->nvars != 1) { set_error(error, TR_MATH_UNSUPPORTED); return NULL; }
    tr_math_budget dflt;
    const tr_math_budget *bud = budget_of(budget, &dflt);
    tr_math_need need;
    pow_plan(a, k, order, bud, &need, NULL);
    int32_t rc = check_need(bud, &need, refusal);
    if (rc) { set_error(error, rc); return NULL; }
    if (k == 0) {                                                     /* the constant 1 = z^0 */
        int64_t zero[2] = {0, 0};
        tr_math_i128 one = tr_i128_set(1);
        tr_gen_poly *r = tr_gen_from_terms(a->nvars, 1, zero, &one, a->repr, NULL, bud, refusal, error);
        if (r) { r->scale[0] = a->scale[0]; r->scale[1] = a->scale[1]; }
        return r;
    }
    tr_gen_poly *g = order ? tr_gen_truncate(a, order, bud, refusal, error) : NULL;
    if (order && !g) return NULL;
    const tr_gen_poly *base = g ? g : a;
    tr_gen_poly *r = NULL;
    for (uint32_t j = 2; j <= k; j++) {
        tr_gen_poly *next = product(r ? r : base, base, order, bud, refusal, error);
        tr_gen_free(r);
        if (!next) { tr_gen_free(g); return NULL; }
        r = next;
    }
    if (!r) {                                                         /* k == 1: a copy */
        uint64_t n = 0;
        gterm *v = poly_view(base, &n), *t = v ? own_terms(v, n, bud, refusal, &rc) : NULL;
        free(v);
        if (!t && base->nonzero) { tr_gen_free(g); set_error(error, rc ? rc : TR_MATH_MEMORY); return NULL; }
        r = poly_make(base->nvars, base->shift, t, n, base->repr, base->repr == TR_GEN_DENSE ? base->span : NULL, bud,
                      refusal, error);
        if (r) copy_meta(r, base);
    } else {
        r->mass = a->mass;
    }
    tr_gen_free(g);
    return r;
}

/* ------------------------------------------------------------------ conversions */
tr_series *tr_gen_to_series(const tr_gen_poly *p, uint32_t order, const tr_math_budget *budget, tr_math_refusal *refusal,
                            int32_t *error) {
    tr_budget_none(refusal);
    if (!poly_valid(p) || order < 1) { set_error(error, TR_MATH_INVALID); return NULL; }
    if (p->nvars != 1) { set_error(error, TR_MATH_UNSUPPORTED); return NULL; }
    tr_math_budget dflt;
    const tr_math_budget *bud = budget_of(budget, &dflt);
    tr_math_need need;
    need_zero(&need);
    need.need[TR_BUDGET_VARIABLES] = 1;
    need.need[TR_BUDGET_DEGREE] = order - 1;
    need.need[TR_BUDGET_SUPPORT] = order;
    need.need[TR_BUDGET_BYTES] = tr_budget_mul((uint64_t)order, 2 * TR_GEN_BIG_BYTES(p->maxlimbs + 1) + 64);
    int32_t rc = check_need(bud, &need, refusal);
    if (rc) { set_error(error, rc); return NULL; }
    tr_bigq **q = calloc((size_t)order, sizeof *q);
    if (!q) { set_error(error, TR_MATH_MEMORY); return NULL; }
    for (uint32_t e = 0; e < order && !rc; e++) {
        uint64_t ex[2] = {e, 0};
        const tr_bigint *c = coeff_at(p, ex);
        if (c) q[e] = tr_bigq_from_bigint(c, bud, refusal, &rc);
        else q[e] = tr_bigq_from_q64(tr_q_int(0), bud, refusal, &rc);
    }
    tr_series *s = rc ? NULL : tr_series_create((const tr_bigq *const *)q, order, order, bud, refusal, &rc);
    for (uint32_t e = 0; e < order; e++) tr_bigq_free(q[e]);
    free(q);
    set_error(error, rc);
    return s;
}

tr_gen_poly *tr_gen_convert(const tr_gen_poly *p, int32_t repr, const tr_math_budget *budget, tr_math_refusal *refusal,
                            int32_t *error) {
    tr_budget_none(refusal);
    if (!poly_valid(p) || (repr != TR_GEN_DENSE && repr != TR_GEN_SPARSE)) { set_error(error, TR_MATH_INVALID); return NULL; }
    tr_math_budget dflt;
    const tr_math_budget *bud = budget_of(budget, &dflt);
    tr_math_need need;
    need_zero(&need);
    uint64_t stored = repr == TR_GEN_DENSE ? p->box : p->nonzero;
    need.need[TR_BUDGET_VARIABLES] = (uint64_t)p->nvars;
    need.need[TR_BUDGET_SUPPORT] = stored;
    need.need[TR_BUDGET_BYTES] = tr_budget_add(tr_budget_mul(p->nonzero, sizeof(gterm)),
                                               poly_bytes(p->nvars, repr, stored, p->nonzero, p->maxlimbs));
    int32_t rc = check_need(bud, &need, refusal);
    if (rc) { set_error(error, rc); return NULL; }
    uint64_t n = 0;
    gterm *v = poly_view(p, &n), *t = v ? own_terms(v, n, bud, refusal, &rc) : NULL;
    free(v);
    if (!t && p->nonzero) { set_error(error, rc ? rc : TR_MATH_MEMORY); return NULL; }
    tr_gen_poly *r = poly_make(p->nvars, p->shift, t, n, repr, repr == TR_GEN_DENSE ? p->span : NULL, bud, refusal, error);
    if (r) copy_meta(r, p);
    return r;
}

tr_gen_poly *tr_gen_marginal(const tr_gen_poly *p, int32_t keep, const tr_math_budget *budget, tr_math_refusal *refusal,
                             int32_t *error) {
    tr_budget_none(refusal);
    if (!poly_valid(p) || p->nvars != 2 || keep < 0 || keep > 1) { set_error(error, TR_MATH_INVALID); return NULL; }
    tr_math_budget dflt;
    const tr_math_budget *bud = budget_of(budget, &dflt);
    tr_math_need need;
    need_zero(&need);
    need.need[TR_BUDGET_VARIABLES] = 1;
    need.need[TR_BUDGET_SUPPORT] = p->nonzero;
    need.need[TR_BUDGET_BYTES] = tr_budget_add(tr_budget_mul(p->nonzero, 2 * sizeof(gterm)),
                                               poly_bytes(1, p->repr, p->repr == TR_GEN_DENSE ? p->span[keep] : p->nonzero,
                                                          p->nonzero, p->maxlimbs + 1));
    int32_t rc = check_need(bud, &need, refusal);
    if (rc) { set_error(error, rc); return NULL; }
    uint64_t n = 0;
    gterm *v = poly_view(p, &n);
    if (!v && p->nonzero) { set_error(error, TR_MATH_MEMORY); return NULL; }
    gterm *t = zalloc(tr_budget_mul(n, sizeof(gterm))), *tmp = zalloc(tr_budget_mul(n, sizeof(gterm)));
    if (!t || !tmp) { free(v); free(t); free(tmp); set_error(error, TR_MATH_MEMORY); return NULL; }
    for (uint64_t i = 0; i < n; i++) { t[i].e[0] = v[i].e[keep]; t[i].e[1] = 0; t[i].c = v[i].c; }   /* borrowed */
    term_sort(t, tmp, n);
    free(tmp); free(v);
    uint64_t m = 0;
    for (uint64_t i = 0; i < n && !rc;) {
        uint64_t j = i;
        tr_bigint *s = tr_bigint_from_i64(0, bud, refusal, &rc);
        while (j < n && t[j].e[0] == t[i].e[0] && !rc) { rc = step(&s, t[j].c, 0, bud, refusal); j++; }
        t[m].e[0] = t[i].e[0]; t[m].e[1] = 0; t[m].c = s;                /* owned from here on */
        m++;
        i = j;
    }
    if (rc) { free_terms(t, m); set_error(error, rc); return NULL; }
    int64_t s[2] = {p->shift[keep], 0};
    uint64_t span[2] = {p->span[keep], 1};
    tr_gen_poly *r = poly_make(1, s, t, m, p->repr, p->repr == TR_GEN_DENSE ? span : NULL, bud, refusal, error);
    if (r) { copy_meta(r, p); r->scale[0] = p->scale[keep]; r->scale[1] = 0; }
    return r;
}

/* ------------------------------------------------------------------ the CPU references of the kernels */
int32_t tr_gen_conv_i128(const int64_t *a, uint64_t na, const int64_t *b, uint64_t nb, tr_math_i128 *out,
                         const tr_math_budget *budget, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!a || !b || !out || !na || !nb) return TR_MATH_INVALID;
    tr_math_budget dflt;
    int32_t rc = tr_budget_check_one(budget_of(budget, &dflt), TR_BUDGET_OUTPUT, na + nb - 1, refusal);
    if (rc) return rc;
    if (!tr_gen_conv_exact(a, na, b, nb)) return TR_MATH_OVERFLOW;
    for (uint64_t k = 0; k < na + nb - 1; k++) out[k] = tr_i128_set(tr_gen_conv_one(a, na, b, nb, k));
    return TR_MATH_OK;
}

int32_t tr_gen_eval_i128(const int64_t *c, uint64_t n, int64_t shift, const tr_gen_point *z, uint64_t m,
                         tr_gen_value *out, const tr_math_budget *budget, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!c || !n || (m && (!z || !out))) return TR_MATH_INVALID;
    tr_math_budget dflt;
    int32_t rc = tr_budget_check_one(budget_of(budget, &dflt), TR_BUDGET_OUTPUT, m, refusal);
    if (rc) return rc;
    for (uint64_t i = 0; i < m; i++) tr_gen_eval_one(c, n, shift, z[i], &out[i]);
    return TR_MATH_OK;
}
