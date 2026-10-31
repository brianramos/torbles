/* DISC-002: Moment-Gram polynomial identity discovery; see math_ident.h.
 * SPDX-License-Identifier: Apache-2.0 */
#include "math_ident.h"
#include "mathematics.h"
#include "math_domain.h"
#include "math_handles.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

TR_MATH_ABI(tr_ident_spec, TR_IDENT_ABI);   /* the ctypes boundary of DISC-002 (math_handles.h) */
TR_MATH_ABI(tr_ident_moment_view, TR_IDENT_ABI);
TR_MATH_ABI(tr_ident_plan_info, TR_IDENT_ABI);
TR_MATH_ABI(tr_ident_gram_states, TR_IDENT_ABI);
TR_MATH_ABI(tr_ident_gram_info, TR_IDENT_ABI);
TR_MATH_ABI(tr_ident_kernel_info, TR_IDENT_ABI);
TR_MATH_ABI(tr_ident_poly, TR_IDENT_ABI);
TR_MATH_ABI(tr_ident_cand_view, TR_IDENT_ABI);
TR_MATH_ABI(tr_ident_item, TR_IDENT_ABI);
TR_MATH_ABI(tr_ident_set_info, TR_IDENT_ABI);
TR_MATH_ABI(tr_ident_quad, TR_IDENT_ABI);

_Static_assert(sizeof(tr_ident_spec) == 72, "tr_ident_spec ABI");
_Static_assert(sizeof(tr_ident_moment_view) == 104, "tr_ident_moment_view ABI");
_Static_assert(sizeof(tr_ident_plan_info) == 296, "tr_ident_plan_info ABI");
_Static_assert(sizeof(tr_ident_gram_states) == 64, "tr_ident_gram_states ABI");
_Static_assert(sizeof(tr_ident_gram_info) == 384, "tr_ident_gram_info ABI");
_Static_assert(sizeof(tr_ident_kernel_info) == 56, "tr_ident_kernel_info ABI");
_Static_assert(sizeof(tr_ident_poly) == 24, "tr_ident_poly ABI");
_Static_assert(sizeof(tr_ident_cand_view) == 88, "tr_ident_cand_view ABI");
_Static_assert(sizeof(tr_ident_item) == 96, "tr_ident_item ABI");
_Static_assert(sizeof(tr_ident_set_info) == 104, "tr_ident_set_info ABI");
_Static_assert(sizeof(tr_ident_quad) == 296, "tr_ident_quad ABI");

#define PLAN_MAGIC UINT32_C(0x4e4c5049)
#define GRAM_MAGIC UINT32_C(0x4d524749)
#define KER_MAGIC UINT32_C(0x524b4449)
#define SET_MAGIC UINT32_C(0x54455349)
#define I128_LIMIT ((((unsigned __int128)1) << 127) - 1)
#define LIN_ROUNDS 8

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
static void need_zero(tr_math_need *n) { for (int i = 0; i < TR_BUDGET_RESOURCES; i++) n->need[i] = 0; }
static int32_t check_one(const tr_math_budget *b, int32_t res, uint64_t need, tr_math_refusal *r) {
    tr_math_budget dflt;
    return tr_budget_check_one(budget_or(b, &dflt), res, need, r);
}
static int32_t check_need(const tr_math_budget *b, const tr_math_need *n, tr_math_refusal *r) {
    tr_math_budget dflt;
    return tr_budget_check(budget_or(b, &dflt), n, r);
}
static uint64_t mag_of(const tr_scope_view *v, int32_t c) {      /* max(|min|, |max|, 1) of a scope column */
    const tr_scope_view_column *k = &v->col[c];
    uint64_t m = k->nvalid ? (tr_abs_u64(k->min) > tr_abs_u64(k->max) ? tr_abs_u64(k->min) : tr_abs_u64(k->max)) : 0;
    return m < 1 ? 1 : m;
}
static int view_ok(const tr_scope_view *v) {
    return v && v->ngroups >= 1 && (!v->rows || v->selection) && v->ncols >= 0 && v->ncols <= TR_SCOPE_MAX_COLUMNS;
}
static int column_ok(const tr_scope_view *v, int32_t c) {
    return c >= 0 && c < v->ncols && (!v->rows || v->col[c].codes || v->col[c].values);
}

/* ------------------------------------------------------------------ exact big-number steps with a sticky error */
typedef struct { const tr_math_budget *b; tr_math_refusal *r; int32_t rc; } ictx;
static void ifail(ictx *x, int32_t rc) { if (!x->rc) x->rc = rc ? rc : TR_MATH_MEMORY; }
#define LIVE(x) (!(x)->rc)
static tr_bigint *zk(ictx *x, tr_bigint *v, int32_t e) { if (!v) ifail(x, e); return v; }
static tr_bigint *Z_i64(ictx *x, int64_t v) { int32_t e = 0; return x->rc ? NULL : zk(x, tr_bigint_from_i64(v, x->b, x->r, &e), e); }
static tr_bigint *Z_i128(ictx *x, __int128 v) {
    int32_t e = 0;
    return x->rc ? NULL : zk(x, tr_bigint_from_i128(tr_i128_set(v), x->b, x->r, &e), e);
}
static tr_bigint *Z_add(ictx *x, const tr_bigint *a, const tr_bigint *c) { int32_t e = 0; return x->rc ? NULL : zk(x, tr_bigint_add(a, c, x->b, x->r, &e), e); }
static tr_bigint *Z_mul(ictx *x, const tr_bigint *a, const tr_bigint *c) { int32_t e = 0; return x->rc ? NULL : zk(x, tr_bigint_mul(a, c, x->b, x->r, &e), e); }
static tr_bigint *Z_gcd(ictx *x, const tr_bigint *a, const tr_bigint *c) { int32_t e = 0; return x->rc ? NULL : zk(x, tr_bigint_gcd(a, c, x->b, x->r, &e), e); }
static tr_bigint *Z_div(ictx *x, const tr_bigint *a, const tr_bigint *c) { int32_t e = 0; return x->rc ? NULL : zk(x, tr_bigint_divexact(a, c, x->b, x->r, &e), e); }
static void Z_set(tr_bigint **slot, tr_bigint *v) { if (v) { tr_bigint_free(*slot); *slot = v; } }
static void Z_free_n(tr_bigint **v, uint64_t n) { if (!v) return; for (uint64_t i = 0; i < n; i++) tr_bigint_free(v[i]); free(v); }
static void Q_free_n(tr_bigq **v, uint64_t n) { if (!v) return; for (uint64_t i = 0; i < n; i++) tr_bigq_free(v[i]); free(v); }

/* ================================================================ plans */
struct tr_ident_plan_s {
    uint32_t magic;
    int32_t k, nb, nm, degree, weight, repr, nprimes, proven, width;
    int32_t var[TR_IDENT_MAX_VARS];
    uint8_t *bexp, *mexp;            /* nb x k, nm x k */
    int32_t *bparent, *bvar, *mleft, *mright, *map;   /* map: nb x nb -> moment */
    tr_gfp primes[TR_IDENT_MAX_PRIMES];
    uint64_t rows, selected, digest[2];               /* the scope the plan was proven on */
    uint64_t bytes;
};
static int plan_valid(const tr_ident_plan *p) { return p && p->magic == PLAN_MAGIC; }
void tr_ident_plan_free(tr_ident_plan *p) {
    if (!plan_valid(p)) return;
    free(p->bexp); free(p->mexp); free(p->bparent); free(p->bvar); free(p->mleft); free(p->mright); free(p->map);
    p->magic = 0;
    free(p);
}
static uint64_t plan_bytes(int32_t k, int32_t nb, int32_t nm) {
    uint64_t b = sizeof(tr_ident_plan);
    b = tr_budget_add(b, tr_budget_mul((uint64_t)nb, (uint64_t)k + 8));
    b = tr_budget_add(b, tr_budget_mul((uint64_t)nm, (uint64_t)k + 8));
    return tr_budget_add(b, tr_budget_mul(tr_budget_mul((uint64_t)nb, (uint64_t)nb), 4));
}
typedef struct { uint8_t e[TR_IDENT_MAX_VARS]; int32_t i, j; } pair_ref;
static int pair_cmp(const void *x, const void *y) {
    const pair_ref *a = (const pair_ref *)x, *b = (const pair_ref *)y;
    int c = tr_ident_cmp(a->e, b->e, TR_IDENT_MAX_VARS);
    if (c) return c;
    return a->i != b->i ? (a->i < b->i ? -1 : 1) : (a->j < b->j ? -1 : a->j > b->j);
}
typedef struct { uint8_t e[TR_IDENT_MAX_VARS]; } mono;
static int mono_cmp(const void *x, const void *y) {
    return tr_ident_cmp(((const mono *)x)->e, ((const mono *)y)->e, TR_IDENT_MAX_VARS);
}
/* the index of e among sorted monomials (n), or -1 */
static int32_t mono_find(const mono *list, int32_t n, const uint8_t *e) {
    int32_t lo = 0, hi = n;
    while (lo < hi) {
        int32_t mid = lo + (hi - lo) / 2;
        int c = tr_ident_cmp(list[mid].e, e, TR_IDENT_MAX_VARS);
        if (!c) return mid;
        if (c < 0) lo = mid + 1; else hi = mid;
    }
    return -1;
}
/* every monomial in k variables of total degree <= d, increasing graded lexicographic order */
static int32_t full_basis(int32_t k, int32_t d, mono *out, int32_t cap) {
    int32_t n = 0;
    for (int32_t t = 0; t <= d; t++) {
        /* exponent vectors of total t in increasing lexicographic order: odometer from (0,..,0,t) */
        uint8_t e[TR_IDENT_MAX_VARS] = {0};
        e[k - 1] = (uint8_t)t;
        for (;;) {
            if (n >= cap) return -1;
            memset(&out[n], 0, sizeof out[n]);
            memcpy(out[n].e, e, (size_t)k);
            n++;
            /* next in increasing lex order with the same total: find the rightmost position i < k-1 that can grow */
            int32_t i = k - 2;
            while (i >= 0) {
                int32_t rest = 0;
                for (int32_t j = i + 1; j < k; j++) rest += e[j];
                if (rest > 0) break;
                i--;
            }
            if (i < 0) break;
            int32_t rest = 0;
            for (int32_t j = i + 1; j < k; j++) rest += e[j];
            e[i]++;
            for (int32_t j = i + 1; j < k; j++) e[j] = 0;
            e[k - 1] = (uint8_t)(rest - 1);
        }
    }
    return n;
}

tr_ident_plan *tr_ident_plan_make(const tr_scope *scope, const tr_ident_spec *s, const tr_math_budget *budget,
                                  tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    tr_scope_view v;
    tr_scope_info si;
    if (!scope || !s || tr_scope_get_view(scope, &v) || tr_scope_get_info(scope, &si) || s->reserved ||
        s->nvars < 1 || (s->nterms && !s->exponents) || s->nterms < 0 || s->degree < 0 || s->nprimes < 0 ||
        (s->width != TR_IDENT_WIDTH_AUTO && s->width != TR_IDENT_WIDTH_EXACT && s->width != TR_IDENT_WIDTH_MODULAR)) {
        set_error(error, TR_MATH_INVALID); return NULL;
    }
    if (s->nvars > TR_IDENT_MAX_VARS || s->nprimes > TR_IDENT_MAX_PRIMES || (!s->nterms && s->degree > TR_IDENT_MAX_DEGREE)) {
        set_error(error, TR_MATH_UNSUPPORTED); return NULL;
    }
    const int32_t k = s->nvars;
    for (int32_t i = 0; i < k; i++) {
        if (s->var[i] < 0 || s->var[i] >= v.ncols || s->var[i] == s->weight) { set_error(error, TR_MATH_INVALID); return NULL; }
        for (int32_t j = 0; j < i; j++) if (s->var[j] == s->var[i]) { set_error(error, TR_MATH_INVALID); return NULL; }
    }
    for (int32_t i = k; i < TR_IDENT_MAX_VARS; i++) if (s->var[i] != 0 && s->var[i] != -1) { set_error(error, TR_MATH_INVALID); return NULL; }
    if (s->weight < -1 || s->weight >= v.ncols) { set_error(error, TR_MATH_INVALID); return NULL; }
    /* stage 1: the basis size against the budget, then the representation limit */
    uint64_t nb64 = s->nterms ? (uint64_t)s->nterms : tr_budget_monomials((uint32_t)k, (uint32_t)s->degree);
    tr_math_need need;
    need_zero(&need);
    need.need[TR_BUDGET_VARIABLES] = (uint64_t)k;
    need.need[TR_BUDGET_SUPPORT] = nb64;
    need.need[TR_BUDGET_TENSOR] = tr_budget_mul(nb64, nb64);
    int32_t rc = check_need(budget, &need, refusal);
    if (rc) { set_error(error, rc); return NULL; }
    if (nb64 > TR_IDENT_MAX_BASIS) { set_error(error, TR_MATH_UNSUPPORTED); return NULL; }
    const int32_t nb = (int32_t)nb64;
    mono basis[TR_IDENT_MAX_BASIS];
    memset(basis, 0, sizeof basis);
    int32_t degree = 0;
    if (s->nterms) {
        for (int32_t b = 0; b < nb; b++) {
            int32_t d = 0;
            for (int32_t i = 0; i < k; i++) { basis[b].e[i] = s->exponents[(uint64_t)b * k + i]; d += basis[b].e[i]; }
            if (d > TR_IDENT_MAX_DEGREE) { set_error(error, TR_MATH_UNSUPPORTED); return NULL; }
            if (d > degree) degree = d;
        }
        qsort(basis, (size_t)nb, sizeof basis[0], mono_cmp);
        for (int32_t b = 1; b < nb; b++) if (!mono_cmp(&basis[b - 1], &basis[b])) { set_error(error, TR_MATH_INVALID); return NULL; }
    } else {
        if (full_basis(k, s->degree, basis, nb) != nb) { set_error(error, TR_MATH_INVALID); return NULL; }
        degree = s->degree;
    }
    /* the moments: distinct products m_i m_j, i <= j */
    const int32_t npairs = nb * (nb + 1) / 2;
    pair_ref *pairs = (pair_ref *)zalloc((uint64_t)npairs * sizeof *pairs);
    if (!pairs) { set_error(error, TR_MATH_MEMORY); return NULL; }
    int32_t np = 0;
    for (int32_t i = 0; i < nb; i++)
        for (int32_t j = i; j < nb; j++) {
            for (int32_t t = 0; t < k; t++) pairs[np].e[t] = (uint8_t)(basis[i].e[t] + basis[j].e[t]);
            pairs[np].i = i; pairs[np].j = j; np++;
        }
    qsort(pairs, (size_t)np, sizeof *pairs, pair_cmp);
    int32_t nm = 0;
    for (int32_t q = 0; q < np; q++) if (!q || tr_ident_cmp(pairs[q - 1].e, pairs[q].e, TR_IDENT_MAX_VARS)) nm++;
    /* the representation: the bound of every moment (the scope's extremes) */
    uint64_t N = v.selected;
    for (int32_t i = 0; i < k; i++) if (v.col[s->var[i]].nvalid < N) N = v.col[s->var[i]].nvalid;
    uint64_t W = 1;
    if (s->weight >= 0) {
        if (v.col[s->weight].nvalid < N) N = v.col[s->weight].nvalid;
        W = mag_of(&v, s->weight);
    }
    int proven = 1;
    for (int32_t q = 0; q < np && proven; q++) {
        if (q && !tr_ident_cmp(pairs[q - 1].e, pairs[q].e, TR_IDENT_MAX_VARS)) continue;
        unsigned __int128 b = tr_bank_satmul((unsigned __int128)N, W, I128_LIMIT);
        for (int32_t t = 0; t < k && b; t++)
            for (uint8_t e = 0; e < pairs[q].e[t]; e++) b = tr_bank_satmul(b, mag_of(&v, s->var[t]), I128_LIMIT);
        if (b > I128_LIMIT) proven = 0;
    }
    int32_t repr = s->width == TR_IDENT_WIDTH_MODULAR ? TR_IDENT_MODULAR : proven ? TR_IDENT_EXACT : TR_IDENT_MODULAR;
    if (s->width == TR_IDENT_WIDTH_EXACT && !proven) { free(pairs); set_error(error, TR_MATH_OVERFLOW); return NULL; }
    int32_t nprimes = repr == TR_IDENT_MODULAR ? (s->nprimes ? s->nprimes : TR_IDENT_DEFAULT_PRIMES) : 0;
    /* stage 2: degree, states, iterations, bytes */
    need.need[TR_BUDGET_DEGREE] = (uint64_t)degree;
    need.need[TR_BUDGET_STATES] = tr_budget_mul((uint64_t)nm, nprimes ? (uint64_t)nprimes : 1);
    need.need[TR_BUDGET_ITERATIONS] = (uint64_t)nprimes;
    need.need[TR_BUDGET_BYTES] = plan_bytes(k, nb, nm);
    rc = check_need(budget, &need, refusal);
    if (rc) { free(pairs); set_error(error, rc); return NULL; }
    tr_ident_plan *p = (tr_ident_plan *)zalloc(sizeof *p);
    if (p) {
        p->bexp = (uint8_t *)zalloc((uint64_t)nb * k); p->mexp = (uint8_t *)zalloc((uint64_t)nm * k);
        p->bparent = (int32_t *)zalloc((uint64_t)nb * 4); p->bvar = (int32_t *)zalloc((uint64_t)nb * 4);
        p->mleft = (int32_t *)zalloc((uint64_t)nm * 4); p->mright = (int32_t *)zalloc((uint64_t)nm * 4);
        p->map = (int32_t *)zalloc((uint64_t)nb * nb * 4);
        p->magic = PLAN_MAGIC;
    }
    if (!p || !p->bexp || !p->mexp || !p->bparent || !p->bvar || !p->mleft || !p->mright || !p->map) {
        tr_ident_plan_free(p); free(pairs); set_error(error, TR_MATH_MEMORY); return NULL;
    }
    p->k = k; p->nb = nb; p->nm = nm; p->degree = degree; p->weight = s->weight; p->repr = repr;
    p->nprimes = nprimes; p->proven = proven; p->width = s->width;
    for (int32_t i = 0; i < TR_IDENT_MAX_VARS; i++) p->var[i] = i < k ? s->var[i] : -1;
    for (int32_t b = 0; b < nb; b++) {
        memcpy(p->bexp + (uint64_t)b * k, basis[b].e, (size_t)k);
        p->bparent[b] = -1; p->bvar[b] = 0;
        for (int32_t i = 0; i < k && p->bparent[b] < 0; i++) {     /* m_b = m_parent * x_i with the parent in the basis */
            if (!basis[b].e[i]) continue;
            mono q = basis[b];
            q.e[i]--;
            int32_t at = mono_find(basis, nb, q.e);
            if (at >= 0) { p->bparent[b] = at; p->bvar[b] = i; }
        }
    }
    int32_t m = -1;
    for (int32_t q = 0; q < np; q++) {
        if (!q || tr_ident_cmp(pairs[q - 1].e, pairs[q].e, TR_IDENT_MAX_VARS)) {
            m++;
            memcpy(p->mexp + (uint64_t)m * k, pairs[q].e, (size_t)k);
            p->mleft[m] = pairs[q].i; p->mright[m] = pairs[q].j;
        }
        p->map[(uint64_t)pairs[q].i * nb + pairs[q].j] = m;
        p->map[(uint64_t)pairs[q].j * nb + pairs[q].i] = m;
    }
    free(pairs);
    if (nprimes) {
        rc = tr_lin_primes((uint64_t)nprimes, p->primes);
        if (rc) { tr_ident_plan_free(p); set_error(error, rc); return NULL; }
    }
    p->rows = si.rows; p->selected = si.selected; p->digest[0] = si.digest[0]; p->digest[1] = si.digest[1];
    p->bytes = plan_bytes(k, nb, nm);
    set_error(error, TR_MATH_OK);
    return p;
}
static tr_ident_plan *plan_clone(const tr_ident_plan *s, int32_t *error) {
    tr_ident_plan *p = (tr_ident_plan *)zalloc(sizeof *p);
    if (!p) { set_error(error, TR_MATH_MEMORY); return NULL; }
    *p = *s;
    p->bexp = (uint8_t *)zalloc((uint64_t)s->nb * s->k); p->mexp = (uint8_t *)zalloc((uint64_t)s->nm * s->k);
    p->bparent = (int32_t *)zalloc((uint64_t)s->nb * 4); p->bvar = (int32_t *)zalloc((uint64_t)s->nb * 4);
    p->mleft = (int32_t *)zalloc((uint64_t)s->nm * 4); p->mright = (int32_t *)zalloc((uint64_t)s->nm * 4);
    p->map = (int32_t *)zalloc((uint64_t)s->nb * s->nb * 4);
    if (!p->bexp || !p->mexp || !p->bparent || !p->bvar || !p->mleft || !p->mright || !p->map) {
        tr_ident_plan_free(p); set_error(error, TR_MATH_MEMORY); return NULL;
    }
    memcpy(p->bexp, s->bexp, (size_t)s->nb * s->k); memcpy(p->mexp, s->mexp, (size_t)s->nm * s->k);
    memcpy(p->bparent, s->bparent, (size_t)s->nb * 4); memcpy(p->bvar, s->bvar, (size_t)s->nb * 4);
    memcpy(p->mleft, s->mleft, (size_t)s->nm * 4); memcpy(p->mright, s->mright, (size_t)s->nm * 4);
    memcpy(p->map, s->map, (size_t)s->nb * s->nb * 4);
    return p;
}
static int plan_same(const tr_ident_plan *a, const tr_ident_plan *b) {
    if (a->k != b->k || a->nb != b->nb || a->nm != b->nm || a->weight != b->weight || a->repr != b->repr ||
        a->nprimes != b->nprimes || memcmp(a->var, b->var, sizeof a->var) || memcmp(a->bexp, b->bexp, (size_t)a->nb * a->k))
        return 0;
    for (int32_t i = 0; i < a->nprimes; i++) if (a->primes[i].p != b->primes[i].p) return 0;
    return 1;
}
int32_t tr_ident_plan_get_info(const tr_ident_plan *p, tr_ident_plan_info *out) {
    if (!plan_valid(p) || !out) return TR_MATH_INVALID;
    memset(out, 0, sizeof *out);
    out->nvars = p->k; out->nbasis = p->nb; out->nmoments = p->nm; out->degree = p->degree;
    out->repr = p->repr; out->nprimes = p->nprimes; out->proven = p->proven; out->weight = p->weight;
    for (int32_t i = 0; i < p->nprimes; i++) out->primes[i] = p->primes[i].p;
    out->bytes = p->bytes;
    return TR_MATH_OK;
}
int32_t tr_ident_plan_get_view(const tr_ident_plan *p, tr_ident_moment_view *v) {
    if (!plan_valid(p) || !v) return TR_MATH_INVALID;
    memset(v, 0, sizeof *v);
    v->nvars = p->k; v->nbasis = p->nb; v->nmoments = p->nm; v->weight = p->weight;
    for (int32_t i = 0; i < TR_IDENT_MAX_VARS; i++) v->var[i] = p->var[i];
    v->repr = p->repr; v->nprimes = p->nprimes;
    v->basis = p->bexp; v->bparent = p->bparent; v->bvar = p->bvar; v->mleft = p->mleft; v->mright = p->mright;
    v->primes = p->primes;
    return TR_MATH_OK;
}
int32_t tr_ident_plan_basis(const tr_ident_plan *p, uint8_t *out, uint64_t capacity) {
    if (!plan_valid(p) || !out || capacity < (uint64_t)p->nb * p->k) return TR_MATH_INVALID;
    memcpy(out, p->bexp, (size_t)p->nb * p->k);
    return TR_MATH_OK;
}
int32_t tr_ident_plan_moments(const tr_ident_plan *p, uint8_t *out, uint64_t capacity) {
    if (!plan_valid(p) || !out || capacity < (uint64_t)p->nm * p->k) return TR_MATH_INVALID;
    memcpy(out, p->mexp, (size_t)p->nm * p->k);
    return TR_MATH_OK;
}
#define GRAM_HEADER_BYTES 192        /* sizeof(struct tr_ident_gram_s), rounded up */
static uint64_t gram_bytes(const tr_ident_plan *p) {
    uint64_t slots = p->repr == TR_IDENT_MODULAR ? (uint64_t)p->nprimes : 1;
    return tr_budget_add(tr_budget_add(GRAM_HEADER_BYTES, p->bytes), tr_budget_mul(tr_budget_mul((uint64_t)p->nm, slots), 16));
}
int32_t tr_ident_plan_need(const tr_ident_plan *p, const tr_scope *scope, int32_t target, tr_math_need *need) {
    tr_scope_view v;
    if (!plan_valid(p) || !need || !scope || tr_scope_get_view(scope, &v)) return TR_MATH_INVALID;
    if (target == TR_IDENT_NEED_DEVICE) {
        tr_ident_moment_view mv;
        tr_ident_plan_get_view(p, &mv);
        tr_ident_gram_device_need(&v, &mv, need);
        return TR_MATH_OK;
    }
    if (target != TR_IDENT_NEED_CPU && target != TR_IDENT_NEED_ADOPT) return TR_MATH_INVALID;
    need_zero(need);
    uint64_t slots = p->repr == TR_IDENT_MODULAR ? (uint64_t)p->nprimes : 1;
    need->need[TR_BUDGET_VARIABLES] = (uint64_t)p->k;
    need->need[TR_BUDGET_DEGREE] = (uint64_t)p->degree;
    need->need[TR_BUDGET_SUPPORT] = (uint64_t)p->nb;
    need->need[TR_BUDGET_TENSOR] = tr_budget_mul((uint64_t)p->nb, (uint64_t)p->nb);
    need->need[TR_BUDGET_STATES] = tr_budget_mul((uint64_t)p->nm, slots);
    need->need[TR_BUDGET_ITERATIONS] = p->repr == TR_IDENT_MODULAR ? slots : 0;
    uint64_t bytes = gram_bytes(p);
    if (target == TR_IDENT_NEED_CPU)                  /* the accumulators and the basis values of one row */
        bytes = tr_budget_add(bytes, tr_budget_add(tr_budget_mul(tr_budget_mul((uint64_t)p->nm, slots), 16),
                                                   tr_budget_mul((uint64_t)p->nb, 16)));
    need->need[TR_BUDGET_BYTES] = bytes;
    return TR_MATH_OK;
}

/* ================================================================ Grams */
struct tr_ident_gram_s {
    uint32_t magic;
    int32_t sign, source, nslots;
    tr_ident_plan *plan;             /* owned copy */
    uint64_t rows, selected, domain, positive, zero, negative;
    int64_t wmin, wmax;
    uint64_t digest[2];
    __int128 *exact;                 /* nm (EXACT) */
    uint64_t *res;                   /* nm x nprimes (MODULAR) */
    uint64_t bytes;
};
_Static_assert(sizeof(struct tr_ident_gram_s) <= GRAM_HEADER_BYTES, "gram header bytes");
static int gram_valid(const tr_ident_gram *g) { return g && g->magic == GRAM_MAGIC; }
void tr_ident_gram_free(tr_ident_gram *g) {
    if (!gram_valid(g)) return;
    tr_ident_plan_free(g->plan);
    free(g->exact); free(g->res);
    g->magic = 0;
    free(g);
}
static tr_ident_gram *gram_alloc(const tr_ident_plan *p, int32_t *error) {
    tr_ident_gram *g = (tr_ident_gram *)zalloc(sizeof *g);
    if (!g) { set_error(error, TR_MATH_MEMORY); return NULL; }
    g->magic = GRAM_MAGIC;
    g->plan = plan_clone(p, error);
    g->nslots = p->repr == TR_IDENT_MODULAR ? p->nprimes : 1;
    if (g->plan) {
        if (p->repr == TR_IDENT_EXACT) g->exact = (__int128 *)zalloc((uint64_t)p->nm * sizeof(__int128));
        else g->res = (uint64_t *)zalloc((uint64_t)p->nm * (uint64_t)p->nprimes * 8);
    }
    if (!g->plan || (p->repr == TR_IDENT_EXACT ? !g->exact : !g->res)) {
        tr_ident_gram_free(g); set_error(error, TR_MATH_MEMORY); return NULL;
    }
    g->sign = TR_IDENT_NONNEGATIVE;
    g->wmin = 1; g->wmax = 1;
    g->bytes = gram_bytes(p);
    return g;
}
static int32_t gram_scope_check(const tr_scope *scope, const tr_ident_plan *p, tr_scope_view *v, tr_scope_info *si) {
    if (!scope || !plan_valid(p) || tr_scope_get_view(scope, v) || tr_scope_get_info(scope, si)) return TR_MATH_INVALID;
    if (si->digest[0] != p->digest[0] || si->digest[1] != p->digest[1] || si->rows != p->rows || si->selected != p->selected)
        return TR_MATH_DOMAIN_ERROR;                           /* the plan's proof is about another scope */
    for (int32_t i = 0; i < p->k; i++) if (!column_ok(v, p->var[i])) return TR_MATH_INVALID;
    if (p->weight >= 0 && !column_ok(v, p->weight)) return TR_MATH_INVALID;
    return TR_MATH_OK;
}
/* the Gram from states (shared by the CPU pass and the adoption of a readback): consistency checks, then the moments */
static tr_ident_gram *gram_from_acc(const tr_scope_view *v, const tr_scope_info *si, const tr_ident_plan *p,
                                    const tr_ident_gram_states *st, int32_t source, int32_t *error) {
    const int32_t slots = p->repr == TR_IDENT_MODULAR ? p->nprimes : 1;
    if (!st || st->nmoments != p->nm || st->nslots != slots || (p->nm && !st->acc) ||
        st->domain != st->positive + st->zero + st->negative || st->domain > v->selected ||
        st->positive > st->domain || (st->domain && st->wmin > st->wmax) ||
        (p->weight < 0 && (st->positive != st->domain || st->wmin != 1 || st->wmax != 1))) {
        set_error(error, TR_MATH_INVALID); return NULL;
    }
    tr_ident_gram *g = gram_alloc(p, error);
    if (!g) return NULL;
    for (int32_t m = 0; m < p->nm; m++) {
        if (p->repr == TR_IDENT_EXACT) {
            __int128 x = (__int128)(((unsigned __int128)(uint64_t)st->acc[2 * m + 1] << 64) | (uint64_t)st->acc[2 * m]);
            /* the proven bound of the moment */
            unsigned __int128 b = tr_bank_satmul((unsigned __int128)st->domain,
                                                 p->weight >= 0 ? (tr_abs_u64(st->wmin) > tr_abs_u64(st->wmax) ? tr_abs_u64(st->wmin) : tr_abs_u64(st->wmax)) : 1,
                                                 I128_LIMIT);
            for (int32_t t = 0; t < p->k && b; t++)
                for (uint8_t e = 0; e < p->mexp[(uint64_t)m * p->k + t]; e++) b = tr_bank_satmul(b, mag_of(v, p->var[t]), I128_LIMIT);
            if (tr_cert_mag(x) > b) { tr_ident_gram_free(g); set_error(error, TR_MATH_INVALID); return NULL; }
            g->exact[m] = x;
        } else {
            for (int32_t q = 0; q < p->nprimes; q++) {
                const int64_t *w = st->acc + 2 * ((uint64_t)m * slots + q);
                if (w[1] < 0) { tr_ident_gram_free(g); set_error(error, TR_MATH_INVALID); return NULL; }
                g->res[(uint64_t)m * p->nprimes + q] = tr_gfp_reduce_u128_hd(&p->primes[q], (uint64_t)w[1], (uint64_t)w[0]);
            }
        }
    }
    g->source = source;
    g->rows = si->rows; g->selected = si->selected; g->domain = st->domain;
    g->positive = st->positive; g->zero = st->zero; g->negative = st->negative;
    g->wmin = st->domain ? st->wmin : (p->weight >= 0 ? 0 : 1);
    g->wmax = st->domain ? st->wmax : (p->weight >= 0 ? 0 : 1);
    g->sign = st->negative ? TR_IDENT_SIGNED : TR_IDENT_NONNEGATIVE;
    g->digest[0] = si->digest[0]; g->digest[1] = si->digest[1];
    set_error(error, TR_MATH_OK);
    return g;
}
tr_ident_gram *tr_ident_gram_build(const tr_scope *scope, const tr_ident_plan *p, const tr_math_budget *budget,
                                   tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    tr_scope_view v;
    tr_scope_info si;
    int32_t rc = gram_scope_check(scope, p, &v, &si);
    if (rc) { set_error(error, rc); return NULL; }
    tr_math_need need;
    tr_ident_plan_need(p, scope, TR_IDENT_NEED_CPU, &need);
    rc = check_need(budget, &need, refusal);
    if (rc) { set_error(error, rc); return NULL; }
    tr_ident_moment_view mv;
    tr_ident_plan_get_view(p, &mv);
    const int32_t slots = p->repr == TR_IDENT_MODULAR ? p->nprimes : 1;
    int64_t *acc = (int64_t *)zalloc((uint64_t)p->nm * slots * 16);
    __int128 *val = (__int128 *)zalloc((uint64_t)p->nb * sizeof(__int128));
    __int128 *ex = (__int128 *)zalloc((uint64_t)p->nm * sizeof(__int128));   /* EXACT: one signed 128-bit sum per moment */
    uint64_t *mval = (uint64_t *)zalloc((uint64_t)p->nb * 8), *racc = (uint64_t *)zalloc((uint64_t)p->nm * slots * 8);
    if (!acc || !val || !ex || !mval || !racc) {
        free(acc); free(val); free(ex); free(mval); free(racc); set_error(error, TR_MATH_MEMORY); return NULL;
    }
    tr_ident_gram_states st;
    memset(&st, 0, sizeof st);
    st.nmoments = p->nm; st.nslots = slots; st.wmin = INT64_MAX; st.wmax = INT64_MIN; st.acc = acc;
    const uint64_t nw = (v.rows + 63) / 64;
    for (uint64_t w = 0; w < nw; w++) {
        uint64_t bits = v.selection[w];
        while (bits) {
            uint64_t r = w * 64 + (uint64_t)__builtin_ctzll(bits);
            bits &= bits - 1;
            int64_t wt;
            if (!tr_ident_gram_row(&v, &mv, r, &wt)) continue;
            st.domain++;
            if (wt > 0) st.positive++; else if (wt < 0) st.negative++; else st.zero++;
            if (wt < st.wmin) st.wmin = wt;
            if (wt > st.wmax) st.wmax = wt;
            if (!wt) continue;
            if (p->repr == TR_IDENT_EXACT) {
                tr_ident_basis_exact(&v, &mv, r, val);
                for (int32_t m = 0; m < p->nm; m++) ex[m] += (__int128)wt * (val[p->mleft[m]] * val[p->mright[m]]);
            } else {
                for (int32_t q = 0; q < p->nprimes; q++) {
                    const tr_gfp *f = &p->primes[q];
                    tr_ident_basis_mod(&v, &mv, f, r, mval);
                    uint64_t wm = tr_gfp_reduce_i64_hd(f, wt);
                    uint64_t *ra = racc + (uint64_t)q;
                    for (int32_t m = 0; m < p->nm; m++) {
                        uint64_t t = tr_gfp_mul_hd(f, wm, tr_gfp_mul_hd(f, mval[p->mleft[m]], mval[p->mright[m]]));
                        ra[(uint64_t)m * slots] = tr_gfp_add_hd(f, ra[(uint64_t)m * slots], t);
                    }
                }
            }
        }
    }
    if (!st.domain) { st.wmin = p->weight >= 0 ? 0 : 1; st.wmax = st.wmin; }
    if (p->weight < 0 && st.domain) { st.wmin = 1; st.wmax = 1; }
    if (p->repr == TR_IDENT_MODULAR)                         /* residues (already reduced) as the states' sums */
        for (uint64_t i = 0; i < (uint64_t)p->nm * slots; i++) { acc[2 * i] = (int64_t)racc[i]; acc[2 * i + 1] = 0; }
    else
        for (int32_t m = 0; m < p->nm; m++) { __int128 x = ex[m]; acc[2 * m] = (int64_t)(uint64_t)x; acc[2 * m + 1] = (int64_t)(x >> 64); }
    tr_ident_gram *g = gram_from_acc(&v, &si, p, &st, TR_IDENT_SOURCE_FUSED, error);
    free(acc); free(val); free(ex); free(mval); free(racc);
    return g;
}
tr_ident_gram *tr_ident_gram_from_states(const tr_scope *scope, const tr_ident_plan *p, const tr_ident_gram_states *st,
                                         const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    tr_scope_view v;
    tr_scope_info si;
    int32_t rc = gram_scope_check(scope, p, &v, &si);
    if (rc) { set_error(error, rc); return NULL; }
    tr_math_need need;
    tr_ident_plan_need(p, scope, TR_IDENT_NEED_ADOPT, &need);
    rc = check_need(budget, &need, refusal);
    if (rc) { set_error(error, rc); return NULL; }
    return gram_from_acc(&v, &si, p, st, TR_IDENT_SOURCE_ADOPTED, error);
}
/* the bank term of a moment (exponents over the bank's columns), or -1 */
static int32_t bank_term(const uint8_t *bexps, int32_t nterms, int32_t ncols, const tr_ident_plan *p, int32_t m) {
    for (int32_t j = 0; j < nterms; j++) {
        const uint8_t *e = bexps + (uint64_t)j * ncols;
        int ok = 1;
        for (int32_t c = 0; c < ncols && ok; c++) {
            uint8_t want = 0;
            for (int32_t t = 0; t < p->k; t++) if (p->var[t] == c) want = p->mexp[(uint64_t)m * p->k + t];
            ok = e[c] == want;
        }
        if (ok) return j;
    }
    return -1;
}
tr_ident_gram *tr_ident_gram_from_bank(const tr_scope *scope, const tr_ident_plan *p, const tr_bank *bank,
                                       const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    tr_scope_view v;
    tr_scope_info si;
    int32_t rc = gram_scope_check(scope, p, &v, &si);
    if (rc) { set_error(error, rc); return NULL; }
    tr_bank_info bi;
    tr_bank_states bs;
    if (!bank || tr_bank_get_info(bank, &bi) || tr_bank_get_states(bank, &bs)) { set_error(error, TR_MATH_INVALID); return NULL; }
    if (p->weight >= 0) { set_error(error, TR_MATH_UNSUPPORTED); return NULL; }
    if (p->repr != TR_IDENT_EXACT) { set_error(error, TR_MATH_OVERFLOW); return NULL; }
    uint32_t want = 0;
    for (int32_t t = 0; t < p->k; t++) want |= UINT32_C(1) << p->var[t];
    if (!bi.scoped || bi.digest[0] != si.digest[0] || bi.digest[1] != si.digest[1] || bi.mode != TR_BANK_JOINT ||
        bi.joint != want || bi.ncols != v.ncols) { set_error(error, TR_MATH_DOMAIN_ERROR); return NULL; }
    tr_math_need need;
    tr_ident_plan_need(p, scope, TR_IDENT_NEED_ADOPT, &need);
    need.need[TR_BUDGET_BYTES] = tr_budget_add(need.need[TR_BUDGET_BYTES], tr_budget_mul((uint64_t)bi.nterms, (uint64_t)bi.ncols));
    rc = check_need(budget, &need, refusal);
    if (rc) { set_error(error, rc); return NULL; }
    uint8_t *bexps = (uint8_t *)zalloc((uint64_t)bi.nterms * bi.ncols);
    int64_t *acc = (int64_t *)zalloc((uint64_t)p->nm * 16);
    if (!bexps || !acc) { free(bexps); free(acc); set_error(error, TR_MATH_MEMORY); return NULL; }
    tr_bank_get_exponents(bank, bexps, (uint64_t)bi.nterms * bi.ncols);
    tr_ident_gram_states st;
    memset(&st, 0, sizeof st);
    st.nmoments = p->nm; st.nslots = 1; st.acc = acc; st.wmin = 1; st.wmax = 1;
    rc = TR_MATH_OK;
    uint64_t domain = 0;
    for (int32_t m = 0; m < p->nm && !rc; m++) {
        int32_t j = bank_term(bexps, bi.nterms, bi.ncols, p, m);
        if (j < 0) { rc = TR_MATH_INVALID; break; }           /* the bank lacks a moment */
        __int128 s = 0;
        uint64_t cnt = 0;
        for (uint64_t g = 0; g < bs.ngroups; g++) {
            s += tr_i128_get(bs.sum[g * (uint64_t)bs.nterms + j]);   /* proven: the pooled sum fits */
            cnt += bs.count[g * (uint64_t)bs.nterms + j];
        }
        if (m && cnt != domain) rc = TR_MATH_INVALID;          /* JOINT: every term counts the complete rows */
        domain = cnt;
        acc[2 * m] = (int64_t)(uint64_t)s; acc[2 * m + 1] = (int64_t)(s >> 64);
    }
    free(bexps);
    if (rc) { free(acc); set_error(error, rc); return NULL; }
    st.domain = st.positive = domain;
    if (!domain) { st.wmin = 1; st.wmax = 1; }
    tr_ident_gram *g = gram_from_acc(&v, &si, p, &st, TR_IDENT_SOURCE_BANK, error);
    free(acc);
    return g;
}
tr_ident_gram *tr_ident_gram_from_lin(const tr_scope *scope, const tr_ident_plan *p, const tr_lin_moments *lm,
                                      const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    tr_scope_view v;
    tr_scope_info si;
    int32_t rc = gram_scope_check(scope, p, &v, &si);
    if (rc) { set_error(error, rc); return NULL; }
    tr_lin_info li;
    tr_lin_states ls;
    if (!lm || tr_lin_moments_get_info(lm, &li) || tr_lin_moments_get_states(lm, &ls)) { set_error(error, TR_MATH_INVALID); return NULL; }
    if (p->weight >= 0 || p->degree > 1) { set_error(error, TR_MATH_UNSUPPORTED); return NULL; }
    if (p->repr != TR_IDENT_EXACT) { set_error(error, TR_MATH_OVERFLOW); return NULL; }
    if (!li.scoped || li.digest[0] != si.digest[0] || li.digest[1] != si.digest[1] || li.policy != TR_LIN_LISTWISE ||
        li.target != -1 || li.nfeatures != p->k) { set_error(error, TR_MATH_DOMAIN_ERROR); return NULL; }
    int32_t aug[TR_IDENT_MAX_VARS];                           /* the augmented LIN column of each variable */
    for (int32_t t = 0; t < p->k; t++) {
        aug[t] = -1;
        for (int32_t a = 1; a < li.k; a++) if (li.column[a] == p->var[t]) aug[t] = a;
        if (aug[t] < 0) { set_error(error, TR_MATH_DOMAIN_ERROR); return NULL; }
    }
    tr_math_need need;
    tr_ident_plan_need(p, scope, TR_IDENT_NEED_ADOPT, &need);
    rc = check_need(budget, &need, refusal);
    if (rc) { set_error(error, rc); return NULL; }
    int64_t *acc = (int64_t *)zalloc((uint64_t)p->nm * 16);
    if (!acc) { set_error(error, TR_MATH_MEMORY); return NULL; }
    tr_ident_gram_states st;
    memset(&st, 0, sizeof st);
    st.nmoments = p->nm; st.nslots = 1; st.acc = acc; st.wmin = 1; st.wmax = 1;
    uint64_t domain = 0;
    for (uint64_t g = 0; g < ls.ngroups; g++) domain += ls.count[g * (uint64_t)ls.nslots + tr_lin_gram_slot(li.k, 0, 0)];
    for (int32_t m = 0; m < p->nm && !rc; m++) {
        int32_t f[2] = {0, 0}, n = 0;                         /* the moment's two augmented columns (0: the constant) */
        for (int32_t t = 0; t < p->k; t++)
            for (uint8_t e = 0; e < p->mexp[(uint64_t)m * p->k + t]; e++) { if (n < 2) f[n] = aug[t]; n++; }
        if (n > 2) { rc = TR_MATH_INVALID; break; }
        int32_t a = f[0] < f[1] ? f[0] : f[1], b = f[0] < f[1] ? f[1] : f[0];   /* n == 1: (0, column) */
        int32_t slot = tr_lin_gram_slot(li.k, a, b);
        __int128 s = 0;
        for (uint64_t g = 0; g < ls.ngroups; g++) {
            const uint64_t *w = ls.sum + (g * (uint64_t)ls.nslots + slot) * (uint64_t)ls.words;
            uint64_t top = ls.words > 2 ? w[2] : (uint64_t)((int64_t)w[1] >> 63);
            if (top != (uint64_t)((int64_t)w[1] >> 63)) { rc = TR_MATH_OVERFLOW; break; }
            s += (__int128)(((unsigned __int128)w[1] << 64) | w[0]);
        }
        acc[2 * m] = (int64_t)(uint64_t)s; acc[2 * m + 1] = (int64_t)(s >> 64);
    }
    if (rc) { free(acc); set_error(error, rc); return NULL; }
    st.domain = st.positive = domain;
    tr_ident_gram *g = gram_from_acc(&v, &si, p, &st, TR_IDENT_SOURCE_LIN, error);
    free(acc);
    return g;
}
tr_ident_gram *tr_ident_gram_combine(const tr_ident_gram *a, const tr_ident_gram *b, int32_t sign,
                                     const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    if (!gram_valid(a) || !gram_valid(b) || (sign != 1 && sign != -1)) { set_error(error, TR_MATH_INVALID); return NULL; }
    if (!plan_same(a->plan, b->plan)) { set_error(error, TR_MATH_DOMAIN_ERROR); return NULL; }
    int32_t rc = check_one(budget, TR_BUDGET_BYTES, a->bytes, refusal);
    if (rc) { set_error(error, rc); return NULL; }
    if (sign < 0 && (b->domain > a->domain || b->positive > a->positive || b->zero > a->zero || b->negative > a->negative)) {
        set_error(error, TR_MATH_INVALID); return NULL;        /* a deletion of rows a does not hold */
    }
    tr_ident_gram *g = gram_alloc(a->plan, error);
    if (!g) return NULL;
    const tr_ident_plan *p = a->plan;
    for (int32_t m = 0; m < p->nm; m++) {
        if (p->repr == TR_IDENT_EXACT) {
            __int128 r;
            int ov = sign > 0 ? __builtin_add_overflow(a->exact[m], b->exact[m], &r) : __builtin_sub_overflow(a->exact[m], b->exact[m], &r);
            if (ov || tr_cert_mag(r) > I128_LIMIT) { tr_ident_gram_free(g); set_error(error, TR_MATH_OVERFLOW); return NULL; }
            g->exact[m] = r;
        } else
            for (int32_t q = 0; q < p->nprimes; q++) {
                const tr_gfp *f = &p->primes[q];
                uint64_t x = a->res[(uint64_t)m * p->nprimes + q], y = b->res[(uint64_t)m * p->nprimes + q];
                g->res[(uint64_t)m * p->nprimes + q] = sign > 0 ? tr_gfp_add_hd(f, x, y) : tr_gfp_sub_hd(f, x, y);
            }
    }
    g->source = TR_IDENT_SOURCE_COMBINED;
    g->rows = a->rows;
    g->selected = sign > 0 ? a->selected + b->selected : (a->selected >= b->selected ? a->selected - b->selected : 0);
    g->domain = sign > 0 ? a->domain + b->domain : a->domain - b->domain;
    g->positive = sign > 0 ? a->positive + b->positive : a->positive - b->positive;
    g->zero = sign > 0 ? a->zero + b->zero : a->zero - b->zero;
    g->negative = sign > 0 ? a->negative + b->negative : a->negative - b->negative;
    /* the weights involved: a's, and b's (negated for a deletion: the deleted rows weigh -w) */
    int have = 0;
    int64_t lo = 0, hi = 0;
    if (a->domain) { lo = a->wmin; hi = a->wmax; have = 1; }
    if (b->domain) {
        int64_t bl = sign > 0 ? b->wmin : (b->wmax == INT64_MIN ? INT64_MAX : -b->wmax);
        int64_t bh = sign > 0 ? b->wmax : (b->wmin == INT64_MIN ? INT64_MAX : -b->wmin);
        if (!have || bl < lo) lo = bl;
        if (!have || bh > hi) hi = bh;
        have = 1;
    }
    g->wmin = have ? lo : a->wmin; g->wmax = have ? hi : a->wmax;
    g->sign = (sign < 0 || a->sign == TR_IDENT_SIGNED || b->sign == TR_IDENT_SIGNED) ? TR_IDENT_SIGNED : TR_IDENT_NONNEGATIVE;
    set_error(error, TR_MATH_OK);
    return g;
}
int32_t tr_ident_gram_get_info(const tr_ident_gram *g, tr_ident_gram_info *out) {
    if (!gram_valid(g) || !out) return TR_MATH_INVALID;
    memset(out, 0, sizeof *out);
    const tr_ident_plan *p = g->plan;
    out->rows = g->rows; out->selected = g->selected; out->domain = g->domain;
    out->positive = g->positive; out->zero = g->zero; out->negative = g->negative;
    out->wmin = g->wmin; out->wmax = g->wmax; out->bytes = g->bytes;
    out->digest[0] = g->digest[0]; out->digest[1] = g->digest[1];
    out->nvars = p->k; out->nbasis = p->nb; out->nmoments = p->nm; out->degree = p->degree;
    out->repr = p->repr; out->nprimes = p->nprimes; out->sign = g->sign; out->source = g->source;
    out->weight = p->weight;
    for (int32_t i = 0; i < p->nprimes; i++) out->primes[i] = p->primes[i].p;
    return TR_MATH_OK;
}
int32_t tr_ident_gram_get_states(const tr_ident_gram *g, tr_ident_gram_states *st, int64_t *out, uint64_t capacity) {
    if (!gram_valid(g) || !st) return TR_MATH_INVALID;
    const tr_ident_plan *p = g->plan;
    uint64_t words = (uint64_t)p->nm * g->nslots * 2;
    if (capacity < words || (words && !out)) return TR_MATH_INVALID;
    memset(st, 0, sizeof *st);
    st->nmoments = p->nm; st->nslots = g->nslots; st->domain = g->domain; st->positive = g->positive;
    st->zero = g->zero; st->negative = g->negative; st->wmin = g->wmin; st->wmax = g->wmax; st->acc = out;
    for (int32_t m = 0; m < p->nm; m++)
        for (int32_t q = 0; q < g->nslots; q++) {
            uint64_t i = (uint64_t)m * g->nslots + q;
            if (p->repr == TR_IDENT_EXACT) { out[2 * i] = (int64_t)(uint64_t)g->exact[m]; out[2 * i + 1] = (int64_t)(g->exact[m] >> 64); }
            else { out[2 * i] = (int64_t)g->res[i]; out[2 * i + 1] = 0; }
        }
    return TR_MATH_OK;
}
int32_t tr_ident_gram_matrix(const tr_ident_gram *g, int32_t slot, uint64_t *out, uint64_t capacity) {
    if (!gram_valid(g) || !out) return TR_MATH_INVALID;
    const tr_ident_plan *p = g->plan;
    const uint64_t nb = (uint64_t)p->nb;
    if (slot < 0) {
        if (p->repr != TR_IDENT_EXACT) return TR_MATH_DOMAIN_ERROR;
        if (capacity < nb * nb * 2) return TR_MATH_INVALID;
        for (uint64_t i = 0; i < nb * nb; i++) {
            __int128 x = g->exact[p->map[i]];
            out[2 * i] = (uint64_t)x; out[2 * i + 1] = (uint64_t)(x >> 64);
        }
        return TR_MATH_OK;
    }
    if (p->repr != TR_IDENT_MODULAR || slot >= p->nprimes || capacity < nb * nb) return TR_MATH_INVALID;
    for (uint64_t i = 0; i < nb * nb; i++) out[i] = g->res[(uint64_t)p->map[i] * p->nprimes + slot];
    return TR_MATH_OK;
}
int32_t tr_ident_gram_equal(const tr_ident_gram *a, const tr_ident_gram *b) {
    if (!gram_valid(a) || !gram_valid(b)) return TR_MATH_INVALID;
    if (!plan_same(a->plan, b->plan) || a->domain != b->domain || a->positive != b->positive || a->zero != b->zero ||
        a->negative != b->negative || a->wmin != b->wmin || a->wmax != b->wmax || a->sign != b->sign) return 0;
    const tr_ident_plan *p = a->plan;
    if (p->repr == TR_IDENT_EXACT) return !memcmp(a->exact, b->exact, (size_t)p->nm * sizeof(__int128));
    return !memcmp(a->res, b->res, (size_t)p->nm * p->nprimes * 8);
}

/* ================================================================ the nullspace */
struct tr_ident_kernel_s {
    uint32_t magic;
    int32_t nb, rank, nullity, method, exact_rank, primes, agreeing, unresolved;
    uint64_t pivots, free, resolved;
    tr_bigq **vec;                   /* nullity x nb; a NULL row: unresolved */
    uint64_t bytes;
};
static int ker_valid(const tr_ident_kernel *k) { return k && k->magic == KER_MAGIC; }
void tr_ident_kernel_free(tr_ident_kernel *k) {
    if (!ker_valid(k)) return;
    Q_free_n(k->vec, (uint64_t)k->nullity * k->nb);
    k->magic = 0;
    free(k);
}
static tr_ident_kernel *ker_alloc(int32_t nb, int32_t nullity, int32_t *error) {
    tr_ident_kernel *k = (tr_ident_kernel *)zalloc(sizeof *k);
    if (!k) { set_error(error, TR_MATH_MEMORY); return NULL; }
    k->magic = KER_MAGIC; k->nb = nb; k->nullity = nullity;
    k->vec = (tr_bigq **)zalloc((uint64_t)(nullity ? nullity : 1) * nb * sizeof *k->vec);
    if (!k->vec) { free(k); set_error(error, TR_MATH_MEMORY); return NULL; }
    return k;
}
tr_lin_batch *tr_ident_gram_system(const tr_ident_gram *g, const tr_math_budget *budget, tr_math_refusal *refusal,
                                   int32_t *error) {
    tr_budget_none(refusal);
    if (!gram_valid(g)) { set_error(error, TR_MATH_INVALID); return NULL; }
    const tr_ident_plan *p = g->plan;
    if (p->repr != TR_IDENT_EXACT) { set_error(error, TR_MATH_DOMAIN_ERROR); return NULL; }
    const int32_t nb = p->nb;
    const uint64_t na = (uint64_t)nb * nb;
    int8_t *sa = (int8_t *)zalloc(na), *sb = (int8_t *)zalloc((uint64_t)nb);
    uint32_t *nla = (uint32_t *)zalloc(na * 4), *dla = (uint32_t *)zalloc(na * 4);
    uint32_t *nlb = (uint32_t *)zalloc((uint64_t)nb * 4), *dlb = (uint32_t *)zalloc((uint64_t)nb * 4);
    uint64_t *la = (uint64_t *)zalloc(na * 16), *lb = (uint64_t *)zalloc(8);
    tr_lin_batch *B = NULL;
    if (sa && sb && nla && dla && nlb && dlb && la && lb) {
        uint64_t nl = 0;
        for (uint64_t i = 0; i < na; i++) {
            __int128 x = g->exact[p->map[i]];
            unsigned __int128 m = tr_cert_mag(x);
            sa[i] = (int8_t)(x > 0 ? 1 : x < 0 ? -1 : 0);
            uint32_t n = m ? ((m >> 64) ? 2 : 1) : 0;
            if (n >= 1) la[nl++] = (uint64_t)m;
            if (n == 2) la[nl++] = (uint64_t)(m >> 64);
            nla[i] = n;
        }
        tr_lin_qbuf qa, qb;
        qa.count = na; qa.nlimbs = nl; qa.sign = sa; qa.nlen = nla; qa.dlen = dla; qa.limbs = la;
        qb.count = (uint64_t)nb; qb.nlimbs = 0; qb.sign = sb; qb.nlen = nlb; qb.dlen = dlb; qb.limbs = lb;
        int32_t rows = nb, unk = nb;
        B = tr_lin_batch_systems(1, &rows, &unk, &qa, &qb, budget, refusal, error);
    } else set_error(error, TR_MATH_MEMORY);
    free(sa); free(sb); free(nla); free(dla); free(nlb); free(dlb); free(la); free(lb);
    return B;
}
tr_ident_kernel *tr_ident_kernel_from_lin(const tr_ident_gram *g, const tr_lin_result *res, const tr_math_budget *budget,
                                          tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    tr_lin_result_info ri;
    if (!gram_valid(g) || !res || tr_lin_result_get_info(res, &ri)) { set_error(error, TR_MATH_INVALID); return NULL; }
    const int32_t nb = g->plan->nb;
    if (ri.what != TR_LIN_RESULT_SOLVE || ri.cols != nb || ri.rows != nb || !ri.certified ||
        (ri.status != TR_LIN_UNIQUE && ri.status != TR_LIN_FAMILY)) { set_error(error, TR_MATH_DOMAIN_ERROR); return NULL; }
    uint64_t count = 0;
    const tr_bigq *const *null = tr_lin_result_part(res, TR_LIN_PART_NULL, &count);
    if (count != (uint64_t)ri.nullity * nb || (ri.nullity && !null)) { set_error(error, TR_MATH_INVALID); return NULL; }
    int32_t rc = check_one(budget, TR_BUDGET_OUTPUT, (uint64_t)ri.nullity, refusal);
    if (rc) { set_error(error, rc); return NULL; }
    tr_ident_kernel *k = ker_alloc(nb, ri.nullity, error);
    if (!k) return NULL;
    k->rank = ri.rank; k->method = TR_IDENT_KERNEL_LIN; k->exact_rank = 1; k->primes = (int32_t)ri.primes;
    k->agreeing = (int32_t)ri.primes; k->pivots = ri.pivots;
    k->free = (nb >= 64 ? UINT64_MAX : ((UINT64_C(1) << nb) - 1)) & ~ri.pivots;
    k->resolved = ri.nullity >= 64 ? UINT64_MAX : ((UINT64_C(1) << ri.nullity) - 1);
    for (uint64_t i = 0; i < count; i++) {
        int32_t e = 0;
        k->vec[i] = tr_bigq_copy(null[i], budget, refusal, &e);
        if (!k->vec[i]) { tr_ident_kernel_free(k); set_error(error, e ? e : TR_MATH_MEMORY); return NULL; }
    }
    k->bytes = sizeof *k + count * 64;
    set_error(error, TR_MATH_OK);
    return k;
}
/* the mixed-radix order of pivot masks: the mask whose lowest differing bit is set comes first (LIN-002's key) */
static int mask_less(uint64_t a, uint64_t b) { uint64_t d = a ^ b; return d && (a & d & (~d + 1)); }
static tr_ident_kernel *kernel_modular(const tr_ident_gram *g, const tr_math_budget *budget, tr_math_refusal *refusal,
                                       int32_t *error) {
    const tr_ident_plan *p = g->plan;
    const int32_t nb = p->nb, np = p->nprimes;
    const uint64_t cells = (uint64_t)nb * nb;
    uint64_t *a = (uint64_t *)zalloc((uint64_t)np * cells * 8), *hdr = (uint64_t *)zalloc((uint64_t)np * 4 * 8);
    uint64_t *r = (uint64_t *)zalloc((uint64_t)np * 8);
    tr_gfp *fs = (tr_gfp *)zalloc((uint64_t)np * sizeof(tr_gfp));
    if (!a || !hdr || !r || !fs) { free(a); free(hdr); free(r); free(fs); set_error(error, TR_MATH_MEMORY); return NULL; }
    for (int32_t q = 0; q < np; q++) {
        uint64_t *m = a + (uint64_t)q * cells;
        for (uint64_t i = 0; i < cells; i++) m[i] = g->res[(uint64_t)p->map[i] * np + q];
        tr_lin_modular_rref(&p->primes[q], m, nb, nb, hdr + 4 * (uint64_t)q);
        uint64_t inv = tr_gfp_inv_hd(&p->primes[q], hdr[4 * q + 3]);   /* the RREF itself: D^-1 (D RREF) */
        for (uint64_t t = 0; t < hdr[4 * q]; t++)
            for (int32_t j = 0; j < nb; j++) m[t * nb + j] = tr_gfp_mul_hd(&p->primes[q], m[t * nb + j], inv);
    }
    int32_t best = 0;
    for (int32_t q = 1; q < np; q++) {
        const uint64_t *h = hdr + 4 * (uint64_t)q, *hb = hdr + 4 * (uint64_t)best;
        if (h[0] > hb[0] || (h[0] == hb[0] && mask_less(h[1], hb[1]))) best = q;
    }
    const uint64_t rank = hdr[4 * (uint64_t)best], piv = hdr[4 * (uint64_t)best + 1];
    int32_t idx[TR_IDENT_MAX_PRIMES], na = 0;
    for (int32_t q = 0; q < np; q++) if (hdr[4 * q] == rank && hdr[4 * q + 1] == piv) { idx[na] = q; fs[na] = p->primes[q]; na++; }
    const int32_t nullity = nb - (int32_t)rank;
    tr_ident_kernel *k = ker_alloc(nb, nullity, error);
    if (!k) { free(a); free(hdr); free(r); free(fs); return NULL; }
    k->rank = (int32_t)rank; k->method = TR_IDENT_KERNEL_MODULAR; k->exact_rank = 0; k->primes = np; k->agreeing = na;
    k->pivots = piv; k->free = (nb >= 64 ? UINT64_MAX : ((UINT64_C(1) << nb) - 1)) & ~piv;
    int32_t pivcol[TR_IDENT_MAX_BASIS], npiv = 0;
    for (int32_t j = 0; j < nb; j++) if ((piv >> j) & 1) pivcol[npiv++] = j;
    ictx x = {budget, refusal, 0};
    int32_t f = 0;
    for (int32_t col = 0; col < nb && LIVE(&x); col++) {
        if ((piv >> col) & 1) continue;
        tr_bigq **vec = k->vec + (uint64_t)f * nb;
        int ok = 1;
        for (int32_t t = 0; t < npiv && ok && LIVE(&x); t++) {
            if (pivcol[t] > col) continue;                    /* RREF: zero left of a pivot */
            for (int32_t i = 0; i < na; i++) r[i] = a[(uint64_t)idx[i] * cells + (uint64_t)t * nb + col];
            tr_bigint *u = NULL, *M = NULL;
            int32_t e = tr_carrier_crt(r, fs, (uint32_t)na, 0, &u, &M, budget, refusal);
            if (e) { ifail(&x, e); break; }
            int32_t e2 = 0;
            tr_bigq *q = tr_carrier_ratrecon(u, M, NULL, NULL, budget, refusal, &e2);
            tr_bigint_free(u); tr_bigint_free(M);
            if (!q) { if (e2 == TR_MATH_NO_SOLUTION) ok = 0; else ifail(&x, e2); break; }
            int32_t e3 = 0;
            vec[pivcol[t]] = tr_bigq_neg(q, budget, refusal, &e3);   /* x_p = -RREF[t][f] */
            tr_bigq_free(q);
            if (!vec[pivcol[t]]) ifail(&x, e3);
        }
        if (LIVE(&x) && ok) {
            int32_t e = 0;
            tr_bigint *one = tr_bigint_from_i64(1, budget, refusal, &e), *zero = one ? tr_bigint_from_i64(0, budget, refusal, &e) : NULL;
            tr_bigint *den = one ? tr_bigint_from_i64(1, budget, refusal, &e) : NULL;
            if (!one || !zero || !den) ifail(&x, e);
            for (int32_t j = 0; j < nb && LIVE(&x); j++) {
                if (vec[j]) continue;
                vec[j] = tr_bigq_from_bigints(j == col ? one : zero, den, budget, refusal, &e);
                if (!vec[j]) ifail(&x, e);
            }
            tr_bigint_free(one); tr_bigint_free(zero); tr_bigint_free(den);
            if (LIVE(&x)) k->resolved |= UINT64_C(1) << f;
        } else if (LIVE(&x)) {
            for (int32_t j = 0; j < nb; j++) { tr_bigq_free(vec[j]); vec[j] = NULL; }
            k->unresolved++;
        }
        f++;
    }
    free(a); free(hdr); free(r); free(fs);
    if (x.rc) { tr_ident_kernel_free(k); set_error(error, x.rc); return NULL; }
    k->bytes = sizeof *k + (uint64_t)nullity * nb * 64;
    set_error(error, TR_MATH_OK);
    return k;
}
tr_ident_kernel *tr_ident_kernel_build(const tr_ident_gram *g, const tr_math_budget *budget, tr_math_refusal *refusal,
                                       int32_t *error) {
    tr_budget_none(refusal);
    if (!gram_valid(g)) { set_error(error, TR_MATH_INVALID); return NULL; }
    const tr_ident_plan *p = g->plan;
    tr_math_need need;
    need_zero(&need);
    need.need[TR_BUDGET_TENSOR] = tr_budget_mul((uint64_t)p->nb, (uint64_t)p->nb + 1);
    need.need[TR_BUDGET_OUTPUT] = (uint64_t)p->nb;
    if (p->repr == TR_IDENT_MODULAR) {
        need.need[TR_BUDGET_ITERATIONS] = (uint64_t)p->nprimes;
        need.need[TR_BUDGET_BYTES] = tr_budget_mul(tr_budget_mul((uint64_t)p->nprimes, (uint64_t)p->nb * p->nb), 8);
    }
    int32_t rc = check_need(budget, &need, refusal);
    if (rc) { set_error(error, rc); return NULL; }
    if (p->repr == TR_IDENT_MODULAR) return kernel_modular(g, budget, refusal, error);
    tr_lin_batch *B = tr_ident_gram_system(g, budget, refusal, error);
    if (!B) return NULL;
    uint64_t np = tr_lin_batch_primes_needed(B);
    tr_ident_kernel *k = NULL;
    rc = TR_MATH_NOT_CONVERGED;
    for (int round = 0; round < LIN_ROUNDS && rc == TR_MATH_NOT_CONVERGED; round++) {
        rc = check_one(budget, TR_BUDGET_ITERATIONS, np, refusal);
        if (rc) break;
        tr_lin_batch_view bv;
        tr_lin_batch_get_view(B, &bv);
        uint64_t words = tr_lin_batch_words(&bv, np);
        rc = check_one(budget, TR_BUDGET_BYTES, tr_budget_add(tr_budget_mul(words, 8), tr_budget_mul(np, sizeof(tr_gfp))), refusal);
        if (rc) break;
        tr_gfp *fs = (tr_gfp *)zalloc(np * sizeof(tr_gfp));
        uint64_t *res = (uint64_t *)zalloc(words * 8);
        if (!fs || !res) { free(fs); free(res); rc = TR_MATH_MEMORY; break; }
        rc = tr_lin_primes(np, fs);
        if (!rc) rc = tr_lin_batch_residues(B, fs, np, res, budget, refusal);
        tr_lin_result *out = NULL;
        uint64_t more = 0;
        if (!rc) rc = tr_lin_batch_finish(B, fs, np, res, &out, &more, budget, refusal);
        if (!rc) {
            k = tr_ident_kernel_from_lin(g, out, budget, refusal, &rc);
            tr_lin_result_free(out);
        } else if (rc == TR_MATH_NOT_CONVERGED) np = more;
        free(fs); free(res);
    }
    tr_lin_batch_free(B);
    if (!k) { set_error(error, rc ? rc : TR_MATH_NOT_CONVERGED); return NULL; }
    set_error(error, TR_MATH_OK);
    return k;
}
int32_t tr_ident_kernel_get_info(const tr_ident_kernel *k, tr_ident_kernel_info *out) {
    if (!ker_valid(k) || !out) return TR_MATH_INVALID;
    memset(out, 0, sizeof *out);
    out->nbasis = k->nb; out->rank = k->rank; out->nullity = k->nullity; out->method = k->method;
    out->exact_rank = k->exact_rank; out->primes = k->primes; out->agreeing = k->agreeing; out->unresolved = k->unresolved;
    out->pivots = k->pivots; out->free = k->free; out->bytes = k->bytes;
    return TR_MATH_OK;
}
uint64_t tr_ident_kernel_resolved(const tr_ident_kernel *k) { return ker_valid(k) ? k->resolved : 0; }
/* a kernel's vectors with zeros in place of unresolved rows (owned copies for the LIN-002 exporter) */
static tr_bigq **kernel_rows(const tr_ident_kernel *k, int32_t *rc) {
    uint64_t n = (uint64_t)k->nullity * k->nb;
    tr_bigq **out = (tr_bigq **)zalloc((n ? n : 1) * sizeof *out);
    if (!out) { *rc = TR_MATH_MEMORY; return NULL; }
    tr_bigint *zero = tr_bigint_from_i64(0, NULL, NULL, rc), *one = zero ? tr_bigint_from_i64(1, NULL, NULL, rc) : NULL;
    for (uint64_t i = 0; i < n && one && !*rc; i++) {
        out[i] = k->vec[i] ? tr_bigq_copy(k->vec[i], NULL, NULL, rc) : tr_bigq_from_bigints(zero, one, NULL, NULL, rc);
        if (!out[i] && !*rc) *rc = TR_MATH_MEMORY;
    }
    tr_bigint_free(zero); tr_bigint_free(one);
    if (*rc) { Q_free_n(out, n); return NULL; }
    return out;
}
uint64_t tr_ident_kernel_limbs(const tr_ident_kernel *k) {
    if (!ker_valid(k)) return 0;
    int32_t rc = 0;
    tr_bigq **rows = kernel_rows(k, &rc);
    if (!rows) return 0;
    uint64_t n = (uint64_t)k->nullity * k->nb, l = tr_lin_q_limbs((const tr_bigq *const *)rows, n);
    Q_free_n(rows, n);
    return l;
}
int32_t tr_ident_kernel_export(const tr_ident_kernel *k, tr_lin_qbuf *out) {
    if (!ker_valid(k) || !out) return TR_MATH_INVALID;
    int32_t rc = 0;
    tr_bigq **rows = kernel_rows(k, &rc);
    if (!rows) return rc;
    uint64_t n = (uint64_t)k->nullity * k->nb;
    rc = tr_lin_q_export((const tr_bigq *const *)rows, n, out);
    Q_free_n(rows, n);
    return rc;
}

/* ================================================================ exact distinct support */
typedef struct { uint64_t key, row; } fprint;
static uint64_t mix64(uint64_t z) {
    z = (z ^ (z >> 30)) * UINT64_C(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94D049BB133111EB);
    return z ^ (z >> 31);
}
static int same_point(const tr_scope_view *v, int32_t nvars, const int32_t *var, uint32_t mask, uint64_t a, uint64_t b) {
    for (int32_t i = 0; i < nvars; i++)
        if (((mask >> i) & 1) && tr_scope_value(v, var[i], a) != tr_scope_value(v, var[i], b)) return 0;
    return 1;
}
static int support_row(const tr_scope_view *v, int32_t nvars, const int32_t *var, uint32_t mask, int32_t weight, uint64_t r) {
    for (int32_t i = 0; i < nvars; i++) if (((mask >> i) & 1) && !tr_scope_present(v, var[i], r)) return 0;
    return weight < 0 || tr_scope_present(v, weight, r);
}
#define RADIX_BUCKETS 65536
/* stable LSD radix sort of n fingerprints by key (16-bit digits; tmp: n scratch entries, cnt: RADIX_BUCKETS) */
static fprint *radix_sort(fprint *a, fprint *tmp, uint64_t n, uint64_t *cnt) {
    for (int pass = 0; pass < 4; pass++) {
        const int shift = 16 * pass;
        memset(cnt, 0, RADIX_BUCKETS * sizeof *cnt);
        for (uint64_t i = 0; i < n; i++) cnt[(a[i].key >> shift) & 0xffff]++;
        uint64_t sum = 0;
        for (uint64_t b = 0; b < RADIX_BUCKETS; b++) { uint64_t c = cnt[b]; cnt[b] = sum; sum += c; }
        for (uint64_t i = 0; i < n; i++) tmp[cnt[(a[i].key >> shift) & 0xffff]++] = a[i];
        fprint *t = a; a = tmp; tmp = t;
    }
    return a;                                                 /* four passes: the sorted data is back in the input */
}
/* bytes of a support count over n rows (the fingerprints, their scratch, the representative flags, the weight sums,
 * the radix counters) */
static uint64_t support_bytes(uint64_t n, int weighted) {
    uint64_t per = 2 * sizeof(fprint) + 1 + (weighted ? sizeof(__int128) : 0);
    return tr_budget_add(tr_budget_mul(n, per), RADIX_BUCKETS * 8);
}
static int32_t support_view(const tr_scope_view *v, int32_t nvars, const int32_t *var, uint32_t mask, int32_t weight,
                            const tr_math_budget *budget, uint64_t *points, uint64_t *rows, tr_math_refusal *refusal) {
    const uint64_t nw = (v->rows + 63) / 64;
    uint64_t n = 0;
    for (uint64_t w = 0; w < nw; w++) {
        uint64_t bits = v->selection[w];
        while (bits) {
            uint64_t r = w * 64 + (uint64_t)__builtin_ctzll(bits);
            bits &= bits - 1;
            n += (uint64_t)support_row(v, nvars, var, mask, weight, r);
        }
    }
    int32_t rc = check_one(budget, TR_BUDGET_BYTES, support_bytes(n, weight >= 0), refusal);
    if (rc) return rc;
    fprint *f = (fprint *)zalloc((n ? n : 1) * sizeof *f), *tmp = (fprint *)zalloc((n ? n : 1) * sizeof *tmp);
    uint64_t *cnt = (uint64_t *)zalloc(RADIX_BUCKETS * 8);
    uint8_t *rep = (uint8_t *)zalloc(n ? n : 1);
    __int128 *wsum = weight >= 0 ? (__int128 *)zalloc((n ? n : 1) * sizeof(__int128)) : NULL;
    if (!f || !tmp || !cnt || !rep || (weight >= 0 && !wsum)) {
        free(f); free(tmp); free(cnt); free(rep); free(wsum); return TR_MATH_MEMORY;
    }
    uint64_t at = 0;
    for (uint64_t w = 0; w < nw; w++) {
        uint64_t bits = v->selection[w];
        while (bits) {
            uint64_t r = w * 64 + (uint64_t)__builtin_ctzll(bits);
            bits &= bits - 1;
            if (!support_row(v, nvars, var, mask, weight, r)) continue;
            uint64_t h = UINT64_C(0x243F6A8885A308D3);
            for (int32_t i = 0; i < nvars; i++)
                if ((mask >> i) & 1) h = mix64(h ^ ((uint64_t)tr_scope_value(v, var[i], r) + UINT64_C(0x9E3779B97F4A7C15) * (uint64_t)(i + 1)));
            f[at].key = h; f[at].row = r; at++;
        }
    }
    fprint *sorted = radix_sort(f, tmp, n, cnt);
    uint64_t count = 0;
    for (uint64_t i = 0; i < n;) {
        uint64_t j = i;
        while (j < n && sorted[j].key == sorted[i].key) j++;
        for (uint64_t a = i; a < j; a++) {                    /* equal fingerprints: compared value by value */
            uint64_t owner = a;
            for (uint64_t b = i; b < a; b++)
                if (rep[b] && same_point(v, nvars, var, mask, sorted[a].row, sorted[b].row)) { owner = b; break; }
            if (owner == a) rep[a] = 1;
            if (wsum) wsum[owner] += (__int128)tr_scope_value(v, weight, sorted[a].row);   /* |sum| < 2^63 * 2^32 */
        }
        for (uint64_t a = i; a < j; a++) if (rep[a] && (!wsum || wsum[a] != 0)) count++;
        i = j;
    }
    free(f); free(tmp); free(cnt); free(rep); free(wsum);
    *points = count;
    *rows = n;
    return TR_MATH_OK;
}
int32_t tr_ident_support(const tr_scope *scope, int32_t nvars, const int32_t *var, uint32_t mask, int32_t weight,
                         const tr_math_budget *budget, uint64_t *points, uint64_t *rows, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    tr_scope_view v;
    if (!scope || tr_scope_get_view(scope, &v) || !var || !points || !rows || nvars < 1 || nvars > TR_IDENT_MAX_VARS ||
        (nvars < 32 && (mask >> nvars)) || weight < -1 || weight >= v.ncols) return TR_MATH_INVALID;
    for (int32_t i = 0; i < nvars; i++) if (((mask >> i) & 1) && !column_ok(&v, var[i])) return TR_MATH_INVALID;
    if (weight >= 0 && !column_ok(&v, weight)) return TR_MATH_INVALID;
    return support_view(&v, nvars, var, mask, weight, budget, points, rows, refusal);
}

/* ================================================================ identities */
struct tr_ident_set_s {
    uint32_t magic;
    int32_t k, ncand, nb;
    int32_t var[TR_IDENT_MAX_VARS];
    uint64_t nterms;
    uint64_t *offset;                /* [ncand + 1] */
    uint32_t *support;
    int32_t *words;
    uint8_t *exps;                   /* nterms x k, each identity's terms in decreasing graded order */
    int64_t *coefs;
    uint8_t *cexp;                   /* the FND-007 claims: canonical terms over the claim's columns */
    int64_t *ccoef;
    uint64_t *coff;                  /* [ncand] byte offset of identity i's claim exponents */
    int32_t *ccols;                  /* [ncand] claim columns */
    tr_ident_item *items;
    tr_ident_set_info info;
    uint64_t bytes;
};
static int set_valid(const tr_ident_set *s) { return s && s->magic == SET_MAGIC; }
void tr_ident_set_free(tr_ident_set *s) {
    if (!set_valid(s)) return;
    free(s->offset); free(s->support); free(s->words); free(s->exps); free(s->coefs); free(s->cexp); free(s->ccoef);
    free(s->coff); free(s->ccols); free(s->items);
    s->magic = 0;
    free(s);
}
/* a working identity before it is stored */
typedef struct {
    int32_t origin, lead, vector, resolved, representable;
    uint64_t nterms;                 /* nonzero terms (T), also when not representable */
    uint8_t *e;                      /* stored terms (representable): nterms x k, decreasing graded */
    int64_t *c;
    tr_bigint **big;                 /* kernel identities: the primitive integer vector over the basis (nb) */
    int32_t sos, known, consequence;
} work;
static void work_free(work *w, int32_t nb) {
    free(w->e); free(w->c);
    Z_free_n(w->big, w->big ? (uint64_t)nb : 0);
    w->e = NULL; w->c = NULL; w->big = NULL;
}
/* the bound sum_t |c_t| prod_v M_v^e and its words (0 past TR_IDENT_MAX_WORDS) */
static int32_t eval_words(const tr_scope_view *v, int32_t k, const int32_t *var, const uint8_t *e, const int64_t *c,
                          uint64_t nt, uint64_t *bits) {
    enum { WW = TR_IDENT_MAX_WORDS + 2 };
    uint64_t total[WW] = {0}, term[WW];
    int sat = 0;
    for (uint64_t t = 0; t < nt && !sat; t++) {
        memset(term, 0, sizeof term);
        term[0] = tr_abs_u64(c[t]);
        for (int32_t i = 0; i < k && !sat; i++) {
            uint64_t m = mag_of(v, var[i]);
            for (uint8_t q = 0; q < e[t * k + i] && !sat; q++) {
                unsigned __int128 carry = 0;
                for (int32_t w = 0; w < WW; w++) { unsigned __int128 x = (unsigned __int128)term[w] * m + carry; term[w] = (uint64_t)x; carry = x >> 64; }
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
    if (sat) { *bits = (uint64_t)64 * WW; return 0; }
    uint64_t b = 0;
    for (int32_t w = WW; w-- > 0;) if (total[w]) { b = (uint64_t)w * 64 + (uint64_t)(64 - __builtin_clzll(total[w])); break; }
    *bits = b;
    for (int32_t W = 2; W <= TR_IDENT_MAX_WORDS; W++) if (b <= (uint64_t)(64 * W - 1)) return W;
    return 0;
}
/* the primitive integer vector of a rational kernel vector: denominators cleared, content removed */
static tr_bigint **primitive(ictx *x, tr_bigq *const *v, int32_t n) {
    tr_bigint **out = (tr_bigint **)zalloc((uint64_t)n * sizeof *out);
    if (!out) { ifail(x, TR_MATH_MEMORY); return NULL; }
    tr_bigint *L = Z_i64(x, 1), *G = Z_i64(x, 0);
    for (int32_t i = 0; i < n && LIVE(x); i++) {
        if (!tr_bigq_sign(v[i])) continue;
        const tr_bigint *d = tr_bigq_denominator(v[i]);
        tr_bigint *g = Z_gcd(x, L, d), *t = g ? Z_div(x, L, g) : NULL, *nl = t ? Z_mul(x, t, d) : NULL;
        tr_bigint_free(g); tr_bigint_free(t);
        Z_set(&L, nl);
    }
    for (int32_t i = 0; i < n && LIVE(x); i++) {
        if (!tr_bigq_sign(v[i])) { out[i] = Z_i64(x, 0); continue; }
        tr_bigint *t = Z_div(x, L, tr_bigq_denominator(v[i]));
        out[i] = t ? Z_mul(x, tr_bigq_numerator(v[i]), t) : NULL;
        tr_bigint_free(t);
        tr_bigint *g = out[i] ? Z_gcd(x, G, out[i]) : NULL;
        Z_set(&G, g);
    }
    if (LIVE(x) && tr_bigint_sign(G) > 0)
        for (int32_t i = 0; i < n && LIVE(x); i++) {
            if (!tr_bigint_sign(out[i])) continue;
            tr_bigint *q = Z_div(x, out[i], G);
            Z_set(&out[i], q);
        }
    tr_bigint_free(L); tr_bigint_free(G);
    if (x->rc) { Z_free_n(out, (uint64_t)n); return NULL; }
    return out;
}
/* exact c^T G c (c integer over the basis) as a big integer */
static tr_bigint *quad_exact(ictx *x, const tr_ident_gram *g, tr_bigint *const *c) {
    const tr_ident_plan *p = g->plan;
    const int32_t nb = p->nb;
    tr_bigint *total = Z_i64(x, 0);
    for (int32_t i = 0; i < nb && LIVE(x); i++) {
        if (!tr_bigint_sign(c[i])) continue;
        tr_bigint *h = Z_i64(x, 0);
        for (int32_t j = 0; j < nb && LIVE(x); j++) {
            __int128 gij = g->exact[p->map[(uint64_t)i * nb + j]];
            if (!gij || !tr_bigint_sign(c[j])) continue;
            tr_bigint *G = Z_i128(x, gij), *t = G ? Z_mul(x, G, c[j]) : NULL, *s = t ? Z_add(x, h, t) : NULL;
            tr_bigint_free(G); tr_bigint_free(t);
            Z_set(&h, s);
        }
        tr_bigint *t = h ? Z_mul(x, h, c[i]) : NULL, *s = t ? Z_add(x, total, t) : NULL;
        tr_bigint_free(h); tr_bigint_free(t);
        Z_set(&total, s);
    }
    if (x->rc) { tr_bigint_free(total); return NULL; }
    return total;
}
static int32_t sos_verdict(const tr_ident_gram *g, int zero) {
    if (g->plan->repr != TR_IDENT_EXACT) return TR_IDENT_SOS_MODULAR;
    if (!zero) return TR_IDENT_SOS_NONZERO;
    return g->sign == TR_IDENT_SIGNED ? TR_IDENT_SOS_SIGNED : TR_IDENT_SOS_CERTIFIED;
}
/* canonical terms of a given polynomial: merged, zeros dropped, decreasing graded order (into owned arrays) */
static int32_t canon_terms(int32_t k, const tr_ident_poly *pl, uint8_t **eo, int64_t **co, uint64_t *nt) {
    *eo = NULL; *co = NULL; *nt = 0;
    if (pl->nterms > TR_CERT_MAX_TERMS) return TR_MATH_UNSUPPORTED;
    if (pl->nterms && (!pl->exponents || !pl->coefficients)) return TR_MATH_INVALID;
    uint64_t n = pl->nterms;
    pair_ref *r = (pair_ref *)zalloc((n ? n : 1) * sizeof *r);
    uint8_t *e = (uint8_t *)zalloc((n ? n : 1) * (uint64_t)k);
    int64_t *c = (int64_t *)zalloc((n ? n : 1) * 8);
    if (!r || !e || !c) { free(r); free(e); free(c); return TR_MATH_MEMORY; }
    for (uint64_t t = 0; t < n; t++) { memcpy(r[t].e, pl->exponents + t * k, (size_t)k); r[t].i = (int32_t)t; r[t].j = 0; }
    qsort(r, (size_t)n, sizeof *r, pair_cmp);
    uint64_t m = 0;
    int32_t rc = TR_MATH_OK;
    for (uint64_t i = n; i-- > 0 && !rc;) {                  /* decreasing order */
        __int128 s = pl->coefficients[r[i].i];
        while (i > 0 && !tr_ident_cmp(r[i - 1].e, r[i].e, TR_IDENT_MAX_VARS)) { i--; s += pl->coefficients[r[i].i]; }
        if (s < INT64_MIN || s > INT64_MAX) rc = TR_MATH_OVERFLOW;
        else if (s) { memcpy(e + m * k, r[i].e, (size_t)k); c[m++] = (int64_t)s; }
    }
    free(r);
    if (rc) { free(e); free(c); return rc; }
    *eo = e; *co = c; *nt = m;
    return TR_MATH_OK;
}
/* ------------------------------------------------------------------ spans of multiples within the basis */
/* An echelon basis (each row's pivot: its largest monomial, distinct) of polynomials over the basis, by fraction-free
 * elimination in 128 bits with the content removed after each step; an overflow is reported (-1), never wrapped. */
typedef struct {
    int32_t nb, k, ne, reserved;
    const mono *basis;
    int32_t at[TR_IDENT_MAX_BASIS];                  /* the echelon row of each pivot column, or -1 */
    __int128 rows[TR_IDENT_MAX_BASIS][TR_IDENT_MAX_BASIS];
} span_t;
static void span_init(span_t *S, const mono *basis, int32_t nb, int32_t k) {
    S->nb = nb; S->k = k; S->ne = 0; S->basis = basis;
    for (int32_t b = 0; b < TR_IDENT_MAX_BASIS; b++) S->at[b] = -1;
}
/* 1: row reduces to zero (it lies in the span); 0: a new pivot remains (row reduced); -1: overflow */
static int span_reduce(const span_t *S, __int128 *row) {
    const int32_t nb = S->nb;
    for (;;) {
        int32_t piv = -1;
        for (int32_t j = nb; j-- > 0;) if (row[j]) { piv = j; break; }
        if (piv < 0) return 1;
        if (S->at[piv] < 0) return 0;
        const __int128 *e = S->rows[S->at[piv]];
        __int128 a = e[piv], c = row[piv];
        unsigned __int128 g = 0;
        for (int32_t j = 0; j < nb; j++) {
            __int128 x1, x2, y;
            if (__builtin_mul_overflow(a, row[j], &x1) || __builtin_mul_overflow(c, e[j], &x2) ||
                __builtin_sub_overflow(x1, x2, &y)) return -1;
            row[j] = y;
            unsigned __int128 m = tr_cert_mag(y), h = g;
            while (m) { unsigned __int128 t = h % m; h = m; m = t; }
            g = h;
        }
        if (g > 1) for (int32_t j = 0; j < nb; j++) row[j] /= (__int128)g;
    }
}
static int span_insert(span_t *S, __int128 *row) {
    int r = span_reduce(S, row);
    if (r) return r < 0 ? -1 : 0;
    int32_t piv = -1;
    for (int32_t j = S->nb; j-- > 0;) if (row[j]) { piv = j; break; }
    S->at[piv] = S->ne;
    memcpy(S->rows[S->ne], row, (size_t)S->nb * sizeof(__int128));
    S->ne++;
    return 0;
}
/* the coefficients of a polynomial (terms in the basis, leading term first) as a row; 0 when a term is outside */
static int span_row(const span_t *S, const uint8_t *e, const int64_t *c, uint64_t nt, __int128 *row) {
    for (int32_t j = 0; j < S->nb; j++) row[j] = 0;
    for (uint64_t t = 0; t < nt; t++) {
        mono m;
        memset(&m, 0, sizeof m);
        memcpy(m.e, e + t * S->k, (size_t)S->k);
        int32_t j = mono_find(S->basis, S->nb, m.e);
        if (j < 0) return 0;
        row[j] = c[t];
    }
    return 1;
}
/* adds every multiple u * poly whose terms all lie in the basis (u = m_b / lead for the basis monomials m_b the
 * leading monomial divides); -1 on overflow */
static int span_multiples(span_t *S, const uint8_t *e, const int64_t *c, uint64_t nt) {
    if (!nt) return 0;
    const int32_t k = S->k;
    __int128 row[TR_IDENT_MAX_BASIS];
    for (int32_t b = 0; b < S->nb; b++) {
        mono u;
        memset(&u, 0, sizeof u);
        int ok = 1;
        for (int32_t i = 0; i < k && ok; i++) {
            if (S->basis[b].e[i] < e[i]) ok = 0;
            else u.e[i] = (uint8_t)(S->basis[b].e[i] - e[i]);
        }
        if (!ok) continue;
        for (int32_t j = 0; j < S->nb; j++) row[j] = 0;
        for (uint64_t t = 0; t < nt && ok; t++) {
            mono m;
            memset(&m, 0, sizeof m);
            for (int32_t i = 0; i < k; i++) {
                int sum = u.e[i] + e[t * k + i];
                if (sum > 255) ok = 0;
                m.e[i] = (uint8_t)sum;
            }
            int32_t j = ok ? mono_find(S->basis, S->nb, m.e) : -1;
            if (j < 0) ok = 0; else row[j] = c[t];
        }
        if (ok && span_insert(S, row) < 0) return -1;
    }
    return 0;
}
/* The exact labels of the kernel identities w[0..nk) (increasing leading monomial) against the declared relations
 * d[0..nd): known = in the span of the declared multiples; consequence = in the span of those and of the multiples
 * of the earlier generators; else a generator, whose multiples join the span. *exact = 0 when the 128-bit elimination
 * overflowed (labels then undecided: all 0). */
static int32_t label_kernel(const tr_ident_plan *p, work *w, int32_t nk, const work *d, int32_t nd, int32_t *exact,
                            int32_t *krank) {
    mono basis[TR_IDENT_MAX_BASIS];
    memset(basis, 0, sizeof basis);
    for (int32_t b = 0; b < p->nb; b++) memcpy(basis[b].e, p->bexp + (uint64_t)b * p->k, (size_t)p->k);
    span_t *S = (span_t *)zalloc(sizeof *S), *SK = (span_t *)zalloc(sizeof *SK);
    if (!S || !SK) { free(S); free(SK); return TR_MATH_MEMORY; }
    span_init(S, basis, p->nb, p->k);
    int ok = 1;
    for (int32_t r = 0; r < nd && ok; r++) if (span_multiples(S, d[r].e, d[r].c, d[r].nterms) < 0) ok = 0;
    *krank = S->ne;
    memcpy(SK, S, sizeof *S);
    __int128 row[TR_IDENT_MAX_BASIS], r1[TR_IDENT_MAX_BASIS];
    for (int32_t i = 0; i < nk && ok; i++) {
        work *it = &w[i];
        if (!it->e || !it->representable || !it->resolved || !span_row(S, it->e, it->c, it->nterms, row)) continue;
        memcpy(r1, row, sizeof row);
        int inK = span_reduce(SK, r1);
        memcpy(r1, row, sizeof row);
        int inS = inK == 1 ? 1 : span_reduce(S, r1);
        if (inK < 0 || inS < 0) { ok = 0; break; }
        if (inK == 1) it->known = 1;
        else if (inS == 1) it->consequence = 1;
        else if (span_multiples(S, it->e, it->c, it->nterms) < 0) ok = 0;
    }
    free(S); free(SK);
    *exact = ok;
    if (!ok) for (int32_t i = 0; i < nk; i++) { w[i].known = 0; w[i].consequence = 0; }
    return TR_MATH_OK;
}
static int divides(const uint8_t *a, const uint8_t *b, int32_t k) {   /* monomial a divides b */
    for (int32_t i = 0; i < k; i++) if (a[i] > b[i]) return 0;
    return 1;
}
/* builds the set from working identities (takes their arrays); labels and counts over `scope` */
static tr_ident_set *set_build(const tr_scope_view *v, const tr_scope *scope, int32_t k, const int32_t *var,
                               const tr_ident_plan *plan, int32_t weight, work *w, int32_t n, int32_t nknown,
                               int32_t labels, int32_t known_rank, const tr_ident_kernel *ker,
                               const tr_ident_gram *gram, const tr_math_budget *budget, tr_math_refusal *refusal,
                               int32_t *error) {
    uint64_t nterms = 0;
    for (int32_t i = 0; i < n; i++) if (w[i].e) nterms += w[i].nterms;
    uint64_t bytes = tr_budget_add(sizeof(tr_ident_set), tr_budget_mul((uint64_t)n, sizeof(tr_ident_item) + 64));
    bytes = tr_budget_add(bytes, tr_budget_mul(nterms, 2 * ((uint64_t)k + 8)));
    int32_t rc = check_one(budget, TR_BUDGET_BYTES, bytes, refusal);
    if (rc) { set_error(error, rc); return NULL; }
    tr_ident_set *s = (tr_ident_set *)zalloc(sizeof *s);
    if (s) {
        s->magic = SET_MAGIC;
        s->offset = (uint64_t *)zalloc(((uint64_t)n + 1) * 8); s->support = (uint32_t *)zalloc((uint64_t)(n ? n : 1) * 4);
        s->words = (int32_t *)zalloc((uint64_t)(n ? n : 1) * 4);
        s->exps = (uint8_t *)zalloc((nterms ? nterms : 1) * (uint64_t)k); s->coefs = (int64_t *)zalloc((nterms ? nterms : 1) * 8);
        s->cexp = (uint8_t *)zalloc((nterms ? nterms : 1) * (uint64_t)k); s->ccoef = (int64_t *)zalloc((nterms ? nterms : 1) * 8);
        s->coff = (uint64_t *)zalloc((uint64_t)(n ? n : 1) * 8); s->ccols = (int32_t *)zalloc((uint64_t)(n ? n : 1) * 4);
        s->items = (tr_ident_item *)zalloc((uint64_t)(n ? n : 1) * sizeof(tr_ident_item));
    }
    if (!s || !s->offset || !s->support || !s->words || !s->exps || !s->coefs || !s->cexp || !s->ccoef || !s->coff ||
        !s->ccols || !s->items) { tr_ident_set_free(s); set_error(error, TR_MATH_MEMORY); return NULL; }
    s->k = k; s->ncand = n; s->nb = plan ? plan->nb : 0; s->nterms = nterms;
    for (int32_t i = 0; i < TR_IDENT_MAX_VARS; i++) s->var[i] = i < k ? var[i] : -1;
    const uint32_t full = (uint32_t)((UINT64_C(1) << k) - 1);
    /* the support counts, one pass per distinct support mask */
    uint32_t mask_seen[TR_IDENT_MAX_CANDIDATES + TR_IDENT_MAX_KNOWN + 64];
    uint64_t mask_pts[TR_IDENT_MAX_CANDIDATES + TR_IDENT_MAX_KNOWN + 64], mask_rows[TR_IDENT_MAX_CANDIDATES + TR_IDENT_MAX_KNOWN + 64];
    int32_t nmask = 0;
    uint64_t t0 = 0, coff = 0;
    for (int32_t i = 0; i < n && !rc; i++) {
        tr_ident_item *it = &s->items[i];
        it->index = i; it->origin = w[i].origin; it->lead = w[i].lead; it->vector = w[i].vector;
        it->nterms = (int32_t)w[i].nterms; it->representable = w[i].representable; it->resolved = w[i].resolved;
        it->sos = w[i].sos; it->parent = -1;
        s->offset[i] = t0;
        uint32_t sup = 0;
        int32_t deg = 0;
        if (w[i].e) {
            memcpy(s->exps + t0 * k, w[i].e, (size_t)(w[i].nterms * k));
            memcpy(s->coefs + t0, w[i].c, (size_t)(w[i].nterms * 8));
            for (uint64_t t = 0; t < w[i].nterms; t++) {
                int32_t d = 0;
                for (int32_t q = 0; q < k; q++) if (w[i].e[t * k + q]) { sup |= UINT32_C(1) << q; d += w[i].e[t * k + q]; }
                if (d > deg) deg = d;
            }
        } else if (w[i].big && plan) {                        /* not representable: support from the big vector */
            for (int32_t b = 0; b < plan->nb; b++) {
                if (!tr_bigint_sign(w[i].big[b])) continue;
                int32_t d = 0;
                for (int32_t q = 0; q < k; q++) if (plan->bexp[(uint64_t)b * k + q]) { sup |= UINT32_C(1) << q; d += plan->bexp[(uint64_t)b * k + q]; }
                if (d > deg) deg = d;
            }
        }
        if (!sup) sup = full;                                 /* a constant: the domain of every variable */
        it->support = sup; it->degree = deg;
        int32_t nc = 0;
        for (int32_t q = 0; q < k; q++) nc += (sup >> q) & 1;
        it->ncols = nc;
        s->support[i] = sup;
        /* the FND-007 claim over the support columns: canonical terms */
        s->coff[i] = coff; s->ccols[i] = nc;
        if (w[i].e && w[i].nterms) {
            uint8_t *pe = (uint8_t *)zalloc(w[i].nterms * (uint64_t)nc);
            uint64_t m = 0;
            if (!pe) { rc = TR_MATH_MEMORY; break; }
            for (uint64_t t = 0; t < w[i].nterms; t++) {
                int32_t c = 0;
                for (int32_t q = 0; q < k; q++) if ((sup >> q) & 1) pe[t * nc + c++] = w[i].e[t * k + q];
            }
            rc = tr_cert_poly_canonical(nc, w[i].nterms, pe, w[i].c, 0, 0, s->cexp + coff, s->ccoef + t0, &m);
            free(pe);
            if (!rc && m != w[i].nterms) rc = TR_MATH_INVALID;
            if (rc) break;
            coff += w[i].nterms * (uint64_t)nc;
        }
        if (w[i].e) t0 += w[i].nterms;
        /* the evaluation width */
        if (w[i].e && w[i].representable && w[i].resolved) {
            uint64_t bits = 0;
            it->words = eval_words(v, k, var, w[i].e, w[i].c, w[i].nterms, &bits);
            it->bound_bits = bits;
        }
        s->words[i] = it->words;
        /* distinct points of its support over its domain */
        int32_t at = -1;
        for (int32_t q = 0; q < nmask; q++) if (mask_seen[q] == sup) at = q;
        if (at < 0) {
            at = nmask++;
            mask_seen[at] = sup;
            rc = support_view(v, k, var, sup, -1, budget, &mask_pts[at], &mask_rows[at], refusal);
            if (rc) break;
        }
        it->points = mask_pts[at]; it->domain = mask_rows[at];
        it->forced = (uint64_t)it->nterms > it->points ? (uint64_t)it->nterms - it->points : 0;
        it->underdetermined = it->forced > 0;
    }
    s->offset[n] = t0;
    if (rc) { tr_ident_set_free(s); set_error(error, rc); return NULL; }
    /* labels (decided by the caller) and, informationally, the identity whose leading monomial divides */
    for (int32_t i = 0; i < n && plan; i++) {
        tr_ident_item *it = &s->items[i];
        it->known = w[i].known; it->consequence = w[i].consequence;
        if (it->origin != TR_IDENT_ORIGIN_KERNEL || it->lead < 0) continue;
        const uint8_t *L = plan->bexp + (uint64_t)it->lead * k;
        for (int32_t j = 0; j < n && it->parent == -1; j++) {
            const tr_ident_item *o = &s->items[j];
            if (j == i || o->origin == TR_IDENT_ORIGIN_GIVEN || o->lead < 0 || o->lead == it->lead) continue;
            if (divides(plan->bexp + (uint64_t)o->lead * k, L, k)) it->parent = o->origin == TR_IDENT_ORIGIN_DECLARED ? -2 : j;
        }
    }
    /* the basis level */
    tr_ident_set_info *si = &s->info;
    si->nvars = k; si->nbasis = plan ? plan->nb : 0; si->ncand = n; si->nknown = nknown; si->known_rank = known_rank;
    si->labels = labels;
    if (ker) { si->rank = ker->rank; si->nullity = ker->nullity; si->exact_rank = ker->exact_rank; si->unresolved = ker->unresolved; }
    if (gram) { si->repr = gram->plan->repr; si->sign = gram->sign; }
    for (int32_t i = 0; i < n; i++) {
        const tr_ident_item *it = &s->items[i];
        if (it->origin == TR_IDENT_ORIGIN_KERNEL) {
            if (it->known) si->known++;
            else if (it->consequence) si->consequences++;
            else si->generators++;
        }
        si->underdetermined += it->underdetermined;
        si->unrepresentable += !it->representable;
    }
    if (plan) {
        uint64_t pts = 0, rows = 0;
        rc = support_view(v, k, var, full, weight, budget, &pts, &rows, refusal);
        if (rc) { tr_ident_set_free(s); set_error(error, rc); return NULL; }
        si->points = pts; si->domain = rows;
        si->forced = (uint64_t)plan->nb > pts ? (uint64_t)plan->nb - pts : 0;
    }
    s->bytes = bytes;
    si->bytes = bytes;
    (void)scope;
    set_error(error, TR_MATH_OK);
    return s;
}
static int32_t set_scope_check(const tr_scope *scope, int32_t k, const int32_t *var, tr_scope_view *v) {
    if (!scope || tr_scope_get_view(scope, v) || k < 1 || k > TR_IDENT_MAX_VARS || !var) return TR_MATH_INVALID;
    for (int32_t i = 0; i < k; i++) {
        if (!column_ok(v, var[i])) return TR_MATH_INVALID;
        for (int32_t j = 0; j < i; j++) if (var[j] == var[i]) return TR_MATH_INVALID;
    }
    return TR_MATH_OK;
}
static int32_t declared_work(int32_t k, const tr_ident_plan *plan, const tr_ident_poly *known, int32_t nknown,
                             const tr_ident_gram *gram, work *w, const tr_math_budget *budget, tr_math_refusal *refusal) {
    for (int32_t r = 0; r < nknown; r++) {
        memset(&w[r], 0, sizeof w[r]);
        w[r].origin = TR_IDENT_ORIGIN_DECLARED; w[r].lead = -1; w[r].vector = -1; w[r].resolved = 1; w[r].representable = 1;
        int32_t rc = canon_terms(k, &known[r], &w[r].e, &w[r].c, &w[r].nterms);
        if (rc) return rc;
        for (uint64_t t = 0; t < w[r].nterms; t++) {
            int32_t d = 0;
            for (int32_t i = 0; i < k; i++) d += w[r].e[t * k + i];
            if (d > 255) return TR_MATH_UNSUPPORTED;
        }
        if (plan && w[r].nterms) {                            /* its leading monomial in the basis */
            for (int32_t b = 0; b < plan->nb; b++) if (!memcmp(plan->bexp + (uint64_t)b * k, w[r].e, (size_t)k)) w[r].lead = b;
        }
    }
    (void)budget; (void)refusal; (void)gram;
    return TR_MATH_OK;
}
/* identity i's coefficients over the basis (int64), or TR_MATH_DOMAIN_ERROR when a term is outside the basis */
static int32_t basis_coefs(const tr_ident_plan *p, int32_t k, const uint8_t *e, const int64_t *c, uint64_t nt, int64_t *out) {
    for (int32_t b = 0; b < p->nb; b++) out[b] = 0;
    for (uint64_t t = 0; t < nt; t++) {
        int32_t at = -1;
        for (int32_t b = 0; b < p->nb && at < 0; b++) if (!memcmp(p->bexp + (uint64_t)b * k, e + t * k, (size_t)k)) at = b;
        if (at < 0) return TR_MATH_DOMAIN_ERROR;
        out[at] = c[t];
    }
    return TR_MATH_OK;
}
/* the sos verdict of an int64 polynomial in the Gram's basis (TR_IDENT_SOS_NONE when outside the basis span) */
static int32_t sos_of_terms(const tr_ident_gram *g, int32_t k, const uint8_t *e, const int64_t *c, uint64_t nt,
                            const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *rc) {
    const tr_ident_plan *p = g->plan;
    int64_t co[TR_IDENT_MAX_BASIS];
    if (basis_coefs(p, k, e, c, nt, co)) return TR_IDENT_SOS_NONE;
    if (p->repr != TR_IDENT_EXACT) return TR_IDENT_SOS_MODULAR;
    ictx x = {budget, refusal, 0};
    tr_bigint *big[TR_IDENT_MAX_BASIS];
    memset(big, 0, sizeof big);
    for (int32_t b = 0; b < p->nb; b++) big[b] = Z_i64(&x, co[b]);
    tr_bigint *q = LIVE(&x) ? quad_exact(&x, g, big) : NULL;
    int32_t verdict = q ? sos_verdict(g, tr_bigint_sign(q) == 0) : TR_IDENT_SOS_NONE;
    tr_bigint_free(q);
    for (int32_t b = 0; b < p->nb; b++) tr_bigint_free(big[b]);
    if (x.rc) *rc = x.rc;
    return verdict;
}
tr_ident_set *tr_ident_propose(const tr_scope *scope, const tr_ident_gram *g, const tr_ident_kernel *ker,
                               const tr_ident_poly *known, int32_t nknown, const tr_math_budget *budget,
                               tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    if (!gram_valid(g) || !ker_valid(ker) || ker->nb != g->plan->nb || nknown < 0 || (nknown && !known)) {
        set_error(error, TR_MATH_INVALID); return NULL;
    }
    if (nknown > TR_IDENT_MAX_KNOWN) { set_error(error, TR_MATH_UNSUPPORTED); return NULL; }
    const tr_ident_plan *p = g->plan;
    const int32_t k = p->k, nb = p->nb;
    tr_scope_view v;
    int32_t rc = set_scope_check(scope, k, p->var, &v);
    if (!rc && p->weight >= 0 && !column_ok(&v, p->weight)) rc = TR_MATH_INVALID;
    if (rc) { set_error(error, rc); return NULL; }
    const int32_t n = ker->nullity + nknown;
    rc = check_one(budget, TR_BUDGET_OUTPUT, (uint64_t)n, refusal);
    if (rc) { set_error(error, rc); return NULL; }
    work *w = (work *)zalloc((uint64_t)(n ? n : 1) * sizeof *w);
    if (!w) { set_error(error, TR_MATH_MEMORY); return NULL; }
    ictx x = {budget, refusal, 0};
    int32_t f = 0;
    for (int32_t col = 0; col < nb && f < ker->nullity && LIVE(&x); col++) {
        if (!((ker->free >> col) & 1)) continue;
        work *it = &w[f];
        it->origin = TR_IDENT_ORIGIN_KERNEL; it->lead = col; it->vector = f;
        it->resolved = (int32_t)((ker->resolved >> f) & 1);
        it->representable = 1;
        if (!it->resolved) { it->representable = 1; f++; continue; }
        it->big = primitive(&x, ker->vec + (uint64_t)f * nb, nb);
        if (!it->big) break;
        uint64_t nt = 0;
        for (int32_t b = 0; b < nb; b++) {
            if (!tr_bigint_sign(it->big[b])) continue;
            nt++;
            int64_t c;
            if (tr_bigint_to_i64(it->big[b], &c)) it->representable = 0;
        }
        it->nterms = nt;
        if (it->representable) {
            it->e = (uint8_t *)zalloc((nt ? nt : 1) * (uint64_t)k);
            it->c = (int64_t *)zalloc((nt ? nt : 1) * 8);
            if (!it->e || !it->c) { ifail(&x, TR_MATH_MEMORY); break; }
            uint64_t t = 0;
            for (int32_t b = nb; b-- > 0;) {                  /* decreasing graded order: the leading term first */
                if (!tr_bigint_sign(it->big[b])) continue;
                memcpy(it->e + t * k, p->bexp + (uint64_t)b * k, (size_t)k);
                tr_bigint_to_i64(it->big[b], &it->c[t]);
                t++;
            }
        }
        /* the Gram's own verdict */
        if (p->repr == TR_IDENT_EXACT) {
            tr_bigint *q = quad_exact(&x, g, it->big);
            if (q) it->sos = sos_verdict(g, tr_bigint_sign(q) == 0);
            tr_bigint_free(q);
        } else it->sos = TR_IDENT_SOS_MODULAR;
        f++;
    }
    if (!x.rc) x.rc = declared_work(k, p, known, nknown, g, w + ker->nullity, budget, refusal);
    for (int32_t r = 0; r < nknown && !x.rc; r++) {
        work *d = &w[ker->nullity + r];
        if (d->nterms) {
            int32_t e = 0;
            d->sos = sos_of_terms(g, k, d->e, d->c, d->nterms, budget, refusal, &e);
            if (e) x.rc = e;
        }
    }
    int32_t krank = 0, labels = 0;
    if (!x.rc) x.rc = label_kernel(p, w, ker->nullity, w + ker->nullity, nknown, &labels, &krank);
    tr_ident_set *s = NULL;
    if (!x.rc) s = set_build(&v, scope, k, p->var, p, p->weight, w, n, nknown, labels, krank, ker, g, budget, refusal, &x.rc);
    for (int32_t i = 0; i < n; i++) work_free(&w[i], nb);
    free(w);
    if (!s) { set_error(error, x.rc ? x.rc : TR_MATH_MEMORY); return NULL; }
    set_error(error, TR_MATH_OK);
    return s;
}
tr_ident_set *tr_ident_candidates(const tr_scope *scope, int32_t nvars, const int32_t *var, const tr_ident_poly *polys,
                                  int32_t npolys, const tr_ident_gram *gram, const tr_math_budget *budget,
                                  tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    tr_scope_view v;
    int32_t rc = set_scope_check(scope, nvars, var, &v);
    if (!rc && (npolys < 0 || (npolys && !polys))) rc = TR_MATH_INVALID;
    if (!rc && gram && (!gram_valid(gram) || gram->plan->k != nvars || memcmp(gram->plan->var, var, (size_t)nvars * 4)))
        rc = TR_MATH_DOMAIN_ERROR;
    if (rc) { set_error(error, rc); return NULL; }
    if (npolys > TR_IDENT_MAX_CANDIDATES) { set_error(error, TR_MATH_UNSUPPORTED); return NULL; }
    rc = check_one(budget, TR_BUDGET_OUTPUT, (uint64_t)npolys, refusal);
    if (rc) { set_error(error, rc); return NULL; }
    work *w = (work *)zalloc((uint64_t)(npolys ? npolys : 1) * sizeof *w);
    if (!w) { set_error(error, TR_MATH_MEMORY); return NULL; }
    rc = declared_work(nvars, gram ? gram->plan : NULL, polys, npolys, NULL, w, budget, refusal);
    for (int32_t i = 0; i < npolys && !rc; i++) {
        w[i].origin = TR_IDENT_ORIGIN_GIVEN;
        if (gram && w[i].nterms) w[i].sos = sos_of_terms(gram, nvars, w[i].e, w[i].c, w[i].nterms, budget, refusal, &rc);
    }
    tr_ident_set *s = NULL;
    if (!rc) s = set_build(&v, scope, nvars, var, NULL, -1, w, npolys, 0, 0, 0, NULL, gram, budget, refusal, &rc);
    for (int32_t i = 0; i < npolys; i++) work_free(&w[i], 0);
    free(w);
    if (!s) { set_error(error, rc ? rc : TR_MATH_MEMORY); return NULL; }
    set_error(error, TR_MATH_OK);
    return s;
}
int32_t tr_ident_set_get_info(const tr_ident_set *s, tr_ident_set_info *out) {
    if (!set_valid(s) || !out) return TR_MATH_INVALID;
    *out = s->info;
    return TR_MATH_OK;
}
int32_t tr_ident_set_items(const tr_ident_set *s, tr_ident_item *out, int32_t capacity) {
    if (!set_valid(s) || (s->ncand && !out) || capacity < s->ncand) return TR_MATH_INVALID;
    memcpy(out, s->items, (size_t)s->ncand * sizeof *out);
    return s->ncand;
}
int32_t tr_ident_set_get_view(const tr_ident_set *s, tr_ident_cand_view *v) {
    if (!set_valid(s) || !v) return TR_MATH_INVALID;
    memset(v, 0, sizeof *v);
    v->ncand = s->ncand; v->nvars = s->k;
    for (int32_t i = 0; i < TR_IDENT_MAX_VARS; i++) v->var[i] = s->var[i];
    v->nterms = s->nterms; v->offset = s->offset; v->support = s->support; v->words = s->words;
    v->exponents = s->exps; v->coefficients = s->coefs;
    return TR_MATH_OK;
}
int32_t tr_ident_set_poly(const tr_ident_set *s, int32_t i, uint8_t *e, int64_t *c, uint64_t capacity, uint64_t *nterms) {
    if (!set_valid(s) || i < 0 || i >= s->ncand || !nterms) return TR_MATH_INVALID;
    uint64_t n = s->offset[i + 1] - s->offset[i];
    *nterms = n;
    if (capacity < n || (n && (!e || !c))) return TR_MATH_INVALID;
    memcpy(e, s->exps + s->offset[i] * s->k, (size_t)(n * s->k));
    memcpy(c, s->coefs + s->offset[i], (size_t)(n * 8));
    return TR_MATH_OK;
}
int32_t tr_ident_claim(const tr_ident_set *s, int32_t i, tr_cert_claim *out) {
    if (!set_valid(s) || i < 0 || i >= s->ncand || !out) return TR_MATH_INVALID;
    const tr_ident_item *it = &s->items[i];
    if (!it->representable || !it->resolved) return TR_MATH_UNSUPPORTED;
    memset(out, 0, sizeof *out);
    out->kind = TR_CLAIM_POLYNOMIAL; out->policy = TR_CERT_NULL_SKIP; out->carrier = TR_CARRIER_I128;
    int32_t c = 0;
    for (int32_t q = 0; q < s->k; q++) if ((it->support >> q) & 1) out->col[c++] = s->var[q];
    for (int32_t q = c; q < TR_CERT_MAX_COLS; q++) out->col[q] = -1;
    out->ncols = c;
    out->nterms = s->offset[i + 1] - s->offset[i];
    out->exponents = s->cexp + s->coff[i];
    out->coefficients = s->ccoef + s->offset[i];
    return TR_MATH_OK;
}

/* ================================================================ the quadratic form */
int32_t tr_ident_quadratic(const tr_ident_gram *g, const tr_ident_set *s, int32_t i, tr_ident_quad *out, uint64_t *limbs,
                           uint64_t capacity) {
    if (!gram_valid(g) || !set_valid(s) || i < 0 || i >= s->ncand || !out) return TR_MATH_INVALID;
    const tr_ident_plan *p = g->plan;
    if (p->k != s->k || memcmp(p->var, s->var, sizeof p->var)) return TR_MATH_DOMAIN_ERROR;
    if (!s->items[i].representable || !s->items[i].resolved) return TR_MATH_UNSUPPORTED;
    int64_t co[TR_IDENT_MAX_BASIS];
    uint64_t t0 = s->offset[i], nt = s->offset[i + 1] - t0;
    int32_t rc = basis_coefs(p, s->k, s->exps + t0 * s->k, s->coefs + t0, nt, co);
    if (rc) return rc;
    memset(out, 0, sizeof *out);
    const int32_t nb = p->nb;
    if (p->repr == TR_IDENT_MODULAR) {
        for (int32_t q = 0; q < p->nprimes; q++) {
            const tr_gfp *f = &p->primes[q];
            uint64_t acc = 0;
            for (int32_t a = 0; a < nb; a++) {
                if (!co[a]) continue;
                uint64_t h = 0;
                for (int32_t b = 0; b < nb; b++)
                    if (co[b]) h = tr_gfp_add_hd(f, h, tr_gfp_mul_hd(f, g->res[(uint64_t)p->map[(uint64_t)a * nb + b] * p->nprimes + q],
                                                                     tr_gfp_reduce_i64_hd(f, co[b])));
                acc = tr_gfp_add_hd(f, acc, tr_gfp_mul_hd(f, h, tr_gfp_reduce_i64_hd(f, co[a])));
            }
            out->residues[q] = acc;
        }
        out->sos = TR_IDENT_SOS_MODULAR;
        out->numeric = NAN; out->scale = NAN; out->relative = NAN;
        return TR_MATH_OK;
    }
    /* exact */
    ictx x = {NULL, NULL, 0};
    tr_bigint *big[TR_IDENT_MAX_BASIS];
    memset(big, 0, sizeof big);
    for (int32_t b = 0; b < nb; b++) big[b] = Z_i64(&x, co[b]);
    tr_bigint *q = LIVE(&x) ? quad_exact(&x, g, big) : NULL;
    for (int32_t b = 0; b < nb; b++) tr_bigint_free(big[b]);
    if (!q) return x.rc ? x.rc : TR_MATH_MEMORY;
    out->exact = 1;
    out->sign = tr_bigint_sign(q);
    out->sos = sos_verdict(g, out->sign == 0);
    uint64_t cnt = tr_bigint_limb_count(q);
    int32_t sg = 0;
    out->nlimbs = (int32_t)cnt;
    rc = TR_MATH_OK;
    if (limbs) rc = capacity < cnt ? TR_MATH_INVALID : tr_bigint_export(q, &sg, limbs, capacity, &cnt);
    tr_bigint_free(q);
    if (rc) return rc;
    /* binary64: the entries rounded to doubles, a fixed order (rows, then columns) */
    double value = 0.0, scale = 0.0;
    for (int32_t a = 0; a < nb; a++) {
        if (!co[a]) continue;
        double h = 0.0, hs = 0.0;
        for (int32_t b = 0; b < nb; b++) {
            if (!co[b]) continue;
            double gab = (double)g->exact[p->map[(uint64_t)a * nb + b]];
            h += gab * (double)co[b];
            hs += fabs(gab) * fabs((double)co[b]);
        }
        value += (double)co[a] * h;
        scale += fabs((double)co[a]) * hs;
    }
    out->numeric = value; out->scale = scale;
    out->relative = scale > 0 ? fabs(value) / scale : (value == 0 ? 0.0 : INFINITY);
    return TR_MATH_OK;
}

/* ================================================================ verification and certificates */
static int cand_ok(const tr_scope_view *v, const tr_ident_cand_view *c) {
    if (!c || c->ncand < 0 || c->nvars < 1 || c->nvars > TR_IDENT_MAX_VARS || (c->ncand && (!c->offset || !c->support || !c->words)) ||
        (c->nterms && (!c->exponents || !c->coefficients))) return 0;
    for (int32_t i = 0; i < c->nvars; i++) if (!column_ok(v, c->var[i])) return 0;
    if (c->ncand && (c->offset[0] != 0 || c->offset[c->ncand] != c->nterms)) return 0;
    for (int32_t j = 0; j < c->ncand; j++) {
        if (c->offset[j] > c->offset[j + 1] || (c->nvars < 32 && (c->support[j] >> c->nvars)) || !c->support[j] ||
            (c->words[j] != 0 && (c->words[j] < 2 || c->words[j] > TR_IDENT_MAX_WORDS))) return 0;
    }
    return 1;
}
int32_t tr_ident_verify_need(const tr_scope_view *v, const tr_ident_cand_view *c, int32_t target, tr_math_need *need) {
    if (!view_ok(v) || !need || !cand_ok(v, c)) return TR_MATH_INVALID;
    if (target == TR_IDENT_NEED_DEVICE) { tr_ident_verify_device_need(v, c, need); return TR_MATH_OK; }
    if (target != TR_IDENT_NEED_CPU) return TR_MATH_INVALID;
    need_zero(need);
    need->need[TR_BUDGET_VARIABLES] = (uint64_t)c->nvars;
    need->need[TR_BUDGET_SUPPORT] = c->nterms;
    need->need[TR_BUDGET_OUTPUT] = (uint64_t)c->ncand;
    return TR_MATH_OK;
}
int32_t tr_ident_verify_view(const tr_scope_view *v, const tr_ident_cand_view *c, const tr_math_budget *budget,
                             tr_cert_evidence *out, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!view_ok(v) || !cand_ok(v, c) || (c->ncand && !out)) return TR_MATH_INVALID;
    tr_math_need need;
    tr_ident_verify_need(v, c, TR_IDENT_NEED_CPU, &need);
    int32_t rc = check_need(budget, &need, refusal);
    if (rc) return rc;
    for (int32_t j = 0; j < c->ncand; j++) {
        memset(&out[j], 0, sizeof out[j]);
        out[j].rows = v->rows; out[j].selected = v->selected; out[j].witness = TR_CERT_NONE; out[j].unit = TR_CERT_UNIT_ROWS;
    }
    const uint64_t nw = (v->rows + 63) / 64;
    for (uint64_t w = 0; w < nw; w++) {
        uint64_t bits = v->selection[w];
        while (bits) {
            uint64_t r = w * 64 + (uint64_t)__builtin_ctzll(bits);
            bits &= bits - 1;
            for (int32_t j = 0; j < c->ncand; j++) {
                int st = tr_ident_cand_row(v, c, j, r);
                tr_cert_evidence *e = &out[j];
                if (st == TR_CERT_ROW_EXCLUDED) { e->excluded++; continue; }
                if (st == TR_CERT_ROW_OVERFLOW) { e->overflow++; continue; }
                e->checked++;
                if (st == TR_CERT_ROW_VIOLATES) { e->violations++; if (r < e->witness) e->witness = r; }
            }
        }
    }
    for (int32_t j = 0; j < c->ncand; j++) out[j].holds = !out[j].overflow && !out[j].violations;
    return TR_MATH_OK;
}
int32_t tr_ident_certify(const tr_ident_set *s, const tr_scope *scope, const tr_cert_evidence *evidence, int32_t backend,
                         const tr_math_budget *budget, int32_t *status, tr_cert **certs, int32_t *codes,
                         tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    tr_scope_view v;
    if (!set_valid(s) || !scope || tr_scope_get_view(scope, &v) || (s->ncand && (!status || !certs || !codes)) ||
        (backend != TR_CERT_BY_CPU && backend != TR_CERT_BY_DEVICE)) return TR_MATH_INVALID;
    tr_ident_cand_view cv;
    tr_ident_set_get_view(s, &cv);
    if (!cand_ok(&v, &cv)) return TR_MATH_INVALID;
    for (int32_t i = 0; i < s->ncand; i++) { status[i] = TR_IDENT_PENDING; certs[i] = NULL; codes[i] = 0; }
    tr_cert_evidence *own = NULL;
    if (!evidence) {
        own = (tr_cert_evidence *)zalloc((uint64_t)(s->ncand ? s->ncand : 1) * sizeof *own);
        if (!own) return TR_MATH_MEMORY;
        int32_t rc = tr_ident_verify_view(&v, &cv, budget, own, refusal);
        if (rc) { free(own); return rc; }
        evidence = own;
        backend = TR_CERT_BY_CPU;
    }
    int32_t hard = TR_MATH_OK;
    for (int32_t i = 0; i < s->ncand && !hard; i++) {
        const tr_ident_item *it = &s->items[i];
        const tr_cert_evidence *e = &evidence[i];
        if (e->rows != v.rows || e->selected != v.selected || e->unit != TR_CERT_UNIT_ROWS ||
            tr_budget_add(tr_budget_add(e->checked, e->excluded), e->overflow) != v.selected) { hard = TR_MATH_INVALID; break; }
        if (!it->resolved) { status[i] = TR_IDENT_UNRESOLVED; codes[i] = TR_MATH_NOT_CONVERGED; continue; }
        if (!it->representable) { status[i] = TR_IDENT_UNREPRESENTABLE; codes[i] = TR_MATH_UNSUPPORTED; continue; }
        if (e->violations) { status[i] = TR_IDENT_REJECTED; codes[i] = TR_MATH_DOMAIN_ERROR; continue; }
        if (!it->words || e->overflow) { status[i] = TR_IDENT_UNDECIDED; codes[i] = TR_MATH_OVERFLOW; continue; }
        status[i] = TR_IDENT_VERIFIED;
        tr_cert_claim claim;
        tr_ident_claim(s, i, &claim);
        int32_t err = 0;
        tr_cert *c;
        if (it->words == 2) c = tr_cert_adopt(scope, &claim, e, backend, NULL, 0, budget, refusal, &err);
        else {                                                 /* FND-007 checks it itself in 128 bits */
            tr_cert_evidence fe;
            c = tr_cert_snapshot(scope, &claim, NULL, 0, &fe, budget, refusal, &err);
        }
        certs[i] = c;
        if (!c) {
            if (err == TR_MATH_OVERFLOW || err == TR_MATH_DOMAIN_ERROR || err == TR_MATH_UNSUPPORTED) codes[i] = err;
            else hard = err ? err : TR_MATH_MEMORY;
        }
    }
    free(own);
    if (hard) {
        for (int32_t i = 0; i < s->ncand; i++) { tr_cert_free(certs[i]); certs[i] = NULL; status[i] = TR_IDENT_PENDING; codes[i] = 0; }
        return hard;
    }
    return TR_MATH_OK;
}
tr_cert *tr_ident_hypothesis(const tr_scope *scope, const tr_ident_gram *g, const tr_ident_set *s, int32_t i, double tolerance,
                             const tr_cert_sample *sample, tr_ident_quad *quad, tr_cert_evidence *evidence,
                             const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    tr_ident_quad q;
    if (!scope || !sample || !(tolerance >= 0)) { set_error(error, TR_MATH_INVALID); return NULL; }
    int32_t rc = tr_ident_quadratic(g, s, i, &q, NULL, 0);
    if (rc) { set_error(error, rc); return NULL; }
    if (quad) *quad = q;
    if (!(q.relative <= tolerance)) { set_error(error, TR_MATH_DOMAIN_ERROR); return NULL; }   /* not near zero (or NaN) */
    tr_cert_claim claim;
    rc = tr_ident_claim(s, i, &claim);
    if (rc) { set_error(error, rc); return NULL; }
    return tr_cert_hypothesis(scope, &claim, sample, NULL, 0, evidence, budget, refusal, error);
}
