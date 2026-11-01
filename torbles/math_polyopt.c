/* OPT-002: bounded convex and polynomial optimization; see math_polyopt.h.
 * SPDX-License-Identifier: Apache-2.0 */
#include "math_polyopt.h"
#include "math_inv.h"
#include "math_lin.h"
#include "math_handles.h"
#include <stdlib.h>
#include <string.h>
#include <time.h>

TR_MATH_ABI(tr_polyopt_info, 1);         /* the ctypes boundary of OPT-002 (math_handles.h) */
TR_MATH_ABI(tr_polyopt_spec, 1);
TR_MATH_ABI(tr_polyopt_encl_view, 1);
TR_MATH_ABI(tr_polyopt_box_out, 1);

_Static_assert(sizeof(tr_polyopt_info) == 128, "tr_polyopt_info ABI");
_Static_assert(sizeof(tr_polyopt_spec) == 40, "tr_polyopt_spec ABI");
_Static_assert(sizeof(tr_polyopt_encl_view) == 560, "tr_polyopt_encl_view ABI");
_Static_assert(sizeof(tr_polyopt_box_out) == 32, "tr_polyopt_box_out ABI");

#define QP_MAGIC UINT32_C(0x5051504f)
#define PP_MAGIC UINT32_C(0x5050504f)
#define PR_MAGIC UINT32_C(0x5250504f)

static void set_error(int32_t *error, int32_t code) { if (error) *error = code; }
static void *zalloc(uint64_t bytes) {
    if (bytes > (uint64_t)SIZE_MAX / 2) return NULL;
    return calloc(1, bytes ? (size_t)bytes : 1);
}

/* ================================================================ exact arithmetic with a sticky error */
typedef struct {
    const tr_math_budget *b;
    tr_math_refusal *r;
    int32_t rc, reserved;
    uint64_t iters, limit;                   /* pivots / steps / boxes and their budget */
    uint64_t lps;
} pctx;
#define LIVE(c) (!(c)->rc)
static void pfail(pctx *c, int32_t rc) { if (!c->rc) c->rc = rc ? rc : TR_MATH_MEMORY; }
static void pctx_init(pctx *c, const tr_math_budget *b, tr_math_refusal *r) {
    c->b = b; c->r = r; c->rc = 0; c->reserved = 0; c->iters = 0; c->lps = 0;
    c->limit = b ? b->limit[TR_BUDGET_ITERATIONS] : ((uint64_t)1 << 20);
}
/* one more step within the iteration budget (a refusal reports the need) */
static int tick(pctx *c) {
    if (c->rc) return 0;
    if (c->iters >= c->limit) {
        if (c->r) { c->r->resource = TR_BUDGET_ITERATIONS; c->r->refused = 1; c->r->need = c->iters + 1; c->r->limit = c->limit; }
        pfail(c, TR_MATH_BUDGET);
        return 0;
    }
    c->iters++;
    return 1;
}
static tr_bigq *qk(pctx *c, tr_bigq *v, int32_t e) { if (!v) pfail(c, e); return v; }
static tr_bigq *Q_int(pctx *c, int64_t v) {
    tr_num_rational q; q.numerator = v; q.denominator = 1;
    int32_t e = 0;
    return c->rc ? NULL : qk(c, tr_bigq_from_q64(q, c->b, c->r, &e), e);
}
static tr_bigq *Q_q64(pctx *c, int64_t n, int64_t d) {
    tr_num_rational q; q.numerator = n; q.denominator = d;
    int32_t e = 0;
    return c->rc ? NULL : qk(c, tr_bigq_from_q64(q, c->b, c->r, &e), e);
}
static tr_bigq *Q_copy(pctx *c, const tr_bigq *a) {
    int32_t e = 0;
    if (c->rc) return NULL;
    if (!a) return Q_int(c, 0);
    return qk(c, tr_bigq_copy(a, c->b, c->r, &e), e);
}
static int Q_zero(const tr_bigq *a) { return !a || tr_bigq_sign(a) == 0; }
static int Q_sgn(const tr_bigq *a) { return a ? tr_bigq_sign(a) : 0; }
static int Q_cmp(const tr_bigq *a, const tr_bigq *b) {      /* NULL = 0 */
    if (!a && !b) return 0;
    if (!a) return -tr_bigq_sign(b);
    if (!b) return tr_bigq_sign(a);
    return tr_bigq_cmp(a, b);
}
static tr_bigq *Q_add(pctx *c, const tr_bigq *a, const tr_bigq *b) {
    int32_t e = 0;
    if (c->rc) return NULL;
    if (Q_zero(a)) return Q_copy(c, b);
    if (Q_zero(b)) return Q_copy(c, a);
    return qk(c, tr_bigq_add(a, b, c->b, c->r, &e), e);
}
static tr_bigq *Q_sub(pctx *c, const tr_bigq *a, const tr_bigq *b) {
    int32_t e = 0;
    if (c->rc) return NULL;
    if (Q_zero(b)) return Q_copy(c, a);
    if (Q_zero(a)) return qk(c, tr_bigq_neg(b, c->b, c->r, &e), e);
    return qk(c, tr_bigq_sub(a, b, c->b, c->r, &e), e);
}
static tr_bigq *Q_mul(pctx *c, const tr_bigq *a, const tr_bigq *b) {
    int32_t e = 0;
    if (c->rc) return NULL;
    if (Q_zero(a) || Q_zero(b)) return Q_int(c, 0);
    return qk(c, tr_bigq_mul(a, b, c->b, c->r, &e), e);
}
static tr_bigq *Q_div(pctx *c, const tr_bigq *a, const tr_bigq *b) {
    int32_t e = 0;
    if (c->rc) return NULL;
    if (Q_zero(b)) { pfail(c, TR_MATH_INVALID); return NULL; }
    if (Q_zero(a)) return Q_int(c, 0);
    return qk(c, tr_bigq_div(a, b, c->b, c->r, &e), e);
}
static tr_bigq *Q_neg(pctx *c, const tr_bigq *a) {
    int32_t e = 0;
    if (c->rc) return NULL;
    if (Q_zero(a)) return Q_int(c, 0);
    return qk(c, tr_bigq_neg(a, c->b, c->r, &e), e);
}
static void Q_set(tr_bigq **slot, tr_bigq *v) { if (v) { tr_bigq_free(*slot); *slot = v; } }
/* *acc += a b (acc may be NULL = 0); a zero result is stored as NULL */
static void Q_fma(pctx *c, tr_bigq **acc, const tr_bigq *a, const tr_bigq *b) {
    if (c->rc || Q_zero(a) || Q_zero(b)) return;
    tr_bigq *p = Q_mul(c, a, b);
    if (!p) return;
    if (!*acc) { *acc = p; return; }
    tr_bigq *s = Q_add(c, *acc, p);
    tr_bigq_free(p);
    if (!s) return;
    tr_bigq_free(*acc);
    if (Q_zero(s)) { tr_bigq_free(s); *acc = NULL; } else *acc = s;
}
/* *acc -= a b */
static void Q_fms(pctx *c, tr_bigq **acc, const tr_bigq *a, const tr_bigq *b) {
    if (c->rc || Q_zero(a) || Q_zero(b)) return;
    tr_bigq *p = Q_mul(c, a, b);
    if (!p) return;
    tr_bigq *s = *acc ? Q_sub(c, *acc, p) : Q_neg(c, p);
    tr_bigq_free(p);
    if (!s) return;
    tr_bigq_free(*acc);
    if (Q_zero(s)) { tr_bigq_free(s); *acc = NULL; } else *acc = s;
}
static tr_bigq **Q_arr(pctx *c, uint64_t n) {
    if (c->rc) return NULL;
    tr_bigq **v = (tr_bigq **)zalloc(n * sizeof *v);
    if (!v) pfail(c, TR_MATH_MEMORY);
    return v;
}
static void Q_free(tr_bigq **v, uint64_t n) {
    if (!v) return;
    for (uint64_t i = 0; i < n; i++) tr_bigq_free(v[i]);
    free(v);
}
/* a dense owned copy (NULL entries become 0) */
static tr_bigq **Q_full(pctx *c, const tr_bigq *const *v, uint64_t n) {
    tr_bigq **o = Q_arr(c, n);
    for (uint64_t i = 0; o && i < n && LIVE(c); i++) o[i] = Q_copy(c, v ? v[i] : NULL);
    return o;
}
static tr_bigq *Q_dot(pctx *c, const tr_bigq *const *a, const tr_bigq *const *b, int32_t n) {
    tr_bigq *acc = NULL;
    for (int32_t i = 0; i < n && LIVE(c); i++) Q_fma(c, &acc, a[i], b[i]);
    return acc ? acc : Q_int(c, 0);
}
/* a rational from an exact double (dyadic) */
static tr_bigq *Q_double(pctx *c, double x) {
    int32_t e = 0;
    if (c->rc) return NULL;
    return qk(c, tr_bigq_from_double(x, c->b, c->r, &e), e);
}
static int budget_need(const tr_math_budget *b, tr_math_refusal *r, const tr_math_need *need) {
    tr_math_budget dflt;
    if (!b) { tr_budget_default(&dflt); b = &dflt; }
    return tr_budget_check(b, need, r);
}

/* ================================================================ results */
struct tr_polyopt_result_s {
    uint32_t magic;
    tr_polyopt_info info;
    tr_bigq **part[TR_POLYOPT_PARTS];
    uint64_t npart[TR_POLYOPT_PARTS];
    int32_t *active;
    int32_t nactive, reserved;
    uint64_t nleaves;
    tr_bigq **leaf_lo, **leaf_hi;            /* nleaves x n */
    int32_t *leaf_con;
};
static int res_valid(const tr_polyopt_result *r) { return r && r->magic == PR_MAGIC; }
void tr_polyopt_result_free(tr_polyopt_result *r) {
    if (!res_valid(r)) return;
    for (int p = 0; p < TR_POLYOPT_PARTS; p++) Q_free(r->part[p], r->npart[p]);
    Q_free(r->leaf_lo, r->nleaves * (uint64_t)r->info.n);
    Q_free(r->leaf_hi, r->nleaves * (uint64_t)r->info.n);
    free(r->leaf_con);
    free(r->active);
    r->magic = 0;
    free(r);
}
static tr_polyopt_result *res_new(pctx *c, int32_t n) {
    if (c->rc) return NULL;
    tr_polyopt_result *r = (tr_polyopt_result *)zalloc(sizeof *r);
    if (!r) { pfail(c, TR_MATH_MEMORY); return NULL; }
    r->magic = PR_MAGIC;
    r->info.n = n;
    r->info.unique = -1;
    return r;
}
/* stores v (owned, n values; NULL entries become 0) as a part */
static void res_set(pctx *c, tr_polyopt_result *r, int32_t part, tr_bigq **v, uint64_t n) {
    if (!r || !v) { Q_free(v, v ? n : 0); return; }
    for (uint64_t i = 0; i < n && LIVE(c); i++) if (!v[i]) v[i] = Q_int(c, 0);
    Q_free(r->part[part], r->npart[part]);
    r->part[part] = v; r->npart[part] = n;
}
static void res_one(pctx *c, tr_polyopt_result *r, int32_t part, const tr_bigq *v) {
    tr_bigq **a = Q_arr(c, 1);
    if (a) a[0] = Q_copy(c, v);
    res_set(c, r, part, a, 1);
}
static tr_polyopt_result *res_done(pctx *c, tr_polyopt_result *r, int32_t *error) {
    if (c->rc) { tr_polyopt_result_free(r); set_error(error, c->rc); return NULL; }
    if (!r) { set_error(error, TR_MATH_MEMORY); return NULL; }
    r->info.lps = c->lps;
    set_error(error, TR_MATH_OK);
    return r;
}
int32_t tr_polyopt_result_get_info(const tr_polyopt_result *r, tr_polyopt_info *out) {
    if (!res_valid(r) || !out) return TR_MATH_INVALID;
    *out = r->info;
    return TR_MATH_OK;
}
const tr_bigq *const *tr_polyopt_result_part(const tr_polyopt_result *r, int32_t part, uint64_t *count) {
    if (count) *count = 0;
    if (!res_valid(r) || part < 0 || part >= TR_POLYOPT_PARTS || !r->part[part]) return NULL;
    if (count) *count = r->npart[part];
    return (const tr_bigq *const *)r->part[part];
}
const tr_bigq *tr_polyopt_result_value(const tr_polyopt_result *r, int32_t part, uint64_t i) {
    if (!res_valid(r) || part < 0 || part >= TR_POLYOPT_PARTS || !r->part[part] || i >= r->npart[part]) return NULL;
    return r->part[part][i];
}
int32_t tr_polyopt_result_leaf(const tr_polyopt_result *r, uint64_t i, const tr_bigq **lo, const tr_bigq **hi,
                               int32_t *constraint) {
    if (!res_valid(r) || i >= r->nleaves || !lo || !hi) return TR_MATH_INVALID;
    for (int32_t v = 0; v < r->info.n; v++) {
        lo[v] = r->leaf_lo[i * (uint64_t)r->info.n + v];
        hi[v] = r->leaf_hi[i * (uint64_t)r->info.n + v];
    }
    if (constraint) *constraint = r->leaf_con[i];
    return TR_MATH_OK;
}
int32_t tr_polyopt_result_active(const tr_polyopt_result *r, int32_t *rows, int32_t capacity) {
    if (!res_valid(r)) return TR_MATH_INVALID;
    for (int32_t i = 0; rows && i < r->nactive && i < capacity; i++) rows[i] = r->active[i];
    return r->nactive;
}

/* ================================================================ the exact simplex method */
/* min cost^T x subject to A x <= b (mi rows) and E x = e (p rows), x in R^n free (cost NULL: a feasible vertex).
 * Tableau over x = u - v (u, v >= 0), slacks s (A u - A v + s = b) and one artificial per row; a row with a negative
 * right-hand side is negated (sigma = -1). Bland's rule (smallest entering column, smallest leaving basic index)
 * prevents cycling. Entries NULL = 0. */
typedef struct {
    int32_t status;                          /* TR_POLYOPT_OPTIMAL / INFEASIBLE / UNBOUNDED */
    int32_t strict;                          /* optimal: every nonbasic reduced cost > 0 (a unique optimum) */
    tr_bigq **x;                             /* n: the optimum, the anchor of a ray, or a feasible vertex */
    tr_bigq **lambda, **mu;                  /* mi, p multipliers (optimal) */
    tr_bigq **y, **z;                        /* mi, p Farkas certificate (infeasible) */
    tr_bigq **ray;                           /* n (unbounded) */
    int32_t *active;                         /* mi flags: slack nonbasic at 0 (optimal / feasible vertex) */
} lp_out;
static void lp_out_free(lp_out *o, int32_t n, int32_t mi, int32_t p) {
    Q_free(o->x, (uint64_t)n); Q_free(o->lambda, (uint64_t)mi); Q_free(o->mu, (uint64_t)p);
    Q_free(o->y, (uint64_t)mi); Q_free(o->z, (uint64_t)p); Q_free(o->ray, (uint64_t)n);
    free(o->active);
    memset(o, 0, sizeof *o);
}

typedef struct {
    int32_t R, ncol, art;                    /* rows, columns (the RHS is column ncol), first artificial column */
    tr_bigq ***T;                            /* R rows of ncol + 1 entries */
    tr_bigq **d;                             /* reduced costs, ncol + 1 (d[ncol] = -objective) */
    int32_t *basis, *init;
    int8_t *sigma;
} tab_t;
static void tab_free(tab_t *t) {
    if (t->T) for (int32_t r = 0; r < t->R; r++) Q_free(t->T[r], (uint64_t)t->ncol + 1);
    free(t->T);
    Q_free(t->d, (uint64_t)t->ncol + 1);
    free(t->basis); free(t->init); free(t->sigma);
    memset(t, 0, sizeof *t);
}
static void tab_pivot(pctx *c, tab_t *t, int32_t r, int32_t j) {
    tr_bigq **row = t->T[r];
    tr_bigq *inv = NULL;
    {
        tr_bigq *one = Q_int(c, 1);
        inv = Q_div(c, one, row[j]);
        tr_bigq_free(one);
    }
    for (int32_t k = 0; k <= t->ncol && LIVE(c); k++)
        if (!Q_zero(row[k])) Q_set(&row[k], Q_mul(c, row[k], inv));
    tr_bigq_free(inv);
    for (int32_t i = 0; i < t->R && LIVE(c); i++) {
        if (i == r || Q_zero(t->T[i][j])) continue;
        tr_bigq *f = Q_copy(c, t->T[i][j]);
        for (int32_t k = 0; k <= t->ncol && LIVE(c); k++) if (!Q_zero(row[k])) Q_fms(c, &t->T[i][k], f, row[k]);
        tr_bigq_free(f);
        if (!Q_zero(t->T[i][j])) pfail(c, TR_MATH_INVALID);
    }
    if (LIVE(c) && !Q_zero(t->d[j])) {
        tr_bigq *f = Q_copy(c, t->d[j]);
        for (int32_t k = 0; k <= t->ncol && LIVE(c); k++) if (!Q_zero(row[k])) Q_fms(c, &t->d[k], f, row[k]);
        tr_bigq_free(f);
    }
    t->basis[r] = j;
}
/* the reduced costs of the cost vector cc (ncol entries, NULL = 0) for the current basis */
static void tab_costs(pctx *c, tab_t *t, tr_bigq *const *cc) {
    for (int32_t k = 0; k <= t->ncol && LIVE(c); k++) {
        tr_bigq_free(t->d[k]);
        t->d[k] = (k < t->ncol && !Q_zero(cc[k])) ? Q_copy(c, cc[k]) : NULL;
    }
    for (int32_t r = 0; r < t->R && LIVE(c); r++) {
        const tr_bigq *cb = cc[t->basis[r]];
        if (Q_zero(cb)) continue;
        for (int32_t k = 0; k <= t->ncol && LIVE(c); k++) Q_fms(c, &t->d[k], cb, t->T[r][k]);
    }
}
/* Bland iterations until optimal (0) or unbounded (1, *entering set) */
static int tab_iterate(pctx *c, tab_t *t, int32_t *entering) {
    while (LIVE(c)) {
        int32_t j = -1;
        for (int32_t k = 0; k < t->art; k++) if (Q_sgn(t->d[k]) < 0) { j = k; break; }
        if (j < 0) return 0;
        int32_t best = -1;
        for (int32_t r = 0; r < t->R && LIVE(c); r++) {
            if (Q_sgn(t->T[r][j]) <= 0) continue;
            if (best < 0) { best = r; continue; }
            /* rhs_r / a_r < rhs_b / a_b  <=>  rhs_r a_b < rhs_b a_r (a > 0) */
            tr_bigq *l = Q_mul(c, t->T[r][t->ncol], t->T[best][j]), *h = Q_mul(c, t->T[best][t->ncol], t->T[r][j]);
            int cmp = LIVE(c) ? Q_cmp(l, h) : 0;
            tr_bigq_free(l); tr_bigq_free(h);
            if (cmp < 0 || (cmp == 0 && t->basis[r] < t->basis[best])) best = r;
        }
        if (!LIVE(c)) return 0;
        if (best < 0) { *entering = j; return 1; }
        if (!tick(c)) return 0;
        tab_pivot(c, t, best, j);
    }
    return 0;
}
static tr_bigq *tab_value(const tab_t *t, int32_t col) {     /* borrowed: the value of a basic column, NULL = 0 */
    for (int32_t r = 0; r < t->R; r++) if (t->basis[r] == col) return t->T[r][t->ncol];
    return NULL;
}
static void lp_core(pctx *c, int32_t n, int32_t mi, const tr_bigq *const *A, const tr_bigq *const *b, int32_t p,
                    const tr_bigq *const *E, const tr_bigq *const *e, const tr_bigq *const *cost, lp_out *o) {
    memset(o, 0, sizeof *o);
    if (c->rc) return;
    c->lps++;
    tab_t t;
    memset(&t, 0, sizeof t);
    t.R = mi + p;
    t.art = 2 * n + mi;
    t.ncol = t.art + t.R;
    tr_math_need need;
    memset(&need, 0, sizeof need);
    need.need[TR_BUDGET_VARIABLES] = (uint64_t)n;
    need.need[TR_BUDGET_TENSOR] = tr_budget_mul((uint64_t)t.R + 1, (uint64_t)t.ncol + 1);
    need.need[TR_BUDGET_BYTES] = tr_budget_mul(need.need[TR_BUDGET_TENSOR], 8);
    int32_t rc = budget_need(c->b, c->r, &need);
    if (rc) { pfail(c, rc); return; }
    t.T = (tr_bigq ***)zalloc((uint64_t)(t.R ? t.R : 1) * sizeof *t.T);
    t.d = Q_arr(c, (uint64_t)t.ncol + 1);
    t.basis = (int32_t *)zalloc((uint64_t)(t.R ? t.R : 1) * 4);
    t.init = (int32_t *)zalloc((uint64_t)(t.R ? t.R : 1) * 4);
    t.sigma = (int8_t *)zalloc((uint64_t)(t.R ? t.R : 1));
    if (!t.T || !t.d || !t.basis || !t.init || !t.sigma) { pfail(c, TR_MATH_MEMORY); tab_free(&t); return; }
    for (int32_t r = 0; r < t.R && LIVE(c); r++) t.T[r] = Q_arr(c, (uint64_t)t.ncol + 1);
    int need_phase1 = 0;
    for (int32_t r = 0; r < t.R && LIVE(c); r++) {
        int eq = r >= mi;
        const tr_bigq *const *row = eq ? E + (uint64_t)(r - mi) * n : A + (uint64_t)r * n;
        const tr_bigq *rhs = eq ? e[r - mi] : b[r];
        int8_t s = Q_sgn(rhs) < 0 ? -1 : 1;
        t.sigma[r] = s;
        for (int32_t j = 0; j < n && LIVE(c); j++) {
            if (Q_zero(row[j])) continue;
            t.T[r][j] = s > 0 ? Q_copy(c, row[j]) : Q_neg(c, row[j]);
            t.T[r][n + j] = s > 0 ? Q_neg(c, row[j]) : Q_copy(c, row[j]);
        }
        if (!eq) t.T[r][2 * n + r] = Q_int(c, s);
        if (!Q_zero(rhs)) t.T[r][t.ncol] = s > 0 ? Q_copy(c, rhs) : Q_neg(c, rhs);
        if (!eq && s > 0) { t.basis[r] = 2 * n + r; t.init[r] = 2 * n + r; }
        else { t.T[r][t.art + r] = Q_int(c, 1); t.basis[r] = t.art + r; t.init[r] = t.art + r; need_phase1 = 1; }
    }
    tr_bigq **cc = Q_arr(c, (uint64_t)t.ncol + 1);
    int32_t entering = -1;
    if (need_phase1 && LIVE(c)) {                               /* phase 1: minimize the artificials */
        for (int32_t r = 0; r < t.R && LIVE(c); r++) cc[t.art + r] = Q_int(c, 1);
        tab_costs(c, &t, cc);
        tab_iterate(c, &t, &entering);                          /* bounded below by 0: never unbounded */
        if (LIVE(c) && Q_sgn(t.d[t.ncol]) < 0) {                /* the minimum sum of artificials is > 0 */
            o->status = TR_POLYOPT_INFEASIBLE;
            o->y = Q_arr(c, (uint64_t)(mi ? mi : 1));
            o->z = Q_arr(c, (uint64_t)(p ? p : 1));
            for (int32_t r = 0; r < t.R && LIVE(c); r++) {
                /* pi_r = c(init_r) - d(init_r); the certificate is -sigma_r pi_r */
                tr_bigq *ci = Q_int(c, t.init[r] >= t.art ? 1 : 0), *pi = Q_sub(c, ci, t.d[t.init[r]]);
                tr_bigq *w = t.sigma[r] > 0 ? Q_neg(c, pi) : Q_copy(c, pi);
                tr_bigq_free(ci); tr_bigq_free(pi);
                if (r < mi) o->y[r] = w; else o->z[r - mi] = w;
            }
            Q_free(cc, (uint64_t)t.ncol + 1);
            tab_free(&t);
            return;
        }
        for (int32_t r = 0; r < t.R && LIVE(c); r++) {          /* drive basic artificials (at 0) out */
            if (t.basis[r] < t.art) continue;
            for (int32_t k = 0; k < t.art; k++)
                if (!Q_zero(t.T[r][k])) { if (tick(c)) tab_pivot(c, &t, r, k); break; }
        }
        for (int32_t r = 0; r < t.R && LIVE(c); r++) { tr_bigq_free(cc[t.art + r]); cc[t.art + r] = NULL; }
    }
    if (!LIVE(c)) { Q_free(cc, (uint64_t)t.ncol + 1); tab_free(&t); return; }
    int unbounded = 0;
    if (cost) {                                                 /* phase 2 */
        for (int32_t j = 0; j < n && LIVE(c); j++) {
            if (Q_zero(cost[j])) continue;
            cc[j] = Q_copy(c, cost[j]);
            cc[n + j] = Q_neg(c, cost[j]);
        }
        tab_costs(c, &t, cc);
        unbounded = tab_iterate(c, &t, &entering);
    }
    o->x = Q_arr(c, (uint64_t)(n ? n : 1));
    for (int32_t j = 0; j < n && LIVE(c); j++) o->x[j] = Q_sub(c, tab_value(&t, j), tab_value(&t, n + j));
    o->active = (int32_t *)zalloc((uint64_t)(mi ? mi : 1) * 4);
    if (!o->active) pfail(c, TR_MATH_MEMORY);
    for (int32_t i = 0; i < mi && LIVE(c); i++) o->active[i] = Q_zero(tab_value(&t, 2 * n + i));
    if (unbounded && LIVE(c)) {
        o->status = TR_POLYOPT_UNBOUNDED;
        o->ray = Q_arr(c, (uint64_t)(n ? n : 1));
        for (int32_t j = 0; j < n && LIVE(c); j++) o->ray[j] = Q_int(c, 0);
        if (entering < n) Q_set(&o->ray[entering], Q_int(c, 1));
        else if (entering < 2 * n) Q_set(&o->ray[entering - n], Q_int(c, -1));
        for (int32_t r = 0; r < t.R && LIVE(c); r++) {
            int32_t bc = t.basis[r];
            if (bc < n) Q_set(&o->ray[bc], Q_sub(c, o->ray[bc], t.T[r][entering]));
            else if (bc < 2 * n) Q_set(&o->ray[bc - n], Q_add(c, o->ray[bc - n], t.T[r][entering]));
        }
    } else if (LIVE(c)) {
        o->status = TR_POLYOPT_OPTIMAL;                         /* (a feasible vertex when cost is NULL) */
        if (cost) {
            o->lambda = Q_arr(c, (uint64_t)(mi ? mi : 1));
            o->mu = Q_arr(c, (uint64_t)(p ? p : 1));
            for (int32_t r = 0; r < t.R && LIVE(c); r++) {      /* pi_r = -d(init_r); multiplier -sigma_r pi_r */
                tr_bigq *w = t.sigma[r] > 0 ? Q_copy(c, t.d[t.init[r]]) : Q_neg(c, t.d[t.init[r]]);
                if (r < mi) o->lambda[r] = w; else o->mu[r - mi] = w;
            }
            /* every nonbasic column with a positive reduced cost: the optimum is unique (the mirror column of a basic
             * split variable moves no x and is skipped) */
            uint8_t *basic = (uint8_t *)zalloc((uint64_t)t.ncol + 1);
            int strict = basic != NULL;
            for (int32_t r = 0; basic && r < t.R; r++) basic[t.basis[r]] = 1;
            for (int32_t k = 0; strict && k < t.art; k++) {
                if (basic[k]) continue;
                if (k < 2 * n && basic[k < n ? k + n : k - n]) continue;
                if (Q_sgn(t.d[k]) <= 0) strict = 0;
            }
            free(basic);
            o->strict = strict;
        }
    }
    Q_free(cc, (uint64_t)t.ncol + 1);
    tab_free(&t);
}

/* ================================================================ convex problems */
struct tr_polyopt_qp_s {
    uint32_t magic;
    int32_t n, m, p, mi;                     /* user rows m; inequality rows mi = m + bound rows */
    int32_t hzero, reserved;
    int32_t *bvar;                           /* bound row i (m <= i < mi): its variable */
    int8_t *bside;                           /* -1 lower bound, +1 upper bound */
    tr_bigq **H, **g, *c0;                   /* n x n, n, 1 (dense, owned) */
    tr_bigq **A, **b;                        /* mi x n, mi */
    tr_bigq **E, **e;                        /* p x n, p */
};
static int qp_valid(const tr_polyopt_qp *q) { return q && q->magic == QP_MAGIC; }
void tr_polyopt_qp_free(tr_polyopt_qp *q) {
    if (!qp_valid(q)) return;
    const uint64_t n = (uint64_t)q->n;
    Q_free(q->H, n * n); Q_free(q->g, n); tr_bigq_free(q->c0);
    Q_free(q->A, (uint64_t)q->mi * n); Q_free(q->b, (uint64_t)q->mi);
    Q_free(q->E, (uint64_t)q->p * n); Q_free(q->e, (uint64_t)q->p);
    free(q->bvar); free(q->bside);
    q->magic = 0;
    free(q);
}
int32_t tr_polyopt_qp_dims(const tr_polyopt_qp *q, int32_t *n, int32_t *m, int32_t *p) {
    if (!qp_valid(q)) return TR_MATH_INVALID;
    if (n) *n = q->n;
    if (m) *m = q->m;
    if (p) *p = q->p;
    return TR_MATH_OK;
}
tr_polyopt_qp *tr_polyopt_qp_create(int32_t n, const tr_bigq *const *H, const tr_bigq *const *g, const tr_bigq *c0,
                                    int32_t m, const tr_bigq *const *A, const tr_bigq *const *b, int32_t p,
                                    const tr_bigq *const *E, const tr_bigq *const *e, const tr_bigq *const *lo,
                                    const tr_bigq *const *hi, const tr_math_budget *budget, tr_math_refusal *refusal,
                                    int32_t *error) {
    tr_budget_none(refusal);
    if (n < 1 || m < 0 || p < 0 || (m && (!A || !b)) || (p && (!E || !e))) { set_error(error, TR_MATH_INVALID); return NULL; }
    int32_t nb = 0;
    for (int32_t i = 0; i < n; i++) nb += (lo && lo[i]) + (hi && hi[i]);
    if (n > TR_POLYOPT_MAX_VARS || (int64_t)m + nb + p > TR_POLYOPT_MAX_ROWS) { set_error(error, TR_MATH_UNSUPPORTED); return NULL; }
    tr_math_need need;
    memset(&need, 0, sizeof need);
    need.need[TR_BUDGET_VARIABLES] = (uint64_t)n;
    need.need[TR_BUDGET_DEGREE] = 2;
    need.need[TR_BUDGET_TENSOR] = (uint64_t)(n + m + nb + p + 1) * (uint64_t)(n + 1);
    int32_t rc = budget_need(budget, refusal, &need);
    if (rc) { set_error(error, rc); return NULL; }
    pctx c;
    pctx_init(&c, budget, refusal);
    tr_polyopt_qp *q = (tr_polyopt_qp *)zalloc(sizeof *q);
    if (!q) { set_error(error, TR_MATH_MEMORY); return NULL; }
    q->magic = QP_MAGIC; q->n = n; q->m = m; q->p = p; q->mi = m + nb;
    q->H = Q_full(&c, H, (uint64_t)n * n);
    q->g = Q_full(&c, g, (uint64_t)n);
    q->c0 = Q_copy(&c, c0);
    q->A = Q_arr(&c, (uint64_t)(q->mi ? q->mi : 1) * n);
    q->b = Q_arr(&c, (uint64_t)(q->mi ? q->mi : 1));
    q->E = Q_full(&c, E, (uint64_t)p * n);
    q->e = Q_full(&c, e, (uint64_t)p);
    q->bvar = (int32_t *)zalloc((uint64_t)(nb ? nb : 1) * 4);
    q->bside = (int8_t *)zalloc((uint64_t)(nb ? nb : 1));
    if (!q->bvar || !q->bside) pfail(&c, TR_MATH_MEMORY);
    for (int32_t i = 0; i < m && LIVE(&c); i++) {
        for (int32_t j = 0; j < n && LIVE(&c); j++) q->A[(uint64_t)i * n + j] = Q_copy(&c, A[(uint64_t)i * n + j]);
        q->b[i] = Q_copy(&c, b[i]);
    }
    int32_t row = m;
    for (int32_t i = 0; i < n && LIVE(&c); i++)
        for (int32_t side = -1; side <= 1; side += 2) {
            const tr_bigq *bd = side < 0 ? (lo ? lo[i] : NULL) : (hi ? hi[i] : NULL);
            if (!bd) continue;
            for (int32_t j = 0; j < n && LIVE(&c); j++) q->A[(uint64_t)row * n + j] = Q_int(&c, j == i ? side : 0);
            q->b[row] = side < 0 ? Q_neg(&c, bd) : Q_copy(&c, bd);
            q->bvar[row - m] = i; q->bside[row - m] = (int8_t)side;
            row++;
        }
    q->hzero = 1;
    for (uint64_t k = 0; k < (uint64_t)n * n && LIVE(&c); k++) if (!Q_zero(q->H[k])) q->hzero = 0;
    for (int32_t i = 0; i < n && LIVE(&c); i++)                  /* H must be symmetric */
        for (int32_t j = i + 1; j < n; j++)
            if (Q_cmp(q->H[(uint64_t)i * n + j], q->H[(uint64_t)j * n + i])) pfail(&c, TR_MATH_INVALID);
    if (c.rc) { tr_polyopt_qp_free(q); set_error(error, c.rc); return NULL; }
    set_error(error, TR_MATH_OK);
    return q;
}
tr_polyopt_qp *tr_polyopt_qp_from_quad(const tr_opt_quad *quad, const tr_bigq *const *linear, int32_t m,
                                       const tr_bigq *const *A, const tr_bigq *const *b, int32_t p,
                                       const tr_bigq *const *E, const tr_bigq *const *e, const tr_bigq *const *lo,
                                       const tr_bigq *const *hi, const tr_math_budget *budget, tr_math_refusal *refusal,
                                       int32_t *error) {
    tr_budget_none(refusal);
    int32_t n = tr_opt_quad_dim(quad);
    if (n < 1) { set_error(error, TR_MATH_INVALID); return NULL; }
    if (n > TR_POLYOPT_MAX_VARS) { set_error(error, TR_MATH_UNSUPPORTED); return NULL; }
    pctx c;
    pctx_init(&c, budget, refusal);
    tr_bigq **parts[3] = {NULL, NULL, NULL};
    uint64_t counts[3] = {(uint64_t)n * n, (uint64_t)n, 1};
    for (int k = 0; k < 3 && LIVE(&c); k++) {                   /* the objective's exact parts A, b, c */
        uint64_t cnt = tr_opt_quad_count(quad, k), limbs = tr_opt_quad_limbs(quad, k);
        if (cnt != counts[k]) { pfail(&c, TR_MATH_INVALID); break; }
        int8_t *sign = (int8_t *)zalloc(cnt);
        uint32_t *nl = (uint32_t *)zalloc(cnt * 4), *dl = (uint32_t *)zalloc(cnt * 4);
        uint64_t *lb = (uint64_t *)zalloc((limbs ? limbs : 1) * 8);
        tr_lin_qbuf qb;
        qb.count = cnt; qb.nlimbs = limbs; qb.sign = sign; qb.nlen = nl; qb.dlen = dl; qb.limbs = lb;
        parts[k] = Q_arr(&c, cnt);
        int32_t rc = (!sign || !nl || !dl || !lb) ? TR_MATH_MEMORY : tr_opt_quad_export(quad, k, &qb);
        if (!rc && parts[k]) rc = tr_lin_q_import(&qb, cnt, parts[k], budget, refusal);
        if (rc) pfail(&c, rc);
        free(sign); free(nl); free(dl); free(lb);
    }
    tr_bigq **g = Q_arr(&c, (uint64_t)n);
    for (int32_t i = 0; g && parts[1] && i < n && LIVE(&c); i++) {
        g[i] = Q_neg(&c, parts[1][i]);
        if (linear && linear[i]) Q_set(&g[i], Q_add(&c, g[i], linear[i]));
    }
    tr_polyopt_qp *q = NULL;
    if (LIVE(&c))
        q = tr_polyopt_qp_create(n, (const tr_bigq *const *)parts[0], (const tr_bigq *const *)g, parts[2][0], m, A, b,
                                 p, E, e, lo, hi, budget, refusal, &c.rc);
    Q_free(parts[0], counts[0]); Q_free(parts[1], counts[1]); Q_free(parts[2], counts[2]);
    Q_free(g, (uint64_t)n);
    set_error(error, c.rc);
    return c.rc ? NULL : q;
}

/* q(x) and g(x) = H x + g */
static tr_bigq **qp_grad(pctx *c, const tr_polyopt_qp *q, tr_bigq *const *x) {
    const int32_t n = q->n;
    tr_bigq **gr = Q_arr(c, (uint64_t)n);
    for (int32_t i = 0; gr && i < n && LIVE(c); i++) {
        tr_bigq *acc = Q_copy(c, q->g[i]);
        if (Q_zero(acc)) { tr_bigq_free(acc); acc = NULL; }
        for (int32_t j = 0; j < n && LIVE(c); j++) Q_fma(c, &acc, q->H[(uint64_t)i * n + j], x[j]);
        gr[i] = acc ? acc : Q_int(c, 0);
    }
    return gr;
}
static tr_bigq *qp_value_at(pctx *c, const tr_polyopt_qp *q, tr_bigq *const *x) {
    const int32_t n = q->n;
    tr_bigq *quad = NULL, *lin = NULL;
    for (int32_t i = 0; i < n && LIVE(c); i++) {
        Q_fma(c, &lin, q->g[i], x[i]);
        if (Q_zero(x[i]) || q->hzero) continue;
        tr_bigq *hx = NULL;
        for (int32_t j = 0; j < n && LIVE(c); j++) Q_fma(c, &hx, q->H[(uint64_t)i * n + j], x[j]);
        Q_fma(c, &quad, hx, x[i]);
        tr_bigq_free(hx);
    }
    tr_bigq *half = Q_q64(c, 1, 2), *h = Q_mul(c, quad, half), *s = Q_add(c, h, lin), *v = Q_add(c, s, q->c0);
    tr_bigq_free(quad); tr_bigq_free(lin); tr_bigq_free(half); tr_bigq_free(h); tr_bigq_free(s);
    return v;
}
tr_bigq *tr_polyopt_qp_value(const tr_polyopt_qp *q, const tr_bigq *const *x, const tr_math_budget *budget,
                             tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    if (!qp_valid(q) || !x) { set_error(error, TR_MATH_INVALID); return NULL; }
    for (int32_t i = 0; i < q->n; i++) if (!x[i]) { set_error(error, TR_MATH_INVALID); return NULL; }
    pctx c;
    pctx_init(&c, budget, refusal);
    tr_bigq *v = qp_value_at(&c, q, (tr_bigq *const *)x);
    if (c.rc) { tr_bigq_free(v); set_error(error, c.rc); return NULL; }
    set_error(error, TR_MATH_OK);
    return v;
}
/* row i of the inequality block (i < mi) or of the equalities (mi + j): its coefficients */
static const tr_bigq *const *qp_row(const tr_polyopt_qp *q, int32_t i) {
    return (const tr_bigq *const *)(i < q->mi ? q->A + (uint64_t)i * q->n : q->E + (uint64_t)(i - q->mi) * q->n);
}

/* ---- exact verification of the answers (independent of how they were found) ---- */
static int verify_feasible(pctx *c, const tr_polyopt_qp *q, tr_bigq *const *x, int32_t *active) {
    int ok = 1;
    for (int32_t i = 0; i < q->mi && ok && LIVE(c); i++) {
        tr_bigq *ax = Q_dot(c, qp_row(q, i), (const tr_bigq *const *)x, q->n);
        int cmp = LIVE(c) ? Q_cmp(ax, q->b[i]) : 0;
        if (cmp > 0) ok = 0;
        if (active) active[i] = cmp == 0;
        tr_bigq_free(ax);
    }
    for (int32_t j = 0; j < q->p && ok && LIVE(c); j++) {
        tr_bigq *ex = Q_dot(c, qp_row(q, q->mi + j), (const tr_bigq *const *)x, q->n);
        if (LIVE(c) && Q_cmp(ex, q->e[j])) ok = 0;
        tr_bigq_free(ex);
    }
    return ok && LIVE(c);
}
/* KKT: feasibility, lambda >= 0, complementary slackness, H x + g + A^T lambda + E^T mu = 0 */
static int verify_kkt(pctx *c, const tr_polyopt_qp *q, tr_bigq *const *x, tr_bigq *const *lambda, tr_bigq *const *mu) {
    const int32_t n = q->n;
    int32_t *act = (int32_t *)zalloc((uint64_t)(q->mi ? q->mi : 1) * 4);
    if (!act) { pfail(c, TR_MATH_MEMORY); return 0; }
    int ok = verify_feasible(c, q, x, act);
    for (int32_t i = 0; i < q->mi && ok; i++) if (Q_sgn(lambda[i]) < 0 || (Q_sgn(lambda[i]) > 0 && !act[i])) ok = 0;
    tr_bigq **r = ok ? qp_grad(c, q, x) : NULL;
    for (int32_t i = 0; r && i < q->mi && LIVE(c); i++)
        for (int32_t j = 0; j < n && LIVE(c); j++) { Q_fma(c, &r[j], q->A[(uint64_t)i * n + j], lambda[i]); }
    for (int32_t i = 0; r && i < q->p && LIVE(c); i++)
        for (int32_t j = 0; j < n && LIVE(c); j++) { Q_fma(c, &r[j], q->E[(uint64_t)i * n + j], mu[i]); }
    for (int32_t j = 0; r && j < n && ok; j++) if (!Q_zero(r[j])) ok = 0;
    Q_free(r, r ? (uint64_t)n : 0);
    free(act);
    return ok && LIVE(c);
}
/* Farkas: y >= 0, A^T y + E^T z = 0, b^T y + e^T z < 0 */
static int verify_farkas(pctx *c, const tr_polyopt_qp *q, tr_bigq *const *y, tr_bigq *const *z) {
    const int32_t n = q->n;
    int ok = 1;
    for (int32_t i = 0; i < q->mi && ok; i++) if (Q_sgn(y[i]) < 0) ok = 0;
    tr_bigq **s = Q_arr(c, (uint64_t)n), *v = NULL;
    for (int32_t i = 0; s && i < q->mi && LIVE(c); i++) {
        for (int32_t j = 0; j < n && LIVE(c); j++) Q_fma(c, &s[j], q->A[(uint64_t)i * n + j], y[i]);
        Q_fma(c, &v, q->b[i], y[i]);
    }
    for (int32_t i = 0; s && i < q->p && LIVE(c); i++) {
        for (int32_t j = 0; j < n && LIVE(c); j++) Q_fma(c, &s[j], q->E[(uint64_t)i * n + j], z[i]);
        Q_fma(c, &v, q->e[i], z[i]);
    }
    for (int32_t j = 0; s && j < n && ok; j++) if (!Q_zero(s[j])) ok = 0;
    if (Q_sgn(v) >= 0) ok = 0;
    Q_free(s, s ? (uint64_t)n : 0);
    tr_bigq_free(v);
    return ok && LIVE(c);
}
/* a ray: x0 feasible, A d <= 0, E d = 0, and d^T H d < 0 (*curv) or d^T H d = 0 and g(x0)^T d < 0 (*rate): then
 * q(x0 + s d) = q(x0) + s g(x0)^T d + s^2 d^T H d / 2 -> -inf */
static int ray_check(pctx *c, const tr_polyopt_qp *q, tr_bigq *const *x0, tr_bigq *const *d, tr_bigq **rate,
                     tr_bigq **curv) {
    const int32_t n = q->n;
    int ok = verify_feasible(c, q, x0, NULL), nz = 0;
    for (int32_t j = 0; j < n; j++) if (!Q_zero(d[j])) nz = 1;
    if (!nz) ok = 0;
    for (int32_t i = 0; i < q->mi + q->p && ok && LIVE(c); i++) {
        tr_bigq *ad = Q_dot(c, qp_row(q, i), (const tr_bigq *const *)d, n);
        int s = Q_sgn(ad);
        if (i < q->mi ? s > 0 : s != 0) ok = 0;
        tr_bigq_free(ad);
    }
    tr_bigq *k = NULL;                                            /* d^T H d */
    for (int32_t i = 0; i < n && ok && !q->hzero && LIVE(c); i++) {
        if (Q_zero(d[i])) continue;
        tr_bigq *hd = Q_dot(c, (const tr_bigq *const *)(q->H + (uint64_t)i * n), (const tr_bigq *const *)d, n);
        Q_fma(c, &k, hd, d[i]);
        tr_bigq_free(hd);
    }
    if (!k && LIVE(c)) k = Q_int(c, 0);
    tr_bigq **gr = ok ? qp_grad(c, q, x0) : NULL;
    tr_bigq *r = gr ? Q_dot(c, (const tr_bigq *const *)gr, (const tr_bigq *const *)d, n) : NULL;
    if (ok && LIVE(c)) {
        int ks = Q_sgn(k);
        if (ks > 0 || (ks == 0 && Q_sgn(r) >= 0)) ok = 0;
    }
    Q_free(gr, gr ? (uint64_t)n : 0);
    if (ok && LIVE(c)) { *rate = r; *curv = k; } else { tr_bigq_free(r); tr_bigq_free(k); }
    return ok && LIVE(c);
}

/* ---- results of convex problems ---- */
static void split_rows(pctx *c, tr_polyopt_result *r, const tr_polyopt_qp *q, tr_bigq *const *w, int32_t part_m,
                       int32_t part_lo, int32_t part_hi) {
    const int32_t n = q->n;
    tr_bigq **um = Q_arr(c, (uint64_t)(q->m ? q->m : 1)), **lo = Q_arr(c, (uint64_t)n), **hi = Q_arr(c, (uint64_t)n);
    for (int32_t i = 0; um && i < q->m && LIVE(c); i++) um[i] = Q_copy(c, w[i]);
    for (int32_t i = q->m; lo && hi && i < q->mi && LIVE(c); i++) {
        int32_t v = q->bvar[i - q->m];
        Q_set(q->bside[i - q->m] < 0 ? &lo[v] : &hi[v], Q_copy(c, w[i]));
    }
    res_set(c, r, part_m, um, (uint64_t)q->m);
    res_set(c, r, part_lo, lo, (uint64_t)n);
    res_set(c, r, part_hi, hi, (uint64_t)n);
}
static void result_farkas(pctx *c, tr_polyopt_result *r, const tr_polyopt_qp *q, tr_bigq *const *y, tr_bigq *const *z) {
    if (!verify_farkas(c, q, y, z)) { pfail(c, TR_MATH_INVALID); return; }
    r->info.status = TR_POLYOPT_INFEASIBLE; r->info.feasibility = TR_POLYOPT_NOT_FEASIBLE; r->info.certificate = 1;
    split_rows(c, r, q, y, TR_POLYOPT_PART_FARKAS_Y, TR_POLYOPT_PART_FARKAS_LO, TR_POLYOPT_PART_FARKAS_HI);
    res_set(c, r, TR_POLYOPT_PART_FARKAS_Z, Q_full(c, (const tr_bigq *const *)z, (uint64_t)q->p), (uint64_t)q->p);
}
static void result_ray(pctx *c, tr_polyopt_result *r, const tr_polyopt_qp *q, tr_bigq *const *x0, tr_bigq *const *d) {
    tr_bigq *rate = NULL, *curv = NULL;
    if (!ray_check(c, q, x0, d, &rate, &curv)) { if (LIVE(c)) pfail(c, TR_MATH_INVALID); return; }
    r->info.status = TR_POLYOPT_UNBOUNDED; r->info.feasibility = TR_POLYOPT_FEASIBLE; r->info.certificate = 1;
    res_set(c, r, TR_POLYOPT_PART_ANCHOR, Q_full(c, (const tr_bigq *const *)x0, (uint64_t)q->n), (uint64_t)q->n);
    res_set(c, r, TR_POLYOPT_PART_DIRECTION, Q_full(c, (const tr_bigq *const *)d, (uint64_t)q->n), (uint64_t)q->n);
    res_one(c, r, TR_POLYOPT_PART_RATE, rate);
    res_one(c, r, TR_POLYOPT_PART_CURVATURE, curv);
    tr_bigq_free(rate); tr_bigq_free(curv);
}
/* a candidate ray of a nonconvex problem: the result when it certifies unboundedness, else nothing (0) */
static int try_ray(pctx *c, tr_polyopt_result *r, const tr_polyopt_qp *q, tr_bigq *const *x0, tr_bigq *const *d) {
    tr_bigq *rate = NULL, *curv = NULL;
    if (!ray_check(c, q, x0, d, &rate, &curv)) return 0;
    tr_bigq_free(rate); tr_bigq_free(curv);
    result_ray(c, r, q, x0, d);
    return LIVE(c);
}
static void result_optimum(pctx *c, tr_polyopt_result *r, const tr_polyopt_qp *q, tr_bigq *const *x,
                           tr_bigq *const *lambda, tr_bigq *const *mu) {
    if (!verify_kkt(c, q, x, lambda, mu)) { pfail(c, TR_MATH_INVALID); return; }
    const int32_t n = q->n;
    r->info.status = TR_POLYOPT_OPTIMAL; r->info.feasibility = TR_POLYOPT_FEASIBLE; r->info.kkt = 1;
    tr_bigq *v = qp_value_at(c, q, x);
    res_set(c, r, TR_POLYOPT_PART_POINT, Q_full(c, (const tr_bigq *const *)x, (uint64_t)n), (uint64_t)n);
    res_one(c, r, TR_POLYOPT_PART_VALUE, v);
    res_one(c, r, TR_POLYOPT_PART_LOWER, v);
    res_one(c, r, TR_POLYOPT_PART_UPPER, v);
    tr_bigq_free(v);
    split_rows(c, r, q, lambda, TR_POLYOPT_PART_LAMBDA, TR_POLYOPT_PART_LAMBDA_LO, TR_POLYOPT_PART_LAMBDA_HI);
    res_set(c, r, TR_POLYOPT_PART_MU, Q_full(c, (const tr_bigq *const *)mu, (uint64_t)q->p), (uint64_t)q->p);
    res_set(c, r, TR_POLYOPT_PART_GRADIENT, qp_grad(c, q, x), (uint64_t)n);
    tr_bigq **sl = Q_arr(c, (uint64_t)(q->m ? q->m : 1));
    r->active = (int32_t *)zalloc((uint64_t)(q->mi ? q->mi : 1) * 4);
    if (!r->active) pfail(c, TR_MATH_MEMORY);
    r->nactive = 0;
    for (int32_t i = 0; i < q->mi && LIVE(c); i++) {
        tr_bigq *ax = Q_dot(c, qp_row(q, i), (const tr_bigq *const *)x, n), *s = Q_sub(c, q->b[i], ax);
        if (Q_zero(s)) {
            int32_t code = i < q->m ? i : q->m + (q->bside[i - q->m] < 0 ? 0 : n) + q->bvar[i - q->m];
            r->active[r->nactive++] = code;
        }
        if (i < q->m && sl) sl[i] = s; else tr_bigq_free(s);
        tr_bigq_free(ax);
    }
    for (int32_t a = 1; a < r->nactive; a++)                      /* active rows in increasing code order */
        for (int32_t k = a; k > 0 && r->active[k - 1] > r->active[k]; k--) {
            int32_t t = r->active[k]; r->active[k] = r->active[k - 1]; r->active[k - 1] = t;
        }
    r->info.active = r->nactive;
    res_set(c, r, TR_POLYOPT_PART_SLACK, sl, (uint64_t)q->m);
}

/* ---- uniqueness: the optimal set S = {x feasible : H x = H x*, g^T x = g^T x*} is a point iff its projections
 * are; a second point of S when it is not ---- */
static void uniqueness(pctx *c, tr_polyopt_result *r, const tr_polyopt_qp *q, tr_bigq *const *xs) {
    const int32_t n = q->n;
    int32_t extra = q->hzero ? 0 : n;
    int32_t pe = q->p + extra + 1;
    tr_bigq **E = Q_arr(c, (uint64_t)pe * n), **e = Q_arr(c, (uint64_t)pe), **cost = Q_arr(c, (uint64_t)n);
    for (int32_t i = 0; E && i < q->p && LIVE(c); i++) {
        for (int32_t j = 0; j < n && LIVE(c); j++) E[(uint64_t)i * n + j] = Q_copy(c, q->E[(uint64_t)i * n + j]);
        e[i] = Q_copy(c, q->e[i]);
    }
    for (int32_t k = 0; E && k < extra && LIVE(c); k++) {        /* H x = H x* */
        for (int32_t j = 0; j < n && LIVE(c); j++) E[(uint64_t)(q->p + k) * n + j] = Q_copy(c, q->H[(uint64_t)k * n + j]);
        e[q->p + k] = Q_dot(c, (const tr_bigq *const *)(q->H + (uint64_t)k * n), (const tr_bigq *const *)xs, n);
    }
    for (int32_t j = 0; E && j < n && LIVE(c); j++) E[(uint64_t)(pe - 1) * n + j] = Q_copy(c, q->g[j]);
    if (e) e[pe - 1] = Q_dot(c, (const tr_bigq *const *)q->g, (const tr_bigq *const *)xs, n);
    int found = 0;
    for (int32_t k = 0; k < n && !found && LIVE(c); k++)
        for (int32_t s = -1; s <= 1 && !found && LIVE(c); s += 2) {
            for (int32_t j = 0; j < n; j++) { tr_bigq_free(cost[j]); cost[j] = NULL; }
            cost[k] = Q_int(c, s);                               /* s = -1: maximize x_k; s = 1: minimize */
            lp_out o;
            lp_core(c, n, q->mi, (const tr_bigq *const *)q->A, (const tr_bigq *const *)q->b, pe,
                    (const tr_bigq *const *)E, (const tr_bigq *const *)e, (const tr_bigq *const *)cost, &o);
            if (LIVE(c) && o.status == TR_POLYOPT_UNBOUNDED) {
                found = 1;                                       /* S is unbounded: x0 + d differs from x0 */
                tr_bigq **pt = Q_arr(c, (uint64_t)n);
                for (int32_t j = 0; pt && j < n && LIVE(c); j++) pt[j] = Q_add(c, o.x[j], o.ray[j]);
                int same = 1;
                for (int32_t j = 0; pt && j < n && LIVE(c); j++) if (Q_cmp(pt[j], xs[j])) same = 0;
                if (same) for (int32_t j = 0; pt && j < n && LIVE(c); j++) Q_set(&pt[j], Q_add(c, pt[j], o.ray[j]));
                res_set(c, r, TR_POLYOPT_PART_FACE, pt, (uint64_t)n);
            } else if (LIVE(c) && o.status == TR_POLYOPT_OPTIMAL && Q_cmp(o.x[k], xs[k]) != 0) {
                found = 1;
                res_set(c, r, TR_POLYOPT_PART_FACE, Q_full(c, (const tr_bigq *const *)o.x, (uint64_t)n), (uint64_t)n);
            } else if (LIVE(c) && o.status != TR_POLYOPT_OPTIMAL) pfail(c, TR_MATH_INVALID);   /* x* is in S */
            lp_out_free(&o, n, q->mi, pe);
        }
    if (LIVE(c)) r->info.unique = !found;
    if (found && r->part[TR_POLYOPT_PART_FACE]) {                /* the second point is optimal: checked exactly */
        tr_bigq *v1 = qp_value_at(c, q, r->part[TR_POLYOPT_PART_FACE]), *v0 = qp_value_at(c, q, xs);
        if (LIVE(c) && (Q_cmp(v1, v0) || !verify_feasible(c, q, r->part[TR_POLYOPT_PART_FACE], NULL))) pfail(c, TR_MATH_INVALID);
        tr_bigq_free(v1); tr_bigq_free(v0);
    }
    Q_free(E, (uint64_t)pe * n); Q_free(e, (uint64_t)pe); Q_free(cost, (uint64_t)n);
}

/* ---- the active-set method for H != 0 (PSD) from a feasible point ---- */
static tr_bigq *Q_zero_const(pctx *c) { return Q_int(c, 0); }
/* rank of the rows `rows` (codes as qp_row) */
static int32_t rows_rank(pctx *c, const tr_polyopt_qp *q, const int32_t *rows, int32_t k, const tr_bigq *zero) {
    const int32_t n = q->n;
    if (!k) return 0;
    const tr_bigq **M = (const tr_bigq **)zalloc((uint64_t)k * n * sizeof *M), **z = (const tr_bigq **)zalloc((uint64_t)k * sizeof *z);
    if (!M || !z) { free(M); free(z); pfail(c, TR_MATH_MEMORY); return 0; }
    for (int32_t a = 0; a < k; a++) {
        const tr_bigq *const *row = qp_row(q, rows[a]);
        for (int32_t j = 0; j < n; j++) M[(uint64_t)a * n + j] = row[j];
        z[a] = zero;
    }
    int32_t e = 0, rank = 0;
    tr_lin_result *s = tr_lin_q_solve(k, n, M, z, c->b, c->r, &e);
    if (!s) pfail(c, e);
    else { tr_lin_result_info ii; tr_lin_result_get_info(s, &ii); rank = ii.rank; tr_lin_result_free(s); }
    free(M); free(z);
    return rank;
}
static void active_set(pctx *c, tr_polyopt_result *r, const tr_polyopt_qp *q, tr_bigq **x, const int32_t *active0) {
    const int32_t n = q->n, mi = q->mi, p = q->p;
    tr_bigq *zero = Q_zero_const(c);
    int32_t *W = (int32_t *)zalloc((uint64_t)(n + p + 1) * 4), w = 0;
    uint8_t *inW = (uint8_t *)zalloc((uint64_t)(mi + p + 1));
    if (!W || !inW) pfail(c, TR_MATH_MEMORY);
    for (int32_t j = 0; j < p && w < n && LIVE(c); j++) {        /* independent equalities, then active rows */
        W[w] = mi + j;
        if (rows_rank(c, q, W, w + 1, zero) == w + 1) { inW[mi + j] = 1; w++; }
    }
    for (int32_t i = 0; i < mi && w < n && LIVE(c); i++) {
        if (!active0[i]) continue;
        W[w] = i;
        if (rows_rank(c, q, W, w + 1, zero) == w + 1) { inW[i] = 1; w++; }
    }
    int done = 0;
    /* the active-set steps: at most 100 (n + rows + 1) within the iteration budget; past them NOT_CONVERGED */
    uint64_t cap = (uint64_t)100 * (uint64_t)(n + mi + p + 1), steps = 0;
    while (!done && LIVE(c)) {
        if (steps++ >= cap) { pfail(c, TR_MATH_NOT_CONVERGED); break; }
        if (!tick(c)) break;
        const int32_t K = n + w;
        tr_bigq **cx = qp_grad(c, q, x);
        const tr_bigq **M = (const tr_bigq **)zalloc((uint64_t)K * K * sizeof *M);
        tr_bigq **rhs = Q_arr(c, (uint64_t)K);
        if (!M) pfail(c, TR_MATH_MEMORY);
        for (int32_t a = 0; M && rhs && a < K && LIVE(c); a++) {
            for (int32_t b2 = 0; b2 < K; b2++) {
                const tr_bigq *v = zero;
                if (a < n && b2 < n) v = q->H[(uint64_t)a * n + b2];
                else if (a < n && b2 >= n) v = qp_row(q, W[b2 - n])[a];
                else if (a >= n && b2 < n) v = qp_row(q, W[a - n])[b2];
                M[(uint64_t)a * K + b2] = v;
            }
            rhs[a] = a < n ? Q_neg(c, cx[a]) : Q_int(c, 0);
        }
        int32_t e = 0;
        tr_lin_result *s = LIVE(c) ? tr_lin_q_solve(K, K, M, (const tr_bigq *const *)rhs, c->b, c->r, &e) : NULL;
        if (!s && LIVE(c)) pfail(c, e);
        tr_lin_result_info si;
        if (s) tr_lin_result_get_info(s, &si);
        tr_bigq **dir = NULL;
        int full = 0;                                            /* 1: a full step (EQP minimizer) may be taken */
        if (s && LIVE(c)) {
            uint64_t cnt = 0;
            if (si.status == TR_LIN_NONE) {                      /* a zero-curvature descent direction */
                const tr_bigq *const *y = tr_lin_result_part(s, TR_LIN_PART_CERT, &cnt);
                dir = Q_full(c, y, (uint64_t)n);
            } else {
                const tr_bigq *const *sol = tr_lin_result_part(s, TR_LIN_PART_X, &cnt);
                int zerostep = 1;
                for (int32_t j = 0; j < n; j++) if (!Q_zero(sol[j])) zerostep = 0;
                if (zerostep) {
                    int32_t drop = -1;                           /* the smallest inequality row with nu < 0 */
                    for (int32_t a = 0; a < w; a++)
                        if (W[a] < mi && Q_sgn(sol[n + a]) < 0 && (drop < 0 || W[a] < W[drop])) drop = a;
                    if (drop < 0) {
                        tr_bigq **lam = Q_arr(c, (uint64_t)(mi ? mi : 1)), **mu = Q_arr(c, (uint64_t)(p ? p : 1));
                        for (int32_t a = 0; lam && mu && a < w && LIVE(c); a++) {
                            if (W[a] < mi) lam[W[a]] = Q_copy(c, sol[n + a]);
                            else mu[W[a] - mi] = Q_copy(c, sol[n + a]);
                        }
                        for (int32_t i = 0; lam && i < mi && LIVE(c); i++) if (!lam[i]) lam[i] = Q_int(c, 0);
                        for (int32_t i = 0; mu && i < p && LIVE(c); i++) if (!mu[i]) mu[i] = Q_int(c, 0);
                        if (LIVE(c)) result_optimum(c, r, q, x, lam, mu);
                        Q_free(lam, (uint64_t)(mi ? mi : 1)); Q_free(mu, (uint64_t)(p ? p : 1));
                        done = 1;
                    } else {
                        inW[W[drop]] = 0;
                        for (int32_t a = drop; a + 1 < w; a++) W[a] = W[a + 1];
                        w--;
                    }
                } else { dir = Q_full(c, sol, (uint64_t)n); full = 1; }
            }
        }
        tr_lin_result_free(s);
        free(M);
        Q_free(rhs, (uint64_t)K);
        Q_free(cx, (uint64_t)n);
        if (dir && LIVE(c)) {                                    /* the ratio test along dir */
            int32_t block = -1;
            tr_bigq *alpha = NULL;
            for (int32_t i = 0; i < mi && LIVE(c); i++) {
                if (inW[i]) continue;
                tr_bigq *ad = Q_dot(c, qp_row(q, i), (const tr_bigq *const *)dir, n);
                if (Q_sgn(ad) > 0) {
                    tr_bigq *ax = Q_dot(c, qp_row(q, i), (const tr_bigq *const *)x, n), *sl = Q_sub(c, q->b[i], ax);
                    tr_bigq *t = Q_div(c, sl, ad);
                    if (LIVE(c) && (!alpha || Q_cmp(t, alpha) < 0)) { tr_bigq_free(alpha); alpha = t; t = NULL; block = i; }
                    tr_bigq_free(t); tr_bigq_free(ax); tr_bigq_free(sl);
                }
                tr_bigq_free(ad);
            }
            if (!alpha && !full) {                               /* unbounded below along a feasible ray */
                if (LIVE(c)) result_ray(c, r, q, x, dir);
                done = 1;
            } else if (LIVE(c)) {
                tr_bigq *one = Q_int(c, 1);
                int take_block = alpha && (!full || Q_cmp(alpha, one) < 0);
                const tr_bigq *step = take_block ? alpha : one;
                for (int32_t j = 0; j < n && LIVE(c); j++) {
                    tr_bigq *dx = Q_mul(c, step, dir[j]);
                    Q_set(&x[j], Q_add(c, x[j], dx));
                    tr_bigq_free(dx);
                }
                if (take_block && LIVE(c)) { inW[block] = 1; W[w++] = block; }
                tr_bigq_free(one);
            }
            tr_bigq_free(alpha);
        }
        Q_free(dir, (uint64_t)n);
    }
    if (!done && c->rc == TR_MATH_BUDGET && c->iters >= c->limit) c->rc = TR_MATH_NOT_CONVERGED;
    free(W); free(inW);
    tr_bigq_free(zero);
}

/* ---- handing an indefinite problem to branch and bound ---- */
struct tr_polyopt_poly_s;
static tr_polyopt_result *bb_run(pctx *c, const struct tr_polyopt_poly_s *P, const tr_bigq *tol, uint64_t batch,
                                 tr_polyopt_boxes_fn fn, double time_limit_ms);
static struct tr_polyopt_poly_s *poly_from_dense(pctx *c, int32_t n, int32_t npolys, int32_t ncons,
                                                 tr_bigq ***coef, const int32_t (*deg)[TR_POLYOPT_BB_VARS],
                                                 tr_bigq *const *lo, tr_bigq *const *hi);
/* x = base + N y (base NULL: 0), N stored as dim rows of length n */
static tr_bigq **map_back(pctx *c, tr_bigq *const *base, tr_bigq *const *N, int32_t dim, tr_bigq *const *y, int32_t n) {
    tr_bigq **x = Q_full(c, (const tr_bigq *const *)base, (uint64_t)n);
    for (int32_t k = 0; x && y && k < dim && LIVE(c); k++)
        for (int32_t j = 0; j < n && LIVE(c); j++) Q_fma(c, &x[j], N[(uint64_t)k * n + j], y[k]);
    for (int32_t j = 0; x && j < n && LIVE(c); j++) if (!x[j]) x[j] = Q_int(c, 0);
    return x;
}
/* R = N^T H N (dim x dim) */
static tr_bigq **reduced_hessian(pctx *c, const tr_polyopt_qp *q, tr_bigq *const *N, int32_t dim) {
    const int32_t n = q->n;
    tr_bigq **R = Q_arr(c, (uint64_t)dim * dim), **HN = Q_arr(c, (uint64_t)dim * n);
    for (int32_t b = 0; R && HN && b < dim && LIVE(c); b++)          /* (H N_b) */
        for (int32_t i = 0; i < n && LIVE(c); i++) {
            tr_bigq *acc = NULL;
            for (int32_t j = 0; j < n && LIVE(c); j++) Q_fma(c, &acc, q->H[(uint64_t)i * n + j], N[(uint64_t)b * n + j]);
            HN[(uint64_t)b * n + i] = acc ? acc : Q_int(c, 0);
        }
    for (int32_t a = 0; R && HN && a < dim && LIVE(c); a++)
        for (int32_t b = a; b < dim && LIVE(c); b++) {
            tr_bigq *v = Q_dot(c, (const tr_bigq *const *)(N + (uint64_t)a * n), (const tr_bigq *const *)(HN + (uint64_t)b * n), n);
            R[(uint64_t)a * dim + b] = v;
            if (b != a) R[(uint64_t)b * dim + a] = Q_copy(c, v);
        }
    Q_free(HN, HN ? (uint64_t)dim * n : 0);
    return R;
}
/* E^T mu = -(H x + g): the multipliers of the equalities at x (p = 0: none) */
static tr_bigq **eq_multipliers(pctx *c, const tr_polyopt_qp *q, tr_bigq *const *x) {
    const int32_t n = q->n, p = q->p;
    if (!p) return Q_arr(c, 1);
    tr_bigq **gr = qp_grad(c, q, x), **ng = Q_arr(c, (uint64_t)n), **mu = NULL;
    const tr_bigq **Et = (const tr_bigq **)zalloc((uint64_t)n * p * sizeof *Et);
    if (!Et) pfail(c, TR_MATH_MEMORY);
    for (int32_t j = 0; Et && gr && ng && j < n && LIVE(c); j++) {
        for (int32_t i = 0; i < p; i++) Et[(uint64_t)j * p + i] = q->E[(uint64_t)i * n + j];
        ng[j] = Q_neg(c, gr[j]);
    }
    int32_t e = 0;
    tr_lin_result *s = (Et && LIVE(c)) ? tr_lin_q_solve(n, p, Et, (const tr_bigq *const *)ng, c->b, c->r, &e) : NULL;
    if (!s && LIVE(c)) pfail(c, e ? e : TR_MATH_MEMORY);
    if (s) {
        tr_lin_result_info si;
        tr_lin_result_get_info(s, &si);
        uint64_t cnt = 0;
        if (si.status == TR_LIN_NONE) pfail(c, TR_MATH_INVALID);  /* x is not stationary on the affine set */
        else mu = Q_full(c, tr_lin_result_part(s, TR_LIN_PART_X, &cnt), (uint64_t)p);
        tr_lin_result_free(s);
    }
    free(Et); Q_free(ng, ng ? (uint64_t)n : 0); Q_free(gr, gr ? (uint64_t)n : 0);
    return mu;
}
/* the feasible set is the affine set xp + N y (no inequality varies on it): decided by the inertia of R = N^T H N.
 * A negative direction v of R: the ray N v (CURVATURE). R PSD: q is convex on the feasible set; R y = -N^T g(xp)
 * consistent gives the optimum xp + N y (KKT with lambda = 0, unique iff R is definite, else a second optimal point
 * along ker R), inconsistent gives k in ker R with g(xp)^T N k = -1: the ray N k (RATE). */
static void affine_case(pctx *c, tr_polyopt_result *r, const tr_polyopt_qp *q, tr_bigq *const *xp, tr_bigq *const *N,
                        int32_t dim, tr_bigq *const *R, int definite, tr_bigq *const *neg) {
    const int32_t n = q->n, mi = q->mi;
    r->info.method = TR_POLYOPT_ACTIVE_SET;
    if (neg) {
        tr_bigq **d = map_back(c, NULL, N, dim, neg, n);
        if (d && LIVE(c)) result_ray(c, r, q, xp, d);
        Q_free(d, d ? (uint64_t)n : 0);
        return;
    }
    r->info.convex = 1;                                          /* convex on the feasible set */
    tr_bigq **gx = qp_grad(c, q, xp), **ngy = Q_arr(c, (uint64_t)dim);
    for (int32_t k = 0; gx && ngy && k < dim && LIVE(c); k++) {
        tr_bigq *t = Q_dot(c, (const tr_bigq *const *)(N + (uint64_t)k * n), (const tr_bigq *const *)gx, n);
        ngy[k] = Q_neg(c, t);
        tr_bigq_free(t);
    }
    Q_free(gx, gx ? (uint64_t)n : 0);
    int32_t e = 0;
    tr_lin_result *s = LIVE(c) ? tr_lin_q_solve(dim, dim, (const tr_bigq *const *)R, (const tr_bigq *const *)ngy,
                                                c->b, c->r, &e) : NULL;
    if (!s && LIVE(c)) pfail(c, e ? e : TR_MATH_MEMORY);
    Q_free(ngy, ngy ? (uint64_t)dim : 0);
    if (!s) return;
    tr_lin_result_info si;
    tr_lin_result_get_info(s, &si);
    uint64_t cnt = 0, nn = 0;
    if (si.status == TR_LIN_NONE) {
        const tr_bigq *const *k = tr_lin_result_part(s, TR_LIN_PART_CERT, &cnt);
        tr_bigq **kk = k && cnt == (uint64_t)dim ? Q_full(c, k, (uint64_t)dim) : NULL;
        tr_bigq **d = kk ? map_back(c, NULL, N, dim, kk, n) : NULL;
        if (d && LIVE(c)) result_ray(c, r, q, xp, d);
        else if (LIVE(c)) pfail(c, TR_MATH_INVALID);
        Q_free(kk, kk ? (uint64_t)dim : 0); Q_free(d, d ? (uint64_t)n : 0);
    } else {
        const tr_bigq *const *y = tr_lin_result_part(s, TR_LIN_PART_X, &cnt);
        const tr_bigq *const *ker = tr_lin_result_part(s, TR_LIN_PART_NULL, &nn);
        tr_bigq **yy = Q_full(c, y, (uint64_t)dim);
        tr_bigq **x = yy ? map_back(c, xp, N, dim, yy, n) : NULL;
        tr_bigq **lam = Q_full(c, NULL, (uint64_t)(mi ? mi : 1));
        tr_bigq **mu = x ? eq_multipliers(c, q, x) : NULL;
        if (x && lam && mu && LIVE(c)) result_optimum(c, r, q, x, lam, mu);
        if (LIVE(c)) r->info.unique = definite ? 1 : 0;
        if (LIVE(c) && !definite) {                               /* a second optimal point along ker R */
            tr_bigq **k = ker && nn >= (uint64_t)dim ? Q_full(c, ker, (uint64_t)dim) : NULL;
            tr_bigq **f = k ? map_back(c, x, N, dim, k, n) : NULL;
            tr_bigq *v0 = f ? qp_value_at(c, q, x) : NULL, *v1 = f ? qp_value_at(c, q, f) : NULL;
            if (!f || !verify_feasible(c, q, f, NULL) || Q_cmp(v0, v1)) { if (LIVE(c)) pfail(c, TR_MATH_INVALID); }
            else res_set(c, r, TR_POLYOPT_PART_FACE, f, (uint64_t)n), f = NULL;
            tr_bigq_free(v0); tr_bigq_free(v1);
            Q_free(k, k ? (uint64_t)dim : 0); Q_free(f, f ? (uint64_t)n : 0);
        }
        Q_free(yy, yy ? (uint64_t)dim : 0); Q_free(x, x ? (uint64_t)n : 0);
        Q_free(lam, (uint64_t)(mi ? mi : 1)); Q_free(mu, mu ? (uint64_t)(q->p ? q->p : 1) : 0);
    }
    tr_lin_result_free(s);
}
static void handoff(pctx *c, tr_polyopt_result **out, const tr_polyopt_qp *q, const tr_bigq *tol, uint64_t batch,
                    tr_polyopt_boxes_fn fn, double time_limit_ms) {
    const int32_t n = q->n, mi = q->mi, p = q->p;
    tr_polyopt_result *r = *out;
    lp_out o;
    lp_core(c, n, mi, (const tr_bigq *const *)q->A, (const tr_bigq *const *)q->b, p, (const tr_bigq *const *)q->E,
            (const tr_bigq *const *)q->e, NULL, &o);
    if (!LIVE(c)) { lp_out_free(&o, n, mi, p); return; }
    if (o.status == TR_POLYOPT_INFEASIBLE) { result_farkas(c, r, q, o.y, o.z); lp_out_free(&o, n, mi, p); return; }
    tr_bigq **xf = o.x;                                          /* a feasible vertex: the anchor of rays */
    o.x = NULL;
    lp_out_free(&o, n, mi, p);
    /* x = xp + N y: the equalities eliminated exactly */
    int32_t dim = n;
    tr_bigq **xp = NULL, **N = NULL;
    if (p) {
        int32_t e = 0;
        tr_lin_result *s = tr_lin_q_solve(p, n, (const tr_bigq *const *)q->E, (const tr_bigq *const *)q->e, c->b, c->r, &e);
        if (!s) { pfail(c, e); Q_free(xf, (uint64_t)n); return; }
        tr_lin_result_info si;
        tr_lin_result_get_info(s, &si);
        uint64_t cnt = 0, nn = 0;
        const tr_bigq *const *x0 = tr_lin_result_part(s, TR_LIN_PART_X, &cnt);
        const tr_bigq *const *ker = tr_lin_result_part(s, TR_LIN_PART_NULL, &nn);
        if (si.status == TR_LIN_NONE) pfail(c, TR_MATH_INVALID);         /* the LP found a feasible point */
        xp = Q_full(c, x0, (uint64_t)n);
        dim = si.status == TR_LIN_UNIQUE ? 0 : (int32_t)(nn / (uint64_t)n);
        N = dim ? Q_full(c, ker, (uint64_t)dim * n) : NULL;
        tr_lin_result_free(s);
    } else {
        xp = Q_full(c, NULL, (uint64_t)n);
        N = Q_arr(c, (uint64_t)n * n);
        for (int32_t i = 0; N && i < n && LIVE(c); i++)
            for (int32_t j = 0; j < n && LIVE(c); j++) N[(uint64_t)i * n + j] = Q_int(c, i == j);
    }
    if (!LIVE(c)) { Q_free(xp, (uint64_t)n); Q_free(N, (uint64_t)dim * n); Q_free(xf, (uint64_t)n); return; }
    if (dim == 0) {                                              /* a single feasible point: its value */
        tr_bigq **lam = Q_full(c, NULL, (uint64_t)(mi ? mi : 1)), **mu = NULL;
        /* the point is optimal; multipliers from E^T mu = -(H x + g) */
        tr_bigq **gr = qp_grad(c, q, xp);
        const tr_bigq **Et = (const tr_bigq **)zalloc((uint64_t)n * (p ? p : 1) * sizeof *Et);
        tr_bigq **ng = Q_arr(c, (uint64_t)n);
        for (int32_t j = 0; Et && ng && j < n && LIVE(c); j++) {
            for (int32_t i = 0; i < p; i++) Et[(uint64_t)j * p + i] = q->E[(uint64_t)i * n + j];
            ng[j] = Q_neg(c, gr[j]);
        }
        int32_t e = 0;
        tr_lin_result *s = (Et && LIVE(c)) ? tr_lin_q_solve(n, p, Et, (const tr_bigq *const *)ng, c->b, c->r, &e) : NULL;
        if (!s && LIVE(c)) pfail(c, e ? e : TR_MATH_MEMORY);
        uint64_t cnt = 0;
        if (s) mu = Q_full(c, tr_lin_result_part(s, TR_LIN_PART_X, &cnt), (uint64_t)p);
        tr_lin_result_free(s);
        free(Et); Q_free(ng, (uint64_t)n); Q_free(gr, (uint64_t)n);
        if (mu && LIVE(c)) { result_optimum(c, r, q, xp, lam, mu); r->info.unique = 1; r->info.method = TR_POLYOPT_SIMPLEX; }
        Q_free(lam, (uint64_t)(mi ? mi : 1)); Q_free(mu, (uint64_t)p);
        Q_free(xp, (uint64_t)n); Q_free(xf, (uint64_t)n);
        return;
    }
    /* the reduced rows (A N) y <= b - A xp and the reduced Hessian R = N^T H N with its exact inertia */
    tr_bigq **AN = Q_arr(c, (uint64_t)(mi ? mi : 1) * dim), **bb = Q_arr(c, (uint64_t)(mi ? mi : 1));
    int effective = 0;                                           /* some inequality varies on the affine set */
    for (int32_t i = 0; AN && bb && i < mi && LIVE(c); i++) {
        for (int32_t k = 0; k < dim && LIVE(c); k++) {
            tr_bigq *acc = NULL;
            for (int32_t j = 0; j < n && LIVE(c); j++) Q_fma(c, &acc, q->A[(uint64_t)i * n + j], N[(uint64_t)k * n + j]);
            AN[(uint64_t)i * dim + k] = acc ? acc : Q_int(c, 0);
            if (!Q_zero(AN[(uint64_t)i * dim + k])) effective = 1;
        }
        tr_bigq *ax = Q_dot(c, qp_row(q, i), (const tr_bigq *const *)xp, n);
        bb[i] = Q_sub(c, q->b[i], ax);
        tr_bigq_free(ax);
    }
    tr_bigq **R = reduced_hessian(c, q, (tr_bigq *const *)N, dim);
    tr_lin_result_info ri;
    memset(&ri, 0, sizeof ri);
    tr_bigq **neg = NULL;                                        /* v with v^T R v < 0 */
    if (R && LIVE(c)) {
        int32_t e = 0;
        tr_lin_result *ine = tr_lin_q_inertia(dim, (const tr_bigq *const *)R, c->b, c->r, &e);
        if (!ine) pfail(c, e ? e : TR_MATH_MEMORY);
        else {
            tr_lin_result_get_info(ine, &ri);
            uint64_t cnt = 0;
            const tr_bigq *const *v = ri.negative ? tr_lin_result_part(ine, TR_LIN_PART_NEG, &cnt) : NULL;
            if (v && cnt == (uint64_t)dim) neg = Q_full(c, v, (uint64_t)dim);
            else if (ri.negative) pfail(c, TR_MATH_INVALID);
            tr_lin_result_free(ine);
        }
    }
    int decided = 0;
    if (LIVE(c) && !effective) {                                 /* the feasible set is the affine set xp + N y */
        affine_case(c, r, q, xp, N, dim, R, ri.positive == dim, neg);
        decided = 1;
    }
    for (int32_t sg = 1; LIVE(c) && !decided && neg && sg >= -1; sg -= 2) {   /* +-v a recession direction */
        int rec = 1;
        for (int32_t i = 0; i < mi && rec && LIVE(c); i++) {
            tr_bigq *t = Q_dot(c, (const tr_bigq *const *)(AN + (uint64_t)i * dim), (const tr_bigq *const *)neg, dim);
            if (sg * Q_sgn(t) > 0) rec = 0;
            tr_bigq_free(t);
        }
        if (!rec || !LIVE(c)) continue;
        tr_bigq **d = map_back(c, NULL, N, dim, neg, n);
        for (int32_t j = 0; d && sg < 0 && j < n && LIVE(c); j++) Q_set(&d[j], Q_neg(c, d[j]));
        if (d && LIVE(c) && try_ray(c, r, q, xf, d)) decided = 1;
        Q_free(d, d ? (uint64_t)n : 0);
    }
    if (LIVE(c) && !decided && (dim > TR_POLYOPT_BB_VARS || mi > TR_POLYOPT_BB_CONS)) pfail(c, TR_MATH_UNSUPPORTED);
    /* the box of y by 2 dim exact LPs; an unbounded one offers its ray */
    tr_bigq *ylo[TR_POLYOPT_BB_VARS] = {0}, *yhi[TR_POLYOPT_BB_VARS] = {0};
    tr_bigq **cost = LIVE(c) && !decided ? Q_arr(c, (uint64_t)dim) : NULL;
    for (int32_t k = 0; cost && k < dim && LIVE(c) && !decided; k++)
        for (int32_t s = 1; s >= -1 && LIVE(c) && !decided; s -= 2) {
            for (int32_t j = 0; j < dim; j++) { tr_bigq_free(cost[j]); cost[j] = NULL; }
            cost[k] = Q_int(c, s);
            lp_out lo2;
            lp_core(c, dim, mi, (const tr_bigq *const *)AN, (const tr_bigq *const *)bb, 0, NULL, NULL,
                    (const tr_bigq *const *)cost, &lo2);
            if (LIVE(c) && lo2.status == TR_POLYOPT_UNBOUNDED) {
                tr_bigq **x0 = map_back(c, xp, N, dim, lo2.x, n), **d = map_back(c, NULL, N, dim, lo2.ray, n);
                if (x0 && d && LIVE(c) && try_ray(c, r, q, x0, d)) decided = 1;
                else if (LIVE(c)) pfail(c, TR_MATH_UNSUPPORTED); /* unbounded, nonconvex, no ray: undecided */
                Q_free(x0, x0 ? (uint64_t)n : 0); Q_free(d, d ? (uint64_t)n : 0);
            } else if (LIVE(c) && lo2.status != TR_POLYOPT_OPTIMAL) pfail(c, TR_MATH_INVALID);
            else if (LIVE(c)) { if (s > 0) ylo[k] = Q_copy(c, lo2.x[k]); else yhi[k] = Q_copy(c, lo2.x[k]); }
            lp_out_free(&lo2, dim, mi, 0);
        }
    Q_free(cost, cost ? (uint64_t)dim : 0);
    Q_free(neg, neg ? (uint64_t)dim : 0);
    Q_free(xf, (uint64_t)n);
    if (decided || !LIVE(c)) {
        Q_free(R, R ? (uint64_t)dim * dim : 0);
        for (int32_t k = 0; k < TR_POLYOPT_BB_VARS; k++) { tr_bigq_free(ylo[k]); tr_bigq_free(yhi[k]); }
        Q_free(AN, (uint64_t)(mi ? mi : 1) * dim); Q_free(bb, (uint64_t)(mi ? mi : 1));
        Q_free(xp, (uint64_t)n); Q_free(N, (uint64_t)dim * n);
        return;
    }
    /* the objective in y: y^T (N^T H N) y / 2 + (N^T (H xp + g))^T y + q(xp), as dense polynomials */
    tr_bigq **polys[TR_POLYOPT_BB_POLYS];
    int32_t deg[TR_POLYOPT_BB_POLYS][TR_POLYOPT_BB_VARS];
    memset(polys, 0, sizeof polys);
    memset(deg, 0, sizeof deg);
    int32_t ncons = 0;
    uint32_t osize = 1;
    for (int32_t k = 0; k < dim; k++) { deg[0][k] = 2; osize *= 3; }
    polys[0] = Q_arr(c, osize);
    tr_bigq **gr = qp_grad(c, q, xp);
    tr_bigq *q0 = qp_value_at(c, q, xp);
    if (polys[0] && LIVE(c)) {
        polys[0][0] = Q_copy(c, q0);
        uint32_t st[TR_POLYOPT_BB_VARS];
        uint32_t s1 = 1;
        for (int32_t k = 0; k < dim; k++) { st[k] = s1; s1 *= 3; }
        for (int32_t a = 0; a < dim && LIVE(c); a++) {
            tr_bigq *lin = NULL;                                  /* N_a^T (H xp + g) */
            for (int32_t j = 0; j < n && LIVE(c); j++) Q_fma(c, &lin, N[(uint64_t)a * n + j], gr[j]);
            polys[0][st[a]] = lin;
            for (int32_t b2 = a; b2 < dim && LIVE(c); b2++) {
                const tr_bigq *rab = R[(uint64_t)a * dim + b2];  /* N_a^T H N_b */
                if (Q_zero(rab)) continue;
                tr_bigq *hab = Q_copy(c, rab);
                uint32_t idx = st[a] + st[b2];
                if (a == b2) { tr_bigq *half = Q_q64(c, 1, 2); Q_set(&polys[0][idx], Q_mul(c, hab, half)); tr_bigq_free(half); tr_bigq_free(hab); }
                else polys[0][idx] = hab;                         /* (y_a y_b) appears twice in the quadratic form */
            }
        }
    }
    Q_free(gr, (uint64_t)n);
    tr_bigq_free(q0);
    for (int32_t i = 0; i < mi && LIVE(c); i++) {                /* (A N)_i y - bb_i <= 0 */
        int nz = 0;
        for (int32_t k = 0; k < dim; k++) if (!Q_zero(AN[(uint64_t)i * dim + k])) nz = 1;
        if (!nz) continue;                                        /* 0 <= bb_i holds at the feasible point */
        int32_t qi = 1 + ncons++;
        uint32_t sz = 1;
        for (int32_t k = 0; k < dim; k++) { deg[qi][k] = 1; sz *= 2; }
        polys[qi] = Q_arr(c, sz);
        if (!polys[qi]) break;
        polys[qi][0] = Q_neg(c, bb[i]);
        for (int32_t k = 0; k < dim && LIVE(c); k++) polys[qi][(uint32_t)1 << k] = Q_copy(c, AN[(uint64_t)i * dim + k]);
    }
    struct tr_polyopt_poly_s *P = LIVE(c) ? poly_from_dense(c, dim, 1 + ncons, ncons, polys,
                                                            (const int32_t (*)[TR_POLYOPT_BB_VARS])deg, ylo, yhi) : NULL;
    for (int32_t qi = 0; qi <= ncons; qi++) {
        uint32_t sz = 1;
        for (int32_t k = 0; k < dim; k++) sz *= (uint32_t)deg[qi][k] + 1;
        Q_free(polys[qi], sz);
    }
    tr_polyopt_result *br = P ? bb_run(c, P, tol, batch, fn, time_limit_ms) : NULL;
    if (br && LIVE(c)) {                                          /* the witness mapped back to x */
        uint64_t cnt = 0;
        const tr_bigq *const *yw = tr_polyopt_result_part(br, TR_POLYOPT_PART_POINT, &cnt);
        if (yw) {
            tr_bigq **xw = Q_full(c, (const tr_bigq *const *)xp, (uint64_t)n);
            for (int32_t k = 0; xw && k < dim && LIVE(c); k++)
                for (int32_t j = 0; j < n && LIVE(c); j++) Q_fma(c, &xw[j], N[(uint64_t)k * n + j], yw[k]);
            for (int32_t j = 0; xw && j < n && LIVE(c); j++) if (!xw[j]) xw[j] = Q_int(c, 0);
            if (xw && LIVE(c)) {                                  /* exactly feasible, with the reported value */
                tr_bigq *v = qp_value_at(c, q, xw);
                if (!verify_feasible(c, q, xw, NULL) || Q_cmp(v, br->part[TR_POLYOPT_PART_UPPER][0])) pfail(c, TR_MATH_INVALID);
                tr_bigq_free(v);
            }
            res_set(c, br, TR_POLYOPT_PART_POINT, xw, (uint64_t)n);
        }
        br->info.n = n; br->info.m = q->m; br->info.p = q->p; br->info.handed = 1; br->info.convex = 0;
        if (br->nleaves) {                                        /* leaves are boxes of y: not exported for x */
            Q_free(br->leaf_lo, br->nleaves * (uint64_t)dim); Q_free(br->leaf_hi, br->nleaves * (uint64_t)dim);
            free(br->leaf_con);
            br->leaf_lo = br->leaf_hi = NULL; br->leaf_con = NULL; br->nleaves = 0;
        }
        tr_polyopt_result_free(*out);
        *out = br;
    } else tr_polyopt_result_free(br);
    tr_polyopt_poly_free(P);
    Q_free(R, (uint64_t)dim * dim);
    for (int32_t k = 0; k < TR_POLYOPT_BB_VARS; k++) { tr_bigq_free(ylo[k]); tr_bigq_free(yhi[k]); }
    Q_free(AN, (uint64_t)(mi ? mi : 1) * dim); Q_free(bb, (uint64_t)(mi ? mi : 1));
    Q_free(xp, (uint64_t)n); Q_free(N, (uint64_t)dim * n);
}

tr_polyopt_result *tr_polyopt_qp_solve(const tr_polyopt_qp *q, int32_t flags, const tr_bigq *tolerance, uint64_t batch,
                                       tr_polyopt_boxes_fn fn, double time_limit_ms, const tr_math_budget *budget,
                                       tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    if (!qp_valid(q) || (flags & ~TR_POLYOPT_NO_UNIQUENESS) || (tolerance && Q_sgn(tolerance) < 0) ||
        !(time_limit_ms >= 0)) { set_error(error, TR_MATH_INVALID); return NULL; }
    pctx c;
    pctx_init(&c, budget, refusal);
    const int32_t n = q->n;
    tr_polyopt_result *r = res_new(&c, n);
    if (!r) { set_error(error, c.rc); return NULL; }
    r->info.m = q->m; r->info.p = q->p;
    int pd = 0;
    if (!q->hzero) {
        int32_t e = 0;
        tr_lin_result *ine = tr_lin_q_inertia(n, (const tr_bigq *const *)q->H, budget, refusal, &e);
        if (!ine) { pfail(&c, e); return res_done(&c, r, error); }
        tr_lin_result_info ii;
        tr_lin_result_get_info(ine, &ii);
        tr_lin_result_free(ine);
        if (ii.negative) {                                       /* not convex: certified branch and bound */
            r->info.convex = 0;
            handoff(&c, &r, q, tolerance, batch, fn, time_limit_ms);
            if (r) r->info.iterations = c.iters;
            return res_done(&c, r, error);
        }
        pd = ii.positive == n;
        if (2 * n > TR_LIN_MAX_UNKNOWNS + 1) { pfail(&c, TR_MATH_UNSUPPORTED); return res_done(&c, r, error); }
    }
    r->info.convex = 1;
    lp_out o;
    lp_core(&c, n, q->mi, (const tr_bigq *const *)q->A, (const tr_bigq *const *)q->b, q->p,
            (const tr_bigq *const *)q->E, (const tr_bigq *const *)q->e,
            q->hzero ? (const tr_bigq *const *)q->g : NULL, &o);
    if (LIVE(&c)) {
        if (o.status == TR_POLYOPT_INFEASIBLE) { r->info.method = TR_POLYOPT_SIMPLEX; result_farkas(&c, r, q, o.y, o.z); }
        else if (q->hzero) {
            r->info.method = TR_POLYOPT_SIMPLEX;
            if (o.status == TR_POLYOPT_UNBOUNDED) result_ray(&c, r, q, o.x, o.ray);
            else {
                result_optimum(&c, r, q, o.x, o.lambda, o.mu);
                if (o.strict) r->info.unique = 1;
                else if (!(flags & TR_POLYOPT_NO_UNIQUENESS)) uniqueness(&c, r, q, r->part[TR_POLYOPT_PART_POINT]);
            }
        } else {
            r->info.method = TR_POLYOPT_ACTIVE_SET;
            tr_bigq **x = Q_full(&c, (const tr_bigq *const *)o.x, (uint64_t)n);
            if (x) active_set(&c, r, q, x, o.active);
            if (LIVE(&c) && r->info.status == TR_POLYOPT_OPTIMAL) {
                if (pd) r->info.unique = 1;
                else if (!(flags & TR_POLYOPT_NO_UNIQUENESS)) uniqueness(&c, r, q, r->part[TR_POLYOPT_PART_POINT]);
            }
            Q_free(x, (uint64_t)n);
        }
    }
    lp_out_free(&o, n, q->mi, q->p);
    if (r) r->info.iterations = c.iters;
    return res_done(&c, r, error);
}

/* ================================================================ polynomial problems */
typedef struct {
    int32_t deg[TR_POLYOPT_BB_VARS];
    uint32_t size, stride[TR_POLYOPT_BB_VARS];
    tr_bigq **c;                             /* size exact coefficients (NULL = 0) */
} dpoly;
struct tr_polyopt_poly_s {
    uint32_t magic;
    int32_t n, ncons, neq, npolys;
    int32_t user[TR_POLYOPT_BB_CONS];        /* internal constraint -> the caller's index */
    dpoly P[TR_POLYOPT_BB_POLYS];
    tr_bigq *lo[TR_POLYOPT_BB_VARS], *hi[TR_POLYOPT_BB_VARS], *w[TR_POLYOPT_BB_VARS];
    double *clo, *chi;
    tr_polyopt_encl_view view;
};
static int poly_valid(const tr_polyopt_poly *P) { return P && P->magic == PP_MAGIC; }
void tr_polyopt_poly_free(tr_polyopt_poly *P) {
    if (!poly_valid(P)) return;
    for (int32_t q = 0; q < P->npolys; q++) Q_free(P->P[q].c, P->P[q].size);
    for (int32_t v = 0; v < TR_POLYOPT_BB_VARS; v++) { tr_bigq_free(P->lo[v]); tr_bigq_free(P->hi[v]); tr_bigq_free(P->w[v]); }
    free(P->clo); free(P->chi);
    P->magic = 0;
    free(P);
}
int32_t tr_polyopt_poly_get_view(const tr_polyopt_poly *P, tr_polyopt_encl_view *v) {
    if (!poly_valid(P) || !v) return TR_MATH_INVALID;
    *v = P->view;
    return TR_MATH_OK;
}
int32_t tr_polyopt_poly_dims(const tr_polyopt_poly *P, int32_t *n, int32_t *ncons, int32_t *neq) {
    if (!poly_valid(P)) return TR_MATH_INVALID;
    if (n) *n = P->n;
    if (ncons) *ncons = P->ncons;
    if (neq) *neq = P->neq;
    return TR_MATH_OK;
}
/* the enclosure arrays and the view of a problem whose dense polynomials and box are set */
static void poly_finish(pctx *c, tr_polyopt_poly *P) {
    uint32_t total = 0;
    for (int32_t q = 0; q < P->npolys; q++) total += P->P[q].size;
    P->clo = (double *)zalloc((uint64_t)total * 8);
    P->chi = (double *)zalloc((uint64_t)total * 8);
    if (!P->clo || !P->chi) { pfail(c, TR_MATH_MEMORY); return; }
    tr_polyopt_encl_view *v = &P->view;
    memset(v, 0, sizeof *v);
    v->n = P->n; v->npolys = P->npolys; v->ncons = P->ncons; v->neq = P->neq;
    uint32_t off = 0;
    for (int32_t q = 0; q < P->npolys && LIVE(c); q++) {
        v->offset[q] = off;
        for (int32_t i = 0; i < P->n; i++) v->deg[q][i] = (uint8_t)P->P[q].deg[i];
        for (uint32_t k = 0; k < P->P[q].size; k++) {
            tr_math_interval iv = {0.0, 0.0};
            if (!Q_zero(P->P[q].c[k])) tr_bigq_enclose(P->P[q].c[k], &iv);
            P->clo[off + k] = iv.lo; P->chi[off + k] = iv.hi;
        }
        off += P->P[q].size;
    }
    v->offset[P->npolys] = off;
    v->clo = P->clo; v->chi = P->chi;
    for (int32_t i = 0; i < P->n && LIVE(c); i++) P->w[i] = Q_sub(c, P->hi[i], P->lo[i]);
    if (LIVE(c) && !tr_polyopt_view_valid(v)) pfail(c, TR_MATH_INVALID);
}
static void dpoly_shape(dpoly *d, int32_t n) {
    uint32_t s = 1;
    for (int32_t i = 0; i < n; i++) { d->stride[i] = s; s *= (uint32_t)d->deg[i] + 1; }
    d->size = s;
}
/* takes the dense coefficient arrays coef[q] (ownership stays with the caller: copied) */
static struct tr_polyopt_poly_s *poly_from_dense(pctx *c, int32_t n, int32_t npolys, int32_t ncons, tr_bigq ***coef,
                                                 const int32_t (*deg)[TR_POLYOPT_BB_VARS], tr_bigq *const *lo,
                                                 tr_bigq *const *hi) {
    if (c->rc) return NULL;
    tr_polyopt_poly *P = (tr_polyopt_poly *)zalloc(sizeof *P);
    if (!P) { pfail(c, TR_MATH_MEMORY); return NULL; }
    P->magic = PP_MAGIC; P->n = n; P->npolys = npolys; P->ncons = ncons; P->neq = npolys - 1 - ncons;
    for (int32_t k = 0; k < ncons; k++) P->user[k] = k;
    for (int32_t q = 0; q < npolys && LIVE(c); q++) {
        for (int32_t i = 0; i < n; i++) P->P[q].deg[i] = deg[q][i];
        dpoly_shape(&P->P[q], n);
        if (P->P[q].size > TR_POLYOPT_DENSE_MAX) { pfail(c, TR_MATH_UNSUPPORTED); break; }
        P->P[q].c = Q_arr(c, P->P[q].size);
        for (uint32_t k = 0; P->P[q].c && k < P->P[q].size && LIVE(c); k++)
            if (!Q_zero(coef[q][k])) P->P[q].c[k] = Q_copy(c, coef[q][k]);
    }
    for (int32_t i = 0; i < n && LIVE(c); i++) { P->lo[i] = Q_copy(c, lo[i]); P->hi[i] = Q_copy(c, hi[i]); }
    if (LIVE(c)) poly_finish(c, P);
    if (c->rc) { tr_polyopt_poly_free(P); return NULL; }
    return P;
}
/* the terms of one spec: (exponents, coefficient) pairs via a callback-free two-pass reader */
typedef struct { uint8_t e[TR_POLYOPT_BB_VARS]; tr_bigq *c; } term_t;
static term_t *spec_terms(pctx *c, int32_t n, const int32_t *vars, const tr_polyopt_spec *s, int32_t *count) {
    *count = 0;
    if (c->rc) return NULL;
    if (s->slice) {
        int32_t nt = tr_math_poly_term_count(s->slice);
        if (nt < 0 || !vars) { pfail(c, TR_MATH_INVALID); return NULL; }
        term_t *t = (term_t *)zalloc((uint64_t)(nt ? nt : 1) * sizeof *t);
        if (!t) { pfail(c, TR_MATH_MEMORY); return NULL; }
        for (int32_t k = 0; k < nt && LIVE(c); k++) {
            tr_math_term mt;
            if (tr_math_poly_get_term(s->slice, k, &mt)) { pfail(c, TR_MATH_INVALID); break; }
            for (int32_t slot = 0; slot < TR_MATH_VARIABLES; slot++) {
                if (!mt.exponents[slot]) continue;
                int32_t v = -1;
                for (int32_t i = 0; i < n; i++) if (vars[i] == slot) v = i;
                if (v < 0) { pfail(c, TR_MATH_DOMAIN_ERROR); break; }
                t[k].e[v] = mt.exponents[slot];
            }
            t[k].c = Q_q64(c, mt.coefficient.numerator, mt.coefficient.denominator);
        }
        *count = nt;
        return t;
    }
    if (s->nterms < 0 || s->nterms > TR_POLYOPT_MAX_TERMS || (s->nterms && (!s->exps || !s->coef))) {
        pfail(c, s->nterms > TR_POLYOPT_MAX_TERMS ? TR_MATH_UNSUPPORTED : TR_MATH_INVALID);
        return NULL;
    }
    term_t *t = (term_t *)zalloc((uint64_t)(s->nterms ? s->nterms : 1) * sizeof *t);
    if (!t) { pfail(c, TR_MATH_MEMORY); return NULL; }
    for (int32_t k = 0; k < s->nterms && LIVE(c); k++) {
        if (!s->coef[k]) { pfail(c, TR_MATH_INVALID); break; }
        for (int32_t i = 0; i < n; i++) t[k].e[i] = s->exps[(uint64_t)k * n + i];
        t[k].c = Q_copy(c, s->coef[k]);
    }
    *count = s->nterms;
    return t;
}
static void terms_free(term_t *t, int32_t n) { for (int32_t k = 0; t && k < n; k++) tr_bigq_free(t[k].c); free(t); }
/* spec -> dense polynomial (sign: +1 or -1 applied, rhs subtracted before the sign) */
static void spec_dense(pctx *c, int32_t n, const int32_t *vars, const tr_polyopt_spec *s, int32_t sign, int rhs,
                       dpoly *d) {
    int32_t nt = 0;
    term_t *t = spec_terms(c, n, vars, s, &nt);
    if (c->rc) { terms_free(t, nt); return; }
    for (int32_t i = 0; i < n; i++) d->deg[i] = 0;
    for (int32_t k = 0; k < nt; k++) {
        if (Q_zero(t[k].c)) continue;
        for (int32_t i = 0; i < n; i++) if (t[k].e[i] > d->deg[i]) d->deg[i] = t[k].e[i];
    }
    for (int32_t i = 0; i < n; i++) if (d->deg[i] > TR_POLYOPT_MAX_DEGREE) pfail(c, TR_MATH_UNSUPPORTED);
    if (LIVE(c)) {
        uint64_t sz = 1;
        for (int32_t i = 0; i < n; i++) sz *= (uint64_t)d->deg[i] + 1;
        if (sz > TR_POLYOPT_DENSE_MAX) pfail(c, TR_MATH_UNSUPPORTED);
    }
    if (LIVE(c)) {
        dpoly_shape(d, n);
        d->c = Q_arr(c, d->size);
        for (int32_t k = 0; d->c && k < nt && LIVE(c); k++) {
            if (Q_zero(t[k].c)) continue;
            uint32_t idx = 0;
            for (int32_t i = 0; i < n; i++) idx += (uint32_t)t[k].e[i] * d->stride[i];
            Q_set(&d->c[idx], d->c[idx] ? Q_add(c, d->c[idx], t[k].c) : Q_copy(c, t[k].c));
        }
        if (rhs && s->rhs && d->c && LIVE(c)) Q_set(&d->c[0], Q_sub(c, d->c[0], s->rhs));
        if (sign < 0) for (uint32_t k = 0; d->c && k < d->size && LIVE(c); k++) if (d->c[k]) Q_set(&d->c[k], Q_neg(c, d->c[k]));
        for (uint32_t k = 0; d->c && k < d->size; k++) if (d->c[k] && Q_zero(d->c[k])) { tr_bigq_free(d->c[k]); d->c[k] = NULL; }
    }
    terms_free(t, nt);
}
tr_polyopt_poly *tr_polyopt_poly_create(int32_t n, const int32_t *vars, const tr_polyopt_spec *objective,
                                        int32_t ncons, const tr_polyopt_spec *cons, const tr_bigq *const *lo,
                                        const tr_bigq *const *hi, const tr_math_budget *budget,
                                        tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    if (n < 1 || !objective || ncons < 0 || (ncons && !cons) || !lo || !hi) { set_error(error, TR_MATH_INVALID); return NULL; }
    if (n > TR_POLYOPT_BB_VARS || ncons > TR_POLYOPT_BB_CONS) { set_error(error, TR_MATH_UNSUPPORTED); return NULL; }
    for (int32_t i = 0; i < n; i++) if (!lo[i] || !hi[i]) { set_error(error, TR_MATH_UNSUPPORTED); return NULL; }
    for (int32_t k = 0; k < ncons; k++)
        if (cons[k].kind < TR_POLYOPT_LE || cons[k].kind > TR_POLYOPT_EQ) { set_error(error, TR_MATH_INVALID); return NULL; }
    tr_math_need need;
    memset(&need, 0, sizeof need);
    need.need[TR_BUDGET_VARIABLES] = (uint64_t)n;
    uint64_t terms = 0;
    for (int32_t k = -1; k < ncons; k++) {
        const tr_polyopt_spec *s = k < 0 ? objective : &cons[k];
        terms += s->slice ? (uint64_t)(tr_math_poly_term_count(s->slice) > 0 ? tr_math_poly_term_count(s->slice) : 0)
                          : (uint64_t)(s->nterms > 0 ? s->nterms : 0);
    }
    need.need[TR_BUDGET_SUPPORT] = terms;
    need.need[TR_BUDGET_TENSOR] = (uint64_t)(ncons + 1) * TR_POLYOPT_DENSE_MAX;
    int32_t rc = budget_need(budget, refusal, &need);
    if (rc) { set_error(error, rc); return NULL; }
    pctx c;
    pctx_init(&c, budget, refusal);
    tr_polyopt_poly *P = (tr_polyopt_poly *)zalloc(sizeof *P);
    if (!P) { set_error(error, TR_MATH_MEMORY); return NULL; }
    P->magic = PP_MAGIC; P->n = n; P->npolys = 1 + ncons;
    spec_dense(&c, n, vars, objective, 1, 0, &P->P[0]);
    int32_t q = 1;
    for (int32_t pass = 0; pass < 2 && LIVE(&c); pass++)          /* inequalities first, then equalities */
        for (int32_t k = 0; k < ncons && LIVE(&c); k++) {
            int eq = cons[k].kind == TR_POLYOPT_EQ;
            if (eq != pass) continue;
            spec_dense(&c, n, vars, &cons[k], cons[k].kind == TR_POLYOPT_GE ? -1 : 1, 1, &P->P[q]);
            P->user[q - 1] = k;
            if (eq) P->neq++; else P->ncons++;
            q++;
        }
    int32_t maxdeg = 0;
    for (int32_t a = 0; a < P->npolys; a++)
        for (int32_t i = 0; i < n; i++) if (P->P[a].deg[i] > maxdeg) maxdeg = P->P[a].deg[i];
    if (LIVE(&c)) {
        need.need[TR_BUDGET_DEGREE] = (uint64_t)maxdeg;
        rc = budget_need(budget, refusal, &need);
        if (rc) pfail(&c, rc);
    }
    for (int32_t i = 0; i < n && LIVE(&c); i++) {
        P->lo[i] = Q_copy(&c, lo[i]); P->hi[i] = Q_copy(&c, hi[i]);
        if (LIVE(&c) && Q_cmp(P->lo[i], P->hi[i]) > 0) pfail(&c, TR_MATH_INFEASIBLE);
    }
    if (LIVE(&c)) poly_finish(&c, P);
    if (c.rc) { tr_polyopt_poly_free(P); set_error(error, c.rc); return NULL; }
    set_error(error, TR_MATH_OK);
    return P;
}
/* exact value of dense polynomial d at x */
static tr_bigq *dpoly_eval(pctx *c, const dpoly *d, int32_t n, tr_bigq *const *x) {
    tr_bigq *pw[TR_POLYOPT_BB_VARS][TR_POLYOPT_MAX_DEGREE + 1];
    memset(pw, 0, sizeof pw);
    for (int32_t i = 0; i < n && LIVE(c); i++) {
        pw[i][0] = Q_int(c, 1);
        for (int32_t e = 1; e <= d->deg[i] && LIVE(c); e++) pw[i][e] = Q_mul(c, pw[i][e - 1], x[i]);
    }
    tr_bigq *acc = NULL;
    for (uint32_t idx = 0; idx < d->size && LIVE(c); idx++) {
        if (Q_zero(d->c[idx])) continue;
        tr_bigq *t = Q_copy(c, d->c[idx]);
        for (int32_t i = 0; i < n && LIVE(c); i++) {
            uint32_t e = (idx / d->stride[i]) % ((uint32_t)d->deg[i] + 1);
            if (e) Q_set(&t, Q_mul(c, t, pw[i][e]));
        }
        if (!acc) acc = t;
        else { Q_set(&acc, Q_add(c, acc, t)); tr_bigq_free(t); }
    }
    for (int32_t i = 0; i < n; i++) for (int32_t e = 0; e <= TR_POLYOPT_MAX_DEGREE; e++) tr_bigq_free(pw[i][e]);
    return acc ? acc : Q_int(c, 0);
}
tr_bigq *tr_polyopt_poly_value(const tr_polyopt_poly *P, int32_t q, const tr_bigq *const *x, const tr_math_budget *budget,
                               tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    if (!poly_valid(P) || !x || q < 0 || q >= P->npolys) { set_error(error, TR_MATH_INVALID); return NULL; }
    for (int32_t i = 0; i < P->n; i++) if (!x[i]) { set_error(error, TR_MATH_INVALID); return NULL; }
    int32_t internal = q;
    if (q > 0) for (int32_t k = 0; k < P->npolys - 1; k++) if (P->user[k] == q - 1) internal = k + 1;
    pctx c;
    pctx_init(&c, budget, refusal);
    tr_bigq *v = dpoly_eval(&c, &P->P[internal], P->n, (tr_bigq *const *)x);
    if (c.rc) { tr_bigq_free(v); set_error(error, c.rc); return NULL; }
    set_error(error, TR_MATH_OK);
    return v;
}
int tr_polyopt_boxes(const tr_polyopt_encl_view *v, uint64_t nboxes, const double *xlo, const double *xhi,
                     double ubound, tr_polyopt_box_out *out, const tr_math_budget *budget, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!tr_polyopt_view_valid(v) || (nboxes && (!xlo || !xhi || !out)) || ubound != ubound) return TR_MATH_INVALID;
    for (uint64_t i = 0; i < nboxes * (uint64_t)v->n; i++) if (!(xlo[i] <= xhi[i])) return TR_MATH_INVALID;
    tr_math_need need;
    memset(&need, 0, sizeof need);
    need.need[TR_BUDGET_OUTPUT] = nboxes;
    int32_t rc = budget_need(budget, refusal, &need);
    if (rc) return rc;
    tr_math_interval S[TR_POLYOPT_DENSE_MAX];
    for (uint64_t i = 0; i < nboxes; i++)
        tr_polyopt_box(v, xlo + i * (uint64_t)v->n, xhi + i * (uint64_t)v->n, ubound, S, &out[i]);
    return TR_MATH_OK;
}

/* ================================================================ certified branch and bound */
typedef struct {
    double flo, fhi;
    uint64_t seq;
    uint64_t k[TR_POLYOPT_BB_VARS];
    uint8_t j[TR_POLYOPT_BB_VARS];
    int32_t inner;
} bnode;                                     /* 72 bytes */
typedef struct { bnode *a; uint64_t n, cap; } heap_t;
static int node_less(const bnode *x, const bnode *y) { return x->flo < y->flo || (x->flo == y->flo && x->seq < y->seq); }
static int heap_push(heap_t *h, const bnode *x) {
    if (h->n == h->cap) {
        uint64_t cap = h->cap ? 2 * h->cap : 64;
        bnode *na = (bnode *)realloc(h->a, (size_t)cap * sizeof *na);
        if (!na) return 0;
        h->a = na; h->cap = cap;
    }
    uint64_t i = h->n++;
    h->a[i] = *x;
    while (i && node_less(&h->a[i], &h->a[(i - 1) / 2])) {
        bnode t = h->a[i]; h->a[i] = h->a[(i - 1) / 2]; h->a[(i - 1) / 2] = t;
        i = (i - 1) / 2;
    }
    return 1;
}
static bnode heap_pop(heap_t *h) {
    bnode top = h->a[0];
    h->a[0] = h->a[--h->n];
    uint64_t i = 0;
    for (;;) {
        uint64_t l = 2 * i + 1, r = l + 1, m = i;
        if (l < h->n && node_less(&h->a[l], &h->a[m])) m = l;
        if (r < h->n && node_less(&h->a[r], &h->a[m])) m = r;
        if (m == i) break;
        bnode t = h->a[i]; h->a[i] = h->a[m]; h->a[m] = t;
        i = m;
    }
    return top;
}
/* lo0 + w0 num / 2^e exactly */
static tr_bigq *box_coord(pctx *c, const tr_polyopt_poly *P, int32_t v, uint64_t num, uint32_t e) {
    if (c->rc) return NULL;
    int32_t err = 0;
    uint64_t one = 1;
    tr_bigint *bn = num ? tr_bigint_from_limbs(1, &num, 1, c->b, c->r, &err) : tr_bigint_from_i64(0, c->b, c->r, &err);
    tr_bigint *b1 = bn ? tr_bigint_from_limbs(1, &one, 1, c->b, c->r, &err) : NULL;
    tr_bigint *bd = b1 ? tr_bigint_shl(b1, e, c->b, c->r, &err) : NULL;
    tr_bigq *f = bd ? tr_bigq_from_bigints(bn, bd, c->b, c->r, &err) : NULL;
    tr_bigint_free(bn); tr_bigint_free(b1); tr_bigint_free(bd);
    if (!f) { pfail(c, err); return NULL; }
    tr_bigq *t = Q_mul(c, P->w[v], f), *x = Q_add(c, P->lo[v], t);
    tr_bigq_free(f); tr_bigq_free(t);
    return x;
}
/* the double enclosure of a node's box */
static void node_box(pctx *c, const tr_polyopt_poly *P, const bnode *b, double *xlo, double *xhi) {
    for (int32_t v = 0; v < P->n && LIVE(c); v++) {
        tr_bigq *lo = box_coord(c, P, v, b->k[v], b->j[v]), *hi = box_coord(c, P, v, b->k[v] + 1, b->j[v]);
        tr_math_interval a = {-INFINITY, INFINITY}, z = {-INFINITY, INFINITY};
        if (LIVE(c)) { tr_bigq_enclose(lo, &a); tr_bigq_enclose(hi, &z); }
        xlo[v] = a.lo; xhi[v] = z.hi;
        tr_bigq_free(lo); tr_bigq_free(hi);
    }
}
static double now_ms(void) {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}
/* the exact point given by per-variable numerators over 2^e (e.g. a midpoint 2k + 1 over 2^(j + 1)) */
static tr_bigq **node_point(pctx *c, const tr_polyopt_poly *P, const uint64_t *num, const uint32_t *e) {
    tr_bigq **x = Q_arr(c, (uint64_t)P->n);
    for (int32_t v = 0; x && v < P->n && LIVE(c); v++) x[v] = box_coord(c, P, v, num[v], e[v]);
    return x;
}
/* 1 when x satisfies every constraint exactly */
static int point_feasible(pctx *c, const tr_polyopt_poly *P, tr_bigq *const *x) {
    for (int32_t q = 1; q < P->npolys && LIVE(c); q++) {
        tr_bigq *g = dpoly_eval(c, &P->P[q], P->n, x);
        int s = Q_sgn(g);
        tr_bigq_free(g);
        if (q > P->ncons ? s != 0 : s > 0) return 0;
    }
    return LIVE(c);
}
typedef struct {
    const tr_polyopt_poly *P;
    tr_bigq *U;                              /* the incumbent value (NULL: none) */
    tr_bigq **witness;
    tr_bigq *tol, *Utol;                     /* U - tol */
    double ldisc, lunsplit;                  /* smallest lower bound of the discarded / unsplittable boxes */
    heap_t open;
    bnode *leaves; int32_t *leafcon;
    uint64_t nleaves, capleaves, seq;
    uint64_t pruned_inf, pruned_bound, pruned_tol, incumbents;
} bbstate;
static void bb_incumbent(pctx *c, bbstate *s, tr_bigq **x) {
    tr_bigq *v = dpoly_eval(c, &s->P->P[0], s->P->n, x);
    if (LIVE(c) && (!s->U || Q_cmp(v, s->U) < 0)) {
        tr_bigq_free(s->U); s->U = v; v = NULL;
        Q_free(s->witness, (uint64_t)s->P->n); s->witness = x; x = NULL;
        tr_bigq_free(s->Utol); s->Utol = Q_sub(c, s->U, s->tol);
        s->incumbents++;
    }
    tr_bigq_free(v);
    Q_free(x, x ? (uint64_t)s->P->n : 0);
}
/* 1 when the double d >= the rational q (exact) */
static int dge(const tr_bigq *q, double d) {
    int32_t out = 0;
    if (d == INFINITY) return 1;
    if (d == -INFINITY) return 0;
    tr_bigq_cmp_double(q, d, &out);
    return out <= 0;
}
static int dgt(const tr_bigq *q, double d) {
    int32_t out = 0;
    if (d == INFINITY) return 1;
    if (d == -INFINITY) return 0;
    tr_bigq_cmp_double(q, d, &out);
    return out < 0;
}
static void bb_process(pctx *c, bbstate *s, const bnode *nd, const tr_polyopt_box_out *o) {
    const tr_polyopt_poly *P = s->P;
    if (o->status == TR_POLYOPT_BOX_INFEASIBLE) {
        s->pruned_inf++;
        if (s->nleaves == s->capleaves) {
            uint64_t cap = s->capleaves ? 2 * s->capleaves : 64;
            bnode *nl = (bnode *)realloc(s->leaves, (size_t)cap * sizeof *nl);
            int32_t *nc = (int32_t *)realloc(s->leafcon, (size_t)cap * sizeof *nc);
            if (nl) s->leaves = nl;
            if (nc) s->leafcon = nc;
            if (!nl || !nc) { pfail(c, TR_MATH_MEMORY); return; }
            s->capleaves = cap;
        }
        s->leaves[s->nleaves] = *nd;
        s->leafcon[s->nleaves++] = P->user[o->violated];
        return;
    }
    if (o->status == TR_POLYOPT_BOX_BOUND || (s->U && dgt(s->U, o->flo))) { s->pruned_bound++; return; }
    if (s->U && dge(s->Utol, o->flo)) {                          /* cannot improve beyond the tolerance */
        if (o->flo < s->ldisc) s->ldisc = o->flo;
        s->pruned_tol++;
        return;
    }
    uint64_t num[TR_POLYOPT_BB_VARS];
    uint32_t e[TR_POLYOPT_BB_VARS];
    for (int32_t v = 0; v < P->n; v++) { num[v] = 2 * nd->k[v] + 1; e[v] = (uint32_t)nd->j[v] + 1; }
    tr_bigq **mid = node_point(c, P, num, e);
    if (mid && LIVE(c) && (o->inner || point_feasible(c, P, mid))) bb_incumbent(c, s, mid);
    else Q_free(mid, mid ? (uint64_t)P->n : 0);
    bnode b = *nd;
    b.flo = o->flo; b.fhi = o->fhi; b.inner = o->inner; b.seq = s->seq++;
    if (LIVE(c) && !heap_push(&s->open, &b)) pfail(c, TR_MATH_MEMORY);
}
static tr_polyopt_result *bb_univariate(pctx *c, const tr_polyopt_poly *P, const tr_bigq *tol);
static tr_polyopt_result *bb_run(pctx *c, const tr_polyopt_poly *P, const tr_bigq *tol, uint64_t batch,
                                 tr_polyopt_boxes_fn fn, double time_limit_ms) {
    if (c->rc) return NULL;
    if (P->n == 1 && P->npolys == 1) return bb_univariate(c, P, tol);
    const int32_t n = P->n;
    if (!batch) batch = 32;
    if (!fn) fn = tr_polyopt_boxes;
    bbstate s;
    memset(&s, 0, sizeof s);
    s.P = P; s.ldisc = INFINITY; s.lunsplit = INFINITY;
    s.tol = tol ? Q_copy(c, tol) : Q_int(c, 0);
    double t0 = now_ms();
    tr_polyopt_result *r = res_new(c, n);
    if (!r) { tr_bigq_free(s.tol); return NULL; }
    r->info.method = TR_POLYOPT_BRANCH_BOUND;
    r->info.m = P->ncons; r->info.p = P->neq;
    uint64_t cap = 2 * batch;
    double *xlo = (double *)zalloc(cap * n * 8), *xhi = (double *)zalloc(cap * n * 8);
    tr_polyopt_box_out *out = (tr_polyopt_box_out *)zalloc(cap * sizeof *out);
    bnode *kids = (bnode *)zalloc(cap * sizeof *kids);
    if (!xlo || !xhi || !out || !kids) pfail(c, TR_MATH_MEMORY);
    /* initial incumbents: the 2^n corners of the box */
    for (uint32_t m = 0; m < (1u << n) && LIVE(c); m++) {
        uint64_t num[TR_POLYOPT_BB_VARS];
        uint32_t e[TR_POLYOPT_BB_VARS];
        for (int32_t v = 0; v < n; v++) { num[v] = (m >> v) & 1; e[v] = 0; }
        tr_bigq **x = node_point(c, P, num, e);
        if (x && LIVE(c) && point_feasible(c, P, x)) bb_incumbent(c, &s, x);
        else Q_free(x, x ? (uint64_t)n : 0);
    }
    int32_t reason = TR_POLYOPT_REASON_NONE, complete = 0;
    uint64_t evaluated = 0;
    int root_done = 0;
    if (LIVE(c) && c->iters < c->limit) {                        /* the root box */
        memset(&kids[0], 0, sizeof kids[0]);
        node_box(c, P, &kids[0], xlo, xhi);
        double ub = INFINITY;
        if (s.U && LIVE(c)) { tr_math_interval iv; tr_bigq_enclose(s.U, &iv); ub = iv.hi; }
        int32_t rc = LIVE(c) ? fn(&P->view, 1, xlo, xhi, ub, out, c->b, c->r) : 0;
        if (rc) pfail(c, rc);
        c->iters++;
        evaluated = 1;
        root_done = 1;
        if (LIVE(c)) bb_process(c, &s, &kids[0], &out[0]);
    }
    if (!root_done) reason = TR_POLYOPT_REASON_NODES;
    while (LIVE(c) && s.open.n) {
        double L = s.open.a[0].flo;
        if (s.ldisc < L) L = s.ldisc;
        if (s.lunsplit < L) L = s.lunsplit;
        if (s.U && dge(s.Utol, L)) { complete = 1; break; }       /* U - L <= tol */
        if (time_limit_ms > 0 && now_ms() - t0 > time_limit_ms) { reason = TR_POLYOPT_REASON_TIME; break; }
        uint64_t room = c->limit > c->iters ? c->limit - c->iters : 0;
        uint64_t take = batch < room / 2 ? batch : room / 2;
        if (!take) { reason = TR_POLYOPT_REASON_NODES; break; }
        uint64_t nk = 0;
        for (uint64_t t = 0; t < take && s.open.n && LIVE(c); t++) {
            bnode b = heap_pop(&s.open);
            if (s.U && dgt(s.U, b.flo)) { s.pruned_bound++; continue; }
            int32_t sv = -1;
            for (int32_t v = 0; v < n; v++) if (b.j[v] < TR_POLYOPT_DEPTH && (sv < 0 || b.j[v] < b.j[sv])) sv = v;
            if (sv < 0) { if (b.flo < s.lunsplit) s.lunsplit = b.flo; continue; }
            bnode k0 = b, k1 = b;
            k0.j[sv]++; k0.k[sv] = 2 * b.k[sv];
            k1.j[sv]++; k1.k[sv] = 2 * b.k[sv] + 1;
            kids[nk++] = k0; kids[nk++] = k1;
        }
        if (!nk || !LIVE(c)) continue;
        for (uint64_t i = 0; i < nk && LIVE(c); i++) node_box(c, P, &kids[i], xlo + i * n, xhi + i * n);
        double ub = INFINITY;
        if (s.U && LIVE(c)) { tr_math_interval iv; tr_bigq_enclose(s.U, &iv); ub = iv.hi; }
        c->iters += nk;
        evaluated += nk;
        r->info.batches++;
        int32_t rc = LIVE(c) ? fn(&P->view, nk, xlo, xhi, ub, out, c->b, c->r) : 0;
        if (rc) pfail(c, rc);
        for (uint64_t i = 0; i < nk && LIVE(c); i++) bb_process(c, &s, &kids[i], &out[i]);
    }
    if (LIVE(c)) {
        double L = s.open.n ? s.open.a[0].flo : INFINITY;
        if (s.ldisc < L) L = s.ldisc;
        if (s.lunsplit < L) L = s.lunsplit;
        if (!root_done) L = -INFINITY;                            /* nothing was examined: no bound */
        if (root_done && !s.open.n && s.lunsplit == INFINITY) complete = 1;
        if (!complete && reason == TR_POLYOPT_REASON_NONE) reason = s.lunsplit < INFINITY ? TR_POLYOPT_REASON_DEPTH : TR_POLYOPT_REASON_NODES;
        tr_bigq *lower = NULL;
        if (L == -INFINITY) lower = NULL;
        else if (L == INFINITY) lower = s.U ? Q_copy(c, s.U) : NULL;
        else {
            lower = Q_double(c, L);
            if (s.U && lower && Q_cmp(lower, s.U) > 0) Q_set(&lower, Q_copy(c, s.U));
        }
        if (s.U) {
            r->info.status = complete ? TR_POLYOPT_COMPLETE : TR_POLYOPT_INCOMPLETE;
            r->info.feasibility = TR_POLYOPT_FEASIBLE;
            res_set(c, r, TR_POLYOPT_PART_POINT, s.witness, (uint64_t)n); s.witness = NULL;
            res_one(c, r, TR_POLYOPT_PART_VALUE, s.U);
            res_one(c, r, TR_POLYOPT_PART_UPPER, s.U);
            if (lower) res_one(c, r, TR_POLYOPT_PART_LOWER, lower);

        } else if (complete) {                                    /* every box was proven infeasible */
            r->info.status = TR_POLYOPT_INFEASIBLE;
            r->info.feasibility = TR_POLYOPT_NOT_FEASIBLE;
            r->info.certificate = 1;

            tr_math_need need;
            memset(&need, 0, sizeof need);
            need.need[TR_BUDGET_OUTPUT] = s.nleaves;
            int32_t rc = budget_need(c->b, c->r, &need);
            if (rc) pfail(c, rc);
            r->leaf_lo = Q_arr(c, s.nleaves * n); r->leaf_hi = Q_arr(c, s.nleaves * n);
            r->leaf_con = (int32_t *)zalloc((s.nleaves ? s.nleaves : 1) * 4);
            if (!r->leaf_con) pfail(c, TR_MATH_MEMORY);
            for (uint64_t i = 0; i < s.nleaves && LIVE(c); i++) {
                for (int32_t v = 0; v < n && LIVE(c); v++) {
                    r->leaf_lo[i * n + v] = box_coord(c, P, v, s.leaves[i].k[v], s.leaves[i].j[v]);
                    r->leaf_hi[i * n + v] = box_coord(c, P, v, s.leaves[i].k[v] + 1, s.leaves[i].j[v]);
                }
                r->leaf_con[i] = s.leafcon[i];
            }
            r->nleaves = s.nleaves;
        } else {
            r->info.status = TR_POLYOPT_INCOMPLETE;
            r->info.feasibility = TR_POLYOPT_UNKNOWN;
            if (lower) res_one(c, r, TR_POLYOPT_PART_LOWER, lower);
        }
        if (!complete) r->info.reason = reason;
        res_one(c, r, TR_POLYOPT_PART_TOLERANCE, s.tol);
        tr_bigq_free(lower);
        r->info.iterations = evaluated;
        r->info.pruned_infeasible = s.pruned_inf; r->info.pruned_bound = s.pruned_bound;
        r->info.pruned_tolerance = s.pruned_tol; r->info.open = s.open.n + (s.lunsplit < INFINITY);
        r->info.leaves = s.nleaves; r->info.incumbents = s.incumbents;
        r->info.unique = -1;
    }
    free(xlo); free(xhi); free(out); free(kids);
    free(s.open.a); free(s.leaves); free(s.leafcon);
    tr_bigq_free(s.U); tr_bigq_free(s.tol); tr_bigq_free(s.Utol);
    Q_free(s.witness, s.witness ? (uint64_t)n : 0);
    if (c->rc) { tr_polyopt_result_free(r); return NULL; }
    return r;
}

/* ---- univariate problems without constraints: INV-001's roots of f' ---- */
/* a feasible candidate x: U = min(U, f(x)) with its witness */
static void uni_cand(pctx *c, const dpoly *f, const tr_bigq *x, tr_bigq **U, tr_bigq **wit) {
    tr_bigq *xx[1] = {(tr_bigq *)x};
    tr_bigq *v = dpoly_eval(c, f, 1, xx);
    if (LIVE(c) && (!*U || Q_cmp(v, *U) < 0)) { tr_bigq_free(*U); *U = v; Q_set(wit, Q_copy(c, x)); return; }
    tr_bigq_free(v);
}
/* the exact dyadic lower bound of f over [lo, hi] from its enclosure (NULL: no finite bound) */
static tr_bigq *uni_lower(pctx *c, const tr_polyopt_poly *P, const tr_bigq *lo, const tr_bigq *hi) {
    tr_math_interval a, b, X, S[TR_POLYOPT_DENSE_MAX];
    tr_bigq_enclose(lo, &a);
    tr_bigq_enclose(hi, &b);
    X.lo = a.lo; X.hi = b.hi;
    tr_math_interval E = tr_polyopt_enclose(&P->view, 0, &X, S);
    return E.lo == -INFINITY || E.lo != E.lo ? NULL : Q_double(c, E.lo);
}
static tr_polyopt_result *bb_univariate(pctx *c, const tr_polyopt_poly *P, const tr_bigq *tol) {
    const dpoly *f = &P->P[0];
    const int32_t D = f->deg[0];
    tr_polyopt_result *r = res_new(c, 1);
    if (!r) return NULL;
    r->info.method = TR_POLYOPT_ROOTS;
    tr_bigq *tolq = tol ? Q_copy(c, tol) : Q_int(c, 0);
    tr_bigq *U = NULL, *L = NULL, *wit = NULL;
    uni_cand(c, f, P->lo[0], &U, &wit);                          /* the bounds are candidates */
    uni_cand(c, f, P->hi[0], &U, &wit);
    tr_inv_roots *roots = NULL;
    if (D >= 2 && LIVE(c)) {                                     /* the critical points: roots of f' */
        tr_bigq **df = Q_arr(c, (uint64_t)D);
        for (int32_t i = 1; df && i <= D && LIVE(c); i++) {
            tr_bigq *k = Q_int(c, i);
            df[i - 1] = Q_mul(c, f->c[i], k);
            tr_bigq_free(k);
        }
        int32_t e = 0;
        if (LIVE(c)) roots = tr_inv_isolate(D, (const tr_bigq *const *)df, NULL, P->lo[0], P->hi[0], c->b, c->r, &e);
        if (!roots && LIVE(c)) pfail(c, e);
        Q_free(df, (uint64_t)D);
    }
    uint64_t rounds = 0;
    int complete = 0;
    while (LIVE(c)) {
        tr_inv_roots_info ri;
        memset(&ri, 0, sizeof ri);
        if (roots) tr_inv_roots_get_info(roots, &ri);
        for (int32_t i = 0; i < ri.nroots && LIVE(c); i++) {     /* exact critical points, interval endpoints */
            tr_inv_root_info inf;
            const tr_bigq *lo = NULL, *hi = NULL;
            tr_inv_roots_root(roots, i, &inf, &lo, &hi);
            uni_cand(c, f, lo, &U, &wit);
            if (!inf.exact) uni_cand(c, f, hi, &U, &wit);
        }
        tr_bigq_free(L);
        L = Q_copy(c, U);
        tr_bigq *ut = Q_sub(c, U, tolq);
        double maxw = 0;
        int pending = 0, unbounded = 0;
        for (int32_t i = 0; i < ri.nroots && LIVE(c); i++) {     /* f at an inexact critical point: enclosed */
            tr_inv_root_info inf;
            const tr_bigq *lo = NULL, *hi = NULL;
            tr_inv_roots_root(roots, i, &inf, &lo, &hi);
            if (inf.exact) continue;
            tr_bigq *el = uni_lower(c, P, lo, hi);
            if (!el) unbounded = 1;
            else if (Q_cmp(el, L) < 0) Q_set(&L, Q_copy(c, el));
            if (!el || Q_cmp(el, ut) < 0) {
                pending = 1;
                tr_bigq *wd = Q_sub(c, hi, lo);
                tr_math_interval wi = {0.0, 0.0};
                if (wd) tr_bigq_enclose(wd, &wi);
                if (wi.hi > maxw) maxw = wi.hi;
                tr_bigq_free(wd);
            }
            tr_bigq_free(el);
        }
        tr_bigq_free(ut);
        if (unbounded) { tr_bigq_free(L); L = NULL; }
        if (!pending) { complete = 1; break; }
        if (c->iters >= c->limit) break;                          /* the iteration budget: incomplete */
        c->iters++;
        tr_bigq *w = Q_double(c, maxw / 4);
        int32_t e = 0;
        tr_inv_roots *ref = (w && Q_sgn(w) > 0 && LIVE(c)) ? tr_inv_roots_refine(roots, w, NULL, c->b, c->r, &e) : NULL;
        tr_bigq_free(w);
        if (!ref) { if (LIVE(c) && e) pfail(c, e); break; }
        tr_inv_roots_free(roots);
        roots = ref;
        rounds++;
    }
    if (LIVE(c) && U) {
        r->info.status = complete ? TR_POLYOPT_COMPLETE : TR_POLYOPT_INCOMPLETE;
        r->info.feasibility = TR_POLYOPT_FEASIBLE;

        if (!complete) r->info.reason = TR_POLYOPT_REASON_NODES;
        tr_bigq **wv = Q_arr(c, 1);
        if (wv) { wv[0] = wit; wit = NULL; }
        res_set(c, r, TR_POLYOPT_PART_POINT, wv, 1);
        res_one(c, r, TR_POLYOPT_PART_VALUE, U);
        res_one(c, r, TR_POLYOPT_PART_UPPER, U);
        if (L) res_one(c, r, TR_POLYOPT_PART_LOWER, L);
        res_one(c, r, TR_POLYOPT_PART_TOLERANCE, tolq);
        r->info.iterations = rounds;
        r->info.unique = -1;
    }
    tr_inv_roots_free(roots);
    tr_bigq_free(U); tr_bigq_free(L); tr_bigq_free(tolq); tr_bigq_free(wit);
    if (c->rc) { tr_polyopt_result_free(r); return NULL; }
    return r;
}

tr_polyopt_result *tr_polyopt_minimize(const tr_polyopt_poly *P, const tr_bigq *tolerance, uint64_t batch,
                                       tr_polyopt_boxes_fn fn, double time_limit_ms, const tr_math_budget *budget,
                                       tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    if (!poly_valid(P) || (tolerance && Q_sgn(tolerance) < 0) || !(time_limit_ms >= 0) || batch > ((uint64_t)1 << 20)) {
        set_error(error, TR_MATH_INVALID);
        return NULL;
    }
    pctx c;
    pctx_init(&c, budget, refusal);
    tr_polyopt_result *r = bb_run(&c, P, tolerance, batch, fn, time_limit_ms);
    return res_done(&c, r, error);
}
