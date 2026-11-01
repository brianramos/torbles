/* INV-002: constrained minimal interventions; see math_interv.h.
 * SPDX-License-Identifier: Apache-2.0 */
#include "math_interv.h"
#include "math_handles.h"
#include <stdlib.h>
#include <string.h>

TR_MATH_ABI(tr_interv_linear_problem, 1);   /* the ctypes boundary of INV-002 (math_handles.h) */
TR_MATH_ABI(tr_interv_univariate_problem, 1);
TR_MATH_ABI(tr_interv_grid_problem, 1);
TR_MATH_ABI(tr_interv_grid_view, 1);
TR_MATH_ABI(tr_interv_grid_info, 1);
TR_MATH_ABI(tr_interv_info, 1);
TR_MATH_ABI(tr_interv_solution, 1);
TR_MATH_ABI(tr_interv_agg_info, 1);
TR_MATH_ABI(tr_interv_agg_query, 1);

_Static_assert(sizeof(tr_interv_linear_problem) == 152, "tr_interv_linear_problem ABI");
_Static_assert(sizeof(tr_interv_univariate_problem) == 104, "tr_interv_univariate_problem ABI");
_Static_assert(sizeof(tr_interv_grid_problem) == 144, "tr_interv_grid_problem ABI");
_Static_assert(sizeof(tr_interv_grid_view) == 320, "tr_interv_grid_view ABI");
_Static_assert(sizeof(tr_interv_grid_info) == 72, "tr_interv_grid_info ABI");
_Static_assert(sizeof(tr_interv_info) == 112, "tr_interv_info ABI");
_Static_assert(sizeof(tr_interv_solution) == 56, "tr_interv_solution ABI");
_Static_assert(sizeof(tr_interv_agg_info) == 56, "tr_interv_agg_info ABI");
_Static_assert(sizeof(tr_interv_agg_query) == 72, "tr_interv_agg_query ABI");

#define RES_MAGIC UINT32_C(0x56524e49)
#define PLAN_MAGIC UINT32_C(0x50524e49)
#define AGG_MAGIC UINT32_C(0x41524e49)

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
    uint64_t iters, limit;
} ictx;
#define LIVE(c) (!(c)->rc)
static void ifail(ictx *c, int32_t rc) { if (!c->rc) c->rc = rc ? rc : TR_MATH_MEMORY; }
static void ictx_init(ictx *c, const tr_math_budget *b, tr_math_refusal *r) {
    c->b = b; c->r = r; c->rc = 0; c->reserved = 0; c->iters = 0;
    c->limit = b ? b->limit[TR_BUDGET_ITERATIONS] : ((uint64_t)1 << 20);
}
static tr_bigq *qk(ictx *c, tr_bigq *v, int32_t e) { if (!v) ifail(c, e); return v; }
static tr_bigq *Q_int(ictx *c, int64_t v) {
    tr_num_rational q; q.numerator = v; q.denominator = 1;
    int32_t e = 0;
    return c->rc ? NULL : qk(c, tr_bigq_from_q64(q, c->b, c->r, &e), e);
}
static tr_bigq *Q_q64(ictx *c, int64_t n, int64_t d) {
    tr_num_rational q; q.numerator = n; q.denominator = d;
    int32_t e = 0;
    return c->rc ? NULL : qk(c, tr_bigq_from_q64(q, c->b, c->r, &e), e);
}
static int Q_zero(const tr_bigq *a) { return !a || tr_bigq_sign(a) == 0; }
static int Q_sgn(const tr_bigq *a) { return a ? tr_bigq_sign(a) : 0; }
static int Q_cmp(const tr_bigq *a, const tr_bigq *b) {
    if (!a && !b) return 0;
    if (!a) return -tr_bigq_sign(b);
    if (!b) return tr_bigq_sign(a);
    return tr_bigq_cmp(a, b);
}
static tr_bigq *Q_copy(ictx *c, const tr_bigq *a) {
    int32_t e = 0;
    if (c->rc) return NULL;
    if (!a) return Q_int(c, 0);
    return qk(c, tr_bigq_copy(a, c->b, c->r, &e), e);
}
static tr_bigq *Q_add(ictx *c, const tr_bigq *a, const tr_bigq *b) {
    int32_t e = 0;
    if (c->rc) return NULL;
    if (Q_zero(a)) return Q_copy(c, b);
    if (Q_zero(b)) return Q_copy(c, a);
    return qk(c, tr_bigq_add(a, b, c->b, c->r, &e), e);
}
static tr_bigq *Q_neg(ictx *c, const tr_bigq *a) {
    int32_t e = 0;
    if (c->rc) return NULL;
    if (Q_zero(a)) return Q_int(c, 0);
    return qk(c, tr_bigq_neg(a, c->b, c->r, &e), e);
}
static tr_bigq *Q_sub(ictx *c, const tr_bigq *a, const tr_bigq *b) {
    int32_t e = 0;
    if (c->rc) return NULL;
    if (Q_zero(b)) return Q_copy(c, a);
    if (Q_zero(a)) return Q_neg(c, b);
    return qk(c, tr_bigq_sub(a, b, c->b, c->r, &e), e);
}
static tr_bigq *Q_mul(ictx *c, const tr_bigq *a, const tr_bigq *b) {
    int32_t e = 0;
    if (c->rc) return NULL;
    if (Q_zero(a) || Q_zero(b)) return Q_int(c, 0);
    return qk(c, tr_bigq_mul(a, b, c->b, c->r, &e), e);
}
static tr_bigq *Q_div(ictx *c, const tr_bigq *a, const tr_bigq *b) {
    int32_t e = 0;
    if (c->rc) return NULL;
    if (Q_zero(b)) { ifail(c, TR_MATH_INVALID); return NULL; }
    if (Q_zero(a)) return Q_int(c, 0);
    return qk(c, tr_bigq_div(a, b, c->b, c->r, &e), e);
}
static tr_bigq *Q_abs(ictx *c, const tr_bigq *a) { return Q_sgn(a) < 0 ? Q_neg(c, a) : Q_copy(c, a); }
static tr_bigq *Q_mid(ictx *c, const tr_bigq *a, const tr_bigq *b) {
    tr_bigq *s = Q_add(c, a, b), *h = Q_q64(c, 1, 2), *m = Q_mul(c, s, h);
    tr_bigq_free(s); tr_bigq_free(h);
    return m;
}
static void Q_set(tr_bigq **slot, tr_bigq *v) { if (v) { tr_bigq_free(*slot); *slot = v; } }
static void Q_fma(ictx *c, tr_bigq **acc, const tr_bigq *a, const tr_bigq *b) {
    if (c->rc || Q_zero(a) || Q_zero(b)) return;
    tr_bigq *p = Q_mul(c, a, b);
    if (!p) return;
    if (!*acc) { *acc = p; return; }
    tr_bigq *s = Q_add(c, *acc, p);
    tr_bigq_free(p);
    if (s) { tr_bigq_free(*acc); *acc = s; }
}
static tr_bigq **Q_arr(ictx *c, uint64_t n) {
    if (c->rc) return NULL;
    tr_bigq **v = (tr_bigq **)zalloc(n * sizeof *v);
    if (!v) ifail(c, TR_MATH_MEMORY);
    return v;
}
static void Q_free(tr_bigq **v, uint64_t n) {
    if (!v) return;
    for (uint64_t i = 0; i < n; i++) tr_bigq_free(v[i]);
    free(v);
}
static tr_bigq **Q_full(ictx *c, const tr_bigq *const *v, uint64_t n) {
    tr_bigq **o = Q_arr(c, n ? n : 1);
    for (uint64_t i = 0; o && i < n && LIVE(c); i++) o[i] = Q_copy(c, v ? v[i] : NULL);
    return o;
}
static tr_bigq *Q_dot(ictx *c, const tr_bigq *const *a, const tr_bigq *const *b, int32_t n) {
    tr_bigq *acc = NULL;
    for (int32_t i = 0; i < n && LIVE(c); i++) Q_fma(c, &acc, a[i], b[i]);
    return acc ? acc : Q_int(c, 0);
}
static int budget_need(const tr_math_budget *b, tr_math_refusal *r, const tr_math_need *need) {
    tr_math_budget dflt;
    if (!b) { tr_budget_default(&dflt); b = &dflt; }
    return tr_budget_check(b, need, r);
}
/* the comparison of v (sign of f - t) with op */
static int cmp_holds(int32_t op, int s) {
    switch (op) {
    case TR_INTERV_EQ: return s == 0;
    case TR_INTERV_NE: return s != 0;
    case TR_INTERV_LT: return s < 0;
    case TR_INTERV_LE: return s <= 0;
    case TR_INTERV_GT: return s > 0;
    case TR_INTERV_GE: return s >= 0;
    default: return 0;
    }
}
static int32_t cmp_complement(int32_t op) {
    switch (op) {
    case TR_INTERV_EQ: return TR_INTERV_NE;
    case TR_INTERV_NE: return TR_INTERV_EQ;
    case TR_INTERV_LT: return TR_INTERV_GE;
    case TR_INTERV_LE: return TR_INTERV_GT;
    case TR_INTERV_GT: return TR_INTERV_LE;
    default: return TR_INTERV_LT;
    }
}
static int op_valid(int32_t op) { return op >= TR_INTERV_EQ && op <= TR_INTERV_GE; }
static int cost_valid(int32_t cost) { return cost >= TR_INTERV_L1 && cost <= TR_INTERV_L0; }
/* the exact cost of a one-coordinate change |d| (w the weight) */
static tr_bigq *cost_scalar(ictx *c, int32_t cost, const tr_bigq *w, const tr_bigq *d) {
    if (cost == TR_INTERV_L0) return Q_zero(d) ? Q_int(c, 0) : Q_copy(c, w);
    tr_bigq *a = Q_abs(c, d), *v = cost == TR_INTERV_L2 ? Q_mul(c, a, a) : Q_copy(c, a);
    tr_bigq *r = Q_mul(c, w, v);
    tr_bigq_free(a); tr_bigq_free(v);
    return r;
}
/* the exact cost of a change d (n coordinates, weights w) */
static tr_bigq *cost_vector(ictx *c, int32_t cost, int32_t n, tr_bigq *const *w, tr_bigq *const *d) {
    tr_bigq *acc = Q_int(c, 0);
    for (int32_t i = 0; i < n && LIVE(c); i++) {
        tr_bigq *t = cost_scalar(c, cost, w[i], d[i]);
        if (cost == TR_INTERV_LINF) { if (Q_cmp(t, acc) > 0) { tr_bigq_free(acc); acc = t; t = NULL; } }
        else Q_set(&acc, Q_add(c, acc, t));
        tr_bigq_free(t);
    }
    return acc;
}

/* ================================================================ results */
typedef struct {
    tr_interv_solution s;
    tr_bigq **dlo, **dhi, **xlo, **xhi;      /* n each */
    tr_bigq *clo, *chi, *vlo, *vhi;
    uint64_t support;
} sol_t;
struct tr_interv_result_s {
    uint32_t magic;
    tr_interv_info info;
    sol_t *sol;
    int32_t nsol, cap;
    tr_bigq **part[TR_INTERV_PARTS];
    uint64_t npart[TR_INTERV_PARTS];
};
static int res_valid(const tr_interv_result *r) { return r && r->magic == RES_MAGIC; }
static void sol_free(sol_t *s, int32_t n) {
    Q_free(s->dlo, (uint64_t)n); Q_free(s->dhi, (uint64_t)n); Q_free(s->xlo, (uint64_t)n); Q_free(s->xhi, (uint64_t)n);
    tr_bigq_free(s->clo); tr_bigq_free(s->chi); tr_bigq_free(s->vlo); tr_bigq_free(s->vhi);
    memset(s, 0, sizeof *s);
}
void tr_interv_result_free(tr_interv_result *r) {
    if (!res_valid(r)) return;
    for (int32_t i = 0; i < r->nsol; i++) sol_free(&r->sol[i], r->info.n);
    free(r->sol);
    for (int p = 0; p < TR_INTERV_PARTS; p++) Q_free(r->part[p], r->npart[p]);
    r->magic = 0;
    free(r);
}
static tr_interv_result *res_new(ictx *c, int32_t method, int32_t n, int32_t cost) {
    if (c->rc) return NULL;
    tr_interv_result *r = (tr_interv_result *)zalloc(sizeof *r);
    if (!r) { ifail(c, TR_MATH_MEMORY); return NULL; }
    r->magic = RES_MAGIC;
    r->info.method = method; r->info.n = n; r->info.cost = cost;
    r->info.claim = TR_INTERV_CLAIM_ALGEBRAIC;
    r->info.unique = -1;
    r->info.exact = 1;
    return r;
}
/* a new solution slot (zeroed) */
static sol_t *res_add(ictx *c, tr_interv_result *r) {
    if (c->rc || !r) return NULL;
    if (r->nsol == r->cap) {
        int32_t cap = r->cap ? 2 * r->cap : 4;
        sol_t *ns = (sol_t *)realloc(r->sol, (size_t)cap * sizeof *ns);
        if (!ns) { ifail(c, TR_MATH_MEMORY); return NULL; }
        r->sol = ns; r->cap = cap;
    }
    sol_t *s = &r->sol[r->nsol++];
    memset(s, 0, sizeof *s);
    r->info.nsolutions = r->nsol;
    return s;
}
static void res_part(ictx *c, tr_interv_result *r, int32_t part, tr_bigq **v, uint64_t n) {
    if (!r || !v) { Q_free(v, v ? n : 0); return; }
    for (uint64_t i = 0; i < n && LIVE(c); i++) if (!v[i]) v[i] = Q_int(c, 0);
    Q_free(r->part[part], r->npart[part]);
    r->part[part] = v; r->npart[part] = n;
}
static void res_one(ictx *c, tr_interv_result *r, int32_t part, const tr_bigq *v) {
    if (!v) return;
    tr_bigq **a = Q_arr(c, 1);
    if (a) a[0] = Q_copy(c, v);
    res_part(c, r, part, a, 1);
}
static tr_interv_result *res_done(ictx *c, tr_interv_result *r, int32_t *error) {
    if (c->rc) { tr_interv_result_free(r); set_error(error, c->rc); return NULL; }
    if (!r) { set_error(error, TR_MATH_MEMORY); return NULL; }
    for (int32_t i = 0; i < r->nsol; i++) if (!r->sol[i].s.exact) r->info.exact = 0;
    set_error(error, TR_MATH_OK);
    return r;
}
int32_t tr_interv_result_get_info(const tr_interv_result *r, tr_interv_info *out) {
    if (!res_valid(r) || !out) return TR_MATH_INVALID;
    *out = r->info;
    return TR_MATH_OK;
}
int32_t tr_interv_result_solution(const tr_interv_result *r, int32_t i, tr_interv_solution *out, const tr_bigq **dlo,
                                  const tr_bigq **dhi, const tr_bigq **xlo, const tr_bigq **xhi, const tr_bigq **clo,
                                  const tr_bigq **chi, const tr_bigq **vlo, const tr_bigq **vhi) {
    if (!res_valid(r) || i < 0 || i >= r->nsol || !out) return TR_MATH_INVALID;
    const sol_t *s = &r->sol[i];
    *out = s->s;
    for (int32_t k = 0; k < r->info.n; k++) {
        if (dlo) dlo[k] = s->dlo ? s->dlo[k] : NULL;
        if (dhi) dhi[k] = s->dhi ? s->dhi[k] : NULL;
        if (xlo) xlo[k] = s->xlo ? s->xlo[k] : NULL;
        if (xhi) xhi[k] = s->xhi ? s->xhi[k] : NULL;
    }
    if (clo) *clo = s->clo;
    if (chi) *chi = s->chi;
    if (vlo) *vlo = s->vlo;
    if (vhi) *vhi = s->vhi;
    return TR_MATH_OK;
}
uint64_t tr_interv_result_support(const tr_interv_result *r, int32_t i) {
    return res_valid(r) && i >= 0 && i < r->nsol ? r->sol[i].support : 0;
}
const tr_bigq *const *tr_interv_result_part(const tr_interv_result *r, int32_t part, uint64_t *count) {
    if (count) *count = 0;
    if (!res_valid(r) || part < 0 || part >= TR_INTERV_PARTS || !r->part[part]) return NULL;
    if (count) *count = r->npart[part];
    return (const tr_bigq *const *)r->part[part];
}
/* an exact solution: d (owned, n), x = x0 + d, cost, value (owned or NULL) */
static sol_t *sol_exact(ictx *c, tr_interv_result *r, int32_t n, tr_bigq **d, tr_bigq *const *x0, tr_bigq *cost,
                        tr_bigq *value) {
    sol_t *s = res_add(c, r);
    if (!s) { Q_free(d, (uint64_t)n); tr_bigq_free(cost); tr_bigq_free(value); return NULL; }
    s->s.exact = 1; s->s.attained = 1;
    s->dlo = d;
    s->dhi = Q_full(c, (const tr_bigq *const *)d, (uint64_t)n);
    s->xlo = Q_arr(c, (uint64_t)n);
    for (int32_t i = 0; s->xlo && i < n && LIVE(c); i++) s->xlo[i] = Q_add(c, x0[i], d[i]);
    s->xhi = Q_full(c, (const tr_bigq *const *)s->xlo, (uint64_t)n);
    s->clo = cost; s->chi = Q_copy(c, cost);
    s->vlo = value; s->vhi = value ? Q_copy(c, value) : NULL;
    return s;
}

/* ================================================================ 1. linear targets */
/* the problem in the change d: rows A d <= b, E d = e, box lo <= d <= hi, ||d||_1 <= l1, target a^T d op r */
typedef struct {
    int32_t n, cost, m, p;
    tr_bigq **w, **A, **b, **E, **e, **lo, **hi, **a, **x0;
    tr_bigq *l1, *r;
} dprob;
static void dprob_free(dprob *P) {
    const uint64_t n = (uint64_t)P->n;
    Q_free(P->w, n); Q_free(P->A, (uint64_t)P->m * n); Q_free(P->b, (uint64_t)P->m);
    Q_free(P->E, (uint64_t)P->p * n); Q_free(P->e, (uint64_t)P->p);
    Q_free(P->lo, n); Q_free(P->hi, n); Q_free(P->a, n); Q_free(P->x0, n);
    tr_bigq_free(P->l1); tr_bigq_free(P->r);
    memset(P, 0, sizeof *P);
}
/* an LP / QP under construction (rows on nv variables; NULL entries 0) */
typedef struct {
    int32_t nv, m, p, mcap, pcap;
    tr_bigq **A, **b, **E, **e, **lo, **hi, **H, **g;
} lpb;
static void lpb_free(lpb *L) {
    const uint64_t nv = (uint64_t)L->nv;
    Q_free(L->A, (uint64_t)L->mcap * nv); Q_free(L->b, (uint64_t)L->mcap);
    Q_free(L->E, (uint64_t)L->pcap * nv); Q_free(L->e, (uint64_t)L->pcap);
    Q_free(L->lo, nv); Q_free(L->hi, nv); Q_free(L->H, nv * nv); Q_free(L->g, nv);
    memset(L, 0, sizeof *L);
}
static void lpb_init(ictx *c, lpb *L, int32_t nv) {
    memset(L, 0, sizeof *L);
    L->nv = nv;
    L->lo = Q_arr(c, (uint64_t)nv); L->hi = Q_arr(c, (uint64_t)nv); L->g = Q_arr(c, (uint64_t)nv);
}
/* a row sum coef[k] v[col[k]] (<= rhs, or = rhs when eq) */
static void lpb_row(ictx *c, lpb *L, int eq, int32_t k, const int32_t *col, tr_bigq *const *coef, const tr_bigq *rhs) {
    if (c->rc) return;
    int32_t *cnt = eq ? &L->p : &L->m, *cap = eq ? &L->pcap : &L->mcap;
    tr_bigq ***M = eq ? &L->E : &L->A, ***R = eq ? &L->e : &L->b;
    if (*cnt == *cap) {
        int32_t nc = *cap ? 2 * *cap : 16;
        tr_bigq **nm = (tr_bigq **)zalloc((uint64_t)nc * L->nv * sizeof *nm), **nr = (tr_bigq **)zalloc((uint64_t)nc * sizeof *nr);
        if (!nm || !nr) { free(nm); free(nr); ifail(c, TR_MATH_MEMORY); return; }
        if (*M) memcpy(nm, *M, (size_t)*cap * L->nv * sizeof *nm);
        if (*R) memcpy(nr, *R, (size_t)*cap * sizeof *nr);
        free(*M); free(*R);
        *M = nm; *R = nr; *cap = nc;
    }
    tr_bigq **row = *M + (uint64_t)*cnt * L->nv;
    for (int32_t j = 0; j < k && LIVE(c); j++) {
        if (Q_zero(coef[j])) continue;
        Q_set(&row[col[j]], row[col[j]] ? Q_add(c, row[col[j]], coef[j]) : Q_copy(c, coef[j]));
    }
    (*R)[*cnt] = Q_copy(c, rhs);
    (*cnt)++;
}
static tr_polyopt_result *lpb_solve(ictx *c, lpb *L, int32_t flags) {
    if (c->rc) return NULL;
    int32_t e = 0;
    tr_polyopt_qp *q = tr_polyopt_qp_create(L->nv, (const tr_bigq *const *)L->H, (const tr_bigq *const *)L->g, NULL,
                                            L->m, (const tr_bigq *const *)L->A, (const tr_bigq *const *)L->b, L->p,
                                            (const tr_bigq *const *)L->E, (const tr_bigq *const *)L->e,
                                            (const tr_bigq *const *)L->lo, (const tr_bigq *const *)L->hi, c->b, c->r, &e);
    if (!q) { ifail(c, e); return NULL; }
    tr_polyopt_result *res = tr_polyopt_qp_solve(q, flags, NULL, 0, NULL, 0, c->b, c->r, &e);
    tr_polyopt_qp_free(q);
    if (!res) ifail(c, e);
    else {
        tr_polyopt_info pi;
        tr_polyopt_result_get_info(res, &pi);
        c->iters += pi.iterations;
    }
    return res;
}
/* the variables: d at 0..n-1, u (|d| bounds) at n..2n-1 when the L1 cost or the L1 ball needs them, s (the LINF
 * level) last */
typedef struct { int32_t nv, u, s; } layout_t;
static layout_t layout_of(const dprob *P, int32_t cost) {
    layout_t L;
    L.nv = P->n; L.u = -1; L.s = -1;
    if (cost == TR_INTERV_L1 || P->l1) { L.u = L.nv; L.nv += P->n; }
    if (cost == TR_INTERV_LINF) { L.s = L.nv; L.nv += 1; }
    return L;
}
enum { T_NONE = -1 };
/* the constraints of the change and the target (op closed: EQ, LE, GE or T_NONE: no target row) */
static void build_rows(ictx *c, const dprob *P, int32_t cost, int32_t op, const layout_t *Y, lpb *L) {
    const int32_t n = P->n;
    int32_t *col = (int32_t *)zalloc((uint64_t)(2 * n + 2) * 4);
    tr_bigq **coef = Q_arr(c, (uint64_t)(2 * n + 2));
    if (!col || !coef) { free(col); Q_free(coef, (uint64_t)(2 * n + 2)); ifail(c, TR_MATH_MEMORY); return; }
    for (int32_t i = 0; i < n; i++) col[i] = i;
    for (int32_t i = 0; i < P->m && LIVE(c); i++)
        lpb_row(c, L, 0, n, col, P->A + (uint64_t)i * n, P->b[i]);
    for (int32_t i = 0; i < P->p && LIVE(c); i++)
        lpb_row(c, L, 1, n, col, P->E + (uint64_t)i * n, P->e[i]);
    for (int32_t i = 0; i < n && LIVE(c); i++) {                 /* the box on d */
        if (P->lo[i]) L->lo[i] = Q_copy(c, P->lo[i]);
        if (P->hi[i]) L->hi[i] = Q_copy(c, P->hi[i]);
    }
    if (op == TR_INTERV_EQ || op == TR_INTERV_LE) lpb_row(c, L, op == TR_INTERV_EQ, n, col, P->a, P->r);
    else if (op == TR_INTERV_GE) {
        tr_bigq **na = Q_arr(c, (uint64_t)n);
        for (int32_t i = 0; na && i < n && LIVE(c); i++) na[i] = Q_neg(c, P->a[i]);
        tr_bigq *nr = Q_neg(c, P->r);
        lpb_row(c, L, 0, n, col, na, nr);
        Q_free(na, (uint64_t)n); tr_bigq_free(nr);
    }
    if (Y->u >= 0) {                                             /* u_i >= d_i, u_i >= -d_i */
        tr_bigq *one = Q_int(c, 1), *mone = Q_int(c, -1), *zero = Q_int(c, 0);
        for (int32_t i = 0; i < n && LIVE(c); i++) {
            int32_t cc[2] = {i, Y->u + i};
            tr_bigq *k1[2] = {one, mone}, *k2[2] = {mone, mone};
            lpb_row(c, L, 0, 2, cc, k1, zero);
            lpb_row(c, L, 0, 2, cc, k2, zero);
        }
        if (P->l1) {
            for (int32_t i = 0; i < n; i++) { col[i] = Y->u + i; coef[i] = one; }
            lpb_row(c, L, 0, n, col, coef, P->l1);
            for (int32_t i = 0; i < n; i++) { col[i] = i; coef[i] = NULL; }
        }
        tr_bigq_free(one); tr_bigq_free(mone); tr_bigq_free(zero);
    }
    if (Y->s >= 0) {                                             /* +-w_i d_i <= s */
        tr_bigq *mone = Q_int(c, -1), *zero = Q_int(c, 0);
        for (int32_t i = 0; i < n && LIVE(c); i++) {
            if (Q_zero(P->w[i])) continue;
            int32_t cc[2] = {i, Y->s};
            tr_bigq *nw = Q_neg(c, P->w[i]);
            tr_bigq *k1[2] = {P->w[i], mone}, *k2[2] = {nw, mone};
            lpb_row(c, L, 0, 2, cc, k1, zero);
            lpb_row(c, L, 0, 2, cc, k2, zero);
            tr_bigq_free(nw);
        }
        tr_bigq_free(mone); tr_bigq_free(zero);
    }
    free(col);
    for (int32_t i = 0; i < 2 * n + 2; i++) coef[i] = NULL;
    Q_free(coef, (uint64_t)(2 * n + 2));
}
/* the objective of cost on the layout */
static void build_cost(ictx *c, const dprob *P, int32_t cost, const layout_t *Y, lpb *L) {
    const int32_t n = P->n;
    if (cost == TR_INTERV_L1) for (int32_t i = 0; i < n && LIVE(c); i++) L->g[Y->u + i] = Q_copy(c, P->w[i]);
    else if (cost == TR_INTERV_LINF) L->g[Y->s] = Q_int(c, 1);
    else if (cost == TR_INTERV_L2) {
        L->H = Q_arr(c, (uint64_t)L->nv * L->nv);
        tr_bigq *two = Q_int(c, 2);
        for (int32_t i = 0; L->H && i < n && LIVE(c); i++) L->H[(uint64_t)i * L->nv + i] = Q_mul(c, two, P->w[i]);
        tr_bigq_free(two);
    }
}
/* the rows of the optimal set {cost(d) <= c*} (L2: d_i = d*_i where w_i > 0) */
static void build_optset(ictx *c, const dprob *P, int32_t cost, const layout_t *Y, lpb *L, const tr_bigq *cstar,
                         tr_bigq *const *dstar) {
    const int32_t n = P->n;
    if (cost == TR_INTERV_L1) {
        int32_t *col = (int32_t *)zalloc((uint64_t)(n ? n : 1) * 4);
        if (!col) { ifail(c, TR_MATH_MEMORY); return; }
        for (int32_t i = 0; i < n; i++) col[i] = Y->u + i;
        lpb_row(c, L, 0, n, col, P->w, cstar);
        free(col);
    } else if (cost == TR_INTERV_LINF) {
        tr_bigq *one = Q_int(c, 1);
        int32_t cc[1] = {Y->s};
        tr_bigq *k[1] = {one};
        lpb_row(c, L, 0, 1, cc, k, cstar);
        tr_bigq_free(one);
    } else {
        tr_bigq *one = Q_int(c, 1);
        for (int32_t i = 0; i < n && LIVE(c); i++) {
            if (Q_zero(P->w[i])) continue;
            int32_t cc[1] = {i};
            tr_bigq *k[1] = {one};
            lpb_row(c, L, 1, 1, cc, k, dstar[i]);
        }
        tr_bigq_free(one);
    }
}
static tr_bigq **res_d(ictx *c, const tr_polyopt_result *R, int32_t n) {
    uint64_t cnt = 0;
    const tr_bigq *const *x = tr_polyopt_result_part(R, TR_POLYOPT_PART_POINT, &cnt);
    if (!x || cnt < (uint64_t)n) { ifail(c, TR_MATH_INVALID); return NULL; }
    return Q_full(c, x, (uint64_t)n);
}
static int r_status(const tr_polyopt_result *R) { tr_polyopt_info i; tr_polyopt_result_get_info(R, &i); return i.status; }
/* min cost over the closure (op EQ / LE / GE): status, *cstar, *dstar */
static int32_t solve_closed(ictx *c, const dprob *P, int32_t cost, int32_t op, tr_bigq **cstar, tr_bigq ***dstar) {
    layout_t Y = layout_of(P, cost);
    lpb L;
    lpb_init(c, &L, Y.nv);
    build_rows(c, P, cost, op, &Y, &L);
    build_cost(c, P, cost, &Y, &L);
    tr_polyopt_result *R = lpb_solve(c, &L, TR_POLYOPT_NO_UNIQUENESS);
    lpb_free(&L);
    int32_t st = R ? r_status(R) : 0;
    if (R && st == TR_POLYOPT_OPTIMAL) {
        *dstar = res_d(c, R, P->n);
        *cstar = *dstar ? cost_vector(c, cost, P->n, P->w, *dstar) : NULL;
    } else if (R && st == TR_POLYOPT_UNBOUNDED) ifail(c, TR_MATH_INVALID);   /* a cost >= 0 is bounded below */
    tr_polyopt_result_free(R);
    return LIVE(c) ? st : 0;
}
/* max sgn a^T d over the constraints (with the target closure and the optimal set when cstar is given): status;
 * the maximum in *vmax (NULL when unbounded) and the maximizer in *at (owned) */
static int32_t max_target(ictx *c, const dprob *P, int32_t cost, int sgn, int32_t closed, const tr_bigq *cstar,
                          tr_bigq *const *dstar, tr_bigq **vmax, tr_bigq ***at) {
    layout_t Y = cstar ? layout_of(P, cost) : layout_of(P, TR_INTERV_L2);
    lpb L;
    lpb_init(c, &L, Y.nv);
    build_rows(c, P, cost, closed, &Y, &L);
    if (cstar) build_optset(c, P, cost, &Y, &L, cstar, dstar);
    for (int32_t i = 0; i < P->n && LIVE(c); i++) L.g[i] = sgn > 0 ? Q_neg(c, P->a[i]) : Q_copy(c, P->a[i]);
    tr_polyopt_result *R = lpb_solve(c, &L, TR_POLYOPT_NO_UNIQUENESS);
    lpb_free(&L);
    int32_t st = R ? r_status(R) : 0;
    *vmax = NULL; *at = NULL;
    if (R && st == TR_POLYOPT_OPTIMAL) {
        *at = res_d(c, R, P->n);
        *vmax = *at ? Q_dot(c, (const tr_bigq *const *)P->a, (const tr_bigq *const *)*at, P->n) : NULL;
        if (*vmax && sgn < 0) Q_set(vmax, Q_neg(c, *vmax));
    } else if (R && st == TR_POLYOPT_UNBOUNDED) {                /* a feasible point far along the ray */
        uint64_t cnt = 0;
        const tr_bigq *const *x0 = tr_polyopt_result_part(R, TR_POLYOPT_PART_ANCHOR, &cnt);
        const tr_bigq *const *dd = tr_polyopt_result_part(R, TR_POLYOPT_PART_DIRECTION, &cnt);
        tr_bigq *ax = Q_dot(c, (const tr_bigq *const *)P->a, x0, P->n), *ad = Q_dot(c, (const tr_bigq *const *)P->a, dd, P->n);
        if (sgn < 0) { Q_set(&ax, Q_neg(c, ax)); Q_set(&ad, Q_neg(c, ad)); }
        /* sgn a^T (x0 + k d) > sgn r for k = (|sgn r - sgn a^T x0| / (sgn a^T d)) + 1 */
        tr_bigq *sr = sgn > 0 ? Q_copy(c, P->r) : Q_neg(c, P->r), *gap = Q_sub(c, sr, ax), *ag = Q_abs(c, gap);
        tr_bigq *k = Q_div(c, ag, ad), *one = Q_int(c, 1);
        Q_set(&k, Q_add(c, k, one));
        *at = Q_arr(c, (uint64_t)P->n);
        for (int32_t i = 0; *at && i < P->n && LIVE(c); i++) {
            tr_bigq *t = Q_mul(c, k, dd[i]);
            (*at)[i] = Q_add(c, x0[i], t);
            tr_bigq_free(t);
        }
        tr_bigq_free(ax); tr_bigq_free(ad); tr_bigq_free(sr); tr_bigq_free(gap); tr_bigq_free(ag); tr_bigq_free(k);
        tr_bigq_free(one);
    }
    tr_polyopt_result_free(R);
    return LIVE(c) ? st : 0;
}
/* one candidate of a linear problem */
typedef struct { int32_t status; tr_bigq *cost; tr_bigq **d; } cand_t;
static void cand_free(cand_t *k, int32_t n) { tr_bigq_free(k->cost); Q_free(k->d, k->d ? (uint64_t)n : 0); memset(k, 0, sizeof *k); }
/* the strict target sgn (a^T d - r) > 0 (GT: sgn 1, LT: sgn -1) */
static void solve_strict(ictx *c, const dprob *P, int32_t cost, int sgn, cand_t *out) {
    memset(out, 0, sizeof *out);
    out->status = TR_INTERV_INFEASIBLE;
    tr_bigq *vmax = NULL, **at = NULL;
    int32_t st = max_target(c, P, cost, sgn, T_NONE, NULL, NULL, &vmax, &at);
    tr_bigq *sr = sgn > 0 ? Q_copy(c, P->r) : Q_neg(c, P->r);
    int nonempty = st == TR_POLYOPT_UNBOUNDED || (st == TR_POLYOPT_OPTIMAL && Q_cmp(vmax, sr) > 0);
    tr_bigq_free(vmax); Q_free(at, at ? (uint64_t)P->n : 0);
    if (!nonempty || !LIVE(c)) { tr_bigq_free(sr); return; }
    tr_bigq *cstar = NULL, **dstar = NULL;
    st = solve_closed(c, P, cost, sgn > 0 ? TR_INTERV_GE : TR_INTERV_LE, &cstar, &dstar);
    if (st != TR_POLYOPT_OPTIMAL || !LIVE(c)) { tr_bigq_free(sr); tr_bigq_free(cstar); Q_free(dstar, dstar ? (uint64_t)P->n : 0); return; }
    tr_bigq *ad = Q_dot(c, (const tr_bigq *const *)P->a, (const tr_bigq *const *)dstar, P->n);
    if (sgn < 0) Q_set(&ad, Q_neg(c, ad));
    if (Q_cmp(ad, sr) > 0) { out->status = TR_INTERV_OPTIMAL; out->cost = cstar; out->d = dstar; }
    else {
        st = max_target(c, P, cost, sgn, sgn > 0 ? TR_INTERV_GE : TR_INTERV_LE, cstar, dstar, &vmax, &at);
        if (st == TR_POLYOPT_UNBOUNDED || (st == TR_POLYOPT_OPTIMAL && Q_cmp(vmax, sr) > 0)) {
            out->status = TR_INTERV_OPTIMAL; out->cost = cstar; out->d = at; at = NULL;
            Q_free(dstar, (uint64_t)P->n);
        } else { out->status = TR_INTERV_UNATTAINED; out->cost = cstar; out->d = dstar; }
        tr_bigq_free(vmax); Q_free(at, at ? (uint64_t)P->n : 0);
    }
    tr_bigq_free(ad); tr_bigq_free(sr);
}
/* uniqueness of an optimum over its optimal set (closed target op): 1, or 0 with a second optimal change *face */
static int unique_change(ictx *c, const dprob *P, int32_t cost, int32_t closed, const tr_bigq *cstar,
                         tr_bigq *const *dstar, tr_bigq ***face) {
    const int32_t n = P->n;
    *face = NULL;
    for (int32_t k = 0; k < n && LIVE(c); k++)
        for (int32_t s = -1; s <= 1 && LIVE(c); s += 2) {
            layout_t Y = layout_of(P, cost);
            lpb L;
            lpb_init(c, &L, Y.nv);
            build_rows(c, P, cost, closed, &Y, &L);
            build_optset(c, P, cost, &Y, &L, cstar, dstar);
            L.g[k] = Q_int(c, s);                                 /* s = -1: max d_k; 1: min d_k */
            tr_polyopt_result *R = lpb_solve(c, &L, TR_POLYOPT_NO_UNIQUENESS);
            lpb_free(&L);
            int32_t st = R ? r_status(R) : 0;
            if (R && st == TR_POLYOPT_UNBOUNDED) {
                uint64_t cnt = 0;
                const tr_bigq *const *x0 = tr_polyopt_result_part(R, TR_POLYOPT_PART_ANCHOR, &cnt);
                const tr_bigq *const *dd = tr_polyopt_result_part(R, TR_POLYOPT_PART_DIRECTION, &cnt);
                tr_bigq **f = Q_arr(c, (uint64_t)n);
                for (int32_t i = 0; f && i < n && LIVE(c); i++) f[i] = Q_add(c, x0[i], dd[i]);
                *face = f;
            } else if (R && st == TR_POLYOPT_OPTIMAL) {
                tr_bigq **x = res_d(c, R, n);
                if (x && Q_cmp(x[k], dstar[k])) *face = x;
                else Q_free(x, x ? (uint64_t)n : 0);
            } else if (R && LIVE(c)) ifail(c, TR_MATH_INVALID);  /* d* is in the optimal set */
            tr_polyopt_result_free(R);
            if (*face) return 0;
        }
    return 1;
}
/* the shared front end: the problem in d */
static void make_dprob(ictx *c, const tr_interv_linear_problem *pr, tr_bigq *const *a, dprob *P) {
    const int32_t n = pr->n;
    memset(P, 0, sizeof *P);
    P->n = n; P->cost = pr->cost; P->m = pr->m; P->p = pr->p;
    P->x0 = Q_full(c, pr->x0, (uint64_t)n);
    P->w = Q_arr(c, (uint64_t)n);
    for (int32_t i = 0; P->w && i < n && LIVE(c); i++) {
        P->w[i] = pr->weight && pr->weight[i] ? Q_copy(c, pr->weight[i]) : Q_int(c, 1);
        if (Q_sgn(P->w[i]) < 0 || (pr->cost == TR_INTERV_L0 && Q_sgn(P->w[i]) == 0)) ifail(c, TR_MATH_INVALID);
    }
    P->a = Q_full(c, (const tr_bigq *const *)a, (uint64_t)n);
    P->A = Q_full(c, pr->A, (uint64_t)pr->m * n);
    P->b = Q_arr(c, (uint64_t)(pr->m ? pr->m : 1));
    for (int32_t i = 0; P->b && i < pr->m && LIVE(c); i++) {     /* A (x0 + d) <= b: A d <= b - A x0 */
        tr_bigq *ax = Q_dot(c, pr->A + (uint64_t)i * n, (const tr_bigq *const *)P->x0, n);
        P->b[i] = Q_sub(c, pr->b[i], ax);
        tr_bigq_free(ax);
    }
    P->E = Q_full(c, pr->E, (uint64_t)pr->p * n);
    P->e = Q_arr(c, (uint64_t)(pr->p ? pr->p : 1));
    for (int32_t i = 0; P->e && i < pr->p && LIVE(c); i++) {
        tr_bigq *ex = Q_dot(c, pr->E + (uint64_t)i * n, (const tr_bigq *const *)P->x0, n);
        P->e[i] = Q_sub(c, pr->e[i], ex);
        tr_bigq_free(ex);
    }
    P->lo = Q_arr(c, (uint64_t)n); P->hi = Q_arr(c, (uint64_t)n);
    for (int32_t i = 0; P->lo && P->hi && i < n && LIVE(c); i++) {      /* the boxes on x and d, and the Linf ball */
        tr_bigq *lo = NULL, *hi = NULL;
        if (pr->xlo && pr->xlo[i]) lo = Q_sub(c, pr->xlo[i], P->x0[i]);
        if (pr->xhi && pr->xhi[i]) hi = Q_sub(c, pr->xhi[i], P->x0[i]);
        if (pr->dlo && pr->dlo[i] && (!lo || Q_cmp(pr->dlo[i], lo) > 0)) Q_set(&lo, Q_copy(c, pr->dlo[i]));
        if (pr->dhi && pr->dhi[i] && (!hi || Q_cmp(pr->dhi[i], hi) < 0)) Q_set(&hi, Q_copy(c, pr->dhi[i]));
        if (pr->linf) {
            tr_bigq *nl = Q_neg(c, pr->linf);
            if (!lo || Q_cmp(nl, lo) > 0) Q_set(&lo, nl); else tr_bigq_free(nl);
            if (!hi || Q_cmp(pr->linf, hi) < 0) Q_set(&hi, Q_copy(c, pr->linf));
        }
        P->lo[i] = lo; P->hi[i] = hi;
    }
    P->l1 = pr->l1 ? Q_copy(c, pr->l1) : NULL;
    if (pr->l1 && Q_sgn(pr->l1) < 0) ifail(c, TR_MATH_INVALID);
    if (pr->linf && Q_sgn(pr->linf) < 0) ifail(c, TR_MATH_INVALID);
}
static int linear_checks(const tr_interv_linear_problem *pr) {
    if (!pr || pr->n < 1 || !cost_valid(pr->cost) || !op_valid(pr->op) || (pr->flip != 0 && pr->flip != 1) ||
        pr->m < 0 || pr->p < 0 || (pr->m && (!pr->A || !pr->b)) || (pr->p && (!pr->E || !pr->e)) || !pr->x0)
        return TR_MATH_INVALID;
    if (pr->semantics == TR_INTERV_CAUSAL) return TR_MATH_UNSUPPORTED;
    if (pr->semantics != TR_INTERV_ALGEBRAIC) return TR_MATH_INVALID;
    if (pr->n > TR_INTERV_MAX_DIMS) return TR_MATH_UNSUPPORTED;
    for (int32_t i = 0; i < pr->n; i++) if (!pr->x0[i]) return TR_MATH_INVALID;
    for (int32_t i = 0; i < pr->m; i++) { if (!pr->b[i]) return TR_MATH_INVALID; for (int32_t j = 0; j < pr->n; j++) if (!pr->A[(uint64_t)i * pr->n + j]) return TR_MATH_INVALID; }
    for (int32_t i = 0; i < pr->p; i++) { if (!pr->e[i]) return TR_MATH_INVALID; for (int32_t j = 0; j < pr->n; j++) if (!pr->E[(uint64_t)i * pr->n + j]) return TR_MATH_INVALID; }
    return TR_MATH_OK;
}
/* the L0 cost: supports in increasing weight */
typedef struct { tr_bigq *w; uint32_t mask; } mask_t;
static int mask_cmp(const void *x, const void *y) {
    const mask_t *a = (const mask_t *)x, *b = (const mask_t *)y;
    int c = tr_bigq_cmp(a->w, b->w);
    return c ? c : (a->mask < b->mask ? -1 : a->mask > b->mask);
}
/* the problem restricted to the coordinates of mask (the others 0): 0 when the box excludes 0 elsewhere */
static int restrict_dprob(ictx *c, const dprob *P, uint32_t mask, dprob *Q, int32_t *map) {
    const int32_t n = P->n;
    memset(Q, 0, sizeof *Q);
    int32_t k = 0;
    for (int32_t i = 0; i < n; i++) {
        if ((mask >> i) & 1) { map[k++] = i; continue; }
        if ((P->lo[i] && Q_sgn(P->lo[i]) > 0) || (P->hi[i] && Q_sgn(P->hi[i]) < 0)) return 0;
    }
    if (!k) return -1;
    Q->n = k; Q->cost = P->cost; Q->m = P->m; Q->p = P->p;
    Q->w = Q_arr(c, (uint64_t)k); Q->a = Q_arr(c, (uint64_t)k); Q->lo = Q_arr(c, (uint64_t)k); Q->hi = Q_arr(c, (uint64_t)k);
    Q->x0 = Q_arr(c, (uint64_t)k);
    for (int32_t j = 0; Q->w && Q->a && Q->lo && Q->hi && Q->x0 && j < k && LIVE(c); j++) {
        Q->w[j] = Q_copy(c, P->w[map[j]]); Q->a[j] = Q_copy(c, P->a[map[j]]); Q->x0[j] = Q_copy(c, P->x0[map[j]]);
        if (P->lo[map[j]]) Q->lo[j] = Q_copy(c, P->lo[map[j]]);
        if (P->hi[map[j]]) Q->hi[j] = Q_copy(c, P->hi[map[j]]);
    }
    Q->A = Q_arr(c, (uint64_t)(P->m ? P->m : 1) * k); Q->b = Q_full(c, (const tr_bigq *const *)P->b, (uint64_t)P->m);
    for (int32_t i = 0; Q->A && i < P->m && LIVE(c); i++)
        for (int32_t j = 0; j < k; j++) Q->A[(uint64_t)i * k + j] = Q_copy(c, P->A[(uint64_t)i * n + map[j]]);
    Q->E = Q_arr(c, (uint64_t)(P->p ? P->p : 1) * k); Q->e = Q_full(c, (const tr_bigq *const *)P->e, (uint64_t)P->p);
    for (int32_t i = 0; Q->E && i < P->p && LIVE(c); i++)
        for (int32_t j = 0; j < k; j++) Q->E[(uint64_t)i * k + j] = Q_copy(c, P->E[(uint64_t)i * n + map[j]]);
    Q->l1 = P->l1 ? Q_copy(c, P->l1) : NULL;
    Q->r = Q_copy(c, P->r);
    return 1;
}
/* is the zero change feasible (constraints and the target op) */
static int zero_feasible(ictx *c, const dprob *P, int32_t op) {
    for (int32_t i = 0; i < P->n; i++)
        if ((P->lo[i] && Q_sgn(P->lo[i]) > 0) || (P->hi[i] && Q_sgn(P->hi[i]) < 0)) return 0;
    for (int32_t i = 0; i < P->m; i++) if (Q_sgn(P->b[i]) < 0) return 0;
    for (int32_t i = 0; i < P->p; i++) if (Q_sgn(P->e[i]) != 0) return 0;
    if (P->l1 && Q_sgn(P->l1) < 0) return 0;
    (void)c;
    return cmp_holds(op, -Q_sgn(P->r));                          /* a^T 0 - r = -r */
}
/* a witness of the restricted problem for op (closed or strict; NE: either side): owned in *d, or NULL */
static tr_bigq **support_witness(ictx *c, const dprob *Q, int32_t op) {
    tr_bigq *vmax = NULL, **at = NULL;
    if (op == TR_INTERV_EQ || op == TR_INTERV_LE || op == TR_INTERV_GE) {
        layout_t Y = layout_of(Q, TR_INTERV_L2);
        lpb L;
        lpb_init(c, &L, Y.nv);
        build_rows(c, Q, TR_INTERV_L2, op, &Y, &L);
        tr_polyopt_result *R = lpb_solve(c, &L, TR_POLYOPT_NO_UNIQUENESS);
        lpb_free(&L);
        tr_bigq **d = R && r_status(R) == TR_POLYOPT_OPTIMAL ? res_d(c, R, Q->n) : NULL;
        tr_polyopt_result_free(R);
        return d;
    }
    for (int s = 1; s >= -1; s -= 2) {
        if ((op == TR_INTERV_GT && s < 0) || (op == TR_INTERV_LT && s > 0)) continue;
        int32_t st = max_target(c, Q, TR_INTERV_L2, s, T_NONE, NULL, NULL, &vmax, &at);
        tr_bigq *sr = s > 0 ? Q_copy(c, Q->r) : Q_neg(c, Q->r);
        int ok = st == TR_POLYOPT_UNBOUNDED || (st == TR_POLYOPT_OPTIMAL && Q_cmp(vmax, sr) > 0);
        tr_bigq_free(sr); tr_bigq_free(vmax); vmax = NULL;
        if (ok && at) return at;
        Q_free(at, at ? (uint64_t)Q->n : 0); at = NULL;
    }
    return NULL;
}
static void linear_l0(ictx *c, tr_interv_result *r, const dprob *P, int32_t op, const tr_bigq *a0t_base) {
    const int32_t n = P->n;
    (void)a0t_base;
    uint32_t nm = (uint32_t)1 << n;
    mask_t *ms = (mask_t *)zalloc((uint64_t)nm * sizeof *ms);
    if (!ms) { ifail(c, TR_MATH_MEMORY); return; }
    for (uint32_t m = 0; m < nm && LIVE(c); m++) {
        ms[m].mask = m;
        if (!m) { ms[m].w = Q_int(c, 0); continue; }
        int32_t low = __builtin_ctz(m);
        ms[m].w = Q_add(c, ms[m & (m - 1)].w, P->w[low]);
    }
    if (LIVE(c)) qsort(ms, nm, sizeof *ms, mask_cmp);
    tr_bigq *best = NULL;
    uint64_t examined = 0;
    uint32_t next = nm;
    int stopped = 0;
    int32_t *map = (int32_t *)zalloc((uint64_t)n * 4);
    if (!map) ifail(c, TR_MATH_MEMORY);
    for (uint32_t t = 0; t < nm && LIVE(c); t++) {
        if (best && tr_bigq_cmp(ms[t].w, best) > 0) { next = t; break; }   /* every support of the minimal weight */
        if (c->iters >= c->limit) { next = t; stopped = 1; break; }    /* the iteration budget: incomplete */
        c->iters++;
        examined++;
        uint32_t m = ms[t].mask;
        tr_bigq **d = NULL;
        if (!m) { if (zero_feasible(c, P, op)) d = Q_full(c, NULL, (uint64_t)n); }
        else {
            dprob Q;
            int ok = restrict_dprob(c, P, m, &Q, map);
            if (ok > 0) {
                tr_bigq **dq = support_witness(c, &Q, op);
                if (dq) {
                    d = Q_full(c, NULL, (uint64_t)n);
                    for (int32_t j = 0; d && j < Q.n && LIVE(c); j++) Q_set(&d[map[j]], Q_copy(c, dq[j]));
                    Q_free(dq, (uint64_t)Q.n);
                }
            }
            dprob_free(&Q);
        }
        if (!d || !LIVE(c)) { Q_free(d, d ? (uint64_t)n : 0); continue; }
        if (!best) best = Q_copy(c, ms[t].w);
        sol_t *s = sol_exact(c, r, n, d, P->x0, Q_copy(c, ms[t].w), NULL);
        if (s) s->support = m;
    }
    if (LIVE(c) && r) {
        r->info.examined = examined;
        r->info.total = nm;
        r->info.next = next;
        r->info.complete = !stopped;
        if (best) {
            r->info.status = stopped ? TR_INTERV_INCOMPLETE : TR_INTERV_OPTIMAL;
            if (stopped) r->info.reason = TR_INTERV_REASON_ITERATIONS;   /* the minimum is proven, ties may be missing */
            res_one(c, r, TR_INTERV_PART_LOWER, best);
            res_one(c, r, TR_INTERV_PART_UPPER, best);
            r->info.unique = r->nsol == 1 && Q_zero(best) ? 1 : (r->nsol > 1 ? 0 : -1);
        } else if (!stopped) r->info.status = TR_INTERV_INFEASIBLE;
        else {
            r->info.status = TR_INTERV_INCOMPLETE;
            r->info.reason = TR_INTERV_REASON_ITERATIONS;
            res_one(c, r, TR_INTERV_PART_LOWER, ms[next].w);     /* no support of smaller weight is feasible */
        }
    }
    for (uint32_t m = 0; m < nm; m++) tr_bigq_free(ms[m].w);
    free(ms); free(map);
    tr_bigq_free(best);
}
static tr_interv_result *linear_core(ictx *c, const tr_interv_linear_problem *pr, tr_bigq *const *a, const tr_bigq *a0) {
    const int32_t n = pr->n;
    dprob P;
    make_dprob(c, pr, a, &P);
    tr_interv_result *r = res_new(c, pr->cost == TR_INTERV_L0 ? TR_INTERV_M_SUPPORTS :
                                     pr->cost == TR_INTERV_L2 ? TR_INTERV_M_QP : TR_INTERV_M_LP, n, pr->cost);
    tr_bigq *f0 = Q_dot(c, (const tr_bigq *const *)P.a, (const tr_bigq *const *)P.x0, n);
    if (a0) Q_set(&f0, Q_add(c, f0, a0));
    tr_bigq *t = Q_copy(c, pr->t);
    int32_t op = pr->op;
    tr_bigq *diff = Q_sub(c, f0, t);
    int member = LIVE(c) ? cmp_holds(pr->op, Q_sgn(diff)) : 0;
    tr_bigq_free(diff);
    if (pr->flip) op = member ? cmp_complement(pr->op) : pr->op;
    if (r) { r->info.op = op; r->info.base_member = member; }
    P.r = Q_sub(c, t, f0);                                       /* a^T d op r */
    res_one(c, r, TR_INTERV_PART_BASE, f0);
    if (LIVE(c) && pr->cost == TR_INTERV_L0) { linear_l0(c, r, &P, op, NULL); }
    else if (LIVE(c)) {
        cand_t cand[2];
        memset(cand, 0, sizeof cand);
        int ncand = 0;
        int32_t closed = T_NONE;
        if (op == TR_INTERV_EQ || op == TR_INTERV_LE || op == TR_INTERV_GE) {
            closed = op;
            int32_t st = solve_closed(c, &P, pr->cost, op, &cand[0].cost, &cand[0].d);
            cand[0].status = st == TR_POLYOPT_OPTIMAL ? TR_INTERV_OPTIMAL : TR_INTERV_INFEASIBLE;
            ncand = 1;
        } else if (op == TR_INTERV_LT || op == TR_INTERV_GT) {
            solve_strict(c, &P, pr->cost, op == TR_INTERV_GT ? 1 : -1, &cand[0]);
            closed = op == TR_INTERV_GT ? TR_INTERV_GE : TR_INTERV_LE;
            ncand = 1;
        } else {                                                 /* NE: both strict sides */
            solve_strict(c, &P, pr->cost, -1, &cand[0]);
            solve_strict(c, &P, pr->cost, 1, &cand[1]);
            ncand = 2;
        }
        tr_bigq *best = NULL;
        int any_attained = 0;
        for (int k = 0; k < ncand && LIVE(c); k++) {
            if (cand[k].status == TR_INTERV_INFEASIBLE) continue;
            if (!best || Q_cmp(cand[k].cost, best) < 0) { tr_bigq_free(best); best = Q_copy(c, cand[k].cost); }
        }
        for (int k = 0; k < ncand && best && LIVE(c); k++)
            if (cand[k].status == TR_INTERV_OPTIMAL && !Q_cmp(cand[k].cost, best)) any_attained = 1;
        if (!best && r) r->info.status = TR_INTERV_INFEASIBLE;
        else if (r) {
            r->info.status = any_attained ? TR_INTERV_OPTIMAL : TR_INTERV_UNATTAINED;
            r->info.complete = 1;
            int listed = 0;
            for (int k = 0; k < ncand && LIVE(c); k++) {
                if (cand[k].status == TR_INTERV_INFEASIBLE || Q_cmp(cand[k].cost, best)) continue;
                if (any_attained && cand[k].status != TR_INTERV_OPTIMAL) continue;
                int dup = 0;                                     /* the same change from both sides of a NE */
                for (int32_t q = 0; q < r->nsol && !dup; q++) {
                    int same = 1;
                    for (int32_t i = 0; i < n && same; i++) if (Q_cmp(r->sol[q].dlo[i], cand[k].d[i])) same = 0;
                    dup = same;
                }
                if (dup) continue;
                tr_bigq *fx = NULL;
                tr_bigq **xs = Q_arr(c, (uint64_t)n);
                for (int32_t i = 0; xs && i < n && LIVE(c); i++) xs[i] = Q_add(c, P.x0[i], cand[k].d[i]);
                fx = xs ? Q_dot(c, (const tr_bigq *const *)P.a, (const tr_bigq *const *)xs, n) : NULL;
                if (a0) Q_set(&fx, Q_add(c, fx, a0));
                Q_free(xs, (uint64_t)n);
                sol_t *s = sol_exact(c, r, n, cand[k].d, P.x0, Q_copy(c, cand[k].cost), fx);
                cand[k].d = NULL;
                if (s) s->s.attained = cand[k].status == TR_INTERV_OPTIMAL;
                listed++;
            }
            res_one(c, r, TR_INTERV_PART_LOWER, best);
            res_one(c, r, TR_INTERV_PART_UPPER, best);
            if (listed > 1) r->info.unique = 0;
            else if (r->info.status == TR_INTERV_OPTIMAL && LIVE(c)) {      /* the optimal set: a point or a face */
                int32_t cl = closed;
                if (op == TR_INTERV_NE) {
                    tr_bigq *ad = Q_dot(c, (const tr_bigq *const *)P.a, (const tr_bigq *const *)r->sol[0].dlo, n);
                    cl = Q_cmp(ad, P.r) > 0 ? TR_INTERV_GE : TR_INTERV_LE;
                    tr_bigq_free(ad);
                }
                tr_bigq **face = NULL;
                int u = unique_change(c, &P, pr->cost, cl, best, r->sol[0].dlo, &face);
                r->info.unique = u;
                if (face) {
                    tr_bigq *fc = cost_vector(c, pr->cost, n, P.w, face);
                    if (LIVE(c) && Q_cmp(fc, best)) ifail(c, TR_MATH_INVALID);      /* checked: equally cheap */
                    res_part(c, r, TR_INTERV_PART_FACE, face, (uint64_t)n);
                    res_one(c, r, TR_INTERV_PART_FACE_COST, fc);
                    tr_bigq_free(fc);
                }
            }
        }
        tr_bigq_free(best);
        for (int k = 0; k < 2; k++) cand_free(&cand[k], n);
    }
    tr_bigq_free(f0); tr_bigq_free(t);
    if (r) r->info.iterations = c->iters;
    dprob_free(&P);
    return r;
}
tr_interv_result *tr_interv_linear(const tr_interv_linear_problem *pr, const tr_math_budget *budget,
                                   tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    int32_t rc = linear_checks(pr);
    if (rc) { set_error(error, rc); return NULL; }
    tr_math_need need;
    memset(&need, 0, sizeof need);
    need.need[TR_BUDGET_VARIABLES] = (uint64_t)pr->n;
    need.need[TR_BUDGET_DEGREE] = 1;
    need.need[TR_BUDGET_TENSOR] = (uint64_t)(pr->m + pr->p + 1) * (uint64_t)(pr->n + 1);
    rc = budget_need(budget, refusal, &need);
    if (rc) { set_error(error, rc); return NULL; }
    ictx c;
    ictx_init(&c, budget, refusal);
    tr_bigq **a = Q_full(&c, pr->a, (uint64_t)pr->n);
    tr_interv_result *r = LIVE(&c) ? linear_core(&c, pr, a, pr->a0) : NULL;
    Q_free(a, (uint64_t)pr->n);
    return res_done(&c, r, error);
}
tr_interv_result *tr_interv_linear_poly(const tr_interv_linear_problem *pr, const tr_math_poly *f, const int32_t *vars,
                                        const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    int32_t rc = linear_checks(pr);
    if (!rc && (!f || !vars)) rc = TR_MATH_INVALID;
    int32_t nt = rc ? 0 : tr_math_poly_term_count(f);
    if (nt < 0) rc = TR_MATH_INVALID;
    if (rc) { set_error(error, rc); return NULL; }
    ictx c;
    ictx_init(&c, budget, refusal);
    tr_bigq **a = Q_full(&c, NULL, (uint64_t)pr->n), *a0 = Q_int(&c, 0);
    for (int32_t k = 0; k < nt && LIVE(&c); k++) {
        tr_math_term term;
        if (tr_math_poly_get_term(f, k, &term)) { ifail(&c, TR_MATH_INVALID); break; }
        int32_t deg = 0, at = -1;
        for (int32_t s = 0; s < TR_MATH_VARIABLES; s++) {
            if (!term.exponents[s]) continue;
            int32_t v = -1;
            for (int32_t i = 0; i < pr->n; i++) if (vars[i] == s) v = i;
            if (v < 0) { ifail(&c, TR_MATH_DOMAIN_ERROR); break; }
            deg += term.exponents[s]; at = v;
        }
        if (!LIVE(&c)) break;
        if (deg > 1) { ifail(&c, TR_MATH_UNSUPPORTED); break; }
        tr_bigq *cf = Q_q64(&c, term.coefficient.numerator, term.coefficient.denominator);
        if (deg == 0) Q_set(&a0, Q_add(&c, a0, cf)); else Q_set(&a[at], Q_add(&c, a[at], cf));
        tr_bigq_free(cf);
    }
    tr_interv_result *r = NULL;
    if (LIVE(&c)) {
        tr_math_need need;
        memset(&need, 0, sizeof need);
        need.need[TR_BUDGET_VARIABLES] = (uint64_t)pr->n;
        need.need[TR_BUDGET_DEGREE] = 1;
        need.need[TR_BUDGET_TENSOR] = (uint64_t)(pr->m + pr->p + 1) * (uint64_t)(pr->n + 1);
        rc = budget_need(budget, refusal, &need);
        if (rc) ifail(&c, rc);
    }
    if (LIVE(&c)) r = linear_core(&c, pr, a, a0);
    Q_free(a, (uint64_t)pr->n); tr_bigq_free(a0);
    return res_done(&c, r, error);
}

/* ================================================================ 2. univariate polynomial targets */
typedef struct { int32_t deg; tr_bigq **c; } upoly;              /* low degree first; deg -1: the zero polynomial */
static void up_free(upoly *p) { Q_free(p->c, p->c ? (uint64_t)(p->deg + 1 > 0 ? p->deg + 1 : 1) : 0); p->c = NULL; p->deg = -1; }
static void up_trim(upoly *p) { while (p->deg >= 0 && Q_zero(p->c[p->deg])) { tr_bigq_free(p->c[p->deg]); p->c[p->deg] = NULL; p->deg--; } }
static tr_bigq *up_eval(ictx *c, const upoly *p, const tr_bigq *x) {
    tr_bigq *acc = Q_int(c, 0);
    for (int32_t i = p->deg; i >= 0 && LIVE(c); i--) {
        tr_bigq *t = Q_mul(c, acc, x);
        Q_set(&acc, Q_add(c, t, p->c[i]));
        tr_bigq_free(t);
    }
    return acc;
}
static int up_sign(ictx *c, const upoly *p, const tr_bigq *x) {
    tr_bigq *v = up_eval(c, p, x);
    int s = Q_sgn(v);
    tr_bigq_free(v);
    return s;
}
/* q(y) = p(s + y) (reflect: p(s - y)) by repeated synthetic division */
static upoly up_shift(ictx *c, const upoly *p, const tr_bigq *s, int reflect) {
    upoly q;
    q.deg = p->deg;
    q.c = Q_full(c, (const tr_bigq *const *)p->c, (uint64_t)(p->deg + 1 > 0 ? p->deg + 1 : 1));
    for (int32_t i = 0; q.c && i < q.deg && LIVE(c); i++)
        for (int32_t k = q.deg - 1; k >= i && LIVE(c); k--) {
            tr_bigq *t = Q_mul(c, s, q.c[k + 1]);
            Q_set(&q.c[k], Q_add(c, q.c[k], t));
            tr_bigq_free(t);
        }
    if (reflect) for (int32_t k = 1; q.c && k <= q.deg && LIVE(c); k += 2) Q_set(&q.c[k], Q_neg(c, q.c[k]));
    return q;
}
/* the monic gcd of a and b over Q (Euclid) */
static upoly up_gcd(ictx *c, const upoly *a0, const upoly *b0) {
    upoly a, b;
    a.deg = a0->deg; a.c = Q_full(c, (const tr_bigq *const *)a0->c, (uint64_t)(a0->deg + 1 > 0 ? a0->deg + 1 : 1));
    b.deg = b0->deg; b.c = Q_full(c, (const tr_bigq *const *)b0->c, (uint64_t)(b0->deg + 1 > 0 ? b0->deg + 1 : 1));
    up_trim(&a); up_trim(&b);
    while (b.deg >= 0 && LIVE(c)) {                               /* a <- a mod b, then swap */
        while (a.deg >= b.deg && a.deg >= 0 && LIVE(c)) {
            tr_bigq *f = Q_div(c, a.c[a.deg], b.c[b.deg]);
            int32_t sh = a.deg - b.deg;
            for (int32_t k = 0; k <= b.deg && LIVE(c); k++) {
                tr_bigq *t = Q_mul(c, f, b.c[k]);
                Q_set(&a.c[k + sh], Q_sub(c, a.c[k + sh], t));
                tr_bigq_free(t);
            }
            tr_bigq_free(f);
            up_trim(&a);
        }
        upoly t = a; a = b; b = t;
    }
    up_free(&b);
    if (a.deg >= 0 && LIVE(c)) {
        tr_bigq *lc = Q_copy(c, a.c[a.deg]);
        for (int32_t k = 0; k <= a.deg && LIVE(c); k++) Q_set(&a.c[k], Q_div(c, a.c[k], lc));
        tr_bigq_free(lc);
    }
    return a;
}
/* the elements of the domain partition: roots of F (by index) and the finite domain ends that are not roots */
typedef struct {
    int32_t kind;                            /* 0: root, 1: the domain's lower end, 2: its upper end */
    int32_t root;
    tr_bigq *at;                             /* an exact position (an end, or a root proved to be x0); NULL: read root */
    int32_t feasible;
} elem_t;
typedef struct {
    ictx *c;
    upoly F;
    tr_inv_roots *roots;
    elem_t *e;
    int32_t ne;
    int32_t op;
} upart;
/* the position of element i: lo == hi when exact (borrowed) */
static int elem_pos(const upart *U, int32_t i, const tr_bigq **lo, const tr_bigq **hi, tr_inv_root_info *ri) {
    const elem_t *e = &U->e[i];
    tr_inv_root_info dummy;
    if (!ri) ri = &dummy;
    memset(ri, 0, sizeof *ri);
    if (e->at) { *lo = *hi = e->at; ri->exact = 1; if (e->kind == 0) { tr_inv_root_info t; const tr_bigq *a, *b; tr_inv_roots_root(U->roots, e->root, &t, &a, &b); *ri = t; ri->exact = 1; } return 1; }
    tr_inv_roots_root(U->roots, e->root, ri, lo, hi);
    return ri->exact;
}
static int upart_refine(upart *U) {
    ictx *c = U->c;
    tr_bigq *w = NULL;
    tr_inv_roots_info inf;
    tr_inv_roots_get_info(U->roots, &inf);
    for (int32_t i = 0; i < inf.nroots && LIVE(c); i++) {         /* half the widest isolating interval */
        tr_inv_root_info ri;
        const tr_bigq *lo, *hi;
        tr_inv_roots_root(U->roots, i, &ri, &lo, &hi);
        if (ri.exact) continue;
        tr_bigq *d = Q_sub(c, hi, lo);
        if (!w || Q_cmp(d, w) > 0) { tr_bigq_free(w); w = d; } else tr_bigq_free(d);
    }
    if (!w) return 0;
    tr_bigq *h = Q_q64(c, 1, 2);
    Q_set(&w, Q_mul(c, w, h));
    tr_bigq_free(h);
    int32_t e = 0;
    if (c->iters >= c->limit) { tr_bigq_free(w); return 0; }
    c->iters++;
    tr_inv_roots *r = LIVE(c) ? tr_inv_roots_refine(U->roots, w, NULL, c->b, c->r, &e) : NULL;
    tr_bigq_free(w);
    if (!r) { if (LIVE(c)) ifail(c, e); return 0; }
    tr_inv_roots_free(U->roots);
    U->roots = r;
    return 1;
}
/* the gcd tie test: distance intervals R (right) and L (left) of two roots of F; 1 when the distances are equal */
static int roots_tie(ictx *c, const upoly *F, const tr_bigq *x0, const tr_bigq *jlo, const tr_bigq *jhi) {
    upoly PR = up_shift(c, F, x0, 0), PL = up_shift(c, F, x0, 1);
    upoly G = up_gcd(c, &PR, &PL);
    int tie = 0;
    if (G.deg >= 1 && LIVE(c) && Q_cmp(jlo, jhi) < 0) {
        int32_t e = 0;
        tr_inv_roots *gr = tr_inv_isolate(G.deg + 1, (const tr_bigq *const *)G.c, NULL, jlo, jhi, c->b, c->r, &e);
        if (!gr) ifail(c, e);
        else {
            tr_inv_roots_info gi;
            tr_inv_roots_get_info(gr, &gi);
            for (int32_t i = 0; i < gi.nroots; i++) {
                tr_inv_root_info ri;
                const tr_bigq *lo, *hi;
                tr_inv_roots_root(gr, i, &ri, &lo, &hi);
                if (!ri.boundary) tie = 1;                        /* a common root strictly inside both intervals */
            }
            tr_inv_roots_free(gr);
        }
    }
    up_free(&PR); up_free(&PL); up_free(&G);
    return tie;
}
/* a candidate: the nearest feasible point on one side */
typedef struct {
    int32_t found, side, attained, elem;     /* elem: the element at the candidate's position */
} ucand;
static int seg_feasible(upart *U, int32_t i, int32_t j, const tr_bigq *lower_end, const tr_bigq *upper_end) {
    /* the open segment between element i and element j (i = -1: (-inf, e_j); j = ne: (e_i, +inf)) */
    ictx *c = U->c;
    tr_bigq *q = NULL;
    if (i < 0) { tr_bigq *one = Q_int(c, 1); q = Q_sub(c, upper_end, one); tr_bigq_free(one); }
    else if (j >= U->ne) { tr_bigq *one = Q_int(c, 1); q = Q_add(c, lower_end, one); tr_bigq_free(one); }
    else q = Q_mid(c, lower_end, upper_end);
    int s = up_sign(c, &U->F, q);
    tr_bigq_free(q);
    return cmp_holds(U->op, s);
}
tr_interv_result *tr_interv_univariate(const tr_interv_univariate_problem *pr, const tr_math_budget *budget,
                                       tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    if (!pr || !cost_valid(pr->cost) || !op_valid(pr->op) || (pr->flip != 0 && pr->flip != 1) || !pr->x0 ||
        (!pr->slice && (pr->ncoef < 1 || !pr->coef)) || (pr->weight && Q_sgn(pr->weight) <= 0) ||
        (pr->width && Q_sgn(pr->width) <= 0)) { set_error(error, pr && pr->semantics == TR_INTERV_CAUSAL ? TR_MATH_UNSUPPORTED : TR_MATH_INVALID); return NULL; }
    if (pr->semantics == TR_INTERV_CAUSAL) { set_error(error, TR_MATH_UNSUPPORTED); return NULL; }
    if (pr->semantics != TR_INTERV_ALGEBRAIC) { set_error(error, TR_MATH_INVALID); return NULL; }
    ictx c;
    ictx_init(&c, budget, refusal);
    upart U;
    memset(&U, 0, sizeof U);
    U.c = &c; U.F.deg = -1;
    /* F = f - t */
    if (pr->slice) {
        int32_t nt = tr_math_poly_term_count(pr->slice);
        if (nt < 0 || pr->variable < 0 || pr->variable >= TR_MATH_VARIABLES) ifail(&c, TR_MATH_INVALID);
        U.F.deg = TR_MATH_DEGREE;
        U.F.c = Q_arr(&c, TR_MATH_DEGREE + 1);
        for (int32_t k = 0; k < nt && LIVE(&c); k++) {
            tr_math_term term;
            if (tr_math_poly_get_term(pr->slice, k, &term)) { ifail(&c, TR_MATH_INVALID); break; }
            for (int32_t s = 0; s < TR_MATH_VARIABLES; s++) if (s != pr->variable && term.exponents[s]) ifail(&c, TR_MATH_DOMAIN_ERROR);
            int32_t e = term.exponents[pr->variable];
            tr_bigq *cf = Q_q64(&c, term.coefficient.numerator, term.coefficient.denominator);
            if (U.F.c) Q_set(&U.F.c[e], U.F.c[e] ? Q_add(&c, U.F.c[e], cf) : Q_copy(&c, cf));
            tr_bigq_free(cf);
        }
    } else {
        U.F.deg = pr->ncoef - 1;
        U.F.c = Q_full(&c, pr->coef, (uint64_t)pr->ncoef);
    }
    if (U.F.c && LIVE(&c)) {
        for (int32_t k = 0; k <= U.F.deg; k++) if (!U.F.c[k]) U.F.c[k] = Q_int(&c, 0);
        if (pr->t) Q_set(&U.F.c[0], Q_sub(&c, U.F.c[0], pr->t));
    }
    if (LIVE(&c)) {
        tr_math_need need;
        memset(&need, 0, sizeof need);
        up_trim(&U.F);
        need.need[TR_BUDGET_VARIABLES] = 1;
        need.need[TR_BUDGET_DEGREE] = (uint64_t)(U.F.deg > 0 ? U.F.deg : 0);
        int32_t rc = budget_need(budget, refusal, &need);
        if (rc) ifail(&c, rc);
    }
    tr_bigq *w = pr->weight ? Q_copy(&c, pr->weight) : Q_int(&c, 1);
    tr_interv_result *r = res_new(&c, TR_INTERV_M_ROOTS, 1, pr->cost);
    int s0 = LIVE(&c) ? up_sign(&c, &U.F, pr->x0) : 0;
    int member = cmp_holds(pr->op, s0);
    U.op = pr->flip ? (member ? cmp_complement(pr->op) : pr->op) : pr->op;
    if (r) { r->info.op = U.op; r->info.base_member = member; }
    tr_bigq *fx0 = Q_copy(&c, pr->x0);
    if (LIVE(&c)) {                                              /* f(x0) = F(x0) + t */
        tr_bigq *v = up_eval(&c, &U.F, pr->x0);
        if (pr->t) Q_set(&v, Q_add(&c, v, pr->t));
        res_one(&c, r, TR_INTERV_PART_BASE, v);
        tr_bigq_free(v);
    }
    tr_bigq_free(fx0);
    /* the domain of x: [max(x0 + dlo, xlo), min(x0 + dhi, xhi)] */
    tr_bigq *Xlo = NULL, *Xhi = NULL;
    if (pr->dlo) Xlo = Q_add(&c, pr->x0, pr->dlo);
    if (pr->xlo && (!Xlo || Q_cmp(pr->xlo, Xlo) > 0)) Q_set(&Xlo, Q_copy(&c, pr->xlo));
    if (pr->dhi) Xhi = Q_add(&c, pr->x0, pr->dhi);
    if (pr->xhi && (!Xhi || Q_cmp(pr->xhi, Xhi) < 0)) Q_set(&Xhi, Q_copy(&c, pr->xhi));
    int inside = (!Xlo || Q_cmp(pr->x0, Xlo) >= 0) && (!Xhi || Q_cmp(pr->x0, Xhi) <= 0);
    int done = 0;
    if (LIVE(&c) && Xlo && Xhi && Q_cmp(Xlo, Xhi) > 0) { r->info.status = TR_INTERV_INFEASIBLE; r->info.complete = 1; done = 1; }
    if (LIVE(&c) && !done && inside && cmp_holds(U.op, s0)) {    /* no change is needed */
        tr_bigq **d = Q_arr(&c, 1);
        if (d) d[0] = Q_int(&c, 0);
        tr_bigq *x0a[1] = {(tr_bigq *)pr->x0};
        tr_bigq *v = up_eval(&c, &U.F, pr->x0);
        if (pr->t) Q_set(&v, Q_add(&c, v, pr->t));
        sol_exact(&c, r, 1, d, x0a, Q_int(&c, 0), v);
        r->info.status = TR_INTERV_OPTIMAL; r->info.unique = 1; r->info.complete = 1;
        done = 1;
    }
    if (LIVE(&c) && !done) {
        int32_t e = 0;
        U.roots = tr_inv_isolate(U.F.deg >= 0 ? U.F.deg + 1 : 1, (const tr_bigq *const *)U.F.c, NULL, Xlo, Xhi, c.b, c.r, &e);
        if (U.F.deg < 0 && !U.roots) {                           /* the zero polynomial */
            tr_bigq *z = Q_int(&c, 0);
            U.roots = tr_inv_isolate(1, (const tr_bigq *const *)&z, NULL, Xlo, Xhi, c.b, c.r, &e);
            tr_bigq_free(z);
        }
        if (!U.roots) ifail(&c, e);
    }
    tr_inv_roots_info ri0;
    memset(&ri0, 0, sizeof ri0);
    if (U.roots) tr_inv_roots_get_info(U.roots, &ri0);
    if (LIVE(&c) && !done && ri0.kind != TR_INV_FINITE) {
        /* constant sign (or F = 0) over the domain: feasible everywhere or nowhere; the nearest domain point */
        int s = ri0.kind == TR_INV_WHOLE ? 0 : up_sign(&c, &U.F, Xlo ? Xlo : (Xhi ? Xhi : pr->x0));
        if (ri0.kind == TR_INV_EMPTY && !Xlo && !Xhi) s = up_sign(&c, &U.F, pr->x0);
        if (!cmp_holds(U.op, s)) { r->info.status = TR_INTERV_INFEASIBLE; r->info.complete = 1; }
        else {
            const tr_bigq *pt = Xlo && Q_cmp(pr->x0, Xlo) < 0 ? Xlo : Xhi;
            tr_bigq **d = Q_arr(&c, 1);
            if (d) d[0] = Q_sub(&c, pt, pr->x0);
            tr_bigq *x0a[1] = {(tr_bigq *)pr->x0};
            tr_bigq *v = up_eval(&c, &U.F, pt);
            if (pr->t) Q_set(&v, Q_add(&c, v, pr->t));
            tr_bigq *cost = d ? cost_scalar(&c, pr->cost, w, d[0]) : NULL;
            sol_t *sl = sol_exact(&c, r, 1, d, x0a, cost, v);
            if (sl) sl->s.side = Q_cmp(pt, pr->x0) > 0 ? 1 : -1;
            r->info.status = TR_INTERV_OPTIMAL; r->info.complete = 1;
            r->info.unique = pr->cost == TR_INTERV_L0 ? (Q_cmp(Xlo, Xhi) == 0 ? 1 : 0) : 1;
            res_one(&c, r, TR_INTERV_PART_LOWER, cost);
            res_one(&c, r, TR_INTERV_PART_UPPER, cost);
        }
        done = 1;
    }
    if (LIVE(&c) && !done) {
        /* the partition: the lower end (unless a root), the roots, the upper end (unless a root) */
        int32_t nr = ri0.nroots;
        U.e = (elem_t *)zalloc((uint64_t)(nr + 2) * sizeof *U.e);
        if (!U.e) ifail(&c, TR_MATH_MEMORY);
        int lo_root = 0, hi_root = 0;
        for (int32_t i = 0; i < nr; i++) {
            tr_inv_root_info ri;
            const tr_bigq *a, *b;
            tr_inv_roots_root(U.roots, i, &ri, &a, &b);
            if (ri.boundary & TR_INV_AT_LO) lo_root = 1;
            if (ri.boundary & TR_INV_AT_HI) hi_root = 1;
        }
        if (U.e && Xlo && !lo_root) { U.e[U.ne].kind = 1; U.e[U.ne].at = Q_copy(&c, Xlo); U.ne++; }
        for (int32_t i = 0; U.e && i < nr; i++) { U.e[U.ne].kind = 0; U.e[U.ne].root = i; U.ne++; }
        if (U.e && Xhi && !hi_root) { U.e[U.ne].kind = 2; U.e[U.ne].at = Q_copy(&c, Xhi); U.ne++; }
        /* x0 inside an isolating interval: either it is that root (F(x0) = 0) or refine until it is outside; then
         * refine until neighbouring elements are separated */
        for (int round = 0; LIVE(&c) && round < 4096; round++) {
            int again = 0;
            for (int32_t i = 0; i < U.ne && LIVE(&c); i++) {
                const tr_bigq *lo, *hi;
                if (elem_pos(&U, i, &lo, &hi, NULL)) continue;
                if (Q_cmp(lo, pr->x0) < 0 && Q_cmp(pr->x0, hi) < 0) {
                    if (s0 == 0) { U.e[i].at = Q_copy(&c, pr->x0); continue; }
                    again = 1;
                }
            }
            for (int32_t i = 0; i + 1 < U.ne && LIVE(&c) && !again; i++) {
                const tr_bigq *a1, *b1, *a2, *b2;
                int x1 = elem_pos(&U, i, &a1, &b1, NULL), x2 = elem_pos(&U, i + 1, &a2, &b2, NULL);
                if (Q_cmp(b1, a2) >= 0 && !(x1 && x2)) again = 1;
            }
            if (!again) break;
            if (!upart_refine(&U)) { if (LIVE(&c)) ifail(&c, TR_MATH_NOT_CONVERGED); break; }
        }
        for (int32_t i = 0; i < U.ne && LIVE(&c); i++) {         /* the points' feasibility */
            if (U.e[i].kind == 0) U.e[i].feasible = cmp_holds(U.op, 0);
            else U.e[i].feasible = cmp_holds(U.op, up_sign(&c, &U.F, U.e[i].at));
        }
        /* pieces 0 .. 2 ne: odd 2i + 1 = element i, even 2i = the open segment (e_{i-1}, e_i) (0: (-inf, e_0) when
         * the domain is unbounded below, 2 ne: (e_{ne-1}, +inf) when unbounded above) */
        int32_t np = 2 * U.ne, x0piece = 0, below = 0;
        for (int32_t i = 0; i < U.ne && LIVE(&c); i++) {
            const tr_bigq *lo, *hi;
            int ex = elem_pos(&U, i, &lo, &hi, NULL);
            if (ex && Q_cmp(lo, pr->x0) == 0) { x0piece = 2 * i + 1; below = -1; break; }
            if (ex ? Q_cmp(lo, pr->x0) < 0 : Q_cmp(hi, pr->x0) <= 0) below = i + 1;
        }
        if (below >= 0) x0piece = 2 * below;
        ucand cd[2];
        memset(cd, 0, sizeof cd);
        for (int k = 0; k < 2 && LIVE(&c); k++) {
            int side = k == 0 ? 1 : -1;
            cd[k].side = side;
            for (int32_t q = x0piece + side; q >= 0 && q <= np && LIVE(&c); q += side) {
                if ((q == 0 && Xlo) || (q == np && Xhi)) continue;              /* no segment past a finite end */
                if (q & 1) {
                    int32_t i = (q - 1) / 2;
                    if (U.e[i].feasible) { cd[k].found = 1; cd[k].attained = 1; cd[k].elem = i; break; }
                    continue;
                }
                int32_t i = q / 2 - 1, j = q / 2;                            /* the segment (e_i, e_j) */
                const tr_bigq *ilo = NULL, *ihi = NULL, *jlo = NULL, *jhi = NULL;
                if (i >= 0) elem_pos(&U, i, &ilo, &ihi, NULL);
                if (j < U.ne) elem_pos(&U, j, &jlo, &jhi, NULL);
                if (seg_feasible(&U, i, j, ihi, jlo)) {                       /* the near end: an infimum */
                    int32_t near = side > 0 ? i : j;
                    if (near >= 0 && near < U.ne) { cd[k].found = 1; cd[k].attained = 0; cd[k].elem = near; }
                    break;
                }
            }
        }
        /* compare the two sides' distances exactly */
        int pick[2] = {cd[0].found, cd[1].found};
        for (int round = 0; pick[0] && pick[1] && LIVE(&c) && round < 4096; round++) {
            const tr_bigq *ra, *rb, *la, *lb;
            int rex = elem_pos(&U, cd[0].elem, &ra, &rb, NULL), lex = elem_pos(&U, cd[1].elem, &la, &lb, NULL);
            tr_bigq *Rlo = Q_sub(&c, ra, pr->x0), *Rhi = Q_sub(&c, rb, pr->x0);
            tr_bigq *Llo = Q_sub(&c, pr->x0, lb), *Lhi = Q_sub(&c, pr->x0, la);
            int decided = 1;
            if (!LIVE(&c)) decided = 1;
            else if (rex && lex) { int cmp = Q_cmp(Rlo, Llo); if (cmp < 0) pick[1] = 0; else if (cmp > 0) pick[0] = 0; }
            else if (Q_cmp(Rhi, Llo) <= 0) pick[1] = 0;          /* one side open: strictly nearer */
            else if (Q_cmp(Lhi, Rlo) <= 0) pick[0] = 0;
            else if (rex || lex) {                               /* an exact distance v inside the other's interval */
                const tr_bigq *v = rex ? Rlo : Llo;
                tr_bigq *pt = rex ? Q_sub(&c, pr->x0, v) : Q_add(&c, pr->x0, v);   /* the mirrored point */
                if (up_sign(&c, &U.F, pt) != 0) decided = 0;     /* zero: the other root is exactly there, a tie */
                tr_bigq_free(pt);
            } else {
                tr_bigq *jlo = Q_cmp(Rlo, Llo) > 0 ? Rlo : Llo, *jhi = Q_cmp(Rhi, Lhi) < 0 ? Rhi : Lhi;
                if (!roots_tie(&c, &U.F, pr->x0, jlo, jhi)) decided = 0;
            }
            tr_bigq_free(Rlo); tr_bigq_free(Rhi); tr_bigq_free(Llo); tr_bigq_free(Lhi);
            if (decided) break;
            if (!upart_refine(&U)) { if (LIVE(&c)) ifail(&c, TR_MATH_NOT_CONVERGED); break; }
        }
        /* an attained candidate beats an infimum at the same distance */
        if (pick[0] && pick[1] && cd[0].attained != cd[1].attained) pick[cd[0].attained ? 1 : 0] = 0;
        if (pr->width && LIVE(&c)) {                              /* the reported intervals refined to the width */
            int32_t e = 0;
            tr_inv_roots *ref = tr_inv_roots_refine(U.roots, pr->width, NULL, c.b, c.r, &e);
            if (!ref) ifail(&c, e); else { tr_inv_roots_free(U.roots); U.roots = ref; }
        }
        int any = 0, attained = 0;
        for (int k = 0; k < 2 && LIVE(&c); k++) {
            if (!pick[k]) continue;
            any = 1;
            if (cd[k].attained) attained = 1;
            tr_inv_root_info ri;
            const tr_bigq *lo, *hi;
            int ex = elem_pos(&U, cd[k].elem, &lo, &hi, &ri);
            sol_t *s = res_add(&c, r);
            if (!s) break;
            s->s.exact = ex; s->s.attained = cd[k].attained; s->s.side = cd[k].side;
            s->s.how = U.e[cd[k].elem].kind == 0 ? ri.how : 0;
            s->s.multiplicity = U.e[cd[k].elem].kind == 0 ? ri.multiplicity : 0;
            s->xlo = Q_arr(&c, 1); s->xhi = Q_arr(&c, 1); s->dlo = Q_arr(&c, 1); s->dhi = Q_arr(&c, 1);
            if (!s->xlo || !s->xhi || !s->dlo || !s->dhi) break;
            s->xlo[0] = Q_copy(&c, lo); s->xhi[0] = Q_copy(&c, hi);
            s->dlo[0] = Q_sub(&c, lo, pr->x0); s->dhi[0] = Q_sub(&c, hi, pr->x0);
            tr_bigq *alo = Q_abs(&c, s->dlo[0]), *ahi = Q_abs(&c, s->dhi[0]);
            if (Q_cmp(alo, ahi) > 0) { tr_bigq *t = alo; alo = ahi; ahi = t; }
            s->clo = cost_scalar(&c, pr->cost, w, alo); s->chi = cost_scalar(&c, pr->cost, w, ahi);
            tr_bigq_free(alo); tr_bigq_free(ahi);
            if (ex) {                                             /* f at the point: t at a root */
                tr_bigq *v = up_eval(&c, &U.F, lo);
                if (pr->t) Q_set(&v, Q_add(&c, v, pr->t));
                s->vlo = v; s->vhi = Q_copy(&c, v);
            } else if (U.e[cd[k].elem].kind == 0) { s->vlo = pr->t ? Q_copy(&c, pr->t) : Q_int(&c, 0); s->vhi = Q_copy(&c, s->vlo); }
        }
        if (LIVE(&c)) {
            r->info.complete = 1;
            if (!any) r->info.status = TR_INTERV_INFEASIBLE;
            else {
                r->info.status = attained ? TR_INTERV_OPTIMAL : TR_INTERV_UNATTAINED;
                res_one(&c, r, TR_INTERV_PART_LOWER, r->sol[0].clo);
                res_one(&c, r, TR_INTERV_PART_UPPER, r->sol[0].chi);
                if (r->nsol > 1) r->info.unique = 0;
                else if (!attained) r->info.unique = -1;
                else if (pr->cost != TR_INTERV_L0) r->info.unique = 1;
                else {                                            /* L0: unique iff the feasible set is one point */
                    int points = 0, segs = 0;
                    for (int32_t i = 0; i < U.ne; i++) points += U.e[i].feasible;
                    for (int32_t q = 0; q <= 2 * U.ne && LIVE(&c); q += 2) {
                        if ((q == 0 && Xlo) || (q == 2 * U.ne && Xhi)) continue;
                        int32_t i = q / 2 - 1, j = q / 2;
                        const tr_bigq *a1 = NULL, *b1 = NULL, *a2 = NULL, *b2 = NULL;
                        if (i >= 0) elem_pos(&U, i, &a1, &b1, NULL);
                        if (j < U.ne) elem_pos(&U, j, &a2, &b2, NULL);
                        segs += seg_feasible(&U, i, j, b1, a2);
                    }
                    r->info.unique = points == 1 && segs == 0 ? 1 : 0;
                }
            }
        }
    }
    if (r) r->info.iterations = c.iters;
    for (int32_t i = 0; U.e && i < U.ne; i++) tr_bigq_free(U.e[i].at);
    free(U.e);
    tr_inv_roots_free(U.roots);
    up_free(&U.F);
    tr_bigq_free(w); tr_bigq_free(Xlo); tr_bigq_free(Xhi);
    return res_done(&c, r, error);
}

/* ================================================================ 3. finite grids */
typedef struct { uint8_t e[TR_INTERV_GRID_DIMS]; tr_bigq *c; } gterm;
typedef struct { int32_t nt; gterm *t; uint8_t allowed; } gcond;
static void gcond_free(gcond *g) { for (int32_t k = 0; g->t && k < g->nt; k++) tr_bigq_free(g->t[k].c); free(g->t); g->t = NULL; g->nt = 0; }
/* the terms of a spec over nd coordinates (vars: the first-slice slots of slice specs), plus a constant */
static void gcond_read(ictx *c, int32_t nd, const int32_t *vars, const tr_polyopt_spec *s, const tr_bigq *minus,
                       gcond *g) {
    memset(g, 0, sizeof *g);
    int32_t nt = s->slice ? tr_math_poly_term_count(s->slice) : s->nterms;
    if (nt < 0 || nt > TR_POLYOPT_MAX_TERMS || (!s->slice && nt && (!s->exps || !s->coef)) || (s->slice && !vars)) {
        ifail(c, nt > TR_POLYOPT_MAX_TERMS ? TR_MATH_UNSUPPORTED : TR_MATH_INVALID);
        return;
    }
    g->t = (gterm *)zalloc((uint64_t)(nt + 1) * sizeof *g->t);
    if (!g->t) { ifail(c, TR_MATH_MEMORY); return; }
    for (int32_t k = 0; k < nt && LIVE(c); k++) {
        if (s->slice) {
            tr_math_term mt;
            if (tr_math_poly_get_term(s->slice, k, &mt)) { ifail(c, TR_MATH_INVALID); break; }
            for (int32_t slot = 0; slot < TR_MATH_VARIABLES; slot++) {
                if (!mt.exponents[slot]) continue;
                int32_t v = -1;
                for (int32_t d = 0; d < nd; d++) if (vars[d] == slot) v = d;
                if (v < 0) { ifail(c, TR_MATH_DOMAIN_ERROR); break; }
                g->t[g->nt].e[v] = mt.exponents[slot];
            }
            g->t[g->nt].c = Q_q64(c, mt.coefficient.numerator, mt.coefficient.denominator);
        } else {
            if (!s->coef[k]) { ifail(c, TR_MATH_INVALID); break; }
            for (int32_t d = 0; d < nd; d++) g->t[g->nt].e[d] = s->exps[(uint64_t)k * nd + d];
            g->t[g->nt].c = Q_copy(c, s->coef[k]);
        }
        g->nt++;
    }
    if (minus && LIVE(c)) { g->t[g->nt].c = Q_neg(c, minus); g->nt++; }
}
/* the exact value of a condition at the point x (nd rationals) */
static tr_bigq *gcond_eval(ictx *c, const gcond *g, int32_t nd, tr_bigq *const *x) {
    tr_bigq *acc = Q_int(c, 0);
    for (int32_t k = 0; k < g->nt && LIVE(c); k++) {
        if (Q_zero(g->t[k].c)) continue;
        tr_bigq *t = Q_copy(c, g->t[k].c);
        for (int32_t d = 0; d < nd && LIVE(c); d++)
            for (uint8_t q = 0; q < g->t[k].e[d]; q++) Q_set(&t, Q_mul(c, t, x[d]));
        Q_set(&acc, Q_add(c, acc, t));
        tr_bigq_free(t);
    }
    return acc;
}
static uint8_t allowed_of(int32_t op) {
    switch (op) {
    case TR_INTERV_EQ: return TR_INTERV_SIGN_ZERO;
    case TR_INTERV_NE: return TR_INTERV_SIGN_NEG | TR_INTERV_SIGN_POS;
    case TR_INTERV_LT: return TR_INTERV_SIGN_NEG;
    case TR_INTERV_LE: return TR_INTERV_SIGN_NEG | TR_INTERV_SIGN_ZERO;
    case TR_INTERV_GT: return TR_INTERV_SIGN_POS;
    default: return TR_INTERV_SIGN_POS | TR_INTERV_SIGN_ZERO;
    }
}
struct tr_interv_grid_plan_s {
    uint32_t magic;
    int32_t cost, nconds, reserved;
    tr_interv_grid_view v;
    tr_interv_grid_info info;
    tr_gfp *fields;
    uint64_t *garner, *half, *cres, *vres, *order;
    uint8_t *exps;
    int64_t *ctab;
    tr_bigq **values, **x0, *D;
    gcond target;                            /* f (without t): the value at a solution */
};
static int plan_valid(const tr_interv_grid_plan *p) { return p && p->magic == PLAN_MAGIC; }
void tr_interv_grid_plan_free(tr_interv_grid_plan *p) {
    if (!plan_valid(p)) return;
    free(p->fields); free(p->garner); free(p->half); free(p->cres); free(p->vres); free(p->order); free(p->exps);
    free(p->ctab);
    Q_free(p->values, p->v.nvalues); Q_free(p->x0, (uint64_t)p->v.ndims); tr_bigq_free(p->D);
    gcond_free(&p->target);
    p->magic = 0;
    free(p);
}
int32_t tr_interv_grid_get_view(const tr_interv_grid_plan *p, tr_interv_grid_view *v) {
    if (!plan_valid(p) || !v) return TR_MATH_INVALID;
    *v = p->v;
    return TR_MATH_OK;
}
int32_t tr_interv_grid_get_info(const tr_interv_grid_plan *p, tr_interv_grid_info *out) {
    if (!plan_valid(p) || !out) return TR_MATH_INVALID;
    *out = p->info;
    return TR_MATH_OK;
}
uint64_t tr_interv_grid_device_need(const tr_interv_grid_view *v, uint64_t count) {
    return v ? tr_interv_grid_device_bytes(v, count) : 0;
}
int32_t tr_interv_grid_coords(const tr_interv_grid_plan *p, uint64_t idx, uint64_t *coords) {
    if (!plan_valid(p) || !coords) return TR_MATH_INVALID;
    tr_interv_digits(&p->v, idx, coords);
    return TR_MATH_OK;
}
/* lcm of the denominators of n rationals (a positive integer as a rational) */
static tr_bigq *lcm_den(ictx *c, tr_bigq *const *v, uint64_t n) {
    tr_bigq *L = Q_int(c, 1);
    for (uint64_t i = 0; i < n && LIVE(c); i++) {
        if (!v[i]) continue;
        const tr_bigint *d = tr_bigq_denominator(v[i]), *l = tr_bigq_numerator(L);
        int32_t e = 0;
        tr_bigint *g = tr_bigint_gcd(l, d, c->b, c->r, &e);
        tr_bigint *q = g ? tr_bigint_divexact(d, g, c->b, c->r, &e) : NULL;
        tr_bigint *m = q ? tr_bigint_mul(l, q, c->b, c->r, &e) : NULL;
        tr_bigq *nl = m ? tr_bigq_from_bigint(m, c->b, c->r, &e) : NULL;
        tr_bigint_free(g); tr_bigint_free(q); tr_bigint_free(m);
        if (!nl) { ifail(c, e); break; }
        tr_bigq_free(L); L = nl;
    }
    return L;
}
typedef struct { const tr_bigq *v; uint64_t i; } sval;
static int sval_cmp(const void *x, const void *y) {
    const sval *a = (const sval *)x, *b = (const sval *)y;
    int c = tr_bigq_cmp(a->v, b->v);
    return c ? c : (a->i < b->i ? -1 : a->i > b->i);
}
typedef struct { int64_t cost; uint64_t idx; } cpt;
static int cpt_cmp(const void *x, const void *y) {
    const cpt *a = (const cpt *)x, *b = (const cpt *)y;
    if (a->cost != b->cost) return a->cost < b->cost ? -1 : 1;
    return a->idx < b->idx ? -1 : a->idx > b->idx;
}
/* the natural interval enclosure of a condition over the box X (outward, FND-009) */
static tr_math_interval gcond_enclose(const gcond *g, int32_t nd, const tr_math_interval *X, const tr_math_interval *C) {
    tr_math_interval acc = {0.0, 0.0};
    for (int32_t k = 0; k < g->nt; k++) {
        tr_math_interval t = C[k];
        if (t.lo == 0 && t.hi == 0) continue;
        for (int32_t d = 0; d < nd; d++) if (g->t[k].e[d]) t = tr_interval_mul(t, tr_interval_pown(X[d], g->t[k].e[d]));
        acc = tr_interval_add(acc, t);
    }
    return tr_polyopt_clean(acc);
}
tr_interv_grid_plan *tr_interv_grid_prepare(const tr_interv_grid_problem *pr, const tr_math_budget *budget,
                                            tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    if (!pr || pr->ndims < 1 || !cost_valid(pr->cost) || !op_valid(pr->op) || (pr->flip != 0 && pr->flip != 1) ||
        pr->ncons < 0 || pr->nrows < 0 || !pr->sizes || !pr->values || !pr->x0 || (pr->ncons && !pr->cons) ||
        (pr->nrows && (!pr->A || !pr->b))) { set_error(error, TR_MATH_INVALID); return NULL; }
    if (pr->semantics == TR_INTERV_CAUSAL) { set_error(error, TR_MATH_UNSUPPORTED); return NULL; }
    if (pr->semantics != TR_INTERV_ALGEBRAIC) { set_error(error, TR_MATH_INVALID); return NULL; }
    if (pr->ndims > TR_INTERV_GRID_DIMS || 1 + pr->ncons + pr->nrows > TR_INTERV_CONDS) { set_error(error, TR_MATH_UNSUPPORTED); return NULL; }
    const int32_t nd = pr->ndims;
    uint64_t grid = 1, nvalues = 0;
    for (int32_t d = 0; d < nd; d++) {
        if (pr->sizes[d] < 1) { set_error(error, TR_MATH_INVALID); return NULL; }
        grid = tr_budget_mul(grid, pr->sizes[d]); nvalues = tr_budget_add(nvalues, pr->sizes[d]);
    }
    for (uint64_t i = 0; i < nvalues; i++) if (!pr->values[i]) { set_error(error, TR_MATH_INVALID); return NULL; }
    for (int32_t d = 0; d < nd; d++) if (!pr->x0[d]) { set_error(error, TR_MATH_INVALID); return NULL; }
    tr_math_need need;
    memset(&need, 0, sizeof need);
    need.need[TR_BUDGET_VARIABLES] = (uint64_t)nd;
    need.need[TR_BUDGET_STATES] = grid;
    need.need[TR_BUDGET_BYTES] = tr_budget_add(tr_budget_mul(grid, 24), tr_budget_mul(nvalues, 8 * (TR_INV_PRIMES_MAX + 2)));
    int32_t rc = budget_need(budget, refusal, &need);
    if (rc) { set_error(error, rc); return NULL; }
    if (grid > TR_INTERV_GRID_MAX) { set_error(error, TR_MATH_UNSUPPORTED); return NULL; }
    ictx c;
    ictx_init(&c, budget, refusal);
    tr_interv_grid_plan *P = (tr_interv_grid_plan *)zalloc(sizeof *P);
    if (!P) { set_error(error, TR_MATH_MEMORY); return NULL; }
    P->magic = PLAN_MAGIC; P->cost = pr->cost;
    tr_interv_grid_view *v = &P->v;
    v->ndims = nd; v->nvalues = nvalues; v->combine = pr->cost == TR_INTERV_LINF;
    uint64_t off = 0;
    for (int32_t d = 0; d < nd; d++) { v->size[d] = pr->sizes[d]; v->voff[d] = off; off += pr->sizes[d]; }
    P->values = Q_full(&c, pr->values, nvalues);
    P->x0 = Q_full(&c, pr->x0, (uint64_t)nd);
    /* the conditions: the target f - t, the polynomial constraints, the rows */
    gcond conds[TR_INTERV_CONDS];
    memset(conds, 0, sizeof conds);
    int32_t nc = 0;
    tr_bigq *t = pr->t ? Q_copy(&c, pr->t) : Q_int(&c, 0);
    gcond_read(&c, nd, pr->vars, &pr->target, NULL, &P->target);
    gcond_read(&c, nd, pr->vars, &pr->target, t, &conds[nc]);
    tr_bigq *f0 = LIVE(&c) ? gcond_eval(&c, &conds[0], nd, P->x0) : NULL;
    int member = LIVE(&c) ? cmp_holds(pr->op, Q_sgn(f0)) : 0;
    tr_bigq_free(f0);
    int32_t op = pr->flip ? (member ? cmp_complement(pr->op) : pr->op) : pr->op;
    conds[nc++].allowed = allowed_of(op);
    for (int32_t k = 0; k < pr->ncons && LIVE(&c); k++) {
        const tr_polyopt_spec *s = &pr->cons[k];
        if (s->kind < TR_POLYOPT_LE || s->kind > TR_POLYOPT_EQ) { ifail(&c, TR_MATH_INVALID); break; }
        gcond_read(&c, nd, pr->vars, s, s->rhs, &conds[nc]);
        conds[nc++].allowed = s->kind == TR_POLYOPT_LE ? allowed_of(TR_INTERV_LE) :
                              s->kind == TR_POLYOPT_GE ? allowed_of(TR_INTERV_GE) : allowed_of(TR_INTERV_EQ);
    }
    for (int32_t i = 0; i < pr->nrows && LIVE(&c); i++) {
        gcond *g = &conds[nc];
        g->t = (gterm *)zalloc((uint64_t)(nd + 1) * sizeof *g->t);
        if (!g->t) { ifail(&c, TR_MATH_MEMORY); break; }
        for (int32_t d = 0; d < nd && LIVE(&c); d++) {
            if (!pr->A[(uint64_t)i * nd + d]) { ifail(&c, TR_MATH_INVALID); break; }
            g->t[g->nt].e[d] = 1; g->t[g->nt].c = Q_copy(&c, pr->A[(uint64_t)i * nd + d]); g->nt++;
        }
        if (!pr->b[i]) ifail(&c, TR_MATH_INVALID);
        g->t[g->nt].c = Q_neg(&c, pr->b[i]); g->nt++;
        g->allowed = allowed_of(TR_INTERV_LE);
        nc++;
    }
    tr_bigq_free(t);
    v->nconds = nc;
    P->info.ndims = nd; P->info.nconds = nc; P->info.grid = grid; P->info.base_member = member; P->info.op = op;
    uint32_t nterms = 0;
    for (int32_t k = 0; k < nc; k++) nterms += (uint32_t)conds[k].nt;
    if (LIVE(&c)) {
        need.need[TR_BUDGET_SUPPORT] = nterms;
        rc = budget_need(budget, refusal, &need);
        if (rc) ifail(&c, rc);
    }
    /* cleared numerators of the coordinates: x = m / Q_d */
    tr_bigq **Qd = Q_arr(&c, (uint64_t)nd), **num = Q_arr(&c, nvalues ? nvalues : 1), **Mx = Q_arr(&c, (uint64_t)nd);
    for (int32_t d = 0; d < nd && LIVE(&c); d++) {
        Qd[d] = lcm_den(&c, P->values + v->voff[d], v->size[d]);
        Mx[d] = Q_int(&c, 0);
        for (uint64_t i = 0; i < v->size[d] && LIVE(&c); i++) {
            num[v->voff[d] + i] = Q_mul(&c, P->values[v->voff[d] + i], Qd[d]);
            tr_bigq *a = Q_abs(&c, num[v->voff[d] + i]);
            if (Q_cmp(a, Mx[d]) > 0) Q_set(&Mx[d], a); else tr_bigq_free(a);
        }
    }
    /* the integer forms H_c = L prod Q_d^deg_d P(m / Q) and their value bound */
    tr_bigq **C = Q_arr(&c, nterms ? nterms : 1);
    P->exps = (uint8_t *)zalloc((uint64_t)(nterms ? nterms : 1) * nd);
    if (!P->exps) ifail(&c, TR_MATH_MEMORY);
    uint64_t bound_bits = 1;
    uint32_t tpos = 0;
    for (int32_t k = 0; k < nc && LIVE(&c); k++) {
        v->cond_off[k] = tpos;
        v->allowed[k] = conds[k].allowed;
        int32_t deg[TR_INTERV_GRID_DIMS] = {0};
        tr_bigq **cf = Q_arr(&c, (uint64_t)(conds[k].nt ? conds[k].nt : 1));
        for (int32_t q = 0; cf && q < conds[k].nt; q++) {
            cf[q] = conds[k].t[q].c;
            if (Q_zero(cf[q])) continue;
            for (int32_t d = 0; d < nd; d++) if (conds[k].t[q].e[d] > deg[d]) deg[d] = conds[k].t[q].e[d];
        }
        tr_bigq *L = cf ? lcm_den(&c, cf, (uint64_t)conds[k].nt) : NULL;
        if (cf) { for (int32_t q = 0; q < conds[k].nt; q++) cf[q] = NULL; free(cf); }
        tr_bigq *bnd = Q_int(&c, 0);
        for (int32_t q = 0; q < conds[k].nt && LIVE(&c); q++) {
            tr_bigq *ct = Q_mul(&c, conds[k].t[q].c, L), *mb = Q_int(&c, 1);
            for (int32_t d = 0; d < nd && LIVE(&c); d++) {
                for (int32_t e = conds[k].t[q].e[d]; e < deg[d]; e++) Q_set(&ct, Q_mul(&c, ct, Qd[d]));
                for (int32_t e = 0; e < conds[k].t[q].e[d]; e++) Q_set(&mb, Q_mul(&c, mb, Mx[d]));
            }
            tr_bigq *a = Q_abs(&c, ct), *tb = Q_mul(&c, a, mb);
            Q_set(&bnd, Q_add(&c, bnd, tb));
            tr_bigq_free(a); tr_bigq_free(tb); tr_bigq_free(mb);
            C[tpos] = ct;
            for (int32_t d = 0; d < nd; d++) P->exps[(uint64_t)tpos * nd + d] = conds[k].t[q].e[d];
            tpos++;
        }
        if (LIVE(&c)) {
            uint64_t bits = tr_bigint_bit_length(tr_bigq_numerator(bnd)) + 1;
            if (bits > bound_bits) bound_bits = bits;
        }
        tr_bigq_free(bnd); tr_bigq_free(L);
    }
    v->cond_off[nc] = tpos;
    v->nterms = tpos;
    uint64_t kk = (bound_bits + 2 + 60) / 61;
    if (kk < 1) kk = 1;
    if (LIVE(&c) && kk > TR_INV_PRIMES_MAX) ifail(&c, TR_MATH_OVERFLOW);
    const uint32_t k = (uint32_t)kk;
    v->nprimes = (int32_t)k;
    P->info.nprimes = (int32_t)k; P->info.bound_bits = bound_bits;
    P->fields = (tr_gfp *)zalloc((uint64_t)k * sizeof(tr_gfp));
    P->garner = (uint64_t *)zalloc((uint64_t)k * k * 8);
    P->half = (uint64_t *)zalloc((uint64_t)k * 8);
    P->cres = (uint64_t *)zalloc((uint64_t)k * (tpos ? tpos : 1) * 8);
    P->vres = (uint64_t *)zalloc((uint64_t)k * (nvalues ? nvalues : 1) * 8);
    if (!P->fields || !P->garner || !P->half || !P->cres || !P->vres) ifail(&c, TR_MATH_MEMORY);
    for (uint32_t j = 0; j < k && LIVE(&c); j++) {
        uint64_t pj = 0;
        if (tr_inv_prime((int32_t)j, &pj)) { ifail(&c, TR_MATH_INVALID); break; }
        tr_gfp_fill(pj, &P->fields[j]);
    }
    for (uint32_t j = 0; j < k && LIVE(&c); j++)
        for (uint32_t s = 0; s < j; s++)
            P->garner[(uint64_t)j * k + s] = tr_gfp_inv_hd(&P->fields[j], P->fields[s].p % P->fields[j].p);
    for (uint32_t j = 0; j < k && LIVE(&c); j++) {                /* the Garner digits of (M - 1) / 2 */
        const tr_gfp *F = &P->fields[j];
        uint64_t x = (F->p - 1) / 2;
        for (uint32_t s = 0; s < j; s++) {
            uint64_t dt = P->half[s] >= F->p ? P->half[s] - F->p : P->half[s];
            x = tr_gfp_mul_hd(F, tr_gfp_sub_hd(F, x, dt), P->garner[(uint64_t)j * k + s]);
        }
        P->half[j] = x;
    }
    for (uint32_t j = 0; j < k && LIVE(&c); j++) {
        for (uint32_t q = 0; q < tpos; q++)
            if (tr_bigint_mod_u64(tr_bigq_numerator(C[q]), P->fields[j].p, &P->cres[(uint64_t)j * tpos + q])) ifail(&c, TR_MATH_INVALID);
        for (uint64_t i = 0; i < nvalues; i++)
            if (tr_bigint_mod_u64(tr_bigq_numerator(num[i]), P->fields[j].p, &P->vres[(uint64_t)j * nvalues + i])) ifail(&c, TR_MATH_INVALID);
    }
    /* the cost tables (integers: the exact costs times the common denominator D) */
    tr_bigq **terms = Q_arr(&c, nvalues ? nvalues : 1);
    for (int32_t d = 0; d < nd && LIVE(&c); d++) {
        tr_bigq *wd = pr->weight && pr->weight[d] ? Q_copy(&c, pr->weight[d]) : Q_int(&c, 1);
        if (Q_sgn(wd) < 0 || (pr->cost == TR_INTERV_L0 && Q_sgn(wd) == 0)) ifail(&c, TR_MATH_INVALID);
        for (uint64_t i = 0; i < v->size[d] && LIVE(&c); i++) {
            tr_bigq *dd = Q_sub(&c, P->values[v->voff[d] + i], P->x0[d]);
            terms[v->voff[d] + i] = cost_scalar(&c, pr->cost, wd, dd);
            tr_bigq_free(dd);
        }
        tr_bigq_free(wd);
    }
    P->D = lcm_den(&c, terms, nvalues);
    P->ctab = (int64_t *)zalloc((nvalues ? nvalues : 1) * 8);
    if (!P->ctab) ifail(&c, TR_MATH_MEMORY);
    {
        tr_bigq *total = Q_int(&c, 0);
        for (int32_t d = 0; d < nd && LIVE(&c); d++) {
            tr_bigq *mx = Q_int(&c, 0);
            for (uint64_t i = 0; i < v->size[d] && LIVE(&c); i++) {
                tr_bigq *it = Q_mul(&c, terms[v->voff[d] + i], P->D);
                int64_t iv = 0;
                tr_num_rational q;
                if (tr_bigq_to_q64(it, &q) || q.denominator != 1) ifail(&c, TR_MATH_OVERFLOW);
                else iv = q.numerator;
                P->ctab[v->voff[d] + i] = iv;
                if (Q_cmp(it, mx) > 0) Q_set(&mx, it); else tr_bigq_free(it);
            }
            if (v->combine) { if (Q_cmp(mx, total) > 0) Q_set(&total, mx); else tr_bigq_free(mx); }
            else { Q_set(&total, Q_add(&c, total, mx)); tr_bigq_free(mx); }
        }
        tr_num_rational q;
        if (LIVE(&c) && (tr_bigq_to_q64(total, &q) || q.numerator == INT64_MAX)) ifail(&c, TR_MATH_OVERFLOW);
        tr_bigq_free(total);
    }
    Q_free(terms, nvalues ? nvalues : 1);
    v->ctab = P->ctab;                                            /* the cost of a point is read below */
    /* the enclosure pre-pass: blocks of sorted values per coordinate; a block where some condition's enclosure
     * excludes every allowed sign holds no feasible point */
    uint64_t per = 1;
    while (1) { uint64_t t2 = 1, nx = per + 1; int ok = 1; for (int32_t d = 0; d < nd && ok; d++) { t2 *= nx; if (t2 > TR_INTERV_BLOCKS) ok = 0; } if (!ok) break; per = nx; }
    uint64_t nb[TR_INTERV_GRID_DIMS], nblocks = 1;
    sval *sorted = (sval *)zalloc((nvalues ? nvalues : 1) * sizeof *sorted);
    if (!sorted) ifail(&c, TR_MATH_MEMORY);
    for (int32_t d = 0; d < nd && LIVE(&c); d++) {
        uint64_t want = (v->size[d] + 3) / 4;                     /* blocks of about four values or more */
        nb[d] = want < per ? want : per;
        if (!nb[d]) nb[d] = 1;
        nblocks *= nb[d];
        for (uint64_t i = 0; i < v->size[d]; i++) { sorted[v->voff[d] + i].v = P->values[v->voff[d] + i]; sorted[v->voff[d] + i].i = i; }
        qsort(sorted + v->voff[d], (size_t)v->size[d], sizeof *sorted, sval_cmp);
    }
    tr_math_interval *CE = (tr_math_interval *)zalloc((uint64_t)(nterms ? nterms : 1) * sizeof *CE);
    if (!CE) ifail(&c, TR_MATH_MEMORY);
    uint32_t cpos = 0;
    for (int32_t q = 0; q < nc && LIVE(&c); q++)
        for (int32_t t2 = 0; t2 < conds[q].nt; t2++) {
            tr_math_interval iv = {0.0, 0.0};
            if (!Q_zero(conds[q].t[t2].c)) tr_bigq_enclose(conds[q].t[t2].c, &iv);
            CE[cpos++] = iv;
        }
    cpt *pts = (cpt *)zalloc((grid ? grid : 1) * sizeof *pts);
    if (!pts) ifail(&c, TR_MATH_MEMORY);
    uint64_t npts = 0, pruned_blocks = 0;
    for (uint64_t b = 0; b < nblocks && LIVE(&c); b++) {
        uint64_t bd[TR_INTERV_GRID_DIMS], rem = b, lo[TR_INTERV_GRID_DIMS], hi[TR_INTERV_GRID_DIMS];
        for (int32_t d = nd; d-- > 0;) { bd[d] = rem % nb[d]; rem /= nb[d]; }
        tr_math_interval X[TR_INTERV_GRID_DIMS];
        for (int32_t d = 0; d < nd; d++) {
            lo[d] = bd[d] * v->size[d] / nb[d]; hi[d] = (bd[d] + 1) * v->size[d] / nb[d];
            tr_math_interval a, z;
            tr_bigq_enclose(sorted[v->voff[d] + lo[d]].v, &a);
            tr_bigq_enclose(sorted[v->voff[d] + hi[d] - 1].v, &z);
            X[d].lo = a.lo; X[d].hi = z.hi;
        }
        int dead = 0;
        cpos = 0;
        for (int32_t q = 0; q < nc && !dead; q++) {
            tr_math_interval E = gcond_enclose(&conds[q], nd, X, CE + cpos);
            cpos += (uint32_t)conds[q].nt;
            uint8_t poss = (E.lo < 0 ? TR_INTERV_SIGN_NEG : 0) | (E.lo <= 0 && E.hi >= 0 ? TR_INTERV_SIGN_ZERO : 0) |
                           (E.hi > 0 ? TR_INTERV_SIGN_POS : 0);
            if (!(poss & conds[q].allowed)) dead = 1;
        }
        if (dead) { pruned_blocks++; continue; }
        uint64_t cnt[TR_INTERV_GRID_DIMS], tot = 1;              /* every point of the block */
        for (int32_t d = 0; d < nd; d++) { cnt[d] = hi[d] - lo[d]; tot *= cnt[d]; }
        for (uint64_t t2 = 0; t2 < tot; t2++) {
            uint64_t r2 = t2, idx = 0;
            uint64_t dig[TR_INTERV_GRID_DIMS];
            for (int32_t d = nd; d-- > 0;) { dig[d] = sorted[v->voff[d] + lo[d] + r2 % cnt[d]].i; r2 /= cnt[d]; }
            for (int32_t d = 0; d < nd; d++) idx = idx * v->size[d] + dig[d];
            pts[npts].idx = idx;
            pts[npts].cost = tr_interv_point_cost(v, dig);
            npts++;
        }
    }
    if (LIVE(&c)) qsort(pts, (size_t)npts, sizeof *pts, cpt_cmp);
    P->order = (uint64_t *)zalloc((npts ? npts : 1) * 8);
    if (!P->order) ifail(&c, TR_MATH_MEMORY);
    for (uint64_t i = 0; P->order && i < npts; i++) P->order[i] = pts[i].idx;
    free(pts); free(sorted); free(CE);
    v->total = npts;
    P->info.total = npts; P->info.pruned = grid - npts; P->info.blocks = nblocks; P->info.blocks_pruned = pruned_blocks;
    v->fields = P->fields; v->garner = P->garner; v->half = P->half; v->exps = P->exps; v->cres = P->cres;
    v->vres = P->vres; v->ctab = P->ctab; v->order = P->order;
    Q_free(C, nterms ? nterms : 1); Q_free(Qd, (uint64_t)nd); Q_free(num, nvalues ? nvalues : 1); Q_free(Mx, (uint64_t)nd);
    for (int32_t q = 0; q < TR_INTERV_CONDS; q++) gcond_free(&conds[q]);
    if (LIVE(&c) && !tr_interv_grid_view_valid(v)) ifail(&c, TR_MATH_INVALID);
    if (c.rc) { tr_interv_grid_plan_free(P); set_error(error, c.rc); return NULL; }
    set_error(error, TR_MATH_OK);
    return P;
}
int tr_interv_grid_eval(const tr_interv_grid_view *v, uint64_t start, uint64_t count, int64_t match, uint64_t *feasible,
                        uint64_t *ties, int64_t *mincost, const tr_math_budget *budget, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!tr_interv_grid_view_valid(v) || start > v->total || count > v->total - start || !mincost ||
        (count && (!feasible || !ties))) return TR_MATH_INVALID;
    tr_math_need need;
    memset(&need, 0, sizeof need);
    need.need[TR_BUDGET_OUTPUT] = count;
    int32_t rc = budget_need(budget, refusal, &need);
    if (rc) return rc;
    uint64_t words = (count + 63) / 64;
    memset(feasible, 0, (size_t)words * 8);
    memset(ties, 0, (size_t)words * 8);
    int64_t best = INT64_MAX;
    for (uint64_t i = 0; i < count; i++) {
        int64_t cost = 0;
        if (tr_interv_grid_point(v, v->order[start + i], &cost)) {
            feasible[i >> 6] |= (uint64_t)1 << (i & 63);
            if (cost < best) best = cost;
        }
    }
    int64_t m = match == INT64_MIN ? best : match;
    for (uint64_t i = 0; i < count; i++) {
        if (!((feasible[i >> 6] >> (i & 63)) & 1)) continue;
        uint64_t dig[TR_INTERV_GRID_DIMS];
        tr_interv_digits(v, v->order[start + i], dig);
        if (tr_interv_point_cost(v, dig) == m) ties[i >> 6] |= (uint64_t)1 << (i & 63);
    }
    *mincost = best;
    return TR_MATH_OK;
}
tr_interv_result *tr_interv_grid_search(const tr_interv_grid_plan *P, uint64_t start, tr_interv_grid_fn fn,
                                        const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    if (!plan_valid(P) || start > P->v.total) { set_error(error, TR_MATH_INVALID); return NULL; }
    if (!fn) fn = tr_interv_grid_eval;
    ictx c;
    ictx_init(&c, budget, refusal);
    const tr_interv_grid_view *v = &P->v;
    const int32_t nd = v->ndims;
    uint64_t out_limit = budget ? budget->limit[TR_BUDGET_OUTPUT] : ((uint64_t)1 << 24);
    tr_interv_result *r = res_new(&c, TR_INTERV_M_GRID, nd, P->cost);
    if (r) {
        r->info.op = P->info.op; r->info.base_member = P->info.base_member; r->info.pruned = P->info.pruned;
        r->info.total = v->total; r->info.bound_bits = P->info.bound_bits; r->info.nprimes = P->info.nprimes;
    }
    tr_bigq *f0 = LIVE(&c) ? gcond_eval(&c, &P->target, nd, P->x0) : NULL;
    res_one(&c, r, TR_INTERV_PART_BASE, f0);
    tr_bigq_free(f0);
    uint64_t chunk = 4096, rank = start, examined = 0;
    int64_t cstar = INT64_MIN;
    tr_math_budget eb;                                            /* the evaluator's own budget: output is the ties' */
    if (budget) eb = *budget; else tr_budget_default(&eb);
    eb.limit[TR_BUDGET_OUTPUT] = UINT64_MAX;
    int complete = 0, reason = TR_INTERV_REASON_NONE;
    uint64_t *tie_ranks = NULL, ntie = 0, captie = 0;
    while (LIVE(&c)) {
        if (rank >= v->total) { complete = 1; break; }
        uint64_t room = c.limit > c.iters ? c.limit - c.iters : 0;
        if (!room) { reason = TR_INTERV_REASON_ITERATIONS; break; }
        uint64_t count = chunk;
        if (count > v->total - rank) count = v->total - rank;
        if (count > room) count = room;
        uint64_t words = (count + 63) / 64;
        uint64_t *fe = (uint64_t *)zalloc(words * 8), *ti = (uint64_t *)zalloc(words * 8);
        int64_t mc = INT64_MAX;
        if (!fe || !ti) ifail(&c, TR_MATH_MEMORY);
        int32_t rc2 = LIVE(&c) ? fn(v, rank, count, cstar, fe, ti, &mc, &eb, refusal) : 0;
        if (rc2) ifail(&c, rc2);
        c.iters += count; examined += count;
        if (LIVE(&c) && cstar == INT64_MIN && mc != INT64_MAX) cstar = mc;
        int over = 0;
        uint64_t stop_at = 0;
        for (uint64_t i = 0; LIVE(&c) && cstar != INT64_MIN && i < count; i++) {
            if (!((ti[i >> 6] >> (i & 63)) & 1)) continue;
            if (ntie >= out_limit) { over = 1; stop_at = rank + i; break; }   /* the output budget: stop before it */
            if (ntie == captie) {
                uint64_t cap = captie ? 2 * captie : 64;
                uint64_t *nt = (uint64_t *)realloc(tie_ranks, (size_t)cap * 8);
                if (!nt) { ifail(&c, TR_MATH_MEMORY); break; }
                tie_ranks = nt; captie = cap;
            }
            tie_ranks[ntie++] = rank + i;
        }
        if (over) {                                               /* resume at the first unlisted tie */
            reason = TR_INTERV_REASON_OUTPUT;
            free(fe); free(ti);
            rank = stop_at;
            break;
        }
        int ended = 0;
        if (cstar != INT64_MIN && LIVE(&c)) {                     /* the block of the minimal cost has ended */
            uint64_t dig[TR_INTERV_GRID_DIMS];
            tr_interv_digits(v, v->order[rank + count - 1], dig);
            if (tr_interv_point_cost(v, dig) > cstar) ended = 1;
        }
        free(fe); free(ti);
        rank += count;
        if (ended) { complete = 1; break; }
        if (chunk < ((uint64_t)1 << 20)) chunk *= 2;
    }
    if (LIVE(&c) && r) {
        r->info.examined = examined;
        r->info.next = rank;
        r->info.iterations = c.iters;
        r->info.complete = complete;
        r->info.reason = complete ? TR_INTERV_REASON_NONE : reason;
        tr_bigq *cq = NULL;
        if (cstar != INT64_MIN) {
            tr_bigq *ci = Q_int(&c, cstar);
            cq = Q_div(&c, ci, P->D);
            tr_bigq_free(ci);
            r->info.status = complete ? TR_INTERV_OPTIMAL : TR_INTERV_INCOMPLETE;
            res_one(&c, r, TR_INTERV_PART_LOWER, cq);
            res_one(&c, r, TR_INTERV_PART_UPPER, cq);
        } else if (complete) r->info.status = TR_INTERV_INFEASIBLE;
        else {
            r->info.status = TR_INTERV_INCOMPLETE;
            uint64_t dig[TR_INTERV_GRID_DIMS];                    /* no examined or pruned point of lower cost */
            tr_interv_digits(v, v->order[rank], dig);
            tr_bigq *ci = Q_int(&c, tr_interv_point_cost(v, dig));
            tr_bigq *lb = Q_div(&c, ci, P->D);
            res_one(&c, r, TR_INTERV_PART_LOWER, lb);
            tr_bigq_free(ci); tr_bigq_free(lb);
        }
        for (uint64_t q = 0; q < ntie && LIVE(&c); q++) {         /* every tie, exactly */
            uint64_t idx = v->order[tie_ranks[q]], dig[TR_INTERV_GRID_DIMS];
            tr_interv_digits(v, idx, dig);
            tr_bigq **d = Q_arr(&c, (uint64_t)nd), **x = Q_arr(&c, (uint64_t)nd);
            for (int32_t k = 0; d && x && k < nd && LIVE(&c); k++) {
                x[k] = Q_copy(&c, P->values[v->voff[k] + dig[k]]);
                d[k] = Q_sub(&c, x[k], P->x0[k]);
            }
            tr_bigq *fx = x ? gcond_eval(&c, &P->target, nd, x) : NULL;
            Q_free(x, (uint64_t)nd);
            sol_t *s = sol_exact(&c, r, nd, d, P->x0, Q_copy(&c, cq), fx);
            if (s) s->s.index = idx;
        }
        if (r->info.status == TR_INTERV_OPTIMAL) r->info.unique = ntie == 1 ? 1 : 0;
        else if (r->nsol > 1) r->info.unique = 0;
        tr_bigq_free(cq);
    }
    free(tie_ranks);
    return res_done(&c, r, error);
}

/* ================================================================ 4. aggregates whose rows cross the predicate */
struct tr_interv_agg_s {
    uint32_t magic, reserved;
    tr_interv_agg_info info;
    uint64_t K;                              /* distinct values of x */
    int64_t *xs;                             /* K sorted distinct logical values */
    uint64_t *pcnt, *pvc;                    /* K + 1 prefix counts of rows / of rows with v present */
    __int128 *pvs, *pxs;                     /* K + 1 prefix sums of v and of x (logical) */
};
static int agg_valid(const tr_interv_agg *a) { return a && a->magic == AGG_MAGIC; }
void tr_interv_agg_free(tr_interv_agg *a) {
    if (!agg_valid(a)) return;
    free(a->xs); free(a->pcnt); free(a->pvc); free(a->pvs); free(a->pxs);
    a->magic = 0;
    free(a);
}
int32_t tr_interv_agg_get_info(const tr_interv_agg *a, tr_interv_agg_info *out) {
    if (!agg_valid(a) || !out) return TR_MATH_INVALID;
    *out = a->info;
    return TR_MATH_OK;
}
typedef struct { int64_t x, v; int32_t vp, reserved; } arow;
static int arow_cmp(const void *p, const void *q) {
    const arow *a = (const arow *)p, *b = (const arow *)q;
    return a->x < b->x ? -1 : a->x > b->x;
}
tr_interv_agg *tr_interv_agg_build(const tr_scope *scope, int32_t feature, int32_t value, const tr_math_budget *budget,
                                   tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    tr_scope_view sv;
    if (!scope || tr_scope_get_view(scope, &sv)) { set_error(error, TR_MATH_INVALID); return NULL; }
    if (feature < 0 || feature >= sv.ncols || value < -1 || value >= sv.ncols) { set_error(error, TR_MATH_INVALID); return NULL; }
    uint64_t present = sv.col[feature].nvalid;
    tr_math_need need;
    memset(&need, 0, sizeof need);
    need.need[TR_BUDGET_VARIABLES] = value >= 0 && value != feature ? 2 : 1;
    need.need[TR_BUDGET_STATES] = present;
    need.need[TR_BUDGET_BYTES] = tr_budget_add(tr_budget_mul(present, sizeof(arow)), tr_budget_mul(present + 1, 56));
    int32_t rc = budget_need(budget, refusal, &need);
    if (rc) { set_error(error, rc); return NULL; }
    tr_interv_agg *A = (tr_interv_agg *)zalloc(sizeof *A);
    arow *rows = (arow *)zalloc((present ? present : 1) * sizeof *rows);
    if (!A || !rows) { free(A); free(rows); set_error(error, TR_MATH_MEMORY); return NULL; }
    A->magic = AGG_MAGIC;
    uint64_t n = 0, vpresent = 0;
    for (uint64_t r = 0; r < sv.rows; r++) {
        if (!tr_scope_is_selected(&sv, r) || !tr_scope_present(&sv, feature, r)) continue;
        if (n >= present) break;
        rows[n].x = tr_scope_value(&sv, feature, r);
        if (value >= 0 && tr_scope_present(&sv, value, r)) { rows[n].vp = 1; rows[n].v = tr_scope_value(&sv, value, r); vpresent++; }
        n++;
    }
    qsort(rows, (size_t)n, sizeof *rows, arow_cmp);
    uint64_t K = 0;
    for (uint64_t i = 0; i < n; i++) if (!i || rows[i].x != rows[i - 1].x) K++;
    A->K = K;
    A->xs = (int64_t *)zalloc((K ? K : 1) * 8);
    A->pcnt = (uint64_t *)zalloc((K + 1) * 8); A->pvc = (uint64_t *)zalloc((K + 1) * 8);
    A->pvs = (__int128 *)zalloc((K + 1) * 16); A->pxs = (__int128 *)zalloc((K + 1) * 16);
    if (!A->xs || !A->pcnt || !A->pvc || !A->pvs || !A->pxs) { free(rows); tr_interv_agg_free(A); set_error(error, TR_MATH_MEMORY); return NULL; }
    uint64_t k = 0;
    for (uint64_t i = 0; i < n; i++) {
        if (i && rows[i].x != rows[i - 1].x) k++;
        A->xs[k] = rows[i].x;
        A->pcnt[k + 1]++;
        A->pxs[k + 1] += rows[i].x;
        if (rows[i].vp) { A->pvc[k + 1]++; A->pvs[k + 1] += rows[i].v; }
    }
    for (uint64_t j = 0; j < K; j++) {                           /* prefix sums (|sum| < 2^127 for < 2^63 rows) */
        A->pcnt[j + 1] += A->pcnt[j]; A->pvc[j + 1] += A->pvc[j];
        A->pvs[j + 1] += A->pvs[j]; A->pxs[j + 1] += A->pxs[j];
    }
    free(rows);
    A->info.rows = sv.selected; A->info.present = n; A->info.distinct = K; A->info.vpresent = vpresent;
    A->info.bytes = (K + 1) * 56;
    A->info.xscale = sv.col[feature].scale;
    A->info.vscale = value >= 0 ? sv.col[value].scale : 0;
    A->info.has_value = value >= 0;
    A->info.shifted = value == feature;
    set_error(error, TR_MATH_OK);
    return A;
}
static tr_bigq *Q_i128(ictx *c, __int128 v) {
    int32_t e = 0;
    tr_bigint *b = tr_bigint_from_i128(tr_i128_set(v), c->b, c->r, &e);
    tr_bigq *q = b ? tr_bigq_from_bigint(b, c->b, c->r, &e) : NULL;
    tr_bigint_free(b);
    return qk(c, q, e);
}
static tr_bigq *Q_pow10(ictx *c, int32_t s) {
    tr_bigq *r = Q_int(c, 1), *ten = Q_int(c, 10);
    for (int32_t i = 0; i < s && LIVE(c); i++) Q_set(&r, Q_mul(c, r, ten));
    tr_bigq_free(ten);
    return r;
}
/* a piece of d: the threshold u = c - d equals x_k (point, kind 1) or lies strictly between x_{k-1} and x_k (gap k,
 * kind 0; k = 0: below every value, k = K: above) */
typedef struct { int32_t kind; uint64_t k; } apiece;
static apiece piece_of(ictx *c, const tr_interv_agg *A, const tr_bigq *u10) {   /* u10 = u 10^xscale */
    uint64_t lo = 0, hi = A->K;                                   /* the first value >= u */
    while (lo < hi && LIVE(c)) {
        uint64_t mid = lo + (hi - lo) / 2;
        tr_bigq *xm = Q_int(c, A->xs[mid]);
        int cmp = Q_cmp(xm, u10);
        tr_bigq_free(xm);
        if (cmp < 0) lo = mid + 1; else hi = mid;
    }
    apiece p;
    p.kind = 0; p.k = lo;
    if (lo < A->K) {
        tr_bigq *xm = Q_int(c, A->xs[lo]);
        if (!Q_cmp(xm, u10)) p.kind = 1;
        tr_bigq_free(xm);
    }
    return p;
}
/* the index range [i, j) selected by the predicate on a piece, and the excluded index (NE on a point; else -1) */
static void piece_range(const tr_interv_agg *A, apiece p, int32_t op, uint64_t *i, uint64_t *j, int64_t *ex) {
    const uint64_t K = A->K, k = p.k;
    *ex = -1;
    if (!p.kind) {
        switch (op) {
        case TR_INTERV_GE: case TR_INTERV_GT: *i = k; *j = K; break;
        case TR_INTERV_LE: case TR_INTERV_LT: *i = 0; *j = k; break;
        case TR_INTERV_EQ: *i = 0; *j = 0; break;
        default: *i = 0; *j = K; break;
        }
        return;
    }
    switch (op) {
    case TR_INTERV_GE: *i = k; *j = K; break;
    case TR_INTERV_GT: *i = k + 1; *j = K; break;
    case TR_INTERV_LE: *i = 0; *j = k + 1; break;
    case TR_INTERV_LT: *i = 0; *j = k; break;
    case TR_INTERV_EQ: *i = k; *j = k + 1; break;
    default: *i = 0; *j = K; *ex = (int64_t)k; break;
    }
}
static int in_range(uint64_t q, uint64_t i, uint64_t j, int64_t ex) { return q >= i && q < j && (int64_t)q != ex; }
/* the aggregate on a piece: A(d) = alpha + beta d (exact), NULL flag, rows in the predicate */
typedef struct { tr_bigq *alpha, *beta; int null; uint64_t rows; } aval;
static void aval_free(aval *v) { tr_bigq_free(v->alpha); tr_bigq_free(v->beta); memset(v, 0, sizeof *v); }
static aval piece_value(ictx *c, const tr_interv_agg *A, apiece p, int32_t agg, int32_t op) {
    aval r;
    memset(&r, 0, sizeof r);
    uint64_t i, j;
    int64_t ex;
    piece_range(A, p, op, &i, &j, &ex);
    uint64_t N = A->pcnt[j] - A->pcnt[i], Nv = A->pvc[j] - A->pvc[i];
    __int128 Sv = A->pvs[j] - A->pvs[i], Sx = A->pxs[j] - A->pxs[i];
    if (ex >= 0) {
        N -= A->pcnt[ex + 1] - A->pcnt[ex]; Nv -= A->pvc[ex + 1] - A->pvc[ex];
        Sv -= A->pvs[ex + 1] - A->pvs[ex]; Sx -= A->pxs[ex + 1] - A->pxs[ex];
    }
    r.rows = N;
    const int shifted = A->info.shifted;
    if (agg == TR_INTERV_COUNT_STAR) { r.alpha = Q_int(c, (int64_t)N); r.beta = Q_int(c, 0); return r; }
    if (agg == TR_INTERV_COUNT) { r.alpha = Q_int(c, (int64_t)Nv); r.beta = Q_int(c, 0); return r; }
    if (!Nv) { r.null = 1; r.alpha = Q_int(c, 0); r.beta = Q_int(c, 0); return r; }
    tr_bigq *S = Q_i128(c, shifted ? Sx : Sv), *sc = Q_pow10(c, shifted ? A->info.xscale : A->info.vscale);
    tr_bigq *Sval = Q_div(c, S, sc);
    tr_bigq_free(S); tr_bigq_free(sc);
    if (agg == TR_INTERV_SUM) { r.alpha = Sval; r.beta = Q_int(c, shifted ? (int64_t)N : 0); return r; }
    tr_bigq *cnt = Q_int(c, (int64_t)Nv);                         /* AVG */
    r.alpha = Q_div(c, Sval, cnt);
    r.beta = Q_int(c, shifted ? 1 : 0);
    tr_bigq_free(cnt); tr_bigq_free(Sval);
    return r;
}
static tr_bigq *aval_at(ictx *c, const aval *v, const tr_bigq *d) {
    tr_bigq *t = Q_mul(c, v->beta, d), *s = Q_add(c, v->alpha, t);
    tr_bigq_free(t);
    return s;
}
int32_t tr_interv_agg_value(const tr_interv_agg *A, int32_t aggregate, int32_t pred_op, const tr_bigq *cq,
                            const tr_bigq *d, tr_bigq **value, int32_t *null, uint64_t *rows,
                            const tr_math_budget *budget, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!agg_valid(A) || aggregate < TR_INTERV_COUNT_STAR || aggregate > TR_INTERV_AVG || !op_valid(pred_op) || !cq ||
        !d || !value || !null || (aggregate != TR_INTERV_COUNT_STAR && !A->info.has_value)) return TR_MATH_INVALID;
    ictx c;
    ictx_init(&c, budget, refusal);
    tr_bigq *u = Q_sub(&c, cq, d), *sc = Q_pow10(&c, A->info.xscale), *u10 = Q_mul(&c, u, sc);
    apiece p = piece_of(&c, A, u10);
    aval v = piece_value(&c, A, p, aggregate, pred_op);
    *null = v.null;
    *value = v.null ? NULL : aval_at(&c, &v, d);
    if (rows) *rows = v.rows;
    aval_free(&v);
    tr_bigq_free(u); tr_bigq_free(sc); tr_bigq_free(u10);
    if (c.rc) { tr_bigq_free(*value); *value = NULL; return c.rc; }
    return TR_MATH_OK;
}
/* an interval of d with open or closed ends (lo / hi NULL: infinite) */
typedef struct { tr_bigq *lo, *hi; int lc, hc, empty; } dset;
static void dset_free(dset *s) { tr_bigq_free(s->lo); tr_bigq_free(s->hi); memset(s, 0, sizeof *s); }
/* s = s intersected with [lo, hi] given by (lo, lc) and (hi, hc) (NULL: no bound) */
static void dset_meet(ictx *c, dset *s, const tr_bigq *lo, int lc, const tr_bigq *hi, int hc) {
    if (s->empty) return;
    if (lo) {
        int cmp = s->lo ? Q_cmp(lo, s->lo) : 1;
        if (cmp > 0) { Q_set(&s->lo, Q_copy(c, lo)); if (!s->lo) s->lo = Q_copy(c, lo); s->lc = lc; }
        else if (cmp == 0) s->lc = s->lc && lc;
    }
    if (hi) {
        int cmp = s->hi ? Q_cmp(hi, s->hi) : -1;
        if (cmp < 0) { Q_set(&s->hi, Q_copy(c, hi)); if (!s->hi) s->hi = Q_copy(c, hi); s->hc = hc; }
        else if (cmp == 0) s->hc = s->hc && hc;
    }
    if (s->lo && s->hi) {
        int cmp = Q_cmp(s->lo, s->hi);
        if (cmp > 0 || (cmp == 0 && !(s->lc && s->hc))) s->empty = 1;
    }
}
/* the feasible part of one piece within the region: up to two intervals */
static int piece_feasible(ictx *c, const tr_interv_agg *A, apiece p, const tr_interv_agg_query *q, int32_t op,
                          const tr_bigq *dL, const tr_bigq *dR, const tr_bigq *A_lo, const tr_bigq *A_hi, dset *out) {
    memset(out, 0, 2 * sizeof *out);
    out[0].empty = out[1].empty = 1;
    aval v = piece_value(c, A, p, q->aggregate, q->pred_op);
    int count = 0;
    if (!v.null && LIVE(c)) {
        dset s;
        memset(&s, 0, sizeof s);
        if (p.kind) { s.lo = Q_copy(c, dL); s.hi = Q_copy(c, dL); s.lc = s.hc = 1; }    /* a point piece */
        else { if (dL) s.lo = Q_copy(c, dL); if (dR) s.hi = Q_copy(c, dR); s.lc = s.hc = 0; }
        dset_meet(c, &s, A_lo, 1, A_hi, 1);
        if (!Q_sgn(v.beta) || p.kind) {                           /* constant on the piece (or a point) */
            tr_bigq *val = p.kind ? aval_at(c, &v, dL) : Q_copy(c, v.alpha), *df = Q_sub(c, val, q->t);
            if (cmp_holds(op, Q_sgn(df)) && !s.empty) { out[0] = s; memset(&s, 0, sizeof s); count = 1; }
            tr_bigq_free(val); tr_bigq_free(df);
        } else {                                                  /* A(d) - t = beta (d - tau), beta > 0 */
            tr_bigq *num = Q_sub(c, q->t, v.alpha), *tau = Q_div(c, num, v.beta);
            tr_bigq_free(num);
            dset s2 = s;
            s2.lo = s.lo ? Q_copy(c, s.lo) : NULL; s2.hi = s.hi ? Q_copy(c, s.hi) : NULL;
            switch (op) {
            case TR_INTERV_EQ: dset_meet(c, &s, tau, 1, tau, 1); break;
            case TR_INTERV_GT: dset_meet(c, &s, tau, 0, NULL, 0); break;
            case TR_INTERV_GE: dset_meet(c, &s, tau, 1, NULL, 0); break;
            case TR_INTERV_LT: dset_meet(c, &s, NULL, 0, tau, 0); break;
            case TR_INTERV_LE: dset_meet(c, &s, NULL, 0, tau, 1); break;
            default: dset_meet(c, &s, NULL, 0, tau, 0); dset_meet(c, &s2, tau, 0, NULL, 0); break;
            }
            if (!s.empty) { out[count++] = s; memset(&s, 0, sizeof s); }
            if (op == TR_INTERV_NE && !s2.empty) { out[count++] = s2; memset(&s2, 0, sizeof s2); }
            dset_free(&s2);
            tr_bigq_free(tau);
        }
        dset_free(&s);
    }
    aval_free(&v);
    return count;
}
tr_interv_result *tr_interv_agg_solve(const tr_interv_agg *A, const tr_interv_agg_query *q, const tr_math_budget *budget,
                                      tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    if (!agg_valid(A) || !q || q->aggregate < TR_INTERV_COUNT_STAR || q->aggregate > TR_INTERV_AVG ||
        !op_valid(q->pred_op) || !op_valid(q->op) || !cost_valid(q->cost) || (q->flip != 0 && q->flip != 1) || !q->c ||
        !q->t || (q->weight && Q_sgn(q->weight) <= 0) || (q->aggregate != TR_INTERV_COUNT_STAR && !A->info.has_value) ||
        (q->dlo && q->dhi && Q_cmp(q->dlo, q->dhi) > 0)) {
        set_error(error, q && q->semantics == TR_INTERV_CAUSAL ? TR_MATH_UNSUPPORTED : TR_MATH_INVALID);
        return NULL;
    }
    if (q->semantics == TR_INTERV_CAUSAL) { set_error(error, TR_MATH_UNSUPPORTED); return NULL; }
    if (q->semantics != TR_INTERV_ALGEBRAIC) { set_error(error, TR_MATH_INVALID); return NULL; }
    ictx c;
    ictx_init(&c, budget, refusal);
    tr_interv_result *r = res_new(&c, TR_INTERV_M_AGGREGATE, 1, q->cost);
    tr_bigq *w = q->weight ? Q_copy(&c, q->weight) : Q_int(&c, 1), *zero = Q_int(&c, 0);
    tr_bigq *sc = Q_pow10(&c, A->info.xscale);
    /* d at value index k: c - x_k */
    const uint64_t K = A->K;
    tr_bigq *u0 = Q_mul(&c, q->c, sc);
    apiece p0 = piece_of(&c, A, u0);
    tr_bigq_free(u0);
    aval v0 = piece_value(&c, A, p0, q->aggregate, q->pred_op);
    tr_bigq *df0 = v0.null ? NULL : Q_sub(&c, v0.alpha, q->t);
    int member = !v0.null && cmp_holds(q->op, Q_sgn(df0));
    int32_t op = q->flip ? (member ? cmp_complement(q->op) : q->op) : q->op;
    tr_bigq_free(df0);
    if (r) { r->info.op = op; r->info.base_member = member; }
    if (!v0.null) res_one(&c, r, TR_INTERV_PART_BASE, v0.alpha);
    uint64_t base_rows = v0.rows;
    /* the pieces in increasing d: j even = gap K - j / 2, j odd = point K - 1 - (j - 1) / 2 */
    const uint64_t np = 2 * K + 1;
    uint64_t j0 = p0.kind ? 2 * (K - 1 - p0.k) + 1 : 2 * (K - p0.k);
    struct { int found, attained; tr_bigq *d; } best[2];
    memset(best, 0, sizeof best);
    for (int k2 = 0; k2 < 2 && LIVE(&c); k2++) {
        int side = k2 == 0 ? 1 : -1;
        /* the region: [max(0, dlo), dhi] on the right, [dlo, min(0, dhi)] on the left */
        const tr_bigq *Rlo = side > 0 ? (q->dlo && Q_sgn(q->dlo) > 0 ? q->dlo : zero) : q->dlo;
        const tr_bigq *Rhi = side > 0 ? q->dhi : (q->dhi && Q_sgn(q->dhi) < 0 ? q->dhi : zero);
        if (Rlo && Rhi && Q_cmp(Rlo, Rhi) > 0) continue;
        /* start at the piece holding the region's near end */
        const tr_bigq *nearv = side > 0 ? Rlo : Rhi;
        uint64_t js = j0;
        if (nearv && Q_cmp(nearv, zero)) {
            tr_bigq *un = Q_sub(&c, q->c, nearv), *u10 = Q_mul(&c, un, sc);
            apiece pn = piece_of(&c, A, u10);
            js = pn.kind ? 2 * (K - 1 - pn.k) + 1 : 2 * (K - pn.k);
            tr_bigq_free(un); tr_bigq_free(u10);
        }
        for (int64_t j = (int64_t)js; j >= 0 && j < (int64_t)np && LIVE(&c); j += side) {
            apiece p;
            tr_bigq *dL = NULL, *dR = NULL;                       /* the piece's d range (point: dL) */
            if (j & 1) {
                p.kind = 1; p.k = K - 1 - (uint64_t)(j - 1) / 2;
                tr_bigq *x = Q_int(&c, A->xs[p.k]), *xv = Q_div(&c, x, sc);
                dL = Q_sub(&c, q->c, xv);
                tr_bigq_free(x); tr_bigq_free(xv);
            } else {
                p.kind = 0; p.k = K - (uint64_t)j / 2;
                if (p.k < K) { tr_bigq *x = Q_int(&c, A->xs[p.k]), *xv = Q_div(&c, x, sc); dL = Q_sub(&c, q->c, xv); tr_bigq_free(x); tr_bigq_free(xv); }
                if (p.k > 0) { tr_bigq *x = Q_int(&c, A->xs[p.k - 1]), *xv = Q_div(&c, x, sc); dR = Q_sub(&c, q->c, xv); tr_bigq_free(x); tr_bigq_free(xv); }
            }
            /* past the region: stop */
            int past = side > 0 ? (Rhi && dL && Q_cmp(dL, Rhi) > 0) : (Rlo && (p.kind ? Q_cmp(dL, Rlo) < 0 : (dR && Q_cmp(dR, Rlo) < 0)));
            dset fs[2];
            int nf = past ? 0 : piece_feasible(&c, A, p, q, op, dL, dR, Rlo, Rhi, fs);
            for (int t = 0; t < nf && LIVE(&c); t++) {
                const tr_bigq *e = side > 0 ? fs[t].lo : fs[t].hi;
                int closed = side > 0 ? fs[t].lc : fs[t].hc;
                if (!e) continue;                                 /* unbounded away from 0: no nearest point */
                int better = !best[k2].found || (side > 0 ? Q_cmp(e, best[k2].d) < 0 : Q_cmp(e, best[k2].d) > 0) ||
                             (!Q_cmp(e, best[k2].d) && closed && !best[k2].attained);
                if (better) { tr_bigq_free(best[k2].d); best[k2].d = Q_copy(&c, e); best[k2].found = 1; best[k2].attained = closed; }
            }
            for (int t = 0; t < nf; t++) dset_free(&fs[t]);
            tr_bigq_free(dL); tr_bigq_free(dR);
            if (past || best[k2].found) break;
        }
    }
    /* the nearer side(s): every minimal change */
    int pick[2] = {best[0].found, best[1].found};
    if (pick[0] && pick[1]) {
        tr_bigq *a0 = Q_abs(&c, best[0].d), *a1 = Q_abs(&c, best[1].d);
        int cmp = Q_cmp(a0, a1);
        if (cmp < 0) pick[1] = 0; else if (cmp > 0) pick[0] = 0;
        else if (!Q_sgn(best[0].d)) pick[1] = 0;                  /* d = 0 found from both sides */
        else if (best[0].attained != best[1].attained) pick[best[0].attained ? 1 : 0] = 0;
        tr_bigq_free(a0); tr_bigq_free(a1);
    }
    int any = 0, attained = 0;
    for (int k2 = 0; k2 < 2 && LIVE(&c); k2++) {
        if (!pick[k2]) continue;
        any = 1;
        if (best[k2].attained) attained = 1;
    }
    for (int k2 = 0; k2 < 2 && LIVE(&c); k2++) {
        if (!pick[k2] || (attained && !best[k2].attained)) continue;
        tr_bigq **d = Q_arr(&c, 1);
        if (d) d[0] = Q_copy(&c, best[k2].d);
        tr_bigq *val = NULL;
        int32_t isnull = 0;
        uint64_t rows = 0;
        tr_interv_agg_value(A, q->aggregate, q->pred_op, q->c, best[k2].d, &val, &isnull, &rows, budget, refusal);
        tr_bigq *x0a[1] = {zero};
        sol_t *s = sol_exact(&c, r, 1, d, x0a, d ? cost_scalar(&c, q->cost, w, d[0]) : NULL, val);
        if (!s) break;
        s->s.attained = best[k2].attained;
        s->s.side = Q_sgn(best[k2].d);
        s->s.null = isnull;
        s->s.rows = rows;
        /* the rows that entered and left the predicate between d = 0 and d */
        tr_bigq *un = Q_sub(&c, q->c, best[k2].d), *u10 = Q_mul(&c, un, sc);
        apiece pd = piece_of(&c, A, u10);
        tr_bigq_free(un); tr_bigq_free(u10);
        uint64_t i0, j0r, i1, j1;
        int64_t e0, e1;
        piece_range(A, p0, q->pred_op, &i0, &j0r, &e0);
        piece_range(A, pd, q->pred_op, &i1, &j1, &e1);
        for (uint64_t x = 0; x < K; x++) {
            int in0 = in_range(x, i0, j0r, e0), in1 = in_range(x, i1, j1, e1);
            uint64_t cnt = A->pcnt[x + 1] - A->pcnt[x];
            if (in1 && !in0) s->s.entered += cnt;
            if (in0 && !in1) s->s.left += cnt;
        }
        uint64_t crossed = 0;                                     /* breakpoints strictly between 0 and d */
        for (uint64_t x = 0; x < K && LIVE(&c); x++) {
            tr_bigq *xv0 = Q_int(&c, A->xs[x]), *xv = Q_div(&c, xv0, sc), *bk = Q_sub(&c, q->c, xv);
            int between = Q_sgn(best[k2].d) > 0 ? (Q_sgn(bk) > 0 && Q_cmp(bk, best[k2].d) < 0)
                                                 : (Q_sgn(bk) < 0 && Q_cmp(bk, best[k2].d) > 0);
            crossed += (uint64_t)between;
            tr_bigq_free(xv0); tr_bigq_free(xv); tr_bigq_free(bk);
        }
        if (r && crossed > r->info.crossed) r->info.crossed = crossed;
    }
    if (LIVE(&c) && r) {
        r->info.complete = 1;
        r->info.total = np;
        if (!any) r->info.status = TR_INTERV_INFEASIBLE;
        else {
            r->info.status = attained ? TR_INTERV_OPTIMAL : TR_INTERV_UNATTAINED;
            res_one(&c, r, TR_INTERV_PART_LOWER, r->sol[0].clo);
            res_one(&c, r, TR_INTERV_PART_UPPER, r->sol[0].clo);
            r->info.unique = r->nsol > 1 ? 0 : (attained && q->cost != TR_INTERV_L0 ? 1 : -1);
            if (attained && r->nsol == 1 && !Q_sgn(r->sol[0].dlo[0])) r->info.unique = 1;
        }
        r->info.examined = base_rows;
    }
    for (int k2 = 0; k2 < 2; k2++) tr_bigq_free(best[k2].d);
    aval_free(&v0);
    tr_bigq_free(w); tr_bigq_free(zero); tr_bigq_free(sc);
    return res_done(&c, r, error);
}
