/* OPT-001: exact minimizers of quadratic objectives over R^n or a box; see math_opt.h.
 * SPDX-License-Identifier: Apache-2.0 */
#include "math_opt.h"
#include "math_domain.h"
#include "math_handles.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

TR_MATH_ABI(tr_opt_box, 1);              /* the ctypes boundary of OPT-001 (math_handles.h) */
TR_MATH_ABI(tr_opt_info, 1);
TR_MATH_ABI(tr_opt_face_view, 1);

_Static_assert(sizeof(tr_opt_box) == 120, "tr_opt_box ABI");
_Static_assert(sizeof(tr_opt_info) == 104, "tr_opt_info ABI");
_Static_assert(sizeof(tr_opt_face_view) == 1464, "tr_opt_face_view ABI");

#define QUAD_MAGIC UINT32_C(0x5154504f)
#define ORES_MAGIC UINT32_C(0x5254504f)

static void set_error(int32_t *error, int32_t code) { if (error) *error = code; }
static void *zalloc(uint64_t bytes) {
    if (bytes > (uint64_t)SIZE_MAX) return NULL;
    return calloc(1, bytes ? (size_t)bytes : 1);
}

/* ================================================================ exact arithmetic with a sticky error */
typedef struct { const tr_math_budget *b; tr_math_refusal *r; int32_t rc; } octx;
static void ofail(octx *x, int32_t rc) { if (!x->rc) x->rc = rc ? rc : TR_MATH_MEMORY; }
static tr_bigq *qk(octx *x, tr_bigq *v, int32_t e) { if (!v) ofail(x, e); return v; }
#define LIVE(x) (!(x)->rc)
static tr_bigq *Q_q64(octx *x, int64_t n, int64_t d) {
    tr_num_rational q; q.numerator = n; q.denominator = d;
    int32_t e = 0;
    return x->rc ? NULL : qk(x, tr_bigq_from_q64(q, x->b, x->r, &e), e);
}
static tr_bigq *Q_int(octx *x, int64_t v) { return Q_q64(x, v, 1); }
static tr_bigq *Q_copy(octx *x, const tr_bigq *a) { int32_t e = 0; return x->rc ? NULL : qk(x, tr_bigq_copy(a, x->b, x->r, &e), e); }
static tr_bigq *Q_add(octx *x, const tr_bigq *a, const tr_bigq *c) { int32_t e = 0; return x->rc ? NULL : qk(x, tr_bigq_add(a, c, x->b, x->r, &e), e); }
static tr_bigq *Q_sub(octx *x, const tr_bigq *a, const tr_bigq *c) { int32_t e = 0; return x->rc ? NULL : qk(x, tr_bigq_sub(a, c, x->b, x->r, &e), e); }
static tr_bigq *Q_mul(octx *x, const tr_bigq *a, const tr_bigq *c) { int32_t e = 0; return x->rc ? NULL : qk(x, tr_bigq_mul(a, c, x->b, x->r, &e), e); }
static tr_bigq *Q_neg(octx *x, const tr_bigq *a) { int32_t e = 0; return x->rc ? NULL : qk(x, tr_bigq_neg(a, x->b, x->r, &e), e); }
static void Q_fma(octx *x, tr_bigq **acc, const tr_bigq *a, const tr_bigq *c) {     /* *acc += a c */
    if (x->rc || !tr_bigq_sign(a) || !tr_bigq_sign(c)) return;
    tr_bigq *p = Q_mul(x, a, c), *s = Q_add(x, *acc, p);
    tr_bigq_free(p);
    if (s) { tr_bigq_free(*acc); *acc = s; }
}
static void Q_set(tr_bigq **slot, tr_bigq *v) { if (v) { tr_bigq_free(*slot); *slot = v; } }
static tr_bigq **Q_arr(octx *x, uint64_t n) {
    if (x->rc) return NULL;
    tr_bigq **v = (tr_bigq **)zalloc(n * sizeof *v);
    if (!v) ofail(x, TR_MATH_MEMORY);
    return v;
}
static tr_bigq **Q_zeros(octx *x, uint64_t n) {
    tr_bigq **v = Q_arr(x, n);
    for (uint64_t i = 0; v && i < n && LIVE(x); i++) v[i] = Q_int(x, 0);
    return v;
}
static int Q_cmp(const tr_bigq *a, const tr_bigq *b) { return tr_bigq_cmp(a, b); }

/* ================================================================ objectives */
struct tr_opt_quad_s {
    uint32_t magic;
    int32_t n, ndesign;
    int32_t design[TR_LIN_MAX_COLS];
    tr_bigq **A, **b, *c;
};
static int quad_valid(const tr_opt_quad *q) { return q && q->magic == QUAD_MAGIC; }
void tr_opt_quad_free(tr_opt_quad *q) {
    if (!quad_valid(q)) return;
    tr_lin_q_free(q->A, (uint64_t)q->n * (uint64_t)q->n);
    tr_lin_q_free(q->b, (uint64_t)q->n);
    tr_bigq_free(q->c);
    q->magic = 0;
    free(q);
}
static int32_t quad_need(int32_t n, const tr_math_budget *budget, tr_math_refusal *refusal) {
    tr_math_need need;
    memset(&need, 0, sizeof need);
    need.need[TR_BUDGET_VARIABLES] = (uint64_t)n;
    need.need[TR_BUDGET_DEGREE] = 2;
    need.need[TR_BUDGET_TENSOR] = (uint64_t)n * (uint64_t)(n + 1);         /* A and b */
    tr_math_budget dflt;
    if (!budget) { tr_budget_default(&dflt); budget = &dflt; }
    return tr_budget_check(budget, &need, refusal);
}
static tr_opt_quad *quad_alloc(octx *x, int32_t n) {
    if (x->rc) return NULL;
    tr_opt_quad *q = (tr_opt_quad *)zalloc(sizeof *q);
    if (!q) { ofail(x, TR_MATH_MEMORY); return NULL; }
    q->n = n;
    q->A = Q_zeros(x, (uint64_t)n * (uint64_t)n);
    q->b = Q_zeros(x, (uint64_t)n);
    q->c = Q_int(x, 0);
    q->magic = QUAD_MAGIC;
    return q;
}
static tr_opt_quad *quad_done(octx *x, tr_opt_quad *q, int32_t *error) {
    if (LIVE(x) && q) {                                          /* A must be symmetric */
        for (int32_t i = 0; i < q->n && LIVE(x); i++)
            for (int32_t j = i + 1; j < q->n; j++)
                if (Q_cmp(q->A[(uint64_t)i * q->n + j], q->A[(uint64_t)j * q->n + i])) { ofail(x, TR_MATH_INVALID); break; }
    }
    if (x->rc) { tr_opt_quad_free(q); set_error(error, x->rc); return NULL; }
    set_error(error, TR_MATH_OK);
    return q;
}

tr_opt_quad *tr_opt_quad_create(int32_t n, const tr_lin_qbuf *A, const tr_lin_qbuf *b, const tr_lin_qbuf *c,
                                const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    if (n < 1 || !A || !b || !c) { set_error(error, TR_MATH_INVALID); return NULL; }
    if (n > TR_OPT_MAX_PARAMS) { set_error(error, TR_MATH_UNSUPPORTED); return NULL; }
    int32_t rc = quad_need(n, budget, refusal);
    if (rc) { set_error(error, rc); return NULL; }
    octx x = {budget, refusal, 0};
    tr_opt_quad *q = (tr_opt_quad *)zalloc(sizeof *q);
    if (!q) { set_error(error, TR_MATH_MEMORY); return NULL; }
    q->n = n;
    q->A = Q_arr(&x, (uint64_t)n * (uint64_t)n); q->b = Q_arr(&x, (uint64_t)n);
    tr_bigq *cv[1] = {NULL};
    q->magic = QUAD_MAGIC;
    if (LIVE(&x)) { rc = tr_lin_q_import(A, (uint64_t)n * (uint64_t)n, q->A, budget, refusal); if (rc) ofail(&x, rc); }
    if (LIVE(&x)) { rc = tr_lin_q_import(b, (uint64_t)n, q->b, budget, refusal); if (rc) ofail(&x, rc); }
    if (LIVE(&x)) { rc = tr_lin_q_import(c, 1, cv, budget, refusal); if (rc) ofail(&x, rc); }
    q->c = cv[0];
    if (LIVE(&x) && !q->c) ofail(&x, TR_MATH_INVALID);
    return quad_done(&x, q, error);
}

tr_opt_quad *tr_opt_quad_from_poly(const tr_math_poly *p, int32_t n, const int32_t *vars, const tr_math_budget *budget,
                                   tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    int32_t nt = p ? tr_math_poly_term_count(p) : TR_MATH_INVALID;
    if (nt < 0 || n < 1 || n > TR_MATH_VARIABLES || !vars) { set_error(error, TR_MATH_INVALID); return NULL; }
    int32_t pos[TR_MATH_VARIABLES];
    for (int32_t v = 0; v < TR_MATH_VARIABLES; v++) pos[v] = -1;
    for (int32_t i = 0; i < n; i++) {
        if (vars[i] < 0 || vars[i] >= TR_MATH_VARIABLES || pos[vars[i]] >= 0) { set_error(error, TR_MATH_INVALID); return NULL; }
        pos[vars[i]] = i;
    }
    int32_t rc = quad_need(n, budget, refusal);
    if (rc) { set_error(error, rc); return NULL; }
    octx x = {budget, refusal, 0};
    tr_opt_quad *q = quad_alloc(&x, n);
    for (int32_t t = 0; t < nt && LIVE(&x); t++) {
        tr_math_term term;
        if (tr_math_poly_get_term(p, t, &term)) { ofail(&x, TR_MATH_INVALID); break; }
        int32_t deg = 0, at[2] = {-1, -1}, na = 0;
        for (int32_t v = 0; v < TR_MATH_VARIABLES; v++) {
            int32_t e = term.exponents[v];
            if (!e) continue;
            if (pos[v] < 0) { ofail(&x, TR_MATH_INVALID); break; }          /* a variable that is not a parameter */
            deg += e;
            for (int32_t k = 0; k < e && na < 2; k++) at[na++] = pos[v];
        }
        if (!LIVE(&x)) break;
        if (deg > 2) { ofail(&x, TR_MATH_UNSUPPORTED); break; }
        tr_bigq *cf = Q_q64(&x, term.coefficient.numerator, term.coefficient.denominator);
        if (deg == 0) Q_set(&q->c, Q_add(&x, q->c, cf));
        else if (deg == 1) Q_set(&q->b[at[0]], Q_sub(&x, q->b[at[0]], cf));
        else if (at[0] == at[1]) {
            tr_bigq *two = Q_int(&x, 2), *t2 = Q_mul(&x, two, cf);
            uint64_t ii = (uint64_t)at[0] * n + at[0];
            Q_set(&q->A[ii], Q_add(&x, q->A[ii], t2));
            tr_bigq_free(two); tr_bigq_free(t2);
        } else {
            uint64_t ij = (uint64_t)at[0] * n + at[1], ji = (uint64_t)at[1] * n + at[0];
            Q_set(&q->A[ij], Q_add(&x, q->A[ij], cf));
            Q_set(&q->A[ji], Q_add(&x, q->A[ji], cf));
        }
        tr_bigq_free(cf);
    }
    return quad_done(&x, q, error);
}

tr_opt_quad *tr_opt_quad_from_moments(const tr_lin_moments *s, const tr_lin_fit_spec *fs, const tr_math_budget *budget,
                                      tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    int32_t dim = 0, design[TR_LIN_MAX_COLS];
    tr_bigq **A = NULL, **b = NULL, *yty = NULL;
    int32_t rc = tr_lin_q_normal(s, fs, &dim, &A, NULL, &b, &yty, NULL, NULL, design, budget, refusal);
    if (rc) { set_error(error, rc); return NULL; }
    rc = quad_need(dim, budget, refusal);
    octx x = {budget, refusal, rc};
    tr_opt_quad *q = (tr_opt_quad *)zalloc(sizeof *q);
    if (!q) ofail(&x, TR_MATH_MEMORY);
    if (q) {                                                    /* loss = t^T (2A) t / 2 - (2b)^T t + yty */
        q->magic = QUAD_MAGIC; q->n = dim; q->ndesign = dim;
        memcpy(q->design, design, (size_t)dim * sizeof(int32_t));
        q->A = Q_arr(&x, (uint64_t)dim * (uint64_t)dim); q->b = Q_arr(&x, (uint64_t)dim);
        tr_bigq *two = Q_int(&x, 2);
        for (uint64_t i = 0; q->A && i < (uint64_t)dim * (uint64_t)dim && LIVE(&x); i++) q->A[i] = Q_mul(&x, two, A[i]);
        for (int32_t i = 0; q->b && i < dim && LIVE(&x); i++) q->b[i] = Q_mul(&x, two, b[i]);
        tr_bigq_free(two);
        q->c = yty; yty = NULL;
    }
    tr_lin_q_free(A, (uint64_t)dim * (uint64_t)dim); tr_lin_q_free(b, (uint64_t)dim); tr_bigq_free(yty);
    return quad_done(&x, q, error);
}

int32_t tr_opt_quad_dim(const tr_opt_quad *q) { return quad_valid(q) ? q->n : TR_MATH_INVALID; }
/* the parts of an objective as borrowed values (ACTIVE: the design columns of a moment objective, as owned ints) */
static int32_t quad_part(const tr_opt_quad *q, int32_t part, const tr_bigq *const **v, uint64_t *n) {
    if (!quad_valid(q)) return TR_MATH_INVALID;
    if (part == TR_OPT_PART_A) { *v = (const tr_bigq *const *)q->A; *n = (uint64_t)q->n * (uint64_t)q->n; }
    else if (part == TR_OPT_PART_B) { *v = (const tr_bigq *const *)q->b; *n = (uint64_t)q->n; }
    else if (part == TR_OPT_PART_C) { *v = (const tr_bigq *const *)&q->c; *n = 1; }
    else return TR_MATH_INVALID;
    return TR_MATH_OK;
}
uint64_t tr_opt_quad_count(const tr_opt_quad *q, int32_t part) {
    const tr_bigq *const *v; uint64_t n = 0;
    if (quad_valid(q) && part == TR_OPT_PART_ACTIVE) return (uint64_t)q->ndesign;
    return quad_part(q, part, &v, &n) ? 0 : n;
}
uint64_t tr_opt_quad_limbs(const tr_opt_quad *q, int32_t part) {
    const tr_bigq *const *v; uint64_t n = 0;
    if (quad_valid(q) && part == TR_OPT_PART_ACTIVE) return (uint64_t)q->ndesign;
    return quad_part(q, part, &v, &n) ? 0 : tr_lin_q_limbs(v, n);
}
int32_t tr_opt_quad_export(const tr_opt_quad *q, int32_t part, tr_lin_qbuf *out) {
    if (quad_valid(q) && part == TR_OPT_PART_ACTIVE) {
        octx x = {NULL, NULL, 0};
        tr_bigq **d = Q_arr(&x, (uint64_t)q->ndesign);
        for (int32_t i = 0; d && i < q->ndesign && LIVE(&x); i++) d[i] = Q_int(&x, q->design[i]);
        int32_t rc = x.rc ? x.rc : tr_lin_q_export((const tr_bigq *const *)d, (uint64_t)q->ndesign, out);
        tr_lin_q_free(d, d ? (uint64_t)q->ndesign : 0);
        return rc;
    }
    const tr_bigq *const *v; uint64_t n = 0;
    int32_t rc = quad_part(q, part, &v, &n);
    return rc ? rc : tr_lin_q_export(v, n, out);
}

/* q(t) = t^T A t / 2 - b^T t + c and g(t) = A t - b */
static tr_bigq *quad_at(octx *x, const tr_opt_quad *q, tr_bigq *const *t) {
    const int32_t n = q->n;
    tr_bigq *tat = Q_int(x, 0), *bt = Q_int(x, 0);
    for (int32_t i = 0; i < n && LIVE(x); i++) {
        Q_fma(x, &bt, q->b[i], t[i]);
        if (!tr_bigq_sign(t[i])) continue;
        for (int32_t j = 0; j < n && LIVE(x); j++) {
            tr_bigq *p = Q_mul(x, t[i], q->A[(uint64_t)i * n + j]);
            Q_fma(x, &tat, p, t[j]);
            tr_bigq_free(p);
        }
    }
    tr_bigq *half = Q_q64(x, 1, 2), *h = Q_mul(x, tat, half), *d = Q_sub(x, h, bt), *v = Q_add(x, d, q->c);
    tr_bigq_free(tat); tr_bigq_free(bt); tr_bigq_free(half); tr_bigq_free(h); tr_bigq_free(d);
    return v;
}
static tr_bigq **grad_at(octx *x, const tr_opt_quad *q, tr_bigq *const *t) {
    const int32_t n = q->n;
    tr_bigq **g = Q_arr(x, (uint64_t)n);
    for (int32_t i = 0; g && i < n && LIVE(x); i++) {
        g[i] = Q_neg(x, q->b[i]);
        for (int32_t j = 0; j < n && LIVE(x); j++) Q_fma(x, &g[i], q->A[(uint64_t)i * n + j], t[j]);
    }
    return g;
}
static tr_bigq *quad_form(octx *x, const tr_opt_quad *q, tr_bigq *const *d) {        /* d^T A d */
    const int32_t n = q->n;
    tr_bigq *acc = Q_int(x, 0);
    for (int32_t i = 0; i < n && LIVE(x); i++) {
        if (!tr_bigq_sign(d[i])) continue;
        for (int32_t j = 0; j < n && LIVE(x); j++) {
            tr_bigq *p = Q_mul(x, d[i], q->A[(uint64_t)i * n + j]);
            Q_fma(x, &acc, p, d[j]);
            tr_bigq_free(p);
        }
    }
    return acc;
}
static tr_bigq *dot(octx *x, tr_bigq *const *a, tr_bigq *const *b, int32_t n) {
    tr_bigq *acc = Q_int(x, 0);
    for (int32_t i = 0; i < n && LIVE(x); i++) Q_fma(x, &acc, a[i], b[i]);
    return acc;
}

int32_t tr_opt_quad_value(const tr_opt_quad *q, const tr_lin_qbuf *t, tr_lin_qbuf *out, const tr_math_budget *budget,
                          tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!quad_valid(q) || !t || !out) return TR_MATH_INVALID;
    octx x = {budget, refusal, 0};
    tr_bigq **tv = Q_arr(&x, (uint64_t)q->n);
    if (LIVE(&x)) { int32_t rc = tr_lin_q_import(t, (uint64_t)q->n, tv, budget, refusal); if (rc) ofail(&x, rc); }
    tr_bigq *v = LIVE(&x) ? quad_at(&x, q, tv) : NULL;
    int32_t rc = x.rc ? x.rc : tr_lin_q_export((const tr_bigq *const *)&v, 1, out);
    tr_bigq_free(v);
    tr_lin_q_free(tv, tv ? (uint64_t)q->n : 0);
    return rc;
}

/* ================================================================ results */
struct tr_opt_result_s {
    uint32_t magic;
    tr_opt_info info;
    tr_bigq **q[TR_OPT_PARTS];
    uint64_t nq[TR_OPT_PARTS];
};
static int ores_valid(const tr_opt_result *r) { return r && r->magic == ORES_MAGIC; }
void tr_opt_result_free(tr_opt_result *r) {
    if (!ores_valid(r)) return;
    for (int p = 0; p < TR_OPT_PARTS; p++) tr_lin_q_free(r->q[p], r->nq[p]);
    r->magic = 0;
    free(r);
}
static tr_opt_result *ores_new(octx *x, const tr_opt_quad *q) {
    if (x->rc) return NULL;
    tr_opt_result *r = (tr_opt_result *)zalloc(sizeof *r);
    if (!r) { ofail(x, TR_MATH_MEMORY); return NULL; }
    r->magic = ORES_MAGIC; r->info.n = q->n;
    return r;
}
static void ores_set(tr_opt_result *r, int32_t part, tr_bigq **v, uint64_t n) {
    if (!r) { tr_lin_q_free(v, v ? n : 0); return; }
    tr_lin_q_free(r->q[part], r->nq[part]);
    r->q[part] = v; r->nq[part] = v ? n : 0;
}
int32_t tr_opt_result_get_info(const tr_opt_result *r, tr_opt_info *out) {
    if (!ores_valid(r) || !out) return TR_MATH_INVALID;
    *out = r->info;
    return TR_MATH_OK;
}
uint64_t tr_opt_result_count(const tr_opt_result *r, int32_t part) {
    return ores_valid(r) && part >= 0 && part < TR_OPT_PARTS ? r->nq[part] : 0;
}
uint64_t tr_opt_result_limbs(const tr_opt_result *r, int32_t part) {
    if (!ores_valid(r) || part < 0 || part >= TR_OPT_PARTS) return 0;
    return tr_lin_q_limbs((const tr_bigq *const *)r->q[part], r->nq[part]);
}
int32_t tr_opt_result_export(const tr_opt_result *r, int32_t part, tr_lin_qbuf *out) {
    if (!ores_valid(r) || part < 0 || part >= TR_OPT_PARTS || !r->q[part]) return TR_MATH_INVALID;
    return tr_lin_q_export((const tr_bigq *const *)r->q[part], r->nq[part], out);
}
static tr_opt_result *ores_done(octx *x, tr_opt_result *r, int32_t *error) {
    if (x->rc) { tr_opt_result_free(r); set_error(error, x->rc); return NULL; }
    set_error(error, TR_MATH_OK);
    return r;
}
static tr_bigq **copy_vec(octx *x, const tr_bigq *const *v, uint64_t n) {
    tr_bigq **c = Q_arr(x, n);
    for (uint64_t i = 0; c && i < n && LIVE(x); i++) c[i] = Q_copy(x, v[i]);
    return c;
}

/* an UNBOUNDED result: direction d, anchor t0 (owned), verified: the rate (d^T A d < 0, or d^T A d = 0 and
 * g(t0)^T d < 0) and feasibility along the ray (d = 0 on the bounded coordinates) */
static void unbounded(octx *x, tr_opt_result *r, const tr_opt_quad *q, tr_bigq **d, tr_bigq **t0, int32_t cert,
                      const int32_t *isbounded) {
    const int32_t n = q->n;
    tr_bigq *curv = quad_form(x, q, d), *rate = NULL;
    if (cert == TR_OPT_CERT_CURVATURE) {
        if (LIVE(x) && tr_bigq_sign(curv) >= 0) ofail(x, TR_MATH_INVALID);
        rate = Q_copy(x, curv);
    } else {
        tr_bigq **g = grad_at(x, q, t0);
        rate = g ? dot(x, g, d, n) : NULL;
        if (LIVE(x) && (tr_bigq_sign(curv) != 0 || tr_bigq_sign(rate) >= 0)) ofail(x, TR_MATH_INVALID);
        tr_lin_q_free(g, g ? (uint64_t)n : 0);
    }
    for (int32_t i = 0; isbounded && i < n && LIVE(x); i++) if (isbounded[i] && tr_bigq_sign(d[i])) ofail(x, TR_MATH_INVALID);
    tr_bigq_free(curv);
    if (x->rc) { tr_lin_q_free(d, d ? (uint64_t)n : 0); tr_lin_q_free(t0, t0 ? (uint64_t)n : 0); tr_bigq_free(rate); return; }
    r->info.status = TR_OPT_UNBOUNDED; r->info.certificate = cert;
    ores_set(r, TR_OPT_PART_DIRECTION, d, (uint64_t)n);
    ores_set(r, TR_OPT_PART_ANCHOR, t0, (uint64_t)n);
    tr_bigq **rv = Q_arr(x, 1);
    if (rv) rv[0] = rate; else tr_bigq_free(rate);
    ores_set(r, TR_OPT_PART_RATE, rv, 1);
}
/* an optimum: point (owned), family (owned, nf x n), active (n ints) */
static void optimum(octx *x, tr_opt_result *r, const tr_opt_quad *q, tr_bigq **point, tr_bigq **family, int32_t nfam,
                    const int32_t *active) {
    const int32_t n = q->n;
    tr_bigq *v = quad_at(x, q, point);
    tr_bigq **g = grad_at(x, q, point), **vv = Q_arr(x, 1), **act = Q_arr(x, (uint64_t)n);
    for (int32_t i = 0; act && i < n && LIVE(x); i++) act[i] = Q_int(x, active ? active[i] : 0);
    if (vv) vv[0] = v; else tr_bigq_free(v);
    r->info.family = nfam;
    ores_set(r, TR_OPT_PART_POINT, point, (uint64_t)n);
    ores_set(r, TR_OPT_PART_FAMILY, family, (uint64_t)nfam * (uint64_t)n);
    ores_set(r, TR_OPT_PART_VALUE, vv, 1);
    ores_set(r, TR_OPT_PART_GRADIENT, g, (uint64_t)n);
    ores_set(r, TR_OPT_PART_ACTIVE, act, (uint64_t)n);
}

/* ================================================================ unconstrained */
static tr_opt_result *minimize_free(octx *x, const tr_opt_quad *q) {
    const int32_t n = q->n;
    tr_opt_result *r = ores_new(x, q);
    if (!r) return NULL;
    r->info.method = TR_OPT_METHOD_UNCONSTRAINED; r->info.free = n; r->info.faces = 1;
    int32_t e = 0;
    tr_lin_result *ine = tr_lin_q_inertia(n, (const tr_bigq *const *)q->A, x->b, x->r, &e);
    if (!ine) { ofail(x, e); return r; }
    tr_lin_result_info ii;
    tr_lin_result_get_info(ine, &ii);
    r->info.convex = ii.negative == 0;
    uint64_t cnt = 0;
    if (ii.negative) {                                         /* a direction of negative curvature */
        const tr_bigq *const *neg = tr_lin_result_part(ine, TR_LIN_PART_NEG, &cnt);
        unbounded(x, r, q, copy_vec(x, neg, (uint64_t)n), Q_zeros(x, (uint64_t)n), TR_OPT_CERT_CURVATURE, NULL);
        tr_lin_result_free(ine);
        return r;
    }
    tr_lin_result *sol = tr_lin_q_solve(n, n, (const tr_bigq *const *)q->A, (const tr_bigq *const *)q->b, x->b, x->r, &e);
    if (!sol) { ofail(x, e); tr_lin_result_free(ine); return r; }
    tr_lin_result_info si;
    tr_lin_result_get_info(sol, &si);
    if (si.status == TR_LIN_NONE) {                            /* PSD and b outside range(A): a flat descent */
        const tr_bigq *const *ker = tr_lin_result_part(ine, TR_LIN_PART_KERNEL, &cnt);
        int32_t found = -1;
        tr_bigq *bd = NULL;
        for (uint64_t f = 0; ker && f < cnt / (uint64_t)n && found < 0 && LIVE(x); f++) {
            tr_bigq_free(bd);
            bd = dot(x, (tr_bigq *const *)q->b, (tr_bigq *const *)(ker + f * n), n);
            if (LIVE(x) && tr_bigq_sign(bd)) found = (int32_t)f;
        }
        if (found < 0) ofail(x, TR_MATH_INVALID);              /* impossible for symmetric A */
        tr_bigq **d = LIVE(x) ? copy_vec(x, ker + (uint64_t)found * n, (uint64_t)n) : NULL;
        if (d && LIVE(x) && tr_bigq_sign(bd) < 0)
            for (int32_t i = 0; i < n && LIVE(x); i++) Q_set(&d[i], Q_neg(x, d[i]));
        tr_bigq_free(bd);
        unbounded(x, r, q, d, Q_zeros(x, (uint64_t)n), TR_OPT_CERT_SLOPE, NULL);
    } else {
        const tr_bigq *const *xs = tr_lin_result_part(sol, TR_LIN_PART_X, &cnt);
        uint64_t nfc = 0;
        const tr_bigq *const *fam = tr_lin_result_part(sol, TR_LIN_PART_NULL, &nfc);
        tr_bigq **point = copy_vec(x, xs, (uint64_t)n), **family = nfc ? copy_vec(x, fam, nfc) : NULL;
        optimum(x, r, q, point, family, (int32_t)(nfc / (uint64_t)n), NULL);
        r->info.status = si.status == TR_LIN_UNIQUE ? TR_OPT_UNIQUE : TR_OPT_NONUNIQUE;
        r->info.optima = 1;
        for (int32_t i = 0; r->q[TR_OPT_PART_GRADIENT] && i < n && LIVE(x); i++)       /* KKT: the gradient is 0 */
            if (tr_bigq_sign(r->q[TR_OPT_PART_GRADIENT][i])) ofail(x, TR_MATH_INVALID);
        r->info.kkt = LIVE(x);
    }
    tr_lin_result_free(sol);
    tr_lin_result_free(ine);
    return r;
}

/* ================================================================ boxes */
typedef struct {
    int32_t n, nk, nu;
    int32_t K[TR_OPT_MAX_BOX], U[TR_OPT_MAX_BOX], bounded[TR_OPT_MAX_BOX], degenerate[TR_OPT_MAX_BOX];
    tr_bigq *lo[TR_OPT_MAX_BOX], *hi[TR_OPT_MAX_BOX];
    uint64_t faces;
} boxctx;
static void box_free(boxctx *B) { for (int32_t i = 0; i < TR_OPT_MAX_BOX; i++) { tr_bigq_free(B->lo[i]); tr_bigq_free(B->hi[i]); } }
/* validates and imports a box: TR_MATH_UNSUPPORTED (n > 8, a one-sided bound), TR_MATH_INFEASIBLE (lo > hi) */
static int32_t box_load(const tr_opt_quad *q, const tr_opt_box *box, boxctx *B, const tr_math_budget *budget,
                        tr_math_refusal *refusal) {
    memset(B, 0, sizeof *B);
    if (!quad_valid(q) || !box || box->n != q->n || box->reserved || !box->has_lo || !box->has_hi) return TR_MATH_INVALID;
    if (q->n > TR_OPT_MAX_BOX) return TR_MATH_UNSUPPORTED;
    const int32_t n = q->n;
    B->n = n;
    for (int32_t i = 0; i < n; i++) {
        if (box->has_lo[i] > 1 || box->has_hi[i] > 1) return TR_MATH_INVALID;
        if (box->has_lo[i] != box->has_hi[i]) return TR_MATH_UNSUPPORTED;
    }
    tr_bigq *lo[TR_OPT_MAX_BOX], *hi[TR_OPT_MAX_BOX];
    int32_t rc = tr_lin_q_import(&box->lo, (uint64_t)n, lo, budget, refusal);
    if (rc) return rc;
    rc = tr_lin_q_import(&box->hi, (uint64_t)n, hi, budget, refusal);
    if (rc) { for (int32_t i = 0; i < n; i++) tr_bigq_free(lo[i]); return rc; }
    int32_t infeasible = 0;
    for (int32_t i = 0; i < n; i++) {
        if (box->has_lo[i]) {
            B->bounded[i] = 1; B->K[B->nk++] = i; B->lo[i] = lo[i]; B->hi[i] = hi[i];
            int c = Q_cmp(lo[i], hi[i]);
            if (c > 0) infeasible = 1;
            B->degenerate[i] = c == 0;
        } else { B->U[B->nu++] = i; tr_bigq_free(lo[i]); tr_bigq_free(hi[i]); }
    }
    if (infeasible) { box_free(B); return TR_MATH_INFEASIBLE; }
    B->faces = 1;
    for (int32_t k = 0; k < B->nk; k++) B->faces *= 3;
    return TR_MATH_OK;
}

int32_t tr_opt_face_view_get(const tr_opt_quad *q, const tr_opt_box *box, tr_opt_face_view *v) {
    if (!v) return TR_MATH_INVALID;
    boxctx B;
    int32_t rc = box_load(q, box, &B, NULL, NULL);
    if (rc) return rc;
    memset(v, 0, sizeof *v);
    const int32_t n = q->n;
    v->n = n; v->nk = B.nk; v->faces = B.faces;
    for (int32_t k = 0; k < B.nk; k++) v->kidx[k] = B.K[k];
    tr_math_interval iv;
    for (int32_t i = 0; i < n && !rc; i++) {
        for (int32_t j = 0; j < n && !rc; j++) {
            rc = tr_bigq_enclose(q->A[(uint64_t)i * n + j], &iv);
            v->alo[i * n + j] = iv.lo; v->ahi[i * n + j] = iv.hi;
        }
        if (!rc) rc = tr_bigq_enclose(q->b[i], &iv);
        v->blo[i] = iv.lo; v->bhi[i] = iv.hi;
        if (!rc && B.bounded[i]) {
            rc = tr_bigq_enclose(B.lo[i], &iv); v->llo[i] = iv.lo; v->lhi[i] = iv.hi;
            if (!rc) rc = tr_bigq_enclose(B.hi[i], &iv);
            v->ulo[i] = iv.lo; v->uhi[i] = iv.hi;
            if (B.degenerate[i]) v->degenerate |= (uint32_t)1 << i;
        }
    }
    box_free(&B);
    return rc;
}

int32_t tr_opt_faces(const tr_opt_face_view *v, uint8_t *status) {
    if (!v || !status || v->n < 1 || v->n > TR_OPT_MAX_BOX || v->nk < 0 || v->nk > v->n) return TR_MATH_INVALID;
    for (uint64_t f = 0; f < v->faces; f++) status[f] = (uint8_t)tr_opt_face_filter(v, f);
    return TR_MATH_OK;
}

/* boundedness with free coordinates U: an UNBOUNDED certificate when one exists (returns 1), else 0 */
static int precheck(octx *x, tr_opt_result *r, const tr_opt_quad *q, const boxctx *B) {
    const int32_t n = q->n, nu = B->nu;
    if (!nu) return 0;
    const tr_bigq *AUU[TR_OPT_MAX_BOX * TR_OPT_MAX_BOX];
    for (int32_t a = 0; a < nu; a++) for (int32_t c = 0; c < nu; c++) AUU[a * nu + c] = q->A[(uint64_t)B->U[a] * n + B->U[c]];
    int32_t e = 0;
    tr_lin_result *ine = tr_lin_q_inertia(nu, AUU, x->b, x->r, &e);
    if (!ine) { ofail(x, e); return 0; }
    tr_lin_result_info ii;
    tr_lin_result_get_info(ine, &ii);
    uint64_t cnt = 0;
    int out = 0;
    tr_bigq **t0 = Q_zeros(x, (uint64_t)n);                       /* a feasible anchor: bounded coordinates at lo */
    for (int32_t k = 0; t0 && k < B->nk && LIVE(x); k++) Q_set(&t0[B->K[k]], Q_copy(x, B->lo[B->K[k]]));
    if (ii.negative && LIVE(x)) {
        const tr_bigq *const *neg = tr_lin_result_part(ine, TR_LIN_PART_NEG, &cnt);
        tr_bigq **d = Q_zeros(x, (uint64_t)n);
        for (int32_t a = 0; d && a < nu && LIVE(x); a++) Q_set(&d[B->U[a]], Q_copy(x, neg[a]));
        unbounded(x, r, q, d, t0, TR_OPT_CERT_CURVATURE, B->bounded);
        t0 = NULL; out = 1;
    } else if (LIVE(x)) {
        const tr_bigq *const *ker = tr_lin_result_part(ine, TR_LIN_PART_KERNEL, &cnt);
        for (uint64_t f = 0; ker && f < cnt / (uint64_t)nu && !out && LIVE(x); f++) {
            const tr_bigq *const *v = ker + f * nu;
            /* l(t_K) = sum_i t_i w_i - b_U . v, w_i = (A_KU v)_i, at t_K = lo, else one coordinate moved to hi */
            tr_bigq *w[TR_OPT_MAX_BOX] = {0}, *l = Q_int(x, 0);
            for (int32_t a = 0; a < nu && LIVE(x); a++) {
                tr_bigq *t = Q_mul(x, q->b[B->U[a]], v[a]);
                Q_set(&l, Q_sub(x, l, t));
                tr_bigq_free(t);
            }
            for (int32_t k = 0; k < B->nk && LIVE(x); k++) {
                int32_t i = B->K[k];
                w[i] = Q_int(x, 0);
                for (int32_t a = 0; a < nu && LIVE(x); a++) Q_fma(x, &w[i], q->A[(uint64_t)i * n + B->U[a]], v[a]);
                Q_fma(x, &l, w[i], B->lo[i]);
            }
            int32_t moved = -1;
            if (LIVE(x) && !tr_bigq_sign(l))
                for (int32_t k = 0; k < B->nk && moved < 0 && LIVE(x); k++) {
                    int32_t i = B->K[k];
                    if (tr_bigq_sign(w[i]) && !B->degenerate[i]) moved = i;
                }
            if (moved >= 0) {
                tr_bigq *span = Q_sub(x, B->hi[moved], B->lo[moved]);
                Q_fma(x, &l, w[moved], span);
                tr_bigq_free(span);
            }
            if (LIVE(x) && tr_bigq_sign(l)) {
                if (moved >= 0) Q_set(&t0[moved], Q_copy(x, B->hi[moved]));
                tr_bigq **d = Q_zeros(x, (uint64_t)n);
                for (int32_t a = 0; d && a < nu && LIVE(x); a++) Q_set(&d[B->U[a]], tr_bigq_sign(l) > 0 ? Q_neg(x, v[a]) : Q_copy(x, v[a]));
                unbounded(x, r, q, d, t0, TR_OPT_CERT_SLOPE, B->bounded);
                t0 = NULL; out = 1;
            }
            for (int32_t i = 0; i < TR_OPT_MAX_BOX; i++) tr_bigq_free(w[i]);
            tr_bigq_free(l);
        }
    }
    tr_lin_q_free(t0, t0 ? (uint64_t)n : 0);
    tr_lin_result_free(ine);
    return out;
}

typedef struct { tr_bigq *value; tr_bigq **point, **family; int32_t nfam; int32_t active[TR_OPT_MAX_BOX]; } candidate;
static void cand_free(candidate *c, int32_t n) {
    tr_bigq_free(c->value); tr_lin_q_free(c->point, c->point ? (uint64_t)n : 0);
    tr_lin_q_free(c->family, c->family ? (uint64_t)c->nfam * (uint64_t)n : 0);
    memset(c, 0, sizeof *c);
}

/* the exact candidate of one face (0: none; 1: *c filled) */
static int face_exact(octx *x, const tr_opt_quad *q, const boxctx *B, uint64_t face, candidate *c) {
    const int32_t n = q->n;
    int32_t state[TR_OPT_MAX_BOX], F[TR_OPT_MAX_BOX], nf = 0;
    for (int32_t i = 0; i < n; i++) state[i] = 2;
    uint64_t f = face;
    for (int32_t k = 0; k < B->nk; k++) { state[B->K[k]] = (int32_t)(f % 3); f /= 3; }
    for (int32_t i = 0; i < n; i++) if (state[i] == 2) F[nf++] = i;
    tr_bigq **t = Q_arr(x, (uint64_t)n);
    for (int32_t i = 0; t && i < n && LIVE(x); i++) t[i] = state[i] == 2 ? Q_int(x, 0) : Q_copy(x, state[i] ? B->hi[i] : B->lo[i]);
    tr_bigq **family = NULL;
    int32_t nfam = 0, ok = LIVE(x) != 0;
    if (nf && ok) {
        const tr_bigq *AFF[TR_OPT_MAX_BOX * TR_OPT_MAX_BOX];
        tr_bigq **rF = Q_arr(x, (uint64_t)nf);
        for (int32_t a = 0; a < nf && LIVE(x); a++) {
            for (int32_t b = 0; b < nf; b++) AFF[a * nf + b] = q->A[(uint64_t)F[a] * n + F[b]];
            rF[a] = Q_copy(x, q->b[F[a]]);
            for (int32_t i = 0; i < n && LIVE(x); i++) {
                if (state[i] == 2) continue;
                tr_bigq *p = Q_mul(x, q->A[(uint64_t)F[a] * n + i], t[i]);
                Q_set(&rF[a], Q_sub(x, rF[a], p));
                tr_bigq_free(p);
            }
        }
        int32_t e = 0;
        tr_lin_result *sol = LIVE(x) ? tr_lin_q_solve(nf, nf, AFF, (const tr_bigq *const *)rF, x->b, x->r, &e) : NULL;
        if (!sol && LIVE(x)) ofail(x, e);
        tr_lin_result_info si;
        if (sol) tr_lin_result_get_info(sol, &si);
        ok = sol && LIVE(x) && si.status != TR_LIN_NONE;
        uint64_t cnt = 0, nfc = 0;
        const tr_bigq *const *xs = ok ? tr_lin_result_part(sol, TR_LIN_PART_X, &cnt) : NULL;
        const tr_bigq *const *ker = ok ? tr_lin_result_part(sol, TR_LIN_PART_NULL, &nfc) : NULL;
        nfam = (int32_t)(nfc / (uint64_t)nf);
        for (int32_t v = 0; ok && v < nfam; v++)                    /* the kernel must vanish on bounded coordinates */
            for (int32_t a = 0; a < nf; a++) if (B->bounded[F[a]] && tr_bigq_sign(ker[(uint64_t)v * nf + a])) ok = 0;
        for (int32_t a = 0; ok && a < nf && LIVE(x); a++) {
            Q_set(&t[F[a]], Q_copy(x, xs[a]));
            int32_t i = F[a];
            if (B->bounded[i] && (Q_cmp(t[i], B->lo[i]) < 0 || Q_cmp(t[i], B->hi[i]) > 0)) ok = 0;
        }
        if (ok && nfam) {
            family = Q_zeros(x, (uint64_t)nfam * (uint64_t)n);
            for (int32_t v = 0; family && v < nfam && LIVE(x); v++)
                for (int32_t a = 0; a < nf && LIVE(x); a++) Q_set(&family[(uint64_t)v * n + F[a]], Q_copy(x, ker[(uint64_t)v * nf + a]));
        }
        tr_lin_result_free(sol);
        tr_lin_q_free(rF, rF ? (uint64_t)nf : 0);
    }
    if (ok && LIVE(x)) {                                         /* KKT signs at the fixed coordinates */
        tr_bigq **g = grad_at(x, q, t);
        for (int32_t i = 0; g && i < n && ok && LIVE(x); i++) {
            if (state[i] == 2) { if (tr_bigq_sign(g[i])) ofail(x, TR_MATH_INVALID); continue; }
            if (B->degenerate[i]) continue;
            if ((state[i] == 0 && tr_bigq_sign(g[i]) < 0) || (state[i] == 1 && tr_bigq_sign(g[i]) > 0)) ok = 0;
        }
        tr_lin_q_free(g, g ? (uint64_t)n : 0);
    }
    if (!ok || !LIVE(x)) {
        tr_lin_q_free(t, t ? (uint64_t)n : 0);
        tr_lin_q_free(family, family ? (uint64_t)nfam * (uint64_t)n : 0);
        return 0;
    }
    c->value = quad_at(x, q, t); c->point = t; c->family = family; c->nfam = nfam;
    for (int32_t i = 0; i < n; i++) c->active[i] = state[i] == 2 ? 0 : (state[i] == 0 || B->degenerate[i]) ? -1 : 1;
    return 1;
}

static int same_point(tr_bigq *const *a, tr_bigq *const *b, int32_t n) {
    for (int32_t i = 0; i < n; i++) if (Q_cmp(a[i], b[i])) return 0;
    return 1;
}

static tr_opt_result *minimize_box(octx *x, const tr_opt_quad *q, const boxctx *B, const uint8_t *status) {
    const int32_t n = q->n;
    tr_opt_result *r = ores_new(x, q);
    if (!r) return NULL;
    r->info.method = TR_OPT_METHOD_FACES; r->info.bounded = B->nk; r->info.free = B->nu; r->info.faces = B->faces;
    int32_t e = 0;
    tr_lin_result *ine = tr_lin_q_inertia(n, (const tr_bigq *const *)q->A, x->b, x->r, &e);
    if (!ine) { ofail(x, e); return r; }
    tr_lin_result_info ii;
    tr_lin_result_get_info(ine, &ii);
    r->info.convex = ii.negative == 0;
    tr_lin_result_free(ine);
    if (precheck(x, r, q, B) || !LIVE(x)) return r;
    candidate best;
    memset(&best, 0, sizeof best);
    tr_bigq **optima = NULL;
    uint64_t nopt = 0;
    for (uint64_t face = 0; face < B->faces && LIVE(x); face++) {
        r->info.filtered[status[face]]++;
        if (status[face] != TR_OPT_FACE_CANDIDATE && status[face] != TR_OPT_FACE_UNDECIDED) continue;
        r->info.exact++;
        candidate c;
        memset(&c, 0, sizeof c);
        if (!face_exact(x, q, B, face, &c)) continue;
        int cmp = best.value ? Q_cmp(c.value, best.value) : -1;
        if (cmp < 0) {
            cand_free(&best, n);
            tr_lin_q_free(optima, nopt * (uint64_t)n);
            best = c;
            optima = copy_vec(x, (const tr_bigq *const *)c.point, (uint64_t)n);
            nopt = optima ? 1 : 0;
            continue;
        }
        if (cmp == 0) {
            int seen = 0;
            for (uint64_t o = 0; o < nopt && !seen; o++) seen = same_point(optima + o * n, c.point, n);
            if (!seen) {
                tr_bigq **grown = (tr_bigq **)realloc(optima, (size_t)((nopt + 1) * (uint64_t)n) * sizeof *grown);
                if (!grown) ofail(x, TR_MATH_MEMORY);
                else {
                    optima = grown;
                    for (int32_t i = 0; i < n; i++) optima[nopt * n + i] = Q_copy(x, c.point[i]);
                    nopt++;
                }
            }
        }
        cand_free(&c, n);
    }
    if (LIVE(x) && !best.value) ofail(x, TR_MATH_INVALID);      /* a bounded problem has an optimal face */
    if (!LIVE(x)) { cand_free(&best, n); tr_lin_q_free(optima, optima ? nopt * (uint64_t)n : 0); return r; }
    optimum(x, r, q, best.point, best.family, best.nfam, best.active);
    best.point = NULL; best.family = NULL;
    r->info.status = nopt > 1 || best.nfam ? TR_OPT_NONUNIQUE : TR_OPT_UNIQUE;
    r->info.optima = (int32_t)nopt; r->info.kkt = 1;
    ores_set(r, TR_OPT_PART_OPTIMA, optima, nopt * (uint64_t)n);
    cand_free(&best, n);
    return r;
}

tr_opt_result *tr_opt_minimize(const tr_opt_quad *q, const tr_opt_box *box, const uint8_t *status, uint64_t nstatus,
                               const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    if (!quad_valid(q)) { set_error(error, TR_MATH_INVALID); return NULL; }
    tr_math_budget dflt;
    if (!budget) { tr_budget_default(&dflt); budget = &dflt; }
    octx x = {budget, refusal, 0};
    tr_math_need need;
    memset(&need, 0, sizeof need);
    need.need[TR_BUDGET_VARIABLES] = (uint64_t)q->n;
    need.need[TR_BUDGET_DEGREE] = 2;
    need.need[TR_BUDGET_TENSOR] = (uint64_t)q->n * (uint64_t)(q->n + 1);   /* the augmented system [A | b] */
    need.need[TR_BUDGET_OUTPUT] = (uint64_t)q->n;
    if (!box) {
        int32_t rc = tr_budget_check(budget, &need, refusal);
        if (rc) { set_error(error, rc); return NULL; }
        return ores_done(&x, minimize_free(&x, q), error);
    }
    boxctx B;
    int32_t rc = box_load(q, box, &B, budget, refusal);
    if (rc) { set_error(error, rc); return NULL; }
    need.need[TR_BUDGET_ITERATIONS] = B.faces;
    need.need[TR_BUDGET_OUTPUT] = tr_budget_mul((uint64_t)q->n, B.faces + 1);
    need.need[TR_BUDGET_BYTES] = tr_budget_add(B.faces, sizeof(tr_opt_face_view));
    rc = tr_budget_check(budget, &need, refusal);
    if (!rc && status && nstatus != B.faces) rc = TR_MATH_INVALID;
    for (uint64_t f = 0; !rc && status && f < nstatus; f++) if (status[f] >= TR_OPT_FACE_STATES) rc = TR_MATH_INVALID;
    if (rc) { box_free(&B); set_error(error, rc); return NULL; }
    uint8_t *own = NULL;
    if (!status) {
        tr_opt_face_view v;
        own = (uint8_t *)zalloc(B.faces);
        rc = own ? tr_opt_face_view_get(q, box, &v) : TR_MATH_MEMORY;
        if (!rc) rc = tr_opt_faces(&v, own);
        if (rc) { free(own); box_free(&B); set_error(error, rc); return NULL; }
        status = own;
    }
    tr_opt_result *r = minimize_box(&x, q, &B, status);
    free(own);
    box_free(&B);
    return ores_done(&x, r, error);
}
