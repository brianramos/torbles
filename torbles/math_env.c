/* ENV-001: univariate affine / quadratic group envelopes; see math_env.h.
 * SPDX-License-Identifier: Apache-2.0 */
#include "math_env.h"
#include "math_domain.h"
#include "math_handles.h"
#include <stdlib.h>
#include <string.h>

TR_MATH_ABI(tr_env_info, 1);             /* the ctypes boundary of ENV-001 (math_handles.h) */
TR_MATH_ABI(tr_env_region, 1);
TR_MATH_ABI(tr_env_point_info, 1);
TR_MATH_ABI(tr_env_class_info, 1);
TR_MATH_ABI(tr_env_answer, 1);
TR_MATH_ABI(tr_env_view, 1);

_Static_assert(sizeof(tr_env_info) == 96, "tr_env_info ABI");
_Static_assert(sizeof(tr_env_region) == 56, "tr_env_region ABI");
_Static_assert(sizeof(tr_env_point_info) == 24, "tr_env_point_info ABI");
_Static_assert(sizeof(tr_env_class_info) == 24, "tr_env_class_info ABI");
_Static_assert(sizeof(tr_env_answer) == 32, "tr_env_answer ABI");
_Static_assert(sizeof(tr_env_view) == 184, "tr_env_view ABI");

#define ENV_MAGIC UINT32_C(0x31564e45)
#define LOOKUP_MAGIC UINT32_C(0x4b4f4f4c)
#define NONE TR_ENV_NONE
enum { P_NONE = 0, P_RAT = 1, P_ALG = 2, P_NEG = 3, P_POS = 4 };

static void set_error(int32_t *error, int32_t code) { if (error) *error = code; }
static void *zalloc(uint64_t bytes) {
    if (bytes > (uint64_t)SIZE_MAX) return NULL;
    return calloc(1, bytes ? (size_t)bytes : 1);
}
static const tr_math_budget *budget_or(const tr_math_budget *b, tr_math_budget *dflt) {
    if (b) return b;
    tr_budget_default(dflt);
    return dflt;
}
static void need_zero(tr_math_need *n) { for (int i = 0; i < TR_BUDGET_RESOURCES; i++) n->need[i] = 0; }

/* ================================================================== exact arithmetic with a sticky error */
typedef struct {
    int32_t rc;
    const tr_math_budget *budget;
    tr_math_refusal *refusal;
    uint64_t iterations, limit;
} ctx;
#define TMAX 40
typedef struct { tr_bigint *p[TMAX]; int n; } tmp;     /* temporaries freed together */
static tr_bigint *keep(ctx *c, tmp *t, tr_bigint *x, int32_t e) {
    if (!x) { if (!c->rc) c->rc = e ? e : TR_MATH_MEMORY; return NULL; }
    if (t) {
        if (t->n >= TMAX) { tr_bigint_free(x); if (!c->rc) c->rc = TR_MATH_INVALID; return NULL; }
        t->p[t->n++] = x;
    }
    return x;
}
static void tmp_free(tmp *t) { for (int i = 0; i < t->n; i++) tr_bigint_free(t->p[i]); t->n = 0; }
#define OP2(name, fn) static tr_bigint *name(ctx *c, tmp *t, const tr_bigint *a, const tr_bigint *b) { \
        int32_t e = 0;                                                                                \
        return c->rc ? NULL : keep(c, t, fn(a, b, c->budget, c->refusal, &e), e); }
#define OP1(name, fn) static tr_bigint *name(ctx *c, tmp *t, const tr_bigint *a) { \
        int32_t e = 0;                                                             \
        return c->rc ? NULL : keep(c, t, fn(a, c->budget, c->refusal, &e), e); }
OP2(zmul, tr_bigint_mul)
OP2(zadd, tr_bigint_add)
OP2(zsub, tr_bigint_sub)
OP2(zgcd, tr_bigint_gcd)
OP2(zdivx, tr_bigint_divexact)
OP1(zneg, tr_bigint_neg)
OP1(zcopy, tr_bigint_copy)
OP1(zisqrt, tr_bigint_isqrt)
static tr_bigint *zi64(ctx *c, tmp *t, int64_t v) {
    if (c->rc) return NULL;
    int32_t e = 0;
    return keep(c, t, tr_bigint_from_i64(v, c->budget, c->refusal, &e), e);
}
static tr_bigint *zshl(ctx *c, tmp *t, const tr_bigint *a, uint64_t k) {
    if (c->rc) return NULL;
    int32_t e = 0;
    return keep(c, t, tr_bigint_shl(a, k, c->budget, c->refusal, &e), e);
}
static int zs(ctx *c, const tr_bigint *a) { return c->rc || !a ? 0 : tr_bigint_sign(a); }
static int zcmp(ctx *c, const tr_bigint *a, const tr_bigint *b) { return c->rc || !a || !b ? 0 : tr_bigint_cmp(a, b); }

/* sign(X + Y sqrt(D)) for integers X, Y and a non-square D > 0 (X^2 = Y^2 D only when both are 0) */
static int sign1(ctx *c, const tr_bigint *X, const tr_bigint *Y, const tr_bigint *D) {
    if (c->rc) return 0;
    int sx = zs(c, X), sy = zs(c, Y);
    if (!sy) return sx;
    if (!sx || sx == sy) return sy;
    tmp t = {{0}, 0};
    tr_bigint *x2 = zmul(c, &t, X, X), *y2 = zmul(c, &t, Y, Y), *y2d = zmul(c, &t, y2, D);
    int r = zcmp(c, x2, y2d);
    tmp_free(&t);
    return r > 0 ? sx : sy;
}
/* sign(R + U sqrt(D1) + W sqrt(D2)) for integers and non-square D1, D2 > 0 */
static int sign2(ctx *c, const tr_bigint *R, const tr_bigint *U, const tr_bigint *D1, const tr_bigint *W,
                 const tr_bigint *D2) {
    int s1 = sign1(c, R, U, D1), s2 = zs(c, W);
    if (c->rc) return 0;
    if (!s1) return s2;
    if (!s2 || s1 == s2) return s1;
    tmp t = {{0}, 0};                     /* |R + U sqrt(D1)| against |W sqrt(D2)|: (R^2 + U^2 D1 - W^2 D2) + 2 R U sqrt(D1) */
    tr_bigint *rr = zmul(c, &t, R, R), *uu = zmul(c, &t, U, U), *uud = zmul(c, &t, uu, D1);
    tr_bigint *ww = zmul(c, &t, W, W), *wwd = zmul(c, &t, ww, D2);
    tr_bigint *x = zsub(c, &t, zadd(c, &t, rr, uud), wwd), *ru = zmul(c, &t, R, U), *y = zadd(c, &t, ru, ru);
    int s3 = sign1(c, x, y, D1);
    tmp_free(&t);
    return s3 > 0 ? s1 : s3 < 0 ? s2 : 0;
}

/* ================================================================== points of the parameter line */
typedef struct {
    int32_t kind, sigma;           /* P_*; ALG: -1 the smaller, +1 the larger root */
    tr_bigint *n, *d;              /* RAT: n / d, d > 0 */
    tr_bigint *q[3], *disc;        /* ALG: primitive q0 + q1 t + q2 t^2, q2 > 0, disc non-square */
} pt;
static void pt_clear(pt *p) {
    tr_bigint_free(p->n); tr_bigint_free(p->d);
    for (int k = 0; k < 3; k++) tr_bigint_free(p->q[k]);
    tr_bigint_free(p->disc);
    memset(p, 0, sizeof *p);
}
static void pt_move(pt *dst, pt *src) { pt_clear(dst); *dst = *src; memset(src, 0, sizeof *src); }
static int32_t pt_copy(ctx *c, const pt *s, pt *o) {
    memset(o, 0, sizeof *o);
    o->kind = s->kind; o->sigma = s->sigma;
    if (s->kind == P_RAT) { o->n = zcopy(c, NULL, s->n); o->d = zcopy(c, NULL, s->d); }
    if (s->kind == P_ALG) {
        for (int k = 0; k < 3; k++) o->q[k] = zcopy(c, NULL, s->q[k]);
        o->disc = zcopy(c, NULL, s->disc);
    }
    if (c->rc) pt_clear(o);
    return c->rc;
}
/* sign(alpha - n / d) for an algebraic alpha = (-q1 + s sqrt(disc)) / (2 q2) and d > 0:
 * d (2 q2) (alpha - n / d) = (-q1 d - 2 q2 n) + s d sqrt(disc) */
static int alg_minus_rat(ctx *c, const pt *a, const tr_bigint *n, const tr_bigint *d) {
    tmp t = {{0}, 0};
    tr_bigint *q1d = zmul(c, &t, a->q[1], d), *q2n = zmul(c, &t, a->q[2], n), *s = zadd(c, &t, q1d, zadd(c, &t, q2n, q2n));
    tr_bigint *X = zneg(c, &t, s), *Y = a->sigma < 0 ? zneg(c, &t, d) : zcopy(c, &t, d);
    int r = sign1(c, X, Y, a->disc);
    tmp_free(&t);
    return r;
}
static int same_q(ctx *c, const pt *a, const pt *b) {
    return !zcmp(c, a->q[0], b->q[0]) && !zcmp(c, a->q[1], b->q[1]) && !zcmp(c, a->q[2], b->q[2]);
}
/* -1, 0, 1: a < b, a == b, a > b (exact) */
static int pt_cmp(ctx *c, const pt *a, const pt *b) {
    if (c->rc) return 0;
    if (a->kind == P_NEG) return b->kind == P_NEG ? 0 : -1;
    if (a->kind == P_POS) return b->kind == P_POS ? 0 : 1;
    if (b->kind == P_NEG) return 1;
    if (b->kind == P_POS) return -1;
    if (a->kind == P_RAT && b->kind == P_RAT) {
        tmp t = {{0}, 0};
        int r = zs(c, zsub(c, &t, zmul(c, &t, a->n, b->d), zmul(c, &t, b->n, a->d)));
        tmp_free(&t);
        return r;
    }
    if (a->kind == P_ALG && b->kind == P_RAT) return alg_minus_rat(c, a, b->n, b->d);
    if (a->kind == P_RAT && b->kind == P_ALG) return -alg_minus_rat(c, b, a->n, a->d);
    if (same_q(c, a, b)) return a->sigma < b->sigma ? -1 : a->sigma > b->sigma;
    /* (alpha_a - alpha_b) 2 q2a q2b = (q2a q1b - q2b q1a) + sa q2b sqrt(Da) - sb q2a sqrt(Db): distinct irreducible
     * polynomials share no root, so this is never 0 */
    tmp t = {{0}, 0};
    tr_bigint *R = zsub(c, &t, zmul(c, &t, a->q[2], b->q[1]), zmul(c, &t, b->q[2], a->q[1]));
    tr_bigint *U = a->sigma < 0 ? zneg(c, &t, b->q[2]) : zcopy(c, &t, b->q[2]);
    tr_bigint *Wv = b->sigma < 0 ? zcopy(c, &t, a->q[2]) : zneg(c, &t, a->q[2]);
    int r = sign2(c, R, U, a->disc, Wv, b->disc);
    tmp_free(&t);
    return r;
}
/* the sign of the integer polynomial P = (p0, p1, p2) at x (+-inf: as t -> +-inf) */
static int sign_at(ctx *c, tr_bigint *const P[3], const pt *x) {
    if (c->rc) return 0;
    if (x->kind == P_NEG || x->kind == P_POS) {
        for (int k = 2; k >= 0; k--) {
            int s = zs(c, P[k]);
            if (s) return (x->kind == P_NEG && (k & 1)) ? -s : s;
        }
        return 0;
    }
    tmp t = {{0}, 0};
    int r;
    if (x->kind == P_RAT) {                                    /* p0 d^2 + p1 n d + p2 n^2 */
        tr_bigint *dd = zmul(c, &t, x->d, x->d), *nd = zmul(c, &t, x->n, x->d), *nn = zmul(c, &t, x->n, x->n);
        tr_bigint *s = zadd(c, &t, zadd(c, &t, zmul(c, &t, P[0], dd), zmul(c, &t, P[1], nd)), zmul(c, &t, P[2], nn));
        r = zs(c, s);
    } else {                                                   /* q2 P - p2 Q = r1 t + r0, Q(alpha) = 0 */
        tr_bigint *r1 = zsub(c, &t, zmul(c, &t, x->q[2], P[1]), zmul(c, &t, P[2], x->q[1]));
        tr_bigint *r0 = zsub(c, &t, zmul(c, &t, x->q[2], P[0]), zmul(c, &t, P[2], x->q[0]));
        int s1 = zs(c, r1);
        if (!s1) r = zs(c, r0);
        else {                                                 /* sign(r1) sign(alpha - (-r0 / r1)) */
            tr_bigint *n = s1 > 0 ? zneg(c, &t, r0) : zcopy(c, &t, r0), *d = s1 > 0 ? zcopy(c, &t, r1) : zneg(c, &t, r1);
            r = c->rc ? 0 : s1 * alg_minus_rat(c, x, n, d);
        }
    }
    tmp_free(&t);
    return c->rc ? 0 : r;
}
/* the sign of D just right of x, where D(x) = 0: the sign of D'(x), else of D's leading coefficient */
static int sign_right(ctx *c, tr_bigint *const D[3], const pt *x) {
    if (x->kind == P_NEG) return sign_at(c, D, x);
    tmp t = {{0}, 0};
    tr_bigint *zero = zi64(c, &t, 0), *two = zadd(c, &t, D[2], D[2]);
    tr_bigint *dp[3] = {D[1], two, zero};
    int s = c->rc ? 0 : sign_at(c, dp, x);
    if (!s) s = zs(c, D[2]);
    tmp_free(&t);
    return s;
}

/* a perfect square? (quadratic residue filters, then the integer square root) */
static int qres(uint64_t r, uint64_t m) {
    for (uint64_t x = 0; x < m; x++) if (x * x % m == r % m) return 1;
    return 0;
}
static int is_square(ctx *c, const tr_bigint *a, tr_bigint **root) {
    *root = NULL;
    uint64_t r = 0;
    if (c->rc || tr_bigint_mod_u64(a, UINT64_C(2882880), &r)) return 0;           /* 64 * 63 * 65 * 11 */
    if (!qres(r % 64, 64) || !qres(r % 63, 63) || !qres(r % 65, 65) || !qres(r % 11, 11)) return 0;
    tmp t = {{0}, 0};
    tr_bigint *s = zisqrt(c, NULL, a), *ss = zmul(c, &t, s, s);
    int sq = !c->rc && !zcmp(c, ss, a);
    tmp_free(&t);
    if (sq) *root = s; else tr_bigint_free(s);
    return sq;
}
/* a rational point n / d with d made positive (owns copies) */
static void pt_rat(ctx *c, pt *p, const tr_bigint *n, const tr_bigint *d) {
    memset(p, 0, sizeof *p);
    p->kind = P_RAT;
    if (zs(c, d) < 0) { p->n = zneg(c, NULL, n); p->d = zneg(c, NULL, d); }
    else { p->n = zcopy(c, NULL, n); p->d = zcopy(c, NULL, d); }
    if (c->rc) pt_clear(p);
}
/* the distinct real roots of a nonzero integer polynomial D of degree <= 2, increasing: 0, 1 or 2 points */
static int roots(ctx *c, tr_bigint *const D[3], pt out[2]) {
    if (c->rc) return 0;
    int s2 = zs(c, D[2]), s1 = zs(c, D[1]);
    if (!s2) {
        if (!s1) return 0;
        tmp t = {{0}, 0};
        pt_rat(c, &out[0], zneg(c, &t, D[0]), D[1]);
        tmp_free(&t);
        return c->rc ? 0 : 1;
    }
    tmp t = {{0}, 0};
    tr_bigint *disc = zsub(c, &t, zmul(c, &t, D[1], D[1]), zshl(c, &t, zmul(c, &t, D[2], D[0]), 2));
    int sd = zs(c, disc), cnt = 0;
    tr_bigint *two_a = zadd(c, &t, D[2], D[2]), *mb = zneg(c, &t, D[1]);
    if (c->rc || sd < 0) { tmp_free(&t); return 0; }
    tr_bigint *s = NULL;
    if (sd == 0) { pt_rat(c, &out[0], mb, two_a); cnt = 1; }
    else if (is_square(c, disc, &s)) {
        keep(c, &t, s, 0);
        pt lo, hi;
        pt_rat(c, &lo, zsub(c, &t, mb, s), two_a);
        pt_rat(c, &hi, zadd(c, &t, mb, s), two_a);
        if (s2 > 0) { out[0] = lo; out[1] = hi; } else { out[0] = hi; out[1] = lo; }
        cnt = 2;
    } else if (!c->rc) {                                       /* irrational: the primitive part, q2 > 0 */
        tr_bigint *g = zgcd(c, &t, zgcd(c, &t, D[0], D[1]), D[2]);
        for (int r = 0; r < 2; r++) {
            out[r].kind = P_ALG; out[r].sigma = r ? 1 : -1;
            for (int k = 0; k < 3; k++) {
                tr_bigint *q = zdivx(c, &t, D[k], g);
                out[r].q[k] = s2 > 0 ? zcopy(c, NULL, q) : zneg(c, NULL, q);
            }
            tr_bigint *qd = zsub(c, &t, zmul(c, &t, out[r].q[1], out[r].q[1]),
                                 zshl(c, &t, zmul(c, &t, out[r].q[2], out[r].q[0]), 2));
            out[r].disc = zcopy(c, NULL, qd);
        }
        cnt = 2;
    }
    tmp_free(&t);
    if (c->rc) { for (int r = 0; r < 2; r++) pt_clear(&out[r]); return 0; }
    return cnt;
}

/* ================================================================== classes */
typedef struct {
    tr_bigint *P[3], *d;           /* internal integer numerators (negated for an upper envelope), d > 0 */
    tr_bigq *orig[3];              /* the given coefficients */
    uint64_t first, count, winner;  /* members: groups[first .. first + count) */
    int32_t degree;
} cls_t;
static void cls_clear(cls_t *k) {
    for (int i = 0; i < 3; i++) { tr_bigint_free(k->P[i]); tr_bigq_free(k->orig[i]); }
    tr_bigint_free(k->d);
    memset(k, 0, sizeof *k);
}
/* D = d_b P_a - d_a P_b: the sign of f_a - f_b (times d_a d_b > 0); one iteration */
static void diff(ctx *c, tmp *t, const cls_t *a, const cls_t *b, tr_bigint *D[3]) {
    for (int k = 0; k < 3; k++) D[k] = zsub(c, t, zmul(c, t, b->d, a->P[k]), zmul(c, t, a->d, b->P[k]));
    c->iterations++;
    if (!c->rc && c->iterations > c->limit) {                  /* the sweep would need more: refused at limit + 1 */
        c->rc = TR_MATH_BUDGET;
        if (c->refusal) { c->refusal->resource = TR_BUDGET_ITERATIONS; c->refusal->refused = 1;
                          c->refusal->need = c->iterations; c->refusal->limit = c->limit; }
    }
}

/* ================================================================== the envelope object */
typedef struct {
    int32_t kind, boundary;
    uint64_t point, right, cls, first, ncls;
} reg_t;
struct tr_env_s {
    uint32_t magic;
    int32_t kind, upper, tie, degree, lo_bounded, hi_bounded;
    uint64_t ngroups, present, nclasses, nregions, npoints, npieces, nalg, iterations, bytes;
    cls_t *cls;
    uint64_t *members;             /* the classes' member groups */
    pt *pts;                       /* the points (internal exact form) */
    tr_bigq **pval;                /* RAT points: the value (canonical) */
    tr_bigq **plo, **phi;          /* ALG points: INV-001's isolating interval */
    int32_t *psign, *pboundary;    /* ALG: the sign of q at plo; boundary flags */
    uint64_t *pregion;
    reg_t *regs;
    uint64_t *pcls;                /* the classes of the points */
    uint64_t *rwinner, *rmembers;
    tr_bigq *lo, *hi;
};
static int env_valid(const tr_env *e) { return e && e->magic == ENV_MAGIC; }
static void env_release(tr_env *e) {
    if (!e) return;
    for (uint64_t i = 0; e->cls && i < e->nclasses; i++) cls_clear(&e->cls[i]);
    for (uint64_t i = 0; e->pts && i < e->npoints; i++) {
        pt_clear(&e->pts[i]);
        if (e->pval) tr_bigq_free(e->pval[i]);
        if (e->plo) tr_bigq_free(e->plo[i]);
        if (e->phi) tr_bigq_free(e->phi[i]);
    }
    free(e->cls); free(e->members); free(e->pts); free(e->pval); free(e->plo); free(e->phi); free(e->psign);
    free(e->pboundary); free(e->pregion); free(e->regs); free(e->pcls); free(e->rwinner); free(e->rmembers);
    tr_bigq_free(e->lo); tr_bigq_free(e->hi);
    e->magic = 0;
    free(e);
}
void tr_env_free(tr_env *e) { if (env_valid(e)) env_release(e); }

int32_t tr_env_need(uint64_t ngroups, uint64_t present, int32_t degree, tr_math_need *need) {
    if (!need || degree < 0) return TR_MATH_INVALID;
    need_zero(need);
    need->need[TR_BUDGET_VARIABLES] = 1;
    need->need[TR_BUDGET_DEGREE] = (uint64_t)degree;
    need->need[TR_BUDGET_SUPPORT] = present;
    need->need[TR_BUDGET_STATES] = ngroups;
    /* the class table, the member and sorting arrays, the per-class scratch of the sweep */
    need->need[TR_BUDGET_BYTES] = tr_budget_add(sizeof(tr_env), tr_budget_mul(present, sizeof(cls_t) + 8 * 4));
    return TR_MATH_OK;
}

/* ------------------------------------------------------------------ the sweep's growing output */
typedef struct {
    pt *pts; uint64_t np, cap_p;
    reg_t *regs; uint64_t nr, cap_r;
    uint64_t *pcls; uint64_t npc, cap_pc;
    int32_t *pbound;
    uint64_t count, points, pieces, alg, limit, fixed;
    int storing;
} out_t;
static void out_clear(out_t *o) {
    for (uint64_t i = 0; i < o->np; i++) pt_clear(&o->pts[i]);
    free(o->pts); free(o->regs); free(o->pcls); free(o->pbound);
    memset(o, 0, sizeof *o);
}
static uint64_t out_bytes(const out_t *o, uint64_t cap_p, uint64_t cap_r, uint64_t cap_pc) {
    uint64_t b = tr_budget_add(o->fixed, tr_budget_mul(cap_p, sizeof(pt) + 4));
    b = tr_budget_add(b, tr_budget_mul(cap_r, sizeof(reg_t)));
    return tr_budget_add(b, tr_budget_mul(cap_pc, 8));
}
/* room for one more region, point and k point classes (grown by doubling, budgeted) */
static int32_t out_room(ctx *c, out_t *o, uint64_t k) {
    uint64_t cp = o->cap_p, cr = o->cap_r, cc = o->cap_pc;
    while (o->np + 1 > cp) cp = cp ? 2 * cp : 16;
    while (o->nr + 1 > cr) cr = cr ? 2 * cr : 16;
    while (o->npc + k > cc) cc = cc ? 2 * cc : 64;
    if (cp == o->cap_p && cr == o->cap_r && cc == o->cap_pc) return TR_MATH_OK;
    int32_t rc = tr_budget_check_one(c->budget, TR_BUDGET_BYTES, out_bytes(o, cp, cr, cc), c->refusal);
    if (rc) return rc;
    if (cp != o->cap_p) {
        pt *p = realloc(o->pts, (size_t)cp * sizeof(pt));
        int32_t *b = realloc(o->pbound, (size_t)cp * sizeof(int32_t));
        if (p) o->pts = p;
        if (b) o->pbound = b;
        if (!p || !b) return TR_MATH_MEMORY;
        o->cap_p = cp;
    }
    if (cr != o->cap_r) { reg_t *r = realloc(o->regs, (size_t)cr * sizeof(reg_t)); if (!r) return TR_MATH_MEMORY; o->regs = r; o->cap_r = cr; }
    if (cc != o->cap_pc) { uint64_t *q = realloc(o->pcls, (size_t)cc * 8); if (!q) return TR_MATH_MEMORY; o->pcls = q; o->cap_pc = cc; }
    return TR_MATH_OK;
}
static void emit_piece(ctx *c, out_t *o, uint64_t cls, int closed) {
    o->count++; o->pieces++;
    if (o->storing && o->count > o->limit) o->storing = 0;     /* past the output budget: count only */
    if (!o->storing || c->rc) return;
    int32_t rc = out_room(c, o, 0);
    if (rc) { c->rc = rc; return; }
    reg_t *r = &o->regs[o->nr++];
    memset(r, 0, sizeof *r);
    r->kind = TR_ENV_PIECE; r->cls = cls; r->ncls = 1;
    r->point = o->np ? o->np - 1 : NONE;                       /* the last point emitted (or an unbounded end) */
    r->right = closed ? o->np : NONE;                          /* the next point emitted */
}
/* a point with its classes M[0..m) (sorted); takes a canonical copy of x */
static void emit_point(ctx *c, out_t *o, const pt *x, const uint64_t *M, uint64_t m, int32_t boundary) {
    o->count++; o->points++;
    if (x->kind == P_ALG) o->alg++;
    if (o->storing && o->count > o->limit) o->storing = 0;
    if (!o->storing || c->rc) return;
    int32_t rc = out_room(c, o, m);
    if (rc) { c->rc = rc; return; }
    pt *p = &o->pts[o->np];
    if (x->kind == P_RAT) {                                    /* canonical: gcd(|n|, d) = 1 */
        tmp t = {{0}, 0};
        tr_bigint *g = zgcd(c, &t, x->n, x->d);
        memset(p, 0, sizeof *p);
        p->kind = P_RAT;
        p->n = zdivx(c, NULL, x->n, g); p->d = zdivx(c, NULL, x->d, g);
        tmp_free(&t);
        if (c->rc) { pt_clear(p); return; }
    } else if (pt_copy(c, x, p)) return;
    o->pbound[o->np] = boundary;
    o->np++;
    reg_t *r = &o->regs[o->nr++];
    memset(r, 0, sizeof *r);
    r->kind = TR_ENV_POINT; r->boundary = boundary; r->point = o->np - 1; r->right = NONE;
    r->first = o->npc; r->ncls = m; r->cls = M[0];
    memcpy(o->pcls + o->npc, M, (size_t)m * 8);
    o->npc += m;
}

static int cmp_u64(const void *a, const void *b) {
    uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
    return x < y ? -1 : x > y;
}
/* the class smallest just right of x among M[0..m) (all equal at x) */
static uint64_t right_min(ctx *c, const cls_t *K, const uint64_t *M, uint64_t m, const pt *x) {
    uint64_t best = M[0];
    for (uint64_t i = 1; i < m && !c->rc; i++) {
        tmp t = {{0}, 0};
        tr_bigint *D[3];
        diff(c, &t, &K[M[i]], &K[best], D);
        if (!c->rc && sign_right(c, D, x) < 0) best = M[i];
        tmp_free(&t);
    }
    return best;
}
/* the classes attaining the minimum at a rational x: argmin into M (sorted), *m */
static void min_set(ctx *c, const cls_t *K, uint64_t n, const pt *x, uint64_t *M, uint64_t *m) {
    uint64_t best = 0, cnt = 1;
    M[0] = 0;
    for (uint64_t k = 1; k < n && !c->rc; k++) {
        tmp t = {{0}, 0};
        tr_bigint *D[3];
        diff(c, &t, &K[k], &K[best], D);
        int s = c->rc ? 0 : sign_at(c, D, x);
        tmp_free(&t);
        if (s < 0) { best = k; cnt = 1; M[0] = k; }
        else if (s == 0) M[cnt++] = k;
    }
    qsort(M, (size_t)cnt, 8, cmp_u64);
    *m = cnt;
}

/* the left-to-right sweep over the classes K[0..n) on [lo, hi] (NULL point pointers: unbounded) */
static void sweep(ctx *c, const cls_t *K, uint64_t n, const pt *lo, const pt *hi, out_t *o) {
    uint64_t *M = zalloc(n * 8), *tied = zalloc(n * 8), m = 0, w = 0;
    pt x, best, r[2];
    memset(&x, 0, sizeof x); memset(&best, 0, sizeof best); memset(r, 0, sizeof r);
    if (!M || !tied) { c->rc = TR_MATH_MEMORY; goto done; }
    if (lo) {
        if (pt_copy(c, lo, &x)) goto done;
        min_set(c, K, n, &x, M, &m);
        int same = hi && !c->rc && pt_cmp(c, lo, hi) == 0;
        emit_point(c, o, &x, M, m, TR_ENV_AT_LO | (same ? TR_ENV_AT_HI : 0));
        if (same || c->rc) goto done;
        w = right_min(c, K, M, m, &x);
    } else {
        x.kind = P_NEG;
        for (uint64_t k = 1; k < n && !c->rc; k++) {           /* the class smallest as t -> -inf */
            tmp t = {{0}, 0};
            tr_bigint *D[3];
            diff(c, &t, &K[k], &K[w], D);
            if (!c->rc && sign_at(c, D, &x) < 0) w = k;
            tmp_free(&t);
        }
    }
    while (!c->rc) {
        uint64_t nt = 0;
        pt_clear(&best);
        for (uint64_t k = 0; k < n && !c->rc; k++) {
            if (k == w) continue;
            tmp t = {{0}, 0};
            tr_bigint *D[3];
            diff(c, &t, &K[k], &K[w], D);
            int cnt = roots(c, D, r);
            tmp_free(&t);
            for (int i = 0; i < cnt && !c->rc; i++) {
                if (pt_cmp(c, &r[i], &x) <= 0) continue;        /* at or before the current position */
                if (hi && pt_cmp(c, &r[i], hi) >= 0) break;      /* at or past hi: not an interior point */
                int s = best.kind ? pt_cmp(c, &r[i], &best) : -1;
                if (s < 0) { pt_move(&best, &r[i]); nt = 0; tied[nt++] = k; }
                else if (s == 0) tied[nt++] = k;
                break;                                         /* only the first root past x matters */
            }
            for (int i = 0; i < 2; i++) pt_clear(&r[i]);
        }
        if (c->rc) break;
        if (!best.kind) {                                      /* w attains the envelope up to hi */
            emit_piece(c, o, w, hi != NULL);
            if (hi && !c->rc) {
                M[0] = w; m = 1;
                for (uint64_t k = 0; k < n && !c->rc; k++) {
                    if (k == w) continue;
                    tmp t = {{0}, 0};
                    tr_bigint *D[3];
                    diff(c, &t, &K[k], &K[w], D);
                    if (!c->rc && sign_at(c, D, hi) == 0) M[m++] = k;
                    tmp_free(&t);
                }
                qsort(M, (size_t)m, 8, cmp_u64);
                emit_point(c, o, hi, M, m, TR_ENV_AT_HI);
            }
            break;
        }
        emit_piece(c, o, w, 1);
        M[0] = w; m = 1;
        for (uint64_t i = 0; i < nt; i++) M[m++] = tied[i];
        qsort(M, (size_t)m, 8, cmp_u64);
        emit_point(c, o, &best, M, m, 0);
        if (c->rc) break;
        w = right_min(c, K, M, m, &best);
        pt_move(&x, &best);
    }
done:
    pt_clear(&x); pt_clear(&best); pt_clear(&r[0]); pt_clear(&r[1]);
    free(M); free(tied);
}

/* ------------------------------------------------------------------ classes from rational coefficients */
static int cls_less(ctx *c, const cls_t *a, const cls_t *b) {   /* the canonical order (d, C, B, A) */
    int r = zcmp(c, a->d, b->d);
    for (int k = 2; !r && k >= 0; k--) r = zcmp(c, a->P[k], b->P[k]);
    return r;
}
static void merge_sort(ctx *c, const cls_t *K, uint64_t *idx, uint64_t *scratch, uint64_t n) {
    if (n < 2) return;
    uint64_t h = n / 2;
    merge_sort(c, K, idx, scratch, h);
    merge_sort(c, K, idx + h, scratch, n - h);
    uint64_t i = 0, j = h, w = 0;
    while (i < h || j < n) {
        if (j >= n || (i < h && cls_less(c, &K[idx[i]], &K[idx[j]]) <= 0)) scratch[w++] = idx[i++];
        else scratch[w++] = idx[j++];
    }
    memcpy(idx, scratch, (size_t)n * 8);
}
/* the internal form of one function: (A + B t + C t^2) / d, d = lcm of the denominators (then gcd(A, B, C, d) = 1) */
static void cls_make(ctx *c, cls_t *k, const tr_bigq *const *q, int upper) {
    tmp t = {{0}, 0};
    tr_bigint *d = zi64(c, &t, 1);
    for (int i = 0; i < 3; i++) {
        const tr_bigint *den = tr_bigq_denominator(q[i]);
        tr_bigint *g = zgcd(c, &t, d, den);
        d = zmul(c, &t, zdivx(c, &t, d, g), den);
    }
    k->d = zcopy(c, NULL, d);
    k->degree = 0;
    for (int i = 0; i < 3; i++) {
        tr_bigint *a = zmul(c, &t, tr_bigq_numerator(q[i]), zdivx(c, &t, d, tr_bigq_denominator(q[i])));
        k->P[i] = upper ? zneg(c, NULL, a) : zcopy(c, NULL, a);
        if (zs(c, a)) k->degree = i;
        int32_t e = 0;
        if (!c->rc) { k->orig[i] = tr_bigq_copy(q[i], c->budget, c->refusal, &e); if (!k->orig[i]) c->rc = e ? e : TR_MATH_MEMORY; }
    }
    tmp_free(&t);
}

/* INV-001's isolating interval of an algebraic point and the sign of q at its lower end */
static void isolate(ctx *c, const pt *p, tr_bigq **lo, tr_bigq **hi, int32_t *sign_lo) {
    tr_bigq *coef[3] = {NULL, NULL, NULL};
    int32_t e = 0;
    for (int k = 0; k < 3 && !c->rc; k++) {
        coef[k] = tr_bigq_from_bigint(p->q[k], c->budget, c->refusal, &e);
        if (!coef[k]) c->rc = e ? e : TR_MATH_MEMORY;
    }
    tr_inv_roots *roots_ = NULL;
    tr_inv_roots_info info;
    if (!c->rc) {                                     /* INV-001 works within the iterations the sweep left */
        tr_math_budget b = *c->budget;
        b.limit[TR_BUDGET_ITERATIONS] = c->limit > c->iterations ? c->limit - c->iterations : 0;
        roots_ = tr_inv_isolate(3, (const tr_bigq *const *)coef, NULL, NULL, NULL, &b, c->refusal, &e);
        if (!roots_) {
            c->rc = e ? e : TR_MATH_MEMORY;
            if (c->rc == TR_MATH_BUDGET && c->refusal && c->refusal->resource == TR_BUDGET_ITERATIONS) {
                c->refusal->need = c->limit + 1;          /* the build's convention: refused at limit + 1 */
                c->refusal->limit = c->limit;
            }
        } else if (!tr_inv_roots_get_info(roots_, &info)) c->iterations += info.iterations;
    }
    tr_inv_root_info ri;
    const tr_bigq *l = NULL, *h = NULL;
    if (!c->rc && (tr_inv_roots_get_info(roots_, &info) || info.nroots != 2 ||
                   tr_inv_roots_root(roots_, p->sigma < 0 ? 0 : 1, &ri, &l, &h) || ri.exact)) c->rc = TR_MATH_INVALID;
    if (!c->rc) {
        *lo = tr_bigq_copy(l, c->budget, c->refusal, &e);
        if (*lo) *hi = tr_bigq_copy(h, c->budget, c->refusal, &e);
        if (!*lo || !*hi) c->rc = e ? e : TR_MATH_MEMORY;
    }
    if (!c->rc) {
        pt at;
        pt_rat(c, &at, tr_bigq_numerator(*lo), tr_bigq_denominator(*lo));
        tr_bigint *const *q = p->q;
        *sign_lo = sign_at(c, q, &at);
        pt_clear(&at);
        if (!c->rc && *sign_lo == 0) c->rc = TR_MATH_INVALID;      /* lo is never a root of the irreducible q */
    }
    tr_inv_roots_free(roots_);
    for (int k = 0; k < 3; k++) tr_bigq_free(coef[k]);
}

/* the whole build from per-group rational coefficients (coef[g * 3 + k], NULL = 0; present[g]) */
static tr_env *build(uint64_t ngroups, const tr_bigq *const *coef, const uint8_t *present, const tr_bigq *lo,
                     const tr_bigq *hi, int32_t upper, int32_t tie, const tr_math_budget *budget,
                     tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    tr_math_budget dflt;
    budget = budget_or(budget, &dflt);
    if ((upper != 0 && upper != 1) || (tie != TR_ENV_TIE_SMALLEST && tie != TR_ENV_TIE_LARGEST) ||
        (lo && hi && tr_bigq_cmp(lo, hi) > 0)) { set_error(error, TR_MATH_INVALID); return NULL; }
    uint64_t np_ = 0;
    int32_t degree = 0;
    for (uint64_t g = 0; g < ngroups; g++) {
        if (present && !present[g]) continue;
        np_++;
        for (int k = 0; k < 3; k++) if (coef[g * 3 + k] && tr_bigq_sign(coef[g * 3 + k]) && k > degree) degree = k;
    }
    tr_math_need need;
    tr_env_need(ngroups, np_, degree, &need);
    int32_t rc = tr_budget_check(budget, &need, refusal);
    if (rc) { set_error(error, rc); return NULL; }
    ctx c = {TR_MATH_OK, budget, refusal, 0, budget->limit[TR_BUDGET_ITERATIONS]};
    tr_env *e = zalloc(sizeof *e);
    cls_t *raw = zalloc(tr_budget_mul(np_, sizeof(cls_t)));
    uint64_t *gid = zalloc(tr_budget_mul(np_, 8)), *idx = zalloc(tr_budget_mul(np_, 8)), *scr = zalloc(tr_budget_mul(np_, 8));
    tr_bigq *zero = NULL;
    int32_t ze = 0;
    zero = tr_bigq_from_q64(tr_q_int(0), budget, refusal, &ze);
    out_t o;
    memset(&o, 0, sizeof o);
    if (!e || !raw || !gid || !idx || !scr || !zero) { c.rc = TR_MATH_MEMORY; goto fail; }
    e->ngroups = ngroups; e->present = np_; e->upper = upper; e->tie = tie; e->degree = degree;
    e->lo_bounded = lo != NULL; e->hi_bounded = hi != NULL;
    {
        int32_t er = 0;
        if (lo && !(e->lo = tr_bigq_copy(lo, budget, refusal, &er))) { c.rc = er ? er : TR_MATH_MEMORY; goto fail; }
        if (hi && !(e->hi = tr_bigq_copy(hi, budget, refusal, &er))) { c.rc = er ? er : TR_MATH_MEMORY; goto fail; }
    }
    uint64_t q = 0;
    for (uint64_t g = 0; g < ngroups && !c.rc; g++) {
        if (present && !present[g]) continue;
        const tr_bigq *row[3];
        for (int k = 0; k < 3; k++) row[k] = coef[g * 3 + k] ? coef[g * 3 + k] : zero;
        cls_make(&c, &raw[q], row, upper);
        gid[q] = g; idx[q] = q; q++;
    }
    if (c.rc) goto fail;
    if (!np_) { e->kind = TR_ENV_EMPTY; goto finish; }
    merge_sort(&c, raw, idx, scr, np_);
    /* classes: runs of equal internal forms; members in increasing group order */
    uint64_t ncls = 0;
    for (uint64_t i = 0; i < np_; i++) if (!i || cls_less(&c, &raw[idx[i - 1]], &raw[idx[i]])) ncls++;
    e->cls = zalloc(tr_budget_mul(ncls, sizeof(cls_t)));
    e->members = zalloc(tr_budget_mul(np_, 8));
    if (!e->cls || !e->members) { c.rc = TR_MATH_MEMORY; goto fail; }
    /* order the classes by their smallest member (deterministic output) */
    uint64_t *first_of = scr, cnum = 0;                         /* reuse scratch: run starts */
    for (uint64_t i = 0; i < np_; i++) if (!i || cls_less(&c, &raw[idx[i - 1]], &raw[idx[i]])) first_of[cnum++] = i;
    uint64_t *order = zalloc(tr_budget_mul(ncls, 8)), *minm = zalloc(tr_budget_mul(ncls, 8));
    if (!order || !minm) { free(order); free(minm); c.rc = TR_MATH_MEMORY; goto fail; }
    for (uint64_t r = 0; r < ncls; r++) {
        uint64_t a = first_of[r], b = r + 1 < ncls ? first_of[r + 1] : np_, mn = UINT64_MAX;
        for (uint64_t i = a; i < b; i++) if (gid[idx[i]] < mn) mn = gid[idx[i]];
        minm[r] = mn; order[r] = r;
    }
    for (uint64_t i = 1; i < ncls; i++) {                       /* insertion by smallest member (distinct) */
        uint64_t v = order[i], j = i;
        while (j > 0 && minm[order[j - 1]] > minm[v]) { order[j] = order[j - 1]; j--; }
        order[j] = v;
    }
    uint64_t at = 0;
    for (uint64_t r = 0; r < ncls; r++) {
        uint64_t run = order[r], a = first_of[run], b = run + 1 < ncls ? first_of[run + 1] : np_;
        cls_t *k = &e->cls[r];
        *k = raw[idx[a]];                                      /* move the representative */
        memset(&raw[idx[a]], 0, sizeof(cls_t));
        k->first = at; k->count = b - a;
        for (uint64_t i = a; i < b; i++) e->members[at++] = gid[idx[i]];
        qsort(e->members + k->first, (size_t)k->count, 8, cmp_u64);
        k->winner = tie == TR_ENV_TIE_SMALLEST ? e->members[k->first] : e->members[k->first + k->count - 1];
    }
    free(order); free(minm);
    e->nclasses = ncls;
    /* the sweep */
    pt plo, phi;
    memset(&plo, 0, sizeof plo); memset(&phi, 0, sizeof phi);
    if (lo) pt_rat(&c, &plo, tr_bigq_numerator(lo), tr_bigq_denominator(lo));
    if (hi) pt_rat(&c, &phi, tr_bigq_numerator(hi), tr_bigq_denominator(hi));
    o.limit = budget->limit[TR_BUDGET_OUTPUT]; o.storing = 1; o.fixed = need.need[TR_BUDGET_BYTES];
    if (!c.rc) sweep(&c, e->cls, ncls, lo ? &plo : NULL, hi ? &phi : NULL, &o);
    pt_clear(&plo); pt_clear(&phi);
    if (c.rc) goto fail;
    if (o.count > o.limit) {                                   /* the exact region count, refused */
        c.rc = TR_MATH_BUDGET;
        if (refusal) { refusal->resource = TR_BUDGET_OUTPUT; refusal->refused = 1; refusal->need = o.count;
                       refusal->limit = o.limit; }
        goto fail;
    }
    e->kind = TR_ENV_FINITE;
    e->npoints = o.np; e->nregions = o.nr; e->npieces = o.pieces; e->nalg = o.alg;
    e->pts = o.pts; e->regs = o.regs; e->pcls = o.pcls; e->pboundary = o.pbound;
    o.pts = NULL; o.regs = NULL; o.pcls = NULL; o.pbound = NULL; o.np = 0;
    e->pval = zalloc(tr_budget_mul(e->npoints, sizeof(void *)));
    e->plo = zalloc(tr_budget_mul(e->npoints, sizeof(void *)));
    e->phi = zalloc(tr_budget_mul(e->npoints, sizeof(void *)));
    e->psign = zalloc(tr_budget_mul(e->npoints, 4));
    e->pregion = zalloc(tr_budget_mul(e->npoints, 8));
    e->rwinner = zalloc(tr_budget_mul(e->nregions, 8));
    e->rmembers = zalloc(tr_budget_mul(e->nregions, 8));
    if (!e->pval || !e->plo || !e->phi || !e->psign || !e->pregion || !e->rwinner || !e->rmembers) { c.rc = TR_MATH_MEMORY; goto fail; }
    for (uint64_t i = 0; i < e->npoints && !c.rc; i++) {
        pt *p = &e->pts[i];
        int32_t er = 0;
        if (p->kind == P_RAT) {
            e->pval[i] = tr_bigq_from_bigints(p->n, p->d, budget, refusal, &er);
            if (!e->pval[i]) c.rc = er ? er : TR_MATH_MEMORY;
        } else isolate(&c, p, &e->plo[i], &e->phi[i], &e->psign[i]);
    }
    for (uint64_t r = 0; r < e->nregions; r++) {
        reg_t *g = &e->regs[r];
        if (g->kind == TR_ENV_POINT) e->pregion[g->point] = r;
        uint64_t win = NONE, mem = 0;
        for (uint64_t i = 0; i < g->ncls; i++) {
            const cls_t *k = &e->cls[g->kind == TR_ENV_POINT ? e->pcls[g->first + i] : g->cls];
            mem += k->count;
            if (win == NONE || (tie == TR_ENV_TIE_SMALLEST ? k->winner < win : k->winner > win)) win = k->winner;
        }
        e->rwinner[r] = win; e->rmembers[r] = mem;
    }
    if (c.rc) goto fail;
    e->iterations = c.iterations;                              /* the sweep's and INV-001's isolation */
finish:
    e->bytes = out_bytes(&o, o.cap_p, o.cap_r, o.cap_pc);      /* the largest amount the growth checked */
    if (e->bytes < need.need[TR_BUDGET_BYTES]) e->bytes = need.need[TR_BUDGET_BYTES];
    e->magic = ENV_MAGIC;
    for (uint64_t i = 0; raw && i < np_; i++) cls_clear(&raw[i]);
    free(raw); free(gid); free(idx); free(scr); tr_bigq_free(zero); out_clear(&o);
    set_error(error, TR_MATH_OK);
    return e;
fail:
    for (uint64_t i = 0; raw && i < np_; i++) cls_clear(&raw[i]);
    free(raw); free(gid); free(idx); free(scr); tr_bigq_free(zero); out_clear(&o);
    env_release(e);
    set_error(error, c.rc ? c.rc : TR_MATH_MEMORY);
    return NULL;
}

tr_env *tr_env_build(uint64_t ngroups, int32_t ncoef, const tr_bigq *const *coef, const uint8_t *present,
                     const tr_bigq *lo, const tr_bigq *hi, int32_t upper, int32_t tie, const tr_math_budget *budget,
                     tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    if (ncoef < 1 || ncoef > 8 || (ngroups && !coef) || ngroups > TR_SCOPE_MAX_GROUPS) {
        set_error(error, ngroups > TR_SCOPE_MAX_GROUPS ? TR_MATH_UNSUPPORTED : TR_MATH_INVALID); return NULL;
    }
    for (uint64_t g = 0; g < ngroups; g++) {                  /* a nonzero coefficient of degree >= 3 */
        if (present && !present[g]) continue;
        for (int32_t k = 3; k < ncoef; k++) {
            const tr_bigq *x = coef[g * (uint64_t)ncoef + (uint64_t)k];
            if (x && tr_bigq_sign(x)) {                       /* over the degree budget, else beyond the scope */
                tr_math_budget dflt;
                const tr_math_budget *b = budget_or(budget, &dflt);
                if (tr_budget_check_one(b, TR_BUDGET_DEGREE, (uint64_t)k, refusal)) { set_error(error, TR_MATH_BUDGET); return NULL; }
                set_error(error, TR_MATH_UNSUPPORTED); return NULL;
            }
        }
    }
    const tr_bigq **c3 = zalloc(tr_budget_mul(tr_budget_mul(ngroups, 3), sizeof(void *)));
    if (!c3) { set_error(error, TR_MATH_MEMORY); return NULL; }
    for (uint64_t g = 0; g < ngroups; g++)
        for (int32_t k = 0; k < 3 && k < ncoef; k++) c3[g * 3 + (uint64_t)k] = coef[g * (uint64_t)ncoef + (uint64_t)k];
    tr_env *e = build(ngroups, (const tr_bigq *const *)c3, present, lo, hi, upper, tie, budget, refusal, error);
    free(c3);
    return e;
}

tr_env *tr_env_from_polys(uint64_t ngroups, const tr_math_poly *const *polys, int32_t variable, const tr_bigq *lo,
                          const tr_bigq *hi, int32_t upper, int32_t tie, const tr_math_budget *budget,
                          tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    if ((ngroups && !polys) || variable < 0 || variable >= TR_MATH_VARIABLES) { set_error(error, TR_MATH_INVALID); return NULL; }
    tr_math_budget dflt;
    budget = budget_or(budget, &dflt);
    tr_bigq **coef = zalloc(tr_budget_mul(tr_budget_mul(ngroups, 3), sizeof(void *)));
    uint8_t *present = zalloc(ngroups);
    int32_t rc = coef && present ? TR_MATH_OK : TR_MATH_MEMORY;
    for (uint64_t g = 0; g < ngroups && !rc; g++) {
        if (!polys[g]) continue;
        present[g] = 1;
        int32_t terms = tr_math_poly_term_count(polys[g]);
        if (terms < 0) { rc = TR_MATH_INVALID; break; }
        for (int32_t i = 0; i < terms && !rc; i++) {
            tr_math_term term;
            if (tr_math_poly_get_term(polys[g], i, &term)) { rc = TR_MATH_INVALID; break; }
            int32_t deg = 0;
            for (int32_t v = 0; v < TR_MATH_VARIABLES; v++) {
                if (!term.exponents[v]) continue;
                if (v != variable) { rc = TR_MATH_UNSUPPORTED; break; }   /* a parameter dimension above 1 */
                deg = term.exponents[v];
            }
            if (rc) break;
            if (deg > 2) {
                rc = tr_budget_check_one(budget, TR_BUDGET_DEGREE, (uint64_t)deg, refusal);
                if (!rc) rc = TR_MATH_UNSUPPORTED;
                break;
            }
            int32_t e = 0;
            tr_num_rational q = {term.coefficient.numerator, term.coefficient.denominator};
            coef[g * 3 + (uint64_t)deg] = tr_bigq_from_q64(q, budget, refusal, &e);
            if (!coef[g * 3 + (uint64_t)deg]) rc = e ? e : TR_MATH_MEMORY;
        }
    }
    tr_env *env = NULL;
    if (!rc) env = build(ngroups, (const tr_bigq *const *)coef, present, lo, hi, upper, tie, budget, refusal, error);
    else set_error(error, rc);
    for (uint64_t i = 0; coef && i < ngroups * 3; i++) tr_bigq_free(coef[i]);
    free(coef); free(present);
    return env;
}

tr_env *tr_env_from_bank(const tr_bank *bank, const int32_t *term, const int64_t *mult, int32_t per,
                         const tr_bigq *lo, const tr_bigq *hi, int32_t upper, int32_t tie, const tr_math_budget *budget,
                         tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    tr_bank_info bi;
    tr_bank_states st;
    if (!bank || !term || !mult || tr_bank_get_info(bank, &bi) || tr_bank_get_states(bank, &st) ||
        (per < TR_ENV_PER_ROWS || per >= bi.nterms)) { set_error(error, TR_MATH_INVALID); return NULL; }
    for (int k = 0; k < 3; k++) if (term[k] >= bi.nterms) { set_error(error, TR_MATH_INVALID); return NULL; }
    tr_math_budget dflt;
    budget = budget_or(budget, &dflt);
    const uint64_t G = bi.ngroups, m = (uint64_t)bi.nterms;
    tr_bigq **coef = zalloc(tr_budget_mul(tr_budget_mul(G, 3), sizeof(void *)));
    uint8_t *present = zalloc(G);
    int32_t rc = coef && present ? TR_MATH_OK : TR_MATH_MEMORY;
    for (uint64_t g = 0; g < G && !rc; g++) {
        uint64_t rows = st.rows[g];
        if (!rows) continue;                                    /* an absent group */
        __int128 den = per == TR_ENV_PER_NONE ? 1 : per == TR_ENV_PER_ROWS ? (__int128)rows : (__int128)st.count[g * m + (uint64_t)per];
        if (!den) continue;                                     /* an undefined mean: empty */
        present[g] = 1;
        int32_t e = 0;
        tr_bigint *d = tr_bigint_from_i128(tr_i128_set(den), budget, refusal, &e);
        if (!d) { rc = e ? e : TR_MATH_MEMORY; break; }
        for (int k = 0; k < 3 && !rc; k++) {
            if (term[k] < 0 || !mult[k]) continue;
            tr_bigint *s = tr_bigint_from_i128(st.sum[g * m + (uint64_t)term[k]], budget, refusal, &e), *f = NULL, *n = NULL;
            if (s) f = tr_bigint_from_i64(mult[k], budget, refusal, &e);
            if (f) n = tr_bigint_mul(s, f, budget, refusal, &e);
            if (n) coef[g * 3 + (uint64_t)k] = tr_bigq_from_bigints(n, d, budget, refusal, &e);
            if (!coef[g * 3 + (uint64_t)k]) rc = e ? e : TR_MATH_MEMORY;
            tr_bigint_free(s); tr_bigint_free(f); tr_bigint_free(n);
        }
        tr_bigint_free(d);
    }
    tr_env *env = NULL;
    if (!rc) env = build(G, (const tr_bigq *const *)coef, present, lo, hi, upper, tie, budget, refusal, error);
    else set_error(error, rc);
    for (uint64_t i = 0; coef && i < G * 3; i++) tr_bigq_free(coef[i]);
    free(coef); free(present);
    return env;
}

/* ------------------------------------------------------------------ reading */
int32_t tr_env_get_info(const tr_env *e, tr_env_info *out) {
    if (!env_valid(e) || !out) return TR_MATH_INVALID;
    memset(out, 0, sizeof *out);
    out->groups = e->ngroups; out->present = e->present; out->classes = e->nclasses; out->regions = e->nregions;
    out->pieces = e->npieces; out->points = e->npoints; out->iterations = e->iterations; out->bytes = e->bytes;
    out->kind = e->kind; out->upper = e->upper; out->tie = e->tie; out->degree = e->degree;
    out->lo_bounded = e->lo_bounded; out->hi_bounded = e->hi_bounded; out->algebraic = (int32_t)e->nalg;
    return TR_MATH_OK;
}
int32_t tr_env_region_get(const tr_env *e, uint64_t i, tr_env_region *out) {
    if (!env_valid(e) || !out || i >= e->nregions) return TR_MATH_INVALID;
    const reg_t *r = &e->regs[i];
    memset(out, 0, sizeof *out);
    out->kind = r->kind; out->boundary = r->boundary; out->point = r->point; out->right = r->right;
    out->classes = r->ncls; out->cls = r->kind == TR_ENV_POINT ? e->pcls[r->first] : r->cls;
    out->winner = e->rwinner[i]; out->members = e->rmembers[i];
    return TR_MATH_OK;
}
int32_t tr_env_region_classes(const tr_env *e, uint64_t i, uint64_t *out, uint64_t capacity) {
    if (!env_valid(e) || !out || i >= e->nregions || capacity < e->regs[i].ncls) return TR_MATH_INVALID;
    const reg_t *r = &e->regs[i];
    if (r->kind == TR_ENV_PIECE) out[0] = r->cls;
    else memcpy(out, e->pcls + r->first, (size_t)r->ncls * 8);
    return TR_MATH_OK;
}
int32_t tr_env_class_get(const tr_env *e, uint64_t k, tr_env_class_info *out) {
    if (!env_valid(e) || !out || k >= e->nclasses) return TR_MATH_INVALID;
    memset(out, 0, sizeof *out);
    out->members = e->cls[k].count; out->winner = e->cls[k].winner; out->degree = e->cls[k].degree;
    return TR_MATH_OK;
}
int32_t tr_env_class_members(const tr_env *e, uint64_t k, uint64_t *out, uint64_t capacity) {
    if (!env_valid(e) || !out || k >= e->nclasses || capacity < e->cls[k].count) return TR_MATH_INVALID;
    memcpy(out, e->members + e->cls[k].first, (size_t)e->cls[k].count * 8);
    return TR_MATH_OK;
}
const tr_bigq *tr_env_class_coef(const tr_env *e, uint64_t k, int32_t i) {
    if (!env_valid(e) || k >= e->nclasses || i < 0 || i > 2) return NULL;
    return e->cls[k].orig[i];
}
int32_t tr_env_point_get(const tr_env *e, uint64_t p, tr_env_point_info *out) {
    if (!env_valid(e) || !out || p >= e->npoints) return TR_MATH_INVALID;
    memset(out, 0, sizeof *out);
    out->kind = e->pts[p].kind == P_RAT ? TR_ENV_RATIONAL : TR_ENV_ALGEBRAIC;
    out->root = e->pts[p].sigma > 0; out->boundary = e->pboundary[p]; out->sign_lo = e->psign[p];
    out->region = e->pregion[p];
    return TR_MATH_OK;
}
const tr_bigq *tr_env_point_value(const tr_env *e, uint64_t p) {
    return env_valid(e) && p < e->npoints ? e->pval[p] : NULL;
}
const tr_bigint *tr_env_point_poly(const tr_env *e, uint64_t p, int32_t k) {
    if (!env_valid(e) || p >= e->npoints || k < 0 || k > 2 || e->pts[p].kind != P_ALG) return NULL;
    return e->pts[p].q[k];
}
int32_t tr_env_point_interval(const tr_env *e, uint64_t p, const tr_bigq **lo, const tr_bigq **hi) {
    if (!env_valid(e) || !lo || !hi || p >= e->npoints || e->pts[p].kind != P_ALG) return TR_MATH_INVALID;
    *lo = e->plo[p]; *hi = e->phi[p];
    return TR_MATH_OK;
}
int32_t tr_env_point_refine(const tr_env *e, uint64_t p, const tr_bigq *width, tr_bigq **lo, tr_bigq **hi,
                            const tr_math_budget *budget, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!env_valid(e) || !width || !lo || !hi || p >= e->npoints || e->pts[p].kind != P_ALG || tr_bigq_sign(width) <= 0)
        return TR_MATH_INVALID;
    tr_math_budget dflt;
    budget = budget_or(budget, &dflt);
    int32_t err = 0;
    tr_bigq *coef[3] = {NULL, NULL, NULL};
    for (int k = 0; k < 3; k++) {
        coef[k] = tr_bigq_from_bigint(e->pts[p].q[k], budget, refusal, &err);
        if (!coef[k]) { for (int i = 0; i < k; i++) tr_bigq_free(coef[i]); return err ? err : TR_MATH_MEMORY; }
    }
    /* isolate within the stored interval (one root of q there), then refine with INV-001's exact signs */
    tr_inv_roots *r = tr_inv_isolate(3, (const tr_bigq *const *)coef, NULL, e->plo[p], e->phi[p], budget, refusal, &err);
    tr_inv_roots *f = r ? tr_inv_roots_refine(r, width, NULL, budget, refusal, &err) : NULL;
    int32_t rc = f ? TR_MATH_OK : (err ? err : TR_MATH_MEMORY);
    tr_inv_roots_info info;
    tr_inv_root_info ri;
    const tr_bigq *l = NULL, *h = NULL;
    if (!rc && (tr_inv_roots_get_info(f, &info) || info.nroots != 1 || tr_inv_roots_root(f, 0, &ri, &l, &h))) rc = TR_MATH_INVALID;
    if (!rc) {
        *lo = tr_bigq_copy(l, budget, refusal, &err);
        *hi = *lo ? tr_bigq_copy(h, budget, refusal, &err) : NULL;
        if (!*lo || !*hi) { tr_bigq_free(*lo); *lo = NULL; rc = err ? err : TR_MATH_MEMORY; }
    }
    tr_inv_roots_free(r); tr_inv_roots_free(f);
    for (int k = 0; k < 3; k++) tr_bigq_free(coef[k]);
    return rc;
}
const tr_bigq *tr_env_bound(const tr_env *e, int32_t hi) { return env_valid(e) ? (hi ? e->hi : e->lo) : NULL; }

/* the region of the envelope containing the exact point x (NONE: outside) */
static uint64_t locate(ctx *c, const tr_env *e, const pt *x) {
    uint64_t lo = 0, hi = e->npoints;
    int at = 0;
    while (lo < hi && !c->rc) {
        uint64_t mid = lo + (hi - lo) / 2;
        int s = pt_cmp(c, x, &e->pts[mid]);
        if (s <= 0) { hi = mid; at = s == 0; } else lo = mid + 1;
    }
    if (c->rc) return NONE;
    if (at) return e->pregion[lo];
    if ((lo == 0 && e->lo_bounded) || (lo == e->npoints && e->hi_bounded)) return NONE;
    /* the piece between points lo - 1 and lo: the region after point lo - 1, or the first region */
    return lo == 0 ? 0 : e->pregion[lo - 1] + 1;
}
int32_t tr_env_value(const tr_env *e, const tr_bigq *t, tr_bigq **out, uint64_t *region, int32_t *status,
                     const tr_math_budget *budget, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!env_valid(e) || !t || !out || !region || !status) return TR_MATH_INVALID;
    *out = NULL; *region = NONE; *status = TR_ENV_Q_OK;
    if (e->kind == TR_ENV_EMPTY) { *status = TR_ENV_Q_EMPTY; return TR_MATH_OK; }
    tr_math_budget dflt;
    budget = budget_or(budget, &dflt);
    ctx c = {TR_MATH_OK, budget, refusal, 0, UINT64_MAX};
    pt x;
    pt_rat(&c, &x, tr_bigq_numerator(t), tr_bigq_denominator(t));
    uint64_t r = c.rc ? NONE : locate(&c, e, &x);
    pt_clear(&x);
    if (c.rc) return c.rc;
    if (r == NONE) { *status = TR_ENV_Q_OUTSIDE; return TR_MATH_OK; }
    *region = r;
    const reg_t *g = &e->regs[r];
    const cls_t *k = &e->cls[g->kind == TR_ENV_POINT ? e->pcls[g->first] : g->cls];
    int32_t err = 0;                                           /* a + b t + c t^2 of the class's given coefficients */
    tr_bigq *t2 = tr_bigq_mul(t, t, budget, refusal, &err), *bt = NULL, *ct = NULL, *s = NULL, *v = NULL;
    if (t2) bt = tr_bigq_mul(k->orig[1], t, budget, refusal, &err);
    if (bt) ct = tr_bigq_mul(k->orig[2], t2, budget, refusal, &err);
    if (ct) s = tr_bigq_add(k->orig[0], bt, budget, refusal, &err);
    if (s) v = tr_bigq_add(s, ct, budget, refusal, &err);
    tr_bigq_free(t2); tr_bigq_free(bt); tr_bigq_free(ct); tr_bigq_free(s);
    if (!v) return err ? err : TR_MATH_MEMORY;
    *out = v;
    return TR_MATH_OK;
}

/* ================================================================== lookups */
struct tr_env_lookup_s {
    uint32_t magic;
    tr_env_view view;
    uint64_t bound_bits, bytes;
    tr_gfp *fields;
    uint64_t *garner, *half, *bres, *point_region, *piece_region, *rwinner, *rclasses, *cres, *cwinner, *qnum, *qden;
    int32_t *bkind;
    int8_t *bsign, *qsign;
};
static int lookup_valid(const tr_env_lookup *l) { return l && l->magic == LOOKUP_MAGIC; }
void tr_env_lookup_free(tr_env_lookup *l) {
    if (!lookup_valid(l)) return;
    free(l->fields); free(l->garner); free(l->half); free(l->bres); free(l->point_region); free(l->piece_region);
    free(l->rwinner); free(l->rclasses); free(l->cres); free(l->cwinner); free(l->qnum); free(l->qden);
    free(l->bkind); free(l->bsign); free(l->qsign);
    l->magic = 0;
    free(l);
}
static uint64_t bits_of(const tr_bigint *x) { return tr_bigint_bit_length(x); }
/* bits(|a| + |b| + |c|) of up to three integers (NULL = 0), bounded above by max bits + 2 */
static uint64_t bits_sum(const tr_bigint *a, const tr_bigint *b, const tr_bigint *c) {
    uint64_t m = 0;
    const tr_bigint *v[3] = {a, b, c};
    for (int i = 0; i < 3; i++) if (v[i] && bits_of(v[i]) > m) m = bits_of(v[i]);
    return m + 2;
}
/* the query points: big rationals, or int64 numerators and positive int64 denominators */
typedef struct { uint64_t n; const tr_bigq *const *big; const int64_t *num, *den; } qsrc;
static uint64_t bits_u64(uint64_t x) { uint64_t b = 0; while (x) { b++; x >>= 1; } return b; }
/* the query limbs and bits (max over the queries of max(bits |u|, bits v)) */
static int32_t query_sizes(const qsrc *q, uint32_t *limbs, uint64_t *qbits) {
    uint64_t lm = 1, qb = 1;
    if ((q->n && !q->big && (!q->num || !q->den))) return TR_MATH_INVALID;
    for (uint64_t i = 0; i < q->n; i++) {
        if (q->big) {
            if (!q->big[i]) return TR_MATH_INVALID;
            const tr_bigint *u = tr_bigq_numerator(q->big[i]), *v = tr_bigq_denominator(q->big[i]);
            uint64_t a = tr_bigint_limb_count(u), b = tr_bigint_limb_count(v);
            if (a > lm) lm = a;
            if (b > lm) lm = b;
            if (bits_of(u) > qb) qb = bits_of(u);
            if (bits_of(v) > qb) qb = bits_of(v);
        } else {
            if (q->den[i] < 1) return TR_MATH_INVALID;
            uint64_t a = bits_u64(tr_abs_u64(q->num[i])), b = bits_u64((uint64_t)q->den[i]);
            if (a > qb) qb = a;
            if (b > qb) qb = b;
        }
    }
    if (lm > TR_INV_POINT_LIMBS_MAX) return TR_MATH_OVERFLOW;
    *limbs = (uint32_t)lm; *qbits = qb;
    return TR_MATH_OK;
}
/* the queries' sign, |u| and v limbs */
static int32_t query_fill(const qsrc *q, uint32_t limbs, int8_t *sign, uint64_t *qn, uint64_t *qd) {
    for (uint64_t i = 0; i < q->n; i++) {
        if (q->big) {
            uint64_t cnt = 0;
            int32_t sg = 0;
            const tr_bigint *u = tr_bigq_numerator(q->big[i]), *d = tr_bigq_denominator(q->big[i]);
            if (tr_bigint_export(u, &sg, qn + i * limbs, limbs, &cnt) || tr_bigint_export(d, &sg, qd + i * limbs, limbs, &cnt))
                return TR_MATH_INVALID;
            sign[i] = (int8_t)tr_bigint_sign(u);
        } else {
            qn[i * limbs] = tr_abs_u64(q->num[i]); qd[i * limbs] = (uint64_t)q->den[i];
            sign[i] = (int8_t)(q->num[i] > 0 ? 1 : q->num[i] < 0 ? -1 : 0);
        }
    }
    return TR_MATH_OK;
}
/* B (|S| < 2^B for every comparison of the job) */
static uint64_t job_bits(const tr_env *e, int32_t mode, uint64_t qbits) {
    uint64_t B = 0;
    for (uint64_t i = 0; i < e->npoints; i++) {
        uint64_t b;
        if (e->pts[i].kind == P_RAT) b = bits_sum(e->pts[i].n, e->pts[i].d, NULL) + qbits;
        else {
            uint64_t l = bits_sum(tr_bigq_numerator(e->plo[i]), tr_bigq_denominator(e->plo[i]), NULL) + qbits;
            uint64_t h = bits_sum(tr_bigq_numerator(e->phi[i]), tr_bigq_denominator(e->phi[i]), NULL) + qbits;
            uint64_t q = bits_sum(e->pts[i].q[0], e->pts[i].q[1], e->pts[i].q[2]) + 2 * qbits;
            b = l > h ? l : h;
            if (q > b) b = q;
        }
        if (b > B) B = b;
    }
    if (mode == TR_ENV_ALL) {
        uint64_t md = 0, mc = 0;
        for (uint64_t k = 0; k < e->nclasses; k++) {
            if (bits_of(e->cls[k].d) > md) md = bits_of(e->cls[k].d);
            uint64_t s = bits_sum(e->cls[k].P[0], e->cls[k].P[1], e->cls[k].P[2]);
            if (s > mc) mc = s;
        }
        uint64_t b = md + mc + 2 * qbits + 1;
        if (b > B) B = b;
    }
    return B ? B : 1;
}
static uint64_t lookup_bytes(uint64_t k, uint64_t nb, uint64_t nr, uint64_t nc, uint64_t npts, uint64_t limbs, int32_t mode) {
    uint64_t b = tr_budget_add(sizeof(tr_env_lookup), tr_budget_mul(k, sizeof(tr_gfp) + 16));
    b = tr_budget_add(b, tr_budget_mul(tr_budget_mul(k, k), 8));
    b = tr_budget_add(b, tr_budget_mul(nb, 4 + 1 + 8 + 8));
    b = tr_budget_add(b, tr_budget_mul(tr_budget_mul(k, nb), 8 * TR_ENV_SLOTS));
    b = tr_budget_add(b, tr_budget_add(8, tr_budget_mul(nr, 16)));
    if (mode == TR_ENV_ALL) b = tr_budget_add(b, tr_budget_add(tr_budget_mul(tr_budget_mul(k, nc), 32), tr_budget_mul(nc, 8)));
    return tr_budget_add(b, tr_budget_add(npts, tr_budget_mul(tr_budget_mul(npts, limbs), 16)));
}
static int32_t lookup_need(const tr_env *e, const qsrc *q, int32_t mode, int32_t target, tr_math_need *need) {
    const uint64_t npoints = q->n;
    if (!env_valid(e) || !need || (mode != TR_ENV_SEARCH && mode != TR_ENV_ALL) || (target != 0 && target != 1))
        return TR_MATH_INVALID;
    uint32_t limbs = 1;
    uint64_t qbits = 1, k = 0;
    int32_t rc = query_sizes(q, &limbs, &qbits);
    if (rc) return rc;
    if (e->kind == TR_ENV_FINITE) {
        k = (job_bits(e, mode, qbits) + 2 + 60) / 61;
        if (k > TR_ENV_PRIMES_MAX) return TR_MATH_OVERFLOW;   /* the multimodular representation cannot hold it */
    }
    need_zero(need);
    need->need[TR_BUDGET_OUTPUT] = npoints;
    uint64_t bytes = lookup_bytes(k, e->npoints, e->nregions, e->nclasses, npoints, limbs, mode);
    if (target == 1) {
        tr_env_view v;
        memset(&v, 0, sizeof v);
        v.nprimes = (uint32_t)k; v.nbreaks = e->npoints; v.nregions = e->nregions; v.nclasses = e->nclasses;
        v.npoints = npoints; v.limbs = limbs; v.mode = mode;
        bytes = tr_budget_add(bytes, tr_env_view_device_bytes(&v));
    }
    need->need[TR_BUDGET_BYTES] = bytes;
    return TR_MATH_OK;
}
/* the residue of a big integer modulo every prime: out[j * stride] */
static void residues(const tr_bigint *x, const tr_gfp *F, uint32_t k, uint64_t *out, uint64_t stride) {
    for (uint32_t j = 0; j < k; j++) { uint64_t r = 0; tr_bigint_mod_u64(x, F[j].p, &r); out[(uint64_t)j * stride] = r; }
}
static void residues_neg(const tr_bigint *x, const tr_gfp *F, uint32_t k, uint64_t *out, uint64_t stride) {
    for (uint32_t j = 0; j < k; j++) { uint64_t r = 0; tr_bigint_mod_u64(x, F[j].p, &r); out[(uint64_t)j * stride] = r ? F[j].p - r : 0; }
}
int32_t tr_env_lookup_need(const tr_env *e, uint64_t npoints, const tr_bigq *const *points, int32_t mode,
                           int32_t target, tr_math_need *need) {
    qsrc q = {npoints, points, NULL, NULL};
    if (npoints && !points) return TR_MATH_INVALID;
    return lookup_need(e, &q, mode, target, need);
}
int32_t tr_env_lookup_need_q64(const tr_env *e, uint64_t npoints, const int64_t *num, const int64_t *den, int32_t mode,
                               int32_t target, tr_math_need *need) {
    qsrc q = {npoints, NULL, num, den};
    return lookup_need(e, &q, mode, target, need);
}
static tr_env_lookup *lookup_create(const tr_env *e, const qsrc *q, int32_t mode, const tr_math_budget *budget,
                                    tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    const uint64_t npoints = q->n;
    tr_math_need need;
    int32_t rc = lookup_need(e, q, mode, 0, &need);
    tr_math_budget dflt;
    if (!rc) rc = tr_budget_check(budget_or(budget, &dflt), &need, refusal);
    if (rc) { set_error(error, rc); return NULL; }
    budget = budget_or(budget, &dflt);
    uint32_t limbs = 1;
    uint64_t qbits = 1;
    query_sizes(q, &limbs, &qbits);
    tr_env_lookup *l = zalloc(sizeof *l);
    if (!l) { set_error(error, TR_MATH_MEMORY); return NULL; }
    tr_env_view *v = &l->view;
    v->npoints = npoints; v->nbreaks = e->npoints; v->nregions = e->nregions; v->nclasses = e->nclasses;
    v->limbs = limbs; v->mode = mode; v->kind = e->kind; v->lo_bounded = e->lo_bounded; v->hi_bounded = e->hi_bounded;
    v->tie = e->tie;
    l->qsign = zalloc(npoints); l->qnum = zalloc(tr_budget_mul(tr_budget_mul(npoints, limbs), 8));
    l->qden = zalloc(tr_budget_mul(tr_budget_mul(npoints, limbs), 8));
    if (!l->qsign || !l->qnum || !l->qden) { rc = TR_MATH_MEMORY; goto fail; }
    if ((rc = query_fill(q, limbs, l->qsign, l->qnum, l->qden))) goto fail;
    v->qsign = l->qsign; v->qnum = l->qnum; v->qden = l->qden;
    if (e->kind == TR_ENV_EMPTY) goto done;
    l->bound_bits = job_bits(e, mode, qbits);
    const uint32_t k = (uint32_t)((l->bound_bits + 2 + 60) / 61);
    v->nprimes = k;
    l->fields = zalloc((uint64_t)k * sizeof(tr_gfp)); l->garner = zalloc((uint64_t)k * k * 8); l->half = zalloc((uint64_t)k * 8);
    if (!l->fields || !l->garner || !l->half) { rc = TR_MATH_MEMORY; goto fail; }
    for (uint32_t j = 0; j < k; j++) {
        uint64_t p = 0;
        if (tr_inv_prime((int32_t)j, &p) || tr_gfp_field(p, &l->fields[j])) { rc = TR_MATH_INVALID; goto fail; }
    }
    for (uint32_t j = 0; j < k; j++)
        for (uint32_t t = 0; t < j; t++) l->garner[(uint64_t)j * k + t] = tr_gfp_inv_hd(&l->fields[j], l->fields[t].p % l->fields[j].p);
    {   /* the mixed-radix digits of (M - 1) / 2, M = prod p */
        int32_t er = 0;
        tr_bigint *M = tr_bigint_from_i64(1, budget, refusal, &er);
        for (uint32_t j = 0; j < k && M; j++) {
            tr_bigint *pj = tr_bigint_from_limbs(1, &l->fields[j].p, 1, budget, refusal, &er), *n = NULL;
            if (pj) n = tr_bigint_mul(M, pj, budget, refusal, &er);
            tr_bigint_free(pj); tr_bigint_free(M); M = n;
        }
        tr_bigint *H = M ? tr_bigint_shr(M, 1, budget, refusal, &er) : NULL;      /* (M - 1) / 2: M is odd */
        tr_bigint_free(M);
        for (uint32_t j = 0; j < k && H; j++) {
            tr_bigint *pj = tr_bigint_from_limbs(1, &l->fields[j].p, 1, budget, refusal, &er), *q = NULL, *r = NULL;
            if (pj && !tr_bigint_divmod(H, pj, TR_DIV_FLOOR, &q, &r, budget, refusal)) {
                uint64_t cnt = 0;
                int32_t sg = 0;
                if (tr_bigint_export(r, &sg, &l->half[j], 1, &cnt)) l->half[j] = 0;   /* r < p_j: one limb */
                tr_bigint_free(H); H = q;
            } else { tr_bigint_free(H); H = NULL; }
            tr_bigint_free(pj); tr_bigint_free(r);
        }
        if (!H) { rc = er ? er : TR_MATH_MEMORY; goto fail; }
        tr_bigint_free(H);
    }
    v->fields = l->fields; v->garner = l->garner; v->half = l->half;
    /* the points' comparison polynomials */
    const uint64_t nb = e->npoints, stride = nb * TR_ENV_SLOTS;
    l->bkind = zalloc(nb * 4); l->bsign = zalloc(nb); l->bres = zalloc(tr_budget_mul(tr_budget_mul(k, nb), 8 * TR_ENV_SLOTS));
    l->point_region = zalloc(nb * 8); l->piece_region = zalloc((nb + 1) * 8);
    l->rwinner = zalloc(e->nregions * 8); l->rclasses = zalloc(e->nregions * 8);
    if (!l->bkind || !l->bsign || !l->bres || !l->point_region || !l->piece_region || !l->rwinner || !l->rclasses) {
        rc = TR_MATH_MEMORY; goto fail;
    }
    for (uint64_t i = 0; i < nb; i++) {
        uint64_t *base = l->bres + i * TR_ENV_SLOTS;
        if (e->pts[i].kind == P_RAT) {                         /* t - n / d: (-n) v + d u */
            l->bkind[i] = TR_ENV_RATIONAL;
            residues_neg(e->pts[i].n, l->fields, k, base, stride);
            residues(e->pts[i].d, l->fields, k, base + 1, stride);
        } else {
            l->bkind[i] = TR_ENV_ALGEBRAIC; l->bsign[i] = (int8_t)e->psign[i];
            residues_neg(tr_bigq_numerator(e->plo[i]), l->fields, k, base, stride);
            residues(tr_bigq_denominator(e->plo[i]), l->fields, k, base + 1, stride);
            residues_neg(tr_bigq_numerator(e->phi[i]), l->fields, k, base + 2, stride);
            residues(tr_bigq_denominator(e->phi[i]), l->fields, k, base + 3, stride);
            for (int q = 0; q < 3; q++) residues(e->pts[i].q[q], l->fields, k, base + 4 + q, stride);
        }
        l->point_region[i] = e->pregion[i];
    }
    for (uint64_t i = 0; i <= nb; i++) {                       /* the piece before point i / after the last */
        uint64_t r;
        if (i == 0) r = e->lo_bounded ? NONE : 0;
        else if (i == nb) r = e->hi_bounded ? NONE : e->pregion[nb - 1] + 1;
        else r = e->pregion[i - 1] + 1;
        if (r != NONE && (r >= e->nregions || e->regs[r].kind != TR_ENV_PIECE)) r = NONE;
        l->piece_region[i] = r;
    }
    for (uint64_t r = 0; r < e->nregions; r++) { l->rwinner[r] = e->rwinner[r]; l->rclasses[r] = e->regs[r].ncls; }
    v->bkind = l->bkind; v->bsign = l->bsign; v->bres = l->bres; v->point_region = l->point_region;
    v->piece_region = l->piece_region; v->rwinner = l->rwinner; v->rclasses = l->rclasses;
    if (mode == TR_ENV_ALL) {
        const uint64_t nc = e->nclasses, cs = nc * 4;
        l->cres = zalloc(tr_budget_mul(tr_budget_mul(k, nc), 32)); l->cwinner = zalloc(nc * 8);
        if (!l->cres || !l->cwinner) { rc = TR_MATH_MEMORY; goto fail; }
        for (uint64_t c = 0; c < nc; c++) {
            for (int q = 0; q < 3; q++) residues(e->cls[c].P[q], l->fields, k, l->cres + c * 4 + q, cs);
            residues(e->cls[c].d, l->fields, k, l->cres + c * 4 + 3, cs);
            l->cwinner[c] = e->cls[c].winner;
        }
        v->cres = l->cres; v->cwinner = l->cwinner;
    }
done:
    l->bytes = need.need[TR_BUDGET_BYTES];
    l->magic = LOOKUP_MAGIC;
    set_error(error, TR_MATH_OK);
    return l;
fail:
    l->magic = LOOKUP_MAGIC;
    tr_env_lookup_free(l);
    set_error(error, rc);
    return NULL;
}
tr_env_lookup *tr_env_lookup_create(const tr_env *e, uint64_t npoints, const tr_bigq *const *points, int32_t mode,
                                    const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error) {
    qsrc q = {npoints, points, NULL, NULL};
    if (npoints && !points) { set_error(error, TR_MATH_INVALID); return NULL; }
    return lookup_create(e, &q, mode, budget, refusal, error);
}
tr_env_lookup *tr_env_lookup_create_q64(const tr_env *e, uint64_t npoints, const int64_t *num, const int64_t *den,
                                        int32_t mode, const tr_math_budget *budget, tr_math_refusal *refusal,
                                        int32_t *error) {
    qsrc q = {npoints, NULL, num, den};
    return lookup_create(e, &q, mode, budget, refusal, error);
}
int32_t tr_env_lookup_get_view(const tr_env_lookup *l, tr_env_view *out) {
    if (!lookup_valid(l) || !out) return TR_MATH_INVALID;
    *out = l->view;
    return TR_MATH_OK;
}
uint64_t tr_env_lookup_bound_bits(const tr_env_lookup *l) { return lookup_valid(l) ? l->bound_bits : 0; }
int32_t tr_env_lookup_eval(const tr_env_view *v, tr_env_answer *out, const tr_math_budget *budget, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!tr_env_view_valid(v) || (v->npoints && !out)) return TR_MATH_INVALID;
    tr_math_budget dflt;
    int32_t rc = tr_budget_check_one(budget_or(budget, &dflt), TR_BUDGET_OUTPUT, v->npoints, refusal);
    if (rc) return rc;
    for (uint64_t i = 0; i < v->npoints; i++) tr_env_answer_one(v, i, &out[i]);
    return TR_MATH_OK;
}
