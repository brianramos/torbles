/* UNC-002: explicit approximations with certified error records; see math_approx.h.
 * SPDX-License-Identifier: Apache-2.0 */
#include "math_approx.h"
#include "mathematics.h"
#include "math_domain.h"
#include "math_handles.h"
#include <stdlib.h>
#include <string.h>

TR_MATH_ABI(tr_approx_info, TR_APPROX_ABI);           /* the ctypes boundary of UNC-002 (math_handles.h) */
TR_MATH_ABI(tr_approx_term, TR_APPROX_ABI);
TR_MATH_ABI(tr_approx_eval_view, TR_APPROX_ABI);
TR_MATH_ABI(tr_approx_group, TR_APPROX_ABI);
TR_MATH_ABI(tr_approx_eval_stats, TR_APPROX_ABI);
TR_MATH_ABI(tr_approx_filter_stats, TR_APPROX_ABI);

_Static_assert(sizeof(tr_approx_info) == 160, "tr_approx_info ABI");
_Static_assert(sizeof(tr_approx_term) == 24, "tr_approx_term ABI");
_Static_assert(sizeof(tr_approx_eval_view) == 104, "tr_approx_eval_view ABI");
_Static_assert(sizeof(tr_approx_group) == 64, "tr_approx_group ABI");
_Static_assert(sizeof(tr_approx_eval_stats) == 56, "tr_approx_eval_stats ABI");
_Static_assert(sizeof(tr_approx_filter_stats) == 128, "tr_approx_filter_stats ABI");

#define APPROX_MAGIC UINT32_C(0x58525041)
#define WORK (TR_APPROX_PREC + 32)          /* significant bits of intermediate enclosures */
#define QBYTES 160                           /* the budgeted bytes of one exact rational's handles */
#define SERIES_CAP 256                       /* the internal degree of a product or composition before truncation */

struct tr_approx_s {
    uint32_t magic;
    int32_t kind, function, basis, method, norm, label, provenance, operation;
    uint32_t assumptions;
    int32_t degree, taylor, quantized, has_budget, exact, has_relative;
    int32_t rows, cols, rank, probes, nterms, nkept;
    uint64_t seed, id, origins[2], bytes;
    tr_bigq *a, *b, *mid, *rad;              /* the interval (SERIES, FLOAT) */
    tr_bigq **coef;                          /* degree + 1: SERIES in its basis; FLOAT the exact monomial-x rounded ones */
    double *dcoef;                           /* FLOAT: binary64 coefficients */
    tr_math_interval *civ;                   /* the evaluation view's coefficients */
    tr_bigq *error, *budget, *lipschitz, *derivative, *range_lo, *range_hi, *relative, *l2;
    tr_bigq *frange_lo, *frange_hi;          /* a certified range of the exact base function (monotonicity), or NULL */
    int32_t term_kind[TR_APPROX_MAX_TERMS], term_scaled[TR_APPROX_MAX_TERMS];
    uint64_t term_origin[TR_APPROX_MAX_TERMS];
    tr_bigq *term_q[TR_APPROX_MAX_TERMS];
    int64_t *A;                              /* LOWRANK: the input, rows x cols */
    tr_bigq **entry;                         /* LOWRANK: the approximant, rows x cols */
    tr_bigq *frob2, *frob, *spectral, *maxentry, *failure;
    tr_bigq *probe[TR_APPROX_MAX_PROBES];
    int32_t *prow, *pcol, *kept;             /* CROSS pivots; SKETCH kept columns of Y */
    int8_t *omega;                           /* SKETCH: cols x rank */
    tr_approx_eval_view view;
};

static void set_error(int32_t *error, int32_t code) { if (error) *error = code; }
static const tr_math_budget *budget_or(const tr_math_budget *b, tr_math_budget *dflt) {
    if (b) return b;
    tr_budget_default(dflt);
    return dflt;
}
static void need_zero(tr_math_need *n) { for (int i = 0; i < TR_BUDGET_RESOURCES; i++) n->need[i] = 0; }
static int approx_valid(const tr_approx *x) { return x && x->magic == APPROX_MAGIC; }

/* ------------------------------------------------------------------ exact rationals with a sticky error */
typedef struct { const tr_math_budget *b; tr_math_refusal *r; int32_t err; } qc;
static void qfail(qc *c, int32_t e) { if (!c->err) c->err = e ? e : TR_MATH_MEMORY; }
static tr_bigq *qwrap(qc *c, tr_bigq *x, int32_t e) { if (!x) qfail(c, e); return x; }
static tr_bigint *bwrap(qc *c, tr_bigint *x, int32_t e) { if (!x) qfail(c, e); return x; }
static void qset(tr_bigq **dst, tr_bigq *v) { tr_bigq_free(*dst); *dst = v; }
static void bset(tr_bigint **dst, tr_bigint *v) { tr_bigint_free(*dst); *dst = v; }

static tr_bigint *bint(qc *c, int64_t v) {
    if (c->err) return NULL;
    int32_t e = 0;
    return bwrap(c, tr_bigint_from_i64(v, c->b, c->r, &e), e);
}
static tr_bigint *bi128(qc *c, __int128 v) {
    if (c->err) return NULL;
    int32_t e = 0;
    return bwrap(c, tr_bigint_from_i128(tr_i128_set(v), c->b, c->r, &e), e);
}
#define BBIN(name, fn) static tr_bigint *name(qc *c, const tr_bigint *x, const tr_bigint *y) { \
    if (c->err) return NULL;                                                                    \
    if (!x || !y) { qfail(c, TR_MATH_MEMORY); return NULL; }                                    \
    int32_t e = 0; return bwrap(c, fn(x, y, c->b, c->r, &e), e); }
BBIN(badd, tr_bigint_add)
BBIN(bsub, tr_bigint_sub)
BBIN(bmul, tr_bigint_mul)
BBIN(bdivexact, tr_bigint_divexact)
static tr_bigint *bdup(qc *c, const tr_bigint *x) {
    if (c->err) return NULL;
    if (!x) { qfail(c, TR_MATH_MEMORY); return NULL; }
    int32_t e = 0;
    return bwrap(c, tr_bigint_copy(x, c->b, c->r, &e), e);
}
static tr_bigint *babs(qc *c, const tr_bigint *x) {
    if (c->err) return NULL;
    if (!x) { qfail(c, TR_MATH_MEMORY); return NULL; }
    int32_t e = 0;
    return bwrap(c, tr_bigint_abs(x, c->b, c->r, &e), e);
}
static tr_bigint *bshl(qc *c, const tr_bigint *x, uint64_t k) {
    if (c->err) return NULL;
    if (!x) { qfail(c, TR_MATH_MEMORY); return NULL; }
    int32_t e = 0;
    return bwrap(c, tr_bigint_shl(x, k, c->b, c->r, &e), e);
}
static tr_bigint *bisqrt(qc *c, const tr_bigint *x) {
    if (c->err) return NULL;
    if (!x) { qfail(c, TR_MATH_MEMORY); return NULL; }
    int32_t e = 0;
    return bwrap(c, tr_bigint_isqrt(x, c->b, c->r, &e), e);
}
/* floor(x / y) for y > 0 */
static tr_bigint *bfloordiv(qc *c, const tr_bigint *x, const tr_bigint *y) {
    if (c->err) return NULL;
    if (!x || !y) { qfail(c, TR_MATH_MEMORY); return NULL; }
    tr_bigint *q = NULL;
    int32_t e = tr_bigint_divmod(x, y, TR_DIV_FLOOR, &q, NULL, c->b, c->r);
    if (e) { qfail(c, e); return NULL; }
    return q;
}
/* |x| <= |y| comparison of magnitudes */
static int bcmpabs(qc *c, const tr_bigint *x, const tr_bigint *y) {
    tr_bigint *ax = babs(c, x), *ay = babs(c, y);
    int r = c->err ? 0 : tr_bigint_cmp(ax, ay);
    tr_bigint_free(ax); tr_bigint_free(ay);
    return r;
}

static tr_bigq *qint(qc *c, int64_t v) {
    if (c->err) return NULL;
    int32_t e = 0;
    tr_num_rational q; q.numerator = v; q.denominator = 1;
    return qwrap(c, tr_bigq_from_q64(q, c->b, c->r, &e), e);
}
static tr_bigq *qfrac(qc *c, int64_t n, int64_t d) {
    if (c->err) return NULL;
    tr_num_rational q;
    if (!tr_q_make(n, d, &q)) { qfail(c, TR_MATH_OVERFLOW); return NULL; }
    int32_t e = 0;
    return qwrap(c, tr_bigq_from_q64(q, c->b, c->r, &e), e);
}
static tr_bigq *qbig(qc *c, const tr_bigint *n, const tr_bigint *d) {
    if (c->err) return NULL;
    if (!n || !d) { qfail(c, TR_MATH_MEMORY); return NULL; }
    int32_t e = 0;
    return qwrap(c, tr_bigq_from_bigints(n, d, c->b, c->r, &e), e);
}
static tr_bigq *qbigint(qc *c, const tr_bigint *n) {
    if (c->err) return NULL;
    if (!n) { qfail(c, TR_MATH_MEMORY); return NULL; }
    int32_t e = 0;
    return qwrap(c, tr_bigq_from_bigint(n, c->b, c->r, &e), e);
}
#define QBIN(name, fn) static tr_bigq *name(qc *c, const tr_bigq *x, const tr_bigq *y) {          \
    if (c->err) return NULL;                                                                    \
    if (!x || !y) { qfail(c, TR_MATH_MEMORY); return NULL; }                                    \
    int32_t e = 0; return qwrap(c, fn(x, y, c->b, c->r, &e), e); }
QBIN(qadd, tr_bigq_add)
QBIN(qsub, tr_bigq_sub)
QBIN(qmul, tr_bigq_mul)
QBIN(qdiv, tr_bigq_div)
static tr_bigq *qcopy(qc *c, const tr_bigq *x) {
    if (c->err) return NULL;
    if (!x) { qfail(c, TR_MATH_MEMORY); return NULL; }
    int32_t e = 0;
    return qwrap(c, tr_bigq_copy(x, c->b, c->r, &e), e);
}
static tr_bigq *qneg(qc *c, const tr_bigq *x) {
    if (c->err) return NULL;
    if (!x) { qfail(c, TR_MATH_MEMORY); return NULL; }
    int32_t e = 0;
    return qwrap(c, tr_bigq_neg(x, c->b, c->r, &e), e);
}
static tr_bigq *qpow(qc *c, const tr_bigq *x, int64_t k) {
    if (c->err) return NULL;
    if (!x) { qfail(c, TR_MATH_MEMORY); return NULL; }
    int32_t e = 0;
    return qwrap(c, tr_bigq_pow(x, k, c->b, c->r, &e), e);
}
static int qsgn(const tr_bigq *x) { return x ? tr_bigq_sign(x) : 0; }
static int qcmp(qc *c, const tr_bigq *x, const tr_bigq *y) { return c->err || !x || !y ? 0 : tr_bigq_cmp(x, y); }
static tr_bigq *qabs(qc *c, const tr_bigq *x) { return qsgn(x) < 0 ? qneg(c, x) : qcopy(c, x); }
static tr_bigq *qmulint(qc *c, const tr_bigq *x, int64_t k) {
    tr_bigq *t = qint(c, k), *r = qmul(c, x, t);
    tr_bigq_free(t);
    return r;
}
static tr_bigq *qdivint(qc *c, const tr_bigq *x, int64_t k) {
    tr_bigq *t = qint(c, k), *r = qdiv(c, x, t);
    tr_bigq_free(t);
    return r;
}
static tr_bigq *qmax(qc *c, const tr_bigq *x, const tr_bigq *y) { return qcmp(c, x, y) >= 0 ? qcopy(c, x) : qcopy(c, y); }
static tr_bigq *qmin(qc *c, const tr_bigq *x, const tr_bigq *y) { return qcmp(c, x, y) <= 0 ? qcopy(c, x) : qcopy(c, y); }
/* 2^k for k in Z */
static tr_bigq *q2pow(qc *c, int64_t k) {
    tr_bigint *one = bint(c, 1), *p = bshl(c, one, (uint64_t)(k < 0 ? -k : k));
    tr_bigq *r = k >= 0 ? qbig(c, p, one) : qbig(c, one, p);
    tr_bigint_free(one); tr_bigint_free(p);
    return r;
}
/* m / 2^s for a big integer m */
static tr_bigq *qdyadic(qc *c, const tr_bigint *m, int64_t s) {
    tr_bigint *one = bint(c, 1);
    tr_bigint *p = bshl(c, one, (uint64_t)(s < 0 ? -s : s));
    tr_bigq *r = NULL;
    if (s >= 0) r = qbig(c, m, p);
    else { tr_bigint *n = bmul(c, m, p); r = qbig(c, n, one); tr_bigint_free(n); }
    tr_bigint_free(one); tr_bigint_free(p);
    return r;
}
/* x rounded to a multiple of 2^-s: dir < 0 down, > 0 up, 0 to nearest (ties up) */
static tr_bigq *qgrid(qc *c, const tr_bigq *x, int64_t s, int dir) {
    if (c->err) return NULL;
    if (!x) { qfail(c, TR_MATH_MEMORY); return NULL; }
    const tr_bigint *n = tr_bigq_numerator(x), *d = tr_bigq_denominator(x);
    tr_bigint *num = s > 0 ? bshl(c, n, (uint64_t)s) : bdup(c, n);
    tr_bigint *den = s < 0 ? bshl(c, d, (uint64_t)-s) : bdup(c, d);
    tr_bigint *m = NULL;
    if (dir == 0) {                                     /* floor((2 num + den) / (2 den)) */
        tr_bigint *n2 = bshl(c, num, 1), *d2 = bshl(c, den, 1), *t = badd(c, n2, den);
        m = bfloordiv(c, t, d2);
        tr_bigint_free(n2); tr_bigint_free(d2); tr_bigint_free(t);
    } else if (dir < 0) m = bfloordiv(c, num, den);
    else {                                              /* ceil = floor((num + den - 1) / den) */
        tr_bigint *one = bint(c, 1), *t = badd(c, num, den), *t2 = bsub(c, t, one);
        m = bfloordiv(c, t2, den);
        tr_bigint_free(one); tr_bigint_free(t); tr_bigint_free(t2);
    }
    tr_bigq *r = qdyadic(c, m, s);
    tr_bigint_free(num); tr_bigint_free(den); tr_bigint_free(m);
    return r;
}
/* x rounded to `bits` significant bits (dir as qgrid); 0 stays 0 */
static tr_bigq *qround(qc *c, const tr_bigq *x, int32_t bits, int dir) {
    if (c->err) return NULL;
    if (!x) { qfail(c, TR_MATH_MEMORY); return NULL; }
    if (!qsgn(x)) return qcopy(c, x);
    int64_t e = (int64_t)tr_bigint_bit_length(tr_bigq_numerator(x)) - (int64_t)tr_bigint_bit_length(tr_bigq_denominator(x));
    return qgrid(c, x, (int64_t)bits - e, dir);
}
/* the floor / ceiling of sqrt(x) at 2^-w resolution: lo = isqrt(n d 4^w) / (d 2^w), hi = lo + 1 / (d 2^w) */
static tr_bigq *qsqrt_dir(qc *c, const tr_bigq *x, int32_t w, int up) {
    if (c->err) return NULL;
    if (!x || qsgn(x) < 0) { qfail(c, TR_MATH_DOMAIN_ERROR); return NULL; }
    const tr_bigint *n = tr_bigq_numerator(x), *d = tr_bigq_denominator(x);
    tr_bigint *t = bmul(c, n, d), *t2 = bshl(c, t, (uint64_t)(2 * w)), *s = bisqrt(c, t2);
    if (up && !c->err) {                               /* s + 1 unless s^2 is the radicand (an exact root) */
        tr_bigint *sq = bmul(c, s, s);
        if (!c->err && tr_bigint_cmp(sq, t2) != 0) { tr_bigint *one = bint(c, 1); bset(&s, badd(c, s, one)); tr_bigint_free(one); }
        tr_bigint_free(sq);
    }
    tr_bigint *den = bshl(c, d, (uint64_t)w);
    tr_bigq *r = qbig(c, s, den);
    tr_bigint_free(t); tr_bigint_free(t2); tr_bigint_free(s); tr_bigint_free(den);
    return r;
}
static double q_hi(const tr_bigq *x) {
    tr_math_interval iv;
    if (!x || tr_bigq_enclose(x, &iv)) return INFINITY;
    return iv.hi;
}
static double q_lo(const tr_bigq *x) {
    tr_math_interval iv;
    if (!x || tr_bigq_enclose(x, &iv)) return -INFINITY;
    return iv.lo;
}
static tr_bigq **qvec(qc *c, int32_t n) {
    if (c->err) return NULL;
    tr_bigq **v = calloc((size_t)(n > 0 ? n : 1), sizeof *v);
    if (!v) { qfail(c, TR_MATH_MEMORY); return NULL; }
    for (int32_t i = 0; i < n; i++) v[i] = qint(c, 0);
    return v;
}
static void qvec_free(tr_bigq **v, int32_t n) {
    if (!v) return;
    for (int32_t i = 0; i < n; i++) tr_bigq_free(v[i]);
    free(v);
}
/* acc += x */
static void qacc(qc *c, tr_bigq **acc, const tr_bigq *x) { qset(acc, qadd(c, *acc, x)); }
static void qaccabs(qc *c, tr_bigq **acc, const tr_bigq *x) {
    tr_bigq *t = qabs(c, x);
    qacc(c, acc, t);
    tr_bigq_free(t);
}

/* ------------------------------------------------------------------ dyadic enclosures [lo, hi] */
typedef struct { tr_bigq *lo, *hi; } qiv;
static void qiv_free(qiv *v) { tr_bigq_free(v->lo); tr_bigq_free(v->hi); v->lo = v->hi = NULL; }
static qiv qiv_point(qc *c, const tr_bigq *x) { qiv r; r.lo = qcopy(c, x); r.hi = qcopy(c, x); return r; }
static qiv qiv_copy(qc *c, qiv x) { qiv r; r.lo = qcopy(c, x.lo); r.hi = qcopy(c, x.hi); return r; }
static void qiv_round(qc *c, qiv *v) { qset(&v->lo, qround(c, v->lo, WORK, -1)); qset(&v->hi, qround(c, v->hi, WORK, 1)); }
static void qiv_set(qiv *dst, qiv v) { qiv_free(dst); *dst = v; }
static qiv qiv_add(qc *c, qiv x, qiv y) { qiv r; r.lo = qadd(c, x.lo, y.lo); r.hi = qadd(c, x.hi, y.hi); qiv_round(c, &r); return r; }
static qiv qiv_neg(qc *c, qiv x) { qiv r; r.lo = qneg(c, x.hi); r.hi = qneg(c, x.lo); return r; }
static qiv qiv_mul(qc *c, qiv x, qiv y) {
    tr_bigq *p[4] = {qmul(c, x.lo, y.lo), qmul(c, x.lo, y.hi), qmul(c, x.hi, y.lo), qmul(c, x.hi, y.hi)};
    int lo = 0, hi = 0;
    for (int i = 1; i < 4; i++) { if (qcmp(c, p[i], p[lo]) < 0) lo = i; if (qcmp(c, p[i], p[hi]) > 0) hi = i; }
    qiv r; r.lo = qcopy(c, p[lo]); r.hi = qcopy(c, p[hi]);
    for (int i = 0; i < 4; i++) tr_bigq_free(p[i]);
    qiv_round(c, &r);
    return r;
}
/* x times an exact rational k */
static qiv qiv_scale(qc *c, qiv x, const tr_bigq *k) {
    qiv r;
    if (qsgn(k) >= 0) { r.lo = qmul(c, x.lo, k); r.hi = qmul(c, x.hi, k); }
    else { r.lo = qmul(c, x.hi, k); r.hi = qmul(c, x.lo, k); }
    qiv_round(c, &r);
    return r;
}
/* x widened by +- e (e >= 0) */
static qiv qiv_widen(qc *c, qiv x, const tr_bigq *e) {
    qiv r; r.lo = qsub(c, x.lo, e); r.hi = qadd(c, x.hi, e); qiv_round(c, &r); return r;
}
static qiv qiv_sqr(qc *c, qiv x) {
    tr_bigq *a = qmul(c, x.lo, x.lo), *b = qmul(c, x.hi, x.hi);
    qiv r;
    if (qsgn(x.lo) >= 0) { r.lo = a; r.hi = b; }
    else if (qsgn(x.hi) <= 0) { r.lo = b; r.hi = a; }
    else { r.lo = qint(c, 0); r.hi = qmax(c, a, b); tr_bigq_free(a); tr_bigq_free(b); }
    qiv_round(c, &r);
    return r;
}
static void qiv_clamp(qc *c, qiv *v, int64_t lo, int64_t hi) {
    tr_bigq *l = qint(c, lo), *h = qint(c, hi);
    if (qcmp(c, v->lo, l) < 0) qset(&v->lo, qcopy(c, l));
    if (qcmp(c, v->hi, h) > 0) qset(&v->hi, qcopy(c, h));
    tr_bigq_free(l); tr_bigq_free(h);
}
/* |x| / 2^s <= 1/2: the reduction exponent s and r = x / 2^s */
static tr_bigq *reduce_half(qc *c, const tr_bigq *x, int32_t *s) {
    tr_bigq *r = qcopy(c, x), *ar = qabs(c, r), *half = qfrac(c, 1, 2);
    *s = 0;
    while (!c->err && qcmp(c, ar, half) > 0) {
        qset(&r, qdivint(c, r, 2)); qset(&ar, qabs(c, r)); (*s)++;
    }
    tr_bigq_free(ar); tr_bigq_free(half);
    return r;
}
/* |r|^k / k! (r exact) */
static tr_bigq *taylor_tail(qc *c, const tr_bigq *r, int32_t k) {
    tr_bigq *ar = qabs(c, r), *t = qpow(c, ar, k);
    for (int32_t i = 2; i <= k && !c->err; i++) qset(&t, qdivint(c, t, i));
    tr_bigq_free(ar);
    return t;
}
#define SERIES_K 48
/* e^m: r = m / 2^s (|r| <= 1/2), sum_{i <= K} r^i / i! +- 2 |r|^(K+1) / (K+1)!, squared s times */
static qiv qiv_exp(qc *c, const tr_bigq *m) {
    int32_t s;
    tr_bigq *r = reduce_half(c, m, &s), *one = qint(c, 1);
    qiv sum = qiv_point(c, one), term = qiv_point(c, one);
    for (int32_t i = 1; i <= SERIES_K && !c->err; i++) {
        tr_bigq *k = qdivint(c, r, i);
        qiv_set(&term, qiv_scale(c, term, k));
        qiv_set(&sum, qiv_add(c, sum, term));
        tr_bigq_free(k);
    }
    tr_bigq *rem = taylor_tail(c, r, SERIES_K + 1);
    qset(&rem, qmulint(c, rem, 2));
    qiv_set(&sum, qiv_widen(c, sum, rem));
    for (int32_t j = 0; j < s && !c->err; j++) qiv_set(&sum, qiv_sqr(c, sum));   /* e^r > 0: squaring is monotone */
    tr_bigq_free(r); tr_bigq_free(one); tr_bigq_free(rem); qiv_free(&term);
    return sum;
}
/* sin m and cos m: Taylor at r = m / 2^s (remainders |r|^(K+1) / (K+1)!), then s double-angle steps */
static void qiv_sincos(qc *c, const tr_bigq *m, qiv *S, qiv *Cs) {
    int32_t s;
    tr_bigq *r = reduce_half(c, m, &s), *one = qint(c, 1), *zero = qint(c, 0);
    qiv sn = qiv_point(c, zero), cs = qiv_point(c, one), term = qiv_point(c, one);
    for (int32_t i = 1; i <= SERIES_K && !c->err; i++) {
        tr_bigq *k = qdivint(c, r, i);
        qiv_set(&term, qiv_scale(c, term, k));
        tr_bigq_free(k);
        int sign = (i % 4 == 1 || i % 4 == 0) ? 1 : -1;  /* sin: + at 1, - at 3 (mod 4); cos: + at 0, - at 2 */
        qiv t = sign > 0 ? qiv_copy(c, term) : qiv_neg(c, term);
        if (i % 2) qiv_set(&sn, qiv_add(c, sn, t)); else qiv_set(&cs, qiv_add(c, cs, t));
        qiv_free(&t);
    }
    tr_bigq *rem = taylor_tail(c, r, SERIES_K + 1);
    qiv_set(&sn, qiv_widen(c, sn, rem)); qiv_set(&cs, qiv_widen(c, cs, rem));
    for (int32_t j = 0; j < s && !c->err; j++) {    /* sin 2u = 2 sin u cos u, cos 2u = 1 - 2 sin^2 u */
        tr_bigq *two = qint(c, 2);
        qiv p = qiv_mul(c, sn, cs), sq = qiv_sqr(c, sn);
        qiv s2 = qiv_scale(c, p, two);
        tr_bigq *mtwo = qint(c, -2);
        qiv t = qiv_scale(c, sq, mtwo);
        qiv c2; c2.lo = qadd(c, t.lo, one); c2.hi = qadd(c, t.hi, one);
        qiv_set(&sn, s2); qiv_set(&cs, c2);
        qiv_clamp(c, &sn, -1, 1); qiv_clamp(c, &cs, -1, 1);
        qiv_free(&p); qiv_free(&sq); qiv_free(&t);
        tr_bigq_free(two); tr_bigq_free(mtwo);
    }
    qiv_clamp(c, &sn, -1, 1); qiv_clamp(c, &cs, -1, 1);
    *S = sn; *Cs = cs;
    tr_bigq_free(r); tr_bigq_free(one); tr_bigq_free(zero); tr_bigq_free(rem); qiv_free(&term);
}
#define ATANH_K 72
/* 2 atanh(z) for |z| <= 1/3: 2 sum_{i <= K} z^(2i+1) / (2i+1) +- 2 (1/3)^(2K+3) 9 / (8 (2K+3)) */
static qiv qiv_atanh2(qc *c, const tr_bigq *z) {
    qiv term = qiv_point(c, z), sum = qiv_point(c, z);
    qiv zz = qiv_point(c, z);
    qiv z2 = qiv_mul(c, zz, zz);
    for (int32_t i = 1; i <= ATANH_K && !c->err; i++) {
        qiv_set(&term, qiv_mul(c, term, z2));
        tr_bigq *k = qfrac(c, 1, 2 * i + 1);
        qiv t = qiv_scale(c, term, k);
        qiv_set(&sum, qiv_add(c, sum, t));
        qiv_free(&t); tr_bigq_free(k);
    }
    tr_bigq *third = qfrac(c, 1, 3), *rem = qpow(c, third, 2 * ATANH_K + 3);
    qset(&rem, qmulint(c, rem, 18));
    qset(&rem, qdivint(c, rem, 8 * (2 * ATANH_K + 3)));
    qiv_set(&sum, qiv_widen(c, sum, rem));
    tr_bigq *two = qint(c, 2);
    qiv r = qiv_scale(c, sum, two);
    qiv_free(&term); qiv_free(&sum); qiv_free(&zz); qiv_free(&z2);
    tr_bigq_free(third); tr_bigq_free(rem); tr_bigq_free(two);
    return r;
}
/* log m for m > 0: m = 2^e u, u in (1/2, 2), z = (u - 1) / (u + 1) in (-1/3, 1/3), log m = e log 2 + 2 atanh(z) */
static qiv qiv_log(qc *c, const tr_bigq *m) {
    qiv out; out.lo = out.hi = NULL;
    if (c->err) return out;
    if (qsgn(m) <= 0) { qfail(c, TR_MATH_DOMAIN_ERROR); return out; }
    int64_t e = (int64_t)tr_bigint_bit_length(tr_bigq_numerator(m)) - (int64_t)tr_bigint_bit_length(tr_bigq_denominator(m));
    tr_bigq *p = q2pow(c, -e), *u = qmul(c, m, p), *one = qint(c, 1);
    tr_bigq *num = qsub(c, u, one), *den = qadd(c, u, one), *z = qdiv(c, num, den), *third = qfrac(c, 1, 3);
    qiv lu = qiv_atanh2(c, z), l2 = qiv_atanh2(c, third);
    tr_bigq *ek = qint(c, e);
    qiv el = qiv_scale(c, l2, ek);
    out = qiv_add(c, el, lu);
    qiv_free(&lu); qiv_free(&l2); qiv_free(&el);
    tr_bigq_free(p); tr_bigq_free(u); tr_bigq_free(one); tr_bigq_free(num); tr_bigq_free(den); tr_bigq_free(z);
    tr_bigq_free(third); tr_bigq_free(ek);
    return out;
}
static qiv qiv_sqrt(qc *c, const tr_bigq *m) {
    qiv r; r.lo = qsqrt_dir(c, m, WORK, 0); r.hi = qsqrt_dir(c, m, WORK, 1); return r;
}

/* ------------------------------------------------------------------ digests (ids; a pre-filter only) */
static uint64_t mix(uint64_t h, uint64_t w) {
    h ^= w + UINT64_C(0x9E3779B97F4A7C15) + (h << 6) + (h >> 2);
    h *= UINT64_C(0xBF58476D1CE4E5B9);
    return h ^ (h >> 31);
}
static uint64_t mix_bigint(uint64_t h, const tr_bigint *x) {
    uint64_t n = tr_bigint_limb_count(x), buf[16];
    int32_t sign = 0;
    uint64_t cnt = 0;
    h = mix(h, (uint64_t)tr_bigint_sign(x));
    h = mix(h, n);
    if (n <= 16 && !tr_bigint_export(x, &sign, buf, 16, &cnt)) for (uint64_t i = 0; i < cnt; i++) h = mix(h, buf[i]);
    else h = mix(h, tr_bigint_bit_length(x));
    return h;
}
static uint64_t mix_q(uint64_t h, const tr_bigq *q) {
    if (!q) return mix(h, 0);
    return mix_bigint(mix_bigint(h, tr_bigq_numerator(q)), tr_bigq_denominator(q));
}

/* ------------------------------------------------------------------ objects */
static tr_approx *approx_alloc(int32_t kind) {
    tr_approx *x = calloc(1, sizeof *x);
    if (!x) return NULL;
    x->kind = kind; x->label = TR_APPROX_LABEL_CERTIFIED; x->provenance = TR_APPROX_DETERMINISTIC;
    return x;
}
static void approx_release(tr_approx *x) {
    if (!x) return;
    tr_bigq_free(x->a); tr_bigq_free(x->b); tr_bigq_free(x->mid); tr_bigq_free(x->rad);
    qvec_free(x->coef, x->degree + 1);
    free(x->dcoef); free(x->civ);
    tr_bigq_free(x->error); tr_bigq_free(x->budget); tr_bigq_free(x->lipschitz); tr_bigq_free(x->derivative);
    tr_bigq_free(x->range_lo); tr_bigq_free(x->range_hi); tr_bigq_free(x->relative); tr_bigq_free(x->l2);
    tr_bigq_free(x->frange_lo); tr_bigq_free(x->frange_hi);
    for (int32_t i = 0; i < x->nterms; i++) tr_bigq_free(x->term_q[i]);
    free(x->A);
    if (x->entry) qvec_free(x->entry, x->rows * x->cols);
    tr_bigq_free(x->frob2); tr_bigq_free(x->frob); tr_bigq_free(x->spectral); tr_bigq_free(x->maxentry);
    tr_bigq_free(x->failure);
    for (int32_t i = 0; i < TR_APPROX_MAX_PROBES; i++) tr_bigq_free(x->probe[i]);
    free(x->prow); free(x->pcol); free(x->kept); free(x->omega);
    x->magic = 0;
    free(x);
}
void tr_approx_free(tr_approx *x) { if (approx_valid(x)) approx_release(x); }

/* one error term: merged with an equal (kind, origin) term; 0 or TR_MATH_UNSUPPORTED past the representation limit */
static int32_t term_add(qc *c, tr_approx *x, int32_t kind, uint64_t origin, int32_t scaled, const tr_bigq *mag) {
    if (c->err || !mag) return c->err ? c->err : TR_MATH_MEMORY;
    if (!qsgn(mag)) return TR_MATH_OK;                 /* an exact zero contributes nothing */
    for (int32_t i = 0; i < x->nterms; i++)
        if (x->term_kind[i] == kind && x->term_origin[i] == origin) {
            qset(&x->term_q[i], qadd(c, x->term_q[i], mag));
            x->term_scaled[i] |= scaled;
            return c->err;
        }
    if (x->nterms >= TR_APPROX_MAX_TERMS) return TR_MATH_UNSUPPORTED;
    x->term_kind[x->nterms] = kind; x->term_origin[x->nterms] = origin; x->term_scaled[x->nterms] = scaled;
    x->term_q[x->nterms] = qcopy(c, mag);
    x->nterms++;
    return c->err;
}
/* every term of src times factor (NULL: 1) */
static int32_t terms_carry(qc *c, tr_approx *x, const tr_approx *src, const tr_bigq *factor) {
    for (int32_t i = 0; i < src->nterms; i++) {
        tr_bigq *m = factor ? qmul(c, src->term_q[i], factor) : qcopy(c, src->term_q[i]);
        int scaled = src->term_scaled[i];
        if (factor) { int32_t cv = 0; if (tr_bigq_cmp_double(factor, 1.0, &cv) || cv != 0) scaled = 1; }
        int32_t rc = term_add(c, x, src->term_kind[i], src->term_origin[i], scaled, m);
        tr_bigq_free(m);
        if (rc) return rc;
    }
    return c->err;
}
static void error_total(qc *c, tr_approx *x) {
    qset(&x->error, qint(c, 0));
    for (int32_t i = 0; i < x->nterms; i++) qacc(c, &x->error, x->term_q[i]);
    x->exact = !c->err && qsgn(x->error) == 0;
}

/* ------------------------------------------------------------------ basis conversions (exact) */
/* sum m_k t^k (k <= deg) in the Chebyshev basis: u_k = t u_{k-1}, t T_0 = T_1, t T_j = (T_{j+1} + T_{j-1}) / 2 */
static tr_bigq **mono_to_cheb(qc *c, tr_bigq *const *m, int32_t deg) {
    tr_bigq **out = qvec(c, deg + 1), **u = qvec(c, deg + 2), **v = qvec(c, deg + 2);
    if (c->err) { qvec_free(out, deg + 1); qvec_free(u, deg + 2); qvec_free(v, deg + 2); return NULL; }
    qset(&u[0], qint(c, 1));
    qset(&out[0], qcopy(c, m[0]));
    for (int32_t k = 1; k <= deg && !c->err; k++) {
        for (int32_t j = 0; j <= k; j++) qset(&v[j], qint(c, 0));
        for (int32_t j = 0; j < k && !c->err; j++) {
            if (!qsgn(u[j])) continue;
            if (j == 0) { qacc(c, &v[1], u[0]); continue; }
            tr_bigq *h = qdivint(c, u[j], 2);
            qacc(c, &v[j + 1], h); qacc(c, &v[j - 1], h);
            tr_bigq_free(h);
        }
        tr_bigq **t = u; u = v; v = t;
        if (!qsgn(m[k])) continue;
        for (int32_t j = 0; j <= k && !c->err; j++) {
            if (!qsgn(u[j])) continue;
            tr_bigq *p = qmul(c, m[k], u[j]);
            qacc(c, &out[j], p);
            tr_bigq_free(p);
        }
    }
    qvec_free(u, deg + 2); qvec_free(v, deg + 2);
    if (c->err) { qvec_free(out, deg + 1); return NULL; }
    return out;
}
/* sum c_j T_j (j <= deg) in monomials of t: T_{j+1} = 2 t T_j - T_{j-1} */
static tr_bigq **cheb_to_mono(qc *c, tr_bigq *const *cc, int32_t deg) {
    tr_bigq **out = qvec(c, deg + 1), **p0 = qvec(c, deg + 2), **p1 = qvec(c, deg + 2), **p2 = qvec(c, deg + 2);
    if (c->err) { qvec_free(out, deg + 1); qvec_free(p0, deg + 2); qvec_free(p1, deg + 2); qvec_free(p2, deg + 2); return NULL; }
    qset(&p0[0], qint(c, 1));                          /* T_0 */
    qset(&out[0], qcopy(c, cc[0]));
    if (deg >= 1) {
        qset(&p1[1], qint(c, 1));                      /* T_1 */
        qset(&out[1], qcopy(c, cc[1]));
    }
    for (int32_t j = 2; j <= deg && !c->err; j++) {
        for (int32_t i = 0; i <= j; i++) {
            tr_bigq *t = i >= 1 ? qmulint(c, p1[i - 1], 2) : qint(c, 0);
            qset(&p2[i], qsub(c, t, p0[i]));
            tr_bigq_free(t);
        }
        tr_bigq **t = p0; p0 = p1; p1 = p2; p2 = t;
        if (!qsgn(cc[j])) continue;
        for (int32_t i = 0; i <= j; i++) {
            if (!qsgn(p1[i])) continue;
            tr_bigq *q = qmul(c, cc[j], p1[i]);
            qacc(c, &out[i], q);
            tr_bigq_free(q);
        }
    }
    qvec_free(p0, deg + 2); qvec_free(p1, deg + 2); qvec_free(p2, deg + 2);
    if (c->err) { qvec_free(out, deg + 1); return NULL; }
    return out;
}
/* sum m_k t^k with t = alpha x + beta as monomials in x (polynomial Horner) */
static tr_bigq **mono_substitute(qc *c, tr_bigq *const *m, int32_t deg, const tr_bigq *alpha, const tr_bigq *beta) {
    tr_bigq **acc = qvec(c, deg + 1), **nx = qvec(c, deg + 1);
    if (c->err) { qvec_free(acc, deg + 1); qvec_free(nx, deg + 1); return NULL; }
    qset(&acc[0], qcopy(c, m[deg]));
    for (int32_t k = deg - 1; k >= 0 && !c->err; k--) {
        int32_t d = deg - 1 - k;                        /* acc has degree d before this step */
        for (int32_t i = 0; i <= d + 1; i++) {
            tr_bigq *s = i <= d ? qmul(c, acc[i], beta) : qint(c, 0);
            if (i >= 1) { tr_bigq *t = qmul(c, acc[i - 1], alpha); qacc(c, &s, t); tr_bigq_free(t); }
            qset(&nx[i], s);
        }
        qacc(c, &nx[0], m[k]);
        tr_bigq **t = acc; acc = nx; nx = t;
    }
    qvec_free(nx, deg + 1);
    if (c->err) { qvec_free(acc, deg + 1); return NULL; }
    return acc;
}
/* Chebyshev product of a (da) and b (db): degree da + db */
static tr_bigq **cheb_mul(qc *c, tr_bigq *const *x, int32_t dx, tr_bigq *const *y, int32_t dy) {
    tr_bigq **out = qvec(c, dx + dy + 1);
    for (int32_t i = 0; i <= dx && !c->err; i++) {
        if (!qsgn(x[i])) continue;
        for (int32_t j = 0; j <= dy && !c->err; j++) {
            if (!qsgn(y[j])) continue;
            tr_bigq *p = qmul(c, x[i], y[j]), *h = qdivint(c, p, 2);
            qacc(c, &out[i + j], h);
            qacc(c, &out[i > j ? i - j : j - i], h);
            tr_bigq_free(p); tr_bigq_free(h);
        }
    }
    if (c->err) { qvec_free(out, dx + dy + 1); return NULL; }
    return out;
}
/* a series' coefficients in the Chebyshev basis (a copy) */
static tr_bigq **as_cheb(qc *c, const tr_approx *x) {
    if (x->basis == TR_APPROX_CHEBYSHEV) {
        tr_bigq **v = qvec(c, x->degree + 1);
        for (int32_t j = 0; v && j <= x->degree; j++) qset(&v[j], qcopy(c, x->coef[j]));
        return v;
    }
    return mono_to_cheb(c, x->coef, x->degree);
}
/* the exact value of a series at t (Clenshaw / Horner) */
static tr_bigq *series_at(qc *c, const tr_approx *x, const tr_bigq *t) {
    int32_t n = x->degree;
    if (x->basis == TR_APPROX_CHEBYSHEV) {
        tr_bigq *b1 = qint(c, 0), *b2 = qint(c, 0), *t2 = qmulint(c, t, 2);
        for (int32_t k = n; k >= 1 && !c->err; k--) {
            tr_bigq *p = qmul(c, t2, b1), *s = qadd(c, x->coef[k], p), *b0 = qsub(c, s, b2);
            tr_bigq_free(p); tr_bigq_free(s);
            qset(&b2, b1); b1 = b0;
        }
        tr_bigq *p = qmul(c, t, b1), *s = qadd(c, x->coef[0], p), *r = qsub(c, s, b2);
        tr_bigq_free(p); tr_bigq_free(s); tr_bigq_free(b1); tr_bigq_free(b2); tr_bigq_free(t2);
        return r;
    }
    tr_bigq *acc = qcopy(c, x->coef[n]);
    for (int32_t k = n - 1; k >= 0 && !c->err; k--) { qset(&acc, qmul(c, acc, t)); qacc(c, &acc, x->coef[k]); }
    return acc;
}

/* ------------------------------------------------------------------ finishing a series / float */
/* range, relative and L2 bounds, the evaluation view, bytes; then the budget in the declared norm */
static int32_t series_finish(qc *c, tr_approx *x, const tr_bigq *budget_q) {
    error_total(c, x);
    if (c->err) return c->err;
    /* S' = sum_{j >= 1} |c_j| over the polynomial's own basis (|T_j|, |t^j| <= 1 on [-1, 1]); FLOAT: sum |a_j| B^j */
    tr_bigq *sp = qint(c, 0), *c0 = NULL;
    if (x->kind == TR_APPROX_FLOAT) {
        tr_bigq *B = NULL, *aa = qabs(c, x->a), *bb = qabs(c, x->b);
        B = qmax(c, aa, bb);
        tr_bigq *pw = qcopy(c, B);
        for (int32_t j = 1; j <= x->degree && !c->err; j++) {
            tr_bigq *t = qabs(c, x->coef[j]), *u = qmul(c, t, pw);
            qacc(c, &sp, u);
            tr_bigq_free(t); tr_bigq_free(u);
            qset(&pw, qmul(c, pw, B));
        }
        c0 = qcopy(c, x->coef[0]);
        tr_bigq_free(B); tr_bigq_free(aa); tr_bigq_free(bb); tr_bigq_free(pw);
    } else {
        for (int32_t j = 1; j <= x->degree && !c->err; j++) qaccabs(c, &sp, x->coef[j]);
        c0 = qcopy(c, x->coef[0]);
    }
    tr_bigq *w = qadd(c, sp, x->error);
    qset(&x->range_lo, qsub(c, c0, w));
    qset(&x->range_hi, qadd(c, c0, w));
    tr_bigq_free(sp); tr_bigq_free(c0); tr_bigq_free(w);
    /* intersected with the base function's own certified range (both enclose the exact function's values) */
    if (x->frange_lo && qcmp(c, x->frange_lo, x->range_lo) > 0) qset(&x->range_lo, qcopy(c, x->frange_lo));
    if (x->frange_hi && qcmp(c, x->frange_hi, x->range_hi) < 0) qset(&x->range_hi, qcopy(c, x->frange_hi));
    x->has_relative = 0;
    qset(&x->relative, NULL);
    if (!c->err && (qsgn(x->range_lo) > 0 || qsgn(x->range_hi) < 0)) {
        tr_bigq *m = qsgn(x->range_lo) > 0 ? qcopy(c, x->range_lo) : qneg(c, x->range_hi);
        qset(&x->relative, qdiv(c, x->error, m));
        tr_bigq_free(m);
        x->has_relative = 1;
    }
    tr_bigq *len = qsub(c, x->b, x->a), *sq = qsqrt_dir(c, len, 64, 1);
    qset(&x->l2, qmul(c, sq, x->error));
    tr_bigq_free(len); tr_bigq_free(sq);
    x->assumptions |= TR_APPROX_ASSUME_DOMAIN;
    if (c->err) return c->err;
    /* the evaluation view */
    int32_t n = x->degree + 1;
    free(x->civ);
    x->civ = calloc((size_t)n, sizeof *x->civ);
    if (!x->civ) return TR_MATH_MEMORY;
    for (int32_t j = 0; j < n; j++) {
        if (x->kind == TR_APPROX_FLOAT) x->civ[j] = tr_approx_iv(x->dcoef[j], x->dcoef[j]);
        else if (tr_bigq_enclose(x->coef[j], &x->civ[j])) return TR_MATH_INVALID;
    }
    tr_approx_eval_view *v = &x->view;
    memset(v, 0, sizeof *v);
    v->mode = x->kind == TR_APPROX_FLOAT ? TR_APPROX_EVAL_FLOAT : TR_APPROX_EVAL_SERIES;
    v->basis = x->basis; v->ncoef = n; v->coef = x->civ;
    if (tr_bigq_enclose(x->mid, &v->mid) || tr_bigq_enclose(x->rad, &v->rad)) return TR_MATH_INVALID;
    v->dom.lo = q_hi(x->a); v->dom.hi = q_lo(x->b);
    v->range.lo = q_lo(x->range_lo); v->range.hi = q_hi(x->range_hi);
    v->err = q_hi(x->error);
    v->lip = q_hi(x->lipschitz);
    if (x->kind == TR_APPROX_FLOAT) x->assumptions |= TR_APPROX_ASSUME_NO_OVERFLOW;
    x->bytes = sizeof *x + (uint64_t)n * (sizeof(tr_math_interval) + QBYTES + 8) + (uint64_t)(x->nterms + 12) * QBYTES;
    /* the budget, in the declared norm */
    if (budget_q) {
        qset(&x->budget, qcopy(c, budget_q));
        x->has_budget = 1;
        const tr_bigq *val = x->norm == TR_APPROX_NORM_REL ? x->relative : x->norm == TR_APPROX_NORM_L2 ? x->l2 : x->error;
        if (x->norm == TR_APPROX_NORM_REL && !x->has_relative) return TR_MATH_DOMAIN_ERROR;   /* |f| may vanish */
        if (c->err) return c->err;
        if (tr_bigq_cmp(val, budget_q) > 0) return TR_MATH_NOT_CONVERGED;
    }
    return c->err;
}
static tr_approx *approx_done(qc *c, tr_approx *x, int32_t rc, int32_t *error) {
    if (!rc) rc = c->err;
    if (rc) { approx_release(x); set_error(error, rc); return NULL; }
    x->magic = APPROX_MAGIC;
    set_error(error, TR_MATH_OK);
    return x;
}
static int budget_ok(const tr_bigq *b) { return !b || (tr_bigq_is_handle(b) == 1 && tr_bigq_sign(b) >= 0); }
static int q_ok(const tr_bigq *q) { return q && tr_bigq_is_handle(q) == 1; }
/* the size of an endpoint we accept (bit lengths of its parts) */
static int q_small(const tr_bigq *q, uint64_t bits) {
    return tr_bigint_bit_length(tr_bigq_numerator(q)) <= bits && tr_bigint_bit_length(tr_bigq_denominator(q)) <= bits;
}

/* ------------------------------------------------------------------ series of the supported functions */
/* the Taylor coefficient enclosures g_k = f^(k)(m) r^k / k!, k <= N, and the remainder bound, the Lipschitz and
 * derivative bounds of f on [a, b] */
static int32_t taylor_coefficients(qc *c, int32_t f, const tr_bigq *a, const tr_bigq *b, const tr_bigq *m,
                                   const tr_bigq *r, int32_t N, qiv *g, tr_bigq **rem, tr_bigq **lip, tr_bigq **der) {
    tr_bigq *one = qint(c, 1), *pk = qint(c, 1);       /* pk = r^k / k! */
    qiv base[4];
    for (int i = 0; i < 4; i++) base[i].lo = base[i].hi = NULL;
    tr_bigq *ratio = NULL;                             /* r / m (LOG, SQRT, RECIP) */
    if (f == TR_APPROX_EXP) base[0] = qiv_exp(c, m);
    else if (f == TR_APPROX_SIN || f == TR_APPROX_COS) {
        qiv s, co;
        qiv_sincos(c, m, &s, &co);
        qiv ns = qiv_neg(c, s), nc = qiv_neg(c, co);
        if (f == TR_APPROX_SIN) { base[0] = s; base[1] = co; base[2] = ns; base[3] = nc; }
        else { base[0] = co; base[1] = ns; base[2] = nc; base[3] = s; }
    } else if (f == TR_APPROX_LOG) { base[0] = qiv_log(c, m); ratio = qdiv(c, r, m); }
    else if (f == TR_APPROX_SQRT) { base[0] = qiv_sqrt(c, m); ratio = qdiv(c, r, m); }
    else ratio = qdiv(c, r, m);
    tr_bigq *binom = qint(c, 1);                       /* SQRT: binom(1/2, k) */
    for (int32_t k = 0; k <= N && !c->err; k++) {
        if (k > 0) { qset(&pk, qmul(c, pk, r)); qset(&pk, qdivint(c, pk, k)); }
        switch (f) {
        case TR_APPROX_EXP: g[k] = qiv_scale(c, base[0], pk); break;
        case TR_APPROX_SIN: case TR_APPROX_COS: g[k] = qiv_scale(c, base[k % 4], pk); break;
        case TR_APPROX_LOG:
            if (!k) { g[0].lo = qcopy(c, base[0].lo); g[0].hi = qcopy(c, base[0].hi); }
            else {                                     /* (-1)^(k-1) (r/m)^k / k */
                tr_bigq *t = qpow(c, ratio, k);
                qset(&t, qdivint(c, t, (k % 2) ? k : -k));
                g[k] = qiv_point(c, t);
                tr_bigq_free(t);
            }
            break;
        case TR_APPROX_SQRT: {                         /* binom(1/2, k) sqrt(m) (r/m)^k */
            if (k > 0) {
                tr_bigq *h = qfrac(c, 1, 2), *km = qint(c, k - 1), *top = qsub(c, h, km);
                qset(&binom, qmul(c, binom, top)); qset(&binom, qdivint(c, binom, k));
                tr_bigq_free(h); tr_bigq_free(km); tr_bigq_free(top);
            }
            tr_bigq *t = qpow(c, ratio, k);
            qset(&t, qmul(c, t, binom));
            g[k] = qiv_scale(c, base[0], t);
            tr_bigq_free(t);
            break;
        }
        default: {                                     /* RECIP: (-1)^k r^k / m^(k+1) = (-1)^k (r/m)^k / m */
            tr_bigq *t = qpow(c, ratio, k);
            qset(&t, qdiv(c, t, m));
            if (k % 2) qset(&t, qneg(c, t));
            g[k] = qiv_point(c, t);
            tr_bigq_free(t);
            break;
        }
        }
    }
    /* the remainder: Lagrange M r^(N+1) / (N+1)! with M >= sup |f^(N+1)| on [a, b]; for LOG, SQRT, RECIP also the
     * integral form: |f^(N+1)(s)| = K |s|^(alpha - N - 1), |x - s| / |s| <= r / |m| between m and x, so
     * |R_N| <= (K / N!) (r / |m|)^N r max |s|^(alpha - 1) (alpha = 0, 1/2, -1), which converges whenever 0 is
     * outside [a, b]; the smaller bound is used. Lipschitz L >= sup |f'| */
    tr_bigq *rn = taylor_tail(c, r, N + 1);           /* r^(N+1) / (N+1)! (r > 0) */
    tr_bigq *am = NULL, *rm = NULL, *geo = NULL, *geor = NULL;   /* (r/|m|)^N r: LOG, SQRT, RECIP (m != 0) */
    if (f == TR_APPROX_LOG || f == TR_APPROX_SQRT || f == TR_APPROX_RECIP) {
        am = qabs(c, m); rm = qdiv(c, r, am); geo = qpow(c, rm, N); geor = qmul(c, geo, r);
    }
    switch (f) {
    case TR_APPROX_EXP: {
        qiv eb = qiv_exp(c, b);
        *der = qcopy(c, eb.hi); *lip = qcopy(c, eb.hi);
        *rem = qmul(c, eb.hi, rn);
        qiv_free(&eb);
        break;
    }
    case TR_APPROX_SIN: case TR_APPROX_COS:
        *der = qint(c, 1); *lip = qint(c, 1); *rem = qcopy(c, rn);
        break;
    case TR_APPROX_LOG: {                              /* M = N! / a^(N+1): rem = (r/a)^(N+1) / (N+1); K / N! = 1 */
        tr_bigq *ra = qdiv(c, r, a), *t = qpow(c, ra, N + 1);
        *rem = qdivint(c, t, N + 1);
        tr_bigq *ig = qdiv(c, geor, a);
        if (qcmp(c, ig, *rem) < 0) qset(rem, qcopy(c, ig));
        tr_bigq_free(ig);
        tr_bigq *fact = qint(c, 1);
        for (int32_t i = 2; i <= N && !c->err; i++) qset(&fact, qmulint(c, fact, i));
        tr_bigq *ap = qpow(c, a, N + 1);
        *der = qdiv(c, fact, ap);
        *lip = qdiv(c, one, a);
        tr_bigq_free(ra); tr_bigq_free(t); tr_bigq_free(fact); tr_bigq_free(ap);
        break;
    }
    case TR_APPROX_SQRT: {                             /* |binom(1/2, N+1)| (N+1)! sqrt(a) / a^(N+1) */
        tr_bigq *h = qfrac(c, 1, 2), *km = qint(c, N), *top = qsub(c, h, km), *bn = qmul(c, binom, top);
        qset(&bn, qdivint(c, bn, N + 1));
        tr_bigq *abn = qabs(c, bn), *sa = qsqrt_dir(c, a, WORK, 1), *sal = qsqrt_dir(c, a, WORK, 0);
        tr_bigq *ra = qdiv(c, r, a), *t = qpow(c, ra, N + 1);
        qset(&t, qmul(c, t, abn)); *rem = qmul(c, t, sa);
        tr_bigq *ig = qmul(c, geor, abn);              /* K / N! = |binom(1/2, N+1)| (N+1), max s^(-1/2) = 1/sqrt(a) */
        qset(&ig, qmulint(c, ig, N + 1));
        qset(&ig, qdiv(c, ig, sal));
        if (qcmp(c, ig, *rem) < 0) qset(rem, qcopy(c, ig));
        tr_bigq_free(ig);
        tr_bigq *fact = qint(c, 1);
        for (int32_t i = 2; i <= N + 1 && !c->err; i++) qset(&fact, qmulint(c, fact, i));
        tr_bigq *ap = qpow(c, a, N + 1), *d0 = qmul(c, abn, fact);
        qset(&d0, qmul(c, d0, sa)); *der = qdiv(c, d0, ap);
        tr_bigq *two_sal = qmulint(c, sal, 2);
        *lip = qdiv(c, one, two_sal);                  /* 1 / (2 sqrt(a)) <= 1 / (2 sqrt_lo(a)) */
        tr_bigq_free(h); tr_bigq_free(km); tr_bigq_free(top); tr_bigq_free(bn); tr_bigq_free(abn); tr_bigq_free(sa);
        tr_bigq_free(sal); tr_bigq_free(ra); tr_bigq_free(t); tr_bigq_free(fact); tr_bigq_free(ap); tr_bigq_free(d0);
        tr_bigq_free(two_sal);
        break;
    }
    default: {                                         /* RECIP: M = (N+1)! / mu^(N+2): rem = r^(N+1) / mu^(N+2) */
        tr_bigq *aa = qabs(c, a), *ab = qabs(c, b), *mu = qmin(c, aa, ab);
        tr_bigq *t = qpow(c, r, N + 1), *mp = qpow(c, mu, N + 2);
        *rem = qdiv(c, t, mp);
        tr_bigq *mu2 = qmul(c, mu, mu), *ig = qmulint(c, geor, N + 1);   /* K / N! = N + 1, max |s|^-2 = 1/mu^2 */
        qset(&ig, qdiv(c, ig, mu2));
        if (qcmp(c, ig, *rem) < 0) qset(rem, qcopy(c, ig));
        tr_bigq_free(ig); tr_bigq_free(mu2);
        tr_bigq *fact = qint(c, 1);
        for (int32_t i = 2; i <= N + 1 && !c->err; i++) qset(&fact, qmulint(c, fact, i));
        *der = qdiv(c, fact, mp);
        tr_bigq *m2 = qmul(c, mu, mu);
        *lip = qdiv(c, one, m2);
        tr_bigq_free(aa); tr_bigq_free(ab); tr_bigq_free(mu); tr_bigq_free(t); tr_bigq_free(mp); tr_bigq_free(fact);
        tr_bigq_free(m2);
        break;
    }
    }
    tr_bigq_free(rn); tr_bigq_free(one); tr_bigq_free(pk); tr_bigq_free(ratio); tr_bigq_free(binom);
    tr_bigq_free(am); tr_bigq_free(rm); tr_bigq_free(geo); tr_bigq_free(geor);
    for (int i = 0; i < 4; i++) qiv_free(&base[i]);
    return c->err;
}

/* degree = the approximant's degree, support = the coefficients computed (N + 1: the Taylor terms behind a series,
 * the internal product of an operation), bytes = 160 per exact rational of the conversion scratch */
static int32_t series_need(int32_t degree, int32_t N, tr_math_need *need) {
    need_zero(need);
    need->need[TR_BUDGET_DEGREE] = (uint64_t)degree;
    need->need[TR_BUDGET_SUPPORT] = (uint64_t)N + 1;
    need->need[TR_BUDGET_BYTES] = tr_budget_add(tr_budget_mul(tr_budget_mul((uint64_t)N + 2, (uint64_t)N + 2), QBYTES),
                                                sizeof(tr_approx) + 4096);
    return TR_MATH_OK;
}

tr_approx *tr_approx_series(int32_t function, const tr_bigq *a, const tr_bigq *b, int32_t degree, int32_t basis,
                            int32_t taylor, int32_t norm, const tr_bigq *budget_q, const tr_math_budget *budget,
                            tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    if (function < TR_APPROX_EXP || function > TR_APPROX_RECIP || !q_ok(a) || !q_ok(b) || degree < 0 ||
        (basis != TR_APPROX_TAYLOR && basis != TR_APPROX_CHEBYSHEV) || taylor < 0 || !budget_ok(budget_q) ||
        (norm != TR_APPROX_NORM_ABS && norm != TR_APPROX_NORM_REL && norm != TR_APPROX_NORM_L2) || tr_bigq_cmp(a, b) >= 0) {
        set_error(error, TR_MATH_INVALID); return NULL;
    }
    if (degree > TR_APPROX_MAX_DEGREE || taylor > TR_APPROX_MAX_TAYLOR) { set_error(error, TR_MATH_UNSUPPORTED); return NULL; }
    int32_t N = basis == TR_APPROX_TAYLOR ? degree : (taylor ? taylor : TR_APPROX_ECONOMIZE(degree));
    if ((basis == TR_APPROX_TAYLOR && taylor && taylor != degree) || N < degree) { set_error(error, TR_MATH_INVALID); return NULL; }
    if (N > TR_APPROX_MAX_TAYLOR) N = TR_APPROX_MAX_TAYLOR;
    if (!q_small(a, 256) || !q_small(b, 256)) { set_error(error, TR_MATH_UNSUPPORTED); return NULL; }
    /* domains */
    if ((function == TR_APPROX_LOG || function == TR_APPROX_SQRT) && tr_bigq_sign(a) <= 0) { set_error(error, TR_MATH_DOMAIN_ERROR); return NULL; }
    if (function == TR_APPROX_RECIP && tr_bigq_sign(a) <= 0 && tr_bigq_sign(b) >= 0) { set_error(error, TR_MATH_DOMAIN_ERROR); return NULL; }
    if (function == TR_APPROX_EXP || function == TR_APPROX_SIN || function == TR_APPROX_COS) {
        int32_t s1 = 0, s2 = 0;
        if (tr_bigq_cmp_double(a, -1024.0, &s1) || tr_bigq_cmp_double(b, 1024.0, &s2) || s1 < 0 || s2 > 0) {
            set_error(error, TR_MATH_UNSUPPORTED); return NULL;
        }
    }
    tr_math_need need;
    series_need(degree, N, &need);
    tr_math_budget dflt;
    const tr_math_budget *bud = budget_or(budget, &dflt);
    int32_t rc = tr_budget_check(bud, &need, refusal);
    if (rc) { set_error(error, rc); return NULL; }
    qc ctx = {bud, refusal, 0}, *c = &ctx;
    tr_approx *x = approx_alloc(TR_APPROX_SERIES);
    if (!x) { set_error(error, TR_MATH_MEMORY); return NULL; }
    x->function = function; x->basis = basis; x->norm = norm; x->degree = degree; x->taylor = N;
    x->assumptions = TR_APPROX_ASSUME_DOMAIN | TR_APPROX_ASSUME_DERIVATIVE | TR_APPROX_ASSUME_LIPSCHITZ;
    x->a = qcopy(c, a); x->b = qcopy(c, b);
    tr_bigq *s = qadd(c, a, b), *d = qsub(c, b, a);
    x->mid = qdivint(c, s, 2); x->rad = qdivint(c, d, 2);
    tr_bigq_free(s); tr_bigq_free(d);
    qiv *g = calloc((size_t)N + 1, sizeof *g);
    if (!g) return approx_done(c, x, TR_MATH_MEMORY, error);
    tr_bigq *rem = NULL;
    rc = taylor_coefficients(c, function, a, b, x->mid, x->rad, N, g, &rem, &x->lipschitz, &x->derivative);
    if (!rc) {                                         /* the exact range: monotone functions, |sin|, |cos| <= 1 */
        if (function == TR_APPROX_EXP) {
            qiv ea = qiv_exp(c, a), eb = qiv_exp(c, b);
            x->frange_lo = qcopy(c, ea.lo); x->frange_hi = qcopy(c, eb.hi);
            qiv_free(&ea); qiv_free(&eb);
        } else if (function == TR_APPROX_LOG) {
            qiv la = qiv_log(c, a), lb = qiv_log(c, b);
            x->frange_lo = qcopy(c, la.lo); x->frange_hi = qcopy(c, lb.hi);
            qiv_free(&la); qiv_free(&lb);
        } else if (function == TR_APPROX_SQRT) {
            x->frange_lo = qsqrt_dir(c, a, WORK, 0); x->frange_hi = qsqrt_dir(c, b, WORK, 1);
        } else if (function == TR_APPROX_RECIP) {          /* decreasing on each side of 0 */
            tr_bigq *one = qint(c, 1);
            x->frange_lo = qdiv(c, one, b); x->frange_hi = qdiv(c, one, a);
            tr_bigq_free(one);
        } else { x->frange_lo = qint(c, -1); x->frange_hi = qint(c, 1); }
    }
    /* representatives at the dyadic midpoints; the half-widths are the coefficient error (|t^k| <= 1) */
    tr_bigq **mono = rc ? NULL : qvec(c, N + 1), *cerr = qint(c, 0);
    for (int32_t k = 0; mono && k <= N && !c->err; k++) {
        tr_bigq *sum = qadd(c, g[k].lo, g[k].hi), *mid = qdivint(c, sum, 2);
        qset(&mono[k], qround(c, mid, TR_APPROX_PREC, 0));
        tr_bigq *e1 = qsub(c, g[k].hi, mono[k]), *e2 = qsub(c, mono[k], g[k].lo), *e = qmax(c, e1, e2);
        qacc(c, &cerr, e);
        tr_bigq_free(sum); tr_bigq_free(mid); tr_bigq_free(e1); tr_bigq_free(e2); tr_bigq_free(e);
    }
    for (int32_t k = 0; k <= N; k++) qiv_free(&g[k]);
    free(g);
    uint64_t id = mix(mix(mix(mix(UINT64_C(0x5345524945530001), (uint64_t)function), (uint64_t)basis), (uint64_t)degree), (uint64_t)N);
    id = mix_q(mix_q(id, a), b);
    x->id = id;
    if (!rc && !c->err) {
        if (basis == TR_APPROX_CHEBYSHEV) {
            tr_bigq **ch = mono_to_cheb(c, mono, N), *tail = qint(c, 0);
            for (int32_t j = degree + 1; ch && j <= N; j++) qaccabs(c, &tail, ch[j]);
            if (ch) { x->coef = qvec(c, degree + 1); for (int32_t j = 0; x->coef && j <= degree; j++) qset(&x->coef[j], qcopy(c, ch[j])); }
            qvec_free(ch, N + 1);
            if (!rc) rc = term_add(c, x, TR_APPROX_TERM_ECONOMIZATION, id, 0, tail);
            tr_bigq_free(tail);
        } else {
            x->coef = qvec(c, degree + 1);
            for (int32_t j = 0; x->coef && j <= degree; j++) qset(&x->coef[j], qcopy(c, mono[j]));
        }
        if (!rc) rc = term_add(c, x, TR_APPROX_TERM_REMAINDER, id, 0, rem);
        if (!rc) rc = term_add(c, x, TR_APPROX_TERM_COEFFICIENT, id, 0, cerr);
    }
    qvec_free(mono, N + 1);
    tr_bigq_free(rem); tr_bigq_free(cerr);
    if (!rc && !c->err && !x->coef) rc = TR_MATH_MEMORY;
    if (!rc) rc = series_finish(c, x, budget_q);
    return approx_done(c, x, rc, error);
}

tr_approx *tr_approx_poly(const tr_bigq *const *coef, int32_t degree, const tr_bigq *a, const tr_bigq *b,
                          const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    if (!coef || degree < 0 || !q_ok(a) || !q_ok(b) || tr_bigq_cmp(a, b) >= 0) { set_error(error, TR_MATH_INVALID); return NULL; }
    if (degree > TR_APPROX_MAX_DEGREE) { set_error(error, TR_MATH_UNSUPPORTED); return NULL; }
    for (int32_t j = 0; j <= degree; j++) if (!q_ok(coef[j])) { set_error(error, TR_MATH_INVALID); return NULL; }
    tr_math_need need;
    series_need(degree, degree, &need);
    tr_math_budget dflt;
    const tr_math_budget *bud = budget_or(budget, &dflt);
    int32_t rc = tr_budget_check(bud, &need, refusal);
    if (rc) { set_error(error, rc); return NULL; }
    qc ctx = {bud, refusal, 0}, *c = &ctx;
    tr_approx *x = approx_alloc(TR_APPROX_SERIES);
    if (!x) { set_error(error, TR_MATH_MEMORY); return NULL; }
    x->function = TR_APPROX_POLY; x->basis = TR_APPROX_CHEBYSHEV; x->norm = TR_APPROX_NORM_ABS; x->degree = degree;
    x->assumptions = TR_APPROX_ASSUME_DOMAIN | TR_APPROX_ASSUME_LIPSCHITZ;
    x->a = qcopy(c, a); x->b = qcopy(c, b);
    tr_bigq *s = qadd(c, a, b), *d = qsub(c, b, a);
    x->mid = qdivint(c, s, 2); x->rad = qdivint(c, d, 2);
    tr_bigq_free(s); tr_bigq_free(d);
    /* p(mid + rad t) as monomials of t, then Chebyshev */
    tr_bigq **cx = qvec(c, degree + 1);
    for (int32_t j = 0; cx && j <= degree; j++) qset(&cx[j], qcopy(c, coef[j]));
    tr_bigq **mt = cx ? mono_substitute(c, cx, degree, x->rad, x->mid) : NULL;
    x->coef = mt ? mono_to_cheb(c, mt, degree) : NULL;
    /* Lipschitz: sup |p'| <= sum_j j |a_j| B^(j-1), B = max(|a|, |b|) */
    tr_bigq *aa = qabs(c, a), *bb = qabs(c, b), *B = qmax(c, aa, bb), *pw = qint(c, 1);
    x->lipschitz = qint(c, 0);
    for (int32_t j = 1; j <= degree && !c->err; j++) {
        tr_bigq *t = qabs(c, coef[j]), *u = qmulint(c, t, j), *v = qmul(c, u, pw);
        qacc(c, &x->lipschitz, v);
        tr_bigq_free(t); tr_bigq_free(u); tr_bigq_free(v);
        qset(&pw, qmul(c, pw, B));
    }
    uint64_t id = mix(mix(UINT64_C(0x504f4c5900000001), (uint64_t)degree), 0);
    for (int32_t j = 0; j <= degree; j++) id = mix_q(id, coef[j]);
    x->id = mix_q(mix_q(id, a), b);
    qvec_free(cx, degree + 1); qvec_free(mt, degree + 1);
    tr_bigq_free(aa); tr_bigq_free(bb); tr_bigq_free(B); tr_bigq_free(pw);
    if (!c->err && !x->coef) rc = TR_MATH_MEMORY;
    if (!rc) rc = series_finish(c, x, NULL);
    return approx_done(c, x, rc, error);
}

/* ------------------------------------------------------------------ operations on series */
static int series_operand(const tr_approx *x) { return approx_valid(x) && x->kind == TR_APPROX_SERIES; }
static tr_approx *series_from(qc *c, const tr_approx *f, int32_t op, uint64_t o1, uint64_t o2) {
    tr_approx *x = approx_alloc(TR_APPROX_SERIES);
    if (!x) return NULL;
    x->function = TR_APPROX_COMPOSED; x->basis = TR_APPROX_CHEBYSHEV; x->norm = f->norm; x->operation = op;
    x->assumptions = f->assumptions; x->origins[0] = o1; x->origins[1] = o2;
    x->a = qcopy(c, f->a); x->b = qcopy(c, f->b); x->mid = qcopy(c, f->mid); x->rad = qcopy(c, f->rad);
    x->id = mix(mix(mix(UINT64_C(0x4f50455200000001), (uint64_t)op), o1), o2);
    x->label = f->label; x->provenance = f->provenance;
    return x;
}
/* coefficients cf (degree n) installed, truncated to TR_APPROX_MAX_DEGREE with the exact tail as a TRUNCATION term */
static int32_t install(qc *c, tr_approx *x, tr_bigq **cf, int32_t n) {
    int32_t d = n > TR_APPROX_MAX_DEGREE ? TR_APPROX_MAX_DEGREE : n;
    while (d > 0 && !qsgn(cf[d])) d--;                 /* trailing zeros (an exact cancellation) are not degree */
    tr_bigq *tail = qint(c, 0);
    for (int32_t j = d + 1; j <= n; j++) qaccabs(c, &tail, cf[j]);
    int32_t rc = term_add(c, x, TR_APPROX_TERM_TRUNCATION, x->id, 0, tail);
    tr_bigq_free(tail);
    x->degree = d;
    x->coef = qvec(c, d + 1);
    for (int32_t j = 0; x->coef && j <= d; j++) qset(&x->coef[j], qcopy(c, cf[j]));
    if (!x->coef && !rc) rc = TR_MATH_MEMORY;
    return rc ? rc : c->err;
}
static int same_interval(const tr_approx *f, const tr_approx *g) { return !tr_bigq_cmp(f->a, g->a) && !tr_bigq_cmp(f->b, g->b); }
static int32_t op_prepare(const tr_approx *f, const tr_approx *g, const tr_bigq *budget_q, const tr_math_budget *budget,
                          tr_math_refusal *refusal, int32_t degree, const tr_math_budget **bud, tr_math_budget *dflt) {
    tr_budget_none(refusal);
    if (!series_operand(f) || (g && !series_operand(g)) || !budget_ok(budget_q)) return TR_MATH_INVALID;
    tr_math_need need;
    series_need(degree > TR_APPROX_MAX_DEGREE ? TR_APPROX_MAX_DEGREE : degree, degree, &need);
    *bud = budget_or(budget, dflt);
    return tr_budget_check(*bud, &need, refusal);
}

tr_approx *tr_approx_add(const tr_approx *f, const tr_approx *g, const tr_bigq *budget_q, const tr_math_budget *budget,
                         tr_math_refusal *refusal, int32_t *error) {
    tr_math_budget dflt;
    const tr_math_budget *bud;
    int32_t rc = op_prepare(f, g, budget_q, budget, refusal, f && g && approx_valid(f) && approx_valid(g) ?
                            (f->degree > g->degree ? f->degree : g->degree) : 0, &bud, &dflt);
    if (rc) { set_error(error, rc); return NULL; }
    if (!same_interval(f, g)) { set_error(error, TR_MATH_DOMAIN_ERROR); return NULL; }
    qc ctx = {bud, refusal, 0}, *c = &ctx;
    tr_approx *x = series_from(c, f, TR_APPROX_OP_ADD, f->id, g->id);
    if (!x) { set_error(error, TR_MATH_MEMORY); return NULL; }
    x->assumptions |= g->assumptions;
    int32_t n = f->degree > g->degree ? f->degree : g->degree;
    tr_bigq **cf = as_cheb(c, f), **cg = as_cheb(c, g), **out = qvec(c, n + 1);
    for (int32_t j = 0; out && cf && cg && j <= n; j++) {
        if (j <= f->degree) qacc(c, &out[j], cf[j]);
        if (j <= g->degree) qacc(c, &out[j], cg[j]);
    }
    if (!rc) rc = terms_carry(c, x, f, NULL);
    if (!rc) rc = terms_carry(c, x, g, NULL);
    if (!rc && out) rc = install(c, x, out, n);
    x->lipschitz = qadd(c, f->lipschitz, g->lipschitz);
    x->label = f->label == TR_APPROX_LABEL_PROBABILISTIC || g->label == TR_APPROX_LABEL_PROBABILISTIC ? TR_APPROX_LABEL_PROBABILISTIC : TR_APPROX_LABEL_CERTIFIED;
    qvec_free(cf, f->degree + 1); qvec_free(cg, g->degree + 1); qvec_free(out, n + 1);
    if (!rc) rc = series_finish(c, x, budget_q);
    return approx_done(c, x, rc, error);
}

tr_approx *tr_approx_scale(const tr_approx *f, const tr_bigq *k, const tr_bigq *budget_q, const tr_math_budget *budget,
                           tr_math_refusal *refusal, int32_t *error) {
    tr_math_budget dflt;
    const tr_math_budget *bud;
    int32_t rc = op_prepare(f, NULL, budget_q, budget, refusal, f && approx_valid(f) ? f->degree : 0, &bud, &dflt);
    if (!rc && !q_ok(k)) rc = TR_MATH_INVALID;
    if (rc) { set_error(error, rc); return NULL; }
    qc ctx = {bud, refusal, 0}, *c = &ctx;
    tr_approx *x = series_from(c, f, TR_APPROX_OP_SCALE, f->id, mix_q(0, k));
    if (!x) { set_error(error, TR_MATH_MEMORY); return NULL; }
    tr_bigq **cf = as_cheb(c, f), **out = qvec(c, f->degree + 1), *ak = qabs(c, k);
    for (int32_t j = 0; out && cf && j <= f->degree; j++) qset(&out[j], qmul(c, cf[j], k));
    rc = terms_carry(c, x, f, ak);
    if (!rc && out) rc = install(c, x, out, f->degree);
    x->lipschitz = qmul(c, f->lipschitz, ak);
    qvec_free(cf, f->degree + 1); qvec_free(out, f->degree + 1); tr_bigq_free(ak);
    if (!rc) rc = series_finish(c, x, budget_q);
    return approx_done(c, x, rc, error);
}

/* sum |c_j| over j >= from */
static tr_bigq *abs_sum(qc *c, tr_bigq *const *cf, int32_t n, int32_t from) {
    tr_bigq *s = qint(c, 0);
    for (int32_t j = from; j <= n; j++) qaccabs(c, &s, cf[j]);
    return s;
}

tr_approx *tr_approx_mul(const tr_approx *f, const tr_approx *g, const tr_bigq *budget_q, const tr_math_budget *budget,
                         tr_math_refusal *refusal, int32_t *error) {
    tr_math_budget dflt;
    const tr_math_budget *bud;
    int32_t deg = f && g && approx_valid(f) && approx_valid(g) ? f->degree + g->degree : 0;
    int32_t rc = op_prepare(f, g, budget_q, budget, refusal, deg, &bud, &dflt);
    if (rc) { set_error(error, rc); return NULL; }
    if (!same_interval(f, g)) { set_error(error, TR_MATH_DOMAIN_ERROR); return NULL; }
    qc ctx = {bud, refusal, 0}, *c = &ctx;
    tr_approx *x = series_from(c, f, TR_APPROX_OP_MUL, f->id, g->id);
    if (!x) { set_error(error, TR_MATH_MEMORY); return NULL; }
    x->assumptions |= g->assumptions;
    tr_bigq **cf = as_cheb(c, f), **cg = as_cheb(c, g);
    tr_bigq **out = cf && cg ? cheb_mul(c, cf, f->degree, cg, g->degree) : NULL;
    /* |fg - P_F P_G| <= S_G e_F + (S_F + e_F) e_G, S = sum |c_j| >= sup |P| */
    tr_bigq *SF = cf ? abs_sum(c, cf, f->degree, 0) : NULL, *SG = cg ? abs_sum(c, cg, g->degree, 0) : NULL;
    tr_bigq *supf = qadd(c, SF, f->error), *supg = qadd(c, SG, g->error);
    rc = terms_carry(c, x, f, SG);
    if (!rc) rc = terms_carry(c, x, g, supf);
    if (!rc && out) rc = install(c, x, out, deg);
    tr_bigq *l1 = qmul(c, supf, g->lipschitz), *l2 = qmul(c, supg, f->lipschitz);
    x->lipschitz = qadd(c, l1, l2);
    x->label = f->label == TR_APPROX_LABEL_PROBABILISTIC || g->label == TR_APPROX_LABEL_PROBABILISTIC ? TR_APPROX_LABEL_PROBABILISTIC : TR_APPROX_LABEL_CERTIFIED;
    qvec_free(cf, f->degree + 1); qvec_free(cg, g->degree + 1); qvec_free(out, deg + 1);
    tr_bigq_free(SF); tr_bigq_free(SG); tr_bigq_free(supf); tr_bigq_free(supg); tr_bigq_free(l1); tr_bigq_free(l2);
    if (!rc) rc = series_finish(c, x, budget_q);
    return approx_done(c, x, rc, error);
}

tr_approx *tr_approx_compose(const tr_approx *f, const tr_approx *g, const tr_bigq *budget_q, const tr_math_budget *budget,
                             tr_math_refusal *refusal, int32_t *error) {
    tr_math_budget dflt;
    const tr_math_budget *bud;
    int32_t deg = f && g && approx_valid(f) && approx_valid(g) ? f->degree * g->degree : 0;
    int32_t rc = op_prepare(f, g, budget_q, budget, refusal, deg, &bud, &dflt);
    if (rc) { set_error(error, rc); return NULL; }
    if (deg > SERIES_CAP) { set_error(error, TR_MATH_UNSUPPORTED); return NULL; }
    qc ctx = {bud, refusal, 0}, *c = &ctx;
    /* the assumption: the exact inner range widened by the inner error (so both g(x) and P_G(x)) lies in the outer
     * interval, where the outer error bound and its Lipschitz bound hold */
    tr_bigq *glo = qsub(c, g->range_lo, g->error), *ghi = qadd(c, g->range_hi, g->error);
    int inside = !c->err && qcmp(c, glo, f->a) >= 0 && qcmp(c, ghi, f->b) <= 0;
    tr_bigq_free(glo); tr_bigq_free(ghi);
    if (c->err) { set_error(error, c->err); return NULL; }
    if (!inside) { set_error(error, TR_MATH_DOMAIN_ERROR); return NULL; }
    tr_approx *x = series_from(c, g, TR_APPROX_OP_COMPOSE, f->id, g->id);
    if (!x) { set_error(error, TR_MATH_MEMORY); return NULL; }
    x->assumptions |= f->assumptions | TR_APPROX_ASSUME_RANGE | TR_APPROX_ASSUME_LIPSCHITZ;
    tr_bigq **cf = as_cheb(c, f), **cg = as_cheb(c, g);
    /* u(t) = (P_G(t) - mid_F) / rad_F, then Clenshaw over Chebyshev series: b_k = c_k + 2 u b_{k+1} - b_{k+2} */
    int32_t dg = g->degree, n = f->degree;
    tr_bigq **u = qvec(c, dg + 1);
    for (int32_t j = 0; u && cg && j <= dg; j++) {
        tr_bigq *t = j ? qcopy(c, cg[j]) : qsub(c, cg[0], f->mid);
        qset(&u[j], qdiv(c, t, f->rad));
        tr_bigq_free(t);
    }
    int32_t cap = deg + 1;
    tr_bigq **b1 = qvec(c, cap), **b2 = qvec(c, cap), **u2 = qvec(c, dg + 1);
    for (int32_t j = 0; u2 && u && j <= dg; j++) qset(&u2[j], qmulint(c, u[j], 2));
    int32_t d1 = 0, d2 = 0;                            /* degrees of b1 = b_{k+1}, b2 = b_{k+2} */
    if (b1 && cf && n >= 1) qset(&b1[0], qcopy(c, cf[n]));   /* b_n = c_n */
    for (int32_t k = n - 1; k >= 1 && !c->err && cf; k--) {
        tr_bigq **p = cheb_mul(c, u2, dg, b1, d1);     /* degree dg + d1 */
        int32_t dp = dg + d1, d0 = dp > d2 ? dp : d2;
        tr_bigq **b0 = qvec(c, cap);
        for (int32_t j = 0; b0 && p && j <= d0 && j < cap; j++) {
            if (j <= dp) qacc(c, &b0[j], p[j]);
            if (j <= d2) { tr_bigq *t = qneg(c, b2[j]); qacc(c, &b0[j], t); tr_bigq_free(t); }
        }
        if (b0) qacc(c, &b0[0], cf[k]);
        qvec_free(p, dp + 1);
        qvec_free(b2, cap); b2 = b1; d2 = d1; b1 = b0; d1 = d0 < cap ? d0 : cap - 1;
    }
    tr_bigq **out = qvec(c, cap);
    if (out && cf && !c->err && n == 0) qset(&out[0], qcopy(c, cf[0]));
    else if (out && cf && !c->err) {                   /* c_0 + u b_1 - b_2 */
        tr_bigq **p = cheb_mul(c, u, dg, b1, d1);
        for (int32_t j = 0; p && j <= dg + d1 && j < cap; j++) qacc(c, &out[j], p[j]);
        for (int32_t j = 0; j <= d2; j++) { tr_bigq *t = qneg(c, b2[j]); qacc(c, &out[j], t); tr_bigq_free(t); }
        qacc(c, &out[0], cf[0]);
        qvec_free(p, dg + d1 + 1);
    }
    rc = terms_carry(c, x, f, NULL);
    if (!rc) rc = terms_carry(c, x, g, f->lipschitz);  /* the inner error through the outer Lipschitz bound */
    if (!rc && out) rc = install(c, x, out, deg);
    x->lipschitz = qmul(c, f->lipschitz, g->lipschitz);
    x->label = f->label == TR_APPROX_LABEL_PROBABILISTIC || g->label == TR_APPROX_LABEL_PROBABILISTIC ? TR_APPROX_LABEL_PROBABILISTIC : TR_APPROX_LABEL_CERTIFIED;
    qvec_free(cf, f->degree + 1); qvec_free(cg, g->degree + 1); qvec_free(u, dg + 1); qvec_free(u2, dg + 1);
    qvec_free(b1, cap); qvec_free(b2, cap); qvec_free(out, cap);
    if (!rc) rc = series_finish(c, x, budget_q);
    return approx_done(c, x, rc, error);
}

tr_approx *tr_approx_quantize(const tr_approx *f, int32_t bits, const tr_bigq *budget_q, const tr_math_budget *budget,
                              tr_math_refusal *refusal, int32_t *error) {
    tr_math_budget dflt;
    const tr_math_budget *bud;
    int32_t rc = op_prepare(f, NULL, budget_q, budget, refusal, f && approx_valid(f) ? f->degree : 0, &bud, &dflt);
    if (!rc && (bits < 1 || bits > 512)) rc = TR_MATH_INVALID;
    if (rc) { set_error(error, rc); return NULL; }
    qc ctx = {bud, refusal, 0}, *c = &ctx;
    tr_approx *x = series_from(c, f, TR_APPROX_OP_QUANTIZE, f->id, (uint64_t)bits);
    if (!x) { set_error(error, TR_MATH_MEMORY); return NULL; }
    x->function = f->function; x->basis = f->basis; x->taylor = f->taylor; x->quantized = bits;
    tr_bigq **out = qvec(c, f->degree + 1), *qe = qint(c, 0);
    for (int32_t j = 0; out && j <= f->degree; j++) {
        qset(&out[j], qgrid(c, f->coef[j], bits, 0));
        tr_bigq *d = qsub(c, f->coef[j], out[j]);
        qaccabs(c, &qe, d);
        tr_bigq_free(d);
    }
    rc = terms_carry(c, x, f, NULL);
    if (!rc) rc = term_add(c, x, TR_APPROX_TERM_QUANTIZATION, x->id, 0, qe);
    x->degree = f->degree; x->coef = out; out = NULL;
    x->lipschitz = qcopy(c, f->lipschitz); x->derivative = f->derivative ? qcopy(c, f->derivative) : NULL;
    if (f->frange_lo) { x->frange_lo = qcopy(c, f->frange_lo); x->frange_hi = qcopy(c, f->frange_hi); }
    tr_bigq_free(qe);
    if (!rc && !x->coef) rc = TR_MATH_MEMORY;
    if (!rc) rc = series_finish(c, x, budget_q);
    return approx_done(c, x, rc, error);
}

tr_approx *tr_approx_float(const tr_approx *f, const tr_bigq *budget_q, const tr_math_budget *budget,
                           tr_math_refusal *refusal, int32_t *error) {
    tr_math_budget dflt;
    const tr_math_budget *bud;
    int32_t rc = op_prepare(f, NULL, budget_q, budget, refusal, f && approx_valid(f) ? f->degree : 0, &bud, &dflt);
    if (rc) { set_error(error, rc); return NULL; }
    qc ctx = {bud, refusal, 0}, *c = &ctx;
    tr_approx *x = series_from(c, f, TR_APPROX_OP_FLOAT, f->id, 0);
    if (!x) { set_error(error, TR_MATH_MEMORY); return NULL; }
    x->kind = TR_APPROX_FLOAT; x->function = f->function; x->basis = TR_APPROX_MONOMIAL_X;
    int32_t n = f->degree;
    x->degree = n; x->taylor = f->taylor; x->quantized = f->quantized;
    tr_bigq **mt = f->basis == TR_APPROX_CHEBYSHEV ? cheb_to_mono(c, f->coef, n) : NULL;
    tr_bigq *one = qint(c, 1), *alpha = qdiv(c, one, f->rad), *mm = qneg(c, f->mid), *beta = qmul(c, mm, alpha);
    tr_bigq **ax = mono_substitute(c, f->basis == TR_APPROX_CHEBYSHEV ? mt : f->coef, n, alpha, beta);
    x->coef = qvec(c, n + 1);
    x->dcoef = calloc((size_t)n + 1, sizeof *x->dcoef);
    if (!x->dcoef) rc = TR_MATH_MEMORY;
    /* round to binary64 (nearest), the coefficient rounding sum |a_j - fl(a_j)| B^j and Horner's sum |fl(a_j)| B^j */
    tr_bigq *aa = qabs(c, f->a), *bb = qabs(c, f->b), *B = qmax(c, aa, bb), *pw = qint(c, 1);
    tr_bigq *cround = qint(c, 0), *hsum = qint(c, 0);
    for (int32_t j = 0; !rc && ax && x->coef && j <= n && !c->err; j++) {
        tr_math_interval iv;
        if (tr_bigq_enclose(ax[j], &iv) || !(iv.lo > -INFINITY) || !(iv.hi < INFINITY)) { rc = TR_MATH_OVERFLOW; break; }
        int32_t e = 0;
        tr_bigq *lo = qwrap(c, tr_bigq_from_double(iv.lo, c->b, c->r, &e), e);
        tr_bigq *hi = qwrap(c, tr_bigq_from_double(iv.hi, c->b, c->r, &e), e);
        tr_bigq *dl = qsub(c, ax[j], lo), *dh = qsub(c, hi, ax[j]);
        int pick_lo = qcmp(c, dl, dh) <= 0;
        x->dcoef[j] = pick_lo ? iv.lo : iv.hi;
        qset(&x->coef[j], qcopy(c, pick_lo ? lo : hi));
        tr_bigq *d = pick_lo ? qcopy(c, dl) : qcopy(c, dh), *t = qmul(c, d, pw);
        qacc(c, &cround, t);
        tr_bigq *ac = qabs(c, x->coef[j]), *t2 = qmul(c, ac, pw);
        qacc(c, &hsum, t2);
        tr_bigq_free(lo); tr_bigq_free(hi); tr_bigq_free(dl); tr_bigq_free(dh); tr_bigq_free(d); tr_bigq_free(t);
        tr_bigq_free(ac); tr_bigq_free(t2);
        if (j < n) qset(&pw, qmul(c, pw, B));
    }
    /* gamma_2n = 2n u / (1 - 2n u), u = 2^-53; underflow (n + 1) 2^-1074 max(1, B)^n */
    tr_bigq *horner = NULL;
    if (!rc && !c->err) {
        tr_bigq *u = q2pow(c, -53), *nu = qmulint(c, u, 2 * (int64_t)(n > 0 ? n : 1)), *den = qsub(c, one, nu);
        tr_bigq *gamma = qdiv(c, nu, den);
        horner = qmul(c, gamma, hsum);
        tr_bigq *mb = qmax(c, one, B), *mbn = qpow(c, mb, n), *tiny = q2pow(c, -1074), *uf = qmulint(c, tiny, n + 1);
        qset(&uf, qmul(c, uf, mbn));
        qacc(c, &horner, uf);
        tr_bigq_free(u); tr_bigq_free(nu); tr_bigq_free(den); tr_bigq_free(gamma); tr_bigq_free(mb); tr_bigq_free(mbn);
        tr_bigq_free(tiny); tr_bigq_free(uf);
    }
    if (!rc) rc = terms_carry(c, x, f, NULL);
    if (!rc) rc = term_add(c, x, TR_APPROX_TERM_ROUNDING, x->id, 0, cround);
    if (!rc) rc = term_add(c, x, TR_APPROX_TERM_ROUNDING, x->id ^ 1, 0, horner);
    x->lipschitz = qcopy(c, f->lipschitz);
    qvec_free(mt, n + 1); qvec_free(ax, n + 1);
    tr_bigq_free(one); tr_bigq_free(alpha); tr_bigq_free(mm); tr_bigq_free(beta); tr_bigq_free(aa); tr_bigq_free(bb);
    tr_bigq_free(B); tr_bigq_free(pw); tr_bigq_free(cround); tr_bigq_free(hsum); tr_bigq_free(horner);
    if (!rc && !x->coef) rc = TR_MATH_MEMORY;
    x->frange_lo = qcopy(c, f->range_lo); x->frange_hi = qcopy(c, f->range_hi);   /* the same exact function */
    if (!rc) rc = series_finish(c, x, budget_q);
    return approx_done(c, x, rc, error);
}

/* ------------------------------------------------------------------ low rank */
static uint64_t sm64(uint64_t z) {
    z += UINT64_C(0x9E3779B97F4A7C15);
    z = (z ^ (z >> 30)) * UINT64_C(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94D049BB133111EB);
    return z ^ (z >> 31);
}
/* +1 / -1 at (stream, index): the sign bit of splitmix64(splitmix64(seed ^ stream constant) + index) */
static int rademacher(uint64_t seed, uint64_t stream, uint64_t index) {
    uint64_t s = sm64(seed ^ (UINT64_C(0xD1B54A32D192ED03) * (stream + 1)));
    return (sm64(s + index) >> 63) ? -1 : 1;
}
static int32_t lowrank_need(int32_t rows, int32_t cols, int32_t rank, int32_t probes, tr_math_need *need) {
    need_zero(need);
    uint64_t t = (uint64_t)rows * (uint64_t)cols;
    need->need[TR_BUDGET_TENSOR] = t;
    need->need[TR_BUDGET_SUPPORT] = (uint64_t)rank;
    need->need[TR_BUDGET_ITERATIONS] = (uint64_t)probes;
    /* the approximant, the elimination's (or the projection's) scratch and the factors: 4 rationals per entry */
    need->need[TR_BUDGET_BYTES] = tr_budget_add(tr_budget_mul(tr_budget_mul(t, 4), QBYTES), sizeof(tr_approx) + 8 * t + 4096);
    return TR_MATH_OK;
}
/* the norm bounds of a residual given as integers M (n entries) over the common denominator d (> 0 magnitude) */
static void residual_norms(qc *c, tr_approx *x, tr_bigint **M, int32_t rows, int32_t cols, const tr_bigint *d) {
    tr_bigint *F = bint(c, 0), *mx = bint(c, 0);
    tr_bigint **colsum = calloc((size_t)cols, sizeof *colsum), *rowmax = bint(c, 0), *colmax = bint(c, 0);
    if (!colsum) { qfail(c, TR_MATH_MEMORY); return; }
    for (int32_t j = 0; j < cols; j++) colsum[j] = bint(c, 0);
    for (int32_t i = 0; i < rows && !c->err; i++) {
        tr_bigint *rs = bint(c, 0);
        for (int32_t j = 0; j < cols && !c->err; j++) {
            const tr_bigint *m = M[i * cols + j];
            if (!tr_bigint_sign(m)) continue;
            tr_bigint *sq = bmul(c, m, m), *am = babs(c, m);
            bset(&F, badd(c, F, sq));
            if (tr_bigint_cmp(am, mx) > 0) bset(&mx, bdup(c, am));
            bset(&rs, badd(c, rs, am));
            bset(&colsum[j], badd(c, colsum[j], am));
            tr_bigint_free(sq); tr_bigint_free(am);
        }
        if (!c->err && tr_bigint_cmp(rs, rowmax) > 0) bset(&rowmax, bdup(c, rs));
        tr_bigint_free(rs);
    }
    for (int32_t j = 0; j < cols && !c->err; j++) if (tr_bigint_cmp(colsum[j], colmax) > 0) bset(&colmax, bdup(c, colsum[j]));
    tr_bigint *ad = babs(c, d), *d2 = bmul(c, ad, ad);
    qset(&x->frob2, qbig(c, F, d2));
    tr_bigq *fq = qcopy(c, x->frob2);
    qset(&x->frob, qsqrt_dir(c, fq, TR_APPROX_PREC, 1));    /* exact when ||S||_F^2 is a square (0 included) */
    qset(&x->maxentry, qbig(c, mx, ad));
    tr_bigint *p = bmul(c, rowmax, colmax);
    tr_bigq *pq = qbig(c, p, d2), *sp = qsqrt_dir(c, pq, TR_APPROX_PREC, 1);
    qset(&x->spectral, qmin(c, x->frob, sp));
    tr_bigint_free(F); tr_bigint_free(mx); tr_bigint_free(rowmax); tr_bigint_free(colmax); tr_bigint_free(ad);
    tr_bigint_free(d2); tr_bigint_free(p);
    tr_bigq_free(fq); tr_bigq_free(pq); tr_bigq_free(sp);
    for (int32_t j = 0; j < cols; j++) tr_bigint_free(colsum[j]);
    free(colsum);
}
/* CROSS: k steps of Bareiss elimination with complete pivoting */
static int32_t lowrank_cross(qc *c, tr_approx *x, int32_t k) {
    int32_t R = x->rows, Cn = x->cols, n = R * Cn;
    tr_bigint **M = calloc((size_t)n, sizeof *M);
    int32_t *pr = malloc((size_t)R * sizeof *pr), *pc = malloc((size_t)Cn * sizeof *pc);
    x->prow = calloc((size_t)(k > 0 ? k : 1), sizeof *x->prow); x->pcol = calloc((size_t)(k > 0 ? k : 1), sizeof *x->pcol);
    if (!M || !pr || !pc || !x->prow || !x->pcol) { free(M); free(pr); free(pc); return TR_MATH_MEMORY; }
    for (int32_t i = 0; i < n; i++) M[i] = bint(c, x->A[i]);
    for (int32_t i = 0; i < R; i++) pr[i] = i;
    for (int32_t j = 0; j < Cn; j++) pc[j] = j;
#define AT(i, j) M[pr[i] * Cn + pc[j]]
    tr_bigint *prev = bint(c, 1);
    int32_t r = 0;
    for (int32_t t = 0; t < k && !c->err; t++) {
        int32_t bi = -1, bj = -1;                      /* the largest |entry|; ties: smaller physical row, column */
        for (int32_t i = t; i < R && !c->err; i++)
            for (int32_t j = t; j < Cn && !c->err; j++) {
                if (!tr_bigint_sign(AT(i, j))) continue;
                if (bi < 0) { bi = i; bj = j; continue; }
                int cmp = bcmpabs(c, AT(i, j), AT(bi, bj));
                if (cmp > 0 || (cmp == 0 && (pr[i] < pr[bi] || (pr[i] == pr[bi] && pc[j] < pc[bj])))) { bi = i; bj = j; }
            }
        if (bi < 0) break;                             /* the trailing matrix is zero: the rank is t */
        int32_t s = pr[t]; pr[t] = pr[bi]; pr[bi] = s;
        s = pc[t]; pc[t] = pc[bj]; pc[bj] = s;
        const tr_bigint *p = AT(t, t);
        for (int32_t i = t + 1; i < R && !c->err; i++)
            for (int32_t j = t + 1; j < Cn && !c->err; j++) {
                tr_bigint *a1 = bmul(c, p, AT(i, j)), *a2 = bmul(c, AT(i, t), AT(t, j)), *df = bsub(c, a1, a2);
                tr_bigint *q = bdivexact(c, df, prev);
                bset(&AT(i, j), q);
                tr_bigint_free(a1); tr_bigint_free(a2); tr_bigint_free(df);
            }
        bset(&prev, bdup(c, p));
        x->prow[t] = pr[t]; x->pcol[t] = pc[t];
        r = t + 1;
    }
    x->rank = r;
    /* the residual: the trailing entries over prev (zero on the pivot rows and columns), in physical positions */
    tr_bigint **S = calloc((size_t)n, sizeof *S);
    if (!S) { qfail(c, TR_MATH_MEMORY); }
    for (int32_t i = 0; S && i < n; i++) S[i] = bint(c, 0);
    for (int32_t i = r; S && i < R && !c->err; i++)
        for (int32_t j = r; j < Cn && !c->err; j++) bset(&S[pr[i] * Cn + pc[j]], bdup(c, AT(i, j)));
#undef AT
    if (S && !c->err) {
        residual_norms(c, x, S, R, Cn, prev);
        x->entry = qvec(c, n);
        for (int32_t i = 0; x->entry && i < n && !c->err; i++) {   /* A - S / prev */
            tr_bigq *s = qbig(c, S[i], prev), *a = qint(c, x->A[i]);
            qset(&x->entry[i], qsub(c, a, s));
            tr_bigq_free(s); tr_bigq_free(a);
        }
    }
    for (int32_t i = 0; i < n; i++) { tr_bigint_free(M[i]); if (S) tr_bigint_free(S[i]); }
    free(M); free(S); free(pr); free(pc); tr_bigint_free(prev);
    return c->err;
}
/* SKETCH: Y = A Omega, its independent columns, the exact projection, the probes */
static int32_t lowrank_sketch(qc *c, tr_approx *x, int32_t k) {
    int32_t R = x->rows, Cn = x->cols, n = R * Cn;
    x->omega = malloc((size_t)Cn * (size_t)k);
    x->kept = calloc((size_t)k, sizeof *x->kept);
    if (!x->omega || !x->kept) return TR_MATH_MEMORY;
    for (int32_t i = 0; i < Cn; i++)
        for (int32_t j = 0; j < k; j++) x->omega[i * k + j] = (int8_t)rademacher(x->seed, 0, (uint64_t)i * (uint64_t)k + (uint64_t)j);
    /* Y (exact integers in 128 bits: |entries| <= 64 * 2^63) */
    tr_bigq **Y = qvec(c, R * k);
    for (int32_t i = 0; Y && i < R; i++)
        for (int32_t j = 0; j < k && !c->err; j++) {
            __int128 s = 0;
            for (int32_t l = 0; l < Cn; l++) s += (__int128)x->A[i * Cn + l] * x->omega[l * k + j];
            tr_bigint *v = bi128(c, s);
            qset(&Y[i * k + j], qbigint(c, v));
            tr_bigint_free(v);
        }
    /* the independent columns of Y in order (Gaussian elimination by columns over the rationals) */
    tr_bigq **E = qvec(c, R * k);
    int32_t *prow = calloc((size_t)k, sizeof *prow), nk = 0;
    if (!prow) qfail(c, TR_MATH_MEMORY);
    for (int32_t i = 0; E && Y && i < R * k; i++) qset(&E[i], qcopy(c, Y[i]));
    for (int32_t j = 0; j < k && E && prow && !c->err; j++) {
        for (int32_t p = 0; p < nk && !c->err; p++) {
            int32_t pc = x->kept[p], rr = prow[p];
            if (!qsgn(E[rr * k + j])) continue;
            tr_bigq *f = qdiv(c, E[rr * k + j], E[rr * k + pc]);
            for (int32_t i = 0; i < R && !c->err; i++) {
                if (!qsgn(E[i * k + pc])) continue;
                tr_bigq *t = qmul(c, f, E[i * k + pc]);
                qset(&E[i * k + j], qsub(c, E[i * k + j], t));
                tr_bigq_free(t);
            }
            tr_bigq_free(f);
        }
        int32_t piv = -1;
        for (int32_t i = 0; i < R && piv < 0; i++) {
            int used = 0;
            for (int32_t p = 0; p < nk; p++) used |= prow[p] == i;
            if (!used && qsgn(E[i * k + j])) piv = i;
        }
        if (piv >= 0) { x->kept[nk] = j; prow[nk] = piv; nk++; }
    }
    x->nkept = nk; x->rank = nk;
    qvec_free(E, R * k); free(prow);
    /* G = Y'^T Y', H = Y'^T A; X = G^-1 H by Gauss-Jordan (G is positive definite: no pivoting needed) */
    tr_bigq **G = qvec(c, nk * nk), **H = qvec(c, nk * Cn);
    for (int32_t p = 0; G && H && Y && p < nk && !c->err; p++) {
        for (int32_t q = 0; q < nk && !c->err; q++)
            for (int32_t i = 0; i < R; i++) { tr_bigq *t = qmul(c, Y[i * k + x->kept[p]], Y[i * k + x->kept[q]]); qacc(c, &G[p * nk + q], t); tr_bigq_free(t); }
        for (int32_t l = 0; l < Cn && !c->err; l++)
            for (int32_t i = 0; i < R; i++) {
                if (!x->A[i * Cn + l]) continue;
                tr_bigq *t = qmulint(c, Y[i * k + x->kept[p]], x->A[i * Cn + l]);
                qacc(c, &H[p * Cn + l], t);
                tr_bigq_free(t);
            }
    }
    for (int32_t p = 0; G && H && p < nk && !c->err; p++) {
        tr_bigq *piv = qcopy(c, G[p * nk + p]);
        for (int32_t q = 0; q < nk; q++) qset(&G[p * nk + q], qdiv(c, G[p * nk + q], piv));
        for (int32_t l = 0; l < Cn; l++) qset(&H[p * Cn + l], qdiv(c, H[p * Cn + l], piv));
        tr_bigq_free(piv);
        for (int32_t s = 0; s < nk && !c->err; s++) {
            if (s == p || !qsgn(G[s * nk + p])) continue;
            tr_bigq *f = qcopy(c, G[s * nk + p]);
            for (int32_t q = 0; q < nk; q++) { tr_bigq *t = qmul(c, f, G[p * nk + q]); qset(&G[s * nk + q], qsub(c, G[s * nk + q], t)); tr_bigq_free(t); }
            for (int32_t l = 0; l < Cn; l++) { tr_bigq *t = qmul(c, f, H[p * Cn + l]); qset(&H[s * Cn + l], qsub(c, H[s * Cn + l], t)); tr_bigq_free(t); }
            tr_bigq_free(f);
        }
    }
    /* the approximant Y' X and the probes ||E w||^2, E = A - Y' X */
    x->entry = qvec(c, n);
    for (int32_t i = 0; x->entry && H && Y && i < R && !c->err; i++)
        for (int32_t l = 0; l < Cn && !c->err; l++)
            for (int32_t p = 0; p < nk; p++) {
                tr_bigq *t = qmul(c, Y[i * k + x->kept[p]], H[p * Cn + l]);
                qacc(c, &x->entry[i * Cn + l], t);
                tr_bigq_free(t);
            }
    tr_bigq *mx = qint(c, 0);
    for (int32_t pi = 0; pi < x->probes && x->entry && !c->err; pi++) {
        tr_bigq *sq = qint(c, 0);
        for (int32_t i = 0; i < R && !c->err; i++) {
            tr_bigq *acc = qint(c, 0);
            for (int32_t l = 0; l < Cn && !c->err; l++) {
                tr_bigq *a = qint(c, x->A[i * Cn + l]), *e = qsub(c, a, x->entry[i * Cn + l]);
                if (rademacher(x->seed, 1, (uint64_t)pi * (uint64_t)Cn + (uint64_t)l) > 0) qacc(c, &acc, e);
                else qset(&acc, qsub(c, acc, e));
                tr_bigq_free(a); tr_bigq_free(e);
            }
            tr_bigq *a2 = qmul(c, acc, acc);
            qacc(c, &sq, a2);
            tr_bigq_free(acc); tr_bigq_free(a2);
        }
        x->probe[pi] = sq;
        if (qcmp(c, sq, mx) > 0) qset(&mx, qcopy(c, sq));
    }
    /* with probability >= 1 - (13/16)^r: ||E||_F^2 <= 4 max_i ||E w_i||^2 */
    tr_bigq *four = qmulint(c, mx, 4);
    qset(&x->frob, qsqrt_dir(c, four, TR_APPROX_PREC, 1));
    qset(&x->spectral, qcopy(c, x->frob));
    qset(&x->maxentry, qcopy(c, x->frob));
    tr_bigq *ratio = qfrac(c, 13, 16);
    qset(&x->failure, qpow(c, ratio, x->probes));
    qvec_free(Y, R * k); qvec_free(G, nk * nk); qvec_free(H, nk * Cn);
    tr_bigq_free(mx); tr_bigq_free(four); tr_bigq_free(ratio);
    return c->err;
}

tr_approx *tr_approx_lowrank(const int64_t *A, int32_t rows, int32_t cols, int32_t rank, int32_t method, uint64_t seed,
                             int32_t probes, int32_t norm, const tr_bigq *budget_q, const tr_math_budget *budget,
                             tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    if (!A || rows < 1 || cols < 1 || rank < 1 || (method != TR_APPROX_CROSS && method != TR_APPROX_SKETCH) ||
        !budget_ok(budget_q) || (norm != TR_APPROX_NORM_FROBENIUS && norm != TR_APPROX_NORM_SPECTRAL && norm != TR_APPROX_NORM_MAX) ||
        (method == TR_APPROX_SKETCH && probes < 1)) { set_error(error, TR_MATH_INVALID); return NULL; }
    if (rows > TR_APPROX_MAX_DIM || cols > TR_APPROX_MAX_DIM || probes > TR_APPROX_MAX_PROBES) { set_error(error, TR_MATH_UNSUPPORTED); return NULL; }
    if (rank > rows || rank > cols) { set_error(error, TR_MATH_INVALID); return NULL; }
    if (method == TR_APPROX_CROSS) probes = 0;
    tr_math_need need;
    lowrank_need(rows, cols, rank, probes, &need);
    tr_math_budget dflt;
    const tr_math_budget *bud = budget_or(budget, &dflt);
    int32_t rc = tr_budget_check(bud, &need, refusal);
    if (rc) { set_error(error, rc); return NULL; }
    qc ctx = {bud, refusal, 0}, *c = &ctx;
    tr_approx *x = approx_alloc(TR_APPROX_LOWRANK);
    if (!x) { set_error(error, TR_MATH_MEMORY); return NULL; }
    x->function = TR_APPROX_MATRIX; x->method = method; x->norm = norm; x->rows = rows; x->cols = cols;
    x->probes = probes; x->seed = method == TR_APPROX_SKETCH ? seed : 0;
    x->A = malloc((size_t)rows * (size_t)cols * sizeof *x->A);
    if (!x->A) return approx_done(c, x, TR_MATH_MEMORY, error);
    memcpy(x->A, A, (size_t)rows * (size_t)cols * sizeof *x->A);
    uint64_t id = mix(mix(mix(mix(UINT64_C(0x4c4f5752414e4b01), (uint64_t)method), (uint64_t)rank), (uint64_t)rows), (uint64_t)cols);
    for (int32_t i = 0; i < rows * cols; i++) id = mix(id, (uint64_t)A[i]);
    x->id = mix(mix(id, x->seed), (uint64_t)probes);
    x->degree = rank;
    rc = method == TR_APPROX_CROSS ? lowrank_cross(c, x, rank) : lowrank_sketch(c, x, rank);
    if (!rc && !c->err) {
        const tr_bigq *val = norm == TR_APPROX_NORM_FROBENIUS ? x->frob : norm == TR_APPROX_NORM_SPECTRAL ? x->spectral : x->maxentry;
        if (method == TR_APPROX_SKETCH) {
            x->label = TR_APPROX_LABEL_PROBABILISTIC; x->provenance = TR_APPROX_RANDOMIZED;
            x->assumptions = TR_APPROX_ASSUME_INDEPENDENT;
            rc = term_add(c, x, TR_APPROX_TERM_PROBABILISTIC, x->id, 0, val);
        } else rc = term_add(c, x, TR_APPROX_TERM_RESIDUAL, x->id, 0, val);
        error_total(c, x);
        x->bytes = sizeof *x + (uint64_t)rows * (uint64_t)cols * (8 + QBYTES) + (uint64_t)(rank + 1) * 8 * (uint64_t)cols;
        if (!rc && budget_q) {
            qset(&x->budget, qcopy(c, budget_q));
            x->has_budget = 1;
            if (!c->err && tr_bigq_cmp(x->error, budget_q) > 0) rc = TR_MATH_NOT_CONVERGED;
        }
    }
    return approx_done(c, x, rc, error);
}

/* ------------------------------------------------------------------ reading */
int32_t tr_approx_get_info(const tr_approx *x, tr_approx_info *out) {
    if (!approx_valid(x) || !out) return TR_MATH_INVALID;
    memset(out, 0, sizeof *out);
    out->kind = x->kind; out->function = x->function; out->basis = x->basis; out->method = x->method;
    out->norm = x->norm; out->label = x->label; out->provenance = x->provenance; out->assumptions = x->assumptions;
    out->degree = x->degree; out->nterms = x->nterms; out->taylor = x->taylor; out->quantized = x->quantized;
    out->rows = x->rows; out->cols = x->cols; out->rank = x->rank; out->probes = x->probes;
    out->has_budget = x->has_budget; out->has_relative = x->has_relative; out->exact = x->exact;
    out->operation = x->operation; out->seed = x->seed; out->id = x->id;
    out->origins[0] = x->origins[0]; out->origins[1] = x->origins[1]; out->bytes = x->bytes;
    out->error_hi = q_hi(x->error); out->budget_hi = x->budget ? q_hi(x->budget) : 0.0;
    out->failure_hi = x->failure ? q_hi(x->failure) : 0.0;
    out->range_lo = x->range_lo ? q_lo(x->range_lo) : 0.0; out->range_hi = x->range_hi ? q_hi(x->range_hi) : 0.0;
    return TR_MATH_OK;
}
int32_t tr_approx_get_term(const tr_approx *x, int32_t i, tr_approx_term *out) {
    if (!approx_valid(x) || !out || i < 0 || i >= x->nterms) return TR_MATH_INVALID;
    out->kind = x->term_kind[i]; out->scaled = x->term_scaled[i]; out->origin = x->term_origin[i];
    out->hi = q_hi(x->term_q[i]);
    return TR_MATH_OK;
}
const tr_bigq *tr_approx_get_q(const tr_approx *x, int32_t which, int32_t index) {
    if (!approx_valid(x)) return NULL;
    switch (which) {
    case TR_APPROX_Q_ERROR: return x->error;
    case TR_APPROX_Q_TERM: return index >= 0 && index < x->nterms ? x->term_q[index] : NULL;
    case TR_APPROX_Q_BUDGET: return x->budget;
    case TR_APPROX_Q_A: return x->a;
    case TR_APPROX_Q_B: return x->b;
    case TR_APPROX_Q_COEF: return x->coef && index >= 0 && index <= x->degree ? x->coef[index] : NULL;
    case TR_APPROX_Q_LIPSCHITZ: return x->lipschitz;
    case TR_APPROX_Q_DERIVATIVE: return x->derivative;
    case TR_APPROX_Q_RANGE_LO: return x->range_lo;
    case TR_APPROX_Q_RANGE_HI: return x->range_hi;
    case TR_APPROX_Q_RELATIVE: return x->relative;
    case TR_APPROX_Q_L2: return x->l2;
    case TR_APPROX_Q_FROBENIUS2: return x->frob2;
    case TR_APPROX_Q_FROBENIUS: return x->frob;
    case TR_APPROX_Q_SPECTRAL: return x->spectral;
    case TR_APPROX_Q_MAXENTRY: return x->maxentry;
    case TR_APPROX_Q_FAILURE: return x->failure;
    case TR_APPROX_Q_PROBE: return index >= 0 && index < x->probes ? x->probe[index] : NULL;
    case TR_APPROX_Q_ENTRY: return x->entry && index >= 0 && index < x->rows * x->cols ? x->entry[index] : NULL;
    default: return NULL;
    }
}
int32_t tr_approx_get_pivots(const tr_approx *x, int32_t *rows_out, int32_t *cols_out, int32_t capacity) {
    if (!approx_valid(x) || x->kind != TR_APPROX_LOWRANK || x->method != TR_APPROX_CROSS || capacity < x->rank ||
        (x->rank && (!rows_out || !cols_out))) return TR_MATH_INVALID;
    for (int32_t i = 0; i < x->rank; i++) { rows_out[i] = x->prow[i]; cols_out[i] = x->pcol[i]; }
    return TR_MATH_OK;
}
int32_t tr_approx_get_sketch(const tr_approx *x, int8_t *omega, int32_t capacity, int32_t *kept, int32_t kept_capacity) {
    if (!approx_valid(x) || x->kind != TR_APPROX_LOWRANK || x->method != TR_APPROX_SKETCH) return TR_MATH_INVALID;
    int32_t n = x->cols * x->degree;
    if (!omega || capacity < n || !kept || kept_capacity < x->nkept) return TR_MATH_INVALID;
    memcpy(omega, x->omega, (size_t)n);
    for (int32_t i = 0; i < x->nkept; i++) kept[i] = x->kept[i];
    return x->nkept;
}
tr_bigq *tr_approx_value(const tr_approx *x, const tr_bigq *at, int32_t index, const tr_math_budget *budget,
                         tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    if (!approx_valid(x)) { set_error(error, TR_MATH_INVALID); return NULL; }
    tr_math_budget dflt;
    qc ctx = {budget_or(budget, &dflt), refusal, 0}, *c = &ctx;
    if (x->kind == TR_APPROX_LOWRANK) {
        if (index < 0 || index >= x->rows * x->cols) { set_error(error, TR_MATH_INVALID); return NULL; }
        tr_bigq *r = qcopy(c, x->entry[index]);
        set_error(error, r ? TR_MATH_OK : c->err);
        return r;
    }
    if (!q_ok(at)) { set_error(error, TR_MATH_INVALID); return NULL; }
    if (tr_bigq_cmp(at, x->a) < 0 || tr_bigq_cmp(at, x->b) > 0) { set_error(error, TR_MATH_DOMAIN_ERROR); return NULL; }
    tr_bigq *r = NULL;
    if (x->kind == TR_APPROX_FLOAT) {                  /* the polynomial with the binary64 coefficients, exactly */
        r = qcopy(c, x->coef[x->degree]);
        for (int32_t k = x->degree - 1; k >= 0 && !c->err; k--) { qset(&r, qmul(c, r, at)); qacc(c, &r, x->coef[k]); }
    } else {
        tr_bigq *d = qsub(c, at, x->mid), *t = qdiv(c, d, x->rad);
        r = series_at(c, x, t);
        tr_bigq_free(d); tr_bigq_free(t);
    }
    if (c->err) { tr_bigq_free(r); set_error(error, c->err); return NULL; }
    set_error(error, TR_MATH_OK);
    return r;
}
int32_t tr_approx_get_eval_view(const tr_approx *x, tr_approx_eval_view *out) {
    if (!approx_valid(x) || !out || x->kind == TR_APPROX_LOWRANK) return TR_MATH_INVALID;
    *out = x->view;
    return TR_MATH_OK;
}
int32_t tr_approx_eval_doubles(const tr_approx *x, const double *xs, uint64_t n, double *out_value, double *out_bound,
                               int8_t *out_status) {
    if (!approx_valid(x) || x->kind != TR_APPROX_FLOAT || (n && (!xs || !out_value || !out_bound || !out_status))) return TR_MATH_INVALID;
    const tr_approx_eval_view *v = &x->view;
    for (uint64_t i = 0; i < n; i++) {
        double xv = xs[i];
        int32_t s1 = 0, s2 = 0;
        out_value[i] = 0.0; out_bound[i] = 0.0;
        if (!(xv == xv) || tr_bigq_cmp_double(x->a, xv, &s1) || tr_bigq_cmp_double(x->b, xv, &s2) || s1 > 0 || s2 < 0) {
            out_status[i] = TR_APPROX_ROW_DOMAIN; continue;
        }
        double acc = v->coef[v->ncoef - 1].lo;
        for (int32_t k = v->ncoef - 2; k >= 0; k--) acc = fma(acc, xv, v->coef[k].lo);
        if (!(acc > -INFINITY && acc < INFINITY)) { out_status[i] = TR_APPROX_ROW_OVERFLOW; continue; }
        out_value[i] = acc; out_bound[i] = v->err; out_status[i] = TR_APPROX_ROW_OK;
    }
    return TR_MATH_OK;
}

/* ------------------------------------------------------------------ evaluation over rows */
static int view_ok(const tr_approx_eval_view *v) {
    return v && (v->mode == TR_APPROX_EVAL_SERIES || v->mode == TR_APPROX_EVAL_FLOAT) && v->ncoef >= 1 &&
           v->ncoef <= TR_APPROX_MAX_DEGREE + 1 && v->coef && v->rad.lo > 0;
}
int32_t tr_approx_eval_view_rows(const tr_approx_eval_view *v, const tr_scope_view *sv, int32_t col, int32_t fbits,
                                 double *lo, double *hi, int8_t *status, tr_approx_group *groups,
                                 tr_approx_eval_stats *stats) {
    if (!view_ok(v) || !sv || !stats || col < 0 || col >= sv->ncols || fbits < 0 || fbits > TR_APPROX_MAX_FBITS ||
        (sv->rows && !sv->selection)) return TR_MATH_INVALID;
    if (groups && !tr_approx_sums_fit(v, fbits, sv->selected)) return TR_MATH_OVERFLOW;
    memset(stats, 0, sizeof *stats);
    stats->rows = sv->rows; stats->selected = sv->selected; stats->fbits = fbits; stats->groups = (int32_t)sv->ngroups;
    if (groups) memset(groups, 0, (size_t)sv->ngroups * sizeof *groups);
    __int128 *acc = groups ? calloc((size_t)sv->ngroups * 2 + 1, sizeof *acc) : NULL;
    if (groups && !acc) return TR_MATH_MEMORY;
    for (uint64_t row = 0; row < sv->rows; row++) {
        if (!tr_scope_is_selected(sv, row)) {
            if (lo) lo[row] = 0.0;
            if (hi) hi[row] = 0.0;
            if (status) status[row] = TR_APPROX_ROW_UNSELECTED;
            continue;
        }
        tr_math_interval F;
        double val = 0.0;
        int st = tr_approx_eval_row(v, sv, col, row, &F, &val);
        uint32_t g = tr_scope_group(sv, row);
        if (st != TR_APPROX_ROW_OK) { F.lo = F.hi = 0.0; }
        if (lo) lo[row] = F.lo;
        if (hi) hi[row] = F.hi;
        if (status) status[row] = (int8_t)st;
        if (st == TR_APPROX_ROW_OK) stats->ok++;
        else if (st == TR_APPROX_ROW_NULL) stats->nulls++;
        else if (st == TR_APPROX_ROW_DOMAIN) stats->domain++;
        else stats->overflow++;
        if (groups) {
            tr_approx_group *G = &groups[g];
            G->rows++;
            if (st == TR_APPROX_ROW_OK) {
                __int128 a, b;
                tr_approx_fixed(F, fbits, &a, &b);
                acc[2 * (uint64_t)g] += a; acc[2 * (uint64_t)g + 1] += b;
                G->ok++;
            } else if (st == TR_APPROX_ROW_NULL) G->nulls++;
            else G->refused++;
        }
    }
    for (uint64_t g = 0; groups && g < sv->ngroups; g++) {
        tr_math_i128 a = tr_i128_set(acc[2 * g]), b = tr_i128_set(acc[2 * g + 1]);
        groups[g].lo[0] = a.lo; groups[g].lo[1] = a.hi; groups[g].hi[0] = b.lo; groups[g].hi[1] = b.hi;
    }
    free(acc);
    return TR_MATH_OK;
}
int32_t tr_approx_eval_need(const tr_approx *x, const tr_scope *scope, int32_t col, int32_t target, tr_math_need *need) {
    tr_scope_view sv;
    if (!approx_valid(x) || x->kind == TR_APPROX_LOWRANK || !scope || tr_scope_get_view(scope, &sv) || !need ||
        col < 0 || col >= sv.ncols || (target != TR_APPROX_NEED_CPU && target != TR_APPROX_NEED_DEVICE)) return TR_MATH_INVALID;
    need_zero(need);
    need->need[TR_BUDGET_OUTPUT] = sv.rows;
    need->need[TR_BUDGET_BYTES] = target == TR_APPROX_NEED_CPU
        ? tr_budget_add(tr_budget_mul(sv.ngroups, 32 + sizeof(tr_approx_group)), 64)
        : tr_approx_eval_device_bytes(&sv, col, x->view.ncoef);
    return TR_MATH_OK;
}
int32_t tr_approx_eval_rows(const tr_approx *x, const tr_scope *scope, int32_t col, int32_t fbits, double *lo, double *hi,
                            int8_t *status, tr_approx_group *groups, tr_approx_eval_stats *stats,
                            const tr_math_budget *budget, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    tr_math_need need;
    int32_t rc = tr_approx_eval_need(x, scope, col, TR_APPROX_NEED_CPU, &need);
    if (rc) return rc;
    tr_math_budget dflt;
    rc = tr_budget_check(budget_or(budget, &dflt), &need, refusal);
    if (rc) return rc;
    tr_scope_view sv;
    tr_scope_get_view(scope, &sv);
    return tr_approx_eval_view_rows(&x->view, &sv, col, fbits, lo, hi, status, groups, stats);
}

/* ------------------------------------------------------------------ approximate predicates in UNC-001 */
/* x rounded (dir) to a dyadic int64 rational m / 2^s with |m| < 2^62, s <= 62 */
static int q64_dyadic(qc *c, const tr_bigq *x, int dir, tr_num_rational *out) {
    if (c->err || !x) return 0;
    int64_t e = qsgn(x) ? (int64_t)tr_bigint_bit_length(tr_bigq_numerator(x)) - (int64_t)tr_bigint_bit_length(tr_bigq_denominator(x)) : 0;
    if (e > 60) return 0;
    int64_t s = 61 - (e > 0 ? e : 0);
    if (s > 62) s = 62;
    tr_bigq *g = qgrid(c, x, s, dir);
    tr_num_rational q;
    int ok = g && !tr_bigq_to_q64(g, &q);
    tr_bigq_free(g);
    if (ok) *out = q;
    return ok;
}
static int unc_node(tr_unc_node *nodes, int32_t *n, int32_t op, int32_t a, int32_t b, tr_num_rational k) {
    if (*n >= TR_UNC_MAX_NODES) return -1;
    tr_unc_node *nd = &nodes[*n];
    nd->op = op; nd->a = a; nd->b = b; nd->reserved = 0; nd->k = k;
    return (*n)++;
}
/* compare a binary64 enclosure endpoint with T ([Tlo, Thi] its enclosure): 1 certainly op-true, 0 certainly false */
static int filter_row(int32_t op, tr_math_interval F, tr_math_interval T) {
    switch (op) {
    case TR_APPROX_GT: return F.lo > T.hi ? 1 : (F.hi <= T.lo ? 0 : -1);
    case TR_APPROX_GE: return F.lo >= T.hi ? 1 : (F.hi < T.lo ? 0 : -1);
    case TR_APPROX_LT: return F.hi < T.lo ? 1 : (F.lo >= T.hi ? 0 : -1);
    default: return F.hi <= T.lo ? 1 : (F.lo > T.hi ? 0 : -1);
    }
}
int32_t tr_approx_filter(const tr_unc_state *s, const tr_scope *scope, int32_t col, const tr_approx *f, int32_t op,
                         tr_num_rational threshold, uint64_t *bitmap, uint64_t *undecided, tr_approx_filter_stats *stats,
                         const tr_math_budget *budget, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    tr_unc_state_info si;
    tr_scope_view sv;
    if (!s || tr_unc_state_get_info(s, &si) || !si.scoped || !scope || tr_scope_get_view(scope, &sv) || !approx_valid(f) ||
        f->kind != TR_APPROX_SERIES || op < TR_APPROX_GT || op > TR_APPROX_LE || !tr_unc_canonical(threshold) || !stats ||
        col < 0 || col >= si.ncols || (sv.rows && (!bitmap || !undecided))) return TR_MATH_INVALID;
    tr_cert_staleness st;
    int32_t rc = tr_unc_state_fresh(s, scope, &st);
    if (rc) return rc;
    if (f->degree > (TR_UNC_MAX_NODES - 7) / 3) return TR_MATH_UNSUPPORTED;
    tr_math_need need;
    need_zero(&need);
    need.need[TR_BUDGET_CIRCUIT] = (uint64_t)3 * (uint64_t)f->degree + 7;
    need.need[TR_BUDGET_OUTPUT] = si.ngroups;
    need.need[TR_BUDGET_BYTES] = tr_budget_add(tr_budget_mul(si.ngroups, 2 * sizeof(tr_unc_result) + 8),
                                               tr_budget_mul((uint64_t)TR_UNC_MAX_NODES + 1, sizeof(tr_unc_value)));
    tr_math_budget dflt;
    const tr_math_budget *bud = budget_or(budget, &dflt);
    rc = tr_budget_check(bud, &need, refusal);
    if (rc) return rc;
    qc ctx = {bud, refusal, 0}, *c = &ctx;
    /* the program: t = (x - mid) / rad, Horner in t with dyadic int64 coefficients (their rounding added to e) */
    tr_num_rational mid, rad, a64, b64;
    if (tr_bigq_to_q64(f->mid, &mid) || tr_bigq_to_q64(f->rad, &rad) || tr_bigq_to_q64(f->a, &a64) ||
        tr_bigq_to_q64(f->b, &b64)) return TR_MATH_UNSUPPORTED;
    tr_bigq **mt = f->basis == TR_APPROX_CHEBYSHEV ? cheb_to_mono(c, f->coef, f->degree) : NULL;
    tr_bigq *const *m = f->basis == TR_APPROX_CHEBYSHEV ? mt : f->coef;
    tr_num_rational kq[TR_APPROX_MAX_DEGREE + 1];
    tr_bigq *eps = qcopy(c, f->error);
    for (int32_t k = 0; m && k <= f->degree && !c->err; k++) {
        if (!q64_dyadic(c, m[k], 0, &kq[k])) { rc = TR_MATH_UNSUPPORTED; break; }
        tr_bigq *kb = qfrac(c, kq[k].numerator, kq[k].denominator), *d = qsub(c, m[k], kb);
        qaccabs(c, &eps, d);                          /* |t^k| <= 1 */
        tr_bigq_free(kb); tr_bigq_free(d);
    }
    qvec_free(mt, f->degree + 1);
    tr_num_rational tc, tp;                           /* the certain-side and possible-side thresholds */
    if (!rc && !c->err) {
        tr_bigq *T = qfrac(c, threshold.numerator, threshold.denominator), *up = qadd(c, T, eps), *dn = qsub(c, T, eps);
        int gt = op == TR_APPROX_GT || op == TR_APPROX_GE;
        /* f > T certainly when p > T + e (rounded up); possibly when p > T - e (rounded down); mirrored for < */
        if (!q64_dyadic(c, gt ? up : dn, gt ? 1 : -1, &tc) || !q64_dyadic(c, gt ? dn : up, gt ? -1 : 1, &tp)) rc = TR_MATH_UNSUPPORTED;
        tr_bigq_free(T); tr_bigq_free(up); tr_bigq_free(dn);
    }
    double eps_hi = q_hi(eps);
    tr_bigq_free(eps);
    if (!rc) rc = c->err;
    if (rc) return rc;
    tr_unc_node nodes[2][TR_UNC_MAX_NODES];
    tr_unc_program prog[2];
    int32_t cmp = op == TR_APPROX_GT ? TR_UNC_GT : op == TR_APPROX_GE ? TR_UNC_GE : op == TR_APPROX_LT ? TR_UNC_LT : TR_UNC_LE;
    for (int w = 0; w < 2; w++) {
        int32_t n = 0;
        tr_num_rational z = tr_q_int(0);
        int32_t x = unc_node(nodes[w], &n, TR_UNC_COL, col, -1, z);
        int32_t cm = unc_node(nodes[w], &n, TR_UNC_CONST, -1, -1, mid);
        int32_t df = unc_node(nodes[w], &n, TR_UNC_SUB, x, cm, z);
        int32_t t = unc_node(nodes[w], &n, TR_UNC_DIV, df, -1, rad);
        int32_t acc = unc_node(nodes[w], &n, TR_UNC_CONST, -1, -1, kq[f->degree]);
        for (int32_t k = f->degree - 1; k >= 0; k--) {
            int32_t mu = unc_node(nodes[w], &n, TR_UNC_MUL, acc, t, z);
            int32_t ck = unc_node(nodes[w], &n, TR_UNC_CONST, -1, -1, kq[k]);
            acc = unc_node(nodes[w], &n, TR_UNC_ADD, mu, ck, z);
        }
        int32_t th = unc_node(nodes[w], &n, TR_UNC_CONST, -1, -1, w == 0 ? tc : tp);
        int32_t root = unc_node(nodes[w], &n, cmp, acc, th, z);
        if (root < 0) return TR_MATH_UNSUPPORTED;
        memset(&prog[w], 0, sizeof prog[w]);
        prog[w].nnodes = n; prog[w].ncols = si.ncols; prog[w].carrier = TR_UNC_BINARY64; prog[w].precision = 53;
        prog[w].rank = TR_UNC_MAX_TERMS; prog[w].domains = TR_UNC_DOM_ALL; prog[w].root = root;
        prog[w].budget = tr_unc_qv(0, 0); prog[w].nodes = nodes[w];
    }
    uint64_t G = si.ngroups;
    tr_unc_result *res = calloc((size_t)G * 2 + 1, sizeof *res);
    int32_t *dec = calloc((size_t)G + 1, sizeof *dec);
    if (!res || !dec) { free(res); free(dec); return TR_MATH_MEMORY; }
    rc = tr_unc_eval(s, &prog[0], 0, G, res, bud, refusal);
    if (!rc) rc = tr_unc_eval(s, &prog[1], 0, G, res + G, bud, refusal);
    memset(stats, 0, sizeof *stats);
    stats->rows = sv.rows; stats->selected = sv.selected; stats->degree = f->degree; stats->error_used = eps_hi;
    const tr_unc_input *in = tr_unc_state_inputs(s);
    for (uint64_t g = 0; !rc && g < G; g++) {
        const tr_unc_input *ix = &in[g * (uint64_t)si.ncols + (uint64_t)col];
        if (!ix->rows) { dec[g] = TR_UNC_EMPTY; stats->groups_empty++; continue; }
        if (!ix->present) { dec[g] = TR_UNC_NONE; stats->groups_none++; continue; }   /* every row NULL: never TRUE */
        tr_num_rational lo, hi;
        int64_t ten = tr_unc_pow10(ix->scale);
        int inside = tr_q_make(ix->min, ten, &lo) && tr_q_make(ix->max, ten, &hi) && tr_q_cmp(lo, a64) >= 0 && tr_q_cmp(hi, b64) <= 0;
        if (!inside) { dec[g] = TR_UNC_UNKNOWN; stats->groups_outside++; continue; }
        if (res[g].decision == TR_UNC_ALL) { dec[g] = TR_UNC_ALL; stats->groups_all++; }
        else if (res[G + g].decision == TR_UNC_NONE) { dec[g] = TR_UNC_NONE; stats->groups_none++; }
        else { dec[g] = TR_UNC_UNKNOWN; stats->groups_unknown++; }
    }
    free(res);
    if (!rc) {
        tr_math_interval T = tr_unc_enclose(threshold);
        int32_t scol = si.cols[col];
        const tr_approx_eval_view *v = &f->view;
        uint64_t nw = (sv.rows + 63) / 64;
        for (uint64_t w = 0; w < nw; w++) { bitmap[w] = 0; undecided[w] = 0; }
        for (uint64_t row = 0; row < sv.rows; row++) {
            if (!tr_scope_is_selected(&sv, row)) continue;
            int32_t d = dec[tr_scope_group(&sv, row)];
            if (d == TR_UNC_ALL) { bitmap[row >> 6] |= (uint64_t)1 << (row & 63); stats->rows_proven++; stats->rows_true++; continue; }
            if (d == TR_UNC_NONE) { stats->rows_proven++; continue; }
            stats->rows_evaluated++;
            tr_math_interval F;
            double val;
            int stt = tr_approx_eval_row(v, &sv, scol, row, &F, &val);
            if (stt == TR_APPROX_ROW_NULL) { stats->rows_null++; continue; }
            if (stt != TR_APPROX_ROW_OK) {
                undecided[row >> 6] |= (uint64_t)1 << (row & 63); stats->rows_undecided++;
                if (stt == TR_APPROX_ROW_DOMAIN) stats->rows_domain++;
                continue;
            }
            int r = filter_row(op, F, T);
            if (r == 1) { bitmap[row >> 6] |= (uint64_t)1 << (row & 63); stats->rows_true++; }
            else if (r == 0) stats->rows_false++;
            else { undecided[row >> 6] |= (uint64_t)1 << (row & 63); stats->rows_undecided++; }
        }
    }
    free(dec);
    return rc;
}

/* ------------------------------------------------------------------ FND-007 */
int32_t tr_approx_authorize_exact(const tr_approx *x, uint64_t field, tr_cert_decision *out) {
    if (!approx_valid(x) || !out) return TR_MATH_INVALID;
    memset(out, 0, sizeof *out);
    out->authorized = 0;
    out->reason = x->label == TR_APPROX_LABEL_PROBABILISTIC ? TR_CERT_REFUSE_STATISTICAL : TR_CERT_REFUSE_APPROXIMATE;
    out->kind = x->label; out->claim = 0; out->policy = 0; out->premise = -1; out->field = field;
    out->stale.column = -1; out->stale.dim = -1;
    return TR_MATH_DOMAIN_ERROR;
}

tr_cert *tr_approx_relation(const tr_approx *f, const tr_scope *scope, int32_t xcol, int32_t ycol,
                            const tr_bigq *data_tolerance, int32_t bits, tr_cert_evidence *evidence,
                            const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    tr_scope_view sv;
    if (!approx_valid(f) || (f->kind != TR_APPROX_SERIES && f->kind != TR_APPROX_FLOAT) || !scope ||
        tr_scope_get_view(scope, &sv) || xcol < 0 || ycol < 0 || xcol >= sv.ncols || ycol >= sv.ncols || xcol == ycol ||
        bits < 0 || bits > 60 || (data_tolerance && (!q_ok(data_tolerance) || tr_bigq_sign(data_tolerance) < 0))) {
        set_error(error, TR_MATH_INVALID); return NULL;
    }
    int32_t n = f->degree, sx = sv.col[xcol].scale, sy = sv.col[ycol].scale;
    if (n > 16 || (int64_t)sx * n + sy > 18) { set_error(error, TR_MATH_UNSUPPORTED); return NULL; }
    tr_math_budget dflt;
    const tr_math_budget *bud = budget_or(budget, &dflt);
    qc ctx = {bud, refusal, 0}, *c = &ctx;
    /* p as monomials in x (exact), each to a multiple of 2^-bits: delta = sum |a_j - m_j 2^-bits| B^j */
    tr_bigq **ax = NULL;
    if (f->kind == TR_APPROX_FLOAT) {
        ax = qvec(c, n + 1);
        for (int32_t j = 0; ax && j <= n; j++) qset(&ax[j], qcopy(c, f->coef[j]));
    } else {
        tr_bigq **mt = f->basis == TR_APPROX_CHEBYSHEV ? cheb_to_mono(c, f->coef, n) : NULL;
        tr_bigq *one = qint(c, 1), *alpha = qdiv(c, one, f->rad), *mm = qneg(c, f->mid), *beta = qmul(c, mm, alpha);
        ax = mono_substitute(c, f->basis == TR_APPROX_CHEBYSHEV ? mt : f->coef, n, alpha, beta);
        qvec_free(mt, n + 1);
        tr_bigq_free(one); tr_bigq_free(alpha); tr_bigq_free(mm); tr_bigq_free(beta);
    }
    tr_bigq *aa = qabs(c, f->a), *bb = qabs(c, f->b), *B = qmax(c, aa, bb), *pw = qint(c, 1), *delta = qint(c, 0);
    int64_t mcoef[17];
    int32_t rc = 0;
    for (int32_t j = 0; ax && j <= n && !c->err; j++) {
        tr_bigq *g = qgrid(c, ax[j], bits, 0), *d = qsub(c, ax[j], g), *ad = qabs(c, d), *t = qmul(c, ad, pw);
        qacc(c, &delta, t);
        tr_bigq *p2 = q2pow(c, bits), *mq = qmul(c, g, p2);
        tr_num_rational q;
        if (!mq || tr_bigq_to_q64(mq, &q) || q.denominator != 1) rc = TR_MATH_UNSUPPORTED;
        else mcoef[j] = q.numerator;
        tr_bigq_free(g); tr_bigq_free(d); tr_bigq_free(ad); tr_bigq_free(t); tr_bigq_free(p2); tr_bigq_free(mq);
        qset(&pw, qmul(c, pw, B));
    }
    /* D (y - p(x)) = 2^bits 10^(sx n) Y - sum_j m_j 10^(sy + sx (n - j)) X^j with D = 10^sy 2^bits 10^(sx n) */
    int64_t coef[18];
    uint8_t exps[18 * 2];
    uint64_t nt = 0, tol = 0;
    if (!rc && !c->err) {
        __int128 yc = ((__int128)1 << bits) * (__int128)tr_unc_pow10(sx * n);
        if (yc > INT64_MAX) rc = TR_MATH_UNSUPPORTED;
        for (int32_t j = 0; !rc && j <= n; j++) {
            __int128 v = -(__int128)mcoef[j] * tr_unc_pow10(sy + sx * (n - j));
            if (v > INT64_MAX || v < INT64_MIN) { rc = TR_MATH_UNSUPPORTED; break; }
            if (!v) continue;
            coef[nt] = (int64_t)v; exps[2 * nt] = (uint8_t)j; exps[2 * nt + 1] = 0; nt++;
        }
        if (!rc) { coef[nt] = (int64_t)yc; exps[2 * nt] = 0; exps[2 * nt + 1] = 1; nt++; }
        /* the tolerance: ceil(D (e + delta + data tolerance)), at least 1 (the claim is always approximate) */
        tr_bigq *e = qadd(c, f->error, delta);
        if (data_tolerance) qacc(c, &e, data_tolerance);
        tr_bigq *D = q2pow(c, bits);
        qset(&D, qmulint(c, D, tr_unc_pow10(sy)));
        qset(&D, qmulint(c, D, tr_unc_pow10(sx * n)));
        tr_bigq *t = qmul(c, D, e), *tg = qgrid(c, t, 0, 1);
        tr_num_rational q;
        if (!rc && (!tg || tr_bigq_to_q64(tg, &q) || q.denominator != 1 || q.numerator < 0)) rc = TR_MATH_UNSUPPORTED;
        else if (!rc) tol = q.numerator < 1 ? 1 : (uint64_t)q.numerator;
        tr_bigq_free(e); tr_bigq_free(D); tr_bigq_free(t); tr_bigq_free(tg);
    }
    qvec_free(ax, n + 1);
    tr_bigq_free(aa); tr_bigq_free(bb); tr_bigq_free(B); tr_bigq_free(pw); tr_bigq_free(delta);
    if (!rc) rc = c->err;
    if (rc) { set_error(error, rc); return NULL; }
    uint8_t cexps[18 * 2];
    int64_t ccoef[18];
    uint64_t cterms = 0;
    rc = tr_cert_poly_canonical(2, nt, exps, coef, 0, 0, cexps, ccoef, &cterms);
    if (rc) { set_error(error, rc); return NULL; }
    tr_cert_claim claim;
    memset(&claim, 0, sizeof claim);
    claim.kind = TR_CLAIM_POLYNOMIAL; claim.policy = TR_CERT_NULL_SKIP; claim.ncols = 2;
    for (int i = 0; i < TR_CERT_MAX_COLS; i++) claim.col[i] = -1;
    claim.col[0] = xcol; claim.col[1] = ycol;
    claim.carrier = TR_CARRIER_I128; claim.field = 0; claim.tolerance = tol;
    claim.nterms = cterms; claim.exponents = cexps; claim.coefficients = ccoef;
    return tr_cert_snapshot(scope, &claim, NULL, 0, evidence, bud, refusal, error);
}
