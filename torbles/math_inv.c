/* INV-001: linear and univariate polynomial inverse answers; every contract is in math_inv.h.
 * SPDX-License-Identifier: Apache-2.0 */
#include "math_inv.h"
#include "math_handles.h"
#include <stdlib.h>
#include <string.h>

TR_MATH_ABI(tr_inv_linear_info, TR_INV_ABI);      /* the ctypes boundary of INV-001 (math_handles.h) */
TR_MATH_ABI(tr_inv_roots_info, TR_INV_ABI);
TR_MATH_ABI(tr_inv_root_info, TR_INV_ABI);
TR_MATH_ABI(tr_inv_signs_view, TR_INV_ABI);
TR_MATH_ABI(tr_inv_sweep_view, TR_INV_ABI);
TR_MATH_ABI(tr_inv_sweep_info, TR_INV_ABI);

_Static_assert(sizeof(tr_inv_linear_info) == 40 && sizeof(tr_inv_roots_info) == 48 && sizeof(tr_inv_root_info) == 24 &&
               sizeof(tr_inv_signs_view) == 96 && sizeof(tr_inv_sweep_view) == 128 && sizeof(tr_inv_sweep_info) == 64,
               "INV-001 ABI");

#define LINEAR_MAGIC 0x31564e49u   /* "INV1" */
#define ROOTS_MAGIC 0x31544f52u    /* "ROT1" */
#define SIGNS_MAGIC 0x31474953u    /* "SIG1" */
#define PLAN_MAGIC 0x31505753u     /* "SWP1" */
#define SWEEP_MAGIC 0x31525753u    /* "SWR1" */

/* the 128 largest primes below 2^62 (deterministic Miller-Rabin, bases 2..37), decreasing: every one in (2^61, 2^62),
 * so a product of k of them exceeds 2^(61 k) and p_t < 2 p_j for any two of them */
static const uint64_t TR_INV_PRIMES[TR_INV_PRIMES_MAX] = {
    0x3fffffffffffffc7ULL, 0x3fffffffffffffa9ULL, 0x3fffffffffffff8bULL, 0x3fffffffffffff71ULL,
    0x3fffffffffffff67ULL, 0x3fffffffffffff59ULL, 0x3fffffffffffff55ULL, 0x3fffffffffffff3dULL,
    0x3fffffffffffff35ULL, 0x3ffffffffffffeefULL, 0x3ffffffffffffee1ULL, 0x3ffffffffffffec3ULL,
    0x3ffffffffffffe45ULL, 0x3ffffffffffffe1dULL, 0x3ffffffffffffe11ULL, 0x3ffffffffffffdc1ULL,
    0x3ffffffffffffdbbULL, 0x3ffffffffffffda5ULL, 0x3ffffffffffffd87ULL, 0x3ffffffffffffd69ULL,
    0x3ffffffffffffd03ULL, 0x3ffffffffffffcfbULL, 0x3ffffffffffffcf7ULL, 0x3ffffffffffffce9ULL,
    0x3ffffffffffffcd3ULL, 0x3ffffffffffffcc1ULL, 0x3ffffffffffffc65ULL, 0x3ffffffffffffc2bULL,
    0x3ffffffffffffc1fULL, 0x3ffffffffffffc17ULL, 0x3ffffffffffffc11ULL, 0x3ffffffffffffc07ULL,
    0x3ffffffffffffb53ULL, 0x3ffffffffffffb27ULL, 0x3ffffffffffffaf3ULL, 0x3ffffffffffffab7ULL,
    0x3ffffffffffffa67ULL, 0x3ffffffffffffa15ULL, 0x3ffffffffffff9efULL, 0x3ffffffffffff9d9ULL,
    0x3ffffffffffff9d3ULL, 0x3ffffffffffff9c5ULL, 0x3ffffffffffff9afULL, 0x3ffffffffffff977ULL,
    0x3ffffffffffff95fULL, 0x3ffffffffffff95bULL, 0x3ffffffffffff959ULL, 0x3ffffffffffff8e1ULL,
    0x3ffffffffffff8a7ULL, 0x3ffffffffffff889ULL, 0x3ffffffffffff87dULL, 0x3ffffffffffff805ULL,
    0x3ffffffffffff7e7ULL, 0x3ffffffffffff7c9ULL, 0x3ffffffffffff7a3ULL, 0x3ffffffffffff775ULL,
    0x3ffffffffffff757ULL, 0x3ffffffffffff739ULL, 0x3ffffffffffff713ULL, 0x3ffffffffffff6d1ULL,
    0x3ffffffffffff6c1ULL, 0x3ffffffffffff6b9ULL, 0x3ffffffffffff6a3ULL, 0x3ffffffffffff68bULL,
    0x3ffffffffffff631ULL, 0x3ffffffffffff613ULL, 0x3ffffffffffff5e9ULL, 0x3ffffffffffff59bULL,
    0x3ffffffffffff58dULL, 0x3ffffffffffff53fULL, 0x3ffffffffffff527ULL, 0x3ffffffffffff517ULL,
    0x3ffffffffffff4d3ULL, 0x3ffffffffffff4b5ULL, 0x3ffffffffffff491ULL, 0x3ffffffffffff431ULL,
    0x3ffffffffffff41fULL, 0x3ffffffffffff36bULL, 0x3ffffffffffff34dULL, 0x3ffffffffffff349ULL,
    0x3ffffffffffff347ULL, 0x3ffffffffffff341ULL, 0x3ffffffffffff30bULL, 0x3ffffffffffff2cfULL,
    0x3ffffffffffff23fULL, 0x3ffffffffffff22fULL, 0x3ffffffffffff227ULL, 0x3ffffffffffff221ULL,
    0x3ffffffffffff215ULL, 0x3ffffffffffff1a9ULL, 0x3ffffffffffff187ULL, 0x3ffffffffffff149ULL,
    0x3ffffffffffff12bULL, 0x3ffffffffffff125ULL, 0x3ffffffffffff0dfULL, 0x3ffffffffffff0a3ULL,
    0x3fffffffffffefbdULL, 0x3fffffffffffef69ULL, 0x3fffffffffffef4dULL, 0x3fffffffffffef33ULL,
    0x3fffffffffffeee7ULL, 0x3fffffffffffeecdULL, 0x3fffffffffffee7bULL, 0x3fffffffffffee33ULL,
    0x3fffffffffffee0dULL, 0x3fffffffffffeddfULL, 0x3fffffffffffedcbULL, 0x3fffffffffffed9dULL,
    0x3fffffffffffed53ULL, 0x3fffffffffffed31ULL, 0x3fffffffffffed2bULL, 0x3fffffffffffed07ULL,
    0x3fffffffffffecefULL, 0x3fffffffffffeccbULL, 0x3fffffffffffecb3ULL, 0x3fffffffffffec95ULL,
    0x3fffffffffffec81ULL, 0x3fffffffffffec7bULL, 0x3fffffffffffec75ULL, 0x3fffffffffffec41ULL,
    0x3fffffffffffec11ULL, 0x3fffffffffffebf3ULL, 0x3fffffffffffebdfULL, 0x3fffffffffffeb6fULL,
    0x3fffffffffffeb15ULL, 0x3fffffffffffeaefULL, 0x3fffffffffffeabbULL, 0x3fffffffffffeaa7ULL,
};

int32_t tr_inv_primes_verify(void) {
    for (int32_t i = 0; i < TR_INV_PRIMES_MAX; i++) {
        uint64_t p = TR_INV_PRIMES[i];
        if (p <= (UINT64_C(1) << 61) || p >= (UINT64_C(1) << 62) || tr_gfp_is_prime(p) != 1) return 0;
        if (i && p >= TR_INV_PRIMES[i - 1]) return 0;
    }
    return 1;
}
int32_t tr_inv_prime(int32_t i, uint64_t *p) {
    if (i < 0 || i >= TR_INV_PRIMES_MAX || !p) return TR_MATH_INVALID;
    *p = TR_INV_PRIMES[i];
    return TR_MATH_OK;
}

/* ================================================================ a sticky error context over the carriers */
typedef struct {
    const tr_math_budget *b;
    tr_math_refusal *r;
    int32_t err;
    uint64_t iters;              /* exact evaluations / generated inequalities, against limit[ITERATIONS] */
    tr_math_budget dflt;
} ctx_t;

static void ctx_init(ctx_t *c, const tr_math_budget *b, tr_math_refusal *r) {
    tr_budget_none(r);
    tr_budget_default(&c->dflt);
    c->b = b ? b : &c->dflt;
    c->r = r; c->err = 0; c->iters = 0;
}
static void cfail(ctx_t *c, int32_t e) { if (!c->err) c->err = e ? e : TR_MATH_MEMORY; }
/* n more iterations; TR_MATH_BUDGET (resource iterations, need = the count) past the limit */
static int tick(ctx_t *c, uint64_t n) {
    if (c->err) return 0;
    c->iters = tr_budget_add(c->iters, n);
    if (c->iters > c->b->limit[TR_BUDGET_ITERATIONS]) {
        if (c->r) { c->r->resource = TR_BUDGET_ITERATIONS; c->r->refused = 1; c->r->need = c->iters;
                    c->r->limit = c->b->limit[TR_BUDGET_ITERATIONS]; }
        c->err = TR_MATH_BUDGET;
        return 0;
    }
    return 1;
}
static int check_one(ctx_t *c, int32_t resource, uint64_t need) {
    if (c->err) return 0;
    int32_t rc = tr_budget_check_one(c->b, resource, need, c->r);
    if (rc) { c->err = rc; return 0; }
    return 1;
}
static void *cmalloc(ctx_t *c, size_t n) {
    if (c->err) return NULL;
    void *p = calloc(1, n ? n : 1);
    if (!p) cfail(c, TR_MATH_MEMORY);
    return p;
}

#define BI_RET(expr) do { if (c->err) return NULL; int32_t e_ = 0; tr_bigint *x_ = (expr); if (!x_) cfail(c, e_); return x_; } while (0)
#define BQ_RET(expr) do { if (c->err) return NULL; int32_t e_ = 0; tr_bigq *x_ = (expr); if (!x_) cfail(c, e_); return x_; } while (0)
static tr_bigint *B_i64(ctx_t *c, int64_t v) { BI_RET(tr_bigint_from_i64(v, c->b, c->r, &e_)); }
static tr_bigint *B_copy(ctx_t *c, const tr_bigint *a) { BI_RET(tr_bigint_copy(a, c->b, c->r, &e_)); }
static tr_bigint *B_neg(ctx_t *c, const tr_bigint *a) { BI_RET(tr_bigint_neg(a, c->b, c->r, &e_)); }
static tr_bigint *B_abs(ctx_t *c, const tr_bigint *a) { BI_RET(tr_bigint_abs(a, c->b, c->r, &e_)); }
static tr_bigint *B_add(ctx_t *c, const tr_bigint *a, const tr_bigint *x) { BI_RET(tr_bigint_add(a, x, c->b, c->r, &e_)); }
static tr_bigint *B_sub(ctx_t *c, const tr_bigint *a, const tr_bigint *x) { BI_RET(tr_bigint_sub(a, x, c->b, c->r, &e_)); }
static tr_bigint *B_mul(ctx_t *c, const tr_bigint *a, const tr_bigint *x) { BI_RET(tr_bigint_mul(a, x, c->b, c->r, &e_)); }
static tr_bigint *B_gcd(ctx_t *c, const tr_bigint *a, const tr_bigint *x) { BI_RET(tr_bigint_gcd(a, x, c->b, c->r, &e_)); }
static tr_bigint *B_divexact(ctx_t *c, const tr_bigint *a, const tr_bigint *x) {
    BI_RET(tr_bigint_divexact(a, x, c->b, c->r, &e_));
}
static tr_bigint *B_pow(ctx_t *c, const tr_bigint *a, uint64_t e) { BI_RET(tr_bigint_pow(a, e, c->b, c->r, &e_)); }
static tr_bigint *B_shl(ctx_t *c, const tr_bigint *a, uint64_t k) { BI_RET(tr_bigint_shl(a, k, c->b, c->r, &e_)); }
static tr_bigint *B_floordiv(ctx_t *c, const tr_bigint *a, const tr_bigint *x) {
    if (c->err) return NULL;
    tr_bigint *q = NULL;
    int32_t rc = tr_bigint_divmod(a, x, TR_DIV_FLOOR, &q, NULL, c->b, c->r);
    if (rc) { cfail(c, rc); return NULL; }
    return q;
}
static tr_bigint *B_mul_i64(ctx_t *c, const tr_bigint *a, int64_t k) {
    tr_bigint *t = B_i64(c, k), *x = B_mul(c, a, t);
    tr_bigint_free(t);
    return x;
}
static tr_bigq *Q_i64(ctx_t *c, int64_t v) { tr_num_rational q = {v, 1}; BQ_RET(tr_bigq_from_q64(q, c->b, c->r, &e_)); }
static tr_bigq *Q_frac(ctx_t *c, const tr_bigint *n, const tr_bigint *d) { BQ_RET(tr_bigq_from_bigints(n, d, c->b, c->r, &e_)); }
static tr_bigq *Q_int(ctx_t *c, const tr_bigint *n) { BQ_RET(tr_bigq_from_bigint(n, c->b, c->r, &e_)); }
static tr_bigq *Q_copy(ctx_t *c, const tr_bigq *a) { BQ_RET(tr_bigq_copy(a, c->b, c->r, &e_)); }
static tr_bigq *Q_neg(ctx_t *c, const tr_bigq *a) { BQ_RET(tr_bigq_neg(a, c->b, c->r, &e_)); }
static tr_bigq *Q_add(ctx_t *c, const tr_bigq *a, const tr_bigq *x) { BQ_RET(tr_bigq_add(a, x, c->b, c->r, &e_)); }
static tr_bigq *Q_sub(ctx_t *c, const tr_bigq *a, const tr_bigq *x) { BQ_RET(tr_bigq_sub(a, x, c->b, c->r, &e_)); }
static tr_bigq *Q_mul(ctx_t *c, const tr_bigq *a, const tr_bigq *x) { BQ_RET(tr_bigq_mul(a, x, c->b, c->r, &e_)); }
static tr_bigq *Q_div(ctx_t *c, const tr_bigq *a, const tr_bigq *x) { BQ_RET(tr_bigq_div(a, x, c->b, c->r, &e_)); }
static int Q_sign(const tr_bigq *q) { return q ? tr_bigq_sign(q) : 0; }
static int Q_cmp(ctx_t *c, const tr_bigq *a, const tr_bigq *x) {
    if (c->err) return 0;
    int32_t r = tr_bigq_cmp(a, x);
    if (r < -1 || r > 1) { cfail(c, r); return 0; }
    return r;
}
static int Q_zero(const tr_bigq *q) { return !q || tr_bigq_sign(q) == 0; }
/* (a + b) / 2 */
static tr_bigq *Q_mid(ctx_t *c, const tr_bigq *a, const tr_bigq *x) {
    tr_bigq *s = Q_add(c, a, x), *two = Q_i64(c, 2), *m = Q_div(c, s, two);
    tr_bigq_free(s); tr_bigq_free(two);
    return m;
}
static tr_bigint *bset(tr_bigint *old, tr_bigint *v) { tr_bigint_free(old); return v; }
static tr_bigq *qset(tr_bigq *old, tr_bigq *v) { tr_bigq_free(old); return v; }
static int bsign(const tr_bigint *x) { return x ? tr_bigint_sign(x) : 0; }

/* ================================================================ integer polynomials over tr_bigint */
typedef struct { int32_t deg; tr_bigint **c; } ipoly;   /* deg -1: the zero polynomial */

static void ip_free(ipoly *p) {
    if (p->c) { for (int32_t i = 0; i <= p->deg; i++) tr_bigint_free(p->c[i]); free(p->c); }
    p->c = NULL; p->deg = -1;
}
static int ip_alloc(ctx_t *c, ipoly *p, int32_t deg) {
    p->deg = -1; p->c = NULL;
    if (deg < 0) return !c->err;
    p->c = (tr_bigint **)cmalloc(c, (size_t)(deg + 1) * sizeof *p->c);
    if (!p->c) return 0;
    p->deg = deg;
    return 1;
}
/* drops zero leading coefficients (and NULL entries, which only occur after an error) */
static void ip_trim(ipoly *p) {
    while (p->deg >= 0 && bsign(p->c[p->deg]) == 0) { tr_bigint_free(p->c[p->deg]); p->c[p->deg] = NULL; p->deg--; }
}
static int ip_ok(ctx_t *c, ipoly *p) {
    if (c->err) return 0;
    for (int32_t i = 0; i <= p->deg; i++) if (!p->c[i]) { cfail(c, TR_MATH_MEMORY); return 0; }
    return 1;
}
static ipoly ip_copy(ctx_t *c, const ipoly *a) {
    ipoly r;
    if (!ip_alloc(c, &r, a->deg)) return r;
    for (int32_t i = 0; i <= a->deg; i++) r.c[i] = B_copy(c, a->c[i]);
    ip_ok(c, &r);
    return r;
}
static ipoly ip_deriv(ctx_t *c, const ipoly *a) {
    ipoly r;
    if (!ip_alloc(c, &r, a->deg - 1)) return r;
    for (int32_t i = 1; i <= a->deg; i++) r.c[i - 1] = B_mul_i64(c, a->c[i], i);
    if (ip_ok(c, &r)) ip_trim(&r);
    return r;
}
static ipoly ip_sub(ctx_t *c, const ipoly *a, const ipoly *x) {
    int32_t d = a->deg > x->deg ? a->deg : x->deg;
    ipoly r;
    if (!ip_alloc(c, &r, d)) return r;
    for (int32_t i = 0; i <= d; i++) {
        if (i <= a->deg && i <= x->deg) r.c[i] = B_sub(c, a->c[i], x->c[i]);
        else if (i <= a->deg) r.c[i] = B_copy(c, a->c[i]);
        else r.c[i] = B_neg(c, x->c[i]);
    }
    if (ip_ok(c, &r)) ip_trim(&r);
    return r;
}
static ipoly ip_mul(ctx_t *c, const ipoly *a, const ipoly *x) {
    ipoly r;
    r.deg = -1; r.c = NULL;
    if (a->deg < 0 || x->deg < 0) return r;
    if (!ip_alloc(c, &r, a->deg + x->deg)) return r;
    for (int32_t i = 0; i <= r.deg; i++) r.c[i] = B_i64(c, 0);
    for (int32_t i = 0; i <= a->deg && !c->err; i++)
        for (int32_t j = 0; j <= x->deg && !c->err; j++) {
            tr_bigint *t = B_mul(c, a->c[i], x->c[j]);
            r.c[i + j] = bset(r.c[i + j], B_add(c, r.c[i + j], t));
            tr_bigint_free(t);
        }
    if (ip_ok(c, &r)) ip_trim(&r);
    return r;
}
/* gcd of the coefficients (>= 0; 0 for the zero polynomial) */
static tr_bigint *ip_content(ctx_t *c, const ipoly *a) {
    tr_bigint *g = B_i64(c, 0);
    for (int32_t i = 0; i <= a->deg && !c->err; i++) {
        g = bset(g, B_gcd(c, g, a->c[i]));
        if (g && tr_bigint_bit_length(g) == 1) break;            /* 1 */
    }
    return g;
}
/* a / content(a), keeping the sign (positive_lc: also make the leading coefficient positive) */
static void ip_primitive(ctx_t *c, ipoly *a, int positive_lc) {
    if (a->deg < 0 || c->err) return;
    tr_bigint *g = ip_content(c, a);
    if (!g) return;
    int negate = positive_lc && bsign(a->c[a->deg]) < 0;
    if (tr_bigint_bit_length(g) > 1)
        for (int32_t i = 0; i <= a->deg; i++) a->c[i] = bset(a->c[i], B_divexact(c, a->c[i], g));
    if (negate)
        for (int32_t i = 0; i <= a->deg; i++) a->c[i] = bset(a->c[i], B_neg(c, a->c[i]));
    tr_bigint_free(g);
    ip_ok(c, a);
}
/* the pseudo-remainder lc(x)^(deg a - deg x + 1) a mod x (deg x >= 0) */
static ipoly ip_prem(ctx_t *c, const ipoly *a, const ipoly *x) {
    ipoly r = ip_copy(c, a);
    if (c->err || x->deg < 0) return r;
    int32_t steps = a->deg - x->deg + 1, done = 0;
    const tr_bigint *l = x->c[x->deg];
    while (r.deg >= x->deg && !c->err) {
        int32_t shift = r.deg - x->deg;
        tr_bigint *lr = B_copy(c, r.c[r.deg]);
        for (int32_t i = 0; i <= r.deg && !c->err; i++) r.c[i] = bset(r.c[i], B_mul(c, r.c[i], l));
        for (int32_t i = 0; i <= x->deg && !c->err; i++) {
            tr_bigint *t = B_mul(c, lr, x->c[i]);
            r.c[i + shift] = bset(r.c[i + shift], B_sub(c, r.c[i + shift], t));
            tr_bigint_free(t);
        }
        tr_bigint_free(lr);
        if (!ip_ok(c, &r)) break;
        ip_trim(&r);
        done++;
    }
    if (!c->err && steps - done > 0 && r.deg >= 0) {
        tr_bigint *f = B_pow(c, l, (uint64_t)(steps - done));
        for (int32_t i = 0; i <= r.deg && !c->err; i++) r.c[i] = bset(r.c[i], B_mul(c, r.c[i], f));
        tr_bigint_free(f);
        ip_ok(c, &r);
    }
    return r;
}
/* the primitive gcd with a positive leading coefficient (primitive PRS); gcd(0, 0) = 0 */
static ipoly ip_gcd(ctx_t *c, const ipoly *a, const ipoly *x) {
    ipoly A = ip_copy(c, a), B = ip_copy(c, x);
    ip_primitive(c, &A, 1); ip_primitive(c, &B, 1);
    if (A.deg < B.deg) { ipoly t = A; A = B; B = t; }
    while (B.deg >= 0 && !c->err) {
        ipoly R = ip_prem(c, &A, &B);
        ip_free(&A);
        A = B;
        ip_primitive(c, &R, 1);
        B = R;
    }
    ip_free(&B);
    ip_primitive(c, &A, 1);
    return A;
}
/* a / x for x | a in Z[x] (NO_SOLUTION from the carriers if it does not divide: an internal error) */
static ipoly ip_divexact(ctx_t *c, const ipoly *a, const ipoly *x) {
    ipoly q;
    q.deg = -1; q.c = NULL;
    if (c->err || a->deg < 0) return q;
    if (x->deg < 0 || a->deg < x->deg) { cfail(c, TR_MATH_INVALID); return q; }
    ipoly r = ip_copy(c, a);
    if (!ip_alloc(c, &q, a->deg - x->deg)) { ip_free(&r); return q; }
    const tr_bigint *l = x->c[x->deg];
    for (int32_t k = a->deg - x->deg; k >= 0 && !c->err; k--) {
        q.c[k] = B_divexact(c, r.c[x->deg + k], l);
        for (int32_t i = 0; i <= x->deg && !c->err; i++) {
            tr_bigint *t = B_mul(c, q.c[k], x->c[i]);
            r.c[i + k] = bset(r.c[i + k], B_sub(c, r.c[i + k], t));
            tr_bigint_free(t);
        }
    }
    if (ip_ok(c, &r)) { ip_trim(&r); if (r.deg >= 0) cfail(c, TR_MATH_INVALID); }
    ip_free(&r);
    if (ip_ok(c, &q)) ip_trim(&q);
    return q;
}
/* the sign of v^deg a(u/v) (v > 0): homogeneous Horner, one iteration */
static int ip_sign_uv(ctx_t *c, const ipoly *a, const tr_bigint *u, const tr_bigint *v) {
    if (a->deg < 0 || !tick(c, 1)) return 0;
    tr_bigint *h = B_copy(c, a->c[a->deg]), *vp = NULL;
    int one = tr_bigint_bit_length(v) == 1;
    for (int32_t i = a->deg - 1; i >= 0 && !c->err; i--) {
        h = bset(h, B_mul(c, h, u));
        if (one) h = bset(h, B_add(c, h, a->c[i]));
        else {
            vp = vp ? bset(vp, B_mul(c, vp, v)) : B_copy(c, v);
            tr_bigint *t = B_mul(c, a->c[i], vp);
            h = bset(h, B_add(c, h, t));
            tr_bigint_free(t);
        }
    }
    int s = c->err ? 0 : bsign(h);
    tr_bigint_free(h); tr_bigint_free(vp);
    return s;
}
static int ip_sign_q(ctx_t *c, const ipoly *a, const tr_bigq *x) {
    return ip_sign_uv(c, a, tr_bigq_numerator(x), tr_bigq_denominator(x));
}
/* L * f for the rational coefficients q[0..n-1] (NULL = 0), L = lcm of the denominators (> 0: the signs are kept) */
static ipoly ip_from_rationals(ctx_t *c, int32_t n, const tr_bigq *const *q) {
    ipoly r;
    r.deg = -1; r.c = NULL;
    tr_bigint *L = B_i64(c, 1);
    for (int32_t i = 0; i < n && !c->err; i++) {
        if (Q_zero(q[i])) continue;
        const tr_bigint *d = tr_bigq_denominator(q[i]);
        tr_bigint *g = B_gcd(c, L, d), *t = B_divexact(c, d, g);
        L = bset(L, B_mul(c, L, t));
        tr_bigint_free(g); tr_bigint_free(t);
    }
    if (!ip_alloc(c, &r, n - 1)) { tr_bigint_free(L); return r; }
    for (int32_t i = 0; i < n && !c->err; i++) {
        if (Q_zero(q[i])) { r.c[i] = B_i64(c, 0); continue; }
        tr_bigint *t = B_divexact(c, L, tr_bigq_denominator(q[i]));
        r.c[i] = B_mul(c, tr_bigq_numerator(q[i]), t);
        tr_bigint_free(t);
    }
    tr_bigint_free(L);
    if (ip_ok(c, &r)) ip_trim(&r);
    return r;
}
static uint64_t ip_abs_sum_bits(ctx_t *c, const ipoly *a) {
    tr_bigint *s = B_i64(c, 0);
    for (int32_t i = 0; i <= a->deg && !c->err; i++) {
        tr_bigint *t = B_abs(c, a->c[i]);
        s = bset(s, B_add(c, s, t));
        tr_bigint_free(t);
    }
    uint64_t bits = s ? tr_bigint_bit_length(s) : 0;
    tr_bigint_free(s);
    return bits;
}

/* ================================================================ batched exact signs: the job */
struct tr_inv_signs_s {
    uint32_t magic, reserved;
    tr_inv_signs_view v;
    uint64_t bound_bits;
    tr_gfp *fields;
    uint64_t *garner, *half, *coef, *num, *den;
    uint32_t *offset, *poly;
    int8_t *sign;
};

void tr_inv_signs_free(tr_inv_signs *s) {
    if (!s || s->magic != SIGNS_MAGIC) return;
    s->magic = 0;
    free(s->fields); free(s->garner); free(s->half); free(s->coef); free(s->num); free(s->den);
    free(s->offset); free(s->poly); free(s->sign);
    free(s);
}
/* Garner digits of the residues r[0..k-1] (the procedure of tr_inv_sign_point) */
static void garner_digits(const tr_gfp *fields, const uint64_t *garner, uint32_t k, const uint64_t *r, uint64_t *dig) {
    for (uint32_t j = 0; j < k; j++) {
        const tr_gfp *F = &fields[j];
        uint64_t x = r[j] % F->p;
        for (uint32_t t = 0; t < j; t++) {
            uint64_t dt = dig[t] >= F->p ? dig[t] - F->p : dig[t];
            x = tr_gfp_mul_hd(F, tr_gfp_sub_hd(F, x, dt), garner[(uint64_t)j * k + t]);
        }
        dig[j] = x;
    }
}
static int fill_fields(ctx_t *c, uint32_t k, tr_gfp **fields, uint64_t **garner, uint64_t **half) {
    *fields = (tr_gfp *)cmalloc(c, (size_t)k * sizeof(tr_gfp));
    if (garner) *garner = (uint64_t *)cmalloc(c, (size_t)k * k * 8);
    if (half) *half = (uint64_t *)cmalloc(c, (size_t)k * 8);
    uint64_t *res = half ? (uint64_t *)cmalloc(c, (size_t)k * 8) : NULL;
    if (c->err) { free(res); return 0; }
    for (uint32_t j = 0; j < k; j++) tr_gfp_fill(TR_INV_PRIMES[j], &(*fields)[j]);
    if (garner)
        for (uint32_t j = 0; j < k; j++)
            for (uint32_t t = 0; t < j; t++)
                (*garner)[(uint64_t)j * k + t] = tr_gfp_inv_hd(&(*fields)[j], TR_INV_PRIMES[t] % TR_INV_PRIMES[j]);
    if (half) {
        for (uint32_t j = 0; j < k; j++) res[j] = (TR_INV_PRIMES[j] - 1) / 2;   /* (M - 1) / 2 = -1/2 mod p_j */
        garner_digits(*fields, *garner, k, res, *half);
    }
    free(res);
    return 1;
}
static uint32_t primes_for(ctx_t *c, uint64_t bits) {
    uint64_t k = (bits + 60) / 61;
    if (k < 1) k = 1;
    if (k > TR_INV_PRIMES_MAX) { cfail(c, TR_MATH_OVERFLOW); return 0; }
    return (uint32_t)k;
}
static uint64_t limbs_of(const tr_bigint *x) { uint64_t n = tr_bigint_limb_count(x); return n ? n : 1; }
static void export_limbs(const tr_bigint *x, uint64_t *out, uint32_t limbs) {
    int32_t sign = 0;
    uint64_t count = 0;
    memset(out, 0, (size_t)limbs * 8);
    if (bsign(x)) tr_bigint_export(x, &sign, out, limbs, &count);
}

/* the job of integer polynomials polys[0..np-1] at the points (rationals) with their polynomial index */
static tr_inv_signs *signs_build(ctx_t *c, uint32_t np, const ipoly *polys, uint64_t n, const tr_bigq *const *points,
                                 const uint32_t *poly) {
    if (c->err) return NULL;
    if (np < 1 || (n && (!points || !poly))) { cfail(c, TR_MATH_INVALID); return NULL; }
    uint64_t *abits = (uint64_t *)cmalloc(c, (size_t)np * 8);
    uint64_t total = 0, bound = 0, limbs = 1;
    for (uint32_t f = 0; f < np && !c->err; f++) {
        abits[f] = ip_abs_sum_bits(c, &polys[f]);
        total += (uint64_t)(polys[f].deg >= 0 ? polys[f].deg + 1 : 1);
    }
    for (uint64_t i = 0; i < n && !c->err; i++) {
        if (poly[i] >= np || !points[i]) { cfail(c, TR_MATH_INVALID); break; }
        const tr_bigint *u = tr_bigq_numerator(points[i]), *v = tr_bigq_denominator(points[i]);
        uint64_t ub = tr_bigint_bit_length(u), vb = tr_bigint_bit_length(v), mb = ub > vb ? ub : vb;
        int32_t d = polys[poly[i]].deg > 0 ? polys[poly[i]].deg : 0;
        uint64_t b = tr_budget_add(abits[poly[i]], tr_budget_mul((uint64_t)d, mb));
        if (b > bound) bound = b;
        uint64_t lu = limbs_of(u), lv = limbs_of(v);
        if (lu > limbs) limbs = lu;
        if (lv > limbs) limbs = lv;
    }
    free(abits);
    if (c->err) return NULL;
    uint32_t k = primes_for(c, tr_budget_add(bound, 2));
    if (!k) return NULL;
    if (limbs > TR_INV_POINT_LIMBS_MAX || total > UINT32_MAX) { cfail(c, TR_MATH_OVERFLOW); return NULL; }
    tr_math_need need;
    memset(&need, 0, sizeof need);
    need.need[TR_BUDGET_OUTPUT] = n;
    uint64_t bytes = sizeof(tr_inv_signs) + (uint64_t)k * (sizeof(tr_gfp) + 8 + 8 * (uint64_t)k);
    bytes = tr_budget_add(bytes, tr_budget_mul((uint64_t)k, tr_budget_mul(total, 8)));
    bytes = tr_budget_add(bytes, tr_budget_mul((uint64_t)np + 1, 4));
    bytes = tr_budget_add(bytes, tr_budget_mul(n, 5 + 16 * limbs));
    need.need[TR_BUDGET_BYTES] = bytes;
    int32_t rc = tr_budget_check(c->b, &need, c->r);
    if (rc) { c->err = rc; return NULL; }
    tr_inv_signs *s = (tr_inv_signs *)cmalloc(c, sizeof *s);
    if (!s) return NULL;
    s->magic = SIGNS_MAGIC;
    fill_fields(c, k, &s->fields, &s->garner, &s->half);
    s->offset = (uint32_t *)cmalloc(c, ((size_t)np + 1) * 4);
    s->coef = (uint64_t *)cmalloc(c, (size_t)k * (size_t)total * 8);
    s->poly = (uint32_t *)cmalloc(c, (size_t)(n ? n : 1) * 4);
    s->sign = (int8_t *)cmalloc(c, (size_t)(n ? n : 1));
    s->num = (uint64_t *)cmalloc(c, (size_t)(n ? n : 1) * limbs * 8);
    s->den = (uint64_t *)cmalloc(c, (size_t)(n ? n : 1) * limbs * 8);
    if (c->err) { tr_inv_signs_free(s); return NULL; }
    uint32_t off = 0;
    for (uint32_t f = 0; f < np; f++) {
        s->offset[f] = off;
        int32_t d = polys[f].deg;
        for (uint32_t j = 0; j < k; j++) {
            uint64_t *row = s->coef + (uint64_t)j * total + off;
            if (d < 0) { row[0] = 0; continue; }
            for (int32_t i = 0; i <= d; i++) tr_bigint_mod_u64(polys[f].c[i], TR_INV_PRIMES[j], &row[i]);
        }
        off += (uint32_t)(d >= 0 ? d + 1 : 1);
    }
    s->offset[np] = off;
    for (uint64_t i = 0; i < n; i++) {
        const tr_bigint *u = tr_bigq_numerator(points[i]), *v = tr_bigq_denominator(points[i]);
        s->poly[i] = poly[i];
        s->sign[i] = (int8_t)bsign(u);
        export_limbs(u, s->num + i * limbs, (uint32_t)limbs);
        export_limbs(v, s->den + i * limbs, (uint32_t)limbs);
    }
    s->bound_bits = bound;
    tr_inv_signs_view *v = &s->v;
    v->npoints = n; v->nprimes = k; v->npolys = np; v->limbs = (uint32_t)limbs; v->total = (uint32_t)total;
    v->fields = s->fields; v->garner = s->garner; v->half = s->half; v->offset = s->offset; v->coef = s->coef;
    v->poly = s->poly; v->sign = s->sign; v->num = s->num; v->den = s->den;
    return s;
}

tr_inv_signs *tr_inv_signs_create(uint32_t npolys, const uint32_t *ncoef, const tr_bigq *const *coef, uint64_t npoints,
                                  const tr_bigq *const *points, const uint32_t *poly, const tr_math_budget *budget,
                                  tr_math_refusal *refusal, int32_t *error) {
    ctx_t c;
    ctx_init(&c, budget, refusal);
    tr_inv_signs *s = NULL;
    if (npolys < 1 || !ncoef || !coef) cfail(&c, TR_MATH_INVALID);
    ipoly *P = c.err ? NULL : (ipoly *)cmalloc(&c, (size_t)npolys * sizeof *P);
    uint64_t at = 0;
    for (uint32_t f = 0; f < npolys && !c.err; f++) {
        if (ncoef[f] < 1 || ncoef[f] > (uint32_t)INT32_MAX) { cfail(&c, TR_MATH_INVALID); break; }
        if (!check_one(&c, TR_BUDGET_DEGREE, ncoef[f] - 1)) break;
        P[f] = ip_from_rationals(&c, (int32_t)ncoef[f], coef + at);
        at += ncoef[f];
    }
    if (!c.err) s = signs_build(&c, npolys, P, npoints, points, poly);
    if (P) for (uint32_t f = 0; f < npolys; f++) ip_free(&P[f]);
    free(P);
    if (error) *error = c.err;
    if (c.err) { tr_inv_signs_free(s); return NULL; }
    return s;
}
int32_t tr_inv_signs_get_view(const tr_inv_signs *s, tr_inv_signs_view *out) {
    if (!s || s->magic != SIGNS_MAGIC || !out) return TR_MATH_INVALID;
    *out = s->v;
    return TR_MATH_OK;
}
uint64_t tr_inv_signs_bound_bits(const tr_inv_signs *s) { return s && s->magic == SIGNS_MAGIC ? s->bound_bits : 0; }
uint64_t tr_inv_signs_device_need(const tr_inv_signs_view *v) { return v ? tr_inv_signs_device_bytes(v) : TR_BUDGET_UNKNOWN; }

int32_t tr_inv_signs_eval(const tr_inv_signs_view *v, int8_t *signs, const tr_math_budget *budget,
                          tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!tr_inv_signs_view_valid(v) || (v->npoints && !signs)) return TR_MATH_INVALID;
    tr_math_budget dflt;
    if (!budget) { tr_budget_default(&dflt); budget = &dflt; }
    int32_t rc = tr_budget_check_one(budget, TR_BUDGET_OUTPUT, v->npoints, refusal);
    if (rc) return rc;
    for (uint64_t i = 0; i < v->npoints; i++) signs[i] = (int8_t)tr_inv_sign_point(v, i);
    return TR_MATH_OK;
}

/* ================================================================ linear targets */
struct tr_inv_linear_s {
    uint32_t magic, reserved;
    tr_inv_linear_info info;
    tr_bigq **point;           /* n */
    tr_bigq **dir;             /* dims x n */
    int32_t *free_unknown;     /* dims */
    tr_bigq **rlo, **rhi;      /* dims (NULL = unbounded) */
};
static void qfree_array(tr_bigq **a, uint64_t n) {
    if (!a) return;
    for (uint64_t i = 0; i < n; i++) tr_bigq_free(a[i]);
    free(a);
}
void tr_inv_linear_free(tr_inv_linear *s) {
    if (!s || s->magic != LINEAR_MAGIC) return;
    s->magic = 0;
    int32_t n = s->info.unknowns, k = s->info.dims;
    qfree_array(s->point, (uint64_t)n);
    qfree_array(s->dir, (uint64_t)k * (uint64_t)n);
    qfree_array(s->rlo, (uint64_t)k);
    qfree_array(s->rhi, (uint64_t)k);
    free(s->free_unknown);
    free(s);
}

/* Fourier-Motzkin systems: inequality i is sum_j a[i][j] y_j <= a[i][k] */
typedef struct { tr_bigq **a; } ineq_t;
typedef struct { ineq_t *v; uint64_t n, cap; int32_t k; int infeasible; } sys_t;

static void ineq_free(ineq_t *q, int32_t k) { qfree_array(q->a, (uint64_t)k + 1); q->a = NULL; }
static void sys_free(sys_t *s) {
    for (uint64_t i = 0; i < s->n; i++) ineq_free(&s->v[i], s->k);
    free(s->v);
    s->v = NULL; s->n = s->cap = 0;
}
static void sys_init(sys_t *s, int32_t k) { s->v = NULL; s->n = s->cap = 0; s->k = k; s->infeasible = 0; }
/* normalize (divide by |first nonzero coefficient|), drop trivial ones, mark infeasibility, merge duplicates (keeping
 * the smaller right-hand side); takes ownership of q */
static void sys_add(ctx_t *c, sys_t *s, ineq_t q) {
    int32_t k = s->k, f = -1;
    if (c->err || !q.a) { ineq_free(&q, k); return; }
    for (int32_t j = 0; j <= k; j++) if (!q.a[j]) { cfail(c, TR_MATH_MEMORY); ineq_free(&q, k); return; }
    for (int32_t j = 0; j < k; j++) if (!Q_zero(q.a[j])) { f = j; break; }
    if (f < 0) {                                                  /* 0 <= rhs */
        if (Q_sign(q.a[k]) < 0) s->infeasible = 1;
        ineq_free(&q, k);
        return;
    }
    {                                                             /* divide by |a_f| > 0 */
        tr_bigq *m = Q_sign(q.a[f]) < 0 ? Q_neg(c, q.a[f]) : Q_copy(c, q.a[f]);
        for (int32_t j = 0; j <= k; j++) if (!Q_zero(q.a[j])) q.a[j] = qset(q.a[j], Q_div(c, q.a[j], m));
        tr_bigq_free(m);
        for (int32_t j = 0; j <= k; j++) if (!q.a[j]) { cfail(c, TR_MATH_MEMORY); ineq_free(&q, k); return; }
    }
    for (uint64_t i = 0; i < s->n && !c->err; i++) {
        int same = 1;
        for (int32_t j = 0; j < k && same; j++) same = Q_cmp(c, s->v[i].a[j], q.a[j]) == 0;
        if (!same) continue;
        if (Q_cmp(c, q.a[k], s->v[i].a[k]) < 0) { tr_bigq *t = s->v[i].a[k]; s->v[i].a[k] = q.a[k]; q.a[k] = t; }
        ineq_free(&q, k);
        return;
    }
    if (s->n == s->cap) {
        uint64_t cap = s->cap ? 2 * s->cap : 16;
        ineq_t *v = (ineq_t *)realloc(s->v, (size_t)cap * sizeof *v);
        if (!v) { cfail(c, TR_MATH_MEMORY); ineq_free(&q, k); return; }
        s->v = v; s->cap = cap;
    }
    s->v[s->n++] = q;
}
static ineq_t ineq_new(ctx_t *c, int32_t k) {
    ineq_t q;
    q.a = (tr_bigq **)cmalloc(c, ((size_t)k + 1) * sizeof *q.a);
    return q;
}
/* eliminates y_e; the size of the new system is budgeted (support) and counted (iterations) before it is built */
static sys_t fm_eliminate(ctx_t *c, const sys_t *s, int32_t e) {
    sys_t t;
    sys_init(&t, s->k);
    t.infeasible = s->infeasible;
    if (c->err) return t;
    uint64_t np = 0, nn = 0, nz = 0;
    for (uint64_t i = 0; i < s->n; i++) {
        int sg = Q_sign(s->v[i].a[e]);
        if (sg > 0) np++; else if (sg < 0) nn++; else nz++;
    }
    uint64_t need = tr_budget_add(nz, tr_budget_mul(np, nn));
    if (!check_one(c, TR_BUDGET_SUPPORT, need) || !tick(c, need)) return t;
    int32_t k = s->k;
    for (uint64_t i = 0; i < s->n && !c->err; i++) {
        if (Q_sign(s->v[i].a[e]) != 0) continue;
        ineq_t q = ineq_new(c, k);
        for (int32_t j = 0; j <= k && q.a; j++) q.a[j] = Q_copy(c, s->v[i].a[j]);
        sys_add(c, &t, q);
    }
    for (uint64_t p = 0; p < s->n && !c->err; p++) {
        if (Q_sign(s->v[p].a[e]) <= 0) continue;
        for (uint64_t m = 0; m < s->n && !c->err; m++) {
            if (Q_sign(s->v[m].a[e]) >= 0) continue;
            tr_bigq *wp = Q_neg(c, s->v[m].a[e]);              /* > 0 */
            const tr_bigq *wm = s->v[p].a[e];                   /* > 0 */
            ineq_t q = ineq_new(c, k);
            for (int32_t j = 0; j <= k && q.a && !c->err; j++) {
                if (j == e) { q.a[j] = Q_i64(c, 0); continue; }
                tr_bigq *x = Q_mul(c, wp, s->v[p].a[j]), *y = Q_mul(c, wm, s->v[m].a[j]);
                q.a[j] = Q_add(c, x, y);
                tr_bigq_free(x); tr_bigq_free(y);
            }
            tr_bigq_free(wp);
            sys_add(c, &t, q);
        }
    }
    return t;
}
/* the bounds of y_e from the inequalities of s with every other variable given (vals; NULL entries are not used: the
 * system involves only y_e then). *lo / *hi NULL = unbounded. 0 when lo > hi (infeasible). */
static int fm_bounds(ctx_t *c, const sys_t *s, int32_t e, tr_bigq *const *vals, int32_t nvals, tr_bigq **lo, tr_bigq **hi) {
    *lo = NULL; *hi = NULL;
    if (s->infeasible) return 0;
    for (uint64_t i = 0; i < s->n && !c->err; i++) {
        const ineq_t *q = &s->v[i];
        tr_bigq *rhs = Q_copy(c, q->a[s->k]);
        for (int32_t j = 0; j < nvals && !c->err; j++) {
            if (j == e || Q_zero(q->a[j])) continue;
            tr_bigq *t = Q_mul(c, q->a[j], vals[j]);
            rhs = qset(rhs, Q_sub(c, rhs, t));
            tr_bigq_free(t);
        }
        int sg = Q_sign(q->a[e]);
        if (sg == 0) {                                          /* 0 <= rhs once the others are fixed */
            int bad = Q_sign(rhs) < 0;
            tr_bigq_free(rhs);
            if (bad) { tr_bigq_free(*lo); tr_bigq_free(*hi); *lo = *hi = NULL; return 0; }
            continue;
        }
        tr_bigq *bnd = Q_div(c, rhs, q->a[e]);
        tr_bigq_free(rhs);
        if (sg > 0) {
            if (!*hi || Q_cmp(c, bnd, *hi) < 0) { tr_bigq_free(*hi); *hi = bnd; } else tr_bigq_free(bnd);
        } else {
            if (!*lo || Q_cmp(c, bnd, *lo) > 0) { tr_bigq_free(*lo); *lo = bnd; } else tr_bigq_free(bnd);
        }
    }
    if (c->err) return 0;
    if (*lo && *hi && Q_cmp(c, *lo, *hi) > 0) return 0;
    return 1;
}
/* 0 when inside [lo, hi], else the closer bound (the witness rule: the parameter nearest to the particular solution) */
static tr_bigq *fm_pick(ctx_t *c, const tr_bigq *lo, const tr_bigq *hi) {
    if (lo && Q_sign(lo) > 0) return Q_copy(c, lo);
    if (hi && Q_sign(hi) < 0) return Q_copy(c, hi);
    return Q_i64(c, 0);
}

/* Gauss-Jordan elimination of the m x (n + 1) system M in place, skipping the columns marked in skip (NULL: none);
 * the pivot is a nonzero entry, so nothing is ever divided by zero. Returns the rank. */
static int32_t gauss_jordan(ctx_t *c, tr_bigq **M, int32_t m, int32_t n, const int32_t *skip, int32_t *pivcol,
                            int32_t *is_pivot) {
    int32_t W = n + 1, rank = 0;
    for (int32_t col = 0; col < n && rank < m && !c->err; col++) {
        if (skip && skip[col]) continue;
        int32_t r = -1;
        for (int32_t i = rank; i < m; i++) if (!Q_zero(M[(size_t)i * W + col])) { r = i; break; }
        if (r < 0) continue;
        if (r != rank)
            for (int32_t j = 0; j < W; j++) {
                tr_bigq *t = M[(size_t)r * W + j]; M[(size_t)r * W + j] = M[(size_t)rank * W + j]; M[(size_t)rank * W + j] = t;
            }
        tr_bigq *piv = Q_copy(c, M[(size_t)rank * W + col]);
        for (int32_t j = 0; j < W && !c->err; j++)
            if (!Q_zero(M[(size_t)rank * W + j]))
                M[(size_t)rank * W + j] = qset(M[(size_t)rank * W + j], Q_div(c, M[(size_t)rank * W + j], piv));
        tr_bigq_free(piv);
        for (int32_t i = 0; i < m && !c->err; i++) {
            if (i == rank || Q_zero(M[(size_t)i * W + col])) continue;
            tr_bigq *f = Q_copy(c, M[(size_t)i * W + col]);
            for (int32_t j = 0; j < W && !c->err; j++) {
                if (Q_zero(M[(size_t)rank * W + j])) continue;
                tr_bigq *t = Q_mul(c, f, M[(size_t)rank * W + j]);
                M[(size_t)i * W + j] = qset(M[(size_t)i * W + j], Q_sub(c, M[(size_t)i * W + j], t));
                tr_bigq_free(t);
            }
            tr_bigq_free(f);
        }
        if (pivcol) pivcol[rank] = col;
        if (is_pivot) is_pivot[col] = 1;
        rank++;
    }
    return rank;
}
/* after elimination: no row 0 = c with c != 0 */
static int consistent(tr_bigq *const *M, int32_t m, int32_t n, int32_t rank) {
    for (int32_t i = rank; i < m; i++) if (!Q_zero(M[(size_t)i * (n + 1) + n])) return 0;
    return 1;
}
static int gauss_consistent(ctx_t *c, tr_bigq **M, int32_t m, int32_t n) {
    int32_t rank = gauss_jordan(c, M, m, n, NULL, NULL, NULL);
    return !c->err && consistent(M, m, n, rank);
}

static tr_inv_linear *linear_core(ctx_t *c, int32_t m, int32_t n, tr_bigq **M /* m x (n+1), owned */,
                                  const tr_bigq *const *lo, const tr_bigq *const *hi) {
    tr_inv_linear *out = (tr_inv_linear *)cmalloc(c, sizeof *out);
    int32_t W = n + 1;
    int32_t *fixed = (int32_t *)cmalloc(c, (size_t)n * sizeof(int32_t));
    int32_t *pivcol = (int32_t *)cmalloc(c, (size_t)(m > 0 ? m : 1) * sizeof(int32_t));
    int32_t *is_pivot = (int32_t *)cmalloc(c, (size_t)n * sizeof(int32_t));
    tr_bigq **p = (tr_bigq **)cmalloc(c, (size_t)n * sizeof *p);
    sys_t sys, *chain = NULL;
    sys_init(&sys, 0);
    tr_bigq **ystar = NULL, **plo = NULL, **phi = NULL;
    int32_t rank = 0, nfixed = 0, k = 0;
    tr_bigq **orig = NULL;                                      /* the system without the box (fixed unknowns) */
    if (c->err) goto done;
    out->magic = LINEAR_MAGIC;
    out->info.unknowns = n; out->info.equations = m;
    for (int32_t j = 0; j < n && lo && hi && !orig && !c->err; j++)
        if (lo[j] && hi[j] && Q_cmp(c, lo[j], hi[j]) == 0) {
            orig = (tr_bigq **)cmalloc(c, (size_t)(m > 0 ? m : 1) * W * sizeof *orig);
            for (int32_t i = 0; orig && i < m * W; i++) orig[i] = Q_copy(c, M[i]);
        }
    /* the domain: lo <= hi; fixed coordinates substituted into the right-hand sides */
    for (int32_t j = 0; j < n && !c->err; j++) {
        const tr_bigq *l = lo ? lo[j] : NULL, *h = hi ? hi[j] : NULL;
        if (l && h) {
            int cmp = Q_cmp(c, l, h);
            if (cmp > 0) { cfail(c, TR_MATH_INVALID); break; }
            if (cmp == 0) {
                fixed[j] = 1; nfixed++;
                for (int32_t i = 0; i < m && !c->err; i++) {
                    tr_bigq **e = &M[(size_t)i * W + j];
                    if (Q_zero(*e)) continue;
                    tr_bigq *t = Q_mul(c, *e, l);
                    M[(size_t)i * W + n] = qset(M[(size_t)i * W + n], Q_sub(c, M[(size_t)i * W + n], t));
                    tr_bigq_free(t);
                    *e = qset(*e, Q_i64(c, 0));
                }
            }
        }
    }
    rank = gauss_jordan(c, M, m, n, fixed, pivcol, is_pivot);
    if (c->err) goto done;
    out->info.rank = rank; out->info.fixed = nfixed;
    if (!consistent(M, m, n, rank)) {
        /* inconsistent once the box fixed some unknowns: the equations alone may still have solutions (outside) */
        out->info.kind = TR_INV_EMPTY;
        out->info.reason = orig && gauss_consistent(c, orig, m, n) ? TR_INV_OUTSIDE : TR_INV_INCONSISTENT;
        goto done;
    }
    if (rank == 0) { out->info.kind = TR_INV_WHOLE; goto done; }
    /* the particular solution (free unknowns 0, fixed ones their value) and the directions */
    for (int32_t j = 0; j < n && !c->err; j++) {
        if (fixed[j]) p[j] = Q_copy(c, lo[j]);
        else p[j] = Q_i64(c, 0);
    }
    for (int32_t r = 0; r < rank && !c->err; r++) p[pivcol[r]] = qset(p[pivcol[r]], Q_copy(c, M[(size_t)r * W + n]));
    for (int32_t j = 0; j < n; j++) if (!fixed[j] && !is_pivot[j]) k++;
    if (c->err) goto done;
    if (k == 0) {                                               /* one solution: inside the box or not */
        int inside = 1;
        for (int32_t j = 0; j < n && inside && !c->err; j++) {
            if (lo && lo[j] && Q_cmp(c, p[j], lo[j]) < 0) inside = 0;
            if (hi && hi[j] && Q_cmp(c, p[j], hi[j]) > 0) inside = 0;
        }
        if (c->err) goto done;
        if (!inside) { out->info.kind = TR_INV_EMPTY; out->info.reason = TR_INV_OUTSIDE; goto done; }
        out->info.kind = TR_INV_UNIQUE;
        out->point = p; p = NULL;
        goto done;
    }
    out->info.dims = k;
    out->free_unknown = (int32_t *)cmalloc(c, (size_t)k * sizeof(int32_t));
    out->dir = (tr_bigq **)cmalloc(c, (size_t)k * n * sizeof(tr_bigq *));
    out->rlo = (tr_bigq **)cmalloc(c, (size_t)k * sizeof(tr_bigq *));
    out->rhi = (tr_bigq **)cmalloc(c, (size_t)k * sizeof(tr_bigq *));
    if (c->err) goto done;
    for (int32_t j = 0, d = 0; j < n; j++) if (!fixed[j] && !is_pivot[j]) out->free_unknown[d++] = j;
    for (int32_t d = 0; d < k && !c->err; d++) {
        int32_t f = out->free_unknown[d];
        for (int32_t j = 0; j < n; j++) out->dir[(size_t)d * n + j] = Q_i64(c, j == f ? 1 : 0);
        for (int32_t r = 0; r < rank && !c->err; r++)
            if (!Q_zero(M[(size_t)r * W + f]))
                out->dir[(size_t)d * n + pivcol[r]] = qset(out->dir[(size_t)d * n + pivcol[r]], Q_neg(c, M[(size_t)r * W + f]));
    }
    /* the box as inequalities over y: lo_j - p_j <= sum_d v_d[j] y_d <= hi_j - p_j */
    sys_init(&sys, k);
    for (int32_t j = 0; j < n && !c->err; j++) {
        if (fixed[j]) continue;
        for (int side = 0; side < 2 && !c->err; side++) {
            const tr_bigq *bound = side ? (hi ? hi[j] : NULL) : (lo ? lo[j] : NULL);
            if (!bound) continue;
            ineq_t q = ineq_new(c, k);
            if (!q.a) break;
            for (int32_t d = 0; d < k; d++)
                q.a[d] = side ? Q_copy(c, out->dir[(size_t)d * n + j]) : Q_neg(c, out->dir[(size_t)d * n + j]);
            q.a[k] = side ? Q_sub(c, bound, p[j]) : Q_sub(c, p[j], bound);
            sys_add(c, &sys, q);
        }
    }
    if (!check_one(c, TR_BUDGET_SUPPORT, sys.n)) goto done;
    /* the witness: eliminate y_{k-1} .. y_1 (chain[e] holds the variables 0..e), then back-substitute */
    chain = (sys_t *)cmalloc(c, (size_t)k * sizeof(sys_t));
    ystar = (tr_bigq **)cmalloc(c, (size_t)k * sizeof(tr_bigq *));
    plo = (tr_bigq **)cmalloc(c, (size_t)k * sizeof(tr_bigq *));
    phi = (tr_bigq **)cmalloc(c, (size_t)k * sizeof(tr_bigq *));
    if (c->err) goto done;
    for (int32_t e = 0; e < k; e++) sys_init(&chain[e], k);
    chain[k - 1] = sys; sys_init(&sys, 0);
    for (int32_t e = k - 1; e >= 1 && !c->err; e--) chain[e - 1] = fm_eliminate(c, &chain[e], e);
    if (c->err) goto done;
    int feasible = 1;
    for (int32_t e = 0; e < k && feasible && !c->err; e++) {
        tr_bigq *l = NULL, *h = NULL;
        if (!fm_bounds(c, &chain[e], e, ystar, e, &l, &h)) feasible = 0;
        else ystar[e] = fm_pick(c, l, h);
        tr_bigq_free(l); tr_bigq_free(h);
    }
    if (c->err) goto done;
    if (!feasible) { out->info.kind = TR_INV_EMPTY; out->info.reason = TR_INV_OUTSIDE; goto done; }
    /* the exact range of every parameter: project onto y_d (eliminate the others) */
    for (int32_t d = 0; d < k && !c->err; d++) {
        if (d == 0) { fm_bounds(c, &chain[0], 0, NULL, 0, &plo[0], &phi[0]); continue; }
        sys_t cur;
        sys_init(&cur, k);
        for (uint64_t i = 0; i < chain[k - 1].n && !c->err; i++) {
            ineq_t q = ineq_new(c, k);
            for (int32_t j = 0; j <= k && q.a; j++) q.a[j] = Q_copy(c, chain[k - 1].v[i].a[j]);
            sys_add(c, &cur, q);
        }
        for (int32_t e = 0; e < k && !c->err; e++) {
            if (e == d) continue;
            sys_t nx = fm_eliminate(c, &cur, e);
            sys_free(&cur);
            cur = nx;
        }
        if (!c->err) fm_bounds(c, &cur, d, NULL, 0, &plo[d], &phi[d]);
        sys_free(&cur);
    }
    if (c->err) goto done;
    out->point = p; p = NULL;
    int single = 1;
    for (int32_t d = 0; d < k && !c->err; d++) {
        for (int32_t j = 0; j < n && !c->err; j++) {
            if (Q_zero(ystar[d]) || Q_zero(out->dir[(size_t)d * n + j])) continue;
            tr_bigq *t = Q_mul(c, ystar[d], out->dir[(size_t)d * n + j]);
            out->point[j] = qset(out->point[j], Q_add(c, out->point[j], t));
            tr_bigq_free(t);
        }
        out->rlo[d] = plo[d] ? Q_sub(c, plo[d], ystar[d]) : NULL;
        out->rhi[d] = phi[d] ? Q_sub(c, phi[d], ystar[d]) : NULL;
        if (!plo[d] || !phi[d] || Q_cmp(c, plo[d], phi[d]) != 0) single = 0;
    }
    if (c->err) goto done;
    if (single) {                                               /* every projection is one point */
        out->info.kind = TR_INV_UNIQUE;
        qfree_array(out->dir, (uint64_t)k * n); out->dir = NULL;
        qfree_array(out->rlo, (uint64_t)k); out->rlo = NULL;
        qfree_array(out->rhi, (uint64_t)k); out->rhi = NULL;
        free(out->free_unknown); out->free_unknown = NULL;
        out->info.dims = 0;
    } else out->info.kind = TR_INV_FAMILY;
done:
    if (out) out->info.iterations = c->iters;
    sys_free(&sys);
    if (chain) for (int32_t e = 0; e < k; e++) sys_free(&chain[e]);
    free(chain);
    qfree_array(ystar, (uint64_t)k); qfree_array(plo, (uint64_t)k); qfree_array(phi, (uint64_t)k);
    qfree_array(p, (uint64_t)n);
    if (orig) qfree_array(orig, (uint64_t)m * W);
    free(fixed); free(pivcol); free(is_pivot);
    if (c->err) { if (out) { if (!out->magic) out->magic = LINEAR_MAGIC; tr_inv_linear_free(out); } return NULL; }
    return out;
}

static int linear_check(ctx_t *c, int32_t m, int32_t n) {
    if (m < 0 || n < 1) { cfail(c, TR_MATH_INVALID); return 0; }
    tr_math_need need;
    memset(&need, 0, sizeof need);
    need.need[TR_BUDGET_VARIABLES] = (uint64_t)n;
    need.need[TR_BUDGET_TENSOR] = tr_budget_mul((uint64_t)m, (uint64_t)n + 1);
    need.need[TR_BUDGET_BYTES] = tr_budget_mul(need.need[TR_BUDGET_TENSOR], 8 + sizeof(void *));
    int32_t rc = tr_budget_check(c->b, &need, c->r);
    if (rc) { c->err = rc; return 0; }
    if (n > TR_INV_LINEAR_MAX) { cfail(c, TR_MATH_UNSUPPORTED); return 0; }
    return 1;
}
tr_inv_linear *tr_inv_linear_solve(int32_t m, int32_t n, const tr_bigq *const *A, const tr_bigq *const *b,
                                   const tr_bigq *const *lo, const tr_bigq *const *hi, const tr_math_budget *budget,
                                   tr_math_refusal *refusal, int32_t *error) {
    ctx_t c;
    ctx_init(&c, budget, refusal);
    tr_inv_linear *out = NULL;
    if (linear_check(&c, m, n) && (!m || (A && b))) {
        int32_t W = n + 1;
        tr_bigq **M = (tr_bigq **)cmalloc(&c, (size_t)(m > 0 ? m : 1) * W * sizeof *M);
        for (int32_t i = 0; i < m && !c.err; i++) {
            for (int32_t j = 0; j < n; j++) {
                const tr_bigq *a = A[(size_t)i * n + j];
                M[(size_t)i * W + j] = a ? Q_copy(&c, a) : Q_i64(&c, 0);
            }
            M[(size_t)i * W + n] = b[i] ? Q_copy(&c, b[i]) : Q_i64(&c, 0);
        }
        if (!c.err) out = linear_core(&c, m, n, M, lo, hi);
        if (M) qfree_array(M, (uint64_t)m * W);
    } else if (!c.err) cfail(&c, TR_MATH_INVALID);
    if (error) *error = c.err;
    return c.err ? NULL : out;
}
tr_inv_linear *tr_inv_linear_from_polys(int32_t m, const tr_math_poly *const *f, const tr_bigq *const *target,
                                        int32_t n, const int32_t *unknowns, const tr_bigq *const *lo,
                                        const tr_bigq *const *hi, const tr_math_budget *budget,
                                        tr_math_refusal *refusal, int32_t *error) {
    ctx_t c;
    ctx_init(&c, budget, refusal);
    tr_inv_linear *out = NULL;
    int32_t slot[TR_MATH_VARIABLES];
    for (int32_t v = 0; v < TR_MATH_VARIABLES; v++) slot[v] = -1;
    if (!unknowns || (m && !f)) cfail(&c, TR_MATH_INVALID);
    else if (linear_check(&c, m, n)) {
        if (n > TR_MATH_VARIABLES) cfail(&c, TR_MATH_INVALID);
        for (int32_t j = 0; j < n && !c.err; j++) {
            if (unknowns[j] < 0 || unknowns[j] >= TR_MATH_VARIABLES || slot[unknowns[j]] >= 0) cfail(&c, TR_MATH_INVALID);
            else slot[unknowns[j]] = j;
        }
    }
    int32_t W = n + 1;
    tr_bigq **M = c.err ? NULL : (tr_bigq **)cmalloc(&c, (size_t)(m > 0 ? m : 1) * W * sizeof *M);
    for (int32_t i = 0; i < m && !c.err; i++) {
        int32_t nt = tr_math_poly_term_count(f[i]);
        if (nt < 0) { cfail(&c, TR_MATH_INVALID); break; }
        for (int32_t j = 0; j < W; j++) M[(size_t)i * W + j] = Q_i64(&c, 0);
        if (target && target[i]) M[(size_t)i * W + n] = qset(M[(size_t)i * W + n], Q_copy(&c, target[i]));
        for (int32_t t = 0; t < nt && !c.err; t++) {
            tr_math_term term;
            if (tr_math_poly_get_term(f[i], t, &term)) { cfail(&c, TR_MATH_INVALID); break; }
            int32_t deg = 0, var = -1;
            for (int32_t v = 0; v < TR_MATH_VARIABLES; v++)
                if (term.exponents[v]) {
                    deg += term.exponents[v]; var = v;
                    if (slot[v] < 0) { cfail(&c, TR_MATH_DOMAIN_ERROR); break; }
                }
            if (c.err) break;
            if (deg > 1) { cfail(&c, TR_MATH_UNSUPPORTED); break; }
            tr_num_rational q = {term.coefficient.numerator, term.coefficient.denominator};
            int32_t e = 0;
            tr_bigq *a = tr_bigq_from_q64(q, c.b, c.r, &e);
            if (!a) { cfail(&c, e); break; }
            if (deg == 0) M[(size_t)i * W + n] = qset(M[(size_t)i * W + n], Q_sub(&c, M[(size_t)i * W + n], a));
            else M[(size_t)i * W + slot[var]] = qset(M[(size_t)i * W + slot[var]], Q_add(&c, M[(size_t)i * W + slot[var]], a));
            tr_bigq_free(a);
        }
    }
    if (!c.err) out = linear_core(&c, m, n, M, lo, hi);
    if (M) qfree_array(M, (uint64_t)m * W);
    if (error) *error = c.err;
    return c.err ? NULL : out;
}
int32_t tr_inv_linear_get_info(const tr_inv_linear *s, tr_inv_linear_info *out) {
    if (!s || s->magic != LINEAR_MAGIC || !out) return TR_MATH_INVALID;
    *out = s->info;
    return TR_MATH_OK;
}
const tr_bigq *tr_inv_linear_point(const tr_inv_linear *s, int32_t j) {
    if (!s || s->magic != LINEAR_MAGIC || !s->point || j < 0 || j >= s->info.unknowns) return NULL;
    return s->point[j];
}
const tr_bigq *tr_inv_linear_direction(const tr_inv_linear *s, int32_t d, int32_t j) {
    if (!s || s->magic != LINEAR_MAGIC || !s->dir || d < 0 || d >= s->info.dims || j < 0 || j >= s->info.unknowns)
        return NULL;
    return s->dir[(size_t)d * s->info.unknowns + j];
}
int32_t tr_inv_linear_free_unknown(const tr_inv_linear *s, int32_t d) {
    if (!s || s->magic != LINEAR_MAGIC || !s->free_unknown || d < 0 || d >= s->info.dims) return TR_MATH_INVALID;
    return s->free_unknown[d];
}
int32_t tr_inv_linear_range(const tr_inv_linear *s, int32_t d, const tr_bigq **lo, const tr_bigq **hi) {
    if (!s || s->magic != LINEAR_MAGIC || !s->rlo || d < 0 || d >= s->info.dims || !lo || !hi) return TR_MATH_INVALID;
    *lo = s->rlo[d]; *hi = s->rhi[d];
    return TR_MATH_OK;
}

/* ================================================================ univariate roots */
typedef struct {
    tr_bigq *lo, *hi;            /* exact: hi == NULL, lo the value */
    int32_t exact, mult, boundary, sign_lo, factor, how;
} root_t;
struct tr_inv_roots_s {
    uint32_t magic, reserved;
    tr_inv_roots_info info;
    tr_bigq *blo, *bhi;          /* the declared bounds (NULL = unbounded) */
    int32_t nfactors;
    ipoly *factors;              /* factor i: multiplicity i + 1 (degree 0: no roots of that multiplicity) */
    root_t *roots;
    int32_t nroots, cap;
};
void tr_inv_roots_free(tr_inv_roots *s) {
    if (!s || s->magic != ROOTS_MAGIC) return;
    s->magic = 0;
    tr_bigq_free(s->blo); tr_bigq_free(s->bhi);
    if (s->factors) for (int32_t i = 0; i < s->nfactors; i++) ip_free(&s->factors[i]);
    free(s->factors);
    for (int32_t i = 0; i < s->nroots; i++) { tr_bigq_free(s->roots[i].lo); tr_bigq_free(s->roots[i].hi); }
    free(s->roots);
    free(s);
}
static root_t *add_root(ctx_t *c, tr_inv_roots *s) {
    if (c->err) return NULL;
    if (!check_one(c, TR_BUDGET_OUTPUT, (uint64_t)s->nroots + 1)) return NULL;
    if (s->nroots == s->cap) {
        int32_t cap = s->cap ? 2 * s->cap : 8;
        root_t *r = (root_t *)realloc(s->roots, (size_t)cap * sizeof *r);
        if (!r) { cfail(c, TR_MATH_MEMORY); return NULL; }
        s->roots = r; s->cap = cap;
    }
    root_t *r = &s->roots[s->nroots++];
    memset(r, 0, sizeof *r);
    return r;
}
/* the factor whose root is r (exact): a_i(r) == 0 */
static int factor_at(ctx_t *c, const tr_inv_roots *s, const tr_bigq *r) {
    for (int32_t i = 0; i < s->nfactors && !c->err; i++)
        if (s->factors[i].deg > 0 && ip_sign_q(c, &s->factors[i], r) == 0) return i;
    if (!c->err) cfail(c, TR_MATH_INVALID);                     /* unreachable: r is a root of the product */
    return 0;
}
static void set_exact(ctx_t *c, tr_inv_roots *s, root_t *r, tr_bigq *value, int how) {
    tr_bigq_free(r->lo); tr_bigq_free(r->hi);
    r->lo = value; r->hi = NULL; r->exact = 1; r->how = how; r->sign_lo = 0;
    if (!value) { cfail(c, TR_MATH_MEMORY); return; }
    r->factor = factor_at(c, s, value);
    r->mult = r->factor + 1;
    if (s->blo && Q_cmp(c, value, s->blo) == 0) r->boundary |= TR_INV_AT_LO;
    if (s->bhi && Q_cmp(c, value, s->bhi) == 0) r->boundary |= TR_INV_AT_HI;
}

/* Sturm sequence of the square-free Q: S0 = Q, S1 = Q', S_{j+1} = -(positive) rem(S_{j-1}, S_j) */
typedef struct { ipoly *s; int32_t n; } sturm_t;
static void sturm_free(sturm_t *st) { for (int32_t i = 0; i < st->n; i++) ip_free(&st->s[i]); free(st->s); st->s = NULL; st->n = 0; }
static sturm_t sturm_build(ctx_t *c, const ipoly *Q) {
    sturm_t st;
    st.n = 0;
    st.s = (ipoly *)cmalloc(c, (size_t)(Q->deg + 2) * sizeof(ipoly));
    if (!st.s) return st;
    st.s[0] = ip_copy(c, Q); st.n = 1;
    st.s[1] = ip_deriv(c, Q); st.n = 2;
    ip_primitive(c, &st.s[1], 0);
    while (!c->err && st.s[st.n - 1].deg > 0 && st.n < Q->deg + 2) {
        const ipoly *a = &st.s[st.n - 2], *x = &st.s[st.n - 1];
        ipoly r = ip_prem(c, a, x);
        if (c->err || r.deg < 0) { ip_free(&r); break; }
        int32_t delta = a->deg - x->deg + 1;
        int keep = bsign(x->c[x->deg]) < 0 && (delta & 1);       /* lc^delta < 0: prem has the opposite sign */
        if (!keep) for (int32_t i = 0; i <= r.deg; i++) r.c[i] = bset(r.c[i], B_neg(c, r.c[i]));
        ip_primitive(c, &r, 0);
        st.s[st.n++] = r;
    }
    return st;
}
/* sign variations of the sequence at x (zeros skipped) */
static int sturm_var(ctx_t *c, const sturm_t *st, const tr_bigq *x) {
    int last = 0, v = 0;
    for (int32_t i = 0; i < st->n && !c->err; i++) {
        int s = ip_sign_q(c, &st->s[i], x);
        if (!s) continue;
        if (last && s != last) v++;
        last = s;
    }
    return v;
}
/* the simplest fraction (smallest denominator) in the open interval (a, b), a < b: continued fractions */
static tr_bigq *simplest(ctx_t *c, const tr_bigq *a, const tr_bigq *b) {
    int sa = Q_sign(a), sb = Q_sign(b);
    if (sa < 0 && sb > 0) return Q_i64(c, 0);
    if (sb <= 0) {
        tr_bigq *na = Q_neg(c, b), *nb = Q_neg(c, a), *s = c->err ? NULL : simplest(c, na, nb), *r = Q_neg(c, s);
        tr_bigq_free(na); tr_bigq_free(nb); tr_bigq_free(s);
        return r;
    }
    tr_bigint *an = B_copy(c, tr_bigq_numerator(a)), *ad = B_copy(c, tr_bigq_denominator(a));
    tr_bigint *bn = B_copy(c, tr_bigq_numerator(b)), *bd = B_copy(c, tr_bigq_denominator(b));
    int binf = 0;
    tr_bigint *p0 = B_i64(c, 0), *q0 = B_i64(c, 1), *p1 = B_i64(c, 1), *q1 = B_i64(c, 0);   /* convergents */
    for (int guard = 0; !c->err && guard < 1 << 20; guard++) {
        tr_bigint *f = B_floordiv(c, an, ad), *f1 = NULL, *t = NULL, *last = NULL;
        if (c->err) { tr_bigint_free(f); break; }
        tr_bigint *one = B_i64(c, 1);
        f1 = B_add(c, f, one);
        tr_bigint_free(one);
        int stop = binf;
        if (!stop) { t = B_mul(c, f1, bd); stop = !c->err && tr_bigint_cmp(t, bn) < 0; tr_bigint_free(t); }
        last = stop ? f1 : f;
        tr_bigint *np = B_mul(c, last, p1), *nq = B_mul(c, last, q1);   /* p = last p1 + p0, q = last q1 + q0 */
        np = bset(np, B_add(c, np, p0)); nq = bset(nq, B_add(c, nq, q0));
        tr_bigint_free(p0); tr_bigint_free(q0);
        p0 = p1; q0 = q1; p1 = np; q1 = nq;
        if (stop) { tr_bigint_free(f); tr_bigint_free(f1); break; }
        /* (a, b) within (f, f + 1]: next interval (1 / (b - f), 1 / (a - f)) */
        tr_bigint *fb = B_mul(c, f, bd), *fa = B_mul(c, f, ad);
        tr_bigint *nbn = B_sub(c, bn, fb), *nan_ = B_sub(c, an, fa);
        int ainf = bsign(nan_) == 0;
        tr_bigint *new_an = bd, *new_ad = nbn, *new_bn = ad, *new_bd = nan_;
        tr_bigint_free(an); tr_bigint_free(bn); tr_bigint_free(fb); tr_bigint_free(fa); tr_bigint_free(f); tr_bigint_free(f1);
        an = new_an; ad = new_ad; bn = new_bn; bd = new_bd;
        binf = ainf;
    }
    tr_bigq *r = Q_frac(c, p1, q1);
    tr_bigint_free(an); tr_bigint_free(ad); tr_bigint_free(bn); tr_bigint_free(bd);
    tr_bigint_free(p0); tr_bigint_free(q0); tr_bigint_free(p1); tr_bigint_free(q1);
    return r;
}
/* rational detection of an isolated root (see the header): exact candidate, irrational certificate, or exact split */
static void classify(ctx_t *c, tr_inv_roots *s, root_t *r) {
    const ipoly *P = &s->factors[r->factor];
    if (P->deg == 1) {                                          /* -c0 / c1 */
        tr_bigint *n = B_neg(c, P->c[0]);
        set_exact(c, s, r, Q_frac(c, n, P->c[1]), TR_INV_HOW_LINEAR);
        tr_bigint_free(n);
        return;
    }
    tr_bigint *L = B_abs(c, P->c[P->deg]);
    while (!c->err) {
        tr_bigq *cand = simplest(c, r->lo, r->hi);
        if (c->err) { tr_bigq_free(cand); break; }
        if (tr_bigint_cmp(tr_bigq_denominator(cand), L) > 0) { tr_bigq_free(cand); r->how = TR_INV_HOW_IRRATIONAL; break; }
        int sg = ip_sign_q(c, P, cand);
        if (c->err) { tr_bigq_free(cand); break; }
        if (sg == 0) { set_exact(c, s, r, cand, TR_INV_HOW_CANDIDATE); break; }
        if (sg == r->sign_lo) r->lo = qset(r->lo, cand); else r->hi = qset(r->hi, cand);
        tr_bigq *m = Q_mid(c, r->lo, r->hi);
        int sm = ip_sign_q(c, P, m);
        if (c->err) { tr_bigq_free(m); break; }
        if (sm == 0) { set_exact(c, s, r, m, TR_INV_HOW_EVALUATED); break; }
        if (sm == r->sign_lo) r->lo = qset(r->lo, m); else r->hi = qset(r->hi, m);
    }
    tr_bigint_free(L);
}
static void sort_roots(tr_inv_roots *s, ctx_t *c) {   /* insertion sort by position (exact value or lower bound) */
    for (int32_t i = 1; i < s->nroots && !c->err; i++) {
        root_t x = s->roots[i];
        int32_t j = i - 1;
        while (j >= 0 && Q_cmp(c, s->roots[j].lo, x.lo) > 0) { s->roots[j + 1] = s->roots[j]; j--; }
        s->roots[j + 1] = x;
    }
}

static tr_inv_roots *isolate_core(ctx_t *c, ipoly *F /* f - target, integer, owned */, const tr_bigq *lo,
                                  const tr_bigq *hi) {
    tr_inv_roots *s = (tr_inv_roots *)cmalloc(c, sizeof *s);
    ipoly Q, D, B, C, Dd, P1;
    Q.deg = D.deg = B.deg = C.deg = Dd.deg = P1.deg = -1;
    Q.c = D.c = B.c = C.c = Dd.c = P1.c = NULL;
    sturm_t st;
    st.s = NULL; st.n = 0;
    typedef struct { tr_bigq *a, *b; int va, vb, az, bz; } span_t;
    span_t *stack = NULL;
    int32_t top = 0, cap = 0;
    tr_bigq *L = NULL, *H = NULL;
    if (!s) goto done;
    s->magic = ROOTS_MAGIC;
    if (lo && hi && Q_cmp(c, lo, hi) > 0) { cfail(c, TR_MATH_INVALID); goto done; }
    s->blo = lo ? Q_copy(c, lo) : NULL;
    s->bhi = hi ? Q_copy(c, hi) : NULL;
    s->info.lo_bounded = lo != NULL; s->info.hi_bounded = hi != NULL;
    s->info.degree = F->deg;
    if (F->deg < 0) { s->info.kind = TR_INV_WHOLE; goto done; }
    if (F->deg == 0) { s->info.kind = TR_INV_EMPTY; s->info.reason = TR_INV_CONSTANT; goto done; }
    if (!check_one(c, TR_BUDGET_DEGREE, (uint64_t)F->deg)) goto done;
    ip_primitive(c, F, 1);
    /* Yun's square-free factorization over Z[x]: F = prod a_i^i (up to a constant) */
    s->factors = (ipoly *)cmalloc(c, (size_t)F->deg * sizeof(ipoly));
    if (!s->factors) goto done;
    for (int32_t i = 0; i < F->deg; i++) { s->factors[i].deg = -1; s->factors[i].c = NULL; }
    D = ip_deriv(c, F);
    B = ip_gcd(c, F, &D);
    C = ip_divexact(c, F, &B);
    {
        ipoly DB = ip_divexact(c, &D, &B), Cd = ip_deriv(c, &C);
        Dd = ip_sub(c, &DB, &Cd);
        ip_free(&DB); ip_free(&Cd);
    }
    int32_t nf = 0;
    while (!c->err && C.deg > 0 && nf < F->deg) {
        ipoly A = ip_gcd(c, &C, &Dd);
        ipoly C2 = ip_divexact(c, &C, &A);
        ipoly DA = ip_divexact(c, &Dd, &A), C2d = ip_deriv(c, &C2);
        ip_free(&Dd);
        Dd = ip_sub(c, &DA, &C2d);
        ip_free(&DA); ip_free(&C2d); ip_free(&C);
        C = C2;
        s->factors[nf++] = A;
    }
    s->nfactors = nf;
    s->info.nfactors = nf;
    if (c->err) goto done;
    /* the square-free part Q = prod a_i and its Sturm sequence */
    Q = ip_copy(c, &s->factors[0]);
    for (int32_t i = 1; i < nf && !c->err; i++) {
        if (s->factors[i].deg <= 0) continue;
        ipoly t = ip_mul(c, &Q, &s->factors[i]);
        ip_free(&Q);
        Q = t;
    }
    ip_primitive(c, &Q, 1);
    if (c->err) goto done;
    st = sturm_build(c, &Q);
    /* the effective interval: a declared bound, or the Cauchy bound 2^(bits(max |q_i|) + 1) (no root on it) */
    uint64_t mb = 0;
    for (int32_t i = 0; i < Q.deg; i++) { uint64_t bl = tr_bigint_bit_length(Q.c[i]); if (bl > mb) mb = bl; }
    {
        tr_bigint *one = B_i64(c, 1), *R = B_shl(c, one, mb + 1);
        tr_bigq *Rq = Q_int(c, R);
        L = lo ? Q_copy(c, lo) : Q_neg(c, Rq);
        H = hi ? Q_copy(c, hi) : Q_copy(c, Rq);
        tr_bigint_free(one); tr_bigint_free(R); tr_bigq_free(Rq);
    }
    if (c->err) goto done;
    int qL = ip_sign_q(c, &Q, L), qH = ip_sign_q(c, &Q, H);
    if (c->err) goto done;
    if (qL == 0) { root_t *r = add_root(c, s); if (r) set_exact(c, s, r, Q_copy(c, L), TR_INV_HOW_EVALUATED); }
    if (qH == 0 && Q_cmp(c, L, H) != 0) { root_t *r = add_root(c, s); if (r) set_exact(c, s, r, Q_copy(c, H), TR_INV_HOW_EVALUATED); }
    if (c->err || Q_cmp(c, L, H) == 0) goto finish;
    cap = 64;
    stack = (span_t *)cmalloc(c, (size_t)cap * sizeof *stack);
    if (!stack) goto done;
    stack[0].a = L; stack[0].b = H; L = H = NULL;
    stack[0].va = sturm_var(c, &st, stack[0].a); stack[0].vb = sturm_var(c, &st, stack[0].b);
    stack[0].az = qL == 0; stack[0].bz = qH == 0;
    top = 1;
    while (top > 0 && !c->err) {
        span_t sp = stack[--top];
        int cnt = sp.va - sp.vb - (sp.bz ? 1 : 0);
        if (cnt <= 0) { tr_bigq_free(sp.a); tr_bigq_free(sp.b); continue; }
        if (cnt == 1 && !sp.az && !sp.bz) {
            root_t *r = add_root(c, s);
            if (!r) { tr_bigq_free(sp.a); tr_bigq_free(sp.b); break; }
            r->lo = sp.a; r->hi = sp.b; r->how = TR_INV_HOW_ISOLATED;
            for (int32_t i = 0; i < nf && !c->err; i++) {       /* the factor with the sign change */
                if (s->factors[i].deg <= 0) continue;
                int x = ip_sign_q(c, &s->factors[i], r->lo), y = ip_sign_q(c, &s->factors[i], r->hi);
                if (x && y && x != y) { r->factor = i; r->mult = i + 1; r->sign_lo = x; break; }
            }
            if (!c->err && !r->sign_lo) cfail(c, TR_MATH_INVALID);   /* unreachable for a square-free Q */
            if (!c->err) classify(c, s, r);
            continue;
        }
        tr_bigq *m = Q_mid(c, sp.a, sp.b);
        int qm = ip_sign_q(c, &Q, m), vm = sturm_var(c, &st, m);
        if (c->err) { tr_bigq_free(m); tr_bigq_free(sp.a); tr_bigq_free(sp.b); break; }
        if (qm == 0) { root_t *r = add_root(c, s); if (r) set_exact(c, s, r, Q_copy(c, m), TR_INV_HOW_EVALUATED); }
        if (top + 2 > cap) {
            cap *= 2;
            span_t *ns = (span_t *)realloc(stack, (size_t)cap * sizeof *ns);
            if (!ns) { cfail(c, TR_MATH_MEMORY); tr_bigq_free(m); tr_bigq_free(sp.a); tr_bigq_free(sp.b); break; }
            stack = ns;
        }
        span_t right = {Q_copy(c, m), sp.b, vm, sp.vb, qm == 0, sp.bz}, left = {sp.a, m, sp.va, vm, sp.az, qm == 0};
        stack[top++] = right;
        stack[top++] = left;
    }
finish:
    if (!c->err) {
        sort_roots(s, c);
        int32_t ex = 0;
        for (int32_t i = 0; i < s->nroots; i++) ex += s->roots[i].exact;
        s->info.nroots = s->nroots;
        s->info.exact = ex;
        s->info.kind = s->nroots ? TR_INV_FINITE : TR_INV_EMPTY;
    }
done:
    if (s) s->info.iterations = c->iters;
    for (int32_t i = 0; i < top; i++) { tr_bigq_free(stack[i].a); tr_bigq_free(stack[i].b); }
    free(stack);
    tr_bigq_free(L); tr_bigq_free(H);
    sturm_free(&st);
    ip_free(&Q); ip_free(&D); ip_free(&B); ip_free(&C); ip_free(&Dd); ip_free(&P1);
    ip_free(F);
    if (c->err) { tr_inv_roots_free(s); return NULL; }
    return s;
}

tr_inv_roots *tr_inv_isolate(int32_t ncoef, const tr_bigq *const *coef, const tr_bigq *target, const tr_bigq *lo,
                             const tr_bigq *hi, const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error) {
    ctx_t c;
    ctx_init(&c, budget, refusal);
    tr_inv_roots *s = NULL;
    if (ncoef < 1 || !coef) cfail(&c, TR_MATH_INVALID);
    else {
        int32_t top = ncoef - 1;                              /* the degree before cancellation is budgeted first */
        while (top > 0 && Q_zero(coef[top])) top--;
        if (check_one(&c, TR_BUDGET_DEGREE, (uint64_t)top)) {
            tr_bigq **g = (tr_bigq **)cmalloc(&c, (size_t)ncoef * sizeof *g);
            for (int32_t i = 0; i < ncoef && !c.err; i++) g[i] = coef[i] ? Q_copy(&c, coef[i]) : Q_i64(&c, 0);
            if (!c.err && target && g) g[0] = qset(g[0], Q_sub(&c, g[0], target));
            ipoly F;
            F.deg = -1; F.c = NULL;
            if (!c.err) F = ip_from_rationals(&c, ncoef, (const tr_bigq *const *)g);
            if (g) qfree_array(g, (uint64_t)ncoef);
            if (!c.err) s = isolate_core(&c, &F, lo, hi);
            else ip_free(&F);
        }
    }
    if (error) *error = c.err;
    return c.err ? NULL : s;
}
tr_inv_roots *tr_inv_isolate_poly(const tr_math_poly *p, int32_t variable, const tr_bigq *target, const tr_bigq *lo,
                                  const tr_bigq *hi, const tr_math_budget *budget, tr_math_refusal *refusal,
                                  int32_t *error) {
    ctx_t c;
    ctx_init(&c, budget, refusal);
    tr_inv_roots *s = NULL;
    int32_t nt = tr_math_poly_term_count(p), deg = 0;
    if (nt < 0 || variable < 0 || variable >= TR_MATH_VARIABLES) cfail(&c, TR_MATH_INVALID);
    tr_bigq *coef[TR_MATH_DEGREE + 1];
    for (int32_t i = 0; i <= TR_MATH_DEGREE; i++) coef[i] = NULL;
    for (int32_t t = 0; t < nt && !c.err; t++) {
        tr_math_term term;
        if (tr_math_poly_get_term(p, t, &term)) { cfail(&c, TR_MATH_INVALID); break; }
        for (int32_t v = 0; v < TR_MATH_VARIABLES; v++)
            if (v != variable && term.exponents[v]) cfail(&c, TR_MATH_DOMAIN_ERROR);
        if (c.err) break;
        int32_t e = term.exponents[variable];
        if (e > deg) deg = e;
        tr_num_rational q = {term.coefficient.numerator, term.coefficient.denominator};
        int32_t er = 0;
        coef[e] = tr_bigq_from_q64(q, c.b, c.r, &er);
        if (!coef[e]) cfail(&c, er);
    }
    if (!c.err) {
        s = tr_inv_isolate(deg + 1, (const tr_bigq *const *)coef, target, lo, hi, c.b, refusal, &c.err);
    }
    for (int32_t i = 0; i <= TR_MATH_DEGREE; i++) tr_bigq_free(coef[i]);
    if (error) *error = c.err;
    return c.err ? NULL : s;
}

static tr_inv_roots *roots_copy(ctx_t *c, const tr_inv_roots *s) {
    tr_inv_roots *o = (tr_inv_roots *)cmalloc(c, sizeof *o);
    if (!o) return NULL;
    o->magic = ROOTS_MAGIC;
    o->info = s->info;
    o->blo = s->blo ? Q_copy(c, s->blo) : NULL;
    o->bhi = s->bhi ? Q_copy(c, s->bhi) : NULL;
    o->nfactors = s->nfactors;
    if (s->nfactors) {
        o->factors = (ipoly *)cmalloc(c, (size_t)s->nfactors * sizeof(ipoly));
        for (int32_t i = 0; i < s->nfactors && o->factors; i++) o->factors[i] = ip_copy(c, &s->factors[i]);
    }
    if (s->nroots) {
        o->roots = (root_t *)cmalloc(c, (size_t)s->nroots * sizeof(root_t));
        o->cap = s->nroots;
        for (int32_t i = 0; i < s->nroots && o->roots; i++) {
            o->roots[i] = s->roots[i];
            o->roots[i].lo = Q_copy(c, s->roots[i].lo);
            o->roots[i].hi = s->roots[i].hi ? Q_copy(c, s->roots[i].hi) : NULL;
            o->nroots = i + 1;
        }
    }
    if (c->err) { tr_inv_roots_free(o); return NULL; }
    return o;
}
tr_inv_roots *tr_inv_roots_refine(const tr_inv_roots *s, const tr_bigq *width, tr_inv_signs_fn fn,
                                  const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error) {
    ctx_t c;
    ctx_init(&c, budget, refusal);
    tr_inv_roots *o = NULL;
    tr_bigq **pts = NULL;
    uint32_t *pidx = NULL;
    int32_t *which = NULL;
    int8_t *sg = NULL;
    uint64_t base = 0;                                         /* this call's iterations are budgeted on their own */
    if (!s || s->magic != ROOTS_MAGIC || !width || Q_sign(width) <= 0) cfail(&c, TR_MATH_INVALID);
    if (!c.err) { base = s->info.iterations; o = roots_copy(&c, s); }
    int32_t n = o ? o->nroots : 0;
    if (!c.err && n) {
        pts = (tr_bigq **)cmalloc(&c, (size_t)n * sizeof *pts);
        pidx = (uint32_t *)cmalloc(&c, (size_t)n * 4);
        which = (int32_t *)cmalloc(&c, (size_t)n * 4);
        sg = (int8_t *)cmalloc(&c, (size_t)n);
    }
    while (!c.err && n) {
        int32_t cnt = 0;
        for (int32_t i = 0; i < n && !c.err; i++) {
            root_t *r = &o->roots[i];
            if (r->exact) continue;
            tr_bigq *w = Q_sub(&c, r->hi, r->lo);
            int wide = Q_cmp(&c, w, width) > 0;
            tr_bigq_free(w);
            if (!wide) continue;
            pts[cnt] = Q_mid(&c, r->lo, r->hi);
            pidx[cnt] = (uint32_t)r->factor;
            which[cnt++] = i;
        }
        if (c.err || !cnt) break;
        if (!tick(&c, (uint64_t)cnt)) break;
        tr_inv_signs *job = signs_build(&c, (uint32_t)o->nfactors, o->factors, (uint64_t)cnt,
                                        (const tr_bigq *const *)pts, pidx);
        if (job) {
            int rc = fn ? fn(&job->v, sg, c.b, c.r) : tr_inv_signs_eval(&job->v, sg, c.b, c.r);
            if (rc) cfail(&c, rc);
        }
        tr_inv_signs_free(job);
        for (int32_t q = 0; q < cnt; q++) {
            root_t *r = &o->roots[which[q]];
            tr_bigq *m = pts[q];
            pts[q] = NULL;
            if (c.err) { tr_bigq_free(m); continue; }
            if (sg[q] == 0) { tr_bigq_free(r->hi); r->hi = NULL; tr_bigq_free(r->lo); r->lo = m; r->exact = 1;
                              r->how = TR_INV_HOW_EVALUATED; r->sign_lo = 0; o->info.exact++; }
            else if (sg[q] == r->sign_lo) r->lo = qset(r->lo, m);
            else r->hi = qset(r->hi, m);
        }
        o->info.rounds++;
    }
    if (o) o->info.iterations = tr_budget_add(base, c.iters);
    if (pts) qfree_array(pts, (uint64_t)n);
    free(pidx); free(which); free(sg);
    if (error) *error = c.err;
    if (c.err) { tr_inv_roots_free(o); return NULL; }
    return o;
}
int32_t tr_inv_roots_get_info(const tr_inv_roots *s, tr_inv_roots_info *out) {
    if (!s || s->magic != ROOTS_MAGIC || !out) return TR_MATH_INVALID;
    *out = s->info;
    return TR_MATH_OK;
}
int32_t tr_inv_roots_root(const tr_inv_roots *s, int32_t i, tr_inv_root_info *out, const tr_bigq **lo,
                          const tr_bigq **hi) {
    if (!s || s->magic != ROOTS_MAGIC || i < 0 || i >= s->nroots || !out || !lo || !hi) return TR_MATH_INVALID;
    const root_t *r = &s->roots[i];
    out->exact = r->exact; out->multiplicity = r->mult; out->boundary = r->boundary; out->sign_lo = r->sign_lo;
    out->factor = r->factor; out->how = r->how;
    *lo = r->lo; *hi = r->exact ? r->lo : r->hi;
    return TR_MATH_OK;
}
int32_t tr_inv_roots_factor_degree(const tr_inv_roots *s, int32_t i) {
    if (!s || s->magic != ROOTS_MAGIC || i < 0 || i >= s->nfactors) return TR_MATH_INVALID;
    return s->factors[i].deg;
}
const tr_bigint *tr_inv_roots_factor_coef(const tr_inv_roots *s, int32_t i, int32_t j) {
    if (!s || s->magic != ROOTS_MAGIC || i < 0 || i >= s->nfactors || j < 0 || j > s->factors[i].deg) return NULL;
    return s->factors[i].c[j];
}
const tr_bigq *tr_inv_roots_bound(const tr_inv_roots *s, int32_t hi) {
    if (!s || s->magic != ROOTS_MAGIC) return NULL;
    return hi ? s->bhi : s->blo;
}

/* ================================================================ finite domains */
struct tr_inv_sweep_plan_s {
    uint32_t magic, reserved;
    tr_inv_sweep_view v;
    uint64_t bound_bits;
    tr_gfp *fields;
    uint8_t *exps;
    uint64_t *cres, *vres;
};
struct tr_inv_sweep_s {
    uint32_t magic, reserved;
    tr_inv_sweep_info info;
    uint64_t size[TR_INV_SWEEP_DIMS];
    uint64_t *index;
};
void tr_inv_sweep_plan_free(tr_inv_sweep_plan *p) {
    if (!p || p->magic != PLAN_MAGIC) return;
    p->magic = 0;
    free(p->fields); free(p->exps); free(p->cres); free(p->vres);
    free(p);
}
void tr_inv_sweep_free(tr_inv_sweep *s) {
    if (!s || s->magic != SWEEP_MAGIC) return;
    s->magic = 0;
    free(s->index);
    free(s);
}
uint64_t tr_inv_sweep_device_need(const tr_inv_sweep_view *v) { return v ? tr_inv_sweep_device_bytes(v) : TR_BUDGET_UNKNOWN; }

static tr_bigint *lcm_into(ctx_t *c, tr_bigint *L, const tr_bigint *d) {
    tr_bigint *g = B_gcd(c, L, d), *t = B_divexact(c, d, g), *r = B_mul(c, L, t);
    tr_bigint_free(g); tr_bigint_free(t); tr_bigint_free(L);
    return r;
}
tr_inv_sweep_plan *tr_inv_sweep_prepare(const tr_math_poly *f, const tr_bigq *target, int32_t ndims,
                                        const int32_t *variables, const uint64_t *sizes, const tr_bigq *const *values,
                                        uint64_t start, const tr_math_budget *budget, tr_math_refusal *refusal,
                                        int32_t *error) {
    ctx_t c;
    ctx_init(&c, budget, refusal);
    tr_inv_sweep_plan *p = NULL;
    int32_t nt = tr_math_poly_term_count(f), slot[TR_MATH_VARIABLES];
    uint64_t total = 1, nvalues = 0;
    tr_bigq **coef = NULL;                       /* g = f - target: terms */
    uint8_t *exps = NULL;
    tr_bigint **h = NULL, **mval = NULL, *D[TR_INV_SWEEP_DIMS] = {NULL, NULL, NULL}, *Mx[TR_INV_SWEEP_DIMS] = {NULL, NULL, NULL};
    tr_bigint *Lc = NULL, *bound = NULL;
    uint32_t nterms = 0;
    for (int32_t v = 0; v < TR_MATH_VARIABLES; v++) slot[v] = -1;
    if (nt < 0 || ndims < 1 || !variables || !sizes || !values) { cfail(&c, TR_MATH_INVALID); goto done; }
    if (!check_one(&c, TR_BUDGET_VARIABLES, (uint64_t)ndims)) goto done;
    if (ndims > TR_INV_SWEEP_DIMS) { cfail(&c, TR_MATH_UNSUPPORTED); goto done; }
    for (int32_t d = 0; d < ndims; d++) {
        if (variables[d] < 0 || variables[d] >= TR_MATH_VARIABLES || slot[variables[d]] >= 0 || sizes[d] < 1 ||
            sizes[d] > TR_INV_SWEEP_VALUES_MAX) { cfail(&c, TR_MATH_INVALID); goto done; }
        slot[variables[d]] = d;
        total *= sizes[d];
        nvalues += sizes[d];
    }
    for (uint64_t i = 0; i < nvalues; i++) if (!values[i]) { cfail(&c, TR_MATH_INVALID); goto done; }
    if (start > total) { cfail(&c, TR_MATH_INVALID); goto done; }
    /* the terms of g = f - target, in the searched coordinates */
    coef = (tr_bigq **)cmalloc(&c, (size_t)(nt + 1) * sizeof *coef);
    exps = (uint8_t *)cmalloc(&c, (size_t)(nt + 1) * (size_t)ndims);
    if (c.err) goto done;
    int32_t constant = -1;
    for (int32_t t = 0; t < nt && !c.err; t++) {
        tr_math_term term;
        if (tr_math_poly_get_term(f, t, &term)) { cfail(&c, TR_MATH_INVALID); break; }
        int any = 0;
        for (int32_t v = 0; v < TR_MATH_VARIABLES; v++) {
            if (!term.exponents[v]) continue;
            if (slot[v] < 0) { cfail(&c, TR_MATH_DOMAIN_ERROR); break; }
            exps[(size_t)nterms * ndims + slot[v]] = term.exponents[v];
            any = 1;
        }
        if (c.err) break;
        tr_num_rational q = {term.coefficient.numerator, term.coefficient.denominator};
        int32_t e = 0;
        coef[nterms] = tr_bigq_from_q64(q, c.b, c.r, &e);
        if (!coef[nterms]) { cfail(&c, e); break; }
        if (!any) constant = (int32_t)nterms;
        nterms++;
    }
    if (c.err) goto done;
    if (target && !Q_zero(target)) {
        if (constant >= 0) coef[constant] = qset(coef[constant], Q_sub(&c, coef[constant], target));
        else { coef[nterms] = Q_neg(&c, target); memset(exps + (size_t)nterms * ndims, 0, (size_t)ndims); nterms++; }
    }
    if (!check_one(&c, TR_BUDGET_SUPPORT, nterms)) goto done;
    /* clear denominators: c_t = L coef_t; grid numerators m = value * D_d; h_t = c_t prod_d D_d^(E_d - e_td) */
    Lc = B_i64(&c, 1);
    for (uint32_t t = 0; t < nterms && !c.err; t++) if (!Q_zero(coef[t])) Lc = lcm_into(&c, Lc, tr_bigq_denominator(coef[t]));
    uint32_t E[TR_INV_SWEEP_DIMS] = {0, 0, 0};
    for (uint32_t t = 0; t < nterms; t++)
        for (int32_t d = 0; d < ndims; d++) if (exps[(size_t)t * ndims + d] > E[d]) E[d] = exps[(size_t)t * ndims + d];
    mval = (tr_bigint **)cmalloc(&c, (size_t)nvalues * sizeof *mval);
    h = (tr_bigint **)cmalloc(&c, (size_t)(nterms ? nterms : 1) * sizeof *h);
    if (c.err) goto done;
    uint64_t base = 0;
    for (int32_t d = 0; d < ndims && !c.err; d++) {
        D[d] = B_i64(&c, 1);
        for (uint64_t i = 0; i < sizes[d] && !c.err; i++) D[d] = lcm_into(&c, D[d], tr_bigq_denominator(values[base + i]));
        Mx[d] = B_i64(&c, 0);
        for (uint64_t i = 0; i < sizes[d] && !c.err; i++) {
            tr_bigint *t = B_divexact(&c, D[d], tr_bigq_denominator(values[base + i]));
            mval[base + i] = B_mul(&c, tr_bigq_numerator(values[base + i]), t);
            tr_bigint_free(t);
            tr_bigint *a = B_abs(&c, mval[base + i]);
            if (!c.err && tr_bigint_cmp(a, Mx[d]) > 0) Mx[d] = bset(Mx[d], a); else tr_bigint_free(a);
        }
        base += sizes[d];
    }
    bound = B_i64(&c, 0);
    for (uint32_t t = 0; t < nterms && !c.err; t++) {
        tr_bigint *x = Q_zero(coef[t]) ? B_i64(&c, 0) : B_divexact(&c, Lc, tr_bigq_denominator(coef[t]));
        x = bset(x, B_mul(&c, x, tr_bigq_numerator(coef[t])));
        tr_bigint *bt = B_abs(&c, x);
        for (int32_t d = 0; d < ndims && !c.err; d++) {
            uint32_t e = exps[(size_t)t * ndims + d];
            if (E[d] > e) { tr_bigint *pw = B_pow(&c, D[d], E[d] - e); x = bset(x, B_mul(&c, x, pw)); bt = bset(bt, B_mul(&c, bt, pw)); tr_bigint_free(pw); }
            if (e) { tr_bigint *pw = B_pow(&c, Mx[d], e); bt = bset(bt, B_mul(&c, bt, pw)); tr_bigint_free(pw); }
        }
        h[t] = x;
        bound = bset(bound, B_add(&c, bound, bt));
        tr_bigint_free(bt);
    }
    if (c.err) goto done;
    uint64_t bbits = tr_bigint_bit_length(bound);              /* |H(m)| <= bound < 2^bbits */
    uint32_t k = primes_for(&c, bbits);
    if (!k) goto done;
    uint64_t remaining = total - start, limit = c.b->limit[TR_BUDGET_ITERATIONS];
    uint64_t count = remaining < limit ? remaining : limit;    /* truncation, reported, never a refusal */
    tr_math_need need;
    memset(&need, 0, sizeof need);
    uint64_t bytes = sizeof(tr_inv_sweep_plan) + (uint64_t)k * sizeof(tr_gfp) + (uint64_t)nterms * (uint64_t)ndims;
    bytes = tr_budget_add(bytes, tr_budget_mul(tr_budget_mul(k, tr_budget_add(nterms, nvalues)), 8));
    bytes = tr_budget_add(bytes, tr_budget_bitmap_bytes(count));
    need.need[TR_BUDGET_BYTES] = bytes;
    int32_t rc = tr_budget_check(c.b, &need, c.r);
    if (rc) { c.err = rc; goto done; }
    p = (tr_inv_sweep_plan *)cmalloc(&c, sizeof *p);
    if (!p) goto done;
    p->magic = PLAN_MAGIC;
    fill_fields(&c, k, &p->fields, NULL, NULL);
    p->exps = (uint8_t *)cmalloc(&c, (size_t)(nterms ? nterms : 1) * (size_t)ndims);
    p->cres = (uint64_t *)cmalloc(&c, (size_t)k * (nterms ? nterms : 1) * 8);
    p->vres = (uint64_t *)cmalloc(&c, (size_t)k * nvalues * 8);
    if (c.err) goto done;
    memcpy(p->exps, exps, (size_t)nterms * (size_t)ndims);
    for (uint32_t j = 0; j < k; j++) {
        for (uint32_t t = 0; t < nterms; t++) tr_bigint_mod_u64(h[t], TR_INV_PRIMES[j], &p->cres[(uint64_t)j * nterms + t]);
        for (uint64_t i = 0; i < nvalues; i++) tr_bigint_mod_u64(mval[i], TR_INV_PRIMES[j], &p->vres[(uint64_t)j * nvalues + i]);
    }
    p->bound_bits = bbits;
    tr_inv_sweep_view *v = &p->v;
    v->start = start; v->count = count; v->total = total; v->nvalues = nvalues;
    for (int32_t d = 0, off = 0; d < TR_INV_SWEEP_DIMS; d++) {
        v->size[d] = d < ndims ? sizes[d] : 1;
        v->voff[d] = (uint64_t)off;
        if (d < ndims) off += (int32_t)sizes[d];
    }
    v->nprimes = k; v->nterms = nterms; v->ndims = ndims;
    v->fields = p->fields; v->exps = p->exps; v->cres = p->cres; v->vres = p->vres;
done:
    if (coef) qfree_array(coef, (uint64_t)nt + 1);
    free(exps);
    if (h) { for (uint32_t t = 0; t < nterms; t++) tr_bigint_free(h[t]); free(h); }
    if (mval) { for (uint64_t i = 0; i < nvalues; i++) tr_bigint_free(mval[i]); free(mval); }
    for (int32_t d = 0; d < TR_INV_SWEEP_DIMS; d++) { tr_bigint_free(D[d]); tr_bigint_free(Mx[d]); }
    tr_bigint_free(Lc); tr_bigint_free(bound);
    if (error) *error = c.err;
    if (c.err) { if (p) tr_inv_sweep_plan_free(p); return NULL; }
    return p;
}
int32_t tr_inv_sweep_get_view(const tr_inv_sweep_plan *p, tr_inv_sweep_view *out) {
    if (!p || p->magic != PLAN_MAGIC || !out) return TR_MATH_INVALID;
    *out = p->v;
    return TR_MATH_OK;
}
int32_t tr_inv_sweep_eval(const tr_inv_sweep_view *v, uint64_t *bitmap) {
    if (!tr_inv_sweep_view_valid(v) || (v->count && !bitmap)) return TR_MATH_INVALID;
    memset(bitmap, 0, (size_t)((v->count + 63) / 64) * 8);
    for (uint64_t i = 0; i < v->count; i++)
        if (tr_inv_sweep_point(v, v->start + i)) bitmap[i >> 6] |= (uint64_t)1 << (i & 63);
    return TR_MATH_OK;
}
tr_inv_sweep *tr_inv_sweep_finish(const tr_inv_sweep_plan *p, const uint64_t *bitmap, const tr_math_budget *budget,
                                  tr_math_refusal *refusal, int32_t *error) {
    ctx_t c;
    ctx_init(&c, budget, refusal);
    tr_inv_sweep *s = NULL;
    if (!p || p->magic != PLAN_MAGIC || (p->v.count && !bitmap)) { cfail(&c, TR_MATH_INVALID); goto out; }
    const tr_inv_sweep_view *v = &p->v;
    uint64_t limit = c.b->limit[TR_BUDGET_OUTPUT], found = 0, searched = v->count;
    for (uint64_t w = 0; w < (v->count + 63) / 64; w++) found += (uint64_t)__builtin_popcountll(bitmap[w]);
    uint64_t keep = found < limit ? found : limit;
    if (!check_one(&c, TR_BUDGET_BYTES, tr_budget_add(sizeof *s, tr_budget_mul(keep, 8)))) goto out;
    s = (tr_inv_sweep *)cmalloc(&c, sizeof *s);
    if (s) s->index = (uint64_t *)cmalloc(&c, (size_t)(keep ? keep : 1) * 8);
    if (c.err) goto out;
    s->magic = SWEEP_MAGIC;
    uint64_t listed = 0;
    int32_t reason = TR_INV_SEARCH_COMPLETE;
    for (uint64_t i = 0; i < v->count; i++) {
        if (!((bitmap[i >> 6] >> (i & 63)) & 1)) continue;
        if (listed == keep && found > limit) { searched = i; reason = TR_INV_SEARCH_OUTPUT; break; }
        s->index[listed++] = v->start + i;
    }
    if (reason == TR_INV_SEARCH_COMPLETE && v->start + v->count < v->total) reason = TR_INV_SEARCH_ITERATIONS;
    s->info.total = v->total; s->info.start = v->start; s->info.searched = searched; s->info.next = v->start + searched;
    s->info.matches = listed; s->info.bound_bits = p->bound_bits;
    s->info.complete = reason == TR_INV_SEARCH_COMPLETE; s->info.reason = reason;
    s->info.ndims = v->ndims; s->info.nprimes = (int32_t)v->nprimes;
    for (int32_t d = 0; d < TR_INV_SWEEP_DIMS; d++) s->size[d] = v->size[d];
out:
    if (error) *error = c.err;
    if (c.err) { if (s) { s->magic = SWEEP_MAGIC; tr_inv_sweep_free(s); } return NULL; }
    return s;
}
int32_t tr_inv_sweep_get_info(const tr_inv_sweep *s, tr_inv_sweep_info *out) {
    if (!s || s->magic != SWEEP_MAGIC || !out) return TR_MATH_INVALID;
    *out = s->info;
    return TR_MATH_OK;
}
int32_t tr_inv_sweep_match(const tr_inv_sweep *s, uint64_t i, uint64_t *index, uint64_t *coords) {
    if (!s || s->magic != SWEEP_MAGIC || i >= s->info.matches || !index) return TR_MATH_INVALID;
    uint64_t x = s->index[i], rem = x;
    *index = x;
    if (coords)
        for (int32_t d = s->info.ndims; d-- > 0;) { coords[d] = rem % s->size[d]; rem /= s->size[d]; }
    return TR_MATH_OK;
}
