/* KERNEL-002: approximate spectral feature kernels; every contract is in math_kapprox.h.
 * SPDX-License-Identifier: Apache-2.0 */
#include "math_kapprox.h"
#include "mathematics.h"
#include "math_handles.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

TR_MATH_ABI(tr_kapprox_spec, TR_KAPPROX_ABI);        /* the ctypes boundary of KERNEL-002 (math_handles.h) */
TR_MATH_ABI(tr_kapprox_plan_view, TR_KAPPROX_ABI);
TR_MATH_ABI(tr_kapprox_group, TR_KAPPROX_ABI);
TR_MATH_ABI(tr_kapprox_request, TR_KAPPROX_ABI);
TR_MATH_ABI(tr_kapprox_result, TR_KAPPROX_ABI);
TR_MATH_ABI(tr_kapprox_info, TR_KAPPROX_ABI);

_Static_assert(sizeof(tr_kapprox_spec) == 176 && sizeof(tr_kapprox_plan_view) == 312 &&
               sizeof(tr_kapprox_group) == 80 && sizeof(tr_kapprox_request) == 40 &&
               sizeof(tr_kapprox_result) == 144 && sizeof(tr_kapprox_info) == 208, "KERNEL-002 ABI");

#define PLAN_MAGIC 0x3150414bu      /* "KAP1" */
#define STATE_MAGIC 0x3153414bu     /* "KAS1" */
#define LN_TERMS 40                 /* atanh terms of the certified logarithm (tail below 2^-131) */
#define FBITS_MAX 56                /* the finest frequency grid 2^-56 */

/* floor(2^576 / ln 2), little-endian limbs (the tests recompute it independently with Python's decimal module) */
static const uint64_t INV_LN2_576[10] = {
    0x5fc529264c2fb3abULL, 0x897f5e06a7be7366ULL, 0xd52173cc1895213fULL, 0x49b25eeb82d7c167ULL, 0xbc3887eeaa2ed9acULL,
    0x164a2cd9a342648fULL, 0xd6aef551bad2b4b1ULL, 0x7d0ffda0d23a7d11ULL, 0x71547652b82fe177ULL, 0x1ULL};

static void seterr(int32_t *e, int32_t v) { if (e) *e = v; }
static const tr_math_budget *budget_of(const tr_math_budget *b, tr_math_budget *store) {
    if (b) return b;
    tr_budget_default(store);
    return store;
}
static void *zalloc(uint64_t bytes) {
    if (bytes > (uint64_t)SIZE_MAX / 2) return NULL;
    return calloc(1, bytes ? (size_t)bytes : 1);
}
static void need_zero(tr_math_need *n) { memset(n, 0, sizeof *n); }
static int32_t bitlen128(unsigned __int128 x) {
    uint64_t hi = (uint64_t)(x >> 64), lo = (uint64_t)x;
    return hi ? 128 - __builtin_clzll(hi) : lo ? 64 - __builtin_clzll(lo) : 0;
}
static int32_t bitlen192(const uint64_t *s) {
    if (s[2]) return 192 - __builtin_clzll(s[2]);
    return bitlen128(((unsigned __int128)s[1] << 64) | s[0]);
}

/* ================================================================ exact big numbers (an arena per computation) */
typedef struct {
    const tr_math_budget *b;
    tr_math_refusal *r;
    int32_t rc;
    void **obj;
    uint8_t *kind;                  /* 0 bigint, 1 bigq */
    size_t n, cap;
} cx;
static void cx_init(cx *c, const tr_math_budget *b, tr_math_refusal *r) {
    c->b = b; c->r = r; c->rc = 0; c->obj = NULL; c->kind = NULL; c->n = c->cap = 0;
}
static void obj_free(void *x, int kind) { if (kind) tr_bigq_free((tr_bigq *)x); else tr_bigint_free((tr_bigint *)x); }
static void cx_done(cx *c) {
    for (size_t i = 0; i < c->n; i++) obj_free(c->obj[i], c->kind[i]);
    free(c->obj); free(c->kind);
    c->obj = NULL; c->kind = NULL; c->n = c->cap = 0;
}
static void *cx_track(cx *c, void *x, int kind, int32_t e) {
    if (!x) { if (!c->rc) c->rc = e ? e : TR_MATH_MEMORY; return NULL; }
    if (c->n == c->cap) {
        size_t nc = c->cap ? 2 * c->cap : 64;
        void **o = (void **)realloc(c->obj, nc * sizeof(void *));
        if (o) c->obj = o;
        uint8_t *k = o ? (uint8_t *)realloc(c->kind, nc) : NULL;
        if (k) c->kind = k;
        if (!o || !k) { obj_free(x, kind); if (!c->rc) c->rc = TR_MATH_MEMORY; return NULL; }
        c->cap = nc;
    }
    c->obj[c->n] = x; c->kind[c->n] = (uint8_t)kind; c->n++;
    return x;
}
static tr_bigint *Zk(cx *c, tr_bigint *x, int32_t e) { return (tr_bigint *)cx_track(c, x, 0, e); }
static tr_bigq *Qk(cx *c, tr_bigq *x, int32_t e) { return (tr_bigq *)cx_track(c, x, 1, e); }
static tr_bigint *Z_i64(cx *c, int64_t v) {
    if (c->rc) return NULL;
    int32_t e = 0;
    return Zk(c, tr_bigint_from_i64(v, c->b, c->r, &e), e);
}
static tr_bigint *Z_i128(cx *c, __int128 v) {
    if (c->rc) return NULL;
    int32_t e = 0;
    return Zk(c, tr_bigint_from_i128(tr_i128_set(v), c->b, c->r, &e), e);
}
static tr_bigint *Z_u128(cx *c, unsigned __int128 v) {
    if (c->rc) return NULL;
    uint64_t l[2] = {(uint64_t)v, (uint64_t)(v >> 64)};
    int32_t e = 0;
    return Zk(c, tr_bigint_from_limbs(v ? 1 : 0, l, 2, c->b, c->r, &e), e);
}
static tr_bigint *Z_limbs(cx *c, const uint64_t *l, uint64_t n) {
    if (c->rc) return NULL;
    int32_t e = 0;
    return Zk(c, tr_bigint_from_limbs(1, l, n, c->b, c->r, &e), e);
}
static tr_bigint *Z_twos(cx *c, const uint64_t *w, uint64_t n) {
    if (c->rc) return NULL;
    int32_t e = 0;
    return Zk(c, tr_bigint_from_twos(w, n, c->b, c->r, &e), e);
}
#define ZBIN(name, fn) \
    static tr_bigint *name(cx *c, const tr_bigint *a, const tr_bigint *b) { \
        int32_t e = 0; \
        if (c->rc || !a || !b) return NULL; \
        return Zk(c, fn(a, b, c->b, c->r, &e), e); \
    }
ZBIN(Z_add, tr_bigint_add)
ZBIN(Z_sub, tr_bigint_sub)
ZBIN(Z_mul, tr_bigint_mul)
#undef ZBIN
static tr_bigint *Z_shl(cx *c, const tr_bigint *a, uint64_t k) {
    if (c->rc || !a) return NULL;
    int32_t e = 0;
    return Zk(c, tr_bigint_shl(a, k, c->b, c->r, &e), e);
}
static tr_bigint *Z_isqrt(cx *c, const tr_bigint *a) {
    if (c->rc || !a) return NULL;
    int32_t e = 0;
    return Zk(c, tr_bigint_isqrt(a, c->b, c->r, &e), e);
}
static tr_bigint *Z_pow(cx *c, const tr_bigint *a, uint64_t k) {
    if (c->rc || !a) return NULL;
    int32_t e = 0;
    return Zk(c, tr_bigint_pow(a, k, c->b, c->r, &e), e);
}
static tr_bigint *Z_floordiv(cx *c, const tr_bigint *a, const tr_bigint *b) {
    if (c->rc || !a || !b) return NULL;
    tr_bigint *q = NULL;
    int32_t rc = tr_bigint_divmod(a, b, TR_DIV_FLOOR, &q, NULL, c->b, c->r);
    if (rc) { if (!c->rc) c->rc = rc; return NULL; }
    return Zk(c, q, 0);
}
static tr_bigint *Z_neg(cx *c, const tr_bigint *a) {
    if (c->rc || !a) return NULL;
    int32_t e = 0;
    return Zk(c, tr_bigint_neg(a, c->b, c->r, &e), e);
}
static tr_bigint *Z_ceildiv(cx *c, const tr_bigint *a, const tr_bigint *b) { return Z_neg(c, Z_floordiv(c, Z_neg(c, a), b)); }
static tr_bigint *Z_pow10(cx *c, uint64_t k) { return Z_pow(c, Z_i64(c, 10), k); }
static tr_bigq *Q_z(cx *c, const tr_bigint *a) {
    if (c->rc || !a) return NULL;
    int32_t e = 0;
    return Qk(c, tr_bigq_from_bigint(a, c->b, c->r, &e), e);
}
static tr_bigq *Q_frac(cx *c, const tr_bigint *n, const tr_bigint *d) {
    if (c->rc || !n || !d) return NULL;
    int32_t e = 0;
    return Qk(c, tr_bigq_from_bigints(n, d, c->b, c->r, &e), e);
}
static tr_bigq *Q_q64(cx *c, tr_num_rational q) {
    if (c->rc) return NULL;
    int32_t e = 0;
    return Qk(c, tr_bigq_from_q64(q, c->b, c->r, &e), e);
}
static tr_bigq *Q_int(cx *c, int64_t v) { tr_num_rational q = {v, 1}; return Q_q64(c, q); }
#define QBIN(name, fn) \
    static tr_bigq *name(cx *c, const tr_bigq *a, const tr_bigq *b) { \
        int32_t e = 0; \
        if (c->rc || !a || !b) return NULL; \
        return Qk(c, fn(a, b, c->b, c->r, &e), e); \
    }
QBIN(Q_add, tr_bigq_add)
QBIN(Q_sub, tr_bigq_sub)
QBIN(Q_mul, tr_bigq_mul)
QBIN(Q_div, tr_bigq_div)
#undef QBIN
static tr_bigq *Q_neg(cx *c, const tr_bigq *a) {
    if (c->rc || !a) return NULL;
    int32_t e = 0;
    return Qk(c, tr_bigq_neg(a, c->b, c->r, &e), e);
}
static tr_bigq *Q_abs(cx *c, tr_bigq *a) { return (!a || c->rc) ? NULL : tr_bigq_sign(a) < 0 ? Q_neg(c, a) : a; }
/* 2^k for any integer k */
static tr_bigq *Q_pow2(cx *c, int64_t k) {
    tr_bigint *one = Z_i64(c, 1);
    if (k >= 0) return Q_z(c, Z_shl(c, one, (uint64_t)k));
    return Q_frac(c, one, Z_shl(c, one, (uint64_t)(-k)));
}
static int Q_cmp(const tr_bigq *a, const tr_bigq *b) { return tr_bigq_cmp(a, b); }
static int Q_is_zero(const tr_bigq *a) { return tr_bigq_sign(a) == 0; }
/* an untracked copy owned by the caller */
static tr_bigq *Q_out(cx *c, const tr_bigq *a) {
    if (c->rc || !a) return NULL;
    int32_t e = 0;
    tr_bigq *x = tr_bigq_copy(a, c->b, c->r, &e);
    if (!x && !c->rc) c->rc = e ? e : TR_MATH_MEMORY;
    return x;
}
/* floor(q) as a bigint */
static tr_bigint *Q_floor(cx *c, const tr_bigq *q) {
    if (c->rc || !q) return NULL;
    return Z_floordiv(c, tr_bigq_numerator(q), tr_bigq_denominator(q));
}
static tr_bigint *Q_ceil(cx *c, const tr_bigq *q) {
    if (c->rc || !q) return NULL;
    return Z_ceildiv(c, tr_bigq_numerator(q), tr_bigq_denominator(q));
}
/* an exact rational >= sqrt(q) for q >= 0: (isqrt(n d 4^64) + 1) / (d 2^64) */
static tr_bigq *Q_sqrt_up(cx *c, const tr_bigq *q) {
    if (c->rc || !q) return NULL;
    if (Q_is_zero(q)) return Q_int(c, 0);
    tr_bigint *x = Z_shl(c, Z_mul(c, tr_bigq_numerator(q), tr_bigq_denominator(q)), 128);
    tr_bigint *r = Z_add(c, Z_isqrt(c, x), Z_i64(c, 1));
    return Q_frac(c, r, Z_shl(c, tr_bigq_denominator(q), 64));
}

/* ================================================================ certified fixed-point transforms (host only) */
static uint64_t sm64(uint64_t z) {
    z += UINT64_C(0x9E3779B97F4A7C15);
    z = (z ^ (z >> 30)) * UINT64_C(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94D049BB133111EB);
    return z ^ (z >> 31);
}
static uint64_t gen_key(uint64_t seed) { return sm64(seed ^ UINT64_C(0xD1B54A32D192ED03)); }   /* stream 0 */
static unsigned __int128 gen_u128(uint64_t key, uint64_t i) {
    return (unsigned __int128)sm64(key + i) | ((unsigned __int128)sm64(key + i + 1) << 64);
}
/* floor(a 2^120 / b) for 0 <= a < b < 2^126 (binary long division) */
static unsigned __int128 div_shift120(unsigned __int128 a, unsigned __int128 b) {
    unsigned __int128 q = 0, r = a;
    for (int i = 0; i < 120; i++) {
        r <<= 1; q <<= 1;
        if (r >= b) { r -= b; q |= 1; }
    }
    return q;
}
/* ln(M / 2^120) for 2^120 <= M < 2^121 in [lo, hi] / 2^120: 2 atanh(y), y = (M - 2^120) / (M + 2^120) < 1/3 floored,
 * y^2 floored, t_k = floor(t_(k-1) y^2) and floor(t_k / (2k + 1)) are each at most 4 ulps below their exact values, the
 * tail after LN_TERMS terms is below an ulp: atanh in (S, S + 4 (LN_TERMS + 1) + 1] */
static void ln_mantissa(unsigned __int128 M, unsigned __int128 *lo, unsigned __int128 *hi) {
    const unsigned __int128 one = (unsigned __int128)1 << TR_CALC_F;
    if (M == one) { *lo = 0; *hi = 0; return; }
    unsigned __int128 y = div_shift120(M - one, M + one);
    unsigned __int128 y2 = tr_calc_fmul(y, y), t = y, S = y;
    for (uint32_t k = 1; k <= LN_TERMS; k++) {
        t = tr_calc_fmul(t, y2);
        S += tr_calc_divsmall(t, 2 * k + 1);
    }
    *lo = 2 * S;
    *hi = 2 * S + 8 * (LN_TERMS + 1) + 2;
}
/* the mantissa M / 2^120 in [1, 2) of x = 2^e (M / 2^120); trunc: low bits dropped (ln then up to 1 ulp larger) */
static void mantissa(unsigned __int128 x, int32_t e, unsigned __int128 *M, int *trunc) {
    if (e <= TR_CALC_F) { *M = x << (TR_CALC_F - e); *trunc = 0; }
    else { *M = x >> (e - TR_CALC_F); *trunc = (x & ((((unsigned __int128)1) << (e - TR_CALC_F)) - 1)) != 0; }
}
/* -ln U for U = (2N + 1) / 2^129: [lo, hi] / 2^120 (-ln U = (129 - e) ln 2 - ln m, 2N + 1 = 2^e m) */
static void neg_ln_uniform(unsigned __int128 N, unsigned __int128 *lo, unsigned __int128 *hi) {
    int32_t e;
    unsigned __int128 M;
    int trunc = 0;
    if (N >> 127) { e = 128; M = N >> 7; trunc = 1; }                /* (2N + 1) >> 8 = N >> 7: bits dropped */
    else { unsigned __int128 x = 2 * N + 1; e = bitlen128(x) - 1; mantissa(x, e, &M, &trunc); }
    unsigned __int128 a, b;
    ln_mantissa(M, &a, &b);
    if (trunc) b += 1;
    unsigned __int128 k = (unsigned __int128)(129 - e);
    unsigned __int128 L = k * TR_CALC_LN2, H = k * (TR_CALC_LN2 + 1);
    *lo = L > b ? L - b : 0;                                          /* -ln U > 0 */
    *hi = H - a;
}
/* ln n for an integer n >= 1: [lo, hi] / 2^120 */
static void ln_int(unsigned __int128 n, unsigned __int128 *lo, unsigned __int128 *hi) {
    int32_t e = bitlen128(n) - 1;
    unsigned __int128 M, a, b;
    int trunc;
    mantissa(n, e, &M, &trunc);
    ln_mantissa(M, &a, &b);
    if (trunc) b += 1;
    *lo = (unsigned __int128)e * TR_CALC_LN2 + a;
    *hi = (unsigned __int128)e * (TR_CALC_LN2 + 1) + b;
}
/* sin and cos of 2 pi U (quarter = 0) or pi U (quarter = 1) for U = (2N + 1) / 2^129: the exact quarter-turn
 * reduction of SPEC-001 (tr_fourier_finish) and CALC-003's certified series */
static void sincos_uniform(unsigned __int128 N, int half, tr_calc_fx *sn, tr_calc_fx *cs) {
    uint64_t acc[6] = {0, 0, 0, 0, 0, 0};
    unsigned __int128 f;
    uint64_t q;
    if (!half) { q = (uint64_t)(N >> 126); f = (N << 2) + 2; }       /* 4U = (2N + 1) / 2^127 */
    else { q = (uint64_t)(N >> 127); f = (N << 1) + 1; }             /* 2U = (2N + 1) / 2^128 */
    acc[1] = (uint64_t)f; acc[2] = (uint64_t)(f >> 64); acc[3] = q;
    tr_fourier_finish(acc, 0, sn, cs);
}

/* ================================================================ the generator */
typedef struct {
    int32_t kernel, d, p;
    uint64_t key, npairs;
    tr_bigint *slo, *shi;           /* GAUSSIAN: sqrt(2 gamma) in [slo, shi] / 2^120 */
    tr_bigq *gam, *eta_num, *eta, *pbad;
} gen_t;
/* the constants of the generator (tracked in c): the precision p, the a priori bounds eta_num (the enclosure's half
 * width), eta (cell + enclosure + rounding) and P_bad (see the header comment) */
static int32_t gen_setup(cx *c, int32_t kernel, tr_num_rational gamma, int32_t d, uint64_t seed, gen_t *g) {
    memset(g, 0, sizeof *g);
    g->kernel = kernel; g->d = d; g->key = gen_key(seed); g->npairs = (uint64_t)(d + 1) / 2;
    g->gam = Q_q64(c, gamma);
    tr_bigq *omax = NULL, *cell = NULL, *num = NULL;
    if (kernel == TR_KAPPROX_GAUSSIAN) {
        tr_bigint *n2 = Z_shl(c, Z_i64(c, gamma.numerator), 241), *den = Z_i64(c, gamma.denominator);   /* 2 gamma 2^240 */
        g->slo = Z_isqrt(c, Z_floordiv(c, n2, den));
        g->shi = Z_add(c, Z_isqrt(c, Z_ceildiv(c, n2, den)), Z_i64(c, 1));
        tr_bigq *splus = Q_frac(c, g->shi, Z_shl(c, Z_i64(c, 1), 120));
        omax = Q_mul(c, splus, Q_int(c, 14));                          /* |z| <= 13.4 on every cell */
        cell = Q_mul(c, splus, Q_pow2(c, -62));
        num = Q_add(c, Q_mul(c, splus, Q_pow2(c, -53)), Q_pow2(c, -110));
        g->pbad = Q_mul(c, Q_int(c, (int64_t)g->npairs), Q_pow2(c, -64));   /* U1 < 2^-64 */
    } else if (kernel == TR_KAPPROX_LAPLACE) {
        omax = Q_mul(c, g->gam, Q_pow2(c, 31));                        /* |cot pi U| <= 2^31 on the good cells */
        cell = Q_mul(c, g->gam, Q_pow2(c, -65));
        num = Q_add(c, Q_mul(c, g->gam, Q_pow2(c, -47)), Q_pow2(c, -110));
        g->pbad = Q_mul(c, Q_int(c, d), Q_pow2(c, -31));               /* U < 2^-32 or U > 1 - 2^-32 */
    } else {
        omax = Q_mul(c, g->gam, Q_int(c, 90));                         /* -ln U <= 89.5 */
        cell = Q_mul(c, g->gam, Q_pow2(c, -64));
        num = Q_add(c, Q_mul(c, g->gam, Q_pow2(c, -100)), Q_pow2(c, -110));
        g->pbad = Q_mul(c, Q_int(c, d), Q_pow2(c, -64));               /* U < 2^-64 */
    }
    tr_bigint *top = Q_ceil(c, omax);
    if (c->rc) return c->rc;
    int32_t bits = (int32_t)tr_bigint_bit_length(top);
    int32_t p = 61 - bits;
    if (p > FBITS_MAX) p = FBITS_MAX;
    if (p < 1) return TR_MATH_UNSUPPORTED;                             /* frequencies past 2^60: refused */
    g->p = p;
    g->eta_num = num;
    g->eta = Q_add(c, Q_add(c, cell, num), Q_pow2(c, -(int64_t)p - 1));
    return c->rc;
}
/* omega~ = round(mid 2^p) / 2^p of an enclosure [lo, hi] whose half width must be within eta_num */
static int32_t round_omega(cx *c, const gen_t *g, tr_bigq *lo, tr_bigq *hi, tr_num_rational *out) {
    tr_bigq *half = Q_pow2(c, -1);
    tr_bigq *hw = Q_mul(c, Q_sub(c, hi, lo), half);
    if (c->rc) return c->rc;
    if (tr_bigq_sign(hw) < 0 || Q_cmp(hw, g->eta_num) > 0) return TR_MATH_OVERFLOW;   /* never: the a priori bound */
    tr_bigq *x = Q_add(c, Q_mul(c, Q_mul(c, Q_add(c, lo, hi), half), Q_pow2(c, g->p)), half);
    tr_bigint *M = Q_floor(c, x);
    if (c->rc) return c->rc;
    int64_t m;
    if (tr_bigint_to_i64(M, &m) || m > (INT64_C(1) << 62) || m < -(INT64_C(1) << 62)) return TR_MATH_OVERFLOW;
    if (!tr_q_make(m, INT64_C(1) << g->p, out)) return TR_MATH_OVERFLOW;
    return TR_MATH_OK;
}
static tr_bigq *Q_fx(cx *c, __int128 v) { return Q_frac(c, Z_i128(c, v), Z_shl(c, Z_i64(c, 1), TR_CALC_F)); }
/* frequency f: its d coordinates */
static int32_t gen_draw(cx *c, const gen_t *g, uint64_t f, tr_num_rational *out) {
    const __int128 one = (__int128)1 << TR_CALC_F;
    if (g->kernel == TR_KAPPROX_GAUSSIAN) {
        for (uint64_t q = 0; q < g->npairs; q++) {
            uint64_t base = 4 * (f * g->npairs + q);
            unsigned __int128 N1 = gen_u128(g->key, base), N2 = gen_u128(g->key, base + 2), Llo, Lhi;
            neg_ln_uniform(N1, &Llo, &Lhi);
            tr_bigint *rlo = Z_isqrt(c, Z_shl(c, Z_u128(c, 2 * Llo), 120));
            tr_bigint *rhi = Z_add(c, Z_isqrt(c, Z_shl(c, Z_u128(c, 2 * Lhi), 120)), Z_i64(c, 1));
            tr_calc_fx sn, cs;
            sincos_uniform(N2, 0, &sn, &cs);
            for (int k = 0; k < 2 && 2 * q + (uint64_t)k < (uint64_t)g->d; k++) {
                tr_calc_fx v = k ? sn : cs;
                __int128 vl = v.c - (__int128)v.r, vh = v.c + (__int128)v.r;
                if (vl < -one) vl = -one;
                if (vh > one) vh = one;                                /* |sin|, |cos| <= 1 */
                tr_bigint *zl, *zh, *Vl = Z_i128(c, vl), *Vh = Z_i128(c, vh);
                if (vl >= 0) { zl = Z_mul(c, rlo, Vl); zh = Z_mul(c, rhi, Vh); }
                else if (vh <= 0) { zl = Z_mul(c, rhi, Vl); zh = Z_mul(c, rlo, Vh); }
                else { zl = Z_mul(c, rhi, Vl); zh = Z_mul(c, rhi, Vh); }  /* z in [zl, zh] / 2^240 */
                if (c->rc) return c->rc;
                tr_bigint *ol, *oh;                                    /* omega = z sqrt(2 gamma) */
                if (tr_bigint_sign(zl) >= 0) { ol = Z_mul(c, zl, g->slo); oh = Z_mul(c, zh, g->shi); }
                else if (tr_bigint_sign(zh) <= 0) { ol = Z_mul(c, zl, g->shi); oh = Z_mul(c, zh, g->slo); }
                else { ol = Z_mul(c, zl, g->shi); oh = Z_mul(c, zh, g->shi); }
                tr_bigint *den = Z_shl(c, Z_i64(c, 1), 360);
                int32_t rc = c->rc ? c->rc : round_omega(c, g, Q_frac(c, ol, den), Q_frac(c, oh, den), &out[2 * q + (uint64_t)k]);
                if (rc) return rc;
            }
        }
        return c->rc;
    }
    for (int32_t j = 0; j < g->d; j++) {
        if (g->kernel == TR_KAPPROX_LAPLACE) {
            uint64_t base = 2 * (f * (uint64_t)g->d + (uint64_t)j);
            unsigned __int128 N = gen_u128(g->key, base), edge = (unsigned __int128)1 << 96;
            if (N < edge || N >= (unsigned __int128)0 - edge) { out[j].numerator = 0; out[j].denominator = 1; continue; }
            tr_calc_fx sn, cs;
            sincos_uniform(N, 1, &sn, &cs);                            /* sin, cos of pi U */
            __int128 sl = sn.c - (__int128)sn.r, sh = sn.c + (__int128)sn.r;
            __int128 cl = cs.c - (__int128)cs.r, ch = cs.c + (__int128)cs.r;
            if (sl <= 0) return TR_MATH_OVERFLOW;                     /* never on a good cell: sin >= 2^-31 */
            tr_bigq *Sl = Q_fx(c, sl), *Sh = Q_fx(c, sh), *Cl = Q_fx(c, cl), *Ch = Q_fx(c, ch);
            tr_bigq *lo = Q_div(c, Cl, cl >= 0 ? Sh : Sl), *hi = Q_div(c, Ch, ch >= 0 ? Sl : Sh);
            int32_t rc = c->rc ? c->rc : round_omega(c, g, Q_mul(c, lo, g->gam), Q_mul(c, hi, g->gam), &out[j]);
            if (rc) return rc;
        } else {
            uint64_t base = 3 * (f * (uint64_t)g->d + (uint64_t)j);
            unsigned __int128 N = gen_u128(g->key, base), Llo, Lhi;
            int neg = (int)(sm64(g->key + base + 2) >> 63);
            neg_ln_uniform(N, &Llo, &Lhi);
            tr_bigq *lo = Q_mul(c, Q_fx(c, (__int128)Llo), g->gam), *hi = Q_mul(c, Q_fx(c, (__int128)Lhi), g->gam);
            if (neg) { tr_bigq *t = Q_neg(c, hi); hi = Q_neg(c, lo); lo = t; }
            int32_t rc = c->rc ? c->rc : round_omega(c, g, lo, hi, &out[j]);
            if (rc) return rc;
        }
    }
    return c->rc;
}
static int32_t kernel_ok(int32_t k) { return k == TR_KAPPROX_GAUSSIAN || k == TR_KAPPROX_LAPLACE || k == TR_KAPPROX_CAUCHY; }
static int q_positive(tr_num_rational q) { return q.denominator > 0 && q.numerator > 0; }

int32_t tr_kapprox_generate(int32_t kernel, tr_num_rational gamma, int32_t ndims, uint64_t count, uint64_t seed,
                            tr_num_rational *out, int32_t *fbits, const tr_math_budget *budget, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    tr_math_budget dflt;
    budget = budget_of(budget, &dflt);
    if (!kernel_ok(kernel) || !q_positive(gamma) || ndims < 1 || !count || (count && !out)) return TR_MATH_INVALID;
    if (ndims > TR_KAPPROX_MAX_DIMS || count > TR_KAPPROX_MAX_FREQS) return TR_MATH_UNSUPPORTED;
    tr_num_rational g;
    if (!tr_q_make(gamma.numerator, gamma.denominator, &g)) return TR_MATH_OVERFLOW;
    tr_math_need need;
    need_zero(&need);
    need.need[TR_BUDGET_ITERATIONS] = tr_budget_mul(count, (uint64_t)ndims);
    need.need[TR_BUDGET_BYTES] = tr_budget_add(tr_budget_mul(need.need[TR_BUDGET_ITERATIONS], 16), 4096);
    int32_t rc = tr_budget_check(budget, &need, refusal);
    if (rc) return rc;
    cx c;
    cx_init(&c, budget, refusal);
    gen_t gen;
    rc = gen_setup(&c, kernel, g, ndims, seed, &gen);
    for (uint64_t f = 0; f < count && !rc; f++) {
        cx w;
        cx_init(&w, budget, refusal);
        rc = gen_draw(&w, &gen, f, out + f * (uint64_t)ndims);
        cx_done(&w);
    }
    if (!rc && fbits) *fbits = gen.p;
    cx_done(&c);
    return rc;
}

/* ================================================================ TAYLOR items: the multi-indices |a| <= N */
typedef struct {
    uint8_t *exps;
    int32_t *parent, *axis, *cls;
    uint32_t *mult;
    uint64_t n;
} items_t;
static void items_free(items_t *it) {
    free(it->exps); free(it->parent); free(it->axis); free(it->cls); free(it->mult);
    memset(it, 0, sizeof *it);
}
static uint64_t items_count(int32_t d, int32_t N) { return tr_budget_monomials((uint32_t)d, (uint32_t)N); }
static uint64_t items_bytes(uint64_t n, int32_t d) { return tr_budget_mul(n, tr_budget_add((uint64_t)d, 16)); }
/* breadth first by degree: the children of item i are i + e_j for j >= its last axis, so every multi-index appears
 * once and its parent (the last unit removed) comes first; mult = |a|! / prod a_j! */
static int32_t items_make(int32_t d, int32_t N, items_t *it) {
    memset(it, 0, sizeof *it);
    uint64_t n = items_count(d, N);
    if (n == TR_BUDGET_UNKNOWN) return TR_MATH_BUDGET;
    it->exps = (uint8_t *)zalloc(n * (uint64_t)d);
    it->parent = (int32_t *)zalloc(n * 4); it->axis = (int32_t *)zalloc(n * 4); it->cls = (int32_t *)zalloc(n * 4);
    it->mult = (uint32_t *)zalloc(n * 4);
    if (!it->exps || !it->parent || !it->axis || !it->cls || !it->mult) { items_free(it); return TR_MATH_MEMORY; }
    static const uint64_t fact[TR_KAPPROX_MAX_DEGREE + 1] = {1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880, 3628800,
                                                             39916800, 479001600};
    it->parent[0] = -1; it->axis[0] = -1; it->cls[0] = 0; it->mult[0] = 1;
    uint64_t count = 1, start = 0, end = 1;
    for (int32_t k = 1; k <= N; k++) {
        for (uint64_t i = start; i < end; i++) {
            int32_t last = i == 0 ? 0 : it->axis[i];
            for (int32_t j = last; j < d; j++) {
                uint64_t t = count++;
                memcpy(it->exps + t * (uint64_t)d, it->exps + i * (uint64_t)d, (size_t)d);
                it->exps[t * (uint64_t)d + (uint64_t)j]++;
                it->parent[t] = (int32_t)i; it->axis[t] = j; it->cls[t] = k;
                uint64_t m = fact[k];
                for (int32_t q = 0; q < d; q++) m /= fact[it->exps[t * (uint64_t)d + (uint64_t)q]];
                it->mult[t] = (uint32_t)m;
            }
        }
        start = end; end = count;
    }
    it->n = count;
    return count == n ? TR_MATH_OK : TR_MATH_INVALID;
}
static int32_t items_copy(const items_t *src, int32_t d, items_t *dst) {
    memset(dst, 0, sizeof *dst);
    uint64_t n = src->n;
    dst->exps = (uint8_t *)zalloc(n * (uint64_t)d);
    dst->parent = (int32_t *)zalloc(n * 4); dst->axis = (int32_t *)zalloc(n * 4); dst->cls = (int32_t *)zalloc(n * 4);
    dst->mult = (uint32_t *)zalloc(n * 4);
    if (!dst->exps || !dst->parent || !dst->axis || !dst->cls || !dst->mult) { items_free(dst); return TR_MATH_MEMORY; }
    memcpy(dst->exps, src->exps, (size_t)(n * (uint64_t)d));
    memcpy(dst->parent, src->parent, (size_t)n * 4); memcpy(dst->axis, src->axis, (size_t)n * 4);
    memcpy(dst->cls, src->cls, (size_t)n * 4); memcpy(dst->mult, src->mult, (size_t)n * 4);
    dst->n = n;
    return TR_MATH_OK;
}

/* ================================================================ frequency sets */
static int freq_cmp(const tr_num_rational *a, const tr_num_rational *b, int32_t d) {
    for (int32_t j = 0; j < d; j++) { int c = tr_q_cmp(a[j], b[j]); if (c) return c; }
    return 0;
}
/* idx = 0..n-1 sorted by frequency (bottom-up merge sort, stable) */
static int32_t sort_freqs(const tr_num_rational *om, uint64_t n, int32_t d, uint64_t *idx) {
    uint64_t *tmp = (uint64_t *)zalloc(n * 8);
    if (!tmp) return TR_MATH_MEMORY;
    for (uint64_t i = 0; i < n; i++) idx[i] = i;
    for (uint64_t w = 1; w < n; w *= 2) {
        for (uint64_t lo = 0; lo < n; lo += 2 * w) {
            uint64_t mid = lo + w < n ? lo + w : n, hi = lo + 2 * w < n ? lo + 2 * w : n, i = lo, j = mid, o = lo;
            while (i < mid && j < hi) tmp[o++] = freq_cmp(om + idx[j] * d, om + idx[i] * d, d) < 0 ? idx[j++] : idx[i++];
            while (i < mid) tmp[o++] = idx[i++];
            while (j < hi) tmp[o++] = idx[j++];
        }
        memcpy(idx, tmp, (size_t)n * 8);
    }
    free(tmp);
    return TR_MATH_OK;
}

/* ================================================================ the spec */
static int32_t spec_check(tr_kapprox_spec *s, int32_t ncols) {
    if (!kernel_ok(s->kernel) || (s->method != TR_KAPPROX_TAYLOR && s->method != TR_KAPPROX_RFF) || s->reserved ||
        s->ndims < 1 || (s->weight != -1 && (s->weight < 0 || s->weight >= ncols)) ||
        (s->has_radius != 0 && s->has_radius != 1) || (s->has_budget != 0 && s->has_budget != 1))
        return TR_MATH_INVALID;
    if (s->ndims > TR_KAPPROX_MAX_DIMS) return TR_MATH_UNSUPPORTED;
    for (int32_t j = 0; j < TR_KAPPROX_MAX_DIMS; j++) {
        if (j >= s->ndims) { if (s->column[j]) return TR_MATH_INVALID; continue; }
        if (s->column[j] < 0 || s->column[j] >= ncols) return TR_MATH_INVALID;
        for (int32_t i = 0; i < j; i++) if (s->column[i] == s->column[j]) return TR_MATH_INVALID;
    }
    if (!q_positive(s->gamma)) return TR_MATH_INVALID;
    if (!tr_q_make(s->gamma.numerator, s->gamma.denominator, &s->gamma)) return TR_MATH_OVERFLOW;
    if (s->has_radius) {
        if (s->radius2.denominator <= 0 || s->radius2.numerator < 0) return TR_MATH_INVALID;
        if (!tr_q_make(s->radius2.numerator, s->radius2.denominator, &s->radius2)) return TR_MATH_OVERFLOW;
    } else { s->radius2.numerator = 0; s->radius2.denominator = 0; }
    if (s->has_budget) {
        if (s->budget.denominator <= 0 || s->budget.numerator < 0) return TR_MATH_INVALID;
        if (!tr_q_make(s->budget.numerator, s->budget.denominator, &s->budget)) return TR_MATH_OVERFLOW;
    } else { s->budget.numerator = 0; s->budget.denominator = 0; }
    if (s->method == TR_KAPPROX_TAYLOR) {
        if (s->kernel != TR_KAPPROX_GAUSSIAN) return TR_MATH_UNSUPPORTED;   /* no deterministic features offered */
        if (s->degree < 0) return TR_MATH_INVALID;
        if (s->degree > TR_KAPPROX_MAX_DEGREE) return TR_MATH_UNSUPPORTED;
        s->nfreq = 0; s->seed = 0; s->failure.numerator = 0; s->failure.denominator = 0;
    } else {
        if (s->degree) return TR_MATH_INVALID;
        if (!s->nfreq) return TR_MATH_INVALID;
        if (s->nfreq > TR_KAPPROX_MAX_FREQS) return TR_MATH_UNSUPPORTED;
        if (!q_positive(s->failure) || s->failure.numerator >= s->failure.denominator) return TR_MATH_INVALID;
        if (!tr_q_make(s->failure.numerator, s->failure.denominator, &s->failure)) return TR_MATH_OVERFLOW;
    }
    return TR_MATH_OK;
}

/* ================================================================ plans */
struct tr_kapprox_plan_s {
    uint32_t magic, reserved;
    tr_kapprox_spec spec;
    tr_kapprox_plan_view v;
    int32_t xscale, wscale, fbits, reserved2;
    uint64_t rows, selected, ngroups;
    uint64_t E[3];                  /* sum_j max |X_j|^2 over the column extremes (192-bit) */
    uint64_t T[3];                  /* the declared domain floor(radius2 10^(2 scale)), clamped to 2^192 - 1 */
    unsigned char *key;
    uint64_t keybytes;
    items_t it;                     /* TAYLOR */
    tr_fourier_plan *fourier;       /* RFF */
    tr_num_rational *omega;         /* RFF: m distinct frequencies (sorted), m x d */
    uint32_t *fmult;
    uint64_t m, D;
};
void tr_kapprox_plan_free(tr_kapprox_plan *p) {
    if (!p || p->magic != PLAN_MAGIC) return;
    p->magic = 0;
    items_free(&p->it);
    tr_fourier_plan_free(p->fourier);
    free(p->omega); free(p->fmult); free(p->key);
    free(p);
}
int32_t tr_kapprox_plan_get_view(const tr_kapprox_plan *p, tr_kapprox_plan_view *out) {
    if (!p || p->magic != PLAN_MAGIC || !out) return TR_MATH_INVALID;
    *out = p->v;
    return TR_MATH_OK;
}
/* bytes of a state */
static uint64_t state_bytes(const tr_kapprox_spec *s, uint64_t G, uint64_t nitems, uint64_t m, uint64_t keybytes) {
    uint64_t b = tr_budget_add(4096, tr_budget_mul(G, sizeof(tr_kapprox_group)));
    b = tr_budget_add(b, tr_budget_mul(tr_budget_mul(G, nitems), 32));
    if (s->method == TR_KAPPROX_TAYLOR) b = tr_budget_add(b, items_bytes(nitems, s->ndims));
    else b = tr_budget_add(b, tr_budget_mul(tr_budget_mul(m, (uint64_t)s->ndims), 16 + 32 + 4 + 1 + 4));
    return tr_budget_add(b, tr_budget_mul(keybytes, 2));
}
/* x into 3 limbs, or all ones when it does not fit */
static void z_to_192(const tr_bigint *x, uint64_t *out) {
    uint64_t l[8];
    int32_t sg;
    uint64_t cnt = 0;
    out[0] = out[1] = out[2] = 0;
    if (!x || tr_bigint_sign(x) <= 0) return;
    if (tr_bigint_limb_count(x) > 3 || tr_bigint_export(x, &sg, l, 8, &cnt)) { out[0] = out[1] = out[2] = UINT64_MAX; return; }
    for (uint64_t i = 0; i < cnt && i < 3; i++) out[i] = l[i];
}
static int32_t plan_common(const tr_scope *scope, tr_kapprox_plan *p, cx *c) {
    tr_scope_info info;
    int32_t rc = tr_scope_get_info(scope, &info);
    if (rc) return rc;
    p->rows = info.rows; p->selected = info.selected; p->ngroups = info.ngroups;
    const tr_kapprox_spec *s = &p->spec;
    uint64_t mw = 1;
    p->wscale = 0;
    if (s->weight >= 0) {
        tr_scope_column_info wi;
        rc = tr_scope_column_get(scope, s->weight, &wi);
        if (rc) return rc;
        p->wscale = wi.scale;
        mw = 0;
        if (wi.nvalid) { uint64_t a = tr_abs_u64(wi.min), b = tr_abs_u64(wi.max); mw = a > b ? a : b; }
    }
    for (int32_t j = 0; j < s->ndims; j++) {
        tr_scope_column_info ci;
        rc = tr_scope_column_get(scope, s->column[j], &ci);
        if (rc) return rc;
        if (j == 0) p->xscale = ci.scale;
        else if (ci.scale != p->xscale) return TR_MATH_DOMAIN_ERROR;   /* one decimal scale for |x|^2 */
        if (ci.nvalid) {
            uint64_t a = tr_abs_u64(ci.min), b = tr_abs_u64(ci.max), m = a > b ? a : b;
            tr_kapprox_add192(p->E, (unsigned __int128)m * m);
        }
    }
    unsigned __int128 W = (unsigned __int128)info.selected * mw;
    if (bitlen128(W) > 123) return TR_MATH_OVERFLOW;                  /* the sums' scale proof needs W < 2^124 */
    p->v.scale = (int32_t)(124 - bitlen128(W) < TR_CALC_F ? 124 - bitlen128(W) : TR_CALC_F);
    p->T[0] = p->T[1] = p->T[2] = UINT64_MAX;
    if (s->has_radius) {
        tr_bigint *t = Z_floordiv(c, Z_mul(c, Z_i64(c, s->radius2.numerator), Z_pow10(c, 2 * (uint64_t)p->xscale)),
                                  Z_i64(c, s->radius2.denominator));
        if (c->rc) return c->rc;
        z_to_192(t, p->T);
    }
    p->keybytes = tr_scope_key_size(scope, TR_SCOPE_KEY_ALL);
    p->key = (unsigned char *)malloc(p->keybytes ? (size_t)p->keybytes : 1);
    if (!p->key) return TR_MATH_MEMORY;
    return tr_scope_key_write(scope, TR_SCOPE_KEY_ALL, p->key, p->keybytes);
}
/* TAYLOR: b, Kq, tmax and the items */
static int32_t plan_taylor(tr_kapprox_plan *p, const tr_math_budget *budget, tr_math_refusal *refusal, cx *c) {
    const tr_kapprox_spec *s = &p->spec;
    uint64_t X[3];                                                     /* max(E, T): rows and points stay below */
    memcpy(X, p->E, sizeof X);
    if (s->has_radius && tr_kapprox_cmp192(p->T, X) > 0) memcpy(X, p->T, sizeof X);
    int32_t b = 0;
    if (X[0] | X[1] | X[2]) {
        uint64_t y[3] = {X[0], X[1], X[2]};                            /* b = ceil(bitlen(X - 1) / 2) */
        if (y[0]-- == 0) { if (y[1]-- == 0) y[2]--; }
        b = (bitlen192(y) + 1) / 2;
    }
    if (b > TR_CALC_F) return TR_MATH_UNSUPPORTED;
    tr_bigint *num = Z_mul(c, Z_i64(c, s->gamma.numerator), Z_limbs(c, INV_LN2_576, 10));
    tr_bigint *den = Z_shl(c, Z_mul(c, Z_i64(c, s->gamma.denominator), Z_pow10(c, 2 * (uint64_t)p->xscale)), 256);
    tr_bigint *kq = Z_floordiv(c, num, den);
    tr_bigint *Xz = Z_limbs(c, X, 3);
    tr_bigint *tm = Z_ceildiv(c, Z_mul(c, Z_mul(c, Z_i64(c, s->gamma.numerator), Xz), Z_i64(c, 1443)),
                              Z_mul(c, Z_mul(c, Z_i64(c, s->gamma.denominator), Z_i64(c, 1000)),
                                    Z_pow10(c, 2 * (uint64_t)p->xscale)));
    if (c->rc) return c->rc;
    int32_t sg;
    uint64_t cnt = 0, lim[8];
    if (tr_bigint_limb_count(kq) > TR_KAPPROX_KQ_LIMBS) return TR_MATH_UNSUPPORTED;
    memset(p->v.kq, 0, sizeof p->v.kq);
    if (tr_bigint_sign(kq) > 0) {
        if (tr_bigint_export(kq, &sg, lim, 8, &cnt)) return TR_MATH_INVALID;
        for (uint64_t i = 0; i < cnt; i++) p->v.kq[i] = lim[i];
    }
    if (tr_bigint_bit_length(tm) > 24) return TR_MATH_UNSUPPORTED;     /* exp(-gamma |x|^2) below 2^-(2^24) */
    int64_t t64 = 0;
    tr_bigint_to_i64(tm, &t64);
    if ((uint64_t)t64 + 1 >= TR_KAPPROX_NMAX) return TR_MATH_UNSUPPORTED;
    p->v.tmax = (uint64_t)t64 + 1;
    p->v.shift = b;
    uint64_t n = items_count(s->ndims, s->degree);
    tr_math_need need;
    need_zero(&need);
    need.need[TR_BUDGET_VARIABLES] = (uint64_t)s->ndims + (s->weight >= 0);
    need.need[TR_BUDGET_DEGREE] = (uint64_t)s->degree;
    need.need[TR_BUDGET_SUPPORT] = n;
    need.need[TR_BUDGET_STATES] = tr_budget_mul(p->ngroups, n);
    need.need[TR_BUDGET_TENSOR] = tr_budget_mul(p->selected, n);
    uint64_t bytes = tr_budget_add(sizeof(tr_kapprox_plan), items_bytes(n, s->ndims));
    bytes = tr_budget_add(bytes, tr_budget_mul(n, sizeof(tr_calc_fx)));            /* the per-row memo */
    bytes = tr_budget_add(bytes, tr_budget_mul(tr_budget_mul(p->ngroups, n), 32));  /* the sums */
    need.need[TR_BUDGET_BYTES] = tr_budget_add(bytes, state_bytes(s, p->ngroups, n, 0, p->keybytes));
    int32_t rc = tr_budget_check(budget, &need, refusal);
    if (rc) return rc;
    rc = items_make(s->ndims, s->degree, &p->it);
    if (rc) return rc;
    p->v.nitems = p->it.n;
    p->v.exps = p->it.exps; p->v.parent = p->it.parent; p->v.axis = p->it.axis;
    return TR_MATH_OK;
}
/* RFF: the draws, deduplicated with multiplicities, and SPEC-001's plan */
static int32_t plan_rff(const tr_scope *scope, tr_kapprox_plan *p, const tr_math_budget *budget,
                        tr_math_refusal *refusal, cx *c) {
    const tr_kapprox_spec *s = &p->spec;
    const int32_t d = s->ndims;
    const uint64_t D = s->nfreq, nd = D * (uint64_t)d;
    tr_math_need need;
    need_zero(&need);
    need.need[TR_BUDGET_VARIABLES] = (uint64_t)d + (s->weight >= 0);
    need.need[TR_BUDGET_SUPPORT] = D;
    need.need[TR_BUDGET_ITERATIONS] = nd;
    uint64_t bytes = tr_budget_add(sizeof(tr_kapprox_plan), tr_budget_mul(nd, 16 * 2));
    bytes = tr_budget_add(bytes, tr_budget_mul(D, 8 * 2 + 4));
    need.need[TR_BUDGET_BYTES] = tr_budget_add(bytes, state_bytes(s, p->ngroups, 2 * D, D, p->keybytes));
    int32_t rc = tr_budget_check(budget, &need, refusal);
    if (rc) return rc;
    gen_t g;
    rc = gen_setup(c, s->kernel, s->gamma, d, s->seed, &g);
    if (rc) return rc;
    p->fbits = g.p;
    tr_num_rational *raw = (tr_num_rational *)zalloc(nd * sizeof *raw);
    uint64_t *idx = (uint64_t *)zalloc(D * 8);
    p->omega = (tr_num_rational *)zalloc(nd * sizeof *raw);
    p->fmult = (uint32_t *)zalloc(D * 4);
    if (!raw || !idx || !p->omega || !p->fmult) rc = TR_MATH_MEMORY;
    for (uint64_t f = 0; f < D && !rc; f++) {
        cx w;
        cx_init(&w, budget, refusal);
        rc = gen_draw(&w, &g, f, raw + f * (uint64_t)d);
        cx_done(&w);
    }
    if (!rc) rc = sort_freqs(raw, D, d, idx);
    uint64_t m = 0;
    for (uint64_t i = 0; i < D && !rc; i++) {
        const tr_num_rational *w = raw + idx[i] * (uint64_t)d;
        if (m && !freq_cmp(p->omega + (m - 1) * (uint64_t)d, w, d)) { p->fmult[m - 1]++; continue; }
        memcpy(p->omega + m * (uint64_t)d, w, (size_t)d * sizeof *w);
        p->fmult[m++] = 1;
    }
    free(raw); free(idx);
    if (rc) return rc;
    p->m = m; p->D = D;
    tr_fourier_spec fs;
    memset(&fs, 0, sizeof fs);
    fs.ndims = d; fs.weight = s->weight; fs.unit = TR_FOURIER_RADIAN;
    for (int32_t j = 0; j < d; j++) fs.column[j] = s->column[j];
    p->fourier = tr_fourier_plan_create(scope, &fs, m, p->omega, budget, refusal, &rc);
    if (!p->fourier) return rc ? rc : TR_MATH_MEMORY;
    rc = tr_fourier_plan_get_view(p->fourier, &p->v.fourier);
    if (rc) return rc;
    p->v.scale = p->v.fourier.scale;
    p->v.nitems = 2 * m;
    return TR_MATH_OK;
}
tr_kapprox_plan *tr_kapprox_plan_create(const tr_scope *scope, const tr_kapprox_spec *spec, const tr_math_budget *budget,
                                        tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    tr_math_budget dflt;
    budget = budget_of(budget, &dflt);
    tr_scope_info info;
    if (!scope || !spec || tr_scope_get_info(scope, &info)) { seterr(error, TR_MATH_INVALID); return NULL; }
    tr_kapprox_plan *p = (tr_kapprox_plan *)zalloc(sizeof *p);
    if (!p) { seterr(error, TR_MATH_MEMORY); return NULL; }
    p->magic = PLAN_MAGIC;
    p->spec = *spec;
    int32_t rc = spec_check(&p->spec, info.ncols);
    cx c;
    cx_init(&c, budget, refusal);
    if (!rc) rc = plan_common(scope, p, &c);
    if (!rc) rc = p->spec.method == TR_KAPPROX_TAYLOR ? plan_taylor(p, budget, refusal, &c)
                                                      : plan_rff(scope, p, budget, refusal, &c);
    cx_done(&c);
    if (rc) { tr_kapprox_plan_free(p); seterr(error, rc); return NULL; }
    tr_kapprox_plan_view *v = &p->v;
    v->method = p->spec.method; v->ndims = p->spec.ndims; v->weight = p->spec.weight; v->degree = p->spec.degree;
    v->xscale = p->xscale; v->ngroups = p->ngroups;
    for (int32_t j = 0; j < p->spec.ndims; j++) v->column[j] = p->spec.column[j];
    seterr(error, TR_MATH_OK);
    return p;
}

/* ================================================================ the sums on the CPU (the reference of the kernels) */
static int add_i128(tr_math_i128 *acc, __int128 x) {
    __int128 s;
    if (__builtin_add_overflow(tr_i128_get(*acc), x, &s)) return 0;
    *acc = tr_i128_set(s);
    return 1;
}
int32_t tr_kapprox_sums(const tr_kapprox_plan_view *pv, const tr_scope_view *v, tr_kapprox_group *groups, uint64_t *maxsq,
                        tr_math_i128 *sums, uint64_t *frows, uint64_t *fvalid, tr_math_i128 *fwsum) {
    if (!tr_kapprox_plan_view_valid(pv, v) || !groups || !maxsq || !sums) return TR_MATH_INVALID;
    const int taylor = pv->method == TR_KAPPROX_TAYLOR;
    if (!taylor && (!frows || !fvalid || !fwsum)) return TR_MATH_INVALID;
    if (taylor && (!pv->parent || !pv->axis)) return TR_MATH_INVALID;
    const uint64_t G = pv->ngroups, n = pv->nitems;
    const int32_t d = pv->ndims;
    memset(groups, 0, (size_t)G * sizeof *groups);
    maxsq[0] = maxsq[1] = maxsq[2] = 0;
    tr_calc_fx *memo = NULL;
    if (taylor) {
        memset(sums, 0, (size_t)(G * n * 2) * sizeof *sums);
        memo = (tr_calc_fx *)zalloc(n * sizeof *memo);
        if (!memo) return TR_MATH_MEMORY;
    }
    int32_t rc = TR_MATH_OK;
    int64_t X[TR_KAPPROX_MAX_DIMS];
    for (uint64_t r = 0; r < v->rows && !rc; r++) {
        if (!tr_scope_is_selected(v, r)) continue;
        uint32_t g = tr_scope_group(v, r);
        if ((uint64_t)g >= G) { rc = TR_MATH_INVALID; break; }
        tr_kapprox_group *gr = &groups[g];
        gr->rows++;
        if (!tr_kapprox_row_ok(pv, v, r)) continue;
        int64_t w = tr_kapprox_row_weight(pv, v, r);
        gr->valid++;
        if (!add_i128(&gr->W, w) || !add_i128(&gr->A, (__int128)tr_abs_u64(w))) { rc = TR_MATH_OVERFLOW; break; }
        tr_kapprox_add192(gr->Q, (unsigned __int128)tr_abs_u64(w) * tr_abs_u64(w));
        for (int32_t j = 0; j < d; j++) X[j] = tr_scope_value(v, pv->column[j], r);
        uint64_t sq[3];
        tr_kapprox_sumsq(X, d, sq);
        if (tr_kapprox_cmp192(sq, maxsq) > 0) memcpy(maxsq, sq, sizeof sq);
        if (!taylor) continue;
        int64_t e;
        tr_calc_fx m;
        if (!tr_kapprox_prefactor(pv, X, &e, &m)) { rc = TR_MATH_OVERFLOW; break; }
        memo[0] = m;
        for (uint64_t t = 1; t < n; t++) memo[t] = tr_calc_fx_mul(memo[pv->parent[t]], tr_kapprox_coord(pv, X[pv->axis[t]]));
        tr_math_i128 *acc = sums + (uint64_t)g * n * 2;
        for (uint64_t t = 0; t < n; t++) {
            __int128 L, H;
            if (!tr_kapprox_term(w, e, memo[t], pv->scale, &L, &H) || !add_i128(&acc[2 * t], L) ||
                !add_i128(&acc[2 * t + 1], H)) { rc = TR_MATH_OVERFLOW; break; }
        }
    }
    free(memo);
    if (!rc && !taylor) rc = tr_fourier_sums(&pv->fourier, v, sums, frows, fvalid, fwsum);
    return rc;
}

/* ================================================================ states */
struct tr_kapprox_state_s {
    uint32_t magic, reserved;
    tr_kapprox_spec spec;
    tr_kapprox_plan_view v;         /* pointers into this state's arrays */
    int32_t xscale, wscale, fbits, ncoef;
    uint64_t ngroups, nitems, m, D, rows, selected, id, bytes;
    uint64_t maxsq[3], E[3], T[3], tdom[3];   /* tdom: the points' domain (the declared T, else maxsq) */
    tr_kapprox_group *groups;
    tr_math_i128 *sums;             /* G x nitems x 2 */
    items_t it;                     /* TAYLOR */
    tr_num_rational *omega;         /* RFF: m x d */
    uint32_t *fmult;
    uint64_t *k;                    /* RFF: SPEC-001's reduction constants, m d x 4 limbs */
    int32_t *ksign;
    uint8_t *kexact;
    tr_bigq *gamma, *rho2, *U, *per_pair, *h, *beta, *eta, *eta_num, *pbad, *failure, *budget;
    tr_bigq **coef;                 /* per class */
    tr_cert_record *source;         /* NULL: unscoped */
    uint8_t *meta;
    uint64_t meta_bytes;
};
static int state_ok(const tr_kapprox_state *s) { return s && s->magic == STATE_MAGIC; }
static void state_release(tr_kapprox_state *s) {
    if (!s) return;
    s->magic = 0;
    free(s->groups); free(s->sums); items_free(&s->it);
    free(s->omega); free(s->fmult); free(s->k); free(s->ksign); free(s->kexact);
    tr_bigq *q[] = {s->gamma, s->rho2, s->U, s->per_pair, s->h, s->beta, s->eta, s->eta_num, s->pbad, s->failure, s->budget};
    for (size_t i = 0; i < sizeof q / sizeof q[0]; i++) tr_bigq_free(q[i]);
    if (s->coef) for (int32_t i = 0; i < s->ncoef; i++) tr_bigq_free(s->coef[i]);
    free(s->coef);
    tr_cert_record_free(s->source);
    free(s->meta);
    free(s);
}
void tr_kapprox_free(tr_kapprox_state *s) { if (state_ok(s)) state_release(s); }

/* the exact constants of a state: the domain, the per-pair bound and its parts, the class coefficients; the budget */
static int32_t state_constants(tr_kapprox_state *s, const tr_math_budget *budget, tr_math_refusal *refusal) {
    cx c;
    cx_init(&c, budget, refusal);
    const tr_kapprox_spec *sp = &s->spec;
    tr_bigq *gam = Q_q64(&c, sp->gamma);
    tr_bigq *ten2 = Q_z(&c, Z_pow10(&c, 2 * (uint64_t)s->xscale));
    tr_bigq *rho2 = sp->has_radius ? Q_q64(&c, sp->radius2) : Q_div(&c, Q_z(&c, Z_limbs(&c, s->maxsq, 3)), ten2);
    memcpy(s->tdom, sp->has_radius ? s->T : s->maxsq, sizeof s->tdom);
    tr_bigq *per = NULL, *U = NULL, *h = NULL, *beta = NULL, *eta = NULL, *etan = NULL, *pbad = NULL, *fail = NULL;
    int32_t ncoef = sp->method == TR_KAPPROX_TAYLOR ? sp->degree + 1 : 1;
    tr_bigq *coef[TR_KAPPROX_MAX_DEGREE + 1] = {NULL};
    if (sp->method == TR_KAPPROX_TAYLOR) {
        const int32_t N = sp->degree;
        U = Q_mul(&c, Q_mul(&c, Q_int(&c, 2), gam), rho2);
        if (!c.rc && Q_is_zero(U)) per = Q_int(&c, 0);
        else if (!c.rc && Q_cmp(U, Q_int(&c, N + 2)) >= 0) per = Q_int(&c, 1);
        else if (!c.rc) {
            /* e^-U T_N(U) <= U^(N+1) / (N+1)! (N+2) / (N+2-U) / P_(N+32)(U) */
            tr_bigq *term = Q_int(&c, 1), *P = Q_int(&c, 1), *T = NULL;
            for (int32_t k = 1; k <= N + 32 && !c.rc; k++) {
                term = Q_div(&c, Q_mul(&c, term, U), Q_int(&c, k));
                if (k == N + 1) T = term;
                P = Q_add(&c, P, term);
            }
            T = Q_div(&c, Q_mul(&c, T, Q_int(&c, N + 2)), Q_sub(&c, Q_int(&c, N + 2), U));
            per = Q_div(&c, T, P);
            if (!c.rc && Q_cmp(per, Q_int(&c, 1)) > 0) per = Q_int(&c, 1);
        }
        /* coef_k = V^k / k!, V = 2 gamma 4^b / 10^(2 scale) */
        tr_bigq *V = Q_div(&c, Q_mul(&c, Q_mul(&c, Q_int(&c, 2), gam), Q_pow2(&c, 2 * (int64_t)s->v.shift)), ten2);
        tr_bigq *ck = Q_int(&c, 1);
        for (int32_t k = 0; k <= N && !c.rc; k++) {
            if (k) ck = Q_div(&c, Q_mul(&c, ck, V), Q_int(&c, k));
            coef[k] = ck;
        }
    } else {
        gen_t g;
        int32_t rc = gen_setup(&c, sp->kernel, sp->gamma, sp->ndims, sp->seed, &g);
        if (!rc && g.p != s->fbits) rc = TR_MATH_INVALID;
        if (rc && !c.rc) c.rc = rc;
        eta = g.eta; etan = g.eta_num; pbad = g.pbad;
        fail = Q_q64(&c, sp->failure);
        /* ln(2 / delta) <= hi(ln 2 den) - lo(ln num) */
        unsigned __int128 a, b, x, y;
        ln_int((unsigned __int128)2 * (uint64_t)sp->failure.denominator, &a, &b);
        ln_int((unsigned __int128)(uint64_t)sp->failure.numerator, &x, &y);
        tr_bigq *L = Q_sub(&c, Q_fx(&c, (__int128)b), Q_fx(&c, (__int128)x));
        h = Q_sqrt_up(&c, Q_div(&c, L, Q_int(&c, (int64_t)(2 * s->D))));
        int64_t sd = 1;
        while (sd * sd < sp->ndims) sd++;                              /* ceil(sqrt(d)) */
        beta = Q_add(&c, Q_mul(&c, Q_int(&c, 2), pbad),
                     Q_mul(&c, Q_mul(&c, Q_mul(&c, Q_int(&c, 2), Q_sqrt_up(&c, rho2)), Q_int(&c, sd)), eta));
        per = Q_add(&c, Q_mul(&c, Q_int(&c, 2), h), beta);
        coef[0] = Q_div(&c, Q_int(&c, 1), Q_int(&c, (int64_t)s->D));
    }
    tr_bigq *bud = sp->has_budget ? Q_q64(&c, sp->budget) : NULL;
    int32_t rc = c.rc;
    if (!rc && bud && Q_cmp(per, bud) > 0) rc = TR_MATH_NOT_CONVERGED;   /* the certified bound exceeds the budget */
    if (!rc) {
        s->coef = (tr_bigq **)zalloc((uint64_t)ncoef * sizeof(tr_bigq *));
        if (!s->coef) rc = TR_MATH_MEMORY;
    }
    if (!rc) {
        s->ncoef = ncoef;
        for (int32_t k = 0; k < ncoef; k++) s->coef[k] = Q_out(&c, coef[k]);
        s->gamma = Q_out(&c, gam); s->rho2 = Q_out(&c, rho2); s->per_pair = Q_out(&c, per);
        if (U) s->U = Q_out(&c, U);
        if (h) s->h = Q_out(&c, h);
        if (beta) s->beta = Q_out(&c, beta);
        if (eta) s->eta = Q_out(&c, eta);
        if (etan) s->eta_num = Q_out(&c, etan);
        if (pbad) s->pbad = Q_out(&c, pbad);
        if (fail) s->failure = Q_out(&c, fail);
        if (bud) s->budget = Q_out(&c, bud);
        rc = c.rc;
    }
    cx_done(&c);
    return rc;
}
static uint64_t mix_words(uint64_t h, const void *p, uint64_t bytes) {
    const uint8_t *b = (const uint8_t *)p;
    for (uint64_t i = 0; i < bytes; i += 8) {
        uint64_t w = 0;
        for (uint64_t k = 0; k < 8 && i + k < bytes; k++) w |= (uint64_t)b[i + k] << (8 * k);
        h = sm64(h ^ w);
    }
    return h;
}
/* the view's pointers into the state, the id and the bytes */
static void state_view(tr_kapprox_state *s) {
    tr_kapprox_plan_view *v = &s->v;
    v->ngroups = s->ngroups; v->nitems = s->nitems;
    if (s->spec.method == TR_KAPPROX_TAYLOR) { v->exps = s->it.exps; v->parent = s->it.parent; v->axis = s->it.axis; }
    else {
        v->fourier.k = s->k; v->fourier.ksign = s->ksign; v->fourier.kexact = s->kexact;
        v->fourier.nfreq = s->m; v->fourier.ngroups = s->ngroups;
    }
    uint64_t h = UINT64_C(0x4b45524e454c3032);                         /* "KERNEL02" */
    h = mix_words(h, &s->spec, sizeof s->spec);
    h = mix_words(h, s->maxsq, sizeof s->maxsq);
    h = mix_words(h, s->groups, s->ngroups * sizeof *s->groups);
    h = mix_words(h, s->sums, s->ngroups * s->nitems * 32);
    if (s->omega) h = mix_words(h, s->omega, s->m * (uint64_t)s->spec.ndims * sizeof *s->omega);
    if (s->fmult) h = mix_words(h, s->fmult, s->m * 4);
    s->id = h;
    s->bytes = state_bytes(&s->spec, s->ngroups, s->nitems, s->m, 0);
}
/* the checks every adoption and every load apply to group statistics and sums */
static int32_t check_groups(const tr_kapprox_group *gr, uint64_t G, const tr_math_i128 *sums, uint64_t nitems,
                            uint64_t selected) {
    uint64_t total = 0;
    for (uint64_t g = 0; g < G; g++) {
        const tr_kapprox_group *x = &gr[g];
        __int128 W = tr_i128_get(x->W), A = tr_i128_get(x->A);
        if (x->valid > x->rows || A < 0 || (W < 0 ? -W : W) > A || x->reserved) return TR_MATH_INVALID;
        unsigned __int128 Q2 = ((unsigned __int128)x->Q[1] << 64) | x->Q[0];
        if (!x->Q[2] && Q2 < (unsigned __int128)A) return TR_MATH_INVALID;   /* |w| <= w^2 for integers */
        unsigned __int128 lo, hi = tr_calc_mulhi((unsigned __int128)A, (unsigned __int128)A, &lo);   /* Q <= A^2 */
        uint64_t asq[4] = {(uint64_t)lo, (uint64_t)(lo >> 64), (uint64_t)hi, (uint64_t)(hi >> 64)};
        if (!asq[3] && tr_kapprox_cmp192(x->Q, asq) > 0) return TR_MATH_INVALID;
        if (!x->valid && (W || A || x->Q[0] || x->Q[1] || x->Q[2])) return TR_MATH_INVALID;
        for (uint64_t i = 0; i < nitems; i++) {
            __int128 L = tr_i128_get(sums[(g * nitems + i) * 2]), H = tr_i128_get(sums[(g * nitems + i) * 2 + 1]);
            if (L > H || (!x->valid && (L || H))) return TR_MATH_INVALID;
        }
        total += x->rows;
    }
    return total == selected ? TR_MATH_OK : TR_MATH_INVALID;
}
static tr_kapprox_state *state_new(void) {
    tr_kapprox_state *s = (tr_kapprox_state *)zalloc(sizeof *s);
    if (s) s->magic = STATE_MAGIC;
    return s;
}
static int32_t state_arrays(tr_kapprox_state *s) {
    const uint64_t G = s->ngroups, n = s->nitems, md = s->m * (uint64_t)s->spec.ndims;
    s->groups = (tr_kapprox_group *)zalloc(G * sizeof *s->groups);
    s->sums = (tr_math_i128 *)zalloc(G * n * 32);
    if (!s->groups || !s->sums) return TR_MATH_MEMORY;
    if (s->spec.method == TR_KAPPROX_RFF) {
        s->omega = (tr_num_rational *)zalloc(md * sizeof *s->omega);
        s->fmult = (uint32_t *)zalloc(s->m * 4);
        s->k = (uint64_t *)zalloc(md * 32); s->ksign = (int32_t *)zalloc(md * 4); s->kexact = (uint8_t *)zalloc(md);
        if (!s->omega || !s->fmult || !s->k || !s->ksign || !s->kexact) return TR_MATH_MEMORY;
    }
    return TR_MATH_OK;
}

tr_kapprox_state *tr_kapprox_from_sums(const tr_kapprox_plan *p, const tr_scope *scope, const tr_kapprox_group *groups,
                                       const uint64_t *maxsq, const tr_math_i128 *sums, const uint64_t *frows,
                                       const uint64_t *fvalid, const tr_math_i128 *fwsum, const tr_math_budget *budget,
                                       tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    tr_math_budget dflt;
    budget = budget_of(budget, &dflt);
    if (!p || p->magic != PLAN_MAGIC || !scope || !groups || !maxsq || !sums) { seterr(error, TR_MATH_INVALID); return NULL; }
    const int taylor = p->spec.method == TR_KAPPROX_TAYLOR;
    if (!taylor && (!frows || !fvalid || !fwsum)) { seterr(error, TR_MATH_INVALID); return NULL; }
    if (!tr_scope_key_equal(scope, TR_SCOPE_KEY_ALL, p->key, p->keybytes)) { seterr(error, TR_MATH_STALE); return NULL; }
    const uint64_t G = p->ngroups, n = p->v.nitems;
    int32_t rc = check_groups(groups, G, sums, n, p->selected);
    if (!rc && tr_kapprox_cmp192(maxsq, p->E) > 0) rc = TR_MATH_INVALID;     /* within the column extremes */
    for (uint64_t g = 0; g < G && !rc && !taylor; g++)               /* SPEC-001's counters agree with ours */
        if (frows[g] != groups[g].rows || fvalid[g] != groups[g].valid ||
            tr_i128_get(fwsum[g]) != tr_i128_get(groups[g].W)) rc = TR_MATH_INVALID;
    if (!rc && !taylor) {                                              /* SPEC-001's adoption checks of its sums */
        tr_fourier_state *fs = tr_fourier_from_sums(p->fourier, scope, sums, frows, fvalid, fwsum, budget, refusal, &rc);
        if (fs) { tr_fourier_free(fs); rc = TR_MATH_OK; }
        else if (!rc) rc = TR_MATH_MEMORY;
    }
    if (!rc && p->spec.has_radius && tr_kapprox_cmp192(maxsq, p->T) > 0) rc = TR_MATH_DOMAIN_ERROR;   /* outside */
    if (rc) { seterr(error, rc); return NULL; }
    tr_math_need need;
    need_zero(&need);
    need.need[TR_BUDGET_BYTES] = state_bytes(&p->spec, G, n, p->m, p->keybytes);
    rc = tr_budget_check(budget, &need, refusal);
    if (rc) { seterr(error, rc); return NULL; }
    tr_kapprox_state *s = state_new();
    if (!s) { seterr(error, TR_MATH_MEMORY); return NULL; }
    s->spec = p->spec; s->v = p->v;
    s->xscale = p->xscale; s->wscale = p->wscale; s->fbits = p->fbits;
    s->ngroups = G; s->nitems = n; s->m = p->m; s->D = p->D; s->rows = p->rows; s->selected = p->selected;
    memcpy(s->maxsq, maxsq, sizeof s->maxsq); memcpy(s->E, p->E, sizeof s->E); memcpy(s->T, p->T, sizeof s->T);
    rc = state_arrays(s);
    if (!rc) {
        memcpy(s->groups, groups, (size_t)G * sizeof *groups);
        memcpy(s->sums, sums, (size_t)(G * n) * 32);
        if (taylor) rc = items_copy(&p->it, p->spec.ndims, &s->it);
        else {
            const uint64_t md = p->m * (uint64_t)p->spec.ndims;
            memcpy(s->omega, p->omega, (size_t)md * sizeof *s->omega);
            memcpy(s->fmult, p->fmult, (size_t)p->m * 4);
            memcpy(s->k, p->v.fourier.k, (size_t)md * 32);
            memcpy(s->ksign, p->v.fourier.ksign, (size_t)md * 4);
            memcpy(s->kexact, p->v.fourier.kexact, (size_t)md);
        }
    }
    if (!rc) rc = state_constants(s, budget, refusal);
    if (!rc) {
        tr_math_budget unlimited;
        for (int i = 0; i < TR_BUDGET_RESOURCES; i++) unlimited.limit[i] = UINT64_MAX;
        s->source = tr_cert_record_scope(scope, NULL, -1, TR_CERT_RECORD_GROUPS, NULL, 0, &unlimited, NULL, &rc);
        if (s->source) rc = TR_MATH_OK;
        else if (!rc) rc = TR_MATH_MEMORY;
    }
    if (rc) { state_release(s); seterr(error, rc); return NULL; }
    state_view(s);
    seterr(error, TR_MATH_OK);
    return s;
}

tr_kapprox_state *tr_kapprox_build(const tr_scope *scope, const tr_kapprox_spec *spec, const tr_math_budget *budget,
                                   tr_math_refusal *refusal, int32_t *error) {
    tr_kapprox_plan *p = tr_kapprox_plan_create(scope, spec, budget, refusal, error);
    if (!p) return NULL;
    const uint64_t G = p->ngroups, n = p->v.nitems;
    const int taylor = p->spec.method == TR_KAPPROX_TAYLOR;
    tr_kapprox_group *groups = (tr_kapprox_group *)zalloc(G * sizeof *groups);
    tr_math_i128 *sums = (tr_math_i128 *)zalloc(G * n * 32);
    uint64_t *frows = taylor ? NULL : (uint64_t *)zalloc(G * 8), *fvalid = taylor ? NULL : (uint64_t *)zalloc(G * 8);
    tr_math_i128 *fwsum = taylor ? NULL : (tr_math_i128 *)zalloc(G * 16);
    uint64_t maxsq[3];
    tr_scope_view v;
    int32_t rc = groups && sums && (taylor || (frows && fvalid && fwsum)) ? tr_scope_get_view(scope, &v) : TR_MATH_MEMORY;
    if (!rc) rc = tr_kapprox_sums(&p->v, &v, groups, maxsq, sums, frows, fvalid, fwsum);
    tr_kapprox_state *s = NULL;
    if (!rc) s = tr_kapprox_from_sums(p, scope, groups, maxsq, sums, frows, fvalid, fwsum, budget, refusal, error);
    else seterr(error, rc);
    free(groups); free(sums); free(frows); free(fvalid); free(fwsum);
    tr_kapprox_plan_free(p);
    return s;
}

int32_t tr_kapprox_need(const tr_scope *scope, const tr_kapprox_spec *spec, int32_t target, tr_math_need *need) {
    if (!need || (target != TR_KAPPROX_NEED_CPU && target != TR_KAPPROX_NEED_DEVICE)) return TR_MATH_INVALID;
    tr_math_budget unlimited;
    for (int i = 0; i < TR_BUDGET_RESOURCES; i++) unlimited.limit[i] = UINT64_MAX;
    int32_t rc;
    tr_kapprox_plan *p = tr_kapprox_plan_create(scope, spec, &unlimited, NULL, &rc);
    if (!p) return rc;
    need_zero(need);
    const tr_kapprox_spec *s = &p->spec;
    const uint64_t G = p->ngroups, n = p->v.nitems;
    need->need[TR_BUDGET_VARIABLES] = (uint64_t)s->ndims + (s->weight >= 0);
    if (target == TR_KAPPROX_NEED_DEVICE) {
        tr_scope_view v;
        rc = tr_scope_get_view(scope, &v);
        uint64_t slices = tr_kapprox_slices(v.rows, 1);
        need->need[TR_BUDGET_STATES] = tr_budget_mul(G, n);
        need->need[TR_BUDGET_BYTES] = rc ? TR_BUDGET_UNKNOWN : tr_kapprox_device_bytes(&p->v, &v, slices);
    } else {
        need->need[TR_BUDGET_DEGREE] = (uint64_t)s->degree;
        need->need[TR_BUDGET_SUPPORT] = s->method == TR_KAPPROX_TAYLOR ? n : s->nfreq;
        need->need[TR_BUDGET_STATES] = tr_budget_mul(G, n);
        need->need[TR_BUDGET_TENSOR] = tr_budget_mul(p->selected, s->method == TR_KAPPROX_TAYLOR ? n : p->m);
        if (s->method == TR_KAPPROX_RFF) need->need[TR_BUDGET_ITERATIONS] = tr_budget_mul(s->nfreq, (uint64_t)s->ndims);
        uint64_t bytes = tr_budget_add(sizeof(tr_kapprox_plan), state_bytes(s, G, n, p->m, p->keybytes));
        if (s->method == TR_KAPPROX_TAYLOR)
            bytes = tr_budget_add(tr_budget_add(bytes, items_bytes(n, s->ndims)),
                                  tr_budget_add(tr_budget_mul(n, sizeof(tr_calc_fx)), tr_budget_mul(tr_budget_mul(G, n), 32)));
        else {
            bytes = tr_budget_add(bytes, tr_budget_add(tr_budget_mul(tr_budget_mul(s->nfreq, (uint64_t)s->ndims), 32),
                                                       tr_budget_mul(s->nfreq, 20)));
            /* SPEC-001's plan checks its own bytes: its exact need, read from its refusal of a zero byte budget */
            tr_math_budget probe;
            for (int i = 0; i < TR_BUDGET_RESOURCES; i++) probe.limit[i] = UINT64_MAX;
            probe.limit[TR_BUDGET_BYTES] = 0;
            tr_math_refusal pr;
            int32_t e2 = 0;
            tr_fourier_spec fs;
            memset(&fs, 0, sizeof fs);
            fs.ndims = s->ndims; fs.weight = s->weight; fs.unit = TR_FOURIER_RADIAN;
            for (int32_t j = 0; j < s->ndims; j++) fs.column[j] = s->column[j];
            tr_fourier_plan *fp = tr_fourier_plan_create(scope, &fs, p->m, p->omega, &probe, &pr, &e2);
            if (fp) tr_fourier_plan_free(fp);
            else if (e2 == TR_MATH_BUDGET && pr.resource == TR_BUDGET_BYTES && pr.need > bytes) bytes = pr.need;
        }
        need->need[TR_BUDGET_BYTES] = bytes;
    }
    tr_kapprox_plan_free(p);
    return rc;
}

/* ================================================================ reading */
int32_t tr_kapprox_get_info(const tr_kapprox_state *s, tr_kapprox_info *out) {
    if (!state_ok(s) || !out) return TR_MATH_INVALID;
    memset(out, 0, sizeof *out);
    const int taylor = s->spec.method == TR_KAPPROX_TAYLOR;
    out->kernel = s->spec.kernel; out->method = s->spec.method; out->ndims = s->spec.ndims; out->weight = s->spec.weight;
    out->degree = s->spec.degree; out->scale = s->v.scale; out->wscale = s->wscale; out->xscale = s->xscale;
    out->shift = s->v.shift; out->fbits = s->fbits;
    out->label = taylor ? TR_APPROX_LABEL_CERTIFIED : TR_APPROX_LABEL_PROBABILISTIC;
    out->provenance = taylor ? TR_APPROX_DETERMINISTIC : TR_APPROX_RANDOMIZED;
    out->norm = TR_APPROX_NORM_ABS; out->has_radius = s->spec.has_radius; out->has_budget = s->spec.has_budget;
    out->scoped = s->source != NULL;
    out->assumptions = TR_APPROX_ASSUME_DOMAIN | TR_APPROX_ASSUME_NO_OVERFLOW | (taylor ? 0u : (uint32_t)TR_APPROX_ASSUME_INDEPENDENT);
    out->generator = taylor ? 0 : TR_KAPPROX_GENERATOR;
    out->ngroups = s->ngroups; out->nitems = s->nitems; out->nfreq = s->m; out->count = s->D; out->seed = s->spec.seed;
    out->selected = s->selected; out->rows = s->rows; out->bytes = s->bytes; out->id = s->id;
    memcpy(out->maxsq, s->maxsq, sizeof out->maxsq);
    out->tmax = s->v.tmax;
    tr_math_interval iv;
    out->per_pair_hi = tr_bigq_enclose(s->per_pair, &iv) ? INFINITY : iv.hi;
    out->hoeffding_hi = s->h && !tr_bigq_enclose(s->h, &iv) ? iv.hi : 0.0;
    out->bias_hi = s->beta && !tr_bigq_enclose(s->beta, &iv) ? iv.hi : 0.0;
    out->radius2_hi = tr_bigq_enclose(s->rho2, &iv) ? INFINITY : iv.hi;
    return TR_MATH_OK;
}
int32_t tr_kapprox_get_spec(const tr_kapprox_state *s, tr_kapprox_spec *out) {
    if (!state_ok(s) || !out) return TR_MATH_INVALID;
    *out = s->spec;
    return TR_MATH_OK;
}
int32_t tr_kapprox_get_group(const tr_kapprox_state *s, uint64_t g, tr_kapprox_group *out) {
    if (!state_ok(s) || g >= s->ngroups || !out) return TR_MATH_INVALID;
    *out = s->groups[g];
    return TR_MATH_OK;
}
int32_t tr_kapprox_get_sums(const tr_kapprox_state *s, uint64_t g, tr_math_i128 *out, uint64_t capacity) {
    if (!state_ok(s) || g >= s->ngroups || !out || capacity < 2 * s->nitems) return TR_MATH_INVALID;
    memcpy(out, s->sums + g * s->nitems * 2, (size_t)s->nitems * 32);
    return TR_MATH_OK;
}
int32_t tr_kapprox_get_frequency(const tr_kapprox_state *s, uint64_t f, tr_num_rational *out, uint32_t *mult) {
    if (!state_ok(s) || s->spec.method != TR_KAPPROX_RFF || f >= s->m || !out) return TR_MATH_INVALID;
    memcpy(out, s->omega + f * (uint64_t)s->spec.ndims, (size_t)s->spec.ndims * sizeof *out);
    if (mult) *mult = s->fmult[f];
    return TR_MATH_OK;
}
const tr_bigq *tr_kapprox_get_q(const tr_kapprox_state *s, int32_t which, int32_t index) {
    if (!state_ok(s)) return NULL;
    switch (which) {
    case TR_KAPPROX_Q_PER_PAIR: return s->per_pair;
    case TR_KAPPROX_Q_HOEFFDING: return s->h;
    case TR_KAPPROX_Q_BIAS: return s->beta;
    case TR_KAPPROX_Q_RADIUS2: return s->rho2;
    case TR_KAPPROX_Q_U: return s->U;
    case TR_KAPPROX_Q_FAILURE: return s->failure;
    case TR_KAPPROX_Q_BUDGET: return s->budget;
    case TR_KAPPROX_Q_GAMMA: return s->gamma;
    case TR_KAPPROX_Q_ETA: return s->eta;
    case TR_KAPPROX_Q_PBAD: return s->pbad;
    case TR_KAPPROX_Q_ETA_NUM: return s->eta_num;
    case TR_KAPPROX_Q_COEF: return index >= 0 && index < s->ncoef ? s->coef[index] : NULL;
    default: return NULL;
    }
}

/* ================================================================ answers */
static int32_t cls_of(const tr_kapprox_state *s, uint64_t t) { return s->spec.method == TR_KAPPROX_TAYLOR ? s->it.cls[t] : 0; }
static uint32_t mult_of(const tr_kapprox_state *s, uint64_t t) {
    return s->spec.method == TR_KAPPROX_TAYLOR ? s->it.mult[t] : s->fmult[t / 2];
}
static __int128 mag128(__int128 x) { return x < 0 ? -x : x; }
/* sum_c coef_c (cen_c -+ rad_c) / 2^den over the classes: [lo, hi] */
static void combine(cx *c, const tr_kapprox_state *s, const uint64_t *cen, const uint64_t *rad, int64_t den,
                    tr_bigq **lo, tr_bigq **hi) {
    tr_bigq *L = Q_int(c, 0), *H = Q_int(c, 0);
    for (int32_t k = 0; k < s->ncoef && !c->rc; k++) {
        int zero = 1;
        for (int i = 0; i < TR_KERNEL_ACC_WORDS; i++) zero = zero && !cen[k * TR_KERNEL_ACC_WORDS + i] && !rad[k * TR_KERNEL_ACC_WORDS + i];
        if (zero) continue;
        tr_bigint *zc = Z_twos(c, cen + k * TR_KERNEL_ACC_WORDS, TR_KERNEL_ACC_WORDS);
        tr_bigint *zr = Z_twos(c, rad + k * TR_KERNEL_ACC_WORDS, TR_KERNEL_ACC_WORDS);
        L = Q_add(c, L, Q_mul(c, s->coef[k], Q_z(c, Z_sub(c, zc, zr))));
        H = Q_add(c, H, Q_mul(c, s->coef[k], Q_z(c, Z_add(c, zc, zr))));
    }
    tr_bigq *scale = Q_pow2(c, -den);
    *lo = Q_mul(c, L, scale);
    *hi = Q_mul(c, H, scale);
}
/* K(g, h) = sum_t coef mult Phi_g[t] Phi_h[t] enclosed: 4 Phi Phi' in [M M' -+ (|M| R' + |M'| R + R R')] with M = L + H,
 * R = H - L, summed per class in KERNEL-001's 384-bit accumulators; logical weight units */
static int32_t inner(cx *c, const tr_kapprox_state *s, uint64_t g, uint64_t h, tr_bigq **lo, tr_bigq **hi) {
    const uint64_t n = s->nitems;
    uint64_t *cen = (uint64_t *)zalloc((uint64_t)s->ncoef * TR_KERNEL_ACC_WORDS * 8);
    uint64_t *rad = (uint64_t *)zalloc((uint64_t)s->ncoef * TR_KERNEL_ACC_WORDS * 8);
    if (!cen || !rad) { free(cen); free(rad); return TR_MATH_MEMORY; }
    const tr_math_i128 *A = s->sums + g * n * 2, *B = s->sums + h * n * 2;
    for (uint64_t t = 0; t < n; t++) {
        __int128 La = tr_i128_get(A[2 * t]), Ha = tr_i128_get(A[2 * t + 1]), Lb = tr_i128_get(B[2 * t]),
                 Hb = tr_i128_get(B[2 * t + 1]);
        if ((!La && !Ha) || (!Lb && !Hb)) continue;
        __int128 Ma = La + Ha, Ra = Ha - La, Mb = Lb + Hb, Rb = Hb - Lb;
        uint64_t *ce = cen + (uint64_t)cls_of(s, t) * TR_KERNEL_ACC_WORDS, *ra = rad + (uint64_t)cls_of(s, t) * TR_KERNEL_ACC_WORDS;
        uint32_t m = mult_of(s, t);
        tr_kernel_acc_add_prod(ce, Ma, Mb, m);
        if (Rb) tr_kernel_acc_add_prod(ra, mag128(Ma), Rb, m);
        if (Ra) tr_kernel_acc_add_prod(ra, mag128(Mb), Ra, m);
        if (Ra && Rb) tr_kernel_acc_add_prod(ra, Ra, Rb, m);
    }
    combine(c, s, cen, rad, 2 * (int64_t)s->v.scale + 2, lo, hi);
    free(cen); free(rad);
    return c->rc;
}
/* sum_i w_i k~(x_i, z) of group g at a point's features (value (c +- r) 2^e / 2^120): logical weight units */
static int32_t witness_sum(cx *c, const tr_kapprox_state *s, uint64_t g, const tr_calc_fx *pf, int64_t e, tr_bigq **lo,
                           tr_bigq **hi) {
    const uint64_t n = s->nitems;
    uint64_t *cen = (uint64_t *)zalloc((uint64_t)s->ncoef * TR_KERNEL_ACC_WORDS * 8);
    uint64_t *rad = (uint64_t *)zalloc((uint64_t)s->ncoef * TR_KERNEL_ACC_WORDS * 8);
    if (!cen || !rad) { free(cen); free(rad); return TR_MATH_MEMORY; }
    const tr_math_i128 *A = s->sums + g * n * 2;
    for (uint64_t t = 0; t < n; t++) {
        __int128 La = tr_i128_get(A[2 * t]), Ha = tr_i128_get(A[2 * t + 1]);
        if (!La && !Ha) continue;
        __int128 Ma = La + Ha, Ra = Ha - La, pc = pf[t].c, pr = (__int128)pf[t].r;
        uint64_t *ce = cen + (uint64_t)cls_of(s, t) * TR_KERNEL_ACC_WORDS, *ra = rad + (uint64_t)cls_of(s, t) * TR_KERNEL_ACC_WORDS;
        uint32_t m = mult_of(s, t);
        if (pc) tr_kernel_acc_add_prod(ce, Ma, pc, m);
        if (pr) tr_kernel_acc_add_prod(ra, mag128(Ma), pr, m);
        if (Ra && pc) tr_kernel_acc_add_prod(ra, mag128(pc), Ra, m);
        if (Ra && pr) tr_kernel_acc_add_prod(ra, Ra, pr, m);
    }
    combine(c, s, cen, rad, (int64_t)s->v.scale + 1 + TR_CALC_F - e, lo, hi);
    free(cen); free(rad);
    return c->rc;
}
/* the features of a point (logical values Z): DOMAIN_ERROR outside the domain */
static int32_t point_features(const tr_kapprox_state *s, const int64_t *Z, tr_calc_fx *out, int64_t *e) {
    const int32_t d = s->spec.ndims;
    uint64_t sq[3];
    tr_kapprox_sumsq(Z, d, sq);
    if (tr_kapprox_cmp192(sq, s->tdom) > 0) return TR_MATH_DOMAIN_ERROR;
    if (s->spec.method == TR_KAPPROX_TAYLOR) {
        tr_calc_fx m;
        if (!tr_kapprox_prefactor(&s->v, Z, e, &m)) return TR_MATH_OVERFLOW;
        out[0] = m;
        for (uint64_t t = 1; t < s->nitems; t++)
            out[t] = tr_calc_fx_mul(out[s->it.parent[t]], tr_kapprox_coord(&s->v, Z[s->it.axis[t]]));
        return TR_MATH_OK;
    }
    *e = 0;
    for (uint64_t f = 0; f < s->m; f++) {                             /* tr_fourier_row_angle of the point */
        uint64_t acc[6] = {0, 0, 0, 0, 0, 0};
        int lost = 0, inexact = 0, any = 0;
        const uint64_t e0 = f * (uint64_t)d;
        tr_calc_fx sn, cs;
        for (int32_t j = 0; j < d; j++) {
            int32_t ks = s->ksign[e0 + (uint64_t)j];
            if (Z[j] == 0 || ks == 0) continue;
            if (!s->kexact[e0 + (uint64_t)j]) inexact = 1;
            tr_fourier_term(acc, s->k + 4 * (e0 + (uint64_t)j), (Z[j] < 0) != (ks < 0), tr_abs_u64(Z[j]), 0, &lost);
            any = 1;
        }
        if (!any) { sn.c = 0; sn.r = 0; cs.c = (__int128)1 << TR_CALC_F; cs.r = 0; }
        else tr_fourier_finish(acc, inexact || lost, &sn, &cs);
        out[2 * f] = cs; out[2 * f + 1] = sn;
    }
    return TR_MATH_OK;
}
typedef struct { tr_bigq *W, *A, *Q; } gstat;
static gstat group_stats(cx *c, const tr_kapprox_state *s, uint64_t g) {
    gstat r;
    r.W = Q_z(c, Z_i128(c, tr_i128_get(s->groups[g].W)));
    r.A = Q_z(c, Z_i128(c, tr_i128_get(s->groups[g].A)));
    r.Q = Q_z(c, Z_limbs(c, s->groups[g].Q, 3));
    return r;
}
/* [lo, hi] x q (sign aware) */
static void iv_scale(cx *c, tr_bigq **lo, tr_bigq **hi, const tr_bigq *q) {
    tr_bigq *a = Q_mul(c, *lo, q), *b = Q_mul(c, *hi, q);
    if (c->rc) return;
    if (tr_bigq_sign(q) < 0) { *lo = b; *hi = a; } else { *lo = a; *hi = b; }
}
static void iv_div(cx *c, tr_bigq **lo, tr_bigq **hi, const tr_bigq *q) {
    if (c->rc) return;
    tr_bigq *inv = Q_div(c, Q_int(c, 1), q);
    iv_scale(c, lo, hi, inv);
}
static int req_valid(const tr_kapprox_state *s, const tr_kapprox_request *q, uint64_t npoints) {
    if (q->reserved || q->a >= s->ngroups) return 0;
    switch (q->op) {
    case TR_KAPPROX_OP_SUM:
    case TR_KAPPROX_OP_MEAN: return q->b < s->ngroups && !(q->flags & ~(TR_KAPPROX_SELF | TR_KAPPROX_UNORDERED));
    case TR_KAPPROX_OP_MMD2: return q->b < s->ngroups && !(q->flags & ~TR_KAPPROX_UNBIASED);
    case TR_KAPPROX_OP_WITNESS: return !q->flags && q->point < npoints;
    default: return 0;
    }
}
static void answer(const tr_kapprox_state *s, const tr_kapprox_request *q, const tr_calc_fx *pf, const int64_t *pe,
                   const int32_t *pst, uint64_t npoints, tr_kapprox_result *r, const tr_math_budget *budget,
                   tr_math_refusal *refusal) {
    memset(r, 0, sizeof *r);
    r->label = s->spec.method == TR_KAPPROX_TAYLOR ? TR_APPROX_LABEL_CERTIFIED : TR_APPROX_LABEL_PROBABILISTIC;
    if (!req_valid(s, q, npoints)) { r->status = TR_MATH_INVALID; return; }
    if (q->op == TR_KAPPROX_OP_WITNESS && pst[q->point]) { r->status = pst[q->point]; return; }
    cx c;
    cx_init(&c, budget, refusal);
    tr_bigq *lo = NULL, *hi = NULL, *F = NULL, *R = NULL, *pairs = NULL;
    int empty = 0;
    gstat a = group_stats(&c, s, q->a), b = q->op == TR_KAPPROX_OP_WITNESS ? a : group_stats(&c, s, q->b);
    if (q->op == TR_KAPPROX_OP_SUM || q->op == TR_KAPPROX_OP_MEAN) {
        int32_t rc = inner(&c, s, q->a, q->b, &lo, &hi);
        if (rc && !c.rc) c.rc = rc;
        if (q->a != q->b) {
            pairs = Q_mul(&c, a.W, b.W); F = Q_mul(&c, a.A, b.A); R = Q_mul(&c, Q_int(&c, 2), F);
        } else {
            int self = (q->flags & TR_KAPPROX_SELF) != 0, unordered = (q->flags & TR_KAPPROX_UNORDERED) != 0;
            pairs = Q_mul(&c, a.W, a.W); F = Q_mul(&c, a.A, a.A); R = F;
            if (!self) { lo = Q_sub(&c, lo, a.Q); hi = Q_sub(&c, hi, a.Q); pairs = Q_sub(&c, pairs, a.Q); }
            else if (unordered) { lo = Q_add(&c, lo, a.Q); hi = Q_add(&c, hi, a.Q); pairs = Q_add(&c, pairs, a.Q); }
            if (unordered) {
                tr_bigq *half = Q_pow2(&c, -1);
                lo = Q_mul(&c, lo, half); hi = Q_mul(&c, hi, half); pairs = Q_mul(&c, pairs, half);
                F = Q_mul(&c, F, half); R = Q_mul(&c, R, half);
            }
        }
        if (q->op == TR_KAPPROX_OP_MEAN && !c.rc) {
            if (Q_is_zero(pairs)) empty = 1;
            else { iv_div(&c, &lo, &hi, pairs); tr_bigq *ap = Q_abs(&c, pairs); F = Q_div(&c, F, ap); R = Q_div(&c, R, ap); }
        }
        if (!c.rc) {                                                  /* value units: / 10^(2 wscale) */
            tr_bigq *u = Q_div(&c, Q_int(&c, 1), Q_z(&c, Z_pow10(&c, 2 * (uint64_t)s->wscale)));
            pairs = Q_mul(&c, pairs, u);
            if (q->op == TR_KAPPROX_OP_SUM) { lo = Q_mul(&c, lo, u); hi = Q_mul(&c, hi, u); F = Q_mul(&c, F, u); R = Q_mul(&c, R, u); }
        }
    } else if (q->op == TR_KAPPROX_OP_MMD2) {
        tr_bigq *al = NULL, *ah = NULL, *bl = NULL, *bh = NULL, *xl = NULL, *xh = NULL;
        int32_t rc = inner(&c, s, q->a, q->a, &al, &ah);
        if (!rc) rc = inner(&c, s, q->b, q->b, &bl, &bh);
        if (!rc) rc = inner(&c, s, q->a, q->b, &xl, &xh);
        if (rc && !c.rc) c.rc = rc;
        int unbiased = (q->flags & TR_KAPPROX_UNBIASED) != 0;
        tr_bigq *Pab = Q_mul(&c, a.W, b.W), *da = Q_mul(&c, a.W, a.W), *db = Q_mul(&c, b.W, b.W);
        if (unbiased) {
            da = Q_sub(&c, da, a.Q); db = Q_sub(&c, db, b.Q);
            al = Q_sub(&c, al, a.Q); ah = Q_sub(&c, ah, a.Q); bl = Q_sub(&c, bl, b.Q); bh = Q_sub(&c, bh, b.Q);
        }
        if (!c.rc && (Q_is_zero(da) || Q_is_zero(db) || Q_is_zero(Pab))) empty = 1;
        else if (!c.rc) {
            iv_div(&c, &al, &ah, da); iv_div(&c, &bl, &bh, db);
            iv_div(&c, &xl, &xh, Pab); iv_scale(&c, &xl, &xh, Q_int(&c, -2));
            lo = Q_add(&c, Q_add(&c, al, bl), xl); hi = Q_add(&c, Q_add(&c, ah, bh), xh);
            tr_bigq *ada = Q_abs(&c, da), *adb = Q_abs(&c, db), *apab = Q_abs(&c, Pab);
            tr_bigq *t1 = Q_div(&c, Q_mul(&c, a.A, a.A), ada), *t2 = Q_div(&c, Q_mul(&c, b.A, b.A), adb);
            tr_bigq *t3 = Q_div(&c, Q_mul(&c, a.A, b.A), apab);
            if (unbiased) { F = Q_add(&c, Q_add(&c, t1, t2), Q_mul(&c, Q_int(&c, 2), t3)); R = Q_add(&c, Q_add(&c, t1, t2), Q_mul(&c, Q_int(&c, 4), t3)); }
            else { F = Q_add(&c, Q_add(&c, t1, t2), Q_mul(&c, Q_int(&c, 2), t3)); R = F; }   /* (A_a/|W_a| + A_b/|W_b|)^2 */
        }
    } else {
        if (!c.rc && Q_is_zero(a.W)) empty = 1;
        else {
            int32_t rc = witness_sum(&c, s, q->a, pf + q->point * s->nitems, pe[q->point], &lo, &hi);
            if (rc && !c.rc) c.rc = rc;
            iv_div(&c, &lo, &hi, a.W);
            tr_bigq *aw = Q_abs(&c, a.W);
            F = Q_div(&c, a.A, aw); R = Q_mul(&c, Q_int(&c, 2), F);
        }
    }
    if (!c.rc) {
        r->empty = empty;
        if (pairs) r->pairs = Q_out(&c, pairs);
        if (!empty) {
            tr_bigq *term[TR_KAPPROX_MAX_TERMS] = {NULL, NULL, NULL, NULL};
            if (s->spec.method == TR_KAPPROX_TAYLOR) {
                r->nterms = 1; r->term_kind[0] = TR_APPROX_TERM_REMAINDER; term[0] = Q_mul(&c, s->per_pair, F);
            } else {
                r->nterms = 2;
                r->term_kind[0] = TR_APPROX_TERM_PROBABILISTIC; term[0] = Q_mul(&c, s->h, R);
                r->term_kind[1] = TR_APPROX_TERM_QUANTIZATION; term[1] = Q_mul(&c, s->beta, F);
            }
            tr_bigq *err = Q_int(&c, 0);
            for (int32_t i = 0; i < r->nterms; i++) err = Q_add(&c, err, term[i]);
            tr_bigq *blo = Q_sub(&c, lo, err), *bhi = Q_add(&c, hi, err);
            r->lo = Q_out(&c, lo); r->hi = Q_out(&c, hi); r->error = Q_out(&c, err);
            r->bound_lo = Q_out(&c, blo); r->bound_hi = Q_out(&c, bhi);
            for (int32_t i = 0; i < r->nterms; i++) r->term[i] = Q_out(&c, term[i]);
            if (s->failure) r->failure = Q_out(&c, s->failure);
            tr_math_interval iv;
            if (!c.rc) {
                r->lo_d = tr_bigq_enclose(r->lo, &iv) ? -INFINITY : iv.lo;
                r->hi_d = tr_bigq_enclose(r->hi, &iv) ? INFINITY : iv.hi;
                r->error_d = tr_bigq_enclose(r->error, &iv) ? INFINITY : iv.hi;
            }
        }
    }
    if (c.rc) {
        int32_t rc = c.rc;
        tr_kapprox_result_free(r, 1);
        memset(r, 0, sizeof *r);
        r->status = rc;
    }
    cx_done(&c);
}
void tr_kapprox_result_free(tr_kapprox_result *r, uint64_t n) {
    if (!r) return;
    for (uint64_t i = 0; i < n; i++) {
        tr_bigq *q[] = {r[i].lo, r[i].hi, r[i].error, r[i].bound_lo, r[i].bound_hi, r[i].failure, r[i].pairs};
        for (size_t k = 0; k < sizeof q / sizeof q[0]; k++) tr_bigq_free(q[k]);
        for (int k = 0; k < TR_KAPPROX_MAX_TERMS; k++) tr_bigq_free(r[i].term[k]);
        r[i].lo = r[i].hi = r[i].error = r[i].bound_lo = r[i].bound_hi = r[i].failure = r[i].pairs = NULL;
        for (int k = 0; k < TR_KAPPROX_MAX_TERMS; k++) r[i].term[k] = NULL;
    }
}
int32_t tr_kapprox_query(const tr_kapprox_state *s, const tr_kapprox_request *req, uint64_t n, const int64_t *points,
                         uint64_t npoints, tr_kapprox_result *out, const tr_math_budget *budget, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    tr_math_budget dflt;
    budget = budget_of(budget, &dflt);
    if (!state_ok(s) || (n && (!req || !out)) || (npoints && !points)) return TR_MATH_INVALID;
    tr_math_need need;
    need_zero(&need);
    need.need[TR_BUDGET_OUTPUT] = n;
    need.need[TR_BUDGET_TENSOR] = tr_budget_mul(npoints, s->nitems);
    need.need[TR_BUDGET_BYTES] = tr_budget_add(tr_budget_mul(need.need[TR_BUDGET_TENSOR], sizeof(tr_calc_fx)),
                                               tr_budget_mul(npoints, 12));
    int32_t rc = tr_budget_check(budget, &need, refusal);
    if (rc) return rc;
    tr_calc_fx *pf = npoints ? (tr_calc_fx *)zalloc(npoints * s->nitems * sizeof *pf) : NULL;
    int64_t *pe = npoints ? (int64_t *)zalloc(npoints * 8) : NULL;
    int32_t *pst = npoints ? (int32_t *)zalloc(npoints * 4) : NULL;
    if (npoints && (!pf || !pe || !pst)) { free(pf); free(pe); free(pst); return TR_MATH_MEMORY; }
    for (uint64_t p = 0; p < npoints; p++)
        pst[p] = point_features(s, points + p * (uint64_t)s->spec.ndims, pf + p * s->nitems, &pe[p]);
    for (uint64_t i = 0; i < n; i++) answer(s, &req[i], pf, pe, pst, npoints, &out[i], budget, refusal);
    free(pf); free(pe); free(pst);
    return TR_MATH_OK;                                                /* refusal: the last answer refused, if any */
}

/* ================================================================ FND-007 */
int32_t tr_kapprox_authorize_exact(const tr_kapprox_state *s, uint64_t field, tr_cert_decision *out) {
    if (!state_ok(s) || !out) return TR_MATH_INVALID;
    memset(out, 0, sizeof *out);
    const int taylor = s->spec.method == TR_KAPPROX_TAYLOR;
    out->authorized = 0;
    out->reason = taylor ? TR_CERT_REFUSE_APPROXIMATE : TR_CERT_REFUSE_STATISTICAL;
    out->kind = taylor ? TR_APPROX_LABEL_CERTIFIED : TR_APPROX_LABEL_PROBABILISTIC;
    out->claim = 0; out->policy = 0; out->premise = -1; out->field = field;
    out->stale.column = -1; out->stale.dim = -1;
    return TR_MATH_DOMAIN_ERROR;
}
int32_t tr_kapprox_fresh(const tr_kapprox_state *s, const tr_scope *current, tr_cert_staleness *out) {
    if (!state_ok(s) || !current) return TR_MATH_INVALID;
    tr_cert_staleness local;
    if (!out) out = &local;
    memset(out, 0, sizeof *out);
    out->column = out->dim = -1;
    if (!s->source) return TR_MATH_UNSUPPORTED;
    return tr_cert_record_fresh(s->source, current, NULL, 0, TR_CERT_MATCH_EXACT, out);
}

/* ================================================================ persistence (FND-011's record format) */
typedef struct { uint8_t *p; uint64_t len, cap; int fail; } wbuf;
static void w64(wbuf *w, uint64_t v) {
    if (w->p) {
        if (w->len + 8 > w->cap) { w->fail = 1; w->len += 8; return; }
        for (int i = 0; i < 8; i++) w->p[w->len + (uint64_t)i] = (uint8_t)(v >> (8 * i));
    }
    w->len += 8;
}
static void wi(wbuf *w, int64_t v) { w64(w, (uint64_t)v); }
static void wq(wbuf *w, tr_num_rational q) { wi(w, q.numerator); wi(w, q.denominator); }
static void w_spec(wbuf *w, const tr_kapprox_spec *s) {
    wi(w, s->kernel); wi(w, s->method); wi(w, s->ndims); wi(w, s->weight);
    wi(w, s->degree); wi(w, s->has_radius); wi(w, s->has_budget); wi(w, s->reserved);
    for (int j = 0; j < TR_KAPPROX_MAX_DIMS; j++) wi(w, s->column[j]);
    wq(w, s->gamma); wq(w, s->radius2); wq(w, s->failure); wq(w, s->budget);
    w64(w, s->nfreq); w64(w, s->seed);
}
static void basis_section(wbuf *w, const tr_kapprox_state *s) {
    w_spec(w, &s->spec);
    wi(w, s->v.scale); wi(w, s->v.shift); wi(w, s->xscale); wi(w, s->wscale); wi(w, s->fbits);
    w64(w, s->ngroups); w64(w, s->nitems); w64(w, s->m); w64(w, s->D); w64(w, s->selected); w64(w, s->rows);
    w64(w, s->v.tmax);
    for (int i = 0; i < TR_KAPPROX_KQ_LIMBS; i++) w64(w, s->v.kq[i]);
    for (int i = 0; i < 3; i++) w64(w, s->maxsq[i]);
    for (int i = 0; i < 3; i++) w64(w, s->E[i]);
    for (int i = 0; i < 3; i++) w64(w, s->T[i]);
    if (s->spec.method != TR_KAPPROX_RFF) return;
    const uint64_t md = s->m * (uint64_t)s->spec.ndims;
    for (uint64_t i = 0; i < md; i++) wq(w, s->omega[i]);
    for (uint64_t f = 0; f < s->m; f++) w64(w, s->fmult[f]);
    for (uint64_t i = 0; i < 4 * md; i++) w64(w, s->k[i]);
    for (uint64_t i = 0; i < md; i++) wi(w, s->ksign[i]);
    for (uint64_t i = 0; i < md; i++) w64(w, s->kexact[i]);
}
static void payload_section(wbuf *w, const tr_kapprox_state *s) {
    for (uint64_t g = 0; g < s->ngroups; g++) {
        const tr_kapprox_group *x = &s->groups[g];
        w64(w, x->rows); w64(w, x->valid); wi(w, x->W.lo); wi(w, x->W.hi); wi(w, x->A.lo); wi(w, x->A.hi);
        w64(w, x->Q[0]); w64(w, x->Q[1]); w64(w, x->Q[2]);
    }
    for (uint64_t i = 0; i < s->ngroups * s->nitems * 2; i++) { wi(w, s->sums[i].lo); wi(w, s->sums[i].hi); }
}
static uint64_t pad8(uint64_t n) { return n > UINT64_MAX - 7 ? UINT64_MAX : (n + 7) & ~(uint64_t)7; }
static uint64_t le64(const uint8_t *p) { uint64_t v = 0; for (int i = 0; i < 8; i++) v |= (uint64_t)p[i] << (8 * i); return v; }
static void put64(uint8_t *p, uint64_t v) { for (int i = 0; i < 8; i++) p[i] = (uint8_t)(v >> (8 * i)); }
static void put32(uint8_t *p, uint32_t v) { for (int i = 0; i < 4; i++) p[i] = (uint8_t)(v >> (8 * i)); }
/* FND-011's checksum: every little-endian word but words 12 and 13, weighted by tr_persist_weight */
static void record_sum(const uint8_t *d, uint64_t total, uint64_t out[2]) {
    uint64_t a = 0, b = 0;
    for (uint64_t i = 0; i < total / 8; i++) {
        if (i == TR_PERSIST_CHECKSUM_WORD || i == TR_PERSIST_CHECKSUM_WORD + 1) continue;
        uint64_t w = le64(d + 8 * i);
        a += w * tr_persist_weight(i, 0); b += w * tr_persist_weight(i, 1);
    }
    out[0] = a; out[1] = b;
}
int32_t tr_kapprox_save(const tr_kapprox_state *s, const uint8_t *meta, uint64_t meta_bytes, uint8_t *out,
                        uint64_t capacity, uint64_t *size, const tr_math_budget *budget, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    tr_math_budget dflt;
    budget = budget_of(budget, &dflt);
    if (!state_ok(s) || !size || (meta_bytes && !meta)) return TR_MATH_INVALID;
    wbuf cb = {NULL, 0, 0, 0}, cp = {NULL, 0, 0, 0};
    basis_section(&cb, s); payload_section(&cp, s);
    uint64_t sbytes = s->source ? tr_cert_record_serial_size(s->source) : 0;
    uint64_t total = TR_PERSIST_HEADER;
    total = tr_budget_add(total, pad8(sbytes)); total = tr_budget_add(total, pad8(cb.len));
    total = tr_budget_add(total, pad8(meta_bytes)); total = tr_budget_add(total, pad8(cp.len));
    tr_math_need need;
    need_zero(&need);
    need.need[TR_BUDGET_BYTES] = tr_persist_save_need(total);
    int32_t rc = tr_budget_check(budget, &need, refusal);
    if (rc) return rc;
    *size = total;
    if (!out) return TR_MATH_OK;
    if (capacity < total) return TR_MATH_INVALID;
    uint8_t *d = out;
    memset(d, 0, (size_t)total);
    put64(d, TR_PERSIST_MAGIC); put32(d + 8, TR_PERSIST_FORMAT); put32(d + 12, TR_PERSIST_HEADER);
    put64(d + 16, TR_PERSIST_ENDIAN); put32(d + 24, TR_KAPPROX_TYPE); put32(d + 28, TR_KAPPROX_TYPE_VERSION);
    put32(d + 32, TR_CARRIER_INTERVAL); put32(d + 36, 0);                  /* an approximate state: exact 0 */
    put32(d + 40, (uint32_t)((s->source ? TR_PERSIST_SOURCE : 0) | TR_PERSIST_IMMUTABLE));
    put64(d + 48, sbytes); put64(d + 56, cb.len); put64(d + 64, meta_bytes); put64(d + 72, cp.len); put64(d + 80, total);
    uint64_t at = TR_PERSIST_HEADER;
    if (s->source && tr_cert_record_serialize(s->source, d + at, sbytes, NULL)) return TR_MATH_INVALID;
    at += pad8(sbytes);
    wbuf wb = {d + at, 0, cb.len, 0};
    basis_section(&wb, s);
    at += pad8(cb.len);
    if (meta_bytes) memcpy(d + at, meta, (size_t)meta_bytes);
    at += pad8(meta_bytes);
    wbuf wp = {d + at, 0, cp.len, 0};
    payload_section(&wp, s);
    if (wb.fail || wp.fail || wb.len != cb.len || wp.len != cp.len) return TR_MATH_INVALID;
    uint64_t sum[2];
    record_sum(d, total, sum);
    put64(d + 96, sum[0]); put64(d + 104, sum[1]);
    return TR_MATH_OK;
}

typedef struct { const uint8_t *p; uint64_t len, at; int fail; } rbuf;
static uint64_t r64(rbuf *r) {
    if (r->fail || r->at + 8 > r->len) { r->fail = 1; return 0; }
    uint64_t v = le64(r->p + r->at);
    r->at += 8;
    return v;
}
static int64_t ri(rbuf *r) { return (int64_t)r64(r); }
static int32_t ri32(rbuf *r) {
    int64_t v = ri(r);
    if (v < INT32_MIN || v > INT32_MAX) { r->fail = 1; return 0; }
    return (int32_t)v;
}
static tr_num_rational rq(rbuf *r) { tr_num_rational q; q.numerator = ri(r); q.denominator = ri(r); return q; }
static int32_t r_spec(rbuf *r, tr_kapprox_spec *s) {
    memset(s, 0, sizeof *s);
    s->kernel = ri32(r); s->method = ri32(r); s->ndims = ri32(r); s->weight = ri32(r);
    s->degree = ri32(r); s->has_radius = ri32(r); s->has_budget = ri32(r); s->reserved = ri32(r);
    for (int j = 0; j < TR_KAPPROX_MAX_DIMS; j++) s->column[j] = ri32(r);
    s->gamma = rq(r); s->radius2 = rq(r); s->failure = rq(r); s->budget = rq(r);
    s->nfreq = r64(r); s->seed = r64(r);
    return r->fail ? TR_MATH_INVALID : TR_MATH_OK;
}
static int status_fail(tr_persist_status *st, int32_t reason) { if (st) st->reason = reason; return 0; }
/* the sections into a new state (structural checks; MALFORMED on any) */
static int32_t parse_state(tr_kapprox_state *s, rbuf *b, rbuf *p) {
    int32_t rc = r_spec(b, &s->spec);
    if (rc) return rc;
    tr_kapprox_spec canon = s->spec;
    if (spec_check(&canon, TR_SCOPE_MAX_COLUMNS) || memcmp(&canon, &s->spec, sizeof canon)) return TR_MATH_INVALID;
    s->v.scale = ri32(b); s->v.shift = ri32(b); s->xscale = ri32(b); s->wscale = ri32(b); s->fbits = ri32(b);
    s->ngroups = r64(b); s->nitems = r64(b); s->m = r64(b); s->D = r64(b); s->selected = r64(b); s->rows = r64(b);
    s->v.tmax = r64(b);
    for (int i = 0; i < TR_KAPPROX_KQ_LIMBS; i++) s->v.kq[i] = r64(b);
    for (int i = 0; i < 3; i++) s->maxsq[i] = r64(b);
    for (int i = 0; i < 3; i++) s->E[i] = r64(b);
    for (int i = 0; i < 3; i++) s->T[i] = r64(b);
    if (b->fail) return TR_MATH_INVALID;
    const tr_kapprox_spec *sp = &s->spec;
    const int taylor = sp->method == TR_KAPPROX_TAYLOR;
    if (s->ngroups < 1 || s->ngroups > TR_SCOPE_MAX_GROUPS || s->v.scale < 1 || s->v.scale > TR_CALC_SCALE_MAX ||
        s->xscale < 0 || s->xscale > TR_SCOPE_MAX_SCALE || s->wscale < 0 || s->wscale > TR_SCOPE_MAX_SCALE ||
        s->selected > s->rows || tr_kapprox_cmp192(s->maxsq, s->E) > 0) return TR_MATH_INVALID;
    if (taylor) {
        if (s->m || s->D || s->fbits || s->v.shift < 0 || s->v.shift > TR_CALC_F || s->v.tmax >= TR_KAPPROX_NMAX ||
            s->nitems != items_count(sp->ndims, sp->degree)) return TR_MATH_INVALID;
    } else {
        if (s->v.shift || s->v.tmax || s->D != sp->nfreq || s->m < 1 || s->m > s->D || s->nitems != 2 * s->m ||
            s->fbits < 1 || s->fbits > FBITS_MAX) return TR_MATH_INVALID;
        for (int i = 0; i < TR_KAPPROX_KQ_LIMBS; i++) if (s->v.kq[i]) return TR_MATH_INVALID;
    }
    if (s->ngroups > UINT64_MAX / 64 / (s->nitems ? s->nitems : 1) || s->m > UINT64_MAX / 128 / (uint64_t)sp->ndims)
        return TR_MATH_INVALID;
    /* the sections' exact sizes before anything is allocated */
    const uint64_t md = s->m * (uint64_t)sp->ndims;
    uint64_t want_b = 8 * (34 + 12 + TR_KAPPROX_KQ_LIMBS + 9) + (taylor ? 0 : 8 * (2 * md + s->m + 4 * md + md + md));
    uint64_t want_p = 8 * (9 * s->ngroups + 4 * s->ngroups * s->nitems);
    if (b->len != want_b || p->len != want_p) return TR_MATH_INVALID;
    if (taylor) {
        rc = items_make(sp->ndims, sp->degree, &s->it);
        if (rc) return rc;
    }
    rc = state_arrays(s);
    if (rc) return rc;
    if (!taylor) {
        uint64_t total = 0;
        for (uint64_t i = 0; i < md; i++) {
            s->omega[i] = rq(b);
            tr_num_rational cq;
            if (b->fail || s->omega[i].denominator <= 0 || !tr_q_make(s->omega[i].numerator, s->omega[i].denominator, &cq) ||
                cq.numerator != s->omega[i].numerator || cq.denominator != s->omega[i].denominator) return TR_MATH_INVALID;
        }
        for (uint64_t f = 0; f < s->m; f++) {
            uint64_t m = r64(b);
            if (!m || m > s->D) return TR_MATH_INVALID;
            s->fmult[f] = (uint32_t)m; total += m;
            if (f && freq_cmp(s->omega + (f - 1) * (uint64_t)sp->ndims, s->omega + f * (uint64_t)sp->ndims, sp->ndims) >= 0)
                return TR_MATH_INVALID;                                /* the canonical (sorted, distinct) set */
        }
        if (total != s->D) return TR_MATH_INVALID;
        for (uint64_t i = 0; i < 4 * md; i++) s->k[i] = r64(b);
        for (uint64_t i = 0; i < md; i++) {
            int64_t v = ri(b);
            if (v < -1 || v > 1) return TR_MATH_INVALID;
            s->ksign[i] = (int32_t)v;
        }
        for (uint64_t i = 0; i < md; i++) {
            uint64_t v = r64(b);
            if (v > 1) return TR_MATH_INVALID;
            s->kexact[i] = (uint8_t)v;
        }
        s->v.fourier.ndims = sp->ndims; s->v.fourier.weight = sp->weight; s->v.fourier.scale = s->v.scale;
        s->v.fourier.unit = TR_FOURIER_RADIAN;
        for (int32_t j = 0; j < sp->ndims; j++) s->v.fourier.column[j] = sp->column[j];
    }
    if (b->fail || b->at != b->len) return TR_MATH_INVALID;
    for (uint64_t g = 0; g < s->ngroups; g++) {
        tr_kapprox_group *x = &s->groups[g];
        x->rows = r64(p); x->valid = r64(p); x->W.lo = ri(p); x->W.hi = ri(p); x->A.lo = ri(p); x->A.hi = ri(p);
        x->Q[0] = r64(p); x->Q[1] = r64(p); x->Q[2] = r64(p);
    }
    for (uint64_t i = 0; i < s->ngroups * s->nitems * 2 && !p->fail; i++) { s->sums[i].lo = ri(p); s->sums[i].hi = ri(p); }
    if (p->fail || p->at != p->len) return TR_MATH_INVALID;
    if (check_groups(s->groups, s->ngroups, s->sums, s->nitems, s->selected)) return TR_MATH_INVALID;
    if (sp->has_radius && tr_kapprox_cmp192(s->maxsq, s->T) > 0) return TR_MATH_INVALID;
    tr_kapprox_plan_view *v = &s->v;
    v->method = sp->method; v->ndims = sp->ndims; v->weight = sp->weight; v->degree = sp->degree;
    v->xscale = s->xscale;
    for (int32_t j = 0; j < sp->ndims; j++) v->column[j] = sp->column[j];
    return TR_MATH_OK;
}
/* with an origin: the plan rebuilt from the recorded spec (the frequencies redrawn from the seed) agrees */
static int32_t matches_plan(const tr_kapprox_state *s, const tr_scope *origin, const tr_math_budget *budget,
                            tr_math_refusal *refusal) {
    int32_t rc;
    tr_kapprox_plan *p = tr_kapprox_plan_create(origin, &s->spec, budget, refusal, &rc);
    if (!p) return rc;
    int ok = p->v.scale == s->v.scale && p->v.shift == s->v.shift && p->xscale == s->xscale && p->wscale == s->wscale &&
             p->fbits == s->fbits && p->ngroups == s->ngroups && p->v.nitems == s->nitems && p->m == s->m &&
             p->D == s->D && p->selected == s->selected && p->rows == s->rows && p->v.tmax == s->v.tmax &&
             !memcmp(p->v.kq, s->v.kq, sizeof p->v.kq) && !memcmp(p->E, s->E, sizeof p->E) &&
             !memcmp(p->T, s->T, sizeof p->T);
    if (ok && s->spec.method == TR_KAPPROX_RFF) {
        const uint64_t md = s->m * (uint64_t)s->spec.ndims;
        ok = !memcmp(p->omega, s->omega, (size_t)md * sizeof *s->omega) && !memcmp(p->fmult, s->fmult, (size_t)s->m * 4) &&
             !memcmp(p->v.fourier.k, s->k, (size_t)md * 32) && !memcmp(p->v.fourier.ksign, s->ksign, (size_t)md * 4) &&
             !memcmp(p->v.fourier.kexact, s->kexact, (size_t)md);
    }
    tr_kapprox_plan_free(p);
    return ok ? TR_MATH_OK : TR_MATH_INVALID;
}
tr_kapprox_state *tr_kapprox_load(const uint8_t *data, uint64_t bytes, const tr_scope *origin, const tr_math_budget *budget,
                                  tr_math_refusal *refusal, tr_persist_status *status, tr_cert_staleness *stale,
                                  int32_t *error) {
    tr_budget_none(refusal);
    tr_math_budget dflt;
    budget = budget_of(budget, &dflt);
    tr_persist_status local;
    if (!status) status = &local;
    if (stale) { memset(stale, 0, sizeof *stale); stale->column = stale->dim = -1; }
    tr_persist_header h;
    int32_t rc = tr_persist_header_read(data, bytes, &h, status);       /* truncated, magic, byte order, version, sizes */
    if (rc) { seterr(error, rc); return NULL; }
    uint64_t sum[2];
    record_sum(data, h.total_bytes, sum);
    status->stored[0] = h.checksum[0]; status->stored[1] = h.checksum[1];
    status->computed[0] = sum[0]; status->computed[1] = sum[1];
    if (sum[0] != h.checksum[0] || sum[1] != h.checksum[1]) { status_fail(status, TR_PERSIST_CORRUPT); seterr(error, TR_MATH_INVALID); return NULL; }
    if (h.type != TR_KAPPROX_TYPE) { status_fail(status, TR_PERSIST_TYPE); seterr(error, TR_MATH_DOMAIN_ERROR); return NULL; }
    if (h.type_version != TR_KAPPROX_TYPE_VERSION) { status_fail(status, TR_PERSIST_VERSION); seterr(error, TR_MATH_UNSUPPORTED); return NULL; }
    if (h.exact != 0 || h.carrier != TR_CARRIER_INTERVAL) { status_fail(status, TR_PERSIST_MALFORMED); seterr(error, TR_MATH_INVALID); return NULL; }
    tr_math_need need;
    need_zero(&need);
    need.need[TR_BUDGET_BYTES] = tr_persist_load_need(h.total_bytes);
    rc = tr_budget_check(budget, &need, refusal);
    if (rc) { seterr(error, rc); return NULL; }
    tr_kapprox_state *s = state_new();
    if (!s) { seterr(error, TR_MATH_MEMORY); return NULL; }
    uint64_t at = TR_PERSIST_HEADER;
    if (h.flags & TR_PERSIST_SOURCE) {
        uint64_t used = 0;
        tr_math_budget unlimited;
        for (int i = 0; i < TR_BUDGET_RESOURCES; i++) unlimited.limit[i] = UINT64_MAX;
        s->source = tr_cert_record_deserialize(data + at, h.source_bytes, &used, &unlimited, NULL, &rc);
        if (!s->source || used != h.source_bytes) rc = rc && rc != TR_MATH_INVALID ? rc : TR_MATH_INVALID;
        else rc = TR_MATH_OK;
    }
    at += pad8(h.source_bytes);
    rbuf b = {data + at, h.basis_bytes, 0, 0};
    at += pad8(h.basis_bytes);
    const uint8_t *meta = data + at;
    at += pad8(h.metadata_bytes);
    rbuf p = {data + at, h.payload_bytes, 0, 0};
    if (!rc) rc = parse_state(s, &b, &p);
    if (rc) {
        if (rc == TR_MATH_INVALID) status_fail(status, TR_PERSIST_MALFORMED);
        state_release(s); seterr(error, rc); return NULL;
    }
    if (h.metadata_bytes) {
        s->meta = (uint8_t *)zalloc(h.metadata_bytes);
        if (!s->meta) { state_release(s); seterr(error, TR_MATH_MEMORY); return NULL; }
        memcpy(s->meta, meta, (size_t)h.metadata_bytes);
        s->meta_bytes = h.metadata_bytes;
    }
    if (origin) {
        if (!s->source) rc = TR_MATH_DOMAIN_ERROR;                       /* an unscoped record has no source */
        else {
            tr_cert_staleness st;
            memset(&st, 0, sizeof st);
            rc = tr_cert_record_fresh(s->source, origin, NULL, 0, TR_CERT_MATCH_EXACT, &st);
            if (stale) *stale = st;
        }
        if (!rc) {
            rc = matches_plan(s, origin, budget, refusal);
            if (rc == TR_MATH_INVALID) status_fail(status, TR_PERSIST_MALFORMED);
        }
        if (rc) { state_release(s); seterr(error, rc); return NULL; }
    }
    rc = state_constants(s, budget, refusal);
    if (rc) { if (rc == TR_MATH_INVALID) status_fail(status, TR_PERSIST_MALFORMED); state_release(s); seterr(error, rc); return NULL; }
    state_view(s);
    status->reason = TR_PERSIST_OK;
    seterr(error, TR_MATH_OK);
    return s;
}
int32_t tr_kapprox_get_metadata(const tr_kapprox_state *s, uint8_t *out, uint64_t capacity, uint64_t *length) {
    if (!state_ok(s) || !length) return TR_MATH_INVALID;
    *length = s->meta_bytes;
    if (!out) return TR_MATH_OK;
    if (capacity < s->meta_bytes) return TR_MATH_INVALID;
    if (s->meta_bytes) memcpy(out, s->meta, (size_t)s->meta_bytes);
    return TR_MATH_OK;
}
