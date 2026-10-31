/* CALC-003: non-polynomial finite-separation calculus.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Closed families whose dependence on k parameters separates into a few retained data coefficients, over the
 * selected rows of a scope (math_scope.h), per dense group:
 *
 *   PHASE   F(phi_1..phi_k) = sum_r w_r prod_i sin(omega_i x_ri + phi_i)
 *           = sum_S C_S prod_{i in S} sin(phi_i) prod_{i not in S} cos(phi_i),
 *           C_S = sum_r w_r prod_{i in S} cos(theta_ri) prod_{i not in S} sin(theta_ri),  theta_ri = omega_i x_ri
 *           (k = 1: F = A sin(phi) + B cos(phi), A = C_1 = sum w cos(theta), B = C_0 = sum w sin(theta)); 2^k
 *           coefficients (the product expansion), budgeted.
 *   EXP     G(theta_1..theta_k) = sum_r w_r exp(sum_i (theta_i + a_i x_ri)) = E exp(theta_1 + ... + theta_k),
 *           E = sum_r w_r exp(sum_i a_i x_ri); one coefficient.
 *
 * x_ri is the exact logical value of scope column column[i] at row r divided by 10^scale (a decimal), w_r the exact
 * logical value of the weight column (1 without one) divided by 10^its scale; a row counts when it is selected and
 * every column the family reads is present (SQL: NULLs are skipped). omega_i / a_i are exact rationals (a double's
 * exact value or a Fraction), fixed by the state.
 *
 * Exactness. The separation identities and the parameter calculus are exact: d/dphi_i maps (sin, cos) of phi_i to
 * (cos, -sin), so a derivative or an antiderivative (the one in the span: no constant) is an exact permutation and
 * negation of the coefficients; d/dtheta_i of the exponential family leaves E unchanged. The coefficients themselves
 * are transcendental: each is returned as an exact rational enclosure [lo, hi] / (2^scale 10^wscale) (lo, hi exact
 * 128-bit integers) that contains the exact value, and as its outward-rounded double interval.
 *
 * Precision contract (per row, then exact sums):
 *   reduction  t = x_ri omega_i (2/pi) (PHASE) or sum_i a_i x_ri / ln 2 (EXP) is computed as X K with X the exact
 *              logical value and K a 64.192 fixed-point constant computed natively from the exact rational with a
 *              512-bit 2/pi or 1/ln 2 (|K - K_exact| <= 2^-191); the integer part of t exactly, its fraction to 128
 *              bits (error < 2^-127); K must be below 2^64 (else TR_MATH_UNSUPPORTED).
 *   series     sin, cos on [0, pi/2] (44 Taylor terms) and 2^f on [0, 1) (exp of f ln 2, 40 terms) in 120-bit fixed
 *              point, every truncation and floor bounded (4 ulps per term, the series tail), plus the reduction
 *              error: a radius of at most 2^-112 per sin / cos value and a relative radius of at most 2^-112 per
 *              exponential (midpoint-radius, 120 fractional bits)
 *   products   the 2^k products of the factors in midpoint-radius arithmetic (each product's radius adds the
 *              factors' radii, rigorously)
 *   sums       each row's enclosure times w_r is turned outward into integers floor(lo 2^s), ceil(hi 2^s) and summed
 *              exactly in 128 bits; the scale s is the largest the host can prove keeps every sum below 2^126 in
 *              magnitude (from the scope's column extremes and the row count), capped at 120 (TR_CALC_F)
 * So every coefficient enclosure contains the exact value; its width is at most rows * (|w|max 2^-108 + 2^(1-s))
 * (k <= 16 factors: a product's radius stays below 2^11 ulps of 2^-120).
 * Evaluations at double points use certified sin, cos and exp of the doubles (|phi| < 2^62, else
 * TR_MATH_UNSUPPORTED; exp overflows to [DBL_MAX, inf] and underflows to [0, 2^-1074]) and outward-rounded interval
 * arithmetic (math_carriers.h): every evaluated value is an interval that contains the exact result.
 *
 * Identity. A state records the exact family, the factors' columns and frequencies / rates, the weight column and
 * the scope's exact descriptor (tr_scope_key_*). tr_calc_require decides whether the state answers a request: 0 (the
 * same identity), TR_MATH_DOMAIN_ERROR (another frequency, rate, column, weight, family or scope: a new construction
 * is needed), TR_MATH_UNSUPPORTED (outside the retained basis: a derivative in a frequency or a rate needs sums
 * weighted by x, which no state of these families retains).
 *
 * CUDA (math_calc_gpu.cu, tr_mgpu_calc_sums): the coefficient sums over the rows, with exactly the CPU's per-row
 * integers (the same inline functions below) summed with exact 128-bit atomics, so the device sums equal the CPU's
 * bit for bit whatever the thread order.
 *
 * Budgets (math_budgets.h, before allocating): variables = k, support = coefficients (2^k or 1), states = groups x
 * coefficients, tensor = selected rows x coefficients (the per-row expansion), bytes = the state and the sums (CUDA:
 * the device copies, tr_calc_device_bytes); degree = |order| of a derivative request; evaluations: output = points,
 * tensor = points x coefficients. Refusal TR_MATH_BUDGET with the tr_math_refusal. More than TR_CALC_FACTORS_MAX
 * factors is TR_MATH_UNSUPPORTED; an exponent bound past TR_CALC_NMAX is TR_MATH_OVERFLOW.
 *
 * Errors: TR_MATH_INVALID, TR_MATH_MEMORY, TR_MATH_OVERFLOW, TR_MATH_BUDGET, TR_MATH_DOMAIN_ERROR, TR_MATH_UNSUPPORTED,
 * TR_MATH_STALE (the plan's scope is not the one given). Objects are immutable and individually owned. */
#ifndef TORBLES_MATH_CALC_H
#define TORBLES_MATH_CALC_H
#include <stdint.h>
#include <stddef.h>
#include "math_num.h"
#include "math_budgets.h"
#include "math_carriers.h"
#include "math_scope.h"
#ifdef __cplusplus
extern "C" {
#endif

#define TR_CALC_ABI 1
#define TR_CALC_FACTORS_MAX 16
#define TR_CALC_F 120                 /* fractional bits of the fixed point */
#define TR_CALC_SIN_TERMS 44
#define TR_CALC_EXP_TERMS 40
#define TR_CALC_NMAX (INT64_C(1) << 24)   /* the largest binary exponent of an exponential row value */
#define TR_CALC_SCALE_MAX 120

enum { TR_CALC_PHASE = 1, TR_CALC_EXP = 2 };

/* a family over a scope: factor i reads scope column column[i]; weight: a scope column or -1 (every weight 1);
 * domega[i]: the requested derivative order in factor i's frequency / rate (0: the retained basis) */
typedef struct {
    int32_t family, nfactors, weight, reserved;
    int32_t column[TR_CALC_FACTORS_MAX];
    int32_t domega[TR_CALC_FACTORS_MAX];
} tr_calc_spec;                       /* 144 bytes */

/* the plan of a coefficient build, as the CPU loop and the CUDA kernels read it */
typedef struct {
    int32_t family, nfactors, weight, scale;   /* scale s: row values enter as floor / ceil(value * 2^s) */
    uint64_t nsubsets, ngroups;               /* coefficients per group (2^k or 1), groups */
    int64_t nmax;                             /* EXP: the largest binary exponent of a row value (proven) */
    int32_t ksign[TR_CALC_FACTORS_MAX];       /* the sign of K_i (0: K_i = 0) */
    int32_t column[TR_CALC_FACTORS_MAX];
    uint64_t k[TR_CALC_FACTORS_MAX][4];       /* |K_i| = (k3 2^192 + k2 2^128 + k1 2^64 + k0) / 2^192 */
} tr_calc_plan_view;                          /* 680 bytes */

typedef struct {
    int32_t family, nparams;     /* nparams: the parameters of this state (after definite integrals) */
    int32_t nfactors, exact;     /* factors of the construction; exact: coefficients carry exact scaled bounds */
    int32_t scale, wscale;       /* the bounds are lo / (2^scale 10^wscale) .. hi / (2^scale 10^wscale) */
    int32_t weight, reserved;
    uint64_t ngroups, ncoef;
    int32_t order[TR_CALC_FACTORS_MAX];   /* the net derivative order applied to each remaining parameter */
    int32_t param[TR_CALC_FACTORS_MAX];   /* the factor each remaining parameter belongs to */
} tr_calc_info;                           /* 176 bytes */

typedef struct tr_calc_plan_s tr_calc_plan;
typedef struct tr_calc_state_s tr_calc_state;

/* the plan of spec over scope with omega[i] (PHASE: the frequency of factor i; EXP: its rate), budgeted */
tr_calc_plan *tr_calc_plan_create(const tr_scope *, const tr_calc_spec *, const tr_bigq *const *omega,
                                  const tr_math_budget *, tr_math_refusal *, int32_t *error);
int32_t tr_calc_plan_get_view(const tr_calc_plan *, tr_calc_plan_view *);
/* the exact sums of a plan over the scope (the CPU reference of tr_mgpu_calc_sums): sums[2 (g nsubsets + S)] = the
 * sum of the floors, [+1] the sum of the ceilings; rows[g] selected rows, valid[g] rows that count */
int32_t tr_calc_sums(const tr_calc_plan_view *, const tr_scope_view *, tr_math_i128 *sums, uint64_t *rows,
                     uint64_t *valid);
/* the state from sums (computed by tr_calc_sums or the CUDA kernels): consistency-checked, budgeted */
tr_calc_state *tr_calc_from_sums(const tr_calc_plan *, const tr_scope *, const tr_math_i128 *sums, const uint64_t *rows,
                                 const uint64_t *valid, const tr_math_budget *, tr_math_refusal *, int32_t *error);
/* plan + tr_calc_sums + tr_calc_from_sums on the CPU */
tr_calc_state *tr_calc_build(const tr_scope *, const tr_calc_spec *, const tr_bigq *const *omega,
                             const tr_math_budget *, tr_math_refusal *, int32_t *error);
void tr_calc_plan_free(tr_calc_plan *);

int32_t tr_calc_get_info(const tr_calc_state *, tr_calc_info *);
int32_t tr_calc_group(const tr_calc_state *, uint64_t g, uint64_t *rows, uint64_t *valid);
/* coefficient S of group g: its outward-rounded double interval, and (exact states) its exact scaled bounds */
int32_t tr_calc_coefficient(const tr_calc_state *, uint64_t g, uint64_t S, tr_math_interval *out);
int32_t tr_calc_coefficient_exact(const tr_calc_state *, uint64_t g, uint64_t S, tr_math_i128 *lo, tr_math_i128 *hi);
/* the derivative of order `order` in parameter `param` (order < 0: |order| antiderivatives), an exact transform */
tr_calc_state *tr_calc_derive(const tr_calc_state *, int32_t param, int32_t order, const tr_math_budget *,
                              tr_math_refusal *, int32_t *error);
/* the definite integral over parameter param from a to b: a state with one parameter less (interval coefficients) */
tr_calc_state *tr_calc_integrate(const tr_calc_state *, int32_t param, double a, double b, const tr_math_budget *,
                                 tr_math_refusal *, int32_t *error);
/* values at n points (n x nparams doubles, row-major; nparams == 0: the constant), enclosures */
int32_t tr_calc_evaluate(const tr_calc_state *, uint64_t g, uint64_t n, const double *points, tr_math_interval *out,
                         const tr_math_budget *, tr_math_refusal *);
/* definite integrals over param from a[j] to b[j], the other parameters at rest[j * (nparams - 1) ..] (NULL when
 * nparams == 1), n pairs */
int32_t tr_calc_definite(const tr_calc_state *, uint64_t g, int32_t param, uint64_t n, const double *a, const double *b,
                         const double *rest, tr_math_interval *out, const tr_math_budget *, tr_math_refusal *);
/* whether the state answers spec / omega over scope: 0, TR_MATH_DOMAIN_ERROR (a new construction is needed) or
 * TR_MATH_UNSUPPORTED (outside the retained basis) */
int32_t tr_calc_require(const tr_calc_state *, const tr_scope *, const tr_calc_spec *, const tr_bigq *const *omega);
void tr_calc_free(tr_calc_state *);

/* certified elementary functions (the evaluators of the states) */
int32_t tr_calc_sincos(double x, tr_math_interval *s, tr_math_interval *c);
int32_t tr_calc_exp(double x, tr_math_interval *out);
/* the per-row factor values of a plan with one frequency / rate: sin and cos of omega X / 10^scale, or
 * exp(a X / 10^scale), enclosed (the very functions the sums use) */
int32_t tr_calc_sincos_scaled(const tr_bigq *omega, int64_t X, int32_t scale, tr_math_interval *s, tr_math_interval *c);
int32_t tr_calc_exp_scaled(const tr_bigq *a, int64_t X, int32_t scale, tr_math_interval *out);

/* bytes the CUDA build copies to and allocates on the device: the scope columns it reads, the selection and the
 * group ids, the per-row factors (PHASE: rows x k x 48 + rows), the sums and the counts (saturating) */
static inline uint64_t tr_calc_device_bytes(const tr_calc_plan_view *p, const tr_scope_view *v) {
    uint64_t n = v->rows, b = tr_budget_bitmap_bytes(n);
    if (v->groups) b = tr_budget_add(b, tr_budget_mul(n, 4));
    uint32_t used = 0;
    for (int32_t i = 0; i < p->nfactors; i++) used |= (uint32_t)1 << p->column[i];
    if (p->weight >= 0) used |= (uint32_t)1 << p->weight;
    for (int32_t c = 0; c < v->ncols; c++) {
        if (!((used >> c) & 1)) continue;
        b = tr_budget_add(b, tr_budget_mul(n, v->col[c].values ? 8 : 4));
        if (v->col[c].valid) b = tr_budget_add(b, n);
    }
    if (p->family == TR_CALC_PHASE)
        b = tr_budget_add(b, tr_budget_add(tr_budget_mul(tr_budget_mul(n, (uint64_t)p->nfactors), 48), n));
    b = tr_budget_add(b, tr_budget_mul(tr_budget_mul(p->ngroups, p->nsubsets), 32));
    return tr_budget_add(b, tr_budget_mul(p->ngroups, 16));
}
uint64_t tr_calc_device_need(const tr_calc_plan_view *, const tr_scope_view *);
/* a plan view consistent with a scope view (host code of both libraries) */
static inline int tr_calc_plan_view_valid(const tr_calc_plan_view *pl, const tr_scope_view *v) {
    if (!pl || !v || (pl->family != TR_CALC_PHASE && pl->family != TR_CALC_EXP) || pl->nfactors < 1 ||
        pl->nfactors > TR_CALC_FACTORS_MAX || pl->ngroups != v->ngroups || pl->weight < -1 || pl->weight >= v->ncols ||
        pl->nsubsets != (pl->family == TR_CALC_PHASE ? ((uint64_t)1 << pl->nfactors) : 1) ||
        (v->rows && !v->selection) || (v->rows && v->ngroups > 1 && !v->groups)) return 0;
    for (int32_t i = 0; i < pl->nfactors; i++) if (pl->column[i] < 0 || pl->column[i] >= v->ncols) return 0;
    return 1;
}

/* ================================================================ shared fixed-point functions (CPU and CUDA) */
/* a value in [(c - r) / 2^F, (c + r) / 2^F] */
typedef struct { __int128 c; uint64_t r; } tr_calc_fx;
typedef struct { uint64_t lo, hi, r; } tr_calc_fx_words;      /* its storage in device arrays */

/* floor(pi / 2 * 2^120) and floor(ln 2 * 2^120) */
#define TR_CALC_PIO2 ((((unsigned __int128)0x01921fb54442d184ULL) << 64) | (unsigned __int128)0x69898cc51701b839ULL)
#define TR_CALC_LN2 ((((unsigned __int128)0x00b17217f7d1cf79ULL) << 64) | (unsigned __int128)0xabc9e3b39803f2f6ULL)

/* the top 128 bits of a * b (and the low 128 in *lo when lo != NULL) */
TR_NUM_HD unsigned __int128 tr_calc_mulhi(unsigned __int128 a, unsigned __int128 b, unsigned __int128 *lo) {
    uint64_t a0 = (uint64_t)a, a1 = (uint64_t)(a >> 64), b0 = (uint64_t)b, b1 = (uint64_t)(b >> 64);
    unsigned __int128 p00 = (unsigned __int128)a0 * b0, p01 = (unsigned __int128)a0 * b1;
    unsigned __int128 p10 = (unsigned __int128)a1 * b0, p11 = (unsigned __int128)a1 * b1;
    unsigned __int128 mid = (p00 >> 64) + (uint64_t)p01 + (uint64_t)p10;
    if (lo) *lo = (mid << 64) | (uint64_t)p00;
    return p11 + (p01 >> 64) + (p10 >> 64) + (mid >> 64);
}
/* floor(a b / 2^120) for a, b < 2^124 */
TR_NUM_HD unsigned __int128 tr_calc_fmul(unsigned __int128 a, unsigned __int128 b) {
    unsigned __int128 lo;
    unsigned __int128 hi = tr_calc_mulhi(a, b, &lo);
    return (hi << 8) | (lo >> 120);
}
/* floor(x / k) for 1 <= k < 2^32, by 32-bit digits (no 128-bit division) */
TR_NUM_HD unsigned __int128 tr_calc_divsmall(unsigned __int128 x, uint32_t k) {
    uint64_t hi = (uint64_t)(x >> 64), lo = (uint64_t)x;
    uint64_t qh = hi / k, r = hi % k;
    uint64_t m = (r << 32) | (lo >> 32), q1 = m / k;
    r = m % k;
    uint64_t l = (r << 32) | (lo & 0xffffffffULL), q0 = l / k;
    return ((unsigned __int128)qh << 64) | (unsigned __int128)((q1 << 32) | q0);
}
/* sin r and cos r for r = R / 2^120 in [0, pi/2]: alternating Taylor sums of t_k = floor(floor(t_{k-1} R / 2^F) / k)
 * (each t_k is within 4 ulps below r^k / k!); the radius covers every term's error and the tail */
TR_NUM_HD void tr_calc_sincos_frac(unsigned __int128 R, __int128 *s, __int128 *c, uint64_t *rad) {
    if (R == 0) { *s = 0; *c = (__int128)1 << TR_CALC_F; *rad = 0; return; }
    unsigned __int128 t = (unsigned __int128)1 << TR_CALC_F;
    __int128 S = 0, Cc = (__int128)t;
    for (uint32_t k = 1; k <= TR_CALC_SIN_TERMS; k++) {
        t = tr_calc_divsmall(tr_calc_fmul(t, R), k);
        __int128 v = (__int128)t;
        switch (k & 3) { case 1: S += v; break; case 2: Cc -= v; break; case 3: S -= v; break; default: Cc += v; break; }
    }
    unsigned __int128 tail = tr_calc_divsmall(tr_calc_fmul(t, R), TR_CALC_SIN_TERMS + 1);
    *s = S; *c = Cc;
    *rad = 4 * (TR_CALC_SIN_TERMS / 2) + (uint64_t)tail + 5;
}
/* exp(r) for r = R / 2^120 in [0, ln 2): positive Taylor terms (each within 4 ulps below), the tail <= 2 t_{N+1} */
TR_NUM_HD void tr_calc_exp_frac(unsigned __int128 R, __int128 *e, uint64_t *rad) {
    unsigned __int128 t = (unsigned __int128)1 << TR_CALC_F;
    if (R == 0) { *e = (__int128)t; *rad = 0; return; }
    __int128 E = (__int128)t;
    for (uint32_t k = 1; k <= TR_CALC_EXP_TERMS; k++) {
        t = tr_calc_divsmall(tr_calc_fmul(t, R), k);
        E += (__int128)t;
    }
    unsigned __int128 tail = tr_calc_divsmall(tr_calc_fmul(t, R), TR_CALC_EXP_TERMS + 1);
    *e = E;
    *rad = 4 * TR_CALC_EXP_TERMS + 2 * ((uint64_t)tail + 4) + 1;
}

/* p[0..4] = x * K (K: 4 limbs) */
TR_NUM_HD void tr_calc_mulk(uint64_t x, const uint64_t *k, uint64_t *p) {
    unsigned __int128 carry = 0;
    for (int i = 0; i < 4; i++) {
        unsigned __int128 t = (unsigned __int128)x * k[i] + carry;
        p[i] = (uint64_t)t;
        carry = t >> 64;
    }
    p[4] = (uint64_t)carry;
}
/* bits pos .. pos + 63 of the n-limb number w (zeros outside it; pos may be negative) */
TR_NUM_HD uint64_t tr_calc_bits64(const uint64_t *w, int32_t n, int32_t pos) {
    int32_t q = pos >= 0 ? pos / 64 : -((-pos + 63) / 64), r = pos - 64 * q;
    uint64_t lo = (q >= 0 && q < n) ? w[q] : 0, hi = (q + 1 >= 0 && q + 1 < n) ? w[q + 1] : 0;
    return r ? (lo >> r) | (hi << (64 - r)) : lo;
}

/* sin and cos of t pi / 2 for t = sign * |X| K * 2^sh (|X| 2^sh <= 2^64: the error of t stays below 2^-127):
 * the quadrant from the exact integer part, the fraction to 128 bits, r = f pi / 2, Taylor on [0, pi/2] */
TR_NUM_HD void tr_calc_sincos_reduce(const uint64_t *k, int32_t ksign, int64_t X, int32_t sh, tr_calc_fx *sn,
                                     tr_calc_fx *cs) {
    if (X == 0 || ksign == 0) { sn->c = 0; sn->r = 0; cs->c = (__int128)1 << TR_CALC_F; cs->r = 0; return; }
    uint64_t p[5];
    tr_calc_mulk(tr_abs_u64(X), k, p);
    int32_t bp = 192 - sh;                                       /* the binary point of t in p */
    uint32_t q = (uint32_t)(tr_calc_bits64(p, 5, bp) & 3);
    unsigned __int128 f = ((unsigned __int128)tr_calc_bits64(p, 5, bp - 64) << 64) | tr_calc_bits64(p, 5, bp - 128);
    unsigned __int128 R = tr_calc_mulhi(f, TR_CALC_PIO2, NULL);  /* r = f pi / 2: 2 ulps below at most */
    __int128 S, Cc, sv, cv;
    uint64_t rad;
    tr_calc_sincos_frac(R, &S, &Cc, &rad);
    rad += 3;                                                    /* r (2 ulps) and t (below 2^-127 * pi / 2) */
    switch (q) {
    case 0: sv = S; cv = Cc; break;
    case 1: sv = Cc; cv = -S; break;
    case 2: sv = -S; cv = -Cc; break;
    default: sv = -Cc; cv = S; break;
    }
    if ((X < 0) != (ksign < 0)) sv = -sv;                         /* sin(-t) = -sin(t), cos(-t) = cos(t) */
    sn->c = sv; sn->r = rad; cs->c = cv; cs->r = rad;
}

/* 2^t for t = sum_i sign_i |X_i| K_i (the EXP reduction): *n = floor(t) and *m = 2^frac(t) in [1, 2]; 0 when |floor t|
 * reaches TR_CALC_NMAX (the host's bound excludes it) */
TR_NUM_HD int tr_calc_exp2_reduce(const tr_calc_plan_view *pl, const int64_t *X, int64_t *n, tr_calc_fx *m) {
    uint64_t acc[6] = {0, 0, 0, 0, 0, 0};
    int any = 0;
    for (int32_t i = 0; i < pl->nfactors; i++) {
        if (X[i] == 0 || pl->ksign[i] == 0) continue;
        uint64_t p[6];
        tr_calc_mulk(tr_abs_u64(X[i]), pl->k[i], p);
        p[5] = 0;
        if ((X[i] < 0) != (pl->ksign[i] < 0)) {                   /* two's complement negation */
            unsigned c = 1;
            for (int j = 0; j < 6; j++) { uint64_t v = ~p[j] + c; c = c && v == 0; p[j] = v; }
        }
        unsigned __int128 carry = 0;
        for (int j = 0; j < 6; j++) { carry += (unsigned __int128)acc[j] + p[j]; acc[j] = (uint64_t)carry; carry >>= 64; }
        any = 1;
    }
    if (!any) { *n = 0; m->c = (__int128)1 << TR_CALC_F; m->r = 0; return 1; }
    int neg = (int)(acc[5] >> 63);
    if (neg) { unsigned c = 1; for (int j = 0; j < 6; j++) { uint64_t v = ~acc[j] + c; c = c && v == 0; acc[j] = v; } }
    uint64_t N = acc[3], high = acc[4] | acc[5];
    if (high || N >= (uint64_t)TR_CALC_NMAX) return 0;
    unsigned __int128 F = ((unsigned __int128)acc[2] << 64) | acc[1];
    int64_t e = (int64_t)N;
    if (neg) { if (F) { e = -e - 1; F = (unsigned __int128)0 - F; } else e = -e; }
    unsigned __int128 R = tr_calc_mulhi(F, TR_CALC_LN2, NULL);   /* r = f ln 2 < ln 2 */
    __int128 E;
    uint64_t rad;
    tr_calc_exp_frac(R, &E, &rad);
    if (R) rad += 5;                                             /* r (2 ulps, exp' < 2) and t (below 2^-127) */
    *n = e; m->c = E; m->r = rad;
    return 1;
}

/* the midpoint-radius product (each factor's magnitude below 2^122, radii below 2^40): with ia = ceil(|a.c| / 2^112)
 * (so |a.c| / 2^F <= ia / 256), the error |a.c| eb + |b.c| ea + ea eb in ulps is at most
 * ceil((ia b.r + ib a.r) / 256) + ceil(a.r b.r / 2^F), plus one ulp for the truncated center */
TR_NUM_HD tr_calc_fx tr_calc_fx_mul(tr_calc_fx a, tr_calc_fx b) {
    unsigned __int128 ma = a.c < 0 ? (unsigned __int128)0 - (unsigned __int128)a.c : (unsigned __int128)a.c;
    unsigned __int128 mb = b.c < 0 ? (unsigned __int128)0 - (unsigned __int128)b.c : (unsigned __int128)b.c;
    unsigned __int128 mc = tr_calc_fmul(ma, mb);
    const unsigned __int128 unit = ((unsigned __int128)1 << (TR_CALC_F - 8)) - 1;
    uint64_t ia = (uint64_t)((ma + unit) >> (TR_CALC_F - 8)), ib = (uint64_t)((mb + unit) >> (TR_CALC_F - 8));
    tr_calc_fx out;
    out.c = ((a.c < 0) != (b.c < 0)) ? -(__int128)mc : (__int128)mc;
    out.r = ((ia * b.r + ib * a.r + 255) >> 8) + (uint64_t)(((unsigned __int128)a.r * b.r) >> TR_CALC_F) + 2;
    return out;
}
TR_NUM_HD tr_calc_fx tr_calc_fx_neg(tr_calc_fx a) { a.c = -a.c; return a; }

/* floor (up = 0) or ceil (up = 1) of sign * mag * 2^-k (mag: 3 limbs), as an int128; 0 when it does not fit */
TR_NUM_HD int tr_calc_round_shift(int neg, const uint64_t *mag, int32_t k, int up, __int128 *out) {
    unsigned __int128 q;
    int rem = 0;
    if (k <= 0) {                                                /* exact left shift by -k: fits 127 bits or 0 */
        int32_t s = -k;
        unsigned __int128 v = ((unsigned __int128)mag[1] << 64) | mag[0];
        if (mag[2] || s >= 127 || (s && (v >> (127 - s)))) return 0;
        q = v << s;
    } else {
        uint64_t w0 = tr_calc_bits64(mag, 3, k), w1 = tr_calc_bits64(mag, 3, k + 64), w2 = tr_calc_bits64(mag, 3, k + 128);
        if (w2) return 0;
        q = ((unsigned __int128)w1 << 64) | w0;
        for (int32_t j = 0; j < 3 && !rem; j++) {                /* any bit below position k */
            int32_t lo = 64 * j, hi = lo + 64;
            if (lo >= k) break;
            uint64_t word = mag[j];
            if (hi > k) word &= (((uint64_t)1 << (k - lo)) - 1);
            rem = word != 0;
        }
    }
    if ((up && !neg) || (!up && neg)) q += rem;                  /* away from zero toward the rounding direction */
    if (q > (((unsigned __int128)1 << 127) - 1)) return 0;
    *out = neg ? -(__int128)q : (__int128)q;
    return 1;
}
/* [L, H] containing w * v * 2^(e - F) for v = (c +- r) / 2^F: 1, or 0 when a bound does not fit int128 */
TR_NUM_HD int tr_calc_scaled(int64_t w, tr_calc_fx v, int32_t e, __int128 *L, __int128 *H) {
    __int128 lo = v.c - (__int128)v.r, hi = v.c + (__int128)v.r;
    __int128 a = w >= 0 ? lo : hi, b = w >= 0 ? hi : lo;          /* w a <= w b */
    uint64_t aw = tr_abs_u64(w);
    int32_t k = TR_CALC_F - e;
    __int128 x[2] = {a, b};
    __int128 res[2];
    for (int j = 0; j < 2; j++) {
        int neg = (x[j] < 0) != (w < 0) && x[j] != 0 && w != 0;
        unsigned __int128 m = x[j] < 0 ? (unsigned __int128)0 - (unsigned __int128)x[j] : (unsigned __int128)x[j];
        uint64_t m0 = (uint64_t)m, m1 = (uint64_t)(m >> 64);
        unsigned __int128 p0 = (unsigned __int128)m0 * aw, p1 = (unsigned __int128)m1 * aw + (p0 >> 64);
        uint64_t mag[3] = {(uint64_t)p0, (uint64_t)p1, (uint64_t)(p1 >> 64)};
        if (!tr_calc_round_shift(neg, mag, k, j, &res[j])) return 0;
    }
    *L = res[0]; *H = res[1];
    return 1;
}

/* the values of one row: PHASE factors (sin, cos of theta_ri), its weight and whether it counts */
TR_NUM_HD int tr_calc_row_ok(const tr_calc_plan_view *pl, const tr_scope_view *v, uint64_t r) {
    if (!tr_scope_is_selected(v, r)) return 0;
    for (int32_t i = 0; i < pl->nfactors; i++) if (!tr_scope_present(v, pl->column[i], r)) return 0;
    return pl->weight < 0 || tr_scope_present(v, pl->weight, r);
}
TR_NUM_HD int64_t tr_calc_row_weight(const tr_calc_plan_view *pl, const tr_scope_view *v, uint64_t r) {
    return pl->weight < 0 ? 1 : tr_scope_value(v, pl->weight, r);
}
/* the PHASE product of subset S: factor i contributes cos(theta_i) when bit i of S is set, else sin(theta_i);
 * fac[2 i] = sin, fac[2 i + 1] = cos; the factors are multiplied in order i = 0 .. k - 1 */
TR_NUM_HD tr_calc_fx tr_calc_subset(const tr_calc_fx *fac, int32_t k, uint64_t S) {
    tr_calc_fx p = fac[(S & 1) ? 1 : 0];
    for (int32_t i = 1; i < k; i++) p = tr_calc_fx_mul(p, fac[2 * i + ((S >> i) & 1)]);
    return p;
}
TR_NUM_HD void tr_calc_fx_store(tr_calc_fx_words *w, tr_calc_fx x) {
    w->lo = (uint64_t)(unsigned __int128)x.c; w->hi = (uint64_t)((unsigned __int128)x.c >> 64); w->r = x.r;
}
TR_NUM_HD tr_calc_fx tr_calc_fx_load(const tr_calc_fx_words *w) {
    tr_calc_fx x;
    x.c = (__int128)(((unsigned __int128)w->hi << 64) | w->lo);
    x.r = w->r;
    return x;
}
/* the EXP contribution [L, H] of row r (counted rows only); 0 on an exponent or int128 overflow */
TR_NUM_HD int tr_calc_exp_row(const tr_calc_plan_view *pl, const tr_scope_view *v, uint64_t r, __int128 *L, __int128 *H) {
    int64_t X[TR_CALC_FACTORS_MAX];
    for (int32_t i = 0; i < pl->nfactors; i++) X[i] = tr_scope_value(v, pl->column[i], r);
    int64_t n;
    tr_calc_fx m;
    if (!tr_calc_exp2_reduce(pl, X, &n, &m)) return 0;
    int64_t e = n + pl->scale;
    if (e < -100000) e = -100000;                                /* far below the scale: the bounds are 0 / +-1 */
    return tr_calc_scaled(tr_calc_row_weight(pl, v, r), m, (int32_t)e, L, H);
}

#ifdef __cplusplus
}
#endif
#endif
