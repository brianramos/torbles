/* KERNEL-002: approximate spectral feature kernels. Selected kernels without an exact finite feature map are offered
 * through explicit, declared finite-feature approximations whose every answer carries a UNC-002-style approximation
 * record; exact polynomial kernels stay KERNEL-001's (math_kernel.h), answered exactly there.
 * SPDX-License-Identifier: Apache-2.0
 *
 * ---- Kernels (values x = logical / 10^scale of d scope columns sharing one decimal scale; gamma > 0 exact) ----------
 *   GAUSSIAN  k(x, y) = exp(-gamma |x - y|^2)                 spectral density N(0, 2 gamma I)
 *   LAPLACE   k(x, y) = exp(-gamma |x - y|_1)                 product of Cauchy densities of scale gamma
 *   CAUCHY    k(x, y) = prod_j 1 / (1 + gamma^2 (x_j - y_j)^2)  product of Laplace densities of scale gamma
 * k(x, x) = 1 for all three. Weights w (a scope column, value w / 10^wscale; 1 without one) multiply each row.
 *
 * ---- TAYLOR: deterministic features with a certified bound (GAUSSIAN only) ---------------------------------------
 * exp(-gamma |x - y|^2) = E(x) E(y) exp(s), E(x) = exp(-gamma |x|^2), s = 2 gamma x.y, and exp(s) is truncated at the
 * declared degree N (<= TR_KAPPROX_MAX_DEGREE): k_N(x, y) = E(x) E(y) sum_{n <= N} s^n / n!, a finite-feature kernel
 * <phi(x), phi(y)> over the C(d + N, N) multi-indices a, |a| <= N (KERNEL-001's sorted-multi-index evaluation with
 * multiplicities |a|! / a!): with u = X / 2^b (X the logical values, 2^b >= every |X_j| and the domain's radius),
 *     K_N(g, h) = sum_k V^k / k! sum_{|a| = k} (k! / a!) Phi_g[a] Phi_h[a],  Phi_g[a] = sum_{i in g} w_i E(x_i) u_i^a,
 * V = 2 gamma 4^b / 10^(2 scale). Declared domain: the ball |x|^2 <= rho^2 (radius2 declared, else the data's largest
 * |x|^2, exact); every counted row and witness point must lie in it (else TR_MATH_DOMAIN_ERROR). Certified per pair:
 *     |k - k_N| <= eps = e^-U T_N(U) = P(Poisson(U) > N),  U = 2 gamma rho^2  (T_N the exponential tail),
 * since E(x) E(y) <= e^-|s| and u -> e^-u T_N(u) is increasing; eps is bounded above by the exact rational
 * min(1, U^(N+1) / (N+1)! (N+2) / (N+2-U) / sum_{n <= N+32} U^n / n!) (1 when U >= N + 2). Label CERTIFIED.
 *
 * ---- RFF: random Fourier features (all three kernels) ---------------------------------------------------------------
 * k(x - y) = E_omega cos(omega.(x - y)); D frequencies are drawn by the native seeded generator, deduplicated with
 * multiplicities m_f, and the approximant k^(x, y) = (1 / D) sum_f m_f cos(omega_f.(x - y)) is evaluated through
 * SPEC-001's fixed spectral basis (math_fourier.h: F_g(omega) = C + i S per group, certified enclosures accumulated as
 * exact scaled integers, CPU equal to CUDA): K^(g, h) = (1 / D) sum_f m_f (C_g C_h + S_g S_h).
 * Generator (TR_KAPPROX_GENERATOR): 64-bit words w_i = splitmix64(key + i), key = splitmix64(seed ^ 0xD1B54A32D192ED03)
 * (UNC-002's stream convention, stream 0), splitmix64(z) the finalizer of z + 0x9E3779B97F4A7C15. A 128-bit integer
 * N = w_k + 2^64 w_(k+1) gives the uniform U = (2 N + 1) / 2^129 (a cell midpoint).
 *   GAUSSIAN  per pair of coordinates (2q, 2q+1) of frequency f, words 4 (f ceil(d/2) + q) + 0..3: U1, U2;
 *             z = sqrt(-2 ln U1) (cos, sin)(2 pi U2) (Box-Muller), omega = sqrt(2 gamma) z;
 *   LAPLACE   coordinate j: words 2 (f d + j) + 0..1: omega = gamma cot(pi U) (0 when U < 2^-32 or U > 1 - 2^-32);
 *   CAUCHY    coordinate j: words 3 (f d + j) + 0..2: omega = +-gamma (-ln U), the sign from bit 63 of the third word.
 * Every transform is evaluated with certified fixed-point enclosures (ln by atanh, sqrt by integer square roots, sin and
 * cos by CALC-003's series via SPEC-001's exact quarter-turn reduction) and rounded to the dyadic omega~ = M / 2^p, p
 * the state's fbits (|M| < 2^62); the per-sample enclosure half-width is checked against its a priori bound.
 * Error per kernel value, for a fixed pair, under TR_APPROX_ASSUME_INDEPENDENT (the generator's words treated as
 * i.i.d. uniform): with probability >= 1 - delta over the draw,
 *     |k^ - k| <= 2 h + beta,  h = sqrt(ln(2 / delta) / (2 D)) (Hoeffding, bounded above exactly),
 *     beta = 2 P_bad + 2 rho sqrt(d) eta (deterministic: E[min(2, |omega~ - omega| |x - y|)] by the coupling of each
 *            cell with an exact sample of the spectral density; eta bounds |omega~_j - omega_j| on the good cells
 *            (cell width, enclosure, rounding), P_bad the probability of the others; rho as for TAYLOR).
 * Per answer the Hoeffding term is R h (R the range of its per-frequency term, below) with failure probability delta
 * (per answer: a union bound gives the simultaneous statement). Label PROBABILISTIC.
 *
 * ---- Answers (tr_kapprox_query; groups a, b; weights w of a's rows, v of b's; A = sum |w|, W = sum w, Q = sum w^2) -
 *   SUM       sum over the selected pairs of w_i v_l k(x_i, y_l) (KERNEL-001's pair semantics: a != b every pair; a == b
 *             ordered pairs i != l unless SELF, UNORDERED halves; k(x, x) = 1 so the diagonal is Q exactly), value units
 *             (/ 10^(2 wscale)); pairs = the exact weighted pair count. Weight factor F = A_a A_b (A^2 within a group,
 *             halved when unordered), Hoeffding range R = 2 A_a A_b (A^2 within a group).
 *   MEAN      SUM / pairs (empty when pairs = 0): F / |pairs|, R / |pairs|.
 *   MMD2      biased K_aa / W_a^2 + K_bb / W_b^2 - 2 K_ab / (W_a W_b): F = R = (A_a / |W_a| + A_b / |W_b|)^2; UNBIASED
 *             (K_aa - Q_a) / (W_a^2 - Q_a) + (K_bb - Q_b) / (W_b^2 - Q_b) - 2 K_ab / (W_a W_b): F = A_a^2 / |W_a^2 - Q_a|
 *             + A_b^2 / |W_b^2 - Q_b| + 2 A_a A_b / |W_a W_b|, R the same with 4 A_a A_b (empty when a denominator is 0).
 *   WITNESS   mu_a(z) = sum_i w_i k(x_i, z) / W_a at a point (logical values; inside the domain): F = A_a / |W_a|,
 *             R = 2 A_a / |W_a|.
 * Each answer: [lo, hi], the exact rational enclosure of the approximant's value (from the certified feature sums);
 * the error terms (TAYLOR: TR_APPROX_TERM_REMAINDER eps F; RFF: TR_APPROX_TERM_PROBABILISTIC h R and
 * TR_APPROX_TERM_QUANTIZATION beta F) and their exact sum; the bound [lo - error, hi + error] for the exact kernel
 * quantity (certified, or with probability >= 1 - failure); norm ABS; the label. The approximant is never reported as
 * the exact value: tr_kapprox_authorize_exact always refuses (FND-007 reason APPROXIMATE, STATISTICAL for RFF).
 * A requested error budget below the per-pair bound (eps, or 2 h + beta) refuses the state (TR_MATH_NOT_CONVERGED).
 *
 * ---- Exactness of the feature sums (per row, then exact 128-bit sums; CPU and CUDA share the inline code below) ------
 * TAYLOR: t = gamma |x|^2 / ln 2 = (sum_j X_j^2) Kq with the exact 192-bit sum of squares and Kq = gamma / (10^(2 scale)
 * ln 2) in 64.320 fixed point rounded down (computed from floor(2^576 / ln 2): Kq's error is below 2^-319, so t's error,
 * fraction truncation included, stays below 2^-127);
 * E = 2^-t = 2^e m with m = 2^(1 - frac(t)) from CALC-003's certified 120-bit exp (math_calc.h: tr_calc_exp_frac,
 * radius + 8 ulps for the reduction); u_j = X_j 2^(120 - b) exactly; each monomial m u^a in CALC-003's midpoint-radius
 * products (the coordinates in non-decreasing order); w E u^a 2^S rounded outward to integers (tr_calc_scaled) and
 * summed exactly, S = min(120, 124 - bits(selected x max |w|)) (no sum leaves 2^126). RFF: SPEC-001's sums exactly.
 * Answers combine the interval sums exactly (midpoint-radius in 384-bit integers per coefficient class with KERNEL-001's
 * accumulator, then FND-009 big rationals): [lo, hi] contains the approximant's exact value.
 *
 * ---- Persistence (FND-011's record format, math_persist.h) ----------------------------------------------------------
 * tr_kapprox_save writes a version 1 record: FND-011's header (magic, format, endianness marker, type tag
 * TR_KAPPROX_TYPE, which FND-011's typed loader does not know and refuses, carrier TR_CARRIER_INTERVAL: enclosures,
 * exact 0), the FND-007 source record of the scope (when scoped), the basis (the spec: kernel, method, gamma, degree,
 * radius, budget, seed, count, failure; the derived constants and the frequencies with their multiplicities and SPEC-001
 * reduction constants), the caller's metadata and the payload (group statistics, feature sums), FND-011's two-lane
 * checksum. tr_kapprox_load checks in FND-011's order (tr_persist_header_read, the checksum, the type, the type
 * version, the sections) and rebuilds the identical state (the same answers, the same bytes when saved again); with
 * an origin scope it requires the recorded source to be fresh (FND-007 exact comparison, else TR_MATH_STALE).
 *
 * ---- Budgets (math_budgets.h, before allocation) --------------------------------------------------------------------
 *   plan / build: variables = d (+ weight), degree = N, support = TAYLOR items / RFF distinct frequencies, states =
 *     groups x items, tensor = selected rows x items (the per-row expansion), iterations = D d (RFF draws), bytes = the
 *     plan, the sums and the state (SPEC-001's own needs for RFF); CUDA: the device copies (tr_kapprox_device_bytes)
 *   query: output = requests, tensor = points x items (point features), bytes = the point features; every big rational
 *     an answer creates is FND-009-budgeted (bytes / 8 limbs)
 *   save / load: bytes = the record (twice for a load)
 * Representation limits (TR_MATH_UNSUPPORTED): d > 16, N > 12, D > 2^20, TAYLOR for LAPLACE / CAUCHY, a bandwidth whose
 * frequencies leave 2^61 or whose Gaussian factor underflows past 2^-(2^24), 2^b past 2^120.
 * Errors: TR_MATH_INVALID, _MEMORY, _OVERFLOW (a sum outside its proof: never with a valid plan), _BUDGET, _DOMAIN_ERROR
 * (columns of different scales, data or points outside the declared domain, an exact rewrite asked for), _UNSUPPORTED,
 * _STALE, _NOT_CONVERGED. Objects are immutable and individually owned (constructors return NULL and set *error). */
#ifndef TORBLES_MATH_KAPPROX_H
#define TORBLES_MATH_KAPPROX_H
#include <stdint.h>
#include <string.h>
#include "math_num.h"
#include "math_budgets.h"
#include "math_scope.h"
#include "math_carriers.h"
#include "math_calc.h"
#include "math_fourier.h"
#include "math_kernel.h"
#include "math_approx.h"
#include "math_certificates.h"
#include "math_persist.h"
#ifdef __cplusplus
extern "C" {
#endif

#define TR_KAPPROX_ABI 1
#define TR_KAPPROX_MAX_DIMS 16                     /* SPEC-001's */
#define TR_KAPPROX_MAX_DEGREE 12                   /* multinomials |a|! / a! < 2^32 */
#define TR_KAPPROX_MAX_FREQS (UINT64_C(1) << 20)
#define TR_KAPPROX_MAX_TERMS 4                     /* error terms of one answer */
#define TR_KAPPROX_KQ_LIMBS 6                      /* 64.320 fixed point */
#define TR_KAPPROX_NMAX (UINT64_C(1) << 24)        /* the largest binary exponent of exp(-gamma |x|^2) */
#define TR_KAPPROX_TYPE 0x4b02                     /* FND-011 record type tag of a KERNEL-002 state */
#define TR_KAPPROX_TYPE_VERSION 1
#define TR_KAPPROX_GENERATOR 1                     /* splitmix64 words, certified transforms (see above) */
#define TR_KAPPROX_NOROW INT64_MIN                 /* a prefactor slot of a row that does not count */

enum { TR_KAPPROX_GAUSSIAN = 1, TR_KAPPROX_LAPLACE = 2, TR_KAPPROX_CAUCHY = 3 };      /* kernels */
enum { TR_KAPPROX_TAYLOR = 1, TR_KAPPROX_RFF = 2 };                                    /* methods */
enum { TR_KAPPROX_OP_SUM = 1, TR_KAPPROX_OP_MEAN = 2, TR_KAPPROX_OP_MMD2 = 3, TR_KAPPROX_OP_WITNESS = 4 };
enum { TR_KAPPROX_SELF = 1, TR_KAPPROX_UNORDERED = 2, TR_KAPPROX_UNBIASED = 4 };       /* = KERNEL-001's flags */
enum { TR_KAPPROX_NEED_CPU = 0, TR_KAPPROX_NEED_DEVICE = 1 };
/* the exact rationals of a state (tr_kapprox_get_q) */
enum { TR_KAPPROX_Q_PER_PAIR = 1, TR_KAPPROX_Q_HOEFFDING = 2, TR_KAPPROX_Q_BIAS = 3, TR_KAPPROX_Q_RADIUS2 = 4,
       TR_KAPPROX_Q_U = 5, TR_KAPPROX_Q_FAILURE = 6, TR_KAPPROX_Q_BUDGET = 7, TR_KAPPROX_Q_GAMMA = 8,
       TR_KAPPROX_Q_ETA = 9, TR_KAPPROX_Q_PBAD = 10, TR_KAPPROX_Q_COEF = 11, TR_KAPPROX_Q_ETA_NUM = 12 };

/* ------------------------------------------------------------------ structs crossing the ctypes boundary */
typedef struct {
    int32_t kernel, method, ndims, weight;       /* TR_KAPPROX_*; d scope columns; the weight column or -1 */
    int32_t degree, has_radius, has_budget, reserved;   /* TAYLOR N; radius2 / budget given */
    int32_t column[TR_KAPPROX_MAX_DIMS];
    tr_num_rational gamma;                       /* > 0 */
    tr_num_rational radius2;                     /* has_radius: the domain |x|^2 <= radius2 (value units) */
    tr_num_rational failure;                     /* RFF: delta, 0 < delta < 1 */
    tr_num_rational budget;                      /* has_budget: the error budget of the per-pair bound */
    uint64_t nfreq, seed;                        /* RFF: D, the generator's seed */
} tr_kapprox_spec;                               /* 176 bytes */

/* the plan as the CPU loop and the kernels read it */
typedef struct {
    int32_t method, ndims, weight, degree;
    int32_t scale, shift, xscale, reserved;      /* scale: sums are of floor / ceil(value 2^scale); shift b: u = X 2^-b */
    uint64_t ngroups, nitems;                    /* features per group: TAYLOR items, RFF 2 x distinct frequencies */
    uint64_t tmax;                               /* TAYLOR: the proven bound of floor(t) */
    uint64_t kq[TR_KAPPROX_KQ_LIMBS];            /* TAYLOR: Kq 2^320, rounded down */
    int32_t column[TR_KAPPROX_MAX_DIMS];
    const uint8_t *exps;                         /* TAYLOR: nitems x ndims exponents (canonical item order) */
    const int32_t *parent, *axis;                /* TAYLOR, host only: item t = parent[t] times u_axis[t] */
    tr_fourier_plan_view fourier;                /* RFF: SPEC-001's plan view */
} tr_kapprox_plan_view;                          /* 312 bytes */

typedef struct {
    uint64_t rows, valid;                        /* selected rows; counted rows (every column and the weight present) */
    tr_math_i128 W, A;                           /* sum w, sum |w| (logical weight units) */
    uint64_t Q[3];                               /* sum w^2, 192-bit unsigned */
    uint64_t reserved;
} tr_kapprox_group;                              /* 80 bytes */

typedef struct { int32_t op, flags; uint64_t a, b, point; int64_t reserved; } tr_kapprox_request;   /* 40 bytes */

/* an answer; every tr_bigq is owned by the caller (tr_kapprox_result_free, or taken over) */
typedef struct {
    int32_t status, empty, label, nterms;        /* 0 or a native code; no value (a zero denominator); TR_APPROX_LABEL_* */
    int32_t term_kind[TR_KAPPROX_MAX_TERMS];     /* TR_APPROX_TERM_* */
    double lo_d, hi_d, error_d;                  /* RD(lo), RU(hi), RU(error): display only */
    tr_bigq *lo, *hi;                            /* the approximant's exact enclosure */
    tr_bigq *error;                              /* the sum of the terms */
    tr_bigq *bound_lo, *bound_hi;                /* lo - error, hi + error: the exact quantity's bound */
    tr_bigq *failure;                            /* RFF: delta (NULL: certified) */
    tr_bigq *pairs;                              /* SUM / MEAN: the exact weighted pair count (value units) */
    tr_bigq *term[TR_KAPPROX_MAX_TERMS];
} tr_kapprox_result;                             /* 144 bytes */

typedef struct {
    int32_t kernel, method, ndims, weight;
    int32_t degree, scale, wscale, xscale;
    int32_t shift, fbits, label, provenance;     /* fbits: frequency fraction bits p; TR_APPROX_LABEL_*, _DETERMINISTIC / _RANDOMIZED */
    int32_t norm, has_radius, has_budget, scoped;
    uint32_t assumptions, generator;             /* TR_APPROX_ASSUME_*; TR_KAPPROX_GENERATOR (RFF) */
    uint64_t ngroups, nitems, nfreq, count, seed;   /* nfreq distinct frequencies, count = D with multiplicity */
    uint64_t selected, rows, bytes, id;
    uint64_t maxsq[3], tmax;                     /* the largest sum_j X_j^2 of a counted row (192-bit) */
    double per_pair_hi, hoeffding_hi, bias_hi, radius2_hi;   /* RU of the exact rationals */
} tr_kapprox_info;                               /* 208 bytes */

typedef struct tr_kapprox_plan_s tr_kapprox_plan;
typedef struct tr_kapprox_state_s tr_kapprox_state;

/* ================================================================ shared row code (CPU loops and CUDA kernels) */
/* whether row r counts (selected; every column and the weight present) and its weight */
TR_NUM_HD int tr_kapprox_row_ok(const tr_kapprox_plan_view *pv, const tr_scope_view *v, uint64_t r) {
    if (!tr_scope_is_selected(v, r)) return 0;
    for (int32_t j = 0; j < pv->ndims; j++) if (!tr_scope_present(v, pv->column[j], r)) return 0;
    return pv->weight < 0 || tr_scope_present(v, pv->weight, r);
}
TR_NUM_HD int64_t tr_kapprox_row_weight(const tr_kapprox_plan_view *pv, const tr_scope_view *v, uint64_t r) {
    return pv->weight < 0 ? 1 : tr_scope_value(v, pv->weight, r);
}
/* s[3] = sum_j X_j^2 exactly (192-bit unsigned) */
TR_NUM_HD void tr_kapprox_sumsq(const int64_t *X, int32_t d, uint64_t *s) {
    s[0] = s[1] = s[2] = 0;
    for (int32_t j = 0; j < d; j++) {
        uint64_t a = tr_abs_u64(X[j]);
        unsigned __int128 p = (unsigned __int128)a * a;
        unsigned __int128 lo = (unsigned __int128)s[0] + (uint64_t)p;
        s[0] = (uint64_t)lo;
        unsigned __int128 mid = (unsigned __int128)s[1] + (uint64_t)(p >> 64) + (uint64_t)(lo >> 64);
        s[1] = (uint64_t)mid;
        s[2] += (uint64_t)(mid >> 64);
    }
}
/* -1, 0, 1 for 192-bit unsigned a vs b */
TR_NUM_HD int tr_kapprox_cmp192(const uint64_t *a, const uint64_t *b) {
    for (int i = 2; i >= 0; i--) if (a[i] != b[i]) return a[i] < b[i] ? -1 : 1;
    return 0;
}
/* acc[3] += x (192-bit unsigned, x < 2^128) */
TR_NUM_HD void tr_kapprox_add192(uint64_t *acc, unsigned __int128 x) {
    unsigned __int128 lo = (unsigned __int128)acc[0] + (uint64_t)x;
    acc[0] = (uint64_t)lo;
    unsigned __int128 mid = (unsigned __int128)acc[1] + (uint64_t)(x >> 64) + (uint64_t)(lo >> 64);
    acc[1] = (uint64_t)mid;
    acc[2] += (uint64_t)(mid >> 64);
}
/* E = exp(-gamma |x|^2) = 2^e m for the logical values X of a row or point: t = (sum X_j^2) Kq (192 x 384 bits, the
 * fraction kept to 128 bits), e = -floor(t) - 1 and m = 2^(1 - frac t) = exp((1 - frac t) ln 2) by CALC-003's series
 * (radius + 8 ulps: the R truncation, t's error below 2^-127 and Kq rounded down); X = 0 gives E = 1 exactly. 1, or 0
 * when floor(t) exceeds the plan's proven tmax. */
TR_NUM_HD int tr_kapprox_prefactor(const tr_kapprox_plan_view *pv, const int64_t *X, int64_t *e, tr_calc_fx *m) {
    uint64_t s[3];
    tr_kapprox_sumsq(X, pv->ndims, s);
    if (!(s[0] | s[1] | s[2])) { *e = 0; m->c = (__int128)1 << TR_CALC_F; m->r = 0; return 1; }
    uint64_t P[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    for (int i = 0; i < 3; i++) {
        unsigned __int128 carry = 0;
        for (int j = 0; j < TR_KAPPROX_KQ_LIMBS; j++) {
            unsigned __int128 t = (unsigned __int128)s[i] * pv->kq[j] + P[i + j] + carry;
            P[i + j] = (uint64_t)t;
            carry = t >> 64;
        }
        for (int k = i + TR_KAPPROX_KQ_LIMBS; carry && k < 9; k++) {
            unsigned __int128 t = (unsigned __int128)P[k] + carry;
            P[k] = (uint64_t)t;
            carry = t >> 64;
        }
    }
    if (P[6] | P[7] | P[8] || P[5] > pv->tmax) return 0;
    uint64_t n = P[5];
    unsigned __int128 F = ((unsigned __int128)P[4] << 64) | P[3];   /* frac(t) 2^128 (bits below dropped: inexact) */
    if (F == 0) {                                                    /* 2^-t in 2^-n [1 - 2^-127, 1] */
        *e = -(int64_t)n; m->c = (__int128)1 << TR_CALC_F; m->r = 1;
        return 1;
    }
    unsigned __int128 G = (unsigned __int128)0 - F;                 /* (1 - frac t) 2^128 */
    unsigned __int128 R = tr_calc_mulhi(G, TR_CALC_LN2, NULL);      /* (1 - frac t) ln 2 2^120 < ln 2 2^120 */
    __int128 E;
    uint64_t rad;
    tr_calc_exp_frac(R, &E, &rad);
    *e = -(int64_t)n - 1;
    m->c = E; m->r = rad + 8;
    return 1;
}
/* u = X 2^-b exactly (|X| <= 2^b, b <= 120) */
TR_NUM_HD tr_calc_fx tr_kapprox_coord(const tr_kapprox_plan_view *pv, int64_t X) {
    tr_calc_fx u;
    u.c = (__int128)X * ((__int128)1 << (TR_CALC_F - pv->shift));
    u.r = 0;
    return u;
}
/* m u^a of item t: the coordinates multiplied in non-decreasing order (the CPU's memoized chain gives the same bits) */
TR_NUM_HD tr_calc_fx tr_kapprox_monomial(const tr_kapprox_plan_view *pv, const int64_t *X, tr_calc_fx m, uint64_t t) {
    tr_calc_fx v = m;
    const uint8_t *a = pv->exps + t * (uint64_t)pv->ndims;
    for (int32_t j = 0; j < pv->ndims; j++) {
        if (!a[j]) continue;
        tr_calc_fx u = tr_kapprox_coord(pv, X[j]);
        for (uint8_t k = 0; k < a[j]; k++) v = tr_calc_fx_mul(v, u);
    }
    return v;
}
/* [L, H] containing w E u^a 2^S for a row's prefactor (e, value v = m u^a): 1, or 0 when a bound does not fit */
TR_NUM_HD int tr_kapprox_term(int64_t w, int64_t e, tr_calc_fx v, int32_t scale, __int128 *L, __int128 *H) {
    int64_t k = e + scale;
    if (k < -(int64_t)TR_KAPPROX_NMAX - 256) k = -(int64_t)TR_KAPPROX_NMAX - 256;   /* far below: 0 / +-1 */
    return tr_calc_scaled(w, v, (int32_t)k, L, H);
}

/* the row slices of the device counts and feature kernels (rows s, s + slices, ...): about 2^19 / work work items,
 * at least 32 rows each */
static inline uint64_t tr_kapprox_slices(uint64_t rows, uint64_t work) {
    uint64_t s = (UINT64_C(1) << 19) / (work ? work : 1), cap = (rows + 31) / 32;
    if (s > cap) s = cap;
    return s ? s : 1;
}
/* bytes the CUDA build copies to and allocates on the device (saturating): the selection, the group ids, the columns
 * read and their validity, the group statistics (80 a group) and the per-slice maxima (24 a slice); TAYLOR also the
 * per-row prefactors (32 a row), the exponents and the sums (32 a cell); RFF: the larger of these and SPEC-001's */
static inline uint64_t tr_kapprox_device_bytes(const tr_kapprox_plan_view *p, const tr_scope_view *v, uint64_t slices) {
    uint64_t n = v->rows, b = tr_budget_bitmap_bytes(n);
    if (v->groups) b = tr_budget_add(b, tr_budget_mul(n, 4));
    uint32_t used = 0;
    for (int32_t i = 0; i < p->ndims; i++) used |= (uint32_t)1 << p->column[i];
    if (p->weight >= 0) used |= (uint32_t)1 << p->weight;
    for (int32_t c = 0; c < v->ncols; c++) {
        if (!((used >> c) & 1)) continue;
        b = tr_budget_add(b, tr_budget_mul(n, v->col[c].values ? 8 : 4));
        if (v->col[c].valid) b = tr_budget_add(b, n);
    }
    b = tr_budget_add(b, tr_budget_add(tr_budget_mul(p->ngroups, 80), tr_budget_mul(slices, 24)));
    b = tr_budget_add(b, 8);
    if (p->method == TR_KAPPROX_TAYLOR) {
        b = tr_budget_add(b, tr_budget_mul(n, 32));
        b = tr_budget_add(b, tr_budget_mul(p->nitems, (uint64_t)p->ndims));
        b = tr_budget_add(b, tr_budget_mul(tr_budget_mul(p->ngroups, p->nitems), 32));
        return b;
    }
    uint64_t f = tr_fourier_device_bytes(&p->fourier, v);
    return f > b ? f : b;
}
/* a plan view consistent with a scope view (host code of both libraries) */
static inline int tr_kapprox_plan_view_valid(const tr_kapprox_plan_view *p, const tr_scope_view *v) {
    if (!p || !v || (p->method != TR_KAPPROX_TAYLOR && p->method != TR_KAPPROX_RFF) || p->ndims < 1 ||
        p->ndims > TR_KAPPROX_MAX_DIMS || p->ngroups != v->ngroups || p->weight < -1 || p->weight >= v->ncols ||
        (v->rows && !v->selection) || (v->rows && v->ngroups > 1 && !v->groups) || p->nitems < 1) return 0;
    for (int32_t i = 0; i < p->ndims; i++) if (p->column[i] < 0 || p->column[i] >= v->ncols) return 0;
    if (p->method == TR_KAPPROX_TAYLOR)
        return p->exps && p->shift >= 0 && p->shift <= TR_CALC_F && p->degree >= 0 && p->degree <= TR_KAPPROX_MAX_DEGREE &&
               p->tmax < TR_KAPPROX_NMAX && p->scale <= TR_CALC_SCALE_MAX;
    return tr_fourier_plan_view_valid(&p->fourier, v) && p->nitems == 2 * p->fourier.nfreq;
}

/* ================================================================ exported (math_kapprox.c) */
/* the plan of a state: the spec checked and canonical, the RFF frequencies drawn (and SPEC-001's plan), the TAYLOR
 * items and constants; budgeted */
tr_kapprox_plan *tr_kapprox_plan_create(const tr_scope *, const tr_kapprox_spec *, const tr_math_budget *,
                                        tr_math_refusal *, int32_t *error);
int32_t tr_kapprox_plan_get_view(const tr_kapprox_plan *, tr_kapprox_plan_view *);
void tr_kapprox_plan_free(tr_kapprox_plan *);
/* the sums of a plan over the scope (the CPU reference of tr_mgpu_kapprox_sums): groups[G], maxsq[3], sums (TAYLOR:
 * G x nitems x 2 (floor, ceil); RFF: SPEC-001's G x m x 4), and for RFF SPEC-001's rows / valid / weight sums */
int32_t tr_kapprox_sums(const tr_kapprox_plan_view *, const tr_scope_view *, tr_kapprox_group *groups, uint64_t *maxsq,
                        tr_math_i128 *sums, uint64_t *frows, uint64_t *fvalid, tr_math_i128 *fwsum);
/* the state from sums (computed by tr_kapprox_sums or the CUDA kernels), after consistency checks */
tr_kapprox_state *tr_kapprox_from_sums(const tr_kapprox_plan *, const tr_scope *, const tr_kapprox_group *groups,
                                       const uint64_t *maxsq, const tr_math_i128 *sums, const uint64_t *frows,
                                       const uint64_t *fvalid, const tr_math_i128 *fwsum, const tr_math_budget *,
                                       tr_math_refusal *, int32_t *error);
/* plan + sums + from_sums on the CPU */
tr_kapprox_state *tr_kapprox_build(const tr_scope *, const tr_kapprox_spec *, const tr_math_budget *,
                                   tr_math_refusal *, int32_t *error);
/* the needs of a build: target CPU, or DEVICE (the device copies of tr_mgpu_kapprox_sums) */
int32_t tr_kapprox_need(const tr_scope *, const tr_kapprox_spec *, int32_t target, tr_math_need *);
/* the generator alone: count x ndims draws (before deduplication) as canonical rationals; *fbits = p */
int32_t tr_kapprox_generate(int32_t kernel, tr_num_rational gamma, int32_t ndims, uint64_t count, uint64_t seed,
                            tr_num_rational *out, int32_t *fbits, const tr_math_budget *, tr_math_refusal *);

int32_t tr_kapprox_get_info(const tr_kapprox_state *, tr_kapprox_info *);
int32_t tr_kapprox_get_spec(const tr_kapprox_state *, tr_kapprox_spec *);
int32_t tr_kapprox_get_group(const tr_kapprox_state *, uint64_t g, tr_kapprox_group *);
/* the raw feature sums of group g: nitems x 2 (floor, ceil) over 2^scale */
int32_t tr_kapprox_get_sums(const tr_kapprox_state *, uint64_t g, tr_math_i128 *out, uint64_t capacity);
/* frequency f (RFF): ndims exact rationals and its multiplicity */
int32_t tr_kapprox_get_frequency(const tr_kapprox_state *, uint64_t f, tr_num_rational *out, uint32_t *mult);
/* an exact rational of the state (TR_KAPPROX_Q_*; index: the class of Q_COEF), borrowed; NULL when not applicable */
const tr_bigq *tr_kapprox_get_q(const tr_kapprox_state *, int32_t which, int32_t index);
/* n requests; points: npoints x ndims logical values (witness points). out[n]: every answer's own status. */
int32_t tr_kapprox_query(const tr_kapprox_state *, const tr_kapprox_request *requests, uint64_t n, const int64_t *points,
                         uint64_t npoints, tr_kapprox_result *out, const tr_math_budget *, tr_math_refusal *);
void tr_kapprox_result_free(tr_kapprox_result *, uint64_t n);
/* FND-007: never authorized (TR_MATH_DOMAIN_ERROR; reason APPROXIMATE, STATISTICAL for a probabilistic state) */
int32_t tr_kapprox_authorize_exact(const tr_kapprox_state *, uint64_t field, tr_cert_decision *out);
/* the recorded source against a current scope: TR_MATH_OK, TR_MATH_STALE (reasons), TR_MATH_UNSUPPORTED (unscoped) */
int32_t tr_kapprox_fresh(const tr_kapprox_state *, const tr_scope *current, tr_cert_staleness *out);
/* the record (out NULL: only *size); TR_MATH_INVALID when capacity is smaller */
int32_t tr_kapprox_save(const tr_kapprox_state *, const uint8_t *meta, uint64_t meta_bytes, uint8_t *out,
                        uint64_t capacity, uint64_t *size, const tr_math_budget *, tr_math_refusal *);
tr_kapprox_state *tr_kapprox_load(const uint8_t *data, uint64_t bytes, const tr_scope *origin, const tr_math_budget *,
                                  tr_math_refusal *, tr_persist_status *status, tr_cert_staleness *stale,
                                  int32_t *error);
/* the metadata a loaded state was saved with (*length its size; INVALID when capacity is smaller) */
int32_t tr_kapprox_get_metadata(const tr_kapprox_state *, uint8_t *out, uint64_t capacity, uint64_t *length);
void tr_kapprox_free(tr_kapprox_state *);

/* ------------------------------------------------------------------ the CUDA library (math_kapprox_gpu.cu)
 * tr_mgpu_kapprox_sums: exactly tr_kapprox_sums on the device (a counts kernel; TAYLOR: a prefactor kernel and a
 * feature kernel over (item, row slice) with exact 128-bit atomics; RFF: SPEC-001's tr_mgpu_fourier_sums) */
int tr_mgpu_kapprox_sums(const tr_kapprox_plan_view *, const tr_scope_view *, tr_kapprox_group *groups,
                         uint64_t *maxsq, tr_math_i128 *sums, uint64_t *frows, uint64_t *fvalid, tr_math_i128 *fwsum,
                         const tr_math_budget *, tr_math_refusal *);
#ifdef __cplusplus
}
#endif
#endif
