/* PROB-001: small-support partition and likelihood functions: declared finite-support probability models evaluated
 * from retained statistics (the counts of a DIST-001 histogram aligned to a declared support), with exact rationals
 * for categorical models and certified enclosures for exponential families, for many parameter requests at once, on
 * the CPU (math_prob.c) and in CUDA (math_prob_gpu.cu).
 * SPDX-License-Identifier: Apache-2.0
 *
 * ---- The declared support ----------------------------------------------------------------------------------------
 * A support (tr_prob_support) is a declared finite set of d points x_0 < ... < x_(d-1) of k = 1..3 coordinates, each
 * coordinate a logical int64 value or NULL (the NULL category of DIST-001: a value of its own, after every value), in
 * DIST-001 cell order (coordinate 0 most significant, values increasing, NULL last), with a decimal scale per
 * coordinate (value = logical / 10^scale). It is declared, never inferred from data: a point that no row has stays in
 * the support with mass 0. tr_prob_support_create sorts the given points natively (perm[i] = the input position of
 * point i) and refuses duplicates (TR_MATH_INVALID). Its identity is the exact content (k, d, scales, points); the
 * 128-bit digest is a pre-filter only.
 *
 * ---- Models ------------------------------------------------------------------------------------------------------
 * A model (tr_prob_model) over a support, with m = 0..16 features T(x) in Q^m (one row of rationals per point; stored
 * natively as integers N_ij = T_j(x_i) D_j over the least common denominator D_j of feature j, each in int64 or
 * TR_MATH_OVERFLOW):
 *   CATEGORICAL  p_w(x_i) = w_i / Z, Z = sum_i w_i: the weights w (rationals >= 0) are the parameters of a request
 *                (TR_PROB_WEIGHTS, unnormalized) or exact probabilities (TR_PROB_PROBABILITIES: sum w = 1 exactly, else
 *                the request is refused with TR_MATH_DOMAIN_ERROR). The gradient and Hessian are those of the
 *                exponential tilt p_(w, theta)(x) ~ w(x) exp(theta . T(x)) at theta = 0 (observed T totals minus
 *                N E_w[T], -N Cov_w[T]); m = 0 means the one-hot statistic T(x_i) = e_i, i.e. the natural parameters
 *                eta_i = ln w_i: gradient n_i - N p_i, Hessian -N (diag(p) - p p^T) (d and d (d + 1) / 2 entries).
 *                Everything but log Z and the log-likelihood is an exact rational.
 *   EXPFAM       p_theta(x_i) = h_i exp(theta . T(x_i)) / Z(theta), Z(theta) = sum_i h_i exp(theta . T(x_i)), with a
 *                declared base measure h >= 0 (rationals; stored as integers H_i = h_i Dh, Dh their lcm; default 1)
 *                and m >= 1 features; theta (m rationals, numerators and denominators in int64) is a request's
 *                parameter. Z, log Z, the log-likelihood, its gradient and Hessian are certified enclosures.
 *   A base measure that is zero everywhere (Z = 0 for every theta) is refused at creation (TR_MATH_DOMAIN_ERROR); a
 *   categorical request whose weights are all 0 (Z = 0) is refused with its own status TR_MATH_DOMAIN_ERROR.
 *
 * ---- Retained statistics -----------------------------------------------------------------------------------------
 * tr_prob_stats_build aligns a DIST-001 histogram (math_dist.h) of the same k and scales (else TR_MATH_DOMAIN_ERROR)
 * with a declared support, without reading a row: per group g and support point i the mass n_gi (mass COUNT: the
 * cell's rows; WEIGHT: its exact weight sum, every present weight >= 0 or TR_MATH_DOMAIN_ERROR, a group total at or
 * past 2^64 TR_MATH_UNSUPPORTED), N_g = sum_i n_gi, and the group's selected rows (0: an absent group). Cells with a
 * NULL in a coordinate of the declared exclusion mask are dropped first (listwise; their mass is reported, never
 * counted). Every other occupied cell (count > 0) must be a declared support point: an observed value outside the
 * declared support is refused (TR_MATH_DOMAIN_ERROR, the first such cell in *outside). Support points without rows
 * keep mass 0. The state's identity is its support (exact), the histogram's scope digest, the mass kind, the
 * exclusion mask and the groups: a model over another support is refused (tr_prob_stats_require:
 * TR_MATH_DOMAIN_ERROR), so a support change is always a different state. With a scope (the histogram's own: same
 * digest, groups and rows, else TR_MATH_INVALID) the state also records the scope's data source (GEN-002's rule,
 * math_conv.h: an engine table's address and identity token, or an array scope's row descriptor) and its grouping.
 *
 * ---- Dependent events --------------------------------------------------------------------------------------------
 * A joint model over (x, y) is evaluated on the statistics of the joint histogram of (x, y): the dependence of the
 * coordinates is in the joint counts, never inferred. tr_prob_stats_product forms the product of two statistics
 * states (per group g: n(x_i, y_j) = n_a(x_i) n_b(y_j), the CROSS JOIN of the two bags within the group, over the
 * product support) and is the only route from marginal statistics to a joint state. Its use as the joint of x and y
 * is an independence assumption, so it is checked natively the way GEN-002 checks a product of factors: states whose
 * scopes have the same data source (one table or one array scope: rows that can be paired) multiply only under an
 * independence declaration (tr_conv_declare) that covers that source (tr_conv_decl_covers); otherwise
 * TR_MATH_DOMAIN_ERROR with *check (GEN-002's tr_conv_check: TR_CONV_NEEDS_DECLARATION, TR_CONV_NOT_COVERED). One
 * source needs the same grouping (TR_CONV_GROUPING_DIFFERS), different sources the same number of groups
 * (TR_CONV_KEYS_DIFFER). States of different sources form their Cartesian product without a declaration (GEN-002).
 * The result records whether it was declared. States without a source cannot be multiplied (TR_MATH_INVALID).
 *
 * ---- Evaluation (per request r: group g, parameters; N = N_g, Tobs = sum_i n_gi T(x_i) exactly) ---------------------
 *   Z, log Z               the normalizer and its logarithm (with the declared denominators: Z = sum_i h_i e_i)
 *   loglik                 sum_i n_gi log p(x_i): -infinity (flag TR_PROB_NEG_INF, never a finite number) when a point
 *                          of zero model probability (w_i = 0 or h_i = 0) has n_gi > 0; zero masses contribute 0
 *   gradient (WANT_GRAD)   Tobs - N E[T] (m entries; one-hot: d), undefined (NaN) with NEG_INF
 *   Hessian (WANT_HESS)    -N Cov[T], the packed upper triangle (entry (j, l), j <= l, at j mm - j (j + 1) / 2 + l)
 * An absent group or N = 0 is flagged TR_PROB_EMPTY (loglik 0, gradient and Hessian 0: the empty sample).
 * Each request has its own status: 0, TR_MATH_INVALID (a group out of range, a denominator < 1), TR_MATH_DOMAIN_ERROR
 * (a negative weight, Z = 0, probabilities not summing to 1), TR_MATH_OVERFLOW (a common denominator or a sum that
 * cannot be proven to fit, an exponent past 2^(2^24): theta's int64 parts always give K_j < 2^256).
 *
 * Exactness and precision.
 *   CATEGORICAL  per request the weights are put over their lcm Dr (W_i = w_i Dr in int64, else OVERFLOW); the sums
 *                Zw = sum W_i, S_j = sum W_i N_ij, S_jl = sum W_i N_ij N_il are exact in 128 bits (proven before the
 *                sums: bitlen(d) + bitlen(max W) + bitlen(max |feature product|) <= 126, else OVERFLOW); Z = Zw / Dr,
 *                the gradient (Tobs_j Zw - N S_j) / (D_j Zw) and the Hessian -N (S_jl Zw - S_j S_l) / (D_j D_l Zw^2)
 *                are exact rationals (256-bit numerators). log-likelihood = sum n_i ln W_i - N ln Zw.
 *   EXPFAM       e_i = exp(theta . T(x_i)) = 2^(t_i), t_i = sum_j N_ij K_j with K_j = theta_j / (D_j ln 2) as the 64.192
 *                fixed-point constant of CALC-003 (math_calc.h: floor(|num| floor(2^512 / ln 2) / (den D_j 2^320)),
 *                error <= 2^-191), reduced and exponentiated by CALC-003's own tr_calc_exp2_reduce: 2^t = 2^n m with
 *                n = floor(t) and m in [1, 2] in 120-bit midpoint-radius fixed point (relative radius <= 2^-112).
 *                Per request, n_ref = max_i (n_i + bitlen(H_i)) over the points with H_i > 0 (an exact maximum: a
 *                first pass), and u_i = (H_i / 2^bitlen(H_i)) m_i 2^(n_i + bitlen(H_i) - n_ref) < 2 (the reference
 *                point's u >= 1/2). The sums S_0 = sum u_i, S_j = sum u_i N_ij, S_jl = sum u_i N_ij N_il are
 *                accumulated as outward integers floor(lo 2^s_q), ceil(hi 2^s_q) of each term, exactly in 128 bits,
 *                at the scales s_0 = s = min(120, 124 - bitlen(d)), s_j = s - b_j, s_jl = s - b_j - b_l (b_j =
 *                bitlen(max_i |N_ij|)), which prove every partial sum below 2^126 in any order. Then
 *                  Z = 2^(n_ref - s) [S_0lo, S_0hi] / Dh,  log Z = (n_ref - s) ln 2 + ln S_0 - ln Dh,
 *                  loglik = sum_i n_i ln H_i + theta . Tobs - N ((n_ref - s) ln 2 + ln S_0),
 *                  grad_j = Tobs_j - (N 2^b_j / D_j) S_j / S_0,   Hess_jl = -N 2^(b_j + b_l) (S_jl S_0 - S_j S_l) /
 *                  (D_j D_l S_0^2), the bounds of the ratios taken at the corners of the sums' boxes, the numerators
 *                  exact 256-bit integers (no cancellation is rounded); a variance's upper bound is clamped at 0.
 *   Logarithms   DIST-003's certified 120-bit logarithm (math_info.h: tr_info_ln, 2 atanh series, <= 144 ulps of
 *                2^-120) for integers below 2^64, and the same series on the 120-bit mantissa for 128-bit integers
 *                (tr_prob_ln128: the series at the truncated mantissa F and at F + 1, so hi - lo <= E + 149 <= 277
 *                ulps for M = 2^E f, E <= 128; exact mantissas: E + 80). The log sums sum n_i ln W_i (CATEGORICAL,
 *                a device sum at scale sL = min(120, 119 - bitlen(N_g))) and sum n_i ln H_i (EXPFAM, exact 256-bit
 *                products at 2^-120) are exact integer sums of certified bounds.
 * Every answer carries the double enclosures [RD, RU] of Z (may be [DBL_MAX, inf] or [0, 2^-1074] past the double
 * range), log Z and the log-likelihood; the exact form of Z (zlo, zhi, zexp, zden); the exact 2^-120 bounds of log Z and
 * of the transcendental part of the log-likelihood; double enclosures of the gradient and Hessian entries; and the
 * reported widths (width[0]: an upper bound of (zhi - zlo) / zlo; width[1..4]: of log Z, loglik, the widest gradient
 * and Hessian enclosures). tr_prob_exact returns the exact rational bounds of any entry (CPU, FND-009 big integers).
 *
 * CUDA (math_prob_gpu.cu, tr_mgpu_prob_sums): the sums of a plan: one work item per (request, TR_PROB_CHUNK support
 * points). EXPFAM: launch 1 computes n_i + bitlen(H_i) of each point (the integer part of CALC-003's reduction) and an
 * atomic maximum per request; launch 2 the exponentials (tr_calc_exp2_reduce), the outward integers and local 128-bit
 * sums, flushed with exact two-word atomics (tr_atomic_add_wide). CATEGORICAL: one launch of exact sums and log bounds.
 * Flags (an exponent overflow, a zero-probability point with mass) by atomic OR. Every sum is an exact integer proven
 * to stay below 2^126, so the device sums equal the CPU's (tr_prob_sums) bit for bit whatever the thread order; the
 * host finishes both with the same tr_prob_finish.
 *
 * Budgets (math_budgets.h), checked before allocation: support: variables = k, support = d, bytes; model: support = d,
 * variables = m, tensor = d m, bytes; statistics: support = d, states = G d, bytes (+ the source descriptor); product:
 * support = da db, states = G da db, bytes; evaluation: output = n x (3 + gradient + Hessian entries), tensor = n d
 * (the per-request work), states = n x sums, bytes = the plan (680 bytes of reduction constants per EXPFAM request,
 * 8 d per CATEGORICAL request), the sums and the finish scratch; CUDA: bytes = the device copies (tr_prob_device_bytes).
 * Refusal: TR_MATH_BUDGET with the tr_math_refusal; sizes past 64 bits are refused whatever the budget.
 *
 * Errors: TR_MATH_INVALID, _MEMORY, _OVERFLOW, _BUDGET, _DOMAIN_ERROR, _UNSUPPORTED (k > 3, m > 16), TR_MATH_DEVICE.
 * Constructors return NULL and set *error, leaving their inputs unchanged; objects are immutable and individually
 * owned; read-only calls may run concurrently. TR_MATH_STALE is the caller's (the Python layer checks the scope).
 *
 * For PROB-002: the support (tr_prob_support_get_view), the retained statistics (tr_prob_stats_get_view: dense masses
 * per group over the declared support), the models (tr_prob_model_get_view) and the evaluation entry points
 * (tr_prob_plan_create, tr_prob_sums, tr_prob_finish, tr_prob_exact, tr_mgpu_prob_sums) are public. */
#ifndef TORBLES_MATH_PROB_H
#define TORBLES_MATH_PROB_H
#include <stdint.h>
#include <string.h>
#include "math_num.h"
#include "math_budgets.h"
#include "math_scope.h"
#include "math_dist.h"
#include "math_calc.h"
#include "math_info.h"
#include "math_carriers.h"
#include "math_conv.h"
#ifdef __cplusplus
extern "C" {
#endif

#define TR_PROB_ABI 1
#define TR_PROB_MAX_DIMS 3
#define TR_PROB_MAX_FEATURES 16                    /* = TR_CALC_FACTORS_MAX: CALC-003's reduction */
#define TR_PROB_MAX_SUMS 155                       /* 1 + 16 + 136 (+ 2 log bounds for CATEGORICAL) */
#define TR_PROB_CHUNK 64                           /* support points per CUDA work item */
#define TR_PROB_F 120                              /* fractional bits of the fixed point and the log bounds */
#define TR_PROB_MAX_POINTS (UINT64_C(1) << 32)     /* support points (indices fit 32 bits) */
enum { TR_PROB_CATEGORICAL = 1, TR_PROB_EXPFAM = 2 };
enum { TR_PROB_WEIGHTS = 0, TR_PROB_PROBABILITIES = 1 };                 /* CATEGORICAL request parameters */
enum { TR_PROB_WANT_GRAD = 1, TR_PROB_WANT_HESS = 2 };
enum { TR_PROB_NEG_INF = 1,         /* the log-likelihood is -infinity: a point of zero probability has mass */
       TR_PROB_EMPTY = 2,           /* an absent group or N = 0: the empty sample */
       TR_PROB_EXP_OVERFLOW = 4,    /* an exponent past 2^(2^24) (CALC-003's bound): status TR_MATH_OVERFLOW */
       TR_PROB_INCONSISTENT = 8 };  /* sums that violate their proof (impossible for CPU sums): status INVALID */
enum { TR_PROB_NEED_CPU = 0, TR_PROB_NEED_DEVICE = 1 };
/* exact entries of an answer (tr_prob_exact): 0 Z, 1 log Z, 2 loglik, 3 + j gradient j, 3 + mm + p Hessian entry p */
enum { TR_PROB_ENTRY_Z = 0, TR_PROB_ENTRY_LOGZ = 1, TR_PROB_ENTRY_LOGLIK = 2, TR_PROB_ENTRY_GRAD = 3 };

/* ---------------------------------------------------------------- views and descriptions */
typedef struct {                   /* 64 bytes */
    int32_t k, reserved;
    uint64_t d;                    /* points */
    int32_t scale[3], reserved2;
    const int64_t *value;          /* [d x k] row-major, 0 where NULL */
    const uint8_t *nulls;          /* [d] bit c: coordinate c is NULL */
    uint64_t digest[2];            /* pre-filter of the exact content */
} tr_prob_support_view;

typedef struct {                   /* 136 bytes */
    int32_t kind, m;               /* TR_PROB_CATEGORICAL / _EXPFAM; features */
    uint64_t d;
    int32_t scale, reserved;       /* EXPFAM: s = min(120, 124 - bitlen(d)) */
    int32_t bits[16];              /* b_j = bitlen(max_i |N_ij|) */
    const int64_t *feat;           /* [d x m] N_ij = T_j(x_i) D_j */
    const int64_t *fden;           /* [m] D_j */
    const int64_t *base;           /* EXPFAM: [d] H_i = h_i Dh (CATEGORICAL: NULL) */
    const int32_t *hbits;          /* EXPFAM: [d] bitlen(H_i) */
    const tr_math_i128 *lnh;       /* EXPFAM: [2 d] lo <= ln(H_i) 2^120 <= hi (0, 0 where H_i <= 1) */
    int64_t hden;                  /* EXPFAM: Dh */
} tr_prob_model_view;

typedef struct {                   /* 64 bytes */
    uint64_t ngroups, d;
    int32_t k, mass;               /* TR_DIST_MASS_COUNT / _WEIGHT */
    uint32_t exclude;              /* coordinates whose NULL cells were dropped */
    int32_t declared;              /* a product made under an independence declaration */
    const uint64_t *counts;        /* [G x d] n_gi */
    const uint64_t *total;         /* [G] N_g */
    const uint64_t *rows;          /* [G] selected rows (0: an absent group); products: rows_a x rows_b */
    const uint64_t *dropped;       /* [G] mass dropped by the exclusion */
} tr_prob_stats_view;

typedef struct {                   /* 104 bytes */
    uint64_t ngroups, present, d, total, bytes;   /* total: sum of N_g (saturating) */
    uint64_t digest[2];            /* the histogram's scope digest (0 when unscoped or a product) */
    uint64_t support_digest[2];
    int32_t k, mass, exclude, scoped;   /* scoped: a data source is recorded */
    int32_t declared, product, source_kind, keyed;   /* TR_SCOPE_TABLE / _ARRAYS (0: none); keyed: G > 1 */
} tr_prob_stats_info;

/* the first observed cell outside the declared support (filled when a build is refused for it) (48 bytes) */
typedef struct {
    uint64_t group, mass;
    int64_t value[3];
    uint8_t nulls, reserved[7];
} tr_prob_outside;

/* what the sums read (kernels: by value, pointers swapped for device copies) (208 bytes) */
typedef struct {
    int32_t kind, what, m, mm;     /* mm: gradient entries (m, or d for one-hot CATEGORICAL) */
    uint64_t n, d, ngroups;
    uint64_t nsums;                /* int128 sums per request (tr_prob_nsums) */
    int32_t scale, reserved;       /* EXPFAM: s */
    int32_t bits[16];
    const int64_t *feat;           /* [d x m] */
    const int64_t *base;           /* EXPFAM: [d] */
    const int32_t *hbits;          /* EXPFAM: [d] */
    const uint64_t *counts;        /* CATEGORICAL: [G x d] masses (the log sums) */
    const uint64_t *group;         /* [n] */
    const int32_t *status;         /* [n] 0: evaluate */
    const tr_calc_plan_view *exp;  /* EXPFAM: [n] request r's reduction constants (CALC-003's layout) */
    const int64_t *weight;         /* CATEGORICAL: [n x d] W_ri */
    const int32_t *lscale;         /* CATEGORICAL: [n] sL */
    uint64_t reserved2[2];
} tr_prob_plan_view;

/* one answer (296 bytes); see the header comment */
typedef struct {
    int32_t status, flags;
    uint64_t group, mass, rows;    /* the group, its mass N and selected rows */
    int64_t zexp, zden;            /* Z in [zlo, zhi] 2^zexp / zden (CATEGORICAL: zlo = zhi = Zw, zexp = 0, zden = Dr) */
    tr_math_i128 zlo, zhi;
    tr_dist_i256 logz_lo, logz_hi; /* log Z in [logz_lo, logz_hi] / 2^120 */
    tr_dist_i256 ll_lo, ll_hi;     /* the log-likelihood without theta . Tobs in [ll_lo, ll_hi] / 2^120 */
    tr_math_interval z, logz, loglik;
    double width[5];
} tr_prob_answer;

/* ---------------------------------------------------------------- sizes */
TR_NUM_HD uint64_t tr_prob_tri(uint64_t m) { return m * (m + 1) / 2; }
/* int128 sums per request: EXPFAM 2 (lo, hi) per sum (S_0, then S_j, S_jl as wanted); CATEGORICAL the exact sums
 * (Zw, S_j, S_jl) then the two log bounds */
TR_NUM_HD uint64_t tr_prob_nsums(int32_t kind, int32_t m, int32_t what) {
    uint64_t q = 1;
    if (m && (what & (TR_PROB_WANT_GRAD | TR_PROB_WANT_HESS))) q += (uint64_t)m;
    if (m && (what & TR_PROB_WANT_HESS)) q += tr_prob_tri((uint64_t)m);
    return kind == TR_PROB_EXPFAM ? 2 * q : q + 2;
}
/* the packed upper-triangle index of (j, l), j <= l < mm */
TR_NUM_HD uint64_t tr_prob_pack(uint64_t mm, uint64_t j, uint64_t l) { return j * mm - j * (j + 1) / 2 + l; }
/* the sum index of S_j (j < m), and of S_jl (j <= l < m), after S_0 */
TR_NUM_HD uint64_t tr_prob_q_feature(int32_t j) { return 1 + (uint64_t)j; }
TR_NUM_HD uint64_t tr_prob_q_pair(int32_t m, int32_t j, int32_t l) {
    return 1 + (uint64_t)m + tr_prob_pack((uint64_t)m, (uint64_t)j, (uint64_t)l);
}
TR_NUM_HD int32_t tr_prob_bitlen64(uint64_t x) { return x ? 64 - tr_info_clz64(x) : 0; }
TR_NUM_HD int32_t tr_prob_bitlen128(unsigned __int128 x) {
    uint64_t hi = (uint64_t)(x >> 64);
    return hi ? 64 + tr_prob_bitlen64(hi) : tr_prob_bitlen64((uint64_t)x);
}

/* ---------------------------------------------------------------- 256-bit helpers (on top of math_dist.h) */
/* floor (up = 0) or ceil (up = 1) of v / 2^k for a signed 256-bit v (|v| < 2^255) and k >= 0, as an int128: 1, or 0
 * (and *out = 0) when it does not fit */
TR_NUM_HD int tr_prob_shift_round(tr_dist_i256 v, int64_t k, int up, __int128 *out) {
    if (up) v = tr_dist_w_negate_bits(v);                           /* ceil(v / 2^k) = -floor(-v / 2^k) */
    int vneg = tr_dist_w_negative(v);
    uint64_t ext = vneg ? ~(uint64_t)0 : 0;
    tr_dist_i256 q;
    if (k >= 256) {
        for (int i = 0; i < 4; i++) q.w[i] = ext;                    /* floor: -1 or 0 */
    } else {
        int32_t s = (int32_t)(k / 64), b = (int32_t)(k % 64);
        for (int i = 0; i < 4; i++) {
            uint64_t lo = i + s < 4 ? v.w[i + s] : ext, hi = i + s + 1 < 4 ? v.w[i + s + 1] : ext;
            q.w[i] = b ? (lo >> b) | (hi << (64 - b)) : lo;
        }
    }
    if (up) q = tr_dist_w_negate_bits(q);
    uint64_t top = (q.w[1] >> 63) ? ~(uint64_t)0 : 0;
    if (q.w[2] != top || q.w[3] != top) { *out = 0; return 0; }
    *out = (__int128)(((unsigned __int128)q.w[1] << 64) | q.w[0]);
    return 1;
}
TR_NUM_HD tr_dist_i256 tr_prob_w_shl(tr_dist_i256 v, int32_t k) {     /* v 2^k for 0 <= k < 256 (bits past 2^255 lost) */
    tr_dist_i256 r;
    int32_t s = k / 64, b = k % 64;
    for (int i = 3; i >= 0; i--) {
        uint64_t lo = i - s >= 0 ? v.w[i - s] : 0, lo2 = i - s - 1 >= 0 ? v.w[i - s - 1] : 0;
        r.w[i] = b ? (lo << b) | (lo2 >> (64 - b)) : lo;
    }
    return r;
}

/* ---------------------------------------------------------------- the certified logarithm of 128-bit integers */
/* lo <= ln(M) 2^120 <= hi for 1 <= M < 2^128: below 2^64 DIST-003's tr_info_ln itself; above, M = 2^E f with the
 * 120-bit mantissa f in [sqrt(2)/2, sqrt(2)] truncated (F <= f 2^120 < F + 1) and DIST-003's series (each value at
 * most 34 ulps below its exact one) on F for the lower and on F + 1 (+ 2 TR_INFO_LOG_RAD) for the upper bound, E ln 2
 * from floor(ln 2 2^120) + [0, E): hi - lo <= E + 80 + 69 (E <= 128) */
TR_NUM_HD void tr_prob_ln_mantissa(unsigned __int128 F, unsigned __int128 *slo, unsigned __int128 *shi, int *neg) {
    if (F == TR_INFO_ONE) { *slo = *shi = 0; *neg = 0; return; }       /* ln 1 = 0: z = 0, the series is exact */
    *neg = F < TR_INFO_ONE;
    unsigned __int128 num = *neg ? TR_INFO_ONE - F : F - TR_INFO_ONE, den = F + TR_INFO_ONE;
    unsigned __int128 Z = tr_info_fdiv(num, den), W = tr_info_fmul(Z, Z), T = Z, S = Z;
    for (uint32_t j = 1; j <= TR_INFO_LOG_TERMS; j++) {
        T = tr_info_fmul(T, W);
        S += tr_info_divsmall(T, 2 * j + 1);
    }
    *slo = 2 * S; *shi = 2 * S + 2 * TR_INFO_LOG_RAD;                 /* |series part| in [slo, shi] */
}
TR_NUM_HD void tr_prob_ln128(unsigned __int128 M, __int128 *lo, __int128 *hi) {
    if ((M >> 64) == 0) {
        unsigned __int128 a, b;
        tr_info_ln((uint64_t)M, &a, &b);
        *lo = (__int128)a; *hi = (__int128)b;
        return;
    }
    int32_t e = tr_prob_bitlen128(M) - 1;                             /* 64 .. 127 */
    unsigned __int128 F;                                               /* floor(M 2^(120 - e)), in [2^120, 2^121) */
    int inexact = 0;
    if (e <= TR_INFO_F) F = M << (TR_INFO_F - e);                      /* exact */
    else {
        F = M >> (e - TR_INFO_F);
        inexact = (M & ((((unsigned __int128)1) << (e - TR_INFO_F)) - 1)) != 0;
    }
    if (F > TR_INFO_SQRT2) { inexact |= (int)(F & 1); F >>= 1; e += 1; }   /* floor(M 2^(120 - e - 1)) */
    unsigned __int128 a0, a1, b0, b1;
    int n0, n1;
    tr_prob_ln_mantissa(F, &a0, &a1, &n0);
    tr_prob_ln_mantissa(F + (unsigned __int128)inexact, &b0, &b1, &n1);
    __int128 blo = (__int128)((unsigned __int128)(uint64_t)e * TR_INFO_LN2);
    __int128 bhi = blo + (__int128)e;
    *lo = n0 ? blo - (__int128)a1 : blo + (__int128)a0;              /* ln f >= the lower bound at F */
    *hi = n1 ? bhi - (__int128)b0 : bhi + (__int128)b1;              /* ln f <= the upper bound at F + 1 */
}

/* ---------------------------------------------------------------- EXPFAM: per (request, point) terms */
/* n_i + bitlen(H_i) with n_i = floor(t_i) exactly as tr_calc_exp2_reduce computes it (the same accumulation of
 * sum_j sign_j |X_j| K_j); 0 when |floor t| reaches TR_CALC_NMAX */
TR_NUM_HD int tr_prob_exp2_floor(const tr_calc_plan_view *pl, const int64_t *X, int64_t *n) {
    uint64_t acc[6] = {0, 0, 0, 0, 0, 0};
    int any = 0;
    for (int32_t i = 0; i < pl->nfactors; i++) {
        if (X[i] == 0 || pl->ksign[i] == 0) continue;
        uint64_t p[6];
        tr_calc_mulk(tr_abs_u64(X[i]), pl->k[i], p);
        p[5] = 0;
        if ((X[i] < 0) != (pl->ksign[i] < 0)) {
            unsigned c = 1;
            for (int j = 0; j < 6; j++) { uint64_t v = ~p[j] + c; c = c && v == 0; p[j] = v; }
        }
        unsigned __int128 carry = 0;
        for (int j = 0; j < 6; j++) { carry += (unsigned __int128)acc[j] + p[j]; acc[j] = (uint64_t)carry; carry >>= 64; }
        any = 1;
    }
    if (!any) { *n = 0; return 1; }
    int neg = (int)(acc[5] >> 63);
    if (neg) { unsigned c = 1; for (int j = 0; j < 6; j++) { uint64_t v = ~acc[j] + c; c = c && v == 0; acc[j] = v; } }
    uint64_t N = acc[3], high = acc[4] | acc[5];
    if (high || N >= (uint64_t)TR_CALC_NMAX) return 0;
    unsigned __int128 F = ((unsigned __int128)acc[2] << 64) | acc[1];
    int64_t e = (int64_t)N;
    if (neg) { if (F) e = -e - 1; else e = -e; }
    *n = e;
    return 1;
}
/* request r's reference exponent contribution of point i (H_i > 0): 1 with *e = n_i + bitlen(H_i), or 0 on an exponent
 * overflow */
TR_NUM_HD int tr_prob_exp_ref(const tr_prob_plan_view *v, uint64_t r, uint64_t i, int64_t *e) {
    int64_t n;
    if (!tr_prob_exp2_floor(&v->exp[r], v->feat + i * (uint64_t)v->m, &n)) return 0;
    *e = n + v->hbits[i];
    return 1;
}
/* u = (H_i / 2^bitlen(H_i)) m_i, the point's value relative to 2^(n_i + bitlen(H_i)), and that exponent minus n_ref
 * (<= 0 for every point of the request): 1, or 0 on an exponent overflow */
TR_NUM_HD int tr_prob_exp_value(const tr_prob_plan_view *v, uint64_t r, uint64_t i, int64_t nref, tr_calc_fx *u,
                                int64_t *rel) {
    int64_t n;
    tr_calc_fx mant;
    if (!tr_calc_exp2_reduce(&v->exp[r], v->feat + i * (uint64_t)v->m, &n, &mant)) return 0;
    int32_t hb = v->hbits[i];
    tr_calc_fx h;
    h.c = (__int128)((unsigned __int128)(uint64_t)v->base[i] << (TR_PROB_F - hb));    /* exact, in [2^119, 2^120) */
    h.r = 0;
    *u = tr_calc_fx_mul(h, mant);
    *rel = n + hb - nref;
    return 1;
}
/* [L, H] containing F u 2^(rel + s - 120) (u = (c +- r) / 2^120): floor and ceil, exact; 0 when a bound does not fit
 * int128 (impossible under the scale proof) */
TR_NUM_HD int tr_prob_scaled(__int128 F, tr_calc_fx u, int64_t rel, int32_t s, __int128 *L, __int128 *H) {
    if (F == 0) { *L = 0; *H = 0; return 1; }
    __int128 lo = u.c - (__int128)u.r, hi = u.c + (__int128)u.r;
    __int128 a = F > 0 ? lo : hi, b = F > 0 ? hi : lo;                /* F a <= F b */
    int64_t k = (int64_t)TR_PROB_F - (int64_t)s - rel;                /* >= 0: s <= 120, rel <= 0 */
    if (k < 0) return 0;
    return tr_prob_shift_round(tr_dist_w_mul128(F, a), k, 0, L) && tr_prob_shift_round(tr_dist_w_mul128(F, b), k, 1, H);
}
/* CATEGORICAL: the bounds floor(n lo / 2^(120 - sL)) <= n ln(W) 2^sL <= ceil(n hi / 2^(120 - sL)) (n >= 1, W >= 1;
 * lo <= ln(W) 2^120 <= hi by DIST-003's tr_info_ln) */
TR_NUM_HD void tr_prob_log_term(uint64_t n, int64_t W, int32_t sL, __int128 *L, __int128 *H) {
    unsigned __int128 lo, hi;
    tr_info_ln((uint64_t)W, &lo, &hi);
    tr_prob_shift_round(tr_dist_w_mul_u128((unsigned __int128)n, lo), TR_PROB_F - sL, 0, L);
    tr_prob_shift_round(tr_dist_w_mul_u128((unsigned __int128)n, hi), TR_PROB_F - sL, 1, H);
}
/* the sums of a request (EXPFAM: lo-hi pairs; CATEGORICAL: the exact sums before the two log bounds) */
TR_NUM_HD uint64_t tr_prob_nq(const tr_prob_plan_view *v) {
    return v->kind == TR_PROB_EXPFAM ? v->nsums / 2 : v->nsums - 2;
}
/* the terms of one EXPFAM point (H_i > 0) into acc[2 nq] in the order S_0, S_j (j < m), S_jl (j <= l): 1, -1 on an
 * exponent overflow, 0 when a bound is not representable (impossible under the scale proof) */
TR_NUM_HD int tr_prob_exp_point(const tr_prob_plan_view *v, uint64_t r, uint64_t i, int64_t nref, __int128 *acc) {
    tr_calc_fx u;
    int64_t rel;
    if (!tr_prob_exp_value(v, r, i, nref, &u, &rel)) return -1;
    if (rel > 0) return 0;
    const int64_t *f = v->feat + i * (uint64_t)v->m;
    const uint64_t nq = tr_prob_nq(v);
    __int128 L, H;
    if (!tr_prob_scaled(1, u, rel, v->scale, &L, &H)) return 0;
    acc[0] += L; acc[1] += H;
    uint64_t q = 1;
    for (int32_t j = 0; j < v->m && q < nq; j++, q++) {
        if (!tr_prob_scaled((__int128)f[j], u, rel, v->scale - v->bits[j], &L, &H)) return 0;
        acc[2 * q] += L; acc[2 * q + 1] += H;
    }
    for (int32_t j = 0; j < v->m && q < nq; j++)
        for (int32_t l = j; l < v->m; l++, q++) {
            if (!tr_prob_scaled((__int128)f[j] * f[l], u, rel, v->scale - v->bits[j] - v->bits[l], &L, &H)) return 0;
            acc[2 * q] += L; acc[2 * q + 1] += H;
        }
    return 1;
}
/* the exact terms of one CATEGORICAL point into acc[nq] (Zw, S_j, S_jl), and its log bounds into lg[2] when its mass
 * is positive: 1, or 0 for a zero weight with positive mass (the log-likelihood is -infinity) */
TR_NUM_HD int tr_prob_cat_point(const tr_prob_plan_view *v, uint64_t r, uint64_t i, __int128 *acc, __int128 *lg) {
    const int64_t w = v->weight[r * v->d + i];
    const uint64_t nq = tr_prob_nq(v), cnt = v->counts[v->group[r] * v->d + i];
    acc[0] += w;
    if (nq > 1) {
        const int64_t *f = v->feat + i * (uint64_t)v->m;
        uint64_t q = 1;
        for (int32_t j = 0; j < v->m && q < nq; j++, q++) acc[q] += (__int128)w * f[j];
        for (int32_t j = 0; j < v->m && q < nq; j++)
            for (int32_t l = j; l < v->m; l++, q++) {
                tr_dist_i256 pr = tr_dist_w_mul128((__int128)f[j] * f[l], (__int128)w);   /* fits int128: the proof */
                acc[q] += (__int128)(((unsigned __int128)pr.w[1] << 64) | pr.w[0]);
            }
    }
    if (!cnt) return 1;
    if (!w) return 0;
    __int128 L, H;
    tr_prob_log_term(cnt, w, v->lscale[r], &L, &H);
    lg[0] += L; lg[1] += H;
    return 1;
}

/* ---------------------------------------------------------------- device bytes (host code of both libraries) */
static inline uint64_t tr_prob_device_bytes(const tr_prob_plan_view *v) {
    uint64_t b = tr_budget_mul(tr_budget_mul(v->d, (uint64_t)v->m), 8);
    b = tr_budget_add(b, tr_budget_mul(v->n, 8 + 4 + 8 + 8));             /* group, status, nref, flags */
    b = tr_budget_add(b, tr_budget_mul(tr_budget_mul(v->n, v->nsums), 16));
    if (v->kind == TR_PROB_EXPFAM) {
        b = tr_budget_add(b, tr_budget_mul(v->d, 12));
        b = tr_budget_add(b, tr_budget_mul(v->n, sizeof(tr_calc_plan_view)));
    } else {
        b = tr_budget_add(b, tr_budget_mul(tr_budget_mul(v->ngroups, v->d), 8));
        b = tr_budget_add(b, tr_budget_mul(tr_budget_mul(v->n, v->d), 8));
        b = tr_budget_add(b, tr_budget_mul(v->n, 4));
    }
    return b;
}
/* a plan view whose sizes and pointers are consistent (host code of both libraries) */
static inline int tr_prob_plan_view_valid(const tr_prob_plan_view *v) {
    if (!v || (v->kind != TR_PROB_CATEGORICAL && v->kind != TR_PROB_EXPFAM) || v->m < 0 ||
        v->m > TR_PROB_MAX_FEATURES || v->d < 1 || v->d > TR_PROB_MAX_POINTS || v->ngroups < 1) return 0;
    if (v->nsums != tr_prob_nsums(v->kind, v->m, v->what) || v->nsums > 2 * TR_PROB_MAX_SUMS) return 0;
    if (v->n && (!v->group || !v->status)) return 0;
    if (v->m && !v->feat) return 0;
    if (v->kind == TR_PROB_EXPFAM) {
        if (v->m < 1 || !v->base || !v->hbits || (v->n && !v->exp) || v->scale > TR_PROB_F) return 0;
        for (int32_t j = 0; j < v->m; j++) if (v->bits[j] < 0 || v->bits[j] > 63) return 0;
    } else if (!v->counts || (v->n && (!v->weight || !v->lscale))) return 0;
    return 1;
}

/* ---------------------------------------------------------------- objects (libalg) */
typedef struct tr_prob_support_s tr_prob_support;
typedef struct tr_prob_model_s tr_prob_model;
typedef struct tr_prob_stats_s tr_prob_stats;
typedef struct tr_prob_plan_s tr_prob_plan;

/* d points of k coordinates (values [d x k], nulls [d] or NULL: none), in any order; perm (may be NULL): [d] */
tr_prob_support *tr_prob_support_create(int32_t k, uint64_t d, const int64_t *values, const uint8_t *nulls,
                                        const int32_t *scales, uint64_t *perm, const tr_math_budget *,
                                        tr_math_refusal *, int32_t *error);
/* the product support A x B (k = ka + kb <= 3): points (a_i, b_j), i major */
tr_prob_support *tr_prob_support_product(const tr_prob_support *a, const tr_prob_support *b, const tr_math_budget *,
                                         tr_math_refusal *, int32_t *error);
int32_t tr_prob_support_get_view(const tr_prob_support *, tr_prob_support_view *);
/* 1 and *index when the point (values [k], nulls mask) is in the support, 0 when not, or TR_MATH_INVALID */
int32_t tr_prob_support_find(const tr_prob_support *, const int64_t *values, uint8_t nulls, uint64_t *index);
int32_t tr_prob_support_equal(const tr_prob_support *, const tr_prob_support *);   /* 1 / 0 (exact content) */
void tr_prob_support_free(tr_prob_support *);

/* kind CATEGORICAL (m >= 0, no base) or EXPFAM (m >= 1, base num / den [d] or NULL: every h_i = 1); features
 * fnum / fden [d x m] row-major (fden NULL: integers) */
tr_prob_model *tr_prob_model_create(const tr_prob_support *, int32_t kind, int32_t m, const int64_t *fnum,
                                    const int64_t *fden, const int64_t *bnum, const int64_t *bden,
                                    const tr_math_budget *, tr_math_refusal *, int32_t *error);
int32_t tr_prob_model_get_view(const tr_prob_model *, tr_prob_model_view *);
const tr_prob_support *tr_prob_model_support(const tr_prob_model *);    /* borrowed */
void tr_prob_model_free(tr_prob_model *);

/* the retained statistics of histogram h over the declared support (see the header comment); scope: h's own scope or
 * NULL (no data source); outside (may be NULL): the first observed cell outside the support */
tr_prob_stats *tr_prob_stats_build(const tr_dist_hist *h, const tr_scope *scope, const tr_prob_support *,
                                   int32_t mass, uint32_t exclude, const tr_math_budget *, tr_math_refusal *,
                                   tr_prob_outside *outside, int32_t *error);
int32_t tr_prob_stats_need(const tr_dist_hist *h, const tr_scope *scope, const tr_prob_support *, int32_t mass,
                           uint32_t exclude, tr_math_need *need);
/* the product of two states (see Dependent events); sa / sb: the scopes they were built with */
tr_prob_stats *tr_prob_stats_product(const tr_prob_stats *a, const tr_scope *sa, const tr_prob_stats *b,
                                     const tr_scope *sb, const tr_conv_decl *decl, const tr_math_budget *,
                                     tr_math_refusal *, tr_conv_check *check, int32_t *error);
int32_t tr_prob_stats_get_view(const tr_prob_stats *, tr_prob_stats_view *);
int32_t tr_prob_stats_get_info(const tr_prob_stats *, tr_prob_stats_info *);
const tr_prob_support *tr_prob_stats_support(const tr_prob_stats *);    /* borrowed */
/* 0 when the model's support is the state's (exact), TR_MATH_DOMAIN_ERROR otherwise (a new state is needed) */
int32_t tr_prob_stats_require(const tr_prob_stats *, const tr_prob_model *);
int32_t tr_prob_stats_equal(const tr_prob_stats *, const tr_prob_stats *);
void tr_prob_stats_free(tr_prob_stats *);

/* a plan of n requests: groups [n]; parameters pnum / pden: EXPFAM [n x m] theta, CATEGORICAL [n x d] weights (mode
 * TR_PROB_WEIGHTS / _PROBABILITIES); what: TR_PROB_WANT_*. Budgeted (target CPU), each request validated (its status) */
tr_prob_plan *tr_prob_plan_create(const tr_prob_stats *, const tr_prob_model *, uint64_t n, const uint64_t *groups,
                                  const int64_t *pnum, const int64_t *pden, int32_t mode, int32_t what,
                                  const tr_math_budget *, tr_math_refusal *, int32_t *error);
/* the needs of a plan of n requests (target TR_PROB_NEED_CPU: plan, sums and finish; _DEVICE: the device copies) */
int32_t tr_prob_plan_need(const tr_prob_stats *, const tr_prob_model *, uint64_t n, int32_t what, int32_t target,
                          tr_math_need *need);
int32_t tr_prob_plan_get_view(const tr_prob_plan *, tr_prob_plan_view *);
/* request r's status after validation */
int32_t tr_prob_plan_status(const tr_prob_plan *, uint64_t r);
/* the CPU sums (the reference of tr_mgpu_prob_sums): sums [n x nsums] (zeroed here), nref [n], flags [n] */
int32_t tr_prob_sums(const tr_prob_plan_view *, tr_math_i128 *sums, int64_t *nref, uint64_t *flags);
/* answers [n], gradient enclosures [n x mm] and Hessian enclosures [n x mm (mm + 1) / 2] (NULL when not wanted) from
 * the sums (CPU or device: checked against their proofs) */
int32_t tr_prob_finish(const tr_prob_plan *, const tr_math_i128 *sums, const int64_t *nref, const uint64_t *flags,
                       tr_prob_answer *out, tr_math_interval *grad, tr_math_interval *hess, const tr_math_budget *,
                       tr_math_refusal *);
/* the exact rational bounds lo = lo_num / lo_den <= value <= hi_num / hi_den of entry `entry` of request r (big
 * integers owned by the caller; lo == hi for an exact value: *exact = 1); a NEG_INF log-likelihood or an undefined
 * gradient / Hessian entry leaves them NULL and returns TR_MATH_OK with *flags */
int32_t tr_prob_exact(const tr_prob_plan *, uint64_t r, const tr_math_i128 *sums, const int64_t *nref,
                      const uint64_t *flags, uint64_t entry, const tr_math_budget *, tr_math_refusal *,
                      tr_bigint **lo_num, tr_bigint **lo_den, tr_bigint **hi_num, tr_bigint **hi_den, int32_t *exact,
                      int32_t *eflags);
void tr_prob_plan_free(tr_prob_plan *);
uint64_t tr_prob_device_need(const tr_prob_plan_view *);   /* tr_prob_device_bytes, exported */
/* the certified bounds of tr_prob_ln128 (M = hi 2^64 + lo; out: lo low, lo high, hi low, hi high words), for tests */
int32_t tr_prob_ln_bounds(uint64_t lo, uint64_t hi, int64_t out[4]);

/* ---------------------------------------------------------------- the CUDA library (math_prob_gpu.cu) */
/* the sums of tr_prob_sums on the device; budget: bytes = tr_prob_device_bytes, states = n x nsums */
int tr_mgpu_prob_sums(const tr_prob_plan_view *view, const tr_math_budget *budget, tr_math_i128 *sums, int64_t *nref,
                      uint64_t *flags, tr_math_refusal *refusal);
#ifdef __cplusplus
}
#endif
#endif
