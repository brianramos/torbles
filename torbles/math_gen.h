/* GEN-001: generating polynomials of finite-support distributions, from DIST-001 histograms (math_dist.h) or explicit
 * terms, with exact big-integer coefficients (FND-009 carriers, math_carriers.h); the 128-bit convolution and the
 * batched evaluation also in CUDA.
 * SPDX-License-Identifier: Apache-2.0
 *
 * The state. G(z) = sum over the support of c(v) z^v, one or two variables (z = (z_1, z_2) for a 2-D support): c(v)
 * is the bag multiplicity of the value v (the rows of a histogram cell: TR_DIST_MASS_COUNT) or its weight sum
 * (TR_DIST_MASS_WEIGHT, any sign); v ranges over logical integers (decimal values times 10^scale). A cell with a NULL
 * coordinate has no value and is not a term: its rows are reported as `nulls`, so G(1) = rows - nulls. Negative
 * values use a declared shift s (per variable): G(z) = z^s P(z), P(z) = sum c_e z^e over the exponents e = v - s >= 0
 * (a Laurent polynomial when s < 0). The shift is the smallest present value (TR_GEN_SHIFT_MIN) or given (explicit:
 * it must not exceed any present value, else TR_MATH_DOMAIN_ERROR); every operation keeps the values v exactly
 * (products add shifts). Representations: TR_GEN_SPARSE (the nonzero terms in lexicographic order of e) or
 * TR_GEN_DENSE (every exponent of the box [0, span_1) x [0, span_2), zeros included; the box: the hull of the
 * exponents unless converted to a larger one). Coefficients are exact (FND-009 big integers); zero terms are dropped
 * from a sparse state.
 *
 * Requests (exact):
 *   coefficient c(v) (0 off the support); export of every term
 *   evaluation G(p/q) at a rational point per variable (q >= 1), as a canonical big rational: the homogeneous Horner
 *     N = sum c_e p^e q^(D - e) over the terms (D = the largest exponent; gaps skipped with p^gap, q^gap: the work is
 *     the number of terms), divided by q^D, times p^s / q^s (s >= 0) or q^-s / p^-s (s < 0); a value
 *     v < 0 with a nonzero coefficient at z = 0 is a pole (TR_MATH_NO_SOLUTION). G(1) = the total mass.
 *   moments sum c(v) v_1^k1 v_2^k2 (raw: the Euler operator (z d/dz)^k at z = 1) or sum c(v) (v_1)_k1 (v_2)_k2
 *     (falling factorials: the k-th derivative at z = 1, the factorial moments)
 *   euler (z_c d/dz_c)^k G: coefficients c(v) v_c^k; derivative d^k/dz_c^k G: coefficients c(v) (v_c)_k, the shift
 *     decreases by k (Laurent); terms that vanish are dropped
 *   products G_a G_b: the convolution of the coefficients (the distribution of the sum of independent draws, masses
 *     multiplied), shifts added; powers G^k (k-fold convolution: sums of k i.i.d. draws), sequential products;
 *     optionally truncated to exponents e < order (one variable): the truncated series P mod z^order
 *   series P(z) mod z^order as an FND-009 truncated series (tr_series, one variable); truncation; dense <-> sparse;
 *     the marginal G(z_c, 1) of a 2-D state
 * Products accumulate in 128 bits when every input coefficient fits int64 and min(terms) max|a| max|b| < 2^127 (the
 * shared tr_gen_conv_one), else in big integers; both are exact and give the same state. A box (dense accumulation)
 * is used when it fits the support budget and is at most 4 x pairs + 65536 cells, else the pairs are sorted.
 *
 * Budgets (math_budgets.h), checked before allocation: variables = 1 or 2; support = the stored terms (dense: the box);
 * tensor = the coefficient pairs of a product; degree = the order of a moment / Euler / derivative request; bytes =
 * every allocation, coefficients counted as TR_GEN_BIG_BYTES(limbs) with limb bounds from the inputs (a product's
 * coefficient: L_a + L_b + 1 limbs; a power's chain step by step: its largest step is checked before the first
 * product). Over budget: TR_MATH_BUDGET with the tr_math_refusal. A value outside int64 (shifts, exponent sums) is
 * TR_MATH_OVERFLOW.
 *
 * CUDA (math_gen_gpu.cu): tr_mgpu_gen_conv (one work item per output coefficient of the dense convolution of two int64
 * arrays, an exact 128-bit sum: the host proves min(na, nb) max|a| max|b| < 2^127, else TR_MATH_OVERFLOW) and
 * tr_mgpu_gen_eval (one work item per rational point: the checked 128-bit homogeneous Horner of a 1-D dense state,
 * each point's own status: 0, TR_MATH_OVERFLOW past int128, TR_MATH_NO_SOLUTION at a pole, TR_MATH_INVALID for
 * q < 1). Their CPU references tr_gen_conv_i128 / tr_gen_eval_i128 run the same inline functions, so both equal bit
 * for bit; where the 128-bit path succeeds it equals the exact big-integer result.
 *
 * Errors: TR_MATH_INVALID, _MEMORY, _OVERFLOW, _BUDGET, _DOMAIN_ERROR, _NO_SOLUTION, _UNSUPPORTED (a histogram of 3
 * coordinates, a 2-D series), TR_MATH_DEVICE. Constructors return NULL and set *error; objects are immutable. */
#ifndef TORBLES_MATH_GEN_H
#define TORBLES_MATH_GEN_H
#include <stdint.h>
#include "math_num.h"
#include "math_budgets.h"
#include "math_dist.h"
#ifdef __cplusplus
extern "C" {
#endif

#define TR_GEN_ABI 1
#define TR_GEN_MAX_VARS 2
#define TR_GEN_BIG_BYTES(limbs) (32 + 8 * (uint64_t)(limbs))   /* an FND-009 big integer of `limbs` limbs */
#define TR_GEN_BOX_FACTOR 4                                    /* a product uses a box of at most 4 x pairs */
#define TR_GEN_BOX_SLACK 65536                                 /*   + 65536 cells (else it sorts the pairs) */
enum { TR_GEN_DENSE = 1, TR_GEN_SPARSE = 2 };

/* a state's description (152 bytes) */
typedef struct {
    uint64_t nterms, nonzero, box;          /* stored terms, nonzero terms, box cells (prod span) */
    uint64_t span[2];                       /* the box: exponents e_c in [0, span_c) */
    int64_t shift[2];                       /* G = z^shift P(z); values v_c = e_c + shift_c */
    int64_t vmin[2], vmax[2];               /* extremes of the values of the nonzero terms (none: 0, -1) */
    uint64_t rows, nulls;                   /* from a histogram group: its rows, and those with a NULL coordinate */
    uint64_t maxbits, bytes;                /* the largest |coefficient|'s bit length; bytes owned */
    int32_t nvars, repr, mass, nonneg;      /* mass: TR_DIST_MASS_* (0: explicit terms); nonneg: every c >= 0 */
    int32_t scale[2], wscale, reserved;
} tr_gen_info;

/* a rational point p / q (16 bytes) and a 128-bit value num / den with its status (40 bytes) */
typedef struct { int64_t num, den; } tr_gen_point;
typedef struct { int32_t status, reserved; tr_math_i128 num, den; } tr_gen_value;

/* ---------------------------------------------------------------- checked 128-bit arithmetic (CPU and CUDA) */
TR_NUM_HD int tr_gen_mul128(__int128 a, __int128 b, __int128 *r) {
    tr_dist_i256 p = tr_dist_w_mul_u128(tr_dist_mag128(a), tr_dist_mag128(b));
    if (p.w[2] || p.w[3]) return 0;
    unsigned __int128 m = ((unsigned __int128)p.w[1] << 64) | p.w[0], lim = (unsigned __int128)1 << 127;
    int neg = (a < 0) != (b < 0) && m != 0;
    if (m > lim || (m == lim && !neg)) return 0;
    *r = neg ? (__int128)((unsigned __int128)0 - m) : (__int128)m;
    return 1;
}
TR_NUM_HD int tr_gen_add128(__int128 a, __int128 b, __int128 *r) {
    __int128 v = (__int128)((unsigned __int128)a + (unsigned __int128)b);
    if ((a < 0) == (b < 0) && (v < 0) != (a < 0)) return 0;
    *r = v;
    return 1;
}
/* base^e checked (square and multiply), 1 or 0 on overflow */
TR_NUM_HD int tr_gen_pow128(__int128 base, uint64_t e, __int128 *r) {
    __int128 acc = 1;
    while (e) {
        if ((e & 1) && !tr_gen_mul128(acc, base, &acc)) return 0;
        e >>= 1;
        if (e && !tr_gen_mul128(base, base, &base)) return 0;
    }
    *r = acc;
    return 1;
}

/* ---------------------------------------------------------------- the shared kernels */
/* c_k = sum_i a_i b_(k - i) of two dense int64 arrays (exact when min(na, nb) max|a| max|b| < 2^127) */
TR_NUM_HD __int128 tr_gen_conv_one(const int64_t *a, uint64_t na, const int64_t *b, uint64_t nb, uint64_t k) {
    uint64_t lo = k >= nb ? k - nb + 1 : 0, hi = k < na ? k : na - 1;
    __int128 s = 0;
    for (uint64_t i = lo; i <= hi; i++) s += (__int128)a[i] * (__int128)b[k - i];
    return s;
}
/* G(p / q) of the 1-D dense state c[0..n) (exponents e = 0..n-1) with shift s, checked in 128 bits: num / den (den > 0,
 * not reduced) and the status (0, -3 overflow, -10 a pole at z = 0, -1 q < 1) */
TR_NUM_HD void tr_gen_eval_one(const int64_t *c, uint64_t n, int64_t s, tr_gen_point z, tr_gen_value *out) {
    out->status = 0; out->reserved = 0;
    out->num = tr_i128_set(0); out->den = tr_i128_set(1);
    if (z.den < 1 || n < 1) { out->status = -1; return; }
    if (z.num == 0) {                                   /* G(0): the coefficient of v = 0, a pole for v < 0 */
        __int128 v0 = 0;
        for (uint64_t e = 0; e < n; e++) {
            if (!c[e]) continue;
            __int128 v = (__int128)e + s;
            if (v < 0) { out->status = -10; return; }
            if (v == 0) v0 = c[e];
        }
        out->num = tr_i128_set(v0);
        return;
    }
    __int128 N = c[n - 1], Q = 1, t;
    for (uint64_t e = n - 1; e-- > 0;) {                /* N = sum c_e p^e q^(n-1-e), Q = q^(n-1) */
        if (!tr_gen_mul128(Q, z.den, &Q) || !tr_gen_mul128(N, z.num, &N) || !tr_gen_mul128(c[e], Q, &t) ||
            !tr_gen_add128(N, t, &N)) { out->status = -3; return; }
    }
    __int128 f;
    uint64_t as = s < 0 ? (uint64_t)0 - (uint64_t)s : (uint64_t)s;
    if (s >= 0) {                                       /* p^s N / (q^s Q) */
        if (!tr_gen_pow128(z.num, as, &f) || !tr_gen_mul128(N, f, &N) || !tr_gen_pow128(z.den, as, &f) ||
            !tr_gen_mul128(Q, f, &Q)) { out->status = -3; return; }
    } else {                                            /* q^-s N / (p^-s Q) */
        if (!tr_gen_pow128(z.den, as, &f) || !tr_gen_mul128(N, f, &N) || !tr_gen_pow128(z.num, as, &f) ||
            !tr_gen_mul128(Q, f, &Q)) { out->status = -3; return; }
        if (Q < 0) {
            if (!tr_gen_mul128(N, -1, &N) || !tr_gen_mul128(Q, -1, &Q)) { out->status = -3; return; }
        }
    }
    out->num = tr_i128_set(N); out->den = tr_i128_set(Q);
}
/* 1 when the 128-bit convolution of a and b is exact: min(na, nb) max|a| max|b| < 2^127 */
static inline int tr_gen_conv_exact(const int64_t *a, uint64_t na, const int64_t *b, uint64_t nb) {
    uint64_t ma = 0, mb = 0;
    for (uint64_t i = 0; i < na; i++) if (tr_abs_u64(a[i]) > ma) ma = tr_abs_u64(a[i]);
    for (uint64_t i = 0; i < nb; i++) if (tr_abs_u64(b[i]) > mb) mb = tr_abs_u64(b[i]);
    tr_dist_i256 p = tr_dist_w_mul_u128((unsigned __int128)ma * mb, (unsigned __int128)(na < nb ? na : nb));
    return !p.w[2] && !p.w[3] && !(p.w[1] >> 63);
}

/* ---------------------------------------------------------------- objects (libalg) */
typedef struct tr_gen_poly_s tr_gen_poly;
struct tr_bigint_s;
struct tr_series_s;

/* the generating polynomial of group g of a 1-D or 2-D histogram; shift NULL: the smallest present value per variable */
tr_gen_poly *tr_gen_from_hist(const tr_dist_hist *, uint64_t group, int32_t mass, int32_t repr, const int64_t *shift,
                              const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error);
/* from n explicit terms: values [n x nvars], coefficients (duplicates add, zeros dropped) */
tr_gen_poly *tr_gen_from_terms(int32_t nvars, uint64_t n, const int64_t *values, const tr_math_i128 *coef, int32_t repr,
                               const int64_t *shift, const tr_math_budget *budget, tr_math_refusal *refusal,
                               int32_t *error);
/* a 1-D state from a dense coefficient array (exponents 0..n-1, the CUDA convolution's output) with that shift */
tr_gen_poly *tr_gen_from_dense_i128(int64_t shift, uint64_t n, const tr_math_i128 *coef, int32_t repr,
                                    const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error);
int32_t tr_gen_get_info(const tr_gen_poly *, tr_gen_info *);
/* every nonzero term: values [nonzero x nvars], signs, limb offsets [nonzero + 1] and magnitude limbs */
int32_t tr_gen_export_size(const tr_gen_poly *, uint64_t *nonzero, uint64_t *limbs);
int32_t tr_gen_export(const tr_gen_poly *, int64_t *values, int32_t *signs, uint64_t *loff, uint64_t *limbs,
                      uint64_t nonzero_capacity, uint64_t limb_capacity);
/* exact answers as FND-009 big integers owned by the caller */
int32_t tr_gen_coeff(const tr_gen_poly *, const int64_t *value, const tr_math_budget *, tr_math_refusal *,
                     struct tr_bigint_s **out);
int32_t tr_gen_eval(const tr_gen_poly *, const int64_t *num, const int64_t *den, const tr_math_budget *,
                    tr_math_refusal *, struct tr_bigint_s **onum, struct tr_bigint_s **oden);
int32_t tr_gen_moment(const tr_gen_poly *, const uint32_t *k, int32_t falling, const tr_math_budget *,
                      tr_math_refusal *, struct tr_bigint_s **out);
tr_gen_poly *tr_gen_euler(const tr_gen_poly *, int32_t coord, uint32_t k, const tr_math_budget *, tr_math_refusal *,
                          int32_t *error);
tr_gen_poly *tr_gen_derivative(const tr_gen_poly *, int32_t coord, uint32_t k, const tr_math_budget *,
                               tr_math_refusal *, int32_t *error);
/* products and powers; order 0: complete, else (one variable) only the exponents e < order */
tr_gen_poly *tr_gen_mul(const tr_gen_poly *, const tr_gen_poly *, uint64_t order, const tr_math_budget *,
                        tr_math_refusal *, int32_t *error);
tr_gen_poly *tr_gen_pow(const tr_gen_poly *, uint32_t k, uint64_t order, const tr_math_budget *, tr_math_refusal *,
                        int32_t *error);
/* what they check before allocating (a power: its largest step); *terms (may be NULL): the result's term bound */
int32_t tr_gen_mul_need(const tr_gen_poly *, const tr_gen_poly *, uint64_t order, const tr_math_budget *,
                        tr_math_need *need, uint64_t *terms);
int32_t tr_gen_pow_need(const tr_gen_poly *, uint32_t k, uint64_t order, const tr_math_budget *, tr_math_need *need,
                        uint64_t *terms);
tr_gen_poly *tr_gen_truncate(const tr_gen_poly *, uint64_t order, const tr_math_budget *, tr_math_refusal *,
                             int32_t *error);
struct tr_series_s *tr_gen_to_series(const tr_gen_poly *, uint32_t order, const tr_math_budget *, tr_math_refusal *,
                                     int32_t *error);
tr_gen_poly *tr_gen_convert(const tr_gen_poly *, int32_t repr, const tr_math_budget *, tr_math_refusal *,
                            int32_t *error);
tr_gen_poly *tr_gen_marginal(const tr_gen_poly *, int32_t keep, const tr_math_budget *, tr_math_refusal *,
                             int32_t *error);
/* 1 when both are the same Laurent polynomial (the same values and coefficients; representation, shift and box aside) */
int32_t tr_gen_equal(const tr_gen_poly *, const tr_gen_poly *);
/* a 1-D state's coefficients over its box as int64 (TR_MATH_OVERFLOW when one does not fit); capacity >= span */
int32_t tr_gen_dense_i64(const tr_gen_poly *, int64_t *out, uint64_t capacity);
void tr_gen_free(tr_gen_poly *);
/* the CPU references of the kernels: out[na + nb - 1] (TR_MATH_OVERFLOW when the 128-bit proof fails; budget: output =
 * na + nb - 1), and m points (budget: output = m), each with its own status */
int32_t tr_gen_conv_i128(const int64_t *a, uint64_t na, const int64_t *b, uint64_t nb, tr_math_i128 *out,
                         const tr_math_budget *, tr_math_refusal *);
int32_t tr_gen_eval_i128(const int64_t *c, uint64_t n, int64_t shift, const tr_gen_point *z, uint64_t m,
                         tr_gen_value *out, const tr_math_budget *, tr_math_refusal *);

/* ---------------------------------------------------------------- the CUDA library (math_gen_gpu.cu) */
/* the same results on the device; budget: output (coefficients / points), bytes = the device copies */
int tr_mgpu_gen_conv(const int64_t *a, uint64_t na, const int64_t *b, uint64_t nb, tr_math_i128 *out,
                     const tr_math_budget *budget, tr_math_refusal *refusal);
int tr_mgpu_gen_eval(const int64_t *c, uint64_t n, int64_t shift, const tr_gen_point *z, uint64_t m, tr_gen_value *out,
                     const tr_math_budget *budget, tr_math_refusal *refusal);
#ifdef __cplusplus
}
#endif
#endif
