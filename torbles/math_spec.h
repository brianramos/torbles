/* SPEC-002: exact Walsh-Hadamard spectra of bounded Boolean histograms (math_lattice.h projections, or any vector of
 * 2^k exact integers): forward and inverse transforms under a declared normalization, Parseval identities, parity
 * correlations of condition subsets, coefficient truncation as an explicit approximation with its exact error, and
 * the GF(2) cross-check of constant parities with FIELD-001 (math_field.h), on the CPU (math_spec.c) and on a CUDA
 * device (math_spec_gpu.cu: the butterfly passes), with the same results bit for bit.
 * SPDX-License-Identifier: Apache-2.0
 *
 * ---- Normalization (declared) ---------------------------------------------------------------------------------
 * For a vector v over the 2^k subsets x of k conditions (bit i = condition i; a histogram's cell counts) and the
 * characters chi_S(x) = (-1)^{|S and x|}:
 *   forward   W(S) = sum_x v(x) chi_S(x)               exact integers (unnormalized)
 *   inverse   v(x) = 2^-k sum_S W(S) chi_S(x)          exact dyadic rationals num / 2^d, canonical (num odd or d = 0;
 *                                                      0 = 0 / 2^0): integers when W is a forward transform of integers
 *   Parseval  sum_S W(S)^2 = 2^k sum_x v(x)^2          (tr_spec_parseval: both sides exact in 256 bits)
 * With the probability normalization p = v / N (N = W(empty) = sum_x v(x)): the Fourier coefficient is W(S) / N and
 * sum_S (W(S) / N)^2 = 2^k sum_x p(x)^2: the same identity divided by N^2.
 * The fast transform: k butterfly passes, pass j maps each pair (x, x + 2^j), bit j clear in x, to (a + b, a - b);
 * a pass's pairs are disjoint, the passes run in order (one launch each on a device). Its cost is the full domain
 * whatever the occupancy: 2^k states, k passes, k 2^(k-1) butterflies (tr_spec_cost); a row count never enters it.
 *
 * ---- Exactness: signed values never wrap ------------------------------------------------------------------------
 * Values are tr_math_i128; every butterfly is checked (an overflow is TR_MATH_OVERFLOW and nothing is written). For
 * inputs with sum_x |v(x)| < 2^63 (every count histogram) every intermediate and output is bounded by that sum; for
 * any int64 inputs by 2^(k+63) < 2^127. Energies are accumulated in 256 bits (TR_MATH_OVERFLOW past it).
 *
 * ---- Parity correlations (tr_spec_parity) ---------------------------------------------------------------------
 * For a set S of conditions on a weight vector (counts) with total N: coefficient W(S); even = (N + W(S)) / 2 and
 * odd = (N - W(S)) / 2, the weight of the points whose conditions in S have an even / odd number of TRUE values
 * (exact for integer weights: N and W(S) have the same parity); correlation = W(S) / N canonical (num / den, den > 0,
 * gcd 1; 0 / 1 when N = 0) = E[chi_S] under the normalized weights = P(even) - P(odd). tr_spec_correlations reads a
 * full spectrum (W(empty) = N); tr_spec_coefficients computes the requested W(S) and N directly from the values
 * (2^k terms each: the construction of new frequencies without the full transform).
 *
 * ---- Truncation: an explicit approximation (tr_spec_truncate) ---------------------------------------------------
 * Keeping the m coefficients of largest |W(S)| (ties: the smaller S first) is an approximation v_m(x) = 2^-k
 * sum_{S kept} W(S) chi_S(x), never presented as exact: the result states the full domain (states = 2^k), what was
 * kept, and the exact error: residual = sum_{S dropped} W(S)^2 (256-bit), so sum_x (v(x) - v_m(x))^2 = residual / 2^k
 * (Parseval, given reduced as l2_num / 2^l2_den_log2), and max_x |v(x) - v_m(x)| <= sum_{S dropped} |W(S)| / 2^k
 * (bound_num / 2^bound_den_log2). exact = 1 only when every dropped coefficient is 0. tr_spec_evaluate evaluates a
 * kept expansion at points (the fixed-basis actions: m terms per point, exact dyadic values).
 *
 * ---- Constant parities are FIELD-001 parity dependencies (tr_spec_constant_parities) ----------------------------
 * On non-negative weights with N > 0 (a count histogram), chi_S is constant on the support (the occupied cells)
 * exactly when |W(S)| = N: W(S) = N (XOR of the conditions of S is 0 on every point) or W(S) = -N (it is 1). These
 * are the affine GF(2) relations of the occupied cells: the native cross-check inserts every occupied cell as the
 * vector [1, x_0, .., x_{k-1}] into FIELD-001's echelon basis (tr_field_basis_insert), takes its canonical relation
 * basis (tr_field_basis_relations: vars, constant), enumerates the span and reports agree = 1 when the span is exactly
 * the set found from the spectrum (2^nbasis relations, the empty one included). rank = the affine rank of the support.
 *
 * ---- Budgets (math_budgets.h), checked before anything is allocated --------------------------------------------
 *   walsh / inverse (tr_spec_need): variables = k, tensor = 2^k (the states), iterations = k (passes), bytes = 16 2^k
 *     (the scratch; on a device its copy).
 *   coefficients: variables = k, tensor = n 2^k (terms), output = n, bytes = 96 n.   correlations: nothing allocated.
 *   truncate: variables = k, tensor = 2^k, output = m, bytes = 24 2^k (the order).   evaluate: tensor = m x npoints,
 *     output = npoints, bytes = 24 npoints.   constant parities: variables = k, tensor = 2^k, iterations = k, output =
 *     2^k (the relations' capacity), bytes = 17 2^k (the spectrum and the span marks).   parseval: nothing allocated.
 * Representation limits (with a budget that admits them): TR_MATH_UNSUPPORTED for k > TR_SPEC_MAX_K (30).
 *
 * Errors: TR_MATH_INVALID (-1: malformed input, a capacity too small), _MEMORY (-2), _OVERFLOW (-3), _BUDGET (-5),
 * TR_MATH_DOMAIN_ERROR (-6: constant parities of a vector with a negative entry or without weight), TR_MATH_DEVICE
 * (-11), TR_MATH_UNSUPPORTED (-12). Functions only read their inputs and write their outputs, and write them only on
 * success. */
#ifndef TORBLES_MATH_SPEC_H
#define TORBLES_MATH_SPEC_H
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "math_num.h"
#include "math_budgets.h"
#ifdef __cplusplus
extern "C" {
#endif

#define TR_SPEC_ABI 1
#define TR_SPEC_MAX_K 30
enum { TR_SPEC_FORWARD = 0, TR_SPEC_INVERSE = 1 };       /* the normalization a cost describes */

typedef struct { tr_math_i128 num; int64_t den_log2; } tr_spec_dyadic;              /* 24 bytes: num / 2^den_log2 */
typedef struct {
    uint64_t states;          /* 2^k: the full domain */
    uint64_t passes;          /* k */
    uint64_t butterflies;     /* k 2^(k-1) */
    uint64_t bytes;           /* the scratch (host) or the device copy */
    int32_t k, normalization; /* TR_SPEC_FORWARD (unnormalized) / TR_SPEC_INVERSE (divided by 2^k) */
} tr_spec_cost;               /* 40 bytes */
typedef struct {
    uint64_t cells[4];        /* sum_x v(x)^2, 256-bit, least significant word first */
    uint64_t spectrum[4];     /* sum_S W(S)^2 */
    int32_t k, holds;         /* holds: spectrum == 2^k cells exactly */
} tr_spec_energy;             /* 72 bytes */
typedef struct {
    uint64_t vars;            /* S */
    int32_t order, reserved;  /* |S| */
    tr_math_i128 coefficient; /* W(S) */
    tr_math_i128 even, odd;   /* (N + W(S)) / 2, (N - W(S)) / 2 */
    tr_math_i128 corr_num, corr_den;   /* W(S) / N canonical */
} tr_spec_parity;             /* 96 bytes */
typedef struct {
    uint64_t states, kept, nonzero, dropped;   /* 2^k; coefficients kept; nonzero in the spectrum; nonzero dropped */
    uint64_t residual[4], total[4];            /* sum over the dropped / all S of W(S)^2 (256-bit) */
    uint64_t l2_num[4];                        /* residual / 2^k reduced: sum_x (v - v_m)^2 = l2_num / 2^l2_den_log2 */
    tr_math_i128 residual_l1;                  /* sum over the dropped S of |W(S)| */
    tr_math_i128 bound_num;                    /* residual_l1 / 2^k reduced: max_x |v - v_m| <= bound */
    int32_t l2_den_log2, bound_den_log2;
    int32_t k, exact;                          /* exact: no nonzero coefficient was dropped */
} tr_spec_truncation;         /* 176 bytes */
typedef struct { uint64_t vars; int32_t constant, reserved; } tr_spec_relation;     /* 16 bytes: XOR over vars = c */
typedef struct {
    tr_math_i128 total;       /* N = W(empty) */
    uint64_t support;         /* occupied cells */
    uint64_t constant;        /* sets S with |W(S)| = N (the relations from the spectrum) */
    int32_t k, rank;          /* the affine GF(2) rank of the support (FIELD-001) */
    int32_t nbasis, agree;    /* relations in the FIELD-001 basis; 1 when its span is exactly the constant set */
} tr_spec_parities;           /* 48 bytes */

/* ------------------------------------------------------------------ shared helpers (CPU code, CUDA host and kernels) */
TR_NUM_HD int tr_spec_add128(__int128 a, __int128 b, __int128 *r) {
    __int128 v = (__int128)((unsigned __int128)a + (unsigned __int128)b);
    if ((a >= 0) == (b >= 0) && (v >= 0) != (a >= 0)) return 0;
    *r = v; return 1;
}
TR_NUM_HD int tr_spec_sub128(__int128 a, __int128 b, __int128 *r) {
    __int128 v = (__int128)((unsigned __int128)a - (unsigned __int128)b);
    if ((a >= 0) != (b >= 0) && (v >= 0) != (a >= 0)) return 0;
    *r = v; return 1;
}
/* one Walsh-Hadamard butterfly on the pair (lo, hi): (a, b) -> (a + b, a - b); 0 on overflow (cells then unspecified) */
TR_NUM_HD int tr_spec_butterfly(tr_math_i128 *v, uint64_t lo, uint64_t hi) {
    __int128 a = tr_i128_get(v[lo]), b = tr_i128_get(v[hi]), s, d;
    if (!tr_spec_add128(a, b, &s) || !tr_spec_sub128(a, b, &d)) return 0;
    v[lo] = tr_i128_set(s); v[hi] = tr_i128_set(d);
    return 1;
}
/* the pair p (0 <= p < 2^(k-1)) of pass j: lo has bit j clear */
TR_NUM_HD uint64_t tr_spec_pair_lo(uint64_t p, int32_t j) {
    return ((p >> j) << (j + 1)) | (p & (((uint64_t)1 << j) - 1));
}
/* v / 2^k as a canonical dyadic rational */
TR_NUM_HD tr_spec_dyadic tr_spec_dyadic_of(__int128 v, int32_t k) {
    tr_spec_dyadic r;
    int32_t t = 0;
    if (v == 0) { r.num = tr_i128_set(0); r.den_log2 = 0; return r; }
    unsigned __int128 m = (unsigned __int128)v;
    while (t < k && !(m & 1)) { m >>= 1; t++; }
    r.num = tr_i128_set(v >> t);                 /* exact: the low t bits are zero */
    r.den_log2 = k - t;
    return r;
}
/* the cost of one transform over 2^k values */
TR_NUM_HD void tr_spec_cost_of(int32_t k, int32_t normalization, uint64_t bytes, tr_spec_cost *c) {
    c->states = (uint64_t)1 << k; c->passes = (uint64_t)k;
    c->butterflies = k ? (uint64_t)k << (k - 1) : 0; c->bytes = bytes;
    c->k = k; c->normalization = normalization;
}
/* the needs of a forward or inverse transform (see Budgets) */
static inline int32_t tr_spec_base_need(int32_t k, tr_math_need *need) {
    if (k < 0 || !need) return -1;
    for (int i = 0; i < TR_BUDGET_RESOURCES; i++) need->need[i] = 0;
    uint64_t n = tr_budget_power2((uint32_t)k);
    need->need[TR_BUDGET_VARIABLES] = (uint64_t)k;
    need->need[TR_BUDGET_TENSOR] = n;
    need->need[TR_BUDGET_ITERATIONS] = (uint64_t)k;
    need->need[TR_BUDGET_BYTES] = tr_budget_mul(n, 16);
    return 0;
}

/* ------------------------------------------------------------------ transforms (libalg) */
int32_t tr_spec_need(int32_t k, tr_math_need *);
/* out[2^k] = W (forward, unnormalized); out may equal in; cost may be NULL */
int32_t tr_spec_walsh(int32_t k, const tr_math_i128 *in, tr_math_i128 *out, const tr_math_budget *, tr_spec_cost *cost,
                      tr_math_refusal *);
/* out[2^k] = 2^-k H in, canonical dyadic rationals */
int32_t tr_spec_walsh_inverse(int32_t k, const tr_math_i128 *in, tr_spec_dyadic *out, const tr_math_budget *,
                              tr_spec_cost *cost, tr_math_refusal *);
/* the energies of a vector and a spectrum of 2^k values and whether Parseval's identity holds between them */
int32_t tr_spec_parseval(int32_t k, const tr_math_i128 *values, const tr_math_i128 *spectrum, tr_spec_energy *out);
/* correlations of the requested sets from a full spectrum (N = spectrum[0]) */
int32_t tr_spec_correlations(int32_t k, const tr_math_i128 *spectrum, const uint64_t *sets, int32_t n,
                             tr_spec_parity *out);
/* the same with W(S) and N summed directly from the values (no full transform) */
int32_t tr_spec_coefficients(int32_t k, const tr_math_i128 *values, const uint64_t *sets, int32_t n,
                             const tr_math_budget *, tr_spec_parity *out, tr_math_refusal *);
int32_t tr_spec_coefficients_need(int32_t k, const uint64_t *sets, int32_t n, tr_math_need *);
/* keeps the m largest coefficients: sets[m], coef[m] (in the kept order), out (the error); m <= 2^k */
int32_t tr_spec_truncate(int32_t k, const tr_math_i128 *spectrum, uint64_t m, const tr_math_budget *, uint64_t *sets,
                         tr_math_i128 *coef, tr_spec_truncation *out, tr_math_refusal *);
/* out[i] = 2^-k sum_j coef[j] chi_{sets[j]}(points[i]) (points < 2^k) */
int32_t tr_spec_evaluate(int32_t k, const uint64_t *sets, const tr_math_i128 *coef, uint64_t m, const uint64_t *points,
                         uint64_t npoints, const tr_math_budget *, tr_spec_dyadic *out, tr_math_refusal *);
/* the relations from the spectrum (relations[capacity >= out->constant], increasing S; capacity 2^k always
 * suffices) and the FIELD-001 basis (basis[capacity >= k + 1]) of non-negative weights values[2^k] */
int32_t tr_spec_constant_parities(int32_t k, const tr_math_i128 *values, const tr_math_budget *,
                                  tr_spec_relation *relations, uint64_t capacity, tr_spec_relation *basis,
                                  int32_t basis_capacity, tr_spec_parities *out, tr_math_refusal *);

/* ------------------------------------------------------------------ the CUDA library (math_spec_gpu.cu)
 * tr_mgpu_spec_walsh / _inverse: tr_spec_walsh / tr_spec_walsh_inverse on the device (the values copied once, k
 * launches of one work item per pair, an overflow flag raised by any butterfly that would wrap; the inverse's
 * canonical dyadic rationals are formed on the host from the device's integers). */
int tr_mgpu_spec_walsh(int32_t k, const tr_math_i128 *in, tr_math_i128 *out, const tr_math_budget *,
                       tr_spec_cost *cost, tr_math_refusal *);
int tr_mgpu_spec_walsh_inverse(int32_t k, const tr_math_i128 *in, tr_spec_dyadic *out, const tr_math_budget *,
                               tr_spec_cost *cost, tr_math_refusal *);
#ifdef __cplusplus
}
#endif
#endif
