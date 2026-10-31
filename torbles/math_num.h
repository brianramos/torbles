/* Exact numeric carriers shared by the native mathematical modules, on the CPU and inside CUDA kernels.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Integers: int64 values with 128-bit intermediates; every operation reports overflow instead of wrapping.
 * Rationals: canonical (numerator, denominator > 0, gcd 1) pairs of int64, the representation of mathematics.h.
 * Wide sums: tr_math_i128 (two's complement in two words) for exact accumulation past 2^63.
 * Intervals: [lo, hi] doubles that contain the exact real result, widened by one ulp per operation (outward
 *   rounding); a floating answer is only ever returned with such an enclosure.
 * Error codes extend mathematics.h / math_domain.h; codes are never reused with another meaning. */
#ifndef TORBLES_MATH_NUM_H
#define TORBLES_MATH_NUM_H
#include <stdint.h>
#include <math.h>
#include <float.h>

#if defined(__CUDACC__) && !defined(EMULATE)
#define TR_NUM_HD __host__ __device__ static inline
#else
#define TR_NUM_HD static inline
#endif

/* error codes after mathematics.h (0, -1, -2, -3, -5) and math_domain.h (-6, -7, -8) */
enum { TR_MATH_SINGULAR = -9,        /* a system has no unique solution (singular, rank deficient) */
       TR_MATH_NO_SOLUTION = -10,    /* an inverse / preimage query has no answer in its declared domain */
       TR_MATH_DEVICE = -11,         /* the CUDA device failed or is absent (never a silent CPU fallback) */
       TR_MATH_UNSUPPORTED = -12,    /* the request is outside the capability's declared scope */
       TR_MATH_STALE = -13,          /* a state's source changed (revision, rows or scope) since the state was built */
       TR_MATH_NOT_CONVERGED = -14,  /* an iterative method did not certify its result within its iteration budget */
       TR_MATH_INFEASIBLE = -15,     /* the constraints of an optimization or search admit no point */
       TR_MATH_UNBOUNDED = -16 };    /* an objective is unbounded below on its declared domain */

typedef struct { int64_t lo; int64_t hi; } tr_math_i128;   /* value = hi * 2^64 + (uint64_t)lo */
typedef struct { double lo, hi; } tr_math_interval;          /* lo <= exact value <= hi */

/* ---------------------------------------------------------------- checked int64 */
TR_NUM_HD int tr_add_i64(int64_t a, int64_t b, int64_t *r) {
    __int128 s = (__int128)a + b;
    if (s > INT64_MAX || s < INT64_MIN) return 0;
    *r = (int64_t)s; return 1;
}
TR_NUM_HD int tr_sub_i64(int64_t a, int64_t b, int64_t *r) {
    __int128 s = (__int128)a - b;
    if (s > INT64_MAX || s < INT64_MIN) return 0;
    *r = (int64_t)s; return 1;
}
TR_NUM_HD int tr_mul_i64(int64_t a, int64_t b, int64_t *r) {
    __int128 p = (__int128)a * b;
    if (p > INT64_MAX || p < INT64_MIN) return 0;
    *r = (int64_t)p; return 1;
}
TR_NUM_HD uint64_t tr_gcd_u64(uint64_t a, uint64_t b) {
    while (b) { uint64_t t = a % b; a = b; b = t; }
    return a;
}
TR_NUM_HD uint64_t tr_abs_u64(int64_t a) { return a < 0 ? (uint64_t)0 - (uint64_t)a : (uint64_t)a; }

/* ---------------------------------------------------------------- 128-bit accumulators */
TR_NUM_HD __int128 tr_i128_get(tr_math_i128 v) { return (__int128)(((unsigned __int128)(uint64_t)v.hi << 64) | (uint64_t)v.lo); }
TR_NUM_HD tr_math_i128 tr_i128_set(__int128 v) { tr_math_i128 r; r.lo = (int64_t)(uint64_t)v; r.hi = (int64_t)(v >> 64); return r; }
TR_NUM_HD int tr_i128_fits_i64(tr_math_i128 v) { __int128 x = tr_i128_get(v); return x >= INT64_MIN && x <= INT64_MAX; }

/* ---------------------------------------------------------------- canonical rationals */
typedef struct { int64_t numerator, denominator; } tr_num_rational;  /* same layout as tr_math_rational */

/* canonical n/d from 128-bit parts: reduce, sign into the numerator; 0 on overflow or d == 0 */
TR_NUM_HD int tr_q_make128(__int128 n, __int128 d, tr_num_rational *r) {
    if (d == 0) return 0;
    if (d < 0) { n = -n; d = -d; }
    unsigned __int128 a = n < 0 ? (unsigned __int128)(-n) : (unsigned __int128)n, b = (unsigned __int128)d;
    unsigned __int128 x = a, y = b;
    while (y) { unsigned __int128 t = x % y; x = y; y = t; }
    if (x > 1) { n /= (__int128)x; d /= (__int128)x; }
    if (n > INT64_MAX || n < INT64_MIN || d > INT64_MAX) return 0;
    r->numerator = (int64_t)n; r->denominator = (int64_t)d; return 1;
}
TR_NUM_HD int tr_q_make(int64_t n, int64_t d, tr_num_rational *r) { return tr_q_make128(n, d, r); }
TR_NUM_HD int tr_q_add(tr_num_rational a, tr_num_rational b, tr_num_rational *r) {
    return tr_q_make128((__int128)a.numerator * b.denominator + (__int128)b.numerator * a.denominator,
                        (__int128)a.denominator * b.denominator, r);
}
TR_NUM_HD int tr_q_sub(tr_num_rational a, tr_num_rational b, tr_num_rational *r) {
    return tr_q_make128((__int128)a.numerator * b.denominator - (__int128)b.numerator * a.denominator,
                        (__int128)a.denominator * b.denominator, r);
}
TR_NUM_HD int tr_q_mul(tr_num_rational a, tr_num_rational b, tr_num_rational *r) {
    return tr_q_make128((__int128)a.numerator * b.numerator, (__int128)a.denominator * b.denominator, r);
}
TR_NUM_HD int tr_q_div(tr_num_rational a, tr_num_rational b, tr_num_rational *r) {
    if (b.numerator == 0) return 0;
    return tr_q_make128((__int128)a.numerator * b.denominator, (__int128)a.denominator * b.numerator, r);
}
TR_NUM_HD int tr_q_cmp(tr_num_rational a, tr_num_rational b) {   /* -1, 0, 1 (denominators positive) */
    __int128 l = (__int128)a.numerator * b.denominator, r = (__int128)b.numerator * a.denominator;
    return l < r ? -1 : l > r ? 1 : 0;
}
TR_NUM_HD tr_num_rational tr_q_int(int64_t v) { tr_num_rational r; r.numerator = v; r.denominator = 1; return r; }

/* ---------------------------------------------------------------- outward-rounded intervals */
/* the next representable double below / above x (bit manipulation: the same on the host and in CUDA device code) */
TR_NUM_HD double tr_step(double x, int up) {
    union { double d; int64_t i; } u; u.d = x;
    if (x != x) return x;
    if (x == 0) { u.i = 1; return up ? u.d : -u.d; }              /* the smallest subnormal */
    if ((x > 0) == (up != 0)) { if (u.i == INT64_MAX || u.i == (int64_t)0x7ff0000000000000LL) return x; u.i++; }
    else u.i--;
    if (x < 0 && (u.i & INT64_MAX) == 0) return up ? -0.0 : x;     /* stepping a negative value to zero */
    return u.d;
}
TR_NUM_HD double tr_down(double x) { return x == -INFINITY ? x : tr_step(x, 0); }
TR_NUM_HD double tr_up(double x) { return x == INFINITY ? x : tr_step(x, 1); }
TR_NUM_HD tr_math_interval tr_iv(double lo, double hi) { tr_math_interval r; r.lo = lo; r.hi = hi; return r; }
/* an exact int64 or rational as an enclosing interval */
TR_NUM_HD tr_math_interval tr_iv_int(int64_t v) { double d = (double)v; return tr_iv((int64_t)d == v ? d : tr_down(d), (int64_t)d == v ? d : tr_up(d)); }
TR_NUM_HD tr_math_interval tr_iv_q(tr_num_rational q) {
    double n = (double)q.numerator, d = (double)q.denominator, x = n / d;
    return tr_iv(tr_down(tr_down(x)), tr_up(tr_up(x)));         /* conversion of n and d, and the division */
}
TR_NUM_HD tr_math_interval tr_iv_add(tr_math_interval a, tr_math_interval b) { return tr_iv(tr_down(a.lo + b.lo), tr_up(a.hi + b.hi)); }
TR_NUM_HD tr_math_interval tr_iv_sub(tr_math_interval a, tr_math_interval b) { return tr_iv(tr_down(a.lo - b.hi), tr_up(a.hi - b.lo)); }
TR_NUM_HD tr_math_interval tr_iv_mul(tr_math_interval a, tr_math_interval b) {
    double p[4] = {a.lo * b.lo, a.lo * b.hi, a.hi * b.lo, a.hi * b.hi};
    double lo = p[0], hi = p[0];
    for (int i = 1; i < 4; i++) { if (p[i] < lo) lo = p[i]; if (p[i] > hi) hi = p[i]; }
    return tr_iv(tr_down(lo), tr_up(hi));
}
TR_NUM_HD int tr_iv_contains_zero(tr_math_interval a) { return a.lo <= 0 && a.hi >= 0; }
TR_NUM_HD tr_math_interval tr_iv_div(tr_math_interval a, tr_math_interval b) {   /* b must not contain 0 */
    double q[4] = {a.lo / b.lo, a.lo / b.hi, a.hi / b.lo, a.hi / b.hi};
    double lo = q[0], hi = q[0];
    for (int i = 1; i < 4; i++) { if (q[i] < lo) lo = q[i]; if (q[i] > hi) hi = q[i]; }
    return tr_iv(tr_down(lo), tr_up(hi));
}
TR_NUM_HD tr_math_interval tr_iv_sqrt(tr_math_interval a) {   /* a.lo >= 0 */
    return tr_iv(a.lo <= 0 ? 0.0 : tr_down(sqrt(a.lo)), tr_up(sqrt(a.hi)));
}
#endif
