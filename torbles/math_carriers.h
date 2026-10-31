/* Numeric carriers and exactness conversions (FND-009).
 * SPDX-License-Identifier: Apache-2.0
 *
 * The carriers extend the checked int64 rationals of the first slice (mathematics.h, math_num.h). Each has one law
 * and one exactness contract, and nothing converts implicitly: a binary operation takes operands of one carrier (one
 * modulus for GF(p), one order for series), and every change of carrier is an explicit call whose kind is fixed by
 * the conversion matrix below (tr_carrier_conversion).
 *
 *   carrier      law                         exact?                          stored as
 *   I64          Z, checked                  exact; overflow reported        int64_t
 *   I128         Z, checked                  exact; overflow reported        tr_math_i128 (math_num.h)
 *   BIGINT       Z                           exact; size budgeted            tr_bigint (sign + 64-bit limbs)
 *   Q64          Q, canonical, checked       exact; overflow reported        tr_num_rational (= tr_math_rational)
 *   BIGQ         Q, canonical                exact; size budgeted            tr_bigq (two tr_bigint)
 *   GFP          GF(p), prime p < 2^62       exact                           tr_gfp_elem / uint64_t residues
 *   BINARY64     the dyadic value of a finite IEEE double (exact, but no arithmetic: only converted)
 *   INTERVAL     real enclosure              APPROXIMATE, certified          tr_math_interval [lo, hi]
 *   CINTERVAL    complex enclosure           APPROXIMATE, certified          tr_carrier_cinterval (a rectangle)
 *   SERIES       Q[[x]] / x^N                exact up to O(x^N)              tr_series (BIGQ coefficients)
 *
 * Error codes (all existing ones; none is new):
 *   TR_MATH_INVALID (-1)        malformed input: NULL or not a live handle, a malformed decimal, a value >= p, ...
 *   TR_MATH_MEMORY (-2)         an allocation failed
 *   TR_MATH_OVERFLOW (-3)       narrowing: the exact value does not fit the fixed-width carrier (int64, int128, int64
 *                               rationals), or a checked fixed-width operation overflowed. Never wrapped.
 *   TR_MATH_BUDGET (-5)         a size budget refused the request before allocation (tr_math_refusal says which)
 *   TR_MATH_CARRIER_ERROR (-7)  carriers mixed (different carriers, moduli or orders, or a handle of another
 *                               carrier where one is expected) or a forbidden conversion
 *   TR_MATH_NO_SOLUTION (-10)   the exact answer does not exist in the carrier: division by zero, the inverse of 0
 *                               in GF(p), a divisor interval containing 0, the square root of an interval with a
 *                               negative part, a non-integral rational narrowed to an integer, a rational whose
 *                               denominator is 0 mod p, a series without a reciprocal / exp / log, a failed rational
 *                               reconstruction, an empty intersection
 *   TR_MATH_UNSUPPORTED (-12)   outside the declared scope (a prime p >= 2^62, ...)
 *
 * Objects (tr_bigint, tr_bigq, tr_series) are immutable once built and individually owned (free with their _free
 * function, which accepts NULL). A constructor or operation returns NULL and sets *error (error may be NULL) without
 * changing its inputs; on success *error is TR_MATH_OK. Outputs through pointers are written only on success
 * (except the output arrays of the checked Horner evaluations, see there).
 *
 * Size budgets (math_budgets.h). Every function that allocates takes `const tr_math_budget *budget` (NULL = the
 * defaults of math_budgets.h) and `tr_math_refusal *refusal` (may be NULL). Before allocating, it computes an exact
 * upper bound on the limbs of every big number it will create (its result and its scratch) and refuses with
 * TR_MATH_BUDGET (resource TR_BUDGET_BYTES, need = 8 * the largest such bound, limit = limit[TR_BUDGET_BYTES]) when it
 * exceeds the limit: limit[TR_BUDGET_BYTES] / 8 is the limb budget of one big number. The bounds (L(x) = limbs of |x|,
 * 0 for zero; bits(x) = bit length of |x|):
 *   from_i64, from_i128: 1, 2           from_limbs: n (as given)    from_twos: n
 *   from_decimal: (3322 * digits + 63999) / 64000 + 1                 copy, neg, abs: La
 *   add, sub: max(La, Lb) + 1           mul: La + Lb                divmod, divexact: max(La, Lb) + 2
 *   gcd: max(La, Lb) + 1                shl(k): La + k / 64 + 1 (0 when a = 0)     shr(k): La + 1
 *   pow(e): ceil(e * bits(a) / 64) + 1 (e = 0 or a in {0, 1, -1}: 1)  to_decimal: La (a scratch copy)
 *   isqrt, dot, horner, batch: the bounds of their elementary steps   crt: k + 1     crt_combine: Lm + 1
 * Rationals, series, CRT and rational reconstruction budget every big number they create the same way (the reported
 * need is that of the first refused number; refusals are deterministic). Series also bound their order: a series of
 * order N needs degree N - 1 (<= limit[TR_BUDGET_DEGREE]) and support N (<= limit[TR_BUDGET_SUPPORT]). A hard cap of
 * TR_BIGINT_MAX_LIMBS limbs per number applies whatever the budget (reported as the limit when it is the smaller).
 * Comparisons (tr_bigq_cmp, tr_bigq_cmp_double, tr_bigq_in_interval) and enclosures allocate scratch bounded by
 * the sizes of their inputs (which were budgeted when they were made) and take no budget.
 *
 * Intervals. A valid interval has lo <= hi, no NaN, lo != +inf and hi != -inf. Every operation returns an enclosure
 * of the exact set of results (outward rounding, never round-to-nearest alone): endpoints are the tightest doubles
 * (RD and RU of the exact endpoint, from exact TwoSum / fma error terms) for sums always, and for products,
 * quotients and square roots whenever the rounded endpoint has magnitude >= 2^-969 or overflows; below 2^-969 they
 * are stepped out by one more ulp. An infinite endpoint means "unbounded". Infinite endpoints follow the
 * convention 0 * inf = 0 (an interval [0, 0] times an unbounded one is [0, 0]). Intervals are the APPROXIMATE
 * carriers: their values are never presented as exact rationals, and no conversion from them to an exact carrier
 * exists. Complex intervals are rectangles re x im; their operations are interval evaluations of the usual formulas,
 * so they enclose the exact result (they may be wider than the true image).
 *
 * The inline TR_NUM_HD helpers (GF(p) arithmetic with Barrett reduction, the tight q64 enclosure, interval
 * arithmetic, exact 192-bit accumulation) are shared by the engine library, the CUDA kernels of math_carriers_gpu.cu
 * and later capabilities' device code; math_carriers.c exports the entry points built on them (batched GF(p),
 * tr_carrier_interval_op, ...) for other modules and the Python bindings (torbles/math_carriers.py). The interval
 * helpers are safe under floating contraction (no a * b + c expression; error terms use explicit fma), so they keep
 * their contract with GCC's default -ffp-contract and nvcc's default --fmad. */
#ifndef TORBLES_MATH_CARRIERS_H
#define TORBLES_MATH_CARRIERS_H
#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include "math_num.h"
#include "math_budgets.h"
#include "math_domain.h"
#ifdef __cplusplus
extern "C" {
#endif

#define TR_CARRIERS_ABI 1
#define TR_BIGINT_MAX_LIMBS ((uint64_t)1 << 31)     /* hard cap of one big number: 2^37 bits */
#define TR_GFP_MAX ((uint64_t)1 << 62)              /* GF(p) needs p < 2^62 */

/* ---------------------------------------------------------------- carriers, laws, conversion matrix */
enum { TR_CARRIER_I64 = 1, TR_CARRIER_I128 = 2, TR_CARRIER_BIGINT = 3, TR_CARRIER_Q64 = 4, TR_CARRIER_BIGQ = 5,
       TR_CARRIER_GFP = 6, TR_CARRIER_BINARY64 = 7, TR_CARRIER_INTERVAL = 8, TR_CARRIER_CINTERVAL = 9,
       TR_CARRIER_SERIES = 10, TR_CARRIER_LAST = 10 };
enum { TR_LAW_INTEGER = 1, TR_LAW_RATIONAL = 2, TR_LAW_PRIME_FIELD = 3, TR_LAW_DYADIC = 4,
       TR_LAW_REAL_ENCLOSURE = 5, TR_LAW_COMPLEX_ENCLOSURE = 6, TR_LAW_TRUNCATED_SERIES = 7 };
/* kinds of a conversion from -> to:
 *   FORBIDDEN    refused with TR_MATH_CARRIER_ERROR (no function performs it)
 *   IDENTITY     the same carrier (GF(p) -> GF(q) with q != p is refused at run time with TR_MATH_CARRIER_ERROR)
 *   LOSSLESS     every value converts and converts back unchanged
 *   NARROWING    exact when the value fits; otherwise TR_MATH_OVERFLOW (range) or TR_MATH_NO_SOLUTION (a non-integral
 *                rational to an integer carrier; a complex interval with a nonzero imaginary part to a real one)
 *   ENCLOSING    an exact value to its tight enclosure (lo = RD(x), hi = RU(x)); the result is APPROXIMATE
 *   REDUCTION    an explicit map into GF(p) given the field: x mod p, n * d^-1 mod p (d = 0 mod p: NO_SOLUTION)
 *   LIFT         the explicit canonical lift of a residue v in [0, p) to the integer v (or the symmetric lift)
 *   TRUNCATION   a series of order N to an order M <= N (M > N: NO_SOLUTION; the lost coefficients are unknown) */
enum { TR_CONVERT_FORBIDDEN = 0, TR_CONVERT_IDENTITY = 1, TR_CONVERT_LOSSLESS = 2, TR_CONVERT_NARROWING = 3,
       TR_CONVERT_ENCLOSING = 4, TR_CONVERT_REDUCTION = 5, TR_CONVERT_LIFT = 6, TR_CONVERT_TRUNCATION = 7 };

/* a carrier's law and exactness contract (a constant table) */
typedef struct {
    int32_t carrier, law;
    int32_t exact;            /* 1: values are exact; 0: values are enclosures (approximate) */
    int32_t certified;        /* 1: an approximate value's enclosure is rigorous (always 1 here) */
    int32_t precision_bits;   /* stored precision: 64, 128, 62 (GF(p) residues), 53 (binary64 endpoints),
                                 0 = unbounded (size budgeted; series: exact coefficients up to the order) */
    int32_t reserved;
    char name[24];
} tr_carrier_desc;
int32_t tr_carrier_describe(int32_t carrier, tr_carrier_desc *out);   /* 0, or TR_MATH_INVALID */
/* the kind of from -> to (TR_CONVERT_*), or TR_MATH_INVALID for an unknown carrier */
int32_t tr_carrier_conversion(int32_t from, int32_t to);
/* 0 when the kind is not FORBIDDEN, else TR_MATH_CARRIER_ERROR (TR_MATH_INVALID for an unknown carrier) */
int32_t tr_carrier_check_conversion(int32_t from, int32_t to);
/* binary operations combine one carrier: returns the carrier when a == b, else TR_MATH_CARRIER_ERROR */
int32_t tr_carrier_combine(int32_t a, int32_t b);
/* the sizes of tr_gfp, tr_gfp_elem, tr_carrier_cinterval, tr_carrier_desc, tr_math_interval, tr_math_i128 and
 * tr_num_rational (the first n of them) for a bindings self-check; returns TR_CARRIERS_ABI */
int32_t tr_carrier_abi(uint64_t *sizes, int32_t n);

/* operation codes of the dispatching entry points */
enum { TR_OP_ADD = 1, TR_OP_SUB = 2, TR_OP_MUL = 3, TR_OP_DIV = 4, TR_OP_NEG = 5, TR_OP_INV = 6, TR_OP_SQR = 7,
       TR_OP_SQRT = 8, TR_OP_ABS = 9, TR_OP_POW = 10, TR_OP_HULL = 11, TR_OP_INTERSECT = 12, TR_OP_CONJ = 13,
       TR_OP_GCD = 14, TR_OP_MOD = 15 };
/* rounding of integer division */
enum { TR_DIV_FLOOR = 0,     /* q = floor(a / b), r = a - q b has the sign of b (Python's // and %) */
       TR_DIV_TRUNC = 1 };   /* q = trunc(a / b), r has the sign of a (C's / and %) */
/* three-valued comparison of enclosures */
enum { TR_CMP_LESS = -1, TR_CMP_EQUAL = 0, TR_CMP_GREATER = 1, TR_CMP_UNKNOWN = 2 };

/* ---------------------------------------------------------------- checked fixed-width integers and rationals */
/* a op b with op ADD, SUB, MUL, NEG (b ignored), DIV (TR_DIV_FLOOR quotient; b == 0: NO_SOLUTION); OVERFLOW past
 * the carrier, never wrapped. *out written only on success. */
int32_t tr_carrier_i64_op(int32_t op, int64_t a, int64_t b, int64_t *out);
int32_t tr_carrier_i128_op(int32_t op, tr_math_i128 a, tr_math_i128 b, tr_math_i128 *out);
int32_t tr_carrier_i128_cmp(tr_math_i128 a, tr_math_i128 b);                 /* -1, 0, 1 */
/* canonical int64 rationals (math_num.h): op ADD, SUB, MUL, DIV, NEG, INV; inputs need not be canonical (d != 0,
 * else INVALID), the result is canonical; OVERFLOW when it does not fit; division by 0: NO_SOLUTION */
int32_t tr_carrier_q64_op(int32_t op, tr_num_rational a, tr_num_rational b, tr_num_rational *out);
int32_t tr_carrier_q64_cmp(tr_num_rational a, tr_num_rational b, int32_t *out);   /* *out -1/0/1; INVALID if d == 0 */
/* narrowing between fixed-width carriers: OVERFLOW out of range; q64 -> i64 is NO_SOLUTION unless integral */
int32_t tr_carrier_i128_to_i64(tr_math_i128 v, int64_t *out);
int32_t tr_carrier_q64_to_i64(tr_num_rational q, int64_t *out);

/* ---------------------------------------------------------------- arbitrary-precision integers */
typedef struct tr_bigint_s tr_bigint;   /* sign + magnitude in little-endian 64-bit limbs, normalized */

tr_bigint *tr_bigint_from_i64(int64_t v, const tr_math_budget *, tr_math_refusal *, int32_t *error);
tr_bigint *tr_bigint_from_i128(tr_math_i128 v, const tr_math_budget *, tr_math_refusal *, int32_t *error);
/* value = sign * sum(limbs[i] * 2^(64 i)); sign is -1, 0 or 1 (0, or an all-zero magnitude, is the value 0) */
tr_bigint *tr_bigint_from_limbs(int32_t sign, const uint64_t *limbs, uint64_t n, const tr_math_budget *,
                                tr_math_refusal *, int32_t *error);
/* n words of a two's complement integer, least significant first (e.g. the 3 words of tr_carrier_dot_wide) */
tr_bigint *tr_bigint_from_twos(const uint64_t *words, uint64_t n, const tr_math_budget *, tr_math_refusal *,
                               int32_t *error);
/* [+-]?[0-9]+ of exactly `length` bytes (no spaces, no other characters): else TR_MATH_INVALID */
tr_bigint *tr_bigint_from_decimal(const char *text, uint64_t length, const tr_math_budget *, tr_math_refusal *,
                                  int32_t *error);
int32_t tr_bigint_to_i64(const tr_bigint *, int64_t *out);            /* OVERFLOW outside [-2^63, 2^63) */
int32_t tr_bigint_to_i128(const tr_bigint *, tr_math_i128 *out);      /* OVERFLOW outside [-2^127, 2^127) */
/* the normalized magnitude limbs (count = limb count) and the sign; INVALID when capacity < the limb count */
int32_t tr_bigint_export(const tr_bigint *, int32_t *sign, uint64_t *limbs, uint64_t capacity, uint64_t *count);
/* an upper bound on the decimal length, including the sign and the terminating NUL */
uint64_t tr_bigint_decimal_size(const tr_bigint *);
/* the decimal text, NUL-terminated; *length = its length without the NUL; INVALID when capacity is too small */
int32_t tr_bigint_to_decimal(const tr_bigint *, char *out, uint64_t capacity, uint64_t *length,
                             const tr_math_budget *, tr_math_refusal *);
int32_t tr_bigint_sign(const tr_bigint *);                            /* -1, 0, 1 (INVALID for a bad handle) */
uint64_t tr_bigint_limb_count(const tr_bigint *);
uint64_t tr_bigint_bit_length(const tr_bigint *);                     /* of |x|; 0 for 0 */
int32_t tr_bigint_cmp(const tr_bigint *, const tr_bigint *);          /* -1, 0, 1 (INVALID for a bad handle) */
int32_t tr_bigint_is_handle(const void *);                            /* 1 when this is a live tr_bigint */
tr_bigint *tr_bigint_copy(const tr_bigint *, const tr_math_budget *, tr_math_refusal *, int32_t *error);
tr_bigint *tr_bigint_neg(const tr_bigint *, const tr_math_budget *, tr_math_refusal *, int32_t *error);
tr_bigint *tr_bigint_abs(const tr_bigint *, const tr_math_budget *, tr_math_refusal *, int32_t *error);
tr_bigint *tr_bigint_add(const tr_bigint *, const tr_bigint *, const tr_math_budget *, tr_math_refusal *, int32_t *error);
tr_bigint *tr_bigint_sub(const tr_bigint *, const tr_bigint *, const tr_math_budget *, tr_math_refusal *, int32_t *error);
tr_bigint *tr_bigint_mul(const tr_bigint *, const tr_bigint *, const tr_math_budget *, tr_math_refusal *, int32_t *error);
/* quotient and remainder with the rounding TR_DIV_FLOOR or TR_DIV_TRUNC; q or r may be NULL (not wanted); b == 0 is
 * TR_MATH_NO_SOLUTION. a = q b + r and |r| < |b| always. */
int32_t tr_bigint_divmod(const tr_bigint *a, const tr_bigint *b, int32_t rounding, tr_bigint **q, tr_bigint **r,
                         const tr_math_budget *, tr_math_refusal *);
/* a / b when b divides a exactly; NO_SOLUTION when it does not (or b == 0) */
tr_bigint *tr_bigint_divexact(const tr_bigint *a, const tr_bigint *b, const tr_math_budget *, tr_math_refusal *,
                              int32_t *error);
/* gcd(|a|, |b|) >= 0 (gcd(0, 0) = 0), Lehmer's algorithm */
tr_bigint *tr_bigint_gcd(const tr_bigint *, const tr_bigint *, const tr_math_budget *, tr_math_refusal *, int32_t *error);
tr_bigint *tr_bigint_shl(const tr_bigint *, uint64_t bits, const tr_math_budget *, tr_math_refusal *, int32_t *error);
/* floor(a / 2^bits) (an arithmetic shift: -1 >> k = -1, like Python) */
tr_bigint *tr_bigint_shr(const tr_bigint *, uint64_t bits, const tr_math_budget *, tr_math_refusal *, int32_t *error);
/* a^e (0^0 = 1); the size bound is checked before anything is computed */
tr_bigint *tr_bigint_pow(const tr_bigint *, uint64_t e, const tr_math_budget *, tr_math_refusal *, int32_t *error);
/* floor(sqrt(a)) for a >= 0 (a < 0: NO_SOLUTION) */
tr_bigint *tr_bigint_isqrt(const tr_bigint *, const tr_math_budget *, tr_math_refusal *, int32_t *error);
/* the canonical residue a mod m in [0, m) for m >= 1 (m == 0: INVALID) */
int32_t tr_bigint_mod_u64(const tr_bigint *, uint64_t m, uint64_t *out);
/* the tight enclosure [RD(a), RU(a)] (APPROXIMATE carrier; [DBL_MAX, inf] past the double range) */
int32_t tr_bigint_enclose(const tr_bigint *, tr_math_interval *out);
/* exact sum(a[i] * b[i]) and Horner's c[0] + x (c[1] + x (...)) over big integers */
tr_bigint *tr_bigint_dot(const tr_bigint *const *a, const tr_bigint *const *b, uint64_t n, const tr_math_budget *,
                         tr_math_refusal *, int32_t *error);
tr_bigint *tr_bigint_horner(const tr_bigint *const *coef, uint32_t ncoef, const tr_bigint *x,
                            const tr_math_budget *, tr_math_refusal *, int32_t *error);
/* elementwise out[i] = a[i] op b[i] for op ADD, SUB, MUL, DIV and MOD (floor rounding), GCD; every output is owned
 * by the caller; all or nothing: on failure every output made so far is freed and out[] is left unchanged */
int32_t tr_bigint_batch(int32_t op, const tr_bigint *const *a, const tr_bigint *const *b, uint64_t n,
                        tr_bigint **out, const tr_math_budget *, tr_math_refusal *);
void tr_bigint_free(tr_bigint *);
void tr_bigint_free_array(tr_bigint **x, uint64_t n);   /* frees each x[i] (NULL allowed) and sets it NULL */

/* ---------------------------------------------------------------- arbitrary-precision rationals */
typedef struct tr_bigq_s tr_bigq;   /* canonical: denominator > 0, gcd(|numerator|, denominator) = 1, 0 = 0/1 */

/* n / d canonicalized (d == 0: NO_SOLUTION) */
tr_bigq *tr_bigq_from_bigints(const tr_bigint *n, const tr_bigint *d, const tr_math_budget *, tr_math_refusal *,
                              int32_t *error);
tr_bigq *tr_bigq_from_bigint(const tr_bigint *, const tr_math_budget *, tr_math_refusal *, int32_t *error);
/* the first slice's int64 rationals (tr_num_rational has the layout of tr_math_rational): lossless; d == 0 INVALID */
tr_bigq *tr_bigq_from_q64(tr_num_rational q, const tr_math_budget *, tr_math_refusal *, int32_t *error);
/* the exact dyadic value of a finite double (NaN, inf: INVALID) */
tr_bigq *tr_bigq_from_double(double x, const tr_math_budget *, tr_math_refusal *, int32_t *error);
int32_t tr_bigq_to_q64(const tr_bigq *, tr_num_rational *out);       /* narrowing: OVERFLOW unless both fit int64 */
/* narrowing to an integer: NO_SOLUTION unless the denominator is 1 */
tr_bigint *tr_bigq_to_bigint(const tr_bigq *, const tr_math_budget *, tr_math_refusal *, int32_t *error);
const tr_bigint *tr_bigq_numerator(const tr_bigq *);                 /* borrowed, valid while the rational lives */
const tr_bigint *tr_bigq_denominator(const tr_bigq *);
int32_t tr_bigq_sign(const tr_bigq *);
int32_t tr_bigq_cmp(const tr_bigq *, const tr_bigq *);               /* -1, 0, 1 (INVALID for a bad handle) */
/* exact comparison with a finite double: -1, 0, 1 (q < x, ==, >); INVALID for NaN */
int32_t tr_bigq_cmp_double(const tr_bigq *, double x, int32_t *out);
int32_t tr_bigq_is_handle(const void *);
tr_bigq *tr_bigq_copy(const tr_bigq *, const tr_math_budget *, tr_math_refusal *, int32_t *error);
tr_bigq *tr_bigq_neg(const tr_bigq *, const tr_math_budget *, tr_math_refusal *, int32_t *error);
tr_bigq *tr_bigq_inv(const tr_bigq *, const tr_math_budget *, tr_math_refusal *, int32_t *error);   /* 0: NO_SOLUTION */
tr_bigq *tr_bigq_add(const tr_bigq *, const tr_bigq *, const tr_math_budget *, tr_math_refusal *, int32_t *error);
tr_bigq *tr_bigq_sub(const tr_bigq *, const tr_bigq *, const tr_math_budget *, tr_math_refusal *, int32_t *error);
tr_bigq *tr_bigq_mul(const tr_bigq *, const tr_bigq *, const tr_math_budget *, tr_math_refusal *, int32_t *error);
tr_bigq *tr_bigq_div(const tr_bigq *, const tr_bigq *, const tr_math_budget *, tr_math_refusal *, int32_t *error);
tr_bigq *tr_bigq_pow(const tr_bigq *, int64_t e, const tr_math_budget *, tr_math_refusal *, int32_t *error);
/* the tight enclosure [RD(q), RU(q)]: lo == hi exactly when q is a double; APPROXIMATE */
int32_t tr_bigq_enclose(const tr_bigq *, tr_math_interval *out);
/* 1 when lo <= q <= hi exactly (endpoints may be infinite), 0 otherwise; INVALID for an invalid interval */
int32_t tr_bigq_in_interval(const tr_bigq *, tr_math_interval iv);
void tr_bigq_free(tr_bigq *);

/* ---------------------------------------------------------------- prime fields GF(p), p < 2^62 */
/* A field made by tr_gfp_field (p verified prime): Barrett constants for x < 2^(2k), k = bit length of p. */
typedef struct { uint64_t p, mu, r64; uint32_t k, reserved; } tr_gfp;   /* mu = floor(2^(2k) / p), r64 = 2^64 mod p */
typedef struct { uint64_t p, v; } tr_gfp_elem;   /* a residue v in [0, p) tagged with its modulus */

int32_t tr_gfp_is_prime(uint64_t n);   /* 1 / 0: deterministic Miller-Rabin, bases 2..37 (exact below 3.3e24) */
/* the field GF(p): INVALID unless p is prime, UNSUPPORTED for a prime >= 2^62 */
int32_t tr_gfp_field(uint64_t p, tr_gfp *out);
/* reductions (REDUCTION conversions): an integer to its residue; a rational n/d to n * d^-1 (d = 0 mod p: NO_SOLUTION) */
int32_t tr_gfp_from_u64(const tr_gfp *, uint64_t x, tr_gfp_elem *out);
int32_t tr_gfp_from_i64(const tr_gfp *, int64_t x, tr_gfp_elem *out);
int32_t tr_gfp_from_i128(const tr_gfp *, tr_math_i128 x, tr_gfp_elem *out);
int32_t tr_gfp_from_q64(const tr_gfp *, tr_num_rational q, tr_gfp_elem *out);
int32_t tr_gfp_from_bigint(const tr_gfp *, const tr_bigint *, tr_gfp_elem *out);
int32_t tr_gfp_from_bigq(const tr_gfp *, const tr_bigq *, tr_gfp_elem *out);
/* element arithmetic: CARRIER_ERROR when the moduli differ, INVALID when v >= p or p is not a field modulus;
 * div / inv of 0: NO_SOLUTION; pow with e < 0 inverts first (0^e, e < 0: NO_SOLUTION), 0^0 = 1 */
int32_t tr_gfp_add(tr_gfp_elem a, tr_gfp_elem b, tr_gfp_elem *out);
int32_t tr_gfp_sub(tr_gfp_elem a, tr_gfp_elem b, tr_gfp_elem *out);
int32_t tr_gfp_mul(tr_gfp_elem a, tr_gfp_elem b, tr_gfp_elem *out);
int32_t tr_gfp_div(tr_gfp_elem a, tr_gfp_elem b, tr_gfp_elem *out);
int32_t tr_gfp_neg(tr_gfp_elem a, tr_gfp_elem *out);
int32_t tr_gfp_inv(tr_gfp_elem a, tr_gfp_elem *out);
int32_t tr_gfp_pow(tr_gfp_elem a, int64_t e, tr_gfp_elem *out);
int32_t tr_gfp_equal(tr_gfp_elem a, tr_gfp_elem b);   /* 1 / 0, or CARRIER_ERROR for different moduli */
/* LIFT conversions: v in [0, p), or symmetric: v - p when v > p / 2 (in (-p/2, p/2]) */
int32_t tr_gfp_lift_i64(tr_gfp_elem a, int32_t symmetric, int64_t *out);
tr_bigint *tr_gfp_lift(tr_gfp_elem a, int32_t symmetric, const tr_math_budget *, tr_math_refusal *, int32_t *error);
/* batched over arrays of residues of one field (every input < p, else INVALID; nothing is allocated):
 *   vec: out[i] = a[i] op b[i], op ADD, SUB, MUL, DIV (b[i] = 0: NO_SOLUTION), NEG, INV (a[i] = 0: NO_SOLUTION;
 *        b unused and may be NULL for NEG / INV); INVALID takes precedence over NO_SOLUTION; out is written only on
 *        success (out may alias a or b)
 *   dot: sum(a[i] b[i]) mod p          matvec: y = A x mod p, A row-major rows x cols (y must not alias A or x)
 *   reduce_i64: out[i] = x[i] mod p in [0, p) */
int32_t tr_gfp_vec(const tr_gfp *, int32_t op, const uint64_t *a, const uint64_t *b, uint64_t n, uint64_t *out);
int32_t tr_gfp_dot(const tr_gfp *, const uint64_t *a, const uint64_t *b, uint64_t n, uint64_t *out);
int32_t tr_gfp_matvec(const tr_gfp *, const uint64_t *A, uint64_t rows, uint64_t cols, const uint64_t *x, uint64_t *y);
int32_t tr_gfp_reduce_i64(const tr_gfp *, const int64_t *x, uint64_t n, uint64_t *out);

/* Chinese remaindering and rational reconstruction (the steps of exact multi-modular solvers).
 * crt: the unique x with x = residues[i] mod fields[i].p for k >= 1 distinct primes (a repeated prime: INVALID), in
 *   [0, M) or, symmetric, in (-M/2, M/2]; M = the product of the primes. Garner's mixed-radix algorithm. *x and *modulus
 *   (either may be NULL) are owned by the caller.
 * crt_combine: one incremental step: from x mod m (0 <= x < m) and residue mod p (p prime to m: else INVALID) to the
 *   unique x' in [0, m p).
 * ratrecon: the fraction n / d with |n| <= N, 0 < d <= D, gcd(n, d) = 1, gcd(d, m) = 1 and n = u d (mod m), found by
 *   the extended Euclidean algorithm; it is unique when 2 N D < m. N, D NULL: N = D = isqrt((m - 1) / 2). m >= 2.
 *   NO_SOLUTION when no such fraction exists. */
int32_t tr_carrier_crt(const uint64_t *residues, const tr_gfp *fields, uint32_t k, int32_t symmetric, tr_bigint **x,
                       tr_bigint **modulus, const tr_math_budget *, tr_math_refusal *);
int32_t tr_carrier_crt_combine(const tr_bigint *x, const tr_bigint *m, uint64_t residue, const tr_gfp *field,
                               tr_bigint **x_out, tr_bigint **m_out, const tr_math_budget *, tr_math_refusal *);
tr_bigq *tr_carrier_ratrecon(const tr_bigint *u, const tr_bigint *m, const tr_bigint *N, const tr_bigint *D,
                             const tr_math_budget *, tr_math_refusal *, int32_t *error);

/* ---------------------------------------------------------------- matched expressions: fixed width, wide, exact */
/* sum(a[i] b[i]): dot_i64 checked in int64 (OVERFLOW at the first partial product or sum past int64), dot_i128
 * checked in 128 bits (OVERFLOW past it), dot_wide exact in 192-bit two's complement (out[0] least significant;
 * never overflows for n < 2^64). Horner c[0] + x (c[1] + ... ) per point x[i]: checked int64 / checked 128-bit; on
 * OVERFLOW the entries before the failing point have been written (the array's contents are then unspecified). */
int32_t tr_carrier_dot_i64(const int64_t *a, const int64_t *b, uint64_t n, int64_t *out);
int32_t tr_carrier_dot_i128(const int64_t *a, const int64_t *b, uint64_t n, tr_math_i128 *out);
int32_t tr_carrier_dot_wide(const int64_t *a, const int64_t *b, uint64_t n, uint64_t out[3]);
int32_t tr_carrier_horner_i64(const int64_t *coef, uint32_t ncoef, const int64_t *x, uint64_t n, int64_t *out);
int32_t tr_carrier_horner_i128(const int64_t *coef, uint32_t ncoef, const int64_t *x, uint64_t n, tr_math_i128 *out);
/* exact per-group sums of int64 values in 128 bits (|sum| < 2^127 for n < 2^64): groups NULL = one group (ngroups
 * must then be 1); a group id >= ngroups is INVALID; out has ngroups entries, written only on success */
int32_t tr_carrier_sum_i64(const int64_t *x, uint64_t n, const uint32_t *groups, uint64_t ngroups, tr_math_i128 *out);

/* ---------------------------------------------------------------- intervals and complex intervals (APPROXIMATE) */
typedef struct { tr_math_interval re, im; } tr_carrier_cinterval;   /* re.lo <= Re z <= re.hi, im likewise */

/* op on enclosures: interval ADD, SUB, MUL, DIV (b containing 0: NO_SOLUTION), NEG, SQR, SQRT (a.lo < 0:
 * NO_SOLUTION), ABS, POW (integer exponent n; n < 0 needs a without 0), INV, HULL, INTERSECT (empty: NO_SOLUTION).
 * INVALID for an invalid interval. b may be NULL for unary ops. */
int32_t tr_carrier_interval_op(int32_t op, const tr_math_interval *a, const tr_math_interval *b, int64_t n,
                               tr_math_interval *out);
int32_t tr_carrier_interval_cmp(const tr_math_interval *a, const tr_math_interval *b);   /* TR_CMP_* or INVALID */
/* complex rectangles: ADD, SUB, MUL, DIV and INV (NO_SOLUTION when the divisor's rectangle contains 0, or when the
 * lower bound of its squared modulus rounds to 0), NEG, CONJ, SQR */
int32_t tr_carrier_cinterval_op(int32_t op, const tr_carrier_cinterval *a, const tr_carrier_cinterval *b,
                                tr_carrier_cinterval *out);
/* |z| = sqrt(re^2 + im^2), enclosed */
int32_t tr_carrier_cinterval_abs(const tr_carrier_cinterval *a, tr_math_interval *out);
/* narrowing complex -> real: NO_SOLUTION unless im is exactly [0, 0] */
int32_t tr_carrier_cinterval_real(const tr_carrier_cinterval *a, tr_math_interval *out);
/* ENCLOSING conversions of exact fixed-width values (tight: [RD(x), RU(x)]) */
int32_t tr_carrier_enclose_i64(int64_t v, tr_math_interval *out);
int32_t tr_carrier_enclose_i128(tr_math_i128 v, tr_math_interval *out);
int32_t tr_carrier_enclose_q64(tr_num_rational q, tr_math_interval *out);   /* d == 0: INVALID */
int32_t tr_carrier_enclose_q64_vec(const tr_num_rational *q, uint64_t n, tr_math_interval *out);

/* ---------------------------------------------------------------- truncated power series over Q */
/* f = c[0] + c[1] x + ... + c[N-1] x^(N-1) + O(x^N), order N >= 1; coefficients are exact canonical BIGQ. A series
 * is exact up to O(x^N): every operation returns the exact coefficients of the result mod x^M with M stated below.
 * Two series combine only with the same order (else TR_MATH_CARRIER_ERROR): truncate one explicitly first.
 *   add, sub, mul (truncated convolution), scale: order N        derivative: order N - 1 (N >= 2, else INVALID)
 *   integral: order N + 1, zero constant term                    truncate(M): M <= N (else NO_SOLUTION)
 *   reciprocal: c[0] != 0 (else NO_SOLUTION)                     compose f(g): g's c[0] = 0 (else NO_SOLUTION)
 *   exp: c[0] = 0 (else NO_SOLUTION: exp(c) is not rational)     log: c[0] = 1 (else NO_SOLUTION)
 * The budget bounds the order (degree, support) and every coefficient (bytes), see above. */
typedef struct tr_series_s tr_series;
/* coefficients 0..count-1 from coef (count <= order, else INVALID; the rest are 0); each is deep-copied */
tr_series *tr_series_create(const tr_bigq *const *coef, uint32_t count, uint32_t order, const tr_math_budget *,
                            tr_math_refusal *, int32_t *error);
/* the constant c + O(x^order) (a LOSSLESS conversion of an exact rational) */
tr_series *tr_series_constant(const tr_bigq *c, uint32_t order, const tr_math_budget *, tr_math_refusal *,
                              int32_t *error);
uint32_t tr_series_order(const tr_series *);                         /* 0 for a bad handle */
const tr_bigq *tr_series_coeff(const tr_series *, uint32_t k);       /* borrowed; NULL when k >= order */
int32_t tr_series_is_handle(const void *);
int32_t tr_series_equal(const tr_series *, const tr_series *);       /* 1 / 0, CARRIER_ERROR for different orders */
tr_series *tr_series_truncate(const tr_series *, uint32_t order, const tr_math_budget *, tr_math_refusal *, int32_t *error);
tr_series *tr_series_add(const tr_series *, const tr_series *, const tr_math_budget *, tr_math_refusal *, int32_t *error);
tr_series *tr_series_sub(const tr_series *, const tr_series *, const tr_math_budget *, tr_math_refusal *, int32_t *error);
tr_series *tr_series_mul(const tr_series *, const tr_series *, const tr_math_budget *, tr_math_refusal *, int32_t *error);
tr_series *tr_series_scale(const tr_series *, const tr_bigq *, const tr_math_budget *, tr_math_refusal *, int32_t *error);
tr_series *tr_series_derivative(const tr_series *, const tr_math_budget *, tr_math_refusal *, int32_t *error);
tr_series *tr_series_integral(const tr_series *, const tr_math_budget *, tr_math_refusal *, int32_t *error);
tr_series *tr_series_reciprocal(const tr_series *, const tr_math_budget *, tr_math_refusal *, int32_t *error);
tr_series *tr_series_compose(const tr_series *outer, const tr_series *inner, const tr_math_budget *,
                             tr_math_refusal *, int32_t *error);
tr_series *tr_series_exp(const tr_series *, const tr_math_budget *, tr_math_refusal *, int32_t *error);
tr_series *tr_series_log(const tr_series *, const tr_math_budget *, tr_math_refusal *, int32_t *error);
void tr_series_free(tr_series *);

/* ---------------------------------------------------------------- CUDA entry points (math_carriers_gpu.cu) */
/* The data-parallel kernels, equal bit for bit to the CPU functions above (same validation, same codes, outputs written
 * only on success). Each budgets its device copies before allocating (resource bytes, need = tr_carrier_kernel_bytes)
 * and matvec also its dense matrix (resource tensor, need = rows * cols); sum_i64 its accumulators (states = ngroups).
 *   int tr_mgpu_carriers_gfp_vec(const tr_gfp *, int32_t op, const uint64_t *a, const uint64_t *b, uint64_t n,
 *                                uint64_t *out, const tr_math_budget *, tr_math_refusal *)
 *   int tr_mgpu_carriers_gfp_dot(const tr_gfp *, const uint64_t *a, const uint64_t *b, uint64_t n, uint64_t *out, ...)
 *   int tr_mgpu_carriers_gfp_matvec(const tr_gfp *, const uint64_t *A, uint64_t rows, uint64_t cols,
 *                                   const uint64_t *x, uint64_t *y, ...)
 *   int tr_mgpu_carriers_gfp_reduce_i64(const tr_gfp *, const int64_t *x, uint64_t n, uint64_t *out, ...)
 *   int tr_mgpu_carriers_enclose_q64(const tr_num_rational *q, uint64_t n, tr_math_interval *out, ...)
 *   int tr_mgpu_carriers_sum_i64(const int64_t *x, uint64_t n, const uint32_t *groups, uint64_t ngroups,
 *                                tr_math_i128 *out, ...)
 *   int tr_mgpu_carriers_dot_wide(const int64_t *a, const int64_t *b, uint64_t n, uint64_t out[3], ...)
 * Order independence: one output slot per element (vec, reduce, enclose), or per-thread partial sums combined with
 * exact 128-bit / 192-bit atomic additions (dot, matvec, sum, dot_wide), whose totals do not depend on the order. */
enum { TR_CARRIER_KERNEL_GFP_VEC = 1, TR_CARRIER_KERNEL_GFP_DOT = 2, TR_CARRIER_KERNEL_GFP_MATVEC = 3,
       TR_CARRIER_KERNEL_GFP_REDUCE = 4, TR_CARRIER_KERNEL_ENCLOSE_Q64 = 5, TR_CARRIER_KERNEL_SUM_I64 = 6,
       TR_CARRIER_KERNEL_DOT_WIDE = 7 };
#define TR_CARRIERS_CHUNK 256   /* matvec: columns per work item */
/* bytes a kernel copies to / allocates on the device (saturating). n = elements (matvec: rows), m = matvec cols or
 * sum ngroups, flag = vec unary (NEG, INV) or sum with groups:
 *   vec 8n (a) + 8n (b, binary ops) + 8n (out) + 8     dot 16n + 24      matvec 8 rows cols + 8 cols + 24 rows + 8
 *   reduce 16n      enclose 32n + 8      sum 8n + 4n (groups) + 16 ngroups + 8      dot_wide 16n + 24 */
static inline uint64_t tr_carrier_kernel_bytes(int32_t kernel, uint64_t n, uint64_t m, int32_t flag) {
    switch (kernel) {
    case TR_CARRIER_KERNEL_GFP_VEC: return tr_budget_add(tr_budget_mul(n, flag ? 16 : 24), 8);
    case TR_CARRIER_KERNEL_GFP_DOT: return tr_budget_add(tr_budget_mul(n, 16), 24);
    case TR_CARRIER_KERNEL_GFP_MATVEC:
        return tr_budget_add(tr_budget_add(tr_budget_mul(tr_budget_mul(n, m), 8), tr_budget_mul(m, 8)),
                             tr_budget_add(tr_budget_mul(n, 24), 8));
    case TR_CARRIER_KERNEL_GFP_REDUCE: return tr_budget_mul(n, 16);
    case TR_CARRIER_KERNEL_ENCLOSE_Q64: return tr_budget_add(tr_budget_mul(n, 32), 8);
    case TR_CARRIER_KERNEL_SUM_I64: return tr_budget_add(tr_budget_add(tr_budget_mul(n, flag ? 12 : 8), tr_budget_mul(m, 16)), 8);
    case TR_CARRIER_KERNEL_DOT_WIDE: return tr_budget_add(tr_budget_mul(n, 16), 24);
    default: return TR_BUDGET_UNKNOWN;
    }
}
uint64_t tr_carrier_device_bytes(int32_t kernel, uint64_t n, uint64_t m, int32_t flag);   /* exported for the bindings */

/* the Barrett constants of p (host code), and whether a field struct carries consistent ones (p in [2, 2^62)): the
 * batched functions check this on every call; primality is tr_gfp_field's (CRT re-proves it) */
static inline void tr_gfp_fill(uint64_t p, tr_gfp *f) {
    uint32_t k = (uint32_t)(64 - __builtin_clzll(p));
    f->p = p; f->k = k; f->reserved = 0;
    f->mu = (uint64_t)(((unsigned __int128)1 << (2 * k)) / p);
    f->r64 = (uint64_t)(((unsigned __int128)1 << 64) % p);
}
static inline int tr_gfp_consistent(const tr_gfp *f) {
    if (!f || f->p < 2 || f->p >= TR_GFP_MAX) return 0;
    tr_gfp g;
    tr_gfp_fill(f->p, &g);
    return g.k == f->k && g.mu == f->mu && g.r64 == f->r64;
}

/* ================================================================ inline helpers (host and CUDA device code) */

/* ---------------------------------------------------------------- GF(p): Barrett reduction (HAC 14.42) */
/* x mod p for x < 2^(2k) (any product of two residues): q3 = ((x >> (k-1)) * mu) >> (k+1) is floor(x/p) or at most 2
 * less, so r = x - q3 p lies in [0, 3p) and fits 64 bits; two conditional subtractions finish. */
TR_NUM_HD uint64_t tr_gfp_barrett_hd(const tr_gfp *f, unsigned __int128 x) {
    uint64_t q1 = (uint64_t)(x >> (f->k - 1));
    uint64_t q3 = (uint64_t)(((unsigned __int128)q1 * f->mu) >> (f->k + 1));
    uint64_t r = (uint64_t)x - q3 * f->p;
    if (r >= f->p) r -= f->p;
    if (r >= f->p) r -= f->p;
    return r;
}
TR_NUM_HD uint64_t tr_gfp_add_hd(const tr_gfp *f, uint64_t a, uint64_t b) { uint64_t r = a + b; return r >= f->p ? r - f->p : r; }
TR_NUM_HD uint64_t tr_gfp_sub_hd(const tr_gfp *f, uint64_t a, uint64_t b) { return a >= b ? a - b : a + (f->p - b); }
TR_NUM_HD uint64_t tr_gfp_neg_hd(const tr_gfp *f, uint64_t a) { return a ? f->p - a : 0; }
TR_NUM_HD uint64_t tr_gfp_mul_hd(const tr_gfp *f, uint64_t a, uint64_t b) { return tr_gfp_barrett_hd(f, (unsigned __int128)a * b); }
TR_NUM_HD uint64_t tr_gfp_pow_hd(const tr_gfp *f, uint64_t a, uint64_t e) {
    uint64_t r = 1;
    while (e) { if (e & 1) r = tr_gfp_mul_hd(f, r, a); a = tr_gfp_mul_hd(f, a, a); e >>= 1; }
    return r;
}
/* a^-1 for 0 < a < p by the extended Euclidean algorithm (|Bezout coefficients| <= p, so no product overflows) */
TR_NUM_HD uint64_t tr_gfp_inv_hd(const tr_gfp *f, uint64_t a) {
    int64_t t = 0, nt = 1;
    uint64_t r = f->p, nr = a;
    while (nr) {
        uint64_t q = r / nr, tr = r - q * nr;
        int64_t tt = t - (int64_t)q * nt;
        t = nt; nt = tt; r = nr; nr = tr;
    }
    return t < 0 ? (uint64_t)(t + (int64_t)f->p) : (uint64_t)t;
}
TR_NUM_HD uint64_t tr_gfp_reduce_i64_hd(const tr_gfp *f, int64_t x) {
    uint64_t m = tr_abs_u64(x) % f->p;
    return x < 0 && m ? f->p - m : m;
}
/* (hi 2^64 + lo) mod p = ((hi mod p) r64 + lo mod p) mod p */
TR_NUM_HD uint64_t tr_gfp_reduce_u128_hd(const tr_gfp *f, uint64_t hi, uint64_t lo) {
    return tr_gfp_add_hd(f, tr_gfp_mul_hd(f, hi % f->p, f->r64), lo % f->p);
}

/* ---------------------------------------------------------------- exact wide accumulation */
/* 192-bit two's complement words w[0..2] += x (a sign-extended 128-bit value) */
TR_NUM_HD void tr_wide_add128(uint64_t *w, __int128 x) {
    uint64_t x0 = (uint64_t)x, x1 = (uint64_t)((unsigned __int128)x >> 64), x2 = x < 0 ? ~(uint64_t)0 : 0;
    uint64_t s0 = w[0] + x0, c0 = s0 < x0;
    uint64_t y1 = x1 + c0, c1a = y1 < c0;
    uint64_t s1 = w[1] + y1, c1 = s1 < y1;
    w[0] = s0; w[1] = s1; w[2] += x2 + c1 + c1a;
}

/* ---------------------------------------------------------------- tight enclosure of an int64 rational */
TR_NUM_HD int tr_carrier_bitlen_u64(uint64_t x) {
#if defined(__CUDA_ARCH__)
    return x ? 64 - __clzll((long long)x) : 0;
#else
    return x ? 64 - __builtin_clzll(x) : 0;
#endif
}
/* [RD(n/d), RU(n/d)] for d != 0 (any signs, not necessarily canonical); returns 0, or -1 (TR_MATH_INVALID) for d == 0.
 * Integer arithmetic only: q = a / b in [2^E, 2^(E+1)) with E exact, t = floor(q 2^(52-E)) in [2^52, 2^53) from a
 * double estimate corrected by exact 128-bit products, lo = t 2^(E-52), hi = lo when exact else (t + 1) 2^(E-52).
 * |q| lies in [2^-63, 2^63], so every endpoint is a normal double and ldexp is exact. The same bits on every device. */
TR_NUM_HD int tr_interval_enclose_q64(int64_t n, int64_t d, tr_math_interval *out) {
    if (d == 0) return -1;
    if (n == 0) { out->lo = 0.0; out->hi = 0.0; return 0; }
    int neg = (n < 0) != (d < 0);
    uint64_t a = tr_abs_u64(n), b = tr_abs_u64(d);
    int e0 = tr_carrier_bitlen_u64(a) - tr_carrier_bitlen_u64(b);
    int ge = e0 >= 0 ? ((unsigned __int128)a >= ((unsigned __int128)b << e0))
                     : (((unsigned __int128)a << (-e0)) >= (unsigned __int128)b);
    int E = ge ? e0 : e0 - 1, s = 52 - E;
    unsigned __int128 A = s >= 0 ? (unsigned __int128)a << s : (unsigned __int128)a;
    unsigned __int128 B = s >= 0 ? (unsigned __int128)b : (unsigned __int128)b << (-s);
    double est = ldexp((double)a / (double)b, s);
    uint64_t t = est < 4503599627370496.0 ? 4503599627370496ULL
               : est >= 9007199254740991.0 ? 9007199254740991ULL : (uint64_t)est;
    while ((unsigned __int128)t * B > A) t--;
    while ((unsigned __int128)(t + 1) * B <= A) t++;
    int exact = (unsigned __int128)t * B == A;
    double lo = ldexp((double)t, -s), hi = exact ? lo : ldexp((double)(t + 1), -s);
    if (neg) { out->lo = -hi; out->hi = -lo; } else { out->lo = lo; out->hi = hi; }
    return 0;
}

/* ---------------------------------------------------------------- interval arithmetic with tight outward rounding */
TR_NUM_HD int tr_interval_valid(tr_math_interval a) {
    return a.lo == a.lo && a.hi == a.hi && a.lo <= a.hi && a.lo != INFINITY && a.hi != -INFINITY;
}
TR_NUM_HD tr_math_interval tr_interval_point(double x) { tr_math_interval r; r.lo = x; r.hi = x; return r; }
TR_NUM_HD double tr_carrier_succ(double x) { return x == INFINITY ? x : tr_step(x, 1); }
TR_NUM_HD double tr_carrier_pred(double x) { return x == -INFINITY ? x : tr_step(x, 0); }
#define TR_CARRIER_TINY 2.004168360008973e-292      /* 2^-969: below it products and quotients are stepped out */
/* RD / RU of x + y (TwoSum: the rounding error is exact when the sum is finite) */
TR_NUM_HD double tr_add_rd(double x, double y) {
    double s = x + y;
    if (s == INFINITY) return (x == INFINITY || y == INFINITY) ? s : DBL_MAX;
    if (s == -INFINITY) return s;
    double bb = s - x, err = (x - (s - bb)) + (y - bb);
    return err < 0 ? tr_carrier_pred(s) : s;
}
TR_NUM_HD double tr_add_ru(double x, double y) {
    double s = x + y;
    if (s == -INFINITY) return (x == -INFINITY || y == -INFINITY) ? s : -DBL_MAX;
    if (s == INFINITY) return s;
    double bb = s - x, err = (x - (s - bb)) + (y - bb);
    return err > 0 ? tr_carrier_succ(s) : s;
}
/* RD / RU of x * y with 0 * inf = 0 (TwoProd with fma: exact error unless the product is tiny or overflows) */
TR_NUM_HD double tr_mul_dir(double x, double y, int up) {
    if (x == 0 || y == 0) return 0.0;
    double p = x * y;
    if (x == INFINITY || x == -INFINITY || y == INFINITY || y == -INFINITY) return p;
    if (p == INFINITY) return up ? p : DBL_MAX;
    if (p == -INFINITY) return up ? -DBL_MAX : p;
    if (fabs(p) < TR_CARRIER_TINY) return up ? tr_carrier_succ(p) : tr_carrier_pred(p);
    double e = fma(x, y, -p);
    if (up) return e > 0 ? tr_carrier_succ(p) : p;
    return e < 0 ? tr_carrier_pred(p) : p;
}
/* RD / RU of x / y, y != 0, with finite / inf = 0 and inf / finite = inf (exact remainder x - q y by fma) */
TR_NUM_HD double tr_div_dir(double x, double y, int up) {
    if (x == 0) return 0.0;
    double q = x / y;
    if (x == INFINITY || x == -INFINITY || y == INFINITY || y == -INFINITY) return q;
    if (q == INFINITY) return up ? q : DBL_MAX;
    if (q == -INFINITY) return up ? -DBL_MAX : q;
    if (fabs(q) < TR_CARRIER_TINY || fabs(x) < TR_CARRIER_TINY || fabs(y) < TR_CARRIER_TINY)
        return up ? tr_carrier_succ(q) : tr_carrier_pred(q);
    double r = fma(-q, y, x);                      /* x - q y exactly; x / y - q has the sign of r / y */
    int above = (r > 0) == (y > 0) && r != 0, below = (r < 0) == (y > 0) && r != 0;
    if (up) return above ? tr_carrier_succ(q) : q;
    return below ? tr_carrier_pred(q) : q;
}
TR_NUM_HD tr_math_interval tr_interval_add(tr_math_interval a, tr_math_interval b) {
    tr_math_interval r; r.lo = tr_add_rd(a.lo, b.lo); r.hi = tr_add_ru(a.hi, b.hi); return r;
}
TR_NUM_HD tr_math_interval tr_interval_neg(tr_math_interval a) { tr_math_interval r; r.lo = -a.hi; r.hi = -a.lo; return r; }
TR_NUM_HD tr_math_interval tr_interval_sub(tr_math_interval a, tr_math_interval b) { return tr_interval_add(a, tr_interval_neg(b)); }
TR_NUM_HD tr_math_interval tr_interval_mul(tr_math_interval a, tr_math_interval b) {
    tr_math_interval r;
    if (a.lo >= 0) {
        if (b.lo >= 0)      { r.lo = tr_mul_dir(a.lo, b.lo, 0); r.hi = tr_mul_dir(a.hi, b.hi, 1); }
        else if (b.hi <= 0) { r.lo = tr_mul_dir(a.hi, b.lo, 0); r.hi = tr_mul_dir(a.lo, b.hi, 1); }
        else                { r.lo = tr_mul_dir(a.hi, b.lo, 0); r.hi = tr_mul_dir(a.hi, b.hi, 1); }
    } else if (a.hi <= 0) {
        if (b.lo >= 0)      { r.lo = tr_mul_dir(a.lo, b.hi, 0); r.hi = tr_mul_dir(a.hi, b.lo, 1); }
        else if (b.hi <= 0) { r.lo = tr_mul_dir(a.hi, b.hi, 0); r.hi = tr_mul_dir(a.lo, b.lo, 1); }
        else                { r.lo = tr_mul_dir(a.lo, b.hi, 0); r.hi = tr_mul_dir(a.lo, b.lo, 1); }
    } else {
        if (b.lo >= 0)      { r.lo = tr_mul_dir(a.lo, b.hi, 0); r.hi = tr_mul_dir(a.hi, b.hi, 1); }
        else if (b.hi <= 0) { r.lo = tr_mul_dir(a.hi, b.lo, 0); r.hi = tr_mul_dir(a.lo, b.lo, 1); }
        else {
            double l1 = tr_mul_dir(a.lo, b.hi, 0), l2 = tr_mul_dir(a.hi, b.lo, 0);
            double h1 = tr_mul_dir(a.lo, b.lo, 1), h2 = tr_mul_dir(a.hi, b.hi, 1);
            r.lo = l1 < l2 ? l1 : l2; r.hi = h1 > h2 ? h1 : h2;
        }
    }
    if (r.lo == 0) r.lo = 0.0;                     /* no -0 endpoints */
    if (r.hi == 0) r.hi = 0.0;
    return r;
}
/* b must not contain 0 (the caller checks: tr_iv_contains_zero) */
TR_NUM_HD tr_math_interval tr_interval_div(tr_math_interval a, tr_math_interval b) {
    tr_math_interval r;
    if (b.lo > 0) {
        if (a.lo >= 0)      { r.lo = tr_div_dir(a.lo, b.hi, 0); r.hi = tr_div_dir(a.hi, b.lo, 1); }
        else if (a.hi <= 0) { r.lo = tr_div_dir(a.lo, b.lo, 0); r.hi = tr_div_dir(a.hi, b.hi, 1); }
        else                { r.lo = tr_div_dir(a.lo, b.lo, 0); r.hi = tr_div_dir(a.hi, b.lo, 1); }
    } else {
        if (a.lo >= 0)      { r.lo = tr_div_dir(a.hi, b.hi, 0); r.hi = tr_div_dir(a.lo, b.lo, 1); }
        else if (a.hi <= 0) { r.lo = tr_div_dir(a.hi, b.lo, 0); r.hi = tr_div_dir(a.lo, b.hi, 1); }
        else                { r.lo = tr_div_dir(a.hi, b.hi, 0); r.hi = tr_div_dir(a.lo, b.hi, 1); }
    }
    if (r.lo == 0) r.lo = 0.0;
    if (r.hi == 0) r.hi = 0.0;
    return r;
}
TR_NUM_HD tr_math_interval tr_interval_abs(tr_math_interval a) {
    if (a.lo >= 0) return a;
    if (a.hi <= 0) return tr_interval_neg(a);
    tr_math_interval r; r.lo = 0.0; r.hi = -a.lo > a.hi ? -a.lo : a.hi; return r;
}
TR_NUM_HD tr_math_interval tr_interval_sqr(tr_math_interval a) {
    tr_math_interval m = tr_interval_abs(a), r;
    r.lo = tr_mul_dir(m.lo, m.lo, 0); r.hi = tr_mul_dir(m.hi, m.hi, 1);
    return r;
}
/* a.lo >= 0 (the caller checks); sqrt is correctly rounded and x - s^2 is exact away from the subnormal range */
TR_NUM_HD double tr_sqrt_dir(double x, int up) {
    if (x == 0 || x == INFINITY) return x == 0 ? 0.0 : x;
    double s = sqrt(x);
    if (x < TR_CARRIER_TINY || x > 1e300) return up ? tr_carrier_succ(s) : tr_carrier_pred(s);
    double e = fma(-s, s, x);                      /* x - s^2: sqrt(x) > s exactly when e > 0 */
    if (up) return e > 0 ? tr_carrier_succ(s) : s;
    return e < 0 ? tr_carrier_pred(s) : s;
}
TR_NUM_HD tr_math_interval tr_interval_sqrt(tr_math_interval a) {
    tr_math_interval r; r.lo = tr_sqrt_dir(a.lo, 0); r.hi = tr_sqrt_dir(a.hi, 1); return r;
}
/* x^n for x >= 0 rounded down / up (monotone repeated products) */
TR_NUM_HD double tr_pow_dir(double x, uint64_t n, int up) {
    double r = 1.0;
    while (n) { if (n & 1) r = tr_mul_dir(r, x, up); x = tr_mul_dir(x, x, up); n >>= 1; }
    return r;
}
/* a^n for n >= 0 (n == 0: [1, 1]) */
TR_NUM_HD tr_math_interval tr_interval_pown(tr_math_interval a, uint64_t n) {
    tr_math_interval r;
    if (n == 0) { r.lo = 1.0; r.hi = 1.0; return r; }
    if (n % 2 == 0) {
        tr_math_interval m = tr_interval_abs(a);
        r.lo = tr_pow_dir(m.lo, n, 0); r.hi = tr_pow_dir(m.hi, n, 1);
        return r;
    }
    r.lo = a.lo >= 0 ? tr_pow_dir(a.lo, n, 0) : -tr_pow_dir(-a.lo, n, 1);
    r.hi = a.hi >= 0 ? tr_pow_dir(a.hi, n, 1) : -tr_pow_dir(-a.hi, n, 0);
    if (r.lo == 0) r.lo = 0.0;
    if (r.hi == 0) r.hi = 0.0;
    return r;
}
TR_NUM_HD tr_math_interval tr_interval_hull(tr_math_interval a, tr_math_interval b) {
    tr_math_interval r; r.lo = a.lo < b.lo ? a.lo : b.lo; r.hi = a.hi > b.hi ? a.hi : b.hi; return r;
}

#ifdef __cplusplus
}
#endif
#endif
