/* INV-001: linear and univariate polynomial inverse answers ("which parameters give this target").
 * SPDX-License-Identifier: Apache-2.0
 *
 * Three answers, all exact (rationals and big integers of math_carriers.h; nothing is rounded):
 *
 * 1. Linear targets. m affine equations a_i . x = b_i in n unknowns over Q, inside a declared box domain
 *    lo_j <= x_j <= hi_j (each bound a rational or unbounded). Gauss-Jordan elimination with exact rationals (a pivot
 *    is a nonzero entry, so nothing is ever divided by zero), coordinates fixed by the box (lo_j == hi_j) substituted
 *    first. The answer is one of
 *      EMPTY    no solution in the domain: the equations are inconsistent (reason INCONSISTENT), or their solutions
 *               all lie outside the box (reason OUTSIDE)
 *      UNIQUE   exactly one solution in the domain (the point)
 *      WHOLE    every point of the domain is a solution (every equation reduces to 0 = 0)
 *      FAMILY   an affine family {x0 + sum_d y_d v_d} intersected with the box: x0 is a witness inside the domain, the
 *               directions v_d are a basis of the solutions of the homogeneous system (v_d has a 1 at its free unknown
 *               and 0 at the other free unknowns), and range_d is the exact range of y_d over the family in the box.
 *    In a box, the family is decided exactly by Fourier-Motzkin elimination over the parameters y (feasibility, the
 *    witness by back-substitution, the projection of each parameter); a family whose every projection is one point
 *    is UNIQUE.
 *
 * 2. Univariate polynomial targets. The real roots of f(t) - target in the closed bounds [lo, hi] (each a rational or
 *    unbounded; an unbounded side uses the Cauchy bound, so every real root is found). f - target is made a primitive
 *    integer polynomial; its square-free factorization f - target = c prod_i a_i^i (Yun's algorithm, exact divisions
 *    over Z[x]) gives the multiplicities; the distinct roots are isolated with the Sturm sequence of the square-free
 *    part (signed pseudo-remainders, primitive parts) by bisection at exact rational points. A root is EXACT only when
 *    an exact evaluation proves it: a boundary or a split point where the polynomial is exactly 0, a degree-one
 *    factor, or the rational candidate of the rational root theorem (the fraction with the smallest denominator in
 *    the isolating interval: a rational root u/v of a primitive factor has v | lc, so when that fraction's denominator
 *    exceeds |lc| the root is certified irrational, and otherwise the candidate is evaluated exactly). Every other
 *    root is an open isolating interval (lo, hi) with rational endpoints, exactly one root of f - target inside, and
 *    a certified sign change of its square-free factor (sign_lo at lo, -sign_lo at hi). Boundary roots exactly at lo
 *    or hi are included. The zero polynomial is WHOLE (every point of the bounds), a nonzero constant EMPTY.
 *    Refinement bisects the isolating intervals to a requested width; the sign at each split point comes from the
 *    batched exact evaluator below (CPU or CUDA, the same function), so both give the same intervals.
 *
 * 3. Finite domains. An exhaustive search for the exact solutions of f(x) = target on a grid of values per coordinate
 *    (1..3 coordinates of a first-slice polynomial, the grid in lexicographic order: the first coordinate is the most
 *    significant). Grid points start .. start + count - 1 are examined, count = min(remaining, iteration budget); a
 *    truncated search is reported as incomplete with the next index to resume from (never silently partial). The
 *    matches are listed in index order up to the output budget; a search that would list more stops before that
 *    match (incomplete, reason OUTPUT) so the listed prefix is always complete.
 *
 * Batched exact evaluation (CPU and CUDA). An integer polynomial P of degree d is evaluated at a rational point u/v
 * (v > 0) as S = v^d P(u/v) = sum a_i u^i v^(d-i), whose sign is the sign of P(u/v). S is computed modulo k primes p_j
 * in (2^61, 2^62) (Barrett arithmetic of math_carriers.h), with k chosen by the host from a proven bound: |S| < 2^B
 * with B = bits(sum |a_i|) + d * max(bits(|u|), bits(v)), and 61 k >= B + 2, so M = prod p_j > 2 |S| + 1. The sign of
 * S follows from its residues alone: S = 0 iff every residue is 0; otherwise Garner's mixed-radix digits of S mod M
 * are compared with those of H = (M - 1) / 2 (S > 0 iff S mod M <= H). The finite-domain search uses the same moduli
 * on H(m) = an integer polynomial in the grid numerators (denominators cleared), where H(m) = 0 iff every residue is 0
 * (|H(m)| < M by the same kind of bound). Both per-point functions below are shared by the CPU library and the CUDA
 * kernels of math_inv_gpu.cu, so the CPU and the device give the same signs and matches bit for bit.
 *
 * Budgets (math_budgets.h; checked before allocating, refusal TR_MATH_BUDGET with the tr_math_refusal):
 *   linear     variables = n unknowns, tensor = m * (n + 1), support = inequalities of one Fourier-Motzkin system,
 *              iterations = inequalities generated over all eliminations; bytes per big number (math_carriers.h)
 *   roots      degree = deg(f - target), iterations = exact evaluations (bisection, rational detection, refinement
 *              rounds x roots), output = roots; bytes per big number
 *   signs      output = points, bytes = the job's arrays (tr_inv_signs_bytes; the CUDA copies tr_inv_signs_device_bytes)
 *   sweep      variables = coordinates (<= TR_INV_SWEEP_DIMS, else TR_MATH_UNSUPPORTED), support = terms,
 *              iterations = grid points examined (TRUNCATES, never refuses), output = matches listed (truncates the
 *              search before the first unlisted match), bytes = the plan and its bitmap
 * More moduli than TR_INV_PRIMES_MAX (a value bound past about 7800 bits) or a point past TR_INV_POINT_LIMBS_MAX limbs
 * is TR_MATH_OVERFLOW: the fixed multimodular representation cannot hold the exact value (never wrapped).
 *
 * Errors: TR_MATH_INVALID (malformed input: lo > hi, a NULL argument, a zero denominator), TR_MATH_MEMORY,
 * TR_MATH_OVERFLOW, TR_MATH_BUDGET, TR_MATH_DOMAIN_ERROR (a polynomial depends on a variable that is not an unknown /
 * searched coordinate), TR_MATH_UNSUPPORTED (a linear target of degree > 1, more than 3 searched coordinates).
 * Objects are immutable and individually owned; constructors return NULL and set *error (error may be NULL). */
#ifndef TORBLES_MATH_INV_H
#define TORBLES_MATH_INV_H
#include <stdint.h>
#include <stddef.h>
#include "math_num.h"
#include "math_budgets.h"
#include "math_carriers.h"
#include "mathematics.h"
#ifdef __cplusplus
extern "C" {
#endif

#define TR_INV_ABI 1
#define TR_INV_PRIMES_MAX 128           /* moduli of the multimodular evaluations: values below 2^7800 */
#define TR_INV_POINT_LIMBS_MAX 128      /* 64-bit limbs of a point's numerator or denominator (8192 bits) */
#define TR_INV_SWEEP_DIMS 3             /* searched coordinates of a finite domain */
#define TR_INV_SWEEP_VALUES_MAX (UINT64_C(1) << 20)   /* values of one coordinate */
#define TR_INV_LINEAR_MAX 64            /* unknowns of a linear target */

enum { TR_INV_EMPTY = 0, TR_INV_UNIQUE = 1, TR_INV_WHOLE = 2, TR_INV_FAMILY = 3, TR_INV_FINITE = 4 };
enum { TR_INV_REASON_NONE = 0, TR_INV_INCONSISTENT = 1, TR_INV_OUTSIDE = 2, TR_INV_CONSTANT = 3 };
enum { TR_INV_SEARCH_COMPLETE = 0, TR_INV_SEARCH_ITERATIONS = 1, TR_INV_SEARCH_OUTPUT = 2 };
enum { TR_INV_AT_LO = 1, TR_INV_AT_HI = 2 };              /* boundary flags of a root */
enum { TR_INV_HOW_ISOLATED = 0, TR_INV_HOW_EVALUATED = 1, TR_INV_HOW_LINEAR = 2, TR_INV_HOW_CANDIDATE = 3,
       TR_INV_HOW_IRRATIONAL = 4 };                         /* how a root's status was proved */

/* the moduli table: 1 when its TR_INV_PRIMES_MAX entries are distinct primes in (2^61, 2^62) in decreasing order
 * (verified with tr_gfp_is_prime, deterministic Miller-Rabin), else 0; prime i in *p (i < TR_INV_PRIMES_MAX) */
int32_t tr_inv_primes_verify(void);
int32_t tr_inv_prime(int32_t i, uint64_t *p);

/* ------------------------------------------------------------------ linear targets */
typedef struct tr_inv_linear_s tr_inv_linear;
typedef struct {
    int32_t kind, reason;        /* TR_INV_EMPTY / UNIQUE / WHOLE / FAMILY; TR_INV_REASON_* for EMPTY */
    int32_t unknowns, equations;
    int32_t rank;                /* of the system after the fixed coordinates were substituted */
    int32_t dims;                /* FAMILY: parameters y_d (free unknowns); 0 otherwise */
    int32_t fixed;               /* unknowns fixed by the domain (lo == hi) */
    int32_t reserved;
    uint64_t iterations;         /* Fourier-Motzkin inequalities generated */
} tr_inv_linear_info;            /* 40 bytes */

/* A (m x n, row-major) and b (m): the equations A x = b; an entry NULL is 0. lo / hi (n entries, or NULL arrays):
 * the box, an entry NULL is unbounded. m >= 0, n >= 1. */
tr_inv_linear *tr_inv_linear_solve(int32_t m, int32_t n, const tr_bigq *const *A, const tr_bigq *const *b,
                                   const tr_bigq *const *lo, const tr_bigq *const *hi, const tr_math_budget *,
                                   tr_math_refusal *, int32_t *error);
/* the equations f_i = target_i of first-slice polynomials of degree <= 1 in the unknowns (variable slots
 * unknowns[0..n-1], distinct, 0..7); a term in another variable is TR_MATH_DOMAIN_ERROR, a term of degree > 1
 * TR_MATH_UNSUPPORTED */
tr_inv_linear *tr_inv_linear_from_polys(int32_t m, const tr_math_poly *const *f, const tr_bigq *const *target,
                                        int32_t n, const int32_t *unknowns, const tr_bigq *const *lo,
                                        const tr_bigq *const *hi, const tr_math_budget *, tr_math_refusal *,
                                        int32_t *error);
int32_t tr_inv_linear_get_info(const tr_inv_linear *, tr_inv_linear_info *);
/* UNIQUE: the solution; FAMILY: the witness x0 (inside the domain); NULL otherwise (borrowed) */
const tr_bigq *tr_inv_linear_point(const tr_inv_linear *, int32_t j);
/* FAMILY: coordinate j of direction d (borrowed) */
const tr_bigq *tr_inv_linear_direction(const tr_inv_linear *, int32_t d, int32_t j);
/* FAMILY: the unknown whose offset from x0 is y_d (direction d has 1 there) */
int32_t tr_inv_linear_free_unknown(const tr_inv_linear *, int32_t d);
/* FAMILY: the exact range [lo, hi] of y_d over the family in the domain (it contains 0); NULL = unbounded */
int32_t tr_inv_linear_range(const tr_inv_linear *, int32_t d, const tr_bigq **lo, const tr_bigq **hi);
void tr_inv_linear_free(tr_inv_linear *);

/* ------------------------------------------------------------------ batched exact signs */
/* A job: npolys integer polynomials and npoints rational points u/v, each point with its polynomial. Built natively
 * (tr_inv_signs_create); the view is what the CPU evaluator and the CUDA kernel read. */
typedef struct {
    uint64_t npoints;
    uint32_t nprimes;            /* k moduli (fields[0..k-1]) */
    uint32_t npolys;
    uint32_t limbs;              /* limbs of every numerator and denominator magnitude (>= 1) */
    uint32_t total;              /* coefficients of all polynomials: offset[npolys] */
    const tr_gfp *fields;        /* k */
    const uint64_t *garner;      /* k x k: garner[j * k + t] = p_t^-1 mod p_j for t < j */
    const uint64_t *half;        /* k mixed-radix digits of (M - 1) / 2 */
    const uint32_t *offset;      /* npolys + 1: polynomial f has coefficients offset[f] .. offset[f+1]-1 (low first) */
    const uint64_t *coef;        /* k x total: coef[j * total + offset[f] + i] = a_{f,i} mod p_j */
    const uint32_t *poly;        /* npoints */
    const int8_t *sign;          /* npoints: the sign of u (-1, 0, 1) */
    const uint64_t *num;         /* npoints x limbs: |u| */
    const uint64_t *den;         /* npoints x limbs: v > 0 */
} tr_inv_signs_view;             /* 96 bytes */
typedef struct tr_inv_signs_s tr_inv_signs;
/* the signature of the sign evaluators (tr_inv_signs_eval on the CPU, tr_mgpu_inv_signs in the CUDA library):
 * signs[i] = sign of polynomial poly[i] at point i; 0 or an error code */
typedef int (*tr_inv_signs_fn)(const tr_inv_signs_view *, int8_t *signs, const tr_math_budget *, tr_math_refusal *);

/* npolys rational polynomials (ncoef[f] coefficients each, low degree first, concatenated in coef; NULL = 0; each made
 * a positive multiple of an integer polynomial, which keeps its signs) and npoints rational points (poly[i] < npolys) */
tr_inv_signs *tr_inv_signs_create(uint32_t npolys, const uint32_t *ncoef, const tr_bigq *const *coef, uint64_t npoints,
                                  const tr_bigq *const *points, const uint32_t *poly, const tr_math_budget *,
                                  tr_math_refusal *, int32_t *error);
int32_t tr_inv_signs_get_view(const tr_inv_signs *, tr_inv_signs_view *);
/* the CPU evaluator (budget: output = npoints) */
int32_t tr_inv_signs_eval(const tr_inv_signs_view *, int8_t *signs, const tr_math_budget *, tr_math_refusal *);
uint64_t tr_inv_signs_bound_bits(const tr_inv_signs *);       /* B of the proof (|S| < 2^B) */
void tr_inv_signs_free(tr_inv_signs *);

/* bytes the CUDA evaluation copies to and allocates on the device (saturating) */
static inline uint64_t tr_inv_signs_device_bytes(const tr_inv_signs_view *v) {
    uint64_t k = v->nprimes, b = tr_budget_mul(k, sizeof(tr_gfp));
    b = tr_budget_add(b, tr_budget_mul(tr_budget_mul(k, k), 8));
    b = tr_budget_add(b, tr_budget_mul(k, 8));
    b = tr_budget_add(b, tr_budget_mul((uint64_t)v->npolys + 1, 4));
    b = tr_budget_add(b, tr_budget_mul(tr_budget_mul(k, v->total), 8));
    b = tr_budget_add(b, tr_budget_mul(v->npoints, 4 + 1 + 1));                 /* poly, sign, output */
    b = tr_budget_add(b, tr_budget_mul(tr_budget_mul(v->npoints, v->limbs), 16));
    return b;
}
uint64_t tr_inv_signs_device_need(const tr_inv_signs_view *);  /* exported for the bindings */

/* ------------------------------------------------------------------ univariate roots */
typedef struct tr_inv_roots_s tr_inv_roots;
typedef struct {
    int32_t kind;                /* TR_INV_EMPTY (reason CONSTANT for a nonzero constant), FINITE or WHOLE */
    int32_t reason;
    int32_t degree;              /* of f - target (-1: the zero polynomial) */
    int32_t nroots;              /* distinct roots in the bounds */
    int32_t nfactors;            /* square-free factors: factor i holds the roots of multiplicity i + 1 */
    int32_t exact;               /* roots identified exactly */
    int32_t lo_bounded, hi_bounded;
    uint64_t iterations;         /* exact evaluations so far (isolation, rational detection, refinement) */
    uint64_t rounds;             /* refinement rounds (batched sign evaluations) */
} tr_inv_roots_info;             /* 48 bytes */
typedef struct {
    int32_t exact;               /* 1: lo == hi == the root; 0: lo < root < hi */
    int32_t multiplicity;
    int32_t boundary;            /* TR_INV_AT_LO | TR_INV_AT_HI: the root is a declared bound */
    int32_t sign_lo;             /* inexact: the sign of the root's square-free factor at lo (at hi: -sign_lo) */
    int32_t factor;              /* its square-free factor (multiplicity - 1) */
    int32_t how;                 /* TR_INV_HOW_* */
} tr_inv_root_info;              /* 24 bytes */

/* ncoef coefficients of f (low degree first; NULL entries are 0), target (NULL = 0), bounds lo, hi (NULL = unbounded;
 * lo <= hi else INVALID) */
tr_inv_roots *tr_inv_isolate(int32_t ncoef, const tr_bigq *const *coef, const tr_bigq *target, const tr_bigq *lo,
                             const tr_bigq *hi, const tr_math_budget *, tr_math_refusal *, int32_t *error);
/* the same for a first-slice polynomial in one variable slot (a term in another variable: TR_MATH_DOMAIN_ERROR) */
tr_inv_roots *tr_inv_isolate_poly(const tr_math_poly *, int32_t variable, const tr_bigq *target, const tr_bigq *lo,
                                  const tr_bigq *hi, const tr_math_budget *, tr_math_refusal *, int32_t *error);
/* every inexact root's interval bisected to a width <= width (> 0); the signs come from fn (NULL: tr_inv_signs_eval).
 * Iterations: rounds x roots refined. A new object; the input is unchanged. */
tr_inv_roots *tr_inv_roots_refine(const tr_inv_roots *, const tr_bigq *width, tr_inv_signs_fn fn,
                                  const tr_math_budget *, tr_math_refusal *, int32_t *error);
int32_t tr_inv_roots_get_info(const tr_inv_roots *, tr_inv_roots_info *);
/* root i in increasing order: its description and its bounds (exact: *lo == *hi == the value; borrowed) */
int32_t tr_inv_roots_root(const tr_inv_roots *, int32_t i, tr_inv_root_info *, const tr_bigq **lo, const tr_bigq **hi);
/* square-free factor i (primitive, positive leading coefficient): its degree, and coefficient j (borrowed) */
int32_t tr_inv_roots_factor_degree(const tr_inv_roots *, int32_t i);
const tr_bigint *tr_inv_roots_factor_coef(const tr_inv_roots *, int32_t i, int32_t j);
/* the bounds used (NULL = unbounded) */
const tr_bigq *tr_inv_roots_bound(const tr_inv_roots *, int32_t hi);
void tr_inv_roots_free(tr_inv_roots *);

/* ------------------------------------------------------------------ finite domains */
typedef struct {
    uint64_t start, count;       /* grid points start .. start + count - 1 are examined */
    uint64_t total;              /* grid points: the product of the sizes */
    uint64_t nvalues;            /* values of every coordinate */
    uint64_t size[TR_INV_SWEEP_DIMS];
    uint64_t voff[TR_INV_SWEEP_DIMS];   /* coordinate d's values: vres[j * nvalues + voff[d] + i] */
    uint32_t nprimes, nterms;
    int32_t ndims, reserved;
    const tr_gfp *fields;        /* k */
    const uint8_t *exps;         /* nterms x ndims */
    const uint64_t *cres;        /* k x nterms: the integer coefficients mod p_j */
    const uint64_t *vres;        /* k x nvalues: the grid numerators mod p_j */
} tr_inv_sweep_view;             /* 128 bytes */
typedef struct {
    uint64_t total, start, searched, next, matches;
    uint64_t bound_bits;         /* the proof: |H(m)| < 2^bound_bits at every grid point */
    int32_t complete, reason;    /* TR_INV_SEARCH_* */
    int32_t ndims, nprimes;
} tr_inv_sweep_info;             /* 64 bytes */
typedef struct tr_inv_sweep_plan_s tr_inv_sweep_plan;
typedef struct tr_inv_sweep_s tr_inv_sweep;

/* f = target on the grid: variables[d] (distinct slots of f), sizes[d] values each (values concatenated, coordinate 0
 * first), grid points from index start on */
tr_inv_sweep_plan *tr_inv_sweep_prepare(const tr_math_poly *f, const tr_bigq *target, int32_t ndims,
                                        const int32_t *variables, const uint64_t *sizes, const tr_bigq *const *values,
                                        uint64_t start, const tr_math_budget *, tr_math_refusal *, int32_t *error);
int32_t tr_inv_sweep_get_view(const tr_inv_sweep_plan *, tr_inv_sweep_view *);
/* the CPU search: bit (i - start) of bitmap ((count + 63) / 64 words, zeroed here) set when grid point i matches */
int32_t tr_inv_sweep_eval(const tr_inv_sweep_view *, uint64_t *bitmap);
/* the answer from a bitmap (the CPU's or the device's): the matches in index order up to the output budget */
tr_inv_sweep *tr_inv_sweep_finish(const tr_inv_sweep_plan *, const uint64_t *bitmap, const tr_math_budget *,
                                  tr_math_refusal *, int32_t *error);
int32_t tr_inv_sweep_get_info(const tr_inv_sweep *, tr_inv_sweep_info *);
/* match i: its grid index and the value index of each coordinate */
int32_t tr_inv_sweep_match(const tr_inv_sweep *, uint64_t i, uint64_t *index, uint64_t *coords);
void tr_inv_sweep_plan_free(tr_inv_sweep_plan *);
void tr_inv_sweep_free(tr_inv_sweep *);
static inline uint64_t tr_inv_sweep_device_bytes(const tr_inv_sweep_view *v) {
    uint64_t k = v->nprimes, b = tr_budget_mul(k, sizeof(tr_gfp));
    b = tr_budget_add(b, tr_budget_mul((uint64_t)v->nterms, (uint64_t)(v->ndims > 0 ? v->ndims : 0)));
    b = tr_budget_add(b, tr_budget_mul(tr_budget_mul(k, v->nterms), 8));
    b = tr_budget_add(b, tr_budget_mul(tr_budget_mul(k, v->nvalues), 8));
    return tr_budget_add(b, tr_budget_bitmap_bytes(v->count));
}
uint64_t tr_inv_sweep_device_need(const tr_inv_sweep_view *);

/* well-formed views (host code of both libraries: the CPU evaluators and the CUDA entry points check the same) */
static inline int tr_inv_signs_view_valid(const tr_inv_signs_view *v) {
    if (!v || v->nprimes < 1 || v->nprimes > TR_INV_PRIMES_MAX || v->npolys < 1 || v->limbs < 1 ||
        v->limbs > TR_INV_POINT_LIMBS_MAX || !v->fields || !v->garner || !v->half || !v->offset || !v->coef)
        return 0;
    if (v->npoints && (!v->poly || !v->sign || !v->num || !v->den)) return 0;
    if (v->offset[0] != 0 || v->offset[v->npolys] != v->total) return 0;
    for (uint32_t f = 0; f < v->npolys; f++) if (v->offset[f + 1] <= v->offset[f]) return 0;
    for (uint32_t j = 0; j < v->nprimes; j++) if (!tr_gfp_consistent(&v->fields[j])) return 0;
    for (uint64_t i = 0; i < v->npoints; i++) if (v->poly[i] >= v->npolys) return 0;
    return 1;
}
static inline int tr_inv_sweep_view_valid(const tr_inv_sweep_view *v) {
    if (!v || v->ndims < 1 || v->ndims > TR_INV_SWEEP_DIMS || v->nprimes < 1 || v->nprimes > TR_INV_PRIMES_MAX ||
        !v->fields || (v->nterms && (!v->exps || !v->cres)) || !v->vres || v->start > v->total ||
        v->count > v->total - v->start) return 0;
    uint64_t total = 1, off = 0;
    for (int32_t d = 0; d < v->ndims; d++) {
        if (v->size[d] < 1 || v->voff[d] != off) return 0;
        total *= v->size[d]; off += v->size[d];
    }
    if (total != v->total || off != v->nvalues) return 0;
    for (uint32_t j = 0; j < v->nprimes; j++) if (!tr_gfp_consistent(&v->fields[j])) return 0;
    return 1;
}

/* ================================================================ shared per-point functions (CPU and CUDA) */
/* a magnitude of n little-endian limbs mod p: Horner over 2^64 with (r 2^64 + w) mod p = (r r64 + w) mod p */
TR_NUM_HD uint64_t tr_inv_limbs_mod(const tr_gfp *f, const uint64_t *w, uint32_t n) {
    uint64_t r = 0;
    for (uint32_t i = n; i-- > 0;) r = tr_gfp_reduce_u128_hd(f, r, w[i]);
    return r;
}

/* the sign (-1, 0, 1) of polynomial poly[i] at point i: residues of S = v^d P(u/v), Garner digits, comparison with
 * the digits of (M - 1) / 2 from the most significant one */
TR_NUM_HD int tr_inv_sign_point(const tr_inv_signs_view *v, uint64_t i) {
    uint64_t dig[TR_INV_PRIMES_MAX];
    const uint32_t k = v->nprimes, f = v->poly[i], c0 = v->offset[f], d = v->offset[f + 1] - c0 - 1;
    const uint64_t *num = v->num + i * (uint64_t)v->limbs, *den = v->den + i * (uint64_t)v->limbs;
    int zero = 1;
    for (uint32_t j = 0; j < k; j++) {
        const tr_gfp *F = &v->fields[j];
        uint64_t u = tr_inv_limbs_mod(F, num, v->limbs), w = tr_inv_limbs_mod(F, den, v->limbs);
        if (v->sign[i] < 0) u = tr_gfp_neg_hd(F, u);
        const uint64_t *c = v->coef + (uint64_t)j * v->total + c0;
        uint64_t h = c[d], wp = 1;
        for (uint32_t t = d; t-- > 0;) {                       /* h = sum c_s u^(s-t) w^(d-s) over s >= t */
            wp = tr_gfp_mul_hd(F, wp, w);
            h = tr_gfp_add_hd(F, tr_gfp_mul_hd(F, h, u), tr_gfp_mul_hd(F, c[t], wp));
        }
        if (h) zero = 0;
        const uint64_t *g = v->garner + (uint64_t)j * k;       /* digit j = (((h - d0) / p0 - d1) / p1 ...) mod p_j */
        uint64_t x = h;
        for (uint32_t t = 0; t < j; t++) {
            uint64_t dt = dig[t] >= F->p ? dig[t] - F->p : dig[t];   /* p_t < 2 p_j: one subtraction reduces */
            x = tr_gfp_mul_hd(F, tr_gfp_sub_hd(F, x, dt), g[t]);
        }
        dig[j] = x;
    }
    if (zero) return 0;
    for (uint32_t j = k; j-- > 0;)
        if (dig[j] != v->half[j]) return dig[j] < v->half[j] ? 1 : -1;
    return 1;                                                 /* S mod M == H: positive */
}

/* 1 when grid point idx is an exact solution: H(m) mod p_j == 0 for every modulus */
TR_NUM_HD int tr_inv_sweep_point(const tr_inv_sweep_view *v, uint64_t idx) {
    uint64_t digit[TR_INV_SWEEP_DIMS] = {0, 0, 0}, rem = idx;
    for (int32_t d = v->ndims; d-- > 0;) { digit[d] = rem % v->size[d]; rem /= v->size[d]; }
    for (uint32_t j = 0; j < v->nprimes; j++) {
        const tr_gfp *F = &v->fields[j];
        const uint64_t *vr = v->vres + (uint64_t)j * v->nvalues, *c = v->cres + (uint64_t)j * v->nterms;
        uint64_t x[TR_INV_SWEEP_DIMS] = {0, 0, 0}, s = 0;
        for (int32_t d = 0; d < v->ndims; d++) x[d] = vr[v->voff[d] + digit[d]];
        for (uint32_t t = 0; t < v->nterms; t++) {
            uint64_t m = c[t];
            const uint8_t *e = v->exps + (uint64_t)t * (uint64_t)v->ndims;
            for (int32_t d = 0; d < v->ndims && m; d++)
                for (uint32_t q = 0; q < e[d]; q++) m = tr_gfp_mul_hd(F, m, x[d]);
            s = tr_gfp_add_hd(F, s, m);
        }
        if (s) return 0;                                       /* a nonzero residue: H(m) != 0 */
    }
    return 1;
}

#ifdef __cplusplus
}
#endif
#endif
