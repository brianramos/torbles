/* DYN-001 / DYN-002: finite closed-observable dynamics and repeated operator powers, on the CPU (math_dyn.c) and in
 * CUDA (math_dyn_gpu.cu). Built on SYM-001 (math_sym.h): its declared geometry, its monomial bases, its exact affine
 * maps and their action matrices, its retained moment states and its 512-bit request evaluator.
 * SPDX-License-Identifier: Apache-2.0
 *
 * ================================================================ DYN-001: closed-observable dynamics
 * Declared dynamics x_{t+1} = F(x_t) act on the n declared coordinates of a SYM-001 geometry (tr_sym_geometry), in
 * LOGICAL units (value = logical / 10^scale; maps are declared in value units and converted exactly):
 *   AFFINE      an exact SYM-001 map (tr_sym_map: translations, scalings, reflections, permutations, affine maps,
 *               rotations by exact rational points of the unit circle or multiples of 90 degrees, and their
 *               compositions): x'_i = (sum_j a_ij x_j + b_i) / q_i (int128). Interval maps (irrational rotations) and
 *               value permutations are refused (TR_MATH_UNSUPPORTED): DYN-001 dynamics are exact.
 *   POLYNOMIAL  x'_i = sum_k c_ik theta^p_k x^e_k for every declared output coordinate i (undeclared coordinates are
 *               fixed), exact int64 rationals c in value units, total degree <= TR_DYN_MAX_DEGREE in x; with a declared
 *               parameter theta (parametric = 1) also degree p <= TR_DYN_MAX_PDEGREE in theta. Converted exactly to
 *               logical units (c 10^(s_i - sum_j s_j e_j)), merged, each coordinate's terms over one common int128
 *               denominator q_i (reduced). Only NUMERIC coordinates may be read or written (TR_MATH_DOMAIN_ERROR, as
 *               SYM-001's metric operations).
 * The observables are a SYM-001 basis B = (x^a)_a of nb monomials (tr_sym_basis, NUMERIC coordinates only). B is CLOSED
 * under F when every phi_a(F(x)) = prod_i F_i(x)^a_i lies in span(B) as a polynomial; the TRANSFER OPERATOR K is then
 * the exact matrix with
 *     phi(F(x)) = K phi(x) for every point x, hence moments(F(rows)) = K moments(rows), moments_t = K^t moments_0
 * (moments(rows) = sum_r w_r phi(x_r), any weights). EXACT operators are int128 numerators N_j over one common
 * denominator D > 0, canonical (gcd of D and every entry 1); K = N / D (parametric: K(theta) = sum_j theta^j N_j / D).
 * AFFINE maps take SYM-001's action matrix (tr_sym_action_make: its closure decision, witness and canonical (N, D));
 * POLYNOMIAL maps are expanded here exactly (int128, every product and sum checked: TR_MATH_OVERFLOW past it). The
 * closure decision is exact and complete: the first basis term whose image has a monomial outside B is refused with
 * TR_MATH_DOMAIN_ERROR and the witness (tr_sym_closure: the term and the missing monomial, degrees up to 64).
 * A nonlinear F can have a closed basis (e.g. x' = x/2, y' = y/3 + x^2 on {1, x, y, x^2}: Koopman-invariant
 * observables); a basis that is not closed is refused unless the caller asks for a TRUNCATION under an explicit
 * approximation contract:
 *   TRUNCATED   K^ keeps the terms of every phi_a(F(x)) that lie in B and drops the others. The contract is a declared
 *               box (tr_dyn_box: lo_i <= x_i <= hi_i, exact rationals in value units, for every coordinate of the reach
 *               set: the basis coordinates closed under "F_i reads x_j"; an unbounded one is TR_MATH_DOMAIN_ERROR) and
 *               its certified bound: for every x in the box and every basis term a,
 *                   |phi_a(F(x)) - (K^ phi(x))_a| <= eps_a = sum_dropped |c| prod_i M_i^b_i,  M_i = max(|lo_i|, |hi_i|)
 *               (exact FND-009 big rationals, logical units). The box is checked F-invariant (F(box) inside box) by exact
 *               rational interval arithmetic: when it is, every evolved row stays in the box (`invariant`), else only
 *               one step is certified. A truncated operator is APPROXIMATE (label TR_DYN_LABEL_CERTIFIED = 5, UNC-002's
 *               value, after FND-007's kinds): tr_dyn_authorize_exact refuses it (TR_MATH_DOMAIN_ERROR, decision
 *               reason TR_CERT_REFUSE_APPROXIMATE), its evolution carries the error bound, and every exact use (powers,
 *               characteristic polynomials, recurrences, mixtures, substitution, CUDA) refuses it (TR_MATH_DOMAIN_ERROR):
 *               uncertainty is never dropped by composition. A parametric map cannot be truncated (UNSUPPORTED).
 * Mixtures (DYN-002 parameterized evolution): K = sum_j w_j K_j for exact rational weights w_j >= 0 with sum w_j = 1
 * (checked exactly) over exact operators of one basis and geometry: the expected moments of a random dynamics choosing
 * F_j with probability w_j independently at each step. Substitution: K(theta0) of a parametric operator at an exact
 * rational theta0 (int128, canonical).
 *
 * Evolution of retained moment states (tr_sym_state: an FND-004 bank or a TENSOR-001 moment tensor, SYM-001), never
 * touching rows: the state and the operator must share the basis and the scope (geometry digest; an abstract
 * geometry has no state), and m_t = K^t m_0 is returned exactly per group as integer numerators over D^t
 * (num = N^t m_0, unique, so every method gives the same integers):
 *   STEPPING   t exact big-integer products v <- N v (FND-009-sized limbs, no allocation per operation);
 *   SQUARING   N^t by exact repeated squaring (DYN-002), then N^t m_0;
 *   MODULAR    (N^t m_0) mod p for k primes of LIN-002 (tr_lin_primes, [2^61, 2^62)), Chinese remaindering within the
 *              proven bound |N^t m_0| <= ||N||_inf^t max|m_0| (symmetric range, sum of log2 p > bound + 1) and the
 *              check of every value modulo TR_DYN_CHECK_PRIMES further primes (tr_dyn_evolution_adopt: the device
 *              residues are adopted only when the check holds, TR_MATH_INVALID otherwise);
 *   AUTO       the method of the smallest estimated cost: each method's structural work (loop steps, the nonzero
 *              products of N's pattern, limb products from the proven sizes, residues, Garner steps, primes) times
 *              unit costs fitted to timed runs, in 1/16 ns (the estimates are in the info; they bound nothing).
 * A truncated operator's evolution: m^_t = K^^t m_0 and the certified componentwise bound E_t with
 *     |moments_t(rows) - m^_t| <= E_t,   E_0 = 0,  E_t = |K^| E_{t-1} + W eps   (W: the group's contributing rows)
 * under the stated assumptions, each checked: the state unweighted (TR_MATH_UNSUPPORTED otherwise), the scope it was
 * built from given (equal digest) with every bounded coordinate's selected extremes inside the box
 * (TR_MATH_DOMAIN_ERROR otherwise), and steps <= 1 unless the box is F-invariant (TR_MATH_DOMAIN_ERROR).
 * A parametric operator's evolution keeps the moments as polynomials in theta (degree t p): tr_dyn_evolution_at
 * substitutes an exact rational theta0 (equal to evolving K(theta0)).
 * Explicit row evolution (tr_dyn_rows_evolve, the reference): every selected row whose map and basis coordinates
 * (and weight) are present is evolved exactly for t steps (one common denominator Q_t for all rows: x_t = X_t / Q_t,
 * Q_{t+1} = L Q_t^d, L = lcm q_i, d the map's degree) and its moments summed per group: exact numerators over
 * Q_t^|a| per basis term.
 *
 * ================================================================ DYN-002: powers, semigroups, recurrences
 * Powers K^n (tr_dyn_power: integer numerators N^n over D^n, unique, never reduced; K^0 = I over 1):
 *   SQUARING   exact repeated squaring in big integers (binary method, every intermediate bounded before allocation);
 *   MODULAR    N^n mod p per prime (the CUDA kernels: one work item per (operator, prime, entry) and product), Chinese
 *              remaindering within |N^n|_max <= ||N||_inf^n and the check modulo further primes (tr_dyn_power_adopt).
 * Composition (tr_dyn_power_compose: (A_num B_num) / (A_den B_den)) and exact equality (cross products: A_num B_den
 * = B_num A_den entrywise) make the semigroup identities checkable: tr_dyn_semigroup verifies K^0 = I, K^1 = K and
 * K^m K^n = K^(m+n) exactly.
 * Characteristic polynomial (tr_dyn_charpoly): chi_K(x) = det(x I - K) = sum_k a_k D^(k - nb) x^k with chi_N(y) =
 * sum_k a_k y^k computed modulo primes (Hessenberg reduction and its recurrence, O(nb^3) per prime), reconstructed
 * within |a_(nb-k)| <= C(nb, k) R^k (R = ||N||_inf, Hadamard) and checked modulo a further prime.
 * Recurrences (tr_dyn_recurrence): the sequence s_t = c^T K^t v (c an int64 functional, v int128: a state's group
 * moments or given) satisfies s_(t+d) = sum_(k<d) r_k s_(t+k) for
 *   CHARPOLY   d = nb, the coefficients of chi_K (Cayley-Hamilton);
 *   MINIMAL    the minimal recurrence of the sequence: Berlekamp-Massey over Q (FND-009 rationals) on u_t = c^T N^t v,
 *              t < 2 nb (its linear complexity is <= nb, so 2 nb terms determine it); its coefficients are integers
 *              (Gauss: a monic divisor of chi_N), checked, and the recurrence is verified on every computed term.
 * Any term s_T = u_T / D^T is evaluated by polynomial reduction (x^T mod the recurrence's polynomial by square and
 * multiply in Z[x], O(d^2 log T) big-integer operations): u_T = sum_k w_k u_k.
 * Floating powers (tr_dyn_float_power, APPROXIMATE): K^n (or K^n m_g for every group of a state) by repeated squaring
 * in outward-rounded binary64 interval arithmetic (FND-009), from tight enclosures of N / D: every returned midpoint
 * m has a rigorous bound |exact - m| <= r. The declared tolerance (ABS: max r <= tol; REL: max r <= tol max|m|) is
 * checked after every product; a bound past it, or a non-finite one, is UNSTABLE: refused (TR_MATH_NOT_CONVERGED,
 * the info filled with the exponent reached and the bound), never returned.
 *
 * ================================================================ budgets and errors
 * Budgets (math_budgets.h), checked before allocation (TR_MATH_BUDGET with the tr_math_refusal; tr_dyn_need reports
 * every need): maps: variables, degree, support = terms, bytes; operators: variables, degree (the image degree),
 * support = nb, tensor = (p + 1) nb^2, states = the largest expansion, bytes; powers: support = nb, tensor = nb^2,
 * iterations = n (the step count), output = nb^2, bytes = every big number from its proven bit bound (coefficient
 * growth: n bitlen(||N||_inf) bits per numerator, n bitlen(D) for the denominator) and the scratch (modular: the
 * residues); evolutions: iterations = t, output = groups x nb (x (t p + 1) parametric; with the errors of a truncation
 * twice), states = groups x nb, degree = t p, bytes as powers; charpoly / recurrences: support = nb, iterations = the
 * terms computed, output = the coefficients, bytes; terms: iterations = T, bytes; floating powers: tensor = nb^2,
 * iterations = n, output, bytes; CUDA: bytes = every device copy (tr_dyn_*_device_bytes).
 * Errors: TR_MATH_INVALID (malformed input), _MEMORY, _OVERFLOW (an exact int128 value past its carrier), _BUDGET,
 * TR_MATH_DOMAIN_ERROR (a basis not closed: the witness; an undeclared or categorical coordinate; a state of another
 * scope or basis; an approximate operator in an exact use; a contract that does not hold), _UNSUPPORTED (beyond the
 * representation limits, interval or value maps, weighted truncation, a parametric truncation), _NOT_CONVERGED (an
 * unstable floating power), TR_MATH_DEVICE (CUDA). Objects are immutable and individually owned; constructors return
 * NULL and set *error, leaving their inputs unchanged; read-only calls may run concurrently. */
#ifndef TORBLES_MATH_DYN_H
#define TORBLES_MATH_DYN_H
#include <stdint.h>
#include <string.h>
#include "math_num.h"
#include "math_budgets.h"
#include "math_scope.h"
#include "math_carriers.h"
#include "math_certificates.h"
#include "math_sym.h"
#ifdef __cplusplus
extern "C" {
#endif

#define TR_DYN_ABI 1
#define TR_DYN_MAX_COORDS TR_SYM_MAX_COORDS   /* 32 declared coordinates */
#define TR_DYN_MAX_DEGREE 8                   /* total degree of a map term in the coordinates */
#define TR_DYN_MAX_PDEGREE 8                  /* degree of a map term in its parameter */
#define TR_DYN_MAX_TERMS 4096                 /* terms of a polynomial map (all coordinates) */
#define TR_DYN_MAX_OPERATORS 64               /* operators of a mixture or of one device batch */
#define TR_DYN_MAX_PRIMES 4096                /* = LIN-002's prime table */
#define TR_DYN_PRIME_BITS 61                  /* every prime lies in [2^61, 2^62) (LIN-002's tr_lin_primes) */
#define TR_DYN_CHECK_PRIMES 1                 /* primes beyond the bound: the check of a reconstruction */

enum { TR_DYN_MAP_AFFINE = 1, TR_DYN_MAP_POLYNOMIAL = 2 };                         /* map kinds */
enum { TR_DYN_OP_EXACT = 1, TR_DYN_OP_TRUNCATED = 2 };                             /* operator kinds */
enum { TR_DYN_FROM_MAP = 1, TR_DYN_FROM_MIXTURE = 2, TR_DYN_FROM_SUBSTITUTION = 3 };   /* where an operator comes from */
enum { TR_DYN_LABEL_EXACT = 0, TR_DYN_LABEL_CERTIFIED = 5 };                       /* 5 = UNC-002 TR_APPROX_LABEL_CERTIFIED */
enum { TR_DYN_AUTO = 0, TR_DYN_SQUARING = 1, TR_DYN_MODULAR = 2, TR_DYN_STEPPING = 3, TR_DYN_FLOAT = 4 };   /* methods */
enum { TR_DYN_CHARPOLY = 1, TR_DYN_MINIMAL = 2 };                                  /* recurrence kinds */
enum { TR_DYN_EV_OPERATOR = 1, TR_DYN_EV_ROWS = 2 };                               /* evolutions */
enum { TR_DYN_NORM_ABS = 1, TR_DYN_NORM_REL = 2 };                                 /* floating tolerances */
enum { TR_DYN_NEED_CPU = 0, TR_DYN_NEED_DEVICE = 1 };
enum { TR_DYN_REQ_POWER = 1, TR_DYN_REQ_EVOLVE = 2, TR_DYN_REQ_FLOAT = 3, TR_DYN_REQ_CHARPOLY = 4,
       TR_DYN_REQ_RECURRENCE = 5, TR_DYN_REQ_APPLY = 6 };                          /* tr_dyn_need requests */
enum { TR_DYN_PART_MOMENTS = 1, TR_DYN_PART_DENOMINATORS = 2, TR_DYN_PART_ERRORS = 3, TR_DYN_PART_ERROR_DEN = 4,
       TR_DYN_PART_NUMERATORS = 5, TR_DYN_PART_DENOMINATOR = 6, TR_DYN_PART_COEFFICIENTS = 7, TR_DYN_PART_INITIAL = 8,
       TR_DYN_PART_EPSILON = 9 };                                                  /* exported parts */

/* ------------------------------------------------------------------ structs crossing the ctypes boundary */
typedef struct {
    int32_t coord, pexp;                    /* the output coordinate (geometry index); the parameter's exponent */
    uint8_t exps[TR_DYN_MAX_COORDS];        /* exponents of the coordinates */
    tr_num_rational coef;                   /* exact coefficient, value units */
} tr_dyn_term;                              /* 56 bytes */

typedef struct {
    int32_t n, kind;                        /* coordinates; TR_DYN_MAP_* */
    int32_t degree, pdegree;                /* total degree in the coordinates; degree in the parameter */
    int32_t nterms, parametric;             /* stored terms (logical units, merged, identities included); 1: theta */
    uint32_t touched, reserved;             /* bit i: coordinate i is not fixed */
    uint64_t digest[2];                     /* the geometry's scope digest (0: an abstract geometry) */
    uint64_t bytes;
} tr_dyn_map_info;                          /* 56 bytes */

typedef struct {
    int32_t coord, pexp;
    uint8_t exps[TR_DYN_MAX_COORDS];
    tr_math_i128 num, den;                  /* logical units: num / den (den = the coordinate's q_i) */
} tr_dyn_map_term;                          /* 72 bytes */

typedef struct {
    int32_t n, reserved;                    /* coordinates of the geometry */
    uint32_t declared, reserved2;           /* bit i: coordinate i is bounded */
    tr_num_rational lo[TR_DYN_MAX_COORDS];  /* value units, lo <= hi */
    tr_num_rational hi[TR_DYN_MAX_COORDS];
} tr_dyn_box;                               /* 1040 bytes */

typedef struct {
    int32_t nb, n, kind, label;             /* basis size; coordinates; TR_DYN_OP_*; TR_DYN_LABEL_* */
    int32_t source, closed;                 /* TR_DYN_FROM_*; 1: the basis is closed under the map */
    int32_t pdegree, invariant;             /* K(theta) = sum_(j <= pdegree) theta^j N_j / D; truncated: box F-invariant */
    int32_t sym_action, reserved;           /* 1: SYM-001's tr_sym_action_make built the matrix */
    tr_math_i128 den;                       /* D > 0 */
    uint64_t nonzeros, dropped;             /* nonzero entries of every N_j; truncated: dropped expansion terms */
    uint64_t norm_bits;                     /* bitlen(||N||_inf) (parametric: of sum_j |N_j|) */
    uint64_t digest[2];                     /* the geometry's scope digest */
    uint64_t bytes;
} tr_dyn_op_info;                           /* 104 bytes */

typedef struct {
    uint64_t nb, exponent;                  /* the matrix size; n (composites: the sum of the exponents) */
    int32_t method, primes;                 /* TR_DYN_SQUARING / _MODULAR; primes used (reconstruction + check) */
    int32_t identity, checked;              /* 1: equals I; modular: 1 when the check primes agreed */
    int32_t backend, reserved;              /* TR_CERT_BY_CPU / _DEVICE (who computed the residues) */
    uint64_t bound_bits, max_bits, den_bits;   /* the proven numerator bound; the largest actual numerator; D^n */
    uint64_t products, bytes;               /* matrix products performed (per prime for MODULAR) */
    uint64_t cost_squaring, cost_modular;   /* the estimates AUTO compares (1/16 ns) */
    uint64_t digest[2];
} tr_dyn_power_info;                        /* 112 bytes */

typedef struct {
    uint64_t ngroups, steps;
    int32_t nb, method, label, kind;        /* TR_DYN_LABEL_*; TR_DYN_EV_* */
    int32_t primes, invariant;              /* modular primes; truncated: the box was F-invariant */
    int32_t pdegree, checked;               /* parametric: the moments' degree in theta; modular check */
    int32_t backend, weighted;              /* TR_CERT_BY_*; 1: weighted moments */
    uint64_t bound_bits, max_bits, bytes;   /* proven bound of the numerators; the largest actual one */
    uint64_t rows, used;                    /* selected and contributing rows (all groups) */
    uint64_t cost_stepping, cost_squaring, cost_modular;   /* the estimates AUTO compares (1/16 ns) */
} tr_dyn_evolve_info;                       /* 120 bytes */

typedef struct {
    uint64_t exponent, reached;             /* n; the exponent certified within the tolerance (n when stable) */
    int32_t norm, stable;                   /* TR_DYN_NORM_*; 1: within the tolerance */
    int32_t products, groups;               /* interval matrix products; groups (moments) or 0 (the power) */
    double tolerance, error, magnitude;     /* declared; RU of the certified error (ABS: max r; REL: max r / max |m|);
                                               RU of max |m| */
} tr_dyn_float_info;                        /* 56 bytes */

typedef struct {
    int32_t kind, degree, nb, primes;       /* TR_DYN_CHARPOLY / _MINIMAL; d; charpoly: primes used */
    int32_t sequence, checked;              /* 1: a sequence (c, v) is attached; 1: the checks held */
    uint64_t terms, bound_bits, bytes;      /* sequence terms computed; charpoly: the coefficient bound */
} tr_dyn_rec_info;                          /* 48 bytes */

typedef struct {
    uint64_t m, n;
    int32_t identity, one, product, holds;  /* K^0 = I; K^1 = K; K^m K^n = K^(m+n); all three */
    int32_t method, reserved;
    uint64_t products;
} tr_dyn_semigroup_evidence;                /* 48 bytes */

/* views of host memory passed to the CUDA library (pointers swapped for device copies there) */
typedef struct {
    int32_t nb, nops;
    uint64_t ngroups;
    const tr_math_i128 *num;                /* nops x nb x nb, row-major: the operators' numerators */
    const tr_math_i128 *moments;            /* ngroups x nb */
} tr_dyn_apply_view;                        /* 32 bytes */

typedef struct {
    int32_t nb, nops;
    uint64_t nprimes, exponent;
    const tr_math_i128 *num;                /* nops x nb x nb */
    const tr_gfp *primes;                   /* nprimes fields (tr_gfp_field) */
} tr_dyn_mod_view;                          /* 40 bytes */

typedef struct tr_dyn_map_s tr_dyn_map;
typedef struct tr_dyn_operator_s tr_dyn_operator;
typedef struct tr_dyn_power_s tr_dyn_power;
typedef struct tr_dyn_evolution_s tr_dyn_evolution;
typedef struct tr_dyn_recurrence_s tr_dyn_recurrence;

/* ------------------------------------------------------------------ shared helpers (CPU loops and kernels) */
/* x mod p in [0, p) for an exact int128 */
TR_NUM_HD uint64_t tr_dyn_reduce_i128(const tr_gfp *f, tr_math_i128 v) {
    __int128 x = tr_i128_get(v);
    unsigned __int128 m = x < 0 ? (unsigned __int128)0 - (unsigned __int128)x : (unsigned __int128)x;
    uint64_t r = tr_gfp_reduce_u128_hd(f, (uint64_t)(m >> 64), (uint64_t)m);
    return x < 0 ? tr_gfp_neg_hd(f, r) : r;
}
/* (A B)_ij mod p for nb x nb residue matrices (row-major) */
TR_NUM_HD uint64_t tr_dyn_mulmod_entry(const tr_gfp *f, const uint64_t *A, const uint64_t *B, int32_t nb, int32_t i,
                                       int32_t j) {
    uint64_t acc = 0;
    const uint64_t *a = A + (uint64_t)i * (uint64_t)nb;
    for (int32_t k = 0; k < nb; k++) {
        uint64_t x = a[k];
        if (!x) continue;
        uint64_t y = B[(uint64_t)k * (uint64_t)nb + (uint64_t)j];
        if (y) acc = tr_gfp_add_hd(f, acc, tr_gfp_mul_hd(f, x, y));
    }
    return acc;
}
/* (P m)_i mod p for a residue matrix P and reduced moments m (residues) */
TR_NUM_HD uint64_t tr_dyn_applymod_entry(const tr_gfp *f, const uint64_t *P, const uint64_t *m, int32_t nb, int32_t i) {
    uint64_t acc = 0;
    const uint64_t *a = P + (uint64_t)i * (uint64_t)nb;
    for (int32_t k = 0; k < nb; k++)
        if (a[k] && m[k]) acc = tr_gfp_add_hd(f, acc, tr_gfp_mul_hd(f, a[k], m[k]));
    return acc;
}
/* one exact entry of the batched application: out = sum_j N_op[i][j] m_g[j] (512-bit; never overflows within the
 * limits: |N|, |m| < 2^127, nb <= 4096); work item w = (op x ngroups + g) x nb + i. 1, or 0 on overflow. */
TR_NUM_HD int tr_dyn_apply_entry(const tr_dyn_apply_view *v, uint64_t w, tr_sym_wide *out) {
    const uint64_t nb = (uint64_t)v->nb, i = w % nb, g = (w / nb) % v->ngroups, op = w / (nb * v->ngroups);
    tr_sym_action_view a;
    a.carrier = TR_SYM_EXACT; a.permutation = 0; a.nb = v->nb; a.reserved = 0;
    a.den = tr_i128_set(1); a.num = v->num + op * nb * nb; a.iv = 0; a.index = 0;
    return tr_sym_row_exact(&a, v->moments + g * nb, v->nb, (int32_t)i, out);
}
/* host checks of the views (0 when well formed) and the device copies of each entry point (bytes, saturating) */
static inline int32_t tr_dyn_apply_view_check(const tr_dyn_apply_view *v) {
    if (!v || v->nb < 1 || v->nb > TR_SYM_MAX_BASIS || v->nops < 1 || v->nops > TR_DYN_MAX_OPERATORS ||
        v->ngroups < 1 || !v->num || !v->moments) return -1;
    return 0;
}
static inline int32_t tr_dyn_mod_view_check(const tr_dyn_mod_view *v) {
    if (!v || v->nb < 1 || v->nb > TR_SYM_MAX_BASIS || v->nops < 1 || v->nops > TR_DYN_MAX_OPERATORS ||
        v->nprimes < 1 || v->nprimes > TR_DYN_MAX_PRIMES || !v->num || !v->primes) return -1;
    for (uint64_t q = 0; q < v->nprimes; q++) if (!tr_gfp_consistent(&v->primes[q])) return -1;
    return 0;
}
static inline uint64_t tr_dyn_apply_device_bytes(const tr_dyn_apply_view *v) {
    uint64_t nb = (uint64_t)v->nb, e = tr_budget_mul(nb, nb);
    uint64_t b = tr_budget_mul(tr_budget_mul((uint64_t)v->nops, e), 16);
    b = tr_budget_add(b, tr_budget_mul(tr_budget_mul(v->ngroups, nb), 16));
    b = tr_budget_add(b, tr_budget_mul(tr_budget_mul(tr_budget_mul((uint64_t)v->nops, v->ngroups), nb), sizeof(tr_sym_wide)));
    return tr_budget_add(b, 8);
}
/* powmod (ngroups 0) or evolvemod: the operators, the primes, three residue matrices per (operator, prime), and for
 * evolvemod the moments, their residues and the outputs */
static inline uint64_t tr_dyn_mod_device_bytes(const tr_dyn_mod_view *v, uint64_t ngroups) {
    uint64_t nb = (uint64_t)v->nb, e = tr_budget_mul(nb, nb), nmat = tr_budget_mul((uint64_t)v->nops, v->nprimes);
    uint64_t b = tr_budget_mul(tr_budget_mul((uint64_t)v->nops, e), 16);
    b = tr_budget_add(b, tr_budget_mul(v->nprimes, sizeof(tr_gfp)));
    b = tr_budget_add(b, tr_budget_mul(tr_budget_mul(nmat, e), 24));
    if (ngroups) {
        uint64_t gn = tr_budget_mul(ngroups, nb);
        b = tr_budget_add(b, tr_budget_mul(gn, 16));
        b = tr_budget_add(b, tr_budget_mul(tr_budget_mul(v->nprimes, gn), 8));
        b = tr_budget_add(b, tr_budget_mul(tr_budget_mul(nmat, gn), 8));
    }
    return b;
}

/* ------------------------------------------------------------------ maps (libalg) */
/* an exact affine SYM-001 map of the geometry geo (the geometry the map was declared in) */
tr_dyn_map *tr_dyn_map_from_sym(const tr_sym_map *, const tr_sym_geometry *geo, const tr_math_budget *,
                                tr_math_refusal *, int32_t *error);
/* a polynomial map: nterms declared terms (value units; several terms per output coordinate; equal monomials merge) */
tr_dyn_map *tr_dyn_map_polynomial(const tr_sym_geometry *, const tr_dyn_term *terms, int32_t nterms, int32_t parametric,
                                  const tr_math_budget *, tr_math_refusal *, int32_t *error);
int32_t tr_dyn_map_get_info(const tr_dyn_map *, tr_dyn_map_info *);
int32_t tr_dyn_map_get_terms(const tr_dyn_map *, tr_dyn_map_term *out, uint64_t capacity);   /* logical units */
void tr_dyn_map_free(tr_dyn_map *);

/* ------------------------------------------------------------------ operators (libalg) */
/* the needs of tr_dyn_operator_make (truncate: the box, or NULL) */
int32_t tr_dyn_operator_need(const tr_sym_basis *, const tr_dyn_map *, const tr_dyn_box *truncate, tr_math_need *);
/* the transfer operator of map on basis: exact (truncate NULL; not closed: TR_MATH_DOMAIN_ERROR with *closure), or
 * truncated under the box contract. closure may be NULL. */
tr_dyn_operator *tr_dyn_operator_make(const tr_sym_basis *, const tr_dyn_map *, const tr_dyn_box *truncate,
                                      const tr_math_budget *, tr_math_refusal *, tr_sym_closure *closure, int32_t *error);
/* sum_j w_j K_j: exact operators of one basis and geometry, weights >= 0 summing to 1 exactly */
tr_dyn_operator *tr_dyn_operator_mixture(const tr_dyn_operator *const *ops, const tr_num_rational *weights, int32_t n,
                                         const tr_math_budget *, tr_math_refusal *, int32_t *error);
/* K(theta) of a parametric operator (a non-parametric one is copied) */
tr_dyn_operator *tr_dyn_operator_substitute(const tr_dyn_operator *, tr_num_rational theta, const tr_math_budget *,
                                            tr_math_refusal *, int32_t *error);
int32_t tr_dyn_operator_get_info(const tr_dyn_operator *, tr_dyn_op_info *);
/* N_j / D as a SYM-001 action view (pointers into the operator): SYM-001's evaluator applies it */
int32_t tr_dyn_operator_get_view(const tr_dyn_operator *, int32_t j, tr_sym_action_view *);
const tr_sym_basis *tr_dyn_operator_get_basis(const tr_dyn_operator *);       /* borrowed */
/* TR_MATH_OK when geo is exactly the operator's geometry (columns, kinds, units, scales, digest), else
 * TR_MATH_DOMAIN_ERROR: a state of the same scope declared with its coordinates in another order would otherwise
 * share the scope digest and even the exponent rows of a full basis (the Python layer checks every state's geometry) */
int32_t tr_dyn_operator_check_geometry(const tr_dyn_operator *, const tr_sym_geometry *geo);
/* the contract of a truncated operator (value units) and, per basis row, eps_a (TR_DYN_PART_EPSILON: packed
 * numerator / denominator pairs, see tr_dyn_export) */
int32_t tr_dyn_operator_get_box(const tr_dyn_operator *, tr_dyn_box *);
/* FND-007: exact operators are authorized (a polynomial identity over Q: kind TR_CERT_LAW, claim POLYNOMIAL);
 * truncated ones refused (TR_MATH_DOMAIN_ERROR, reason TR_CERT_REFUSE_APPROXIMATE, kind TR_DYN_LABEL_CERTIFIED) */
int32_t tr_dyn_authorize_exact(const tr_dyn_operator *, tr_cert_decision *);
void tr_dyn_operator_free(tr_dyn_operator *);

/* ------------------------------------------------------------------ requests: needs and method choice (libalg) */
/* the needs of a request (TR_DYN_REQ_*) on an operator: n = the exponent / steps / terms, state for EVOLVE and APPLY
 * (and the FLOAT moments; NULL: the power), method TR_DYN_* (AUTO is resolved first: *chosen), target
 * TR_DYN_NEED_CPU / _DEVICE. */
int32_t tr_dyn_need(int32_t request, const tr_dyn_operator *, const tr_sym_state *, uint64_t n, int32_t method,
                    int32_t target, tr_math_need *need, int32_t *chosen);
/* the primes a modular power (state NULL) or evolution needs: reconstruction + check, and the proven bound */
int32_t tr_dyn_mod_plan(const tr_dyn_operator *, const tr_sym_state *, uint64_t n, uint64_t *nprimes, uint64_t *bound_bits);
/* the first count primes below 2^62 (LIN-002's tr_lin_primes), each verified prime */
int32_t tr_dyn_primes(uint64_t count, tr_gfp *out);
/* the views of the CUDA entry points (pointers into the operator, the state and the primes) */
int32_t tr_dyn_mod_bind(const tr_dyn_operator *, uint64_t n, const tr_gfp *primes, uint64_t nprimes, tr_dyn_mod_view *);
int32_t tr_dyn_apply_bind(const tr_sym_state *, const tr_dyn_operator *, tr_dyn_apply_view *);
/* the CPU references of the kernels (equal to tr_mgpu_dyn_* bit for bit) */
int32_t tr_dyn_apply_view_eval(const tr_dyn_apply_view *, tr_sym_wide *out, const tr_math_budget *, tr_math_refusal *);
int32_t tr_dyn_powmod_view_eval(const tr_dyn_mod_view *, uint64_t *out, const tr_math_budget *, tr_math_refusal *);
int32_t tr_dyn_evolvemod_view_eval(const tr_dyn_mod_view *, const tr_math_i128 *moments, uint64_t ngroups,
                                   uint64_t *out, const tr_math_budget *, tr_math_refusal *);

/* ------------------------------------------------------------------ evolution (libalg) */
/* m_t = K^t m_0 for every group (method TR_DYN_*); a truncated operator needs scope (the state's) and gives the bound.
 * The state must be one of the operator's scope and basis (digest, basis); its geometry: tr_dyn_operator_check_geometry */
tr_dyn_evolution *tr_dyn_evolve(const tr_sym_state *, const tr_dyn_operator *, uint64_t steps, int32_t method,
                                const tr_scope *scope, const tr_math_budget *, tr_math_refusal *, int32_t *error);
/* the modular evolution from residues computed elsewhere (tr_mgpu_dyn_evolvemod for this operator: nprimes x ngroups
 * x nb), checked; backend TR_CERT_BY_* */
tr_dyn_evolution *tr_dyn_evolution_adopt(const tr_sym_state *, const tr_dyn_operator *, uint64_t steps,
                                         const tr_gfp *primes, uint64_t nprimes, const uint64_t *residues,
                                         int32_t backend, const tr_math_budget *, tr_math_refusal *, int32_t *error);
/* a parametric evolution at theta0 */
tr_dyn_evolution *tr_dyn_evolution_at(const tr_dyn_evolution *, tr_num_rational theta, const tr_math_budget *,
                                      tr_math_refusal *, int32_t *error);
/* the exact moments of the explicitly evolved rows (weight: a scope column or -1) */
int32_t tr_dyn_rows_need(const tr_scope *, const tr_dyn_map *, const tr_sym_basis *, uint64_t steps, int32_t weight,
                         tr_math_need *);
tr_dyn_evolution *tr_dyn_rows_evolve(const tr_scope *, const tr_dyn_map *, const tr_sym_basis *, uint64_t steps,
                                     int32_t weight, const tr_math_budget *, tr_math_refusal *, int32_t *error);
int32_t tr_dyn_evolution_get_info(const tr_dyn_evolution *, tr_dyn_evolve_info *);
int32_t tr_dyn_evolution_get_rows(const tr_dyn_evolution *, uint64_t *rows, uint64_t *used, uint64_t capacity);
int32_t tr_dyn_evolution_authorize_exact(const tr_dyn_evolution *, tr_cert_decision *);
void tr_dyn_evolution_free(tr_dyn_evolution *);

/* ------------------------------------------------------------------ powers (libalg) */
tr_dyn_power *tr_dyn_power_make(const tr_dyn_operator *, uint64_t n, int32_t method, const tr_math_budget *,
                                tr_math_refusal *, int32_t *error);
/* K^n from residues computed elsewhere (tr_mgpu_dyn_powmod for this operator: nprimes x nb x nb), checked */
tr_dyn_power *tr_dyn_power_adopt(const tr_dyn_operator *, uint64_t n, const tr_gfp *primes, uint64_t nprimes,
                                 const uint64_t *residues, int32_t backend, const tr_math_budget *, tr_math_refusal *,
                                 int32_t *error);
tr_dyn_power *tr_dyn_power_compose(const tr_dyn_power *after, const tr_dyn_power *before, const tr_math_budget *,
                                   tr_math_refusal *, int32_t *error);
int32_t tr_dyn_power_equal(const tr_dyn_power *, const tr_dyn_power *);      /* 1 / 0 (TR_MATH_INVALID) */
int32_t tr_dyn_power_get_info(const tr_dyn_power *, tr_dyn_power_info *);
void tr_dyn_power_free(tr_dyn_power *);
/* K^0 = I, K^1 = K, K^m K^n = K^(m+n), exactly, with powers made by method */
int32_t tr_dyn_semigroup(const tr_dyn_operator *, uint64_t m, uint64_t n, int32_t method, tr_dyn_semigroup_evidence *,
                         const tr_math_budget *, tr_math_refusal *);

/* ------------------------------------------------------------------ characteristic polynomials and recurrences */
tr_dyn_recurrence *tr_dyn_charpoly(const tr_dyn_operator *, const tr_math_budget *, tr_math_refusal *, int32_t *error);
/* s_t = c^T K^t v: functional c (nb int64), v = the moments of group `group` of state (or vector: nb int128 when
 * state is NULL) */
tr_dyn_recurrence *tr_dyn_recurrence_make(const tr_dyn_operator *, const int64_t *functional, const tr_sym_state *,
                                          uint64_t group, const tr_math_i128 *vector, int32_t kind,
                                          const tr_math_budget *, tr_math_refusal *, int32_t *error);
int32_t tr_dyn_recurrence_get_info(const tr_dyn_recurrence *, tr_dyn_rec_info *);
int32_t tr_dyn_recurrence_term_need(const tr_dyn_recurrence *, uint64_t T, tr_math_need *);
/* s_T exactly: packed numerator and denominator (see tr_dyn_export) */
int32_t tr_dyn_recurrence_term(const tr_dyn_recurrence *, uint64_t T, uint64_t *out, uint64_t capacity, uint64_t *words,
                               const tr_math_budget *, tr_math_refusal *);
void tr_dyn_recurrence_free(tr_dyn_recurrence *);

/* ------------------------------------------------------------------ floating powers (APPROXIMATE, certified) */
/* K^n (state NULL: mid / rad nb x nb) or K^n m_g (mid / rad ngroups x nb) */
int32_t tr_dyn_float_power(const tr_dyn_operator *, uint64_t n, const tr_sym_state *, double tolerance, int32_t norm,
                           double *mid, double *rad, tr_dyn_float_info *, const tr_math_budget *, tr_math_refusal *);

/* ------------------------------------------------------------------ exports
 * Packed exact integers: per number one header word (bits 0..62 the limb count L, bit 63 the sign) and its L
 * magnitude limbs, least significant first (zero: L = 0). object: an operator (EPSILON: nb numerator / denominator
 * pairs), a power (NUMERATORS nb x nb, DENOMINATOR), an evolution (MOMENTS ngroups x nb (x (pdegree + 1)),
 * DENOMINATORS nb, ERRORS ngroups x nb, ERROR_DEN 1) or a recurrence (COEFFICIENTS, INITIAL: d numerator / denominator
 * pairs). *words: the words written; with out NULL (capacity 0) only the words needed. */
int32_t tr_dyn_export(const void *object, int32_t part, uint64_t *out, uint64_t capacity, uint64_t *words);

/* ------------------------------------------------------------------ the CUDA library (math_dyn_gpu.cu)
 * tr_mgpu_dyn_apply: out[(op x ngroups + g) x nb + i] = sum_j N_op[i][j] m_g[j] exactly (512 bits), one work item per
 * output entry running tr_dyn_apply_entry; equal to tr_dyn_apply_view_eval. Budget: output, bytes = the device copies.
 * tr_mgpu_dyn_powmod: out[(op x nprimes + q) x nb^2 + i nb + j] = (N_op^exponent)_ij mod p_q: the residues reduced on
 * the device, then binary exponentiation driven by the host, every product one launch with one work item per
 * (operator, prime, entry) running tr_dyn_mulmod_entry; equal to tr_dyn_powmod_view_eval.
 * tr_mgpu_dyn_evolvemod: the same powers, then out[((op x nprimes + q) x ngroups + g) x nb + i] = (N_op^e m_g)_i mod
 * p_q, one work item per entry running tr_dyn_applymod_entry; equal to tr_dyn_evolvemod_view_eval.
 * Every output slot is written by one work item: no result depends on the thread order. 0 or a native code. */
int tr_mgpu_dyn_apply(const tr_dyn_apply_view *view, tr_sym_wide *out, const tr_math_budget *, tr_math_refusal *);
int tr_mgpu_dyn_powmod(const tr_dyn_mod_view *view, uint64_t *out, const tr_math_budget *, tr_math_refusal *);
int tr_mgpu_dyn_evolvemod(const tr_dyn_mod_view *view, const tr_math_i128 *moments, uint64_t ngroups, uint64_t *out,
                          const tr_math_budget *, tr_math_refusal *);
#ifdef __cplusplus
}
#endif
#endif
