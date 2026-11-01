/* SYM-001 / SYM-002: closed-basis transformation actions on polynomial moment states, and certified data symmetries
 * with orbit reductions, on the CPU (math_sym.c) and in CUDA (math_sym_gpu.cu).
 * SPDX-License-Identifier: Apache-2.0
 *
 * ================================================================ declared geometry
 * A geometry (tr_sym_geometry) is the only coordinate system any transformation sees: n <= TR_SYM_MAX_COORDS
 * declared coordinates, each a scope column (math_scope.h) with a declared kind and unit label:
 *   NUMERIC      a measured quantity: its value is the scope's exact decimal value (logical / 10^scale, the scale
 *                read from the scope), its unit a label (equal labels: commensurable coordinates);
 *   CATEGORICAL  a code with no metric (a dictionary or foreign-key code, any category): its unit is its domain label.
 * Nothing else is inferred: neither the order nor the adjacency of codes, nor a metric between columns. A
 * transformation names scope columns; a column that is not declared is refused (TR_MATH_DOMAIN_ERROR), and so is
 * every metric operation on a CATEGORICAL coordinate (translation, scaling, reflection, rotation, affine maps): only
 * coordinate permutations among coordinates of one kind and unit and (SYM-002) value permutations apply to it.
 * Rotations need two NUMERIC coordinates of the same unit; permutations coordinates of one kind and unit.
 *
 * ================================================================ SYM-001: closed-basis actions
 * Transformations (tr_sym_transform, declared in value units with exact int64 rationals):
 *   IDENTITY; TRANSLATE x_c += t_c; SCALE x_c *= f_c; REFLECT x_c -> 2 a_c - x_c; PERMUTE x'_{c_i} = x_{c_perm[i]};
 *   AFFINE x'_C = A x_C + b over k named coordinates (the others fixed);
 *   ROTATE of the plane (x, y) = (c_0, c_1) about a center (cx, cy): x' = cx + c (x - cx) - s (y - cy),
 *     y' = cy + s (x - cx) + c (y - cy), declared either by an exact rational point (c, s) of the unit circle
 *     (c^2 + s^2 = 1 verified exactly, e.g. (3/5, 4/5)) or by an angle in degrees (an exact rational): multiples of
 *     90 are exact; any other angle gives certified enclosures of cos and sin (Taylor series with a rigorous
 *     remainder in outward-rounded binary64 interval arithmetic, pi enclosed by its two neighbouring doubles), and
 *     the map and everything derived from it is an INTERVAL (approximate, certified, 53-bit endpoints);
 *   VALUES (SYM-002 only) a permutation of finitely many values of one coordinate (every other value fixed).
 * A map (tr_sym_map) is a transformation in LOGICAL units of the geometry: with D = diag(10^scale_i) the affine map
 * becomes x' = D A D^-1 x + D b, stored exactly per output coordinate as x'_i = (sum_j a_ij x_j + b_i) / q_i
 * (int128, q_i > 0, each row reduced by its gcd; TR_MATH_OVERFLOW past int128, never wrapped) or as interval rows.
 * Maps compose exactly: (T o S)(x) = T(S(x)); an interval operand makes the composition an interval map.
 *
 * Bases (tr_sym_basis): nb <= TR_SYM_MAX_BASIS distinct monomials x^a over the n coordinates (exponent rows, total
 * degree <= TR_SYM_MAX_DEGREE), in the caller's order; phi(x) = (x^a)_a. tr_sym_basis_full gives every monomial of
 * total degree <= d (homogeneous: exactly d) over a coordinate subset, graded and in FND-004's order. A basis is
 * CLOSED under a map T when every (T x)^a, a polynomial of degree <= |a|, lies in the span of the basis.
 *
 * Actions (tr_sym_action): the matrix M_T with phi(T x) = M_T phi(x) for every point x, hence
 *     moments(T(rows)) = M_T moments(rows)        (moments(rows) = sum_r w_r phi(x_r), any weights)
 * Row a of M_T is the expansion of (T x)^a = prod_i ((sum_j a_ij x_j + b_i) / q_i)^{a_i} (exact multinomial
 * expansion in int128, every intermediate checked). An expansion term outside the basis means the basis is not
 * closed under T: TR_MATH_DOMAIN_ERROR with the witness (the basis term and the missing monomial, tr_sym_closure).
 *   EXACT     M = N / D: int128 numerators N (nb x nb, row-major) and one common denominator D > 0, canonical (gcd of
 *             D and every entry 1): equal actions have identical (N, D). A coordinate permutation gives an index
 *             permutation of the basis (permutation = 1, index[i] with (M m)_i = m_index[i]).
 *   INTERVAL  every entry an enclosure (lo <= exact entry <= hi) of the action of the exact rotation, computed by the
 *             same expansion in interval arithmetic; a structurally zero entry is exactly [0, 0].
 * Composition M_{T o S} = M_T M_S (tr_sym_action_compose: exact products in 512 bits, reduced by the gcd with FND-009
 * big integers, narrowed to int128 or refused with TR_MATH_OVERFLOW; an interval operand gives interval products).
 * tr_sym_action_identity is I; tr_sym_action_equal compares exactly (EXACT: identical canonical (N, D); INTERVAL:
 * identical endpoints). The action of a composed map and the product of the actions are computed independently
 * and agree exactly (both are tested against transformed rows).
 *
 * States (tr_sym_state): retained per-group moments over a basis of NUMERIC coordinates, never rescanned:
 *   from a FND-004 bank (math_bank.h): its terms as the basis (every column with a positive exponent a declared
 *     NUMERIC coordinate); mode JOINT (listwise), or SQL when every term of every group was valid on all its rows
 *     (otherwise the moments cover different rows and cannot be transformed together: TR_MATH_DOMAIN_ERROR);
 *   from a TENSOR-001 moment tensor (math_tensor.h), LISTWISE, every axis listing the same columns: declared NUMERIC
 *     coordinates, plus at most once the constant (-1). With the constant the basis is every monomial of degree <= m
 *     (the tensor's order) over those coordinates, without it the homogeneous monomials of degree m (closed under
 *     linear maps only); the moment of x^a is the entry at the sorted multi-index with m - |a| constants.
 * Per group: rows (bag multiplicity, present iff rows > 0), used (contributing rows) and the moments (int128); the
 * constant term's moment is the contributing rows or, weighted, the weight sum. The state and the geometry must
 * come from one scope (equal descriptor digests).
 *
 * Requests (tr_sym_request, batched): (action index or -1 for the untransformed moments, group, a basis term or a
 * functional row F of int64 coefficients over the basis, flags MEAN). The answer is the exact rational
 *     F . (M_T m_g) = sum_i F_i sum_j N_ij m_gj / D             (TERM: the transformed moment of one term)
 * as a 512-bit two's complement numerator over a 512-bit denominator (> 0, not reduced), divided under MEAN by the
 * constant term's moment (the contributing rows or the weight sum; 0: empty; a basis without the constant term:
 * TR_MATH_UNSUPPORTED). With nb <= 4096 every product and sum fits 512 bits (|N|, |m| <= 2^127: |sum| < 2^341), and
 * the arithmetic still checks it. INTERVAL actions answer an enclosure (the moments enclosed exactly, then interval
 * products and sums with outward rounding). The CPU loop and the CUDA kernel share tr_sym_eval_request: the same
 * results bit for bit (the same operation order; the interval helpers of math_carriers.h use explicit fma).
 *
 * ================================================================ SYM-002: certified symmetries and orbits
 * A declared group (tr_sym_group) is generated by up to TR_SYM_MAX_GENERATORS maps that send the integer domain into
 * itself: affine maps whose logical form is integral (q_i = 1, int64 coefficients) and invertible (det A != 0, decided
 * exactly), e.g. coordinate permutations, reflections x -> 2a - x, rotations by multiples of 90 degrees, other
 * integral SYM-001 maps; or VALUES permutations. A map that is not integral or not invertible is refused
 * (TR_MATH_DOMAIN_ERROR), an interval map too.
 *
 * The support (tr_sym_support) of a scope over a geometry: the tuples (x_1..x_n) of the selected rows, as the sorted
 * distinct tuples with their multiplicities (rows) and smallest rows. NULL policy: SKIP (rows with a NULL coordinate
 * are excluded: the claim covers the complete rows) or FORBID (a NULL coordinate refuses the support:
 * TR_MATH_DOMAIN_ERROR with the smallest such row). It keeps an FND-007 record of the scope (tr_cert_record_scope
 * over the geometry's columns with the exact selection bits: identity, revision, rows, selection, NULL domains).
 *
 * Verification (exhaustive and exact): generator g preserves the multiset of rows iff every row's image occurs with
 * the multiplicity of the row's own tuple: mult(g(v)) = mult(v) for every support tuple v (g is injective, so this is
 * equivalent to equal multisets). An image outside int64 or absent is a MISSING orbit member; an image present with
 * another multiplicity a BROKEN multiplicity. Evidence per generator (tr_sym_evidence): rows and tuples checked,
 * violating rows (the multiplicities of the violating tuples), missing and broken tuples, the witness = the smallest
 * violating row, its tuple, its image's multiplicity and its kind; holds = no violation. The group is a symmetry iff
 * every generator holds. The CPU loop and the CUDA kernel (work items (generator, tuple slice), one atomic per
 * counter and an atomic minimum for the witness) share tr_sym_check_tuple and tr_sym_evidence_finish.
 *
 * Certificates (tr_sym_cert): made only when every generator holds, of FND-007 kind TR_CERT_SNAPSHOT: the group, the
 * evidence, the NULL policy, the backend that verified it and a copy of the support's record. Freshness
 * (tr_sym_cert_fresh): FND-007's exact comparison (tr_cert_record_fresh, TR_CERT_MATCH_EXACT) with every reason. A
 * broken multiplicity or a missing orbit member refuses the certificate (TR_MATH_DOMAIN_ERROR, the evidence with the
 * witness row); a certificate of data that changed is stale (TR_MATH_STALE) and authorizes nothing.
 *
 * Orbits (tr_sym_orbits), only under a certificate fresh for the current scope and made for this support: the
 * connected components of v ~ g(v) over the support (union-find; under the certificate every generator permutes the
 * support preserving multiplicities, so these are the orbits of the group), each with its representative (the
 * lexicographically smallest tuple), its size (distinct tuples) and its multiplicity (rows of each member: equal by
 * the certificate). Queries (tr_sym_poly_query): a polynomial p over the coordinates (int64 coefficients, canonical
 * terms), SUM (sum over the rows of p) or COUNT (rows with p OP constant). Under the reduction a query is answered
 * as sum over orbits of size x multiplicity x f(representative) only when it is proven invariant under every
 * generator as a law (not on a sample): p o g = p as polynomials for an affine g (the exact expansion of p(A x + b));
 * p with x_c = a equal to p with x_c = sigma(a) as polynomials for every value a a VALUES generator moves. A query not
 * proven invariant is refused under the reduction (status TR_MATH_DOMAIN_ERROR, the generator named); the exhaustive
 * evaluation over every selected row (tr_sym_rows_query) answers any query and is the check.
 *
 * ================================================================ budgets and errors
 * Budgets (math_budgets.h), checked before allocation: geometry and maps: variables = coordinates; bases: variables,
 * degree, support = nb, bytes; actions: variables, degree, support = nb, tensor = nb^2, bytes = the matrix and the
 * expansion scratch (bounded per row from the coordinates its expansion can reach); compositions: tensor = nb^2,
 * bytes = the 512-bit products and the result; states: states = groups x nb, bytes; requests: output = requests, bytes
 * = results (CUDA: the device copies); groups: support = moved values, bytes; supports: support = selected rows,
 * bytes = tuples, sort index and scratch, the record; verification: output = generators (CUDA: bytes = the device
 * copies); orbits: support = distinct tuples, bytes; queries: support = terms, degree, output = queries, bytes = the
 * invariance expansions. Refusal: TR_MATH_BUDGET with the tr_math_refusal.
 * Errors: TR_MATH_INVALID (malformed input), _MEMORY, _OVERFLOW (an exact value past its carrier: int128 map or action
 * entries, a query value), _BUDGET, TR_MATH_DOMAIN_ERROR (an undeclared column, a metric operation on a categorical
 * coordinate, incommensurable units, a basis not closed, moments of different rows, a group that is not a
 * symmetry, a query not invariant), _UNSUPPORTED (beyond the representation limits, a value permutation in an action,
 * MEAN without a constant term), TR_MATH_STALE, TR_MATH_DEVICE (CUDA). Objects are immutable and individually owned;
 * constructors return NULL and set *error, leaving their inputs unchanged; read-only calls may run concurrently.
 *
 * DYN-001 (finite closed-observable dynamics) builds on the public basis, maps, actions and their composition. */
#ifndef TORBLES_MATH_SYM_H
#define TORBLES_MATH_SYM_H
#include <stdint.h>
#include <string.h>
#include "math_num.h"
#include "math_budgets.h"
#include "math_scope.h"
#include "math_carriers.h"
#include "math_certificates.h"
#include "math_bank.h"
#include "math_tensor.h"
#ifdef __cplusplus
extern "C" {
#endif

#define TR_SYM_ABI 1
#define TR_SYM_MAX_COORDS 32              /* declared coordinates (= TR_SCOPE_MAX_COLUMNS) */
#define TR_SYM_MAX_DEGREE 8               /* total degree of a basis monomial (4-bit exponents) */
#define TR_SYM_MAX_BASIS 4096             /* monomials of a basis */
#define TR_SYM_MAX_GENERATORS 16          /* generators of a declared group */
#define TR_SYM_MAX_VALUES 65536           /* values one VALUES generator moves */
#define TR_SYM_MAX_TERMS 4096             /* terms of a query polynomial */
#define TR_SYM_WORDS 8                    /* 512-bit two's complement answers */
#define TR_SYM_NONE UINT64_MAX            /* no witness */
#define TR_SYM_PRECISION 53               /* interval endpoints: binary64 */

enum { TR_SYM_NUMERIC = 1, TR_SYM_CATEGORICAL = 2 };                         /* coordinate kinds */
enum { TR_SYM_T_IDENTITY = 0, TR_SYM_T_TRANSLATE = 1, TR_SYM_T_SCALE = 2, TR_SYM_T_REFLECT = 3,
       TR_SYM_T_PERMUTE = 4, TR_SYM_T_AFFINE = 5, TR_SYM_T_ROTATE = 6, TR_SYM_T_VALUES = 7 };   /* transformations */
enum { TR_SYM_ROT_COSSIN = 1, TR_SYM_ROT_DEGREES = 2 };                      /* how a rotation is declared */
enum { TR_SYM_MAP_AFFINE = 1, TR_SYM_MAP_VALUES = 2 };                       /* map kinds */
enum { TR_SYM_EXACT = 1, TR_SYM_INTERVAL = 2 };                              /* carriers of maps and actions */
enum { TR_SYM_FROM_BANK = 1, TR_SYM_FROM_TENSOR = 2 };                       /* sources of a state */
enum { TR_SYM_Q_TERM = 1, TR_SYM_Q_FUNCTIONAL = 2 };                         /* request kinds */
enum { TR_SYM_Q_MEAN = 1 };                                                  /* request flags */
enum { TR_SYM_GEN_AFFINE = 1, TR_SYM_GEN_VALUES = 2 };                       /* generator kinds */
enum { TR_SYM_NULL_FORBID = 1, TR_SYM_NULL_SKIP = 2 };                       /* = TR_CERT_NULL_FORBID / _SKIP */
enum { TR_SYM_HOLDS = 0, TR_SYM_MISSING = 1, TR_SYM_BROKEN = 2 };            /* a tuple under a generator */
enum { TR_SYM_SUM = 1, TR_SYM_COUNT = 2 };                                   /* query kinds */
enum { TR_SYM_LT = 1, TR_SYM_LE = 2, TR_SYM_EQ = 3, TR_SYM_NE = 4, TR_SYM_GE = 5, TR_SYM_GT = 6 };   /* COUNT ops */
enum { TR_SYM_NEED_CPU = 0, TR_SYM_NEED_DEVICE = 1 };

/* ------------------------------------------------------------------ structs crossing the ctypes boundary */
typedef struct {
    int32_t n, ncols;                       /* declared coordinates 1..32; the scope's columns */
    int32_t col[TR_SYM_MAX_COORDS];         /* scope column of coordinate i (distinct) */
    int32_t kind[TR_SYM_MAX_COORDS];        /* TR_SYM_NUMERIC / TR_SYM_CATEGORICAL */
    int32_t unit[TR_SYM_MAX_COORDS];        /* unit (NUMERIC) or domain (CATEGORICAL) label >= 0 */
    int32_t scale[TR_SYM_MAX_COORDS];       /* the column's decimal scale (from the scope) */
    uint64_t digest[2];                     /* the scope's descriptor digest (0: an abstract geometry) */
} tr_sym_geometry;                          /* 536 bytes */

typedef struct {
    int32_t kind, k;                        /* TR_SYM_T_*; columns named (scope columns, declared coordinates) */
    int32_t col[TR_SYM_MAX_COORDS];
    int32_t perm[TR_SYM_MAX_COORDS];        /* PERMUTE: coordinate col[i] receives the value of col[perm[i]] */
    int32_t mode, reserved;                 /* ROTATE: TR_SYM_ROT_COSSIN / TR_SYM_ROT_DEGREES */
    tr_num_rational value[TR_SYM_MAX_COORDS];   /* TRANSLATE offsets, SCALE factors, REFLECT centers (value units) */
    tr_num_rational cosine, sine, degrees;  /* ROTATE */
    tr_num_rational center[2];              /* ROTATE: the center (value units) */
    uint64_t nvalues;                       /* VALUES: the moved values */
    const tr_num_rational *A, *b;           /* AFFINE: k x k (row-major) and k (value units) */
    const tr_num_rational *keys, *targets;  /* VALUES: value -> its image (value units) */
} tr_sym_transform;                         /* 904 bytes */

typedef struct {
    int32_t term, closed;                   /* the basis term whose image leaves the basis (-1); 1: closed */
    int32_t reserved[2];
    uint8_t missing[TR_SYM_MAX_COORDS];     /* exponents of a monomial of that image outside the basis */
} tr_sym_closure;                           /* 48 bytes */

typedef struct {
    int32_t n, kind;                        /* coordinates; TR_SYM_MAP_* */
    int32_t carrier, integral;              /* TR_SYM_EXACT / _INTERVAL; 1: integer logical coefficients */
    int32_t permutation, invertible;        /* 1: a coordinate permutation; 1: det A != 0 (-1: undecided) */
    int32_t coord, precision;               /* VALUES: the coordinate; INTERVAL: 53, EXACT: 0 */
    uint32_t touched, reserved;             /* bit i: coordinate i is not fixed */
    uint64_t nvalues, bytes;
} tr_sym_map_info;                          /* 56 bytes */

typedef struct {
    int32_t n, nb, degree, constant;        /* coordinates; monomials; largest total degree; index of 1 (-1) */
    int32_t homogeneous, reserved;          /* 1: every monomial has the same degree */
    uint32_t coords, reserved2;             /* bit i: coordinate i has a positive exponent somewhere */
    uint64_t bytes;
} tr_sym_basis_info;                        /* 40 bytes */

typedef struct {
    int32_t nb, n, carrier, permutation;    /* basis size; coordinates; TR_SYM_EXACT / _INTERVAL; index permutation */
    int32_t monomial, identity, precision, approximate;   /* one nonzero per row; I; 53 for intervals; 1: interval */
    tr_math_i128 den;                       /* EXACT: the common denominator D > 0 */
    uint64_t nonzeros, bytes;
} tr_sym_action_info;                       /* 64 bytes */

typedef struct {
    int32_t carrier, permutation, nb, reserved;
    tr_math_i128 den;                       /* EXACT */
    const tr_math_i128 *num;                /* EXACT: nb x nb, row-major */
    const tr_math_interval *iv;             /* INTERVAL: nb x nb */
    const int32_t *index;                   /* permutation: (M m)_i = m_index[i] */
} tr_sym_action_view;                       /* 56 bytes */

typedef struct {
    uint64_t ngroups, present, rows, used, bytes;
    int32_t nb, n, source, weighted;        /* basis size; coordinates; TR_SYM_FROM_*; 1: weighted moments */
    int32_t constant, reserved;             /* the constant term's index (-1) */
    uint64_t digest[2];                     /* the scope's descriptor digest */
} tr_sym_state_info;                        /* 80 bytes */

typedef struct {
    uint64_t ngroups;
    int32_t nb, nactions, nfunc, constant;
    const tr_math_i128 *moments;            /* ngroups x nb */
    const tr_sym_action_view *actions;      /* nactions (device: views of device copies) */
    const int64_t *functionals;             /* nfunc x nb */
} tr_sym_query_view;                        /* 48 bytes */

typedef struct {
    int32_t action, kind;                   /* -1: no action; TR_SYM_Q_TERM / _FUNCTIONAL */
    int32_t index, flags;                   /* the term or functional row; TR_SYM_Q_MEAN */
    uint64_t group;
} tr_sym_request;                           /* 24 bytes */

typedef struct { uint64_t w[TR_SYM_WORDS]; } tr_sym_wide;   /* 512-bit two's complement, w[0] least significant */

typedef struct {
    int32_t status, empty;                  /* native code; 1: MEAN over a zero weight */
    int32_t carrier, reserved;              /* TR_SYM_EXACT / _INTERVAL */
    tr_sym_wide num, den;                   /* EXACT: num / den, den > 0, not reduced */
    tr_math_interval iv;                    /* INTERVAL: lo <= exact <= hi */
} tr_sym_result;                            /* 160 bytes */

typedef struct {
    int32_t kind, coord;                    /* TR_SYM_GEN_*; VALUES: the coordinate */
    uint64_t at, nvalues;                   /* AFFINE: A (n x n) and b (n) at coef[at]; VALUES: keys/targets[at] */
} tr_sym_gen;                               /* 24 bytes */

typedef struct {
    int32_t ngen, n;
    const tr_sym_gen *gens;
    const int64_t *coef;                    /* AFFINE: logical integer coefficients */
    const int64_t *keys, *targets;          /* VALUES: logical values, keys strictly increasing per generator */
} tr_sym_group_view;                        /* 40 bytes */

typedef struct {
    int32_t ngen, n;
    uint64_t ncoef, nvalues, bytes;
} tr_sym_group_info;                        /* 32 bytes */

typedef struct {
    uint64_t distinct;
    int32_t n, reserved;
    const int64_t *tuples;                  /* distinct x n, lexicographically increasing */
    const uint64_t *mult;                   /* rows of each tuple */
    const uint64_t *first;                  /* the smallest row of each tuple */
} tr_sym_support_view;                      /* 40 bytes */

typedef struct {
    uint64_t rows, selected, checked, excluded, distinct, bytes;   /* checked: complete rows; excluded: SKIP */
    int32_t n, policy;
} tr_sym_support_info;                      /* 56 bytes */

typedef struct {
    uint64_t rows, tuples;                  /* rows (sum of multiplicities) and distinct tuples checked */
    uint64_t violations;                    /* rows of the violating tuples */
    uint64_t missing, broken;               /* violating tuples: image absent; image of another multiplicity */
    uint64_t witness;                       /* the smallest violating row (TR_SYM_NONE) */
    uint64_t witness_tuple;                 /* its tuple (TR_SYM_NONE) */
    uint64_t image_mult;                    /* the multiplicity of the witness's image (0: missing) */
    int32_t generator, kind;                /* the generator; TR_SYM_MISSING / _BROKEN (TR_SYM_HOLDS without one) */
    int32_t holds, reserved;
} tr_sym_evidence;                          /* 80 bytes */

typedef struct {
    int32_t kind, backend;                  /* TR_CERT_SNAPSHOT; TR_CERT_BY_CPU / _DEVICE */
    int32_t ngen, n;
    int32_t policy, holds;
    uint64_t rows, selected, checked, excluded, distinct, bytes;
} tr_sym_cert_info;                         /* 72 bytes */

typedef struct {
    uint64_t norbits, distinct, rows, largest, smallest, bytes;    /* largest / smallest orbit size */
    int32_t n, ngen;
} tr_sym_orbits_info;                       /* 56 bytes */

typedef struct {
    uint64_t norbits, distinct;
    int32_t n, reserved;
    const uint64_t *rep;                    /* the representative's support tuple, per orbit */
    const uint64_t *size;                   /* distinct tuples per orbit */
    const uint64_t *mult;                   /* rows per member */
    const int64_t *tuples;                  /* the support tuples (distinct x n) */
    const uint64_t *orbit;                  /* the orbit of each support tuple */
} tr_sym_orbits_view;                       /* 64 bytes */

typedef struct {
    int32_t kind, op;                       /* TR_SYM_SUM / _COUNT; COUNT: TR_SYM_LT .. _GT */
    int32_t nterms, reserved;
    int64_t constant;                       /* COUNT: p(x) OP constant */
    const uint8_t *exponents;               /* nterms x n (logical units) */
    const int64_t *coefficients;            /* nterms */
} tr_sym_poly_query;                        /* 40 bytes */

typedef struct {
    int32_t status, invariant;              /* native code; 1: proven invariant under every generator */
    int32_t generator, reserved;            /* the first generator breaking invariance (-1) */
    uint64_t evaluations, rows;             /* polynomial evaluations performed; rows covered */
    tr_sym_wide value;                      /* SUM: the exact sum; COUNT: the rows */
} tr_sym_poly_answer;                       /* 96 bytes */

typedef struct tr_sym_basis_s tr_sym_basis;
typedef struct tr_sym_map_s tr_sym_map;
typedef struct tr_sym_action_s tr_sym_action;
typedef struct tr_sym_state_s tr_sym_state;
typedef struct tr_sym_group_s tr_sym_group;
typedef struct tr_sym_support_s tr_sym_support;
typedef struct tr_sym_cert_s tr_sym_cert;
typedef struct tr_sym_orbits_s tr_sym_orbits;

/* ------------------------------------------------------------------ 512-bit exact arithmetic (CPU and kernels) */
TR_NUM_HD void tr_sym_w_zero(tr_sym_wide *a) { for (int i = 0; i < TR_SYM_WORDS; i++) a->w[i] = 0; }
TR_NUM_HD void tr_sym_w_from_i128(__int128 x, tr_sym_wide *a) {
    uint64_t fill = x < 0 ? ~(uint64_t)0 : 0;
    a->w[0] = (uint64_t)x;
    a->w[1] = (uint64_t)((unsigned __int128)x >> 64);
    for (int i = 2; i < TR_SYM_WORDS; i++) a->w[i] = fill;
}
TR_NUM_HD int tr_sym_w_negative(const tr_sym_wide *a) { return (int)(a->w[TR_SYM_WORDS - 1] >> 63); }
TR_NUM_HD int tr_sym_w_is_zero(const tr_sym_wide *a) {
    uint64_t o = 0;
    for (int i = 0; i < TR_SYM_WORDS; i++) o |= a->w[i];
    return o == 0;
}
/* two's complement negation (the caller keeps |a| < 2^511) */
TR_NUM_HD void tr_sym_w_negate(const tr_sym_wide *a, tr_sym_wide *out) {
    uint64_t carry = 1;
    for (int i = 0; i < TR_SYM_WORDS; i++) {
        uint64_t v = ~a->w[i] + carry;
        carry = carry && v == 0;
        out->w[i] = v;
    }
}
/* out = a + b; 0 on signed overflow (out is then unspecified) */
TR_NUM_HD int tr_sym_w_add(const tr_sym_wide *a, const tr_sym_wide *b, tr_sym_wide *out) {
    int sa = tr_sym_w_negative(a), sb = tr_sym_w_negative(b);
    uint64_t carry = 0;
    for (int i = 0; i < TR_SYM_WORDS; i++) {
        uint64_t s = a->w[i] + carry, c1 = s < carry;
        uint64_t t = s + b->w[i], c2 = t < s;
        out->w[i] = t;
        carry = c1 | c2;
    }
    return !(sa == sb && tr_sym_w_negative(out) != sa);
}
/* the exact product of two int128 values (|product| <= 2^254), sign-extended to 512 bits */
TR_NUM_HD void tr_sym_w_mul128(__int128 x, __int128 y, tr_sym_wide *out) {
    int neg = (x < 0) != (y < 0);
    unsigned __int128 a = x < 0 ? (unsigned __int128)0 - (unsigned __int128)x : (unsigned __int128)x;
    unsigned __int128 b = y < 0 ? (unsigned __int128)0 - (unsigned __int128)y : (unsigned __int128)y;
    uint64_t a0 = (uint64_t)a, a1 = (uint64_t)(a >> 64), b0 = (uint64_t)b, b1 = (uint64_t)(b >> 64);
    unsigned __int128 p00 = (unsigned __int128)a0 * b0, p01 = (unsigned __int128)a0 * b1;
    unsigned __int128 p10 = (unsigned __int128)a1 * b0, p11 = (unsigned __int128)a1 * b1;
    unsigned __int128 mid = (p00 >> 64) + (uint64_t)p01 + (uint64_t)p10;
    unsigned __int128 top = (mid >> 64) + (p01 >> 64) + (p10 >> 64) + (uint64_t)p11;
    tr_sym_wide r;
    tr_sym_w_zero(&r);
    r.w[0] = (uint64_t)p00;
    r.w[1] = (uint64_t)mid;
    r.w[2] = (uint64_t)top;
    r.w[3] = (uint64_t)(top >> 64) + (uint64_t)(p11 >> 64);
    if (neg) tr_sym_w_negate(&r, out); else *out = r;
}
/* out = a * b; 0 when the exact product needs more than 511 bits of magnitude (out unspecified) */
TR_NUM_HD int tr_sym_w_mul(const tr_sym_wide *a, const tr_sym_wide *b, tr_sym_wide *out) {
    int na = tr_sym_w_negative(a), nb = tr_sym_w_negative(b);
    tr_sym_wide ma, mb;
    if (na) tr_sym_w_negate(a, &ma); else ma = *a;
    if (nb) tr_sym_w_negate(b, &mb); else mb = *b;
    if (tr_sym_w_negative(&ma) || tr_sym_w_negative(&mb)) return 0;            /* -2^511: out of range */
    uint64_t r[2 * TR_SYM_WORDS];
    for (int i = 0; i < 2 * TR_SYM_WORDS; i++) r[i] = 0;
    for (int i = 0; i < TR_SYM_WORDS; i++) {
        if (!ma.w[i]) continue;
        uint64_t carry = 0;
        for (int j = 0; j < TR_SYM_WORDS; j++) {
            unsigned __int128 t = (unsigned __int128)ma.w[i] * mb.w[j] + r[i + j] + carry;
            r[i + j] = (uint64_t)t;
            carry = (uint64_t)(t >> 64);
        }
        r[i + TR_SYM_WORDS] = carry;
    }
    for (int i = TR_SYM_WORDS; i < 2 * TR_SYM_WORDS; i++) if (r[i]) return 0;
    if (r[TR_SYM_WORDS - 1] >> 63) return 0;
    tr_sym_wide p;
    for (int i = 0; i < TR_SYM_WORDS; i++) p.w[i] = r[i];
    if (na != nb) tr_sym_w_negate(&p, out); else *out = p;
    return 1;
}

/* ------------------------------------------------------------------ enclosures of exact values (CPU and kernels) */
/* [RD(x), RU(x)] of an int64 (a point when |x| < 2^53: exactly representable) */
TR_NUM_HD tr_math_interval tr_sym_iv_i64(int64_t x) {
    tr_math_interval r;
    double d = (double)x;
    if (x > -((int64_t)1 << 53) && x < ((int64_t)1 << 53)) { r.lo = d; r.hi = d; return r; }
    r.lo = tr_carrier_pred(d); r.hi = tr_carrier_succ(d);     /* the conversion rounds by at most half an ulp */
    return r;
}
TR_NUM_HD tr_math_interval tr_sym_iv_u64(uint64_t x) {
    tr_math_interval r;
    double d = (double)x;
    if (x < ((uint64_t)1 << 53)) { r.lo = d; r.hi = d; return r; }
    r.lo = tr_carrier_pred(d); r.hi = tr_carrier_succ(d);
    return r;
}
/* an enclosure of an int128: hi 2^64 + lo with both parts enclosed (scaling by 2^64 is exact) */
TR_NUM_HD tr_math_interval tr_sym_iv_i128(__int128 x) {
    if (x > -((__int128)1 << 53) && x < ((__int128)1 << 53)) return tr_sym_iv_i64((int64_t)x);
    tr_math_interval h = tr_sym_iv_i64((int64_t)(x >> 64)), l = tr_sym_iv_u64((uint64_t)x), s;
    s.lo = 18446744073709551616.0; s.hi = s.lo;
    return tr_interval_add(tr_interval_mul(h, s), l);
}

/* ------------------------------------------------------------------ the request evaluator (CPU and kernels) */
/* row i of the action applied to group moments m: an exact numerator (the action's denominator is separate) */
TR_NUM_HD int tr_sym_row_exact(const tr_sym_action_view *a, const tr_math_i128 *m, int32_t nb, int32_t i, tr_sym_wide *s) {
    if (!a) { tr_sym_w_from_i128(tr_i128_get(m[i]), s); return 1; }
    if (a->permutation) { tr_sym_w_from_i128(tr_i128_get(m[a->index[i]]), s); return 1; }
    tr_sym_w_zero(s);
    const tr_math_i128 *row = a->num + (uint64_t)i * (uint64_t)nb;
    for (int32_t j = 0; j < nb; j++) {
        __int128 c = tr_i128_get(row[j]);
        if (c == 0) continue;
        tr_sym_wide p;
        tr_sym_w_mul128(c, tr_i128_get(m[j]), &p);
        if (!tr_sym_w_add(s, &p, s)) return 0;
    }
    return 1;
}
TR_NUM_HD tr_math_interval tr_sym_row_interval(const tr_sym_action_view *a, const tr_math_i128 *m, int32_t nb, int32_t i) {
    tr_math_interval s;
    s.lo = 0.0; s.hi = 0.0;
    const tr_math_interval *row = a->iv + (uint64_t)i * (uint64_t)nb;
    for (int32_t j = 0; j < nb; j++) {
        if (row[j].lo == 0.0 && row[j].hi == 0.0) continue;
        __int128 x = tr_i128_get(m[j]);
        if (x == 0) continue;
        s = tr_interval_add(s, tr_interval_mul(row[j], tr_sym_iv_i128(x)));
    }
    return s;
}
/* One request (see the header comment): out->status 0, or TR_MATH_INVALID (-1: a request out of range),
 * TR_MATH_UNSUPPORTED (-12: MEAN without a constant term), TR_MATH_OVERFLOW (-3: unreachable within the limits) */
TR_NUM_HD void tr_sym_eval_request(const tr_sym_query_view *v, const tr_sym_request *r, tr_sym_result *out) {
    tr_sym_w_zero(&out->num);
    tr_sym_w_zero(&out->den);
    out->status = 0; out->empty = 0; out->carrier = TR_SYM_EXACT; out->reserved = 0;
    out->iv.lo = 0.0; out->iv.hi = 0.0;
    const int32_t nb = v->nb;
    if (r->group >= v->ngroups || r->action < -1 || r->action >= v->nactions || (r->flags & ~TR_SYM_Q_MEAN) ||
        (r->kind != TR_SYM_Q_TERM && r->kind != TR_SYM_Q_FUNCTIONAL) || r->index < 0 ||
        (r->kind == TR_SYM_Q_TERM && r->index >= nb) || (r->kind == TR_SYM_Q_FUNCTIONAL && r->index >= v->nfunc)) {
        out->status = -1;
        return;
    }
    if ((r->flags & TR_SYM_Q_MEAN) && v->constant < 0) { out->status = -12; return; }
    const tr_math_i128 *m = v->moments + r->group * (uint64_t)nb;
    const tr_sym_action_view *a = r->action >= 0 ? &v->actions[r->action] : 0;
    const int64_t *F = r->kind == TR_SYM_Q_FUNCTIONAL ? v->functionals + (uint64_t)r->index * (uint64_t)nb : 0;
    const __int128 cm = (r->flags & TR_SYM_Q_MEAN) ? tr_i128_get(m[v->constant]) : 1;
    if ((r->flags & TR_SYM_Q_MEAN) && cm == 0) { out->empty = 1; return; }
    if (a && a->carrier == TR_SYM_INTERVAL) {
        out->carrier = TR_SYM_INTERVAL;
        tr_math_interval acc;
        acc.lo = 0.0; acc.hi = 0.0;
        if (!F) acc = tr_sym_row_interval(a, m, nb, r->index);
        else
            for (int32_t i = 0; i < nb; i++) {
                if (!F[i]) continue;
                acc = tr_interval_add(acc, tr_interval_mul(tr_sym_iv_i64(F[i]), tr_sym_row_interval(a, m, nb, i)));
            }
        if (cm != 1) acc = tr_interval_div(acc, tr_sym_iv_i128(cm));
        out->iv = acc;
        return;
    }
    tr_sym_wide num, den, t, f;
    tr_sym_w_zero(&num);
    if (!F) {
        if (!tr_sym_row_exact(a, m, nb, r->index, &num)) { out->status = -3; return; }
    } else
        for (int32_t i = 0; i < nb; i++) {
            if (!F[i]) continue;
            if (!tr_sym_row_exact(a, m, nb, i, &t)) { out->status = -3; return; }
            tr_sym_w_from_i128((__int128)F[i], &f);
            if (!tr_sym_w_mul(&f, &t, &t) || !tr_sym_w_add(&num, &t, &num)) { out->status = -3; return; }
        }
    tr_sym_w_mul128(a ? tr_i128_get(a->den) : 1, cm, &den);
    if (tr_sym_w_negative(&den)) { tr_sym_w_negate(&den, &den); tr_sym_w_negate(&num, &num); }
    out->num = num; out->den = den;
}

/* ------------------------------------------------------------------ symmetry checks (CPU and kernels) */
TR_NUM_HD int tr_sym_tuple_cmp(const int64_t *a, const int64_t *b, int32_t n) {
    for (int32_t i = 0; i < n; i++) if (a[i] != b[i]) return a[i] < b[i] ? -1 : 1;
    return 0;
}
/* the support tuple equal to y, or TR_SYM_NONE (binary search of the sorted tuples) */
TR_NUM_HD uint64_t tr_sym_find(const tr_sym_support_view *s, const int64_t *y) {
    uint64_t lo = 0, hi = s->distinct;
    while (lo < hi) {
        uint64_t mid = lo + (hi - lo) / 2;
        int c = tr_sym_tuple_cmp(s->tuples + mid * (uint64_t)s->n, y, s->n);
        if (c == 0) return mid;
        if (c < 0) lo = mid + 1; else hi = mid;
    }
    return TR_SYM_NONE;
}
/* the image y = g(x) under generator gen: 1, or 0 when a coordinate of the image leaves int64 (no tuple of any
 * support lies there) */
TR_NUM_HD int tr_sym_image(const tr_sym_group_view *g, int32_t gen, const int64_t *x, int64_t *y) {
    const tr_sym_gen *G = &g->gens[gen];
    const int32_t n = g->n;
    for (int32_t i = 0; i < n; i++) y[i] = x[i];
    if (G->kind == TR_SYM_GEN_VALUES) {
        const int64_t *k = g->keys + G->at;
        uint64_t lo = 0, hi = G->nvalues;
        int64_t v = x[G->coord];
        while (lo < hi) {
            uint64_t mid = lo + (hi - lo) / 2;
            if (k[mid] == v) { y[G->coord] = g->targets[G->at + mid]; return 1; }
            if (k[mid] < v) lo = mid + 1; else hi = mid;
        }
        return 1;
    }
    const int64_t *A = g->coef + G->at, *b = A + (uint64_t)n * (uint64_t)n;
    for (int32_t i = 0; i < n; i++) {
        unsigned __int128 acc = (unsigned __int128)(__int128)b[i];
        for (int32_t j = 0; j < n; j++) {
            int64_t a = A[(uint64_t)i * (uint64_t)n + (uint64_t)j];
            if (!a) continue;
            __int128 p = (__int128)a * x[j];                        /* |p| <= 2^126 */
            __int128 before = (__int128)acc;
            acc += (unsigned __int128)p;
            __int128 after = (__int128)acc;
            if ((p < 0) == (before < 0) && (after < 0) != (before < 0)) return 0;   /* past int128 */
        }
        __int128 v = (__int128)acc;
        if (v < INT64_MIN || v > INT64_MAX) return 0;
        y[i] = (int64_t)v;
    }
    return 1;
}
/* support tuple t under generator gen: TR_SYM_HOLDS, _MISSING (image absent) or _BROKEN (image present with another
 * multiplicity); *image_mult = the image's multiplicity (0 when absent). y: scratch for n coordinates. */
TR_NUM_HD int tr_sym_check_tuple(const tr_sym_support_view *s, const tr_sym_group_view *g, int32_t gen, uint64_t t,
                                 int64_t *y, uint64_t *image_mult) {
    *image_mult = 0;
    if (!tr_sym_image(g, gen, s->tuples + t * (uint64_t)s->n, y)) return TR_SYM_MISSING;
    uint64_t u = tr_sym_find(s, y);
    if (u == TR_SYM_NONE) return TR_SYM_MISSING;
    *image_mult = s->mult[u];
    return s->mult[u] == s->mult[t] ? TR_SYM_HOLDS : TR_SYM_BROKEN;
}
/* Completes the raw evidence of generator j (rows, tuples, violations, missing, broken, witness: what the kernel
 * accumulates) with the witness's tuple, kind and image multiplicity and holds; TR_MATH_INVALID when the raw evidence
 * is inconsistent with the support (counts, or a witness that is not the first row of a violating tuple). Host code:
 * the support view's arrays are host memory. */
static inline int32_t tr_sym_evidence_finish(const tr_sym_support_view *s, const tr_sym_group_view *g, int32_t j,
                                             tr_sym_evidence *e) {
    uint64_t total = 0;
    for (uint64_t t = 0; t < s->distinct; t++) total += s->mult[t];
    e->generator = j; e->reserved = 0;
    e->witness_tuple = TR_SYM_NONE; e->image_mult = 0; e->kind = TR_SYM_HOLDS;
    if (e->rows != total || e->tuples != s->distinct || e->missing + e->broken > e->tuples ||
        e->violations > e->rows || ((e->missing + e->broken) == 0) != (e->violations == 0) ||
        (e->violations == 0) != (e->witness == TR_SYM_NONE)) return -1;
    e->holds = e->violations == 0;
    if (e->holds) return 0;
    int64_t y[TR_SYM_MAX_COORDS];
    for (uint64_t t = 0; t < s->distinct; t++) {
        if (s->first[t] != e->witness) continue;
        uint64_t im = 0;
        int k = tr_sym_check_tuple(s, g, j, t, y, &im);
        if (k == TR_SYM_HOLDS) return -1;
        e->witness_tuple = t; e->kind = k; e->image_mult = im;
        return 0;
    }
    return -1;
}
/* ------------------------------------------------------------------ view checks and device needs (host code) */
/* TR_MATH_OK when a query view is well formed (sizes, pointers, every action's carrier and denominator) */
static inline int32_t tr_sym_query_view_check(const tr_sym_query_view *v) {
    if (!v || v->ngroups < 1 || v->nb < 1 || v->nb > TR_SYM_MAX_BASIS || !v->moments || v->nactions < 0 ||
        (v->nactions && !v->actions) || v->nfunc < 0 || (v->nfunc && !v->functionals) || v->constant < -1 ||
        v->constant >= v->nb) return -1;
    for (int32_t i = 0; i < v->nactions; i++) {
        const tr_sym_action_view *a = &v->actions[i];
        if (a->nb != v->nb || (a->carrier != TR_SYM_EXACT && a->carrier != TR_SYM_INTERVAL) ||
            (a->carrier == TR_SYM_EXACT && (!a->num || tr_i128_get(a->den) <= 0)) ||
            (a->carrier == TR_SYM_INTERVAL && !a->iv) || (a->permutation && !a->index)) return -1;
        if (a->permutation)
            for (int32_t j = 0; j < a->nb; j++) if (a->index[j] < 0 || a->index[j] >= a->nb) return -1;
    }
    return 0;
}
/* the needs of a batch of n requests: output = n; bytes = the results (CPU) or every device copy (CUDA: moments,
 * each action's matrix and index, the action views, the functionals, the requests and the results) */
static inline void tr_sym_query_needs(const tr_sym_query_view *v, uint64_t n, int32_t target, tr_math_need *need) {
    for (int i = 0; i < TR_BUDGET_RESOURCES; i++) need->need[i] = 0;
    need->need[TR_BUDGET_OUTPUT] = n;
    uint64_t bytes = tr_budget_mul(n, sizeof(tr_sym_result));
    if (target == TR_SYM_NEED_DEVICE) {
        const uint64_t nb = (uint64_t)v->nb;
        bytes = tr_budget_add(bytes, tr_budget_mul(n, sizeof(tr_sym_request)));
        bytes = tr_budget_add(bytes, tr_budget_mul(tr_budget_mul(v->ngroups, nb), 16));
        bytes = tr_budget_add(bytes, tr_budget_mul((uint64_t)v->nactions, sizeof(tr_sym_action_view)));
        for (int32_t i = 0; i < v->nactions; i++) {
            bytes = tr_budget_add(bytes, tr_budget_mul(tr_budget_mul(nb, nb), 16));
            if (v->actions[i].permutation) bytes = tr_budget_add(bytes, tr_budget_mul(nb, 4));
        }
        bytes = tr_budget_add(bytes, tr_budget_mul(tr_budget_mul((uint64_t)v->nfunc, nb), 8));
    }
    need->need[TR_BUDGET_BYTES] = bytes;
}
/* TR_MATH_OK when a support view and a group view (with the lengths of its coefficient and value arrays) are well
 * formed: every generator's arrays inside them, every VALUES coordinate inside the tuples */
static inline int32_t tr_sym_views_check(const tr_sym_support_view *s, const tr_sym_group_view *g, uint64_t ncoef,
                                         uint64_t nvalues) {
    if (!s || !g || s->n < 1 || s->n > TR_SYM_MAX_COORDS || g->n != s->n || g->ngen < 1 ||
        g->ngen > TR_SYM_MAX_GENERATORS || !g->gens || (s->distinct && (!s->tuples || !s->mult || !s->first)) ||
        (ncoef && !g->coef) || (nvalues && (!g->keys || !g->targets))) return -1;
    const uint64_t block = (uint64_t)s->n * (uint64_t)s->n + (uint64_t)s->n;
    for (int32_t j = 0; j < g->ngen; j++) {
        const tr_sym_gen *G = &g->gens[j];
        if (G->kind == TR_SYM_GEN_AFFINE) { if (G->at > ncoef || ncoef - G->at < block) return -1; }
        else if (G->kind == TR_SYM_GEN_VALUES) {
            if (G->coord < 0 || G->coord >= g->n || G->at > nvalues || nvalues - G->at < G->nvalues) return -1;
            for (uint64_t i = 1; i < G->nvalues; i++) if (g->keys[G->at + i - 1] >= g->keys[G->at + i]) return -1;
        } else return -1;
    }
    return 0;
}
/* the device copies of tr_mgpu_sym_verify */
static inline uint64_t tr_sym_verify_device_bytes(const tr_sym_support_view *s, const tr_sym_group_view *g,
                                                  uint64_t ncoef, uint64_t nvalues) {
    uint64_t b = tr_budget_mul(s->distinct, tr_budget_add(tr_budget_mul((uint64_t)s->n, 8), 16));
    b = tr_budget_add(b, tr_budget_mul((uint64_t)g->ngen, sizeof(tr_sym_gen) + 6 * 8));
    b = tr_budget_add(b, tr_budget_mul(ncoef, 8));
    return tr_budget_add(b, tr_budget_mul(nvalues, 16));
}
/* the needs of a verification: variables = coordinates, output = generators, bytes = the evidence (CPU) or the device
 * copies (CUDA) */
static inline void tr_sym_verify_needs(const tr_sym_support_view *s, const tr_sym_group_view *g, uint64_t ncoef,
                                       uint64_t nvalues, int32_t target, tr_math_need *need) {
    for (int i = 0; i < TR_BUDGET_RESOURCES; i++) need->need[i] = 0;
    need->need[TR_BUDGET_VARIABLES] = (uint64_t)s->n;
    need->need[TR_BUDGET_OUTPUT] = (uint64_t)g->ngen;
    need->need[TR_BUDGET_BYTES] = target == TR_SYM_NEED_DEVICE ? tr_sym_verify_device_bytes(s, g, ncoef, nvalues)
                                                               : tr_budget_mul((uint64_t)g->ngen, sizeof(tr_sym_evidence));
}

/* ------------------------------------------------------------------ geometry (libalg) */
/* Declares n coordinates over a scope: cols (distinct scope columns), kinds (TR_SYM_NUMERIC / _CATEGORICAL), units
 * (labels >= 0); the scales and the scope's digest are read from the scope. TR_MATH_INVALID for a malformed
 * declaration, TR_MATH_UNSUPPORTED past TR_SYM_MAX_COORDS. */
int32_t tr_sym_geometry_make(const tr_scope *, const int32_t *cols, const int32_t *kinds, const int32_t *units,
                             int32_t n, tr_sym_geometry *out);
/* a geometry without data (columns 0..n-1, declared scales 0..18, digest 0): for actions on bases alone */
int32_t tr_sym_geometry_abstract(int32_t n, const int32_t *kinds, const int32_t *units, const int32_t *scales,
                                 tr_sym_geometry *out);
int32_t tr_sym_geometry_check(const tr_sym_geometry *);

/* ------------------------------------------------------------------ bases (libalg) */
tr_sym_basis *tr_sym_basis_create(int32_t n, const uint8_t *exponents, int32_t nb, const tr_math_budget *,
                                  tr_math_refusal *, int32_t *error);
/* every monomial of total degree <= degree (homogeneous: exactly degree) over the coordinates of mask (bit i:
 * coordinate i), graded, each degree in FND-004's order (math_bank.py monomials) */
tr_sym_basis *tr_sym_basis_full(int32_t n, uint32_t mask, int32_t degree, int32_t homogeneous,
                                const tr_math_budget *, tr_math_refusal *, int32_t *error);
int32_t tr_sym_basis_need(int32_t n, int32_t nb, int32_t degree, tr_math_need *);
int32_t tr_sym_basis_get_info(const tr_sym_basis *, tr_sym_basis_info *);
int32_t tr_sym_basis_get_exponents(const tr_sym_basis *, uint8_t *out, uint64_t capacity);   /* nb x n */
int32_t tr_sym_basis_find(const tr_sym_basis *, const uint8_t *exponents);   /* the index, or -1 */
/* TR_MATH_OK when every coordinate with a positive exponent is a declared NUMERIC coordinate of the geometry (the
 * geometry's n coordinates), else TR_MATH_DOMAIN_ERROR (a basis over a categorical code has no metric meaning) */
int32_t tr_sym_basis_check(const tr_sym_basis *, const tr_sym_geometry *);
int32_t tr_sym_basis_equal(const tr_sym_basis *, const tr_sym_basis *);       /* 1: the same monomials in order */
tr_sym_basis *tr_sym_basis_clone(const tr_sym_basis *, int32_t *error);
void tr_sym_basis_free(tr_sym_basis *);

/* ------------------------------------------------------------------ maps (libalg) */
tr_sym_map *tr_sym_map_make(const tr_sym_geometry *, const tr_sym_transform *, const tr_math_budget *,
                            tr_math_refusal *, int32_t *error);
tr_sym_map *tr_sym_map_identity(int32_t n, int32_t *error);
/* after o before (before applied first); VALUES maps do not compose with others (TR_MATH_UNSUPPORTED) */
tr_sym_map *tr_sym_map_compose(const tr_sym_map *after, const tr_sym_map *before, const tr_math_budget *,
                               tr_math_refusal *, int32_t *error);
int32_t tr_sym_map_get_info(const tr_sym_map *, tr_sym_map_info *);
/* EXACT affine: a (n x n), b (n), q (n): x'_i = (sum_j a_ij x_j + b_i) / q_i; INTERVAL: the enclosures */
int32_t tr_sym_map_get_exact(const tr_sym_map *, tr_math_i128 *a, tr_math_i128 *b, tr_math_i128 *q);
int32_t tr_sym_map_get_interval(const tr_sym_map *, tr_math_interval *a, tr_math_interval *b);
int32_t tr_sym_map_get_values(const tr_sym_map *, int64_t *keys, int64_t *targets, uint64_t capacity);
void tr_sym_map_free(tr_sym_map *);

/* ------------------------------------------------------------------ actions (libalg) */
int32_t tr_sym_action_need(const tr_sym_basis *, const tr_sym_map *, tr_math_need *);
/* M_T on the basis; TR_MATH_DOMAIN_ERROR when the basis is not closed under the map (closure: the witness; may be
 * NULL), TR_MATH_OVERFLOW past int128, TR_MATH_UNSUPPORTED for a VALUES map */
tr_sym_action *tr_sym_action_make(const tr_sym_basis *, const tr_sym_map *, const tr_math_budget *, tr_math_refusal *,
                                  tr_sym_closure *closure, int32_t *error);
tr_sym_action *tr_sym_action_identity(const tr_sym_basis *, const tr_math_budget *, tr_math_refusal *, int32_t *error);
/* M_after M_before (the action of after o before); both on one basis (else TR_MATH_DOMAIN_ERROR) */
tr_sym_action *tr_sym_action_compose(const tr_sym_action *after, const tr_sym_action *before, const tr_math_budget *,
                                     tr_math_refusal *, int32_t *error);
int32_t tr_sym_action_equal(const tr_sym_action *, const tr_sym_action *);   /* 1 / 0 (TR_MATH_INVALID) */
int32_t tr_sym_action_get_info(const tr_sym_action *, tr_sym_action_info *);
int32_t tr_sym_action_get_view(const tr_sym_action *, tr_sym_action_view *); /* pointers into the action */
const tr_sym_basis *tr_sym_action_get_basis(const tr_sym_action *);          /* borrowed */
void tr_sym_action_free(tr_sym_action *);

/* ------------------------------------------------------------------ states and requests (libalg) */
tr_sym_state *tr_sym_state_from_bank(const tr_bank *, const tr_sym_geometry *, const tr_math_budget *,
                                     tr_math_refusal *, int32_t *error);
tr_sym_state *tr_sym_state_from_tensor(const tr_tensor *, const tr_sym_geometry *, const tr_math_budget *,
                                       tr_math_refusal *, int32_t *error);
int32_t tr_sym_state_get_info(const tr_sym_state *, tr_sym_state_info *);
const tr_sym_basis *tr_sym_state_get_basis(const tr_sym_state *);            /* borrowed */
/* the view of the moments (actions and functionals NULL); pointers into the state */
int32_t tr_sym_state_get_view(const tr_sym_state *, tr_sym_query_view *);
int32_t tr_sym_state_get_rows(const tr_sym_state *, uint64_t *rows, uint64_t *used, uint64_t capacity);
void tr_sym_state_free(tr_sym_state *);
/* the needs of tr_sym_query (TR_SYM_NEED_CPU) or tr_mgpu_sym_query (TR_SYM_NEED_DEVICE: its device copies) */
int32_t tr_sym_query_need(const tr_sym_query_view *, uint64_t nrequests, int32_t target, tr_math_need *);
/* the requests over a view (the CPU reference of the kernel) */
int32_t tr_sym_query_view_eval(const tr_sym_query_view *, const tr_sym_request *, uint64_t n, tr_sym_result *out,
                               const tr_math_budget *, tr_math_refusal *);
/* the view of a state with actions (every action on the state's basis, else TR_MATH_DOMAIN_ERROR) and functionals:
 * views[nactions] is filled; the view points into the state, the actions and functionals */
int32_t tr_sym_query_bind(const tr_sym_state *, const tr_sym_action *const *actions, int32_t nactions,
                          const int64_t *functionals, int32_t nfunc, tr_sym_action_view *views, tr_sym_query_view *out);
int32_t tr_sym_query(const tr_sym_state *, const tr_sym_action *const *actions, int32_t nactions,
                     const int64_t *functionals, int32_t nfunc, const tr_sym_request *, uint64_t n, tr_sym_result *out,
                     const tr_math_budget *, tr_math_refusal *);

/* ------------------------------------------------------------------ SYM-002 (libalg) */
tr_sym_group *tr_sym_group_create(const tr_sym_map *const *generators, int32_t ngen, const tr_math_budget *,
                                  tr_math_refusal *, int32_t *error);
int32_t tr_sym_group_get_info(const tr_sym_group *, tr_sym_group_info *);
int32_t tr_sym_group_get_view(const tr_sym_group *, tr_sym_group_view *);
void tr_sym_group_free(tr_sym_group *);
/* the support of the scope's selected rows over the geometry; FORBID with a NULL coordinate: TR_MATH_DOMAIN_ERROR and
 * *null_row the smallest such row (may be NULL) */
tr_sym_support *tr_sym_support_build(const tr_scope *, const tr_sym_geometry *, int32_t policy, const tr_math_budget *,
                                     tr_math_refusal *, uint64_t *null_row, int32_t *error);
int32_t tr_sym_support_need(const tr_scope *, const tr_sym_geometry *, tr_math_need *);
int32_t tr_sym_support_get_info(const tr_sym_support *, tr_sym_support_info *);
int32_t tr_sym_support_get_view(const tr_sym_support *, tr_sym_support_view *);
const tr_cert_record *tr_sym_support_get_record(const tr_sym_support *);   /* borrowed */
int32_t tr_sym_support_fresh(const tr_sym_support *, const tr_scope *current, tr_cert_staleness *);
void tr_sym_support_free(tr_sym_support *);
/* the evidence of every generator (out[ngen]) over views (ncoef, nvalues: the lengths of the group view's arrays,
 * tr_sym_group_get_info); the CPU reference of the kernel */
int32_t tr_sym_verify_need(const tr_sym_support_view *, const tr_sym_group_view *, uint64_t ncoef, uint64_t nvalues,
                           int32_t target, tr_math_need *);
int32_t tr_sym_verify_view(const tr_sym_support_view *, const tr_sym_group_view *, uint64_t ncoef, uint64_t nvalues,
                           const tr_math_budget *, tr_sym_evidence *out, tr_math_refusal *);
/* a certificate when every generator holds; else TR_MATH_DOMAIN_ERROR (evidence: every generator's, may be NULL) */
tr_sym_cert *tr_sym_certify(const tr_sym_support *, const tr_sym_group *, const tr_math_budget *, tr_math_refusal *,
                            tr_sym_evidence *evidence, int32_t *error);
/* a certificate from evidence computed elsewhere (tr_mgpu_sym_verify) after consistency checks; backend TR_CERT_BY_*;
 * finished: the checked evidence (may be NULL) */
tr_sym_cert *tr_sym_adopt(const tr_sym_support *, const tr_sym_group *, const tr_sym_evidence *raw, int32_t backend,
                          const tr_math_budget *, tr_math_refusal *, tr_sym_evidence *finished, int32_t *error);
int32_t tr_sym_cert_get_info(const tr_sym_cert *, tr_sym_cert_info *);
int32_t tr_sym_cert_get_evidence(const tr_sym_cert *, tr_sym_evidence *out, int32_t capacity);
const tr_cert_record *tr_sym_cert_get_record(const tr_sym_cert *);         /* borrowed */
int32_t tr_sym_cert_fresh(const tr_sym_cert *, const tr_scope *current, tr_cert_staleness *);
void tr_sym_cert_free(tr_sym_cert *);
/* the orbits of the certified group on the support; TR_MATH_STALE (stale: the reasons) when the certificate is not
 * fresh for current, TR_MATH_DOMAIN_ERROR when it was not made for this support */
tr_sym_orbits *tr_sym_orbits_build(const tr_sym_support *, const tr_sym_cert *, const tr_scope *current,
                                   const tr_math_budget *, tr_math_refusal *, tr_cert_staleness *stale, int32_t *error);
int32_t tr_sym_orbits_get_info(const tr_sym_orbits *, tr_sym_orbits_info *);
int32_t tr_sym_orbits_get_view(const tr_sym_orbits *, tr_sym_orbits_view *);
void tr_sym_orbits_free(tr_sym_orbits *);
/* 1 when the query's polynomial is proven invariant under every generator, 0 when generator *generator breaks it,
 * else a native error (an invariance expansion past int128: TR_MATH_OVERFLOW) */
int32_t tr_sym_poly_invariant(const tr_sym_group *, const tr_sym_poly_query *, int32_t *generator,
                              const tr_math_budget *, tr_math_refusal *);
/* invariant queries on the orbit representatives (the certificate re-checked fresh for current: TR_MATH_STALE);
 * per answer status 0, TR_MATH_DOMAIN_ERROR (not invariant: refused) or TR_MATH_OVERFLOW */
int32_t tr_sym_orbit_query(const tr_sym_orbits *, const tr_scope *current, const tr_sym_poly_query *, int32_t nq,
                           tr_sym_poly_answer *out, const tr_math_budget *, tr_math_refusal *, tr_cert_staleness *stale);
/* the exhaustive evaluation of any query over every selected row of the scope (complete rows under SKIP) */
int32_t tr_sym_rows_query(const tr_scope *, const tr_sym_geometry *, int32_t policy, const tr_sym_poly_query *,
                          int32_t nq, tr_sym_poly_answer *out, const tr_math_budget *, tr_math_refusal *);

/* ------------------------------------------------------------------ the CUDA library (math_sym_gpu.cu)
 * tr_mgpu_sym_query: the requests on a view of host memory (tr_sym_query_bind), copied to the device; one work item
 * per request runs tr_sym_eval_request; out[n] equals tr_sym_query_view_eval's. Budget: output = n, bytes = the
 * device copies (tr_sym_query_need, TR_SYM_NEED_DEVICE).
 * tr_mgpu_sym_verify: every generator's evidence (out[ngen], finished with tr_sym_evidence_finish) over a support view
 * and a group view of host memory; work items (generator, tuple slice) count with one atomic per counter and take the
 * smallest violating row with an atomic minimum; equal to tr_sym_verify_view (adopt it with tr_sym_adopt). Budget:
 * output = generators, bytes = the device copies. Both return 0 or a native code (TR_MATH_DEVICE when the device
 * fails). */
int tr_mgpu_sym_query(const tr_sym_query_view *view, const tr_sym_request *requests, uint64_t n, tr_sym_result *out,
                      const tr_math_budget *budget, tr_math_refusal *refusal);
int tr_mgpu_sym_verify(const tr_sym_support_view *support, const tr_sym_group_view *group, uint64_t ncoef,
                       uint64_t nvalues, const tr_math_budget *budget, tr_sym_evidence *out, tr_math_refusal *refusal);
#ifdef __cplusplus
}
#endif
#endif
