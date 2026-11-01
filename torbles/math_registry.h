/* FND-005: the native module and operator registry of the mathematical capabilities.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Carriers and laws. A carrier is the algebraic structure a value lives in (checked int64 integers, exact 128-bit
 * sums, residues modulo 2^128, canonical rationals, multiplicities, min/max lattices, stored codes, row sets, big
 * integers and rationals, GF(p), complex and exact-endpoint intervals, truncated series, Kleene's three-valued logic).
 * tr_reg_carrier_laws states the laws each carrier's operations satisfy as implemented, by every implementation in
 * that carrier (a result that is returned is exact; an overflow is refused, never wrapped). tr_reg_conversions lists
 * the declared carrier changes: EXACT (always lossless), CHECKED (lossless or refused: a narrowing, a nonnegative
 * check, or a residue read back under a proven bound) and ENCLOSING (a rigorous enclosure, not lossless) apply at any
 * step; LIFT (a residue to its integer representative: lossless per value but not a homomorphism, XOR is not integer
 * addition), REDUCTION (an exact value to its residue: a homomorphism, not injective) and ENCODING (three-valued
 * outcomes to two-valued ones or to digits: UNKNOWN has no Boolean value, so the NULL encoding must be declared)
 * apply only at a step of their own explicit operator (lift, residue, null_encode). Every operator
 * (tr_reg_operators) states the laws it requires of the carrier it runs in, and some run only in a fixed input
 * carrier or produce a fixed output carrier.
 *
 * A composition is a chain of (operator, carrier) steps over a value that starts in an input carrier.
 * tr_reg_check_laws accepts it only when, at every step: the operator exists; its fixed input carrier, if it has one,
 * is the step's carrier; the value's carrier equals the step's carrier or one declared conversion admitted at this
 * step leads there (EXACT or CHECKED; ENCLOSING too under the enclosure contract; LIFT, REDUCTION, ENCODING only when
 * the step's operator is the conversion's own); and the step's carrier has every law the operator requires (plus
 * EXACT under the exact contract, EXACT or ENCLOSURE under the enclosure contract). The value then takes the
 * operator's output carrier. Otherwise it refuses at the first failing step, naming the missing law (the lowest
 * missing bit) or the missing conversion (from, to), with TR_MATH_CARRIER_ERROR (-7), or TR_MATH_OPERATOR_ERROR (-8)
 * for an unknown operator.
 *
 * Modules. A module descriptor (tr_reg_module, math_registry.c) states what one native module computes in, the laws it
 * relies on and guarantees, the operators it implements in which carriers on which devices (tr_reg_impls below,
 * shared with the CUDA library), the state that suffices for its requests, the proofs it depends on, its CPU and CUDA
 * entry points and its cost coefficients. Ids 1 .. 9 are the representations of the registry's own request kinds
 * (module id == representation id: the two first-slice built-ins, FND-010, DISC-001, SCAN, HISTOGRAM, POWER,
 * BITPLANE, EMPTY), whose candidates the FND-008 planner costs; ids from TR_REG_M_SCOPE on are the implemented
 * families (representation 0: each answers its own requests through its own API, the planner has no candidates for
 * them and their cost coefficients are zeros, cost_basis TR_REG_COST_NONE). A family adds one id before TR_REG_M_END,
 * one descriptor in math_registry.c and its rows in tr_reg_impls.
 *
 * Requests. tr_reg_request is "aggregate f_j(x) over the selected valid rows" for one to four integer polynomials
 * f_j (degree <= 8, int64 coefficients, low to high) of the logical value x = code + offset of one column (a DECIMAL
 * column's logical value is its scaled integer), aggregate SUM (exact, 128 bits), MIN or MAX (int64). NULL validity
 * (rows whose validity flag is 0), bag multiplicity (the selected rows, each counted once per row) and group presence
 * (whether any row is selected) are reported separately: selected, valid and nulls. The executable representations:
 *   SCAN       every f_j at every selected valid row by Horner's rule in checked int64 (128-bit products), folded
 *              exactly in 128 bits (or min/max);
 *   HISTOGRAM  the selected valid rows counted per code (state: ncodes counts), contracted as
 *              sum_c count_c * f(c + offset) (or min/max over the occupied codes), the same checked evaluation;
 *   POWER      sparse power sums S_k = sum_r x_r^k modulo 2^128 for the exponents k of the polynomials' support
 *              (state), contracted as sum_k a_k S_k;
 *   BITPLANE   the factored state of the code's bit planes: for every set S of at most D planes (D >= degree), the
 *              number of selected valid rows whose code has every bit of S; f(c + offset) = sum_S w_S prod_{j in S}
 *              bit_j(c) with the Moebius weights w_S = sum_{T subset of S} (-1)^(|S|-|T|) f(T + offset) (T read as
 *              the integer of its bits), and the sum is sum_S w_S count_S, computed modulo 2^128;
 *   EMPTY      no scan: only for a selection proven empty by an exact count (it recounts the selection natively).
 * POWER and BITPLANE compute residues modulo 2^128 (a ring homomorphism of the integers) and read the result back as
 * a signed 128-bit integer only when |result| < 2^127 is proven before execution from the code domain and an upper
 * bound of the selected rows (tr_reg_bound_ok): |x| <= M = max(|offset|, |offset + ncodes - 1|) and
 * |f_j(x)| <= sum_k |a_jk| M^k. The residue is then the exact sum. SCAN and HISTOGRAM refuse (TR_MATH_OVERFLOW) when
 * the checked evaluation of a value leaves int64 at any Horner step; their sums never overflow (fewer than 2^64 rows
 * of int64 values). A code >= ncodes in a selected valid row is invalid input (TR_MATH_INVALID; checked first). The
 * logical domain offset + ncodes - 1 must fit int64 (else TR_MATH_OVERFLOW before anything runs). A plan
 * (tr_reg_candidates) admits SCAN and HISTOGRAM only when tr_reg_eval_bound_ok proves no evaluation can refuse.
 *
 * Each representation is a chain of steps (tr_reg_chains: family 0 for SUM, 1 for MIN/MAX, 2 the multiplicity chain
 * of the histogram), checked like any composition: POWER and BITPLANE refuse MIN/MAX (the min/max lattice is not
 * distributive; multiplicities do not convert to it) with the law or conversion named.
 *
 * Transforms through the registry (tr_reg_transform, tr_mgpu_registry_transform): a subset-lattice zeta or Moebius
 * transform (LATTICE-002) or a Walsh-Hadamard transform (SPEC-002) of 2^k values in a declared carrier: the chain
 * [(op, carrier)] from the values' carrier is checked like any composition, the module implementing op in that
 * carrier on the device is taken from tr_reg_impls, and the explanation names it, its entry point, the device and
 * any refusal (law, conversion, implementation, bound, budget, overflow).
 *
 * Budgets (math_budgets.h) are checked before anything is allocated (tr_reg_need): degree and support of the
 * polynomials, states (histogram cells, power sums, plane sets), circuit (BITPLANE: Moebius terms sum_S 2^|S|),
 * output (the polynomials) and bytes (state, scratch and, on CUDA, the device copies). Results, explanations, states,
 * candidates, descriptors and implementation rows are plain structs of fixed layout (sizes checked in
 * math_registry.c). */
#ifndef TORBLES_MATH_REGISTRY_H
#define TORBLES_MATH_REGISTRY_H
#include <stdint.h>
#include "math_num.h"
#include "math_budgets.h"
#include "math_disc.h"
#ifdef __cplusplus
extern "C" {
#endif

#define TR_REG_ABI 2                         /* 2: descriptors, explanations, conversions, implementation rows */
/* native codes (mathematics.h, math_domain.h, math_num.h): the CUDA library does not include the first slice */
#define TR_REG_E_INVALID (-1)
#define TR_REG_E_MEMORY (-2)
#define TR_REG_E_OVERFLOW (-3)
#define TR_REG_E_BUDGET (-5)
#define TR_REG_E_CARRIER (-7)
#define TR_REG_E_OPERATOR (-8)
#define TR_REG_E_UNSUPPORTED (-12)

/* ------------------------------------------------------------------ carriers, laws, conversions */
enum { TR_REG_C_ANY = 0,                                             /* operator input: any carrier */
       TR_REG_C_INTEGER = 1, TR_REG_C_RATIONAL = 2, TR_REG_C_BOOLEAN = 3, TR_REG_C_GF2 = 4,   /* = math_domain.h */
       TR_REG_C_COUNT = 5,     /* multiplicities: natural numbers (uint64, overflow refused); histogram and occupancy
                                  cell counts */
       TR_REG_C_INT128 = 6,    /* exact signed 128-bit sums */
       TR_REG_C_MOD128 = 7,    /* residues modulo 2^128 (exact ring, no order) */
       TR_REG_C_INTERVAL = 8,  /* outward-rounded double intervals (math_num.h stepped, FND-009 tight) */
       TR_REG_C_FLOAT64 = 9,   /* IEEE doubles */
       TR_REG_C_MINMAX = 10,   /* int64 with min (or max) as the fold and integer multiplication */
       TR_REG_C_CODE = 11,     /* stored codes of an order-preserving codec */
       TR_REG_C_ROWSET = 12,   /* row bitmaps: union, intersection */
       TR_REG_C_BIGINT = 13,   /* exact integers beyond 128 bits: FND-009 big integers (size budgeted) and the
                                  192- / 256-bit words of LIN-001, DIST-002, RESP-002, MULTI-001, SPEC-002 (checked) */
       TR_REG_C_BIGQ = 14,     /* exact rationals beyond int64: FND-009 big rationals (size budgeted), the families'
                                  wide canonical rationals (checked) */
       TR_REG_C_GFP = 15,      /* GF(p), p prime < 2^62 (FND-009): a field of characteristic p, no order */
       TR_REG_C_CINTERVAL = 16,/* complex rectangles re x im of FND-009 tight intervals: enclosures */
       TR_REG_C_SERIES = 17,   /* truncated power series over Q (exact up to the order): FND-009 univariate series,
                                  JET-001/002 multivariate jets (fixed-denominator int64 numerators, checked) */
       TR_REG_C_KLEENE = 18,   /* SQL three-valued logic: FALSE < UNKNOWN < TRUE, Kleene's AND (min), OR (max), NOT */
       TR_REG_C_QINTERVAL = 19,/* intervals with exact rational endpoints (exact endpoint sums): enclosures */
       TR_REG_CARRIERS = 20 };
enum { TR_LAW_ASSOCIATIVE = 1 << 0,      /* (a + b) + c = a + (b + c) */
       TR_LAW_COMMUTATIVE = 1 << 1,      /* a + b = b + a */
       TR_LAW_IDENTITY = 1 << 2,         /* 0 + a = a: the empty bag has a value */
       TR_LAW_INVERSE = 1 << 3,          /* a + (-a) = 0: contributions can be removed */
       TR_LAW_DISTRIBUTIVE = 1 << 4,     /* a * (b + c) = a*b + a*c, * associative with a unit (a semiring) */
       TR_LAW_IDEMPOTENT = 1 << 5,       /* a + a = a: multiplicity can be ignored */
       TR_LAW_MUL_IDEMPOTENT = 1 << 6,   /* a * a = a: products of indicators over multisets are over sets */
       TR_LAW_ORDERED = 1 << 7,          /* its comparison is a total order with a <= b => a + c <= b + c (codes:
                                            the codec preserves the order of values) */
       TR_LAW_DIVISION = 1 << 8,         /* every element but 0 has a * inverse (with INVERSE: a field) */
       TR_LAW_CHAR0 = 1 << 9,            /* the multiples n*1 (n >= 0) are pairwise distinct */
       TR_LAW_EXACT = 1 << 10,           /* results are exact; overflow is refused, never wrapped or rounded */
       TR_LAW_ENCLOSURE = 1 << 11,       /* results are rigorous enclosures of the exact real value */
       TR_LAW_COMPLEMENT = 1 << 12,      /* every a has a complement b: a + b = 1 and a * b = 0 (excluded middle:
                                            Boolean, not Kleene; two-valued cells partition the points) */
       TR_LAW_CHAR2 = 1 << 13 };         /* a + a = 0: characteristic 2 (XOR, not integer addition) */
#define TR_REG_LAWS 14
#define TR_REG_RING (TR_LAW_ASSOCIATIVE | TR_LAW_COMMUTATIVE | TR_LAW_IDENTITY | TR_LAW_DISTRIBUTIVE)
#define TR_REG_FOLD (TR_LAW_ASSOCIATIVE | TR_LAW_COMMUTATIVE | TR_LAW_IDENTITY)
#define TR_REG_FIELD (TR_REG_RING | TR_LAW_INVERSE | TR_LAW_DIVISION)
#define TR_REG_LATTICE (TR_REG_RING | TR_LAW_IDEMPOTENT | TR_LAW_MUL_IDEMPOTENT)   /* a distributive lattice */
static const uint32_t tr_reg_carrier_laws[TR_REG_CARRIERS] = {
    0u,
    TR_REG_RING | TR_LAW_INVERSE | TR_LAW_ORDERED | TR_LAW_CHAR0 | TR_LAW_EXACT,                          /* integer */
    TR_REG_FIELD | TR_LAW_ORDERED | TR_LAW_CHAR0 | TR_LAW_EXACT,                                          /* rational */
    TR_REG_LATTICE | TR_LAW_ORDERED | TR_LAW_DIVISION | TR_LAW_COMPLEMENT | TR_LAW_EXACT,                 /* boolean */
    TR_REG_FIELD | TR_LAW_MUL_IDEMPOTENT | TR_LAW_COMPLEMENT | TR_LAW_CHAR2 | TR_LAW_EXACT,               /* gf2 */
    TR_REG_RING | TR_LAW_ORDERED | TR_LAW_CHAR0 | TR_LAW_EXACT,                                           /* count */
    TR_REG_RING | TR_LAW_INVERSE | TR_LAW_ORDERED | TR_LAW_CHAR0 | TR_LAW_EXACT,                          /* int128 */
    TR_REG_RING | TR_LAW_INVERSE | TR_LAW_EXACT,                                                          /* mod128 */
    TR_LAW_COMMUTATIVE | TR_LAW_CHAR0 | TR_LAW_ENCLOSURE,                                                 /* interval */
    TR_LAW_COMMUTATIVE | TR_LAW_IDENTITY,                                                                 /* float64 */
    TR_LAW_ASSOCIATIVE | TR_LAW_COMMUTATIVE | TR_LAW_IDENTITY | TR_LAW_IDEMPOTENT | TR_LAW_ORDERED | TR_LAW_EXACT,  /* minmax */
    TR_LAW_ORDERED | TR_LAW_EXACT,                                                                        /* code */
    TR_REG_LATTICE | TR_LAW_COMPLEMENT | TR_LAW_EXACT,                                                    /* rowset */
    TR_REG_RING | TR_LAW_INVERSE | TR_LAW_ORDERED | TR_LAW_CHAR0 | TR_LAW_EXACT,                          /* bigint */
    TR_REG_FIELD | TR_LAW_ORDERED | TR_LAW_CHAR0 | TR_LAW_EXACT,                                          /* bigq */
    TR_REG_FIELD | TR_LAW_EXACT,                                                                          /* gfp */
    TR_LAW_COMMUTATIVE | TR_LAW_IDENTITY | TR_LAW_CHAR0 | TR_LAW_ENCLOSURE,                               /* cinterval */
    TR_REG_RING | TR_LAW_INVERSE | TR_LAW_CHAR0 | TR_LAW_EXACT,                                           /* series */
    TR_REG_LATTICE | TR_LAW_ORDERED | TR_LAW_EXACT,                                                       /* kleene */
    TR_REG_FOLD | TR_LAW_CHAR0 | TR_LAW_ENCLOSURE,                                                        /* qinterval */
};
enum { TR_CONV_EXACT = 1, TR_CONV_CHECKED = 2, TR_CONV_ENCLOSING = 3,     /* applied at any step */
       TR_CONV_LIFT = 4,         /* a residue to its integer representative (only by a lift step) */
       TR_CONV_REDUCTION = 5,    /* an exact value to its residue (only by a residue step) */
       TR_CONV_ENCODING = 6,     /* three-valued outcomes by a declared NULL encoding (only by a null_encode step) */
       TR_REG_CONV_KINDS = 7 };
/* op: 0 for a conversion applied at any step, else the only operator that applies it. 16 bytes. */
typedef struct { int32_t from, to, kind, op; } tr_reg_conversion;

/* ------------------------------------------------------------------ operators */
enum { TR_OP_POLY_ADD = 1, TR_OP_POLY_SUB = 2, TR_OP_POLY_MUL = 3, TR_OP_POLY_POW = 4, TR_OP_POLY_DERIVATIVE = 5,
       TR_OP_POLY_INTEGRATE = 6, TR_OP_POLY_SUBSTITUTE = 7, TR_OP_POLY_EVALUATE = 8, TR_OP_BANK_REDUCE = 9,  /* = math_modules.h */
       TR_OP_CONVERT = 10,        /* change carrier only (the conversion is checked) */
       TR_OP_EVAL = 11,           /* a polynomial's value at each data point (checked) */
       TR_OP_FOLD = 12,           /* fold a bag of values in any order or chunking */
       TR_OP_MERGE = 13,          /* merge partial states of threads, blocks or devices */
       TR_OP_SCALE_COUNT = 14,    /* a value times its multiplicity */
       TR_OP_DISTINCT_FOLD = 15,  /* fold each distinct value once, multiplicity ignored */
       TR_OP_SUBTRACT = 16,       /* remove a sub-bag's contribution */
       TR_OP_DIVIDE = 17,         /* exact quotients */
       TR_OP_ORDER = 18,          /* comparisons, ranks, prefixes */
       TR_OP_POWERS = 19,         /* x^k by products */
       TR_OP_CONTRACT = 20,       /* sum_i w_i s_i of a state with weights */
       TR_OP_HISTOGRAM = 21,      /* rows per exact code */
       TR_OP_BITS = 22,           /* a code as its bit planes (row sets) */
       TR_OP_AND = 23,            /* intersections of planes */
       TR_OP_POPCOUNT = 24,       /* the rows of a row set */
       TR_OP_MARK_CELLS = 25,     /* one bit per occupied cell (FND-010) */
       TR_OP_VERIFY = 26,         /* exact equality checks of a relation (DISC-001) */
       TR_OP_EMPTY = 27,          /* the value of the empty bag (needs a proof of emptiness) */
       TR_OP_LIFT = 28,           /* explicit: a residue to its integer representative (LIFT conversions) */
       TR_OP_RESIDUE = 29,        /* explicit: an exact value to its residue modulo p or 2 (REDUCTION conversions) */
       TR_OP_NULL_ENCODE = 30,    /* explicit: three-valued outcomes by a declared NULL encoding (ENCODING) */
       TR_OP_MULTIPLY = 31,       /* the carrier's product (matrix products, convolutions) */
       TR_OP_EUCLID = 32,         /* division with remainder, exact division, gcd (integers) */
       TR_OP_SOLVE = 33,          /* exact solution sets of linear systems (elimination over a field) */
       TR_OP_INERTIA = 34,        /* the signs of a symmetric matrix's eigenvalues (an ordered field) */
       TR_OP_MINIMIZE = 35,       /* exact minimizers with their certificates (an ordered field) */
       TR_OP_ROOTS = 36,          /* real roots isolated by Sturm sequences: exact or exact-endpoint intervals */
       TR_OP_CRT = 37,            /* residues modulo several primes to the integer in a proven range (Garner) */
       TR_OP_RATRECON = 38,       /* a residue to the rational it reconstructs within a proven bound */
       TR_OP_ENCLOSE = 39,        /* a real computation evaluated as a rigorous enclosure */
       TR_OP_COEFFICIENTS = 40,   /* the exact coefficients of a truncated series (jet readouts) */
       TR_OP_DIFFERENTIATE = 41,  /* a parameter derivative of a separated family: its coefficients mapped exactly */
       TR_OP_PREDICATE = 42,      /* a row condition on exact values: a three-valued outcome */
       TR_OP_LOGIC = 43,          /* AND / OR / NOT (and SQL's XOR) of outcomes, word-parallel */
       TR_OP_COFACTOR = 44,       /* conditions of a predicate replaced by constants */
       TR_OP_DIFFERENCE = 45,     /* finite differences: sum_t (-1)^(m - |t|) f|S:=t (XOR derivative in GF(2)) */
       TR_OP_QUANTIFY = 46,       /* quantifier answers from TRUE / FALSE / UNKNOWN counts */
       TR_OP_TEST = 47,           /* a two-valued test of exact counts (zero, equality) */
       TR_OP_OCCUPANCY = 48,      /* points per cell of k two-valued axes */
       TR_OP_ZETA = 49,           /* superset or subset sums over the subset lattice */
       TR_OP_MOBIUS = 50,         /* their inversion: alternating sums */
       TR_OP_WALSH = 51,          /* the Walsh-Hadamard transform: sums with characters +-1 */
       TR_OP_ROLLUP = 52,         /* fine states added into their coarse group along a refinement map */
       TR_OP_REFINE = 53,         /* the coarse group of each fine group: a refinement map, certified on the rows */
       TR_OP_GROUP = 54,          /* dense group ids of key codes */
       TR_OP_UPDATE = 55,         /* signed insert / delete / weight-change batches merged into a state */
       TR_OP_PREFIX = 56,         /* prefix sums over a sorted support */
       TR_OP_ATTRIBUTE = 57,      /* attribution indices: rational-weighted sums of a game's marginal contributions or Moebius coefficients */
       TR_OP_COMPILE = 58,        /* a circuit compiled into a canonical decision diagram under a variable order */
       TR_OP_MODEL_COUNT = 59,    /* the weighted number of models of a function over a declared assignment universe */
       TR_OP_PROPAGATE = 60,      /* source changes propagated through a provenance DAG by the update laws of its semiring */
       TR_OP_EXPLAIN = 61,        /* explanations of a monotone lineage: sufficient, necessary and why-not source sets */
       TR_OP_PROBABILITY = 62,    /* the probability of a lineage under independent source tuples (a weighted model count) */
       TR_OP_NORMALIZE = 63,      /* normalizers, log-likelihoods, gradients and Hessians of a declared finite-support model from retained masses */
       TR_OP_FIXPOINT = 64,       /* the least fixed point of a monotone frontier iteration within a declared round bound */
       TR_OP_RELAX = 65,          /* path relaxations in an ordered idempotent semiring ((min, +), (max, +)) with improving-cycle refusal */
       TR_OP_BOUNDARY = 66,       /* signed incidences of a chain complex (d o d = 0) */
       TR_OP_HOMOLOGY = 67,       /* ranks, Betti numbers and cycle bases over a field */
       TR_OP_PERSISTENCE = 68,    /* pairs of a bounded filtration by column reduction */
       TR_REG_OPERATORS = 69 };
#define TR_REG_OPERATOR_WORDS 2    /* operator masks: bit op % 64 of word op / 64 */
typedef struct { int32_t id, input, output; uint32_t required; } tr_reg_operator;  /* input/output 0: the step's carrier */
static const tr_reg_operator tr_reg_operators[TR_REG_OPERATORS] = {
    {0, 0, 0, 0u},
    {TR_OP_POLY_ADD, 0, 0, TR_LAW_ASSOCIATIVE | TR_LAW_COMMUTATIVE | TR_LAW_IDENTITY},
    {TR_OP_POLY_SUB, 0, 0, TR_LAW_ASSOCIATIVE | TR_LAW_COMMUTATIVE | TR_LAW_IDENTITY | TR_LAW_INVERSE},
    {TR_OP_POLY_MUL, 0, 0, TR_REG_RING},
    {TR_OP_POLY_POW, 0, 0, TR_REG_RING},
    {TR_OP_POLY_DERIVATIVE, 0, 0, TR_REG_RING},
    {TR_OP_POLY_INTEGRATE, 0, 0, TR_REG_RING | TR_LAW_DIVISION | TR_LAW_CHAR0},
    {TR_OP_POLY_SUBSTITUTE, 0, 0, TR_REG_RING},
    {TR_OP_POLY_EVALUATE, 0, 0, TR_REG_RING},
    {TR_OP_BANK_REDUCE, 0, 0, TR_REG_RING},
    {TR_OP_CONVERT, 0, 0, 0u},
    {TR_OP_EVAL, 0, 0, TR_REG_RING},
    {TR_OP_FOLD, 0, 0, TR_LAW_ASSOCIATIVE | TR_LAW_COMMUTATIVE | TR_LAW_IDENTITY},
    {TR_OP_MERGE, 0, 0, TR_LAW_ASSOCIATIVE | TR_LAW_COMMUTATIVE},
    {TR_OP_SCALE_COUNT, 0, 0, TR_REG_RING},
    {TR_OP_DISTINCT_FOLD, 0, 0, TR_LAW_ASSOCIATIVE | TR_LAW_COMMUTATIVE | TR_LAW_IDEMPOTENT},
    {TR_OP_SUBTRACT, 0, 0, TR_LAW_ASSOCIATIVE | TR_LAW_COMMUTATIVE | TR_LAW_INVERSE},
    {TR_OP_DIVIDE, 0, 0, TR_LAW_DIVISION},
    {TR_OP_ORDER, 0, 0, TR_LAW_ORDERED},
    {TR_OP_POWERS, 0, 0, TR_LAW_ASSOCIATIVE | TR_LAW_DISTRIBUTIVE},
    {TR_OP_CONTRACT, 0, 0, TR_REG_RING},
    {TR_OP_HISTOGRAM, TR_REG_C_CODE, TR_REG_C_COUNT, TR_LAW_EXACT},
    {TR_OP_BITS, TR_REG_C_CODE, TR_REG_C_ROWSET, TR_LAW_ORDERED | TR_LAW_EXACT},
    {TR_OP_AND, TR_REG_C_ROWSET, TR_REG_C_ROWSET, TR_LAW_MUL_IDEMPOTENT},
    {TR_OP_POPCOUNT, TR_REG_C_ROWSET, TR_REG_C_COUNT, TR_LAW_EXACT},
    {TR_OP_MARK_CELLS, TR_REG_C_CODE, TR_REG_C_ROWSET, TR_LAW_EXACT},
    {TR_OP_VERIFY, TR_REG_C_INTEGER, TR_REG_C_COUNT, TR_LAW_ORDERED | TR_LAW_EXACT},
    {TR_OP_EMPTY, 0, 0, TR_LAW_IDENTITY},
    {TR_OP_LIFT, 0, 0, 0u},
    {TR_OP_RESIDUE, 0, 0, 0u},
    {TR_OP_NULL_ENCODE, 0, 0, 0u},
    {TR_OP_MULTIPLY, 0, 0, TR_LAW_DISTRIBUTIVE},
    {TR_OP_EUCLID, 0, 0, TR_REG_RING | TR_LAW_INVERSE | TR_LAW_ORDERED | TR_LAW_CHAR0},
    {TR_OP_SOLVE, 0, 0, TR_REG_FIELD},
    {TR_OP_INERTIA, 0, 0, TR_REG_FIELD | TR_LAW_ORDERED},
    {TR_OP_MINIMIZE, 0, 0, TR_REG_FIELD | TR_LAW_ORDERED},
    {TR_OP_ROOTS, 0, TR_REG_C_QINTERVAL, TR_REG_FIELD | TR_LAW_ORDERED | TR_LAW_CHAR0},
    {TR_OP_CRT, TR_REG_C_GFP, TR_REG_C_BIGINT, TR_LAW_EXACT},
    {TR_OP_RATRECON, TR_REG_C_GFP, TR_REG_C_BIGQ, TR_LAW_EXACT},
    {TR_OP_ENCLOSE, 0, 0, TR_LAW_ENCLOSURE},
    {TR_OP_COEFFICIENTS, TR_REG_C_SERIES, TR_REG_C_BIGQ, TR_LAW_EXACT},
    {TR_OP_DIFFERENTIATE, 0, 0, 0u},
    {TR_OP_PREDICATE, 0, TR_REG_C_KLEENE, TR_LAW_ORDERED},
    {TR_OP_LOGIC, 0, 0, TR_REG_LATTICE},
    {TR_OP_COFACTOR, 0, 0, TR_REG_LATTICE},
    {TR_OP_DIFFERENCE, 0, 0, TR_REG_FOLD | TR_LAW_INVERSE},
    {TR_OP_QUANTIFY, TR_REG_C_COUNT, TR_REG_C_KLEENE, TR_LAW_EXACT},
    {TR_OP_TEST, TR_REG_C_COUNT, TR_REG_C_BOOLEAN, TR_LAW_EXACT},
    {TR_OP_OCCUPANCY, 0, TR_REG_C_COUNT, TR_LAW_COMPLEMENT | TR_LAW_EXACT},
    {TR_OP_ZETA, 0, 0, TR_REG_FOLD},
    {TR_OP_MOBIUS, 0, 0, TR_REG_FOLD | TR_LAW_INVERSE},
    {TR_OP_WALSH, 0, 0, TR_REG_FOLD | TR_LAW_INVERSE | TR_LAW_CHAR0},
    {TR_OP_ROLLUP, 0, 0, TR_REG_FOLD},
    {TR_OP_REFINE, TR_REG_C_CODE, TR_REG_C_CODE, TR_LAW_EXACT},
    {TR_OP_GROUP, TR_REG_C_CODE, TR_REG_C_CODE, TR_LAW_EXACT},
    {TR_OP_UPDATE, 0, 0, TR_REG_FOLD | TR_LAW_INVERSE},
    {TR_OP_PREFIX, 0, 0, TR_REG_FOLD | TR_LAW_ORDERED},
    {TR_OP_ATTRIBUTE, 0, 0, TR_REG_FIELD | TR_LAW_CHAR0},
    {TR_OP_COMPILE, 0, 0, TR_REG_LATTICE},
    {TR_OP_MODEL_COUNT, 0, 0, TR_REG_RING},
    {TR_OP_PROPAGATE, 0, 0, TR_REG_RING},
    {TR_OP_EXPLAIN, 0, 0, TR_REG_LATTICE},
    {TR_OP_PROBABILITY, 0, 0, TR_REG_RING},
    {TR_OP_NORMALIZE, 0, 0, TR_REG_FOLD},
    {TR_OP_FIXPOINT, 0, 0, TR_REG_LATTICE},
    {TR_OP_RELAX, 0, 0, TR_LAW_ASSOCIATIVE | TR_LAW_COMMUTATIVE | TR_LAW_IDEMPOTENT | TR_LAW_ORDERED},
    {TR_OP_BOUNDARY, 0, 0, TR_REG_RING | TR_LAW_INVERSE},
    {TR_OP_HOMOLOGY, 0, 0, TR_REG_FIELD},
    {TR_OP_PERSISTENCE, 0, 0, TR_REG_FIELD},
};

/* the declared conversions (EXACT / CHECKED / ENCLOSING at any step; LIFT / REDUCTION / ENCODING by their operator) */
#define TR_REG_CONVERSIONS 59
#define TR__C(x) TR_REG_C_##x
#define TR__V(f, t, k, op) {TR__C(f), TR__C(t), TR_CONV_##k, op}
static const tr_reg_conversion tr_reg_conversions[TR_REG_CONVERSIONS] = {
    TR__V(CODE, INTEGER, CHECKED, 0),        /* logical = code + offset, refused outside int64 */
    TR__V(INTEGER, INT128, EXACT, 0),
    TR__V(INTEGER, RATIONAL, EXACT, 0),      /* n / 1 */
    TR__V(INTEGER, MOD128, EXACT, 0),        /* the residue (a ring homomorphism, injective on int64) */
    TR__V(INTEGER, MINMAX, EXACT, 0),
    TR__V(INTEGER, INTERVAL, ENCLOSING, 0),
    TR__V(INT128, INTEGER, CHECKED, 0),      /* narrowing */
    TR__V(INT128, RATIONAL, CHECKED, 0),
    TR__V(INT128, MOD128, EXACT, 0),
    TR__V(MOD128, INT128, CHECKED, 0),       /* only under a proven bound |v| < 2^127 */
    TR__V(COUNT, INTEGER, CHECKED, 0),
    TR__V(COUNT, INT128, EXACT, 0),
    TR__V(COUNT, MOD128, EXACT, 0),
    TR__V(BOOLEAN, COUNT, EXACT, 0),         /* the indicator 0 / 1 */
    TR__V(BOOLEAN, GF2, EXACT, 0),           /* the same two elements (each step's laws are checked) */
    TR__V(GF2, BOOLEAN, EXACT, 0),
    TR__V(RATIONAL, INTERVAL, ENCLOSING, 0),
    TR__V(INTEGER, BIGINT, EXACT, 0),
    TR__V(INT128, BIGINT, EXACT, 0),
    TR__V(COUNT, BIGINT, EXACT, 0),
    TR__V(BIGINT, INTEGER, CHECKED, 0),      /* narrowing */
    TR__V(BIGINT, INT128, CHECKED, 0),
    TR__V(INTEGER, COUNT, CHECKED, 0),       /* a nonnegative check (a negative multiplicity is refused) */
    TR__V(RATIONAL, BIGQ, EXACT, 0),
    TR__V(INTEGER, BIGQ, EXACT, 0),
    TR__V(INT128, BIGQ, EXACT, 0),
    TR__V(BIGINT, BIGQ, EXACT, 0),
    TR__V(BIGQ, RATIONAL, CHECKED, 0),       /* narrowing: numerator and denominator within int64 */
    TR__V(BIGQ, BIGINT, CHECKED, 0),         /* an integral rational only */
    TR__V(INT128, INTERVAL, ENCLOSING, 0),
    TR__V(BIGINT, INTERVAL, ENCLOSING, 0),
    TR__V(BIGQ, INTERVAL, ENCLOSING, 0),     /* the tight [RD(x), RU(x)] */
    TR__V(FLOAT64, BIGQ, CHECKED, 0),        /* the exact dyadic value of a finite double (NaN, inf refused) */
    TR__V(FLOAT64, INTERVAL, CHECKED, 0),    /* a finite double as a point interval */
    TR__V(INTERVAL, CINTERVAL, EXACT, 0),    /* im = [0, 0] */
    TR__V(CINTERVAL, INTERVAL, CHECKED, 0),  /* only when im = [0, 0] */
    TR__V(INTEGER, QINTERVAL, EXACT, 0),     /* the point [x, x] */
    TR__V(RATIONAL, QINTERVAL, EXACT, 0),
    TR__V(BIGQ, QINTERVAL, EXACT, 0),
    TR__V(INTERVAL, QINTERVAL, EXACT, 0),    /* double endpoints are exact dyadic rationals */
    TR__V(QINTERVAL, INTERVAL, ENCLOSING, 0),/* endpoints rounded outward */
    TR__V(RATIONAL, SERIES, EXACT, 0),       /* a constant series */
    TR__V(BIGQ, SERIES, EXACT, 0),
    TR__V(BOOLEAN, KLEENE, EXACT, 0),        /* two-valued logic embeds into Kleene's */
    TR__V(ROWSET, GF2, EXACT, 0),            /* a row bitmap read as GF(2) bits (XOR replaces union) */
    TR__V(GF2, ROWSET, EXACT, 0),
    TR__V(KLEENE, BOOLEAN, ENCODING, TR_OP_NULL_ENCODE),   /* UNKNOWN as FALSE (IS TRUE), or kept apart */
    TR__V(KLEENE, CODE, ENCODING, TR_OP_NULL_ENCODE),      /* the digit 0 FALSE, 1 TRUE, 2 UNKNOWN */
    TR__V(GF2, INTEGER, LIFT, TR_OP_LIFT),   /* 0 / 1: 1 + 1 is 0 in GF(2), 2 in the integers */
    TR__V(GF2, COUNT, LIFT, TR_OP_LIFT),
    TR__V(GFP, INTEGER, LIFT, TR_OP_LIFT),   /* the representative in [0, p) or the symmetric one */
    TR__V(GFP, BIGINT, LIFT, TR_OP_LIFT),
    TR__V(INTEGER, GFP, REDUCTION, TR_OP_RESIDUE),
    TR__V(INT128, GFP, REDUCTION, TR_OP_RESIDUE),
    TR__V(BIGINT, GFP, REDUCTION, TR_OP_RESIDUE),
    TR__V(RATIONAL, GFP, REDUCTION, TR_OP_RESIDUE),        /* n d^-1 mod p (d = 0 mod p refused) */
    TR__V(BIGQ, GFP, REDUCTION, TR_OP_RESIDUE),
    TR__V(INTEGER, GF2, REDUCTION, TR_OP_RESIDUE),         /* x mod 2 */
    TR__V(COUNT, GF2, REDUCTION, TR_OP_RESIDUE),           /* the parity of a count */
};
#undef TR__V
/* the declared conversion from -> to, NULL when none is declared */
static inline const tr_reg_conversion *tr_reg_conversion_find(int32_t from, int32_t to) {
    for (int i = 0; i < TR_REG_CONVERSIONS; i++)
        if (tr_reg_conversions[i].from == from && tr_reg_conversions[i].to == to) return &tr_reg_conversions[i];
    return 0;
}
/* the declared conversion from -> to (its kind), 0 when none is declared */
static inline int32_t tr_reg_conversion_kind(int32_t from, int32_t to) {
    const tr_reg_conversion *c = tr_reg_conversion_find(from, to);
    return c ? c->kind : 0;
}

/* ------------------------------------------------------------------ the composition check */
enum { TR_REG_CONTRACT_EXACT = 0, TR_REG_CONTRACT_ENCLOSURE = 1 };
enum { TR_REG_REFUSE_NONE = 0, TR_REG_REFUSE_LAW = 1, TR_REG_REFUSE_CONVERSION = 2, TR_REG_REFUSE_OPERATOR = 3,
       TR_REG_REFUSE_CARRIER = 4,          /* the operator runs only in another carrier */
       TR_REG_REFUSE_IMPLEMENTATION = 5,   /* no module implements the operator in that carrier (on that device) */
       TR_REG_REFUSE_BUDGET = 6, TR_REG_REFUSE_OVERFLOW = 7,
       TR_REG_REFUSE_BOUND = 8,            /* a bound needed before execution is not proven */
       TR_REG_REFUSE_STALE = 9,            /* a state's scope, source or revision differs */
       TR_REG_REFUSE_PROOF = 10,           /* a required proof is absent (emptiness not proven) */
       TR_REG_REFUSE_DEVICE = 11, TR_REG_REFUSE_INVALID = 12, TR_REG_REFUSE_UNSUPPORTED = 13,
       TR_REG_REFUSE_MEMORY = 14, TR_REG_REFUSALS = 15 };
typedef struct { int32_t op, carrier; } tr_reg_step;
/* kind TR_REG_REFUSE_*; code the native code (0 accepted); step the refused step (-1 none) with its operator and
 * carrier; law the first missing law (a TR_LAW_* bit); from/to the missing conversion. 32 bytes. */
typedef struct { int32_t kind, code, step, op, carrier, law, from, to; } tr_reg_refusal;

static inline void tr_reg_refusal_none(tr_reg_refusal *r) {
    if (r) { r->kind = TR_REG_REFUSE_NONE; r->code = 0; r->step = -1; r->op = 0; r->carrier = 0; r->law = 0; r->from = 0; r->to = 0; }
}
static inline int32_t tr_reg_refuse(tr_reg_refusal *r, int32_t kind, int32_t code, int32_t step, int32_t op,
                                    int32_t carrier, int32_t law, int32_t from, int32_t to) {
    if (r) { r->kind = kind; r->code = code; r->step = step; r->op = op; r->carrier = carrier; r->law = law; r->from = from; r->to = to; }
    return code;
}
/* whether the declared conversion c may change the carrier at a step of operator op under the contract */
static inline int tr_reg_conversion_admitted(const tr_reg_conversion *c, int32_t op, int32_t contract) {
    if (!c) return 0;
    if (c->op) return op == c->op;                              /* LIFT / REDUCTION / ENCODING: their own step only */
    return c->kind != TR_CONV_ENCLOSING || contract == TR_REG_CONTRACT_ENCLOSURE;
}
/* the laws and conversions of a chain (see the header comment); *output (may be NULL) gets the final carrier */
static inline int32_t tr_reg_check_laws(const tr_reg_step *steps, int32_t n, int32_t input, int32_t contract,
                                        int32_t *output, tr_reg_refusal *r) {
    tr_reg_refusal_none(r);
    if ((n && !steps) || n < 0 || input < 1 || input >= TR_REG_CARRIERS ||
        (contract != TR_REG_CONTRACT_EXACT && contract != TR_REG_CONTRACT_ENCLOSURE))
        return tr_reg_refuse(r, TR_REG_REFUSE_INVALID, TR_REG_E_INVALID, -1, 0, input, 0, 0, 0);
    int32_t cur = input;
    for (int32_t i = 0; i < n; i++) {
        int32_t op = steps[i].op, c = steps[i].carrier;
        if (op < 1 || op >= TR_REG_OPERATORS)
            return tr_reg_refuse(r, TR_REG_REFUSE_OPERATOR, TR_REG_E_OPERATOR, i, op, c, 0, 0, 0);
        if (c < 1 || c >= TR_REG_CARRIERS)
            return tr_reg_refuse(r, TR_REG_REFUSE_INVALID, TR_REG_E_INVALID, i, op, c, 0, 0, 0);
        const tr_reg_operator *o = &tr_reg_operators[op];
        if (o->input && o->input != c)
            return tr_reg_refuse(r, TR_REG_REFUSE_CARRIER, TR_REG_E_CARRIER, i, op, c, 0, 0, 0);
        if (cur != c && !tr_reg_conversion_admitted(tr_reg_conversion_find(cur, c), op, contract))
            return tr_reg_refuse(r, TR_REG_REFUSE_CONVERSION, TR_REG_E_CARRIER, i, op, c, 0, cur, c);
        uint32_t have = tr_reg_carrier_laws[c], need = o->required;
        if (contract == TR_REG_CONTRACT_EXACT) need |= TR_LAW_EXACT;
        else if (!(have & (TR_LAW_EXACT | TR_LAW_ENCLOSURE))) need |= TR_LAW_ENCLOSURE;
        uint32_t missing = need & ~have;
        if (missing)
            return tr_reg_refuse(r, TR_REG_REFUSE_LAW, TR_REG_E_CARRIER, i, op, c, (int32_t)(missing & (0u - missing)), 0, 0);
        cur = o->output ? o->output : c;
    }
    if (output) *output = cur;
    return 0;
}

/* ------------------------------------------------------------------ representations and their chains */
enum { TR_REG_R_FORMAL = 1,      /* checked rational polynomials (math_modules.h, FND-002) */
       TR_REG_R_MOMENTS = 2,     /* scalar degree-two moment banks (math_modules.h, FND-003) */
       TR_REG_R_CELLS = 3,       /* the distinct-cell pre-pass (math_budgets.h, FND-010) */
       TR_REG_R_VERIFY = 4,      /* relation verification (math_disc.h, DISC-001) */
       TR_REG_R_SCAN = 5, TR_REG_R_HISTOGRAM = 6, TR_REG_R_POWER = 7, TR_REG_R_BITPLANE = 8, TR_REG_R_EMPTY = 9,
       TR_REG_REPRESENTATIONS = 10 };
/* The rows of the planner's coefficient tables [TR_REG_MODULES][2][TR_COST_COMPONENTS] (math_planner.h): the modules
 * with candidates, ids 1 .. 9, where module id == representation id. The family modules (ids TR_REG_M_SCOPE ..
 * TR_REG_M_END - 1) have no candidates. */
#define TR_REG_MODULES TR_REG_REPRESENTATIONS
#define TR_REG_CHAIN 6
typedef struct { int32_t input, len; tr_reg_step step[TR_REG_CHAIN]; } tr_reg_chain;   /* 56 bytes */
#define TR__S0 {0, 0}
#define TR__EMPTY_CHAIN {0, 0, {TR__S0, TR__S0, TR__S0, TR__S0, TR__S0, TR__S0}}
/* [representation][0 SUM family, 1 MIN/MAX family, 2 multiplicities] */
static const tr_reg_chain tr_reg_chains[TR_REG_REPRESENTATIONS][3] = {
    {TR__EMPTY_CHAIN, TR__EMPTY_CHAIN, TR__EMPTY_CHAIN},
    {{TR_REG_C_RATIONAL, 3, {{TR_OP_POLY_MUL, TR_REG_C_RATIONAL}, {TR_OP_POLY_INTEGRATE, TR_REG_C_RATIONAL},
                             {TR_OP_POLY_EVALUATE, TR_REG_C_RATIONAL}, TR__S0, TR__S0, TR__S0}},
     TR__EMPTY_CHAIN, TR__EMPTY_CHAIN},
    {{TR_REG_C_CODE, 4, {{TR_OP_CONVERT, TR_REG_C_INTEGER}, {TR_OP_POWERS, TR_REG_C_INTEGER},
                         {TR_OP_FOLD, TR_REG_C_INTEGER}, {TR_OP_BANK_REDUCE, TR_REG_C_RATIONAL}, TR__S0, TR__S0}},
     TR__EMPTY_CHAIN, TR__EMPTY_CHAIN},
    {{TR_REG_C_CODE, 3, {{TR_OP_MARK_CELLS, TR_REG_C_CODE}, {TR_OP_DISTINCT_FOLD, TR_REG_C_ROWSET},
                         {TR_OP_POPCOUNT, TR_REG_C_ROWSET}, TR__S0, TR__S0, TR__S0}},
     TR__EMPTY_CHAIN, TR__EMPTY_CHAIN},
    {{TR_REG_C_CODE, 4, {{TR_OP_CONVERT, TR_REG_C_INTEGER}, {TR_OP_VERIFY, TR_REG_C_INTEGER},
                         {TR_OP_FOLD, TR_REG_C_COUNT}, {TR_OP_MERGE, TR_REG_C_COUNT}, TR__S0, TR__S0}},
     {TR_REG_C_CODE, 3, {{TR_OP_CONVERT, TR_REG_C_INTEGER}, {TR_OP_FOLD, TR_REG_C_MINMAX},
                         {TR_OP_MERGE, TR_REG_C_MINMAX}, TR__S0, TR__S0, TR__S0}},
     TR__EMPTY_CHAIN},
    {{TR_REG_C_CODE, 4, {{TR_OP_CONVERT, TR_REG_C_INTEGER}, {TR_OP_EVAL, TR_REG_C_INTEGER},
                         {TR_OP_FOLD, TR_REG_C_INT128}, {TR_OP_MERGE, TR_REG_C_INT128}, TR__S0, TR__S0}},
     {TR_REG_C_CODE, 4, {{TR_OP_CONVERT, TR_REG_C_INTEGER}, {TR_OP_EVAL, TR_REG_C_INTEGER},
                         {TR_OP_FOLD, TR_REG_C_MINMAX}, {TR_OP_MERGE, TR_REG_C_MINMAX}, TR__S0, TR__S0}},
     TR__EMPTY_CHAIN},
    {{TR_REG_C_CODE, 5, {{TR_OP_CONVERT, TR_REG_C_INTEGER}, {TR_OP_EVAL, TR_REG_C_INTEGER},
                         {TR_OP_SCALE_COUNT, TR_REG_C_INT128}, {TR_OP_FOLD, TR_REG_C_INT128},
                         {TR_OP_MERGE, TR_REG_C_INT128}, TR__S0}},
     {TR_REG_C_CODE, 4, {{TR_OP_CONVERT, TR_REG_C_INTEGER}, {TR_OP_EVAL, TR_REG_C_INTEGER},
                         {TR_OP_DISTINCT_FOLD, TR_REG_C_MINMAX}, {TR_OP_MERGE, TR_REG_C_MINMAX}, TR__S0, TR__S0}},
     {TR_REG_C_CODE, 3, {{TR_OP_HISTOGRAM, TR_REG_C_CODE}, {TR_OP_FOLD, TR_REG_C_COUNT},
                         {TR_OP_MERGE, TR_REG_C_COUNT}, TR__S0, TR__S0, TR__S0}}},
    {{TR_REG_C_CODE, 6, {{TR_OP_CONVERT, TR_REG_C_INTEGER}, {TR_OP_POWERS, TR_REG_C_MOD128},
                         {TR_OP_FOLD, TR_REG_C_MOD128}, {TR_OP_MERGE, TR_REG_C_MOD128},
                         {TR_OP_CONTRACT, TR_REG_C_MOD128}, {TR_OP_CONVERT, TR_REG_C_INT128}}},
     {TR_REG_C_CODE, 5, {{TR_OP_CONVERT, TR_REG_C_INTEGER}, {TR_OP_POWERS, TR_REG_C_MINMAX},
                         {TR_OP_FOLD, TR_REG_C_MINMAX}, {TR_OP_MERGE, TR_REG_C_MINMAX},
                         {TR_OP_CONTRACT, TR_REG_C_MINMAX}, TR__S0}},
     TR__EMPTY_CHAIN},
    {{TR_REG_C_CODE, 6, {{TR_OP_BITS, TR_REG_C_CODE}, {TR_OP_AND, TR_REG_C_ROWSET},
                         {TR_OP_POPCOUNT, TR_REG_C_ROWSET}, {TR_OP_MERGE, TR_REG_C_COUNT},
                         {TR_OP_CONTRACT, TR_REG_C_MOD128}, {TR_OP_CONVERT, TR_REG_C_INT128}}},
     {TR_REG_C_CODE, 5, {{TR_OP_BITS, TR_REG_C_CODE}, {TR_OP_AND, TR_REG_C_ROWSET},
                         {TR_OP_POPCOUNT, TR_REG_C_ROWSET}, {TR_OP_MERGE, TR_REG_C_COUNT},
                         {TR_OP_CONTRACT, TR_REG_C_MINMAX}, TR__S0}},
     TR__EMPTY_CHAIN},
    {{TR_REG_C_CODE, 2, {{TR_OP_CONVERT, TR_REG_C_INTEGER}, {TR_OP_EMPTY, TR_REG_C_INT128},
                         TR__S0, TR__S0, TR__S0, TR__S0}},
     {TR_REG_C_CODE, 2, {{TR_OP_CONVERT, TR_REG_C_INTEGER}, {TR_OP_EMPTY, TR_REG_C_MINMAX},
                         TR__S0, TR__S0, TR__S0, TR__S0}},
     TR__EMPTY_CHAIN},
};

/* ------------------------------------------------------------------ requests, results, explanations */
#define TR_REG_POLYS 4
#define TR_REG_DEGREE 8                      /* TR_MATH_DEGREE */
enum { TR_REG_SUM = 1, TR_REG_MIN = 2, TR_REG_MAX = 3 };
enum { TR_REG_DEV_CPU = 0, TR_REG_DEV_CUDA = 1, TR_REG_DEV_EMULATION = 2 };   /* where an execution ran */
enum { TR_REG_ON_CPU = 1, TR_REG_ON_CUDA = 2 };                               /* device masks */
#define TR_REG_ON_BOTH (TR_REG_ON_CPU | TR_REG_ON_CUDA)
enum { TR_REG_FINAL_CPU = 0, TR_REG_FINAL_GPU = 1 };                          /* where a result is finalized */
#define TR_REG_ENTRY_NAME 40                 /* entry point names (39 characters and the terminator) */
/* 384 bytes; pointers are borrowed (the caller keeps them alive during a call) */
typedef struct {
    uint64_t rows, ncodes;          /* physical rows; the code domain: every code < ncodes (<= 2^32) */
    const uint32_t *codes;          /* rows stored codes */
    const uint32_t *validity;       /* rows flags, 0 = NULL; NULL pointer: no NULLs */
    const uint64_t *selected;       /* ceil(rows / 64) words of selected rows (bits past rows ignored); NULL: every row */
    int64_t offset;                 /* logical value = code + offset */
    uint64_t source, revision;      /* identity of the column's source and its revision (state reuse) */
    int32_t width, aggregate, npolys, reserved;   /* width: bits of a code (ncodes <= 2^width, <= 32) */
    int32_t degree[TR_REG_POLYS];
    int64_t coef[TR_REG_POLYS][TR_REG_DEGREE + 1];   /* f_j(x) = sum_k coef[j][k] x^k */
} tr_reg_request;
/* SUM: the exact sum; MIN/MAX: the extreme as a sign-extended int64, meaningful only when valid > 0. 96 bytes. */
typedef struct {
    tr_math_i128 value[TR_REG_POLYS];
    uint64_t selected, valid, nulls;      /* selected rows; of them valid and NULL */
    int32_t npolys, aggregate;
} tr_reg_result;
/* What ran (status 0) or why nothing ran (status 1): the module (its id; representation 0 for a family module), the
 * device, the entry point that computed the result (the family's own function when the registry dispatched to one)
 * and the refusal. 176 bytes. */
typedef struct {
    int32_t abi, status, module, representation;
    int32_t device, finalize, reused, code;     /* device TR_REG_DEV_*; reused: a cached state was contracted */
    tr_reg_refusal refusal;                     /* why refused (kind, step, law, conversion) */
    tr_math_refusal budget;                     /* a budget refusal: resource, need, limit */
    uint64_t rows_scanned, words_scanned, state_elements, state_bytes, device_bytes, native_calls;
    char entry[TR_REG_ENTRY_NAME];
} tr_reg_explain;
/* a built state (see tr_reg_state_build): 104 bytes */
typedef struct {
    int32_t representation, degree, width, has_validity;   /* degree: BITPLANE D (POWER: highest exponent) */
    uint32_t support, has_selection;                        /* support: POWER exponent mask (bit k: S_k kept) */
    uint64_t elements, bytes, rows, ncodes, selected, valid, nulls, source, revision;
    int64_t offset;
} tr_reg_state_info;

/* the entry point that computes (copied, truncated to TR_REG_ENTRY_NAME - 1 characters) */
static inline void tr_reg_explain_entry(tr_reg_explain *e, const char *name) {
    if (!e) return;
    int32_t i = 0;
    for (; name && name[i] && i < TR_REG_ENTRY_NAME - 1; i++) e->entry[i] = name[i];
    for (; i < TR_REG_ENTRY_NAME; i++) e->entry[i] = 0;
}
/* an explanation of module `module` running representation `repr` (0 for a family module) */
static inline void tr_reg_explain_begin(tr_reg_explain *e, int32_t module, int32_t repr, int32_t device,
                                        int32_t finalize) {
    if (!e) return;
    e->abi = TR_REG_ABI; e->status = 0; e->module = module; e->representation = repr;
    e->device = device; e->finalize = finalize; e->reused = 0; e->code = 0;
    tr_reg_refusal_none(&e->refusal); tr_budget_none(&e->budget);
    e->rows_scanned = e->words_scanned = e->state_elements = e->state_bytes = e->device_bytes = e->native_calls = 0;
    tr_reg_explain_entry(e, 0);
}
static inline void tr_reg_explain_init(tr_reg_explain *e, int32_t repr, int32_t device, int32_t finalize) {
    tr_reg_explain_begin(e, repr, repr, device, finalize);   /* modules 1 .. 9: module id == representation id */
}
static inline int32_t tr_reg_explain_refuse(tr_reg_explain *e, int32_t kind, int32_t code) {
    if (e) {
        e->status = 1; e->code = code;
        if (e->refusal.kind == TR_REG_REFUSE_NONE) { e->refusal.kind = kind; e->refusal.code = code; }
    }
    return code;
}
static inline void tr_reg_result_init(tr_reg_result *res, const tr_reg_request *r) {
    for (int j = 0; j < TR_REG_POLYS; j++) { res->value[j].lo = 0; res->value[j].hi = 0; }
    res->selected = res->valid = res->nulls = 0;
    res->npolys = r->npolys; res->aggregate = r->aggregate;
}

/* ------------------------------------------------------------------ exact helpers shared with the kernels */
typedef unsigned __int128 tr_u128;
/* the rows of word w that exist */
TR_NUM_HD uint64_t tr_reg_rowmask(uint64_t rows, uint64_t w) {
    uint64_t left = rows - w * 64;
    return left >= 64 ? ~(uint64_t)0 : (((uint64_t)1 << left) - 1);
}
/* f(x) by Horner's rule, every step checked to fit int64 (128-bit products): 1 with *out, or 0 (overflow) */
TR_NUM_HD int tr_reg_eval(const int64_t *a, int32_t d, int64_t x, int64_t *out) {
    int64_t acc = a[d];
    for (int32_t k = d - 1; k >= 0; k--) {
        __int128 v = (__int128)acc * x + a[k];
        if (v > INT64_MAX || v < INT64_MIN) return 0;
        acc = (int64_t)v;
    }
    *out = acc;
    return 1;
}
/* f(x) modulo 2^128 (x any 128-bit integer; unsigned arithmetic wraps by definition) */
TR_NUM_HD tr_u128 tr_reg_eval_mod(const int64_t *a, int32_t d, __int128 x) {
    tr_u128 xr = (tr_u128)x, acc = (tr_u128)(__int128)a[d];
    for (int32_t k = d - 1; k >= 0; k--) acc = acc * xr + (tr_u128)(__int128)a[k];
    return acc;
}
TR_NUM_HD tr_math_i128 tr_reg_i128_of(tr_u128 v) {
    tr_math_i128 r; r.lo = (int64_t)(uint64_t)v; r.hi = (int64_t)(uint64_t)(v >> 64); return r;
}
TR_NUM_HD tr_math_i128 tr_reg_i128_of_i64(int64_t v) { tr_math_i128 r; r.lo = v; r.hi = v < 0 ? -1 : 0; return r; }
TR_NUM_HD int tr_reg_bit(const uint64_t *bits, uint64_t i) { return (int)((bits[i >> 6] >> (i & 63)) & 1); }

/* the exponents a request uses (bit k: some coefficient of x^k is nonzero), bit 0 always (the count) */
static inline uint32_t tr_reg_support(const tr_reg_request *r) {
    uint32_t m = 1;
    for (int32_t j = 0; j < r->npolys; j++)
        for (int32_t k = 0; k <= r->degree[j]; k++) if (r->coef[j][k]) m |= 1u << k;
    return m;
}
static inline int32_t tr_reg_max_degree(const tr_reg_request *r) {
    int32_t d = 0;
    for (int32_t j = 0; j < r->npolys; j++) if (r->degree[j] > d) d = r->degree[j];
    return d;
}
static inline uint64_t tr_reg_terms(const tr_reg_request *r) {
    uint64_t t = 0;
    for (int32_t j = 0; j < r->npolys; j++) t += (uint64_t)r->degree[j] + 1;
    return t;
}
static inline uint64_t tr_reg_words(const tr_reg_request *r) { return (r->rows + 63) / 64; }
static inline int tr_reg_popcount32(uint32_t x) { return __builtin_popcount(x); }
/* the plane sets of BITPLANE: sum_{s <= min(D, width)} C(width, s), and the Moebius terms sum_s C(width, s) 2^s */
static inline uint64_t tr_reg_plane_sets(int32_t width, int32_t D) {
    uint64_t p = 0;
    for (int32_t s = 0; s <= D && s <= width; s++) p = tr_budget_add(p, tr_budget_binomial((uint64_t)width, (uint64_t)s));
    return p;
}
static inline uint64_t tr_reg_plane_terms(int32_t width, int32_t D) {
    uint64_t p = 0;
    for (int32_t s = 0; s <= D && s <= width; s++)
        p = tr_budget_add(p, tr_budget_mul(tr_budget_binomial((uint64_t)width, (uint64_t)s), (uint64_t)1 << s));
    return p;
}
/* the plane sets in their fixed order: by size, then increasing mask (Gosper's successor); returns how many */
static inline uint64_t tr_reg_plane_sets_fill(int32_t width, int32_t D, uint32_t *out, uint64_t cap) {
    uint64_t n = 0, limit = (uint64_t)1 << width;
    for (int32_t s = 0; s <= D && s <= width; s++) {
        uint64_t x = s == 0 ? 0 : (((uint64_t)1 << s) - 1);
        while (x < limit || (s == 0 && x == 0)) {
            if (n < cap) out[n] = (uint32_t)x;
            n++;
            if (s == 0) break;
            uint64_t c = x & (0 - x), y = x + c;
            x = (((y ^ x) >> 2) / c) | y;
        }
    }
    return n;
}
/* the request's validity (TR_MATH_INVALID), kernel limits (TR_MATH_BUDGET: degree > 8, with *refusal) and logical
 * domain (TR_MATH_OVERFLOW: offset + ncodes - 1 outside int64) */
static inline int32_t tr_reg_request_check(const tr_reg_request *r, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!r || (r->rows && !r->codes) || r->npolys < 1 || r->npolys > TR_REG_POLYS ||
        r->aggregate < TR_REG_SUM || r->aggregate > TR_REG_MAX || r->ncodes < 1 || r->ncodes > ((uint64_t)1 << 32) ||
        r->width < 0 || r->width > 32 || (r->width < 32 && r->ncodes > ((uint64_t)1 << r->width)))
        return TR_REG_E_INVALID;
    for (int32_t j = 0; j < r->npolys; j++) {
        if (r->degree[j] < 0) return TR_REG_E_INVALID;
        if (r->degree[j] > TR_REG_DEGREE) {
            if (refusal) { refusal->resource = TR_BUDGET_DEGREE; refusal->refused = 1;
                           refusal->need = (uint64_t)r->degree[j]; refusal->limit = TR_REG_DEGREE; }
            return TR_REG_E_BUDGET;
        }
    }
    if ((__int128)r->offset + (__int128)(r->ncodes - 1) > INT64_MAX) return TR_REG_E_OVERFLOW;
    return 0;
}
/* the proven bound of POWER and BITPLANE: 1 when, for every j, rows_bound * sum_k |a_jk| M^k <= 2^127 - 1 */
static inline tr_u128 tr_reg_sat_mul(tr_u128 a, tr_u128 b) {
    const tr_u128 top = ~(tr_u128)0;
    return a && b > top / a ? top : a * b;
}
static inline tr_u128 tr_reg_sat_add(tr_u128 a, tr_u128 b) { const tr_u128 top = ~(tr_u128)0; return a > top - b ? top : a + b; }
static inline int tr_reg_bound_ok(const tr_reg_request *r, uint64_t rows_bound) {
    uint64_t lo = tr_abs_u64(r->offset);
    __int128 high = (__int128)r->offset + (__int128)(r->ncodes - 1);
    uint64_t hi = high < 0 ? (uint64_t)(-high) : (uint64_t)high;
    tr_u128 M = lo > hi ? lo : hi, limit = (((tr_u128)1) << 127) - 1;
    for (int32_t j = 0; j < r->npolys; j++) {
        tr_u128 sum = 0, p = 1;
        for (int32_t k = 0; k <= r->degree[j]; k++) {
            sum = tr_reg_sat_add(sum, tr_reg_sat_mul(tr_abs_u64(r->coef[j][k]), p));
            p = tr_reg_sat_mul(p, M);
        }
        if (tr_reg_sat_mul(sum, rows_bound) > limit) return 0;
    }
    return 1;
}
/* the proven bound of SCAN and HISTOGRAM in a plan: 1 when, for every j and every |x| <= M, each Horner step stays
 * within int64: B_d = |a_d|, B_k = B_(k+1) M + |a_k| <= 2^63 - 1 (then no checked evaluation can refuse; a direct
 * execution checks each value instead and refuses only when one actually leaves int64) */
static inline int tr_reg_eval_bound_ok(const tr_reg_request *r) {
    uint64_t lo = tr_abs_u64(r->offset);
    __int128 high = (__int128)r->offset + (__int128)(r->ncodes - 1);
    uint64_t hi = high < 0 ? (uint64_t)(-high) : (uint64_t)high;
    tr_u128 M = lo > hi ? lo : hi, limit = (tr_u128)INT64_MAX;
    for (int32_t j = 0; j < r->npolys; j++) {
        tr_u128 b = tr_abs_u64(r->coef[j][r->degree[j]]);
        if (b > limit) return 0;
        for (int32_t k = r->degree[j] - 1; k >= 0; k--) {
            b = tr_reg_sat_add(tr_reg_sat_mul(b, M), tr_abs_u64(r->coef[j][k]));
            if (b > limit) return 0;
        }
    }
    return 1;
}
/* the chain of a representation for the request's aggregate (family 0 SUM, 1 MIN/MAX) */
static inline const tr_reg_chain *tr_reg_chain_of(int32_t repr, int32_t aggregate) {
    if (repr < 1 || repr >= TR_REG_REPRESENTATIONS) return 0;
    return &tr_reg_chains[repr][aggregate == TR_REG_SUM ? 0 : 1];
}
/* the representation's laws and conversions for this request; also the histogram's multiplicity chain */
static inline int32_t tr_reg_check_representation(int32_t repr, const tr_reg_request *r, tr_reg_refusal *refusal) {
    const tr_reg_chain *ch = tr_reg_chain_of(repr, r->aggregate);
    if (!ch || !ch->len) return tr_reg_refuse(refusal, TR_REG_REFUSE_UNSUPPORTED, TR_MATH_UNSUPPORTED, -1, 0, 0, 0, 0, 0);
    int32_t rc = tr_reg_check_laws(ch->step, ch->len, ch->input, TR_REG_CONTRACT_EXACT, 0, refusal);
    if (rc || repr != TR_REG_R_HISTOGRAM) return rc;
    const tr_reg_chain *aux = &tr_reg_chains[repr][2];
    return tr_reg_check_laws(aux->step, aux->len, aux->input, TR_REG_CONTRACT_EXACT, 0, refusal);
}

/* ------------------------------------------------------------------ budget needs */
#define TR_REG_ACC_BYTES 160            /* device accumulators: 4 x 16 (sums) + 4 x 8 (extremes) + 8 x 8 (counters) */
/* bytes of the device copies of a request's inputs: codes, validity, selection */
static inline uint64_t tr_reg_input_bytes(const tr_reg_request *r) {
    uint64_t b = tr_budget_mul(r->rows, 4);
    if (r->validity) b = tr_budget_add(b, tr_budget_mul(r->rows, 4));
    if (r->selected) b = tr_budget_add(b, tr_budget_bitmap_bytes(r->rows));
    return b;
}
/* state elements of a representation: HISTOGRAM ncodes, POWER the support's exponents, BITPLANE the plane sets */
static inline uint64_t tr_reg_state_elements(const tr_reg_request *r, int32_t repr, uint32_t support, int32_t D) {
    if (repr == TR_REG_R_HISTOGRAM) return r->ncodes;
    if (repr == TR_REG_R_POWER) return (uint64_t)tr_reg_popcount32(support);
    if (repr == TR_REG_R_BITPLANE) return tr_reg_plane_sets(r->width, D);
    return 0;
}
/* What a representation needs: variables 1, degree, support (terms), output (polynomials), states, circuit
 * (BITPLANE Moebius terms) and bytes: the state (8 per histogram cell or plane count, 16 per power sum, 4 per plane
 * set mask), and on CUDA the device copies of the inputs, the device state, the host copy a CPU finalization reads
 * and the accumulators. A reused state allocates nothing. support/D: the state's exponents and plane-set degree. */
static inline void tr_reg_need(const tr_reg_request *r, int32_t repr, int32_t device, int32_t finalize, int32_t reuse,
                               uint32_t support, int32_t D, tr_math_need *n) {
    for (int i = 0; i < TR_BUDGET_RESOURCES; i++) n->need[i] = 0;
    n->need[TR_BUDGET_VARIABLES] = 1;
    n->need[TR_BUDGET_DEGREE] = (uint64_t)tr_reg_max_degree(r);
    n->need[TR_BUDGET_SUPPORT] = tr_reg_terms(r);
    n->need[TR_BUDGET_OUTPUT] = (uint64_t)r->npolys;
    if (repr == TR_REG_R_EMPTY || repr == TR_REG_R_SCAN) {
        if (repr == TR_REG_R_SCAN && device == TR_REG_ON_CUDA)
            n->need[TR_BUDGET_BYTES] = tr_budget_add(tr_reg_input_bytes(r), TR_REG_ACC_BYTES);
        return;
    }
    uint64_t elements = tr_reg_state_elements(r, repr, support, D), cell = repr == TR_REG_R_POWER ? 16 : 8;
    n->need[TR_BUDGET_STATES] = elements;
    if (repr == TR_REG_R_BITPLANE) n->need[TR_BUDGET_CIRCUIT] = tr_reg_plane_terms(r->width, D);
    if (reuse) return;
    uint64_t state = tr_budget_mul(elements, cell);
    if (repr == TR_REG_R_BITPLANE) state = tr_budget_add(state, tr_budget_mul(elements, 4));
    if (device != TR_REG_ON_CUDA) { n->need[TR_BUDGET_BYTES] = state; return; }
    uint64_t bytes = tr_budget_add(tr_reg_input_bytes(r), TR_REG_ACC_BYTES);
    bytes = tr_budget_add(bytes, state);                                          /* the device state */
    if (finalize == TR_REG_FINAL_CPU) bytes = tr_budget_add(bytes, tr_budget_mul(elements, cell));   /* host copy */
    n->need[TR_BUDGET_BYTES] = bytes;
}
/* the exact number of selected rows of a request (popcount of the selection; every row when NULL) */
static inline uint64_t tr_reg_selected_count(const tr_reg_request *r) {
    if (!r->selected) return r->rows;
    uint64_t n = 0, words = tr_reg_words(r);
    for (uint64_t w = 0; w < words; w++) n += (uint64_t)__builtin_popcountll(r->selected[w] & tr_reg_rowmask(r->rows, w));
    return n;
}

/* ------------------------------------------------------------------ CPU finalization (shared by both libraries) */
/* HISTOGRAM: sum_c count_c f(c + offset), or min/max over the occupied codes; TR_MATH_OVERFLOW when the checked
 * evaluation of an occupied code leaves int64. *evaluated: the occupied codes evaluated. */
static inline int32_t tr_reg_hist_contract(const tr_reg_request *r, const uint64_t *counts, tr_reg_result *res,
                                           uint64_t *evaluated) {
    __int128 sum[TR_REG_POLYS] = {0, 0, 0, 0};
    int64_t ext[TR_REG_POLYS];
    for (int32_t j = 0; j < TR_REG_POLYS; j++) ext[j] = r->aggregate == TR_REG_MIN ? INT64_MAX : INT64_MIN;
    uint64_t occupied = 0;
    int overflow = 0;
    for (uint64_t c = 0; c < r->ncodes; c++) {
        uint64_t n = counts[c];
        if (!n) continue;
        occupied++;
        int64_t x = (int64_t)((__int128)c + r->offset);
        for (int32_t j = 0; j < r->npolys; j++) {
            int64_t v;
            if (!tr_reg_eval(r->coef[j], r->degree[j], x, &v)) { overflow = 1; continue; }
            if (r->aggregate == TR_REG_SUM) sum[j] += (__int128)v * (__int128)n;
            else if (r->aggregate == TR_REG_MIN) { if (v < ext[j]) ext[j] = v; }
            else if (v > ext[j]) ext[j] = v;
        }
    }
    if (evaluated) *evaluated = occupied;
    if (overflow) return TR_REG_E_OVERFLOW;
    for (int32_t j = 0; j < r->npolys; j++)
        res->value[j] = r->aggregate == TR_REG_SUM ? tr_reg_i128_of((tr_u128)sum[j]) : tr_reg_i128_of_i64(ext[j]);
    return 0;
}
/* POWER: sum_k a_k S_k modulo 2^128 (sums: the residues of the exponents in support, increasing, two words each) */
static inline void tr_reg_power_contract(const tr_reg_request *r, uint32_t support, const uint64_t *sums,
                                         tr_reg_result *res) {
    for (int32_t j = 0; j < r->npolys; j++) {
        tr_u128 acc = 0;
        int32_t idx = 0;
        for (int32_t k = 0; k <= TR_REG_DEGREE; k++) {
            if (!((support >> k) & 1)) continue;
            if (k <= r->degree[j] && r->coef[j][k]) {
                tr_u128 s = ((tr_u128)sums[2 * idx + 1] << 64) | sums[2 * idx];
                acc += (tr_u128)(__int128)r->coef[j][k] * s;
            }
            idx++;
        }
        res->value[j] = tr_reg_i128_of(acc);
    }
}
/* BITPLANE: sum_S w_S count_S modulo 2^128 with w_S = sum_{T subset of S} (-1)^(|S|-|T|) f(T + offset) (zero when
 * |S| > degree); *terms: the Moebius terms evaluated */
static inline void tr_reg_plane_contract(const tr_reg_request *r, const uint32_t *sets, const uint64_t *counts,
                                         uint64_t nsets, tr_reg_result *res, uint64_t *terms) {
    uint64_t evaluated = 0;
    for (int32_t j = 0; j < r->npolys; j++) {
        tr_u128 acc = 0;
        for (uint64_t s = 0; s < nsets; s++) {
            uint32_t S = sets[s];
            int size = tr_reg_popcount32(S);
            if (size > r->degree[j] || !counts[s]) continue;
            tr_u128 w = 0;
            uint32_t T = S;
            for (;;) {
                tr_u128 g = tr_reg_eval_mod(r->coef[j], r->degree[j], (__int128)T + r->offset);
                if ((size - tr_reg_popcount32(T)) & 1) w -= g; else w += g;
                evaluated++;
                if (!T) break;
                T = (T - 1) & S;
            }
            acc += w * (tr_u128)counts[s];
        }
        res->value[j] = tr_reg_i128_of(acc);
    }
    if (terms) *terms = evaluated;
}

/* ------------------------------------------------------------------ cost components and candidates */
enum { TR_COST_WORDS = 0,        /* 64-row selection words visited */
       TR_COST_SCAN = 1,         /* row-columns read: rows read x columns (codes, validity) */
       TR_COST_CHAIN = 2,        /* per-row operator steps: Horner steps, increments, powers, plane bits */
       TR_COST_COMBINE = 3,      /* one-time algebraic combination: Moebius terms of the plane weights */
       TR_COST_CONSTRUCT = 4,    /* state elements constructed (allocated, initialized, accumulated into) */
       TR_COST_REUSE = 5,        /* reusing a cached state: words compared to prove its scope, plus one */
       TR_COST_FUSION = 6,       /* passes saved by fusing the batch's outputs (a cost <= 0) */
       TR_COST_MATERIALIZE = 7,  /* output bytes */
       TR_COST_TRANSFER = 8,     /* host <-> device bytes */
       TR_COST_CPU_FINAL = 9,    /* finalization on the CPU: state elements and terms contracted */
       TR_COST_GPU_FINAL = 10,   /* finalization on the device */
       TR_COST_FIXED = 11,       /* calls, allocations and kernel launches */
       TR_COST_COMPONENTS = 12 };
#define TR_REG_MAX_CANDIDATES 16
enum { TR_REG_REQ_FORMAL = 1, TR_REG_REQ_MOMENTS = 2, TR_REG_REQ_CELLS = 4, TR_REG_REQ_VERIFY = 8,
       TR_REG_REQ_POLYSUM = 16,
       TR_REG_REQ_FAMILY = 32 };   /* the family's own requests (its API); not planned by FND-008 */
#define TR_REG_REQUEST_KINDS 6
/* One way to execute a request. The registry fills its identity, validity (refusal: law, conversion, proof, bound,
 * stale), budget needs and exact work amounts per cost component; the planner (math_planner.h) fills status, reason,
 * the budget refusal and the costs. Components in `shared` (a bit per component) are counted for `batch` unfused
 * passes; the fusion component subtracts the batch - 1 passes a fused execution saves. 376 bytes. */
typedef struct {
    int32_t module, representation, device, finalize;   /* device TR_REG_ON_CPU / TR_REG_ON_CUDA */
    int32_t reuse, batch, status, reason;
    int32_t code, valid;                                /* code: native code of a rejection; valid: registry check */
    uint32_t shared, reserved;
    tr_reg_refusal refusal;
    tr_math_refusal budget;
    tr_math_need need;
    uint64_t amount[TR_COST_COMPONENTS];
    double cost[TR_COST_COMPONENTS], total;
} tr_reg_candidate;

/* ------------------------------------------------------------------ module descriptors (math_registry.c) */
/* family module ids (representation 0); a new family takes the next id before TR_REG_M_END */
enum { TR_REG_M_SCOPE = 10,             /* FND-004 data scopes */
       TR_REG_M_BANK = 11,              /* FND-004 grouped moment banks */
       TR_REG_M_CARRIERS = 12,          /* FND-009 numeric carriers */
       TR_REG_M_HANDLES = 13,           /* FND-013 handle tables, sparse integer polynomials */
       TR_REG_M_LIN_MOMENTS = 14,       /* LIN-001 grouped moment matrices */
       TR_REG_M_LIN_SOLVE = 15,         /* LIN-002 solves, inertia, covariance, eigensystems */
       TR_REG_M_OPT = 16,               /* OPT-001 quadratic minimizers */
       TR_REG_M_JET_DUAL = 17,          /* JET-001 dual numbers */
       TR_REG_M_JET_TAYLOR = 18,        /* JET-002 multivariate Taylor jets */
       TR_REG_M_INV = 19,               /* INV-001 inverse answers */
       TR_REG_M_CALC = 20,              /* CALC-003 finite-separation calculus */
       TR_REG_M_MULTI = 21,             /* MULTI-001 hierarchy rollups and contrasts */
       TR_REG_M_DELTA = 22,             /* DELTA-001 signed state updates */
       TR_REG_M_BOOL = 23,              /* BOOL-001 predicate cofactors and differences */
       TR_REG_M_FIELD = 24,             /* FIELD-001 GF(2) primitives and parity dependencies */
       TR_REG_M_DIST_HIST = 25,         /* DIST-001 reusable histograms */
       TR_REG_M_DIST_PREFIX = 26,       /* DIST-002 prefix distribution calculus */
       TR_REG_M_RESP = 27,              /* RESP-002 piecewise loss curves */
       TR_REG_M_QUANT_GROUPS = 28,      /* QUANT-001 quantified group predicates */
       TR_REG_M_QUANT_COVER = 29,       /* QUANT-002 relational division */
       TR_REG_M_LATTICE_HIST = 30,      /* LATTICE-001 occupancy histograms */
       TR_REG_M_LATTICE_TRANSFORM = 31, /* LATTICE-002 subset-lattice transforms */
       TR_REG_M_SPEC_WALSH = 32,        /* SPEC-002 Walsh spectra */
       TR_REG_M_INFLUENCE = 33,         /* BOOL-002 influence and interactions */
       TR_REG_M_PLANNER = 34,           /* FND-008 the planner's selectivity sample */
       TR_REG_M_DIST_INFO = 35,         /* DIST-003 entropy, mutual information, contingency statistics */
       TR_REG_M_PAIR = 36,              /* PAIR-001 sorted-distribution pairwise summaries */
       TR_REG_M_GEN = 37,               /* GEN-001 generating polynomials of finite supports */
       TR_REG_M_CERT = 38,              /* FND-007 proof certificates and evidence scope */
       TR_REG_M_PERSIST = 39,           /* FND-011 state serialization */
       TR_REG_M_CONV = 40,              /* GEN-002 convolution and bounded formal series */
       TR_REG_M_PROV = 41,              /* PROV-001 factorized provenance payloads */
       TR_REG_M_BILINEAR = 42,          /* DELTA-002 bilinear/join and nonlinear update laws */
       TR_REG_M_ENV = 43,               /* ENV-001 univariate group envelopes */
       TR_REG_M_ATTR = 44,              /* ATTR-001 sparse interaction and contribution attribution */
       TR_REG_M_COUNT = 45,             /* COUNT-001 observed-domain constraint counts and witnesses */
       TR_REG_M_PARITY = 46,            /* FIELD-002 parity relationship discovery and substitution */
       TR_REG_M_UNC = 47,               /* UNC-001 intervals, affine forms, congruences, pruning */
       TR_REG_M_PRE = 48,               /* PRE-001 affine, square, residue, piecewise maps to codes */
       TR_REG_M_IDENT = 49,             /* DISC-002 Gram nullspaces, verified identities */
       TR_REG_M_GEOM = 50,              /* GEOM-001 Groebner bases, reduction certificates */
       TR_REG_M_TENSOR_STATE = 51,      /* TENSOR-001 sparse and factored moment tensors */
       TR_REG_M_TENSOR_CONTRACT = 52,   /* TENSOR-002 costed tensor contraction planning */
       TR_REG_M_KERNEL = 53,            /* KERNEL-001 exact finite-feature kernel embeddings */
       TR_REG_M_SYMCOUNT = 54,          /* COUNT-002 symbolic full-domain algebraic model counting */
       TR_REG_M_EXPLAIN = 55,           /* PROV-002 explanation specialization and deletion propagation */
       TR_REG_M_FOURIER = 56,           /* SPEC-001 fixed spectral basis states and phase actions */
       TR_REG_M_LINUPD = 57,            /* LIN-003 leave-group-out and low-rank model updates */
       TR_REG_M_PROB = 58,              /* PROB-001 small-support partition and likelihood functions */
       TR_REG_M_SYM_ACTION = 59,        /* SYM-001 closed-basis transformation actions */
       TR_REG_M_SYM_SYMMETRY = 60,      /* SYM-002 certified data symmetries and orbits */
       TR_REG_M_APPROX = 61,            /* UNC-002 certified series, low rank, rounding, composition */
       TR_REG_M_REFINE = 62,            /* MULTI-002 coarse-to-fine bounds, localized rollups */
       TR_REG_M_INTERV = 63,            /* INV-002 minimal-cost changes attaining a target, algebraic only */
       TR_REG_M_POLYOPT = 64,           /* OPT-002 exact QPs and LPs, certified branch and bound */
       TR_REG_M_GRAPH_LAYOUT = 65,      /* GRAPH-001 graph layouts, degrees, frontiers and semiring products */
       TR_REG_M_GRAPH_PATHS = 66,       /* GRAPH-002 reachability, bounded walk counts, paths with cycle refusal, energies */
       TR_REG_M_ELIM = 67,              /* GEOM-002 generator sets, reconstructions, local forms */
       TR_REG_M_PREIMG = 68,            /* PRE-002 roots, residues, bit cubes, fallbacks */
       TR_REG_M_TOPO_COMPLEX = 69,      /* TOPO-001 declared occupancy complexes and local topology */
       TR_REG_M_TOPO_HOMOLOGY = 70,     /* TOPO-002 bounded cycles, homology and persistence */
       TR_REG_M_END = 71 };             /* module ids are 1 .. TR_REG_M_END - 1 */
#define TR_REG_IMPLS 32                  /* implementation rows a descriptor carries */
#define TR_REG_ENTRIES 12                /* entry points per device a descriptor names */
/* An operator a module implements in a carrier on devices (TR_REG_ON_*). 16 bytes. */
typedef struct { int32_t op, carrier, devices, module; } tr_reg_impl;
enum { TR_REG_STATE_NONE = 0, TR_REG_STATE_FORMAL = 1, TR_REG_STATE_MOMENT_BANK = 2, TR_REG_STATE_CELL_BITMAP = 3,
       TR_REG_STATE_EVIDENCE = 4, TR_REG_STATE_HISTOGRAM = 5, TR_REG_STATE_POWER_SUMS = 6, TR_REG_STATE_PLANE_COUNTS = 7,
       TR_REG_STATE_EMPTY_PROOF = 8,
       TR_REG_STATE_SCOPE = 9,            /* selected rows, exact logical values, validity, dense group ids */
       TR_REG_STATE_GROUPED_MOMENTS = 10, /* per group: rows, per term counts and exact sums */
       TR_REG_STATE_VALUES = 11,          /* immutable carrier values (big numbers, residues, enclosures, series) */
       TR_REG_STATE_HANDLE_TABLE = 12,    /* owned handles to immutable states */
       TR_REG_STATE_GRAM = 13,            /* per group: Gram slots (counts, exact sums of products) */
       TR_REG_STATE_QUADRATIC = 14,       /* an exact quadratic objective [A | b | c] */
       TR_REG_STATE_SEPARATED = 15,       /* retained separation coefficients (exact-endpoint enclosures) */
       TR_REG_STATE_JETS = 16,            /* per group: rows, count and the sums of a plan's jet coefficients */
       TR_REG_STATE_HIERARCHY = 17,       /* banks of every level and their refinement maps */
       TR_REG_STATE_MAINTAINED = 18,      /* the cells of a weighted bag, with an identity and a sign promise */
       TR_REG_STATE_CONDITION_MASKS = 19, /* value and valid words per condition (or the assignments) */
       TR_REG_STATE_GF2_BITS = 20,        /* packed GF(2) bit variables or matrices */
       TR_REG_STATE_JOINT_HISTOGRAM = 21, /* per group: cells of k coordinates (NULL a category), counts, weights */
       TR_REG_STATE_PREFIX_SUMS = 22,     /* per group: a sorted support and prefix sums of m x^k */
       TR_REG_STATE_QUANTIFIER = 23,      /* per group: TRUE, FALSE, UNKNOWN rows and declared presence */
       TR_REG_STATE_COVERAGE = 24,        /* per group: the categories present (masks or sorted sets) */
       TR_REG_STATE_OCCUPANCY = 25,       /* 2^k (or 3^k) cell counts of Boolean axes, per-axis counts */
       TR_REG_STATE_LATTICE_VECTOR = 26,  /* 2^k exact values over the subsets of k axes */
       TR_REG_STATE_SPECTRUM = 27,        /* 2^k Walsh coefficients */
       TR_REG_STATE_CONTINGENCY = 28,     /* per group: marginal and contingency tables of a joint histogram */
       TR_REG_STATE_GENERATING = 29,      /* exact coefficients of a generating polynomial (declared shifts) */
       TR_REG_STATE_CERTIFICATE = 30,     /* a claim, its kind (law, schema, snapshot, hypothesis) and its scope */
       TR_REG_STATE_RECORD = 31,          /* a versioned, checksummed serialized state */
       TR_REG_STATE_PROVENANCE = 32,      /* a shared derivation DAG over source-tuple variables */
       TR_REG_STATE_JOIN_TERMS = 33,      /* per key and group: the side states of a bilinear join and its cross terms */
       TR_REG_STATE_ENVELOPE = 34,        /* the winning group per parameter piece and the exact breakpoints */
       TR_REG_STATE_GAME = 35,              /* a declared cooperative game (coalition values or a provenance plan with players) and its Moebius support */
       TR_REG_STATE_CONSTRAINT_COUNTS = 36, /* per constraint: TRUE / FALSE / UNKNOWN points, witnesses, rows covered */
       TR_REG_STATE_PARITY_RELATIONS = 37,  /* certified parity relations, rewritten predicates */
       TR_REG_STATE_ABSTRACT = 38,          /* per group and column: rows, present, min, max, gcd */
       TR_REG_STATE_CODE_SET = 39,          /* spans of stored codes, branch programs and laws */
       TR_REG_STATE_IDEAL = 40,             /* a polynomial basis with cofactors over relation certificates */
       TR_REG_STATE_MOMENT_TENSOR = 41,     /* per group: dense, sparse or factored moment or count tensors */
       TR_REG_STATE_CONTRACTION_PLAN = 42,  /* costed contraction orders, step shapes, bytes, lifetimes */
       TR_REG_STATE_KERNEL_FEATURES = 43,   /* per group: a feature tensor, weight sums, self-pair terms */
       TR_REG_STATE_DECISION_DIAGRAM = 44,  /* a reduced ordered BDD over a variable order (or the messages of an elimination) and its exact weighted counts */
       TR_REG_STATE_EXPLANATION = 45,       /* a retained provenance cone with reverse edges and its baseline under a base valuation */
       TR_REG_STATE_SPECTRAL_BASIS = 46,    /* per group: exact scaled bounds of the Fourier features of a fixed frequency set, rows, weight sums */
       TR_REG_STATE_SOLVED_SYSTEM = 47,     /* an exact solved system: A, b, its canonical solution set, A^-1, det(A), the last update's denominator */
       TR_REG_STATE_SUPPORT_MASSES = 48,    /* per group: the mass of every point of a declared finite support (zeros kept), N, rows */
       TR_REG_STATE_BASIS_ACTION = 49,      /* a closed monomial basis, exact or interval action matrices, retained moments */
       TR_REG_STATE_SYMMETRY = 50,          /* a declared group, its multiset evidence and scope record, orbit summaries */
       TR_REG_STATE_APPROXIMATION = 51,     /* an approximant, its norm, assumptions, budget, error terms, provenance */
       TR_REG_STATE_REFINEMENT = 52,        /* per level: cells, abstract inputs and links; maintained leaves */
       TR_REG_STATE_INTERVENTION = 53,      /* grid plans (residues, cost order) and shift aggregates (prefix sums) */
       TR_REG_STATE_PROGRAM = 54,           /* an objective with linear or polynomial constraints and bounds */
       TR_REG_STATE_GRAPH_LAYOUT = 55,      /* a vertex dictionary, CSR / CSC entries with multiplicities, canonical edges and incidence lists of a snapshot */
       TR_REG_STATE_GRAPH_FIXPOINT = 56,    /* reach sets and levels, bounded walk tables, path trees and certified iterates of a layout */
       TR_REG_STATE_ELIMINATION = 57,       /* generators, certified reconstructions, row checks */
       TR_REG_STATE_CUBICAL_COMPLEX = 58,   /* occupancy, cells, signed incidences of a declared grid */
       TR_REG_STATE_CHAIN_REDUCTION = 59,   /* ranks, Betti numbers, cycles, persistence pairs */
       TR_REG_STATE_KINDS = 60 };
enum { TR_PROOF_DOMAIN = 1,        /* a typed domain / formal context contract (FND-001) */
       TR_PROOF_SNAPSHOT = 2,      /* the state's source identity and revision */
       TR_PROOF_SCOPE = 4,         /* the exact selected rows of a reused state */
       TR_PROOF_BOUND = 8,         /* a magnitude bound proven before execution */
       TR_PROOF_EXACT_COUNT = 16,  /* an exact selected-row count */
       TR_PROOF_CATALOG = 32,      /* a relationship catalog's snapshot (revision, rows, used rows) */
       TR_PROOF_CELL_DOMAIN = 64,  /* every id inside the declared domain */
       TR_PROOF_REFINEMENT = 128,  /* a refinement map certified (or declared and verified) on the rows */
       TR_PROOF_PRIME = 256,       /* moduli verified prime (deterministic Miller-Rabin) */
       TR_PROOF_IDENTITY = 512,    /* the exact construction identity of a reused state (compared byte for byte) */
       TR_PROOF_HANDLE = 1024,     /* a live handle retained for the operation (a closed one is refused) */
       TR_PROOF_NULL_POLICY = 2048 };   /* a declared NULL policy: SQL or listwise validity, a NULL encoding */
#define TR_REG_PROOFS 12
enum { TR_REG_COST_NONE = 0,       /* zeros: nothing calibrated (the planner has no candidates for the module) */
       TR_REG_COST_ESTIMATE = 1 }; /* hand-set relative estimates, checked (not calibrated) by the planner benchmark */
/* 1832 bytes. cost[device][component]: relative work units per unit of the component's amount, NOT time
 * (device 0 the CPU, 1 a CUDA device; host-side components of a CUDA execution use the host's rate). key: a unique
 * short name (the representation's name for ids 1 .. 9); family: the source family torbles/math_<family>.* (the
 * first slice: mathematics). impl / nimpl / operators are filled from tr_reg_impls. */
typedef struct {
    int32_t abi, id, representation, state_kind;
    int32_t requests, devices, nimpl, finalize;          /* requests TR_REG_REQ_*; devices TR_REG_ON_*; finalize bits */
    uint32_t carriers, laws_relied, laws_guaranteed, proofs;
    uint64_t operators[TR_REG_OPERATOR_WORDS];           /* bit op: implemented */
    int32_t cost_basis, reserved;
    tr_reg_impl impl[TR_REG_IMPLS];
    double cost[2][TR_COST_COMPONENTS];
    char name[40], capability[16], family[16], key[24];
    char cpu_entry[TR_REG_ENTRIES][TR_REG_ENTRY_NAME], cuda_entry[TR_REG_ENTRIES][TR_REG_ENTRY_NAME];
} tr_reg_module;

/* ------------------------------------------------------------------ implementations (shared with the CUDA library) */
/* Every operator a module implements in a carrier, with the devices that run it ({op, carrier, devices, module}).
 * The composition check with implementations (tr_reg_check_chain), the module lookup (tr_reg_find) and the
 * registry's transforms read this table; a descriptor's impl list is its module's rows, in this order. */
#define TR__I(m, op, c, d) {TR_OP_##op, TR_REG_C_##c, d, m}
#define TR__CPU TR_REG_ON_CPU
#define TR__ALL TR_REG_ON_BOTH
static const tr_reg_impl tr_reg_impls[] = {
    /* 1 FND-002 checked rational polynomials (the operator set of math_modules.c) */
    TR__I(TR_REG_R_FORMAL, POLY_ADD, RATIONAL, TR__CPU), TR__I(TR_REG_R_FORMAL, POLY_SUB, RATIONAL, TR__CPU),
    TR__I(TR_REG_R_FORMAL, POLY_MUL, RATIONAL, TR__CPU), TR__I(TR_REG_R_FORMAL, POLY_POW, RATIONAL, TR__CPU),
    TR__I(TR_REG_R_FORMAL, POLY_DERIVATIVE, RATIONAL, TR__CPU), TR__I(TR_REG_R_FORMAL, POLY_INTEGRATE, RATIONAL, TR__CPU),
    TR__I(TR_REG_R_FORMAL, POLY_SUBSTITUTE, RATIONAL, TR__CPU), TR__I(TR_REG_R_FORMAL, POLY_EVALUATE, RATIONAL, TR__CPU),
    /* 2 FND-003 scalar integer moments */
    TR__I(TR_REG_R_MOMENTS, CONVERT, INTEGER, TR__CPU), TR__I(TR_REG_R_MOMENTS, POWERS, INTEGER, TR__CPU),
    TR__I(TR_REG_R_MOMENTS, FOLD, INTEGER, TR__CPU), TR__I(TR_REG_R_MOMENTS, BANK_REDUCE, RATIONAL, TR__CPU),
    /* 3 FND-010 distinct cell pre-pass */
    TR__I(TR_REG_R_CELLS, MARK_CELLS, CODE, TR__ALL), TR__I(TR_REG_R_CELLS, DISTINCT_FOLD, ROWSET, TR__ALL),
    TR__I(TR_REG_R_CELLS, POPCOUNT, ROWSET, TR__ALL),
    /* 4 DISC-001 relation verification */
    TR__I(TR_REG_R_VERIFY, CONVERT, INTEGER, TR__ALL), TR__I(TR_REG_R_VERIFY, VERIFY, INTEGER, TR__ALL),
    TR__I(TR_REG_R_VERIFY, FOLD, COUNT, TR__ALL), TR__I(TR_REG_R_VERIFY, MERGE, COUNT, TR__ALL),
    TR__I(TR_REG_R_VERIFY, FOLD, MINMAX, TR__ALL), TR__I(TR_REG_R_VERIFY, MERGE, MINMAX, TR__ALL),
    /* 5 .. 9 FND-005 / FND-008 the representations of a polynomial aggregate */
    TR__I(TR_REG_R_SCAN, CONVERT, INTEGER, TR__ALL), TR__I(TR_REG_R_SCAN, EVAL, INTEGER, TR__ALL),
    TR__I(TR_REG_R_SCAN, FOLD, INT128, TR__ALL), TR__I(TR_REG_R_SCAN, MERGE, INT128, TR__ALL),
    TR__I(TR_REG_R_SCAN, FOLD, MINMAX, TR__ALL), TR__I(TR_REG_R_SCAN, MERGE, MINMAX, TR__ALL),
    TR__I(TR_REG_R_HISTOGRAM, CONVERT, INTEGER, TR__ALL), TR__I(TR_REG_R_HISTOGRAM, EVAL, INTEGER, TR__ALL),
    TR__I(TR_REG_R_HISTOGRAM, HISTOGRAM, CODE, TR__ALL), TR__I(TR_REG_R_HISTOGRAM, FOLD, COUNT, TR__ALL),
    TR__I(TR_REG_R_HISTOGRAM, MERGE, COUNT, TR__ALL), TR__I(TR_REG_R_HISTOGRAM, SCALE_COUNT, INT128, TR__ALL),
    TR__I(TR_REG_R_HISTOGRAM, FOLD, INT128, TR__ALL), TR__I(TR_REG_R_HISTOGRAM, MERGE, INT128, TR__ALL),
    TR__I(TR_REG_R_HISTOGRAM, DISTINCT_FOLD, MINMAX, TR__ALL), TR__I(TR_REG_R_HISTOGRAM, MERGE, MINMAX, TR__ALL),
    TR__I(TR_REG_R_POWER, CONVERT, INTEGER, TR__ALL), TR__I(TR_REG_R_POWER, POWERS, MOD128, TR__ALL),
    TR__I(TR_REG_R_POWER, FOLD, MOD128, TR__ALL), TR__I(TR_REG_R_POWER, MERGE, MOD128, TR__ALL),
    TR__I(TR_REG_R_POWER, CONTRACT, MOD128, TR__ALL), TR__I(TR_REG_R_POWER, CONVERT, INT128, TR__ALL),
    TR__I(TR_REG_R_BITPLANE, BITS, CODE, TR__ALL), TR__I(TR_REG_R_BITPLANE, AND, ROWSET, TR__ALL),
    TR__I(TR_REG_R_BITPLANE, POPCOUNT, ROWSET, TR__ALL), TR__I(TR_REG_R_BITPLANE, MERGE, COUNT, TR__ALL),
    TR__I(TR_REG_R_BITPLANE, CONTRACT, MOD128, TR__ALL), TR__I(TR_REG_R_BITPLANE, CONVERT, INT128, TR__ALL),
    TR__I(TR_REG_R_EMPTY, CONVERT, INTEGER, TR__CPU), TR__I(TR_REG_R_EMPTY, EMPTY, INT128, TR__CPU),
    TR__I(TR_REG_R_EMPTY, EMPTY, MINMAX, TR__CPU),
    /* 10 FND-004 data scope: logical values, group ids, refinement maps, selected rows */
    TR__I(TR_REG_M_SCOPE, CONVERT, INTEGER, TR__CPU), TR__I(TR_REG_M_SCOPE, GROUP, CODE, TR__CPU),
    TR__I(TR_REG_M_SCOPE, REFINE, CODE, TR__CPU), TR__I(TR_REG_M_SCOPE, POPCOUNT, ROWSET, TR__CPU),
    /* 11 FND-004 grouped moment bank: monomials, exact 128-bit sums and counts, chunk merges, rollups */
    TR__I(TR_REG_M_BANK, POWERS, INTEGER, TR__ALL), TR__I(TR_REG_M_BANK, FOLD, INT128, TR__ALL),
    TR__I(TR_REG_M_BANK, MERGE, INT128, TR__CPU), TR__I(TR_REG_M_BANK, FOLD, COUNT, TR__ALL),
    TR__I(TR_REG_M_BANK, MERGE, COUNT, TR__CPU), TR__I(TR_REG_M_BANK, ROLLUP, INT128, TR__ALL),
    TR__I(TR_REG_M_BANK, ROLLUP, COUNT, TR__ALL),
    /* 12 FND-009 numeric carriers */
    TR__I(TR_REG_M_CARRIERS, MULTIPLY, INTEGER, TR__ALL),      /* checked int64; exact 192-bit dot products */
    TR__I(TR_REG_M_CARRIERS, FOLD, INT128, TR__ALL),           /* exact grouped 128-bit sums */
    TR__I(TR_REG_M_CARRIERS, FOLD, BIGINT, TR__ALL),           /* the 192-bit dot accumulator; big sums */
    TR__I(TR_REG_M_CARRIERS, MULTIPLY, BIGINT, TR__CPU), TR__I(TR_REG_M_CARRIERS, EUCLID, BIGINT, TR__CPU),
    TR__I(TR_REG_M_CARRIERS, DIVIDE, RATIONAL, TR__CPU), TR__I(TR_REG_M_CARRIERS, MULTIPLY, BIGQ, TR__CPU),
    TR__I(TR_REG_M_CARRIERS, DIVIDE, BIGQ, TR__CPU),
    TR__I(TR_REG_M_CARRIERS, RESIDUE, GFP, TR__ALL),           /* int64 (CUDA), int128, big, rationals mod p */
    TR__I(TR_REG_M_CARRIERS, MULTIPLY, GFP, TR__ALL), TR__I(TR_REG_M_CARRIERS, DIVIDE, GFP, TR__ALL),
    TR__I(TR_REG_M_CARRIERS, FOLD, GFP, TR__ALL),              /* dot products, matrix-vector products mod p */
    TR__I(TR_REG_M_CARRIERS, LIFT, INTEGER, TR__CPU), TR__I(TR_REG_M_CARRIERS, LIFT, BIGINT, TR__CPU),
    TR__I(TR_REG_M_CARRIERS, CRT, GFP, TR__CPU), TR__I(TR_REG_M_CARRIERS, RATRECON, GFP, TR__CPU),
    TR__I(TR_REG_M_CARRIERS, CONVERT, INTEGER, TR__CPU), TR__I(TR_REG_M_CARRIERS, CONVERT, INT128, TR__CPU),
    TR__I(TR_REG_M_CARRIERS, CONVERT, BIGINT, TR__CPU),        /* the conversion matrix: lossless and narrowing */
    TR__I(TR_REG_M_CARRIERS, CONVERT, RATIONAL, TR__CPU), TR__I(TR_REG_M_CARRIERS, CONVERT, BIGQ, TR__CPU),
    TR__I(TR_REG_M_CARRIERS, CONVERT, INTERVAL, TR__ALL),      /* tight enclosures of exact values */
    TR__I(TR_REG_M_CARRIERS, ENCLOSE, INTERVAL, TR__CPU), TR__I(TR_REG_M_CARRIERS, ENCLOSE, CINTERVAL, TR__CPU),
    TR__I(TR_REG_M_CARRIERS, CONVERT, CINTERVAL, TR__CPU),
    TR__I(TR_REG_M_CARRIERS, FOLD, SERIES, TR__CPU), TR__I(TR_REG_M_CARRIERS, MULTIPLY, SERIES, TR__CPU),
    TR__I(TR_REG_M_CARRIERS, POLY_DERIVATIVE, SERIES, TR__CPU), TR__I(TR_REG_M_CARRIERS, POLY_SUBSTITUTE, SERIES, TR__CPU),
    TR__I(TR_REG_M_CARRIERS, COEFFICIENTS, SERIES, TR__CPU),
    /* 13 FND-013 sparse integer polynomials evaluated exactly in 128 bits (overflow flagged, never wrapped) */
    TR__I(TR_REG_M_HANDLES, EVAL, INT128, TR__ALL),
    /* 14 LIN-001 grouped Gram matrices: products, exact 128- or 192-bit sums, merges, deletions */
    TR__I(TR_REG_M_LIN_MOMENTS, MULTIPLY, INTEGER, TR__ALL), TR__I(TR_REG_M_LIN_MOMENTS, FOLD, INT128, TR__ALL),
    TR__I(TR_REG_M_LIN_MOMENTS, FOLD, BIGINT, TR__ALL), TR__I(TR_REG_M_LIN_MOMENTS, FOLD, COUNT, TR__ALL),
    TR__I(TR_REG_M_LIN_MOMENTS, MERGE, INT128, TR__CPU), TR__I(TR_REG_M_LIN_MOMENTS, MERGE, BIGINT, TR__CPU),
    TR__I(TR_REG_M_LIN_MOMENTS, SUBTRACT, INT128, TR__CPU), TR__I(TR_REG_M_LIN_MOMENTS, SUBTRACT, BIGINT, TR__CPU),
    /* 15 LIN-002 Bareiss over Z (the rational solution set), elimination mod primes, CRT, inertia, eigen bounds */
    TR__I(TR_REG_M_LIN_SOLVE, SOLVE, BIGQ, TR__CPU), TR__I(TR_REG_M_LIN_SOLVE, EUCLID, BIGINT, TR__CPU),
    TR__I(TR_REG_M_LIN_SOLVE, RESIDUE, GFP, TR__CPU), TR__I(TR_REG_M_LIN_SOLVE, SOLVE, GFP, TR__ALL),
    TR__I(TR_REG_M_LIN_SOLVE, CRT, GFP, TR__CPU), TR__I(TR_REG_M_LIN_SOLVE, INERTIA, BIGQ, TR__CPU),
    TR__I(TR_REG_M_LIN_SOLVE, CONVERT, INTERVAL, TR__CPU), TR__I(TR_REG_M_LIN_SOLVE, ENCLOSE, INTERVAL, TR__CPU),
    /* 16 OPT-001 exact minimizers; the face filter's interval elimination (CUDA) only discards faces */
    TR__I(TR_REG_M_OPT, MINIMIZE, BIGQ, TR__CPU), TR__I(TR_REG_M_OPT, INERTIA, BIGQ, TR__CPU),
    TR__I(TR_REG_M_OPT, SOLVE, BIGQ, TR__CPU), TR__I(TR_REG_M_OPT, CONVERT, INTERVAL, TR__ALL),
    TR__I(TR_REG_M_OPT, ENCLOSE, INTERVAL, TR__ALL),
    /* 17 JET-001 dual numbers (order 1) and 18 JET-002 Taylor jets: truncated series, exact readouts */
    TR__I(TR_REG_M_JET_DUAL, EVAL, SERIES, TR__ALL), TR__I(TR_REG_M_JET_DUAL, FOLD, SERIES, TR__ALL),
    TR__I(TR_REG_M_JET_DUAL, MERGE, SERIES, TR__CPU), TR__I(TR_REG_M_JET_DUAL, COEFFICIENTS, SERIES, TR__CPU),
    TR__I(TR_REG_M_JET_DUAL, CONVERT, RATIONAL, TR__CPU), TR__I(TR_REG_M_JET_DUAL, RESIDUE, GFP, TR__CPU),
    TR__I(TR_REG_M_JET_DUAL, CONVERT, INTERVAL, TR__CPU),
    TR__I(TR_REG_M_JET_TAYLOR, EVAL, SERIES, TR__ALL), TR__I(TR_REG_M_JET_TAYLOR, FOLD, SERIES, TR__ALL),
    TR__I(TR_REG_M_JET_TAYLOR, MERGE, SERIES, TR__CPU), TR__I(TR_REG_M_JET_TAYLOR, COEFFICIENTS, SERIES, TR__CPU),
    TR__I(TR_REG_M_JET_TAYLOR, CONVERT, RATIONAL, TR__CPU), TR__I(TR_REG_M_JET_TAYLOR, RESIDUE, GFP, TR__CPU),
    TR__I(TR_REG_M_JET_TAYLOR, CONVERT, INTERVAL, TR__CPU),
    /* 19 INV-001 linear targets, Sturm isolation, multimodular signs and sweeps */
    TR__I(TR_REG_M_INV, SOLVE, BIGQ, TR__CPU), TR__I(TR_REG_M_INV, ROOTS, BIGQ, TR__CPU),
    TR__I(TR_REG_M_INV, EUCLID, BIGINT, TR__CPU), TR__I(TR_REG_M_INV, RESIDUE, GFP, TR__ALL),
    TR__I(TR_REG_M_INV, EVAL, GFP, TR__ALL), TR__I(TR_REG_M_INV, CRT, GFP, TR__ALL),
    /* 20 CALC-003 certified per-row enclosures summed exactly as integer bounds; exact derivatives */
    TR__I(TR_REG_M_CALC, ENCLOSE, QINTERVAL, TR__ALL), TR__I(TR_REG_M_CALC, FOLD, QINTERVAL, TR__ALL),
    TR__I(TR_REG_M_CALC, DIFFERENTIATE, QINTERVAL, TR__CPU),
    TR__I(TR_REG_M_CALC, CONVERT, INTERVAL, TR__CPU), TR__I(TR_REG_M_CALC, ENCLOSE, INTERVAL, TR__CPU),
    /* 21 MULTI-001 certified maps, exact rollups, 192-bit updates, canonical rational contrasts */
    TR__I(TR_REG_M_MULTI, REFINE, CODE, TR__ALL), TR__I(TR_REG_M_MULTI, ROLLUP, INT128, TR__CPU),
    TR__I(TR_REG_M_MULTI, ROLLUP, COUNT, TR__CPU), TR__I(TR_REG_M_MULTI, SUBTRACT, BIGINT, TR__CPU),
    TR__I(TR_REG_M_MULTI, FOLD, BIGINT, TR__CPU), TR__I(TR_REG_M_MULTI, SUBTRACT, BIGQ, TR__ALL),
    TR__I(TR_REG_M_MULTI, DIVIDE, BIGQ, TR__ALL),
    /* 22 DELTA-001 signed updates of rows, counts and sums; the nonnegative promise */
    TR__I(TR_REG_M_DELTA, UPDATE, INTEGER, TR__ALL), TR__I(TR_REG_M_DELTA, UPDATE, INT128, TR__ALL),
    TR__I(TR_REG_M_DELTA, ORDER, INTEGER, TR__ALL), TR__I(TR_REG_M_DELTA, CONVERT, COUNT, TR__ALL),
    /* 23 BOOL-001 row conditions, Kleene evaluation, cofactors, strict differences (signed and XOR) */
    TR__I(TR_REG_M_BOOL, PREDICATE, INTEGER, TR__ALL), TR__I(TR_REG_M_BOOL, PREDICATE, RATIONAL, TR__ALL),
    TR__I(TR_REG_M_BOOL, LOGIC, KLEENE, TR__ALL), TR__I(TR_REG_M_BOOL, COFACTOR, KLEENE, TR__ALL),
    TR__I(TR_REG_M_BOOL, NULL_ENCODE, BOOLEAN, TR__ALL), TR__I(TR_REG_M_BOOL, CONVERT, COUNT, TR__ALL),
    TR__I(TR_REG_M_BOOL, DIFFERENCE, INTEGER, TR__ALL), TR__I(TR_REG_M_BOOL, DIFFERENCE, GF2, TR__ALL),
    TR__I(TR_REG_M_BOOL, FOLD, COUNT, TR__ALL), TR__I(TR_REG_M_BOOL, FOLD, INT128, TR__ALL),
    /* 24 FIELD-001 GF(2) elimination, products and syndromes, parities; weights are integer counts */
    TR__I(TR_REG_M_FIELD, SOLVE, GF2, TR__CPU), TR__I(TR_REG_M_FIELD, MULTIPLY, GF2, TR__ALL),
    TR__I(TR_REG_M_FIELD, FOLD, GF2, TR__ALL), TR__I(TR_REG_M_FIELD, BITS, CODE, TR__CPU),
    TR__I(TR_REG_M_FIELD, CONVERT, GF2, TR__ALL), TR__I(TR_REG_M_FIELD, NULL_ENCODE, BOOLEAN, TR__ALL),
    TR__I(TR_REG_M_FIELD, LIFT, COUNT, TR__ALL), TR__I(TR_REG_M_FIELD, FOLD, COUNT, TR__ALL),
    /* 25 DIST-001 cell counts and exact weight sums; merges and rollups; the cell order */
    TR__I(TR_REG_M_DIST_HIST, HISTOGRAM, CODE, TR__ALL), TR__I(TR_REG_M_DIST_HIST, FOLD, COUNT, TR__ALL),
    TR__I(TR_REG_M_DIST_HIST, FOLD, INT128, TR__ALL), TR__I(TR_REG_M_DIST_HIST, MERGE, COUNT, TR__CPU),
    TR__I(TR_REG_M_DIST_HIST, MERGE, INT128, TR__CPU), TR__I(TR_REG_M_DIST_HIST, ROLLUP, COUNT, TR__CPU),
    TR__I(TR_REG_M_DIST_HIST, ROLLUP, INT128, TR__CPU), TR__I(TR_REG_M_DIST_HIST, ORDER, INTEGER, TR__ALL),
    /* 26 DIST-002 prefix sums, ranks and quantiles, exact fractions */
    TR__I(TR_REG_M_DIST_PREFIX, PREFIX, COUNT, TR__CPU), TR__I(TR_REG_M_DIST_PREFIX, PREFIX, INT128, TR__CPU),
    TR__I(TR_REG_M_DIST_PREFIX, ORDER, INTEGER, TR__ALL), TR__I(TR_REG_M_DIST_PREFIX, SUBTRACT, INT128, TR__ALL),
    TR__I(TR_REG_M_DIST_PREFIX, DIVIDE, BIGQ, TR__ALL),
    /* 27 RESP-002 loss values and slopes (256-bit numerators), pieces and minimizer sets */
    TR__I(TR_REG_M_RESP, ORDER, INTEGER, TR__ALL), TR__I(TR_REG_M_RESP, SUBTRACT, INT128, TR__ALL),
    TR__I(TR_REG_M_RESP, EVAL, BIGINT, TR__ALL), TR__I(TR_REG_M_RESP, DIVIDE, BIGQ, TR__ALL),
    TR__I(TR_REG_M_RESP, MINIMIZE, BIGQ, TR__CPU),
    /* 28 QUANT-001 TRUE / FALSE / UNKNOWN counts per group and the SQL quantifiers */
    TR__I(TR_REG_M_QUANT_GROUPS, PREDICATE, INTEGER, TR__ALL), TR__I(TR_REG_M_QUANT_GROUPS, LOGIC, KLEENE, TR__ALL),
    TR__I(TR_REG_M_QUANT_GROUPS, NULL_ENCODE, CODE, TR__ALL), TR__I(TR_REG_M_QUANT_GROUPS, HISTOGRAM, CODE, TR__ALL),
    TR__I(TR_REG_M_QUANT_GROUPS, FOLD, COUNT, TR__ALL), TR__I(TR_REG_M_QUANT_GROUPS, MERGE, COUNT, TR__CPU),
    TR__I(TR_REG_M_QUANT_GROUPS, QUANTIFY, COUNT, TR__CPU),
    /* 29 QUANT-002 presence masks (or sorted sets), the coverage test against a required set */
    TR__I(TR_REG_M_QUANT_COVER, DISTINCT_FOLD, ROWSET, TR__ALL), TR__I(TR_REG_M_QUANT_COVER, AND, ROWSET, TR__ALL),
    TR__I(TR_REG_M_QUANT_COVER, POPCOUNT, ROWSET, TR__ALL), TR__I(TR_REG_M_QUANT_COVER, ORDER, CODE, TR__CPU),
    TR__I(TR_REG_M_QUANT_COVER, FOLD, COUNT, TR__ALL), TR__I(TR_REG_M_QUANT_COVER, TEST, COUNT, TR__ALL),
    /* 30 LATTICE-001 axes in Kleene logic, the declared NULL convention, 2^k or 3^k cells */
    TR__I(TR_REG_M_LATTICE_HIST, LOGIC, KLEENE, TR__ALL), TR__I(TR_REG_M_LATTICE_HIST, NULL_ENCODE, BOOLEAN, TR__ALL),
    TR__I(TR_REG_M_LATTICE_HIST, OCCUPANCY, BOOLEAN, TR__ALL), TR__I(TR_REG_M_LATTICE_HIST, NULL_ENCODE, CODE, TR__ALL),
    TR__I(TR_REG_M_LATTICE_HIST, HISTOGRAM, CODE, TR__ALL), TR__I(TR_REG_M_LATTICE_HIST, FOLD, COUNT, TR__ALL),
    TR__I(TR_REG_M_LATTICE_HIST, MERGE, COUNT, TR__CPU),
    /* 31 LATTICE-002 zeta and Moebius transforms (count, checked int64, int128; big integers on the CPU) */
    TR__I(TR_REG_M_LATTICE_TRANSFORM, ZETA, COUNT, TR__ALL), TR__I(TR_REG_M_LATTICE_TRANSFORM, ZETA, INTEGER, TR__ALL),
    TR__I(TR_REG_M_LATTICE_TRANSFORM, ZETA, INT128, TR__ALL), TR__I(TR_REG_M_LATTICE_TRANSFORM, ZETA, BIGINT, TR__CPU),
    TR__I(TR_REG_M_LATTICE_TRANSFORM, MOBIUS, INTEGER, TR__ALL),
    TR__I(TR_REG_M_LATTICE_TRANSFORM, MOBIUS, INT128, TR__ALL),
    TR__I(TR_REG_M_LATTICE_TRANSFORM, MOBIUS, BIGINT, TR__CPU),
    /* 32 SPEC-002 Walsh-Hadamard in checked 128 bits, dyadic inverses, 256-bit energies, constant parities */
    TR__I(TR_REG_M_SPEC_WALSH, WALSH, INT128, TR__ALL), TR__I(TR_REG_M_SPEC_WALSH, DIVIDE, BIGQ, TR__ALL),
    TR__I(TR_REG_M_SPEC_WALSH, MULTIPLY, BIGINT, TR__CPU), TR__I(TR_REG_M_SPEC_WALSH, FOLD, BIGINT, TR__CPU),
    TR__I(TR_REG_M_SPEC_WALSH, SOLVE, GF2, TR__CPU),
    /* 33 BOOL-002 XOR derivatives (influence) and signed mixed differences under two measures */
    TR__I(TR_REG_M_INFLUENCE, LOGIC, KLEENE, TR__ALL), TR__I(TR_REG_M_INFLUENCE, COFACTOR, KLEENE, TR__ALL),
    TR__I(TR_REG_M_INFLUENCE, NULL_ENCODE, BOOLEAN, TR__ALL), TR__I(TR_REG_M_INFLUENCE, DIFFERENCE, GF2, TR__ALL),
    TR__I(TR_REG_M_INFLUENCE, DIFFERENCE, INTEGER, TR__ALL), TR__I(TR_REG_M_INFLUENCE, FOLD, COUNT, TR__ALL),
    TR__I(TR_REG_M_INFLUENCE, FOLD, INT128, TR__ALL),
    /* 34 FND-008 the selected rows of sampled words, counted exactly */
    TR__I(TR_REG_M_PLANNER, POPCOUNT, ROWSET, TR__ALL), TR__I(TR_REG_M_PLANNER, FOLD, COUNT, TR__ALL),
    /* 35 DIST-003 tables from histogram states; certified log terms summed exactly; exact chi-square on the CPU */
    TR__I(TR_REG_M_DIST_INFO, MERGE, COUNT, TR__CPU), TR__I(TR_REG_M_DIST_INFO, ENCLOSE, QINTERVAL, TR__ALL),
    TR__I(TR_REG_M_DIST_INFO, FOLD, QINTERVAL, TR__ALL), TR__I(TR_REG_M_DIST_INFO, DIVIDE, BIGQ, TR__CPU),
    TR__I(TR_REG_M_DIST_INFO, ENCLOSE, INTERVAL, TR__CPU),
    /* 36 PAIR-001 one pass or merge over prefix states, exact rationals */
    TR__I(TR_REG_M_PAIR, ORDER, INTEGER, TR__ALL), TR__I(TR_REG_M_PAIR, CONTRACT, INT128, TR__ALL),
    TR__I(TR_REG_M_PAIR, DIVIDE, BIGQ, TR__ALL),
    /* 37 GEN-001 exact coefficients: convolution, evaluation, moments, series */
    TR__I(TR_REG_M_GEN, COEFFICIENTS, SERIES, TR__CPU), TR__I(TR_REG_M_GEN, MULTIPLY, INT128, TR__ALL),
    TR__I(TR_REG_M_GEN, MULTIPLY, BIGINT, TR__CPU), TR__I(TR_REG_M_GEN, EVAL, INT128, TR__ALL),
    TR__I(TR_REG_M_GEN, EVAL, BIGQ, TR__CPU), TR__I(TR_REG_M_GEN, DIFFERENTIATE, BIGINT, TR__CPU),
    TR__I(TR_REG_M_GEN, CONVERT, SERIES, TR__CPU),
    /* 38 FND-007 exhaustive snapshot evidence; polynomial claims checked exactly in 128 bits */
    TR__I(TR_REG_M_CERT, VERIFY, INTEGER, TR__ALL), TR__I(TR_REG_M_CERT, FOLD, COUNT, TR__ALL),
    TR__I(TR_REG_M_CERT, EVAL, INT128, TR__CPU),
    /* 39 FND-011 records: position-weighted checksums of 64-bit words */
    TR__I(TR_REG_M_PERSIST, CONVERT, INTEGER, TR__CPU), TR__I(TR_REG_M_PERSIST, FOLD, MOD128, TR__ALL),
    /* 40 GEN-002 declared products, per-key conditional products, bounded series and walks */
    TR__I(TR_REG_M_CONV, MULTIPLY, INT128, TR__ALL), TR__I(TR_REG_M_CONV, MULTIPLY, BIGINT, TR__CPU),
    TR__I(TR_REG_M_CONV, FOLD, BIGINT, TR__ALL), TR__I(TR_REG_M_CONV, COEFFICIENTS, SERIES, TR__CPU),
    /* 41 PROV-001 the DAG evaluated in B, N or N[X] */
    TR__I(TR_REG_M_PROV, LOGIC, BOOLEAN, TR__ALL), TR__I(TR_REG_M_PROV, EVAL, COUNT, TR__ALL),
    TR__I(TR_REG_M_PROV, MULTIPLY, BIGINT, TR__CPU),
    /* 42 DELTA-002 Delta(AB) = Delta(A) B + A Delta(B) + Delta(A) Delta(B), exactly; nonlinear finalizers keep their own state */
    TR__I(TR_REG_M_BILINEAR, UPDATE, INT128, TR__ALL), TR__I(TR_REG_M_BILINEAR, MULTIPLY, INT128, TR__ALL),
    TR__I(TR_REG_M_BILINEAR, FOLD, BIGINT, TR__ALL), TR__I(TR_REG_M_BILINEAR, UPDATE, INTEGER, TR__CPU),
    TR__I(TR_REG_M_BILINEAR, CONVERT, COUNT, TR__CPU),   /* multiplicities: signed updates, nonnegativity checked */
    /* 43 ENV-001 exact breakpoints (rational, or certified algebraic intervals) and lookups */
    TR__I(TR_REG_M_ENV, ORDER, BIGQ, TR__ALL), TR__I(TR_REG_M_ENV, EVAL, BIGQ, TR__ALL),
    TR__I(TR_REG_M_ENV, ROOTS, BIGQ, TR__CPU),
    /* 44 ATTR-001 sparse interaction and contribution attribution */
    TR__I(TR_REG_M_ATTR, EVAL, COUNT, TR__ALL), TR__I(TR_REG_M_ATTR, LOGIC, BOOLEAN, TR__ALL),
    TR__I(TR_REG_M_ATTR, COFACTOR, KLEENE, TR__CPU), TR__I(TR_REG_M_ATTR, ZETA, INT128, TR__CPU),
    TR__I(TR_REG_M_ATTR, MOBIUS, INT128, TR__CPU), TR__I(TR_REG_M_ATTR, MULTIPLY, INT128, TR__CPU),
    TR__I(TR_REG_M_ATTR, FOLD, MOD128, TR__ALL), TR__I(TR_REG_M_ATTR, FOLD, INT128, TR__ALL),
    TR__I(TR_REG_M_ATTR, ATTRIBUTE, BIGQ, TR__ALL), TR__I(TR_REG_M_ATTR, ORDER, INT128, TR__ALL),
    TR__I(TR_REG_M_ATTR, ORDER, BIGQ, TR__ALL),
    /* 45 COUNT-001 observed-domain constraint counts and witnesses */
    TR__I(TR_REG_M_COUNT, PREDICATE, INTEGER, TR__ALL), TR__I(TR_REG_M_COUNT, PREDICATE, RATIONAL, TR__ALL),
    TR__I(TR_REG_M_COUNT, LOGIC, KLEENE, TR__ALL), TR__I(TR_REG_M_COUNT, POPCOUNT, ROWSET, TR__ALL),
    TR__I(TR_REG_M_COUNT, FOLD, COUNT, TR__ALL), TR__I(TR_REG_M_COUNT, ORDER, COUNT, TR__ALL),
    /* 46 FIELD-002 parity relationship discovery and substitution */
    TR__I(TR_REG_M_PARITY, SOLVE, GF2, TR__CPU), TR__I(TR_REG_M_PARITY, BITS, CODE, TR__CPU),
    TR__I(TR_REG_M_PARITY, VERIFY, INTEGER, TR__ALL), TR__I(TR_REG_M_PARITY, POLY_SUBSTITUTE, GF2, TR__CPU),
    TR__I(TR_REG_M_PARITY, LOGIC, KLEENE, TR__ALL), TR__I(TR_REG_M_PARITY, POPCOUNT, ROWSET, TR__ALL),
    TR__I(TR_REG_M_PARITY, FOLD, COUNT, TR__ALL), TR__I(TR_REG_M_PARITY, ORDER, COUNT, TR__ALL),
    /* 47 UNC-001 intervals, affine forms, congruences, pruning */
    TR__I(TR_REG_M_UNC, ENCLOSE, QINTERVAL, TR__ALL), TR__I(TR_REG_M_UNC, ENCLOSE, INTERVAL, TR__ALL),
    TR__I(TR_REG_M_UNC, PREDICATE, RATIONAL, TR__ALL), TR__I(TR_REG_M_UNC, LOGIC, KLEENE, TR__ALL),
    TR__I(TR_REG_M_UNC, EVAL, RATIONAL, TR__ALL), TR__I(TR_REG_M_UNC, FOLD, COUNT, TR__ALL),
    TR__I(TR_REG_M_UNC, EUCLID, INTEGER, TR__CPU),
    /* 48 PRE-001 affine, square, residue, piecewise maps to codes */
    TR__I(TR_REG_M_PRE, DIVIDE, BIGQ, TR__CPU), TR__I(TR_REG_M_PRE, EUCLID, BIGINT, TR__CPU),
    TR__I(TR_REG_M_PRE, ORDER, CODE, TR__ALL), TR__I(TR_REG_M_PRE, EVAL, RATIONAL, TR__ALL),
    TR__I(TR_REG_M_PRE, PREDICATE, RATIONAL, TR__ALL), TR__I(TR_REG_M_PRE, FOLD, COUNT, TR__ALL),
    /* 49 DISC-002 Gram nullspaces, verified identities */
    TR__I(TR_REG_M_IDENT, FOLD, INT128, TR__ALL), TR__I(TR_REG_M_IDENT, FOLD, GFP, TR__ALL),
    TR__I(TR_REG_M_IDENT, MERGE, INT128, TR__CPU), TR__I(TR_REG_M_IDENT, SUBTRACT, INT128, TR__CPU),
    TR__I(TR_REG_M_IDENT, SOLVE, BIGQ, TR__CPU), TR__I(TR_REG_M_IDENT, CRT, GFP, TR__CPU),
    TR__I(TR_REG_M_IDENT, RATRECON, GFP, TR__CPU), TR__I(TR_REG_M_IDENT, EVAL, INTEGER, TR__ALL),
    TR__I(TR_REG_M_IDENT, VERIFY, INTEGER, TR__ALL), TR__I(TR_REG_M_IDENT, CONTRACT, BIGINT, TR__CPU),
    /* 50 GEOM-001 Groebner bases, reduction certificates */
    TR__I(TR_REG_M_GEOM, POLY_MUL, RATIONAL, TR__CPU), TR__I(TR_REG_M_GEOM, POLY_SUB, RATIONAL, TR__CPU),
    TR__I(TR_REG_M_GEOM, DIVIDE, RATIONAL, TR__CPU), TR__I(TR_REG_M_GEOM, POLY_DERIVATIVE, RATIONAL, TR__CPU),
    TR__I(TR_REG_M_GEOM, MULTIPLY, INT128, TR__ALL), TR__I(TR_REG_M_GEOM, FOLD, INT128, TR__ALL),
    TR__I(TR_REG_M_GEOM, EVAL, INTEGER, TR__ALL), TR__I(TR_REG_M_GEOM, VERIFY, INTEGER, TR__ALL),
    /* 51 TENSOR-001 sparse and factored moment tensors */
    TR__I(TR_REG_M_TENSOR_STATE, FOLD, INT128, TR__ALL), TR__I(TR_REG_M_TENSOR_STATE, FOLD, COUNT, TR__ALL),
    TR__I(TR_REG_M_TENSOR_STATE, POWERS, INTEGER, TR__ALL), TR__I(TR_REG_M_TENSOR_STATE, HISTOGRAM, CODE, TR__ALL),
    TR__I(TR_REG_M_TENSOR_STATE, MERGE, INT128, TR__CPU), TR__I(TR_REG_M_TENSOR_STATE, MULTIPLY, INT128, TR__CPU),
    /* 52 TENSOR-002 costed tensor contraction planning */
    TR__I(TR_REG_M_TENSOR_CONTRACT, CONTRACT, INT128, TR__ALL), TR__I(TR_REG_M_TENSOR_CONTRACT, MULTIPLY, INT128, TR__ALL),
    TR__I(TR_REG_M_TENSOR_CONTRACT, FOLD, INT128, TR__ALL),
    /* 53 KERNEL-001 exact finite-feature kernel embeddings */
    TR__I(TR_REG_M_KERNEL, FOLD, INT128, TR__ALL), TR__I(TR_REG_M_KERNEL, FOLD, BIGINT, TR__ALL),
    TR__I(TR_REG_M_KERNEL, POWERS, INTEGER, TR__ALL), TR__I(TR_REG_M_KERNEL, HISTOGRAM, CODE, TR__ALL),
    TR__I(TR_REG_M_KERNEL, CONTRACT, BIGINT, TR__ALL), TR__I(TR_REG_M_KERNEL, SUBTRACT, BIGINT, TR__ALL),
    TR__I(TR_REG_M_KERNEL, MULTIPLY, BIGINT, TR__ALL), TR__I(TR_REG_M_KERNEL, DIVIDE, BIGQ, TR__ALL),
    /* 54 COUNT-002 symbolic full-domain algebraic model counting */
    TR__I(TR_REG_M_SYMCOUNT, COMPILE, BOOLEAN, TR__CPU), TR__I(TR_REG_M_SYMCOUNT, LOGIC, KLEENE, TR__ALL),
    TR__I(TR_REG_M_SYMCOUNT, MODEL_COUNT, BIGINT, TR__ALL), TR__I(TR_REG_M_SYMCOUNT, MODEL_COUNT, BIGQ, TR__CPU),
    TR__I(TR_REG_M_SYMCOUNT, FOLD, COUNT, TR__ALL), TR__I(TR_REG_M_SYMCOUNT, MULTIPLY, BIGINT, TR__CPU),
    TR__I(TR_REG_M_SYMCOUNT, ORDER, BIGQ, TR__CPU),
    /* 55 PROV-002 explanation specialization and deletion propagation */
    TR__I(TR_REG_M_EXPLAIN, PROPAGATE, COUNT, TR__ALL), TR__I(TR_REG_M_EXPLAIN, PROPAGATE, BOOLEAN, TR__ALL),
    TR__I(TR_REG_M_EXPLAIN, EXPLAIN, BOOLEAN, TR__CPU), TR__I(TR_REG_M_EXPLAIN, PROBABILITY, BIGQ, TR__CPU),
    TR__I(TR_REG_M_EXPLAIN, ENCLOSE, INTERVAL, TR__CPU),
    /* 56 SPEC-001 fixed spectral basis states and phase actions */
    TR__I(TR_REG_M_FOURIER, ENCLOSE, QINTERVAL, TR__ALL), TR__I(TR_REG_M_FOURIER, FOLD, QINTERVAL, TR__ALL),
    TR__I(TR_REG_M_FOURIER, FOLD, INT128, TR__ALL), TR__I(TR_REG_M_FOURIER, FOLD, COUNT, TR__ALL),
    TR__I(TR_REG_M_FOURIER, DIFFERENTIATE, BIGQ, TR__CPU), TR__I(TR_REG_M_FOURIER, ENCLOSE, CINTERVAL, TR__CPU),
    TR__I(TR_REG_M_FOURIER, CONVERT, CINTERVAL, TR__CPU), TR__I(TR_REG_M_FOURIER, WALSH, INT128, TR__CPU),
    /* 57 LIN-003 leave-group-out and low-rank model updates */
    TR__I(TR_REG_M_LINUPD, FOLD, BIGINT, TR__CPU), TR__I(TR_REG_M_LINUPD, SUBTRACT, BIGINT, TR__CPU),
    TR__I(TR_REG_M_LINUPD, SOLVE, BIGQ, TR__CPU), TR__I(TR_REG_M_LINUPD, UPDATE, BIGQ, TR__CPU),
    TR__I(TR_REG_M_LINUPD, DIVIDE, BIGQ, TR__CPU), TR__I(TR_REG_M_LINUPD, RESIDUE, GFP, TR__ALL),
    TR__I(TR_REG_M_LINUPD, SOLVE, GFP, TR__ALL), TR__I(TR_REG_M_LINUPD, CRT, GFP, TR__CPU),
    TR__I(TR_REG_M_LINUPD, ENCLOSE, INTERVAL, TR__CPU),
    /* 58 PROB-001 small-support partition and likelihood functions */
    TR__I(TR_REG_M_PROB, FOLD, COUNT, TR__CPU), TR__I(TR_REG_M_PROB, MULTIPLY, COUNT, TR__CPU),
    TR__I(TR_REG_M_PROB, CONTRACT, INT128, TR__ALL), TR__I(TR_REG_M_PROB, ENCLOSE, QINTERVAL, TR__ALL),
    TR__I(TR_REG_M_PROB, NORMALIZE, QINTERVAL, TR__ALL), TR__I(TR_REG_M_PROB, NORMALIZE, BIGQ, TR__CPU),
    TR__I(TR_REG_M_PROB, DIVIDE, BIGQ, TR__CPU), TR__I(TR_REG_M_PROB, ENCLOSE, INTERVAL, TR__CPU),
    /* 59 SYM-001 closed-basis transformation actions */
    TR__I(TR_REG_M_SYM_ACTION, MULTIPLY, INT128, TR__ALL), TR__I(TR_REG_M_SYM_ACTION, CONTRACT, INT128, TR__ALL),
    TR__I(TR_REG_M_SYM_ACTION, DIVIDE, RATIONAL, TR__ALL), TR__I(TR_REG_M_SYM_ACTION, ENCLOSE, INTERVAL, TR__ALL),
    TR__I(TR_REG_M_SYM_ACTION, POWERS, INTEGER, TR__CPU), TR__I(TR_REG_M_SYM_ACTION, EUCLID, BIGINT, TR__CPU),
    /* 60 SYM-002 certified data symmetries and orbits */
    TR__I(TR_REG_M_SYM_SYMMETRY, VERIFY, INTEGER, TR__ALL), TR__I(TR_REG_M_SYM_SYMMETRY, FOLD, COUNT, TR__ALL),
    TR__I(TR_REG_M_SYM_SYMMETRY, EVAL, INTEGER, TR__CPU), TR__I(TR_REG_M_SYM_SYMMETRY, SCALE_COUNT, INT128, TR__CPU),
    TR__I(TR_REG_M_SYM_SYMMETRY, FOLD, INT128, TR__CPU),
    /* 61 UNC-002 certified series, low rank, rounding, composition */
    TR__I(TR_REG_M_APPROX, ENCLOSE, INTERVAL, TR__ALL), TR__I(TR_REG_M_APPROX, FOLD, INT128, TR__ALL),
    TR__I(TR_REG_M_APPROX, FOLD, COUNT, TR__ALL), TR__I(TR_REG_M_APPROX, ENCLOSE, QINTERVAL, TR__CPU),
    TR__I(TR_REG_M_APPROX, EVAL, BIGQ, TR__CPU), TR__I(TR_REG_M_APPROX, MULTIPLY, BIGQ, TR__CPU),
    TR__I(TR_REG_M_APPROX, SOLVE, BIGQ, TR__CPU), TR__I(TR_REG_M_APPROX, PREDICATE, RATIONAL, TR__CPU),
    /* 62 MULTI-002 coarse-to-fine bounds, localized rollups */
    TR__I(TR_REG_M_REFINE, ENCLOSE, QINTERVAL, TR__ALL), TR__I(TR_REG_M_REFINE, ROLLUP, INT128, TR__ALL),
    TR__I(TR_REG_M_REFINE, ROLLUP, COUNT, TR__ALL), TR__I(TR_REG_M_REFINE, PREDICATE, RATIONAL, TR__CPU),
    TR__I(TR_REG_M_REFINE, ORDER, BIGINT, TR__CPU), TR__I(TR_REG_M_REFINE, SUBTRACT, BIGINT, TR__CPU),
    /* 63 INV-002 minimal-cost changes attaining a target, algebraic only */
    TR__I(TR_REG_M_INTERV, MINIMIZE, BIGQ, TR__CPU), TR__I(TR_REG_M_INTERV, ROOTS, BIGQ, TR__CPU),
    TR__I(TR_REG_M_INTERV, EVAL, GFP, TR__ALL), TR__I(TR_REG_M_INTERV, CRT, GFP, TR__ALL),
    TR__I(TR_REG_M_INTERV, ORDER, INTEGER, TR__ALL), TR__I(TR_REG_M_INTERV, FOLD, MINMAX, TR__ALL),
    TR__I(TR_REG_M_INTERV, ENCLOSE, INTERVAL, TR__CPU), TR__I(TR_REG_M_INTERV, PREFIX, INT128, TR__CPU),
    /* 64 OPT-002 exact QPs and LPs, certified branch and bound */
    TR__I(TR_REG_M_POLYOPT, MINIMIZE, BIGQ, TR__CPU), TR__I(TR_REG_M_POLYOPT, SOLVE, BIGQ, TR__CPU),
    TR__I(TR_REG_M_POLYOPT, INERTIA, BIGQ, TR__CPU), TR__I(TR_REG_M_POLYOPT, ROOTS, BIGQ, TR__CPU),
    TR__I(TR_REG_M_POLYOPT, EVAL, BIGQ, TR__CPU), TR__I(TR_REG_M_POLYOPT, ORDER, BIGQ, TR__CPU),
    TR__I(TR_REG_M_POLYOPT, ENCLOSE, INTERVAL, TR__ALL),
    /* 65 GRAPH-001 graph layouts, degrees, frontiers and semiring products */
    TR__I(TR_REG_M_GRAPH_LAYOUT, ORDER, INTEGER, TR__CPU), TR__I(TR_REG_M_GRAPH_LAYOUT, FOLD, COUNT, TR__ALL),
    TR__I(TR_REG_M_GRAPH_LAYOUT, MULTIPLY, INT128, TR__ALL), TR__I(TR_REG_M_GRAPH_LAYOUT, LOGIC, BOOLEAN, TR__ALL),
    TR__I(TR_REG_M_GRAPH_LAYOUT, MULTIPLY, GF2, TR__ALL), TR__I(TR_REG_M_GRAPH_LAYOUT, DISTINCT_FOLD, MINMAX, TR__ALL),
    TR__I(TR_REG_M_GRAPH_LAYOUT, ENCLOSE, INTERVAL, TR__ALL), TR__I(TR_REG_M_GRAPH_LAYOUT, POPCOUNT, ROWSET, TR__ALL),
    /* 66 GRAPH-002 reachability, bounded walk counts, paths with cycle refusal, energies */
    TR__I(TR_REG_M_GRAPH_PATHS, FIXPOINT, BOOLEAN, TR__ALL), TR__I(TR_REG_M_GRAPH_PATHS, FIXPOINT, ROWSET, TR__ALL),
    TR__I(TR_REG_M_GRAPH_PATHS, MULTIPLY, BIGINT, TR__ALL), TR__I(TR_REG_M_GRAPH_PATHS, ORDER, BIGINT, TR__ALL),
    TR__I(TR_REG_M_GRAPH_PATHS, RELAX, MINMAX, TR__ALL), TR__I(TR_REG_M_GRAPH_PATHS, FOLD, BIGINT, TR__ALL),
    TR__I(TR_REG_M_GRAPH_PATHS, ENCLOSE, INTERVAL, TR__CPU),
    /* 67 GEOM-002 generator sets, reconstructions, local forms */
    TR__I(TR_REG_M_ELIM, DIVIDE, RATIONAL, TR__CPU), TR__I(TR_REG_M_ELIM, POLY_SUB, RATIONAL, TR__CPU),
    TR__I(TR_REG_M_ELIM, POLY_SUBSTITUTE, RATIONAL, TR__CPU), TR__I(TR_REG_M_ELIM, EVAL, INTEGER, TR__ALL),
    TR__I(TR_REG_M_ELIM, VERIFY, INTEGER, TR__ALL), TR__I(TR_REG_M_ELIM, ORDER, INTEGER, TR__ALL),
    TR__I(TR_REG_M_ELIM, FOLD, COUNT, TR__ALL),
    /* 68 PRE-002 roots, residues, bit cubes, fallbacks */
    TR__I(TR_REG_M_PREIMG, ROOTS, BIGQ, TR__CPU), TR__I(TR_REG_M_PREIMG, EVAL, BIGINT, TR__CPU),
    TR__I(TR_REG_M_PREIMG, EUCLID, INTEGER, TR__CPU), TR__I(TR_REG_M_PREIMG, RESIDUE, INTEGER, TR__ALL),
    TR__I(TR_REG_M_PREIMG, BITS, CODE, TR__CPU), TR__I(TR_REG_M_PREIMG, ORDER, CODE, TR__ALL),
    TR__I(TR_REG_M_PREIMG, PREDICATE, RATIONAL, TR__ALL), TR__I(TR_REG_M_PREIMG, FOLD, COUNT, TR__ALL),
    /* 69 TOPO-001 declared occupancy complexes and local topology */
    TR__I(TR_REG_M_TOPO_COMPLEX, CONVERT, INTEGER, TR__ALL), TR__I(TR_REG_M_TOPO_COMPLEX, MARK_CELLS, CODE, TR__ALL),
    TR__I(TR_REG_M_TOPO_COMPLEX, DISTINCT_FOLD, ROWSET, TR__ALL), TR__I(TR_REG_M_TOPO_COMPLEX, POPCOUNT, ROWSET, TR__ALL),
    TR__I(TR_REG_M_TOPO_COMPLEX, FOLD, COUNT, TR__ALL), TR__I(TR_REG_M_TOPO_COMPLEX, ORDER, COUNT, TR__ALL),
    TR__I(TR_REG_M_TOPO_COMPLEX, BOUNDARY, INTEGER, TR__ALL), TR__I(TR_REG_M_TOPO_COMPLEX, ORDER, INTEGER, TR__ALL),
    /* 70 TOPO-002 bounded cycles, homology and persistence */
    TR__I(TR_REG_M_TOPO_HOMOLOGY, BOUNDARY, INTEGER, TR__ALL), TR__I(TR_REG_M_TOPO_HOMOLOGY, BOUNDARY, GF2, TR__ALL),
    TR__I(TR_REG_M_TOPO_HOMOLOGY, VERIFY, INTEGER, TR__ALL), TR__I(TR_REG_M_TOPO_HOMOLOGY, FOLD, COUNT, TR__ALL),
    TR__I(TR_REG_M_TOPO_HOMOLOGY, RESIDUE, GFP, TR__CPU), TR__I(TR_REG_M_TOPO_HOMOLOGY, SOLVE, GF2, TR__CPU),
    TR__I(TR_REG_M_TOPO_HOMOLOGY, HOMOLOGY, GF2, TR__CPU), TR__I(TR_REG_M_TOPO_HOMOLOGY, HOMOLOGY, GFP, TR__CPU),
    TR__I(TR_REG_M_TOPO_HOMOLOGY, HOMOLOGY, RATIONAL, TR__CPU), TR__I(TR_REG_M_TOPO_HOMOLOGY, PERSISTENCE, GF2, TR__CPU),
    TR__I(TR_REG_M_TOPO_HOMOLOGY, PERSISTENCE, GFP, TR__CPU), TR__I(TR_REG_M_TOPO_HOMOLOGY, PERSISTENCE, RATIONAL, TR__CPU),
    TR__I(TR_REG_M_TOPO_HOMOLOGY, ORDER, INTEGER, TR__CPU),
};
#undef TR__I
#undef TR__CPU
#undef TR__ALL
#undef TR__C
#define TR_REG_IMPL_ROWS ((int32_t)(sizeof tr_reg_impls / sizeof tr_reg_impls[0]))
/* the first module implementing op in carrier on one of the devices (its id), or 0 */
static inline int32_t tr_reg_find_impl(int32_t op, int32_t carrier, int32_t devices) {
    for (int32_t i = 0; i < TR_REG_IMPL_ROWS; i++)
        if (tr_reg_impls[i].op == op && tr_reg_impls[i].carrier == carrier && (tr_reg_impls[i].devices & devices))
            return tr_reg_impls[i].module;
    return 0;
}

/* ------------------------------------------------------------------ transforms through the registry */
enum { TR_REG_UP = 0, TR_REG_DOWN = 1 };   /* zeta / Moebius over supersets (UP) or subsets (DOWN); Walsh: UP */
#define TR_REG_TRANSFORM_MAX_K 30            /* = TR_LATTICE_MAX_AXES = TR_SPEC_MAX_K (math_registry.c) */
/* The checks a transform makes before anything is read (both libraries): op / variant / k, the composition
 * [(op, carrier)] from the values' carrier `input` (laws, conversions), the implementing module on `device`
 * (tr_reg_impls; *module gets its id) and the carriers the registry's entry carries (count, integer = checked int64,
 * int128; a big-integer transform takes FND-009 handles through math_lattice's own API). */
static inline int32_t tr_reg_transform_check(int32_t op, int32_t variant, int32_t carrier, int32_t input, int32_t k,
                                             const tr_math_i128 *in, const tr_math_i128 *out, int32_t device,
                                             int32_t *module, tr_reg_explain *ex) {
    *module = 0;
    if ((op != TR_OP_ZETA && op != TR_OP_MOBIUS && op != TR_OP_WALSH) || (variant != TR_REG_UP && variant != TR_REG_DOWN) ||
        (op == TR_OP_WALSH && variant != TR_REG_UP) || k < 0 || !in || !out)
        return tr_reg_explain_refuse(ex, TR_REG_REFUSE_INVALID, TR_REG_E_INVALID);
    tr_reg_step step = {op, carrier};
    tr_reg_refusal law;
    int32_t rc = tr_reg_check_laws(&step, 1, input, TR_REG_CONTRACT_EXACT, 0, &law);
    if (rc) { if (ex) ex->refusal = law; return tr_reg_explain_refuse(ex, law.kind, rc); }
    *module = tr_reg_find_impl(op, carrier, device);
    if (ex) ex->module = *module;
    if (!*module) {
        if (ex) tr_reg_refuse(&ex->refusal, TR_REG_REFUSE_IMPLEMENTATION, TR_REG_E_CARRIER, 0, op, carrier, 0, 0, 0);
        return tr_reg_explain_refuse(ex, TR_REG_REFUSE_IMPLEMENTATION, TR_REG_E_CARRIER);
    }
    if (carrier != TR_REG_C_COUNT && carrier != TR_REG_C_INTEGER && carrier != TR_REG_C_INT128)
        return tr_reg_explain_refuse(ex, TR_REG_REFUSE_UNSUPPORTED, TR_REG_E_UNSUPPORTED);
    return 0;
}
/* whether a 128-bit value is an element of a carrier the transform's values may be declared in */
static inline int tr_reg_value_in(int32_t carrier, __int128 v) {
    switch (carrier) {
    case TR_REG_C_COUNT: return v >= 0 && v <= (__int128)UINT64_MAX;
    case TR_REG_C_INTEGER: return v >= (__int128)INT64_MIN && v <= (__int128)INT64_MAX;
    case TR_REG_C_BOOLEAN: case TR_REG_C_GF2: return v == 0 || v == 1;
    case TR_REG_C_CODE: return v >= 0 && v <= (__int128)UINT32_MAX;
    default: return 1;                                        /* int128 (and big integers given in 128 bits) */
    }
}
/* After the module's budget check (k <= TR_REG_TRANSFORM_MAX_K), before the module runs: every value an element of
 * its declared carrier `input` (else invalid input), converted to the step's carrier (a checked narrowing: else
 * TR_MATH_OVERFLOW, refusal kind OVERFLOW naming input -> carrier), and for the count carrier the exact bound: every
 * zeta output is a sum of inputs, the largest being their total, so total <= 2^64 - 1 is proven before anything
 * runs (else TR_MATH_OVERFLOW, refusal kind BOUND). A module's own TR_MATH_OVERFLOW is then an intermediate or an
 * output leaving the step's carrier. */
static inline int32_t tr_reg_transform_values(int32_t op, int32_t carrier, int32_t input, int32_t k,
                                              const tr_math_i128 *in, tr_reg_explain *ex) {
    const uint64_t n = (uint64_t)1 << k;
    tr_u128 total = 0;
    for (uint64_t i = 0; i < n; i++) {
        __int128 v = tr_i128_get(in[i]);
        if (!tr_reg_value_in(input, v)) return tr_reg_explain_refuse(ex, TR_REG_REFUSE_INVALID, TR_REG_E_INVALID);
        if (!tr_reg_value_in(carrier, v)) {
            if (ex) tr_reg_refuse(&ex->refusal, TR_REG_REFUSE_OVERFLOW, TR_REG_E_OVERFLOW, 0, op, carrier, 0, input, carrier);
            return tr_reg_explain_refuse(ex, TR_REG_REFUSE_OVERFLOW, TR_REG_E_OVERFLOW);
        }
        if (carrier == TR_REG_C_COUNT) total += (tr_u128)v;   /* < 2^30 x 2^64: exact */
    }
    if (carrier == TR_REG_C_COUNT && total > (tr_u128)UINT64_MAX) {
        if (ex) tr_reg_refuse(&ex->refusal, TR_REG_REFUSE_BOUND, TR_REG_E_OVERFLOW, 0, op, TR_REG_C_COUNT, 0, 0, 0);
        return tr_reg_explain_refuse(ex, TR_REG_REFUSE_BOUND, TR_REG_E_OVERFLOW);
    }
    return 0;
}

/* ------------------------------------------------------------------ exported (math_registry.c) */
int32_t tr_reg_module_count(void);
int32_t tr_reg_module_get(int32_t index, tr_reg_module *);          /* index 0 .. count-1, in id order */
int32_t tr_reg_module_by_id(int32_t id, tr_reg_module *);
int32_t tr_reg_chain_get(int32_t representation, int32_t slot, tr_reg_chain *);
int32_t tr_reg_carrier_info(int32_t carrier, uint32_t *laws, char *name, int32_t capacity);
int32_t tr_reg_operator_info(int32_t op, tr_reg_operator *, char *name, int32_t capacity);
int32_t tr_reg_conversion_get(int32_t index, tr_reg_conversion *);
int32_t tr_reg_law_name(int32_t bit_index, char *name, int32_t capacity);
/* the names of the other enumerations: TR_REG_NAMES_* table, index (a kind, or a bit index for proofs and requests) */
enum { TR_REG_NAMES_STATES = 1, TR_REG_NAMES_PROOFS = 2, TR_REG_NAMES_REQUESTS = 3, TR_REG_NAMES_CONVERSIONS = 4,
       TR_REG_NAMES_COST_BASES = 5, TR_REG_NAMES_REFUSALS = 6 };
int32_t tr_reg_name(int32_t table, int32_t index, char *name, int32_t capacity);
/* the implementation rows (index 0 .. tr_reg_impl_count() - 1) */
int32_t tr_reg_impl_count(void);
int32_t tr_reg_impl_get(int32_t index, tr_reg_impl *);
/* the module implementing op in carrier on one of the devices (its id), or 0 */
int32_t tr_reg_find(int32_t op, int32_t carrier, int32_t devices);
/* the composition check (laws and conversions); with require_impl also that some module implements every step */
int32_t tr_reg_check_chain(const tr_reg_step *, int32_t n, int32_t input, int32_t contract, int32_t require_impl,
                           int32_t *output, tr_reg_refusal *);
/* the code domain of raw codes: ncodes = max + 1 (1 when rows == 0), width = bits of max */
int32_t tr_reg_code_domain(const uint32_t *codes, uint64_t rows, uint64_t *ncodes, int32_t *width);

struct atable_s;
typedef struct tr_reg_column_s tr_reg_column;
/* the decoded codes (and validity flags) of one engine column (codes <= 32 bits): rows x 4 bytes each, budgeted
 * (bytes) before allocation; NULL with *error on failure. valid_col -1: no validity column. */
tr_reg_column *tr_reg_column_create(const struct atable_s *, int32_t col, int32_t valid_col, int64_t offset,
                                    const tr_math_budget *, tr_math_refusal *, int32_t *error);
/* fills the request's column fields (rows, codes, validity, offset, ncodes, width, source, revision); TR_MATH_STALE
 * when the table's revision or row count changed since the column was decoded */
int32_t tr_reg_column_request(const tr_reg_column *, const struct atable_s *, tr_reg_request *);
uint64_t tr_reg_column_bytes(const tr_reg_column *);
void tr_reg_column_free(tr_reg_column *);

typedef struct tr_reg_state_s tr_reg_state;
/* Builds the state of HISTOGRAM, POWER (support: exponent mask, 0 = the request's) or BITPLANE (D: plane-set degree,
 * 0 = the request's) for the request's scope on the CPU: immutable and owned, independent of the request's buffers. */
tr_reg_state *tr_reg_state_build(const tr_reg_request *, int32_t representation, uint32_t support, int32_t D,
                                 const tr_math_budget *, tr_reg_explain *, int32_t *error);
/* A state built on a CUDA device (tr_mgpu_registry_state): validated against the request and copied */
tr_reg_state *tr_reg_state_import(const tr_reg_request *, const tr_reg_state_info *, const void *raw, uint64_t bytes,
                                  int32_t *error);
int32_t tr_reg_state_get_info(const tr_reg_state *, tr_reg_state_info *);
/* the raw bytes of a state (*bytes), after its build's budget check on the device (TR_REG_ON_CPU or _CUDA) passed:
 * a caller sizes the buffer of tr_mgpu_registry_state with it, so nothing is allocated before the check */
int32_t tr_reg_state_size(const tr_reg_request *, int32_t representation, uint32_t support, int32_t D, int32_t device,
                          const tr_math_budget *, uint64_t *bytes, tr_math_refusal *);
/* the raw arrays: HISTOGRAM ncodes counts, POWER (lo, hi) residues per kept exponent, BITPLANE counts per plane set */
int32_t tr_reg_state_raw(const tr_reg_state *, void *out, uint64_t capacity);
/* 1 when the state covers exactly the request's scope (source, revision, rows, code domain, validity, selection), 0
 * when not, < 0 invalid */
int32_t tr_reg_state_matches(const tr_reg_state *, const tr_reg_request *);
void tr_reg_state_free(tr_reg_state *);
/* executes a representation on the CPU (reuse: a matching state, contracted without a scan); explanation always */
int32_t tr_reg_execute(const tr_reg_request *, int32_t representation, const tr_reg_state *reuse,
                       const tr_math_budget *, tr_reg_result *, tr_reg_explain *);
/* the candidates of a request (bounded: at most TR_REG_MAX_CANDIDATES; budget iterations = candidates searched), in
 * the order EMPTY, SCAN, HISTOGRAM, POWER, BITPLANE on the CPU, the reuse of `cached`, then SCAN, HISTOGRAM (GPU and
 * CPU finalization), POWER, BITPLANE on CUDA. A candidate is valid when its chains pass the composition check and
 * its proofs hold: EMPTY empty_proven (an exact zero count), SCAN / HISTOGRAM tr_reg_eval_bound_ok, POWER / BITPLANE
 * tr_reg_bound_ok(rows_bound), a reuse a matching and sufficient state. estimate: selected rows the amounts assume;
 * rows_bound: a proven upper bound of the selected rows; cached: a state offered for reuse, or NULL. */
int32_t tr_reg_candidates(const tr_reg_request *, const tr_reg_state *cached, int32_t devices, uint64_t estimate,
                          uint64_t rows_bound, int32_t empty_proven, const tr_math_budget *, tr_reg_candidate *out,
                          int32_t capacity, int32_t *count, tr_math_refusal *);

/* DISC-001 verification through the registry: tr_disc_verify with an explanation (module TR_REG_R_VERIFY) */
int32_t tr_reg_verify(const tr_disc_view *, int64_t scale, int64_t bias, tr_disc_evidence *, tr_reg_explain *);
/* its candidates: the CPU verifier and the CUDA kernel (device copies: tr_disc_bytes) */
int32_t tr_reg_candidates_verify(const tr_disc_view *, int32_t devices, const tr_math_budget *, tr_reg_candidate *out,
                                 int32_t capacity, int32_t *count, tr_math_refusal *);
/* A transform of 2^k values through the registry on the CPU (see the header comment): op TR_OP_ZETA, TR_OP_MOBIUS
 * (variant TR_REG_UP: over supersets, TR_REG_DOWN: over subsets) or TR_OP_WALSH (TR_REG_UP), carrier the step's
 * (count, integer = checked int64, int128), input the values' carrier; the module's budget (variables k, tensor 2^k,
 * iterations k, bytes 16 2^k) is checked before anything is allocated. out is written only on success. The
 * explanation names the module (LATTICE-002 or SPEC-002), the entry point that ran and any refusal. The same on a
 * CUDA device: tr_mgpu_registry_transform (math_registry_gpu.cu). */
int32_t tr_reg_transform(int32_t op, int32_t variant, int32_t carrier, int32_t input, int32_t k, const tr_math_i128 *in,
                         tr_math_i128 *out, const tr_math_budget *, tr_math_refusal *, tr_reg_explain *);
#ifdef __cplusplus
}
#endif
#endif
