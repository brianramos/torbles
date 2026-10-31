/* FND-005: the native module and operator registry of the mathematical capabilities.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Carriers and laws. A carrier is the algebraic structure a value lives in (checked int64 integers, exact 128-bit
 * sums, residues modulo 2^128, canonical rationals, multiplicities, min/max lattices, stored codes, row sets, ...).
 * tr_reg_carrier_laws states the laws each carrier's operations satisfy as implemented (a result that is returned is
 * exact; an overflow is refused, never wrapped). tr_reg_conversions lists the declared carrier changes: EXACT (always
 * lossless), CHECKED (lossless or refused: a narrowing, or a residue read back under a proven bound) and ENCLOSING (a
 * rigorous enclosure, not lossless). Every operator (tr_reg_operators) states the laws it requires of the carrier it
 * runs in, and some run only in a fixed input carrier.
 *
 * A composition is a chain of (operator, carrier) steps over a value that starts in an input carrier.
 * tr_reg_check_chain accepts it only when, at every step: the operator exists; its fixed input carrier, if it has
 * one, is the step's carrier; the value's carrier equals the step's carrier or one declared lossless conversion
 * (EXACT or CHECKED; ENCLOSING too under the enclosure contract) leads there; and the step's carrier has every law
 * the operator requires (plus EXACT under the exact contract, EXACT or ENCLOSURE under the enclosure contract). The
 * value then takes the operator's output carrier. Otherwise it refuses at the first failing step, naming the missing
 * law (the lowest missing bit) or the missing conversion (from, to), with TR_MATH_CARRIER_ERROR (-7), or
 * TR_MATH_OPERATOR_ERROR (-8) for an unknown operator.
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
 * Budgets (math_budgets.h) are checked before anything is allocated (tr_reg_need): degree and support of the
 * polynomials, states (histogram cells, power sums, plane sets), circuit (BITPLANE: Moebius terms sum_S 2^|S|),
 * output (the polynomials) and bytes (state, scratch and, on CUDA, the device copies). Results, explanations, states
 * and candidates are plain structs of fixed layout (sizes checked in math_registry.c). */
#ifndef TORBLES_MATH_REGISTRY_H
#define TORBLES_MATH_REGISTRY_H
#include <stdint.h>
#include "math_num.h"
#include "math_budgets.h"
#include "math_disc.h"
#ifdef __cplusplus
extern "C" {
#endif

#define TR_REG_ABI 1
/* native codes (mathematics.h, math_domain.h, math_num.h): the CUDA library does not include the first slice */
#define TR_REG_E_INVALID (-1)
#define TR_REG_E_MEMORY (-2)
#define TR_REG_E_OVERFLOW (-3)
#define TR_REG_E_BUDGET (-5)
#define TR_REG_E_CARRIER (-7)
#define TR_REG_E_OPERATOR (-8)

/* ------------------------------------------------------------------ carriers, laws, conversions */
enum { TR_REG_C_ANY = 0,                                             /* operator input: any carrier */
       TR_REG_C_INTEGER = 1, TR_REG_C_RATIONAL = 2, TR_REG_C_BOOLEAN = 3, TR_REG_C_GF2 = 4,   /* = math_domain.h */
       TR_REG_C_COUNT = 5,     /* multiplicities: natural numbers (uint64, overflow refused) */
       TR_REG_C_INT128 = 6,    /* exact signed 128-bit sums */
       TR_REG_C_MOD128 = 7,    /* residues modulo 2^128 (exact ring, no order) */
       TR_REG_C_INTERVAL = 8,  /* outward-rounded double intervals (math_num.h) */
       TR_REG_C_FLOAT64 = 9,   /* IEEE doubles */
       TR_REG_C_MINMAX = 10,   /* int64 with min (or max) as the fold and integer multiplication */
       TR_REG_C_CODE = 11,     /* stored codes of an order-preserving codec */
       TR_REG_C_ROWSET = 12,   /* row bitmaps: union, intersection */
       TR_REG_CARRIERS = 13 };
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
       TR_LAW_ENCLOSURE = 1 << 11 };     /* results are rigorous enclosures of the exact real value */
#define TR_REG_LAWS 12
#define TR_REG_RING (TR_LAW_ASSOCIATIVE | TR_LAW_COMMUTATIVE | TR_LAW_IDENTITY | TR_LAW_DISTRIBUTIVE)
static const uint32_t tr_reg_carrier_laws[TR_REG_CARRIERS] = {
    0u,
    TR_REG_RING | TR_LAW_INVERSE | TR_LAW_ORDERED | TR_LAW_CHAR0 | TR_LAW_EXACT,                          /* integer */
    TR_REG_RING | TR_LAW_INVERSE | TR_LAW_ORDERED | TR_LAW_DIVISION | TR_LAW_CHAR0 | TR_LAW_EXACT,        /* rational */
    TR_REG_RING | TR_LAW_IDEMPOTENT | TR_LAW_MUL_IDEMPOTENT | TR_LAW_ORDERED | TR_LAW_DIVISION | TR_LAW_EXACT,  /* boolean */
    TR_REG_RING | TR_LAW_INVERSE | TR_LAW_MUL_IDEMPOTENT | TR_LAW_DIVISION | TR_LAW_EXACT,               /* gf2 */
    TR_REG_RING | TR_LAW_ORDERED | TR_LAW_CHAR0 | TR_LAW_EXACT,                                           /* count */
    TR_REG_RING | TR_LAW_INVERSE | TR_LAW_ORDERED | TR_LAW_CHAR0 | TR_LAW_EXACT,                          /* int128 */
    TR_REG_RING | TR_LAW_INVERSE | TR_LAW_EXACT,                                                          /* mod128 */
    TR_LAW_COMMUTATIVE | TR_LAW_CHAR0 | TR_LAW_ENCLOSURE,                                                 /* interval */
    TR_LAW_COMMUTATIVE | TR_LAW_IDENTITY,                                                                 /* float64 */
    TR_LAW_ASSOCIATIVE | TR_LAW_COMMUTATIVE | TR_LAW_IDENTITY | TR_LAW_IDEMPOTENT | TR_LAW_ORDERED | TR_LAW_EXACT,  /* minmax */
    TR_LAW_ORDERED | TR_LAW_EXACT,                                                                        /* code */
    TR_REG_RING | TR_LAW_IDEMPOTENT | TR_LAW_MUL_IDEMPOTENT | TR_LAW_EXACT,                               /* rowset */
};
enum { TR_CONV_EXACT = 1, TR_CONV_CHECKED = 2, TR_CONV_ENCLOSING = 3 };
typedef struct { int32_t from, to, kind; } tr_reg_conversion;
#define TR_REG_CONVERSIONS 17
static const tr_reg_conversion tr_reg_conversions[TR_REG_CONVERSIONS] = {
    {TR_REG_C_CODE, TR_REG_C_INTEGER, TR_CONV_CHECKED},        /* logical = code + offset, refused outside int64 */
    {TR_REG_C_INTEGER, TR_REG_C_INT128, TR_CONV_EXACT},
    {TR_REG_C_INTEGER, TR_REG_C_RATIONAL, TR_CONV_EXACT},      /* n / 1 */
    {TR_REG_C_INTEGER, TR_REG_C_MOD128, TR_CONV_EXACT},        /* the residue (a ring homomorphism) */
    {TR_REG_C_INTEGER, TR_REG_C_MINMAX, TR_CONV_EXACT},
    {TR_REG_C_INTEGER, TR_REG_C_INTERVAL, TR_CONV_ENCLOSING},
    {TR_REG_C_INT128, TR_REG_C_INTEGER, TR_CONV_CHECKED},      /* narrowing */
    {TR_REG_C_INT128, TR_REG_C_RATIONAL, TR_CONV_CHECKED},
    {TR_REG_C_INT128, TR_REG_C_MOD128, TR_CONV_EXACT},
    {TR_REG_C_MOD128, TR_REG_C_INT128, TR_CONV_CHECKED},       /* only under a proven bound |v| < 2^127 */
    {TR_REG_C_COUNT, TR_REG_C_INTEGER, TR_CONV_CHECKED},
    {TR_REG_C_COUNT, TR_REG_C_INT128, TR_CONV_EXACT},
    {TR_REG_C_COUNT, TR_REG_C_MOD128, TR_CONV_EXACT},
    {TR_REG_C_BOOLEAN, TR_REG_C_COUNT, TR_CONV_EXACT},         /* the indicator 0 / 1 */
    {TR_REG_C_BOOLEAN, TR_REG_C_GF2, TR_CONV_EXACT},           /* the same two elements (each step's laws are checked) */
    {TR_REG_C_GF2, TR_REG_C_BOOLEAN, TR_CONV_EXACT},
    {TR_REG_C_RATIONAL, TR_REG_C_INTERVAL, TR_CONV_ENCLOSING},
};
/* the declared conversion from -> to (its kind), 0 when none is declared */
static inline int32_t tr_reg_conversion_kind(int32_t from, int32_t to) {
    for (int i = 0; i < TR_REG_CONVERSIONS; i++)
        if (tr_reg_conversions[i].from == from && tr_reg_conversions[i].to == to) return tr_reg_conversions[i].kind;
    return 0;
}

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
       TR_REG_OPERATORS = 28 };
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
};

/* ------------------------------------------------------------------ the composition check */
enum { TR_REG_CONTRACT_EXACT = 0, TR_REG_CONTRACT_ENCLOSURE = 1 };
enum { TR_REG_REFUSE_NONE = 0, TR_REG_REFUSE_LAW = 1, TR_REG_REFUSE_CONVERSION = 2, TR_REG_REFUSE_OPERATOR = 3,
       TR_REG_REFUSE_CARRIER = 4,          /* the operator runs only in another carrier */
       TR_REG_REFUSE_IMPLEMENTATION = 5,   /* no module implements the operator in that carrier */
       TR_REG_REFUSE_BUDGET = 6, TR_REG_REFUSE_OVERFLOW = 7,
       TR_REG_REFUSE_BOUND = 8,            /* |result| < 2^127 not proven for a mod 2^128 representation */
       TR_REG_REFUSE_STALE = 9,            /* a state's scope, source or revision differs */
       TR_REG_REFUSE_PROOF = 10,           /* a required proof is absent (emptiness not proven) */
       TR_REG_REFUSE_DEVICE = 11, TR_REG_REFUSE_INVALID = 12, TR_REG_REFUSE_UNSUPPORTED = 13,
       TR_REG_REFUSE_MEMORY = 14 };
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
        if (cur != c) {
            int32_t kind = tr_reg_conversion_kind(cur, c);
            if (!kind || (kind == TR_CONV_ENCLOSING && contract == TR_REG_CONTRACT_EXACT))
                return tr_reg_refuse(r, TR_REG_REFUSE_CONVERSION, TR_REG_E_CARRIER, i, op, c, 0, cur, c);
        }
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
#define TR_REG_MODULES TR_REG_REPRESENTATIONS      /* module id == representation id, 1 .. 9 */
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
enum { TR_REG_FINAL_CPU = 0, TR_REG_FINAL_GPU = 1 };                          /* where a result is finalized */
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
/* what ran (status 0) or why nothing ran (status 1). 136 bytes. */
typedef struct {
    int32_t abi, status, module, representation;
    int32_t device, finalize, reused, code;     /* device TR_REG_DEV_*; reused: a cached state was contracted */
    tr_reg_refusal refusal;                     /* why refused (kind, step, law, conversion) */
    tr_math_refusal budget;                     /* a budget refusal: resource, need, limit */
    uint64_t rows_scanned, words_scanned, state_elements, state_bytes, device_bytes, native_calls;
} tr_reg_explain;
/* a built state (see tr_reg_state_build): 104 bytes */
typedef struct {
    int32_t representation, degree, width, has_validity;   /* degree: BITPLANE D (POWER: highest exponent) */
    uint32_t support, has_selection;                        /* support: POWER exponent mask (bit k: S_k kept) */
    uint64_t elements, bytes, rows, ncodes, selected, valid, nulls, source, revision;
    int64_t offset;
} tr_reg_state_info;

static inline void tr_reg_explain_init(tr_reg_explain *e, int32_t repr, int32_t device, int32_t finalize) {
    if (!e) return;
    e->abi = TR_REG_ABI; e->status = 0; e->module = repr; e->representation = repr;
    e->device = device; e->finalize = finalize; e->reused = 0; e->code = 0;
    tr_reg_refusal_none(&e->refusal); tr_budget_none(&e->budget);
    e->rows_scanned = e->words_scanned = e->state_elements = e->state_bytes = e->device_bytes = e->native_calls = 0;
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
       TR_REG_REQ_POLYSUM = 16 };
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
#define TR_REG_IMPLS 12
typedef struct { int32_t op, carrier; } tr_reg_impl;     /* an operator a module implements, in that carrier */
enum { TR_REG_STATE_NONE = 0, TR_REG_STATE_FORMAL = 1, TR_REG_STATE_MOMENT_BANK = 2, TR_REG_STATE_CELL_BITMAP = 3,
       TR_REG_STATE_EVIDENCE = 4, TR_REG_STATE_HISTOGRAM = 5, TR_REG_STATE_POWER_SUMS = 6, TR_REG_STATE_PLANE_COUNTS = 7,
       TR_REG_STATE_EMPTY_PROOF = 8 };
enum { TR_PROOF_DOMAIN = 1,        /* a typed domain / formal context contract (FND-001) */
       TR_PROOF_SNAPSHOT = 2,      /* the state's source identity and revision */
       TR_PROOF_SCOPE = 4,         /* the exact selected rows of a reused state */
       TR_PROOF_BOUND = 8,         /* a magnitude bound proven before execution */
       TR_PROOF_EXACT_COUNT = 16,  /* an exact selected-row count */
       TR_PROOF_CATALOG = 32,      /* a relationship catalog's snapshot (revision, rows, used rows) */
       TR_PROOF_CELL_DOMAIN = 64 };/* every id inside the declared domain */
/* 528 bytes. cost[device][component]: relative work units per unit of the component's amount, NOT time
 * (device 0 the CPU, 1 a CUDA device; host-side components of a CUDA execution use the host's rate). */
typedef struct {
    int32_t abi, id, representation, state_kind;
    int32_t requests, devices, nimpl, finalize;          /* requests TR_REG_REQ_*; devices TR_REG_ON_*; finalize bits */
    uint32_t carriers, laws_relied, laws_guaranteed, proofs;
    uint64_t operators;                                  /* bit op: implemented */
    tr_reg_impl impl[TR_REG_IMPLS];
    double cost[2][TR_COST_COMPONENTS];
    char name[40], capability[16];
    char cpu_entry[2][32], cuda_entry[2][32];
} tr_reg_module;

/* ------------------------------------------------------------------ exported (math_registry.c) */
int32_t tr_reg_module_count(void);
int32_t tr_reg_module_get(int32_t index, tr_reg_module *);          /* index 0 .. count-1, in id order */
int32_t tr_reg_module_by_id(int32_t id, tr_reg_module *);
int32_t tr_reg_chain_get(int32_t representation, int32_t slot, tr_reg_chain *);
int32_t tr_reg_carrier_info(int32_t carrier, uint32_t *laws, char *name, int32_t capacity);
int32_t tr_reg_operator_info(int32_t op, tr_reg_operator *, char *name, int32_t capacity);
int32_t tr_reg_conversion_get(int32_t index, tr_reg_conversion *);
int32_t tr_reg_law_name(int32_t bit_index, char *name, int32_t capacity);
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
#ifdef __cplusplus
}
#endif
#endif
