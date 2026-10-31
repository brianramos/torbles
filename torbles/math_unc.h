/* UNC-001: intervals, affine forms and congruence reasoning (certified abstract states).
 * SPDX-License-Identifier: Apache-2.0
 *
 * An abstract state describes, per group of a scope (FND-004) and per column, the set of values the group's selected
 * rows can hold: the selected rows, the present (non-NULL) rows, the minimum and maximum logical value and the gcd of
 * the differences between present values (tr_unc_input). A program (tr_unc_program: nodes in topological order) is an
 * expression over the columns; evaluating it abstractly propagates SOUND facts through every node: every value a
 * concrete row can produce lies in the node's abstract value. Nothing is guessed: what cannot be proved is "unknown".
 *
 * Domains of a numeric node (all of them hold at once; their intersection is used, a reduced product):
 *   nullness     MAYBE_NULL (some row can be NULL) and MAYBE_VALUE (some row can have a value): definitely NULL,
 *                definitely not NULL, unknown; SQL semantics (an operation with a NULL operand is NULL).
 *   interval     [lo, hi] closed, possibly unbounded: exact canonical int64 rationals (carrier EXACT) or outward-rounded
 *                doubles (carrier BINARY64, the tight FND-009 helpers of math_carriers.h).
 *   affine form  center + sum_k c_k e_k over noise symbols e_k in [-1, 1]: symbol j < TR_UNC_FRESH is column j (its
 *                value is center_j + radius_j e_j in the group), symbol TR_UNC_FRESH + k is the approximation symbol
 *                of node k (it absorbs the node's nonlinear remainder, its rounding errors and condensed terms; it is
 *                a function of the row, so a node used twice keeps its correlation: x - x = 0, (x y) - (x y) = 0).
 *                At most `rank` terms are kept (the smallest are condensed into the node's symbol); a node whose
 *                approximation error (sum |c_k| over approximation symbols) passes the error budget collapses to its
 *                own symbol. Exact rational coefficients (EXACT) or doubles with every rounding error bounded by the
 *                FND-009 directed operations (BINARY64), optionally truncated to `precision` bits (the truncation
 *                error is bounded the same way).
 *   congruence   value in r + m Z (rationals; m = 0: exactly r): Granger's rules (+, -, *, scalar division, joins by
 *                gcd, squares r^2 + gcd(2 r m, m^2) Z) and the CRT meet (tr_unc_cong_meet). GF(p) is the case m = p.
 *   zero fact    ZERO (every value is 0), NONZERO (0 is impossible: excluded by the interval or the congruence),
 *                unknown.
 * A boolean node (comparison, AND, OR, NOT, IS NULL, IS NOT NULL) holds the set of outcomes its rows can have:
 * TRUE, FALSE, NULL (SQL three-valued logic). A comparison a op b decides on d = a - b as an affine form (so
 * correlated sides cancel: x + 1 > x is always TRUE), intersected with the interval difference and the congruence of
 * d (an equality is impossible when 0 is not in r + m Z).
 *
 * Transfer functions (soundness: every concrete result is enclosed; exhaustively tested on small domains):
 *   ADD, SUB, NEG     exact on every domain (intervals: endpoint sums; forms: term-wise; congruence: (ra +- rb, gcd))
 *   MUL               intervals: the four endpoint products (0 * inf = 0); forms: a0 b0 + a0 B + b0 A plus the
 *                     remainder rad(A) rad(B) on the node's symbol; congruence: (ra rb, gcd(ma mb, ma rb, mb ra))
 *   DIV k             scalar division by a nonzero rational constant (a division by 0 is refused at the program check)
 *   MIN, MAX          the side that is provably smaller / larger, else (a + b)/2 -+ |a - b|/2 with |.| below; joins
 *   ABS, SQR          intervals exact; forms: the secant slope alpha with [m1, m2] = the exact range of f(x) - alpha x
 *                     on the operand's interval, f = alpha x + (m1 + m2)/2 +- (m2 - m1)/2
 *   MOD k             x - k floor(x / k) for a positive rational k: an exact shift when floor(x/k) is constant on the
 *                     interval, else [0, k) (tightened by the congruence r + gcd(m, k) Z)
 *   LT LE EQ NE GE GT on d = a - b;  AND, OR, NOT on outcome sets;  IS NULL, IS NOT NULL on nullness.
 * Overflow (EXACT): an int64 rational that does not fit widens soundly (an interval bound to +-inf or to a bound kept
 * by sign, a form or congruence to unknown), flagged TR_UNC_WIDENED, never wrapped and never a guess.
 *
 * Decisions (a boolean root, per group): ALL (every selected row's predicate is TRUE: the filter keeps the whole
 * group), NONE (no row can be TRUE: the group is pruned), UNKNOWN (rows must be evaluated concretely), EMPTY (no
 * selected row). Unsound pruning is impossible by construction: ALL needs the outcome set {TRUE}, NONE needs TRUE to
 * be impossible, and the concrete fallback (tr_unc_rows) evaluates every row of an UNKNOWN group exactly.
 *
 * Concrete rows (tr_unc_rows, the CPU reference of the CUDA kernel): the program evaluated exactly on each selected
 * row (int64 rationals with 128-bit intermediates, SQL NULLs); a row whose exact value leaves int64 rationals is
 * counted as overflow (TR_MATH_OVERFLOW: never a guessed bit).
 *
 * Overflow decisions (tr_unc_sum_fits): every partial sum over a group of values in [lo, hi] lies in
 * [rows min(lo, 0), rows max(hi, 0)]; with values that are integer multiples of the unit (decided by the congruence)
 * the accumulator of value / unit is proven to fit `bits` signed bits, or not proven.
 *
 * Certificates (FND-007): tr_unc_identity emits a LAW (ring identity) when the symbolic affine evaluation of a
 * polynomial node is exact: D E(x) - D L(x) = 0 in Z[x] (E the node's expansion, L its exact affine form), decided by
 * FND-007's canonicalization (x - x = 0 is the law x - x = 0). tr_unc_certify emits SNAPSHOT certificates of data
 * facts the analysis proved in every group: NOT NULL (a column), a congruence modulo a prime p (a GF(p) polynomial
 * claim D E - D r = 0, NULL rows skipped) and a constant (D E - D c = 0 over Z); FND-007 re-verifies each claim on
 * every selected row before it issues the certificate. Interval bounds and pruning decisions have no FND-007 claim
 * kind: they are valid while the state's recorded scope is fresh (tr_unc_state_fresh: FND-007's record comparison).
 *
 * Budgets (math_budgets.h, checked before allocation; TR_MATH_BUDGET with the tr_math_refusal):
 *   program      circuit = nodes, variables = columns, support = rank (terms of a form)
 *   state        states = groups x columns, bytes = the inputs, the accumulators and the scope record
 *   evaluation   output = groups evaluated, bytes = the results (CUDA: the device copies and the per-group scratch)
 *   rows         bytes = the bitmap (CUDA: the device copies of the columns read, ids, decisions and the bitmap)
 *   identity     support = terms of the expansion (also <= TR_CERT_MAX_TERMS), degree = its degree
 * Representation limits: 64 nodes, 32 columns, 16 terms per form; programs past them are TR_MATH_UNSUPPORTED.
 *
 * Errors: TR_MATH_INVALID (malformed program, inputs or arguments), _MEMORY, _OVERFLOW (an exact row evaluation or a
 * certificate coefficient left int64), _BUDGET, _UNSUPPORTED, _DOMAIN_ERROR (a fact the analysis did not prove; an
 * expression that is not an affine identity), TR_MATH_STALE (the state's scope changed). Objects are immutable and
 * individually owned (constructors return NULL and set *error); read-only calls may run concurrently.
 *
 * The CUDA library (math_unc_gpu.cu): tr_mgpu_unc_eval evaluates many groups at once (one work item per group, one
 * output slot each, its node values in a per-group device scratch), tr_mgpu_unc_rows the concrete fallback (one
 * work item per 64-row word, one bitmap word each, counts by atomics). Both call the inline functions below, so they
 * equal the CPU bit for bit (the double operations are FND-009's contraction-safe directed operations). */
#ifndef TORBLES_MATH_UNC_H
#define TORBLES_MATH_UNC_H
#include <stdint.h>
#include <string.h>
#include <math.h>
#include "math_num.h"
#include "math_budgets.h"
#include "math_scope.h"
#include "math_carriers.h"
#include "math_certificates.h"
#ifdef __cplusplus
extern "C" {
#endif

#define TR_UNC_ABI 1
#define TR_UNC_MAX_NODES 64          /* nodes of a program (representation limit) */
#define TR_UNC_MAX_TERMS 16          /* terms of one affine form (the largest rank) */
#define TR_UNC_MAX_COLS 32           /* columns a program reads */
#define TR_UNC_FRESH 32              /* the approximation symbol of node k is TR_UNC_FRESH + k */
#define TR_UNC_MAX_SCALE 18

/* node operations */
enum { TR_UNC_COL = 1, TR_UNC_CONST = 2, TR_UNC_ADD = 3, TR_UNC_SUB = 4, TR_UNC_MUL = 5, TR_UNC_NEG = 6,
       TR_UNC_DIV = 7,               /* a / k, k a nonzero rational constant */
       TR_UNC_MIN = 8, TR_UNC_MAX = 9, TR_UNC_ABS = 10, TR_UNC_SQR = 11,
       TR_UNC_MOD = 12,              /* a - k floor(a / k), k a positive rational constant */
       TR_UNC_LT = 13, TR_UNC_LE = 14, TR_UNC_EQ = 15, TR_UNC_NE = 16, TR_UNC_GE = 17, TR_UNC_GT = 18,
       TR_UNC_AND = 19, TR_UNC_OR = 20, TR_UNC_NOT = 21, TR_UNC_ISNULL = 22, TR_UNC_NOTNULL = 23,
       TR_UNC_OPS = 24 };
enum { TR_UNC_EXACT = 1, TR_UNC_BINARY64 = 2 };                                   /* carriers */
enum { TR_UNC_DOM_AFFINE = 1, TR_UNC_DOM_CONGRUENCE = 2, TR_UNC_DOM_ALL = 3,     /* optional domains */
       TR_UNC_DOM_SYMBOLIC = 4 };    /* internal (tr_unc_symbolic): column symbols are unbounded variables */
enum { TR_UNC_NUMERIC = 1, TR_UNC_BOOLEAN = 2 };                                  /* node kinds */
enum { TR_UNC_MAYBE_NULL = 1, TR_UNC_MAYBE_VALUE = 2 };                           /* nullness bits */
enum { TR_UNC_TRUE = 1, TR_UNC_FALSE = 2, TR_UNC_NULL = 4 };                      /* outcome bits */
enum { TR_UNC_ZERO_UNKNOWN = 0, TR_UNC_ZERO = 1, TR_UNC_NONZERO = 2 };
enum { TR_UNC_UNKNOWN = 0, TR_UNC_ALL = 1, TR_UNC_NONE = 2, TR_UNC_EMPTY = 3 };   /* decisions (0: evaluate rows) */
enum { TR_UNC_LO_INF = 1, TR_UNC_HI_INF = 2, TR_UNC_AFFINE_TOP = 4, TR_UNC_CONG_TOP = 8,
       TR_UNC_WIDENED = 16,          /* an exact operation overflowed int64 rationals and was widened soundly */
       TR_UNC_CONDENSED = 32,        /* terms past the rank were condensed into the node's symbol */
       TR_UNC_BUDGET = 64,           /* the approximation error passed the error budget: the form collapsed */
       TR_UNC_ROUNDED = 128 };       /* BINARY64: rounding errors were bounded into the node's symbol */
enum { TR_UNC_FACT_NOT_NULL = 1, TR_UNC_FACT_CONGRUENCE = 2, TR_UNC_FACT_CONSTANT = 3 };
enum { TR_UNC_NEED_CPU = 0, TR_UNC_NEED_DEVICE = 1 };

/* ------------------------------------------------------------------ structs crossing the ctypes boundary */
typedef struct { int32_t op, a, b, reserved; tr_num_rational k; } tr_unc_node;   /* 32 bytes; a, b: earlier nodes
                                                        (COL: a = the column); k: CONST value, DIV / MOD constant */
typedef struct {
    int32_t nnodes, ncols;           /* nodes in topological order; columns the inputs have (COL a < ncols) */
    int32_t carrier, precision;      /* TR_UNC_EXACT (precision 0) or TR_UNC_BINARY64 (2..53 coefficient bits) */
    int32_t rank, domains;           /* terms kept per form (1..16); TR_UNC_DOM_* bits (intervals always) */
    int32_t root, reserved;          /* the result node */
    tr_num_rational budget;          /* the error budget (approximation radius per form); denominator 0: none */
    const tr_unc_node *nodes;
} tr_unc_program;                    /* 56 bytes */

/* one column of one group: the abstract input */
typedef struct {
    uint64_t rows;                   /* selected rows of the group (equal for every column of a group) */
    uint64_t present;                /* rows where the column is present (not NULL) */
    int64_t min, max;                /* logical extremes of the present values (ignored when present == 0) */
    uint64_t modulus;                /* gcd of (value - min) over the present values: 0 = a single value */
    int32_t scale, reserved;         /* value = logical / 10^scale */
} tr_unc_input;                      /* 48 bytes */

typedef struct { int32_t sym, reserved; int64_t num, den; double d; } tr_unc_term;   /* 32 bytes: EXACT num/den,
                                                                                         BINARY64 d */
typedef struct {
    int32_t kind, null, outcomes, zero;
    int32_t flags, nterms;
    tr_num_rational lo, hi;          /* EXACT interval (unless LO_INF / HI_INF) */
    double dlo, dhi;                 /* BINARY64 interval; EXACT: the enclosure [RD(lo), RU(hi)] */
    tr_num_rational cr, cm;          /* congruence r + m Z (unless CONG_TOP) */
    tr_num_rational center;          /* EXACT affine center */
    double dcenter, reserved;        /* BINARY64 affine center */
    tr_unc_term term[TR_UNC_MAX_TERMS];   /* sorted by symbol */
} tr_unc_value;                      /* 648 bytes */

typedef struct {
    int32_t kind, decision;          /* decision: boolean roots (TR_UNC_ALL / _NONE / _UNKNOWN / _EMPTY) */
    int32_t outcomes, null;
    int32_t zero, flags;
    int32_t nterms, fresh;           /* the root's affine terms; how many are approximation symbols */
    tr_num_rational lo, hi;
    double dlo, dhi;
    tr_num_rational cr, cm;
    tr_num_rational center, radius, error;   /* EXACT: center, sum |c_k|, sum over approximation symbols */
    double dcenter, dradius, derror;         /* BINARY64 (EXACT: enclosures rounded outward) */
    uint64_t rows, reserved;
} tr_unc_result;                     /* 200 bytes */

typedef struct {
    int32_t nnodes, nlive, root, reserved;
    int32_t slot[TR_UNC_MAX_NODES];  /* the scratch slot of each node's value */
    int32_t kind[TR_UNC_MAX_NODES];  /* TR_UNC_NUMERIC / _BOOLEAN */
} tr_unc_plan;                       /* 528 bytes */

typedef struct {
    uint64_t rows, selected;         /* the scope's physical / selected rows */
    uint64_t rows_proven;            /* selected rows decided by their group's proof (ALL or NONE) */
    uint64_t rows_evaluated;         /* selected rows evaluated concretely (the fallback) */
    uint64_t rows_true;              /* rows whose predicate is TRUE (bits set) */
    uint64_t overflow, first_overflow;   /* rows whose exact evaluation left int64 rationals; the smallest */
    uint64_t groups_all, groups_none, groups_unknown, groups_empty;   /* decisions (tr_unc_filter) */
} tr_unc_filter_stats;               /* 88 bytes */

typedef struct {
    int32_t proven, integral;        /* 1: every partial sum fits `bits`; 1: values are multiples of the unit */
    int32_t bits, bounded;           /* the accumulator width; 1: the bounds below are finite */
    tr_math_i128 lo, hi;             /* bounds of every partial sum, in units (when bounded and integral) */
} tr_unc_sum;                        /* 48 bytes */

typedef struct {
    uint64_t ngroups, bytes;
    int32_t ncols, scoped;           /* scoped: 1 when built from a scope (its record is kept) */
    int32_t cols[TR_UNC_MAX_COLS];   /* the scope columns (scoped) */
} tr_unc_state_info;                 /* 152 bytes */

typedef struct tr_unc_state_s tr_unc_state;

/* ================================================================ inline helpers (host and CUDA device code) */

/* ---------------------------------------------------------------- int64 rationals */
TR_NUM_HD tr_num_rational tr_unc_qv(int64_t n, int64_t d) { tr_num_rational r; r.numerator = n; r.denominator = d; return r; }
TR_NUM_HD int tr_unc_qsign(tr_num_rational a) { return a.numerator > 0 ? 1 : (a.numerator < 0 ? -1 : 0); }
TR_NUM_HD int tr_unc_qneg(tr_num_rational a, tr_num_rational *r) {
    if (a.numerator == INT64_MIN) return 0;
    r->numerator = -a.numerator; r->denominator = a.denominator; return 1;
}
TR_NUM_HD int tr_unc_qabs(tr_num_rational a, tr_num_rational *r) {
    if (a.numerator >= 0) { *r = a; return 1; }
    return tr_unc_qneg(a, r);
}
TR_NUM_HD int tr_unc_qeq(tr_num_rational a, tr_num_rational b) {   /* canonical rationals */
    return a.numerator == b.numerator && a.denominator == b.denominator;
}
/* floor(n / d) for d > 0 */
TR_NUM_HD __int128 tr_unc_floordiv(__int128 n, __int128 d) {
    __int128 q = n / d;
    if (n % d != 0 && n < 0) q -= 1;
    return q;
}
TR_NUM_HD int64_t tr_unc_qfloor(tr_num_rational a) { return (int64_t)tr_unc_floordiv(a.numerator, a.denominator); }
TR_NUM_HD int tr_unc_qceil(tr_num_rational a, int64_t *out) {
    __int128 q = -tr_unc_floordiv(-(__int128)a.numerator, a.denominator);
    if (q > INT64_MAX) return 0;
    *out = (int64_t)q; return 1;
}
TR_NUM_HD unsigned __int128 tr_unc_gcd128(unsigned __int128 a, unsigned __int128 b) {
    while (b) { unsigned __int128 t = a % b; a = b; b = t; }
    return a;
}
/* the nonnegative generator of a Z + b Z in Q; 0 when it does not fit */
TR_NUM_HD int tr_unc_qgcd(tr_num_rational a, tr_num_rational b, tr_num_rational *r) {
    if (a.numerator == 0) return tr_unc_qabs(b, r);
    if (b.numerator == 0) return tr_unc_qabs(a, r);
    unsigned __int128 qa = (uint64_t)a.denominator, qb = (uint64_t)b.denominator;
    unsigned __int128 L = qa / tr_unc_gcd128(qa, qb) * qb;
    unsigned __int128 A = (unsigned __int128)tr_abs_u64(a.numerator) * (L / qa);
    unsigned __int128 B = (unsigned __int128)tr_abs_u64(b.numerator) * (L / qb);
    unsigned __int128 n = tr_unc_gcd128(A, B), h = tr_unc_gcd128(n, L);
    n /= h; L /= h;
    if (n > (unsigned __int128)INT64_MAX || L > (unsigned __int128)INT64_MAX) return 0;
    r->numerator = (int64_t)n; r->denominator = (int64_t)L; return 1;
}
/* r mod m in [0, m) for m > 0 */
TR_NUM_HD int tr_unc_qmod(tr_num_rational r, tr_num_rational m, tr_num_rational *out) {
    tr_num_rational q, t;
    if (!tr_q_div(r, m, &q)) return 0;
    if (!tr_q_mul(m, tr_q_int(tr_unc_qfloor(q)), &t)) return 0;
    return tr_q_sub(r, t, out);
}
TR_NUM_HD int64_t tr_unc_pow10(int32_t s) { int64_t p = 1; for (int32_t i = 0; i < s; i++) p *= 10; return p; }

/* ---------------------------------------------------------------- doubles (FND-009 directed operations only) */
TR_NUM_HD double tr_unc_dmax(double a, double b) { return a > b ? a : b; }
TR_NUM_HD double tr_unc_dmin(double a, double b) { return a < b ? a : b; }
TR_NUM_HD double tr_unc_nz(double x) { return x == 0 ? 0.0 : x; }   /* no -0 */
/* x truncated toward zero to p significant bits (p >= 53: x) */
TR_NUM_HD double tr_unc_trunc(double x, int32_t p) {
    if (p >= 53 || x == 0 || x != x || x == INFINITY || x == -INFINITY) return x;
    union { double d; uint64_t u; } v; v.d = x;
    v.u &= ~(((uint64_t)1 << (53 - p)) - 1);
    return v.d;
}
TR_NUM_HD tr_math_interval tr_unc_enclose(tr_num_rational q) {
    tr_math_interval r; r.lo = -INFINITY; r.hi = INFINITY;
    tr_interval_enclose_q64(q.numerator, q.denominator, &r);
    return r;
}
TR_NUM_HD double tr_unc_enclose_lo(tr_num_rational q) { return tr_unc_enclose(q).lo; }
TR_NUM_HD double tr_unc_enclose_hi(tr_num_rational q) { return tr_unc_enclose(q).hi; }
TR_NUM_HD double tr_unc_half_dir(double x, int up) { return tr_mul_dir(x, 0.5, up); }

/* ---------------------------------------------------------------- scalars of either carrier, with an error account */
/* EXACT: q. BINARY64: the enclosure [lo, hi] of the scalar (a point for a computed coefficient). */
typedef struct { tr_num_rational q; double lo, hi; } tr_unc_s;
typedef struct { int32_t carrier, precision, fail, reserved; double err; } tr_unc_ctx;   /* err: BINARY64 rounding
                                                                                          errors (an upper bound) */
TR_NUM_HD tr_unc_s tr_unc_s_zero(void) { tr_unc_s r; r.q = tr_q_int(0); r.lo = r.hi = 0.0; return r; }
TR_NUM_HD tr_unc_s tr_unc_s_point(double d) { tr_unc_s r; r.q = tr_q_int(0); r.lo = r.hi = tr_unc_nz(d); return r; }
TR_NUM_HD tr_unc_s tr_unc_s_rat(tr_num_rational q) { tr_unc_s r; tr_math_interval e = tr_unc_enclose(q); r.q = q; r.lo = e.lo; r.hi = e.hi; return r; }
TR_NUM_HD tr_unc_s tr_unc_s_iv(double lo, double hi) { tr_unc_s r; r.q = tr_q_int(0); r.lo = lo; r.hi = hi; return r; }
TR_NUM_HD tr_unc_ctx tr_unc_ctx_make(int32_t carrier, int32_t precision) {
    tr_unc_ctx c; c.carrier = carrier; c.precision = carrier == TR_UNC_BINARY64 ? precision : 53;
    c.fail = 0; c.reserved = 0; c.err = 0.0; return c;
}
/* a point in [lo, hi] (truncated to the precision), the distance to the ends accounted as error */
TR_NUM_HD tr_unc_s tr_unc_s_round(tr_unc_ctx *c, double lo, double hi) {
    if (!(lo > -INFINITY) || !(hi < INFINITY)) { c->fail = 1; return tr_unc_s_zero(); }
    double v = lo;
    if (lo != hi && hi > 0 && lo < 0) v = 0.0;            /* a straddling enclosure: the point 0 */
    if (c->precision < 53) v = tr_unc_trunc(v, c->precision);
    double e = tr_unc_dmax(tr_add_ru(hi, -v), tr_add_ru(v, -lo));
    if (e > 0) c->err = tr_add_ru(c->err, e);
    if (c->err == INFINITY) c->fail = 1;
    return tr_unc_s_point(v);
}
TR_NUM_HD tr_unc_s tr_unc_s_mul(tr_unc_ctx *c, tr_unc_s a, tr_unc_s b) {
    if (c->carrier == TR_UNC_EXACT) { tr_unc_s r = tr_unc_s_zero(); if (!tr_q_mul(a.q, b.q, &r.q)) c->fail = 1; return r; }
    tr_math_interval x, y, p; x.lo = a.lo; x.hi = a.hi; y.lo = b.lo; y.hi = b.hi;
    p = tr_interval_mul(x, y);
    return tr_unc_s_round(c, p.lo, p.hi);
}
TR_NUM_HD tr_unc_s tr_unc_s_add(tr_unc_ctx *c, tr_unc_s a, tr_unc_s b) {
    if (c->carrier == TR_UNC_EXACT) { tr_unc_s r = tr_unc_s_zero(); if (!tr_q_add(a.q, b.q, &r.q)) c->fail = 1; return r; }
    return tr_unc_s_round(c, tr_add_rd(a.lo, b.lo), tr_add_ru(a.hi, b.hi));
}
TR_NUM_HD tr_unc_s tr_unc_s_neg(tr_unc_ctx *c, tr_unc_s a) {
    tr_unc_s r = tr_unc_s_zero();
    if (c->carrier == TR_UNC_EXACT) { if (!tr_unc_qneg(a.q, &r.q)) c->fail = 1; return r; }
    r.lo = tr_unc_nz(-a.hi); r.hi = tr_unc_nz(-a.lo); return r;
}
/* an upper bound of |a| (EXACT: |a|) */
TR_NUM_HD tr_unc_s tr_unc_s_mag(tr_unc_ctx *c, tr_unc_s a) {
    tr_unc_s r = tr_unc_s_zero();
    if (c->carrier == TR_UNC_EXACT) { if (!tr_unc_qabs(a.q, &r.q)) c->fail = 1; return r; }
    r.lo = r.hi = tr_unc_dmax(fabs(a.lo), fabs(a.hi)); return r;
}
/* acc + |a| rounded up (radius accumulators; acc >= 0) */
TR_NUM_HD tr_unc_s tr_unc_s_addmag(tr_unc_ctx *c, tr_unc_s acc, tr_unc_s a) {
    tr_unc_s m = tr_unc_s_mag(c, a);
    if (c->carrier == TR_UNC_EXACT) { tr_unc_s r = tr_unc_s_zero(); if (!tr_q_add(acc.q, m.q, &r.q)) c->fail = 1; return r; }
    double s = tr_add_ru(acc.hi, m.hi);
    if (s == INFINITY) c->fail = 1;
    return tr_unc_s_point(s);
}
/* the comparison of two points / exact values */
TR_NUM_HD int tr_unc_s_cmp(const tr_unc_ctx *c, tr_unc_s a, tr_unc_s b) {
    if (c->carrier == TR_UNC_EXACT) return tr_q_cmp(a.q, b.q);
    return a.lo < b.lo ? -1 : (a.lo > b.lo ? 1 : 0);
}
TR_NUM_HD int tr_unc_s_iszero(const tr_unc_ctx *c, tr_unc_s a) {
    return c->carrier == TR_UNC_EXACT ? a.q.numerator == 0 : (a.lo == 0 && a.hi == 0);
}
TR_NUM_HD tr_unc_s tr_unc_s_term(const tr_unc_term *t) {
    tr_unc_s r; r.q.numerator = t->num; r.q.denominator = t->den; r.lo = r.hi = t->d; return r;
}
TR_NUM_HD void tr_unc_term_set(tr_unc_term *t, int32_t sym, tr_unc_s s) {
    t->sym = sym; t->reserved = 0; t->num = s.q.numerator; t->den = s.q.denominator ? s.q.denominator : 1; t->d = s.lo;
}

/* ---------------------------------------------------------------- values */
TR_NUM_HD void tr_unc_value_clear(tr_unc_value *v, int32_t kind) {
    v->kind = kind; v->null = 0; v->outcomes = 0; v->zero = TR_UNC_ZERO_UNKNOWN; v->flags = 0; v->nterms = 0;
    v->lo = tr_q_int(0); v->hi = tr_q_int(0); v->dlo = 0.0; v->dhi = 0.0;
    v->cr = tr_q_int(0); v->cm = tr_q_int(0); v->center = tr_q_int(0); v->dcenter = 0.0; v->reserved = 0.0;
    for (int32_t i = 0; i < TR_UNC_MAX_TERMS; i++) {
        v->term[i].sym = 0; v->term[i].reserved = 0; v->term[i].num = 0; v->term[i].den = 1; v->term[i].d = 0.0;
    }
}
/* the unknown numeric value with the given nullness (every domain at its top) */
TR_NUM_HD void tr_unc_value_top(tr_unc_value *v, int32_t null) {
    tr_unc_value_clear(v, TR_UNC_NUMERIC);
    v->null = null;
    v->flags = TR_UNC_LO_INF | TR_UNC_HI_INF | TR_UNC_AFFINE_TOP | TR_UNC_CONG_TOP;
    v->dlo = -INFINITY; v->dhi = INFINITY;
}
/* no value can occur (only the nullness remains) */
TR_NUM_HD void tr_unc_set_novalue(tr_unc_value *v) {
    int32_t n = v->null & ~TR_UNC_MAYBE_VALUE;
    tr_unc_value_clear(v, TR_UNC_NUMERIC);
    v->null = n;
    v->flags = TR_UNC_AFFINE_TOP | TR_UNC_CONG_TOP;
}
TR_NUM_HD tr_unc_s tr_unc_s_center(const tr_unc_value *v) {
    tr_unc_s r; r.q = v->center; r.lo = r.hi = v->dcenter; return r;
}

/* ---------------------------------------------------------------- exact bounds with infinities */
typedef struct { tr_num_rational v; int32_t inf; } tr_unc_xb;   /* inf: -1 = -inf, 0 = finite v, +1 = +inf */
TR_NUM_HD tr_unc_xb tr_unc_xb_of(tr_num_rational v, int32_t inf) { tr_unc_xb r; r.v = v; r.inf = inf; return r; }
TR_NUM_HD tr_unc_xb tr_unc_xlo(const tr_unc_value *v) { return tr_unc_xb_of(v->lo, (v->flags & TR_UNC_LO_INF) ? -1 : 0); }
TR_NUM_HD tr_unc_xb tr_unc_xhi(const tr_unc_value *v) { return tr_unc_xb_of(v->hi, (v->flags & TR_UNC_HI_INF) ? 1 : 0); }
TR_NUM_HD int tr_unc_xsign(tr_unc_xb a) { return a.inf ? a.inf : tr_unc_qsign(a.v); }
TR_NUM_HD int tr_unc_xcmp(tr_unc_xb a, tr_unc_xb b) {
    if (a.inf || b.inf) return a.inf < b.inf ? -1 : (a.inf > b.inf ? 1 : 0);
    return tr_q_cmp(a.v, b.v);
}
TR_NUM_HD void tr_unc_set_lo(tr_unc_value *v, tr_unc_xb b) {
    if (b.inf < 0) { v->flags |= TR_UNC_LO_INF; v->lo = tr_q_int(0); } else { v->flags &= ~TR_UNC_LO_INF; v->lo = b.v; }
}
TR_NUM_HD void tr_unc_set_hi(tr_unc_value *v, tr_unc_xb b) {
    if (b.inf > 0) { v->flags |= TR_UNC_HI_INF; v->hi = tr_q_int(0); } else { v->flags &= ~TR_UNC_HI_INF; v->hi = b.v; }
}
/* a lower (up = 0) / upper (up = 1) bound of a + b from bounds of the same side; past int64: a bound kept by sign */
TR_NUM_HD tr_unc_xb tr_unc_xadd(tr_unc_xb a, tr_unc_xb b, int up, int32_t *widened) {
    int inf = up ? 1 : -1;
    if (a.inf || b.inf) return tr_unc_xb_of(tr_q_int(0), inf);
    tr_num_rational s;
    if (tr_q_add(a.v, b.v, &s)) return tr_unc_xb_of(s, 0);
    *widened = 1;
    int sa = tr_unc_qsign(a.v), sb = tr_unc_qsign(b.v);
    if (!up) {                        /* a, b >= 0: a + b >= max(a, b); a >= 0: a + b >= b */
        if (sa >= 0 && sb >= 0) return tr_q_cmp(a.v, b.v) >= 0 ? a : b;
        if (sa >= 0) return b;
        if (sb >= 0) return a;
    } else {
        if (sa <= 0 && sb <= 0) return tr_q_cmp(a.v, b.v) <= 0 ? a : b;
        if (sa <= 0) return b;
        if (sb <= 0) return a;
    }
    return tr_unc_xb_of(tr_q_int(0), inf);
}
/* -a as an upper (up = 1) or lower bound; -(-2^63 / d) does not fit: +inf above, (2^63 - 1) / d below */
TR_NUM_HD tr_unc_xb tr_unc_xneg(tr_unc_xb a, int up, int32_t *widened) {
    if (a.inf) return tr_unc_xb_of(tr_q_int(0), -a.inf);
    tr_num_rational r = tr_q_int(0);
    if (tr_unc_qneg(a.v, &r)) return tr_unc_xb_of(r, 0);
    *widened = 1;
    if (up) return tr_unc_xb_of(tr_q_int(0), 1);
    tr_q_make(INT64_MAX, a.v.denominator, &r);
    return tr_unc_xb_of(r, 0);
}
/* an enclosure [plo, phi] of the product of two bounds (0 * inf = 0; an overflowed product keeps its sign) */
TR_NUM_HD void tr_unc_xmul(tr_unc_xb a, tr_unc_xb b, tr_unc_xb *plo, tr_unc_xb *phi, int32_t *widened) {
    int sa = tr_unc_xsign(a), sb = tr_unc_xsign(b), s = sa * sb;
    if (sa == 0 || sb == 0) { *plo = *phi = tr_unc_xb_of(tr_q_int(0), 0); return; }
    if (a.inf || b.inf) { *plo = *phi = tr_unc_xb_of(tr_q_int(0), s); return; }
    tr_num_rational p;
    if (tr_q_mul(a.v, b.v, &p)) { *plo = *phi = tr_unc_xb_of(p, 0); return; }
    *widened = 1;
    if (s > 0) { *plo = tr_unc_xb_of(tr_q_int(0), 0); *phi = tr_unc_xb_of(tr_q_int(0), 1); }
    else { *plo = tr_unc_xb_of(tr_q_int(0), -1); *phi = tr_unc_xb_of(tr_q_int(0), 0); }
}
/* [alo, ahi] x [blo, bhi]: the hull of the endpoint products */
TR_NUM_HD void tr_unc_xmul_iv(tr_unc_xb alo, tr_unc_xb ahi, tr_unc_xb blo, tr_unc_xb bhi, tr_unc_xb *lo, tr_unc_xb *hi,
                              int32_t *widened) {
    tr_unc_xb l[4], h[4];
    tr_unc_xmul(alo, blo, &l[0], &h[0], widened);
    tr_unc_xmul(alo, bhi, &l[1], &h[1], widened);
    tr_unc_xmul(ahi, blo, &l[2], &h[2], widened);
    tr_unc_xmul(ahi, bhi, &l[3], &h[3], widened);
    *lo = l[0]; *hi = h[0];
    for (int i = 1; i < 4; i++) {
        if (tr_unc_xcmp(l[i], *lo) < 0) *lo = l[i];
        if (tr_unc_xcmp(h[i], *hi) > 0) *hi = h[i];
    }
}

/* ---------------------------------------------------------------- congruences r + m Z */
/* canonical (r mod m in [0, m) for m > 0); 0 when it does not fit */
TR_NUM_HD int tr_unc_cong_norm(tr_num_rational *r, tr_num_rational m) {
    if (m.numerator == 0) return 1;
    tr_num_rational x;
    if (!tr_unc_qmod(*r, m, &x)) return 0;
    *r = x; return 1;
}
/* 0 when x is provably outside r + m Z, else 1 (also when int64 rationals cannot decide it) */
TR_NUM_HD int tr_unc_cong_contains(tr_num_rational r, tr_num_rational m, tr_num_rational x) {
    tr_num_rational d, q;
    if (!tr_q_sub(x, r, &d)) return 1;
    if (m.numerator == 0) return d.numerator == 0;
    if (!tr_q_div(d, m, &q)) return 1;
    return q.denominator == 1;
}
/* the CRT meet of r1 + m1 Z and r2 + m2 Z: 1 (the intersection is r + m Z), 0 (empty), -1 (does not fit int64) */
TR_NUM_HD int tr_unc_cong_meet_hd(tr_num_rational r1, tr_num_rational m1, tr_num_rational r2, tr_num_rational m2,
                                  tr_num_rational *r, tr_num_rational *m) {
    if (tr_unc_qsign(m1) < 0 || tr_unc_qsign(m2) < 0) return -1;
    if (m1.numerator == 0 || m2.numerator == 0) {        /* a constant meets a class: the constant or nothing */
        tr_num_rational c = m1.numerator == 0 ? r1 : r2, rr = m1.numerator == 0 ? r2 : r1, mm = m1.numerator == 0 ? m2 : m1;
        tr_num_rational d, q;
        if (!tr_q_sub(c, rr, &d)) return -1;
        if (mm.numerator != 0) { if (!tr_q_div(d, mm, &q)) return -1; if (q.denominator != 1) return 0; }
        else if (d.numerator != 0) return 0;
        *r = c; *m = tr_q_int(0); return 1;
    }
    /* a common denominator L: classes of integers R1 + M1 Z and R2 + M2 Z, then the integer CRT */
    unsigned __int128 d1 = (uint64_t)r1.denominator, e1 = (uint64_t)m1.denominator, d2 = (uint64_t)r2.denominator, e2 = (uint64_t)m2.denominator;
    unsigned __int128 L = d1 / tr_unc_gcd128(d1, e1) * e1;
    L = L / tr_unc_gcd128(L, d2) * d2;
    if (L > (unsigned __int128)INT64_MAX) return -1;
    L = L / tr_unc_gcd128(L, e2) * e2;
    if (L > (unsigned __int128)INT64_MAX) return -1;
    __int128 R1 = (__int128)r1.numerator * (__int128)(L / d1), R2 = (__int128)r2.numerator * (__int128)(L / d2);
    __int128 M1 = (__int128)m1.numerator * (__int128)(L / e1), M2 = (__int128)m2.numerator * (__int128)(L / e2);
    if (R1 > INT64_MAX || R1 < INT64_MIN || R2 > INT64_MAX || R2 < INT64_MIN || M1 > INT64_MAX || M2 > INT64_MAX) return -1;
    __int128 g0 = (__int128)tr_unc_gcd128((unsigned __int128)M1, (unsigned __int128)M2), diff = R2 - R1;
    if (diff % g0 != 0) return 0;                        /* r1 != r2 (mod gcd(m1, m2)): disjoint */
    __int128 a = M1 / g0, b = M2 / g0, old_s = 1, s = 0, old_r = a, rr = b;
    while (rr) { __int128 qq = old_r / rr, t = old_r - qq * rr; old_r = rr; rr = t; t = old_s - qq * s; old_s = s; s = t; }
    __int128 lcm = a * M2;                                /* M1 M2 / g < 2^126 */
    __int128 tq = (diff / g0) % b, inv = old_s % b;       /* M1 t = R2 - R1 (mod M2): t = tq inv (mod b) */
    if (tq < 0) tq += b;
    if (inv < 0) inv += b;
    __int128 tt = (__int128)(((unsigned __int128)tq * (unsigned __int128)inv) % (unsigned __int128)b);
    __int128 x = R1 + M1 * tt, xm = x % lcm;
    if (xm < 0) xm += lcm;
    if (!tr_q_make128(xm, (__int128)L, r) || !tr_q_make128(lcm, (__int128)L, m)) return -1;
    return 1;
}

/* ---------------------------------------------------------------- affine forms */
/* a form being built inside one node: terms of the operands (never the node's own symbol) and a pending error */
typedef struct {
    tr_unc_s center, err;            /* err >= 0: attached to the node's symbol when the node's form is finished */
    int32_t n, top;
    tr_unc_term t[2 * TR_UNC_MAX_TERMS + 2];
} tr_unc_aff;
/* sum |c_k| over every symbol (all) or over the approximation symbols, rounded up */
TR_NUM_HD tr_unc_s tr_unc_radius(tr_unc_ctx *c, const tr_unc_term *t, int32_t n, int all) {
    tr_unc_s r = tr_unc_s_zero();
    for (int32_t i = 0; i < n; i++)
        if (all || t[i].sym >= TR_UNC_FRESH) r = tr_unc_s_addmag(c, r, tr_unc_s_term(&t[i]));
    return r;
}
/* out = alpha A + beta B + gamma (mode 0) or the product center a0 b0 with terms b0 A + a0 B (mode 1); B may be
 * NULL; the operation's rounding errors become the pending error */
TR_NUM_HD void tr_unc_aff_lin(const tr_unc_program *p, const tr_unc_value *a, tr_unc_s alpha, const tr_unc_value *b,
                              tr_unc_s beta, tr_unc_s gamma, int mode, tr_unc_aff *out) {
    out->n = 0; out->top = 0; out->center = tr_unc_s_zero(); out->err = tr_unc_s_zero();
    if ((a->flags & TR_UNC_AFFINE_TOP) || (b && (b->flags & TR_UNC_AFFINE_TOP))) { out->top = 1; return; }
    tr_unc_ctx c = tr_unc_ctx_make(p->carrier, p->precision);
    int32_t i = 0, j = 0, n = 0, na = a->nterms, nb = b ? b->nterms : 0;
    while (i < na || j < nb) {
        int32_t sa = i < na ? a->term[i].sym : INT32_MAX, sb = j < nb ? b->term[j].sym : INT32_MAX, sym = sa < sb ? sa : sb;
        tr_unc_s v = tr_unc_s_zero();
        int have = 0;
        if (sa == sym) { v = tr_unc_s_mul(&c, alpha, tr_unc_s_term(&a->term[i])); i++; have = 1; }
        if (sb == sym) {
            tr_unc_s w = tr_unc_s_mul(&c, beta, tr_unc_s_term(&b->term[j]));
            v = have ? tr_unc_s_add(&c, v, w) : w;
            j++;
        }
        if (!tr_unc_s_iszero(&c, v)) tr_unc_term_set(&out->t[n++], sym, v);
    }
    out->n = n;
    tr_unc_s ctr;
    if (mode == 1) ctr = tr_unc_s_mul(&c, tr_unc_s_center(a), tr_unc_s_center(b));
    else {
        ctr = tr_unc_s_mul(&c, alpha, tr_unc_s_center(a));
        if (b) ctr = tr_unc_s_add(&c, ctr, tr_unc_s_mul(&c, beta, tr_unc_s_center(b)));
        if (!tr_unc_s_iszero(&c, gamma)) ctr = tr_unc_s_add(&c, ctr, gamma);
        else if (c.carrier == TR_UNC_BINARY64 && ctr.lo != ctr.hi) ctr = tr_unc_s_round(&c, ctr.lo, ctr.hi);
    }
    if (c.fail) { out->top = 1; return; }
    out->center = ctr;
    if (c.carrier == TR_UNC_BINARY64) out->err = tr_unc_s_point(c.err);
}
/* the node's form: the pending error and `extra` (>= 0) on the node's symbol, condensed past the rank, collapsed past
 * the error budget */
TR_NUM_HD void tr_unc_aff_finish(const tr_unc_program *p, int32_t node, tr_unc_aff *f, tr_unc_s extra, tr_unc_value *r) {
    r->nterms = 0;
    if (f->top) { r->flags |= TR_UNC_AFFINE_TOP; return; }
    tr_unc_ctx c = tr_unc_ctx_make(p->carrier, 53);
    int32_t fresh = TR_UNC_FRESH + node, n = f->n;
    tr_unc_s add = tr_unc_s_addmag(&c, f->err, extra);
    if (p->carrier == TR_UNC_BINARY64 && f->err.hi > 0) r->flags |= TR_UNC_ROUNDED;
    int32_t keep = p->rank - 1, extra_term = !tr_unc_s_iszero(&c, add);
    if (n + extra_term > p->rank) {                      /* condense: keep the rank - 1 largest |c_k| (ties: lower symbol) */
        int32_t kept[2 * TR_UNC_MAX_TERMS + 2];
        for (int32_t i = 0; i < n; i++) kept[i] = 0;
        for (int32_t k = 0; k < keep; k++) {
            int32_t best = -1;
            for (int32_t i = 0; i < n; i++) {
                if (kept[i]) continue;
                if (best < 0) { best = i; continue; }
                tr_unc_s x = tr_unc_s_mag(&c, tr_unc_s_term(&f->t[i])), y = tr_unc_s_mag(&c, tr_unc_s_term(&f->t[best]));
                if (tr_unc_s_cmp(&c, x, y) > 0) best = i;
            }
            if (best >= 0) kept[best] = 1;
        }
        int32_t m = 0;
        for (int32_t i = 0; i < n; i++) {
            if (kept[i]) f->t[m++] = f->t[i];
            else add = tr_unc_s_addmag(&c, add, tr_unc_s_term(&f->t[i]));
        }
        n = m;
        r->flags |= TR_UNC_CONDENSED;
    }
    if (c.fail) { r->flags |= TR_UNC_AFFINE_TOP | (p->carrier == TR_UNC_EXACT ? TR_UNC_WIDENED : 0); return; }
    if (!tr_unc_s_iszero(&c, add)) {                     /* the node's symbol: after every operand symbol */
        int32_t pos = n;
        while (pos > 0 && f->t[pos - 1].sym > fresh) { f->t[pos] = f->t[pos - 1]; pos--; }
        tr_unc_term_set(&f->t[pos], fresh, add);
        n++;
    }
    if (n > TR_UNC_MAX_TERMS) { r->flags |= TR_UNC_AFFINE_TOP; return; }
    if (p->budget.denominator > 0) {                     /* the error budget on the approximation radius */
        tr_unc_s e = tr_unc_radius(&c, f->t, n, 0), lim = p->carrier == TR_UNC_EXACT ? tr_unc_s_rat(p->budget) : tr_unc_s_point(tr_unc_enclose_lo(p->budget));
        if (!c.fail && tr_unc_s_cmp(&c, e, lim) > 0) {
            tr_unc_s all = tr_unc_radius(&c, f->t, n, 1);
            n = 0;
            if (!tr_unc_s_iszero(&c, all)) { tr_unc_term_set(&f->t[0], fresh, all); n = 1; }
            r->flags |= TR_UNC_BUDGET;
        }
    }
    if (c.fail) { r->flags |= TR_UNC_AFFINE_TOP | (p->carrier == TR_UNC_EXACT ? TR_UNC_WIDENED : 0); return; }
    for (int32_t i = 0; i < n; i++) r->term[i] = f->t[i];
    r->nterms = n;
    r->center = f->center.q; r->dcenter = f->center.lo;
    r->flags &= ~TR_UNC_AFFINE_TOP;
}
/* the range [c0 - rad, c0 + rad] of a finished form (EXACT: *lo, *hi; BINARY64: *dlo, *dhi); 0 when unbounded */
TR_NUM_HD int tr_unc_aff_range(const tr_unc_value *v, int32_t carrier, tr_unc_xb *lo, tr_unc_xb *hi, double *dlo, double *dhi) {
    tr_unc_ctx c = tr_unc_ctx_make(carrier, 53);
    tr_unc_s rad = tr_unc_radius(&c, v->term, v->nterms, 1);
    if (c.fail) return 0;
    if (carrier == TR_UNC_EXACT) {
        tr_num_rational l, h;
        if (!tr_q_sub(v->center, rad.q, &l) || !tr_q_add(v->center, rad.q, &h)) return 0;
        *lo = tr_unc_xb_of(l, 0); *hi = tr_unc_xb_of(h, 0);
        return 1;
    }
    *dlo = tr_add_rd(v->dcenter, -rad.hi); *dhi = tr_add_ru(v->dcenter, rad.hi);
    return *dlo > -INFINITY && *dhi < INFINITY;
}

/* ---------------------------------------------------------------- the reduced product of one numeric value */
TR_NUM_HD void tr_unc_normalize(const tr_unc_program *p, tr_unc_value *v) {
    if (!(v->null & TR_UNC_MAYBE_VALUE)) { tr_unc_set_novalue(v); return; }
    if (!(p->domains & TR_UNC_DOM_AFFINE)) { v->flags |= TR_UNC_AFFINE_TOP; v->nterms = 0; }
    if (!(p->domains & TR_UNC_DOM_CONGRUENCE)) v->flags |= TR_UNC_CONG_TOP;
    int cong = !(v->flags & TR_UNC_CONG_TOP);
    int z0;
    if (p->carrier == TR_UNC_EXACT) {
        tr_unc_xb lo = tr_unc_xlo(v), hi = tr_unc_xhi(v), al, ah;
        double dl, dh;
        if (!(v->flags & TR_UNC_AFFINE_TOP) && (!(p->domains & TR_UNC_DOM_SYMBOLIC) || v->nterms == 0) &&
            tr_unc_aff_range(v, TR_UNC_EXACT, &al, &ah, &dl, &dh)) {
            if (tr_unc_xcmp(al, lo) > 0) lo = al;
            if (tr_unc_xcmp(ah, hi) < 0) hi = ah;
        }
        if (cong && v->cm.numerator == 0) {               /* an exact constant */
            tr_unc_xb c = tr_unc_xb_of(v->cr, 0);
            if (tr_unc_xcmp(c, lo) < 0 || tr_unc_xcmp(c, hi) > 0) { tr_unc_set_novalue(v); return; }
            lo = hi = c;
        } else if (cong) {                                /* the bounds moved onto the class r + m Z */
            tr_num_rational t, k, s;
            int64_t ce;
            if (!lo.inf && tr_q_sub(lo.v, v->cr, &t) && tr_q_div(t, v->cm, &k) && tr_unc_qceil(k, &ce) &&
                tr_q_mul(v->cm, tr_q_int(ce), &s) && tr_q_add(v->cr, s, &t)) lo = tr_unc_xb_of(t, 0);
            if (!hi.inf && tr_q_sub(hi.v, v->cr, &t) && tr_q_div(t, v->cm, &k) &&
                tr_q_mul(v->cm, tr_q_int(tr_unc_qfloor(k)), &s) && tr_q_add(v->cr, s, &t)) hi = tr_unc_xb_of(t, 0);
        }
        if (tr_unc_xcmp(lo, hi) > 0) { tr_unc_set_novalue(v); return; }
        tr_unc_set_lo(v, lo); tr_unc_set_hi(v, hi);
        v->dlo = lo.inf ? -INFINITY : tr_unc_enclose_lo(lo.v);
        v->dhi = hi.inf ? INFINITY : tr_unc_enclose_hi(hi.v);
        if (!lo.inf && !hi.inf && tr_unc_qeq(lo.v, hi.v)) {   /* a point: the form and the class are exact */
            if (cong && !tr_unc_cong_contains(v->cr, v->cm, lo.v)) { tr_unc_set_novalue(v); return; }
            if (p->domains & TR_UNC_DOM_AFFINE) { v->center = lo.v; v->nterms = 0; v->flags &= ~TR_UNC_AFFINE_TOP; }
            if (p->domains & TR_UNC_DOM_CONGRUENCE) { v->cr = lo.v; v->cm = tr_q_int(0); v->flags &= ~TR_UNC_CONG_TOP; cong = 1; }
        }
        z0 = tr_unc_xsign(lo) <= 0 && tr_unc_xsign(hi) >= 0;
        if (z0 && cong && !tr_unc_cong_contains(v->cr, v->cm, tr_q_int(0))) z0 = 0;
        v->zero = !z0 ? TR_UNC_NONZERO : (!lo.inf && !hi.inf && lo.v.numerator == 0 && hi.v.numerator == 0 ? TR_UNC_ZERO : TR_UNC_ZERO_UNKNOWN);
        return;
    }
    double lo = v->dlo, hi = v->dhi, al = 0, ah = 0;
    tr_unc_xb xl, xh;
    if (!(v->flags & TR_UNC_AFFINE_TOP) && (!(p->domains & TR_UNC_DOM_SYMBOLIC) || v->nterms == 0) &&
        tr_unc_aff_range(v, TR_UNC_BINARY64, &xl, &xh, &al, &ah)) {
        lo = tr_unc_dmax(lo, al); hi = tr_unc_dmin(hi, ah);
    }
    if (cong && v->cm.numerator == 0) {                   /* an exact constant: within its enclosure */
        lo = tr_unc_dmax(lo, tr_unc_enclose_lo(v->cr)); hi = tr_unc_dmin(hi, tr_unc_enclose_hi(v->cr));
    }
    if (lo > hi) { tr_unc_set_novalue(v); return; }
    v->dlo = tr_unc_nz(lo); v->dhi = tr_unc_nz(hi);
    v->flags = (v->flags & ~(TR_UNC_LO_INF | TR_UNC_HI_INF)) | (lo == -INFINITY ? TR_UNC_LO_INF : 0) | (hi == INFINITY ? TR_UNC_HI_INF : 0);
    if (lo == hi && (p->domains & TR_UNC_DOM_AFFINE)) { v->dcenter = v->dlo; v->nterms = 0; v->flags &= ~TR_UNC_AFFINE_TOP; }
    z0 = lo <= 0 && hi >= 0;
    if (z0 && cong && !tr_unc_cong_contains(v->cr, v->cm, tr_q_int(0))) z0 = 0;
    v->zero = !z0 ? TR_UNC_NONZERO : (lo == 0 && hi == 0 ? TR_UNC_ZERO : TR_UNC_ZERO_UNKNOWN);
}

/* ---------------------------------------------------------------- convex approximations */
/* |x| or x^2 on [l, u] (finite) as alpha x + zeta +- delta: alpha a secant slope, [m1, m2] the exact (EXACT) or
 * rigorously enclosed (BINARY64) range of f(x) - alpha x on [l, u]; 0 when it cannot be formed */
TR_NUM_HD int tr_unc_convex(int32_t carrier, int abs, tr_unc_s l, tr_unc_s u, tr_unc_s *alpha, tr_unc_s *zeta, tr_unc_s *delta) {
    if (carrier == TR_UNC_EXACT) {
        tr_num_rational al, m1, m2, s, w;
        if (abs) {                                       /* l < 0 < u: slope (u + l) / (u - l); f - alpha x at l, u, 0 */
            tr_num_rational one = tr_q_int(1), t, nl, fl, fu;
            if (!tr_q_add(u.q, l.q, &s) || !tr_q_sub(u.q, l.q, &w) || !tr_q_div(s, w, &al) || !tr_q_add(one, al, &t) ||
                !tr_unc_qneg(l.q, &nl) || !tr_q_mul(nl, t, &fl) || !tr_q_sub(one, al, &t) || !tr_q_mul(u.q, t, &fu)) return 0;
            m1 = tr_q_int(0); m2 = tr_q_int(0);
            if (tr_q_cmp(fl, m1) < 0) m1 = fl;
            if (tr_q_cmp(fu, m1) < 0) m1 = fu;
            if (tr_q_cmp(fl, m2) > 0) m2 = fl;
            if (tr_q_cmp(fu, m2) > 0) m2 = fu;
        } else {                                         /* slope l + u; f - alpha x = -l u at the ends, -mid^2 inside */
            tr_num_rational mid, lu;
            if (!tr_q_add(l.q, u.q, &al) || !tr_q_div(al, tr_q_int(2), &mid) || !tr_q_mul(l.q, u.q, &lu) ||
                !tr_unc_qneg(lu, &m2) || !tr_q_mul(mid, mid, &s) || !tr_unc_qneg(s, &m1)) return 0;
        }
        tr_num_rational half = tr_unc_qv(1, 2), z, d;
        if (!tr_q_add(m1, m2, &s) || !tr_q_mul(s, half, &z) || !tr_q_sub(m2, m1, &w) || !tr_q_mul(w, half, &d)) return 0;
        *alpha = tr_unc_s_zero(); alpha->q = al;
        *zeta = tr_unc_s_zero(); zeta->q = z;
        *delta = tr_unc_s_zero(); delta->q = d;
        return 1;
    }
    double L = l.lo, U = u.hi, al, m1, m2;
    if (!(L > -INFINITY) || !(U < INFINITY)) return 0;
    if (abs) {
        double s = tr_add_rd(U, L), w = tr_add_ru(U, -L);
        al = w > 0 ? s / w : 0.0;                         /* any slope is rigorous below; clamped to [-1, 1] */
        if (!(al >= -1.0)) al = -1.0;
        if (!(al <= 1.0)) al = 1.0;
        /* f(l) - al l = -l (1 + al), f(u) - al u = u (1 - al), and 0 at the kink */
        tr_math_interval NL = tr_interval_point(-L), UU = tr_interval_point(U), T1, T2;
        T1.lo = tr_add_rd(1.0, al); T1.hi = tr_add_ru(1.0, al); T2.lo = tr_add_rd(1.0, -al); T2.hi = tr_add_ru(1.0, -al);
        tr_math_interval FL = tr_interval_mul(NL, T1), FU = tr_interval_mul(UU, T2);
        m1 = tr_unc_dmin(0.0, tr_unc_dmin(FL.lo, FU.lo));
        m2 = tr_unc_dmax(0.0, tr_unc_dmax(FL.hi, FU.hi));
    } else {
        al = tr_add_rd(L, U);                             /* g(x) = x^2 - al x: maxima at the ends (convex) */
        m2 = -INFINITY;
        double xs[2]; xs[0] = L; xs[1] = U;
        for (int i = 0; i < 2; i++) {
            tr_math_interval X = tr_interval_point(xs[i]), D;
            D.lo = tr_add_rd(xs[i], -al); D.hi = tr_add_ru(xs[i], -al);
            m2 = tr_unc_dmax(m2, tr_interval_mul(X, D).hi);
        }
        double h = tr_unc_half_dir(fabs(al), 1);          /* the global minimum of g is -(al/2)^2 */
        m1 = -tr_mul_dir(h, h, 1);
    }
    if (!(m1 > -INFINITY) || !(m2 < INFINITY)) return 0;
    double z = tr_unc_half_dir(tr_add_rd(m1, m2), 0);
    if (z < m1) z = m1;
    if (z > m2) z = m2;
    *alpha = tr_unc_s_point(al);
    *zeta = tr_unc_s_point(z);
    *delta = tr_unc_s_point(tr_unc_dmax(tr_add_ru(m2, -z), tr_add_ru(z, -m1)));
    return 1;
}
/* the operand's reduced interval as scalars (EXACT q; BINARY64 points); 0 when unbounded */
TR_NUM_HD int tr_unc_bounds(const tr_unc_program *p, const tr_unc_value *a, tr_unc_s *l, tr_unc_s *u) {
    if (a->flags & (TR_UNC_LO_INF | TR_UNC_HI_INF)) return 0;
    if (p->carrier == TR_UNC_EXACT) { *l = tr_unc_s_zero(); l->q = a->lo; *u = tr_unc_s_zero(); u->q = a->hi; }
    else { *l = tr_unc_s_point(a->dlo); *u = tr_unc_s_point(a->dhi); }
    return 1;
}

/* the input of column j as a value */
TR_NUM_HD void tr_unc_from_input(const tr_unc_program *p, int32_t j, const tr_unc_input *in, tr_unc_value *v) {
    tr_unc_value_clear(v, TR_UNC_NUMERIC);
    v->null = (in->present < in->rows ? TR_UNC_MAYBE_NULL : 0) | (in->present > 0 ? TR_UNC_MAYBE_VALUE : 0);
    if (!in->present) { tr_unc_set_novalue(v); return; }
    int64_t ten = tr_unc_pow10(in->scale);
    tr_num_rational lo, hi;
    tr_q_make(in->min, ten, &lo); tr_q_make(in->max, ten, &hi);
    v->cr = lo;
    if (!(in->modulus <= (uint64_t)INT64_MAX && tr_q_make((int64_t)in->modulus, ten, &v->cm) && tr_unc_cong_norm(&v->cr, v->cm)))
        v->flags |= TR_UNC_CONG_TOP;
    if (p->carrier == TR_UNC_EXACT) {
        v->lo = lo; v->hi = hi;
        tr_num_rational c, r;
        if (tr_q_make128((__int128)in->min + in->max, (__int128)2 * ten, &c) && tr_q_make128((__int128)in->max - in->min, (__int128)2 * ten, &r)) {
            v->center = c;
            if (r.numerator) { tr_unc_s s = tr_unc_s_zero(); s.q = r; tr_unc_term_set(&v->term[0], j, s); v->nterms = 1; }
        } else v->flags |= TR_UNC_AFFINE_TOP | TR_UNC_WIDENED;
    } else {
        double dl = tr_unc_enclose_lo(lo), dh = tr_unc_enclose_hi(hi);
        v->dlo = dl; v->dhi = dh;
        double c = tr_unc_half_dir(tr_add_rd(dl, dh), 0);  /* any center works: the radius covers both ends */
        if (c < dl) c = dl;
        if (c > dh) c = dh;
        if (p->precision < 53) c = tr_unc_trunc(c, p->precision);
        double r = tr_unc_dmax(tr_add_ru(dh, -c), tr_add_ru(c, -dl));
        if (p->precision < 53) {                          /* a radius >= r with p significant bits */
            double t = tr_unc_trunc(r, p->precision);
            if (t < r) { union { double d; uint64_t u; } w; w.d = t; w.u += (uint64_t)1 << (53 - p->precision); t = w.d; }
            r = t;
        }
        v->dcenter = tr_unc_nz(c);
        if (r > 0) { tr_unc_term_set(&v->term[0], j, tr_unc_s_point(r)); v->nterms = 1; }
    }
    tr_unc_normalize(p, v);
}

/* congruence results (exact rationals; any failure leaves the result unknown) */
TR_NUM_HD void tr_unc_cong_set(tr_unc_value *r, int ok, tr_num_rational cr, tr_num_rational cm) {
    if (ok && tr_unc_qsign(cm) < 0 && !tr_unc_qneg(cm, &cm)) ok = 0;
    if (ok && !tr_unc_cong_norm(&cr, cm)) ok = 0;
    if (!ok) { r->flags |= TR_UNC_CONG_TOP; return; }
    r->flags &= ~TR_UNC_CONG_TOP; r->cr = cr; r->cm = cm;
}
TR_NUM_HD int tr_unc_cong_top(const tr_unc_value *a) { return (a->flags & TR_UNC_CONG_TOP) != 0; }
/* the join of two classes (the value is one of them): r1 + gcd(m1, m2, r1 - r2) Z */
TR_NUM_HD void tr_unc_cong_join(const tr_unc_value *a, const tr_unc_value *b, tr_unc_value *r) {
    if (tr_unc_cong_top(a) || tr_unc_cong_top(b)) { r->flags |= TR_UNC_CONG_TOP; return; }
    tr_num_rational d, g, h;
    int ok = tr_q_sub(a->cr, b->cr, &d) && tr_unc_qgcd(a->cm, b->cm, &g) && tr_unc_qgcd(g, d, &h);
    tr_unc_cong_set(r, ok, a->cr, h);
}
/* the range of the form a - b (no storage): EXACT [*lo, *hi], BINARY64 [*dlo, *dhi]; 0 when unknown */
TR_NUM_HD int tr_unc_diff_range(const tr_unc_program *p, const tr_unc_value *a, const tr_unc_value *b, tr_unc_xb *lo,
                                tr_unc_xb *hi, double *dlo, double *dhi) {
    if ((a->flags & TR_UNC_AFFINE_TOP) || (b->flags & TR_UNC_AFFINE_TOP)) return 0;
    tr_unc_ctx c = tr_unc_ctx_make(p->carrier, 53);
    tr_unc_s one = tr_unc_s_rat(tr_q_int(1)), mone = tr_unc_s_neg(&c, one), rad = tr_unc_s_zero();
    int32_t i = 0, j = 0, na = a->nterms, nb = b->nterms;
    while (i < na || j < nb) {
        int32_t sa = i < na ? a->term[i].sym : INT32_MAX, sb = j < nb ? b->term[j].sym : INT32_MAX, sym = sa < sb ? sa : sb;
        tr_unc_s v = tr_unc_s_zero();
        if (sa == sym) { v = tr_unc_s_term(&a->term[i]); i++; }
        if (sb == sym) {
            tr_unc_s w = tr_unc_s_mul(&c, mone, tr_unc_s_term(&b->term[j]));
            if (c.carrier == TR_UNC_EXACT) { if (!tr_q_add(v.q, w.q, &v.q)) c.fail = 1; }
            else v = tr_unc_s_iv(tr_add_rd(v.lo, w.lo), tr_add_ru(v.hi, w.hi));
            j++;
        }
        rad = tr_unc_s_addmag(&c, rad, v);
    }
    if (c.carrier == TR_UNC_EXACT) {
        tr_num_rational d0, l, h;
        if (c.fail || !tr_q_sub(a->center, b->center, &d0) || !tr_q_sub(d0, rad.q, &l) || !tr_q_add(d0, rad.q, &h)) return 0;
        *lo = tr_unc_xb_of(l, 0); *hi = tr_unc_xb_of(h, 0);
        return 1;
    }
    double c0l = tr_add_rd(a->dcenter, -b->dcenter), c0h = tr_add_ru(a->dcenter, -b->dcenter);
    *dlo = tr_add_rd(c0l, -rad.hi); *dhi = tr_add_ru(c0h, rad.hi);
    return !c.fail && *dlo > -INFINITY && *dhi < INFINITY;
}

/* ---------------------------------------------------------------- one numeric operation (operands evaluated) */
TR_NUM_HD void tr_unc_numeric(const tr_unc_program *p, int32_t k, const tr_unc_node *nd, const tr_unc_value *a,
                              const tr_unc_value *b, tr_unc_value *r) {
    tr_unc_ctx c = tr_unc_ctx_make(p->carrier, p->precision);
    tr_unc_s z = tr_unc_s_zero(), one = tr_unc_s_rat(tr_q_int(1)), mone = tr_unc_s_rat(tr_q_int(-1));
    int32_t na = a->null, nb = b ? b->null : TR_UNC_MAYBE_VALUE;
    tr_unc_value_clear(r, TR_UNC_NUMERIC);
    r->null = ((na | nb) & TR_UNC_MAYBE_NULL) | ((na & nb) & TR_UNC_MAYBE_VALUE);
    r->flags = TR_UNC_CONG_TOP | TR_UNC_AFFINE_TOP;
    if (!(r->null & TR_UNC_MAYBE_VALUE)) { tr_unc_set_novalue(r); return; }
    int32_t widened = 0;
    tr_unc_xb lo = tr_unc_xb_of(tr_q_int(0), -1), hi = tr_unc_xb_of(tr_q_int(0), 1);
    tr_math_interval A, B, R;
    A.lo = a->dlo; A.hi = a->dhi;
    if (b) { B.lo = b->dlo; B.hi = b->dhi; } else B = A;
    R.lo = -INFINITY; R.hi = INFINITY;
    tr_num_rational t1, t2, t3;
    tr_unc_aff f;
    tr_unc_s extra = z;
    f.top = 1; f.n = 0;
    switch (nd->op) {
    case TR_UNC_ADD: case TR_UNC_SUB: {
        int sub = nd->op == TR_UNC_SUB;
        if (sub && a == b && (p->domains & TR_UNC_DOM_AFFINE)) {   /* a node minus itself (correlation): exactly 0 */
            lo = hi = tr_unc_xb_of(tr_q_int(0), 0);
            R.lo = R.hi = 0.0;
            f.top = 0; f.n = 0; f.center = z; f.err = z;
            tr_unc_cong_set(r, 1, tr_q_int(0), tr_q_int(0));
            break;
        }
        if (p->carrier == TR_UNC_EXACT) {
            tr_unc_xb bl = sub ? tr_unc_xneg(tr_unc_xhi(b), 0, &widened) : tr_unc_xlo(b);
            tr_unc_xb bh = sub ? tr_unc_xneg(tr_unc_xlo(b), 1, &widened) : tr_unc_xhi(b);
            lo = tr_unc_xadd(tr_unc_xlo(a), bl, 0, &widened); hi = tr_unc_xadd(tr_unc_xhi(a), bh, 1, &widened);
        } else R = sub ? tr_interval_sub(A, B) : tr_interval_add(A, B);
        tr_unc_aff_lin(p, a, one, b, sub ? mone : one, z, 0, &f);
        if (!tr_unc_cong_top(a) && !tr_unc_cong_top(b)) {
            int ok = (sub ? tr_q_sub(a->cr, b->cr, &t1) : tr_q_add(a->cr, b->cr, &t1)) && tr_unc_qgcd(a->cm, b->cm, &t2);
            tr_unc_cong_set(r, ok, t1, t2);
        }
        break;
    }
    case TR_UNC_NEG:
        if (p->carrier == TR_UNC_EXACT) { lo = tr_unc_xneg(tr_unc_xhi(a), 0, &widened); hi = tr_unc_xneg(tr_unc_xlo(a), 1, &widened); }
        else R = tr_interval_neg(A);
        tr_unc_aff_lin(p, a, mone, NULL, z, z, 0, &f);
        if (!tr_unc_cong_top(a)) tr_unc_cong_set(r, tr_unc_qneg(a->cr, &t1), t1, a->cm);
        break;
    case TR_UNC_MUL: {
        if (a->zero == TR_UNC_ZERO || b->zero == TR_UNC_ZERO) {   /* an exact zero factor: exactly 0 (NULL aside) */
            lo = hi = tr_unc_xb_of(tr_q_int(0), 0);
            R.lo = R.hi = 0.0;
            f.top = 0; f.n = 0; f.center = z; f.err = z;
            tr_unc_cong_set(r, 1, tr_q_int(0), tr_q_int(0));
            break;
        }
        if (p->carrier == TR_UNC_EXACT) tr_unc_xmul_iv(tr_unc_xlo(a), tr_unc_xhi(a), tr_unc_xlo(b), tr_unc_xhi(b), &lo, &hi, &widened);
        else R = tr_interval_mul(A, B);
        tr_unc_aff_lin(p, a, tr_unc_s_center(b), b, tr_unc_s_center(a), z, 1, &f);
        if (!f.top) {                                     /* the remainder rad(A) rad(B) */
            tr_unc_s ra = tr_unc_radius(&c, a->term, a->nterms, 1), rb = tr_unc_radius(&c, b->term, b->nterms, 1);
            if (p->carrier == TR_UNC_EXACT) { if (!tr_q_mul(ra.q, rb.q, &extra.q)) f.top = 1; }
            else extra = tr_unc_s_point(tr_mul_dir(ra.hi, rb.hi, 1));
            if (c.fail) f.top = 1;
        }
        if (!tr_unc_cong_top(a) && !tr_unc_cong_top(b)) {   /* (ra rb, gcd(ma mb, ma rb, mb ra)) */
            tr_num_rational g, h;
            int ok = tr_q_mul(a->cr, b->cr, &t1) && tr_q_mul(a->cm, b->cm, &t2) && tr_q_mul(a->cm, b->cr, &t3) &&
                     tr_unc_qgcd(t2, t3, &g) && tr_q_mul(b->cm, a->cr, &t3) && tr_unc_qgcd(g, t3, &h);
            tr_unc_cong_set(r, ok, t1, h);
        }
        break;
    }
    case TR_UNC_DIV: {
        tr_num_rational inv = tr_q_int(0);
        tr_q_div(tr_q_int(1), nd->k, &inv);              /* k != 0 (program check): 1 / k fits int64 */
        if (p->carrier == TR_UNC_EXACT) {
            tr_unc_xb kb = tr_unc_xb_of(inv, 0);
            tr_unc_xmul_iv(tr_unc_xlo(a), tr_unc_xhi(a), kb, kb, &lo, &hi, &widened);
        } else R = tr_interval_div(A, tr_unc_enclose(nd->k));
        tr_unc_aff_lin(p, a, tr_unc_s_rat(inv), NULL, z, z, 0, &f);
        if (!tr_unc_cong_top(a)) {
            tr_num_rational ak;
            int ok = tr_q_div(a->cr, nd->k, &t1) && tr_unc_qabs(nd->k, &ak) && tr_q_div(a->cm, ak, &t2);
            tr_unc_cong_set(r, ok, t1, t2);
        }
        break;
    }
    case TR_UNC_MIN: case TR_UNC_MAX: {
        int mx = nd->op == TR_UNC_MAX;
        if (p->carrier == TR_UNC_EXACT) {
            tr_unc_xb al = tr_unc_xlo(a), ah = tr_unc_xhi(a), bl = tr_unc_xlo(b), bh = tr_unc_xhi(b);
            if (mx) { lo = tr_unc_xcmp(al, bl) >= 0 ? al : bl; hi = tr_unc_xcmp(ah, bh) >= 0 ? ah : bh; }
            else { lo = tr_unc_xcmp(al, bl) <= 0 ? al : bl; hi = tr_unc_xcmp(ah, bh) <= 0 ? ah : bh; }
        } else {
            R.lo = mx ? tr_unc_dmax(A.lo, B.lo) : tr_unc_dmin(A.lo, B.lo);
            R.hi = mx ? tr_unc_dmax(A.hi, B.hi) : tr_unc_dmin(A.hi, B.hi);
        }
        /* d = a - b on every row: its interval difference intersected with the form's range */
        tr_unc_s dl, dh;
        int finite;
        if (p->carrier == TR_UNC_EXACT) {
            int32_t w = 0;
            tr_unc_xb xl = tr_unc_xadd(tr_unc_xlo(a), tr_unc_xneg(tr_unc_xhi(b), 0, &w), 0, &w);
            tr_unc_xb xh = tr_unc_xadd(tr_unc_xhi(a), tr_unc_xneg(tr_unc_xlo(b), 1, &w), 1, &w), fl, fh;
            double u1, u2;
            if (tr_unc_diff_range(p, a, b, &fl, &fh, &u1, &u2) &&
                (!(p->domains & TR_UNC_DOM_SYMBOLIC) || tr_unc_xcmp(fl, fh) == 0)) {   /* symbolic: exact differences */
                if (tr_unc_xcmp(fl, xl) > 0) xl = fl;
                if (tr_unc_xcmp(fh, xh) < 0) xh = fh;
            }
            finite = !xl.inf && !xh.inf;
            dl = tr_unc_s_zero(); dl.q = xl.v; dh = tr_unc_s_zero(); dh.q = xh.v;
            if (!xh.inf && tr_unc_qsign(xh.v) <= 0) finite = 2;            /* a <= b everywhere */
            else if (!xl.inf && tr_unc_qsign(xl.v) >= 0) finite = 3;       /* a >= b everywhere */
        } else {
            tr_math_interval D = tr_interval_sub(A, B);
            double u1 = 0, u2 = 0;
            tr_unc_xb fl, fh;
            if (tr_unc_diff_range(p, a, b, &fl, &fh, &u1, &u2) && (!(p->domains & TR_UNC_DOM_SYMBOLIC) || u1 == u2)) {
                D.lo = tr_unc_dmax(D.lo, u1); D.hi = tr_unc_dmin(D.hi, u2);
            }
            finite = D.lo > -INFINITY && D.hi < INFINITY;
            dl = tr_unc_s_point(D.lo); dh = tr_unc_s_point(D.hi);
            if (D.hi <= 0) finite = 2;
            else if (D.lo >= 0) finite = 3;
        }
        if (finite >= 2) {                                /* one side is the result on every row */
            const tr_unc_value *pick = (finite == 2) != mx ? a : b;
            tr_unc_aff_lin(p, pick, one, NULL, z, z, 0, &f);
            if (!tr_unc_cong_top(pick)) tr_unc_cong_set(r, 1, pick->cr, pick->cm);
        } else {
            /* min = (a + b)/2 - |d|/2, max = (a + b)/2 + |d|/2 with |d| = alpha d + zeta +- delta on [dl, dh]:
             * ((1 -+ alpha)/2) a + ((1 +- alpha)/2) b -+ zeta / 2, pending error delta / 2 */
            tr_unc_s al, ze, de;
            if (finite && tr_unc_convex(p->carrier, 1, dl, dh, &al, &ze, &de)) {
                tr_unc_s ca, cb, g;
                if (p->carrier == TR_UNC_EXACT) {
                    tr_num_rational h = tr_unc_qv(1, 2), s1, s2;
                    int ok = tr_q_sub(tr_q_int(1), al.q, &s1) && tr_q_add(tr_q_int(1), al.q, &s2);
                    ca = tr_unc_s_zero(); cb = tr_unc_s_zero(); g = tr_unc_s_zero();
                    ok = ok && tr_q_mul(mx ? s2 : s1, h, &ca.q) && tr_q_mul(mx ? s1 : s2, h, &cb.q) && tr_q_mul(ze.q, h, &g.q);
                    if (ok && !mx) ok = tr_unc_qneg(g.q, &g.q);
                    ok = ok && tr_q_mul(de.q, h, &extra.q);
                    if (ok) tr_unc_aff_lin(p, a, ca, b, cb, g, 0, &f);
                } else {
                    double s1l = tr_add_rd(1.0, -al.hi), s1h = tr_add_ru(1.0, -al.lo), s2l = tr_add_rd(1.0, al.lo), s2h = tr_add_ru(1.0, al.hi);
                    tr_unc_s c1 = tr_unc_s_iv(tr_unc_half_dir(s1l, 0), tr_unc_half_dir(s1h, 1));
                    tr_unc_s c2 = tr_unc_s_iv(tr_unc_half_dir(s2l, 0), tr_unc_half_dir(s2h, 1));
                    ca = mx ? c2 : c1; cb = mx ? c1 : c2;
                    g = tr_unc_s_iv(tr_unc_half_dir(ze.lo, 0), tr_unc_half_dir(ze.hi, 1));
                    if (!mx) g = tr_unc_s_iv(tr_unc_nz(-g.hi), tr_unc_nz(-g.lo));
                    extra = tr_unc_s_point(tr_unc_half_dir(de.hi, 1));
                    tr_unc_aff_lin(p, a, ca, b, cb, g, 0, &f);
                }
            }
            tr_unc_cong_join(a, b, r);
        }
        break;
    }
    case TR_UNC_ABS: {
        if (p->carrier == TR_UNC_EXACT) {
            tr_unc_xb al = tr_unc_xlo(a), ah = tr_unc_xhi(a);
            if (tr_unc_xsign(al) >= 0) { lo = al; hi = ah; }
            else if (tr_unc_xsign(ah) <= 0) { lo = tr_unc_xneg(ah, 0, &widened); hi = tr_unc_xneg(al, 1, &widened); }
            else { tr_unc_xb nl = tr_unc_xneg(al, 1, &widened); lo = tr_unc_xb_of(tr_q_int(0), 0); hi = tr_unc_xcmp(nl, ah) >= 0 ? nl : ah; }
        } else R = tr_interval_abs(A);
        int pos = p->carrier == TR_UNC_EXACT ? (!(a->flags & TR_UNC_LO_INF) && tr_unc_qsign(a->lo) >= 0) : a->dlo >= 0;
        int neg = p->carrier == TR_UNC_EXACT ? (!(a->flags & TR_UNC_HI_INF) && tr_unc_qsign(a->hi) <= 0) : a->dhi <= 0;
        if (pos) tr_unc_aff_lin(p, a, one, NULL, z, z, 0, &f);
        else if (neg) tr_unc_aff_lin(p, a, mone, NULL, z, z, 0, &f);
        else {
            tr_unc_s l, u, al, ze, de;
            if (tr_unc_bounds(p, a, &l, &u) && tr_unc_convex(p->carrier, 1, l, u, &al, &ze, &de)) {
                tr_unc_aff_lin(p, a, al, NULL, z, ze, 0, &f);
                extra = de;
            }
        }
        if (!tr_unc_cong_top(a)) {
            if (pos) tr_unc_cong_set(r, 1, a->cr, a->cm);
            else if (neg) tr_unc_cong_set(r, tr_unc_qneg(a->cr, &t1), t1, a->cm);
            else {                                        /* +-x: r + gcd(m, 2 r) Z */
                int ok = tr_q_mul(a->cr, tr_q_int(2), &t1) && tr_unc_qgcd(a->cm, t1, &t2);
                tr_unc_cong_set(r, ok, a->cr, t2);
            }
        }
        break;
    }
    case TR_UNC_SQR: {
        if (p->carrier == TR_UNC_EXACT) {
            tr_unc_xb al = tr_unc_xlo(a), ah = tr_unc_xhi(a), pl, ph, ql, qh;
            tr_unc_xmul(al, al, &pl, &ph, &widened);
            tr_unc_xmul(ah, ah, &ql, &qh, &widened);
            if (tr_unc_xsign(al) >= 0) { lo = pl; hi = qh; }
            else if (tr_unc_xsign(ah) <= 0) { lo = ql; hi = ph; }
            else { lo = tr_unc_xb_of(tr_q_int(0), 0); hi = tr_unc_xcmp(ph, qh) >= 0 ? ph : qh; }
        } else R = tr_interval_sqr(A);
        tr_unc_s l, u, al, ze, de;
        if (tr_unc_bounds(p, a, &l, &u) && tr_unc_convex(p->carrier, 0, l, u, &al, &ze, &de)) {
            tr_unc_aff_lin(p, a, al, NULL, z, ze, 0, &f);
            extra = de;
        }
        if (!tr_unc_cong_top(a)) {                        /* (r^2, gcd(2 r m, m^2)) */
            int ok = tr_q_mul(a->cr, a->cr, &t1) && tr_q_mul(a->cr, a->cm, &t2) && tr_q_mul(t2, tr_q_int(2), &t2) &&
                     tr_q_mul(a->cm, a->cm, &t3) && tr_unc_qgcd(t2, t3, &t3);
            tr_unc_cong_set(r, ok, t1, t3);
        }
        break;
    }
    case TR_UNC_MOD: {
        /* floor(x / k) constant on the interval: an exact shift; else [0, k) */
        tr_num_rational K = nd->k, q, nk;
        int shift = 0;
        int64_t fl = 0;
        if (!(a->flags & (TR_UNC_LO_INF | TR_UNC_HI_INF))) {
            if (p->carrier == TR_UNC_EXACT) {
                tr_num_rational ql, qh;
                if (tr_q_div(a->lo, K, &ql) && tr_q_div(a->hi, K, &qh) && tr_unc_qfloor(ql) == tr_unc_qfloor(qh)) { shift = 1; fl = tr_unc_qfloor(ql); }
            } else {
                tr_math_interval Q = tr_interval_div(A, tr_unc_enclose(K));
                if (Q.lo > -9.0e18 && Q.hi < 9.0e18 && floor(Q.lo) == floor(Q.hi)) { shift = 1; fl = (int64_t)floor(Q.lo); }
            }
        }
        if (shift && tr_q_mul(K, tr_q_int(fl), &nk) && tr_unc_qneg(nk, &q)) {   /* x - fl k */
            if (p->carrier == TR_UNC_EXACT) {
                lo = tr_unc_xadd(tr_unc_xlo(a), tr_unc_xb_of(q, 0), 0, &widened);
                hi = tr_unc_xadd(tr_unc_xhi(a), tr_unc_xb_of(q, 0), 1, &widened);
            } else {
                R = tr_interval_add(A, tr_unc_enclose(q));
                R.lo = tr_unc_dmax(R.lo, 0.0); R.hi = tr_unc_dmin(R.hi, tr_unc_enclose_hi(K));
            }
            tr_unc_aff_lin(p, a, one, NULL, z, tr_unc_s_rat(q), 0, &f);
            if (!tr_unc_cong_top(a)) tr_unc_cong_set(r, tr_q_add(a->cr, q, &t1), t1, a->cm);
        } else {
            if (!tr_unc_cong_top(a)) { int ok = tr_unc_qgcd(a->cm, K, &t2); tr_unc_cong_set(r, ok, a->cr, t2); }
            tr_num_rational top = K;
            if (!tr_unc_cong_top(r) && r->cm.numerator) {   /* the largest element of r + m Z below k */
                tr_num_rational d, mm;
                if (tr_q_sub(K, r->cr, &d) && tr_unc_qmod(d, r->cm, &mm) && !tr_q_sub(K, mm.numerator ? mm : r->cm, &top)) top = K;
            }
            if (p->carrier == TR_UNC_EXACT) { lo = tr_unc_xb_of(tr_q_int(0), 0); hi = tr_unc_xb_of(top, 0); }
            else { R.lo = 0.0; R.hi = tr_unc_enclose_hi(top); }
            /* the form: the node's symbol around [0, top] */
            f.top = 0; f.n = 0; f.err = z;
            if (p->carrier == TR_UNC_EXACT) {
                tr_num_rational h2;
                if (tr_q_mul(top, tr_unc_qv(1, 2), &h2)) { f.center = z; f.center.q = h2; extra = z; extra.q = h2; }
                else f.top = 1;
            } else {
                double hh = tr_unc_half_dir(tr_unc_enclose_hi(top), 1);
                f.center = tr_unc_s_point(hh); extra = tr_unc_s_point(hh);
            }
        }
        break;
    }
    default:
        break;
    }
    tr_unc_aff_finish(p, k, &f, extra, r);
    if (p->carrier == TR_UNC_EXACT) { tr_unc_set_lo(r, lo); tr_unc_set_hi(r, hi); }
    else {
        r->dlo = R.lo; r->dhi = R.hi;
        r->flags = (r->flags & ~(TR_UNC_LO_INF | TR_UNC_HI_INF)) | (R.lo == -INFINITY ? TR_UNC_LO_INF : 0) | (R.hi == INFINITY ? TR_UNC_HI_INF : 0);
    }
    if (widened) r->flags |= TR_UNC_WIDENED;
    tr_unc_normalize(p, r);
}

/* ---------------------------------------------------------------- boolean nodes (SQL three-valued outcome sets) */
TR_NUM_HD int32_t tr_unc_and3(int32_t x, int32_t y) {
    int32_t r = 0;
    if ((x & TR_UNC_FALSE) || (y & TR_UNC_FALSE)) r |= TR_UNC_FALSE;
    if ((x & TR_UNC_TRUE) && (y & TR_UNC_TRUE)) r |= TR_UNC_TRUE;
    if (((x & TR_UNC_NULL) && (y & (TR_UNC_TRUE | TR_UNC_NULL))) || ((y & TR_UNC_NULL) && (x & (TR_UNC_TRUE | TR_UNC_NULL)))) r |= TR_UNC_NULL;
    return r;
}
TR_NUM_HD int32_t tr_unc_not3(int32_t x) {
    return ((x & TR_UNC_TRUE) ? TR_UNC_FALSE : 0) | ((x & TR_UNC_FALSE) ? TR_UNC_TRUE : 0) | (x & TR_UNC_NULL);
}
TR_NUM_HD void tr_unc_boolean_set(tr_unc_value *r, int32_t outcomes) {
    tr_unc_value_clear(r, TR_UNC_BOOLEAN);
    r->outcomes = outcomes;
    r->null = ((outcomes & TR_UNC_NULL) ? TR_UNC_MAYBE_NULL : 0) | ((outcomes & (TR_UNC_TRUE | TR_UNC_FALSE)) ? TR_UNC_MAYBE_VALUE : 0);
    r->flags = TR_UNC_AFFINE_TOP | TR_UNC_CONG_TOP;
}
/* a comparison a op b on d = a - b (its form cancels correlated sides; its class decides equalities) */
TR_NUM_HD void tr_unc_compare(const tr_unc_program *p, int32_t k, const tr_unc_node *nd, const tr_unc_value *a,
                              const tr_unc_value *b, tr_unc_value *d, tr_unc_value *r) {
    int32_t maybe_value = (a->null & b->null) & TR_UNC_MAYBE_VALUE, maybe_null = (a->null | b->null) & TR_UNC_MAYBE_NULL;
    int32_t out = maybe_null ? TR_UNC_NULL : 0;
    if (maybe_value) {
        tr_unc_node sub; sub.op = TR_UNC_SUB; sub.a = nd->a; sub.b = nd->b; sub.reserved = 0; sub.k = tr_q_int(0);
        tr_unc_numeric(p, k, &sub, a, b, d);
        int lt0 = 0, le0 = 0, gt0 = 0, ge0 = 0, z0 = 0, nz = 0;
        if (d->null & TR_UNC_MAYBE_VALUE) {
            if (p->carrier == TR_UNC_EXACT) {
                int sl = (d->flags & TR_UNC_LO_INF) ? -1 : tr_unc_qsign(d->lo), sh = (d->flags & TR_UNC_HI_INF) ? 1 : tr_unc_qsign(d->hi);
                lt0 = sl < 0; le0 = sl <= 0; gt0 = sh > 0; ge0 = sh >= 0;
            } else { lt0 = d->dlo < 0; le0 = d->dlo <= 0; gt0 = d->dhi > 0; ge0 = d->dhi >= 0; }
            z0 = d->zero != TR_UNC_NONZERO; nz = d->zero != TR_UNC_ZERO;
        }
        int t = 0, f = 0;
        switch (nd->op) {
        case TR_UNC_LT: t = lt0; f = ge0; break;
        case TR_UNC_LE: t = le0; f = gt0; break;
        case TR_UNC_GT: t = gt0; f = le0; break;
        case TR_UNC_GE: t = ge0; f = lt0; break;
        case TR_UNC_EQ: t = z0; f = nz; break;
        default: t = nz; f = z0; break;               /* NE */
        }
        out |= (t ? TR_UNC_TRUE : 0) | (f ? TR_UNC_FALSE : 0);
    }
    tr_unc_boolean_set(r, out);
}
/* node k of the program; tmp: scratch for a comparison's difference */
TR_NUM_HD void tr_unc_node_eval(const tr_unc_program *p, const tr_unc_plan *plan, int32_t k, const tr_unc_input *in,
                                tr_unc_value *slots, tr_unc_value *tmp) {
    const tr_unc_node *nd = &p->nodes[k];
    tr_unc_value *r = &slots[plan->slot[k]];
    int unary = nd->op == TR_UNC_NEG || nd->op == TR_UNC_DIV || nd->op == TR_UNC_ABS || nd->op == TR_UNC_SQR ||
                nd->op == TR_UNC_MOD || nd->op == TR_UNC_NOT || nd->op == TR_UNC_ISNULL || nd->op == TR_UNC_NOTNULL;
    int leaf = nd->op == TR_UNC_COL || nd->op == TR_UNC_CONST;
    const tr_unc_value *a = leaf ? NULL : &slots[plan->slot[nd->a]];
    const tr_unc_value *b = leaf || unary ? NULL : &slots[plan->slot[nd->b]];
    switch (nd->op) {
    case TR_UNC_COL: tr_unc_from_input(p, nd->a, &in[nd->a], r); return;
    case TR_UNC_CONST: {
        tr_unc_value_clear(r, TR_UNC_NUMERIC);
        r->null = in[0].rows ? TR_UNC_MAYBE_VALUE : 0;
        r->cr = nd->k; r->cm = tr_q_int(0);
        tr_unc_aff f;
        f.top = 0; f.n = 0; f.err = tr_unc_s_zero();
        if (p->carrier == TR_UNC_EXACT) { r->lo = r->hi = nd->k; f.center = tr_unc_s_zero(); f.center.q = nd->k; }
        else {
            tr_math_interval e = tr_unc_enclose(nd->k);
            r->dlo = e.lo; r->dhi = e.hi;
            tr_unc_ctx c = tr_unc_ctx_make(p->carrier, p->precision);
            f.center = tr_unc_s_round(&c, e.lo, e.hi);
            f.err = tr_unc_s_point(c.err);
        }
        tr_unc_aff_finish(p, k, &f, tr_unc_s_zero(), r);
        tr_unc_normalize(p, r);
        return;
    }
    case TR_UNC_LT: case TR_UNC_LE: case TR_UNC_EQ: case TR_UNC_NE: case TR_UNC_GE: case TR_UNC_GT:
        tr_unc_compare(p, k, nd, a, b, tmp, r); return;
    case TR_UNC_AND: tr_unc_boolean_set(r, tr_unc_and3(a->outcomes, b->outcomes)); return;
    case TR_UNC_OR: tr_unc_boolean_set(r, tr_unc_not3(tr_unc_and3(tr_unc_not3(a->outcomes), tr_unc_not3(b->outcomes)))); return;
    case TR_UNC_NOT: tr_unc_boolean_set(r, tr_unc_not3(a->outcomes)); return;
    case TR_UNC_ISNULL: case TR_UNC_NOTNULL: {
        int32_t isn = nd->op == TR_UNC_ISNULL;
        int32_t t = isn ? (a->null & TR_UNC_MAYBE_NULL) : (a->null & TR_UNC_MAYBE_VALUE);
        int32_t f = isn ? (a->null & TR_UNC_MAYBE_VALUE) : (a->null & TR_UNC_MAYBE_NULL);
        tr_unc_boolean_set(r, (t ? TR_UNC_TRUE : 0) | (f ? TR_UNC_FALSE : 0));
        return;
    }
    case TR_UNC_MUL:
        if (nd->a == nd->b) {                             /* a * a: the square's rules */
            tr_unc_node sq = *nd;
            sq.op = TR_UNC_SQR;
            tr_unc_numeric(p, k, &sq, a, NULL, r);
            return;
        }
        tr_unc_numeric(p, k, nd, a, b, r); return;
    default: tr_unc_numeric(p, k, nd, a, b, r); return;
    }
}
/* the per-group result of the root value */
TR_NUM_HD void tr_unc_result_of(const tr_unc_program *p, const tr_unc_value *v, uint64_t rows, tr_unc_result *out) {
    out->kind = v->kind; out->outcomes = v->outcomes; out->null = v->null; out->zero = v->zero; out->flags = v->flags;
    out->nterms = v->nterms; out->fresh = 0;
    for (int32_t i = 0; i < v->nterms; i++) out->fresh += v->term[i].sym >= TR_UNC_FRESH;
    out->lo = v->lo; out->hi = v->hi; out->dlo = v->dlo; out->dhi = v->dhi; out->cr = v->cr; out->cm = v->cm;
    out->rows = rows; out->reserved = 0;
    out->center = v->center; out->radius = tr_q_int(0); out->error = tr_q_int(0);
    out->dcenter = v->dcenter; out->dradius = 0.0; out->derror = 0.0;
    if (v->kind == TR_UNC_NUMERIC && (v->null & TR_UNC_MAYBE_VALUE) && !(v->flags & TR_UNC_AFFINE_TOP)) {
        tr_unc_ctx c = tr_unc_ctx_make(p->carrier, 53);
        tr_unc_s ra = tr_unc_radius(&c, v->term, v->nterms, 1), re = tr_unc_radius(&c, v->term, v->nterms, 0);
        if (c.fail) out->flags |= TR_UNC_AFFINE_TOP;
        else if (p->carrier == TR_UNC_EXACT) {
            out->radius = ra.q; out->error = re.q;
            out->dcenter = tr_unc_enclose_lo(v->center); out->dradius = tr_unc_enclose_hi(ra.q); out->derror = tr_unc_enclose_hi(re.q);
        } else { out->dradius = ra.hi; out->derror = re.hi; }
    }
    if (v->kind == TR_UNC_BOOLEAN) {
        if (!rows) out->decision = TR_UNC_EMPTY;
        else if (v->outcomes == TR_UNC_TRUE) out->decision = TR_UNC_ALL;
        else if (!(v->outcomes & TR_UNC_TRUE)) out->decision = TR_UNC_NONE;
        else out->decision = TR_UNC_UNKNOWN;
    } else out->decision = rows ? TR_UNC_UNKNOWN : TR_UNC_EMPTY;
}
/* one group: in[ncols] its inputs, slots[plan->nlive + 1] its scratch (the last one: a comparison's difference) */
TR_NUM_HD void tr_unc_eval_group(const tr_unc_program *p, const tr_unc_plan *plan, const tr_unc_input *in,
                                 tr_unc_value *slots, tr_unc_result *out) {
    for (int32_t k = 0; k < p->nnodes; k++) tr_unc_node_eval(p, plan, k, in, slots, &slots[plan->nlive]);
    tr_unc_result_of(p, &slots[plan->slot[p->root]], in[0].rows, out);
}

/* ---------------------------------------------------------------- the scratch plan */
/* operands of a node: the count, *a, *b (-1 when absent) */
TR_NUM_HD int tr_unc_operands(const tr_unc_node *nd, int32_t *a, int32_t *b) {
    *a = -1; *b = -1;
    switch (nd->op) {
    case TR_UNC_COL: case TR_UNC_CONST: return 0;
    case TR_UNC_NEG: case TR_UNC_DIV: case TR_UNC_ABS: case TR_UNC_SQR: case TR_UNC_MOD: case TR_UNC_NOT:
    case TR_UNC_ISNULL: case TR_UNC_NOTNULL: *a = nd->a; return 1;
    default: *a = nd->a; *b = nd->b; return 2;
    }
}
/* a node's value lives from its evaluation to its last use (the root to the end); identity: slot k = node k */
TR_NUM_HD void tr_unc_plan_slots(const tr_unc_program *p, tr_unc_plan *plan, int identity) {
    int32_t last[TR_UNC_MAX_NODES], owner[TR_UNC_MAX_NODES], nslots = 0;
    for (int32_t k = 0; k < p->nnodes; k++) last[k] = k;
    for (int32_t k = 0; k < p->nnodes; k++) {
        int32_t a, b;
        tr_unc_operands(&p->nodes[k], &a, &b);
        if (a >= 0) last[a] = k;
        if (b >= 0) last[b] = k;
    }
    last[p->root] = p->nnodes;
    for (int32_t s = 0; s < TR_UNC_MAX_NODES; s++) owner[s] = -1;
    for (int32_t k = 0; k < p->nnodes; k++) {
        if (identity) { plan->slot[k] = k; continue; }
        int32_t s = 0;
        while (s < nslots && last[owner[s]] >= k) s++;    /* the first slot whose value is dead before node k */
        if (s == nslots) nslots++;
        owner[s] = k; plan->slot[k] = s;
    }
    plan->nlive = identity ? p->nnodes : nslots;
}

/* ================================================================ concrete rows (exact; SQL NULLs) */
enum { TR_UNC_ROW_NULL = 0, TR_UNC_ROW_VALUE = 1, TR_UNC_ROW_OVERFLOW = -1 };
typedef struct { tr_num_rational v; int32_t state, truth; } tr_unc_rowval;   /* truth: boolean nodes (1/0) */
/* the program's root at one selected row: TR_UNC_ROW_VALUE (*out), _NULL or _OVERFLOW. cols[j]: the scope column of
 * program column j; vals: scratch of p->nnodes entries */
TR_NUM_HD int tr_unc_row_eval(const tr_unc_program *p, const tr_scope_view *v, const int32_t *cols, uint64_t row,
                              tr_unc_rowval *vals, tr_unc_rowval *out) {
    for (int32_t k = 0; k < p->nnodes; k++) {
        const tr_unc_node *nd = &p->nodes[k];
        tr_unc_rowval *r = &vals[k];
        r->state = TR_UNC_ROW_VALUE; r->truth = 0; r->v = tr_q_int(0);
        int32_t ia, ib;
        int nops = tr_unc_operands(nd, &ia, &ib);
        const tr_unc_rowval *a = nops >= 1 ? &vals[ia] : NULL, *b = nops == 2 ? &vals[ib] : NULL;
        if ((a && a->state == TR_UNC_ROW_OVERFLOW) || (b && b->state == TR_UNC_ROW_OVERFLOW)) { r->state = TR_UNC_ROW_OVERFLOW; continue; }
        int an = a && a->state == TR_UNC_ROW_NULL, bn = b && b->state == TR_UNC_ROW_NULL, ok = 1;
        tr_num_rational t;
        switch (nd->op) {
        case TR_UNC_COL: {
            int32_t c = cols[nd->a];
            if (!tr_scope_present(v, c, row)) { r->state = TR_UNC_ROW_NULL; break; }
            ok = tr_q_make(tr_scope_value(v, c, row), tr_unc_pow10(v->col[c].scale), &r->v);
            break;
        }
        case TR_UNC_CONST: r->v = nd->k; break;
        case TR_UNC_AND: case TR_UNC_OR: {               /* FALSE AND x = FALSE, TRUE OR x = TRUE, else NULL wins */
            int dom = nd->op == TR_UNC_AND ? 0 : 1;
            if ((!an && a->truth == dom) || (!bn && b->truth == dom)) { r->truth = dom; break; }
            if (an || bn) { r->state = TR_UNC_ROW_NULL; break; }
            r->truth = 1 - dom;
            break;
        }
        case TR_UNC_NOT: if (an) r->state = TR_UNC_ROW_NULL; else r->truth = !a->truth; break;
        case TR_UNC_ISNULL: r->truth = an; break;
        case TR_UNC_NOTNULL: r->truth = !an; break;
        default:
            if (an || bn) { r->state = TR_UNC_ROW_NULL; break; }
            switch (nd->op) {
            case TR_UNC_ADD: ok = tr_q_add(a->v, b->v, &r->v); break;
            case TR_UNC_SUB: ok = tr_q_sub(a->v, b->v, &r->v); break;
            case TR_UNC_MUL: ok = tr_q_mul(a->v, b->v, &r->v); break;
            case TR_UNC_NEG: ok = tr_unc_qneg(a->v, &r->v); break;
            case TR_UNC_DIV: ok = tr_q_div(a->v, nd->k, &r->v); break;
            case TR_UNC_MIN: r->v = tr_q_cmp(a->v, b->v) <= 0 ? a->v : b->v; break;
            case TR_UNC_MAX: r->v = tr_q_cmp(a->v, b->v) >= 0 ? a->v : b->v; break;
            case TR_UNC_ABS: ok = tr_unc_qabs(a->v, &r->v); break;
            case TR_UNC_SQR: ok = tr_q_mul(a->v, a->v, &r->v); break;
            case TR_UNC_MOD: ok = tr_q_div(a->v, nd->k, &t) && tr_q_mul(nd->k, tr_q_int(tr_unc_qfloor(t)), &t) && tr_q_sub(a->v, t, &r->v); break;
            default: {                                    /* comparisons */
                int s = tr_q_cmp(a->v, b->v);
                switch (nd->op) {
                case TR_UNC_LT: r->truth = s < 0; break;
                case TR_UNC_LE: r->truth = s <= 0; break;
                case TR_UNC_EQ: r->truth = s == 0; break;
                case TR_UNC_NE: r->truth = s != 0; break;
                case TR_UNC_GE: r->truth = s >= 0; break;
                default: r->truth = s > 0; break;
                }
            }
            }
        }
        if (!ok) r->state = TR_UNC_ROW_OVERFLOW;
    }
    *out = vals[p->root];
    return out->state;
}
/* one 64-row word: bit r set when the predicate is TRUE at selected row r (decisions per group, NULL: every row
 * concretely); acc[0..3] += proven, evaluated, true, overflow rows; *first_overflow = min(the overflowing rows) */
TR_NUM_HD uint64_t tr_unc_rows_word(const tr_unc_program *p, const tr_scope_view *v, const int32_t *cols,
                                    const int32_t *decisions, uint64_t w, tr_unc_rowval *vals, uint64_t acc[4],
                                    uint64_t *first_overflow) {
    uint64_t bits = v->selection[w], out = 0;
    while (bits) {
        int bit = tr_ipoly_ctz(bits);
        bits &= bits - 1;
        uint64_t row = w * 64 + (uint64_t)bit;
        int32_t d = decisions ? decisions[tr_scope_group(v, row)] : TR_UNC_UNKNOWN;
        if (d == TR_UNC_ALL) { out |= (uint64_t)1 << bit; acc[0]++; acc[2]++; continue; }
        if (d == TR_UNC_NONE || d == TR_UNC_EMPTY) { acc[0]++; continue; }
        tr_unc_rowval res;
        int st = tr_unc_row_eval(p, v, cols, row, vals, &res);
        acc[1]++;
        if (st == TR_UNC_ROW_OVERFLOW) { acc[3]++; if (row < *first_overflow) *first_overflow = row; continue; }
        if (st == TR_UNC_ROW_VALUE && res.truth) { out |= (uint64_t)1 << bit; acc[2]++; }
    }
    return out;
}
/* ---------------------------------------------------------------- validation (host code of both libraries) */
static inline int tr_unc_canonical(tr_num_rational q) {
    if (q.denominator <= 0) return 0;
    tr_num_rational c;
    return tr_q_make(q.numerator, q.denominator, &c) && c.numerator == q.numerator && c.denominator == q.denominator;
}

static inline int tr_unc_kind_ok(int32_t op, int32_t ka, int32_t kb, int nops) {
    switch (op) {
    case TR_UNC_COL: case TR_UNC_CONST: return 1;
    case TR_UNC_NEG: case TR_UNC_DIV: case TR_UNC_ABS: case TR_UNC_SQR: case TR_UNC_MOD: return ka == TR_UNC_NUMERIC;
    case TR_UNC_NOT: return ka == TR_UNC_BOOLEAN;
    case TR_UNC_ISNULL: case TR_UNC_NOTNULL: return nops == 1;
    case TR_UNC_AND: case TR_UNC_OR: return ka == TR_UNC_BOOLEAN && kb == TR_UNC_BOOLEAN;
    default: return ka == TR_UNC_NUMERIC && kb == TR_UNC_NUMERIC;
    }
}
static inline int32_t tr_unc_check(const tr_unc_program *p, int symbolic, tr_unc_plan *plan, tr_math_need *need) {
    if (need) for (int i = 0; i < TR_BUDGET_RESOURCES; i++) need->need[i] = 0;
    if (!p || p->nnodes < 1 || !p->nodes || p->ncols < 1 || p->reserved) return TR_MATH_INVALID;
    if (p->nnodes > TR_UNC_MAX_NODES || p->ncols > TR_UNC_MAX_COLS) return TR_MATH_UNSUPPORTED;
    if (p->carrier == TR_UNC_EXACT) { if (p->precision != 0) return TR_MATH_INVALID; }
    else if (p->carrier == TR_UNC_BINARY64) { if (p->precision < 2 || p->precision > 53) return TR_MATH_INVALID; }
    else return TR_MATH_INVALID;
    if (p->rank > TR_UNC_MAX_TERMS) return TR_MATH_UNSUPPORTED;
    if (p->rank < 1) return TR_MATH_INVALID;
    if (p->domains < 0 || p->domains > (symbolic ? 7 : TR_UNC_DOM_ALL)) return TR_MATH_INVALID;
    if (p->budget.denominator < 0 || (p->budget.denominator == 0 && p->budget.numerator != 0) ||
        (p->budget.denominator > 0 && (!tr_unc_canonical(p->budget) || p->budget.numerator < 0))) return TR_MATH_INVALID;
    if (p->root < 0 || p->root >= p->nnodes) return TR_MATH_INVALID;
    int32_t kind[TR_UNC_MAX_NODES];
    for (int32_t k = 0; k < p->nnodes; k++) {
        const tr_unc_node *nd = &p->nodes[k];
        if (nd->op < 1 || nd->op >= TR_UNC_OPS || nd->reserved) return TR_MATH_INVALID;
        int32_t a, b;
        int n = tr_unc_operands(nd, &a, &b);
        if ((n >= 1 && (a < 0 || a >= k)) || (n == 2 && (b < 0 || b >= k))) return TR_MATH_INVALID;
        if (!tr_unc_kind_ok(nd->op, n >= 1 ? kind[a] : 0, n == 2 ? kind[b] : 0, n)) return TR_MATH_INVALID;
        if (nd->op == TR_UNC_COL && (nd->a < 0 || nd->a >= p->ncols)) return TR_MATH_INVALID;
        if ((nd->op == TR_UNC_CONST || nd->op == TR_UNC_DIV || nd->op == TR_UNC_MOD) && !tr_unc_canonical(nd->k)) return TR_MATH_INVALID;
        if (nd->op == TR_UNC_DIV && nd->k.numerator == 0) return TR_MATH_INVALID;        /* a zero divisor */
        if (nd->op == TR_UNC_DIV) { tr_num_rational inv = tr_q_int(0); if (!tr_q_div(tr_q_int(1), nd->k, &inv)) return TR_MATH_UNSUPPORTED; }
        if (nd->op == TR_UNC_MOD && nd->k.numerator <= 0) return TR_MATH_INVALID;
        kind[k] = (nd->op >= TR_UNC_LT && nd->op <= TR_UNC_NOTNULL) ? TR_UNC_BOOLEAN : TR_UNC_NUMERIC;
    }
    if (plan) {
        memset(plan, 0, sizeof *plan);
        plan->nnodes = p->nnodes; plan->root = p->root;
        for (int32_t k = 0; k < p->nnodes; k++) plan->kind[k] = kind[k];
        tr_unc_plan_slots(p, plan, 0);
    }
    if (need) {
        need->need[TR_BUDGET_CIRCUIT] = (uint64_t)p->nnodes;
        need->need[TR_BUDGET_VARIABLES] = (uint64_t)p->ncols;
        need->need[TR_BUDGET_SUPPORT] = (uint64_t)p->rank;
    }
    return TR_MATH_OK;
}
static inline int tr_unc_input_ok(const tr_unc_input *in) {
    if (in->reserved || in->scale < 0 || in->scale > TR_UNC_MAX_SCALE || in->present > in->rows) return 0;
    if (!in->present) return 1;
    if (in->min > in->max) return 0;
    uint64_t span = (uint64_t)in->max - (uint64_t)in->min;
    if ((in->modulus == 0) != (span == 0)) return 0;
    return in->modulus == 0 || span % in->modulus == 0;
}
#define TR_UNC_GPU_CHUNK 4096           /* groups per launch of tr_mgpu_unc_eval (its scratch: chunk x (live + 1)) */
/* the device copies of tr_mgpu_unc_eval (bytes; UNKNOWN past 64 bits) */
static inline uint64_t tr_unc_eval_device_bytes(uint64_t ngroups, int32_t ncols, int32_t nnodes, int32_t nlive) {
    uint64_t chunk = ngroups < TR_UNC_GPU_CHUNK ? ngroups : TR_UNC_GPU_CHUNK;
    uint64_t b = tr_budget_mul(tr_budget_mul(ngroups, (uint64_t)(ncols > 0 ? ncols : 1)), sizeof(tr_unc_input));
    b = tr_budget_add(b, tr_budget_mul(ngroups, sizeof(tr_unc_result)));
    b = tr_budget_add(b, tr_budget_mul((uint64_t)(nnodes > 0 ? nnodes : 1), sizeof(tr_unc_node)));
    return tr_budget_add(b, tr_budget_mul(tr_budget_mul(chunk ? chunk : 1, (uint64_t)nlive + 1), sizeof(tr_unc_value)));
}
/* the device copies of tr_mgpu_unc_rows: the selection, the columns read (and validity), the ids, the decisions, the
 * program, the bitmap and the accumulators */
static inline uint64_t tr_unc_rows_device_bytes(const tr_scope_view *v, const int32_t *cols, int32_t ncols, int32_t nnodes,
                                                int decisions) {
    uint64_t rows = v->rows, b = tr_budget_mul(tr_budget_bitmap_bytes(rows), 2);
    uint32_t seen = 0;
    for (int32_t j = 0; j < ncols; j++) {
        int32_t c = cols[j];
        if (c < 0 || c >= v->ncols || c >= 32 || ((seen >> c) & 1)) continue;
        seen |= (uint32_t)1 << c;
        b = tr_budget_add(b, tr_budget_mul(rows, v->col[c].values ? 8 : 4));
        if (v->col[c].valid) b = tr_budget_add(b, rows);
    }
    if (decisions) {
        b = tr_budget_add(b, tr_budget_mul(v->ngroups, 4));
        if (v->groups) b = tr_budget_add(b, tr_budget_mul(rows, 4));
    }
    b = tr_budget_add(b, tr_budget_mul((uint64_t)(nnodes > 0 ? nnodes : 1), sizeof(tr_unc_node)));
    return tr_budget_add(b, tr_budget_add(tr_budget_mul((uint64_t)TR_UNC_MAX_COLS, 4), 64));
}

/* ------------------------------------------------------------------ exported (math_unc.c) */
/* Validates a program (topological operands of the right kinds, COL < ncols, DIV k != 0, MOD k > 0, carrier and
 * precision, rank 1..16, budget >= 0, root) and writes its scratch plan and needs (circuit = nodes, variables =
 * columns, support = rank). plan / need may be NULL. */
int32_t tr_unc_program_check(const tr_unc_program *, tr_unc_plan *plan, tr_math_need *need);
/* The state of a scope over its columns cols[ncols] (scope column indices, distinct): per group and column the
 * selected rows, present rows, extremes and the gcd of differences; the scope's record (FND-007) for freshness. */
tr_unc_state *tr_unc_state_from_scope(const tr_scope *, const int32_t *cols, int32_t ncols, const tr_math_budget *,
                                      tr_math_refusal *, int32_t *error);
/* A state from explicit inputs (ngroups x ncols, group-major), validated: present <= rows, min <= max, modulus
 * divides max - min (0 iff min == max), scale 0..18, equal rows within a group. */
tr_unc_state *tr_unc_state_from_inputs(uint64_t ngroups, int32_t ncols, const tr_unc_input *inputs,
                                       const tr_math_budget *, tr_math_refusal *, int32_t *error);
int32_t tr_unc_state_need(uint64_t ngroups, int32_t ncols, uint64_t rows, int32_t scoped, tr_math_need *need);
int32_t tr_unc_state_get_info(const tr_unc_state *, tr_unc_state_info *);
const tr_unc_input *tr_unc_state_inputs(const tr_unc_state *);       /* borrowed: ngroups x ncols */
/* TR_MATH_OK, or TR_MATH_STALE with FND-007's reasons; TR_MATH_INVALID for a state built from inputs */
int32_t tr_unc_state_fresh(const tr_unc_state *, const tr_scope *current, tr_cert_staleness *out);
void tr_unc_state_free(tr_unc_state *);
/* Groups first .. first + count - 1 evaluated abstractly into out[count]. The program's ncols must equal the
 * state's. Budget: output = count, bytes = the scratch. */
int32_t tr_unc_eval(const tr_unc_state *, const tr_unc_program *, uint64_t first, uint64_t count, tr_unc_result *out,
                    const tr_math_budget *, tr_math_refusal *);
/* the needs of tr_unc_eval (TR_UNC_NEED_CPU) or tr_mgpu_unc_eval (TR_UNC_NEED_DEVICE) */
int32_t tr_unc_eval_need(const tr_unc_program *, uint64_t ngroups, int32_t ncols, int32_t target, tr_math_need *need);
/* every node's value for one group (values[capacity >= nnodes]) */
int32_t tr_unc_trace(const tr_unc_state *, const tr_unc_program *, uint64_t group, tr_unc_value *values, int32_t capacity);
/* The symbolic affine form of a node (columns are the exact symbols x_j: center 0, coefficient 1, no bounds), in the
 * EXACT carrier without condensation or budget: exact = no approximation symbol remains. */
int32_t tr_unc_symbolic(const tr_unc_program *, int32_t node, tr_unc_value *out, int32_t *exact);
/* Concrete rows: bitmap (ceil(rows / 64) words) of the selected rows whose predicate is TRUE, the program's columns
 * read from the scope columns cols[program ncols]; decisions (per group, from tr_unc_eval of the same scope's state)
 * or NULL. TR_MATH_OVERFLOW when a row's exact value leaves int64 rationals (stats say how many; the bitmap is then
 * unspecified). */
int32_t tr_unc_rows(const tr_scope *, const int32_t *cols, const tr_unc_program *, const int32_t *decisions,
                    uint64_t *bitmap, tr_unc_filter_stats *, const tr_math_budget *, tr_math_refusal *);
/* The pruning filter: the state must be fresh for the scope; groups decided by the evaluation, UNKNOWN groups row by
 * row. stats also count the groups by decision. */
int32_t tr_unc_filter(const tr_unc_state *, const tr_scope *, const tr_unc_program *, uint64_t *bitmap,
                      tr_unc_filter_stats *, const tr_math_budget *, tr_math_refusal *);
/* the overflow decision of a SUM over a group of a numeric result, in units of `unit` (> 0), a signed accumulator of
 * bits (2..128) */
int32_t tr_unc_sum_fits(const tr_unc_result *, int32_t bits, tr_num_rational unit, tr_unc_sum *out);
/* the CRT meet; TR_MATH_NO_SOLUTION when the classes are disjoint, TR_MATH_OVERFLOW past int64 rationals */
int32_t tr_unc_cong_meet(tr_num_rational r1, tr_num_rational m1, tr_num_rational r2, tr_num_rational m2,
                         tr_num_rational *r, tr_num_rational *m);
/* the LAW certificate of an exact affine identity of a polynomial node (see the header comment) */
tr_cert *tr_unc_identity(const tr_unc_program *, int32_t node, const tr_math_budget *, tr_math_refusal *, int32_t *error);
/* the SNAPSHOT certificate of a fact proved for the program's root in every group of the state (fact TR_UNC_FACT_*;
 * modulus: the prime of a congruence), verified by FND-007 on the scope (which must be the state's, fresh) */
tr_cert *tr_unc_certify(const tr_unc_state *, const tr_scope *, const tr_unc_program *, int32_t fact, uint64_t modulus,
                        tr_cert_evidence *evidence, const tr_math_budget *, tr_math_refusal *, int32_t *error);

/* ------------------------------------------------------------------ the CUDA library (math_unc_gpu.cu) */
/* inputs: ngroups x ncols (host); out[ngroups]. Budget (NULL = defaults): output = groups, bytes =
 * tr_unc_eval_device_bytes. 0 or a native error code (TR_MATH_DEVICE when the device fails). */
int tr_mgpu_unc_eval(const tr_unc_input *inputs, uint64_t ngroups, int32_t ncols, const tr_unc_program *,
                     tr_unc_result *out, const tr_math_budget *, tr_math_refusal *);
/* the concrete rows of tr_unc_rows on a scope view (decisions: view->ngroups entries or NULL) */
int tr_mgpu_unc_rows(const tr_scope_view *, const int32_t *cols, const tr_unc_program *, const int32_t *decisions,
                     uint64_t *bitmap, tr_unc_filter_stats *, const tr_math_budget *, tr_math_refusal *);
#ifdef __cplusplus
}
#endif
#endif
