/* BOOL-001: predicate DAGs over k conditions; their cofactors, XOR (Boolean) derivatives and signed first and mixed
 * differences, with exact measures over a declared domain, on the CPU (math_bool.c) and on a CUDA device
 * (math_bool_gpu.cu) with the same counts bit for bit.
 * SPDX-License-Identifier: Apache-2.0
 *
 * ---- Truth values -------------------------------------------------------------------------------------------------
 * SQL's three-valued logic: TRUE, FALSE, UNKNOWN. At a point a truth value is two bits, value and valid: TRUE = (1, 1),
 * FALSE = (0, 1), UNKNOWN = (0, 0); value implies valid. A word holds 64 points (bit p of word w is point 64 w + p).
 * The connectives are Kleene's strong logic, which is SQL's: NOT UNKNOWN = UNKNOWN; AND is FALSE when either side is
 * FALSE, else UNKNOWN when either side is, else TRUE; OR dually; XOR(a, b) is UNKNOWN when either side is, else
 * a != b (SQL's `a <> b` on booleans). On words (value v, valid k), tr_bool_apply:
 *   NOT  v = k & ~v, k = k                    AND  v = va & vb, k = v | (ka & ~va) | (kb & ~vb)
 *   OR   v = va | vb, k = v | (ka & ~va & kb & ~vb)                 XOR  k = ka & kb, v = (va ^ vb) & k
 *
 * ---- Domains (tr_bool_masks, read as a tr_bool_domain) ------------------------------------------------------------
 * ASSIGNMENTS (encoded variables): the points are the 2^k assignments a = 0 .. 2^k - 1 of k two-valued variables;
 *   condition i at a is bit i of a, always known; every point counts once (the uniform measure, total 2^k). Its words
 *   are generated, never stored: bit p of word w is assignment 64 w + p, so condition i < 6 is the fixed pattern of
 *   bit i of p (0xAAAA.., 0xCCCC.., ...) and condition i >= 6 is bit i - 6 of w (tr_bool_var_word).
 * ROWS (logical features of table rows): the points are the physical rows of a data scope (math_scope.h); only the
 *   selected rows count, each once (bag multiplicity: the empirical measure, total = the selected rows). Condition i
 *   is a native row predicate over the scope's columns, evaluated once into a value and a valid bitmap of the rows
 *   (canonical: value within valid within the selection; bits of unselected rows are zero). The scope's group ids are
 *   ignored (one measure over all selected rows).
 *
 * ---- Conditions (tr_bool_condition) -------------------------------------------------------------------------------
 * Over the scope's exact logical values (decimal value = logical / 10^scale). A constant is a rational num/den in value
 * units (den >= 1); comparisons are exact (128-bit cross products x * den op num * 10^scale, never rounded):
 *   EQ NE LT LE GT GE      x op num[0]/den[0]          BETWEEN   num[0]/den[0] <= x <= num[1]/den[1] (SQL BETWEEN)
 *   IS_NULL IS_NOT_NULL    never UNKNOWN               BIT       bit num[1] (0..63) of (logical - num[0]) mod 2^64
 *   COL_EQ .. COL_GE       x op y, y the value of column col2 (scales aligned exactly)
 * A comparison or BIT on a row where its column (or col2) is NULL is UNKNOWN (SQL). Fields a condition does not use
 * must be zero (num, den: 0 / 1 as unused constants), col2 = -1 unless COL_*.
 *
 * ---- Programs (tr_bool_program, read as a tr_bool_program_view) ---------------------------------------------------
 * An immutable DAG of nnodes tr_bool_node; node i refers only to nodes < i, so a node with several parents is a shared
 * subexpression: CONST a (0 FALSE, 1 TRUE, 2 UNKNOWN: SQL's NULL literal), VAR a (condition a < k), NOT a, AND a b,
 * OR a b, XOR a b; operand fields a node does not use are -1. The output is node root. The program stores per node
 * deps (bit i set: condition i is read at or below the node; k <= 64) and the schedule: the nodes the root reaches, in
 * increasing order (the others are kept but never evaluated).
 *
 * ---- Cofactors and differences (tr_bool_query -> tr_bool_result) --------------------------------------------------
 * A query names a set S of m conditions (vars: bit i = condition i; m = |S| <= TR_BOOL_MAX_ORDER). For an assignment t
 * of S (bit j of t is the value given to the j-th condition of S in increasing index order; |t| = the conditions of
 * S set TRUE), the cofactor f|S:=t is the program with the conditions of S replaced by the constants TRUE / FALSE,
 * evaluated at the point with the logic above on the conditions outside S. A cofactor never reads the conditions of S
 * at the point: a row where a condition of S is UNKNOWN (its column NULL) has definite cofactors wherever the
 * conditions outside S that they read are known. At each point:
 *   XOR derivative     dS f = XOR over the 2^m assignments t of f|S:=t       (m = 1: f|1 XOR f|0)
 *   signed difference  DS f = sum over t of (-1)^(m - |t|) [f|S:=t]           (m = 1: f|1 - f|0 in {-1, 0, 1})
 * with [TRUE] = 1, [FALSE] = 0. DS f is an integer in [-2^(m-1), 2^(m-1)] (m = 0: DS f = [f] in {0, 1}), and
 * dS f = DS f mod 2: the XOR derivative counts the points where DS f is odd, not where DS f != 0 (f = x0 XOR x1 over
 * S = {0, 1}: DS f = -2 and dS f = 0 at every point). Strictness: when any of the 2^m cofactors is UNKNOWN at a point,
 * both differences are UNKNOWN there; the point is counted in `unknown` and in no other difference count (an UNKNOWN is
 * never read as FALSE or as 0).
 * Measures, exact counts of points (the probability under the domain's measure is count / points):
 *   points                        selected rows, or 2^k
 *   f_true, f_false, f_unknown    the program itself (f_true + f_false + f_unknown = points)
 *   known, unknown                points where every cofactor over S is known / where one is UNKNOWN
 *   xor_true, xor_false           dS f over the known points (xor_true + xor_false = known)
 *   histogram                     bin b = the known points where DS f = lo + b; nbins = 2^m + 1 and lo = -2^(m-1),
 *                                 or nbins = 2 and lo = 0 for m = 0 (the bins sum to known)
 *   nonzero, sum                  known points with DS f != 0; the exact sum of DS f over the known points (128-bit)
 *   cofactors (TR_BOOL_COFACTORS) for every t, the points where f|S:=t is TRUE, FALSE and UNKNOWN (3 counts per t)
 * Output layout: queries in order; query q's bins start at results[q].hist (the sum of the earlier queries' nbins) and
 * its cofactor counts at results[q].cof (3 per assignment, t increasing; UINT64_MAX without TR_BOOL_COFACTORS);
 * tr_bool_layout gives both totals.
 *
 * ---- Execution --------------------------------------------------------------------------------------------------
 * Shared-mask execution (tr_bool_differences, tr_mgpu_bool_differences): the condition words are built once (the
 * masks) and each 64-point word is evaluated with bitwise operations only: the schedule once (the base words), then for
 * each assignment t of S only the cone of S (the scheduled nodes whose deps meet S; tr_bool_eval_cone) with the
 * conditions of S as constant words; the root's 2^m words are combined by bit-sliced counters (tr_bool_acc_*), split
 * into value classes (tr_bool_acc_leaves) and counted with popcounts. Every count is an integer, so the CPU, the
 * emulation and the device agree bit for bit in any thread order. Independent evaluation (tr_bool_direct): one
 * cofactor at a time, point by point, the conditions read again from the scope's values and the program evaluated with
 * scalar three-valued logic (no masks, no sharing): the reference baseline.
 *
 * ---- Budgets (math_budgets.h), checked before anything is allocated ----------------------------------------------
 *   masks, ROWS (tr_bool_masks_need, TR_BOOL_NEED_CPU and _ADOPT): variables = k, bytes = the object, the selection
 *     copy and the value and valid words (8 nw (1 + 2k)); TR_BOOL_NEED_DEVICE (tr_mgpu_bool_masks): bytes = its device
 *     copies (the selection, the columns the conditions read and their validity, the conditions, the output words).
 *   masks, ASSIGNMENTS: variables = k, support = 2^k (the points enumerated), bytes = the object.
 *   program: variables = k, circuit = nnodes, bytes = the object, 28 bytes per node and 1 byte of scratch per node.
 *   differences (tr_bool_differences_need): variables = k, degree = the largest order m, circuit = nnodes, support =
 *     2^k on ASSIGNMENTS (0 on ROWS: the rows were budgeted with the scope), tensor = sum over queries of 2^m (the
 *     cofactors evaluated per point), output = nq + total bins + total cofactor counts, bytes = CPU: the plans, the
 *     cone lists and the node words (16 per node); CUDA: the device copies and scratch (tr_bool_device_bytes).
 *   direct: variables = k, circuit = nnodes, support = 2^k on ASSIGNMENTS, bytes = 2 nnodes + k.
 * Representation limits (with a budget that admits them): TR_MATH_UNSUPPORTED for k > 64 (ROWS) or > 63
 * (ASSIGNMENTS), nnodes > TR_BOOL_MAX_NODES, m > TR_BOOL_MAX_ORDER, nq > TR_BOOL_MAX_QUERIES.
 *
 * Errors: TR_MATH_INVALID (-1, malformed input), _MEMORY (-2), _BUDGET (-5, with the tr_math_refusal),
 * TR_MATH_DOMAIN_ERROR (-6: a program over another number of conditions than the domain), TR_MATH_DEVICE (-11),
 * TR_MATH_UNSUPPORTED (-12). Objects are immutable once built and individually owned; constructors return NULL and
 * set *error, leaving their inputs unchanged; masks copy what they need from the scope (it may be freed after). The
 * freshness of a ROWS domain is its scope's (tr_scope_fresh; the Python layer checks it before every use). Read-only
 * calls on one object may run concurrently. */
#ifndef TORBLES_MATH_BOOL_H
#define TORBLES_MATH_BOOL_H
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "math_num.h"
#include "math_budgets.h"
#include "math_scope.h"
#ifdef __cplusplus
extern "C" {
#endif

#define TR_BOOL_ABI 1
#define TR_BOOL_MAX_CONDITIONS 64          /* conditions of a ROWS domain (deps and queries are uint64 masks) */
#define TR_BOOL_MAX_ASSIGN_VARS 63         /* variables of an ASSIGNMENTS domain (2^k points fit uint64) */
#define TR_BOOL_MAX_NODES (1 << 24)
#define TR_BOOL_MAX_ORDER 20               /* conditions of one difference (2^m cofactors per point) */
#define TR_BOOL_MAX_QUERIES (1 << 16)
#define TR_BOOL_COUNTER_BITS (TR_BOOL_MAX_ORDER + 1)   /* bit-sliced counters of up to 2^(m-1) cofactor words */
#define TR_BOOL_VALUE_BITS (TR_BOOL_MAX_ORDER + 2)     /* two's complement DS f */
#define TR_BOOL_NONE UINT64_MAX

enum { TR_BOOL_ROWS = 1, TR_BOOL_ASSIGNMENTS = 2 };                      /* domain kinds */
enum { TR_BOOL_FALSE = 0, TR_BOOL_TRUE = 1, TR_BOOL_UNKNOWN = 2 };       /* scalar truth values (CONST a) */
enum { TR_BOOL_CONST = 1, TR_BOOL_VAR = 2, TR_BOOL_NOT = 3, TR_BOOL_AND = 4, TR_BOOL_OR = 5, TR_BOOL_XOR = 6 };
enum { TR_BOOL_EQ = 1, TR_BOOL_NE = 2, TR_BOOL_LT = 3, TR_BOOL_LE = 4, TR_BOOL_GT = 5, TR_BOOL_GE = 6,
       TR_BOOL_BETWEEN = 7, TR_BOOL_IS_NULL = 8, TR_BOOL_IS_NOT_NULL = 9, TR_BOOL_BIT = 10,
       TR_BOOL_COL_EQ = 11, TR_BOOL_COL_NE = 12, TR_BOOL_COL_LT = 13, TR_BOOL_COL_LE = 14, TR_BOOL_COL_GT = 15,
       TR_BOOL_COL_GE = 16 };                                            /* condition ops */
enum { TR_BOOL_COFACTORS = 1 };                                          /* query flags */
enum { TR_BOOL_NEED_CPU = 0, TR_BOOL_NEED_DEVICE = 1, TR_BOOL_NEED_ADOPT = 2 };   /* need targets */

/* one native row predicate (see Conditions) */
typedef struct {
    int32_t op, col, col2, reserved;   /* TR_BOOL_EQ..; scope column; second column of COL_* (else -1); 0 */
    int64_t num[2], den[2];            /* constants num/den in value units; BIT: num[0] base, num[1] bit index */
} tr_bool_condition;                   /* 48 bytes */

typedef struct { int32_t op, a, b, reserved; } tr_bool_node;   /* 16 bytes (see Programs) */

/* a program as the CPU code and the kernels read it (pointers into the program; valid while it lives) */
typedef struct {
    const tr_bool_node *nodes;    /* nnodes */
    const uint64_t *deps;         /* nnodes: the conditions read at or below each node */
    const uint32_t *schedule;     /* nsched reachable nodes, increasing; the last is root */
    int32_t nnodes, nsched, root, k;
} tr_bool_program_view;           /* 40 bytes */

/* a domain as the CPU code and the kernels read it (pointers into the masks; valid while they live) */
typedef struct {
    uint64_t points;              /* physical rows, or 2^k assignments */
    uint64_t nw;                  /* words: ceil(points / 64) */
    uint64_t selected;            /* points that count: selected rows, or 2^k */
    int32_t kind, k;              /* TR_BOOL_ROWS / TR_BOOL_ASSIGNMENTS; conditions */
    const uint64_t *selection;    /* ROWS: nw words (bits past the rows clear); ASSIGNMENTS: NULL */
    const uint64_t *value;        /* ROWS: k * nw words, condition-major (value[i * nw + w]); ASSIGNMENTS: NULL */
    const uint64_t *valid;        /* ROWS: k * nw words, or NULL when every condition is known on every selected row */
} tr_bool_domain;                 /* 56 bytes */

typedef struct { uint64_t vars; int32_t flags, reserved; } tr_bool_query;   /* 16 bytes: S and TR_BOOL_COFACTORS */

typedef struct {
    uint64_t vars;                          /* S */
    int32_t order, nbins;                   /* m; histogram bins */
    int64_t lo;                             /* the value of bin 0 */
    uint64_t points;                        /* selected rows, or 2^k */
    uint64_t f_true, f_false, f_unknown;    /* the program itself */
    uint64_t known, unknown;                /* points where every cofactor over S is known / one is UNKNOWN */
    uint64_t xor_true, xor_false;           /* the XOR derivative over S on the known points */
    uint64_t nonzero;                       /* known points with DS f != 0 */
    tr_math_i128 sum;                       /* the exact sum of DS f over the known points */
    uint64_t hist, cof;                     /* offsets of this query's bins / cofactor counts (cof: TR_BOOL_NONE) */
} tr_bool_result;                           /* 128 bytes */

typedef struct {
    uint64_t points, nw, selected, bytes;   /* bytes the masks retain */
    uint64_t scope_digest[2];               /* the descriptor digest of the scope (ROWS; 0 for ASSIGNMENTS) */
    int32_t kind, k;
    int32_t has_valid, reserved;            /* 0: no condition is UNKNOWN on a selected row */
} tr_bool_masks_info;                       /* 64 bytes */

typedef struct {
    int32_t nnodes, nsched, root, k;
    uint64_t support;                       /* deps of the root: the conditions the program reads */
    uint64_t bytes;
} tr_bool_program_info;                     /* 32 bytes */

typedef struct tr_bool_masks_s tr_bool_masks;
typedef struct tr_bool_program_s tr_bool_program;

/* ------------------------------------------------------------------ shared helpers (CPU code, CUDA host and kernels) */
TR_NUM_HD int tr_bool_popc(uint64_t x) {
#if defined(__CUDA_ARCH__)
    return __popcll((unsigned long long)x);
#else
    return __builtin_popcountll(x);
#endif
}
TR_NUM_HD int tr_bool_ctz(uint64_t x) {   /* x != 0 */
#if defined(__CUDA_ARCH__)
    return __ffsll((long long)x) - 1;
#else
    return __builtin_ctzll(x);
#endif
}
TR_NUM_HD int64_t tr_bool_pow10(int32_t s) { int64_t p = 1; for (int32_t i = 0; i < s; i++) p *= 10; return p; }
TR_NUM_HD int tr_bool_compare(int32_t op, __int128 l, __int128 r) {
    switch (op) {
    case TR_BOOL_EQ: return l == r;
    case TR_BOOL_NE: return l != r;
    case TR_BOOL_LT: return l < r;
    case TR_BOOL_LE: return l <= r;
    case TR_BOOL_GT: return l > r;
    default: return l >= r;
    }
}
/* condition c at a row of the scope view: TR_BOOL_FALSE, _TRUE or _UNKNOWN (the caller visits selected rows only) */
TR_NUM_HD int tr_bool_condition_row(const tr_scope_view *v, const tr_bool_condition *c, uint64_t row) {
    int present = tr_scope_present(v, c->col, row);
    if (c->op == TR_BOOL_IS_NULL) return !present;
    if (c->op == TR_BOOL_IS_NOT_NULL) return present;
    if (!present) return TR_BOOL_UNKNOWN;
    int64_t x = tr_scope_value(v, c->col, row);
    if (c->op >= TR_BOOL_COL_EQ) {
        if (!tr_scope_present(v, c->col2, row)) return TR_BOOL_UNKNOWN;
        int64_t y = tr_scope_value(v, c->col2, row);
        int32_t sx = v->col[c->col].scale, sy = v->col[c->col2].scale, s = sx > sy ? sx : sy;
        return tr_bool_compare(c->op - TR_BOOL_COL_EQ + TR_BOOL_EQ, (__int128)x * tr_bool_pow10(s - sx),
                               (__int128)y * tr_bool_pow10(s - sy));
    }
    if (c->op == TR_BOOL_BIT) return (int)((((uint64_t)x - (uint64_t)c->num[0]) >> c->num[1]) & 1);
    int64_t p = tr_bool_pow10(v->col[c->col].scale);
    __int128 l0 = (__int128)x * c->den[0], r0 = (__int128)c->num[0] * p;
    if (c->op == TR_BOOL_BETWEEN) return l0 >= r0 && (__int128)x * c->den[1] <= (__int128)c->num[1] * p;
    return tr_bool_compare(c->op, l0, r0);
}
/* the value and valid words of condition c at word w (bits of unselected rows are zero) */
TR_NUM_HD void tr_bool_condition_word(const tr_scope_view *v, const tr_bool_condition *c, uint64_t w,
                                      uint64_t *value, uint64_t *valid) {
    uint64_t bits = v->selection[w], val = 0, ok = 0;
    while (bits) {
        int b = tr_bool_ctz(bits);
        bits &= bits - 1;
        int r = tr_bool_condition_row(v, c, w * 64 + (uint64_t)b);
        if (r != TR_BOOL_UNKNOWN) { ok |= (uint64_t)1 << b; if (r) val |= (uint64_t)1 << b; }
    }
    *value = val; *valid = ok;
}
/* the word of points that count */
TR_NUM_HD uint64_t tr_bool_selection_word(const tr_bool_domain *d, uint64_t w) {
    if (d->kind == TR_BOOL_ROWS) return d->selection[w];
    if (d->points >= 64) return ~(uint64_t)0;
    return ((uint64_t)1 << d->points) - 1;                    /* k < 6: one partial word */
}
/* condition i at word w; ASSIGNMENTS words are generated (and masked by the caller like stored ones) */
TR_NUM_HD void tr_bool_var_word(const tr_bool_domain *d, int32_t i, uint64_t w, uint64_t *value, uint64_t *valid) {
    if (d->kind == TR_BOOL_ROWS) {
        uint64_t at = (uint64_t)i * d->nw + w;
        *value = d->value[at];
        *valid = d->valid ? d->valid[at] : d->selection[w];
        return;
    }
    if (i < 6) {                                              /* bit p of the word set iff bit i of p is */
        switch (i) {
        case 0: *value = UINT64_C(0xAAAAAAAAAAAAAAAA); break;
        case 1: *value = UINT64_C(0xCCCCCCCCCCCCCCCC); break;
        case 2: *value = UINT64_C(0xF0F0F0F0F0F0F0F0); break;
        case 3: *value = UINT64_C(0xFF00FF00FF00FF00); break;
        case 4: *value = UINT64_C(0xFFFF0000FFFF0000); break;
        default: *value = UINT64_C(0xFFFFFFFF00000000); break;
        }
    } else {
        *value = ((w >> (i - 6)) & 1) ? ~(uint64_t)0 : 0;
    }
    *valid = ~(uint64_t)0;
}
/* one connective on words (see Truth values); CONST and VAR are handled by the callers */
TR_NUM_HD void tr_bool_apply(int32_t op, uint64_t va, uint64_t ka, uint64_t vb, uint64_t kb, uint64_t *v, uint64_t *k) {
    switch (op) {
    case TR_BOOL_NOT: *v = ka & ~va; *k = ka; break;
    case TR_BOOL_AND: *v = va & vb; *k = (va & vb) | (ka & ~va) | (kb & ~vb); break;
    case TR_BOOL_OR: *v = va | vb; *k = (va | vb) | (ka & ~va & kb & ~vb); break;
    default: *k = ka & kb; *v = (va ^ vb) & ka & kb; break;           /* XOR */
    }
}
TR_NUM_HD void tr_bool_const_word(int32_t a, uint64_t *v, uint64_t *k) {
    *v = a == TR_BOOL_TRUE ? ~(uint64_t)0 : 0;
    *k = a == TR_BOOL_UNKNOWN ? 0 : ~(uint64_t)0;
}
/* the base words of every scheduled node at word w: val[2 i] = value, val[2 i + 1] = valid */
TR_NUM_HD void tr_bool_eval_base(const tr_bool_program_view *p, const tr_bool_domain *d, uint64_t w, uint64_t *val) {
    for (int32_t s = 0; s < p->nsched; s++) {
        uint32_t i = p->schedule[s];
        const tr_bool_node *n = &p->nodes[i];
        uint64_t v, k;
        if (n->op == TR_BOOL_CONST) tr_bool_const_word(n->a, &v, &k);
        else if (n->op == TR_BOOL_VAR) tr_bool_var_word(d, n->a, w, &v, &k);
        else tr_bool_apply(n->op, val[2 * (uint64_t)n->a], val[2 * (uint64_t)n->a + 1],
                           n->b >= 0 ? val[2 * (uint64_t)n->b] : 0, n->b >= 0 ? val[2 * (uint64_t)n->b + 1] : 0, &v, &k);
        val[2 * (uint64_t)i] = v; val[2 * (uint64_t)i + 1] = k;
    }
}
/* the cone nodes cone[0..ncone) (increasing) at word w: the conditions of S = vars take the constant values of the
 * assignment t (actual = 0), or their own words again (actual = 1: restores the base words) */
TR_NUM_HD void tr_bool_eval_cone(const tr_bool_program_view *p, const tr_bool_domain *d, const uint32_t *cone,
                                 uint32_t ncone, uint64_t vars, uint64_t t, int actual, uint64_t w, uint64_t *val) {
    for (uint32_t c = 0; c < ncone; c++) {
        uint32_t i = cone[c];
        const tr_bool_node *n = &p->nodes[i];
        uint64_t v, k;
        if (n->op == TR_BOOL_VAR) {
            if (!((vars >> n->a) & 1) || actual) tr_bool_var_word(d, n->a, w, &v, &k);
            else {
                int j = tr_bool_popc(vars & (((uint64_t)1 << n->a) - 1));   /* the position of the condition in S */
                v = ((t >> j) & 1) ? ~(uint64_t)0 : 0;
                k = ~(uint64_t)0;
            }
        } else if (n->op == TR_BOOL_CONST) tr_bool_const_word(n->a, &v, &k);
        else tr_bool_apply(n->op, val[2 * (uint64_t)n->a], val[2 * (uint64_t)n->a + 1],
                           n->b >= 0 ? val[2 * (uint64_t)n->b] : 0, n->b >= 0 ? val[2 * (uint64_t)n->b + 1] : 0, &v, &k);
        val[2 * (uint64_t)i] = v; val[2 * (uint64_t)i + 1] = k;
    }
}

/* the combination of the 2^m cofactor words of one query at one word */
typedef struct {
    uint64_t known, parity;                                     /* every cofactor known; XOR of the cofactor values */
    uint64_t pos[TR_BOOL_COUNTER_BITS], neg[TR_BOOL_COUNTER_BITS];   /* bit-sliced counts of TRUE cofactors, by sign */
} tr_bool_acc;
TR_NUM_HD void tr_bool_acc_begin(tr_bool_acc *a, int32_t m, uint64_t sel) {
    a->known = sel; a->parity = 0;
    for (int32_t b = 0; b <= m; b++) { a->pos[b] = 0; a->neg[b] = 0; }
}
/* adds the root's words for assignment t (v and k masked by the selection) */
TR_NUM_HD void tr_bool_acc_add(tr_bool_acc *a, int32_t m, uint64_t t, uint64_t v, uint64_t k) {
    a->known &= k;
    a->parity ^= v;
    uint64_t *c = ((m - tr_bool_popc(t)) & 1) ? a->neg : a->pos, carry = v;
    for (int32_t b = 0; carry && b <= m; b++) { uint64_t x = c[b] & carry; c[b] ^= carry; carry = x; }
}
/* the value classes of DS f = pos - neg over the known points: (value, count) pairs, at most 64 (|value| <= 2^19);
 * returns their number. The split visits the bits of DS f from the sign bit down and prunes empty sides; its stack
 * holds at most one pending side per bit plus the two children of the current one: B + 1 <= TR_BOOL_VALUE_BITS + 1. */
TR_NUM_HD int tr_bool_acc_leaves(const tr_bool_acc *a, int32_t m, int32_t *values, uint32_t *counts) {
    const int32_t B = m + 2;
    uint64_t D[TR_BOOL_VALUE_BITS], carry = ~(uint64_t)0;      /* D = pos + ~neg + 1, two's complement in B bits */
    for (int32_t b = 0; b < B; b++) {
        uint64_t x = b <= m ? a->pos[b] : 0, y = ~(b <= m ? a->neg[b] : 0);
        D[b] = x ^ y ^ carry;
        carry = (x & y) | (carry & (x ^ y));
    }
    uint64_t sm[TR_BOOL_VALUE_BITS + 2];
    uint32_t sp_[TR_BOOL_VALUE_BITS + 2];
    int32_t sb[TR_BOOL_VALUE_BITS + 2];
    int sp = 0, n = 0;
    if (a->known) { sm[0] = a->known; sb[0] = B - 1; sp_[0] = 0; sp = 1; }
    while (sp) {                                                /* depth-first split by the bits of D, empty sides pruned */
        sp--;
        uint64_t mask = sm[sp];
        uint32_t prefix = sp_[sp];
        int32_t b = sb[sp];
        if (b < 0) {
            int32_t value = (int32_t)prefix;
            if ((prefix >> (B - 1)) & 1) value -= (int32_t)1 << B;
            values[n] = value; counts[n] = (uint32_t)tr_bool_popc(mask); n++;
            continue;
        }
        uint64_t one = mask & D[b], zero = mask & ~D[b];
        if (zero) { sm[sp] = zero; sb[sp] = b - 1; sp_[sp] = prefix; sp++; }
        if (one) { sm[sp] = one; sb[sp] = b - 1; sp_[sp] = prefix | ((uint32_t)1 << b); sp++; }
    }
    return n;
}
/* histogram shape of order m */
TR_NUM_HD int32_t tr_bool_nbins(int32_t m) { return m ? (int32_t)(((int64_t)1 << m) + 1) : 2; }
TR_NUM_HD int64_t tr_bool_lo(int32_t m) { return m ? -((int64_t)1 << (m - 1)) : 0; }

/* the per-query plan the evaluators read (built on the host by tr_bool_plan_queries) */
typedef struct {
    uint64_t vars;
    int32_t order, flags;
    uint32_t cone, ncone;        /* the query's cone: cones[cone .. cone + ncone) */
    int32_t nbins, reserved;
    int64_t lo;
    uint64_t hist, cof;          /* output offsets (cof: TR_BOOL_NONE) */
} tr_bool_plan;                  /* 56 bytes */

/* ------------------------------------------------------------------ host-side validation and planning (both libraries) */
/* 0, or TR_MATH_INVALID (-1) for a malformed view, TR_MATH_UNSUPPORTED (-12) beyond the representation limits. A view
 * is checked completely (the deps and schedule against the nodes), in O(nnodes) and without allocating. */
static inline int32_t tr_bool_program_check(const tr_bool_program_view *p) {
    if (!p || p->k < 0 || p->nnodes < 1 || !p->nodes || !p->deps || !p->schedule || p->nsched < 1 ||
        p->nsched > p->nnodes || p->root < 0 || p->root >= p->nnodes) return -1;
    if (p->k > TR_BOOL_MAX_CONDITIONS || p->nnodes > TR_BOOL_MAX_NODES) return -12;
    uint64_t prev = 0;
    for (int32_t s = 0; s < p->nsched; s++) {
        uint32_t i = p->schedule[s];
        if (i >= (uint32_t)p->nnodes || (s && i <= prev)) return -1;
        prev = i;
    }
    if (p->schedule[p->nsched - 1] != (uint32_t)p->root) return -1;
    for (int32_t i = 0; i < p->nnodes; i++) {
        const tr_bool_node *n = &p->nodes[i];
        uint64_t d = 0;
        if (n->reserved != 0) return -1;
        switch (n->op) {
        case TR_BOOL_CONST: if (n->a < 0 || n->a > 2 || n->b != -1) return -1; break;
        case TR_BOOL_VAR:
            if (n->a < 0 || n->a >= p->k || n->b != -1) return -1;
            d = (uint64_t)1 << n->a; break;
        case TR_BOOL_NOT: if (n->a < 0 || n->a >= i || n->b != -1) return -1; d = p->deps[n->a]; break;
        case TR_BOOL_AND: case TR_BOOL_OR: case TR_BOOL_XOR:
            if (n->a < 0 || n->a >= i || n->b < 0 || n->b >= i) return -1;
            d = p->deps[n->a] | p->deps[n->b]; break;
        default: return -1;
        }
        if (p->deps[i] != d) return -1;
    }
    /* the schedule ends at the root and is closed under operands, so every node it evaluates reads only nodes that
     * were evaluated before it (tr_bool_program_get_view gives exactly the nodes the root reaches) */
    for (int32_t s = p->nsched - 1; s >= 0; s--) {
        const tr_bool_node *n = &p->nodes[p->schedule[s]];
        int32_t ops[2] = {n->op == TR_BOOL_CONST || n->op == TR_BOOL_VAR ? -1 : n->a,
                          n->op >= TR_BOOL_AND ? n->b : -1};
        for (int j = 0; j < 2; j++) {
            if (ops[j] < 0) continue;
            int32_t lo = 0, hi = s - 1, found = 0;            /* binary search in the increasing schedule */
            while (lo <= hi) {
                int32_t mid = lo + (hi - lo) / 2;
                if (p->schedule[mid] == (uint32_t)ops[j]) { found = 1; break; }
                if (p->schedule[mid] < (uint32_t)ops[j]) lo = mid + 1; else hi = mid - 1;
            }
            if (!found) return -1;
        }
    }
    return 0;
}
/* 0, or TR_MATH_INVALID / TR_MATH_UNSUPPORTED for a malformed domain view */
static inline int32_t tr_bool_domain_check(const tr_bool_domain *d) {
    if (!d || d->k < 0) return -1;
    if (d->kind == TR_BOOL_ASSIGNMENTS) {
        if (d->k > TR_BOOL_MAX_ASSIGN_VARS) return -12;
        uint64_t points = (uint64_t)1 << d->k;
        if (d->points != points || d->selected != points || d->nw != (points + 63) / 64 || d->selection || d->value ||
            d->valid) return -1;
        return 0;
    }
    if (d->kind != TR_BOOL_ROWS) return -1;
    if (d->k > TR_BOOL_MAX_CONDITIONS) return -12;
    if (d->nw != (d->points + 63) / 64 || (d->nw && !d->selection) || (d->nw && d->k && !d->value)) return -1;
    return 0;
}
/* checks the queries and fills plans[nq]; with cones == NULL only counts: *ncones = the total cone length, and the
 * output totals. Returns 0 or TR_MATH_INVALID / TR_MATH_UNSUPPORTED. */
static inline int32_t tr_bool_plan_queries(const tr_bool_program_view *p, const tr_bool_query *q, int32_t nq,
                                           tr_bool_plan *plans, uint32_t *cones, uint64_t *ncones,
                                           uint64_t *hist_total, uint64_t *cof_total, uint64_t *tensor,
                                           int32_t *max_order) {
    if (nq < 0 || (nq && !q)) return -1;
    if (nq > TR_BOOL_MAX_QUERIES) return -12;
    uint64_t at = 0, hist = 0, cof = 0, tens = 0;
    int32_t top = 0;
    for (int32_t j = 0; j < nq; j++) {
        uint64_t vars = q[j].vars;
        if ((p->k < 64 && (vars >> p->k)) || q[j].reserved != 0 || (q[j].flags & ~TR_BOOL_COFACTORS)) return -1;
        int32_t m = tr_bool_popc(vars);
        if (m > TR_BOOL_MAX_ORDER) return -12;
        if (m > top) top = m;
        uint32_t nc = 0;
        for (int32_t s = 0; s < p->nsched; s++) {
            uint32_t i = p->schedule[s];
            if (p->deps[i] & vars) { if (cones) cones[at + nc] = i; nc++; }
        }
        if (plans) {
            tr_bool_plan *P = &plans[j];
            P->vars = vars; P->order = m; P->flags = q[j].flags; P->cone = (uint32_t)at; P->ncone = nc;
            P->nbins = tr_bool_nbins(m); P->reserved = 0; P->lo = tr_bool_lo(m);
            P->hist = hist; P->cof = (q[j].flags & TR_BOOL_COFACTORS) ? cof : TR_BOOL_NONE;
        }
        at += nc;
        hist = tr_budget_add(hist, (uint64_t)tr_bool_nbins(m));
        if (q[j].flags & TR_BOOL_COFACTORS) cof = tr_budget_add(cof, tr_budget_mul(3, (uint64_t)1 << m));
        tens = tr_budget_add(tens, (uint64_t)1 << m);
    }
    if (ncones) *ncones = at;
    if (hist_total) *hist_total = hist;
    if (cof_total) *cof_total = cof;
    if (tensor) *tensor = tens;
    if (max_order) *max_order = top;
    return 0;
}
/* the needs of a differences request whose cone total and output totals are known (see Budgets); host_bytes: the CPU
 * scratch (plans, cones, node words) */
static inline void tr_bool_differences_base_need(const tr_bool_program_view *p, const tr_bool_domain *d, int32_t nq,
                                                 uint64_t ncones, uint64_t hist_total, uint64_t cof_total,
                                                 uint64_t tensor, int32_t max_order, tr_math_need *need) {
    for (int i = 0; i < TR_BUDGET_RESOURCES; i++) need->need[i] = 0;
    need->need[TR_BUDGET_VARIABLES] = (uint64_t)p->k;
    need->need[TR_BUDGET_DEGREE] = (uint64_t)max_order;
    need->need[TR_BUDGET_CIRCUIT] = (uint64_t)p->nnodes;
    need->need[TR_BUDGET_SUPPORT] = d->kind == TR_BOOL_ASSIGNMENTS ? d->points : 0;
    need->need[TR_BUDGET_TENSOR] = tensor;
    need->need[TR_BUDGET_OUTPUT] = tr_budget_add(tr_budget_add((uint64_t)nq, hist_total), cof_total);
    need->need[TR_BUDGET_BYTES] = tr_budget_add(tr_budget_add(tr_budget_mul((uint64_t)nq, sizeof(tr_bool_plan)),
                                                              tr_budget_mul(ncones, 4)),
                                                tr_budget_mul((uint64_t)p->nnodes, 16));
}

/* ------------------------------------------------------------------ the CUDA work split (shared with the host need) */
#define TR_BOOL_WORK (UINT64_C(1) << 14)          /* about this many work items (query, word slice) per launch */
#define TR_BOOL_MIN_WORDS 4                        /* at least this many words per work item */
#define TR_BOOL_SCRATCH (UINT64_C(1) << 26)        /* at most this many bytes of node words on the device */
/* slices per query: work items = nq * slices, each with its own node words (16 per node) */
static inline uint64_t tr_bool_slices(uint64_t nw, int32_t nq, int32_t nnodes) {
    uint64_t q = nq > 0 ? (uint64_t)nq : 1, per = tr_budget_mul((uint64_t)nnodes, 16);
    uint64_t s = TR_BOOL_WORK / q, cap = (nw + TR_BOOL_MIN_WORDS - 1) / TR_BOOL_MIN_WORDS;
    uint64_t room = per ? TR_BOOL_SCRATCH / per / q : s;
    if (s > cap) s = cap;
    if (s > room) s = room;
    return s ? s : 1;
}
/* the device bytes of tr_mgpu_bool_differences (the host plans and cones are counted as well) */
static inline uint64_t tr_bool_device_bytes(const tr_bool_program_view *p, const tr_bool_domain *d, int32_t nq,
                                            uint64_t ncones, uint64_t hist_total, uint64_t cof_total) {
    uint64_t bytes = 0, words = d->nw;
    if (d->kind == TR_BOOL_ROWS) {
        bytes = tr_budget_mul(words, 8);
        bytes = tr_budget_add(bytes, tr_budget_mul(tr_budget_mul((uint64_t)d->k, words), d->valid ? 16 : 8));
    }
    bytes = tr_budget_add(bytes, tr_budget_mul((uint64_t)p->nnodes, 16));                 /* nodes */
    bytes = tr_budget_add(bytes, tr_budget_mul((uint64_t)p->nsched, 4));                  /* schedule */
    bytes = tr_budget_add(bytes, tr_budget_mul(tr_budget_mul((uint64_t)nq, sizeof(tr_bool_plan)), 2));   /* + host */
    bytes = tr_budget_add(bytes, tr_budget_mul(ncones, 8));                               /* cones, + host */
    uint64_t items = tr_budget_mul((uint64_t)(nq > 0 ? nq : 0), tr_bool_slices(words, nq, p->nnodes));
    bytes = tr_budget_add(bytes, tr_budget_mul(items, tr_budget_mul((uint64_t)p->nnodes, 16)));   /* node words */
    bytes = tr_budget_add(bytes, tr_budget_mul((uint64_t)nq, 160));                       /* accumulators, + host */
    bytes = tr_budget_add(bytes, tr_budget_mul(tr_budget_add(hist_total, cof_total), 8));  /* bins, cofactor counts */
    return bytes;
}
/* the device bytes of tr_mgpu_bool_masks: the selection, every column a condition reads (codes or values, validity),
 * the conditions and the output words */
static inline uint64_t tr_bool_masks_device_bytes(const tr_scope_view *v, const tr_bool_condition *c, int32_t k) {
    uint64_t words = (v->rows + 63) / 64, bytes = tr_budget_mul(words, 8), used = 0;
    for (int32_t i = 0; i < k; i++) {
        if (c[i].col >= 0 && c[i].col < 64) used |= (uint64_t)1 << c[i].col;
        if (c[i].op >= TR_BOOL_COL_EQ && c[i].col2 >= 0 && c[i].col2 < 64) used |= (uint64_t)1 << c[i].col2;
    }
    for (int32_t col = 0; col < v->ncols && col < TR_SCOPE_MAX_COLUMNS; col++) {
        if (!((used >> col) & 1)) continue;
        bytes = tr_budget_add(bytes, tr_budget_mul(v->rows, v->col[col].values ? 8 : 4));
        if (v->col[col].valid) bytes = tr_budget_add(bytes, v->rows);
    }
    bytes = tr_budget_add(bytes, tr_budget_mul((uint64_t)k, sizeof(tr_bool_condition)));
    return tr_budget_add(bytes, tr_budget_mul(tr_budget_mul((uint64_t)k, words), 16));
}
/* 0, or TR_MATH_INVALID for a condition the scope view cannot evaluate (column out of range, bad op or constants) */
static inline int32_t tr_bool_condition_check(const tr_scope_view *v, const tr_bool_condition *c) {
    if (c->reserved != 0 || c->col < 0 || c->col >= v->ncols) return -1;
    if (c->op < TR_BOOL_EQ || c->op > TR_BOOL_COL_GE) return -1;
    if (c->op >= TR_BOOL_COL_EQ) { if (c->col2 < 0 || c->col2 >= v->ncols) return -1; }
    else if (c->col2 != -1) return -1;
    if (c->den[0] < 1 || c->den[1] < 1) return -1;
    if (c->op == TR_BOOL_BIT && (c->num[1] < 0 || c->num[1] > 63 || c->den[0] != 1 || c->den[1] != 1)) return -1;
    if ((c->op == TR_BOOL_IS_NULL || c->op == TR_BOOL_IS_NOT_NULL || c->op >= TR_BOOL_COL_EQ) &&
        (c->num[0] || c->num[1] || c->den[0] != 1 || c->den[1] != 1)) return -1;
    if (c->op >= TR_BOOL_EQ && c->op <= TR_BOOL_GE && (c->num[1] || c->den[1] != 1)) return -1;
    return 0;
}

/* ------------------------------------------------------------------ masks (libalg) */
/* ROWS: the k conditions over the selected rows of the scope; budget NULL = defaults */
tr_bool_masks *tr_bool_masks_build(const tr_scope *, const tr_bool_condition *, int32_t k, const tr_math_budget *,
                                   tr_math_refusal *, int32_t *error);
/* ASSIGNMENTS: the 2^k assignments of k variables (k <= TR_BOOL_MAX_ASSIGN_VARS) */
tr_bool_masks *tr_bool_masks_assignments(int32_t k, const tr_math_budget *, tr_math_refusal *, int32_t *error);
/* Adopts words computed elsewhere (the readback of tr_mgpu_bool_masks) as the masks of the conditions over the scope:
 * value and valid are k * nw words each (valid NULL: every condition known); checked canonical (value within valid
 * within the scope's selection) and copied. */
tr_bool_masks *tr_bool_masks_adopt(const tr_scope *, const tr_bool_condition *, int32_t k, const uint64_t *value,
                                   const uint64_t *valid, const tr_math_budget *, tr_math_refusal *, int32_t *error);
/* the needs checked by tr_bool_masks_build (TR_BOOL_NEED_CPU), tr_mgpu_bool_masks (_DEVICE) or tr_bool_masks_adopt
 * (_ADOPT); scope NULL: tr_bool_masks_assignments */
int32_t tr_bool_masks_need(const tr_scope *, const tr_bool_condition *, int32_t k, int32_t target, tr_math_need *);
int32_t tr_bool_masks_get_info(const tr_bool_masks *, tr_bool_masks_info *);
int32_t tr_bool_masks_get_domain(const tr_bool_masks *, tr_bool_domain *);
/* condition i over the points that count: counts[0..2] = TRUE, FALSE, UNKNOWN */
int32_t tr_bool_masks_counts(const tr_bool_masks *, int32_t i, uint64_t counts[3]);
/* 1 when both describe the same domain with identical words, 0 otherwise */
int32_t tr_bool_masks_equal(const tr_bool_masks *, const tr_bool_masks *);
void tr_bool_masks_free(tr_bool_masks *);

/* ------------------------------------------------------------------ programs (libalg) */
tr_bool_program *tr_bool_program_create(int32_t k, const tr_bool_node *, int32_t nnodes, int32_t root,
                                        const tr_math_budget *, tr_math_refusal *, int32_t *error);
int32_t tr_bool_program_need(int32_t k, int32_t nnodes, tr_math_need *);
int32_t tr_bool_program_get_view(const tr_bool_program *, tr_bool_program_view *);
int32_t tr_bool_program_get_info(const tr_bool_program *, tr_bool_program_info *);
void tr_bool_program_free(tr_bool_program *);

/* ------------------------------------------------------------------ cofactors and differences (libalg) */
/* the output totals of a list of queries for a program (bins, cofactor counts); also checks the queries */
int32_t tr_bool_layout(const tr_bool_program *, const tr_bool_query *, int32_t nq, uint64_t *hist_total,
                       uint64_t *cof_total);
/* the needs checked by tr_bool_differences (TR_BOOL_NEED_CPU) or tr_mgpu_bool_differences (TR_BOOL_NEED_DEVICE) */
int32_t tr_bool_differences_need(const tr_bool_masks *, const tr_bool_program *, const tr_bool_query *, int32_t nq,
                                 int32_t target, tr_math_need *);
/* Shared-mask execution of nq queries over the masks: results[nq], hist[hist_total], cof[cof_total] (tr_bool_layout;
 * NULL when the total is 0). A program over another k than the masks: TR_MATH_DOMAIN_ERROR. */
int32_t tr_bool_differences(const tr_bool_masks *, const tr_bool_program *, const tr_bool_query *, int32_t nq,
                            const tr_math_budget *, tr_bool_result *results, uint64_t *hist, uint64_t *cof,
                            tr_math_refusal *);
/* the same on views (for native callers that hold views; both are checked completely) */
int32_t tr_bool_differences_view(const tr_bool_domain *, const tr_bool_program_view *, const tr_bool_query *,
                                 int32_t nq, const tr_math_budget *, tr_bool_result *results, uint64_t *hist,
                                 uint64_t *cof, tr_math_refusal *);
/* Independent evaluation of one cofactor f|S:=t (vars = S, values = t as a bit per condition of S, bit i = condition
 * i; values outside vars must be 0), point by point: over the selected rows of the scope with the conditions read from
 * its values, or (scope NULL, conditions NULL) over the 2^k assignments. counts[0..2] = TRUE, FALSE, UNKNOWN. */
int32_t tr_bool_direct(const tr_scope *, const tr_bool_condition *, const tr_bool_program *, uint64_t vars,
                       uint64_t values, const tr_math_budget *, uint64_t counts[3], tr_math_refusal *);

/* ------------------------------------------------------------------ the CUDA library (math_bool_gpu.cu, libmathgpu)
 * tr_mgpu_bool_masks: the condition words of tr_bool_masks_build over a scope view (tr_scope_get_view; host pointers,
 *   copied to the device and budgeted with tr_bool_masks_device_bytes), written to value and valid (k * nw words each;
 *   then adopt them with tr_bool_masks_adopt). One work item per (condition, word).
 * tr_mgpu_bool_differences: tr_bool_differences_view on the device (views of host memory, copied; budgeted with the
 *   needs of tr_bool_differences_base_need, its bytes replaced by tr_bool_device_bytes). Work items (query, word
 *   slice) evaluate their words as the CPU does and add their counts with integer atomics. */
int tr_mgpu_bool_masks(const tr_scope_view *, const tr_bool_condition *, int32_t k, const tr_math_budget *,
                       uint64_t *value, uint64_t *valid, tr_math_refusal *);
int tr_mgpu_bool_differences(const tr_bool_domain *, const tr_bool_program_view *, const tr_bool_query *, int32_t nq,
                             const tr_math_budget *, tr_bool_result *results, uint64_t *hist, uint64_t *cof,
                             tr_math_refusal *);
#ifdef __cplusplus
}
#endif
#endif
