/* ATTR-002: bounded explanation and repair search: redundant predicates, minimal distinguishing sets and least-cost
 * feasible repairs, exact where the search finishes, explicitly incomplete with proven bounds where a declared state
 * budget stops it; on the CPU (math_repair.c) and with batched CUDA evaluation (math_repair_gpu.cu), bit for bit.
 * SPDX-License-Identifier: Apache-2.0
 *
 * ---- Subset problems (one immutable tr_repair_problem, read by the evaluators as a tr_repair_view) --------------
 * Elements 0..n-1 (n <= TR_REPAIR_MAX_ELEMENTS: subsets are uint64 masks); `allowed` the elements a solution may use.
 * A problem decides, for a subset S of the allowed elements, whether S is feasible, with exact counts:
 *   HIT      a hitting set: S meets every mask of a list of inclusion-minimal masks M (count = masks missed). Built by
 *            SEPARATE  rows (or groups) A and B of a BOOL-001 domain: S separates them when every pair (a, b) differs
 *                      on a condition of S (convention VALUES: the three-valued values TRUE / FALSE / UNKNOWN differ;
 *                      SELECTED: one is TRUE and the other is not, SQL WHERE). M = the minimal difference sets of the
 *                      distinct patterns of A and B; a pair with no difference makes the problem infeasible, with a
 *                      witness pair of rows. EXPLAIN: A = the points a BOOL-001 program selects, B = the other points:
 *                      S explains membership (membership is a function of the conditions of S).
 *            KEEP      a conjunction AND_{i in C} c_i over a domain: S (a subset of C) keeps the selection exactly iff it
 *                      meets the failure set {i in C : c_i not TRUE} of every point the conjunction rejects (M = the
 *                      minimal failure sets): the conditions outside a minimum S are jointly redundant.
 *            MASKS     declared masks.
 *   CIRCUIT  a BOOL-001 program f over a domain, every condition outside S replaced by its declared baseline constant
 *            (TRUE / FALSE): S is feasible iff f_S selects exactly the points f selects (count = points where they
 *            differ; SQL three-valued logic: selected means TRUE); allowed by default: the conditions f reads.
 *   GAME     an ATTR-001 game (copied): EXPLAIN v(S) op t, REPAIR v(P - S) op t (count = 0 / 1, value = v).
 *   DELETE   declared deletable rows r_0..r_{n-1} of a scope (n <= 64), the other selected rows fixed: S feasible iff
 *            AGG over the qualifying rows that remain op t, AGG one of COUNT(*), COUNT(v), SUM(v), AVG(v) (SQL: a NULL v
 *            is not counted, SUM / AVG of no value is NULL and attains no target), qualifying = selected and TRUE under
 *            a BOOL-001 program (or every selected row); t an exact rational in value units (count = rows remaining).
 * Redundancy (tr_repair_redundancy): element i of `allowed` is redundant iff S = allowed - {i} is feasible: for a
 * CIRCUIT problem, removing condition i (its baseline constant) leaves the selected set unchanged, certified by the
 * exact count of points where the selections differ (0), with the first witness rows otherwise (gained: selected only
 * without condition i; lost: the reverse); for a GAME problem built on a LATTICE-002 conjunction game, v(C - i) = v(C).
 *
 * ---- Searches (tr_repair_search, the shared drivers below) --------------------------------------------------------
 * Costs: positive exact rationals per element (unit by default), scaled natively to integers over their common
 * denominator `den` (the scaled costs and their total must fit int64, else TR_MATH_OVERFLOW). The minimum cost over the
 * feasible subsets of the allowed elements, every minimum-cost subset (all ties; listed up to max_solutions, counted
 * exactly when the search completes) and its cost.
 *   EXHAUSTIVE    every subset in increasing (cost, mask) order (unit costs: levels |S| = 0, 1, ... in increasing mask
 *                 order; weighted: the sorted order of the 2^m subsets, m <= TR_REPAIR_MAX_SORTED), chunk by chunk;
 *                 the first feasible rank gives the minimum, the rest of its cost level the ties.
 *   BRANCH_BOUND  best-first over include / exclude decisions in a declared element order, nodes ordered by (lower
 *                 bound, I, X); a node's bound is cost(I) plus an admissible estimate: HIT: a greedy packing of masks
 *                 missed by I that are disjoint on the undecided elements (each needs its cheapest undecided element);
 *                 DELETE with SUM / COUNT / AVG and an order comparison: the fractional (LP) relaxation of the one
 *                 linear constraint, rounded up; otherwise the cheapest undecided element when I is infeasible. Nodes
 *                 are pruned only soundly: a bound above the incumbent (ties kept), no undecided element left, a
 *                 monotone problem whose completion I + U is infeasible, or an infeasible relaxation. A feasible I is a
 *                 solution and is not expanded (positive costs).
 *   Orders (branch and bound): INDEX, COST (cheapest first), SHAPLEY / BANZHAF (decreasing ATTR-001 index of a declared
 *   game over the same elements, exact: tr_attr_exhaustive, or the sparse support for large provenance games). The
 *   order changes the work, never the answer.
 * Every state (a subset or a node's evaluation) counts against max_states. The answer: FOUND (the minimum cost exact,
 * its multiplicity exact, the solutions listed), NONE (proven: every subset examined or pruned soundly), or INCOMPLETE
 * when max_states stopped the search: lower <= minimum <= upper, both proven (upper: a feasible subset found, else
 * none), the explored states and open nodes counted, and the multiplicity bounded when lower = upper (at least the
 * ties found; at most those plus the subsets of that cost not examined: exhaustive, the rest of the cost level; branch
 * and bound, every subset below an open node at that bound; otherwise 0 and unknown). Never a claimed minimum without
 * proof.
 *
 * ---- Continuous repairs (INV-002, math_interv.h; CPU) -------------------------------------------------------------
 * EDIT       value edits of declared rows (n <= TR_INTERV_MAX_DIMS) of the value column v so that SUM(v) or AVG(v)
 *            over the qualifying rows op t (qualification does not read v; the edited values must be present): the
 *            exact LP / projection of tr_interv_linear (costs l1, linf, l2, l0 with weights; bounds on the edits).
 * THRESHOLD  the constant c of one declared comparison condition x op c of a conjunction (the others fixed: the rows
 *            they keep TRUE) changed to c' so that AGG(v) op t over the rows that then qualify: INV-002's aggregate
 *            state (tr_interv_agg_solve: rows entering and leaving the predicate decided exactly, c' = c - d), cost
 *            w |c' - c|; over several declared conditions the minimum cost and every condition attaining it.
 * Results are INV-002 results: algebraic interventions, never causal claims.
 *
 * ---- CUDA ---------------------------------------------------------------------------------------------------------
 * tr_mgpu_repair_eval: one work item per candidate subset (exact counts and values into its own slot); tr_mgpu_repair_scan:
 * one work item per candidate, the feasible bitmap by atomic OR, the smallest feasible cost by atomic minimum, then
 * the ties of the minimum cost (atomic OR): independent of thread order; tr_mgpu_repair_search runs the shared drivers
 * with them. Equal to the CPU bit for bit.
 *
 * ---- Budgets (math_budgets.h, checked before allocation; TR_MATH_BUDGET with the refusal) ------------------------
 *   problems   variables = elements; tensor = pattern pairs (SEPARATE / EXPLAIN: |patterns A| x |patterns B|);
 *              support = the masks kept (MASKS: the declared ones, copied before they are minimized; the others:
 *              the minimal difference / failure sets); circuit = program nodes; bytes = the copies and the scratch
 *   evaluate   output = candidates, bytes (CUDA: every device copy and the scratch chunk)
 *   search     iterations = max_states (the declared exploration budget: a larger space gives INCOMPLETE, never a
 *              refusal), output = max_solutions, bytes = the frontier (2 max_states nodes), the batches and the sorted
 *              order (16 x 2^m); SHAPLEY order: the game's exhaustive or support needs (ATTR-001)
 *   redundancy output = allowed elements x (1 + max_witnesses)
 *   threshold  variables = conditions, output = changeable conditions, bytes = the rows' flags; then INV-002's needs
 *   edit       INV-002's needs (tr_interv_linear)
 * Representation limits (TR_MATH_UNSUPPORTED): more than TR_REPAIR_MAX_ELEMENTS elements, a weighted exhaustive
 * search above TR_REPAIR_MAX_SORTED allowed elements, a program above TR_REPAIR_MAX_NODES nodes for the device.
 * Errors: TR_MATH_INVALID, _MEMORY, _OVERFLOW (an exact value outside its carrier, never wrapped), _BUDGET,
 * _UNSUPPORTED, TR_MATH_DOMAIN_ERROR (a program over another number of conditions than the domain), TR_MATH_INFEASIBLE
 * (an edit whose SUM / AVG has no present value to change), TR_MATH_DEVICE. A separation with two indistinguishable
 * rows is a problem like the others: tr_repair_problem_info reports it infeasible with the witness pair, and its
 * searches answer NONE (proven). */
#ifndef TORBLES_MATH_REPAIR_H
#define TORBLES_MATH_REPAIR_H
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "math_num.h"
#include "math_budgets.h"
#include "math_bool.h"
#include "math_attr.h"
#include "math_interv.h"
#include "math_scope.h"
#ifdef __cplusplus
extern "C" {
#endif

#define TR_REPAIR_ABI 1
#define TR_REPAIR_MAX_ELEMENTS 62           /* subsets are uint64 masks */
#define TR_REPAIR_MAX_SORTED 20             /* weighted exhaustive search: 2^m subsets sorted by cost */
#define TR_REPAIR_MAX_NODES 4096            /* program nodes evaluated on the device */
#define TR_REPAIR_BATCH 1024                /* candidates per evaluation batch of the drivers */
#define TR_REPAIR_DEVICE_SCRATCH (UINT64_C(1) << 26)   /* bytes of device scratch per launch (at most) */
#define TR_REPAIR_NONE UINT64_MAX
#define TR_REPAIR_UNKNOWN UINT64_MAX

enum { TR_REPAIR_HIT = 1, TR_REPAIR_CIRCUIT = 2, TR_REPAIR_GAME = 3, TR_REPAIR_DELETE = 4 };          /* evaluators */
enum { TR_REPAIR_SRC_MASKS = 1, TR_REPAIR_SRC_SEPARATE = 2, TR_REPAIR_SRC_EXPLAIN = 3, TR_REPAIR_SRC_KEEP = 4,
       TR_REPAIR_SRC_PROGRAM = 5, TR_REPAIR_SRC_GAME = 6, TR_REPAIR_SRC_DELETE = 7 };                /* sources */
enum { TR_REPAIR_VALUES = 1, TR_REPAIR_SELECTED = 2 };                                               /* separation */
enum { TR_REPAIR_EXPLAIN = 1, TR_REPAIR_REPAIR = 2 };                                                /* game kinds */
enum { TR_REPAIR_EQ = 1, TR_REPAIR_NE = 2, TR_REPAIR_LT = 3, TR_REPAIR_LE = 4, TR_REPAIR_GT = 5, TR_REPAIR_GE = 6 };
enum { TR_REPAIR_COUNT_STAR = 1, TR_REPAIR_COUNT = 2, TR_REPAIR_SUM = 3, TR_REPAIR_AVG = 4 };          /* aggregates */
enum { TR_REPAIR_AUTO = 0, TR_REPAIR_EXHAUSTIVE = 1, TR_REPAIR_BRANCH_BOUND = 2 };                   /* methods */
enum { TR_REPAIR_ORDER_INDEX = 0, TR_REPAIR_ORDER_COST = 1, TR_REPAIR_ORDER_SHAPLEY = 2,
       TR_REPAIR_ORDER_BANZHAF = 3 };
enum { TR_REPAIR_FOUND = 1, TR_REPAIR_NONE_FEASIBLE = 2, TR_REPAIR_INCOMPLETE = 3 };                  /* statuses */
enum { TR_REPAIR_STOP_COMPLETE = 1, TR_REPAIR_STOP_BUDGET = 2, TR_REPAIR_STOP_EXHAUSTED = 3 };
enum { TR_REPAIR_NEED_CPU = 0, TR_REPAIR_NEED_DEVICE = 1 };

/* ------------------------------------------------------------------ structs crossing the ctypes boundary */
/* a problem as the evaluators read it (pointers into the problem; valid while it lives) */
typedef struct {
    int32_t kind, n, op, agg;                /* TR_REPAIR_HIT..; elements; GAME / DELETE comparison; DELETE aggregate */
    int32_t explain, monotone, vscale, reserved;   /* GAME explain (1) / repair (0); +1: supersets of a feasible S are
                                                      feasible; DELETE: the value column's decimal scale */
    uint64_t allowed;                        /* the elements a solution may use */
    uint64_t nmasks;                         /* HIT */
    const uint64_t *masks;                   /* HIT: nmasks inclusion-minimal masks */
    tr_bool_domain dom;                      /* CIRCUIT */
    tr_bool_program_view prog;               /* CIRCUIT */
    uint64_t baseline;                       /* CIRCUIT: bit i = condition i's constant outside S (1 TRUE) */
    const uint64_t *target;                  /* CIRCUIT: dom.nw words, the points f selects (TRUE) */
    tr_attr_game_view game;                  /* GAME */
    tr_math_i128 threshold;                  /* GAME */
    const uint8_t *qual, *vvalid;            /* DELETE: n flags: the row qualifies; its value is present */
    const int64_t *vval;                     /* DELETE: n logical values */
    uint64_t base_count, base_vcount;        /* DELETE: the qualifying fixed rows; those with a value */
    tr_math_i128 base_sum;                   /* DELETE: the sum of their values (logical) */
    tr_math_i128 tnum;                       /* DELETE: t = tnum / tden; SUM and AVG compare against tnum 10^vscale */
    int64_t tden, reserved2;
} tr_repair_view;                            /* 472 bytes */

typedef struct {
    tr_math_i128 value;                      /* HIT: masks missed; CIRCUIT: points f_S selects; GAME: v; DELETE: the
                                                aggregate's numerator (COUNTs: the count; SUM, AVG: the logical sum) */
    uint64_t count;                          /* HIT: masks missed; CIRCUIT: points where f_S and f differ; GAME: 1 if
                                                infeasible; DELETE: qualifying rows remaining */
    int32_t feasible, null;                  /* DELETE: the aggregate is SQL NULL */
} tr_repair_out;                             /* 32 bytes */

typedef struct {
    int32_t method, order, reserved0, reserved1;   /* TR_REPAIR_AUTO / _EXHAUSTIVE / _BRANCH_BOUND; TR_REPAIR_ORDER_* */
    uint64_t max_states, max_solutions;      /* states evaluated; minimum solutions listed */
    const tr_bigq *const *cost;              /* n positive costs (NULL: unit costs) */
} tr_repair_query;                           /* 40 bytes */

/* a prepared search (costs scaled, order fixed) as the drivers read it */
typedef struct {
    int32_t method, order, n, m;             /* the method that runs; m = allowed elements */
    uint64_t allowed, max_states, max_solutions;
    int64_t den;                             /* costs are cost[i] / den */
    int64_t unit;                            /* 1: every allowed cost is 1 */
    const int64_t *cost;                     /* n integer costs (>= 1 on the allowed elements) */
    const uint32_t *perm;                    /* m: the allowed elements in branching order */
} tr_repair_plan;                            /* 72 bytes */

typedef struct {
    int32_t status, reason, method, order;   /* TR_REPAIR_FOUND / _NONE_FEASIBLE / _INCOMPLETE; TR_REPAIR_STOP_* */
    int32_t n, m, complete, reserved;        /* complete: the multiplicity is exact */
    int64_t lower, upper;                    /* proven lower <= minimum <= upper (cost numerators over den); upper -1:
                                                no feasible subset found */
    int64_t den, reserved2;
    uint64_t multiplicity, mult_hi;          /* minimum solutions found; at most (TR_REPAIR_UNKNOWN: unknown) */
    uint64_t explored, open, pruned, listed; /* states evaluated; open nodes at the stop; nodes pruned; solutions written */
    uint64_t total;                          /* 2^m (saturated) */
} tr_repair_info;                            /* 120 bytes */

typedef struct {
    int32_t kind, source, n, convention;     /* TR_REPAIR_HIT..; TR_REPAIR_SRC_*; elements; separation convention */
    uint64_t allowed, nmasks;                /* HIT: minimal masks */
    uint64_t patterns_a, patterns_b, pairs;  /* SEPARATE / EXPLAIN / KEEP: distinct patterns; pairs compared */
    uint64_t witness_a, witness_b;           /* an infeasible separation: rows a, b with no difference (else NONE) */
    uint64_t points, selected;               /* CIRCUIT / DELETE: points of the domain; points f selects */
    uint64_t bytes;
    int32_t infeasible, monotone;            /* 1: no subset of the allowed elements is feasible (proven) */
} tr_repair_problem_info;                    /* 104 bytes */

typedef struct {
    int32_t element, redundant;              /* the condition; 1: removing it changes nothing */
    uint64_t count;                          /* points where the selections differ (masks missed, ...) */
    uint64_t gained, lost;                   /* CIRCUIT: selected only without the condition / only with it */
    uint64_t witnesses;                      /* witness rows written (gained ones first) */
    tr_math_i128 value;                      /* the evaluator's value at allowed - {i} */
} tr_repair_removal;                         /* 56 bytes */

typedef struct tr_repair_problem_s tr_repair_problem;

/* ================================================================ shared evaluation (CPU code and kernels) */
TR_NUM_HD int tr_repair_test(int32_t op, int s) {   /* s = sign(value - t) */
    switch (op) {
    case TR_REPAIR_EQ: return s == 0;
    case TR_REPAIR_NE: return s != 0;
    case TR_REPAIR_LT: return s < 0;
    case TR_REPAIR_LE: return s <= 0;
    case TR_REPAIR_GT: return s > 0;
    default: return s >= 0;
    }
}
/* |a| as 128 bits, and the 256-bit magnitude of a b in r[0..3] (least significant first) */
TR_NUM_HD unsigned __int128 tr_repair_uabs(__int128 a) {
    return a < 0 ? (unsigned __int128)0 - (unsigned __int128)a : (unsigned __int128)a;
}
TR_NUM_HD void tr_repair_umul(unsigned __int128 x, unsigned __int128 y, uint64_t *r) {
    uint64_t a[2] = {(uint64_t)x, (uint64_t)(x >> 64)}, b[2] = {(uint64_t)y, (uint64_t)(y >> 64)};
    r[0] = r[1] = r[2] = r[3] = 0;
    for (int i = 0; i < 2; i++) {
        unsigned __int128 carry = 0;
        for (int j = 0; j < 2; j++) {
            unsigned __int128 t = (unsigned __int128)a[i] * b[j] + r[i + j] + carry;
            r[i + j] = (uint64_t)t;
            carry = t >> 64;
        }
        r[i + 2] += (uint64_t)carry;
    }
}
/* the sign of a b - c d, exactly (|a|, |b|, |c|, |d| < 2^127) */
TR_NUM_HD int tr_repair_cmp_prod(__int128 a, __int128 b, __int128 c, __int128 d) {
    const int sp = (a == 0 || b == 0) ? 0 : ((a < 0) != (b < 0) ? -1 : 1);
    const int sq = (c == 0 || d == 0) ? 0 : ((c < 0) != (d < 0) ? -1 : 1);
    if (sp != sq) return sp > sq ? 1 : -1;
    if (!sp) return 0;
    uint64_t p[4], q[4];
    tr_repair_umul(tr_repair_uabs(a), tr_repair_uabs(b), p);
    tr_repair_umul(tr_repair_uabs(c), tr_repair_uabs(d), q);
    for (int i = 3; i >= 0; i--)
        if (p[i] != q[i]) return (p[i] > q[i] ? 1 : -1) * sp;
    return 0;
}
/* CIRCUIT: the words of f_S at word w (value and valid of every scheduled node into val: 2 words per node) */
TR_NUM_HD uint64_t tr_repair_circuit_word(const tr_repair_view *v, uint64_t S, uint64_t w, uint64_t *val) {
    const tr_bool_program_view *p = &v->prog;
    for (int32_t s = 0; s < p->nsched; s++) {
        const uint32_t i = p->schedule[s];
        const tr_bool_node *nd = &p->nodes[i];
        uint64_t x, k;
        if (nd->op == TR_BOOL_CONST) tr_bool_const_word(nd->a, &x, &k);
        else if (nd->op == TR_BOOL_VAR) {
            if ((S >> nd->a) & 1) tr_bool_var_word(&v->dom, nd->a, w, &x, &k);
            else tr_bool_const_word(((v->baseline >> nd->a) & 1) ? TR_BOOL_TRUE : TR_BOOL_FALSE, &x, &k);
        } else tr_bool_apply(nd->op, val[2 * (uint64_t)nd->a], val[2 * (uint64_t)nd->a + 1],
                             nd->b >= 0 ? val[2 * (uint64_t)nd->b] : 0, nd->b >= 0 ? val[2 * (uint64_t)nd->b + 1] : 0,
                             &x, &k);
        val[2 * (uint64_t)i] = x; val[2 * (uint64_t)i + 1] = k;
    }
    const uint32_t root = (uint32_t)p->root;
    return val[2 * (uint64_t)root] & val[2 * (uint64_t)root + 1] & tr_bool_selection_word(&v->dom, w);
}
/* the evaluation of subset S (scratch: tr_repair_scratch_words words) */
TR_NUM_HD void tr_repair_eval_one(const tr_repair_view *v, uint64_t S, uint64_t *scratch, tr_repair_out *out) {
    out->value = tr_i128_set(0); out->count = 0; out->feasible = 0; out->null = 0;
    if (v->kind == TR_REPAIR_HIT) {
        uint64_t miss = 0;
        for (uint64_t m = 0; m < v->nmasks; m++) if (!(v->masks[m] & S)) miss++;
        out->count = miss; out->value = tr_i128_set((__int128)miss); out->feasible = miss == 0;
        return;
    }
    if (v->kind == TR_REPAIR_CIRCUIT) {
        uint64_t diff = 0, sel = 0;
        for (uint64_t w = 0; w < v->dom.nw; w++) {
            if (!tr_bool_selection_word(&v->dom, w)) continue;
            const uint64_t x = tr_repair_circuit_word(v, S, w, scratch);
            diff += (uint64_t)tr_bool_popc(x ^ v->target[w]);
            sel += (uint64_t)tr_bool_popc(x);
        }
        out->count = diff; out->value = tr_i128_set((__int128)sel); out->feasible = diff == 0;
        return;
    }
    if (v->kind == TR_REPAIR_GAME) {
        const uint64_t all = v->game.n >= 64 ? ~(uint64_t)0 : (((uint64_t)1 << v->game.n) - 1);
        __int128 val = 0;
        if (!tr_attr_eval(&v->game, v->explain ? S : (all & ~S), (unsigned __int128 *)scratch, &val)) {
            out->null = 1; return;                                    /* impossible for a created game */
        }
        const __int128 t = tr_i128_get(v->threshold);
        out->value = tr_i128_set(val);
        out->feasible = tr_repair_test(v->op, val < t ? -1 : val > t ? 1 : 0);
        out->count = !out->feasible;
        return;
    }
    /* DELETE */
    uint64_t cnt = v->base_count, vc = v->base_vcount;
    __int128 sum = tr_i128_get(v->base_sum);
    for (int32_t i = 0; i < v->n; i++) {
        if (((S >> i) & 1) || !v->qual[i]) continue;
        cnt++;
        if (v->vvalid[i]) { vc++; sum += v->vval[i]; }           /* |sum| < 2^126: proven by the host */
    }
    out->count = cnt;
    const __int128 tn = tr_i128_get(v->tnum), td = v->tden;
    int s;
    if (v->agg == TR_REPAIR_COUNT_STAR || v->agg == TR_REPAIR_COUNT) {
        const __int128 c = (__int128)(v->agg == TR_REPAIR_COUNT_STAR ? cnt : vc);
        out->value = tr_i128_set(c);
        s = tr_repair_cmp_prod(c, td, tn, 1);
    } else {
        out->value = tr_i128_set(sum);
        if (!vc) { out->null = 1; return; }                          /* SUM / AVG of no value: NULL, no target */
        s = v->agg == TR_REPAIR_SUM ? tr_repair_cmp_prod(sum, td, tn, 1) : tr_repair_cmp_prod(sum, td, tn, (__int128)vc);
    }
    out->feasible = tr_repair_test(v->op, s);
}

/* ------------------------------------------------------------------ host helpers (both libraries) */
/* scratch words of one evaluation: CIRCUIT 2 per program node, GAME (provenance) 2 per plan node */
static inline uint64_t tr_repair_scratch_words(const tr_repair_view *v) {
    if (v->kind == TR_REPAIR_CIRCUIT) return tr_budget_mul(2, (uint64_t)(v->prog.nnodes > 0 ? v->prog.nnodes : 1));
    if (v->kind == TR_REPAIR_GAME && v->game.kind == TR_ATTR_GAME_PROV)
        return tr_budget_mul(2, v->game.plan.nodes ? v->game.plan.nodes : 1);
    return 2;
}
static inline uint64_t tr_repair_mask_all(int32_t n) { return n >= 64 ? ~(uint64_t)0 : (((uint64_t)1 << n) - 1); }
/* 0 when a view is well formed, else TR_MATH_INVALID / TR_MATH_UNSUPPORTED */
static inline int32_t tr_repair_view_check(const tr_repair_view *v) {
    if (!v || v->n < 1 || v->reserved || v->reserved2) return -1;
    if (v->n > TR_REPAIR_MAX_ELEMENTS) return -12;
    if (v->allowed & ~tr_repair_mask_all(v->n)) return -1;
    switch (v->kind) {
    case TR_REPAIR_HIT: return v->nmasks && !v->masks ? -1 : 0;
    case TR_REPAIR_CIRCUIT: {
        int32_t rc = tr_bool_domain_check(&v->dom);
        if (!rc) rc = tr_bool_program_check(&v->prog);
        if (rc) return rc;
        if (v->prog.k != v->dom.k || v->dom.k != v->n || !v->target) return -1;
        return 0;
    }
    case TR_REPAIR_GAME: {
        int32_t rc = tr_attr_view_check(&v->game, 1);
        if (rc) return rc;
        if (v->game.n != v->n || v->op < TR_REPAIR_EQ || v->op > TR_REPAIR_GE) return -1;
        return 0;
    }
    case TR_REPAIR_DELETE:
        if (!v->qual || !v->vvalid || !v->vval || v->tden < 1 || v->agg < TR_REPAIR_COUNT_STAR || v->agg > TR_REPAIR_AVG ||
            v->op < TR_REPAIR_EQ || v->op > TR_REPAIR_GE) return -1;
        return 0;
    default:
        return -1;
    }
}

/* the evaluators the drivers call: per candidate results, or the scan (feasible bits, minimum feasible cost by atomic
 * minimum, tie bits of the cost `match`; match = INT64_MIN: the scan's own minimum) */
typedef int32_t (*tr_repair_batch_fn)(void *ctx, const uint64_t *masks, uint64_t count, tr_repair_out *out);
typedef int32_t (*tr_repair_scan_fn)(void *ctx, const uint64_t *masks, const int64_t *costs, uint64_t count,
                                     int64_t match, uint64_t *feasible, uint64_t *ties, int64_t *mincost);

/* ------------------------------------------------------------------ the shared drivers (CPU and device evaluators) */
static inline int64_t tr_repair_cost_of(const tr_repair_plan *pl, uint64_t S) {
    int64_t c = 0;
    for (uint64_t b = S; b; b &= b - 1) c += pl->cost[__builtin_ctzll(b)];    /* the total fits int64 (checked) */
    return c;
}
/* the subset of the allowed elements with index bits `idx` (bit j: the j-th allowed element by increasing index) */
static inline uint64_t tr_repair_deposit(uint64_t allowed, uint64_t idx) {
    uint64_t out = 0;
    for (uint64_t a = allowed; a && idx; a &= a - 1, idx >>= 1) if (idx & 1) out |= a & (~a + 1);
    return out;
}
static inline uint64_t tr_repair_next_mask(uint64_t s) {      /* Gosper: the next mask with the same popcount */
    uint64_t c = s & (~s + 1), r = s + c;
    return (((r ^ s) >> 2) / c) | r;
}
static inline uint64_t tr_repair_binom(int32_t n, int32_t k) {
    if (k < 0 || k > n) return 0;
    if (k > n - k) k = n - k;
    unsigned __int128 r = 1;
    for (int32_t i = 1; i <= k; i++) { r = r * (unsigned __int128)(n - k + i) / (unsigned __int128)i; if (r > UINT64_MAX) return UINT64_MAX; }
    return (uint64_t)r;
}
static inline void tr_repair_add_solution(uint64_t S, int64_t cost, uint64_t *solutions, int64_t *sol_cost,
                                          uint64_t max_solutions, tr_repair_info *r, int64_t *best) {
    if (*best < 0 || cost < *best) { *best = cost; r->multiplicity = 0; r->listed = 0; }
    if (cost != *best) return;
    if (r->listed < max_solutions) { solutions[r->listed] = S; if (sol_cost) sol_cost[r->listed] = cost; r->listed++; }
    r->multiplicity++;
}
typedef struct { int64_t cost; uint64_t mask; } tr_repair_ranked;
static inline int tr_repair_ranked_cmp(const void *a, const void *b) {
    const tr_repair_ranked *x = (const tr_repair_ranked *)a, *y = (const tr_repair_ranked *)b;
    if (x->cost != y->cost) return x->cost < y->cost ? -1 : 1;
    return x->mask < y->mask ? -1 : x->mask > y->mask;
}

/* the rank generator of the exhaustive search: unit costs by levels (Gosper over the m allowed positions), weighted
 * by the sorted order of the 2^m subsets */
typedef struct {
    const tr_repair_plan *pl;
    const tr_repair_ranked *ord;
    uint64_t total, rank;                    /* weighted */
    int32_t level, reserved;                 /* unit */
    uint64_t lpos, ltotal, lmask;
} tr_repair_gen;
static inline void tr_repair_gen_level(tr_repair_gen *g) {
    g->lpos = 0; g->ltotal = tr_repair_binom(g->pl->m, g->level);
    g->lmask = g->level ? (((uint64_t)1 << g->level) - 1) : 0;
}
static inline int64_t tr_repair_gen_peek(const tr_repair_gen *g) {   /* the next rank's cost, INT64_MAX: none */
    if (g->pl->unit) return g->level <= g->pl->m ? (int64_t)g->level : INT64_MAX;
    return g->rank < g->total ? g->ord[g->rank].cost : INT64_MAX;
}
/* the ranks of cost `cost` not generated yet (the multiplicity bound of a stopped exhaustive search) */
static inline uint64_t tr_repair_gen_left(const tr_repair_gen *g, int64_t cost) {
    if (g->pl->unit) return g->level == cost && g->level <= g->pl->m ? g->ltotal - g->lpos : 0;
    uint64_t left = 0;
    for (uint64_t q = g->rank; q < g->total && g->ord[q].cost == cost; q++) left++;
    return left;
}
static inline uint64_t tr_repair_sat_add(uint64_t a, uint64_t b) {   /* saturating at TR_REPAIR_UNKNOWN */
    return a >= TR_REPAIR_UNKNOWN - b ? TR_REPAIR_UNKNOWN : a + b;
}
static inline uint64_t tr_repair_gen_fill(tr_repair_gen *g, uint64_t cap, uint64_t *masks, int64_t *costs) {
    uint64_t n = 0;
    if (g->pl->unit) {
        while (n < cap && g->level <= g->pl->m) {
            masks[n] = tr_repair_deposit(g->pl->allowed, g->lmask); costs[n] = g->level; n++;
            if (++g->lpos < g->ltotal) g->lmask = tr_repair_next_mask(g->lmask);
            else { g->level++; if (g->level <= g->pl->m) tr_repair_gen_level(g); break; }   /* one level per chunk */
        }
        return n;
    }
    while (n < cap && g->rank < g->total) { masks[n] = g->ord[g->rank].mask; costs[n] = g->ord[g->rank].cost; n++; g->rank++; }
    return n;
}
/* EXHAUSTIVE: every subset of the allowed elements in increasing (cost, mask) order through the scan evaluator; the
 * first feasible rank gives the minimum, the rest of its cost level every tie; past max_states: INCOMPLETE with the
 * cost of the first unexamined rank as the proven lower bound */
static inline int32_t tr_repair_exhaustive_run(const tr_repair_plan *pl, tr_repair_scan_fn scan, void *ctx,
                                               uint64_t batch, uint64_t *solutions, int64_t *sol_cost,
                                               tr_repair_info *info) {
    const int32_t m = pl->m;
    tr_repair_info r;
    memset(&r, 0, sizeof r);
    r.method = TR_REPAIR_EXHAUSTIVE; r.order = pl->order; r.n = pl->n; r.m = m; r.den = pl->den; r.upper = -1;
    r.total = m >= 64 ? UINT64_MAX : (uint64_t)1 << m;
    uint64_t *masks = (uint64_t *)calloc((size_t)batch, 8), *fe = (uint64_t *)calloc((size_t)(batch + 63) / 64, 8);
    uint64_t *ti = (uint64_t *)calloc((size_t)(batch + 63) / 64, 8);
    int64_t *costs = (int64_t *)calloc((size_t)batch, 8);
    tr_repair_ranked *ord = NULL;
    int32_t rc = masks && fe && ti && costs ? 0 : -2;
    if (!rc && !pl->unit) {
        ord = (tr_repair_ranked *)calloc((size_t)r.total, sizeof *ord);
        if (!ord) rc = -2;
        for (uint64_t i = 0; !rc && i < r.total; i++) { ord[i].mask = tr_repair_deposit(pl->allowed, i); ord[i].cost = tr_repair_cost_of(pl, ord[i].mask); }
        if (!rc) qsort(ord, (size_t)r.total, sizeof *ord, tr_repair_ranked_cmp);
    }
    tr_repair_gen g;
    memset(&g, 0, sizeof g);
    g.pl = pl; g.ord = ord; g.total = r.total;
    if (pl->unit) tr_repair_gen_level(&g);
    int64_t best = -1;
    uint64_t explored = 0;
    while (!rc) {
        const int64_t next = tr_repair_gen_peek(&g);
        if (next == INT64_MAX || (best >= 0 && next > best)) break;       /* exhausted, or every tie examined */
        const uint64_t room = pl->max_states - explored;
        if (!room) {                                                       /* every cheaper rank was examined */
            r.status = TR_REPAIR_INCOMPLETE; r.reason = TR_REPAIR_STOP_BUDGET;
            r.lower = best >= 0 ? best : next; r.upper = best;
            if (best >= 0) r.mult_hi = tr_repair_sat_add(r.multiplicity, tr_repair_gen_left(&g, best));
            else { r.multiplicity = 0; r.mult_hi = TR_REPAIR_UNKNOWN; }
            break;
        }
        const uint64_t cnt = tr_repair_gen_fill(&g, room < batch ? room : batch, masks, costs);
        int64_t mn = INT64_MAX;
        rc = scan(ctx, masks, costs, cnt, best >= 0 ? best : INT64_MIN, fe, ti, &mn);
        if (rc) break;
        explored += cnt;
        if (best < 0 && mn != INT64_MAX) best = mn;
        for (uint64_t q = 0; q < cnt && best >= 0; q++)
            if ((ti[q >> 6] >> (q & 63)) & 1) tr_repair_add_solution(masks[q], costs[q], solutions, sol_cost, pl->max_solutions, &r, &best);
    }
    if (!rc && !r.status) {
        r.complete = 1;
        if (best >= 0) { r.status = TR_REPAIR_FOUND; r.reason = TR_REPAIR_STOP_COMPLETE; r.lower = r.upper = best; r.mult_hi = r.multiplicity; }
        else { r.status = TR_REPAIR_NONE_FEASIBLE; r.reason = TR_REPAIR_STOP_EXHAUSTED; r.lower = r.upper = -1; }
    }
    r.explored = explored;
    free(masks); free(fe); free(ti); free(costs); free(ord);
    if (rc) return rc;
    *info = r;
    return 0;
}

/* BRANCH_BOUND: best-first over include / exclude decisions with admissible bounds (the batch evaluator decides the
 * nodes' subsets); the frontier is a binary heap on (lower bound, I, X) */
typedef struct {
    uint64_t I, X;
    int64_t lb;
    int32_t depth;
    int8_t fi, fc, pad[2];                   /* F(I) / F(I + U): -1 unknown, 0, 1 */
} tr_repair_node;
static inline int tr_repair_node_less(const tr_repair_node *a, const tr_repair_node *b) {
    if (a->lb != b->lb) return a->lb < b->lb;
    if (a->I != b->I) return a->I < b->I;
    return a->X < b->X;
}
typedef struct { tr_repair_node *v; uint64_t n, cap; } tr_repair_heap;
static inline int tr_repair_heap_push(tr_repair_heap *h, tr_repair_node x) {
    if (h->n == h->cap) {
        uint64_t nc = h->cap ? 2 * h->cap : 1024;
        tr_repair_node *nv = (tr_repair_node *)realloc(h->v, (size_t)nc * sizeof *nv);
        if (!nv) return -2;
        h->v = nv; h->cap = nc;
    }
    uint64_t i = h->n++;
    while (i) {
        uint64_t p = (i - 1) / 2;
        if (!tr_repair_node_less(&x, &h->v[p])) break;
        h->v[i] = h->v[p]; i = p;
    }
    h->v[i] = x;
    return 0;
}
static inline tr_repair_node tr_repair_heap_pop(tr_repair_heap *h) {
    tr_repair_node top = h->v[0], x = h->v[--h->n];
    uint64_t i = 0;
    for (;;) {
        uint64_t l = 2 * i + 1, r = l + 1, m = i;
        const tr_repair_node *best = &x;
        if (l < h->n && tr_repair_node_less(&h->v[l], best)) { m = l; best = &h->v[l]; }
        if (r < h->n && tr_repair_node_less(&h->v[r], best)) m = r;
        if (m == i) break;
        h->v[i] = h->v[m]; i = m;
    }
    if (h->n) h->v[i] = x;
    return top;
}
/* the admissible estimate of the cost still needed below a node (I, U undecided allowed): 0 or more, or -1 when no
 * completion can be feasible (sound pruning); fi: F(I) when known (-1 unknown) */
static inline int64_t tr_repair_bound(const tr_repair_view *v, const tr_repair_plan *pl, uint64_t I, uint64_t U, int fi) {
    if (fi == 1) return 0;
    if (v->kind == TR_REPAIR_HIT) {
        uint64_t used = 0;
        int64_t h = 0;
        for (uint64_t q = 0; q < v->nmasks; q++) {
            const uint64_t mk = v->masks[q];
            if (mk & I) continue;
            const uint64_t r = mk & U;
            if (!r) return -1;                                  /* a mask no completion meets */
            if (r & used) continue;
            used |= r;
            int64_t c = INT64_MAX;
            for (uint64_t b = r; b; b &= b - 1) { int64_t w = pl->cost[__builtin_ctzll(b)]; if (w < c) c = w; }
            h += c;
        }
        return h;
    }
    if (v->kind == TR_REPAIR_DELETE && v->op != TR_REPAIR_NE) {
        /* the remaining aggregate as base_A + sum over kept rows of a_r op' T: delete D from U with sum_D a_r >= need
         * (<= ops) or sum_D a_r <= -need' (>= ops): the fractional knapsack bound, rounded up */
        const __int128 tn = tr_i128_get(v->tnum), td = v->tden;
        __int128 tot = 0;                                       /* sum of a over the rows not deleted (|.| < 2^125:
                                                                   proven when the problem was made) */
        __int128 a[TR_REPAIR_MAX_ELEMENTS + 2];
        for (int32_t i = 0; i < v->n; i++) {
            a[i] = 0;
            if (!v->qual[i]) continue;
            if (v->agg == TR_REPAIR_COUNT_STAR) a[i] = td;
            else if (v->agg == TR_REPAIR_COUNT) a[i] = v->vvalid[i] ? td : 0;
            else if (v->agg == TR_REPAIR_SUM) a[i] = v->vvalid[i] ? (__int128)v->vval[i] * td : 0;
            else a[i] = v->vvalid[i] ? (__int128)v->vval[i] * td - tn : 0;      /* AVG: sum (v td - tn) op 0 */
        }
        __int128 base;
        if (v->agg == TR_REPAIR_COUNT_STAR) base = (__int128)v->base_count * td - tn;
        else if (v->agg == TR_REPAIR_COUNT) base = (__int128)v->base_vcount * td - tn;
        else if (v->agg == TR_REPAIR_SUM) base = tr_i128_get(v->base_sum) * td - tn;
        else base = tr_i128_get(v->base_sum) * td - tn * (__int128)v->base_vcount;
        for (int32_t i = 0; i < v->n; i++) if (!((I >> i) & 1)) tot += a[i];
        /* value(S) = base + tot - sum_{D} a (D = deleted beyond I); the target: value op 0 */
        const __int128 cur = base + tot;
        int sgn;                                                /* +1: decrease value to <= 0; -1: increase to >= 0 */
        if (v->op == TR_REPAIR_LE || v->op == TR_REPAIR_LT) sgn = 1;
        else if (v->op == TR_REPAIR_GE || v->op == TR_REPAIR_GT) sgn = -1;
        else sgn = cur > 0 ? 1 : -1;                            /* EQ: the side the value must move to */
        __int128 need = sgn > 0 ? cur : -cur;                   /* sum_D (sgn a_r) >= need (relaxing strictness) */
        if (need <= 0) return fi == 0 ? -2 : 0;                 /* -2: the generic estimate applies */
        /* greedy by the ratio cost / (sgn a) over the undecided rows with sgn a > 0, in exact cross products */
        uint64_t left = U;
        __int128 got = 0;
        int64_t full = 0;
        while (left) {
            int32_t bi = -1;
            for (uint64_t b = left; b; b &= b - 1) {
                int32_t i = __builtin_ctzll(b);
                __int128 g = sgn > 0 ? a[i] : -a[i];
                if (g <= 0) continue;
                if (bi < 0) { bi = i; continue; }
                __int128 gb = sgn > 0 ? a[bi] : -a[bi];
                /* cost_i / g_i < cost_bi / g_bi */
                if (tr_repair_cmp_prod((__int128)pl->cost[i], gb, (__int128)pl->cost[bi], g) < 0) bi = i;
            }
            if (bi < 0) break;
            left &= ~((uint64_t)1 << bi);
            __int128 g = sgn > 0 ? a[bi] : -a[bi];
            if (got + g >= need) {                              /* the fractional last item: ceil(cost (need - got) / g) */
                const __int128 rem = need - got, w = (__int128)pl->cost[bi];
                int64_t lo = 0, hi = pl->cost[bi];                  /* rem <= g: the smallest q with q g >= w rem */
                while (lo < hi) {
                    int64_t mid = lo + (hi - lo) / 2;
                    if (tr_repair_cmp_prod((__int128)mid, g, w, rem) >= 0) hi = mid; else lo = mid + 1;
                }
                return full + lo;
            }
            got += g; full += pl->cost[bi];
        }
        return -1;                                              /* even deleting every useful row cannot reach it */
    }
    return -2;                                                  /* the generic estimate */
}
static inline int64_t tr_repair_generic(const tr_repair_plan *pl, uint64_t U, int fi) {
    if (fi == 1) return 0;
    if (fi == -1) return 0;
    if (!U) return -1;
    int64_t c = INT64_MAX;
    for (uint64_t b = U; b; b &= b - 1) { int64_t w = pl->cost[__builtin_ctzll(b)]; if (w < c) c = w; }
    return c;
}
static inline uint64_t tr_repair_undecided(const tr_repair_plan *pl, int32_t depth) {
    uint64_t U = 0;
    for (int32_t j = depth; j < pl->m; j++) U |= (uint64_t)1 << pl->perm[j];
    return U;
}
static inline int32_t tr_repair_bnb_run(const tr_repair_view *v, const tr_repair_plan *pl, tr_repair_batch_fn eval,
                                        void *ctx, uint64_t batch, uint64_t *solutions, int64_t *sol_cost,
                                        tr_repair_info *info) {
    tr_repair_info r;
    memset(&r, 0, sizeof r);
    r.method = TR_REPAIR_BRANCH_BOUND; r.order = pl->order; r.n = pl->n; r.m = pl->m; r.den = pl->den; r.upper = -1;
    r.total = pl->m >= 64 ? UINT64_MAX : (uint64_t)1 << pl->m;
    const int monotone = v->monotone > 0;
    tr_repair_heap h = {NULL, 0, 0};
    tr_repair_node *pop = (tr_repair_node *)calloc((size_t)batch, sizeof *pop);
    uint64_t *masks = (uint64_t *)calloc((size_t)(2 * batch), 8);
    int32_t *slot = (int32_t *)calloc((size_t)(2 * batch), 4);
    tr_repair_out *out = (tr_repair_out *)calloc((size_t)(2 * batch), sizeof *out);
    int32_t rc = pop && masks && slot && out ? 0 : -2;
    tr_repair_node root;
    memset(&root, 0, sizeof root);
    root.fi = -1; root.fc = -1; root.lb = 0;
    {
        int64_t e = tr_repair_bound(v, pl, 0, tr_repair_undecided(pl, 0), -1);
        if (e == -1) root.lb = -1; else if (e > 0) root.lb = e;
    }
    if (!rc && root.lb >= 0) rc = tr_repair_heap_push(&h, root);
    else if (!rc) r.pruned++;
    int64_t best = -1;
    uint64_t explored = 0;
    while (!rc && h.n) {
        if (best >= 0 && h.v[0].lb > best) break;                          /* every open node is costlier: done */
        /* a batch of the best nodes, and the subsets they still need decided */
        uint64_t np = 0, nm = 0;
        while (np < batch && h.n && !(best >= 0 && h.v[0].lb > best)) {
            tr_repair_node x = h.v[0];
            uint64_t need = (x.fi < 0) + (monotone && x.fc < 0 && x.fi != 1);
            if (explored + nm + need > pl->max_states) break;
            pop[np] = tr_repair_heap_pop(&h);
            if (pop[np].fi < 0) { masks[nm] = pop[np].I; slot[nm] = (int32_t)(2 * np); nm++; }
            if (monotone && pop[np].fc < 0) {
                masks[nm] = pop[np].I | tr_repair_undecided(pl, pop[np].depth); slot[nm] = (int32_t)(2 * np + 1); nm++;
            }
            np++;
        }
        if (!np) {                                                         /* the state budget is exhausted */
            r.status = TR_REPAIR_INCOMPLETE; r.reason = TR_REPAIR_STOP_BUDGET;
            break;
        }
        if (nm) {
            rc = eval(ctx, masks, nm, out);
            if (rc) break;
            explored += nm;
            for (uint64_t q = 0; q < nm; q++) {
                tr_repair_node *x = &pop[slot[q] / 2];
                if (slot[q] % 2 == 0) x->fi = (int8_t)(out[q].feasible != 0); else x->fc = (int8_t)(out[q].feasible != 0);
            }
        }
        for (uint64_t q = 0; q < np && !rc; q++) {
            tr_repair_node *x = &pop[q];
            const int64_t cI = tr_repair_cost_of(pl, x->I);
            if (x->fi == 1) {                                              /* a solution: not expanded */
                if (best < 0 || cI <= best) tr_repair_add_solution(x->I, cI, solutions, sol_cost, pl->max_solutions, &r, &best);
                continue;
            }
            if (monotone && x->fc == 0) { r.pruned++; continue; }          /* no completion is feasible */
            if (x->depth >= pl->m) continue;                               /* every element decided, infeasible */
            const uint32_t e = pl->perm[x->depth];
            const uint64_t bit = (uint64_t)1 << e;
            tr_repair_node c1 = *x, c2 = *x;
            c1.I |= bit; c1.depth++; c1.fi = -1; c1.fc = monotone ? 1 : -1;  /* the same completion: feasible */
            c2.X |= bit; c2.depth++; c2.fi = 0; c2.fc = -1;                  /* the same I: infeasible */
            for (int ch = 0; ch < 2 && !rc; ch++) {
                tr_repair_node *y = ch ? &c2 : &c1;
                const uint64_t U = tr_repair_undecided(pl, y->depth);
                const int64_t base = tr_repair_cost_of(pl, y->I);
                int64_t e2 = tr_repair_bound(v, pl, y->I, U, y->fi);
                if (e2 == -2) e2 = tr_repair_generic(pl, U, y->fi);
                if (e2 == -1 && y->fi != 1) { r.pruned++; continue; }      /* no completion (sound) */
                if (e2 < 0) e2 = 0;
                y->lb = base + e2;
                if (best >= 0 && y->lb > best) { r.pruned++; continue; }
                rc = tr_repair_heap_push(&h, *y);
            }
        }
    }
    if (!rc) {
        r.explored = explored; r.open = h.n;
        if (r.status == TR_REPAIR_INCOMPLETE) {
            int64_t lo = h.n ? h.v[0].lb : (best >= 0 ? best : 0);
            if (best >= 0 && best < lo) lo = best;
            r.lower = lo; r.upper = best;
            r.complete = 0;
            if (best >= 0 && lo == best) {        /* the minimum is best: the ties found, at most every open subset more */
                uint64_t hi = r.multiplicity;
                for (uint64_t q = 0; q < h.n; q++) {
                    if (h.v[q].lb > best) continue;
                    const int32_t u = pl->m - h.v[q].depth;
                    hi = tr_repair_sat_add(hi, u >= 63 ? TR_REPAIR_UNKNOWN : (uint64_t)1 << u);
                }
                r.mult_hi = hi;
            } else { r.multiplicity = 0; r.mult_hi = TR_REPAIR_UNKNOWN; }   /* the minimum is not known yet */
            if (best >= 0 && h.n && h.v[0].lb > best) {                    /* every open node is costlier after all */
                r.status = TR_REPAIR_FOUND; r.reason = TR_REPAIR_STOP_COMPLETE; r.lower = r.upper = best; r.complete = 1;
                r.mult_hi = r.multiplicity;
            }
        } else if (best >= 0) {
            r.status = TR_REPAIR_FOUND; r.reason = TR_REPAIR_STOP_COMPLETE; r.lower = r.upper = best; r.complete = 1;
            r.mult_hi = r.multiplicity;
        } else {
            r.status = TR_REPAIR_NONE_FEASIBLE; r.reason = TR_REPAIR_STOP_EXHAUSTED; r.lower = r.upper = -1; r.complete = 1;
        }
    }
    free(h.v); free(pop); free(masks); free(slot); free(out);
    if (rc) return rc;
    *info = r;
    return 0;
}
/* 0 when a prepared search fits its view (the CPU and the device check it alike) */
static inline int32_t tr_repair_plan_check(const tr_repair_view *v, const tr_repair_plan *pl) {
    if (!pl || !pl->cost || !pl->perm || pl->n != v->n || pl->allowed != v->allowed ||
        pl->m != __builtin_popcountll(v->allowed) || pl->den < 1 || pl->max_states < 1 ||
        (pl->method != TR_REPAIR_EXHAUSTIVE && pl->method != TR_REPAIR_BRANCH_BOUND) ||
        pl->order < TR_REPAIR_ORDER_INDEX || pl->order > TR_REPAIR_ORDER_BANZHAF)
        return -1;
    uint64_t seen = 0;
    int64_t total = 0;
    for (int32_t j = 0; j < pl->m; j++) {
        const uint32_t e = pl->perm[j];
        if (e >= (uint32_t)v->n || !((v->allowed >> e) & 1) || ((seen >> e) & 1)) return -1;
        seen |= (uint64_t)1 << e;
        if (pl->cost[e] < 1 || pl->cost[e] > (INT64_C(1) << 62) - total) return -1;
        total += pl->cost[e];
        if (pl->unit && pl->cost[e] != 1) return -1;
    }
    if (pl->method == TR_REPAIR_EXHAUSTIVE && !pl->unit && pl->m > TR_REPAIR_MAX_SORTED) return -12;
    return 0;
}
/* the search: a hitting set with a mask no allowed element meets is infeasible at once (proven); else the plan's
 * method with the given evaluators */
static inline int32_t tr_repair_run(const tr_repair_view *v, const tr_repair_plan *pl, tr_repair_batch_fn eval,
                                    tr_repair_scan_fn scan, void *ctx, uint64_t batch, uint64_t *solutions,
                                    int64_t *sol_cost, tr_repair_info *info) {
    if (v->kind == TR_REPAIR_HIT) {
        for (uint64_t q = 0; q < v->nmasks; q++)
            if (!(v->masks[q] & v->allowed)) {
                memset(info, 0, sizeof *info);
                info->status = TR_REPAIR_NONE_FEASIBLE; info->reason = TR_REPAIR_STOP_EXHAUSTED; info->method = pl->method;
                info->order = pl->order; info->n = pl->n; info->m = pl->m; info->complete = 1;
                info->lower = info->upper = -1; info->den = pl->den;
                info->total = pl->m >= 64 ? UINT64_MAX : (uint64_t)1 << pl->m;
                return 0;
            }
    }
    if (pl->method == TR_REPAIR_EXHAUSTIVE) return tr_repair_exhaustive_run(pl, scan, ctx, batch, solutions, sol_cost, info);
    return tr_repair_bnb_run(v, pl, eval, ctx, batch, solutions, sol_cost, info);
}
/* the needs of a search (both libraries add their device copies) */
static inline void tr_repair_search_need(const tr_repair_plan *pl, tr_math_need *need) {
    for (int i = 0; i < TR_BUDGET_RESOURCES; i++) need->need[i] = 0;
    need->need[TR_BUDGET_VARIABLES] = (uint64_t)pl->n;
    need->need[TR_BUDGET_ITERATIONS] = pl->max_states;
    need->need[TR_BUDGET_OUTPUT] = pl->max_solutions;
    uint64_t b = tr_budget_add(tr_budget_mul(TR_REPAIR_BATCH, 8 + 16 + 32 + 4 + 16), tr_budget_mul(pl->max_solutions, 16));
    if (pl->method == TR_REPAIR_EXHAUSTIVE && !pl->unit)
        b = tr_budget_add(b, tr_budget_mul(pl->m >= 64 ? TR_BUDGET_UNKNOWN : (uint64_t)1 << pl->m, sizeof(tr_repair_ranked)));
    if (pl->method == TR_REPAIR_BRANCH_BOUND)
        b = tr_budget_add(b, tr_budget_mul(tr_budget_mul(pl->max_states, 2), sizeof(tr_repair_node)));
    need->need[TR_BUDGET_BYTES] = b;
}

/* bytes the device evaluation of `batch` candidates copies to and allocates on the device (saturating) */
static inline uint64_t tr_repair_view_device_bytes(const tr_repair_view *v, uint64_t batch) {
    uint64_t b = 0;
    if (v->kind == TR_REPAIR_HIT) b = tr_budget_mul(v->nmasks, 8);
    else if (v->kind == TR_REPAIR_CIRCUIT) {
        const uint64_t nw = v->dom.nw;
        if (v->dom.kind == TR_BOOL_ROWS)
            b = tr_budget_mul(nw, tr_budget_add(1, tr_budget_mul((uint64_t)v->dom.k, v->dom.valid ? 2 : 1)));
        b = tr_budget_mul(tr_budget_add(b, nw), 8);
        b = tr_budget_add(b, tr_budget_mul((uint64_t)(v->prog.nnodes > 0 ? v->prog.nnodes : 0), 16 + 8));
        b = tr_budget_add(b, tr_budget_mul((uint64_t)(v->prog.nsched > 0 ? v->prog.nsched : 0), 4));
    } else if (v->kind == TR_REPAIR_GAME) b = tr_attr_game_device_bytes(&v->game);
    else b = tr_budget_mul((uint64_t)v->n, 10);
    const uint64_t row = tr_budget_mul(tr_repair_scratch_words(v), 8);
    uint64_t chunk = row >= TR_REPAIR_DEVICE_SCRATCH ? 1 : TR_REPAIR_DEVICE_SCRATCH / row;
    if (chunk > batch) chunk = batch ? batch : 1;
    b = tr_budget_add(b, tr_budget_mul(chunk, row));
    b = tr_budget_add(b, tr_budget_mul(batch, 8 + 8 + sizeof(tr_repair_out)));
    return tr_budget_add(b, tr_budget_add(tr_budget_mul(tr_budget_bitmap_bytes(batch), 2), 16));
}

/* ================================================================ libalg */
/* the changed threshold of a comparison condition after the feature shift d of an INV-002 threshold result: c - d */
tr_bigq *tr_repair_threshold_after(const tr_bool_condition *condition, const tr_bigq *shift, const tr_math_budget *,
                                   tr_math_refusal *, int32_t *error);
/* problems: masks (HIT), separation (SEPARATE: A, B words over the domain's points; EXPLAIN: program, A = its TRUE
 * points), conjunction keep sets (KEEP: the conjunction of the conditions in `conjunction`), programs (CIRCUIT), games,
 * deletions; `allowed` 0 = every element (KEEP: the conjunction's conditions; CIRCUIT: the program's conditions) */
tr_repair_problem *tr_repair_masks(int32_t n, uint64_t nmasks, const uint64_t *masks, uint64_t allowed,
                                   const tr_math_budget *, tr_math_refusal *, int32_t *error);
tr_repair_problem *tr_repair_separate(const tr_bool_domain *, const uint64_t *a_words, const uint64_t *b_words,
                                      int32_t convention, uint64_t allowed, const tr_math_budget *, tr_math_refusal *,
                                      int32_t *error);
tr_repair_problem *tr_repair_explain(const tr_bool_domain *, const tr_bool_program_view *, int32_t convention,
                                     uint64_t allowed, const tr_math_budget *, tr_math_refusal *, int32_t *error);
tr_repair_problem *tr_repair_keep(const tr_bool_domain *, uint64_t conjunction, const tr_math_budget *,
                                  tr_math_refusal *, int32_t *error);
tr_repair_problem *tr_repair_program(const tr_bool_domain *, const tr_bool_program_view *, uint64_t baseline,
                                     uint64_t allowed, const tr_math_budget *, tr_math_refusal *, int32_t *error);
tr_repair_problem *tr_repair_game(const tr_attr_game *, int32_t kind, int32_t op, tr_math_i128 threshold,
                                  uint64_t allowed, const tr_math_budget *, tr_math_refusal *, int32_t *error);
/* deletions: the domain's selected rows of `scope` (the domain built over it), qualifying under `program` (NULL: every
 * selected row), value column `value` of the scope (-1: COUNT(*) only), the n candidate rows `rows` */
tr_repair_problem *tr_repair_delete(const tr_scope *, const tr_bool_domain *, const tr_bool_program_view *,
                                    int32_t value, int32_t agg, int32_t op, const tr_bigq *t, int32_t n,
                                    const uint64_t *rows, const tr_math_budget *, tr_math_refusal *, int32_t *error);
/* the words of the selected rows of group `group` of a scope (SEPARATE between groups) */
int32_t tr_repair_group_words(const tr_scope *, uint64_t group, uint64_t *words, uint64_t nwords);
int32_t tr_repair_problem_get_info(const tr_repair_problem *, tr_repair_problem_info *);
int32_t tr_repair_problem_get_view(const tr_repair_problem *, tr_repair_view *);
int32_t tr_repair_problem_masks(const tr_repair_problem *, uint64_t *out, uint64_t capacity);
void tr_repair_problem_free(tr_repair_problem *);
/* the CPU evaluators (budget: output = count) */
int32_t tr_repair_evaluate(const tr_repair_view *, const uint64_t *masks, uint64_t count, tr_repair_out *out,
                           const tr_math_budget *, tr_math_refusal *);
int32_t tr_repair_scan(const tr_repair_view *, const uint64_t *masks, const int64_t *costs, uint64_t count,
                       int64_t match, uint64_t *feasible, uint64_t *ties, int64_t *mincost, const tr_math_budget *,
                       tr_math_refusal *);
/* a prepared search: costs scaled, the method (AUTO: exhaustive when 2^m <= max_states, else branch and bound) and
 * the order (SHAPLEY / BANZHAF: of `order_game`, or of the problem's own game) */
int32_t tr_repair_plan_need(const tr_repair_problem *, const tr_repair_query *, tr_math_need *);
int32_t tr_repair_prepare(const tr_repair_problem *, const tr_repair_query *, const tr_attr_game *order_game,
                          int64_t *cost_out, uint32_t *perm_out, tr_repair_plan *plan, const tr_math_budget *,
                          tr_math_refusal *);
/* the search on the CPU (solutions: max_solutions masks and their cost numerators) */
int32_t tr_repair_search(const tr_repair_problem *, const tr_repair_plan *, uint64_t *solutions, int64_t *sol_cost,
                         tr_repair_info *, const tr_math_budget *, tr_math_refusal *);
/* redundancy of every allowed element: out[m] in increasing element order; witnesses (CIRCUIT): per element up to
 * max_witnesses rows in rows[i * max_witnesses ..], gained ones first */
int32_t tr_repair_redundancy(const tr_repair_problem *, tr_repair_removal *out, uint64_t capacity, uint64_t *rows,
                             uint64_t max_witnesses, const tr_math_budget *, tr_math_refusal *);
/* EDIT: value edits of the rows `rows` (n <= 16) of column `value` for SUM / AVG over the qualifying rows op t; cost,
 * weights (NULL: 1) and bounds on the edits dlo / dhi (NULL entries: none): an INV-002 result */
tr_interv_result *tr_repair_edit(const tr_scope *, const tr_bool_domain *, const tr_bool_program_view *, int32_t value,
                                 int32_t agg, int32_t op, const tr_bigq *t, int32_t n, const uint64_t *rows,
                                 int32_t cost, const tr_bigq *const *weights, const tr_bigq *const *dlo,
                                 const tr_bigq *const *dhi, const tr_math_budget *, tr_math_refusal *, int32_t *error);
/* THRESHOLD: the conjunction of the k conditions (BOOL-001 comparisons with constants, over the scope) and the
 * declared changeable ones (`which`, a mask); AGG(v) op t over the rows the conjunction then selects: one INV-002
 * result per declared condition in results[] (increasing condition order, owned by the caller), the minimum cost over
 * them (*best_lo / *best_hi borrowed from a result: lo = hi unless that result is an infimum) and `attaining` (a mask
 * of the conditions whose cost equals the minimum) */
int32_t tr_repair_threshold(const tr_scope *, const tr_bool_condition *conditions, int32_t k, uint64_t which,
                            int32_t value, int32_t agg, int32_t op, const tr_bigq *t, const tr_bigq *const *weights,
                            tr_interv_result **results, uint64_t *attaining, const tr_math_budget *, tr_math_refusal *);

/* ================================================================ the CUDA library (math_repair_gpu.cu) */
int tr_mgpu_repair_eval(const tr_repair_view *, const uint64_t *masks, uint64_t count, tr_repair_out *out,
                        const tr_math_budget *, tr_math_refusal *);
int tr_mgpu_repair_scan(const tr_repair_view *, const uint64_t *masks, const int64_t *costs, uint64_t count,
                        int64_t match, uint64_t *feasible, uint64_t *ties, int64_t *mincost, const tr_math_budget *,
                        tr_math_refusal *);
int tr_mgpu_repair_search(const tr_repair_view *, const tr_repair_plan *, uint64_t *solutions, int64_t *sol_cost,
                          tr_repair_info *, const tr_math_budget *, tr_math_refusal *);
/* the needs of the device evaluation of `count` candidates (output, bytes) and of a device search */
int32_t tr_repair_device_need(const tr_repair_view *, uint64_t count, const tr_repair_plan *, tr_math_need *);
#ifdef __cplusplus
}
#endif
#endif
