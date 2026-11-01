/* MULTI-002: adaptive refinement and change localization over MULTI-001 hierarchies.
 * SPDX-License-Identifier: Apache-2.0
 *
 * A refinement state (tr_refine_state, immutable) holds, for every level of a MULTI-001 hierarchy (math_multi.h), the
 * exact FND-004 cells of each group (rows, per term count and 128-bit sum) and the UNC-001 abstract input of each
 * group and column (rows, present values, min, max, gcd of the differences: math_unc.h; in logical units), rolled up
 * exactly through the hierarchy's maps (min of mins, max of maxes, gcd(m1, m2, min1 - min2)), plus the parent of every
 * group and the children of every coarse group (ids increasing) and the DELTA-001 maintained state of the leaves.
 *
 * Requests (tr_refine_request) about the groups of a target level L, answered from the coarsest level down:
 *   THRESHOLD  A(t) op T          A = sum_i c_i atom_i, atoms SUM(term j), COUNT(term j), ROWS, int64 coefficients;
 *                                 op >, >=, <, <=; T an int64 rational (as integers: A > T <=> A >= floor(T) + 1, ...)
 *   RATIO      A(t) / D(t) op T   decided as C = q A - p D (T = p / q) op 0 where D > 0, the flipped op where D < 0;
 *                                 D = 0 is UNDEFINED (never in the answer, like a SQL NULL comparison)
 *   TOPK       the k target groups with the largest (descending) or smallest A, ties by the smaller group id
 * Region bounds (tr_refine_region_bound, the inline function the CPU loop and the CUDA kernel share): for a region R at
 * a level above L, every present descendant t at level L satisfies A(t) in [lo, hi]. The atoms are split into
 * validity classes (SQL: the terms' column sets; listwise: one class; ROWS: every row); a class V with n = its valid
 * rows in R, its exact total E = sum_{atoms in V} c_i atom_i(R), a per-row interval [elo, ehi] of its row expression
 * e_V = sum c_i term_i (+ the COUNT / ROWS coefficients) from UNC-001's exact abstract evaluation of e_V on R's inputs
 * (affine forms and congruences: correlated terms cancel), and t holding k of R's V-rows (k in [kmin, n], kmin = 1
 * when every row of R is valid for V): A_V(t) <= max_k min(k ehi, E - (n - k) elo) and A_V(t) >= min_k max(k elo,
 * E - (n - k) ehi), evaluated exactly at the endpoints and at the crossing of the two lines (a concave / convex
 * piecewise-linear function of k); A's bound is the sum of its classes' bounds. When every atom's term is valid on
 * every row of R the atoms form one class (exact correlation of the whole aggregate). Values are exact 256-bit
 * integers (|A| < 2^197); an unbounded side (UNC-001 widened) is +-inf. At L itself the bound is the exact value.
 * Waves (tr_refine_query): the regions of one level at a time, coarse to fine, in increasing group id. A region whose
 * bound decides every descendant is final: ALL (every descendant satisfies), NONE (none does); TOPK prunes a region
 * whose upper bound is below tau, the k-th largest of the wave's witnesses (each region has a present descendant at
 * least that large: its lower bound, or the mean of its present target descendants, A being linear: the region's own
 * exact aggregate over their count). The other regions are expanded into their present children, at most
 * expand_limit expansions in
 * all (0: no limit); a region left when the limit is reached is UNRESOLVED with its sound bound. At L the groups are
 * exact: TRUE / FALSE / UNDEFINED, or TOPK's IN / OUT (OUT when k better groups are known, IN when its rank is
 * certainly at most k, UNRESOLVED otherwise). Every entry of the answer carries its bound; the answer membership of the
 * target groups is the union of the ALL regions' descendants and the exact TRUE / IN groups, the unresolved ones
 * marked apart. tr_refine_fine answers the same request from the complete states of level L (the baseline).
 *
 * Change localization (tr_refine_update): a DELTA-001 delta of the leaves (math_delta.h) is applied to the leaves'
 * maintained state (DELTA-001 checks its compatibility, the overflow proof and the sign promise); each touched leaf's
 * exact change (rows, counts, 192-bit sums, present values, the inserted values' min, max and gcd, the deleted
 * values) is added along its path only: one work item per affected region sums its affected children's changes
 * (tr_refine_rollup_region; level by level). Extrema stay sound enclosures after deletions (flagged: not tight), the
 * gcd stays a valid congruence. The maps are carried (status CARRIED, as MULTI-001's tr_multi_hier_update without
 * levels: a touched leaf keeps its parent, which the delta's rows are assumed to share). The new state reports every
 * changed region with its exact change. An update the local path cannot take (the delta's schema, the overflow
 * proof, a touched leaf that becomes present without a parent) falls back to a complete rebuild from the new levels
 * when they are given (strategy REBUILD, maps certified on them, the reason kept), else it is refused; a
 * contradicted nonnegative promise is refused (TR_MATH_DOMAIN_ERROR with the violation), never rebuilt around.
 * Multiplicities are nonnegative (DELTA-001's TR_DELTA_NONNEGATIVE): the region bounds count rows; a signed measure
 * is TR_MATH_UNSUPPORTED.
 *
 * Budgets (math_budgets.h, checked before allocation): build: states = groups x (terms + columns), bytes = every
 * level's cells, inputs, maps and child lists; query: support = atoms, circuit = the classes' program nodes, output =
 * the entries (at most the groups of all levels), iterations = expansions, bytes = the frontier and the entries;
 * update: states = the affected regions x (terms + columns), bytes = the new state and the changes. CUDA: the device
 * copies (tr_refine_bounds_device_bytes, tr_refine_rollup_device_bytes).
 *
 * Errors: TR_MATH_INVALID, _MEMORY, _OVERFLOW, _BUDGET, _UNSUPPORTED (more atoms or nodes than represented, a ratio
 * whose combined coefficients leave int64), TR_MATH_DOMAIN_ERROR (bindings of another hierarchy or scope; an update
 * the local path cannot take and no rebuild was given; a contradicted promise). */
#ifndef TORBLES_MATH_REFINE_H
#define TORBLES_MATH_REFINE_H
#include <stdint.h>
#include <string.h>
#include "math_num.h"
#include "math_budgets.h"
#include "math_scope.h"
#include "math_bank.h"
#include "math_unc.h"
#include "math_multi.h"
#include "math_delta.h"
#ifdef __cplusplus
extern "C" {
#endif

#define TR_REFINE_ABI 1
#define TR_REFINE_MAX_ATOMS 4            /* atoms of A and of D */
#define TR_REFINE_MAX_PLAN_ATOMS (6 * TR_REFINE_MAX_ATOMS)   /* a plan's atoms: C = q A - p D in its classes and its
                                                            merged class, D in its classes and its merged class */
#define TR_REFINE_MAX_CLASSES 18         /* validity classes of a plan (two aggregates, merged classes included) */
#define TR_REFINE_MAX_NODES 512          /* program nodes of a plan (each class program <= TR_UNC_MAX_NODES) */
#define TR_REFINE_GPU_CHUNK 1024         /* regions per launch of tr_mgpu_refine_bounds (their UNC-001 scratch) */
#define TR_REFINE_NO_ID UINT64_MAX

enum { TR_REFINE_THRESHOLD = 1, TR_REFINE_RATIO = 2, TR_REFINE_TOPK = 3 };
enum { TR_REFINE_GT = 1, TR_REFINE_GE = 2, TR_REFINE_LT = 3, TR_REFINE_LE = 4 };
enum { TR_REFINE_SUM = 1, TR_REFINE_COUNT = 2, TR_REFINE_ROWS = 3 };                           /* atoms */
enum { TR_REFINE_ALL = 1, TR_REFINE_NONE = 2, TR_REFINE_UNRESOLVED = 3, TR_REFINE_TRUE = 4, TR_REFINE_FALSE = 5,
       TR_REFINE_UNDEFINED = 6, TR_REFINE_IN = 7, TR_REFINE_OUT = 8 };                         /* entry decisions */
enum { TR_REFINE_LO_INF = 1, TR_REFINE_HI_INF = 2, TR_REFINE_DLO_INF = 4, TR_REFINE_DHI_INF = 8,
       TR_REFINE_EXACT = 16, TR_REFINE_MERGED = 32 };                                          /* bound flags */
enum { TR_REFINE_LOCAL = 2, TR_REFINE_COPY = 3, TR_REFINE_REBUILD = 4 };                       /* update strategies */
enum { TR_REFINE_REASON_NONE = 0, TR_REFINE_REASON_SCHEMA = 1, TR_REFINE_REASON_OVERFLOW = 2,
       TR_REFINE_REASON_NEW_GROUP = 3 };                                                       /* why a rebuild */
enum { TR_REFINE_NEED_CPU = 0, TR_REFINE_NEED_DEVICE = 1 };

/* ------------------------------------------------------------------ structs crossing the ctypes boundary */
typedef struct { int32_t kind, term; int64_t coef; } tr_refine_atom;                           /* 16 bytes */
typedef struct {
    int32_t kind, level, op, natoms;             /* TR_REFINE_THRESHOLD / RATIO / TOPK; the target level; op; atoms of A */
    int32_t ndatoms, descending, reserved;       /* RATIO: atoms of D; TOPK: 1 largest first, 0 smallest first */
    uint32_t k;                                  /* TOPK */
    tr_num_rational threshold;                   /* THRESHOLD, RATIO */
    uint64_t expand_limit;                       /* region expansions (0: no limit) */
    tr_refine_atom atoms[TR_REFINE_MAX_ATOMS], datoms[TR_REFINE_MAX_ATOMS];
} tr_refine_request;                             /* 184 bytes */

/* a class of a compiled plan: its atoms (plan atoms[first .. first + n)), the term whose count is its rows (-1: rows),
 * whether it is the merged class, its UNC-001 program (plan nodes[first_node .. + nnodes), root) */
typedef struct { int32_t first_atom, natoms, count_term, merged, first_node, nnodes, root, nlive; } tr_refine_class;   /* 32 */
/* the compiled request: aggregate 0 (A, or C = q A - p D) and aggregate 1 (D): classes[first[a] .. + nclasses[a]),
 * the merged class (or -1); every pointer is host memory (the CUDA entry copies them) */
typedef struct {
    int32_t naggs, ncols, nterms, mode;
    int32_t first[2], nclasses[2], merged[2];
    int32_t natoms, nclass, nnodes, nlive;       /* totals; nlive: the largest program scratch */
    const tr_refine_atom *atoms;
    const tr_refine_class *classes;
    const tr_unc_node *nodes;
} tr_refine_plan;                                /* 80 bytes */

/* one level of a state as the kernels read it */
typedef struct {
    uint64_t ngroups;
    int32_t nterms, ncols;
    const uint64_t *rows;                        /* [ngroups] */
    const uint64_t *count;                       /* [ngroups * nterms] */
    const tr_math_i128 *sum;                     /* [ngroups * nterms] */
    const tr_unc_input *inputs;                  /* [ngroups * ncols], logical units (scale 0) */
} tr_refine_level_view;                          /* 48 bytes */

/* the bound of one region (or the exact value of a target group): 256-bit two's complement words */
typedef struct {
    int32_t flags, reserved;                     /* TR_REFINE_*_INF, EXACT, MERGED */
    uint32_t group, level;
    uint64_t lo[4], hi[4];                       /* aggregate 0 */
    uint64_t dlo[4], dhi[4];                     /* aggregate 1 (RATIO) */
} tr_refine_bound;                               /* 144 bytes */

typedef struct {
    int32_t level, decision, flags, reserved;
    uint32_t group, rank;                        /* TOPK IN: its smallest possible rank (1-based; exact when
                                                    nothing is unresolved), else 0 */
    uint64_t rows;                               /* the region's rows */
    uint64_t lo[4], hi[4];                       /* the bound of A (C for a ratio), or the exact value */
    uint64_t dlo[4], dhi[4];                     /* RATIO: the bound of D */
} tr_refine_entry;                               /* 160 bytes */

typedef struct {
    uint64_t evaluated[TR_MULTI_MAX_LEVELS];     /* region bounds evaluated per level (at the target: exact values) */
    uint64_t expanded[TR_MULTI_MAX_LEVELS];      /* regions expanded per level */
    uint64_t entries, answer, unresolved;        /* entries; target groups in the answer; unresolved entries */
    uint64_t unresolved_groups;                  /* present target groups under unresolved entries */
    uint64_t work;                               /* cells read: regions x (terms + columns) */
    uint64_t fine_work;                          /* the complete fine-grain evaluation's cells: target groups x terms */
    uint64_t enumerated;                         /* links followed: TOPK descendant counts, the answer membership */
    int32_t complete, waves;                     /* 1: nothing unresolved; waves run */
    int32_t target, levels;
} tr_refine_stats;                               /* 200 bytes */

typedef struct {
    int32_t levels, ncols, nterms, mode;
    int32_t status, promise, strategy, reason;   /* MULTI-001 status of the maps; DELTA-001 promise; the last update */
    uint64_t groups[TR_MULTI_MAX_LEVELS], present[TR_MULTI_MAX_LEVELS];
    uint64_t level_bytes[TR_MULTI_MAX_LEVELS];   /* retained bytes per level (cells, inputs, maps, children) */
    uint64_t changed[TR_MULTI_MAX_LEVELS];       /* the last update's changed regions per level */
    uint64_t loose[TR_MULTI_MAX_LEVELS];         /* groups whose extrema are enclosures (after deletions) */
    uint64_t bytes, rows, step, touched;         /* retained bytes; selected rows; updates; leaves the last delta touched */
    uint64_t work, rebuild_work;                 /* the last update's cells visited; a complete rollup's cells */
    uint64_t identity[2];                        /* the leaves' DELTA-001 identity */
} tr_refine_info;                                /* 416 bytes */

typedef struct { int32_t level, reserved; uint32_t group, reserved2; int64_t drows; } tr_refine_change;   /* 24 bytes */

/* the change of one affected region during a localized update (192-bit sums as three words) */
typedef struct { uint64_t count; int64_t min, max; uint64_t mod; uint64_t dels; } tr_refine_ins;   /* 40 bytes */
/* the localized update as the rollup kernel reads it: per level l the affected regions (ids increasing), their
 * affected children (indices into level l - 1's list, CSR), and the changes in flat arrays at offset off[l] */
typedef struct {
    int32_t levels, nterms, ncols, reserved;
    uint64_t naff[TR_MULTI_MAX_LEVELS], off[TR_MULTI_MAX_LEVELS];
    const uint32_t *ids;                         /* [total]: level l's affected regions at off[l] */
    const uint64_t *child_off;                   /* [total + levels]: CSR offsets of each region (level >= 1) */
    const uint32_t *child_idx;                   /* indices into the level below (relative to its off) */
    tr_math_i128 *drows;                         /* [total] */
    tr_math_i128 *dcount;                        /* [total * nterms] */
    uint64_t *dsum;                              /* [total * nterms * 3] */
    tr_math_i128 *dpresent;                      /* [total * ncols] */
    tr_refine_ins *ins;                          /* [total * ncols] */
} tr_refine_update_view;                         /* 208 bytes */

typedef struct tr_refine_state_s tr_refine_state;
typedef struct tr_refine_query_s tr_refine_query;
typedef struct tr_refine_update_s tr_refine_update;

/* ================================================================ exact 256-bit integers (host and device) */
typedef struct { uint64_t w[4]; } tr_refine_w;   /* two's complement */
TR_NUM_HD tr_refine_w tr_refine_w_i128(__int128 x) {
    tr_refine_w r; uint64_t s = x < 0 ? UINT64_MAX : 0;
    r.w[0] = (uint64_t)x; r.w[1] = (uint64_t)((unsigned __int128)x >> 64); r.w[2] = s; r.w[3] = s; return r;
}
TR_NUM_HD tr_refine_w tr_refine_w_add(tr_refine_w a, tr_refine_w b) {
    tr_refine_w r; uint64_t c = 0;
    for (int i = 0; i < 4; i++) { uint64_t s = a.w[i] + b.w[i], c1 = s < a.w[i]; r.w[i] = s + c; c = c1 | (r.w[i] < s); }
    return r;
}
TR_NUM_HD tr_refine_w tr_refine_w_neg(tr_refine_w a) {
    tr_refine_w r; uint64_t c = 1;
    for (int i = 0; i < 4; i++) { uint64_t v = ~a.w[i] + c; c = c && v == 0; r.w[i] = v; }
    return r;
}
TR_NUM_HD tr_refine_w tr_refine_w_sub(tr_refine_w a, tr_refine_w b) { return tr_refine_w_add(a, tr_refine_w_neg(b)); }
TR_NUM_HD int tr_refine_w_neg_p(tr_refine_w a) { return (a.w[3] >> 63) != 0; }
TR_NUM_HD int tr_refine_w_cmp(tr_refine_w a, tr_refine_w b) {
    int na = tr_refine_w_neg_p(a), nb = tr_refine_w_neg_p(b);
    if (na != nb) return na ? -1 : 1;
    for (int i = 3; i >= 0; i--) if (a.w[i] != b.w[i]) return a.w[i] < b.w[i] ? -1 : 1;
    return 0;
}
/* a (|a| < 2^191) times a signed 64-bit k */
TR_NUM_HD tr_refine_w tr_refine_w_mul64(tr_refine_w a, int64_t k) {
    int neg = tr_refine_w_neg_p(a) != (k < 0);
    tr_refine_w m = tr_refine_w_neg_p(a) ? tr_refine_w_neg(a) : a;
    uint64_t kk = k < 0 ? (uint64_t)0 - (uint64_t)k : (uint64_t)k, carry = 0;
    tr_refine_w r;
    for (int i = 0; i < 4; i++) {
        unsigned __int128 p = (unsigned __int128)m.w[i] * kk + carry;
        r.w[i] = (uint64_t)p; carry = (uint64_t)(p >> 64);
    }
    return neg ? tr_refine_w_neg(r) : r;
}
/* floor(a / d) for a >= 0 and d >= 1 (64-bit), as a uint64 when it fits (else UINT64_MAX) */
TR_NUM_HD uint64_t tr_refine_w_div64(tr_refine_w a, uint64_t d) {
    unsigned __int128 r = 0;
    uint64_t q[4];
    for (int i = 3; i >= 0; i--) { unsigned __int128 cur = (r << 64) | a.w[i]; q[i] = (uint64_t)(cur / d); r = cur % d; }
    return (q[1] | q[2] | q[3]) ? UINT64_MAX : q[0];
}
/* an extended value: v, or -inf / +inf */
typedef struct { tr_refine_w v; int32_t inf; } tr_refine_x;
TR_NUM_HD tr_refine_x tr_refine_x_of(tr_refine_w v, int32_t inf) { tr_refine_x r; r.v = v; r.inf = inf; return r; }
TR_NUM_HD int tr_refine_x_cmp(tr_refine_x a, tr_refine_x b) {
    if (a.inf || b.inf) return a.inf < b.inf ? -1 : (a.inf > b.inf ? 1 : 0);
    return tr_refine_w_cmp(a.v, b.v);
}
TR_NUM_HD tr_refine_x tr_refine_x_add(tr_refine_x a, tr_refine_x b) {   /* never -inf + inf (callers keep sides) */
    if (a.inf || b.inf) return tr_refine_x_of(tr_refine_w_i128(0), a.inf ? a.inf : b.inf);
    return tr_refine_x_of(tr_refine_w_add(a.v, b.v), 0);
}

/* ================================================================ region bounds (host and device) */
/* an atom's exact value in a group: c * (sum, count or rows) */
TR_NUM_HD tr_refine_w tr_refine_atom_value(const tr_refine_atom *a, const tr_refine_level_view *lv, uint64_t g) {
    __int128 v;
    if (a->kind == TR_REFINE_ROWS) v = (__int128)lv->rows[g];
    else if (a->kind == TR_REFINE_COUNT) v = (__int128)lv->count[g * (uint64_t)lv->nterms + (uint64_t)a->term];
    else v = tr_i128_get(lv->sum[g * (uint64_t)lv->nterms + (uint64_t)a->term]);
    return tr_refine_w_mul64(tr_refine_w_i128(v), a->coef);
}
/* k e as an extended value (k = 0: exactly 0; e infinite: +-inf by the sign); |k e| < 2^64 2^63 = 2^127 */
TR_NUM_HD tr_refine_x tr_refine_ke(uint64_t k, __int128 e, int32_t einf) {
    if (!k) return tr_refine_x_of(tr_refine_w_i128(0), 0);
    if (einf) return tr_refine_x_of(tr_refine_w_i128(0), einf);
    return tr_refine_x_of(tr_refine_w_i128((__int128)k * e), 0);
}
/* E - (n - k) e as an extended value (k = n: exactly E) */
TR_NUM_HD tr_refine_x tr_refine_rest(tr_refine_w E, uint64_t n, uint64_t k, __int128 e, int32_t einf) {
    if (k == n) return tr_refine_x_of(E, 0);
    if (einf) return tr_refine_x_of(tr_refine_w_i128(0), -einf);
    tr_refine_x t = tr_refine_ke(n - k, e, 0);
    return tr_refine_x_of(tr_refine_w_sub(E, t.v), 0);
}
/* the bound of one class: every descendant's A_V lies in [*lo, *hi] (see the header comment) */
TR_NUM_HD void tr_refine_class_bound(tr_refine_w E, uint64_t n, uint64_t kmin, __int128 elo, int32_t lo_inf,
                                     __int128 ehi, int32_t hi_inf, tr_refine_x *lo, tr_refine_x *hi) {
    if (!n) { *lo = *hi = tr_refine_x_of(tr_refine_w_i128(0), 0); return; }
    if (kmin > n) kmin = n;
    uint64_t cand[8];
    int nc = 0;
    cand[nc++] = kmin; cand[nc++] = n;
    if (kmin + 1 <= n) cand[nc++] = kmin + 1;
    if (n >= 1 && n - 1 >= kmin) cand[nc++] = n - 1;
    if (!lo_inf && !hi_inf && ehi > elo) {           /* the crossings of the two lines */
        uint64_t span = (uint64_t)(ehi - elo);
        tr_refine_w a = tr_refine_w_sub(E, tr_refine_ke(n, elo, 0).v);   /* E - n elo: k* = a / span (upper) */
        if (!tr_refine_w_neg_p(a)) {
            uint64_t k = tr_refine_w_div64(a, span);
            if (k != UINT64_MAX) { if (k >= kmin && k <= n) cand[nc++] = k; if (k + 1 >= kmin && k + 1 <= n) cand[nc++] = k + 1; }
        }
        tr_refine_w b = tr_refine_w_sub(tr_refine_ke(n, ehi, 0).v, E);   /* n ehi - E: k** = b / span (lower) */
        if (!tr_refine_w_neg_p(b)) {
            uint64_t k = tr_refine_w_div64(b, span);
            if (k != UINT64_MAX) { if (k >= kmin && k <= n) cand[nc++] = k; if (k + 1 >= kmin && k + 1 <= n) cand[nc++] = k + 1; }
        }
    }
    tr_refine_x best_hi = tr_refine_x_of(tr_refine_w_i128(0), 0), best_lo = best_hi;
    for (int i = 0; i < nc; i++) {
        uint64_t k = cand[i];
        tr_refine_x u1 = tr_refine_ke(k, ehi, hi_inf), u2 = tr_refine_rest(E, n, k, elo, lo_inf ? -1 : 0);
        tr_refine_x l1 = tr_refine_ke(k, elo, lo_inf ? -1 : 0), l2 = tr_refine_rest(E, n, k, ehi, hi_inf);
        tr_refine_x u = tr_refine_x_cmp(u1, u2) <= 0 ? u1 : u2, l = tr_refine_x_cmp(l1, l2) >= 0 ? l1 : l2;
        if (!i || tr_refine_x_cmp(u, best_hi) > 0) best_hi = u;
        if (!i || tr_refine_x_cmp(l, best_lo) < 0) best_lo = l;
    }
    *lo = best_lo; *hi = best_hi;
}
/* the per-row interval of a class's expression in a region (UNC-001, exact carrier): integers [*elo, *ehi] */
TR_NUM_HD void tr_refine_class_rows(const tr_refine_plan *pl, const tr_refine_class *cl, const tr_refine_level_view *lv,
                                    uint64_t g, tr_unc_value *scratch, __int128 *elo, int32_t *lo_inf, __int128 *ehi,
                                    int32_t *hi_inf) {
    tr_unc_program p;
    p.nnodes = cl->nnodes; p.ncols = lv->ncols; p.carrier = TR_UNC_EXACT; p.precision = 0;
    p.rank = TR_UNC_MAX_TERMS; p.domains = TR_UNC_DOM_ALL; p.root = cl->root; p.reserved = 0;
    p.budget = tr_unc_qv(0, 0); p.nodes = pl->nodes + cl->first_node;
    tr_unc_plan plan;
    plan.nnodes = p.nnodes; plan.root = p.root; plan.reserved = 0;
    tr_unc_plan_slots(&p, &plan, 0);
    tr_unc_result res;
    tr_unc_eval_group(&p, &plan, lv->inputs + g * (uint64_t)lv->ncols, scratch, &res);
    *lo_inf = (res.flags & TR_UNC_LO_INF) || !(res.null & TR_UNC_MAYBE_VALUE) ? -1 : 0;
    *hi_inf = (res.flags & TR_UNC_HI_INF) || !(res.null & TR_UNC_MAYBE_VALUE) ? 1 : 0;
    *elo = 0; *ehi = 0;
    if (!*lo_inf) *elo = -tr_unc_floordiv(-(__int128)res.lo.numerator, res.lo.denominator);   /* ceil: values are integers */
    if (!*hi_inf) *ehi = tr_unc_floordiv(res.hi.numerator, res.hi.denominator);
}
/* the bound of aggregate a of region g (exact: its exact value) */
TR_NUM_HD void tr_refine_agg_bound(const tr_refine_plan *pl, int32_t agg, const tr_refine_level_view *lv, uint64_t g,
                                   int exact, tr_unc_value *scratch, tr_refine_x *lo, tr_refine_x *hi, int32_t *merged) {
    *merged = 0;
    if (exact) {
        tr_refine_w s = tr_refine_w_i128(0);
        for (int32_t c = 0; c < pl->nclasses[agg]; c++) {
            const tr_refine_class *cl = &pl->classes[pl->first[agg] + c];
            if (cl->merged) continue;
            for (int32_t i = 0; i < cl->natoms; i++) s = tr_refine_w_add(s, tr_refine_atom_value(&pl->atoms[cl->first_atom + i], lv, g));
        }
        *lo = *hi = tr_refine_x_of(s, 0);
        return;
    }
    uint64_t rows = lv->rows[g], m = (uint64_t)lv->nterms;
    int all = pl->merged[agg] >= 0;                  /* the merged class applies when every atom is valid on every row */
    for (int32_t c = 0; all && c < pl->nclasses[agg]; c++) {
        const tr_refine_class *cl = &pl->classes[pl->first[agg] + c];
        if (cl->merged) continue;
        for (int32_t i = 0; i < cl->natoms; i++) {
            const tr_refine_atom *a = &pl->atoms[cl->first_atom + i];
            if (a->kind != TR_REFINE_ROWS && lv->count[g * m + (uint64_t)a->term] != rows) all = 0;
        }
    }
    *lo = *hi = tr_refine_x_of(tr_refine_w_i128(0), 0);
    for (int32_t c = 0; c < pl->nclasses[agg]; c++) {
        const tr_refine_class *cl = &pl->classes[pl->first[agg] + c];
        if (cl->merged != all) continue;
        tr_refine_w E = tr_refine_w_i128(0);
        for (int32_t i = 0; i < cl->natoms; i++) E = tr_refine_w_add(E, tr_refine_atom_value(&pl->atoms[cl->first_atom + i], lv, g));
        uint64_t n = cl->count_term < 0 ? rows : lv->count[g * m + (uint64_t)cl->count_term];
        uint64_t kmin = n == rows ? 1 : 0;           /* every row valid: a present descendant has k = its rows >= 1 */
        __int128 elo, ehi;
        int32_t li, hi_i;
        tr_refine_class_rows(pl, cl, lv, g, scratch, &elo, &li, &ehi, &hi_i);
        tr_refine_x l, h;
        tr_refine_class_bound(E, n, kmin, elo, li, ehi, hi_i, &l, &h);
        *lo = tr_refine_x_add(*lo, l); *hi = tr_refine_x_add(*hi, h);
    }
    *merged = all;
}
/* one region: out (see tr_refine_bound) */
TR_NUM_HD void tr_refine_region_bound(const tr_refine_plan *pl, const tr_refine_level_view *lv, uint32_t level, uint32_t g,
                                      int exact, tr_unc_value *scratch, tr_refine_bound *out) {
    tr_refine_x lo, hi, dlo, dhi;
    int32_t merged = 0, m2 = 0;
    tr_refine_agg_bound(pl, 0, lv, g, exact, scratch, &lo, &hi, &merged);
    out->flags = (lo.inf ? TR_REFINE_LO_INF : 0) | (hi.inf ? TR_REFINE_HI_INF : 0) | (exact ? TR_REFINE_EXACT : 0) |
                 (merged ? TR_REFINE_MERGED : 0);
    out->reserved = 0; out->group = g; out->level = level;
    for (int i = 0; i < 4; i++) { out->lo[i] = lo.v.w[i]; out->hi[i] = hi.v.w[i]; out->dlo[i] = 0; out->dhi[i] = 0; }
    if (pl->naggs > 1) {
        tr_refine_agg_bound(pl, 1, lv, g, exact, scratch, &dlo, &dhi, &m2);
        out->flags |= (dlo.inf ? TR_REFINE_DLO_INF : 0) | (dhi.inf ? TR_REFINE_DHI_INF : 0);
        for (int i = 0; i < 4; i++) { out->dlo[i] = dlo.v.w[i]; out->dhi[i] = dhi.v.w[i]; }
    }
}

/* ================================================================ localized rollups (host and device) */
TR_NUM_HD void tr_refine_ins_merge(tr_refine_ins *a, const tr_refine_ins *b) {
    a->dels += b->dels;
    if (!b->count) return;
    if (!a->count) { uint64_t d = a->dels; *a = *b; a->dels = d; return; }
    uint64_t d = a->min > b->min ? (uint64_t)a->min - (uint64_t)b->min : (uint64_t)b->min - (uint64_t)a->min;
    a->mod = tr_gcd_u64(tr_gcd_u64(a->mod, b->mod), d);
    if (b->min < a->min) a->min = b->min;
    if (b->max > a->max) a->max = b->max;
    a->count += b->count;
}
/* 192-bit words += another's */
TR_NUM_HD void tr_refine_add192(uint64_t *w, const uint64_t *x) {
    uint64_t c = 0;
    for (int i = 0; i < 3; i++) { uint64_t s = w[i] + x[i], c1 = s < w[i]; w[i] = s + c; c = c1 | (w[i] < s); }
}
/* the change of affected region i of level l (>= 1): the sum of its affected children's changes */
TR_NUM_HD void tr_refine_rollup_region(const tr_refine_update_view *u, int32_t l, uint64_t i) {
    uint64_t me = u->off[l] + i, below = u->off[l - 1], m = (uint64_t)u->nterms, nc = (uint64_t)u->ncols;
    __int128 dr = 0;
    for (uint64_t j = 0; j < m; j++) { u->dcount[me * m + j] = tr_i128_set(0); for (int w = 0; w < 3; w++) u->dsum[(me * m + j) * 3 + w] = 0; }
    for (uint64_t c = 0; c < nc; c++) {
        u->dpresent[me * nc + c] = tr_i128_set(0);
        tr_refine_ins *z = &u->ins[me * nc + c];
        z->count = 0; z->min = 0; z->max = 0; z->mod = 0; z->dels = 0;
    }
    uint64_t a = u->child_off[me + (uint64_t)l], b = u->child_off[me + (uint64_t)l + 1];
    for (uint64_t k = a; k < b; k++) {
        uint64_t ch = below + u->child_idx[k];
        dr += tr_i128_get(u->drows[ch]);
        for (uint64_t j = 0; j < m; j++) {
            u->dcount[me * m + j] = tr_i128_set(tr_i128_get(u->dcount[me * m + j]) + tr_i128_get(u->dcount[ch * m + j]));
            tr_refine_add192(&u->dsum[(me * m + j) * 3], &u->dsum[(ch * m + j) * 3]);
        }
        for (uint64_t c = 0; c < nc; c++) {
            u->dpresent[me * nc + c] = tr_i128_set(tr_i128_get(u->dpresent[me * nc + c]) + tr_i128_get(u->dpresent[ch * nc + c]));
            tr_refine_ins_merge(&u->ins[me * nc + c], &u->ins[ch * nc + c]);
        }
    }
    u->drows[me] = tr_i128_set(dr);
}

/* the device copies of tr_mgpu_refine_bounds (and their host staging): the n regions' cells and inputs gathered,
 * the plan, the groups, the outputs and the UNC-001 scratch of a chunk */
static inline uint64_t tr_refine_bounds_device_bytes(const tr_refine_level_view *lv, const tr_refine_plan *pl, uint64_t n) {
    uint64_t m = (uint64_t)(lv->nterms > 0 ? lv->nterms : 1);
    uint64_t b = tr_budget_mul(2, tr_budget_add(tr_budget_mul(n, 8), tr_budget_mul(tr_budget_mul(n, m), 24)));
    b = tr_budget_add(b, tr_budget_mul(2, tr_budget_mul(tr_budget_mul(n, (uint64_t)(lv->ncols > 0 ? lv->ncols : 1)),
                                                        sizeof(tr_unc_input))));
    b = tr_budget_add(b, tr_budget_add(tr_budget_mul((uint64_t)pl->natoms + 1, sizeof(tr_refine_atom)),
                                       tr_budget_mul((uint64_t)pl->nclass + 1, sizeof(tr_refine_class))));
    b = tr_budget_add(b, tr_budget_mul((uint64_t)pl->nnodes + 1, sizeof(tr_unc_node)));
    b = tr_budget_add(b, tr_budget_mul(n, 4 + sizeof(tr_refine_bound)));
    uint64_t chunk = n < TR_REFINE_GPU_CHUNK ? n : TR_REFINE_GPU_CHUNK;
    return tr_budget_add(b, tr_budget_mul(tr_budget_mul(chunk ? chunk : 1, (uint64_t)pl->nlive + 1), sizeof(tr_unc_value)));
}
/* the device copies of tr_mgpu_refine_rollup: the view's arrays */
static inline uint64_t tr_refine_rollup_device_bytes(const tr_refine_update_view *u, uint64_t nchild) {
    uint64_t total = 0;
    for (int32_t l = 0; l < u->levels; l++) total = tr_budget_add(total, u->naff[l]);
    uint64_t m = (uint64_t)u->nterms, c = (uint64_t)u->ncols;
    uint64_t per = tr_budget_add(tr_budget_add(4 + 8 + 16, tr_budget_mul(m, 16 + 24)), tr_budget_mul(c, 16 + sizeof(tr_refine_ins)));
    return tr_budget_add(tr_budget_mul(total, per), tr_budget_add(tr_budget_mul(nchild, 4), 8 * (uint64_t)u->levels));
}

/* ================================================================ exported (math_refine.c) */
/* the state of a MULTI-001 hierarchy whose level-0 bank is the bank of scope0 (its UNC-001 inputs are built from it;
 * promise: DELTA-001's TR_DELTA_NONNEGATIVE for the leaves; TR_DELTA_SIGNED is TR_MATH_UNSUPPORTED) */
tr_refine_state *tr_refine_build(const tr_multi_hier *, const tr_scope *scope0, int32_t promise, const tr_math_budget *,
                                 tr_math_refusal *, int32_t *error);
int32_t tr_refine_build_need(const tr_multi_hier *, const tr_scope *scope0, tr_math_need *);
int32_t tr_refine_get_info(const tr_refine_state *, tr_refine_info *);
/* level l's cells and inputs (borrowed) */
int32_t tr_refine_get_level(const tr_refine_state *, int32_t level, tr_refine_level_view *);
/* parent[g] of level l (< levels - 1) and the children CSR of level l (>= 1): offsets[groups + 1], ids */
int32_t tr_refine_get_links(const tr_refine_state *, int32_t level, const uint32_t **parent, const uint64_t **child_off,
                            const uint32_t **child_ids);
/* 1: group g of level l has exact extrema (no deletion since they were computed), 0: an enclosure */
int32_t tr_refine_tight(const tr_refine_state *, int32_t level, uint64_t g);
void tr_refine_free(tr_refine_state *);

/* Queries. begin validates and compiles the request (budget: support, circuit, output, iterations, bytes); wave
 * describes the pending wave (0: the query is finished); supply adopts its bounds (computed by tr_refine_eval_bounds
 * or tr_mgpu_refine_bounds) and decides it; run computes every remaining wave on the CPU. */
tr_refine_query *tr_refine_begin(const tr_refine_state *, const tr_refine_request *, const tr_math_budget *,
                                 tr_math_refusal *, int32_t *error);
/* the needs tr_refine_begin checks (the request compiled) */
int32_t tr_refine_query_need(const tr_refine_state *, const tr_refine_request *, tr_math_need *);
/* the needs tr_mgpu_refine_bounds checks for n regions (output, device bytes) */
int32_t tr_refine_bounds_need(const tr_refine_level_view *, const tr_refine_plan *, uint64_t n, tr_math_need *);
int32_t tr_refine_wave(const tr_refine_query *, int32_t *level, int32_t *exact, const uint32_t **groups, uint64_t *n);
int32_t tr_refine_supply(tr_refine_query *, const tr_refine_bound *bounds, uint64_t n);
int32_t tr_refine_run(tr_refine_query *);
int32_t tr_refine_get_plan(const tr_refine_query *, tr_refine_plan *);
int32_t tr_refine_get_stats(const tr_refine_query *, tr_refine_stats *);
/* entries in decision order (out NULL and capacity 0: the count only) */
int32_t tr_refine_get_entries(const tr_refine_query *, tr_refine_entry *out, uint64_t capacity, uint64_t *count);
/* the target groups in the answer and those under unresolved entries (ceil(groups / 64) words each; either NULL) */
int32_t tr_refine_get_answer(tr_refine_query *, uint64_t *answer, uint64_t *unresolved, uint64_t capacity);
void tr_refine_query_free(tr_refine_query *);
/* the bounds of n regions of one level (exact: the target level), the CPU reference of tr_mgpu_refine_bounds */
int32_t tr_refine_eval_bounds(const tr_refine_level_view *, const tr_refine_plan *, int32_t level, int32_t exact,
                              const uint32_t *groups, uint64_t n, tr_refine_bound *out);
/* the complete fine-grain answer of the request from level L's states (the baseline): answer bitmap, the exact value
 * of every target group (out[groups], may be NULL), stats.fine_work */
int32_t tr_refine_fine(const tr_refine_state *, const tr_refine_request *, uint64_t *answer, uint64_t capacity,
                       tr_refine_bound *values, tr_refine_stats *, const tr_math_budget *, tr_math_refusal *);

/* Change localization. update = prepare + the CPU rollup + finish; levels (NULL, or nlevels new levels) enable the
 * complete rebuild when the local path is not compatible. The CUDA path: prepare, tr_mgpu_refine_rollup on the view,
 * finish. violation (may be NULL): a contradicted promise. */
tr_refine_state *tr_refine_update_state(const tr_refine_state *, const tr_delta *, const tr_multi_level *levels,
                                        int32_t nlevels, const tr_math_budget *, tr_math_refusal *,
                                        tr_delta_violation *violation, int32_t *error);
tr_refine_update *tr_refine_update_prepare(const tr_refine_state *, const tr_delta *, const tr_math_budget *,
                                           tr_math_refusal *, tr_delta_violation *violation, int32_t *reason,
                                           int32_t *error);
/* the needs of a localized update, resource by resource the largest of its checks: DELTA-001's apply, prepare's
 * arrays and finish's new state (not of a rebuild) */
int32_t tr_refine_update_need(const tr_refine_state *, const tr_delta *, tr_math_need *);
/* the needs tr_mgpu_refine_rollup checks (states, device bytes) */
int32_t tr_refine_rollup_need(const tr_refine_update_view *, uint64_t nchild, tr_math_need *);
int32_t tr_refine_update_view_get(tr_refine_update *, tr_refine_update_view *, uint64_t *nchild);
/* the CPU rollup of every level (the reference of the kernel) */
int32_t tr_refine_update_rollup(tr_refine_update *);
/* adopts the coarse changes tr_mgpu_refine_rollup wrote into the view: checked to conserve every level's totals
 * (rows, counts, sums, present values: each level's sum equals the leaves'), else TR_MATH_INVALID */
int32_t tr_refine_update_adopt(tr_refine_update *);
tr_refine_state *tr_refine_update_finish(tr_refine_update *, const tr_math_budget *, tr_math_refusal *, int32_t *error);
void tr_refine_update_free(tr_refine_update *);
/* the last update's changes: changes[count] and per change dcount[nterms] (int64) and dsum[nterms * 3] (192-bit);
 * every pointer NULL and capacity 0: the count only */
int32_t tr_refine_get_changes(const tr_refine_state *, tr_refine_change *out, int64_t *dcount, uint64_t *dsum,
                              uint64_t capacity, uint64_t *count);

/* ------------------------------------------------------------------ the CUDA library (math_refine_gpu.cu) */
/* the bounds of n regions (groups[n] of the level lv describes) with the compiled plan: out[n], equal to
 * tr_refine_eval_bounds. Budget: output = n, bytes = tr_refine_bounds_device_bytes. */
int tr_mgpu_refine_bounds(const tr_refine_level_view *lv, const tr_refine_plan *plan, int32_t level, int32_t exact,
                          const uint32_t *groups, uint64_t n, tr_refine_bound *out, const tr_math_budget *budget,
                          tr_math_refusal *refusal);
/* the localized rollup of every level >= 1 of an update view (its level-0 changes given), written into the view's
 * arrays (host memory), equal to tr_refine_update_rollup. Budget: bytes = tr_refine_rollup_device_bytes. */
int tr_mgpu_refine_rollup(tr_refine_update_view *view, uint64_t nchild, const tr_math_budget *budget,
                          tr_math_refusal *refusal);
#ifdef __cplusplus
}
#endif
#endif
