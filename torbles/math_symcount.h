/* COUNT-002: symbolic counting over a declared full assignment domain (the assignment universe, not observed rows):
 * exact model counts and weighted model counts of BOOL-001 predicate circuits by compilation to reduced ordered
 * binary decision diagrams within a node / step budget, of finite-domain factor systems (constraint and weight tables)
 * by variable elimination within a width budget, with witnesses, brute-force verification and a bounded Shannon
 * expansion that returns certified bounds instead of an extrapolation; on the CPU (math_symcount.c) and, for the
 * data-parallel parts (the brute-force enumeration of 2^k assignments, the level-by-level count of a compiled
 * diagram), on a CUDA device (math_symcount_gpu.cu) with the same values bit for bit.
 * SPDX-License-Identifier: Apache-2.0
 *
 * ---- The domain: the assignment universe ---------------------------------------------------------------------------
 * Every count here is over a DECLARED universe of assignments, never over stored rows: the 2^k assignments of k
 * two-valued variables (a program's conditions 0..k-1; bit i of an assignment is variable i), or the prod_i d_i
 * assignments of n finite-domain variables (variable i takes 0 .. d_i - 1). Observed-row counts are COUNT-001's
 * (math_count.h); an answer here always says which universe it counts (tr_symc_answer_info.universe_vars and the
 * TOTAL value: 2^k or prod d_i assignments, or their total weight).
 *
 * ---- Weights (exact, nonnegative) ---------------------------------------------------------------------------------
 * A literal weight is a rational num/den (int64, num >= 0, den >= 1). Variable v has w(v = 1) = num1/den1 and
 * w(v = 0) = num0/den0 (NULL weights: 1 and 1, model counting). The weight of an assignment is the product of its
 * literals' weights; a weighted count is the sum over the assignments of a class. Natively every variable is scaled
 * to integers over a common denominator d_v = lcm(den1, den0) (< 2^126): a_v = num1 d_v / den1, b_v = num0 d_v /
 * den0 (each < 2^126, s_v = a_v + b_v < 2^127): the count is N / D with N = sum over the class of prod (a_v or b_v)
 * and D = prod d_v, N computed exactly in L 64-bit limbs where the host proves every intermediate value below
 * prod s_v < 2^(64 L) first (nonnegative weights: every partial count is at most the universe's total), then reduced
 * to a canonical rational with FND-009's big integers. A probability p = n/d is the weight pair (n/d, (d - n)/d):
 * then s_v = d_v and the total weight is 1.
 * Factor systems (dependent weights): factor f over a scope of variables has a table of nonnegative rationals, one
 * per joint assignment of its scope (C order: the last scope variable varies fastest); the weight of an assignment
 * is the product of its factors' entries (a zero entry forbids it: a hard constraint); Z = the sum over the universe.
 * Natively each factor is scaled to integers over D_f = lcm of its denominators (< 2^126).
 *
 * ---- Programs and their three values -------------------------------------------------------------------------------
 * A program is a BOOL-001 DAG (math_bool.h: CONST, VAR, NOT, AND, OR, XOR; Kleene's three-valued logic). At every
 * assignment it is TRUE, FALSE or UNKNOWN (UNKNOWN arises only from the UNKNOWN constant); the classes partition the
 * universe and TRUE + FALSE + UNKNOWN = TOTAL in every weighting. Compiled two-valued (no reachable UNKNOWN constant:
 * one root T, FALSE = TOTAL - TRUE) or three-valued (two roots: T = [f is TRUE], F = [f is FALSE], each a Boolean
 * function: NOT (T, F) = (F, T); AND = (Ta Tb, Fa | Fb); OR = (Ta | Tb, Fa Fb); XOR = (Ta Fb | Fa Tb, Ta Tb | Fa Fb);
 * VAR = (x, NOT x); TRUE = (1, 0), FALSE = (0, 1), UNKNOWN = (0, 0)).
 *
 * ---- Decision diagrams (tr_symc_mgr builds, tr_symc_dd is the immutable result) -----------------------------------
 * A reduced ordered BDD over nlevels levels: level i tests variable order[i]; node 0 is FALSE, node 1 is TRUE (both
 * at level nlevels); an internal node (level, lo, hi) has lo != hi (reduced) and is unique (hash-consed): the
 * diagram of a function under an order is canonical. Apply (AND, OR, XOR; NOT f = f XOR TRUE) is the Shannon
 * recursion on the top level with a lossy computed table, iterative (an explicit stack of at most nlevels + 1
 * frames). Orders: NATURAL (level i = variable i), GIVEN (a permutation), DFS (first appearance in a depth-first walk
 * of the program from its root, left operand first; the variables it never reads after, in increasing index). The
 * immutable diagram keeps the nodes the roots reach, renumbered from the deepest level up (terminals 0, 1, then the
 * nodes of level nlevels-1, ..., level 0; within a level by creation), so a node's children have smaller ids and the
 * nodes of level l are [first[l], first[l] + count[l]).
 * Counting (tr_symc_node_count, shared by the CPU loop and the kernel): V(FALSE) = 0, V(TRUE) = 1 and for a node at
 * level l, V = a_l V(hi) G(hi) + b_l V(lo) G(lo) with G(c) = prod of s_m over the levels skipped by the edge (l < m <
 * level(c)); a root r contributes V(r) times the s_m of the levels above it. CPU: nodes in increasing id; CUDA: one
 * launch per level from the deepest, one work item per node into its own L-limb slot.
 * Witnesses: the first m assignments of a root's class in lexicographic order of the levels (level 0 most
 * significant, 0 before 1), by a depth-first walk that never backtracks (every node but FALSE has a model); returned
 * as k-bit masks over the variables (bit v = variable v), k <= 64.
 *
 * ---- Brute force (verification) ----------------------------------------------------------------------------------
 * Every assignment of the k <= 63 variables evaluated: 64 per word with BOOL-001's word evaluator on its ASSIGNMENTS
 * domain, classes counted with popcounts, weighted sums per word as H(word) sum_p Lo(p) (the high variables' weight
 * product times the low six variables' weight of each assignment of the class). CUDA: work items are slices of words,
 * local sums, one 64-bit atomic per count and a multi-word atomic add per weighted sum (exact modulo 2^(64 L), exact
 * because the host proves the total below 2^(64 L)).
 *
 * ---- Bounded Shannon expansion (tr_symc_search): exact or certified bounds, never extrapolated --------------------
 * Branches are partial assignments of the levels 0 .. depth-1, breadth first; a branch is evaluated once (one
 * evaluation of the budget) by propagating the sets of possible values ({FALSE, TRUE, UNKNOWN} subsets) of every
 * program node, the unassigned variables {FALSE, TRUE}: an over-approximation, so a singleton is the program's value
 * at every completion and the branch's total weight is exact for that class. Undecided branches split on the next
 * level. When max_evaluations is reached the unevaluated branches stay open: each adds its weight to the upper bound
 * of every class its parent left possible. Values: lo <= class count <= hi for each class, exact (lo = hi) when no
 * branch is open; the open branches and evaluations are reported.
 *
 * ---- Variable elimination over factors (tr_symc_factors_count) ---------------------------------------------------
 * Order: GIVEN (a permutation), MIN_FILL or MIN_DEGREE (greedy on the interaction graph, ties to the smallest index).
 * Each table goes to the bucket of its first-eliminated variable; eliminating x multiplies its bucket's tables and
 * sums x out (a message over the union of their scopes minus x, into the bucket of its first-eliminated variable; a
 * constant multiplies Z). Width = the largest joint table |scope(bucket) + x| (tensor), refused before anything is
 * computed with the exact need. Witnesses: the first m models in lexicographic order of the reverse elimination order
 * (the last-eliminated variable most significant), decoded without backtracking: x takes the values v whose bucket
 * product at (v, the decoded later variables) is nonzero (every such branch completes to a model). INDICATOR mode
 * counts the models (every nonzero entry read as 1).
 *
 * ---- Answers (tr_symc_answer: immutable) ---------------------------------------------------------------------------
 * Canonical nonnegative rationals (FND-009 big integers; an integer has denominator 1) in a layout per method:
 *   BDD, BRUTE      [TRUE, FALSE, UNKNOWN, TOTAL]                         (TOTAL = prod s_v / prod d_v; 2^k unweighted)
 *   SEARCH          [TRUE lo, TRUE hi, FALSE lo, FALSE hi, UNKNOWN lo, UNKNOWN hi, TOTAL]
 *   ELIM, FBRUTE    [Z, ASSIGNMENTS]                                       (ASSIGNMENTS = prod d_i)
 * with status EXACT or INCOMPLETE (SEARCH with open branches) and the witnesses asked for.
 *
 * ---- Budgets (math_budgets.h), checked before allocation (TR_MATH_BUDGET with the tr_math_refusal) ---------------
 *   manager: circuit = nodes (terminals included; a node past the limit refuses the operation with need = the node
 *     that did not fit), iterations = apply steps (Shannon expansions not answered by a terminal case or the computed
 *     table), bytes = node, unique-table, cache and stack arrays (checked at every growth); the levels are bounded by
 *     TR_SYMC_MAX_LEVELS (a program's k is its variables need)
 *   count: circuit = nodes, output = roots, bytes = node values (nodes x L x 8) and scratch (CUDA: every device copy,
 *     the values and the per-node scratch of the widest level)
 *   witnesses: output = m; brute force: variables = k, support = 2^k, circuit = program nodes, bytes = scratch
 *   search: iterations = max_evaluations, bytes = the branch queue and the scratch
 *   factors: variables = n, support = table entries, tensor = the largest table, bytes = the object; elimination:
 *     tensor = the width, iterations = the products computed, bytes = every message (L limbs per entry) and scratch,
 *     output = witnesses; factor brute force: support = prod d_i
 * Representation limits (TR_MATH_UNSUPPORTED): programs over k > 64 variables (63 for brute force), more than
 * TR_SYMC_MAX_LEVELS levels, numbers past TR_SYMC_MAX_LIMBS limbs, factor scopes of more than TR_SYMC_MAX_SCOPE
 * variables, domains past TR_SYMC_MAX_DOMAIN, factor denominators past 2^126.
 * Errors: TR_MATH_INVALID (-1), _MEMORY (-2), _OVERFLOW (-3: a carrier overflow; never wrapped), _BUDGET (-5),
 * TR_MATH_DEVICE (-11), _UNSUPPORTED (-12). Objects are immutable once built (the manager is a builder owned by one
 * caller); constructors return NULL and set *error. */
#ifndef TORBLES_MATH_SYMCOUNT_H
#define TORBLES_MATH_SYMCOUNT_H
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "math_num.h"
#include "math_budgets.h"
#include "math_bool.h"
#ifdef __cplusplus
extern "C" {
#endif

#define TR_SYMC_ABI 1
#define TR_SYMC_MAX_PROGRAM_VARS 64        /* variables of a compiled program (witness masks are uint64) */
#define TR_SYMC_MAX_BRUTE_VARS 63          /* variables of a brute-force enumeration */
#define TR_SYMC_MAX_LEVELS (UINT32_C(1) << 24)
#define TR_SYMC_MAX_LIMBS 4096             /* 64-bit limbs of an exact count (262144 bits) */
#define TR_SYMC_MAX_SCOPE 31               /* variables of one factor */
#define TR_SYMC_MAX_DOMAIN (UINT32_C(1) << 16)
#define TR_SYMC_MAX_FACTOR_VARS (UINT32_C(1) << 20)
#define TR_SYMC_MAX_HEURISTIC 4096         /* variables of a MIN_FILL / MIN_DEGREE order (adjacency bitsets) */
#define TR_SYMC_DEVICE_ITEMS (UINT64_C(1) << 14)   /* brute-force work items per launch (at most) */
#define TR_SYMC_NONE UINT32_MAX

enum { TR_SYMC_FALSE = 0, TR_SYMC_TRUE = 1 };                                    /* terminal node ids */
enum { TR_SYMC_AND = 1, TR_SYMC_OR = 2, TR_SYMC_XOR = 3 };                       /* apply operations */
enum { TR_SYMC_ORDER_NATURAL = 0, TR_SYMC_ORDER_GIVEN = 1, TR_SYMC_ORDER_DFS = 2,
       TR_SYMC_ORDER_MIN_FILL = 3, TR_SYMC_ORDER_MIN_DEGREE = 4 };               /* orders */
enum { TR_SYMC_M_BDD = 1, TR_SYMC_M_BRUTE = 2, TR_SYMC_M_SEARCH = 3, TR_SYMC_M_ELIM = 4, TR_SYMC_M_FBRUTE = 5 };
enum { TR_SYMC_EXACT = 1, TR_SYMC_INCOMPLETE = 2 };                              /* answer status */
enum { TR_SYMC_BY_CPU = 1, TR_SYMC_BY_DEVICE = 2 };                              /* who computed the raw values */
enum { TR_SYMC_NEED_CPU = 0, TR_SYMC_NEED_DEVICE = 1 };
enum { TR_SYMC_WEIGHTED = 1, TR_SYMC_INDICATOR = 2 };                            /* factor modes */

/* ------------------------------------------------------------------ structs crossing the ctypes boundary */
typedef struct { uint32_t level, lo, hi, reserved; } tr_symc_node;              /* 16 bytes */
typedef struct { int64_t num1, den1, num0, den0; } tr_symc_weight;             /* 32 bytes: w(v=1), w(v=0) */

/* a diagram as the CPU loops and the kernels read it (pointers into the diagram; valid while it lives) */
typedef struct {
    const tr_symc_node *node;      /* nnodes: 0 FALSE, 1 TRUE, then the nodes of level nlevels-1 .. 0 */
    const uint32_t *roots;         /* nroots */
    const uint32_t *order;         /* nlevels: the variable tested at each level */
    const uint64_t *first, *count; /* nlevels: the nodes of level l are [first[l], first[l] + count[l]) */
    uint64_t nnodes;
    uint32_t nlevels, nroots, nvars, three_valued;   /* nvars: variables of the universe (order entries < nvars) */
} tr_symc_dd_view;                 /* 64 bytes */

typedef struct {
    uint64_t nnodes, peak_nodes, steps, bytes;   /* diagram nodes; nodes the manager held; apply steps; bytes owned */
    uint64_t width;                              /* the most nodes on one level */
    uint32_t nlevels, nroots, nvars, three_valued;
} tr_symc_dd_info;                 /* 56 bytes */

typedef struct {
    int32_t method, status, weighted, three_valued;
    int32_t backend, nvalues;
    uint32_t limbs, universe_vars;   /* the largest numerator / denominator; variables of the universe */
    uint64_t witnesses, witness_width;   /* witnesses held; entries per witness (1: a mask; n: one value per variable) */
    uint64_t nodes, width;           /* BDD: nodes and the widest level; ELIM: the largest joint table */
    uint64_t evaluations, open;      /* SEARCH: branches evaluated, open branches; ELIM: products computed */
} tr_symc_answer_info;               /* 80 bytes */

typedef struct { uint64_t scope_at, table_at; uint32_t nscope, reserved; } tr_symc_factor;   /* 24 bytes */

typedef struct {
    uint64_t entries, bytes, assignments_lo, assignments_hi;   /* table entries; bytes; prod d_i (128 bits, or 0 / 0
                                                                   with assignments_wide set when it does not fit) */
    uint32_t nvars, nfactors, max_scope, assignments_wide;
} tr_symc_factors_info;              /* 48 bytes */

typedef struct tr_symc_mgr_s tr_symc_mgr;
typedef struct tr_symc_dd_s tr_symc_dd;
typedef struct tr_symc_answer_s tr_symc_answer;
typedef struct tr_symc_factors_s tr_symc_factors;

/* ------------------------------------------------------------------ exact L-limb unsigned integers (CPU and CUDA) */
TR_NUM_HD void tr_symc_w_zero(uint64_t *x, uint32_t L) { for (uint32_t i = 0; i < L; i++) x[i] = 0; }
TR_NUM_HD void tr_symc_w_copy(uint64_t *d, const uint64_t *s, uint32_t L) { for (uint32_t i = 0; i < L; i++) d[i] = s[i]; }
TR_NUM_HD void tr_symc_w_set128(uint64_t *x, uint32_t L, uint64_t lo, uint64_t hi) {
    tr_symc_w_zero(x, L);
    x[0] = lo;
    if (L > 1) x[1] = hi;
}
TR_NUM_HD int tr_symc_w_is_zero(const uint64_t *x, uint32_t L) {
    for (uint32_t i = 0; i < L; i++) if (x[i]) return 0;
    return 1;
}
/* acc += x; 1 when the sum leaves L limbs */
TR_NUM_HD int tr_symc_w_add(uint64_t *acc, const uint64_t *x, uint32_t L) {
    uint64_t carry = 0;
    for (uint32_t i = 0; i < L; i++) {
        uint64_t y = x[i] + carry, c1 = y < carry;
        uint64_t s = acc[i] + y, c2 = s < y;
        acc[i] = s;
        carry = c1 | c2;
    }
    return carry != 0;
}
/* acc -= x; 1 when x > acc (a borrow out) */
TR_NUM_HD int tr_symc_w_sub(uint64_t *acc, const uint64_t *x, uint32_t L) {
    uint64_t borrow = 0;
    for (uint32_t i = 0; i < L; i++) {
        uint64_t y = x[i] + borrow, b1 = y < borrow;
        uint64_t d = acc[i] - y, b2 = acc[i] < y;
        acc[i] = d;
        borrow = b1 | b2;
    }
    return borrow != 0;
}
/* out = x (whi 2^64 + wlo), out distinct from x; 1 when the product leaves L limbs */
TR_NUM_HD int tr_symc_w_mul128(const uint64_t *x, uint32_t L, uint64_t wlo, uint64_t whi, uint64_t *out) {
    unsigned __int128 carry = 0;
    for (uint32_t i = 0; i < L; i++) {
        unsigned __int128 p = (unsigned __int128)x[i] * wlo + carry;
        out[i] = (uint64_t)p;
        carry = p >> 64;
    }
    int over = carry != 0;
    if (whi) {
        if (x[L - 1]) over = 1;
        carry = 0;
        for (uint32_t i = 0; i + 1 < L; i++) {
            unsigned __int128 p = (unsigned __int128)x[i] * whi + out[i + 1] + carry;
            out[i + 1] = (uint64_t)p;
            carry = p >> 64;
        }
        if (carry) over = 1;
    }
    return over;
}
/* x *= w through the scratch t (L limbs) */
TR_NUM_HD int tr_symc_w_scale(uint64_t *x, uint64_t *t, uint32_t L, uint64_t wlo, uint64_t whi) {
    int over = tr_symc_w_mul128(x, L, wlo, whi, t);
    tr_symc_w_copy(x, t, L);
    return over;
}
/* out = x y (schoolbook, out distinct from x and y); 1 when the product leaves L limbs */
TR_NUM_HD int tr_symc_w_mul(const uint64_t *x, const uint64_t *y, uint32_t L, uint64_t *out) {
    tr_symc_w_zero(out, L);
    int over = 0;
    uint32_t nx = L, ny = L;
    while (nx && !x[nx - 1]) nx--;
    while (ny && !y[ny - 1]) ny--;
    if (!nx || !ny) return 0;
    if (nx + ny - 1 > L) over = 1;
    for (uint32_t i = 0; i < nx; i++) {
        unsigned __int128 carry = 0;
        if (!x[i]) continue;
        for (uint32_t j = 0; j < ny; j++) {
            if (i + j >= L) { if (y[j] || carry) over = 1; continue; }
            unsigned __int128 p = (unsigned __int128)x[i] * y[j] + out[i + j] + carry;
            out[i + j] = (uint64_t)p;
            carry = p >> 64;
        }
        for (uint32_t t = i + ny; carry; t++) {
            if (t >= L) { over = 1; break; }
            unsigned __int128 p = (unsigned __int128)out[t] + carry;
            out[t] = (uint64_t)p;
            carry = p >> 64;
        }
    }
    return over;
}

/* ------------------------------------------------------------------ the diagram count (CPU and CUDA) */
/* scaled weights per level (or per variable), 4 words each: a (w(1)) lo, hi, b (w(0)) lo, hi; s = a + b < 2^127 */
TR_NUM_HD void tr_symc_level_sum(const uint64_t *w, uint32_t l, uint64_t *lo, uint64_t *hi) {
    unsigned __int128 a = ((unsigned __int128)w[4 * (uint64_t)l + 1] << 64) | w[4 * (uint64_t)l];
    unsigned __int128 b = ((unsigned __int128)w[4 * (uint64_t)l + 3] << 64) | w[4 * (uint64_t)l + 2];
    unsigned __int128 s = a + b;
    *lo = (uint64_t)s; *hi = (uint64_t)(s >> 64);
}
TR_NUM_HD uint32_t tr_symc_level(const tr_symc_dd_view *d, uint32_t id) {
    return id < 2 ? d->nlevels : d->node[id].level;
}
/* x *= prod of s_m for from <= m < to (the levels an edge skips) */
TR_NUM_HD int tr_symc_skip(const uint64_t *w, uint32_t from, uint32_t to, uint64_t *x, uint64_t *t, uint32_t L) {
    int over = 0;
    for (uint32_t m = from; m < to; m++) {
        uint64_t lo, hi;
        tr_symc_level_sum(w, m, &lo, &hi);
        over |= tr_symc_w_scale(x, t, L, lo, hi);
    }
    return over;
}
/* the value of node i >= 2 from its children's values (values[c L .. c L + L) for c < i; terminals 0 and 1 hold 0
 * and 1); tmp: 2 L words. Returns 1 on an overflow (excluded by the host's bound). */
TR_NUM_HD int tr_symc_node_count(const tr_symc_dd_view *d, const uint64_t *w, uint32_t L, uint64_t *values, uint64_t i,
                                 uint64_t *tmp) {
    const tr_symc_node n = d->node[i];
    uint64_t *out = values + i * (uint64_t)L, *a = tmp, *t = tmp + L;
    int over = 0;
    tr_symc_w_copy(a, values + (uint64_t)n.hi * L, L);
    over |= tr_symc_skip(w, n.level + 1, tr_symc_level(d, n.hi), a, t, L);
    over |= tr_symc_w_scale(a, t, L, w[4 * (uint64_t)n.level], w[4 * (uint64_t)n.level + 1]);
    tr_symc_w_copy(out, a, L);
    tr_symc_w_copy(a, values + (uint64_t)n.lo * L, L);
    over |= tr_symc_skip(w, n.level + 1, tr_symc_level(d, n.lo), a, t, L);
    over |= tr_symc_w_scale(a, t, L, w[4 * (uint64_t)n.level + 2], w[4 * (uint64_t)n.level + 3]);
    over |= tr_symc_w_add(out, a, L);
    return over;
}
/* the total of root r: V(root) times the s_m of the levels above it, into out (L words); tmp: L words */
TR_NUM_HD int tr_symc_root_count(const tr_symc_dd_view *d, const uint64_t *w, uint32_t L, const uint64_t *values,
                                 uint32_t r, uint64_t *out, uint64_t *tmp) {
    uint32_t id = d->roots[r];
    tr_symc_w_copy(out, values + (uint64_t)id * L, L);
    return tr_symc_skip(w, 0, tr_symc_level(d, id), out, tmp, L);
}
/* the terminal values (the rest of values is written by the counts) */
TR_NUM_HD void tr_symc_terminals(uint64_t *values, uint32_t L) {
    tr_symc_w_zero(values, 2 * L);
    values[L] = 1;
}

/* ------------------------------------------------------------------ brute force (CPU and CUDA) */
/* the class words of the program's root at word w of its ASSIGNMENTS domain (val: 2 nnodes words of scratch) */
TR_NUM_HD void tr_symc_brute_classes(const tr_bool_program_view *p, const tr_bool_domain *d, uint64_t w, uint64_t *val,
                                     uint64_t *t, uint64_t *f, uint64_t *u) {
    tr_bool_eval_base(p, d, w, val);
    uint64_t v = val[2 * (uint64_t)p->root], k = val[2 * (uint64_t)p->root + 1], sel = tr_bool_selection_word(d, w);
    *t = v & k & sel;
    *f = k & ~v & sel;
    *u = sel & ~k;
}
/* acc += the weight of every assignment of word w whose bit is set in bits (w per variable, 4 words each; k
 * variables): H(w) sum over the set bits p of Lo(p); scratch: 3 L words. Returns 1 on an overflow. */
TR_NUM_HD int tr_symc_brute_weigh(const uint64_t *wts, int32_t k, uint64_t word, uint64_t bits, uint32_t L,
                                  uint64_t *acc, uint64_t *scratch) {
    if (!bits) return 0;
    uint64_t *h = scratch, *s = scratch + L, *t = scratch + 2 * L;
    int over = 0;
    tr_symc_w_zero(s, L);
    s[0] = 1;
    for (int32_t j = 6; j < k; j++) {                       /* the high variables: bit j - 6 of the word index */
        int bit = (int)((word >> (j - 6)) & 1);
        over |= tr_symc_w_scale(s, t, L, wts[4 * (uint64_t)j + (bit ? 0 : 2)], wts[4 * (uint64_t)j + (bit ? 1 : 3)]);
    }
    while (bits) {
        int p = tr_bool_ctz(bits);
        bits &= bits - 1;
        tr_symc_w_copy(h, s, L);
        for (int32_t j = 0; j < 6 && j < k; j++) {
            int bit = (p >> j) & 1;
            over |= tr_symc_w_scale(h, t, L, wts[4 * (uint64_t)j + (bit ? 0 : 2)], wts[4 * (uint64_t)j + (bit ? 1 : 3)]);
        }
        over |= tr_symc_w_add(acc, h, L);
    }
    return over;
}
/* the domain view of the 2^k assignments (k <= TR_SYMC_MAX_BRUTE_VARS) */
TR_NUM_HD tr_bool_domain tr_symc_assignments(int32_t k) {
    tr_bool_domain d;
    d.points = (uint64_t)1 << k; d.nw = (d.points + 63) / 64; d.selected = d.points; d.kind = TR_BOOL_ASSIGNMENTS;
    d.k = k; d.selection = NULL; d.value = NULL; d.valid = NULL;
    return d;
}

/* ------------------------------------------------------------------ host-side checks and sizes (both libraries) */
/* 0, or TR_MATH_INVALID for a malformed diagram view: terminals, levels within range and increasing along every edge,
 * children below their parent, the level ranges, the order and the roots */
static inline int32_t tr_symc_dd_view_check(const tr_symc_dd_view *d) {
    if (!d || d->nnodes < 2 || !d->node || (d->nroots && !d->roots) || (d->nlevels && (!d->order || !d->first ||
        !d->count)) || d->nlevels > TR_SYMC_MAX_LEVELS) return -1;
    uint64_t at = 2;
    for (uint32_t l = d->nlevels; l-- > 0;) {
        if (d->first[l] != at || d->count[l] > d->nnodes - at) return -1;
        at += d->count[l];
        if (d->order[l] >= d->nvars) return -1;
    }
    if (at != d->nnodes) return -1;
    for (uint64_t i = 2; i < d->nnodes; i++) {
        const tr_symc_node n = d->node[i];
        if (n.level >= d->nlevels || n.lo >= i || n.hi >= i || n.lo == n.hi || n.reserved) return -1;
        if (i < d->first[n.level] || i >= d->first[n.level] + d->count[n.level]) return -1;
        if (tr_symc_level(d, n.lo) <= n.level || tr_symc_level(d, n.hi) <= n.level) return -1;
    }
    for (uint32_t r = 0; r < d->nroots; r++) if (d->roots[r] >= d->nnodes) return -1;
    return 0;
}
/* 0, or TR_MATH_INVALID for scaled weights that break the contract (each weight below 2^126) */
static inline int32_t tr_symc_scaled_check(const uint64_t *w, uint64_t n) {
    if (n && !w) return -1;
    for (uint64_t l = 0; l < n; l++) if ((w[4 * l + 1] >> 62) || (w[4 * l + 3] >> 62)) return -1;
    return 0;
}
/* the bytes tr_mgpu_symcount_levels copies to and allocates on the device */
static inline uint64_t tr_symc_levels_device_bytes(const tr_symc_dd_view *d, uint32_t L) {
    uint64_t wide = 0;
    for (uint32_t l = 0; l < d->nlevels; l++) if (d->count[l] > wide) wide = d->count[l];
    uint64_t b = tr_budget_mul(d->nnodes, sizeof(tr_symc_node));
    b = tr_budget_add(b, tr_budget_mul(tr_budget_add(d->nnodes, d->nroots), tr_budget_mul(L, 8)));   /* values, roots */
    b = tr_budget_add(b, tr_budget_mul(d->nlevels, 32));                                               /* weights */
    b = tr_budget_add(b, tr_budget_mul((uint64_t)d->nroots, 4));
    b = tr_budget_add(b, tr_budget_mul(tr_budget_add(wide, d->nroots), tr_budget_mul(2 * (uint64_t)L, 8))); /* tmp */
    return tr_budget_add(b, 16);
}
/* brute-force work items and the bytes tr_mgpu_symcount_brute copies and allocates */
static inline uint64_t tr_symc_brute_items(uint64_t nw) { return nw < TR_SYMC_DEVICE_ITEMS ? (nw ? nw : 1) : TR_SYMC_DEVICE_ITEMS; }
static inline uint64_t tr_symc_brute_device_bytes(const tr_bool_program_view *p, uint32_t L, int weighted) {
    uint64_t items = tr_symc_brute_items(((uint64_t)1 << p->k) / 64 + 1);
    uint64_t b = tr_budget_add(tr_budget_mul((uint64_t)p->nnodes, 16 + 8), tr_budget_mul((uint64_t)p->nsched, 4));
    b = tr_budget_add(b, tr_budget_mul((uint64_t)p->k, 32));
    b = tr_budget_add(b, tr_budget_mul(items, tr_budget_mul((uint64_t)p->nnodes, 16)));        /* node words */
    if (weighted) b = tr_budget_add(b, tr_budget_mul(items, tr_budget_mul(5 * (uint64_t)L, 8)));   /* sums, scratch */
    return tr_budget_add(b, tr_budget_add(tr_budget_mul(2 * (uint64_t)L, 8), 24));
}

/* ------------------------------------------------------------------ decision diagrams (libalg) */
/* a builder over nlevels levels: order[l] = the variable of level l (distinct, < nvars); budget NULL = defaults */
tr_symc_mgr *tr_symc_mgr_create(uint32_t nvars, uint32_t nlevels, const uint32_t *order, const tr_math_budget *,
                                tr_math_refusal *, int32_t *error);
/* the node of the positive literal of variable var (it must be ordered) */
int32_t tr_symc_mgr_var(tr_symc_mgr *, uint32_t var, uint32_t *node, tr_math_refusal *);
/* f op g (TR_SYMC_AND / _OR / _XOR); NOT f is tr_symc_mgr_apply(XOR, f, TR_SYMC_TRUE) */
int32_t tr_symc_mgr_apply(tr_symc_mgr *, int32_t op, uint32_t f, uint32_t g, uint32_t *out, tr_math_refusal *);
/* the nodes the manager holds, its apply steps */
int32_t tr_symc_mgr_stats(const tr_symc_mgr *, uint64_t *nodes, uint64_t *steps);
/* the immutable diagram of the roots (renumbered from the deepest level) */
tr_symc_dd *tr_symc_mgr_finish(const tr_symc_mgr *, const uint32_t *roots, uint32_t nroots, uint32_t three_valued,
                               const tr_math_budget *, tr_math_refusal *, int32_t *error);
void tr_symc_mgr_free(tr_symc_mgr *);

/* the order of a program (TR_SYMC_ORDER_NATURAL / _GIVEN with order[k] / _DFS) into out[k] */
int32_t tr_symc_program_order(const tr_bool_program_view *, int32_t kind, const uint32_t *order, uint32_t *out);
/* a BOOL-001 program compiled into its diagram over all k variables (two roots T, F when three-valued) */
tr_symc_dd *tr_symc_compile(const tr_bool_program_view *, int32_t order_kind, const uint32_t *order,
                            const tr_math_budget *, tr_math_refusal *, int32_t *error);
int32_t tr_symc_dd_get_view(const tr_symc_dd *, tr_symc_dd_view *);
int32_t tr_symc_dd_get_info(const tr_symc_dd *, tr_symc_dd_info *);
void tr_symc_dd_free(tr_symc_dd *);

/* the scaled weights of the levels (4 words per level, see Weights) and the limbs L of the count: weights[nvars] per
 * variable (NULL: model counting) */
int32_t tr_symc_dd_prepare(const tr_symc_dd_view *, const tr_symc_weight *weights, uint64_t *scaled, uint32_t *limbs);
/* the needs of a count on the CPU (TR_SYMC_NEED_CPU) or on the device (_DEVICE) */
int32_t tr_symc_dd_count_need(const tr_symc_dd *, const tr_symc_weight *, int32_t target, tr_math_need *);
/* the raw root totals on the CPU (out: nroots x L words), the reference of tr_mgpu_symcount_levels */
int32_t tr_symc_dd_levels(const tr_symc_dd_view *, const uint64_t *scaled, uint32_t L, uint64_t *roots_out,
                          const tr_math_budget *, tr_math_refusal *);
/* the answer [TRUE, FALSE, UNKNOWN, TOTAL] from raw root totals (computed by tr_symc_dd_levels or the device) */
tr_symc_answer *tr_symc_dd_adopt(const tr_symc_dd *, const tr_symc_weight *, uint32_t L, const uint64_t *roots,
                                 int32_t backend, const tr_math_budget *, tr_math_refusal *, int32_t *error);
/* prepare + levels + adopt on the CPU */
tr_symc_answer *tr_symc_dd_count(const tr_symc_dd *, const tr_symc_weight *, const tr_math_budget *, tr_math_refusal *,
                                 int32_t *error);
/* the first m assignments of root r's class in lexicographic order of the levels (masks, nvars <= 64) */
int32_t tr_symc_dd_witnesses(const tr_symc_dd *, uint32_t root, uint64_t m, uint64_t *out, uint64_t *count,
                             const tr_math_budget *, tr_math_refusal *);

/* ------------------------------------------------------------------ brute force and bounded search (libalg) */
/* the scaled weights of the k variables (4 words each) and L; weights NULL: unit weights */
int32_t tr_symc_brute_prepare(const tr_bool_program_view *, const tr_symc_weight *, uint64_t *scaled, uint32_t *limbs);
int32_t tr_symc_brute_need(const tr_bool_program_view *, const tr_symc_weight *, int32_t target, tr_math_need *);
/* the raw values on the CPU: counts[3] (TRUE, FALSE, UNKNOWN assignments), wsum[2 L] (TRUE, FALSE weights; weighted
 * only: scaled may be NULL for unit weights, then wsum is not written) */
int32_t tr_symc_brute_raw(const tr_bool_program_view *, const uint64_t *scaled, uint32_t L, uint64_t *counts,
                          uint64_t *wsum, const tr_math_budget *, tr_math_refusal *);
tr_symc_answer *tr_symc_brute_adopt(const tr_bool_program_view *, const tr_symc_weight *, uint32_t L,
                                    const uint64_t *counts, const uint64_t *wsum, int32_t backend,
                                    const tr_math_budget *, tr_math_refusal *, int32_t *error);
tr_symc_answer *tr_symc_brute(const tr_bool_program_view *, const tr_symc_weight *, const tr_math_budget *,
                              tr_math_refusal *, int32_t *error);
tr_symc_answer *tr_symc_search(const tr_bool_program_view *, int32_t order_kind, const uint32_t *order,
                               const tr_symc_weight *, uint64_t max_evaluations, const tr_math_budget *,
                               tr_math_refusal *, int32_t *error);

/* ------------------------------------------------------------------ factor systems (libalg) */
/* n variables with domains[n]; nf factors whose scopes are scope[f.scope_at ..] and tables num/den[f.table_at ..]
 * (C order); entries rationals num >= 0, den >= 1 */
tr_symc_factors *tr_symc_factors_create(uint32_t nvars, const uint32_t *domains, uint32_t nfactors,
                                        const tr_symc_factor *factors, const uint32_t *scope, const int64_t *num,
                                        const int64_t *den, uint64_t nentries, const tr_math_budget *,
                                        tr_math_refusal *, int32_t *error);
int32_t tr_symc_factors_get_info(const tr_symc_factors *, tr_symc_factors_info *);
/* the elimination order (kind GIVEN with order[n], MIN_FILL, MIN_DEGREE) into out[n] and its width */
int32_t tr_symc_factors_order(const tr_symc_factors *, int32_t kind, const uint32_t *order, uint32_t *out,
                              uint64_t *width, const tr_math_budget *, tr_math_refusal *);
int32_t tr_symc_factors_need(const tr_symc_factors *, int32_t kind, const uint32_t *order, int32_t mode,
                             uint64_t witnesses, tr_math_need *);
tr_symc_answer *tr_symc_factors_count(const tr_symc_factors *, int32_t order_kind, const uint32_t *order, int32_t mode,
                                      uint64_t witnesses, const tr_math_budget *, tr_math_refusal *, int32_t *error);
tr_symc_answer *tr_symc_factors_brute(const tr_symc_factors *, int32_t mode, uint64_t witnesses,
                                      const tr_math_budget *, tr_math_refusal *, int32_t *error);
void tr_symc_factors_free(tr_symc_factors *);

/* ------------------------------------------------------------------ answers (libalg) */
int32_t tr_symc_answer_get_info(const tr_symc_answer *, tr_symc_answer_info *);
/* value i as canonical limbs: *nlimbs / *dlimbs used (0 limbs: zero), INVALID when cap is too small */
int32_t tr_symc_answer_value(const tr_symc_answer *, uint32_t i, uint64_t *num, uint64_t *den, uint32_t cap,
                             uint32_t *nlimbs, uint32_t *dlimbs);
/* the witnesses: masks (witness_width 1) or values per variable (witness_width n), as uint64 */
int32_t tr_symc_answer_witnesses(const tr_symc_answer *, uint64_t *out, uint64_t capacity);
/* the order the answer used (levels or the elimination order), out[universe_vars] */
int32_t tr_symc_answer_order(const tr_symc_answer *, uint32_t *out, uint64_t capacity);
void tr_symc_answer_free(tr_symc_answer *);

/* ------------------------------------------------------------------ the CUDA library (math_symcount_gpu.cu)
 * tr_mgpu_symcount_levels: the root totals of tr_symc_dd_levels (views of host memory, copied): one launch per level
 *   from the deepest, one work item per node (tr_symc_node_count into its own slot), then one per root; budget:
 *   output = roots, bytes = tr_symc_levels_device_bytes. Adopt the readback with tr_symc_dd_adopt.
 * tr_mgpu_symcount_brute: the raw values of tr_symc_brute_raw: work items over slices of words, counts with one
 *   64-bit atomic each, weighted sums with a multi-word atomic add; budget: support = 2^k, bytes =
 *   tr_symc_brute_device_bytes. Adopt with tr_symc_brute_adopt. */
int tr_mgpu_symcount_levels(const tr_symc_dd_view *, const uint64_t *scaled, uint32_t L, uint64_t *roots_out,
                            const tr_math_budget *, tr_math_refusal *);
int tr_mgpu_symcount_brute(const tr_bool_program_view *, const uint64_t *scaled, uint32_t L, uint64_t *counts,
                           uint64_t *wsum, const tr_math_budget *, tr_math_refusal *);
#ifdef __cplusplus
}
#endif
#endif
