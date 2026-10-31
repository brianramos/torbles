/* ATTR-001: sparse interaction and contribution attribution of declared cooperative games, exactly (rationals), on
 * the CPU (math_attr.c) and on a CUDA device (math_attr_gpu.cu: many coalitions evaluated at once, weighted marginal
 * contributions accumulated with 128-bit atomics), with the same results bit for bit.
 * SPDX-License-Identifier: Apache-2.0
 *
 * ---- The game (declared, never inferred) --------------------------------------------------------------------------
 * A game is n players P = {0 .. n-1} and an exact integer value v(S) for every coalition S of players (bit p of a
 * coalition mask: player p present). Its baseline is the empty coalition: every attribution splits v(P) - v(empty)
 * among the players (or their sets). Two kinds:
 *   TABLE  v given at every coalition (2^n exact int128 values, n <= TR_ATTR_MAX_TABLE), declared from
 *            VALUES       given values (e.g. LATTICE-002 coalition values)
 *            CONJUNCTION  a LATTICE-001 histogram: v(S) = the points TRUE on every axis of S (zeta-up of a projection:
 *                         COUNT(*) WHERE c_i AND .. for i in S; absent conditions are dropped; v(empty) = every point)
 *            DISJUNCTION  v(S) = the points TRUE on at least one axis of S (N - zeta-down at the complement:
 *                         COUNT(*) WHERE c_i OR ..; absent conditions are FALSE; v(empty) = 0)
 *            PREDICATE    a BOOL-001 program f over k conditions on a BOOL-001 domain: v(S) = the points where
 *                         f(conditions of S at the point, the absent conditions replaced by their baseline constant
 *                         TRUE / FALSE) is TRUE under SQL three-valued logic (UNKNOWN is not TRUE)
 *   PROV   a PROV-001 evaluation plan (the cone of some output annotations): each store variable (a source tuple) is
 *          owned by one player, or is no player and fixed present (1) or absent (0) (var_player); coalition S is the
 *          0/1 valuation x = 1 iff its owner is in S (or it is fixed present); v(S) = sum_j w_j a_j(S), a_j the
 *          annotation of output j in N (COUNT: multiplicities, e.g. COUNT(*); weights give SUM(col)) or in B (BOOL:
 *          existence, 0/1), w_j int64 weights. v(empty): every player's tuples deleted, the fixed ones as declared.
 * Certified facts recorded at creation (used by the bounds, never assumed): lo <= v(S) <= hi and |v(S)| <= bound for
 * every S (TABLE: the exact extremes; PROV: N and B annotations are monotone in the valuation, so each a_j lies
 * between its values at empty and at P); monotone = +1 / -1 when v never decreases / increases as players join
 * (PROV: weights all >= 0 / <= 0; TABLE: checked on every edge of the lattice), 0 otherwise; sign = 1 when every
 * Moebius coefficient is >= 0 (PROV in N with weights >= 0: m(T) is a sum of monomial coefficients); degree = a bound
 * on |T| for m(T) != 0 (PROV in N: the largest number of player variables in a derivation; otherwise the players the
 * cone reaches; TABLE: n). A PROV game whose values at P leave 128 bits is refused (TR_MATH_OVERFLOW).
 *
 * ---- Attribution indices (exact rationals, tr_attr_q: canonical num / den in 128 bits) ---------------------------
 * From the Moebius (Harsanyi) transform m(T) = sum_{S subset of T} (-1)^{|T - S|} v(S), v(S) = sum_{T subset of S} m(T):
 *   SHAPLEY               phi_i = sum_{T contains i} m(T) / |T|
 *                               = sum_{S not containing i} |S|! (n-|S|-1)! / n! (v(S + i) - v(S))
 *   BANZHAF               beta_i = sum_{T contains i} m(T) / 2^{|T|-1} = 2^{-(n-1)} sum_{S not containing i} (v(S+i) - v(S))
 *   MOBIUS                m(S)
 *   SHAPLEY_INTERACTION   I(S) = sum_{T superset of S} m(T) / (|T| - |S| + 1)            (Grabisch 1997; I({i}) = phi_i)
 *   BANZHAF_INTERACTION   B(S) = sum_{T superset of S} m(T) / 2^{|T| - |S|}               (B({i}) = beta_i)
 *   SHAPLEY_TAYLOR (k)    ST(S) = m(S) for |S| < k; sum_{T superset of S} m(T) / C(|T|, k) for |S| = k
 *                         (Sundararajan et al. 2020: efficient, sum_{1 <= |S| <= k} ST(S) = v(P) - v(empty))
 * Identities checked natively and reported (tr_attr_identity): efficiency sum_i phi_i = v(P) - v(empty), the
 * Shapley-Taylor efficiency, and for a support built from a game sum_T m(T) = v(P), m(empty) = v(empty) against the
 * game's own evaluation. An attribution is an algebraic decomposition of v(P) - v(empty) under the declared game,
 * baseline and weighting; it has no automatic causal interpretation.
 *
 * ---- Sparse Moebius supports (tr_attr_support) --------------------------------------------------------------------
 * The nonzero m(T) as terms (T a sorted player list, a nonzero int128 coefficient), in canonical order (by |T|, then
 * colex: the largest player first, as LATTICE-002 orders its sets). Built from
 *   EXPANSION   (PROV in N) the plan's cone expanded bottom-up in the multilinear algebra over players (x^2 = x under
 *               0/1 valuations; a player's variables merge): PLUS adds term lists, TIMES multiplies (set union,
 *               coefficients multiplied), equal sets merge exactly; m = sum_j w_j (the expansion of output j). Its
 *               size is the provenance polynomial's player-set support, never the 2^n coalitions.
 *   WITNESSES   (PROV in B) per output the minimal player sets of its expansion W_1..W_w (the witnesses), then
 *               v_j = 1 - prod_i (1 - x_{W_i}) expanded (inclusion-exclusion over the witnesses: at most 2^w terms)
 *   TRANSFORM   (TABLE; or any game with n <= TR_ATTR_MAX_TABLE: its 2^n values) the LATTICE-002 fast Moebius transform
 *   TERMS       given terms (merged, zeros dropped, sorted)
 * Indices from a support cost O(sum over terms of |T|) (values) or O(sum over terms of subsets up to the order)
 * (interactions: every set S with a nonzero index is a subset of a term; the others are exactly 0 and not listed).
 *
 * ---- Enumeration (n <= TR_ATTR_MAX_ENUM: coalitions are uint64 masks) --------------------------------------------
 * values      v at given coalitions.
 * exhaustive  (n <= TR_ATTR_MAX_EXHAUSTIVE) every coalition once; per coalition S of size s with value v, player i
 *             receives the weighted marginal-contribution term c_i(S) v with c_i(S) = (s-1)! (n-s)! (i in S) or
 *             -s! (n-1-s)! (i not in S) over the common denominator n! (Shapley) and +-1 over 2^{n-1} (Banzhaf),
 *             summed exactly in 128-bit two's complement (wrapping sums are exact because the host proves the
 *             result bound first: |numerator| <= 2 n! bound < 2^127 and 2^n bound < 2^127, else TR_MATH_OVERFLOW).
 *             The CUDA kernel: work items are aligned slices of TR_ATTR_SLICE masks; a slice accumulates locally (the
 *             high players' membership is constant on a slice) and adds with one tr_atomic_add_wide per counter.
 * explore     bounded exploration with an explicit budget of evaluated coalitions (max_coalitions): whole levels
 *             (coalitions of one size) in a fixed order while they fit; v(empty) and v(P) come from the game. The
 *             result is exact, or certified bounds lo <= index <= hi (never presented as exact), with the explored
 *             states counted:
 *               MOEBIUS regime (sign certified): level n-1, then levels 1 .. g-1 (bottom up; g = the degree bound;
 *                 nothing for g = 0). With levels 0..d known, every m(T) with |T| <= d is computed exactly; U_i = v(P)
 *                 - v(P - i) - sum_{known T contains i} m(T) >= 0 is the unknown mass on the sets containing i, all of
 *                 size d+1 .. g: phi_i in [K_i + U_i / g, K_i + U_i / (d + 1)], beta_i in [K_i + U_i / 2^{g-1},
 *                 K_i + U_i / 2^d] (without level n-1: the total unknown mass R = v(P) - sum_known m bounds every U_i,
 *                 the lower bounds stay K_i). Exact when every bound is a point: d >= g (the degree bound proves the
 *                 other coefficients zero), d = g - 1 with level n-1 (the unknown sets all have g players), or U_i = 0.
 *               RANGE regime (otherwise): levels n-1, 1, n-2, 2, ... (the two ends first: Shapley weights every
 *                 level 1/n); per explored level the exact level sums; each unexplored level s widens phi_i by
 *                 +-(hi - lo) / n and beta_i by [C(n-1,s-1) lo - C(n-1,s) hi, C(n-1,s-1) hi - C(n-1,s) lo] / 2^{n-1}
 *                 (v in [lo, hi] certified); for a monotone game both indices are also clamped to [0, v(P) - v(empty)]
 *                 (increasing; the reverse when decreasing): every marginal contribution lies there. Exact when every
 *                 level is explored.
 * search      the minimum coalitions C with a declared property: EXPLAIN (C alone suffices: v(C) op threshold) or
 *             REPAIR (removing C suffices: v(P - C) op threshold), op one of = <> < <= > >=. Levels |C| = 0, 1, ...
 *             in increasing mask order; v(empty), v(P) come from the game; at most max_coalitions evaluations. The
 *             result: FOUND (the minimum size and the number of minimum solutions exact), NONE (proven: every level
 *             explored, or a monotone game whose extreme value fails), or INCOMPLETE with certified bounds: no solution
 *             of size < size_lo; a solution of size size_hi exists (C = P when it is one; -1: none known); when
 *             size_lo = size_hi, mult_lo <= the number of minimum solutions <= mult_hi.
 * explore and search share one driver (tr_attr_explore_run / tr_attr_search_run below) with the CPU's and the
 * device's batch evaluators, so both paths take the same decisions on the same exact values.
 *
 * ---- Budgets (math_budgets.h), checked before anything is allocated (TR_MATH_BUDGET with the tr_math_refusal) ----
 *   table game: variables = n, tensor = 2^n, bytes = the object; conjunction / disjunction: + iterations = k (the
 *     transform passes) and the transform scratch; predicate: tensor = 2^k coalitions x the domain's words, circuit =
 *     the scheduled nodes, bytes = the object and 16 per program node
 *   provenance game: variables = n, circuit = cone nodes, support = cone children, output = outputs, bytes = the
 *     copies (plan, weights, var_player) and the evaluation scratch
 *   values: output = coalitions, bytes = scratch; exhaustive: iterations = 2^n, output = 2n, bytes = scratch;
 *     CUDA: bytes = every device copy, the accumulators and the scratch chunk (tr_attr_view_need, NEED_DEVICE)
 *   support: circuit = cone nodes, support = the term bound (per node: sum / product of the children's bounds, at
 *     most 2^(players reached)), degree = the largest set, bytes = every memo bound + the largest raw product;
 *     witnesses: tensor = sum_j terms_j^2 (the minimization) + sum_j 2^{w_j} (checked after the witnesses are known,
 *     before the products); transform: variables = n, tensor = 2^n, iterations = n (2^n evaluations for a PROV game)
 *   indices: support = terms, output = n, bytes = the accumulators; interactions: tensor = the (set, term) pairs,
 *     output = the candidate sets, bytes = the candidates and their sort
 *   explore: iterations = max_coalitions, tensor = the Moebius inversion terms, output = 4n, bytes = the stored values
 *   search: iterations = max_coalitions, output = max_solutions, bytes = the batch buffers and the solutions
 * Representation limits (TR_MATH_UNSUPPORTED): n > TR_ATTR_MAX_PLAYERS; enumeration with n > TR_ATTR_MAX_ENUM; table
 * games with n > TR_ATTR_MAX_TABLE; exhaustive with n > TR_ATTR_MAX_EXHAUSTIVE; more than 62 minimal witnesses of one
 * output. Errors: TR_MATH_INVALID (-1), _MEMORY (-2), _OVERFLOW (-3: an exact value outside the 128-bit carriers,
 * reported, never wrapped), _BUDGET (-5), TR_MATH_DOMAIN_ERROR (-6: a projection the histogram did not record, a
 * program over another number of conditions), TR_MATH_DEVICE (-11), TR_MATH_UNSUPPORTED (-12). Objects are immutable
 * and individually owned (constructors return NULL and set *error); outputs are written only on success. */
#ifndef TORBLES_MATH_ATTR_H
#define TORBLES_MATH_ATTR_H
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "math_num.h"
#include "math_budgets.h"
#include "math_prov.h"
#include "math_lattice.h"
#ifdef __cplusplus
extern "C" {
#endif

#define TR_ATTR_ABI 1
#define TR_ATTR_MAX_ENUM 62                     /* players of a game whose coalitions are enumerated */
#define TR_ATTR_MAX_TABLE 24                    /* players of a table game (2^n values) */
#define TR_ATTR_MAX_EXHAUSTIVE 32               /* players of an exhaustive enumeration */
#define TR_ATTR_MAX_PLAYERS (UINT32_C(1) << 24) /* players of a provenance game or a support */
#define TR_ATTR_MAX_WITNESSES 62                /* minimal witnesses of one output (inclusion-exclusion) */
#define TR_ATTR_PRESENT UINT32_MAX              /* var_player: no player, always present */
#define TR_ATTR_ABSENT (UINT32_MAX - 1)         /* var_player: no player, always absent */
#define TR_ATTR_SLICE 64                        /* masks per work item of the exhaustive kernel (aligned) */
#define TR_ATTR_DEVICE_SCRATCH (UINT64_C(1) << 26)   /* bytes of device evaluation scratch per launch (at most) */
#define TR_ATTR_BATCH 4096                      /* coalitions per evaluation batch of the drivers (CPU) */
#define TR_ATTR_DEVICE_BATCH (UINT64_C(1) << 16)     /* coalitions per evaluation batch of the drivers (device) */
#define TR_ATTR_UNKNOWN UINT64_MAX              /* an unknown multiplicity bound */

enum { TR_ATTR_GAME_TABLE = 1, TR_ATTR_GAME_PROV = 2 };
enum { TR_ATTR_SRC_VALUES = 1, TR_ATTR_SRC_CONJUNCTION = 2, TR_ATTR_SRC_DISJUNCTION = 3, TR_ATTR_SRC_PREDICATE = 4,
       TR_ATTR_SRC_PROVENANCE = 5 };
enum { TR_ATTR_SHAPLEY = 1, TR_ATTR_BANZHAF = 2, TR_ATTR_MOBIUS = 3, TR_ATTR_SHAPLEY_INTERACTION = 4,
       TR_ATTR_BANZHAF_INTERACTION = 5, TR_ATTR_SHAPLEY_TAYLOR = 6 };
enum { TR_ATTR_SUPPORT_AUTO = 0, TR_ATTR_SUPPORT_EXPANSION = 1, TR_ATTR_SUPPORT_WITNESSES = 2,
       TR_ATTR_SUPPORT_TRANSFORM = 3, TR_ATTR_SUPPORT_TERMS = 4 };
enum { TR_ATTR_ID_NONE = 0, TR_ATTR_ID_EFFICIENCY = 1, TR_ATTR_ID_TAYLOR_EFFICIENCY = 2, TR_ATTR_ID_MOBIUS_SUM = 3,
       TR_ATTR_ID_MOBIUS_EMPTY = 4 };
enum { TR_ATTR_REGIME_MOEBIUS = 1, TR_ATTR_REGIME_RANGE = 2 };
enum { TR_ATTR_EXPLAIN = 1, TR_ATTR_REPAIR = 2 };
enum { TR_ATTR_EQ = 1, TR_ATTR_NE = 2, TR_ATTR_LT = 3, TR_ATTR_LE = 4, TR_ATTR_GT = 5, TR_ATTR_GE = 6 };
enum { TR_ATTR_FOUND = 1, TR_ATTR_NONE = 2, TR_ATTR_INCOMPLETE = 3 };
enum { TR_ATTR_STOP_LEVEL = 1, TR_ATTR_STOP_BUDGET = 2, TR_ATTR_STOP_EXHAUSTED = 3, TR_ATTR_STOP_MONOTONE = 4 };
enum { TR_ATTR_NEED_CPU = 0, TR_ATTR_NEED_DEVICE = 1 };
enum { TR_ATTR_OP_VALUES = 1, TR_ATTR_OP_EXHAUSTIVE = 2, TR_ATTR_OP_EXPLORE = 3, TR_ATTR_OP_SEARCH = 4 };

/* ------------------------------------------------------------------ structs crossing the ctypes boundary */
typedef struct { tr_math_i128 num, den; } tr_attr_q;     /* canonical: den > 0, gcd(|num|, den) = 1 (32 bytes) */

typedef struct {
    tr_attr_q lo, hi, expected;        /* the identity's two sides: lo <= expected <= hi (exact: lo = hi) */
    int32_t kind, holds, exact, reserved;
} tr_attr_identity;                    /* 112 bytes */

/* a game as the CPU loops and the kernels read it (pointers into the game; valid while it lives) */
typedef struct {
    int32_t kind, semiring, n, monotone;     /* TABLE / PROV; TR_PROV_COUNT / _BOOL (PROV); players; +1 / -1 / 0 */
    int32_t sign, degree, source, reserved;  /* Moebius >= 0 certified; |T| bound; TR_ATTR_SRC_* */
    uint64_t nvars;                          /* PROV: entries of var_player (the store's variables at plan time) */
    tr_prov_plan_view plan;                  /* PROV: the cone of the outputs */
    const int64_t *weights;                  /* PROV: plan.noutputs */
    const uint32_t *var_player;              /* PROV: nvars: a player, TR_ATTR_PRESENT or TR_ATTR_ABSENT */
    const tr_math_i128 *table;               /* TABLE: 2^n values, index = coalition mask */
    tr_math_i128 lo, hi, bound;              /* certified: lo <= v(S) <= hi, |v(S)| <= bound */
    tr_math_i128 empty, full;                /* v(empty), v(P) */
} tr_attr_game_view;                         /* 200 bytes */

typedef struct {
    int32_t kind, semiring, n, source;       /* as the view */
    int32_t monotone, sign, degree, projection;   /* projection: LATTICE-001's (CONJUNCTION / DISJUNCTION) */
    uint64_t baseline;                       /* PREDICATE: bit i = the constant of absent condition i (1 TRUE) */
    uint64_t nvars, present, absent;         /* PROV: variables, fixed present, fixed absent */
    uint64_t noutputs, nodes, bytes;         /* PROV: outputs, cone nodes; bytes owned */
    int32_t signed_weights, players_used;    /* PROV: a weight < 0; players owning at least one cone variable */
    tr_math_i128 lo, hi, bound, empty, full;
} tr_attr_game_info;                         /* 176 bytes */

typedef struct {
    int32_t n, method, degree, nonnegative;  /* players; TR_ATTR_SUPPORT_*; the largest |T|; every coefficient >= 0 */
    uint64_t terms, pairs, bytes;            /* terms, player entries, bytes owned */
    uint64_t candidates;                     /* raw terms before the final merge (expansion), 2^n (transform) */
    int32_t from_game, reconstructs;         /* reconstructs: sum = the game's v(P) and empty = its v(empty) */
    tr_math_i128 sum, empty;                 /* sum_T m(T) (= v(P)); m(empty) (= v(empty)) */
    tr_math_i128 game_full, game_empty;      /* the game's own v(P), v(empty) (from_game) */
} tr_attr_support_info;                      /* 120 bytes */

typedef struct {
    int32_t index, min_order, max_order, n;
    uint64_t sets, pairs, candidates, bytes; /* listed sets (nonzero indices), their entries, (set, term) pairs */
    tr_attr_identity identity;               /* SHAPLEY_TAYLOR: efficiency; MOBIUS (from order 0): the sum = v(P) */
} tr_attr_table_info;                        /* 160 bytes */

typedef struct {
    int32_t n, reserved;
    uint64_t coalitions;                     /* evaluated: 2^n */
    tr_math_i128 shapley_den, banzhaf_den;   /* n!, 2^(n-1): the common denominators of the numerators */
    tr_attr_identity efficiency;             /* sum_i phi_i = v(P) - v(empty) */
} tr_attr_exhaustive_info;                   /* 160 bytes */

typedef struct {
    int32_t regime, exact, n, degree;        /* TR_ATTR_REGIME_*; 1 when every bound is a point; the degree bound */
    int32_t bottom, levels, reserved0, reserved1;   /* levels 0..bottom all known; levels known */
    uint64_t explored, total, level_mask, budget;   /* evaluated here; 2^n (saturated); bit s: level s known; max */
    tr_math_i128 lo, hi, empty, full;        /* the certified range used (RANGE); v(empty), v(P) */
    tr_attr_identity efficiency;             /* sum lo <= v(P) - v(empty) <= sum hi */
} tr_attr_explore_info;                      /* 240 bytes */

typedef struct {
    int32_t kind, op, reserved0, reserved1;  /* TR_ATTR_EXPLAIN / _REPAIR; TR_ATTR_EQ .. _GE */
    tr_math_i128 threshold;
    uint64_t max_coalitions, max_solutions;  /* evaluations allowed; minimum solutions listed */
} tr_attr_search_query;                      /* 48 bytes */

typedef struct {
    int32_t status, reason, size_lo, size_hi;   /* TR_ATTR_FOUND / _NONE / _INCOMPLETE; TR_ATTR_STOP_*; bounds */
    uint64_t mult_lo, mult_hi;                  /* minimum solutions (when size_lo == size_hi; else 0, UNKNOWN) */
    uint64_t explored, listed;                  /* coalitions evaluated here; solutions written */
    int32_t level, complete, monotone, n;       /* level being explored at the stop; levels fully explored */
    uint64_t level_explored, level_total;
    tr_math_i128 empty, full;
} tr_attr_search_info;                          /* 112 bytes */

typedef struct tr_attr_game_s tr_attr_game;
typedef struct tr_attr_support_s tr_attr_support;
typedef struct tr_attr_table_s tr_attr_table;

/* ------------------------------------------------------------------ shared evaluation (CPU code and kernels) */
TR_NUM_HD int tr_attr_add128(__int128 a, __int128 b, __int128 *r) {
    __int128 v = (__int128)((unsigned __int128)a + (unsigned __int128)b);
    if ((a >= 0) == (b >= 0) && (v >= 0) != (a >= 0)) return 0;
    *r = v; return 1;
}
TR_NUM_HD int tr_attr_mul128(__int128 a, __int128 b, __int128 *r) {   /* checked signed product */
    if (a == 0 || b == 0) { *r = 0; return 1; }
    unsigned __int128 x = a < 0 ? (unsigned __int128)0 - (unsigned __int128)a : (unsigned __int128)a;
    unsigned __int128 y = b < 0 ? (unsigned __int128)0 - (unsigned __int128)b : (unsigned __int128)b;
    const unsigned __int128 lim = ((unsigned __int128)1 << 127) - 1;
    if (x > lim / y) {
        if ((a < 0) != (b < 0) && x * y == lim + 1 && x <= ((unsigned __int128)1 << 127) / y) {
            *r = (__int128)((unsigned __int128)0 - (lim + 1)); return 1;   /* exactly -2^127 */
        }
        return 0;
    }
    unsigned __int128 p = x * y;
    *r = (a < 0) != (b < 0) ? -(__int128)p : (__int128)p;
    return 1;
}
/* v(S) of a game (S: bit p = player p present; n <= TR_ATTR_MAX_ENUM): 1 and *value, or 0 when a node value leaves
 * 128 bits (impossible for a created game: every node is monotone and was evaluated at P). scratch: plan.nodes */
TR_NUM_HD int tr_attr_eval(const tr_attr_game_view *g, uint64_t S, unsigned __int128 *scratch, __int128 *value) {
    if (g->kind == TR_ATTR_GAME_TABLE) { *value = tr_i128_get(g->table[S]); return 1; }
    const tr_prov_plan_view *p = &g->plan;
    const unsigned __int128 top = ~(unsigned __int128)0;
    const int boolean = g->semiring == TR_PROV_BOOL;
    for (uint64_t i = 0; i < p->nodes; i++) {
        const tr_prov_node nd = p->node[i];
        unsigned __int128 x;
        if (nd.op == TR_PROV_ZERO) x = 0;
        else if (nd.op == TR_PROV_ONE) x = 1;
        else if (nd.op == TR_PROV_VAR) {
            uint32_t q = g->var_player[nd.arg];
            x = q == TR_ATTR_PRESENT ? 1 : q == TR_ATTR_ABSENT ? 0 : (unsigned __int128)((S >> q) & 1);
        } else if (nd.op == TR_PROV_PLUS) {
            x = 0;
            for (uint32_t c = 0; c < nd.n; c++) {
                unsigned __int128 y = scratch[p->child[nd.arg + c]];
                if (boolean) { if (y) { x = 1; break; } continue; }
                if (y > top - x) return 0;
                x += y;
            }
        } else {
            x = 1;
            for (uint32_t c = 0; c < nd.n && x; c++) {
                unsigned __int128 y = scratch[p->child[nd.arg + c]];
                if (boolean) { x = y ? 1 : 0; continue; }
                if (y && x > top / y) return 0;
                x *= y;
            }
        }
        scratch[i] = x;
    }
    __int128 v = 0;
    const unsigned __int128 lim = ((unsigned __int128)1 << 127) - 1;
    for (uint64_t j = 0; j < p->noutputs; j++) {
        unsigned __int128 y = scratch[p->output[j]];
        int64_t w = g->weights[j];
        if (!y || !w) continue;
        if (y > lim) return 0;
        __int128 t;
        if (!tr_attr_mul128((__int128)y, (__int128)w, &t) || !tr_attr_add128(v, t, &v)) return 0;
    }
    *value = v;
    return 1;
}
TR_NUM_HD int tr_attr_popc(uint64_t x) { return tr_bool_popc(x); }

/* ------------------------------------------------------------------ host helpers (both libraries) */
static inline uint64_t tr_attr_mask_all(int32_t n) { return n >= 64 ? ~(uint64_t)0 : (((uint64_t)1 << n) - 1); }
/* C(n, k) for n <= 62 (exact in uint64) */
static inline uint64_t tr_attr_binom(int32_t n, int32_t k) {
    if (k < 0 || k > n) return 0;
    if (k > n - k) k = n - k;
    unsigned __int128 r = 1;
    for (int32_t i = 1; i <= k; i++) r = r * (unsigned __int128)(n - k + i) / (unsigned __int128)i;
    return (uint64_t)r;
}
/* Gosper's hack: the next mask with the same popcount (increasing order = colex order) */
static inline uint64_t tr_attr_next_mask(uint64_t s) {
    uint64_t c = s & (~s + 1), r = s + c;
    return (((r ^ s) >> 2) / c) | r;
}
/* the colex rank of mask s within its level (sum over its j-th smallest player c_j of C(c_j, j + 1)) */
static inline uint64_t tr_attr_colex_rank(uint64_t s) {
    uint64_t r = 0;
    int32_t j = 0;
    while (s) {
        int32_t c = __builtin_ctzll(s);
        s &= s - 1;
        r += tr_attr_binom(c, j + 1);
        j++;
    }
    return r;
}
static inline int tr_attr_test(int32_t op, __int128 v, __int128 t) {
    switch (op) {
    case TR_ATTR_EQ: return v == t;
    case TR_ATTR_NE: return v != t;
    case TR_ATTR_LT: return v < t;
    case TR_ATTR_LE: return v <= t;
    case TR_ATTR_GT: return v > t;
    default: return v >= t;
    }
}
static inline void tr_attr_need_zero(tr_math_need *n) { for (int i = 0; i < TR_BUDGET_RESOURCES; i++) n->need[i] = 0; }
/* tr_budget_check, refusing a size past 64 bits even under an unlimited budget */
static inline int32_t tr_attr_check(const tr_math_budget *b, const tr_math_need *n, tr_math_refusal *r) {
    tr_math_budget dflt;
    if (!b) { tr_budget_default(&dflt); b = &dflt; }
    int32_t rc = tr_budget_check(b, n, r);
    if (rc) return rc;
    for (int32_t i = 0; i < TR_BUDGET_RESOURCES; i++)
        if (n->need[i] == TR_BUDGET_UNKNOWN) {
            if (r) { r->resource = i; r->refused = 1; r->need = n->need[i]; r->limit = b->limit[i]; }
            return TR_MATH_BUDGET_CODE;
        }
    return 0;
}

/* exact rationals in 128 bits (canonical results; partial sums keep a lcm denominator) */
static inline unsigned __int128 tr_attr_ugcd(unsigned __int128 a, unsigned __int128 b) {
    while (b) { unsigned __int128 t = a % b; a = b; b = t; }
    return a;
}
static inline unsigned __int128 tr_attr_uabs(__int128 x) {
    return x < 0 ? (unsigned __int128)0 - (unsigned __int128)x : (unsigned __int128)x;
}
typedef struct { __int128 num, den; } tr_attr_qacc;   /* den > 0 */
static inline tr_attr_qacc tr_attr_qzero(void) { tr_attr_qacc q; q.num = 0; q.den = 1; return q; }
static inline void tr_attr_qreduce(tr_attr_qacc *q) {
    unsigned __int128 g = tr_attr_ugcd(tr_attr_uabs(q->num), (unsigned __int128)q->den);
    if (q->num == 0) { q->den = 1; return; }
    if (g > 1) { q->num /= (__int128)g; q->den /= (__int128)g; }
}
/* q += a / d (d > 0); 1, or 0 when the exact sum leaves 128 bits */
static inline int tr_attr_qadd(tr_attr_qacc *q, __int128 a, __int128 d) {
    if (a == 0) return 1;
    for (int attempt = 0; attempt < 2; attempt++) {
        unsigned __int128 g = tr_attr_ugcd((unsigned __int128)q->den, (unsigned __int128)d);
        __int128 dq = q->den / (__int128)g, dd = d / (__int128)g, den, x, y, num;
        if (tr_attr_mul128(dq, d, &den) && tr_attr_mul128(q->num, dd, &x) && tr_attr_mul128(a, dq, &y) &&
            tr_attr_add128(x, y, &num)) {
            q->num = num; q->den = den;
            if (attempt) tr_attr_qreduce(q);
            return 1;
        }
        tr_attr_qreduce(q);                       /* retry once from the reduced partial sum */
        unsigned __int128 h = tr_attr_ugcd(tr_attr_uabs(a), (unsigned __int128)d);
        if (h > 1) { a /= (__int128)h; d /= (__int128)h; }
    }
    return 0;
}
static inline int tr_attr_qadd_q(tr_attr_qacc *q, tr_attr_qacc b) { return tr_attr_qadd(q, b.num, b.den); }
static inline tr_attr_q tr_attr_qout(tr_attr_qacc q) {
    tr_attr_qreduce(&q);
    tr_attr_q r;
    r.num = tr_i128_set(q.num); r.den = tr_i128_set(q.den);
    return r;
}
static inline int tr_attr_qcmp(tr_attr_qacc a, tr_attr_qacc b) {   /* -1, 0, 1, or 2 when undecided in 128 bits */
    __int128 l, r;
    tr_attr_qreduce(&a); tr_attr_qreduce(&b);
    if (!tr_attr_mul128(a.num, b.den, &l) || !tr_attr_mul128(b.num, a.den, &r)) {
        tr_attr_qacc d = a;                       /* the sign of a - b */
        if (!tr_attr_qadd(&d, -b.num, b.den)) return 2;
        return d.num < 0 ? -1 : d.num > 0;
    }
    return l < r ? -1 : l > r;
}
static inline tr_attr_identity tr_attr_identity_make(int32_t kind, tr_attr_qacc lo, tr_attr_qacc hi, tr_attr_qacc expected) {
    tr_attr_identity id;
    memset(&id, 0, sizeof id);
    id.kind = kind;
    int a = tr_attr_qcmp(lo, expected), b = tr_attr_qcmp(expected, hi), c = tr_attr_qcmp(lo, hi);
    id.exact = c == 0;
    id.holds = a != 2 && b != 2 && a <= 0 && b <= 0;
    id.lo = tr_attr_qout(lo); id.hi = tr_attr_qout(hi); id.expected = tr_attr_qout(expected);
    return id;
}

/* exact signed 256-bit accumulation (two's complement in two unsigned 128-bit words) for identity checks */
typedef struct { unsigned __int128 lo, hi; } tr_attr_w256;
static inline void tr_attr_w256_add(tr_attr_w256 *a, __int128 x) {
    unsigned __int128 xl = (unsigned __int128)x, xh = x < 0 ? ~(unsigned __int128)0 : 0, old = a->lo;
    a->lo += xl;
    a->hi += xh + (a->lo < old);
}
static inline tr_attr_w256 tr_attr_w256_mul(__int128 a, __int128 b) {   /* the exact product a b */
    int neg = (a < 0) != (b < 0);
    unsigned __int128 x = tr_attr_uabs(a), y = tr_attr_uabs(b);
    unsigned __int128 m = ((unsigned __int128)1 << 64) - 1;
    unsigned __int128 x0 = x & m, x1 = x >> 64, y0 = y & m, y1 = y >> 64;
    unsigned __int128 p00 = x0 * y0, p01 = x0 * y1, p10 = x1 * y0, p11 = x1 * y1;
    unsigned __int128 mid = (p00 >> 64) + (p01 & m) + (p10 & m);
    tr_attr_w256 r;
    r.lo = (p00 & m) | (mid << 64);
    r.hi = p11 + (p01 >> 64) + (p10 >> 64) + (mid >> 64);
    if (neg) { r.lo = ~r.lo + 1; r.hi = ~r.hi + (r.lo == 0); }
    return r;
}

/* the Shapley weights over n!: pos[s] = (s-1)! (n-s)! (s >= 1), neg[s] = s! (n-1-s)! (s <= n-1), *fact = n!;
 * 0 when n! >= 2^126 (n > 33) */
static inline int tr_attr_shapley_weights(int32_t n, __int128 *pos, __int128 *neg, __int128 *fact) {
    __int128 f[TR_ATTR_MAX_EXHAUSTIVE + 2];
    f[0] = 1;
    for (int32_t i = 1; i <= n && i <= TR_ATTR_MAX_EXHAUSTIVE + 1; i++)
        if (!tr_attr_mul128(f[i - 1], i, &f[i]) || f[i] >= ((__int128)1 << 125)) return 0;
    if (n > TR_ATTR_MAX_EXHAUSTIVE) return 0;
    for (int32_t s = 0; s <= n; s++) {
        pos[s] = s >= 1 ? f[s - 1] * f[n - s] : 0;
        neg[s] = s <= n - 1 ? f[s] * f[n - 1 - s] : 0;
    }
    *fact = f[n];
    return 1;
}
/* the exhaustive bound checks: 2 n! bound < 2^127 (Shapley) and 2^n bound < 2^127 (Banzhaf); 0 or TR_MATH_OVERFLOW */
static inline int32_t tr_attr_exhaustive_bound(const tr_attr_game_view *g) {
    __int128 pos[TR_ATTR_MAX_EXHAUSTIVE + 1], neg[TR_ATTR_MAX_EXHAUSTIVE + 1], fact, b = tr_i128_get(g->bound), x;
    if (!tr_attr_shapley_weights(g->n, pos, neg, &fact)) return -3;
    if (!tr_attr_mul128(fact, b, &x) || !tr_attr_mul128(x, 2, &x)) return -3;
    if (g->n >= 126 || !tr_attr_mul128(b, (__int128)1 << g->n, &x)) return -3;
    return 0;
}
/* the rationals and the efficiency identity of exhaustive numerators (Shapley over n!, Banzhaf over 2^(n-1)) */
static inline void tr_attr_exhaustive_finish(const tr_attr_game_view *g, const tr_math_i128 *shn, const tr_math_i128 *bzn,
                                             tr_attr_q *shapley, tr_attr_q *banzhaf, tr_attr_exhaustive_info *info) {
    __int128 pos[TR_ATTR_MAX_EXHAUSTIVE + 1], neg[TR_ATTR_MAX_EXHAUSTIVE + 1], fact = 1;
    tr_attr_shapley_weights(g->n, pos, neg, &fact);
    const __int128 bden = (__int128)1 << (g->n - 1);
    tr_attr_w256 sum = {0, 0};
    for (int32_t i = 0; i < g->n; i++) {
        tr_attr_qacc a, b;
        a.num = tr_i128_get(shn[i]); a.den = fact;
        b.num = tr_i128_get(bzn[i]); b.den = bden;
        if (shapley) shapley[i] = tr_attr_qout(a);
        if (banzhaf) banzhaf[i] = tr_attr_qout(b);
        tr_attr_w256_add(&sum, a.num);
    }
    if (!info) return;
    const __int128 diff = tr_i128_get(g->full) - tr_i128_get(g->empty);   /* |diff| <= 2 bound: no overflow */
    tr_attr_w256 want = tr_attr_w256_mul(fact, diff);
    memset(info, 0, sizeof *info);
    info->n = g->n;
    info->coalitions = (uint64_t)1 << g->n;
    info->shapley_den = tr_i128_set(fact); info->banzhaf_den = tr_i128_set(bden);
    tr_attr_qacc e, s;
    e.num = diff; e.den = 1;
    const int fits = sum.hi == ((sum.lo >> 127) ? ~(unsigned __int128)0 : 0);   /* the sum fits int128 */
    s.num = fits ? (__int128)sum.lo : 0; s.den = fits ? fact : 1;
    info->efficiency = tr_attr_identity_make(TR_ATTR_ID_EFFICIENCY, s, s, e);
    info->efficiency.holds = fits && sum.lo == want.lo && sum.hi == want.hi;
    info->efficiency.exact = 1;
}

/* device bytes of the game's copies (table or plan, weights, var_player) */
static inline uint64_t tr_attr_game_device_bytes(const tr_attr_game_view *g) {
    if (g->kind == TR_ATTR_GAME_TABLE) return tr_budget_mul(tr_budget_power2((uint32_t)g->n), 16);
    uint64_t b = tr_budget_add(tr_budget_mul(g->plan.nodes, 16), tr_budget_mul(tr_budget_add(g->plan.nchild, g->plan.noutputs), 4));
    b = tr_budget_add(b, tr_budget_mul(g->plan.noutputs, 8));
    return tr_budget_add(b, tr_budget_mul(g->nvars, 4));
}
/* evaluation scratch per coalition (bytes) and coalitions per device launch within TR_ATTR_DEVICE_SCRATCH */
static inline uint64_t tr_attr_scratch_row(const tr_attr_game_view *g) {
    return g->kind == TR_ATTR_GAME_PROV ? tr_budget_mul(g->plan.nodes ? g->plan.nodes : 1, 16) : 0;
}
static inline uint64_t tr_attr_device_chunk(const tr_attr_game_view *g, uint64_t items) {
    uint64_t per = tr_attr_scratch_row(g), c = per ? (per >= TR_ATTR_DEVICE_SCRATCH ? 1 : TR_ATTR_DEVICE_SCRATCH / per)
                                                    : (UINT64_C(1) << 22);
    return items < c ? (items ? items : 1) : c;
}
/* 0 when a view is well formed (and n within the enumeration limit when enumerate), else TR_MATH_INVALID or
 * TR_MATH_UNSUPPORTED */
static inline int32_t tr_attr_view_check(const tr_attr_game_view *g, int enumerate) {
    if (!g || g->n < 1) return -1;
    if (g->kind == TR_ATTR_GAME_TABLE) {
        if (g->n > TR_ATTR_MAX_TABLE) return -12;
        if (!g->table) return -1;
    } else if (g->kind == TR_ATTR_GAME_PROV) {
        if ((uint32_t)g->n > TR_ATTR_MAX_PLAYERS) return -12;
        if ((g->semiring != TR_PROV_COUNT && g->semiring != TR_PROV_BOOL) || tr_prov_plan_view_check(&g->plan) ||
            g->nvars != g->plan.nvars || (g->nvars && !g->var_player) || (g->plan.noutputs && !g->weights)) return -1;
        for (uint64_t x = 0; x < g->nvars; x++)
            if (g->var_player[x] < TR_ATTR_ABSENT && g->var_player[x] >= (uint32_t)g->n) return -1;
    } else return -1;
    if (enumerate && g->n > TR_ATTR_MAX_ENUM) return -12;
    return 0;
}

/* the needs of an enumeration op (TR_ATTR_OP_*) on a view, CPU or DEVICE (arg: coalitions for VALUES,
 * max_coalitions for EXPLORE / SEARCH); the drivers add their own host buffers (bytes) */
static inline int32_t tr_attr_view_need(const tr_attr_game_view *v, int32_t op, uint64_t arg, int32_t target,
                                        tr_math_need *need) {
    if (!v || !need || (target != TR_ATTR_NEED_CPU && target != TR_ATTR_NEED_DEVICE)) return -1;
    const int32_t n = v->n;
    const uint64_t row = tr_attr_scratch_row(v), dev = target == TR_ATTR_NEED_DEVICE ? tr_attr_game_device_bytes(v) : 0;
    tr_attr_need_zero(need);
    need->need[TR_BUDGET_VARIABLES] = (uint64_t)n;
    switch (op) {
    case TR_ATTR_OP_VALUES:
        need->need[TR_BUDGET_OUTPUT] = arg;
        need->need[TR_BUDGET_BYTES] = target == TR_ATTR_NEED_CPU
            ? tr_budget_add(row ? row : 16, tr_budget_mul(arg, 16))
            : tr_budget_add(tr_budget_add(dev, 8), tr_budget_add(tr_budget_mul(arg, 24), tr_budget_mul(tr_attr_device_chunk(v, arg), row)));
        return 0;
    case TR_ATTR_OP_EXHAUSTIVE: {
        const uint64_t total = n >= 64 ? TR_BUDGET_UNKNOWN : (uint64_t)1 << n;
        const uint64_t slices = total == TR_BUDGET_UNKNOWN ? total : (total + TR_ATTR_SLICE - 1) / TR_ATTR_SLICE;
        need->need[TR_BUDGET_ITERATIONS] = total;
        need->need[TR_BUDGET_OUTPUT] = tr_budget_mul((uint64_t)n, 2);
        need->need[TR_BUDGET_BYTES] = target == TR_ATTR_NEED_CPU
            ? tr_budget_add(row ? row : 16, tr_budget_mul((uint64_t)n, 64))
            : tr_budget_add(tr_budget_add(dev, tr_budget_mul((uint64_t)n + 1, 32)),
                            tr_budget_add(tr_budget_add(tr_budget_mul((uint64_t)n, 32), 40), tr_budget_mul(tr_attr_device_chunk(v, slices), row)));
        return 0;
    }
    case TR_ATTR_OP_EXPLORE:
    case TR_ATTR_OP_SEARCH:
        need->need[TR_BUDGET_ITERATIONS] = arg;
        need->need[TR_BUDGET_BYTES] = target == TR_ATTR_NEED_CPU ? (row ? row : 16)
            : tr_budget_add(tr_budget_add(dev, 8), tr_budget_add(tr_budget_mul(TR_ATTR_DEVICE_BATCH, 24),
                                                                 tr_budget_mul(tr_attr_device_chunk(v, TR_ATTR_DEVICE_BATCH), row)));
        return 0;
    default:
        return -1;
    }
}

/* ------------------------------------------------------------------ the shared drivers (CPU and device evaluators) */
/* a batch evaluator: values[i] = v(masks[i]) for i < count; 0 or a native error code */
typedef int32_t (*tr_attr_eval_fn)(void *ctx, const uint64_t *masks, uint64_t count, tr_math_i128 *values);

/* binomials C(a, b), a, b <= 63, as a table (63 * 64 + b) */
static inline uint64_t *tr_attr_binom_table(void) {
    uint64_t *t = (uint64_t *)calloc(64 * 64, 8);
    if (!t) return NULL;
    for (int32_t a = 0; a < 64; a++) {
        t[a * 64] = 1;
        for (int32_t b = 1; b <= a; b++) t[a * 64 + b] = t[(a - 1) * 64 + b - 1] + (b <= a - 1 ? t[(a - 1) * 64 + b] : 0);
    }
    return t;
}
static inline uint64_t tr_attr_rank_t(const uint64_t *C, uint64_t s) {   /* colex rank with the table */
    uint64_t r = 0;
    int32_t j = 0;
    while (s) {
        int32_t c = __builtin_ctzll(s);
        s &= s - 1;
        j++;
        r += c >= j ? C[c * 64 + j] : 0;
    }
    return r;
}

/* The levels an exploration evaluates, in order, while they fit max_coalitions (levels 0 and n are the game's):
 * MOEBIUS: n-1, then 1 .. min(degree - 1, n-2); RANGE: n-1, 1, n-2, 2, ... Returns the coalitions planned. */
static inline uint64_t tr_attr_explore_levels(const tr_attr_game_view *g, uint64_t max_coalitions, int32_t *order,
                                              int32_t *nlevels) {
    const int32_t n = g->n;
    int32_t cand[TR_ATTR_MAX_ENUM + 2], nc = 0, k = 0;
    if (g->sign) {
        /* levels 1 .. g-1 suffice with level n-1: the unknown sets then all have exactly g players (the bounds meet);
         * a constant game (g = 0) needs nothing */
        if (n >= 2 && g->degree > 0) cand[nc++] = n - 1;
        int32_t top = g->degree - 1 < n - 2 ? g->degree - 1 : n - 2;
        for (int32_t s = 1; s <= top; s++) cand[nc++] = s;
    } else {
        for (int32_t a = 1, b = n - 1; a <= b; a++, b--) { cand[nc++] = b; if (a != b) cand[nc++] = a; }
    }
    uint64_t used = 0;
    for (int32_t i = 0; i < nc; i++) {
        uint64_t c = tr_attr_binom(n, cand[i]);
        if (c > max_coalitions - used) break;
        used += c;
        order[k++] = cand[i];
    }
    *nlevels = k;
    return used;
}

static inline int32_t tr_attr_explore_run(const tr_attr_game_view *g, uint64_t max_coalitions, uint64_t batch,
                                          tr_attr_eval_fn eval, void *ctx, tr_attr_q *sh_lo, tr_attr_q *sh_hi,
                                          tr_attr_q *bz_lo, tr_attr_q *bz_hi, tr_attr_explore_info *info,
                                          const tr_math_budget *budget, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    int32_t rc = tr_attr_view_check(g, 1);
    if (rc) return rc;
    if (!sh_lo || !sh_hi || !bz_lo || !bz_hi || !info || !eval || batch < 1) return -1;
    const int32_t n = g->n;
    const int moebius = g->sign != 0;
    const int32_t gdeg = g->degree < 0 ? 0 : g->degree < n ? g->degree : n;
    int32_t order[TR_ATTR_MAX_ENUM + 2], nlevels = 0;
    const uint64_t planned = tr_attr_explore_levels(g, max_coalitions, order, &nlevels);
    uint64_t level_mask = (uint64_t)1 | ((uint64_t)1 << n);
    for (int32_t i = 0; i < nlevels; i++) level_mask |= (uint64_t)1 << order[i];
    const int all_known = level_mask == tr_attr_mask_all(n + 1);
    int32_t bottom = 0;
    while (bottom < n && ((level_mask >> (bottom + 1)) & 1)) bottom++;
    uint64_t sets = 0;                                  /* the sets of the truncated Moebius inversion */
    if (moebius && !all_known) for (int32_t t = 1; t <= bottom; t++) sets = tr_budget_add(sets, tr_attr_binom(n, t));
    tr_math_need need;
    tr_attr_need_zero(&need);
    need.need[TR_BUDGET_VARIABLES] = (uint64_t)n;
    need.need[TR_BUDGET_ITERATIONS] = max_coalitions;
    need.need[TR_BUDGET_TENSOR] = tr_budget_mul(sets, (uint64_t)n);
    need.need[TR_BUDGET_OUTPUT] = tr_budget_mul((uint64_t)n, 4);
    uint64_t bytes = tr_budget_add(tr_budget_mul(planned, 16), tr_budget_mul(batch, 8));
    bytes = tr_budget_add(bytes, tr_budget_add(tr_budget_mul(tr_budget_add(sets, 1), 16), 64 * 64 * 8));
    need.need[TR_BUDGET_BYTES] = tr_budget_add(bytes, tr_budget_mul((uint64_t)n, 6 * sizeof(tr_attr_qacc) + 32));
    rc = tr_attr_check(budget, &need, refusal);
    if (rc) return rc;
    const __int128 V = tr_i128_get(g->bound), lo = tr_i128_get(g->lo), hi = tr_i128_get(g->hi);
    const __int128 empty = tr_i128_get(g->empty), full = tr_i128_get(g->full);
    __int128 chk;
    if (!tr_attr_mul128(V, (__int128)1 << n, &chk)) return -3;   /* level sums and Moebius terms fit 128 bits */
    uint64_t *masks = (uint64_t *)calloc((size_t)batch, 8);
    tr_math_i128 *vals = (tr_math_i128 *)calloc((size_t)(planned ? planned : 1), 16);
    uint64_t *C = tr_attr_binom_table();
    tr_attr_qacc *acc = (tr_attr_qacc *)calloc((size_t)n * 6, sizeof(tr_attr_qacc));
    __int128 *col = (__int128 *)calloc((size_t)n, sizeof(__int128));
    if (!masks || !vals || !C || !acc || !col) { free(masks); free(vals); free(C); free(acc); free(col); return -2; }
    tr_attr_qacc *ks = acc, *kb = acc + n, *slo = acc + 2 * n, *shi = acc + 3 * n, *blo = acc + 4 * n, *bhi = acc + 5 * n;
    uint64_t start[TR_ATTR_MAX_ENUM + 2], at = 0;     /* start of level s in vals */
    for (int32_t s = 0; s <= n; s++) start[s] = UINT64_MAX;
    for (int32_t li = 0; li < nlevels && !rc; li++) {  /* evaluate the planned levels, each in increasing mask order */
        const int32_t s = order[li];
        const uint64_t total = tr_attr_binom(n, s);
        uint64_t m = tr_attr_mask_all(s), done = 0;
        start[s] = at;
        while (done < total && !rc) {
            uint64_t c = total - done < batch ? total - done : batch;
            for (uint64_t j = 0; j < c; j++) { masks[j] = m; if (done + j + 1 < total) m = tr_attr_next_mask(m); }
            rc = eval(ctx, masks, c, vals + at + done);
            done += c;
        }
        at += total;
    }
    for (int32_t i = 0; i < n; i++) { ks[i] = tr_attr_qzero(); kb[i] = tr_attr_qzero(); }
    int exact = 1;
    const __int128 nn = n, bden = (__int128)1 << (n - 1);
    if (!rc && (all_known || !moebius)) {
        /* exact level sums of every known level; each unknown level (RANGE) bounded through [lo, hi] */
        for (int32_t i = 0; i < n && !rc; i++)        /* level 0 (i absent) and level n (i present) */
            if (!tr_attr_qadd(&ks[i], -empty, nn) || !tr_attr_qadd(&ks[i], full, nn) ||
                !tr_attr_qadd(&kb[i], full - empty, bden)) rc = -3;
        __int128 w_sh = 0;
        tr_attr_qacc bw_lo = tr_attr_qzero(), bw_hi = tr_attr_qzero();
        for (int32_t s = 1; s <= n - 1 && !rc; s++) {
            const uint64_t total = tr_attr_binom(n, s);
            const __int128 c1 = (__int128)tr_attr_binom(n - 1, s - 1), c0 = (__int128)tr_attr_binom(n - 1, s);
            if (!((level_mask >> s) & 1)) {
                __int128 a, b, x;
                exact = 0;
                if (!tr_attr_add128(w_sh, hi - lo, &w_sh) ||
                    !tr_attr_mul128(c1, lo, &a) || !tr_attr_mul128(c0, hi, &b) || !tr_attr_add128(a, -b, &x) ||
                    !tr_attr_qadd(&bw_lo, x, bden) ||
                    !tr_attr_mul128(c1, hi, &a) || !tr_attr_mul128(c0, lo, &b) || !tr_attr_add128(a, -b, &x) ||
                    !tr_attr_qadd(&bw_hi, x, bden)) rc = -3;
                continue;
            }
            __int128 B = 0;
            for (int32_t i = 0; i < n; i++) col[i] = 0;
            uint64_t S = tr_attr_mask_all(s);
            for (uint64_t r = 0; r < total && !rc; r++) {
                __int128 v = tr_i128_get(vals[start[s] + r]);
                if (!tr_attr_add128(B, v, &B)) rc = -3;
                for (uint64_t b = S; b && !rc; b &= b - 1) {
                    int32_t i = __builtin_ctzll(b);
                    if (!tr_attr_add128(col[i], v, &col[i])) rc = -3;
                }
                if (r + 1 < total) S = tr_attr_next_mask(S);
            }
            for (int32_t i = 0; i < n && !rc; i++)       /* + A_i / (n C(n-1, s-1)) - (B - A_i) / (n C(n-1, s)) */
                if (!tr_attr_qadd(&ks[i], col[i], nn * c1) || !tr_attr_qadd(&ks[i], -(B - col[i]), nn * c0) ||
                    !tr_attr_qadd(&kb[i], col[i] - (B - col[i]), bden)) rc = -3;
        }
        /* a monotone game's marginal contributions lie between 0 and v(P) - v(empty): so do both indices */
        tr_attr_qacc mlo = tr_attr_qzero(), mhi = tr_attr_qzero();
        if (g->monotone > 0) mhi.num = full - empty;
        if (g->monotone < 0) mlo.num = full - empty;
        for (int32_t i = 0; i < n && !rc; i++) {
            slo[i] = ks[i]; shi[i] = ks[i]; blo[i] = kb[i]; bhi[i] = kb[i];
            if (exact) continue;
            if (!tr_attr_qadd(&slo[i], -w_sh, nn) || !tr_attr_qadd(&shi[i], w_sh, nn) ||
                !tr_attr_qadd_q(&blo[i], bw_lo) || !tr_attr_qadd_q(&bhi[i], bw_hi)) { rc = -3; break; }
            if (g->monotone) {
                if (tr_attr_qcmp(slo[i], mlo) < 0) slo[i] = mlo;
                if (tr_attr_qcmp(shi[i], mhi) > 0) shi[i] = mhi;
                if (tr_attr_qcmp(blo[i], mlo) < 0) blo[i] = mlo;
                if (tr_attr_qcmp(bhi[i], mhi) > 0) bhi[i] = mhi;
            }
        }
    } else if (!rc) {
        /* MOEBIUS: m(T) for |T| <= bottom by the truncated butterflies (pass j: f(T) -= f(T - j) for T containing j) */
        const int32_t d = bottom;
        uint64_t fs[TR_ATTR_MAX_ENUM + 2];
        fs[0] = 0;
        for (int32_t t = 1; t <= d + 1; t++) fs[t] = fs[t - 1] + tr_attr_binom(n, t - 1);
        __int128 *f = (__int128 *)calloc((size_t)(fs[d + 1] ? fs[d + 1] : 1), sizeof(__int128));
        if (!f) rc = -2;
        if (!rc) {
            f[0] = empty;
            for (int32_t t = 1; t <= d; t++)
                for (uint64_t r = 0; r < tr_attr_binom(n, t); r++) f[fs[t] + r] = tr_i128_get(vals[start[t] + r]);
            for (int32_t j = 0; j < n && !rc; j++)
                for (int32_t t = 1; t <= d && !rc; t++) {
                    const uint64_t total = tr_attr_binom(n, t);
                    uint64_t T = tr_attr_mask_all(t);
                    for (uint64_t r = 0; r < total; r++) {
                        if ((T >> j) & 1) {
                            __int128 *x = &f[fs[t] + r];
                            if (!tr_attr_add128(*x, -f[fs[t - 1] + tr_attr_rank_t(C, T & ~((uint64_t)1 << j))], x)) { rc = -3; break; }
                        }
                        if (r + 1 < total) T = tr_attr_next_mask(T);
                    }
                }
        }
        __int128 known = empty;                         /* sum of the known coefficients */
        for (int32_t i = 0; i < n; i++) col[i] = 0;     /* Q_i: the known mass on the sets containing i */
        for (int32_t t = 1; t <= d && !rc; t++) {
            const uint64_t total = tr_attr_binom(n, t);
            uint64_t T = tr_attr_mask_all(t);
            for (uint64_t r = 0; r < total && !rc; r++) {
                const __int128 mt = f[fs[t] + r];
                if (mt) {
                    if (!tr_attr_add128(known, mt, &known)) rc = -3;
                    for (uint64_t b = T; b && !rc; b &= b - 1) {
                        int32_t i = __builtin_ctzll(b);
                        if (!tr_attr_qadd(&ks[i], mt, t) || !tr_attr_qadd(&kb[i], mt, (__int128)1 << (t - 1)) ||
                            !tr_attr_add128(col[i], mt, &col[i])) rc = -3;
                    }
                }
                if (r + 1 < total) T = tr_attr_next_mask(T);
            }
        }
        const int whole = d >= gdeg;                    /* the degree bound proves every other coefficient zero */
        const int have_top = n >= 2 && ((level_mask >> (n - 1)) & 1);
        __int128 R = 0;
        if (!rc && !whole && (!tr_attr_add128(full, -known, &R) || R < 0)) rc = R < 0 ? -1 : -3;
        for (int32_t i = 0; i < n && !rc; i++) {
            slo[i] = ks[i]; shi[i] = ks[i]; blo[i] = kb[i]; bhi[i] = kb[i];
            if (whole) continue;
            __int128 U = R;
            if (have_top) {                              /* U_i = v(P) - v(P - i) - Q_i */
                const uint64_t S = tr_attr_mask_all(n) & ~((uint64_t)1 << i);
                const __int128 vi = tr_i128_get(vals[start[n - 1] + tr_attr_rank_t(C, S)]);
                if (!tr_attr_add128(full, -vi, &U) || !tr_attr_add128(U, -col[i], &U)) { rc = -3; break; }
                if (U < 0) { rc = -1; break; }           /* the sign certificate failed: never for a valid game */
            }
            if (U == 0) continue;
            exact = 0;
            if (have_top && (!tr_attr_qadd(&slo[i], U, gdeg) || !tr_attr_qadd(&blo[i], U, (__int128)1 << (gdeg - 1)))) { rc = -3; break; }
            if (!tr_attr_qadd(&shi[i], U, d + 1) || !tr_attr_qadd(&bhi[i], U, (__int128)1 << d)) { rc = -3; break; }
        }
        free(f);
    }
    if (!rc) {
        tr_attr_qacc a = tr_attr_qzero(), b = tr_attr_qzero(), e;
        e.num = full - empty; e.den = 1;
        exact = 1;                                      /* exact: every bound is a point (the bounds may meet: d + 1 = g) */
        for (int32_t i = 0; i < n; i++)
            if (tr_attr_qcmp(slo[i], shi[i]) || tr_attr_qcmp(blo[i], bhi[i])) exact = 0;
        for (int32_t i = 0; i < n && !rc; i++) if (!tr_attr_qadd_q(&a, slo[i]) || !tr_attr_qadd_q(&b, shi[i])) rc = -3;
        if (!rc) {
            for (int32_t i = 0; i < n; i++) {
                sh_lo[i] = tr_attr_qout(slo[i]); sh_hi[i] = tr_attr_qout(shi[i]);
                bz_lo[i] = tr_attr_qout(blo[i]); bz_hi[i] = tr_attr_qout(bhi[i]);
            }
            memset(info, 0, sizeof *info);
            info->regime = moebius ? TR_ATTR_REGIME_MOEBIUS : TR_ATTR_REGIME_RANGE;
            info->exact = exact; info->n = n; info->degree = gdeg;
            info->bottom = bottom;
            int32_t lv = 0;
            for (int32_t s = 0; s <= n; s++) lv += (int32_t)((level_mask >> s) & 1);
            info->levels = lv;
            info->explored = planned; info->total = (uint64_t)1 << n;
            info->level_mask = level_mask; info->budget = max_coalitions;
            info->lo = g->lo; info->hi = g->hi; info->empty = g->empty; info->full = g->full;
            info->efficiency = tr_attr_identity_make(TR_ATTR_ID_EFFICIENCY, a, b, e);
        }
    }
    free(masks); free(vals); free(C); free(acc); free(col);
    return rc;
}

static inline int32_t tr_attr_search_run(const tr_attr_game_view *g, const tr_attr_search_query *q, uint64_t batch,
                                         tr_attr_eval_fn eval, void *ctx, uint64_t *solutions,
                                         tr_attr_search_info *info, const tr_math_budget *budget,
                                         tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    int32_t rc = tr_attr_view_check(g, 1);
    if (rc) return rc;
    if (!q || !info || !eval || batch < 1 || (q->kind != TR_ATTR_EXPLAIN && q->kind != TR_ATTR_REPAIR) ||
        q->op < TR_ATTR_EQ || q->op > TR_ATTR_GE || q->reserved0 || q->reserved1 || (q->max_solutions && !solutions))
        return -1;
    const int32_t n = g->n;
    tr_math_need need;
    tr_attr_need_zero(&need);
    need.need[TR_BUDGET_VARIABLES] = (uint64_t)n;
    need.need[TR_BUDGET_ITERATIONS] = q->max_coalitions;
    need.need[TR_BUDGET_OUTPUT] = q->max_solutions;
    need.need[TR_BUDGET_BYTES] = tr_budget_add(tr_budget_mul(batch, 32), tr_budget_mul(q->max_solutions, 8));
    rc = tr_attr_check(budget, &need, refusal);
    if (rc) return rc;
    uint64_t *masks = (uint64_t *)calloc((size_t)batch, 8), *cand = (uint64_t *)calloc((size_t)batch, 8);
    tr_math_i128 *vals = (tr_math_i128 *)calloc((size_t)batch, 16);
    if (!masks || !cand || !vals) { free(masks); free(cand); free(vals); return -2; }
    const uint64_t all = tr_attr_mask_all(n);
    const __int128 th = tr_i128_get(q->threshold), empty = tr_i128_get(g->empty), full = tr_i128_get(g->full);
    const int explain = q->kind == TR_ATTR_EXPLAIN;
    tr_attr_search_info r;
    memset(&r, 0, sizeof r);
    r.n = n; r.monotone = g->monotone; r.empty = g->empty; r.full = g->full; r.size_hi = -1;
    uint64_t listed = 0, explored = 0;
    /* C = P (known from the game): explain tests v(P), repair tests v(empty) */
    const int anchor = tr_attr_test(q->op, explain ? full : empty, th);
    const int up = g->monotone > 0, down = g->monotone < 0;
    const int ge = q->op == TR_ATTR_GE || q->op == TR_ATTR_GT, le = q->op == TR_ATTR_LE || q->op == TR_ATTR_LT;
    /* monotone games: the extreme value fails, so does every coalition */
    const int none = !anchor && (explain ? ((up && ge) || (down && le)) : ((up && le) || (down && ge)));
    if (none) {
        r.status = TR_ATTR_NONE; r.reason = TR_ATTR_STOP_MONOTONE; r.size_lo = n + 1;
    } else {
        for (int32_t d = 0; d < n && !rc; d++) {
            const uint64_t total = tr_attr_binom(n, d);
            uint64_t found = 0, done = 0, m = tr_attr_mask_all(d);
            r.level = d; r.level_total = total;
            if (d == 0) {                                   /* C = empty (known): explain v(empty), repair v(P) */
                done = 1;
                if (tr_attr_test(q->op, explain ? empty : full, th)) {
                    found = 1;
                    if (listed < q->max_solutions) solutions[listed++] = 0;
                }
            }
            while (done < total && !rc) {
                const uint64_t left = q->max_coalitions - explored;
                if (!left) break;
                uint64_t c = total - done < batch ? total - done : batch;
                if (c > left) c = left;
                for (uint64_t j = 0; j < c; j++) {
                    cand[j] = m;
                    masks[j] = explain ? m : (all & ~m);
                    if (done + j + 1 < total) m = tr_attr_next_mask(m);
                }
                rc = eval(ctx, masks, c, vals);
                if (rc) break;
                explored += c; done += c;
                for (uint64_t j = 0; j < c; j++)
                    if (tr_attr_test(q->op, tr_i128_get(vals[j]), th)) {
                        found++;
                        if (listed < q->max_solutions) solutions[listed++] = cand[j];
                    }
            }
            if (rc) break;
            r.level_explored = done;
            if (done == total) {
                r.complete = d + 1;
                if (found) {                                /* every smaller level had none: the minimum, exactly */
                    r.status = TR_ATTR_FOUND; r.reason = TR_ATTR_STOP_LEVEL;
                    r.size_lo = r.size_hi = d; r.mult_lo = r.mult_hi = found;
                    break;
                }
                r.size_lo = d + 1;
                continue;
            }
            r.status = TR_ATTR_INCOMPLETE; r.reason = TR_ATTR_STOP_BUDGET; r.complete = d; r.size_lo = d;
            if (found) { r.size_hi = d; r.mult_lo = found; r.mult_hi = found + (total - done); }
            else { r.size_hi = anchor ? n : -1; r.mult_lo = 0; r.mult_hi = TR_ATTR_UNKNOWN; }
            break;
        }
        if (!rc && !r.status) {                             /* every level below n explored without a solution */
            r.level = n; r.level_total = 1; r.level_explored = 1; r.complete = n + 1;
            if (anchor) {
                r.status = TR_ATTR_FOUND; r.reason = TR_ATTR_STOP_LEVEL; r.size_lo = r.size_hi = n;
                r.mult_lo = r.mult_hi = 1;
                if (listed < q->max_solutions) solutions[listed++] = all;
            } else {
                r.status = TR_ATTR_NONE; r.reason = TR_ATTR_STOP_EXHAUSTED; r.size_lo = n + 1;
            }
        }
    }
    free(masks); free(cand); free(vals);
    if (rc) return rc;
    r.explored = explored; r.listed = listed;
    *info = r;
    return 0;
}

/* ------------------------------------------------------------------ objects (libalg) */
/* a table game of 2^n values (index = coalition mask) */
tr_attr_game *tr_attr_game_table(int32_t n, const tr_math_i128 *values, const tr_math_budget *, tr_math_refusal *,
                                 int32_t *error);
/* LATTICE-002 coalition values of a histogram's projection (TR_LATTICE_COMPLETE or _TRUE_SETS):
 * rule TR_ATTR_SRC_CONJUNCTION or _DISJUNCTION */
tr_attr_game *tr_attr_game_lattice(const tr_lattice_hist *, int32_t projection, int32_t rule, const tr_math_budget *,
                                   tr_math_refusal *, int32_t *error);
/* the predicate game of a BOOL-001 program over a domain (absent condition i takes bit i of baseline: 1 TRUE) */
tr_attr_game *tr_attr_game_predicate(const tr_bool_domain *, const tr_bool_program_view *, uint64_t baseline,
                                     const tr_math_budget *, tr_math_refusal *, int32_t *error);
/* a provenance game over an evaluation plan's view (copied): semiring TR_PROV_COUNT / _BOOL, weights[noutputs],
 * var_player[plan.nvars] */
tr_attr_game *tr_attr_game_prov(const tr_prov_plan_view *, int32_t semiring, const int64_t *weights,
                                const uint32_t *var_player, int32_t n, const tr_math_budget *, tr_math_refusal *,
                                int32_t *error);
int32_t tr_attr_game_need(int32_t kind, int32_t n, const tr_prov_plan_view *plan, tr_math_need *need);
int32_t tr_attr_game_get_info(const tr_attr_game *, tr_attr_game_info *);
int32_t tr_attr_game_get_view(const tr_attr_game *, tr_attr_game_view *);
void tr_attr_game_free(tr_attr_game *);
/* the distinct variables of a plan's cone, increasing (out NULL: count only) */
int32_t tr_attr_plan_variables(const tr_prov_plan_view *, uint32_t *out, uint64_t capacity, uint64_t *count);

/* coalition values; the exhaustive Shapley / Banzhaf values (numerators: Shapley over n!, Banzhaf over 2^(n-1)) */
int32_t tr_attr_values(const tr_attr_game *, const uint64_t *masks, uint64_t count, tr_math_i128 *out,
                       const tr_math_budget *, tr_math_refusal *);
int32_t tr_attr_exhaustive(const tr_attr_game *, tr_math_i128 *shapley_num, tr_math_i128 *banzhaf_num,
                           tr_attr_q *shapley, tr_attr_q *banzhaf, tr_attr_exhaustive_info *,
                           const tr_math_budget *, tr_math_refusal *);
/* the rationals and identity of numerators computed elsewhere (tr_mgpu_attr_exhaustive) */
int32_t tr_attr_exhaustive_adopt(const tr_attr_game *, const tr_math_i128 *shapley_num, const tr_math_i128 *banzhaf_num,
                                 tr_attr_q *shapley, tr_attr_q *banzhaf, tr_attr_exhaustive_info *);
int32_t tr_attr_explore(const tr_attr_game *, uint64_t max_coalitions, tr_attr_q *shapley_lo, tr_attr_q *shapley_hi,
                        tr_attr_q *banzhaf_lo, tr_attr_q *banzhaf_hi, tr_attr_explore_info *,
                        const tr_math_budget *, tr_math_refusal *);
int32_t tr_attr_search(const tr_attr_game *, const tr_attr_search_query *, uint64_t *solutions,
                       tr_attr_search_info *, const tr_math_budget *, tr_math_refusal *);
/* the needs of op (TR_ATTR_OP_*) on target CPU / DEVICE (explore, search: max_coalitions = arg) */
int32_t tr_attr_enum_need(const tr_attr_game *, int32_t op, uint64_t arg, int32_t target, tr_math_need *);

/* sparse Moebius supports */
tr_attr_support *tr_attr_support_build(const tr_attr_game *, int32_t method, const tr_math_budget *,
                                       tr_math_refusal *, int32_t *error);
int32_t tr_attr_support_need(const tr_attr_game *, int32_t method, tr_math_need *);
tr_attr_support *tr_attr_support_terms(int32_t n, uint64_t nterms, const uint64_t *offset, const uint32_t *players,
                                       const tr_math_i128 *coef, const tr_math_budget *, tr_math_refusal *,
                                       int32_t *error);
int32_t tr_attr_support_get_info(const tr_attr_support *, tr_attr_support_info *);
int32_t tr_attr_support_get(const tr_attr_support *, uint64_t *offset, uint32_t *players, tr_math_i128 *coef,
                            uint64_t terms_capacity, uint64_t pairs_capacity);
void tr_attr_support_free(tr_attr_support *);
/* per-player values (TR_ATTR_SHAPLEY / _BANZHAF) from a support; id: Shapley's efficiency */
int32_t tr_attr_support_values(const tr_attr_support *, int32_t index, tr_attr_q *out, tr_attr_identity *id,
                               const tr_math_budget *, tr_math_refusal *);
/* every set S with min_order <= |S| <= max_order and a nonzero index (MOBIUS, SHAPLEY_INTERACTION,
 * BANZHAF_INTERACTION, SHAPLEY_TAYLOR of order max_order), canonical order */
tr_attr_table *tr_attr_interactions(const tr_attr_support *, int32_t index, int32_t min_order, int32_t max_order,
                                    const tr_math_budget *, tr_math_refusal *, int32_t *error);
int32_t tr_attr_interactions_need(const tr_attr_support *, int32_t index, int32_t min_order, int32_t max_order,
                                  tr_math_need *);
int32_t tr_attr_table_get_info(const tr_attr_table *, tr_attr_table_info *);
int32_t tr_attr_table_get(const tr_attr_table *, uint64_t *offset, uint32_t *players, tr_attr_q *values,
                          uint64_t sets_capacity, uint64_t pairs_capacity);
void tr_attr_table_free(tr_attr_table *);

/* ------------------------------------------------------------------ the CUDA library (math_attr_gpu.cu) */
/* the same results as tr_attr_values / tr_attr_exhaustive (numerators) / tr_attr_explore / tr_attr_search from a
 * game view (host pointers, copied to the device); budgets: the CPU needs with bytes = the device copies */
int tr_mgpu_attr_values(const tr_attr_game_view *, const uint64_t *masks, uint64_t count, tr_math_i128 *out,
                        const tr_math_budget *, tr_math_refusal *);
int tr_mgpu_attr_exhaustive(const tr_attr_game_view *, tr_math_i128 *shapley_num, tr_math_i128 *banzhaf_num,
                            const tr_math_budget *, tr_math_refusal *);
int tr_mgpu_attr_explore(const tr_attr_game_view *, uint64_t max_coalitions, tr_attr_q *shapley_lo,
                         tr_attr_q *shapley_hi, tr_attr_q *banzhaf_lo, tr_attr_q *banzhaf_hi, tr_attr_explore_info *,
                         const tr_math_budget *, tr_math_refusal *);
int tr_mgpu_attr_search(const tr_attr_game_view *, const tr_attr_search_query *, uint64_t *solutions,
                        tr_attr_search_info *, const tr_math_budget *, tr_math_refusal *);
#ifdef __cplusplus
}
#endif
#endif
