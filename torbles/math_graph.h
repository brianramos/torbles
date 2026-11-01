/* GRAPH-001: declared graph layouts and reductions (a vertex dictionary, CSR / CSC / incidence forms built natively
 * from the edge rows of an FND-004 scope; degrees, frontier expansion, neighborhood aggregates and sparse
 * matrix-vector products over declared semirings) and GRAPH-002: bounded fixed points, paths and graph energies
 * (reachability and transitive closure, bounded walk counts, shortest / longest paths with cycle refusal, the exact
 * Dirichlet energy, PageRank and power iteration with declared tolerances), on the CPU (math_graph.c) and on a CUDA
 * device (math_graph_gpu.cu), with the same results bit for bit.
 * SPDX-License-Identifier: Apache-2.0
 *
 * ---- Layouts (GRAPH-001) -----------------------------------------------------------------------------------------
 * Source: an FND-004 scope (math_scope.h; one group) whose selected rows are edge rows: spec.src and spec.dst are the
 * key columns (exact logical integers of one decimal scale, else TR_MATH_DOMAIN_ERROR), spec.weight an optional
 * weight column (-1: unweighted, every weight is 1; value = logical / denominator, denominator = 10^scale x
 * spec.denominator). NULL policy: TR_GRAPH_NULLS_SKIP (a row with a NULL endpoint or weight is no edge; counted in
 * null_endpoints / null_weights) or TR_GRAPH_NULLS_FORBID (TR_MATH_DOMAIN_ERROR). An optional vertex scope adds the
 * selected present keys of its column spec.vertex (isolated vertices; same scale). NULL is never a vertex.
 * Vertex dictionary: the distinct keys in increasing order; the dense id of a key is its rank (0 .. n-1).
 * Edges: directed: tail = src, head = dst. Undirected: the pair {src, dst}, oriented tail = min id, head = max id.
 * Self-loops are kept (TR_GRAPH_LOOPS_KEEP) or dropped (TR_GRAPH_LOOPS_DROP, counted). Multi-edges are kept as
 * multiplicities (TR_GRAPH_MULTI_KEEP: one canonical edge per distinct (tail, head, w) with m = its rows) unless
 * deduplication is declared (TR_GRAPH_MULTI_DEDUP: one edge per (tail, head), m = 1, the weights of parallel rows
 * combined by spec.combine: REFUSE (different weights are TR_MATH_DOMAIN_ERROR), MIN, MAX, SUM (checked int64)).
 * Adjacency matrix A (exact integers): directed A[u][v] = sum of m over the edges u -> v. Undirected: an edge {u, v}
 * appears in both directions (A[u][v] = A[v][u] = sum of m over the edges {u, v}) and a self-loop appears ONCE
 * (A[u][u] = sum of m over the loops at u). Weighted: the same sums of m w. Stored: the entries (u, v, w, m) of A (one
 * per distinct (u, v, w): directed the canonical edges, undirected each non-loop edge twice and each loop once) as CSR
 * (out-entries of u by increasing (v, w)) and CSC (in-entries of v by increasing (u, w)); the canonical edge list
 * (the incidence form: edge e = (tail, head, w, m), sorted by (tail, head, w)) and the vertex-incidence lists (the
 * edges of each vertex, increasing id, with role tail / head / loop: the oriented incidence matrix B[head][e] = +1,
 * B[tail][e] = -1, a loop column zero). Degrees: out(u) = sum_v A[u][v], in(v) = sum_u A[u][v], total = out + in
 * (directed) or the row sum (undirected: a loop counts once; the handshake degree is total + loops), loops(u) =
 * A[u][u]; weight modes ONE (multiplicities), EDGE (sums of m w), DISTINCT (distinct neighbors).
 * Identity: (spec, the edge scope's exact FND-004 descriptor (source identity and revision, selection, column specs;
 * an array scope's data), the vertex scope's descriptor, the dictionary). tr_graph_layout_same compares it byte for
 * byte; tr_graph_layout_fresh refuses (TR_MATH_STALE) unless the given scopes are exactly the recorded ones and fresh
 * against their tables (tr_scope_fresh). Every state of a layout (vertex sets, vectors, walk tables, paths) is indexed
 * by its dense ids and meaningful only for a layout of the same identity.
 *
 * ---- Semiring products (GRAPH-001) ---------------------------------------------------------------------------------
 * TR_GRAPH_FORWARD: y_v = (+)_{u -> v} a (x) x_u (y = A^T x: v aggregates its in-entries; expansion along the edges);
 * TR_GRAPH_BACKWARD: y_u = (+)_{u -> v} a (x) x_v (y = A x). Entry values a by weight mode: EDGE (the weight w of the
 * entry, with its multiplicity m), ONE (the semiring's one per parallel edge), DISTINCT (one per distinct neighbor).
 *   COUNT (+, x) over Z: x int64, y exact int128 = sum c x_u, c = m w (EDGE), m (ONE), 1 (DISTINCT); every product and
 *     sum checked (TR_MATH_OVERFLOW; *where = the lowest failing vertex).
 *   BOOL (or, and): x, y bitsets; an entry is a one (weights, multiplicities ignored).
 *   GF2 (xor, and): y = sum (c mod 2) x_u mod 2: FIELD-001's A^T x (tr_graph_gf2_matrix exports A mod 2 packed); EDGE
 *     needs integral weights (denominator 1, else TR_MATH_DOMAIN_ERROR).
 *   MIN_PLUS / MAX_PLUS (min or max, +) over Z with +inf / -inf: y = min (max) over the entries of w' + x_u, w' = w
 *     (EDGE) or 0 (ONE, DISTINCT: the semiring's one); int64 values, every sum checked to fit int64; idempotent.
 *   FLOAT (+, x) in binary64: y = the fma accumulation, from 0 in the stored entry order, of c_e x_u with c_e =
 *     RN(RN(m w) / RN(D)) (EDGE), RN(m) (ONE), 1 (DISTINCT); err_v = RU(gamma(k + 4) RU(sum |c_e x_u|)) (k terms,
 *     gamma(k) = k u / (1 - k u), u = 2^-53) bounds |y_v - y*_v| for the exact real y*_v = sum (m w / D) x_u (EDGE).
 * Validity: x_valid (NULL: every x present): a NULL x_u gives no term; y_valid bit v = at least one term (the empty sum
 * is the semiring's zero: 0, false, +inf, -inf). Frontier expansion: next = (FORWARD / BACKWARD BOOL product of the
 * frontier) AND NOT exclude, with its popcount.
 *
 * ---- Bounded fixed points, paths and energies (GRAPH-002) ----------------------------------------------------------
 * Reach (tr_graph_reach): from a source set S, along the direction, the vertices reachable by walks of length >= 0
 * (reflexive: R*, level 0 at S) or >= 1 (R+: a source is in it only on a cycle); level = the shortest such walk. An
 * expansion step from a nonempty frontier is a round; rounds = the steps executed (the last finds nothing: the fixed
 * point); after max_rounds steps with a nonempty frontier: TR_MATH_NOT_CONVERGED, nothing returned (max_rounds = n
 * always suffices). Closure (tr_graph_closure): the reach set of each of k sources: row i of a k x n bit matrix
 * (FIELD-001 layout, stride words per row), 64 sources per lane word; rounds = the largest over the lanes.
 * Walks (tr_graph_walks): w_0 = 1_S, w_(l+1)[v] = sum_{u -> v} m w_l[u] (FORWARD; BACKWARD reversed; multiplicities
 * count, weights do not) = ((A^T)^l 1_S)[v]: rows[l][v] (optional), totals[v] = sum_{l <= L} w_l[v], lengths[l] =
 * sum_v w_l[v] and grand = sum_l lengths[l]: the coefficients of the walk generating functions truncated at z^(L+1),
 * exact unsigned integers of `limbs` 64-bit words (least significant first) under a declared bound B (limbs words):
 * every coefficient <= B, else TR_MATH_OVERFLOW with the first violation (by step l; in a step the rows (lowest
 * vertex), then the totals (lowest vertex), then the step's length total; the grand total last). L = UNBOUNDED: walks
 * of every length, finite iff no cycle is reachable from S (then L = the longest walk from S, <= n - 1, found by the
 * support pre-pass of tr_graph_walks_need); a reachable cycle makes the counts diverge: TR_MATH_NO_SOLUTION, never a
 * finite answer.
 * Paths (tr_graph_paths): MIN_PLUS (shortest) / MAX_PLUS (longest) walk weights from S (0 at S) in the entry weights
 * (int64 units of 1/D; every sum checked: TR_MATH_OVERFLOW). BELLMAN_FORD: synchronous rounds (round k: d_k[v] = the
 * best of d_(k-1)[v] and d_(k-1)[u] + w over the in-entries; only vertices with a changed in-neighbor are evaluated);
 * a change in round n (n vertices) proves a reachable cycle of negative (MIN_PLUS) or positive (MAX_PLUS) weight:
 * TR_MATH_UNBOUNDED with the cycle (v_0 -> v_1 -> ... -> v_0 along entries, from the round predecessors) and its
 * weight, never a finite answer; zero-weight cycles converge. DIJKSTRA (CPU only): MIN_PLUS with every weight >= 0
 * (else TR_MATH_DOMAIN_ERROR); AUTO: Dijkstra on the CPU when allowed, else Bellman-Ford. Results: dist, reached,
 * hops = the fewest edges among the optimal walks, pred = the smallest u with d[u] + w(u, v) = d[v] and hops[u] + 1 =
 * hops[v] (TR_GRAPH_NONE at hops 0): the same for both algorithms and both devices.
 * Energy (tr_graph_energy): E(x) = sum_e m_e w_e (x_head - x_tail)^2 over the canonical edges = x^T L x, L = the
 * Laplacian of the underlying undirected multigraph (each arc an undirected edge; a loop contributes 0): exact,
 * 320-bit two's complement (value E / (D 10^(2 s_x)) for x in units of 10^-s_x).
 * PageRank (tr_graph_pagerank, binary64): transition P[v][u] = m w / W_u along the direction (weights must be >= 0,
 * else TR_MATH_DOMAIN_ERROR; W_u = 0: u is dangling), teleport distribution p = uniform over a set T (default every
 * vertex), damping d in (0, 1]: G r = d (P r + D(r) p) + (1 - d) S(r) p, D(r) = the dangling mass, S(r) = sum r.
 * Iteration from r_0 = uniform over a start set (default T): r_(k+1)[v] = fma(d, y_v, [v in T] tele) with y_v the fma
 * accumulation of c_e r[u] over the entries (c_e = RN(RN(m w) / RN(W_u))) and tele = fma(d, D, (1 - d) S) / |T|; every
 * sum of n values (D, S, the change delta = sum |r_(k+1) - r_k|) is a fixed blocked sum (blocks of TR_GRAPH_BLOCK in
 * index order, then the block sums in order). It stops at the first k with delta <= tol and returns r_k; after
 * max_iter iterations without that: TR_MATH_NOT_CONVERGED, nothing returned. The result carries a rigorous a
 * posteriori bound (tr_graph_pagerank_certify, outward-rounded intervals over the exact rationals of G):
 * ||r - r*||_1 <= residual / (S (1 - d)) + |1 - S| with residual >= ||G r - r||_1 (bound = +inf when d = 1).
 * Power iteration (tr_graph_power): x_(k+1) = y / ||y||_1, y = A^T x_k (FORWARD; EDGE weights >= 0 as doubles), from
 * x_0 uniform over a start set; lambda = ||y||_1; the same stopping rule and refusal; y = 0 (a nilpotent part) is
 * TR_MATH_NO_SOLUTION. Certified: bound >= ||A^T x - lambda x||_1; for an undirected layout (A symmetric) some
 * eigenvalue lies within radius >= ||A x - lambda x||_2 / ||x||_2 of lambda (else radius = +inf).
 *
 * ---- Budgets (math_budgets.h), checked before anything is allocated ----------------------------------------------
 *   layout (tr_graph_layout_need): variables = columns read, support = edge rows kept, states = the vertex bound
 *     (2 x kept + vertex rows), output = the entry bound, bytes = every allocation of the build counted as if none
 *     were freed (object, dictionary and key-slot sort, tuples, both adjacency forms, edges, incidence lists, the
 *     recorded descriptors): an upper bound computed from the counts of a pass over the rows.
 *   degrees / expand / spmv / energy: output = n (energy 1), tensor = the entries visited, bytes = the CPU scratch
 *     (FLOAT: the coefficients); CUDA: bytes = the device copies (tr_graph_device_bytes).
 *   gf2 matrix: tensor = output = n x n bits.
 *   reach / closure: iterations = max_rounds, output = n (closure: k x n bits, tensor = k x n), bytes = the frontier
 *     lists and bitsets (closure: 3 lane words per vertex and batch).
 *   walks: degree = L, iterations = L, output = the coefficients returned ((L + 1) n when the rows are kept, + n +
 *     L + 2), bytes = the two step vectors (2 n (limbs + 2) words), the lists and the kept rows.
 *   paths: iterations = n (rounds), output = n, bytes = the round state (double-buffered) and the lists / heap.
 *   pagerank / power: iterations = max_iter, output = n, bytes = 3 n doubles, the coefficients and the block sums.
 * Over budget: TR_MATH_BUDGET with the refusal. Representation limits (TR_MATH_UNSUPPORTED): more than
 * TR_GRAPH_MAX_VERTICES vertices or TR_GRAPH_MAX_ENTRIES entries, 2 x kept + vertex rows past 2^32 - 1 (the key slots
 * of the build), a grouped scope, more than TR_GRAPH_MAX_LIMBS limbs.
 *
 * Errors: TR_MATH_INVALID (-1), _MEMORY (-2), _OVERFLOW (-3), _BUDGET (-5), _DOMAIN_ERROR (-6), TR_MATH_NO_SOLUTION
 * (-10: a divergent walk count, a vanished power iterate, a key that is no vertex), _DEVICE (-11), _UNSUPPORTED (-12),
 * _STALE (-13), _NOT_CONVERGED (-14), _UNBOUNDED (-16: a reachable negative / positive cycle). Layouts are immutable,
 * individually owned objects (the constructor returns NULL and sets *error); outputs are written only on success
 * (refusals leave them unspecified); read-only calls on one layout may run concurrently. */
#ifndef TORBLES_MATH_GRAPH_H
#define TORBLES_MATH_GRAPH_H
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "math_num.h"
#include "math_budgets.h"
#ifdef __cplusplus
extern "C" {
#endif

#define TR_GRAPH_ABI 1
#define TR_GRAPH_NONE UINT32_MAX                    /* no vertex (a predecessor at a root, an unreached level) */
#define TR_GRAPH_UNBOUNDED UINT32_MAX               /* walks of any length */
#define TR_GRAPH_MAX_VERTICES (UINT64_C(4294967294))    /* ids < 2^32 - 1 (TR_GRAPH_NONE stays free) */
#define TR_GRAPH_MAX_ENTRIES (UINT64_C(4294967295))     /* entries and canonical edges (edge ids fit uint32) */
#define TR_GRAPH_MAX_LIMBS 32                       /* walk counts: at most 2048 bits */
#define TR_GRAPH_BLOCK 4096                         /* the fixed block of the floating reductions */
#define TR_GRAPH_ENERGY_LIMBS 5                     /* the 320-bit energy accumulator */
#define TR_GRAPH_LANES 64                           /* closure sources per lane word */

/* local copies of the native codes (the CUDA library does not include the first slice) */
#define TR_GRAPH_E_INVALID (-1)
#define TR_GRAPH_E_MEMORY (-2)
#define TR_GRAPH_E_OVERFLOW (-3)
#define TR_GRAPH_E_BUDGET (-5)
#define TR_GRAPH_E_DOMAIN (-6)
#define TR_GRAPH_E_NO_SOLUTION (-10)
#define TR_GRAPH_E_DEVICE (-11)
#define TR_GRAPH_E_UNSUPPORTED (-12)
#define TR_GRAPH_E_STALE (-13)
#define TR_GRAPH_E_NOT_CONVERGED (-14)
#define TR_GRAPH_E_UNBOUNDED (-16)

enum { TR_GRAPH_DIRECTED = 1, TR_GRAPH_UNDIRECTED = 2 };
enum { TR_GRAPH_MULTI_KEEP = 0, TR_GRAPH_MULTI_DEDUP = 1 };
enum { TR_GRAPH_LOOPS_KEEP = 0, TR_GRAPH_LOOPS_DROP = 1 };
enum { TR_GRAPH_COMBINE_REFUSE = 0, TR_GRAPH_COMBINE_MIN = 1, TR_GRAPH_COMBINE_MAX = 2, TR_GRAPH_COMBINE_SUM = 3 };
enum { TR_GRAPH_NULLS_SKIP = 0, TR_GRAPH_NULLS_FORBID = 1 };
enum { TR_GRAPH_FORWARD = 1, TR_GRAPH_BACKWARD = 2 };
enum { TR_GRAPH_W_EDGE = 1, TR_GRAPH_W_ONE = 2, TR_GRAPH_W_DISTINCT = 3 };
enum { TR_GRAPH_COUNT = 1, TR_GRAPH_BOOL = 2, TR_GRAPH_GF2 = 3, TR_GRAPH_MIN_PLUS = 4, TR_GRAPH_MAX_PLUS = 5,
       TR_GRAPH_FLOAT = 6 };                                                      /* semirings */
enum { TR_GRAPH_DEG_OUT = 1, TR_GRAPH_DEG_IN = 2, TR_GRAPH_DEG_TOTAL = 3, TR_GRAPH_DEG_LOOPS = 4 };
enum { TR_GRAPH_AUTO = 0, TR_GRAPH_BELLMAN_FORD = 1, TR_GRAPH_DIJKSTRA = 2 };
enum { TR_GRAPH_ROLE_TAIL = 0, TR_GRAPH_ROLE_HEAD = 1, TR_GRAPH_ROLE_LOOP = 2 };   /* incidence entry = id << 2 | role */
enum { TR_GRAPH_V_NONE = 0, TR_GRAPH_V_ROW = 1, TR_GRAPH_V_TOTAL = 2, TR_GRAPH_V_LENGTH = 3, TR_GRAPH_V_GRAND = 4 };
enum { TR_GRAPH_NEED_CPU = 0, TR_GRAPH_NEED_DEVICE = 1 };

/* the declared layout (48 bytes) */
typedef struct {
    int32_t direction;        /* TR_GRAPH_DIRECTED / _UNDIRECTED */
    int32_t multi;            /* TR_GRAPH_MULTI_KEEP / _DEDUP */
    int32_t loops;            /* TR_GRAPH_LOOPS_KEEP / _DROP */
    int32_t combine;          /* TR_GRAPH_COMBINE_* (deduplicated weights) */
    int32_t nulls;            /* TR_GRAPH_NULLS_SKIP / _FORBID */
    int32_t src, dst;         /* key columns of the edge scope */
    int32_t weight;           /* weight column, or -1 (unweighted) */
    int32_t vertex;           /* key column of the vertex scope (used when one is given) */
    int32_t reserved;         /* 0 */
    int64_t denominator;      /* >= 1: weight value = logical / (10^scale x denominator) */
} tr_graph_spec;

/* a layout's description (184 bytes) */
typedef struct {
    uint64_t vertices, entries, edges;      /* n; adjacency entries per form; canonical edges */
    uint64_t rows, kept;                    /* selected edge rows; rows that are edges */
    uint64_t null_endpoints, null_weights;  /* selected rows without an edge (NULLS_SKIP) */
    uint64_t loops, dropped_loops;          /* loop rows kept / dropped */
    uint64_t merged;                        /* rows folded into another row's edge (deduplication) */
    uint64_t multiplicity;                  /* sum of the canonical edges' multiplicities */
    uint64_t vertex_rows;                   /* selected present keys of the vertex scope */
    uint64_t isolated;                      /* vertices without an entry */
    uint64_t bytes;                         /* bytes the layout retains */
    int64_t wmin, wmax;                     /* extremes of the entry weights (wmin > wmax without entries) */
    int64_t denominator;                    /* the weight unit: 10^weight_scale x spec.denominator */
    uint64_t digest[2];                     /* digest of the identity (a pre-filter; identity is exact) */
    int32_t directed, weighted, multi, loops_policy, combine, nulls, key_scale, weight_scale;
} tr_graph_info;

/* the plain-C view of a layout for CPU loops and CUDA kernels (pointers into the layout; 168 bytes): forward pulls
 * read the in-entries (CSC), backward pulls the out-entries (CSR); undirected layouts share one form */
typedef struct {
    uint64_t n, entries, edges;
    const uint64_t *in_off;        /* [n + 1] */
    const uint32_t *in_nbr;        /* [entries] the tail of each in-entry */
    const int64_t *in_w;           /* [entries] logical weights (1 when unweighted) */
    const uint64_t *in_m;          /* [entries] multiplicities >= 1 */
    const uint64_t *out_off;
    const uint32_t *out_nbr;       /* the head of each out-entry */
    const int64_t *out_w;
    const uint64_t *out_m;
    const uint32_t *e_tail, *e_head;   /* [edges] canonical edges */
    const int64_t *e_w;
    const uint64_t *e_m;
    int64_t denominator;
    int32_t directed, weighted;
    int64_t wmin, wmax;
    uint64_t max_in, max_out;      /* the most entries of one column / row */
} tr_graph_view;

/* rounds of a fixed point (32 bytes) */
typedef struct { uint64_t rounds, reached, frontier_max, work; } tr_graph_rounds;

/* a walk table's description (56 bytes) */
typedef struct {
    uint64_t lengths;               /* steps computed + 1 (L + 1; unbounded: the longest walk + 1) */
    uint64_t violation_length;      /* the first coefficient past the bound (kind != V_NONE) */
    uint64_t violation_vertex;
    int32_t status, kind;           /* 0 / TR_MATH_OVERFLOW; TR_GRAPH_V_* */
    uint32_t limbs, unbounded;
    uint64_t maxbits;               /* bits of the largest coefficient computed */
    uint64_t work;                  /* entries visited */
} tr_graph_walks_info;

/* a path problem's outcome (64 bytes) */
typedef struct {
    uint64_t rounds, reached, cycle_length;
    tr_math_i128 cycle_weight;      /* the refused cycle's weight (logical units) */
    uint64_t witness;               /* the vertex changed in round n (TR_GRAPH_NONE: converged) */
    int32_t status, algorithm;      /* 0 / TR_MATH_UNBOUNDED; the algorithm that ran */
    uint64_t work;
} tr_graph_paths_info;

/* an iteration's outcome (72 bytes) */
typedef struct {
    uint64_t iterations;
    double delta;                   /* the last ||x_k - x_(k-1)||_1 (blocked sum, computed) */
    double value;                   /* pagerank: S of the result; power: lambda */
    double bound;                   /* pagerank: >= ||r - r*||_1 ; power: >= ||A^T x - lambda x||_1 */
    double residual;                /* pagerank: >= ||G r - r||_1 ; power: >= ||A^T x - lambda x||_2 */
    double radius;                  /* power, undirected: eigenvalue inclusion radius (else +inf); pagerank: 0 */
    int32_t status, converged;
    uint64_t dangling;              /* vertices without an out-weight (pagerank) */
    uint64_t reserved;
} tr_graph_iter_info;

typedef struct tr_graph_layout_s tr_graph_layout;
struct tr_scope_s;
struct atable_s;

/* ------------------------------------------------------------------ shared helpers (CPU code, CUDA host and kernels) */
TR_NUM_HD int tr_graph_bit(const uint64_t *b, uint64_t i) { return (int)((b[i >> 6] >> (i & 63)) & 1); }
TR_NUM_HD int tr_graph_popc(uint64_t x) {
#if defined(__CUDA_ARCH__)
    return __popcll((unsigned long long)x);
#else
    return __builtin_popcountll(x);
#endif
}
TR_NUM_HD int tr_graph_ctz(uint64_t x) {   /* x != 0 */
#if defined(__CUDA_ARCH__)
    return __ffsll((long long)x) - 1;
#else
    return __builtin_ctzll(x);
#endif
}
TR_NUM_HD uint64_t tr_graph_words(uint64_t n) { return (n + 63) / 64; }

/* the pull form of a direction (the entries a vertex aggregates) and its push form (where its value goes) */
typedef struct { const uint64_t *off; const uint32_t *nbr; const int64_t *w; const uint64_t *m; } tr_graph_adj;
TR_NUM_HD tr_graph_adj tr_graph_pull(const tr_graph_view *g, int32_t dir) {
    tr_graph_adj a;
    if (dir == TR_GRAPH_FORWARD) { a.off = g->in_off; a.nbr = g->in_nbr; a.w = g->in_w; a.m = g->in_m; }
    else { a.off = g->out_off; a.nbr = g->out_nbr; a.w = g->out_w; a.m = g->out_m; }
    return a;
}
TR_NUM_HD tr_graph_adj tr_graph_push(const tr_graph_view *g, int32_t dir) {
    return tr_graph_pull(g, dir == TR_GRAPH_FORWARD ? TR_GRAPH_BACKWARD : TR_GRAPH_FORWARD);
}

/* checked 128-bit arithmetic: 1 and *r, or 0 on overflow (never wrapped) */
TR_NUM_HD int tr_graph_add128(__int128 a, __int128 b, __int128 *r) {
    __int128 v = (__int128)((unsigned __int128)a + (unsigned __int128)b);
    if ((a >= 0) == (b >= 0) && (v >= 0) != (a >= 0)) return 0;
    *r = v; return 1;
}
TR_NUM_HD unsigned __int128 tr_graph_abs128(__int128 a) {
    return a < 0 ? (unsigned __int128)0 - (unsigned __int128)a : (unsigned __int128)a;
}
/* a (int128) x b (int64), exactly or 0 */
TR_NUM_HD int tr_graph_mul128(__int128 a, int64_t b, __int128 *r) {
    if (a == 0 || b == 0) { *r = 0; return 1; }
    unsigned __int128 ua = tr_graph_abs128(a);
    uint64_t ub = tr_abs_u64(b), alo = (uint64_t)ua, ahi = (uint64_t)(ua >> 64);
    unsigned __int128 plo = (unsigned __int128)alo * ub, phi = (unsigned __int128)ahi * ub;
    unsigned __int128 mid = phi + (plo >> 64);             /* < 2^128: (2^64 - 1)^2 + 2^64 - 1 */
    const unsigned __int128 top = (unsigned __int128)1 << 63;
    int neg = (a < 0) != (b < 0);
    if (mid > top || (mid == top && (!neg || (uint64_t)plo))) return 0;
    unsigned __int128 mag = (mid << 64) | (uint64_t)plo;
    *r = neg ? (__int128)(~mag + 1) : (__int128)mag;
    return 1;
}
/* the entry coefficient of the counting semiring: m w (EDGE), m (ONE); |m w| < 2^127 always */
TR_NUM_HD __int128 tr_graph_count_coef(uint64_t m, int64_t w, int32_t wmode) {
    if (wmode == TR_GRAPH_W_EDGE) {
        unsigned __int128 p = (unsigned __int128)m * tr_abs_u64(w);
        return w < 0 ? (__int128)(~p + 1) : (__int128)p;
    }
    return (__int128)m;
}

/* the degree of v over one form: ONE sum m, EDGE sum m w, DISTINCT distinct neighbors; loops: only entries v -> v.
 * returns 1, or 0 on overflow */
TR_NUM_HD int tr_graph_row_degree(tr_graph_adj a, uint64_t v, int32_t wmode, int loops, __int128 *out) {
    __int128 s = 0;
    uint32_t prev = TR_GRAPH_NONE;
    for (uint64_t e = a.off[v]; e < a.off[v + 1]; e++) {
        uint32_t u = a.nbr[e];
        if (loops && u != v) continue;
        if (wmode == TR_GRAPH_W_DISTINCT) { if (u != prev) s += 1; prev = u; continue; }
        if (!tr_graph_add128(s, tr_graph_count_coef(a.m[e], a.w[e], wmode), &s)) return 0;
    }
    *out = s;
    return 1;
}

/* ---- one pull per vertex for every semiring (a row of A^T or A), in the stored entry order ----
 * each returns 1, or 0 on overflow; *has = at least one term */
TR_NUM_HD int tr_graph_row_count(tr_graph_adj a, uint64_t v, int32_t wmode, const int64_t *x, const uint64_t *xv,
                                 __int128 *y, int *has) {
    __int128 acc = 0;
    int any = 0;
    uint32_t prev = TR_GRAPH_NONE;
    for (uint64_t e = a.off[v]; e < a.off[v + 1]; e++) {
        uint32_t u = a.nbr[e];
        if (wmode == TR_GRAPH_W_DISTINCT) { if (u == prev) continue; prev = u; }
        if (xv && !tr_graph_bit(xv, u)) continue;
        any = 1;
        __int128 c = wmode == TR_GRAPH_W_DISTINCT ? 1 : tr_graph_count_coef(a.m[e], a.w[e], wmode), t;
        if (!tr_graph_mul128(c, x[u], &t) || !tr_graph_add128(acc, t, &acc)) return 0;
    }
    *y = acc; *has = any;
    return 1;
}
TR_NUM_HD int tr_graph_row_bool(tr_graph_adj a, uint64_t v, const uint64_t *x, const uint64_t *xv, int *has) {
    int any = 0;
    for (uint64_t e = a.off[v]; e < a.off[v + 1]; e++) {
        uint32_t u = a.nbr[e];
        if (xv && !tr_graph_bit(xv, u)) continue;
        any = 1;
        if (tr_graph_bit(x, u)) { *has = 1; return 1; }
    }
    *has = any;
    return 0;
}
TR_NUM_HD int tr_graph_row_gf2(tr_graph_adj a, uint64_t v, int32_t wmode, const uint64_t *x, const uint64_t *xv,
                               int *has) {
    int any = 0, s = 0;
    uint32_t prev = TR_GRAPH_NONE;
    for (uint64_t e = a.off[v]; e < a.off[v + 1]; e++) {
        uint32_t u = a.nbr[e];
        if (wmode == TR_GRAPH_W_DISTINCT) { if (u == prev) continue; prev = u; }
        if (xv && !tr_graph_bit(xv, u)) continue;
        any = 1;
        int c = wmode == TR_GRAPH_W_DISTINCT ? 1 : (int)(a.m[e] & 1);
        if (wmode == TR_GRAPH_W_EDGE) c &= (int)((uint64_t)a.w[e] & 1);
        s ^= c & tr_graph_bit(x, u);
    }
    *has = any;
    return s;
}
TR_NUM_HD int tr_graph_row_tropical(tr_graph_adj a, uint64_t v, int32_t wmode, int32_t maxplus, const int64_t *x,
                                    const uint64_t *xv, int64_t *y, int *has) {
    int any = 0;
    int64_t best = 0;
    for (uint64_t e = a.off[v]; e < a.off[v + 1]; e++) {
        uint32_t u = a.nbr[e];
        if (xv && !tr_graph_bit(xv, u)) continue;
        __int128 c = (__int128)x[u] + (wmode == TR_GRAPH_W_EDGE ? a.w[e] : 0);
        if (c > INT64_MAX || c < INT64_MIN) return 0;
        int64_t t = (int64_t)c;
        if (!any || (maxplus ? t > best : t < best)) best = t;
        any = 1;
    }
    *y = best; *has = any;
    return 1;
}
/* gamma(k) = k u / (1 - k u) rounded up (u = 2^-53; +inf when k u >= 1) */
TR_NUM_HD double tr_graph_gamma(uint64_t k) {
    if (k >= ((uint64_t)1 << 52)) return INFINITY;
    return tr_up((double)k / (double)(((uint64_t)1 << 53) - k));   /* k / (2^53 - k): exact operands, one division */
}
TR_NUM_HD double tr_graph_row_float(tr_graph_adj a, uint64_t v, int32_t wmode, const double *coef, const double *x,
                                    const uint64_t *xv, double *err, int *has) {
    double acc = 0.0, mag = 0.0;
    uint64_t k = 0;
    uint32_t prev = TR_GRAPH_NONE;
    for (uint64_t e = a.off[v]; e < a.off[v + 1]; e++) {
        uint32_t u = a.nbr[e];
        if (wmode == TR_GRAPH_W_DISTINCT) { if (u == prev) continue; prev = u; }
        if (xv && !tr_graph_bit(xv, u)) continue;
        double c = wmode == TR_GRAPH_W_DISTINCT ? 1.0 : coef[e];
        acc = fma(c, x[u], acc);
        mag = tr_up(fma(fabs(c), fabs(x[u]), mag));
        k++;
    }
    *has = k > 0;
    *err = k ? tr_up(tr_graph_gamma(k + 4) * mag) : 0.0;
    return acc;
}

/* ---- exact unsigned multiword integers (64-bit limbs, least significant first) ---- */
/* acc[0..na) += m x[0..nx) (na >= nx + 1); returns the carry out of acc (0 when na suffices) */
TR_NUM_HD uint64_t tr_graph_limbs_mac(uint64_t *acc, uint32_t na, const uint64_t *x, uint32_t nx, uint64_t m) {
    uint64_t carry = 0;
    uint32_t i = 0;
    for (; i < nx; i++) {
        unsigned __int128 p = (unsigned __int128)m * x[i] + acc[i] + carry;
        acc[i] = (uint64_t)p;
        carry = (uint64_t)(p >> 64);
    }
    for (; carry && i < na; i++) { uint64_t s = acc[i] + carry; carry = s < carry; acc[i] = s; }
    return carry;
}
/* acc[0..na) += x[0..nx) (na >= nx); returns the carry out */
TR_NUM_HD uint64_t tr_graph_limbs_add(uint64_t *acc, uint32_t na, const uint64_t *x, uint32_t nx) {
    uint64_t carry = 0;
    uint32_t i = 0;
    for (; i < nx; i++) {
        uint64_t s = acc[i] + x[i], c1 = s < x[i];
        uint64_t t = s + carry, c2 = t < carry;
        acc[i] = t; carry = c1 + c2;
    }
    for (; carry && i < na; i++) { uint64_t s = acc[i] + carry; carry = s < carry; acc[i] = s; }
    return carry;
}
/* a[0..na) <= b[0..nb) */
TR_NUM_HD int tr_graph_limbs_le(const uint64_t *a, uint32_t na, const uint64_t *b, uint32_t nb) {
    uint32_t n = na > nb ? na : nb;
    for (uint32_t i = n; i-- > 0;) {
        uint64_t x = i < na ? a[i] : 0, y = i < nb ? b[i] : 0;
        if (x != y) return x < y;
    }
    return 1;
}
TR_NUM_HD int tr_graph_limbs_zero(const uint64_t *a, uint32_t n) {
    for (uint32_t i = 0; i < n; i++) if (a[i]) return 0;
    return 1;
}
TR_NUM_HD int tr_graph_clz(uint64_t x) {   /* x != 0 */
#if defined(__CUDA_ARCH__)
    return __clzll((long long)x);
#else
    return __builtin_clzll(x);
#endif
}
TR_NUM_HD uint64_t tr_graph_limbs_bits(const uint64_t *a, uint32_t n) {
    for (uint32_t i = n; i-- > 0;)
        if (a[i]) return (uint64_t)i * 64 + 64 - (uint64_t)tr_graph_clz(a[i]);
    return 0;
}
/* one walk step at v: acc[0..W+2) = sum over the pulled entries of m prev[u] (prev: n x W limbs) */
TR_NUM_HD void tr_graph_row_walk(tr_graph_adj a, uint64_t v, uint32_t W, const uint64_t *prev, uint64_t *acc) {
    for (uint32_t i = 0; i < W + 2; i++) acc[i] = 0;
    for (uint64_t e = a.off[v]; e < a.off[v + 1]; e++) {
        const uint64_t *x = prev + (uint64_t)a.nbr[e] * W;
        if (!tr_graph_limbs_zero(x, W)) tr_graph_limbs_mac(acc, W + 2, x, W, a.m[e]);
    }
}

/* ---- paths: one synchronous relaxation at v (the best strict improvement of d over its pulled entries; the first
 * entry, i.e. the smallest u, among equal candidates); returns 1, or 0 on overflow ---- */
TR_NUM_HD int tr_graph_row_relax(tr_graph_adj a, uint64_t v, int32_t maxplus, const int64_t *d, const uint64_t *fin,
                                 int64_t *nd, uint32_t *np, int *improved) {
    int bfin = tr_graph_bit(fin, v), imp = 0;
    int64_t best = bfin ? d[v] : 0;
    uint32_t bp = TR_GRAPH_NONE;
    for (uint64_t e = a.off[v]; e < a.off[v + 1]; e++) {
        uint32_t u = a.nbr[e];
        if (!tr_graph_bit(fin, u)) continue;
        __int128 c = (__int128)d[u] + a.w[e];
        if (c > INT64_MAX || c < INT64_MIN) return 0;
        int64_t t = (int64_t)c;
        if (!bfin || (maxplus ? t > best : t < best)) { best = t; bfin = 1; bp = u; imp = 1; }
    }
    *nd = best; *np = bp; *improved = imp;
    return 1;
}
/* the canonical predecessor of a reached v: the smallest u with d[u] + w = d[v] and hops[u] + 1 = hops[v] */
TR_NUM_HD uint32_t tr_graph_row_pred(tr_graph_adj a, uint64_t v, const int64_t *d, const uint64_t *fin,
                                     const uint32_t *hops) {
    if (!tr_graph_bit(fin, v) || hops[v] == 0) return TR_GRAPH_NONE;
    for (uint64_t e = a.off[v]; e < a.off[v + 1]; e++) {
        uint32_t u = a.nbr[e];
        if (tr_graph_bit(fin, u) && hops[u] + 1 == hops[v] && (__int128)d[u] + a.w[e] == (__int128)d[v]) return u;
    }
    return TR_GRAPH_NONE;
}

/* ---- closure: the lanes of the in-neighbors' frontier words (64 sources per word) ---- */
TR_NUM_HD uint64_t tr_graph_row_lanes(tr_graph_adj a, uint64_t v, const uint64_t *front) {
    uint64_t s = 0;
    for (uint64_t e = a.off[v]; e < a.off[v + 1]; e++) s |= front[a.nbr[e]];
    return s;
}

/* ---- floating iterations: fixed-order sums (explicit fma, no contraction left to the compiler) ---- */
TR_NUM_HD double tr_graph_row_dot(tr_graph_adj a, uint64_t v, const double *coef, const double *x) {
    double acc = 0.0;
    for (uint64_t e = a.off[v]; e < a.off[v + 1]; e++) acc = fma(coef[e], x[a.nbr[e]], acc);
    return acc;
}
/* block b of n values: sum of x[i] (mask: only the set bits; NULL all), of |x[i] - y[i]| when y */
TR_NUM_HD double tr_graph_block(const double *x, const double *y, const uint64_t *mask, uint64_t n, uint64_t b) {
    uint64_t lo = b * TR_GRAPH_BLOCK, hi = lo + TR_GRAPH_BLOCK < n ? lo + TR_GRAPH_BLOCK : n;
    double s = 0.0;
    for (uint64_t i = lo; i < hi; i++) {
        if (mask && !tr_graph_bit(mask, i)) continue;
        s += y ? fabs(x[i] - y[i]) : x[i];
    }
    return s;
}
TR_NUM_HD double tr_graph_blocks_total(const double *part, uint64_t nb) {
    double s = 0.0;
    for (uint64_t b = 0; b < nb; b++) s += part[b];
    return s;
}
TR_NUM_HD uint64_t tr_graph_nblocks(uint64_t n) { return (n + TR_GRAPH_BLOCK - 1) / TR_GRAPH_BLOCK; }
TR_NUM_HD double tr_graph_pr_tele(double d, double dangling, double mass, double t) {
    return fma(d, dangling, (1.0 - d) * mass) / t;
}
TR_NUM_HD double tr_graph_pr_update(double d, double y, double tele, int in_t) { return fma(d, y, in_t ? tele : 0.0); }

/* ---- the exact energy: 320-bit two's complement accumulation of m w (x_h - x_t)^2 ---- */
TR_NUM_HD void tr_graph_energy_term(uint64_t m, int64_t w, int64_t xt, int64_t xh, uint64_t *t /* 5 limbs */) {
    __int128 diff = (__int128)xh - xt;                     /* |diff| < 2^64 */
    unsigned __int128 ad = tr_graph_abs128(diff);
    uint64_t dl = (uint64_t)ad;                            /* ad < 2^64 */
    unsigned __int128 sq = (unsigned __int128)dl * dl;     /* < 2^128 */
    unsigned __int128 mw = (unsigned __int128)m * tr_abs_u64(w);   /* < 2^127 */
    uint64_t a0 = (uint64_t)mw, a1 = (uint64_t)(mw >> 64), b0 = (uint64_t)sq, b1 = (uint64_t)(sq >> 64);
    unsigned __int128 p00 = (unsigned __int128)a0 * b0, p01 = (unsigned __int128)a0 * b1,
                      p10 = (unsigned __int128)a1 * b0, p11 = (unsigned __int128)a1 * b1;
    uint64_t r[5];
    r[0] = (uint64_t)p00;
    unsigned __int128 c = (p00 >> 64) + (uint64_t)p01 + (uint64_t)p10;
    r[1] = (uint64_t)c;
    c = (c >> 64) + (p01 >> 64) + (p10 >> 64) + (uint64_t)p11;
    r[2] = (uint64_t)c;
    c = (c >> 64) + (p11 >> 64);
    r[3] = (uint64_t)c;
    r[4] = 0;
    if (w < 0 && (r[0] | r[1] | r[2] | r[3])) {            /* negate in 320-bit two's complement */
        uint64_t carry = 1;
        for (int i = 0; i < 5; i++) { uint64_t s = ~r[i] + carry; carry = carry && s == 0; r[i] = s; }
    }
    for (int i = 0; i < 5; i++) t[i] = r[i];
}
TR_NUM_HD void tr_graph_energy_add(uint64_t *acc, const uint64_t *t) {    /* mod 2^320 */
    uint64_t carry = 0;
    for (int i = 0; i < 5; i++) {
        uint64_t s = acc[i] + t[i], c1 = s < t[i];
        uint64_t u = s + carry, c2 = u < carry;
        acc[i] = u; carry = c1 + c2;
    }
}

/* ------------------------------------------------------------------ host-side checks and sizes (both libraries) */
static inline int32_t tr_graph_dir_ok(int32_t dir) { return dir == TR_GRAPH_FORWARD || dir == TR_GRAPH_BACKWARD; }
static inline int32_t tr_graph_view_check(const tr_graph_view *g) {
    if (!g || (g->n && (!g->in_off || !g->out_off)) || (g->entries && (!g->in_nbr || !g->in_w || !g->in_m ||
        !g->out_nbr || !g->out_w || !g->out_m)) || (g->edges && (!g->e_tail || !g->e_head || !g->e_w || !g->e_m)) ||
        g->denominator < 1 || g->n > TR_GRAPH_MAX_VERTICES || g->entries > TR_GRAPH_MAX_ENTRIES)
        return TR_GRAPH_E_INVALID;
    return 0;
}
static inline uint64_t tr_graph_adj_bytes(uint64_t n, uint64_t entries) {
    return tr_budget_add(tr_budget_mul(tr_budget_add(n, 1), 8), tr_budget_mul(entries, 20));
}
static inline uint64_t tr_graph_bitmap(uint64_t n) { return tr_budget_bitmap_bytes(n); }
static inline void tr_graph_zero_need(tr_math_need *need) {
    for (int i = 0; i < TR_BUDGET_RESOURCES; i++) need->need[i] = 0;
}
/* the bytes of the value vectors of a semiring: x (and y) per vertex */
static inline uint64_t tr_graph_value_bytes(int32_t semiring, uint64_t n) {
    if (semiring == TR_GRAPH_BOOL || semiring == TR_GRAPH_GF2) return tr_graph_bitmap(n);
    if (semiring == TR_GRAPH_COUNT) return tr_budget_mul(n, 8);
    return tr_budget_mul(n, 8);
}
static inline uint64_t tr_graph_out_bytes(int32_t semiring, uint64_t n) {
    if (semiring == TR_GRAPH_BOOL || semiring == TR_GRAPH_GF2) return tr_graph_bitmap(n);
    if (semiring == TR_GRAPH_COUNT) return tr_budget_mul(n, 16);
    if (semiring == TR_GRAPH_FLOAT) return tr_budget_mul(n, 16);   /* y and err */
    return tr_budget_mul(n, 8);
}
/* the needs of a product (spmv), on the CPU (scratch: FLOAT coefficients) or the device (the copies) */
static inline void tr_graph_spmv_base_need(const tr_graph_view *g, int32_t semiring, int32_t target, tr_math_need *need) {
    tr_graph_zero_need(need);
    need->need[TR_BUDGET_TENSOR] = g->entries;
    need->need[TR_BUDGET_OUTPUT] = g->n;
    uint64_t bytes = semiring == TR_GRAPH_FLOAT ? tr_budget_mul(g->entries, 8) : 0;
    if (target == TR_GRAPH_NEED_DEVICE) {
        bytes = tr_budget_add(bytes, tr_graph_adj_bytes(g->n, g->entries));
        bytes = tr_budget_add(bytes, tr_budget_add(tr_graph_value_bytes(semiring, g->n), tr_graph_bitmap(g->n)));
        bytes = tr_budget_add(bytes, tr_budget_add(tr_graph_out_bytes(semiring, g->n), tr_graph_bitmap(g->n)));
        bytes = tr_budget_add(bytes, 16);
    }
    need->need[TR_BUDGET_BYTES] = bytes;
}
static inline void tr_graph_degrees_base_need(const tr_graph_view *g, int32_t target, tr_math_need *need) {
    tr_graph_zero_need(need);
    need->need[TR_BUDGET_TENSOR] = tr_budget_mul(g->entries, 2);
    need->need[TR_BUDGET_OUTPUT] = g->n;
    if (target == TR_GRAPH_NEED_DEVICE)
        need->need[TR_BUDGET_BYTES] = tr_budget_add(tr_budget_mul(tr_graph_adj_bytes(g->n, g->entries), 2),
                                                    tr_budget_add(tr_budget_mul(g->n, 16), 16));
}
static inline void tr_graph_expand_base_need(const tr_graph_view *g, int32_t target, tr_math_need *need) {
    tr_graph_zero_need(need);
    need->need[TR_BUDGET_TENSOR] = g->entries;
    need->need[TR_BUDGET_OUTPUT] = g->n;
    if (target == TR_GRAPH_NEED_DEVICE)
        need->need[TR_BUDGET_BYTES] = tr_budget_add(tr_graph_adj_bytes(g->n, g->entries),
                                                    tr_budget_add(tr_budget_mul(tr_graph_bitmap(g->n), 3), 16));
}
/* frontier algorithms: the lists (4 bytes per vertex, three of them) and marks; on the device also both forms and a
 * touch counter per vertex */
static inline uint64_t tr_graph_frontier_bytes(const tr_graph_view *g, int32_t device) {
    uint64_t bytes = tr_budget_add(tr_budget_mul(g->n, 12), tr_budget_mul(tr_graph_bitmap(g->n), 2));
    if (device) bytes = tr_budget_add(tr_budget_add(bytes, tr_budget_mul(g->n, 8)),
                                      tr_budget_mul(tr_graph_adj_bytes(g->n, g->entries), 2));
    return tr_budget_add(bytes, 64);
}
static inline void tr_graph_reach_base_need(const tr_graph_view *g, uint64_t max_rounds, int32_t target,
                                            tr_math_need *need) {
    tr_graph_zero_need(need);
    need->need[TR_BUDGET_ITERATIONS] = max_rounds;
    need->need[TR_BUDGET_OUTPUT] = g->n;
    uint64_t bytes = tr_graph_frontier_bytes(g, target == TR_GRAPH_NEED_DEVICE);
    if (target == TR_GRAPH_NEED_DEVICE)
        bytes = tr_budget_add(bytes, tr_budget_add(tr_budget_mul(g->n, 4), tr_graph_bitmap(g->n)));
    need->need[TR_BUDGET_BYTES] = bytes;
}
static inline void tr_graph_closure_base_need(const tr_graph_view *g, uint64_t k, uint64_t max_rounds, int32_t target,
                                              tr_math_need *need) {
    tr_graph_zero_need(need);
    uint64_t cells = tr_budget_mul(k, g->n);
    need->need[TR_BUDGET_TENSOR] = cells;
    need->need[TR_BUDGET_ITERATIONS] = max_rounds;
    need->need[TR_BUDGET_OUTPUT] = cells;
    uint64_t batches = (k + TR_GRAPH_LANES - 1) / TR_GRAPH_LANES;
    uint64_t bytes = tr_budget_add(tr_graph_frontier_bytes(g, target == TR_GRAPH_NEED_DEVICE),
                                   tr_budget_mul(tr_budget_mul(g->n, 8), batches > 0 ? 3 : 0));
    if (target == TR_GRAPH_NEED_DEVICE) bytes = tr_budget_add(bytes, tr_budget_mul(g->n, 8));
    need->need[TR_BUDGET_BYTES] = bytes;
}
/* walks: L steps (L + 1 rows), W limbs, rows kept or not */
static inline void tr_graph_walks_base_need(const tr_graph_view *g, uint64_t L, uint32_t W, int32_t keep_rows,
                                            int32_t target, tr_math_need *need) {
    tr_graph_zero_need(need);
    uint64_t rows = tr_budget_add(L, 1);
    need->need[TR_BUDGET_DEGREE] = L;
    need->need[TR_BUDGET_ITERATIONS] = L;
    uint64_t coefs = tr_budget_add(tr_budget_add(g->n, rows), 1);
    if (keep_rows) coefs = tr_budget_add(coefs, tr_budget_mul(rows, g->n));
    need->need[TR_BUDGET_OUTPUT] = coefs;
    uint64_t bytes = tr_budget_mul(tr_budget_mul(g->n, (uint64_t)W + 2), 16);    /* the two step vectors */
    bytes = tr_budget_add(bytes, tr_graph_frontier_bytes(g, target == TR_GRAPH_NEED_DEVICE));
    if (target == TR_GRAPH_NEED_DEVICE) {
        bytes = tr_budget_add(bytes, tr_budget_mul(tr_budget_mul(g->n, W), 8));   /* totals */
        bytes = tr_budget_add(bytes, tr_budget_mul(tr_budget_mul(rows, (uint64_t)W + 2), 8));   /* length sums */
        bytes = tr_budget_add(bytes, tr_budget_mul(W, 8));                                     /* the bound */
        if (keep_rows) bytes = tr_budget_add(bytes, tr_budget_mul(tr_budget_mul(rows, g->n), tr_budget_mul(W, 8)));
    }
    need->need[TR_BUDGET_BYTES] = bytes;
}
static inline void tr_graph_paths_base_need(const tr_graph_view *g, int32_t target, tr_math_need *need) {
    tr_graph_zero_need(need);
    need->need[TR_BUDGET_ITERATIONS] = g->n;
    need->need[TR_BUDGET_OUTPUT] = g->n;
    uint64_t bytes = tr_budget_add(tr_budget_mul(g->n, 40), tr_graph_frontier_bytes(g, target == TR_GRAPH_NEED_DEVICE));
    if (target == TR_GRAPH_NEED_DEVICE) bytes = tr_budget_add(bytes, tr_budget_mul(tr_graph_bitmap(g->n), 2));
    need->need[TR_BUDGET_BYTES] = bytes;
}
static inline void tr_graph_energy_base_need(const tr_graph_view *g, int32_t target, tr_math_need *need) {
    tr_graph_zero_need(need);
    need->need[TR_BUDGET_TENSOR] = g->edges;
    need->need[TR_BUDGET_OUTPUT] = 1;
    if (target == TR_GRAPH_NEED_DEVICE)
        need->need[TR_BUDGET_BYTES] = tr_budget_add(tr_budget_mul(g->edges, 24),
                                                    tr_budget_add(tr_budget_mul(g->n, 8), 64));
}
static inline void tr_graph_iter_base_need(const tr_graph_view *g, uint64_t max_iter, int32_t target,
                                           tr_math_need *need) {
    tr_graph_zero_need(need);
    need->need[TR_BUDGET_TENSOR] = g->entries;
    need->need[TR_BUDGET_ITERATIONS] = max_iter;
    need->need[TR_BUDGET_OUTPUT] = g->n;
    uint64_t bytes = tr_budget_add(tr_budget_mul(g->n, 24), tr_budget_mul(g->entries, 8));
    bytes = tr_budget_add(bytes, tr_budget_add(tr_budget_mul(tr_graph_nblocks(g->n), 16), tr_graph_bitmap(g->n) * 2));
    if (target == TR_GRAPH_NEED_DEVICE) bytes = tr_budget_add(bytes, tr_graph_adj_bytes(g->n, g->entries));
    need->need[TR_BUDGET_BYTES] = bytes;
}

/* the walk count limbs of a declared bound (its words without leading zeros, at least 1) */
static inline uint32_t tr_graph_bound_limbs(const uint64_t *bound, uint32_t limbs) {
    uint32_t w = limbs;
    while (w > 1 && bound[w - 1] == 0) w--;
    return w ? w : 1;
}

/* host: the negative / positive cycle through the round predecessors from a vertex changed in round n: follow pred n
 * times (that lands on the cycle), then around it. cycle[0..len) in edge order (pred[cycle[i + 1]] = cycle[i]);
 * returns the length, or 0 when no cycle is found (inconsistent input). */
static inline uint64_t tr_graph_cycle(const uint32_t *pred, uint64_t n, uint64_t witness, uint32_t *cycle,
                                      uint64_t capacity) {
    uint64_t x = witness;
    for (uint64_t i = 0; i < n; i++) {
        if (x >= n || pred[x] == TR_GRAPH_NONE) return 0;
        x = pred[x];
    }
    uint64_t len = 0, y = x;
    do {                                                   /* collect backwards, then reverse into edge order */
        if (len >= capacity || len >= n) return 0;
        cycle[len++] = (uint32_t)y;
        y = pred[y];
        if (y == TR_GRAPH_NONE) return 0;
    } while (y != x);
    for (uint64_t i = 0; i < len / 2; i++) { uint32_t t = cycle[i]; cycle[i] = cycle[len - 1 - i]; cycle[len - 1 - i] = t; }
    return len;
}
/* host: the weight of a cycle (cycle[i] -> cycle[i + 1 mod len]) using the best parallel entry (min, or max) of each
 * step in the pull form; 0 and *w, or TR_MATH_INVALID when a step is not an entry */
static inline int32_t tr_graph_cycle_weight(const tr_graph_view *g, int32_t dir, int32_t maxplus, const uint32_t *cycle,
                                            uint64_t len, __int128 *w) {
    tr_graph_adj a = tr_graph_pull(g, dir);
    __int128 s = 0;
    for (uint64_t i = 0; i < len; i++) {
        uint32_t u = cycle[i], v = cycle[(i + 1) % len];
        int found = 0;
        int64_t best = 0;
        for (uint64_t e = a.off[v]; e < a.off[v + 1]; e++)
            if (a.nbr[e] == u && (!found || (maxplus ? a.w[e] > best : a.w[e] < best))) { best = a.w[e]; found = 1; }
        if (!found) return TR_GRAPH_E_INVALID;
        s += best;
    }
    *w = s;
    return 0;
}

/* ------------------------------------------------------------------ host helpers shared by both libraries
 * (coefficients of the floating products, the walk horizon, the rigorous certificates of the iterations; host code
 * only, the same source in both libraries, every floating operation explicit: no product feeds an addition) */
static inline void *tr_graph_alloc(uint64_t bytes) {
    if (bytes > (uint64_t)SIZE_MAX) return NULL;
    return calloc(1, bytes ? (size_t)bytes : 1);
}
/* the FLOAT coefficients of the pull form of dir: RN(RN(m w) / RN(D)) (EDGE), RN(m) (ONE), 1 (DISTINCT) */
static inline void tr_graph_float_coef(const tr_graph_view *g, int32_t dir, int32_t weights, double *coef) {
    tr_graph_adj a = tr_graph_pull(g, dir);
    const double den = (double)g->denominator;
    for (uint64_t e = 0; e < g->entries; e++) {
        if (weights == TR_GRAPH_W_EDGE) coef[e] = (double)tr_graph_count_coef(a.m[e], a.w[e], TR_GRAPH_W_EDGE) / den;
        else if (weights == TR_GRAPH_W_ONE) coef[e] = (double)a.m[e];
        else coef[e] = 1.0;
    }
}
/* the out-weights W_u = sum over the push entries of u of m w (int128, checked; TR_MATH_DOMAIN_ERROR on a negative
 * weight, TR_MATH_OVERFLOW past int128) */
static inline int32_t tr_graph_out_weights(const tr_graph_view *g, int32_t dir, __int128 *W) {
    tr_graph_adj p = tr_graph_push(g, dir);
    for (uint64_t u = 0; u < g->n; u++) {
        __int128 s = 0;
        for (uint64_t e = p.off[u]; e < p.off[u + 1]; e++) {
            if (p.w[e] < 0) return TR_GRAPH_E_DOMAIN;
            if (!tr_graph_add128(s, tr_graph_count_coef(p.m[e], p.w[e], TR_GRAPH_W_EDGE), &s)) return TR_GRAPH_E_OVERFLOW;
        }
        W[u] = s;
    }
    return 0;
}
/* PageRank: coef[e] = RN(RN(m w) / RN(W_u)) of the pull entries (u the entry's tail along dir; 0 when W_u = 0) and
 * the dangling set (W_u = 0); W: scratch of n int128 */
static inline int32_t tr_graph_pagerank_coef(const tr_graph_view *g, int32_t dir, __int128 *W, double *coef,
                                             uint64_t *dangling, uint64_t *ndangling) {
    int32_t rc = tr_graph_out_weights(g, dir, W);
    if (rc) return rc;
    tr_graph_adj a = tr_graph_pull(g, dir);
    uint64_t nd = 0;
    for (uint64_t w = 0; w < tr_graph_words(g->n); w++) dangling[w] = 0;
    for (uint64_t u = 0; u < g->n; u++)
        if (W[u] == 0) { dangling[u >> 6] |= (uint64_t)1 << (u & 63); nd++; }
    for (uint64_t e = 0; e < g->entries; e++) {
        __int128 Wu = W[a.nbr[e]];
        coef[e] = Wu == 0 ? 0.0 : (double)tr_graph_count_coef(a.m[e], a.w[e], TR_GRAPH_W_EDGE) / (double)Wu;
    }
    *ndangling = nd;
    return 0;
}
/* power iteration: coef[e] = RN(RN(m w) / RN(D)) of the pull entries; TR_MATH_DOMAIN_ERROR on a negative weight */
static inline int32_t tr_graph_power_coef(const tr_graph_view *g, int32_t dir, double *coef) {
    tr_graph_adj a = tr_graph_pull(g, dir);
    for (uint64_t e = 0; e < g->entries; e++) if (a.w[e] < 0) return TR_GRAPH_E_DOMAIN;
    tr_graph_float_coef(g, dir, TR_GRAPH_W_EDGE, coef);
    return 0;
}
/* an exact int128 as an enclosing interval (a point when it is a double) */
static inline tr_math_interval tr_graph_iv_i128(__int128 x) {
    double d = (double)x;
    if (d >= 0x1p127) return tr_iv(tr_down(d), d);
    __int128 back = (__int128)d;
    if (back == x) return tr_iv(d, d);
    return back > x ? tr_iv(tr_down(d), d) : tr_iv(d, tr_up(d));
}
static inline double tr_graph_iv_mag(tr_math_interval a) {   /* max |t| over the interval */
    double l = fabs(a.lo), h = fabs(a.hi);
    return l > h ? l : h;
}
static inline uint64_t tr_graph_count_bits(const uint64_t *bits, uint64_t n) {
    uint64_t c = 0;
    for (uint64_t w = 0; w < tr_graph_words(n); w++) c += (uint64_t)tr_graph_popc(bits[w]);
    return c;
}
/* the rigorous a posteriori bounds of a PageRank vector r (see the header comment): info->residual >= ||G r - r||_1,
 * info->bound >= ||r - r*||_1 (+inf when d = 1 or sum r is not certified positive), info->value = sum r (computed);
 * W: scratch of n int128 */
static inline int32_t tr_graph_pagerank_certify(const tr_graph_view *g, int32_t dir, double d, const uint64_t *teleport,
                                                const double *r, __int128 *W, tr_graph_iter_info *info) {
    const uint64_t n = g->n;
    int32_t rc = tr_graph_out_weights(g, dir, W);
    if (rc) return rc;
    uint64_t t = teleport ? tr_graph_count_bits(teleport, n) : n;
    if (!t) return TR_GRAPH_E_INVALID;
    tr_math_interval s = tr_iv(0, 0), D = tr_iv(0, 0);
    for (uint64_t v = 0; v < n; v++) {
        s = tr_iv_add(s, tr_iv(r[v], r[v]));
        if (W[v] == 0) D = tr_iv_add(D, tr_iv(r[v], r[v]));
    }
    tr_math_interval one = tr_iv(1.0, 1.0), dd = tr_iv(d, d), omd = tr_iv_sub(one, dd), tt = tr_graph_iv_i128(t);
    tr_math_interval tele = tr_iv_div(tr_iv_add(tr_iv_mul(dd, D), tr_iv_mul(omd, s)), tt);
    tr_graph_adj a = tr_graph_pull(g, dir);
    double res = 0.0;
    for (uint64_t v = 0; v < n; v++) {
        tr_math_interval acc = tr_iv(0, 0);
        for (uint64_t e = a.off[v]; e < a.off[v + 1]; e++) {
            uint32_t u = a.nbr[e];
            if (W[u] == 0) continue;
            tr_math_interval pe = tr_iv_div(tr_graph_iv_i128(tr_graph_count_coef(a.m[e], a.w[e], TR_GRAPH_W_EDGE)),
                                            tr_graph_iv_i128(W[u]));
            acc = tr_iv_add(acc, tr_iv_mul(pe, tr_iv(r[u], r[u])));
        }
        tr_math_interval gv = tr_iv_mul(dd, acc);
        if (!teleport || tr_graph_bit(teleport, v)) gv = tr_iv_add(gv, tele);
        res = tr_up(res + tr_graph_iv_mag(tr_iv_sub(gv, tr_iv(r[v], r[v]))));
    }
    info->residual = res;
    if (omd.lo <= 0 || s.lo <= 0) info->bound = INFINITY;
    else {
        double first = tr_up(res / tr_down(s.lo * omd.lo));
        double off = tr_graph_iv_mag(tr_iv_sub(s, one));
        info->bound = tr_up(first + off);
    }
    info->radius = 0.0;
    return 0;
}
/* the rigorous residuals of a power iterate x with the estimate lambda: info->bound >= ||A' x - lambda x||_1,
 * info->residual >= ||A' x - lambda x||_2 (A' = A^T forward, A backward, exact weights m w / D), info->radius >=
 * residual / ||x||_2 for an undirected layout (an eigenvalue of the symmetric A lies within it), else +inf */
static inline int32_t tr_graph_power_certify(const tr_graph_view *g, int32_t dir, const double *x, double lambda,
                                             tr_graph_iter_info *info) {
    tr_graph_adj a = tr_graph_pull(g, dir);
    tr_math_interval den = tr_graph_iv_i128(g->denominator), lam = tr_iv(lambda, lambda), norm2 = tr_iv(0, 0);
    double r1 = 0.0, r2 = 0.0;
    for (uint64_t v = 0; v < g->n; v++) {
        tr_math_interval acc = tr_iv(0, 0);
        for (uint64_t e = a.off[v]; e < a.off[v + 1]; e++) {
            tr_math_interval ae = tr_iv_div(tr_graph_iv_i128(tr_graph_count_coef(a.m[e], a.w[e], TR_GRAPH_W_EDGE)), den);
            acc = tr_iv_add(acc, tr_iv_mul(ae, tr_iv(x[a.nbr[e]], x[a.nbr[e]])));
        }
        tr_math_interval xv = tr_iv(x[v], x[v]);
        double m = tr_graph_iv_mag(tr_iv_sub(acc, tr_iv_mul(lam, xv)));
        r1 = tr_up(r1 + m);
        r2 = tr_up(r2 + tr_up(m * m));
        norm2 = tr_iv_add(norm2, tr_iv_mul(xv, xv));
    }
    info->bound = r1;
    info->residual = tr_up(sqrt(r2));
    double nlo = norm2.lo > 0 ? tr_down(sqrt(norm2.lo)) : 0.0;
    info->radius = (!g->directed && nlo > 0) ? tr_up(info->residual / nlo) : INFINITY;
    return 0;
}
/* The walk horizon of a source set along dir: *L = the longest walk from S when no cycle is reachable from S (the
 * reachable part is a DAG: Kahn's order, longest paths), else TR_MATH_NO_SOLUTION (the walk counts diverge). The
 * scratch (12 n bytes and a bitmap: tr_graph_horizon_bytes) is the caller's budget. */
static inline uint64_t tr_graph_horizon_bytes(uint64_t n) {
    return tr_budget_add(tr_budget_mul(n, 12), tr_budget_bitmap_bytes(n));
}
static inline int32_t tr_graph_horizon(const tr_graph_view *g, int32_t dir, const uint64_t *sources, uint64_t *L,
                                       uint64_t *work) {
    const uint64_t n = g->n;
    uint64_t *R = (uint64_t *)tr_graph_alloc(tr_budget_bitmap_bytes(n));
    uint32_t *Q = (uint32_t *)tr_graph_alloc(n * 4), *deg = (uint32_t *)tr_graph_alloc(n * 4),
             *len = (uint32_t *)tr_graph_alloc(n * 4);
    int32_t rc = 0;
    if (!R || !Q || !deg || !len) rc = TR_GRAPH_E_MEMORY;
    tr_graph_adj push = tr_graph_push(g, dir), pull = tr_graph_pull(g, dir);
    uint64_t qn = 0, w = 0, best = 0;
    if (!rc) {
        for (uint64_t v = 0; v < n; v++) if (tr_graph_bit(sources, v)) { R[v >> 6] |= (uint64_t)1 << (v & 63); Q[qn++] = (uint32_t)v; }
        for (uint64_t i = 0; i < qn; i++) {
            uint32_t u = Q[i];
            for (uint64_t e = push.off[u]; e < push.off[u + 1]; e++) {
                uint32_t v = push.nbr[e];
                w++;
                if (!tr_graph_bit(R, v)) { R[v >> 6] |= (uint64_t)1 << (v & 63); Q[qn++] = v; }
            }
        }
        uint64_t reached = qn, head = 0, done = 0;
        qn = 0;
        for (uint64_t v = 0; v < n; v++) {
            if (!tr_graph_bit(R, v)) continue;
            uint32_t c = 0;
            for (uint64_t e = pull.off[v]; e < pull.off[v + 1]; e++) c += (uint32_t)tr_graph_bit(R, pull.nbr[e]);
            deg[v] = c;
            if (!c) Q[qn++] = (uint32_t)v;
        }
        while (head < qn) {
            uint32_t u = Q[head++];
            done++;
            if (len[u] > best) best = len[u];
            for (uint64_t e = push.off[u]; e < push.off[u + 1]; e++) {
                uint32_t v = push.nbr[e];
                w++;
                if (len[u] + 1 > len[v]) len[v] = len[u] + 1;
                if (--deg[v] == 0) Q[qn++] = v;
            }
        }
        if (done < reached) rc = TR_GRAPH_E_NO_SOLUTION;     /* a cycle is reachable: every count diverges */
    }
    free(R); free(Q); free(deg); free(len);
    if (!rc) { *L = best; if (work) *work = w; }
    return rc;
}

/* ------------------------------------------------------------------ layouts (GRAPH-001, libalg) */
tr_graph_layout *tr_graph_layout_build(const struct tr_scope_s *edges, const struct tr_scope_s *vertices,
                                       const tr_graph_spec *spec, const tr_math_budget *, tr_math_refusal *,
                                       int32_t *error);
int32_t tr_graph_layout_need(const struct tr_scope_s *edges, const struct tr_scope_s *vertices,
                             const tr_graph_spec *spec, tr_math_need *);
int32_t tr_graph_layout_get_info(const tr_graph_layout *, tr_graph_info *);
int32_t tr_graph_layout_get_view(const tr_graph_layout *, tr_graph_view *);
/* the dictionary: keys[0..n) in increasing order (key of id i) */
int32_t tr_graph_layout_keys(const tr_graph_layout *, int64_t *keys, uint64_t capacity);
/* the ids of keys[0..count): TR_MATH_NO_SOLUTION when one is no vertex (*missing = its index) */
int32_t tr_graph_layout_ids(const tr_graph_layout *, const int64_t *keys, uint64_t count, uint32_t *ids,
                            uint64_t *missing);
/* the vertex-incidence lists: offsets[n + 1], entries[2 x edges] (edge id << 2 | role); *count written */
int32_t tr_graph_layout_incidence(const tr_graph_layout *, uint64_t *offsets, uint64_t *entries, uint64_t capacity,
                                  uint64_t *count);
/* 1 when both layouts have the same identity (spec, descriptors, dictionary), else 0 */
int32_t tr_graph_layout_same(const tr_graph_layout *, const tr_graph_layout *);
/* TR_MATH_OK when edges (and vertices, or NULL) are the recorded scopes and fresh against their tables (table NULL
 * for an array scope; rows NULL = every row), TR_MATH_STALE otherwise */
int32_t tr_graph_layout_fresh(const tr_graph_layout *, const struct tr_scope_s *edges, const struct atable_s *etable,
                              const uint64_t *erows, const struct tr_scope_s *vertices, const struct atable_s *vtable,
                              const uint64_t *vrows);
void tr_graph_layout_free(tr_graph_layout *);
/* rationals num[i] / den[i] (den > 0) on their least common denominator: scaled[i] = num[i] D / den[i]; checked
 * (TR_MATH_OVERFLOW) */
int32_t tr_graph_rationals(uint64_t count, const int64_t *num, const int64_t *den, int64_t *scaled, int64_t *denominator);

/* ------------------------------------------------------------------ reductions (GRAPH-001, libalg) */
int32_t tr_graph_degrees(const tr_graph_layout *, int32_t kind, int32_t weights, tr_math_i128 *out,
                         const tr_math_budget *, tr_math_refusal *);
int32_t tr_graph_expand(const tr_graph_layout *, int32_t dir, const uint64_t *frontier, const uint64_t *exclude,
                        uint64_t *next, uint64_t *count, const tr_math_budget *, tr_math_refusal *);
/* x / y by semiring: COUNT int64 / tr_math_i128, BOOL and GF2 bitsets, MIN_PLUS / MAX_PLUS int64 / int64, FLOAT
 * double / double (err: double[n], FLOAT only); where: the lowest vertex of an overflow */
int32_t tr_graph_spmv(const tr_graph_layout *, int32_t semiring, int32_t dir, int32_t weights, const void *x,
                      const uint64_t *x_valid, void *y, uint64_t *y_valid, double *err, uint64_t *where,
                      const tr_math_budget *, tr_math_refusal *);
int32_t tr_graph_need(const tr_graph_layout *, int32_t request, int32_t semiring, uint64_t a, uint64_t b, uint32_t c,
                      int32_t target, tr_math_need *);
enum { TR_GRAPH_REQ_DEGREES = 1, TR_GRAPH_REQ_EXPAND = 2, TR_GRAPH_REQ_SPMV = 3, TR_GRAPH_REQ_GF2 = 4,
       TR_GRAPH_REQ_REACH = 5, TR_GRAPH_REQ_CLOSURE = 6, TR_GRAPH_REQ_WALKS = 7, TR_GRAPH_REQ_PATHS = 8,
       TR_GRAPH_REQ_ENERGY = 9, TR_GRAPH_REQ_ITERATE = 10 };
/* A mod 2 (n x n bits, row u = the out-entries of u, FIELD-001's packed layout: stride words per row) */
int32_t tr_graph_gf2_matrix(const tr_graph_layout *, int32_t weights, uint64_t *words, uint64_t stride,
                            const tr_math_budget *, tr_math_refusal *);

/* ------------------------------------------------------------------ fixed points, paths, energies (GRAPH-002) */
int32_t tr_graph_reach(const tr_graph_layout *, int32_t dir, const uint64_t *sources, int32_t reflexive,
                       uint64_t max_rounds, uint64_t *reached, uint32_t *level, tr_graph_rounds *,
                       const tr_math_budget *, tr_math_refusal *);
int32_t tr_graph_closure(const tr_graph_layout *, int32_t dir, const uint32_t *sources, uint64_t k, int32_t reflexive,
                         uint64_t max_rounds, uint64_t *rows, uint64_t stride, tr_graph_rounds *,
                         const tr_math_budget *, tr_math_refusal *);
/* the needs of a walk table and its rows: *lengths = L + 1 (unbounded: the support pre-pass finds the longest walk,
 * or TR_MATH_NO_SOLUTION when a cycle is reachable) */
int32_t tr_graph_walks_need(const tr_graph_layout *, int32_t dir, const uint64_t *sources, uint32_t max_length,
                            uint32_t limbs, int32_t keep_rows, int32_t target, tr_math_need *, uint64_t *lengths);
/* rows [lengths x n x W] or NULL, totals [n x W], lengths_out [lengths x W], grand [W]; W = limbs of the bound */
int32_t tr_graph_walks(const tr_graph_layout *, int32_t dir, const uint64_t *sources, uint32_t max_length,
                       const uint64_t *bound, uint32_t limbs, uint64_t *rows, uint64_t rows_capacity, uint64_t *totals,
                       uint64_t *lengths, uint64_t lengths_capacity, uint64_t *grand, tr_graph_walks_info *,
                       const tr_math_budget *, tr_math_refusal *);
int32_t tr_graph_paths(const tr_graph_layout *, int32_t semiring, int32_t algorithm, int32_t dir,
                       const uint64_t *sources, int64_t *dist, uint64_t *reached, uint32_t *hops, uint32_t *pred,
                       uint32_t *cycle, uint64_t cycle_capacity, tr_graph_paths_info *, const tr_math_budget *,
                       tr_math_refusal *);
/* out[5]: the 320-bit two's complement energy of x[n] (int64 logical values) */
int32_t tr_graph_energy(const tr_graph_layout *, const int64_t *x, uint64_t *out, const tr_math_budget *,
                        tr_math_refusal *);
int32_t tr_graph_pagerank(const tr_graph_layout *, int32_t dir, double damping, const uint64_t *teleport,
                          const uint64_t *start, double tol, uint64_t max_iter, double *rank, tr_graph_iter_info *,
                          const tr_math_budget *, tr_math_refusal *);
int32_t tr_graph_power(const tr_graph_layout *, int32_t dir, const uint64_t *start, double tol, uint64_t max_iter,
                       double *vec, tr_graph_iter_info *, const tr_math_budget *, tr_math_refusal *);

/* ------------------------------------------------------------------ the CUDA library (math_graph_gpu.cu)
 * Every entry takes the layout's view (host pointers; the arrays it needs are copied), checks the budget (the device
 * bytes of tr_graph_*_base_need with TR_GRAPH_NEED_DEVICE) and writes exactly what the CPU function writes:
 *   tr_mgpu_graph_degrees  one work item per vertex (its row / column; total = both)
 *   tr_mgpu_graph_expand   one work item per frontier word: its vertices push their entries into the next set by
 *                          atomic OR; then one work item per word masks it and adds its count by atomic add
 *   tr_mgpu_graph_spmv     one work item per vertex: the shared pull of the semiring, one output slot each; the FLOAT
 *                          coefficients computed on the host (tr_graph_float_coef) and copied; overflow: atomic min
 *   tr_mgpu_graph_reach / _closure / _walks / _paths: one launch per round: candidates (the push of the changed
 *                          vertices, deduplicated by atomic OR marks, appended by an atomic counter: the list's order
 *                          is not used), then the shared pull at each candidate into its own slot, then the commit;
 *                          violations and witnesses by atomic minimum; the host reads the round's counters
 *   tr_mgpu_graph_energy   one work item per edge slice, 320-bit local sums added with a multiword atomic add
 *   tr_mgpu_graph_pagerank / _power: per iteration the block sums (one work item per block), their total (one work
 *                          item), the pull update (one work item per vertex), the change; the host reads delta */
int tr_mgpu_graph_degrees(const tr_graph_view *, int32_t kind, int32_t weights, tr_math_i128 *out,
                          const tr_math_budget *, tr_math_refusal *);
int tr_mgpu_graph_expand(const tr_graph_view *, int32_t dir, const uint64_t *frontier, const uint64_t *exclude,
                         uint64_t *next, uint64_t *count, const tr_math_budget *, tr_math_refusal *);
int tr_mgpu_graph_spmv(const tr_graph_view *, int32_t semiring, int32_t dir, int32_t weights, const void *x,
                       const uint64_t *x_valid, void *y, uint64_t *y_valid, double *err, uint64_t *where,
                       const tr_math_budget *, tr_math_refusal *);
int tr_mgpu_graph_reach(const tr_graph_view *, int32_t dir, const uint64_t *sources, int32_t reflexive,
                        uint64_t max_rounds, uint64_t *reached, uint32_t *level, tr_graph_rounds *,
                        const tr_math_budget *, tr_math_refusal *);
int tr_mgpu_graph_closure(const tr_graph_view *, int32_t dir, const uint32_t *sources, uint64_t k, int32_t reflexive,
                          uint64_t max_rounds, uint64_t *rows, uint64_t stride, tr_graph_rounds *,
                          const tr_math_budget *, tr_math_refusal *);
int tr_mgpu_graph_walks(const tr_graph_view *, int32_t dir, const uint64_t *sources, uint32_t max_length,
                        const uint64_t *bound, uint32_t limbs, uint64_t *rows, uint64_t rows_capacity, uint64_t *totals,
                        uint64_t *lengths, uint64_t lengths_capacity, uint64_t *grand, tr_graph_walks_info *,
                        const tr_math_budget *, tr_math_refusal *);
int tr_mgpu_graph_paths(const tr_graph_view *, int32_t semiring, int32_t dir, const uint64_t *sources, int64_t *dist,
                        uint64_t *reached, uint32_t *hops, uint32_t *pred, uint32_t *cycle, uint64_t cycle_capacity,
                        tr_graph_paths_info *, const tr_math_budget *, tr_math_refusal *);
int tr_mgpu_graph_energy(const tr_graph_view *, const int64_t *x, uint64_t *out, const tr_math_budget *,
                         tr_math_refusal *);
int tr_mgpu_graph_pagerank(const tr_graph_view *, int32_t dir, double damping, const uint64_t *teleport,
                           const uint64_t *start, double tol, uint64_t max_iter, double *rank, tr_graph_iter_info *,
                           const tr_math_budget *, tr_math_refusal *);
int tr_mgpu_graph_power(const tr_graph_view *, int32_t dir, const uint64_t *start, double tol, uint64_t max_iter,
                        double *vec, tr_graph_iter_info *, const tr_math_budget *, tr_math_refusal *);
#ifdef __cplusplus
}
#endif
#endif
