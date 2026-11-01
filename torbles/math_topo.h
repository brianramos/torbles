/* TOPO-001: declared occupancy complexes and local topology; TOPO-002: bounded cycles, homology and persistence.
 * On the CPU (math_topo.c) and on a CUDA device (math_topo_gpu.cu: occupancy, cells and incidences, Euler counts,
 * local invariants, boundary-composition checks), with the same results bit for bit.
 * SPDX-License-Identifier: Apache-2.0
 *
 * ---- Declared geometry (tr_topo_grid) ----------------------------------------------------------------------------
 * The geometry is declared, never inferred: dim (2 or 3) coordinate columns of a scope (math_scope.h; axis a is scope
 * column col[a], its exact logical integer value, decimal scale 0), the extent lo[a] <= x_a <= hi[a] of the top cells
 * along each axis (n_a = hi[a] - lo[a] + 1 positions), the adjacency of the dual graph (4 or 8 in 2D; 6, 18 or 26 in
 * 3D) and optionally a value column (any exact column) for filtrations. Nothing else is geometry: the physical row
 * order, the stored codes (logical - codec offset), their bit planes and the torus layout of the columns are storage
 * and are never read as coordinates. A decimal column (scale > 0: its logical integer is a scaled storage form) is
 * refused as a coordinate (TR_MATH_DOMAIN_ERROR), as is a repeated coordinate column.
 *
 * ---- Cell convention -----------------------------------------------------------------------------------------------
 * A selected row whose coordinates are present and inside the extent occupies the closed unit top cell
 * [x_0, x_0 + 1] x .. x [x_{dim-1}, x_{dim-1} + 1]. A selected row with a NULL coordinate (null_rows) or with a present
 * coordinate outside the extent (outside_rows) occupies nothing; both are counted apart (NULL is decided first).
 * Rows of one cell collapse to its occupancy (duplicates = valid_rows - occupied). The complex is the closure of the
 * occupied top cells (every face of every occupied top cell): a cubical complex whose underlying space is the union
 * of the closed occupied cells; its 0-cells are lattice points, its 1-cells unit edges, and so on.
 * A cell is identified by Khalimsky coordinates c_a in 0 .. 2 n_a: c_a = 2 (x - lo_a) + 1 when the cell spans
 * [x, x + 1] along axis a, c_a = 2 (x - lo_a) when it sits at the coordinate x; its dimension is the number of odd
 * c_a, its anchor (lower corner) lo_a + floor(c_a / 2), its axes the odd c_a. Linear index L = c_0 + K_0 (c_1 + K_1 c_2)
 * with K_a = 2 n_a + 1 (axis 0 fastest). The k-cells are numbered 0 .. n_k - 1 in increasing L: a canonical order that
 * depends on the declared coordinates only (the top cells: increasing position p = i_0 + n_0 (i_1 + n_1 i_2), i_a =
 * x_a - lo_a).
 *
 * ---- Incidences and orientation -------------------------------------------------------------------------------------
 * A k-cell spanning the axes a_0 < .. < a_{k-1} has 2k faces of dimension k - 1: for i = 0 .. k-1, slot 2i holds the
 * minus face (c_{a_i} - 1) with sign -(-1)^i and slot 2i + 1 the plus face (c_{a_i} + 1) with sign +(-1)^i:
 *     d [x_0, x_0 + 1] x .. = sum_i (-1)^i (face_i^+ - face_i^-)        (the product orientation of the axes)
 * so d_{k-1} d_k = 0 over Z (hence over every field); GF(2) reads every sign as 1. The incidences are stored (face
 * index and sign per slot) and are what tr_topo_check verifies and the reductions read.
 *
 * ---- TOPO-001 results ------------------------------------------------------------------------------------------------
 *   counts (tr_topo_counts)  the row accounting, cells per dimension n_k, the Euler characteristic
 *                            chi = n_0 - n_1 + n_2 - n_3 (exact), the boundary cells per dimension and their chi.
 *   boundary cells           a cell of dimension k < dim lies on the topological boundary of the union iff one of the
 *                            2^(dim-k) top-cell positions around it is unoccupied or outside the extent (equivalently,
 *                            it is a face of a (dim-1)-cell with exactly one occupied coface); they form a subcomplex.
 *   local invariants         per cell (tr_topo_local): star[j] = the j-cells of the complex having it as a face
 *                            (star[k] = 1), the link counts (the link of a k-cell has one (j-k-1)-simplex per j-coface:
 *                            link[i] = star[k+1+i]) and the link's Euler characteristic sum_{j>k} (-1)^(j-k-1) star[j]
 *                            (2D: 0 at an interior vertex (a circle), 1 on a boundary arc, 2 at a pinch of two arcs),
 *                            the boundary flag and, for top cells, the dual degree under the declared adjacency.
 *   adjacency                the dual graph: occupied top cells, adjacent when they share an edge (4, 2D) / a face
 *                            (6, 3D), an edge (18), or a vertex (8, 2D; 26, 3D). Its components are labelled by their
 *                            smallest top-cell index. With 8 / 26 adjacency they are exactly the connected components
 *                            of the union (H_0 of the complex); with 4 / 6 / 18 they may be finer.
 * None of these is a homology claim: Betti numbers come from TOPO-002's reductions only, never from counts (two
 * complexes with equal occupancy, equal n_k and equal chi can have different Betti numbers).
 *
 * ---- TOPO-002: chain complexes, fields, homology ---------------------------------------------------------------------
 * The boundary matrices d_k (rows: the (k-1)-cells, columns: the k-cells, in their canonical order) of a complex, or of
 * an explicit integral chain complex (tr_topo_chain: given sparse integer matrices, d_{k-1} d_k = 0 verified over Z at
 * creation, top dimension <= TR_TOPO_MAX_CHAIN), over a declared field (tr_topo_field):
 *   TR_TOPO_GF2   GF(2). TR_TOPO_PACKED: the ranks through FIELD-001's packed matrices (tr_field_rank on d_k as a
 *                 tr_field_matrix, tr_topo_boundary_packed); TR_TOPO_REDUCTION: the sparse column reduction (XOR).
 *   TR_TOPO_GFP   GF(p) for a declared prime 3 <= p < 2^62, verified prime (FND-009 tr_gfp_field): coefficients mod p.
 *   TR_TOPO_Q     Q, exactly: fraction-free reduction with primitive integer columns (a column is scaled by nonzero
 *                 integers and divided by the gcd of its entries, which changes no pivot: the same pairs and ranks as
 *                 elimination over Q, as LIN-002's Bareiss elimination is). Entries are checked to fit int64 with
 *                 128-bit intermediates; an entry that does not fit is refused (TR_MATH_OVERFLOW), never wrapped.
 * rank_k = rank d_k (rank d_0 = rank d_{top+1} = 0) and the Betti numbers b_k = n_k - rank_k - rank_{k+1} are exact in the
 * declared field. They depend on it when the integral homology has torsion (RP^2: GF(2) gives 1, 1, 1, Q 1, 0, 0;
 * subcomplexes of the 2D / 3D grid have none, so every field agrees there).
 * The reduction is the standard column algorithm: the cells are totally ordered (dimension and canonical index for
 * homology, the filtration order below for persistence); for each k the columns of d_k are reduced left to right: while
 * column j is nonzero and its lowest row (the latest face in the order) is the lowest row of an earlier reduced column
 * i, column j becomes column j - (c_j / c_i) column i (GF(2): XOR; GF(p): mod p; Q: the fraction-free form). A nonzero
 * reduced column j pairs its lowest face (positive, born) with j (negative, the death); a zero column is positive. The
 * positive k-cells that are no column's lowest face are the essential classes: their number is b_k of the (final)
 * complex, and, with V tracking (cycles bit k), V_j (the k-chain with R_j = d V_j = 0, V_j[j] != 0) is a k-cycle; these
 * cycles form a basis of H_k (H_1: edge cycles). Representatives are normalized: GF(p) generator coefficient 1, Q
 * primitive with a positive generator coefficient; entries sorted by cell index.
 *
 * ---- Bounded filtrations and persistence (tr_topo_filtration) --------------------------------------------------------
 * m = nlevels >= 1 declared levels 0 .. m-1. Sources: TR_TOPO_SUBLEVEL (the complex's value column: a top cell enters at
 * the first level l with v <= thresholds[l], v = the minimum of its rows' present values; thresholds strictly
 * increasing logical values), TR_TOPO_SUPERLEVEL (v = the maximum, the first l with v >= thresholds[l], thresholds
 * strictly decreasing), TR_TOPO_LEVELS (explicit levels: one per top cell of a complex, or one per cell of every
 * dimension of an explicit chain complex, dimension-major; TR_TOPO_NEVER = never enters). In a complex every lower cell
 * enters with the earliest of the occupied top cells containing it (the closure); a top cell without a present value,
 * past the last threshold or at TR_TOPO_NEVER never enters (counted in never[]): the filtration is bounded, nothing is
 * claimed past level m - 1. An explicit chain's levels must be a filtration (a face no later than its cofaces), else
 * TR_MATH_DOMAIN_ERROR. The total order is (level, dimension, index) with the declared tie rule for the index:
 * TR_TOPO_TIES_INDEX increasing, TR_TOPO_TIES_REVERSE decreasing; both are filtrations, so the pairs are deterministic.
 * Pairs (tr_topo_pair): (dim k, birth level, death level, birth k-cell, death (k+1)-cell) for every nonzero reduced
 * column, birth <= death; zero-length pairs (birth == death) are reported and counted (they depend on the tie rule;
 * the pairs with birth < death and the essential classes (death TR_TOPO_NEVER, death_cell TR_TOPO_NONE) do not).
 * Pairs are sorted by (dim, birth, death, birth cell), the essential classes after them by (dim, birth, cell).
 *
 * ---- Budgets (math_budgets.h), checked before anything is allocated (refused: TR_MATH_BUDGET + refusal) -------------
 *   scan (tr_topo_count, the first phase of a build): variables = dim, states = positions (the occupancy bitmap),
 *     bytes = the bitmap and its rank directory (16 bytes per 64 positions) and the object; CUDA: + the device copies.
 *   build (after the scan, data-dependent: tr_topo_build_needs): output = cells sum n_k, tensor = incidence entries
 *     sum 2k n_k, bytes = the complex (8 per cell, 5 per incidence, 16 per top cell with a value column, the bitmaps)
 *     and its FND-007 record (made first, which checks its own bytes). CUDA (tr_mgpu_topo_build): bytes = the device
 *     copies; the adopted complex (tr_topo_adopt) as the CPU build.
 *   local / adjacency / check / boundary: output = the entries written; bytes = scratch; CUDA: the device copies.
 *   homology / persistence (tr_topo_reduce_need): states = the cells of the filtration (matrix rows and columns),
 *     tensor = their boundary entries (PACKED: the largest n_{k-1} n_k), support = the levels m (1 for homology), bytes =
 *     the fixed arrays; during the reduction (deterministic, so the refusal point is too): iterations = column
 *     additions (a step past the limit is refused, need = limit + 1), bytes = the fixed arrays plus every reduced
 *     column, V column and scratch buffer held (refused when an allocation would exceed the limit); after the
 *     reduction, before anything is written: output = pairs + essential classes + representative classes + entries.
 *     A refusal returns nothing: no partial pairs, ranks or cycles.
 *   explicit chain (tr_topo_chain_create): states = sum n_k, tensor = sum nnz, bytes = the copies and the d o d scratch.
 * Representation limits (TR_MATH_UNSUPPORTED with a budget that admits them): positions > TR_TOPO_MAX_POSITIONS, an
 * explicit chain past TR_TOPO_MAX_CHAIN dimensions or with 2^32 cells or entries in one dimension, p >= 2^62.
 *
 * Errors: TR_MATH_INVALID (-1: malformed input, a composite p), _MEMORY (-2), _OVERFLOW (-3: a Q entry past int64, an
 * explicit chain's d o d past int128), _BUDGET (-5), TR_MATH_DOMAIN_ERROR (-6: a decimal or repeated coordinate, an
 * explicit chain that is not one (d o d != 0), levels that are not a filtration), TR_MATH_DEVICE (-11),
 * TR_MATH_UNSUPPORTED (-12), TR_MATH_STALE (-13: tr_topo_fresh). Objects are immutable and individually owned
 * (constructors return NULL and set *error, inputs unchanged). A complex keeps an FND-007 record of its scope (the
 * coordinate and value columns with the exact selection): tr_topo_fresh compares it with a current scope and reports
 * every reason (identity, revision, rows, selection, NULL domain, columns: another layout of other columns is refused
 * there). */
#ifndef TORBLES_MATH_TOPO_H
#define TORBLES_MATH_TOPO_H
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "math_num.h"
#include "math_budgets.h"
#include "math_scope.h"
#include "math_field.h"
#include "math_certificates.h"
#ifdef __cplusplus
extern "C" {
#endif

#define TR_TOPO_ABI 1
#define TR_TOPO_MAX_DIM 3                               /* grids are 2D or 3D */
#define TR_TOPO_MAX_CHAIN 7                             /* explicit chain complexes: dimensions 0 .. 7 */
#define TR_TOPO_MAX_POSITIONS (UINT64_C(1) << 28)       /* top-cell positions of an extent (n_k < 2^32 then) */
#define TR_TOPO_NEVER (-1)                              /* the level of a cell that never enters */
#define TR_TOPO_NONE UINT64_MAX                         /* no cell, no witness */
#define TR_TOPO_CHUNK 256                               /* candidate cells per work item of the device build */
#define TR_TOPO_E_INVALID (-1)                          /* native codes (mathematics.h, math_domain.h): the CUDA */
#define TR_TOPO_E_MEMORY (-2)                           /* library does not include the first slice */
#define TR_TOPO_E_OVERFLOW (-3)
#define TR_TOPO_E_BUDGET (-5)
#define TR_TOPO_E_DOMAIN (-6)

enum { TR_TOPO_GF2 = 1, TR_TOPO_GFP = 2, TR_TOPO_Q = 3 };                    /* coefficient fields */
enum { TR_TOPO_REDUCTION = 1, TR_TOPO_PACKED = 2 };                         /* homology methods */
enum { TR_TOPO_SUBLEVEL = 1, TR_TOPO_SUPERLEVEL = 2, TR_TOPO_LEVELS = 3 };  /* filtration sources */
enum { TR_TOPO_TIES_INDEX = 1, TR_TOPO_TIES_REVERSE = 2 };                  /* tie rules within a level and dimension */
enum { TR_TOPO_BY_CPU = 0, TR_TOPO_BY_DEVICE = 1 };                         /* who built a complex */
enum { TR_TOPO_NEED_CPU = 0, TR_TOPO_NEED_DEVICE = 1 };
enum { TR_TOPO_ROW_UNSELECTED = 0, TR_TOPO_ROW_NULL = 1, TR_TOPO_ROW_OUTSIDE = 2, TR_TOPO_ROW_INSIDE = 3 };

/* ------------------------------------------------------------------ structs crossing the ctypes boundary */
typedef struct {
    int32_t dim, adjacency;           /* 2 or 3; 4 / 8 (2D), 6 / 18 / 26 (3D) */
    int32_t col[3];                   /* scope columns of the axes 0 .. dim-1 (-1 past dim) */
    int32_t value;                    /* scope column of the filtration value, or -1 */
    int64_t lo[3], hi[3];             /* the extent of the top cells (0, 0 past dim) */
} tr_topo_grid;                       /* 72 bytes */

typedef struct {
    uint64_t selected;                /* selected rows of the scope */
    uint64_t null_rows;               /* selected rows with a NULL coordinate */
    uint64_t outside_rows;            /* selected rows with present coordinates outside the extent */
    uint64_t valid_rows;              /* selected rows inside the extent */
    uint64_t occupied;                /* distinct occupied top cells (n_dim) */
    uint64_t duplicates;              /* valid_rows - occupied */
    uint64_t value_nulls;             /* valid rows whose value is NULL (a value column declared) */
    uint64_t valued;                  /* occupied top cells with at least one present value */
} tr_topo_rows;                       /* 64 bytes */

typedef struct {
    tr_topo_rows rows;
    uint64_t positions;               /* prod n_a */
    uint64_t candidates;              /* prod (2 n_a + 1): the Khalimsky grid */
    uint64_t n[4];                    /* cells per dimension (0 past dim) */
    uint64_t boundary[4];             /* cells per dimension on the topological boundary of the union */
    int64_t euler;                    /* sum (-1)^k n_k */
    int64_t boundary_euler;           /* sum (-1)^k boundary_k */
    int32_t dim, adjacency;
} tr_topo_counts;                     /* 168 bytes */

typedef struct {
    int32_t dim, adjacency;
    uint64_t ext[3];                  /* n_a (1 past dim) */
    uint64_t kdim[3];                 /* K_a = 2 n_a + 1 (1 past dim) */
    uint64_t kstride[3];              /* 1, K_0, K_0 K_1 */
    uint64_t pstride[3];              /* 1, n_0, n_0 n_1 */
    uint64_t positions, words, candidates;
    uint64_t n[4];                    /* cells per dimension */
    const uint64_t *occupancy;        /* words: bit p = position p occupied (bits past positions clear) */
    const uint64_t *cells[4];         /* n[k] increasing linear indices */
    const uint32_t *face[4];          /* k >= 1: 2k n[k] face indices into dimension k - 1, [cell][slot] */
    const int8_t *sign[4];            /* k >= 1: 2k n[k] orientation signs */
} tr_topo_view;                       /* 264 bytes: also a CUDA kernel argument */

typedef struct {
    tr_topo_counts counts;
    uint64_t cells, incidences;       /* sum n_k, sum 2k n_k */
    uint64_t bytes;                   /* bytes the complex retains (its FND-007 record included) */
    int64_t lo[3], hi[3];
    int32_t col[3], value;            /* the declared columns */
    int32_t backend, has_value;       /* TR_TOPO_BY_CPU / _DEVICE; 1 with a value column */
} tr_topo_info;                       /* 264 bytes */

typedef struct {
    uint8_t star[4];                  /* star[j]: j-cells of the complex having the cell as a face (0 for j < k, j > dim) */
    int8_t link_euler;                /* sum_{j > k} (-1)^(j-k-1) star[j] */
    uint8_t boundary;                 /* 1: on the topological boundary of the union */
    uint8_t degree;                   /* top cells: occupied neighbours in the declared adjacency (else 0) */
    uint8_t dim;
} tr_topo_local;                      /* 8 bytes */

typedef struct {
    uint64_t cells;                   /* cells evaluated */
    uint64_t boundary[4];             /* boundary cells per dimension */
    uint64_t dual_edges;              /* sum of the top cells' degrees / 2 */
    int64_t link_euler_sum[4];        /* per dimension: the sum of link_euler */
} tr_topo_local_totals;               /* 80 bytes */

typedef struct {
    uint64_t tops, edges, components; /* dual graph: occupied top cells, adjacent pairs, components */
    int32_t adjacency, reserved;
} tr_topo_adjacency;                  /* 32 bytes */

typedef struct {
    uint64_t checked[4];              /* k-cells whose d_{k-1} d_k was evaluated (k = 2 .. dim) */
    uint64_t violations_z[4];         /* of them with a nonzero integer coefficient */
    uint64_t violations_gf2[4];       /* of them with a (k-2)-cell met an odd number of times */
    uint64_t witness_z[4], witness_gf2[4];   /* the smallest violating k-cell (TR_TOPO_NONE) */
    uint64_t invalid;                 /* k-cells with an incidence entry out of range or a sign other than +-1 */
    uint64_t terms;                   /* composition terms evaluated */
    int32_t holds_z, holds_gf2;       /* no violation and no invalid entry */
} tr_topo_dd;                         /* 184 bytes */

typedef struct {
    int32_t field, method;            /* TR_TOPO_GF2 / _GFP / _Q; TR_TOPO_REDUCTION / _PACKED */
    uint64_t prime;                   /* GFP: the declared prime (else 0) */
    uint32_t cycles;                  /* bit k: a basis of H_k as k-cycles (REDUCTION only) */
    int32_t reserved;
} tr_topo_field;                      /* 24 bytes */

typedef struct {
    int32_t source, ties;             /* TR_TOPO_SUBLEVEL / _SUPERLEVEL / _LEVELS; TR_TOPO_TIES_* */
    uint64_t nlevels;                 /* m >= 1 */
    const int64_t *thresholds;        /* value sources: m logical values, strictly monotone */
    const int64_t *levels;            /* LEVELS: per top cell (complex) or per cell, dimension-major (chain) */
} tr_topo_filtration;                 /* 32 bytes */

typedef struct {
    int32_t top, field, method, kind; /* kind: 1 a complex, 2 an explicit chain */
    uint64_t prime;
    uint64_t n[TR_TOPO_MAX_CHAIN + 1];          /* cells per dimension */
    uint64_t rank[TR_TOPO_MAX_CHAIN + 2];       /* rank d_k, k = 0 .. top + 1 */
    uint64_t betti[TR_TOPO_MAX_CHAIN + 1];
    int64_t euler;
    uint64_t steps;                   /* column additions (0 for PACKED) */
    uint64_t peak_bytes;              /* the most bytes held at once */
    uint64_t classes, entries;        /* representatives: classes, their entries */
    uint32_t cycles;
    int32_t reserved;
} tr_topo_homology_info;              /* 272 bytes */

typedef struct {
    int32_t dim, reserved;
    uint64_t generator;               /* the essential positive cell (index within dim) */
    uint64_t offset, length;          /* its entries [offset, offset + length) */
} tr_topo_class;                      /* 32 bytes */

typedef struct {
    int32_t top, field, source, ties;
    uint64_t prime, nlevels;
    uint64_t entered[TR_TOPO_MAX_CHAIN + 1];    /* cells per dimension in the filtration */
    uint64_t never[TR_TOPO_MAX_CHAIN + 1];      /* cells per dimension that never enter */
    uint64_t rank[TR_TOPO_MAX_CHAIN + 2];       /* rank of d_k over the cells that enter */
    uint64_t betti[TR_TOPO_MAX_CHAIN + 1];      /* of the final complex: essential classes per dimension */
    uint64_t pairs, essential, zero_length;
    uint64_t steps, peak_bytes;
} tr_topo_persistence_info;           /* 336 bytes */

typedef struct {
    int32_t dim, reserved;            /* the class's dimension (its birth cell's) */
    int64_t birth, death;             /* levels; death = TR_TOPO_NEVER for an essential class */
    uint64_t birth_cell, death_cell;  /* indices within dim and dim + 1 (TR_TOPO_NONE for an essential class) */
} tr_topo_pair;                       /* 40 bytes */

typedef struct {
    int32_t top, reserved;
    uint64_t n[TR_TOPO_MAX_CHAIN + 1];
    uint64_t nnz[TR_TOPO_MAX_CHAIN + 1];        /* entries of d_k */
    int64_t euler;
    uint64_t bytes;
} tr_topo_chain_info;                 /* 152 bytes */

/* The arrays of a device build (tr_mgpu_topo_build writes them; tr_topo_adopt validates and copies them): capacities
 * n[k] must equal the counts of tr_mgpu_topo_counts. */
typedef struct {
    int32_t dim, has_value;
    uint64_t positions, words;
    uint64_t n[4];
    uint64_t *occupancy;              /* words */
    uint64_t *cells[4];               /* n[k] */
    uint32_t *face[4];                /* 2k n[k] */
    int8_t *sign[4];                  /* 2k n[k] */
    int64_t *vmin, *vmax;             /* n[dim] (has_value): extremes of each top cell's present values (min > max:
                                         none) */
    tr_topo_rows rows;
} tr_topo_states;                     /* 240 bytes */

typedef struct tr_topo_complex_s tr_topo_complex;
typedef struct tr_topo_chain_s tr_topo_chain;
typedef struct tr_topo_homology_s tr_topo_homology;
typedef struct tr_topo_persistence_s tr_topo_persistence;

/* ------------------------------------------------------------------ geometry (host and device) */
TR_NUM_HD int tr_topo_bit(const uint64_t *bits, uint64_t i) { return (int)((bits[i >> 6] >> (i & 63)) & 1); }
TR_NUM_HD int tr_topo_popc3(uint32_t x) { return (int)((x & 1) + ((x >> 1) & 1) + ((x >> 2) & 1)); }
static inline int tr_topo_adjacency_ok(int32_t dim, int32_t adj) {
    return dim == 2 ? (adj == 4 || adj == 8) : dim == 3 ? (adj == 6 || adj == 18 || adj == 26) : 0;
}
/* the geometry of a grid into a view (pointers NULL, counts 0): 0, TR_MATH_INVALID or TR_MATH_UNSUPPORTED (-12) */
static inline int32_t tr_topo_view_init(const tr_topo_grid *g, tr_topo_view *v) {
    if (!g || !v || (g->dim != 2 && g->dim != 3) || !tr_topo_adjacency_ok(g->dim, g->adjacency)) return TR_TOPO_E_INVALID;
    memset(v, 0, sizeof *v);
    v->dim = g->dim; v->adjacency = g->adjacency;
    uint64_t positions = 1, candidates = 1;
    for (int32_t a = 0; a < 3; a++) {
        if (a >= g->dim) {
            if (g->col[a] != -1 || g->lo[a] != 0 || g->hi[a] != 0) return TR_TOPO_E_INVALID;
            v->ext[a] = 1; v->kdim[a] = 1;
        } else {
            if (g->hi[a] < g->lo[a]) return TR_TOPO_E_INVALID;
            __int128 n = (__int128)g->hi[a] - (__int128)g->lo[a] + 1;
            if (n > (__int128)TR_TOPO_MAX_POSITIONS) return -12;
            v->ext[a] = (uint64_t)n; v->kdim[a] = 2 * (uint64_t)n + 1;
        }
        v->pstride[a] = positions; v->kstride[a] = candidates;
        positions = tr_budget_mul(positions, v->ext[a]);
        candidates = tr_budget_mul(candidates, v->kdim[a]);
        if (positions > TR_TOPO_MAX_POSITIONS) return -12;
    }
    v->positions = positions; v->candidates = candidates; v->words = (positions + 63) / 64;
    return 0;
}
/* the grid against a scope's view (geometry, then columns: in range, distinct, decimal scale 0; the value column in
 * range or -1): 0, TR_MATH_INVALID, TR_MATH_DOMAIN_ERROR (a decimal or repeated coordinate), TR_MATH_UNSUPPORTED */
static inline int32_t tr_topo_grid_check_view(const tr_scope_view *s, const tr_topo_grid *g, tr_topo_view *v) {
    if (!s || !g || !v) return TR_TOPO_E_INVALID;
    int32_t rc = tr_topo_view_init(g, v);
    if (rc) return rc;
    for (int32_t a = 0; a < g->dim; a++) {
        if (g->col[a] < 0 || g->col[a] >= s->ncols) return TR_TOPO_E_INVALID;
        for (int32_t b = 0; b < a; b++) if (g->col[b] == g->col[a]) return TR_TOPO_E_DOMAIN;
        if (s->col[g->col[a]].scale != 0) return TR_TOPO_E_DOMAIN;     /* a decimal is not a coordinate */
    }
    if (g->value < -1 || g->value >= s->ncols) return TR_TOPO_E_INVALID;
    if (s->rows && !s->selection) return TR_TOPO_E_INVALID;
    return 0;
}
/* the Khalimsky coordinates of linear index L (0 past dim); returns the cell's dimension */
TR_NUM_HD int32_t tr_topo_decode(const tr_topo_view *v, uint64_t L, uint64_t *c) {
    int32_t k = 0;
    for (int32_t a = 0; a < 3; a++) {
        if (a < v->dim) { c[a] = L % v->kdim[a]; L /= v->kdim[a]; k += (int32_t)(c[a] & 1); }
        else c[a] = 0;
    }
    return k;
}
/* the position of the top cell with (odd) Khalimsky coordinates t */
TR_NUM_HD uint64_t tr_topo_top_position(const tr_topo_view *v, const uint64_t *t) {
    uint64_t p = 0;
    for (int32_t a = 0; a < v->dim; a++) p += ((t[a] - 1) / 2) * v->pstride[a];
    return p;
}
/* The occupancy of the 2^m top-cell positions around cell c, m = its degenerate (even) axes in increasing order:
 * bit b chooses, for the j-th degenerate axis a, c_a - 1 (bit j of b clear) or c_a + 1 (set). *inside: the positions
 * inside the extent. The cell is in the complex iff the pattern is nonzero; a cell of dimension < dim is on the
 * boundary iff the pattern is not 2^(2^m) - 1. */
TR_NUM_HD uint32_t tr_topo_pattern(const tr_topo_view *v, const uint64_t *c, int32_t *m_out, uint32_t *inside_out) {
    int32_t deg[3], m = 0;
    for (int32_t a = 0; a < v->dim; a++) if (!(c[a] & 1)) deg[m++] = a;
    uint32_t pattern = 0, inside = 0;
    for (uint32_t b = 0; b < (1u << m); b++) {
        uint64_t t[3] = {c[0], c[1], c[2]};
        int ok = 1;
        for (int32_t j = 0; j < m; j++) {
            int32_t a = deg[j];
            if ((b >> j) & 1) { if (c[a] >= 2 * v->ext[a]) ok = 0; else t[a] = c[a] + 1; }
            else { if (c[a] == 0) ok = 0; else t[a] = c[a] - 1; }
        }
        if (!ok) continue;
        inside |= 1u << b;
        if (tr_topo_bit(v->occupancy, tr_topo_top_position(v, t))) pattern |= 1u << b;
    }
    *m_out = m;
    *inside_out = inside;
    return pattern;
}
TR_NUM_HD int tr_topo_is_boundary(int32_t m, uint32_t pattern) {
    return m > 0 && pattern != ((m == 3) ? 0xFFu : (m == 2) ? 0xFu : 0x3u);
}
/* star[j] for a k-cell of the given pattern: the j-cofaces (j = k + |s|) span s of the degenerate axes in directions
 * e; one is in the complex iff an occupied position around the cell agrees with e on s */
TR_NUM_HD void tr_topo_star(uint32_t pattern, int32_t m, int32_t k, uint8_t *star) {
    for (int32_t j = 0; j < 4; j++) star[j] = 0;
    uint32_t full = 1u << m;
    for (uint32_t s = 0; s < full; s++) {
        uint32_t e = 0;
        do {
            int hit = 0;
            for (uint32_t b = 0; b < full && !hit; b++) if ((b & s) == e && ((pattern >> b) & 1)) hit = 1;
            if (hit) star[k + tr_topo_popc3(s)]++;
            e = (e - s) & s;
        } while (e != 0);
    }
}
TR_NUM_HD int8_t tr_topo_link_euler(const uint8_t *star, int32_t k, int32_t dim) {
    int32_t x = 0;
    for (int32_t j = k + 1; j <= dim; j++) x += ((j - k - 1) & 1) ? -(int32_t)star[j] : (int32_t)star[j];
    return (int8_t)x;
}
/* the faces of the k-cell L (coordinates c): slot 2i minus, 2i + 1 plus face of its i-th axis, signs -(-1)^i, (-1)^i */
TR_NUM_HD int32_t tr_topo_faces(const tr_topo_view *v, uint64_t L, const uint64_t *c, uint64_t *face, int8_t *sign) {
    int32_t i = 0;
    for (int32_t a = 0; a < v->dim; a++) {
        if (!(c[a] & 1)) continue;
        int8_t s = (i & 1) ? (int8_t)-1 : (int8_t)1;
        face[2 * i] = L - v->kstride[a]; sign[2 * i] = (int8_t)-s;
        face[2 * i + 1] = L + v->kstride[a]; sign[2 * i + 1] = s;
        i++;
    }
    return i;
}
/* the index of L among n increasing values, or TR_TOPO_NONE */
TR_NUM_HD uint64_t tr_topo_find(const uint64_t *cells, uint64_t n, uint64_t L) {
    uint64_t lo = 0, hi = n;
    while (lo < hi) { uint64_t mid = lo + (hi - lo) / 2; if (cells[mid] < L) lo = mid + 1; else hi = mid; }
    return lo < n && cells[lo] == L ? lo : TR_TOPO_NONE;
}
/* the occupied neighbours of the top cell with coordinates c under the view's adjacency */
TR_NUM_HD uint32_t tr_topo_degree(const tr_topo_view *v, const uint64_t *c) {
    uint32_t deg = 0, combos = v->dim == 3 ? 27u : 9u;
    for (uint32_t o = 0; o < combos; o++) {
        int32_t off[3] = {(int32_t)(o % 3) - 1, (int32_t)((o / 3) % 3) - 1, v->dim == 3 ? (int32_t)(o / 9) - 1 : 0};
        int32_t nz = (off[0] != 0) + (off[1] != 0) + (off[2] != 0);
        if (!nz) continue;
        if ((v->adjacency == 4 || v->adjacency == 6) && nz != 1) continue;
        if (v->adjacency == 18 && nz > 2) continue;
        uint64_t p = 0;
        int ok = 1;
        for (int32_t a = 0; a < v->dim && ok; a++) {
            int64_t i = (int64_t)((c[a] - 1) / 2) + off[a];
            if (i < 0 || (uint64_t)i >= v->ext[a]) ok = 0;
            else p += (uint64_t)i * v->pstride[a];
        }
        if (ok && tr_topo_bit(v->occupancy, p)) deg++;
    }
    return deg;
}
/* the local invariants of the cell with linear index L */
TR_NUM_HD void tr_topo_local_cell(const tr_topo_view *v, uint64_t L, tr_topo_local *out) {
    uint64_t c[3];
    int32_t k = tr_topo_decode(v, L, c), m;
    uint32_t inside;
    uint32_t pattern = tr_topo_pattern(v, c, &m, &inside);
    tr_topo_star(pattern, m, k, out->star);
    out->link_euler = tr_topo_link_euler(out->star, k, v->dim);
    out->boundary = (uint8_t)tr_topo_is_boundary(m, pattern);
    out->degree = (uint8_t)(k == v->dim ? tr_topo_degree(v, c) : 0);
    out->dim = (uint8_t)k;
}
/* A selected row: TR_TOPO_ROW_NULL (a NULL coordinate), _OUTSIDE (present, outside the extent) or _INSIDE (with the
 * position *pos); TR_TOPO_ROW_UNSELECTED otherwise. Only the logical values of the declared columns are read. */
TR_NUM_HD int tr_topo_row(const tr_scope_view *s, const tr_topo_grid *g, const tr_topo_view *v, uint64_t row,
                          uint64_t *pos) {
    if (!tr_scope_is_selected(s, row)) return TR_TOPO_ROW_UNSELECTED;
    for (int32_t a = 0; a < g->dim; a++) if (!tr_scope_present(s, g->col[a], row)) return TR_TOPO_ROW_NULL;
    uint64_t p = 0;
    for (int32_t a = 0; a < g->dim; a++) {
        int64_t x = tr_scope_value(s, g->col[a], row);
        if (x < g->lo[a] || x > g->hi[a]) return TR_TOPO_ROW_OUTSIDE;
        p += ((uint64_t)x - (uint64_t)g->lo[a]) * v->pstride[a];
    }
    *pos = p;
    return TR_TOPO_ROW_INSIDE;
}
/* d_{k-1} d_k of the k-cell j (k >= 2) from the view's incidences: *zero_z (every integer coefficient is 0), *zero_2
 * (every (k-2)-cell met an even number of times), *invalid (an entry out of range or a sign not +-1); returns the
 * terms evaluated */
TR_NUM_HD uint32_t tr_topo_dd_cell(const tr_topo_view *v, int32_t k, uint64_t j, int *zero_z, int *zero_2, int *invalid) {
    uint64_t idx[24];
    int32_t coef[24];
    uint32_t cnt[24], terms = 0;
    int32_t used = 0;
    *invalid = 0;
    const uint32_t *F = v->face[k] + (uint64_t)2 * (uint64_t)k * j;
    const int8_t *S = v->sign[k] + (uint64_t)2 * (uint64_t)k * j;
    for (int32_t s = 0; s < 2 * k; s++) {
        uint64_t f = F[s];
        int32_t sf = S[s];
        if (f >= v->n[k - 1] || (sf != 1 && sf != -1)) { *invalid = 1; continue; }
        const uint32_t *G = v->face[k - 1] + (uint64_t)2 * (uint64_t)(k - 1) * f;
        const int8_t *T = v->sign[k - 1] + (uint64_t)2 * (uint64_t)(k - 1) * f;
        for (int32_t t = 0; t < 2 * (k - 1); t++) {
            uint64_t g = G[t];
            int32_t tg = T[t];
            if (g >= v->n[k - 2] || (tg != 1 && tg != -1)) { *invalid = 1; continue; }
            terms++;
            int32_t u = 0;
            while (u < used && idx[u] != g) u++;
            if (u == used) { idx[used] = g; coef[used] = 0; cnt[used] = 0; used++; }
            coef[u] += sf * tg;
            cnt[u]++;
        }
    }
    *zero_z = 1; *zero_2 = 1;
    for (int32_t u = 0; u < used; u++) {
        if (coef[u]) *zero_z = 0;
        if (cnt[u] & 1) *zero_2 = 0;
    }
    return terms;
}
static inline void tr_topo_dd_init(tr_topo_dd *e) {
    memset(e, 0, sizeof *e);
    for (int32_t k = 0; k < 4; k++) { e->witness_z[k] = TR_TOPO_NONE; e->witness_gf2[k] = TR_TOPO_NONE; }
}
static inline void tr_topo_dd_finish(tr_topo_dd *e) {
    uint64_t vz = 0, v2 = 0;
    for (int32_t k = 0; k < 4; k++) { vz += e->violations_z[k]; v2 += e->violations_gf2[k]; }
    e->holds_z = !vz && !e->invalid;
    e->holds_gf2 = !v2 && !e->invalid;
}
/* 0 when a view is consistent enough to read (dimension, geometry, pointers for its counts), else TR_MATH_INVALID */
static inline int32_t tr_topo_view_check(const tr_topo_view *v) {
    if (!v || (v->dim != 2 && v->dim != 3) || !tr_topo_adjacency_ok(v->dim, v->adjacency)) return TR_TOPO_E_INVALID;
    if (v->words != (v->positions + 63) / 64 || (v->words && !v->occupancy)) return TR_TOPO_E_INVALID;
    for (int32_t k = 0; k <= v->dim; k++) {
        if (v->n[k] && !v->cells[k]) return TR_TOPO_E_INVALID;
        if (k >= 1 && v->n[k] && (!v->face[k] || !v->sign[k])) return TR_TOPO_E_INVALID;
        if (v->n[k] > UINT32_MAX) return TR_TOPO_E_INVALID;
    }
    for (int32_t k = v->dim + 1; k < 4; k++) if (v->n[k]) return TR_TOPO_E_INVALID;
    return 0;
}
static inline uint64_t tr_topo_cells_total(const uint64_t *n) { return tr_budget_add(tr_budget_add(n[0], n[1]), tr_budget_add(n[2], n[3])); }
static inline uint64_t tr_topo_incidences(const uint64_t *n) {
    return tr_budget_add(tr_budget_add(tr_budget_mul(2, n[1]), tr_budget_mul(4, n[2])), tr_budget_mul(6, n[3]));
}

/* ------------------------------------------------------------------ needs (shared with the CUDA host code) */
/* bytes the device copies of one scope column take (codes or values, and the validity flags) */
static inline uint64_t tr_topo_column_bytes(const tr_scope_view *s, int32_t c) {
    uint64_t b = tr_budget_mul(s->rows, s->col[c].values ? 8 : 4);
    return s->col[c].valid ? tr_budget_add(b, s->rows) : b;
}
/* the scan (the occupancy and the Euler counts): variables = dim, states = positions, bytes = the occupancy bitmap,
 * its rank directory and (a value column) the valued bitmap; device: + the copies of the selection and the columns */
static inline void tr_topo_scan_need(const tr_scope_view *s, const tr_topo_grid *g, const tr_topo_view *v,
                                     int32_t device, tr_math_need *need) {
    for (int i = 0; i < TR_BUDGET_RESOURCES; i++) need->need[i] = 0;
    need->need[TR_BUDGET_VARIABLES] = (uint64_t)g->dim;
    need->need[TR_BUDGET_STATES] = v->positions;
    uint64_t bytes = tr_budget_add(tr_budget_mul(v->words, g->value >= 0 ? 24 : 16), 512);
    if (device) {
        bytes = tr_budget_add(bytes, tr_budget_mul((s->rows + 63) / 64, 8));
        for (int32_t a = 0; a < g->dim; a++) bytes = tr_budget_add(bytes, tr_topo_column_bytes(s, g->col[a]));
        if (g->value >= 0) bytes = tr_budget_add(bytes, tr_topo_column_bytes(s, g->value));
        bytes = tr_budget_add(bytes, 256);
    }
    need->need[TR_BUDGET_BYTES] = bytes;
}
/* the complex after the scan: output = cells, tensor = incidences, bytes = what it retains (the record apart) */
static inline uint64_t tr_topo_complex_bytes(const tr_topo_counts *c, int32_t has_value) {
    uint64_t words = (c->positions + 63) / 64;
    uint64_t b = tr_budget_add(tr_budget_mul(words, 16), 1024);
    b = tr_budget_add(b, tr_budget_mul(tr_topo_cells_total(c->n), 8));
    b = tr_budget_add(b, tr_budget_mul(tr_topo_incidences(c->n), 5));
    if (has_value) b = tr_budget_add(b, tr_budget_mul(c->n[c->dim], 16));
    return b;
}
/* the device build: every device copy (scope columns, selection, occupancy, the per-position value extremes, chunk
 * counts, cells, incidences, top-cell values) */
static inline uint64_t tr_topo_build_device_bytes(const tr_scope_view *s, const tr_topo_grid *g, const tr_topo_view *v,
                                                  const tr_topo_counts *c) {
    uint64_t bytes = tr_budget_add(tr_budget_mul((s->rows + 63) / 64, 8), tr_budget_mul(v->words, 8));
    for (int32_t a = 0; a < g->dim; a++) bytes = tr_budget_add(bytes, tr_topo_column_bytes(s, g->col[a]));
    if (g->value >= 0) {
        bytes = tr_budget_add(bytes, tr_topo_column_bytes(s, g->value));
        bytes = tr_budget_add(bytes, tr_budget_mul(v->positions, 16));
        bytes = tr_budget_add(bytes, tr_budget_mul(c->n[g->dim], 16));
    }
    uint64_t chunks = (v->candidates + TR_TOPO_CHUNK - 1) / TR_TOPO_CHUNK;
    bytes = tr_budget_add(bytes, tr_budget_mul(chunks, 32));
    bytes = tr_budget_add(bytes, tr_budget_mul(tr_topo_cells_total(c->n), 8));
    bytes = tr_budget_add(bytes, tr_budget_mul(tr_topo_incidences(c->n), 5));
    return tr_budget_add(bytes, 256);
}
/* the local invariants on a device: the occupancy, the cells, the outputs */
static inline uint64_t tr_topo_local_device_bytes(const tr_topo_view *v) {
    uint64_t cells = tr_topo_cells_total(v->n);
    return tr_budget_add(tr_budget_add(tr_budget_mul(v->words, 8), tr_budget_mul(cells, 16)), 256);
}
/* the composition check on a device: the incidences of dimensions 1 .. dim and the accumulators */
static inline uint64_t tr_topo_dd_device_bytes(const tr_topo_view *v) {
    return tr_budget_add(tr_budget_mul(tr_topo_incidences(v->n), 5), 512);
}

/* ------------------------------------------------------------------ TOPO-001 (libalg) */
int32_t tr_topo_grid_check(const tr_scope *, const tr_topo_grid *);
/* the needs of the scan (device: of tr_mgpu_topo_counts) */
int32_t tr_topo_scan_needs(const tr_scope *, const tr_topo_grid *, int32_t device, tr_math_need *);
/* the needs of a build with these counts (device 1: the device copies of tr_mgpu_topo_build; device 0: the complex) */
int32_t tr_topo_build_needs(const tr_scope *, const tr_topo_grid *, const tr_topo_counts *, int32_t device,
                            tr_math_need *);
/* the occupancy, row accounting, cells and boundary cells per dimension and Euler characteristics, without the
 * complex (budget: the scan) */
int32_t tr_topo_count(const tr_scope *, const tr_topo_grid *, const tr_math_budget *, tr_topo_counts *,
                      tr_math_refusal *);
tr_topo_complex *tr_topo_build(const tr_scope *, const tr_topo_grid *, const tr_math_budget *, tr_math_refusal *,
                               int32_t *error);
/* a complex from a device build's arrays, validated (occupancy, increasing cells of the right dimension, every face of
 * every cell at its slot with its sign, every lower cell a face of a higher one, the row accounting) and copied */
tr_topo_complex *tr_topo_adopt(const tr_scope *, const tr_topo_grid *, const tr_topo_states *, const tr_math_budget *,
                               tr_math_refusal *, int32_t *error);
void tr_topo_free(tr_topo_complex *);
int32_t tr_topo_get_info(const tr_topo_complex *, tr_topo_info *);
int32_t tr_topo_get_view(const tr_topo_complex *, tr_topo_view *);
int32_t tr_topo_get_grid(const tr_topo_complex *, tr_topo_grid *);
/* the k-cells' anchors (n_k x dim logical coordinates), axes masks (bit a) and linear indices (each may be NULL) */
int32_t tr_topo_cells(const tr_topo_complex *, int32_t k, int64_t *anchors, uint8_t *axes, uint64_t *linear,
                      uint64_t capacity);
/* the value extremes of the top cells (has_value): vmin[i] > vmax[i] when top cell i has no present value */
int32_t tr_topo_values(const tr_topo_complex *, int64_t *vmin, int64_t *vmax, uint64_t capacity);
/* per-cell local invariants of every cell, dimension-major (out: capacity >= sum n_k; NULL: totals only); budget:
 * output = cells */
int32_t tr_topo_local_get(const tr_topo_complex *, const tr_math_budget *, tr_topo_local *out, uint64_t capacity,
                      tr_topo_local_totals *, tr_math_refusal *);
int32_t tr_topo_local_view(const tr_topo_view *, tr_topo_local *out, uint64_t capacity, tr_topo_local_totals *);
/* the dual graph under `adjacency` (0: the declared one; any valid for the dimension): labels (n_dim, the smallest top
 * index of each component), pairs (2 x capacity uint32: (i, j), i < j, increasing; NULL: counted only); budget:
 * output = labels + listed pairs, bytes = the union-find (8 per top cell) */
int32_t tr_topo_adjacency_get(const tr_topo_complex *, int32_t adjacency, const tr_math_budget *, uint32_t *labels,
                              uint32_t *pairs, uint64_t pair_capacity, tr_topo_adjacency *, tr_math_refusal *);
/* FND-007 freshness of the complex against a current scope: TR_MATH_OK or TR_MATH_STALE with the reasons */
int32_t tr_topo_fresh(const tr_topo_complex *, const tr_scope *current, tr_cert_staleness *);
const tr_cert_record *tr_topo_get_record(const tr_topo_complex *);   /* borrowed */
/* 1 when two complexes hold the same grid, rows accounting, occupancy, cells, incidences and values */
int32_t tr_topo_equal(const tr_topo_complex *, const tr_topo_complex *);

/* ------------------------------------------------------------------ TOPO-002 (libalg) */
/* d o d = 0 over Z and GF(2) for every k-cell, k = 2 .. dim (budget: tensor = composition terms) */
int32_t tr_topo_check(const tr_topo_complex *, const tr_math_budget *, tr_topo_dd *, tr_math_refusal *);
int32_t tr_topo_check_view(const tr_topo_view *, const tr_math_budget *, tr_topo_dd *, tr_math_refusal *);
/* d_k as coordinates (row, col, value): over Z (prime 0: signs) or GF(p) (residues; p = 2: every entry 1); budget:
 * output = 2k n_k entries */
int32_t tr_topo_boundary(const tr_topo_complex *, int32_t k, uint64_t prime, const tr_math_budget *, uint64_t *rows,
                         uint64_t *cols, int64_t *vals, uint64_t capacity, tr_math_refusal *);
/* d_k over GF(2) as FIELD-001's packed matrix: out rows = n_{k-1}, cols = n_k (contents overwritten); budget:
 * tensor = rows x cols */
int32_t tr_topo_boundary_packed(const tr_topo_complex *, const tr_topo_chain *, int32_t k, const tr_math_budget *,
                                tr_field_matrix *out, tr_math_refusal *);
/* an explicit integral chain complex: top (0 .. TR_TOPO_MAX_CHAIN), n[0..top], d_k (k = 1 .. top) as CSC: ptr[k]
 * (n[k] + 1 offsets), row[k] (increasing within a column, < n[k-1]), val[k] (nonzero); d_{k-1} d_k = 0 verified over Z */
tr_topo_chain *tr_topo_chain_create(int32_t top, const uint64_t *n, const uint64_t *const *ptr,
                                    const uint32_t *const *row, const int64_t *const *val, const tr_math_budget *,
                                    tr_math_refusal *, int32_t *error);
void tr_topo_chain_free(tr_topo_chain *);
int32_t tr_topo_chain_get_info(const tr_topo_chain *, tr_topo_chain_info *);
/* the static needs of a homology (filtration NULL) or persistence request over a complex or an explicit chain */
int32_t tr_topo_reduce_need(const tr_topo_complex *, const tr_topo_chain *, const tr_topo_field *,
                            const tr_topo_filtration *, tr_math_need *);
/* exactly one of cx / chain */
tr_topo_homology *tr_topo_homology_build(const tr_topo_complex *, const tr_topo_chain *, const tr_topo_field *,
                                         const tr_math_budget *, tr_math_refusal *, int32_t *error);
int32_t tr_topo_homology_get_info(const tr_topo_homology *, tr_topo_homology_info *);
int32_t tr_topo_homology_classes(const tr_topo_homology *, tr_topo_class *out, uint64_t capacity);
int32_t tr_topo_homology_entries(const tr_topo_homology *, uint64_t *cells, int64_t *coefs, uint64_t capacity);
void tr_topo_homology_free(tr_topo_homology *);
tr_topo_persistence *tr_topo_persistence_build(const tr_topo_complex *, const tr_topo_chain *, const tr_topo_field *,
                                               const tr_topo_filtration *, const tr_math_budget *, tr_math_refusal *,
                                               int32_t *error);
int32_t tr_topo_persistence_get_info(const tr_topo_persistence *, tr_topo_persistence_info *);
/* the finite pairs then the essential classes (capacity >= pairs + essential) */
int32_t tr_topo_persistence_pairs(const tr_topo_persistence *, tr_topo_pair *out, uint64_t capacity);
/* the Betti numbers of K_level (0 <= level < m): classes born at or before it and not dead by it (top + 1 values) */
int32_t tr_topo_persistence_betti(const tr_topo_persistence *, int64_t level, uint64_t *betti, uint64_t capacity);
/* the entry level of every k-cell (TR_TOPO_NEVER) */
int32_t tr_topo_persistence_levels(const tr_topo_persistence *, int32_t k, int64_t *levels, uint64_t capacity);
void tr_topo_persistence_free(tr_topo_persistence *);

/* ------------------------------------------------------------------ the CUDA library (math_topo_gpu.cu)
 *   tr_mgpu_topo_counts  the occupancy (one work item per row slice: atomic OR of the bit, local counts added once)
 *                        and the cells / boundary cells per dimension (one work item per candidate slice, local
 *                        counts added once): tr_topo_count's counts
 *   tr_mgpu_topo_build   the occupancy and the per-position value extremes (atomic min / max), per-chunk cell counts
 *                        (one work item per chunk of TR_TOPO_CHUNK candidates, its own slots), their exclusive prefix
 *                        on the host, the cells written in increasing order (one writer per slot), the incidences
 *                        (one work item per cell: binary search of each face, its own slots), the top-cell values
 *   tr_mgpu_topo_local   one work item per cell: tr_topo_local_cell into its own slot, totals by atomics
 *   tr_mgpu_topo_dd      one work item per k-cell (k = 2 .. dim): tr_topo_dd_cell, counts by atomics, the smallest
 *                        violating cell by atomic minimum
 * The column reductions (homology, persistence) run on the CPU only. */
int tr_mgpu_topo_counts(const tr_scope_view *, const tr_topo_grid *, const tr_math_budget *, tr_topo_counts *,
                        tr_math_refusal *);
int tr_mgpu_topo_build(const tr_scope_view *, const tr_topo_grid *, const tr_math_budget *, tr_topo_states *,
                       tr_math_refusal *);
int tr_mgpu_topo_local(const tr_topo_view *, const tr_math_budget *, tr_topo_local *out, uint64_t capacity,
                       tr_topo_local_totals *, tr_math_refusal *);
int tr_mgpu_topo_dd(const tr_topo_view *, const tr_math_budget *, tr_topo_dd *, tr_math_refusal *);
#ifdef __cplusplus
}
#endif
#endif
