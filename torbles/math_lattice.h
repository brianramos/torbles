/* LATTICE-001: bounded k-condition occupancy histograms (the exact bag count of every cell of k Boolean axes over a
 * BOOL-001 domain) and LATTICE-002: their subset-lattice transforms (zeta sums over supersets or subsets, Moebius
 * inversion, inclusion-exclusion counts, Moebius coefficients of set functions up to an order), on the CPU
 * (math_lattice.c) and on a CUDA device (math_lattice_gpu.cu: the histogram scan, the butterfly passes), with the
 * same results bit for bit.
 * SPDX-License-Identifier: Apache-2.0
 *
 * ---- Axes (tr_lattice_axis) ---------------------------------------------------------------------------------------
 * A histogram has k axes over a BOOL-001 domain (math_bool.h: the selected rows of a scope with k0 conditions, or the
 * 2^k0 assignments). Axis i is condition `index` of the domain (TR_LATTICE_CONDITION) or the output of program
 * `index` of an array of BOOL-001 program views over the domain's k0 conditions (TR_LATTICE_PROGRAM: a compound
 * predicate, SQL three-valued logic). At a point an axis is TRUE, FALSE or UNKNOWN. The axes keep their given order:
 * axis i is bit i of a two-valued cell id and digit i (base 3) of a three-valued one. An axis may repeat a condition.
 *
 * ---- Cells and the NULL convention ------------------------------------------------------------------------------
 *   TR_LATTICE_APART (two-valued cells): 2^k cells; cell c holds the points where every axis is known and axis i is
 *     TRUE exactly when bit i of c is set. A point where some axis is UNKNOWN is in no cell: it is counted apart
 *     (unknown = points - complete), and per axis in its UNKNOWN count. UNKNOWN is never read as FALSE.
 *   TR_LATTICE_THREE (three-valued cells): 3^k cells; cell c = sum_i d_i 3^i holds the points whose axis i has the
 *     value d_i: 0 = FALSE, 1 = TRUE, 2 = UNKNOWN. Every point is in exactly one cell.
 * Counts are exact bag counts (uint64): every selected row (or assignment) counts once. Per axis the histogram also
 * keeps its TRUE / FALSE / UNKNOWN counts over all points (axis[3 i + 0..2]); points = complete + unknown.
 *
 * ---- Projections to the subset lattice (tr_lattice_hist_vector) ------------------------------------------------
 * The transforms act on 2^k vectors indexed by subsets A of the axes (bit i = axis i):
 *   TR_LATTICE_COMPLETE      the two-valued cells: h(A) = complete points TRUE exactly on A (both conventions)
 *   TR_LATTICE_TRUE_SETS     (THREE only) h(A) = points whose set of TRUE axes is A (FALSE and UNKNOWN both outside:
 *                            SQL's WHERE semantics, declared: a conjunction is TRUE iff each conjunct is TRUE)
 *   TR_LATTICE_FALSE_SETS    (THREE only) h(A) = points whose set of FALSE axes is A
 *   TR_LATTICE_UNKNOWN_SETS  (THREE only) h(A) = points whose set of UNKNOWN axes is A (A = 0: the complete points)
 * An APART histogram has no per-cell record of its unknown points: its other projections are TR_MATH_DOMAIN_ERROR.
 *
 * ---- Transforms (LATTICE-002) -------------------------------------------------------------------------------------
 * On a vector v over the 2^k subsets (exact integers, tr_math_i128):
 *   ZETA_UP      (Z+ v)(S) = sum_{A superset of S} v(A)        (on a projection: points TRUE on at least S, the
 *                                                               intersection count of S; Z+ v(empty) = the total)
 *   ZETA_DOWN    (Z- v)(S) = sum_{A subset of S} v(A)          (points TRUE on nothing outside S)
 *   MOBIUS_UP    the inverse of ZETA_UP:   v(S) = sum_{A superset of S} (-1)^{|A \ S|} (Z+ v)(A)  (exclusive regions)
 *   MOBIUS_DOWN  the inverse of ZETA_DOWN: m(S) = sum_{A subset of S} (-1)^{|S \ A|} v(A)  (Harsanyi dividends of a
 *                                                               set function v: its coalition interactions)
 * Fast transforms: k butterfly passes; pass j visits the 2^(k-1) pairs (A, A + 2^j) with bit j clear in A and updates
 * one cell of each pair (ZETA_UP: v(A) += v(A + 2^j); ZETA_DOWN: v(A + 2^j) += v(A); the Moebius passes subtract).
 * A pass's pairs are disjoint, so a pass is order independent; the passes run in order (one launch each on a device).
 * Direct coefficients (tr_lattice_coefficients): the same sums for requested sets only, without the full transform
 * (2^(k - |S|) terms for the UP transforms, 2^|S| for the DOWN ones). tr_lattice_order_sets enumerates the sets of
 * sizes min..max (by size, then increasing mask): the coalitions of bounded order.
 *
 * ---- Carriers: signed intermediates never wrap ------------------------------------------------------------------
 * Values and every intermediate are exact signed integers. TR_LATTICE_I128 (default): two's complement 128-bit cells
 * (tr_math_i128), every addition and subtraction checked: an overflow is TR_MATH_OVERFLOW and nothing is written. For
 * inputs that fit int64 no overflow is possible (|any intermediate| <= 2^k max |v| <= 2^(k+63) < 2^127 for k <= 63).
 * TR_LATTICE_I64: the same computation, every input and every output checked to fit int64 (TR_MATH_OVERFLOW
 * otherwise; the intermediates are 128-bit). TR_LATTICE_BIGINT (tr_lattice_transform_bigint, CPU): FND-009
 * arbitrary-precision integers (math_carriers.h), exact for any input; each number's limbs are budgeted by FND-009.
 *
 * ---- Inclusion-exclusion counts (tr_lattice_counts) --------------------------------------------------------------
 * On a projection h of a histogram (N = sum_A h(A)), for each requested set S of m axes: the marginal of h on S
 * (2^m cells), its superset sums I(T) = points TRUE on every axis of T (T subset of S), and
 *   all      = I(S)                              (S empty: N)
 *   any      = sum_{nonempty T subset of S} (-1)^{|T|+1} I(T)  (the union, summed in the order of |T|: the Bonferroni
 *              partial sums; peak = the largest |partial sum|, which may exceed int64 while the result does not)
 *   none     = N - any                           exactly = h(S)   (TRUE on S and on no other axis)
 * On TR_LATTICE_TRUE_SETS these are SQL's COUNT(*) WHERE c1 AND .. AND cm (all), WHERE c1 OR .. OR cm (any) and the
 * points where no axis of S is TRUE (none: FALSE or UNKNOWN); on TR_LATTICE_COMPLETE the same over the complete
 * points. All in checked 128-bit arithmetic.
 *
 * ---- Budgets (math_budgets.h), checked before anything is allocated --------------------------------------------
 *   histogram (tr_lattice_hist_need): variables = k, states = cells (2^k or 3^k), circuit = the nodes of the programs
 *     (each evaluated once per word, whatever the number of axes reading it), support = 2^k0 on an ASSIGNMENTS
 *     domain, output = cells, bytes = the object (8 per cell, 40 per axis, 256) and the scratch (16 per program node,
 *     16 per axis); TR_LATTICE_NEED_DEVICE: bytes = the device copies and scratch and the adopted object
 *     (tr_lattice_hist_device_bytes).
 *   from counts / merge: variables = k, states = output = cells, bytes = the object.
 *   vector: nothing is allocated (the caller's 2^k output; the histogram's cells were budgeted when it was made).
 *   transform (tr_lattice_transform_need): variables = k, tensor = 2^k values, iterations = k passes, bytes = 16 2^k
 *     (the scratch; on a device the device copy); BIGINT: bytes = 8 2^k (the handles; each number's limbs are FND-009's).
 *   coefficients: variables = k, tensor = the terms summed, output = n.
 *   order sets: variables = k, output = the sets.
 *   counts: variables = k, degree = the largest m, tensor = 2^k (the projection) + the cells read per request
 *     (2^k + m 2^m + 2^m each), output = n, bytes = 8 2^k + 16 2^m.
 * Representation limits (with a budget that admits them): TR_MATH_UNSUPPORTED for k > TR_LATTICE_MAX_AXES (30), more
 * than TR_LATTICE_MAX_REQUESTS sets per call.
 *
 * Errors: TR_MATH_INVALID (-1: malformed input, a program over another k than the domain), _MEMORY (-2), _OVERFLOW
 * (-3), _BUDGET (-5 with the refusal), TR_MATH_DOMAIN_ERROR (-6: a projection the convention does not record, merging
 * histograms of different shapes), TR_MATH_DEVICE (-11), TR_MATH_UNSUPPORTED (-12). Histograms are immutable,
 * individually owned objects (constructors return NULL and set *error, leaving their inputs unchanged); outputs
 * through pointers are written only on success. The freshness of a histogram over rows is its scope's (the Python
 * layer checks the masks' scope before every use). */
#ifndef TORBLES_MATH_LATTICE_H
#define TORBLES_MATH_LATTICE_H
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "math_num.h"
#include "math_budgets.h"
#include "math_bool.h"
#ifdef __cplusplus
extern "C" {
#endif

#define TR_LATTICE_ABI 1
#define TR_LATTICE_MAX_AXES 30                /* axes of a histogram, variables of a transform */
#define TR_LATTICE_MAX_REQUESTS (1 << 24)     /* sets per coefficients / counts call */
#define TR_LATTICE_LOCAL_CELLS 64             /* the device keeps per-thread counts up to this many cells */
#define TR_LATTICE_POINTS_SUM UINT64_MAX      /* tr_lattice_hist_from_counts: points = the sum of the cells */

enum { TR_LATTICE_CONDITION = 1, TR_LATTICE_PROGRAM = 2 };                          /* axis kinds */
enum { TR_LATTICE_APART = 0, TR_LATTICE_THREE = 1 };                                /* NULL conventions */
enum { TR_LATTICE_COMPLETE = 1, TR_LATTICE_TRUE_SETS = 2, TR_LATTICE_FALSE_SETS = 3,
       TR_LATTICE_UNKNOWN_SETS = 4 };                                               /* projections */
enum { TR_LATTICE_ZETA_UP = 1, TR_LATTICE_ZETA_DOWN = 2, TR_LATTICE_MOBIUS_UP = 3, TR_LATTICE_MOBIUS_DOWN = 4 };
enum { TR_LATTICE_I64 = 1, TR_LATTICE_I128 = 2, TR_LATTICE_BIGINT = 3 };            /* carriers */
enum { TR_LATTICE_NEED_CPU = 0, TR_LATTICE_NEED_DEVICE = 1 };

typedef struct { int32_t kind, index, reserved0, reserved1; } tr_lattice_axis;     /* 16 bytes */

typedef struct {
    uint64_t ncells;          /* 2^k or 3^k */
    uint64_t points;          /* selected rows, assignments, or the given total */
    uint64_t complete;        /* points where every axis is known */
    uint64_t unknown;         /* points - complete */
    uint64_t occupied;        /* nonzero cells */
    uint64_t bytes;           /* bytes the histogram retains */
    int32_t k, nulls;         /* axes; TR_LATTICE_APART / TR_LATTICE_THREE */
    int32_t kind, reserved;   /* TR_BOOL_ROWS / TR_BOOL_ASSIGNMENTS, or 0 (from counts, merged) */
} tr_lattice_hist_info;       /* 64 bytes */

typedef struct {
    uint64_t vars;                      /* S */
    int32_t order, reserved;            /* m = |S| */
    uint64_t terms;                     /* inclusion-exclusion terms summed: 2^m - 1 */
    tr_math_i128 all, any, none, exactly;
    tr_math_i128 peak;                  /* the largest |Bonferroni partial sum| of `any` */
} tr_lattice_count;                     /* 104 bytes */

typedef struct tr_lattice_hist_s tr_lattice_hist;
struct tr_bigint_s;                     /* FND-009 (math_carriers.h) */

/* ------------------------------------------------------------------ shared helpers (CPU code, CUDA host and kernels) */
/* 3^i for i <= TR_LATTICE_MAX_AXES (fits uint64) */
TR_NUM_HD uint64_t tr_lattice_pow3(int32_t i) { uint64_t p = 1; for (int32_t j = 0; j < i; j++) p *= 3; return p; }
/* the cells of k axes under a convention: 2^k or 3^k (k <= TR_LATTICE_MAX_AXES), UINT64_MAX past the limit */
TR_NUM_HD uint64_t tr_lattice_cells(int32_t k, int32_t nulls) {
    if (k < 0 || k > TR_LATTICE_MAX_AXES) return UINT64_MAX;
    return nulls == TR_LATTICE_THREE ? tr_lattice_pow3(k) : (uint64_t)1 << k;
}
/* checked 128-bit addition and subtraction: 1 and *r, or 0 on overflow (never wrapped) */
TR_NUM_HD int tr_lattice_add128(__int128 a, __int128 b, __int128 *r) {
    __int128 v = (__int128)((unsigned __int128)a + (unsigned __int128)b);
    if ((a >= 0) == (b >= 0) && (v >= 0) != (a >= 0)) return 0;
    *r = v; return 1;
}
TR_NUM_HD int tr_lattice_sub128(__int128 a, __int128 b, __int128 *r) {
    __int128 v = (__int128)((unsigned __int128)a - (unsigned __int128)b);
    if ((a >= 0) != (b >= 0) && (v >= 0) != (a >= 0)) return 0;
    *r = v; return 1;
}
/* the pair p (0 <= p < 2^(k-1)) of pass j: lo has bit j clear, hi = lo + 2^j */
TR_NUM_HD uint64_t tr_lattice_pair_lo(uint64_t p, int32_t j) {
    uint64_t low = p & (((uint64_t)1 << j) - 1);
    return ((p >> j) << (j + 1)) | low;
}
/* one butterfly of a transform op on the pair (lo, hi): 1, or 0 on overflow (the cells are then unspecified) */
TR_NUM_HD int tr_lattice_butterfly(int32_t op, tr_math_i128 *v, uint64_t lo, uint64_t hi) {
    __int128 a = tr_i128_get(v[lo]), b = tr_i128_get(v[hi]), r = 0;
    int ok;
    switch (op) {
    case TR_LATTICE_ZETA_UP: ok = tr_lattice_add128(a, b, &r); if (ok) v[lo] = tr_i128_set(r); break;
    case TR_LATTICE_ZETA_DOWN: ok = tr_lattice_add128(b, a, &r); if (ok) v[hi] = tr_i128_set(r); break;
    case TR_LATTICE_MOBIUS_UP: ok = tr_lattice_sub128(a, b, &r); if (ok) v[lo] = tr_i128_set(r); break;
    default: ok = tr_lattice_sub128(b, a, &r); if (ok) v[hi] = tr_i128_set(r); break;    /* MOBIUS_DOWN */
    }
    return ok;
}
/* The value and valid words of every axis at word w of the domain, masked by the selection sel: each program is
 * evaluated once (tr_bool_eval_base into its part of scratch: 2 words per node, the programs one after the other),
 * then axis i reads its condition's words or its program's root words. */
TR_NUM_HD void tr_lattice_word(const tr_bool_domain *d, const tr_lattice_axis *axes, int32_t k,
                               const tr_bool_program_view *progs, int32_t nprogs, uint64_t w, uint64_t sel,
                               uint64_t *scratch, uint64_t *value, uint64_t *valid) {
    uint64_t root[TR_LATTICE_MAX_AXES], at = 0;
    for (int32_t j = 0; j < nprogs; j++) {
        tr_bool_eval_base(&progs[j], d, w, scratch + at);
        root[j] = at + 2 * (uint64_t)progs[j].root;
        at += 2 * (uint64_t)progs[j].nnodes;
    }
    for (int32_t i = 0; i < k; i++) {
        uint64_t v, kk;
        if (axes[i].kind == TR_LATTICE_CONDITION) tr_bool_var_word(d, axes[i].index, w, &v, &kk);
        else { v = scratch[root[axes[i].index]]; kk = scratch[root[axes[i].index] + 1]; }
        value[i] = v & sel; valid[i] = kk & sel;
    }
}
/* the cell of point bit b of a word whose axis words are value[0..k), valid[0..k) (APART: every axis known there) */
TR_NUM_HD uint64_t tr_lattice_cell_of(const uint64_t *value, const uint64_t *valid, int32_t k, int32_t nulls, int b) {
    uint64_t cell = 0;
    if (nulls == TR_LATTICE_APART) {
        for (int32_t i = 0; i < k; i++) cell |= ((value[i] >> b) & 1) << i;
        return cell;
    }
    uint64_t mult = 1;
    for (int32_t i = 0; i < k; i++) {
        uint64_t digit = ((valid[i] >> b) & 1) ? ((value[i] >> b) & 1) : 2;
        cell += digit * mult;
        mult *= 3;
    }
    return cell;
}

/* ------------------------------------------------------------------ host-side validation (both libraries) */
/* 0, or TR_MATH_INVALID (or the domain's TR_MATH_UNSUPPORTED) for axes the domain and programs cannot give (complete
 * view checks); the axis limits are tr_lattice_hist_limits, checked after the budget */
static inline int32_t tr_lattice_axes_check(const tr_bool_domain *d, const tr_lattice_axis *axes, int32_t k,
                                            const tr_bool_program_view *progs, int32_t nprogs, int32_t nulls) {
    int32_t rc = tr_bool_domain_check(d);
    if (rc) return rc;
    if (k < 0 || (k && !axes) || nprogs < 0 || (nprogs && !progs) || (nulls != TR_LATTICE_APART && nulls != TR_LATTICE_THREE))
        return -1;
    for (int32_t j = 0; j < nprogs; j++) {
        rc = tr_bool_program_check(&progs[j]);
        if (rc) return rc;
        if (progs[j].k != d->k) return -1;
    }
    for (int32_t i = 0; i < k; i++) {
        const tr_lattice_axis *a = &axes[i];
        if (a->reserved0 || a->reserved1) return -1;
        if (a->kind == TR_LATTICE_CONDITION) { if (a->index < 0 || a->index >= d->k) return -1; }
        else if (a->kind == TR_LATTICE_PROGRAM) { if (a->index < 0 || a->index >= nprogs) return -1; }
        else return -1;
    }
    return 0;
}
/* TR_MATH_UNSUPPORTED (-12) past the representation limits of a histogram, else 0 */
static inline int32_t tr_lattice_hist_limits(int32_t k, int32_t nprogs) {
    return k > TR_LATTICE_MAX_AXES || nprogs > TR_LATTICE_MAX_AXES ? -12 : 0;
}
/* the program nodes evaluated per word (every program once) */
static inline uint64_t tr_lattice_prog_nodes(const tr_bool_program_view *progs, int32_t nprogs) {
    uint64_t n = 0;
    for (int32_t j = 0; j < nprogs; j++) n = tr_budget_add(n, (uint64_t)progs[j].nnodes);
    return n;
}
static inline uint64_t tr_lattice_object_bytes(uint64_t ncells, int32_t k) {
    return tr_budget_add(tr_budget_add(256, tr_budget_mul(ncells, 8)), tr_budget_mul((uint64_t)(k > 0 ? k : 0), 40));
}
/* the needs of a histogram build (see Budgets); the device bytes are filled by tr_lattice_hist_device_bytes */
static inline void tr_lattice_hist_base_need(const tr_bool_domain *d, int32_t k, const tr_bool_program_view *progs,
                                             int32_t nprogs, int32_t nulls, tr_math_need *need) {
    for (int i = 0; i < TR_BUDGET_RESOURCES; i++) need->need[i] = 0;
    uint64_t cells = tr_lattice_cells(k, nulls), nodes = tr_lattice_prog_nodes(progs, nprogs);
    need->need[TR_BUDGET_VARIABLES] = (uint64_t)k;
    need->need[TR_BUDGET_SUPPORT] = d->kind == TR_BOOL_ASSIGNMENTS ? d->points : 0;
    need->need[TR_BUDGET_CIRCUIT] = nodes;
    need->need[TR_BUDGET_OUTPUT] = cells;
    need->need[TR_BUDGET_STATES] = cells;
    need->need[TR_BUDGET_BYTES] = tr_budget_add(tr_lattice_object_bytes(cells, k),
                                                tr_budget_add(tr_budget_mul(nodes, 16), tr_budget_mul((uint64_t)k, 16)));
}
/* the CUDA work split of the histogram scan: slices of words, each with its own node words */
#define TR_LATTICE_WORK (UINT64_C(1) << 14)
#define TR_LATTICE_MIN_WORDS 4
#define TR_LATTICE_SCRATCH (UINT64_C(1) << 26)
static inline uint64_t tr_lattice_slices(uint64_t nw, uint64_t nodes) {
    uint64_t s = TR_LATTICE_WORK, cap = (nw + TR_LATTICE_MIN_WORDS - 1) / TR_LATTICE_MIN_WORDS;
    uint64_t per = tr_budget_mul(nodes, 16);
    if (s > cap) s = cap;
    if (per && s > TR_LATTICE_SCRATCH / per) s = TR_LATTICE_SCRATCH / per;
    return s ? s : 1;
}
/* the device bytes of tr_mgpu_lattice_hist: the domain words, the programs, the axes, the cells and axis counts, the
 * per-slice node words, and the histogram object the counts are adopted into (tr_lattice_hist_from_counts) */
static inline uint64_t tr_lattice_hist_device_bytes(const tr_bool_domain *d, int32_t k,
                                                    const tr_bool_program_view *progs, int32_t nprogs, int32_t nulls) {
    uint64_t bytes = 0, nw = d->nw;
    if (d->kind == TR_BOOL_ROWS) {
        bytes = tr_budget_mul(nw, 8);
        bytes = tr_budget_add(bytes, tr_budget_mul(tr_budget_mul((uint64_t)d->k, nw), d->valid ? 16 : 8));
    }
    for (int32_t j = 0; j < nprogs; j++)
        bytes = tr_budget_add(bytes, tr_budget_add(tr_budget_mul((uint64_t)progs[j].nnodes, 16),
                                                   tr_budget_mul((uint64_t)progs[j].nsched, 4)));
    bytes = tr_budget_add(bytes, tr_budget_mul((uint64_t)(nprogs > 0 ? nprogs : 0), sizeof(tr_bool_program_view)));
    bytes = tr_budget_add(bytes, tr_budget_mul((uint64_t)(k > 0 ? k : 0), sizeof(tr_lattice_axis) + 24));
    bytes = tr_budget_add(bytes, tr_budget_mul(tr_lattice_cells(k, nulls), 8));
    uint64_t nodes = tr_lattice_prog_nodes(progs, nprogs);
    bytes = tr_budget_add(bytes, tr_budget_mul(tr_lattice_slices(nw, nodes), tr_budget_mul(nodes, 16)));
    return tr_budget_add(bytes, tr_lattice_object_bytes(tr_lattice_cells(k, nulls), k));   /* + the adopted object */
}
/* the needs of a transform over 2^k values (see Budgets); device: the same (the device copy replaces the scratch) */
static inline int32_t tr_lattice_transform_base_need(int32_t k, int32_t op, int32_t carrier, tr_math_need *need) {
    if (k < 0 || op < TR_LATTICE_ZETA_UP || op > TR_LATTICE_MOBIUS_DOWN ||
        carrier < TR_LATTICE_I64 || carrier > TR_LATTICE_BIGINT) return -1;
    for (int i = 0; i < TR_BUDGET_RESOURCES; i++) need->need[i] = 0;
    uint64_t n = tr_budget_power2((uint32_t)k);
    need->need[TR_BUDGET_VARIABLES] = (uint64_t)k;
    need->need[TR_BUDGET_TENSOR] = n;
    need->need[TR_BUDGET_ITERATIONS] = (uint64_t)k;
    need->need[TR_BUDGET_BYTES] = tr_budget_mul(n, carrier == TR_LATTICE_BIGINT ? 8 : 16);
    return 0;
}

/* ------------------------------------------------------------------ histograms (LATTICE-001, libalg) */
/* The histogram of k axes over a domain view (tr_bool_masks_get_domain), with nprogs program views for the program
 * axes (tr_bool_program_get_view; every program over the domain's k0 conditions); budget NULL = defaults. */
tr_lattice_hist *tr_lattice_hist_build(const tr_bool_domain *, const tr_lattice_axis *axes, int32_t k,
                                       const tr_bool_program_view *progs, int32_t nprogs, int32_t nulls,
                                       const tr_math_budget *, tr_math_refusal *, int32_t *error);
/* the needs checked by tr_lattice_hist_build (TR_LATTICE_NEED_CPU) or tr_mgpu_lattice_hist (_DEVICE) */
int32_t tr_lattice_hist_need(const tr_bool_domain *, const tr_lattice_axis *axes, int32_t k,
                             const tr_bool_program_view *progs, int32_t nprogs, int32_t nulls, int32_t target,
                             tr_math_need *);
/* A histogram from given counts (the readback of tr_mgpu_lattice_hist, or synthetic data): cells[ncells], points the
 * total (TR_LATTICE_POINTS_SUM: the sum of the cells), axis[3k] the per-axis TRUE / FALSE / UNKNOWN counts or NULL (then derived from the cells: THREE exactly;
 * APART only when no point is unknown, points == sum of the cells). Checked consistent (TR_MATH_INVALID otherwise):
 * THREE: the cells sum to points and give exactly the axis counts; APART: the cells sum to complete <= points, every
 * axis count is at least what the cells give and the unknown points lie between max_i UNKNOWN_i and sum_i
 * UNKNOWN_i. kind: TR_BOOL_ROWS, TR_BOOL_ASSIGNMENTS or 0 (provenance only). */
tr_lattice_hist *tr_lattice_hist_from_counts(int32_t k, int32_t nulls, int32_t kind, uint64_t points,
                                             const uint64_t *cells, const uint64_t *axis, const tr_math_budget *,
                                             tr_math_refusal *, int32_t *error);
/* the histogram of the union of two disjoint point sets: same k and convention (else TR_MATH_DOMAIN_ERROR); counts
 * add (TR_MATH_OVERFLOW past uint64) */
tr_lattice_hist *tr_lattice_hist_merge(const tr_lattice_hist *, const tr_lattice_hist *, const tr_math_budget *,
                                       tr_math_refusal *, int32_t *error);
int32_t tr_lattice_hist_get_info(const tr_lattice_hist *, tr_lattice_hist_info *);
/* copies of the cells (ncells) and of the axis counts (3k) */
int32_t tr_lattice_hist_cells(const tr_lattice_hist *, uint64_t *cells, uint64_t capacity);
int32_t tr_lattice_hist_axis_counts(const tr_lattice_hist *, uint64_t *axis, uint64_t capacity);
/* 1 when both hold the same shape and counts, 0 otherwise */
int32_t tr_lattice_hist_equal(const tr_lattice_hist *, const tr_lattice_hist *);
/* the 2^k projection (see Projections) as counts / as tr_math_i128 values (the transform input) */
int32_t tr_lattice_hist_vector(const tr_lattice_hist *, int32_t projection, uint64_t *out, uint64_t capacity);
int32_t tr_lattice_hist_vector_i128(const tr_lattice_hist *, int32_t projection, tr_math_i128 *out, uint64_t capacity);
void tr_lattice_hist_free(tr_lattice_hist *);

/* ------------------------------------------------------------------ transforms (LATTICE-002, libalg) */
int32_t tr_lattice_transform_need(int32_t k, int32_t op, int32_t carrier, tr_math_need *);
/* out[2^k] = op(in[2^k]) by the fast transform (carrier TR_LATTICE_I64 or _I128); out may equal in */
int32_t tr_lattice_transform(int32_t k, int32_t op, int32_t carrier, const tr_math_i128 *in, tr_math_i128 *out,
                             const tr_math_budget *, tr_math_refusal *);
/* the same over FND-009 big integers: out[2^k] receives new owned handles (all or nothing) */
int32_t tr_lattice_transform_bigint(int32_t k, int32_t op, const struct tr_bigint_s *const *in,
                                    struct tr_bigint_s **out, const tr_math_budget *, tr_math_refusal *);
/* out[j] = (op v)(sets[j]) by its direct sum (no full transform); carrier as for tr_lattice_transform */
int32_t tr_lattice_coefficients(int32_t k, int32_t op, int32_t carrier, const tr_math_i128 *values,
                                const uint64_t *sets, int32_t n, const tr_math_budget *, tr_math_i128 *out,
                                tr_math_refusal *);
/* the needs of tr_lattice_coefficients (tensor = the terms of the direct sums) */
int32_t tr_lattice_coefficients_need(int32_t k, int32_t op, const uint64_t *sets, int32_t n, tr_math_need *);
/* the sets of sizes min_order..max_order of k axes, by size then increasing mask: *n of them (sets NULL: count only;
 * capacity too small: TR_MATH_INVALID) */
int32_t tr_lattice_order_sets(int32_t k, int32_t min_order, int32_t max_order, const tr_math_budget *, uint64_t *sets,
                              uint64_t capacity, uint64_t *n, tr_math_refusal *);
/* inclusion-exclusion counts of the requested sets on a projection (see Inclusion-exclusion counts) */
int32_t tr_lattice_counts(const tr_lattice_hist *, int32_t projection, const uint64_t *sets, int32_t n,
                          const tr_math_budget *, tr_lattice_count *out, tr_math_refusal *);
int32_t tr_lattice_counts_need(const tr_lattice_hist *, int32_t projection, const uint64_t *sets, int32_t n,
                               tr_math_need *);

/* ------------------------------------------------------------------ the CUDA library (math_lattice_gpu.cu)
 * tr_mgpu_lattice_hist: the histogram scan of tr_lattice_hist_build on the device (views of host memory, copied;
 *   budgeted with tr_lattice_hist_base_need, bytes = tr_lattice_hist_device_bytes): work items are slices of 64-point
 *   words; each computes the cell of every point of its words from the axis words and adds it with an integer atomic
 *   (or, for at most TR_LATTICE_LOCAL_CELLS cells, into per-thread counts added once at the end). Writes cells[ncells]
 *   and axis[3k]; adopt them with tr_lattice_hist_from_counts.
 * tr_mgpu_lattice_transform: tr_lattice_transform on the device: k launches, one per butterfly pass, one work item per
 *   pair, an overflow flag raised by any checked addition that would wrap (then TR_MATH_OVERFLOW, nothing written). */
int tr_mgpu_lattice_hist(const tr_bool_domain *, const tr_lattice_axis *axes, int32_t k,
                         const tr_bool_program_view *progs, int32_t nprogs, int32_t nulls, const tr_math_budget *,
                         uint64_t *cells, uint64_t *axis, tr_math_refusal *);
int tr_mgpu_lattice_transform(int32_t k, int32_t op, int32_t carrier, const tr_math_i128 *in, tr_math_i128 *out,
                              const tr_math_budget *, tr_math_refusal *);
#ifdef __cplusplus
}
#endif
#endif
