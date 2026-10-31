/* DIST-001 / DIST-002: reusable scalar and joint histograms over a data scope, and the prefix calculus of a sorted
 * support (CDF, tails, ranges, exact quantiles, trimmed sums, range integrals), on the CPU and in CUDA.
 * SPDX-License-Identifier: Apache-2.0
 *
 * ---------------------------------------------------------------- DIST-001: the histogram state
 * Source. A histogram summarizes a FND-004 scope (math_scope.h): its selected rows, its dense groups g < G, and
 * k = 1..3 of its value columns (the coordinates: logical int64 values, decimals as scaled integers with their scale),
 * optionally with a weight column (any scope column, logical int64, its own scale).
 *
 * Cells. A cell of group g is a k-tuple whose coordinate c is either a value or NULL: the NULL category of each column
 * is a coordinate value of its own, never merged into a value, so (3, NULL), (NULL, NULL) and (3, 5) are three cells
 * (SQL GROUP BY semantics). Per cell: count (its rows, the bag multiplicity), and when weighted wcount (its rows whose
 * weight is present) and wsum (the exact sum of those weights, signed 128 bits: SQL SUM, NULL weights skipped).
 * Per group: rows[g] (its selected rows = the sum of its cells' counts). Presence is explicit: group g is present iff
 * rows[g] > 0 (a group whose values are all NULL is present, with its rows in NULL cells); an absent group has no
 * rows and no occupied cell, which is not an empty histogram of a present group.
 *
 * Order. Cells are ordered lexicographically, coordinate 0 most significant; within a coordinate values increase and
 * NULL comes after every value (NULLS LAST).
 *
 * Representations (TR_DIST_DENSE, TR_DIST_SPARSE), exact images of each other:
 *   dense:  a bounded domain per coordinate c: the values lo[c] .. lo[c] + span[c] - 1, plus a NULL slot (index
 *           span[c]) when nullslot[c]; dim[c] = span[c] + nullslot[c]; cells = prod dim[c] per group; group g owns
 *           the array entries g * cells .. (g + 1) * cells - 1 in cell order (index = sum_c i_c * stride[c], stride of
 *           the last coordinate 1), structural zeros included. The scope domain (TR_DIST_DOMAIN_SCOPE) of a coordinate
 *           is [min, max] of its selected present values with a NULL slot iff a selected value is NULL (no value:
 *           span 0); an explicit domain (TR_DIST_DOMAIN_EXPLICIT) must contain every selected value and NULL
 *           (else TR_MATH_DOMAIN_ERROR before anything is read).
 *   sparse: per group the occupied cells (count > 0) in cell order: offset[g] .. offset[g + 1] - 1 (CSR), each with
 *           its coordinates value[c][i] (0 where NULL) and the NULL mask nulls[i] (bit c: coordinate c is NULL).
 * Conversions (tr_dist_hist_to_dense / _to_sparse) preserve every occupied cell exactly; a dense domain that would
 * drop an occupied cell is refused (TR_MATH_DOMAIN_ERROR), an over-budget dense joint domain with TR_MATH_BUDGET.
 *
 * Construction. tr_dist_hist_init creates a builder (owned by one caller, calls serialized) over a scope that must
 * outlive it; tr_dist_hist_accumulate adds the selected rows of a physical row range (each row once); merging a
 * builder of the same scope descriptor and spec adds its rows (disjoint coverage, else TR_MATH_INVALID);
 * tr_dist_hist_finish (every row covered) makes the immutable histogram. tr_dist_hist_build is init + accumulate(0,
 * rows) + finish. Every route gives the same state bit for bit. The CPU strategy (TR_DIST_STRATEGY_*): dense states
 * are counted directly (one pass); a sparse state is counted in a dense scratch over the scope domain and compacted
 * (COUNT, chosen by AUTO when groups x cells <= 4 x selected rows + 65536) or built by sorting one entry per row and
 * reducing equal cells (SORT: any domain, e.g. near-row-sized or wide joint supports). info.scans counts the passes
 * over the scope's rows (1 per build; a state derived from another one: 0), info.passes those over the scratch.
 *
 * Merge, rollup, marginal. tr_dist_hist_merge adds two finished histograms of the same shape (k, representation,
 * weightedness, scales, groups, and for dense the same domain): the union of two disjoint bags (e.g. partitions of a
 * distributed table); the result is unscoped. tr_dist_hist_rollup maps fine groups to coarse ones (tr_scope_rollup_map
 * gives the refinement map between two scopes of the same rows): cells add, a coarse group is present iff one of its
 * fine groups is, and with `origin` (the coarse scope) the result is that scope's histogram. tr_dist_hist_marginal
 * keeps a subset of the coordinates and sums the others out (NULL categories included): it equals the histogram of
 * the kept columns built directly.
 *
 * Exactness. Counts are exact uint64; weight sums are exact signed 128-bit integers: before accumulating, the build
 * proves selected rows x max |w| <= 2^127 - 1 from the scope's extremes (else TR_MATH_OVERFLOW). Merges and rollups
 * check their sums (TR_MATH_OVERFLOW, never wrapped).
 *
 * Budgets (math_budgets.h), checked before allocation: variables = k (+1 weighted); support = cells per group (dense:
 * cells; sparse: the largest group's occupied cells); states = groups x cells held (dense and the COUNT scratch:
 * G x cells; SORT and derived sparse states: the occupied cells); bytes = every allocation. A sparse build checks the
 * scratch first and the scratch plus the final state once the occupied cells are counted, before the state is
 * allocated (the refusal reports the first failing check). The CUDA path checks its device copies
 * (tr_dist_hist_need, TR_DIST_NEED_DEVICE) and the adoption its own state (TR_DIST_NEED_ADOPT).
 *
 * CUDA (math_dist_gpu.cu). tr_mgpu_dist_hist_build writes exactly the states tr_dist_hist_build computes: the row counts
 * of the groups (a slice launch) and, dense, one atomic add per row into its cell (128-bit weight sums with the
 * two-word atomic of math_cuda.h); sparse, the same counting over the scope domain into a device scratch followed by a
 * compaction in three launches (chunk occupancy, one serial scan of the chunk counts, the ordered writes of each chunk
 * with the group offsets), with no barrier inside a kernel; a domain whose scratch exceeds the budget is refused there
 * (the CPU SORT strategy has no such bound). tr_dist_hist_from_states adopts the readback after consistency checks.
 *
 * ---------------------------------------------------------------- DIST-002: the prefix state and its requests
 * From a 1-D histogram (k = 1), per group: the sorted distinct present values x_1 < ... < x_d with mass m_i > 0
 * (TR_DIST_MASS_COUNT: the cell count; TR_DIST_MASS_WEIGHT: the cell's weight sum, which needs every present weight
 * >= 0: TR_MATH_DOMAIN_ERROR otherwise; cells of zero weight are not part of the weighted support), and the inclusive
 * prefix sums P_k[i] = sum_{j <= i} m_j x_j^k for k = 0..degree (degree <= TR_DIST_MAX_DEGREE), signed 128 bits.
 * NULL values are not in the support (nulls[g] counts their rows; rows with a NULL weight carry no weight mass).
 * Exactness: the build proves T x X^k <= 2^127 - 1 for every kept k (T = the group's total mass, X = max |x|), else
 * TR_MATH_OVERFLOW; every request's sums then fit 128 bits; quantile values and ratios are exact rationals
 * (numerator 256-bit two's complement, denominator > 0, not necessarily in lowest terms).
 * Semantics in terms of the expanded multiset (each value x_i repeated m_i times: rows, or the weight's logical
 * integer for weights, i.e. replication in units of 10^-wscale) with order statistics y_1 <= ... <= y_M, M = sum m_i:
 *   RANGE      the mass of the values in an interval: bounds a, b rational (num / den, den >= 1, logical units),
 *              each closed or open (TR_DIST_LO_CLOSED, TR_DIST_HI_CLOSED) or absent (TR_DIST_LO_NONE, _HI_NONE);
 *              CDF(t) = RANGE(-inf, t] (closed) or (-inf, t) (open), tails [t, inf) or (t, inf); answer mass and the
 *              fraction mass / M (M = 0: empty, no fraction).
 *   MOMENT     sum m_i x_i^k over the interval (k = power <= degree): a range integral; exact integer.
 *   QUANTILE   level p = a / b, 0 <= p <= 1, rule TR_DIST_Q_DISC (SQL PERCENTILE_DISC: y_r, r = max(1, ceil(p M))),
 *              _CONT (SQL PERCENTILE_CONT: h = p (M - 1), y_{j+1} + (h - j)(y_{j+2} - y_{j+1}), j = floor(h)),
 *              _LOWER (y_{j+1}), _HIGHER (y_{j+1} when h is an integer, else y_{j+2}); ties are equal order
 *              statistics; M = 0 is empty (SQL NULL).
 *   RANK       y_r for 1 <= r <= M (a = r).
 *   TRIMMED    sum_{r = r1}^{r2} y_r^k (a = r1, b = r2, 1 <= r1 <= r2 + 1, r2 <= M; r1 = r2 + 1 sums nothing);
 *   TRIMMED_MEAN  that sum over r2 - r1 + 1 ranks (empty when there is none).
 * Each request has its own status (0 or TR_MATH_INVALID / _UNSUPPORTED), so a batch never fails as a whole. The
 * evaluation (tr_dist_eval) is shared by the CPU loop and the CUDA kernel (one work item per request: binary searches
 * over the prefix arrays), so both give identical results; tr_mgpu_dist_prefix_upload keeps a prefix state on the
 * device for any number of batches (budget: bytes of the copies; a batch: output = requests, bytes = its copies).
 *
 * Errors: TR_MATH_INVALID, _MEMORY, _OVERFLOW, _BUDGET (with the tr_math_refusal), _UNSUPPORTED, _DOMAIN_ERROR,
 * TR_MATH_STALE is the caller's (the Python layer checks scope freshness), TR_MATH_DEVICE. Constructors return NULL
 * and set *error, leaving their inputs unchanged. Objects are immutable; read-only calls may run concurrently. */
#ifndef TORBLES_MATH_DIST_H
#define TORBLES_MATH_DIST_H
#include <stdint.h>
#include <string.h>
#include "math_num.h"
#include "math_budgets.h"
#include "math_scope.h"
#ifdef __cplusplus
extern "C" {
#endif

#define TR_DIST_ABI 1
#define TR_DIST_MAX_DIMS 3
#define TR_DIST_MAX_DEGREE 4
#define TR_DIST_MAX_ROWS UINT64_C(4294967295)        /* rows of a scope a histogram reads (ids and ranks fit 32 bits) */
#define TR_DIST_COUNT_FACTOR 4                         /* AUTO counts a sparse state densely when G x cells <= */
#define TR_DIST_COUNT_SLACK 65536                      /*   4 x selected rows + 65536 */
#define TR_DIST_CHUNK 1024                             /* cells per chunk of the device compaction (at least) */
#define TR_DIST_SCAN_CHUNKS 65536                      /* chunks scanned serially (at most) */
enum { TR_DIST_DENSE = 1, TR_DIST_SPARSE = 2 };
enum { TR_DIST_DOMAIN_SCOPE = 0, TR_DIST_DOMAIN_EXPLICIT = 1 };
enum { TR_DIST_STRATEGY_AUTO = 0, TR_DIST_STRATEGY_COUNT = 1, TR_DIST_STRATEGY_SORT = 2,
       TR_DIST_STRATEGY_DIRECT = 3,   /* a dense state counted in place */
       TR_DIST_STRATEGY_DERIVED = 4,  /* converted, merged, rolled up, projected */
       TR_DIST_STRATEGY_DEVICE = 5 }; /* adopted from the CUDA path */
enum { TR_DIST_NEED_CPU = 0, TR_DIST_NEED_DEVICE = 1, TR_DIST_NEED_ADOPT = 2 };
enum { TR_DIST_MASS_COUNT = 1, TR_DIST_MASS_WEIGHT = 2 };
enum { TR_DIST_OP_RANGE = 1, TR_DIST_OP_MOMENT = 2, TR_DIST_OP_QUANTILE = 3, TR_DIST_OP_RANK = 4,
       TR_DIST_OP_TRIMMED = 5, TR_DIST_OP_TRIMMED_MEAN = 6 };
enum { TR_DIST_LO_CLOSED = 1, TR_DIST_HI_CLOSED = 2, TR_DIST_LO_NONE = 4, TR_DIST_HI_NONE = 8 };
enum { TR_DIST_Q_DISC = 1, TR_DIST_Q_CONT = 2, TR_DIST_Q_LOWER = 3, TR_DIST_Q_HIGHER = 4 };

/* what to build (96 bytes) */
typedef struct {
    int32_t k;                  /* coordinates, 1..3 */
    int32_t col[3];             /* the scope column of coordinate c (c < k; the rest -1) */
    int32_t weight;             /* the scope column of the weight, or -1 */
    int32_t repr;               /* TR_DIST_DENSE / TR_DIST_SPARSE */
    int32_t domain;             /* dense: TR_DIST_DOMAIN_SCOPE / _EXPLICIT (sparse: ignored) */
    int32_t nullslot[3];        /* explicit domain: 1 = coordinate c has a NULL slot */
    int64_t lo[3], hi[3];       /* explicit domain: inclusive logical bounds (hi = lo - 1: no value) */
    int32_t strategy;           /* sparse on the CPU: TR_DIST_STRATEGY_AUTO / _COUNT / _SORT */
    int32_t reserved;           /* 0 */
} tr_dist_spec;

/* The states of a histogram (192 bytes): the read-only view of a built one (tr_dist_hist_get_states: pointers into
 * it, valid while it lives) or the output a CUDA build writes (host arrays of `capacity` entries). */
typedef struct {
    uint64_t ngroups;           /* G */
    uint64_t ncells;            /* dense: G x cells; sparse: the occupied cells (written) */
    uint64_t capacity;          /* entries each cell array holds (>= ncells) */
    uint64_t cells;             /* dense: cells per group; sparse: 0 */
    int32_t k, repr, weighted, reserved;
    int64_t lo[3];              /* dense: the domain */
    uint64_t span[3];           /* dense: values of coordinate c */
    uint64_t dim[3];            /* dense: span + nullslot (the NULL slot is index span) */
    uint64_t *rows;             /* [G] selected rows of each group */
    uint64_t *offset;           /* sparse: [G + 1] (offset[G] = ncells); dense: NULL */
    int64_t *value[3];          /* sparse: [capacity] coordinate c of each cell, 0 where NULL; dense: NULL */
    uint8_t *nulls;             /* sparse: [capacity] bit c: coordinate c is NULL; dense: NULL */
    uint64_t *count;            /* [capacity] rows of each cell */
    uint64_t *wcount;           /* weighted: [capacity] rows of the cell whose weight is present */
    tr_math_i128 *wsum;         /* weighted: [capacity] exact sum of the present weights */
} tr_dist_states;

/* a histogram's description (232 bytes) */
typedef struct {
    uint64_t ngroups, present, rows;   /* groups, groups with rows, selected rows */
    uint64_t ncells, occupied;         /* represented cells, cells with rows */
    uint64_t max_cells;                /* dense: cells per group; sparse: the largest group's cells */
    uint64_t bytes;                    /* bytes owned */
    uint64_t scans, passes;            /* passes over the scope's rows / over the scratch when it was built */
    uint64_t digest[2];                /* the scope descriptor's digest (0 when unscoped) */
    int64_t lo[3], hi[3];              /* dense: the domain; sparse: extremes of the occupied present values */
    uint64_t dim[3];                   /* dense: span + nullslot; sparse: 0 */
    int32_t k, repr, weighted, scoped;
    int32_t col[3], weight;            /* the scope columns it read (-1 unused or unknown) */
    int32_t scale[3], wscale;          /* decimal scales */
    int32_t nullslot[3], strategy;     /* dense: NULL slots; TR_DIST_STRATEGY_* of its construction */
    int32_t wnonneg, reserved;         /* weighted: every present weight is >= 0 */
} tr_dist_hist_info;

/* ---------------------------------------------------------------- the prefix state (DIST-002) */
/* 72 bytes; pointers into a prefix state, or device copies (tr_mgpu_dist_prefix_upload) */
typedef struct {
    uint64_t ngroups, nsupport;        /* groups, support entries over all groups */
    int32_t degree, mass, scale, wscale;   /* moments 0..degree, TR_DIST_MASS_*, value and mass scales */
    const uint64_t *rows;              /* [G] selected rows */
    const uint64_t *nulls;             /* [G] rows whose value is NULL */
    const uint64_t *offset;            /* [G + 1] the group's support entries */
    const int64_t *value;              /* [nsupport] increasing within a group */
    const tr_math_i128 *prefix;        /* [nsupport x (degree + 1)]: P_k[i] at prefix[i * (degree + 1) + k] */
} tr_dist_prefix_view;

typedef struct {
    uint64_t ngroups, present, nsupport, max_support;   /* groups, groups with rows, entries, the largest group's */
    uint64_t rows, nulls, bytes;
    uint64_t digest[2];                /* the histogram's scope digest (0 when unscoped) */
    int32_t degree, mass, scale, wscale;
    int32_t scoped, reserved;
} tr_dist_prefix_info;                 /* 96 bytes */

/* one request (64 bytes) and its answer (104 bytes); see the header comment */
typedef struct {
    int32_t op, flags;                 /* TR_DIST_OP_*; TR_DIST_LO_* | TR_DIST_HI_* for RANGE / MOMENT */
    uint64_t group;
    int64_t a_num, a_den;              /* lower bound / level p / rank r1 (den 1) */
    int64_t b_num, b_den;              /* upper bound / rank r2 */
    int32_t power, rule;               /* k for MOMENT / TRIMMED*; TR_DIST_Q_* for QUANTILE */
    int64_t reserved;                  /* 0 */
} tr_dist_request;

typedef struct { uint64_t w[4]; } tr_dist_i256;   /* signed 256-bit two's complement, w[0] least significant */

typedef struct {
    int32_t status;                    /* 0, or this request's error code */
    int32_t empty;                     /* 1: no answer (SQL NULL): an empty mass, no rank */
    uint64_t lo, hi;                   /* positions within the group's support: RANGE / MOMENT the entries [lo, hi);
                                          QUANTILE the entries of the order statistics used; RANK / TRIMMED* the
                                          entries of the first and last rank */
    tr_math_i128 mass;                 /* RANGE / MOMENT: mass in range; QUANTILE / RANK: the rank used (lower one);
                                          TRIMMED*: the ranks summed */
    tr_math_i128 total;                /* the group's total mass M */
    tr_dist_i256 num;                  /* the answer: num / den */
    tr_math_i128 den;                  /* > 0 */
} tr_dist_result;

/* ---------------------------------------------------------------- exact 256-bit helpers (CPU code and kernels) */
TR_NUM_HD tr_dist_i256 tr_dist_w_zero(void) { tr_dist_i256 r; r.w[0] = r.w[1] = r.w[2] = r.w[3] = 0; return r; }
TR_NUM_HD tr_dist_i256 tr_dist_w_from_i128(__int128 v) {
    tr_dist_i256 r;
    unsigned __int128 u = (unsigned __int128)v;
    r.w[0] = (uint64_t)u; r.w[1] = (uint64_t)(u >> 64);
    r.w[2] = r.w[3] = v < 0 ? UINT64_MAX : 0;
    return r;
}
TR_NUM_HD int tr_dist_w_negative(tr_dist_i256 a) { return (int)(a.w[3] >> 63); }
TR_NUM_HD int tr_dist_w_is_zero(tr_dist_i256 a) { return !(a.w[0] | a.w[1] | a.w[2] | a.w[3]); }
/* a + b: 1, or 0 on signed overflow (*r unchanged) */
TR_NUM_HD int tr_dist_w_add(tr_dist_i256 a, tr_dist_i256 b, tr_dist_i256 *r) {
    tr_dist_i256 s;
    uint64_t c = 0;
    for (int i = 0; i < 4; i++) {
        uint64_t x = a.w[i] + c, c1 = x < c, y = x + b.w[i], c2 = y < x;
        s.w[i] = y; c = c1 | c2;
    }
    int sa = tr_dist_w_negative(a);
    if (sa == tr_dist_w_negative(b) && tr_dist_w_negative(s) != sa) return 0;
    *r = s; return 1;
}
/* a - b: 1, or 0 on signed overflow */
TR_NUM_HD int tr_dist_w_sub(tr_dist_i256 a, tr_dist_i256 b, tr_dist_i256 *r) {
    tr_dist_i256 s;
    uint64_t borrow = 0;
    for (int i = 0; i < 4; i++) {
        uint64_t x = a.w[i] - b.w[i], b1 = a.w[i] < b.w[i], y = x - borrow, b2 = x < borrow;
        s.w[i] = y; borrow = b1 | b2;
    }
    int sa = tr_dist_w_negative(a);
    if (sa != tr_dist_w_negative(b) && tr_dist_w_negative(s) != sa) return 0;
    *r = s; return 1;
}
/* two's complement negation of the 256-bit pattern (exact for every value but -2^255) */
TR_NUM_HD tr_dist_i256 tr_dist_w_negate_bits(tr_dist_i256 a) {
    tr_dist_i256 r;
    uint64_t c = 1;
    for (int i = 0; i < 4; i++) { uint64_t x = ~a.w[i] + c; c = c && x == 0; r.w[i] = x; }
    return r;
}
/* the unsigned 256-bit product of two unsigned 128-bit integers */
TR_NUM_HD tr_dist_i256 tr_dist_w_mul_u128(unsigned __int128 x, unsigned __int128 y) {
    uint64_t x0 = (uint64_t)x, x1 = (uint64_t)(x >> 64), y0 = (uint64_t)y, y1 = (uint64_t)(y >> 64);
    unsigned __int128 p00 = (unsigned __int128)x0 * y0, p01 = (unsigned __int128)x0 * y1;
    unsigned __int128 p10 = (unsigned __int128)x1 * y0, p11 = (unsigned __int128)x1 * y1;
    unsigned __int128 mid = (p00 >> 64) + (uint64_t)p01 + (uint64_t)p10;
    unsigned __int128 high = (p01 >> 64) + (p10 >> 64) + (uint64_t)p11 + (mid >> 64);
    tr_dist_i256 r;
    r.w[0] = (uint64_t)p00; r.w[1] = (uint64_t)mid; r.w[2] = (uint64_t)high;
    r.w[3] = (uint64_t)(p11 >> 64) + (uint64_t)(high >> 64);
    return r;
}
TR_NUM_HD unsigned __int128 tr_dist_mag128(__int128 a) {
    return a < 0 ? (unsigned __int128)0 - (unsigned __int128)a : (unsigned __int128)a;
}
/* the exact product of two signed 128-bit integers (|a b| <= 2^254 always fits) */
TR_NUM_HD tr_dist_i256 tr_dist_w_mul128(__int128 a, __int128 b) {
    tr_dist_i256 p = tr_dist_w_mul_u128(tr_dist_mag128(a), tr_dist_mag128(b));
    return ((a < 0) != (b < 0)) ? tr_dist_w_negate_bits(p) : p;
}
/* a * b for a 256-bit a and a 64-bit b: 1, or 0 on overflow */
TR_NUM_HD int tr_dist_w_mul64(tr_dist_i256 a, int64_t b, tr_dist_i256 *r) {
    int na = tr_dist_w_negative(a), nb = b < 0;
    tr_dist_i256 m = na ? tr_dist_w_negate_bits(a) : a;          /* |a| as an unsigned pattern (2^255 for the min) */
    uint64_t ub = tr_abs_u64(b), carry = 0;
    tr_dist_i256 p;
    for (int i = 0; i < 4; i++) {
        unsigned __int128 t = (unsigned __int128)m.w[i] * ub + carry;
        p.w[i] = (uint64_t)t; carry = (uint64_t)(t >> 64);
    }
    if (carry) return 0;
    int neg = na != nb && !tr_dist_w_is_zero(p);
    if (p.w[3] >> 63) {                                            /* |product| >= 2^255: only -2^255 fits */
        if (!neg || p.w[3] != (UINT64_C(1) << 63) || p.w[2] || p.w[1] || p.w[0]) return 0;
        *r = p; return 1;
    }
    *r = neg ? tr_dist_w_negate_bits(p) : p;
    return 1;
}
/* -1, 0, 1 */
TR_NUM_HD int tr_dist_w_cmp(tr_dist_i256 a, tr_dist_i256 b) {
    int na = tr_dist_w_negative(a), nb = tr_dist_w_negative(b);
    if (na != nb) return na ? -1 : 1;
    for (int i = 3; i >= 0; i--) if (a.w[i] != b.w[i]) return a.w[i] < b.w[i] ? -1 : 1;
    return 0;
}
/* floor(x * a / b) and the remainder, for x < 2^128, a <= b, 1 <= b < 2^64 (the quotient is <= x) */
TR_NUM_HD void tr_dist_muldiv(unsigned __int128 x, uint64_t a, uint64_t b, unsigned __int128 *q, uint64_t *rem) {
    tr_dist_i256 p = tr_dist_w_mul_u128(x, (unsigned __int128)a);  /* < 2^192 */
    uint64_t r = 0, qw[3];
    for (int i = 2; i >= 0; i--) {
        unsigned __int128 cur = ((unsigned __int128)r << 64) | p.w[i];
        qw[i] = (uint64_t)(cur / b); r = (uint64_t)(cur % b);
    }
    *q = ((unsigned __int128)qw[1] << 64) | qw[0];
    *rem = r;
}
TR_NUM_HD __int128 tr_dist_floor_div(int64_t n, int64_t d) {      /* d >= 1 */
    __int128 q = (__int128)n / d;
    if ((__int128)n % d != 0 && n < 0) q -= 1;
    return q;
}
TR_NUM_HD __int128 tr_dist_ceil_div(int64_t n, int64_t d) {       /* d >= 1 */
    __int128 q = (__int128)n / d;
    if ((__int128)n % d != 0 && n > 0) q += 1;
    return q;
}

/* ---------------------------------------------------------------- the build plan (host: CPU and CUDA host code) */
/* The resolved spec over a scope view: coordinates, weight, the dense domain (also the counting scratch of a sparse
 * build), strides. cells = TR_BUDGET_UNKNOWN when the product of the dims does not fit 64 bits. Passed by value to the
 * kernels (no pointers). */
typedef struct {
    int32_t k, repr, weighted, strategy;
    int32_t col[3], weight;
    int32_t nullslot[3], scale[3], wscale, wnonneg;
    int64_t lo[3];
    uint64_t span[3], dim[3], stride[3], cells;
} tr_dist_plan;

static inline int tr_dist_plan_col_ok(const tr_scope_view *v, int32_t c) { return c >= 0 && c < v->ncols; }
/* 0, or TR_MATH_INVALID (malformed), _UNSUPPORTED (k outside 1..3, more rows than TR_DIST_MAX_ROWS),
 * _DOMAIN_ERROR (an explicit domain without a selected value or NULL), _OVERFLOW (weights cannot be summed exactly) */
static inline int32_t tr_dist_plan_make(const tr_scope_view *v, const tr_dist_spec *s, tr_dist_plan *p) {
    if (!v || !s || !p) return -1;
    memset(p, 0, sizeof *p);
    if (s->k < 1 || s->k > TR_DIST_MAX_DIMS) return -12;
    if ((s->repr != TR_DIST_DENSE && s->repr != TR_DIST_SPARSE) || s->reserved ||
        (s->domain != TR_DIST_DOMAIN_SCOPE && s->domain != TR_DIST_DOMAIN_EXPLICIT) ||
        s->strategy < TR_DIST_STRATEGY_AUTO || s->strategy > TR_DIST_STRATEGY_SORT ||
        (s->weight != -1 && !tr_dist_plan_col_ok(v, s->weight))) return -1;
    if (v->rows > TR_DIST_MAX_ROWS) return -12;
    p->k = s->k; p->repr = s->repr; p->strategy = s->strategy; p->weight = s->weight; p->weighted = s->weight >= 0;
    p->wnonneg = 1;
    for (int c = 0; c < 3; c++) { p->col[c] = -1; p->nullslot[c] = 0; }
    int explicit_domain = s->repr == TR_DIST_DENSE && s->domain == TR_DIST_DOMAIN_EXPLICIT;
    uint64_t cells = 1;
    for (int c = 0; c < s->k; c++) {
        if (!tr_dist_plan_col_ok(v, s->col[c])) return -1;
        const tr_scope_view_column *vc = &v->col[s->col[c]];
        p->col[c] = s->col[c]; p->scale[c] = vc->scale;
        int has_null = vc->valid != NULL;                 /* a selected value is NULL (math_scope.h) */
        if (explicit_domain) {
            if ((s->nullslot[c] != 0 && s->nullslot[c] != 1) || (__int128)s->hi[c] < (__int128)s->lo[c] - 1) return -1;
            if (vc->nvalid && (vc->min < s->lo[c] || vc->max > s->hi[c])) return -6;       /* TR_MATH_DOMAIN_ERROR */
            if (has_null && !s->nullslot[c]) return -6;
            p->lo[c] = s->lo[c];
            p->span[c] = s->hi[c] >= s->lo[c] ? (uint64_t)s->hi[c] - (uint64_t)s->lo[c] + 1 : 0;
            if (s->hi[c] >= s->lo[c] && p->span[c] == 0) p->span[c] = TR_BUDGET_UNKNOWN;   /* 2^64 values */
            p->nullslot[c] = s->nullslot[c];
        } else {
            p->lo[c] = vc->nvalid ? vc->min : 0;
            p->span[c] = vc->nvalid ? (uint64_t)vc->max - (uint64_t)vc->min + 1 : 0;
            if (vc->nvalid && p->span[c] == 0) p->span[c] = TR_BUDGET_UNKNOWN;
            p->nullslot[c] = has_null;
        }
        p->dim[c] = tr_budget_add(p->span[c], (uint64_t)p->nullslot[c]);
        if (p->span[c] == TR_BUDGET_UNKNOWN) p->dim[c] = TR_BUDGET_UNKNOWN;
    }
    for (int c = s->k - 1; c >= 0; c--) {
        p->stride[c] = cells;
        cells = p->dim[c] == 0 ? 0 : (cells == 0 ? 0 : tr_budget_mul(cells, p->dim[c]));
    }
    for (int c = 0; c < s->k; c++) if (p->dim[c] == 0) cells = 0;
    p->cells = cells;
    if (p->weighted) {
        const tr_scope_view_column *w = &v->col[s->weight];
        p->wscale = w->scale;
        p->wnonneg = !w->nvalid || w->min >= 0;
        if (w->nvalid) {
            uint64_t m = tr_abs_u64(w->min) > tr_abs_u64(w->max) ? tr_abs_u64(w->min) : tr_abs_u64(w->max);
            unsigned __int128 limit = ((unsigned __int128)1 << 127) - 1;
            if (m && (unsigned __int128)v->selected > limit / m) return -3;   /* TR_MATH_OVERFLOW */
        }
    }
    return 0;
}

/* the cell of a selected row within its group's dense domain (the row's values lie in the domain) */
TR_NUM_HD uint64_t tr_dist_row_cell(const tr_scope_view *v, const tr_dist_plan *p, uint64_t row) {
    uint64_t idx = 0;
    for (int c = 0; c < p->k; c++) {
        int32_t col = p->col[c];
        uint64_t i = tr_scope_present(v, col, row) ? (uint64_t)tr_scope_value(v, col, row) - (uint64_t)p->lo[c]
                                                   : p->span[c];
        idx += i * p->stride[c];
    }
    return idx;
}
/* coordinate c of dense cell idx (within a group): its value, or *null = 1 for the NULL slot */
TR_NUM_HD int64_t tr_dist_plan_coord(const tr_dist_plan *p, uint64_t idx, int c, int *null) {
    uint64_t i = (idx / p->stride[c]) % p->dim[c];
    *null = i == p->span[c];
    return *null ? 0 : (int64_t)((uint64_t)p->lo[c] + i);
}
/* the dense domain of a states struct as a plan (for decoding cells of a built dense histogram) */
TR_NUM_HD void tr_dist_states_plan(const tr_dist_states *s, tr_dist_plan *p) {
    uint64_t cells = 1;
    p->k = s->k;
    for (int c = s->k - 1; c >= 0; c--) {
        p->lo[c] = s->lo[c]; p->span[c] = s->span[c]; p->dim[c] = s->dim[c];
        p->nullslot[c] = (int32_t)(s->dim[c] - s->span[c]);
        p->stride[c] = cells; cells *= s->dim[c];
    }
    p->cells = s->cells;
}

/* the canonical spec: unused coordinates -1, the domain fields only for an explicit dense domain, the strategy only
 * for a sparse state (two specs that build the same state normalize equal) */
static inline void tr_dist_spec_normalize(const tr_dist_spec *s, tr_dist_spec *n) {
    *n = *s;
    for (int c = 0; c < 3; c++) {
        if (c >= s->k) { n->col[c] = -1; n->lo[c] = n->hi[c] = 0; n->nullslot[c] = 0; }
        if (s->repr != TR_DIST_DENSE || s->domain != TR_DIST_DOMAIN_EXPLICIT) { n->lo[c] = n->hi[c] = 0; n->nullslot[c] = 0; }
    }
    if (s->repr != TR_DIST_DENSE) n->domain = TR_DIST_DOMAIN_SCOPE;
    if (s->repr == TR_DIST_DENSE) n->strategy = TR_DIST_STRATEGY_AUTO;
}
/* the bytes of a device copy of a prefix state: rows, nulls, offsets, values, prefix sums */
static inline uint64_t tr_dist_prefix_copy_bytes(const tr_dist_prefix_view *v) {
    uint64_t b = tr_budget_add(tr_budget_mul(v->ngroups, 16), tr_budget_mul(tr_budget_add(v->ngroups, 1), 8));
    return tr_budget_add(b, tr_budget_mul(v->nsupport, tr_budget_add(8, tr_budget_mul((uint64_t)v->degree + 1, 16))));
}
/* the distinct scope columns a plan reads (coordinates, then the weight): their count */
static inline int tr_dist_plan_columns(const tr_dist_plan *p, int32_t cols[4]) {
    int n = 0;
    for (int c = 0; c <= p->k; c++) {
        int32_t col = c < p->k ? p->col[c] : p->weight;
        if (col < 0) continue;
        int seen = 0;
        for (int i = 0; i < n; i++) seen |= cols[i] == col;
        if (!seen) cols[n++] = col;
    }
    return n;
}
/* bytes of one cell's counters: count, and wcount + wsum when weighted */
static inline uint64_t tr_dist_cell_bytes(int weighted) { return weighted ? 32 : 8; }
/* chunk length and number of chunks of the device compaction over `total` dense cells */
static inline uint64_t tr_dist_chunk_len(uint64_t total) {
    uint64_t c = (total + TR_DIST_SCAN_CHUNKS - 1) / TR_DIST_SCAN_CHUNKS;
    return c < TR_DIST_CHUNK ? TR_DIST_CHUNK : c;
}
/* The needs of tr_mgpu_dist_hist_build: variables = k (+1), support = cells per group of the device array (the
 * output when dense, the counting scratch when sparse), states = groups x those cells, bytes = its device copies (the
 * selection, group ids, the columns read with their validity, the row counts, the cell counters; sparse: the output
 * of `capacity` cells, the offsets and the chunk counters). capacity = min(selected rows, G x cells): a bound on the
 * occupied cells. */
static inline void tr_dist_device_need(const tr_scope_view *v, const tr_dist_plan *p, tr_math_need *need,
                                       uint64_t *capacity) {
    for (int i = 0; i < TR_BUDGET_RESOURCES; i++) need->need[i] = 0;
    uint64_t n = v->rows, G = v->ngroups, total = tr_budget_mul(G, p->cells), cb = tr_dist_cell_bytes(p->weighted);
    uint64_t bytes = tr_budget_add(tr_budget_bitmap_bytes(n), tr_budget_mul(G, 8));
    if (v->groups) bytes = tr_budget_add(bytes, tr_budget_mul(n, 4));
    int32_t cols[4];
    int nc = tr_dist_plan_columns(p, cols);
    for (int i = 0; i < nc; i++) {
        bytes = tr_budget_add(bytes, tr_budget_mul(n, v->col[cols[i]].values ? 8 : 4));
        if (v->col[cols[i]].valid) bytes = tr_budget_add(bytes, n);
    }
    bytes = tr_budget_add(bytes, tr_budget_mul(total, cb));
    uint64_t cap = total;
    if (p->repr == TR_DIST_SPARSE) {
        cap = v->selected < total ? v->selected : total;
        uint64_t chunks = total == TR_BUDGET_UNKNOWN ? TR_BUDGET_UNKNOWN : (total + tr_dist_chunk_len(total) - 1) / tr_dist_chunk_len(total);
        bytes = tr_budget_add(bytes, tr_budget_mul(cap, tr_budget_add(tr_budget_mul((uint64_t)p->k, 8), 1 + cb)));
        bytes = tr_budget_add(bytes, tr_budget_add(tr_budget_mul(tr_budget_add(G, 1), 8), tr_budget_add(tr_budget_mul(chunks, 16), 8)));
    }
    need->need[TR_BUDGET_VARIABLES] = (uint64_t)(p->k + p->weighted);
    need->need[TR_BUDGET_SUPPORT] = p->cells;
    need->need[TR_BUDGET_STATES] = total;
    need->need[TR_BUDGET_BYTES] = bytes;
    if (capacity) *capacity = cap;
}

/* ---------------------------------------------------------------- prefix requests (CPU loop and CUDA kernel) */
TR_NUM_HD __int128 tr_dist_px(const tr_dist_prefix_view *v, uint64_t i, int k) {
    return tr_i128_get(v->prefix[i * (uint64_t)(v->degree + 1) + (uint64_t)k]);
}
/* the exclusive prefix before position i of the group starting at b (0 at the group's start), as a bit pattern */
TR_NUM_HD unsigned __int128 tr_dist_ex(const tr_dist_prefix_view *v, uint64_t b, uint64_t i, int k) {
    return i == b ? 0 : (unsigned __int128)tr_dist_px(v, i - 1, k);
}
TR_NUM_HD uint64_t tr_dist_lower_bound(const tr_dist_prefix_view *v, uint64_t b, uint64_t e, __int128 x) {
    while (b < e) { uint64_t m = b + (e - b) / 2; if ((__int128)v->value[m] < x) b = m + 1; else e = m; }
    return b;
}
TR_NUM_HD uint64_t tr_dist_upper_bound(const tr_dist_prefix_view *v, uint64_t b, uint64_t e, __int128 x) {
    while (b < e) { uint64_t m = b + (e - b) / 2; if ((__int128)v->value[m] <= x) b = m + 1; else e = m; }
    return b;
}
/* the entry holding rank r (1 <= r <= M): the first i with P_0[i] >= r */
TR_NUM_HD uint64_t tr_dist_rank_entry(const tr_dist_prefix_view *v, uint64_t b, uint64_t e, unsigned __int128 r) {
    while (b < e) { uint64_t m = b + (e - b) / 2; if ((unsigned __int128)tr_dist_px(v, m, 0) < r) b = m + 1; else e = m; }
    return b;
}
/* x^k as a bit pattern (exact when the prefix proof covers k) */
TR_NUM_HD unsigned __int128 tr_dist_powk(int64_t x, int k) {
    unsigned __int128 r = 1, u = (unsigned __int128)(__int128)x;
    for (int i = 0; i < k; i++) r *= u;
    return r;
}
/* sum of y_r^k over ranks 1..R (R <= M) of the group [b, e), as a bit pattern; *at = the entry of rank R */
TR_NUM_HD unsigned __int128 tr_dist_rank_sum(const tr_dist_prefix_view *v, uint64_t b, uint64_t e, unsigned __int128 R,
                                             int k, uint64_t *at) {
    if (R == 0) { *at = b; return 0; }
    uint64_t i = tr_dist_rank_entry(v, b, e, R);
    *at = i;
    unsigned __int128 before = tr_dist_ex(v, b, i, 0);
    return tr_dist_ex(v, b, i, k) + (R - before) * tr_dist_powk(v->value[i], k);
}
/* the integer bounds L <= x <= U of an interval request (empty when L > U) */
TR_NUM_HD void tr_dist_bounds(const tr_dist_request *q, __int128 *L, __int128 *U) {
    const __int128 big = (__int128)1 << 100;
    *L = -big; *U = big;
    if (!(q->flags & TR_DIST_LO_NONE))
        *L = (q->flags & TR_DIST_LO_CLOSED) ? tr_dist_ceil_div(q->a_num, q->a_den) : tr_dist_floor_div(q->a_num, q->a_den) + 1;
    if (!(q->flags & TR_DIST_HI_NONE))
        *U = (q->flags & TR_DIST_HI_CLOSED) ? tr_dist_floor_div(q->b_num, q->b_den) : tr_dist_ceil_div(q->b_num, q->b_den) - 1;
}
TR_NUM_HD void tr_dist_result_init(tr_dist_result *r) {
    r->status = 0; r->empty = 0; r->lo = 0; r->hi = 0;
    r->mass = tr_i128_set(0); r->total = tr_i128_set(0);
    r->num = tr_dist_w_zero(); r->den = tr_i128_set(1);
}
/* one request against the prefix view: the shared evaluation of the CPU loop and the CUDA kernel */
TR_NUM_HD void tr_dist_eval(const tr_dist_prefix_view *v, const tr_dist_request *q, tr_dist_result *r) {
    tr_dist_result_init(r);
    if (q->group >= v->ngroups || q->reserved) { r->status = -1; return; }
    const uint64_t b = v->offset[q->group], e = v->offset[q->group + 1];
    const unsigned __int128 M = e > b ? (unsigned __int128)tr_dist_px(v, e - 1, 0) : 0;
    r->total = tr_i128_set((__int128)M);
    switch (q->op) {
    case TR_DIST_OP_RANGE:
    case TR_DIST_OP_MOMENT: {
        if (q->flags & ~(TR_DIST_LO_CLOSED | TR_DIST_HI_CLOSED | TR_DIST_LO_NONE | TR_DIST_HI_NONE)) { r->status = -1; return; }
        if ((!(q->flags & TR_DIST_LO_NONE) && q->a_den < 1) || (!(q->flags & TR_DIST_HI_NONE) && q->b_den < 1)) {
            r->status = -1; return;
        }
        int k = q->op == TR_DIST_OP_MOMENT ? q->power : 0;
        if (k < 0 || k > TR_DIST_MAX_DEGREE) { r->status = -1; return; }
        if (k > v->degree) { r->status = -12; return; }
        __int128 L, U;
        tr_dist_bounds(q, &L, &U);
        uint64_t lo = tr_dist_lower_bound(v, b, e, L), hi = L > U ? lo : tr_dist_upper_bound(v, b, e, U);
        if (hi < lo) hi = lo;
        r->lo = lo - b; r->hi = hi - b;
        unsigned __int128 mass = tr_dist_ex(v, b, hi, 0) - tr_dist_ex(v, b, lo, 0);
        r->mass = tr_i128_set((__int128)mass);
        if (q->op == TR_DIST_OP_RANGE) {
            if (M == 0) { r->empty = 1; return; }
            r->num = tr_dist_w_from_i128((__int128)mass); r->den = tr_i128_set((__int128)M);
        } else {
            r->num = tr_dist_w_from_i128((__int128)(tr_dist_ex(v, b, hi, k) - tr_dist_ex(v, b, lo, k)));
        }
        return;
    }
    case TR_DIST_OP_QUANTILE: {
        if (q->a_den < 1 || q->a_num < 0 || q->a_num > q->a_den || q->rule < TR_DIST_Q_DISC || q->rule > TR_DIST_Q_HIGHER) {
            r->status = -1; return;
        }
        if (M == 0) { r->empty = 1; return; }
        uint64_t a = (uint64_t)q->a_num, d = (uint64_t)q->a_den, rem;
        unsigned __int128 j;
        if (q->rule == TR_DIST_Q_DISC) {
            tr_dist_muldiv(M, a, d, &j, &rem);                    /* ceil(p M), at least 1 */
            unsigned __int128 rank = j + (rem != 0);
            if (rank < 1) rank = 1;
            uint64_t i = tr_dist_rank_entry(v, b, e, rank);
            r->lo = r->hi = i - b; r->mass = tr_i128_set((__int128)rank);
            r->num = tr_dist_w_from_i128(v->value[i]);
            return;
        }
        tr_dist_muldiv(M - 1, a, d, &j, &rem);                    /* h = p (M - 1) = j + rem / d */
        uint64_t i1 = tr_dist_rank_entry(v, b, e, j + 1);
        r->lo = r->hi = i1 - b; r->mass = tr_i128_set((__int128)(j + 1));
        if (rem == 0 || q->rule == TR_DIST_Q_LOWER) { r->num = tr_dist_w_from_i128(v->value[i1]); return; }
        uint64_t i2 = tr_dist_rank_entry(v, b, e, j + 2);
        r->hi = i2 - b;
        if (q->rule == TR_DIST_Q_HIGHER) { r->num = tr_dist_w_from_i128(v->value[i2]); return; }
        /* (d - rem) y_lo + rem y_hi over d: each term below 2^126 in magnitude */
        __int128 n = (__int128)(d - rem) * v->value[i1] + (__int128)rem * v->value[i2];
        r->num = tr_dist_w_from_i128(n); r->den = tr_i128_set((__int128)d);
        return;
    }
    case TR_DIST_OP_RANK: {
        if (q->a_den != 1 || q->a_num < 1) { r->status = -1; return; }
        if ((unsigned __int128)q->a_num > M) { r->empty = 1; return; }
        uint64_t i = tr_dist_rank_entry(v, b, e, (unsigned __int128)q->a_num);
        r->lo = r->hi = i - b; r->mass = tr_i128_set(q->a_num);
        r->num = tr_dist_w_from_i128(v->value[i]);
        return;
    }
    case TR_DIST_OP_TRIMMED:
    case TR_DIST_OP_TRIMMED_MEAN: {
        int k = q->power;
        if (q->a_den != 1 || q->b_den != 1 || k < 0 || k > TR_DIST_MAX_DEGREE || q->a_num < 1 || q->b_num < 0 ||
            (__int128)q->a_num > (__int128)q->b_num + 1) { r->status = -1; return; }
        if (k > v->degree) { r->status = -12; return; }
        if ((unsigned __int128)q->b_num > M) { r->status = -1; return; }
        uint64_t i1, i2;
        unsigned __int128 hi = tr_dist_rank_sum(v, b, e, (unsigned __int128)q->b_num, k, &i2);
        unsigned __int128 lo = tr_dist_rank_sum(v, b, e, (unsigned __int128)(q->a_num - 1), k, &i1);
        uint64_t count = (uint64_t)((__int128)q->b_num - q->a_num + 1);
        r->mass = tr_i128_set((__int128)count);
        if (count) { r->lo = tr_dist_rank_entry(v, b, e, (unsigned __int128)q->a_num) - b; r->hi = i2 - b; }
        r->num = tr_dist_w_from_i128((__int128)(hi - lo));
        if (q->op == TR_DIST_OP_TRIMMED_MEAN) {
            if (!count) { r->empty = 1; r->num = tr_dist_w_zero(); return; }
            r->den = tr_i128_set((__int128)count);
        }
        (void)i1;
        return;
    }
    default:
        r->status = -1;
    }
}

/* ---------------------------------------------------------------- histograms (libalg) */
typedef struct tr_dist_hist_s tr_dist_hist;
typedef struct tr_dist_hist_builder_s tr_dist_hist_builder;
typedef struct tr_dist_prefix_s tr_dist_prefix;

tr_dist_hist_builder *tr_dist_hist_init(const tr_scope *, const tr_dist_spec *, const tr_math_budget *budget,
                                        tr_math_refusal *refusal, int32_t *error);
int32_t tr_dist_hist_accumulate(tr_dist_hist_builder *, uint64_t begin, uint64_t end);
int32_t tr_dist_hist_merge_builder(tr_dist_hist_builder *into, const tr_dist_hist_builder *from);
/* budget: for the final state of a sparse build (NULL = the init budget's defaults) */
tr_dist_hist *tr_dist_hist_finish(tr_dist_hist_builder *, const tr_math_budget *budget, tr_math_refusal *refusal,
                                  int32_t *error);
void tr_dist_hist_builder_free(tr_dist_hist_builder *);
tr_dist_hist *tr_dist_hist_build(const tr_scope *, const tr_dist_spec *, const tr_math_budget *budget,
                                 tr_math_refusal *refusal, int32_t *error);
/* TR_DIST_NEED_CPU: what init checks (a sparse build adds its final state at finish); TR_DIST_NEED_DEVICE: the device
 * copies of tr_mgpu_dist_hist_build; TR_DIST_NEED_ADOPT: the largest adopted state (sparse: one cell per selected
 * row at most). *capacity (may be NULL): the cell capacity the CUDA build's output needs. */
int32_t tr_dist_hist_need(const tr_scope *, const tr_dist_spec *, int32_t target, tr_math_need *need, uint64_t *capacity);
/* adopts states computed elsewhere (the CUDA readback) as the histogram of origin under spec, after checks */
tr_dist_hist *tr_dist_hist_from_states(const tr_scope *origin, const tr_dist_spec *spec, const tr_dist_states *,
                                       const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error);
/* dense over the domain lo/hi/nullslot (NULL: the hull of the occupied cells); TR_MATH_DOMAIN_ERROR when an
 * occupied cell lies outside it */
tr_dist_hist *tr_dist_hist_to_dense(const tr_dist_hist *, const int64_t *lo, const int64_t *hi, const int32_t *nullslot,
                                    const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error);
tr_dist_hist *tr_dist_hist_to_sparse(const tr_dist_hist *, const tr_math_budget *budget, tr_math_refusal *refusal,
                                     int32_t *error);
tr_dist_hist *tr_dist_hist_merge(const tr_dist_hist *, const tr_dist_hist *, const tr_math_budget *budget,
                                 tr_math_refusal *refusal, int32_t *error);
tr_dist_hist *tr_dist_hist_rollup(const tr_dist_hist *, const uint32_t *map, uint64_t coarse_groups,
                                  const tr_scope *origin, const tr_math_budget *budget, tr_math_refusal *refusal,
                                  int32_t *error);
/* keep the coordinates whose bit is set in mask (at least one), in their order */
tr_dist_hist *tr_dist_hist_marginal(const tr_dist_hist *, uint32_t mask, const tr_math_budget *budget,
                                    tr_math_refusal *refusal, int32_t *error);
tr_dist_hist *tr_dist_hist_clone(const tr_dist_hist *, int32_t *error);
/* 1 when both have the same shape (k, representation, weightedness, scales, groups, dense domain) and states */
int32_t tr_dist_hist_equal(const tr_dist_hist *, const tr_dist_hist *);
int32_t tr_dist_hist_get_info(const tr_dist_hist *, tr_dist_hist_info *);
int32_t tr_dist_hist_get_states(const tr_dist_hist *, tr_dist_states *);
void tr_dist_hist_free(tr_dist_hist *);

/* ---------------------------------------------------------------- prefix states (libalg) */
tr_dist_prefix *tr_dist_prefix_build(const tr_dist_hist *, int32_t mass, int32_t degree, const tr_math_budget *budget,
                                     tr_math_refusal *refusal, int32_t *error);
int32_t tr_dist_prefix_need(const tr_dist_hist *, int32_t mass, int32_t degree, tr_math_need *need);
int32_t tr_dist_prefix_get_view(const tr_dist_prefix *, tr_dist_prefix_view *);
int32_t tr_dist_prefix_get_info(const tr_dist_prefix *, tr_dist_prefix_info *);
int32_t tr_dist_prefix_equal(const tr_dist_prefix *, const tr_dist_prefix *);
void tr_dist_prefix_free(tr_dist_prefix *);
/* n requests (output = n, bytes = 0: the caller owns both arrays); 0 or TR_MATH_BUDGET / _INVALID */
int32_t tr_dist_query(const tr_dist_prefix *, const tr_dist_request *requests, uint64_t n, tr_dist_result *out,
                      const tr_math_budget *budget, tr_math_refusal *refusal);
/* the bytes a device copy of the prefix state takes (tr_mgpu_dist_prefix_upload's need) */
uint64_t tr_dist_prefix_device_bytes(const tr_dist_prefix_view *);

/* ---------------------------------------------------------------- the CUDA library (math_dist_gpu.cu)
 * tr_mgpu_dist_hist_build: view = tr_scope_get_view of the scope; out: host arrays, out->ngroups = view->ngroups,
 * out->k / repr / weighted as the spec, out->capacity = the cells they hold (dense: G x cells; sparse: the capacity
 * tr_dist_hist_need reports). Writes exactly the states of tr_dist_hist_build (adopt them with
 * tr_dist_hist_from_states). A prefix state is uploaded once and queried by any number of batches; the device copy is
 * independent of the host state. All return 0 or a native error code (TR_MATH_DEVICE when the device fails). */
typedef struct tr_mgpu_dist_prefix_s tr_mgpu_dist_prefix;
int tr_mgpu_dist_hist_build(const tr_scope_view *view, const tr_dist_spec *spec, const tr_math_budget *budget,
                            tr_dist_states *out, tr_math_refusal *refusal);
tr_mgpu_dist_prefix *tr_mgpu_dist_prefix_upload(const tr_dist_prefix_view *view, const tr_math_budget *budget,
                                                tr_math_refusal *refusal, int *error);
int tr_mgpu_dist_prefix_device_view(const tr_mgpu_dist_prefix *, tr_dist_prefix_view *out);
uint64_t tr_mgpu_dist_prefix_bytes(const tr_mgpu_dist_prefix *);
void tr_mgpu_dist_prefix_free(tr_mgpu_dist_prefix *);
int tr_mgpu_dist_query(const tr_mgpu_dist_prefix *, const tr_dist_request *requests, uint64_t n, tr_dist_result *out,
                       const tr_math_budget *budget, tr_math_refusal *refusal);
#ifdef __cplusplus
}
#endif
#endif
