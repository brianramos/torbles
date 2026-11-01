/* INV-002: constrained minimal interventions ("the cheapest change of the inputs that attains a target").
 * SPDX-License-Identifier: Apache-2.0
 *
 * Every answer is an ALGEBRAIC intervention on the function as given: the change d of declared parameters or
 * features (x = x0 + d) at minimum cost that makes the function attain a target, with no claim about what would
 * happen in the world if the inputs were changed (no causal model is involved; a request for causal semantics is
 * TR_MATH_UNSUPPORTED). Results carry TR_INTERV_CLAIM_ALGEBRAIC.
 *
 * Costs of the change d (rational weights w_i >= 0; L0: w_i > 0): L1 sum w_i |d_i|, LINF max w_i |d_i|, L2 the
 * squared sum w_i d_i^2, L0 the weighted number of changed coordinates sum w_i [d_i != 0].
 * Targets: f(x) op t (op EQ, NE, LT, LE, GT, GE), or a predicate membership change (flip: the complement of
 * "f(x0) op t" is the target: the base point's membership, decided exactly, is reported).
 *
 * 1. Linear targets f(x) = a^T x + a0 in n <= TR_INTERV_MAX_DIMS coordinates, under A x <= b, E x = e, boxes on x and
 *    on d (either side optional) and the polyhedral norm balls ||d||_1 <= r1, ||d||_inf <= rinf: exact LPs (L1: the
 *    variables (d, u), u >= +-d, cost w^T u; LINF: (d, s), +-w_i d_i <= s, cost s) and the exact projection QP (L2,
 *    H = 2 diag(w)) by OPT-002's simplex and active-set methods (math_polyopt.h), KKT-verified there; L0 by
 *    enumerating the supports in increasing weight (an exact feasibility LP each; every support of the minimal
 *    weight is reported, with a witness). A strict target (LT, GT; NE as both) is solved on its closure: the strict
 *    set must be nonempty (an LP), and the infimum is attained iff some optimal point satisfies it strictly (an LP
 *    over the optimal set), else the answer is UNATTAINED with the exact infimum and its boundary point.
 *    Uniqueness of the optimal change is decided exactly (max / min d_k over the optimal set, 2n LPs); a nonunique
 *    answer carries a second optimal change.
 * 2. Univariate polynomial targets (one coordinate; first-slice polynomials or rational coefficients): the roots of
 *    f - t in the domain by INV-001 (exact rational roots, otherwise certified isolating intervals); the feasible set
 *    is a union of points and open intervals between them, each decided by an exact sign; the nearest feasible
 *    points on each side of x0 are compared exactly (a tie between irrational distances is decided by the gcd of
 *    (f - t)(x0 + y) and (f - t)(x0 - y) having a root in both intervals); every minimal change is reported.
 * 3. Finite grids (1..TR_INTERV_GRID_DIMS coordinates, values per coordinate, any first-slice polynomial target and
 *    polynomial or linear constraints): exhaustive search in increasing cost. Blocks of the grid whose UNC-001-style
 *    outward-rounded interval enclosure (FND-009) proves a condition impossible are pruned soundly first; the other
 *    points are ordered by (exact cost, grid index) and examined in that order, chunk by chunk, each condition decided
 *    by its exact sign (an integer polynomial in the cleared numerators, residues modulo the verified primes of
 *    INV-001 and Garner's digits). The first feasible rank gives the minimum; every feasible point of the same cost
 *    is a tie and every one is listed. Past the iteration budget the answer is INCOMPLETE with proven bounds: no
 *    point of cost < lower is feasible (every such point was examined or pruned) and upper is the best found.
 *    The per-point evaluation (tr_interv_grid_point) is shared by the CPU and the CUDA kernels.
 * 4. Aggregates of a query whose rows move across its predicate when a feature is shifted: over the selected rows of
 *    an FND-004 scope, A(d) = COUNT(*), COUNT(v), SUM(v) or AVG(v) WHERE x + d op c (v may be x itself, then shifted
 *    too), SQL semantics (NULL x never qualifies, SUM / AVG of no value is NULL and attains no target). A(d) is
 *    piecewise (constant or linear in d) with the exact breakpoints c - x_k; every piece is decided exactly, so rows
 *    entering and leaving the predicate are part of the objective (never the frozen row set of d = 0). The state (the
 *    sorted distinct x with counts and exact sums, prefix sums) is built once and answers many targets.
 *
 * Budgets (math_budgets.h, checked before allocating; TR_MATH_BUDGET with the refusal):
 *   linear      variables = n, iterations = LP pivots (and L0: supports examined), tensor = the LP tableaux
 *   univariate  degree, iterations = exact evaluations of INV-001
 *   grid        variables = coordinates, support = terms, states = grid points, bytes = the plan (order, residues),
 *               iterations = points examined (TRUNCATES: INCOMPLETE), output = ties listed (TRUNCATES)
 *   aggregate   states = distinct values, bytes = the state
 * Errors: TR_MATH_INVALID, _MEMORY, _OVERFLOW (a cost table or a multimodular bound past its representation),
 * _BUDGET, _UNSUPPORTED (sizes, causal semantics, a linear target of degree > 1), _DOMAIN_ERROR. */
#ifndef TORBLES_MATH_INTERV_H
#define TORBLES_MATH_INTERV_H
#include <stdint.h>
#include <stddef.h>
#include "math_num.h"
#include "math_budgets.h"
#include "math_carriers.h"
#include "math_inv.h"
#include "math_polyopt.h"
#include "math_scope.h"
#include "mathematics.h"
#ifdef __cplusplus
extern "C" {
#endif

#define TR_INTERV_ABI 1
#define TR_INTERV_MAX_DIMS 16            /* coordinates of a linear problem */
#define TR_INTERV_GRID_DIMS 8            /* coordinates of a grid */
#define TR_INTERV_GRID_MAX (UINT64_C(1) << 24)   /* grid points */
#define TR_INTERV_CONDS 16               /* conditions of a grid search: the target and the constraints */
#define TR_INTERV_BLOCKS 4096            /* blocks of the enclosure pre-pass */
#define TR_INTERV_CLAIM_ALGEBRAIC 1

enum { TR_INTERV_L1 = 1, TR_INTERV_LINF = 2, TR_INTERV_L2 = 3, TR_INTERV_L0 = 4 };                 /* costs */
enum { TR_INTERV_EQ = 0, TR_INTERV_NE = 1, TR_INTERV_LT = 2, TR_INTERV_LE = 3, TR_INTERV_GT = 4,
       TR_INTERV_GE = 5 };                                                                        /* comparisons */
enum { TR_INTERV_ALGEBRAIC = 1, TR_INTERV_CAUSAL = 2 };                                           /* semantics */
enum { TR_INTERV_OPTIMAL = 1, TR_INTERV_UNATTAINED = 2, TR_INTERV_INFEASIBLE = 3, TR_INTERV_INCOMPLETE = 4 };
enum { TR_INTERV_M_LP = 1, TR_INTERV_M_QP = 2, TR_INTERV_M_SUPPORTS = 3, TR_INTERV_M_ROOTS = 4, TR_INTERV_M_GRID = 5,
       TR_INTERV_M_AGGREGATE = 6 };                                                               /* methods */
enum { TR_INTERV_REASON_NONE = 0, TR_INTERV_REASON_ITERATIONS = 1, TR_INTERV_REASON_OUTPUT = 2 };
enum { TR_INTERV_COUNT_STAR = 1, TR_INTERV_COUNT = 2, TR_INTERV_SUM = 3, TR_INTERV_AVG = 4 };    /* aggregates */
enum { TR_INTERV_SIGN_NEG = 1, TR_INTERV_SIGN_ZERO = 2, TR_INTERV_SIGN_POS = 4 };                /* allowed signs */
enum { TR_INTERV_PART_LOWER = 0, TR_INTERV_PART_UPPER = 1,   /* bounds of the optimal cost (incomplete: proven) */
       TR_INTERV_PART_BASE = 2,                              /* f(x0), or the aggregate at d = 0 (absent: NULL) */
       TR_INTERV_PART_FACE = 3,                              /* nonunique: a second optimal change d (n) */
       TR_INTERV_PART_FACE_COST = 4,
       TR_INTERV_PARTS = 5 };

/* ------------------------------------------------------------------ structs crossing the ctypes boundary */
typedef struct {
    int32_t n, cost, op, flip;               /* coordinates; TR_INTERV_L1..; comparison; 1: a membership change */
    int32_t m, p, semantics, reserved;       /* rows of A x <= b and E x = e; TR_INTERV_ALGEBRAIC */
    const tr_bigq *const *x0;                /* n: the base point */
    const tr_bigq *const *weight;            /* n (NULL array or entries: 1) */
    const tr_bigq *const *a;                 /* n: f(x) = a^T x + a0 (NULL entries 0) */
    const tr_bigq *a0, *t;                   /* NULL: 0 */
    const tr_bigq *const *A, *const *b;      /* m x n, m (on the point x) */
    const tr_bigq *const *E, *const *e;      /* p x n, p */
    const tr_bigq *const *xlo, *const *xhi;  /* boxes on x and on d: n each, NULL arrays or entries: none */
    const tr_bigq *const *dlo, *const *dhi;
    const tr_bigq *l1, *linf;                /* ||d||_1 <= l1, ||d||_inf <= linf (NULL: none) */
} tr_interv_linear_problem;                  /* 152 bytes */

typedef struct {
    const tr_math_poly *slice;               /* f as a first-slice polynomial in variable slot `variable`, or NULL: */
    int32_t variable, ncoef;                 /* ncoef coefficients (low degree first) */
    const tr_bigq *const *coef;
    const tr_bigq *x0, *t, *weight;          /* the base value, the target (NULL 0), the cost weight (NULL 1) */
    int32_t cost, op, flip, semantics;
    const tr_bigq *dlo, *dhi, *xlo, *xhi;    /* the domain of d and of x (NULL: unbounded) */
    const tr_bigq *width;                    /* refine reported isolating intervals to this width (NULL: as found) */
} tr_interv_univariate_problem;              /* 104 bytes */

typedef struct {
    int32_t ndims, cost, op, flip;
    int32_t ncons, semantics, nrows, reserved;
    const int32_t *vars;                     /* the first-slice variable slots of the coordinates (slice specs) */
    const uint64_t *sizes;                   /* values per coordinate */
    const tr_bigq *const *values;            /* concatenated, coordinate 0 first */
    const tr_bigq *const *x0;                /* ndims */
    const tr_bigq *const *weight;            /* ndims (NULL array or entries: 1) */
    tr_polyopt_spec target;                  /* f (its kind and rhs ignored) */
    const tr_bigq *t;                        /* the target value (NULL 0) */
    const tr_polyopt_spec *cons;             /* ncons polynomial constraints (kind LE / GE / EQ, rhs) */
    const tr_bigq *const *A, *const *b;      /* nrows x ndims, nrows: A x <= b on the point */
} tr_interv_grid_problem;                    /* 144 bytes */

/* the search job as the CPU evaluator and the CUDA kernels read it */
typedef struct {
    uint64_t total;                          /* points in the order (after the enclosure pruning) */
    uint64_t nvalues;                        /* values of every coordinate */
    uint64_t size[TR_INTERV_GRID_DIMS];
    uint64_t voff[TR_INTERV_GRID_DIMS];      /* coordinate d's values: index voff[d] + i */
    int32_t ndims, nconds, nprimes, combine; /* combine: 0 the sum of the coordinate costs, 1 their maximum */
    uint32_t nterms, reserved;
    uint32_t cond_off[TR_INTERV_CONDS + 1];  /* condition c's terms: cond_off[c] .. cond_off[c + 1] - 1 */
    uint8_t allowed[TR_INTERV_CONDS];        /* condition c holds when the sign of H_c is in TR_INTERV_SIGN_* */
    const tr_gfp *fields;                    /* k */
    const uint64_t *garner, *half;           /* k x k, k: Garner's constants and the digits of (M - 1) / 2 */
    const uint8_t *exps;                     /* nterms x ndims */
    const uint64_t *cres;                    /* k x nterms: the integer coefficients mod p_j */
    const uint64_t *vres;                    /* k x nvalues: the cleared numerators mod p_j */
    const int64_t *ctab;                     /* nvalues: the integer cost of each coordinate value (x the scale D) */
    const uint64_t *order;                   /* total: the grid index of each rank (increasing cost, then index) */
} tr_interv_grid_view;                       /* 320 bytes */

typedef struct {
    uint64_t grid, total, pruned;            /* grid points; points in the order; points pruned by enclosures */
    uint64_t blocks, blocks_pruned;
    uint64_t bound_bits;                     /* |H_c| < 2^bound_bits at every point */
    int32_t ndims, nconds, nprimes, base_member;
    int32_t op, reserved;
} tr_interv_grid_info;                       /* 72 bytes */

typedef struct {
    int32_t method, status, cost, op;        /* op: the comparison solved (after a flip) */
    int32_t n, nsolutions, unique, base_member;   /* unique: 1, 0 (several optima), -1 not decided */
    int32_t claim, complete, reason, exact;  /* claim: TR_INTERV_CLAIM_ALGEBRAIC; exact: every solution is exact */
    int32_t nprimes, reserved;
    uint64_t examined, pruned, total, next;  /* grid: ranks examined / pruned points / ranks / the rank to resume */
    uint64_t iterations, bound_bits, crossed;   /* aggregate: breakpoints between 0 and the answer */
} tr_interv_info;                            /* 112 bytes */

typedef struct {
    int32_t exact, attained, side, how;      /* exact: d exact (else open isolating intervals); attained: 0 = an
                                                infimum; side: -1 / +1 the direction of a one-coordinate change;
                                                how: INV-001's TR_INV_HOW_* of a root */
    int32_t multiplicity, null;              /* a root's multiplicity; aggregate: the value at d is SQL NULL */
    uint64_t index;                          /* grid: the grid index */
    uint64_t rows, entered, left;            /* aggregate: rows in the predicate at d; rows that entered / left */
} tr_interv_solution;                        /* 56 bytes */

typedef struct {
    uint64_t rows, present, distinct;        /* selected rows, rows with x present, distinct x values */
    uint64_t vpresent, bytes;
    int32_t xscale, vscale, has_value, shifted;   /* shifted: the value column is the feature itself */
} tr_interv_agg_info;                        /* 56 bytes */

typedef struct {
    int32_t aggregate, pred_op, op, flip;    /* TR_INTERV_COUNT_STAR..; x + d pred_op c; A(d) op t */
    int32_t cost, semantics, reserved[2];
    const tr_bigq *c, *t, *weight, *dlo, *dhi;   /* value units; weight NULL 1; domain of d NULL unbounded */
} tr_interv_agg_query;                       /* 72 bytes */

typedef struct tr_interv_result_s tr_interv_result;
typedef struct tr_interv_grid_plan_s tr_interv_grid_plan;
typedef struct tr_interv_agg_s tr_interv_agg;

/* the grid evaluators (tr_interv_grid_eval on the CPU, tr_mgpu_interv_grid in the CUDA library): ranks start ..
 * start + count - 1: feasible bit i set when rank start + i satisfies every condition; *mincost = the smallest cost
 * among them (INT64_MAX: none); tie bit i set when it is feasible and its cost equals `match` (INT64_MIN: *mincost) */
typedef int (*tr_interv_grid_fn)(const tr_interv_grid_view *, uint64_t start, uint64_t count, int64_t match,
                                 uint64_t *feasible, uint64_t *ties, int64_t *mincost, const tr_math_budget *,
                                 tr_math_refusal *);

/* ================================================================ shared per-point functions (CPU and CUDA) */
/* the coordinates (value indices) of grid index idx: the first coordinate most significant */
TR_NUM_HD void tr_interv_digits(const tr_interv_grid_view *v, uint64_t idx, uint64_t *digit) {
    for (int32_t d = v->ndims; d-- > 0;) { digit[d] = idx % v->size[d]; idx /= v->size[d]; }
}
TR_NUM_HD int64_t tr_interv_point_cost(const tr_interv_grid_view *v, const uint64_t *digit) {
    int64_t c = 0;
    for (int32_t d = 0; d < v->ndims; d++) {
        int64_t t = v->ctab[v->voff[d] + digit[d]];
        if (v->combine) { if (t > c) c = t; } else c += t;     /* the host proved the sum fits int64 */
    }
    return c;
}
/* the sign (-1, 0, 1) of condition c at the point: residues of H_c, Garner digits compared with those of (M-1)/2 */
TR_NUM_HD int tr_interv_cond_sign(const tr_interv_grid_view *v, int32_t c, const uint64_t *digit) {
    uint64_t dig[TR_INV_PRIMES_MAX];
    const uint32_t k = (uint32_t)v->nprimes;
    int zero = 1;
    for (uint32_t j = 0; j < k; j++) {
        const tr_gfp *F = &v->fields[j];
        const uint64_t *vr = v->vres + (uint64_t)j * v->nvalues, *cr = v->cres + (uint64_t)j * v->nterms;
        uint64_t x[TR_INTERV_GRID_DIMS], h = 0;
        for (int32_t d = 0; d < v->ndims; d++) x[d] = vr[v->voff[d] + digit[d]];
        for (uint32_t t = v->cond_off[c]; t < v->cond_off[c + 1]; t++) {
            uint64_t m = cr[t];
            const uint8_t *e = v->exps + (uint64_t)t * (uint64_t)v->ndims;
            for (int32_t d = 0; d < v->ndims && m; d++)
                for (uint32_t q = 0; q < e[d]; q++) m = tr_gfp_mul_hd(F, m, x[d]);
            h = tr_gfp_add_hd(F, h, m);
        }
        if (h) zero = 0;
        const uint64_t *g = v->garner + (uint64_t)j * k;
        uint64_t y = h;
        for (uint32_t t = 0; t < j; t++) {
            uint64_t dt = dig[t] >= F->p ? dig[t] - F->p : dig[t];   /* p_t < 2 p_j: one subtraction reduces */
            y = tr_gfp_mul_hd(F, tr_gfp_sub_hd(F, y, dt), g[t]);
        }
        dig[j] = y;
    }
    if (zero) return 0;
    for (uint32_t j = k; j-- > 0;)
        if (dig[j] != v->half[j]) return dig[j] < v->half[j] ? 1 : -1;
    return 1;
}
/* 1 when grid index idx satisfies every condition; *cost its integer cost */
TR_NUM_HD int tr_interv_grid_point(const tr_interv_grid_view *v, uint64_t idx, int64_t *cost) {
    uint64_t digit[TR_INTERV_GRID_DIMS];
    tr_interv_digits(v, idx, digit);
    *cost = tr_interv_point_cost(v, digit);
    for (int32_t c = 0; c < v->nconds; c++) {
        int s = tr_interv_cond_sign(v, c, digit);
        int bit = s < 0 ? TR_INTERV_SIGN_NEG : s == 0 ? TR_INTERV_SIGN_ZERO : TR_INTERV_SIGN_POS;
        if (!(v->allowed[c] & bit)) return 0;
    }
    return 1;
}
/* well-formed views (host code of both libraries) */
static inline int tr_interv_grid_view_valid(const tr_interv_grid_view *v) {
    if (!v || v->ndims < 1 || v->ndims > TR_INTERV_GRID_DIMS || v->nconds < 0 || v->nconds > TR_INTERV_CONDS ||
        v->nprimes < 1 || v->nprimes > TR_INV_PRIMES_MAX || (v->combine != 0 && v->combine != 1) || !v->fields ||
        !v->garner || !v->half || !v->vres || !v->ctab || (v->total && !v->order) || (v->nterms && (!v->exps || !v->cres)))
        return 0;
    uint64_t total = 1, off = 0;
    for (int32_t d = 0; d < v->ndims; d++) {
        if (v->size[d] < 1 || v->voff[d] != off) return 0;
        total *= v->size[d]; off += v->size[d];
    }
    if (off != v->nvalues || v->cond_off[0] != 0 || v->cond_off[v->nconds] != v->nterms || total > TR_INTERV_GRID_MAX)
        return 0;
    for (int32_t c = 0; c < v->nconds; c++) if (v->cond_off[c + 1] < v->cond_off[c]) return 0;
    for (int32_t j = 0; j < v->nprimes; j++) if (!tr_gfp_consistent(&v->fields[j])) return 0;
    for (uint64_t r = 0; r < v->total; r++) if (v->order[r] >= total) return 0;
    return 1;
}
/* bytes the CUDA evaluation of count ranks copies to and allocates on the device */
static inline uint64_t tr_interv_grid_device_bytes(const tr_interv_grid_view *v, uint64_t count) {
    uint64_t k = (uint64_t)v->nprimes, b = tr_budget_mul(k, sizeof(tr_gfp));
    b = tr_budget_add(b, tr_budget_mul(tr_budget_add(tr_budget_mul(k, k), k), 8));
    b = tr_budget_add(b, tr_budget_mul((uint64_t)v->nterms, (uint64_t)v->ndims));
    b = tr_budget_add(b, tr_budget_mul(tr_budget_mul(k, (uint64_t)v->nterms), 8));
    b = tr_budget_add(b, tr_budget_mul(tr_budget_mul(k, v->nvalues), 8));
    b = tr_budget_add(b, tr_budget_mul(v->nvalues, 8));
    b = tr_budget_add(b, tr_budget_mul(count, 8));                       /* the order slice */
    b = tr_budget_add(b, tr_budget_mul(tr_budget_bitmap_bytes(count), 2));
    return tr_budget_add(b, 8);
}

/* ================================================================ libalg */
/* 1. linear targets */
tr_interv_result *tr_interv_linear(const tr_interv_linear_problem *, const tr_math_budget *, tr_math_refusal *,
                                   int32_t *error);
/* the same with f a first-slice polynomial of degree <= 1 in the variable slots vars[0..n) (a term in another
 * variable: TR_MATH_DOMAIN_ERROR, degree > 1: TR_MATH_UNSUPPORTED); pr->a and pr->a0 are ignored */
tr_interv_result *tr_interv_linear_poly(const tr_interv_linear_problem *, const tr_math_poly *f, const int32_t *vars,
                                        const tr_math_budget *, tr_math_refusal *, int32_t *error);
/* 2. univariate polynomial targets */
tr_interv_result *tr_interv_univariate(const tr_interv_univariate_problem *, const tr_math_budget *,
                                       tr_math_refusal *, int32_t *error);
/* 3. finite grids */
tr_interv_grid_plan *tr_interv_grid_prepare(const tr_interv_grid_problem *, const tr_math_budget *, tr_math_refusal *,
                                            int32_t *error);
int32_t tr_interv_grid_get_view(const tr_interv_grid_plan *, tr_interv_grid_view *);
int32_t tr_interv_grid_get_info(const tr_interv_grid_plan *, tr_interv_grid_info *);
/* the CPU evaluator (the kernels' code; budget: output = count) */
int tr_interv_grid_eval(const tr_interv_grid_view *, uint64_t start, uint64_t count, int64_t match, uint64_t *feasible,
                        uint64_t *ties, int64_t *mincost, const tr_math_budget *, tr_math_refusal *);
/* the search from rank start (fn NULL: tr_interv_grid_eval); iterations = ranks examined, output = ties listed */
tr_interv_result *tr_interv_grid_search(const tr_interv_grid_plan *, uint64_t start, tr_interv_grid_fn fn,
                                        const tr_math_budget *, tr_math_refusal *, int32_t *error);
/* grid index idx: the value index of each coordinate (coords[ndims]) */
int32_t tr_interv_grid_coords(const tr_interv_grid_plan *, uint64_t idx, uint64_t *coords);
void tr_interv_grid_plan_free(tr_interv_grid_plan *);
uint64_t tr_interv_grid_device_need(const tr_interv_grid_view *, uint64_t count);
/* 4. aggregates over a scope: feature x and value v (scope columns; value -1: COUNT(*) only; value == feature: the
 * shifted feature) */
tr_interv_agg *tr_interv_agg_build(const tr_scope *, int32_t feature, int32_t value, const tr_math_budget *,
                                   tr_math_refusal *, int32_t *error);
int32_t tr_interv_agg_get_info(const tr_interv_agg *, tr_interv_agg_info *);
tr_interv_result *tr_interv_agg_solve(const tr_interv_agg *, const tr_interv_agg_query *, const tr_math_budget *,
                                      tr_math_refusal *, int32_t *error);
/* A(d) exactly: *value (owned; NULL with *null = 1 for SQL NULL) and the rows in the predicate */
int32_t tr_interv_agg_value(const tr_interv_agg *, int32_t aggregate, int32_t pred_op, const tr_bigq *c,
                            const tr_bigq *d, tr_bigq **value, int32_t *null, uint64_t *rows,
                            const tr_math_budget *, tr_math_refusal *);
void tr_interv_agg_free(tr_interv_agg *);
/* results */
int32_t tr_interv_result_get_info(const tr_interv_result *, tr_interv_info *);
/* solution i: its description, the change d (lo[n], hi[n]: equal when exact), the point x = x0 + d (xlo, xhi; NULL
 * arrays allowed), its cost (clo, chi) and the value of the function at it (vlo, vhi; NULL: SQL NULL), borrowed */
int32_t tr_interv_result_solution(const tr_interv_result *, int32_t i, tr_interv_solution *, const tr_bigq **dlo,
                                  const tr_bigq **dhi, const tr_bigq **xlo, const tr_bigq **xhi, const tr_bigq **clo,
                                  const tr_bigq **chi, const tr_bigq **vlo, const tr_bigq **vhi);
/* L0: the support of solution i (bit k: coordinate k changes) */
uint64_t tr_interv_result_support(const tr_interv_result *, int32_t i);
const tr_bigq *const *tr_interv_result_part(const tr_interv_result *, int32_t part, uint64_t *count);
void tr_interv_result_free(tr_interv_result *);

/* ------------------------------------------------------------------ the CUDA library (math_interv_gpu.cu)
 * the evaluator of tr_interv_grid_fn: one work item per rank (two launches: feasibility with the minimal cost by
 * atomic minimum and the feasibility bitmap by atomic OR, then the ties of the minimal cost), equal to
 * tr_interv_grid_eval bit for bit; the device copies are budgeted (also output = count). */
int tr_mgpu_interv_grid(const tr_interv_grid_view *view, uint64_t start, uint64_t count, int64_t match,
                        uint64_t *feasible, uint64_t *ties, int64_t *mincost, const tr_math_budget *budget,
                        tr_math_refusal *refusal);
#ifdef __cplusplus
}
#endif
#endif
