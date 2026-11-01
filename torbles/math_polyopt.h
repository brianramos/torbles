/* OPT-002: bounded convex and polynomial optimization with feasibility and optimality evidence.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Two methods, both exact in what they claim (FND-009 big rationals, outward-rounded intervals):
 *
 * 1. Convex structure: minimize q(x) = x^T H x / 2 + g^T x + c0 (H symmetric positive semidefinite, H = 0 for a
 *    linear program) subject to A x <= b (m rows), E x = e (p rows) and per-variable bounds lo_i <= x_i, x_i <= hi_i
 *    (either side optional), n <= TR_POLYOPT_MAX_VARS (the declared scope is n <= 8; larger requests are bounded by
 *    the budgets). Exact rational simplex (tableau over u - v = x and slacks,
 *    phase 1 with artificials, Bland's rule: no cycling) and, for H != 0, an exact primal active-set method from the
 *    phase-1 vertex (equality-constrained steps solved exactly by LIN-002, zero-curvature descent directions from its
 *    inconsistency certificate, ratio tests, multiplier drops by the smallest index; an iteration budget bounds it).
 *    Every answer is verified exactly before it is returned:
 *      OPTIMAL     x*, multipliers lambda >= 0 (inequalities and bounds) and mu (equalities) with stationarity
 *                  H x* + g + A^T lambda + E^T mu = 0, primal feasibility and complementary slackness
 *                  lambda_i (b_i - A_i x*) = 0 (KKT); uniqueness decided: the optimal set is {x feasible : H x = H x*,
 *                  g^T x = g^T x*} and it is a point iff max / min x_k over it equals x*_k for every k (2n exact LPs,
 *                  skipped when H is positive definite or every nonbasic reduced cost of the LP is positive); a
 *                  nonunique answer carries a second optimal point (FACE)
 *      INFEASIBLE  a Farkas certificate y >= 0 (inequalities and bounds), z: A^T y + E^T z = 0, b^T y + e^T z < 0
 *      UNBOUNDED   a feasible anchor x0 and a ray d: A d <= 0, E d = 0 and either d^T H d < 0 (CURVATURE) or
 *                  d^T H d = 0 and g(x0)^T d < 0 (RATE), so q(x0 + s d) -> -inf (for H PSD: d^T H d = 0 iff H d = 0)
 *    An indefinite H is not convex. The equalities are eliminated exactly (x = x_p + N y, R = N^T H N). When no
 *    inequality varies on the affine set {E x = e} (no rows and no bounds, or rows constant on it) the problem is
 *    decided exactly by the inertia of R (LIN-002): a negative direction of R is a CURVATURE ray; otherwise q is
 *    convex on the feasible set and R y = -N^T g(x_p) gives the optimum (KKT-verified, unique iff R is definite) or a
 *    RATE ray. Otherwise a negative direction of R that is a recession direction of the polyhedron, or a ray of the
 *    LPs bounding it, certifies unboundedness when one is found; else the problem is handed to method 2 when the
 *    feasible polyhedron is bounded (its box found by 2 dim exact LPs) and dim <= TR_POLYOPT_BB_VARS; otherwise
 *    TR_MATH_UNSUPPORTED (an unbounded nonconvex problem without such a direction is not decided). Never a local
 *    optimum presented as a global one.
 *
 * 2. Small polynomial problems: minimize f(x) over a box (every bound finite, rational) subject to g_k(x) <= 0 and
 *    h_l(x) = 0 (n <= TR_POLYOPT_BB_VARS, per-polynomial dense size <= TR_POLYOPT_DENSE_MAX), by certified best-first
 *    branch and bound. A box's enclosures (the natural interval extension intersected with the centered form: the
 *    Taylor expansion at a double point of the box, both in FND-009 outward-rounded intervals) prune it soundly:
 *    INFEASIBLE when some g_k > 0 or 0 not in h_l on the whole box, BOUND when its lower bound exceeds RU(U) for the
 *    incumbent value U. Incumbents are exact: f at a box midpoint whose constraints hold exactly (bigq evaluation).
 *    The answer encloses the global minimum f* in [lower, upper]: upper = f(witness) exactly at a feasible witness,
 *    lower = the smallest lower bound of any box not pruned (an exact dyadic rational); COMPLETE only when
 *    upper - lower <= the declared tolerance, else INCOMPLETE with the reason (the node budget, a time limit, the
 *    depth limit), never a false global optimum. Feasibility is FEASIBLE (witness), NOT_FEASIBLE (a certificate: the
 *    leaf boxes partitioning the box, each with a constraint its enclosure proves violated) or UNKNOWN. Univariate
 *    problems without constraints are solved by INV-001's roots of f' (exact rational critical points, isolating
 *    intervals refined until the gap is within the tolerance). The enclosure of a batch of boxes is the data-parallel
 *    part: tr_polyopt_boxes on the CPU and tr_mgpu_polyopt_boxes on CUDA run the same inline functions below (one
 *    output slot per box, FND-009's contraction-safe directed operations only), so they agree bit for bit.
 *
 * Budgets (math_budgets.h, checked before allocating; TR_MATH_BUDGET with the refusal):
 *   QP / LP     variables = n, tensor = the tableau (rows x columns), iterations = pivots + active-set steps (also
 *               the LPs of the uniqueness test and of the hand-off), bytes per big number (math_carriers.h)
 *   polynomial  variables = n, degree, support = terms, iterations = boxes evaluated (TRUNCATES: INCOMPLETE),
 *               output = leaves of an infeasibility certificate, bytes = open boxes and leaves
 *   boxes       output = boxes, bytes = the job (CUDA: tr_polyopt_device_bytes)
 * Errors: TR_MATH_INVALID, _MEMORY, _OVERFLOW, _BUDGET, _UNSUPPORTED (sizes past the limits, an unbounded box, an
 * indefinite QP that cannot be handed off), _NOT_CONVERGED (the active-set iteration budget). Objects are immutable
 * and individually owned; constructors return NULL and set *error. */
#ifndef TORBLES_MATH_POLYOPT_H
#define TORBLES_MATH_POLYOPT_H
#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include "math_num.h"
#include "math_budgets.h"
#include "math_carriers.h"
#include "math_opt.h"
#include "mathematics.h"
#ifdef __cplusplus
extern "C" {
#endif

#define TR_POLYOPT_ABI 1
#define TR_POLYOPT_MAX_VARS 64           /* QP / LP variables (representation limit; the declared scope is n <= 8) */
#define TR_POLYOPT_MAX_ROWS 256          /* QP / LP rows: inequalities, bounds and equalities */
#define TR_POLYOPT_BB_VARS 4             /* polynomial branch and bound */
#define TR_POLYOPT_BB_CONS 64            /* constraints of a polynomial problem */
#define TR_POLYOPT_BB_POLYS (1 + TR_POLYOPT_BB_CONS)
#define TR_POLYOPT_DENSE_MAX 256         /* dense coefficients of one polynomial: prod (deg_v + 1) */
#define TR_POLYOPT_MAX_DEGREE 8          /* per variable */
#define TR_POLYOPT_MAX_TERMS 4096
#define TR_POLYOPT_DEPTH 60              /* bisections of one variable */

enum { TR_POLYOPT_OPTIMAL = 1, TR_POLYOPT_INFEASIBLE = 2, TR_POLYOPT_UNBOUNDED = 3, TR_POLYOPT_COMPLETE = 4,
       TR_POLYOPT_INCOMPLETE = 5 };                                                      /* statuses */
enum { TR_POLYOPT_SIMPLEX = 1, TR_POLYOPT_ACTIVE_SET = 2, TR_POLYOPT_BRANCH_BOUND = 3, TR_POLYOPT_ROOTS = 4 };
enum { TR_POLYOPT_FEASIBLE = 1, TR_POLYOPT_NOT_FEASIBLE = 2, TR_POLYOPT_UNKNOWN = 3 };   /* feasibility */
enum { TR_POLYOPT_REASON_NONE = 0, TR_POLYOPT_REASON_NODES = 1, TR_POLYOPT_REASON_TIME = 2,
       TR_POLYOPT_REASON_DEPTH = 3 };
enum { TR_POLYOPT_LE = 0, TR_POLYOPT_GE = 1, TR_POLYOPT_EQ = 2 };                        /* constraint kinds */
enum { TR_POLYOPT_BOX_KEEP = 0, TR_POLYOPT_BOX_INFEASIBLE = 1, TR_POLYOPT_BOX_BOUND = 2 }; /* box statuses */
enum { TR_POLYOPT_NO_UNIQUENESS = 1 };                                                   /* solve flags */
enum { TR_POLYOPT_PART_POINT = 0,        /* the optimum / the witness (n) */
       TR_POLYOPT_PART_VALUE = 1,        /* its exact objective value (1) */
       TR_POLYOPT_PART_LOWER = 2,        /* certified lower bound of the minimum (1) */
       TR_POLYOPT_PART_UPPER = 3,        /* upper bound: the value at the witness (1) */
       TR_POLYOPT_PART_LAMBDA = 4,       /* multipliers of A x <= b (m) */
       TR_POLYOPT_PART_MU = 5,           /* multipliers of E x = e (p) */
       TR_POLYOPT_PART_LAMBDA_LO = 6,    /* multipliers of the lower bounds (n; 0 without a bound) */
       TR_POLYOPT_PART_LAMBDA_HI = 7,    /* multipliers of the upper bounds (n) */
       TR_POLYOPT_PART_FARKAS_Y = 8,     /* infeasible: y >= 0 for A x <= b (m) */
       TR_POLYOPT_PART_FARKAS_Z = 9,     /* infeasible: z for E x = e (p) */
       TR_POLYOPT_PART_FARKAS_LO = 10,   /* infeasible: weights of the lower bounds (n) */
       TR_POLYOPT_PART_FARKAS_HI = 11,   /* infeasible: weights of the upper bounds (n) */
       TR_POLYOPT_PART_ANCHOR = 12,      /* unbounded: feasible x0 (n) */
       TR_POLYOPT_PART_DIRECTION = 13,   /* unbounded: the ray d (n) */
       TR_POLYOPT_PART_RATE = 14,        /* unbounded: g(x0)^T d (< 0 when the curvature is 0) (1) */
       TR_POLYOPT_PART_FACE = 15,        /* nonunique: a second optimal point (n) */
       TR_POLYOPT_PART_GRADIENT = 16,    /* H x* + g (n) */
       TR_POLYOPT_PART_SLACK = 17,       /* b - A x* (m) */
       TR_POLYOPT_PART_TOLERANCE = 18,   /* the declared tolerance (1) */
       TR_POLYOPT_PART_CURVATURE = 19,   /* unbounded: d^T H d <= 0 (1) */
       TR_POLYOPT_PARTS = 20 };

/* ------------------------------------------------------------------ structs crossing the ctypes boundary */
typedef struct {
    int32_t status, method, n, m;            /* TR_POLYOPT_*; variables; inequality rows A x <= b */
    int32_t p, unique, convex, kkt;          /* equality rows; 1 unique optimum; H PSD; KKT verified exactly */
    int32_t feasibility, reason, certificate, handed;   /* certificate: Farkas / ray / leaves verified; handed: an
                                                           indefinite QP solved by branch and bound */
    int32_t active, reserved;                /* active inequality rows (bounds included) at the optimum */
    uint64_t iterations;                     /* pivots and active-set steps; branch and bound: boxes evaluated */
    uint64_t lps;                            /* exact LPs solved (phases, uniqueness, hand-off box) */
    uint64_t pruned_infeasible, pruned_bound, pruned_tolerance;
    uint64_t open, leaves, incumbents, batches;
} tr_polyopt_info;                           /* 128 bytes */

/* a polynomial of a problem: a first-slice polynomial (with the variable slots of the call) or explicit terms */
typedef struct {
    const tr_math_poly *slice;               /* or NULL: the explicit terms */
    int32_t nterms, kind;                    /* explicit terms; constraints: TR_POLYOPT_LE / GE / EQ */
    const uint8_t *exps;                     /* nterms x n exponents */
    const tr_bigq *const *coef;              /* nterms coefficients */
    const tr_bigq *rhs;                      /* constraints: poly kind rhs (NULL = 0); objective: ignored */
} tr_polyopt_spec;                           /* 40 bytes */

/* the enclosure job: dense coefficient enclosures of every polynomial (0: objective, 1 .. ncons: g <= 0 rows,
 * then neq equality rows), by exponent index sum_v e_v stride_v (stride_0 = 1, stride_{v+1} = stride_v (deg_v + 1)) */
typedef struct {
    int32_t n, npolys, ncons, neq;
    uint32_t offset[TR_POLYOPT_BB_POLYS + 1];
    uint8_t deg[TR_POLYOPT_BB_POLYS][TR_POLYOPT_BB_VARS];
    const double *clo, *chi;                 /* offset[npolys] values each */
} tr_polyopt_encl_view;                      /* 560 bytes */

typedef struct {
    double flo, fhi;                         /* enclosure of the objective over the box */
    int32_t status, violated;                /* TR_POLYOPT_BOX_*; the first constraint proven violated (-1) */
    int32_t inner, reserved;                 /* 1: every inequality proven to hold on the whole box, no equality */
} tr_polyopt_box_out;                        /* 32 bytes */

typedef struct tr_polyopt_qp_s tr_polyopt_qp;
typedef struct tr_polyopt_poly_s tr_polyopt_poly;
typedef struct tr_polyopt_result_s tr_polyopt_result;

/* the signature of the batched enclosure (tr_polyopt_boxes on the CPU, tr_mgpu_polyopt_boxes in the CUDA library):
 * boxes i < nboxes are [xlo[i n + v], xhi[i n + v]]; ubound: RU of the incumbent (INFINITY: none) */
typedef int (*tr_polyopt_boxes_fn)(const tr_polyopt_encl_view *, uint64_t nboxes, const double *xlo,
                                   const double *xhi, double ubound, tr_polyopt_box_out *out,
                                   const tr_math_budget *, tr_math_refusal *);

/* ================================================================ inline enclosures (host and CUDA device code) */
TR_NUM_HD int tr_polyopt_finite(double x) { return x == x && x != INFINITY && x != -INFINITY; }
TR_NUM_HD tr_math_interval tr_polyopt_iv(double lo, double hi) { tr_math_interval r; r.lo = lo; r.hi = hi; return r; }
TR_NUM_HD int tr_polyopt_iv_zero(tr_math_interval a) { return a.lo == 0 && a.hi == 0; }
/* a sound result: a NaN endpoint becomes unbounded */
TR_NUM_HD tr_math_interval tr_polyopt_clean(tr_math_interval a) {
    if (a.lo != a.lo) a.lo = -INFINITY;
    if (a.hi != a.hi) a.hi = INFINITY;
    return a;
}
/* the strides of polynomial q (returns its dense size) */
TR_NUM_HD uint32_t tr_polyopt_strides(const tr_polyopt_encl_view *v, int32_t q, uint32_t *stride) {
    uint32_t s = 1;
    for (int32_t i = 0; i < v->n; i++) { stride[i] = s; s *= (uint32_t)v->deg[q][i] + 1; }
    return s;
}
/* powers P[i][e] = X_i^e (outward), e <= deg */
TR_NUM_HD void tr_polyopt_powers(const tr_math_interval *X, int32_t n, const uint8_t *deg,
                                 tr_math_interval P[TR_POLYOPT_BB_VARS][TR_POLYOPT_MAX_DEGREE + 1]) {
    for (int32_t i = 0; i < n; i++)
        for (int32_t e = 0; e <= (int32_t)deg[i]; e++) P[i][e] = tr_interval_pown(X[i], (uint64_t)e);
}
/* sum_idx C[idx] prod_i P[i][e_i(idx)] over the nonzero coefficients */
TR_NUM_HD tr_math_interval tr_polyopt_sum(const tr_math_interval *C, uint32_t size, int32_t n, const uint32_t *stride,
                                          const uint8_t *deg,
                                          tr_math_interval P[TR_POLYOPT_BB_VARS][TR_POLYOPT_MAX_DEGREE + 1]) {
    tr_math_interval acc = tr_polyopt_iv(0.0, 0.0);
    for (uint32_t idx = 0; idx < size; idx++) {
        if (tr_polyopt_iv_zero(C[idx])) continue;
        tr_math_interval t = C[idx];
        for (int32_t i = 0; i < n; i++) {
            uint32_t e = (idx / stride[i]) % ((uint32_t)deg[i] + 1);
            if (e) t = tr_interval_mul(t, P[i][e]);
        }
        acc = tr_interval_add(acc, t);
    }
    return acc;
}
/* The enclosure of polynomial q over the box X (outward-rounded, sound for every point of X): the natural
 * extension intersected with the centered form at m (m_i: a double of X_i chosen by directed operations; the shift
 * x = y + m is computed in interval arithmetic, Y = X - m). S: scratch of TR_POLYOPT_DENSE_MAX intervals. */
TR_NUM_HD tr_math_interval tr_polyopt_enclose(const tr_polyopt_encl_view *v, int32_t q, const tr_math_interval *X,
                                              tr_math_interval *S) {
    const int32_t n = v->n;
    uint32_t stride[TR_POLYOPT_BB_VARS];
    uint32_t size = tr_polyopt_strides(v, q, stride);
    const double *clo = v->clo + v->offset[q], *chi = v->chi + v->offset[q];
    tr_math_interval P[TR_POLYOPT_BB_VARS][TR_POLYOPT_MAX_DEGREE + 1], Y[TR_POLYOPT_BB_VARS];
    int finite = 1;
    for (int32_t i = 0; i < n; i++) if (!tr_polyopt_finite(X[i].lo) || !tr_polyopt_finite(X[i].hi)) finite = 0;
    for (uint32_t idx = 0; idx < size; idx++) {
        S[idx] = tr_polyopt_iv(clo[idx], chi[idx]);
        if (!tr_polyopt_finite(clo[idx]) || !tr_polyopt_finite(chi[idx])) finite = 0;
    }
    if (!finite) return tr_polyopt_iv(-INFINITY, INFINITY);
    tr_polyopt_powers(X, n, v->deg[q], P);
    tr_math_interval nat = tr_polyopt_clean(tr_polyopt_sum(S, size, n, stride, v->deg[q], P));
    for (int32_t i = 0; i < n; i++) {                          /* the expansion point and the shifted box */
        double m = tr_mul_dir(tr_add_rd(X[i].lo, X[i].hi), 0.5, 0);
        if (!tr_polyopt_finite(m)) return nat;
        Y[i] = tr_interval_sub(X[i], tr_polyopt_iv(m, m));
        const uint32_t D = v->deg[q][i];
        if (!D) continue;
        tr_math_interval M = tr_polyopt_iv(m, m);
        for (uint32_t base = 0; base < size; base++) {         /* Taylor shift of every fiber along variable i */
            if ((base / stride[i]) % (D + 1)) continue;
            for (uint32_t r = 0; r < D; r++)
                for (uint32_t k = D; k-- > r;)
                    S[base + k * stride[i]] = tr_interval_add(S[base + k * stride[i]],
                                                              tr_interval_mul(M, S[base + (k + 1) * stride[i]]));
        }
    }
    tr_polyopt_powers(Y, n, v->deg[q], P);
    tr_math_interval cen = tr_polyopt_clean(tr_polyopt_sum(S, size, n, stride, v->deg[q], P));
    tr_math_interval r;
    r.lo = nat.lo > cen.lo ? nat.lo : cen.lo;
    r.hi = nat.hi < cen.hi ? nat.hi : cen.hi;
    if (r.lo > r.hi) r = nat;                                  /* unreachable for sound enclosures */
    return r;
}
/* one box: the objective's enclosure, the constraint decisions and the pruning status */
TR_NUM_HD void tr_polyopt_box(const tr_polyopt_encl_view *v, const double *xlo, const double *xhi, double ubound,
                              tr_math_interval *S, tr_polyopt_box_out *out) {
    tr_math_interval X[TR_POLYOPT_BB_VARS];
    for (int32_t i = 0; i < v->n; i++) X[i] = tr_polyopt_iv(xlo[i], xhi[i]);
    out->violated = -1; out->inner = v->neq == 0; out->reserved = 0;
    for (int32_t k = 1; k < v->npolys; k++) {
        tr_math_interval g = tr_polyopt_enclose(v, k, X, S);
        int eq = k > v->ncons;
        if (eq ? (g.lo > 0 || g.hi < 0) : g.lo > 0) { out->violated = k - 1; break; }
        if (!eq && !(g.hi <= 0)) out->inner = 0;
    }
    tr_math_interval f = tr_polyopt_enclose(v, 0, X, S);
    out->flo = f.lo; out->fhi = f.hi;
    if (out->violated >= 0) { out->status = TR_POLYOPT_BOX_INFEASIBLE; out->inner = 0; }
    else out->status = f.lo > ubound ? TR_POLYOPT_BOX_BOUND : TR_POLYOPT_BOX_KEEP;
}
/* well-formed jobs (host code of both libraries) */
static inline int tr_polyopt_view_valid(const tr_polyopt_encl_view *v) {
    if (!v || v->n < 1 || v->n > TR_POLYOPT_BB_VARS || v->npolys < 1 || v->npolys > TR_POLYOPT_BB_POLYS ||
        v->ncons < 0 || v->neq < 0 || 1 + v->ncons + v->neq != v->npolys || !v->clo || !v->chi || v->offset[0] != 0)
        return 0;
    for (int32_t q = 0; q < v->npolys; q++) {
        uint32_t s = 1;
        for (int32_t i = 0; i < v->n; i++) {
            if (v->deg[q][i] > TR_POLYOPT_MAX_DEGREE) return 0;
            s *= (uint32_t)v->deg[q][i] + 1;
        }
        if (s > TR_POLYOPT_DENSE_MAX || v->offset[q + 1] != v->offset[q] + s) return 0;
    }
    return 1;
}
/* bytes the CUDA evaluation copies to and allocates on the device */
static inline uint64_t tr_polyopt_device_bytes(const tr_polyopt_encl_view *v, uint64_t nboxes) {
    uint64_t b = tr_budget_mul((uint64_t)v->offset[v->npolys], 16);
    b = tr_budget_add(b, tr_budget_mul(tr_budget_mul(nboxes, (uint64_t)v->n), 16));
    b = tr_budget_add(b, tr_budget_mul(nboxes, sizeof(tr_polyopt_box_out)));
    return tr_budget_add(b, tr_budget_mul(nboxes, 16 * TR_POLYOPT_DENSE_MAX));   /* the per-box scratch */
}

/* ================================================================ libalg */
/* ---- convex problems ---- */
/* q(x) = x^T H x / 2 + g^T x + c0 (H: n x n row-major, symmetric; NULL = 0, NULL entries 0; g: n, NULL entries 0),
 * A (m x n) x <= b, E (p x n) x = e, bounds lo / hi (n each, NULL arrays or entries: none). n <= 64, and m, the
 * bounds and p together at most 256 rows. */
tr_polyopt_qp *tr_polyopt_qp_create(int32_t n, const tr_bigq *const *H, const tr_bigq *const *g, const tr_bigq *c0,
                                    int32_t m, const tr_bigq *const *A, const tr_bigq *const *b, int32_t p,
                                    const tr_bigq *const *E, const tr_bigq *const *e, const tr_bigq *const *lo,
                                    const tr_bigq *const *hi, const tr_math_budget *, tr_math_refusal *,
                                    int32_t *error);
/* the objective of an OPT-001 quadratic (q(t) = t^T A t / 2 - b^T t + c: H = A, g = -b) plus `linear` (n, NULL) */
tr_polyopt_qp *tr_polyopt_qp_from_quad(const tr_opt_quad *, const tr_bigq *const *linear, int32_t m,
                                       const tr_bigq *const *A, const tr_bigq *const *b, int32_t p,
                                       const tr_bigq *const *E, const tr_bigq *const *e, const tr_bigq *const *lo,
                                       const tr_bigq *const *hi, const tr_math_budget *, tr_math_refusal *,
                                       int32_t *error);
int32_t tr_polyopt_qp_dims(const tr_polyopt_qp *, int32_t *n, int32_t *m, int32_t *p);
/* The exact solution (see the header comment). flags: TR_POLYOPT_NO_UNIQUENESS skips the uniqueness test (unique is
 * then 0 unless H is positive definite). An indefinite H is handed to branch and bound with `tolerance` (NULL = 0),
 * `batch` boxes per round (0 = 32), `fn` the enclosure (NULL: tr_polyopt_boxes) and time_limit_ms (0: none). */
tr_polyopt_result *tr_polyopt_qp_solve(const tr_polyopt_qp *, int32_t flags, const tr_bigq *tolerance, uint64_t batch,
                                       tr_polyopt_boxes_fn fn, double time_limit_ms, const tr_math_budget *,
                                       tr_math_refusal *, int32_t *error);
/* The value of the objective at x (n values) and the slacks b - A x (m), exactly (owned by the caller). */
tr_bigq *tr_polyopt_qp_value(const tr_polyopt_qp *, const tr_bigq *const *x, const tr_math_budget *,
                             tr_math_refusal *, int32_t *error);
void tr_polyopt_qp_free(tr_polyopt_qp *);

/* ---- polynomial problems ---- */
/* objective and ncons constraints (spec.kind) over n variables (vars: the first-slice variable slots of specs given
 * as slices, NULL when every spec is explicit), box lo / hi (n each, every bound finite, lo <= hi) */
tr_polyopt_poly *tr_polyopt_poly_create(int32_t n, const int32_t *vars, const tr_polyopt_spec *objective,
                                        int32_t ncons, const tr_polyopt_spec *cons, const tr_bigq *const *lo,
                                        const tr_bigq *const *hi, const tr_math_budget *, tr_math_refusal *,
                                        int32_t *error);
int32_t tr_polyopt_poly_get_view(const tr_polyopt_poly *, tr_polyopt_encl_view *);
int32_t tr_polyopt_poly_dims(const tr_polyopt_poly *, int32_t *n, int32_t *ncons, int32_t *neq);
/* the exact value of the objective (q = 0) or of constraint q - 1 (its left side minus rhs) at x */
tr_bigq *tr_polyopt_poly_value(const tr_polyopt_poly *, int32_t q, const tr_bigq *const *x, const tr_math_budget *,
                               tr_math_refusal *, int32_t *error);
/* the CPU enclosure (the kernel's code); budget: output = nboxes */
int tr_polyopt_boxes(const tr_polyopt_encl_view *, uint64_t nboxes, const double *xlo, const double *xhi,
                     double ubound, tr_polyopt_box_out *out, const tr_math_budget *, tr_math_refusal *);
/* certified branch and bound (see the header comment); tolerance NULL = 0; batch 0 = 32 boxes per round */
tr_polyopt_result *tr_polyopt_minimize(const tr_polyopt_poly *, const tr_bigq *tolerance, uint64_t batch,
                                       tr_polyopt_boxes_fn fn, double time_limit_ms, const tr_math_budget *,
                                       tr_math_refusal *, int32_t *error);
void tr_polyopt_poly_free(tr_polyopt_poly *);

/* ---- results ---- */
int32_t tr_polyopt_result_get_info(const tr_polyopt_result *, tr_polyopt_info *);
/* a part (TR_POLYOPT_PART_*): count values, borrowed (NULL when absent) */
const tr_bigq *const *tr_polyopt_result_part(const tr_polyopt_result *, int32_t part, uint64_t *count);
const tr_bigq *tr_polyopt_result_value(const tr_polyopt_result *, int32_t part, uint64_t i);
/* infeasibility leaf i: its box (borrowed lo[n], hi[n] through the out arrays) and the violated constraint */
int32_t tr_polyopt_result_leaf(const tr_polyopt_result *, uint64_t i, const tr_bigq **lo, const tr_bigq **hi,
                               int32_t *constraint);
/* active inequality rows at the optimum: row indices (< m: A rows; m + i: lower bound of x_i; m + n + i: upper) */
int32_t tr_polyopt_result_active(const tr_polyopt_result *, int32_t *rows, int32_t capacity);
void tr_polyopt_result_free(tr_polyopt_result *);

/* ------------------------------------------------------------------ the CUDA library (math_polyopt_gpu.cu)
 * out[i] = tr_polyopt_box of box i (one work item per box, its own scratch slice and output slot), equal to
 * tr_polyopt_boxes bit for bit; the device copies are budgeted (budget NULL = defaults; also output = nboxes). */
int tr_mgpu_polyopt_boxes(const tr_polyopt_encl_view *view, uint64_t nboxes, const double *xlo, const double *xhi,
                          double ubound, tr_polyopt_box_out *out, const tr_math_budget *budget,
                          tr_math_refusal *refusal);
#ifdef __cplusplus
}
#endif
#endif
