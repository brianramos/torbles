/* ENV-002: bounded multivariate parameter regions. For group functions affine in a parameter vector theta of declared
 * dimension d (1..TR_REGION_MAX_DIMS), the exact regions of a declared box or polytope where each group attains the
 * minimum (or the maximum), their vertices, facets and ties, and point location of many parameter values on the CPU
 * and on a CUDA device with the same answers bit for bit.
 * SPDX-License-Identifier: Apache-2.0
 *
 * ---- Input ----------------------------------------------------------------------------------------------------------
 * Groups g < ngroups, each a function f_g(theta) = a_g0 + sum_k a_gk theta_k with exact rational coefficients (FND-009
 * tr_bigq), or none: an empty group (absent; it never attains anything). The parameter set
 *     P = {theta : lo_k <= theta_k <= hi_k (k < d), A theta <= b}
 * is a box with finite rational bounds lo_k < hi_k and at most TR_REGION_MAX_ROWS further rows; P must have a nonempty
 * interior, decided by an exact LP (TR_MATH_INFEASIBLE when P is empty, TR_MATH_UNSUPPORTED when it has an empty
 * interior). Lower regions (the minimizer) or upper regions (the maximizer: the functions are negated internally).
 * With ncoef = 1 + d + d (d + 1) / 2 a group also has the coefficients h_kl of the quadratic monomials theta_k theta_l
 * (k <= l, in lexicographic order); a nonzero one is TR_MATH_UNSUPPORTED unless the problem declares the explicit
 * PIECEWISE_LINEAR contract (below). Nothing is ever approximated without it.
 *
 * ---- Classes, regions, ties ---------------------------------------------------------------------------------------
 * Identical functions (equal canonical forms (A_0 + sum_k A_k theta_k) / D, gcd 1, D > 0) form one class; classes are
 * ordered by their smallest member group. The region of class c is the convex polytope
 *     R_c = {theta in P : f_c(theta) <= f_c'(theta) for every class c'}
 * which is empty, of full dimension d, or of lower dimension (a class that attains the minimum only on a face). Every
 * point of P lies in some region and the nonempty regions form a polyhedral complex (two regions meet in a common
 * face). Where two or more classes attain the minimum the point is a tie: the tie policy TR_REGION_TIE_SMALLEST
 * (_LARGEST) names the smallest (largest) member group of the tied classes as the winner, and every answer reports all
 * of the tied classes as well. Reported per nonempty region: its class, its dimension, its vertices (exact rationals)
 * and an irredundant description by linear inequalities with primitive integer coefficients n theta <= c (full
 * dimension: one per facet, with its sources: the faces of the box (or cell), rows of A, and the classes c' whose
 * constraint f_c <= f_c' is that hyperplane, i.e. the classes tied with c along the facet; lower dimension: a basis of
 * its implicit equalities and one inequality per relative facet), and the exact LP evidence of its nonemptiness: the
 * optimum s* of max s subject to n_j theta + s <= c_j over its defining constraints (OPT-002's exact simplex,
 * math_polyopt.h) with its optimal point; s* > 0 (an interior point), s* = 0 (nonempty without interior), s* < 0
 * (empty) must agree with the vertex enumeration (else TR_MATH_INVALID: an internal inconsistency). Globally: the
 * distinct vertices, each with the classes attaining the minimum there (the regions having it as a vertex), the tie
 * policy's winner and the walls of P through it; and the boundaries: every facet of a full-dimensional region once,
 * with its hyperplane, its vertices, the classes tied on the whole facet (those tied at all of its vertices), the winner
 * and its kind (WALL: on a face of P; CELL: on an internal face of a contract's cell; INNER: between regions).
 *
 * ---- Construction (exact, CPU) --------------------------------------------------------------------------------------
 * 1. A candidate index per cell: a kd-tree over a dyadic lattice of 2^TR_REGION_LATTICE steps per axis. A node keeps
 *    the classes c with min_B f_c <= min_c' max_B f_c' (exact corner values of affine functions over the closed box B:
 *    a superset of the classes attaining the minimum anywhere in B); a node that a row of P excludes keeps none. A node
 *    is a leaf at <= TR_REGION_LEAF candidates, at the lattice resolution, or when d + 1 consecutive splits removed
 *    no candidate.
 * 2. For each class c that is a candidate of some leaf: its neighbors N(c), the co-candidates of the leaves where c is
 *    one, and Q_c = {theta in P : f_c <= f_c' for c' in N(c)}. If R_c is nonempty then Q_c = R_c (a segment from a
 *    point of R_c to a point of Q_c outside R_c would leave R_c at a point where c ties with a class that is a
 *    co-candidate there, a contradiction), and a point of Q_c lies in R_c iff c is a candidate of a leaf containing it:
 *    so R_c is decided exactly from one point of Q_c.
 * 3. The vertices of Q_c by the double description method: P's polytope clipped exactly by each constraint in
 *    homogeneous integer coordinates (FND-009 big integers), adjacency by the combinatorial test; the exact LP above;
 *    dimension, facets and equalities from exact ranks of the vertex-constraint incidences.
 * 4. Vertices deduplicated exactly; the classes tied at a vertex are the regions having it as a vertex (the complex
 *    property); the classes tied on a facet are those tied at all of its vertices.
 *
 * ---- Point location (CPU and CUDA run the same inline functions below) --------------------------------------------
 * A lookup job holds the residues of every integer it compares modulo k primes in (2^61, 2^62) (INV-001's verified
 * table, tr_inv_prime) with Garner's mixed-radix constants. A parameter value theta = (u_1 / v_1, ..., u_d / v_d) is the
 * homogeneous integer point x_k = u_k prod_{j != k} v_j, w = prod_k v_k; every comparison is the exact sign of an
 * integer linear form at (x, w) or of S_c D_b - S_b D_c (S_c = A_0 w + sum_k A_k x_k), k chosen from a proven bound
 * |value| < 2^B with 61 k >= B + 2 (more than TR_REGION_PRIMES_MAX moduli: TR_MATH_OVERFLOW). Modes:
 *   INDEX    the value's cell, then the kd-tree descent (theta_k <= split: left) and the leaf's candidates evaluated
 *   REGIONS  every region of the cell tested against its irredundant description (the regions containing theta)
 *   ALL      every class of the cell evaluated (the verification path: repeated all-group evaluation)
 * All three give the same answer: status (OK, OUTSIDE: not in P, EMPTY: no present group), kind (INTERIOR: one class
 * attains the minimum, BOUNDARY: two or more), the cell, the tied classes (the count, and the first max_ties of them in
 * increasing order), the winner (tie policy), its class and its region. One work item per value writes its own slots,
 * so the answers do not depend on the order of the threads.
 *
 * ---- Quadratic functions under the explicit PIECEWISE_LINEAR contract (approximate, labelled) -------------------
 * With contract PIECEWISE_LINEAR, K = pieces cells per axis and an error budget E (required), the box is divided into
 * K^d equal cells; on a cell (midpoint m, half widths r) each function q (affine and quadratic coefficients) is replaced
 * by the affine L = q(m) + grad q(m) . (theta - m) + (lo + hi) / 2, where [lo, hi] encloses the quadratic remainder
 * sum_{k<=l} h_kl y_k y_l over |y_k| <= r_k (h_kk y_k^2 in [min(0, h_kk r_k^2), max(0, h_kk r_k^2)], h_kl y_k y_l in
 * [-|h_kl| r_k r_l, |h_kl| r_k r_l] for k < l): |q - L| <= eps = (hi - lo) / 2 on the cell, exactly. The largest eps must
 * not exceed E (else TR_MATH_NOT_CONVERGED: a bound that is not achieved is refused, never reported). Regions, ties and
 * lookups are then those of the enclosures, cell by cell (a value on an internal cell face belongs to the lowest cell
 * containing it), and the result is labelled APPROXIMATE as UNC-002 labels its approximations: at every theta the
 * winner's true value is within 2 eps of the true envelope (envelope_error), and tr_region_authorize refuses an exact
 * rewrite (FND-007 reason APPROXIMATE). An affine function has eps = 0.
 *
 * ---- Budgets (math_budgets.h) ------------------------------------------------------------------------------------
 * Before anything is allocated: variables = d, degree (1, or 2 under the contract), states = cells x groups, bytes =
 * the function tables. During the build each of these is counted exactly and refused at limit + 1 (the reported need
 * is limit + 1; nothing partial is returned, never a truncated set of regions): output = nonempty regions, support =
 * region-vertex incidences, circuit = index nodes, iterations = index nodes + clipping steps + LP pivots, bytes = the
 * tables as they grow (OPT-002 checks each LP's tableau and pivots itself). Lookups: output = values, bytes = the job
 * (CUDA: and its device copies, tr_region_view_device_bytes).
 *
 * Errors: TR_MATH_INVALID (malformed input, lo >= hi), _MEMORY, _OVERFLOW (more than TR_REGION_PRIMES_MAX moduli),
 * _BUDGET (with the tr_math_refusal), _UNSUPPORTED (d > 4, quadratic terms without the contract, a degree above 2,
 * more than TR_REGION_MAX_ROWS rows, P with an empty interior, a region LP above TR_REGION_LP_ROWS rows whose vertex
 * enumeration is empty), _INFEASIBLE (P empty), _NOT_CONVERGED (the contract's error above its budget), TR_MATH_DOMAIN_ERROR
 * (tr_region_authorize of an approximate set). Objects are immutable and individually owned; constructors return NULL
 * and set *error, inputs unchanged. */
#ifndef TORBLES_MATH_REGION_H
#define TORBLES_MATH_REGION_H
#include <stdint.h>
#include <stddef.h>
#include "math_num.h"
#include "math_budgets.h"
#include "math_carriers.h"
#include "math_inv.h"
#include "math_certificates.h"
#include "mathematics.h"
#ifdef __cplusplus
extern "C" {
#endif

#define TR_REGION_ABI 1
#define TR_REGION_MAX_DIMS 4
#define TR_REGION_FORM (TR_REGION_MAX_DIMS + 1)     /* coefficients of a linear form in (x_1..x_d, w) */
#define TR_REGION_MAX_COEF 15                       /* 1 + d + d (d + 1) / 2 at d = 4 */
#define TR_REGION_MAX_ROWS 64
#define TR_REGION_MAX_PIECES 16
#define TR_REGION_MAX_CELLS 4096
#define TR_REGION_PRIMES_MAX 64                     /* moduli of the lookups: values below about 2^3900 */
#define TR_REGION_LIMBS_MAX 64                      /* 64-bit limbs of a parameter value's numerator or denominator */
#define TR_REGION_LEAF 8                            /* candidates of an index leaf */
#define TR_REGION_LATTICE 20                        /* lattice bits per axis of the index */
#define TR_REGION_LP_ROWS 255                       /* rows of one certifying LP (OPT-002: 256 with the bound s <= 1) */
#define TR_REGION_NONE UINT64_MAX

enum { TR_REGION_LOWER = 0, TR_REGION_UPPER = 1 };
enum { TR_REGION_TIE_SMALLEST = 0, TR_REGION_TIE_LARGEST = 1 };
enum { TR_REGION_EXACT = 0, TR_REGION_PIECEWISE_LINEAR = 1 };               /* contracts */
enum { TR_REGION_CLAIM_EXACT = 1, TR_REGION_CLAIM_APPROXIMATE = 2 };
enum { TR_REGION_EMPTY = 0, TR_REGION_FINITE = 1 };                        /* set kinds */
enum { TR_REGION_INDEX = 1, TR_REGION_REGIONS = 2, TR_REGION_ALL = 3 };    /* lookup modes */
enum { TR_REGION_Q_OK = 0, TR_REGION_Q_OUTSIDE = 1, TR_REGION_Q_EMPTY = 2 };   /* answer status (or < 0: an error) */
enum { TR_REGION_INTERIOR = 1, TR_REGION_BOUNDARY = 2 };                  /* answer kinds */
/* constraint sources: theta_k >= the cell's lower bound (index k), <= its upper bound (k), row i of A, class c' */
enum { TR_REGION_SRC_LOWER = 1, TR_REGION_SRC_UPPER = 2, TR_REGION_SRC_ROW = 3, TR_REGION_SRC_CLASS = 4 };
enum { TR_REGION_WALL = 1, TR_REGION_CELL = 2, TR_REGION_INNER = 3 };     /* boundary kinds */
enum { TR_REGION_NEED_CPU = 0, TR_REGION_NEED_DEVICE = 1 };

/* ------------------------------------------------------------------ structs crossing the ctypes boundary */
typedef struct {
    int32_t d, ncoef;                /* dimension; coefficients per group: 1 + d, or 1 + d + d (d + 1) / 2 */
    int32_t nrows, upper;            /* rows of A theta <= b; 1: the maximizer's regions */
    int32_t tie, contract;           /* TR_REGION_TIE_*; TR_REGION_EXACT / _PIECEWISE_LINEAR */
    int32_t pieces, reserved;        /* cells per axis under the contract (1..TR_REGION_MAX_PIECES); 0 */
    uint64_t ngroups;
    const tr_bigq *const *coef;      /* ngroups x ncoef: a_0, a_1..a_d, then h_kl (k <= l); NULL entries 0 */
    const uint8_t *present;          /* ngroups flags (0: an empty group), or NULL: every group */
    const tr_bigq *const *lo;        /* d finite lower bounds */
    const tr_bigq *const *hi;        /* d finite upper bounds (lo_k < hi_k) */
    const tr_bigq *const *A;         /* nrows x d (NULL entries 0) */
    const tr_bigq *const *b;         /* nrows */
    const tr_bigq *error_budget;     /* PIECEWISE_LINEAR: the largest enclosure error permitted (required) */
} tr_region_problem;                 /* 96 bytes */

typedef struct {
    uint64_t groups, present, classes, cells;
    uint64_t regions, full, lower, vertices;     /* nonempty regions: of dimension d / below; distinct vertices */
    uint64_t incidences, boundaries, nodes, leaves;   /* region-vertex records; facets once; index nodes, leaves */
    uint64_t candidates, lps, pivots, clips;     /* leaf candidate entries; certifying LPs and their pivots; clips */
    uint64_t iterations, bytes;                  /* the work counted against the budget; the largest bytes checked */
    int32_t d, kind, upper, tie;                 /* kind: TR_REGION_EMPTY / _FINITE */
    int32_t claim, contract, pieces, degree;     /* TR_REGION_CLAIM_*; the largest degree given */
    int32_t nrows, unverified;                   /* rows of A; regions decided by vertex enumeration alone */
} tr_region_info;                                /* 184 bytes */

typedef struct {
    uint64_t cell, members, winner, region;      /* region: TR_REGION_NONE when R_c is empty */
    int32_t degree, exact, reserved[2];          /* the group functions' degree; exact: eps = 0 */
} tr_region_class_info;                          /* 48 bytes */

typedef struct {
    uint64_t cls, cell, winner, members;         /* class; cell; the tie policy's member; member groups */
    uint64_t vertices, inequalities, equalities, lp_rows;   /* lp_rows 0: decided by vertex enumeration alone */
    int32_t dim, full, lp_sign, reserved;        /* affine dimension; dim == d; the sign of the LP optimum s* */
} tr_region_region_info;                         /* 80 bytes */

typedef struct {
    uint64_t cell, classes, winner, walls;       /* tied classes; the tie policy's group; walls of P through it */
} tr_region_vertex_info;                         /* 32 bytes */

typedef struct {
    uint64_t cell, classes, winner, vertices;    /* tied classes on the whole facet; the winner; its vertices */
    uint64_t region;                             /* the region whose halfspace the stored hyperplane is */
    int32_t kind, reserved;                      /* TR_REGION_WALL / _CELL / _INNER */
} tr_region_boundary_info;                       /* 48 bytes */

typedef struct {
    int32_t status, kind;                        /* TR_REGION_Q_*; TR_REGION_INTERIOR / _BOUNDARY */
    uint64_t cell, winner, cls, region, tied;    /* tied: classes attaining the minimum at the value */
} tr_region_answer;                              /* 48 bytes */

/* the lookup job as the CPU evaluator and the CUDA kernel read it (pointers into a tr_region_lookup). Residue tables
 * are prime-major: entry e of a table of n entries of s words has the word t for prime j at [(j * n + e) * s + t]. */
typedef struct {
    uint64_t npoints, nclasses, nregions, nnodes;
    uint64_t ncells, ncons, ncand, max_ties;     /* region constraint entries; leaf candidates; ties per value */
    int32_t d, nprimes, mode, tie;
    int32_t kind, nwalls, pieces, limbs;         /* nwalls = 2 d + rows; limbs per numerator / denominator */
    const tr_gfp *fields;                        /* k */
    const uint64_t *garner, *half;               /* k x k: p_t^-1 mod p_j (t < j); k digits of (M - 1) / 2 */
    const uint64_t *wres;                        /* walls: forms of d + 1 words (<= 0 inside P) */
    const uint64_t *gres;                        /* cell grid: d x (pieces - 1) boundaries of 2 words (q, p): q x - p w */
    const uint64_t *cell_root;                   /* ncells: the index root of each cell */
    const uint64_t *cell_class;                  /* ncells + 1: the classes of cell j are [cell_class[j], ...[j + 1]) */
    const uint64_t *cell_region;                 /* ncells + 1 */
    const int32_t *naxis;                        /* nnodes: the split axis, or -1 for a leaf */
    const uint64_t *nlink;                       /* nnodes x 2: the children, or a leaf's (candidate offset, count) */
    const uint64_t *nres;                        /* nodes: 2 words (q, p): theta_axis <= p / q goes left */
    const uint64_t *cand;                        /* ncand class ids, increasing per leaf */
    const uint64_t *cres;                        /* classes: d + 2 words: A_1..A_d, A_0 (the w coefficient), D */
    const uint64_t *cwinner, *cregion;           /* nclasses: the tie policy's member; its region (or NONE) */
    const uint64_t *rclass, *rcons;              /* nregions: the class; nregions + 1: constraint offsets */
    const int8_t *ceq;                           /* ncons: 1 an equality (= 0), 0 an inequality (<= 0) */
    const uint64_t *kres;                        /* region constraints: forms of d + 1 words */
    const int8_t *qsign;                         /* npoints x d: the sign of u */
    const uint64_t *qnum, *qden;                 /* npoints x d x limbs: |u|, v > 0 */
} tr_region_view;                                /* 272 bytes */

typedef struct tr_region_set_s tr_region_set;
typedef struct tr_region_lookup_s tr_region_lookup;

/* ------------------------------------------------------------------ shared exact signs (CPU code and kernels) */
/* the sign (-1, 0, 1) of the integer S with residues r[0..k-1] (|S| < M / 2): Garner's digits against (M - 1) / 2 */
TR_NUM_HD int tr_region_garner_sign(const tr_gfp *fields, const uint64_t *garner, const uint64_t *half, uint32_t k,
                                    const uint64_t *r) {
    uint64_t dig[TR_REGION_PRIMES_MAX];
    int zero = 1;
    for (uint32_t j = 0; j < k; j++) {
        const tr_gfp *F = &fields[j];
        const uint64_t *g = garner + (uint64_t)j * k;
        uint64_t x = r[j];
        if (x) zero = 0;
        for (uint32_t t = 0; t < j; t++) {
            uint64_t dt = dig[t] >= F->p ? dig[t] - F->p : dig[t];      /* p_t < 2 p_j: one subtraction reduces */
            x = tr_gfp_mul_hd(F, tr_gfp_sub_hd(F, x, dt), g[t]);
        }
        dig[j] = x;
    }
    if (zero) return 0;
    for (uint32_t j = k; j-- > 0;)
        if (dig[j] != half[j]) return dig[j] < half[j] ? 1 : -1;
    return 1;
}
TR_NUM_HD int tr_region_sign(const tr_region_view *v, const uint64_t *r) {
    return tr_region_garner_sign(v->fields, v->garner, v->half, (uint32_t)v->nprimes, r);
}
/* the sign of a linear form at the value: sum_k f[k] x_k + f[d] w (table base, n entries of d + 1 words, entry e);
 * xr: the value's residues, prime-major: xr[j * (d + 1) + k] (k = d: w) */
TR_NUM_HD int tr_region_form_sign(const tr_region_view *v, const uint64_t *table, uint64_t n, uint64_t e,
                                  const uint64_t *xr) {
    uint64_t r[TR_REGION_PRIMES_MAX];
    const uint32_t W = (uint32_t)v->d + 1;
    for (uint32_t j = 0; j < (uint32_t)v->nprimes; j++) {
        const tr_gfp *F = &v->fields[j];
        const uint64_t *f = table + ((uint64_t)j * n + e) * W, *x = xr + (uint64_t)j * W;
        uint64_t s = 0;
        for (uint32_t t = 0; t < W; t++) s = tr_gfp_add_hd(F, s, tr_gfp_mul_hd(F, f[t], x[t]));
        r[j] = s;
    }
    return tr_region_sign(v, r);
}
/* the sign of q x_axis - p w for a pair (q, p) (table of n entries of 2 words, entry e) */
TR_NUM_HD int tr_region_split_sign(const tr_region_view *v, const uint64_t *table, uint64_t n, uint64_t e, int32_t axis,
                                   const uint64_t *xr) {
    uint64_t r[TR_REGION_PRIMES_MAX];
    const uint32_t W = (uint32_t)v->d + 1;
    for (uint32_t j = 0; j < (uint32_t)v->nprimes; j++) {
        const tr_gfp *F = &v->fields[j];
        const uint64_t *f = table + ((uint64_t)j * n + e) * 2, *x = xr + (uint64_t)j * W;
        r[j] = tr_gfp_sub_hd(F, tr_gfp_mul_hd(F, f[0], x[axis]), tr_gfp_mul_hd(F, f[1], x[W - 1]));
    }
    return tr_region_sign(v, r);
}
/* S_c = sum_k A_k x_k + A_0 w and D_c of class c, for every prime (s[j], dd[j]) */
TR_NUM_HD void tr_region_class_value(const tr_region_view *v, uint64_t c, const uint64_t *xr, uint64_t *s, uint64_t *dd) {
    const uint32_t W = (uint32_t)v->d + 1, S = W + 1;
    for (uint32_t j = 0; j < (uint32_t)v->nprimes; j++) {
        const tr_gfp *F = &v->fields[j];
        const uint64_t *f = v->cres + ((uint64_t)j * v->nclasses + c) * S, *x = xr + (uint64_t)j * W;
        uint64_t a = 0;
        for (uint32_t t = 0; t < W; t++) a = tr_gfp_add_hd(F, a, tr_gfp_mul_hd(F, f[t], x[t]));
        s[j] = a; dd[j] = f[W];
    }
}
/* the sign of S_a / D_a - S_b / D_b (D > 0): the sign of S_a D_b - S_b D_a */
TR_NUM_HD int tr_region_cmp_values(const tr_region_view *v, const uint64_t *sa, const uint64_t *da, const uint64_t *sb,
                                   const uint64_t *db) {
    uint64_t r[TR_REGION_PRIMES_MAX];
    for (uint32_t j = 0; j < (uint32_t)v->nprimes; j++) {
        const tr_gfp *F = &v->fields[j];
        r[j] = tr_gfp_sub_hd(F, tr_gfp_mul_hd(F, sa[j], db[j]), tr_gfp_mul_hd(F, sb[j], da[j]));
    }
    return tr_region_sign(v, r);
}
/* the residues of value i: x_k = u_k prod_{j != k} v_j, w = prod v_j, for every prime (xr prime-major, d + 1 words) */
TR_NUM_HD void tr_region_value_residues(const tr_region_view *v, uint64_t i, uint64_t *xr) {
    const uint32_t d = (uint32_t)v->d, W = d + 1, L = (uint32_t)v->limbs;
    for (uint32_t j = 0; j < (uint32_t)v->nprimes; j++) {
        const tr_gfp *F = &v->fields[j];
        uint64_t u[TR_REGION_MAX_DIMS], q[TR_REGION_MAX_DIMS], pre[TR_REGION_MAX_DIMS + 1], suf[TR_REGION_MAX_DIMS + 1];
        for (uint32_t k = 0; k < d; k++) {
            const uint64_t at = (i * d + k) * (uint64_t)L;
            u[k] = tr_inv_limbs_mod(F, v->qnum + at, L);
            if (v->qsign[i * d + k] < 0) u[k] = tr_gfp_neg_hd(F, u[k]);
            q[k] = tr_inv_limbs_mod(F, v->qden + at, L);
        }
        pre[0] = 1 % F->p; suf[d] = 1 % F->p;
        for (uint32_t k = 0; k < d; k++) pre[k + 1] = tr_gfp_mul_hd(F, pre[k], q[k]);
        for (uint32_t k = d; k-- > 0;) suf[k] = tr_gfp_mul_hd(F, suf[k + 1], q[k]);
        uint64_t *x = xr + (uint64_t)j * W;
        for (uint32_t k = 0; k < d; k++) x[k] = tr_gfp_mul_hd(F, u[k], tr_gfp_mul_hd(F, pre[k], suf[k + 1]));
        x[d] = pre[d];
    }
}
/* the cell of a value inside the box: per axis, the number of interior grid boundaries strictly below theta_k (a value
 * on an internal face belongs to the lower cell); the first axis is the most significant digit */
TR_NUM_HD uint64_t tr_region_value_cell(const tr_region_view *v, const uint64_t *xr) {
    if (v->pieces <= 1) return 0;
    const uint64_t K = (uint64_t)v->pieces, nb = (uint64_t)v->d * (K - 1);
    uint64_t cell = 0;
    for (int32_t a = 0; a < v->d; a++) {
        uint64_t j = 0;
        for (uint64_t b = 0; b + 1 < K; b++)
            if (tr_region_split_sign(v, v->gres, nb, (uint64_t)a * (K - 1) + b, a, xr) > 0) j = b + 1;
        cell = cell * K + j;
    }
    return cell;
}
/* one value: the answer of the job's mode, and its first max_ties tied classes in ties[0..max_ties) (NONE after them) */
TR_NUM_HD void tr_region_answer_one(const tr_region_view *v, uint64_t i, tr_region_answer *out, uint64_t *ties) {
    out->status = TR_REGION_Q_OK; out->kind = 0;
    out->cell = TR_REGION_NONE; out->winner = TR_REGION_NONE; out->cls = TR_REGION_NONE;
    out->region = TR_REGION_NONE; out->tied = 0;
    for (uint64_t t = 0; ties && t < v->max_ties; t++) ties[t] = TR_REGION_NONE;
    if (v->kind == TR_REGION_EMPTY) { out->status = TR_REGION_Q_EMPTY; return; }
    uint64_t xr[TR_REGION_PRIMES_MAX * TR_REGION_FORM];
    tr_region_value_residues(v, i, xr);
    for (int32_t c = 0; c < v->nwalls; c++)
        if (tr_region_form_sign(v, v->wres, (uint64_t)v->nwalls, (uint64_t)c, xr) > 0) {
            out->status = TR_REGION_Q_OUTSIDE; return;
        }
    const uint64_t cell = tr_region_value_cell(v, xr);
    out->cell = cell;
    uint64_t bs[TR_REGION_PRIMES_MAX], bd[TR_REGION_PRIMES_MAX], s[TR_REGION_PRIMES_MAX], dd[TR_REGION_PRIMES_MAX];
    uint64_t tied = 0, win = TR_REGION_NONE, wcls = TR_REGION_NONE;
    const int smallest = v->tie == TR_REGION_TIE_SMALLEST;
    if (v->mode == TR_REGION_REGIONS) {
        for (uint64_t r = v->cell_region[cell]; r < v->cell_region[cell + 1]; r++) {
            int inside = 1;
            for (uint64_t e = v->rcons[r]; e < v->rcons[r + 1] && inside; e++) {
                int sg = tr_region_form_sign(v, v->kres, v->ncons, e, xr);
                if (v->ceq[e] ? sg != 0 : sg > 0) inside = 0;
            }
            if (!inside) continue;
            const uint64_t c = v->rclass[r];
            if (ties && tied < v->max_ties) ties[tied] = c;
            if (!tied || (smallest ? v->cwinner[c] < win : v->cwinner[c] > win)) { win = v->cwinner[c]; wcls = c; }
            tied++;
        }
    } else {
        uint64_t lo, hi;
        const uint64_t *list = 0;
        if (v->mode == TR_REGION_INDEX) {
            uint64_t node = v->cell_root[cell];
            while (v->naxis[node] >= 0) {
                int sg = tr_region_split_sign(v, v->nres, v->nnodes, node, v->naxis[node], xr);
                node = v->nlink[2 * node + (sg <= 0 ? 0 : 1)];
            }
            lo = 0; hi = v->nlink[2 * node + 1];
            list = v->cand + v->nlink[2 * node];
        } else {
            lo = v->cell_class[cell]; hi = v->cell_class[cell + 1];
        }
        for (uint64_t t = lo; t < hi; t++) {
            const uint64_t c = list ? list[t] : t;
            tr_region_class_value(v, c, xr, s, dd);
            int sg = tied ? tr_region_cmp_values(v, s, dd, bs, bd) : -1;
            if (sg > 0) continue;
            if (sg < 0) {                                                  /* a new minimum */
                for (uint32_t j = 0; j < (uint32_t)v->nprimes; j++) { bs[j] = s[j]; bd[j] = dd[j]; }
                tied = 0;
                for (uint64_t q = 0; ties && q < v->max_ties; q++) ties[q] = TR_REGION_NONE;
            }
            if (ties && tied < v->max_ties) ties[tied] = c;
            if (!tied || (smallest ? v->cwinner[c] < win : v->cwinner[c] > win)) { win = v->cwinner[c]; wcls = c; }
            tied++;
        }
    }
    if (!tied) { out->status = TR_REGION_Q_OUTSIDE; return; }    /* unreachable for a valid job and a value in P */
    out->kind = tied >= 2 ? TR_REGION_BOUNDARY : TR_REGION_INTERIOR;
    out->tied = tied; out->winner = win; out->cls = wcls; out->region = v->cregion[wcls];
}

/* bytes the CUDA lookup copies to / allocates on the device (saturating) */
static inline uint64_t tr_region_view_device_bytes(const tr_region_view *v) {
    const uint64_t k = (uint64_t)v->nprimes, W = (uint64_t)v->d + 1;
    uint64_t b = tr_budget_mul(k, sizeof(tr_gfp));
    b = tr_budget_add(b, tr_budget_mul(tr_budget_add(tr_budget_mul(k, k), k), 8));
    b = tr_budget_add(b, tr_budget_mul(tr_budget_mul(tr_budget_mul(k, (uint64_t)v->nwalls), W), 8));
    if (v->pieces > 1)
        b = tr_budget_add(b, tr_budget_mul(tr_budget_mul(k, (uint64_t)v->d * (uint64_t)(v->pieces - 1)), 16));
    b = tr_budget_add(b, tr_budget_mul(tr_budget_add(tr_budget_mul(v->ncells, 3), 2), 8));
    b = tr_budget_add(b, tr_budget_mul(v->nnodes, 4 + 16));
    b = tr_budget_add(b, tr_budget_mul(tr_budget_mul(k, v->nnodes), 16));
    b = tr_budget_add(b, tr_budget_mul(v->ncand, 8));
    b = tr_budget_add(b, tr_budget_mul(tr_budget_mul(tr_budget_mul(k, v->nclasses), W + 1), 8));
    b = tr_budget_add(b, tr_budget_mul(v->nclasses, 16));
    b = tr_budget_add(b, tr_budget_mul(tr_budget_add(tr_budget_mul(v->nregions, 2), 1), 8));
    b = tr_budget_add(b, tr_budget_mul(v->ncons, 1));
    b = tr_budget_add(b, tr_budget_mul(tr_budget_mul(tr_budget_mul(k, v->ncons), W), 8));
    b = tr_budget_add(b, tr_budget_mul(tr_budget_mul(v->npoints, (uint64_t)v->d), 1));
    b = tr_budget_add(b, tr_budget_mul(tr_budget_mul(tr_budget_mul(v->npoints, (uint64_t)v->d), (uint64_t)v->limbs), 16));
    b = tr_budget_add(b, tr_budget_mul(v->npoints, sizeof(tr_region_answer)));
    return tr_budget_add(b, tr_budget_mul(tr_budget_mul(v->npoints, v->max_ties), 8));
}
/* a well-formed job (checked by the CPU evaluator and the CUDA entry point alike) */
static inline int tr_region_view_valid(const tr_region_view *v) {
    if (!v || v->d < 1 || v->d > TR_REGION_MAX_DIMS || v->mode < TR_REGION_INDEX || v->mode > TR_REGION_ALL ||
        (v->tie != TR_REGION_TIE_SMALLEST && v->tie != TR_REGION_TIE_LARGEST))
        return 0;
    if (v->kind == TR_REGION_EMPTY) return 1;
    if (v->kind != TR_REGION_FINITE || v->nprimes < 1 || v->nprimes > TR_REGION_PRIMES_MAX || v->limbs < 1 ||
        v->limbs > TR_REGION_LIMBS_MAX || v->pieces < 1 || v->pieces > TR_REGION_MAX_PIECES || v->nwalls < 2 * v->d ||
        v->ncells < 1 || !v->fields || !v->garner || !v->half || !v->wres || (v->pieces > 1 && !v->gres) ||
        !v->cell_root || !v->cell_class || !v->cell_region || !v->naxis || !v->nlink || !v->nres || !v->cres ||
        !v->cwinner || !v->cregion || (v->ncand && !v->cand) || (v->nregions && (!v->rclass || !v->rcons)) ||
        (v->ncons && (!v->ceq || !v->kres)) || (v->npoints && (!v->qsign || !v->qnum || !v->qden)))
        return 0;
    for (int32_t j = 0; j < v->nprimes; j++) if (!tr_gfp_consistent(&v->fields[j])) return 0;
    uint64_t cells = 1;
    for (int32_t a = 0; a < v->d; a++) cells *= (uint64_t)v->pieces;
    if (cells != v->ncells || v->cell_class[0] != 0 || v->cell_class[v->ncells] != v->nclasses ||
        v->cell_region[0] != 0 || v->cell_region[v->ncells] != v->nregions)
        return 0;
    for (uint64_t c = 0; c < v->ncells; c++)
        if (v->cell_class[c + 1] < v->cell_class[c] || v->cell_region[c + 1] < v->cell_region[c] ||
            v->cell_root[c] >= v->nnodes) return 0;
    for (uint64_t n = 0; n < v->nnodes; n++) {
        if (v->naxis[n] >= v->d) return 0;
        if (v->naxis[n] >= 0) {
            if (v->nlink[2 * n] >= v->nnodes || v->nlink[2 * n + 1] >= v->nnodes || v->nlink[2 * n] <= n ||
                v->nlink[2 * n + 1] <= n) return 0;
        } else if (v->nlink[2 * n] > v->ncand || v->nlink[2 * n + 1] > v->ncand - v->nlink[2 * n]) return 0;
    }
    for (uint64_t e = 0; e < v->ncand; e++) if (v->cand[e] >= v->nclasses) return 0;
    for (uint64_t c = 0; c < v->nclasses; c++) if (v->cregion[c] != TR_REGION_NONE && v->cregion[c] >= v->nregions) return 0;
    for (uint64_t r = 0; r < v->nregions; r++)
        if (v->rclass[r] >= v->nclasses || v->rcons[r + 1] < v->rcons[r]) return 0;
    if (v->nregions && (v->rcons[0] != 0 || v->rcons[v->nregions] != v->ncons)) return 0;
    return 1;
}

/* ------------------------------------------------------------------ region sets (libalg) */
/* the a-priori needs of a build (variables, degree, states, bytes) */
int32_t tr_region_need(const tr_region_problem *, tr_math_need *need);
tr_region_set *tr_region_build(const tr_region_problem *, const tr_math_budget *, tr_math_refusal *, int32_t *error);
/* first-slice polynomials in the variable slots vars[0..d) (NULL: an empty group; a term in another variable:
 * TR_MATH_DOMAIN_ERROR; degree 2: the contract decides; above 2: TR_MATH_UNSUPPORTED); pr->coef / ncoef are ignored */
tr_region_set *tr_region_from_polys(const tr_region_problem *, const tr_math_poly *const *polys, const int32_t *vars,
                                    const tr_math_budget *, tr_math_refusal *, int32_t *error);
int32_t tr_region_get_info(const tr_region_set *, tr_region_info *);
const tr_bigq *tr_region_bound(const tr_region_set *, int32_t hi, int32_t k);      /* the declared box (borrowed) */
const tr_bigq *tr_region_error(const tr_region_set *, int32_t which);  /* 0: largest eps, 1: 2 eps, 2: the budget */
int32_t tr_region_class_get(const tr_region_set *, uint64_t cls, tr_region_class_info *);
int32_t tr_region_class_members(const tr_region_set *, uint64_t cls, uint64_t *out, uint64_t capacity);
/* coefficient k (0: constant, 1..d: theta_k) of the class's affine function (as given, or its enclosure) and its eps */
const tr_bigq *tr_region_class_coef(const tr_region_set *, uint64_t cls, int32_t k);
const tr_bigq *tr_region_class_eps(const tr_region_set *, uint64_t cls);
int32_t tr_region_region_get(const tr_region_set *, uint64_t region, tr_region_region_info *);
int32_t tr_region_region_vertices(const tr_region_set *, uint64_t region, uint64_t *out, uint64_t capacity);
/* constraint i of the region's description (inequalities first, then equalities): n (d borrowed coefficients of
 * theta) and c of n theta <= c (or = c), and its number of sources */
int32_t tr_region_region_constraint(const tr_region_set *, uint64_t region, uint64_t i, const tr_bigint **normal,
                                    const tr_bigint **rhs, int32_t *equality, uint64_t *nsources);
int32_t tr_region_region_sources(const tr_region_set *, uint64_t region, uint64_t i, int32_t *kinds, uint64_t *index,
                                 uint64_t capacity);
/* the certifying LP's optimal point (d borrowed coordinates) and optimum s* */
int32_t tr_region_region_lp(const tr_region_set *, uint64_t region, const tr_bigq **point, const tr_bigq **margin);
int32_t tr_region_vertex_get(const tr_region_set *, uint64_t vertex, tr_region_vertex_info *);
int32_t tr_region_vertex_point(const tr_region_set *, uint64_t vertex, const tr_bigq **coords);   /* d borrowed */
int32_t tr_region_vertex_classes(const tr_region_set *, uint64_t vertex, uint64_t *out, uint64_t capacity);
/* walls: 2 k (theta_k >= lo_k), 2 k + 1 (theta_k <= hi_k), 2 d + i (row i) */
int32_t tr_region_vertex_walls(const tr_region_set *, uint64_t vertex, uint64_t *out, uint64_t capacity);
int32_t tr_region_boundary_get(const tr_region_set *, uint64_t boundary, tr_region_boundary_info *);
int32_t tr_region_boundary_plane(const tr_region_set *, uint64_t boundary, const tr_bigint **normal, const tr_bigint **rhs);
int32_t tr_region_boundary_vertices(const tr_region_set *, uint64_t boundary, uint64_t *out, uint64_t capacity);
int32_t tr_region_boundary_classes(const tr_region_set *, uint64_t boundary, uint64_t *out, uint64_t capacity);
/* the FND-007 decision on using the regions as an exact result: authorized (an exact build: TR_MATH_OK) or refused
 * with reason TR_CERT_REFUSE_APPROXIMATE (TR_MATH_DOMAIN_ERROR) */
int32_t tr_region_authorize(const tr_region_set *, tr_cert_decision *out);
void tr_region_free(tr_region_set *);

/* ------------------------------------------------------------------ lookups (libalg) */
/* values: npoints x d exact rationals (bigq), or int64 num / den (den >= 1); target TR_REGION_NEED_CPU / _DEVICE */
int32_t tr_region_lookup_need(const tr_region_set *, uint64_t npoints, const tr_bigq *const *values, int32_t mode,
                              uint64_t max_ties, int32_t target, tr_math_need *need);
tr_region_lookup *tr_region_lookup_create(const tr_region_set *, uint64_t npoints, const tr_bigq *const *values,
                                          int32_t mode, uint64_t max_ties, const tr_math_budget *, tr_math_refusal *,
                                          int32_t *error);
int32_t tr_region_lookup_need_q64(const tr_region_set *, uint64_t npoints, const int64_t *num, const int64_t *den,
                                  int32_t mode, uint64_t max_ties, int32_t target, tr_math_need *need);
tr_region_lookup *tr_region_lookup_create_q64(const tr_region_set *, uint64_t npoints, const int64_t *num,
                                              const int64_t *den, int32_t mode, uint64_t max_ties,
                                              const tr_math_budget *, tr_math_refusal *, int32_t *error);
int32_t tr_region_lookup_get_view(const tr_region_lookup *, tr_region_view *);
uint64_t tr_region_lookup_bound_bits(const tr_region_lookup *);
/* the CPU evaluator (output = npoints): out[npoints], ties[npoints x max_ties] (NULL when max_ties = 0) */
int32_t tr_region_lookup_eval(const tr_region_view *, tr_region_answer *out, uint64_t *ties, const tr_math_budget *,
                              tr_math_refusal *);
void tr_region_lookup_free(tr_region_lookup *);

/* ------------------------------------------------------------------ the CUDA library (math_region_gpu.cu) */
int tr_mgpu_region_locate(const tr_region_view *view, tr_region_answer *out, uint64_t *ties,
                          const tr_math_budget *budget, tr_math_refusal *refusal);
#ifdef __cplusplus
}
#endif
#endif
