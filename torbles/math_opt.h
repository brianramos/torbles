/* OPT-001: exact minimizers of quadratic objectives q(t) = t^T A t / 2 - b^T t + c over R^n or a box.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Objectives. A (symmetric), b and c are exact rationals (math_carriers.h): given directly, built from a LIN-001
 * moment state (the least-squares loss sum (y - z^T t)^2 + lambda |t_features|^2 of a fit: A = 2 (Z^T Z + lambda I_f),
 * b = 2 Z^T y, c = y^T y in value units), or read from a first-slice polynomial of total degree <= 2 in its
 * parameters (mathematics.h: c t^e terms; a parameter's square c t_i^2 gives A_ii = 2c, c t_i t_j gives A_ij = c,
 * c t_i gives b_i = -c). A nonsymmetric A, a polynomial of degree > 2 (TR_MATH_UNSUPPORTED) or one in a variable
 * that is not a parameter (TR_MATH_INVALID) is refused.
 *
 * Unconstrained (n <= TR_OPT_MAX_PARAMS), decided exactly from the inertia of A and the exact solution set of
 * A t = b (math_lin.h), never by dividing by zero:
 *   A positive definite                  UNIQUE minimizer t* = A^-1 b, value c - b^T t* / 2;
 *   A PSD singular, b in range(A)        NONUNIQUE: the minimal value on t0 + span(N) (N the kernel basis);
 *   A PSD, b not in range(A)             UNBOUNDED: d in ker A with b^T d > 0, q(s d) = c - s b^T d (SLOPE);
 *   A with a negative eigenvalue         UNBOUNDED: d with d^T A d < 0, q(s d) -> -inf (CURVATURE).
 * Box (n <= TR_OPT_MAX_BOX): each parameter has both bounds (lo_i <= hi_i, rational) or none (free); a one-sided
 * bound is TR_MATH_UNSUPPORTED, lo_i > hi_i is TR_MATH_INFEASIBLE. Boundedness is decided first and exactly: with
 * free coordinates U, q is unbounded iff A_UU has a negative eigenvalue (CURVATURE along U) or some kernel vector v of
 * A_UU has v^T (A_UK t_K - b_U) != 0 at a feasible t_K (SLOPE, a vertex is exhibited); otherwise the minimum is
 * attained. It is then found by enumerating the 3^|K| faces of the bounded coordinates K (each at its lower bound,
 * its upper bound, or free): the minimum lies at a stationary point of some face whose restricted kernel vanishes on
 * the bounded coordinates, so per face the restricted system A_FF t_F = b_F - A_F,fixed t_fixed is solved exactly
 * (math_lin.h), and a candidate is kept when its bounded coordinates are unique and feasible and the KKT conditions
 * hold exactly (the gradient A t - b is 0 on the free coordinates, >= 0 at a lower bound, <= 0 at an upper bound).
 * The minimum over the candidates is the exact optimum (boundary optima included); several distinct optimal points,
 * or directions of constant value along free coordinates, make it NONUNIQUE.
 * Face filter (the data-parallel part: the CUDA kernel of math_opt_gpu.cu, one work item per face, and the same
 * inline code on the CPU): per face, interval Gaussian elimination on the enclosures of A, b and the bounds
 * (math_carriers.h, outward rounding) discards a face only when it rigorously cannot hold the optimum: its unique
 * stationary point lies outside a bound (INFEASIBLE), violates a KKT sign (KKT), or a diagonal entry of A_FF is
 * negative (NOT_PSD); a pivot interval containing 0 leaves the face UNDECIDED (checked exactly). The exact answer does
 * not depend on the filter, which only skips work; both paths give the same statuses bit for bit.
 * Results are verified exactly: an UNBOUNDED certificate (direction d, anchor t0, the curvature d^T A d < 0 or the
 * slope g(t0)^T d < 0 with d^T A d = 0, feasible along the ray), the KKT conditions at the reported optimum.
 * Budgets: variables = n, degree = 2, tensor = n (n + 1) (the augmented system [A | b]), iterations = faces (3^|K|),
 * output = n (a box: n (faces + 1), every optimum found).
 * Errors: TR_MATH_INVALID, _MEMORY, _BUDGET, _UNSUPPORTED, _INFEASIBLE, _DOMAIN_ERROR. */
#ifndef TORBLES_MATH_OPT_H
#define TORBLES_MATH_OPT_H
#include <stdint.h>
#include "math_num.h"
#include "math_budgets.h"
#include "math_carriers.h"
#include "math_lin.h"
#include "mathematics.h"
#ifdef __cplusplus
extern "C" {
#endif

#define TR_OPT_MAX_PARAMS TR_LIN_MAX_UNKNOWNS
#define TR_OPT_MAX_BOX 8
enum { TR_OPT_UNIQUE = 1, TR_OPT_NONUNIQUE = 2, TR_OPT_UNBOUNDED = 3 };
enum { TR_OPT_CERT_CURVATURE = 1, TR_OPT_CERT_SLOPE = 2 };
enum { TR_OPT_METHOD_UNCONSTRAINED = 1, TR_OPT_METHOD_FACES = 2 };
enum { TR_OPT_FACE_CANDIDATE = 0, TR_OPT_FACE_UNDECIDED = 1, TR_OPT_FACE_INFEASIBLE = 2, TR_OPT_FACE_KKT = 3,
       TR_OPT_FACE_NOT_PSD = 4, TR_OPT_FACE_DUPLICATE = 5, TR_OPT_FACE_STATES = 6 };
enum { TR_OPT_PART_A = 0, TR_OPT_PART_B = 1, TR_OPT_PART_C = 2,                    /* the objective (tr_opt_quad) */
       TR_OPT_PART_POINT = 3,      /* an optimal point (n) */
       TR_OPT_PART_VALUE = 4,      /* the optimal value (1) */
       TR_OPT_PART_FAMILY = 5,     /* directions of constant optimal value (family x n) */
       TR_OPT_PART_DIRECTION = 6,  /* UNBOUNDED: the ray's direction d (n) */
       TR_OPT_PART_ANCHOR = 7,     /* UNBOUNDED: the ray's feasible origin t0 (n) */
       TR_OPT_PART_RATE = 8,       /* UNBOUNDED: d^T A d (< 0, curvature) or g(t0)^T d (< 0, slope) (1) */
       TR_OPT_PART_GRADIENT = 9,   /* A t - b at the optimum (n) */
       TR_OPT_PART_ACTIVE = 10,    /* per parameter: -1 at its lower bound, 1 at its upper bound, 0 free (n) */
       TR_OPT_PART_OPTIMA = 11,    /* every distinct optimal point found (optima x n) */
       TR_OPT_PARTS = 12 };

typedef struct {
    int32_t n, reserved;
    uint8_t *has_lo, *has_hi;               /* n flags each (both 1 or both 0) */
    tr_lin_qbuf lo, hi;                     /* n values each (ignored where a parameter has no bound) */
} tr_opt_box;                               /* 120 bytes */

typedef struct {
    int32_t status, n, bounded, free;       /* TR_OPT_*; parameters; finite-bound and free coordinates */
    int32_t certificate, family, optima, kkt;   /* TR_OPT_CERT_*; family directions; distinct optima; KKT checked */
    uint64_t faces, exact;                  /* faces enumerated; faces solved exactly */
    uint64_t filtered[TR_OPT_FACE_STATES];  /* faces per filter status */
    int32_t method, convex;                 /* TR_OPT_METHOD_*; 1 when A is positive semidefinite */
} tr_opt_info;                              /* 104 bytes */

/* the face filter's input (n <= 8): enclosures of A, b and the bounds, by coordinate */
typedef struct {
    int32_t n, nk;
    int32_t kidx[TR_OPT_MAX_BOX];           /* the bounded coordinates, increasing */
    uint64_t faces;                         /* 3^nk */
    uint32_t degenerate, reserved;          /* bit i: lo_i == hi_i (one state only) */
    double alo[TR_OPT_MAX_BOX * TR_OPT_MAX_BOX], ahi[TR_OPT_MAX_BOX * TR_OPT_MAX_BOX];
    double blo[TR_OPT_MAX_BOX], bhi[TR_OPT_MAX_BOX];
    double llo[TR_OPT_MAX_BOX], lhi[TR_OPT_MAX_BOX], ulo[TR_OPT_MAX_BOX], uhi[TR_OPT_MAX_BOX];
} tr_opt_face_view;                         /* 1464 bytes */

typedef struct tr_opt_quad_s tr_opt_quad;
typedef struct tr_opt_result_s tr_opt_result;

/* ------------------------------------------------------------------ the face filter (CPU and CUDA kernel) */
TR_NUM_HD double tr_opt_mig(tr_math_interval x) { return x.lo > 0 ? x.lo : x.hi < 0 ? -x.hi : 0.0; }
TR_NUM_HD tr_math_interval tr_opt_iv(double lo, double hi) { tr_math_interval r; r.lo = lo; r.hi = hi; return r; }
/* the status of face `face` (base-3 digits over the bounded coordinates: 0 lower, 1 upper, 2 free) */
TR_NUM_HD int32_t tr_opt_face_filter(const tr_opt_face_view *v, uint64_t face) {
    const int32_t n = v->n;
    int32_t state[TR_OPT_MAX_BOX], bounded[TR_OPT_MAX_BOX], F[TR_OPT_MAX_BOX], nf = 0;
    tr_math_interval t[TR_OPT_MAX_BOX], M[TR_OPT_MAX_BOX][TR_OPT_MAX_BOX + 1];
    for (int32_t i = 0; i < n; i++) { state[i] = 2; bounded[i] = 0; }
    uint64_t f = face;
    for (int32_t k = 0; k < v->nk; k++) {
        int32_t d = (int32_t)(f % 3), i = v->kidx[k];
        f /= 3;
        if (((v->degenerate >> i) & 1) && d) return TR_OPT_FACE_DUPLICATE;
        state[i] = d; bounded[i] = 1;
    }
    for (int32_t i = 0; i < n; i++) {
        if (state[i] == 2) F[nf++] = i;
        else t[i] = state[i] == 0 ? tr_opt_iv(v->llo[i], v->lhi[i]) : tr_opt_iv(v->ulo[i], v->uhi[i]);
    }
    for (int32_t a = 0; a < nf; a++) if (v->ahi[F[a] * n + F[a]] < 0) return TR_OPT_FACE_NOT_PSD;
    for (int32_t a = 0; a < nf; a++) {
        tr_math_interval r = tr_opt_iv(v->blo[F[a]], v->bhi[F[a]]);
        for (int32_t c = 0; c < nf; c++) M[a][c] = tr_opt_iv(v->alo[F[a] * n + F[c]], v->ahi[F[a] * n + F[c]]);
        for (int32_t i = 0; i < n; i++)
            if (state[i] != 2) r = tr_interval_sub(r, tr_interval_mul(tr_opt_iv(v->alo[F[a] * n + i], v->ahi[F[a] * n + i]), t[i]));
        M[a][nf] = r;
    }
    for (int32_t col = 0; col < nf; col++) {                 /* elimination, pivots of largest mignitude */
        int32_t best = -1;
        double bm = 0.0;
        for (int32_t r = col; r < nf; r++) { double m = tr_opt_mig(M[r][col]); if (m > bm) { bm = m; best = r; } }
        if (best < 0) return TR_OPT_FACE_UNDECIDED;
        if (best != col)
            for (int32_t c = 0; c <= nf; c++) { tr_math_interval x = M[best][c]; M[best][c] = M[col][c]; M[col][c] = x; }
        for (int32_t r = col + 1; r < nf; r++) {
            tr_math_interval q = tr_interval_div(M[r][col], M[col][col]);
            for (int32_t c = col; c <= nf; c++) M[r][c] = tr_interval_sub(M[r][c], tr_interval_mul(q, M[col][c]));
        }
    }
    for (int32_t a = nf; a-- > 0;) {
        tr_math_interval s = M[a][nf];
        for (int32_t c = a + 1; c < nf; c++) s = tr_interval_sub(s, tr_interval_mul(M[a][c], t[F[c]]));
        if (tr_iv_contains_zero(M[a][a])) return TR_OPT_FACE_UNDECIDED;
        t[F[a]] = tr_interval_div(s, M[a][a]);
    }
    for (int32_t a = 0; a < nf; a++) {
        int32_t i = F[a];
        if (bounded[i] && (t[i].hi < v->llo[i] || t[i].lo > v->uhi[i])) return TR_OPT_FACE_INFEASIBLE;
    }
    for (int32_t i = 0; i < n; i++) {                        /* KKT signs at the fixed coordinates */
        if (state[i] == 2 || ((v->degenerate >> i) & 1)) continue;
        tr_math_interval g = tr_interval_neg(tr_opt_iv(v->blo[i], v->bhi[i]));
        for (int32_t j = 0; j < n; j++) g = tr_interval_add(g, tr_interval_mul(tr_opt_iv(v->alo[i * n + j], v->ahi[i * n + j]), t[j]));
        if ((state[i] == 0 && g.hi < 0) || (state[i] == 1 && g.lo > 0)) return TR_OPT_FACE_KKT;
    }
    return TR_OPT_FACE_CANDIDATE;
}
static inline uint64_t tr_opt_faces_bytes(const tr_opt_face_view *v) { return tr_budget_add(v->faces, 8); }

/* ------------------------------------------------------------------ objectives (libalg) */
tr_opt_quad *tr_opt_quad_create(int32_t n, const tr_lin_qbuf *A, const tr_lin_qbuf *b, const tr_lin_qbuf *c,
                                const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error);
/* a first-slice polynomial (mathematics.h) in the parameters vars[0..n) (variable indices, distinct) */
tr_opt_quad *tr_opt_quad_from_poly(const tr_math_poly *, int32_t n, const int32_t *vars, const tr_math_budget *budget,
                                   tr_math_refusal *refusal, int32_t *error);
/* the least-squares loss of a fit of a LIN-001 state (unknowns in the order of the fit's design) */
tr_opt_quad *tr_opt_quad_from_moments(const tr_lin_moments *, const tr_lin_fit_spec *, const tr_math_budget *budget,
                                      tr_math_refusal *refusal, int32_t *error);
int32_t tr_opt_quad_dim(const tr_opt_quad *);
/* parts TR_OPT_PART_A (n x n), _B (n), _C (1); the design columns of a moment objective with part _ACTIVE */
uint64_t tr_opt_quad_count(const tr_opt_quad *, int32_t part);
uint64_t tr_opt_quad_limbs(const tr_opt_quad *, int32_t part);
int32_t tr_opt_quad_export(const tr_opt_quad *, int32_t part, tr_lin_qbuf *out);
/* q(t) exactly for t (n values): one value in *out (owned by the caller's qbuf) */
int32_t tr_opt_quad_value(const tr_opt_quad *, const tr_lin_qbuf *t, tr_lin_qbuf *out, const tr_math_budget *budget,
                          tr_math_refusal *refusal);
void tr_opt_quad_free(tr_opt_quad *);

/* ------------------------------------------------------------------ minimization (libalg) */
/* the filter's input for a box (validated as by tr_opt_minimize); a box without bounded coordinates has 1 face */
int32_t tr_opt_face_view_get(const tr_opt_quad *, const tr_opt_box *, tr_opt_face_view *out);
/* the CPU filter (the kernel's code): status[view->faces] */
int32_t tr_opt_faces(const tr_opt_face_view *, uint8_t *status);
/* the exact minimum; box NULL: unconstrained; status: the filter's statuses (nstatus = faces, e.g. from the CUDA
 * kernel) or NULL (computed here on the CPU) */
tr_opt_result *tr_opt_minimize(const tr_opt_quad *, const tr_opt_box *box, const uint8_t *status, uint64_t nstatus,
                               const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error);
int32_t tr_opt_result_get_info(const tr_opt_result *, tr_opt_info *);
uint64_t tr_opt_result_count(const tr_opt_result *, int32_t part);
uint64_t tr_opt_result_limbs(const tr_opt_result *, int32_t part);
int32_t tr_opt_result_export(const tr_opt_result *, int32_t part, tr_lin_qbuf *out);
void tr_opt_result_free(tr_opt_result *);

/* ------------------------------------------------------------------ the CUDA library (math_opt_gpu.cu)
 * status[view->faces] = tr_opt_face_filter of every face (one work item per face), equal to tr_opt_faces bit for bit;
 * the device copies (tr_opt_faces_bytes) are budgeted (budget NULL = defaults; also iterations = faces). */
int tr_mgpu_opt_faces(const tr_opt_face_view *view, const tr_math_budget *budget, uint8_t *status, tr_math_refusal *refusal);
#ifdef __cplusplus
}
#endif
#endif
