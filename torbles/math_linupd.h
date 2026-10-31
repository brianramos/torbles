/* LIN-003: leave-group-out and low-rank model updates on LIN-001 / LIN-002 states.
 * SPDX-License-Identifier: Apache-2.0
 *
 * ---- Models ---------------------------------------------------------------------------------------------------------
 * A model is an immutable solved square system A x = b with exact rational entries (FND-009 big rationals): the normal
 * equations of a fit of one group of a LIN-001 moment state (value units: A = X^T X + lambda D on the design columns,
 * D the identity on the features, never on the intercept; b = X^T y; LIN-002's fit_spec: group, intercept, feature
 * subset, exact ridge lambda = num / den >= 0), or any square rational system. Its solution set is canonical and
 * verified exactly, as LIN-002's (math_lin.h): UNIQUE (x), FAMILY (x with the free unknowns 0, plus the canonical
 * nullspace basis), NONE (a certificate y with y^T A = 0, y^T b = 1); rank and pivot columns exact. With
 * TR_LINUPD_INVERSE a nonsingular model also keeps A^-1 exactly (fraction-free Gauss-Jordan elimination of
 * [M | diag(L) | c] over big integers, M = diag(L) A the row-scaled integer matrix; A A^-1 = I verified exactly) and
 * det(A). A regression model keeps A0 (no ridge), y^T y, sum y and the rows n, and the statistics of LIN-002's fit
 * (rss = y^T y - 2 x.b + x^T A0 x, tss, r2, sigma2 = rss / (n - rank) without a ridge, y^T y, n, ||X x||^2), defined
 * for common-listwise states (pairwise: y^T y only).
 *
 * ---- Low-rank updates (Sherman-Morrison-Woodbury, exact) --------------------------------------------------------------
 * tr_linupd_update: A' = A + U V^T (U, V: n x r rational, r <= TR_LINUPD_MAX_RANK), b' = b + db. With the inverse
 * kept, the capacitance K = I_r + V^T A^-1 U is formed exactly and det(K) (for r = 1 the update denominator
 * 1 + v^T A^-1 u) decides: det(A') = det(A) det(K), so det(K) = 0 exactly when A' is singular. det(K) != 0: A'^-1 =
 * A^-1 - (A^-1 U) K^-1 (V^T A^-1) and x' = A'^-1 b' (method WOODBURY), x' verified exactly (A' x' = b'). det(K) = 0
 * (or a base model without an inverse) follows the declared contract: TR_LINUPD_REBUILD solves A' x = b' exactly
 * again (LIN-002's canonical solution set: FAMILY / NONE when singular; method REBUILT, fallback = 1, no inverse),
 * TR_LINUPD_FAIL refuses (TR_MATH_SINGULAR) and builds nothing. The denominator det(K) is part of the result.
 * tr_linupd_rows: deleting (sign -1) or inserting (sign +1) r complete rows (value units: the model's feature values in
 * design order, and the target) of a listwise regression model is the update U = sign Z^T, V = Z^T (z = the design
 * row, 1 for the intercept), db = sign Z^T y, with y^T y, sum y and n updated exactly (deleting more rows than n is
 * TR_MATH_INVALID; a deleted row that was never in the data is not detectable from the normal equations). For one
 * row the denominator is 1 - h for a deletion (h = z^T A^-1 z, the leverage) and 1 + h for an insertion.
 * tr_linupd_ridge: lambda -> lambda' is A0 + lambda' D: WOODBURY as the rank-p update (lambda' - lambda) E E^T (E the
 * p penalized columns of the identity), or REBUILD from A0 directly; both give the same exact model.
 * tr_linupd_subtract: the model's normal equations minus (sign -1) or plus (+1) the Gram matrix of a group of a
 * compatible moment state (the same columns, target, policy and decimal scales as the model's origin), rebuilt
 * exactly: the leave-group-out normal equations of a total model.
 *
 * ---- Leave-one-group-out (LOGO) ---------------------------------------------------------------------------------------
 * A LOGO plan of a moment state and a fit spec holds, for each selected group g, the complement states total - g
 * (every slot: counts and 192-bit sums subtract exactly), whose normal equations are those of the data without group
 * g. Its integer form (the kernel's and the verification's): row u of the value-unit system times den 10^(s_u + T)
 * (T = the largest decimal scale of the design and the target), entries M[u][w] = den G[u][w] 10^(T - s_w) (+ num
 * 10^(s_u + T) on a penalized diagonal), c[u] = den G[u][y] 10^(T - s_y). tr_linupd_logo_solve solves every system
 * exactly on the CPU (LIN-002's Bareiss elimination); the CUDA path (tr_mgpu_linupd_logo, one work item per (group,
 * prime)) forms each complement system modulo primes in [2^61, 2^62) from the copied group states and runs LIN-002's
 * modular Gauss-Jordan elimination (tr_lin_modular_rref); tr_linupd_logo_finish chooses the agreeing primes (highest
 * rank, smallest pivot columns), Chinese-remainders the det-scaled reduced row echelon form within the Hadamard bound
 * of each integer system and verifies the canonical solution exactly against it (more primes when short or when a
 * check fails: TR_MATH_NOT_CONVERGED with the primes a rerun needs). Both paths give the same canonical models.
 *
 * ---- Floating variant (a declared numerical contract) ------------------------------------------------------------------
 * tr_linupd_float_solve: x~ in binary64 from an approximate inverse R (Gauss-Jordan with partial pivoting on the
 * midpoint of A's tight enclosure, one refinement step), certified by interval arithmetic (math_carriers.h, outward
 * rounding): alpha >= ||I - R A||_inf over the enclosure of A; when alpha < 1, A is nonsingular and
 * ||x* - x~||_inf <= bound = ||R (b - A x~)||_inf / (1 - alpha) (enclosed residual). Returned only when alpha < 1 and
 * bound <= tol; otherwise TR_MATH_NOT_CONVERGED (the condition bound fails; the info, with the condition estimate
 * kappa >= ||A||_inf ||R||_inf, is written), TR_MATH_SINGULAR for a model whose exact system is singular.
 *
 * ---- Budgets (math_budgets.h, before allocation) -------------------------------------------------------------------
 *   model:   variables = n, tensor = n (2n + 1) (the elimination), output = n (n + 1) (solution, inverse)
 *   update:  variables = n, tensor = n r + r^2 + n^2, output = n (n + 1)
 *   LOGO:    states = groups x slots (the copied states), tensor = systems x n (n + 1), output = systems, bytes = the
 *            copies; CUDA: bytes = the device copies and residues (tr_linupd_logo_device_bytes)
 *   float:   tensor = n^2, bytes = 6 n^2 doubles
 * Refusals: TR_MATH_UNSUPPORTED (n > TR_LIN_MAX_UNKNOWNS, r > TR_LINUPD_MAX_RANK, more than TR_LINUPD_MAX_PRIMES
 * primes), TR_MATH_DOMAIN_ERROR (an incompatible moment state, a rows update of a pairwise or non-regression model),
 * TR_MATH_SINGULAR (FAIL contract), TR_MATH_NOT_CONVERGED, TR_MATH_INVALID, _MEMORY, _BUDGET. Objects are immutable
 * and individually owned. */
#ifndef TORBLES_MATH_LINUPD_H
#define TORBLES_MATH_LINUPD_H
#include <stdint.h>
#include <string.h>
#include "math_num.h"
#include "math_budgets.h"
#include "math_carriers.h"
#include "math_lin.h"
#ifdef __cplusplus
extern "C" {
#endif

#define TR_LINUPD_ABI 1
#define TR_LINUPD_MAX_RANK 64
#define TR_LINUPD_MAX_PRIMES 1024
enum { TR_LINUPD_FAIL = 0, TR_LINUPD_REBUILD = 1 };                    /* the contract of a singular update */
enum { TR_LINUPD_INVERSE = 1 };                                        /* flags: keep A^-1 */
enum { TR_LINUPD_METHOD_WOODBURY = 1, TR_LINUPD_METHOD_REBUILD = 2 };  /* tr_linupd_ridge */
/* how a model's solution was obtained */
enum { TR_LINUPD_BUILT = 1, TR_LINUPD_WOODBURY = 2, TR_LINUPD_REBUILT = 3, TR_LINUPD_SUBTRACTED = 4,
       TR_LINUPD_MODULAR = 5 };
/* parts of a model (tr_linupd_export, packed rationals: math_lin.h tr_lin_qbuf) */
enum { TR_LINUPD_PART_X = 0, TR_LINUPD_PART_NULL = 1, TR_LINUPD_PART_CERT = 2, TR_LINUPD_PART_A = 3,
       TR_LINUPD_PART_A0 = 4, TR_LINUPD_PART_B = 5, TR_LINUPD_PART_INVERSE = 6, TR_LINUPD_PART_STATS = 7,
       TR_LINUPD_PART_DET = 8, TR_LINUPD_PART_DENOMINATOR = 9, TR_LINUPD_PART_DESIGN = 10, TR_LINUPD_PARTS = 11 };

typedef struct {
    int32_t n, kind, rank, method;           /* unknowns; TR_LIN_UNIQUE / FAMILY / NONE; rank; TR_LINUPD_BUILT .. */
    int32_t regression, listwise, intercept, ridged;
    int32_t inverse, fallback, update_rank, denominator_sign;   /* sign of det(K): -1, 0, 1; 2 = no update */
    uint64_t rows, group, pivots, primes;    /* regression rows n; the group (LOGO: the group left out) */
    int32_t design[TR_LIN_MAX_COLS];         /* augmented column of each unknown (regression) */
} tr_linupd_info;                            /* 152 bytes */

/* the LOGO systems as the CUDA kernel reads them */
typedef struct {
    uint64_t ngroups, nsys;                  /* groups of the state; systems (selected groups) */
    int32_t k, nslots, n, ycol;              /* augmented columns, slots, unknowns, augmented column of the target */
    int32_t design[TR_LIN_MAX_COLS];         /* augmented column of each unknown */
    int32_t pen[TR_LIN_MAX_COLS];            /* 1: unknown u carries the ridge */
    int32_t scale[TR_LIN_MAX_COLS];          /* decimal scale of each augmented column */
    int32_t T, reserved;
    int64_t ridge_num, ridge_den;
    const uint64_t *sum;                     /* ngroups x nslots x 3 words (192-bit two's complement) */
    const uint64_t *total;                   /* nslots x 3 words */
    const uint64_t *groups;                  /* nsys group ids */
} tr_linupd_logo_view;                       /* 296 bytes */

typedef struct {
    double alpha, bound, kappa, xnorm, tol;  /* see the floating variant */
    int32_t certified, reason;               /* reason: 0 certified, 1 zero pivot, 2 alpha >= 1, 3 bound > tol */
} tr_linupd_float_info;                      /* 48 bytes */

typedef struct tr_linupd_model_s tr_linupd_model;
typedef struct tr_linupd_logo_s tr_linupd_logo;

/* ------------------------------------------------------------------ models (libalg) */
tr_linupd_model *tr_linupd_fit(const tr_lin_moments *, const tr_lin_fit_spec *, int32_t flags, const tr_math_budget *,
                               tr_math_refusal *, int32_t *error);
tr_linupd_model *tr_linupd_system(int32_t n, const tr_lin_qbuf *A, const tr_lin_qbuf *b, int32_t flags,
                                  const tr_math_budget *, tr_math_refusal *, int32_t *error);
/* U, V: n x r row-major; db: n values or NULL (no change) */
tr_linupd_model *tr_linupd_update(const tr_linupd_model *, int32_t r, const tr_lin_qbuf *U, const tr_lin_qbuf *V,
                                  const tr_lin_qbuf *db, int32_t fallback, const tr_math_budget *, tr_math_refusal *,
                                  int32_t *error);
/* Z: r x f row-major (f = the design's feature unknowns, design order), y: r values */
tr_linupd_model *tr_linupd_rows(const tr_linupd_model *, int32_t sign, int32_t r, const tr_lin_qbuf *Z,
                                const tr_lin_qbuf *y, int32_t fallback, const tr_math_budget *, tr_math_refusal *,
                                int32_t *error);
tr_linupd_model *tr_linupd_ridge(const tr_linupd_model *, int64_t num, int64_t den, int32_t method, int32_t fallback,
                                 const tr_math_budget *, tr_math_refusal *, int32_t *error);
tr_linupd_model *tr_linupd_subtract(const tr_linupd_model *, const tr_lin_moments *, uint64_t group, int32_t sign,
                                    int32_t flags, const tr_math_budget *, tr_math_refusal *, int32_t *error);
int32_t tr_linupd_get_info(const tr_linupd_model *, tr_linupd_info *);
uint64_t tr_linupd_count(const tr_linupd_model *, int32_t part);       /* values of a part (0 when absent) */
uint64_t tr_linupd_limbs(const tr_linupd_model *, int32_t part);       /* limbs its export needs */
int32_t tr_linupd_export(const tr_linupd_model *, int32_t part, tr_lin_qbuf *out);
/* 1 when two models hold the same canonical solution, system, statistics and inverse; 0 otherwise */
int32_t tr_linupd_equal(const tr_linupd_model *, const tr_linupd_model *);
int32_t tr_linupd_float_solve(const tr_linupd_model *, double tol, double *x, tr_linupd_float_info *info,
                              const tr_math_budget *, tr_math_refusal *);
void tr_linupd_free(tr_linupd_model *);

/* ------------------------------------------------------------------ leave-one-group-out */
/* groups: nsel group ids (NULL: every group, nsel ignored); the fit spec's group field is ignored */
tr_linupd_logo *tr_linupd_logo_plan(const tr_lin_moments *, const tr_lin_fit_spec *, uint64_t nsel,
                                    const uint64_t *groups, const tr_math_budget *, tr_math_refusal *, int32_t *error);
int32_t tr_linupd_logo_get_view(const tr_linupd_logo *, tr_linupd_logo_view *);
uint64_t tr_linupd_logo_primes_needed(const tr_linupd_logo *);         /* max over systems, with the margin */
uint64_t tr_linupd_logo_bytes(const tr_linupd_logo *, uint64_t nprimes);   /* the CUDA path's device bytes */
/* out[nsys]: the exact models on the CPU (flags: TR_LINUPD_INVERSE) */
int32_t tr_linupd_logo_solve(const tr_linupd_logo *, int32_t flags, tr_linupd_model **out, const tr_math_budget *,
                             tr_math_refusal *);
/* the CPU reference of tr_mgpu_linupd_logo: residues[nsys * nprimes * (4 + n (n + 1))] */
int32_t tr_linupd_logo_residues(const tr_linupd_logo_view *, const tr_gfp *primes, uint64_t nprimes,
                                uint64_t *residues, const tr_math_budget *, tr_math_refusal *);
/* CRT, reconstruction and exact verification; TR_MATH_NOT_CONVERGED with *need = the primes a rerun needs */
int32_t tr_linupd_logo_finish(const tr_linupd_logo *, const tr_gfp *primes, uint64_t nprimes, const uint64_t *residues,
                              tr_linupd_model **out, uint64_t *need, const tr_math_budget *, tr_math_refusal *);
void tr_linupd_logo_free(tr_linupd_logo *);

/* ------------------------------------------------------------------ the CUDA library (math_linupd_gpu.cu)
 * tr_mgpu_linupd_logo: residues equal to tr_linupd_logo_residues bit for bit (budget NULL = defaults) */
int tr_mgpu_linupd_logo(const tr_linupd_logo_view *, const tr_gfp *primes, uint64_t nprimes, const tr_math_budget *,
                        uint64_t *residues, tr_math_refusal *);

/* ================================================================ shared helpers (CPU code, CUDA host and kernels) */
/* the residue words of one (system, prime): the header and the n x (n + 1) det-scaled reduced matrix */
TR_NUM_HD uint64_t tr_linupd_block(const tr_linupd_logo_view *v) { return 4 + (uint64_t)v->n * (uint64_t)(v->n + 1); }
/* the slot of Gram entry (a, b) of k augmented columns (tr_lin_gram_slot of math_lin.h, usable in device code) */
TR_NUM_HD int32_t tr_linupd_slot(int32_t k, int32_t a, int32_t b) {
    if (a > b) { int32_t t = a; a = b; b = t; }
    return a * k - a * (a - 1) / 2 + (b - a);
}
/* bytes tr_mgpu_linupd_logo copies to and allocates on the device */
static inline uint64_t tr_linupd_logo_device_bytes(const tr_linupd_logo_view *v, uint64_t nprimes) {
    uint64_t b = tr_budget_mul(tr_budget_mul(v->ngroups, (uint64_t)v->nslots), 24);
    b = tr_budget_add(b, tr_budget_add(tr_budget_mul((uint64_t)v->nslots, 24), tr_budget_mul(v->nsys, 8)));
    b = tr_budget_add(b, tr_budget_mul(nprimes, sizeof(tr_gfp)));
    return tr_budget_add(b, tr_budget_mul(tr_budget_mul(tr_budget_mul(v->nsys, nprimes), tr_linupd_block(v)), 8));
}
static inline int tr_linupd_logo_view_valid(const tr_linupd_logo_view *v) {
    if (!v || !v->sum || !v->total || !v->groups || !v->nsys || v->k < 2 || v->k > TR_LIN_MAX_COLS || v->n < 1 ||
        v->n >= v->k || v->ycol != v->k - 1 || v->nslots < v->k * (v->k + 1) / 2 || v->T < 0 || v->T > 36 ||
        v->ridge_num < 0 || v->ridge_den < 1) return 0;
    for (int32_t u = 0; u < v->n; u++) if (v->design[u] < 0 || v->design[u] >= v->ycol) return 0;
    for (int32_t i = 0; i < v->k; i++) if (v->scale[i] < 0 || v->scale[i] > v->T) return 0;
    for (uint64_t s = 0; s < v->nsys; s++) if (v->groups[s] >= v->ngroups) return 0;
    return 1;
}
/* 10^e mod p */
TR_NUM_HD uint64_t tr_linupd_pow10(const tr_gfp *f, int32_t e) {
    uint64_t r = 1 % f->p, t = 10 % f->p;
    for (uint32_t x = (uint32_t)e; x; x >>= 1) { if (x & 1) r = tr_gfp_mul_hd(f, r, t); t = tr_gfp_mul_hd(f, t, t); }
    return r;
}
/* the complement system of LOGO system s modulo p into blk[4 ..] (n x (n + 1), row-major), then its det-scaled
 * reduced row echelon form and the header blk[0..3] (tr_lin_modular_rref) */
TR_NUM_HD void tr_linupd_logo_system(const tr_linupd_logo_view *v, const tr_gfp *f, uint64_t s, uint64_t *blk) {
    const uint64_t g = v->groups[s];
    const int32_t n = v->n, c = n + 1, k = v->k;
    const uint64_t den = (uint64_t)v->ridge_den % f->p, num = (uint64_t)v->ridge_num % f->p;
    for (int32_t u = 0; u < n; u++) {
        const int32_t a = v->design[u];
        for (int32_t w = 0; w < c; w++) {
            const int32_t b = w < n ? v->design[w] : v->ycol;
            const int32_t slot = tr_linupd_slot(k, a, b);
            uint64_t t = tr_lin_reduce_words(f, v->total + 3 * (uint64_t)slot, 3);
            uint64_t q = tr_lin_reduce_words(f, v->sum + 3 * (g * (uint64_t)v->nslots + (uint64_t)slot), 3);
            uint64_t e = tr_gfp_mul_hd(f, tr_gfp_mul_hd(f, den, tr_gfp_sub_hd(f, t, q)), tr_linupd_pow10(f, v->T - v->scale[b]));
            if (w == u && v->pen[u])
                e = tr_gfp_add_hd(f, e, tr_gfp_mul_hd(f, num, tr_linupd_pow10(f, v->scale[a] + v->T)));
            blk[4 + (uint64_t)u * (uint64_t)c + (uint64_t)w] = e;
        }
    }
    tr_lin_modular_rref(f, blk + 4, n, c, blk);
}

#ifdef __cplusplus
}
#endif
#endif
