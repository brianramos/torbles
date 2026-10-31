/* LIN-001 and LIN-002: grouped exact moment matrices and regression systems over a data scope (math_scope.h), and the
 * small native operators that finalize them: exact rational solves, rank, solution sets, inertia, covariance,
 * correlation, regression statistics, and numerical symmetric eigensystems with a declared error contract.
 * SPDX-License-Identifier: Apache-2.0
 *
 * ---- LIN-001: moment states ----------------------------------------------------------------------------------------
 * Spec. d <= TR_LIN_MAX_FEATURES feature columns x1..xd and an optional target y, given as columns of the scope. The
 * augmented columns are z0 = 1 (the constant), z1..zd = the features, z(d+1) = y: k = d + 1 + (y given) columns.
 * The state of group g holds, for k(k+1)/2 Gram slots (a <= b, row-major upper triangle), the count of rows where the
 * slot is valid and the exact sum of z_a z_b over them; together they are the count n, the sums of x and y, X^T X,
 * X^T y and y^T y. `intercept` declares whether the regression design includes z0 (the slots are always kept).
 * Validity is a declared policy:
 *   TR_LIN_LISTWISE  (common-listwise): a row counts only when every feature and the target are present; every slot
 *                    has the same count and the matrix is the Gram matrix of the complete rows (positive semidefinite).
 *   TR_LIN_PAIRWISE  each slot over the rows where its own columns are present (z0 always is): per-slot counts. The
 *                    raw matrix is then the Gram matrix of the data with NULL read as 0 (so it stays positive
 *                    semidefinite), but it is not the Gram matrix of any set of complete rows, and the covariance and
 *                    correlation it defines (pair-complete means) may be indefinite. A fused pairwise state also
 *                    keeps, for every pair a < b of non-constant columns, the pair-complete marginals sum z_a, sum z_b,
 *                    sum z_a^2, sum z_b^2 over the rows where both are present (`marginals`), which covariance and
 *                    correlation use (SQL COVAR / CORR / REGR_* semantics). A state built from a bank has none.
 * Presence (bag multiplicity) is separate: rows[g] = selected rows of group g; a group is present iff rows[g] > 0
 * (with count 0 when none of its rows is valid); an absent group has rows 0.
 * Exactness. Sums are exact integers of the scope's logical values (decimal value = logical / 10^scale, so slot (a, b)
 * carries the scale s_a + s_b). Before accumulating, each slot is proven to fit 128 bits: N * M_a * M_b <= 2^127 - 1
 * (M_c = max(|min_c|, |max_c|, 1) from the scope's extremes, N the rows that can be valid). When some slot is not
 * proven, the accumulation uses 192-bit words, which hold every sum of int64 products over < 2^64 rows. width = 128
 * refuses an unproven request (TR_MATH_OVERFLOW), 192 forces the wide words, 0 chooses. Stored sums are always
 * 192-bit two's complement; nothing wraps. The CUDA path, a bank, and merges of the states of disjoint selections
 * (tr_lin_moments_combine) give the same states bit for bit.
 * Budgets (checked before allocation; tr_lin_moments_need): variables = scope columns used, degree = 2, support =
 * slots, tensor = groups x Gram slots, output = groups, states = groups x slots, bytes = every allocation (CPU) or
 * the device copies (CUDA). d > TR_LIN_MAX_FEATURES is TR_MATH_UNSUPPORTED.
 *
 * ---- LIN-002: exact operators ----------------------------------------------------------------------------------------
 * Systems A x = b (m <= TR_LIN_MAX_ROWS rows, n <= TR_LIN_MAX_UNKNOWNS unknowns) have rational entries (tr_lin_qbuf).
 * Each row is scaled to integers (by the lcm of its denominators); the solution set is then found by fraction-free
 * Gauss-Jordan elimination (Bareiss) over big integers (math_carriers.h) on the CPU, or by Gauss-Jordan elimination
 * modulo several primes (the CUDA kernel, one work item per system and prime) followed on the host by Chinese
 * remaindering of det-scaled reduced row echelon forms within a Hadamard bound. Either way the result is the
 * canonical description of the solution set and is verified exactly before it is returned:
 *   UNIQUE   rank = n: x;
 *   FAMILY   rank < n: x0 (free unknowns 0) + span(N), N the canonical basis (one vector per non-pivot column);
 *   NONE     inconsistent: a certificate y with y^T A = 0 and y^T b = 1;
 * never a division by zero; rank and pivot columns (of the reduced row echelon form) are exact.
 * Inertia of a symmetric rational matrix: symmetric fraction-free elimination with diagonal pivots (a zero diagonal
 * with a nonzero off-diagonal entry is resolved by an integer congruence), so (n+, n-, n0) are exact (Sylvester), the
 * class follows (PD, PSD singular, ND, NSD, indefinite, zero) with certificates: v with v^T A v < 0 (n- > 0), v with
 * v^T A v > 0 (n+ > 0), and an integer kernel basis (n0 vectors), each verified exactly.
 * Moment operators: the normal equations of group g (value units, an optional exact ridge penalty on the features
 * and a feature subset: deletion requests), the regression fit (coefficients as a solution set, RSS = ||y - X b||^2,
 * TSS, R^2, sigma^2 = RSS / (n - rank), the projection's squared norm ||X b||^2; statistics need a common-listwise
 * state), covariance (ddof 0 or 1) and
 * squared correlation with its sign, all exact rationals (undefined entries are reported as such, never divided).
 * Numerical eigensystems (APPROXIMATE): cyclic Jacobi in binary64 on the midpoint of an enclosure of a symmetric
 * matrix; returned with rigorous bounds, valid for every matrix in the enclosure (so for the exact one): per pair the
 * residual ||A v - lambda v||_2 (upper bound), the eigenvalue bound ||A v - lambda v||_2 / ||v||_2 (an eigenvalue of
 * A lies within it of lambda), and the orthogonality error max |V^T V - I|. Nothing more is claimed: exact input
 * moments give no stability guarantee for the eigenvectors (close eigenvalues make them ill-conditioned).
 *
 * Errors: TR_MATH_INVALID, _MEMORY, _OVERFLOW, _BUDGET (with the tr_math_refusal), _UNSUPPORTED, _DOMAIN_ERROR (an
 * operator that the state's policy does not define), _STALE (checked by the Python layer), _NOT_CONVERGED (Jacobi
 * sweeps, or too few primes for a modular batch: rerun with more). Objects are immutable and individually owned. */
#ifndef TORBLES_MATH_LIN_H
#define TORBLES_MATH_LIN_H
#include <stdint.h>
#include <string.h>
#include "math_num.h"
#include "math_budgets.h"
#include "math_scope.h"
#include "math_bank.h"
#include "math_carriers.h"
#ifdef __cplusplus
extern "C" {
#endif

#define TR_LIN_MAX_FEATURES 16
#define TR_LIN_MAX_COLS 18                  /* the constant, 16 features, the target */
#define TR_LIN_MAX_SLOTS 715                /* 171 Gram slots and 544 pair-complete marginals at k = 18 */
#define TR_LIN_MAX_ROWS 64                  /* exact systems: equations */
#define TR_LIN_MAX_UNKNOWNS 63              /* exact systems: unknowns (the augmented matrix has <= 64 columns) */
#define TR_LIN_WORDS 3                      /* stored sums: 192-bit two's complement, least significant word first */
#define TR_LIN_LOCAL_GROUPS 16              /* CUDA: work items keep every group locally up to this many groups */
#define TR_LIN_JACOBI_SWEEPS 64             /* the default sweep cap of the eigensolver */
#define TR_LIN_PRIME_BITS 61                /* every prime of the modular path lies in [2^61, 2^62) */
#define TR_LIN_PRIME_MARGIN 2               /* primes a batch adds beyond the Hadamard bound (unlucky primes) */
enum { TR_LIN_LISTWISE = 0, TR_LIN_PAIRWISE = 1 };                                  /* validity policies */
enum { TR_LIN_NEED_CPU = 0, TR_LIN_NEED_DEVICE = 1, TR_LIN_NEED_ADOPT = 2 };        /* tr_lin_moments_need targets */
enum { TR_LIN_SOURCE_FUSED = 1, TR_LIN_SOURCE_BANK = 2, TR_LIN_SOURCE_ADOPTED = 3, TR_LIN_SOURCE_COMBINED = 4 };
enum { TR_LIN_UNIQUE = 1, TR_LIN_FAMILY = 2, TR_LIN_NONE = 3 };                     /* solution sets */
enum { TR_LIN_PD = 1, TR_LIN_PSD = 2, TR_LIN_ND = 3, TR_LIN_NSD = 4, TR_LIN_INDEFINITE = 5, TR_LIN_ZERO = 6 };
enum { TR_LIN_RESULT_SOLVE = 1, TR_LIN_RESULT_INERTIA = 2, TR_LIN_RESULT_SYSTEM = 3, TR_LIN_RESULT_FIT = 4,
       TR_LIN_RESULT_COVARIANCE = 5, TR_LIN_RESULT_EIGEN = 6 };
enum { TR_LIN_METHOD_BAREISS = 1, TR_LIN_METHOD_MODULAR = 2, TR_LIN_METHOD_JACOBI = 3 };
/* parts of a result (exact: tr_lin_result_export; numerical: tr_lin_result_doubles) */
enum { TR_LIN_PART_X = 0,         /* solve / fit: the (particular) solution, n values */
       TR_LIN_PART_NULL = 1,      /* the canonical basis of the solution directions, nullity x n */
       TR_LIN_PART_CERT = 2,      /* NONE: y with y^T A = 0, y^T b = 1 (m values) */
       TR_LIN_PART_NEG = 3,       /* inertia: v with v^T A v < 0 (n values) */
       TR_LIN_PART_POS = 4,       /* inertia: v with v^T A v > 0 */
       TR_LIN_PART_KERNEL = 5,    /* inertia: an integer basis of the kernel, n0 x n */
       TR_LIN_PART_A = 6,         /* system / fit: the normal matrix (value units, ridge included), n x n */
       TR_LIN_PART_B = 7,         /* system / fit: the right-hand side, n values */
       TR_LIN_PART_STATS = 8,     /* fit: rss, tss, r2, sigma2, yty, n, ||X x||^2 (undefined where not defined) */
       TR_LIN_PART_COV = 9,       /* covariance: p x p (value units) */
       TR_LIN_PART_R2 = 10,       /* covariance: squared correlation, p x p */
       TR_LIN_PART_SIGN = 11,     /* covariance: sign of the correlation (-1, 0, 1), p x p */
       TR_LIN_PART_COUNT = 12,    /* covariance: rows of each entry, p x p */
       TR_LIN_PART_DESIGN = 13,   /* system / fit: the augmented column of each unknown */
       TR_LIN_PART_VALUES = 14,   /* eigen: eigenvalues, descending */
       TR_LIN_PART_VECTORS = 15,  /* eigen: eigenvectors, n x n, row i = vector i */
       TR_LIN_PART_RESIDUAL = 16, /* eigen: upper bounds of ||A v_i - lambda_i v_i||_2 */
       TR_LIN_PART_BOUND = 17,    /* eigen: eigenvalue error bounds */
       TR_LIN_PART_ORTHO = 18,    /* eigen: upper bound of max |V^T V - I| */
       TR_LIN_PART_WIDTH = 19,    /* eigen: max width of the input enclosure (0 for exact binary64 inputs) */
       TR_LIN_PARTS = 20 };
#define TR_LIN_UNDEFINED 2         /* tr_lin_qbuf sign of an undefined value (a variance of 0, a statistic of 0 dof) */
#define TR_LIN_STATS 7             /* values of TR_LIN_PART_STATS */

/* ------------------------------------------------------------------ structs crossing the ctypes boundary */
typedef struct {
    int32_t nfeatures;                      /* d: 0..TR_LIN_MAX_FEATURES */
    int32_t target;                         /* scope column of y, or -1 */
    int32_t feature[TR_LIN_MAX_FEATURES];   /* scope columns of x1..xd (distinct, not the target) */
    int32_t policy, intercept, width, reserved;   /* TR_LIN_LISTWISE / PAIRWISE; 0 / 1; 0, 128 or 192; 0 */
} tr_lin_spec;                              /* 88 bytes */

typedef struct { uint8_t a, b, kind, reserved; uint32_t need; } tr_lin_slot;   /* z_a z_b valid when need's columns
                                                                                   are present; kind 0 = Gram, 1..4 =
                                                                                   pair-complete marginals; 8 bytes */
typedef struct {
    uint64_t ngroups;
    int32_t nslots, words;                  /* words per sum (TR_LIN_WORDS) */
    uint64_t *rows;                         /* [ngroups] */
    uint64_t *count;                        /* [ngroups * nslots] */
    uint64_t *sum;                          /* [ngroups * nslots * words] */
} tr_lin_states;                            /* 40 bytes */

typedef struct {
    uint64_t ngroups, present, rows, bytes;
    uint64_t digest[2];                     /* the scope's descriptor digest (0 when unscoped) */
    int32_t k, nslots, ngram, nfeatures;
    int32_t target, policy, intercept, width;   /* width: 128 or 192, the accumulation used */
    int32_t marginals, source, scoped, reserved;
    int32_t column[TR_LIN_MAX_COLS];        /* scope column of each augmented column (-1: the constant) */
    int32_t scale[TR_LIN_MAX_COLS];         /* decimal scale of each augmented column */
} tr_lin_info;                              /* 240 bytes */

/* packed rationals: value i = sign[i] * |numerator| / denominator; numerator nlen[i] limbs then denominator dlen[i]
 * limbs (dlen 0: denominator 1), value after value, in limbs[0..nlimbs). As an output, nlimbs is the capacity on
 * entry and the limbs written (or needed: TR_MATH_INVALID when too small) on return; sign TR_LIN_UNDEFINED marks an
 * undefined value. */
typedef struct {
    uint64_t count, nlimbs;
    int8_t *sign;
    uint32_t *nlen, *dlen;
    uint64_t *limbs;
} tr_lin_qbuf;                              /* 48 bytes */

typedef struct {
    uint64_t group;
    int32_t intercept;                      /* -1: the state's declaration; 0 / 1 */
    int32_t masked;                         /* 0: every feature; 1: only those of `features` */
    uint32_t features, reserved;            /* bit i: feature i is in the design */
    int64_t ridge_num, ridge_den;           /* the penalty lambda = num / den >= 0 on the features (0 / 1: none) */
} tr_lin_fit_spec;                          /* 40 bytes */

typedef struct {
    int32_t what, status;                   /* TR_LIN_RESULT_*; solve/fit: UNIQUE/FAMILY/NONE; inertia: class */
    int32_t rows, cols, rank, nullity;
    int32_t positive, negative, zero, method;
    uint64_t pivots;                        /* bit j: unknown j is a pivot column */
    uint64_t primes, iterations, used;      /* primes (modular); sweeps (eigen); rows used (fit, covariance) */
    int32_t certified, converged;           /* exact verification done; eigen converged */
} tr_lin_result_info;                       /* 80 bytes */

/* the systems of a modular batch as the CUDA kernel reads them */
typedef struct {
    uint64_t nsys, words, entries;          /* systems; words per entry (two's complement); total entries */
    const uint64_t *data;                   /* entries * words */
    const uint64_t *offset;                 /* [nsys] first entry of each system (row-major m x c) */
    const int32_t *rows, *cols;             /* [nsys] m and c (c = unknowns + 1: the right-hand side) */
} tr_lin_batch_view;                        /* 56 bytes */

typedef struct tr_lin_moments_s tr_lin_moments;
typedef struct tr_lin_result_s tr_lin_result;
typedef struct tr_lin_batch_s tr_lin_batch;

/* ------------------------------------------------------------------ shared helpers (CPU code, CUDA host and kernels) */
/* The augmented columns: col[0] = -1, then the features, then the target. Returns k, or TR_MATH_INVALID (-1) /
 * TR_MATH_UNSUPPORTED (-12). ncols: the scope's columns. */
static inline int32_t tr_lin_columns(const tr_lin_spec *s, int32_t ncols, int32_t *col) {
    if (!s || !col || s->nfeatures < 0 || s->target < -1 || (s->policy != TR_LIN_LISTWISE && s->policy != TR_LIN_PAIRWISE) ||
        (s->intercept != 0 && s->intercept != 1) || (s->width != 0 && s->width != 128 && s->width != 192) ||
        s->reserved != 0 || ncols < 0 || ncols > TR_SCOPE_MAX_COLUMNS) return -1;
    if (s->nfeatures > TR_LIN_MAX_FEATURES) return -12;
    uint64_t seen = 0;
    int32_t k = 0;
    col[k++] = -1;
    for (int32_t i = 0; i < s->nfeatures; i++) {
        int32_t c = s->feature[i];
        if (c < 0 || c >= ncols || ((seen >> c) & 1)) return -1;
        seen |= (uint64_t)1 << c;
        col[k++] = c;
    }
    if (s->target >= 0) {
        if (s->target >= ncols || ((seen >> s->target) & 1)) return -1;
        col[k++] = s->target;
    }
    return k;
}
/* slot of Gram entry (a, b), a <= b < k; of the marginals of pair (a, b), 1 <= a < b < k (four consecutive slots:
 * sum z_a, sum z_b, sum z_a^2, sum z_b^2) */
static inline int32_t tr_lin_gram_slot(int32_t k, int32_t a, int32_t b) { return a * k - a * (a - 1) / 2 + (b - a); }
static inline int32_t tr_lin_pair_slot(int32_t k, int32_t a, int32_t b) {
    return k * (k + 1) / 2 + 4 * ((a - 1) * (k - 1) - a * (a - 1) / 2 + (b - a - 1));
}
static inline int32_t tr_lin_slot_count(int32_t k, int32_t policy, int32_t marginals) {
    return k * (k + 1) / 2 + (policy == TR_LIN_PAIRWISE && marginals && k > 2 ? 2 * (k - 1) * (k - 2) : 0);
}
static inline void tr_lin_set_slot(tr_lin_slot *s, int32_t a, int32_t b, int32_t kind, uint32_t need) {
    s->a = (uint8_t)a; s->b = (uint8_t)b; s->kind = (uint8_t)kind; s->reserved = 0; s->need = need;
}
/* the slot layout (need: bit i = augmented column i must be present; bit 0, the constant, never needed) */
static inline int32_t tr_lin_layout(int32_t k, int32_t policy, int32_t marginals, tr_lin_slot *slots) {
    uint32_t all = 0;
    int32_t n = 0;
    for (int32_t i = 1; i < k; i++) all |= (uint32_t)1 << i;
    for (int32_t a = 0; a < k; a++)
        for (int32_t b = a; b < k; b++)
            tr_lin_set_slot(&slots[n++], a, b, 0, policy == TR_LIN_LISTWISE ? all
                            : ((a ? (uint32_t)1 << a : 0) | (b ? (uint32_t)1 << b : 0)));
    if (policy == TR_LIN_PAIRWISE && marginals)
        for (int32_t a = 1; a < k; a++)
            for (int32_t b = a + 1; b < k; b++) {
                uint32_t need = ((uint32_t)1 << a) | ((uint32_t)1 << b);
                tr_lin_set_slot(&slots[n++], 0, a, 1, need);
                tr_lin_set_slot(&slots[n++], 0, b, 2, need);
                tr_lin_set_slot(&slots[n++], a, a, 3, need);
                tr_lin_set_slot(&slots[n++], b, b, 4, need);
            }
    return n;
}
/* M_c = max(|min|, |max|, 1) of augmented column i (1 for the constant) */
static inline uint64_t tr_lin_magnitude(const tr_scope_view *v, const int32_t *col, int32_t i) {
    if (i == 0) return 1;
    const tr_scope_view_column *c = &v->col[col[i]];
    uint64_t m = c->nvalid ? (tr_abs_u64(c->min) > tr_abs_u64(c->max) ? tr_abs_u64(c->min) : tr_abs_u64(c->max)) : 0;
    return m < 1 ? 1 : m;
}
/* the rows slot s can be valid on: the selected rows, at most the present values of each column it needs */
static inline uint64_t tr_lin_slot_rows(const tr_scope_view *v, const int32_t *col, int32_t k, const tr_lin_slot *s) {
    uint64_t rows = v->selected;
    for (int32_t i = 1; i < k; i++)
        if (((s->need >> i) & 1) && v->col[col[i]].nvalid < rows) rows = v->col[col[i]].nvalid;
    return rows;
}
/* 1 when some slot's sum is not proven to fit 128 bits (N * M_a * M_b > 2^127 - 1), else 0 */
static inline int32_t tr_lin_needs_wide(const tr_scope_view *v, const int32_t *col, int32_t k, const tr_lin_slot *slots,
                                        int32_t nslots) {
    const unsigned __int128 limit = ((unsigned __int128)1 << 127) - 1;
    for (int32_t s = 0; s < nslots; s++) {
        unsigned __int128 b = tr_bank_satmul(tr_lin_magnitude(v, col, slots[s].a), tr_lin_magnitude(v, col, slots[s].b), limit);
        if (tr_bank_satmul(b, tr_lin_slot_rows(v, col, k, &slots[s]), limit) > limit) return 1;
    }
    return 0;
}
/* the accumulation words (2 or 3) for a spec's width, or TR_MATH_OVERFLOW (-3) for width 128 without a proof */
static inline int32_t tr_lin_words(int32_t width, int32_t wide) {
    if (width == 128) return wide ? -3 : 2;
    if (width == 192) return 3;
    return wide ? 3 : 2;
}
static inline uint64_t tr_lin_state_bytes(uint64_t ngroups, int32_t nslots) {
    return tr_budget_add(tr_budget_mul(ngroups, 8), tr_budget_mul(tr_budget_mul(ngroups, (uint64_t)nslots), 8 + 8 * TR_LIN_WORDS));
}
/* the needs of tr_mgpu_lin_moments: bytes = its device copies (selection, group ids, the used columns and their
 * validity, row masks, slots, the column map, the states) */
static inline void tr_lin_device_need(const tr_scope_view *v, const int32_t *col, int32_t k, int32_t nslots, int32_t words,
                                      tr_math_need *need) {
    for (int i = 0; i < TR_BUDGET_RESOURCES; i++) need->need[i] = 0;
    uint64_t n = v->rows, G = v->ngroups;
    uint64_t bytes = tr_budget_add(tr_budget_bitmap_bytes(n), tr_budget_mul(n, 4));                  /* selection, masks */
    if (v->groups) bytes = tr_budget_add(bytes, tr_budget_mul(n, 4));
    for (int32_t i = 1; i < k; i++) {
        bytes = tr_budget_add(bytes, tr_budget_mul(n, v->col[col[i]].values ? 8 : 4));
        if (v->col[col[i]].valid) bytes = tr_budget_add(bytes, n);
    }
    bytes = tr_budget_add(bytes, tr_budget_add(tr_budget_mul((uint64_t)nslots, sizeof(tr_lin_slot)), 4 * TR_LIN_MAX_COLS));
    bytes = tr_budget_add(bytes, tr_budget_add(tr_budget_mul(G, 8),
                                               tr_budget_mul(tr_budget_mul(G, (uint64_t)nslots), 8 + 8 * (uint64_t)words)));
    need->need[TR_BUDGET_VARIABLES] = (uint64_t)(k - 1);
    need->need[TR_BUDGET_DEGREE] = 2;
    need->need[TR_BUDGET_SUPPORT] = (uint64_t)nslots;
    need->need[TR_BUDGET_TENSOR] = tr_budget_mul(G, (uint64_t)(k * (k + 1) / 2));
    need->need[TR_BUDGET_OUTPUT] = G;
    need->need[TR_BUDGET_STATES] = tr_budget_mul(G, (uint64_t)nslots);
    need->need[TR_BUDGET_BYTES] = bytes;
}

/* per row (CPU loop and kernels): the present columns as a mask (bit 0 = the row is selected) and z_a z_b */
TR_NUM_HD uint32_t tr_lin_row_mask(const tr_scope_view *v, const int32_t *col, int32_t k, uint64_t row) {
    if (!tr_scope_is_selected(v, row)) return 0;
    uint32_t m = 1;
    for (int32_t i = 1; i < k; i++) if (tr_scope_present(v, col[i], row)) m |= (uint32_t)1 << i;
    return m;
}
TR_NUM_HD __int128 tr_lin_product(const tr_scope_view *v, const int32_t *col, const tr_lin_slot *s, uint64_t row) {
    __int128 x = s->a ? (__int128)tr_scope_value(v, col[s->a], row) : 1;
    __int128 y = s->b ? (__int128)tr_scope_value(v, col[s->b], row) : 1;
    return x * y;
}
/* 192-bit two's complement helpers: w = 0; w += a (signed 128-bit) */
TR_NUM_HD void tr_lin_w3_add(uint64_t *w, __int128 x) { tr_wide_add128(w, x); }
TR_NUM_HD void tr_lin_w3_set(uint64_t *w, __int128 x) {
    w[0] = (uint64_t)x; w[1] = (uint64_t)((unsigned __int128)x >> 64); w[2] = x < 0 ? ~(uint64_t)0 : 0;
}

/* ------------------------------------------------------------------ modular elimination (CPU reference and kernel) */
/* x mod p for a words-word two's complement integer (least significant word first) */
TR_NUM_HD uint64_t tr_lin_reduce_words(const tr_gfp *f, const uint64_t *w, uint64_t words) {
    uint64_t r = 0;
    for (uint64_t i = words; i-- > 0;) r = tr_gfp_reduce_u128_hd(f, r, w[i]);
    if (words && (w[words - 1] >> 63)) {                       /* negative: minus 2^(64 words) mod p */
        uint64_t t = 1;
        for (uint64_t i = 0; i < words; i++) t = tr_gfp_mul_hd(f, t, f->r64);
        r = tr_gfp_sub_hd(f, r, t);
    }
    return r;
}
/* Gauss-Jordan elimination modulo p of the m x c matrix a (row-major residues, in place; m <= 64, c <= 64): rows
 * 0..r-1 become D times the reduced row echelon form and the rest zero, where D = det(M[I, P]) mod p for the pivot
 * columns P (increasing; a column's pivot is its first nonzero row) and the pivot rows I (increasing order).
 * hdr[0..3] = rank, pivot column mask, pivot row mask, D. */
TR_NUM_HD void tr_lin_modular_rref(const tr_gfp *f, uint64_t *a, int32_t m, int32_t c, uint64_t *hdr) {
    uint8_t rowid[TR_LIN_MAX_ROWS];
    for (int32_t i = 0; i < m; i++) rowid[i] = (uint8_t)i;
    int32_t r = 0;
    uint64_t D = 1, pc = 0, pr = 0;
    for (int32_t col = 0; col < c && r < m; col++) {
        int32_t i = r;
        while (i < m && a[(uint64_t)i * c + col] == 0) i++;
        if (i == m) continue;
        if (i != r) {
            for (int32_t j = 0; j < c; j++) {
                uint64_t t = a[(uint64_t)i * c + j];
                a[(uint64_t)i * c + j] = a[(uint64_t)r * c + j];
                a[(uint64_t)r * c + j] = t;
            }
            uint8_t t = rowid[i]; rowid[i] = rowid[r]; rowid[r] = t;
        }
        uint64_t *pivrow = a + (uint64_t)r * c;
        uint64_t piv = pivrow[col], inv = tr_gfp_inv_hd(f, piv);
        D = tr_gfp_mul_hd(f, D, piv);
        for (int32_t j = col; j < c; j++) pivrow[j] = tr_gfp_mul_hd(f, pivrow[j], inv);
        for (int32_t i2 = 0; i2 < m; i2++) {
            if (i2 == r) continue;
            uint64_t *row = a + (uint64_t)i2 * c, t = row[col];
            if (!t) continue;
            for (int32_t j = col; j < c; j++) row[j] = tr_gfp_sub_hd(f, row[j], tr_gfp_mul_hd(f, t, pivrow[j]));
        }
        pc |= (uint64_t)1 << col;
        r++;
    }
    int parity = 0;
    for (int32_t i = 0; i < r; i++) {
        pr |= (uint64_t)1 << rowid[i];
        for (int32_t j = i + 1; j < r; j++) if (rowid[i] > rowid[j]) parity ^= 1;
    }
    if (parity) D = tr_gfp_neg_hd(f, D);
    for (int32_t i = 0; i < r; i++)
        for (int32_t j = 0; j < c; j++) a[(uint64_t)i * c + j] = tr_gfp_mul_hd(f, a[(uint64_t)i * c + j], D);
    hdr[0] = (uint64_t)r; hdr[1] = pc; hdr[2] = pr; hdr[3] = D;
}
/* residue words of a batch for nprimes primes: per system nprimes blocks of 4 + m c words */
static inline uint64_t tr_lin_batch_words(const tr_lin_batch_view *v, uint64_t nprimes) {
    uint64_t total = 0;
    for (uint64_t s = 0; s < v->nsys; s++)
        total = tr_budget_add(total, tr_budget_mul(nprimes, tr_budget_add(4, (uint64_t)v->rows[s] * (uint64_t)v->cols[s])));
    return total;
}
/* bytes tr_mgpu_lin_eliminate copies to and allocates on the device */
static inline uint64_t tr_lin_batch_device_bytes(const tr_lin_batch_view *v, uint64_t nprimes) {
    uint64_t b = tr_budget_mul(tr_budget_mul(v->entries, v->words), 8);
    b = tr_budget_add(b, tr_budget_mul(v->nsys, 8 + 8 + 4 + 4));
    b = tr_budget_add(b, tr_budget_mul(nprimes, sizeof(tr_gfp)));
    return tr_budget_add(b, tr_budget_mul(tr_lin_batch_words(v, nprimes), 8));
}

/* ------------------------------------------------------------------ LIN-001 (libalg) */
tr_lin_moments *tr_lin_moments_build(const tr_scope *, const tr_lin_spec *, const tr_math_budget *budget,
                                     tr_math_refusal *refusal, int32_t *error);
/* the needs of tr_lin_moments_build (CPU), of tr_mgpu_lin_moments (DEVICE) or of adopting its states (ADOPT) */
int32_t tr_lin_moments_need(const tr_scope *, const tr_lin_spec *, int32_t target, tr_math_need *need);
/* From a bank built from `scope` (math_bank.h; spec columns are the scope's columns): the bank's basis must hold
 * every Gram monomial; mode TR_BANK_JOINT over exactly the spec's columns gives LISTWISE, TR_BANK_SQL gives PAIRWISE
 * (without marginals). Another mode, joint set or scope is TR_MATH_DOMAIN_ERROR, a missing monomial TR_MATH_INVALID. */
tr_lin_moments *tr_lin_moments_from_bank(const tr_scope *scope, const tr_bank *, const tr_lin_spec *,
                                         const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error);
/* Adopts states computed elsewhere (tr_mgpu_lin_moments' readback) after consistency checks (shape, total rows, counts,
 * zero sums without valid rows, the constant slot, the proven bound); origin: the scope they summarize. */
tr_lin_moments *tr_lin_moments_from_states(const tr_scope *origin, const tr_lin_spec *, int32_t marginals,
                                           const tr_lin_states *, const tr_math_budget *budget, tr_math_refusal *refusal,
                                           int32_t *error);
/* a + sign * b (sign 1 or -1) for states of one spec and grouping: merges, and deletions of rows (b summarizing a
 * subset of a's rows); TR_MATH_DOMAIN_ERROR for different specs, TR_MATH_INVALID for a deletion that is not a subset */
tr_lin_moments *tr_lin_moments_combine(const tr_lin_moments *a, const tr_lin_moments *b, int32_t sign,
                                       const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error);
int32_t tr_lin_moments_get_info(const tr_lin_moments *, tr_lin_info *);
int32_t tr_lin_moments_get_states(const tr_lin_moments *, tr_lin_states *);   /* pointers into the state (read only) */
int32_t tr_lin_moments_get_slots(const tr_lin_moments *, tr_lin_slot *out, int32_t capacity);
int32_t tr_lin_moments_equal(const tr_lin_moments *, const tr_lin_moments *);   /* 1: same spec and states */
void tr_lin_moments_free(tr_lin_moments *);

/* ------------------------------------------------------------------ LIN-002 (libalg) */
/* the exact solution set of A x = b: A m x n row-major, b m values (rational) */
tr_lin_result *tr_lin_solve(int32_t m, int32_t n, const tr_lin_qbuf *A, const tr_lin_qbuf *b, const tr_math_budget *budget,
                            tr_math_refusal *refusal, int32_t *error);
/* the exact inertia and class of a symmetric n x n rational matrix (a nonsymmetric one is TR_MATH_INVALID) */
tr_lin_result *tr_lin_inertia(int32_t n, const tr_lin_qbuf *A, const tr_math_budget *budget, tr_math_refusal *refusal,
                              int32_t *error);
/* the normal equations of a fit (parts A, B, DESIGN; info.used = rows of the group's complete rows (listwise)) */
tr_lin_result *tr_lin_system(const tr_lin_moments *, const tr_lin_fit_spec *, const tr_math_budget *budget,
                             tr_math_refusal *refusal, int32_t *error);
/* the exact fit: the solution set of the normal equations, verified, and (listwise) its statistics */
tr_lin_result *tr_lin_fit(const tr_lin_moments *, const tr_lin_fit_spec *, const tr_math_budget *budget,
                          tr_math_refusal *refusal, int32_t *error);
/* covariance (ddof 0 or 1), squared correlation and its sign, per pair of the non-constant augmented columns */
tr_lin_result *tr_lin_covariance(const tr_lin_moments *, uint64_t group, int32_t ddof, const tr_math_budget *budget,
                                 tr_math_refusal *refusal, int32_t *error);
/* Numerical symmetric eigensystem (APPROXIMATE, see the header comment) of the n x n enclosure lo <= A <= hi
 * (row-major, each symmetric); sweeps: the cap (0 = TR_LIN_JACOBI_SWEEPS, at most the budget's iterations). */
tr_lin_result *tr_lin_eigen(int32_t n, const double *lo, const double *hi, uint64_t sweeps, const tr_math_budget *budget,
                            tr_math_refusal *refusal, int32_t *error);
/* the same for an exact rational matrix (enclosed entry by entry) */
tr_lin_result *tr_lin_eigen_exact(int32_t n, const tr_lin_qbuf *A, uint64_t sweeps, const tr_math_budget *budget,
                                  tr_math_refusal *refusal, int32_t *error);
/* PCA of group g: the eigensystem of the covariance (kind 0, exact entries) or correlation (kind 1, enclosed entries;
 * a zero variance is TR_MATH_DOMAIN_ERROR) of the features (the target excluded) */
tr_lin_result *tr_lin_pca(const tr_lin_moments *, uint64_t group, int32_t kind, int32_t ddof, uint64_t sweeps,
                          const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error);
int32_t tr_lin_result_get_info(const tr_lin_result *, tr_lin_result_info *);
/* exact parts (see TR_LIN_PART_*): packed rationals; numerical parts: doubles (count written to *count) */
int32_t tr_lin_result_export(const tr_lin_result *, int32_t part, tr_lin_qbuf *out);
int32_t tr_lin_result_doubles(const tr_lin_result *, int32_t part, double *out, uint64_t capacity, uint64_t *count);
uint64_t tr_lin_result_count(const tr_lin_result *, int32_t part);    /* values of a part (0 when absent) */
uint64_t tr_lin_result_limbs(const tr_lin_result *, int32_t part);    /* limbs an export of an exact part needs */
void tr_lin_result_free(tr_lin_result *);

/* Batches of systems for the modular path: explicit systems (rows[s] x cols[s] augmented [A | b], packed one after
 * the other) or fits of one state. The view is what the kernel reads; need = the primes each system needs. */
tr_lin_batch *tr_lin_batch_systems(int32_t nsys, const int32_t *rows, const int32_t *unknowns, const tr_lin_qbuf *A,
                                   const tr_lin_qbuf *b, const tr_math_budget *budget, tr_math_refusal *refusal,
                                   int32_t *error);
tr_lin_batch *tr_lin_batch_fits(const tr_lin_moments *, int32_t nfits, const tr_lin_fit_spec *, const tr_math_budget *budget,
                                tr_math_refusal *refusal, int32_t *error);
int32_t tr_lin_batch_get_view(const tr_lin_batch *, tr_lin_batch_view *);
uint64_t tr_lin_batch_primes_needed(const tr_lin_batch *);           /* max over systems, with TR_LIN_PRIME_MARGIN */
/* bytes tr_mgpu_lin_eliminate copies to the device for nprimes primes (tr_lin_batch_device_bytes of the view) */
uint64_t tr_lin_batch_bytes(const tr_lin_batch *, uint64_t nprimes);
/* the first `count` primes of the modular path: the largest primes below 2^62, descending (deterministic) */
int32_t tr_lin_primes(uint64_t count, tr_gfp *out);
/* the CPU reference of tr_mgpu_lin_eliminate (the same inline elimination): residues[tr_lin_batch_words] */
int32_t tr_lin_batch_residues(const tr_lin_batch *, const tr_gfp *primes, uint64_t nprimes, uint64_t *residues,
                              const tr_math_budget *budget, tr_math_refusal *refusal);
/* Chinese remaindering, reconstruction and exact verification of every system from the residues; out[nsys] results
 * (solve or fit). TR_MATH_NOT_CONVERGED when some system had too few agreeing primes: *need is then the number of
 * primes a rerun needs (nothing is returned). */
int32_t tr_lin_batch_finish(const tr_lin_batch *, const tr_gfp *primes, uint64_t nprimes, const uint64_t *residues,
                            tr_lin_result **out, uint64_t *need, const tr_math_budget *budget, tr_math_refusal *refusal);
void tr_lin_batch_free(tr_lin_batch *);

/* exact cores on owned rational matrices for other capabilities (OPT-001): A row-major m x n, b m values (tr_bigq);
 * the results are those of tr_lin_solve / tr_lin_inertia; parts read with tr_lin_result_part (borrowed) */
tr_lin_result *tr_lin_q_solve(int32_t m, int32_t n, const tr_bigq *const *A, const tr_bigq *const *b,
                              const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error);
tr_lin_result *tr_lin_q_inertia(int32_t n, const tr_bigq *const *A, const tr_math_budget *budget, tr_math_refusal *refusal,
                                int32_t *error);
const tr_bigq *const *tr_lin_result_part(const tr_lin_result *, int32_t part, uint64_t *count);
/* packed rationals <-> owned tr_bigq arrays (NULL entries: undefined) */
int32_t tr_lin_q_import(const tr_lin_qbuf *in, uint64_t count, tr_bigq **out, const tr_math_budget *budget,
                        tr_math_refusal *refusal);
int32_t tr_lin_q_export(const tr_bigq *const *values, uint64_t count, tr_lin_qbuf *out);
uint64_t tr_lin_q_limbs(const tr_bigq *const *values, uint64_t count);
void tr_lin_q_free(tr_bigq **values, uint64_t count);
/* the least-squares system of a fit as owned rationals (value units): normal matrix with and without the ridge, the
 * right-hand side, y^T y, sum y, n (listwise), the design columns; for OPT-001's loss */
int32_t tr_lin_q_normal(const tr_lin_moments *, const tr_lin_fit_spec *, int32_t *dim, tr_bigq ***A, tr_bigq ***A0,
                        tr_bigq ***b, tr_bigq **yty, tr_bigq **sy, uint64_t *n, int32_t *design,
                        const tr_math_budget *budget, tr_math_refusal *refusal);

/* ------------------------------------------------------------------ the CUDA library (math_lin_gpu.cu)
 * tr_mgpu_lin_moments: view = tr_scope_get_view of the scope; out: host arrays, out->ngroups = view->ngroups,
 *   out->nslots = tr_lin_slot_count(k, policy, marginals), out->words = TR_LIN_WORDS; overwritten with exactly the
 *   states tr_lin_moments_build computes (then adopted by tr_lin_moments_from_states).
 * tr_mgpu_lin_eliminate: residues[tr_lin_batch_words(view, nprimes)] = per system and prime the header and the
 *   det-scaled reduced matrix of tr_lin_modular_rref, equal to tr_lin_batch_residues bit for bit.
 * Both budget their device copies (budget NULL = defaults) and return 0 or a native error code. */
int tr_mgpu_lin_moments(const tr_scope_view *view, const tr_lin_spec *spec, int32_t marginals, const tr_math_budget *budget,
                        tr_lin_states *out, tr_math_refusal *refusal);
int tr_mgpu_lin_eliminate(const tr_lin_batch_view *view, const tr_gfp *primes, uint64_t nprimes, const tr_math_budget *budget,
                          uint64_t *residues, tr_math_refusal *refusal);
#ifdef __cplusplus
}
#endif
#endif
