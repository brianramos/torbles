/* PAIR-002: structured pairwise query planning. A pairwise statistic S = sum over a declared pair set of
 * w_i w_j k(x_i, x_j) is answered exactly by the cheapest admissible of four strategies, each costed through the
 * FND-008 planner (math_planner.h), on the CPU and (support aggregation, direct pairs) in CUDA.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Data. A FND-004 scope (math_scope.h), ncols = 1..TR_PAIRPLAN_MAX_COLUMNS feature columns x (logical integers of the
 * scope) and an optional weight column w (absent: w = 1). Element i is a selected row whose weight is present and,
 * under the LISTWISE policy, whose feature values are all present; under the CATEGORY policy (the DELTA kernel only)
 * a NULL feature value is a category of its own (NULL matches NULL). Group g of the scope holds the elements of
 * group g. Rows with a NULL weight are no element under either policy.
 *
 * The pair set of a request (a, b, flags), declared exactly:
 *   within one group (a == b): ordered pairs (i, j), i != j, or with TR_PAIRPLAN_SELF every (i, j) including the
 *     self-pairs (i, i); without TR_PAIRPLAN_ORDERED each unordered pair {i, j} once (with SELF: {i, i} too),
 *     which needs a symmetric k (an unordered pair has no orientation: an asymmetric kernel is TR_MATH_INVALID);
 *   between two groups (a != b): the pairs (i in a, j in b), each cross pair once, k(x_i, x_j) oriented a first;
 *     with TR_PAIRPLAN_ORDERED both orientations (i, j) and (j, i), i.e. k(x_i, x_j) + k(x_j, x_i). No element is
 *     in two groups, so SELF adds nothing between groups.
 * TR_PAIRPLAN_MEAN divides S by the weighted pair count P (below): the mean over the pairs (empty when P = 0).
 * The weighted pair count P = sum over the pairs of w_i w_j: between W_a W_b (ordered: 2 W_a W_b); within, ordered
 * W^2 - Q (with self-pairs W^2), unordered (W^2 - Q) / 2 (with self-pairs (W^2 + Q) / 2), where W = sum w and
 * Q = sum w^2 over the group's elements (unweighted: n (n - 1), n^2, n (n - 1) / 2, n (n + 1) / 2).
 *
 * Kernels (tr_pairplan_spec.kind), exact integer values of the logical values (decimals of one common scale):
 *   ABS           sum_k |x_k - y_k| (the L1 distance; |x - y| for one column)
 *   MANN_WHITNEY  [x > y] + 1/2 [x = y] (one column; ties count half; asymmetric), kept as 2 k: S has den 2
 *   FEATURE       a KERNEL-001 kernel (math_kernel.h) over the feature columns: POLY (x.y + c)^d, MONOMIAL
 *                 sum_t w_t x^a_t y^a_t, DELTA prod_k [x_k = y_k] (CATEGORY policy)
 *   TABLE         a declared similarity: k(u, v) for a list of value-tuple pairs (int64), a default elsewhere;
 *                 symmetric (one orientation per pair, k(v, u) = k(u, v)) or not
 *   BAND          [max_k |x_k - y_k| <= h]: pairs within L-infinity radius h
 *   SQDIST        sum_k (x_k - y_k)^2
 *   RBF, LAPLACE, OTHER: no exact value (transcendental or undeclared): refused (TR_MATH_UNSUPPORTED), never
 *                 approximated.
 *
 * Strategies (every one returns the same exact S and P; tr_pairplan_make lists every candidate):
 *   SUPPORT   support aggregation: per group the distinct value tuples u (sorted; NULL last per coordinate) with
 *             their multiplicity m(u) (rows), W(u) = sum w and Q(u) = sum w^2; S = sum over entry pairs (u, v) of
 *             mult(u, v) k(u, v) with mult = W_a(u) W_b(v) off the diagonal and, on the diagonal u = v of one group,
 *             W(u)^2 - Q(u) ordered without self-pairs, W(u)^2 with them, (W(u)^2 -+ Q(u)) / 2 unordered (a value of
 *             multiplicity m: m (m - 1), m^2, m (m - 1) / 2, m (m + 1) / 2 pairs). Any kernel. CPU and CUDA.
 *   FEATURES  finite-feature factorization: the KERNEL-001 feature state of a FEATURE kernel (dense or sparse) and its
 *             SUM requests (K, K - Dg, (K -+ Dg) / 2: the same pair set conventions). CPU.
 *   PREFIX    sorted-prefix evaluation: the PAIR-001 summaries of the DIST-002 prefix state of the one feature
 *             column (count mass, or weight mass for nonnegative weights): ABS from ABS_WITHIN / ABS_CROSS (the
 *             diagonal contributes 0), MANN_WHITNEY from U with self-pairs (2U - Q without them; both orientations
 *             by two requests). Needs one column, ABS or MANN_WHITNEY, weights >= 0 and every class mass < 2^64. CPU.
 *   DIRECT    bounded direct pairs: the same evaluator over the uncompressed elements (one entry per row, W = w,
 *             Q = w^2), O(n_a n_b), within the declared pair budget. Any kernel. CPU and CUDA.
 *
 * Exactness. Results are exact rationals num / den in 1024-bit two's complement (den 1, or 2 for MANN_WHITNEY; MEAN:
 * den = that times P, sign normalized so den > 0; not reduced). A request is evaluated only after its magnitude is
 * proven: bits(sum_a |w|) + bits(sum_b |w|) + bits(max |k|) + 2 (+ 1 for both orientations) <= TR_PAIRPLAN_BITS_LIMIT
 * (max |k| bounded from the value extremes), so every product and partial sum fits and no result wraps; otherwise
 * the request is TR_MATH_OVERFLOW (a candidate whose requests cannot all be proven is rejected: refusal BOUND).
 * The FEATURES and PREFIX candidates carry KERNEL-001's and PAIR-001's own proofs.
 *
 * Planning (tr_pairplan_make). The batch's profile is exact: per group the elements n, W, A = sum |w| and Q (one
 * pass over the scope, or a retained table's totals). Distinct supports are exact with a retained state, else exact
 * upper bounds min(n, the cells of the value box). Each candidate has its validity (refusal kind UNSUPPORTED: no
 * structure for the kernel; DOMAIN: negative weights for weight masses; BOUND: an unproven magnitude), its needs
 * (variables, degree, support, tensor, iterations = kernel evaluations: the declared pair budget, output, states,
 * bytes; CUDA: the device copies) and exact amounts per FND-008 cost component under the model below; the FND-008
 * planner (tr_planner_choose) costs them with the family's rates (tr_pairplan_default_rates, or the caller's, passed
 * as its override table in row TR_PAIRPLAN_COST_ROW: relative work units, NOT time), checks the needs against the
 * budget and chooses the cheapest admissible candidate (the first on a tie). A force (strategy, device, repr) runs the
 * cheapest admissible match or refuses with the first match's code (UNSUPPORTED when nothing matches); it is never
 * replaced. With no admissible candidate the plan is refused (TR_MATH_BUDGET with the first over-budget candidate's
 * refusal when one was over budget, else the first rejection's code) and still returned with its evidence.
 *   Amounts: words / scan (selection words and row-values read by a state build; 0 when a retained state is reused),
 *   construct (elements built: records, sort comparisons n ceil(log2(n + 1)), entries, cells, emissions), reuse (1 per
 *   retained state read), chain (kernel operations: SUPPORT / DIRECT visits x (ops(k) + 3), times 8 for a request
 *   summed in 1024-bit words, PREFIX support entries read x 4, FEATURES feature items read x 2), materialize (result
 *   bytes), transfer (CUDA: host <-> device bytes), cpu_final / gpu_final (16 per request), fixed (native calls, state
 *   builds, launches). ops(k): ABS ncols, MANN_WHITNEY 1, BAND ncols, SQDIST 2 ncols, TABLE 1 + 2 ncols
 *   ceil(log2(entries + 1)), POLY ncols + 4 d, MONOMIAL sum_t (1 + 4 |a_t|), DELTA ncols.
 *   Two carriers, the same exact integers: a request proven within TR_PAIRPLAN_NARROW_BITS is summed in 128-bit words
 *   (its kernel values, multiplicities, terms and partial sums are below 2^124), the others in 1024-bit words.
 *   Evaluations (iterations): SUPPORT / DIRECT the entry pairs visited (within ordered d^2, unordered d (d + 1) / 2,
 *   between d_a d_b; d: entries, rows for DIRECT), PREFIX the support entries the PAIR-001 requests read (within d,
 *   between d_a + d_b, per request issued), FEATURES the feature items read (per KERNEL-001 request issued).
 *
 * Retained states (tr_pairplan_retain): a SUPPORT table, a ROWS table (DIRECT), a PREFIX state (the DIST-001 histogram
 * and DIST-002 prefix state PAIR-001 reads) or a FEATURES state (the KERNEL-001 feature state and its evaluation),
 * built by PAIR-002 itself over a scope, so its scope (by identity: the scope must outlive the state), columns,
 * weight, policy and kernel are known exactly; a plan reuses a matching one (construct 0, reuse 1, exact sizes).
 *
 * CUDA (math_pairplan_gpu.cu): a SUPPORT or ROWS table uploaded once (tr_mgpu_pairplan_upload), then per batch one work
 * item per (request, tile of TR_PAIRPLAN_TILE x TR_PAIRPLAN_TILE entry pairs) running the shared tr_pairplan_tile
 * (128- or 1024-bit sums, as on the CPU) into a local 1024-bit sum added to the request's accumulator with a 16-word
 * atomic carry chain (exact modulo 2^1024, hence exact under the proof), and one work item per request finishing it
 * with the shared tr_pairplan_finish: results equal to tr_pairplan_eval_table bit for bit, whatever the thread order.
 *
 * Budgets are checked before allocation (TR_MATH_BUDGET with the tr_math_refusal). Errors: TR_MATH_INVALID, _MEMORY,
 * _OVERFLOW, _BUDGET, _UNSUPPORTED, _DOMAIN_ERROR, _DEVICE. Objects are immutable; constructors return NULL and set
 * *error, leaving their inputs unchanged. */
#ifndef TORBLES_MATH_PAIRPLAN_H
#define TORBLES_MATH_PAIRPLAN_H
#include <stdint.h>
#include <string.h>
#include "math_num.h"
#include "math_budgets.h"
#include "math_scope.h"
#include "math_registry.h"
#include "math_dist.h"
#include "math_pair.h"
#include "math_kernel.h"
#ifdef __cplusplus
extern "C" {
#endif

#define TR_PAIRPLAN_ABI 1
#define TR_PAIRPLAN_MAX_COLUMNS 8
#define TR_PAIRPLAN_TILE 32               /* entry pairs per tile edge (a CUDA work item: up to 32 x 32 visits) */
#define TR_PAIRPLAN_BITS_LIMIT 1020       /* the magnitude proof's bit budget (1024-bit two's complement results) */
#define TR_PAIRPLAN_NARROW_BITS 126       /* a request proven within this many bits is summed in 128-bit words */
#define TR_PAIRPLAN_COST_ROW 1            /* the planner's override-table row the candidates are costed in */
#define TR_PAIRPLAN_MAX_CANDIDATES 8
#define TR_PAIRPLAN_RESULT_BYTES 400
enum { TR_PAIRPLAN_K_ABS = 1, TR_PAIRPLAN_K_MANN_WHITNEY = 2, TR_PAIRPLAN_K_FEATURE = 3, TR_PAIRPLAN_K_TABLE = 4,
       TR_PAIRPLAN_K_BAND = 5, TR_PAIRPLAN_K_SQDIST = 6,
       TR_PAIRPLAN_K_RBF = 16, TR_PAIRPLAN_K_LAPLACE = 17, TR_PAIRPLAN_K_OTHER = 18 };   /* the last three: refused */
enum { TR_PAIRPLAN_ORDERED = 1, TR_PAIRPLAN_SELF = 2, TR_PAIRPLAN_MEAN = 4 };
enum { TR_PAIRPLAN_S_SUPPORT = 1, TR_PAIRPLAN_S_FEATURES = 2, TR_PAIRPLAN_S_PREFIX = 3, TR_PAIRPLAN_S_DIRECT = 4,
       TR_PAIRPLAN_STRATEGIES = 4 };
enum { TR_PAIRPLAN_FORM_SUPPORT = 1, TR_PAIRPLAN_FORM_ROWS = 2 };
enum { TR_PAIRPLAN_LISTWISE = 1, TR_PAIRPLAN_CATEGORY = 2 };

/* ------------------------------------------------------------------ ABI structs */
/* the source columns and the kernel (2808 bytes) */
typedef struct {
    int32_t kind;                              /* TR_PAIRPLAN_K_* */
    int32_t ncols;                             /* 1..TR_PAIRPLAN_MAX_COLUMNS (MANN_WHITNEY: 1) */
    int32_t cols[TR_PAIRPLAN_MAX_COLUMNS];     /* scope columns of the value tuple x (distinct) */
    int32_t weight;                            /* scope column of w, or -1 */
    int32_t symmetric;                         /* TABLE: 1 = k(u, v) = k(v, u) declared (one orientation per pair) */
    int64_t h;                                 /* BAND: the radius, >= 0 */
    int64_t dflt;                              /* TABLE: k of a pair the table does not list */
    uint64_t ntable;                           /* TABLE: declared pairs */
    const int64_t *table;                      /* TABLE: ntable rows of 2 ncols + 1 int64: u, v, k(u, v) */
    tr_kernel_spec kernel;                     /* FEATURE: KERNEL-001's kernel; its columns are cols, in order */
} tr_pairplan_spec;

/* a request (24 bytes): groups a, b (a == b: within one group) */
typedef struct {
    uint64_t a, b;
    int32_t flags, reserved;                   /* TR_PAIRPLAN_ORDERED | _SELF | _MEAN; reserved 0 */
} tr_pairplan_request;

/* its answer (400 bytes) */
typedef struct {
    int32_t status, empty;                     /* 0 or the request's code; empty: MEAN over P = 0 (SQL NULL) */
    uint64_t evaluations;                      /* kernel evaluations (visits, entries read, items read) */
    tr_kernel_wide num, den;                   /* S (or the mean) = num / den, den > 0 */
    tr_kernel_wide pairs;                      /* P, the weighted pair count */
} tr_pairplan_result;

/* the elements of one group (80 bytes) */
typedef struct {
    uint64_t rows, entries;                    /* elements (rows), table entries (rows: = rows) */
    tr_math_i128 W, A;                         /* sum w, sum |w| */
    tr_dist_i256 Q;                            /* sum w^2 (nonnegative) */
} tr_pairplan_totals;

/* A SUPPORT or ROWS table (216 bytes): pointers into a retained state (tr_pairplan_state_table_view) or device copies.
 * Entries of group g: offset[g] .. offset[g + 1] - 1; SUPPORT entries are the distinct tuples in increasing order
 * (coordinate 0 most significant, NULL after every value), ROWS entries the elements in row order. */
typedef struct {
    int32_t form, policy, ncols, reserved;
    uint64_t ngroups, nentries;
    const uint64_t *offset;                    /* [G + 1] */
    const int64_t *value;                      /* [nentries x ncols] (0 where NULL) */
    const uint8_t *nulls;                      /* CATEGORY: [nentries] bit k: coordinate k NULL; LISTWISE: NULL */
    const tr_math_i128 *W;                     /* [nentries] sum w */
    const tr_dist_i256 *Q;                     /* SUPPORT: [nentries] sum w^2; ROWS: NULL (Q = W^2) */
    const uint64_t *count;                     /* SUPPORT: [nentries] rows; ROWS: NULL (1) */
    const tr_pairplan_totals *totals;          /* [G] */
    int64_t lo[TR_PAIRPLAN_MAX_COLUMNS], hi[TR_PAIRPLAN_MAX_COLUMNS];   /* present value extremes (lo > hi: none) */
} tr_pairplan_table_view;

/* A kernel ready for evaluation (1096 bytes): from a plan (tr_pairplan_plan_kernel_view; the table sorted and
 * canonical, owned by the plan) or its device copy. */
typedef struct {
    int32_t kind, ncols, policy, symmetric;    /* TR_PAIRPLAN_K_*; symmetric: k(u, v) = k(v, u) */
    int32_t fkind, degree, nterms, tbits;      /* FEATURE: TR_KERNEL_POLY / _MONOMIAL / _DELTA; tbits: TABLE max bits */
    int64_t c, h, dflt;
    uint64_t ntable;
    const int64_t *table;                      /* [ntable x (2 ncols + 1)], increasing (u, v); symmetric: u <= v */
    int64_t weight[TR_KERNEL_MAX_TERMS];       /* MONOMIAL */
    uint8_t exps[TR_KERNEL_MAX_TERMS][TR_PAIRPLAN_MAX_COLUMNS];
} tr_pairplan_kernel_view;

/* plan options (224 bytes) */
typedef struct {
    int32_t devices;                           /* TR_REG_ON_CPU | TR_REG_ON_CUDA (0: the CPU) */
    int32_t strategies;                        /* bit (1 << s): offer strategy s (0: all) */
    int32_t force_strategy, force_device;      /* 0: no force / any device */
    int32_t force_repr, custom_rates;          /* force_repr: FEATURES TR_TENSOR_DENSE / _SPARSE (0: any) */
    int32_t reserved[2];
    double rates[2][TR_COST_COMPONENTS];       /* custom_rates: [0] CPU, [1] CUDA per FND-008 component */
} tr_pairplan_options;

/* one candidate: FND-008's candidate (amounts, needs, costs, status, reason, refusals) and its sizes (472 bytes) */
typedef struct {
    tr_reg_candidate reg;
    int32_t strategy, device, repr, reuse;     /* reuse: the index of the retained state read, -1: built */
    int32_t exact, policy, form, bound_bits;   /* exact: sizes from retained states (else upper bounds) */
    uint64_t evaluations;                      /* kernel evaluations of the batch (the iterations need) */
    uint64_t native_requests;                  /* PAIR-001 / KERNEL-001 requests issued (else the batch's) */
    uint64_t entries;                          /* entries of the strategy's state (all groups) */
    uint64_t largest;                          /* entries of the largest group */
    uint64_t state_bytes;                      /* the state's bytes (built or reused) */
    uint64_t device_bytes;                     /* CUDA: the device copies (upload and batch) */
    uint64_t work_items;                       /* CUDA: tile work items */
    uint64_t reserved2;
} tr_pairplan_candidate;

/* a plan's summary (104 bytes) */
typedef struct {
    int32_t ncandidates, chosen, code, reason;  /* chosen -1 when refused */
    int32_t forced, devices, kind, policy;
    uint64_t requests, groups, elements, bytes;
    tr_math_refusal refusal;                   /* the plan-level budget refusal */
    int32_t profile_reused, nretained;         /* profile from a retained table; retained states given */
    uint64_t reserved;
} tr_pairplan_plan_info;

/* an execution (64 bytes) */
typedef struct {
    int32_t candidate, strategy, device, built;   /* built: a transient state was built (else retained) */
    uint64_t evaluations, native_requests, state_bytes, requests;
    uint64_t reserved[2];
} tr_pairplan_exec;

/* a retained state's description (136 bytes) */
typedef struct {
    int32_t strategy, form, policy, ncols;
    int32_t weight, repr, mass, kind;           /* mass: PREFIX TR_DIST_MASS_*; kind: the spec's TR_PAIRPLAN_K_* */
    int32_t cols[TR_PAIRPLAN_MAX_COLUMNS];
    uint64_t ngroups, entries, largest, elements, bytes;
    uint64_t reserved[4];
} tr_pairplan_state_info;

typedef struct tr_pairplan_state_s tr_pairplan_state;
typedef struct tr_pairplan_plan_s tr_pairplan_plan;

/* ------------------------------------------------------------------ wide helpers (CPU and kernels) */
TR_NUM_HD uint32_t tr_pairplan_bitlen(unsigned __int128 x) { uint32_t n = 0; while (x) { n++; x >>= 1; } return n; }
TR_NUM_HD void tr_pairplan_w_from_u128(unsigned __int128 v, tr_kernel_wide *r) {
    r->w[0] = (uint64_t)v; r->w[1] = (uint64_t)(v >> 64);
    for (int i = 2; i < TR_KERNEL_WORDS; i++) r->w[i] = 0;
}
TR_NUM_HD void tr_pairplan_w_from_i256(tr_dist_i256 a, tr_kernel_wide *r) {
    uint64_t ext = (a.w[3] >> 63) ? UINT64_MAX : 0;
    for (int i = 0; i < TR_KERNEL_WORDS; i++) r->w[i] = i < 4 ? a.w[i] : ext;
}
/* a / 2 of an even signed 256-bit value (an arithmetic shift) */
TR_NUM_HD tr_dist_i256 tr_pairplan_half(tr_dist_i256 a) {
    tr_dist_i256 r;
    uint64_t sign = (a.w[3] >> 63) ? UINT64_MAX : 0;
    for (int i = 0; i < 4; i++) { uint64_t hi = i < 3 ? a.w[i + 1] : sign; r.w[i] = (a.w[i] >> 1) | (hi << 63); }
    return r;
}
TR_NUM_HD uint64_t tr_pairplan_absdiff(int64_t a, int64_t b) {
    return a > b ? (uint64_t)a - (uint64_t)b : (uint64_t)b - (uint64_t)a;
}
/* The multiplicity of the diagonal entry pair (u, u) of one group: W^2 - Q ordered, W^2 with self-pairs, unordered
 * (W^2 - Q) / 2 and (W^2 + Q) / 2 (W^2 -+ Q = 2 sum_{i<j} w_i w_j (+ 2 Q) is even). Also the group's P from totals. */
TR_NUM_HD tr_dist_i256 tr_pairplan_diag(__int128 W, tr_dist_i256 Q, int ordered, int self) {
    tr_dist_i256 sq = tr_dist_w_mul128(W, W), r = sq;
    if (ordered) { if (!self) tr_dist_w_sub(sq, Q, &r); return r; }
    if (self) tr_dist_w_add(sq, Q, &r); else tr_dist_w_sub(sq, Q, &r);
    return tr_pairplan_half(r);
}

/* ------------------------------------------------------------------ kernel values (CPU and kernels) */
TR_NUM_HD int tr_pairplan_tuple_cmp(const int64_t *a, const int64_t *b, int32_t n) {
    for (int32_t j = 0; j < n; j++) if (a[j] != b[j]) return a[j] < b[j] ? -1 : 1;
    return 0;
}
/* TABLE: k(u, v) by binary search over the canonical rows (symmetric: (min, max) of the tuples) */
TR_NUM_HD int64_t tr_pairplan_table_lookup(const tr_pairplan_kernel_view *k, const int64_t *u, const int64_t *v) {
    const int32_t n = k->ncols;
    if (k->symmetric && tr_pairplan_tuple_cmp(v, u, n) < 0) { const int64_t *t = u; u = v; v = t; }
    uint64_t lo = 0, hi = k->ntable;
    const uint64_t width = 2 * (uint64_t)n + 1;
    while (lo < hi) {
        uint64_t mid = lo + (hi - lo) / 2;
        const int64_t *row = k->table + mid * width;
        int c = tr_pairplan_tuple_cmp(row, u, n);
        if (c == 0) c = tr_pairplan_tuple_cmp(row + n, v, n);
        if (c == 0) return row[2 * n];
        if (c < 0) lo = mid + 1; else hi = mid;
    }
    return k->dflt;
}
/* k(x, y) exactly (MANN_WHITNEY: 2 k); xn, yn: NULL masks (CATEGORY). 1, or 0 on overflow (never after the proof) */
TR_NUM_HD int tr_pairplan_kval(const tr_pairplan_kernel_view *k, const int64_t *x, uint8_t xn, const int64_t *y,
                               uint8_t yn, tr_kernel_wide *out) {
    const int32_t n = k->ncols;
    switch (k->kind) {
    case TR_PAIRPLAN_K_ABS: {
        unsigned __int128 s = 0;
        for (int32_t j = 0; j < n; j++) s += tr_pairplan_absdiff(x[j], y[j]);
        tr_pairplan_w_from_u128(s, out);
        return 1;
    }
    case TR_PAIRPLAN_K_MANN_WHITNEY:
        tr_kernel_w_from_i128(x[0] > y[0] ? 2 : x[0] == y[0] ? 1 : 0, out);
        return 1;
    case TR_PAIRPLAN_K_BAND: {
        int in = 1;
        for (int32_t j = 0; j < n; j++) if (tr_pairplan_absdiff(x[j], y[j]) > (uint64_t)k->h) in = 0;
        tr_kernel_w_from_i128(in, out);
        return 1;
    }
    case TR_PAIRPLAN_K_SQDIST: {
        tr_kernel_w_zero(out);
        for (int32_t j = 0; j < n; j++) {
            unsigned __int128 d = tr_pairplan_absdiff(x[j], y[j]);
            tr_kernel_wide t;
            tr_pairplan_w_from_u128(d * d, &t);                       /* < 2^128 */
            if (!tr_kernel_w_add(out, &t, out)) return 0;
        }
        return 1;
    }
    case TR_PAIRPLAN_K_TABLE:
        tr_kernel_w_from_i128(tr_pairplan_table_lookup(k, x, y), out);
        return 1;
    case TR_PAIRPLAN_K_FEATURE:
        if (k->fkind == TR_KERNEL_DELTA) {
            int eq = 1;
            for (int32_t j = 0; j < n; j++) {
                int nx = (xn >> j) & 1, ny = (yn >> j) & 1;
                if (nx != ny || (!nx && x[j] != y[j])) eq = 0;
            }
            tr_kernel_w_from_i128(eq, out);
            return 1;
        }
        if (k->fkind == TR_KERNEL_POLY) {
            tr_kernel_wide s, t;
            tr_kernel_w_from_i128(k->c, &s);
            for (int32_t j = 0; j < n; j++) {
                tr_kernel_w_from_i128((__int128)x[j] * y[j], &t);     /* |x y| < 2^126 */
                if (!tr_kernel_w_add(&s, &t, &s)) return 0;
            }
            tr_kernel_w_from_i128(1, out);
            for (int32_t e = 0; e < k->degree; e++) if (!tr_kernel_w_mul(out, &s, out)) return 0;
            return 1;
        }
        tr_kernel_w_zero(out);                                       /* MONOMIAL */
        for (int32_t t = 0; t < k->nterms; t++) {
            tr_kernel_wide m;
            tr_kernel_w_from_i128(k->weight[t], &m);
            for (int32_t j = 0; j < n; j++)
                for (int32_t e = 0; e < (int32_t)k->exps[t][j]; e++)
                    if (!tr_kernel_w_mul_i128(&m, (__int128)x[j] * y[j], &m)) return 0;
            if (!tr_kernel_w_add(out, &m, out)) return 0;
        }
        return 1;
    default:
        return 0;
    }
}
/* bits of max |k| over the box lo..hi of the values (an upper bound; MANN_WHITNEY: of 2 k) */
TR_NUM_HD uint32_t tr_pairplan_kernel_bits(const tr_pairplan_kernel_view *k, const int64_t *lo, const int64_t *hi) {
    const int32_t n = k->ncols;
    uint32_t span_bits = 0, mag_bits[TR_PAIRPLAN_MAX_COLUMNS], mag_max = 0;
    unsigned __int128 spans = 0;
    for (int32_t j = 0; j < n; j++) {
        uint64_t span = lo[j] <= hi[j] ? (uint64_t)hi[j] - (uint64_t)lo[j] : 0;
        uint64_t mag = lo[j] <= hi[j] ? (tr_abs_u64(lo[j]) > tr_abs_u64(hi[j]) ? tr_abs_u64(lo[j]) : tr_abs_u64(hi[j])) : 0;
        spans += span;
        uint32_t sb = tr_pairplan_bitlen(span);
        if (sb > span_bits) span_bits = sb;
        mag_bits[j] = tr_pairplan_bitlen(mag);
        if (mag_bits[j] > mag_max) mag_max = mag_bits[j];
    }
    switch (k->kind) {
    case TR_PAIRPLAN_K_ABS: return tr_pairplan_bitlen(spans) + 1;
    case TR_PAIRPLAN_K_MANN_WHITNEY: return 2;
    case TR_PAIRPLAN_K_BAND: return 1;
    case TR_PAIRPLAN_K_SQDIST: return 2 * span_bits + tr_pairplan_bitlen((unsigned __int128)n) + 1;
    case TR_PAIRPLAN_K_TABLE: return (uint32_t)k->tbits;
    case TR_PAIRPLAN_K_FEATURE:
        if (k->fkind == TR_KERNEL_DELTA) return 1;
        if (k->fkind == TR_KERNEL_POLY) {
            if (k->degree == 0) return 1;
            uint32_t s = tr_pairplan_bitlen((unsigned __int128)n) + 2 * mag_max, bc = tr_pairplan_bitlen(tr_abs_u64(k->c));
            return (uint32_t)k->degree * ((s > bc ? s : bc) + 1) + 1;
        } else {
            unsigned __int128 sw = 0;
            uint32_t top = 0;
            for (int32_t t = 0; t < k->nterms; t++) {
                sw += tr_abs_u64(k->weight[t]);
                uint32_t b = 0;
                for (int32_t j = 0; j < n; j++) b += 2u * (uint32_t)k->exps[t][j] * mag_bits[j];
                if (b > top) top = b;
            }
            return tr_pairplan_bitlen(sw) + top + 1;
        }
    default:
        return 4096;
    }
}

/* ------------------------------------------------------------------ requests over a table (CPU and kernels) */
TR_NUM_HD uint64_t tr_pairplan_ceil_tiles(uint64_t d) { return (d + TR_PAIRPLAN_TILE - 1) / TR_PAIRPLAN_TILE; }
/* entry pairs visited: within ordered d^2, unordered d (d + 1) / 2; between d_a d_b (saturating) */
TR_NUM_HD uint64_t tr_pairplan_visits(uint64_t da, uint64_t db, int within, int ordered) {
    unsigned __int128 v = within ? (ordered ? (unsigned __int128)da * da : (unsigned __int128)da * (da + 1) / 2)
                                 : (unsigned __int128)da * db;
    return v > UINT64_MAX ? UINT64_MAX : (uint64_t)v;
}
/* The magnitude proof of a request: bits(sum_a |w|) + bits(sum_b |w|) + bits(max |k|) + 2 (+ 1 for both
 * orientations). |S|, every pair term, multiplicity and partial sum is below 2^(bits - 1). */
TR_NUM_HD uint32_t tr_pairplan_request_bits(const tr_pairplan_table_view *t, const tr_pairplan_kernel_view *k,
                                            const tr_pairplan_request *q) {
    return tr_pairplan_bitlen((unsigned __int128)tr_i128_get(t->totals[q->a].A)) +
           tr_pairplan_bitlen((unsigned __int128)tr_i128_get(t->totals[q->b].A)) +
           tr_pairplan_kernel_bits(k, t->lo, t->hi) + 2 + (q->a != q->b && (q->flags & TR_PAIRPLAN_ORDERED) != 0);
}
/* 0, TR_MATH_INVALID (flags, groups, an unordered pair set with an asymmetric kernel) or TR_MATH_OVERFLOW (the
 * magnitude proof fails) */
TR_NUM_HD int32_t tr_pairplan_check(const tr_pairplan_table_view *t, const tr_pairplan_kernel_view *k,
                                    const tr_pairplan_request *q) {
    if (q->reserved || (q->flags & ~(TR_PAIRPLAN_ORDERED | TR_PAIRPLAN_SELF | TR_PAIRPLAN_MEAN)) ||
        q->a >= t->ngroups || q->b >= t->ngroups) return -1;
    const int within = q->a == q->b, ordered = (q->flags & TR_PAIRPLAN_ORDERED) != 0;
    if (within && !ordered && !k->symmetric) return -1;
    return tr_pairplan_request_bits(t, k, q) > TR_PAIRPLAN_BITS_LIMIT ? -3 : 0;
}
/* k(x, y) in 128 bits, exact for a request proven within TR_PAIRPLAN_NARROW_BITS (every value, product and partial
 * sum of the kernel is then below 2^124: the bounds of tr_pairplan_kernel_bits) */
TR_NUM_HD __int128 tr_pairplan_kval_narrow(const tr_pairplan_kernel_view *k, const int64_t *x, uint8_t xn,
                                           const int64_t *y, uint8_t yn) {
    const int32_t n = k->ncols;
    switch (k->kind) {
    case TR_PAIRPLAN_K_ABS: {
        unsigned __int128 s = 0;
        for (int32_t j = 0; j < n; j++) s += tr_pairplan_absdiff(x[j], y[j]);
        return (__int128)s;
    }
    case TR_PAIRPLAN_K_MANN_WHITNEY: return x[0] > y[0] ? 2 : x[0] == y[0] ? 1 : 0;
    case TR_PAIRPLAN_K_BAND:
        for (int32_t j = 0; j < n; j++) if (tr_pairplan_absdiff(x[j], y[j]) > (uint64_t)k->h) return 0;
        return 1;
    case TR_PAIRPLAN_K_SQDIST: {
        unsigned __int128 s = 0;
        for (int32_t j = 0; j < n; j++) { unsigned __int128 d = tr_pairplan_absdiff(x[j], y[j]); s += d * d; }
        return (__int128)s;
    }
    case TR_PAIRPLAN_K_TABLE: return tr_pairplan_table_lookup(k, x, y);
    case TR_PAIRPLAN_K_FEATURE:
        if (k->fkind == TR_KERNEL_DELTA) {
            for (int32_t j = 0; j < n; j++) {
                int nx = (xn >> j) & 1, ny = (yn >> j) & 1;
                if (nx != ny || (!nx && x[j] != y[j])) return 0;
            }
            return 1;
        }
        if (k->fkind == TR_KERNEL_POLY) {
            if (k->degree == 0) return 1;                                /* (x.y + c)^0: x.y is not bounded */
            __int128 s = k->c, p = 1;
            for (int32_t j = 0; j < n; j++) s += (__int128)x[j] * y[j];
            for (int32_t e = 0; e < k->degree; e++) p *= s;
            return p;
        } else {
            __int128 out = 0;
            for (int32_t t = 0; t < k->nterms; t++) {
                __int128 m = k->weight[t];
                for (int32_t j = 0; j < n; j++)
                    for (int32_t e = 0; e < (int32_t)k->exps[t][j]; e++) m *= (__int128)x[j] * y[j];
                out += m;
            }
            return out;
        }
    default:
        return 0;
    }
}
/* tr_pairplan_visit in 128 bits for a request proven within TR_PAIRPLAN_NARROW_BITS: the same exact term */
TR_NUM_HD __int128 tr_pairplan_visit_narrow(const tr_pairplan_table_view *t, const tr_pairplan_kernel_view *k,
                                            int within, int ordered, int self, uint64_t u, uint64_t v) {
    const __int128 Wu = tr_i128_get(t->W[u]);
    __int128 m;
    if (within && u == v) {
        const __int128 sq = Wu * Wu, Q = t->Q ? (__int128)(((unsigned __int128)t->Q[u].w[1] << 64) | t->Q[u].w[0]) : sq;
        m = ordered ? (self ? sq : sq - Q) : (self ? sq + Q : sq - Q) / 2;   /* W^2 -+ Q is even: exact */
    } else m = Wu * tr_i128_get(t->W[v]);
    if (!m) return 0;
    const int32_t n = t->ncols;
    const int64_t *xu = t->value + u * (uint64_t)n, *xv = t->value + v * (uint64_t)n;
    const uint8_t nu = t->nulls ? t->nulls[u] : 0, nv = t->nulls ? t->nulls[v] : 0;
    __int128 K = tr_pairplan_kval_narrow(k, xu, nu, xv, nv);
    if (!within && ordered) K += k->symmetric ? K : tr_pairplan_kval_narrow(k, xv, nv, xu, nu);
    return m * K;
}
TR_NUM_HD uint64_t tr_pairplan_group_entries(const tr_pairplan_table_view *t, uint64_t g) {
    return t->offset[g + 1] - t->offset[g];
}
/* the tiles of a valid request (0 for an empty group) */
TR_NUM_HD uint64_t tr_pairplan_request_tiles(const tr_pairplan_table_view *t, const tr_pairplan_request *q) {
    uint64_t ta = tr_pairplan_ceil_tiles(tr_pairplan_group_entries(t, q->a));
    if (q->a != q->b) return ta * tr_pairplan_ceil_tiles(tr_pairplan_group_entries(t, q->b));
    return (q->flags & TR_PAIRPLAN_ORDERED) ? ta * ta : ta * (ta + 1) / 2;
}
/* The contribution of the entry pair (u of group a, v of group b) to the request: mult(u, v) times k(u, v) (both
 * orientations: + k(v, u)), added to acc. 1, or 0 on overflow (never after the proof). */
TR_NUM_HD int tr_pairplan_visit(const tr_pairplan_table_view *t, const tr_pairplan_kernel_view *k, int within,
                                int ordered, int self, uint64_t u, uint64_t v, tr_kernel_wide *acc) {
    const __int128 Wu = tr_i128_get(t->W[u]);
    tr_dist_i256 m;
    if (within && u == v) m = tr_pairplan_diag(Wu, t->Q ? t->Q[u] : tr_dist_w_mul128(Wu, Wu), ordered, self);
    else m = tr_dist_w_mul128(Wu, tr_i128_get(t->W[v]));
    if (tr_dist_w_is_zero(m)) return 1;
    const int32_t n = t->ncols;
    const int64_t *xu = t->value + u * (uint64_t)n, *xv = t->value + v * (uint64_t)n;
    const uint8_t nu = t->nulls ? t->nulls[u] : 0, nv = t->nulls ? t->nulls[v] : 0;
    tr_kernel_wide K, K2, M, T;
    if (!tr_pairplan_kval(k, xu, nu, xv, nv, &K)) return 0;
    if (!within && ordered) {
        if (k->symmetric) { K2 = K; if (!tr_kernel_w_add(&K, &K2, &K)) return 0; }
        else if (!tr_pairplan_kval(k, xv, nv, xu, nu, &K2) || !tr_kernel_w_add(&K, &K2, &K)) return 0;
    }
    tr_pairplan_w_from_i256(m, &M);
    if (!tr_kernel_w_mul(&M, &K, &T)) return 0;
    return tr_kernel_w_add(acc, &T, acc);
}
/* Tile i of a valid request: the entry ranges [u0, u1) of group a and [v0, v1) of group b (within, unordered: the
 * tiles on and above the diagonal, row by row). */
TR_NUM_HD void tr_pairplan_tile_range(const tr_pairplan_table_view *t, const tr_pairplan_request *q, uint64_t i,
                                      uint64_t *u0, uint64_t *u1, uint64_t *v0, uint64_t *v1) {
    const uint64_t ba = t->offset[q->a], da = tr_pairplan_group_entries(t, q->a);
    const uint64_t bb = t->offset[q->b], db = tr_pairplan_group_entries(t, q->b);
    uint64_t tu, tv;
    if (q->a == q->b && !(q->flags & TR_PAIRPLAN_ORDERED)) {
        const uint64_t T = tr_pairplan_ceil_tiles(da);
        uint64_t lo = 0, hi = T - 1;                                    /* the last row tu with start(tu) <= i */
        while (lo < hi) {
            uint64_t mid = lo + (hi - lo + 1) / 2, start = mid * T - mid * (mid - 1) / 2;
            if (start <= i) lo = mid; else hi = mid - 1;
        }
        tu = lo; tv = tu + (i - (tu * T - tu * (tu - 1) / 2));
    } else {
        const uint64_t Tv = tr_pairplan_ceil_tiles(db);
        tu = i / Tv; tv = i % Tv;
    }
    *u0 = ba + tu * TR_PAIRPLAN_TILE; *u1 = *u0 + TR_PAIRPLAN_TILE < ba + da ? *u0 + TR_PAIRPLAN_TILE : ba + da;
    *v0 = bb + tv * TR_PAIRPLAN_TILE; *v1 = *v0 + TR_PAIRPLAN_TILE < bb + db ? *v0 + TR_PAIRPLAN_TILE : bb + db;
}
/* The sum over tile i of a valid request into *local (zeroed first) and its visits: 1, or 0 on overflow. A request
 * proven within TR_PAIRPLAN_NARROW_BITS is summed in 128-bit words (the same exact integers), others in 1024 bits. */
TR_NUM_HD int tr_pairplan_tile(const tr_pairplan_table_view *t, const tr_pairplan_kernel_view *k,
                               const tr_pairplan_request *q, uint64_t i, tr_kernel_wide *local, uint64_t *visits) {
    tr_kernel_w_zero(local);
    uint64_t u0, u1, v0, v1, n = 0;
    tr_pairplan_tile_range(t, q, i, &u0, &u1, &v0, &v1);
    const int within = q->a == q->b, ordered = (q->flags & TR_PAIRPLAN_ORDERED) != 0;
    const int self = (q->flags & TR_PAIRPLAN_SELF) != 0, upper = within && !ordered;
    if (tr_pairplan_request_bits(t, k, q) <= TR_PAIRPLAN_NARROW_BITS) {
        __int128 s = 0;
        for (uint64_t u = u0; u < u1; u++)
            for (uint64_t v = upper && v0 < u ? u : v0; v < v1; v++) {
                n++;
                s += tr_pairplan_visit_narrow(t, k, within, ordered, self, u, v);
            }
        tr_kernel_w_from_i128(s, local);
        *visits = n;
        return 1;
    }
    int ok = 1;
    for (uint64_t u = u0; u < u1 && ok; u++)
        for (uint64_t v = upper && v0 < u ? u : v0; v < v1; v++) {   /* unordered: the pairs {u, v}, v >= u */
            n++;
            if (!tr_pairplan_visit(t, k, within, ordered, self, u, v, local)) { ok = 0; break; }
        }
    *visits = n;
    return ok;
}
/* P of a request from the groups' totals */
TR_NUM_HD void tr_pairplan_pair_count(const tr_pairplan_totals *A, const tr_pairplan_totals *B, int within, int ordered,
                                      int self, tr_kernel_wide *out) {
    tr_dist_i256 p;
    if (within) p = tr_pairplan_diag(tr_i128_get(A->W), A->Q, ordered, self);
    else {
        p = tr_dist_w_mul128(tr_i128_get(A->W), tr_i128_get(B->W));
        if (ordered) tr_dist_w_add(p, p, &p);                           /* |2 W_a W_b| < 2^191 */
    }
    tr_pairplan_w_from_i256(p, out);
}
/* The answer of a request from its status (tr_pairplan_check), the sum of its pair terms and its visits */
TR_NUM_HD void tr_pairplan_finish(const tr_pairplan_table_view *t, const tr_pairplan_kernel_view *k,
                                  const tr_pairplan_request *q, int32_t status, const tr_kernel_wide *sum, int overflow,
                                  uint64_t visits, tr_pairplan_result *r) {
    r->status = 0; r->empty = 0; r->evaluations = 0;
    tr_kernel_w_zero(&r->num); tr_kernel_w_from_i128(1, &r->den); tr_kernel_w_zero(&r->pairs);
    if (status || overflow) { r->status = status ? status : -3; return; }
    const int within = q->a == q->b, ordered = (q->flags & TR_PAIRPLAN_ORDERED) != 0;
    tr_pairplan_pair_count(&t->totals[q->a], &t->totals[q->b], within, ordered, (q->flags & TR_PAIRPLAN_SELF) != 0,
                           &r->pairs);
    r->num = *sum; r->evaluations = visits;
    tr_kernel_w_from_i128(k->kind == TR_PAIRPLAN_K_MANN_WHITNEY ? 2 : 1, &r->den);
    if (q->flags & TR_PAIRPLAN_MEAN) {
        if (tr_kernel_w_is_zero(&r->pairs)) {
            r->empty = 1; tr_kernel_w_zero(&r->num); tr_kernel_w_from_i128(1, &r->den); return;
        }
        tr_kernel_wide d = r->den;
        tr_kernel_w_mul(&d, &r->pairs, &r->den);                        /* within the proof */
        if (tr_kernel_w_neg(&r->den)) { tr_kernel_w_negate(&r->den); tr_kernel_w_negate(&r->num); }
    }
}

/* ------------------------------------------------------------------ host sizes (libalg and the CUDA host code) */
static inline uint64_t tr_pairplan_entry_bytes(int32_t form, int32_t policy, int32_t ncols) {
    uint64_t b = 8 * (uint64_t)ncols + 16 + (policy == TR_PAIRPLAN_CATEGORY ? 1 : 0);
    return form == TR_PAIRPLAN_FORM_SUPPORT ? b + 40 : b;                   /* Q (32) and count (8) */
}
/* bytes of a table's arrays (and of its device copy) */
static inline uint64_t tr_pairplan_table_bytes(int32_t form, int32_t policy, int32_t ncols, uint64_t G, uint64_t entries) {
    uint64_t b = tr_budget_add(tr_budget_mul(tr_budget_add(G, 1), 8), tr_budget_mul(G, sizeof(tr_pairplan_totals)));
    return tr_budget_add(b, tr_budget_mul(entries, tr_pairplan_entry_bytes(form, policy, ncols)));
}
/* an upload's bytes: the table's device copy and the host copies of its offsets and totals (the batch's checks) */
static inline uint64_t tr_pairplan_upload_bytes(int32_t form, int32_t policy, int32_t ncols, uint64_t G, uint64_t entries) {
    uint64_t host = tr_budget_add(tr_budget_mul(tr_budget_add(G, 1), 8), tr_budget_mul(G, sizeof(tr_pairplan_totals)));
    return tr_budget_add(tr_pairplan_table_bytes(form, policy, ncols, G, entries), host);
}
/* the device copies of one batch: requests, statuses, tile offsets, accumulators (16 words, visits, flag), results,
 * the kernel view and its table */
static inline uint64_t tr_pairplan_batch_device_bytes(uint64_t n, int32_t ncols, uint64_t ntable) {
    uint64_t per = sizeof(tr_pairplan_request) + 4 + 8 + 8 * TR_KERNEL_WORDS + 16 + sizeof(tr_pairplan_result);
    uint64_t b = tr_budget_add(tr_budget_mul(n, per), 8 + sizeof(tr_pairplan_kernel_view));
    return tr_budget_add(b, tr_budget_mul(ntable, tr_budget_mul(2 * (uint64_t)ncols + 1, 8)));
}

/* ------------------------------------------------------------------ libalg (math_pairplan.c) */
/* the rates (relative work units per amount, NOT time) of the candidates: [0] CPU, [1] CUDA */
void tr_pairplan_default_rates(double rates[2][TR_COST_COMPONENTS]);
/* A retained state of a strategy over a scope (which must outlive it): SUPPORT (the support table), DIRECT (the ROWS
 * table), PREFIX (the DIST-001 histogram and DIST-002 prefix state of cols[0]: count mass, weight mass when weighted),
 * FEATURES (the KERNEL-001 feature state of spec->kernel and its evaluation; repr TR_TENSOR_DENSE / _SPARSE). The
 * spec's kind sets the policy (DELTA: CATEGORY). Budgets: the table (states = elements, bytes: the records and the
 * table, the final table checked once its entries are counted), or the components' own. */
tr_pairplan_state *tr_pairplan_retain(const tr_scope *scope, const tr_pairplan_spec *spec, int32_t strategy,
                                      int32_t repr, const tr_math_budget *budget, tr_math_refusal *refusal,
                                      int32_t *error);
int32_t tr_pairplan_state_get_info(const tr_pairplan_state *, tr_pairplan_state_info *);
/* SUPPORT / ROWS states: the table (pointers into the state) */
int32_t tr_pairplan_state_table_view(const tr_pairplan_state *, tr_pairplan_table_view *);
void tr_pairplan_state_free(tr_pairplan_state *);
/* A plan of n requests (see the header comment): 0 or the plan's code. *out is the plan (with its evidence, also when
 * it is refused), or NULL when the spec, the requests or the options are invalid (the code says why: INVALID,
 * UNSUPPORTED for a kernel without an exact value or a limit, DOMAIN_ERROR for columns of different scales) or the
 * plan object is over budget (TR_MATH_BUDGET with *refusal). retained: nretained states (NULL entries skipped); a
 * matching one is reused. options NULL: the CPU, every strategy, no force, the default rates. */
int32_t tr_pairplan_make(const tr_scope *scope, const tr_pairplan_spec *spec, const tr_pairplan_request *requests,
                         uint64_t n, const tr_pairplan_state *const *retained, int32_t nretained,
                         const tr_pairplan_options *options, const tr_math_budget *budget, tr_pairplan_plan **out,
                         tr_math_refusal *refusal);
int32_t tr_pairplan_plan_get_info(const tr_pairplan_plan *, tr_pairplan_plan_info *);
int32_t tr_pairplan_plan_get(const tr_pairplan_plan *, int32_t index, tr_pairplan_candidate *);
/* the evaluation view of the plan's kernel (the table owned by the plan) */
int32_t tr_pairplan_plan_kernel_view(const tr_pairplan_plan *, tr_pairplan_kernel_view *);
/* The requests of the plan in order (n of them, as given) */
int32_t tr_pairplan_plan_requests(const tr_pairplan_plan *, const tr_pairplan_request **requests, uint64_t *n);
void tr_pairplan_plan_free(tr_pairplan_plan *);
/* Runs a CPU candidate (index -1: the chosen one) on every request: out[n]. A rejected candidate returns its code; a
 * CUDA candidate is TR_MATH_INVALID here (it runs through tr_mgpu_pairplan_query). Retained states are read, others
 * built for the call (their budgets checked) and freed. */
int32_t tr_pairplan_execute(const tr_pairplan_plan *, int32_t index, const tr_math_budget *budget,
                            tr_pairplan_result *out, tr_pairplan_exec *exec, tr_math_refusal *refusal);
/* The CPU evaluation of n requests over a SUPPORT or ROWS table (what the CPU candidates run; the CUDA twin is
 * tr_mgpu_pairplan_query). Budget: output = n, iterations = the visits of the valid requests. The table's policy
 * must be the kernel's (else TR_MATH_INVALID). */
int32_t tr_pairplan_eval_table(const tr_pairplan_table_view *, const tr_pairplan_kernel_view *,
                               const tr_pairplan_request *requests, uint64_t n, tr_pairplan_result *out,
                               const tr_math_budget *budget, tr_math_refusal *refusal);

/* ------------------------------------------------------------------ the CUDA library (math_pairplan_gpu.cu) */
typedef struct tr_mgpu_pairplan_s tr_mgpu_pairplan;
/* the device copy of a table (budget: bytes = tr_pairplan_upload_bytes) */
tr_mgpu_pairplan *tr_mgpu_pairplan_upload(const tr_pairplan_table_view *view, const tr_math_budget *budget,
                                          tr_math_refusal *refusal, int *error);
/* n requests against the device table: exactly tr_pairplan_eval_table's results. Budget: output = n, iterations =
 * the visits, bytes = tr_pairplan_batch_device_bytes. 0 or a native code (TR_MATH_DEVICE when the device fails). */
int tr_mgpu_pairplan_query(const tr_mgpu_pairplan *, const tr_pairplan_kernel_view *kernel,
                           const tr_pairplan_request *requests, uint64_t n, tr_pairplan_result *out,
                           const tr_math_budget *budget, tr_math_refusal *refusal);
uint64_t tr_mgpu_pairplan_bytes(const tr_mgpu_pairplan *);
void tr_mgpu_pairplan_free(tr_mgpu_pairplan *);
#ifdef __cplusplus
}
#endif
#endif
