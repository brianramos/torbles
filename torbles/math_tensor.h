/* TENSOR-001 / TENSOR-002: exact multi-index states (dense, sparse and factored moment and count tensors, per group)
 * and costed contraction planning of bounded tensor networks, on the CPU and in CUDA.
 * SPDX-License-Identifier: Apache-2.0
 *
 * ================================================================ TENSOR-001: the tensor state
 * A tensor T of order m (0..TR_TENSOR_MAX_ORDER) has dims d_0..d_{m-1} (0 <= d_k <= 2^32 - 1) and G >= 1 groups; an
 * entry is T_g[i_0, ..., i_{m-1}] with 0 <= i_k < d_k, an exact signed integer (|T| <= 2^127 - 1, two words). The
 * cells of a group are prod_k d_k (1 for m = 0, a scalar), which must be below 2^64 (else TR_MATH_UNSUPPORTED). Order
 * is row major: axis 0 most significant (the linear index of a multi-index is sum_k i_k stride_k, stride_{m-1} = 1).
 *
 * Kinds (what an entry means; tr_tensor_info.kind):
 *   MOMENT  built from a FND-004 scope (math_scope.h): axis k lists ncols_k <= TR_TENSOR_MAX_FEATURES scope columns
 *           (repeats allowed; -1 is the constant 1), d_k = ncols_k, and
 *               T_g[i_0..i_{m-1}] = sum over the rows r of group g that contribute of  w_r prod_k x_{r, col_k[i_k]}
 *           (w_r = the weight column's logical value, 1 without weight). Policies (declared, never mixed):
 *             LISTWISE  a row contributes iff every referenced column and the weight are present (complete rows);
 *             SQL       a row contributes to an entry iff the entry's factors and the weight are present; the tensor
 *                       then keeps per entry the count of rows that contributed (has_counts: the "validity"
 *                       component, the same moment tensor of the validity indicators), as SQL COUNT(x*y).
 *           Example: one axis list [x1..xn] on all m axes is the m-th moment tensor sum_r x_r^{(x)m}; [-1, x1..xn]
 *           gives the augmented tensor whose slices at leading index 0 are every lower moment (KERNEL-001).
 *   COUNT   joint counts over m categorical axes: axis k reads one scope column with a domain of span_k values
 *           lo_k .. lo_k + span_k - 1 and, when nullslot_k, a NULL slot at index span_k (d_k = span_k + nullslot_k);
 *           T_g[i] = the rows of group g whose coordinates are i (NULL is a category of its own, never merged into a
 *           value), or with a weight the exact sum of their present weights (rows with a NULL weight count nowhere).
 *           Scope domain: [min, max] of the selected present values with a NULL slot iff a selected value is NULL;
 *           an explicit domain must hold every selected value and NULL (else TR_MATH_DOMAIN_ERROR before reading).
 *   GENERIC from arrays, or derived (a contraction, a fixed slice, an outer product, a marginal of moments): plain
 *           index axes, no scope.
 * Per group (scoped tensors): rows[g] = the selected rows of the group (bag multiplicity; group g is present iff
 * rows[g] > 0, an all-NULL group is present with zero entries, an absent group has rows 0 and no entry) and used[g] =
 * the rows that contributed (LISTWISE: the complete rows; SQL and COUNT: the rows whose weight is present). Unscoped
 * tensors have rows = used = 0. NULL validity (counts, NULL slots), bag multiplicity (rows) and presence are separate.
 *
 * Representations (tr_tensor_info.repr), exact images of each other:
 *   DENSE     every entry of every group: value[g * cells + linear index] (structural zeros included), and
 *             count[...] when has_counts.
 *   SPARSE    per group the stored entries in strictly increasing multi-index order (CSR): offset[g] ..
 *             offset[g + 1] - 1, each with its coordinates coord[e * m + k], value[e] (and count[e]). An entry is
 *             stored iff its value is nonzero (has_counts: iff its count is nonzero; its value may then be 0).
 *   FACTORED  a declared sum of rank-one terms per group: T_g = sum_{t of g} weight_t (x)_k f_{t,k}, f_{t,k} in Z^{d_k}
 *             (int64): term t of group g in toffset[g] .. toffset[g + 1] - 1, factor[t * fdim + foff_k + i]
 *             (fdim = sum_k d_k, foff_k = sum_{j<k} d_j). Built for MOMENT under LISTWISE (one term per contributing
 *             row: its weight and its values along each axis), from declared factors (tr_tensor_from_factors), by an
 *             outer product, or converted (one unit term per stored entry). A factored tensor carries a proven bound
 *             on every partial sum of its expansion at any entry: max_g sum_t |weight_t| prod_k max_i |f_{t,k}[i]|
 *             <= 2^127 - 1 (else TR_MATH_OVERFLOW at construction; a conversion's unit terms, one per entry, carry the
 *             source's maximum), so every expansion is exact. Per-entry counts have no factored form (refused).
 * Conversions (tr_tensor_convert) preserve the order, dims, axis metadata, groups, rows, used, counts and every
 * entry exactly; a dense image is built only when asked (and budgeted): nothing expands a factored or sparse tensor
 * into its Cartesian domain implicitly.
 *
 * Exactness. Entries are exact signed 128-bit integers. A build proves before reading the rows that
 *   N x max|w| x prod_k max_{c in axis k} max(|min_c|, |max_c|, 1) <= 2^127 - 1
 * (N the rows that can contribute, from the scope's extremes; COUNT without weight always fits), else TR_MATH_OVERFLOW;
 * so every product and partial sum fits, and chunked, device and CPU builds agree bit for bit. Derived tensors check
 * their sums (marginal, outer product, factored arithmetic) and refuse TR_MATH_OVERFLOW instead of wrapping. Each
 * tensor records bound = the exact max |entry| (dense, sparse) or the factored bound above, which contraction proofs
 * read.
 *
 * Operations (all return a new owned tensor; inputs unchanged): convert; marginal (sum out the axes not kept: values
 * and counts alike, NULL slots included; a COUNT marginal is the count tensor of the kept columns, a MOMENT marginal
 * is GENERIC); slice (per axis a fixed index, removing the axis, or a range [lo, hi), keeping it with d = hi - lo and
 * adjusted metadata; any fixed axis makes the result GENERIC); permute (axis k of the result is axis perm[k]); outer
 * product (order m_A + m_B, groups paired, weights multiplied); ungroup (the G groups become a leading GROUP axis of
 * a one-group tensor); values (the same entries without counts, GENERIC). Entry lookup (tr_tensor_get) works in
 * every form (binary search in sparse, a term sum in factored).
 *
 * Construction (CPU): dense counts in place (MOMENT tensors whose axes share one column list are accumulated on the
 * non-decreasing multi-indices only and completed by symmetry: identical results, fewer products); sparse through a
 * dense scratch (TR_TENSOR_STRATEGY_COUNT, chosen by AUTO when G x cells <= 4 x emissions + 65536) or by sorting one
 * entry per emitted (row, entry) pair (SORT: any domain); factored by one pass collecting the terms. An emission is a
 * (row, multi-index) whose product is nonzero (LISTWISE) or whose factors are present (SQL); a COUNT row emits one
 * cell. CUDA (math_tensor_gpu.cu, tr_mgpu_tensor_build): dense work items (entry, row slice) accumulate locally and
 * add with exact two-word atomics (COUNT: one atomic per row); sparse ones insert every emission into an
 * open-addressing hash table on the device (an atomic compare-and-swap claims a slot, exact atomics add) whose
 * occupied slots the host sorts into the canonical CSR: tr_tensor_from_states adopts the readback after checks.
 *
 * ================================================================ TENSOR-002: contraction planning
 * A network is 1..TR_TENSOR_NET_MAX tensors, each with one label (0..TR_TENSOR_MAX_LABELS - 1) per axis (a grouped
 * tensor, G > 1, has one more label first: its group axis), and an output label list. Its value is the Einstein sum
 *   out[o] = sum over every assignment of the labels not in the output of  prod_t T_t[labels of t]
 * (a label may join any number of tensors; a label appearing in one tensor only and not in the output is summed out).
 * Every label has one dimension across the tensors; labels are distinct within one tensor (a trace is
 * TR_MATH_UNSUPPORTED) and in the output. Tensors with per-entry counts are refused (TR_MATH_DOMAIN_ERROR: counts do
 * not contract; tr_tensor_values drops them explicitly).
 *
 * Orders. A contraction order is a sequence of n - 1 pairwise steps (slots: inputs 0..n-1, the result of step s is
 * slot n + s); step s contracts two live slots A, B: result labels = the labels of A u B still needed by another live
 * slot or by the output (the last step: exactly the output, in its order; others sorted by label id), summed labels
 * = the rest of A u B. A one-tensor network has one reduction step (B = -1). tr_tensor_net_plan enumerates every
 * order when there are at most max_orders (n <= 4: 1, 1, 3, 18 orders), otherwise a bounded search: the two greedy
 * orders (smallest result first, fewest products first) and then depth-first orders in lexicographic step order, up
 * to max_orders (<= the iterations budget); info.complete says which.
 *
 * Candidates. Each order runs with a kernel on a device: CPU dense (every result entry by the shared evaluator
 * tr_tensor_dense_entry), CPU sparse (operands as sorted coordinates: entries matched on the shared labels, products
 * emitted, sorted and reduced; zeros dropped) and, with TR_REG_ON_CUDA, CUDA dense (the same evaluator, one work item
 * per result entry). Inputs not in the kernel's form are converted first (a converted copy is workspace). For every
 * candidate the plan computes, exactly, each step's result shape (labels, dims, cells), summed combinations, stored
 * entries (dense: cells; sparse: an exact upper bound min(cells, matched pairs), matched pairs <= min(n_A x prod of
 * B's other dims, n_B x prod of A's other dims)), bytes (header + tr_tensor_dense_data_bytes / _sparse_data_bytes of
 * that shape), scratch (sparse: TR_TENSOR_PAIR_BYTES per matched pair + TR_TENSOR_KEY_BYTES per operand entry) and the
 * live bytes during the step (converted inputs and intermediates
 * still needed, the result, the scratch: intermediates are freed right after their consumer, which is their
 * materialization lifetime), the peak (the largest live bytes) and the output. Magnitudes are proven per step: |result|
 * <= terms x bound_A x bound_B <= 2^127 - 1 from the inputs' exact bounds, else the candidate is refused
 * (TR_MATH_OVERFLOW, refusal kind TR_REG_REFUSE_BOUND). Needs: variables = labels, degree = the largest number of
 * labels of an operand or result, tensor = the largest dense array the candidate reads or writes (dense kernel: its
 * operands and results), support = the largest sparse operand or result bound (sparse kernel), output = the output's
 * cells (dense) or bound (sparse), bytes = the peak (CUDA: the device copies' peak, the host output and converted
 * inputs).
 *
 * Costs (FND-008). The amounts are exact work counts: scan (operand entries read: dense 2 x products, sparse the
 * operands' entries), chain (products: dense cells x terms, sparse matched pairs), construct (elements built: dense
 * results, sparse pairs x ceil(log2 pairs) for sorting plus the results, converted inputs' entries), materialize
 * (output bytes), transfer (CUDA: host <-> device bytes), fixed (steps; CUDA: launches + 1). The FND-008 planner
 * (tr_planner_choose, math_planner.h) costs them (amount x rate per component, relative work units, NOT time), checks
 * every candidate's needs against the budget (a candidate over budget is rejected with the resource, need and limit),
 * and chooses the cheapest admissible one (the first on a tie): in rounds of TR_REG_MAX_CANDIDATES candidates and a
 * final round over the rounds' winners, so up to TR_TENSOR_MAX_CANDIDATES candidates keep the planner's statuses,
 * reasons, refusals and costs. The family modules have no registry rates (by design), so the plan passes its rates
 * (tr_tensor_default_rates, or the caller's) as the planner's override table in row TR_TENSOR_COST_ROW; the
 * candidates carry that row as their module only during the choice and module 0 (a family candidate) in the
 * evidence. A force (a candidate index) runs exactly that candidate or refuses with its code; it is never replaced.
 * With no admissible candidate the plan is refused (TR_MATH_BUDGET with the first over-budget candidate's refusal
 * when one was over budget, else the first refusal's code) and still returned with all its evidence.
 *
 * Execution: tr_tensor_net_execute runs a CPU candidate (any admissible one, so every order can be checked) and
 * records the actual stored entries, bytes and products of every step and the actual peak; a CUDA candidate runs
 * through tr_tensor_net_program (the plain step program) and tr_mgpu_tensor_contract on the dense views of the inputs,
 * whose output tr_tensor_from_dense adopts. CPU dense, CUDA and CPU sparse executions give the same tensor exactly.
 *
 * Budgets (math_budgets.h) are checked before allocation: builds (variables = columns read, degree = m, tensor =
 * G x cells for dense states and scratches, support = the largest group's stored entries or terms, states = G x
 * cells of a dense state or scratch, bytes = every allocation), conversions and operations (their results), networks
 * (bytes of ungrouped copies), plans (iterations = orders searched, bytes = the plan) and executions (the candidate's
 * needs). Refusals: TR_MATH_BUDGET with the tr_math_refusal. Errors: TR_MATH_INVALID, _MEMORY, _OVERFLOW, _BUDGET,
 * _UNSUPPORTED, _DOMAIN_ERROR, _DEVICE (CUDA). Objects are immutable once built; constructors return NULL and set
 * *error, leaving their inputs unchanged; read-only calls may run concurrently. */
#ifndef TORBLES_MATH_TENSOR_H
#define TORBLES_MATH_TENSOR_H
#include <stdint.h>
#include <string.h>
#include "math_num.h"
#include "math_budgets.h"
#include "math_scope.h"
#include "math_registry.h"
#ifdef __cplusplus
extern "C" {
#endif

#define TR_TENSOR_ABI 1
#define TR_TENSOR_MAX_ORDER 8
#define TR_TENSOR_MAX_FEATURES 64        /* columns of one feature axis */
#define TR_TENSOR_MAX_DIM UINT64_C(4294967295)
#define TR_TENSOR_COUNT_FACTOR 4          /* AUTO counts a sparse build densely when G x cells <= 4 x emissions + */
#define TR_TENSOR_COUNT_SLACK 65536       /*   65536 */
#define TR_TENSOR_NET_MAX 8               /* tensors of a network */
#define TR_TENSOR_MAX_LABELS 16           /* labels of a network (and of one step) */
#define TR_TENSOR_MAX_STEPS TR_TENSOR_NET_MAX
#define TR_TENSOR_PLAN_ORDERS 64          /* orders searched by default */
#define TR_TENSOR_MAX_CANDIDATES (TR_REG_MAX_CANDIDATES * TR_REG_MAX_CANDIDATES)   /* 256 */
#define TR_TENSOR_COST_ROW 1              /* the planner's override-table row the candidates are costed in */
#define TR_TENSOR_PAIR_BYTES 48           /* sparse kernel scratch per matched pair: an emitted record */
#define TR_TENSOR_KEY_BYTES 16            /* and per operand entry: its sort key and position */
enum { TR_TENSOR_GENERIC = 0, TR_TENSOR_MOMENT = 1, TR_TENSOR_COUNT = 2 };
enum { TR_TENSOR_DENSE = 1, TR_TENSOR_SPARSE = 2, TR_TENSOR_FACTORED = 3 };
enum { TR_TENSOR_LISTWISE = 1, TR_TENSOR_SQL = 2 };
enum { TR_TENSOR_AXIS_INDEX = 0, TR_TENSOR_AXIS_FEATURE = 1, TR_TENSOR_AXIS_CATEGORY = 2, TR_TENSOR_AXIS_GROUP = 3 };
enum { TR_TENSOR_DOMAIN_SCOPE = 0, TR_TENSOR_DOMAIN_EXPLICIT = 1 };
enum { TR_TENSOR_NEED_CPU = 0, TR_TENSOR_NEED_DEVICE = 1, TR_TENSOR_NEED_ADOPT = 2 };
enum { TR_TENSOR_STRATEGY_AUTO = 0, TR_TENSOR_STRATEGY_COUNT = 1, TR_TENSOR_STRATEGY_SORT = 2,
       TR_TENSOR_STRATEGY_DIRECT = 3,   /* a dense or factored state built in place */
       TR_TENSOR_STRATEGY_DERIVED = 4,  /* converted, sliced, marginalized, permuted, an outer product */
       TR_TENSOR_STRATEGY_DEVICE = 5,   /* adopted from the CUDA path */
       TR_TENSOR_STRATEGY_ARRAYS = 6,   /* given as arrays */
       TR_TENSOR_STRATEGY_CONTRACTED = 7 };
enum { TR_TENSOR_KERNEL_DENSE = 1, TR_TENSOR_KERNEL_SPARSE = 2 };   /* bits of tr_tensor_plan_options.kernels */

/* ------------------------------------------------------------------ ABI structs */
/* what to build from a scope (736 bytes) */
typedef struct {
    int32_t kind;                 /* TR_TENSOR_MOMENT / TR_TENSOR_COUNT */
    int32_t order;                /* m, 0..TR_TENSOR_MAX_ORDER */
    int32_t repr;                 /* DENSE / SPARSE / FACTORED (MOMENT under LISTWISE only) */
    int32_t policy;               /* MOMENT: LISTWISE / SQL; COUNT: 0 */
    int32_t weight;               /* the scope column of the weight, or -1 */
    int32_t strategy;             /* sparse on the CPU: AUTO / COUNT / SORT (else 0) */
    int32_t domain;               /* COUNT: SCOPE / EXPLICIT (MOMENT: 0) */
    int32_t reserved;             /* 0 */
    int32_t ncols[TR_TENSOR_MAX_ORDER];                          /* MOMENT: 1..64 per axis; COUNT: 1 */
    int8_t cols[TR_TENSOR_MAX_ORDER][TR_TENSOR_MAX_FEATURES];    /* scope columns (-1: the constant 1, MOMENT only) */
    int32_t nullslot[TR_TENSOR_MAX_ORDER];                       /* COUNT explicit: 1 = a NULL slot */
    int64_t lo[TR_TENSOR_MAX_ORDER], hi[TR_TENSOR_MAX_ORDER];    /* COUNT explicit: inclusive bounds (hi = lo - 1: none) */
} tr_tensor_spec;

/* the description of one axis (112 bytes) */
typedef struct {
    int32_t kind;                 /* TR_TENSOR_AXIS_* */
    int32_t ncols;                /* FEATURE: its columns (= dim) */
    int32_t col;                  /* CATEGORY: the scope column (-1 otherwise) */
    int32_t nullslot;             /* CATEGORY: 1 = index span is the NULL slot */
    int64_t lo;                   /* CATEGORY: the value at index 0 */
    uint64_t span;                /* CATEGORY: the values */
    uint64_t dim;
    int32_t scale, reserved;      /* CATEGORY: the column's decimal scale */
    int8_t cols[TR_TENSOR_MAX_FEATURES];   /* FEATURE: scope columns (-1: the constant 1) */
} tr_tensor_axis;

/* a tensor's description (232 bytes) */
typedef struct {
    uint64_t ngroups, present, rows, used;   /* groups, groups with rows, selected rows, contributing rows */
    uint64_t cells, nnz, nterms, bytes;      /* cells per group; stored entries (dense: G x cells); terms; owned bytes */
    uint64_t dims[TR_TENSOR_MAX_ORDER];
    uint64_t fdim;                           /* factored: sum of the dims */
    tr_math_i128 bound;                      /* every |entry| <= bound (dense, sparse: the exact maximum) */
    uint64_t digest[2];                      /* the scope descriptor's digest (0 when unscoped) */
    int32_t kind, repr, order, policy;
    int32_t has_counts, weighted, scoped, strategy;
    int32_t symmetric, weight, wscale, reserved;   /* symmetric: every axis lists the same columns */
} tr_tensor_info;

/* The states of a tensor (224 bytes): pointers into a built one (tr_tensor_get_states, read only, valid while it
 * lives) or the host arrays a CUDA build writes (capacity entries; tr_tensor_from_states adopts them). */
typedef struct {
    uint64_t ngroups, cells, capacity, nnz;  /* dense: nnz = capacity = G x cells; sparse: nnz written */
    int32_t order, repr, has_counts, reserved;
    uint64_t dims[TR_TENSOR_MAX_ORDER];
    uint64_t *rows, *used;                   /* [G] */
    uint64_t *offset;                        /* sparse [G + 1]; factored: the term offsets [G + 1] */
    uint32_t *coord;                         /* sparse [capacity x order] */
    tr_math_i128 *value;                     /* dense [G x cells]; sparse [capacity] */
    uint64_t *count;                         /* has_counts: like value */
    uint64_t nterms, fdim;                   /* factored */
    tr_math_i128 *weight;                    /* factored [nterms] */
    int64_t *factor;                         /* factored [nterms x fdim] */
} tr_tensor_states;

/* ---------------------------------------------------------------- networks and plans (TENSOR-002) */
/* a network: labels per tensor (a grouped tensor: its group label first) and the output (392 bytes) */
typedef struct {
    int32_t ntensors, noutput;
    int32_t nlabels[TR_TENSOR_NET_MAX];
    int32_t labels[TR_TENSOR_NET_MAX][TR_TENSOR_MAX_ORDER + 1];
    int32_t output[TR_TENSOR_MAX_LABELS];
} tr_tensor_net_spec;

/* plan options (216 bytes): rates[0] CPU, rates[1] CUDA per FND-008 cost component when custom_rates */
typedef struct {
    int32_t max_orders;           /* 0: TR_TENSOR_PLAN_ORDERS; at most the iterations budget */
    int32_t devices;              /* TR_REG_ON_CPU | TR_REG_ON_CUDA (0: CPU) */
    int32_t kernels;              /* TR_TENSOR_KERNEL_DENSE | _SPARSE (0: both) */
    int32_t force;                /* -1: none; else the candidate index that must run */
    int32_t custom_rates, reserved;
    double rates[2][TR_COST_COMPONENTS];
} tr_tensor_plan_options;

/* one step of a candidate (128 bytes) */
typedef struct {
    int32_t a, b;                 /* operand slots (inputs 0..n-1, step s's result n + s); b = -1: a reduction */
    int32_t nout, nsum;
    int8_t out[TR_TENSOR_MAX_LABELS], sum[TR_TENSOR_MAX_LABELS];   /* the result's labels in order; summed labels */
    uint64_t cells, terms;        /* result cells; summed combinations per result entry */
    uint64_t nnz;                 /* stored result entries: dense cells, sparse an exact upper bound */
    uint64_t pairs;               /* products: dense cells x terms, sparse matched pairs (upper bound) */
    uint64_t bytes;               /* the result's bytes (dense exact, sparse an upper bound) */
    uint64_t scratch;             /* the step's scratch bytes (sparse: pairs and sort keys) */
    uint64_t live;                /* bytes held while the step runs */
    int32_t free_a, free_b;       /* 1: the operand (an intermediate or a converted input) is freed after the step */
    tr_math_i128 bound;           /* |result entry| <= bound (proven) */
} tr_tensor_step_info;

/* one candidate: FND-008's candidate (amounts, needs, costs, status, reason, refusals) and the steps (1456 bytes) */
typedef struct {
    tr_reg_candidate reg;
    int32_t kernel, device, nsteps, order;    /* TR_TENSOR_KERNEL_*, TR_REG_ON_*, steps, its order's index */
    int32_t converted, reserved;              /* inputs converted to the kernel's form */
    uint64_t peak_bytes, output_bytes, output_cells, output_nnz;   /* output_nnz: dense cells, sparse a bound */
    uint64_t input_bytes;                     /* bytes of the converted input copies */
    tr_math_i128 bound;                       /* |output entry| <= bound */
    tr_tensor_step_info step[TR_TENSOR_MAX_STEPS];
} tr_tensor_candidate;

/* a plan's summary (80 bytes) */
typedef struct {
    int32_t ncandidates, chosen, code, reason;   /* chosen -1 when refused; code the plan's native code */
    int32_t complete, norders, forced, devices;  /* complete: every order enumerated */
    uint64_t total_orders;                       /* orders of the network (saturating) */
    uint64_t bytes;
    tr_math_refusal refusal;                     /* the plan-level budget refusal */
    int32_t ntensors, nlabels;
} tr_tensor_plan_info;

/* an execution's actual sizes (240 bytes) */
typedef struct {
    int32_t candidate, kernel, device, nsteps;
    uint64_t nnz[TR_TENSOR_MAX_STEPS];        /* stored entries of each step's result */
    uint64_t bytes[TR_TENSOR_MAX_STEPS];      /* bytes of each step's result */
    uint64_t pairs[TR_TENSOR_MAX_STEPS];      /* products computed (sparse: matched pairs) */
    uint64_t peak_bytes, output_nnz, output_bytes;
    uint64_t converted_bytes;
} tr_tensor_exec;

/* the dense step program a CUDA candidate (or a CPU dense one) runs (6960 bytes); a work item = one result entry */
typedef struct {
    int32_t a, b, nout, nsum;
    uint64_t cells, terms;
    uint64_t dim_out[TR_TENSOR_MAX_LABELS], dim_sum[TR_TENSOR_MAX_LABELS];
    uint64_t sa_out[TR_TENSOR_MAX_LABELS], sa_sum[TR_TENSOR_MAX_LABELS];   /* element strides in A (0: not in A) */
    uint64_t sb_out[TR_TENSOR_MAX_LABELS], sb_sum[TR_TENSOR_MAX_LABELS];   /* and in B */
    tr_math_i128 bound;
    int32_t free_a, free_b;
    uint64_t a_cells, b_cells;
} tr_tensor_dense_step;
typedef struct {
    int32_t ninputs, nsteps, noutput, reserved;
    uint64_t input_cells[TR_TENSOR_NET_MAX];
    tr_math_i128 input_bound[TR_TENSOR_NET_MAX];
    uint64_t output_cells;
    uint64_t output_dims[TR_TENSOR_MAX_LABELS];
    tr_tensor_dense_step step[TR_TENSOR_MAX_STEPS];
} tr_tensor_program;
/* a dense input of the device contraction (88 bytes) */
typedef struct {
    const tr_math_i128 *value;
    uint64_t cells;
    int32_t order, reserved;
    uint64_t dims[TR_TENSOR_MAX_ORDER];
} tr_tensor_dense_view;

typedef struct tr_tensor_s tr_tensor;
typedef struct tr_tensor_builder_s tr_tensor_builder;
typedef struct tr_tensor_net_s tr_tensor_net;
typedef struct tr_tensor_plan_s tr_tensor_plan;

/* ------------------------------------------------------------------ sizes (shared, exact, saturating) */
typedef unsigned __int128 tr_tensor_u128;
#define TR_TENSOR_I128_MAX ((((tr_tensor_u128)1) << 127) - 1)
/* prod dims (1 for order 0), TR_BUDGET_UNKNOWN past 2^64 - 1 */
static inline uint64_t tr_tensor_cells(const uint64_t *dims, int32_t order) {
    uint64_t c = 1;
    for (int32_t k = 0; k < order; k++) c = tr_budget_mul(c, dims[k]);
    return c;
}
static inline uint64_t tr_tensor_dense_data_bytes(uint64_t ngroups, uint64_t cells, int has_counts) {
    uint64_t e = tr_budget_mul(ngroups, cells);
    return tr_budget_add(tr_budget_mul(ngroups, 16), tr_budget_mul(e, has_counts ? 24 : 16));
}
static inline uint64_t tr_tensor_sparse_data_bytes(uint64_t ngroups, uint64_t nnz, int32_t order, int has_counts) {
    uint64_t per = (uint64_t)order * 4 + 16 + (has_counts ? 8 : 0);
    return tr_budget_add(tr_budget_add(tr_budget_mul(ngroups, 16), tr_budget_mul(tr_budget_add(ngroups, 1), 8)),
                         tr_budget_mul(nnz, per));
}
static inline uint64_t tr_tensor_factored_data_bytes(uint64_t ngroups, uint64_t nterms, uint64_t fdim) {
    return tr_budget_add(tr_budget_add(tr_budget_mul(ngroups, 16), tr_budget_mul(tr_budget_add(ngroups, 1), 8)),
                         tr_budget_mul(nterms, tr_budget_add(16, tr_budget_mul(fdim, 8))));
}
TR_NUM_HD tr_tensor_u128 tr_tensor_mag(__int128 x) { return x < 0 ? (tr_tensor_u128)0 - (tr_tensor_u128)x : (tr_tensor_u128)x; }
/* a * b saturated above limit */
static inline tr_tensor_u128 tr_tensor_satmul(tr_tensor_u128 a, tr_tensor_u128 b, tr_tensor_u128 limit) {
    if (!a || !b) return 0;
    return a > limit / b ? limit + 1 : a * b;
}
static inline tr_tensor_u128 tr_tensor_satadd(tr_tensor_u128 a, tr_tensor_u128 b, tr_tensor_u128 limit) {
    return a > limit - (b > limit ? limit : b) ? limit + 1 : a + b;
}

/* ------------------------------------------------------------------ the resolved build plan (host and kernels) */
/* A spec resolved over a scope view: dims, strides, columns, the COUNT domain. Passed by value to the kernels. */
typedef struct {
    int32_t kind, order, policy, weight;
    int32_t repr, has_counts, symmetric, strategy;
    uint32_t refmask, reserved;               /* bit c: scope column c is a factor of some axis */
    uint64_t dims[TR_TENSOR_MAX_ORDER], stride[TR_TENSOR_MAX_ORDER], cells;
    int32_t ncols[TR_TENSOR_MAX_ORDER];
    int8_t cols[TR_TENSOR_MAX_ORDER][TR_TENSOR_MAX_FEATURES];
    int32_t nullslot[TR_TENSOR_MAX_ORDER];
    int64_t lo[TR_TENSOR_MAX_ORDER];
    uint64_t span[TR_TENSOR_MAX_ORDER];
} tr_tensor_build_plan;

static inline int tr_tensor_col_ok(const tr_scope_view *v, int32_t c) { return c >= 0 && c < v->ncols; }
/* 0, or TR_MATH_INVALID (malformed), _UNSUPPORTED (order, columns, cells >= 2^64, factored outside LISTWISE MOMENT),
 * _DOMAIN_ERROR (an explicit domain that misses a selected value or NULL) */
static inline int32_t tr_tensor_plan_make(const tr_scope_view *v, const tr_tensor_spec *s, tr_tensor_build_plan *p) {
    if (!v || !s || !p) return -1;
    memset(p, 0, sizeof *p);
    if (s->reserved || (s->kind != TR_TENSOR_MOMENT && s->kind != TR_TENSOR_COUNT) ||
        (s->repr != TR_TENSOR_DENSE && s->repr != TR_TENSOR_SPARSE && s->repr != TR_TENSOR_FACTORED) ||
        (s->weight != -1 && !tr_tensor_col_ok(v, s->weight)) ||
        s->strategy < TR_TENSOR_STRATEGY_AUTO || s->strategy > TR_TENSOR_STRATEGY_SORT) return -1;
    if (s->order < 0 || s->order > TR_TENSOR_MAX_ORDER) return -12;
    if (s->kind == TR_TENSOR_MOMENT && s->policy != TR_TENSOR_LISTWISE && s->policy != TR_TENSOR_SQL) return -1;
    if (s->kind == TR_TENSOR_COUNT && (s->policy != 0 ||
        (s->domain != TR_TENSOR_DOMAIN_SCOPE && s->domain != TR_TENSOR_DOMAIN_EXPLICIT))) return -1;
    if (s->kind == TR_TENSOR_MOMENT && s->domain != 0) return -1;
    if (s->repr == TR_TENSOR_FACTORED && (s->kind != TR_TENSOR_MOMENT || s->policy != TR_TENSOR_LISTWISE)) return -12;
    p->kind = s->kind; p->order = s->order; p->policy = s->policy; p->weight = s->weight; p->repr = s->repr;
    p->strategy = s->strategy;
    p->has_counts = s->kind == TR_TENSOR_MOMENT && s->policy == TR_TENSOR_SQL;
    const int32_t m = s->order;
    for (int32_t k = 0; k < m; k++) {
        if (s->kind == TR_TENSOR_MOMENT) {
            if (s->ncols[k] < 1 || s->ncols[k] > TR_TENSOR_MAX_FEATURES) return s->ncols[k] < 1 ? -1 : -12;
            p->ncols[k] = s->ncols[k];
            for (int32_t i = 0; i < s->ncols[k]; i++) {
                int32_t c = s->cols[k][i];
                if (c != -1 && !tr_tensor_col_ok(v, c)) return -1;
                p->cols[k][i] = (int8_t)c;
                if (c >= 0) p->refmask |= UINT32_C(1) << c;
            }
            p->dims[k] = (uint64_t)s->ncols[k];
        } else {
            if (s->ncols[k] != 1 || !tr_tensor_col_ok(v, s->cols[k][0])) return -1;
            int32_t c = s->cols[k][0];
            const tr_scope_view_column *vc = &v->col[c];
            p->ncols[k] = 1; p->cols[k][0] = (int8_t)c;
            int has_null = vc->valid != NULL;
            if (s->domain == TR_TENSOR_DOMAIN_EXPLICIT) {
                if ((s->nullslot[k] != 0 && s->nullslot[k] != 1) || (__int128)s->hi[k] < (__int128)s->lo[k] - 1) return -1;
                if (vc->nvalid && (vc->min < s->lo[k] || vc->max > s->hi[k])) return -6;
                if (has_null && !s->nullslot[k]) return -6;
                p->lo[k] = s->lo[k];
                p->span[k] = s->hi[k] >= s->lo[k] ? (uint64_t)s->hi[k] - (uint64_t)s->lo[k] + 1 : 0;
                if (s->hi[k] >= s->lo[k] && p->span[k] == 0) return -12;          /* 2^64 values */
                p->nullslot[k] = s->nullslot[k];
            } else {
                p->lo[k] = vc->nvalid ? vc->min : 0;
                p->span[k] = vc->nvalid ? (uint64_t)vc->max - (uint64_t)vc->min + 1 : 0;
                if (vc->nvalid && p->span[k] == 0) return -12;
                p->nullslot[k] = has_null;
            }
            if (p->span[k] + (uint64_t)p->nullslot[k] < p->span[k]) return -12;
            p->dims[k] = p->span[k] + (uint64_t)p->nullslot[k];   /* 0 only when no row is selected */
            if (p->dims[k] > TR_TENSOR_MAX_DIM) return -12;
        }
    }
    uint64_t c = 1;
    for (int32_t k = m - 1; k >= 0; k--) {
        p->stride[k] = c;
        c = tr_budget_mul(c, p->dims[k]);
    }
    if (c == TR_BUDGET_UNKNOWN) return -12;
    p->cells = c;
    int sym = s->kind == TR_TENSOR_MOMENT && m >= 2;
    for (int32_t k = 1; sym && k < m; k++) {
        if (p->ncols[k] != p->ncols[0]) { sym = 0; break; }
        for (int32_t i = 0; i < p->ncols[0]; i++) if (p->cols[k][i] != p->cols[0][i]) { sym = 0; break; }
    }
    p->symmetric = sym;
    return 0;
}

/* the rows that can contribute (for the overflow proof): the selected rows, at most the present values of the
 * weight and, under LISTWISE, of every referenced column */
static inline uint64_t tr_tensor_plan_rows(const tr_scope_view *v, const tr_tensor_build_plan *p) {
    uint64_t n = v->selected;
    if (p->weight >= 0 && v->col[p->weight].nvalid < n) n = v->col[p->weight].nvalid;
    if (p->kind == TR_TENSOR_MOMENT && p->policy == TR_TENSOR_LISTWISE)
        for (int32_t c = 0; c < v->ncols; c++)
            if (((p->refmask >> c) & 1) && v->col[c].nvalid < n) n = v->col[c].nvalid;
    return n;
}
static inline uint64_t tr_tensor_col_mag(const tr_scope_view *v, int32_t c) {
    if (c < 0) return 1;
    const tr_scope_view_column *k = &v->col[c];
    uint64_t m = k->nvalid ? (tr_abs_u64(k->min) > tr_abs_u64(k->max) ? tr_abs_u64(k->min) : tr_abs_u64(k->max)) : 0;
    return m < 1 ? 1 : m;
}
/* the proven bound N x max|w| x prod_k max_axis_k (saturated past 2^127 - 1): every entry, product and partial sum */
static inline tr_tensor_u128 tr_tensor_plan_bound(const tr_scope_view *v, const tr_tensor_build_plan *p) {
    const tr_tensor_u128 limit = TR_TENSOR_I128_MAX;
    tr_tensor_u128 b = tr_tensor_plan_rows(v, p);
    if (p->weight >= 0) b = tr_tensor_satmul(b, tr_tensor_col_mag(v, p->weight), limit);
    if (p->kind == TR_TENSOR_MOMENT)
        for (int32_t k = 0; k < p->order; k++) {
            uint64_t m = 1;
            for (int32_t i = 0; i < p->ncols[k]; i++) { uint64_t x = tr_tensor_col_mag(v, p->cols[k][i]); if (x > m) m = x; }
            b = tr_tensor_satmul(b, m, limit);
        }
    return b;
}

/* ------------------------------------------------------------------ per row (CPU loops and kernels) */
TR_NUM_HD int tr_tensor_weight_ok(const tr_scope_view *v, const tr_tensor_build_plan *p, uint64_t row, __int128 *w) {
    if (p->weight < 0) { *w = 1; return 1; }
    if (!tr_scope_present(v, p->weight, row)) return 0;
    *w = tr_scope_value(v, p->weight, row);
    return 1;
}
/* whether a selected row contributes at all (weight present; LISTWISE: every referenced column present) */
TR_NUM_HD int tr_tensor_row_ok(const tr_scope_view *v, const tr_tensor_build_plan *p, uint64_t row, __int128 *w) {
    if (!tr_tensor_weight_ok(v, p, row, w)) return 0;
    if (p->kind == TR_TENSOR_MOMENT && p->policy == TR_TENSOR_LISTWISE)
        for (int32_t c = 0; c < v->ncols; c++)
            if (((p->refmask >> c) & 1) && !tr_scope_present(v, c, row)) return 0;
    return 1;
}
/* coordinate i of axis k at a row: 1 with *x (an admissible factor: present, and nonzero under LISTWISE) */
TR_NUM_HD int tr_tensor_factor(const tr_scope_view *v, const tr_tensor_build_plan *p, int32_t k, uint64_t i,
                               uint64_t row, __int128 *x) {
    int32_t c = p->cols[k][i];
    if (c < 0) { *x = 1; return 1; }
    if (!tr_scope_present(v, c, row)) return 0;
    *x = tr_scope_value(v, c, row);
    return p->policy == TR_TENSOR_SQL || *x != 0;
}
/* the cell (linear index within the group) of a COUNT row */
TR_NUM_HD uint64_t tr_tensor_count_cell(const tr_scope_view *v, const tr_tensor_build_plan *p, uint64_t row) {
    uint64_t idx = 0;
    for (int32_t k = 0; k < p->order; k++) {
        int32_t c = p->cols[k][0];
        uint64_t i = tr_scope_present(v, c, row) ? (uint64_t)tr_scope_value(v, c, row) - (uint64_t)p->lo[k] : p->span[k];
        idx += i * p->stride[k];
    }
    return idx;
}
/* The emissions of a contributing row: (linear index, w x product) for every multi-index whose factors are
 * admissible (COUNT: the row's cell); canonical: non-decreasing multi-indices only (symmetric dense builds). */
typedef struct {
    int64_t i[TR_TENSOR_MAX_ORDER];
    __int128 pre[TR_TENSOR_MAX_ORDER + 1];
    uint64_t idx[TR_TENSOR_MAX_ORDER + 1];
    uint64_t row;
    int32_t canonical, done;
} tr_tensor_emitter;
/* advance axis `level` past its coordinate and refill the deeper axes: 1 with a new emission, 0 at the end */
TR_NUM_HD int tr_tensor_emit_step(const tr_scope_view *v, const tr_tensor_build_plan *p, tr_tensor_emitter *e, int32_t level) {
    const int32_t m = p->order;
    for (;;) {
        __int128 x = 0;
        uint64_t c = (uint64_t)(e->i[level] + 1);
        while (c < p->dims[level] && !tr_tensor_factor(v, p, level, c, e->row, &x)) c++;
        if (c >= p->dims[level]) {
            if (level == 0) { e->done = 1; return 0; }
            level--;
            continue;
        }
        e->i[level] = (int64_t)c;
        e->pre[level + 1] = e->pre[level] * x;
        e->idx[level + 1] = e->idx[level] + c * p->stride[level];
        if (level == m - 1) return 1;
        level++;
        e->i[level] = e->canonical ? (int64_t)c - 1 : -1;
    }
}
TR_NUM_HD int tr_tensor_emit_first(const tr_scope_view *v, const tr_tensor_build_plan *p, uint64_t row, __int128 w,
                                   int32_t canonical, tr_tensor_emitter *e) {
    e->row = row; e->canonical = canonical; e->done = 0;
    e->pre[0] = w; e->idx[0] = 0;
    if (p->kind == TR_TENSOR_COUNT) { e->idx[p->order] = tr_tensor_count_cell(v, p, row); e->pre[p->order] = w; return 1; }
    if (p->order == 0) return 1;
    e->i[0] = -1;
    return tr_tensor_emit_step(v, p, e, 0);
}
TR_NUM_HD int tr_tensor_emit_next(const tr_scope_view *v, const tr_tensor_build_plan *p, tr_tensor_emitter *e) {
    if (p->kind == TR_TENSOR_COUNT || p->order == 0 || e->done) return 0;
    return tr_tensor_emit_step(v, p, e, p->order - 1);
}
/* the number of emissions of a contributing row (non-canonical): prod_k admissible coordinates of axis k */
TR_NUM_HD uint64_t tr_tensor_emissions(const tr_scope_view *v, const tr_tensor_build_plan *p, uint64_t row) {
    if (p->kind == TR_TENSOR_COUNT) return 1;
    uint64_t n = 1;
    for (int32_t k = 0; k < p->order && n; k++) {
        uint64_t a = 0;
        __int128 x;
        for (uint64_t i = 0; i < p->dims[k]; i++) a += (uint64_t)tr_tensor_factor(v, p, k, i, row, &x);
        n *= a;
    }
    return n;
}
/* MOMENT entry e (linear index) at a contributing row: 1 with *x = w x product (its factors admissible), else 0 */
TR_NUM_HD int tr_tensor_entry_at(const tr_scope_view *v, const tr_tensor_build_plan *p, uint64_t e, uint64_t row,
                                 __int128 w, __int128 *x) {
    __int128 prod = w;
    for (int32_t k = 0; k < p->order; k++) {
        uint64_t i = (e / p->stride[k]) % p->dims[k];
        __int128 f;
        if (!tr_tensor_factor(v, p, k, i, row, &f)) return 0;
        prod *= f;
    }
    *x = prod;
    return 1;
}
/* the linear index of the sorted (non-decreasing) multi-index of e: the canonical entry of a symmetric tensor */
TR_NUM_HD uint64_t tr_tensor_canonical(const uint64_t *dims, const uint64_t *stride, int32_t order, uint64_t e) {
    uint64_t c[TR_TENSOR_MAX_ORDER];
    for (int32_t k = 0; k < order; k++) c[k] = (e / stride[k]) % dims[k];
    for (int32_t k = 1; k < order; k++) {                  /* insertion sort of at most 8 coordinates */
        uint64_t x = c[k];
        int32_t j = k - 1;
        while (j >= 0 && c[j] > x) { c[j + 1] = c[j]; j--; }
        c[j + 1] = x;
    }
    uint64_t r = 0;
    for (int32_t k = 0; k < order; k++) r += c[k] * stride[k];
    return r;
}

/* ------------------------------------------------------------------ the dense contraction step (CPU and kernels) */
/* result entry o of a dense step: sum over the summed combinations of A[...] B[...] (B NULL: a reduction); the
 * program proved |result| <= bound <= 2^127 - 1, so the 128-bit products and sums are exact */
TR_NUM_HD __int128 tr_tensor_dense_entry(const tr_tensor_dense_step *s, const tr_math_i128 *A, const tr_math_i128 *B,
                                         uint64_t o) {
    uint64_t oa = 0, ob = 0, rem = o;
    for (int32_t k = s->nout - 1; k >= 0; k--) {
        uint64_t c = rem % s->dim_out[k];
        rem /= s->dim_out[k];
        oa += c * s->sa_out[k];
        ob += c * s->sb_out[k];
    }
    uint64_t cs[TR_TENSOR_MAX_LABELS];
    for (int32_t j = 0; j < s->nsum; j++) cs[j] = 0;
    __int128 acc = 0;
    for (uint64_t t = 0; t < s->terms; t++) {
        __int128 x = tr_i128_get(A[oa]);
        if (B) x *= tr_i128_get(B[ob]);
        acc += x;
        for (int32_t j = s->nsum - 1; j >= 0; j--) {
            cs[j]++;
            oa += s->sa_sum[j];
            ob += s->sb_sum[j];
            if (cs[j] < s->dim_sum[j]) break;
            oa -= s->sa_sum[j] * s->dim_sum[j];
            ob -= s->sb_sum[j] * s->dim_sum[j];
            cs[j] = 0;
        }
    }
    return acc;
}
/* the bytes tr_mgpu_tensor_contract copies and holds on the device at its peak (inputs, live results, the output),
 * plus the host output it writes: its bytes need */
static inline uint64_t tr_tensor_program_device_bytes(const tr_tensor_program *p) {
    uint64_t live = 0, peak = 0, alive[TR_TENSOR_NET_MAX + TR_TENSOR_MAX_STEPS];
    int32_t nslots = p->ninputs + p->nsteps;
    if (nslots > TR_TENSOR_NET_MAX + TR_TENSOR_MAX_STEPS) return TR_BUDGET_UNKNOWN;
    for (int32_t i = 0; i < nslots; i++) alive[i] = 0;
    for (int32_t i = 0; i < p->ninputs; i++) { alive[i] = tr_budget_mul(p->input_cells[i], 16); live = tr_budget_add(live, alive[i]); }
    peak = live;
    for (int32_t s = 0; s < p->nsteps; s++) {
        const tr_tensor_dense_step *st = &p->step[s];
        uint64_t out = tr_budget_mul(st->cells, 16);
        live = tr_budget_add(live, out);
        alive[p->ninputs + s] = out;
        if (live > peak) peak = live;
        if (st->a >= p->ninputs && st->free_a) { live -= alive[st->a]; alive[st->a] = 0; }
        if (st->b >= p->ninputs && st->free_b) { live -= alive[st->b]; alive[st->b] = 0; }
    }
    return tr_budget_add(peak, tr_budget_mul(p->output_cells, 16));
}

/* ------------------------------------------------------------------ device needs (CUDA host code and libalg) */
static inline uint64_t tr_tensor_pow2_at_least(uint64_t n) {
    uint64_t c = 1;
    while (c < n) { if (c > (UINT64_MAX >> 1)) return TR_BUDGET_UNKNOWN; c <<= 1; }
    return c;
}
/* entries a sparse build can hold at most: min(G x cells, selected rows x (COUNT: 1, MOMENT: cells)) */
static inline uint64_t tr_tensor_sparse_capacity(const tr_scope_view *v, const tr_tensor_build_plan *p) {
    uint64_t total = tr_budget_mul(v->ngroups, p->cells);
    uint64_t rows = tr_budget_mul(v->selected, p->kind == TR_TENSOR_COUNT ? 1 : p->cells);
    return rows < total ? rows : total;
}
/* hash slots of the device sparse build: the power of two >= 2 x capacity (load at most 1/2) */
static inline uint64_t tr_tensor_hash_slots(uint64_t capacity) {
    return tr_tensor_pow2_at_least(tr_budget_mul(capacity ? capacity : 1, 2));
}
/* the needs of tr_mgpu_tensor_build: variables (columns read), degree (m), tensor and states (dense: G x cells),
 * support (sparse: the capacity), bytes (device copies of the selection, ids, the columns read and their validity,
 * the row counters, the dense state or the hash table (8 + 16 (+ 8) bytes a slot), the canonical index list of a
 * symmetric dense build; sparse: the host's sorting copy of the occupied slots). *capacity: the output entries. */
static inline void tr_tensor_device_need(const tr_scope_view *v, const tr_tensor_build_plan *p, tr_math_need *need,
                                         uint64_t *capacity) {
    for (int i = 0; i < TR_BUDGET_RESOURCES; i++) need->need[i] = 0;
    uint64_t n = v->rows, G = v->ngroups, bytes = tr_budget_add(tr_budget_bitmap_bytes(n), tr_budget_mul(G, 16));
    if (v->groups) bytes = tr_budget_add(bytes, tr_budget_mul(n, 4));
    uint32_t read = p->refmask;
    if (p->kind == TR_TENSOR_COUNT) for (int32_t k = 0; k < p->order; k++) read |= UINT32_C(1) << p->cols[k][0];
    if (p->weight >= 0) read |= UINT32_C(1) << p->weight;
    int32_t ncols = 0;
    for (int32_t c = 0; c < v->ncols; c++) {
        if (!((read >> c) & 1)) continue;
        ncols++;
        bytes = tr_budget_add(bytes, tr_budget_mul(n, v->col[c].values ? 8 : 4));
        if (v->col[c].valid) bytes = tr_budget_add(bytes, n);
    }
    uint64_t total = tr_budget_mul(G, p->cells), cap = total, per = p->has_counts ? 24 : 16;
    if (p->repr == TR_TENSOR_DENSE) {
        bytes = tr_budget_add(bytes, tr_budget_mul(total, per));
        if (p->symmetric) bytes = tr_budget_add(bytes, tr_budget_mul(p->cells, 8));
        need->need[TR_BUDGET_TENSOR] = total;
        need->need[TR_BUDGET_STATES] = total;
    } else {
        cap = tr_tensor_sparse_capacity(v, p);
        uint64_t slots = tr_tensor_hash_slots(cap);
        bytes = tr_budget_add(bytes, tr_budget_mul(slots, per + 8));
        bytes = tr_budget_add(bytes, tr_budget_mul(cap, per + 8 + 16));     /* host sort of the occupied slots */
        need->need[TR_BUDGET_SUPPORT] = cap;
        need->need[TR_BUDGET_STATES] = slots;
    }
    need->need[TR_BUDGET_VARIABLES] = (uint64_t)ncols;
    need->need[TR_BUDGET_DEGREE] = (uint64_t)p->order;
    need->need[TR_BUDGET_BYTES] = bytes;
    if (capacity) *capacity = cap;
}
/* the 64-bit finalizer of splitmix64 (the device hash table's probe start) */
TR_NUM_HD uint64_t tr_tensor_hash64(uint64_t x) {
    x ^= x >> 30; x *= UINT64_C(0xbf58476d1ce4e5b9);
    x ^= x >> 27; x *= UINT64_C(0x94d049bb133111eb);
    return x ^ (x >> 31);
}

/* ------------------------------------------------------------------ TENSOR-001 (libalg) */
/* rates (relative work units per amount, NOT time) of the contraction candidates: [0] CPU, [1] CUDA */
void tr_tensor_default_rates(double rates[2][TR_COST_COMPONENTS]);
uint64_t tr_tensor_header_bytes(void);           /* the object's fixed bytes, included in every tensor's bytes */
tr_tensor_builder *tr_tensor_init(const tr_scope *, const tr_tensor_spec *, const tr_math_budget *budget,
                                  tr_math_refusal *refusal, int32_t *error);
/* adds the selected rows of [begin, end) (each physical row once over all calls and merged builders) */
int32_t tr_tensor_accumulate(tr_tensor_builder *, uint64_t begin, uint64_t end);
int32_t tr_tensor_merge_builder(tr_tensor_builder *into, const tr_tensor_builder *from);
/* every row covered: the immutable tensor (the builder is spent); budget: the final sparse state */
tr_tensor *tr_tensor_finish(tr_tensor_builder *, const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error);
void tr_tensor_builder_free(tr_tensor_builder *);
tr_tensor *tr_tensor_build(const tr_scope *, const tr_tensor_spec *, const tr_math_budget *budget,
                           tr_math_refusal *refusal, int32_t *error);
/* TR_TENSOR_NEED_CPU: what init checks (sparse: the scratch; the final state is checked at finish); _DEVICE: the
 * device copies of tr_mgpu_tensor_build; _ADOPT: the adopted state. *capacity (may be NULL): output entries. */
int32_t tr_tensor_need(const tr_scope *, const tr_tensor_spec *, int32_t target, tr_math_need *need, uint64_t *capacity);
/* adopts states computed elsewhere (the CUDA readback) as the tensor of `origin` under `spec`, after checks (shape,
 * rows, canonical order, the proven bound) */
tr_tensor *tr_tensor_from_states(const tr_scope *origin, const tr_tensor_spec *spec, const tr_tensor_states *,
                                 const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error);
/* GENERIC tensors from arrays: dense values [G x cells] (counts or NULL); sparse entries in any order (group ids or
 * NULL = 0, coordinates [nnz x order], values, counts or NULL: duplicates are summed, then zeros dropped and entries
 * sorted); declared factors (group ids [nterms] or NULL, weights, factors [nterms x sum dims]) */
tr_tensor *tr_tensor_from_dense(int32_t order, const uint64_t *dims, uint64_t ngroups, const tr_math_i128 *values,
                                const uint64_t *counts, const tr_math_budget *budget, tr_math_refusal *refusal,
                                int32_t *error);
tr_tensor *tr_tensor_from_sparse(int32_t order, const uint64_t *dims, uint64_t ngroups, uint64_t nnz,
                                 const uint32_t *groups, const uint32_t *coords, const tr_math_i128 *values,
                                 const uint64_t *counts, const tr_math_budget *budget, tr_math_refusal *refusal,
                                 int32_t *error);
tr_tensor *tr_tensor_from_factors(int32_t order, const uint64_t *dims, uint64_t ngroups, uint64_t nterms,
                                  const uint32_t *groups, const tr_math_i128 *weights, const int64_t *factors,
                                  const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error);
tr_tensor *tr_tensor_convert(const tr_tensor *, int32_t repr, const tr_math_budget *budget, tr_math_refusal *refusal,
                             int32_t *error);
/* keep the axes whose bit is set in keep (in their order) and sum out the others */
tr_tensor *tr_tensor_marginal(const tr_tensor *, uint32_t keep, const tr_math_budget *budget, tr_math_refusal *refusal,
                              int32_t *error);
/* axis k: fixed at lo[k] (bit k of fix; removed) or the range [lo[k], hi[k]) (kept) */
tr_tensor *tr_tensor_slice(const tr_tensor *, uint32_t fix, const uint64_t *lo, const uint64_t *hi,
                           const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error);
tr_tensor *tr_tensor_permute(const tr_tensor *, const int32_t *perm, const tr_math_budget *budget,
                             tr_math_refusal *refusal, int32_t *error);
tr_tensor *tr_tensor_outer(const tr_tensor *, const tr_tensor *, int32_t repr, const tr_math_budget *budget,
                           tr_math_refusal *refusal, int32_t *error);
tr_tensor *tr_tensor_ungroup(const tr_tensor *, const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error);
tr_tensor *tr_tensor_values(const tr_tensor *, const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error);
tr_tensor *tr_tensor_clone(const tr_tensor *, int32_t *error);
/* entry lookup in any form: *value (and *count when has_counts, else 0) */
int32_t tr_tensor_get(const tr_tensor *, uint64_t group, const uint32_t *coords, tr_math_i128 *value, uint64_t *count);
/* 1 when both hold the same state in the same representation; tr_tensor_same: the same tensor in any representation
 * (kind, order, dims, axes, policy, groups, rows, used, every entry and count) */
int32_t tr_tensor_equal(const tr_tensor *, const tr_tensor *);
int32_t tr_tensor_same(const tr_tensor *, const tr_tensor *);
int32_t tr_tensor_get_info(const tr_tensor *, tr_tensor_info *);
int32_t tr_tensor_get_axis(const tr_tensor *, int32_t axis, tr_tensor_axis *);
int32_t tr_tensor_get_states(const tr_tensor *, tr_tensor_states *);
void tr_tensor_free(tr_tensor *);

/* ------------------------------------------------------------------ TENSOR-002 (libalg) */
/* tensors are borrowed and must outlive the network (grouped ones are copied ungrouped, budgeted) */
tr_tensor_net *tr_tensor_net_create(const tr_tensor *const *tensors, const tr_tensor_net_spec *,
                                    const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error);
void tr_tensor_net_free(tr_tensor_net *);
/* Plans the network (see the header comment). Returns the plan's code; *plan is the plan with its evidence (also
 * when it is refused for want of an admissible candidate), NULL only when the search itself was refused. */
int32_t tr_tensor_net_plan(const tr_tensor_net *, const tr_tensor_plan_options *, const tr_math_budget *budget,
                           tr_tensor_plan **plan, tr_math_refusal *refusal);
int32_t tr_tensor_plan_get_info(const tr_tensor_plan *, tr_tensor_plan_info *);
int32_t tr_tensor_plan_get(const tr_tensor_plan *, int32_t index, tr_tensor_candidate *);
void tr_tensor_plan_free(tr_tensor_plan *);
/* runs candidate index (-1: the chosen one) of a plan of this network on the CPU; a rejected candidate is refused
 * with its code, a CUDA one with TR_MATH_UNSUPPORTED (tr_mgpu_tensor_contract runs it). rec may be NULL. */
tr_tensor *tr_tensor_net_execute(const tr_tensor_net *, const tr_tensor_plan *, int32_t index,
                                 const tr_math_budget *budget, tr_tensor_exec *rec, tr_math_refusal *refusal,
                                 int32_t *error);
/* the dense program of a dense candidate (CPU or CUDA; -1: the chosen one) */
int32_t tr_tensor_net_program(const tr_tensor_net *, const tr_tensor_plan *, int32_t index, tr_tensor_program *out);
/* input i of the network as a dense tensor (a new owned copy when it is not dense; a clone otherwise) */
tr_tensor *tr_tensor_net_input_dense(const tr_tensor_net *, int32_t i, const tr_math_budget *budget,
                                     tr_math_refusal *refusal, int32_t *error);

/* ------------------------------------------------------------------ the CUDA library (math_tensor_gpu.cu)
 * tr_mgpu_tensor_build: view = tr_scope_get_view of the scope; out: host arrays (out->ngroups = G, order and dims
 * as the spec resolves them, capacity = tr_tensor_need(..., TR_TENSOR_NEED_DEVICE) entries; dense: G x cells),
 * overwritten with exactly the states tr_tensor_build computes (adopt them with tr_tensor_from_states).
 * tr_mgpu_tensor_contract: a program of tr_tensor_net_program, the dense inputs (dims as the program's), out: the
 * output's cells; it re-proves the program's bounds from the inputs before launching. rec may be NULL. Both return 0
 * or a native error code (TR_MATH_DEVICE when the device fails). */
int tr_mgpu_tensor_build(const tr_scope_view *view, const tr_tensor_spec *spec, const tr_math_budget *budget,
                         tr_tensor_states *out, tr_math_refusal *refusal);
int tr_mgpu_tensor_contract(const tr_tensor_program *prog, const tr_tensor_dense_view *inputs,
                            const tr_math_budget *budget, tr_math_i128 *out, tr_tensor_exec *rec,
                            tr_math_refusal *refusal);
#ifdef __cplusplus
}
#endif
#endif
