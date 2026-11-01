/* FND-006: fused multi-output basis reduction. Several basis states over one data scope (math_scope.h), moment banks
 * of different bases (math_bank.h) and histograms (math_dist.h), each with its own predicate and grouping, accumulated
 * in ONE traversal of the scope's rows on the CPU (math_reduce.c) or on a CUDA device (math_reduce_gpu.cu); and the
 * competing plan that runs the independent reductions one by one, both costed and selectable.
 * SPDX-License-Identifier: Apache-2.0
 *
 * ---- Request (tr_reduce_request) ---------------------------------------------------------------------------------
 * Over a base scope S (selected rows, ncols exact logical columns with validity, its dense groups):
 *   conditions  up to TR_REDUCE_MAX_CONDITIONS BOOL-001 row predicates (tr_bool_condition, math_bool.h) over the
 *               columns of S, SQL three-valued;
 *   nodes       a predicate DAG of tr_bool_node (CONST, VAR a = condition a, NOT, AND, OR, XOR, Kleene logic) shared
 *               by every output: node i refers to nodes < i only; an output's filter is one node (or -1: none);
 *   groupings   up to TR_REDUCE_MAX_GROUPINGS: the scope's own grouping (scope NULL), or the groups of another scope
 *               of exactly the same rows (same source identity and version, row count and selection bits: an exact
 *               alignment; array scopes are aligned by position), optionally followed by a map of those groups onto
 *               ngroups coarse groups (a rollup; map entries < ngroups);
 *   outputs     up to TR_REDUCE_MAX_OUTPUTS, each a BANK (a FND-004 moment basis: exponent rows first .. first + nterms
 *               - 1 of the request's matrix, total degree <= 4, mode TR_BANK_SQL or TR_BANK_JOINT; nterms 0 is a row
 *               count) or a HIST (a DIST-001 histogram spec over 1..3 columns of S and an optional weight column:
 *               dense over the base scope's domain [min, max] (+ a NULL slot iff S has a selected NULL) or an
 *               explicit domain, or sparse), with its filter and its grouping.
 * Semantics: output o is exactly the independent reduction of the derived scope S_o: the rows of S where o's filter
 * is TRUE (UNKNOWN and FALSE rows do not count: SQL WHERE), grouped by o's grouping: tr_bank_build of S_o over o's
 * basis in o's mode, or tr_dist_hist_build of S_o (a dense output: over o's resolved domain, given explicitly). Every
 * output keeps its own bag multiplicity rows[g], its own validity counts (per term, or per cell and weight) and its own
 * group presence (g present iff rows[g] > 0; an absent group has no rows, an all-NULL present group has rows and zero
 * counts). An empty selection, a never-TRUE filter or an empty group gives zeros, never an error.
 *
 * ---- The fused plan -------------------------------------------------------------------------------------------------
 * One traversal of the selection words. Per 64-row word: every condition read by the scheduled nodes is evaluated once
 * (its value and valid words), every node reachable from a filter once (tr_bool_apply), each output's mask is the
 * selection AND its filter's value word; per selected row of the union of the masks: the presence of every column
 * once, the group id of each grouping once, and each distinct monomial ("product") of all bank bases at most once,
 * when the first output needing it on that row is valid for it (a stamp per product); that one value feeds every
 * output term naming the product. Bank terms accumulate count and an exact 128-bit sum, histogram cells their count,
 * weight count and exact 128-bit weight sum. Shared: the selection, the decoded columns and validity, the predicate
 * words, the group ids, the products; per output: everything it accumulates. Structurally identical conditions and
 * nodes are merged at compile time (hash consing), so equal predicates are evaluated once.
 * The repeated plan (the preserved competing plan): for each output in turn, its predicate by BOOL-001 masks
 * (tr_bool_masks_build) and a word evaluation of its filter, its derived scope S_o (tr_scope_from_arrays: the columns
 * it reads, its grouping's ids, the TRUE rows), then tr_bank_build or tr_dist_hist_build of S_o (the CUDA variant
 * runs tr_mgpu_bank_build / tr_mgpu_dist_hist_build on S_o from the Python layer, then tr_reduce_stage_put_*).
 * Both plans give identical states (the tests compare them with independent oracles); a sparse histogram of the
 * repeated plan may use DIST-001's SORT strategy where the fused plan's counting scratch is over budget.
 *
 * ---- Exactness and atomicity -------------------------------------------------------------------------------------
 * The compile step proves, for every bank output term, N * prod max(|min_c|, |max_c|, 1)^e_c <= 2^127 - 1 (N: the
 * rows the term can be valid on in S, an upper bound for every S_o), and for every weighted histogram selected rows x
 * max |w| <= 2^127 - 1, from the base scope's extremes: otherwise the whole request is refused (TR_MATH_OVERFLOW,
 * *failed = the first refused output) before any state is allocated or computed. A run checks its plan's budget
 * for the whole request (all outputs at once) before allocating states. Partial failures are atomic: a run either
 * returns a result holding every output, or nothing (the stage and every finished output are freed): a sparse output
 * whose final state is over
 * budget (known after counting), a repeated output whose own reduction refuses (its exact bound over S_o, its own
 * budget), a device or adoption failure. Integer sums only: the CPU, the emulation and the device agree bit for bit in
 * any thread order. Adopting a staged result checks its consistency (rows <= selected, count <= rows, a zero sum
 * without valid rows, |sum| <= count x the product bound, the constant term, histogram rows = the sum of their cells,
 * weight counts and bounds) and refuses (TR_MATH_INVALID) otherwise.
 *
 * ---- Plan choice (FND-008 conventions) ------------------------------------------------------------------------------
 * tr_reduce_choose costs the candidates (fused / repeated) x (CPU / CUDA) with FND-008's components (tr_reg_candidate,
 * math_registry.h): words, scan (row-columns read), chain (per-row predicate, product and accumulation steps),
 * construct (state elements), fusion (<= 0: the batch - 1 traversals of the shared components the fused plan saves;
 * the fused candidate counts its shared components for batch = outputs unfused passes, as FND-008 does), materialize
 * (output bytes), transfer (device bytes), fixed (native calls, kernel launches) times hand-set relative coefficients
 * (cost basis "estimate", not calibrated; a table [2 plans][2 devices][12 components] may override them). It refuses
 * candidates over budget, chooses the cheapest admissible one (the first on a tie), and keeps every candidate's
 * amounts, costs, status and reason (TR_PLANNER_S_* / TR_PLANNER_R_*). A force (plan, device: "fused", "repeated",
 * "fused@cuda", "repeated@cpu", ...; the environment variable TORBLES_MATH_REDUCE_FORCE) runs the cheapest admissible
 * match or refuses with the first match's code (TR_MATH_UNSUPPORTED when nothing matches): never replaced.
 *
 * ---- Budgets (math_budgets.h), checked before allocation --------------------------------------------------------
 *   variables = distinct columns read; degree = the largest term degree; support = the bank terms of all outputs plus
 *   the largest histogram's cells per group; circuit = conditions + scheduled nodes; output = outputs; states = the
 *   state elements of all outputs (rows per group, term counts per group, histogram cells per group); bytes = the
 *   plan, the stage (all states, the counting scratch of sparse histograms), the CPU scratch (condition and node words,
 *   masks, product values and stamps) and the result; on CUDA (tr_reduce_device_need) the device copies (view, the
 *   selection, the columns read and their validity, the group ids and maps, the compiled tables, the states) and the
 *   per-work-item scratch. The repeated plan's need: the stage, plus the largest single output's derived scope, masks
 *   and reduction, every resource bounded from the derived scope's shape (tr_bank_device_need / tr_dist_device_need
 *   on CUDA). A plan's need is checked before anything is allocated and bounds every later check, except the occupied
 *   cells of sparse histograms: checked when counted (fused: at the end; repeated: as each state is put, and at the
 *   end).
 * Representation limits (TR_MATH_UNSUPPORTED): more than 1024 outputs, 16 groupings, 64 conditions, 65536 nodes,
 * 65536 terms, 4096 distinct products, a bank term of degree > 4, a histogram outside DIST-001's limits.
 *
 * Errors: TR_MATH_INVALID (malformed input, misaligned groupings, a map entry out of range), _MEMORY, _OVERFLOW,
 * _BUDGET (with the tr_math_refusal), _UNSUPPORTED, _DOMAIN_ERROR (an explicit domain that does not hold the values),
 * TR_MATH_STALE (the Python layer checks freshness), TR_MATH_DEVICE. Objects are immutable once built (a stage is the
 * one mutable object: one owner, calls serialized); constructors return NULL and set *error, inputs unchanged. A plan
 * reads its base scope and the grouping scopes: they must outlive the plan and its stages. */
#ifndef TORBLES_MATH_REDUCE_H
#define TORBLES_MATH_REDUCE_H
#include <stdint.h>
#include <string.h>
#include "math_num.h"
#include "math_budgets.h"
#include "math_scope.h"
#include "math_bank.h"
#include "math_bool.h"
#include "math_dist.h"
#include "math_planner.h"
#ifdef __cplusplus
extern "C" {
#endif

#define TR_REDUCE_ABI 1
#define TR_REDUCE_MAX_OUTPUTS 1024
#define TR_REDUCE_MAX_GROUPINGS 16
#define TR_REDUCE_MAX_CONDITIONS 64
#define TR_REDUCE_MAX_NODES 65536
#define TR_REDUCE_MAX_TERMS 65536
#define TR_REDUCE_MAX_PRODUCTS 4096
#define TR_REDUCE_CANDIDATES 4
enum { TR_REDUCE_BANK = 1, TR_REDUCE_HIST = 2 };                         /* output kinds */
enum { TR_REDUCE_FUSED = 1, TR_REDUCE_REPEATED = 2 };                    /* plans (a candidate's representation) */
enum { TR_REDUCE_ON_CPU = 1, TR_REDUCE_ON_CUDA = 2 };                    /* = TR_REG_ON_* */
enum { TR_REDUCE_NEED_FUSED = 0, TR_REDUCE_NEED_FUSED_DEVICE = 1,        /* tr_reduce_need targets */
       TR_REDUCE_NEED_REPEATED = 2, TR_REDUCE_NEED_REPEATED_DEVICE = 3,
       TR_REDUCE_NEED_FUSED_KERNEL = 4 };                                  /* what tr_mgpu_reduce_fused checks */

/* ---------------------------------------------------------------- the request */
typedef struct {
    const tr_scope *scope;     /* NULL: the base scope's own grouping; else a scope of exactly the same rows */
    const uint32_t *map;       /* NULL, or one coarse id (< ngroups) per group of that grouping */
    uint64_t ngroups;          /* with map: the coarse groups (>= 1); without: 0 */
} tr_reduce_grouping;          /* 24 bytes */

typedef struct {
    int32_t kind;              /* TR_REDUCE_BANK / TR_REDUCE_HIST */
    int32_t filter;            /* the predicate node whose TRUE rows count, or -1: every selected row */
    int32_t grouping;          /* index of its grouping */
    int32_t mode;              /* BANK: TR_BANK_SQL / TR_BANK_JOINT; HIST: 0 */
    int32_t first, nterms;     /* BANK: exponent rows first .. first + nterms - 1; HIST: 0, 0 */
    int32_t reserved[2];       /* 0 */
    tr_dist_spec hist;         /* HIST: columns of the base scope (strategy ignored); BANK: zeros */
} tr_reduce_output;            /* 128 bytes */

typedef struct {
    int32_t nconds, nnodes, ngroupings, noutputs;
    int32_t nexps, reserved;          /* rows of the exponent matrix (each the scope's ncols wide) */
    const tr_bool_condition *conds;   /* nconds */
    const tr_bool_node *nodes;        /* nnodes */
    const tr_reduce_grouping *groupings;
    const tr_reduce_output *outputs;
    const uint8_t *exps;              /* nexps x ncols, row-major */
} tr_reduce_request;                  /* 64 bytes */

/* ---------------------------------------------------------------- the compiled plan (shared with the kernels) */
typedef struct { uint32_t product, need; } tr_reduce_term;   /* a bank term: its product, the columns it needs */

typedef struct {
    int32_t kind, filter, grouping, mode;    /* filter: a compiled node, or -1 */
    int32_t first, nterms;                   /* BANK: compiled terms first .. first + nterms - 1 */
    int32_t hist, reserved;                  /* HIST: its plan; else -1 */
    uint32_t joint, reserved2;               /* JOINT: the columns of the basis */
    uint64_t ngroups;                        /* groups of its grouping */
    uint64_t cells;                          /* HIST: counted cells per group (dense domain, or the sparse scratch) */
    uint64_t rows_at, count_at, wcount_at;   /* offsets in the uint64 arena: rows[G]; count[G x (nterms | cells)];
                                                weighted HIST: wcount[G x cells] */
    uint64_t sum_at;                         /* offset in the 128-bit arena: BANK sum[G x nterms]; weighted HIST wsum */
} tr_reduce_out;                             /* 88 bytes */

typedef struct {
    const uint32_t *ids;       /* per-row ids of the source grouping, or NULL (one source group: every id 0) */
    const uint32_t *map;       /* NULL (identity) or source_groups coarse ids */
    uint64_t source_groups, ngroups;
} tr_reduce_group_view;        /* 32 bytes */

/* The plain-C view of a compiled plan: what the CPU loop and the CUDA kernel read (pointers into the plan and its
 * scopes; the CUDA host copies it with every pointer swapped for a device copy). */
typedef struct {
    tr_scope_view scope;                     /* the base scope */
    const tr_bool_condition *conds;          /* nconds (merged) */
    const tr_bool_node *nodes;               /* nnodes (merged, compiled indices) */
    const uint32_t *schedule;                /* nsched nodes evaluated per word, increasing */
    const uint32_t *condlist;                /* ncondlist conditions the schedule reads */
    const tr_bank_term *products;            /* nproducts distinct monomials */
    const tr_reduce_term *terms;             /* nterms compiled bank terms */
    const tr_reduce_out *outs;               /* noutputs */
    const tr_dist_plan *hplans;              /* nhist histogram plans */
    tr_reduce_group_view groups[TR_REDUCE_MAX_GROUPINGS];
    int32_t nconds, nnodes, nsched, ncondlist;
    int32_t nproducts, nterms, noutputs, nhist;
    int32_t ngroupings, degree;
    uint32_t used, nullable;                 /* columns read; those of them with validity flags */
    uint64_t arena64, arena128;              /* entries of the arenas (uint64 counts, 128-bit sums) */
} tr_reduce_view;                            /* 2736 bytes */

/* the arenas a run writes (host arrays: arena64 uint64 entries, arena128 two-word sums) */
typedef struct {
    uint64_t *a64;
    tr_math_i128 *a128;
    uint64_t n64, n128;
} tr_reduce_arena;                           /* 32 bytes */

/* what a run did (the result's record; the CUDA entry fills the device part) */
typedef struct {
    uint64_t scans;            /* traversals of the scope's rows (fused 1; repeated per output: masks, derive, build) */
    uint64_t kernels;          /* CUDA kernel launches */
    uint64_t native_calls;     /* native reductions and passes run */
    uint64_t decoded_bytes;    /* bytes of decoded columns, validity, group ids, maps and selection words read */
    uint64_t workspace_bytes;  /* scratch: CPU words, masks, products; device copies and per-item scratch */
    uint64_t transfer_bytes;   /* host <-> device copies */
    uint64_t products;         /* product evaluations (fused: each distinct monomial at most once per row) */
    uint64_t accumulations;    /* term and cell accumulations */
    uint64_t rows_visited, words_visited;
    uint64_t items, state_bytes;
    int32_t plan, device;      /* TR_REDUCE_FUSED / _REPEATED; TR_REDUCE_ON_* */
} tr_reduce_stats;             /* 104 bytes */

/* plan-level description */
typedef struct {
    int32_t noutputs, nconds, nnodes, nsched;     /* merged conditions and nodes, nodes evaluated per word */
    int32_t nproducts, nterms, ngroupings, nhist; /* distinct products, bank terms of all outputs */
    int32_t merged_conds, merged_nodes;           /* request conditions / nodes merged into equal ones */
    uint32_t used, nullable;
    uint64_t rows, selected, words;
    uint64_t arena64, arena128;                   /* fused arenas */
    uint64_t states, bytes;                       /* state elements of all outputs; bytes the plan owns */
    uint64_t term_rows;                           /* sum over outputs of their terms (the unshared product count) */
} tr_reduce_plan_info;                            /* 112 bytes */

/* one output of a result */
typedef struct {
    int32_t kind, mode, grouping, filter;
    int32_t nterms, k, repr, weighted;
    uint64_t ngroups, present, rows;   /* groups, present groups, rows over every group */
    uint64_t cells;                    /* BANK: nterms; dense HIST: cells per group; sparse: occupied cells */
    uint64_t state_bytes;
    int64_t lo[3];                     /* dense HIST: the domain */
    uint64_t span[3], dim[3];
    int32_t nullslot[3], scale[3];
    int32_t col[3], weight;            /* HIST: base scope columns */
    int32_t wscale, reserved;
} tr_reduce_output_info;               /* 192 bytes */

/* ---------------------------------------------------------------- plan choice */
typedef struct { int32_t plan, device, reserved[2]; } tr_reduce_force;   /* plan 0: none; device -1: any. 16 bytes */
typedef struct {
    int32_t abi, ncandidates, chosen, forced;      /* chosen: candidate index, -1 refused */
    int32_t reason, code, devices, module;         /* TR_PLANNER_R_*; the plan's code; module: registry id or 0 */
    tr_reduce_force force;
    tr_math_refusal refusal;                       /* the budget refusal of a refused plan */
    tr_reg_candidate candidates[TR_REDUCE_CANDIDATES];
} tr_reduce_evidence;
#define TR_REDUCE_COEFFICIENTS (2 * 2 * TR_COST_COMPONENTS)   /* [plan - 1][device - 1][component] */

typedef struct tr_reduce_plan_s tr_reduce_plan;
typedef struct tr_reduce_stage_s tr_reduce_stage;
typedef struct tr_reduce_result_s tr_reduce_result;

/* ---------------------------------------------------------------- shared helpers (CPU loop, CUDA kernels) */
/* the value and valid words of every condition the schedule reads (cw[2 i], cw[2 i + 1]) and of every scheduled node
 * (nw[2 i], nw[2 i + 1]) at word w */
TR_NUM_HD void tr_reduce_eval_word(const tr_reduce_view *v, uint64_t w, uint64_t *cw, uint64_t *nw) {
    for (int32_t i = 0; i < v->ncondlist; i++) {
        uint32_t c = v->condlist[i];
        tr_bool_condition_word(&v->scope, &v->conds[c], w, &cw[2 * (uint64_t)c], &cw[2 * (uint64_t)c + 1]);
    }
    for (int32_t s = 0; s < v->nsched; s++) {
        uint32_t i = v->schedule[s];
        const tr_bool_node *n = &v->nodes[i];
        uint64_t val, ok;
        if (n->op == TR_BOOL_CONST) tr_bool_const_word(n->a, &val, &ok);
        else if (n->op == TR_BOOL_VAR) { val = cw[2 * (uint64_t)n->a]; ok = cw[2 * (uint64_t)n->a + 1]; }
        else tr_bool_apply(n->op, nw[2 * (uint64_t)n->a], nw[2 * (uint64_t)n->a + 1],
                           n->b >= 0 ? nw[2 * (uint64_t)n->b] : 0, n->b >= 0 ? nw[2 * (uint64_t)n->b + 1] : 0, &val, &ok);
        nw[2 * (uint64_t)i] = val; nw[2 * (uint64_t)i + 1] = ok;
    }
}
/* the rows of word w output o counts: the selection AND its filter's TRUE bits */
TR_NUM_HD uint64_t tr_reduce_mask(const tr_reduce_view *v, const tr_reduce_out *o, uint64_t sel, const uint64_t *nw) {
    (void)v;
    return o->filter < 0 ? sel : sel & nw[2 * (uint64_t)o->filter];
}
/* bit c set: column c (read by the plan) is present at the row */
TR_NUM_HD uint32_t tr_reduce_presence(const tr_reduce_view *v, uint64_t row) {
    uint32_t pm = v->used, nl = v->nullable;
    while (nl) {
        int c = tr_bool_ctz((uint64_t)nl);
        nl &= nl - 1;
        if (!v->scope.col[c].valid[row]) pm &= ~(UINT32_C(1) << c);
    }
    return pm;
}
TR_NUM_HD uint32_t tr_reduce_gid(const tr_reduce_view *v, int32_t g, uint64_t row) {
    const tr_reduce_group_view *q = &v->groups[g];
    uint32_t id = q->ids ? q->ids[row] : 0;
    return q->map ? q->map[id] : id;
}
/* a bank term is valid at a row with presence pm */
TR_NUM_HD int tr_reduce_term_valid(const tr_reduce_out *o, const tr_reduce_term *t, uint32_t pm) {
    return o->mode == TR_BANK_JOINT ? (pm & o->joint) == o->joint : (pm & t->need) == t->need;
}

/* ---------------------------------------------------------------- the CUDA work split (host code of both libraries) */
#define TR_REDUCE_ITEMS (UINT64_C(1) << 16)        /* at most this many work items per launch */
#define TR_REDUCE_MIN_WORDS 4                       /* at least this many 64-row words per work item */
#define TR_REDUCE_SCRATCH (UINT64_C(1) << 26)       /* at most this many bytes of per-item device scratch */
/* bytes of one work item's device scratch: condition and node words, masks and live list, products and stamps,
 * per-output rows accumulators (count, group), per-term accumulators (count, sum) */
static inline uint64_t tr_reduce_item_bytes(const tr_reduce_view *v) {
    uint64_t b = tr_budget_mul((uint64_t)v->nconds, 16);
    b = tr_budget_add(b, tr_budget_mul((uint64_t)v->nnodes, 16));
    b = tr_budget_add(b, tr_budget_mul((uint64_t)v->noutputs, 8 + 8 + 16));
    b = tr_budget_add(b, tr_budget_mul((uint64_t)v->nproducts, 24));
    b = tr_budget_add(b, tr_budget_mul((uint64_t)v->nterms, 24));
    return tr_budget_add(b, 64);
}
static inline uint64_t tr_reduce_items(const tr_reduce_view *v) {
    uint64_t nw = (v->scope.rows + 63) / 64, per = tr_reduce_item_bytes(v);
    uint64_t s = (nw + TR_REDUCE_MIN_WORDS - 1) / TR_REDUCE_MIN_WORDS;
    if (s > TR_REDUCE_ITEMS) s = TR_REDUCE_ITEMS;
    uint64_t room = per ? TR_REDUCE_SCRATCH / per : s;
    if (s > room) s = room;
    return s ? s : 1;
}
/* the needs of tr_mgpu_reduce_fused: bytes = its device copies (the view, the selection, the columns read and their
 * validity, each distinct group-id array, the maps, the compiled tables, the arenas) and the per-item scratch;
 * states = the arena entries; the other resources as the CPU plan's (given in base) */
static inline void tr_reduce_device_need(const tr_reduce_view *v, const tr_math_need *base, tr_math_need *need) {
    *need = *base;
    uint64_t n = v->scope.rows, bytes = tr_budget_add(sizeof(tr_reduce_view), tr_budget_bitmap_bytes(n));
    for (int32_t c = 0; c < v->scope.ncols; c++) {
        if (!((v->used >> c) & 1)) continue;
        bytes = tr_budget_add(bytes, tr_budget_mul(n, v->scope.col[c].values ? 8 : 4));
        if (v->scope.col[c].valid) bytes = tr_budget_add(bytes, n);
    }
    for (int32_t g = 0; g < v->ngroupings; g++) {
        int seen = 0;
        for (int32_t h = 0; h < g; h++) seen |= v->groups[h].ids == v->groups[g].ids;
        if (v->groups[g].ids && !seen) bytes = tr_budget_add(bytes, tr_budget_mul(n, 4));
        if (v->groups[g].map) bytes = tr_budget_add(bytes, tr_budget_mul(v->groups[g].source_groups, 4));
    }
    bytes = tr_budget_add(bytes, tr_budget_mul((uint64_t)v->nconds, sizeof(tr_bool_condition)));
    bytes = tr_budget_add(bytes, tr_budget_mul((uint64_t)v->nnodes, sizeof(tr_bool_node) + 4));
    bytes = tr_budget_add(bytes, tr_budget_mul((uint64_t)v->nconds, 4));
    bytes = tr_budget_add(bytes, tr_budget_mul((uint64_t)v->nproducts, sizeof(tr_bank_term)));
    bytes = tr_budget_add(bytes, tr_budget_mul((uint64_t)v->nterms, sizeof(tr_reduce_term)));
    bytes = tr_budget_add(bytes, tr_budget_mul((uint64_t)v->noutputs, sizeof(tr_reduce_out)));
    bytes = tr_budget_add(bytes, tr_budget_mul((uint64_t)v->nhist, sizeof(tr_dist_plan)));
    bytes = tr_budget_add(bytes, tr_budget_add(tr_budget_mul(v->arena64, 8), tr_budget_mul(v->arena128, 16)));
    bytes = tr_budget_add(bytes, tr_budget_mul(tr_reduce_items(v), tr_reduce_item_bytes(v)));
    need->need[TR_BUDGET_BYTES] = bytes;
}

/* ---------------------------------------------------------------- libalg (math_reduce.c) */
/* Compiles and validates a request over `scope` (see the header comment): merges equal conditions and nodes,
 * resolves the groupings and histogram domains, collects the distinct products, lays out the arenas and proves every
 * bound. It checks the request's own size (circuit, output) before allocating and the plan's bytes after; each
 * plan's needs (tr_reduce_need) are checked when it is chosen or run, so a fused plan over budget leaves the repeated
 * plan selectable. failed (may be NULL): the output refused first (-1 when the request as a whole is). */
tr_reduce_plan *tr_reduce_compile(const tr_scope *scope, const tr_reduce_request *, const tr_math_budget *budget,
                                  tr_math_refusal *refusal, int32_t *failed, int32_t *error);
void tr_reduce_plan_free(tr_reduce_plan *);
int32_t tr_reduce_plan_get_info(const tr_reduce_plan *, tr_reduce_plan_info *);
int32_t tr_reduce_get_view(const tr_reduce_plan *, tr_reduce_view *);
/* the needs of a plan: TR_REDUCE_NEED_FUSED (CPU), _FUSED_DEVICE (the CUDA fused run: the device copies and scratch,
 * plus the host's plan, stage and result), _FUSED_KERNEL (what tr_mgpu_reduce_fused itself checks), _REPEATED (CPU),
 * _REPEATED_DEVICE (the per-output device builds: the largest one's device copies) */
int32_t tr_reduce_need(const tr_reduce_plan *, int32_t target, tr_math_need *);
/* the candidates (fused, repeated) x (devices: TR_REDUCE_ON_* mask), costed, budget-checked and chosen; force NULL
 * reads TORBLES_MATH_REDUCE_FORCE; coefficients NULL: the built-in estimates. Returns the plan's code. */
int32_t tr_reduce_choose(const tr_reduce_plan *, int32_t devices, const tr_math_budget *, const double *coefficients,
                         const tr_reduce_force *, tr_reduce_evidence *);
/* "fused", "repeated", optionally "@cpu" / "@cuda"; the first entry of a comma-separated list; NULL / "": no force */
int32_t tr_reduce_force_parse(const char *text, tr_reduce_force *);
/* the built-in coefficient table (TR_REDUCE_COEFFICIENTS doubles) */
int32_t tr_reduce_coefficients(double *out, int32_t capacity);
/* Runs a plan on the CPU (TR_REDUCE_FUSED or TR_REDUCE_REPEATED); atomic (all outputs or nothing). */
tr_reduce_result *tr_reduce_run(const tr_reduce_plan *, int32_t plan, const tr_math_budget *budget,
                                tr_math_refusal *refusal, int32_t *failed, int32_t *error);

/* staging (the CUDA paths): an empty mutable stage for a plan (TR_REDUCE_FUSED: the plan's arenas, zeroed;
 * TR_REDUCE_REPEATED: room for the outputs put one by one); the fused arenas for the kernel; finish adopts the staged
 * states after the consistency checks (device: TR_REDUCE_ON_*; stats: what the run did, or NULL) and frees the stage
 * in every case. */
tr_reduce_stage *tr_reduce_stage_create(const tr_reduce_plan *, int32_t plan, const tr_math_budget *budget,
                                        tr_math_refusal *refusal, int32_t *error);
int32_t tr_reduce_stage_arena(tr_reduce_stage *, tr_reduce_arena *);
/* the states of an independent reduction of output o's derived scope (a bank of tr_reduce_derive's scope over the
 * output's derived basis, or its histogram), copied into the stage; a sparse histogram's copy (its occupied cells,
 * known only now) is budgeted beside the stage and the plan (TR_MATH_BUDGET with the refusal) */
int32_t tr_reduce_stage_put_bank(tr_reduce_stage *, int32_t output, const tr_bank *);
int32_t tr_reduce_stage_put_hist(tr_reduce_stage *, int32_t output, const tr_dist_hist *, tr_math_refusal *refusal);
tr_reduce_result *tr_reduce_stage_finish(tr_reduce_stage *, int32_t device, const tr_reduce_stats *stats,
                                         const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *failed,
                                         int32_t *error);
void tr_reduce_stage_free(tr_reduce_stage *);
/* The derived scope S_o of output o (the repeated plan): the columns o reads (in increasing base order), its
 * grouping's ids, the rows of the base selection where its filter is TRUE (BOOL-001 masks); exps (capacity >= nterms x
 * columns, may be NULL) receives o's basis over the derived columns, spec (may be NULL) its histogram spec over them
 * (a dense output with its resolved domain made explicit); *ncols the derived columns. */
tr_scope *tr_reduce_derive(const tr_reduce_plan *, int32_t output, uint8_t *exps, uint64_t capacity, tr_dist_spec *spec,
                           int32_t *ncols, const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error);

/* results */
int32_t tr_reduce_result_get_stats(const tr_reduce_result *, tr_reduce_stats *);
int32_t tr_reduce_result_output(const tr_reduce_result *, int32_t output, tr_reduce_output_info *);
/* pointers into the result's states (read only, valid while it lives) */
int32_t tr_reduce_result_bank(const tr_reduce_result *, int32_t output, tr_bank_states *);
int32_t tr_reduce_result_hist(const tr_reduce_result *, int32_t output, tr_dist_states *);
/* 1 when both hold the same outputs with identical states (the record of the run is not compared) */
int32_t tr_reduce_result_equal(const tr_reduce_result *, const tr_reduce_result *);
void tr_reduce_result_free(tr_reduce_result *);

/* ---------------------------------------------------------------- the CUDA library (math_reduce_gpu.cu)
 * The fused pass on the device: view = tr_reduce_get_view (host pointers; copied to the device, budgeted with
 * tr_reduce_device_need against `base`, the plan's CPU need); out = tr_reduce_stage_arena of a TR_REDUCE_FUSED stage
 * (zeroed host arrays), overwritten with exactly the states of the CPU fused pass; stats: kernels, items, device and
 * transfer bytes. Returns 0 or a native code (TR_MATH_DEVICE when the device fails). */
int tr_mgpu_reduce_fused(const tr_reduce_view *view, const tr_math_need *base, const tr_math_budget *budget,
                         tr_reduce_arena *out, tr_reduce_stats *stats, tr_math_refusal *refusal);
/* The same pass with its arenas kept on the device for a later kernel (META-002's finalization): the states object
 * owns them (budgeted as above); _arena gives their device pointers, _copy copies them into host arrays of the same
 * sizes (a TR_REDUCE_FUSED stage's arena), _free releases them. NULL and *error on a refusal or a device failure. */
typedef struct tr_mgpu_reduce_states_s tr_mgpu_reduce_states;
tr_mgpu_reduce_states *tr_mgpu_reduce_fused_device(const tr_reduce_view *view, const tr_math_need *base,
                                                   const tr_math_budget *budget, tr_reduce_stats *stats,
                                                   tr_math_refusal *refusal, int *error);
int tr_mgpu_reduce_states_arena(const tr_mgpu_reduce_states *, tr_reduce_arena *device_out);
int tr_mgpu_reduce_states_copy(const tr_mgpu_reduce_states *, tr_reduce_arena *host_out);
void tr_mgpu_reduce_states_free(tr_mgpu_reduce_states *);
#ifdef __cplusplus
}
#endif
#endif
