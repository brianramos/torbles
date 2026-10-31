/* Search, representation and output budgets of the mathematical modules (FND-010).
 * SPDX-License-Identifier: Apache-2.0
 *
 * A capability bounds every request before it allocates anything. It states what the request needs
 * (tr_math_need: variables, degree, support terms, tensor elements, circuit nodes, iterations, output rows, states,
 * bytes), computed with the exact size formulas below, and compares that with its tr_math_budget. The comparison
 * visits the resources in the fixed order of the enum and refuses (TR_MATH_BUDGET) at the first need above its
 * limit, reporting that resource with its need and its limit: a refusal is deterministic and is decided before any
 * partial result exists. A size that does not fit 64 bits is the need TR_BUDGET_UNKNOWN (UINT64_MAX), above every
 * limit; no formula wraps.
 *
 * Output sizes that depend on the data (the distinct occupied cells of a grouping, say) are counted exactly by a
 * pre-pass (tr_math_budget_count_cells on the CPU, tr_mgpu_budgets_count_cells on a CUDA device) and checked before
 * the output is built. The pre-pass's own bitmap is budgeted first (states = possible cells, bytes = its memory).
 *
 * A representation choice is explicit and costed: tr_math_budget_choose takes an expanded and a factored
 * representation of the same result (their needs and relative costs) and reports which one runs and why. With the
 * fallback disallowed, an expanded representation over budget is refused, never replaced silently.
 *
 * The inline functions are shared by the engine library and the CUDA library (host code); math_budgets.c exports
 * them under tr_math_budget_* for other modules and the Python bindings. */
#ifndef TORBLES_MATH_BUDGETS_H
#define TORBLES_MATH_BUDGETS_H
#include <stdint.h>
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif

enum { TR_BUDGET_VARIABLES = 0,   /* free or observed variables of an expression or state */
       TR_BUDGET_DEGREE = 1,      /* total polynomial / jet / series degree */
       TR_BUDGET_SUPPORT = 2,     /* stored terms, support points or nonzeros */
       TR_BUDGET_TENSOR = 3,      /* dense tensor or matrix elements */
       TR_BUDGET_CIRCUIT = 4,     /* circuit / DAG / program nodes */
       TR_BUDGET_ITERATIONS = 5,  /* iterations of an iterative method or search steps */
       TR_BUDGET_OUTPUT = 6,      /* rows, groups or answers returned */
       TR_BUDGET_STATES = 7,      /* accumulator states (groups x cells) held during a scan */
       TR_BUDGET_BYTES = 8,       /* memory a request allocates */
       TR_BUDGET_RESOURCES = 9 };
#define TR_BUDGET_UNKNOWN UINT64_MAX
#define TR_MATH_BUDGET_CODE (-5)   /* TR_MATH_BUDGET of mathematics.h */

typedef struct { uint64_t limit[TR_BUDGET_RESOURCES]; } tr_math_budget;  /* a need <= its limit passes */
typedef struct { uint64_t need[TR_BUDGET_RESOURCES]; } tr_math_need;     /* 0 for a resource not used */
/* resource is -1 when nothing was refused; need/limit are those of the refused resource */
typedef struct { int32_t resource, refused; uint64_t need, limit; } tr_math_refusal;

enum { TR_REPR_EXPANDED = 0, TR_REPR_FACTORED = 1 };
enum { TR_CHOICE_CHEAPER = 0,    /* both fit: the cheaper runs (expanded on a tie) */
       TR_CHOICE_ONLY_FIT = 1 }; /* only the chosen one fits the budget */
typedef struct {
    int32_t choice, reason;
    tr_math_refusal expanded, factored;   /* each representation's own check */
    double expanded_cost, factored_cost;  /* relative work units as given, NOT time */
} tr_math_choice;

/* ------------------------------------------------------------------ saturating exact sizes */
static inline uint64_t tr_budget_add(uint64_t a, uint64_t b) { uint64_t r; return __builtin_add_overflow(a, b, &r) ? TR_BUDGET_UNKNOWN : r; }
static inline uint64_t tr_budget_mul(uint64_t a, uint64_t b) { uint64_t r; return __builtin_mul_overflow(a, b, &r) ? TR_BUDGET_UNKNOWN : r; }
/* C(n, k), exact, or UNKNOWN past 2^64 - 1 (each partial product C(n-k+i, i) is an integer) */
static inline uint64_t tr_budget_binomial(uint64_t n, uint64_t k) {
    if (k > n) return 0;
    if (k > n - k) k = n - k;
    unsigned __int128 r = 1;
    for (uint64_t i = 1; i <= k; i++) {
        r = r * (n - k + i) / i;
        if (r > UINT64_MAX) return TR_BUDGET_UNKNOWN;
    }
    return (uint64_t)r;
}
/* monomials in n variables of total degree <= d: C(n + d, d) */
static inline uint64_t tr_budget_monomials(uint32_t n, uint32_t d) { return tr_budget_binomial((uint64_t)n + d, d); }
/* monomials of total degree exactly d: C(n + d - 1, d) (1 when d == 0; 0 when n == 0 < d) */
static inline uint64_t tr_budget_homogeneous(uint32_t n, uint32_t d) {
    if (d == 0) return 1;
    if (n == 0) return 0;
    return tr_budget_binomial((uint64_t)n + d - 1, d);
}
/* elements of a dense tensor / cells of a dense joint domain: the product of the dimensions (1 for k == 0) */
static inline uint64_t tr_budget_product(const uint64_t *dims, uint32_t k) {
    uint64_t r = 1;
    for (uint32_t i = 0; i < k; i++) { if (dims[i] == 0) return 0; r = tr_budget_mul(r, dims[i]); }
    return r;
}
/* assignments of k Boolean variables: 2^k */
static inline uint64_t tr_budget_power2(uint32_t k) { return k >= 64 ? TR_BUDGET_UNKNOWN : (uint64_t)1 << k; }
/* bytes of a bitmap over n positions, in whole 64-bit words */
static inline uint64_t tr_budget_bitmap_bytes(uint64_t n) { return tr_budget_mul((n + 63) / 64, 8); }

/* ------------------------------------------------------------------ checks */
static inline void tr_budget_none(tr_math_refusal *r) { if (r) { r->resource = -1; r->refused = 0; r->need = 0; r->limit = 0; } }
/* 0, or TR_MATH_BUDGET at the first resource (enum order) whose need exceeds its limit */
static inline int32_t tr_budget_check(const tr_math_budget *b, const tr_math_need *n, tr_math_refusal *r) {
    tr_budget_none(r);
    for (int32_t i = 0; i < TR_BUDGET_RESOURCES; i++)
        if (n->need[i] > b->limit[i]) {
            if (r) { r->resource = i; r->refused = 1; r->need = n->need[i]; r->limit = b->limit[i]; }
            return TR_MATH_BUDGET_CODE;
        }
    return 0;
}
/* one resource against its limit (the other resources unchanged) */
static inline int32_t tr_budget_check_one(const tr_math_budget *b, int32_t resource, uint64_t need, tr_math_refusal *r) {
    tr_budget_none(r);
    if (resource < 0 || resource >= TR_BUDGET_RESOURCES) return -1;
    if (need > b->limit[resource]) {
        if (r) { r->resource = resource; r->refused = 1; r->need = need; r->limit = b->limit[resource]; }
        return TR_MATH_BUDGET_CODE;
    }
    return 0;
}
static inline void tr_budget_default(tr_math_budget *b) {
    b->limit[TR_BUDGET_VARIABLES] = 64;
    b->limit[TR_BUDGET_DEGREE] = 64;
    b->limit[TR_BUDGET_SUPPORT] = (uint64_t)1 << 20;
    b->limit[TR_BUDGET_TENSOR] = (uint64_t)1 << 26;
    b->limit[TR_BUDGET_CIRCUIT] = (uint64_t)1 << 20;
    b->limit[TR_BUDGET_ITERATIONS] = (uint64_t)1 << 20;
    b->limit[TR_BUDGET_OUTPUT] = (uint64_t)1 << 24;
    b->limit[TR_BUDGET_STATES] = (uint64_t)1 << 24;
    b->limit[TR_BUDGET_BYTES] = (uint64_t)1 << 30;
}
/* needs of the cell-count pre-pass: one bit per possible cell (states), its bitmap (bytes); with device_copies, the
 * device copies of the cell ids and the row bitmap as well */
static inline void tr_budget_count_cells_need(uint64_t n, uint64_t ncells, int has_rows, int device_copies, tr_math_need *need) {
    for (int i = 0; i < TR_BUDGET_RESOURCES; i++) need->need[i] = 0;
    need->need[TR_BUDGET_STATES] = ncells;
    uint64_t bytes = tr_budget_add(tr_budget_bitmap_bytes(ncells), 8);
    if (device_copies) {
        bytes = tr_budget_add(bytes, tr_budget_mul(n, 4));
        if (has_rows) bytes = tr_budget_add(bytes, tr_budget_bitmap_bytes(n));
    }
    need->need[TR_BUDGET_BYTES] = bytes;
}

/* ------------------------------------------------------------------ exported (math_budgets.c) */
void tr_math_budget_default(tr_math_budget *);
/* the fixed limits of the polynomial kernel of mathematics.h: 8 variables, degree 8, 128 terms */
void tr_math_budget_polynomial_kernel(tr_math_budget *);
uint64_t tr_math_budget_binomial(uint64_t n, uint64_t k);
uint64_t tr_math_budget_monomials(uint32_t variables, uint32_t degree);
uint64_t tr_math_budget_homogeneous(uint32_t variables, uint32_t degree);
uint64_t tr_math_budget_product(const uint64_t *dims, uint32_t k);
uint64_t tr_math_budget_power2(uint32_t k);
uint64_t tr_math_budget_add(uint64_t, uint64_t);
uint64_t tr_math_budget_mul(uint64_t, uint64_t);
int32_t tr_math_budget_check(const tr_math_budget *, const tr_math_need *, tr_math_refusal *);
/* Picks the representation that runs. allow_factored == 0 forbids the fallback: an expanded representation over
 * budget is then refused. Costs must be finite and >= 0. Returns 0 (out->choice set) or TR_MATH_BUDGET when no
 * allowed representation fits (out->expanded / out->factored say why). */
int32_t tr_math_budget_choose(const tr_math_budget *, const tr_math_need *expanded, double expanded_cost,
                              const tr_math_need *factored, double factored_cost, int32_t allow_factored,
                              tr_math_choice *out);
/* Exact number of distinct cell ids among the selected rows (rows: bitmap of n bits, NULL = every row). The bitmap of
 * ncells bits is budgeted (states, bytes) before it is allocated; the count is then checked against the output limit.
 * *distinct is written only on success; a refusal reports the exact count as its need. A cell id >= ncells is
 * TR_MATH_INVALID (nothing is returned). */
int32_t tr_math_budget_count_cells(const tr_math_budget *, const uint32_t *cells, const uint64_t *rows, uint64_t n,
                                   uint64_t ncells, uint64_t *distinct, tr_math_refusal *);
#ifdef __cplusplus
}
#endif
#endif
