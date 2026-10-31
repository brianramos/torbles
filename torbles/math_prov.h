/* PROV-001: factorized provenance of small relational queries (selection, union, join, group): every output tuple
 * carries a provenance expression over source-tuple variables in a shared, hash-consed DAG, interpreted in declared
 * semirings (Boolean existence B, bag multiplicity N, polynomial provenance N[X]); on the CPU (math_prov.c) and, for
 * the evaluation of a DAG under many valuations at once, in CUDA (math_prov_gpu.cu).
 * SPDX-License-Identifier: Apache-2.0
 *
 * ---- The store: one shared DAG ------------------------------------------------------------------------------------
 * A tr_prov_store is an append-only DAG of nodes (node i refers only to nodes < i): ZERO (node 0), ONE (node 1), VAR v
 * (source-tuple variable v: one per source row), PLUS (n >= 2 children: alternative derivations) and TIMES (n >= 2
 * children: joint use). Nodes are hash-consed: a node with the same operation and the same multiset of children is
 * one node, whoever asks for it (children sorted; repeats kept: x * x is x^2, a + a is 2a), so a derivation used by
 * many outputs is one node (factorized) and the DAG is never expanded. Constants simplify (PLUS drops ZERO, TIMES
 * drops ONE and is ZERO with a ZERO child; one child is that child); nothing is flattened, so shared nodes stay shared.
 * The store owns its nodes; relations refer to them by id. It is mutable only by appending (operators add nodes) and
 * owned by one caller (calls on one store are serialized by the caller; evaluations read immutable plans).
 * Annotation limits are fixed when the store is made (tr_math_budget: circuit = nodes, support = child entries,
 * bytes = the store's arrays); an operator computes the nodes and children it may add before adding any and refuses
 * with TR_MATH_BUDGET (need = the store's size after the operation's bound) when they do not fit: an annotation is
 * never truncated.
 *
 * ---- Relations: K-relations over the DAG ----------------------------------------------------------------------------
 * A tr_prov_rel is immutable: ncols int64 columns (logical values with a validity flag per value, a decimal scale per
 * column) and distinct tuples, each with its annotation (a node of its store). Tuples are compared with NULL equal to
 * NULL (as GROUP BY / DISTINCT do) and ordered lexicographically, values ascending, NULL last. Operators (each a new
 * relation; the semantics of K-relations, Green, Karvounarakis and Tannen 2007):
 *   source   each selected row of the data is a new variable; equal rows are one tuple annotated PLUS(their VARs)
 *   select   a conjunction of predicates (SQL WHERE): a tuple is kept iff every predicate is TRUE; a comparison with a
 *            NULL operand is UNKNOWN (dropped); IS NULL / IS NOT NULL are never UNKNOWN; annotations are kept
 *   union    UNION ALL: equal tuples of the two inputs merge, annotated PLUS(a, b)
 *   join     equi-join on column pairs (no pairs: the Cartesian product): every pair of matching tuples, columns of the
 *            left then the right, annotated TIMES(a, b); NULL never matches (SQL)
 *   group    GROUP BY columns: one tuple per group present in the input (a group derived from no tuple does not
 *            exist), annotated PLUS(its members); NULL keys form their own group; with no columns, the global group:
 *            one tuple that is always present (SQL's aggregate without GROUP BY), annotated PLUS(every tuple) or ZERO
 * NULL validity, the multiplicity of a tuple (its annotation) and group presence are separate: the multiplicity of a
 * present group under a valuation can be 0 only for the global group.
 *
 * ---- Interpretations (each result is in one declared semiring, TR_PROV_BOOL / _COUNT / _POLY) ----------------------
 *   BOOL   B = ({0, 1}, OR, AND) under a valuation nu: X -> {0, 1} (which sources are present): the tuple exists iff 1
 *   COUNT  N = (naturals, +, x) under nu: X -> N (multiplicities; deletions: 0): the number of derivations, exact in
 *          unsigned 128 bits (a valuation whose value leaves 128 bits has status TR_MATH_OVERFLOW and zero outputs)
 *   POLY   N[X]: the expanded derivation polynomial (monomials over the variables with natural coefficients), the
 *          universal semiring: B and N are its images under nu (x^k -> nu(x), Boolean: x^k -> x)
 * A valuation is a default value for every variable and sorted overrides (variable, value); the 1-valuation (every
 * variable 1) gives in N the bag multiplicity of the query and in B the existence. Witnesses (which sources suffice):
 * the minimal variable sets of the monomials (the Boolean provenance's minimal disjunctive form).
 *
 * ---- Evaluation plans -------------------------------------------------------------------------------------------
 * tr_prov_plan_build snapshots the cone of some output nodes (the nodes they reach, in increasing order, children
 * renumbered within the cone) as an immutable plan; tr_prov_evaluate evaluates it under count valuations in B or N
 * (the shared inline tr_prov_eval_*: one valuation at a time, a scratch value per cone node), the CUDA kernel the same
 * with one work item per valuation (valuations in chunks bounded by the device scratch). Both write the same bytes.
 *
 * ---- Budgets (math_budgets.h), checked before allocation ----------------------------------------------------------
 *   store: its limits (circuit, support, bytes); every operator: the store's nodes / children / bytes after the bound
 *     of the operation (source: rows VARs + runs PLUS; union: merged pairs; join: the matching pairs, counted first;
 *     group: the groups), and its own output = tuples, bytes = the result and the scratch
 *   plan: circuit = cone nodes, support = cone children, output = outputs, bytes
 *   evaluate: output = valuations x outputs, bytes = the scratch (CUDA: the device copies and the scratch chunk)
 *   expand (N[X]) / witnesses: circuit = cone nodes, support = the number of derivations D (the COUNT value at 1,
 *     an upper bound on the monomials; counted in a first pass), degree = the largest monomial degree, bytes = the
 *     memoized polynomials' bound (witnesses: tensor = D^2 for the minimization)
 * Over budget: TR_MATH_BUDGET with the tr_math_refusal; an annotation or expansion is never truncated.
 *
 * Errors: TR_MATH_INVALID (malformed input, a relation of another store), _MEMORY, _OVERFLOW (a coefficient past 64
 * bits, values past int64), _BUDGET, _DOMAIN_ERROR (columns of different decimal scales compared or merged, unions
 * of different arities), _UNSUPPORTED (more than 2^32 - 2 nodes or variables, more than TR_PROV_MAX_COLUMNS
 * columns), TR_MATH_DEVICE. Constructors return NULL and set *error. */
#ifndef TORBLES_MATH_PROV_H
#define TORBLES_MATH_PROV_H
#include <stdint.h>
#include "math_num.h"
#include "math_budgets.h"
#include "math_scope.h"
#ifdef __cplusplus
extern "C" {
#endif

#define TR_PROV_ABI 1
#define TR_PROV_MAX_COLUMNS 64
#define TR_PROV_MAX_IDS UINT32_C(4294967294)          /* nodes and variables are uint32 (UINT32_MAX: none) */
#define TR_PROV_DEVICE_SCRATCH (UINT64_C(1) << 26)    /* bytes of device scratch per launch chunk (at most) */
enum { TR_PROV_ZERO = 0, TR_PROV_ONE = 1, TR_PROV_VAR = 2, TR_PROV_PLUS = 3, TR_PROV_TIMES = 4 };   /* node ops */
enum { TR_PROV_BOOL = 1, TR_PROV_COUNT = 2, TR_PROV_POLY = 3 };                                     /* semirings */
enum { TR_PROV_EQ = 1, TR_PROV_NE = 2, TR_PROV_LT = 3, TR_PROV_LE = 4, TR_PROV_GT = 5, TR_PROV_GE = 6,
       TR_PROV_IS_NULL = 7, TR_PROV_IS_NOT_NULL = 8 };                                              /* predicates */
enum { TR_PROV_OP_SOURCE = 1, TR_PROV_OP_SELECT = 2, TR_PROV_OP_UNION = 3, TR_PROV_OP_JOIN = 4,
       TR_PROV_OP_GROUP = 5 };                                                                      /* relation ops */

/* a node (16 bytes): VAR: arg = the variable; PLUS / TIMES: n children at child[arg .. arg + n) */
typedef struct { uint32_t op, n; uint64_t arg; } tr_prov_node;

/* a predicate (24 bytes): column col op value (in the column's logical units) or op column col2 (col2 >= 0) */
typedef struct { int32_t op, col, col2, reserved; int64_t value; } tr_prov_pred;

/* valuations (40 bytes): valuation w gives every variable fallback[w] (NULL: 1) except the overrides
 * var[offset[w] .. offset[w + 1]) (strictly increasing), which get value[...] */
typedef struct {
    uint64_t count;
    const uint64_t *offset, *value, *fallback;
    const uint32_t *var;
} tr_prov_valuations;

/* an evaluation plan as the CPU and the kernels read it (56 bytes) */
typedef struct {
    uint64_t nodes, nchild, noutputs, nvars;  /* cone nodes, their children, outputs, variables of the store */
    const tr_prov_node *node;                 /* [nodes]: PLUS / TIMES arg = offset into child (cone-local) */
    const uint32_t *child;                    /* [nchild] cone-local node indices, each below its parent's */
    const uint32_t *output;                   /* [noutputs] cone-local indices of the outputs */
} tr_prov_plan_view;

/* descriptions */
typedef struct {
    uint64_t nodes, children, variables, sources;
    uint64_t shared;           /* nodes with at least two parents: derivations reused (factorization) */
    uint64_t hits;             /* node requests answered by an existing node (hash-consing) */
    uint64_t bytes;            /* bytes of the store's arrays */
    uint64_t limit_nodes, limit_children, limit_bytes;
} tr_prov_store_info;          /* 80 bytes */
typedef struct {
    uint64_t tuples, bytes, new_nodes, store_nodes;   /* tuples; bytes owned; nodes its operation added; store size */
    int32_t ncols, global, op, reserved;              /* global: the global group (always present) */
} tr_prov_rel_info;            /* 48 bytes */
typedef struct {
    uint64_t terms, pairs, derivations;  /* monomials, (variable, exponent) pairs, sum of the coefficients */
    uint32_t degree, semiring;           /* the largest total degree; TR_PROV_POLY (witnesses: TR_PROV_BOOL) */
} tr_prov_poly_info;           /* 32 bytes */
typedef struct {
    uint64_t count, noutputs, overflowed;   /* valuations, outputs per valuation, valuations past 128 bits */
    uint64_t chunk;                         /* valuations per launch (CPU: 1) */
    int32_t semiring, reserved;
} tr_prov_eval_info;           /* 40 bytes */

/* ---------------------------------------------------------------- the shared evaluators (CPU and CUDA) */
/* the value of variable x under valuation w */
TR_NUM_HD uint64_t tr_prov_var_value(const tr_prov_valuations *v, uint64_t w, uint32_t x) {
    uint64_t lo = v->offset[w], hi = v->offset[w + 1];
    while (lo < hi) {
        uint64_t m = lo + (hi - lo) / 2;
        if (v->var[m] == x) return v->value[m];
        if (v->var[m] < x) lo = m + 1; else hi = m;
    }
    return v->fallback ? v->fallback[w] : 1;
}
/* N: the plan under valuation w into s[nodes] (unsigned 128 bits) and its outputs out[2 noutputs] (lo, hi words);
 * 0, or -3 (TR_MATH_OVERFLOW: outputs zeroed) when a value leaves 128 bits */
TR_NUM_HD int32_t tr_prov_eval_count(const tr_prov_plan_view *p, const tr_prov_valuations *v, uint64_t w,
                                     unsigned __int128 *s, uint64_t *out) {
    const unsigned __int128 top = ~(unsigned __int128)0;
    int32_t status = 0;
    for (uint64_t i = 0; i < p->nodes && !status; i++) {
        const tr_prov_node nd = p->node[i];
        unsigned __int128 x;
        if (nd.op == TR_PROV_ZERO) x = 0;
        else if (nd.op == TR_PROV_ONE) x = 1;
        else if (nd.op == TR_PROV_VAR) x = tr_prov_var_value(v, w, (uint32_t)nd.arg);
        else if (nd.op == TR_PROV_PLUS) {
            x = 0;
            for (uint32_t c = 0; c < nd.n; c++) {
                unsigned __int128 y = s[p->child[nd.arg + c]];
                if (y > top - x) { status = -3; break; }
                x += y;
            }
        } else {
            x = 1;
            for (uint32_t c = 0; c < nd.n && x; c++) {
                unsigned __int128 y = s[p->child[nd.arg + c]];
                if (y && x > top / y) { status = -3; break; }
                x *= y;
            }
            if (!x) x = 0;
        }
        s[i] = x;
    }
    for (uint64_t j = 0; j < p->noutputs; j++) {
        unsigned __int128 x = status ? 0 : s[p->output[j]];
        out[2 * j] = (uint64_t)x;
        out[2 * j + 1] = (uint64_t)(x >> 64);
    }
    return status;
}
/* B: the plan under valuation w (a variable is present iff its value is nonzero) into s[nodes], outputs out[noutputs] */
TR_NUM_HD void tr_prov_eval_bool(const tr_prov_plan_view *p, const tr_prov_valuations *v, uint64_t w, uint8_t *s,
                                 uint8_t *out) {
    for (uint64_t i = 0; i < p->nodes; i++) {
        const tr_prov_node nd = p->node[i];
        uint8_t x;
        if (nd.op == TR_PROV_ZERO) x = 0;
        else if (nd.op == TR_PROV_ONE) x = 1;
        else if (nd.op == TR_PROV_VAR) x = tr_prov_var_value(v, w, (uint32_t)nd.arg) != 0;
        else if (nd.op == TR_PROV_PLUS) {
            x = 0;
            for (uint32_t c = 0; c < nd.n && !x; c++) x = s[p->child[nd.arg + c]];
        } else {
            x = 1;
            for (uint32_t c = 0; c < nd.n && x; c++) x = s[p->child[nd.arg + c]];
        }
        s[i] = x;
    }
    for (uint64_t j = 0; j < p->noutputs; j++) out[j] = s[p->output[j]];
}
/* host: 0 when the valuations are well formed for nvars variables (offsets nondecreasing, overrides strictly
 * increasing and < nvars), else TR_MATH_INVALID */
static inline int32_t tr_prov_valuations_check(const tr_prov_valuations *v, uint64_t nvars) {
    if (!v || (v->count && !v->offset)) return -1;
    if (!v->count) return 0;
    if (v->offset[0] != 0) return -1;
    for (uint64_t w = 0; w < v->count; w++) {
        if (v->offset[w + 1] < v->offset[w]) return -1;
        if (v->offset[w + 1] > v->offset[w] && (!v->var || !v->value)) return -1;
        for (uint64_t i = v->offset[w]; i < v->offset[w + 1]; i++)
            if ((uint64_t)v->var[i] >= nvars || (i > v->offset[w] && v->var[i] <= v->var[i - 1])) return -1;
    }
    return 0;
}
/* host: 0 when a plan view is well formed (children below their parent and within the child array, VAR variables
 * below nvars, outputs within the cone), else TR_MATH_INVALID */
static inline int32_t tr_prov_plan_view_check(const tr_prov_plan_view *p) {
    if (!p || (p->nodes && !p->node) || (p->nchild && !p->child) || (p->noutputs && !p->output)) return -1;
    for (uint64_t i = 0; i < p->nodes; i++) {
        const tr_prov_node *nd = &p->node[i];
        if (nd->op == TR_PROV_VAR) { if (nd->arg >= p->nvars || nd->n) return -1; continue; }
        if (nd->op == TR_PROV_ZERO || nd->op == TR_PROV_ONE) { if (nd->n) return -1; continue; }
        if ((nd->op != TR_PROV_PLUS && nd->op != TR_PROV_TIMES) || nd->arg > p->nchild || nd->n > p->nchild - nd->arg)
            return -1;
        for (uint32_t c = 0; c < nd->n; c++) if ((uint64_t)p->child[nd->arg + c] >= i) return -1;
    }
    for (uint64_t j = 0; j < p->noutputs; j++) if ((uint64_t)p->output[j] >= p->nodes) return -1;
    return 0;
}
/* host: the bytes of one valuation chunk's scratch, and the bytes the device copies (plan, valuations, outputs) */
static inline uint64_t tr_prov_scratch_width(int32_t semiring) { return semiring == TR_PROV_COUNT ? 16 : 1; }
static inline uint64_t tr_prov_device_bytes(const tr_prov_plan_view *p, const tr_prov_valuations *v, int32_t semiring,
                                            uint64_t chunk) {
    uint64_t overrides = v->count ? v->offset[v->count] : 0, width = tr_prov_scratch_width(semiring);
    uint64_t b = tr_budget_add(tr_budget_mul(p->nodes, 16), tr_budget_mul(tr_budget_add(p->nchild, p->noutputs), 4));
    b = tr_budget_add(b, tr_budget_mul(tr_budget_add(v->count, 1), 8));
    b = tr_budget_add(b, tr_budget_mul(overrides, 12));
    if (v->fallback) b = tr_budget_add(b, tr_budget_mul(v->count, 8));
    b = tr_budget_add(b, tr_budget_mul(tr_budget_mul(v->count, p->noutputs), semiring == TR_PROV_COUNT ? 16 : 1));
    b = tr_budget_add(b, tr_budget_mul(v->count, 4));
    return tr_budget_add(b, tr_budget_mul(tr_budget_mul(chunk, p->nodes), width));
}
/* host: valuations per device launch so that the scratch stays within TR_PROV_DEVICE_SCRATCH (at least 1) */
static inline uint64_t tr_prov_device_chunk(const tr_prov_plan_view *p, uint64_t count, int32_t semiring) {
    uint64_t per = tr_budget_mul(p->nodes ? p->nodes : 1, tr_prov_scratch_width(semiring));
    uint64_t c = per >= TR_PROV_DEVICE_SCRATCH ? 1 : TR_PROV_DEVICE_SCRATCH / per;
    return count < c ? (count ? count : 1) : c;
}

/* ---------------------------------------------------------------- objects (libalg) */
typedef struct tr_prov_store_s tr_prov_store;
typedef struct tr_prov_rel_s tr_prov_rel;
typedef struct tr_prov_plan_s tr_prov_plan;
typedef struct tr_prov_poly_s tr_prov_poly;

/* a store whose annotation limits are limits->limit[CIRCUIT] nodes, [SUPPORT] child entries, [BYTES] bytes (NULL: the
 * defaults of math_budgets.h) */
tr_prov_store *tr_prov_store_create(const tr_math_budget *limits, tr_math_refusal *, int32_t *error);
int32_t tr_prov_store_get_info(const tr_prov_store *, tr_prov_store_info *);
/* nodes [first, first + n) (their children: tr_prov_store_children) */
int32_t tr_prov_store_nodes(const tr_prov_store *, uint64_t first, uint64_t n, tr_prov_node *out);
int32_t tr_prov_store_children(const tr_prov_store *, uint64_t first, uint64_t n, uint32_t *out);
/* the source and the row of variables [first, first + n) */
int32_t tr_prov_store_variables(const tr_prov_store *, uint64_t first, uint64_t n, uint32_t *source, uint64_t *row);
/* parents of node id within the store (how many nodes use it) */
int32_t tr_prov_store_parents(const tr_prov_store *, uint32_t node, uint64_t *parents);
void tr_prov_store_free(tr_prov_store *);

/* a source relation of n rows (selection rows: n bits, NULL = every row) over ncols columns (tr_scope_array: values,
 * validity, scale); its variables are consecutive from *first_var (source id: *source) */
tr_prov_rel *tr_prov_source(tr_prov_store *, uint64_t n, int32_t ncols, const tr_scope_array *columns,
                            const uint64_t *rows, uint32_t *first_var, uint32_t *source, const tr_math_budget *,
                            tr_math_refusal *, int32_t *error);
/* the same from a FND-004 scope: its selected rows and the given scope columns (exact logical values) */
tr_prov_rel *tr_prov_source_scope(tr_prov_store *, const tr_scope *, const int32_t *cols, int32_t ncols,
                                  uint32_t *first_var, uint32_t *source, const tr_math_budget *, tr_math_refusal *,
                                  int32_t *error);
tr_prov_rel *tr_prov_select(tr_prov_store *, const tr_prov_rel *, const tr_prov_pred *preds, uint32_t npreds,
                            const tr_math_budget *, tr_math_refusal *, int32_t *error);
tr_prov_rel *tr_prov_union(tr_prov_store *, const tr_prov_rel *, const tr_prov_rel *, const tr_math_budget *,
                           tr_math_refusal *, int32_t *error);
tr_prov_rel *tr_prov_join(tr_prov_store *, const tr_prov_rel *, const tr_prov_rel *, const int32_t *lcols,
                          const int32_t *rcols, uint32_t nkeys, const tr_math_budget *, tr_math_refusal *,
                          int32_t *error);
tr_prov_rel *tr_prov_group(tr_prov_store *, const tr_prov_rel *, const int32_t *cols, int32_t ncols,
                           const tr_math_budget *, tr_math_refusal *, int32_t *error);
/* what an operator checks (its own needs and the store's after its bound): op = TR_PROV_OP_*, b / cols as the operator */
int32_t tr_prov_need(const tr_prov_store *, int32_t op, const tr_prov_rel *a, const tr_prov_rel *b, const int32_t *lcols,
                     const int32_t *rcols, uint32_t nkeys, tr_math_need *call, tr_math_need *store);
int32_t tr_prov_rel_get_info(const tr_prov_rel *, tr_prov_rel_info *);
/* the tuples (values [tuples x ncols], validity, annotations [tuples]) and the scales [ncols] */
int32_t tr_prov_rel_get(const tr_prov_rel *, int64_t *values, uint8_t *valid, uint32_t *ann, int32_t *scales,
                        uint64_t capacity);
void tr_prov_rel_free(tr_prov_rel *);

/* the cone of n output nodes as an immutable plan */
tr_prov_plan *tr_prov_plan_build(const tr_prov_store *, const uint32_t *nodes, uint64_t n, const tr_math_budget *,
                                 tr_math_refusal *, int32_t *error);
int32_t tr_prov_plan_get_view(const tr_prov_plan *, tr_prov_plan_view *);
void tr_prov_plan_free(tr_prov_plan *);
/* B (out: uint8 [count x noutputs]) or N (out: uint64 [count x noutputs x 2], status [count]) under the valuations */
int32_t tr_prov_evaluate(const tr_prov_plan *, int32_t semiring, const tr_prov_valuations *, void *out,
                         int32_t *status, tr_prov_eval_info *info, const tr_math_budget *, tr_math_refusal *);
/* the expanded N[X] polynomial of a node / its minimal witnesses (B), with their needs */
tr_prov_poly *tr_prov_expand(const tr_prov_store *, uint32_t node, const tr_math_budget *, tr_math_refusal *,
                             int32_t *error);
tr_prov_poly *tr_prov_witnesses(const tr_prov_store *, uint32_t node, const tr_math_budget *, tr_math_refusal *,
                                int32_t *error);
int32_t tr_prov_expand_need(const tr_prov_store *, uint32_t node, int32_t witnesses, tr_math_need *need);
int32_t tr_prov_poly_get_info(const tr_prov_poly *, tr_prov_poly_info *);
/* coefficients [terms], offsets [terms + 1] into the pairs, variables and exponents [pairs] */
int32_t tr_prov_poly_get(const tr_prov_poly *, uint64_t *coef, uint64_t *offset, uint32_t *var, uint32_t *exp,
                         uint64_t terms_capacity, uint64_t pairs_capacity);
void tr_prov_poly_free(tr_prov_poly *);

/* ---------------------------------------------------------------- the CUDA library (math_prov_gpu.cu) */
/* the same outputs as tr_prov_evaluate (one work item per valuation, chunks of tr_prov_device_chunk valuations);
 * budget: output = count x noutputs, bytes = tr_prov_device_bytes */
int tr_mgpu_prov_evaluate(const tr_prov_plan_view *, int32_t semiring, const tr_prov_valuations *, void *out,
                          int32_t *status, tr_prov_eval_info *info, const tr_math_budget *, tr_math_refusal *);
#ifdef __cplusplus
}
#endif
#endif
