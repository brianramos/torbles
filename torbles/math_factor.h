/* PROB-002: factorized probabilistic inference over bounded factor graphs of discrete variables with declared finite
 * supports: the partition function, marginals of any variable set and conditional marginals by sum-product variable
 * elimination, best assignments by max-product (the maximum with every tied assignment, or the k best), exact
 * rationals throughout, along an elimination order chosen by a costed planner (FND-008) under explicit treewidth and
 * intermediate-support budgets; on the CPU (math_factor.c) and, for the data-parallel part (the factor products and
 * sum-outs / max-outs of every elimination step), on a CUDA device (math_factor_gpu.cu), with the same values bit for
 * bit. Alternative backends for weighted model counts reuse COUNT-002 (math_symcount.h: variable elimination, brute
 * force, compiled decision diagrams) and TENSOR-002 (math_tensor.h: costed contraction of the dense factor tables).
 * SPDX-License-Identifier: Apache-2.0
 *
 * ---- The model --------------------------------------------------------------------------------------------------
 * n variables (1..TR_FACTOR_MAX_VARS), variable i with a DECLARED finite support of d_i values (1..TR_FACTOR_MAX_DOMAIN):
 * the logical int64 values (or the NULL category, at most one) with a decimal scale, or, when no values are given, the
 * indices 0..d_i - 1 (scale 0). Values are distinct (TR_MATH_INVALID otherwise). Internally a variable takes the
 * indices 0..d_i - 1 of its declared values, in the given order.
 * m factors (tr_factor_spec), each over a scope of distinct variables (at most TR_FACTOR_MAX_SCOPE), with a table of
 * nonnegative exact rationals, one per joint assignment of the scope in C order (the last scope variable fastest):
 *   TABLE        explicit rationals num/den (int64, num >= 0, den >= 1).
 *   TENSOR       a TENSOR-001 tensor (math_tensor.h) of order = |scope|, dense or sparse, one group's entries: a COUNT
 *                tensor maps each CATEGORY axis to its variable BY VALUE (axis value lo + i, or its NULL slot, must be a
 *                declared value of the variable with the same decimal scale; a declared value the axis does not hold has
 *                weight 0); an INDEX or GROUP axis (GENERIC tensors) by index (its dimension must be d of the variable).
 *                A nonzero cell whose value is not declared is refused (TR_MATH_DOMAIN_ERROR with tr_factor_outside: an
 *                observed value outside the declared support); negative entries (TR_MATH_DOMAIN_ERROR), MOMENT tensors,
 *                per-entry counts and factored forms (TR_MATH_UNSUPPORTED) are refused. A scoped tensor needs its scope.
 *   PROB_MODEL   a PROB-001 categorical model (math_prob.h) with the weights of one request (rationals >= 0; mode
 *                TR_PROB_PROBABILITIES: exactly summing to 1, else TR_MATH_DOMAIN_ERROR): its k <= 3 coordinates are
 *                the scope's variables, each support point maps by value (scales equal) to one cell, which takes the
 *                point's weight; every other cell is 0 (outside the model's declared support). A point of positive
 *                weight whose value is not declared is refused (tr_factor_outside). Exponential families: UNSUPPORTED.
 *   PROB_STATS   PROB-001 retained statistics (one group's masses over the declared support) with the scope they were
 *                built from, mapped like PROB_MODEL. Products of statistics are refused (TR_MATH_UNSUPPORTED).
 * The weight of an assignment x is the product of its factors' entries; Z = sum_x weight(x) (a zero entry is a hard
 * constraint). Natively every factor f is scaled to integers E_f = table x D_f over D_f = the lcm of its denominators
 * (< 2^126, each scaled entry < 2^126, else TR_MATH_UNSUPPORTED); every exact value of an elimination is an integer
 * below 2^(64 L) where the graph proves L from bitlen(prod_f max E_f) + bitlen(prod_i d_i) (counts of assignments
 * included), and the reported rationals divide by D = prod_f D_f with FND-009 big integers (canonical).
 *
 * ---- The factorization is exactly the declared one (GEN-002's rule) ---------------------------------------------
 * A data factor (TENSOR of a scope, PROB_STATS) records its scope's data source (an engine table's address and identity
 * token, or an array scope's row descriptor; math_conv.h). Two data factors of ONE source multiply in the model only
 * under a GEN-002 independence declaration covering that source (tr_conv_decl_covers): their product factors the joint
 * of linked rows into marginals, an independence assumption; otherwise the graph is refused with TR_MATH_DOMAIN_ERROR
 * and tr_factor_check names the pair (reason TR_CONV_NEEDS_DECLARATION / TR_CONV_NOT_COVERED, the source digest,
 * whether it is keyed). Factors of different sources, or without a source (TABLE, PROB_MODEL), multiply freely. The
 * graph records whether a declaration admitted linked factors. Marginals of a variable set are computed from the
 * factor graph by elimination, never as a product of the single marginals.
 *
 * ---- Queries ------------------------------------------------------------------------------------------------------
 * A query shape (tr_factor_shape): the evidence variables (fixed: their values are given per request), the kept
 * variables Q (an ordered set, SUM only) and a mode; a batch of R requests shares the shape, each with its own evidence
 * values. Evidence conditions by restriction (every factor read at the observed values). Answers per request:
 *   PARTITION    (SUM, Q empty) Z_e = sum over the assignments consistent with e of their weight, and P(e) = Z_e / Z
 *                when the normalizer (the PARTITION answer without evidence) is given (status DOMAIN_ERROR if Z = 0)
 *   MARGINAL     (SUM) P(q | e) = Z(q, e) / Z_e for every assignment q of Q (row-major in Q's order), and Z_e
 *   CONDITIONAL  (SUM, the last `given` variables of Q are the condition R) P(q | r, e) = Z(q, r, e) / Z(r, e), undefined
 *                (defined flag 0) where Z(r, e) = 0
 *   MAX          (MAX) the maximum weight w* = max_x weight(x) over the assignments consistent with e, P(x* | e) = w* /
 *                Z_e, the number of maximizers and EVERY tied maximizer (full assignments, sorted lexicographically by
 *                variable index), refused past the output budget with the exact count as the need
 *   KBEST        (KBEST, CPU) the k assignments of largest weight in the canonical order (weight decreasing, then
 *                lexicographic by variable index), with weights, probabilities, and how many assignments tie with the
 *                k-th beyond those returned (zero-weight assignments are never listed)
 * Z_e = 0 makes the conditionals undefined: MARGINAL, CONDITIONAL, MAX and KBEST requests are then refused with their
 * own status TR_MATH_DOMAIN_ERROR (the batch goes on).
 *
 * ---- Elimination programs (shared by the CPU loop and the kernels) ------------------------------------------------
 * An order of the eliminated variables E = (variables not fixed, not kept) is simulated by buckets: a factor goes to
 * the bucket of its first-eliminated variable (its scope restricted by the evidence), step s eliminates order[s]: its
 * operands (the factors and earlier messages of its bucket) are multiplied and the variable summed out (SUM) or maxed
 * out (MAX: the maximum, the minimum index t achieving it (argmax: ties broken by the minimum index, so nothing
 * depends on the thread order), and the number of maximizing completions), giving a message over the union of the
 * operands' scopes minus the variable (sorted by variable index, row-major). Tables of no eliminated variable go to the
 * final step, which multiplies them into the output table over Q (in Q's order; one cell when Q is empty). A MAX
 * program computes the sum chain too (Z_e). Every message is one dense table of the request's arena; a work item is
 * one output cell (tr_factor_cell): out[o] = OP over t of prod_j T_j[base_j + sum_k o_k stride_jk + t sx_j], with
 * exact L-limb products (COUNT-002's tr_symc_w_*), input entries as their 128-bit scaled integers.
 *
 * ---- The costed choice of the order (FND-008) ---------------------------------------------------------------------
 * Candidate orders: EXACT (a dynamic program over the subsets of E, when |E| <= the DP limit and at most 64 variables
 * are free: among the orders whose every step satisfies the per-step budgets, the one of least total joint cells; when
 * none exists, no order fits and the refusal is proven for every order), MIN_FILL, MIN_DEGREE, MIN_WEIGHT (the smallest
 * joint table), WEIGHTED_FILL, NATURAL, REVERSE and a GIVEN order; duplicates merge. Each order runs on the CPU or, with
 * TR_REG_ON_CUDA, on the device. Per candidate the plan computes exactly: width (the induced width: the largest
 * intermediate scope = joint table variables - 1), joint (the largest joint table, cells), support (the largest
 * intermediate table: message or output cells), products, the arena, bytes and device bytes. Needs: variables = the
 * free variables, degree = width (the treewidth budget), support, tensor = joint (the intermediate-support budgets),
 * states = R x arena, output = R x values, bytes (CPU: bases, arena and scratch; CUDA: the device copies and the host
 * readback). Costs (relative work units, NOT time): scan = operand entries read, chain = products x L, construct =
 * message limbs written, materialize = output bytes, transfer = device bytes, fixed = steps (launches), x R; the FND-008
 * planner (tr_planner_choose) checks every candidate's needs against the budget and runs the cheapest admissible one
 * (the first on a tie) with the plan's rates as its override table (row TR_FACTOR_COST_ROW; module 0 in the evidence).
 * A force (a candidate index) runs exactly that candidate or refuses with its code. With no admissible candidate the
 * plan is refused (TR_MATH_BUDGET with the first over-budget candidate's refusal) and still returned with its evidence:
 * every_order says whether the refusal is proven for every order (the DP's infeasibility, or a lower bound past the
 * limit: the minor-min-width treewidth bound, the first step's and the output's table) and min_width / min_joint /
 * min_support are the exact minima over every order (bounds_exact = 1, the DP) or lower bounds (bounds_exact = 0).
 * The plan's own needs: iterations = orders evaluated + DP subsets visited (the DP is skipped, not refused, when it
 * would not fit), bytes = the plan and the DP arrays.
 *
 * ---- Budgets (math_budgets.h), checked before allocation (TR_MATH_BUDGET with the tr_math_refusal) ----------------
 *   graph: variables = n, support = table entries, tensor = the largest table, bytes = the object
 *   plan: iterations, bytes (above); candidates: their needs (above)
 *   program: bytes = its arrays; elimination (CPU): the candidate's needs; CUDA: bytes = tr_factor_device_bytes,
 *     states = R x arena
 *   answers: output = R x values; MAX: output = the maximizers (the exact count, before they are enumerated); KBEST:
 *     support = the longest list held, bytes = every list (checked as they grow), output = |S| (the assignments of
 *     weight at least the k-th's); FND-009's big integers check their own bytes
 *   alternative backends: COUNT-002's and TENSOR-002's own budgets
 * Representation limits (TR_MATH_UNSUPPORTED): more than TR_FACTOR_MAX_VARS variables, a domain past
 * TR_FACTOR_MAX_DOMAIN, scopes past TR_FACTOR_MAX_SCOPE, factor denominators or scaled entries past 2^126, L past
 * TR_FACTOR_MAX_LIMBS; CUDA: L past TR_FACTOR_DEVICE_LIMBS (the candidate is rejected).
 *
 * Errors: TR_MATH_INVALID (-1), _MEMORY (-2), _OVERFLOW (-3: never wrapped), _BUDGET (-5), _DOMAIN_ERROR (-6), _DEVICE
 * (-11), _UNSUPPORTED (-12). Objects are immutable once built and individually owned (borrowed inputs must outlive
 * them: a plan its graph, a program its graph and plan); constructors return NULL and set *error, leaving their inputs
 * unchanged; read-only calls may run concurrently. TR_MATH_STALE is the caller's (the Python layer checks the scopes). */
#ifndef TORBLES_MATH_FACTOR_H
#define TORBLES_MATH_FACTOR_H
#include <stdint.h>
#include <string.h>
#include "math_num.h"
#include "math_budgets.h"
#include "math_symcount.h"
#include "math_registry.h"
#ifdef __cplusplus
extern "C" {
#endif

#define TR_FACTOR_ABI 1
#define TR_FACTOR_MAX_VARS 4096                  /* variables of a graph */
#define TR_FACTOR_MAX_DOMAIN (UINT32_C(1) << 16) /* declared values of one variable */
#define TR_FACTOR_MAX_SCOPE 31                   /* variables of a factor, a message or the output */
#define TR_FACTOR_MAX_LIMBS 4096                 /* 64-bit limbs of an exact value */
#define TR_FACTOR_DEVICE_LIMBS 16                /* limbs a CUDA work item holds in local memory (7 L words) */
#define TR_FACTOR_SCRATCH 7                      /* L-limb words of scratch per work item */
#define TR_FACTOR_DP_DEFAULT 14                  /* the exact subset DP runs for |E| <= this (options.dp_limit) */
#define TR_FACTOR_DP_MAX 24
#define TR_FACTOR_MAX_ORDERS 8                   /* candidate orders (x 2 devices <= TR_REG_MAX_CANDIDATES) */
#define TR_FACTOR_COST_ROW 1                     /* the planner's override-table row of the candidates */
#define TR_FACTOR_NONE UINT32_MAX

enum { TR_FACTOR_TABLE = 1, TR_FACTOR_TENSOR = 2, TR_FACTOR_PROB_MODEL = 3, TR_FACTOR_PROB_STATS = 4 };  /* factors */
enum { TR_FACTOR_SUM = 1, TR_FACTOR_MAX = 2, TR_FACTOR_KBEST = 3 };                          /* elimination modes */
enum { TR_FACTOR_OPD_INPUT = 1, TR_FACTOR_OPD_MESSAGE = 2 };                                 /* operand kinds */
enum { TR_FACTOR_H_EXACT = 0, TR_FACTOR_H_MIN_FILL = 1, TR_FACTOR_H_MIN_DEGREE = 2, TR_FACTOR_H_MIN_WEIGHT = 3,
       TR_FACTOR_H_WEIGHTED_FILL = 4, TR_FACTOR_H_NATURAL = 5, TR_FACTOR_H_REVERSE = 6, TR_FACTOR_H_GIVEN = 7,
       TR_FACTOR_HEURISTICS = 8 };                                                            /* order generators */
enum { TR_FACTOR_A_PARTITION = 1, TR_FACTOR_A_MARGINAL = 2, TR_FACTOR_A_CONDITIONAL = 3, TR_FACTOR_A_MAX = 4,
       TR_FACTOR_A_KBEST = 5, TR_FACTOR_A_COUNT = 6 };                                        /* answer kinds */
enum { TR_FACTOR_ALT_SYMCOUNT = 1,   /* COUNT-002 variable elimination (tr_symc_factors_count) */
       TR_FACTOR_ALT_SYMBRUTE = 2,   /* COUNT-002 brute force over the factor universe (tr_symc_factors_brute) */
       TR_FACTOR_ALT_DIAGRAM = 3,    /* COUNT-002 decision diagram of the parameter encoding (binary variables) */
       TR_FACTOR_ALT_TENSOR = 4 };   /* TENSOR-002 costed contraction of the dense scaled tables */
enum { TR_FACTOR_BY_CPU = 1, TR_FACTOR_BY_DEVICE = 2 };
enum { TR_FACTOR_OUT_VALUE = 1,      /* a value (or the NULL category) the variable does not declare */
       TR_FACTOR_OUT_SCALE = 2,      /* a decimal scale differs from the variable's */
       TR_FACTOR_OUT_DIMS = 3,       /* an index axis whose dimension is not the variable's domain */
       TR_FACTOR_OUT_NEGATIVE = 4 }; /* a negative entry */

/* ------------------------------------------------------------------ structs crossing the ctypes boundary */
/* one factor of a graph (56 bytes); see The model */
typedef struct {
    int32_t kind, mode;              /* TR_FACTOR_*; PROB_MODEL: TR_PROB_WEIGHTS (0) / _PROBABILITIES (1) */
    uint32_t nscope, reserved;
    uint64_t scope_at;               /* its variables vars[scope_at .. scope_at + nscope) (TENSOR: per axis) */
    uint64_t table_at;               /* TABLE: num/den[table_at ..] (prod d entries); PROB_MODEL: the d weights */
    uint64_t group;                  /* TENSOR, PROB_STATS: the group whose entries are the table */
    const void *object;              /* TENSOR: tr_tensor; PROB_MODEL: tr_prob_model; PROB_STATS: tr_prob_stats */
    const void *scope;               /* TENSOR (scoped), PROB_STATS: the tr_scope the state was built from */
} tr_factor_spec;

/* why a factor was refused for its values (32 bytes) */
typedef struct {
    uint32_t factor, axis;           /* the factor and its axis (coordinate) */
    int32_t reason, null;            /* TR_FACTOR_OUT_*; the value is the NULL category */
    int64_t value;                   /* the logical value (or the dimension, TR_FACTOR_OUT_DIMS) */
    uint64_t mass;                   /* the cell's entry (saturated) */
} tr_factor_outside;

/* why linked factors were refused (32 bytes; GEN-002's reasons TR_CONV_*) */
typedef struct {
    int32_t reason, first, second, keyed;   /* first, second: the factors sharing the source (-1: none) */
    uint64_t digest[2];                     /* the shared source's digest */
} tr_factor_check;

typedef struct {                     /* 104 bytes */
    uint64_t entries, bytes, assignments_lo, assignments_hi;   /* prod d_i (both 0 when it passes 128 bits: wide) */
    uint64_t digest[2];              /* of the exact content (domains, scopes, scaled entries, denominators) */
    uint32_t nvars, nfactors, max_scope, wide;
    int32_t declared, linked, sources, data;   /* a declaration admitted linked factors; linked factors; distinct
                                                  sources; data factors */
    uint32_t limbs, reserved;        /* L: limbs of every exact value of an elimination */
    uint64_t max_table, reserved2;
} tr_factor_graph_info;

typedef struct {                     /* 96 bytes */
    int32_t kind, mode;
    uint32_t nscope, linked;         /* linked: shares its data source with another factor */
    uint64_t entries, group, nonzero;
    uint64_t denom[2], maxent[2];    /* D_f and the largest scaled entry (128 bits each) */
    uint64_t source_digest[2];       /* 0 without a data source */
    int32_t source_kind, keyed;      /* TR_SCOPE_TABLE / _ARRAYS (0: none); the source scope has several groups */
} tr_factor_info;

/* a query shape (48 bytes) */
typedef struct {
    int32_t mode;                    /* TR_FACTOR_SUM / _MAX / _KBEST */
    uint32_t nkeep, nfixed, reserved;
    const uint32_t *keep;            /* SUM: the kept variables in output order (MAX, KBEST: none) */
    const uint32_t *fixed;           /* the evidence variables (values per request, in this order) */
    uint64_t requests;               /* R >= 1 */
    uint64_t kbest;                  /* KBEST: k >= 1 */
} tr_factor_shape;

/* plan options (224 bytes): rates[0] CPU, rates[1] CUDA per FND-008 component when custom_rates */
typedef struct {
    int32_t devices;                 /* TR_REG_ON_CPU | TR_REG_ON_CUDA (0: CPU) */
    int32_t heuristics;              /* bits 1 << TR_FACTOR_H_* (0: all but GIVEN; GIVEN is added with an order) */
    int32_t force;                   /* -1: none; else the candidate index that must run */
    int32_t custom_rates;
    uint32_t norder, dp_limit;       /* a given order of E (0: none); the DP limit (0: TR_FACTOR_DP_DEFAULT) */
    const uint32_t *order;
    double rates[2][TR_COST_COMPONENTS];
} tr_factor_plan_options;

/* one candidate (456 bytes): FND-008's candidate and the order's exact sizes (per request) */
typedef struct {
    tr_reg_candidate reg;
    int32_t heuristic, device, nsteps, order;   /* TR_FACTOR_H_* (the first generator of the order), TR_REG_ON_*,
                                                   steps (the final one included), the order's index */
    uint64_t width, joint, support, products, arena;
    uint64_t bytes, device_bytes;
    uint32_t limbs, device_ok;       /* device_ok: the CUDA kernel can hold its limbs */
} tr_factor_candidate;

typedef struct {                     /* 152 bytes */
    int32_t ncandidates, chosen, code, reason;
    int32_t complete, every_order, forced, devices;   /* complete: the exact DP ran over every order */
    int32_t bounds_exact, mode, norders, reserved;
    uint64_t eliminate, keep, fixed, free_vars;       /* variables eliminated, kept, fixed, free */
    uint64_t dp_states, evaluated;                    /* DP subsets visited; orders evaluated */
    uint64_t min_width, min_joint, min_support;       /* exact minima over every order, or lower bounds */
    tr_math_refusal refusal;                          /* the plan-level refusal (24 bytes) */
    uint64_t bytes;
} tr_factor_plan_info;

/* the program as the CPU loop and the kernels read it */
typedef struct {                     /* 56 bytes */
    uint32_t final, var;             /* final: the last step (a product, no variable); var: the eliminated variable */
    uint32_t nout, nopd;             /* output dims; operands */
    uint64_t cells, nterms;          /* output cells; values of the eliminated variable (1 for the final step) */
    uint64_t out_at;                 /* the message's first entry in a request's arena */
    uint64_t opd_at, dim_at;         /* operands opd[opd_at ..]; output dims dims[dim_at ..] (row-major) */
} tr_factor_step;
typedef struct {                     /* 16 bytes */
    uint32_t kind, index;            /* INPUT: the factor (its base per request); MESSAGE: the step it reads */
    uint64_t stride_at;              /* strides[stride_at .. + nout] per output dim, then the eliminated variable's */
} tr_factor_operand;
typedef struct {                     /* 96 bytes */
    uint32_t nsteps, limbs, mode, nfactors;   /* mode: TR_FACTOR_SUM / _MAX */
    uint64_t arena;                  /* entries of one request's arena (every message) */
    uint64_t nopd, nstrides, ndims;
    uint64_t inputs;                 /* input entries (2 words each: the scaled numerators) */
    const tr_factor_step *step;
    const tr_factor_operand *opd;
    const uint64_t *stride, *dims;
    const uint64_t *input;           /* [inputs x 2] */
} tr_factor_program;

typedef struct {                     /* 64 bytes */
    int32_t kind, backend, mode, method;   /* TR_FACTOR_A_*; TR_FACTOR_BY_*; the elimination mode; ALT method or 0 */
    uint64_t requests, nvalues;      /* values per request */
    uint64_t cells, assignments;     /* the output table's cells; assignments held (every request) */
    uint32_t nvars, limbs;           /* values per assignment; the largest numerator / denominator limbs */
    uint32_t nkeep, ngiven;
} tr_factor_answer_info;

typedef struct tr_factor_graph_s tr_factor_graph;
typedef struct tr_factor_plan_s tr_factor_plan;
typedef struct tr_factor_prog_s tr_factor_prog;
typedef struct tr_factor_answer_s tr_factor_answer;

/* ------------------------------------------------------------------ the elimination step (CPU and CUDA) */
/* 1 when a > b, -1 when a < b, 0 when equal (L limbs) */
TR_NUM_HD int tr_factor_w_cmp(const uint64_t *a, const uint64_t *b, uint32_t L) {
    for (uint32_t i = L; i-- > 0;) {
        if (a[i] > b[i]) return 1;
        if (a[i] < b[i]) return -1;
    }
    return 0;
}
TR_NUM_HD void tr_factor_w_one(uint64_t *x, uint32_t L) { tr_symc_w_zero(x, L); x[0] = 1; }
/* x *= y (L limbs, through t); 1 on an overflow */
TR_NUM_HD int tr_factor_w_mulby(uint64_t *x, const uint64_t *y, uint64_t *t, uint32_t L) {
    int over = tr_symc_w_mul(x, y, L, t);
    tr_symc_w_copy(x, t, L);
    return over;
}
/* the entry index of operand j of step st at output coordinates c and t (an input: bases[r nf + factor] + offset;
 * a message: r arena + its out_at + offset) */
TR_NUM_HD uint64_t tr_factor_opd_entry(const tr_factor_program *p, const tr_factor_step *st, const tr_factor_operand *op,
                                       uint64_t r, const uint64_t *bases, const uint64_t *c, uint64_t t) {
    const uint64_t *s = p->stride + op->stride_at;
    uint64_t off = t * s[st->nout];
    for (uint32_t k = 0; k < st->nout; k++) off += c[k] * s[k];
    if (op->kind == TR_FACTOR_OPD_INPUT) return bases[r * (uint64_t)p->nfactors + op->index] + off;
    return r * p->arena + p->step[op->index].out_at + off;
}
/* the output coordinates of cell o (row-major: the last dim fastest) */
TR_NUM_HD void tr_factor_coords(const tr_factor_program *p, const tr_factor_step *st, uint64_t o, uint64_t *c) {
    for (uint32_t k = st->nout; k-- > 0;) {
        uint64_t d = p->dims[st->dim_at + k];
        c[k] = o % d;
        o /= d;
    }
}
/* the products of term t at coordinates c: sp (the sum chain) and, with maxv, mp (the max chain) and cp (the counts
 * of maximizing completions); 0 when an operand entry is zero (the products are then 0), 1 otherwise; *over set on an
 * overflow (excluded by the host's L proof). scratch: one L-limb word (t). */
TR_NUM_HD int tr_factor_term(const tr_factor_program *p, const tr_factor_step *st, uint64_t r, const uint64_t *bases,
                             const uint64_t *c, uint64_t t, const uint64_t *vals, const uint64_t *maxv,
                             const uint64_t *cnts, uint64_t *sp, uint64_t *mp, uint64_t *cp, uint64_t *tmp, int *over) {
    const uint32_t L = p->limbs;
    tr_factor_w_one(sp, L);
    if (maxv) { tr_factor_w_one(mp, L); tr_factor_w_one(cp, L); }
    for (uint32_t j = 0; j < st->nopd; j++) {
        const tr_factor_operand *op = &p->opd[st->opd_at + j];
        uint64_t e = tr_factor_opd_entry(p, st, op, r, bases, c, t);
        if (op->kind == TR_FACTOR_OPD_INPUT) {
            uint64_t lo = p->input[2 * e], hi = p->input[2 * e + 1];
            if (!lo && !hi) return 0;
            *over |= tr_symc_w_mul128(sp, L, lo, hi, tmp); tr_symc_w_copy(sp, tmp, L);
            if (maxv) { *over |= tr_symc_w_mul128(mp, L, lo, hi, tmp); tr_symc_w_copy(mp, tmp, L); }
        } else {
            const uint64_t *v = vals + e * L;
            if (tr_symc_w_is_zero(v, L)) return 0;
            *over |= tr_factor_w_mulby(sp, v, tmp, L);
            if (maxv) {
                *over |= tr_factor_w_mulby(mp, maxv + e * L, tmp, L);
                *over |= tr_factor_w_mulby(cp, cnts + e * L, tmp, L);
            }
        }
    }
    return 1;
}
/* cell o of step s of request r: vals (and, MAX, maxv / cnts / arg) written at the cell's own slot; scratch:
 * TR_FACTOR_SCRATCH L words; returns 1 on an overflow (excluded by the host's proof), else 0 */
TR_NUM_HD int tr_factor_cell(const tr_factor_program *p, uint32_t s, uint64_t r, uint64_t o, const uint64_t *bases,
                             uint64_t *vals, uint64_t *maxv, uint64_t *cnts, uint32_t *arg, uint64_t *scratch) {
    const tr_factor_step *st = &p->step[s];
    const uint32_t L = p->limbs;
    const int mx = p->mode == TR_FACTOR_MAX;
    uint64_t *acc = scratch, *sp = scratch + L, *tmp = scratch + 2 * L, *mp = scratch + 3 * L, *cp = scratch + 4 * L,
             *best = scratch + 5 * L, *bcnt = scratch + 6 * L;
    uint64_t c[TR_FACTOR_MAX_SCOPE];
    tr_factor_coords(p, st, o, c);
    int over = 0, found = 0;
    uint32_t barg = 0;
    tr_symc_w_zero(acc, L);
    if (mx) { tr_symc_w_zero(best, L); tr_symc_w_zero(bcnt, L); }
    for (uint64_t t = 0; t < st->nterms; t++) {
        if (!tr_factor_term(p, st, r, bases, c, t, vals, mx ? maxv : (const uint64_t *)0, cnts, sp, mp, cp, tmp, &over))
            continue;                                       /* a zero product: no sum, no maximizer */
        over |= tr_symc_w_add(acc, sp, L);
        if (!mx) continue;
        int cmp = found ? tr_factor_w_cmp(mp, best, L) : 1;
        if (cmp > 0) { tr_symc_w_copy(best, mp, L); tr_symc_w_copy(bcnt, cp, L); barg = (uint32_t)t; found = 1; }
        else if (cmp == 0) over |= tr_symc_w_add(bcnt, cp, L);
    }
    const uint64_t e = r * p->arena + st->out_at + o;
    tr_symc_w_copy(vals + e * L, acc, L);
    if (mx) { tr_symc_w_copy(maxv + e * L, best, L); tr_symc_w_copy(cnts + e * L, bcnt, L); arg[e] = barg; }
    return over;
}

/* ------------------------------------------------------------------ host-side checks and sizes (both libraries) */
/* 0 when the program view is consistent (steps in arena order, operands reading inputs or earlier steps, dims whose
 * product is the cells, every stride's reach inside its table: inputs within `inputs` when bases are given), else
 * TR_MATH_INVALID; the input bases are checked by tr_factor_bases_check */
static inline int32_t tr_factor_program_check(const tr_factor_program *p) {
    if (!p || p->nsteps < 1 || !p->step || p->limbs < 1 || p->limbs > TR_FACTOR_MAX_LIMBS ||
        (p->mode != TR_FACTOR_SUM && p->mode != TR_FACTOR_MAX) || (p->nopd && !p->opd) || (p->nstrides && !p->stride) ||
        (p->ndims && !p->dims) || (p->inputs && !p->input)) return -1;
    uint64_t at = 0;
    for (uint32_t s = 0; s < p->nsteps; s++) {
        const tr_factor_step *st = &p->step[s];
        if (st->out_at != at || st->nout > TR_FACTOR_MAX_SCOPE || st->dim_at > p->ndims || st->nout > p->ndims - st->dim_at ||
            st->opd_at > p->nopd || st->nopd > p->nopd - st->opd_at || st->nterms < 1 || st->cells < 1) return -1;
        if (st->final != (s + 1 == p->nsteps) || (st->final && st->nterms != 1)) return -1;
        uint64_t cells = 1;
        for (uint32_t k = 0; k < st->nout; k++) {
            uint64_t d = p->dims[st->dim_at + k];
            if (d < 1) return -1;
            cells = tr_budget_mul(cells, d);
        }
        if (cells != st->cells) return -1;
        for (uint32_t j = 0; j < st->nopd; j++) {
            const tr_factor_operand *op = &p->opd[st->opd_at + j];
            if (op->stride_at > p->nstrides || (uint64_t)st->nout + 1 > p->nstrides - op->stride_at) return -1;
            const uint64_t *sd = p->stride + op->stride_at;
            uint64_t reach = tr_budget_mul(st->nterms - 1, sd[st->nout]);
            for (uint32_t k = 0; k < st->nout; k++) reach = tr_budget_add(reach, tr_budget_mul(p->dims[st->dim_at + k] - 1, sd[k]));
            if (op->kind == TR_FACTOR_OPD_MESSAGE) {
                if (op->index >= s || reach >= p->step[op->index].cells) return -1;
            } else if (op->kind == TR_FACTOR_OPD_INPUT) {
                if (op->index >= p->nfactors || reach >= p->inputs) return -1;
            } else return -1;
        }
        at = tr_budget_add(at, st->cells);
    }
    return at == p->arena && p->step[p->nsteps - 1].final ? 0 : -1;
}
/* 0 when every input operand of every request stays inside the inputs from its base, else TR_MATH_INVALID */
static inline int32_t tr_factor_bases_check(const tr_factor_program *p, uint64_t R, const uint64_t *bases) {
    if (R && p->nfactors && !bases) return -1;
    for (uint32_t s = 0; s < p->nsteps; s++) {
        const tr_factor_step *st = &p->step[s];
        for (uint32_t j = 0; j < st->nopd; j++) {
            const tr_factor_operand *op = &p->opd[st->opd_at + j];
            if (op->kind != TR_FACTOR_OPD_INPUT) continue;
            const uint64_t *sd = p->stride + op->stride_at;
            uint64_t reach = tr_budget_mul(st->nterms - 1, sd[st->nout]);
            for (uint32_t k = 0; k < st->nout; k++) reach = tr_budget_add(reach, tr_budget_mul(p->dims[st->dim_at + k] - 1, sd[k]));
            for (uint64_t r = 0; r < R; r++) {
                uint64_t b = bases[r * p->nfactors + op->index];
                if (b >= p->inputs || reach >= p->inputs - b) return -1;
            }
        }
    }
    return 0;
}
/* the bytes tr_mgpu_factor_eliminate copies to and allocates on the device for R requests */
static inline uint64_t tr_factor_device_bytes(const tr_factor_program *p, uint64_t R) {
    uint64_t b = tr_budget_mul(p->nsteps, sizeof(tr_factor_step));
    b = tr_budget_add(b, tr_budget_mul(p->nopd, sizeof(tr_factor_operand)));
    b = tr_budget_add(b, tr_budget_mul(tr_budget_add(p->nstrides, p->ndims), 8));
    b = tr_budget_add(b, tr_budget_mul(p->inputs, 16));
    b = tr_budget_add(b, tr_budget_mul(tr_budget_mul(R, p->nfactors), 8));
    uint64_t entries = tr_budget_mul(R, p->arena), words = tr_budget_mul(entries, p->limbs);
    b = tr_budget_add(b, tr_budget_mul(words, p->mode == TR_FACTOR_MAX ? 24 : 8));
    if (p->mode == TR_FACTOR_MAX) b = tr_budget_add(b, tr_budget_mul(entries, 4));
    return tr_budget_add(b, 16);
}

/* ------------------------------------------------------------------ graphs (libalg) */
/* n variables with domains[n] and their declared values (values / nulls [sum d_i], scales [n]; values NULL: the
 * indices), nf factors (specs; scopes in vars[]; TABLE and PROB_MODEL entries in num / den [nentries]); decl: a GEN-002
 * declaration (struct tr_conv_decl_s) or NULL. outside / check (may be NULL): why a factor or a pair was refused. */
tr_factor_graph *tr_factor_graph_create(uint32_t nvars, const uint32_t *domains, const int64_t *values,
                                        const uint8_t *nulls, const int32_t *scales, uint32_t nfactors,
                                        const tr_factor_spec *specs, const uint32_t *vars, const int64_t *num,
                                        const int64_t *den, uint64_t nentries, const void *decl,
                                        const tr_math_budget *, tr_math_refusal *, tr_factor_outside *outside,
                                        tr_factor_check *check, int32_t *error);
int32_t tr_factor_graph_get_info(const tr_factor_graph *, tr_factor_graph_info *);
int32_t tr_factor_graph_factor(const tr_factor_graph *, uint32_t f, tr_factor_info *);
/* factor f's scaled table: scope (nscope variables) and entries (2 words each), and D_f (2 words) */
int32_t tr_factor_graph_table(const tr_factor_graph *, uint32_t f, uint32_t *scope, uint64_t *entries,
                              uint64_t capacity, uint64_t *denom);
void tr_factor_graph_free(tr_factor_graph *);

/* ------------------------------------------------------------------ plans (libalg) */
/* Plans a query shape (see The costed choice). Returns the plan's code; *plan is the plan with its evidence (also when
 * refused for want of an admissible candidate), NULL only when the shape or the search itself was refused. */
int32_t tr_factor_plan_create(const tr_factor_graph *, const tr_factor_shape *, const tr_factor_plan_options *,
                              const tr_math_budget *, tr_factor_plan **plan, tr_math_refusal *);
int32_t tr_factor_plan_get_info(const tr_factor_plan *, tr_factor_plan_info *);
int32_t tr_factor_plan_get(const tr_factor_plan *, int32_t index, tr_factor_candidate *);
/* the order (eliminated variables) of candidate index into out[capacity], *n its length */
int32_t tr_factor_plan_order(const tr_factor_plan *, int32_t index, uint32_t *out, uint64_t capacity, uint64_t *n);
void tr_factor_plan_free(tr_factor_plan *);
/* rates (relative work units per amount, NOT time) of the candidates: [0] CPU, [1] CUDA */
void tr_factor_default_rates(double rates[2][TR_COST_COMPONENTS]);

/* ------------------------------------------------------------------ programs and elimination (libalg) */
/* the program of candidate index (-1: the chosen one); a rejected candidate is refused with its code */
tr_factor_prog *tr_factor_prog_create(const tr_factor_graph *, const tr_factor_plan *, int32_t index,
                                      const tr_math_budget *, tr_math_refusal *, int32_t *error);
int32_t tr_factor_prog_get_view(const tr_factor_prog *, tr_factor_program *);
/* the candidate the program runs (its index in the plan, its device) */
int32_t tr_factor_prog_candidate(const tr_factor_prog *, int32_t *index, int32_t *device);
/* bases [R x nfactors] of R requests' evidence values [R x nfixed] (value indices, < d of the variable) */
int32_t tr_factor_prog_bases(const tr_factor_prog *, uint64_t R, const uint32_t *evidence, uint64_t *bases);
/* the needs of an elimination of R requests on the CPU (TR_FACTOR_BY_CPU) or the device (_DEVICE) */
int32_t tr_factor_prog_need(const tr_factor_prog *, uint64_t R, int32_t target, tr_math_need *);
/* the raw arena of R requests on the CPU (the reference of tr_mgpu_factor_eliminate): vals [R x arena x L]; MAX: maxv,
 * cnts [R x arena x L], arg [R x arena] */
int32_t tr_factor_eliminate(const tr_factor_program *, uint64_t R, const uint64_t *bases, uint64_t *vals, uint64_t *maxv,
                            uint64_t *cnts, uint32_t *arg, const tr_math_budget *, tr_math_refusal *);
void tr_factor_prog_free(tr_factor_prog *);

/* ------------------------------------------------------------------ answers (libalg) */
/* the answers of R requests from a raw arena (CPU or device: checked); kind PARTITION, MARGINAL, CONDITIONAL (the last
 * `given` kept variables are the condition) for SUM programs, MAX for MAX programs; normalizer: a PARTITION answer of
 * the same graph without evidence (P(e) = Z_e / Z), or NULL */
tr_factor_answer *tr_factor_answer_create(const tr_factor_prog *, int32_t kind, uint32_t given, uint64_t R,
                                          const uint32_t *evidence, const uint64_t *vals, const uint64_t *maxv,
                                          const uint64_t *cnts, const uint32_t *arg, const tr_factor_answer *normalizer,
                                          int32_t backend, const tr_math_budget *, tr_math_refusal *, int32_t *error);
/* the k best assignments of R requests (a MAX or KBEST program, on the CPU): [Z_e, P(e), then per assignment its
 * weight and probability]; normalizer as for tr_factor_answer_create */
tr_factor_answer *tr_factor_kbest(const tr_factor_prog *, uint64_t R, const uint32_t *evidence, uint64_t k,
                                  const tr_factor_answer *normalizer, const tr_math_budget *, tr_math_refusal *,
                                  int32_t *error);
/* Z_e of one evidence (fixed / values: nfixed) by an alternative backend (TR_FACTOR_ALT_*): a COUNT answer [Z_e] */
tr_factor_answer *tr_factor_alternative(const tr_factor_graph *, int32_t method, uint32_t nfixed, const uint32_t *fixed,
                                        const uint32_t *values, const tr_math_budget *, tr_math_refusal *,
                                        int32_t *error);
int32_t tr_factor_answer_get_info(const tr_factor_answer *, tr_factor_answer_info *);
/* request r's status: 0 or a native code (TR_MATH_DOMAIN_ERROR: the evidence or the model has probability zero) */
int32_t tr_factor_answer_status(const tr_factor_answer *, uint64_t r);
/* value i of request r as canonical limbs (0 limbs: zero); returns 1 (defined) or 0 (undefined), or a code */
int32_t tr_factor_answer_value(const tr_factor_answer *, uint64_t r, uint64_t i, uint64_t *num, uint64_t *den,
                               uint32_t cap, uint32_t *nlimbs, uint32_t *dlimbs);
/* value i of the requests r0 .. r0 + n - 1 at once: num / den [n x cap] limbs, nlimbs / dlimbs [n] (dlimbs 0:
 * undefined); TR_MATH_INVALID when cap is smaller than the answer's limbs */
int32_t tr_factor_answer_column(const tr_factor_answer *, uint64_t i, uint64_t r0, uint64_t n, uint64_t *num,
                                uint64_t *den, uint32_t cap, uint32_t *nlimbs, uint32_t *dlimbs);
/* request r's assignments (n values each, the variables' value indices) into out[capacity]; *count of them */
int32_t tr_factor_answer_assignments(const tr_factor_answer *, uint64_t r, uint32_t *out, uint64_t capacity,
                                     uint64_t *count);
/* KBEST: request r's assignments tied with the k-th beyond those returned (saturated) */
int32_t tr_factor_answer_tied(const tr_factor_answer *, uint64_t r, uint64_t *tied);
void tr_factor_answer_free(tr_factor_answer *);

/* ------------------------------------------------------------------ the CUDA library (math_factor_gpu.cu)
 * tr_mgpu_factor_eliminate: the raw arena of tr_factor_eliminate on the device: one launch per step, one work item
 * per (request, output cell) running tr_factor_cell into its own slot (no atomics on the values; overflow flags by an
 * atomic OR, excluded by the host's L proof); the host checks the program, the bases and the budget (bytes =
 * tr_factor_device_bytes, states = R x arena) before allocating; L past TR_FACTOR_DEVICE_LIMBS is TR_MATH_UNSUPPORTED.
 * Adopt the readback with tr_factor_answer_create (backend TR_FACTOR_BY_DEVICE). */
int tr_mgpu_factor_eliminate(const tr_factor_program *, uint64_t R, const uint64_t *bases, uint64_t *vals,
                             uint64_t *maxv, uint64_t *cnts, uint32_t *arg, const tr_math_budget *, tr_math_refusal *);
#ifdef __cplusplus
}
#endif
#endif
