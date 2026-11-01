/* META-002: a family of related queries over one scope compiled into one multi-output program: common
 * subexpressions (canonical conditions, predicate nodes, decoded columns, shared states through FND-006's fused
 * reduction, math_reduce.h) are computed once, while every query keeps its own domain, validity, group presence,
 * values and explanation; the CPU (math_program.c) and the CUDA finalization (math_program_gpu.cu).
 * SPDX-License-Identifier: Apache-2.0
 *
 * ---- Queries (tr_prog_family) -------------------------------------------------------------------------------------
 * Over a base scope S (FND-004): BOOL-001 conditions and a predicate DAG (as in FND-006), groupings (as in FND-006),
 * and queries "SELECT <grouping keys>, <measures> FROM S WHERE <predicate> GROUP BY <grouping>" (scalar: no GROUP BY:
 * exactly one result row, also when no row passes; its grouping has one group). A measure is COUNT(*), COUNT(p),
 * SUM(p), AVG(p) of an integer polynomial p = sum c_t m_t (int64 coefficients, monomials m_t of total degree <= 4 over
 * S's columns, logical values: a DECIMAL factor counts in units of 10^-scale), or HIST: a DIST-001 histogram spec.
 * SQL semantics per group: a row counts for p iff every column written in p is present (SQL evaluates p to NULL
 * otherwise; also a column whose terms cancel: x - x is NULL where x is); COUNT(p) = those rows, SUM(p) = their exact
 * sum or NULL when there is none, AVG(p) = SUM / COUNT as an exact rational or NULL; COUNT(*) = the group's rows. A
 * grouped query reports its present groups (rows > 0); a scalar one its single row (COUNT 0, SUM / AVG NULL when
 * empty). Equal monomials of p are merged (coefficients added exactly, the total must fit int64; zeros dropped); the
 * written columns are kept.
 *
 * ---- Compilation (tr_prog_compile) --------------------------------------------------------------------------------
 * 1. Canonical conditions (universal rewrites, valid on every database of the scope's schema): a comparison with a
 *    constant becomes an interval of logical integers, BETWEEN [L, U] with L, U in int64 (x > 5.5 on a scale-0 column
 *    is x BETWEEN 6 AND 2^63 - 1; x <> 5.5 is the full interval: TRUE on every present value; an empty interval is
 *    [1, 0]), or NE q for an integer q; a column comparison is ordered (col < col2: x > y is y < x). Equal canonical
 *    conditions are one condition.
 * 2. Canonical predicates (hash-consed nodes, rewritten by identities of SQL's three-valued Kleene logic only):
 *    double negation, constants (AND: FALSE absorbs, TRUE drops; OR dually; XOR: FALSE drops, TRUE negates, UNKNOWN
 *    absorbs), De Morgan (NOT moves to the leaves; NOT above XOR), associative-commutative flattening of AND, OR and
 *    XOR chains (operands sorted; AND and OR idempotent, XOR not: x XOR x is UNKNOWN where x is), and negated leaves
 *    folded into complementary conditions where one exists (NOT x >= L is x <= L - 1, NOT IS NULL is IS NOT NULL, NOT
 *    x = q is x <> q, NOT x < y is x >= y). Never excluded middle (x OR NOT x is UNKNOWN where x is). A predicate whose
 *    canonical form is TRUE is no filter. Queries whose predicates have the same canonical node share it (proof:
 *    canonical). Each node's negation is computed once (linear in the DAG, shared subexpressions included).
 * 3. With TR_PROG_PROVE, a bounded universal search (tr_prog_compare's) over the pairs of distinct canonical
 *    predicates with the same condition support merges those proven equivalent (proof: search); pairs left unknown or
 *    budget-limited stay apart (recorded). Iterations (assignments evaluated) are budgeted.
 * 4. States: per (predicate class, grouping) one FND-006 SQL-mode bank holding the monomials of every measure that
 *    reads no column or whose canonical monomials each read exactly its written columns (its per-term validity is
 *    then the measure's), and COUNT(*); per (class, grouping, written column set) one JOINT-mode bank for the other
 *    measures (listwise over exactly that set; a written column no monomial reads gets a degree-1 term); one
 *    histogram per (class, grouping, spec). The FND-006 plan (one fused traversal, or the repeated plan) builds them.
 * 5. Finalization tables: per measure the output it reads, its terms (index in that output's basis, coefficient), the
 *    term whose count is its valid rows, its groups and its first value slot.
 *
 * ---- Finalization (tr_prog_finalize, tr_mgpu_program_finalize, tr_mgpu_program_run) ----------------------------
 * One value per (measure, group) slot (tr_prog_value): rows, count, status (NULL) and num / den: COUNT_STAR rows;
 * COUNT count; SUM sum_t c_t S_t as an exact 256-bit integer (|c| < 2^63, |S| < 2^127, at most 1024 terms: always
 * fits) or NULL; AVG that sum over count. The slot evaluation (tr_prog_eval_slot) is shared by the CPU loop and the
 * kernel (one work item per slot, its own output slot: no atomics), so both give identical values.
 * tr_mgpu_program_run keeps FND-006's fused states on the device (tr_mgpu_reduce_fused_device) and finalizes them
 * there, then copies the states (for adoption and histogram measures) and the values back.
 *
 * ---- Equivalence (tr_prog_compare, tr_prog_snapshot_compare) ------------------------------------------------------
 * Universal (over every database whose columns are NULL only where the schema allows: a table column with a validity
 * column, an array column with a validity array, and otherwise take any int64 logical value): the WHERE predicates of
 * two queries are EQUIVALENT by their canonical form, or by the search: every assignment of TRUE / FALSE / UNKNOWN to
 * the conditions they read on which [p = TRUE] and [q = TRUE] differ is infeasible (no row gives the conditions those
 * values: per column, NULL or a value in the intervals, holes and points the assignment demands); DIFFERENT with a
 * witness row (per column NULL or a logical value) that one accepts and the other does not; UNKNOWN when a differing
 * assignment involves a BIT or column-comparison condition whose feasibility the search does not decide; BUDGET when
 * 3^k assignments exceed the iterations left. The queries' groupings and measures are compared apart (a filter outcome
 * is about the WHERE clauses; equal filters, groupings and measures make the queries equivalent): measures by their
 * values on every database: COUNT(p) by p's written columns (over no column: COUNT(*)), SUM and AVG by the written
 * columns and the canonical polynomial, histograms by the normalized spec.
 * Snapshot: both predicates evaluated on the scope's selected rows (one pass): the rows each accepts, the rows where
 * they differ and the first one. A snapshot equality is a fact about this snapshot only (no universal authority).
 *
 * ---- Lifetimes ------------------------------------------------------------------------------------------------------
 * A program owns its compiled tables and FND-006 plan. Shared states exist only within a run (TR_PROG_LIFETIME_RUN)
 * or, declared, across runs of the same program on its fresh scope (TR_PROG_LIFETIME_SNAPSHOT; the Python layer
 * checks freshness before every reuse); nothing is shared between programs and no cache is consulted implicitly.
 *
 * Budgets (math_budgets.h): before allocating, circuit = conditions + nodes, output = queries, support = polynomial
 * terms; after canonicalization, circuit = canonical conditions + nodes; iterations = the proof search's assignments
 * (a pair past the budget stays apart); once the tables are built, output = queries + value slots and bytes = the
 * program (its tables and FND-006 plan) and the value slots (72 bytes each); FND-006 checks its own needs. Representation limits (TR_MATH_UNSUPPORTED): 4096
 * queries, 16384 measures, 65536 polynomial terms, 1024 terms per measure, 64 canonical conditions.
 * Errors as FND-006. Objects are immutable; constructors return NULL and set *error, inputs unchanged. */
#ifndef TORBLES_MATH_PROGRAM_H
#define TORBLES_MATH_PROGRAM_H
#include <stdint.h>
#include "math_reduce.h"
#ifdef __cplusplus
extern "C" {
#endif

#define TR_PROG_ABI 1
#define TR_PROG_MAX_QUERIES 4096
#define TR_PROG_MAX_MEASURES 16384
#define TR_PROG_MAX_TERMS 65536
#define TR_PROG_MAX_MEASURE_TERMS 1024
enum { TR_PROG_COUNT_STAR = 1, TR_PROG_COUNT = 2, TR_PROG_SUM = 3, TR_PROG_AVG = 4, TR_PROG_HIST = 5 };
enum { TR_PROG_PROVE = 1 };                                                     /* family flags */
enum { TR_PROG_LIFETIME_RUN = 0, TR_PROG_LIFETIME_SNAPSHOT = 1 };
enum { TR_PROG_EQUIVALENT = 1, TR_PROG_DIFFERENT = 2, TR_PROG_UNKNOWN = 3, TR_PROG_BUDGET = 4 };   /* outcomes */
enum { TR_PROG_BY_CANONICAL = 1, TR_PROG_BY_SEARCH = 2, TR_PROG_BY_WITNESS = 3 };                   /* how */

typedef struct {
    int32_t kind;              /* TR_PROG_COUNT_STAR .. TR_PROG_HIST */
    int32_t first, nterms;     /* COUNT / SUM / AVG: the polynomial's terms (rows of exps / coefs) */
    int32_t reserved;          /* 0 */
    tr_dist_spec hist;         /* HIST: over the scope's columns */
} tr_prog_measure;             /* 112 bytes */

typedef struct {
    int32_t filter, grouping;  /* filter: a node of the family's DAG, or -1 */
    int32_t first, nmeasures;  /* its measures */
    int32_t scalar;            /* 1: no GROUP BY (the grouping has one group) */
    int32_t reserved[3];
} tr_prog_query;               /* 32 bytes */

typedef struct {
    int32_t nconds, nnodes, ngroupings, nqueries;
    int32_t nmeasures, nterms, flags, lifetime;
    const tr_bool_condition *conds;
    const tr_bool_node *nodes;
    const tr_reduce_grouping *groupings;
    const tr_prog_query *queries;
    const tr_prog_measure *measures;
    const uint8_t *exps;       /* nterms x the scope's ncols */
    const int64_t *coefs;      /* nterms */
} tr_prog_family;              /* 88 bytes */

/* ---------------------------------------------------------------- finalization (shared with the kernel) */
typedef struct {
    int32_t kind, output, first, nterms;   /* first / nterms: its contraction terms */
    uint32_t count_term;                   /* the term whose count is the measure's valid rows (UINT32_MAX: rows) */
    int32_t query;
    uint64_t groups, slot;                 /* groups of its grouping; its first value slot */
} tr_prog_fmeasure;                        /* 40 bytes */
typedef struct { uint32_t term, reserved; int64_t coef; } tr_prog_fterm;   /* a term of the output's basis */
typedef struct {
    const uint64_t *rows, *count;          /* the output's states: rows[G], count[G x nterms] */
    const tr_math_i128 *sum;               /* sum[G x nterms] */
    uint64_t ngroups;
    int32_t nterms, reserved;
} tr_prog_fout;                            /* 40 bytes */
typedef struct {
    const tr_prog_fmeasure *measures;      /* the value measures, increasing slots */
    const tr_prog_fterm *terms;
    const tr_prog_fout *outs;              /* per FND-006 output */
    int32_t nmeasures, nterms, noutputs, reserved;
    uint64_t nslots;
} tr_prog_final_view;                      /* 48 bytes */
typedef struct {
    int32_t status, kind;                  /* status 0: a value; 1: SQL NULL (no valid row) */
    uint64_t rows, count;
    tr_dist_i256 num;                      /* the value: num / den */
    uint64_t den, reserved;
} tr_prog_value;                           /* 72 bytes */

/* the value of slot i (the CPU loop and the kernel) */
TR_NUM_HD void tr_prog_eval_slot(const tr_prog_final_view *v, uint64_t i, tr_prog_value *out) {
    int32_t lo = 0, hi = v->nmeasures - 1;
    while (lo < hi) {                                       /* the last measure whose first slot is <= i */
        int32_t mid = lo + (hi - lo + 1) / 2;
        if (v->measures[mid].slot <= i) lo = mid; else hi = mid - 1;
    }
    const tr_prog_fmeasure *m = &v->measures[lo];
    const tr_prog_fout *o = &v->outs[m->output];
    const uint64_t g = i - m->slot, rows = o->rows[g];
    out->kind = m->kind; out->rows = rows; out->status = 0; out->den = 1; out->reserved = 0;
    out->num = tr_dist_w_zero();
    uint64_t count = rows;
    if (m->kind != TR_PROG_COUNT_STAR && m->count_term != UINT32_MAX)
        count = o->count[g * (uint64_t)o->nterms + m->count_term];
    out->count = count;
    if (m->kind == TR_PROG_COUNT_STAR || m->kind == TR_PROG_COUNT) {
        out->num = tr_dist_w_from_i128((__int128)count);
        return;
    }
    if (!count) { out->status = 1; return; }               /* SQL: SUM / AVG over no valid row is NULL */
    tr_dist_i256 acc = tr_dist_w_zero();
    for (int32_t t = 0; t < m->nterms; t++) {
        const tr_prog_fterm *ft = &v->terms[m->first + t];
        tr_dist_i256 p = tr_dist_w_mul128((__int128)ft->coef, tr_i128_get(o->sum[g * (uint64_t)o->nterms + ft->term]));
        tr_dist_i256 s;
        if (tr_dist_w_add(acc, p, &s)) acc = s;            /* cannot overflow: |sum| < 2^127 x 2^63 x 1024 */
    }
    out->num = acc;
    if (m->kind == TR_PROG_AVG) out->den = count;
}

/* the device bytes of a finalization (tr_mgpu_program_finalize, tr_mgpu_program_run): its tables, the view, the
 * values and, with_states, a copy of every bank output's states (the outputs whose rows are given) */
static inline uint64_t tr_prog_final_bytes(const tr_prog_final_view *fv, int with_states) {
    uint64_t b = tr_budget_add(tr_budget_mul((uint64_t)fv->nmeasures, sizeof(tr_prog_fmeasure)),
                               tr_budget_mul((uint64_t)fv->nterms, sizeof(tr_prog_fterm)));
    b = tr_budget_add(b, tr_budget_mul((uint64_t)fv->noutputs, sizeof(tr_prog_fout)));
    b = tr_budget_add(b, tr_budget_add(sizeof(tr_prog_final_view), tr_budget_mul(fv->nslots, sizeof(tr_prog_value))));
    for (int32_t o = 0; with_states && o < fv->noutputs; o++) {
        const tr_prog_fout *x = &fv->outs[o];
        if (!x->rows) continue;
        uint64_t gm = tr_budget_mul(x->ngroups, (uint64_t)x->nterms);
        b = tr_budget_add(b, tr_budget_add(tr_budget_mul(x->ngroups, 8), tr_budget_mul(gm, 24)));
    }
    return b;
}

/* ---------------------------------------------------------------- descriptions */
typedef struct {
    int32_t nqueries, nmeasures, noutputs, nclasses;       /* predicate classes after merging */
    int32_t conds_in, conds, nodes_in, nodes;              /* request vs canonical */
    int32_t merged_canonical, merged_search, search_unknown, search_budget;   /* query predicates merged / left */
    int32_t lifetime, flags, nvalue_measures, nhist;
    uint64_t search_iterations, slots, bytes;
} tr_prog_info;                                            /* 88 bytes */
typedef struct {
    int32_t filter;              /* its canonical predicate node (-1: none) */
    int32_t klass;               /* the query whose predicate represents its class (itself when first) */
    int32_t proof;               /* how it joined the class: TR_PROG_BY_CANONICAL / _SEARCH; 0 its own */
    int32_t grouping, nmeasures, scalar;
    uint64_t conditions;         /* canonical conditions its predicate reads */
    uint64_t slots;              /* its value slots */
    uint64_t state_bytes;        /* bytes of the distinct FND-006 states it reads */
    int32_t outputs, sharers;    /* distinct states it reads; the most queries reading one of them (itself too) */
    int32_t class_size, reserved;
} tr_prog_query_info;            /* 64 bytes */
typedef struct {
    int32_t kind, query, output, nterms;   /* the FND-006 output it reads */
    uint32_t count_term, reserved;
    uint64_t slot, groups;
} tr_prog_measure_info;                    /* 40 bytes */
typedef struct {
    int32_t outcome, how;                  /* TR_PROG_EQUIVALENT ..; TR_PROG_BY_* */
    int32_t grouping_equal, measures_equal;
    uint64_t iterations, support;          /* assignments evaluated; conditions either predicate reads */
    uint64_t assign_true, assign_unknown;  /* the witness assignment: condition i TRUE / UNKNOWN (else FALSE) */
    int32_t accepts, ncols;                /* accepts: 1 the first query's predicate is TRUE at the witness, 2 the second's */
    uint64_t null_mask, set_mask;          /* witness row: columns NULL; columns given a value */
    int64_t value[TR_SCOPE_MAX_COLUMNS];   /* the witness's logical values */
} tr_prog_compare_result;                  /* 328 bytes */
typedef struct {
    int32_t equal, grouping_equal, measures_equal, reserved;
    uint64_t rows_a, rows_b, differ, first, words;   /* first: the first differing row (UINT64_MAX: none) */
} tr_prog_snapshot_result;                 /* 56 bytes */

typedef struct {
    int32_t kind, mode, filter, grouping;  /* a FND-006 output of the program: TR_REDUCE_BANK / _HIST, ... */
    int32_t nterms, readers;               /* its basis terms; the queries reading it */
    uint32_t colset, reserved;             /* JOINT: its column set */
    tr_dist_spec hist;                     /* HIST: its spec */
} tr_prog_output_info;                     /* 128 bytes */

typedef struct tr_prog_s tr_prog;

/* ---------------------------------------------------------------- libalg (math_program.c) */
tr_prog *tr_prog_compile(const tr_scope *scope, const tr_prog_family *, const tr_math_budget *budget,
                         tr_math_refusal *refusal, int32_t *failed, int32_t *error);
void tr_prog_free(tr_prog *);
int32_t tr_prog_get_info(const tr_prog *, tr_prog_info *);
int32_t tr_prog_query_get(const tr_prog *, int32_t query, tr_prog_query_info *);
int32_t tr_prog_measure_get(const tr_prog *, int32_t measure, tr_prog_measure_info *);
/* output o of the program's FND-006 plan; exps (capacity >= nterms x ncols, may be NULL) receives its basis */
int32_t tr_prog_output_get(const tr_prog *, int32_t output, tr_prog_output_info *, uint8_t *exps, uint64_t capacity);
/* the program's FND-006 plan (owned by the program: valid while it lives) */
const tr_reduce_plan *tr_prog_reduce_plan(const tr_prog *);
/* the finalization view over a FND-006 result of this program's plan; outs: capacity >= the plan's outputs */
int32_t tr_prog_final_view_get(const tr_prog *, const tr_reduce_result *, tr_prog_fout *outs, int32_t capacity,
                               tr_prog_final_view *);
/* the value slots of a result on the CPU: out[nslots] */
int32_t tr_prog_finalize(const tr_prog *, const tr_reduce_result *, tr_prog_value *out, uint64_t capacity,
                         const tr_math_budget *budget, tr_math_refusal *refusal);
/* The need of a whole run (what Program.run checks before anything is allocated: an upper bound
 * of every later check but the occupied cells of sparse histograms): the FND-006 pass of `plan` (TR_REDUCE_FUSED /
 * _REPEATED) on `device` (TR_REDUCE_ON_*), then the finalization on `finalize` (TR_REG_FINAL_CPU / _GPU): output =
 * the larger of the pass's and the value slots, bytes = the pass's plus the finalization's (CPU: its views and the
 * value slots; GPU: its device copies (the states too unless the fused pass left them on the device) and the value
 * slots on the host), the other resources the pass's. */
int32_t tr_prog_need(const tr_prog *, int32_t plan, int32_t device, int32_t finalize, tr_math_need *);
/* the finalization's own need (output = value slots; bytes: CPU its views and the value slots; GPU device copies with
 * (states = 1) or without the states) */
int32_t tr_prog_final_need(const tr_prog *, int32_t finalize, int32_t states, tr_math_need *);
int32_t tr_prog_compare(const tr_prog *, int32_t qa, int32_t qb, const tr_math_budget *budget,
                        tr_prog_compare_result *, tr_math_refusal *refusal);
int32_t tr_prog_snapshot_compare(const tr_prog *, int32_t qa, int32_t qb, tr_prog_snapshot_result *);

/* ---------------------------------------------------------------- the CUDA library (math_program_gpu.cu)
 * tr_mgpu_program_finalize: the slots of a final view whose states are host arrays (copied to the device), into
 * out[nslots]. tr_mgpu_program_run: the FND-006 fused pass on the device with its states kept there, then the
 * finalization kernel over them: arena (a TR_REDUCE_FUSED stage's host arena) receives the states, out the values;
 * fv: measures and terms as for the CPU, its outs ignored (built from the view's outputs on the device). */
int tr_mgpu_program_finalize(const tr_prog_final_view *fv, const tr_math_budget *budget, tr_prog_value *out,
                             tr_math_refusal *refusal);
int tr_mgpu_program_run(const tr_reduce_view *view, const tr_math_need *base, const tr_prog_final_view *fv,
                        const tr_math_budget *budget, tr_reduce_arena *arena, tr_prog_value *out,
                        tr_reduce_stats *stats, tr_math_refusal *refusal);
#ifdef __cplusplus
}
#endif
#endif
