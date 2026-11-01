/* META-001: scoped query equivalence, implication and counterexamples.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Query parts (tr_equiv_set, built once and shared by every question, check and run):
 *   predicates  BOOL-001 circuits (math_bool.h: CONST, VAR, NOT, AND, OR, XOR nodes; SQL three-valued logic) over
 *               BOOL-001 conditions on scope columns (comparisons with exact constants or another column, BETWEEN,
 *               IS [NOT] NULL, BIT). The set hash-conses them: equal conditions (constants reduced) are one condition,
 *               structurally equal nodes (AND / OR / XOR operands ordered) are one node of a shared DAG, so a
 *               predicate is a root of that DAG and common subexpressions are explicit;
 *   measures    polynomials over scope columns with exact rational coefficients (GEOM-001's tr_geom_poly; the first
 *               slice's tr_math_poly converts with tr_equiv_poly_from_math), canonical: equal measures are one;
 *   queries     (filter predicate or none, measure or none: COUNT): equal pairs are one.
 * A measure reads the scope's exact logical integers (GEOM-001's convention: a decimal column's value is logical /
 * 10^scale); its value on a row is NULL when a column it reads (its canonical support) is NULL (SQL); a query's rows
 * are those where its filter is TRUE (UNKNOWN filters a row out, as WHERE).
 *
 * Questions (tr_equiv_question): two parts of one kind and a relation:
 *   PREDICATE  EQUIVALENT with semantics FILTER (the TRUE rows are the same) or KLEENE (the same truth value
 *              everywhere), or IMPLIES (every TRUE row of a is TRUE in b: containment of the filtered rows);
 *   MEASURE    EQUIVALENT: equal values, with NULL semantics SQL (a row where exactly one side is NULL differs) or
 *              SKIP (rows where a side is NULL excluded: FND-007 policy SKIP);
 *   QUERY      EQUIVALENT (the same filtered rows, and on them equal measures) or IMPLIES (the rows of a are rows of b
 *              with the same measure values); a COUNT query is only compared with a COUNT query.
 *
 * Domains and verdicts (PROVEN, REFUTED with a concrete witness, or UNKNOWN with the reason; every verdict labelled):
 *   SNAPSHOT   a FND-004 scope: both sides evaluated exactly on every selected row (tr_equiv_check): the conditions
 *              once as BOOL-001 masks (CPU, or device words adopted), each question's cone word-parallel, measures
 *              by GEOM-001's exact pair candidates (P = D (f - g) in proven W words). PROVEN means: on these rows.
 *              The proof is labelled SNAPSHOT and carries an FND-007 record of the scope (exact freshness) and, for
 *              measures, the FND-007 snapshot certificate of the POLYNOMIAL claim P = 0 (policy SKIP over the union of
 *              the measures' columns); FND-007 has no predicate claim kind, so a predicate proof is the record and the
 *              exhaustive evidence. REFUTED gives the smallest differing row. With relation certificates (DISC-002
 *              identities, DISC-001 claims), a measure question is first reduced by GEOM-001 (complete Groebner basis):
 *              NF(f - g) = 0 is a PROVEN, label RELATIONS (the derived FND-007 certificate, which holds where the
 *              relations hold: their own scope); a nonzero remainder refutes nothing and the rows decide.
 *   DOMAIN     a declared domain (tr_equiv_domain) per column: an integer range [lo, hi] or a finite set of logical
 *              values, a decimal scale and whether NULL is a value. Universal over that domain:
 *              ENUMERATE   every point of the product (at most the support budget): the snapshot check of a synthetic
 *                          scope of all points (witness: the point);
 *              DECIDE      ranges only. Predicates: per NULL pattern of the nullable columns read (2^m, budgeted) the
 *                          conditions become constants or two-valued atoms, the TRUE and FALSE functions of both sides
 *                          are reduced ordered BDDs over the atoms, and every satisfying cube of their difference is
 *                          checked for an integer point: bounds and difference constraints x - y <= c (Fourier-Motzkin
 *                          restricted to difference constraints, i.e. Bellman-Ford shortest paths: exact over the
 *                          integers), disjunctions (<>, NOT BETWEEN) by branching; the first feasible cube is a
 *                          counterexample, checked by evaluating both sides at the point. BDD equality without any
 *                          theory step holds for any domain: label UNIVERSAL. BIT conditions and comparisons of columns
 *                          of different scales are outside the theory: a cube that needs them gives UNKNOWN.
 *                          Measures: P = D (f - g) canonical; zero is PROVEN UNIVERSAL (an FND-007 RING_IDENTITY law of
 *                          D f - D g); else under SQL semantics a NULL-pattern mismatch (a nullable column read by one
 *                          side only) is a witness, else P vanishes on the box iff it vanishes on the grid
 *                          prod_v S_v, S_v the min(|range|, deg_v P + 1) integers of the range nearest 0 (the
 *                          combinatorial Nullstellensatz): enumerated exactly (budgeted), the first non-root a witness.
 *                          Queries: the filters as predicates (FILTER semantics), then the measures on the filter's
 *                          TRUE region: P = 0 proves it; else the TRUE cubes whose literals bound single columns are
 *                          boxes searched on their own grids; a cube with a column comparison gives UNKNOWN.
 *   A snapshot proof is never a universal claim: the decision procedures never read rows, and a SNAPSHOT proof
 *   authorizes only on the scope it was checked on (tr_equiv_authorize: its FND-007 record fresh), a DOMAIN proof only
 *   on a scope whose selected values lie in the domain (checked row by row), a RELATIONS proof as FND-007 authorizes
 *   its derived certificate; UNIVERSAL always. An unsupported or over-budget universal question is UNKNOWN with the
 *   reason (why BUDGET with the resource, need and limit; UNSUPPORTED), never an exception and never a guess.
 *
 * Families (tr_equiv_family_run): every query's COUNT of filtered rows, measure rows present and exact SUM (W words of
 * the integer form D f; the sum is the value / D) over a scope, SHARED (one scan: the distinct conditions' masks once,
 * the shared DAG once per word, each distinct measure once per row) or SEPARATE (each query alone: its own masks,
 * nodes and measure: one scan per query); the statistics count the scans, condition rows, node words and measure rows
 * actually evaluated.
 *
 * The relationship optimizer (relationships_optimizer.c, through its public tr_optimize): tr_equiv_rewrite lowers a
 * degree <= 2 integer measure to stored codes (logical = code + the catalog's offset), runs the bounded beam search
 * under the catalog's verified rules, and lifts the result back to logical values exactly: a candidate equivalent
 * measure whose equivalence holds where those rules were verified (the catalog's snapshot) and is checked here like
 * any other question. relationships_stage.c (whole-stage decompositions of compiled plans) is an executor
 * transformation, not a query part, and is not used.
 *
 * Budgets (math_budgets.h), checked before allocation or, in the searches, as reached (then UNKNOWN): sets: circuit =
 * the given nodes, support = the given measure terms, output = the given parts, bytes (upper bounds from the given
 * parts), variables = the distinct conditions (checked as each is interned); checks: output = questions, bytes = the
 * plans, cones (TR_EQUIV_MAX_CONE entries per question), pair claims and scratch (BOOL-001's masks, GEOM-001's pairs
 * and FND-007's record check the same budget; CUDA: tr_equiv_device_need); decisions: support = the points enumerated
 * or of one grid, bytes = their synthetic rows, circuit = BDD nodes, iterations = NULL patterns + cubes + branch
 * combinations (each synthetic evaluation is budgeted as a snapshot check); runs: states = the distinct queries
 * (SHARED) or 1, output = the given queries, bytes.
 * Representation limits (TR_MATH_UNSUPPORTED): more than 64 distinct conditions, 4096 parts or questions, a cone of
 * TR_EQUIV_MAX_CONE nodes, measures of one check over more than 8 columns or 256 pairs (GEOM-001). Errors:
 * TR_MATH_INVALID, _MEMORY, _OVERFLOW, _BUDGET, _UNSUPPORTED, _DOMAIN_ERROR, _STALE. Objects are immutable and
 * individually owned; constructors return NULL and set *error. */
#ifndef TORBLES_MATH_EQUIV_H
#define TORBLES_MATH_EQUIV_H
#include <stdint.h>
#include <string.h>
#include "math_num.h"
#include "math_budgets.h"
#include "math_scope.h"
#include "math_bool.h"
#include "math_certificates.h"
#include "math_ident.h"
#include "math_geom.h"
#ifdef __cplusplus
extern "C" {
#endif

#define TR_EQUIV_ABI 1
#define TR_EQUIV_MAX_CONDITIONS 64          /* BOOL-001 ROWS masks */
#define TR_EQUIV_MAX_PARTS 4096
#define TR_EQUIV_MAX_QUESTIONS 4096
#define TR_EQUIV_MAX_CONE 128               /* nodes of one question's two cones (scratch per work item) */
#define TR_EQUIV_MAX_COLUMNS 32
#define TR_EQUIV_SLICE_WORDS 16             /* CUDA: selection words per row block, at least */
#define TR_EQUIV_MAX_SLICES 4096
#define TR_EQUIV_NONE UINT64_MAX

enum { TR_EQUIV_PREDICATE = 1, TR_EQUIV_MEASURE = 2, TR_EQUIV_QUERY = 3 };          /* question kinds */
enum { TR_EQUIV_EQUIVALENT = 1, TR_EQUIV_IMPLIES = 2 };                             /* relations */
enum { TR_EQUIV_FILTER = 1, TR_EQUIV_KLEENE = 2 };                                  /* predicate semantics */
enum { TR_EQUIV_SQL = 1, TR_EQUIV_SKIP = 2 };                                       /* measure NULL semantics */
enum { TR_EQUIV_UNKNOWN = 0, TR_EQUIV_PROVEN = 1, TR_EQUIV_REFUTED = 2 };           /* verdicts */
enum { TR_EQUIV_LABEL_NONE = 0, TR_EQUIV_LABEL_SNAPSHOT = 1, TR_EQUIV_LABEL_DOMAIN = 2, TR_EQUIV_LABEL_UNIVERSAL = 3,
       TR_EQUIV_LABEL_RELATIONS = 4 };
enum { TR_EQUIV_METHOD_ROWS = 1, TR_EQUIV_METHOD_ENUMERATE = 2, TR_EQUIV_METHOD_BDD = 3,
       TR_EQUIV_METHOD_NORMAL_FORM = 4, TR_EQUIV_METHOD_GRID = 5, TR_EQUIV_METHOD_REDUCTION = 6,
       TR_EQUIV_METHOD_CONSTANT = 7 };
enum { TR_EQUIV_WHY_NONE = 0, TR_EQUIV_WHY_BUDGET = 1, TR_EQUIV_WHY_UNSUPPORTED = 2, TR_EQUIV_WHY_UNDECIDED = 3 };
enum { TR_EQUIV_RANGE = 1, TR_EQUIV_VALUES = 2 };                                   /* domain columns */
enum { TR_EQUIV_ENUMERATE = 1, TR_EQUIV_DECIDE = 2 };                               /* domain modes */
enum { TR_EQUIV_SHARED = 1, TR_EQUIV_SEPARATE = 2 };                                /* family runs */
enum { TR_EQUIV_AUTH_OK = 0, TR_EQUIV_AUTH_NOT_PROVEN = 1, TR_EQUIV_AUTH_STALE = 2, TR_EQUIV_AUTH_OUTSIDE = 3,
       TR_EQUIV_AUTH_RELATIONS = 4, TR_EQUIV_AUTH_NO_SCOPE = 5 };                   /* authorization outcomes */
enum { TR_EQUIV_NEED_CPU = 0, TR_EQUIV_NEED_DEVICE = 1 };

/* ------------------------------------------------------------------ structs crossing the ctypes boundary */
typedef struct {
    int32_t nconds, nnodes, root, reserved;     /* root: the output node (< nnodes) */
    const tr_bool_condition *conds;             /* over scope columns */
    const tr_bool_node *nodes;                  /* VAR a: conds[a]; node i refers to nodes < i */
} tr_equiv_pred;                                /* 32 bytes */

typedef struct { int32_t filter, measure; } tr_equiv_query;   /* predicate / measure index of the set, or -1 */

typedef struct {
    int32_t kind, relation, semantics, nulls;   /* TR_EQUIV_* (semantics: predicates; nulls: measures, queries) */
    int32_t a, b;                               /* parts of the set (predicates, measures or queries) */
} tr_equiv_question;                            /* 24 bytes */

/* exhaustive evidence of one question over the rows of a scope */
typedef struct {
    uint64_t rows, selected;                    /* the scope's */
    uint64_t checked, excluded;                 /* rows decided; rows outside the SKIP domain */
    uint64_t differ, undecided;                 /* rows where the sides differ; rows not decided (width) */
    uint64_t witness, null_mismatch;            /* the smallest differing row (NONE); rows with one side NULL */
    uint64_t nulls;                             /* measures: rows where a side is NULL (SKIP: the excluded rows) */
    uint64_t cells[9];                          /* filter sides by truth: (a, b) in {TRUE, FALSE, UNKNOWN}^2 */
} tr_equiv_evidence;                            /* 144 bytes: checked + excluded + undecided = selected */

/* one question of a check as the CPU loop and the kernel read it */
typedef struct {
    int32_t kind, relation, semantics, nulls;
    int32_t ra, rb;                             /* root positions in the cone (-1: no filter, TRUE everywhere) */
    uint32_t cone, ncone;                       /* entries [cone, cone + ncone) */
    int32_t pair, hasa, hasb, constant;         /* GEOM-001 pair (-1: none); the sides carry a measure; pair -1
                                                 * with two measures: two constants, 1 when they differ */
    uint32_t supa, supb;                        /* the scope columns the measures read (a NULL there: NULL) */
} tr_equiv_plan;                                /* 56 bytes */

typedef struct { int32_t op, a, b, reserved; } tr_equiv_entry;   /* VAR a: condition; CONST a; else positions */

typedef struct {
    tr_scope_view rows;                         /* the scope (its selection; the measure columns) */
    tr_bool_domain masks;                       /* the set's conditions over the scope (BOOL-001 words) */
    tr_ident_cand_view cand;                    /* GEOM-001 pair candidates: F_j, G_j, P_j = D (f - g) */
    int32_t nq, reserved;
    uint64_t nentries;
    const tr_equiv_plan *plans;                 /* nq */
    const tr_equiv_entry *entries;              /* nentries */
} tr_equiv_check_view;                          /* 2272 bytes */

typedef struct {
    int32_t col, scale;                         /* scope column (as the set's parts name it), decimal scale */
    int32_t nullable, kind;                     /* 1: NULL is a value; TR_EQUIV_RANGE / TR_EQUIV_VALUES */
    int64_t lo, hi;                             /* RANGE: the logical integers lo..hi */
    uint64_t nvalues;                           /* VALUES: strictly increasing logical values */
    const int64_t *values;
} tr_equiv_column;                              /* 48 bytes */

typedef struct {
    int32_t ncols, mode;                        /* TR_EQUIV_ENUMERATE / TR_EQUIV_DECIDE */
    const tr_equiv_column *cols;
} tr_equiv_domain;                              /* 16 bytes */

typedef struct {
    int32_t verdict, label, method, why;        /* TR_EQUIV_*; why: the reason of an UNKNOWN */
    int32_t resource, point, reserved[2];       /* why BUDGET: the resource; point: 1 when a witness point exists */
    uint64_t need, limit;                       /* why BUDGET: the need reached and the limit */
    uint64_t work;                              /* rows checked / points enumerated / cubes explored / grid points */
    uint64_t nodes, patterns;                   /* BDD nodes built; NULL patterns decided */
    uint64_t witness;                           /* SNAPSHOT: the smallest differing row (NONE) */
} tr_equiv_verdict;                             /* 80 bytes */

typedef struct {
    int32_t authorized, reason;                 /* TR_EQUIV_AUTH_* */
    int32_t label, verdict;
    tr_cert_staleness stale;                    /* a stale snapshot record */
    tr_cert_decision cert;                      /* FND-007's decision of the proof's certificate (if any) */
} tr_equiv_decision;                            /* 112 bytes */

typedef struct {
    int32_t npred, nmeasure, nquery, ncond;     /* distinct parts */
    int32_t nnodes, ncols, reserved[2];         /* shared DAG nodes; scope columns read */
    uint64_t given_preds, given_conds, given_nodes, given_measures, given_queries;   /* before sharing */
    uint64_t filters, measures, pairs;          /* distinct among the queries */
    uint64_t shared_nodes;                      /* DAG nodes read by more than one predicate */
    uint64_t bytes;
} tr_equiv_set_info;                            /* 112 bytes */

typedef struct {
    uint64_t rows, nonnull;                     /* filtered rows (COUNT); among them the measure present */
    int32_t words, filter, measure, pair;       /* the sum's words; shared ids (-1) */
    int64_t denominator;                        /* SUM = value / denominator */
    uint64_t sum[TR_IDENT_MAX_WORDS];           /* the exact sum of D f, two's complement in `words` words */
} tr_equiv_answer;                              /* 104 bytes */

typedef struct {
    uint64_t scans, condition_rows, node_words, measure_rows;   /* what the run evaluated */
    uint64_t queries, filters, measures, pairs, bytes;
} tr_equiv_run_stats;                           /* 72 bytes */

typedef struct {
    int32_t changed, steps, explored, truncated;   /* tr_optimize's trace */
    double before_cost, after_cost;             /* its relative work units (not time) */
    int32_t ncols, reserved;
    int32_t col[TR_GEOM_MAX_VARS];              /* the scope columns of the rewritten measure */
    uint64_t nterms;                            /* its terms */
} tr_equiv_rewrite_info;                        /* 80 bytes */

typedef struct tr_equiv_set_s tr_equiv_set;
typedef struct tr_equiv_check_s tr_equiv_check;
typedef struct tr_equiv_result_s tr_equiv_result;

/* ------------------------------------------------------------------ shared helpers (CPU code, CUDA host and kernels) */
/* local accumulation of one question over some words */
typedef struct {
    uint64_t checked, excluded, differ, undecided, null_mismatch, nulls, witness;
    uint64_t cells[9];
} tr_equiv_acc;
TR_NUM_HD void tr_equiv_acc_init(tr_equiv_acc *a) {
    a->checked = a->excluded = a->differ = a->undecided = a->null_mismatch = a->nulls = 0;
    a->witness = TR_EQUIV_NONE;
    for (int i = 0; i < 9; i++) a->cells[i] = 0;
}
/* a measure side is NULL at a row: a scope column of its support is NULL (sup 0: a constant, never NULL) */
TR_NUM_HD int tr_equiv_null(const tr_scope_view *v, uint32_t sup, uint64_t row) {
    while (sup) {
        int c = tr_bool_ctz((uint64_t)sup);
        sup &= sup - 1;
        if (!tr_scope_present(v, c, row)) return 1;
    }
    return 0;
}
/* the measures of a question at one row (both sides present): 1 differ, 0 equal, -1 undecided */
TR_NUM_HD int tr_equiv_measure_row(const tr_equiv_check_view *v, const tr_equiv_plan *p, uint64_t row) {
    if (p->pair < 0) return p->constant ? 1 : 0;
    int32_t j = 3 * p->pair + 2;
    int32_t W = v->cand.words[j];
    if (W < 2 || W > TR_IDENT_MAX_WORDS) return -1;
    uint64_t acc[TR_IDENT_MAX_WORDS];
    tr_ident_cand_value(&v->rows, &v->cand, j, row, W, acc);
    return tr_ident_w_zero(acc, W) ? 0 : 1;
}
/* the rows of word w where the measures differ (sides present), with NULL handling; adds to the counters
 * excluded / null_mismatch / undecided; returns the differing bits, *ok the decided bits */
TR_NUM_HD uint64_t tr_equiv_measure_bits(const tr_equiv_check_view *v, const tr_equiv_plan *p, uint64_t rows,
                                         uint64_t w, tr_equiv_acc *a, uint64_t *ok) {
    uint64_t diff = 0, decided = 0;
    while (rows) {
        int b = tr_bool_ctz(rows);
        rows &= rows - 1;
        uint64_t r = w * 64 + (uint64_t)b, bit = (uint64_t)1 << b;
        int na = p->hasa && tr_equiv_null(&v->rows, p->supa, r);
        int nb = p->hasb && tr_equiv_null(&v->rows, p->supb, r);
        if (na || nb) {
            a->nulls++;
            if (p->nulls == TR_EQUIV_SKIP) { a->excluded++; continue; }
            decided |= bit;
            if (na != nb) { diff |= bit; a->null_mismatch++; }
            continue;
        }
        int s = tr_equiv_measure_row(v, p, r);
        if (s < 0) { a->undecided++; continue; }
        decided |= bit;
        if (s) diff |= bit;
    }
    *ok = decided;
    return diff;
}
/* question q at selection word w: its cone evaluated on the masks' words (val: 2 x ncone scratch), the differing
 * rows counted and the smallest kept */
TR_NUM_HD void tr_equiv_word(const tr_equiv_check_view *v, int32_t q, uint64_t w, uint64_t *val, tr_equiv_acc *a) {
    const tr_equiv_plan *p = &v->plans[q];
    const uint64_t sel = v->rows.selection[w];
    if (!sel) return;
    if (p->kind != TR_EQUIV_MEASURE) {
        for (uint32_t c = 0; c < p->ncone; c++) {
            const tr_equiv_entry *e = &v->entries[p->cone + c];
            uint64_t x, k;
            if (e->op == TR_BOOL_CONST) tr_bool_const_word(e->a, &x, &k);
            else if (e->op == TR_BOOL_VAR) tr_bool_var_word(&v->masks, e->a, w, &x, &k);
            else tr_bool_apply(e->op, val[2 * (uint64_t)e->a], val[2 * (uint64_t)e->a + 1],
                               e->b >= 0 ? val[2 * (uint64_t)e->b] : 0, e->b >= 0 ? val[2 * (uint64_t)e->b + 1] : 0, &x, &k);
            val[2 * (uint64_t)c] = x; val[2 * (uint64_t)c + 1] = k;
        }
    }
    uint64_t va = ~(uint64_t)0, ka = ~(uint64_t)0, vb = ~(uint64_t)0, kb = ~(uint64_t)0;
    if (p->kind != TR_EQUIV_MEASURE) {
        if (p->ra >= 0) { va = val[2 * (uint64_t)p->ra]; ka = val[2 * (uint64_t)p->ra + 1]; }
        if (p->rb >= 0) { vb = val[2 * (uint64_t)p->rb]; kb = val[2 * (uint64_t)p->rb + 1]; }
    }
    const uint64_t Ta = va & ka & sel, Fa = ~va & ka & sel, Ua = ~ka & sel;
    const uint64_t Tb = vb & kb & sel, Fb = ~vb & kb & sel, Ub = ~kb & sel;
    uint64_t diff = 0;
    if (p->kind != TR_EQUIV_MEASURE) {
        const uint64_t A[3] = {Ta, Fa, Ua}, B[3] = {Tb, Fb, Ub};
        for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) a->cells[3 * i + j] += (uint64_t)tr_bool_popc(A[i] & B[j]);
    }
    if (p->kind == TR_EQUIV_PREDICATE) {
        if (p->relation == TR_EQUIV_IMPLIES) diff = Ta & ~Tb;
        else if (p->semantics == TR_EQUIV_KLEENE) diff = (Ta ^ Tb) | (Fa ^ Fb);
        else diff = Ta ^ Tb;
        a->checked += (uint64_t)tr_bool_popc(sel);
    } else if (p->kind == TR_EQUIV_MEASURE) {
        uint64_t ok = 0;
        diff = tr_equiv_measure_bits(v, p, sel, w, a, &ok);
        a->checked += (uint64_t)tr_bool_popc(ok);
    } else {                                    /* QUERY: the filtered rows, then the measures on them */
        uint64_t rowdiff = p->relation == TR_EQUIV_IMPLIES ? (Ta & ~Tb) : (Ta ^ Tb);
        uint64_t both = Ta & Tb, ok = 0, mdiff = 0;
        uint64_t decided = sel & ~both;         /* rows filtered out on a side are decided by the filters */
        if (p->hasa && both) mdiff = tr_equiv_measure_bits(v, p, both, w, a, &ok);
        else ok = both;
        decided |= ok;
        diff = rowdiff | mdiff;
        a->checked += (uint64_t)tr_bool_popc(decided);
    }
    diff &= sel;
    if (diff) {
        a->differ += (uint64_t)tr_bool_popc(diff);
        uint64_t r = w * 64 + (uint64_t)tr_bool_ctz(diff);
        if (r < a->witness) a->witness = r;
    }
}
/* the device copies of tr_mgpu_equiv_check: the selection, the condition words, the columns the measures read, the
 * pair candidates, plans, entries and accumulators (the cone scratch is local: 2 TR_EQUIV_MAX_CONE words per item) */
static inline void tr_equiv_device_need(const tr_equiv_check_view *v, tr_math_need *need) {
    for (int i = 0; i < TR_BUDGET_RESOURCES; i++) need->need[i] = 0;
    uint64_t rows = v->rows.rows, nw = (rows + 63) / 64;
    uint64_t bytes = tr_budget_mul(nw, 8);
    bytes = tr_budget_add(bytes, tr_budget_mul(tr_budget_mul((uint64_t)(v->masks.k > 0 ? v->masks.k : 0), nw), v->masks.valid ? 16 : 8));
    uint64_t cols = 0;
    for (int32_t i = 0; i < v->cand.nvars && i < TR_IDENT_MAX_VARS; i++)
        if (v->cand.var[i] >= 0 && v->cand.var[i] < 64) cols |= (uint64_t)1 << v->cand.var[i];
    for (int32_t q = 0; q < v->nq; q++) cols |= (uint64_t)(v->plans[q].supa | v->plans[q].supb);
    for (int32_t c = 0; c < v->rows.ncols && c < TR_SCOPE_MAX_COLUMNS; c++) {
        if (!((cols >> c) & 1)) continue;
        bytes = tr_budget_add(bytes, tr_budget_mul(rows, v->rows.col[c].values ? 8 : 4));
        if (v->rows.col[c].valid) bytes = tr_budget_add(bytes, rows);
    }
    uint64_t nc = (uint64_t)(v->cand.ncand > 0 ? v->cand.ncand : 0);
    bytes = tr_budget_add(bytes, tr_budget_add(tr_budget_mul(v->cand.nterms, (uint64_t)v->cand.nvars + 8), tr_budget_mul(nc, 16)));
    uint64_t nq = (uint64_t)(v->nq > 0 ? v->nq : 0);
    bytes = tr_budget_add(bytes, tr_budget_mul(nq, sizeof(tr_equiv_plan) + sizeof(tr_equiv_evidence)));
    bytes = tr_budget_add(bytes, tr_budget_mul(v->nentries, sizeof(tr_equiv_entry)));
    need->need[TR_BUDGET_VARIABLES] = (uint64_t)(v->masks.k > 0 ? v->masks.k : 0);
    need->need[TR_BUDGET_SUPPORT] = v->cand.nterms;
    need->need[TR_BUDGET_OUTPUT] = nq;
    need->need[TR_BUDGET_BYTES] = tr_budget_add(bytes, 64);
}

/* ------------------------------------------------------------------ sets (libalg) */
tr_equiv_set *tr_equiv_set_make(int32_t npred, const tr_equiv_pred *preds, int32_t nmeasure, const tr_geom_poly *measures,
                                int32_t nquery, const tr_equiv_query *queries, const tr_math_budget *,
                                tr_math_refusal *, int32_t *error);
int32_t tr_equiv_set_get_info(const tr_equiv_set *, tr_equiv_set_info *);
/* per given part its shared id: predicates -> root node, measures -> distinct measure, queries -> distinct query;
 * per query also its distinct filter and measure (-1); arrays of the given counts (NULL to skip) */
int32_t tr_equiv_set_ids(const tr_equiv_set *, int32_t *pred_roots, int32_t *measure_ids, int32_t *query_ids,
                         int32_t *query_filters, int32_t *query_measures);
/* the distinct conditions (ncond) and the shared DAG (nnodes) */
int32_t tr_equiv_set_conditions(const tr_equiv_set *, tr_bool_condition *out, int32_t capacity);
int32_t tr_equiv_set_nodes(const tr_equiv_set *, tr_bool_node *out, int32_t capacity);
void tr_equiv_set_free(tr_equiv_set *);
/* a first-slice polynomial (mathematics.h; its variable v < 8 bound to scope column cols[v], -1 unused) as a
 * measure: the variables it reads become out_cols[ncols] (in variable order), exponents nterms x ncols, num / den;
 * capacity: terms */
struct tr_math_poly_s;
int32_t tr_equiv_poly_from_math(const struct tr_math_poly_s *, const int32_t *cols, int32_t *ncols, int32_t *out_cols,
                                uint8_t *exponents, int64_t *num, int64_t *den, uint64_t capacity, uint64_t *nterms);

/* ------------------------------------------------------------------ snapshot checks (libalg) */
/* questions over a scope: the masks of the set's conditions built here (value / valid NULL) or adopted from device
 * words (k x nw each, BOOL-001 layout); measure pairs by GEOM-001 */
tr_equiv_check *tr_equiv_check_make(const tr_equiv_set *, const tr_equiv_question *, int32_t nq, const tr_scope *,
                                    const uint64_t *value, const uint64_t *valid, const tr_math_budget *,
                                    tr_math_refusal *, int32_t *error);
int32_t tr_equiv_check_get_view(const tr_equiv_check *, tr_equiv_check_view *);
int32_t tr_equiv_check_need(const tr_equiv_check *, int32_t target, tr_math_need *);
/* the CPU evaluation of every question (out[nq]) */
int32_t tr_equiv_check_run(const tr_equiv_check *, tr_equiv_evidence *out);
/* the verdicts of a check on its scope (TR_MATH_STALE when it changed since the check was made; computed: a device
 * result to adopt after consistency checks, NULL: evaluated here; backend TR_CERT_BY_*); relations: FND-007
 * certificates tried first on measure questions with SKIP semantics or equal supports (GEOM-001) */
tr_equiv_result *tr_equiv_check_decide(const tr_equiv_check *, const tr_scope *, const tr_equiv_evidence *computed,
                                       int32_t backend, const tr_cert *const *relations, int32_t nrelations,
                                       const tr_math_budget *, tr_math_refusal *, int32_t *error);
void tr_equiv_check_free(tr_equiv_check *);

/* ------------------------------------------------------------------ universal decisions (libalg) */
tr_equiv_result *tr_equiv_decide(const tr_equiv_set *, const tr_equiv_question *, int32_t nq, const tr_equiv_domain *,
                                 const tr_math_budget *, tr_math_refusal *, int32_t *error);
int32_t tr_equiv_result_count(const tr_equiv_result *);
int32_t tr_equiv_result_get(const tr_equiv_result *, int32_t q, tr_equiv_verdict *out, tr_equiv_evidence *evidence);
/* the witness point of question q: values and presence per scope column 0..capacity-1 (columns not read: 0, 1) */
int32_t tr_equiv_result_point(const tr_equiv_result *, int32_t q, int64_t *values, uint8_t *present, int32_t capacity);
/* the FND-007 certificate of question q's proof (borrowed): a law (UNIVERSAL measures), a snapshot (SNAPSHOT
 * measures) or a derived certificate (RELATIONS); NULL otherwise */
const tr_cert *tr_equiv_result_certificate(const tr_equiv_result *, int32_t q);
/* whether question q's proof authorizes an exact rewrite on the current scope (NULL: universal proofs only) */
int32_t tr_equiv_authorize(const tr_equiv_result *, int32_t q, const tr_scope *current, tr_equiv_decision *out);
void tr_equiv_result_free(tr_equiv_result *);

/* ------------------------------------------------------------------ families (libalg) */
int32_t tr_equiv_family_run(const tr_equiv_set *, const tr_scope *, int32_t mode, const tr_math_budget *,
                            tr_equiv_answer *out, tr_equiv_run_stats *stats, tr_math_refusal *);

/* ------------------------------------------------------------------ the relationship optimizer (libalg) */
struct tr_catalog_s;
struct atable_s;
/* measure: an integer polynomial of degree <= 2 over scope columns; stored[c]: the table column of scope column c
 * (nscope entries); the rewritten measure over info->col (exponents nterms x info->ncols, coefficients) */
int32_t tr_equiv_rewrite(struct tr_catalog_s *, const struct atable_s *, const uint64_t *used, const tr_geom_poly *measure,
                         const int32_t *stored, int32_t nscope, tr_equiv_rewrite_info *info, uint8_t *exponents,
                         int64_t *coefficients, uint64_t capacity);

/* ------------------------------------------------------------------ the CUDA library (math_equiv_gpu.cu) */
int tr_mgpu_equiv_check(const tr_equiv_check_view *view, const tr_math_budget *budget, tr_equiv_evidence *out,
                        tr_math_refusal *refusal);
#ifdef __cplusplus
}
#endif
#endif
