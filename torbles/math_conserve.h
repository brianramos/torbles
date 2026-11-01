/* DISC-003: conservation and stability of invariants across revisions.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Tracker (tr_cons_tracker): the invariants of one lineage of revisions of a FND-004 scope (math_scope.h). It holds
 *   - the DISC-002 Gram state of a plan (math_ident.h; unweighted, every moment an exact signed 128-bit sum: the plan
 *     must be EXACT on the starting scope, else TR_MATH_OVERFLOW), kept current under DELTA-001 deltas (math_delta.h):
 *     G' = G + sum_r w_r m(x_r) m(x_r)^T over the delta rows r where every plan variable is present (w_r the row's
 *     weight: +1 insert, -1 delete, another value a weight change), proven before anything is added:
 *     |G_u| + W B_u <= 2^127 - 1 for every moment u, W = sum |w_r| over the delta and B_u = prod_v M_v^{e_uv} from the
 *     delta's column extremes (M_v = max(|min|, |max|, 1)); otherwise TR_MATH_OVERFLOW (nothing wraps). The Gram rows
 *     (the bag multiplicity of the complete rows) are kept as an exact int64 sum;
 *   - a companion DELTA-001 state (a FND-004 JOINT bank of the constant and the plan variables, promise NONNEGATIVE)
 *     updated by the same delta with tr_delta_state_apply: it decides whether a delta applies (column count, scales,
 *     the schema of a SCOPE or DIFF delta, a DIFF from exactly the tracker's current scope) and refuses a delete of rows
 *     that are not there (TR_MATH_DOMAIN_ERROR with the tr_delta_violation), and after a DIFF it is bound to the new
 *     scope (tr_delta_state_bound_to). The Gram-row count must equal its constant term's pooled count (checked);
 *   - up to TR_CONS_MAX_IDENTITIES tracked identities (the representable, resolved items of a DISC-002 tr_ident_set,
 *     each a primitive int64 polynomial over its support columns) with their FND-007 evidence over the current bag
 *     (checked, excluded (a NULL support column: policy SKIP), violations, overflow: sums of multiplicities);
 *   - per identity a persistence record and the evidence classes below; per revision a step record.
 *
 * Revisions. Step 0 is the build: every identity is verified exhaustively on the scope (DISC-002's W-word evaluation)
 * and each verified one certified (FND-007 snapshot, tr_ident_certify). An update (tr_cons_tracker_update) is step
 * s + 1: the delta's rows alone are evaluated (the incremental verification: per identity the weights of the checked,
 * excluded, violating and undecided rows, inserted and deleted separately, and the smallest violating inserted and
 * deleted delta rows: tr_cons_delta_evidence, on the CPU or with tr_mgpu_cons_delta, equal bit for bit) and added to
 * the evidence: checked' = checked + checked_ins - checked_del, and so on (a negative result is a delete of rows the
 * evidence does not hold: TR_MATH_DOMAIN_ERROR). Each identity's evaluation width is proven from the delta's extremes
 * (sum |c_t| prod M_v^e <= 2^(64 W - 1) - 1, W <= 8; else its rows are undecided and the identity UNDECIDED until a
 * rebind). The new revision is
 *   BOUND    after a DIFF (tr_delta_between(current scope, new scope)) with the new scope given (the companion must be
 *            bound to exactly it): the evidence is that scope's exhaustive evidence, by induction (complete at the build,
 *            and a DIFF lists exactly the rows whose contribution changed); a holding identity becomes an FND-007
 *            snapshot certificate of the new scope (tr_cert_adopt when the scope's extremes prove W = 2, else
 *            FND-007's own exhaustive check, which may refuse past 128 bits: certificate code TR_MATH_OVERFLOW);
 *   UNBOUND  after an ARRAYS or SCOPE delta: the bag has no scope, so nothing is certified; holding identities are
 *            HELD (a record only). tr_cons_tracker_rebind(scope) recomputes everything from a scope (the Gram, the
 *            exhaustive evidence, the companion) as a new step and binds the lineage to it again.
 * Statuses of an identity at a revision: CERTIFIED (bound, holds, certificate made), VERIFIED (bound, holds, FND-007
 * refused the certificate), HELD (unbound, holds), FAILED (a violating row), UNDECIDED (undecided rows).
 *
 * Persistence record (tr_cons_record): the steps certified (counts, first, last, the maximal runs of consecutive
 * certified steps, the current streak), held, failed and undecided; the first failure: its step, the table revision,
 * the number of violating rows and the witness row (step 0 or a rebind: the smallest violating physical row of the
 * scope; an update: the smallest violating inserted delta row, else the smallest violating deleted one) with its
 * values on the identity's support columns. A record is a record: it states what held on the revisions it lists, never
 * what holds on another revision.
 *
 * Evidence classes, separate and never converted into one another (tr_cons_tracker_classes, tr_cons_authorize):
 *   DECLARED    a schema guarantee: the identity was declared (a DISC-002 DECLARED item, an item labelled known, or the
 *               caller's declaration at build), and FND-007 SCHEMA certificates of facts about its columns given with
 *               the call. A declaration is not verified evidence: the identity is still verified on every revision and
 *               a violation is reported as such.
 *   SNAPSHOT    the FND-007 snapshot certificate of the current revision, made by the tracker's own verification only;
 *               it describes that one revision.
 *   STABILITY   the persistence record (held on steps s1..sk).
 *   HYPOTHESIS  FND-007 HYPOTHESIS certificates of the identity's claim given with the call (DISC-002
 *               tr_ident_hypothesis: a sample); never counted as a certified revision.
 * Certificates given with a call are sorted into these classes by their FND-007 kind; a snapshot certificate (not the
 * tracker's own), a law (a nonzero identity is never a ring identity), a hypothesis of another polynomial or a schema
 * fact about another column is refused (TR_MATH_DOMAIN_ERROR). Authorization of an exact rewrite on a current scope:
 * only the snapshot certificate, when FND-007 authorizes it (fresh and exact on that scope); a FAILED identity never.
 * Otherwise refused with the class that was available:
 * STABILITY ("a record of the steps it held on, not a proof about this revision"), STATISTICAL (a hypothesis),
 * DECLARED, FAILED, NONE; a stale snapshot reports FND-007's reasons (TR_MATH_STALE).
 *
 * Conservation (tr_cons_law): a declared transformation T of n <= 8 variables (each variable's image a polynomial
 * T_v over the variables; an empty image is the identity) and a quantity q (an int64 polynomial): the composition
 * P = q(T(x)) - q(x) is expanded exactly (128-bit coefficients, every product checked; past int64 at the end or past
 * the support budget: refused). When P is the zero polynomial the FND-007 law RING_IDENTITY of the claim
 * "q(T(x)) - q(x)" (its expanded terms, which FND-007 canonicalizes to zero itself) is the universal certificate that T
 * conserves q. Otherwise T does not conserve q on Z^n: a concrete counterexample assignment is searched on the grid
 * prod_v S_v, S_v the first deg_v(P) + 1 integers of 0, 1, -1, 2, -2, ... (a nonzero P has a non-root there: the
 * combinatorial Nullstellensatz), each point evaluated exactly in W words (bound proven from the grid; W <= 8, else
 * the search is not run); budget: iterations = grid points. On data (variables bound to scope columns), P = 0 is
 * verified exhaustively on every selected row (DISC-002 candidates: evidence, the FND-007 snapshot certificate when it
 * holds) and tr_cons_law_witnesses lists the first violating rows (the perturbations that break conservation).
 *
 * Budgets (math_budgets.h), checked before allocation: tracker builds and updates: variables, support = basis
 * monomials, states = moments + identities, output = identities, bytes = the tracker (moments, identities, records,
 * runs, steps) plus the DISC-002 plan, Gram and set and the companion state; a delta job: bytes = its words and output
 * arrays (CUDA: tr_cons_delta_device_need); conservation: variables = n, degree, support = the largest expansion
 * (before it is formed), iterations = grid points, bytes. Errors: TR_MATH_INVALID, _MEMORY, _OVERFLOW, _BUDGET,
 * _UNSUPPORTED (k > 8, more than 256 identities, a weighted plan), _DOMAIN_ERROR (an incompatible delta, a negative
 * multiplicity, an attached certificate of another claim or kind, a scope the companion is not bound to). Objects are
 * immutable and individually owned; constructors return NULL and set *error. */
#ifndef TORBLES_MATH_CONSERVE_H
#define TORBLES_MATH_CONSERVE_H
#include <stdint.h>
#include <string.h>
#include "math_num.h"
#include "math_budgets.h"
#include "math_scope.h"
#include "math_bank.h"
#include "math_certificates.h"
#include "math_ident.h"
#include "math_delta.h"
#ifdef __cplusplus
extern "C" {
#endif

#define TR_CONS_ABI 1
#define TR_CONS_MAX_VARS 8
#define TR_CONS_MAX_IDENTITIES 256          /* = TR_IDENT_MAX_CANDIDATES */
#define TR_CONS_MAX_STEPS (UINT64_C(1) << 24)
#define TR_CONS_MAX_TERMS 4096              /* the expansion of a conservation check (FND-007 claims: 4096 terms) */
#define TR_CONS_CHUNK 32                    /* CUDA: moments per work item */
#define TR_CONS_CAND_CHUNK 16               /* CUDA: identities per work item */
#define TR_CONS_SLICE_ROWS 2048             /* CUDA: delta rows per slice, at least */
#define TR_CONS_MAX_SLICES 2048
#define TR_CONS_NONE UINT64_MAX

enum { TR_CONS_PENDING = 0, TR_CONS_CERTIFIED = 1, TR_CONS_VERIFIED = 2, TR_CONS_HELD = 3, TR_CONS_FAILED = 4,
       TR_CONS_UNDECIDED = 5, TR_CONS_UNTRACKED = 6 };                         /* statuses */
enum { TR_CONS_STEP_BUILD = 1, TR_CONS_STEP_DIFF = 2, TR_CONS_STEP_SCOPE = 3, TR_CONS_STEP_ARRAYS = 4,
       TR_CONS_STEP_REBIND = 5 };                                              /* step kinds */
enum { TR_CONS_WITNESS_NONE = 0, TR_CONS_WITNESS_SCOPE = 1, TR_CONS_WITNESS_INSERTED = 2,
       TR_CONS_WITNESS_DELETED = 3 };                                          /* where a witness row lives */
enum { TR_CONS_CLASS_NONE = 0, TR_CONS_CLASS_DECLARED = 1, TR_CONS_CLASS_SNAPSHOT = 2, TR_CONS_CLASS_STABILITY = 3,
       TR_CONS_CLASS_HYPOTHESIS = 4 };                                         /* evidence classes */
enum { TR_CONS_OK = 0, TR_CONS_REFUSE_STABILITY = 1, TR_CONS_REFUSE_STATISTICAL = 2, TR_CONS_REFUSE_DECLARED = 3,
       TR_CONS_REFUSE_FAILED = 4, TR_CONS_REFUSE_SNAPSHOT = 5, TR_CONS_REFUSE_NONE = 6,
       TR_CONS_REFUSE_UNTRACKED = 7 };                                         /* authorization outcomes */
enum { TR_CONS_NEED_CPU = 0, TR_CONS_NEED_DEVICE = 1 };

/* ------------------------------------------------------------------ structs crossing the ctypes boundary */
/* the incremental evidence of one identity over the rows of one delta (multiplicities: sums of |weight|) */
typedef struct {
    uint64_t checked_ins, checked_del;          /* support present: evaluated */
    uint64_t excluded_ins, excluded_del;        /* a support column NULL (policy SKIP) */
    uint64_t violations_ins, violations_del;    /* evaluated and nonzero */
    uint64_t overflow_ins, overflow_del;        /* not decided (the evaluation width is not proven) */
    uint64_t witness_ins, witness_del;          /* the smallest violating delta row with weight > 0 / < 0, or NONE */
} tr_cons_delta_evidence;                       /* 80 bytes */

/* what a delta job computes (the CPU loop or tr_mgpu_cons_delta): the Gram's moment increments and the evidence */
typedef struct {
    int32_t nmoments, nident;
    uint64_t gram_ins, gram_del;                /* sums of |weight| over the delta's Gram rows, by sign */
    int64_t *acc;                               /* nmoments x 2 words (lo, hi): sum_r w_r u(x_r), exact */
    tr_cons_delta_evidence *ev;                 /* nident */
} tr_cons_delta_out;                            /* 40 bytes */

/* a delta job as the CPU loop and the kernels read it (pointers into the delta and the job) */
typedef struct {
    tr_scope_view rows;                         /* the delta's rows (tr_delta_get_view) */
    const int64_t *weights;                     /* [rows.rows] */
    tr_ident_moment_view plan;                  /* the tracker's basis and moments (EXACT) */
    tr_ident_cand_view cand;                    /* the tracked identities, words proven for the delta's extremes */
} tr_cons_job_view;                             /* 2296 bytes */

typedef struct {
    int32_t nvars, nbasis, nmoments, nident;
    int32_t tracked, bound, degree, ncols;      /* identities tracked; the current revision has a scope */
    uint64_t step, revision, domain, bytes;     /* step; its table revision (0 unbound); Gram rows; bytes owned */
    uint64_t certified, verified, held, failed, undecided;   /* identities by current status */
    uint64_t scope_digest[2];                   /* the current scope's descriptor digest (0 unbound) */
    uint64_t state_identity[2];                 /* the companion DELTA-001 state's identity */
} tr_cons_info;                                 /* 136 bytes */

typedef struct {
    int32_t index, status;                      /* TR_CONS_* at the current revision */
    int32_t tracked, declared;                  /* declared: the DECLARED class */
    int32_t snapshot, certificate_code;         /* 1: a snapshot certificate of the current revision; 0 or why refused */
    uint64_t certified, verified, held, failed, undecided;   /* steps by status, since step 0 */
    uint64_t first_certified, last_certified;   /* steps (TR_CONS_NONE: never) */
    uint64_t runs, streak;                      /* maximal runs of consecutive certified steps; the current run */
    uint64_t fail_step, fail_revision;          /* the first failure (TR_CONS_NONE: never failed) */
    uint64_t fail_violations, fail_row;         /* violating rows there; the witness row (TR_CONS_WITNESS_*) */
    int32_t fail_source, fail_ncols;
    int64_t fail_weight;                        /* the witness delta row's weight (1 for a scope row) */
    int32_t fail_col[TR_CONS_MAX_VARS];         /* the support columns of the recorded values */
    int64_t fail_value[TR_CONS_MAX_VARS];
    uint8_t fail_valid[TR_CONS_MAX_VARS];
    uint64_t witness_ins, witness_del;          /* the last update's smallest violating inserted / deleted delta rows */
    tr_cert_evidence evidence;                  /* the current bag (rows, selected: the bound scope's; unbound:
                                                   0 and the bag's multiplicity) */
} tr_cons_record;                               /* 328 bytes */

typedef struct {
    uint64_t step, revision;                    /* the step; its scope's table revision (0 unbound, arrays) */
    uint64_t rows, inserted, deleted;           /* delta rows and rows by weight sign (build: the selected rows) */
    int32_t kind, bound;                        /* TR_CONS_STEP_*; 1: the step's bag is a scope */
    uint64_t certified, failed;                 /* identities certified / failed at the step */
    uint64_t domain;                            /* Gram rows after the step */
    uint64_t digest[2];                         /* the scope's descriptor digest (bound) */
    uint64_t delta[2];                          /* the delta's identity (updates) */
} tr_cons_step;                                 /* 104 bytes */

typedef struct {
    int32_t declared, snapshot, stability, hypothesis;   /* 1: that class holds something (hypothesis: a count) */
    int32_t schema, status, reserved[2];        /* schema: FND-007 schema facts given (in DECLARED) */
    uint64_t certified, snapshot_step;          /* certified steps (the record); the snapshot's step (NONE) */
    uint64_t hypothesis_size;                   /* rows drawn by the first hypothesis (0) */
} tr_cons_classes;                              /* 56 bytes */

typedef struct {
    int32_t authorized, reason;                 /* TR_CONS_OK or TR_CONS_REFUSE_*; class: TR_CONS_CLASS_* used */
    int32_t cls, status;
    tr_cert_decision snapshot;                  /* FND-007's decision of the snapshot (reason 0 when none) */
} tr_cons_decision;                             /* 80 bytes */

typedef struct {
    int32_t nvars, conserved;                   /* 1: P = q(T(x)) - q(x) is the zero polynomial */
    int32_t certified, searched;                /* a law certificate exists; the counterexample search ran */
    int32_t found, words;                       /* a counterexample point; the words of its value */
    int32_t degree, reserved;                   /* P's total degree */
    uint64_t q_terms, expanded_terms, residual_terms, points;   /* sizes; grid points evaluated */
    int64_t point[TR_CONS_MAX_VARS];            /* the counterexample assignment */
    uint64_t value[TR_IDENT_MAX_WORDS];         /* P(point), two's complement in `words` words */
    uint64_t bytes;
} tr_cons_law_info;                             /* 200 bytes */

typedef struct tr_cons_tracker_s tr_cons_tracker;
typedef struct tr_cons_job_s tr_cons_job;
typedef struct tr_cons_law_s tr_cons_law;

/* ------------------------------------------------------------------ shared helpers (CPU code, CUDA host and kernels) */
/* evaluation words of an int64 polynomial (terms t0..t1 of a candidate view) under column magnitudes M[v] (the
 * variables of the view): the smallest W in 2..TR_IDENT_MAX_WORDS with sum |c| prod M^e <= 2^(64 W - 1) - 1, else 0 */
static inline int32_t tr_cons_words(const tr_ident_cand_view *c, int32_t j, const uint64_t *M) {
    enum { WW = TR_IDENT_MAX_WORDS + 2 };
    uint64_t total[WW], term[WW];
    for (int32_t w = 0; w < WW; w++) total[w] = 0;
    for (uint64_t t = c->offset[j]; t < c->offset[j + 1]; t++) {
        for (int32_t w = 0; w < WW; w++) term[w] = 0;
        term[0] = tr_abs_u64(c->coefficients[t]);
        const uint8_t *e = c->exponents + t * (uint64_t)c->nvars;
        for (int32_t i = 0; i < c->nvars; i++)
            for (uint8_t q = 0; q < e[i]; q++) {
                unsigned __int128 carry = 0;
                for (int32_t w = 0; w < WW; w++) {
                    unsigned __int128 x = (unsigned __int128)term[w] * M[i] + carry;
                    term[w] = (uint64_t)x; carry = x >> 64;
                }
                if (carry || term[WW - 1]) return 0;
            }
        uint64_t cy = 0;
        for (int32_t w = 0; w < WW; w++) {
            uint64_t s = total[w] + term[w], c1 = s < term[w], s2 = s + cy, c2 = s2 < cy;
            total[w] = s2; cy = c1 | c2;
        }
        if (cy || total[WW - 1]) return 0;
    }
    uint64_t bits = 0;
    for (int32_t w = WW; w-- > 0;)
        if (total[w]) { bits = (uint64_t)w * 64 + (uint64_t)(64 - __builtin_clzll(total[w])); break; }
    for (int32_t W = 2; W <= TR_IDENT_MAX_WORDS; W++) if (bits <= (uint64_t)(64 * W - 1)) return W;
    return 0;
}
/* the magnitude bound max(|min|, |max|, 1) of a view column */
TR_NUM_HD uint64_t tr_cons_mag(const tr_scope_view *v, int32_t c) {
    const tr_scope_view_column *k = &v->col[c];
    uint64_t m = k->nvalid ? (tr_abs_u64(k->min) > tr_abs_u64(k->max) ? tr_abs_u64(k->min) : tr_abs_u64(k->max)) : 0;
    return m < 1 ? 1 : m;
}
/* delta row r is a Gram row of the plan (every plan variable present) */
TR_NUM_HD int tr_cons_gram_row(const tr_scope_view *v, const tr_ident_moment_view *m, uint64_t r) {
    for (int32_t i = 0; i < m->nvars; i++) if (!tr_scope_present(v, m->var[i], r)) return 0;
    return 1;
}
/* identity j at delta row r of weight w: adds |w| to its counters (one evidence record) */
TR_NUM_HD void tr_cons_eval_row(const tr_scope_view *v, const tr_ident_cand_view *c, int32_t j, uint64_t r, int64_t w,
                                tr_cons_delta_evidence *e) {
    if (!w) return;
    uint64_t m = tr_abs_u64(w);
    int st = tr_ident_cand_row(v, c, j, r);
    int ins = w > 0;
    if (st == TR_CERT_ROW_EXCLUDED) { if (ins) e->excluded_ins += m; else e->excluded_del += m; return; }
    if (st == TR_CERT_ROW_OVERFLOW) { if (ins) e->overflow_ins += m; else e->overflow_del += m; return; }
    if (ins) e->checked_ins += m; else e->checked_del += m;
    if (st == TR_CERT_ROW_VIOLATES) {
        if (ins) { e->violations_ins += m; if (r < e->witness_ins) e->witness_ins = r; }
        else { e->violations_del += m; if (r < e->witness_del) e->witness_del = r; }
    }
}
TR_NUM_HD void tr_cons_evidence_init(tr_cons_delta_evidence *e) {
    e->checked_ins = e->checked_del = e->excluded_ins = e->excluded_del = 0;
    e->violations_ins = e->violations_del = e->overflow_ins = e->overflow_del = 0;
    e->witness_ins = e->witness_del = TR_CONS_NONE;
}

/* the device copies of tr_mgpu_cons_delta: the delta's columns the plan and the identities read (values and
 * validity), the weights and the selection, the plan and candidate arrays, the accumulators and evidence */
static inline void tr_cons_delta_device_need(const tr_cons_job_view *j, tr_math_need *need) {
    for (int i = 0; i < TR_BUDGET_RESOURCES; i++) need->need[i] = 0;
    const tr_scope_view *v = &j->rows;
    uint64_t rows = v->rows, bytes = tr_budget_add(tr_budget_bitmap_bytes(rows), tr_budget_mul(rows, 8));
    uint32_t cols = 0;
    for (int32_t i = 0; i < j->plan.nvars; i++) if (j->plan.var[i] >= 0 && j->plan.var[i] < 32) cols |= (uint32_t)1 << j->plan.var[i];
    for (int32_t i = 0; i < j->cand.nvars; i++) if (j->cand.var[i] >= 0 && j->cand.var[i] < 32) cols |= (uint32_t)1 << j->cand.var[i];
    for (int32_t c = 0; c < v->ncols && c < TR_SCOPE_MAX_COLUMNS; c++) {
        if (!((cols >> c) & 1)) continue;
        bytes = tr_budget_add(bytes, tr_budget_mul(rows, v->col[c].values ? 8 : 4));
        if (v->col[c].valid) bytes = tr_budget_add(bytes, rows);
    }
    uint64_t nb = (uint64_t)(j->plan.nbasis > 0 ? j->plan.nbasis : 0), nm = (uint64_t)(j->plan.nmoments > 0 ? j->plan.nmoments : 0);
    uint64_t n = (uint64_t)(j->cand.ncand > 0 ? j->cand.ncand : 0);
    bytes = tr_budget_add(bytes, tr_budget_mul(nb, (uint64_t)j->plan.nvars + 8));
    bytes = tr_budget_add(bytes, tr_budget_mul(nm, 8 + 16));
    bytes = tr_budget_add(bytes, tr_budget_mul(j->cand.nterms, (uint64_t)j->cand.nvars + 8));
    bytes = tr_budget_add(bytes, tr_budget_mul(n, 8 + 4 + 4 + sizeof(tr_cons_delta_evidence)));
    need->need[TR_BUDGET_VARIABLES] = (uint64_t)(j->plan.nvars > j->cand.nvars ? j->plan.nvars : j->cand.nvars);
    need->need[TR_BUDGET_SUPPORT] = nb > j->cand.nterms ? nb : j->cand.nterms;
    need->need[TR_BUDGET_STATES] = tr_budget_add(nm, n);
    need->need[TR_BUDGET_OUTPUT] = n;
    need->need[TR_BUDGET_BYTES] = tr_budget_add(bytes, 64);
}

/* ------------------------------------------------------------------ trackers (libalg) */
/* The tracker of a scope: the DISC-002 plan of spec (unweighted, EXACT on the scope), its Gram (CPU), the identities of
 * set (a DISC-002 set over this scope; NULL: none) verified and certified exhaustively, declared[i] (NULL: from the
 * set's labels) and the companion DELTA-001 state. */
tr_cons_tracker *tr_cons_tracker_make(const tr_scope *scope, const tr_ident_spec *spec, const tr_ident_set *set,
                                      const int32_t *declared, const tr_math_budget *, tr_math_refusal *,
                                      int32_t *error);
/* the job of updating a tracker by a delta (the delta must outlive the job): the update proof and the evaluation
 * widths are decided here, before anything is evaluated */
tr_cons_job *tr_cons_job_make(const tr_cons_tracker *, const tr_delta *, const tr_math_budget *, tr_math_refusal *,
                              int32_t *error);
int32_t tr_cons_job_get_view(const tr_cons_job *, tr_cons_job_view *);
/* the needs of running the job on the CPU or the device */
int32_t tr_cons_job_need(const tr_cons_job *, int32_t target, tr_math_need *);
/* the CPU evaluation into caller arrays (out->acc: nmoments x 2 words, out->ev: nident records) */
int32_t tr_cons_job_run(const tr_cons_job *, tr_cons_delta_out *out);
void tr_cons_job_free(tr_cons_job *);
/* The next revision: the job's result (computed = NULL: evaluated here on the CPU; else a device result, adopted after
 * consistency checks; backend TR_CERT_BY_*), the companion updated by the job's delta, and, after a DIFF, the new
 * scope (required; the companion must be bound to it). violation: the DELTA-001 refusal of a negative multiplicity. */
tr_cons_tracker *tr_cons_tracker_update(const tr_cons_tracker *, const tr_cons_job *, const tr_cons_delta_out *computed,
                                        int32_t backend, const tr_scope *new_scope, const tr_math_budget *,
                                        tr_math_refusal *, tr_delta_violation *violation, int32_t *error);
/* a new step recomputed from a scope of the same schema (Gram, exhaustive evidence, companion): the lineage bound */
tr_cons_tracker *tr_cons_tracker_rebind(const tr_cons_tracker *, const tr_scope *, const tr_math_budget *,
                                        tr_math_refusal *, int32_t *error);
int32_t tr_cons_tracker_get_info(const tr_cons_tracker *, tr_cons_info *);
int32_t tr_cons_tracker_record(const tr_cons_tracker *, int32_t i, tr_cons_record *out);
/* the certified runs of identity i as [first, last] step pairs: writes min(runs, capacity) pairs, returns runs */
int64_t tr_cons_tracker_runs(const tr_cons_tracker *, int32_t i, uint64_t *pairs, uint64_t capacity);
int64_t tr_cons_tracker_steps(const tr_cons_tracker *, tr_cons_step *out, uint64_t capacity);
/* the current revision's snapshot certificate of identity i (borrowed; NULL when none) */
const tr_cert *tr_cons_tracker_certificate(const tr_cons_tracker *, int32_t i);
/* identity i's FND-007 POLYNOMIAL claim (pointers into the tracker) and its terms */
int32_t tr_cons_tracker_claim(const tr_cons_tracker *, int32_t i, tr_cert_claim *out);
/* identity i's evidence classes, with the certificates given (n of them: hypotheses, laws, schema facts) */
int32_t tr_cons_tracker_classes(const tr_cons_tracker *, int32_t i, const tr_cert *const *given, int32_t n,
                                tr_cons_classes *out);
/* the exact rewrite decision for identity i on a current scope (field: 0 = Z, p = GF(p)) with the certificates given */
int32_t tr_cons_authorize(const tr_cons_tracker *, int32_t i, const tr_cert *const *given, int32_t n,
                          const tr_scope *current, uint64_t field, tr_cons_decision *out);
/* the maintained Gram: its moment exponents (nmoments x nvars) and exact sums (nmoments x 2 words) */
int32_t tr_cons_tracker_moments(const tr_cons_tracker *, uint8_t *exponents, int64_t *sums, uint64_t capacity);
/* the DISC-002 Gram of the current revision adopted on its scope (the companion bound to it; a plan EXACT on it) */
tr_ident_gram *tr_cons_tracker_gram(const tr_cons_tracker *, const tr_scope *, const tr_math_budget *,
                                    tr_math_refusal *, int32_t *error);
/* the companion DELTA-001 state (borrowed) */
const tr_delta_state *tr_cons_tracker_state(const tr_cons_tracker *);
void tr_cons_tracker_free(tr_cons_tracker *);

/* ------------------------------------------------------------------ conservation (libalg) */
/* q and the images T_v (images[v].nterms == 0: x_v) over n variables (exponents nterms x n) */
tr_cons_law *tr_cons_conserve(int32_t n, const tr_ident_poly *q, const tr_ident_poly *images,
                              const tr_math_budget *, tr_math_refusal *, int32_t *error);
int32_t tr_cons_law_get_info(const tr_cons_law *, tr_cons_law_info *);
/* P's canonical terms (decreasing graded order): exponents nterms x n, coefficients */
int32_t tr_cons_law_residual(const tr_cons_law *, uint8_t *exponents, int64_t *coefficients, uint64_t capacity,
                             uint64_t *nterms);
/* the universal certificate (borrowed; NULL when q is not conserved) */
const tr_cert *tr_cons_law_certificate(const tr_cons_law *);
/* the DISC-002 candidate of P = 0 over scope columns cols[n] (-1: a variable P does not read) */
tr_ident_set *tr_cons_law_candidates(const tr_cons_law *, const tr_scope *, const int32_t *cols,
                                     const tr_math_budget *, tr_math_refusal *, int32_t *error);
/* exhaustive evidence of P = 0 on the scope's selected rows and the first violating rows (rows[capacity]) */
int32_t tr_cons_law_witnesses(const tr_cons_law *, const tr_scope *, const int32_t *cols, const tr_math_budget *,
                              tr_cert_evidence *evidence, uint64_t *rows, uint64_t capacity, uint64_t *found,
                              tr_math_refusal *);
void tr_cons_law_free(tr_cons_law *);

/* ------------------------------------------------------------------ the CUDA library (math_conserve_gpu.cu) */
/* the job's evaluation on the device (out arrays on the host: out->acc nmoments x 2, out->ev nident) */
int tr_mgpu_cons_delta(const tr_cons_job_view *job, const tr_math_budget *budget, tr_cons_delta_out *out,
                       tr_math_refusal *refusal);
#ifdef __cplusplus
}
#endif
#endif
