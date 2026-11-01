/* GEOM-002: bounded elimination and generator selection.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Relations are GEOM-001's (math_geom.h): FND-007 certificates of polynomial facts about the columns of one table (or
 * array scope), at most TR_ELIM_MAX_RELATIONS over at most TR_ELIM_MAX_VARS columns, one NULL policy. Their ideal I
 * holds every polynomial they imply. Variables V (scope columns; default: the relations' columns) are to be covered.
 *
 * Generators (tr_elim_select). A set G of V is a generator set when every eliminated variable x of E = V \ G has a
 * reconstruction x = f_x(G): a polynomial in G alone with x - f_x in I. It is decided exactly by elimination: under a
 * lex order with E before G (GEOM-001's reduced Groebner basis, complete), the normal form of x lies in Q[G] exactly
 * when such an f_x exists (the elimination property: a polynomial of Q[G] reduces within Q[G]), and then f_x is that
 * normal form; its FND-007 certificate D (x - f_x) = sum D h_k r_k is GEOM-001's derived reduction certificate. A
 * reconstruction is only kept when it is checked on the scope's rows (tr_elim_check: values, NULL mask and range), never
 * assumed from the algebra: a relation certified under the policy SKIP holds on complete rows only, so a column NULL
 * where its generators are present is not reconstructed by them. Two ways to find G:
 *   ORDER    one elimination (one solve) under a given lex sequence: E = the variables whose normal form differs from
 *            themselves (the linear leading monomials of the reduced basis) and whose check holds; G = the rest.
 *   SEARCH   a bounded search by increasing size k = 0, 1, ...: every k-subset G of the relation variables (variables
 *            in no relation are always generators), in lexicographic order of positions, is one solve (one completion
 *            under lex E > G, then the reductions of E); the first size with a set whose every reconstruction exists and
 *            checks is the minimum; with rank_all every set of that size is costed and the cheapest chosen by the
 *            priority (ties to the earliest), else the first is taken. Nothing feasible below max_size + 1 leaves G = V
 *            (nothing eliminated). A set whose solve cannot be completed exactly within the budget is UNDECIDED, never
 *            guessed: refused by the degree / support budget (or GEOM-001's per-solve limits) it is rejected with
 *            reason BUDGET and its refusal, past int64 (or a GEOM-001 representation limit) with reason UNCLAIMABLE;
 *            the search goes on. minimal = 1 only when every smaller set was decided infeasible (no undecided set
 *            below the chosen size; for G = V, none at all and every size searched).
 *
 * Costs (tr_elim_cost), measured on the scope (relative work units of a bit-plane engine, NOT time): a column's width
 * is the bit length of max - min of its selected present values (at least 1) plus one validity plane when it has NULLs;
 * a polynomial p (integer form D p) costs scan = ceil(N / 64) x the widths of the columns it reads (N selected rows),
 * work = N x (terms + multiplications) x words, words = ceil((actual_bits + 1) / 64) from the actual bit length of
 * |D p| measured on every compared row (the evaluation modulo 2^(64 words) is exact when the result fits), total = scan
 * + work. A generator set costs storage = ceil(N / 64) x the generators' widths plus every reconstruction's scan and
 * work; nothing eliminated (G = V) costs storage only. Each eliminated column also states its read path: RECONSTRUCT
 * only when its reconstruction is strictly cheaper than scanning the stored column, else DIRECT.
 *
 * Forms (tr_elim_forms_make): local elimination forms of a query expression q over a selection: the direct q, every
 * local substitution q_S = q(x_S := f_S) for nonempty sets S of the eliminated variables q reads (|S| <= max_eliminate;
 * increasing size, then lexicographic), each GEOM-001's division of q by the reconstruction certificates of S (lex S
 * first, not completed: exactly the substitution), certified by FND-007 through those certificates (depth 2), and
 * optionally the normal form of q under the selection's elimination ideal (its canonical form, certified from the
 * relations). Every alternative is checked against q on every selected row (values and NULL mask) and costed; the
 * cheapest admissible candidate by the priority is chosen, ties to the earliest, so the direct candidate is kept
 * whenever no alternative is strictly cheaper ("expansion loses") or every alternative is rejected. The canonical
 * normal form is a candidate like any other, never assumed cheapest.
 *
 * Checks (tr_elim_check_view; tr_mgpu_elim_check on the device, the same results bit for bit): pairs (T_j, F_j) of a
 * target and a form with integer forms A_j = D_j T_j and B_j = D_j F_j (one D_j > 0), on every selected row: the
 * target is NULL when a column it reads is NULL, the form likewise (SQL propagation); both NULL agree, one NULL is a
 * NULL-domain mismatch (null_lost: the target present, the form NULL; null_extra: the converse); both present compare
 * A and B exactly in W 64-bit words under a bound proven from the scope's extremes (W <= 8; the pair records the
 * magnitudes it was proven for and refuses a view whose extremes exceed them), a difference a value mismatch; a range
 * pair (the target one scope column) also compares B with D lo and D hi, [lo, hi] the target column's extremes on the
 * view: below, above (outside the range) and at_lo, at_hi (the range attained). Holds: no mismatch, nothing undecided,
 * and a ranged pair attains both ends whenever it compared a row. The first mismatch (any kind) is the smallest row.
 *
 * Budgets (math_budgets.h): iterations = solves (the eliminations of a search; the alternatives of a form search,
 * counted before any solve), refused at the solve past the limit (need = limit + 1), never truncated; degree and support
 * (terms) bound every GEOM-001 completion and reduction (a search set past them is undecided as above, an ORDER
 * selection is refused, a form candidate past them is rejected with its refusal and the direct one kept); variables =
 * |V|; output, bytes as GEOM-001's and this module's
 * allocations. Each solve's S-pairs and division steps run under GEOM-001's default iteration limit. Errors:
 * TR_MATH_INVALID, _MEMORY, _OVERFLOW, _BUDGET, _UNSUPPORTED, _DOMAIN_ERROR (relations GEOM-001 refuses; pairs checked
 * on a view their bound does not cover). Objects are immutable and individually owned. */
#ifndef TORBLES_MATH_ELIM_H
#define TORBLES_MATH_ELIM_H
#include <stdint.h>
#include <string.h>
#include "math_num.h"
#include "math_budgets.h"
#include "math_scope.h"
#include "math_certificates.h"
#include "math_ident.h"
#include "math_geom.h"
#ifdef __cplusplus
extern "C" {
#endif

#define TR_ELIM_ABI 1
#define TR_ELIM_MAX_VARS 8                /* variables of a selection (FND-007 claims read at most 8 columns) */
#define TR_ELIM_MAX_RELATIONS 16
#define TR_ELIM_MAX_CANDIDATES 256        /* generator sets of one search (2^8) */
#define TR_ELIM_MAX_FORMS 257             /* the direct form, 255 local forms and the normal form */
#define TR_ELIM_MAX_PAIRS 256             /* pairs of one check batch */
#define TR_ELIM_PAIR_CHUNK 8              /* CUDA: pairs per work item */
#define TR_ELIM_SLICE_WORDS 64            /* CUDA: selection words per row slice, at least */
#define TR_ELIM_MAX_SLICES 2048
#define TR_ELIM_ACC 16                    /* accumulators per pair (see tr_elim_pair_row) */

enum { TR_ELIM_SEARCH = 1, TR_ELIM_ORDER = 2 };
enum { TR_ELIM_PRIORITY_TOTAL = 1, TR_ELIM_PRIORITY_SCAN = 2, TR_ELIM_PRIORITY_WORK = 3, TR_ELIM_PRIORITY_BITS = 4 };
enum { TR_ELIM_GENERATOR = 1, TR_ELIM_ELIMINATED = 2 };                           /* roles */
enum { TR_ELIM_READ_DIRECT = 1, TR_ELIM_READ_RECONSTRUCT = 2 };                   /* read paths */
enum { TR_ELIM_CHOSEN = 1, TR_ELIM_ADMISSIBLE = 2, TR_ELIM_REJECTED = 3 };        /* candidate statuses */
enum { TR_ELIM_R_CHEAPEST = 1,           /* chosen: the cheapest admissible candidate */
       TR_ELIM_R_COSTLIER = 2,           /* admissible, not chosen */
       TR_ELIM_R_NOT_RECONSTRUCTIBLE = 3,/* a variable's normal form leaves the generators */
       TR_ELIM_R_NULL_MISMATCH = 4,      /* the row check found a NULL-domain difference */
       TR_ELIM_R_VALUE_MISMATCH = 5,     /* the row check found a value difference */
       TR_ELIM_R_RANGE = 6,              /* the row check found the range not preserved */
       TR_ELIM_R_BUDGET = 7,             /* past the degree / support budget (refusal recorded): undecided */
       TR_ELIM_R_UNCLAIMABLE = 8,        /* exact values past int64 or a representation limit: undecided */
       TR_ELIM_R_UNDECIDED = 9,          /* rows the exact evaluation could not decide (no proven width) */
       TR_ELIM_R_DIRECT = 10 };          /* nothing eliminated (G = V): no smaller set was feasible */
enum { TR_ELIM_FORM_DIRECT = 1, TR_ELIM_FORM_LOCAL = 2, TR_ELIM_FORM_NORMAL = 3 };
enum { TR_ELIM_NEED_CPU = 0, TR_ELIM_NEED_DEVICE = 1 };

/* ------------------------------------------------------------------ structs crossing the ctypes boundary */
typedef struct {
    int32_t mode;                        /* TR_ELIM_SEARCH or TR_ELIM_ORDER */
    int32_t nvars;                       /* variables to cover (scope columns); 0: every column of the relations */
    int32_t col[TR_ELIM_MAX_VARS];
    int32_t norder;                      /* ORDER: the lex sequence (most significant first; the rest after, in V's order) */
    int32_t order[TR_ELIM_MAX_VARS];
    int32_t max_size;                    /* SEARCH: the largest generator set tried (-1: every size below |V|) */
    int32_t rank_all;                    /* SEARCH: 1 = cost every feasible set of the minimum size; 0 = take the first */
    int32_t priority;                    /* TR_ELIM_PRIORITY_* (0: TOTAL) */
    int32_t reserved[2];
} tr_elim_spec;                          /* 96 bytes */

typedef struct {
    uint64_t rows;                       /* selected rows N */
    uint64_t columns, scan_bits;         /* columns read; the sum of their widths */
    uint64_t terms, multiplications, degree;   /* of the integer form D p */
    uint64_t bound_bits, actual_bits;    /* the proven evaluation bound; the measured largest |D p| (bits) */
    uint64_t words;                      /* evaluation words: ceil((actual_bits + 1) / 64) (the bound's when not measured) */
    uint64_t scan, work, total;          /* ceil(N / 64) x scan_bits; N x (terms + multiplications) x words; scan + work */
    int64_t denominator;                 /* D */
    int32_t measured, reserved;
} tr_elim_cost;                          /* 112 bytes */

typedef struct {
    uint64_t rows, selected;
    uint64_t compared, nulls;            /* both present (values compared) / both NULL (agree) */
    uint64_t null_lost, null_extra;      /* the target present and the form NULL / the target NULL and the form present */
    uint64_t values;                     /* compared rows with A != B */
    uint64_t below, above, at_lo, at_hi; /* a range pair: B < D lo, B > D hi, B == D lo, B == D hi */
    uint64_t undecided;                  /* compared rows not evaluated (no proven width) */
    uint64_t first, first_null, first_value, first_range;   /* the smallest mismatching rows (UINT64_MAX: none) */
    uint32_t bits_target, bits_form;     /* the largest bit lengths of |A| and |B| over the compared rows */
    int32_t ranged, holds;
} tr_elim_check;                         /* 144 bytes */

typedef struct {
    int32_t npairs, reserved;
    uint64_t mag[TR_ELIM_MAX_VARS];      /* per variable of cand: the magnitude its bounds were proven for */
    tr_ident_cand_view cand;             /* 2 npairs candidates: A_j = D_j T_j, then B_j = D_j F_j */
    const int32_t *target;               /* [npairs] the target's scope column of a range pair, or -1 */
    const int64_t *den;                  /* [npairs] D_j > 0 */
} tr_elim_pair_view;                     /* 176 bytes */

typedef struct {
    uint32_t generators, eliminated;     /* masks over the selection's variables */
    int32_t size, status, reason, solve; /* |G|; TR_ELIM_CHOSEN / _ADMISSIBLE / _REJECTED; TR_ELIM_R_*; solve index */
    int32_t failed, reserved;            /* the first variable that failed (-1) */
    uint64_t stored_bits, storage;       /* the generators' widths; ceil(N / 64) x stored_bits */
    uint64_t recon_scan, recon_work;     /* every reconstruction's scan and work (each eliminated column once) */
    uint64_t total;                      /* storage + recon_scan + recon_work */
    uint64_t terms, degree, bits;        /* reconstruction terms, their largest degree, the largest actual bits */
    tr_math_refusal refusal;             /* reason BUDGET (undecided): the solve's refusal (resource, need, limit) */
} tr_elim_candidate;                     /* 120 bytes */

typedef struct {
    int32_t column, role;                /* the scope column; TR_ELIM_GENERATOR / _ELIMINATED */
    int32_t read, claimable;             /* TR_ELIM_READ_*; 1: the reconstruction has an FND-007 certificate */
    uint32_t support, reserved;          /* the generators its reconstruction reads (mask over the variables) */
    uint64_t width, direct;              /* its width; ceil(N / 64) x width (scanning the stored column) */
    tr_elim_cost reconstruction;         /* the reconstruction's measured cost (eliminated columns) */
    tr_elim_check check;                 /* its row check on the selection's scope (eliminated columns) */
} tr_elim_column;                        /* 296 bytes */

typedef struct {
    int32_t nvars, mode, ngenerators, neliminated;
    int32_t col[TR_ELIM_MAX_VARS];       /* the variables (scope columns), in the selection's order */
    uint32_t generators, eliminated;
    int32_t minimum, minimal;            /* the chosen set's size; 1: every smaller set was decided infeasible */
    int32_t undecided, undecided_size;   /* sets not decided within the budget (BUDGET / UNCLAIMABLE); the smallest's
                                            size (-1: none) */
    int32_t ncandidates, chosen;
    int32_t nrelations, priority;
    uint64_t solves, max_degree, max_terms;   /* eliminations; the largest degree and support observed */
    uint64_t rows, direct_storage;       /* selected rows; ceil(N / 64) x every variable's width */
    uint64_t bytes;
} tr_elim_info;                          /* 136 bytes */

typedef struct {
    int32_t max_eliminate;               /* the largest set of variables substituted (-1: every one) */
    int32_t normal_form;                 /* 1: the normal form under the selection's elimination ideal is a candidate */
    int32_t priority;                    /* TR_ELIM_PRIORITY_* (0: TOTAL) */
    int32_t reserved;
} tr_elim_form_spec;                     /* 16 bytes */

typedef struct {
    int32_t kind, status, reason, depth; /* TR_ELIM_FORM_*; TR_ELIM_CHOSEN...; TR_ELIM_R_*; certificate depth (0) */
    uint32_t eliminated, columns;        /* LOCAL: the substituted variables (mask); the scope columns read (mask) */
    int32_t claimable, reserved;
    tr_elim_cost cost;
    tr_elim_check check;                 /* against the direct expression on every selected row */
    tr_math_refusal refusal;             /* a candidate rejected for the budget: resource, need, limit */
} tr_elim_form;                          /* 312 bytes */

typedef struct {
    int32_t ncandidates, chosen, nvars, priority;
    uint32_t reads, reserved;            /* the eliminated variables q reads (mask) */
    uint64_t solves, rows, bytes;
} tr_elim_forms_info;                    /* 48 bytes */

typedef struct tr_elim_selection_s tr_elim_selection;
typedef struct tr_elim_pairs_s tr_elim_pairs;
typedef struct tr_elim_forms_s tr_elim_forms;

/* ------------------------------------------------------------------ shared row check (CPU loop and kernel) */
/* acc[TR_ELIM_ACC]: 0 compared, 1 nulls, 2 null_lost, 3 null_extra, 4 values, 5 below, 6 above, 7 at_lo, 8 at_hi,
 * 9 undecided, 10 first, 11 first_null, 12 first_value, 13 first_range (UINT64_MAX initially), 14 bits_target,
 * 15 bits_form (0 initially) */
TR_NUM_HD void tr_elim_acc_init(uint64_t *acc) {
    for (int i = 0; i < TR_ELIM_ACC; i++) acc[i] = (i >= 10 && i <= 13) ? UINT64_MAX : 0;
}
TR_NUM_HD int tr_elim_present(const tr_scope_view *v, const tr_ident_cand_view *c, uint32_t support, uint64_t row) {
    for (int32_t i = 0; i < c->nvars; i++) if (((support >> i) & 1) && !tr_scope_present(v, c->var[i], row)) return 0;
    return 1;
}
/* x = |m| (W words), negated when m < 0 */
TR_NUM_HD void tr_elim_w_scaled(uint64_t *x, int32_t W, int64_t d, int64_t m) {
    tr_ident_w_set(x, W, tr_abs_u64(d));
    tr_ident_w_mul(x, W, tr_abs_u64(m));
    if ((d < 0) != (m < 0) && m != 0 && d != 0) tr_ident_w_neg(x, W);
}
/* sign of b - a (W words, exact by the proven bound): -1, 0, 1 */
TR_NUM_HD int tr_elim_w_cmp(const uint64_t *b, const uint64_t *a, int32_t W) {
    uint64_t d[TR_IDENT_MAX_WORDS], na[TR_IDENT_MAX_WORDS];
    for (int32_t i = 0; i < W; i++) { d[i] = b[i]; na[i] = a[i]; }
    tr_ident_w_neg(na, W);
    tr_ident_w_add(d, na, W);
    if (tr_ident_w_zero(d, W)) return 0;
    return (d[W - 1] >> 63) ? -1 : 1;
}
/* one selected row of pair j */
TR_NUM_HD void tr_elim_pair_row(const tr_scope_view *v, const tr_elim_pair_view *p, int32_t j, uint64_t row, uint64_t *acc) {
    const tr_ident_cand_view *c = &p->cand;
    const int pa = tr_elim_present(v, c, c->support[2 * j], row), pb = tr_elim_present(v, c, c->support[2 * j + 1], row);
    if (!pa && !pb) { acc[1]++; return; }
    if (pa != pb) {
        acc[pa ? 2 : 3]++;
        if (row < acc[11]) acc[11] = row;
        if (row < acc[10]) acc[10] = row;
        return;
    }
    acc[0]++;
    const int32_t W = c->words[2 * j];
    if (W < 2 || W > TR_IDENT_MAX_WORDS) { acc[9]++; return; }
    uint64_t a[TR_IDENT_MAX_WORDS], b[TR_IDENT_MAX_WORDS];
    tr_ident_cand_value(v, c, 2 * j, row, W, a);
    tr_ident_cand_value(v, c, 2 * j + 1, row, W, b);
    uint32_t ba = tr_geom_bits(a, W), bb = tr_geom_bits(b, W);
    if (ba > acc[14]) acc[14] = ba;
    if (bb > acc[15]) acc[15] = bb;
    int same = 1;
    for (int32_t i = 0; i < W; i++) if (a[i] != b[i]) same = 0;
    if (!same) {
        acc[4]++;
        if (row < acc[12]) acc[12] = row;
        if (row < acc[10]) acc[10] = row;
    }
    const int32_t t = p->target[j];
    if (t < 0 || !v->col[t].nvalid) return;
    uint64_t lim[TR_IDENT_MAX_WORDS];
    tr_elim_w_scaled(lim, W, p->den[j], v->col[t].min);
    int slo = tr_elim_w_cmp(b, lim, W);
    tr_elim_w_scaled(lim, W, p->den[j], v->col[t].max);
    int shi = tr_elim_w_cmp(b, lim, W);
    if (slo == 0) acc[7]++;
    if (shi == 0) acc[8]++;
    if (slo < 0 || shi > 0) {
        acc[slo < 0 ? 5 : 6]++;
        if (row < acc[13]) acc[13] = row;
        if (row < acc[10]) acc[10] = row;
    }
}
/* the structure the row loop relies on (host code of both libraries): 0, TR_MATH_INVALID (-1), or
 * TR_MATH_DOMAIN_ERROR (-6) for a view whose extremes the proven magnitudes do not cover */
static inline int32_t tr_elim_pair_view_check(const tr_scope_view *v, const tr_elim_pair_view *p) {
    if (!v || v->ngroups < 1 || (v->rows && !v->selection) || v->ncols < 0 || v->ncols > TR_SCOPE_MAX_COLUMNS) return -1;
    if (!p || p->npairs < 0 || p->npairs > TR_ELIM_MAX_PAIRS) return -1;
    const tr_ident_cand_view *c = &p->cand;
    if (c->ncand != 2 * p->npairs || c->nvars < 1 || c->nvars > TR_ELIM_MAX_VARS) return -1;
    if (p->npairs && (!c->offset || !c->support || !c->words || !p->target || !p->den ||
                      (c->nterms && (!c->exponents || !c->coefficients)))) return -1;
    for (int32_t i = 0; i < c->nvars; i++) {
        int32_t k = c->var[i];
        if (k < 0 || k >= v->ncols || (v->rows && !v->col[k].codes && !v->col[k].values)) return -1;
        for (int32_t q = 0; q < i; q++) if (c->var[q] == k) return -1;
    }
    if (p->npairs && (c->offset[0] || c->offset[c->ncand] != c->nterms)) return -1;
    for (int32_t j = 0; j < c->ncand; j++)
        if (c->offset[j] > c->offset[j + 1] || (c->nvars < 32 && (c->support[j] >> c->nvars)) ||
            (c->words[j] && (c->words[j] < 2 || c->words[j] > TR_IDENT_MAX_WORDS)) || c->words[j] != c->words[j & ~1])
            return -1;
    for (int32_t j = 0; j < p->npairs; j++) {
        if (p->den[j] <= 0) return -1;
        if (p->target[j] >= 0) {
            int found = 0;
            for (int32_t i = 0; i < c->nvars; i++) found |= c->var[i] == p->target[j];
            if (!found) return -1;
        }
    }
    for (int32_t i = 0; i < c->nvars; i++) {
        const tr_scope_view_column *k = &v->col[c->var[i]];
        if (k->nvalid && (tr_abs_u64(k->min) > p->mag[i] || tr_abs_u64(k->max) > p->mag[i])) return -6;
    }
    return 0;
}
/* the per-pair result from the accumulators */
static inline void tr_elim_check_finish(const tr_scope_view *v, const tr_elim_pair_view *p, int32_t j, const uint64_t *acc,
                                        tr_elim_check *out) {
    memset(out, 0, sizeof *out);
    out->rows = v->rows; out->selected = v->selected;
    out->compared = acc[0]; out->nulls = acc[1]; out->null_lost = acc[2]; out->null_extra = acc[3]; out->values = acc[4];
    out->below = acc[5]; out->above = acc[6]; out->at_lo = acc[7]; out->at_hi = acc[8]; out->undecided = acc[9];
    out->first = acc[10]; out->first_null = acc[11]; out->first_value = acc[12]; out->first_range = acc[13];
    out->bits_target = (uint32_t)acc[14]; out->bits_form = (uint32_t)acc[15];
    out->ranged = p->target[j] >= 0;
    int attained = !out->ranged || !out->compared || (out->at_lo && out->at_hi);
    out->holds = !out->null_lost && !out->null_extra && !out->values && !out->undecided && !out->below && !out->above &&
                 attained;
}

/* the device copies of tr_mgpu_elim_check (and the CPU needs: variables, support, output) */
static inline void tr_elim_check_device_need(const tr_scope_view *v, const tr_elim_pair_view *p, tr_math_need *need) {
    tr_ident_verify_device_need(v, &p->cand, need);
    uint64_t n = (uint64_t)(p->npairs > 0 ? p->npairs : 0);
    need->need[TR_BUDGET_OUTPUT] = n;
    need->need[TR_BUDGET_BYTES] = tr_budget_add(need->need[TR_BUDGET_BYTES], tr_budget_mul(n, 4 + 8 + 8 * TR_ELIM_ACC));
}

/* ------------------------------------------------------------------ generator selection */
tr_elim_selection *tr_elim_select(const tr_scope *, const tr_cert *const *relations, int32_t n, const tr_elim_spec *,
                                  const tr_math_budget *, tr_math_refusal *, int32_t *error);
int32_t tr_elim_selection_get_info(const tr_elim_selection *, tr_elim_info *);
int32_t tr_elim_selection_candidates(const tr_elim_selection *, tr_elim_candidate *out, int32_t capacity);
int32_t tr_elim_selection_column(const tr_elim_selection *, int32_t i, tr_elim_column *out);
/* variable i's reconstruction f (GEOM-001 polynomial over scope columns); TR_MATH_INVALID for a generator */
tr_geom_pol *tr_elim_selection_form(const tr_elim_selection *, int32_t i, int32_t *error);
/* variable i's FND-007 derived certificate D (x - f) = sum D h_k r_k (GEOM-001's tr_geom_certify) */
tr_cert *tr_elim_selection_certify(const tr_elim_selection *, int32_t i, const tr_math_budget *, tr_math_refusal *,
                                   int32_t *error);
/* the pairs (x, f_x) of every eliminated variable (range pairs), for tr_elim_check_view / tr_mgpu_elim_check */
tr_elim_pairs *tr_elim_selection_pairs(const tr_elim_selection *, const tr_scope *, const tr_math_budget *,
                                       tr_math_refusal *, int32_t *error);
void tr_elim_selection_free(tr_elim_selection *);

/* ------------------------------------------------------------------ pairs, checks and costs */
/* n pairs (targets[j], forms[j]) over scope columns; ranged[j] (NULL: none) the scope column of a target that is that
 * one column (range check), or -1 */
tr_elim_pairs *tr_elim_pairs_make(const tr_scope *, const tr_geom_poly *targets, const tr_geom_poly *forms,
                                  const int32_t *ranged, int32_t n, const tr_math_budget *, tr_math_refusal *,
                                  int32_t *error);
int32_t tr_elim_pairs_get_view(const tr_elim_pairs *, tr_elim_pair_view *);
/* bound bits of B_j (the form's integer form) on the scope it was made for */
uint64_t tr_elim_pairs_bound_bits(const tr_elim_pairs *, int32_t j);
void tr_elim_pairs_free(tr_elim_pairs *);
int32_t tr_elim_check_need(const tr_scope_view *, const tr_elim_pair_view *, int32_t target, tr_math_need *);
int32_t tr_elim_check_view(const tr_scope_view *, const tr_elim_pair_view *, const tr_math_budget *, tr_elim_check *out,
                           tr_math_refusal *);
/* the cost of a polynomial on a scope, measured on every selected row where its columns are present */
int32_t tr_elim_cost_of(const tr_scope *, const tr_geom_poly *, const tr_math_budget *, tr_elim_cost *out,
                        tr_math_refusal *);
/* the index of the cheapest of n costs by priority (ties to the earliest) */
int32_t tr_elim_choose(const tr_elim_cost *costs, int32_t n, int32_t priority, int32_t *chosen);

/* ------------------------------------------------------------------ local elimination forms */
tr_elim_forms *tr_elim_forms_make(const tr_scope *, const tr_elim_selection *, const tr_geom_poly *q,
                                  const tr_elim_form_spec *, const tr_math_budget *, tr_math_refusal *, int32_t *error);
int32_t tr_elim_forms_get_info(const tr_elim_forms *, tr_elim_forms_info *);
int32_t tr_elim_forms_candidate(const tr_elim_forms *, int32_t i, tr_elim_form *out);
/* candidate i >= 1: its polynomial (candidate 0 is q itself; TR_MATH_INVALID for one rejected before it was built) */
tr_geom_pol *tr_elim_forms_poly(const tr_elim_forms *, int32_t i, int32_t *error);
/* candidate i's FND-007 derived certificate D (q - p) = ... (TR_MATH_INVALID for the direct candidate) */
tr_cert *tr_elim_forms_certify(const tr_elim_forms *, int32_t i, const tr_math_budget *, tr_math_refusal *,
                               int32_t *error);
void tr_elim_forms_free(tr_elim_forms *);

/* ------------------------------------------------------------------ the CUDA library (math_elim_gpu.cu) */
int tr_mgpu_elim_check(const tr_scope_view *view, const tr_elim_pair_view *pairs, const tr_math_budget *budget,
                       tr_elim_check *out, tr_math_refusal *refusal);
#ifdef __cplusplus
}
#endif
#endif
