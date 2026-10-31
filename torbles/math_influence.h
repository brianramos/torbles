/* BOOL-002: influences and interactions of a predicate (a BOOL-001 program over k conditions): the measure of the
 * points where its XOR derivative in a condition (or a set of conditions) is TRUE, and the signed mixed differences,
 * under the uniform measure of the 2^k assignments and under the empirical measure of the stored rows of a scope, as
 * separate results; on the CPU (math_influence.c) and on a CUDA device (math_influence_gpu.cu), bit for bit equal.
 * SPDX-License-Identifier: Apache-2.0
 *
 * ---- What is measured: coordinate sensitivity, not a causal effect ----------------------------------------------
 * For a set S of m conditions (m = 1: one condition i) and a point, the cofactors f|S:=t, the XOR derivative
 * d_S f = XOR_t f|S:=t and the signed difference D_S f = sum_t (-1)^(m - |t|) [f|S:=t] are BOOL-001's (math_bool.h:
 * SQL three-valued logic; a cofactor never reads the conditions of S at the point; a difference is UNKNOWN where one
 * of its cofactors is). The influence of S is the measure of {d_S f = TRUE}: how often substituting the conditions of
 * S, with every other condition held at the point's own value, changes the predicate. It is a property of the
 * predicate as a function of its conditions (coordinate sensitivity), measured over a declared set of points; it is
 * not the effect of an intervention on the data, and no causal reading is implied.
 *
 * ---- The two measures, never mixed ------------------------------------------------------------------------------
 * The domain decides the measure (measure = the domain kind, carried by every result):
 *   TR_BOOL_ASSIGNMENTS  uniform: the 2^k assignments, each once (points = 2^k). Computed from the truth table of the
 *                        predicate (value and valid words over the assignments, built once): the cofactor words of
 *                        an assignment t of S at word w are the table words with the conditions of S substituted
 *                        (conditions >= 6 select another word, conditions < 6 move bits within the word).
 *   TR_BOOL_ROWS         empirical: the selected rows of a scope, each once (points = the selected rows): the stored
 *                        rows only, never the assignments they do not contain. Computed from the masks by cofactors
 *                        per row: per 64-row word the program once (the base words), then for each set S and
 *                        assignment t only the cone of S (tr_bool_eval_cone).
 * The same set S generally has different influences under the two measures; they coincide only when the selected
 * rows hold every assignment exactly once.
 *
 * ---- Results ---------------------------------------------------------------------------------------------------
 * Per term (tr_influence_term; the first-order terms {i} for every condition with TR_INFLUENCE_FIRST, then the given
 * sets): known / unknown points (every cofactor over S known / one UNKNOWN; known + unknown = points), sensitive (the
 * known points where d_S f is TRUE; known - sensitive points where it is FALSE), positive / negative (known points with
 * D_S f > 0 / < 0; m = 1: f|1 - f|0 = +1 / -1), sum (the exact sum of D_S f over the known points) and measure =
 * sensitive / points as a canonical fraction (0 / 1 without points). UNKNOWN points are reported apart, never as
 * insensitive. With TR_INFLUENCE_FIRST the summary also gives the total influence (sum over i of sensitive_i, and
 * total / points canonical: the expected number of sensitive conditions with an UNKNOWN derivative counted 0) and the
 * sensitivity histogram: bins[b] = the points where every first-order cofactor is known (all_known) and exactly b
 * conditions are sensitive, b = 0..k (max_sensitivity: the largest occupied bin).
 *
 * ---- Budgets (math_budgets.h), checked before anything is allocated --------------------------------------------
 * tr_influence_need: variables = k, degree = the largest m, circuit = nodes, support = 2^k on ASSIGNMENTS (0 on ROWS:
 * the rows were budgeted with the scope), tensor = sum over the terms of 2^m (the cofactors per point), output = the
 * terms (+ the k first-order terms and the k + 1 bins with TR_INFLUENCE_FIRST), bytes = CPU: the plans, cones, node
 * words, accumulators and (ASSIGNMENTS)
 * the truth table; TR_INFLUENCE_NEED_DEVICE: the device copies, the per-item scratch and the totals
 * (tr_influence_device_bytes). tr_influence_sets: variables = k, output = the sets.
 * Representation limits (with a budget that admits them, else BOOL-001's): TR_MATH_UNSUPPORTED for m > 20 or more
 * than TR_INFLUENCE_MAX_TERMS terms. A program over another k than the domain: TR_MATH_DOMAIN_ERROR.
 *
 * Errors: TR_MATH_INVALID (-1), _MEMORY (-2), _BUDGET (-5 with the refusal), TR_MATH_DOMAIN_ERROR (-6),
 * TR_MATH_DEVICE (-11), TR_MATH_UNSUPPORTED (-12). Outputs are written only on success. The freshness of a ROWS domain
 * is its scope's (the Python layer checks it before every use). */
#ifndef TORBLES_MATH_INFLUENCE_H
#define TORBLES_MATH_INFLUENCE_H
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "math_num.h"
#include "math_budgets.h"
#include "math_bool.h"
#ifdef __cplusplus
extern "C" {
#endif

#define TR_INFLUENCE_ABI 1
#define TR_INFLUENCE_MAX_TERMS (TR_BOOL_MAX_QUERIES - TR_BOOL_MAX_CONDITIONS)   /* the first-order terms fit beside */
#define TR_INFLUENCE_COUNTER 7                     /* bit planes of a per-point count of sensitive conditions (<= 64) */
enum { TR_INFLUENCE_FIRST = 1 };                   /* flags: every condition's influence and the sensitivity histogram */
enum { TR_INFLUENCE_NEED_CPU = 0, TR_INFLUENCE_NEED_DEVICE = 1 };

typedef struct {
    uint64_t vars;                      /* S */
    int32_t order, measure;             /* m; TR_BOOL_ASSIGNMENTS (uniform) or TR_BOOL_ROWS (empirical) */
    uint64_t points;                    /* 2^k, or the selected rows */
    uint64_t known, unknown;            /* every cofactor over S known / one UNKNOWN */
    uint64_t sensitive;                 /* known points where d_S f is TRUE */
    uint64_t positive, negative;        /* known points where D_S f > 0 / < 0 */
    tr_math_i128 sum;                   /* sum of D_S f over the known points */
    uint64_t measure_num, measure_den;  /* sensitive / points, canonical */
} tr_influence_term;                    /* 96 bytes */

typedef struct {
    uint64_t points, f_true, f_false, f_unknown;
    uint64_t all_known;                 /* points where every first-order cofactor is known (TR_INFLUENCE_FIRST) */
    uint64_t max_sensitivity;           /* the largest b with bins[b] > 0 (0 without such points) */
    tr_math_i128 total;                 /* sum over the conditions of their sensitive points */
    tr_math_i128 total_num;             /* total / points, canonical */
    uint64_t total_den;
    int32_t k, measure, nterms, flags;
} tr_influence_summary;                 /* 104 bytes */

/* the per-term accumulator of a scan (internal: CPU and device scratch) */
typedef struct { uint64_t known, unknown, sensitive, positive, negative; int64_t sum_lo, sum_hi; uint64_t pad; } tr_influence_acc;

/* ------------------------------------------------------------------ shared helpers (CPU code, CUDA host and kernels) */
/* the within-word pattern of condition i < 6: bit p set iff bit i of p is */
TR_NUM_HD uint64_t tr_influence_pattern(int i) {
    switch (i) {
    case 0: return UINT64_C(0xAAAAAAAAAAAAAAAA);
    case 1: return UINT64_C(0xCCCCCCCCCCCCCCCC);
    case 2: return UINT64_C(0xF0F0F0F0F0F0F0F0);
    case 3: return UINT64_C(0xFF00FF00FF00FF00);
    case 4: return UINT64_C(0xFFFF0000FFFF0000);
    default: return UINT64_C(0xFFFFFFFF00000000);
    }
}
/* the truth-table words of the cofactor f|S:=t at word w (ASSIGNMENTS): bit p is the table at the assignment
 * 64 w + p with the conditions of S set by t (bit j of t: the j-th condition of S in increasing order) */
TR_NUM_HD void tr_influence_subst(const uint64_t *tv, const uint64_t *tk, uint64_t vars, uint64_t t, uint64_t w,
                                  uint64_t *v, uint64_t *k) {
    uint64_t set = 0, rest = vars;
    for (int j = 0; rest; j++) {
        int i = tr_bool_ctz(rest);
        rest &= rest - 1;
        if ((t >> j) & 1) set |= (uint64_t)1 << i;
    }
    uint64_t src = (w & ~(vars >> 6)) | (set >> 6);           /* conditions >= 6 are the bits of the word index */
    uint64_t x = tv[src], y = tk[src];
    for (int i = 0; i < 6; i++) {                             /* conditions < 6 move bits within the word */
        if (!((vars >> i) & 1)) continue;
        uint64_t m1 = tr_influence_pattern(i), m0 = ~m1;
        int sh = 1 << i;
        if ((set >> i) & 1) { x = (x & m1) | ((x & m1) >> sh); y = (y & m1) | ((y & m1) >> sh); }
        else { x = (x & m0) | ((x & m0) << sh); y = (y & m0) | ((y & m0) << sh); }
    }
    *v = x; *k = y;
}
/* adds one word's sensitive-condition indicator to the bit-sliced per-point counter */
TR_NUM_HD void tr_influence_count(uint64_t *c, uint64_t x) {
    for (int b = 0; x && b < TR_INFLUENCE_COUNTER; b++) { uint64_t carry = c[b] & x; c[b] ^= x; x = carry; }
}
/* One word of the scan: the program (ROWS: the base node words in val, then each term's cone per assignment;
 * ASSIGNMENTS: the truth table tv / tk) and every plan's counts, added to acc[np] and fcounts[3]; with histogram (the
 * first `first` plans are then the first-order terms) to bins[first + 1] and *all_known. */
TR_NUM_HD void tr_influence_word(const tr_bool_domain *d, const tr_bool_program_view *p, const tr_bool_plan *plans,
                                 int32_t np, const uint32_t *cones, int32_t first, int32_t histogram,
                                 const uint64_t *tv, const uint64_t *tk, uint64_t w, uint64_t *val,
                                 tr_influence_acc *acc, uint64_t *fcounts, uint64_t *bins, uint64_t *all_known) {
    const uint64_t sel = tr_bool_selection_word(d, w), root = (uint64_t)p->root;
    if (!sel) return;
    uint64_t fv, fk;
    if (d->kind == TR_BOOL_ROWS) {
        tr_bool_eval_base(p, d, w, val);
        fv = val[2 * root] & sel; fk = val[2 * root + 1] & sel;
    } else {
        fv = tv[w] & sel; fk = tk[w] & sel;
    }
    fcounts[0] += (uint64_t)tr_bool_popc(fv);
    fcounts[1] += (uint64_t)tr_bool_popc(fk & ~fv);
    fcounts[2] += (uint64_t)tr_bool_popc(sel & ~fk);
    uint64_t counter[TR_INFLUENCE_COUNTER] = {0}, allk = sel;
    int32_t values[64];
    uint32_t counts[64];
    for (int32_t j = 0; j < np; j++) {
        const tr_bool_plan *P = &plans[j];
        tr_bool_acc A;
        tr_bool_acc_begin(&A, P->order, sel);
        const uint64_t nt = (uint64_t)1 << P->order;
        for (uint64_t t = 0; t < nt; t++) {
            uint64_t rv, rk;
            if (d->kind == TR_BOOL_ROWS) {
                if (P->ncone) tr_bool_eval_cone(p, d, cones + P->cone, P->ncone, P->vars, t, 0, w, val);
                rv = val[2 * root] & sel; rk = val[2 * root + 1] & sel;
            } else {
                tr_influence_subst(tv, tk, P->vars, t, w, &rv, &rk);
                rv &= sel; rk &= sel;
            }
            tr_bool_acc_add(&A, P->order, t, rv, rk);
        }
        if (d->kind == TR_BOOL_ROWS && P->ncone)                 /* restore the base words */
            tr_bool_eval_cone(p, d, cones + P->cone, P->ncone, P->vars, 0, 1, w, val);
        const uint64_t known = A.known, sens = A.parity & known;
        tr_influence_acc *a = &acc[j];
        a->known += (uint64_t)tr_bool_popc(known);
        a->unknown += (uint64_t)tr_bool_popc(sel & ~known);
        a->sensitive += (uint64_t)tr_bool_popc(sens);
        int n = tr_bool_acc_leaves(&A, P->order, values, counts);
        tr_math_i128 cur;
        cur.lo = a->sum_lo; cur.hi = a->sum_hi;
        __int128 sum = tr_i128_get(cur);                          /* |sum| <= 2^(m-1) points < 2^84 */
        for (int l = 0; l < n; l++) {
            if (values[l] > 0) a->positive += counts[l];
            else if (values[l] < 0) a->negative += counts[l];
            sum += (__int128)values[l] * (__int128)counts[l];
        }
        tr_math_i128 s = tr_i128_set(sum);
        a->sum_lo = s.lo; a->sum_hi = s.hi;
        if (j < first) { tr_influence_count(counter, sens); allk &= known; }
    }
    if (histogram) {                                             /* k = 0: every point has 0 sensitive conditions */
        *all_known += (uint64_t)tr_bool_popc(allk);
        for (int32_t b = 0; b <= first; b++) {                   /* the points with exactly b sensitive conditions */
            uint64_t mask = allk;
            for (int i = 0; i < TR_INFLUENCE_COUNTER && mask; i++) mask &= ((b >> i) & 1) ? counter[i] : ~counter[i];
            bins[b] += (uint64_t)tr_bool_popc(mask);
        }
    }
}

/* ------------------------------------------------------------------ host-side planning (both libraries) */
/* the queries of a request: the first-order terms (flags & FIRST), then the sets; *nq their number */
static inline int32_t tr_influence_queries(const tr_bool_program_view *p, const uint64_t *sets, int32_t nterms,
                                           int32_t flags, tr_bool_query *q, int32_t *nq) {
    if (nterms < 0 || (nterms && !sets) || (flags & ~TR_INFLUENCE_FIRST)) return -1;
    if (nterms > TR_INFLUENCE_MAX_TERMS) return -12;
    int32_t at = 0;
    if (flags & TR_INFLUENCE_FIRST)
        for (int32_t i = 0; i < p->k; i++) { if (q) { q[at].vars = (uint64_t)1 << i; q[at].flags = 0; q[at].reserved = 0; } at++; }
    for (int32_t j = 0; j < nterms; j++) {
        if (p->k < 64 && (sets[j] >> p->k)) return -1;
        if (q) { q[at].vars = sets[j]; q[at].flags = 0; q[at].reserved = 0; }
        at++;
    }
    *nq = at;
    return 0;
}
/* the CUDA work split: word slices, each with its node words, its accumulators and its bins */
#define TR_INFLUENCE_WORK (UINT64_C(1) << 14)
#define TR_INFLUENCE_MIN_WORDS 4
#define TR_INFLUENCE_SCRATCH (UINT64_C(1) << 26)
static inline uint64_t tr_influence_item_bytes(const tr_bool_program_view *p, int32_t np) {
    return tr_budget_add(tr_budget_add(tr_budget_mul((uint64_t)p->nnodes, 16), tr_budget_mul((uint64_t)np, sizeof(tr_influence_acc))),
                         tr_budget_mul((uint64_t)p->k + 5, 8));      /* bins (k + 1), f counts (3), all_known (1) */
}
static inline uint64_t tr_influence_slices(uint64_t nw, const tr_bool_program_view *p, int32_t np) {
    uint64_t s = TR_INFLUENCE_WORK, cap = (nw + TR_INFLUENCE_MIN_WORDS - 1) / TR_INFLUENCE_MIN_WORDS;
    uint64_t per = tr_influence_item_bytes(p, np);
    if (s > cap) s = cap;
    if (per && s > TR_INFLUENCE_SCRATCH / per) s = TR_INFLUENCE_SCRATCH / per;
    return s ? s : 1;
}
static inline uint64_t tr_influence_device_bytes(const tr_bool_domain *d, const tr_bool_program_view *p, int32_t np,
                                                 uint64_t ncones) {
    uint64_t bytes = 0, nw = d->nw;
    if (d->kind == TR_BOOL_ROWS) {
        bytes = tr_budget_mul(nw, 8);
        bytes = tr_budget_add(bytes, tr_budget_mul(tr_budget_mul((uint64_t)d->k, nw), d->valid ? 16 : 8));
    } else {
        bytes = tr_budget_mul(nw, 16);                                       /* the truth table */
    }
    bytes = tr_budget_add(bytes, tr_budget_mul((uint64_t)p->nnodes, 16));
    bytes = tr_budget_add(bytes, tr_budget_mul((uint64_t)p->nsched, 4));
    bytes = tr_budget_add(bytes, tr_budget_mul(tr_budget_mul((uint64_t)np, sizeof(tr_bool_plan)), 2));   /* + host */
    bytes = tr_budget_add(bytes, tr_budget_mul(ncones, 8));                                            /* + host */
    uint64_t items = tr_influence_slices(nw, p, np);
    bytes = tr_budget_add(bytes, tr_budget_mul(items, tr_influence_item_bytes(p, np)));
    bytes = tr_budget_add(bytes, tr_budget_mul(tr_influence_item_bytes(p, np), 2));                    /* totals + host */
    return bytes;
}
/* the needs of a request whose plans are known (see Budgets) */
static inline void tr_influence_base_need(const tr_bool_domain *d, const tr_bool_program_view *p, int32_t np,
                                          int32_t nterms, int32_t flags, uint64_t ncones, uint64_t tensor,
                                          int32_t max_order, tr_math_need *need) {
    for (int i = 0; i < TR_BUDGET_RESOURCES; i++) need->need[i] = 0;
    need->need[TR_BUDGET_VARIABLES] = (uint64_t)p->k;
    need->need[TR_BUDGET_DEGREE] = (uint64_t)max_order;
    need->need[TR_BUDGET_CIRCUIT] = (uint64_t)p->nnodes;
    need->need[TR_BUDGET_SUPPORT] = d->kind == TR_BOOL_ASSIGNMENTS ? d->points : 0;
    need->need[TR_BUDGET_TENSOR] = tensor;
    need->need[TR_BUDGET_OUTPUT] = tr_budget_add((uint64_t)nterms, (flags & TR_INFLUENCE_FIRST) ? 2 * (uint64_t)p->k + 1 : 0);
    uint64_t bytes = tr_budget_add(tr_budget_mul((uint64_t)np, sizeof(tr_bool_plan) + sizeof(tr_influence_acc)),
                                   tr_budget_mul(ncones, 4));
    bytes = tr_budget_add(bytes, tr_budget_mul((uint64_t)p->nnodes, 16));
    bytes = tr_budget_add(bytes, tr_budget_mul((uint64_t)p->k + 1, 8));
    if (d->kind == TR_BOOL_ASSIGNMENTS) bytes = tr_budget_add(bytes, tr_budget_mul(d->nw, 16));
    need->need[TR_BUDGET_BYTES] = bytes;
}
static inline uint64_t tr_influence_gcd(uint64_t a, uint64_t b) { while (b) { uint64_t t = a % b; a = b; b = t; } return a; }
/* the results of the totals (shared by both libraries) */
static inline void tr_influence_finish(const tr_bool_domain *d, const tr_bool_program_view *p, const tr_bool_plan *plans,
                                       int32_t np, int32_t first, int32_t nterms, int32_t flags,
                                       const tr_influence_acc *acc, const uint64_t *fcounts, const uint64_t *bins,
                                       uint64_t all_known, tr_influence_summary *summary, tr_influence_term *first_terms,
                                       uint64_t *sensitivity, tr_influence_term *terms) {
    const uint64_t points = d->selected;
    unsigned __int128 total = 0;
    for (int32_t j = 0; j < np; j++) {
        tr_influence_term *T = j < first ? &first_terms[j] : &terms[j - first];
        const tr_influence_acc *a = &acc[j];
        memset(T, 0, sizeof *T);
        T->vars = plans[j].vars; T->order = plans[j].order; T->measure = d->kind; T->points = points;
        T->known = a->known; T->unknown = a->unknown; T->sensitive = a->sensitive;
        T->positive = a->positive; T->negative = a->negative;
        T->sum.lo = a->sum_lo; T->sum.hi = a->sum_hi;
        uint64_t g = points ? tr_influence_gcd(a->sensitive, points) : 1;
        T->measure_num = points ? a->sensitive / g : 0;
        T->measure_den = points ? points / g : 1;
        if (j < first) total += a->sensitive;
    }
    memset(summary, 0, sizeof *summary);
    summary->points = points; summary->f_true = fcounts[0]; summary->f_false = fcounts[1]; summary->f_unknown = fcounts[2];
    const int histogram = (flags & TR_INFLUENCE_FIRST) != 0;
    summary->all_known = histogram ? all_known : 0;
    summary->total = tr_i128_set((__int128)total);
    unsigned __int128 g = points ? (unsigned __int128)points : 1, x = total;   /* gcd(total, points) */
    while (x) { unsigned __int128 t = g % x; g = x; x = t; }
    if (!points) { summary->total_num = tr_i128_set(0); summary->total_den = 1; }
    else { summary->total_num = tr_i128_set((__int128)(total / g)); summary->total_den = (uint64_t)(points / g); }
    summary->k = p->k; summary->measure = d->kind; summary->nterms = nterms; summary->flags = flags;
    if (histogram) {
        uint64_t top = 0;
        for (int32_t b = 0; b <= first; b++) { sensitivity[b] = bins[b]; if (bins[b]) top = (uint64_t)b; }
        summary->max_sensitivity = top;
    }
}

/* ------------------------------------------------------------------ entry points (libalg) */
/* the needs checked by tr_influence_compute (TR_INFLUENCE_NEED_CPU) or tr_mgpu_influence_compute (_DEVICE) */
int32_t tr_influence_need(const tr_bool_domain *, const tr_bool_program_view *, const uint64_t *sets, int32_t nterms,
                          int32_t flags, int32_t target, tr_math_need *);
/* The influences over the domain's measure: summary, first[k] and sensitivity[k + 1] with TR_INFLUENCE_FIRST (else
 * unused, may be NULL), terms[nterms] for the given sets. */
int32_t tr_influence_compute(const tr_bool_domain *, const tr_bool_program_view *, const uint64_t *sets, int32_t nterms,
                             int32_t flags, const tr_math_budget *, tr_influence_summary *summary,
                             tr_influence_term *first, uint64_t *sensitivity, tr_influence_term *terms,
                             tr_math_refusal *);
/* the sets of sizes min_order..max_order of k conditions (by size, then increasing mask): *n of them (sets NULL:
 * count only; capacity too small: TR_MATH_INVALID) */
int32_t tr_influence_sets(int32_t k, int32_t min_order, int32_t max_order, const tr_math_budget *, uint64_t *sets,
                          uint64_t capacity, uint64_t *n, tr_math_refusal *);

/* ------------------------------------------------------------------ the CUDA library (math_influence_gpu.cu)
 * tr_mgpu_influence_compute: tr_influence_compute on the device. ASSIGNMENTS: one launch builds the truth table (one
 * work item per word), a second scans it. ROWS: the masks' words are copied once. The scan's work items are word
 * slices; each evaluates every term of its words "all conditions at once" (tr_influence_word: the program once per
 * word, then the cones / substituted table words, bitwise with popcounts) into its own accumulators, then adds them
 * to the totals with one integer atomic per nonzero counter (the 128-bit sums with tr_atomic_add_wide). */
int tr_mgpu_influence_compute(const tr_bool_domain *, const tr_bool_program_view *, const uint64_t *sets,
                              int32_t nterms, int32_t flags, const tr_math_budget *, tr_influence_summary *summary,
                              tr_influence_term *first, uint64_t *sensitivity, tr_influence_term *terms,
                              tr_math_refusal *);
#ifdef __cplusplus
}
#endif
#endif
