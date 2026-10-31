/* BOOL-002: influences and interactions of a predicate under the uniform and the empirical measure; see
 * math_influence.h.
 * SPDX-License-Identifier: Apache-2.0 */
#include "math_influence.h"
#include "mathematics.h"
#include "math_domain.h"
#include "math_handles.h"
#include <stdlib.h>
#include <string.h>

TR_MATH_ABI(tr_influence_term, TR_INFLUENCE_ABI);      /* the ctypes boundary of BOOL-002 (math_handles.h) */
TR_MATH_ABI(tr_influence_summary, TR_INFLUENCE_ABI);

_Static_assert(sizeof(tr_influence_term) == 96, "tr_influence_term ABI");
_Static_assert(sizeof(tr_influence_summary) == 104, "tr_influence_summary ABI");
_Static_assert(sizeof(tr_influence_acc) == 64, "tr_influence_acc layout");

static void *zalloc(uint64_t bytes) {
    if (bytes > (uint64_t)SIZE_MAX) return NULL;
    return calloc(1, bytes ? (size_t)bytes : 1);
}
static const tr_math_budget *budget_or_default(const tr_math_budget *b, tr_math_budget *dflt) {
    if (b) return b;
    tr_budget_default(dflt);
    return dflt;
}

/* the checks of a request; its queries (allocated, *nq of them) and plan totals */
static int32_t request(const tr_bool_domain *d, const tr_bool_program_view *p, const uint64_t *sets, int32_t nterms,
                       int32_t flags, tr_bool_query **queries, int32_t *nq, uint64_t *ncones, uint64_t *tensor,
                       int32_t *top) {
    int32_t rc = tr_bool_domain_check(d);
    if (!rc) rc = tr_bool_program_check(p);
    if (rc) return rc;
    if (d->k != p->k) return TR_MATH_DOMAIN_ERROR;
    rc = tr_influence_queries(p, sets, nterms, flags, NULL, nq);
    if (rc) return rc;
    tr_bool_query *q = zalloc((uint64_t)(*nq > 0 ? *nq : 1) * sizeof *q);
    if (!q) return TR_MATH_MEMORY;
    tr_influence_queries(p, sets, nterms, flags, q, nq);
    rc = tr_bool_plan_queries(p, q, *nq, NULL, NULL, ncones, NULL, NULL, tensor, top);
    if (rc) { free(q); return rc; }
    *queries = q;
    return TR_MATH_OK;
}

int32_t tr_influence_need(const tr_bool_domain *d, const tr_bool_program_view *p, const uint64_t *sets, int32_t nterms,
                          int32_t flags, int32_t target, tr_math_need *need) {
    if (!need) return TR_MATH_INVALID;
    tr_bool_query *q = NULL;
    int32_t nq = 0, top = 0;
    uint64_t ncones = 0, tensor = 0;
    int32_t rc = request(d, p, sets, nterms, flags, &q, &nq, &ncones, &tensor, &top);
    if (rc) return rc;
    free(q);
    tr_influence_base_need(d, p, nq, nterms, flags, ncones, tensor, top, need);
    if (target == TR_INFLUENCE_NEED_DEVICE) need->need[TR_BUDGET_BYTES] = tr_influence_device_bytes(d, p, nq, ncones);
    else if (target != TR_INFLUENCE_NEED_CPU) return TR_MATH_INVALID;
    return TR_MATH_OK;
}

int32_t tr_influence_compute(const tr_bool_domain *d, const tr_bool_program_view *p, const uint64_t *sets,
                             int32_t nterms, int32_t flags, const tr_math_budget *budget,
                             tr_influence_summary *summary, tr_influence_term *first, uint64_t *sensitivity,
                             tr_influence_term *terms, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    tr_bool_query *q = NULL;
    int32_t nq = 0, top = 0;
    uint64_t ncones = 0, tensor = 0;
    int32_t rc = request(d, p, sets, nterms, flags, &q, &nq, &ncones, &tensor, &top);
    if (rc) return rc;
    const int32_t nfirst = (flags & TR_INFLUENCE_FIRST) ? p->k : 0;
    if (!summary || (nfirst && (!first || !sensitivity)) || (nterms && !terms)) { free(q); return TR_MATH_INVALID; }
    tr_math_need need;
    tr_influence_base_need(d, p, nq, nterms, flags, ncones, tensor, top, &need);
    tr_math_budget dflt;
    rc = tr_budget_check(budget_or_default(budget, &dflt), &need, refusal);
    if (rc) { free(q); return rc; }
    tr_bool_plan *plans = zalloc((uint64_t)(nq > 0 ? nq : 1) * sizeof *plans);
    uint32_t *cones = zalloc(ncones * 4);
    uint64_t *val = zalloc((uint64_t)p->nnodes * 16);
    tr_influence_acc *acc = zalloc((uint64_t)(nq > 0 ? nq : 1) * sizeof *acc);
    uint64_t *bins = zalloc(((uint64_t)p->k + 1) * 8);
    uint64_t *table = d->kind == TR_BOOL_ASSIGNMENTS ? zalloc(d->nw * 16) : NULL;
    if (!plans || !cones || !val || !acc || !bins || (d->kind == TR_BOOL_ASSIGNMENTS && !table)) {
        free(q); free(plans); free(cones); free(val); free(acc); free(bins); free(table);
        return TR_MATH_MEMORY;
    }
    tr_bool_plan_queries(p, q, nq, plans, cones, NULL, NULL, NULL, NULL, NULL);
    free(q);
    const uint64_t *tv = NULL, *tk = NULL;
    if (table) {                                              /* the truth table of the 2^k assignments, once */
        for (uint64_t w = 0; w < d->nw; w++) {
            tr_bool_eval_base(p, d, w, val);
            table[w] = val[2 * (uint64_t)p->root];
            table[d->nw + w] = val[2 * (uint64_t)p->root + 1];
        }
        tv = table; tk = table + d->nw;
    }
    uint64_t fcounts[3] = {0, 0, 0}, all_known = 0;
    for (uint64_t w = 0; w < d->nw; w++)
        tr_influence_word(d, p, plans, nq, cones, nfirst, (flags & TR_INFLUENCE_FIRST) != 0, tv, tk, w, val, acc,
                          fcounts, bins, &all_known);
    tr_influence_finish(d, p, plans, nq, nfirst, nterms, flags, acc, fcounts, bins, all_known, summary, first,
                        sensitivity, terms);
    free(plans); free(cones); free(val); free(acc); free(bins); free(table);
    return TR_MATH_OK;
}

int32_t tr_influence_sets(int32_t k, int32_t min_order, int32_t max_order, const tr_math_budget *budget, uint64_t *sets,
                          uint64_t capacity, uint64_t *n, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (k < 0 || k > TR_BOOL_MAX_CONDITIONS || min_order < 0 || max_order < min_order || !n) return TR_MATH_INVALID;
    int32_t top = max_order < k ? max_order : k;
    uint64_t count = 0;
    for (int32_t m = min_order; m <= top; m++) count = tr_budget_add(count, tr_budget_binomial((uint64_t)k, (uint64_t)m));
    tr_math_need need;
    for (int i = 0; i < TR_BUDGET_RESOURCES; i++) need.need[i] = 0;
    need.need[TR_BUDGET_VARIABLES] = (uint64_t)k;
    need.need[TR_BUDGET_OUTPUT] = count;
    tr_math_budget dflt;
    int32_t rc = tr_budget_check(budget_or_default(budget, &dflt), &need, refusal);
    if (rc) return rc;
    if (!sets) { *n = count; return TR_MATH_OK; }
    if (capacity < count) return TR_MATH_INVALID;
    uint64_t at = 0;
    const unsigned __int128 limit = (unsigned __int128)1 << k;   /* k <= 64: 128-bit masks never overflow */
    for (int32_t m = min_order; m <= top; m++) {
        unsigned __int128 s = ((unsigned __int128)1 << m) - 1;   /* Gosper's hack: the m-subsets, increasing */
        if (m == 0) { sets[at++] = 0; continue; }
        while (s < limit) {
            sets[at++] = (uint64_t)s;
            unsigned __int128 c = s & (~s + 1), r = s + c;
            s = (((r ^ s) >> 2) / c) | r;
        }
    }
    *n = at;
    return TR_MATH_OK;
}
