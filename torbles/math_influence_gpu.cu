/* BOOL-002 on a CUDA device: influences and interactions of a predicate under the uniform measure (from its truth
 * table) and the empirical measure (cofactors per row of the masks), with exactly the CPU's counts
 * (math_influence.h).
 * SPDX-License-Identifier: Apache-2.0
 *
 * ASSIGNMENTS: tr_influence_table_kernel builds the truth table (one work item per 64-assignment word: the program on
 * the generated condition words). The scan, both domains: work items are word slices (slice s takes the words s,
 * s + S, ...); each owns its node words and its accumulators (one per term, plus the sensitivity bins and the
 * predicate's counts) in the scratch, evaluates every term of each of its words at once with the shared
 * tr_influence_word (the program once per word, then per term and assignment the cone or the substituted table words,
 * bitwise, counted with popcounts), and at the end adds each nonzero counter to the totals with one integer atomic
 * (the 128-bit signed sums with tr_atomic_add_wide). Integer additions commute: the counts do not depend on thread
 * order. */
#include "math_cuda.h"
#include "math_influence.h"
#include "math_handles.h"

TR_MGPU_ABI(tr_influence_term, TR_INFLUENCE_ABI);      /* structs passed to the CUDA library (math_handles.h) */
TR_MGPU_ABI(tr_influence_summary, TR_INFLUENCE_ABI);

TR_KERNEL void tr_influence_table_kernel(tr_bool_domain d, tr_bool_program_view p, uint64_t *scratch, uint64_t items,
                                         uint64_t *table) {
    TR_GRID_LOOP(item, items) {
        uint64_t *val = scratch + item * 2 * (uint64_t)p.nnodes;
        for (uint64_t w = item; w < d.nw; w += items) {
            tr_bool_eval_base(&p, &d, w, val);
            table[w] = val[2 * (uint64_t)p.root];
            table[d.nw + w] = val[2 * (uint64_t)p.root + 1];
        }
    }
}

/* the item layout of the scratch: node words (2 nnodes), accumulators (np), bins (k + 1), f counts (3), all_known */
TR_KERNEL void tr_influence_scan_kernel(tr_bool_domain d, tr_bool_program_view p, const tr_bool_plan *plans,
                                        int32_t np, const uint32_t *cones, int32_t first, int32_t histogram,
                                        const uint64_t *table,
                                        uint64_t items, uint64_t item_words, uint64_t *scratch, tr_influence_acc *acc,
                                        uint64_t *totals) {
    TR_GRID_LOOP(item, items) {
        uint64_t *base = scratch + item * item_words;
        uint64_t *val = base;
        tr_influence_acc *a = (tr_influence_acc *)(void *)(base + 2 * (uint64_t)p.nnodes);
        uint64_t *bins = (uint64_t *)(void *)(a + np), *fcounts = bins + p.k + 1, *allk = fcounts + 3;
        const uint64_t *tv = table, *tk = table ? table + d.nw : NULL;
        for (uint64_t w = item; w < d.nw; w += items)
            tr_influence_word(&d, &p, plans, np, cones, first, histogram, tv, tk, w, val, a, fcounts, bins, allk);
        for (int32_t j = 0; j < np; j++) {
            const tr_influence_acc *x = &a[j];
            tr_influence_acc *g = &acc[j];
            if (x->known) tr_atomic_add_u64(&g->known, x->known);
            if (x->unknown) tr_atomic_add_u64(&g->unknown, x->unknown);
            if (x->sensitive) tr_atomic_add_u64(&g->sensitive, x->sensitive);
            if (x->positive) tr_atomic_add_u64(&g->positive, x->positive);
            if (x->negative) tr_atomic_add_u64(&g->negative, x->negative);
            tr_math_i128 s;
            s.lo = x->sum_lo; s.hi = x->sum_hi;
            __int128 v = tr_i128_get(s);
            if (v) tr_atomic_add_wide(&g->sum_lo, &g->sum_hi, v);
        }
        for (int32_t b = 0; b < p.k + 5; b++) if (bins[b]) tr_atomic_add_u64(&totals[b], bins[b]);   /* bins, f, allk */
    }
}

#ifdef __cplusplus
extern "C" {
#endif
int tr_mgpu_influence_compute(const tr_bool_domain *dom, const tr_bool_program_view *prog, const uint64_t *sets,
                              int32_t nterms, int32_t flags, const tr_math_budget *budget,
                              tr_influence_summary *summary, tr_influence_term *first, uint64_t *sensitivity,
                              tr_influence_term *terms, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    int rc = tr_bool_domain_check(dom);
    if (!rc) rc = tr_bool_program_check(prog);
    if (rc) return rc;
    if (dom->k != prog->k) return -6;                                                       /* TR_MATH_DOMAIN_ERROR */
    int32_t nq = 0, top = 0;
    rc = tr_influence_queries(prog, sets, nterms, flags, NULL, &nq);
    if (rc) return rc;
    const int32_t nfirst = (flags & TR_INFLUENCE_FIRST) ? prog->k : 0;
    if (!summary || (nfirst && (!first || !sensitivity)) || (nterms && !terms)) return -1;   /* TR_MATH_INVALID */
    tr_bool_query *q = (tr_bool_query *)calloc((size_t)(nq > 0 ? nq : 1), sizeof(tr_bool_query));
    if (!q) return -2;                                                                      /* TR_MATH_MEMORY */
    tr_influence_queries(prog, sets, nterms, flags, q, &nq);
    uint64_t ncones = 0, tensor = 0;
    rc = tr_bool_plan_queries(prog, q, nq, NULL, NULL, &ncones, NULL, NULL, &tensor, &top);
    if (rc) { free(q); return rc; }
    tr_math_budget dflt;
    if (!budget) { tr_budget_default(&dflt); budget = &dflt; }
    tr_math_need need;
    tr_influence_base_need(dom, prog, nq, nterms, flags, ncones, tensor, top, &need);
    need.need[TR_BUDGET_BYTES] = tr_influence_device_bytes(dom, prog, nq, ncones);
    rc = tr_budget_check(budget, &need, refusal);
    if (rc) { free(q); return rc; }
    tr_bool_plan *plans = (tr_bool_plan *)calloc((size_t)(nq > 0 ? nq : 1), sizeof(tr_bool_plan));
    uint32_t *cones = (uint32_t *)calloc((size_t)(ncones ? ncones : 1), 4);
    const uint64_t item_words = tr_influence_item_bytes(prog, nq) / 8;
    tr_influence_acc *acc = (tr_influence_acc *)calloc((size_t)(nq > 0 ? nq : 1), sizeof(tr_influence_acc));
    uint64_t *totals = (uint64_t *)calloc((size_t)prog->k + 5, 8);
    if (!plans || !cones || !acc || !totals) { free(q); free(plans); free(cones); free(acc); free(totals); return -2; }
    tr_bool_plan_queries(prog, q, nq, plans, cones, NULL, NULL, NULL, NULL, NULL);
    free(q);
    const uint64_t nw = dom->nw, items = tr_influence_slices(nw, prog, nq);
    tr_bool_domain d = *dom;
    tr_bool_program_view p = *prog;
    p.deps = NULL;                                            /* never read by the kernels (the cones are planned) */
    void *mem[16];
    int nmem = 0;
    uint64_t *dsel = NULL, *dvalue = NULL, *dvalid = NULL, *dtable = NULL, *dscratch = NULL, *dtotals = NULL;
    tr_bool_node *dnodes = NULL;
    uint32_t *dsched = NULL, *dcones = NULL;
    tr_bool_plan *dplans = NULL;
    tr_influence_acc *dacc = NULL;
#define TR_INFLUENCE_DEV(ptr, bytes) do { if (!rc) { rc = tr_cuda_alloc((void **)&(ptr), (size_t)(bytes)); if (!rc) mem[nmem++] = (void *)(ptr); } } while (0)
    if (dom->kind == TR_BOOL_ROWS) {
        uint64_t words = (uint64_t)dom->k * nw;
        TR_INFLUENCE_DEV(dsel, (nw ? nw : 1) * 8);
        if (!rc && nw) rc = tr_cuda_copy_in(dsel, dom->selection, (size_t)(nw * 8));
        TR_INFLUENCE_DEV(dvalue, (words ? words : 1) * 8);
        if (!rc && words) rc = tr_cuda_copy_in(dvalue, dom->value, (size_t)(words * 8));
        if (dom->valid) {
            TR_INFLUENCE_DEV(dvalid, (words ? words : 1) * 8);
            if (!rc && words) rc = tr_cuda_copy_in(dvalid, dom->valid, (size_t)(words * 8));
        }
        d.selection = dsel; d.value = dvalue; d.valid = dvalid;
    }
    TR_INFLUENCE_DEV(dnodes, (uint64_t)prog->nnodes * sizeof(tr_bool_node));
    if (!rc) rc = tr_cuda_copy_in(dnodes, prog->nodes, (size_t)prog->nnodes * sizeof(tr_bool_node));
    TR_INFLUENCE_DEV(dsched, (uint64_t)prog->nsched * 4);
    if (!rc) rc = tr_cuda_copy_in(dsched, prog->schedule, (size_t)prog->nsched * 4);
    p.nodes = dnodes; p.schedule = dsched;
    TR_INFLUENCE_DEV(dplans, (uint64_t)(nq > 0 ? nq : 1) * sizeof(tr_bool_plan));
    if (!rc && nq) rc = tr_cuda_copy_in(dplans, plans, (size_t)nq * sizeof(tr_bool_plan));
    TR_INFLUENCE_DEV(dcones, (ncones ? ncones : 1) * 4);
    if (!rc && ncones) rc = tr_cuda_copy_in(dcones, cones, (size_t)(ncones * 4));
    TR_INFLUENCE_DEV(dscratch, items * item_words * 8);
    TR_INFLUENCE_DEV(dacc, (uint64_t)(nq > 0 ? nq : 1) * sizeof(tr_influence_acc));
    TR_INFLUENCE_DEV(dtotals, ((uint64_t)prog->k + 5) * 8);
    if (dom->kind == TR_BOOL_ASSIGNMENTS) {
        TR_INFLUENCE_DEV(dtable, nw * 16);
        if (!rc) { TR_LAUNCH(tr_influence_table_kernel, items, d, p, dscratch, items, dtable); rc = tr_cuda_sync(); }
        if (!rc) rc = tr_cuda_zero(dscratch, (size_t)(items * item_words * 8));   /* the scan's accumulators start at 0 */
    }
#undef TR_INFLUENCE_DEV
    const int32_t histogram = (flags & TR_INFLUENCE_FIRST) != 0;
    if (!rc && dom->selected) {
        TR_LAUNCH(tr_influence_scan_kernel, items, d, p, dplans, nq, dcones, nfirst, histogram, dtable, items,
                  item_words, dscratch, dacc, dtotals);
        rc = tr_cuda_sync();
    }
    if (!rc) rc = tr_cuda_copy_out(acc, dacc, (size_t)(nq > 0 ? nq : 1) * sizeof(tr_influence_acc));
    if (!rc) rc = tr_cuda_copy_out(totals, dtotals, ((size_t)prog->k + 5) * 8);
    if (!rc) {
        const uint64_t *bins = totals, *fcounts = totals + prog->k + 1;
        tr_influence_finish(dom, prog, plans, nq, nfirst, nterms, flags, acc, fcounts, bins, fcounts[3], summary,
                            first, sensitivity, terms);
    }
    for (int i = 0; i < nmem; i++) tr_cuda_free(mem[i]);
    free(plans); free(cones); free(acc); free(totals);
    return rc;
}
#ifdef __cplusplus
}
#endif
