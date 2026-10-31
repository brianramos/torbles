/* COUNT-001 on a CUDA device: the counts and smallest violating points of every constraint, and the bounded
 * extraction of the satisfying and the smallest violating points, with exactly the CPU's results (math_count.h).
 * SPDX-License-Identifier: Apache-2.0
 *
 * Counts: work items (constraint j, word slice s): slice s of S takes the words s, s + S, ...; a work item owns 16
 * bytes of node words per node (its slot of the scratch), evaluates the cone of constraint j at each of its words with
 * bitwise operations (a value and a valid word per node, SQL three-valued logic), splits the constraint's words into
 * satisfied, violated and UNKNOWN points, counts them with popcounts and keeps its first violating point (its words are
 * visited in increasing order), then adds its counts with one atomic each and its witness with an atomic minimum.
 * Extraction: work items (constraint j, chunk c of contiguous words): a first launch writes the chunk's satisfying and
 * violating counts into its own two slots; the host forms their exclusive prefix per constraint (and checks the totals
 * against the counts it was given); a second launch evaluates each chunk that holds requested ranks again and writes
 * every requested point at its rank (rank = the chunk's prefix + the points before it in the chunk): each output slot
 * is written by exactly one work item. Integer atomics commute and every other output has one writer, so the results do
 * not depend on thread order. */
#include "math_cuda.h"
#include "math_count.h"
#include "math_handles.h"

TR_MGPU_ABI(tr_count_constraint, TR_COUNT_ABI);   /* structs passed to the CUDA library (math_handles.h) */
TR_MGPU_ABI(tr_count_request, TR_COUNT_ABI);
TR_MGPU_ABI(tr_count_result, TR_COUNT_ABI);
TR_MGPU_ABI(tr_count_set_view, TR_COUNT_ABI);

/* acc[4 j + 0..3] = satisfied, violated, unknown, the smallest violating point (initially UINT64_MAX) */
TR_KERNEL void tr_count_counts_kernel(tr_bool_domain d, tr_count_set_view s, uint64_t items, uint64_t slices,
                                      uint64_t *scratch, uint64_t *acc) {
    TR_GRID_LOOP(item, items) {
        const int32_t j = (int32_t)(item / slices);
        const uint64_t sl = item % slices;
        const tr_count_constraint c = s.constraints[j];
        const tr_bool_program_view p = tr_count_cone_view(&s, j);
        uint64_t *val = scratch + item * 2 * (uint64_t)s.nnodes;
        uint64_t ns = 0, nv = 0, nu = 0, witness = UINT64_MAX;
        for (uint64_t w = sl; w < d.nw; w += slices) {
            uint64_t sel = tr_bool_selection_word(&d, w);
            if (!sel) continue;
            tr_bool_eval_base(&p, &d, w, val);
            uint64_t sat = 0, vio = 0, unk = 0;
            tr_count_classes(&c, val, sel, &sat, &vio, &unk);
            ns += (uint64_t)tr_bool_popc(sat); nv += (uint64_t)tr_bool_popc(vio); nu += (uint64_t)tr_bool_popc(unk);
            if (vio && witness == UINT64_MAX) witness = w * 64 + (uint64_t)tr_bool_ctz(vio);
        }
        if (ns) tr_atomic_add_u64(&acc[4 * (uint64_t)j], ns);
        if (nv) { tr_atomic_add_u64(&acc[4 * (uint64_t)j + 1], nv); tr_atomic_min_u64(&acc[4 * (uint64_t)j + 3], witness); }
        if (nu) tr_atomic_add_u64(&acc[4 * (uint64_t)j + 2], nu);
    }
}

/* want[4 j + 0..3] = rows_n, wit_n, rows_at, wit_at of constraint j; cs / cv: one slot per (constraint, chunk) */
TR_KERNEL void tr_count_chunk_kernel(tr_bool_domain d, tr_count_set_view s, const uint64_t *want, uint64_t items,
                                     uint64_t chunks, uint64_t cw, uint64_t *scratch, uint64_t *cs, uint64_t *cv) {
    TR_GRID_LOOP(item, items) {
        const int32_t j = (int32_t)(item / chunks);
        const uint64_t ch = item % chunks;
        uint64_t ns = 0, nv = 0;
        if (want[4 * (uint64_t)j] || want[4 * (uint64_t)j + 1]) {
            const tr_count_constraint c = s.constraints[j];
            const tr_bool_program_view p = tr_count_cone_view(&s, j);
            uint64_t *val = scratch + item * 2 * (uint64_t)s.nnodes;
            const uint64_t end = (ch + 1) * cw < d.nw ? (ch + 1) * cw : d.nw;
            for (uint64_t w = ch * cw; w < end; w++) {
                uint64_t sel = tr_bool_selection_word(&d, w);
                if (!sel) continue;
                tr_bool_eval_base(&p, &d, w, val);
                uint64_t sat = 0, vio = 0, unk = 0;
                tr_count_classes(&c, val, sel, &sat, &vio, &unk);
                ns += (uint64_t)tr_bool_popc(sat); nv += (uint64_t)tr_bool_popc(vio);
            }
        }
        cs[item] = ns; cv[item] = nv;
    }
}

/* ps / pv: the exclusive prefixes of cs / cv per constraint */
TR_KERNEL void tr_count_write_kernel(tr_bool_domain d, tr_count_set_view s, const uint64_t *want, const uint64_t *ps,
                                     const uint64_t *pv, uint64_t items, uint64_t chunks, uint64_t cw,
                                     uint64_t *scratch, uint64_t *points, uint64_t *witnesses) {
    TR_GRID_LOOP(item, items) {
        const int32_t j = (int32_t)(item / chunks);
        const uint64_t ch = item % chunks;
        const uint64_t rows_n = want[4 * (uint64_t)j], wit_n = want[4 * (uint64_t)j + 1];
        const uint64_t rows_at = want[4 * (uint64_t)j + 2], wit_at = want[4 * (uint64_t)j + 3];
        uint64_t rs = ps[item], rv = pv[item];
        if (!(rs < rows_n) && !(rv < wit_n)) continue;
        const tr_count_constraint c = s.constraints[j];
        const tr_bool_program_view p = tr_count_cone_view(&s, j);
        uint64_t *val = scratch + item * 2 * (uint64_t)s.nnodes;
        const uint64_t end = (ch + 1) * cw < d.nw ? (ch + 1) * cw : d.nw;
        for (uint64_t w = ch * cw; w < end && (rs < rows_n || rv < wit_n); w++) {
            uint64_t sel = tr_bool_selection_word(&d, w);
            if (!sel) continue;
            tr_bool_eval_base(&p, &d, w, val);
            uint64_t sat = 0, vio = 0, unk = 0;
            tr_count_classes(&c, val, sel, &sat, &vio, &unk);
            for (uint64_t bits = sat; bits && rs < rows_n; bits &= bits - 1)
                points[rows_at + rs++] = w * 64 + (uint64_t)tr_bool_ctz(bits);
            for (uint64_t bits = vio; bits && rv < wit_n; bits &= bits - 1)
                witnesses[wit_at + rv++] = w * 64 + (uint64_t)tr_bool_ctz(bits);
        }
    }
}

/* the device copies of a domain view and a set view (pointers swapped) */
typedef struct {
    tr_bool_domain d;
    tr_count_set_view s;
    void *mem[24];
    int nmem;
} tr_count_device;

static void tr_count_device_free(tr_count_device *g) {
    for (int i = 0; i < g->nmem; i++) tr_cuda_free(g->mem[i]);
    g->nmem = 0;
}
static int tr_count_device_alloc(tr_count_device *g, void **ptr, uint64_t bytes) {
    int rc = tr_cuda_alloc(ptr, (size_t)(bytes ? bytes : 1));
    if (!rc) g->mem[g->nmem++] = *ptr;
    return rc;
}
static int tr_count_device_copy(const tr_bool_domain *dom, const tr_count_set_view *set, tr_count_device *g) {
    memset(g, 0, sizeof *g);
    g->d = *dom; g->s = *set;
    int rc = 0;
    const uint64_t nw = dom->nw;
    if (dom->kind == TR_BOOL_ROWS) {
        uint64_t words = (uint64_t)dom->k * nw;
        uint64_t *sel = NULL, *value = NULL, *valid = NULL;
        rc = tr_count_device_alloc(g, (void **)&sel, nw * 8);
        if (!rc && nw) rc = tr_cuda_copy_in(sel, dom->selection, (size_t)(nw * 8));
        if (!rc) rc = tr_count_device_alloc(g, (void **)&value, words * 8);
        if (!rc && words) rc = tr_cuda_copy_in(value, dom->value, (size_t)(words * 8));
        if (!rc && dom->valid) {
            rc = tr_count_device_alloc(g, (void **)&valid, words * 8);
            if (!rc && words) rc = tr_cuda_copy_in(valid, dom->valid, (size_t)(words * 8));
        }
        g->d.selection = sel; g->d.value = value; g->d.valid = valid;
    }
    tr_bool_node *nodes = NULL;
    uint32_t *sched = NULL, *cones = NULL;
    tr_count_constraint *cons = NULL;
    uint64_t *at = NULL;
    const uint64_t ncones = set->cone_at[set->nc];
    if (!rc) rc = tr_count_device_alloc(g, (void **)&nodes, (uint64_t)set->nnodes * sizeof(tr_bool_node));
    if (!rc) rc = tr_cuda_copy_in(nodes, set->nodes, (size_t)set->nnodes * sizeof(tr_bool_node));
    if (!rc) rc = tr_count_device_alloc(g, (void **)&sched, (uint64_t)set->nsched * 4);
    if (!rc) rc = tr_cuda_copy_in(sched, set->schedule, (size_t)set->nsched * 4);
    if (!rc) rc = tr_count_device_alloc(g, (void **)&cons, (uint64_t)set->nc * sizeof(tr_count_constraint));
    if (!rc) rc = tr_cuda_copy_in(cons, set->constraints, (size_t)set->nc * sizeof(tr_count_constraint));
    if (!rc) rc = tr_count_device_alloc(g, (void **)&cones, ncones * 4);
    if (!rc && ncones) rc = tr_cuda_copy_in(cones, set->cones, (size_t)(ncones * 4));
    if (!rc) rc = tr_count_device_alloc(g, (void **)&at, ((uint64_t)set->nc + 1) * 8);
    if (!rc) rc = tr_cuda_copy_in(at, set->cone_at, (size_t)((uint64_t)set->nc + 1) * 8);
    g->s.nodes = nodes; g->s.schedule = sched; g->s.constraints = cons; g->s.cones = cones; g->s.cone_at = at;
    g->s.deps = NULL;                                         /* never read by the kernels (the cones are planned) */
    if (rc) tr_count_device_free(g);
    return rc;
}
/* the common checks of both entry points */
static int tr_count_device_check(const tr_bool_domain *d, const tr_count_set_view *s) {
    int rc = tr_bool_domain_check(d);
    if (!rc) rc = tr_count_set_check(s);
    if (rc) return rc;
    return d->k != s->k ? -6 : 0;                                                            /* TR_MATH_DOMAIN_ERROR */
}
static void tr_count_device_need(const tr_bool_domain *d, const tr_count_set_view *s, tr_math_need *need) {
    for (int i = 0; i < TR_BUDGET_RESOURCES; i++) need->need[i] = 0;
    need->need[TR_BUDGET_VARIABLES] = (uint64_t)s->k;
    need->need[TR_BUDGET_CIRCUIT] = (uint64_t)s->nnodes;
    need->need[TR_BUDGET_SUPPORT] = d->kind == TR_BOOL_ASSIGNMENTS ? d->points : 0;
    need->need[TR_BUDGET_OUTPUT] = (uint64_t)s->nc;
}

#ifdef __cplusplus
extern "C" {
#endif
int tr_mgpu_count_counts(const tr_bool_domain *dom, const tr_count_set_view *set, const tr_math_budget *budget,
                         tr_count_result *results, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    int rc = tr_count_device_check(dom, set);
    if (rc) return rc;
    if (!results) return -1;                                                                 /* TR_MATH_INVALID */
    tr_math_budget dflt;
    if (!budget) { tr_budget_default(&dflt); budget = &dflt; }
    tr_math_need need;
    tr_count_device_need(dom, set, &need);
    need.need[TR_BUDGET_BYTES] = tr_count_device_bytes(dom, set);
    rc = tr_budget_check(budget, &need, refusal);
    if (rc) return rc;
    for (int32_t j = 0; j < set->nc; j++) {
        tr_count_result *R = &results[j];
        memset(R, 0, sizeof *R);
        R->kind = set->constraints[j].kind; R->a = set->constraints[j].a; R->b = set->constraints[j].b;
        R->points = dom->selected; R->witness = TR_COUNT_NONE;
    }
    if (!dom->selected || !dom->nw) return 0;                                  /* nothing counts: every count is zero */
    const uint64_t slices = tr_count_slices(dom->nw, set->nc, set->nnodes), items = (uint64_t)set->nc * slices;
    uint64_t *acc = (uint64_t *)calloc((size_t)set->nc, 32);
    if (!acc) return -2;                                                                     /* TR_MATH_MEMORY */
    for (int32_t j = 0; j < set->nc; j++) acc[4 * (uint64_t)j + 3] = UINT64_MAX;
    tr_count_device g;
    rc = tr_count_device_copy(dom, set, &g);
    uint64_t *dscratch = NULL, *dacc = NULL;
    if (!rc) rc = tr_count_device_alloc(&g, (void **)&dscratch, items * 16 * (uint64_t)set->nnodes);
    if (!rc) rc = tr_count_device_alloc(&g, (void **)&dacc, (uint64_t)set->nc * 32);
    if (!rc) rc = tr_cuda_copy_in(dacc, acc, (size_t)set->nc * 32);
    if (!rc) { TR_LAUNCH(tr_count_counts_kernel, items, g.d, g.s, items, slices, dscratch, dacc); rc = tr_cuda_sync(); }
    if (!rc) rc = tr_cuda_copy_out(acc, dacc, (size_t)set->nc * 32);
    if (!rc)
        for (int32_t j = 0; j < set->nc; j++) {
            tr_count_result *R = &results[j];
            const uint64_t *a = acc + 4 * (uint64_t)j;
            R->satisfied = a[0]; R->violated = a[1]; R->unknown = a[2];
            R->witness = a[1] ? a[3] : TR_COUNT_NONE;
        }
    tr_count_device_free(&g);
    free(acc);
    return rc;
}

int tr_mgpu_count_extract(const tr_bool_domain *dom, const tr_count_set_view *set, const tr_count_request *q,
                          tr_count_result *results, const tr_math_budget *budget, uint64_t *points,
                          uint64_t *witnesses, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    int rc = tr_count_device_check(dom, set);
    if (rc) return rc;
    if (!results || !q) return -1;
    for (int32_t j = 0; j < set->nc; j++) {                   /* the counts are this set's */
        const tr_count_constraint *c = &set->constraints[j];
        if (results[j].kind != c->kind || results[j].a != c->a || results[j].b != c->b ||
            results[j].points != dom->selected) return -1;
    }
    uint64_t pt = 0, wt = 0;
    rc = tr_count_layout(results, set->nc, q, &pt, &wt);
    if (rc) return rc;
    if ((pt && !points) || (wt && !witnesses)) return -1;
    if (!pt && !wt) return 0;                                 /* nothing requested: the layout alone */
    tr_math_budget dflt;
    if (!budget) { tr_budget_default(&dflt); budget = &dflt; }
    tr_math_need need;
    tr_count_device_need(dom, set, &need);
    need.need[TR_BUDGET_OUTPUT] = tr_budget_add((uint64_t)set->nc, tr_budget_add(pt, wt));
    need.need[TR_BUDGET_BYTES] = tr_count_extract_device_bytes(dom, set, pt, wt);
    rc = tr_budget_check(budget, &need, refusal);
    if (rc) return rc;
    const int32_t nc = set->nc;
    const uint64_t cw = tr_count_chunk_words(dom->nw, nc, set->nnodes), chunks = tr_count_chunks(dom->nw, cw);
    const uint64_t items = (uint64_t)nc * chunks;
    uint64_t *want = (uint64_t *)calloc((size_t)nc, 32);
    uint64_t *cs = (uint64_t *)calloc((size_t)(items ? items : 1), 8), *cv = (uint64_t *)calloc((size_t)(items ? items : 1), 8);
    uint64_t *ps = (uint64_t *)calloc((size_t)(items ? items : 1), 8), *pv = (uint64_t *)calloc((size_t)(items ? items : 1), 8);
    if (!want || !cs || !cv || !ps || !pv) { free(want); free(cs); free(cv); free(ps); free(pv); return -2; }
    for (int32_t j = 0; j < nc; j++) {
        uint64_t *x = want + 4 * (uint64_t)j;
        x[0] = results[j].rows_n; x[1] = results[j].wit_n; x[2] = results[j].rows_at; x[3] = results[j].wit_at;
    }
    tr_count_device g;
    rc = tr_count_device_copy(dom, set, &g);
    uint64_t *dwant = NULL, *dscratch = NULL, *dcs = NULL, *dcv = NULL, *dps = NULL, *dpv = NULL, *dpts = NULL, *dwit = NULL;
    if (!rc) rc = tr_count_device_alloc(&g, (void **)&dwant, (uint64_t)nc * 32);
    if (!rc) rc = tr_cuda_copy_in(dwant, want, (size_t)nc * 32);
    if (!rc) rc = tr_count_device_alloc(&g, (void **)&dscratch, items * 16 * (uint64_t)set->nnodes);
    if (!rc) rc = tr_count_device_alloc(&g, (void **)&dcs, items * 8);
    if (!rc) rc = tr_count_device_alloc(&g, (void **)&dcv, items * 8);
    if (!rc) rc = tr_count_device_alloc(&g, (void **)&dps, items * 8);
    if (!rc) rc = tr_count_device_alloc(&g, (void **)&dpv, items * 8);
    if (!rc) rc = tr_count_device_alloc(&g, (void **)&dpts, pt * 8);
    if (!rc) rc = tr_count_device_alloc(&g, (void **)&dwit, wt * 8);
    if (!rc && items) {
        TR_LAUNCH(tr_count_chunk_kernel, items, g.d, g.s, dwant, items, chunks, cw, dscratch, dcs, dcv);
        rc = tr_cuda_sync();
    }
    if (!rc && items) rc = tr_cuda_copy_out(cs, dcs, (size_t)(items * 8));
    if (!rc && items) rc = tr_cuda_copy_out(cv, dcv, (size_t)(items * 8));
    for (int32_t j = 0; j < nc && !rc; j++) {                  /* exclusive prefixes, checked against the counts */
        uint64_t s = 0, v = 0;
        for (uint64_t ch = 0; ch < chunks; ch++) {
            uint64_t at = (uint64_t)j * chunks + ch;
            ps[at] = s; pv[at] = v;
            s += cs[at]; v += cv[at];
        }
        int wanted = results[j].rows_n || results[j].wit_n;
        if (wanted && (s != results[j].satisfied || v != results[j].violated)) rc = -1;   /* not this domain's counts */
    }
    if (!rc && items) rc = tr_cuda_copy_in(dps, ps, (size_t)(items * 8));
    if (!rc && items) rc = tr_cuda_copy_in(dpv, pv, (size_t)(items * 8));
    if (!rc && items) {
        TR_LAUNCH(tr_count_write_kernel, items, g.d, g.s, dwant, dps, dpv, items, chunks, cw, dscratch, dpts, dwit);
        rc = tr_cuda_sync();
    }
    if (!rc && pt) rc = tr_cuda_copy_out(points, dpts, (size_t)(pt * 8));
    if (!rc && wt) rc = tr_cuda_copy_out(witnesses, dwit, (size_t)(wt * 8));
    tr_count_device_free(&g);
    free(want); free(cs); free(cv); free(ps); free(pv);
    return rc;
}
#ifdef __cplusplus
}
#endif
