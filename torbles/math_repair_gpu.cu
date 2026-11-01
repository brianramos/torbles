/* ATTR-002 on a CUDA device: batched evaluation of candidate condition subsets and repairs (one work item per
 * candidate), the minimum feasible cost and its ties collected with atomics, and the shared exhaustive and
 * branch-and-bound drivers of math_repair.h running on them; with exactly the results of math_repair.c.
 * SPDX-License-Identifier: Apache-2.0
 *
 * eval    one work item per candidate subset: the shared tr_repair_eval_one (a hitting-set count over the minimal
 *         masks, a word-parallel BOOL-001 program with the absent conditions at their baseline compared with the
 *         target selection by popcounts, an ATTR-001 game value, or the aggregate over the rows a deletion keeps) into
 *         its own slot; scratch rows within TR_REPAIR_DEVICE_SCRATCH bytes per launch.
 * scan    launch 1: one work item per candidate, the feasible bitmap by atomic OR and the smallest feasible cost by
 *         atomic minimum; launch 2: the tie bitmap (feasible and of the cost `match`, or of the minimum) by atomic OR.
 *         Nothing depends on the order of the threads.
 * search  the drivers of math_repair.h with these evaluators (the same batches, the same exact values: the same
 *         decisions as the CPU).
 * The host checks the view, the plan and the budget (bytes = every device copy, the batch buffers and the scratch
 * chunk) before anything is allocated on the device. */
#include "math_cuda.h"
#include "math_repair.h"
#include "math_handles.h"

TR_MGPU_ABI(tr_repair_view, 1);          /* structs passed to the CUDA library (math_handles.h) */
TR_MGPU_ABI(tr_repair_out, 1);
TR_MGPU_ABI(tr_repair_plan, 1);
TR_MGPU_ABI(tr_repair_info, 1);

TR_KERNEL void tr_repair_eval_kernel(tr_repair_view v, const uint64_t *masks, uint64_t first, uint64_t count,
                                     uint64_t *scratch, uint64_t row, tr_repair_out *out) {
    TR_GRID_LOOP(i, count) tr_repair_eval_one(&v, masks[first + i], scratch + i * row, &out[first + i]);
}
TR_KERNEL void tr_repair_scan_kernel(tr_repair_view v, const uint64_t *masks, const int64_t *costs, uint64_t first,
                                     uint64_t count, uint64_t *scratch, uint64_t row, uint64_t *feasible,
                                     int64_t *mincost) {
    TR_GRID_LOOP(i, count) {
        tr_repair_out o;
        const uint64_t q = first + i;
        tr_repair_eval_one(&v, masks[q], scratch + i * row, &o);
        if (o.feasible) {
            tr_atomic_or_u64(&feasible[q >> 6], (uint64_t)1 << (q & 63));
            tr_atomic_min_i64(mincost, costs[q]);
        }
    }
}
TR_KERNEL void tr_repair_tie_kernel(const int64_t *costs, uint64_t count, const uint64_t *feasible,
                                    const int64_t *match, uint64_t *ties) {
    TR_GRID_LOOP(q, count)
        if (((feasible[q >> 6] >> (q & 63)) & 1) && costs[q] == *match) tr_atomic_or_u64(&ties[q >> 6], (uint64_t)1 << (q & 63));
}

/* the view's device copies (pointers replaced) and the buffers of `cap` candidates */
typedef struct {
    tr_repair_view d;
    void *mem[20];
    int nmem;
    uint64_t cap, row, chunk;
    uint64_t *dmasks, *dscratch, *dfe, *dti;
    int64_t *dcosts, *dmin;
    tr_repair_out *dout;
} tr_repair_dev;
static void tr_repair_dev_close(tr_repair_dev *e) {
    for (int i = 0; i < e->nmem; i++) tr_cuda_free(e->mem[i]);
    e->nmem = 0;
}
static int tr_repair_dev_put(tr_repair_dev *e, const void *src, size_t bytes, void **dst) {
    *dst = NULL;
    if (!src || !bytes) return 0;
    int rc = tr_cuda_alloc(dst, bytes);
    if (!rc) { e->mem[e->nmem++] = *dst; rc = tr_cuda_copy_in(*dst, src, bytes); }
    return rc;
}
static int tr_repair_dev_alloc(tr_repair_dev *e, size_t bytes, void **dst) {
    int rc = tr_cuda_alloc(dst, bytes ? bytes : 1);
    if (!rc) e->mem[e->nmem++] = *dst;
    return rc;
}
static int tr_repair_dev_open(const tr_repair_view *v, uint64_t cap, tr_repair_dev *e) {
    memset(e, 0, sizeof *e);
    e->d = *v; e->cap = cap ? cap : 1;
    int rc = 0;
    void *p;
    if (v->kind == TR_REPAIR_HIT) {
        rc = tr_repair_dev_put(e, v->masks, (size_t)(v->nmasks * 8), &p); e->d.masks = (const uint64_t *)p;
    } else if (v->kind == TR_REPAIR_CIRCUIT) {
        const uint64_t nw = v->dom.nw, k = (uint64_t)v->dom.k;
        if (v->dom.kind == TR_BOOL_ROWS) {
            rc = tr_repair_dev_put(e, v->dom.selection, (size_t)(nw * 8), &p); e->d.dom.selection = (const uint64_t *)p;
            if (!rc) { rc = tr_repair_dev_put(e, v->dom.value, (size_t)(k * nw * 8), &p); e->d.dom.value = (const uint64_t *)p; }
            if (!rc && v->dom.valid) { rc = tr_repair_dev_put(e, v->dom.valid, (size_t)(k * nw * 8), &p); e->d.dom.valid = (const uint64_t *)p; }
        }
        if (!rc) { rc = tr_repair_dev_put(e, v->prog.nodes, (size_t)v->prog.nnodes * sizeof(tr_bool_node), &p); e->d.prog.nodes = (const tr_bool_node *)p; }
        if (!rc) { rc = tr_repair_dev_put(e, v->prog.deps, (size_t)v->prog.nnodes * 8, &p); e->d.prog.deps = (const uint64_t *)p; }
        if (!rc) { rc = tr_repair_dev_put(e, v->prog.schedule, (size_t)v->prog.nsched * 4, &p); e->d.prog.schedule = (const uint32_t *)p; }
        if (!rc) { rc = tr_repair_dev_put(e, v->target, (size_t)(nw * 8), &p); e->d.target = (const uint64_t *)p; }
    } else if (v->kind == TR_REPAIR_GAME) {
        const tr_attr_game_view *g = &v->game;
        if (g->kind == TR_ATTR_GAME_TABLE) {
            rc = tr_repair_dev_put(e, g->table, (size_t)(((uint64_t)1 << g->n) * 16), &p); e->d.game.table = (const tr_math_i128 *)p;
        } else {
            const tr_prov_plan_view *pl = &g->plan;
            rc = tr_repair_dev_put(e, pl->node, (size_t)pl->nodes * sizeof(tr_prov_node), &p); e->d.game.plan.node = (const tr_prov_node *)p;
            if (!rc) { rc = tr_repair_dev_put(e, pl->child, (size_t)pl->nchild * 4, &p); e->d.game.plan.child = (const uint32_t *)p; }
            if (!rc) { rc = tr_repair_dev_put(e, pl->output, (size_t)pl->noutputs * 4, &p); e->d.game.plan.output = (const uint32_t *)p; }
            if (!rc) { rc = tr_repair_dev_put(e, g->weights, (size_t)pl->noutputs * 8, &p); e->d.game.weights = (const int64_t *)p; }
            if (!rc) { rc = tr_repair_dev_put(e, g->var_player, (size_t)g->nvars * 4, &p); e->d.game.var_player = (const uint32_t *)p; }
        }
    } else {
        rc = tr_repair_dev_put(e, v->qual, (size_t)v->n, &p); e->d.qual = (const uint8_t *)p;
        if (!rc) { rc = tr_repair_dev_put(e, v->vvalid, (size_t)v->n, &p); e->d.vvalid = (const uint8_t *)p; }
        if (!rc) { rc = tr_repair_dev_put(e, v->vval, (size_t)v->n * 8, &p); e->d.vval = (const int64_t *)p; }
    }
    e->row = tr_repair_scratch_words(v);
    const uint64_t rb = e->row * 8;
    e->chunk = rb >= TR_REPAIR_DEVICE_SCRATCH ? 1 : TR_REPAIR_DEVICE_SCRATCH / rb;
    if (e->chunk > e->cap) e->chunk = e->cap;
    const uint64_t words = (e->cap + 63) / 64;
    if (!rc) rc = tr_repair_dev_alloc(e, (size_t)(e->cap * 8), (void **)&e->dmasks);
    if (!rc) rc = tr_repair_dev_alloc(e, (size_t)(e->cap * 8), (void **)&e->dcosts);
    if (!rc) rc = tr_repair_dev_alloc(e, (size_t)(e->cap * sizeof(tr_repair_out)), (void **)&e->dout);
    if (!rc) rc = tr_repair_dev_alloc(e, (size_t)(words * 8), (void **)&e->dfe);
    if (!rc) rc = tr_repair_dev_alloc(e, (size_t)(words * 8), (void **)&e->dti);
    if (!rc) rc = tr_repair_dev_alloc(e, 16, (void **)&e->dmin);
    if (!rc) rc = tr_repair_dev_alloc(e, (size_t)(e->chunk * rb), (void **)&e->dscratch);
    return rc;
}
static int32_t tr_repair_dev_batch(void *ctx, const uint64_t *masks, uint64_t count, tr_repair_out *out) {
    tr_repair_dev *e = (tr_repair_dev *)ctx;
    if (count > e->cap) return -1;
    int rc = tr_cuda_copy_in(e->dmasks, masks, (size_t)count * 8);
    for (uint64_t first = 0; first < count && !rc; first += e->chunk) {
        const uint64_t c = count - first < e->chunk ? count - first : e->chunk;
        TR_LAUNCH(tr_repair_eval_kernel, c, e->d, e->dmasks, first, c, e->dscratch, e->row, e->dout);
        rc = tr_cuda_sync();
    }
    if (!rc) rc = tr_cuda_copy_out(out, e->dout, (size_t)count * sizeof(tr_repair_out));
    return rc;
}
static int32_t tr_repair_dev_scan(void *ctx, const uint64_t *masks, const int64_t *costs, uint64_t count, int64_t match,
                                  uint64_t *feasible, uint64_t *ties, int64_t *mincost) {
    tr_repair_dev *e = (tr_repair_dev *)ctx;
    if (count > e->cap) return -1;
    const uint64_t words = (count + 63) / 64;
    const int64_t init[2] = {INT64_MAX, 0};
    int rc = tr_cuda_copy_in(e->dmasks, masks, (size_t)count * 8);
    if (!rc) rc = tr_cuda_copy_in(e->dcosts, costs, (size_t)count * 8);
    if (!rc) rc = tr_cuda_zero(e->dfe, (size_t)(words ? words : 1) * 8);
    if (!rc) rc = tr_cuda_zero(e->dti, (size_t)(words ? words : 1) * 8);
    if (!rc) rc = tr_cuda_copy_in(e->dmin, init, 16);
    for (uint64_t first = 0; first < count && !rc; first += e->chunk) {
        const uint64_t c = count - first < e->chunk ? count - first : e->chunk;
        TR_LAUNCH(tr_repair_scan_kernel, c, e->d, e->dmasks, e->dcosts, first, c, e->dscratch, e->row, e->dfe, e->dmin);
        rc = tr_cuda_sync();
    }
    int64_t mn = INT64_MAX;
    if (!rc) rc = tr_cuda_copy_out(&mn, e->dmin, 8);
    if (!rc) {
        const int64_t m = match == INT64_MIN ? mn : match;
        rc = tr_cuda_copy_in(e->dmin + 1, &m, 8);
        if (!rc && count) { TR_LAUNCH(tr_repair_tie_kernel, count, e->dcosts, count, e->dfe, e->dmin + 1, e->dti); rc = tr_cuda_sync(); }
    }
    if (!rc) rc = tr_cuda_copy_out(feasible, e->dfe, (size_t)words * 8);
    if (!rc) rc = tr_cuda_copy_out(ties, e->dti, (size_t)words * 8);
    if (!rc) *mincost = mn;
    return rc;
}

#ifdef __cplusplus
extern "C" {
#endif
int tr_mgpu_repair_eval(const tr_repair_view *view, const uint64_t *masks, uint64_t count, tr_repair_out *out,
                        const tr_math_budget *budget, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    int32_t rc = tr_repair_view_check(view);
    if (rc) return rc;
    if (count && (!masks || !out)) return -1;
    if (view->kind == TR_REPAIR_CIRCUIT && view->prog.nnodes > TR_REPAIR_MAX_NODES) return -12;
    tr_math_budget dflt;
    if (!budget) { tr_budget_default(&dflt); budget = &dflt; }
    tr_math_need need;
    for (int i = 0; i < TR_BUDGET_RESOURCES; i++) need.need[i] = 0;
    need.need[TR_BUDGET_OUTPUT] = count;
    need.need[TR_BUDGET_BYTES] = tr_repair_view_device_bytes(view, count);
    if ((rc = tr_budget_check(budget, &need, refusal)) || !count) return rc;
    tr_repair_dev e;
    rc = tr_repair_dev_open(view, count, &e);
    if (!rc) rc = tr_repair_dev_batch(&e, masks, count, out);
    tr_repair_dev_close(&e);
    return rc;
}
int tr_mgpu_repair_scan(const tr_repair_view *view, const uint64_t *masks, const int64_t *costs, uint64_t count,
                        int64_t match, uint64_t *feasible, uint64_t *ties, int64_t *mincost,
                        const tr_math_budget *budget, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    int32_t rc = tr_repair_view_check(view);
    if (rc) return rc;
    if ((count && (!masks || !costs || !feasible || !ties)) || !mincost) return -1;
    if (view->kind == TR_REPAIR_CIRCUIT && view->prog.nnodes > TR_REPAIR_MAX_NODES) return -12;
    tr_math_budget dflt;
    if (!budget) { tr_budget_default(&dflt); budget = &dflt; }
    tr_math_need need;
    for (int i = 0; i < TR_BUDGET_RESOURCES; i++) need.need[i] = 0;
    need.need[TR_BUDGET_OUTPUT] = count;
    need.need[TR_BUDGET_BYTES] = tr_repair_view_device_bytes(view, count);
    if ((rc = tr_budget_check(budget, &need, refusal))) return rc;
    if (!count) { *mincost = INT64_MAX; return 0; }
    tr_repair_dev e;
    rc = tr_repair_dev_open(view, count, &e);
    if (!rc) rc = tr_repair_dev_scan(&e, masks, costs, count, match, feasible, ties, mincost);
    tr_repair_dev_close(&e);
    return rc;
}
int tr_mgpu_repair_search(const tr_repair_view *view, const tr_repair_plan *plan, uint64_t *solutions, int64_t *sol_cost,
                          tr_repair_info *info, const tr_math_budget *budget, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    int32_t rc = tr_repair_view_check(view);
    if (rc) return rc;
    if (!plan || !info || (plan->max_solutions && !solutions)) return -1;
    if (view->kind == TR_REPAIR_CIRCUIT && view->prog.nnodes > TR_REPAIR_MAX_NODES) return -12;
    tr_math_budget dflt;
    if (!budget) { tr_budget_default(&dflt); budget = &dflt; }
    if ((rc = tr_repair_plan_check(view, plan))) return rc;           /* the same checks as the CPU */
    tr_math_need need;
    tr_repair_search_need(plan, &need);
    need.need[TR_BUDGET_BYTES] = tr_budget_add(need.need[TR_BUDGET_BYTES], tr_repair_view_device_bytes(view, 2 * TR_REPAIR_BATCH));
    if ((rc = tr_budget_check(budget, &need, refusal))) return rc;
    tr_repair_dev e;
    rc = tr_repair_dev_open(view, 2 * TR_REPAIR_BATCH, &e);
    if (!rc) rc = tr_repair_run(view, plan, tr_repair_dev_batch, tr_repair_dev_scan, &e, TR_REPAIR_BATCH, solutions,
                                sol_cost, info);
    tr_repair_dev_close(&e);
    return rc;
}
#ifdef __cplusplus
}
#endif
