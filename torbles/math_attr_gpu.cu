/* ATTR-001 on a CUDA device: a declared game's value function on many coalitions at once (each work item one
 * coalition: a 0/1 valuation of the provenance DAG, or a table lookup), and the exact Shapley / Banzhaf numerators of
 * every coalition's weighted marginal-contribution terms accumulated with 128-bit atomics; with exactly the results of
 * math_attr.c (math_attr.h).
 * SPDX-License-Identifier: Apache-2.0
 *
 * values      one work item per coalition: tr_attr_eval (shared with the CPU) into its own scratch row and its own
 *             output slot (nothing depends on the thread order); launch chunks keep the scratch within
 *             TR_ATTR_DEVICE_SCRATCH bytes.
 * exhaustive  work items are aligned slices of TR_ATTR_SLICE coalition masks (the low 6 bits vary, the high players'
 *             membership is constant on a slice). Player i's Shapley numerator over n! is sum_{S contains i} (pos(s) +
 *             neg(s)) v(S) - sum_S neg(s) v(S) (pos(s) = (s-1)! (n-s)!, neg(s) = s! (n-1-s)!), its Banzhaf numerator over
 *             2^{n-1} is sum_{S contains i} 2 v(S) - sum_S v(S): a slice sums its terms locally (one counter per low
 *             player, one for the high players, one for the player-independent sums) and adds each nonzero counter
 *             once with tr_atomic_add_wide (exact modulo 2^128; exact because the host proved |numerator| < 2^127).
 * explore, search: the shared drivers of math_attr.h with a device batch evaluator (the values kernel).
 * The host checks the view, the budget (bytes = every device copy, the accumulators and the scratch chunk) and the
 * overflow bounds before allocating anything on the device. */
#include "math_cuda.h"
#include "math_attr.h"
#include "math_handles.h"

TR_MGPU_ABI(tr_attr_q, TR_ATTR_ABI);              /* structs passed to the CUDA library (math_handles.h) */
TR_MGPU_ABI(tr_attr_identity, TR_ATTR_ABI);
TR_MGPU_ABI(tr_attr_game_view, TR_ATTR_ABI);
TR_MGPU_ABI(tr_attr_explore_info, TR_ATTR_ABI);
TR_MGPU_ABI(tr_attr_search_query, TR_ATTR_ABI);
TR_MGPU_ABI(tr_attr_search_info, TR_ATTR_ABI);

TR_KERNEL void tr_attr_values_kernel(tr_attr_game_view g, const uint64_t *masks, uint64_t first, uint64_t count,
                                     unsigned __int128 *scratch, uint64_t row, tr_math_i128 *out, uint64_t *flag) {
    TR_GRID_LOOP(i, count) {
        __int128 v = 0;
        if (!tr_attr_eval(&g, masks[first + i], scratch + i * row, &v)) { tr_atomic_or_u64(flag, 1); v = 0; }
        out[first + i] = tr_i128_set(v);
    }
}

TR_KERNEL void tr_attr_exhaustive_kernel(tr_attr_game_view g, uint64_t first, uint64_t nslices, uint64_t total,
                                         const tr_math_i128 *pos, const tr_math_i128 *neg, unsigned __int128 *scratch,
                                         uint64_t row, int64_t *sh, int64_t *bz, int64_t *tot, uint64_t *flag) {
    TR_GRID_LOOP(i, nslices) {
        const uint64_t base = (first + i) * TR_ATTR_SLICE;
        const uint64_t end = total - base < TR_ATTR_SLICE ? total : base + TR_ATTR_SLICE;
        const int32_t n = g.n, low = n < 6 ? n : 6;
        unsigned __int128 xl[6], yl[6], xh = 0, yh = 0, tn = 0, bn = 0;
        for (int32_t b = 0; b < 6; b++) { xl[b] = 0; yl[b] = 0; }
        for (uint64_t S = base; S < end; S++) {
            __int128 v = 0;
            if (!tr_attr_eval(&g, S, scratch + i * row, &v)) { tr_atomic_or_u64(flag, 1); v = 0; }
            const int s = tr_attr_popc(S);
            const unsigned __int128 vv = (unsigned __int128)v;
            const unsigned __int128 ps = (unsigned __int128)tr_i128_get(pos[s]), ng = (unsigned __int128)tr_i128_get(neg[s]);
            const unsigned __int128 w = (ps + ng) * vv, wn = ng * vv, two = vv + vv;
            tn += wn; bn += vv;
            for (int32_t b = 0; b < low; b++) if ((S >> b) & 1) { xl[b] += w; yl[b] += two; }
            xh += w; yh += two;
        }
        for (int32_t b = 0; b < low; b++) {
            if (xl[b]) tr_atomic_add_wide(&sh[2 * b], &sh[2 * b + 1], (__int128)xl[b]);
            if (yl[b]) tr_atomic_add_wide(&bz[2 * b], &bz[2 * b + 1], (__int128)yl[b]);
        }
        for (int32_t p = 6; p < n; p++)
            if ((base >> p) & 1) {
                if (xh) tr_atomic_add_wide(&sh[2 * p], &sh[2 * p + 1], (__int128)xh);
                if (yh) tr_atomic_add_wide(&bz[2 * p], &bz[2 * p + 1], (__int128)yh);
            }
        if (tn) tr_atomic_add_wide(&tot[0], &tot[1], (__int128)tn);
        if (bn) tr_atomic_add_wide(&tot[2], &tot[3], (__int128)bn);
    }
}

/* the game's device copies (pointers of the view replaced) */
typedef struct {
    tr_attr_game_view d;
    void *mem[6];
} tr_attr_dev_game;
static void tr_attr_dev_free(tr_attr_dev_game *g) {
    for (int i = 0; i < 6; i++) { tr_cuda_free(g->mem[i]); g->mem[i] = NULL; }
}
static int tr_attr_dev_copy(const tr_attr_game_view *h, tr_attr_dev_game *g) {
    int rc = 0;
    g->d = *h;
    for (int i = 0; i < 6; i++) g->mem[i] = NULL;
    if (h->kind == TR_ATTR_GAME_TABLE) {
        const size_t bytes = (size_t)(((uint64_t)1 << h->n) * 16);
        rc = tr_cuda_alloc(&g->mem[0], bytes);
        if (!rc) rc = tr_cuda_copy_in(g->mem[0], h->table, bytes);
        g->d.table = (const tr_math_i128 *)g->mem[0];
        return rc;
    }
    const tr_prov_plan_view *p = &h->plan;
    rc = tr_cuda_alloc(&g->mem[0], (size_t)p->nodes * sizeof(tr_prov_node));
    if (!rc) rc = tr_cuda_alloc(&g->mem[1], (size_t)p->nchild * 4);
    if (!rc) rc = tr_cuda_alloc(&g->mem[2], (size_t)p->noutputs * 4);
    if (!rc) rc = tr_cuda_alloc(&g->mem[3], (size_t)p->noutputs * 8);
    if (!rc) rc = tr_cuda_alloc(&g->mem[4], (size_t)h->nvars * 4);
    if (!rc) rc = tr_cuda_copy_in(g->mem[0], p->node, (size_t)p->nodes * sizeof(tr_prov_node));
    if (!rc) rc = tr_cuda_copy_in(g->mem[1], p->child, (size_t)p->nchild * 4);
    if (!rc) rc = tr_cuda_copy_in(g->mem[2], p->output, (size_t)p->noutputs * 4);
    if (!rc) rc = tr_cuda_copy_in(g->mem[3], h->weights, (size_t)p->noutputs * 8);
    if (!rc) rc = tr_cuda_copy_in(g->mem[4], h->var_player, (size_t)h->nvars * 4);
    g->d.plan.node = (const tr_prov_node *)g->mem[0];
    g->d.plan.child = (const uint32_t *)g->mem[1];
    g->d.plan.output = (const uint32_t *)g->mem[2];
    g->d.weights = (const int64_t *)g->mem[3];
    g->d.var_player = (const uint32_t *)g->mem[4];
    return rc;
}

/* a device batch evaluator: the game's copies and buffers for TR_ATTR_DEVICE_BATCH coalitions */
typedef struct {
    tr_attr_dev_game game;
    uint64_t *dmasks, *dflag, chunk, row;
    tr_math_i128 *dvals;
    unsigned __int128 *dscratch;
} tr_attr_dev_eval;
static void tr_attr_dev_eval_free(tr_attr_dev_eval *e) {
    tr_attr_dev_free(&e->game);
    tr_cuda_free(e->dmasks); tr_cuda_free(e->dflag); tr_cuda_free(e->dvals); tr_cuda_free(e->dscratch);
    e->dmasks = NULL; e->dflag = NULL; e->dvals = NULL; e->dscratch = NULL;
}
static int tr_attr_dev_eval_open(const tr_attr_game_view *h, uint64_t batch, tr_attr_dev_eval *e) {
    e->dmasks = NULL; e->dflag = NULL; e->dvals = NULL; e->dscratch = NULL;
    e->row = h->kind == TR_ATTR_GAME_PROV ? (h->plan.nodes ? h->plan.nodes : 1) : 0;
    e->chunk = tr_attr_device_chunk(h, batch);
    int rc = tr_attr_dev_copy(h, &e->game);
    if (!rc) rc = tr_cuda_alloc((void **)&e->dmasks, (size_t)batch * 8);
    if (!rc) rc = tr_cuda_alloc((void **)&e->dvals, (size_t)batch * 16);
    if (!rc) rc = tr_cuda_alloc((void **)&e->dflag, 8);
    if (!rc) rc = tr_cuda_alloc((void **)&e->dscratch, (size_t)(e->chunk * (e->row ? e->row : 1) * 16));
    return rc;
}
static int32_t tr_attr_dev_batch(void *ctx, const uint64_t *masks, uint64_t count, tr_math_i128 *values) {
    tr_attr_dev_eval *e = (tr_attr_dev_eval *)ctx;
    uint64_t flag = 0;
    int rc = tr_cuda_copy_in(e->dmasks, masks, (size_t)count * 8);
    if (!rc) rc = tr_cuda_zero(e->dflag, 8);
    for (uint64_t first = 0; first < count && !rc; first += e->chunk) {
        uint64_t c = count - first < e->chunk ? count - first : e->chunk;
        TR_LAUNCH(tr_attr_values_kernel, c, e->game.d, e->dmasks, first, c, e->dscratch, e->row, e->dvals, e->dflag);
        rc = tr_cuda_sync();
    }
    if (!rc) rc = tr_cuda_copy_out(values, e->dvals, (size_t)count * 16);
    if (!rc) rc = tr_cuda_copy_out(&flag, e->dflag, 8);
    if (!rc && flag) rc = -3;                                         /* TR_MATH_OVERFLOW */
    return rc;
}

#ifdef __cplusplus
extern "C" {
#endif
int tr_mgpu_attr_values(const tr_attr_game_view *g, const uint64_t *masks, uint64_t count, tr_math_i128 *out,
                        const tr_math_budget *budget, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    int rc = tr_attr_view_check(g, 1);
    if (rc) return rc;
    if (count && (!masks || !out)) return -1;
    const uint64_t all = tr_attr_mask_all(g->n);
    for (uint64_t i = 0; i < count; i++) if (masks[i] & ~all) return -1;
    tr_math_need need;
    tr_attr_view_need(g, TR_ATTR_OP_VALUES, count, TR_ATTR_NEED_DEVICE, &need);
    rc = tr_attr_check(budget, &need, refusal);
    if (rc) return rc;
    if (!count) return 0;
    tr_attr_dev_eval e;
    rc = tr_attr_dev_eval_open(g, count, &e);
    if (!rc) rc = tr_attr_dev_batch(&e, masks, count, out);
    tr_attr_dev_eval_free(&e);
    return rc;
}

int tr_mgpu_attr_exhaustive(const tr_attr_game_view *g, tr_math_i128 *shapley_num, tr_math_i128 *banzhaf_num,
                            const tr_math_budget *budget, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    int rc = tr_attr_view_check(g, 1);
    if (rc) return rc;
    if (!shapley_num || !banzhaf_num) return -1;
    tr_math_need need;
    tr_attr_view_need(g, TR_ATTR_OP_EXHAUSTIVE, 0, TR_ATTR_NEED_DEVICE, &need);
    rc = tr_attr_check(budget, &need, refusal);
    if (rc) return rc;
    if (g->n > TR_ATTR_MAX_EXHAUSTIVE) return -12;                   /* TR_MATH_UNSUPPORTED */
    rc = tr_attr_exhaustive_bound(g);
    if (rc) return rc;
    const int32_t n = g->n;
    __int128 pos[TR_ATTR_MAX_EXHAUSTIVE + 1], neg[TR_ATTR_MAX_EXHAUSTIVE + 1], fact;
    tr_attr_shapley_weights(n, pos, neg, &fact);
    tr_math_i128 hp[TR_ATTR_MAX_EXHAUSTIVE + 1], hn[TR_ATTR_MAX_EXHAUSTIVE + 1];
    for (int32_t s = 0; s <= n; s++) { hp[s] = tr_i128_set(pos[s]); hn[s] = tr_i128_set(neg[s]); }
    const uint64_t total = (uint64_t)1 << n, slices = (total + TR_ATTR_SLICE - 1) / TR_ATTR_SLICE;
    const uint64_t row = g->kind == TR_ATTR_GAME_PROV ? (g->plan.nodes ? g->plan.nodes : 1) : 0;
    const uint64_t chunk = tr_attr_device_chunk(g, slices);
    tr_attr_dev_game dg;
    tr_math_i128 *dpos = NULL, *dneg = NULL;
    int64_t *dsh = NULL, *dbz = NULL, *dtot = NULL;
    uint64_t *dflag = NULL, flag = 0;
    unsigned __int128 *dscratch = NULL;
    rc = tr_attr_dev_copy(g, &dg);
    if (!rc) rc = tr_cuda_alloc((void **)&dpos, (size_t)(n + 1) * 16);
    if (!rc) rc = tr_cuda_alloc((void **)&dneg, (size_t)(n + 1) * 16);
    if (!rc) rc = tr_cuda_alloc((void **)&dsh, (size_t)n * 16);
    if (!rc) rc = tr_cuda_alloc((void **)&dbz, (size_t)n * 16);
    if (!rc) rc = tr_cuda_alloc((void **)&dtot, 32);
    if (!rc) rc = tr_cuda_alloc((void **)&dflag, 8);
    if (!rc) rc = tr_cuda_alloc((void **)&dscratch, (size_t)(chunk * (row ? row : 1) * 16));
    if (!rc) rc = tr_cuda_copy_in(dpos, hp, (size_t)(n + 1) * 16);
    if (!rc) rc = tr_cuda_copy_in(dneg, hn, (size_t)(n + 1) * 16);
    for (uint64_t first = 0; first < slices && !rc; first += chunk) {
        uint64_t c = slices - first < chunk ? slices - first : chunk;
        TR_LAUNCH(tr_attr_exhaustive_kernel, c, dg.d, first, c, total, dpos, dneg, dscratch, row, dsh, dbz, dtot, dflag);
        rc = tr_cuda_sync();
    }
    tr_math_i128 hs[TR_ATTR_MAX_EXHAUSTIVE], hb[TR_ATTR_MAX_EXHAUSTIVE], ht[2];
    memset(hs, 0, sizeof hs); memset(hb, 0, sizeof hb); memset(ht, 0, sizeof ht);
    if (!rc) rc = tr_cuda_copy_out(hs, dsh, (size_t)n * 16);
    if (!rc) rc = tr_cuda_copy_out(hb, dbz, (size_t)n * 16);
    if (!rc) rc = tr_cuda_copy_out(ht, dtot, 32);
    if (!rc) rc = tr_cuda_copy_out(&flag, dflag, 8);
    tr_attr_dev_free(&dg);
    tr_cuda_free(dpos); tr_cuda_free(dneg); tr_cuda_free(dsh); tr_cuda_free(dbz); tr_cuda_free(dtot);
    tr_cuda_free(dflag); tr_cuda_free(dscratch);
    if (rc) return rc;
    if (flag) return -3;
    const unsigned __int128 tsh = (unsigned __int128)tr_i128_get(ht[0]), tbz = (unsigned __int128)tr_i128_get(ht[1]);
    for (int32_t i = 0; i < n; i++) {
        shapley_num[i] = tr_i128_set((__int128)((unsigned __int128)tr_i128_get(hs[i]) - tsh));
        banzhaf_num[i] = tr_i128_set((__int128)((unsigned __int128)tr_i128_get(hb[i]) - tbz));
    }
    return 0;
}

int tr_mgpu_attr_explore(const tr_attr_game_view *g, uint64_t max_coalitions, tr_attr_q *shapley_lo,
                         tr_attr_q *shapley_hi, tr_attr_q *banzhaf_lo, tr_attr_q *banzhaf_hi, tr_attr_explore_info *info,
                         const tr_math_budget *budget, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    int rc = tr_attr_view_check(g, 1);
    if (rc) return rc;
    tr_math_need need;
    tr_attr_view_need(g, TR_ATTR_OP_EXPLORE, max_coalitions, TR_ATTR_NEED_DEVICE, &need);
    rc = tr_attr_check(budget, &need, refusal);
    if (rc) return rc;
    tr_attr_dev_eval e;
    rc = tr_attr_dev_eval_open(g, TR_ATTR_DEVICE_BATCH, &e);
    if (!rc) rc = tr_attr_explore_run(g, max_coalitions, TR_ATTR_DEVICE_BATCH, tr_attr_dev_batch, &e, shapley_lo,
                                      shapley_hi, banzhaf_lo, banzhaf_hi, info, budget, refusal);
    tr_attr_dev_eval_free(&e);
    return rc;
}

int tr_mgpu_attr_search(const tr_attr_game_view *g, const tr_attr_search_query *q, uint64_t *solutions,
                        tr_attr_search_info *info, const tr_math_budget *budget, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    int rc = tr_attr_view_check(g, 1);
    if (rc) return rc;
    if (!q) return -1;
    tr_math_need need;
    tr_attr_view_need(g, TR_ATTR_OP_SEARCH, q->max_coalitions, TR_ATTR_NEED_DEVICE, &need);
    rc = tr_attr_check(budget, &need, refusal);
    if (rc) return rc;
    tr_attr_dev_eval e;
    rc = tr_attr_dev_eval_open(g, TR_ATTR_DEVICE_BATCH, &e);
    if (!rc) rc = tr_attr_search_run(g, q, TR_ATTR_DEVICE_BATCH, tr_attr_dev_batch, &e, solutions, info, budget, refusal);
    tr_attr_dev_eval_free(&e);
    return rc;
}
#ifdef __cplusplus
}
#endif
