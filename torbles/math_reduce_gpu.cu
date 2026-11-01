/* FND-006 on a CUDA device: the fused multi-output pass, with exactly the states of the CPU fused pass
 * (math_reduce.h, math_reduce.c).
 * SPDX-License-Identifier: Apache-2.0
 *
 * One kernel. A work item is a contiguous slice of 64-row words (at least TR_REDUCE_MIN_WORDS words each) and produces
 * every output's contribution of its rows: per word it evaluates the conditions the plan reads and the scheduled
 * predicate nodes once (tr_reduce_eval_word, shared with the CPU loop), forms each output's mask, and per selected row
 * of their union computes the column presence and each distinct product at most once (a stamp per product in the
 * item's scratch) for every output term that uses it. Bank outputs accumulate locally (the rows, and each term's count
 * and exact 128-bit sum) while the output's group id stays the same and flush with integer atomics when it changes
 * and at the item's end (tr_atomic_add_u64; tr_atomic_add_wide for the 128-bit addends); histogram cells add with one
 * atomic per row and cell (the weight sum with the two-word atomic). Integer atomics commute, so the arenas do not
 * depend on thread order; every product and partial sum is exact because the host proved every term's bound
 * (tr_bank_prove at compile time) before launching. Each item owns its scratch slot (no sharing between threads, no
 * barrier). The host copies the view with every pointer swapped for a device copy: the selection, only the columns
 * the plan reads and their validity, each distinct group-id array once, the maps, the compiled tables; the arenas
 * come back into the stage the host adopts (tr_reduce_stage_finish). */
#include "math_cuda.h"
#include "math_reduce.h"
#include "math_handles.h"

TR_MGPU_ABI(tr_reduce_view, 1);         /* structs passed to the CUDA library (math_handles.h) */
TR_MGPU_ABI(tr_reduce_group_view, 1);
TR_MGPU_ABI(tr_reduce_arena, 1);
TR_MGPU_ABI(tr_reduce_stats, 1);

TR_DEVICE void tr_reduce_flush(const tr_reduce_view *v, int32_t o, uint64_t *racc, uint64_t *rcur, uint64_t *tcnt,
                               uint64_t *tsum, uint64_t *A, int64_t *B) {
    const tr_reduce_out *out = &v->outs[o];
    uint64_t g = rcur[o];
    if (g == UINT64_MAX || !racc[o]) return;
    tr_atomic_add_u64(&A[out->rows_at + g], racc[o]);
    racc[o] = 0;
    if (out->kind != TR_REDUCE_BANK) return;
    const uint64_t m = (uint64_t)out->nterms;
    for (uint64_t j = 0; j < m; j++) {
        uint64_t t = (uint64_t)out->first + j, k = g * m + j;
        if (tcnt[t]) { tr_atomic_add_u64(&A[out->count_at + k], tcnt[t]); tcnt[t] = 0; }
        __int128 s = (__int128)(((unsigned __int128)tsum[2 * t + 1] << 64) | tsum[2 * t]);
        if (s) {
            tr_atomic_add_wide(&B[2 * (out->sum_at + k)], &B[2 * (out->sum_at + k) + 1], s);
            tsum[2 * t] = 0; tsum[2 * t + 1] = 0;
        }
    }
}

TR_KERNEL void tr_reduce_fused_kernel(const tr_reduce_view *v, uint64_t items, uint64_t wpi, uint64_t *scratch,
                                      uint64_t per, uint64_t *A, int64_t *B) {
    TR_GRID_LOOP(i, items) {
        const uint64_t nw = (v->scope.rows + 63) / 64, O = (uint64_t)v->noutputs;
        uint64_t *cw = scratch + i * per;
        uint64_t *nwd = cw + 2 * (uint64_t)v->nconds;
        uint64_t *masks = nwd + 2 * (uint64_t)v->nnodes;
        uint64_t *live = masks + O;
        uint64_t *pval = live + O;
        uint64_t *pstamp = pval + 2 * (uint64_t)v->nproducts;
        uint64_t *racc = pstamp + (uint64_t)v->nproducts;
        uint64_t *rcur = racc + O;
        uint64_t *tcnt = rcur + O;
        uint64_t *tsum = tcnt + (uint64_t)v->nterms;
        for (uint64_t o = 0; o < O; o++) { racc[o] = 0; rcur[o] = UINT64_MAX; }
        for (int32_t j = 0; j < v->nproducts; j++) pstamp[j] = 0;
        for (int32_t t = 0; t < v->nterms; t++) { tcnt[t] = 0; tsum[2 * (uint64_t)t] = 0; tsum[2 * (uint64_t)t + 1] = 0; }
        uint64_t w0 = i * wpi, w1 = w0 + wpi < nw ? w0 + wpi : nw;
        for (uint64_t w = w0; w < w1; w++) {
            uint64_t sel = v->scope.selection[w];
            if (!sel) continue;
            tr_reduce_eval_word(v, w, cw, nwd);
            uint64_t nlive = 0, any = 0;
            for (uint64_t o = 0; o < O; o++) {
                uint64_t m = tr_reduce_mask(v, &v->outs[o], sel, nwd);
                if (m) { live[nlive] = o; masks[nlive] = m; nlive++; any |= m; }
            }
            while (any) {
                int b = tr_bool_ctz(any);
                any &= any - 1;
                uint64_t r = (w << 6) | (uint64_t)b, stamp = r + 1;
                uint32_t pm = tr_reduce_presence(v, r);
                for (uint64_t x = 0; x < nlive; x++) {
                    if (!((masks[x] >> b) & 1)) continue;
                    int32_t o = (int32_t)live[x];
                    const tr_reduce_out *out = &v->outs[o];
                    uint64_t g = tr_reduce_gid(v, out->grouping, r);
                    if (rcur[o] != g) { tr_reduce_flush(v, o, racc, rcur, tcnt, tsum, A, B); rcur[o] = g; }
                    racc[o]++;
                    if (out->kind == TR_REDUCE_BANK) {
                        for (int32_t j = 0; j < out->nterms; j++) {
                            uint64_t t = (uint64_t)out->first + (uint64_t)j;
                            const tr_reduce_term *tm = &v->terms[t];
                            if (!tr_reduce_term_valid(out, tm, pm)) continue;
                            uint64_t q = tm->product;
                            __int128 val;
                            if (pstamp[q] != stamp) {
                                val = tr_bank_term_value(&v->scope, &v->products[q], r);
                                pval[2 * q] = (uint64_t)val;
                                pval[2 * q + 1] = (uint64_t)((unsigned __int128)val >> 64);
                                pstamp[q] = stamp;
                            } else {
                                val = (__int128)(((unsigned __int128)pval[2 * q + 1] << 64) | pval[2 * q]);
                            }
                            tcnt[t]++;
                            __int128 s = (__int128)(((unsigned __int128)tsum[2 * t + 1] << 64) | tsum[2 * t]) + val;
                            tsum[2 * t] = (uint64_t)s;
                            tsum[2 * t + 1] = (uint64_t)((unsigned __int128)s >> 64);
                        }
                    } else {
                        const tr_dist_plan *hp = &v->hplans[out->hist];
                        uint64_t k = g * out->cells + tr_dist_row_cell(&v->scope, hp, r);
                        tr_atomic_add_u64(&A[out->count_at + k], 1);
                        if (hp->weighted && ((pm >> hp->weight) & 1)) {
                            tr_atomic_add_u64(&A[out->wcount_at + k], 1);
                            __int128 wv = tr_scope_value(&v->scope, hp->weight, r);
                            if (wv) tr_atomic_add_wide(&B[2 * (out->sum_at + k)], &B[2 * (out->sum_at + k) + 1], wv);
                        }
                    }
                }
            }
        }
        for (uint64_t o = 0; o < O; o++) tr_reduce_flush(v, (int32_t)o, racc, rcur, tcnt, tsum, A, B);
    }
}

/* the fused pass's states kept on the device (the arenas), for a later kernel to read (META-002's finalization) */
struct tr_mgpu_reduce_states_s {
    uint32_t magic;
    uint64_t *dA;
    int64_t *dB;
    uint64_t n64, n128;
};
#define TR_MGPU_REDUCE_MAGIC 0x53554752u

#ifdef __cplusplus
extern "C" {
#endif
static int tr_reduce_view_ok(const tr_reduce_view *view) {
    return view && view->scope.ncols >= 0 && view->scope.ncols <= TR_SCOPE_MAX_COLUMNS && view->ngroupings >= 1 &&
           view->ngroupings <= TR_REDUCE_MAX_GROUPINGS && view->noutputs >= 0 && view->noutputs <= TR_REDUCE_MAX_OUTPUTS &&
           view->nconds >= 0 && view->nnodes >= 0 && view->nsched >= 0 && view->nsched <= view->nnodes &&
           view->ncondlist >= 0 && view->ncondlist <= view->nconds && view->nproducts >= 0 && view->nterms >= 0 &&
           view->nhist >= 0 && (!view->scope.rows || view->scope.selection) && (!view->noutputs || view->outs) &&
           (!view->nconds || view->conds) && (!view->nnodes || (view->nodes && view->schedule)) &&
           (!view->ncondlist || view->condlist) && (!view->nproducts || view->products) && (!view->nterms || view->terms) &&
           (!view->nhist || view->hplans) && view->arena64 != TR_BUDGET_UNKNOWN && view->arena128 != TR_BUDGET_UNKNOWN;
}
void tr_mgpu_reduce_states_free(tr_mgpu_reduce_states *s) {
    if (!s || s->magic != TR_MGPU_REDUCE_MAGIC) return;
    tr_cuda_free(s->dA); tr_cuda_free(s->dB);
    s->magic = 0;
    free(s);
}
tr_mgpu_reduce_states *tr_mgpu_reduce_fused_device(const tr_reduce_view *view, const tr_math_need *base,
                                                   const tr_math_budget *budget, tr_reduce_stats *stats,
                                                   tr_math_refusal *refusal, int *error) {
    tr_budget_none(refusal);
    if (error) *error = 0;
    if (!tr_reduce_view_ok(view) || !base) { if (error) *error = -1; return NULL; }       /* TR_MATH_INVALID */
    tr_math_need need;
    tr_math_budget dflt;
    if (!budget) { tr_budget_default(&dflt); budget = &dflt; }
    tr_reduce_device_need(view, base, &need);
    int rc = tr_budget_check(budget, &need, refusal);
    if (rc) { if (error) *error = rc; return NULL; }
    tr_mgpu_reduce_states *out = (tr_mgpu_reduce_states *)calloc(1, sizeof *out);
    if (!out) { if (error) *error = -2; return NULL; }                                     /* TR_MATH_MEMORY */
    out->magic = TR_MGPU_REDUCE_MAGIC; out->n64 = view->arena64; out->n128 = view->arena128;
    const uint64_t n = view->scope.rows, nw = (n + 63) / 64;
    tr_reduce_stats st;
    memset(&st, 0, sizeof st);
    st.scans = 1; st.native_calls = 1; st.device = TR_REDUCE_ON_CUDA; st.plan = TR_REDUCE_FUSED;
    void *p = NULL;
    rc = tr_cuda_alloc(&p, (size_t)out->n64 * 8);
    out->dA = (uint64_t *)p;
    if (!rc) { rc = tr_cuda_alloc(&p, (size_t)out->n128 * 16); out->dB = (int64_t *)p; }
    if (!rc && (!view->scope.selected || !view->noutputs)) {   /* nothing to accumulate: the zeroed arenas */
        st.scans = 0;
        st.workspace_bytes = out->n64 * 8 + out->n128 * 16;
        if (stats) *stats = st;
        return out;
    }
    tr_reduce_view d = *view;
    void *dmem[3 * TR_SCOPE_MAX_COLUMNS + 2 * TR_REDUCE_MAX_GROUPINGS + 16];
    int nmem = 0;
    uint64_t moved = 0;
#define TR_REDUCE_DEV(ptr, src, bytes) do {                                                                   \
        if (!rc) {                                                                                             \
            void *p_ = NULL;                                                                                   \
            rc = tr_cuda_alloc(&p_, (size_t)(bytes));                                                          \
            if (!rc) { dmem[nmem++] = p_; if (src) { rc = tr_cuda_copy_in(p_, (src), (size_t)(bytes)); moved += (bytes); } } \
            (ptr) = p_;                                                                                        \
        }                                                                                                      \
    } while (0)
    TR_REDUCE_DEV(p, view->scope.selection, nw * 8);
    d.scope.selection = (const uint64_t *)p;
    d.scope.groups = NULL;
    for (int32_t c = 0; c < view->scope.ncols; c++) {
        tr_scope_view_column *k = &d.scope.col[c];
        if (!((view->used >> c) & 1)) { k->codes = NULL; k->values = NULL; k->valid = NULL; continue; }   /* never read */
        if (view->scope.col[c].values) { TR_REDUCE_DEV(p, view->scope.col[c].values, n * 8); k->values = (const int64_t *)p; }
        else { TR_REDUCE_DEV(p, view->scope.col[c].codes, n * 4); k->codes = (const uint32_t *)p; }
        if (view->scope.col[c].valid) { TR_REDUCE_DEV(p, view->scope.col[c].valid, n); k->valid = (const uint8_t *)p; }
    }
    for (int32_t g = 0; g < view->ngroupings; g++) {
        const uint32_t *ids = view->groups[g].ids;
        d.groups[g].ids = NULL;
        if (ids) {
            for (int32_t h = 0; h < g; h++) if (view->groups[h].ids == ids) d.groups[g].ids = d.groups[h].ids;
            if (!d.groups[g].ids) { TR_REDUCE_DEV(p, ids, n * 4); d.groups[g].ids = (const uint32_t *)p; }
        }
        if (view->groups[g].map) {
            TR_REDUCE_DEV(p, view->groups[g].map, view->groups[g].source_groups * 4);
            d.groups[g].map = (const uint32_t *)p;
        }
    }
    TR_REDUCE_DEV(p, view->conds, (uint64_t)view->nconds * sizeof(tr_bool_condition)); d.conds = (const tr_bool_condition *)p;
    TR_REDUCE_DEV(p, view->nodes, (uint64_t)view->nnodes * sizeof(tr_bool_node)); d.nodes = (const tr_bool_node *)p;
    TR_REDUCE_DEV(p, view->schedule, (uint64_t)view->nsched * 4); d.schedule = (const uint32_t *)p;
    TR_REDUCE_DEV(p, view->condlist, (uint64_t)view->ncondlist * 4); d.condlist = (const uint32_t *)p;
    TR_REDUCE_DEV(p, view->products, (uint64_t)view->nproducts * sizeof(tr_bank_term)); d.products = (const tr_bank_term *)p;
    TR_REDUCE_DEV(p, view->terms, (uint64_t)view->nterms * sizeof(tr_reduce_term)); d.terms = (const tr_reduce_term *)p;
    TR_REDUCE_DEV(p, view->outs, (uint64_t)view->noutputs * sizeof(tr_reduce_out)); d.outs = (const tr_reduce_out *)p;
    TR_REDUCE_DEV(p, view->hplans, (uint64_t)view->nhist * sizeof(tr_dist_plan)); d.hplans = (const tr_dist_plan *)p;
    const uint64_t items = tr_reduce_items(view), wpi = (nw + items - 1) / items;
    const uint64_t per = 2 * (uint64_t)view->nconds + 2 * (uint64_t)view->nnodes + 4 * (uint64_t)view->noutputs +
                         3 * (uint64_t)view->nproducts + 3 * (uint64_t)view->nterms + 8;
    uint64_t *dS = NULL;
    TR_REDUCE_DEV(p, (const void *)NULL, items * per * 8); dS = (uint64_t *)p;
    tr_reduce_view *dv = NULL;
    TR_REDUCE_DEV(p, &d, sizeof d); dv = (tr_reduce_view *)p;
#undef TR_REDUCE_DEV
    if (!rc) {
        TR_LAUNCH(tr_reduce_fused_kernel, items, dv, items, wpi, dS, per, out->dA, out->dB);
        rc = tr_cuda_sync();
    }
    for (int i = 0; i < nmem; i++) tr_cuda_free(dmem[i]);       /* only the arenas stay on the device */
    if (rc) { tr_mgpu_reduce_states_free(out); if (error) *error = rc; return NULL; }
    st.kernels = 1; st.items = items;
    st.workspace_bytes = need.need[TR_BUDGET_BYTES];
    st.transfer_bytes = moved;
    st.decoded_bytes = moved;
    if (stats) *stats = st;
    return out;
}
/* the device arenas (device pointers; valid while the states live) */
int tr_mgpu_reduce_states_arena(const tr_mgpu_reduce_states *s, tr_reduce_arena *out) {
    if (!s || s->magic != TR_MGPU_REDUCE_MAGIC || !out) return -1;
    out->a64 = s->dA; out->a128 = (tr_math_i128 *)s->dB; out->n64 = s->n64; out->n128 = s->n128;
    return 0;
}
/* the arenas copied into host arrays of the same sizes (a stage's arena) */
int tr_mgpu_reduce_states_copy(const tr_mgpu_reduce_states *s, tr_reduce_arena *host) {
    if (!s || s->magic != TR_MGPU_REDUCE_MAGIC || !host || !host->a64 || !host->a128 || host->n64 != s->n64 ||
        host->n128 != s->n128) return -1;
    int rc = tr_cuda_copy_out(host->a64, s->dA, (size_t)s->n64 * 8);
    if (!rc) rc = tr_cuda_copy_out(host->a128, s->dB, (size_t)s->n128 * 16);
    return rc;
}
int tr_mgpu_reduce_fused(const tr_reduce_view *view, const tr_math_need *base, const tr_math_budget *budget,
                         tr_reduce_arena *out, tr_reduce_stats *stats, tr_math_refusal *refusal) {
    if (!view || !out || out->n64 != view->arena64 || out->n128 != view->arena128) return -1;   /* TR_MATH_INVALID */
    int rc = 0;
    tr_reduce_stats st;
    memset(&st, 0, sizeof st);
    tr_mgpu_reduce_states *s = tr_mgpu_reduce_fused_device(view, base, budget, &st, refusal, &rc);
    if (!s) return rc ? rc : -2;
    rc = tr_mgpu_reduce_states_copy(s, out);
    tr_mgpu_reduce_states_free(s);
    st.transfer_bytes += out->n64 * 8 + out->n128 * 16;
    if (stats) *stats = st;
    return rc;
}
#ifdef __cplusplus
}
#endif
