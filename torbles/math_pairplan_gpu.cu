/* PAIR-002 on a CUDA device: the support-aggregation and direct-pair strategies over a SUPPORT or ROWS table uploaded
 * once, with exactly the results of tr_pairplan_eval_table (math_pairplan.h).
 * SPDX-License-Identifier: Apache-2.0
 *
 * A batch runs in two launches without barriers. The tile launch has one work item per (request, tile of
 * TR_PAIRPLAN_TILE x TR_PAIRPLAN_TILE entry pairs): the shared tr_pairplan_tile sums the tile's pair terms
 * mult(u, v) k(u, v) exactly in a local 1024-bit word and adds it to the request's accumulator with a sixteen-word
 * carry chain of 64-bit atomic adds (a local equivalent of math_kernel_gpu.cu's: each word adds its own word plus the
 * carry of the word below, so the words hold the exact sum modulo 2^1024, which is the exact sum because the host
 * proved |S| < 2^1019 before launching: tr_pairplan_check), its visits with an atomic add and an overflow flag with an
 * atomic or. The finish launch has one work item per request running the shared tr_pairplan_finish (the pair count P
 * from the groups' totals, the MEAN normalization), one output slot each. Integer sums do not depend on the order of
 * the threads, so the results equal the CPU's bit for bit. The host checks every request (validity, the magnitude
 * proof) and the batch's budget (output, iterations = the visits, bytes = the device copies) before copying. */
#include "math_cuda.h"
#include "math_pairplan.h"
#include "math_handles.h"

TR_MGPU_ABI(tr_pairplan_request, TR_PAIRPLAN_ABI);       /* structs passed to the CUDA library (math_handles.h) */
TR_MGPU_ABI(tr_pairplan_result, TR_PAIRPLAN_ABI);
TR_MGPU_ABI(tr_pairplan_totals, TR_PAIRPLAN_ABI);
TR_MGPU_ABI(tr_pairplan_table_view, TR_PAIRPLAN_ABI);
TR_MGPU_ABI(tr_pairplan_kernel_view, TR_PAIRPLAN_ABI);

/* dst (16 words) += x modulo 2^1024 */
TR_DEVICE void tr_pairplan_atomic_add_words(uint64_t *dst, const tr_kernel_wide *x) {
    uint64_t carry = 0;
    for (int i = 0; i < TR_KERNEL_WORDS; i++) {
        uint64_t v = x->w[i] + carry, c1 = v < carry;                   /* x + carry wrapped: only MAX + 1 */
        if (v) { uint64_t old = tr_atomic_add_u64(&dst[i], v); carry = c1 | ((old + v) < old); }
        else carry = c1;
    }
}
/* the request of tile work item i: the last r with offset[r] <= i (offset[0] = 0, n >= 1) */
TR_DEVICE uint64_t tr_pairplan_owner(const uint64_t *offset, uint64_t n, uint64_t i) {
    uint64_t lo = 0, hi = n - 1;
    while (lo < hi) {
        uint64_t mid = lo + (hi - lo + 1) / 2;
        if (offset[mid] <= i) lo = mid; else hi = mid - 1;
    }
    return lo;
}

TR_KERNEL void tr_pairplan_tile_kernel(tr_pairplan_table_view t, const tr_pairplan_kernel_view *k,
                                       const tr_pairplan_request *req, const uint64_t *offset, uint64_t n,
                                       uint64_t tiles, uint64_t *acc, uint64_t *visits, uint64_t *overflow) {
    TR_GRID_LOOP(i, tiles) {
        uint64_t r = tr_pairplan_owner(offset, n, i), v = 0;
        tr_kernel_wide local;
        if (!tr_pairplan_tile(&t, k, &req[r], i - offset[r], &local, &v)) tr_atomic_or_u64(&overflow[r], 1);
        else if (!tr_kernel_w_is_zero(&local)) tr_pairplan_atomic_add_words(acc + r * TR_KERNEL_WORDS, &local);
        if (v) tr_atomic_add_u64(&visits[r], v);
    }
}

TR_KERNEL void tr_pairplan_finish_kernel(tr_pairplan_table_view t, const tr_pairplan_kernel_view *k,
                                         const tr_pairplan_request *req, const int32_t *status, const uint64_t *acc,
                                         const uint64_t *visits, const uint64_t *overflow, uint64_t n,
                                         tr_pairplan_result *out) {
    TR_GRID_LOOP(r, n) {
        tr_kernel_wide s;
        for (int w = 0; w < TR_KERNEL_WORDS; w++) s.w[w] = acc[r * TR_KERNEL_WORDS + w];
        tr_pairplan_finish(&t, k, &req[r], status[r], &s, overflow[r] != 0, visits[r], &out[r]);
    }
}

struct tr_mgpu_pairplan_s {
    tr_pairplan_table_view view;           /* device pointers */
    uint64_t *host_offset;                 /* host copies for the checks and the tiles */
    tr_pairplan_totals *host_totals;
    void *mem[8];
    int nmem;
    uint64_t bytes;
};

static int tr_pairplan_view_ok(const tr_pairplan_table_view *v) {
    if (!v || (v->form != TR_PAIRPLAN_FORM_SUPPORT && v->form != TR_PAIRPLAN_FORM_ROWS) ||
        (v->policy != TR_PAIRPLAN_LISTWISE && v->policy != TR_PAIRPLAN_CATEGORY) || v->ncols < 1 ||
        v->ncols > TR_PAIRPLAN_MAX_COLUMNS || v->ngroups < 1 || !v->offset || !v->totals ||
        v->offset[v->ngroups] != v->nentries) return 0;
    if (v->nentries && (!v->value || !v->W)) return 0;
    if (v->nentries && v->form == TR_PAIRPLAN_FORM_SUPPORT && (!v->Q || !v->count)) return 0;
    if (v->nentries && v->policy == TR_PAIRPLAN_CATEGORY && !v->nulls) return 0;
    return 1;
}
static void tr_pairplan_device_free(tr_mgpu_pairplan *d) {
    if (!d) return;
    for (int i = 0; i < d->nmem; i++) tr_cuda_free(d->mem[i]);
    free(d->host_offset); free(d->host_totals);
    free(d);
}
static void tr_pairplan_refuse(tr_math_refusal *r, int32_t resource, uint64_t need, uint64_t limit) {
    if (r) { r->resource = resource; r->refused = 1; r->need = need; r->limit = limit; }
}

#ifdef __cplusplus
extern "C" {
#endif
tr_mgpu_pairplan *tr_mgpu_pairplan_upload(const tr_pairplan_table_view *v, const tr_math_budget *budget,
                                          tr_math_refusal *refusal, int *error) {
    tr_budget_none(refusal);
    if (error) *error = 0;
    if (!tr_pairplan_view_ok(v)) { if (error) *error = -1; return NULL; }        /* TR_MATH_INVALID */
    const uint64_t G = v->ngroups, N = v->nentries;
    tr_math_budget dflt;
    if (!budget) { tr_budget_default(&dflt); budget = &dflt; }
    uint64_t need = tr_pairplan_upload_bytes(v->form, v->policy, v->ncols, G, N);
    if (need == TR_BUDGET_UNKNOWN || need > budget->limit[TR_BUDGET_BYTES]) {
        tr_pairplan_refuse(refusal, TR_BUDGET_BYTES, need, budget->limit[TR_BUDGET_BYTES]);
        if (error) *error = TR_MATH_BUDGET_CODE;
        return NULL;
    }
    tr_mgpu_pairplan *d = (tr_mgpu_pairplan *)calloc(1, sizeof *d);
    if (!d) { if (error) *error = -2; return NULL; }                          /* TR_MATH_MEMORY */
    d->view = *v;
    d->bytes = need;
    d->host_offset = (uint64_t *)malloc((size_t)(G + 1) * 8);
    d->host_totals = (tr_pairplan_totals *)malloc((size_t)G * sizeof(tr_pairplan_totals));
    if (!d->host_offset || !d->host_totals) { tr_pairplan_device_free(d); if (error) *error = -2; return NULL; }
    memcpy(d->host_offset, v->offset, (size_t)(G + 1) * 8);
    memcpy(d->host_totals, v->totals, (size_t)G * sizeof(tr_pairplan_totals));
    int rc = 0;
    const void *src[7] = {v->offset, v->value, v->nulls, v->W, v->Q, v->count, v->totals};
    const uint64_t size[7] = {(G + 1) * 8, N * 8 * (uint64_t)v->ncols, v->nulls ? N : 0, N * 16,
                              v->Q ? N * sizeof(tr_dist_i256) : 0, v->count ? N * 8 : 0, G * sizeof(tr_pairplan_totals)};
    void *dst[7] = {0, 0, 0, 0, 0, 0, 0};
    for (int i = 0; i < 7 && !rc; i++) {
        if (!src[i]) continue;
        rc = tr_cuda_alloc(&dst[i], (size_t)size[i]);
        if (!rc) { d->mem[d->nmem++] = dst[i]; rc = tr_cuda_copy_in(dst[i], src[i], (size_t)size[i]); }
    }
    if (rc) { tr_pairplan_device_free(d); if (error) *error = rc; return NULL; }
    d->view.offset = (const uint64_t *)dst[0]; d->view.value = (const int64_t *)dst[1];
    d->view.nulls = (const uint8_t *)dst[2]; d->view.W = (const tr_math_i128 *)dst[3];
    d->view.Q = (const tr_dist_i256 *)dst[4]; d->view.count = (const uint64_t *)dst[5];
    d->view.totals = (const tr_pairplan_totals *)dst[6];
    return d;
}
uint64_t tr_mgpu_pairplan_bytes(const tr_mgpu_pairplan *d) { return d ? d->bytes : 0; }
void tr_mgpu_pairplan_free(tr_mgpu_pairplan *d) { tr_pairplan_device_free(d); }

int tr_mgpu_pairplan_query(const tr_mgpu_pairplan *d, const tr_pairplan_kernel_view *k,
                           const tr_pairplan_request *req, uint64_t n, tr_pairplan_result *out,
                           const tr_math_budget *budget, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!d || !k || (n && (!req || !out)) || k->policy != d->view.policy || k->ncols != d->view.ncols ||
        (k->ntable && !k->table)) return -1;                                   /* TR_MATH_INVALID */
    tr_pairplan_table_view hv = d->view;                                       /* host copies for the checks */
    hv.offset = d->host_offset; hv.totals = d->host_totals;
    hv.value = NULL; hv.nulls = NULL; hv.W = NULL; hv.Q = NULL; hv.count = NULL;
    tr_math_budget dflt;
    if (!budget) { tr_budget_default(&dflt); budget = &dflt; }
    int32_t *status = (int32_t *)calloc((size_t)(n ? n : 1), 4);
    uint64_t *offset = (uint64_t *)calloc((size_t)(n + 1), 8);
    if (!status || !offset) { free(status); free(offset); return -2; }
    tr_math_need need;
    for (int i = 0; i < TR_BUDGET_RESOURCES; i++) need.need[i] = 0;
    need.need[TR_BUDGET_OUTPUT] = n;
    for (uint64_t r = 0; r < n; r++) {
        status[r] = tr_pairplan_check(&hv, k, &req[r]);
        uint64_t tiles = 0;
        if (!status[r]) {
            tiles = tr_pairplan_request_tiles(&hv, &req[r]);
            need.need[TR_BUDGET_ITERATIONS] = tr_budget_add(need.need[TR_BUDGET_ITERATIONS], tr_pairplan_visits(
                tr_pairplan_group_entries(&hv, req[r].a), tr_pairplan_group_entries(&hv, req[r].b), req[r].a == req[r].b,
                (req[r].flags & TR_PAIRPLAN_ORDERED) != 0));
        }
        offset[r + 1] = tr_budget_add(offset[r], tiles);
    }
    need.need[TR_BUDGET_BYTES] = tr_pairplan_batch_device_bytes(n, k->ncols, k->ntable);
    int rc = tr_budget_check(budget, &need, refusal);
    if (!rc && need.need[TR_BUDGET_BYTES] == TR_BUDGET_UNKNOWN) {
        tr_pairplan_refuse(refusal, TR_BUDGET_BYTES, TR_BUDGET_UNKNOWN, budget->limit[TR_BUDGET_BYTES]);
        rc = TR_MATH_BUDGET_CODE;
    }
    if (rc || !n) { free(status); free(offset); return rc; }
    const uint64_t tiles = offset[n], width = 2 * (uint64_t)k->ncols + 1;
    tr_pairplan_request *dreq = NULL;
    tr_pairplan_result *dres = NULL;
    tr_pairplan_kernel_view *dk = NULL, kc = *k;
    int32_t *dstatus = NULL;
    uint64_t *doff = NULL, *dacc = NULL, *dvis = NULL, *dovf = NULL;
    int64_t *dtable = NULL;
    rc = tr_cuda_alloc((void **)&dreq, (size_t)n * sizeof *dreq);
    if (!rc) rc = tr_cuda_alloc((void **)&dres, (size_t)n * sizeof *dres);
    if (!rc) rc = tr_cuda_alloc((void **)&dstatus, (size_t)n * 4);
    if (!rc) rc = tr_cuda_alloc((void **)&doff, (size_t)(n + 1) * 8);
    if (!rc) rc = tr_cuda_alloc((void **)&dacc, (size_t)n * 8 * TR_KERNEL_WORDS);
    if (!rc) rc = tr_cuda_alloc((void **)&dvis, (size_t)n * 8);
    if (!rc) rc = tr_cuda_alloc((void **)&dovf, (size_t)n * 8);
    if (!rc) rc = tr_cuda_alloc((void **)&dk, sizeof *dk);
    if (!rc && k->ntable) {
        rc = tr_cuda_alloc((void **)&dtable, (size_t)(k->ntable * width * 8));
        if (!rc) rc = tr_cuda_copy_in(dtable, k->table, (size_t)(k->ntable * width * 8));
    }
    kc.table = dtable;
    if (!rc) rc = tr_cuda_copy_in(dk, &kc, sizeof kc);
    if (!rc) rc = tr_cuda_copy_in(dreq, req, (size_t)n * sizeof *dreq);
    if (!rc) rc = tr_cuda_copy_in(dstatus, status, (size_t)n * 4);
    if (!rc) rc = tr_cuda_copy_in(doff, offset, (size_t)(n + 1) * 8);
    if (!rc && tiles) {
        TR_LAUNCH(tr_pairplan_tile_kernel, tiles, d->view, dk, dreq, doff, n, tiles, dacc, dvis, dovf);
        rc = tr_cuda_sync();
    }
    if (!rc) {
        TR_LAUNCH(tr_pairplan_finish_kernel, n, d->view, dk, dreq, dstatus, dacc, dvis, dovf, n, dres);
        rc = tr_cuda_sync();
    }
    if (!rc) rc = tr_cuda_copy_out(out, dres, (size_t)n * sizeof *dres);
    tr_cuda_free(dreq); tr_cuda_free(dres); tr_cuda_free(dstatus); tr_cuda_free(doff); tr_cuda_free(dacc);
    tr_cuda_free(dvis); tr_cuda_free(dovf); tr_cuda_free(dk); tr_cuda_free(dtable);
    free(status); free(offset);
    return rc;
}
#ifdef __cplusplus
}
#endif
