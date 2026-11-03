/* Copyright 2026 Brian Richard RAMOS
 * SPDX-License-Identifier: Apache-2.0 */

__global__ void rows_bitmap_words_g(const gtable_t *t, const qplan_t *plan, uint64_t *out) {
    uint64_t word = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (word < t->nw) out[word] = filter_g(t, plan, word);
}

__global__ void rows_bitmap_semi_g(const gtable_t *t, const qplan_t *plan, uint64_t *out) {
    unsigned lane = threadIdx.x & 31;
    uint64_t word = ((uint64_t)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (word >= t->nw) return;
    uint64_t scope = lane == 0 ? filter_g(t, plan, word) : 0;
    scope = __shfl_sync(0xffffffffu, (unsigned long long)scope, 0);
    uint64_t result = 0;
    for (int half = 0; half < 2; ++half) {
        unsigned row = lane + 32 * half;
        bool keep = (scope >> row) & 1;
        for (int i = 0; i < plan->nsemi && keep; ++i) {
            uint32_t code = value_at(t, plan->semi_col[i], word, row);
            if (plan->semi_kmask[i]) code &= plan->semi_kmask[i];
            keep = code < plan->semi_n[i] && ((plan->semi_bm[i][code >> 6] >> (code & 63)) & 1);
        }
        uint32_t part = __ballot_sync(0xffffffffu, keep);
        result |= (uint64_t)part << (32 * half);
    }
    if (lane == 0) out[word] = result;
}

extern "C" int gpu_rows_bitmap(void *hv, const qplan_t *plan) {
    if (!hv || !plan || !plan->out_rows || plan->agg != A_ROWS || plan->nattr ||
        plan->nranges || plan->nsemi < 0 || plan->nsemi > 8 ||
        plan->xmode < 0 || plan->xmode > 2 || (plan->xmode && !plan->xrows))
        return -1;
    auto *h = (handle_t *)hv;
    auto filter = *plan;
    filter.nsemi = filter.xmode = 0;
    if (!topk_filter_ok(h->host, filter)) return -1;
    if (!h->host.nw) return 0;

    filter.nsemi = plan->nsemi;
    filter.xmode = plan->xmode;
    dtmp_t temp = {};
    uint64_t *out = nullptr;
    int rc = [&]() {
        for (int i = 0; i < filter.nsemi; ++i) {
            if (filter.semi_col[i] < 0 || filter.semi_col[i] >= h->host.k || filter.semi_n[i] > UINT32_MAX ||
                (filter.semi_n[i] && !filter.semi_bm[i]))
                return -1;
            size_t bytes = ((filter.semi_n[i] + 63) / 64) * sizeof(uint64_t);
            auto *copy = dev_copy(filter.semi_bm[i], bytes);
            if (!copy) return -2;
            filter.semi_bm[i] = (const uint64_t *)copy;
            temp.p[temp.n++] = copy;
        }
        if (filter.xmode) {
            auto *copy = dev_copy(filter.xrows, h->host.nw * sizeof(uint64_t));
            if (!copy) return -2;
            filter.xrows = (const uint64_t *)copy;
            temp.p[temp.n++] = copy;
        }
        CKI(cudaMalloc((void **)&out, h->host.nw * sizeof *out));
        CKI(cudaMemcpy(h->dq, &filter, sizeof filter, cudaMemcpyHostToDevice));
        if (filter.nsemi)
            rows_bitmap_semi_g<<<(unsigned)((h->host.nw + 7) / 8), 256>>>(h->dev, h->dq, out);
        else
            rows_bitmap_words_g<<<(unsigned)((h->host.nw + 127) / 128), 128>>>(h->dev, h->dq, out);
        CKI(cudaGetLastError());
        CKI(cudaMemcpy(plan->out_rows, out, h->host.nw * sizeof *out, cudaMemcpyDeviceToHost));
        return 0;
    }();
    if (out) cudaFree(out);
    for (int i = 0; i < temp.n; ++i) dev_free(temp.p[i]);
    return rc;
}
