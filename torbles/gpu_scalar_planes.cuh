/* Copyright 2026 Brian Richard RAMOS
 * SPDX-License-Identifier: Apache-2.0 */
struct gpu_moment_key_t {
    int32_t low, high, shift, transform;
    int64_t offset;
};

DEV void moment_add(uint64_t &lo, uint64_t &hi, uint64_t add_lo, uint64_t add_hi) {
    uint64_t before = lo;
    lo += add_lo;
    hi += add_hi + (lo < before);
}

__global__ void scalar_plane_moments_g(const gtable_t *table, const uint64_t *scope, int nsum,
                                      gpu_moment_key_t first, gpu_moment_key_t second, uint64_t *out) {
    uint64_t word = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t mask = word < table->nw ? scope[word] & rowmask_g(table, word) : 0;
    uint64_t count = popc(mask);
    uint64_t lo[2] = {}, hi[2] = {};
    for (int i = 0; i < nsum && mask; ++i) {
        gpu_moment_key_t key = i == 0 ? first : second;
        for (int half = 0; half < 2; ++half) {
            int col = half == 0 ? key.low : key.high;
            if (col < 0) continue;
            for (int plane = 0; plane < table->width[col]; ++plane) {
                unsigned bit = (half == 0 ? 0 : key.shift) + pw_g(table, col, plane);
                uint64_t n = popc(X(table, col, plane, word) & mask);
                moment_add(lo[i], hi[i], n << bit, bit ? n >> (64 - bit) : 0);
            }
        }
    }
    for (int delta = 16; delta; delta >>= 1) {
        count += __shfl_down_sync(0xffffffffu, (unsigned long long)count, delta);
        for (int i = 0; i < nsum; ++i) {
            uint64_t add_lo = __shfl_down_sync(0xffffffffu, (unsigned long long)lo[i], delta);
            uint64_t add_hi = __shfl_down_sync(0xffffffffu, (unsigned long long)hi[i], delta);
            moment_add(lo[i], hi[i], add_lo, add_hi);
        }
    }
    if ((threadIdx.x & 31) || !count) return;
    ADD_U64(out, count);
    for (int i = 0; i < nsum; ++i) {
        uint64_t prior = ADD_U64(out + 1 + 2 * i, lo[i]);
        ADD_U64(out + 2 + 2 * i, hi[i] + (prior + lo[i] < prior));
    }
}

extern "C" int gpu_scalar_plane_moments(void *hv, const uint64_t *scope, int32_t nsum,
                                       const gpu_moment_key_t *keys, uint64_t *out, double *ms) {
    if (!hv || !scope || !out || nsum < 0 || nsum > 2 || (nsum && !keys)) return -1;
    auto *h = (handle_t *)hv;
    const auto &t = h->host;
    if (t.n > UINT32_MAX) return -1;
    for (int i = 0; i < nsum; ++i) {
        const auto &key = keys[i];
        if (key.low < 0 || key.low >= t.k || key.high < -1 || key.high >= t.k || key.transform ||
            (key.high >= 0 && (key.shift < t.width[key.low] || key.shift > 64 - t.width[key.high])))
            return -1;
    }
    memset(out, 0, 5 * sizeof *out);
    if (ms) *ms = 0;
    if (!t.nw) return 0;
    auto *mask = (uint64_t *)dev_copy(scope, t.nw * sizeof *scope);
    uint64_t *state = nullptr;
    if (!mask) return -2;
    int rc = [&]() {
        CKI(cudaMalloc((void **)&state, 5 * sizeof *state));
        CKI(cudaMemset(state, 0, 5 * sizeof *state));
        scalar_plane_moments_g<<<(unsigned)((t.nw + 127) / 128), 128>>>(
            h->dev, mask, nsum, nsum ? keys[0] : gpu_moment_key_t{},
            nsum == 2 ? keys[1] : gpu_moment_key_t{}, state);
        CKI(cudaGetLastError());
        CKI(cudaMemcpy(out, state, 5 * sizeof *out, cudaMemcpyDeviceToHost));
        return 0;
    }();
    if (state) cudaFree(state);
    dev_free(mask);
    return rc;
}

__global__ void epoch_minute_histogram_g(const gtable_t *t, const uint64_t *scope,
                                         gpu_moment_key_t key, uint64_t remainder,
                                         uint32_t domain, uint32_t *histogram, unsigned *invalid) {
    uint64_t row = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= t->n || !(scope[row >> 6] >> (row & 63) & 1)) return;
    uint64_t code = value_at(t, key.low, row >> 6, row & 63);
    if (key.high >= 0) code |= (uint64_t)value_at(t, key.high, row >> 6, row & 63) << key.shift;
    const uint64_t minute = UINT64_C(60000000);
    uint64_t bucket = code / minute + (code % minute + remainder) / minute;
    if (bucket < domain) atomicAdd(histogram + bucket, 1U);
    else atomicExch(invalid, 1U);
}

extern "C" int gpu_epoch_minute_histogram(void *hv, const uint64_t *scope, const gpu_moment_key_t *key,
                                         uint32_t domain, uint32_t *counts) {
    if (!hv || !scope || !key || !counts || !domain || domain > (1U << 20)) return -1;
    auto *h = (handle_t *)hv;
    const auto &t = h->host;
    if (t.n > UINT32_MAX || key->transform != 2 || key->low < 0 || key->low >= t.k ||
        key->high < -1 || key->high >= t.k ||
        (key->high >= 0 && (key->shift < t.width[key->low] || key->shift > 64 - t.width[key->high])))
        return -1;
    if (!t.n) { memset(counts, 0, domain * sizeof *counts); return 0; }
    const int64_t minute = INT64_C(60000000);
    int64_t rem = key->offset % minute;
    if (rem < 0) rem += minute;
    auto *mask = (uint64_t *)dev_copy(scope, t.nw * sizeof *scope);
    uint32_t *state = nullptr;
    if (!mask) return -2;
    int rc = [&]() {
        CKI(cudaMalloc((void **)&state, (domain + 1) * sizeof *state));
        CKI(cudaMemset(state, 0, (domain + 1) * sizeof *state));
        epoch_minute_histogram_g<<<(unsigned)((t.n + 255) / 256), 256>>>(
            h->dev, mask, *key, rem, domain, state, state + domain);
        CKI(cudaGetLastError());
        uint32_t invalid = 0;
        CKI(cudaMemcpy(&invalid, state + domain, sizeof invalid, cudaMemcpyDeviceToHost));
        if (invalid) return -1;
        CKI(cudaMemcpy(counts, state, domain * sizeof *state, cudaMemcpyDeviceToHost));
        return 0;
    }();
    if (state) cudaFree(state);
    dev_free(mask);
    return rc;
}

struct gpu_point_state_t { uint32_t count; uint64_t ids[256]; };

__global__ void wide_equal_ids_g(const gtable_t *t, const uint64_t *scope, gpu_moment_key_t key,
                                uint32_t low, uint32_t high, gpu_point_state_t *state) {
    uint64_t w = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (w >= t->nw) return;
    uint64_t rows = scope[w] & rowmask_g(t, w);
    if (rows) rows &= cmp_col(t, key.low, w, P_EQ, low);
    if (rows && key.high >= 0) rows &= cmp_col(t, key.high, w, P_EQ, high);
    if (!rows) return;
    unsigned at = atomicAdd(&state->count, popc(rows));
    while (rows && at < 256) {
        state->ids[at++] = w * 64 + ctz64(rows);
        rows &= rows - 1;
    }
}

extern "C" int gpu_plane_equal_ids(void *hv, const uint64_t *scope, const gpu_moment_key_t *key,
                                   uint64_t code, uint64_t *ids, uint32_t *count) {
    if (!hv || !scope || !key || !ids || !count || key->transform) return -1;
    auto *h = (handle_t *)hv;
    const auto &t = h->host;
    if (t.n > UINT32_MAX || key->low < 0 || key->low >= t.k || key->high < -1 || key->high >= t.k)
        return -1;
    int low_width = t.width[key->low], high_width = key->high < 0 ? 0 : t.width[key->high];
    if (key->high >= 0 && (key->shift < low_width || key->shift >= 64 || key->shift > 64 - high_width))
        return -1;
    uint32_t low = (uint32_t)(code & ((UINT64_C(1) << low_width) - 1));
    uint32_t high = key->high < 0 ? 0 : (uint32_t)(code >> key->shift);
    *count = 0;
    if ((high_width < 32 && high >> high_width) ||
        code != (low | (key->high < 0 ? 0 : (uint64_t)high << key->shift)) || !t.nw)
        return 0;
    auto *mask = (uint64_t *)dev_copy(scope, t.nw * sizeof *scope);
    if (!mask) return -2;
    gpu_point_state_t *state = nullptr;
    int rc = [&]() {
        CKI(cudaMalloc((void **)&state, sizeof *state));
        CKI(cudaMemset(state, 0, sizeof state->count));
        wide_equal_ids_g<<<(unsigned)((t.nw + 127) / 128), 128>>>(h->dev, mask, *key, low, high, state);
        CKI(cudaGetLastError());
        CKI(cudaMemcpy(count, &state->count, sizeof *count, cudaMemcpyDeviceToHost));
        if (*count > 256) return 1;
        if (*count) CKI(cudaMemcpy(ids, state->ids, *count * sizeof *ids, cudaMemcpyDeviceToHost));
        return 0;
    }();
    if (state) cudaFree(state);
    dev_free(mask);
    return rc;
}
