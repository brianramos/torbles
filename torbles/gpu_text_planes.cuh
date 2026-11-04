/* Copyright 2026 Brian Richard RAMOS
 * SPDX-License-Identifier: Apache-2.0
 * Code-axis text moments over the executor's resident decoded planes. */
#ifndef EMULATE
namespace {
constexpr int TEXT_THREADS = 256;
constexpr int TEXT_KEYS = 128;
constexpr uint32_t TEXT_DOMAIN = 1u << 27;

template<typename T>
struct text_buffer {
    T *p = nullptr;
    ~text_buffer() { if (p) cudaFree(p); }
    bool alloc(size_t n) { return cudaMalloc((void **)&p, n * sizeof(T)) == cudaSuccess; }
    bool upload(const T *src, size_t n) {
        return alloc(n) && cudaMemcpy(p, src, n * sizeof(T), cudaMemcpyHostToDevice) == cudaSuccess;
    }
    bool zero(size_t n) {
        return alloc(n) && cudaMemset(p, 0, n * sizeof(T)) == cudaSuccess;
    }
    bool download(T *dst, size_t n) const {
        return cudaMemcpy(dst, p, n * sizeof(T), cudaMemcpyDeviceToHost) == cudaSuccess;
    }
};

struct text_timer {
    cudaEvent_t start = nullptr, end = nullptr;
    ~text_timer() { if (start) cudaEventDestroy(start); if (end) cudaEventDestroy(end); }
    bool begin() {
        return cudaEventCreate(&start) == cudaSuccess && cudaEventCreate(&end) == cudaSuccess &&
               cudaEventRecord(start) == cudaSuccess;
    }
    bool finish(double *ms) {
        float elapsed;
        if (cudaGetLastError() != cudaSuccess || cudaEventRecord(end) != cudaSuccess ||
            cudaEventSynchronize(end) != cudaSuccess || cudaEventElapsedTime(&elapsed, start, end) != cudaSuccess)
            return false;
        if (ms) *ms = elapsed;
        return true;
    }
};

__global__ void text_histogram_kernel(const gtable_t *table, const uint64_t *scope, int col,
                                      uint32_t domain, uint32_t *counts, uint32_t *invalid) {
    const uint64_t first = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t step = (uint64_t)gridDim.x * blockDim.x;
    for (uint64_t row = first; row < table->n; row += step) {
        uint64_t word = row >> 6;
        int bit = row & 63;
        if (scope && !((scope[word] >> bit) & 1)) continue;
        uint32_t code = value_at(table, col, word, bit);
        if (code >= domain) {
            atomicExch(invalid, 1u);
            continue;
        }
#if __CUDA_ARCH__ >= 700
        unsigned peers = __match_any_sync(__activemask(), code);
        if ((__ffs(peers) - 1) == (threadIdx.x & 31))
            atomicAdd(&counts[code], (uint32_t)__popc(peers));
#else
        atomicAdd(&counts[code], 1u);
#endif
    }
}

// Push the histogram's counts and moments forward along the regex code map.
__global__ void text_regex_moments_kernel(uint32_t domain, const uint32_t *counts,
                                          const uint32_t *mapping, const uint32_t *lengths,
                                          uint32_t groups, uint32_t *group_counts,
                                          unsigned long long *group_sums, uint32_t *group_min,
                                          uint32_t *invalid) {
    const uint64_t first = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t step = (uint64_t)gridDim.x * blockDim.x;
    for (uint64_t code = first; code < domain; code += step) {
        uint32_t count = counts[code];
        if (!count) continue;
        uint32_t group = mapping[code];
        if (group == UINT32_MAX) continue;
        if (group >= groups) {
            atomicExch(invalid, 1u);
            continue;
        }
        atomicAdd(&group_counts[group], count);
        atomicAdd(&group_sums[group], (unsigned long long)count * lengths[code]);
        atomicMin(&group_min[group], (uint32_t)code);
    }
}

__global__ void text_regex_compact_kernel(uint32_t groups, uint32_t min_count,
                                         const uint32_t *group_counts,
                                         const unsigned long long *group_sums,
                                         const uint32_t *group_min, uint32_t cap,
                                         uint32_t *out_groups, uint32_t *out_counts,
                                         unsigned long long *out_sums, uint32_t *out_min,
                                         uint32_t *out_n) {
    const uint64_t first = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t step = (uint64_t)gridDim.x * blockDim.x;
    for (uint64_t group = first; group < groups; group += step) {
        uint32_t count = group_counts[group];
        if (!count || count < min_count) continue;
        uint32_t slot = atomicAdd(out_n, 1u);
        if (slot >= cap) continue;
        out_groups[slot] = (uint32_t)group;
        out_counts[slot] = count;
        out_sums[slot] = group_sums[group];
        out_min[slot] = group_min[group];
    }
}

__global__ void text_lookup_kernel(const gtable_t *table, const uint64_t *scope,
                                   int key_col, int value_col, uint32_t mask,
                                   const uint32_t *lookup, uint64_t lookup_n,
                                   const uint32_t *keys, uint32_t nkeys,
                                   unsigned long long *sums, uint32_t *invalid) {
    __shared__ unsigned long long local[TEXT_KEYS];
    if (threadIdx.x < nkeys) local[threadIdx.x] = 0;
    __syncthreads();
    const uint64_t first = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t step = (uint64_t)gridDim.x * blockDim.x;
    for (uint64_t row = first; row < table->n; row += step) {
        uint64_t word = row >> 6;
        int bit = row & 63;
        if (scope && !((scope[word] >> bit) & 1)) continue;
        uint32_t code = value_at(table, key_col, word, bit);
        uint32_t lo = 0, hi = nkeys;
        while (lo < hi) {
            uint32_t mid = lo + (hi - lo) / 2;
            if (keys[mid] < code) lo = mid + 1;
            else hi = mid;
        }
        if (lo == nkeys || keys[lo] != code) continue;
        uint32_t index = value_at(table, value_col, word, bit) & mask;
        if (index >= lookup_n) {
            atomicExch(invalid, 1u);
            continue;
        }
        atomicAdd(&local[lo], (unsigned long long)lookup[index]);
    }
    __syncthreads();
    if (threadIdx.x < nkeys && local[threadIdx.x])
        atomicAdd(&sums[threadIdx.x], local[threadIdx.x]);
}

bool text_column(const handle_t *h, int col) {
    return h && h->dev && h->host.n <= UINT32_MAX && col >= 0 && col < h->host.k &&
           h->host.width[col] >= 0 && h->host.width[col] <= 32;
}

unsigned text_blocks(uint64_t rows, unsigned limit) {
    uint64_t blocks = (rows + TEXT_THREADS - 1) / TEXT_THREADS;
    return blocks < limit ? (unsigned)blocks : limit;
}
} // namespace

extern "C" int gpu_plane_code_histogram(void *handle, const uint64_t *scope, int32_t col,
                                          uint32_t domain, uint32_t *counts, double *ms) {
    if (ms) *ms = 0;
    const auto *h = (const handle_t *)handle;
    if (!text_column(h, col) || !domain || domain > (1u << 23) || !counts) return -1;
    if (!h->host.n) {
        memset(counts, 0, (size_t)domain * sizeof(*counts));
        return 0;
    }
    text_buffer<uint64_t> device_scope;
    text_buffer<uint32_t> device_counts, invalid;
    text_timer timer;
    if ((scope && !device_scope.upload(scope, h->host.nw)) ||
        !device_counts.zero(domain) || !invalid.zero(1) || !timer.begin()) return -2;
    text_histogram_kernel<<<text_blocks(h->host.n, 65535), TEXT_THREADS>>>(
        h->dev, device_scope.p, col, domain, device_counts.p, invalid.p);
    uint32_t bad;
    if (!timer.finish(ms) || !invalid.download(&bad, 1)) return -2;
    if (bad) return -1;
    return device_counts.download(counts, domain) ? 0 : -2;
}

extern "C" int gpu_plane_regex_moments(void *handle, const uint64_t *scope, int32_t col,
                                        uint32_t domain, const uint32_t *mapping, const uint32_t *lengths,
                                        uint32_t groups, uint32_t min_count, uint32_t cap,
                                        uint32_t *out_groups, uint32_t *out_counts, uint64_t *out_sums,
                                        uint32_t *out_min, uint32_t *out_n, double *ms) {
    if (ms) *ms = 0;
    if (!out_n) return -1;
    *out_n = 0;
    const auto *h = (const handle_t *)handle;
    if (!text_column(h, col) || !domain || domain > TEXT_DOMAIN || !groups ||
        groups > TEXT_DOMAIN || !mapping || !lengths || !cap ||
        !out_groups || !out_counts || !out_sums || !out_min) return -1;
    const uint32_t threshold = min_count ? min_count : 1;
    const uint32_t possible = (uint32_t)(h->host.n / threshold);
    if (!possible) return 0;
    uint32_t output_cap = cap < groups ? cap : groups;
    if (possible < output_cap) output_cap = possible;
    // Domain, group and output sizes are bounded above, independent of the bitmap.
    size_t bytes = (size_t)domain * 12 + (size_t)groups * 16 + (size_t)output_cap * 20
                 + (scope ? h->host.nw * sizeof(*scope) : 0) + 8;
    size_t available, total;
    if (cudaMemGetInfo(&available, &total) != cudaSuccess) return -2;
    if (bytes > available) return -1;

    text_buffer<uint64_t> device_scope;
    text_buffer<uint32_t> device_mapping, device_lengths, counts, group_counts, group_min;
    text_buffer<uint32_t> ids, compact_counts, compact_min, invalid, n;
    text_buffer<unsigned long long> group_sums, compact_sums;
    text_timer timer;
    if ((scope && !device_scope.upload(scope, h->host.nw)) ||
        !device_mapping.upload(mapping, domain) || !device_lengths.upload(lengths, domain) ||
        !counts.zero(domain) || !group_counts.zero(groups) || !group_sums.zero(groups) ||
        !group_min.alloc(groups) || cudaMemset(group_min.p, 0xff, (size_t)groups * sizeof(uint32_t)) != cudaSuccess ||
        !ids.alloc(output_cap) || !compact_counts.alloc(output_cap) || !compact_sums.alloc(output_cap) ||
        !compact_min.alloc(output_cap) || !invalid.zero(1) || !n.zero(1) || !timer.begin()) return -2;

    text_histogram_kernel<<<text_blocks(h->host.n, 65535), TEXT_THREADS>>>(
        h->dev, device_scope.p, col, domain, counts.p, invalid.p);
    if (cudaGetLastError() != cudaSuccess) return -2;
    text_regex_moments_kernel<<<text_blocks(domain, 65535), TEXT_THREADS>>>(
        domain, counts.p, device_mapping.p, device_lengths.p, groups, group_counts.p,
        group_sums.p, group_min.p, invalid.p);
    if (cudaGetLastError() != cudaSuccess) return -2;
    text_regex_compact_kernel<<<text_blocks(groups, 65535), TEXT_THREADS>>>(
        groups, threshold, group_counts.p, group_sums.p, group_min.p, output_cap, ids.p,
        compact_counts.p, compact_sums.p, compact_min.p, n.p);

    uint32_t bad, count;
    if (!timer.finish(ms) || !invalid.download(&bad, 1) || !n.download(&count, 1)) return -2;
    if (bad || count > output_cap) return -1;
    if (count && (!ids.download(out_groups, count) || !compact_counts.download(out_counts, count) ||
                  !compact_min.download(out_min, count) ||
                  cudaMemcpy(out_sums, compact_sums.p, (size_t)count * sizeof(*out_sums),
                             cudaMemcpyDeviceToHost) != cudaSuccess)) return -2;
    *out_n = count;
    return 0;
}

extern "C" int gpu_plane_lookup_sums(void *handle, const uint64_t *scope, int32_t key_col,
                                      int32_t value_col, uint32_t mask, const uint32_t *lookup,
                                      uint64_t lookup_n, const uint32_t *keys, uint32_t nkeys,
                                      uint64_t *sums, double *ms) {
    if (ms) *ms = 0;
    const auto *h = (const handle_t *)handle;
    if (!text_column(h, key_col) || !text_column(h, value_col) || !lookup || !lookup_n ||
        lookup_n > TEXT_DOMAIN || !keys || !nkeys || nkeys > TEXT_KEYS || !sums) return -1;
    for (uint32_t i = 0; i < nkeys; ++i)
        if ((i && keys[i] <= keys[i - 1]) || (uint64_t)keys[i] >= (UINT64_C(1) << h->host.width[key_col]))
            return -1;
    if (!h->host.n) {
        memset(sums, 0, (size_t)nkeys * sizeof(*sums));
        return 0;
    }
    text_buffer<uint64_t> device_scope;
    text_buffer<uint32_t> device_lookup, device_keys, invalid;
    text_buffer<unsigned long long> device_sums;
    text_timer timer;
    if ((scope && !device_scope.upload(scope, h->host.nw)) ||
        !device_lookup.upload(lookup, lookup_n) || !device_keys.upload(keys, nkeys) ||
        !device_sums.zero(nkeys) || !invalid.zero(1) || !timer.begin()) return -2;
    text_lookup_kernel<<<text_blocks(h->host.n, 4096), TEXT_THREADS>>>(
        h->dev, device_scope.p, key_col, value_col, mask, device_lookup.p, lookup_n,
        device_keys.p, nkeys, device_sums.p, invalid.p);
    uint32_t bad;
    if (!timer.finish(ms) || !invalid.download(&bad, 1)) return -2;
    if (bad) return -1;
    return cudaMemcpy(sums, device_sums.p, (size_t)nkeys * sizeof(*sums),
                      cudaMemcpyDeviceToHost) == cudaSuccess ? 0 : -2;
}
#else
extern "C" int gpu_plane_code_histogram(void *, const uint64_t *, int32_t, uint32_t, uint32_t *, double *) {
    return -1;
}
extern "C" int gpu_plane_regex_moments(void *, const uint64_t *, int32_t, uint32_t,
                                        const uint32_t *, const uint32_t *, uint32_t, uint32_t,
                                        uint32_t, uint32_t *, uint32_t *, uint64_t *, uint32_t *,
                                        uint32_t *, double *) {
    return -1;
}
extern "C" int gpu_plane_lookup_sums(void *, const uint64_t *, int32_t, int32_t, uint32_t,
                                      const uint32_t *, uint64_t, const uint32_t *, uint32_t, uint64_t *, double *) {
    return -1;
}
#endif
