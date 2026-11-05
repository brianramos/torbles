/* Copyright 2026 Brian Richard RAMOS
 * SPDX-License-Identifier: Apache-2.0
 * Code-axis text moments over the executor's resident decoded planes. */
#ifndef EMULATE
#include <cuda/atomic>
#include <memory>

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

struct text_referrer_span {
    uint32_t length;
    uint8_t start, suffix_lf;
};

struct text_referrer_projection {
    uint32_t domain;
    text_buffer<uint32_t> mapping, bytes, chars;
    text_buffer<text_referrer_span> spans;
};

// Python's default ^https?://(?:www\.)?([^/]+)/.*$ with replacement \1.
DEV text_referrer_span text_referrer_project(const uint8_t *s, uint32_t length, uint32_t *chars) {
    uint32_t start = 0;
    if (length >= 7 && s[0] == 'h' && s[1] == 't' && s[2] == 't' && s[3] == 'p') {
        if (s[4] == ':' && s[5] == '/' && s[6] == '/') start = 7;
        else if (length >= 8 && s[4] == 's' && s[5] == ':' && s[6] == '/' && s[7] == '/') start = 8;
    }
    uint32_t slash = length, nchar = 0;
    bool interior_lf = false;
    for (uint32_t i = 0; i < length; ++i) {
        uint8_t c = s[i];
        nchar += (c & 0xc0) != 0x80; // The caller supplies valid UTF-8.
        if (start && i >= start) {
            if (slash == length) {
                if (c == '/') slash = i;
            } else if (c == '\n' && i + 1 < length) {
                interior_lf = true;
            }
        }
    }
    *chars = nchar;
    if (!start || slash == start || slash == length || interior_lf) return {length, 0, 0};
    // If www. is the entire host, the optional prefix must backtrack.
    if (slash - start > 4 && s[start] == 'w' && s[start + 1] == 'w' &&
        s[start + 2] == 'w' && s[start + 3] == '.') start += 4;
    return {slash - start, (uint8_t)start, (uint8_t)(s[length - 1] == '\n')};
}

__global__ void text_referrer_project_kernel(
    const uint8_t *blob, uint64_t blob_bytes, const int64_t *offsets, uint32_t domain,
    text_referrer_span *spans, uint32_t *bytes, uint32_t *chars, uint32_t *invalid) {
    const uint64_t first = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t step = (uint64_t)gridDim.x * blockDim.x;
    for (uint64_t code = first; code < domain; code += step) {
        int64_t begin = offsets[code], end = offsets[code + 1];
        if (begin < 0 || end < begin || (uint64_t)end > blob_bytes || (uint64_t)(end - begin) > UINT32_MAX) {
            atomicExch(invalid, 1u);
            continue;
        }
        uint32_t length = (uint32_t)(end - begin);
        bytes[code] = length;
        spans[code] = text_referrer_project(blob + begin, length, chars + code);
    }
}

DEV uint64_t text_referrer_hash(const uint8_t *s, text_referrer_span span) {
    uint64_t hash = UINT64_C(14695981039346656037);
    for (uint32_t i = 0; i < span.length; ++i)
        hash = (hash ^ s[i]) * UINT64_C(1099511628211);
    if (span.suffix_lf) hash = (hash ^ '\n') * UINT64_C(1099511628211);
    hash ^= hash >> 33;
    hash *= UINT64_C(0xff51afd7ed558ccd);
    return hash ^ (hash >> 33);
}

DEV bool text_referrer_equal(const uint8_t *a, text_referrer_span as,
                             const uint8_t *b, text_referrer_span bs) {
    if (as.length + as.suffix_lf != bs.length + bs.suffix_lf) return false;
    uint32_t common = as.length < bs.length ? as.length : bs.length;
    for (uint32_t i = 0; i < common; ++i)
        if (a[i] != b[i]) return false;
    return as.length == bs.length || (as.length > common ? a[common] : b[common]) == '\n';
}

__global__ void text_referrer_intern_kernel(
    const uint8_t *blob, const int64_t *offsets, uint32_t domain, const text_referrer_span *spans,
    unsigned long long *slots, uint64_t mask, uint32_t *mapping, uint32_t *invalid) {
    cuda::atomic_ref<uint32_t, cuda::thread_scope_device> bad(*invalid);
    if (bad.load(cuda::memory_order_relaxed)) return; // A malformed offset prevents all blob reads.
    const uint64_t first = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t step = (uint64_t)gridDim.x * blockDim.x;
    for (uint64_t code = first; code < domain; code += step) {
        text_referrer_span span = spans[code];
        const uint8_t *s = blob + offsets[code] + span.start;
        uint64_t hash = text_referrer_hash(s, span);
        unsigned long long ticket = (hash & UINT64_C(0xffffffff00000000)) | (code + 1);
        bool found = false;
        for (uint64_t probe = 0; probe <= mask; ++probe) {
            cuda::atomic_ref<unsigned long long, cuda::thread_scope_device> slot(slots[(hash + probe) & mask]);
            unsigned long long current = slot.load(cuda::memory_order_relaxed);
            if (!current && slot.compare_exchange_strong(current, ticket, cuda::memory_order_relaxed)) {
                mapping[code] = (uint32_t)code;
                found = true;
                break;
            }
            if ((current >> 32) != (ticket >> 32)) continue;
            uint32_t rep = (uint32_t)current - 1;
            text_referrer_span other = spans[rep];
            if (text_referrer_equal(s, span, blob + offsets[rep] + other.start, other)) {
                mapping[code] = rep;
                found = true;
                break;
            }
        }
        if (!found) bad.store(1u, cuda::memory_order_relaxed);
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
                                         uint32_t *out_n, const text_referrer_span *spans,
                                         text_referrer_span *out_spans) {
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
        if (spans) out_spans[slot] = spans[group];
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

namespace {
int text_mapped_moments(const handle_t *h, const uint64_t *scope, int32_t col,
                        uint32_t domain, const uint32_t *mapping, const uint32_t *lengths,
                        uint32_t groups, uint32_t min_count, uint32_t cap,
                        uint32_t *out_groups, uint32_t *out_counts, uint64_t *out_sums,
                        uint32_t *out_min, uint32_t *out_n, double *ms,
                        const text_referrer_span *spans = nullptr, uint32_t *out_start = nullptr,
                        uint32_t *out_length = nullptr, uint8_t *out_lf = nullptr) {
    const uint32_t threshold = min_count ? min_count : 1;
    const uint32_t possible = (uint32_t)(h->host.n / threshold);
    if (!possible) return 0;
    uint32_t output_cap = cap < groups ? cap : groups;
    if (possible < output_cap) output_cap = possible;
    size_t bytes = (size_t)domain * 4 + (size_t)groups * 16 + (size_t)output_cap * 20
                 + (scope ? h->host.nw * sizeof(*scope) : 0) + 8;
    if (spans) bytes += (size_t)output_cap * sizeof(*spans);
    size_t available, total;
    if (cudaMemGetInfo(&available, &total) != cudaSuccess) return -2;
    if (bytes > available) return -1;

    text_buffer<uint64_t> device_scope;
    text_buffer<uint32_t> counts, group_counts, group_min;
    text_buffer<uint32_t> ids, compact_counts, compact_min, invalid, n;
    text_buffer<unsigned long long> group_sums, compact_sums;
    text_buffer<text_referrer_span> compact_spans;
    text_timer timer;
    if ((scope && !device_scope.upload(scope, h->host.nw)) ||
        !counts.zero(domain) || !group_counts.zero(groups) || !group_sums.zero(groups) ||
        !group_min.alloc(groups) || cudaMemset(group_min.p, 0xff, (size_t)groups * sizeof(uint32_t)) != cudaSuccess ||
        !ids.alloc(output_cap) || !compact_counts.alloc(output_cap) || !compact_sums.alloc(output_cap) ||
        !compact_min.alloc(output_cap) || (spans && !compact_spans.alloc(output_cap)) ||
        !invalid.zero(1) || !n.zero(1) || !timer.begin()) return -2;

    text_histogram_kernel<<<text_blocks(h->host.n, 65535), TEXT_THREADS>>>(
        h->dev, device_scope.p, col, domain, counts.p, invalid.p);
    if (cudaGetLastError() != cudaSuccess) return -2;
    text_regex_moments_kernel<<<text_blocks(domain, 65535), TEXT_THREADS>>>(
        domain, counts.p, mapping, lengths, groups, group_counts.p,
        group_sums.p, group_min.p, invalid.p);
    if (cudaGetLastError() != cudaSuccess) return -2;
    text_regex_compact_kernel<<<text_blocks(groups, 65535), TEXT_THREADS>>>(
        groups, threshold, group_counts.p, group_sums.p, group_min.p, output_cap, ids.p,
        compact_counts.p, compact_sums.p, compact_min.p, n.p, spans, compact_spans.p);

    uint32_t bad, count;
    if (!timer.finish(ms) || !invalid.download(&bad, 1) || !n.download(&count, 1)) return -2;
    if (bad || count > output_cap) return -1;
    if (count && (!ids.download(out_groups, count) || !compact_counts.download(out_counts, count) ||
                  !compact_min.download(out_min, count) ||
                  cudaMemcpy(out_sums, compact_sums.p, (size_t)count * sizeof(*out_sums),
                             cudaMemcpyDeviceToHost) != cudaSuccess)) return -2;
    if (count && spans) {
        std::unique_ptr<text_referrer_span[]> host(new (std::nothrow) text_referrer_span[count]);
        if (!host || !compact_spans.download(host.get(), count)) return -2;
        for (uint32_t i = 0; i < count; ++i) {
            out_start[i] = host[i].start;
            out_length[i] = host[i].length;
            out_lf[i] = host[i].suffix_lf;
        }
    }
    *out_n = count;
    return 0;
}
} // namespace

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
    if (h->host.n < (min_count ? min_count : 1)) return 0;
    text_buffer<uint32_t> device_mapping, device_lengths;
    if (!device_mapping.upload(mapping, domain) || !device_lengths.upload(lengths, domain)) return -2;
    return text_mapped_moments(h, scope, col, domain, device_mapping.p, device_lengths.p, groups,
                              min_count, cap, out_groups, out_counts, out_sums, out_min, out_n, ms);
}

extern "C" void *gpu_referrer_prepare(const uint8_t *blob, uint64_t blob_bytes, const int64_t *offsets,
                                       uint32_t domain, double *ms) {
    if (ms) *ms = 0;
    if (!offsets || !domain || domain > TEXT_DOMAIN || (!blob && blob_bytes) ||
        offsets[0] != 0 || offsets[domain] < 0 || (uint64_t)offsets[domain] != blob_bytes) return nullptr;
    // Keep open addressing below 2/3 load even for a fully distinct dictionary.
    size_t cells = 1;
    while (cells < (size_t)domain + ((size_t)domain + 1) / 2) cells <<= 1;
    const size_t fixed = ((size_t)domain + 1) * sizeof(*offsets) +
                         (size_t)domain * (3 * sizeof(uint32_t) + sizeof(text_referrer_span)) +
                         cells * sizeof(unsigned long long) + sizeof(uint32_t);
    size_t available, total;
    if (cudaMemGetInfo(&available, &total) != cudaSuccess ||
        fixed >= available || blob_bytes > available - fixed) return nullptr;

    std::unique_ptr<text_referrer_projection> plan(new (std::nothrow) text_referrer_projection);
    if (!plan) return nullptr;
    plan->domain = domain;
    text_buffer<uint8_t> device_blob;
    text_buffer<int64_t> device_offsets;
    text_buffer<unsigned long long> slots;
    text_buffer<uint32_t> invalid;
    text_timer timer;
    if (!(blob_bytes ? device_blob.upload(blob, (size_t)blob_bytes) : device_blob.alloc(1)) ||
        !device_offsets.upload(offsets, (size_t)domain + 1) || !slots.zero(cells) ||
        !invalid.zero(1) || !plan->mapping.alloc(domain) || !plan->bytes.alloc(domain) ||
        !plan->chars.alloc(domain) || !plan->spans.alloc(domain) || !timer.begin()) return nullptr;
    const unsigned blocks = text_blocks(domain, 65535);
    text_referrer_project_kernel<<<blocks, TEXT_THREADS>>>(
        device_blob.p, blob_bytes, device_offsets.p, domain, plan->spans.p,
        plan->bytes.p, plan->chars.p, invalid.p);
    if (cudaGetLastError() != cudaSuccess) return nullptr;
    text_referrer_intern_kernel<<<blocks, TEXT_THREADS>>>(
        device_blob.p, device_offsets.p, domain, plan->spans.p, slots.p, cells - 1,
        plan->mapping.p, invalid.p);
    uint32_t bad;
    if (!timer.finish(ms) || !invalid.download(&bad, 1) || bad) return nullptr;
    return plan.release();
}

extern "C" void gpu_referrer_free(void *projection) {
    delete (text_referrer_projection *)projection;
}

extern "C" int gpu_referrer_moments(void *projection, void *handle, const uint64_t *scope, int32_t col,
                                    int32_t byte_length, uint32_t min_count, uint32_t cap,
                                    uint32_t *out_reps, uint32_t *out_counts, uint64_t *out_sums,
                                    uint32_t *out_min, uint32_t *out_start, uint32_t *out_length,
                                    uint8_t *out_lf, uint32_t *out_n, double *ms) {
    if (ms) *ms = 0;
    if (!out_n) return -1;
    *out_n = 0;
    const auto *plan = (const text_referrer_projection *)projection;
    const auto *h = (const handle_t *)handle;
    if (!plan || !text_column(h, col) || (byte_length != 0 && byte_length != 1) || !cap ||
        !out_reps || !out_counts || !out_sums || !out_min || !out_start || !out_length || !out_lf) return -1;
    const uint32_t *lengths = byte_length ? plan->bytes.p : plan->chars.p;
    return text_mapped_moments(h, scope, col, plan->domain, plan->mapping.p, lengths, plan->domain,
                              min_count, cap, out_reps, out_counts, out_sums, out_min, out_n, ms,
                              plan->spans.p, out_start, out_length, out_lf);
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
extern "C" void *gpu_referrer_prepare(const uint8_t *, uint64_t, const int64_t *, uint32_t, double *) {
    return nullptr;
}
extern "C" void gpu_referrer_free(void *) {}
extern "C" int gpu_referrer_moments(void *, void *, const uint64_t *, int32_t, int32_t,
                                    uint32_t, uint32_t, uint32_t *, uint32_t *, uint64_t *,
                                    uint32_t *, uint32_t *, uint32_t *, uint8_t *, uint32_t *, double *) {
    return -1;
}
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
