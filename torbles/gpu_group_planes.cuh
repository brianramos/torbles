/* Copyright 2026 Brian Richard RAMOS
 * SPDX-License-Identifier: Apache-2.0 */
#pragma once

#ifndef EMULATE
#include <cub/block/block_radix_sort.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_reduce.cuh>

enum { GPU_GROUP_KEYS = 5, GPU_GROUP_SUMS = 2, GPU_GROUP_LIMIT = 1024, GPU_GROUP_SMALL = 16,
       GPU_GROUP_THREADS = 128, GPU_GROUP_ITEMS = 4,
       GPU_GROUP_TILE = GPU_GROUP_THREADS * GPU_GROUP_ITEMS };

struct gpu_group_key_t {
    int32_t low, high, shift, transform;
    int64_t offset;
};

struct gpu_group_plan_t {
    int nkey, nsum;
    gpu_group_key_t key[GPU_GROUP_KEYS];
    int32_t sums[GPU_GROUP_SUMS];
    gpu_group_key_t distinct;
    const uint32_t *projection;
    uint64_t projection_len, projection_mask;
    uint32_t projection_width;
    int projected;
};

struct gpu_group_tuple_t { uint64_t key[GPU_GROUP_KEYS]; };
struct gpu_group_result_t {
    gpu_group_tuple_t tuple;
    uint64_t sums[GPU_GROUP_SUMS];
    uint32_t count, distinct;
};

DEV uint32_t gpu_group_code(const gtable_t *table, int col, uint32_t row) {
    uint64_t word = row >> 6;
    unsigned lane = row & 63;
    uint32_t code = 0;
    for (int b = 0; b < table->width[col]; ++b)
        code |= (uint32_t)((table->P[col][b][word] >> lane) & 1) << pw_g(table, col, b);
    return code;
}

DEV uint64_t gpu_group_value(const gtable_t *table, const gpu_group_key_t &key, uint32_t row) {
    uint64_t value = gpu_group_code(table, key.low, row);
    if (key.high >= 0) value |= (uint64_t)gpu_group_code(table, key.high, row) << key.shift;
    if (key.transform == 1) {
        const int64_t hour = INT64_C(3600000000);
        int64_t us = (int64_t)(value % hour) + key.offset;
        if (us < 0) us += hour;
        if (us >= hour) us -= hour;
        value = (uint64_t)us / UINT64_C(60000000);
    }
    return value;
}

DEV gpu_group_tuple_t gpu_group_key(const gtable_t *table, const gpu_group_plan_t &plan, uint32_t row) {
    gpu_group_tuple_t tuple{};
    for (int i = 0; i < plan.nkey; ++i) {
        uint64_t value = gpu_group_value(table, plan.key[i], row);
        if (plan.key[i].transform == 3) {
            uint64_t index = value & plan.projection_mask;
            value = index < plan.projection_len ? plan.projection[index] : UINT64_MAX;
            if (value >= (UINT64_C(1) << plan.projection_width)) value = UINT64_MAX;
        }
        tuple.key[i] = value;
    }
    return tuple;
}

DEV uint64_t gpu_group_mix(uint64_t x) {
    x = (x ^ (x >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)) * UINT64_C(0x94d049bb133111eb);
    return x ^ (x >> 31);
}

DEV uint64_t gpu_group_hash(gpu_group_tuple_t tuple, int nkey) {
    uint64_t hash = tuple.key[0];
    for (int i = 1; i < nkey; ++i)
        hash ^= gpu_group_mix(tuple.key[i] + UINT64_C(0x9e3779b97f4a7c15) * (unsigned)i);
    return gpu_group_mix(hash);
}

DEV bool gpu_group_equal(gpu_group_tuple_t a, gpu_group_tuple_t b, int nkey) {
    for (int i = 0; i < nkey; ++i) if (a.key[i] != b.key[i]) return false;
    return true;
}

/* Publishing only a representative row is enough: that row's full key is
 * already immutable in the device planes. No lock, decoded key vector or
 * fingerprint-only equivalence is needed. */
template <bool Distinct> __global__ void gpu_group_accumulate(const gtable_t *table, const uint64_t *scope,
                                     gpu_group_plan_t plan, uint32_t slots,
                                     unsigned long long *owners, uint32_t *counts,
                                     uint64_t *sums0, uint64_t *sums1, unsigned long long *pairs,
                                     uint32_t *uniques, unsigned int *invalid) {
    uint64_t row = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= table->n || (scope && !((scope[row >> 6] >> (row & 63)) & 1))) return;
    gpu_group_tuple_t tuple = gpu_group_key(table, plan, (uint32_t)row);
    if (plan.projected >= 0 && tuple.key[plan.projected] == UINT64_MAX) {
        atomicExch(invalid, 1U);
        return;
    }
#if __CUDA_ARCH__ >= 700
    if (!counts && plan.nkey == 1) {
        unsigned peers = __match_any_sync(__activemask(), (unsigned long long)tuple.key[0]);
        if ((threadIdx.x & 31) != (unsigned)(__ffs(peers) - 1)) return;
    }
#endif
    uint64_t hash = gpu_group_hash(tuple, plan.nkey);
    uint32_t pos = (uint32_t)hash & (slots - 1);
    uint64_t tag = hash & UINT64_C(0xffffffff00000000);
    uint64_t token = tag | (row + 1);
    for (;;) {
        uint64_t owner = atomicCAS(owners + pos, 0ULL, (unsigned long long)token);
        if (!owner || ((owner & UINT64_C(0xffffffff00000000)) == tag &&
                       gpu_group_equal(tuple, gpu_group_key(table, plan, (uint32_t)owner - 1), plan.nkey)))
            break;
        pos = (pos + 1) & (slots - 1);
    }
    if (Distinct) {
        uint64_t value = gpu_group_value(table, plan.distinct, (uint32_t)row);
        uint32_t cell = (uint32_t)gpu_group_mix(value ^ gpu_group_mix(pos)) & (slots - 1);
        uint64_t pair = ((uint64_t)pos << 32) | (row + 1);
        for (;;) {
            uint64_t owner = atomicCAS(pairs + cell, 0ULL, (unsigned long long)pair);
            if (!owner) {
                atomicAdd(uniques + pos, 1U);
                break;
            }
            if ((uint32_t)(owner >> 32) == pos &&
                gpu_group_value(table, plan.distinct, (uint32_t)owner - 1) == value) break;
            cell = (cell + 1) & (slots - 1);
        }
    }
    if (counts) atomicAdd(counts + pos, 1U);
    if (plan.nsum > 0) ADD_U64(sums0 + pos, gpu_group_code(table, plan.sums[0], (uint32_t)row));
    if (plan.nsum > 1) ADD_U64(sums1 + pos, gpu_group_code(table, plan.sums[1], (uint32_t)row));
}

/* Score encodes descending count and a slot tie-breaker. Unspecified SQL
 * ties can select any group at the cutoff. Each tile emits only its top k,
 * so recursively taking top k preserves all candidates that can win. */
template <bool First> __global__ void gpu_group_select(const uint32_t *counts,
                  const uint64_t *input, uint32_t n, unsigned k, uint64_t *output) {
    using sort_t = cub::BlockRadixSort<uint64_t, GPU_GROUP_THREADS, GPU_GROUP_ITEMS>;
    __shared__ typename sort_t::TempStorage storage;
    uint64_t score[GPU_GROUP_ITEMS];
    for (int item = 0; item < GPU_GROUP_ITEMS; ++item) {
        uint32_t at = blockIdx.x * GPU_GROUP_TILE + threadIdx.x + item * GPU_GROUP_THREADS;
        score[item] = 0;
        if (at < n) {
            if (First) {
                uint32_t count = counts[at];
                if (count) score[item] = ((uint64_t)count << 32) | (UINT32_MAX - at);
            } else {
                score[item] = input[at];
            }
        }
    }
    sort_t(storage).SortDescending(score);
    for (int item = 0; item < GPU_GROUP_ITEMS; ++item) {
        unsigned rank = threadIdx.x * GPU_GROUP_ITEMS + item;
        if (rank < k) output[blockIdx.x * k + rank] = score[item];
    }
}

__global__ void gpu_group_gather(const gtable_t *table, gpu_group_plan_t plan,
                   const unsigned long long *owners, const uint64_t *scores,
                   const uint32_t *counts, const uint32_t *uniques, const uint64_t *sums0, const uint64_t *sums1, unsigned k, gpu_group_result_t *out) {
    unsigned rank = blockIdx.x * blockDim.x + threadIdx.x;
    if (rank >= k) return;
    uint64_t score = scores[rank];
    gpu_group_result_t entry{};
    if (score) {
        uint32_t slot = UINT32_MAX - (uint32_t)score;
        entry.tuple = gpu_group_key(table, plan, (uint32_t)owners[slot] - 1);
        entry.count = counts[slot];
        if (uniques) entry.distinct = uniques[slot];
        if (plan.nsum > 0) entry.sums[0] = sums0[slot];
        if (plan.nsum > 1) entry.sums[1] = sums1[slot];
    }
    out[rank] = entry;
}

__global__ void gpu_group_scores(const uint32_t *counts, uint32_t slots, uint64_t *scores) {
    uint32_t slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot < slots) {
        uint32_t count = counts[slot];
        scores[slot] = count ? ((uint64_t)count << 32) | (UINT32_MAX - slot) : 0;
    }
}

template <class T> struct gpu_group_buffer {
    T *data = nullptr;
    ~gpu_group_buffer() { if (data) cudaFree(data); }
    bool alloc(size_t n) {
        return !n || cudaMalloc((void **)&data, n * sizeof(T)) == cudaSuccess;
    }
    bool clear(size_t n) {
        return !n || cudaMemset(data, 0, n * sizeof(T)) == cudaSuccess;
    }
};

static bool gpu_group_col(const gtable_t &t, int col) {
    return col >= 0 && col < t.k && t.width[col] > 0 && t.width[col] <= 32;
}

static bool gpu_group_prepare_key(const gtable_t &t, const gpu_group_key_t &key, gpu_group_key_t &out) {
    if (!gpu_group_col(t, key.low) || key.high < -1 || key.transform < 0 || key.transform > 1)
        return false;
    if (key.high >= 0 && (!gpu_group_col(t, key.high) || key.shift < t.width[key.low] ||
                          key.shift >= 64 || t.width[key.high] > 64 - key.shift))
        return false;
    out = key;
    out.offset %= INT64_C(3600000000);
    return true;
}

static uint64_t gpu_group_active(const gtable_t &t, const uint64_t *filter) {
    if (!filter) return t.n;
    uint64_t active = 0;
    for (uint64_t w = 0; w < t.nw; ++w) {
        uint64_t left = t.n - w * 64;
        uint64_t mask = left >= 64 ? UINT64_MAX : (UINT64_C(1) << left) - 1;
        active += __builtin_popcountll(filter[w] & mask);
    }
    return active;
}

static int gpu_group_run(void *hv, const uint64_t *filter, int32_t nkey,
                    const gpu_group_key_t *keys, int32_t nsum, const int32_t *sum_cols,
                    uint32_t k, uint64_t *out_keys, uint64_t *out_counts,
                    uint64_t *out_sums, uint32_t *out_n, const gpu_group_key_t *distinct,
                    uint32_t rank, uint64_t *out_distinct, const uint32_t *projection,
                    uint64_t projection_len, uint64_t projection_mask, uint32_t projection_width) {
    if (!out_n) return -1;
    *out_n = 0;
    if (!hv || nkey < 1 || nkey > GPU_GROUP_KEYS || !keys || nsum < 0 || nsum > GPU_GROUP_SUMS ||
        (nsum && (!sum_cols || !out_sums)) || !k || k > GPU_GROUP_LIMIT || !out_keys || !out_counts ||
        (distinct && !out_distinct) || rank > 1 || (rank && !distinct) ||
        (projection && (!projection_len || projection_len > SIZE_MAX / sizeof *projection ||
                        !projection_width || projection_width > 32)))
        return -1;
    handle_t *h = (handle_t *)hv;
    const gtable_t &t = h->host;
    if (t.n > UINT32_MAX) return -1;
    gpu_group_plan_t plan{};
    plan.nkey = nkey;
    plan.nsum = nsum;
    plan.projected = -1;
    for (int i = 0; i < nkey; ++i) {
        if (keys[i].transform == 3) {
            if (!projection || plan.projected >= 0 || keys[i].high != -1 || !gpu_group_col(t, keys[i].low))
                return -1;
            plan.projected = i;
            plan.key[i] = keys[i];
        } else if (!gpu_group_prepare_key(t, keys[i], plan.key[i])) {
            return -1;
        }
    }
    if ((projection != nullptr) != (plan.projected >= 0)) return -1;
    if (distinct && !gpu_group_prepare_key(t, *distinct, plan.distinct)) return -1;
    for (int i = 0; i < nsum; ++i) {
        if (!gpu_group_col(t, sum_cols[i])) return -1;
        plan.sums[i] = sum_cols[i];
    }
    if (!t.n) return 0;
    uint64_t active = gpu_group_active(t, filter);
    uint64_t capacity = pow2_at_least((active * 4 + 2) / 3);
    if (capacity > (UINT64_C(1) << 27)) return -1;
    uint32_t slots = (uint32_t)capacity;
    bool small = k <= GPU_GROUP_SMALL;
    uint32_t blocks = (slots + GPU_GROUP_TILE - 1) / GPU_GROUP_TILE;
    uint32_t next = (blocks * k + GPU_GROUP_TILE - 1) / GPU_GROUP_TILE;
    size_t first_size = small ? blocks * k : slots;
    size_t second_size = small ? (blocks > 1 ? next * k : 0) : slots;
    gpu_group_buffer<unsigned long long> owners, pairs;
    gpu_group_buffer<uint32_t> counts, uniques, lookup;
    gpu_group_buffer<unsigned int> invalid;
    gpu_group_buffer<uint64_t> scope, sums0, sums1, first, second;
    gpu_group_buffer<gpu_group_result_t> result;
    gpu_group_buffer<unsigned char> sort_scratch;
    if (!owners.alloc(slots) || !counts.alloc(slots) ||
        !pairs.alloc(distinct ? slots : 0) || !uniques.alloc(distinct ? slots : 0) ||
        !lookup.alloc(projection ? projection_len : 0) || !invalid.alloc(projection ? 1 : 0) ||
        !sums0.alloc(nsum > 0 ? slots : 0) || !sums1.alloc(nsum > 1 ? slots : 0) ||
        !scope.alloc(filter ? t.nw : 0) || !first.alloc(first_size) ||
        !second.alloc(second_size) || !result.alloc(k) ||
        !owners.clear(slots) || !counts.clear(slots) ||
        !pairs.clear(distinct ? slots : 0) || !uniques.clear(distinct ? slots : 0) ||
        !invalid.clear(projection ? 1 : 0) ||
        !sums0.clear(nsum > 0 ? slots : 0) || !sums1.clear(nsum > 1 ? slots : 0))
        return -2;
    if (filter && cudaMemcpy(scope.data, filter, t.nw * sizeof *filter, cudaMemcpyHostToDevice) != cudaSuccess)
        return -2;

    if (projection) {
        if (cudaMemcpy(lookup.data, projection, projection_len * sizeof *projection, cudaMemcpyHostToDevice) != cudaSuccess)
            return -2;
        plan.projection = lookup.data;
        plan.projection_len = projection_len;
        plan.projection_mask = projection_mask;
        plan.projection_width = projection_width;
    }
    unsigned grid = (unsigned)((t.n + 255) / 256);
    if (distinct) {
        gpu_group_accumulate<true><<<grid, 256>>>(
            h->dev, scope.data, plan, slots, owners.data, counts.data, sums0.data, sums1.data,
            pairs.data, uniques.data, invalid.data);
    } else {
        gpu_group_accumulate<false><<<grid, 256>>>(
            h->dev, scope.data, plan, slots, owners.data, counts.data, sums0.data, sums1.data,
            nullptr, nullptr, invalid.data);
    }
    if (cudaGetLastError() != cudaSuccess) return -2;
    const uint32_t *rank_counts = rank ? uniques.data : counts.data;
    uint64_t *in = first.data, *out = second.data;
    if (small) {
        gpu_group_select<true><<<blocks, GPU_GROUP_THREADS>>>(rank_counts, nullptr, slots, k, in);
        if (cudaGetLastError() != cudaSuccess) return -2;
        while (blocks > 1) {
            uint32_t n = blocks * k;
            blocks = (n + GPU_GROUP_TILE - 1) / GPU_GROUP_TILE;
            gpu_group_select<false><<<blocks, GPU_GROUP_THREADS>>>(nullptr, in, n, k, out);
            if (cudaGetLastError() != cudaSuccess) return -2;
            std::swap(in, out);
        }
    } else {
        gpu_group_scores<<<(slots + 255) / 256, 256>>>(rank_counts, slots, in);
        if (cudaGetLastError() != cudaSuccess) return -2;
        cub::DoubleBuffer<uint64_t> scores(in, out);
        size_t bytes = 0;
        if (cub::DeviceRadixSort::SortKeysDescending(nullptr, bytes, scores, slots) != cudaSuccess ||
            !sort_scratch.alloc(bytes) ||
            cub::DeviceRadixSort::SortKeysDescending(sort_scratch.data, bytes, scores, slots) != cudaSuccess)
            return -2;
        in = scores.Current();
    }
    gpu_group_gather<<<(k + 127) / 128, 128>>>(h->dev, plan, owners.data, in, counts.data, uniques.data, sums0.data, sums1.data, k, result.data);
    if (cudaGetLastError() != cudaSuccess) return -2;
    gpu_group_result_t host[GPU_GROUP_LIMIT];
    if (cudaMemcpy(host, result.data, k * sizeof *host, cudaMemcpyDeviceToHost) != cudaSuccess)
        return -2;
    if (projection) {
        unsigned int bad;
        if (cudaMemcpy(&bad, invalid.data, sizeof bad, cudaMemcpyDeviceToHost) != cudaSuccess) return -2;
        if (bad) return -1;
    }
    for (unsigned i = 0; i < k && host[i].count; ++i) {
        for (int j = 0; j < nkey; ++j) out_keys[i * nkey + j] = host[i].tuple.key[j];
        for (int j = 0; j < nsum; ++j) out_sums[i * nsum + j] = host[i].sums[j];
        out_counts[i] = host[i].count;
        if (distinct) out_distinct[i] = host[i].distinct;
        ++*out_n;
    }
    return 0;
}
extern "C" int gpu_group_planes(void *hv, const uint64_t *filter, int32_t nkey,
                    const gpu_group_key_t *keys, int32_t nsum, const int32_t *sum_cols,
                    uint32_t k, uint64_t *out_keys, uint64_t *out_counts,
                    uint64_t *out_sums, uint32_t *out_n) {
    return gpu_group_run(hv, filter, nkey, keys, nsum, sum_cols, k, out_keys, out_counts,
                         out_sums, out_n, nullptr, 0, nullptr, nullptr, 0, 0, 0);
}

extern "C" int gpu_group_distinct_planes(void *hv, const uint64_t *filter, int32_t nkey,
                    const gpu_group_key_t *keys, int32_t nsum, const int32_t *sum_cols,
                    uint32_t k, uint64_t *out_keys, uint64_t *out_counts, uint64_t *out_sums,
                    uint32_t *out_n, const gpu_group_key_t *distinct, uint32_t rank,
                    uint64_t *out_distinct, const uint32_t *projection, uint64_t projection_len,
                    uint64_t projection_mask, uint32_t projection_width) {
    return gpu_group_run(hv, filter, nkey, keys, nsum, sum_cols, k, out_keys, out_counts,
                         out_sums, out_n, distinct, rank, out_distinct, projection,
                         projection_len, projection_mask, projection_width);
}

__global__ void gpu_distinct_bits(const gtable_t *table, const uint64_t *scope,
                                   int col, unsigned long long *bits) {
    uint64_t row = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= table->n || (scope && !((scope[row >> 6] >> (row & 63)) & 1))) return;
    uint32_t code = gpu_group_code(table, col, (uint32_t)row);
#if __CUDA_ARCH__ >= 700
    unsigned peers = __match_any_sync(__activemask(), code);
    if ((threadIdx.x & 31) != (unsigned)(__ffs(peers) - 1)) return;
#endif
    atomicOr(bits + (code >> 6), 1ULL << (code & 63));
}

template <bool Popcount>
__global__ void gpu_distinct_partials(const unsigned long long *cells, uint32_t n, uint64_t *out) {
    unsigned thread = threadIdx.x, lane = thread & 31;
    uint32_t start = (blockIdx.x * 4) * blockDim.x + thread;
    uint32_t total = 0;
    for (unsigned item = 0; item < 4; ++item) {
        uint32_t i = start + item * blockDim.x;
        if (i < n) total += Popcount ? __popcll(cells[i]) : cells[i] != 0;
    }
    for (int shift = 16; shift; shift >>= 1)
        total += __shfl_down_sync(0xffffffffU, total, shift);
    __shared__ uint32_t warp[8];
    if (!lane) warp[thread >> 5] = total;
    __syncthreads();
    if (!thread) {
        uint64_t sum = 0;
        for (unsigned w = 0; w < 8; ++w) sum += warp[w];
        out[blockIdx.x] = sum;
    }
}

template <bool Popcount>
static int gpu_distinct_reduce(const unsigned long long *cells, uint32_t n, uint64_t *out) {
    uint32_t blocks = (n + 1023) / 1024;
    gpu_group_buffer<uint64_t> partial, result;
    if (!partial.alloc(blocks)) return -2;
    gpu_distinct_partials<Popcount><<<blocks, 256>>>(cells, n, partial.data);
    if (cudaGetLastError() != cudaSuccess) return -2;
    if (blocks == 1)
        return cudaMemcpy(out, partial.data, sizeof *out, cudaMemcpyDeviceToHost) == cudaSuccess ? 0 : -2;
    gpu_group_buffer<unsigned char> scratch;
    size_t bytes = 0;
    if (!result.alloc(1) || cub::DeviceReduce::Sum(nullptr, bytes, partial.data, result.data, blocks) != cudaSuccess ||
        !scratch.alloc(bytes) || cub::DeviceReduce::Sum(scratch.data, bytes, partial.data, result.data, blocks) != cudaSuccess ||
        cudaMemcpy(out, result.data, sizeof *out, cudaMemcpyDeviceToHost) != cudaSuccess)
        return -2;
    return 0;
}

extern "C" int gpu_distinct_plane_count(void *hv, const uint64_t *filter,
                                         const gpu_group_key_t *key, uint64_t *out) {
    if (!hv || !key || !out) return -1;
    handle_t *h = (handle_t *)hv;
    const gtable_t &t = h->host;
    gpu_group_plan_t plan{};
    plan.nkey = 1;
    plan.projected = -1;
    if (t.n > UINT32_MAX || !gpu_group_prepare_key(t, *key, plan.key[0])) return -1;
    if (!t.n) {
        *out = 0;
        return 0;
    }

    bool dense = !key->transform && key->high == -1 && t.width[key->low] <= 22;
    uint64_t capacity = dense ? ((UINT64_C(1) << t.width[key->low]) + 63) / 64
                              : pow2_at_least((gpu_group_active(t, filter) * 4 + 2) / 3);
    if (capacity > (UINT64_C(1) << 27)) return -1;
    gpu_group_buffer<unsigned long long> cells;
    gpu_group_buffer<uint64_t> scope;
    if (!cells.alloc(capacity) || !cells.clear(capacity) || !scope.alloc(filter ? t.nw : 0) ||
        (filter && cudaMemcpy(scope.data, filter, t.nw * sizeof *filter, cudaMemcpyHostToDevice) != cudaSuccess))
        return -2;
    unsigned blocks = (unsigned)((t.n + 255) / 256);
    if (dense) {
        gpu_distinct_bits<<<blocks, 256>>>(h->dev, scope.data, key->low, cells.data);
        if (cudaGetLastError() != cudaSuccess) return -2;
        return gpu_distinct_reduce<true>(cells.data, (uint32_t)capacity, out);
    }
    gpu_group_accumulate<false><<<blocks, 256>>>(
        h->dev, scope.data, plan, (uint32_t)capacity, cells.data, nullptr, nullptr, nullptr,
        nullptr, nullptr, nullptr);
    if (cudaGetLastError() != cudaSuccess) return -2;
    return gpu_distinct_reduce<false>(cells.data, (uint32_t)capacity, out);
}
#endif

