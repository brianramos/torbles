/* Copyright 2026 Brian Richard RAMOS
 * SPDX-License-Identifier: Apache-2.0 */
#pragma once

#ifndef EMULATE
#include <cub/block/block_radix_sort.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_reduce.cuh>
#include <functional>
#include <memory>

enum { GPU_GROUP_KEYS = 5, GPU_GROUP_SUMS = 2, GPU_GROUP_LIMIT = 1024, GPU_GROUP_SMALL = 16,
       GPU_GROUP_THREADS = 128, GPU_GROUP_ITEMS = 4,
       GPU_GROUP_TILE = GPU_GROUP_THREADS * GPU_GROUP_ITEMS };

struct gpu_group_key_t {
    int32_t low, high, shift, transform;
    int64_t offset;
};

struct gpu_group_sectors_t {
    const uint32_t *words;
    uint32_t shift, bits, stride;
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
    gpu_group_sectors_t sectors;
    const unsigned char *winners;
    unsigned dense_bits;
    unsigned dense_width[GPU_GROUP_KEYS];
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
    if (key.transform == 4) {
        uint64_t word = row >> 6;
        unsigned lane = row & 63;
        uint32_t field = 0;
        int width = table->width[key.low];
        for (int bit = 0; bit < key.offset; ++bit) {
            int significance = key.shift + bit;
            int plane = table->msb[key.low] ? width - 1 - significance : significance;
            field |= (uint32_t)((table->P[key.low][plane][word] >> lane) & 1) << bit;
        }
        return field;
    }
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

DEV uint64_t gpu_group_project(const gpu_group_plan_t &plan, uint32_t raw) {
    uint64_t index = raw & plan.projection_mask;
    if (index >= plan.projection_len) return UINT64_MAX;
    uint64_t value;
    if (plan.sectors.words) {
        uint32_t mask = (UINT32_C(1) << plan.sectors.bits) - 1;
        uint32_t sector = (raw >> plan.sectors.shift) & mask;
        const uint32_t *formula = plan.sectors.words + (size_t)sector * plan.sectors.stride;
        if (formula[0] == 1) {
            uint32_t code = formula[1];
            for (unsigned bit = 0; bit < plan.projection_width; ++bit)
                code ^= (uint32_t)(popc32(raw & formula[2 + bit]) & 1) << bit;
            value = code;
        } else {
            value = plan.projection[index];
        }
    } else {
        value = plan.projection[index];
    }
    return value < (UINT64_C(1) << plan.projection_width) ? value : UINT64_MAX;
}

DEV gpu_group_tuple_t gpu_group_key(const gtable_t *table, const gpu_group_plan_t &plan, uint32_t row) {
    gpu_group_tuple_t tuple{};
    for (int i = 0; i < plan.nkey; ++i) {
        uint64_t value = gpu_group_value(table, plan.key[i], row);
        if (plan.key[i].transform == 3)
            value = gpu_group_project(plan, (uint32_t)value);
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

DEV uint32_t gpu_group_dense_slot(gpu_group_tuple_t tuple, const gpu_group_plan_t &plan) {
    uint32_t slot = 0;
    for (int i = 0; i < plan.nkey; ++i)
        slot = (slot << plan.dense_width[i]) | (uint32_t)tuple.key[i];
    return slot;
}

__global__ void gpu_group_dense_counts(const gtable_t *table, const uint64_t *scope,
                                       gpu_group_plan_t plan, uint32_t *counts) {
    uint64_t row = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    bool eligible = row < table->n && (!scope || ((scope[row >> 6] >> (row & 63)) & 1));
#if __CUDA_ARCH__ >= 700
    unsigned active = __ballot_sync(0xffffffffU, eligible);
#endif
    if (!eligible) return;
    uint32_t slot = gpu_group_dense_slot(gpu_group_key(table, plan, (uint32_t)row), plan);
    unsigned weight = 1;
#if __CUDA_ARCH__ >= 700
    unsigned peers = __match_any_sync(active, slot);
    if ((threadIdx.x & 31) != (unsigned)(__ffs(peers) - 1)) return;
    weight = popc32(peers);
#endif
    atomicAdd(counts + slot, weight);
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
    if (plan.winners && !plan.winners[tuple.key[0]]) return;
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
    uint32_t pos;
    if (plan.dense_bits) {
        pos = gpu_group_dense_slot(tuple, plan);
    } else {
        uint64_t hash = gpu_group_hash(tuple, plan.nkey);
        pos = (uint32_t)hash & (slots - 1);
        uint64_t tag = hash & UINT64_C(0xffffffff00000000);
        uint64_t token = tag | (row + 1);
        for (;;) {
            uint64_t owner = atomicCAS(owners + pos, 0ULL, (unsigned long long)token);
            if (!owner || ((owner & UINT64_C(0xffffffff00000000)) == tag &&
                           gpu_group_equal(tuple, gpu_group_key(table, plan, (uint32_t)owner - 1), plan.nkey)))
                break;
            pos = (pos + 1) & (slots - 1);
        }
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
        if (plan.dense_bits) {
            uint32_t code = slot;
            for (int i = plan.nkey - 1; i >= 0; --i) {
                unsigned width = plan.dense_width[i];
                entry.tuple.key[i] = code & ((UINT32_C(1) << width) - 1);
                code >>= width;
            }
        } else {
            entry.tuple = gpu_group_key(table, plan, (uint32_t)owners[slot] - 1);
        }
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
    if (!gpu_group_col(t, key.low) || key.high < -1) return false;
    if (key.transform == 4) {
        int width = t.width[key.low];
        if (key.high != -1 || key.shift < 0 || key.shift >= width ||
            key.offset < 1 || key.offset > width - key.shift) return false;
        out = key;
        return true;
    }
    if (key.transform < 0 || key.transform > 1) return false;
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

__global__ void gpu_coordinate_counts(const gtable_t *table, const uint64_t *scope,
                                      gpu_group_key_t key, uint32_t *histogram) {
    uint64_t row = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    bool eligible = row < table->n && (!scope || ((scope[row >> 6] >> (row & 63)) & 1));
#if __CUDA_ARCH__ >= 700
    unsigned active = __ballot_sync(0xffffffffU, eligible);
#endif
    if (!eligible) return;
    uint32_t coordinate = (uint32_t)gpu_group_value(table, key, (uint32_t)row);
    unsigned weight = 1;
#if __CUDA_ARCH__ >= 700
    unsigned peers = __match_any_sync(active, coordinate);
    if ((threadIdx.x & 31) != (unsigned)(__ffs(peers) - 1)) return;
    weight = popc32(peers);
#endif
    atomicAdd(histogram + coordinate, weight);
}

/* COUNT fixes its winners directly. COUNT(DISTINCT) can use row counts
 * as upper bounds, subject to an exact certificate after the second pass. */
static uint64_t gpu_coordinate_winners(handle_t *h, const uint64_t *scope, gpu_group_key_t key,
                         uint32_t k, bool distinct_rank, uint64_t active, uint64_t full_capacity,
                         gpu_group_buffer<unsigned char> &device_winners, uint32_t &excluded_bound) {
    uint32_t domain = UINT32_C(1) << key.offset;
    gpu_group_buffer<uint32_t> histogram;
    std::unique_ptr<uint32_t, decltype(&free)> counts((uint32_t *)malloc(domain * sizeof(uint32_t)), free);
    std::unique_ptr<uint64_t, decltype(&free)> scores((uint64_t *)malloc(domain * sizeof(uint64_t)), free);
    if (!counts || !scores || !histogram.alloc(domain) || !histogram.clear(domain)) return 0;
    gpu_coordinate_counts<<<(unsigned)((h->host.n + 255) / 256), 256>>>(
        h->dev, scope, key, histogram.data);
    if (cudaGetLastError() != cudaSuccess ||
        cudaMemcpy(counts.get(), histogram.data, domain * sizeof(uint32_t), cudaMemcpyDeviceToHost) != cudaSuccess)
        return 0;
    uint32_t occupied = 0;
    for (uint32_t code = 0; code < domain; ++code)
        if (counts.get()[code]) scores.get()[occupied++] = ((uint64_t)counts.get()[code] << 32) | (UINT32_MAX - code);
    if (occupied <= k) return 0;
    uint32_t limit = distinct_rank ? std::min(occupied - 1, k * 8) : k;
    std::partial_sort(scores.get(), scores.get() + limit + 1, scores.get() + occupied, std::greater<uint64_t>{});
    uint64_t mass = 0;
    uint32_t keep = 0;
    for (; keep < limit; ++keep) {
        uint64_t next = mass + (scores.get()[keep] >> 32);
        // An unproven DISTINCT pass must save enough work to justify a retry.
        bool too_many_rows = distinct_rank ? next * 8 > active * 3 : next * 4 > active * 3;
        if (too_many_rows || pow2_at_least((next * 4 + 2) / 3) > full_capacity / 2) break;
        mass = next;
    }
    if (keep < k) return 0;

    std::unique_ptr<unsigned char, decltype(&free)> winners((unsigned char *)calloc(domain, 1), free);
    if (!winners || !device_winners.alloc(domain)) return 0;
    for (uint32_t i = 0; i < keep; ++i) winners.get()[UINT32_MAX - (uint32_t)scores.get()[i]] = 1;
    excluded_bound = (uint32_t)(scores.get()[keep] >> 32);
    if (cudaMemcpy(device_winners.data, winners.get(), domain, cudaMemcpyHostToDevice) != cudaSuccess) return 0;
    return mass;
}

enum { GPU_GROUP_UNCERTIFIED = -3 };

static int gpu_group_run_once(void *hv, const uint64_t *filter, int32_t nkey,
                    const gpu_group_key_t *keys, int32_t nsum, const int32_t *sum_cols,
                    uint32_t k, uint64_t *out_keys, uint64_t *out_counts,
                    uint64_t *out_sums, uint32_t *out_n, const gpu_group_key_t *distinct,
                    uint32_t rank, uint64_t *out_distinct, const uint32_t *projection,
                    uint64_t projection_len, uint64_t projection_mask, uint32_t projection_width,
                    gpu_group_sectors_t sectors, uint64_t *selected_rows, bool try_pruning,
                    uint32_t *dense_slots, const uint64_t *resident_scope = nullptr,
                    uint64_t resident_active = 0) {
    if (dense_slots) *dense_slots = 0;
    if (selected_rows) *selected_rows = 0;
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
    size_t sector_words = 0;
    if (sectors.words) {
        if (!projection || sectors.shift >= 32 || sectors.bits > 16 ||
            sectors.bits > 32 - sectors.shift || sectors.stride != 2 + projection_width)
            return -1;
        uint32_t nsectors = UINT32_C(1) << sectors.bits;
        sector_words = (size_t)nsectors * sectors.stride;
        for (uint32_t i = 0; i < nsectors; ++i) {
            const uint32_t *formula = sectors.words + (size_t)i * sectors.stride;
            if (formula[0] > 1 || (formula[0] && formula[1] >= (UINT64_C(1) << projection_width)))
                return -1;
        }
    } else if (sectors.shift || sectors.bits || sectors.stride) {
        return -1;
    }
    if (distinct && !gpu_group_prepare_key(t, *distinct, plan.distinct)) return -1;
    for (int i = 0; i < nsum; ++i) {
        if (!gpu_group_col(t, sum_cols[i])) return -1;
        plan.sums[i] = sum_cols[i];
    }
    if (!t.n) return 0;
    uint64_t active = resident_scope ? resident_active : gpu_group_active(t, filter);
    uint64_t capacity = pow2_at_least((active * 4 + 2) / 3);
    gpu_group_buffer<uint64_t> scope;
    if (!resident_scope && (!scope.alloc(filter ? t.nw : 0) ||
        (filter && cudaMemcpy(scope.data, filter, t.nw * sizeof *filter, cudaMemcpyHostToDevice) != cudaSuccess)))
        return -2;
    const uint64_t *scope_data = resident_scope ? resident_scope : scope.data;
    gpu_group_buffer<unsigned char> device_winners;
    uint64_t winner_mass = 0;
    uint32_t excluded_bound = 0;
    if (selected_rows && distinct) {
        unsigned bits = 0;
        bool coordinate = false;
        for (int i = 0; i < nkey; ++i) {
            const gpu_group_key_t &key = plan.key[i];
            if (key.high >= 0 || (key.transform != 0 && key.transform != 4)) {
                bits = 0;
                break;
            }
            coordinate |= key.transform == 4;
            plan.dense_width[i] = key.transform == 4 ? (unsigned)key.offset : (unsigned)t.width[key.low];
            bits += plan.dense_width[i];
        }
        if (coordinate && bits <= 17) plan.dense_bits = bits;
    }
    if (dense_slots && !distinct) {
        unsigned bits = 0;
        for (int i = 0; i < nkey; ++i) {
            const gpu_group_key_t &key = plan.key[i];
            unsigned width;
            if (key.transform == 1) width = 6;
            else if (key.transform == 4) width = (unsigned)key.offset;
            else if (key.transform == 0 && key.high < 0) width = (unsigned)t.width[key.low];
            else { bits = 0; break; }
            plan.dense_width[i] = width;
            bits += width;
        }
        if (bits && bits <= 22) {
            uint32_t domain = UINT32_C(1) << bits;
            uint64_t state_bytes = (uint64_t)domain * (sizeof(uint32_t) + nsum * sizeof(uint64_t));
            if (domain < capacity / 2 && state_bytes <= (UINT64_C(64) << 20))
                plan.dense_bits = bits;
        }
    }
    if (distinct && plan.dense_bits && nkey == 1 && try_pruning && active > k) {
        winner_mass = gpu_coordinate_winners(h, scope_data, plan.key[0], k, rank == 1, active, capacity,
                                             device_winners, excluded_bound);
        if (winner_mass) {
            capacity = pow2_at_least((winner_mass * 4 + 2) / 3);
            plan.winners = device_winners.data;
        }
    }
    if (!distinct && plan.dense_bits) capacity = 1024;  /* no hash storage */
    if (capacity > (UINT64_C(1) << 27)) return -1;
    uint32_t slots = (uint32_t)capacity;
    uint32_t group_slots = plan.dense_bits ? std::max(k, UINT32_C(1) << plan.dense_bits) : slots;
    uint32_t owner_slots = plan.dense_bits ? 0 : slots;
    bool small = k <= GPU_GROUP_SMALL;
    uint32_t blocks = (group_slots + GPU_GROUP_TILE - 1) / GPU_GROUP_TILE;
    uint32_t next = (blocks * k + GPU_GROUP_TILE - 1) / GPU_GROUP_TILE;
    size_t first_size = small ? blocks * k : group_slots;
    size_t second_size = small ? (blocks > 1 ? next * k : 0) : group_slots;
    gpu_group_buffer<unsigned long long> owners, pairs;
    gpu_group_buffer<uint32_t> counts, uniques, lookup, sector_data;
    gpu_group_buffer<unsigned int> invalid;
    gpu_group_buffer<uint64_t> sums0, sums1, first, second;
    gpu_group_buffer<gpu_group_result_t> result;
    gpu_group_buffer<unsigned char> sort_scratch;
    if (!owners.alloc(owner_slots) || !counts.alloc(group_slots) ||
        !pairs.alloc(distinct ? slots : 0) || !uniques.alloc(distinct ? group_slots : 0) ||
        !lookup.alloc(projection ? projection_len : 0) || !invalid.alloc(projection ? 1 : 0) ||
        !sector_data.alloc(sector_words) ||
        !sums0.alloc(nsum > 0 ? group_slots : 0) || !sums1.alloc(nsum > 1 ? group_slots : 0) ||
        !first.alloc(first_size) ||
        !second.alloc(second_size) || !result.alloc(k) ||
        !owners.clear(owner_slots) || !counts.clear(group_slots) ||
        !pairs.clear(distinct ? slots : 0) || !uniques.clear(distinct ? group_slots : 0) ||
        !invalid.clear(projection ? 1 : 0) ||
        !sums0.clear(nsum > 0 ? group_slots : 0) || !sums1.clear(nsum > 1 ? group_slots : 0))
        return -2;
    if (projection) {
        if (cudaMemcpy(lookup.data, projection, projection_len * sizeof *projection, cudaMemcpyHostToDevice) != cudaSuccess)
            return -2;
        plan.projection = lookup.data;
        plan.projection_len = projection_len;
        plan.projection_mask = projection_mask;
        plan.projection_width = projection_width;
    }
    if (sector_words) {
        if (cudaMemcpy(sector_data.data, sectors.words, sector_words * sizeof *sectors.words,
                       cudaMemcpyHostToDevice) != cudaSuccess)
            return -2;
        plan.sectors = sectors;
        plan.sectors.words = sector_data.data;
    }
    unsigned grid = (unsigned)((t.n + 255) / 256);
    if (plan.dense_bits && !distinct && !nsum) {
        gpu_group_dense_counts<<<grid, 256>>>(h->dev, scope_data, plan, counts.data);
    } else if (distinct) {
        gpu_group_accumulate<true><<<grid, 256>>>(
            h->dev, scope_data, plan, slots, owners.data, counts.data, sums0.data, sums1.data,
            pairs.data, uniques.data, invalid.data);
    } else {
        gpu_group_accumulate<false><<<grid, 256>>>(
            h->dev, scope_data, plan, slots, owners.data, counts.data, sums0.data, sums1.data,
            nullptr, nullptr, invalid.data);
    }
    if (cudaGetLastError() != cudaSuccess) return -2;
    const uint32_t *rank_counts = rank ? uniques.data : counts.data;
    uint64_t *in = first.data, *out = second.data;
    if (small) {
        gpu_group_select<true><<<blocks, GPU_GROUP_THREADS>>>(rank_counts, nullptr, group_slots, k, in);
        if (cudaGetLastError() != cudaSuccess) return -2;
        while (blocks > 1) {
            uint32_t n = blocks * k;
            blocks = (n + GPU_GROUP_TILE - 1) / GPU_GROUP_TILE;
            gpu_group_select<false><<<blocks, GPU_GROUP_THREADS>>>(nullptr, in, n, k, out);
            if (cudaGetLastError() != cudaSuccess) return -2;
            std::swap(in, out);
        }
    } else {
        gpu_group_scores<<<(group_slots + 255) / 256, 256>>>(rank_counts, group_slots, in);
        if (cudaGetLastError() != cudaSuccess) return -2;
        cub::DoubleBuffer<uint64_t> scores(in, out);
        size_t bytes = 0;
        if (cub::DeviceRadixSort::SortKeysDescending(nullptr, bytes, scores, group_slots) != cudaSuccess ||
            !sort_scratch.alloc(bytes) ||
            cub::DeviceRadixSort::SortKeysDescending(sort_scratch.data, bytes, scores, group_slots) != cudaSuccess)
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
    if (winner_mass && rank == 1 && host[k - 1].distinct <= excluded_bound)
        return GPU_GROUP_UNCERTIFIED;
    for (unsigned i = 0; i < k && host[i].count; ++i) {
        for (int j = 0; j < nkey; ++j) out_keys[i * nkey + j] = host[i].tuple.key[j];
        for (int j = 0; j < nsum; ++j) out_sums[i * nsum + j] = host[i].sums[j];
        out_counts[i] = host[i].count;
        if (distinct) out_distinct[i] = host[i].distinct;
        ++*out_n;
    }
    if (selected_rows) *selected_rows = winner_mass;
    if (dense_slots && plan.dense_bits) *dense_slots = group_slots;
    return 0;
}
static int gpu_group_run(void *hv, const uint64_t *filter, int32_t nkey,
                    const gpu_group_key_t *keys, int32_t nsum, const int32_t *sum_cols,
                    uint32_t k, uint64_t *out_keys, uint64_t *out_counts,
                    uint64_t *out_sums, uint32_t *out_n, const gpu_group_key_t *distinct,
                    uint32_t rank, uint64_t *out_distinct, const uint32_t *projection,
                    uint64_t projection_len, uint64_t projection_mask, uint32_t projection_width,
                    gpu_group_sectors_t sectors, uint64_t *selected_rows, uint32_t *dense_slots = nullptr) {
    int rc = gpu_group_run_once(hv, filter, nkey, keys, nsum, sum_cols, k, out_keys, out_counts,
                               out_sums, out_n, distinct, rank, out_distinct, projection,
                               projection_len, projection_mask, projection_width, sectors, selected_rows, true, dense_slots);
    if (rc != GPU_GROUP_UNCERTIFIED) return rc;
    return gpu_group_run_once(hv, filter, nkey, keys, nsum, sum_cols, k, out_keys, out_counts,
                              out_sums, out_n, distinct, rank, out_distinct, projection,
                              projection_len, projection_mask, projection_width, sectors, selected_rows, false, dense_slots);
}

extern "C" int gpu_group_planes(void *hv, const uint64_t *filter, int32_t nkey,
                    const gpu_group_key_t *keys, int32_t nsum, const int32_t *sum_cols,
                    uint32_t k, uint64_t *out_keys, uint64_t *out_counts,
                    uint64_t *out_sums, uint32_t *out_n) {
    return gpu_group_run(hv, filter, nkey, keys, nsum, sum_cols, k, out_keys, out_counts,
                         out_sums, out_n, nullptr, 0, nullptr, nullptr, 0, 0, 0, {}, nullptr);
}

extern "C" int gpu_group_dense_planes(void *hv, const uint64_t *filter, int32_t nkey,
                    const gpu_group_key_t *keys, int32_t nsum, const int32_t *sum_cols,
                    uint32_t k, uint64_t *out_keys, uint64_t *out_counts, uint64_t *out_sums,
                    uint32_t *out_n, uint32_t *dense_slots) {
    return gpu_group_run(hv, filter, nkey, keys, nsum, sum_cols, k, out_keys, out_counts,
                         out_sums, out_n, nullptr, 0, nullptr, nullptr, 0, 0, 0, {}, nullptr, dense_slots);
}

extern "C" int gpu_group_distinct_planes(void *hv, const uint64_t *filter, int32_t nkey,
                    const gpu_group_key_t *keys, int32_t nsum, const int32_t *sum_cols,
                    uint32_t k, uint64_t *out_keys, uint64_t *out_counts, uint64_t *out_sums,
                    uint32_t *out_n, const gpu_group_key_t *distinct, uint32_t rank,
                    uint64_t *out_distinct, const uint32_t *projection, uint64_t projection_len,
                    uint64_t projection_mask, uint32_t projection_width) {
    return gpu_group_run(hv, filter, nkey, keys, nsum, sum_cols, k, out_keys, out_counts,
                         out_sums, out_n, distinct, rank, out_distinct, projection,
                         projection_len, projection_mask, projection_width, {}, nullptr);
}

extern "C" int gpu_group_distinct_sector_planes(void *hv, const uint64_t *filter, int32_t nkey,
                    const gpu_group_key_t *keys, int32_t nsum, const int32_t *sum_cols,
                    uint32_t k, uint64_t *out_keys, uint64_t *out_counts, uint64_t *out_sums,
                    uint32_t *out_n, const gpu_group_key_t *distinct, uint32_t rank,
                    uint64_t *out_distinct, const uint32_t *projection, uint64_t projection_len,
                    uint64_t projection_mask, uint32_t projection_width, const uint32_t *sectors,
                    uint32_t sector_shift, uint32_t sector_bits, uint32_t stride, uint64_t *selected_rows) {
    return gpu_group_run(hv, filter, nkey, keys, nsum, sum_cols, k, out_keys, out_counts,
                         out_sums, out_n, distinct, rank, out_distinct, projection,
                         projection_len, projection_mask, projection_width,
                         {sectors, sector_shift, sector_bits, stride}, selected_rows);
}

struct gpu_group_collision_t {
    unsigned long long *seen[2], *collided[2];
    uint32_t mask;
};

DEV uint32_t gpu_collision_bin(uint64_t hash, uint32_t mask, int i) {
    return (uint32_t)(i ? gpu_group_mix(hash ^ UINT64_C(0xe7037ed1a0b428db)) : hash) & mask;
}

__global__ void gpu_collision_mark(const gtable_t *table, const uint64_t *scope,
                                  gpu_group_plan_t plan, gpu_group_collision_t sketch) {
    uint64_t row = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= table->n || (scope && !((scope[row >> 6] >> (row & 63)) & 1))) return;
    uint64_t hash = gpu_group_hash(gpu_group_key(table, plan, (uint32_t)row), plan.nkey);
    for (int i = 0; i < 2; ++i) {
        uint32_t bin = gpu_collision_bin(hash, sketch.mask, i);
        unsigned long long bit = 1ULL << (bin & 63);
        if (atomicOr(sketch.seen[i] + (bin >> 6), bit) & bit)
            atomicOr(sketch.collided[i] + (bin >> 6), bit);
    }
}

struct gpu_collision_summary_t {
    uint32_t ambiguous, singleton;
    uint32_t rows[GPU_GROUP_SMALL];
};

__global__ void gpu_collision_scope(const gtable_t *table, const uint64_t *scope,
                    gpu_group_plan_t plan, gpu_group_collision_t sketch, uint32_t k,
                    uint32_t *ambiguous_scope, gpu_collision_summary_t *summary) {
    uint64_t row = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned lane = threadIdx.x & 31;
    bool live = row < table->n && (!scope || ((scope[row >> 6] >> (row & 63)) & 1));
    bool ambiguous = false;
    if (live) {
        uint64_t hash = gpu_group_hash(gpu_group_key(table, plan, (uint32_t)row), plan.nkey);
        ambiguous = true;
        for (int i = 0; i < 2 && ambiguous; ++i) {
            uint32_t bin = gpu_collision_bin(hash, sketch.mask, i);
            ambiguous = (sketch.collided[i][bin >> 6] >> (bin & 63)) & 1;
        }
    }
    uint32_t candidates = __ballot_sync(0xffffffffU, ambiguous);
    uint32_t singletons = __ballot_sync(0xffffffffU, live && !ambiguous);
    if (lane || row >= table->n) return;
    ambiguous_scope[row >> 5] = candidates;
    if (candidates) atomicAdd(&summary->ambiguous, (uint32_t)popc32(candidates));
    if (singletons && atomicAdd(&summary->singleton, 0U) < k) {
        uint32_t at = atomicAdd(&summary->singleton, (uint32_t)popc32(singletons));
        for (; singletons && at < k; ++at) {
            summary->rows[at] = (uint32_t)row + ctz32(singletons);
            singletons &= singletons - 1;
        }
    }
}

__global__ void gpu_collision_singletons(const gtable_t *table, gpu_group_plan_t plan,
                  const gpu_collision_summary_t *summary, unsigned start, unsigned count,
                  gpu_group_result_t *out) {
    unsigned at = threadIdx.x;
    if (at >= count) return;
    uint32_t row = summary->rows[start + at];
    gpu_group_result_t result{};
    result.tuple = gpu_group_key(table, plan, row);
    result.count = 1;
    for (int j = 0; j < plan.nsum; ++j) result.sums[j] = gpu_group_code(table, plan.sums[j], row);
    out[at] = result;
}

/* A repeated tuple has at least two arrivals at each of its two bins, so
 * every occurrence survives the intersection. Other candidates still use
 * full key equality. Rows outside the intersection are proven singletons. */
static int gpu_group_collision_try(handle_t *h, const uint64_t *filter, int32_t nkey,
                    const gpu_group_key_t *keys, int32_t nsum, const int32_t *sum_cols,
                    uint32_t k, uint64_t *out_keys, uint64_t *out_counts, uint64_t *out_sums,
                    uint32_t *out_n, uint64_t *collision_rows) {
    const gtable_t &t = h->host;
    if (t.n > UINT32_MAX) return -1;
    uint64_t active = gpu_group_active(t, filter);
    if (active < 65536 || active > (UINT64_C(1) << 25)) return -1;
    gpu_group_plan_t plan{};
    plan.nkey = nkey;
    plan.nsum = nsum;
    plan.projected = -1;
    for (int i = 0; i < nkey; ++i)
        if (!gpu_group_prepare_key(t, keys[i], plan.key[i])) return -1;
    for (int i = 0; i < nsum; ++i) {
        if (!gpu_group_col(t, sum_cols[i])) return -1;
        plan.sums[i] = sum_cols[i];
    }
    uint32_t bins = (uint32_t)pow2_at_least(active * 4), words = bins / 64;
    gpu_group_buffer<unsigned long long> flags;
    gpu_group_buffer<uint64_t> scope, candidate;
    gpu_group_buffer<gpu_collision_summary_t> summary;
    gpu_group_buffer<gpu_group_result_t> device_rows;
    if (!flags.alloc((size_t)words * 4) || !flags.clear((size_t)words * 4) ||
        !scope.alloc(filter ? t.nw : 0) || !candidate.alloc(t.nw) || !candidate.clear(t.nw) ||
        !summary.alloc(1) || !summary.clear(1) ||
        (filter && cudaMemcpy(scope.data, filter, t.nw * sizeof *filter, cudaMemcpyHostToDevice) != cudaSuccess))
        return -2;
    gpu_group_collision_t sketch = {
        {flags.data, flags.data + words}, {flags.data + 2 * words, flags.data + 3 * words}, bins - 1
    };
    unsigned blocks = (unsigned)((t.n + 255) / 256);
    gpu_collision_mark<<<blocks, 256>>>(h->dev, scope.data, plan, sketch);
    if (cudaGetLastError() != cudaSuccess) return -2;
    gpu_collision_scope<<<blocks, 256>>>(h->dev, scope.data, plan, sketch, k,
                                      (uint32_t *)candidate.data, summary.data);
    gpu_collision_summary_t sizes;
    if (cudaGetLastError() != cudaSuccess ||
        cudaMemcpy(&sizes, summary.data, sizeof sizes, cudaMemcpyDeviceToHost) != cudaSuccess)
        return -2;
    if ((uint64_t)sizes.ambiguous * 2 > active) return -1;

    uint64_t staged_keys[GPU_GROUP_SMALL * GPU_GROUP_KEYS], staged_counts[GPU_GROUP_SMALL];
    uint64_t staged_sums[GPU_GROUP_SMALL * GPU_GROUP_SUMS];
    uint32_t found = 0;
    if (sizes.ambiguous) {
        int rc = gpu_group_run_once(h, nullptr, nkey, keys, nsum, sum_cols, k,
                                   staged_keys, staged_counts, staged_sums, &found,
                                   nullptr, 0, nullptr, nullptr, 0, 0, 0, {}, nullptr, false,
                                   nullptr, candidate.data, sizes.ambiguous);
        if (rc) return rc;
    }
    unsigned singles = std::min(k - found, sizes.singleton);
    if (singles) {
        if (!device_rows.alloc(singles)) return -2;
        gpu_collision_singletons<<<1, GPU_GROUP_SMALL>>>(h->dev, plan, summary.data, 0, singles, device_rows.data);
        gpu_group_result_t host[GPU_GROUP_SMALL];
        if (cudaGetLastError() != cudaSuccess ||
            cudaMemcpy(host, device_rows.data, singles * sizeof *host, cudaMemcpyDeviceToHost) != cudaSuccess)
            return -2;
        for (unsigned i = 0; i < singles; ++i) {
            unsigned j = found + i;
            for (int c = 0; c < nkey; ++c) staged_keys[j * nkey + c] = host[i].tuple.key[c];
            for (int m = 0; m < nsum; ++m) staged_sums[j * nsum + m] = host[i].sums[m];
            staged_counts[j] = 1;
        }
    }
    found += singles;
    memcpy(out_keys, staged_keys, found * nkey * sizeof *out_keys);
    memcpy(out_counts, staged_counts, found * sizeof *out_counts);
    if (nsum) memcpy(out_sums, staged_sums, found * nsum * sizeof *out_sums);
    *out_n = found;
    *collision_rows = sizes.ambiguous;
    return 0;
}

/* Opt-in experiment. UINT64_MAX reports that the ordinary path was used. */
extern "C" int gpu_group_collision_planes(void *hv, const uint64_t *filter, int32_t nkey,
                    const gpu_group_key_t *keys, int32_t nsum, const int32_t *sum_cols,
                    uint32_t k, uint64_t *out_keys, uint64_t *out_counts, uint64_t *out_sums,
                    uint32_t *out_n, uint64_t *collision_rows) {
    if (!collision_rows) return -1;
    *collision_rows = UINT64_MAX;
    if (hv && nkey >= 1 && nkey <= GPU_GROUP_KEYS && keys && nsum >= 0 && nsum <= GPU_GROUP_SUMS &&
        (!nsum || (sum_cols && out_sums)) && k && k <= GPU_GROUP_SMALL && out_keys && out_counts && out_n) {
        int rc = gpu_group_collision_try((handle_t *)hv, filter, nkey, keys, nsum, sum_cols, k,
                                         out_keys, out_counts, out_sums, out_n, collision_rows);
        if (!rc) return 0;
    }
    return gpu_group_planes(hv, filter, nkey, keys, nsum, sum_cols, k, out_keys, out_counts, out_sums, out_n);
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

