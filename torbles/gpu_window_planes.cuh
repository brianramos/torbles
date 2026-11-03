/* Copyright 2026 Brian Richard RAMOS
 * SPDX-License-Identifier: Apache-2.0 */
#pragma once

#ifndef EMULATE
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>

enum { GPU_WINDOW_KEYS = 12, GPU_WINDOW_THREADS = 256 };

struct gpu_window_plan_t {
    int32_t cols[GPU_WINDOW_KEYS], desc[GPU_WINDOW_KEYS];
    int32_t npart, mode, measure_col;
    int64_t offset;
};

struct gpu_window_chunk_t { int first, end, bits; };
struct gpu_window_segment_t { int64_t value; uint32_t head; };

struct gpu_window_scan_t {
    __host__ __device__ gpu_window_segment_t operator()(gpu_window_segment_t a, gpu_window_segment_t b) const {
        return {b.head ? b.value : a.value + b.value, a.head | b.head};
    }
};

template <class T> struct gpu_window_buffer_t {
    T *data = nullptr;
    ~gpu_window_buffer_t() { if (data) cudaFree(data); }
    bool alloc(size_t count) {
        return !count || cudaMalloc((void **)&data, count * sizeof(T)) == cudaSuccess;
    }
};

struct gpu_window_events_t {
    cudaEvent_t start = nullptr, end = nullptr;
    ~gpu_window_events_t() {
        if (start) cudaEventDestroy(start);
        if (end) cudaEventDestroy(end);
    }
    bool open() {
        return cudaEventCreate(&start) == cudaSuccess && cudaEventCreate(&end) == cudaSuccess;
    }
};

__global__ void gpu_window_counts(const gtable_t *table, const uint64_t *scope, uint32_t *counts) {
    uint64_t word = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (word < table->nw) counts[word] = popc(scope[word] & rowmask_g(table, word));
}

__global__ void gpu_window_scope_ids(const gtable_t *table, const uint64_t *scope,
                                     const uint32_t *offsets, uint32_t *ids) {
    uint64_t word = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (word >= table->nw) return;
    uint32_t at = offsets[word];
    for (uint64_t mask = scope[word] & rowmask_g(table, word); mask; mask &= mask - 1)
        ids[at++] = (uint32_t)(word * 64 + ctz64(mask));
}

__global__ void gpu_window_iota(uint32_t *ids, uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) ids[i] = i;
}

DEV uint32_t gpu_window_row(const uint32_t *physical, uint32_t compact) {
    return physical ? physical[compact] : compact;
}

DEV uint32_t gpu_window_code(const gtable_t *table, int col, uint32_t row) {
    return value_at(table, col, row >> 6, row & 63);
}

/* Only the current radix chunk is assembled. CUB's stable pair sort
 * preserves preceding less-significant chunks and physical-row tie order. */
__global__ void gpu_window_make_keys(const gtable_t *table, gpu_window_plan_t plan,
                    gpu_window_chunk_t chunk, const uint32_t *physical, const uint32_t *order,
                    uint32_t n, uint64_t *keys) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint32_t row = gpu_window_row(physical, order[i]);
    uint64_t packed = 0;
    for (int k = chunk.first; k < chunk.end; ++k) {
        int col = plan.cols[k], bits = table->width[col];
        uint64_t code = gpu_window_code(table, col, row);
        if (plan.desc[k]) code ^= (UINT64_C(1) << bits) - 1;
        packed = (packed << bits) | code;
    }
    keys[i] = packed;
}

__global__ void gpu_window_make_segments(const gtable_t *table, gpu_window_plan_t plan,
                    const uint32_t *physical, const uint32_t *order, uint32_t n,
                    const uint64_t *sorted_keys, int partition_shift, gpu_window_segment_t *segments) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint32_t row = gpu_window_row(physical, order[i]);
    bool head = i == 0;
    if (i && plan.npart) {
        if (sorted_keys) {
            head = (sorted_keys[i] >> partition_shift) != (sorted_keys[i - 1] >> partition_shift);
        } else {
            uint32_t prev = gpu_window_row(physical, order[i - 1]);
            for (int k = 0; k < plan.npart && !head; ++k) {
                int col = plan.cols[k];
                head = gpu_window_code(table, col, row) != gpu_window_code(table, col, prev);
            }
        }
    }
    int64_t value = plan.mode ? plan.offset + (int64_t)gpu_window_code(table, plan.measure_col, row) : 1;
    segments[i] = {value, (uint32_t)head};
}

__global__ void gpu_window_scatter(const uint32_t *physical, const uint32_t *order,
                    const gpu_window_segment_t *segments, uint32_t n, int64_t *values) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) values[gpu_window_row(physical, order[i])] = segments[i].value;
}

static unsigned gpu_window_blocks(uint64_t n) {
    return (unsigned)((n + GPU_WINDOW_THREADS - 1) / GPU_WINDOW_THREADS);
}

static bool gpu_window_col_ok(const gtable_t &table, int col) {
    return col >= 0 && col < table.k && table.width[col] >= 0 && table.width[col] <= 32;
}

/* With out_ids, return active rows compacted in physical order. Without
 * out_ids, return dense values with zero in rows excluded by the scope. */
extern "C" int gpu_plane_window(void *hv, const uint64_t *bitmap, int32_t nkeys,
                  const int32_t *cols, const int32_t *desc, int32_t npart, int32_t mode,
                  int32_t measure_col, int64_t measure_offset, uint32_t *out_ids, int64_t *out_values,
                  uint64_t cap, uint64_t *nout, double *kernel_ms) {
    if (!nout) return -1;
    *nout = 0;
    if (kernel_ms) *kernel_ms = 0;
    if (!hv || nkeys < 0 || nkeys > GPU_WINDOW_KEYS || (nkeys && (!cols || !desc)) ||
        npart < 0 || npart > nkeys || (mode != 0 && mode != 1)) return -1;
    handle_t *h = (handle_t *)hv;
    const gtable_t &table = h->host;
    if (table.n > INT32_MAX) return -1;
    gpu_window_plan_t plan{};
    plan.npart = npart;
    plan.mode = mode;
    plan.measure_col = measure_col;
    plan.offset = measure_offset;
    for (int k = 0; k < nkeys; ++k) {
        if (!gpu_window_col_ok(table, cols[k]) || (desc[k] != 0 && desc[k] != 1)) return -1;
        plan.cols[k] = cols[k];
        plan.desc[k] = desc[k];
    }
    if (mode && !gpu_window_col_ok(table, measure_col)) return -1;
    uint64_t active = table.n;
    if (bitmap) {
        active = 0;
        for (uint64_t word = 0; word < table.nw; ++word) {
            uint64_t left = table.n - word * 64;
            uint64_t tail = left >= 64 ? UINT64_MAX : (UINT64_C(1) << left) - 1;
            active += __builtin_popcountll(bitmap[word] & tail);
        }
    }
    *nout = active;
    uint64_t output_n = out_ids ? active : table.n;
    if (cap < output_n) return -3;
    if (output_n && !out_values) return -1;
    if (!active) {
        if (output_n) memset(out_values, 0, output_n * sizeof *out_values);
        return 0;
    }
    if (mode) {
        int64_t largest = (int64_t)((UINT64_C(1) << table.width[measure_col]) - 1);
        if (measure_offset > INT64_MAX - largest) return -1;
        int64_t high = measure_offset + largest, n = (int64_t)active;
        if (measure_offset < INT64_MIN / n || high > INT64_MAX / n) return -1;
    }

    /* Chunks run in least-significant tuple order. Each spans up to 64 bits. */
    gpu_window_chunk_t chunks[GPU_WINDOW_KEYS];
    int nchunks = 0;
    if (active > 1) for (int end = nkeys; end > 0;) {
        int first = end, bits = 0;
        while (first > 0 && bits + table.width[cols[first - 1]] <= 64)
            bits += table.width[cols[--first]];
        if (bits) chunks[nchunks++] = {first, end, bits};
        end = first;
    }
    bool compact = active != table.n;
    gpu_window_buffer_t<uint64_t> scope, key0, key1;
    gpu_window_buffer_t<uint32_t> offsets, physical, order0, order1;
    gpu_window_buffer_t<gpu_window_segment_t> segments;
    gpu_window_buffer_t<int64_t> values;
    gpu_window_buffer_t<unsigned char> scratch;
    size_t sort_bytes = 0, scan_bytes = 0, compact_bytes = 0;
    int n = (int)active, nw = (int)table.nw;
    if (!scope.alloc(compact ? nw : 0) || !offsets.alloc(compact ? nw : 0) ||
        !physical.alloc(compact ? active : 0) || !order0.alloc(active) ||
        !order1.alloc(nchunks ? active : 0) || !key0.alloc(nchunks ? active : 0) ||
        !key1.alloc(nchunks ? active : 0) || !segments.alloc(active) || !values.alloc(output_n))
        return -2;
    if (compact &&
        (cudaMemcpy(scope.data, bitmap, table.nw * sizeof *bitmap, cudaMemcpyHostToDevice) != cudaSuccess ||
         cub::DeviceScan::ExclusiveSum(nullptr, compact_bytes, offsets.data, offsets.data, nw) != cudaSuccess))
        return -2;
    for (int c = 0; c < nchunks; ++c) {
        cub::DoubleBuffer<uint64_t> keys(key0.data, key1.data);
        cub::DoubleBuffer<uint32_t> order(order0.data, order1.data);
        size_t bytes = 0;
        if (cub::DeviceRadixSort::SortPairs(nullptr, bytes, keys, order, n, 0, chunks[c].bits) != cudaSuccess)
            return -2;
        sort_bytes = std::max(sort_bytes, bytes);
    }
    if (cub::DeviceScan::InclusiveScan(nullptr, scan_bytes, segments.data, segments.data,
                                       gpu_window_scan_t{}, n) != cudaSuccess ||
        !scratch.alloc(std::max(size_t(1), std::max(compact_bytes, std::max(sort_bytes, scan_bytes)))))
        return -2;

    gpu_window_events_t events;
    if (kernel_ms && (!events.open() || cudaEventRecord(events.start) != cudaSuccess)) return -2;
    if (compact) {
        gpu_window_counts<<<gpu_window_blocks(table.nw), GPU_WINDOW_THREADS>>>(h->dev, scope.data, offsets.data);
        if (cudaGetLastError() != cudaSuccess ||
            cub::DeviceScan::ExclusiveSum(scratch.data, compact_bytes, offsets.data, offsets.data, nw) != cudaSuccess)
            return -2;
        gpu_window_scope_ids<<<gpu_window_blocks(table.nw), GPU_WINDOW_THREADS>>>(
            h->dev, scope.data, offsets.data, physical.data);
        if (cudaGetLastError() != cudaSuccess) return -2;
        if (!out_ids && cudaMemset(values.data, 0, output_n * sizeof *out_values) != cudaSuccess) return -2;
    }
    unsigned blocks = gpu_window_blocks(active);
    gpu_window_iota<<<blocks, GPU_WINDOW_THREADS>>>(order0.data, (uint32_t)active);
    if (cudaGetLastError() != cudaSuccess) return -2;
    uint32_t *order = order0.data, *other = order1.data;
    const uint64_t *sorted_partition_keys = nullptr;
    int partition_shift = 0;
    for (int c = 0; c < nchunks; ++c) {
        gpu_window_chunk_t chunk = chunks[c];
        gpu_window_make_keys<<<blocks, GPU_WINDOW_THREADS>>>(
            h->dev, plan, chunk, physical.data, order, (uint32_t)active, key0.data);
        if (cudaGetLastError() != cudaSuccess) return -2;
        cub::DoubleBuffer<uint64_t> keys(key0.data, key1.data);
        cub::DoubleBuffer<uint32_t> ids(order, other);
        size_t bytes = sort_bytes;
        if (cub::DeviceRadixSort::SortPairs(scratch.data, bytes, keys, ids, n, 0, chunk.bits) != cudaSuccess)
            return -2;
        order = ids.Current();
        other = ids.Alternate();
        if (npart && chunk.first == 0 && chunk.end >= npart) {
            for (int k = npart; k < chunk.end; ++k) partition_shift += table.width[cols[k]];
            if (partition_shift < 64) sorted_partition_keys = keys.Current();
        }
    }
    gpu_window_make_segments<<<blocks, GPU_WINDOW_THREADS>>>(h->dev, plan, physical.data, order,
        (uint32_t)active, sorted_partition_keys, partition_shift, segments.data);
    if (cudaGetLastError() != cudaSuccess ||
        cub::DeviceScan::InclusiveScan(scratch.data, scan_bytes, segments.data, segments.data,
                                       gpu_window_scan_t{}, n) != cudaSuccess)
        return -2;
    gpu_window_scatter<<<blocks, GPU_WINDOW_THREADS>>>(out_ids ? nullptr : physical.data, order, segments.data,
                                                       (uint32_t)active, values.data);
    if (cudaGetLastError() != cudaSuccess) return -2;
    if (kernel_ms) {
        float elapsed;
        if (cudaEventRecord(events.end) != cudaSuccess || cudaEventSynchronize(events.end) != cudaSuccess ||
            cudaEventElapsedTime(&elapsed, events.start, events.end) != cudaSuccess) return -2;
        *kernel_ms = elapsed;
    }
    if (cudaMemcpy(out_values, values.data, output_n * sizeof *out_values, cudaMemcpyDeviceToHost) != cudaSuccess)
        return -2;
    if (out_ids) {
        if (compact) {
            if (cudaMemcpy(out_ids, physical.data, active * sizeof *out_ids, cudaMemcpyDeviceToHost) != cudaSuccess)
                return -2;
        } else {
            for (uint64_t i = 0; i < active; ++i) out_ids[i] = (uint32_t)i;
        }
    }
    return 0;
}
#endif

