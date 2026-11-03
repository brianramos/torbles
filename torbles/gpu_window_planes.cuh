/* Copyright 2026 Brian Richard RAMOS
 * SPDX-License-Identifier: Apache-2.0 */
#pragma once

#ifndef EMULATE
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>
#include <cub/device/device_reduce.cuh>
#include <cub/block/block_reduce.cuh>

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

HD bool gpu_window_qual_pass(int32_t op, int64_t value, int64_t threshold) {
    switch (op) {
        case P_TRUE: return true;
        case P_EQ: return value == threshold;
        case P_NE: return value != threshold;
        case P_LT: return value < threshold;
        case P_LE: return value <= threshold;
        case P_GT: return value > threshold;
        default: return value >= threshold;
    }
}

/* A packed survivor mask over the compact physical-row stream. Each warp
 * writes one word and its population. Offsets reserve stable output ranges. */
__global__ void gpu_window_qual_flags(const int64_t *values, uint32_t n, int32_t op,
                        int64_t threshold, int32_t op2, int64_t threshold2,
                        uint32_t *flags, uint32_t *offsets, uint32_t words) {
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned lane = threadIdx.x & 31;
    bool live = row < n && gpu_window_qual_pass(op, values[row], threshold) &&
                            gpu_window_qual_pass(op2, values[row], threshold2);
    uint32_t mask = __ballot_sync(0xffffffffU, live);
    if (!lane && row < n) {
        flags[row >> 5] = mask;
        offsets[row >> 5] = popc32(mask);
    }
    if (!row) offsets[words] = 0;
}

template <int Shift> DEV uint64_t gpu_window_plane_step(uint64_t code, unsigned lane, uint64_t mask) {
    uint64_t other = __shfl_xor_sync(0xffffffffU, (unsigned long long)code, Shift);
    return (lane & Shift) ? code ^ (((other >> Shift) ^ code) & mask)
                          : code ^ ((((code >> Shift) ^ other) & mask) << Shift);
}

/* Two independent 32x32 butterfly transposes, one in each register limb.
 * Each warp writes packed value planes for its consecutive 32 physical
 * rows. Signed values are rebased into an order-preserving unsigned code. */
__global__ void gpu_window_value_planes(const int64_t *values, uint32_t n, int64_t base,
                            uint32_t width, uint32_t *planes) {
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned lane = threadIdx.x & 31;
    uint32_t word = row >> 5;
    if ((row - lane) >= n) return;  /* entire warp inactive */
    uint64_t code = row < n ? (uint64_t)values[row] - (uint64_t)base : 0;
    code = gpu_window_plane_step<16>(code, lane, UINT64_C(0x0000ffff0000ffff));
    code = gpu_window_plane_step<8>(code, lane, UINT64_C(0x00ff00ff00ff00ff));
    code = gpu_window_plane_step<4>(code, lane, UINT64_C(0x0f0f0f0f0f0f0f0f));
    code = gpu_window_plane_step<2>(code, lane, UINT64_C(0x3333333333333333));
    code = gpu_window_plane_step<1>(code, lane, UINT64_C(0x5555555555555555));
    uint32_t *tile = planes + (size_t)word * width;
    if (lane < width) tile[lane] = (uint32_t)code;
    if (lane + 32 < width) tile[lane + 32] = (uint32_t)(code >> 32);
}

/* P_BETWEEN is an internal constant-false marker after range folding. */
DEV uint32_t gpu_window_plane_result(int32_t op, uint32_t scope, uint32_t lt, uint32_t eq) {
    switch (op) {
        case P_TRUE: return scope;
        case P_BETWEEN: return 0;
        case P_EQ: return eq;
        case P_NE: return scope & ~eq;
        case P_LT: return lt;
        case P_LE: return lt | eq;
        case P_GT: return scope & ~(lt | eq);
        default: return scope & ~lt;
    }
}

DEV void gpu_window_plane_step_compare(uint32_t plane, bool bit, uint32_t &lt, uint32_t &eq) {
    if (bit) {
        lt |= eq & ~plane;
        eq &= plane;
    } else {
        eq &= ~plane;
    }
}

__global__ void gpu_window_plane_qualify(const uint32_t *planes, uint32_t n, uint32_t width,
                    int32_t op, uint64_t threshold, int32_t op2, uint64_t threshold2,
                    uint32_t *flags, uint32_t *offsets, uint32_t words) {
    uint32_t word = blockIdx.x * blockDim.x + threadIdx.x;
    if (word >= words) return;
    uint32_t left = n - word * 32;
    uint32_t scope = left >= 32 ? UINT32_MAX : (UINT32_C(1) << left) - 1;
    uint32_t lt = 0, eq = op == P_TRUE || op == P_BETWEEN ? 0 : scope;
    uint32_t lt2 = 0, eq2 = op2 == P_TRUE || op2 == P_BETWEEN ? 0 : scope;
    const uint32_t *tile = planes + (size_t)word * width;
    for (int bit = (int)width - 1; bit >= 0 && (eq || eq2); --bit) {
        uint32_t p = tile[bit];
        gpu_window_plane_step_compare(p, (threshold >> bit) & 1, lt, eq);
        gpu_window_plane_step_compare(p, (threshold2 >> bit) & 1, lt2, eq2);
    }
    uint32_t mask = gpu_window_plane_result(op, scope, lt, eq)
                  & gpu_window_plane_result(op2, scope, lt2, eq2);
    flags[word] = mask;
    offsets[word] = popc32(mask);
    if (!word) offsets[words] = 0;
}

__global__ void gpu_window_qual_emit(const int64_t *values, const uint32_t *physical, uint32_t n,
                        const uint32_t *flags, const uint32_t *offsets, uint32_t start, uint32_t count,
                        uint32_t *out_ids, int64_t *out_values) {
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;
    uint32_t word = row >> 5;
    uint32_t bit = UINT32_C(1) << (row & 31), mask = flags[word];
    if (!(mask & bit)) return;
    uint32_t at = offsets[word] + popc32(mask & (bit - 1));
    if (at < start || at - start >= count) return;
    at -= start;
    out_ids[at] = gpu_window_row(physical, row);
    out_values[at] = values[row];
}

static unsigned gpu_window_blocks(uint64_t n) {
    return (unsigned)((n + GPU_WINDOW_THREADS - 1) / GPU_WINDOW_THREADS);
}

static bool gpu_window_col_ok(const gtable_t &table, int col) {
    return col >= 0 && col < table.k && table.width[col] >= 0 && table.width[col] <= 32;
}

static uint64_t gpu_window_slice(uint64_t count, uint64_t offset, uint64_t limit) {
    return count > offset ? std::min(count - offset, limit) : 0;
}

static int gpu_window_terminal_copy(int64_t *values, uint32_t *physical, uint32_t count, int descending,
                    uint64_t offset, uint64_t take, uint32_t *out_ids, int64_t *out_values,
                    gpu_window_events_t &events, double *kernel_ms) {
    if (!take) return 0;
    gpu_window_buffer_t<uint32_t> ids, other_ids;
    gpu_window_buffer_t<int64_t> other_values;
    gpu_window_buffer_t<unsigned char> scratch;
    if (descending >= 0 && count > 1) {
        if ((!physical && !ids.alloc(count)) || !other_ids.alloc(count) || !other_values.alloc(count)) return -2;
        cub::DoubleBuffer<int64_t> keys(values, other_values.data);
        cub::DoubleBuffer<uint32_t> rows(physical ? physical : ids.data, other_ids.data);
        size_t bytes = 0;
        int n = (int)count;
        cudaError_t size_rc = descending
            ? cub::DeviceRadixSort::SortPairsDescending(nullptr, bytes, keys, rows, n)
            : cub::DeviceRadixSort::SortPairs(nullptr, bytes, keys, rows, n);
        if (size_rc != cudaSuccess || !scratch.alloc(std::max(size_t(1), bytes))) return -2;
        if (kernel_ms && cudaEventRecord(events.start) != cudaSuccess) return -2;
        if (!physical) {
            gpu_window_iota<<<gpu_window_blocks(count), GPU_WINDOW_THREADS>>>(ids.data, count);
            if (cudaGetLastError() != cudaSuccess) return -2;
        }
        cudaError_t sort_rc = descending
            ? cub::DeviceRadixSort::SortPairsDescending(scratch.data, bytes, keys, rows, n)
            : cub::DeviceRadixSort::SortPairs(scratch.data, bytes, keys, rows, n);
        if (sort_rc != cudaSuccess) return -2;
        values = keys.Current();
        physical = rows.Current();
        if (kernel_ms) {
            float elapsed;
            if (cudaEventRecord(events.end) != cudaSuccess || cudaEventSynchronize(events.end) != cudaSuccess ||
                cudaEventElapsedTime(&elapsed, events.start, events.end) != cudaSuccess) return -2;
            *kernel_ms += elapsed;
        }
    }
    if (cudaMemcpy(out_values, values + offset, take * sizeof *out_values, cudaMemcpyDeviceToHost) != cudaSuccess)
        return -2;
    if (physical) {
        if (cudaMemcpy(out_ids, physical + offset, take * sizeof *out_ids, cudaMemcpyDeviceToHost) != cudaSuccess)
            return -2;
    } else {
        for (uint64_t i = 0; i < take; ++i) out_ids[i] = (uint32_t)(offset + i);
    }
    return 0;
}

/* With out_ids, return active rows compacted in physical order. Without
 * out_ids, return dense values with zero in rows excluded by the scope. */
enum { GPU_WINDOW_COUNT = 1, GPU_WINDOW_MIN, GPU_WINDOW_MAX, GPU_WINDOW_SUM };

struct gpu_window_scalar_op_t {
    int32_t op;
    __host__ __device__ int64_t operator()(int64_t a, int64_t b) const {
        if (op == GPU_WINDOW_MIN) return a < b ? a : b;
        if (op == GPU_WINDOW_MAX) return a > b ? a : b;
        return a + b;
    }
};

DEV int64_t gpu_window_signed(uint64_t bits) {
    uint64_t sign = UINT64_C(1) << 63;
    return bits < sign ? (int64_t)bits : INT64_MIN + (int64_t)(bits - sign);
}

DEV int64_t gpu_window_plane_scalar(const uint32_t *tile, uint32_t width, uint32_t mask,
                                    int64_t base, int32_t op) {
    if (op == GPU_WINDOW_SUM) {
        uint64_t total = (uint64_t)base * popc32(mask);
        for (unsigned bit = 0; bit < width; ++bit)
            total += (uint64_t)popc32(mask & tile[bit]) << bit;
        return gpu_window_signed(total);
    }
    uint64_t code = 0;
    for (int bit = (int)width - 1; bit >= 0; --bit) {
        uint32_t ones = mask & tile[bit], zeros = mask & ~tile[bit];
        if (op == GPU_WINDOW_MIN ? !zeros : !!ones) {
            code |= UINT64_C(1) << bit;
            mask = ones;
        } else {
            mask = zeros;
        }
    }
    return gpu_window_signed(code + (uint64_t)base);
}

__global__ void gpu_window_scalar_partials(const int64_t *values, const uint32_t *planes,
                    const uint32_t *flags, uint32_t n, uint32_t width, int64_t base, int64_t identity,
                    gpu_window_scalar_op_t op, int64_t *partials) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t value = identity;
    if (planes) {
        uint32_t start = i * 32;
        if (start < n) {
            uint32_t left = n - start;
            uint32_t scope = left >= 32 ? UINT32_MAX : (UINT32_C(1) << left) - 1;
            uint32_t mask = flags ? flags[i] : scope;
            if (mask) value = gpu_window_plane_scalar(planes + (size_t)i * width, width, mask, base, op.op);
        }
    } else if (i < n && (!flags || ((flags[i >> 5] >> (i & 31)) & 1))) {
        value = values[i];
    }
    using reduction_t = cub::BlockReduce<int64_t, GPU_WINDOW_THREADS>;
    __shared__ typename reduction_t::TempStorage storage;
    int64_t partial = reduction_t(storage).Reduce(value, op);
    if (!threadIdx.x) partials[blockIdx.x] = partial;
}

static int gpu_window_scalar(const int64_t *values, const uint32_t *planes, const uint32_t *flags,
                    uint32_t n, uint32_t width, int64_t base, int32_t aggregate,
                    int64_t *out_value, gpu_window_events_t &events, double *kernel_ms) {
    unsigned blocks = gpu_window_blocks(planes ? (n + 31U) / 32U : n);
    gpu_window_buffer_t<int64_t> partials, result;
    gpu_window_buffer_t<unsigned char> scratch;
    gpu_window_scalar_op_t op{aggregate};
    int64_t identity = aggregate == GPU_WINDOW_MIN ? INT64_MAX : aggregate == GPU_WINDOW_MAX ? INT64_MIN : 0;
    size_t bytes = 0;
    if (!partials.alloc(blocks) || !result.alloc(blocks > 1 ? 1 : 0)) return -2;
    if (blocks > 1 &&
        (cub::DeviceReduce::Reduce(nullptr, bytes, partials.data, result.data, (int)blocks,
                                   op, identity) != cudaSuccess ||
         !scratch.alloc(std::max(size_t(1), bytes))))
        return -2;
    if (kernel_ms && cudaEventRecord(events.start) != cudaSuccess) return -2;
    gpu_window_scalar_partials<<<blocks, GPU_WINDOW_THREADS>>>(
        values, planes, flags, n, width, base, identity, op, partials.data);
    if (cudaGetLastError() != cudaSuccess) return -2;
    if (blocks > 1 && cub::DeviceReduce::Reduce(scratch.data, bytes, partials.data, result.data,
                                                (int)blocks, op, identity) != cudaSuccess)
        return -2;
    if (kernel_ms) {
        float elapsed;
        if (cudaEventRecord(events.end) != cudaSuccess || cudaEventSynchronize(events.end) != cudaSuccess ||
            cudaEventElapsedTime(&elapsed, events.start, events.end) != cudaSuccess) return -2;
        *kernel_ms += elapsed;
    }
    const int64_t *scalar = blocks > 1 ? result.data : partials.data;
    return cudaMemcpy(out_value, scalar, sizeof *out_value, cudaMemcpyDeviceToHost) == cudaSuccess ? 0 : -2;
}

static int32_t gpu_window_range_op(int32_t op, int64_t threshold, int64_t low, int64_t high) {
    if (op == P_TRUE || (threshold >= low && threshold <= high)) return op;
    return gpu_window_qual_pass(op, threshold < low ? low : high, threshold) ? P_TRUE : P_BETWEEN;
}

static int gpu_plane_window_run(void *hv, const uint64_t *bitmap, int32_t nkeys,
                  const int32_t *cols, const int32_t *desc, int32_t npart, int32_t mode,
                  int32_t measure_col, int64_t measure_offset, uint32_t *out_ids, int64_t *out_values,
                  uint64_t cap, uint64_t *nout, double *kernel_ms, int32_t op, int64_t threshold,
                  int32_t descending, uint64_t offset, uint64_t limit, uint32_t *plane_width = nullptr,
                  int32_t op2 = P_TRUE, int64_t threshold2 = 0, int32_t aggregate = 0) {
    if (plane_width) *plane_width = 0;
    if (!nout) return -1;
    *nout = 0;
    if (kernel_ms) *kernel_ms = 0;
    bool qualify = op != P_TRUE || op2 != P_TRUE, terminal = descending >= -1;
    bool select = qualify || terminal || aggregate;
    if ((op != P_TRUE && (op < P_EQ || op > P_GE)) ||
        (op2 != P_TRUE && (op2 < P_EQ || op2 > P_GE)) ||
        (aggregate && (aggregate < GPU_WINDOW_COUNT || aggregate > GPU_WINDOW_SUM || !out_values)) ||
        (!aggregate && select && cap && (!out_ids || !out_values))) return -1;
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
    if (!qualify && !aggregate) *nout = terminal ? gpu_window_slice(active, offset, limit) : active;
    uint64_t output_n = select ? active : out_ids ? active : table.n;
    if (terminal && !qualify && cap < *nout) return -3;
    if (terminal && ((!qualify && !*nout) || !limit)) return 0;
    if (!select && cap < output_n) return -3;
    if (!select && output_n && !out_values) return -1;
    if (!active) {
        if (aggregate == GPU_WINDOW_COUNT) { *out_values = 0; *nout = 1; }
        else if (output_n) memset(out_values, 0, output_n * sizeof *out_values);
        return 0;
    }
    int64_t code_base = 0, code_high = (int64_t)active;
    if (mode) {
        int64_t largest = (int64_t)((UINT64_C(1) << table.width[measure_col]) - 1);
        if (measure_offset > INT64_MAX - largest) return -1;
        int64_t high = measure_offset + largest, n = (int64_t)active;
        if (measure_offset < INT64_MIN / n || high > INT64_MAX / n) return -1;
        code_base = std::min(INT64_C(0), measure_offset) * n;
        code_high = std::max(INT64_C(0), high) * n;
    }

    if (aggregate == GPU_WINDOW_SUM &&
        (code_base < INT64_MIN / (int64_t)active || code_high > INT64_MAX / (int64_t)active))
        return -1;

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
    gpu_window_buffer_t<uint32_t> offsets, physical, order0, order1, qual_flags, qual_offsets, value_planes;
    gpu_window_buffer_t<gpu_window_segment_t> segments;
    gpu_window_buffer_t<int64_t> values;
    gpu_window_buffer_t<unsigned char> scratch;
    size_t sort_bytes = 0, scan_bytes = 0, compact_bytes = 0, qual_bytes = 0;
    uint32_t value_words = (uint32_t)((active + 31) / 32);
    uint32_t qual_words = qualify ? value_words : 0;
    uint32_t width = 0;
    int32_t packed_op = gpu_window_range_op(op, threshold, code_base, code_high);
    int32_t packed_op2 = gpu_window_range_op(op2, threshold2, code_base, code_high);
    if (plane_width && (aggregate >= GPU_WINDOW_MIN ||
                        (packed_op >= P_EQ && packed_op <= P_GE) ||
                        (packed_op2 >= P_EQ && packed_op2 <= P_GE))) {
        uint64_t span = (uint64_t)code_high - (uint64_t)code_base;
        uint32_t bits = span ? 64 - __builtin_clzll(span) : 1;
        size_t count = (size_t)value_words * bits;
        if (count <= (size_t(256) << 20) / sizeof(uint32_t) && value_planes.alloc(count)) width = bits;
    }
    int n = (int)active, nw = (int)table.nw;
    if (!scope.alloc(compact ? nw : 0) || !offsets.alloc(compact ? nw : 0) ||
        !physical.alloc(compact ? active : 0) || !order0.alloc(active) ||
        !order1.alloc(nchunks ? active : 0) || !key0.alloc(nchunks ? active : 0) ||
        !key1.alloc(nchunks ? active : 0) || !segments.alloc(active) || !values.alloc(output_n) ||
        !qual_flags.alloc(qual_words) || !qual_offsets.alloc(qualify ? qual_words + 1 : 0))
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
    if (qualify &&
        cub::DeviceScan::ExclusiveSum(nullptr, qual_bytes, qual_offsets.data, qual_offsets.data,
                                      (int)qual_words + 1) != cudaSuccess)
        return -2;
    if (cub::DeviceScan::InclusiveScan(nullptr, scan_bytes, segments.data, segments.data,
                                       gpu_window_scan_t{}, n) != cudaSuccess ||
        !scratch.alloc(std::max(size_t(1), std::max(qual_bytes,
                                                   std::max(compact_bytes, std::max(sort_bytes, scan_bytes))))))
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
        if (!select && !out_ids && cudaMemset(values.data, 0, output_n * sizeof *out_values) != cudaSuccess) return -2;
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
    gpu_window_scatter<<<blocks, GPU_WINDOW_THREADS>>>(select || out_ids ? nullptr : physical.data, order, segments.data,
                                                       (uint32_t)active, values.data);
    if (cudaGetLastError() != cudaSuccess) return -2;
    if (width) {
        gpu_window_value_planes<<<blocks, GPU_WINDOW_THREADS>>>(
            values.data, (uint32_t)active, code_base, width, value_planes.data);
        if (cudaGetLastError() != cudaSuccess) return -2;
    }
    if (qualify) {
        if (width) {
            gpu_window_plane_qualify<<<gpu_window_blocks(qual_words), GPU_WINDOW_THREADS>>>(
                value_planes.data, (uint32_t)active, width, packed_op, (uint64_t)threshold - (uint64_t)code_base,
                packed_op2, (uint64_t)threshold2 - (uint64_t)code_base, qual_flags.data, qual_offsets.data, qual_words);
        } else {
            gpu_window_qual_flags<<<blocks, GPU_WINDOW_THREADS>>>(
                values.data, (uint32_t)active, op, threshold, op2, threshold2,
                qual_flags.data, qual_offsets.data, qual_words);
        }
        if (cudaGetLastError() != cudaSuccess ||
            cub::DeviceScan::ExclusiveSum(scratch.data, qual_bytes, qual_offsets.data, qual_offsets.data,
                                          (int)qual_words + 1) != cudaSuccess)
            return -2;
    }
    if (kernel_ms) {
        float elapsed;
        if (cudaEventRecord(events.end) != cudaSuccess || cudaEventSynchronize(events.end) != cudaSuccess ||
            cudaEventElapsedTime(&elapsed, events.start, events.end) != cudaSuccess) return -2;
        *kernel_ms = elapsed;
    }
    if (aggregate) {
        uint32_t survivors = (uint32_t)active;
        if (qualify && cudaMemcpy(&survivors, qual_offsets.data + qual_words, sizeof survivors,
                                  cudaMemcpyDeviceToHost) != cudaSuccess)
            return -2;
        if (plane_width) *plane_width = width;
        if (aggregate == GPU_WINDOW_COUNT) {
            *out_values = survivors;
            *nout = 1;
            return 0;
        }
        if (!survivors) return 0;
        int rc = gpu_window_scalar(values.data, value_planes.data, qual_flags.data,
                                  (uint32_t)active, width, code_base, aggregate,
                                  out_values, events, kernel_ms);
        if (!rc) *nout = 1;
        return rc;
    }
    if (qualify) {
        uint32_t survivors;
        if (cudaMemcpy(&survivors, qual_offsets.data + qual_words, sizeof survivors,
                       cudaMemcpyDeviceToHost) != cudaSuccess)
            return -2;
        uint64_t take = terminal ? gpu_window_slice(survivors, offset, limit) : survivors;
        *nout = take;
        if (plane_width) *plane_width = width;
        if (cap < take) return -3;
        if (!take) return 0;
        bool sort = terminal && descending >= 0;
        uint32_t start = terminal && !sort ? (uint32_t)offset : 0;
        uint32_t emit = sort ? survivors : (uint32_t)take;
        gpu_window_buffer_t<uint32_t> survivor_ids;
        gpu_window_buffer_t<int64_t> survivor_values;
        if (!survivor_ids.alloc(emit) || !survivor_values.alloc(emit)) return -2;
        if (kernel_ms && cudaEventRecord(events.start) != cudaSuccess) return -2;
        gpu_window_qual_emit<<<blocks, GPU_WINDOW_THREADS>>>(values.data, physical.data, (uint32_t)active,
            qual_flags.data, qual_offsets.data, start, emit, survivor_ids.data, survivor_values.data);
        if (cudaGetLastError() != cudaSuccess) return -2;
        if (kernel_ms) {
            float elapsed;
            if (cudaEventRecord(events.end) != cudaSuccess || cudaEventSynchronize(events.end) != cudaSuccess ||
                cudaEventElapsedTime(&elapsed, events.start, events.end) != cudaSuccess) return -2;
            *kernel_ms += elapsed;
        }
        if (sort)
            return gpu_window_terminal_copy(survivor_values.data, survivor_ids.data, survivors, descending,
                                            offset, take, out_ids, out_values, events, kernel_ms);
        if (cudaMemcpy(out_ids, survivor_ids.data, take * sizeof *out_ids, cudaMemcpyDeviceToHost) != cudaSuccess ||
            cudaMemcpy(out_values, survivor_values.data, take * sizeof *out_values,
                       cudaMemcpyDeviceToHost) != cudaSuccess)
            return -2;
        return 0;
    }
    if (terminal)
        return gpu_window_terminal_copy(values.data, physical.data, (uint32_t)active, descending,
                                        offset, *nout, out_ids, out_values, events, kernel_ms);
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

extern "C" int gpu_plane_window(void *hv, const uint64_t *bitmap, int32_t nkeys,
                  const int32_t *cols, const int32_t *desc, int32_t npart, int32_t mode,
                  int32_t measure_col, int64_t measure_offset, uint32_t *out_ids, int64_t *out_values,
                  uint64_t cap, uint64_t *nout, double *kernel_ms) {
    return gpu_plane_window_run(hv, bitmap, nkeys, cols, desc, npart, mode, measure_col,
                                measure_offset, out_ids, out_values, cap, nout, kernel_ms, P_TRUE, 0, -2, 0, 0);
}

extern "C" int gpu_plane_window_qualify(void *hv, const uint64_t *bitmap, int32_t nkeys,
                  const int32_t *cols, const int32_t *desc, int32_t npart, int32_t mode,
                  int32_t measure_col, int64_t measure_offset, uint32_t *out_ids, int64_t *out_values,
                  uint64_t cap, uint64_t *nout, double *kernel_ms, int32_t op, int64_t threshold) {
    if (op < P_EQ || op > P_GE) {
        if (nout) *nout = 0;
        return -1;
    }
    return gpu_plane_window_run(hv, bitmap, nkeys, cols, desc, npart, mode, measure_col,
                                measure_offset, out_ids, out_values, cap, nout, kernel_ms, op, threshold, -2, 0, 0);
}

/* descending=-1 slices physical order. Otherwise a stable signed radix
 * sort orders by the window value, retaining physical ties. */
extern "C" int gpu_plane_window_terminal(void *hv, const uint64_t *bitmap, int32_t nkeys,
                  const int32_t *cols, const int32_t *desc, int32_t npart, int32_t mode,
                  int32_t measure_col, int64_t measure_offset, uint32_t *out_ids, int64_t *out_values,
                  uint64_t cap, uint64_t *nout, double *kernel_ms, int32_t op, int64_t threshold,
                  int32_t descending, uint64_t offset, uint64_t limit) {
    if ((op != P_TRUE && (op < P_EQ || op > P_GE)) || descending < -1 || descending > 1) {
        if (nout) *nout = 0;
        return -1;
    }
    return gpu_plane_window_run(hv, bitmap, nkeys, cols, desc, npart, mode, measure_col,
                                measure_offset, out_ids, out_values, cap, nout, kernel_ms,
                                op, threshold, descending, offset, limit);
}

extern "C" int gpu_plane_window_torus_terminal(void *hv, const uint64_t *bitmap, int32_t nkeys,
                  const int32_t *cols, const int32_t *desc, int32_t npart, int32_t mode,
                  int32_t measure_col, int64_t measure_offset, uint32_t *out_ids, int64_t *out_values,
                  uint64_t cap, uint64_t *nout, double *kernel_ms, int32_t op, int64_t threshold,
                  int32_t descending, uint64_t offset, uint64_t limit, uint32_t *plane_width,
                  int32_t op2, int64_t threshold2) {
    if ((op != P_TRUE && (op < P_EQ || op > P_GE)) || descending < -1 || descending > 1) {
        if (nout) *nout = 0;
        if (plane_width) *plane_width = 0;
        return -1;
    }
    return gpu_plane_window_run(hv, bitmap, nkeys, cols, desc, npart, mode, measure_col,
                                measure_offset, out_ids, out_values, cap, nout, kernel_ms,
                                op, threshold, descending, offset, limit, plane_width, op2, threshold2);
}

/* Scalar result presence in nout: COUNT always has a value, while an empty
 * MIN/MAX/SUM has none. No window or survivor row arrays cross to host. */
extern "C" int gpu_plane_window_aggregate(void *hv, const uint64_t *bitmap, int32_t nkeys,
                  const int32_t *cols, const int32_t *desc, int32_t npart, int32_t mode,
                  int32_t measure_col, int64_t measure_offset, int64_t *out_value,
                  uint64_t *nout, double *kernel_ms, int32_t op, int64_t threshold,
                  int32_t op2, int64_t threshold2, int32_t aggregate, uint32_t *plane_width) {
    if (aggregate < GPU_WINDOW_COUNT || aggregate > GPU_WINDOW_SUM) {
        if (nout) *nout = 0;
        if (plane_width) *plane_width = 0;
        return -1;
    }
    return gpu_plane_window_run(hv, bitmap, nkeys, cols, desc, npart, mode, measure_col,
                                measure_offset, nullptr, out_value, 1, nout, kernel_ms,
                                op, threshold, -2, 0, 0, plane_width, op2, threshold2, aggregate);
}
#endif

