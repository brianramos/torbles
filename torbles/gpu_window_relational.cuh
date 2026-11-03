/* Copyright 2026 Brian Richard RAMOS
 * SPDX-License-Identifier: Apache-2.0 */
#pragma once

#ifndef EMULATE
#include <cub/device/device_select.cuh>

__global__ void gpu_window_group_reduce(const gpu_window_resident_t window, int key,
                                       unsigned long long *counts, unsigned long long *sums) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= window.n) return;
    uint32_t row = gpu_window_row(window.ids, i);
    uint32_t code = gpu_window_code(window.source, key, row);
    atomicAdd(counts + code, 1ULL);
    atomicAdd(sums + code, (unsigned long long)window.values[i]);
}

extern "C" int gpu_plane_window_group(void *handle, int32_t key, uint32_t domain,
                                       uint64_t *counts, int64_t *sums) {
    if (!handle || !domain || domain > (1U << 17) || !counts || !sums || key < 0) return -1;
    const auto &window = *(const gpu_window_resident_t *)handle;
    if (!gpu_window_col_ok(*window.source_host, key) ||
        (UINT64_C(1) << window.source_host->width[key]) != domain) return -1;
    if (!window.n) {
        memset(counts, 0, domain * sizeof *counts);
        memset(sums, 0, domain * sizeof *sums);
        return 0;
    }
    if (window.code_base < INT64_MIN / (int64_t)window.n ||
        window.code_high > INT64_MAX / (int64_t)window.n) return -1;
    gpu_window_buffer_t<unsigned long long> dcounts, dsums;
    if (!dcounts.alloc(domain) || !dsums.alloc(domain) ||
        cudaMemset(dcounts.data, 0, domain * sizeof *counts) != cudaSuccess ||
        cudaMemset(dsums.data, 0, domain * sizeof *sums) != cudaSuccess) return -2;
    gpu_window_group_reduce<<<gpu_window_blocks(window.n), GPU_WINDOW_THREADS>>>(
        window, key, dcounts.data, dsums.data);
    if (cudaGetLastError() != cudaSuccess ||
        cudaMemcpy(counts, dcounts.data, domain * sizeof *counts, cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(sums, dsums.data, domain * sizeof *sums, cudaMemcpyDeviceToHost) != cudaSuccess) return -2;
    return 0;
}

/* Each consumer observes the same exact rank planes and physical scope. */
__global__ void gpu_window_relation_masks(const uint32_t *planes, uint32_t n, unsigned width,
                       int op_a, uint64_t threshold_a, int op_b, uint64_t threshold_b,
                       int kind, uint32_t *selected, unsigned long long *counts) {
    uint32_t word = blockIdx.x * blockDim.x + threadIdx.x;
    if ((uint64_t)word * 32 >= n) return;
    unsigned active = min(32U, n - word * 32);
    uint32_t scope = active == 32 ? UINT32_MAX : (UINT32_C(1) << active) - 1;
    uint32_t eq_a = scope, eq_b = scope, lt_a = 0, lt_b = 0;
    for (int bit = (int)width - 1; bit >= 0; --bit) {
        uint32_t plane = planes[(size_t)word * width + bit];
        if ((threshold_a >> bit) & 1) {
            lt_a |= eq_a & ~plane;
            eq_a &= plane;
        } else {
            eq_a &= ~plane;
        }
        if ((threshold_b >> bit) & 1) {
            lt_b |= eq_b & ~plane;
            eq_b &= plane;
        } else {
            eq_b &= ~plane;
        }
    }
    uint32_t a = gpu_window_plane_result(op_a, scope, lt_a, eq_a);
    uint32_t b = gpu_window_plane_result(op_b, scope, lt_b, eq_b);
    if (kind < 3) {
        selected[word] = kind == 0 ? a | b : kind == 1 ? a & b : a & ~b;
        return;
    }
    if (kind == 4) a &= b;
    if (a) atomicAdd(counts, (unsigned long long)__popc(a));
    if (kind == 3 && b) atomicAdd(counts + 1, (unsigned long long)__popc(b));
}

__global__ void gpu_window_relation_flags(const uint32_t *selected, uint32_t n, uint8_t *flags) {
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n) flags[row] = (selected[row >> 5] >> (row & 31)) & 1;
}

/* UNION, INTERSECT and EXCEPT operate over a unique value domain (global
 * ROW_NUMBER). COUNT JOIN reduces equality on that domain to mask AND. */
extern "C" int gpu_plane_window_relation(void *handle, int32_t kind, int32_t op_a, int64_t threshold_a,
                       int32_t op_b, int64_t threshold_b, int64_t *out, uint64_t cap, uint64_t *nout) {
    if (!handle || kind < 0 || kind > 4 || op_a < P_TRUE || op_a > P_GE ||
        op_b < P_TRUE || op_b > P_GE || !nout || !out) return -1;
    const auto &window = *(const gpu_window_resident_t *)handle;
    *nout = kind == 3 ? 2 : kind == 4 ? 1 : 0;
    if (cap < *nout) return -1;
    if (!window.n) {
        for (uint64_t i = 0; i < *nout; ++i) out[i] = 0;
        return 0;
    }
    if (window.code_base != 0 || window.code_high != (int64_t)window.n) return -1;
    unsigned width = 64 - __builtin_clzll(window.n);
    uint64_t words = (window.n + 31) / 32;
    gpu_window_buffer_t<uint32_t> planes;
    if (!planes.alloc(words * width)) return -2;
    gpu_window_value_planes<<<gpu_window_blocks(window.n), GPU_WINDOW_THREADS>>>(
        window.values, (uint32_t)window.n, 0, width, planes.data);
    if (cudaGetLastError() != cudaSuccess) return -2;
    op_a = gpu_window_range_op(op_a, threshold_a, 0, window.code_high);
    op_b = gpu_window_range_op(op_b, threshold_b, 0, window.code_high);
    if (kind >= 3) {
        gpu_window_buffer_t<unsigned long long> counts;
        if (!counts.alloc(*nout) || cudaMemset(counts.data, 0, *nout * sizeof *out) != cudaSuccess) return -2;
        gpu_window_relation_masks<<<gpu_window_blocks(words), GPU_WINDOW_THREADS>>>(
            planes.data, (uint32_t)window.n, width, op_a, threshold_a, op_b, threshold_b, kind, nullptr, counts.data);
        if (cudaGetLastError() != cudaSuccess ||
            cudaMemcpy(out, counts.data, *nout * sizeof *out, cudaMemcpyDeviceToHost) != cudaSuccess) return -2;
        return 0;
    }
    if (cap < window.n) return -1;
    gpu_window_buffer_t<uint8_t> selected, scratch;
    gpu_window_buffer_t<int64_t> values;
    gpu_window_buffer_t<uint32_t> count, mask;
    if (!selected.alloc(window.n) || !values.alloc(window.n) || !count.alloc(1) || !mask.alloc(words)) return -2;
    gpu_window_relation_masks<<<gpu_window_blocks(words), GPU_WINDOW_THREADS>>>(
        planes.data, (uint32_t)window.n, width, op_a, threshold_a, op_b, threshold_b, kind, mask.data, nullptr);
    if (cudaGetLastError() != cudaSuccess) return -2;
    gpu_window_relation_flags<<<gpu_window_blocks(window.n), GPU_WINDOW_THREADS>>>(
        mask.data, (uint32_t)window.n, selected.data);
    size_t bytes = 0;
    if (cudaGetLastError() != cudaSuccess ||
        cub::DeviceSelect::Flagged(nullptr, bytes, window.values, selected.data, values.data, count.data,
                                   (int)window.n) != cudaSuccess || !scratch.alloc(bytes) ||
        cub::DeviceSelect::Flagged(scratch.data, bytes, window.values, selected.data, values.data, count.data,
                                   (int)window.n) != cudaSuccess) return -2;
    uint32_t size;
    if (cudaMemcpy(&size, count.data, sizeof size, cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(out, values.data, size * sizeof *out, cudaMemcpyDeviceToHost) != cudaSuccess) return -2;
    *nout = size;
    return 0;
}
#endif
