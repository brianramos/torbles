/* CUDA kernels of the mathematical modules, or their CPU emulation (-DEMULATE), from one source.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Kernels are written with these macros only (see docs/MATH_CONVENTIONS.md):
 *   TR_KERNEL void name(args)            a kernel
 *   TR_DEVICE / TR_HD                    device / host+device helpers
 *   TR_GRID_LOOP(i, n) { ... }           grid-stride loop over 0..n-1
 *   TR_LAUNCH(name, n, args...)          launch over n work items (blocks/threads chosen here)
 *   tr_atomic_add_u64 / _i64 / _i128, tr_atomic_min_i64 / _u64, tr_atomic_max_i64 / _u64, tr_atomic_or_u64
 *   tr_popc64(x)                         population count of a 64-bit word (device and emulation)
 *   tr_cuda_alloc / tr_cuda_zero / tr_cuda_copy_in / tr_cuda_copy_out / tr_cuda_free / tr_cuda_sync
 * A kernel's result must not depend on the order of its threads: accumulate with these atomics or write one output
 * slot per thread; no __syncthreads, no shared-memory barriers. In emulation, TORBLES_EMU_SCRAMBLE=1 runs the
 * threads of a launch in a scrambled order so that order dependence is caught by the tests. */
#ifndef TORBLES_MATH_CUDA_H
#define TORBLES_MATH_CUDA_H
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "math_num.h"

#define TR_THREADS 256
#define TR_MAX_BLOCKS 4096

#ifdef EMULATE
  #include <stdio.h>
  #define TR_KERNEL static
  #define TR_DEVICE static inline
  #define TR_HD static inline
  static thread_local uint64_t tr_emu_thread, tr_emu_threads;
  #define TR_GRID_LOOP(i, n) for (uint64_t i = tr_emu_thread; i < (uint64_t)(n); i += tr_emu_threads)
  static inline uint64_t tr_emu_stride(uint64_t total) {           /* a multiplier coprime to total, or 1 */
      static const uint64_t primes[] = {1000003ULL, 998244353ULL, 1000000007ULL, 104729ULL, 7919ULL};
      const char *s = getenv("TORBLES_EMU_SCRAMBLE");
      if (!s || !*s || *s == '0' || total < 2) return 1;
      for (unsigned i = 0; i < sizeof primes / sizeof primes[0]; i++) if (total % primes[i]) return primes[i] % total ? primes[i] % total : 1;
      return 1;
  }
  #define TR_LAUNCH(kernel, n, ...) do {                                                                     \
      uint64_t _n = (uint64_t)(n), _b = (_n + TR_THREADS - 1) / TR_THREADS;                                \
      if (_b > TR_MAX_BLOCKS) _b = TR_MAX_BLOCKS;                                                          \
      if (_b == 0) _b = 1;                                                                                  \
      uint64_t _t = _b * TR_THREADS, _m = tr_emu_stride(_t), _o = _m > 1 ? (_t / 3) : 0;                   \
      tr_emu_threads = _t;                                                                                  \
      for (uint64_t _k = 0; _k < _t; _k++) { tr_emu_thread = (uint64_t)(((unsigned __int128)_k * _m + _o) % _t); kernel(__VA_ARGS__); } \
  } while (0)
  static inline uint64_t tr_atomic_add_u64(uint64_t *p, uint64_t v) { uint64_t o = *p; *p = o + v; return o; }
  static inline void tr_atomic_min_i64(int64_t *p, int64_t v) { if (v < *p) *p = v; }
  static inline void tr_atomic_max_i64(int64_t *p, int64_t v) { if (v > *p) *p = v; }
  static inline uint64_t tr_atomic_or_u64(uint64_t *p, uint64_t v) { uint64_t o = *p; *p = o | v; return o; }
  static inline void tr_atomic_min_u64(uint64_t *p, uint64_t v) { if (v < *p) *p = v; }
  static inline void tr_atomic_max_u64(uint64_t *p, uint64_t v) { if (v > *p) *p = v; }
  static inline int tr_popc64(uint64_t x) { return __builtin_popcountll(x); }
  static inline int tr_cuda_alloc(void **p, size_t bytes) { *p = calloc(1, bytes ? bytes : 1); return *p ? 0 : TR_MATH_DEVICE; }
  static inline int tr_cuda_zero(void *p, size_t bytes) { memset(p, 0, bytes); return 0; }
  static inline int tr_cuda_copy_in(void *dst, const void *src, size_t bytes) { if (bytes) memcpy(dst, src, bytes); return 0; }
  static inline int tr_cuda_copy_out(void *dst, const void *src, size_t bytes) { if (bytes) memcpy(dst, src, bytes); return 0; }
  static inline void tr_cuda_free(void *p) { free(p); }
  static inline int tr_cuda_sync(void) { return 0; }
  static inline int tr_cuda_device_name(char *out, int cap) { snprintf(out, (size_t)cap, "CPU emulation of the CUDA kernels"); return 0; }
  static inline int tr_cuda_emulated(void) { return 1; }
#else
  #include <cuda_runtime.h>
  #include <stdio.h>
  #define TR_KERNEL __global__
  #define TR_DEVICE __device__ static inline
  #define TR_HD __host__ __device__ static inline
  #define TR_GRID_LOOP(i, n) for (uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; i < (uint64_t)(n); \
                                   i += (uint64_t)gridDim.x * blockDim.x)
  #define TR_LAUNCH(kernel, n, ...) do {                                                                     \
      uint64_t _n = (uint64_t)(n), _b = (_n + TR_THREADS - 1) / TR_THREADS;                                \
      if (_b > TR_MAX_BLOCKS) _b = TR_MAX_BLOCKS;                                                          \
      if (_b == 0) _b = 1;                                                                                  \
      kernel<<<(unsigned)_b, TR_THREADS>>>(__VA_ARGS__);                                                   \
  } while (0)
  __device__ static inline uint64_t tr_atomic_add_u64(uint64_t *p, uint64_t v) {
      return (uint64_t)atomicAdd((unsigned long long *)p, (unsigned long long)v); }
  __device__ static inline void tr_atomic_min_i64(int64_t *p, int64_t v) { atomicMin((long long *)p, (long long)v); }
  __device__ static inline void tr_atomic_max_i64(int64_t *p, int64_t v) { atomicMax((long long *)p, (long long)v); }
  __device__ static inline uint64_t tr_atomic_or_u64(uint64_t *p, uint64_t v) {
      return (uint64_t)atomicOr((unsigned long long *)p, (unsigned long long)v); }
  __device__ static inline void tr_atomic_min_u64(uint64_t *p, uint64_t v) { atomicMin((unsigned long long *)p, (unsigned long long)v); }
  __device__ static inline void tr_atomic_max_u64(uint64_t *p, uint64_t v) { atomicMax((unsigned long long *)p, (unsigned long long)v); }
  __device__ static inline int tr_popc64(uint64_t x) { return __popcll((unsigned long long)x); }
  static inline int tr_cuda_alloc(void **p, size_t bytes) {
      if (cudaMalloc(p, bytes ? bytes : 1) != cudaSuccess) { *p = NULL; return TR_MATH_DEVICE; }
      return cudaMemset(*p, 0, bytes ? bytes : 1) == cudaSuccess ? 0 : TR_MATH_DEVICE; }
  static inline int tr_cuda_zero(void *p, size_t bytes) { return cudaMemset(p, 0, bytes) == cudaSuccess ? 0 : TR_MATH_DEVICE; }
  static inline int tr_cuda_copy_in(void *dst, const void *src, size_t bytes) {
      return !bytes || cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice) == cudaSuccess ? 0 : TR_MATH_DEVICE; }
  static inline int tr_cuda_copy_out(void *dst, const void *src, size_t bytes) {
      return !bytes || cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost) == cudaSuccess ? 0 : TR_MATH_DEVICE; }
  static inline void tr_cuda_free(void *p) { if (p) cudaFree(p); }
  static inline int tr_cuda_sync(void) { return cudaDeviceSynchronize() == cudaSuccess && cudaGetLastError() == cudaSuccess ? 0 : TR_MATH_DEVICE; }
  static inline int tr_cuda_device_name(char *out, int cap) {
      int n = 0; cudaDeviceProp p;
      if (cudaGetDeviceCount(&n) != cudaSuccess || n < 1 || cudaGetDeviceProperties(&p, 0) != cudaSuccess) {
          snprintf(out, (size_t)cap, "no CUDA device"); return TR_MATH_DEVICE; }
      snprintf(out, (size_t)cap, "%.200s (compute %d.%d)", p.name, p.major, p.minor); return 0; }
  static inline int tr_cuda_emulated(void) { return 0; }
#endif

/* signed 64-bit accumulation (two's complement; the host proves the sum fits before launching) */
#ifdef EMULATE
static inline void tr_atomic_add_i64(int64_t *p, int64_t x) { tr_atomic_add_u64((uint64_t *)p, (uint64_t)x); }
#else
__device__ static inline void tr_atomic_add_i64(int64_t *p, int64_t x) { tr_atomic_add_u64((uint64_t *)p, (uint64_t)x); }
#endif

/* exact signed 128-bit accumulation in two words: lo += x (mod 2^64), hi += sign extension + carry of that add */
#ifdef EMULATE
static inline void tr_atomic_add_i128(int64_t *lo, int64_t *hi, int64_t x) {
    uint64_t old = tr_atomic_add_u64((uint64_t *)lo, (uint64_t)x);
    uint64_t carry = (old + (uint64_t)x) < old;
    tr_atomic_add_u64((uint64_t *)hi, (uint64_t)(x < 0 ? -1 : 0) + carry);
}
#else
__device__ static inline void tr_atomic_add_i128(int64_t *lo, int64_t *hi, int64_t x) {
    uint64_t old = tr_atomic_add_u64((uint64_t *)lo, (uint64_t)x);
    uint64_t carry = (old + (uint64_t)x) < old;
    tr_atomic_add_u64((uint64_t *)hi, (uint64_t)(x < 0 ? -1 : 0) + carry);
}
#endif
#endif
