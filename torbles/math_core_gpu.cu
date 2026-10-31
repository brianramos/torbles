/* Core of the mathematical CUDA library: device identity and a self-test of the exact accumulators.
 * SPDX-License-Identifier: Apache-2.0 */
#include "math_cuda.h"

#ifdef __cplusplus
extern "C" {
#endif
/* the device the kernels run on, or "CPU emulation of the CUDA kernels" */
const char *tr_mgpu_device_name(void) {
    static char name[256];
    if (!name[0]) tr_cuda_device_name(name, (int)sizeof name);
    return name;
}
int tr_mgpu_emulated(void) { return tr_cuda_emulated(); }
#ifdef __cplusplus
}
#endif

TR_KERNEL void tr_core_selftest_kernel(const int64_t *x, uint64_t n, int64_t *lo, int64_t *hi, int64_t *mn, int64_t *mx) {
    TR_GRID_LOOP(i, n) {
        tr_atomic_add_i128(lo, hi, x[i]);
        tr_atomic_min_i64(mn, x[i]);
        tr_atomic_max_i64(mx, x[i]);
    }
}

#ifdef __cplusplus
extern "C" {
#endif
/* sums n int64 values exactly (128 bits) with their min and max on the device: lo, hi, min, max in out[4] */
int tr_mgpu_core_selftest(const int64_t *x, uint64_t n, int64_t out[4]) {
    int64_t *dx = NULL, *acc = NULL, init[4] = {0, 0, INT64_MAX, INT64_MIN};
    int rc = tr_cuda_alloc((void **)&dx, n * sizeof(int64_t));
    if (!rc) rc = tr_cuda_alloc((void **)&acc, sizeof init);
    if (!rc) rc = tr_cuda_copy_in(dx, x, n * sizeof(int64_t));
    if (!rc) rc = tr_cuda_copy_in(acc, init, sizeof init);
    if (!rc) { TR_LAUNCH(tr_core_selftest_kernel, n, dx, n, acc, acc + 1, acc + 2, acc + 3); rc = tr_cuda_sync(); }
    if (!rc) rc = tr_cuda_copy_out(out, acc, sizeof init);
    tr_cuda_free(dx); tr_cuda_free(acc);
    return rc;
}
#ifdef __cplusplus
}
#endif
