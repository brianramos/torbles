/* FND-011 on a CUDA device: the payload checksum of a persisted state (math_persist.h), computed before a large
 * payload is uploaded and used. The checksum is a position-weighted sum of 64-bit words mod 2^64 (two lanes, an odd
 * multiplier per position: tr_persist_weight), so it does not depend on the order it is accumulated in: work items
 * take contiguous slices of words, sum them locally and add the two partial sums with 64-bit atomics (wrapping
 * addition mod 2^64 commutes). The result equals tr_persist_checksum bit for bit; tr_persist_validate_sum then
 * compares it with the record's checksum. No barrier, no shared memory.
 * SPDX-License-Identifier: Apache-2.0 */
#include "math_cuda.h"
#include "math_persist.h"

#define TR_PERSIST_SLICE 4096                    /* words per work item, at least */
#define TR_PERSIST_MAX_SLICES 65536

TR_KERNEL void tr_persist_checksum_kernel(const uint64_t *w, uint64_t n, uint64_t slices, uint64_t per, uint64_t *acc) {
    TR_GRID_LOOP(s, slices) {
        uint64_t i0 = s * per, i1 = i0 + per < n ? i0 + per : n;
        uint64_t a = 0, b = 0;
        for (uint64_t i = i0; i < i1; i++) {
            uint64_t x = w[i];
            a += x * tr_persist_weight(i, 0);
            b += x * tr_persist_weight(i, 1);
        }
        if (a) tr_atomic_add_u64(&acc[0], a);
        if (b) tr_atomic_add_u64(&acc[1], b);
    }
}

#ifdef __cplusplus
extern "C" {
#endif
int tr_mgpu_persist_checksum(const uint64_t *words, uint64_t n, const tr_math_budget *budget, uint64_t out[2],
                             tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!out || (n && !words)) return -1;                                                     /* TR_MATH_INVALID */
    tr_math_budget dflt;
    if (!budget) { tr_budget_default(&dflt); budget = &dflt; }
    int rc = tr_budget_check_one(budget, TR_BUDGET_BYTES, tr_persist_device_bytes(n), refusal);
    if (rc) return rc;
    out[0] = out[1] = 0;
    if (!n) return 0;
    uint64_t *dw = NULL, *dacc = NULL, acc[2] = {0, 0};
    uint64_t slices = (n + TR_PERSIST_SLICE - 1) / TR_PERSIST_SLICE;
    if (slices > TR_PERSIST_MAX_SLICES) slices = TR_PERSIST_MAX_SLICES;
    uint64_t per = (n + slices - 1) / slices;
    rc = tr_cuda_alloc((void **)&dw, (size_t)n * 8);
    if (!rc) rc = tr_cuda_alloc((void **)&dacc, sizeof acc);
    if (!rc) rc = tr_cuda_copy_in(dw, words, (size_t)n * 8);
    if (!rc) { TR_LAUNCH(tr_persist_checksum_kernel, slices, dw, n, slices, per, dacc); rc = tr_cuda_sync(); }
    if (!rc) rc = tr_cuda_copy_out(acc, dacc, sizeof acc);
    tr_cuda_free(dw); tr_cuda_free(dacc);
    if (rc) return rc;
    out[0] = acc[0]; out[1] = acc[1];
    return 0;
}
#ifdef __cplusplus
}
#endif
