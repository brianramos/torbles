/* SPEC-002 on a CUDA device: the butterfly passes of the fast Walsh-Hadamard transform (tr_spec_walsh,
 * tr_spec_walsh_inverse), with exactly the CPU's results (math_spec.h).
 * SPDX-License-Identifier: Apache-2.0
 *
 * The 2^k values are copied to the device once; pass j is one launch of 2^(k-1) work items, item p updating its own
 * pair (x, x + 2^j) (the pairs of a pass are disjoint, so a pass does not depend on thread order, and the launches
 * order the passes). Each butterfly is checked: an addition or subtraction that would leave the 128-bit range raises
 * an overflow flag (atomic OR) instead of wrapping, and the host then refuses with TR_MATH_OVERFLOW without writing
 * the output. The inverse's canonical dyadic rationals (v / 2^k reduced) are formed on the host from the device's
 * exact integers, with the helper the CPU uses. */
#include "math_cuda.h"
#include "math_spec.h"
#include "math_handles.h"

TR_MGPU_ABI(tr_spec_dyadic, TR_SPEC_ABI);          /* structs passed to the CUDA library (math_handles.h) */
TR_MGPU_ABI(tr_spec_cost, TR_SPEC_ABI);

TR_KERNEL void tr_spec_pass_kernel(tr_math_i128 *v, int32_t j, uint64_t pairs, uint64_t *overflow) {
    TR_GRID_LOOP(p, pairs) {
        uint64_t lo = tr_spec_pair_lo(p, j);
        if (!tr_spec_butterfly(v, lo, lo | ((uint64_t)1 << j))) tr_atomic_or_u64(overflow, 1);
    }
}

/* the device transform of in into the host array res (written only when no butterfly overflowed); the budget is
 * checked before anything is allocated */
static int tr_spec_device_run(int32_t k, const tr_math_i128 *in, tr_math_i128 *res, const tr_math_budget *budget,
                              tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    tr_math_need need;
    if (tr_spec_base_need(k, &need) || !in || !res) return -1;                              /* TR_MATH_INVALID */
    tr_math_budget dflt;
    if (!budget) { tr_budget_default(&dflt); budget = &dflt; }
    int rc = tr_budget_check(budget, &need, refusal);
    if (rc) return rc;
    if (k > TR_SPEC_MAX_K) return -12;                                                      /* TR_MATH_UNSUPPORTED */
    const uint64_t n = (uint64_t)1 << k;
    tr_math_i128 *dv = NULL;
    uint64_t *dflag = NULL, flag = 0;
    rc = tr_cuda_alloc((void **)&dv, (size_t)(n * 16));
    if (!rc) rc = tr_cuda_alloc((void **)&dflag, 8);
    if (!rc) rc = tr_cuda_copy_in(dv, in, (size_t)(n * 16));
    for (int32_t j = 0; j < k && !rc; j++) {                  /* one launch per pass */
        TR_LAUNCH(tr_spec_pass_kernel, n / 2, dv, j, n / 2, dflag);
        rc = tr_cuda_sync();
    }
    if (!rc) rc = tr_cuda_copy_out(&flag, dflag, 8);
    if (!rc && flag) rc = -3;                                                               /* TR_MATH_OVERFLOW */
    if (!rc) rc = tr_cuda_copy_out(res, dv, (size_t)(n * 16));
    tr_cuda_free(dv);
    tr_cuda_free(dflag);
    return rc;
}

#ifdef __cplusplus
extern "C" {
#endif
int tr_mgpu_spec_walsh(int32_t k, const tr_math_i128 *in, tr_math_i128 *out, const tr_math_budget *budget,
                       tr_spec_cost *cost, tr_math_refusal *refusal) {
    int rc = tr_spec_device_run(k, in, out, budget, refusal);
    if (!rc && cost) tr_spec_cost_of(k, TR_SPEC_FORWARD, ((uint64_t)1 << k) * 16, cost);
    return rc;
}

int tr_mgpu_spec_walsh_inverse(int32_t k, const tr_math_i128 *in, tr_spec_dyadic *out, const tr_math_budget *budget,
                               tr_spec_cost *cost, tr_math_refusal *refusal) {
    /* the device's integers land in the first 16 2^k bytes of out (24 per entry), then expand in place from the last
     * entry down: dyadic i (bytes 24 i ..) only covers integers j >= i, which are already converted */
    int rc = tr_spec_device_run(k, in, (tr_math_i128 *)(void *)out, budget, refusal);
    if (rc) return rc;
    const uint64_t n = (uint64_t)1 << k;
    tr_math_i128 *ints = (tr_math_i128 *)(void *)out;
    for (uint64_t i = n; i-- > 0;) {
        tr_math_i128 v;
        memcpy(&v, &ints[i], sizeof v);
        tr_spec_dyadic d = tr_spec_dyadic_of(tr_i128_get(v), k);
        memcpy(&out[i], &d, sizeof d);
    }
    if (cost) tr_spec_cost_of(k, TR_SPEC_INVERSE, n * 16, cost);
    return 0;
}
#ifdef __cplusplus
}
#endif
