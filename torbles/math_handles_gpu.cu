/* FND-013 on a CUDA device: immutable sharing. One immutable sparse integer polynomial is copied once per call into
 * read-only device buffers and evaluated by every thread at its own points, exactly (tr_ipoly_point of
 * math_handles.h: 128-bit values, overflow flagged per point, never wrapped). Each thread writes only its own output
 * slots and adds to one atomic overflow counter, so the result does not depend on thread order. The host keeps the
 * polynomial's handle retained (tr_math_handles_ipoly_view ... tr_math_handles_release) around this call, so a
 * concurrent close cannot free the terms while they are copied.
 * Also the ABI registry of the CUDA library: the structs its entry points receive, as compiled here.
 * SPDX-License-Identifier: Apache-2.0 */
#include "math_cuda.h"
#include "math_handles.h"
#include "math_disc.h"

/* the structs that cross the ctypes boundary into this library (one line each; later families add theirs in their
 * own math_*_gpu.cu) */
TR_MGPU_ABI(tr_math_budget, 1);          /* math_budgets_gpu.cu, math_disc_gpu.cu, this file */
TR_MGPU_ABI(tr_math_refusal, 1);
TR_MGPU_ABI(tr_disc_view, 1);            /* math_disc_gpu.cu */
TR_MGPU_ABI(tr_disc_evidence, 1);
TR_MGPU_ABI(tr_math_ipoly_view, TR_HANDLES_ABI);   /* this file */
TR_MGPU_ABI(tr_math_i128, 1);
TR_MGPU_ABI(tr_math_abi_entry, TR_HANDLES_ABI);

TR_KERNEL void tr_handles_eval_kernel(tr_math_ipoly_view p, const int64_t *points, uint64_t npoints,
                                      tr_math_i128 *values, uint8_t *flags, uint64_t *overflowed) {
    TR_GRID_LOOP(i, npoints) {
        __int128 value = 0;
        int ok = tr_ipoly_point(&p, points + i * (uint64_t)p.nvars, &value);
        values[i] = tr_i128_set(ok ? value : 0);
        flags[i] = (uint8_t)!ok;
        if (!ok) tr_atomic_add_u64(overflowed, 1);
    }
}

#ifdef __cplusplus
extern "C" {
#endif
extern const tr_math_abi_entry __start_tr_mgpu_abi[] __attribute__((visibility("hidden")));
extern const tr_math_abi_entry __stop_tr_mgpu_abi[] __attribute__((visibility("hidden")));

/* every TR_MGPU_ABI entry linked into this library; same contract as tr_math_handles_abi */
int tr_mgpu_handles_abi(tr_math_abi_entry *out, int capacity, unsigned entry_size) {
    if (entry_size != sizeof(tr_math_abi_entry) || capacity < 0 || (capacity && !out)) return -1;
    int n = (int)(__stop_tr_mgpu_abi - __start_tr_mgpu_abi);
    for (int i = 0; i < n && i < capacity; i++) out[i] = __start_tr_mgpu_abi[i];
    return n;
}

/* the contract of tr_math_handles_view_evaluate, on the device. Budget (NULL = defaults): output = npoints, bytes =
 * the device copies (tr_handles_device_bytes), checked before anything is allocated. */
int tr_mgpu_handles_evaluate(const tr_math_ipoly_view *view, const int64_t *points, uint64_t npoints,
                             tr_math_i128 *values, uint8_t *flags, uint64_t *overflowed,
                             const tr_math_budget *budget, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!view || view->nvars < 0 || (view->nterms && (!view->exponents || !view->coefficients)) || !overflowed ||
        (npoints && (!points || !values || !flags))) return -1;
    tr_math_budget dflt;
    if (!budget) { tr_budget_default(&dflt); budget = &dflt; }
    tr_math_need need;
    memset(&need, 0, sizeof need);
    need.need[TR_BUDGET_OUTPUT] = npoints;
    need.need[TR_BUDGET_BYTES] = tr_handles_device_bytes(view, npoints);
    int rc = tr_budget_check(budget, &need, refusal);
    if (rc) return rc;
    size_t nv = (size_t)view->nvars, nt = (size_t)view->nterms, np = (size_t)npoints;
    uint8_t *de = NULL, *dflags = NULL;
    int64_t *dc = NULL, *dpoints = NULL;
    tr_math_i128 *dvalues = NULL;
    uint64_t *dcount = NULL, count = 0;
    /* the immutable state: copied once, read by every thread */
    rc = tr_cuda_alloc((void **)&de, nt * nv);
    if (!rc) rc = tr_cuda_alloc((void **)&dc, nt * sizeof(int64_t));
    if (!rc) rc = tr_cuda_copy_in(de, view->exponents, nt * nv);
    if (!rc) rc = tr_cuda_copy_in(dc, view->coefficients, nt * sizeof(int64_t));
    if (!rc) rc = tr_cuda_alloc((void **)&dpoints, np * nv * sizeof(int64_t));
    if (!rc) rc = tr_cuda_copy_in(dpoints, points, np * nv * sizeof(int64_t));
    if (!rc) rc = tr_cuda_alloc((void **)&dvalues, np * sizeof(tr_math_i128));
    if (!rc) rc = tr_cuda_alloc((void **)&dflags, np);
    if (!rc) rc = tr_cuda_alloc((void **)&dcount, sizeof count);
    if (!rc && npoints) {
        tr_math_ipoly_view d = *view;
        d.exponents = de;
        d.coefficients = dc;
        TR_LAUNCH(tr_handles_eval_kernel, npoints, d, dpoints, npoints, dvalues, dflags, dcount);
        rc = tr_cuda_sync();
    }
    if (!rc) rc = tr_cuda_copy_out(values, dvalues, np * sizeof(tr_math_i128));
    if (!rc) rc = tr_cuda_copy_out(flags, dflags, np);
    if (!rc) rc = tr_cuda_copy_out(&count, dcount, sizeof count);
    tr_cuda_free(de); tr_cuda_free(dc); tr_cuda_free(dpoints); tr_cuda_free(dvalues); tr_cuda_free(dflags);
    tr_cuda_free(dcount);
    if (rc) return rc;
    *overflowed = count;
    return 0;
}
#ifdef __cplusplus
}
#endif
