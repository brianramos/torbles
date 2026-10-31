/* PAIR-001 on a CUDA device: batched pairwise summaries (absolute-difference sums, Mann-Whitney U / AUC, KS,
 * 1-Wasserstein, Cramer-von Mises) against prefix states uploaded once (tr_mgpu_dist_prefix_upload, math_dist_gpu.cu),
 * with exactly the answers of tr_pair_query (math_pair.h).
 * SPDX-License-Identifier: Apache-2.0
 *
 * One work item per request runs the shared tr_pair_eval of math_pair.h (one pass over a class's sorted support, or
 * one merge of two supports, with exact 256-bit sums) and writes its own output slot, so the answers do not depend on
 * the order of the threads. The host checks the batch against the budget (output = requests, bytes = the device
 * copies of the requests and answers) before launching. */
#include "math_cuda.h"
#include "math_pair.h"
#include "math_handles.h"

TR_MGPU_ABI(tr_pair_request, TR_PAIR_ABI);     /* structs passed to the CUDA library (math_handles.h) */
TR_MGPU_ABI(tr_pair_result, TR_PAIR_ABI);

TR_KERNEL void tr_pair_query_kernel(tr_dist_prefix_view a, tr_dist_prefix_view b, const tr_pair_request *req,
                                    uint64_t n, tr_pair_result *out) {
    TR_GRID_LOOP(i, n) tr_pair_eval(&a, &b, &req[i], &out[i]);
}

#ifdef __cplusplus
extern "C" {
#endif
int tr_mgpu_pair_query(const tr_mgpu_dist_prefix *da, const tr_mgpu_dist_prefix *db, const tr_pair_request *req,
                       uint64_t n, tr_pair_result *out, const tr_math_budget *budget, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    tr_dist_prefix_view va, vb;
    if (tr_mgpu_dist_prefix_device_view(da, &va) || (db && tr_mgpu_dist_prefix_device_view(db, &vb)) ||
        (n && (!req || !out))) return -1;                                                 /* TR_MATH_INVALID */
    if (!db) vb = va;
    tr_math_need need;
    for (int i = 0; i < TR_BUDGET_RESOURCES; i++) need.need[i] = 0;
    need.need[TR_BUDGET_OUTPUT] = n;
    need.need[TR_BUDGET_BYTES] = tr_budget_mul(n, sizeof(tr_pair_request) + sizeof(tr_pair_result));
    tr_math_budget dflt;
    if (!budget) { tr_budget_default(&dflt); budget = &dflt; }
    int rc = tr_budget_check(budget, &need, refusal);
    if (!rc && need.need[TR_BUDGET_BYTES] == TR_BUDGET_UNKNOWN) {
        if (refusal) { refusal->resource = TR_BUDGET_BYTES; refusal->refused = 1; refusal->need = TR_BUDGET_UNKNOWN;
                       refusal->limit = budget->limit[TR_BUDGET_BYTES]; }
        rc = TR_MATH_BUDGET_CODE;
    }
    if (rc || !n) return rc;
    tr_pair_request *dreq = NULL;
    tr_pair_result *dres = NULL;
    rc = tr_cuda_alloc((void **)&dreq, (size_t)n * sizeof *dreq);
    if (!rc) rc = tr_cuda_alloc((void **)&dres, (size_t)n * sizeof *dres);
    if (!rc) rc = tr_cuda_copy_in(dreq, req, (size_t)n * sizeof *dreq);
    if (!rc) {
        TR_LAUNCH(tr_pair_query_kernel, n, va, vb, dreq, n, dres);
        rc = tr_cuda_sync();
    }
    if (!rc) rc = tr_cuda_copy_out(out, dres, (size_t)n * sizeof *dres);
    tr_cuda_free(dreq); tr_cuda_free(dres);
    return rc;
}
#ifdef __cplusplus
}
#endif
