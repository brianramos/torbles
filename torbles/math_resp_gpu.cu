/* RESP-002 on a CUDA device: batched exact loss values and one-sided slopes of many theta requests against a prefix
 * state uploaded once (tr_mgpu_dist_prefix_upload, math_dist_gpu.cu), with exactly the answers of tr_resp_eval.
 * SPDX-License-Identifier: Apache-2.0
 *
 * One work item per request runs the shared tr_resp_eval_one of math_resp.h (two binary searches over the group's
 * support, prefix differences, exact 256-bit products) and writes its own output slot, so the answers do not depend
 * on the order of the threads. The host checks the batch against the budget (output = requests, bytes = the device
 * copies of the requests and answers) before launching. */
#include "math_cuda.h"
#include "math_resp.h"
#include "math_handles.h"

TR_MGPU_ABI(tr_resp_request, 1);            /* structs passed to the CUDA library (math_handles.h) */
TR_MGPU_ABI(tr_resp_result, 1);

TR_KERNEL void tr_resp_eval_kernel(tr_dist_prefix_view v, const tr_resp_request *req, uint64_t n, tr_resp_result *out) {
    TR_GRID_LOOP(i, n) tr_resp_eval_one(&v, &req[i], &out[i]);
}

#ifdef __cplusplus
extern "C" {
#endif
int tr_mgpu_resp_eval(const tr_mgpu_dist_prefix *dp, const tr_resp_request *req, uint64_t n, tr_resp_result *out,
                      const tr_math_budget *budget, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    tr_dist_prefix_view v;
    if (tr_mgpu_dist_prefix_device_view(dp, &v) || (n && (!req || !out))) return -1;      /* TR_MATH_INVALID */
    tr_math_need need;
    for (int i = 0; i < TR_BUDGET_RESOURCES; i++) need.need[i] = 0;
    need.need[TR_BUDGET_OUTPUT] = n;
    need.need[TR_BUDGET_BYTES] = tr_budget_mul(n, sizeof(tr_resp_request) + sizeof(tr_resp_result));
    tr_math_budget dflt;
    if (!budget) { tr_budget_default(&dflt); budget = &dflt; }
    int rc = tr_budget_check(budget, &need, refusal);
    if (!rc && need.need[TR_BUDGET_BYTES] == TR_BUDGET_UNKNOWN) {
        if (refusal) { refusal->resource = TR_BUDGET_BYTES; refusal->refused = 1; refusal->need = TR_BUDGET_UNKNOWN;
                       refusal->limit = budget->limit[TR_BUDGET_BYTES]; }
        rc = TR_MATH_BUDGET_CODE;
    }
    if (rc || !n) return rc;
    tr_resp_request *dreq = NULL;
    tr_resp_result *dres = NULL;
    rc = tr_cuda_alloc((void **)&dreq, (size_t)n * sizeof *dreq);
    if (!rc) rc = tr_cuda_alloc((void **)&dres, (size_t)n * sizeof *dres);
    if (!rc) rc = tr_cuda_copy_in(dreq, req, (size_t)n * sizeof *dreq);
    if (!rc) {
        TR_LAUNCH(tr_resp_eval_kernel, n, v, dreq, n, dres);
        rc = tr_cuda_sync();
    }
    if (!rc) rc = tr_cuda_copy_out(out, dres, (size_t)n * sizeof *dres);
    tr_cuda_free(dreq); tr_cuda_free(dres);
    return rc;
}
#ifdef __cplusplus
}
#endif
