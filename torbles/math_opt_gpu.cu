/* OPT-001 on a CUDA device: the face filter of a box-constrained quadratic, one work item per face, with exactly the
 * statuses of tr_opt_faces (math_opt.h: interval Gaussian elimination with outward rounding on enclosures of A, b
 * and the bounds; a face is discarded only when it rigorously cannot hold the optimum). One output slot per face, so
 * the statuses do not depend on thread order. The exact enumeration of the surviving faces is tr_opt_minimize.
 * SPDX-License-Identifier: Apache-2.0 */
#include "math_cuda.h"
#include "math_opt.h"
#include "math_handles.h"

TR_MGPU_ABI(tr_opt_face_view, 1);        /* structs passed to the CUDA library (math_handles.h) */

TR_KERNEL void tr_opt_faces_kernel(tr_opt_face_view v, uint8_t *status) {
    TR_GRID_LOOP(f, v.faces) status[f] = (uint8_t)tr_opt_face_filter(&v, f);
}

#ifdef __cplusplus
extern "C" {
#endif
int tr_mgpu_opt_faces(const tr_opt_face_view *view, const tr_math_budget *budget, uint8_t *status, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!view || !status || view->n < 1 || view->n > TR_OPT_MAX_BOX || view->nk < 0 || view->nk > view->n) return -1;
    uint64_t faces = 1;
    for (int32_t k = 0; k < view->nk; k++) {
        if (view->kidx[k] < 0 || view->kidx[k] >= view->n || (k && view->kidx[k] <= view->kidx[k - 1])) return -1;
        faces *= 3;
    }
    if (faces != view->faces) return -1;
    tr_math_budget dflt;
    if (!budget) { tr_budget_default(&dflt); budget = &dflt; }
    tr_math_need need;
    for (int i = 0; i < TR_BUDGET_RESOURCES; i++) need.need[i] = 0;
    need.need[TR_BUDGET_VARIABLES] = (uint64_t)view->n;
    need.need[TR_BUDGET_ITERATIONS] = faces;
    need.need[TR_BUDGET_BYTES] = tr_opt_faces_bytes(view);
    int rc = tr_budget_check(budget, &need, refusal);
    if (rc) return rc;
    uint8_t *dstatus = NULL;
    rc = tr_cuda_alloc((void **)&dstatus, (size_t)faces);
    if (!rc) { TR_LAUNCH(tr_opt_faces_kernel, faces, *view, dstatus); rc = tr_cuda_sync(); }
    if (!rc) rc = tr_cuda_copy_out(status, dstatus, (size_t)faces);
    tr_cuda_free(dstatus);
    return rc;
}
#ifdef __cplusplus
}
#endif
