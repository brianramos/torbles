/* OPT-002 on a CUDA device: the batched enclosure of branch-and-bound boxes, one work item per box, with exactly the
 * results of tr_polyopt_boxes (math_polyopt.h: the natural interval extension intersected with the centered form,
 * FND-009's contraction-safe directed operations; a box is pruned only when its enclosures prove it). Each work item
 * writes its own output slot and uses its own slice of the scratch, so the results do not depend on thread order.
 * The exact simplex, the active-set method and the bookkeeping of the search stay on the CPU (math_polyopt.c).
 * SPDX-License-Identifier: Apache-2.0 */
#include "math_cuda.h"
#include "math_polyopt.h"
#include "math_handles.h"

TR_MGPU_ABI(tr_polyopt_encl_view, 1);    /* structs passed to the CUDA library (math_handles.h) */
TR_MGPU_ABI(tr_polyopt_box_out, 1);

TR_KERNEL void tr_polyopt_boxes_kernel(tr_polyopt_encl_view v, uint64_t nboxes, const double *xlo, const double *xhi,
                                       double ubound, tr_math_interval *scratch, tr_polyopt_box_out *out) {
    TR_GRID_LOOP(i, nboxes)
        tr_polyopt_box(&v, xlo + i * (uint64_t)v.n, xhi + i * (uint64_t)v.n, ubound,
                       scratch + i * (uint64_t)TR_POLYOPT_DENSE_MAX, &out[i]);
}

#ifdef __cplusplus
extern "C" {
#endif
int tr_mgpu_polyopt_boxes(const tr_polyopt_encl_view *view, uint64_t nboxes, const double *xlo, const double *xhi,
                          double ubound, tr_polyopt_box_out *out, const tr_math_budget *budget,
                          tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!tr_polyopt_view_valid(view) || (nboxes && (!xlo || !xhi || !out)) || ubound != ubound) return TR_MATH_INVALID;
    for (uint64_t i = 0; i < nboxes * (uint64_t)view->n; i++) if (!(xlo[i] <= xhi[i])) return TR_MATH_INVALID;
    tr_math_budget dflt;
    if (!budget) { tr_budget_default(&dflt); budget = &dflt; }
    tr_math_need need;
    for (int i = 0; i < TR_BUDGET_RESOURCES; i++) need.need[i] = 0;
    need.need[TR_BUDGET_OUTPUT] = nboxes;
    need.need[TR_BUDGET_BYTES] = tr_polyopt_device_bytes(view, nboxes);
    int rc = tr_budget_check(budget, &need, refusal);
    if (rc || !nboxes) return rc;
    const uint64_t ncoef = view->offset[view->npolys], nx = nboxes * (uint64_t)view->n;
    double *dclo = NULL, *dchi = NULL, *dxlo = NULL, *dxhi = NULL;
    tr_math_interval *dscratch = NULL;
    tr_polyopt_box_out *dout = NULL;
    rc = tr_cuda_alloc((void **)&dclo, (size_t)(ncoef * 8));
    if (!rc) rc = tr_cuda_alloc((void **)&dchi, (size_t)(ncoef * 8));
    if (!rc) rc = tr_cuda_alloc((void **)&dxlo, (size_t)(nx * 8));
    if (!rc) rc = tr_cuda_alloc((void **)&dxhi, (size_t)(nx * 8));
    if (!rc) rc = tr_cuda_alloc((void **)&dscratch, (size_t)(nboxes * TR_POLYOPT_DENSE_MAX * sizeof(tr_math_interval)));
    if (!rc) rc = tr_cuda_alloc((void **)&dout, (size_t)(nboxes * sizeof(tr_polyopt_box_out)));
    if (!rc) rc = tr_cuda_copy_in(dclo, view->clo, (size_t)(ncoef * 8));
    if (!rc) rc = tr_cuda_copy_in(dchi, view->chi, (size_t)(ncoef * 8));
    if (!rc) rc = tr_cuda_copy_in(dxlo, xlo, (size_t)(nx * 8));
    if (!rc) rc = tr_cuda_copy_in(dxhi, xhi, (size_t)(nx * 8));
    if (!rc) {
        tr_polyopt_encl_view dv = *view;                        /* the view with its pointers swapped for device copies */
        dv.clo = dclo; dv.chi = dchi;
        TR_LAUNCH(tr_polyopt_boxes_kernel, nboxes, dv, nboxes, dxlo, dxhi, ubound, dscratch, dout);
        rc = tr_cuda_sync();
    }
    if (!rc) rc = tr_cuda_copy_out(out, dout, (size_t)(nboxes * sizeof(tr_polyopt_box_out)));
    tr_cuda_free(dclo); tr_cuda_free(dchi); tr_cuda_free(dxlo); tr_cuda_free(dxhi);
    tr_cuda_free(dscratch); tr_cuda_free(dout);
    return rc;
}
#ifdef __cplusplus
}
#endif
