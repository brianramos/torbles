/* ENV-002 on a CUDA device: batched point location of many parameter values against the regions, the index or every
 * class, with exactly the answers of tr_region_lookup_eval (math_region.h).
 * SPDX-License-Identifier: Apache-2.0
 *
 * One work item per value runs the shared tr_region_answer_one: the value's residues modulo the job's primes, the
 * walls of P, the value's cell, then the mode's scan (INDEX: the kd-tree descent and the leaf's candidates; REGIONS:
 * every region of the cell against its irredundant description; ALL: every class of the cell), each comparison the
 * exact sign of an integer from its residues and Garner's mixed-radix digits. Each work item writes its own answer and
 * tie slots, so the answers do not depend on the order of the threads, and they equal the CPU's bit for bit. Ties are
 * resolved by the declared policy inside the work item (the smallest or largest member group of the tied classes),
 * independently of any other work item. The host checks the view and the budget (output = values, bytes = the device
 * copies) before anything is allocated. */
#include "math_cuda.h"
#include "math_region.h"
#include "math_handles.h"

TR_MGPU_ABI(tr_region_view, 1);          /* structs passed to the CUDA library (math_handles.h) */
TR_MGPU_ABI(tr_region_answer, 1);

TR_KERNEL void tr_region_locate_kernel(tr_region_view v, tr_region_answer *out, uint64_t *ties) {
    TR_GRID_LOOP(i, v.npoints) tr_region_answer_one(&v, i, &out[i], v.max_ties ? ties + i * v.max_ties : NULL);
}

#ifdef __cplusplus
extern "C" {
#endif
int tr_mgpu_region_locate(const tr_region_view *view, tr_region_answer *out, uint64_t *ties,
                          const tr_math_budget *budget, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!tr_region_view_valid(view) || (view->npoints && !out) || (view->max_ties && view->npoints && !ties))
        return -1;                                                                         /* TR_MATH_INVALID */
    tr_math_budget dflt;
    if (!budget) { tr_budget_default(&dflt); budget = &dflt; }
    tr_math_need need;
    for (int i = 0; i < TR_BUDGET_RESOURCES; i++) need.need[i] = 0;
    need.need[TR_BUDGET_OUTPUT] = view->npoints;
    need.need[TR_BUDGET_BYTES] = tr_region_view_device_bytes(view);
    int rc = tr_budget_check(budget, &need, refusal);
    if (rc || !view->npoints) return rc;
    tr_region_view d = *view;
    void *dmem[32];
    int nmem = 0;
    const uint64_t k = (uint64_t)(view->nprimes > 0 ? view->nprimes : 0), W = (uint64_t)view->d + 1;
    const uint64_t nb = view->pieces > 1 ? (uint64_t)view->d * (uint64_t)(view->pieces - 1) : 0;
#define TR_REGION_COPY(field, bytes) do {                                                                 \
        const size_t _n = (size_t)(bytes);                                                              \
        void *_p = NULL;                                                                                \
        if (!rc && view->field && _n) {                                                                 \
            rc = tr_cuda_alloc(&_p, _n);                                                                \
            if (!rc) { dmem[nmem++] = _p; rc = tr_cuda_copy_in(_p, view->field, _n); }                  \
        }                                                                                               \
        d.field = (view->field && _n) ? (decltype(d.field))_p : NULL;                                   \
    } while (0)
    if (view->kind == TR_REGION_FINITE) {
        TR_REGION_COPY(fields, k * sizeof(tr_gfp));
        TR_REGION_COPY(garner, k * k * 8);
        TR_REGION_COPY(half, k * 8);
        TR_REGION_COPY(wres, k * (uint64_t)view->nwalls * W * 8);
        TR_REGION_COPY(gres, k * nb * 16);
        TR_REGION_COPY(cell_root, view->ncells * 8);
        TR_REGION_COPY(cell_class, (view->ncells + 1) * 8);
        TR_REGION_COPY(cell_region, (view->ncells + 1) * 8);
        TR_REGION_COPY(naxis, view->nnodes * 4);
        TR_REGION_COPY(nlink, view->nnodes * 16);
        TR_REGION_COPY(nres, k * view->nnodes * 16);
        TR_REGION_COPY(cand, view->ncand * 8);
        TR_REGION_COPY(cres, k * view->nclasses * (W + 1) * 8);
        TR_REGION_COPY(cwinner, view->nclasses * 8);
        TR_REGION_COPY(cregion, view->nclasses * 8);
        TR_REGION_COPY(rclass, view->nregions * 8);
        TR_REGION_COPY(rcons, (view->nregions + 1) * 8);
        TR_REGION_COPY(ceq, view->ncons);
        TR_REGION_COPY(kres, k * view->ncons * W * 8);
        TR_REGION_COPY(qsign, view->npoints * (uint64_t)view->d);
        TR_REGION_COPY(qnum, view->npoints * (uint64_t)view->d * (uint64_t)view->limbs * 8);
        TR_REGION_COPY(qden, view->npoints * (uint64_t)view->d * (uint64_t)view->limbs * 8);
    }
#undef TR_REGION_COPY
    tr_region_answer *dout = NULL;
    uint64_t *dties = NULL;
    const size_t obytes = (size_t)view->npoints * sizeof(tr_region_answer);
    const size_t tbytes = (size_t)(view->npoints * view->max_ties * 8);
    if (!rc) { rc = tr_cuda_alloc((void **)&dout, obytes); if (!rc) dmem[nmem++] = dout; }
    if (!rc && tbytes) { rc = tr_cuda_alloc((void **)&dties, tbytes); if (!rc) dmem[nmem++] = dties; }
    if (!rc) {
        TR_LAUNCH(tr_region_locate_kernel, view->npoints, d, dout, dties);
        rc = tr_cuda_sync();
    }
    if (!rc) rc = tr_cuda_copy_out(out, dout, obytes);
    if (!rc && tbytes) rc = tr_cuda_copy_out(ties, dties, tbytes);
    for (int i = 0; i < nmem; i++) tr_cuda_free(dmem[i]);
    return rc;
}
#ifdef __cplusplus
}
#endif
