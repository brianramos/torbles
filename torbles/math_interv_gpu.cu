/* INV-002 on a CUDA device: the batched exhaustive evaluation of a finite intervention grid in cost order, with
 * exactly the results of tr_interv_grid_eval (math_interv.h). Launch 1: one work item per rank evaluates every
 * condition by its exact sign (residues modulo the verified primes, Garner's digits) and the integer cost of the
 * point, sets its bit of the feasibility bitmap by atomic OR and lowers the minimal feasible cost by atomic minimum.
 * Launch 2: one work item per rank sets its tie bit (feasible and of the matched cost) by atomic OR. Bits and the
 * minimum do not depend on the order of the threads. The plan, the order and the search loop are host code
 * (math_interv.c); the continuous methods (exact LPs, roots, aggregates) run on the CPU.
 * SPDX-License-Identifier: Apache-2.0 */
#include "math_cuda.h"
#include "math_interv.h"
#include "math_handles.h"

TR_MGPU_ABI(tr_interv_grid_view, 1);     /* structs passed to the CUDA library (math_handles.h) */

TR_KERNEL void tr_interv_grid_kernel(tr_interv_grid_view v, uint64_t start, uint64_t count, uint64_t *feasible,
                                     int64_t *mincost) {
    TR_GRID_LOOP(i, count) {
        int64_t cost = 0;
        if (tr_interv_grid_point(&v, v.order[i], &cost)) {        /* the device order holds this chunk's slice */
            tr_atomic_or_u64(&feasible[i >> 6], (uint64_t)1 << (i & 63));
            tr_atomic_min_i64(mincost, cost);
        }
    }
    (void)start;
}
TR_KERNEL void tr_interv_ties_kernel(tr_interv_grid_view v, uint64_t count, const uint64_t *feasible,
                                     const int64_t *match, uint64_t *ties) {
    TR_GRID_LOOP(i, count) {
        if (!((feasible[i >> 6] >> (i & 63)) & 1)) continue;
        uint64_t dig[TR_INTERV_GRID_DIMS];
        tr_interv_digits(&v, v.order[i], dig);
        if (tr_interv_point_cost(&v, dig) == *match) tr_atomic_or_u64(&ties[i >> 6], (uint64_t)1 << (i & 63));
    }
}

#ifdef __cplusplus
extern "C" {
#endif
int tr_mgpu_interv_grid(const tr_interv_grid_view *view, uint64_t start, uint64_t count, int64_t match,
                        uint64_t *feasible, uint64_t *ties, int64_t *mincost, const tr_math_budget *budget,
                        tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!tr_interv_grid_view_valid(view) || start > view->total || count > view->total - start || !mincost ||
        (count && (!feasible || !ties))) return TR_MATH_INVALID;
    tr_math_budget dflt;
    if (!budget) { tr_budget_default(&dflt); budget = &dflt; }
    tr_math_need need;
    for (int i = 0; i < TR_BUDGET_RESOURCES; i++) need.need[i] = 0;
    need.need[TR_BUDGET_OUTPUT] = count;
    need.need[TR_BUDGET_BYTES] = tr_interv_grid_device_bytes(view, count);
    int rc = tr_budget_check(budget, &need, refusal);
    if (rc) return rc;
    const uint64_t k = (uint64_t)view->nprimes, words = (count + 63) / 64;
    if (!count) { *mincost = INT64_MAX; return 0; }
    tr_gfp *dfields = NULL;
    uint64_t *dgarner = NULL, *dhalf = NULL, *dcres = NULL, *dvres = NULL, *dorder = NULL, *dfe = NULL, *dti = NULL;
    uint8_t *dexps = NULL;
    int64_t *dctab = NULL, *dmin = NULL;
    rc = tr_cuda_alloc((void **)&dfields, (size_t)(k * sizeof(tr_gfp)));
    if (!rc) rc = tr_cuda_alloc((void **)&dgarner, (size_t)(k * k * 8));
    if (!rc) rc = tr_cuda_alloc((void **)&dhalf, (size_t)(k * 8));
    if (!rc) rc = tr_cuda_alloc((void **)&dexps, (size_t)((uint64_t)view->nterms * (uint64_t)view->ndims));
    if (!rc) rc = tr_cuda_alloc((void **)&dcres, (size_t)(k * view->nterms * 8));
    if (!rc) rc = tr_cuda_alloc((void **)&dvres, (size_t)(k * view->nvalues * 8));
    if (!rc) rc = tr_cuda_alloc((void **)&dctab, (size_t)(view->nvalues * 8));
    if (!rc) rc = tr_cuda_alloc((void **)&dorder, (size_t)(count * 8));
    if (!rc) rc = tr_cuda_alloc((void **)&dfe, (size_t)(words * 8));
    if (!rc) rc = tr_cuda_alloc((void **)&dti, (size_t)(words * 8));
    if (!rc) rc = tr_cuda_alloc((void **)&dmin, 8);
    if (!rc) rc = tr_cuda_copy_in(dfields, view->fields, (size_t)(k * sizeof(tr_gfp)));
    if (!rc) rc = tr_cuda_copy_in(dgarner, view->garner, (size_t)(k * k * 8));
    if (!rc) rc = tr_cuda_copy_in(dhalf, view->half, (size_t)(k * 8));
    if (!rc && view->nterms) rc = tr_cuda_copy_in(dexps, view->exps, (size_t)((uint64_t)view->nterms * (uint64_t)view->ndims));
    if (!rc && view->nterms) rc = tr_cuda_copy_in(dcres, view->cres, (size_t)(k * view->nterms * 8));
    if (!rc) rc = tr_cuda_copy_in(dvres, view->vres, (size_t)(k * view->nvalues * 8));
    if (!rc) rc = tr_cuda_copy_in(dctab, view->ctab, (size_t)(view->nvalues * 8));
    if (!rc) rc = tr_cuda_copy_in(dorder, view->order + start, (size_t)(count * 8));
    int64_t init = INT64_MAX;
    if (!rc) rc = tr_cuda_copy_in(dmin, &init, 8);
    tr_interv_grid_view dv = *view;                               /* the view with its pointers swapped for device copies */
    dv.fields = dfields; dv.garner = dgarner; dv.half = dhalf; dv.exps = dexps; dv.cres = dcres; dv.vres = dvres;
    dv.ctab = dctab; dv.order = dorder; dv.total = count;
    if (!rc) { TR_LAUNCH(tr_interv_grid_kernel, count, dv, start, count, dfe, dmin); rc = tr_cuda_sync(); }
    int64_t best = INT64_MAX;
    if (!rc) rc = tr_cuda_copy_out(&best, dmin, 8);
    int64_t m = match == INT64_MIN ? best : match;
    if (!rc) rc = tr_cuda_copy_in(dmin, &m, 8);
    if (!rc) { TR_LAUNCH(tr_interv_ties_kernel, count, dv, count, dfe, dmin, dti); rc = tr_cuda_sync(); }
    if (!rc) rc = tr_cuda_copy_out(feasible, dfe, (size_t)(words * 8));
    if (!rc) rc = tr_cuda_copy_out(ties, dti, (size_t)(words * 8));
    if (!rc) *mincost = best;
    tr_cuda_free(dfields); tr_cuda_free(dgarner); tr_cuda_free(dhalf); tr_cuda_free(dexps); tr_cuda_free(dcres);
    tr_cuda_free(dvres); tr_cuda_free(dctab); tr_cuda_free(dorder); tr_cuda_free(dfe); tr_cuda_free(dti);
    tr_cuda_free(dmin);
    return rc;
}
#ifdef __cplusplus
}
#endif
