/* LIN-003 on a CUDA device: batched leave-one-group-out solves. One work item per (system, prime): it forms the
 * complement normal system of its group modulo its prime directly from the copied group states (total - group g per
 * Gram slot, 192-bit words reduced with tr_lin_reduce_words, the integer row scaling den 10^(s_u + T) and the ridge),
 * then runs LIN-002's modular Gauss-Jordan elimination (tr_lin_modular_rref) in its own output block: the header
 * (rank, pivot columns, pivot rows, D) and the det-scaled reduced row echelon form. One output slot per work item,
 * so the residues equal tr_linupd_logo_residues (math_linupd.c, the same inline tr_linupd_logo_system) bit for bit
 * whatever the thread order. The host part (choice of the agreeing primes, Chinese remaindering within the Hadamard
 * bound, exact verification against the integer systems) is tr_linupd_logo_finish in the engine library.
 * SPDX-License-Identifier: Apache-2.0 */
#include "math_cuda.h"
#include "math_linupd.h"
#include "math_handles.h"

TR_MGPU_ABI(tr_linupd_logo_view, TR_LINUPD_ABI);    /* structs passed to the CUDA library (math_handles.h) */

TR_KERNEL void tr_linupd_logo_kernel(tr_linupd_logo_view v, const tr_gfp *primes, uint64_t nprimes, uint64_t *res) {
    const uint64_t blk = tr_linupd_block(&v);
    TR_GRID_LOOP(i, v.nsys * nprimes) {
        uint64_t s = i / nprimes, q = i % nprimes;
        tr_gfp f = primes[q];
        tr_linupd_logo_system(&v, &f, s, res + i * blk);
    }
}

#ifdef __cplusplus
extern "C" {
#endif
int tr_mgpu_linupd_logo(const tr_linupd_logo_view *view, const tr_gfp *primes, uint64_t nprimes,
                        const tr_math_budget *budget, uint64_t *residues, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!tr_linupd_logo_view_valid(view) || !primes || !residues || !nprimes || nprimes > TR_LINUPD_MAX_PRIMES)
        return TR_MATH_INVALID;
    for (uint64_t q = 0; q < nprimes; q++)
        if (!tr_gfp_consistent(&primes[q]) || primes[q].p < ((uint64_t)1 << 61) || (q && primes[q].p >= primes[q - 1].p))
            return TR_MATH_INVALID;
    tr_math_budget dflt;
    if (!budget) { tr_budget_default(&dflt); budget = &dflt; }
    const uint64_t blk = tr_linupd_block(view), total = view->nsys * nprimes * blk;
    tr_math_need need;
    memset(&need, 0, sizeof need);
    need.need[TR_BUDGET_TENSOR] = tr_budget_mul(tr_budget_mul(view->nsys, nprimes), blk);
    need.need[TR_BUDGET_BYTES] = tr_linupd_logo_device_bytes(view, nprimes);
    int rc = tr_budget_check(budget, &need, refusal);
    if (rc) return rc;
    const uint64_t sbytes = view->ngroups * (uint64_t)view->nslots * 24, tbytes = (uint64_t)view->nslots * 24;
    tr_linupd_logo_view d = *view;
    uint64_t *dsum = NULL, *dtotal = NULL, *dgroups = NULL, *dres = NULL;
    tr_gfp *dprimes = NULL;
    rc = tr_cuda_alloc((void **)&dsum, (size_t)sbytes);
    if (!rc) rc = tr_cuda_alloc((void **)&dtotal, (size_t)tbytes);
    if (!rc) rc = tr_cuda_alloc((void **)&dgroups, (size_t)view->nsys * 8);
    if (!rc) rc = tr_cuda_alloc((void **)&dprimes, (size_t)nprimes * sizeof(tr_gfp));
    if (!rc) rc = tr_cuda_alloc((void **)&dres, (size_t)total * 8);
    if (!rc) rc = tr_cuda_copy_in(dsum, view->sum, (size_t)sbytes);
    if (!rc) rc = tr_cuda_copy_in(dtotal, view->total, (size_t)tbytes);
    if (!rc) rc = tr_cuda_copy_in(dgroups, view->groups, (size_t)view->nsys * 8);
    if (!rc) rc = tr_cuda_copy_in(dprimes, primes, (size_t)nprimes * sizeof(tr_gfp));
    d.sum = dsum; d.total = dtotal; d.groups = dgroups;
    if (!rc) {
        TR_LAUNCH(tr_linupd_logo_kernel, view->nsys * nprimes, d, dprimes, nprimes, dres);
        rc = tr_cuda_sync();
    }
    if (!rc) rc = tr_cuda_copy_out(residues, dres, (size_t)total * 8);
    tr_cuda_free(dsum); tr_cuda_free(dtotal); tr_cuda_free(dgroups); tr_cuda_free(dprimes); tr_cuda_free(dres);
    return rc;
}
#ifdef __cplusplus
}
#endif
