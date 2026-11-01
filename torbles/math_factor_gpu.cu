/* PROB-002 on a CUDA device: the factor products and sum-outs / max-outs of every elimination step of a program
 * (math_factor.h), with exactly the raw arena of the CPU loop tr_factor_eliminate (math_factor.c).
 * SPDX-License-Identifier: Apache-2.0
 *
 * tr_mgpu_factor_eliminate: one launch per step (the steps of a program read only earlier messages), one work item per
 * (request, output cell): the shared evaluator tr_factor_cell computes the cell's exact L-limb sum of the products of
 * its operands over the eliminated variable's values (input entries as their 128-bit scaled integers, messages as L-limb
 * values; COUNT-002's exact limb arithmetic) and, for a MAX program, the maximum, its minimum index (argmax ties broken
 * by the minimum index) and the number of maximizing completions, into the cell's own slots. One writer per slot and
 * no atomics on the values, so nothing depends on the order of the threads; an overflow (excluded by the host's L
 * proof) would raise a flag by an atomic OR. Each work item holds 7 L words of scratch in local memory, so L is limited
 * to TR_FACTOR_DEVICE_LIMBS (TR_MATH_UNSUPPORTED beyond). The host checks the program, the bases (every input operand of
 * every request stays inside the inputs) and the budget (bytes = tr_factor_device_bytes, states = R x arena) before
 * allocating anything on the device. */
#include "math_cuda.h"
#include "math_factor.h"
#include "math_handles.h"

TR_MGPU_ABI(tr_factor_step, TR_FACTOR_ABI);       /* structs passed to the CUDA library (math_handles.h) */
TR_MGPU_ABI(tr_factor_operand, TR_FACTOR_ABI);
TR_MGPU_ABI(tr_factor_program, TR_FACTOR_ABI);

TR_KERNEL void tr_factor_step_kernel(tr_factor_program p, uint32_t s, uint64_t R, uint64_t cells, const uint64_t *bases,
                                     uint64_t *vals, uint64_t *maxv, uint64_t *cnts, uint32_t *arg, uint64_t *over) {
    TR_GRID_LOOP(i, R * cells) {
        uint64_t scratch[TR_FACTOR_SCRATCH * TR_FACTOR_DEVICE_LIMBS];
        uint64_t r = i / cells, o = i % cells;
        if (tr_factor_cell(&p, s, r, o, bases, vals, maxv, cnts, arg, scratch)) tr_atomic_or_u64(over, 1);
    }
}

/* tr_budget_check, refusing a size past 64 bits even under an unlimited budget */
static int tr_factor_gpu_check(const tr_math_budget *b, const tr_math_need *n, tr_math_refusal *r) {
    int rc = tr_budget_check(b, n, r);
    if (rc) return rc;
    for (int i = 0; i < TR_BUDGET_RESOURCES; i++)
        if (n->need[i] == TR_BUDGET_UNKNOWN) {
            if (r) { r->resource = i; r->refused = 1; r->need = n->need[i]; r->limit = b->limit[i]; }
            return TR_MATH_BUDGET_CODE;
        }
    return 0;
}

#ifdef __cplusplus
extern "C" {
#endif
int tr_mgpu_factor_eliminate(const tr_factor_program *p, uint64_t R, const uint64_t *bases, uint64_t *vals,
                             uint64_t *maxv, uint64_t *cnts, uint32_t *arg, const tr_math_budget *budget,
                             tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!p || tr_factor_program_check(p) || tr_factor_bases_check(p, R, bases) || (R && !vals) ||
        (R && p->mode == TR_FACTOR_MAX && (!maxv || !cnts || !arg))) return -1;            /* TR_MATH_INVALID */
    if (p->limbs > TR_FACTOR_DEVICE_LIMBS) return -12;                                     /* TR_MATH_UNSUPPORTED */
    tr_math_need need;
    for (int i = 0; i < TR_BUDGET_RESOURCES; i++) need.need[i] = 0;
    need.need[TR_BUDGET_STATES] = tr_budget_mul(R, p->arena);
    need.need[TR_BUDGET_BYTES] = tr_factor_device_bytes(p, R);
    tr_math_budget dflt;
    if (!budget) { tr_budget_default(&dflt); budget = &dflt; }
    int rc = tr_factor_gpu_check(budget, &need, refusal);
    if (rc || !R) return rc;
    const int mx = p->mode == TR_FACTOR_MAX;
    const uint64_t L = p->limbs, entries = R * p->arena, words = entries * L;
    tr_factor_step *dstep = NULL;
    tr_factor_operand *dopd = NULL;
    uint64_t *dstride = NULL, *ddims = NULL, *dinput = NULL, *dbases = NULL, *dvals = NULL, *dmax = NULL, *dcnt = NULL,
             *dover = NULL;
    uint32_t *darg = NULL;
    rc = tr_cuda_alloc((void **)&dstep, (size_t)p->nsteps * sizeof(tr_factor_step));
    if (!rc) rc = tr_cuda_alloc((void **)&dopd, (size_t)(p->nopd ? p->nopd : 1) * sizeof(tr_factor_operand));
    if (!rc) rc = tr_cuda_alloc((void **)&dstride, (size_t)(p->nstrides ? p->nstrides : 1) * 8);
    if (!rc) rc = tr_cuda_alloc((void **)&ddims, (size_t)(p->ndims ? p->ndims : 1) * 8);
    if (!rc) rc = tr_cuda_alloc((void **)&dinput, (size_t)(p->inputs ? p->inputs : 1) * 16);
    if (!rc) rc = tr_cuda_alloc((void **)&dbases, (size_t)(R * p->nfactors ? R * p->nfactors : 1) * 8);
    if (!rc) rc = tr_cuda_alloc((void **)&dvals, (size_t)words * 8);
    if (!rc && mx) rc = tr_cuda_alloc((void **)&dmax, (size_t)words * 8);
    if (!rc && mx) rc = tr_cuda_alloc((void **)&dcnt, (size_t)words * 8);
    if (!rc && mx) rc = tr_cuda_alloc((void **)&darg, (size_t)entries * 4);
    if (!rc) rc = tr_cuda_alloc((void **)&dover, 8);
    if (!rc) rc = tr_cuda_copy_in(dstep, p->step, (size_t)p->nsteps * sizeof(tr_factor_step));
    if (!rc && p->nopd) rc = tr_cuda_copy_in(dopd, p->opd, (size_t)p->nopd * sizeof(tr_factor_operand));
    if (!rc && p->nstrides) rc = tr_cuda_copy_in(dstride, p->stride, (size_t)p->nstrides * 8);
    if (!rc && p->ndims) rc = tr_cuda_copy_in(ddims, p->dims, (size_t)p->ndims * 8);
    if (!rc && p->inputs) rc = tr_cuda_copy_in(dinput, p->input, (size_t)p->inputs * 16);
    if (!rc && p->nfactors) rc = tr_cuda_copy_in(dbases, bases, (size_t)(R * p->nfactors) * 8);
    tr_factor_program dv = *p;
    dv.step = dstep; dv.opd = dopd; dv.stride = dstride; dv.dims = ddims; dv.input = dinput;
    for (uint32_t s = 0; s < p->nsteps && !rc; s++) {           /* the steps in order: later ones read earlier messages */
        const uint64_t cells = p->step[s].cells;
        TR_LAUNCH(tr_factor_step_kernel, R * cells, dv, s, R, cells, (const uint64_t *)dbases, dvals, dmax, dcnt, darg,
                  dover);
        rc = tr_cuda_sync();
    }
    uint64_t over = 0;
    if (!rc) rc = tr_cuda_copy_out(&over, dover, 8);
    if (!rc) rc = tr_cuda_copy_out(vals, dvals, (size_t)words * 8);
    if (!rc && mx) rc = tr_cuda_copy_out(maxv, dmax, (size_t)words * 8);
    if (!rc && mx) rc = tr_cuda_copy_out(cnts, dcnt, (size_t)words * 8);
    if (!rc && mx) rc = tr_cuda_copy_out(arg, darg, (size_t)entries * 4);
    tr_cuda_free(dstep); tr_cuda_free(dopd); tr_cuda_free(dstride); tr_cuda_free(ddims); tr_cuda_free(dinput);
    tr_cuda_free(dbases); tr_cuda_free(dvals); tr_cuda_free(dmax); tr_cuda_free(dcnt); tr_cuda_free(darg);
    tr_cuda_free(dover);
    if (!rc && over) rc = -3;                                                              /* TR_MATH_OVERFLOW */
    return rc;
}
#ifdef __cplusplus
}
#endif
