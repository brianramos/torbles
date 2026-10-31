/* ENV-001 on a CUDA device: batched lookups of the winning group at many parameter values, with exactly the answers
 * of tr_env_lookup_eval (math_env.h).
 * SPDX-License-Identifier: Apache-2.0
 *
 * One work item per query runs the shared tr_env_answer_one: the query's residues modulo the job's primes, then either
 * a binary search over the envelope's sorted points (mode TR_ENV_SEARCH: each comparison the exact sign of an integer
 * polynomial at t from its residues and Garner's mixed-radix digits) or the exact comparison of every class at t (mode
 * TR_ENV_ALL, the verification path). Each work item writes its own answer slot, so the answers do not depend on the
 * order of the threads, and they equal the CPU's bit for bit. The host checks the budget (output = queries, bytes =
 * the device copies) and the view before anything is allocated. */
#include "math_cuda.h"
#include "math_env.h"
#include "math_handles.h"

TR_MGPU_ABI(tr_env_view, 1);             /* structs passed to the CUDA library (math_handles.h) */
TR_MGPU_ABI(tr_env_answer, 1);

TR_KERNEL void tr_env_lookup_kernel(tr_env_view v, tr_env_answer *out) {
    TR_GRID_LOOP(i, v.npoints) tr_env_answer_one(&v, i, &out[i]);
}

#ifdef __cplusplus
extern "C" {
#endif
int tr_mgpu_env_lookup(const tr_env_view *view, tr_env_answer *out, const tr_math_budget *budget,
                       tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!tr_env_view_valid(view) || (view->npoints && !out)) return -1;                       /* TR_MATH_INVALID */
    tr_math_budget dflt;
    if (!budget) { tr_budget_default(&dflt); budget = &dflt; }
    tr_math_need need;
    for (int i = 0; i < TR_BUDGET_RESOURCES; i++) need.need[i] = 0;
    need.need[TR_BUDGET_OUTPUT] = view->npoints;
    need.need[TR_BUDGET_BYTES] = tr_env_view_device_bytes(view);
    int rc = tr_budget_check(budget, &need, refusal);
    if (rc || !view->npoints) return rc;
    tr_env_view d = *view;
    void *dmem[24];
    int nmem = 0;
    const uint64_t k = view->nprimes, nb = view->nbreaks;
#define TR_ENV_COPY(field, bytes) do {                                                                    \
        const size_t _n = (size_t)(bytes);                                                              \
        void *_p = NULL;                                                                                \
        if (!rc && view->field) {                                                                       \
            rc = tr_cuda_alloc(&_p, _n > 0 ? _n : 1);                                                   \
            if (!rc) { dmem[nmem++] = _p; rc = tr_cuda_copy_in(_p, view->field, _n); }                  \
        }                                                                                               \
        d.field = view->field ? (decltype(d.field))_p : NULL;                                           \
    } while (0)
    TR_ENV_COPY(fields, k * sizeof(tr_gfp));
    TR_ENV_COPY(garner, k * k * 8);
    TR_ENV_COPY(half, k * 8);
    TR_ENV_COPY(bkind, nb * 4);
    TR_ENV_COPY(bsign, nb);
    TR_ENV_COPY(bres, k * nb * 8 * TR_ENV_SLOTS);
    TR_ENV_COPY(point_region, nb * 8);
    TR_ENV_COPY(piece_region, (nb + 1) * 8);
    TR_ENV_COPY(rwinner, view->nregions * 8);
    TR_ENV_COPY(rclasses, view->nregions * 8);
    TR_ENV_COPY(cres, view->mode == TR_ENV_ALL ? k * view->nclasses * 32 : 0);
    TR_ENV_COPY(cwinner, view->mode == TR_ENV_ALL ? view->nclasses * 8 : 0);
    TR_ENV_COPY(qsign, view->npoints);
    TR_ENV_COPY(qnum, view->npoints * view->limbs * 8);
    TR_ENV_COPY(qden, view->npoints * view->limbs * 8);
#undef TR_ENV_COPY
    tr_env_answer *dout = NULL;
    const size_t obytes = (size_t)view->npoints * sizeof(tr_env_answer);
    if (!rc) { rc = tr_cuda_alloc((void **)&dout, obytes); if (!rc) dmem[nmem++] = dout; }
    if (!rc) {
        TR_LAUNCH(tr_env_lookup_kernel, view->npoints, d, dout);
        rc = tr_cuda_sync();
    }
    if (!rc) rc = tr_cuda_copy_out(out, dout, obytes);
    for (int i = 0; i < nmem; i++) tr_cuda_free(dmem[i]);
    return rc;
}
#ifdef __cplusplus
}
#endif
