/* GEN-001 on a CUDA device: the exact 128-bit convolution of two coefficient arrays and the batched exact evaluation of
 * a 1-D generating polynomial at many rational points, with exactly the results of tr_gen_conv_i128 and
 * tr_gen_eval_i128 (math_gen.h).
 * SPDX-License-Identifier: Apache-2.0
 *
 *   conv  one work item per output coefficient c_k = sum_i a_i b_(k-i) (the shared tr_gen_conv_one): an exact 128-bit
 *         sum, because the host proves min(na, nb) max|a| max|b| < 2^127 before launching (else TR_MATH_OVERFLOW)
 *   eval  one work item per point p / q: the checked 128-bit homogeneous Horner of tr_gen_eval_one, each point with its
 *         own status (0, TR_MATH_OVERFLOW past int128, TR_MATH_NO_SOLUTION at a pole, TR_MATH_INVALID for q < 1)
 * Every output slot has one writer, so nothing depends on the order of the threads. The host budgets the device copies
 * (bytes) and the outputs (output) before allocating. */
#include "math_cuda.h"
#include "math_gen.h"
#include "math_handles.h"

TR_MGPU_ABI(tr_gen_point, TR_GEN_ABI);         /* structs passed to the CUDA library (math_handles.h) */
TR_MGPU_ABI(tr_gen_value, TR_GEN_ABI);

TR_KERNEL void tr_gen_conv_kernel(const int64_t *a, uint64_t na, const int64_t *b, uint64_t nb, uint64_t nout,
                                  int64_t *out) {
    TR_GRID_LOOP(k, nout) {
        __int128 s = tr_gen_conv_one(a, na, b, nb, k);
        out[2 * k] = (int64_t)(uint64_t)s;
        out[2 * k + 1] = (int64_t)(s >> 64);
    }
}

TR_KERNEL void tr_gen_eval_kernel(const int64_t *c, uint64_t n, int64_t shift, const tr_gen_point *z, uint64_t m,
                                  tr_gen_value *out) {
    TR_GRID_LOOP(i, m) tr_gen_eval_one(c, n, shift, z[i], &out[i]);
}

/* tr_budget_check, refusing a size past 64 bits even under an unlimited budget */
static int tr_gen_gpu_check(const tr_math_budget *b, const tr_math_need *n, tr_math_refusal *r) {
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
int tr_mgpu_gen_conv(const int64_t *a, uint64_t na, const int64_t *b, uint64_t nb, tr_math_i128 *out,
                     const tr_math_budget *budget, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!a || !b || !out || !na || !nb) return -1;                                       /* TR_MATH_INVALID */
    const uint64_t nout = na + nb - 1;
    tr_math_need need;
    for (int i = 0; i < TR_BUDGET_RESOURCES; i++) need.need[i] = 0;
    need.need[TR_BUDGET_OUTPUT] = nout;
    need.need[TR_BUDGET_BYTES] = tr_budget_add(tr_budget_mul(tr_budget_add(na, nb), 8), tr_budget_mul(nout, 16));
    tr_math_budget dflt;
    if (!budget) { tr_budget_default(&dflt); budget = &dflt; }
    int rc = tr_gen_gpu_check(budget, &need, refusal);
    if (rc) return rc;
    if (!tr_gen_conv_exact(a, na, b, nb)) return -3;                                    /* TR_MATH_OVERFLOW */
    int64_t *da = NULL, *db = NULL, *dout = NULL;
    rc = tr_cuda_alloc((void **)&da, (size_t)na * 8);
    if (!rc) rc = tr_cuda_alloc((void **)&db, (size_t)nb * 8);
    if (!rc) rc = tr_cuda_alloc((void **)&dout, (size_t)nout * 16);
    if (!rc) rc = tr_cuda_copy_in(da, a, (size_t)na * 8);
    if (!rc) rc = tr_cuda_copy_in(db, b, (size_t)nb * 8);
    if (!rc) {
        TR_LAUNCH(tr_gen_conv_kernel, nout, da, na, db, nb, nout, dout);
        rc = tr_cuda_sync();
    }
    if (!rc) rc = tr_cuda_copy_out(out, dout, (size_t)nout * 16);
    tr_cuda_free(da); tr_cuda_free(db); tr_cuda_free(dout);
    return rc;
}

int tr_mgpu_gen_eval(const int64_t *c, uint64_t n, int64_t shift, const tr_gen_point *z, uint64_t m, tr_gen_value *out,
                     const tr_math_budget *budget, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!c || !n || (m && (!z || !out))) return -1;
    tr_math_need need;
    for (int i = 0; i < TR_BUDGET_RESOURCES; i++) need.need[i] = 0;
    need.need[TR_BUDGET_OUTPUT] = m;
    need.need[TR_BUDGET_BYTES] = tr_budget_add(tr_budget_mul(n, 8), tr_budget_mul(m, sizeof(tr_gen_point) + sizeof(tr_gen_value)));
    tr_math_budget dflt;
    if (!budget) { tr_budget_default(&dflt); budget = &dflt; }
    int rc = tr_gen_gpu_check(budget, &need, refusal);
    if (rc || !m) return rc;
    int64_t *dc = NULL;
    tr_gen_point *dz = NULL;
    tr_gen_value *dout = NULL;
    rc = tr_cuda_alloc((void **)&dc, (size_t)n * 8);
    if (!rc) rc = tr_cuda_alloc((void **)&dz, (size_t)m * sizeof *dz);
    if (!rc) rc = tr_cuda_alloc((void **)&dout, (size_t)m * sizeof *dout);
    if (!rc) rc = tr_cuda_copy_in(dc, c, (size_t)n * 8);
    if (!rc) rc = tr_cuda_copy_in(dz, z, (size_t)m * sizeof *dz);
    if (!rc) {
        TR_LAUNCH(tr_gen_eval_kernel, m, dc, n, shift, dz, m, dout);
        rc = tr_cuda_sync();
    }
    if (!rc) rc = tr_cuda_copy_out(out, dout, (size_t)m * sizeof *dout);
    tr_cuda_free(dc); tr_cuda_free(dz); tr_cuda_free(dout);
    return rc;
}
#ifdef __cplusplus
}
#endif
