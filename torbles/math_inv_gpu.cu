/* INV-001 on a CUDA device: the data-parallel parts of the inverse answers, with exactly the results of the CPU
 * functions of math_inv.c (tr_inv_signs_eval, tr_inv_sweep_eval):
 *   tr_mgpu_inv_signs   batched exact signs of integer polynomials at rational points (the refinement of isolating
 *                       intervals): one work item per point, one output slot each
 *   tr_mgpu_inv_sweep   the exhaustive finite-domain search: one work item per grid point, an exact zero test, the
 *                       matches set in a bitmap with atomic OR (the bitmap does not depend on thread order)
 * Both evaluate with the shared inline functions of math_inv.h (residues modulo primes in (2^61, 2^62) with Barrett
 * reduction, Garner digits for the sign), so the device and the CPU agree bit for bit; the host checks the views and
 * budgets the device copies (resource bytes, tr_inv_signs_device_bytes / tr_inv_sweep_device_bytes) before
 * allocating anything.
 * SPDX-License-Identifier: Apache-2.0 */
#include "math_cuda.h"
#include "math_inv.h"
#include "math_handles.h"

TR_MGPU_ABI(tr_inv_signs_view, TR_INV_ABI);   /* structs passed to the CUDA library (math_handles.h) */
TR_MGPU_ABI(tr_inv_sweep_view, TR_INV_ABI);

TR_KERNEL void tr_inv_signs_kernel(tr_inv_signs_view v, int8_t *out) {
    TR_GRID_LOOP(i, v.npoints) out[i] = (int8_t)tr_inv_sign_point(&v, i);
}

TR_KERNEL void tr_inv_sweep_kernel(tr_inv_sweep_view v, uint64_t *bits) {
    TR_GRID_LOOP(i, v.count)
        if (tr_inv_sweep_point(&v, v.start + i)) tr_atomic_or_u64(&bits[i >> 6], (uint64_t)1 << (i & 63));
}

#define TR_INV_DEV(ptr, bytes) do { if (!rc) { rc = tr_cuda_alloc((void **)&(ptr), (size_t)(bytes)); if (!rc) mem[nmem++] = (void *)(ptr); } } while (0)
#define TR_INV_IN(dst, src, bytes) do { if (!rc && (bytes) != 0) rc = tr_cuda_copy_in((void *)(dst), (const void *)(src), (size_t)(bytes)); } while (0)

#ifdef __cplusplus
extern "C" {
#endif
static const tr_math_budget *tr_inv_budget(const tr_math_budget *b, tr_math_budget *store) {
    if (b) return b;
    tr_budget_default(store);
    return store;
}

int tr_mgpu_inv_signs(const tr_inv_signs_view *v, int8_t *signs, const tr_math_budget *budget,
                      tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!tr_inv_signs_view_valid(v) || (v->npoints && !signs)) return TR_MATH_INVALID;
    tr_math_budget dflt;
    budget = tr_inv_budget(budget, &dflt);
    tr_math_need need;
    memset(&need, 0, sizeof need);
    need.need[TR_BUDGET_OUTPUT] = v->npoints;
    need.need[TR_BUDGET_BYTES] = tr_inv_signs_device_bytes(v);
    int rc = tr_budget_check(budget, &need, refusal);
    if (rc || !v->npoints) return rc;
    uint64_t k = v->nprimes, n = v->npoints, cells = n * (uint64_t)v->limbs;
    tr_inv_signs_view d = *v;
    void *mem[12];
    int nmem = 0;
    tr_gfp *dfields = NULL;
    uint64_t *dgarner = NULL, *dhalf = NULL, *dcoef = NULL, *dnum = NULL, *dden = NULL;
    uint32_t *doffset = NULL, *dpoly = NULL;
    int8_t *dsign = NULL, *dout = NULL;
    TR_INV_DEV(dfields, k * sizeof(tr_gfp));
    TR_INV_DEV(dgarner, k * k * 8);
    TR_INV_DEV(dhalf, k * 8);
    TR_INV_DEV(doffset, ((uint64_t)v->npolys + 1) * 4);
    TR_INV_DEV(dcoef, k * (uint64_t)v->total * 8);
    TR_INV_DEV(dpoly, n * 4);
    TR_INV_DEV(dsign, n);
    TR_INV_DEV(dnum, cells * 8);
    TR_INV_DEV(dden, cells * 8);
    TR_INV_DEV(dout, n);
    TR_INV_IN(dfields, v->fields, k * sizeof(tr_gfp));
    TR_INV_IN(dgarner, v->garner, k * k * 8);
    TR_INV_IN(dhalf, v->half, k * 8);
    TR_INV_IN(doffset, v->offset, ((uint64_t)v->npolys + 1) * 4);
    TR_INV_IN(dcoef, v->coef, k * (uint64_t)v->total * 8);
    TR_INV_IN(dpoly, v->poly, n * 4);
    TR_INV_IN(dsign, v->sign, n);
    TR_INV_IN(dnum, v->num, cells * 8);
    TR_INV_IN(dden, v->den, cells * 8);
    d.fields = dfields; d.garner = dgarner; d.half = dhalf; d.offset = doffset; d.coef = dcoef; d.poly = dpoly;
    d.sign = dsign; d.num = dnum; d.den = dden;
    if (!rc) { TR_LAUNCH(tr_inv_signs_kernel, n, d, dout); rc = tr_cuda_sync(); }
    if (!rc) rc = tr_cuda_copy_out(signs, dout, (size_t)n);
    for (int i = 0; i < nmem; i++) tr_cuda_free(mem[i]);
    return rc;
}

int tr_mgpu_inv_sweep(const tr_inv_sweep_view *v, uint64_t *bitmap, const tr_math_budget *budget,
                      tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!tr_inv_sweep_view_valid(v) || (v->count && !bitmap)) return TR_MATH_INVALID;
    tr_math_budget dflt;
    budget = tr_inv_budget(budget, &dflt);
    int rc = tr_budget_check_one(budget, TR_BUDGET_BYTES, tr_inv_sweep_device_bytes(v), refusal);
    if (rc) return rc;
    uint64_t words = (v->count + 63) / 64, k = v->nprimes;
    if (!v->count) return TR_MATH_OK;
    tr_inv_sweep_view d = *v;
    void *mem[6];
    int nmem = 0;
    tr_gfp *dfields = NULL;
    uint8_t *dexps = NULL;
    uint64_t *dcres = NULL, *dvres = NULL, *dbits = NULL;
    uint64_t nt = v->nterms, ne = nt * (uint64_t)v->ndims;
    TR_INV_DEV(dfields, k * sizeof(tr_gfp));
    TR_INV_DEV(dexps, ne);
    TR_INV_DEV(dcres, k * nt * 8);
    TR_INV_DEV(dvres, k * v->nvalues * 8);
    TR_INV_DEV(dbits, words * 8);
    TR_INV_IN(dfields, v->fields, k * sizeof(tr_gfp));
    TR_INV_IN(dexps, v->exps, ne);
    TR_INV_IN(dcres, v->cres, k * nt * 8);
    TR_INV_IN(dvres, v->vres, k * v->nvalues * 8);
    d.fields = dfields; d.exps = dexps; d.cres = dcres; d.vres = dvres;
    if (!rc) { TR_LAUNCH(tr_inv_sweep_kernel, v->count, d, dbits); rc = tr_cuda_sync(); }
    if (!rc) rc = tr_cuda_copy_out(bitmap, dbits, (size_t)words * 8);
    for (int i = 0; i < nmem; i++) tr_cuda_free(mem[i]);
    return rc;
}
#ifdef __cplusplus
}
#endif
