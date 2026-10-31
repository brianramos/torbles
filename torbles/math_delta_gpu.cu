/* DELTA-001 on a CUDA device: a delta batch applied to the touched cells of a maintained state, with exactly the
 * cells of tr_delta_state_apply (math_delta.h), and the exact nonnegativity check before anything is exposed.
 * SPDX-License-Identifier: Apache-2.0
 *
 * The host gathers the old cells of the groups the delta touches (tr_delta_state_gather: compact, one row of cells per
 * touched group) and proves before the launch that every addition stays inside its carrier (tr_delta_prove). Then:
 *   1. one work item per delta row adds its weight to its group's bag multiplicity (64-bit two's complement atomic);
 *   2. one work item per (delta row, term) adds the weight to the term's count and weight * value to its sum when the
 *      term is valid on the row (TR_BANK_SQL) or the row is jointly valid (TR_BANK_JOINT): a signed 64-bit atomic and
 *      the two-word 128-bit atomic tr_atomic_add_wide (math_cuda.h). The words hold the exact results modulo 2^64 and
 *      2^128, which are the exact results because the proof bounds them;
 *   3. for a nonnegative measure, one work item per cell (touched group, slot) evaluates tr_delta_reason and the
 *      smallest contradicted cell key is kept with an atomic min.
 * Integer atomics commute, so the cells and the key do not depend on thread order. The host adopts the cells with
 * tr_delta_state_from_cells, which checks the key again exactly on the host before the new state exists. */
#include "math_cuda.h"
#include "math_delta.h"
#include "math_handles.h"

TR_MGPU_ABI(tr_delta_rows, 1);           /* structs passed to the CUDA library (math_handles.h) */

TR_KERNEL void tr_delta_rows_kernel(const uint32_t *local, const int64_t *w, uint64_t k, uint64_t *rows) {
    TR_GRID_LOOP(r, k) {
        if (w[r]) tr_atomic_add_u64(&rows[local[r]], (uint64_t)w[r]);
    }
}

TR_KERNEL void tr_delta_terms_kernel(tr_scope_view v, const tr_bank_term *terms, int32_t nterms, int32_t mode,
                                     uint32_t joint, const int64_t *w, uint64_t *count, int64_t *sum) {
    TR_GRID_LOOP(i, v.rows * (uint64_t)nterms) {
        uint64_t r = i / (uint64_t)nterms, j = i % (uint64_t)nterms;
        int64_t wr = w[r];
        if (!wr) continue;
        tr_bank_term t = terms[j];
        if (mode == TR_BANK_JOINT ? !tr_bank_joint_valid(&v, joint, r) : !tr_bank_term_valid(&v, &t, r)) continue;
        uint64_t k = (uint64_t)tr_scope_group(&v, r) * (uint64_t)nterms + j;
        tr_atomic_add_u64(&count[k], (uint64_t)wr);
        __int128 x = (__int128)wr * tr_bank_term_value(&v, &t, r);
        if (x) tr_atomic_add_wide(&sum[2 * k], &sum[2 * k + 1], x);
    }
}

TR_KERNEL void tr_delta_check_kernel(const uint64_t *rows, const uint64_t *count, const int64_t *sum, uint64_t touched,
                                     const tr_bank_term *terms, int32_t nterms, int32_t mode, uint64_t *violation) {
    const uint64_t width = (uint64_t)nterms + 1;
    uint64_t best = TR_DELTA_NONE;
    TR_GRID_LOOP(i, touched * width) {
        uint64_t u = i / width, s = i % width, k = u * (uint64_t)nterms + s - 1;
        int64_t c = 0;
        __int128 x = 0;
        if (s) {
            tr_math_i128 h;
            h.lo = sum[2 * k]; h.hi = sum[2 * k + 1];
            c = (int64_t)count[k]; x = tr_i128_get(h);
        }
        if (tr_delta_reason(s ? &terms[s - 1] : terms, mode, (int64_t)rows[u], c, x, (int)s) && i < best) best = i;
    }
    if (best != TR_DELTA_NONE) tr_atomic_min_u64(violation, best);
}

#ifdef __cplusplus
extern "C" {
#endif
int tr_mgpu_delta_apply(const tr_scope_view *dv, const tr_delta_rows *dr, const uint8_t *exps, int32_t nterms,
                        int32_t mode, int32_t promise, const tr_bank_states *in, const tr_math_budget *budget,
                        tr_bank_states *out, uint64_t *violation, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!dv || !dr || !in || !out || !violation || (mode != TR_BANK_SQL && mode != TR_BANK_JOINT) ||
        (promise != TR_DELTA_NONNEGATIVE && promise != TR_DELTA_SIGNED) || dv->rows != dr->rows ||
        in->ngroups != dr->touched || out->ngroups != dr->touched || in->nterms != nterms || out->nterms != nterms ||
        dv->ncols < 0 || dv->ncols > TR_SCOPE_MAX_COLUMNS || (dr->touched && (!in->rows || !out->rows)) ||
        (dr->touched && nterms > 0 && (!in->count || !in->sum || !out->count || !out->sum)) ||
        (dr->rows && (!dr->weights || !dr->local || !dr->wabs))) return -1;                     /* TR_MATH_INVALID */
    if (nterms < 0 || nterms > TR_BANK_MAX_TERMS) return nterms < 0 ? -1 : TR_MATH_UNSUPPORTED;
    for (uint64_t r = 0; r < dr->rows; r++) if (dr->local[r] >= dr->touched) return -1;
    tr_bank_term *terms = (tr_bank_term *)calloc((size_t)(nterms > 0 ? nterms : 1), sizeof(tr_bank_term));
    if (!terms) return -2;                                                                       /* TR_MATH_MEMORY */
    uint32_t joint = 0;
    int32_t degree = 0;
    int rc = tr_bank_compile(exps, nterms, dv->ncols, terms, &joint, &degree);
    tr_math_need need;
    tr_math_budget dflt;
    if (!budget) { tr_budget_default(&dflt); budget = &dflt; }
    if (!rc) { tr_delta_device_need(dv, nterms, joint, degree, dr->touched, &need); rc = tr_budget_check(budget, &need, refusal); }
    if (!rc) rc = tr_delta_prove(dv, terms, nterms, mode, joint, in, dr->wabs);                 /* before any launch */
    const uint64_t k = dr->rows, T = dr->touched, m = (uint64_t)(nterms > 0 ? nterms : 0), tm = T * m;
    *violation = TR_DELTA_NONE;
    if (rc || !T) { free(terms); return rc; }
    tr_scope_view d = *dv;
    void *dmem[2 * TR_SCOPE_MAX_COLUMNS + 10];
    int nmem = 0;
    uint64_t *drows = NULL, *dcount = NULL, *dviol = NULL;
    int64_t *dsum = NULL, *dw = NULL;
    uint32_t *dlocal = NULL;
    tr_bank_term *dterms = NULL;
#define TR_DELTA_DEV(ptr, bytes) do { if (!rc) { rc = tr_cuda_alloc((void **)&(ptr), (size_t)(bytes)); if (!rc) dmem[nmem++] = (void *)(ptr); } } while (0)
    TR_DELTA_DEV(dlocal, k * 4);
    if (!rc) rc = tr_cuda_copy_in(dlocal, dr->local, (size_t)k * 4);
    TR_DELTA_DEV(dw, k * 8);
    if (!rc) rc = tr_cuda_copy_in(dw, dr->weights, (size_t)k * 8);
    d.selection = NULL; d.groups = dlocal;           /* every delta row is selected; groups = local ids */
    for (int32_t c = 0; c < dv->ncols; c++) {
        tr_scope_view_column *col = &d.col[c];
        col->codes = NULL;
        if (!((joint >> c) & 1)) { col->values = NULL; col->valid = NULL; continue; }   /* never read */
        int64_t *p = NULL;
        TR_DELTA_DEV(p, k * 8);
        if (!rc) rc = tr_cuda_copy_in(p, dv->col[c].values, (size_t)k * 8);
        col->values = p;
        if (dv->col[c].valid) {
            uint8_t *q = NULL;
            TR_DELTA_DEV(q, k);
            if (!rc) rc = tr_cuda_copy_in(q, dv->col[c].valid, (size_t)k);
            col->valid = q;
        }
    }
    TR_DELTA_DEV(dterms, (m ? m : 1) * sizeof(tr_bank_term));
    if (!rc && m) rc = tr_cuda_copy_in(dterms, terms, (size_t)m * sizeof(tr_bank_term));
    TR_DELTA_DEV(drows, T * 8);
    if (!rc) rc = tr_cuda_copy_in(drows, in->rows, (size_t)T * 8);
    TR_DELTA_DEV(dcount, (tm ? tm : 1) * 8);
    TR_DELTA_DEV(dsum, (tm ? tm : 1) * 16);
    if (!rc && tm) rc = tr_cuda_copy_in(dcount, in->count, (size_t)tm * 8);
    if (!rc && tm) rc = tr_cuda_copy_in(dsum, in->sum, (size_t)tm * 16);
    uint64_t none = TR_DELTA_NONE;
    TR_DELTA_DEV(dviol, 8);
    if (!rc) rc = tr_cuda_copy_in(dviol, &none, 8);
#undef TR_DELTA_DEV
    if (!rc) {
        TR_LAUNCH(tr_delta_rows_kernel, k, dlocal, dw, k, drows);
        if (m) TR_LAUNCH(tr_delta_terms_kernel, k * m, d, dterms, nterms, mode, joint, dw, dcount, dsum);
        rc = tr_cuda_sync();
    }
    if (!rc && promise == TR_DELTA_NONNEGATIVE) {
        TR_LAUNCH(tr_delta_check_kernel, T * (m + 1), drows, dcount, dsum, T, dterms, nterms, mode, dviol);
        rc = tr_cuda_sync();
    }
    if (!rc) rc = tr_cuda_copy_out(out->rows, drows, (size_t)T * 8);
    if (!rc && tm) rc = tr_cuda_copy_out(out->count, dcount, (size_t)tm * 8);
    if (!rc && tm) rc = tr_cuda_copy_out(out->sum, dsum, (size_t)tm * 16);
    if (!rc) rc = tr_cuda_copy_out(violation, dviol, 8);
    for (int i = 0; i < nmem; i++) tr_cuda_free(dmem[i]);
    free(terms);
    return rc;
}
#ifdef __cplusplus
}
#endif
