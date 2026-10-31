/* UNC-001 on a CUDA device: abstract evaluation of many groups at once and the concrete fallback over 64-row words,
 * with the semantics of tr_unc_eval / tr_unc_rows (math_unc.h) and the same results bit for bit.
 * SPDX-License-Identifier: Apache-2.0
 *
 * tr_mgpu_unc_eval: one work item per group evaluates the program through every node (tr_unc_eval_group: intervals,
 * affine forms, congruences, nullness and outcome sets) with its node values in its own slot range of a device
 * scratch (chunks of TR_UNC_GPU_CHUNK groups per launch) and writes its own result slot.
 * tr_mgpu_unc_rows: one work item per 64-row word writes its own bitmap word (rows of a group decided by the proof
 * are set or cleared without evaluation, the others evaluated exactly: tr_unc_rows_word) and adds its counts with
 * one atomic each, the smallest overflowing row with an atomic minimum. No barrier, no shared memory; the host
 * validates the program and the inputs (tr_unc_check, tr_unc_input_ok) and checks the budget before copying. */
#include "math_cuda.h"
#include "math_unc.h"

TR_MGPU_ABI(tr_unc_node, TR_UNC_ABI);              /* structs passed to the CUDA library (math_handles.h) */
TR_MGPU_ABI(tr_unc_program, TR_UNC_ABI);
TR_MGPU_ABI(tr_unc_input, TR_UNC_ABI);
TR_MGPU_ABI(tr_unc_result, TR_UNC_ABI);
TR_MGPU_ABI(tr_unc_filter_stats, TR_UNC_ABI);

TR_KERNEL void tr_unc_eval_kernel(tr_unc_program p, tr_unc_plan plan, const tr_unc_input *in, uint64_t first,
                                  uint64_t count, tr_unc_value *scratch, tr_unc_result *out) {
    TR_GRID_LOOP(i, count) {
        uint64_t g = first + i;
        tr_unc_eval_group(&p, &plan, &in[g * (uint64_t)p.ncols], &scratch[i * ((uint64_t)plan.nlive + 1)], &out[g]);
    }
}

TR_KERNEL void tr_unc_rows_kernel(tr_unc_program p, tr_scope_view v, const int32_t *cols, const int32_t *decisions,
                                  uint64_t nw, uint64_t *bitmap, uint64_t *acc) {
    tr_unc_rowval vals[TR_UNC_MAX_NODES];
    uint64_t local[4] = {0, 0, 0, 0}, first = UINT64_MAX;
    TR_GRID_LOOP(w, nw) bitmap[w] = tr_unc_rows_word(&p, &v, cols, decisions, w, vals, local, &first);
    for (int i = 0; i < 4; i++) if (local[i]) tr_atomic_add_u64(&acc[i], local[i]);
    if (first != UINT64_MAX) tr_atomic_min_u64(&acc[4], first);
}

#ifdef __cplusplus
extern "C" {
#endif
int tr_mgpu_unc_eval(const tr_unc_input *inputs, uint64_t ngroups, int32_t ncols, const tr_unc_program *prog,
                     tr_unc_result *out, const tr_math_budget *budget, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!prog || !inputs || !out || ngroups < 1) return TR_MATH_INVALID;
    tr_unc_plan plan;
    tr_math_need need;
    int rc = tr_unc_check(prog, 0, &plan, &need);
    if (rc) return rc;
    if (ncols != prog->ncols) return TR_MATH_INVALID;
    for (uint64_t g = 0; g < ngroups; g++)
        for (int32_t j = 0; j < ncols; j++) {
            const tr_unc_input *in = &inputs[g * (uint64_t)ncols + (uint64_t)j];
            if (!tr_unc_input_ok(in) || in->rows != inputs[g * (uint64_t)ncols].rows) return TR_MATH_INVALID;
        }
    tr_math_budget dflt;
    if (!budget) { tr_budget_default(&dflt); budget = &dflt; }
    need.need[TR_BUDGET_OUTPUT] = ngroups;
    need.need[TR_BUDGET_BYTES] = tr_unc_eval_device_bytes(ngroups, ncols, prog->nnodes, plan.nlive);
    rc = tr_budget_check(budget, &need, refusal);
    if (rc) return rc;
    uint64_t chunk = ngroups < TR_UNC_GPU_CHUNK ? ngroups : TR_UNC_GPU_CHUNK, cells = ngroups * (uint64_t)ncols;
    tr_unc_input *din = NULL;
    tr_unc_result *dout = NULL;
    tr_unc_node *dnodes = NULL;
    tr_unc_value *dscratch = NULL;
    rc = tr_cuda_alloc((void **)&din, (size_t)cells * sizeof *din);
    if (!rc) rc = tr_cuda_alloc((void **)&dout, (size_t)ngroups * sizeof *dout);
    if (!rc) rc = tr_cuda_alloc((void **)&dnodes, (size_t)prog->nnodes * sizeof *dnodes);
    if (!rc) rc = tr_cuda_alloc((void **)&dscratch, (size_t)chunk * ((size_t)plan.nlive + 1) * sizeof *dscratch);
    if (!rc) rc = tr_cuda_copy_in(din, inputs, (size_t)cells * sizeof *din);
    if (!rc) rc = tr_cuda_copy_in(dnodes, prog->nodes, (size_t)prog->nnodes * sizeof *dnodes);
    tr_unc_program dp = *prog;
    dp.nodes = dnodes;
    for (uint64_t first = 0; !rc && first < ngroups; first += chunk) {
        uint64_t count = ngroups - first < chunk ? ngroups - first : chunk;
        TR_LAUNCH(tr_unc_eval_kernel, count, dp, plan, din, first, count, dscratch, dout);
        rc = tr_cuda_sync();
    }
    if (!rc) rc = tr_cuda_copy_out(out, dout, (size_t)ngroups * sizeof *dout);
    tr_cuda_free(din); tr_cuda_free(dout); tr_cuda_free(dnodes); tr_cuda_free(dscratch);
    return rc;
}

int tr_mgpu_unc_rows(const tr_scope_view *view, const int32_t *cols, const tr_unc_program *prog, const int32_t *decisions,
                     uint64_t *bitmap, tr_unc_filter_stats *stats, const tr_math_budget *budget, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!view || !cols || !prog || !stats || (view->rows && (!bitmap || !view->selection)) || view->ngroups < 1 ||
        (view->rows && view->ngroups > 1 && decisions && !view->groups) || view->ncols < 0 || view->ncols > TR_SCOPE_MAX_COLUMNS)
        return TR_MATH_INVALID;
    tr_math_need need;
    int rc = tr_unc_check(prog, 0, NULL, &need);
    if (rc) return rc;
    for (int32_t j = 0; j < prog->ncols; j++) {
        int32_t c = cols[j];
        if (c < 0 || c >= view->ncols || (view->rows && !view->col[c].codes && !view->col[c].values)) return TR_MATH_INVALID;
    }
    if (decisions)
        for (uint64_t g = 0; g < view->ngroups; g++) if (decisions[g] < TR_UNC_UNKNOWN || decisions[g] > TR_UNC_EMPTY) return TR_MATH_INVALID;
    tr_math_budget dflt;
    if (!budget) { tr_budget_default(&dflt); budget = &dflt; }
    need.need[TR_BUDGET_BYTES] = tr_unc_rows_device_bytes(view, cols, prog->ncols, prog->nnodes, decisions != NULL);
    rc = tr_budget_check(budget, &need, refusal);
    if (rc) return rc;
    memset(stats, 0, sizeof *stats);
    stats->rows = view->rows; stats->selected = view->selected; stats->first_overflow = UINT64_MAX;
    const uint64_t rows = view->rows, nw = (rows + 63) / 64;
    if (!nw) return 0;
    void *dmem[3 * TR_SCOPE_MAX_COLUMNS + 8];
    int nmem = 0;
#define TR_UNC_DEV(ptr, bytes) do { if (!rc) { rc = tr_cuda_alloc((void **)&(ptr), (size_t)(bytes)); if (!rc) dmem[nmem++] = (void *)(ptr); } } while (0)
#define TR_UNC_IN(dst, src, bytes) do { TR_UNC_DEV(dst, bytes); if (!rc && (uint64_t)(bytes) != 0) rc = tr_cuda_copy_in(dst, src, (size_t)(bytes)); } while (0)
    tr_scope_view d = *view;
    uint64_t *dsel = NULL, *dbits = NULL, *dacc = NULL, acc[5] = {0, 0, 0, 0, UINT64_MAX};
    uint32_t *dgroups = NULL;
    int32_t *dcols = NULL, *ddec = NULL;
    tr_unc_node *dnodes = NULL;
    TR_UNC_IN(dsel, view->selection, nw * 8);
    d.selection = dsel;
    d.groups = NULL;
    if (decisions && view->groups) { TR_UNC_IN(dgroups, view->groups, rows * 4); d.groups = dgroups; }
    uint32_t seen = 0;
    for (int32_t j = 0; j < prog->ncols; j++) seen |= (uint32_t)1 << cols[j];
    for (int32_t c = 0; c < view->ncols; c++) {
        tr_scope_view_column *k = &d.col[c];
        if (!((seen >> c) & 1)) { k->codes = NULL; k->values = NULL; k->valid = NULL; continue; }
        if (view->col[c].values) { int64_t *p = NULL; TR_UNC_IN(p, view->col[c].values, rows * 8); k->values = p; }
        else { uint32_t *p = NULL; TR_UNC_IN(p, view->col[c].codes, rows * 4); k->codes = p; }
        if (view->col[c].valid) { uint8_t *p = NULL; TR_UNC_IN(p, view->col[c].valid, rows); k->valid = p; }
    }
    TR_UNC_IN(dcols, cols, (uint64_t)prog->ncols * 4);
    if (decisions) TR_UNC_IN(ddec, decisions, view->ngroups * 4);
    TR_UNC_IN(dnodes, prog->nodes, (uint64_t)prog->nnodes * sizeof(tr_unc_node));
    TR_UNC_DEV(dbits, nw * 8);
    TR_UNC_IN(dacc, acc, sizeof acc);
#undef TR_UNC_IN
#undef TR_UNC_DEV
    tr_unc_program dp = *prog;
    dp.nodes = dnodes;
    if (!rc) { TR_LAUNCH(tr_unc_rows_kernel, nw, dp, d, dcols, ddec, nw, dbits, dacc); rc = tr_cuda_sync(); }
    if (!rc) rc = tr_cuda_copy_out(bitmap, dbits, (size_t)nw * 8);
    if (!rc) rc = tr_cuda_copy_out(acc, dacc, sizeof acc);
    for (int i = 0; i < nmem; i++) tr_cuda_free(dmem[i]);
    if (rc) return rc;
    stats->rows_proven = acc[0]; stats->rows_evaluated = acc[1]; stats->rows_true = acc[2]; stats->overflow = acc[3];
    stats->first_overflow = acc[4];
    return stats->overflow ? TR_MATH_OVERFLOW : 0;
}
#ifdef __cplusplus
}
#endif
