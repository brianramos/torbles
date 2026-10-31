/* PRE-001 on a CUDA device: the transformed predicate evaluated on the stored codes, checked against the direct
 * evaluation of the original predicate on every selected row, with the semantics of tr_pre_scan (math_pre.h) and the
 * same results bit for bit. SPDX-License-Identifier: Apache-2.0
 *
 * One work item per 64-row word writes its own bitmap word (tr_pre_word: membership of each present row's code in
 * the set's spans; with direct = 1 also UNC-001's exact row evaluation of the branch program and the comparison) and
 * adds its counts with one atomic each, the smallest mismatching / undecided row with an atomic minimum. No barrier,
 * no shared memory. The host checks the view's structure and the budget before copying; the set itself is built and
 * validated by tr_pre_preimage (libalg), whose plain view (tr_pre_set_get_view) is passed here. */
#include "math_cuda.h"
#include "math_pre.h"

TR_MGPU_ABI(tr_pre_view, TR_PRE_ABI);              /* structs passed to the CUDA library (math_handles.h) */
TR_MGPU_ABI(tr_pre_codec, TR_PRE_ABI);
TR_MGPU_ABI(tr_pre_span, TR_PRE_ABI);
TR_MGPU_ABI(tr_pre_scan_stats, TR_PRE_ABI);

TR_KERNEL void tr_pre_scan_kernel(tr_pre_view v, tr_scope_view sv, int32_t col, int32_t direct, uint64_t nw,
                                  uint64_t *bitmap, uint64_t *acc) {
    tr_unc_rowval vals[8];
    uint64_t local[6] = {0, 0, 0, 0, 0, 0}, first_m = UINT64_MAX, first_o = UINT64_MAX;
    TR_GRID_LOOP(w, nw) bitmap[w] = tr_pre_word(&v, &sv, col, direct, w, vals, local, &first_m, &first_o);
    for (int i = 0; i < 6; i++) if (local[i]) tr_atomic_add_u64(&acc[i], local[i]);
    if (first_m != UINT64_MAX) tr_atomic_min_u64(&acc[6], first_m);
    if (first_o != UINT64_MAX) tr_atomic_min_u64(&acc[7], first_o);
}

/* the structure the kernel relies on (the set was built by tr_pre_preimage) */
static int tr_pre_gpu_view_ok(const tr_pre_view *v, int direct) {
    if (v->codec.lo > v->codec.hi || v->codec.scale < 0 || v->codec.scale > TR_SCOPE_MAX_SCALE) return 0;
    if (v->nspans && !v->spans) return 0;
    for (uint64_t i = 0; i < v->nspans; i++) {
        const tr_pre_span *s = &v->spans[i];
        if (v->residue ? (s->modulus < 1 || s->modulus != v->spans[0].modulus || s->rlo > s->rhi) : s->lo > s->hi) return 0;
        if (i && (v->residue ? s->rlo <= v->spans[i - 1].rhi : s->lo <= v->spans[i - 1].hi)) return 0;
    }
    if (!direct) return 1;
    if (v->nbranch < 0 || (v->nbranch && (!v->branch_lo || !v->branch_hi || !v->node_off || !v->nodes))) return 0;
    if (v->op < TR_PRE_LT || v->op > TR_PRE_IN || (v->op == TR_PRE_IN && v->nvalues && !v->values)) return 0;
    for (int32_t b = 0; b < v->nbranch; b++) {
        int32_t n = v->node_off[b + 1] - v->node_off[b];
        if (v->node_off[b] < 0 || n < 1 || n > 8 || (uint64_t)v->node_off[b + 1] > v->nnodes) return 0;
        tr_unc_program p;
        memset(&p, 0, sizeof p);
        p.nnodes = n; p.ncols = 1; p.carrier = TR_UNC_EXACT; p.rank = TR_UNC_MAX_TERMS; p.domains = TR_UNC_DOM_ALL;
        p.root = n - 1; p.nodes = v->nodes + v->node_off[b];
        if (tr_unc_check(&p, 0, NULL, NULL)) return 0;
        if (v->branch_lo[b] > v->branch_hi[b] || (b && v->branch_lo[b] <= v->branch_hi[b - 1])) return 0;
    }
    return 1;
}

#ifdef __cplusplus
extern "C" {
#endif
int tr_mgpu_pre_scan(const tr_pre_view *view, const tr_scope_view *sv, int32_t column, int32_t direct, uint64_t *bitmap,
                     tr_pre_scan_stats *stats, const tr_math_budget *budget, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!view || !sv || !stats || column < 0 || column >= sv->ncols || sv->ncols > TR_SCOPE_MAX_COLUMNS ||
        (sv->rows && (!bitmap || !sv->selection || (!sv->col[column].codes && !sv->col[column].values))) ||
        (direct != 0 && direct != 1) || !tr_pre_gpu_view_ok(view, direct) || sv->col[column].scale != view->codec.scale)
        return TR_MATH_INVALID;
    tr_math_budget dflt;
    if (!budget) { tr_budget_default(&dflt); budget = &dflt; }
    int rc = tr_budget_check_one(budget, TR_BUDGET_BYTES, tr_pre_scan_device_bytes(view, sv, column, direct), refusal);
    if (rc) return rc;
    memset(stats, 0, sizeof *stats);
    stats->rows = sv->rows; stats->selected = sv->selected;
    stats->first_mismatch = UINT64_MAX; stats->first_overflow = UINT64_MAX;
    const uint64_t rows = sv->rows, nw = (rows + 63) / 64;
    if (!nw) return 0;
    void *dmem[16];
    int nmem = 0;
#define TR_PRE_DEV(ptr, bytes) do { if (!rc) { rc = tr_cuda_alloc((void **)&(ptr), (size_t)(bytes)); if (!rc) dmem[nmem++] = (void *)(ptr); } } while (0)
#define TR_PRE_IN(dst, src, bytes) do { TR_PRE_DEV(dst, bytes); if (!rc && (uint64_t)(bytes) != 0) rc = tr_cuda_copy_in(dst, src, (size_t)(bytes)); } while (0)
    tr_scope_view d = *sv;
    tr_pre_view dv = *view;
    uint64_t *dsel = NULL, *dbits = NULL, *dacc = NULL, acc[8] = {0, 0, 0, 0, 0, 0, UINT64_MAX, UINT64_MAX};
    tr_pre_span *dspans = NULL;
    tr_num_rational *dvals = NULL;
    int64_t *dblo = NULL, *dbhi = NULL;
    int32_t *doff = NULL;
    tr_unc_node *dnodes = NULL;
    TR_PRE_IN(dsel, sv->selection, nw * 8);
    d.selection = dsel; d.groups = NULL;
    for (int32_t c = 0; c < sv->ncols; c++) {
        tr_scope_view_column *k = &d.col[c];
        if (c != column) { k->codes = NULL; k->values = NULL; k->valid = NULL; continue; }
        if (sv->col[c].values) { int64_t *p = NULL; TR_PRE_IN(p, sv->col[c].values, rows * 8); k->values = p; }
        else { uint32_t *p = NULL; TR_PRE_IN(p, sv->col[c].codes, rows * 4); k->codes = p; }
        if (sv->col[c].valid) { uint8_t *p = NULL; TR_PRE_IN(p, sv->col[c].valid, rows); k->valid = p; }
    }
    if (view->nspans) TR_PRE_IN(dspans, view->spans, view->nspans * sizeof(tr_pre_span));
    dv.spans = dspans;
    if (direct) {
        if (view->nvalues) TR_PRE_IN(dvals, view->values, view->nvalues * sizeof(tr_num_rational));
        if (view->nbranch) {
            TR_PRE_IN(dblo, view->branch_lo, (uint64_t)view->nbranch * 8);
            TR_PRE_IN(dbhi, view->branch_hi, (uint64_t)view->nbranch * 8);
            TR_PRE_IN(doff, view->node_off, ((uint64_t)view->nbranch + 1) * 4);
            TR_PRE_IN(dnodes, view->nodes, view->nnodes * sizeof(tr_unc_node));
        }
    }
    dv.values = dvals; dv.branch_lo = dblo; dv.branch_hi = dbhi; dv.node_off = doff; dv.nodes = dnodes;
    if (!direct) dv.nbranch = 0;
    TR_PRE_DEV(dbits, nw * 8);
    TR_PRE_IN(dacc, acc, sizeof acc);
#undef TR_PRE_IN
#undef TR_PRE_DEV
    if (!rc) { TR_LAUNCH(tr_pre_scan_kernel, nw, dv, d, column, direct, nw, dbits, dacc); rc = tr_cuda_sync(); }
    if (!rc) rc = tr_cuda_copy_out(bitmap, dbits, (size_t)nw * 8);
    if (!rc) rc = tr_cuda_copy_out(acc, dacc, sizeof acc);
    for (int i = 0; i < nmem; i++) tr_cuda_free(dmem[i]);
    if (rc) return rc;
    stats->nulls = acc[0]; stats->in_set = acc[1]; stats->direct_true = acc[2]; stats->mismatches = acc[3];
    stats->overflow = acc[4]; stats->outside = acc[5]; stats->first_mismatch = acc[6]; stats->first_overflow = acc[7];
    return stats->overflow ? TR_MATH_OVERFLOW : 0;
}
#ifdef __cplusplus
}
#endif
