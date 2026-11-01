/* PRE-002 on a CUDA device: the membership scan of a preimage (intervals, residue classes, cubes, a lookup or the
 * direct predicate) over the rows of a scope column, and the exhaustive verification over a small code domain, with the
 * semantics of tr_preimg_scan / tr_preimg_verify (math_preimg.h) and the same results bit for bit.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Scan: one work item per 64-row word writes its own bitmap word (tr_preimg_word: membership of each present row's code,
 * with direct = 1 also the exact evaluation of the original predicate and the comparison) and adds its counts with
 * one atomic each, the smallest mismatching / undecided rows by atomic minimum. Verification: one work item per 64
 * codes of the domain (tr_preimg_verify_word: membership against the direct predicate), counts by atomics and the
 * smallest mismatching code (as an offset from lo) by atomic minimum. No barrier, no shared memory: nothing depends on
 * the order threads run in. The host validates the set's plain view (tr_preimg_view_ok) and budgets the device copies
 * before anything is copied; the set itself is built by tr_preimg_preimage (libalg). */
#include "math_cuda.h"
#include "math_preimg.h"

TR_MGPU_ABI(tr_preimg_view, TR_PREIMG_ABI);        /* structs passed to the CUDA library (math_handles.h) */
TR_MGPU_ABI(tr_preimg_branch, TR_PREIMG_ABI);
TR_MGPU_ABI(tr_preimg_verification, TR_PREIMG_ABI);

TR_KERNEL void tr_preimg_scan_kernel(tr_preimg_view v, tr_scope_view sv, int32_t col, int32_t direct, uint64_t nw,
                                     uint64_t *bitmap, uint64_t *acc) {
    uint64_t local[6] = {0, 0, 0, 0, 0, 0}, first_m = UINT64_MAX, first_u = UINT64_MAX;
    TR_GRID_LOOP(w, nw) bitmap[w] = tr_preimg_word(&v, &sv, col, direct, w, local, &first_m, &first_u);
    for (int i = 0; i < 6; i++) if (local[i]) tr_atomic_add_u64(&acc[i], local[i]);
    if (first_m != UINT64_MAX) tr_atomic_min_u64(&acc[6], first_m);
    if (first_u != UINT64_MAX) tr_atomic_min_u64(&acc[7], first_u);
}

TR_KERNEL void tr_preimg_verify_kernel(tr_preimg_view v, uint64_t total, uint64_t nw, uint64_t *acc) {
    uint64_t local[4] = {0, 0, 0, 0}, first = UINT64_MAX;
    TR_GRID_LOOP(w, nw) tr_preimg_verify_word(&v, w, total, local, &first);
    for (int i = 0; i < 4; i++) if (local[i]) tr_atomic_add_u64(&acc[i], local[i]);
    if (first != UINT64_MAX) tr_atomic_min_u64(&acc[4], first);
}

/* device copies of the set's arrays; dv points at them */
static int tr_preimg_gpu_set(const tr_preimg_view *view, tr_preimg_view *dv, void **mem, int *nmem) {
    int rc = 0;
#define TR_PREIMG_COPY(field, type, count) do { if (!rc && view->field && (uint64_t)(count) != 0) { void *p = NULL; \
        rc = tr_cuda_alloc(&p, (size_t)(count) * sizeof(type)); if (!rc) { mem[(*nmem)++] = p; \
        rc = tr_cuda_copy_in(p, view->field, (size_t)(count) * sizeof(type)); dv->field = (const type *)p; } } \
        else if (!rc) dv->field = NULL; } while (0)
    TR_PREIMG_COPY(branch, tr_preimg_branch, view->nbranch);
    TR_PREIMG_COPY(lookup, uint64_t, (view->nlookup + 63) / 64);
    TR_PREIMG_COPY(icoef, int64_t, view->map == TR_PREIMG_POLYMOD ? (uint64_t)view->degree + 1 : 0);
    TR_PREIMG_COPY(wcoef, uint64_t, view->map == TR_PREIMG_POLY && view->words ? ((uint64_t)view->degree + 1) * (uint64_t)view->words : 0);
    TR_PREIMG_COPY(wthreshold, uint64_t, view->map == TR_PREIMG_POLY && view->words ? (view->nvalues ? view->nvalues : 1) * (uint64_t)view->words : 0);
    TR_PREIMG_COPY(values, tr_num_rational, view->nvalues);
#undef TR_PREIMG_COPY
    return rc;
}

#ifdef __cplusplus
extern "C" {
#endif
int tr_mgpu_preimg_scan(const tr_preimg_view *view, const tr_scope_view *sv, int32_t column, int32_t direct, uint64_t *bitmap,
                        tr_pre_scan_stats *stats, const tr_math_budget *budget, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!view || !sv || !stats || column < 0 || column >= sv->ncols || sv->ncols > TR_SCOPE_MAX_COLUMNS ||
        (sv->rows && (!bitmap || !sv->selection || (!sv->col[column].codes && !sv->col[column].values))) ||
        (direct != 0 && direct != 1) || !tr_preimg_view_ok(view) || sv->col[column].scale != view->codec.scale)
        return -1;                                                              /* TR_MATH_INVALID */
    if (direct && view->map == TR_PREIMG_POLY && !view->words) return -12;      /* TR_MATH_UNSUPPORTED */
    tr_math_budget dflt;
    if (!budget) { tr_budget_default(&dflt); budget = &dflt; }
    int rc = tr_budget_check_one(budget, TR_BUDGET_BYTES, tr_preimg_scan_device_bytes(view, sv, column), refusal);
    if (rc) return rc;
    memset(stats, 0, sizeof *stats);
    stats->rows = sv->rows; stats->selected = sv->selected;
    stats->first_mismatch = UINT64_MAX; stats->first_overflow = UINT64_MAX;
    const uint64_t rows = sv->rows, nw = (rows + 63) / 64;
    if (!nw) return 0;
    void *mem[16];
    int nmem = 0;
#define TR_PREIMG_DEV(ptr, bytes) do { if (!rc) { rc = tr_cuda_alloc((void **)&(ptr), (size_t)(bytes)); if (!rc) mem[nmem++] = (void *)(ptr); } } while (0)
#define TR_PREIMG_IN(dst, src, bytes) do { TR_PREIMG_DEV(dst, bytes); if (!rc && (uint64_t)(bytes) != 0) rc = tr_cuda_copy_in(dst, src, (size_t)(bytes)); } while (0)
    tr_scope_view d = *sv;
    tr_preimg_view dv = *view;
    uint64_t *dsel = NULL, *dbits = NULL, *dacc = NULL, acc[8] = {0, 0, 0, 0, 0, 0, UINT64_MAX, UINT64_MAX};
    TR_PREIMG_IN(dsel, sv->selection, nw * 8);
    d.selection = dsel; d.groups = NULL;
    for (int32_t c = 0; c < sv->ncols; c++) {
        tr_scope_view_column *k = &d.col[c];
        if (c != column) { k->codes = NULL; k->values = NULL; k->valid = NULL; continue; }
        if (sv->col[c].values) { int64_t *p = NULL; TR_PREIMG_IN(p, sv->col[c].values, rows * 8); k->values = p; }
        else { uint32_t *p = NULL; TR_PREIMG_IN(p, sv->col[c].codes, rows * 4); k->codes = p; }
        if (sv->col[c].valid) { uint8_t *p = NULL; TR_PREIMG_IN(p, sv->col[c].valid, rows); k->valid = p; }
    }
    if (!rc) rc = tr_preimg_gpu_set(view, &dv, mem, &nmem);
    TR_PREIMG_DEV(dbits, nw * 8);
    TR_PREIMG_IN(dacc, acc, sizeof acc);
#undef TR_PREIMG_IN
#undef TR_PREIMG_DEV
    if (!rc) { TR_LAUNCH(tr_preimg_scan_kernel, nw, dv, d, column, direct, nw, dbits, dacc); rc = tr_cuda_sync(); }
    if (!rc) rc = tr_cuda_copy_out(bitmap, dbits, (size_t)nw * 8);
    if (!rc) rc = tr_cuda_copy_out(acc, dacc, sizeof acc);
    for (int i = 0; i < nmem; i++) tr_cuda_free(mem[i]);
    if (rc) return rc;
    stats->nulls = acc[0]; stats->in_set = acc[1]; stats->direct_true = acc[2]; stats->mismatches = acc[3];
    stats->overflow = acc[4]; stats->outside = acc[5]; stats->first_mismatch = acc[6]; stats->first_overflow = acc[7];
    return stats->overflow ? -3 : 0;                                            /* TR_MATH_OVERFLOW */
}

int tr_mgpu_preimg_verify(const tr_preimg_view *view, const tr_math_budget *budget, tr_preimg_verification *out,
                          tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!view || !out || !tr_preimg_view_ok(view)) return -1;                   /* TR_MATH_INVALID */
    const __int128 size = (__int128)view->codec.hi - view->codec.lo + 1;
    const __int128 L0 = (__int128)view->codec.lo + view->codec.offset, L1 = (__int128)view->codec.hi + view->codec.offset;
    if (size > (__int128)TR_PREIMG_MAX_VERIFY || L0 < INT64_MIN || L1 > INT64_MAX) return -12;
    if (view->map == TR_PREIMG_POLY && !view->words) return -12;                /* no direct evaluation */
    tr_math_budget dflt;
    if (!budget) { tr_budget_default(&dflt); budget = &dflt; }
    int rc = tr_budget_check_one(budget, TR_BUDGET_ITERATIONS, (uint64_t)size, refusal);
    if (!rc) rc = tr_budget_check_one(budget, TR_BUDGET_BYTES, tr_preimg_set_device_bytes(view), refusal);
    if (rc) return rc;
    memset(out, 0, sizeof *out);
    out->lo = view->codec.lo; out->hi = view->codec.hi;
    const uint64_t total = (uint64_t)size, nw = (total + 63) / 64;
    void *mem[16];
    int nmem = 0;
    tr_preimg_view dv = *view;
    uint64_t *dacc = NULL, acc[5] = {0, 0, 0, 0, UINT64_MAX};
    rc = tr_preimg_gpu_set(view, &dv, mem, &nmem);
    if (!rc) { rc = tr_cuda_alloc((void **)&dacc, sizeof acc); if (!rc) mem[nmem++] = dacc; }
    if (!rc) rc = tr_cuda_copy_in(dacc, acc, sizeof acc);
    if (!rc) { TR_LAUNCH(tr_preimg_verify_kernel, nw, dv, total, nw, dacc); rc = tr_cuda_sync(); }
    if (!rc) rc = tr_cuda_copy_out(acc, dacc, sizeof acc);
    for (int i = 0; i < nmem; i++) tr_cuda_free(mem[i]);
    if (rc) return rc;
    out->codes = acc[0]; out->members = acc[1]; out->truths = acc[2]; out->mismatches = acc[3];
    out->first = acc[4] == UINT64_MAX ? 0 : (int64_t)((uint64_t)view->codec.lo + acc[4]);
    return 0;
}
#ifdef __cplusplus
}
#endif
