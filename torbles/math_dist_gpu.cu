/* DIST-001 / DIST-002 on a CUDA device: histogram construction with exactly the states of tr_dist_hist_build, and
 * batched prefix requests with exactly the answers of tr_dist_query (math_dist.h).
 * SPDX-License-Identifier: Apache-2.0
 *
 * Histograms. Dense: one work item per row adds 1 to its cell's count (and to wcount, and its weight to the cell's
 * 128-bit sum with the two-word atomic of math_cuda.h) at index g * cells + the cell of its values in the scope
 * domain; the bag multiplicity of each group comes from a slice kernel (local counts while the group stays the same,
 * then one atomic). Sparse: the same counting into a device scratch over the scope domain, then a compaction without
 * barriers in three more launches: (1) one work item per chunk of cells counts its occupied cells, (2) one work item
 * scans the chunk counts serially (at most TR_DIST_SCAN_CHUNKS of them) into each chunk's first output position and
 * the total, (3) one work item per chunk writes its occupied cells in order from that position (coordinates decoded
 * from the cell index, counters copied) and the offset of every group whose first cell lies in the chunk. Every output
 * slot has exactly one writer and integer atomics commute, so nothing depends on the order of the threads. The host
 * proves the weight sums fit (tr_dist_plan_make) and checks the device copies against the budget before launching.
 *
 * Requests. A prefix state is uploaded once (tr_mgpu_dist_prefix_upload); each batch runs one work item per request,
 * the shared tr_dist_eval of math_dist.h (binary searches over the prefix arrays), one output slot each. */
#include "math_cuda.h"
#include "math_dist.h"
#include "math_handles.h"

TR_MGPU_ABI(tr_dist_spec, 1);               /* structs passed to the CUDA library (math_handles.h) */
TR_MGPU_ABI(tr_dist_states, 1);
TR_MGPU_ABI(tr_dist_prefix_view, 1);
TR_MGPU_ABI(tr_dist_request, 1);
TR_MGPU_ABI(tr_dist_result, 1);

#define TR_DIST_SLICE_WORK (UINT64_C(1) << 19)
#define TR_DIST_SLICE_ROWS 32
#define TR_DIST_DEVICE_MAGIC UINT32_C(0x56445344)

TR_KERNEL void tr_dist_rows_kernel(tr_scope_view v, uint64_t slices, uint64_t *rows) {
    TR_GRID_LOOP(s, slices) {
        uint32_t cur = 0;
        uint64_t c = 0;
        for (uint64_t r = s; r < v.rows; r += slices) {
            if (!tr_scope_is_selected(&v, r)) continue;
            uint32_t g = tr_scope_group(&v, r);
            if (c && g != cur) { tr_atomic_add_u64(&rows[cur], c); c = 0; }
            cur = g; c++;
        }
        if (c) tr_atomic_add_u64(&rows[cur], c);
    }
}

TR_KERNEL void tr_dist_cells_kernel(tr_scope_view v, tr_dist_plan p, uint64_t *count, uint64_t *wcount, int64_t *wsum) {
    TR_GRID_LOOP(r, v.rows) {
        if (!tr_scope_is_selected(&v, r)) continue;
        uint64_t idx = (uint64_t)tr_scope_group(&v, r) * p.cells + tr_dist_row_cell(&v, &p, r);
        tr_atomic_add_u64(&count[idx], 1);
        if (p.weighted && tr_scope_present(&v, p.weight, r)) {
            tr_atomic_add_u64(&wcount[idx], 1);
            tr_atomic_add_wide(&wsum[2 * idx], &wsum[2 * idx + 1], (__int128)tr_scope_value(&v, p.weight, r));
        }
    }
}

TR_KERNEL void tr_dist_occupancy_kernel(const uint64_t *count, uint64_t total, uint64_t chunk, uint64_t nchunks,
                                        uint64_t *occ) {
    TR_GRID_LOOP(c, nchunks) {
        uint64_t b = c * chunk, e = total - b < chunk ? total : b + chunk, n = 0;
        for (uint64_t i = b; i < e; i++) n += count[i] != 0;
        occ[c] = n;
    }
}

TR_KERNEL void tr_dist_scan_kernel(const uint64_t *occ, uint64_t nchunks, uint64_t *base, uint64_t *offset,
                                   uint64_t ngroups, uint64_t *ncells) {
    TR_GRID_LOOP(t, 1) {
        uint64_t s = 0;
        for (uint64_t c = 0; c < nchunks; c++) { base[c] = s; s += occ[c]; }
        ncells[0] = s;
        offset[ngroups] = s;
    }
}

TR_KERNEL void tr_dist_compact_kernel(tr_dist_plan p, const uint64_t *count, const uint64_t *wcount, const int64_t *wsum,
                                      uint64_t total, uint64_t chunk, uint64_t nchunks, const uint64_t *base,
                                      uint64_t *offset, int64_t *v0, int64_t *v1, int64_t *v2, uint8_t *nulls,
                                      uint64_t *ocount, uint64_t *owcount, int64_t *owsum) {
    TR_GRID_LOOP(c, nchunks) {
        uint64_t b = c * chunk, e = total - b < chunk ? total : b + chunk, pos = base[c];
        for (uint64_t i = b; i < e; i++) {
            uint64_t idx = i % p.cells;
            if (idx == 0) offset[i / p.cells] = pos;              /* the first cell of group i / cells */
            if (!count[i]) continue;
            int null;
            uint8_t mask = 0;
            int64_t *out[3] = {v0, v1, v2};
            for (int k = 0; k < p.k; k++) {
                out[k][pos] = tr_dist_plan_coord(&p, idx, k, &null);
                if (null) mask = (uint8_t)(mask | (1u << k));
            }
            nulls[pos] = mask;
            ocount[pos] = count[i];
            if (p.weighted) { owcount[pos] = wcount[i]; owsum[2 * pos] = wsum[2 * i]; owsum[2 * pos + 1] = wsum[2 * i + 1]; }
            pos++;
        }
    }
}

TR_KERNEL void tr_dist_query_kernel(tr_dist_prefix_view v, const tr_dist_request *req, uint64_t n, tr_dist_result *out) {
    TR_GRID_LOOP(i, n) tr_dist_eval(&v, &req[i], &out[i]);
}

struct tr_mgpu_dist_prefix_s {
    uint32_t magic;
    int32_t reserved;
    tr_dist_prefix_view dev;          /* device pointers */
    uint64_t bytes;
    void *mem[5];
};

static uint64_t tr_dist_slices(uint64_t rows) {
    uint64_t s = TR_DIST_SLICE_WORK, cap = (rows + TR_DIST_SLICE_ROWS - 1) / TR_DIST_SLICE_ROWS;
    if (s > cap) s = cap;
    return s ? s : 1;
}
/* tr_budget_check, refusing a size past 64 bits even under an unlimited budget */
static int tr_dist_gpu_check(const tr_math_budget *b, const tr_math_need *n, tr_math_refusal *r) {
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
int tr_mgpu_dist_hist_build(const tr_scope_view *view, const tr_dist_spec *spec, const tr_math_budget *budget,
                            tr_dist_states *out, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!view || !spec || !out || view->ngroups < 1 || (view->rows && !view->selection) ||
        (view->rows && view->ngroups > 1 && !view->groups) || view->ncols < 0 || view->ncols > TR_SCOPE_MAX_COLUMNS)
        return -1;                                                                    /* TR_MATH_INVALID */
    tr_dist_spec ns;
    tr_dist_spec_normalize(spec, &ns);
    tr_dist_plan p;
    int rc = tr_dist_plan_make(view, &ns, &p);
    if (rc) return rc;
    tr_math_need need;
    uint64_t cap = 0;
    tr_dist_device_need(view, &p, &need, &cap);
    tr_math_budget dflt;
    if (!budget) { tr_budget_default(&dflt); budget = &dflt; }
    rc = tr_dist_gpu_check(budget, &need, refusal);
    if (rc) return rc;
    const uint64_t n = view->rows, G = view->ngroups, total = G * p.cells;
    const int sparse = p.repr == TR_DIST_SPARSE;
    if (out->ngroups != G || out->k != p.k || out->repr != p.repr || out->weighted != p.weighted || !out->rows ||
        out->capacity < (sparse ? cap : total) || (out->capacity && !out->count) ||
        (p.weighted && out->capacity && (!out->wcount || !out->wsum)) || (sparse && !out->offset) ||
        (sparse && out->capacity && !out->nulls)) return -1;
    for (int c = 0; sparse && c < p.k; c++) if (out->capacity && !out->value[c]) return -1;
    out->cells = sparse ? 0 : p.cells;
    for (int c = 0; c < 3; c++) {
        out->lo[c] = c < p.k ? p.lo[c] : 0; out->span[c] = c < p.k ? p.span[c] : 0; out->dim[c] = c < p.k ? p.dim[c] : 0;
    }
    memset(out->rows, 0, (size_t)G * 8);
    if (!view->selected) {                           /* nothing selected: every state is zero, nothing to launch */
        out->ncells = sparse ? 0 : total;
        if (sparse) memset(out->offset, 0, (size_t)(G + 1) * 8);
        else if (total) {
            memset(out->count, 0, (size_t)total * 8);
            if (p.weighted) { memset(out->wcount, 0, (size_t)total * 8); memset(out->wsum, 0, (size_t)total * 16); }
        }
        return 0;
    }
    tr_scope_view d = *view;
    /* at most: selection, ids, 4 columns with validity, rows, 3 counters; sparse: 4 compaction arrays, 3 coordinates,
     * nulls and 3 counters = 25 device allocations */
    void *dmem[32];
    int nmem = 0;
    uint64_t *dsel = NULL, *drows = NULL, *dcount = NULL, *dwcount = NULL;
    uint32_t *dgroups = NULL;
    int64_t *dwsum = NULL;
#define TR_DIST_DEV(ptr, bytes) do { if (!rc) { rc = tr_cuda_alloc((void **)&(ptr), (size_t)(bytes)); if (!rc) dmem[nmem++] = (void *)(ptr); } } while (0)
    TR_DIST_DEV(dsel, ((n + 63) / 64) * 8);
    if (!rc) rc = tr_cuda_copy_in(dsel, view->selection, (size_t)(((n + 63) / 64) * 8));
    d.selection = dsel;
    if (view->groups) {
        TR_DIST_DEV(dgroups, n * 4);
        if (!rc) rc = tr_cuda_copy_in(dgroups, view->groups, (size_t)n * 4);
        d.groups = dgroups;
    }
    int32_t cols[4];
    int ncols = tr_dist_plan_columns(&p, cols);
    for (int32_t c = 0; c < view->ncols; c++) {
        tr_scope_view_column *k = &d.col[c];
        int used = 0;
        for (int i = 0; i < ncols; i++) used |= cols[i] == c;
        if (!used) { k->codes = NULL; k->values = NULL; k->valid = NULL; continue; }   /* never read */
        if (view->col[c].values) {
            int64_t *ptr = NULL;
            TR_DIST_DEV(ptr, n * 8);
            if (!rc) rc = tr_cuda_copy_in(ptr, view->col[c].values, (size_t)n * 8);
            k->values = ptr;
        } else {
            uint32_t *ptr = NULL;
            TR_DIST_DEV(ptr, n * 4);
            if (!rc) rc = tr_cuda_copy_in(ptr, view->col[c].codes, (size_t)n * 4);
            k->codes = ptr;
        }
        if (view->col[c].valid) {
            uint8_t *ptr = NULL;
            TR_DIST_DEV(ptr, n);
            if (!rc) rc = tr_cuda_copy_in(ptr, view->col[c].valid, (size_t)n);
            k->valid = ptr;
        }
    }
    TR_DIST_DEV(drows, G * 8);
    TR_DIST_DEV(dcount, total * 8);
    if (p.weighted) { TR_DIST_DEV(dwcount, total * 8); TR_DIST_DEV(dwsum, total * 16); }
    if (!rc) {
        uint64_t s = tr_dist_slices(n);
        TR_LAUNCH(tr_dist_rows_kernel, s, d, s, drows);
        TR_LAUNCH(tr_dist_cells_kernel, n, d, p, dcount, dwcount, dwsum);
        rc = tr_cuda_sync();
    }
    if (!rc) rc = tr_cuda_copy_out(out->rows, drows, (size_t)G * 8);
    if (!sparse) {
        if (!rc) rc = tr_cuda_copy_out(out->count, dcount, (size_t)total * 8);
        if (!rc && p.weighted) rc = tr_cuda_copy_out(out->wcount, dwcount, (size_t)total * 8);
        if (!rc && p.weighted) rc = tr_cuda_copy_out(out->wsum, dwsum, (size_t)total * 16);
        if (!rc) out->ncells = total;
    } else {
        const uint64_t chunk = tr_dist_chunk_len(total), nchunks = (total + chunk - 1) / chunk;
        uint64_t *docc = NULL, *dbase = NULL, *dn = NULL, *doffset = NULL, *docount = NULL, *dowcount = NULL;
        int64_t *dv[3] = {NULL, NULL, NULL}, *dowsum = NULL;
        uint8_t *dnulls = NULL;
        TR_DIST_DEV(docc, nchunks * 8);
        TR_DIST_DEV(dbase, nchunks * 8);
        TR_DIST_DEV(dn, 8);
        TR_DIST_DEV(doffset, (G + 1) * 8);
        for (int c = 0; c < p.k; c++) TR_DIST_DEV(dv[c], cap * 8);
        TR_DIST_DEV(dnulls, cap);
        TR_DIST_DEV(docount, cap * 8);
        if (p.weighted) { TR_DIST_DEV(dowcount, cap * 8); TR_DIST_DEV(dowsum, cap * 16); }
        if (!rc) {
            TR_LAUNCH(tr_dist_occupancy_kernel, nchunks, dcount, total, chunk, nchunks, docc);
            TR_LAUNCH(tr_dist_scan_kernel, 1, docc, nchunks, dbase, doffset, G, dn);
            TR_LAUNCH(tr_dist_compact_kernel, nchunks, p, dcount, dwcount, dwsum, total, chunk, nchunks, dbase, doffset,
                      dv[0], dv[1], dv[2], dnulls, docount, dowcount, dowsum);
            rc = tr_cuda_sync();
        }
        uint64_t ncells = 0;
        if (!rc) rc = tr_cuda_copy_out(&ncells, dn, 8);
        if (!rc && ncells > cap) rc = -1;
        if (!rc) rc = tr_cuda_copy_out(out->offset, doffset, (size_t)(G + 1) * 8);
        for (int c = 0; c < p.k; c++) if (!rc) rc = tr_cuda_copy_out(out->value[c], dv[c], (size_t)ncells * 8);
        if (!rc) rc = tr_cuda_copy_out(out->nulls, dnulls, (size_t)ncells);
        if (!rc) rc = tr_cuda_copy_out(out->count, docount, (size_t)ncells * 8);
        if (!rc && p.weighted) rc = tr_cuda_copy_out(out->wcount, dowcount, (size_t)ncells * 8);
        if (!rc && p.weighted) rc = tr_cuda_copy_out(out->wsum, dowsum, (size_t)ncells * 16);
        if (!rc) out->ncells = ncells;
    }
#undef TR_DIST_DEV
    for (int i = 0; i < nmem; i++) tr_cuda_free(dmem[i]);
    return rc;
}

tr_mgpu_dist_prefix *tr_mgpu_dist_prefix_upload(const tr_dist_prefix_view *v, const tr_math_budget *budget,
                                                tr_math_refusal *refusal, int *error) {
    tr_budget_none(refusal);
    if (error) *error = 0;
    if (!v || v->ngroups < 1 || v->degree < 0 || v->degree > TR_DIST_MAX_DEGREE || !v->rows || !v->nulls ||
        !v->offset || (v->nsupport && (!v->value || !v->prefix))) { if (error) *error = -1; return NULL; }
    tr_math_need need;
    for (int i = 0; i < TR_BUDGET_RESOURCES; i++) need.need[i] = 0;
    need.need[TR_BUDGET_BYTES] = tr_dist_prefix_copy_bytes(v);
    tr_math_budget dflt;
    if (!budget) { tr_budget_default(&dflt); budget = &dflt; }
    int rc = tr_dist_gpu_check(budget, &need, refusal);
    if (rc) { if (error) *error = rc; return NULL; }
    tr_mgpu_dist_prefix *dp = (tr_mgpu_dist_prefix *)calloc(1, sizeof *dp);
    if (!dp) { if (error) *error = -2; return NULL; }
    const uint64_t G = v->ngroups, n = v->nsupport, w = (uint64_t)v->degree + 1;
    dp->dev = *v;
    uint64_t *rows = NULL, *nulls = NULL, *offset = NULL;
    int64_t *value = NULL;
    tr_math_i128 *prefix = NULL;
    rc = tr_cuda_alloc((void **)&rows, (size_t)G * 8);
    if (!rc) { dp->mem[0] = rows; rc = tr_cuda_alloc((void **)&nulls, (size_t)G * 8); }
    if (!rc) { dp->mem[1] = nulls; rc = tr_cuda_alloc((void **)&offset, (size_t)(G + 1) * 8); }
    if (!rc) { dp->mem[2] = offset; rc = tr_cuda_alloc((void **)&value, (size_t)n * 8); }
    if (!rc) { dp->mem[3] = value; rc = tr_cuda_alloc((void **)&prefix, (size_t)(n * w) * 16); }
    if (!rc) dp->mem[4] = prefix;
    if (!rc) rc = tr_cuda_copy_in(rows, v->rows, (size_t)G * 8);
    if (!rc) rc = tr_cuda_copy_in(nulls, v->nulls, (size_t)G * 8);
    if (!rc) rc = tr_cuda_copy_in(offset, v->offset, (size_t)(G + 1) * 8);
    if (!rc) rc = tr_cuda_copy_in(value, v->value, (size_t)n * 8);
    if (!rc) rc = tr_cuda_copy_in(prefix, v->prefix, (size_t)(n * w) * 16);
    if (rc) {
        for (int i = 0; i < 5; i++) tr_cuda_free(dp->mem[i]);
        free(dp);
        if (error) *error = rc;
        return NULL;
    }
    dp->dev.rows = rows; dp->dev.nulls = nulls; dp->dev.offset = offset; dp->dev.value = value; dp->dev.prefix = prefix;
    dp->bytes = need.need[TR_BUDGET_BYTES];
    dp->magic = TR_DIST_DEVICE_MAGIC;
    return dp;
}

int tr_mgpu_dist_prefix_device_view(const tr_mgpu_dist_prefix *dp, tr_dist_prefix_view *out) {
    if (!dp || dp->magic != TR_DIST_DEVICE_MAGIC || !out) return -1;
    *out = dp->dev;
    return 0;
}

uint64_t tr_mgpu_dist_prefix_bytes(const tr_mgpu_dist_prefix *dp) {
    return dp && dp->magic == TR_DIST_DEVICE_MAGIC ? dp->bytes : 0;
}

void tr_mgpu_dist_prefix_free(tr_mgpu_dist_prefix *dp) {
    if (!dp || dp->magic != TR_DIST_DEVICE_MAGIC) return;
    for (int i = 0; i < 5; i++) tr_cuda_free(dp->mem[i]);
    dp->magic = 0;
    free(dp);
}

int tr_mgpu_dist_query(const tr_mgpu_dist_prefix *dp, const tr_dist_request *req, uint64_t n, tr_dist_result *out,
                       const tr_math_budget *budget, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!dp || dp->magic != TR_DIST_DEVICE_MAGIC || (n && (!req || !out))) return -1;
    tr_math_need need;
    for (int i = 0; i < TR_BUDGET_RESOURCES; i++) need.need[i] = 0;
    need.need[TR_BUDGET_OUTPUT] = n;
    need.need[TR_BUDGET_BYTES] = tr_budget_mul(n, sizeof(tr_dist_request) + sizeof(tr_dist_result));
    tr_math_budget dflt;
    if (!budget) { tr_budget_default(&dflt); budget = &dflt; }
    int rc = tr_dist_gpu_check(budget, &need, refusal);
    if (rc || !n) return rc;
    tr_dist_request *dreq = NULL;
    tr_dist_result *dres = NULL;
    rc = tr_cuda_alloc((void **)&dreq, (size_t)n * sizeof *dreq);
    if (!rc) rc = tr_cuda_alloc((void **)&dres, (size_t)n * sizeof *dres);
    if (!rc) rc = tr_cuda_copy_in(dreq, req, (size_t)n * sizeof *dreq);
    if (!rc) {
        TR_LAUNCH(tr_dist_query_kernel, n, dp->dev, dreq, n, dres);
        rc = tr_cuda_sync();
    }
    if (!rc) rc = tr_cuda_copy_out(out, dres, (size_t)n * sizeof *dres);
    tr_cuda_free(dreq); tr_cuda_free(dres);
    return rc;
}
#ifdef __cplusplus
}
#endif
