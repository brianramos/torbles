/* TOPO-001 / TOPO-002 on a CUDA device: the occupancy of a declared grid, its cells and incidences, the Euler counts,
 * the local invariants and the boundary-composition checks, with exactly the results of math_topo.c (math_topo.h).
 * SPDX-License-Identifier: Apache-2.0
 *
 * Order independence: the occupancy is an atomic OR per row; counts are accumulated per thread and added with one
 * atomic each; the cells are written at positions fixed by an exclusive prefix (on the host) of per-chunk counts that
 * each work item writes into its own slots, in increasing linear index within its chunk; incidences, top-cell values
 * and local invariants are one writer per output slot; the value extremes use atomic minimum / maximum, the
 * composition witnesses atomic minimum. Nothing depends on the order the threads run in. The column reductions of
 * TOPO-002 (homology, persistence) are CPU only. */
#include "math_cuda.h"
#include "math_topo.h"
#include "math_handles.h"

TR_MGPU_ABI(tr_topo_grid, TR_TOPO_ABI);        /* structs passed to the CUDA library (math_handles.h) */
TR_MGPU_ABI(tr_topo_rows, TR_TOPO_ABI);
TR_MGPU_ABI(tr_topo_counts, TR_TOPO_ABI);
TR_MGPU_ABI(tr_topo_states, TR_TOPO_ABI);
TR_MGPU_ABI(tr_topo_view, TR_TOPO_ABI);
TR_MGPU_ABI(tr_topo_local, TR_TOPO_ABI);
TR_MGPU_ABI(tr_topo_local_totals, TR_TOPO_ABI);
TR_MGPU_ABI(tr_topo_dd, TR_TOPO_ABI);

enum { ACC_SELECTED = 0, ACC_NULL = 1, ACC_OUTSIDE = 2, ACC_VALID = 3, ACC_VALUE_NULLS = 4, ACC_OCCUPIED = 5,
       ACC_VALUED = 6, ACC_ROWS = 8 };

typedef struct { uint64_t *cells[4]; uint32_t *face[4]; int8_t *sign[4]; } tr_topo_out;

/* the occupancy: one bit per occupied position (atomic OR), the valued positions, the per-position value extremes
 * (atomic min / max, build only), the row accounting (per-thread counts, one atomic each) */
TR_KERNEL void tr_topo_occ_kernel(tr_scope_view s, tr_topo_grid g, tr_topo_view v, uint64_t *occ, uint64_t *valued,
                                  int64_t *pmin, int64_t *pmax, uint64_t *acc) {
    uint64_t sel = 0, nul = 0, out = 0, val = 0, vnul = 0;
    TR_GRID_LOOP(r, s.rows) {
        uint64_t p = 0;
        int st = tr_topo_row(&s, &g, &v, r, &p);
        if (st == TR_TOPO_ROW_UNSELECTED) continue;
        sel++;
        if (st == TR_TOPO_ROW_NULL) { nul++; continue; }
        if (st == TR_TOPO_ROW_OUTSIDE) { out++; continue; }
        val++;
        tr_atomic_or_u64(&occ[p >> 6], (uint64_t)1 << (p & 63));
        if (g.value >= 0) {
            if (tr_scope_present(&s, g.value, r)) {
                tr_atomic_or_u64(&valued[p >> 6], (uint64_t)1 << (p & 63));
                if (pmin) {
                    int64_t x = tr_scope_value(&s, g.value, r);
                    tr_atomic_min_i64(&pmin[p], x);
                    tr_atomic_max_i64(&pmax[p], x);
                }
            } else vnul++;
        }
    }
    if (sel) tr_atomic_add_u64(&acc[ACC_SELECTED], sel);
    if (nul) tr_atomic_add_u64(&acc[ACC_NULL], nul);
    if (out) tr_atomic_add_u64(&acc[ACC_OUTSIDE], out);
    if (val) tr_atomic_add_u64(&acc[ACC_VALID], val);
    if (vnul) tr_atomic_add_u64(&acc[ACC_VALUE_NULLS], vnul);
}

TR_KERNEL void tr_topo_popc_kernel(const uint64_t *words, uint64_t n, uint64_t *acc) {
    uint64_t c = 0;
    TR_GRID_LOOP(w, n) c += (uint64_t)tr_popc64(words[w]);
    if (c) tr_atomic_add_u64(acc, c);
}

TR_KERNEL void tr_topo_fill_kernel(int64_t *x, uint64_t n, int64_t value) {
    TR_GRID_LOOP(i, n) x[i] = value;
}

/* the cells and boundary cells per dimension: per-thread counts over a strided set of candidates, one atomic each */
TR_KERNEL void tr_topo_count_kernel(tr_topo_view v, uint64_t *acc) {
    uint64_t n[4] = {0, 0, 0, 0}, b[4] = {0, 0, 0, 0};
    TR_GRID_LOOP(L, v.candidates) {
        uint64_t c[3];
        int32_t k = tr_topo_decode(&v, L, c), m;
        uint32_t inside, pat = tr_topo_pattern(&v, c, &m, &inside);
        if (!pat) continue;
        n[k]++;
        if (tr_topo_is_boundary(m, pat)) b[k]++;
    }
    for (int32_t k = 0; k < 4; k++) {
        if (n[k]) tr_atomic_add_u64(&acc[k], n[k]);
        if (b[k]) tr_atomic_add_u64(&acc[4 + k], b[k]);
    }
}

/* per chunk of TR_TOPO_CHUNK candidates: its cells per dimension, in its own four slots */
TR_KERNEL void tr_topo_chunk_kernel(tr_topo_view v, uint64_t nchunks, uint64_t *counts) {
    TR_GRID_LOOP(q, nchunks) {
        uint64_t n[4] = {0, 0, 0, 0}, end = (q + 1) * TR_TOPO_CHUNK;
        if (end > v.candidates) end = v.candidates;
        for (uint64_t L = q * TR_TOPO_CHUNK; L < end; L++) {
            uint64_t c[3];
            int32_t k = tr_topo_decode(&v, L, c), m;
            uint32_t inside;
            if (tr_topo_pattern(&v, c, &m, &inside)) n[k]++;
        }
        for (int32_t k = 0; k < 4; k++) counts[4 * q + (uint64_t)k] = n[k];
    }
}

/* the cells of each chunk at their prefix offsets, in increasing linear index: one writer per slot */
TR_KERNEL void tr_topo_write_kernel(tr_topo_view v, uint64_t nchunks, const uint64_t *offsets, tr_topo_out o) {
    TR_GRID_LOOP(q, nchunks) {
        uint64_t at[4], end = (q + 1) * TR_TOPO_CHUNK;
        for (int32_t k = 0; k < 4; k++) at[k] = offsets[4 * q + (uint64_t)k];
        if (end > v.candidates) end = v.candidates;
        for (uint64_t L = q * TR_TOPO_CHUNK; L < end; L++) {
            uint64_t c[3];
            int32_t k = tr_topo_decode(&v, L, c), m;
            uint32_t inside;
            if (tr_topo_pattern(&v, c, &m, &inside)) o.cells[k][at[k]++] = L;
        }
    }
}

/* the incidences of dimension k: one work item per k-cell, each face found by binary search, its own slots */
TR_KERNEL void tr_topo_inc_kernel(tr_topo_view v, int32_t k, uint32_t *face, int8_t *sign) {
    TR_GRID_LOOP(j, v.n[k]) {
        uint64_t c[3], f[6], L = v.cells[k][j];
        int8_t sg[6];
        tr_topo_decode(&v, L, c);
        tr_topo_faces(&v, L, c, f, sg);
        for (int32_t s = 0; s < 2 * k; s++) {
            uint64_t at = j * 2 * (uint64_t)k + (uint64_t)s;
            face[at] = (uint32_t)tr_topo_find(v.cells[k - 1], v.n[k - 1], f[s]);
            sign[at] = sg[s];
        }
    }
}

/* the value extremes of each top cell from its position's */
TR_KERNEL void tr_topo_topval_kernel(tr_topo_view v, const int64_t *pmin, const int64_t *pmax, int64_t *vmin,
                                     int64_t *vmax) {
    TR_GRID_LOOP(i, v.n[v.dim]) {
        uint64_t c[3];
        tr_topo_decode(&v, v.cells[v.dim][i], c);
        uint64_t p = tr_topo_top_position(&v, c);
        vmin[i] = pmin[p];
        vmax[i] = pmax[p];
    }
}

/* the local invariants: one work item per cell (dimension-major), its own slot; totals by atomics */
TR_KERNEL void tr_topo_local_kernel(tr_topo_view v, uint64_t total, tr_topo_local *out, uint64_t *acc) {
    uint64_t b[4] = {0, 0, 0, 0}, deg = 0;
    int64_t le[4] = {0, 0, 0, 0};
    TR_GRID_LOOP(i, total) {
        int32_t k = 0;
        uint64_t j = i;
        while (k < 3 && j >= v.n[k]) { j -= v.n[k]; k++; }
        tr_topo_local l;
        tr_topo_local_cell(&v, v.cells[k][j], &l);
        if (out) out[i] = l;
        b[k] += l.boundary;
        le[k] += l.link_euler;
        deg += l.degree;
    }
    for (int32_t k = 0; k < 4; k++) {
        if (b[k]) tr_atomic_add_u64(&acc[k], b[k]);
        if (le[k]) tr_atomic_add_i64((int64_t *)&acc[5 + k], le[k]);
    }
    if (deg) tr_atomic_add_u64(&acc[4], deg);
}

/* d_{k-1} d_k of every k-cell, k = 2 .. dim: one work item per cell; counts by atomics, witnesses by atomic minimum */
TR_KERNEL void tr_topo_dd_kernel(tr_topo_view v, uint64_t total, uint64_t *acc) {
    uint64_t checked[4] = {0, 0, 0, 0}, vz[4] = {0, 0, 0, 0}, v2[4] = {0, 0, 0, 0}, wz[4], w2[4], bad = 0, terms = 0;
    for (int32_t k = 0; k < 4; k++) { wz[k] = TR_TOPO_NONE; w2[k] = TR_TOPO_NONE; }
    TR_GRID_LOOP(i, total) {
        int32_t k = 2;
        uint64_t j = i;
        if (j >= v.n[2]) { j -= v.n[2]; k = 3; }
        int zz, z2, inv;
        terms += tr_topo_dd_cell(&v, k, j, &zz, &z2, &inv);
        checked[k]++;
        if (inv) bad++;
        if (!zz || inv) { vz[k]++; if (j < wz[k]) wz[k] = j; }
        if (!z2 || inv) { v2[k]++; if (j < w2[k]) w2[k] = j; }
    }
    for (int32_t k = 2; k < 4; k++) {
        if (checked[k]) tr_atomic_add_u64(&acc[k], checked[k]);
        if (vz[k]) { tr_atomic_add_u64(&acc[4 + k], vz[k]); tr_atomic_min_u64(&acc[12 + k], wz[k]); }
        if (v2[k]) { tr_atomic_add_u64(&acc[8 + k], v2[k]); tr_atomic_min_u64(&acc[16 + k], w2[k]); }
    }
    if (bad) tr_atomic_add_u64(&acc[20], bad);
    if (terms) tr_atomic_add_u64(&acc[21], terms);
}

/* ------------------------------------------------------------------ host side */
typedef struct { void *p[48]; int n; } tr_topo_devmem;
static int tr_topo_dev(tr_topo_devmem *m, void **ptr, uint64_t bytes, int rc) {
    if (rc) return rc;
    if (m->n >= 48) return -1;
    rc = tr_cuda_alloc(ptr, (size_t)bytes);
    if (!rc) m->p[m->n++] = *ptr;
    return rc;
}
static void tr_topo_dev_free(tr_topo_devmem *m) { for (int i = 0; i < m->n; i++) tr_cuda_free(m->p[i]); m->n = 0; }
/* the device copies of the selection and the columns the grid reads (the view's pointers swapped) */
static int tr_topo_dev_scope(tr_topo_devmem *m, const tr_scope_view *view, const tr_topo_grid *g, tr_scope_view *d) {
    int rc = 0;
    *d = *view;
    uint64_t n = view->rows, nw = (n + 63) / 64;
    uint64_t *sel = NULL;
    rc = tr_topo_dev(m, (void **)&sel, nw * 8, rc);
    if (!rc) rc = tr_cuda_copy_in(sel, view->selection, (size_t)(nw * 8));
    d->selection = sel;
    d->groups = NULL;
    for (int32_t c = 0; c < view->ncols; c++) {
        int used = c == g->value;
        for (int32_t a = 0; a < g->dim; a++) used |= c == g->col[a];
        tr_scope_view_column *k = &d->col[c];
        if (!used) { k->codes = NULL; k->values = NULL; k->valid = NULL; continue; }
        if (view->col[c].values) {
            int64_t *p = NULL;
            rc = tr_topo_dev(m, (void **)&p, n * 8, rc);
            if (!rc) rc = tr_cuda_copy_in(p, view->col[c].values, (size_t)(n * 8));
            k->values = p;
        } else {
            uint32_t *p = NULL;
            rc = tr_topo_dev(m, (void **)&p, n * 4, rc);
            if (!rc && n) rc = tr_cuda_copy_in(p, view->col[c].codes, (size_t)(n * 4));
            k->codes = p;
        }
        if (view->col[c].valid) {
            uint8_t *p = NULL;
            rc = tr_topo_dev(m, (void **)&p, n, rc);
            if (!rc) rc = tr_cuda_copy_in(p, view->col[c].valid, (size_t)n);
            k->valid = p;
        }
    }
    return rc;
}
static int tr_topo_check_scope_view(const tr_scope_view *view) {
    if (!view || view->ncols < 0 || view->ncols > TR_SCOPE_MAX_COLUMNS || (view->rows && !view->selection)) return -1;
    for (int32_t c = 0; c < view->ncols; c++)
        if (view->rows && !view->col[c].codes && !view->col[c].values) return -1;
    return 0;
}

#ifdef __cplusplus
extern "C" {
#endif
/* tr_topo_count's counts on the device; budget: the scan with the device copies */
int tr_mgpu_topo_counts(const tr_scope_view *view, const tr_topo_grid *g, const tr_math_budget *budget,
                        tr_topo_counts *out, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    tr_topo_view v;
    if (!out || tr_topo_check_scope_view(view)) return -1;
    int rc = tr_topo_grid_check_view(view, g, &v);
    if (rc) return rc;
    tr_math_budget dflt;
    if (!budget) { tr_budget_default(&dflt); budget = &dflt; }
    tr_math_need need;
    tr_topo_scan_need(view, g, &v, 1, &need);
    rc = tr_budget_check(budget, &need, refusal);
    if (rc) return rc;
    tr_topo_devmem m;
    m.n = 0;
    tr_scope_view d;
    uint64_t *occ = NULL, *valued = NULL, *acc = NULL, host[ACC_ROWS + 8];
    memset(host, 0, sizeof host);
    rc = tr_topo_dev_scope(&m, view, g, &d);
    rc = tr_topo_dev(&m, (void **)&occ, v.words * 8, rc);
    if (g->value >= 0) rc = tr_topo_dev(&m, (void **)&valued, v.words * 8, rc);
    rc = tr_topo_dev(&m, (void **)&acc, sizeof host, rc);
    if (!rc) {
        TR_LAUNCH(tr_topo_occ_kernel, view->rows, d, *g, v, occ, valued, (int64_t *)NULL, (int64_t *)NULL, acc);
        TR_LAUNCH(tr_topo_popc_kernel, v.words, occ, v.words, acc + ACC_OCCUPIED);
        if (valued) TR_LAUNCH(tr_topo_popc_kernel, v.words, valued, v.words, acc + ACC_VALUED);
        tr_topo_view dv = v;
        dv.occupancy = occ;
        TR_LAUNCH(tr_topo_count_kernel, v.candidates, dv, acc + ACC_ROWS);
        rc = tr_cuda_sync();
    }
    if (!rc) rc = tr_cuda_copy_out(host, acc, sizeof host);
    tr_topo_dev_free(&m);
    if (rc) return rc;
    tr_topo_counts c;
    memset(&c, 0, sizeof c);
    c.rows.selected = host[ACC_SELECTED]; c.rows.null_rows = host[ACC_NULL]; c.rows.outside_rows = host[ACC_OUTSIDE];
    c.rows.valid_rows = host[ACC_VALID]; c.rows.value_nulls = host[ACC_VALUE_NULLS];
    c.rows.occupied = host[ACC_OCCUPIED]; c.rows.valued = host[ACC_VALUED];
    c.rows.duplicates = c.rows.valid_rows - c.rows.occupied;
    for (int32_t k = 0; k < 4; k++) { c.n[k] = host[ACC_ROWS + k]; c.boundary[k] = host[ACC_ROWS + 4 + k]; }
    c.positions = v.positions; c.candidates = v.candidates;
    c.euler = (int64_t)c.n[0] - (int64_t)c.n[1] + (int64_t)c.n[2] - (int64_t)c.n[3];
    c.boundary_euler = (int64_t)c.boundary[0] - (int64_t)c.boundary[1] + (int64_t)c.boundary[2] - (int64_t)c.boundary[3];
    c.dim = g->dim; c.adjacency = g->adjacency;
    *out = c;
    return 0;
}

/* the arrays of the complex into st (capacities: the counts of tr_mgpu_topo_counts); budget: output = cells, tensor =
 * incidences, bytes = the device copies */
int tr_mgpu_topo_build(const tr_scope_view *view, const tr_topo_grid *g, const tr_math_budget *budget,
                       tr_topo_states *st, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    tr_topo_view v;
    if (!st || tr_topo_check_scope_view(view)) return -1;
    int rc = tr_topo_grid_check_view(view, g, &v);
    if (rc) return rc;
    const int32_t dim = g->dim;
    if (st->dim != dim || st->has_value != (g->value >= 0) || st->positions != v.positions || st->words != v.words ||
        (v.words && !st->occupancy)) return -1;
    for (int32_t k = 0; k < 4; k++) {
        if (k > dim && st->n[k]) return -1;
        if (k <= dim && st->n[k] && !st->cells[k]) return -1;
        if (k >= 1 && k <= dim && st->n[k] && (!st->face[k] || !st->sign[k])) return -1;
        if (st->n[k] > UINT32_MAX) return -1;
    }
    if (st->has_value && st->n[dim] && (!st->vmin || !st->vmax)) return -1;
    tr_topo_counts cap;
    memset(&cap, 0, sizeof cap);
    for (int32_t k = 0; k < 4; k++) cap.n[k] = st->n[k];
    cap.positions = v.positions; cap.dim = dim;
    tr_math_budget dflt;
    if (!budget) { tr_budget_default(&dflt); budget = &dflt; }
    tr_math_need need;
    for (int i = 0; i < TR_BUDGET_RESOURCES; i++) need.need[i] = 0;
    need.need[TR_BUDGET_VARIABLES] = (uint64_t)dim;
    need.need[TR_BUDGET_STATES] = v.positions;
    need.need[TR_BUDGET_OUTPUT] = tr_topo_cells_total(st->n);
    need.need[TR_BUDGET_TENSOR] = tr_topo_incidences(st->n);
    need.need[TR_BUDGET_BYTES] = tr_topo_build_device_bytes(view, g, &v, &cap);
    rc = tr_budget_check(budget, &need, refusal);
    if (rc) return rc;
    const uint64_t nchunks = (v.candidates + TR_TOPO_CHUNK - 1) / TR_TOPO_CHUNK;
    uint64_t *counts = (uint64_t *)calloc((size_t)(nchunks ? 4 * nchunks : 4), 8);
    if (!counts) return -2;
    tr_topo_devmem m;
    m.n = 0;
    tr_scope_view d;
    uint64_t *occ = NULL, *valued = NULL, *acc = NULL, *dcounts = NULL, host[ACC_ROWS + 8];
    int64_t *pmin = NULL, *pmax = NULL, *tmin = NULL, *tmax = NULL;
    memset(host, 0, sizeof host);
    rc = tr_topo_dev_scope(&m, view, g, &d);
    rc = tr_topo_dev(&m, (void **)&occ, v.words * 8, rc);
    rc = tr_topo_dev(&m, (void **)&acc, sizeof host, rc);
    rc = tr_topo_dev(&m, (void **)&dcounts, (nchunks ? 4 * nchunks : 4) * 8, rc);
    if (st->has_value) {
        rc = tr_topo_dev(&m, (void **)&valued, v.words * 8, rc);
        rc = tr_topo_dev(&m, (void **)&pmin, v.positions * 8, rc);
        rc = tr_topo_dev(&m, (void **)&pmax, v.positions * 8, rc);
        if (!rc) {
            TR_LAUNCH(tr_topo_fill_kernel, v.positions, pmin, v.positions, (int64_t)INT64_MAX);
            TR_LAUNCH(tr_topo_fill_kernel, v.positions, pmax, v.positions, (int64_t)INT64_MIN);
        }
    }
    if (!rc) {
        TR_LAUNCH(tr_topo_occ_kernel, view->rows, d, *g, v, occ, valued, pmin, pmax, acc);
        TR_LAUNCH(tr_topo_popc_kernel, v.words, occ, v.words, acc + ACC_OCCUPIED);
        if (valued) TR_LAUNCH(tr_topo_popc_kernel, v.words, valued, v.words, acc + ACC_VALUED);
        rc = tr_cuda_sync();
    }
    tr_topo_view dv = v;
    dv.occupancy = occ;
    if (!rc) {
        TR_LAUNCH(tr_topo_chunk_kernel, nchunks, dv, nchunks, dcounts);
        rc = tr_cuda_sync();
    }
    if (!rc) rc = tr_cuda_copy_out(counts, dcounts, (size_t)(nchunks * 4 * 8));
    uint64_t total[4] = {0, 0, 0, 0};
    if (!rc) {                                       /* the exclusive prefix per dimension, then the totals */
        for (uint64_t q = 0; q < nchunks; q++)
            for (int32_t k = 0; k < 4; k++) { uint64_t c = counts[4 * q + (uint64_t)k]; counts[4 * q + (uint64_t)k] = total[k]; total[k] += c; }
        for (int32_t k = 0; k < 4; k++) if (total[k] != st->n[k]) rc = -1;      /* the capacities are the counts */
    }
    if (!rc) rc = tr_cuda_copy_in(dcounts, counts, (size_t)(nchunks * 4 * 8));
    tr_topo_out o;
    memset(&o, 0, sizeof o);
    for (int32_t k = 0; k <= dim && !rc; k++) {
        rc = tr_topo_dev(&m, (void **)&o.cells[k], st->n[k] * 8, rc);
        if (k >= 1) {
            rc = tr_topo_dev(&m, (void **)&o.face[k], st->n[k] * 2 * (uint64_t)k * 4, rc);
            rc = tr_topo_dev(&m, (void **)&o.sign[k], st->n[k] * 2 * (uint64_t)k, rc);
        }
        dv.n[k] = st->n[k];
        dv.cells[k] = o.cells[k];
    }
    if (!rc) {
        TR_LAUNCH(tr_topo_write_kernel, nchunks, dv, nchunks, dcounts, o);
        rc = tr_cuda_sync();
    }
    for (int32_t k = 1; k <= dim && !rc; k++) {
        if (st->n[k]) TR_LAUNCH(tr_topo_inc_kernel, st->n[k], dv, k, o.face[k], o.sign[k]);
        rc = tr_cuda_sync();
    }
    if (!rc && st->has_value) {
        rc = tr_topo_dev(&m, (void **)&tmin, st->n[dim] * 8, rc);
        rc = tr_topo_dev(&m, (void **)&tmax, st->n[dim] * 8, rc);
        if (!rc && st->n[dim]) TR_LAUNCH(tr_topo_topval_kernel, st->n[dim], dv, pmin, pmax, tmin, tmax);
        if (!rc) rc = tr_cuda_sync();
    }
    if (!rc) rc = tr_cuda_copy_out(host, acc, sizeof host);
    if (!rc) rc = tr_cuda_copy_out(st->occupancy, occ, (size_t)(v.words * 8));
    for (int32_t k = 0; k <= dim && !rc; k++) {
        rc = tr_cuda_copy_out(st->cells[k], o.cells[k], (size_t)(st->n[k] * 8));
        if (!rc && k >= 1) rc = tr_cuda_copy_out(st->face[k], o.face[k], (size_t)(st->n[k] * 2 * (uint64_t)k * 4));
        if (!rc && k >= 1) rc = tr_cuda_copy_out(st->sign[k], o.sign[k], (size_t)(st->n[k] * 2 * (uint64_t)k));
    }
    if (!rc && st->has_value) {
        rc = tr_cuda_copy_out(st->vmin, tmin, (size_t)(st->n[dim] * 8));
        if (!rc) rc = tr_cuda_copy_out(st->vmax, tmax, (size_t)(st->n[dim] * 8));
    }
    tr_topo_dev_free(&m);
    free(counts);
    if (rc) return rc;
    tr_topo_rows *r = &st->rows;
    memset(r, 0, sizeof *r);
    r->selected = host[ACC_SELECTED]; r->null_rows = host[ACC_NULL]; r->outside_rows = host[ACC_OUTSIDE];
    r->valid_rows = host[ACC_VALID]; r->value_nulls = host[ACC_VALUE_NULLS];
    r->occupied = host[ACC_OCCUPIED]; r->valued = host[ACC_VALUED];
    r->duplicates = r->valid_rows - r->occupied;
    return 0;
}

/* tr_topo_local_view on the device; budget: output = cells (with out), bytes = the device copies */
int tr_mgpu_topo_local(const tr_topo_view *view, const tr_math_budget *budget, tr_topo_local *out, uint64_t capacity,
                       tr_topo_local_totals *tot, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    int rc = tr_topo_view_check(view);
    if (rc) return rc;
    const uint64_t total = tr_topo_cells_total(view->n);
    if (out && capacity < total) return -1;
    tr_math_budget dflt;
    if (!budget) { tr_budget_default(&dflt); budget = &dflt; }
    tr_math_need need;
    for (int i = 0; i < TR_BUDGET_RESOURCES; i++) need.need[i] = 0;
    need.need[TR_BUDGET_OUTPUT] = out ? total : 0;
    need.need[TR_BUDGET_BYTES] = tr_topo_local_device_bytes(view);
    rc = tr_budget_check(budget, &need, refusal);
    if (rc) return rc;
    tr_topo_devmem m;
    m.n = 0;
    tr_topo_view d = *view;
    uint64_t *occ = NULL, *acc = NULL, host[16];
    tr_topo_local *dout = NULL;
    memset(host, 0, sizeof host);
    rc = tr_topo_dev(&m, (void **)&occ, view->words * 8, rc);
    if (!rc) rc = tr_cuda_copy_in(occ, view->occupancy, (size_t)(view->words * 8));
    d.occupancy = occ;
    for (int32_t k = 0; k < 4; k++) {
        d.face[k] = NULL; d.sign[k] = NULL; d.cells[k] = NULL;
        if (k > view->dim) continue;
        uint64_t *p = NULL;
        rc = tr_topo_dev(&m, (void **)&p, view->n[k] * 8, rc);
        if (!rc && view->n[k]) rc = tr_cuda_copy_in(p, view->cells[k], (size_t)(view->n[k] * 8));
        d.cells[k] = p;
    }
    if (out) rc = tr_topo_dev(&m, (void **)&dout, total * sizeof(tr_topo_local), rc);
    rc = tr_topo_dev(&m, (void **)&acc, sizeof host, rc);
    if (!rc && total) {
        TR_LAUNCH(tr_topo_local_kernel, total, d, total, dout, acc);
        rc = tr_cuda_sync();
    }
    if (!rc) rc = tr_cuda_copy_out(host, acc, sizeof host);
    if (!rc && out) rc = tr_cuda_copy_out(out, dout, (size_t)(total * sizeof(tr_topo_local)));
    tr_topo_dev_free(&m);
    if (rc) return rc;
    if (tot) {
        memset(tot, 0, sizeof *tot);
        tot->cells = total;
        for (int32_t k = 0; k < 4; k++) { tot->boundary[k] = host[k]; tot->link_euler_sum[k] = (int64_t)host[5 + k]; }
        tot->dual_edges = host[4] / 2;
    }
    return 0;
}

/* tr_topo_check_view on the device; budget: tensor = composition terms, bytes = the device copies */
int tr_mgpu_topo_dd(const tr_topo_view *view, const tr_math_budget *budget, tr_topo_dd *out, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    int rc = tr_topo_view_check(view);
    if (rc || !out) return -1;
    tr_math_budget dflt;
    if (!budget) { tr_budget_default(&dflt); budget = &dflt; }
    tr_math_need need;
    for (int i = 0; i < TR_BUDGET_RESOURCES; i++) need.need[i] = 0;
    need.need[TR_BUDGET_TENSOR] = tr_budget_add(tr_budget_mul(view->n[2], 8), tr_budget_mul(view->n[3], 24));
    need.need[TR_BUDGET_BYTES] = tr_topo_dd_device_bytes(view);
    rc = tr_budget_check(budget, &need, refusal);
    if (rc) return rc;
    tr_topo_devmem m;
    m.n = 0;
    tr_topo_view d = *view;
    uint64_t *acc = NULL, host[24];
    for (int32_t k = 0; k < 24; k++) host[k] = (k >= 12 && k < 20) ? TR_TOPO_NONE : 0;
    d.occupancy = NULL;
    for (int32_t k = 0; k < 4; k++) {
        d.cells[k] = NULL; d.face[k] = NULL; d.sign[k] = NULL;
        if (k < 1 || k > view->dim) continue;
        uint32_t *f = NULL;
        int8_t *s = NULL;
        uint64_t e = view->n[k] * 2 * (uint64_t)k;
        rc = tr_topo_dev(&m, (void **)&f, e * 4, rc);
        rc = tr_topo_dev(&m, (void **)&s, e, rc);
        if (!rc && e) rc = tr_cuda_copy_in(f, view->face[k], (size_t)(e * 4));
        if (!rc && e) rc = tr_cuda_copy_in(s, view->sign[k], (size_t)e);
        d.face[k] = f; d.sign[k] = s;
    }
    rc = tr_topo_dev(&m, (void **)&acc, sizeof host, rc);
    if (!rc) rc = tr_cuda_copy_in(acc, host, sizeof host);
    const uint64_t total = (view->dim >= 2 ? view->n[2] : 0) + (view->dim >= 3 ? view->n[3] : 0);
    if (!rc && total) {
        TR_LAUNCH(tr_topo_dd_kernel, total, d, total, acc);
        rc = tr_cuda_sync();
    }
    if (!rc) rc = tr_cuda_copy_out(host, acc, sizeof host);
    tr_topo_dev_free(&m);
    if (rc) return rc;
    tr_topo_dd e;
    tr_topo_dd_init(&e);
    for (int32_t k = 2; k < 4; k++) {
        e.checked[k] = host[k]; e.violations_z[k] = host[4 + k]; e.violations_gf2[k] = host[8 + k];
        e.witness_z[k] = host[12 + k]; e.witness_gf2[k] = host[16 + k];
    }
    e.invalid = host[20];
    e.terms = host[21];
    tr_topo_dd_finish(&e);
    *out = e;
    return 0;
}
#ifdef __cplusplus
}
#endif
