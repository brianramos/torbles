/* TENSOR-001 / TENSOR-002 on a CUDA device: moment and count tensors built from a scope (dense cells with exact
 * atomics, or a hashed sparse table), and the dense steps of a contraction program, with exactly the results of
 * tr_tensor_build and tr_tensor_net_execute (math_tensor.h).
 * SPDX-License-Identifier: Apache-2.0
 *
 * Dense builds: a MOMENT work item is (entry e, row slice s): slice s of S takes rows s, s + S, ... and accumulates
 * entry e's products locally while the group stays the same, then adds them to the group's cell with the exact
 * two-word atomic of math_cuda.h (tr_atomic_add_wide; counts with a 64-bit atomic). A symmetric tensor (one column
 * list on every axis) runs work items on its canonical (sorted) entries only and then copies each entry from its
 * canonical one (one work item per cell, reading final values only). A COUNT work item is one row: one atomic add
 * into its cell. Sparse builds: one work item per row inserts each of its emissions into an open-addressing table of
 * 2^k >= 2 x capacity slots: a 64-bit compare-and-swap (a local equivalent of the math_cuda.h atomics, since that
 * header has none) claims the key's slot, exact atomics add into it; the host compacts the occupied slots and sorts
 * them into the canonical CSR (any insertion order gives the same set of keys and sums). Integer atomics commute, so
 * nothing depends on the thread order; every sum is exact because the host proved the build's bound before launching.
 *
 * Contraction: each step is one launch with one work item per result entry (tr_tensor_dense_entry, the CPU's
 * evaluator); the host checks the program's shapes and strides against its inputs and re-proves every step's bound
 * from the inputs' exact maxima before anything is copied. */
#include "math_cuda.h"
#include "math_tensor.h"
#include "math_handles.h"

TR_MGPU_ABI(tr_tensor_spec, 1);          /* structs passed to the CUDA library (math_handles.h) */
TR_MGPU_ABI(tr_tensor_states, 1);
TR_MGPU_ABI(tr_tensor_dense_step, 1);
TR_MGPU_ABI(tr_tensor_program, 1);
TR_MGPU_ABI(tr_tensor_dense_view, 1);
TR_MGPU_ABI(tr_tensor_exec, 1);

#define TR_TENSOR_WORK (UINT64_C(1) << 19)    /* about this many work items per launch */
#define TR_TENSOR_MIN_ROWS 32                  /* and at least this many rows per work item */
#define TR_TENSOR_EMPTY UINT64_MAX             /* a free hash slot */

#ifdef EMULATE
static inline uint64_t tr_tensor_atomic_cas(uint64_t *p, uint64_t cmp, uint64_t val) {
    uint64_t o = *p;
    if (o == cmp) *p = val;
    return o;
}
#else
__device__ static inline uint64_t tr_tensor_atomic_cas(uint64_t *p, uint64_t cmp, uint64_t val) {
    return (uint64_t)atomicCAS((unsigned long long *)p, (unsigned long long)cmp, (unsigned long long)val);
}
#endif

/* selected rows and contributing rows per group (one work item per slice) */
TR_KERNEL void tr_tensor_rows_kernel(tr_scope_view v, tr_tensor_build_plan p, uint64_t slices, uint64_t *rows, uint64_t *used) {
    TR_GRID_LOOP(s, slices) {
        uint32_t cur = 0;
        uint64_t c = 0, u = 0;
        for (uint64_t r = s; r < v.rows; r += slices) {
            if (!tr_scope_is_selected(&v, r)) continue;
            uint32_t g = tr_scope_group(&v, r);
            if (c && g != cur) {
                tr_atomic_add_u64(&rows[cur], c);
                if (u) tr_atomic_add_u64(&used[cur], u);
                c = 0; u = 0;
            }
            cur = g; c++;
            __int128 w;
            u += (uint64_t)tr_tensor_row_ok(&v, &p, r, &w);
        }
        if (c) { tr_atomic_add_u64(&rows[cur], c); if (u) tr_atomic_add_u64(&used[cur], u); }
    }
}

/* MOMENT dense: work item (entry list[j] or j, slice) */
TR_KERNEL void tr_tensor_moment_kernel(tr_scope_view v, tr_tensor_build_plan p, const uint64_t *entries, uint64_t nentries,
                                       uint64_t slices, int64_t *value, uint64_t *count) {
    TR_GRID_LOOP(i, nentries * slices) {
        uint64_t j = i / slices, s = i % slices, e = entries ? entries[j] : j;
        int8_t col[TR_TENSOR_MAX_ORDER];
        for (int32_t k = 0; k < p.order; k++) col[k] = p.cols[k][(e / p.stride[k]) % p.dims[k]];
        uint32_t cur = 0;
        uint64_t c = 0;
        __int128 acc = 0;
        for (uint64_t r = s; r < v.rows; r += slices) {
            if (!tr_scope_is_selected(&v, r)) continue;
            __int128 w;
            if (!tr_tensor_row_ok(&v, &p, r, &w)) continue;
            __int128 x = w;
            int ok = 1;
            for (int32_t k = 0; k < p.order && ok; k++) {
                if (col[k] < 0) continue;
                if (!tr_scope_present(&v, col[k], r)) { ok = 0; break; }
                x *= (__int128)tr_scope_value(&v, col[k], r);
            }
            if (!ok) continue;
            uint32_t g = tr_scope_group(&v, r);
            if (c && g != cur) {
                uint64_t q = (uint64_t)cur * p.cells + e;
                if (acc) tr_atomic_add_wide(&value[2 * q], &value[2 * q + 1], acc);
                if (count) tr_atomic_add_u64(&count[q], c);
                c = 0; acc = 0;
            }
            cur = g; c++; acc += x;
        }
        if (c) {
            uint64_t q = (uint64_t)cur * p.cells + e;
            if (acc) tr_atomic_add_wide(&value[2 * q], &value[2 * q + 1], acc);
            if (count) tr_atomic_add_u64(&count[q], c);
        }
    }
}

/* a symmetric dense state: every cell from its canonical cell (written by the moment kernel, read only here) */
TR_KERNEL void tr_tensor_symmetrize_kernel(tr_tensor_build_plan p, uint64_t ngroups, int64_t *value, uint64_t *count) {
    TR_GRID_LOOP(i, ngroups * p.cells) {
        uint64_t g = i / p.cells, e = i % p.cells;
        uint64_t c = tr_tensor_canonical(p.dims, p.stride, p.order, e);
        if (c == e) continue;
        uint64_t q = g * p.cells + c;
        value[2 * i] = value[2 * q]; value[2 * i + 1] = value[2 * q + 1];
        if (count) count[i] = count[q];
    }
}

/* COUNT dense: one work item per row */
TR_KERNEL void tr_tensor_count_kernel(tr_scope_view v, tr_tensor_build_plan p, int64_t *value) {
    TR_GRID_LOOP(r, v.rows) {
        if (!tr_scope_is_selected(&v, r)) continue;
        __int128 w;
        if (!tr_tensor_row_ok(&v, &p, r, &w)) continue;
        uint64_t q = (uint64_t)tr_scope_group(&v, r) * p.cells + tr_tensor_count_cell(&v, &p, r);
        if (w) tr_atomic_add_wide(&value[2 * q], &value[2 * q + 1], w);
    }
}

TR_KERNEL void tr_tensor_fill_kernel(uint64_t *keys, uint64_t n) {
    TR_GRID_LOOP(i, n) keys[i] = TR_TENSOR_EMPTY;
}

/* sparse: every emission of a row into the hash table (key = group x cells + index) */
TR_KERNEL void tr_tensor_hash_kernel(tr_scope_view v, tr_tensor_build_plan p, uint64_t slots, uint64_t *keys,
                                     int64_t *value, uint64_t *count, uint64_t *full) {
    TR_GRID_LOOP(r, v.rows) {
        if (!tr_scope_is_selected(&v, r)) continue;
        __int128 w;
        if (!tr_tensor_row_ok(&v, &p, r, &w)) continue;
        uint64_t g = tr_scope_group(&v, r);
        tr_tensor_emitter e;
        for (int more = tr_tensor_emit_first(&v, &p, r, w, 0, &e); more; more = tr_tensor_emit_next(&v, &p, &e)) {
            uint64_t key = g * p.cells + e.idx[p.order], slot = tr_tensor_hash64(key) & (slots - 1), probes = 0;
            for (;;) {
                uint64_t prev = tr_tensor_atomic_cas(&keys[slot], TR_TENSOR_EMPTY, key);
                if (prev == TR_TENSOR_EMPTY || prev == key) break;
                slot = (slot + 1) & (slots - 1);
                if (++probes >= slots) { tr_atomic_or_u64(full, 1); slot = slots; break; }
            }
            if (slot >= slots) continue;
            __int128 x = e.pre[p.order];
            if (x) tr_atomic_add_wide(&value[2 * slot], &value[2 * slot + 1], x);
            if (count) tr_atomic_add_u64(&count[slot], 1);
        }
    }
}

/* one dense contraction step: one work item per result entry */
TR_KERNEL void tr_tensor_contract_kernel(tr_tensor_dense_step s, const tr_math_i128 *A, const tr_math_i128 *B,
                                         tr_math_i128 *R) {
    TR_GRID_LOOP(o, s.cells) R[o] = tr_i128_set(tr_tensor_dense_entry(&s, A, B, o));
}

static uint64_t tr_tensor_slices(uint64_t rows, uint64_t work) {
    uint64_t s = TR_TENSOR_WORK / (work ? work : 1), cap = (rows + TR_TENSOR_MIN_ROWS - 1) / TR_TENSOR_MIN_ROWS;
    if (s > cap) s = cap;
    return s ? s : 1;
}
typedef struct { uint64_t key; __int128 v; uint64_t c; } tr_tensor_slot;
static int tr_tensor_slot_cmp(const void *x, const void *y) {
    const tr_tensor_slot *a = (const tr_tensor_slot *)x, *b = (const tr_tensor_slot *)y;
    return a->key < b->key ? -1 : a->key > b->key;
}

#ifdef __cplusplus
extern "C" {
#endif
int tr_mgpu_tensor_build(const tr_scope_view *view, const tr_tensor_spec *spec, const tr_math_budget *budget,
                         tr_tensor_states *out, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!view || !spec || !out || view->ngroups < 1 || (view->rows && !view->selection) ||
        (view->rows && view->ngroups > 1 && !view->groups) || view->ncols < 0 || view->ncols > TR_SCOPE_MAX_COLUMNS)
        return -1;                                                                  /* TR_MATH_INVALID */
    tr_tensor_build_plan p;
    int rc = tr_tensor_plan_make(view, spec, &p);
    if (rc) return rc;
    if (p.repr == TR_TENSOR_FACTORED) return TR_MATH_UNSUPPORTED;
    const uint64_t G = view->ngroups, n = view->rows, total = tr_budget_mul(G, p.cells);
    if (total == TR_BUDGET_UNKNOWN) return TR_MATH_UNSUPPORTED;
    tr_math_need need;
    uint64_t cap = 0;
    tr_tensor_device_need(view, &p, &need, &cap);
    tr_math_budget dflt;
    if (!budget) { tr_budget_default(&dflt); budget = &dflt; }
    rc = tr_budget_check(budget, &need, refusal);
    if (rc) return rc;
    if (tr_tensor_plan_bound(view, &p) > TR_TENSOR_I128_MAX) return -3;            /* TR_MATH_OVERFLOW */
    if (out->ngroups != G || out->order != p.order || out->repr != p.repr || out->has_counts != p.has_counts ||
        !out->rows || !out->used || !out->value || (p.has_counts && !out->count) ||
        (p.repr == TR_TENSOR_DENSE && out->capacity < total) ||
        (p.repr == TR_TENSOR_SPARSE && (out->capacity < cap || !out->offset || (p.order && !out->coord)))) return -1;
    for (int32_t k = 0; k < p.order; k++) if (out->dims[k] != p.dims[k]) return -1;
    out->cells = p.cells;
    memset(out->rows, 0, (size_t)G * 8); memset(out->used, 0, (size_t)G * 8);
    if (p.repr == TR_TENSOR_DENSE) {
        out->nnz = total;
        if (total) memset(out->value, 0, (size_t)total * 16);
        if (p.has_counts && total) memset(out->count, 0, (size_t)total * 8);
    } else {
        out->nnz = 0;
        memset(out->offset, 0, (size_t)(G + 1) * 8);
    }
    if (!view->selected) return 0;                    /* nothing selected: every state is zero */
    /* device copies of the scope's arrays the plan reads */
    tr_scope_view d = *view;
    void *dmem[3 * TR_SCOPE_MAX_COLUMNS + 12];
    int nmem = 0;
#define TR_TENSOR_DEV(ptr, bytes) do { if (!rc) { rc = tr_cuda_alloc((void **)&(ptr), (size_t)(bytes)); if (!rc) dmem[nmem++] = (void *)(ptr); } } while (0)
    uint64_t *dsel = NULL, *drows = NULL, *dused = NULL, *dcount = NULL, *dentries = NULL, *dkeys = NULL, *dfull = NULL;
    uint32_t *dgroups = NULL;
    int64_t *dvalue = NULL;
    TR_TENSOR_DEV(dsel, ((n + 63) / 64) * 8);
    if (!rc) rc = tr_cuda_copy_in(dsel, view->selection, (size_t)(((n + 63) / 64) * 8));
    d.selection = dsel;
    if (view->groups) {
        TR_TENSOR_DEV(dgroups, n * 4);
        if (!rc) rc = tr_cuda_copy_in(dgroups, view->groups, (size_t)n * 4);
        d.groups = dgroups;
    }
    uint32_t read = p.refmask;
    if (p.kind == TR_TENSOR_COUNT) for (int32_t k = 0; k < p.order; k++) read |= UINT32_C(1) << p.cols[k][0];
    if (p.weight >= 0) read |= UINT32_C(1) << p.weight;
    for (int32_t c = 0; c < view->ncols; c++) {
        tr_scope_view_column *k = &d.col[c];
        if (!((read >> c) & 1)) { k->codes = NULL; k->values = NULL; k->valid = NULL; continue; }   /* never read */
        if (view->col[c].values) {
            int64_t *q = NULL;
            TR_TENSOR_DEV(q, n * 8);
            if (!rc) rc = tr_cuda_copy_in(q, view->col[c].values, (size_t)n * 8);
            k->values = q;
        } else {
            uint32_t *q = NULL;
            TR_TENSOR_DEV(q, n * 4);
            if (!rc) rc = tr_cuda_copy_in(q, view->col[c].codes, (size_t)n * 4);
            k->codes = q;
        }
        if (view->col[c].valid) {
            uint8_t *q = NULL;
            TR_TENSOR_DEV(q, n);
            if (!rc) rc = tr_cuda_copy_in(q, view->col[c].valid, (size_t)n);
            k->valid = q;
        }
    }
    TR_TENSOR_DEV(drows, G * 8);
    TR_TENSOR_DEV(dused, G * 8);
    uint64_t slots = 0, *hkeys = NULL, *hcount = NULL, full = 0;
    int64_t *hvalue = NULL;
    if (p.repr == TR_TENSOR_DENSE) {
        TR_TENSOR_DEV(dvalue, total * 16);
        if (p.has_counts) TR_TENSOR_DEV(dcount, total * 8);
    } else {
        slots = tr_tensor_hash_slots(cap);
        TR_TENSOR_DEV(dkeys, slots * 8);
        TR_TENSOR_DEV(dvalue, slots * 16);
        if (p.has_counts) TR_TENSOR_DEV(dcount, slots * 8);
        TR_TENSOR_DEV(dfull, 8);
    }
    uint64_t *canon = NULL, ncanon = 0;
    if (!rc && p.repr == TR_TENSOR_DENSE && p.kind == TR_TENSOR_MOMENT && p.symmetric) {
        canon = (uint64_t *)malloc((size_t)(p.cells ? p.cells : 1) * 8);
        if (!canon) rc = -2;                                                         /* TR_MATH_MEMORY */
        for (uint64_t e = 0; !rc && e < p.cells; e++)
            if (tr_tensor_canonical(p.dims, p.stride, p.order, e) == e) canon[ncanon++] = e;
        TR_TENSOR_DEV(dentries, ncanon * 8);
        if (!rc) rc = tr_cuda_copy_in(dentries, canon, (size_t)ncanon * 8);
    }
#undef TR_TENSOR_DEV
    if (!rc) {
        uint64_t s = tr_tensor_slices(n, 1);
        TR_LAUNCH(tr_tensor_rows_kernel, s, d, p, s, drows, dused);
        if (p.repr == TR_TENSOR_DENSE && p.kind == TR_TENSOR_MOMENT) {
            uint64_t ne = dentries ? ncanon : p.cells, t = tr_tensor_slices(n, ne);
            if (ne) TR_LAUNCH(tr_tensor_moment_kernel, ne * t, d, p, dentries, ne, t, dvalue, dcount);
            rc = tr_cuda_sync();
            if (!rc && dentries && total) TR_LAUNCH(tr_tensor_symmetrize_kernel, total, p, G, dvalue, dcount);
        } else if (p.repr == TR_TENSOR_DENSE) {
            TR_LAUNCH(tr_tensor_count_kernel, n, d, p, dvalue);
        } else {
            TR_LAUNCH(tr_tensor_fill_kernel, slots, dkeys, slots);
            rc = tr_cuda_sync();
            if (!rc) TR_LAUNCH(tr_tensor_hash_kernel, n, d, p, slots, dkeys, dvalue, dcount, dfull);
        }
        if (!rc) rc = tr_cuda_sync();
    }
    if (!rc) rc = tr_cuda_copy_out(out->rows, drows, (size_t)G * 8);
    if (!rc) rc = tr_cuda_copy_out(out->used, dused, (size_t)G * 8);
    if (!rc && p.repr == TR_TENSOR_DENSE) {
        if (total) rc = tr_cuda_copy_out(out->value, dvalue, (size_t)total * 16);
        if (!rc && p.has_counts && total) rc = tr_cuda_copy_out(out->count, dcount, (size_t)total * 8);
    } else if (!rc) {
        hkeys = (uint64_t *)malloc((size_t)slots * 8);
        hvalue = (int64_t *)malloc((size_t)slots * 16);
        if (p.has_counts) hcount = (uint64_t *)malloc((size_t)slots * 8);
        if (!hkeys || !hvalue || (p.has_counts && !hcount)) rc = -2;
        if (!rc) rc = tr_cuda_copy_out(hkeys, dkeys, (size_t)slots * 8);
        if (!rc) rc = tr_cuda_copy_out(hvalue, dvalue, (size_t)slots * 16);
        if (!rc && p.has_counts) rc = tr_cuda_copy_out(hcount, dcount, (size_t)slots * 8);
        if (!rc) rc = tr_cuda_copy_out(&full, dfull, 8);
        if (!rc && full) rc = -1;                    /* unreachable: the table holds twice the capacity */
    }
    for (int i = 0; i < nmem; i++) tr_cuda_free(dmem[i]);
    free(canon);
    if (!rc && p.repr == TR_TENSOR_SPARSE) {        /* the occupied slots, sorted: the canonical CSR */
        tr_tensor_slot *occ = (tr_tensor_slot *)malloc((size_t)(cap ? cap : 1) * sizeof(tr_tensor_slot));
        uint64_t m = 0;
        if (!occ) rc = -2;
        for (uint64_t i = 0; !rc && i < slots; i++) {
            if (hkeys[i] == TR_TENSOR_EMPTY) continue;
            tr_math_i128 w;
            w.lo = hvalue[2 * i]; w.hi = hvalue[2 * i + 1];
            __int128 v = tr_i128_get(w);
            uint64_t c = hcount ? hcount[i] : 0;
            if (hcount ? c == 0 : v == 0) continue;  /* sums that cancelled (or zero weights) store nothing */
            if (m >= cap) { rc = -1; break; }
            occ[m].key = hkeys[i]; occ[m].v = v; occ[m].c = c; m++;
        }
        if (!rc && m > 1) qsort(occ, (size_t)m, sizeof *occ, tr_tensor_slot_cmp);
        for (uint64_t i = 0; !rc && i < m; i++) {
            uint64_t g = occ[i].key / (p.cells ? p.cells : 1), e = occ[i].key % (p.cells ? p.cells : 1);
            out->offset[g + 1]++;
            for (int32_t k = 0; k < p.order; k++) out->coord[i * (uint64_t)p.order + (uint64_t)k] = (uint32_t)((e / p.stride[k]) % p.dims[k]);
            out->value[i] = tr_i128_set(occ[i].v);
            if (p.has_counts) out->count[i] = occ[i].c;
        }
        if (!rc) { for (uint64_t g = 0; g < G; g++) out->offset[g + 1] += out->offset[g]; out->nnz = m; }
        free(occ);
    }
    free(hkeys); free(hvalue); free(hcount);
    return rc;
}

int tr_mgpu_tensor_contract(const tr_tensor_program *prog, const tr_tensor_dense_view *inputs,
                            const tr_math_budget *budget, tr_math_i128 *out, tr_tensor_exec *rec,
                            tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!prog || !inputs || !out || prog->ninputs < 1 || prog->ninputs > TR_TENSOR_NET_MAX || prog->nsteps < 1 ||
        prog->nsteps > TR_TENSOR_MAX_STEPS) return -1;
    const int32_t ni = prog->ninputs, ns = prog->nsteps;
    tr_tensor_u128 bound[TR_TENSOR_NET_MAX + TR_TENSOR_MAX_STEPS];
    uint64_t cells[TR_TENSOR_NET_MAX + TR_TENSOR_MAX_STEPS], max_tensor = 0;
    int consumed[TR_TENSOR_NET_MAX + TR_TENSOR_MAX_STEPS];
    for (int32_t i = 0; i < ni; i++) {               /* the inputs as the program describes them, their exact maxima */
        const tr_tensor_dense_view *v = &inputs[i];
        if ((v->cells && !v->value) || v->order < 0 || v->order > TR_TENSOR_MAX_ORDER || v->cells != prog->input_cells[i] ||
            tr_tensor_cells(v->dims, v->order) != v->cells) return -1;
        tr_tensor_u128 b = 0;
        for (uint64_t e = 0; e < v->cells; e++) { tr_tensor_u128 m = tr_tensor_mag(tr_i128_get(v->value[e])); if (m > b) b = m; }
        bound[i] = b; cells[i] = v->cells; consumed[i] = 0;
        if (v->cells > max_tensor) max_tensor = v->cells;
    }
    for (int32_t s = 0; s < ns; s++) {               /* shapes, strides and the bound of every step */
        const tr_tensor_dense_step *st = &prog->step[s];
        int32_t slot = ni + s;
        if (st->a < 0 || st->a >= slot || st->b < -1 || st->b >= slot || st->a == st->b || consumed[st->a] ||
            (st->b >= 0 && consumed[st->b]) || st->nout < 0 || st->nout > TR_TENSOR_MAX_LABELS || st->nsum < 0 ||
            st->nsum > TR_TENSOR_MAX_LABELS || st->nout + st->nsum > TR_TENSOR_MAX_LABELS) return -1;
        consumed[st->a] = 1;
        if (st->b >= 0) consumed[st->b] = 1;
        uint64_t c = 1, t = 1;
        tr_tensor_u128 ma = 0, mb = 0;
        for (int32_t k = 0; k < st->nout; k++) {
            c = tr_budget_mul(c, st->dim_out[k]);
            if (st->dim_out[k]) { ma += (tr_tensor_u128)(st->dim_out[k] - 1) * st->sa_out[k]; mb += (tr_tensor_u128)(st->dim_out[k] - 1) * st->sb_out[k]; }
        }
        for (int32_t k = 0; k < st->nsum; k++) {
            t = tr_budget_mul(t, st->dim_sum[k]);
            if (st->dim_sum[k]) { ma += (tr_tensor_u128)(st->dim_sum[k] - 1) * st->sa_sum[k]; mb += (tr_tensor_u128)(st->dim_sum[k] - 1) * st->sb_sum[k]; }
        }
        if (c != st->cells || t != st->terms || c == TR_BUDGET_UNKNOWN || t == TR_BUDGET_UNKNOWN) return -1;
        if (c && t && (ma >= cells[st->a] || (st->b >= 0 && mb >= cells[st->b]))) return -1;   /* inside the operands */
        tr_tensor_u128 bb = st->b >= 0 ? bound[st->b] : 1;
        tr_tensor_u128 b = tr_tensor_satmul(tr_tensor_satmul(t, bound[st->a], TR_TENSOR_I128_MAX), bb, TR_TENSOR_I128_MAX);
        if (!t || !bound[st->a] || !bb) b = 0;
        if (b > TR_TENSOR_I128_MAX) return -3;       /* TR_MATH_OVERFLOW: never an unproven sum */
        bound[slot] = b; cells[slot] = c; consumed[slot] = 0;
        if (c > max_tensor) max_tensor = c;
    }
    if (cells[ni + ns - 1] != prog->output_cells) return -1;
    tr_math_need need;
    for (int i = 0; i < TR_BUDGET_RESOURCES; i++) need.need[i] = 0;
    need.need[TR_BUDGET_TENSOR] = max_tensor;
    need.need[TR_BUDGET_OUTPUT] = prog->output_cells;
    need.need[TR_BUDGET_BYTES] = tr_tensor_program_device_bytes(prog);
    tr_math_budget dflt;
    if (!budget) { tr_budget_default(&dflt); budget = &dflt; }
    int rc = tr_budget_check(budget, &need, refusal);
    if (rc) return rc;
    tr_math_i128 *dev[TR_TENSOR_NET_MAX + TR_TENSOR_MAX_STEPS];
    for (int32_t i = 0; i < ni + ns; i++) dev[i] = NULL;
    for (int32_t i = 0; i < ni && !rc; i++) {
        rc = tr_cuda_alloc((void **)&dev[i], (size_t)cells[i] * 16);
        if (!rc) rc = tr_cuda_copy_in(dev[i], inputs[i].value, (size_t)cells[i] * 16);
    }
    if (rec) { memset(rec, 0, sizeof *rec); rec->candidate = -1; rec->kernel = TR_TENSOR_KERNEL_DENSE; rec->device = TR_REG_ON_CUDA; rec->nsteps = ns; }
    uint64_t live = 0, peak = 0;
    for (int32_t i = 0; i < ni; i++) live += cells[i] * 16;
    peak = live;
    for (int32_t s = 0; s < ns && !rc; s++) {
        const tr_tensor_dense_step *st = &prog->step[s];
        int32_t slot = ni + s;
        rc = tr_cuda_alloc((void **)&dev[slot], (size_t)cells[slot] * 16);
        if (!rc && cells[slot]) {
            TR_LAUNCH(tr_tensor_contract_kernel, cells[slot], *st, dev[st->a], st->b >= 0 ? dev[st->b] : (const tr_math_i128 *)NULL,
                      dev[slot]);
            rc = tr_cuda_sync();
        }
        live += cells[slot] * 16;
        if (live > peak) peak = live;
        if (rec) { rec->nnz[s] = cells[slot]; rec->bytes[s] = cells[slot] * 16; rec->pairs[s] = st->cells * st->terms; }
        if (st->a >= ni && st->free_a) { tr_cuda_free(dev[st->a]); dev[st->a] = NULL; live -= cells[st->a] * 16; }
        if (st->b >= ni && st->free_b) { tr_cuda_free(dev[st->b]); dev[st->b] = NULL; live -= cells[st->b] * 16; }
    }
    if (!rc && prog->output_cells) rc = tr_cuda_copy_out(out, dev[ni + ns - 1], (size_t)prog->output_cells * 16);
    for (int32_t i = 0; i < ni + ns; i++) tr_cuda_free(dev[i]);
    if (rec) { rec->peak_bytes = peak; rec->output_nnz = prog->output_cells; rec->output_bytes = prog->output_cells * 16; }
    return rc;
}
#ifdef __cplusplus
}
#endif
