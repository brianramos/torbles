/* LATTICE-001/002 on a CUDA device: the occupancy histogram scan (tr_lattice_hist_build) and the butterfly passes of
 * the fast zeta / Moebius transforms (tr_lattice_transform), with exactly the CPU's results (math_lattice.h).
 * SPDX-License-Identifier: Apache-2.0
 *
 * Histogram: work items are word slices (slice s takes the words s, s + S, s + 2S, ...). A work item owns 16 bytes of
 * node words per program node, evaluates the axes of each of its words like the CPU (tr_lattice_word: every program
 * once, bitwise on a value and a valid word), computes the cell of every point (tr_lattice_cell_of) and adds one to
 * it with an integer atomic; with at most TR_LATTICE_LOCAL_CELLS cells it counts in a per-thread array instead and
 * adds each nonzero count once at the end, like its per-axis TRUE / FALSE / UNKNOWN counts.
 * Transforms: one launch per pass j, one work item per pair (A, A + 2^j) (the pairs of a pass are disjoint), checked
 * 128-bit butterflies (tr_lattice_butterfly) that raise an overflow flag (atomic OR) instead of wrapping; with the
 * int64 carrier a last launch flags every output that does not fit int64. Integer atomics commute and the passes are
 * ordered by their launches, so the results do not depend on thread order. */
#include "math_cuda.h"
#include "math_lattice.h"
#include "math_handles.h"

TR_MGPU_ABI(tr_lattice_axis, TR_LATTICE_ABI);      /* structs passed to the CUDA library (math_handles.h) */

TR_KERNEL void tr_lattice_hist_kernel(tr_bool_domain d, const tr_lattice_axis *axes, int32_t k,
                                      const tr_bool_program_view *progs, int32_t nprogs, int32_t nulls,
                                      uint64_t items, uint64_t nodes, uint64_t *scratch, uint64_t *cells,
                                      uint64_t ncells, uint64_t *axis) {
    TR_GRID_LOOP(item, items) {
        uint64_t *val = scratch + item * 2 * nodes;
        uint64_t value[TR_LATTICE_MAX_AXES + 1], valid[TR_LATTICE_MAX_AXES + 1];
        uint64_t local[TR_LATTICE_LOCAL_CELLS], ax[3 * TR_LATTICE_MAX_AXES + 1];
        const int small = ncells <= TR_LATTICE_LOCAL_CELLS;
        for (int c = 0; c < TR_LATTICE_LOCAL_CELLS; c++) local[c] = 0;
        for (int32_t i = 0; i < 3 * k; i++) ax[i] = 0;
        for (uint64_t w = item; w < d.nw; w += items) {
            uint64_t sel = tr_bool_selection_word(&d, w);
            if (!sel) continue;
            tr_lattice_word(&d, axes, k, progs, nprogs, w, sel, val, value, valid);
            uint64_t known = sel;
            for (int32_t i = 0; i < k; i++) {
                ax[3 * i] += (uint64_t)tr_bool_popc(value[i]);
                ax[3 * i + 1] += (uint64_t)tr_bool_popc(valid[i] & ~value[i]);
                ax[3 * i + 2] += (uint64_t)tr_bool_popc(sel & ~valid[i]);
                known &= valid[i];
            }
            uint64_t bits = nulls == TR_LATTICE_APART ? known : sel;
            while (bits) {
                int b = tr_bool_ctz(bits);
                bits &= bits - 1;
                uint64_t cell = tr_lattice_cell_of(value, valid, k, nulls, b);
                if (small) local[cell]++;
                else tr_atomic_add_u64(&cells[cell], 1);
            }
        }
        if (small)
            for (uint64_t c = 0; c < ncells; c++) if (local[c]) tr_atomic_add_u64(&cells[c], local[c]);
        for (int32_t i = 0; i < 3 * k; i++) if (ax[i]) tr_atomic_add_u64(&axis[i], ax[i]);
    }
}

TR_KERNEL void tr_lattice_pass_kernel(tr_math_i128 *v, int32_t op, int32_t j, uint64_t pairs, uint64_t *overflow) {
    TR_GRID_LOOP(p, pairs) {
        uint64_t lo = tr_lattice_pair_lo(p, j);
        if (!tr_lattice_butterfly(op, v, lo, lo | ((uint64_t)1 << j))) tr_atomic_or_u64(overflow, 1);
    }
}

TR_KERNEL void tr_lattice_fits_kernel(const tr_math_i128 *v, uint64_t n, uint64_t *overflow) {
    TR_GRID_LOOP(i, n) {
        if (!tr_i128_fits_i64(v[i])) tr_atomic_or_u64(overflow, 2);
    }
}

#ifdef __cplusplus
extern "C" {
#endif
int tr_mgpu_lattice_hist(const tr_bool_domain *dom, const tr_lattice_axis *axes, int32_t k,
                         const tr_bool_program_view *progs, int32_t nprogs, int32_t nulls, const tr_math_budget *budget,
                         uint64_t *cells, uint64_t *axis, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    int rc = tr_lattice_axes_check(dom, axes, k, progs, nprogs, nulls);
    if (rc) return rc;
    tr_math_budget dflt;
    if (!budget) { tr_budget_default(&dflt); budget = &dflt; }
    tr_math_need need;
    tr_lattice_hist_base_need(dom, k, progs, nprogs, nulls, &need);
    need.need[TR_BUDGET_BYTES] = tr_lattice_hist_device_bytes(dom, k, progs, nprogs, nulls);
    rc = tr_budget_check(budget, &need, refusal);
    if (rc) return rc;
    rc = tr_lattice_hist_limits(k, nprogs);
    if (rc) return rc;
    const uint64_t ncells = tr_lattice_cells(k, nulls), nw = dom->nw;
    if (!cells || (k && !axis)) return -1;                                                  /* TR_MATH_INVALID */
    memset(cells, 0, (size_t)(ncells * 8));
    if (k) memset(axis, 0, (size_t)k * 24);
    if (!dom->selected) return 0;                             /* nothing counts: every count is zero */
    const uint64_t nodes = tr_lattice_prog_nodes(progs, nprogs), items = tr_lattice_slices(nw, nodes);
    tr_bool_domain d = *dom;
    tr_bool_program_view hp[TR_LATTICE_MAX_AXES];
    void *mem[2 * TR_LATTICE_MAX_AXES + 16];
    int nmem = 0;
    uint64_t *dsel = NULL, *dvalue = NULL, *dvalid = NULL, *dcells = NULL, *daxis = NULL, *dscratch = NULL;
    tr_lattice_axis *daxes = NULL;
    tr_bool_program_view *dprogs = NULL;
#define TR_LATTICE_DEV(ptr, bytes) do { if (!rc) { rc = tr_cuda_alloc((void **)&(ptr), (size_t)(bytes)); if (!rc) mem[nmem++] = (void *)(ptr); } } while (0)
    if (dom->kind == TR_BOOL_ROWS) {
        uint64_t words = (uint64_t)dom->k * nw;
        TR_LATTICE_DEV(dsel, nw * 8);
        if (!rc) rc = tr_cuda_copy_in(dsel, dom->selection, (size_t)(nw * 8));
        TR_LATTICE_DEV(dvalue, (words ? words : 1) * 8);
        if (!rc && words) rc = tr_cuda_copy_in(dvalue, dom->value, (size_t)(words * 8));
        if (dom->valid) {
            TR_LATTICE_DEV(dvalid, (words ? words : 1) * 8);
            if (!rc && words) rc = tr_cuda_copy_in(dvalid, dom->valid, (size_t)(words * 8));
        }
        d.selection = dsel; d.value = dvalue; d.valid = dvalid;
    }
    for (int32_t j = 0; j < nprogs && !rc; j++) {
        tr_bool_node *dn = NULL;
        uint32_t *ds = NULL;
        hp[j] = progs[j];
        TR_LATTICE_DEV(dn, (uint64_t)progs[j].nnodes * sizeof(tr_bool_node));
        if (!rc) rc = tr_cuda_copy_in(dn, progs[j].nodes, (size_t)progs[j].nnodes * sizeof(tr_bool_node));
        TR_LATTICE_DEV(ds, (uint64_t)progs[j].nsched * 4);
        if (!rc) rc = tr_cuda_copy_in(ds, progs[j].schedule, (size_t)progs[j].nsched * 4);
        hp[j].nodes = dn; hp[j].schedule = ds; hp[j].deps = NULL;   /* the evaluators never read deps */
    }
    TR_LATTICE_DEV(dprogs, (uint64_t)(nprogs ? nprogs : 1) * sizeof(tr_bool_program_view));
    if (!rc && nprogs) rc = tr_cuda_copy_in(dprogs, hp, (size_t)nprogs * sizeof(tr_bool_program_view));
    TR_LATTICE_DEV(daxes, (uint64_t)(k ? k : 1) * sizeof(tr_lattice_axis));
    if (!rc && k) rc = tr_cuda_copy_in(daxes, axes, (size_t)k * sizeof(tr_lattice_axis));
    TR_LATTICE_DEV(dcells, ncells * 8);
    TR_LATTICE_DEV(daxis, (uint64_t)(k ? k : 1) * 24);
    TR_LATTICE_DEV(dscratch, items * 16 * (nodes ? nodes : 1));
#undef TR_LATTICE_DEV
    if (!rc) {
        TR_LAUNCH(tr_lattice_hist_kernel, items, d, daxes, k, dprogs, nprogs, nulls, items, nodes, dscratch, dcells,
                  ncells, daxis);
        rc = tr_cuda_sync();
    }
    if (!rc) rc = tr_cuda_copy_out(cells, dcells, (size_t)(ncells * 8));
    if (!rc && k) rc = tr_cuda_copy_out(axis, daxis, (size_t)k * 24);
    for (int i = 0; i < nmem; i++) tr_cuda_free(mem[i]);
    if (rc) { memset(cells, 0, (size_t)(ncells * 8)); if (k) memset(axis, 0, (size_t)k * 24); }
    return rc;
}

int tr_mgpu_lattice_transform(int32_t k, int32_t op, int32_t carrier, const tr_math_i128 *in, tr_math_i128 *out,
                              const tr_math_budget *budget, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    tr_math_need need;
    if (tr_lattice_transform_base_need(k, op, carrier, &need) || carrier == TR_LATTICE_BIGINT || !in || !out)
        return -1;                                                                          /* TR_MATH_INVALID */
    tr_math_budget dflt;
    if (!budget) { tr_budget_default(&dflt); budget = &dflt; }
    int rc = tr_budget_check(budget, &need, refusal);
    if (rc) return rc;
    if (k > TR_LATTICE_MAX_AXES) return -12;                                                /* TR_MATH_UNSUPPORTED */
    const uint64_t n = (uint64_t)1 << k;
    if (carrier == TR_LATTICE_I64)
        for (uint64_t i = 0; i < n; i++) if (!tr_i128_fits_i64(in[i])) return -3;          /* TR_MATH_OVERFLOW */
    tr_math_i128 *dv = NULL;
    uint64_t *dflag = NULL, flag = 0;
    rc = tr_cuda_alloc((void **)&dv, (size_t)(n * 16));
    if (!rc) rc = tr_cuda_alloc((void **)&dflag, 8);
    if (!rc) rc = tr_cuda_copy_in(dv, in, (size_t)(n * 16));
    for (int32_t j = 0; j < k && !rc; j++) {                  /* one launch per pass: the passes are ordered */
        TR_LAUNCH(tr_lattice_pass_kernel, n / 2, dv, op, j, n / 2, dflag);
        rc = tr_cuda_sync();
    }
    if (!rc && carrier == TR_LATTICE_I64) { TR_LAUNCH(tr_lattice_fits_kernel, n, dv, n, dflag); rc = tr_cuda_sync(); }
    if (!rc) rc = tr_cuda_copy_out(&flag, dflag, 8);
    if (!rc && flag) rc = -3;                                                               /* TR_MATH_OVERFLOW */
    if (!rc) rc = tr_cuda_copy_out(out, dv, (size_t)(n * 16));
    tr_cuda_free(dv);
    tr_cuda_free(dflag);
    return rc;
}
#ifdef __cplusplus
}
#endif
