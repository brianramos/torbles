/* MULTI-002 on a CUDA device: batched region bounds and localized rollups, with the semantics of
 * tr_refine_eval_bounds / tr_refine_update_rollup (math_refine.h) and the same results bit for bit.
 * SPDX-License-Identifier: Apache-2.0
 *
 * tr_mgpu_refine_bounds: the host gathers the cells and the UNC-001 inputs of the wave's regions (n of them, in the
 * order given) into compact arrays; one work item per region evaluates tr_refine_region_bound on them (exact 256-bit
 * integers; the UNC-001 abstract evaluation of each class's row expression in the item's own slot range of a device
 * scratch, chunks of TR_REFINE_GPU_CHUNK regions per launch) and writes its own output slot. tr_mgpu_refine_rollup:
 * one launch per level >= 1 (each level reads the one below, finished by the previous launch), one work item per
 * affected region summing its affected children's exact changes (tr_refine_rollup_region: 128-bit counts, 192-bit
 * sums, present values, the inserted values' extremes and gcd) into its own slots. No atomics are needed, no barrier,
 * no shared memory: every result is a function of its inputs only, whatever the order of the threads. The host
 * validates the plan, the view and every index the kernels read, and checks the budget before copying. */
#include "math_cuda.h"
#include "math_refine.h"

TR_MGPU_ABI(tr_refine_atom, TR_REFINE_ABI);          /* structs passed to the CUDA library (math_handles.h) */
TR_MGPU_ABI(tr_refine_class, TR_REFINE_ABI);
TR_MGPU_ABI(tr_refine_plan, TR_REFINE_ABI);
TR_MGPU_ABI(tr_refine_level_view, TR_REFINE_ABI);
TR_MGPU_ABI(tr_refine_bound, TR_REFINE_ABI);
TR_MGPU_ABI(tr_refine_ins, TR_REFINE_ABI);
TR_MGPU_ABI(tr_refine_update_view, TR_REFINE_ABI);

TR_KERNEL void tr_refine_bounds_kernel(tr_refine_plan pl, tr_refine_level_view lv, uint32_t level, int32_t exact,
                                       const uint32_t *groups, uint64_t first, uint64_t count, tr_unc_value *scratch,
                                       tr_refine_bound *out) {
    TR_GRID_LOOP(i, count) {
        uint64_t r = first + i;
        tr_refine_region_bound(&pl, &lv, level, (uint32_t)r, exact, &scratch[i * ((uint64_t)pl.nlive + 1)], &out[r]);
        out[r].group = groups[r];
    }
}

TR_KERNEL void tr_refine_rollup_kernel(tr_refine_update_view u, int32_t l) {
    TR_GRID_LOOP(i, u.naff[l]) tr_refine_rollup_region(&u, l, i);
}

/* the plan is what tr_refine_begin compiles: every index the kernel follows is in range, every program valid */
static int tr_refine_plan_ok(const tr_refine_plan *pl, const tr_refine_level_view *lv) {
    if (pl->naggs < 1 || pl->naggs > 2 || pl->natoms < 1 || pl->natoms > TR_REFINE_MAX_PLAN_ATOMS || pl->nclass < 1 ||
        pl->nclass > TR_REFINE_MAX_CLASSES || pl->nnodes < 1 || pl->nnodes > TR_REFINE_MAX_NODES || pl->nlive < 0 ||
        !pl->atoms || !pl->classes || !pl->nodes || pl->nterms != lv->nterms || pl->ncols != lv->ncols ||
        lv->ncols < 1 || lv->ncols > TR_UNC_MAX_COLS || lv->nterms < 0) return 0;
    for (int32_t i = 0; i < pl->natoms; i++) {
        const tr_refine_atom *a = &pl->atoms[i];
        if (a->kind < TR_REFINE_SUM || a->kind > TR_REFINE_ROWS || (a->kind != TR_REFINE_ROWS && (a->term < 0 || a->term >= lv->nterms)))
            return 0;
    }
    for (int32_t a = 0; a < pl->naggs; a++) {
        if (pl->first[a] < 0 || pl->nclasses[a] < 1 || pl->first[a] + pl->nclasses[a] > pl->nclass) return 0;
        if (pl->merged[a] != -1 && (pl->merged[a] < pl->first[a] || pl->merged[a] >= pl->first[a] + pl->nclasses[a])) return 0;
    }
    for (int32_t c = 0; c < pl->nclass; c++) {
        const tr_refine_class *cl = &pl->classes[c];
        if (cl->first_atom < 0 || cl->natoms < 1 || cl->first_atom + cl->natoms > pl->natoms || cl->first_node < 0 ||
            cl->nnodes < 1 || cl->first_node + cl->nnodes > pl->nnodes || cl->count_term < -1 || cl->count_term >= lv->nterms ||
            (cl->merged != 0 && cl->merged != 1) || cl->nlive < 0 || cl->nlive > pl->nlive) return 0;
        tr_unc_program p;
        memset(&p, 0, sizeof p);
        p.nnodes = cl->nnodes; p.ncols = lv->ncols; p.carrier = TR_UNC_EXACT; p.precision = 0; p.rank = TR_UNC_MAX_TERMS;
        p.domains = TR_UNC_DOM_ALL; p.root = cl->root; p.budget = tr_unc_qv(0, 0); p.nodes = pl->nodes + cl->first_node;
        tr_unc_plan plan;
        if (tr_unc_check(&p, 0, &plan, NULL) || plan.nlive > pl->nlive || plan.kind[p.root] != TR_UNC_NUMERIC) return 0;
    }
    return 1;
}

#ifdef __cplusplus
extern "C" {
#endif
int tr_mgpu_refine_bounds(const tr_refine_level_view *lv, const tr_refine_plan *plan, int32_t level, int32_t exact,
                          const uint32_t *groups, uint64_t n, tr_refine_bound *out, const tr_math_budget *budget,
                          tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!lv || !plan || (n && (!groups || !out)) || level < 0 || level >= TR_MULTI_MAX_LEVELS || (exact != 0 && exact != 1) ||
        (lv->ngroups && (!lv->rows || !lv->inputs || (lv->nterms && (!lv->count || !lv->sum)))) || !tr_refine_plan_ok(plan, lv))
        return TR_MATH_INVALID;
    for (uint64_t i = 0; i < n; i++) if (groups[i] >= lv->ngroups) return TR_MATH_INVALID;
    tr_math_budget dflt;
    if (!budget) { tr_budget_default(&dflt); budget = &dflt; }
    tr_math_need need;
    for (int i = 0; i < TR_BUDGET_RESOURCES; i++) need.need[i] = 0;
    need.need[TR_BUDGET_OUTPUT] = n;
    need.need[TR_BUDGET_BYTES] = tr_refine_bounds_device_bytes(lv, plan, n);
    int rc = tr_budget_check(budget, &need, refusal);
    if (rc || !n) return rc;
    const uint64_t m = (uint64_t)lv->nterms, c = (uint64_t)lv->ncols;
    const uint64_t chunk = n < TR_REFINE_GPU_CHUNK ? n : TR_REFINE_GPU_CHUNK, slots = (uint64_t)plan->nlive + 1;
    /* the regions' cells and inputs, gathered on the host in the order of groups[] */
    uint64_t *hrows = (uint64_t *)malloc((size_t)n * 8), *hcount = (uint64_t *)malloc((size_t)(n * m + 1) * 8);
    tr_math_i128 *hsum = (tr_math_i128 *)malloc((size_t)(n * m + 1) * sizeof(tr_math_i128));
    tr_unc_input *hin = (tr_unc_input *)malloc((size_t)(n * c) * sizeof(tr_unc_input));
    if (!hrows || !hcount || !hsum || !hin) { free(hrows); free(hcount); free(hsum); free(hin); return TR_MATH_MEMORY; }
    for (uint64_t i = 0; i < n; i++) {
        uint64_t g = groups[i];
        hrows[i] = lv->rows[g];
        for (uint64_t j = 0; j < m; j++) { hcount[i * m + j] = lv->count[g * m + j]; hsum[i * m + j] = lv->sum[g * m + j]; }
        for (uint64_t j = 0; j < c; j++) hin[i * c + j] = lv->inputs[g * c + j];
    }
    void *dmem[12];
    int nmem = 0;
#define TR_REFINE_DEV(ptr, bytes) do { if (!rc) { rc = tr_cuda_alloc((void **)&(ptr), (size_t)(bytes)); if (!rc) dmem[nmem++] = (void *)(ptr); } } while (0)
#define TR_REFINE_IN(dst, src, bytes) do { TR_REFINE_DEV(dst, bytes); if (!rc && (uint64_t)(bytes) != 0) rc = tr_cuda_copy_in(dst, src, (size_t)(bytes)); } while (0)
    uint64_t *drows = NULL, *dcount = NULL;
    tr_math_i128 *dsum = NULL;
    tr_unc_input *din = NULL;
    tr_refine_atom *datoms = NULL;
    tr_refine_class *dclasses = NULL;
    tr_unc_node *dnodes = NULL;
    uint32_t *dgroups = NULL;
    tr_refine_bound *dout = NULL;
    tr_unc_value *dscratch = NULL;
    TR_REFINE_IN(drows, hrows, n * 8);
    TR_REFINE_IN(dcount, hcount, n * m * 8);
    TR_REFINE_IN(dsum, hsum, n * m * sizeof(tr_math_i128));
    TR_REFINE_IN(din, hin, n * c * sizeof(tr_unc_input));
    TR_REFINE_IN(datoms, plan->atoms, (uint64_t)plan->natoms * sizeof(tr_refine_atom));
    TR_REFINE_IN(dclasses, plan->classes, (uint64_t)plan->nclass * sizeof(tr_refine_class));
    TR_REFINE_IN(dnodes, plan->nodes, (uint64_t)plan->nnodes * sizeof(tr_unc_node));
    TR_REFINE_IN(dgroups, groups, n * 4);
    TR_REFINE_DEV(dout, n * sizeof(tr_refine_bound));
    TR_REFINE_DEV(dscratch, chunk * slots * sizeof(tr_unc_value));
#undef TR_REFINE_IN
#undef TR_REFINE_DEV
    free(hrows); free(hcount); free(hsum); free(hin);
    tr_refine_level_view dlv = *lv;
    dlv.ngroups = n; dlv.rows = drows; dlv.count = dcount; dlv.sum = dsum; dlv.inputs = din;
    tr_refine_plan dpl = *plan;
    dpl.atoms = datoms; dpl.classes = dclasses; dpl.nodes = dnodes;
    for (uint64_t first = 0; !rc && first < n; first += chunk) {
        uint64_t count = n - first < chunk ? n - first : chunk;
        TR_LAUNCH(tr_refine_bounds_kernel, count, dpl, dlv, (uint32_t)level, exact, dgroups, first, count, dscratch, dout);
        rc = tr_cuda_sync();
    }
    if (!rc) rc = tr_cuda_copy_out(out, dout, (size_t)n * sizeof(tr_refine_bound));
    for (int i = 0; i < nmem; i++) tr_cuda_free(dmem[i]);
    return rc;
}

int tr_mgpu_refine_rollup(tr_refine_update_view *view, uint64_t nchild, const tr_math_budget *budget,
                          tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!view || view->levels < 2 || view->levels > TR_MULTI_MAX_LEVELS || view->nterms < 0 || view->ncols < 1 ||
        view->ncols > TR_SCOPE_MAX_COLUMNS || view->reserved) return TR_MATH_INVALID;
    const int32_t L = view->levels;
    uint64_t total = 0;
    for (int32_t l = 0; l < L; l++) {                 /* the flat layout: level l's regions at off[l] */
        if (view->off[l] != total) return TR_MATH_INVALID;
        total += view->naff[l];
    }
    if (total && (!view->ids || !view->child_off || !view->drows || !view->dpresent || !view->ins ||
                  (view->nterms && (!view->dcount || !view->dsum)))) return TR_MATH_INVALID;
    if (nchild && !view->child_idx) return TR_MATH_INVALID;
    uint64_t prev = 0;
    for (int32_t l = 1; l < L; l++)                   /* each level's CSR: increasing, inside child_idx, children below */
        for (uint64_t i = 0; i <= view->naff[l]; i++) {
            uint64_t o = view->child_off[view->off[l] + (uint64_t)l + i];
            if (o < prev || o > nchild || (i == 0 && l > 1 && o != prev)) return TR_MATH_INVALID;
            if (i > 0) for (uint64_t k = prev; k < o; k++) if (view->child_idx[k] >= view->naff[l - 1]) return TR_MATH_INVALID;
            prev = o;
        }
    if (prev != nchild) return TR_MATH_INVALID;
    tr_math_budget dflt;
    if (!budget) { tr_budget_default(&dflt); budget = &dflt; }
    tr_math_need need;
    for (int i = 0; i < TR_BUDGET_RESOURCES; i++) need.need[i] = 0;
    need.need[TR_BUDGET_STATES] = tr_budget_mul(total, (uint64_t)view->nterms + (uint64_t)view->ncols);
    need.need[TR_BUDGET_BYTES] = tr_refine_rollup_device_bytes(view, nchild);
    int rc = tr_budget_check(budget, &need, refusal);
    if (rc || !total) return rc;
    const uint64_t m = (uint64_t)view->nterms, c = (uint64_t)view->ncols, noff = total + (uint64_t)L;
    void *dmem[8];
    int nmem = 0;
#define TR_REFINE_DEV(ptr, bytes) do { if (!rc) { rc = tr_cuda_alloc((void **)&(ptr), (size_t)(bytes)); if (!rc) dmem[nmem++] = (void *)(ptr); } } while (0)
#define TR_REFINE_IN(dst, src, bytes) do { TR_REFINE_DEV(dst, bytes); if (!rc && (uint64_t)(bytes) != 0) rc = tr_cuda_copy_in(dst, src, (size_t)(bytes)); } while (0)
    uint64_t *doff = NULL, *dsum = NULL;
    uint32_t *didx = NULL;
    tr_math_i128 *drows = NULL, *dcount = NULL, *dpresent = NULL;
    tr_refine_ins *dins = NULL;
    TR_REFINE_IN(doff, view->child_off, noff * 8);
    TR_REFINE_IN(didx, view->child_idx, nchild * 4);
    TR_REFINE_IN(drows, view->drows, total * sizeof(tr_math_i128));
    TR_REFINE_IN(dcount, view->dcount, total * m * sizeof(tr_math_i128));
    TR_REFINE_IN(dsum, view->dsum, total * m * 24);
    TR_REFINE_IN(dpresent, view->dpresent, total * c * sizeof(tr_math_i128));
    TR_REFINE_IN(dins, view->ins, total * c * sizeof(tr_refine_ins));
#undef TR_REFINE_IN
#undef TR_REFINE_DEV
    tr_refine_update_view d = *view;
    d.ids = NULL; d.child_off = doff; d.child_idx = didx; d.drows = drows; d.dcount = dcount; d.dsum = dsum;
    d.dpresent = dpresent; d.ins = dins;
    for (int32_t l = 1; !rc && l < L; l++) {
        if (!view->naff[l]) continue;
        TR_LAUNCH(tr_refine_rollup_kernel, view->naff[l], d, l);
        rc = tr_cuda_sync();
    }
    const uint64_t b0 = view->naff[0], rest = total - b0;   /* the coarse levels' slots back into the view */
    if (!rc) rc = tr_cuda_copy_out(view->drows + b0, drows + b0, (size_t)rest * sizeof(tr_math_i128));
    if (!rc && m) rc = tr_cuda_copy_out(view->dcount + b0 * m, dcount + b0 * m, (size_t)(rest * m) * sizeof(tr_math_i128));
    if (!rc && m) rc = tr_cuda_copy_out(view->dsum + b0 * m * 3, dsum + b0 * m * 3, (size_t)(rest * m) * 24);
    if (!rc) rc = tr_cuda_copy_out(view->dpresent + b0 * c, dpresent + b0 * c, (size_t)(rest * c) * sizeof(tr_math_i128));
    if (!rc) rc = tr_cuda_copy_out(view->ins + b0 * c, dins + b0 * c, (size_t)(rest * c) * sizeof(tr_refine_ins));
    for (int i = 0; i < nmem; i++) tr_cuda_free(dmem[i]);
    return rc;
}
#ifdef __cplusplus
}
#endif
