/* JET-001 / JET-002 on a CUDA device: the jets of one immutable circuit plan at every selected row of a scope, summed
 * exactly per group, with exactly the states of tr_jet_agg_build (math_jet.h).
 * SPDX-License-Identifier: Apache-2.0
 *
 * Work split: W work items (tr_jet_device_items); the rows are processed in launches of min(2^20, 64 W) physical rows
 * (so a launch stays short, e.g. under a display watchdog, also when a large plan leaves few work items), and in the
 * launch over [r0, r1) item w takes the rows r0 + w, r0 + w + W, ... (consecutive threads read consecutive rows). Each
 * row's jet is evaluated by tr_jet_eval, the inline code the CPU runs, in fixed-denominator int64 arithmetic with
 * checked 128-bit intermediates; the workspaces of the work items are interleaved in one device
 * array (word i of item w at i * W + w, so a warp's accesses are contiguous). An item adds its rows' numerators into
 * local exact 128-bit sums while the group id stays the same and flushes them with tr_atomic_add_wide (and the row /
 * valid counts with 64-bit atomics) when it changes and at its end. Integer additions commute, so the states do not
 * depend on thread order; each row's numerators are int64, so every sum over fewer than 2^63 rows fits 128 bits (no
 * proof beyond that is needed). A refused row (TR_MATH_OVERFLOW on the CPU) sets the flag word and the host refuses the
 * whole request, as the CPU does. No shared memory, no barriers. */
#include "math_cuda.h"
#include "math_jet.h"
#include "math_handles.h"

TR_MGPU_ABI(tr_jet_view, 1);            /* structs passed to the CUDA library (math_handles.h) */
TR_MGPU_ABI(tr_jet_step, 1);
TR_MGPU_ABI(tr_jet_states, 1);

/* the local exact sums of one item: word pair (2k, 2k + 1) of its accumulator block, interleaved */
TR_DEVICE void tr_jet_flush(const tr_jet_view *p, int64_t *acc, uint64_t stride, uint32_t g, uint64_t lrows,
                            uint64_t lcount, uint64_t *rows, uint64_t *count, int64_t *sum) {
    const uint64_t S = p->support, Sout = p->out_jet ? p->upto[p->out_pdeg] : 1;   /* zero past the output's degree */
    tr_atomic_add_u64(&rows[g], lrows);
    if (lcount) tr_atomic_add_u64(&count[g], lcount);
    for (uint64_t k = 0; k < Sout; k++) {
        int64_t *lo = &acc[(2 * k) * stride], *hi = &acc[(2 * k + 1) * stride];
        tr_math_i128 w;
        w.lo = *lo; w.hi = *hi;
        __int128 x = tr_i128_get(w);
        if (x) {
            uint64_t i = (uint64_t)g * S + k;
            tr_atomic_add_wide(&sum[2 * i], &sum[2 * i + 1], x);
        }
        *lo = 0; *hi = 0;
    }
}

#define TR_JET_LAUNCH_ROWS (UINT64_C(1) << 20)   /* physical rows per launch, at most */
#define TR_JET_ITEM_ROWS 64                        /* and at most this many per work item */

TR_KERNEL void tr_jet_agg_kernel(tr_jet_view p, tr_scope_view v, uint64_t r0, uint64_t r1, uint64_t items,
                                 int64_t *work, uint64_t *rows, uint64_t *count, int64_t *sum, uint64_t *flags) {
    TR_GRID_LOOP(w, items) {
        const uint64_t S = p.support, stride = items, Sout = p.out_jet ? p.upto[p.out_pdeg] : 1;
        int64_t *ws = work + w;
        int64_t *out = ws + p.workspace * stride;            /* output coefficient k at out[k * stride] */
        int64_t *acc = ws + (p.workspace + S) * stride;      /* local sum k at acc[2k * stride], acc[(2k + 1) * stride] */
        int64_t data[TR_SCOPE_MAX_COLUMNS];
        for (uint64_t k = 0; k < 2 * Sout; k++) acc[k * stride] = 0;
        uint32_t cur = 0;
        uint64_t lrows = 0, lcount = 0;
        int bad = 0;
        for (uint64_t r = r0 + w; r < r1; r += items) {
            if (!tr_scope_is_selected(&v, r)) continue;
            uint32_t g = tr_scope_group(&v, r);
            if (lrows && g != cur) { tr_jet_flush(&p, acc, stride, cur, lrows, lcount, rows, count, sum); lrows = lcount = 0; }
            cur = g; lrows++;
            int ok = 1;
            for (int32_t c = 0; c < v.ncols && ok; c++) {
                if (!((p.columns >> c) & 1)) continue;
                if (!tr_scope_present(&v, c, r)) ok = 0;
                else data[c] = tr_scope_value(&v, c, r);
            }
            if (!ok) continue;                               /* f is NULL on this row */
            lcount++;
            if (!tr_jet_eval(&p, data, ws, stride, out, stride)) { bad = 1; break; }
            for (uint64_t k = 0; k < Sout; k++) {
                int64_t x = out[k * stride];
                if (!x) continue;
                int64_t *lo = &acc[(2 * k) * stride], *hi = &acc[(2 * k + 1) * stride];
                tr_math_i128 s;
                s.lo = *lo; s.hi = *hi;
                s = tr_i128_set(tr_i128_get(s) + x);         /* a local sum of int64 values: |sum| < 2^126 */
                *lo = s.lo; *hi = s.hi;
            }
        }
        if (bad) { tr_atomic_or_u64(flags, 1); continue; }
        if (lrows) tr_jet_flush(&p, acc, stride, cur, lrows, lcount, rows, count, sum);
    }
}

#ifdef __cplusplus
extern "C" {
#endif
int tr_mgpu_jet_aggregate(const tr_jet_view *plan, const tr_scope_view *view, const tr_math_budget *budget,
                          tr_jet_states *out, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!plan || !view || !out || !plan->steps || !plan->degree || !plan->mask || !plan->conv_start ||
        !plan->conv_pairs || !plan->upto || (plan->ncols && !plan->scales) || view->ngroups < 1 || out->ngroups != view->ngroups ||
        out->support != plan->support || plan->support < 1 || !out->rows || !out->count || !out->sum ||
        (view->rows && !view->selection) || (view->rows && view->ngroups > 1 && !view->groups) ||
        view->ncols < 0 || view->ncols > TR_SCOPE_MAX_COLUMNS) return -1;                      /* TR_MATH_INVALID */
    if (view->ncols != plan->ncols) return TR_MATH_DOMAIN_ERROR;
    for (int32_t c = 0; c < view->ncols; c++) if (view->col[c].scale != plan->scales[c]) return TR_MATH_DOMAIN_ERROR;
    if (view->selected >= TR_JET_MAX_ROWS) return TR_MATH_UNSUPPORTED;
    tr_math_need need;
    tr_math_budget dflt;
    if (!budget) { tr_budget_default(&dflt); budget = &dflt; }
    tr_jet_device_need(plan, view, &need);
    int rc = tr_budget_check(budget, &need, refusal);
    if (rc) return rc;
    const uint64_t n = view->rows, G = view->ngroups, S = plan->support, GS = G * S;
    memset(out->rows, 0, (size_t)G * 8);
    memset(out->count, 0, (size_t)G * 8);
    memset(out->sum, 0, (size_t)GS * 16);
    if (!view->selected) return 0;                           /* nothing selected: every state is zero */
    const uint64_t items = tr_jet_device_items(plan, n), words = tr_jet_item_words(plan);
    tr_scope_view d = *view;
    tr_jet_view dp = *plan;
    void *dmem[2 * TR_SCOPE_MAX_COLUMNS + 16];
    int nmem = 0;
    uint64_t *dsel = NULL, *drows = NULL, *dcount = NULL, *dflags = NULL, flags = 0;
    uint32_t *dgroups = NULL, *dstart = NULL, *dpairs = NULL, *dupto = NULL;
    int64_t *dsum = NULL, *dwork = NULL, *dleaf = NULL;
    uint8_t *ddeg = NULL;
    uint64_t *dmask = NULL;
    tr_jet_step *dsteps = NULL;
#define TR_JET_DEV(ptr, bytes) do { if (!rc) { rc = tr_cuda_alloc((void **)&(ptr), (size_t)(bytes)); if (!rc) dmem[nmem++] = (void *)(ptr); } } while (0)
#define TR_JET_IN(dst, src, bytes) do { if (!rc) rc = tr_cuda_copy_in((dst), (src), (size_t)(bytes)); } while (0)
    TR_JET_DEV(dsel, ((n + 63) / 64) * 8);
    TR_JET_IN(dsel, view->selection, ((n + 63) / 64) * 8);
    d.selection = dsel;
    if (view->groups) {
        TR_JET_DEV(dgroups, n * 4);
        TR_JET_IN(dgroups, view->groups, n * 4);
        d.groups = dgroups;
    }
    for (int32_t c = 0; c < view->ncols; c++) {
        tr_scope_view_column *k = &d.col[c];
        if (!((plan->columns >> c) & 1)) { k->codes = NULL; k->values = NULL; k->valid = NULL; continue; }   /* never read */
        if (view->col[c].values) {
            int64_t *q = NULL;
            TR_JET_DEV(q, n * 8);
            TR_JET_IN(q, view->col[c].values, n * 8);
            k->values = q;
        } else {
            uint32_t *q = NULL;
            TR_JET_DEV(q, n * 4);
            TR_JET_IN(q, view->col[c].codes, n * 4);
            k->codes = q;
        }
        if (view->col[c].valid) {
            uint8_t *q = NULL;
            TR_JET_DEV(q, n);
            TR_JET_IN(q, view->col[c].valid, n);
            k->valid = q;
        }
    }
    const uint64_t leaf = (uint64_t)(plan->nparams > 0 ? plan->nparams : 0) * ((uint64_t)plan->directions + 1);
    TR_JET_DEV(dsteps, (uint64_t)plan->nsteps * sizeof(tr_jet_step));
    TR_JET_IN(dsteps, plan->steps, (uint64_t)plan->nsteps * sizeof(tr_jet_step));
    TR_JET_DEV(dleaf, leaf * 8);
    if (leaf) TR_JET_IN(dleaf, plan->leaf, leaf * 8);
    TR_JET_DEV(ddeg, S);
    TR_JET_IN(ddeg, plan->degree, S);
    TR_JET_DEV(dmask, S * 8);
    TR_JET_IN(dmask, plan->mask, S * 8);
    TR_JET_DEV(dstart, (S + 1) * 4);
    TR_JET_IN(dstart, plan->conv_start, (S + 1) * 4);
    TR_JET_DEV(dpairs, plan->pairs * 4);
    TR_JET_IN(dpairs, plan->conv_pairs, plan->pairs * 4);
    TR_JET_DEV(dupto, ((uint64_t)plan->order + 1) * 4);
    TR_JET_IN(dupto, plan->upto, ((uint64_t)plan->order + 1) * 4);
    dp.steps = dsteps; dp.leaf = dleaf; dp.degree = ddeg; dp.mask = dmask; dp.conv_start = dstart;
    dp.conv_pairs = dpairs; dp.upto = dupto; dp.scales = NULL;
    TR_JET_DEV(dwork, items * words * 8);
    TR_JET_DEV(drows, G * 8);
    TR_JET_DEV(dcount, G * 8);
    TR_JET_DEV(dsum, GS * 16);
    TR_JET_DEV(dflags, 8);
#undef TR_JET_DEV
#undef TR_JET_IN
    const uint64_t chunk = items * TR_JET_ITEM_ROWS < TR_JET_LAUNCH_ROWS ? items * TR_JET_ITEM_ROWS : TR_JET_LAUNCH_ROWS;
    for (uint64_t r0 = 0; !rc && r0 < n; r0 += chunk) {
        uint64_t r1 = n - r0 < chunk ? n : r0 + chunk;
        TR_LAUNCH(tr_jet_agg_kernel, items, dp, d, r0, r1, items, dwork, drows, dcount, dsum, dflags);
        rc = tr_cuda_sync();
        if (!rc) rc = tr_cuda_copy_out(&flags, dflags, 8);
        if (!rc && flags) rc = TR_MATH_OVERFLOW;                 /* a refused row: stop at once */
    }
    if (!rc) rc = tr_cuda_copy_out(out->rows, drows, (size_t)G * 8);
    if (!rc) rc = tr_cuda_copy_out(out->count, dcount, (size_t)G * 8);
    if (!rc) rc = tr_cuda_copy_out(out->sum, dsum, (size_t)GS * 16);
    for (int i = 0; i < nmem; i++) tr_cuda_free(dmem[i]);
    return rc;
}
#ifdef __cplusplus
}
#endif
