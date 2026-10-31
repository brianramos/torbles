/* GEN-002 on a CUDA device: batched exact convolutions, the per-key conditional products of two keyed families and
 * walk tables of bounded series, with exactly the results of tr_conv_batch_run and tr_conv_walks_run (math_conv.h).
 * SPDX-License-Identifier: Apache-2.0
 *
 *   batch        one work item per output coefficient t: its pair by a binary search over the output offsets, then the
 *                exact 192-bit sum of tr_conv_coef192 written to its own three words
 *   conditional  the batch kernel over the keys of two families (pair g = key g of each): per-key products in parallel
 *                over keys; the host adopts the readback (tr_conv_conditional_adopt) as the CPU path adopts its own
 *   walks        one launch per length l: work item e computes row l's coefficient e from row l - 1 (192-bit sum, the
 *                declared bound checked; a violation writes 0 and lowers the first-violation word with an atomic
 *                minimum), then one launch for the totals (sum over the rows, 192-bit, bound checked)
 * Every output slot has one writer; the violation and the largest magnitude are atomic minima / maxima, so nothing
 * depends on the order of the threads. The host checks the budgets (output, tensor, iterations and the bytes of the
 * device copies) before allocating anything on the device. */
#include "math_cuda.h"
#include "math_conv.h"
#include "math_handles.h"

TR_MGPU_ABI(tr_conv_batch, TR_CONV_ABI);          /* structs passed to the CUDA library (math_handles.h) */
TR_MGPU_ABI(tr_conv_family_view, TR_CONV_ABI);
TR_MGPU_ABI(tr_conv_walks, TR_CONV_ABI);

TR_KERNEL void tr_conv_batch_kernel(tr_conv_batch b, int64_t *out) {
    TR_GRID_LOOP(t, b.total) tr_conv_batch_one(&b, t, (uint64_t *)out + 3 * t);
}

TR_KERNEL void tr_conv_walk_kernel(const int64_t *step, uint64_t n, const int64_t *prev, int64_t *cur, uint64_t row,
                                   int64_t bound, uint64_t *flags) {
    TR_GRID_LOOP(e, n) {
        uint64_t w[3];
        tr_conv_coef192(prev, n, step, n, e, w);
        int64_t v;
        if (tr_conv_bounded(w, bound, &v)) {
            cur[e] = v;
            tr_atomic_max_u64(&flags[1], tr_abs_u64(v));
        } else {
            cur[e] = 0;
            tr_atomic_min_u64(&flags[0], row * n + e);
        }
    }
}

TR_KERNEL void tr_conv_total_kernel(const int64_t *rows, uint64_t n, uint64_t lengths, int64_t bound, int64_t *total,
                                    uint64_t *flags) {
    TR_GRID_LOOP(e, n) {
        uint64_t w[3] = {0, 0, 0};
        for (uint64_t l = 0; l < lengths; l++) tr_wide_add128(w, (__int128)rows[l * n + e]);
        int64_t v;
        if (tr_conv_bounded(w, bound, &v)) {
            total[e] = v;
            tr_atomic_max_u64(&flags[1], tr_abs_u64(v));
        } else {
            total[e] = 0;
            tr_atomic_min_u64(&flags[0], lengths * n + e);
        }
    }
}

/* tr_budget_check, refusing a size past 64 bits even under an unlimited budget */
static int tr_conv_gpu_check(const tr_math_budget *b, const tr_math_need *n, tr_math_refusal *r) {
    int rc = tr_budget_check(b, n, r);
    if (rc) return rc;
    for (int i = 0; i < TR_BUDGET_RESOURCES; i++)
        if (n->need[i] == TR_BUDGET_UNKNOWN) {
            if (r) { r->resource = i; r->refused = 1; r->need = n->need[i]; r->limit = b->limit[i]; }
            return TR_MATH_BUDGET_CODE;
        }
    return 0;
}

/* a checked batch on the device: inputs, offsets and outputs copied; out[3 * total] */
static int tr_conv_gpu_batch(const tr_conv_batch *b, int64_t *out, const tr_math_budget *budget,
                             tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (tr_conv_batch_check(b) || (b->total && !out)) return -1;                  /* TR_MATH_INVALID */
    tr_math_need need;
    for (int i = 0; i < TR_BUDGET_RESOURCES; i++) need.need[i] = 0;
    need.need[TR_BUDGET_TENSOR] = b->pairs;
    need.need[TR_BUDGET_OUTPUT] = b->total;
    need.need[TR_BUDGET_BYTES] = tr_conv_batch_bytes(b);
    tr_math_budget dflt;
    if (!budget) { tr_budget_default(&dflt); budget = &dflt; }
    int rc = tr_conv_gpu_check(budget, &need, refusal);
    if (rc || !b->total) return rc;
    const uint64_t na = b->aoff[b->npairs], nb = b->boff[b->npairs], np = b->npairs + 1;
    int64_t *da = NULL, *db = NULL, *dout = NULL;
    uint64_t *daoff = NULL, *dboff = NULL, *dooff = NULL;
    rc = tr_cuda_alloc((void **)&da, (size_t)na * 8);
    if (!rc) rc = tr_cuda_alloc((void **)&db, (size_t)nb * 8);
    if (!rc) rc = tr_cuda_alloc((void **)&daoff, (size_t)np * 8);
    if (!rc) rc = tr_cuda_alloc((void **)&dboff, (size_t)np * 8);
    if (!rc) rc = tr_cuda_alloc((void **)&dooff, (size_t)np * 8);
    if (!rc) rc = tr_cuda_alloc((void **)&dout, (size_t)b->total * 24);
    if (!rc) rc = tr_cuda_copy_in(da, b->a, (size_t)na * 8);
    if (!rc) rc = tr_cuda_copy_in(db, b->b, (size_t)nb * 8);
    if (!rc) rc = tr_cuda_copy_in(daoff, b->aoff, (size_t)np * 8);
    if (!rc) rc = tr_cuda_copy_in(dboff, b->boff, (size_t)np * 8);
    if (!rc) rc = tr_cuda_copy_in(dooff, b->ooff, (size_t)np * 8);
    if (!rc) {
        tr_conv_batch d = *b;
        d.a = da; d.b = db; d.aoff = daoff; d.boff = dboff; d.ooff = dooff;
        TR_LAUNCH(tr_conv_batch_kernel, b->total, d, dout);
        rc = tr_cuda_sync();
    }
    if (!rc) rc = tr_cuda_copy_out(out, dout, (size_t)b->total * 24);
    tr_cuda_free(da); tr_cuda_free(db); tr_cuda_free(daoff); tr_cuda_free(dboff); tr_cuda_free(dooff);
    tr_cuda_free(dout);
    return rc;
}

#ifdef __cplusplus
extern "C" {
#endif
int tr_mgpu_conv_batch(const tr_conv_batch *b, int64_t *out, const tr_math_budget *budget, tr_math_refusal *refusal) {
    return tr_conv_gpu_batch(b, out, budget, refusal);
}

int tr_mgpu_conv_conditional(const tr_conv_family_view *a, const tr_conv_family_view *b, const uint64_t *ooff,
                             int64_t *out, const tr_math_budget *budget, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!a || !b || a->keys != b->keys || (a->keys && (!a->offset || !b->offset || !ooff))) return -1;
    tr_conv_batch bt;
    bt.npairs = a->keys; bt.order = 0; bt.aoff = a->offset; bt.boff = b->offset; bt.ooff = ooff;
    bt.a = a->coef; bt.b = b->coef; bt.total = 0; bt.pairs = 0;
    if (a->keys && tr_conv_batch_plan(a->keys, a->offset, b->offset, 0, NULL, &bt.total, &bt.pairs)) return -1;
    return tr_conv_gpu_batch(&bt, out, budget, refusal);
}

int tr_mgpu_conv_walks(const int64_t *step, uint64_t order, uint64_t lengths, int64_t bound, int64_t *rows,
                       int64_t *total, tr_conv_walks *info, const tr_math_budget *budget, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!step || !order || !lengths || !rows || !total || !info || bound < 0) return -1;
    tr_math_need need;
    for (int i = 0; i < TR_BUDGET_RESOURCES; i++) need.need[i] = 0;
    need.need[TR_BUDGET_DEGREE] = order - 1;
    need.need[TR_BUDGET_SUPPORT] = order;
    need.need[TR_BUDGET_ITERATIONS] = lengths - 1;
    need.need[TR_BUDGET_OUTPUT] = tr_budget_mul(lengths, order);
    need.need[TR_BUDGET_BYTES] = tr_conv_walks_bytes(order, lengths);
    tr_math_budget dflt;
    if (!budget) { tr_budget_default(&dflt); budget = &dflt; }
    int rc = tr_conv_gpu_check(budget, &need, refusal);
    if (rc) return rc;
    info->lengths = lengths; info->violation = TR_CONV_NONE; info->status = 0; info->reserved = 0;
    info->bound = bound; info->maxabs = 0;
    if (bound < 1) { info->status = -3; info->violation = 0; return -3; }     /* row 0 (the empty walk) is 1 */
    int64_t *dstep = NULL, *drows = NULL, *dtotal = NULL;
    uint64_t *dflags = NULL, flags[2] = {TR_CONV_NONE, 1};
    rc = tr_cuda_alloc((void **)&dstep, (size_t)order * 8);
    if (!rc) rc = tr_cuda_alloc((void **)&drows, (size_t)(lengths * order) * 8);
    if (!rc) rc = tr_cuda_alloc((void **)&dtotal, (size_t)order * 8);
    if (!rc) rc = tr_cuda_alloc((void **)&dflags, sizeof flags);
    if (!rc) rc = tr_cuda_copy_in(dstep, step, (size_t)order * 8);
    if (!rc) rc = tr_cuda_copy_in(dflags, flags, sizeof flags);
    if (!rc) {                                                        /* row 0: the empty walk, total 0 */
        int64_t one = 1;
        rc = tr_cuda_copy_in(drows, &one, 8);
    }
    for (uint64_t l = 1; l < lengths && !rc; l++) {
        TR_LAUNCH(tr_conv_walk_kernel, order, dstep, order, drows + (l - 1) * order, drows + l * order, l, bound, dflags);
        rc = tr_cuda_sync();
    }
    if (!rc) {
        TR_LAUNCH(tr_conv_total_kernel, order, drows, order, lengths, bound, dtotal, dflags);
        rc = tr_cuda_sync();
    }
    if (!rc) rc = tr_cuda_copy_out(rows, drows, (size_t)(lengths * order) * 8);
    if (!rc) rc = tr_cuda_copy_out(total, dtotal, (size_t)order * 8);
    if (!rc) rc = tr_cuda_copy_out(flags, dflags, sizeof flags);
    tr_cuda_free(dstep); tr_cuda_free(drows); tr_cuda_free(dtotal); tr_cuda_free(dflags);
    if (rc) return rc;
    info->violation = flags[0];
    info->maxabs = (int64_t)flags[1];
    info->status = flags[0] != TR_CONV_NONE ? -3 : 0;                  /* TR_MATH_OVERFLOW */
    return info->status;
}
#ifdef __cplusplus
}
#endif
