/* KERNEL-001 on a CUDA device: the feature state of a finite-feature kernel built from a scope (the TENSOR-001 device
 * build of its moment or count tensor, then the per-group weight sums and self-pair terms), and batched kernel-sum
 * requests, with exactly the results of tr_kernel_build and tr_kernel_query (math_kernel.h).
 * SPDX-License-Identifier: Apache-2.0
 *
 * The self-pair pass: a work item is a row slice (rows s, s + S, ...); it sums W = sum w (128 bits), Q = sum w^2 and
 * Dg = sum w^2 k(x, x) (1024 bits, tr_kernel_self_term) locally while the group stays the same and adds them with
 * exact atomics: the two-word atomic of math_cuda.h for W, and for Q and Dg a sixteen-word carry chain of 64-bit
 * atomic adds (a local generalization of tr_atomic_add_wide: each word adds its own word plus the carry of the word
 * below, so the words hold the exact sum modulo 2^1024, which is the exact sum because the host proved its bound
 * before launching). Requests: one work item per request runs the shared evaluator tr_kernel_eval_request on a view
 * whose arrays were uploaded once; one output slot per request. Nothing depends on the thread order. */
#include "math_cuda.h"
#include "math_kernel.h"
#include "math_handles.h"

TR_MGPU_ABI(tr_kernel_spec, 1);          /* structs passed to the CUDA library (math_handles.h) */
TR_MGPU_ABI(tr_kernel_wide, 1);
TR_MGPU_ABI(tr_kernel_request, 1);
TR_MGPU_ABI(tr_kernel_result, 1);
TR_MGPU_ABI(tr_kernel_view, 1);

#define TR_KERNEL_SLICE_ROWS 64          /* at least this many rows per self-pair work item */
#define TR_KERNEL_SLICES (UINT64_C(1) << 16)

TR_DEVICE void tr_kernel_atomic_add_words(uint64_t *dst, const tr_kernel_wide *x) {
    uint64_t carry = 0;
    for (int i = 0; i < TR_KERNEL_WORDS; i++) {
        uint64_t v = x->w[i] + carry, c1 = v < carry;               /* x + carry wrapped: only MAX + 1 */
        if (v) { uint64_t old = tr_atomic_add_u64(&dst[i], v); carry = c1 | ((old + v) < old); }
        else carry = c1;
    }
}

TR_KERNEL void tr_kernel_self_kernel(tr_scope_view v, const tr_kernel_spec *k, int32_t weight, uint64_t slices,
                                     int64_t *W, uint64_t *Q, uint64_t *Dg, uint64_t *overflow) {
    TR_GRID_LOOP(s, slices) {
        uint32_t cur = 0;
        uint64_t n = 0;
        __int128 w = 0;
        tr_kernel_wide q, d, t;
        tr_kernel_w_zero(&q); tr_kernel_w_zero(&d);
        for (uint64_t r = s; r < v.rows; r += slices) {
            if (!tr_scope_is_selected(&v, r)) continue;
            __int128 wt;
            if (!tr_kernel_row_ok(&v, k, weight, r, &wt)) continue;
            uint32_t g = tr_scope_group(&v, r);
            if (n && g != cur) {
                if (w) tr_atomic_add_wide(&W[2 * cur], &W[2 * cur + 1], w);
                tr_kernel_atomic_add_words(Q + (uint64_t)cur * TR_KERNEL_WORDS, &q);
                tr_kernel_atomic_add_words(Dg + (uint64_t)cur * TR_KERNEL_WORDS, &d);
                n = 0; w = 0; tr_kernel_w_zero(&q); tr_kernel_w_zero(&d);
            }
            cur = g; n++; w += wt;
            tr_kernel_w_from_i128(wt * wt, &t);
            if (!tr_kernel_w_add(&q, &t, &q) || !tr_kernel_self_term(&v, k, r, wt, &t) || !tr_kernel_w_add(&d, &t, &d))
                tr_atomic_or_u64(overflow, 1);
        }
        if (n) {
            if (w) tr_atomic_add_wide(&W[2 * cur], &W[2 * cur + 1], w);
            tr_kernel_atomic_add_words(Q + (uint64_t)cur * TR_KERNEL_WORDS, &q);
            tr_kernel_atomic_add_words(Dg + (uint64_t)cur * TR_KERNEL_WORDS, &d);
        }
    }
}

TR_KERNEL void tr_kernel_query_kernel(tr_kernel_view v, const tr_kernel_request *req, uint64_t n, const int64_t *points,
                                      const uint8_t *valid, tr_kernel_result *out) {
    TR_GRID_LOOP(i, n) tr_kernel_eval_request(&v, &req[i], points, valid, &out[i]);
}

struct tr_mgpu_kernel_s {
    tr_kernel_view view;                 /* device pointers */
    void *mem[12];
    int nmem;
    uint64_t bytes;
};

#ifdef __cplusplus
extern "C" {
#endif
int tr_mgpu_kernel_build(const tr_scope_view *view, const tr_kernel_spec *kernel, int32_t weight, int32_t degree,
                         int32_t repr, const tr_math_budget *budget, tr_tensor_states *tensor, tr_math_i128 *W,
                         tr_kernel_wide *Q, tr_kernel_wide *Dg, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!view || !kernel || !tensor || !W || !Q || !Dg || view->ngroups < 1 || (view->rows && !view->selection) ||
        (view->rows && view->ngroups > 1 && !view->groups) || view->ncols < 0 || view->ncols > TR_SCOPE_MAX_COLUMNS ||
        (weight != -1 && (weight < 0 || weight >= view->ncols)) || degree < 0 ||
        (repr != TR_TENSOR_DENSE && repr != TR_TENSOR_SPARSE)) return -1;
    int rc = tr_kernel_spec_valid(kernel, view->ncols);
    if (rc) return rc;
    if (kernel->kind != TR_KERNEL_DELTA)
        for (int32_t j = 1; j < kernel->ncols; j++)
            if (view->col[kernel->cols[j]].scale != view->col[kernel->cols[0]].scale) return -6;   /* TR_MATH_DOMAIN_ERROR */
    int32_t D = kernel->kind == TR_KERNEL_DELTA ? 0 : (degree ? degree : kernel->degree);
    if ((kernel->kind == TR_KERNEL_DELTA && degree) || (kernel->kind != TR_KERNEL_DELTA && D < kernel->degree)) return -1;
    if (D > TR_KERNEL_MAX_DEGREE) return TR_MATH_UNSUPPORTED;
    if (kernel->kind != TR_KERNEL_DELTA && D == 0) D = 1;          /* as tr_kernel_build: the axes read the columns */
    if (tr_kernel_self_bits(view, kernel, weight) > TR_KERNEL_BITS_LIMIT) return -3;               /* TR_MATH_OVERFLOW */
    tr_tensor_spec spec;
    rc = tr_kernel_tensor_spec(kernel, D, weight, repr, &spec);
    if (rc) return rc;
    /* the self-pair pass's device copies: selection, ids, the kernel's columns and weight, W, Q, Dg, the kernel */
    const uint64_t n = view->rows, G = view->ngroups;
    uint64_t bytes = tr_budget_add(tr_budget_bitmap_bytes(n), tr_budget_mul(G, TR_KERNEL_GROUP_BYTES));
    bytes = tr_budget_add(bytes, sizeof(tr_kernel_spec) + 8);
    if (view->groups) bytes = tr_budget_add(bytes, tr_budget_mul(n, 4));
    uint32_t read = 0;
    for (int32_t j = 0; j < kernel->ncols; j++) read |= UINT32_C(1) << kernel->cols[j];
    if (weight >= 0) read |= UINT32_C(1) << weight;
    for (int32_t c = 0; c < view->ncols; c++) {
        if (!((read >> c) & 1)) continue;
        bytes = tr_budget_add(bytes, tr_budget_mul(n, view->col[c].values ? 8 : 4));
        if (view->col[c].valid) bytes = tr_budget_add(bytes, n);
    }
    tr_math_need need;
    for (int i = 0; i < TR_BUDGET_RESOURCES; i++) need.need[i] = 0;
    need.need[TR_BUDGET_BYTES] = bytes;
    tr_math_budget dflt;
    if (!budget) { tr_budget_default(&dflt); budget = &dflt; }
    rc = tr_budget_check(budget, &need, refusal);
    if (rc) return rc;
    rc = tr_mgpu_tensor_build(view, &spec, budget, tensor, refusal);       /* the feature tensor (TENSOR-001) */
    if (rc) return rc;
    memset(W, 0, (size_t)G * 16);
    memset(Q, 0, (size_t)G * sizeof(tr_kernel_wide));
    memset(Dg, 0, (size_t)G * sizeof(tr_kernel_wide));
    if (!view->selected) return 0;
    tr_scope_view d = *view;
    void *dmem[3 * TR_SCOPE_MAX_COLUMNS + 8];
    int nmem = 0;
#define TR_KERNEL_DEV(ptr, b) do { if (!rc) { rc = tr_cuda_alloc((void **)&(ptr), (size_t)(b)); if (!rc) dmem[nmem++] = (void *)(ptr); } } while (0)
    uint64_t *dsel = NULL, *dQ = NULL, *dD = NULL, *dovf = NULL, ovf = 0;
    uint32_t *dgroups = NULL;
    int64_t *dW = NULL;
    tr_kernel_spec *dk = NULL;
    TR_KERNEL_DEV(dsel, ((n + 63) / 64) * 8);
    if (!rc) rc = tr_cuda_copy_in(dsel, view->selection, (size_t)(((n + 63) / 64) * 8));
    d.selection = dsel;
    if (view->groups) {
        TR_KERNEL_DEV(dgroups, n * 4);
        if (!rc) rc = tr_cuda_copy_in(dgroups, view->groups, (size_t)n * 4);
        d.groups = dgroups;
    }
    for (int32_t c = 0; c < view->ncols; c++) {
        tr_scope_view_column *k = &d.col[c];
        if (!((read >> c) & 1)) { k->codes = NULL; k->values = NULL; k->valid = NULL; continue; }
        if (view->col[c].values) {
            int64_t *q = NULL;
            TR_KERNEL_DEV(q, n * 8);
            if (!rc) rc = tr_cuda_copy_in(q, view->col[c].values, (size_t)n * 8);
            k->values = q;
        } else {
            uint32_t *q = NULL;
            TR_KERNEL_DEV(q, n * 4);
            if (!rc) rc = tr_cuda_copy_in(q, view->col[c].codes, (size_t)n * 4);
            k->codes = q;
        }
        if (view->col[c].valid) {
            uint8_t *q = NULL;
            TR_KERNEL_DEV(q, n);
            if (!rc) rc = tr_cuda_copy_in(q, view->col[c].valid, (size_t)n);
            k->valid = q;
        }
    }
    TR_KERNEL_DEV(dk, sizeof(tr_kernel_spec));
    if (!rc) rc = tr_cuda_copy_in(dk, kernel, sizeof(tr_kernel_spec));
    TR_KERNEL_DEV(dW, G * 16);
    TR_KERNEL_DEV(dQ, G * sizeof(tr_kernel_wide));
    TR_KERNEL_DEV(dD, G * sizeof(tr_kernel_wide));
    TR_KERNEL_DEV(dovf, 8);
#undef TR_KERNEL_DEV
    if (!rc) {
        uint64_t s = (n + TR_KERNEL_SLICE_ROWS - 1) / TR_KERNEL_SLICE_ROWS;
        if (s > TR_KERNEL_SLICES) s = TR_KERNEL_SLICES;
        if (!s) s = 1;
        TR_LAUNCH(tr_kernel_self_kernel, s, d, dk, weight, s, dW, dQ, dD, dovf);
        rc = tr_cuda_sync();
    }
    if (!rc) rc = tr_cuda_copy_out(&ovf, dovf, 8);
    if (!rc) rc = tr_cuda_copy_out(W, dW, (size_t)G * 16);
    if (!rc) rc = tr_cuda_copy_out(Q, dQ, (size_t)G * sizeof(tr_kernel_wide));
    if (!rc) rc = tr_cuda_copy_out(Dg, dD, (size_t)G * sizeof(tr_kernel_wide));
    for (int i = 0; i < nmem; i++) tr_cuda_free(dmem[i]);
    if (!rc && ovf) rc = -3;                       /* unreachable after the proof */
    return rc;
}

tr_mgpu_kernel *tr_mgpu_kernel_upload(const tr_kernel_view *view, uint64_t nnz, const tr_math_budget *budget,
                                      tr_math_refusal *refusal, int *error) {
    tr_budget_none(refusal);
    if (error) *error = 0;
    if (!view || view->ngroups < 1 || view->order < 0 || view->order > TR_TENSOR_MAX_ORDER ||
        (view->repr != TR_TENSOR_DENSE && view->repr != TR_TENSOR_SPARSE) || !view->W || !view->Q || !view->Dg ||
        !view->class_coef || !view->pos_of_axis ||
        (view->nitems && (!view->item_index || !view->item_class || !view->item_mult)) ||
        view->nclass < 1 || view->nclass > TR_KERNEL_MAX_TERMS ||
        (view->kind != TR_KERNEL_MONOMIAL && view->nclass > TR_KERNEL_MAX_CLASSES) ||
        (view->repr == TR_TENSOR_SPARSE && (!view->offset || view->offset[view->ngroups] != nnz))) {
        if (error) *error = -1;
        return NULL;
    }
    const uint64_t G = view->ngroups, entries = view->repr == TR_TENSOR_DENSE ? tr_budget_mul(G, view->cells) : nnz;
    for (int32_t i = 0; i <= TR_TENSOR_MAX_FEATURES; i++)   /* point columns in range */
        if (view->pos_of_axis[i] < -1 || view->pos_of_axis[i] >= view->npos) { if (error) *error = -1; return NULL; }
    for (uint64_t t = 0; t < view->nitems; t++)       /* items inside a group's cells, classes and multiplicities in range */
        if (view->item_index[t] >= view->cells || view->item_class[t] < 0 || view->item_class[t] >= view->nclass ||
            view->item_mult[t] < 1 || view->item_mult[t] > 40320) {
            if (error) *error = -1;
            return NULL;
        }
    tr_math_need need;
    for (int i = 0; i < TR_BUDGET_RESOURCES; i++) need.need[i] = 0;
    need.need[TR_BUDGET_BYTES] = tr_budget_add(tr_kernel_view_bytes(view, nnz), sizeof(tr_mgpu_kernel));
    tr_math_budget dflt;
    if (!budget) { tr_budget_default(&dflt); budget = &dflt; }
    int rc = tr_budget_check(budget, &need, refusal);
    if (rc) { if (error) *error = rc; return NULL; }
    tr_mgpu_kernel *h = (tr_mgpu_kernel *)calloc(1, sizeof *h);
    if (!h) { if (error) *error = -2; return NULL; }
    h->view = *view;
    h->bytes = need.need[TR_BUDGET_BYTES];
#define TR_KERNEL_UP(field, type, count) do { if (!rc) { type *p = NULL; uint64_t b_ = (uint64_t)(count) * sizeof(type); \
        rc = tr_cuda_alloc((void **)&p, (size_t)b_); if (!rc) { h->mem[h->nmem++] = p; rc = tr_cuda_copy_in(p, view->field, (size_t)b_); } \
        h->view.field = p; } } while (0)
    TR_KERNEL_UP(value, tr_math_i128, entries);
    if (view->repr == TR_TENSOR_SPARSE) {
        TR_KERNEL_UP(offset, uint64_t, G + 1);
        TR_KERNEL_UP(coord, uint32_t, nnz * (uint64_t)view->order);
    }
    TR_KERNEL_UP(W, tr_math_i128, G);
    TR_KERNEL_UP(Q, tr_kernel_wide, G);
    TR_KERNEL_UP(Dg, tr_kernel_wide, G);
    TR_KERNEL_UP(class_coef, tr_math_i128, view->nclass);
    TR_KERNEL_UP(pos_of_axis, int32_t, TR_TENSOR_MAX_FEATURES + 1);
    if (view->nitems) {
        TR_KERNEL_UP(item_index, uint64_t, view->nitems);
        TR_KERNEL_UP(item_class, int32_t, view->nitems);
        TR_KERNEL_UP(item_mult, uint32_t, view->nitems);
    }
#undef TR_KERNEL_UP
    if (rc) { for (int i = 0; i < h->nmem; i++) tr_cuda_free(h->mem[i]); free(h); if (error) *error = rc; return NULL; }
    return h;
}

int tr_mgpu_kernel_query(const tr_mgpu_kernel *h, const tr_kernel_request *requests, uint64_t n, const int64_t *points,
                         const uint8_t *valid, uint64_t npoints, tr_kernel_result *out, const tr_math_budget *budget,
                         tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!h || (n && (!requests || !out)) || (npoints && !points)) return -1;
    const uint64_t np = (uint64_t)h->view.npos;
    tr_math_need need;
    for (int i = 0; i < TR_BUDGET_RESOURCES; i++) need.need[i] = 0;
    need.need[TR_BUDGET_OUTPUT] = n;
    uint64_t bytes = tr_budget_mul(n, sizeof(tr_kernel_request) + sizeof(tr_kernel_result));
    bytes = tr_budget_add(bytes, tr_budget_mul(tr_budget_mul(npoints, np), valid ? 9 : 8));
    need.need[TR_BUDGET_BYTES] = bytes;
    tr_math_budget dflt;
    if (!budget) { tr_budget_default(&dflt); budget = &dflt; }
    int rc = tr_budget_check(budget, &need, refusal);
    if (rc || !n) return rc;
    tr_kernel_request *dreq = NULL;
    tr_kernel_result *dout = NULL;
    int64_t *dpts = NULL;
    uint8_t *dval = NULL;
    rc = tr_cuda_alloc((void **)&dreq, (size_t)n * sizeof *dreq);
    if (!rc) rc = tr_cuda_alloc((void **)&dout, (size_t)n * sizeof *dout);
    if (!rc && npoints) rc = tr_cuda_alloc((void **)&dpts, (size_t)(npoints * np) * 8);
    if (!rc && npoints && valid) rc = tr_cuda_alloc((void **)&dval, (size_t)(npoints * np));
    if (!rc) rc = tr_cuda_copy_in(dreq, requests, (size_t)n * sizeof *dreq);
    if (!rc && npoints) rc = tr_cuda_copy_in(dpts, points, (size_t)(npoints * np) * 8);
    if (!rc && dval) rc = tr_cuda_copy_in(dval, valid, (size_t)(npoints * np));
    if (!rc) {
        tr_kernel_view v = h->view;
        v.npoints = npoints;
        TR_LAUNCH(tr_kernel_query_kernel, n, v, dreq, n, dpts, dval, dout);
        rc = tr_cuda_sync();
    }
    if (!rc) rc = tr_cuda_copy_out(out, dout, (size_t)n * sizeof *dout);
    tr_cuda_free(dreq); tr_cuda_free(dout); tr_cuda_free(dpts); tr_cuda_free(dval);
    return rc;
}
uint64_t tr_mgpu_kernel_bytes(const tr_mgpu_kernel *h) { return h ? h->bytes : 0; }
void tr_mgpu_kernel_free(tr_mgpu_kernel *h) {
    if (!h) return;
    for (int i = 0; i < h->nmem; i++) tr_cuda_free(h->mem[i]);
    free(h);
}
#ifdef __cplusplus
}
#endif
