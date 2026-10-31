/* DELTA-002 on a CUDA device: the three cross terms of a bilinear join update (or build) and the dense matrix-product
 * delta, with exactly the words of tr_bilin_cross_eval / tr_bilin_matrix_delta (math_bilinear.h).
 * SPDX-License-Identifier: Apache-2.0
 *
 * Cross terms: one work item per (touched key, output group of A, output group of B, value slot) runs the shared
 * tr_bilin_cross_item (four exact 128 x 128-bit products of the old and new cells, combined into T1 = A'B - AB,
 * T2 = AB' - AB, T3 = A'B' - A'B - AB' + AB in 320 bits) and adds its nonzero terms to the output cell with a
 * five-word atomic add (each word gets the carry out of the word below, from the value its own atomic returned), so
 * the words hold the exact sums modulo 2^320, which are the exact sums (|value| < 2^318 by the header's bound), and do
 * not depend on the order of the threads. Matrices: one work item per output entry (g, h) runs tr_bilin_matrix_entry
 * (a sequential exact dot over K) and writes its own five output slots. The host checks the budgets (device copies,
 * work items) before anything is allocated. */
#include "math_cuda.h"
#include "math_bilinear.h"
#include "math_handles.h"

TR_MGPU_ABI(tr_bilin_cross_view, 1);     /* structs passed to the CUDA library (math_handles.h) */
TR_MGPU_ABI(tr_bilin_wide, 1);

/* w += x (mod 2^320, five words): every word is added with the carry out of the word below */
TR_DEVICE void tr_bilin_atomic_add(uint64_t *w, const uint64_t *x) {
    uint64_t carry = 0;
    for (int i = 0; i < TR_BILIN_WORDS; i++) {
        uint64_t y = x[i] + carry, c = y < carry;         /* x[i] + carry wrapped: then y == 0 and no add below */
        if (y) { uint64_t old = tr_atomic_add_u64(&w[i], y); c += (old + y) < old; }
        carry = c;
    }
}

TR_KERNEL void tr_bilin_cross_kernel(tr_bilin_cross_view v, uint64_t items, uint64_t nv, uint64_t *terms) {
    TR_GRID_LOOP(i, items) {
        uint64_t t[3 * TR_BILIN_WORDS];
        uint64_t idx = tr_bilin_cross_item(&v, i, t);
        if (idx == TR_BILIN_NONE) continue;
        for (int k = 0; k < 3; k++)
            if (!tr_bilin_w_is_zero(t + k * TR_BILIN_WORDS))
                tr_bilin_atomic_add(terms + ((uint64_t)k * nv + idx) * TR_BILIN_WORDS, t + k * TR_BILIN_WORDS);
    }
}

TR_KERNEL void tr_bilin_matrix_kernel(uint64_t G, uint64_t K, uint64_t H, const int64_t *A, const int64_t *B,
                                      const int64_t *dA, const int64_t *dB, tr_bilin_wide *out) {
    TR_GRID_LOOP(e, G * H) tr_bilin_matrix_entry(G, K, H, A, B, dA, dB, e / H, e % H, out);
}

static int tr_bilin_view_valid(const tr_bilin_cross_view *v) {
    if (!v || v->groups_a < 1 || v->groups_b < 1 || v->npairs < 1 || v->npairs > TR_BILIN_MAX_PAIRS ||
        v->values != 1 + 2 * v->npairs || v->terms_a < 0 || v->terms_b < 0 || !v->pairs ||
        v->ngroups != v->groups_a * v->groups_b) return 0;
    for (int ver = 0; ver < 2; ver++) {
        if (v->a_rows[ver] && v->terms_a && (!v->a_count[ver] || !v->a_sum[ver])) return 0;
        if (v->b_rows[ver] && v->terms_b && (!v->b_count[ver] || !v->b_sum[ver])) return 0;
    }
    for (int32_t p = 0; p < v->npairs; p++)
        if (v->pairs[2 * p] < 0 || v->pairs[2 * p] >= v->terms_a || v->pairs[2 * p + 1] < 0 ||
            v->pairs[2 * p + 1] >= v->terms_b) return 0;
    return 1;
}

#ifdef __cplusplus
extern "C" {
#endif
int tr_mgpu_bilin_cross(const tr_bilin_cross_view *view, tr_bilin_wide *terms, const tr_math_budget *budget,
                        tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!tr_bilin_view_valid(view) || !terms) return -1;                                       /* TR_MATH_INVALID */
    tr_math_budget dflt;
    if (!budget) { tr_budget_default(&dflt); budget = &dflt; }
    tr_math_need need;
    for (int i = 0; i < TR_BUDGET_RESOURCES; i++) need.need[i] = 0;
    const uint64_t items = tr_bilin_cross_items(view), nv = view->ngroups * (uint64_t)view->values;
    need.need[TR_BUDGET_ITERATIONS] = items;
    need.need[TR_BUDGET_BYTES] = tr_bilin_cross_device_bytes(view);
    int rc = tr_budget_check(budget, &need, refusal);
    if (rc) return rc;
    tr_bilin_cross_view d = *view;
    void *dmem[16];
    int nmem = 0;
    const uint64_t ca = view->keys * view->groups_a, cb = view->keys * view->groups_b;
    const uint64_t ma = (uint64_t)view->terms_a, mb = (uint64_t)view->terms_b;
#define TR_BILIN_COPY(field, src, bytes) do {                                                              \
        void *_p = NULL;                                                                                \
        const size_t _n = (size_t)(bytes);                                                              \
        if (!rc && (src)) {                                                                             \
            rc = tr_cuda_alloc(&_p, _n > 0 ? _n : 1);                                                   \
            if (!rc) { dmem[nmem++] = _p; rc = tr_cuda_copy_in(_p, (src), _n); }                        \
        }                                                                                               \
        field = (src) ? (decltype(field))_p : NULL;                                                     \
    } while (0)
    TR_BILIN_COPY(d.pairs, view->pairs, (uint64_t)view->npairs * 8);
    for (int ver = 0; ver < 2; ver++) {
        TR_BILIN_COPY(d.a_rows[ver], view->a_rows[ver], ca * 8);
        TR_BILIN_COPY(d.a_count[ver], view->a_rows[ver] ? view->a_count[ver] : NULL, ca * ma * 8);
        TR_BILIN_COPY(d.a_sum[ver], view->a_rows[ver] ? view->a_sum[ver] : NULL, ca * ma * 16);
        TR_BILIN_COPY(d.b_rows[ver], view->b_rows[ver], cb * 8);
        TR_BILIN_COPY(d.b_count[ver], view->b_rows[ver] ? view->b_count[ver] : NULL, cb * mb * 8);
        TR_BILIN_COPY(d.b_sum[ver], view->b_rows[ver] ? view->b_sum[ver] : NULL, cb * mb * 16);
    }
#undef TR_BILIN_COPY
    uint64_t *dterms = NULL;
    const size_t tbytes = (size_t)(3 * nv) * sizeof(tr_bilin_wide);
    if (!rc) { rc = tr_cuda_alloc((void **)&dterms, tbytes); if (!rc) dmem[nmem++] = dterms; }   /* zeroed */
    if (!rc && items) {
        TR_LAUNCH(tr_bilin_cross_kernel, items, d, items, nv, dterms);
        rc = tr_cuda_sync();
    }
    if (!rc) rc = tr_cuda_copy_out(terms, dterms, tbytes);
    for (int i = 0; i < nmem; i++) tr_cuda_free(dmem[i]);
    return rc;
}

int tr_mgpu_bilin_matrix(uint64_t G, uint64_t K, uint64_t H, const int64_t *A, const int64_t *B, const int64_t *dA,
                         const int64_t *dB, tr_bilin_wide *out, const tr_math_budget *budget, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!out || (G && K && !A) || (K && H && !B) || G < 1 || H < 1) return -1;               /* TR_MATH_INVALID */
    if (K >= (UINT64_C(1) << 32)) return TR_MATH_UNSUPPORTED;
    tr_math_budget dflt;
    if (!budget) { tr_budget_default(&dflt); budget = &dflt; }
    tr_math_need need;
    tr_bilin_matrix_need_inline(G, K, H, dA != NULL, dB != NULL, 1, &need);
    int rc = tr_budget_check(budget, &need, refusal);
    if (rc) return rc;
    void *dmem[5];
    int nmem = 0;
    int64_t *dA_ = NULL, *dB_ = NULL, *ddA = NULL, *ddB = NULL;
    tr_bilin_wide *dout = NULL;
    const uint64_t gk = G * K, kh = K * H, gh = G * H;
#define TR_BILIN_IN(ptr, src, n) do {                                                                      \
        if (!rc && (src)) {                                                                             \
            rc = tr_cuda_alloc((void **)&(ptr), (size_t)((n) ? (n) * 8 : 1));                          \
            if (!rc) { dmem[nmem++] = (ptr); rc = tr_cuda_copy_in((ptr), (src), (size_t)((n) * 8)); }   \
        }                                                                                               \
    } while (0)
    TR_BILIN_IN(dA_, A, gk);
    TR_BILIN_IN(dB_, B, kh);
    TR_BILIN_IN(ddA, dA, gk);
    TR_BILIN_IN(ddB, dB, kh);
#undef TR_BILIN_IN
    const size_t obytes = (size_t)(5 * gh) * sizeof(tr_bilin_wide);
    if (!rc) { rc = tr_cuda_alloc((void **)&dout, obytes); if (!rc) dmem[nmem++] = dout; }
    if (!rc) {
        TR_LAUNCH(tr_bilin_matrix_kernel, gh, G, K, H, dA_, dB_, dA ? ddA : NULL, dB ? ddB : NULL, dout);
        rc = tr_cuda_sync();
    }
    if (!rc) rc = tr_cuda_copy_out(out, dout, obytes);
    for (int i = 0; i < nmem; i++) tr_cuda_free(dmem[i]);
    return rc;
}
#ifdef __cplusplus
}
#endif
