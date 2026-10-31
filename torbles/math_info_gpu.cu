/* DIST-003 on a CUDA device: the certified per-cell terms of entropy and chi-square sums over a support's tables and
 * their exact accumulation, with exactly the integers of tr_info_sums (math_info.h).
 * SPDX-License-Identifier: Apache-2.0
 *
 * Two launches over the support's entries (tr_info_view), one work item per TR_INFO_CHUNK consecutive entries:
 *   log  each cell's bounds floor(m lo / 2^8), ceil(m hi / 2^8) with lo <= ln(m) 2^120 <= hi (tr_info_log_term: the
 *        120-bit fixed-point logarithm of math_info.h, integer arithmetic only)
 *   chi  each cell's floor / ceil(O^2 2^128 / (m_a m_b)) (tr_info_chi_term: exact integer division)
 * A work item adds the bounds of its entries locally while their key (group x slots + slot) stays the same (the entries
 * are sorted by table and group) and flushes each local sum with exact 256-bit atomic additions (four 64-bit atomic adds
 * with carries: the total of every word and of its carries is the same in every order). Every bound is a nonnegative
 * integer and every sum stays below 2^190, so the device sums equal the CPU's bit for bit whatever the order of the
 * threads. The host checks the view and budgets the device copies (tr_info_device_bytes) before launching. */
#include "math_cuda.h"
#include "math_info.h"
#include "math_handles.h"

TR_MGPU_ABI(tr_info_view, TR_INFO_ABI);        /* structs passed to the CUDA library (math_handles.h) */

/* exact 256-bit accumulation into four words: word i receives x.w[i] plus the carry out of word i - 1 */
TR_DEVICE void tr_info_atomic_add256(int64_t *w, tr_dist_i256 x) {
    uint64_t carry = 0;
    for (int i = 0; i < 4; i++) {
        uint64_t a = x.w[i] + carry, c1 = a < carry, c2 = 0;
        if (a) {
            uint64_t old = tr_atomic_add_u64((uint64_t *)&w[i], a);
            c2 = (old + a) < old;
        }
        carry = c1 | c2;
    }
}
TR_DEVICE void tr_info_flush(int64_t *sums, uint64_t key, tr_dist_i256 L, tr_dist_i256 H) {
    if (!tr_dist_w_is_zero(L)) tr_info_atomic_add256(&sums[key * 8], L);
    if (!tr_dist_w_is_zero(H)) tr_info_atomic_add256(&sums[key * 8 + 4], H);
}

TR_KERNEL void tr_info_log_kernel(tr_info_view v, uint64_t chunks, int64_t *sums) {
    TR_GRID_LOOP(c, chunks) {
        uint64_t b = c * TR_INFO_CHUNK, e = v.nlog - b < TR_INFO_CHUNK ? v.nlog : b + TR_INFO_CHUNK, key = 0;
        tr_dist_i256 accL = tr_dist_w_zero(), accH = tr_dist_w_zero();
        int have = 0;
        for (uint64_t i = b; i < e; i++) {
            if (have && v.lkey[i] != key) { tr_info_flush(sums, key, accL, accH); accL = accH = tr_dist_w_zero(); }
            key = v.lkey[i]; have = 1;
            tr_dist_i256 L, H;
            tr_info_log_term(v.lmass[i], &L, &H);
            tr_info_add256(accL.w, L);
            tr_info_add256(accH.w, H);
        }
        if (have) tr_info_flush(sums, key, accL, accH);
    }
}

TR_KERNEL void tr_info_chi_kernel(tr_info_view v, uint64_t chunks, int64_t *sums) {
    TR_GRID_LOOP(c, chunks) {
        uint64_t b = c * TR_INFO_CHUNK, e = v.nchi - b < TR_INFO_CHUNK ? v.nchi : b + TR_INFO_CHUNK, key = 0;
        tr_dist_i256 accL = tr_dist_w_zero(), accH = tr_dist_w_zero();
        int have = 0;
        for (uint64_t i = b; i < e; i++) {
            if (have && v.ckey[i] != key) { tr_info_flush(sums, key, accL, accH); accL = accH = tr_dist_w_zero(); }
            key = v.ckey[i]; have = 1;
            tr_dist_i256 L, H;
            tr_info_chi_term(v.co[i], v.cma[i], v.cmb[i], &L, &H);
            tr_info_add256(accL.w, L);
            tr_info_add256(accH.w, H);
        }
        if (have) tr_info_flush(sums, key, accL, accH);
    }
}

#ifdef __cplusplus
extern "C" {
#endif
int tr_mgpu_info_sums(const tr_info_view *view, const tr_math_budget *budget, int64_t *sums, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!tr_info_view_valid(view) || !sums) return -1;                                  /* TR_MATH_INVALID */
    const uint64_t total = view->ngroups * view->nslots;
    for (uint64_t i = 0; i < view->nlog; i++) if (view->lkey[i] >= total || !view->lmass[i]) return -1;
    for (uint64_t i = 0; i < view->nchi; i++)
        if (view->ckey[i] >= total || !view->co[i] || view->co[i] > view->cma[i] || view->co[i] > view->cmb[i]) return -1;
    tr_math_need need;
    for (int i = 0; i < TR_BUDGET_RESOURCES; i++) need.need[i] = 0;
    need.need[TR_BUDGET_STATES] = total;
    need.need[TR_BUDGET_BYTES] = tr_info_device_bytes(view);
    tr_math_budget dflt;
    if (!budget) { tr_budget_default(&dflt); budget = &dflt; }
    int rc = tr_budget_check(budget, &need, refusal);
    if (!rc && need.need[TR_BUDGET_BYTES] == TR_BUDGET_UNKNOWN) {
        if (refusal) { refusal->resource = TR_BUDGET_BYTES; refusal->refused = 1; refusal->need = TR_BUDGET_UNKNOWN;
                       refusal->limit = budget->limit[TR_BUDGET_BYTES]; }
        rc = TR_MATH_BUDGET_CODE;
    }
    if (rc) return rc;
    tr_info_view d = *view;
    uint64_t *lkey = NULL, *lmass = NULL, *ckey = NULL, *co = NULL, *cma = NULL, *cmb = NULL;
    int64_t *dsums = NULL;
    void *mem[7];
    int nmem = 0;
#define TR_INFO_DEV(ptr, src, n) do { if (!rc) { rc = tr_cuda_alloc((void **)&(ptr), (size_t)(n) * 8); \
        if (!rc) { mem[nmem++] = (void *)(ptr); rc = tr_cuda_copy_in((ptr), (src), (size_t)(n) * 8); } } } while (0)
    if (view->nlog) { TR_INFO_DEV(lkey, view->lkey, view->nlog); TR_INFO_DEV(lmass, view->lmass, view->nlog); }
    if (view->nchi) {
        TR_INFO_DEV(ckey, view->ckey, view->nchi); TR_INFO_DEV(co, view->co, view->nchi);
        TR_INFO_DEV(cma, view->cma, view->nchi); TR_INFO_DEV(cmb, view->cmb, view->nchi);
    }
#undef TR_INFO_DEV
    if (!rc) { rc = tr_cuda_alloc((void **)&dsums, (size_t)total * 64); if (!rc) mem[nmem++] = dsums; }   /* zeroed */
    d.lkey = lkey; d.lmass = lmass; d.ckey = ckey; d.co = co; d.cma = cma; d.cmb = cmb;
    if (!rc) {
        uint64_t lc = (view->nlog + TR_INFO_CHUNK - 1) / TR_INFO_CHUNK, cc = (view->nchi + TR_INFO_CHUNK - 1) / TR_INFO_CHUNK;
        if (lc) TR_LAUNCH(tr_info_log_kernel, lc, d, lc, dsums);
        if (cc) TR_LAUNCH(tr_info_chi_kernel, cc, d, cc, dsums);
        rc = tr_cuda_sync();
    }
    if (!rc) rc = tr_cuda_copy_out(sums, dsums, (size_t)total * 64);
    for (int i = 0; i < nmem; i++) tr_cuda_free(mem[i]);
    return rc;
}
#ifdef __cplusplus
}
#endif
