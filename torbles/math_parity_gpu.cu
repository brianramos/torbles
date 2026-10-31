/* FIELD-002 on a CUDA device: the verification of a substituted predicate against the original on every selected row,
 * with exactly the CPU's counts (math_parity.h).
 * SPDX-License-Identifier: Apache-2.0
 *
 * Each work item is one 64-row word (or, when the node words of one work item per word would pass TR_PARITY_SCRATCH,
 * a strided slice of words s, s + S, ...): it evaluates both programs at its word with bitwise operations (a value and
 * a valid word per node: SQL three-valued logic), takes the two root bitmaps, XORs them (value and validity) into the
 * mismatch word, counts every class with popcounts, keeps its first mismatching row, and adds its counts with one
 * integer atomic each and its witness with an atomic minimum. Integer atomics commute: the counts do not depend on
 * thread order. */
#include "math_cuda.h"
#include "math_parity.h"
#include "math_handles.h"

TR_MGPU_ABI(tr_parity_check, TR_PARITY_ABI);      /* structs passed to the CUDA library (math_handles.h) */

/* acc[0..7] = a_true, a_false, a_unknown, b_true, b_false, b_unknown, mismatches, the smallest mismatching row */
TR_KERNEL void tr_parity_verify_kernel(tr_bool_domain d, tr_bool_program_view a, tr_bool_program_view b,
                                       uint64_t items, uint64_t *scratch, uint64_t *acc) {
    TR_GRID_LOOP(item, items) {
        uint64_t *va = scratch + item * 2 * ((uint64_t)a.nnodes + (uint64_t)b.nnodes);
        uint64_t *vb = va + 2 * (uint64_t)a.nnodes;
        const uint64_t ra = (uint64_t)a.root, rb = (uint64_t)b.root;
        uint64_t c[7] = {0, 0, 0, 0, 0, 0, 0}, witness = UINT64_MAX;
        for (uint64_t w = item; w < d.nw; w += items) {
            uint64_t sel = tr_bool_selection_word(&d, w);
            if (!sel) continue;
            tr_bool_eval_base(&a, &d, w, va);
            tr_bool_eval_base(&b, &d, w, vb);
            uint64_t xa = va[2 * ra] & sel, ka = va[2 * ra + 1] & sel, xb = vb[2 * rb] & sel, kb = vb[2 * rb + 1] & sel;
            uint64_t m = tr_parity_mismatch(xa, ka, xb, kb, sel);
            c[0] += (uint64_t)tr_bool_popc(xa); c[1] += (uint64_t)tr_bool_popc(ka & ~xa); c[2] += (uint64_t)tr_bool_popc(sel & ~ka);
            c[3] += (uint64_t)tr_bool_popc(xb); c[4] += (uint64_t)tr_bool_popc(kb & ~xb); c[5] += (uint64_t)tr_bool_popc(sel & ~kb);
            c[6] += (uint64_t)tr_bool_popc(m);
            if (m && witness == UINT64_MAX) witness = w * 64 + (uint64_t)tr_bool_ctz(m);
        }
        for (int i = 0; i < 7; i++) if (c[i]) tr_atomic_add_u64(&acc[i], c[i]);
        if (c[6]) tr_atomic_min_u64(&acc[7], witness);
    }
}

#ifdef __cplusplus
extern "C" {
#endif
int tr_mgpu_parity_verify(const tr_bool_domain *dom, const tr_bool_program_view *pa, const tr_bool_program_view *pb,
                          const tr_math_budget *budget, tr_parity_check *out, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    int rc = tr_bool_domain_check(dom);
    if (!rc) rc = tr_bool_program_check(pa);
    if (!rc) rc = tr_bool_program_check(pb);
    if (rc) return rc;
    if (pa->k != dom->k || pb->k != dom->k) return -6;                                       /* TR_MATH_DOMAIN_ERROR */
    if (!out) return -1;                                                                     /* TR_MATH_INVALID */
    tr_math_budget dflt;
    if (!budget) { tr_budget_default(&dflt); budget = &dflt; }
    tr_math_need need;
    for (int i = 0; i < TR_BUDGET_RESOURCES; i++) need.need[i] = 0;
    need.need[TR_BUDGET_VARIABLES] = (uint64_t)dom->k;
    need.need[TR_BUDGET_CIRCUIT] = tr_budget_add((uint64_t)pa->nnodes, (uint64_t)pb->nnodes);
    need.need[TR_BUDGET_SUPPORT] = dom->kind == TR_BOOL_ASSIGNMENTS ? dom->points : 0;
    need.need[TR_BUDGET_BYTES] = tr_parity_device_bytes(dom, pa, pb);
    rc = tr_budget_check(budget, &need, refusal);
    if (rc) return rc;
    memset(out, 0, sizeof *out);
    out->points = dom->selected; out->witness = TR_PARITY_NONE;
    if (!dom->selected || !dom->nw) { out->equivalent = 1; return 0; }
    const uint64_t nw = dom->nw, items = tr_parity_slices(nw, pa->nnodes, pb->nnodes);
    void *mem[12];
    int nmem = 0;
    tr_bool_domain d = *dom;
    tr_bool_program_view a = *pa, b = *pb;
    a.deps = NULL; b.deps = NULL;                             /* never read by the evaluation */
    uint64_t *dsel = NULL, *dvalue = NULL, *dvalid = NULL, *dscratch = NULL, *dacc = NULL;
    tr_bool_node *na = NULL, *nb = NULL;
    uint32_t *sa = NULL, *sb = NULL;
#define TR_PARITY_DEV(ptr, bytes) do { if (!rc) { rc = tr_cuda_alloc((void **)&(ptr), (size_t)(bytes)); if (!rc) mem[nmem++] = (void *)(ptr); } } while (0)
    if (dom->kind == TR_BOOL_ROWS) {
        uint64_t words = (uint64_t)dom->k * nw;
        TR_PARITY_DEV(dsel, nw * 8);
        if (!rc) rc = tr_cuda_copy_in(dsel, dom->selection, (size_t)(nw * 8));
        TR_PARITY_DEV(dvalue, words * 8);
        if (!rc && words) rc = tr_cuda_copy_in(dvalue, dom->value, (size_t)(words * 8));
        if (dom->valid) {
            TR_PARITY_DEV(dvalid, words * 8);
            if (!rc && words) rc = tr_cuda_copy_in(dvalid, dom->valid, (size_t)(words * 8));
        }
        d.selection = dsel; d.value = dvalue; d.valid = dvalid;
    }
    TR_PARITY_DEV(na, (uint64_t)pa->nnodes * sizeof(tr_bool_node));
    if (!rc) rc = tr_cuda_copy_in(na, pa->nodes, (size_t)pa->nnodes * sizeof(tr_bool_node));
    TR_PARITY_DEV(nb, (uint64_t)pb->nnodes * sizeof(tr_bool_node));
    if (!rc) rc = tr_cuda_copy_in(nb, pb->nodes, (size_t)pb->nnodes * sizeof(tr_bool_node));
    TR_PARITY_DEV(sa, (uint64_t)pa->nsched * 4);
    if (!rc) rc = tr_cuda_copy_in(sa, pa->schedule, (size_t)pa->nsched * 4);
    TR_PARITY_DEV(sb, (uint64_t)pb->nsched * 4);
    if (!rc) rc = tr_cuda_copy_in(sb, pb->schedule, (size_t)pb->nsched * 4);
    a.nodes = na; a.schedule = sa; b.nodes = nb; b.schedule = sb;
    TR_PARITY_DEV(dscratch, items * 16 * ((uint64_t)pa->nnodes + (uint64_t)pb->nnodes));
    TR_PARITY_DEV(dacc, 64);
#undef TR_PARITY_DEV
    uint64_t acc[8] = {0, 0, 0, 0, 0, 0, 0, UINT64_MAX};
    if (!rc) rc = tr_cuda_copy_in(dacc, acc, 64);
    if (!rc) { TR_LAUNCH(tr_parity_verify_kernel, items, d, a, b, items, dscratch, dacc); rc = tr_cuda_sync(); }
    if (!rc) rc = tr_cuda_copy_out(acc, dacc, 64);
    if (!rc) {
        out->a_true = acc[0]; out->a_false = acc[1]; out->a_unknown = acc[2];
        out->b_true = acc[3]; out->b_false = acc[4]; out->b_unknown = acc[5];
        out->mismatches = acc[6]; out->witness = acc[6] ? acc[7] : TR_PARITY_NONE;
        out->equivalent = acc[6] == 0;
    }
    for (int i = 0; i < nmem; i++) tr_cuda_free(mem[i]);
    return rc;
}
#ifdef __cplusplus
}
#endif
