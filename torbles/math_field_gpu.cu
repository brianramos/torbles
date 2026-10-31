/* FIELD-001 on a CUDA device: batched syndromes (matrix-vector products over many vectors) and the exhaustive
 * verification of parity relations over every row, with exactly the CPU's results (math_field.h); the discovery loop
 * of the header with this verification.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Syndromes: one work item per (vector, 64-bit syndrome word) computes its 64 parities with the shared
 * tr_field_syndrome_word and writes its own output word.
 * Verification: work items (relation, word slice): slice s of S takes the words s, s + S, ...; a work item XORs the
 * relation's value words per 64-row word (tr_field_relation_word), counts checked, excluded and violating rows with
 * popcounts and keeps its first violating row (the slice visits its words in increasing order), then adds its counts
 * with one atomic each and its witness with an atomic minimum. Integer atomics commute: the evidence does not depend
 * on thread order. */
#include "math_cuda.h"
#include "math_field.h"
#include "math_handles.h"

TR_MGPU_ABI(tr_field_matrix, TR_FIELD_ABI);        /* structs passed to the CUDA library (math_handles.h) */
TR_MGPU_ABI(tr_field_bits_view, TR_FIELD_ABI);
TR_MGPU_ABI(tr_field_relation, TR_FIELD_ABI);
TR_MGPU_ABI(tr_field_evidence, TR_FIELD_ABI);
TR_MGPU_ABI(tr_field_discover_spec, TR_FIELD_ABI);
TR_MGPU_ABI(tr_field_discovery, TR_FIELD_ABI);

#define TR_FIELD_WORK (UINT64_C(1) << 15)          /* about this many verification work items per launch */
#define TR_FIELD_MIN_WORDS 16                       /* and at least this many words per work item */

TR_KERNEL void tr_field_syndromes_kernel(tr_field_matrix h, tr_field_matrix x, uint64_t *out, uint64_t items,
                                         uint64_t owords) {
    TR_GRID_LOOP(item, items) {
        uint64_t j = item / owords, ow = item % owords;
        out[item] = tr_field_syndrome_word(&h, x.words + j * x.stride, ow);
    }
}

/* acc[4 r + 0..3] = checked, excluded, violations, the smallest violating row */
TR_KERNEL void tr_field_verify_kernel(tr_field_bits_view v, const tr_field_relation *rels, int32_t mode,
                                      uint64_t items, uint64_t slices, uint64_t *acc) {
    TR_GRID_LOOP(item, items) {
        const uint64_t r = item / slices, s = item % slices;
        const tr_field_relation rel = rels[r];
        uint64_t checked = 0, excluded = 0, violations = 0, witness = UINT64_MAX;
        for (uint64_t w = s; w < v.nw; w += slices) {
            uint64_t ok = 0, bad = 0;
            tr_field_relation_word(&v, &rel, mode, w, &ok, &bad);
            checked += (uint64_t)tr_field_popc(ok);
            excluded += (uint64_t)tr_field_popc(v.selection[w] & ~ok);
            violations += (uint64_t)tr_field_popc(bad);
            if (bad && witness == UINT64_MAX) witness = w * 64 + (uint64_t)tr_field_ctz(bad);
        }
        if (checked) tr_atomic_add_u64(&acc[4 * r], checked);
        if (excluded) tr_atomic_add_u64(&acc[4 * r + 1], excluded);
        if (violations) { tr_atomic_add_u64(&acc[4 * r + 2], violations); tr_atomic_min_u64(&acc[4 * r + 3], witness); }
    }
}

/* the bits of a view on the device (pointers swapped), with the relation and accumulator buffers of up to cap
 * relations */
typedef struct {
    tr_field_bits_view d;
    tr_field_relation *rels;
    uint64_t *acc;
    int32_t cap, mode;
    void *mem[5];
    int nmem;
} tr_field_device;

static uint64_t tr_field_slices(uint64_t nw, int32_t n) {
    uint64_t q = n > 0 ? (uint64_t)n : 1, s = TR_FIELD_WORK / q, cap = (nw + TR_FIELD_MIN_WORDS - 1) / TR_FIELD_MIN_WORDS;
    if (s > cap) s = cap;
    return s ? s : 1;
}

static void tr_field_device_free(tr_field_device *g) {
    for (int i = 0; i < g->nmem; i++) tr_cuda_free(g->mem[i]);
    g->nmem = 0;
}

static int tr_field_device_copy(const tr_field_bits_view *v, int32_t cap, int32_t mode, tr_field_device *g) {
    int rc = 0;
    memset(g, 0, sizeof *g);
    g->d = *v; g->cap = cap; g->mode = mode;
    uint64_t *sel = NULL, *value = NULL, *valid = NULL, words = (uint64_t)v->nvars * v->nw;
#define TR_FIELD_DEV(ptr, bytes) do { if (!rc) { rc = tr_cuda_alloc((void **)&(ptr), (size_t)(bytes)); if (!rc) g->mem[g->nmem++] = (void *)(ptr); } } while (0)
    TR_FIELD_DEV(sel, (v->nw ? v->nw : 1) * 8);
    if (!rc && v->nw) rc = tr_cuda_copy_in(sel, v->selection, (size_t)(v->nw * 8));
    TR_FIELD_DEV(value, (words ? words : 1) * 8);
    if (!rc && words) rc = tr_cuda_copy_in(value, v->value, (size_t)(words * 8));
    if (v->valid) {
        TR_FIELD_DEV(valid, (words ? words : 1) * 8);
        if (!rc && words) rc = tr_cuda_copy_in(valid, v->valid, (size_t)(words * 8));
    }
    TR_FIELD_DEV(g->rels, (uint64_t)(cap > 0 ? cap : 1) * sizeof(tr_field_relation));
    TR_FIELD_DEV(g->acc, (uint64_t)(cap > 0 ? cap : 1) * 32);
#undef TR_FIELD_DEV
    g->d.selection = sel; g->d.value = value; g->d.valid = valid;
    if (rc) tr_field_device_free(g);
    return rc;
}

/* n <= g->cap relations over the device bits */
static int tr_field_device_verify(tr_field_device *g, const tr_field_relation *rels, int32_t n, tr_field_evidence *out) {
    if (n <= 0) return 0;
    uint64_t *acc = (uint64_t *)calloc((size_t)n, 32);
    if (!acc) return -2;                                                                       /* TR_MATH_MEMORY */
    for (int32_t i = 0; i < n; i++) acc[4 * i + 3] = UINT64_MAX;
    int rc = tr_cuda_copy_in(g->rels, rels, (size_t)n * sizeof(tr_field_relation));
    if (!rc) rc = tr_cuda_copy_in(g->acc, acc, (size_t)n * 32);
    const uint64_t slices = tr_field_slices(g->d.nw, n), items = (uint64_t)n * slices;
    if (!rc && g->d.nw) { TR_LAUNCH(tr_field_verify_kernel, items, g->d, g->rels, g->mode, items, slices, g->acc); rc = tr_cuda_sync(); }
    if (!rc) rc = tr_cuda_copy_out(acc, g->acc, (size_t)n * 32);
    if (!rc)
        for (int32_t i = 0; i < n; i++) {
            tr_field_evidence *e = &out[i];
            memset(e, 0, sizeof *e);
            e->checked = acc[4 * i]; e->excluded = acc[4 * i + 1]; e->violations = acc[4 * i + 2];
            e->witness = e->violations ? acc[4 * i + 3] : TR_FIELD_NO_ROW;
            e->holds = !e->violations && !e->excluded;
        }
    free(acc);
    return rc;
}

#ifdef __cplusplus
extern "C" {
#endif
int tr_mgpu_field_syndromes(const tr_field_matrix *h, const tr_field_matrix *x, tr_field_matrix *out,
                            const tr_math_budget *budget, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    int rc = tr_field_check_matrix(h);
    if (!rc) rc = tr_field_check_matrix(x);
    if (rc) return rc;
    if (x->cols != h->cols) return -1;
    rc = tr_field_check_output(out, x->rows, h->rows);
    if (rc) return rc;
    uint64_t ob = tr_field_matrix_bytes(out);
    if (tr_field_overlap(h->words, tr_field_matrix_bytes(h), out->words, ob) ||
        tr_field_overlap(x->words, tr_field_matrix_bytes(x), out->words, ob)) return -1;
    tr_math_budget dflt;
    if (!budget) { tr_budget_default(&dflt); budget = &dflt; }
    tr_math_need need;
    for (int i = 0; i < TR_BUDGET_RESOURCES; i++) need.need[i] = 0;
    need.need[TR_BUDGET_TENSOR] = tr_budget_add(tr_budget_mul(h->rows, h->cols), tr_budget_mul(x->rows, x->cols));
    need.need[TR_BUDGET_OUTPUT] = tr_budget_mul(x->rows, h->rows);
    need.need[TR_BUDGET_BYTES] = tr_field_syndromes_device_bytes(h, x);
    rc = tr_budget_check(budget, &need, refusal);
    if (rc) return rc;
    const uint64_t W = tr_field_words(h->cols), O = tr_field_words(h->rows), items = x->rows * O;
    for (uint64_t j = 0; j < out->rows && out->stride; j++) memset(out->words + j * out->stride, 0, (size_t)(out->stride * 8));
    if (!items) return 0;
    const uint64_t hw = h->rows * W, xw = x->rows * W;              /* compact copies: W words per row */
    uint64_t *hc = (uint64_t *)calloc((size_t)(hw > 0 ? hw : 1), 8);
    uint64_t *xc = (uint64_t *)calloc((size_t)(xw > 0 ? xw : 1), 8);
    uint64_t *oc = (uint64_t *)calloc((size_t)items, 8);
    if (!hc || !xc || !oc) { free(hc); free(xc); free(oc); return -2; }                     /* TR_MATH_MEMORY */
    for (uint64_t i = 0; i < h->rows && W; i++) memcpy(hc + i * W, h->words + i * h->stride, (size_t)(W * 8));
    for (uint64_t i = 0; i < x->rows && W; i++) memcpy(xc + i * W, x->words + i * x->stride, (size_t)(W * 8));
    uint64_t *dh = NULL, *dx = NULL, *dout = NULL;
    rc = tr_cuda_alloc((void **)&dh, (size_t)(hw > 0 ? hw : 1) * 8);
    if (!rc) rc = tr_cuda_alloc((void **)&dx, (size_t)(xw > 0 ? xw : 1) * 8);
    if (!rc) rc = tr_cuda_alloc((void **)&dout, (size_t)items * 8);
    if (!rc) rc = tr_cuda_copy_in(dh, hc, (size_t)(hw * 8));
    if (!rc) rc = tr_cuda_copy_in(dx, xc, (size_t)(xw * 8));
    tr_field_matrix dhm = *h, dxm = *x;
    dhm.words = dh; dhm.stride = W; dxm.words = dx; dxm.stride = W;
    if (!rc) { TR_LAUNCH(tr_field_syndromes_kernel, items, dhm, dxm, dout, items, O); rc = tr_cuda_sync(); }
    if (!rc) rc = tr_cuda_copy_out(oc, dout, (size_t)items * 8);
    if (!rc) for (uint64_t j = 0; j < x->rows; j++) memcpy(out->words + j * out->stride, oc + j * O, (size_t)(O * 8));
    tr_cuda_free(dh); tr_cuda_free(dx); tr_cuda_free(dout);
    free(hc); free(xc); free(oc);
    return rc;
}

int tr_mgpu_field_verify(const tr_field_bits_view *v, const tr_field_relation *rels, int32_t n, int32_t mode,
                         const tr_math_budget *budget, tr_field_evidence *out, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    int rc = tr_field_check_bits(v);
    if (!rc) rc = tr_field_check_relations(v, rels, n);
    if (rc) return rc;
    if ((mode != TR_FIELD_SQL && mode != TR_FIELD_LISTWISE) || (n && !out)) return -1;
    tr_math_budget dflt;
    if (!budget) { tr_budget_default(&dflt); budget = &dflt; }
    tr_math_need need;
    for (int i = 0; i < TR_BUDGET_RESOURCES; i++) need.need[i] = 0;
    need.need[TR_BUDGET_OUTPUT] = (uint64_t)n;
    need.need[TR_BUDGET_BYTES] = tr_field_verify_device_bytes(v, n);
    rc = tr_budget_check(budget, &need, refusal);
    if (rc || !n) return rc;
    tr_field_device g;
    rc = tr_field_device_copy(v, n, mode, &g);
    if (!rc) rc = tr_field_device_verify(&g, rels, n, out);
    tr_field_device_free(&g);
    return rc;
}

static int32_t tr_field_device_cb(void *ctx, const tr_field_relation *rels, int32_t n, tr_field_evidence *out) {
    return tr_field_device_verify((tr_field_device *)ctx, rels, n, out);
}

int tr_mgpu_field_discover(const tr_field_bits_view *v, const tr_field_discover_spec *s, const tr_math_budget *budget,
                           tr_field_relation *found, tr_field_evidence *found_ev, int32_t found_cap,
                           tr_field_relation *rejected, tr_field_evidence *rejected_ev, int32_t rejected_cap,
                           tr_field_discovery *stats, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    tr_math_need need;
    int rc = tr_field_discover_need(v, s, 1, &need);
    if (rc) return rc;
    tr_math_budget dflt;
    if (!budget) { tr_budget_default(&dflt); budget = &dflt; }
    rc = tr_budget_check(budget, &need, refusal);
    if (rc) return rc;
    int32_t columns = tr_field_discover_columns(v, s);
    if (columns > 64) return TR_MATH_UNSUPPORTED;
    tr_field_device g;
    rc = tr_field_device_copy(v, columns > 0 ? columns : 1, TR_FIELD_LISTWISE, &g);
    if (!rc) rc = tr_field_discover_run(v, s, tr_field_device_cb, &g, found, found_ev, found_cap, rejected,
                                        rejected_ev, rejected_cap, stats);
    tr_field_device_free(&g);
    return rc;
}
#ifdef __cplusplus
}
#endif
