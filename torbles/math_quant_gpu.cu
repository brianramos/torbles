/* QUANT-001 / QUANT-002 on a CUDA device: the per-group quantifier scan, the per-group category masks and the coverage
 * test, with exactly the CPU's states (math_quant.h).
 * SPDX-License-Identifier: Apache-2.0
 *
 * Work items of the scans are slices of consecutive 64-row words (item s: words s L .. s L + L - 1). A QUANT-001 item
 * owns a scratch slot (2 k condition words and 2 words per program node), evaluates each word like the CPU
 * (tr_quant_predicate_word: the conditions the root reads, then BOOL-001's word evaluator), and keeps a local run of
 * one group: the counts of a word whose selected rows share one group are added with popcounts, a mixed word row by
 * row; the run is added to the group's counters with 64-bit atomics when the group changes and at the end (contiguous
 * groups: a few atomics per item; scattered groups: about one per row). A QUANT-002 item does the same for rows and
 * NULL rows and keeps the pending bits of one mask word, combined into the group's mask with atomic OR when the word
 * changes (OR is idempotent: a duplicated category sets the same bit). The coverage test is one work item per group
 * (AND-NOT and popcount against the required mask) writing its own output slots; parent rows have their own scan.
 * Integer atomics commute and OR is idempotent, so the states do not depend on thread order. */
#include "math_cuda.h"
#include "math_quant.h"
#include "math_handles.h"

TR_MGPU_ABI(tr_quant_counts, TR_QUANT_ABI);          /* structs passed to the CUDA library (math_handles.h) */
TR_MGPU_ABI(tr_quant_cover_arrays, TR_QUANT_ABI);

TR_DEVICE void tr_quant_flush(uint64_t *a, uint64_t *b, uint64_t *c, uint64_t *d, uint32_t g, uint64_t x, uint64_t y,
                              uint64_t z, uint64_t t) {
    if (x) tr_atomic_add_u64(&a[g], x);
    if (y) tr_atomic_add_u64(&b[g], y);
    if (z && c) tr_atomic_add_u64(&c[g], z);
    if (t && d) tr_atomic_add_u64(&d[g], t);
}

TR_KERNEL void tr_quant_groups_kernel(tr_scope_view v, const tr_bool_condition *conds, tr_bool_program_view p,
                                      uint64_t reads, uint64_t items, uint64_t per, uint64_t stride, uint64_t *scratch,
                                      uint64_t *rows, uint64_t *qual, uint64_t *fail, uint64_t *unk) {
    TR_GRID_LOOP(item, items) {
        const uint64_t nw = (v.rows + 63) / 64, first = item * per, last = first + per < nw ? first + per : nw;
        const uint64_t k = (uint64_t)p.k;
        uint64_t *cv = scratch + item * stride, *ck = cv + k, *val = cv + 2 * k;
        uint32_t cur = 0;
        uint64_t cr = 0, cq = 0, cf = 0, cu = 0;            /* the local run of group cur */
        for (uint64_t w = first; w < last; w++) {
            const uint64_t sel = v.selection[w];
            if (!sel) continue;
            uint64_t T, F, U;
            tr_quant_predicate_word(&v, conds, &p, reads, w, cv, ck, val, &T, &F, &U);
            int uniform;
            const uint32_t g = tr_quant_word_group(v.groups, w, sel, &uniform);
            if (uniform) {
                if (cr && g != cur) { tr_quant_flush(rows, qual, fail, unk, cur, cr, cq, cf, cu); cr = cq = cf = cu = 0; }
                cur = g;
                cr += (uint64_t)tr_quant_popc(sel); cq += (uint64_t)tr_quant_popc(T);
                cf += (uint64_t)tr_quant_popc(F); cu += (uint64_t)tr_quant_popc(U);
                continue;
            }
            for (uint64_t bits = sel; bits; bits &= bits - 1) {
                const int b = tr_quant_ctz(bits);
                const uint64_t bit = (uint64_t)1 << b;
                const uint32_t h = v.groups[w * 64 + (uint64_t)b];
                if (cr && h != cur) { tr_quant_flush(rows, qual, fail, unk, cur, cr, cq, cf, cu); cr = cq = cf = cu = 0; }
                cur = h; cr++;
                if (T & bit) cq++; else if (F & bit) cf++; else cu++;
            }
        }
        if (cr) tr_quant_flush(rows, qual, fail, unk, cur, cr, cq, cf, cu);
    }
}

TR_KERNEL void tr_quant_parents_kernel(tr_scope_view v, uint64_t items, uint64_t per, uint64_t *parents) {
    TR_GRID_LOOP(item, items) {
        const uint64_t nw = (v.rows + 63) / 64, first = item * per, last = first + per < nw ? first + per : nw;
        uint32_t cur = 0;
        uint64_t c = 0;
        for (uint64_t w = first; w < last; w++)
            for (uint64_t bits = v.selection[w]; bits; bits &= bits - 1) {
                const uint64_t r = w * 64 + (uint64_t)tr_quant_ctz(bits);
                const uint32_t g = v.groups ? v.groups[r] : 0;
                if (c && g != cur) { tr_atomic_add_u64(&parents[cur], c); c = 0; }
                cur = g; c++;
            }
        if (c) tr_atomic_add_u64(&parents[cur], c);
    }
}

TR_KERNEL void tr_quant_cover_kernel(tr_scope_view v, const uint32_t *cats, uint32_t null_code, uint64_t W,
                                     uint64_t items, uint64_t per, uint64_t *rows, uint64_t *nulls, uint64_t *masks) {
    TR_GRID_LOOP(item, items) {
        const uint64_t nw = (v.rows + 63) / 64, first = item * per, last = first + per < nw ? first + per : nw;
        uint32_t cur = 0;
        uint64_t cr = 0, cn = 0;                            /* the local run of group cur: rows, NULL rows */
        uint64_t mw = 0, mbits = 0;                         /* pending bits of mask word mw */
        for (uint64_t w = first; w < last; w++)
            for (uint64_t bits = v.selection[w]; bits; bits &= bits - 1) {
                const uint64_t r = w * 64 + (uint64_t)tr_quant_ctz(bits);
                const uint32_t g = v.groups ? v.groups[r] : 0, c = cats ? cats[r] : 0;
                if (cr && g != cur) { tr_quant_flush(rows, nulls, NULL, NULL, cur, cr, cn, 0, 0); cr = cn = 0; }
                cur = g; cr++;
                if (c == null_code) { cn++; continue; }
                const uint64_t at = (uint64_t)g * W + (c >> 6);
                if (mbits && at != mw) { tr_atomic_or_u64(&masks[mw], mbits); mbits = 0; }
                mw = at;
                mbits |= (uint64_t)1 << (c & 63);
            }
        if (cr) tr_quant_flush(rows, nulls, NULL, NULL, cur, cr, cn, 0, 0);
        if (mbits) tr_atomic_or_u64(&masks[mw], mbits);
    }
}

TR_KERNEL void tr_quant_test_kernel(const uint64_t *masks, uint64_t G, uint64_t W, const uint64_t *req, uint64_t cm,
                                    uint64_t *distinct, uint64_t *matched, uint64_t *missing, uint64_t *extra) {
    TR_GRID_LOOP(g, G) {
        uint64_t d = 0, m = 0, lack = 0, more = 0;
        const uint64_t *row = masks + g * W;
        for (uint64_t w = 0; w < W; w++) {                   /* AND-NOT and popcount against the required mask */
            const uint64_t x = row[w], r = req[w];
            d += (uint64_t)tr_quant_popc(x); m += (uint64_t)tr_quant_popc(x & r);
            lack += (uint64_t)tr_quant_popc(r & ~x); more += (uint64_t)tr_quant_popc(x & ~r);
        }
        distinct[g] = d; matched[g] = m; missing[g] = lack + cm; extra[g] = more;
    }
}

/* the required mask holds exactly required_codes codes, all below D: 0 or TR_MATH_INVALID */
static int tr_quant_mask_ok(const uint64_t *mask, uint64_t D, uint64_t required_codes) {
    const uint64_t W = (D + 63) / 64;
    uint64_t n = 0;
    for (uint64_t w = 0; w < W; w++) n += (uint64_t)__builtin_popcountll(mask[w]);
    if (W && D % 64 && (mask[W - 1] >> (D % 64))) return -1;
    return n == required_codes ? 0 : -1;
}
/* consistent views (the checks the kernels rely on): 0 or TR_MATH_INVALID */
static int tr_quant_view_ok(const tr_scope_view *v) {
    if (!v || v->ncols < 0 || v->ncols > TR_SCOPE_MAX_COLUMNS || v->ngroups < 1 || (v->rows && !v->selection) ||
        (v->rows && v->ngroups > 1 && !v->groups)) return -1;
    return 0;
}

/* device memory of one call: every allocation recorded for one release */
typedef struct { void *mem[3 * TR_SCOPE_MAX_COLUMNS + 32]; int n; int rc; } tr_quant_dev;
static void *tr_quant_dev_alloc(tr_quant_dev *m, uint64_t bytes) {
    void *p = NULL;
    if (m->rc) return NULL;
    m->rc = tr_cuda_alloc(&p, (size_t)bytes);
    if (!m->rc) m->mem[m->n++] = p;
    return m->rc ? NULL : p;
}
static void *tr_quant_dev_copy(tr_quant_dev *m, const void *src, uint64_t bytes) {
    void *p = tr_quant_dev_alloc(m, bytes);
    if (!m->rc && bytes) m->rc = tr_cuda_copy_in(p, src, (size_t)bytes);
    return p;
}
static void tr_quant_dev_free(tr_quant_dev *m) { for (int i = 0; i < m->n; i++) tr_cuda_free(m->mem[i]); m->n = 0; }
/* a device copy of a view: its selection, its group ids and the columns of cols (codes or values, validity) */
static tr_scope_view tr_quant_dev_view(tr_quant_dev *m, const tr_scope_view *v, uint64_t cols) {
    tr_scope_view d = *v;
    const uint64_t nw = (v->rows + 63) / 64;
    d.selection = (const uint64_t *)tr_quant_dev_copy(m, v->selection, nw * 8);
    d.groups = v->groups ? (const uint32_t *)tr_quant_dev_copy(m, v->groups, v->rows * 4) : NULL;
    for (int32_t c = 0; c < v->ncols; c++) {
        tr_scope_view_column *k = &d.col[c];
        if (!((cols >> c) & 1)) { k->codes = NULL; k->values = NULL; k->valid = NULL; continue; }   /* never read */
        if (v->col[c].values) k->values = (const int64_t *)tr_quant_dev_copy(m, v->col[c].values, v->rows * 8);
        else k->codes = (const uint32_t *)tr_quant_dev_copy(m, v->col[c].codes, v->rows * 4);
        if (v->col[c].valid) k->valid = (const uint8_t *)tr_quant_dev_copy(m, v->col[c].valid, v->rows);
    }
    return d;
}
/* launches the parent scan into parents (device, G words) */
static void tr_quant_dev_parents(tr_quant_dev *m, const tr_scope_view *parents, uint64_t *dparents) {
    if (m->rc || !parents || !parents->selected) return;
    tr_scope_view d = tr_quant_dev_view(m, parents, 0);
    if (m->rc) return;
    const uint64_t nw = (parents->rows + 63) / 64, items = tr_quant_items(nw, 0), per = (nw + items - 1) / items;
    TR_LAUNCH(tr_quant_parents_kernel, items, d, items, per, dparents);
}

#ifdef __cplusplus
extern "C" {
#endif
int tr_mgpu_quant_groups(const tr_scope_view *child, const tr_bool_condition *conds, int32_t k,
                         const tr_bool_program_view *pred, const tr_scope_view *parents, const tr_math_budget *budget,
                         tr_quant_counts *out, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (tr_quant_view_ok(child) || (parents && tr_quant_view_ok(parents)) || k < 0 || (k && !conds)) return -1;
    int rc = tr_bool_program_check(pred);
    if (rc) return rc;
    if (pred->k != k) return -6;                                                         /* TR_MATH_DOMAIN_ERROR */
    if (parents && parents->ngroups > child->ngroups) return -6;
    for (int32_t i = 0; i < k; i++) if (tr_bool_condition_check(child, &conds[i])) return -1;
    const uint64_t G = child->ngroups;
    if (!out || out->ngroups != G || !out->rows || !out->qualifying || !out->failing || !out->unknown || !out->parents)
        return -1;                                                                       /* TR_MATH_INVALID */
    tr_math_budget dflt;
    if (!budget) { tr_budget_default(&dflt); budget = &dflt; }
    tr_math_need need;
    for (int i = 0; i < TR_BUDGET_RESOURCES; i++) need.need[i] = 0;
    need.need[TR_BUDGET_VARIABLES] = (uint64_t)k;
    need.need[TR_BUDGET_CIRCUIT] = (uint64_t)pred->nnodes;
    need.need[TR_BUDGET_STATES] = G;
    need.need[TR_BUDGET_OUTPUT] = G;
    need.need[TR_BUDGET_BYTES] = tr_quant_groups_device_bytes(child, conds, k, pred, parents);
    rc = tr_budget_check(budget, &need, refusal);
    if (rc) return rc;
    uint64_t *host[5] = {out->rows, out->qualifying, out->failing, out->unknown, out->parents};
    for (int a = 0; a < 5; a++) memset(host[a], 0, (size_t)(G * 8));
    if (!child->selected && (!parents || !parents->selected)) return 0;   /* nothing selected: every count is zero */
    const uint64_t reads = pred->deps[pred->root], nw = (child->rows + 63) / 64;
    const uint64_t stride = tr_quant_item_words(k, pred->nnodes), items = tr_quant_items(nw, stride * 8);
    const uint64_t per = (nw + items - 1) / items;
    tr_quant_dev m;
    m.n = 0; m.rc = 0;
    uint64_t *arr = (uint64_t *)tr_quant_dev_alloc(&m, G * 40);
    if (child->selected) {
        tr_scope_view d = tr_quant_dev_view(&m, child, tr_quant_condition_columns(conds, k));
        const tr_bool_condition *dconds = (const tr_bool_condition *)tr_quant_dev_copy(&m, conds, (uint64_t)k * sizeof(tr_bool_condition));
        tr_bool_program_view p = *pred;
        p.nodes = (const tr_bool_node *)tr_quant_dev_copy(&m, pred->nodes, (uint64_t)pred->nnodes * sizeof(tr_bool_node));
        p.schedule = (const uint32_t *)tr_quant_dev_copy(&m, pred->schedule, (uint64_t)pred->nsched * 4);
        p.deps = NULL;                                       /* never read: the root's conditions are `reads` */
        uint64_t *scratch = (uint64_t *)tr_quant_dev_alloc(&m, items * stride * 8);
        if (!m.rc) TR_LAUNCH(tr_quant_groups_kernel, items, d, dconds, p, reads, items, per, stride, scratch, arr, arr + G,
                             arr + 2 * G, arr + 3 * G);
    }
    if (!m.rc) tr_quant_dev_parents(&m, parents, arr + 4 * G);
    rc = m.rc ? m.rc : tr_cuda_sync();
    for (int a = 0; a < 5 && !rc; a++) rc = tr_cuda_copy_out(host[a], arr + (uint64_t)a * G, (size_t)(G * 8));
    tr_quant_dev_free(&m);
    return rc;
}

int tr_mgpu_quant_cover(const tr_scope_view *child, const tr_scope_view *categories, uint32_t null_code,
                        const uint64_t *required, uint64_t required_codes, uint64_t constant_missing,
                        const tr_scope_view *parents, const tr_math_budget *budget, tr_quant_cover_arrays *out,
                        tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (tr_quant_view_ok(child) || tr_quant_view_ok(categories) || (parents && tr_quant_view_ok(parents)) ||
        categories->rows != child->rows) return -1;                                      /* TR_MATH_INVALID */
    if (parents && parents->ngroups > child->ngroups) return -6;                         /* TR_MATH_DOMAIN_ERROR */
    const uint64_t G = child->ngroups, D = categories->ngroups, W = (D + 63) / 64;
    if ((null_code != TR_QUANT_NO_NULL && null_code >= D) || (W && !required) || tr_quant_mask_ok(required, D, required_codes))
        return -1;
    if (!out || out->ngroups != G || out->size != D || out->words != W || !out->rows || !out->nulls || !out->parents ||
        !out->distinct || !out->matched || !out->missing || !out->extra || (G && W && !out->masks)) return -1;
    tr_math_budget dflt;
    if (!budget) { tr_budget_default(&dflt); budget = &dflt; }
    tr_math_need need;
    for (int i = 0; i < TR_BUDGET_RESOURCES; i++) need.need[i] = 0;
    need.need[TR_BUDGET_SUPPORT] = D;
    need.need[TR_BUDGET_STATES] = tr_budget_mul(G, D);
    need.need[TR_BUDGET_OUTPUT] = G;
    need.need[TR_BUDGET_BYTES] = tr_quant_cover_device_bytes(child, categories, parents);
    int rc = tr_budget_check(budget, &need, refusal);
    if (rc) return rc;
    tr_quant_dev m;
    m.n = 0; m.rc = 0;
    uint64_t *arr = (uint64_t *)tr_quant_dev_alloc(&m, G * 56);   /* rows, nulls, parents, distinct, matched, missing, extra */
    uint64_t *dmasks = (uint64_t *)tr_quant_dev_alloc(&m, G * W * 8);
    const uint64_t *dreq = (const uint64_t *)tr_quant_dev_copy(&m, required, W * 8);
    if (!m.rc && child->selected) {
        tr_scope_view d = tr_quant_dev_view(&m, child, 0);
        const uint32_t *dcats = categories->groups ? (const uint32_t *)tr_quant_dev_copy(&m, categories->groups, categories->rows * 4) : NULL;
        const uint64_t nw = (child->rows + 63) / 64, items = tr_quant_items(nw, 0), per = (nw + items - 1) / items;
        if (!m.rc) TR_LAUNCH(tr_quant_cover_kernel, items, d, dcats, null_code, W, items, per, arr, arr + G, dmasks);
    }
    if (!m.rc) tr_quant_dev_parents(&m, parents, arr + 2 * G);
    if (!m.rc) TR_LAUNCH(tr_quant_test_kernel, G, dmasks, G, W, dreq, constant_missing, arr + 3 * G, arr + 4 * G,
                         arr + 5 * G, arr + 6 * G);
    rc = m.rc ? m.rc : tr_cuda_sync();
    uint64_t *host[7] = {out->rows, out->nulls, out->parents, out->distinct, out->matched, out->missing, out->extra};
    for (int a = 0; a < 7 && !rc; a++) rc = tr_cuda_copy_out(host[a], arr + (uint64_t)a * G, (size_t)(G * 8));
    if (!rc && G && W) rc = tr_cuda_copy_out(out->masks, dmasks, (size_t)(G * W * 8));
    tr_quant_dev_free(&m);
    return rc;
}

int tr_mgpu_quant_cover_test(const tr_quant_cover_arrays *in, const uint64_t *required, uint64_t required_codes,
                             uint64_t constant_missing, const tr_math_budget *budget, tr_quant_cover_arrays *out,
                             tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!in || !out || in->words != (in->size + 63) / 64 || (in->ngroups && in->words && !in->masks) ||
        (in->words && !required) || (in->words && tr_quant_mask_ok(required, in->size, required_codes)) ||
        out->ngroups != in->ngroups ||
        out->size != in->size || out->words != in->words || !out->distinct || !out->matched || !out->missing ||
        !out->extra) return -1;                                                          /* TR_MATH_INVALID */
    const uint64_t G = in->ngroups, W = in->words;
    tr_math_budget dflt;
    if (!budget) { tr_budget_default(&dflt); budget = &dflt; }
    tr_math_need need;
    for (int i = 0; i < TR_BUDGET_RESOURCES; i++) need.need[i] = 0;
    need.need[TR_BUDGET_SUPPORT] = in->size;
    need.need[TR_BUDGET_OUTPUT] = G;
    need.need[TR_BUDGET_BYTES] = tr_quant_test_device_bytes(G, in->size);
    int rc = tr_budget_check(budget, &need, refusal);
    if (rc) return rc;
    tr_quant_dev m;
    m.n = 0; m.rc = 0;
    const uint64_t *dmasks = (const uint64_t *)tr_quant_dev_copy(&m, in->masks, G * W * 8);
    const uint64_t *dreq = (const uint64_t *)tr_quant_dev_copy(&m, required, W * 8);
    uint64_t *arr = (uint64_t *)tr_quant_dev_alloc(&m, G * 32);
    if (!m.rc) TR_LAUNCH(tr_quant_test_kernel, G, dmasks, G, W, dreq, constant_missing, arr, arr + G, arr + 2 * G,
                         arr + 3 * G);
    rc = m.rc ? m.rc : tr_cuda_sync();
    uint64_t *host[4] = {out->distinct, out->matched, out->missing, out->extra};
    for (int a = 0; a < 4 && !rc; a++) rc = tr_cuda_copy_out(host[a], arr + (uint64_t)a * G, (size_t)(G * 8));
    tr_quant_dev_free(&m);
    return rc;
}
#ifdef __cplusplus
}
#endif
