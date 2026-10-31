/* DIST-001 / DIST-002: histograms over a data scope and prefix states of a sorted support; see math_dist.h.
 * SPDX-License-Identifier: Apache-2.0 */
#include "math_dist.h"
#include "mathematics.h"
#include "math_domain.h"
#include "math_handles.h"
#include <stdlib.h>
#include <string.h>

TR_MATH_ABI(tr_dist_spec, 1);              /* the ctypes boundary of DIST-001 / DIST-002 (math_handles.h) */
TR_MATH_ABI(tr_dist_states, 1);
TR_MATH_ABI(tr_dist_hist_info, 1);
TR_MATH_ABI(tr_dist_prefix_view, 1);
TR_MATH_ABI(tr_dist_prefix_info, 1);
TR_MATH_ABI(tr_dist_request, 1);
TR_MATH_ABI(tr_dist_result, 1);

_Static_assert(sizeof(tr_dist_spec) == 96, "tr_dist_spec ABI");
_Static_assert(sizeof(tr_dist_states) == 192, "tr_dist_states ABI");
_Static_assert(sizeof(tr_dist_hist_info) == 232, "tr_dist_hist_info ABI");
_Static_assert(sizeof(tr_dist_prefix_view) == 72, "tr_dist_prefix_view ABI");
_Static_assert(sizeof(tr_dist_prefix_info) == 96, "tr_dist_prefix_info ABI");
_Static_assert(sizeof(tr_dist_request) == 64, "tr_dist_request ABI");
_Static_assert(sizeof(tr_dist_result) == 104, "tr_dist_result ABI");
_Static_assert(sizeof(tr_dist_i256) == 32, "tr_dist_i256 ABI");

#define HIST_MAGIC UINT32_C(0x54534944)
#define BUILDER_MAGIC UINT32_C(0x42545344)
#define PREFIX_MAGIC UINT32_C(0x50545344)
#define I128_LIMIT ((((unsigned __int128)1) << 127) - 1)

/* one row (SORT) or one cell (derived sparse states) while sorting: 64 bytes */
typedef struct {
    uint32_t g;
    uint8_t nulls, pad[3];
    int64_t v[3];
    uint64_t count, wcount;
    tr_math_i128 wsum;
} dist_entry;

struct tr_dist_hist_s {
    uint32_t magic;
    int32_t k, repr, weighted, scoped, strategy, wnonneg, wscale;
    int32_t col[3], weight, scale[3], nullslot[3];
    uint64_t ngroups, ncells, cells, occupied, present, rows_total, max_cells, bytes, scans, passes;
    uint64_t digest[2];
    int64_t lo[3];
    uint64_t span[3], dim[3];
    uint64_t *rows, *offset;
    int64_t *value[3];
    uint8_t *nulls;
    uint64_t *count, *wcount;
    tr_math_i128 *wsum;
};

struct tr_dist_hist_builder_s {
    uint32_t magic;
    int32_t strategy, spent, reserved;
    tr_dist_spec spec;
    tr_dist_plan plan;
    const tr_scope *scope;
    tr_scope_view view;
    uint64_t digest[2];
    uint64_t G, total;                 /* groups; G x cells (DIRECT, COUNT) */
    uint64_t *rows, *count, *wcount;
    tr_math_i128 *wsum;
    dist_entry *entries;               /* SORT: one per accumulated selected row */
    uint64_t nentries, capacity;
    uint64_t *covered, ncovered;
    tr_math_need need;                 /* what init checked */
    tr_math_budget budget;
};

struct tr_dist_prefix_s {
    uint32_t magic;
    int32_t degree, mass, scale, wscale, scoped;
    uint64_t ngroups, nsupport, max_support, present, rows_total, nulls_total, bytes;
    uint64_t digest[2];
    uint64_t *rows, *nulls, *offset;
    int64_t *value;
    tr_math_i128 *prefix;
};

/* ------------------------------------------------------------------ small helpers */
static void set_error(int32_t *error, int32_t code) { if (error) *error = code; }
static void *zalloc(uint64_t bytes) {
    if (bytes >= (uint64_t)SIZE_MAX) return NULL;
    return calloc(1, bytes ? (size_t)bytes : 1);
}
static void copy(void *dst, const void *src, uint64_t bytes) { if (bytes) memcpy(dst, src, (size_t)bytes); }
static const tr_math_budget *budget_of(const tr_math_budget *b, tr_math_budget *dflt) {
    if (b) return b;
    tr_budget_default(dflt);
    return dflt;
}
static int hist_valid(const tr_dist_hist *h) { return h && h->magic == HIST_MAGIC; }
static int builder_valid(const tr_dist_hist_builder *b) { return b && b->magic == BUILDER_MAGIC; }
static int prefix_valid(const tr_dist_prefix *p) { return p && p->magic == PREFIX_MAGIC; }
static void need_zero(tr_math_need *n) { for (int i = 0; i < TR_BUDGET_RESOURCES; i++) n->need[i] = 0; }
/* tr_budget_check, and a size that does not fit 64 bits is refused even by an unlimited budget */
static int32_t check_need(const tr_math_budget *b, const tr_math_need *n, tr_math_refusal *r) {
    int32_t rc = tr_budget_check(b, n, r);
    if (rc) return rc;
    for (int32_t i = 0; i < TR_BUDGET_RESOURCES; i++)
        if (n->need[i] == TR_BUDGET_UNKNOWN) {
            if (r) { r->resource = i; r->refused = 1; r->need = n->need[i]; r->limit = b->limit[i]; }
            return TR_MATH_BUDGET;
        }
    return TR_MATH_OK;
}
static uint64_t range_mask(uint64_t w, uint64_t begin, uint64_t end) {   /* the bits of word w inside [begin, end) */
    uint64_t lo = w * 64, a = begin > lo ? begin - lo : 0, z = end - lo < 64 ? end - lo : 64;
    uint64_t m = z >= 64 ? UINT64_MAX : (UINT64_C(1) << z) - 1;
    return a ? m & ~((UINT64_C(1) << a) - 1) : m;
}
static int add_i128(tr_math_i128 *acc, tr_math_i128 x) {     /* 1, or 0 on overflow (acc unchanged) */
    __int128 r;
    if (__builtin_add_overflow(tr_i128_get(*acc), tr_i128_get(x), &r)) return 0;
    *acc = tr_i128_set(r);
    return 1;
}

/* ------------------------------------------------------------------ the histogram object */
static uint64_t hist_bytes(uint64_t G, int repr, int k, int weighted, uint64_t ncells) {
    uint64_t b = tr_budget_add(sizeof(tr_dist_hist), tr_budget_mul(G, 8));
    uint64_t per = tr_dist_cell_bytes(weighted);
    if (repr == TR_DIST_SPARSE) {
        b = tr_budget_add(b, tr_budget_mul(tr_budget_add(G, 1), 8));
        per = tr_budget_add(per, tr_budget_add(tr_budget_mul((uint64_t)k, 8), 1));
    }
    return tr_budget_add(b, tr_budget_mul(ncells, per));
}
static void hist_release(tr_dist_hist *h) {
    if (!h) return;
    free(h->rows); free(h->offset); free(h->nulls); free(h->count); free(h->wcount); free(h->wsum);
    for (int c = 0; c < 3; c++) free(h->value[c]);
    h->magic = 0;
    free(h);
}
void tr_dist_hist_free(tr_dist_hist *h) { if (hist_valid(h)) hist_release(h); }

/* a zeroed histogram of this shape (dense: ncells = G x cells) */
static tr_dist_hist *hist_alloc(uint64_t G, int repr, int k, int weighted, uint64_t ncells) {
    tr_dist_hist *h = zalloc(sizeof *h);
    if (!h) return NULL;
    h->ngroups = G; h->repr = repr; h->k = k; h->weighted = weighted; h->ncells = ncells;
    h->weight = -1;
    for (int c = 0; c < 3; c++) h->col[c] = -1;
    h->rows = zalloc(tr_budget_mul(G, 8));
    h->count = zalloc(tr_budget_mul(ncells, 8));
    int ok = h->rows && h->count;
    if (weighted) {
        h->wcount = zalloc(tr_budget_mul(ncells, 8)); h->wsum = zalloc(tr_budget_mul(ncells, 16));
        ok = ok && h->wcount && h->wsum;
    }
    if (repr == TR_DIST_SPARSE) {
        h->offset = zalloc(tr_budget_mul(tr_budget_add(G, 1), 8)); h->nulls = zalloc(ncells);
        ok = ok && h->offset && h->nulls;
        for (int c = 0; c < k; c++) { h->value[c] = zalloc(tr_budget_mul(ncells, 8)); ok = ok && h->value[c]; }
    }
    h->bytes = hist_bytes(G, repr, k, weighted, ncells);
    h->magic = HIST_MAGIC;
    if (!ok) { hist_release(h); return NULL; }
    return h;
}
/* the metadata of a derived histogram: copied from its source */
static void hist_copy_meta(tr_dist_hist *h, const tr_dist_hist *f) {
    h->scoped = f->scoped; h->digest[0] = f->digest[0]; h->digest[1] = f->digest[1];
    h->wnonneg = f->wnonneg; h->wscale = f->wscale; h->weight = f->weight;
    for (int c = 0; c < 3; c++) { h->col[c] = f->col[c]; h->scale[c] = f->scale[c]; }
}
static void hist_set_domain(tr_dist_hist *h, const tr_dist_plan *p) {
    h->cells = p->cells;
    for (int c = 0; c < 3; c++) {
        h->lo[c] = c < p->k ? p->lo[c] : 0; h->span[c] = c < p->k ? p->span[c] : 0;
        h->dim[c] = c < p->k ? p->dim[c] : 0; h->nullslot[c] = c < p->k ? p->nullslot[c] : 0;
    }
}
static void hist_plan(const tr_dist_hist *h, tr_dist_plan *p) {
    memset(p, 0, sizeof *p);
    p->k = h->k; p->repr = h->repr; p->weighted = h->weighted;
    uint64_t cells = 1;
    for (int c = h->k - 1; c >= 0; c--) {
        p->lo[c] = h->lo[c]; p->span[c] = h->span[c]; p->dim[c] = h->dim[c]; p->nullslot[c] = h->nullslot[c];
        p->stride[c] = cells; cells = tr_budget_mul(cells, h->dim[c]);
    }
    p->cells = h->cells;
}
/* present groups, selected rows, occupied cells, the largest group's cells */
static void hist_tally(tr_dist_hist *h) {
    uint64_t rows = 0, present = 0, occupied = 0, maxc = 0;
    for (uint64_t g = 0; g < h->ngroups; g++) { rows += h->rows[g]; present += h->rows[g] != 0; }
    if (h->repr == TR_DIST_DENSE) {
        for (uint64_t i = 0; i < h->ncells; i++) occupied += h->count[i] != 0;
        maxc = h->cells;
    } else {
        occupied = h->ncells;
        for (uint64_t g = 0; g < h->ngroups; g++) {
            uint64_t c = h->offset[g + 1] - h->offset[g];
            if (c > maxc) maxc = c;
        }
    }
    h->rows_total = rows; h->present = present; h->occupied = occupied; h->max_cells = maxc;
}
/* the cell at array position i (dense: of group g) as an entry */
static void hist_cell(const tr_dist_hist *h, const tr_dist_plan *p, uint64_t g, uint64_t i, dist_entry *e) {
    memset(e, 0, sizeof *e);
    e->g = (uint32_t)g;
    if (h->repr == TR_DIST_DENSE) {
        uint64_t idx = i - g * h->cells;
        for (int c = 0; c < h->k; c++) {
            int null;
            e->v[c] = tr_dist_plan_coord(p, idx, c, &null);
            if (null) e->nulls |= (uint8_t)(1u << c);
        }
    } else {
        e->nulls = h->nulls[i];
        for (int c = 0; c < h->k; c++) e->v[c] = h->value[c][i];
    }
    e->count = h->count[i];
    if (h->weighted) { e->wcount = h->wcount[i]; e->wsum = h->wsum[i]; }
}
static void group_cells(const tr_dist_hist *h, uint64_t g, uint64_t *b, uint64_t *e) {
    if (h->repr == TR_DIST_DENSE) { *b = g * h->cells; *e = *b + h->cells; }
    else { *b = h->offset[g]; *e = h->offset[g + 1]; }
}

/* ------------------------------------------------------------------ sorting entries (stable merge sort) */
static int entry_cmp(const dist_entry *a, const dist_entry *b, int k) {
    if (a->g != b->g) return a->g < b->g ? -1 : 1;
    for (int c = 0; c < k; c++) {
        int na = (a->nulls >> c) & 1, nb = (b->nulls >> c) & 1;
        if (na != nb) return na - nb;                             /* a value before NULL */
        if (!na && a->v[c] != b->v[c]) return a->v[c] < b->v[c] ? -1 : 1;
    }
    return 0;
}
static uint64_t entry_sort(dist_entry *a, dist_entry *tmp, uint64_t n, int k) {
    const uint64_t RUN = 32;
    for (uint64_t s = 0; s < n; s += RUN) {
        uint64_t e = s + RUN < n ? s + RUN : n;
        for (uint64_t i = s + 1; i < e; i++) {
            dist_entry x = a[i];
            uint64_t j = i;
            while (j > s && entry_cmp(&a[j - 1], &x, k) > 0) { a[j] = a[j - 1]; j--; }
            a[j] = x;
        }
    }
    uint64_t passes = n > 1;
    dist_entry *src = a, *dst = tmp;
    for (uint64_t w = RUN; w < n; w *= 2) {
        for (uint64_t s = 0; s < n; s += 2 * w) {
            uint64_t m = s + w < n ? s + w : n, e = s + 2 * w < n ? s + 2 * w : n, i = s, j = m, o = s;
            while (i < m && j < e) dst[o++] = entry_cmp(&src[j], &src[i], k) < 0 ? src[j++] : src[i++];
            while (i < m) dst[o++] = src[i++];
            while (j < e) dst[o++] = src[j++];
        }
        dist_entry *t = src; src = dst; dst = t;
        passes++;
    }
    if (src != a) copy(a, src, n * sizeof *a);
    return passes;
}
/* equal cells of sorted entries added in place: *m distinct; TR_MATH_OVERFLOW when a sum does not fit */
static int32_t entry_reduce(dist_entry *e, uint64_t n, int k, uint64_t *m) {
    uint64_t out = 0;
    for (uint64_t i = 0; i < n; i++) {
        if (out && entry_cmp(&e[out - 1], &e[i], k) == 0) {
            dist_entry *t = &e[out - 1];
            if (__builtin_add_overflow(t->count, e[i].count, &t->count) ||
                __builtin_add_overflow(t->wcount, e[i].wcount, &t->wcount) || !add_i128(&t->wsum, e[i].wsum))
                return TR_MATH_OVERFLOW;
        } else e[out++] = e[i];
    }
    *m = out;
    return TR_MATH_OK;
}
/* the sparse histogram of sorted distinct entries (rows: [G], copied). The final state is budgeted with `pre`
 * (the needs already held: scratch bytes, states) before it is allocated. */
static tr_dist_hist *sparse_from_entries(const dist_entry *e, uint64_t m, uint64_t G, int k, int weighted,
                                         const uint64_t *rows, const tr_math_need *pre, int states_from_pre,
                                         const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error) {
    uint64_t maxc = 0;
    for (uint64_t i = 0; i < m;) {
        uint64_t j = i;
        while (j < m && e[j].g == e[i].g) j++;
        if (j - i > maxc) maxc = j - i;
        i = j;
    }
    tr_math_need need = *pre;
    need.need[TR_BUDGET_SUPPORT] = maxc;
    if (!states_from_pre) need.need[TR_BUDGET_STATES] = m;
    need.need[TR_BUDGET_BYTES] = tr_budget_add(pre->need[TR_BUDGET_BYTES], hist_bytes(G, TR_DIST_SPARSE, k, weighted, m));
    int32_t rc = check_need(budget, &need, refusal);
    if (rc) { set_error(error, rc); return NULL; }
    tr_dist_hist *h = hist_alloc(G, TR_DIST_SPARSE, k, weighted, m);
    if (!h) { set_error(error, TR_MATH_MEMORY); return NULL; }
    copy(h->rows, rows, G * 8);
    uint64_t i = 0;
    for (uint64_t g = 0; g < G; g++) {
        h->offset[g] = i;
        while (i < m && e[i].g == g) {
            h->nulls[i] = e[i].nulls;
            for (int c = 0; c < k; c++) h->value[c][i] = e[i].v[c];
            h->count[i] = e[i].count;
            if (weighted) { h->wcount[i] = e[i].wcount; h->wsum[i] = e[i].wsum; }
            i++;
        }
    }
    h->offset[G] = m;
    hist_tally(h);
    set_error(error, TR_MATH_OK);
    return h;
}

/* ------------------------------------------------------------------ building from a scope */
static int32_t choose_strategy(const tr_dist_plan *p, const tr_scope_view *v) {
    if (p->repr == TR_DIST_DENSE) return TR_DIST_STRATEGY_DIRECT;
    if (p->strategy == TR_DIST_STRATEGY_COUNT || p->strategy == TR_DIST_STRATEGY_SORT) return p->strategy;
    uint64_t total = tr_budget_mul(v->ngroups, p->cells);
    uint64_t limit = tr_budget_add(tr_budget_mul(v->selected, TR_DIST_COUNT_FACTOR), TR_DIST_COUNT_SLACK);
    return p->cells != TR_BUDGET_UNKNOWN && total <= limit ? TR_DIST_STRATEGY_COUNT : TR_DIST_STRATEGY_SORT;
}
/* what init allocates (DIRECT: everything, the histogram object included; COUNT / SORT: the scratch) */
static void init_need(const tr_scope_view *v, const tr_dist_plan *p, int32_t strategy, tr_math_need *need) {
    need_zero(need);
    uint64_t G = v->ngroups, total = tr_budget_mul(G, p->cells);
    uint64_t bytes = tr_budget_add(sizeof(tr_dist_hist_builder), tr_budget_add(tr_budget_bitmap_bytes(v->rows),
                                                                                  tr_budget_mul(G, 8)));
    need->need[TR_BUDGET_VARIABLES] = (uint64_t)(p->k + p->weighted);
    if (strategy == TR_DIST_STRATEGY_DIRECT || strategy == TR_DIST_STRATEGY_COUNT) {
        need->need[TR_BUDGET_STATES] = total;
        bytes = tr_budget_add(bytes, tr_budget_mul(total, tr_dist_cell_bytes(p->weighted)));
        if (strategy == TR_DIST_STRATEGY_DIRECT) {
            need->need[TR_BUDGET_SUPPORT] = p->cells;
            bytes = tr_budget_add(bytes, sizeof(tr_dist_hist));
        }
    } else {
        bytes = tr_budget_add(bytes, tr_budget_mul(tr_budget_mul(v->selected, 2), sizeof(dist_entry)));
    }
    need->need[TR_BUDGET_BYTES] = bytes;
}

static void builder_release(tr_dist_hist_builder *b) {
    if (!b) return;
    free(b->rows); free(b->count); free(b->wcount); free(b->wsum); free(b->entries); free(b->covered);
    b->magic = 0;
    free(b);
}
void tr_dist_hist_builder_free(tr_dist_hist_builder *b) { if (builder_valid(b)) builder_release(b); }

tr_dist_hist_builder *tr_dist_hist_init(const tr_scope *scope, const tr_dist_spec *spec, const tr_math_budget *budget,
                                        tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    tr_scope_view view;
    if (!scope || !spec || tr_scope_get_view(scope, &view)) { set_error(error, TR_MATH_INVALID); return NULL; }
    tr_dist_spec ns;
    tr_dist_spec_normalize(spec, &ns);
    tr_dist_plan plan;
    int32_t rc = tr_dist_plan_make(&view, &ns, &plan);
    if (rc) { set_error(error, rc); return NULL; }
    int32_t strategy = choose_strategy(&plan, &view);
    tr_math_need need;
    init_need(&view, &plan, strategy, &need);
    tr_math_budget dflt;
    const tr_math_budget *bud = budget_of(budget, &dflt);
    rc = check_need(bud, &need, refusal);
    if (rc) { set_error(error, rc); return NULL; }
    tr_scope_info info;
    tr_scope_get_info(scope, &info);
    tr_dist_hist_builder *b = zalloc(sizeof *b);
    if (!b) { set_error(error, TR_MATH_MEMORY); return NULL; }
    b->strategy = strategy; b->spec = ns; b->plan = plan; b->scope = scope; b->view = view;
    b->digest[0] = info.digest[0]; b->digest[1] = info.digest[1];
    b->G = view.ngroups; b->need = need; b->budget = *bud;
    b->rows = zalloc(b->G * 8);
    b->covered = zalloc(((view.rows + 63) / 64) * 8);
    int ok = b->rows && b->covered;
    if (strategy == TR_DIST_STRATEGY_SORT) {
        b->capacity = view.selected;
        b->entries = zalloc(tr_budget_mul(view.selected, sizeof(dist_entry)));
        ok = ok && b->entries;
    } else {
        b->total = tr_budget_mul(b->G, plan.cells);
        b->count = zalloc(b->total * 8);
        ok = ok && b->count;
        if (plan.weighted) {
            b->wcount = zalloc(b->total * 8); b->wsum = zalloc(b->total * 16);
            ok = ok && b->wcount && b->wsum;
        }
    }
    if (!ok) { builder_release(b); set_error(error, TR_MATH_MEMORY); return NULL; }
    b->magic = BUILDER_MAGIC;
    set_error(error, TR_MATH_OK);
    return b;
}

/* the selected rows of [begin, end) (the weight sums are proven to fit by tr_dist_plan_make) */
static void accumulate_range(tr_dist_hist_builder *b, uint64_t begin, uint64_t end) {
    const tr_scope_view *v = &b->view;
    const tr_dist_plan *p = &b->plan;
    const int32_t wc = p->weight;
    for (uint64_t w = begin >> 6; w <= (end - 1) >> 6; w++) {
        uint64_t bits = v->selection[w] & range_mask(w, begin, end);
        while (bits) {
            uint64_t r = (w << 6) | (uint64_t)__builtin_ctzll(bits);
            bits &= bits - 1;
            uint32_t g = tr_scope_group(v, r);
            b->rows[g]++;
            int wvalid = p->weighted && tr_scope_present(v, wc, r);
            int64_t wv = wvalid ? tr_scope_value(v, wc, r) : 0;
            if (b->strategy == TR_DIST_STRATEGY_SORT) {
                dist_entry *e = &b->entries[b->nentries++];
                memset(e, 0, sizeof *e);
                e->g = g;
                for (int c = 0; c < p->k; c++) {
                    if (tr_scope_present(v, p->col[c], r)) e->v[c] = tr_scope_value(v, p->col[c], r);
                    else e->nulls |= (uint8_t)(1u << c);
                }
                e->count = 1;
                if (wvalid) { e->wcount = 1; e->wsum = tr_i128_set(wv); }
                continue;
            }
            uint64_t idx = (uint64_t)g * p->cells + tr_dist_row_cell(v, p, r);
            b->count[idx]++;
            if (wvalid) { b->wcount[idx]++; b->wsum[idx] = tr_i128_set(tr_i128_get(b->wsum[idx]) + wv); }
        }
    }
}

int32_t tr_dist_hist_accumulate(tr_dist_hist_builder *b, uint64_t begin, uint64_t end) {
    if (!builder_valid(b) || b->spent || begin > end || end > b->view.rows) return TR_MATH_INVALID;
    if (begin == end) return TR_MATH_OK;
    for (uint64_t w = begin >> 6; w <= (end - 1) >> 6; w++)
        if (b->covered[w] & range_mask(w, begin, end)) return TR_MATH_INVALID;   /* a row is counted once */
    accumulate_range(b, begin, end);
    for (uint64_t w = begin >> 6; w <= (end - 1) >> 6; w++) b->covered[w] |= range_mask(w, begin, end);
    b->ncovered += end - begin;
    return TR_MATH_OK;
}

int32_t tr_dist_hist_merge_builder(tr_dist_hist_builder *into, const tr_dist_hist_builder *from) {
    if (!builder_valid(into) || !builder_valid(from) || into == from || into->spent || from->spent ||
        into->strategy != from->strategy || memcmp(&into->spec, &from->spec, sizeof into->spec) ||
        memcmp(&into->plan, &from->plan, sizeof into->plan) || into->G != from->G) return TR_MATH_INVALID;
    if (into->scope != from->scope) {
        int32_t same = tr_scope_same(into->scope, from->scope);
        if (same != 1) return same < 0 ? same : TR_MATH_INVALID;
    }
    uint64_t nw = (into->view.rows + 63) / 64;
    for (uint64_t w = 0; w < nw; w++) if (into->covered[w] & from->covered[w]) return TR_MATH_INVALID;
    if (into->strategy == TR_DIST_STRATEGY_SORT && into->nentries + from->nentries > into->capacity) return TR_MATH_INVALID;
    for (uint64_t g = 0; g < into->G; g++) into->rows[g] += from->rows[g];
    if (into->strategy == TR_DIST_STRATEGY_SORT) {
        copy(into->entries + into->nentries, from->entries, from->nentries * sizeof(dist_entry));
        into->nentries += from->nentries;
    } else {
        for (uint64_t i = 0; i < into->total; i++) into->count[i] += from->count[i];
        if (into->plan.weighted)
            for (uint64_t i = 0; i < into->total; i++) {   /* disjoint rows of one scope: within the proof */
                into->wcount[i] += from->wcount[i];
                into->wsum[i] = tr_i128_set(tr_i128_get(into->wsum[i]) + tr_i128_get(from->wsum[i]));
            }
    }
    for (uint64_t w = 0; w < nw; w++) into->covered[w] |= from->covered[w];
    into->ncovered += from->ncovered;
    return TR_MATH_OK;
}

static void hist_from_builder_meta(tr_dist_hist *h, const tr_dist_hist_builder *b) {
    const tr_dist_plan *p = &b->plan;
    h->scoped = 1; h->digest[0] = b->digest[0]; h->digest[1] = b->digest[1];
    h->wnonneg = p->wnonneg; h->wscale = p->wscale; h->weight = p->weight; h->strategy = b->strategy;
    h->scans = 1;
    for (int c = 0; c < 3; c++) { h->col[c] = p->col[c]; h->scale[c] = c < p->k ? p->scale[c] : 0; }
}

tr_dist_hist *tr_dist_hist_finish(tr_dist_hist_builder *b, const tr_math_budget *budget, tr_math_refusal *refusal,
                                  int32_t *error) {
    tr_budget_none(refusal);
    if (!builder_valid(b) || b->spent || b->ncovered != b->view.rows) { set_error(error, TR_MATH_INVALID); return NULL; }
    const tr_math_budget *bud = budget ? budget : &b->budget;
    const tr_dist_plan *p = &b->plan;
    const uint64_t G = b->G;
    tr_dist_hist *h = NULL;
    if (b->strategy == TR_DIST_STRATEGY_DIRECT) {
        h = zalloc(sizeof *h);
        if (!h) { set_error(error, TR_MATH_MEMORY); return NULL; }
        h->ngroups = G; h->repr = TR_DIST_DENSE; h->k = p->k; h->weighted = p->weighted; h->ncells = b->total;
        h->rows = b->rows; h->count = b->count; h->wcount = b->wcount; h->wsum = b->wsum;
        b->rows = NULL; b->count = NULL; b->wcount = NULL; b->wsum = NULL;
        h->bytes = hist_bytes(G, TR_DIST_DENSE, p->k, p->weighted, b->total);
        h->magic = HIST_MAGIC;
        hist_set_domain(h, p);
    } else if (b->strategy == TR_DIST_STRATEGY_COUNT) {
        uint64_t occupied = 0, maxc = 0;
        for (uint64_t g = 0; g < G; g++) {
            uint64_t c = 0;
            for (uint64_t i = g * p->cells; i < (g + 1) * p->cells; i++) c += b->count[i] != 0;
            occupied += c;
            if (c > maxc) maxc = c;
        }
        tr_math_need need = b->need;
        need.need[TR_BUDGET_SUPPORT] = maxc;
        need.need[TR_BUDGET_BYTES] = tr_budget_add(b->need.need[TR_BUDGET_BYTES],
                                                   hist_bytes(G, TR_DIST_SPARSE, p->k, p->weighted, occupied));
        int32_t rc = check_need(bud, &need, refusal);
        if (rc) { set_error(error, rc); return NULL; }
        h = hist_alloc(G, TR_DIST_SPARSE, p->k, p->weighted, occupied);
        if (!h) { set_error(error, TR_MATH_MEMORY); return NULL; }
        copy(h->rows, b->rows, G * 8);
        uint64_t pos = 0;
        for (uint64_t g = 0; g < G; g++) {
            h->offset[g] = pos;
            for (uint64_t idx = 0; idx < p->cells; idx++) {
                uint64_t i = g * p->cells + idx;
                if (!b->count[i]) continue;
                uint8_t nulls = 0;
                for (int c = 0; c < p->k; c++) {
                    int null;
                    h->value[c][pos] = tr_dist_plan_coord(p, idx, c, &null);
                    if (null) nulls |= (uint8_t)(1u << c);
                }
                h->nulls[pos] = nulls;
                h->count[pos] = b->count[i];
                if (p->weighted) { h->wcount[pos] = b->wcount[i]; h->wsum[pos] = b->wsum[i]; }
                pos++;
            }
        }
        h->offset[G] = pos;
        h->passes = 2;
    } else {
        tr_math_need need = b->need;
        dist_entry *tmp = zalloc(tr_budget_mul(b->nentries, sizeof(dist_entry)));
        if (!tmp) { set_error(error, TR_MATH_MEMORY); return NULL; }
        uint64_t passes = entry_sort(b->entries, tmp, b->nentries, p->k), m = 0;
        free(tmp);
        int32_t rc = entry_reduce(b->entries, b->nentries, p->k, &m);
        if (!rc) h = sparse_from_entries(b->entries, m, G, p->k, p->weighted, b->rows, &need, 0, bud, refusal, &rc);
        if (!h) { set_error(error, rc); return NULL; }
        h->passes = passes + 2;
    }
    hist_from_builder_meta(h, b);
    hist_tally(h);
    free(b->entries); b->entries = NULL;
    b->spent = 1;
    set_error(error, TR_MATH_OK);
    return h;
}

tr_dist_hist *tr_dist_hist_build(const tr_scope *scope, const tr_dist_spec *spec, const tr_math_budget *budget,
                                 tr_math_refusal *refusal, int32_t *error) {
    tr_dist_hist_builder *b = tr_dist_hist_init(scope, spec, budget, refusal, error);
    if (!b) return NULL;
    int32_t rc = tr_dist_hist_accumulate(b, 0, b->view.rows);
    tr_dist_hist *h = rc ? NULL : tr_dist_hist_finish(b, budget, refusal, error);
    if (rc) set_error(error, rc);
    builder_release(b);
    return h;
}

int32_t tr_dist_hist_need(const tr_scope *scope, const tr_dist_spec *spec, int32_t target, tr_math_need *need,
                          uint64_t *capacity) {
    tr_scope_view view;
    if (!scope || !spec || !need || tr_scope_get_view(scope, &view) ||
        (target != TR_DIST_NEED_CPU && target != TR_DIST_NEED_DEVICE && target != TR_DIST_NEED_ADOPT)) return TR_MATH_INVALID;
    tr_dist_spec ns;
    tr_dist_spec_normalize(spec, &ns);
    tr_dist_plan plan;
    int32_t rc = tr_dist_plan_make(&view, &ns, &plan);
    if (rc) return rc;
    uint64_t cap = 0;
    if (target == TR_DIST_NEED_CPU) init_need(&view, &plan, choose_strategy(&plan, &view), need);
    else {
        tr_dist_device_need(&view, &plan, need, &cap);
        if (target == TR_DIST_NEED_ADOPT) {
            uint64_t total = tr_budget_mul(view.ngroups, plan.cells);
            need_zero(need);
            need->need[TR_BUDGET_VARIABLES] = (uint64_t)(plan.k + plan.weighted);
            need->need[TR_BUDGET_SUPPORT] = plan.repr == TR_DIST_DENSE ? plan.cells : (cap < plan.cells ? cap : plan.cells);
            need->need[TR_BUDGET_STATES] = plan.repr == TR_DIST_DENSE ? total : cap;
            need->need[TR_BUDGET_BYTES] = hist_bytes(view.ngroups, plan.repr, plan.k, plan.weighted, cap);
        }
    }
    if (capacity) *capacity = cap;
    return TR_MATH_OK;
}

/* ------------------------------------------------------------------ adoption of states computed elsewhere */
tr_dist_hist *tr_dist_hist_from_states(const tr_scope *origin, const tr_dist_spec *spec, const tr_dist_states *st,
                                       const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    tr_scope_view view;
    if (!origin || !spec || !st || tr_scope_get_view(origin, &view)) { set_error(error, TR_MATH_INVALID); return NULL; }
    tr_dist_spec ns;
    tr_dist_spec_normalize(spec, &ns);
    tr_dist_plan p;
    int32_t rc = tr_dist_plan_make(&view, &ns, &p);
    if (rc) { set_error(error, rc); return NULL; }
    const uint64_t G = view.ngroups;
    int bad = st->ngroups != G || st->k != p.k || st->repr != p.repr || st->weighted != p.weighted || !st->rows ||
              st->ncells > st->capacity || (st->ncells && !st->count) || (p.weighted && st->ncells && (!st->wcount || !st->wsum));
    if (!bad && p.repr == TR_DIST_DENSE) {
        bad = p.cells == TR_BUDGET_UNKNOWN || st->cells != p.cells || st->ncells != tr_budget_mul(G, p.cells);
        for (int c = 0; !bad && c < p.k; c++) bad = st->lo[c] != p.lo[c] || st->span[c] != p.span[c] || st->dim[c] != p.dim[c];
    }
    if (!bad && p.repr == TR_DIST_SPARSE) {
        bad = !st->offset || (st->ncells && !st->nulls);
        for (int c = 0; !bad && c < p.k; c++) bad = st->ncells && !st->value[c];
    }
    if (bad) { set_error(error, TR_MATH_INVALID); return NULL; }
    /* the content: rows, cells in order inside the scope domain, weights within the proven bound */
    uint64_t total = 0, maxc = 0, wmax = 0;
    if (p.weighted) {
        const tr_scope_view_column *w = &view.col[p.weight];
        wmax = w->nvalid ? (tr_abs_u64(w->min) > tr_abs_u64(w->max) ? tr_abs_u64(w->min) : tr_abs_u64(w->max)) : 0;
    }
    for (uint64_t g = 0; g < G && !bad; g++) {
        if (__builtin_add_overflow(total, st->rows[g], &total)) { bad = 1; break; }
        uint64_t b, e, sum = 0;
        if (p.repr == TR_DIST_DENSE) { b = g * p.cells; e = b + p.cells; }
        else {
            b = st->offset[g]; e = st->offset[g + 1];
            if ((g == 0 && b != 0) || e < b || e > st->ncells || (g == G - 1 && e != st->ncells)) { bad = 1; break; }
            if (e - b > maxc) maxc = e - b;
        }
        for (uint64_t i = b; i < e && !bad; i++) {
            uint64_t c = st->count[i];
            if (__builtin_add_overflow(sum, c, &sum)) { bad = 1; break; }
            if (p.weighted) {
                __int128 s = tr_i128_get(st->wsum[i]);
                unsigned __int128 mag = tr_dist_mag128(s);
                bad = st->wcount[i] > c || (!st->wcount[i] && s) || mag > (unsigned __int128)st->wcount[i] * wmax;
            }
            if (p.repr == TR_DIST_SPARSE && !bad) {
                uint8_t nulls = st->nulls[i];
                bad = !c || (nulls >> p.k) != 0;
                for (int cc = 0; cc < p.k && !bad; cc++) {
                    int64_t x = st->value[cc][i];
                    if ((nulls >> cc) & 1) bad = x != 0 || !p.nullslot[cc];
                    else bad = p.span[cc] == 0 || x < p.lo[cc] || (uint64_t)x - (uint64_t)p.lo[cc] >= p.span[cc];
                }
                if (!bad && i > b) {                                   /* strictly increasing cell order */
                    dist_entry x, y;
                    memset(&x, 0, sizeof x); memset(&y, 0, sizeof y);
                    x.nulls = st->nulls[i - 1]; y.nulls = nulls;
                    for (int cc = 0; cc < p.k; cc++) { x.v[cc] = st->value[cc][i - 1]; y.v[cc] = st->value[cc][i]; }
                    bad = entry_cmp(&x, &y, p.k) >= 0;
                }
            }
        }
        if (!bad) bad = sum != st->rows[g];
    }
    if (bad || total != view.selected) { set_error(error, TR_MATH_INVALID); return NULL; }
    tr_math_need need;
    need_zero(&need);
    need.need[TR_BUDGET_VARIABLES] = (uint64_t)(p.k + p.weighted);
    need.need[TR_BUDGET_SUPPORT] = p.repr == TR_DIST_DENSE ? p.cells : maxc;
    need.need[TR_BUDGET_STATES] = st->ncells;
    need.need[TR_BUDGET_BYTES] = hist_bytes(G, p.repr, p.k, p.weighted, st->ncells);
    tr_math_budget dflt;
    rc = check_need(budget_of(budget, &dflt), &need, refusal);
    if (rc) { set_error(error, rc); return NULL; }
    tr_dist_hist *h = hist_alloc(G, p.repr, p.k, p.weighted, st->ncells);
    if (!h) { set_error(error, TR_MATH_MEMORY); return NULL; }
    copy(h->rows, st->rows, G * 8);
    copy(h->count, st->count, st->ncells * 8);
    if (p.weighted) { copy(h->wcount, st->wcount, st->ncells * 8); copy(h->wsum, st->wsum, st->ncells * 16); }
    if (p.repr == TR_DIST_SPARSE) {
        copy(h->offset, st->offset, (G + 1) * 8);
        copy(h->nulls, st->nulls, st->ncells);
        for (int c = 0; c < p.k; c++) copy(h->value[c], st->value[c], st->ncells * 8);
    }
    if (p.repr == TR_DIST_DENSE) hist_set_domain(h, &p);
    tr_scope_info info;
    tr_scope_get_info(origin, &info);
    h->scoped = 1; h->digest[0] = info.digest[0]; h->digest[1] = info.digest[1];
    h->wnonneg = p.wnonneg; h->wscale = p.wscale; h->weight = p.weight; h->strategy = TR_DIST_STRATEGY_DEVICE;
    h->scans = 1;
    for (int c = 0; c < 3; c++) { h->col[c] = p.col[c]; h->scale[c] = c < p.k ? p.scale[c] : 0; }
    hist_tally(h);
    set_error(error, TR_MATH_OK);
    return h;
}

/* ------------------------------------------------------------------ representation changes */
/* the hull of the occupied cells: per coordinate [min, max] of its present values (none: lo = 0, hi = -1) and
 * whether some cell is NULL there */
static void hist_hull(const tr_dist_hist *h, int64_t *lo, int64_t *hi, int32_t *nullslot) {
    tr_dist_plan p;
    hist_plan(h, &p);
    int any[3] = {0, 0, 0};
    for (int c = 0; c < 3; c++) { lo[c] = 0; hi[c] = -1; nullslot[c] = 0; }
    for (uint64_t g = 0; g < h->ngroups; g++) {
        uint64_t b, e;
        group_cells(h, g, &b, &e);
        for (uint64_t i = b; i < e; i++) {
            if (!h->count[i]) continue;
            dist_entry x;
            hist_cell(h, &p, g, i, &x);
            for (int c = 0; c < h->k; c++) {
                if ((x.nulls >> c) & 1) { nullslot[c] = 1; continue; }
                if (!any[c] || x.v[c] < lo[c]) lo[c] = x.v[c];
                if (!any[c] || x.v[c] > hi[c]) hi[c] = x.v[c];
                any[c] = 1;
            }
        }
    }
}

tr_dist_hist *tr_dist_hist_to_dense(const tr_dist_hist *h, const int64_t *lo, const int64_t *hi, const int32_t *nullslot,
                                    const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    if (!hist_valid(h) || (lo && (!hi || !nullslot)) || (!lo && (hi || nullslot))) { set_error(error, TR_MATH_INVALID); return NULL; }
    int64_t L[3], H[3];
    int32_t N[3];
    if (lo) for (int c = 0; c < 3; c++) { L[c] = c < h->k ? lo[c] : 0; H[c] = c < h->k ? hi[c] : -1; N[c] = c < h->k ? nullslot[c] : 0; }
    else hist_hull(h, L, H, N);
    tr_dist_plan q;
    memset(&q, 0, sizeof q);
    q.k = h->k; q.repr = TR_DIST_DENSE; q.weighted = h->weighted;
    uint64_t cells = 1;
    for (int c = h->k - 1; c >= 0; c--) {
        if ((N[c] != 0 && N[c] != 1) || (__int128)H[c] < (__int128)L[c] - 1) { set_error(error, TR_MATH_INVALID); return NULL; }
        q.lo[c] = L[c]; q.nullslot[c] = N[c];
        q.span[c] = H[c] >= L[c] ? (uint64_t)H[c] - (uint64_t)L[c] + 1 : 0;
        if (H[c] >= L[c] && q.span[c] == 0) q.span[c] = TR_BUDGET_UNKNOWN;
        q.dim[c] = q.span[c] == TR_BUDGET_UNKNOWN ? TR_BUDGET_UNKNOWN : tr_budget_add(q.span[c], (uint64_t)N[c]);
        q.stride[c] = cells;
        cells = tr_budget_mul(cells, q.dim[c]);
    }
    for (int c = 0; c < h->k; c++) if (q.dim[c] == 0) cells = 0;
    q.cells = cells;
    tr_math_need need;
    need_zero(&need);
    need.need[TR_BUDGET_VARIABLES] = (uint64_t)(h->k + h->weighted);
    need.need[TR_BUDGET_SUPPORT] = cells;
    need.need[TR_BUDGET_STATES] = tr_budget_mul(h->ngroups, cells);
    need.need[TR_BUDGET_BYTES] = hist_bytes(h->ngroups, TR_DIST_DENSE, h->k, h->weighted, need.need[TR_BUDGET_STATES]);
    /* every occupied cell must be representable before anything is allocated */
    tr_dist_plan p;
    hist_plan(h, &p);
    for (uint64_t g = 0; g < h->ngroups; g++) {
        uint64_t b, e;
        group_cells(h, g, &b, &e);
        for (uint64_t i = b; i < e; i++) {
            if (!h->count[i]) continue;
            dist_entry x;
            hist_cell(h, &p, g, i, &x);
            for (int c = 0; c < h->k; c++) {
                int ok = ((x.nulls >> c) & 1) ? q.nullslot[c]
                                              : (q.span[c] && x.v[c] >= q.lo[c] && (uint64_t)x.v[c] - (uint64_t)q.lo[c] < q.span[c]);
                if (!ok) { set_error(error, TR_MATH_DOMAIN_ERROR); return NULL; }
            }
        }
    }
    tr_math_budget dflt;
    int32_t rc = check_need(budget_of(budget, &dflt), &need, refusal);
    if (rc) { set_error(error, rc); return NULL; }
    tr_dist_hist *d = hist_alloc(h->ngroups, TR_DIST_DENSE, h->k, h->weighted, need.need[TR_BUDGET_STATES]);
    if (!d) { set_error(error, TR_MATH_MEMORY); return NULL; }
    hist_copy_meta(d, h);
    hist_set_domain(d, &q);
    copy(d->rows, h->rows, h->ngroups * 8);
    for (uint64_t g = 0; g < h->ngroups; g++) {
        uint64_t b, e;
        group_cells(h, g, &b, &e);
        for (uint64_t i = b; i < e; i++) {
            if (!h->count[i]) continue;
            dist_entry x;
            hist_cell(h, &p, g, i, &x);
            uint64_t idx = g * cells;
            for (int c = 0; c < h->k; c++)
                idx += (((x.nulls >> c) & 1) ? q.span[c] : (uint64_t)x.v[c] - (uint64_t)q.lo[c]) * q.stride[c];
            d->count[idx] = x.count;
            if (h->weighted) { d->wcount[idx] = x.wcount; d->wsum[idx] = x.wsum; }
        }
    }
    d->strategy = TR_DIST_STRATEGY_DERIVED;
    hist_tally(d);
    set_error(error, TR_MATH_OK);
    return d;
}

tr_dist_hist *tr_dist_hist_to_sparse(const tr_dist_hist *h, const tr_math_budget *budget, tr_math_refusal *refusal,
                                     int32_t *error) {
    tr_budget_none(refusal);
    if (!hist_valid(h)) { set_error(error, TR_MATH_INVALID); return NULL; }
    uint64_t occupied = 0, maxc = 0;
    for (uint64_t g = 0; g < h->ngroups; g++) {
        uint64_t b, e, c = 0;
        group_cells(h, g, &b, &e);
        for (uint64_t i = b; i < e; i++) c += h->count[i] != 0;
        occupied += c;
        if (c > maxc) maxc = c;
    }
    tr_math_need need;
    need_zero(&need);
    need.need[TR_BUDGET_VARIABLES] = (uint64_t)(h->k + h->weighted);
    need.need[TR_BUDGET_SUPPORT] = maxc;
    need.need[TR_BUDGET_STATES] = occupied;
    need.need[TR_BUDGET_BYTES] = hist_bytes(h->ngroups, TR_DIST_SPARSE, h->k, h->weighted, occupied);
    tr_math_budget dflt;
    int32_t rc = check_need(budget_of(budget, &dflt), &need, refusal);
    if (rc) { set_error(error, rc); return NULL; }
    tr_dist_hist *s = hist_alloc(h->ngroups, TR_DIST_SPARSE, h->k, h->weighted, occupied);
    if (!s) { set_error(error, TR_MATH_MEMORY); return NULL; }
    hist_copy_meta(s, h);
    copy(s->rows, h->rows, h->ngroups * 8);
    tr_dist_plan p;
    hist_plan(h, &p);
    uint64_t pos = 0;
    for (uint64_t g = 0; g < h->ngroups; g++) {
        uint64_t b, e;
        group_cells(h, g, &b, &e);
        s->offset[g] = pos;
        for (uint64_t i = b; i < e; i++) {
            if (!h->count[i]) continue;
            dist_entry x;
            hist_cell(h, &p, g, i, &x);
            s->nulls[pos] = x.nulls;
            for (int c = 0; c < h->k; c++) s->value[c][pos] = x.v[c];
            s->count[pos] = x.count;
            if (h->weighted) { s->wcount[pos] = x.wcount; s->wsum[pos] = x.wsum; }
            pos++;
        }
    }
    s->offset[h->ngroups] = pos;
    s->strategy = TR_DIST_STRATEGY_DERIVED;
    hist_tally(s);
    set_error(error, TR_MATH_OK);
    return s;
}

/* ------------------------------------------------------------------ merge, rollup, marginal */
static int same_shape(const tr_dist_hist *a, const tr_dist_hist *b) {
    if (a->k != b->k || a->repr != b->repr || a->weighted != b->weighted || a->ngroups != b->ngroups ||
        (a->weighted && a->wscale != b->wscale)) return 0;
    for (int c = 0; c < a->k; c++) if (a->scale[c] != b->scale[c]) return 0;
    return 1;
}
static int same_domain(const tr_dist_hist *a, const tr_dist_hist *b) {
    if (a->repr != TR_DIST_DENSE) return 1;
    for (int c = 0; c < a->k; c++)
        if (a->lo[c] != b->lo[c] || a->span[c] != b->span[c] || a->dim[c] != b->dim[c]) return 0;
    return a->cells == b->cells;
}
/* sparse helpers on cell positions */
static void sparse_cell(const tr_dist_hist *h, uint64_t i, dist_entry *e) {
    memset(e, 0, sizeof *e);
    e->nulls = h->nulls[i];
    for (int c = 0; c < h->k; c++) e->v[c] = h->value[c][i];
    e->count = h->count[i];
    if (h->weighted) { e->wcount = h->wcount[i]; e->wsum = h->wsum[i]; }
}

tr_dist_hist *tr_dist_hist_merge(const tr_dist_hist *a, const tr_dist_hist *b, const tr_math_budget *budget,
                                 tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    if (!hist_valid(a) || !hist_valid(b) || !same_shape(a, b)) { set_error(error, TR_MATH_INVALID); return NULL; }
    if (!same_domain(a, b)) { set_error(error, TR_MATH_DOMAIN_ERROR); return NULL; }
    const uint64_t G = a->ngroups;
    for (uint64_t g = 0; g < G; g++) {
        uint64_t r;
        if (__builtin_add_overflow(a->rows[g], b->rows[g], &r)) { set_error(error, TR_MATH_OVERFLOW); return NULL; }
    }
    tr_math_need need;
    need_zero(&need);
    need.need[TR_BUDGET_VARIABLES] = (uint64_t)(a->k + a->weighted);
    tr_math_budget dflt;
    const tr_math_budget *bud = budget_of(budget, &dflt);
    tr_dist_hist *h;
    if (a->repr == TR_DIST_DENSE) {
        need.need[TR_BUDGET_SUPPORT] = a->cells;
        need.need[TR_BUDGET_STATES] = a->ncells;
        need.need[TR_BUDGET_BYTES] = hist_bytes(G, TR_DIST_DENSE, a->k, a->weighted, a->ncells);
        int32_t rc = check_need(bud, &need, refusal);
        if (rc) { set_error(error, rc); return NULL; }
        h = hist_alloc(G, TR_DIST_DENSE, a->k, a->weighted, a->ncells);
        if (!h) { set_error(error, TR_MATH_MEMORY); return NULL; }
        h->cells = a->cells;
        for (int c = 0; c < 3; c++) { h->lo[c] = a->lo[c]; h->span[c] = a->span[c]; h->dim[c] = a->dim[c]; h->nullslot[c] = a->nullslot[c]; }
        for (uint64_t i = 0; i < a->ncells; i++) {
            if (__builtin_add_overflow(a->count[i], b->count[i], &h->count[i])) { hist_release(h); set_error(error, TR_MATH_OVERFLOW); return NULL; }
            if (a->weighted) {
                h->wcount[i] = a->wcount[i] + b->wcount[i];          /* <= count, which did not overflow */
                h->wsum[i] = a->wsum[i];
                if (!add_i128(&h->wsum[i], b->wsum[i])) { hist_release(h); set_error(error, TR_MATH_OVERFLOW); return NULL; }
            }
        }
    } else {
        uint64_t total = 0, maxc = 0;                      /* pass 1: the merged cells of each group */
        for (uint64_t g = 0; g < G; g++) {
            uint64_t i = a->offset[g], ie = a->offset[g + 1], j = b->offset[g], je = b->offset[g + 1], c = 0;
            while (i < ie || j < je) {
                if (i < ie && j < je) {
                    dist_entry x, y;
                    sparse_cell(a, i, &x); sparse_cell(b, j, &y);
                    int s = entry_cmp(&x, &y, a->k);
                    if (s <= 0) i++;
                    if (s >= 0) j++;
                } else if (i < ie) i++;
                else j++;
                c++;
            }
            total += c;
            if (c > maxc) maxc = c;
        }
        need.need[TR_BUDGET_SUPPORT] = maxc;
        need.need[TR_BUDGET_STATES] = total;
        need.need[TR_BUDGET_BYTES] = hist_bytes(G, TR_DIST_SPARSE, a->k, a->weighted, total);
        int32_t rc = check_need(bud, &need, refusal);
        if (rc) { set_error(error, rc); return NULL; }
        h = hist_alloc(G, TR_DIST_SPARSE, a->k, a->weighted, total);
        if (!h) { set_error(error, TR_MATH_MEMORY); return NULL; }
        uint64_t pos = 0;
        for (uint64_t g = 0; g < G; g++) {
            h->offset[g] = pos;
            uint64_t i = a->offset[g], ie = a->offset[g + 1], j = b->offset[g], je = b->offset[g + 1];
            while (i < ie || j < je) {
                dist_entry x, y;
                int s;
                if (i < ie && j < je) { sparse_cell(a, i, &x); sparse_cell(b, j, &y); s = entry_cmp(&x, &y, a->k); }
                else if (i < ie) { sparse_cell(a, i, &x); s = -1; }
                else { sparse_cell(b, j, &y); s = 1; }
                dist_entry out = s <= 0 ? x : y;
                if (s == 0) {
                    if (__builtin_add_overflow(x.count, y.count, &out.count) || !add_i128(&out.wsum, y.wsum)) {
                        hist_release(h); set_error(error, TR_MATH_OVERFLOW); return NULL;
                    }
                    out.wcount = x.wcount + y.wcount;
                }
                if (s <= 0) i++;
                if (s >= 0) j++;
                h->nulls[pos] = out.nulls;
                for (int c = 0; c < a->k; c++) h->value[c][pos] = out.v[c];
                h->count[pos] = out.count;
                if (a->weighted) { h->wcount[pos] = out.wcount; h->wsum[pos] = out.wsum; }
                pos++;
            }
        }
        h->offset[G] = pos;
    }
    for (uint64_t g = 0; g < G; g++) h->rows[g] = a->rows[g] + b->rows[g];
    h->wnonneg = a->wnonneg && b->wnonneg; h->wscale = a->wscale;
    h->weight = a->weight == b->weight ? a->weight : -1;
    for (int c = 0; c < 3; c++) { h->scale[c] = a->scale[c]; h->col[c] = a->col[c] == b->col[c] ? a->col[c] : -1; }
    h->strategy = TR_DIST_STRATEGY_DERIVED;
    hist_tally(h);
    set_error(error, TR_MATH_OK);
    return h;
}

/* entries of every occupied cell, with the group mapped (map NULL: kept) and the coordinates projected onto keep[] */
static dist_entry *gather_entries(const tr_dist_hist *h, const uint32_t *map, const int *keep, int nkeep, uint64_t *n) {
    uint64_t m = h->repr == TR_DIST_SPARSE ? h->ncells : h->occupied;
    dist_entry *e = zalloc(tr_budget_mul(m, sizeof(dist_entry)));
    if (!e) return NULL;
    tr_dist_plan p;
    hist_plan(h, &p);
    uint64_t o = 0;
    for (uint64_t g = 0; g < h->ngroups; g++) {
        uint64_t b, en;
        group_cells(h, g, &b, &en);
        for (uint64_t i = b; i < en; i++) {
            if (!h->count[i]) continue;
            dist_entry x;
            hist_cell(h, &p, g, i, &x);
            if (map) x.g = map[g];
            if (keep) {
                dist_entry y = x;
                y.nulls = 0;
                for (int c = 0; c < nkeep; c++) {
                    y.v[c] = x.v[keep[c]];
                    if ((x.nulls >> keep[c]) & 1) y.nulls |= (uint8_t)(1u << c);
                }
                for (int c = nkeep; c < 3; c++) y.v[c] = 0;
                x = y;
            }
            e[o++] = x;
        }
    }
    *n = o;
    return e;
}
/* the sparse state of gathered entries: scratch budgeted first, then the final state */
static tr_dist_hist *sparse_by_sorting(const tr_dist_hist *h, const uint32_t *map, uint64_t G, const int *keep, int nkeep,
                                       const uint64_t *rows, uint64_t extra_bytes, const tr_math_budget *bud,
                                       tr_math_refusal *refusal, int32_t *error) {
    int k = keep ? nkeep : h->k;
    uint64_t m = h->repr == TR_DIST_SPARSE ? h->ncells : h->occupied;
    tr_math_need pre;
    need_zero(&pre);
    pre.need[TR_BUDGET_VARIABLES] = (uint64_t)(k + h->weighted);
    pre.need[TR_BUDGET_BYTES] = tr_budget_add(tr_budget_mul(tr_budget_mul(m, 2), sizeof(dist_entry)), extra_bytes);
    int32_t rc = check_need(bud, &pre, refusal);
    if (rc) { set_error(error, rc); return NULL; }
    uint64_t n = 0;
    dist_entry *e = gather_entries(h, map, keep, nkeep, &n);
    dist_entry *tmp = e ? zalloc(tr_budget_mul(n, sizeof(dist_entry))) : NULL;
    if (!e || !tmp) { free(e); free(tmp); set_error(error, TR_MATH_MEMORY); return NULL; }
    uint64_t passes = entry_sort(e, tmp, n, k), distinct = 0;
    free(tmp);
    rc = entry_reduce(e, n, k, &distinct);
    tr_dist_hist *s = NULL;
    if (!rc) s = sparse_from_entries(e, distinct, G, k, h->weighted, rows, &pre, 0, bud, refusal, &rc);
    free(e);
    if (!s) { set_error(error, rc); return NULL; }
    s->passes = passes + 2;
    set_error(error, TR_MATH_OK);
    return s;
}

tr_dist_hist *tr_dist_hist_rollup(const tr_dist_hist *h, const uint32_t *map, uint64_t coarse_groups,
                                  const tr_scope *origin, const tr_math_budget *budget, tr_math_refusal *refusal,
                                  int32_t *error) {
    tr_budget_none(refusal);
    tr_scope_info info;
    if (!hist_valid(h) || !map || coarse_groups < 1 || coarse_groups > TR_SCOPE_MAX_GROUPS ||
        (origin && (tr_scope_get_info(origin, &info) || info.ngroups != coarse_groups || info.selected != h->rows_total))) {
        set_error(error, TR_MATH_INVALID); return NULL;
    }
    if (origin) for (int c = 0; c <= h->k; c++) {
        int32_t col = c < h->k ? h->col[c] : h->weight;
        if ((c < h->k && col < 0) || col >= info.ncols) { set_error(error, TR_MATH_INVALID); return NULL; }
    }
    for (uint64_t g = 0; g < h->ngroups; g++) if (map[g] >= coarse_groups) { set_error(error, TR_MATH_INVALID); return NULL; }
    uint64_t *rows = zalloc(coarse_groups * 8);
    if (!rows) { set_error(error, TR_MATH_MEMORY); return NULL; }
    for (uint64_t g = 0; g < h->ngroups; g++) rows[map[g]] += h->rows[g];   /* <= the total, which fits */
    tr_math_budget dflt;
    const tr_math_budget *bud = budget_of(budget, &dflt);
    tr_dist_hist *r = NULL;
    int32_t rc = TR_MATH_OK;
    if (h->repr == TR_DIST_DENSE) {
        tr_math_need need;
        need_zero(&need);
        need.need[TR_BUDGET_VARIABLES] = (uint64_t)(h->k + h->weighted);
        need.need[TR_BUDGET_SUPPORT] = h->cells;
        need.need[TR_BUDGET_STATES] = tr_budget_mul(coarse_groups, h->cells);
        need.need[TR_BUDGET_BYTES] = tr_budget_add(hist_bytes(coarse_groups, TR_DIST_DENSE, h->k, h->weighted,
                                                              need.need[TR_BUDGET_STATES]), coarse_groups * 8);
        rc = check_need(bud, &need, refusal);
        if (!rc && !(r = hist_alloc(coarse_groups, TR_DIST_DENSE, h->k, h->weighted, need.need[TR_BUDGET_STATES]))) rc = TR_MATH_MEMORY;
        if (!rc) {
            r->cells = h->cells;
            for (int c = 0; c < 3; c++) { r->lo[c] = h->lo[c]; r->span[c] = h->span[c]; r->dim[c] = h->dim[c]; r->nullslot[c] = h->nullslot[c]; }
            copy(r->rows, rows, coarse_groups * 8);
            for (uint64_t g = 0; g < h->ngroups && !rc; g++) {
                if (!h->rows[g]) continue;                          /* an absent fine group holds nothing */
                uint64_t q = map[g];
                for (uint64_t j = 0; j < h->cells; j++) {
                    uint64_t f = g * h->cells + j, t = q * h->cells + j;
                    if (!h->count[f]) continue;
                    if (__builtin_add_overflow(r->count[t], h->count[f], &r->count[t])) { rc = TR_MATH_OVERFLOW; break; }
                    if (h->weighted) {
                        r->wcount[t] += h->wcount[f];
                        if (!add_i128(&r->wsum[t], h->wsum[f])) { rc = TR_MATH_OVERFLOW; break; }
                    }
                }
            }
            if (rc) { hist_release(r); r = NULL; }
        }
    } else {
        r = sparse_by_sorting(h, map, coarse_groups, NULL, 0, rows, coarse_groups * 8, bud, refusal, &rc);
    }
    free(rows);
    if (!r) { set_error(error, rc); return NULL; }
    hist_copy_meta(r, h);
    if (origin) { r->scoped = 1; r->digest[0] = info.digest[0]; r->digest[1] = info.digest[1]; }
    else { r->scoped = 0; r->digest[0] = r->digest[1] = 0; }
    r->strategy = TR_DIST_STRATEGY_DERIVED;
    r->scans = 0;
    hist_tally(r);
    set_error(error, TR_MATH_OK);
    return r;
}

tr_dist_hist *tr_dist_hist_marginal(const tr_dist_hist *h, uint32_t mask, const tr_math_budget *budget,
                                    tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    if (!hist_valid(h) || !mask || (mask >> h->k)) { set_error(error, TR_MATH_INVALID); return NULL; }
    int keep[3], nk = 0;
    for (int c = 0; c < h->k; c++) if ((mask >> c) & 1) keep[nk++] = c;
    tr_math_budget dflt;
    const tr_math_budget *bud = budget_of(budget, &dflt);
    tr_dist_hist *r = NULL;
    int32_t rc = TR_MATH_OK;
    if (h->repr == TR_DIST_DENSE) {
        tr_dist_plan q;
        memset(&q, 0, sizeof q);
        q.k = nk;
        uint64_t cells = 1;
        for (int c = nk - 1; c >= 0; c--) {
            q.lo[c] = h->lo[keep[c]]; q.span[c] = h->span[keep[c]]; q.dim[c] = h->dim[keep[c]]; q.nullslot[c] = h->nullslot[keep[c]];
            q.stride[c] = cells; cells = tr_budget_mul(cells, q.dim[c]);
        }
        for (int c = 0; c < nk; c++) if (q.dim[c] == 0) cells = 0;
        q.cells = cells;
        tr_math_need need;
        need_zero(&need);
        need.need[TR_BUDGET_VARIABLES] = (uint64_t)(nk + h->weighted);
        need.need[TR_BUDGET_SUPPORT] = cells;
        need.need[TR_BUDGET_STATES] = tr_budget_mul(h->ngroups, cells);
        need.need[TR_BUDGET_BYTES] = hist_bytes(h->ngroups, TR_DIST_DENSE, nk, h->weighted, need.need[TR_BUDGET_STATES]);
        rc = check_need(bud, &need, refusal);
        if (!rc && !(r = hist_alloc(h->ngroups, TR_DIST_DENSE, nk, h->weighted, need.need[TR_BUDGET_STATES]))) rc = TR_MATH_MEMORY;
        if (!rc) {
            hist_set_domain(r, &q);
            copy(r->rows, h->rows, h->ngroups * 8);
            tr_dist_plan p;
            hist_plan(h, &p);
            for (uint64_t g = 0; g < h->ngroups && !rc; g++)
                for (uint64_t j = 0; j < h->cells; j++) {
                    uint64_t f = g * h->cells + j;
                    if (!h->count[f]) continue;
                    uint64_t idx = g * cells;
                    for (int c = 0; c < nk; c++) idx += ((j / p.stride[keep[c]]) % p.dim[keep[c]]) * q.stride[c];
                    if (__builtin_add_overflow(r->count[idx], h->count[f], &r->count[idx])) { rc = TR_MATH_OVERFLOW; break; }
                    if (h->weighted) {
                        r->wcount[idx] += h->wcount[f];
                        if (!add_i128(&r->wsum[idx], h->wsum[f])) { rc = TR_MATH_OVERFLOW; break; }
                    }
                }
            if (rc) { hist_release(r); r = NULL; }
        }
    } else {
        r = sparse_by_sorting(h, NULL, h->ngroups, keep, nk, h->rows, 0, bud, refusal, &rc);
    }
    if (!r) { set_error(error, rc); return NULL; }
    hist_copy_meta(r, h);
    for (int c = 0; c < 3; c++) {
        r->col[c] = c < nk ? h->col[keep[c]] : -1;
        r->scale[c] = c < nk ? h->scale[keep[c]] : 0;
    }
    r->strategy = TR_DIST_STRATEGY_DERIVED;
    r->scans = 0;
    hist_tally(r);
    set_error(error, TR_MATH_OK);
    return r;
}

/* ------------------------------------------------------------------ reading */
tr_dist_hist *tr_dist_hist_clone(const tr_dist_hist *h, int32_t *error) {
    if (!hist_valid(h)) { set_error(error, TR_MATH_INVALID); return NULL; }
    tr_dist_hist *c = hist_alloc(h->ngroups, h->repr, h->k, h->weighted, h->ncells);
    if (!c) { set_error(error, TR_MATH_MEMORY); return NULL; }
    uint64_t *rows = c->rows, *offset = c->offset, *count = c->count, *wcount = c->wcount;
    int64_t *value[3] = {c->value[0], c->value[1], c->value[2]};
    uint8_t *nulls = c->nulls;
    tr_math_i128 *wsum = c->wsum;
    *c = *h;                                                      /* metadata; then the owned arrays */
    c->rows = rows; c->offset = offset; c->count = count; c->wcount = wcount; c->nulls = nulls; c->wsum = wsum;
    for (int i = 0; i < 3; i++) c->value[i] = value[i];
    copy(c->rows, h->rows, h->ngroups * 8);
    copy(c->count, h->count, h->ncells * 8);
    if (h->weighted) { copy(c->wcount, h->wcount, h->ncells * 8); copy(c->wsum, h->wsum, h->ncells * 16); }
    if (h->repr == TR_DIST_SPARSE) {
        copy(c->offset, h->offset, (h->ngroups + 1) * 8);
        copy(c->nulls, h->nulls, h->ncells);
        for (int i = 0; i < h->k; i++) copy(c->value[i], h->value[i], h->ncells * 8);
    }
    set_error(error, TR_MATH_OK);
    return c;
}

int32_t tr_dist_hist_equal(const tr_dist_hist *a, const tr_dist_hist *b) {
    if (!hist_valid(a) || !hist_valid(b)) return TR_MATH_INVALID;
    if (!same_shape(a, b) || !same_domain(a, b) || a->ncells != b->ncells) return 0;
    uint64_t n = a->ncells;
    if (memcmp(a->rows, b->rows, (size_t)a->ngroups * 8) || (n && memcmp(a->count, b->count, (size_t)n * 8))) return 0;
    if (a->weighted && n && (memcmp(a->wcount, b->wcount, (size_t)n * 8) || memcmp(a->wsum, b->wsum, (size_t)n * 16))) return 0;
    if (a->repr == TR_DIST_SPARSE) {
        if (memcmp(a->offset, b->offset, (size_t)(a->ngroups + 1) * 8) || (n && memcmp(a->nulls, b->nulls, (size_t)n))) return 0;
        for (int c = 0; c < a->k; c++) if (n && memcmp(a->value[c], b->value[c], (size_t)n * 8)) return 0;
    }
    return 1;
}

int32_t tr_dist_hist_get_info(const tr_dist_hist *h, tr_dist_hist_info *out) {
    if (!hist_valid(h) || !out) return TR_MATH_INVALID;
    memset(out, 0, sizeof *out);
    out->ngroups = h->ngroups; out->present = h->present; out->rows = h->rows_total;
    out->ncells = h->ncells; out->occupied = h->occupied; out->max_cells = h->max_cells; out->bytes = h->bytes;
    out->scans = h->scans; out->passes = h->passes; out->digest[0] = h->digest[0]; out->digest[1] = h->digest[1];
    out->k = h->k; out->repr = h->repr; out->weighted = h->weighted; out->scoped = h->scoped;
    out->weight = h->weight; out->wscale = h->wscale; out->strategy = h->strategy; out->wnonneg = h->wnonneg;
    for (int c = 0; c < 3; c++) { out->col[c] = h->col[c]; out->scale[c] = h->scale[c]; }
    if (h->repr == TR_DIST_DENSE) {
        for (int c = 0; c < h->k; c++) {
            out->lo[c] = h->lo[c]; out->dim[c] = h->dim[c]; out->nullslot[c] = h->nullslot[c];
            out->hi[c] = (int64_t)((uint64_t)h->lo[c] + h->span[c] - 1);
        }
    } else {
        int32_t ns[3];
        hist_hull(h, out->lo, out->hi, ns);
    }
    return TR_MATH_OK;
}

int32_t tr_dist_hist_get_states(const tr_dist_hist *h, tr_dist_states *out) {
    if (!hist_valid(h) || !out) return TR_MATH_INVALID;
    memset(out, 0, sizeof *out);
    out->ngroups = h->ngroups; out->ncells = h->ncells; out->capacity = h->ncells;
    out->cells = h->repr == TR_DIST_DENSE ? h->cells : 0;
    out->k = h->k; out->repr = h->repr; out->weighted = h->weighted;
    for (int c = 0; c < 3; c++) { out->lo[c] = h->lo[c]; out->span[c] = h->span[c]; out->dim[c] = h->dim[c]; }
    out->rows = h->rows; out->offset = h->offset; out->nulls = h->nulls;
    for (int c = 0; c < 3; c++) out->value[c] = h->value[c];
    out->count = h->count; out->wcount = h->wcount; out->wsum = h->wsum;
    return TR_MATH_OK;
}

/* ------------------------------------------------------------------ prefix states (DIST-002) */
static uint64_t prefix_bytes(uint64_t G, uint64_t n, int32_t degree) {
    uint64_t b = tr_budget_add(sizeof(tr_dist_prefix), tr_budget_mul(G, 16));
    b = tr_budget_add(b, tr_budget_mul(tr_budget_add(G, 1), 8));
    return tr_budget_add(b, tr_budget_mul(n, tr_budget_add(8, tr_budget_mul((uint64_t)degree + 1, 16))));
}
static void prefix_release(tr_dist_prefix *p) {
    if (!p) return;
    free(p->rows); free(p->nulls); free(p->offset); free(p->value); free(p->prefix);
    p->magic = 0;
    free(p);
}
void tr_dist_prefix_free(tr_dist_prefix *p) { if (prefix_valid(p)) prefix_release(p); }

/* the support of a 1-D histogram's group: calls back for each NULL cell and each present cell of positive mass */
static unsigned __int128 cell_mass(const tr_dist_hist *h, uint64_t i, int32_t mass) {
    return mass == TR_DIST_MASS_COUNT ? (unsigned __int128)h->count[i] : (unsigned __int128)tr_i128_get(h->wsum[i]);
}
static int cell_present_value(const tr_dist_hist *h, uint64_t g, uint64_t i, int64_t *x) {
    if (h->repr == TR_DIST_DENSE) {
        uint64_t idx = i - g * h->cells;
        if (idx >= h->span[0]) return 0;                            /* the NULL slot */
        *x = (int64_t)((uint64_t)h->lo[0] + idx);
        return 1;
    }
    if (h->nulls[i] & 1) return 0;
    *x = h->value[0][i];
    return 1;
}
/* pass 1 of a prefix build: the entries per group (into counts, may be NULL), the proof, the needs */
static int32_t prefix_plan(const tr_dist_hist *h, int32_t mass, int32_t degree, uint64_t *counts, tr_math_need *need) {
    if (!hist_valid(h) || !need || (mass != TR_DIST_MASS_COUNT && mass != TR_DIST_MASS_WEIGHT) || degree < 0) return TR_MATH_INVALID;
    if (h->k != 1 || degree > TR_DIST_MAX_DEGREE) return TR_MATH_UNSUPPORTED;
    if (mass == TR_DIST_MASS_WEIGHT && !h->weighted) return TR_MATH_INVALID;
    if (mass == TR_DIST_MASS_WEIGHT && !h->wnonneg) return TR_MATH_DOMAIN_ERROR;
    uint64_t total = 0, maxs = 0;
    int32_t rc = TR_MATH_OK;
    for (uint64_t g = 0; g < h->ngroups; g++) {
        uint64_t b, e, n = 0, X = 0;
        unsigned __int128 T = 0;
        group_cells(h, g, &b, &e);
        for (uint64_t i = b; i < e; i++) {
            int64_t x;
            if (!h->count[i] || !cell_present_value(h, g, i, &x)) continue;
            unsigned __int128 m = cell_mass(h, i, mass);
            if (!m) continue;
            if (m > I128_LIMIT - T) return TR_MATH_OVERFLOW;
            T += m; n++;
            if (tr_abs_u64(x) > X) X = tr_abs_u64(x);
        }
        for (int k = 1; k <= degree && T; k++) {                      /* T X^k <= 2^127 - 1 */
            unsigned __int128 bound = T;
            for (int j = 0; j < k; j++) {
                if (X && bound > I128_LIMIT / X) { rc = TR_MATH_OVERFLOW; break; }
                bound *= X;
            }
            if (rc) return rc;
        }
        if (counts) counts[g] = n;
        total += n;
        if (n > maxs) maxs = n;
    }
    need_zero(need);
    need->need[TR_BUDGET_VARIABLES] = 1;
    need->need[TR_BUDGET_DEGREE] = (uint64_t)degree;
    need->need[TR_BUDGET_SUPPORT] = maxs;
    need->need[TR_BUDGET_STATES] = tr_budget_mul(total, (uint64_t)degree + 1);
    need->need[TR_BUDGET_BYTES] = prefix_bytes(h->ngroups, total, degree);
    return TR_MATH_OK;
}

int32_t tr_dist_prefix_need(const tr_dist_hist *h, int32_t mass, int32_t degree, tr_math_need *need) {
    return prefix_plan(h, mass, degree, NULL, need);
}

tr_dist_prefix *tr_dist_prefix_build(const tr_dist_hist *h, int32_t mass, int32_t degree, const tr_math_budget *budget,
                                     tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    tr_math_need need;
    int32_t rc = prefix_plan(h, mass, degree, NULL, &need);
    if (rc) { set_error(error, rc); return NULL; }
    tr_math_budget dflt;
    rc = check_need(budget_of(budget, &dflt), &need, refusal);
    if (rc) { set_error(error, rc); return NULL; }
    const uint64_t G = h->ngroups, n = need.need[TR_BUDGET_STATES] / ((uint64_t)degree + 1);
    tr_dist_prefix *p = zalloc(sizeof *p);
    if (!p) { set_error(error, TR_MATH_MEMORY); return NULL; }
    p->rows = zalloc(G * 8); p->nulls = zalloc(G * 8); p->offset = zalloc((G + 1) * 8);
    p->value = zalloc(n * 8); p->prefix = zalloc(n * ((uint64_t)degree + 1) * 16);
    if (!p->rows || !p->nulls || !p->offset || !p->value || !p->prefix) { prefix_release(p); set_error(error, TR_MATH_MEMORY); return NULL; }
    p->degree = degree; p->mass = mass; p->scale = h->scale[0]; p->wscale = mass == TR_DIST_MASS_WEIGHT ? h->wscale : 0;
    p->scoped = h->scoped; p->digest[0] = h->digest[0]; p->digest[1] = h->digest[1];
    p->ngroups = G; p->nsupport = n; p->max_support = need.need[TR_BUDGET_SUPPORT];
    p->bytes = need.need[TR_BUDGET_BYTES];
    const uint64_t w = (uint64_t)degree + 1;
    uint64_t pos = 0;
    for (uint64_t g = 0; g < G; g++) {
        uint64_t b, e;
        group_cells(h, g, &b, &e);
        p->offset[g] = pos;
        p->rows[g] = h->rows[g];
        unsigned __int128 run[TR_DIST_MAX_DEGREE + 1] = {0};
        for (uint64_t i = b; i < e; i++) {
            int64_t x;
            if (!h->count[i]) continue;
            if (!cell_present_value(h, g, i, &x)) { p->nulls[g] += h->count[i]; continue; }
            unsigned __int128 m = cell_mass(h, i, mass);
            if (!m) continue;
            unsigned __int128 xk = 1, ux = (unsigned __int128)(__int128)x;
            for (int k = 0; k <= degree; k++) {                      /* exact: proven by prefix_plan */
                run[k] += m * xk;
                p->prefix[pos * w + (uint64_t)k] = tr_i128_set((__int128)run[k]);
                xk *= ux;
            }
            p->value[pos++] = x;
        }
        p->rows_total += p->rows[g]; p->nulls_total += p->nulls[g]; p->present += p->rows[g] != 0;
    }
    p->offset[G] = pos;
    p->magic = PREFIX_MAGIC;
    set_error(error, TR_MATH_OK);
    return p;
}

int32_t tr_dist_prefix_get_view(const tr_dist_prefix *p, tr_dist_prefix_view *v) {
    if (!prefix_valid(p) || !v) return TR_MATH_INVALID;
    memset(v, 0, sizeof *v);
    v->ngroups = p->ngroups; v->nsupport = p->nsupport; v->degree = p->degree; v->mass = p->mass;
    v->scale = p->scale; v->wscale = p->wscale;
    v->rows = p->rows; v->nulls = p->nulls; v->offset = p->offset; v->value = p->value; v->prefix = p->prefix;
    return TR_MATH_OK;
}
int32_t tr_dist_prefix_get_info(const tr_dist_prefix *p, tr_dist_prefix_info *out) {
    if (!prefix_valid(p) || !out) return TR_MATH_INVALID;
    memset(out, 0, sizeof *out);
    out->ngroups = p->ngroups; out->present = p->present; out->nsupport = p->nsupport; out->max_support = p->max_support;
    out->rows = p->rows_total; out->nulls = p->nulls_total; out->bytes = p->bytes;
    out->digest[0] = p->digest[0]; out->digest[1] = p->digest[1];
    out->degree = p->degree; out->mass = p->mass; out->scale = p->scale; out->wscale = p->wscale; out->scoped = p->scoped;
    return TR_MATH_OK;
}
int32_t tr_dist_prefix_equal(const tr_dist_prefix *a, const tr_dist_prefix *b) {
    if (!prefix_valid(a) || !prefix_valid(b)) return TR_MATH_INVALID;
    if (a->ngroups != b->ngroups || a->nsupport != b->nsupport || a->degree != b->degree || a->mass != b->mass ||
        a->scale != b->scale || a->wscale != b->wscale) return 0;
    uint64_t G = a->ngroups, n = a->nsupport;
    if (memcmp(a->rows, b->rows, (size_t)G * 8) || memcmp(a->nulls, b->nulls, (size_t)G * 8) ||
        memcmp(a->offset, b->offset, (size_t)(G + 1) * 8)) return 0;
    if (n && (memcmp(a->value, b->value, (size_t)n * 8) ||
              memcmp(a->prefix, b->prefix, (size_t)(n * ((uint64_t)a->degree + 1)) * 16))) return 0;
    return 1;
}
uint64_t tr_dist_prefix_device_bytes(const tr_dist_prefix_view *v) { return v ? tr_dist_prefix_copy_bytes(v) : TR_BUDGET_UNKNOWN; }

int32_t tr_dist_query(const tr_dist_prefix *p, const tr_dist_request *req, uint64_t n, tr_dist_result *out,
                      const tr_math_budget *budget, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!prefix_valid(p) || (n && (!req || !out))) return TR_MATH_INVALID;
    tr_math_budget dflt;
    int32_t rc = tr_budget_check_one(budget_of(budget, &dflt), TR_BUDGET_OUTPUT, n, refusal);
    if (rc) return rc;
    tr_dist_prefix_view v;
    tr_dist_prefix_get_view(p, &v);
    for (uint64_t i = 0; i < n; i++) tr_dist_eval(&v, &req[i], &out[i]);
    return TR_MATH_OK;
}
