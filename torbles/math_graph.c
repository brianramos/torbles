/* GRAPH-001 (declared graph layouts and reductions) and GRAPH-002 (bounded fixed points, paths and graph energies);
 * see math_graph.h for the contract.
 * SPDX-License-Identifier: Apache-2.0 */
#include "math_graph.h"
#include "mathematics.h"
#include "math_domain.h"
#include "math_scope.h"
#include "math_handles.h"
#include <stdlib.h>
#include <string.h>

TR_MATH_ABI(tr_graph_spec, TR_GRAPH_ABI);           /* the ctypes boundary of GRAPH-001/002 (math_handles.h) */
TR_MATH_ABI(tr_graph_info, TR_GRAPH_ABI);
TR_MATH_ABI(tr_graph_view, TR_GRAPH_ABI);
TR_MATH_ABI(tr_graph_rounds, TR_GRAPH_ABI);
TR_MATH_ABI(tr_graph_walks_info, TR_GRAPH_ABI);
TR_MATH_ABI(tr_graph_paths_info, TR_GRAPH_ABI);
TR_MATH_ABI(tr_graph_iter_info, TR_GRAPH_ABI);

_Static_assert(sizeof(tr_graph_spec) == 48, "tr_graph_spec ABI");
_Static_assert(sizeof(tr_graph_info) == 184, "tr_graph_info ABI");
_Static_assert(sizeof(tr_graph_view) == 168, "tr_graph_view ABI");
_Static_assert(sizeof(tr_graph_rounds) == 32, "tr_graph_rounds ABI");
_Static_assert(sizeof(tr_graph_walks_info) == 56, "tr_graph_walks_info ABI");
_Static_assert(sizeof(tr_graph_paths_info) == 64, "tr_graph_paths_info ABI");
_Static_assert(sizeof(tr_graph_iter_info) == 72, "tr_graph_iter_info ABI");

#define LAYOUT_MAGIC UINT32_C(0x47524150)

struct tr_graph_layout_s {
    uint32_t magic;
    tr_graph_spec spec;
    tr_graph_info info;
    int64_t *keys;                        /* the dictionary: n keys, increasing */
    uint64_t *in_off, *out_off;           /* undirected: one form (out_* alias in_*) */
    uint32_t *in_nbr, *out_nbr;
    int64_t *in_w, *out_w;
    uint64_t *in_m, *out_m;
    uint32_t *e_tail, *e_head;            /* directed: e_head / e_w / e_m alias the out form */
    int64_t *e_w;
    uint64_t *e_m;
    uint64_t *inc_off, *inc;              /* vertex-incidence lists */
    unsigned char *ident;                 /* spec | edge descriptor | vertex descriptor (lengths prefixed) */
    uint64_t ident_bytes, edesc_at, edesc_len, vdesc_at, vdesc_len;
    uint64_t max_in, max_out;
    int shared_form, shared_edges, has_vertices;
};

static void set_error(int32_t *error, int32_t code) { if (error) *error = code; }
static void *zalloc(uint64_t bytes) { return tr_graph_alloc(bytes); }
static int layout_ok(const tr_graph_layout *g) { return g && g->magic == LAYOUT_MAGIC; }
static const tr_math_budget *budget_or_default(const tr_math_budget *b, tr_math_budget *dflt) {
    if (b) return b;
    tr_budget_default(dflt);
    return dflt;
}
static int32_t check_need(const tr_math_budget *budget, const tr_math_need *need, tr_math_refusal *refusal) {
    tr_math_budget dflt;
    return tr_budget_check(budget_or_default(budget, &dflt), need, refusal);
}
static void bit_set(uint64_t *b, uint64_t i) { b[i >> 6] |= (uint64_t)1 << (i & 63); }

/* ------------------------------------------------------------------ the layout object */
static void layout_release(tr_graph_layout *g) {
    if (!g) return;
    free(g->keys);
    free(g->in_off); free(g->in_nbr); free(g->in_w); free(g->in_m);
    if (!g->shared_form) { free(g->out_off); free(g->out_nbr); free(g->out_w); free(g->out_m); }
    free(g->e_tail);
    if (!g->shared_edges) { free(g->e_head); free(g->e_w); free(g->e_m); }
    free(g->inc_off); free(g->inc); free(g->ident);
    g->magic = 0;
    free(g);
}
void tr_graph_layout_free(tr_graph_layout *g) { if (layout_ok(g)) layout_release(g); }

static void view_of(const tr_graph_layout *g, tr_graph_view *v) {
    memset(v, 0, sizeof *v);
    v->n = g->info.vertices; v->entries = g->info.entries; v->edges = g->info.edges;
    v->in_off = g->in_off; v->in_nbr = g->in_nbr; v->in_w = g->in_w; v->in_m = g->in_m;
    v->out_off = g->out_off; v->out_nbr = g->out_nbr; v->out_w = g->out_w; v->out_m = g->out_m;
    v->e_tail = g->e_tail; v->e_head = g->e_head; v->e_w = g->e_w; v->e_m = g->e_m;
    v->denominator = g->info.denominator;
    v->directed = g->info.directed; v->weighted = g->info.weighted;
    v->wmin = g->info.wmin; v->wmax = g->info.wmax;
    v->max_in = g->max_in; v->max_out = g->max_out;
}
int32_t tr_graph_layout_get_view(const tr_graph_layout *g, tr_graph_view *v) {
    if (!layout_ok(g) || !v) return TR_MATH_INVALID;
    view_of(g, v);
    return TR_MATH_OK;
}
int32_t tr_graph_layout_get_info(const tr_graph_layout *g, tr_graph_info *out) {
    if (!layout_ok(g) || !out) return TR_MATH_INVALID;
    *out = g->info;
    return TR_MATH_OK;
}
int32_t tr_graph_layout_keys(const tr_graph_layout *g, int64_t *keys, uint64_t capacity) {
    if (!layout_ok(g) || (g->info.vertices && !keys) || capacity < g->info.vertices) return TR_MATH_INVALID;
    if (g->info.vertices) memcpy(keys, g->keys, (size_t)(g->info.vertices * 8));
    return TR_MATH_OK;
}
static int64_t id_of(const int64_t *keys, uint64_t n, int64_t key, int *found) {
    uint64_t lo = 0, hi = n;
    while (lo < hi) {
        uint64_t mid = lo + (hi - lo) / 2;
        if (keys[mid] < key) lo = mid + 1; else hi = mid;
    }
    *found = lo < n && keys[lo] == key;
    return (int64_t)lo;
}
int32_t tr_graph_layout_ids(const tr_graph_layout *g, const int64_t *keys, uint64_t count, uint32_t *ids,
                            uint64_t *missing) {
    if (!layout_ok(g) || (count && (!keys || !ids))) return TR_MATH_INVALID;
    for (uint64_t i = 0; i < count; i++) {
        int found;
        int64_t id = id_of(g->keys, g->info.vertices, keys[i], &found);
        if (!found) { if (missing) *missing = i; return TR_MATH_NO_SOLUTION; }
        ids[i] = (uint32_t)id;
    }
    return TR_MATH_OK;
}
int32_t tr_graph_layout_incidence(const tr_graph_layout *g, uint64_t *offsets, uint64_t *entries, uint64_t capacity,
                                  uint64_t *count) {
    if (!layout_ok(g) || !offsets) return TR_MATH_INVALID;
    uint64_t n = g->info.vertices, total = g->inc_off[n];
    if (capacity < total || (total && !entries)) return TR_MATH_INVALID;
    memcpy(offsets, g->inc_off, (size_t)((n + 1) * 8));
    if (total) memcpy(entries, g->inc, (size_t)(total * 8));
    if (count) *count = total;
    return TR_MATH_OK;
}
int32_t tr_graph_layout_same(const tr_graph_layout *a, const tr_graph_layout *b) {
    if (!layout_ok(a) || !layout_ok(b)) return TR_MATH_INVALID;
    if (a == b) return 1;
    if (a->info.digest[0] != b->info.digest[0] || a->info.digest[1] != b->info.digest[1]) return 0;
    if (a->ident_bytes != b->ident_bytes || memcmp(a->ident, b->ident, (size_t)a->ident_bytes)) return 0;
    if (a->info.vertices != b->info.vertices) return 0;
    return !a->info.vertices || !memcmp(a->keys, b->keys, (size_t)(a->info.vertices * 8));
}
int32_t tr_graph_layout_fresh(const tr_graph_layout *g, const struct tr_scope_s *edges, const struct atable_s *etable,
                              const uint64_t *erows, const struct tr_scope_s *vertices, const struct atable_s *vtable,
                              const uint64_t *vrows) {
    if (!layout_ok(g) || !edges) return TR_MATH_INVALID;
    int32_t same = tr_scope_key_equal(edges, TR_SCOPE_KEY_ALL, g->ident + g->edesc_at, g->edesc_len);
    if (same < 0) return same;
    if (!same) return TR_MATH_STALE;
    int32_t rc = tr_scope_fresh(edges, etable, erows);
    if (rc) return rc;
    if (!g->has_vertices) return vertices ? TR_MATH_STALE : TR_MATH_OK;
    if (!vertices) return TR_MATH_STALE;
    same = tr_scope_key_equal(vertices, TR_SCOPE_KEY_ALL, g->ident + g->vdesc_at, g->vdesc_len);
    if (same < 0) return same;
    if (!same) return TR_MATH_STALE;
    return tr_scope_fresh(vertices, vtable, vrows);
}

/* ------------------------------------------------------------------ the build */
typedef struct {
    tr_scope_view ev, vv;
    int has_vertices, weighted;
    uint64_t rows, kept, null_endpoints, null_weights, dropped_loops, vertex_rows;
    uint64_t edesc, vdesc;            /* descriptor sizes */
    int64_t denominator;
    int32_t key_scale, weight_scale;
} build_plan;

static uint64_t ident_size(const build_plan *p) { return tr_budget_add(tr_budget_add(48 + 16, p->edesc), p->vdesc); }
/* every allocation of the build as if none were freed (n <= vb vertices, edges <= k, entries <= eb): an upper bound */
static uint64_t layout_bytes(const build_plan *p, int directed) {
    uint64_t k = p->kept, vb = tr_budget_add(tr_budget_mul(k, 2), p->vertex_rows);
    uint64_t eb = directed ? k : tr_budget_mul(k, 2);
    uint64_t bytes = tr_budget_add(sizeof(build_plan) + sizeof(struct tr_graph_layout_s), 2048 * 8);
    bytes = tr_budget_add(bytes, tr_budget_mul(vb, 28));      /* keys, slots, their sort scratch, the slot ranks */
    bytes = tr_budget_add(bytes, tr_budget_mul(k, 32 + 24));                    /* tuples (two), canonical edges */
    bytes = tr_budget_add(bytes, tr_budget_mul(tr_budget_add(vb, 1), 16));      /* counts, incidence offsets */
    bytes = tr_budget_add(bytes, tr_budget_mul(eb, 48));                        /* the entry lists (two) */
    bytes = tr_budget_add(bytes, tr_budget_mul(tr_graph_adj_bytes(vb, eb), directed ? 2 : 1));
    bytes = tr_budget_add(bytes, tr_budget_mul(k, directed ? 4 : 24));          /* the canonical edge arrays */
    bytes = tr_budget_add(bytes, tr_budget_mul(k, 16));                         /* the incidence lists */
    return tr_budget_add(bytes, ident_size(p));
}

static int32_t plan_build(const tr_scope *edges, const tr_scope *vertices, const tr_graph_spec *s, build_plan *p) {
    if (!edges || !s) return TR_MATH_INVALID;
    if ((s->direction != TR_GRAPH_DIRECTED && s->direction != TR_GRAPH_UNDIRECTED) ||
        (s->multi != TR_GRAPH_MULTI_KEEP && s->multi != TR_GRAPH_MULTI_DEDUP) ||
        (s->loops != TR_GRAPH_LOOPS_KEEP && s->loops != TR_GRAPH_LOOPS_DROP) ||
        s->combine < TR_GRAPH_COMBINE_REFUSE || s->combine > TR_GRAPH_COMBINE_SUM ||
        (s->nulls != TR_GRAPH_NULLS_SKIP && s->nulls != TR_GRAPH_NULLS_FORBID) || s->reserved || s->denominator < 1)
        return TR_MATH_INVALID;
    memset(p, 0, sizeof *p);
    int32_t rc = tr_scope_get_view(edges, &p->ev);
    if (rc) return rc;
    const tr_scope_view *v = &p->ev;
    if (s->src < 0 || s->src >= v->ncols || s->dst < 0 || s->dst >= v->ncols || s->weight < -1 ||
        s->weight >= v->ncols) return TR_MATH_INVALID;
    if (v->ngroups != 1) return TR_MATH_UNSUPPORTED;                         /* one graph per scope */
    if (v->col[s->src].scale != v->col[s->dst].scale) return TR_MATH_DOMAIN_ERROR;
    p->key_scale = v->col[s->src].scale;
    p->weighted = s->weight >= 0;
    p->weight_scale = p->weighted ? v->col[s->weight].scale : 0;
    int64_t den = s->denominator;
    for (int32_t i = 0; i < p->weight_scale; i++) if (!tr_mul_i64(den, 10, &den)) return TR_MATH_OVERFLOW;
    p->denominator = den;
    if (vertices) {
        rc = tr_scope_get_view(vertices, &p->vv);
        if (rc) return rc;
        if (s->vertex < 0 || s->vertex >= p->vv.ncols) return TR_MATH_INVALID;
        if (p->vv.ngroups != 1) return TR_MATH_UNSUPPORTED;
        if (p->vv.col[s->vertex].scale != p->key_scale) return TR_MATH_DOMAIN_ERROR;
        p->has_vertices = 1;
    }
    for (uint64_t w = 0; w < tr_graph_words(v->rows); w++) {
        uint64_t bits = v->selection[w];
        while (bits) {
            uint64_t r = w * 64 + (uint64_t)tr_graph_ctz(bits);
            bits &= bits - 1;
            p->rows++;
            if (!tr_scope_present(v, s->src, r) || !tr_scope_present(v, s->dst, r)) {
                if (s->nulls == TR_GRAPH_NULLS_FORBID) return TR_MATH_DOMAIN_ERROR;
                p->null_endpoints++;
                continue;
            }
            if (p->weighted && !tr_scope_present(v, s->weight, r)) {
                if (s->nulls == TR_GRAPH_NULLS_FORBID) return TR_MATH_DOMAIN_ERROR;
                p->null_weights++;
                continue;
            }
            if (s->loops == TR_GRAPH_LOOPS_DROP && tr_scope_value(v, s->src, r) == tr_scope_value(v, s->dst, r)) {
                p->dropped_loops++;
                continue;
            }
            p->kept++;
        }
    }
    if (p->has_vertices) {
        const tr_scope_view *q = &p->vv;
        for (uint64_t w = 0; w < tr_graph_words(q->rows); w++) {
            uint64_t bits = q->selection[w];
            while (bits) {
                uint64_t r = w * 64 + (uint64_t)tr_graph_ctz(bits);
                bits &= bits - 1;
                if (tr_scope_present(q, s->vertex, r)) p->vertex_rows++;
            }
        }
    }
    p->edesc = tr_scope_key_size(edges, TR_SCOPE_KEY_ALL);
    p->vdesc = p->has_vertices ? tr_scope_key_size(vertices, TR_SCOPE_KEY_ALL) : 0;
    return TR_MATH_OK;
}
static void plan_need(const build_plan *p, const tr_graph_spec *s, tr_math_need *need) {
    tr_graph_zero_need(need);
    int directed = s->direction == TR_GRAPH_DIRECTED;
    need->need[TR_BUDGET_VARIABLES] = 2 + (uint64_t)p->weighted + (uint64_t)p->has_vertices;
    need->need[TR_BUDGET_SUPPORT] = p->kept;
    need->need[TR_BUDGET_STATES] = tr_budget_add(tr_budget_mul(p->kept, 2), p->vertex_rows);
    need->need[TR_BUDGET_OUTPUT] = directed ? p->kept : tr_budget_mul(p->kept, 2);
    need->need[TR_BUDGET_BYTES] = layout_bytes(p, directed);
}
int32_t tr_graph_layout_need(const struct tr_scope_s *edges, const struct tr_scope_s *vertices,
                             const tr_graph_spec *spec, tr_math_need *need) {
    if (!need) return TR_MATH_INVALID;
    build_plan p;
    int32_t rc = plan_build(edges, vertices, spec, &p);
    if (rc) return rc;
    plan_need(&p, spec, need);
    return TR_MATH_OK;
}

/* stable LSD radix sort of (key, slot) pairs by key (11-bit digits, constant digits skipped); kt, st: n scratch;
 * the sorted pairs end in k, sl */
static void sort_pairs(int64_t *k, uint32_t *sl, int64_t *kt, uint32_t *st, uint64_t n, uint64_t *count /* 2048 */) {
    int64_t *a = k, *at = kt;
    uint32_t *b = sl, *bt = st;
    const uint64_t sign = (uint64_t)1 << 63;
    for (int pass = 0; pass < 6; pass++) {
        const int shift = pass * 11;
        memset(count, 0, 2048 * 8);
        for (uint64_t i = 0; i < n; i++) count[(((uint64_t)a[i] ^ sign) >> shift) & 2047]++;
        int constant = 0;
        for (int dd = 0; dd < 2048; dd++) if (count[dd]) { constant = count[dd] == n; break; }
        if (constant) continue;
        uint64_t at_ = 0;
        for (int dd = 0; dd < 2048; dd++) { uint64_t c = count[dd]; count[dd] = at_; at_ += c; }
        for (uint64_t i = 0; i < n; i++) {
            uint64_t j = count[(((uint64_t)a[i] ^ sign) >> shift) & 2047]++;
            at[j] = a[i]; bt[j] = b[i];
        }
        int64_t *t1 = a; a = at; at = t1;
        uint32_t *t2 = b; b = bt; bt = t2;
    }
    if (a != k) { memcpy(k, a, (size_t)(n * 8)); memcpy(sl, b, (size_t)(n * 4)); }
}

typedef struct { uint32_t u, v; int64_t w; } tuple_t;
typedef struct { uint32_t u, v; int64_t w; uint64_t m; } entry_t;

static int cmp_w(const void *a, const void *b) {
    int64_t x = ((const tuple_t *)a)->w, y = ((const tuple_t *)b)->w;
    return x < y ? -1 : x > y;
}
/* stable counting sort of tuples by the field sel (0: u, 1: v), ids < n */
static void csort_tuples(const tuple_t *in, tuple_t *out, uint64_t k, uint64_t n, uint64_t *cnt, int sel) {
    memset(cnt, 0, (size_t)((n + 1) * 8));
    for (uint64_t i = 0; i < k; i++) cnt[(sel ? in[i].v : in[i].u) + 1]++;
    for (uint64_t i = 0; i < n; i++) cnt[i + 1] += cnt[i];
    for (uint64_t i = 0; i < k; i++) out[cnt[sel ? in[i].v : in[i].u]++] = in[i];
}
static void csort_entries(const entry_t *in, entry_t *out, uint64_t k, uint64_t n, uint64_t *cnt, int sel) {
    memset(cnt, 0, (size_t)((n + 1) * 8));
    for (uint64_t i = 0; i < k; i++) cnt[(sel ? in[i].v : in[i].u) + 1]++;
    for (uint64_t i = 0; i < n; i++) cnt[i + 1] += cnt[i];
    for (uint64_t i = 0; i < k; i++) out[cnt[sel ? in[i].v : in[i].u]++] = in[i];
}

/* 128-bit digest of a byte string (two multiply-xorshift streams over 8-byte words; a pre-filter only) */
static void digest_bytes(const unsigned char *p, uint64_t n, uint64_t *h) {
    uint64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        uint64_t x;
        memcpy(&x, p + i, 8);
        h[0] = (h[0] ^ x) * UINT64_C(0x9e3779b97f4a7c15); h[0] ^= h[0] >> 32;
        h[1] = (h[1] + x) * UINT64_C(0xc2b2ae3d27d4eb4f); h[1] ^= h[1] >> 29;
    }
    for (; i < n; i++) {
        h[0] = (h[0] ^ p[i]) * UINT64_C(0x100000001b3);
        h[1] = (h[1] + p[i]) * UINT64_C(0x9e3779b97f4a7c15) + (h[1] >> 29);
    }
}

/* the adjacency form of sorted entries (by row u, then v, then w): off[n + 1], nbr, w, m */
static int32_t form_from(const entry_t *e, uint64_t k, uint64_t n, int by_v, uint64_t **off, uint32_t **nbr,
                         int64_t **w, uint64_t **m, uint64_t *maxrow) {
    *off = zalloc((n + 1) * 8); *nbr = zalloc(k * 4); *w = zalloc(k * 8); *m = zalloc(k * 8);
    if (!*off || !*nbr || !*w || !*m) return TR_MATH_MEMORY;
    uint64_t best = 0;
    for (uint64_t i = 0; i < k; i++) {
        uint32_t row = by_v ? e[i].v : e[i].u;
        (*off)[row + 1]++;
        (*nbr)[i] = by_v ? e[i].u : e[i].v;
        (*w)[i] = e[i].w; (*m)[i] = e[i].m;
    }
    for (uint64_t r = 0; r < n; r++) { if ((*off)[r + 1] > best) best = (*off)[r + 1]; (*off)[r + 1] += (*off)[r]; }
    *maxrow = best;
    return TR_MATH_OK;
}

tr_graph_layout *tr_graph_layout_build(const struct tr_scope_s *edges, const struct tr_scope_s *vertices,
                                       const tr_graph_spec *spec, const tr_math_budget *budget,
                                       tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    build_plan *p = zalloc(sizeof *p);
    if (!p) { set_error(error, TR_MATH_MEMORY); return NULL; }
    int32_t rc = plan_build(edges, vertices, spec, p);
    if (rc) { free(p); set_error(error, rc); return NULL; }
    tr_math_need need;
    plan_need(p, spec, &need);
    rc = check_need(budget, &need, refusal);
    if (!rc && (p->kept > TR_GRAPH_MAX_ENTRIES || need.need[TR_BUDGET_OUTPUT] > TR_GRAPH_MAX_ENTRIES ||
                need.need[TR_BUDGET_STATES] > UINT32_MAX))        /* the key slots are 32-bit */
        rc = TR_MATH_UNSUPPORTED;
    if (rc) { free(p); set_error(error, rc); return NULL; }
    const int directed = spec->direction == TR_GRAPH_DIRECTED;
    const tr_scope_view *ev = &p->ev;
    const uint64_t k = p->kept, nk = 2 * k + p->vertex_rows;
    tr_graph_layout *g = zalloc(sizeof *g);
    int64_t *keys = zalloc(nk * 8), *ktmp = zalloc(nk * 8);
    uint32_t *slot = zalloc(nk * 4), *stmp = zalloc(nk * 4), *sid = zalloc(nk * 4);
    uint64_t *kcount = zalloc(2048 * 8);
    tuple_t *t = zalloc(k * sizeof(tuple_t)), *t2 = zalloc(k * sizeof(tuple_t));
    entry_t *ent = NULL, *ent2 = NULL;
    uint64_t *cnt = NULL;
    if (!g || !keys || !ktmp || !slot || !stmp || !sid || !kcount || !t || !t2) { rc = TR_MATH_MEMORY; goto fail; }
    g->spec = *spec;
    /* 1. the keys of the kept rows (and the vertex rows), sorted and unique: the dictionary */
    uint64_t at = 0, ti = 0;
    for (uint64_t w = 0; w < tr_graph_words(ev->rows); w++) {
        uint64_t bits = ev->selection[w];
        while (bits) {
            uint64_t r = w * 64 + (uint64_t)tr_graph_ctz(bits);
            bits &= bits - 1;
            if (!tr_scope_present(ev, spec->src, r) || !tr_scope_present(ev, spec->dst, r)) continue;
            if (p->weighted && !tr_scope_present(ev, spec->weight, r)) continue;
            int64_t a = tr_scope_value(ev, spec->src, r), b = tr_scope_value(ev, spec->dst, r);
            if (spec->loops == TR_GRAPH_LOOPS_DROP && a == b) continue;
            slot[at] = (uint32_t)at; keys[at++] = a;              /* slot 2 i: the tail, 2 i + 1: the head */
            slot[at] = (uint32_t)at; keys[at++] = b;
            t[ti].u = 0; t[ti].v = 0;
            t[ti].w = p->weighted ? tr_scope_value(ev, spec->weight, r) : 1;
            ti++;
        }
    }
    if (p->has_vertices) {
        const tr_scope_view *q = &p->vv;
        for (uint64_t w = 0; w < tr_graph_words(q->rows); w++) {
            uint64_t bits = q->selection[w];
            while (bits) {
                uint64_t r = w * 64 + (uint64_t)tr_graph_ctz(bits);
                bits &= bits - 1;
                if (tr_scope_present(q, spec->vertex, r)) { slot[at] = (uint32_t)at; keys[at++] = tr_scope_value(q, spec->vertex, r); }
            }
        }
    }
    sort_pairs(keys, slot, ktmp, stmp, nk, kcount);
    uint64_t n = 0;
    int64_t last = 0;
    for (uint64_t i = 0; i < nk; i++) {                       /* unique keys in place; every slot's rank */
        if (!n || keys[i] != last) { last = keys[i]; keys[n++] = last; }
        sid[slot[i]] = (uint32_t)(n - 1);
    }
    free(ktmp); ktmp = NULL; free(kcount); kcount = NULL; free(stmp); stmp = NULL; free(slot); slot = NULL;
    if (n > TR_GRAPH_MAX_VERTICES) { rc = TR_MATH_UNSUPPORTED; goto fail; }
    /* 2. the endpoints as ids (edge i: slots 2 i and 2 i + 1); undirected: oriented (min, max) */
    for (uint64_t i = 0; i < k; i++) {
        uint32_t u = sid[2 * i], v = sid[2 * i + 1];
        if (!directed && v < u) { uint32_t x = u; u = v; v = x; }
        t[i].u = u; t[i].v = v;
    }
    free(sid); sid = NULL;
    /* 3. sort the tuples by (u, v, w): stable counting sorts by v then u, then w within equal (u, v) */
    cnt = zalloc((n + 1) * 8);
    if (!cnt) { rc = TR_MATH_MEMORY; goto fail; }
    csort_tuples(t, t2, k, n, cnt, 1);
    csort_tuples(t2, t, k, n, cnt, 0);
    for (uint64_t i = 0; i < k;) {
        uint64_t j = i + 1;
        while (j < k && t[j].u == t[i].u && t[j].v == t[i].v) j++;
        if (j - i > 1) {
            if (j - i <= 16) {
                for (uint64_t a = i + 1; a < j; a++) {
                    tuple_t x = t[a];
                    uint64_t b = a;
                    while (b > i && t[b - 1].w > x.w) { t[b] = t[b - 1]; b--; }
                    t[b] = x;
                }
            } else qsort(t + i, (size_t)(j - i), sizeof(tuple_t), cmp_w);
        }
        i = j;
    }
    free(t2); t2 = NULL;
    /* 4. canonical edges: identical (u, v, w) merged into a multiplicity; DEDUP: one edge per (u, v) */
    ent = zalloc((k ? k : 1) * sizeof(entry_t));
    if (!ent) { rc = TR_MATH_MEMORY; goto fail; }
    uint64_t M = 0, loops = 0;
    for (uint64_t i = 0; i < k;) {
        uint64_t j = i + 1;
        if (spec->multi == TR_GRAPH_MULTI_DEDUP) {
            int64_t w = t[i].w;
            while (j < k && t[j].u == t[i].u && t[j].v == t[i].v) {
                int64_t x = t[j].w;
                switch (spec->combine) {
                case TR_GRAPH_COMBINE_REFUSE: if (x != w) { rc = TR_MATH_DOMAIN_ERROR; goto fail; } break;
                case TR_GRAPH_COMBINE_MIN: if (x < w) w = x; break;
                case TR_GRAPH_COMBINE_MAX: if (x > w) w = x; break;
                default: if (!tr_add_i64(w, x, &w)) { rc = TR_MATH_OVERFLOW; goto fail; }
                }
                j++;
            }
            ent[M].u = t[i].u; ent[M].v = t[i].v; ent[M].w = w; ent[M].m = 1;
        } else {
            while (j < k && t[j].u == t[i].u && t[j].v == t[i].v && t[j].w == t[i].w) j++;
            ent[M].u = t[i].u; ent[M].v = t[i].v; ent[M].w = t[i].w; ent[M].m = j - i;
        }
        if (ent[M].u == ent[M].v) loops += ent[M].m;
        M++;
        i = j;
    }
    free(t); t = NULL;
    /* 5. the adjacency forms */
    uint64_t E = M;
    g->info.vertices = n;
    if (directed) {
        rc = form_from(ent, M, n, 0, &g->out_off, &g->out_nbr, &g->out_w, &g->out_m, &g->max_out);
        if (rc) goto fail;
        ent2 = zalloc((M ? M : 1) * sizeof(entry_t));
        if (!ent2) { rc = TR_MATH_MEMORY; goto fail; }
        csort_entries(ent, ent2, M, n, cnt, 1);                 /* by head, stable: tails increasing */
        rc = form_from(ent2, M, n, 1, &g->in_off, &g->in_nbr, &g->in_w, &g->in_m, &g->max_in);
        if (rc) goto fail;
        g->e_tail = zalloc(M * 4);
        if (!g->e_tail) { rc = TR_MATH_MEMORY; goto fail; }
        for (uint64_t i = 0; i < M; i++) g->e_tail[i] = ent[i].u;
        g->e_head = g->out_nbr; g->e_w = g->out_w; g->e_m = g->out_m;
        g->shared_edges = 1;
    } else {
        uint64_t loopsE = 0;
        for (uint64_t i = 0; i < M; i++) loopsE += ent[i].u == ent[i].v;
        E = 2 * M - loopsE;
        entry_t *both = zalloc((E ? E : 1) * sizeof(entry_t));
        ent2 = zalloc((E ? E : 1) * sizeof(entry_t));
        if (!both || !ent2) { free(both); rc = TR_MATH_MEMORY; goto fail; }
        uint64_t q = 0;
        for (uint64_t i = 0; i < M; i++) both[q++] = ent[i];
        for (uint64_t i = 0; i < M; i++)
            if (ent[i].u != ent[i].v) { both[q] = ent[i]; both[q].u = ent[i].v; both[q].v = ent[i].u; q++; }
        csort_entries(both, ent2, E, n, cnt, 1);                /* by column, then stable by row */
        csort_entries(ent2, both, E, n, cnt, 0);
        rc = form_from(both, E, n, 0, &g->in_off, &g->in_nbr, &g->in_w, &g->in_m, &g->max_in);
        free(both);
        if (rc) goto fail;
        g->out_off = g->in_off; g->out_nbr = g->in_nbr; g->out_w = g->in_w; g->out_m = g->in_m;
        g->max_out = g->max_in;
        g->shared_form = 1;
        g->e_tail = zalloc(M * 4); g->e_head = zalloc(M * 4); g->e_w = zalloc(M * 8); g->e_m = zalloc(M * 8);
        if (!g->e_tail || !g->e_head || !g->e_w || !g->e_m) { rc = TR_MATH_MEMORY; goto fail; }
        for (uint64_t i = 0; i < M; i++) {
            g->e_tail[i] = ent[i].u; g->e_head[i] = ent[i].v; g->e_w[i] = ent[i].w; g->e_m[i] = ent[i].m;
        }
    }
    free(ent2); ent2 = NULL;
    /* 6. the vertex-incidence lists (edge order within each vertex) */
    g->inc_off = zalloc((n + 1) * 8);
    if (!g->inc_off) { rc = TR_MATH_MEMORY; goto fail; }
    for (uint64_t i = 0; i < M; i++) {
        g->inc_off[ent[i].u + 1]++;
        if (ent[i].v != ent[i].u) g->inc_off[ent[i].v + 1]++;
    }
    for (uint64_t r = 0; r < n; r++) g->inc_off[r + 1] += g->inc_off[r];
    g->inc = zalloc((g->inc_off[n] ? g->inc_off[n] : 1) * 8);
    if (!g->inc) { rc = TR_MATH_MEMORY; goto fail; }
    memcpy(cnt, g->inc_off, (size_t)(n * 8));
    for (uint64_t i = 0; i < M; i++) {
        if (ent[i].u == ent[i].v) { g->inc[cnt[ent[i].u]++] = (i << 2) | TR_GRAPH_ROLE_LOOP; continue; }
        g->inc[cnt[ent[i].u]++] = (i << 2) | TR_GRAPH_ROLE_TAIL;
        g->inc[cnt[ent[i].v]++] = (i << 2) | TR_GRAPH_ROLE_HEAD;
    }
    /* 7. the description */
    tr_graph_info *in = &g->info;
    in->entries = E; in->edges = M;
    in->rows = p->rows; in->kept = k; in->null_endpoints = p->null_endpoints; in->null_weights = p->null_weights;
    in->loops = loops; in->dropped_loops = p->dropped_loops;
    in->merged = spec->multi == TR_GRAPH_MULTI_DEDUP ? k - M : 0;
    uint64_t mult = 0;
    for (uint64_t i = 0; i < M; i++) mult += ent[i].m;
    in->multiplicity = mult;
    in->vertex_rows = p->vertex_rows;
    uint64_t iso = 0;
    for (uint64_t v = 0; v < n; v++)
        iso += g->in_off[v + 1] == g->in_off[v] && g->out_off[v + 1] == g->out_off[v];
    in->isolated = iso;
    in->wmin = INT64_MAX; in->wmax = INT64_MIN;
    if (!M) { in->wmin = 1; in->wmax = 0; }
    for (uint64_t i = 0; i < M; i++) {
        if (ent[i].w < in->wmin) in->wmin = ent[i].w;
        if (ent[i].w > in->wmax) in->wmax = ent[i].w;
    }
    in->denominator = p->denominator;
    in->directed = directed; in->weighted = p->weighted; in->multi = spec->multi; in->loops_policy = spec->loops;
    in->combine = spec->combine; in->nulls = spec->nulls; in->key_scale = p->key_scale;
    in->weight_scale = p->weight_scale;
    free(ent); ent = NULL;
    free(cnt); cnt = NULL;
    /* 8. the identity: spec | edge descriptor | vertex descriptor, and the dictionary */
    g->ident_bytes = ident_size(p);
    g->ident = zalloc(g->ident_bytes);
    if (!g->ident) { rc = TR_MATH_MEMORY; goto fail; }
    memcpy(g->ident, spec, 48);
    memcpy(g->ident + 48, &p->edesc, 8);
    g->edesc_at = 56; g->edesc_len = p->edesc;
    rc = tr_scope_key_write(edges, TR_SCOPE_KEY_ALL, g->ident + 56, p->edesc);
    if (rc) goto fail;
    memcpy(g->ident + 56 + p->edesc, &p->vdesc, 8);
    g->vdesc_at = 64 + p->edesc; g->vdesc_len = p->vdesc;
    if (p->has_vertices) {
        rc = tr_scope_key_write(vertices, TR_SCOPE_KEY_ALL, g->ident + g->vdesc_at, p->vdesc);
        if (rc) goto fail;
    }
    g->has_vertices = p->has_vertices;
    g->keys = keys; keys = NULL;
    in->digest[0] = UINT64_C(0xcbf29ce484222325); in->digest[1] = UINT64_C(0x6a09e667f3bcc909);
    digest_bytes(g->ident, g->ident_bytes, in->digest);
    digest_bytes((const unsigned char *)g->keys, n * 8, in->digest);
    in->bytes = tr_budget_add(tr_budget_add(sizeof *g, tr_budget_mul(n, 8)),
                              tr_budget_add(tr_budget_mul(tr_graph_adj_bytes(n, E), directed ? 2 : 1),
                                            tr_budget_add(tr_budget_mul(M, directed ? 4 : 24),
                                                          tr_budget_add(tr_budget_mul(n + 1, 8),
                                                                        tr_budget_mul(g->inc_off[n], 8)))));
    in->bytes = tr_budget_add(in->bytes, g->ident_bytes);
    g->magic = LAYOUT_MAGIC;
    free(p);
    set_error(error, TR_MATH_OK);
    return g;
fail:
    free(keys); free(ktmp); free(slot); free(stmp); free(sid); free(kcount); free(t); free(t2); free(ent); free(ent2);
    free(cnt); free(p);
    layout_release(g);
    set_error(error, rc);
    return NULL;
}

int32_t tr_graph_rationals(uint64_t count, const int64_t *num, const int64_t *den, int64_t *scaled,
                           int64_t *denominator) {
    if ((count && (!num || !den || !scaled)) || !denominator) return TR_MATH_INVALID;
    int64_t D = 1;
    for (uint64_t i = 0; i < count; i++) {                 /* the lcm of the reduced denominators */
        if (den[i] <= 0) return TR_MATH_INVALID;
        uint64_t gg = tr_gcd_u64(tr_abs_u64(num[i]), (uint64_t)den[i]);
        int64_t d = den[i] / (int64_t)gg;                  /* gg >= 1: den[i] > 0 */
        uint64_t h = tr_gcd_u64((uint64_t)D, (uint64_t)d);
        if (!tr_mul_i64(D / (int64_t)h, d, &D)) return TR_MATH_OVERFLOW;
    }
    for (uint64_t i = 0; i < count; i++) {
        uint64_t gg = tr_gcd_u64(tr_abs_u64(num[i]), (uint64_t)den[i]);
        int64_t nn = (int64_t)((__int128)num[i] / (__int128)gg), dd = den[i] / (int64_t)gg;
        __int128 v = (__int128)nn * (D / dd);
        if (v > INT64_MAX || v < INT64_MIN) return TR_MATH_OVERFLOW;
        scaled[i] = (int64_t)v;
    }
    *denominator = D;
    return TR_MATH_OK;
}

/* ------------------------------------------------------------------ requests: checks and needs */
static int bits_canonical(const uint64_t *b, uint64_t n) {
    if (!b) return 0;
    uint64_t nw = tr_graph_words(n);
    if (n % 64 && nw && (b[nw - 1] >> (n % 64))) return 0;
    return 1;
}
int32_t tr_graph_need(const tr_graph_layout *g, int32_t request, int32_t semiring, uint64_t a, uint64_t b, uint32_t c,
                      int32_t target, tr_math_need *need) {
    if (!layout_ok(g) || !need || (target != TR_GRAPH_NEED_CPU && target != TR_GRAPH_NEED_DEVICE)) return TR_MATH_INVALID;
    (void)c;
    tr_graph_view v;
    view_of(g, &v);
    int dev = target == TR_GRAPH_NEED_DEVICE;
    switch (request) {
    case TR_GRAPH_REQ_DEGREES: tr_graph_degrees_base_need(&v, target, need); break;
    case TR_GRAPH_REQ_EXPAND: tr_graph_expand_base_need(&v, target, need); break;
    case TR_GRAPH_REQ_SPMV:
        if (semiring < TR_GRAPH_COUNT || semiring > TR_GRAPH_FLOAT) return TR_MATH_INVALID;
        tr_graph_spmv_base_need(&v, semiring, target, need); break;
    case TR_GRAPH_REQ_GF2:
        tr_graph_zero_need(need);
        need->need[TR_BUDGET_TENSOR] = tr_budget_mul(v.n, v.n);
        need->need[TR_BUDGET_OUTPUT] = tr_budget_mul(v.n, v.n);
        break;
    case TR_GRAPH_REQ_REACH: tr_graph_reach_base_need(&v, a, target, need); break;
    case TR_GRAPH_REQ_CLOSURE: tr_graph_closure_base_need(&v, a, b, target, need); break;
    case TR_GRAPH_REQ_PATHS:
        tr_graph_paths_base_need(&v, target, need);
        if (!dev) need->need[TR_BUDGET_BYTES] = tr_budget_add(need->need[TR_BUDGET_BYTES],
                                                              tr_budget_mul(tr_budget_add(v.entries, v.n), 24));
        break;
    case TR_GRAPH_REQ_ENERGY: tr_graph_energy_base_need(&v, target, need); break;
    case TR_GRAPH_REQ_ITERATE:
        tr_graph_iter_base_need(&v, a, target, need);
        need->need[TR_BUDGET_BYTES] = tr_budget_add(need->need[TR_BUDGET_BYTES], tr_budget_mul(v.n, 16));
        break;
    default: return TR_MATH_INVALID;
    }
    return TR_MATH_OK;
}

/* ------------------------------------------------------------------ GRAPH-001 reductions */
int32_t tr_graph_degrees(const tr_graph_layout *g, int32_t kind, int32_t weights, tr_math_i128 *out,
                         const tr_math_budget *budget, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!layout_ok(g) || (g->info.vertices && !out) || kind < TR_GRAPH_DEG_OUT || kind > TR_GRAPH_DEG_LOOPS ||
        weights < TR_GRAPH_W_EDGE || weights > TR_GRAPH_W_DISTINCT) return TR_MATH_INVALID;
    tr_graph_view v;
    view_of(g, &v);
    tr_math_need need;
    tr_graph_degrees_base_need(&v, TR_GRAPH_NEED_CPU, &need);
    int32_t rc = check_need(budget, &need, refusal);
    if (rc) return rc;
    tr_graph_adj in = tr_graph_pull(&v, TR_GRAPH_FORWARD), outf = tr_graph_pull(&v, TR_GRAPH_BACKWARD);
    for (uint64_t x = 0; x < v.n; x++) {
        __int128 a = 0, b = 0;
        int ok = 1;
        if (kind == TR_GRAPH_DEG_OUT) ok = tr_graph_row_degree(outf, x, weights, 0, &a);
        else if (kind == TR_GRAPH_DEG_IN) ok = tr_graph_row_degree(in, x, weights, 0, &a);
        else if (kind == TR_GRAPH_DEG_LOOPS) ok = tr_graph_row_degree(outf, x, weights, 1, &a);
        else {
            ok = tr_graph_row_degree(outf, x, weights, 0, &a);
            if (ok && v.directed) ok = tr_graph_row_degree(in, x, weights, 0, &b) && tr_graph_add128(a, b, &a);
        }
        if (!ok) return TR_MATH_OVERFLOW;
        out[x] = tr_i128_set(a);
    }
    return TR_MATH_OK;
}

int32_t tr_graph_expand(const tr_graph_layout *g, int32_t dir, const uint64_t *frontier, const uint64_t *exclude,
                        uint64_t *next, uint64_t *count, const tr_math_budget *budget, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!layout_ok(g) || !tr_graph_dir_ok(dir)) return TR_MATH_INVALID;
    tr_graph_view v;
    view_of(g, &v);
    uint64_t nw = tr_graph_words(v.n);
    if (nw && (!frontier || !next || !bits_canonical(frontier, v.n) || (exclude && !bits_canonical(exclude, v.n))))
        return TR_MATH_INVALID;
    tr_math_need need;
    tr_graph_expand_base_need(&v, TR_GRAPH_NEED_CPU, &need);
    int32_t rc = check_need(budget, &need, refusal);
    if (rc) return rc;
    tr_graph_adj push = tr_graph_push(&v, dir);
    for (uint64_t w = 0; w < nw; w++) next[w] = 0;
    for (uint64_t w = 0; w < nw; w++) {
        uint64_t bits = frontier[w];
        while (bits) {
            uint64_t u = w * 64 + (uint64_t)tr_graph_ctz(bits);
            bits &= bits - 1;
            for (uint64_t e = push.off[u]; e < push.off[u + 1]; e++) bit_set(next, push.nbr[e]);
        }
    }
    uint64_t c = 0;
    for (uint64_t w = 0; w < nw; w++) {
        if (exclude) next[w] &= ~exclude[w];
        c += (uint64_t)tr_graph_popc(next[w]);
    }
    if (count) *count = c;
    return TR_MATH_OK;
}

static int32_t spmv_args(const tr_graph_view *v, int32_t semiring, int32_t dir, int32_t weights, const void *x,
                         void *y, double *err) {
    if (semiring < TR_GRAPH_COUNT || semiring > TR_GRAPH_FLOAT || !tr_graph_dir_ok(dir) ||
        weights < TR_GRAPH_W_EDGE || weights > TR_GRAPH_W_DISTINCT) return TR_MATH_INVALID;
    if (v->n && (!x || !y || (semiring == TR_GRAPH_FLOAT && !err))) return TR_MATH_INVALID;
    if (semiring == TR_GRAPH_GF2 && weights == TR_GRAPH_W_EDGE && v->denominator != 1) return TR_MATH_DOMAIN_ERROR;
    return TR_MATH_OK;
}
int32_t tr_graph_spmv(const tr_graph_layout *g, int32_t semiring, int32_t dir, int32_t weights, const void *x,
                      const uint64_t *x_valid, void *y, uint64_t *y_valid, double *err, uint64_t *where,
                      const tr_math_budget *budget, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!layout_ok(g)) return TR_MATH_INVALID;
    tr_graph_view v;
    view_of(g, &v);
    int32_t rc = spmv_args(&v, semiring, dir, weights, x, y, err);
    if (rc) return rc;
    if ((semiring == TR_GRAPH_BOOL || semiring == TR_GRAPH_GF2) && v.n && !bits_canonical((const uint64_t *)x, v.n))
        return TR_MATH_INVALID;
    tr_math_need need;
    tr_graph_spmv_base_need(&v, semiring, TR_GRAPH_NEED_CPU, &need);
    rc = check_need(budget, &need, refusal);
    if (rc) return rc;
    double *coef = NULL;
    if (semiring == TR_GRAPH_FLOAT) {
        coef = zalloc(v.entries * 8);
        if (!coef) return TR_MATH_MEMORY;
        tr_graph_float_coef(&v, dir, weights, coef);
    }
    tr_graph_adj a = tr_graph_pull(&v, dir);
    const uint64_t nw = tr_graph_words(v.n);
    if (y_valid) for (uint64_t w = 0; w < nw; w++) y_valid[w] = 0;
    if (semiring == TR_GRAPH_BOOL || semiring == TR_GRAPH_GF2) for (uint64_t w = 0; w < nw; w++) ((uint64_t *)y)[w] = 0;
    for (uint64_t u = 0; u < v.n; u++) {
        int has = 0, ok = 1;
        switch (semiring) {
        case TR_GRAPH_COUNT: {
            __int128 r;
            ok = tr_graph_row_count(a, u, weights, (const int64_t *)x, x_valid, &r, &has);
            if (ok) ((tr_math_i128 *)y)[u] = tr_i128_set(r);
            break;
        }
        case TR_GRAPH_BOOL:
            if (tr_graph_row_bool(a, u, (const uint64_t *)x, x_valid, &has)) bit_set((uint64_t *)y, u);
            break;
        case TR_GRAPH_GF2:
            if (tr_graph_row_gf2(a, u, weights, (const uint64_t *)x, x_valid, &has)) bit_set((uint64_t *)y, u);
            break;
        case TR_GRAPH_MIN_PLUS: case TR_GRAPH_MAX_PLUS: {
            int64_t r;
            ok = tr_graph_row_tropical(a, u, weights, semiring == TR_GRAPH_MAX_PLUS, (const int64_t *)x, x_valid, &r, &has);
            if (ok) ((int64_t *)y)[u] = has ? r : 0;
            break;
        }
        default: {
            double e;
            ((double *)y)[u] = tr_graph_row_float(a, u, weights, coef, (const double *)x, x_valid, &e, &has);
            err[u] = e;
        }
        }
        if (!ok) { free(coef); if (where) *where = u; return TR_MATH_OVERFLOW; }
        if (has && y_valid) bit_set(y_valid, u);
    }
    free(coef);
    return TR_MATH_OK;
}

int32_t tr_graph_gf2_matrix(const tr_graph_layout *g, int32_t weights, uint64_t *words, uint64_t stride,
                            const tr_math_budget *budget, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!layout_ok(g) || weights < TR_GRAPH_W_EDGE || weights > TR_GRAPH_W_DISTINCT) return TR_MATH_INVALID;
    tr_graph_view v;
    view_of(g, &v);
    if ((v.n && !words) || stride < tr_graph_words(v.n)) return TR_MATH_INVALID;
    if (weights == TR_GRAPH_W_EDGE && v.denominator != 1) return TR_MATH_DOMAIN_ERROR;
    tr_math_need need;
    tr_graph_need(g, TR_GRAPH_REQ_GF2, 0, 0, 0, 0, TR_GRAPH_NEED_CPU, &need);
    int32_t rc = check_need(budget, &need, refusal);
    if (rc) return rc;
    memset(words, 0, (size_t)(v.n * stride * 8));
    tr_graph_adj a = tr_graph_pull(&v, TR_GRAPH_BACKWARD);   /* row u: its out-entries */
    for (uint64_t u = 0; u < v.n; u++) {
        uint32_t prev = TR_GRAPH_NONE;
        for (uint64_t e = a.off[u]; e < a.off[u + 1]; e++) {
            uint32_t x = a.nbr[e];
            uint64_t c;
            if (weights == TR_GRAPH_W_DISTINCT) { if (x == prev) continue; prev = x; c = 1; }
            else c = (a.m[e] & 1) & (weights == TR_GRAPH_W_EDGE ? ((uint64_t)a.w[e] & 1) : 1);
            words[u * stride + (x >> 6)] ^= c << (x & 63);
        }
    }
    return TR_MATH_OK;
}

/* ------------------------------------------------------------------ GRAPH-002: frontier machinery (CPU) */
/* candidates: the distinct push neighbors of the active list (marks: a bitmap, left clear afterwards by the caller) */
static uint64_t candidates(tr_graph_adj push, const uint32_t *act, uint64_t na, uint64_t *marks, uint32_t *cand,
                           uint64_t *work) {
    uint64_t nc = 0;
    for (uint64_t i = 0; i < na; i++) {
        uint32_t u = act[i];
        *work += push.off[u + 1] - push.off[u];
        for (uint64_t e = push.off[u]; e < push.off[u + 1]; e++) {
            uint32_t x = push.nbr[e];
            if (!tr_graph_bit(marks, x)) { bit_set(marks, x); cand[nc++] = x; }
        }
    }
    return nc;
}
static void unmark(uint64_t *marks, const uint32_t *cand, uint64_t nc) {
    for (uint64_t i = 0; i < nc; i++) marks[cand[i] >> 6] &= ~((uint64_t)1 << (cand[i] & 63));
}

int32_t tr_graph_reach(const tr_graph_layout *g, int32_t dir, const uint64_t *sources, int32_t reflexive,
                       uint64_t max_rounds, uint64_t *reached, uint32_t *level, tr_graph_rounds *info,
                       const tr_math_budget *budget, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!layout_ok(g) || !tr_graph_dir_ok(dir) || !info) return TR_MATH_INVALID;
    tr_graph_view v;
    view_of(g, &v);
    const uint64_t n = v.n, nw = tr_graph_words(n);
    if (n && (!sources || !reached || !level || !bits_canonical(sources, n))) return TR_MATH_INVALID;
    tr_math_need need;
    tr_graph_reach_base_need(&v, max_rounds, TR_GRAPH_NEED_CPU, &need);
    int32_t rc = check_need(budget, &need, refusal);
    if (rc) return rc;
    uint32_t *buf[2] = {zalloc(n * 4), zalloc(n * 4)};
    if (!buf[0] || !buf[1]) { free(buf[0]); free(buf[1]); return TR_MATH_MEMORY; }
    tr_graph_adj push = tr_graph_push(&v, dir);
    for (uint64_t w = 0; w < nw; w++) reached[w] = 0;
    for (uint64_t x = 0; x < n; x++) level[x] = TR_GRAPH_NONE;
    uint64_t fn = 0, rounds = 0, work = 0, fmax = 0;
    int cur = 0;
    for (uint64_t x = 0; x < n; x++)
        if (tr_graph_bit(sources, x)) {
            buf[0][fn++] = (uint32_t)x;
            if (reflexive) { bit_set(reached, x); level[x] = 0; }
        }
    fmax = fn;
    while (fn) {
        if (rounds == max_rounds) { free(buf[0]); free(buf[1]); return TR_MATH_NOT_CONVERGED; }
        rounds++;
        uint64_t nn = 0;
        const uint32_t *F = buf[cur];
        uint32_t *N = buf[cur ^ 1];
        for (uint64_t i = 0; i < fn; i++) {
            uint32_t u = F[i];
            work += push.off[u + 1] - push.off[u];
            for (uint64_t e = push.off[u]; e < push.off[u + 1]; e++) {
                uint32_t x = push.nbr[e];
                if (!tr_graph_bit(reached, x)) { bit_set(reached, x); level[x] = (uint32_t)rounds; N[nn++] = x; }
            }
        }
        cur ^= 1;
        fn = nn;
        if (fn > fmax) fmax = fn;
    }
    free(buf[0]); free(buf[1]);
    info->rounds = rounds; info->reached = tr_graph_count_bits(reached, n); info->frontier_max = fmax; info->work = work;
    return TR_MATH_OK;
}

/* the rows of a lane batch: row (base + j) bit x = bit j of R[x] */
static void lanes_to_rows(const uint64_t *R, uint64_t n, uint64_t base, uint64_t lanes, uint64_t *rows, uint64_t stride) {
    for (uint64_t x = 0; x < n; x++) {
        uint64_t b = R[x];
        while (b) {
            uint64_t j = (uint64_t)tr_graph_ctz(b);
            b &= b - 1;
            if (j < lanes) rows[(base + j) * stride + (x >> 6)] |= (uint64_t)1 << (x & 63);
        }
    }
}
int32_t tr_graph_closure(const tr_graph_layout *g, int32_t dir, const uint32_t *sources, uint64_t k, int32_t reflexive,
                         uint64_t max_rounds, uint64_t *rows, uint64_t stride, tr_graph_rounds *info,
                         const tr_math_budget *budget, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!layout_ok(g) || !tr_graph_dir_ok(dir) || !info) return TR_MATH_INVALID;
    tr_graph_view v;
    view_of(g, &v);
    const uint64_t n = v.n;
    if ((k && (!sources || !rows)) || stride < tr_graph_words(n)) return TR_MATH_INVALID;
    for (uint64_t i = 0; i < k; i++) if (sources[i] >= n) return TR_MATH_INVALID;
    tr_math_need need;
    tr_graph_closure_base_need(&v, k, max_rounds, TR_GRAPH_NEED_CPU, &need);
    int32_t rc = check_need(budget, &need, refusal);
    if (rc) return rc;
    uint64_t *R = zalloc(n * 8), *F = zalloc(n * 8), *slot = zalloc(n * 8), *marks = zalloc(tr_graph_bitmap(n));
    uint32_t *A = zalloc(n * 4), *Cn = zalloc(n * 4), *NA = zalloc(n * 4);
    if (!R || !F || !slot || !marks || !A || !Cn || !NA) { rc = TR_MATH_MEMORY; goto done; }
    tr_graph_adj push = tr_graph_push(&v, dir), pull = tr_graph_pull(&v, dir);
    memset(rows, 0, (size_t)(k * stride * 8));
    uint64_t rounds = 0, work = 0, fmax = 0, reached = 0;
    for (uint64_t base = 0; base < k; base += TR_GRAPH_LANES) {
        uint64_t lanes = k - base < TR_GRAPH_LANES ? k - base : TR_GRAPH_LANES, na = 0, r = 0;
        memset(R, 0, (size_t)(n * 8)); memset(F, 0, (size_t)(n * 8));
        for (uint64_t j = 0; j < lanes; j++) {
            uint32_t s = sources[base + j];
            if (!F[s]) A[na++] = s;
            F[s] |= (uint64_t)1 << j;
            if (reflexive) R[s] |= (uint64_t)1 << j;
        }
        if (na > fmax) fmax = na;
        while (na) {
            if (r == max_rounds) { rc = TR_MATH_NOT_CONVERGED; goto done; }
            r++;
            uint64_t nc = candidates(push, A, na, marks, Cn, &work);
            for (uint64_t i = 0; i < nc; i++) {
                uint32_t c = Cn[i];
                work += pull.off[c + 1] - pull.off[c];
                slot[i] = tr_graph_row_lanes(pull, c, F) & ~R[c];
            }
            for (uint64_t i = 0; i < na; i++) F[A[i]] = 0;
            uint64_t nn = 0;
            for (uint64_t i = 0; i < nc; i++) {
                uint32_t c = Cn[i];
                if (slot[i]) { R[c] |= slot[i]; F[c] = slot[i]; NA[nn++] = c; }
            }
            unmark(marks, Cn, nc);
            uint32_t *t = A; A = NA; NA = t;
            na = nn;
            if (na > fmax) fmax = na;
        }
        if (r > rounds) rounds = r;
        lanes_to_rows(R, n, base, lanes, rows, stride);
    }
    for (uint64_t i = 0; i < k; i++) reached += tr_graph_count_bits(rows + i * stride, n);
    info->rounds = rounds; info->reached = reached; info->frontier_max = fmax; info->work = work;
done:
    free(R); free(F); free(slot); free(marks); free(A); free(Cn); free(NA);
    return rc;
}

/* walks: the checks, L (the horizon pre-pass for UNBOUNDED, its scratch budgeted first when a budget is given) and the
 * needs for W limbs */
static int32_t walks_plan(const tr_graph_view *v, int32_t dir, const uint64_t *sources, uint32_t max_length,
                          uint32_t W, int32_t keep_rows, int32_t target, int check, const tr_math_budget *budget,
                          tr_math_refusal *refusal, tr_math_need *need, uint64_t *L) {
    if (!tr_graph_dir_ok(dir) || W < 1 || (v->n && (!sources || !bits_canonical(sources, v->n)))) return TR_MATH_INVALID;
    if (max_length == TR_GRAPH_UNBOUNDED) {
        tr_graph_zero_need(need);
        need->need[TR_BUDGET_BYTES] = tr_graph_horizon_bytes(v->n);
        int32_t rc = check ? check_need(budget, need, refusal) : 0;
        if (rc) return rc;
        uint64_t h = 0;
        rc = tr_graph_horizon(v, dir, sources, &h, NULL);
        if (rc) return rc;
        *L = h;
    } else *L = max_length;
    tr_graph_walks_base_need(v, *L, W, keep_rows, target, need);
    return TR_MATH_OK;
}
int32_t tr_graph_walks_need(const tr_graph_layout *g, int32_t dir, const uint64_t *sources, uint32_t max_length,
                            uint32_t limbs, int32_t keep_rows, int32_t target, tr_math_need *need, uint64_t *lengths) {
    if (!layout_ok(g) || !need || (target != TR_GRAPH_NEED_CPU && target != TR_GRAPH_NEED_DEVICE)) return TR_MATH_INVALID;
    tr_graph_view v;
    view_of(g, &v);
    uint64_t L = 0;
    int32_t rc = walks_plan(&v, dir, sources, max_length, limbs, keep_rows, target, 0, NULL, NULL, need, &L);
    if (rc) return rc;
    if (lengths) *lengths = L + 1;
    return TR_MATH_OK;
}

static void walks_violation(tr_graph_walks_info *info, int32_t kind, uint64_t l, uint64_t vtx) {
    info->status = TR_MATH_OVERFLOW; info->kind = kind; info->violation_length = l; info->violation_vertex = vtx;
}
int32_t tr_graph_walks(const tr_graph_layout *g, int32_t dir, const uint64_t *sources, uint32_t max_length,
                       const uint64_t *bound, uint32_t limbs, uint64_t *rows, uint64_t rows_capacity, uint64_t *totals,
                       uint64_t *lengths, uint64_t lengths_capacity, uint64_t *grand, tr_graph_walks_info *info,
                       const tr_math_budget *budget, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!layout_ok(g) || !info || !bound || !grand) return TR_MATH_INVALID;
    tr_graph_view v;
    view_of(g, &v);
    const uint64_t n = v.n;
    tr_math_need need;
    uint64_t L = 0;
    if (limbs < 1) return TR_MATH_INVALID;
    const uint32_t W = tr_graph_bound_limbs(bound, limbs);
    int32_t rc = walks_plan(&v, dir, sources, max_length, W, rows != NULL, TR_GRAPH_NEED_CPU, 1, budget, refusal, &need, &L);
    if (rc) return rc;
    rc = check_need(budget, &need, refusal);
    if (rc) return rc;
    if (W > TR_GRAPH_MAX_LIMBS) return TR_MATH_UNSUPPORTED;
    const uint64_t lens = L + 1;
    if ((n && !totals) || !lengths || lengths_capacity < lens * W || (rows && rows_capacity < lens * n * W))
        return TR_MATH_INVALID;
    memset(info, 0, sizeof *info);
    info->lengths = lens; info->limbs = W; info->unbounded = max_length == TR_GRAPH_UNBOUNDED;
    info->violation_length = TR_GRAPH_NONE; info->violation_vertex = TR_GRAPH_NONE;
    uint64_t *prev = zalloc(n * W * 8), *cur = zalloc(n * W * 8), *marks = zalloc(tr_graph_bitmap(n));
    uint32_t *A = zalloc(n * 4), *Cn = zalloc(n * 4);
    uint64_t acc[TR_GRAPH_MAX_LIMBS + 2], tmp[TR_GRAPH_MAX_LIMBS + 2], lsum[TR_GRAPH_MAX_LIMBS + 2],
             gsum[TR_GRAPH_MAX_LIMBS + 2];
    if (!prev || !cur || !marks || !A || !Cn) { rc = TR_MATH_MEMORY; goto done; }
    memset(totals, 0, (size_t)(n * W * 8));
    memset(lengths, 0, (size_t)(lens * W * 8));
    if (rows) memset(rows, 0, (size_t)(lens * n * W * 8));
    memset(gsum, 0, sizeof gsum);
    tr_graph_adj push = tr_graph_push(&v, dir), pull = tr_graph_pull(&v, dir);
    const uint64_t one[1] = {1};
    uint64_t na = 0, work = 0, maxbits = 0;
    for (uint64_t x = 0; x < n; x++)
        if (tr_graph_bit(sources, x)) {
            if (!tr_graph_limbs_le(one, 1, bound, W)) { walks_violation(info, TR_GRAPH_V_ROW, 0, x); rc = TR_MATH_OVERFLOW; goto done; }
            prev[x * W] = 1; totals[x * W] = 1;
            if (rows) rows[x * W] = 1;
            A[na++] = (uint32_t)x;
            maxbits = 1;
        }
    memset(lsum, 0, sizeof lsum);
    lsum[0] = na;
    if (!tr_graph_limbs_le(lsum, W + 2, bound, W)) { walks_violation(info, TR_GRAPH_V_LENGTH, 0, TR_GRAPH_NONE); rc = TR_MATH_OVERFLOW; goto done; }
    memcpy(lengths, lsum, (size_t)W * 8);
    tr_graph_limbs_add(gsum, W + 2, lsum, W + 2);
    for (uint64_t l = 1; l <= L && na; l++) {
        uint64_t nc = candidates(push, A, na, marks, Cn, &work), rowv = TR_GRAPH_NONE, totv = TR_GRAPH_NONE;
        for (uint64_t i = 0; i < nc; i++) {
            uint32_t c = Cn[i];
            work += pull.off[c + 1] - pull.off[c];
            tr_graph_row_walk(pull, c, W, prev, acc);
            uint64_t bits = tr_graph_limbs_bits(acc, W + 2);
            if (bits > maxbits) maxbits = bits;
            if (!tr_graph_limbs_le(acc, W + 2, bound, W)) { if (c < rowv) rowv = c; continue; }
            memcpy(cur + (uint64_t)c * W, acc, (size_t)W * 8);
        }
        if (rowv != TR_GRAPH_NONE) { walks_violation(info, TR_GRAPH_V_ROW, l, rowv); rc = TR_MATH_OVERFLOW; goto done; }
        memset(lsum, 0, sizeof lsum);
        for (uint64_t i = 0; i < nc; i++) {
            uint32_t c = Cn[i];
            const uint64_t *x = cur + (uint64_t)c * W;
            memset(tmp, 0, (size_t)(W + 1) * 8);
            memcpy(tmp, totals + (uint64_t)c * W, (size_t)W * 8);
            tr_graph_limbs_add(tmp, W + 1, x, W);
            if (!tr_graph_limbs_le(tmp, W + 1, bound, W)) { if (c < totv) totv = c; continue; }
            memcpy(totals + (uint64_t)c * W, tmp, (size_t)W * 8);
            if (rows) memcpy(rows + (l * n + c) * W, x, (size_t)W * 8);
            tr_graph_limbs_add(lsum, W + 2, x, W);
        }
        if (totv != TR_GRAPH_NONE) { walks_violation(info, TR_GRAPH_V_TOTAL, l, totv); rc = TR_MATH_OVERFLOW; goto done; }
        if (!tr_graph_limbs_le(lsum, W + 2, bound, W)) { walks_violation(info, TR_GRAPH_V_LENGTH, l, TR_GRAPH_NONE); rc = TR_MATH_OVERFLOW; goto done; }
        memcpy(lengths + l * W, lsum, (size_t)W * 8);
        tr_graph_limbs_add(gsum, W + 2, lsum, W + 2);
        for (uint64_t i = 0; i < na; i++) memset(prev + (uint64_t)A[i] * W, 0, (size_t)W * 8);
        unmark(marks, Cn, nc);
        uint64_t *t = prev; prev = cur; cur = t;
        memcpy(A, Cn, (size_t)(nc * 4));
        na = nc;
    }
    if (!tr_graph_limbs_le(gsum, W + 2, bound, W)) { walks_violation(info, TR_GRAPH_V_GRAND, lens, TR_GRAPH_NONE); rc = TR_MATH_OVERFLOW; goto done; }
    memcpy(grand, gsum, (size_t)W * 8);
    info->maxbits = maxbits; info->work = work;
done:
    free(prev); free(cur); free(marks); free(A); free(Cn);
    if (rc == TR_MATH_OVERFLOW) info->maxbits = 0;
    return rc;
}

/* ------------------------------------------------------------------ paths */
typedef struct { int64_t d; uint32_t h, v; } heap_t;
static int heap_less(const heap_t *a, const heap_t *b) {
    if (a->d != b->d) return a->d < b->d;
    if (a->h != b->h) return a->h < b->h;
    return a->v < b->v;
}
static void heap_push(heap_t *hp, uint64_t *n, heap_t x) {
    uint64_t i = (*n)++;
    hp[i] = x;
    while (i && heap_less(&hp[i], &hp[(i - 1) / 2])) { heap_t t = hp[i]; hp[i] = hp[(i - 1) / 2]; hp[(i - 1) / 2] = t; i = (i - 1) / 2; }
}
static heap_t heap_pop(heap_t *hp, uint64_t *n) {
    heap_t top = hp[0];
    hp[0] = hp[--(*n)];
    uint64_t i = 0;
    for (;;) {
        uint64_t l = 2 * i + 1, r = l + 1, m = i;
        if (l < *n && heap_less(&hp[l], &hp[m])) m = l;
        if (r < *n && heap_less(&hp[r], &hp[m])) m = r;
        if (m == i) break;
        heap_t t = hp[i]; hp[i] = hp[m]; hp[m] = t; i = m;
    }
    return top;
}
static int32_t paths_dijkstra(const tr_graph_view *v, int32_t dir, const uint64_t *sources, int64_t *dist,
                              uint64_t *fin, uint32_t *hops, tr_graph_paths_info *info) {
    const uint64_t n = v->n;
    heap_t *hp = zalloc((v->entries + n + 1) * sizeof(heap_t));
    uint64_t *done = zalloc(tr_graph_bitmap(n));
    if (!hp || !done) { free(hp); free(done); return TR_MATH_MEMORY; }
    tr_graph_adj push = tr_graph_push(v, dir);
    uint64_t hn = 0, settled = 0, work = 0;
    for (uint64_t x = 0; x < n; x++)
        if (tr_graph_bit(sources, x)) { dist[x] = 0; hops[x] = 0; bit_set(fin, x); heap_t s = {0, 0, (uint32_t)x}; heap_push(hp, &hn, s); }
    while (hn) {
        heap_t t = heap_pop(hp, &hn);
        if (tr_graph_bit(done, t.v) || t.d != dist[t.v] || t.h != hops[t.v]) continue;
        bit_set(done, t.v);
        settled++;
        work += push.off[t.v + 1] - push.off[t.v];
        for (uint64_t e = push.off[t.v]; e < push.off[t.v + 1]; e++) {
            uint32_t x = push.nbr[e];
            __int128 c = (__int128)t.d + push.w[e];
            if (c > INT64_MAX || c < INT64_MIN) { free(hp); free(done); return TR_MATH_OVERFLOW; }
            int64_t cd = (int64_t)c;
            uint32_t ch = t.h + 1;
            if (!tr_graph_bit(fin, x) || cd < dist[x] || (cd == dist[x] && ch < hops[x])) {
                dist[x] = cd; hops[x] = ch; bit_set(fin, x);
                heap_t s = {cd, ch, x};
                heap_push(hp, &hn, s);
            }
        }
    }
    free(hp); free(done);
    info->rounds = settled; info->work = work;
    return TR_MATH_OK;
}
int32_t tr_graph_paths(const tr_graph_layout *g, int32_t semiring, int32_t algorithm, int32_t dir,
                       const uint64_t *sources, int64_t *dist, uint64_t *reached, uint32_t *hops, uint32_t *pred,
                       uint32_t *cycle, uint64_t cycle_capacity, tr_graph_paths_info *info,
                       const tr_math_budget *budget, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!layout_ok(g) || !info || !tr_graph_dir_ok(dir) || (semiring != TR_GRAPH_MIN_PLUS && semiring != TR_GRAPH_MAX_PLUS) ||
        algorithm < TR_GRAPH_AUTO || algorithm > TR_GRAPH_DIJKSTRA) return TR_MATH_INVALID;
    tr_graph_view v;
    view_of(g, &v);
    const uint64_t n = v.n, nw = tr_graph_words(n);
    if (n && (!sources || !dist || !reached || !hops || !pred || !bits_canonical(sources, n))) return TR_MATH_INVALID;
    const int maxplus = semiring == TR_GRAPH_MAX_PLUS;
    int nonneg = !v.entries || v.wmin >= 0;
    if (algorithm == TR_GRAPH_DIJKSTRA && (maxplus || !nonneg)) return TR_MATH_DOMAIN_ERROR;
    int dijkstra = algorithm == TR_GRAPH_DIJKSTRA || (algorithm == TR_GRAPH_AUTO && !maxplus && nonneg);
    tr_math_need need;
    tr_graph_need(g, TR_GRAPH_REQ_PATHS, semiring, 0, 0, 0, TR_GRAPH_NEED_CPU, &need);
    int32_t rc = check_need(budget, &need, refusal);
    if (rc) return rc;
    memset(info, 0, sizeof *info);
    info->witness = TR_GRAPH_NONE;
    info->algorithm = dijkstra ? TR_GRAPH_DIJKSTRA : TR_GRAPH_BELLMAN_FORD;
    for (uint64_t w = 0; w < nw; w++) reached[w] = 0;
    for (uint64_t x = 0; x < n; x++) { dist[x] = 0; hops[x] = TR_GRAPH_NONE; pred[x] = TR_GRAPH_NONE; }
    tr_graph_adj pull = tr_graph_pull(&v, dir), push = tr_graph_push(&v, dir);
    if (dijkstra) {
        rc = paths_dijkstra(&v, dir, sources, dist, reached, hops, info);
        if (rc) return rc;
    } else {
        int64_t *nd = zalloc(n * 8);
        uint32_t *np = zalloc(n * 4), *A = zalloc(n * 4), *Cn = zalloc(n * 4);
        uint8_t *imp = zalloc(n);
        uint64_t *marks = zalloc(tr_graph_bitmap(n));
        if (!nd || !np || !A || !Cn || !imp || !marks) rc = TR_MATH_MEMORY;
        uint64_t na = 0, round = 0, work = 0;
        if (!rc) {
            for (uint64_t x = 0; x < n; x++)
                if (tr_graph_bit(sources, x)) { bit_set(reached, x); hops[x] = 0; A[na++] = (uint32_t)x; }
        }
        while (!rc && na) {
            round++;
            uint64_t nc = candidates(push, A, na, marks, Cn, &work);
            for (uint64_t i = 0; i < nc && !rc; i++) {
                uint32_t c = Cn[i];
                int im = 0;
                work += pull.off[c + 1] - pull.off[c];
                if (!tr_graph_row_relax(pull, c, maxplus, dist, reached, &nd[i], &np[i], &im)) rc = TR_MATH_OVERFLOW;
                imp[i] = (uint8_t)im;
            }
            if (rc) break;
            uint64_t nn = 0, wit = TR_GRAPH_NONE;
            for (uint64_t i = 0; i < nc; i++) {
                uint32_t c = Cn[i];
                if (!imp[i]) continue;
                dist[c] = nd[i]; bit_set(reached, c); hops[c] = (uint32_t)round;
                pred[c] = np[i];
                A[nn++] = c;
                if (c < wit) wit = c;
            }
            unmark(marks, Cn, nc);
            na = nn;
            if (na && round >= n) {                          /* a change in round n: a reachable improving cycle */
                info->witness = wit;
                info->rounds = round; info->work = work;
                info->status = TR_MATH_UNBOUNDED;
                uint32_t *cyc = cycle && cycle_capacity >= n ? cycle : zalloc(n * 4);
                uint64_t len = cyc ? tr_graph_cycle(pred, n, wit, cyc, n) : 0;
                __int128 cw = 0;
                if (len) tr_graph_cycle_weight(&v, dir, maxplus, cyc, len, &cw);
                if (cyc != cycle) free(cyc);
                info->cycle_length = len;
                info->cycle_weight = tr_i128_set(cw);
                rc = TR_MATH_UNBOUNDED;
                break;
            }
        }
        if (!rc) { info->rounds = round; info->work = work; }
        free(nd); free(np); free(A); free(Cn); free(imp); free(marks);
        if (rc) return rc;
    }
    for (uint64_t x = 0; x < n; x++) pred[x] = tr_graph_row_pred(pull, x, dist, reached, hops);
    info->reached = tr_graph_count_bits(reached, n);
    return TR_MATH_OK;
}

/* ------------------------------------------------------------------ energy */
int32_t tr_graph_energy(const tr_graph_layout *g, const int64_t *x, uint64_t *out, const tr_math_budget *budget,
                        tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!layout_ok(g) || !out || (g->info.vertices && !x)) return TR_MATH_INVALID;
    tr_graph_view v;
    view_of(g, &v);
    tr_math_need need;
    tr_graph_energy_base_need(&v, TR_GRAPH_NEED_CPU, &need);
    int32_t rc = check_need(budget, &need, refusal);
    if (rc) return rc;
    uint64_t acc[TR_GRAPH_ENERGY_LIMBS] = {0}, t[TR_GRAPH_ENERGY_LIMBS];
    for (uint64_t e = 0; e < v.edges; e++) {
        tr_graph_energy_term(v.e_m[e], v.e_w[e], x[v.e_tail[e]], x[v.e_head[e]], t);
        tr_graph_energy_add(acc, t);
    }
    memcpy(out, acc, sizeof acc);
    return TR_MATH_OK;
}

/* ------------------------------------------------------------------ PageRank and power iteration (binary64) */
static int32_t iter_args(const tr_graph_view *v, int32_t dir, double tol, uint64_t max_iter, const uint64_t *start) {
    if (!tr_graph_dir_ok(dir) || !(tol >= 0) || max_iter < 1) return TR_MATH_INVALID;
    if (start && !bits_canonical(start, v->n)) return TR_MATH_INVALID;
    if (v->n && start && !tr_graph_count_bits(start, v->n)) return TR_MATH_INVALID;
    return TR_MATH_OK;
}
static void uniform_on(double *x, uint64_t n, const uint64_t *set) {
    uint64_t c = set ? tr_graph_count_bits(set, n) : n;
    double q = 1.0 / (double)c;
    for (uint64_t i = 0; i < n; i++) x[i] = (!set || tr_graph_bit(set, i)) ? q : 0.0;
}
static double blocked(const double *x, const double *y, const uint64_t *mask, uint64_t n, double *part) {
    uint64_t nb = tr_graph_nblocks(n);
    for (uint64_t b = 0; b < nb; b++) part[b] = tr_graph_block(x, y, mask, n, b);
    return tr_graph_blocks_total(part, nb);
}
int32_t tr_graph_pagerank(const tr_graph_layout *g, int32_t dir, double damping, const uint64_t *teleport,
                          const uint64_t *start, double tol, uint64_t max_iter, double *rank, tr_graph_iter_info *info,
                          const tr_math_budget *budget, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!layout_ok(g) || !info) return TR_MATH_INVALID;
    tr_graph_view v;
    view_of(g, &v);
    const uint64_t n = v.n;
    int32_t rc = iter_args(&v, dir, tol, max_iter, start);
    if (rc) return rc;
    if (!(damping > 0 && damping <= 1) || !n || !rank) return TR_MATH_INVALID;
    if (teleport && (!bits_canonical(teleport, n) || !tr_graph_count_bits(teleport, n))) return TR_MATH_INVALID;
    tr_math_need need;
    tr_graph_need(g, TR_GRAPH_REQ_ITERATE, 0, max_iter, 0, 0, TR_GRAPH_NEED_CPU, &need);
    rc = check_need(budget, &need, refusal);
    if (rc) return rc;
    memset(info, 0, sizeof *info);
    double *coef = zalloc(v.entries * 8), *r = zalloc(n * 8), *rn = zalloc(n * 8),
           *part = zalloc(tr_graph_nblocks(n) * 8);
    uint64_t *dang = zalloc(tr_graph_bitmap(n)), nd = 0;
    __int128 *W = zalloc(n * 16);
    if (!coef || !r || !rn || !part || !dang || !W) { rc = TR_MATH_MEMORY; goto done; }
    rc = tr_graph_pagerank_coef(&v, dir, W, coef, dang, &nd);
    if (rc) goto done;
    info->dangling = nd;
    uniform_on(r, n, start ? start : teleport);
    const double t = (double)(teleport ? tr_graph_count_bits(teleport, n) : n);
    tr_graph_adj pull = tr_graph_pull(&v, dir);
    rc = TR_MATH_NOT_CONVERGED;
    for (uint64_t k = 1; k <= max_iter; k++) {
        double D = blocked(r, NULL, dang, n, part), S = blocked(r, NULL, NULL, n, part);
        double tele = tr_graph_pr_tele(damping, D, S, t);
        for (uint64_t x = 0; x < n; x++)
            rn[x] = tr_graph_pr_update(damping, tr_graph_row_dot(pull, x, coef, r), tele, !teleport || tr_graph_bit(teleport, x));
        double delta = blocked(rn, r, NULL, n, part);
        double *tmp = r; r = rn; rn = tmp;
        info->iterations = k; info->delta = delta;
        if (delta <= tol) { rc = TR_MATH_OK; break; }
    }
    if (rc) { info->status = rc; goto done; }
    memcpy(rank, r, (size_t)(n * 8));
    info->value = blocked(r, NULL, NULL, n, part);
    info->converged = 1;
    rc = tr_graph_pagerank_certify(&v, dir, damping, teleport, r, W, info);
done:
    free(coef); free(r); free(rn); free(part); free(dang); free(W);
    return rc;
}
int32_t tr_graph_power(const tr_graph_layout *g, int32_t dir, const uint64_t *start, double tol, uint64_t max_iter,
                       double *vec, tr_graph_iter_info *info, const tr_math_budget *budget, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!layout_ok(g) || !info) return TR_MATH_INVALID;
    tr_graph_view v;
    view_of(g, &v);
    const uint64_t n = v.n;
    int32_t rc = iter_args(&v, dir, tol, max_iter, start);
    if (rc) return rc;
    if (!n || !vec) return TR_MATH_INVALID;
    tr_math_need need;
    tr_graph_need(g, TR_GRAPH_REQ_ITERATE, 0, max_iter, 0, 0, TR_GRAPH_NEED_CPU, &need);
    rc = check_need(budget, &need, refusal);
    if (rc) return rc;
    memset(info, 0, sizeof *info);
    double *coef = zalloc(v.entries * 8), *x = zalloc(n * 8), *y = zalloc(n * 8), *part = zalloc(tr_graph_nblocks(n) * 8);
    if (!coef || !x || !y || !part) { rc = TR_MATH_MEMORY; goto done; }
    rc = tr_graph_power_coef(&v, dir, coef);
    if (rc) goto done;
    uniform_on(x, n, start);
    tr_graph_adj pull = tr_graph_pull(&v, dir);
    rc = TR_MATH_NOT_CONVERGED;
    for (uint64_t k = 1; k <= max_iter; k++) {
        for (uint64_t i = 0; i < n; i++) y[i] = tr_graph_row_dot(pull, i, coef, x);
        double S = blocked(y, NULL, NULL, n, part);
        if (S == 0) { rc = TR_MATH_NO_SOLUTION; info->iterations = k; break; }
        if (!(S < INFINITY)) { rc = TR_MATH_OVERFLOW; info->iterations = k; break; }
        for (uint64_t i = 0; i < n; i++) y[i] = y[i] / S;
        double delta = blocked(y, x, NULL, n, part);
        double *tmp = x; x = y; y = tmp;
        info->iterations = k; info->delta = delta; info->value = S;
        if (delta <= tol) { rc = TR_MATH_OK; break; }
    }
    if (rc) { info->status = rc; goto done; }
    memcpy(vec, x, (size_t)(n * 8));
    info->converged = 1;
    rc = tr_graph_power_certify(&v, dir, x, info->value, info);
done:
    free(coef); free(x); free(y); free(part);
    return rc;
}
