/* TOPO-001 / TOPO-002: declared occupancy complexes, local topology, chain complexes, homology and persistence over a
 * declared field; see math_topo.h (the normative contract).
 * SPDX-License-Identifier: Apache-2.0 */
#include "math_topo.h"
#include "mathematics.h"
#include "math_domain.h"
#include "math_handles.h"
#include "math_carriers.h"
#include "math_field.h"
#include "math_certificates.h"
#include <stdlib.h>
#include <string.h>

TR_MATH_ABI(tr_topo_grid, TR_TOPO_ABI);        /* the ctypes boundary of TOPO-001 / TOPO-002 (math_handles.h) */
TR_MATH_ABI(tr_topo_rows, TR_TOPO_ABI);
TR_MATH_ABI(tr_topo_counts, TR_TOPO_ABI);
TR_MATH_ABI(tr_topo_view, TR_TOPO_ABI);
TR_MATH_ABI(tr_topo_info, TR_TOPO_ABI);
TR_MATH_ABI(tr_topo_local, TR_TOPO_ABI);
TR_MATH_ABI(tr_topo_local_totals, TR_TOPO_ABI);
TR_MATH_ABI(tr_topo_adjacency, TR_TOPO_ABI);
TR_MATH_ABI(tr_topo_dd, TR_TOPO_ABI);
TR_MATH_ABI(tr_topo_field, TR_TOPO_ABI);
TR_MATH_ABI(tr_topo_filtration, TR_TOPO_ABI);
TR_MATH_ABI(tr_topo_homology_info, TR_TOPO_ABI);
TR_MATH_ABI(tr_topo_class, TR_TOPO_ABI);
TR_MATH_ABI(tr_topo_persistence_info, TR_TOPO_ABI);
TR_MATH_ABI(tr_topo_pair, TR_TOPO_ABI);
TR_MATH_ABI(tr_topo_chain_info, TR_TOPO_ABI);
TR_MATH_ABI(tr_topo_states, TR_TOPO_ABI);

_Static_assert(sizeof(tr_topo_grid) == 72, "tr_topo_grid layout");
_Static_assert(sizeof(tr_topo_rows) == 64, "tr_topo_rows layout");
_Static_assert(sizeof(tr_topo_counts) == 168, "tr_topo_counts layout");
_Static_assert(sizeof(tr_topo_view) == 264, "tr_topo_view layout");
_Static_assert(sizeof(tr_topo_info) == 264, "tr_topo_info layout");
_Static_assert(sizeof(tr_topo_local) == 8, "tr_topo_local layout");
_Static_assert(sizeof(tr_topo_local_totals) == 80, "tr_topo_local_totals layout");
_Static_assert(sizeof(tr_topo_adjacency) == 32, "tr_topo_adjacency layout");
_Static_assert(sizeof(tr_topo_dd) == 184, "tr_topo_dd layout");
_Static_assert(sizeof(tr_topo_field) == 24, "tr_topo_field layout");
_Static_assert(sizeof(tr_topo_filtration) == 32, "tr_topo_filtration layout");
_Static_assert(sizeof(tr_topo_homology_info) == 272, "tr_topo_homology_info layout");
_Static_assert(sizeof(tr_topo_class) == 32, "tr_topo_class layout");
_Static_assert(sizeof(tr_topo_persistence_info) == 336, "tr_topo_persistence_info layout");
_Static_assert(sizeof(tr_topo_pair) == 40, "tr_topo_pair layout");
_Static_assert(sizeof(tr_topo_chain_info) == 152, "tr_topo_chain_info layout");
_Static_assert(sizeof(tr_topo_states) == 240, "tr_topo_states layout");

#define COMPLEX_MAGIC UINT64_C(0x544f504f434f4d50)   /* "TOPOCOMP" */
#define CHAIN_MAGIC UINT64_C(0x544f504f43484149)     /* "TOPOCHAI" */
#define HOMOLOGY_MAGIC UINT64_C(0x544f504f484f4d4f)  /* "TOPOHOMO" */
#define PERSIST_MAGIC UINT64_C(0x544f504f50455253)   /* "TOPOPERS" */
#define MAXK (TR_TOPO_MAX_CHAIN + 1)

/* ------------------------------------------------------------------ small helpers */
static void set_error(int32_t *e, int32_t v) { if (e) *e = v; }
static void *zalloc(uint64_t bytes) {
    if (bytes > (uint64_t)SIZE_MAX) return NULL;
    return calloc(1, bytes ? (size_t)bytes : 1);
}
static const tr_math_budget *budget_or(const tr_math_budget *b, tr_math_budget *d) {
    if (b) return b;
    tr_budget_default(d);
    return d;
}
static void need_zero(tr_math_need *n) { for (int i = 0; i < TR_BUDGET_RESOURCES; i++) n->need[i] = 0; }
static void refuse(tr_math_refusal *r, int32_t resource, uint64_t need, uint64_t limit) {
    if (r) { r->resource = resource; r->refused = 1; r->need = need; r->limit = limit; }
}
static uint64_t popcount_words(const uint64_t *w, uint64_t n) {
    uint64_t c = 0;
    for (uint64_t i = 0; i < n; i++) c += (uint64_t)__builtin_popcountll(w[i]);
    return c;
}
static uint64_t u64max(uint64_t a, uint64_t b) { return a > b ? a : b; }

/* ------------------------------------------------------------------ the complex */
struct tr_topo_complex_s {
    uint64_t magic;
    tr_topo_grid grid;
    tr_topo_view v;                   /* geometry and pointers into the arrays below */
    tr_topo_counts counts;
    uint64_t *occupancy, *rankdir;    /* rankdir[w] = occupied positions in words 0 .. w-1 */
    uint64_t *cells[4];
    uint32_t *face[4];
    int8_t *sign[4];
    int64_t *vmin, *vmax;             /* per top cell (has_value) */
    tr_cert_record *record;           /* FND-007: the scope's coordinate and value columns, exact selection */
    uint64_t bytes;
    int32_t backend, has_value;
};

static int complex_ok(const tr_topo_complex *c) { return c && c->magic == COMPLEX_MAGIC; }

void tr_topo_free(tr_topo_complex *c) {
    if (!complex_ok(c)) return;
    free(c->occupancy); free(c->rankdir); free(c->vmin); free(c->vmax);
    for (int k = 0; k < 4; k++) { free(c->cells[k]); free(c->face[k]); free(c->sign[k]); }
    if (c->record) tr_cert_record_free(c->record);
    c->magic = 0;
    free(c);
}
static tr_topo_complex *complex_fail(tr_topo_complex *c, int32_t *error, int32_t rc) {
    if (c) { c->magic = COMPLEX_MAGIC; tr_topo_free(c); }
    set_error(error, rc);
    return NULL;
}

static int32_t grid_check_view(const tr_scope_view *s, const tr_topo_grid *g, tr_topo_view *v) {
    return tr_topo_grid_check_view(s, g, v);
}
int32_t tr_topo_grid_check(const tr_scope *scope, const tr_topo_grid *g) {
    tr_scope_view s;
    tr_topo_view v;
    if (!scope || tr_scope_get_view(scope, &s)) return TR_MATH_INVALID;
    return grid_check_view(&s, g, &v);
}

static void scan_need(const tr_scope_view *s, const tr_topo_grid *g, const tr_topo_view *v, int32_t device,
                      tr_math_need *need) {
    tr_topo_scan_need(s, g, v, device, need);
}
int32_t tr_topo_scan_needs(const tr_scope *scope, const tr_topo_grid *g, int32_t device, tr_math_need *need) {
    tr_scope_view s;
    tr_topo_view v;
    if (!scope || !need || tr_scope_get_view(scope, &s)) return TR_MATH_INVALID;
    int32_t rc = grid_check_view(&s, g, &v);
    if (rc) return rc;
    scan_need(&s, g, &v, device, need);
    return TR_MATH_OK;
}
static void complex_need(const tr_topo_grid *g, const tr_topo_counts *c, uint64_t record_bytes, tr_math_need *need) {
    need_zero(need);
    need->need[TR_BUDGET_VARIABLES] = (uint64_t)g->dim;
    need->need[TR_BUDGET_STATES] = c->positions;
    need->need[TR_BUDGET_OUTPUT] = tr_topo_cells_total(c->n);
    need->need[TR_BUDGET_TENSOR] = tr_topo_incidences(c->n);
    need->need[TR_BUDGET_BYTES] = tr_budget_add(tr_topo_complex_bytes(c, g->value >= 0), record_bytes);
}
/* the FND-007 record of the scope's coordinate and value columns with the exact selection (it budgets itself) */
static tr_cert_record *scope_record(const tr_scope *scope, const tr_topo_grid *g, const tr_math_budget *b,
                                    tr_math_refusal *refusal, int32_t *rc) {
    int32_t cols[4], ncols = 0;
    for (int32_t a = 0; a < g->dim; a++) cols[ncols++] = g->col[a];
    if (g->value >= 0) {
        int dup = 0;
        for (int32_t i = 0; i < ncols; i++) dup |= cols[i] == g->value;
        if (!dup) cols[ncols++] = g->value;
    }
    *rc = 0;
    tr_cert_record *r = tr_cert_record_scope(scope, cols, ncols, TR_CERT_RECORD_SELECTION, NULL, 0, b, refusal, rc);
    if (!r && !*rc) *rc = TR_MATH_MEMORY;
    return r;
}
int32_t tr_topo_build_needs(const tr_scope *scope, const tr_topo_grid *g, const tr_topo_counts *c, int32_t device,
                            tr_math_need *need) {
    tr_scope_view s;
    tr_topo_view v;
    if (!scope || !c || !need || tr_scope_get_view(scope, &s)) return TR_MATH_INVALID;
    int32_t rc = grid_check_view(&s, g, &v);
    if (rc) return rc;
    tr_math_budget all;
    for (int i = 0; i < TR_BUDGET_RESOURCES; i++) all.limit[i] = UINT64_MAX;
    tr_cert_record *r = scope_record(scope, g, &all, NULL, &rc);       /* measured: FND-007 knows its own size */
    if (!r) return rc;
    uint64_t record_bytes = tr_cert_record_bytes(r);
    tr_cert_record_free(r);
    complex_need(g, c, record_bytes, need);
    if (device) need->need[TR_BUDGET_BYTES] = tr_topo_build_device_bytes(&s, g, &v, c);
    return TR_MATH_OK;
}

/* the occupancy pass over the selected rows (valued: the positions with a present value, or NULL) */
static void scan_rows(const tr_scope_view *s, const tr_topo_grid *g, const tr_topo_view *v, uint64_t *occ,
                      uint64_t *valued, tr_topo_rows *rows) {
    memset(rows, 0, sizeof *rows);
    for (uint64_t r = 0; r < s->rows; r++) {
        uint64_t p = 0;
        int st = tr_topo_row(s, g, v, r, &p);
        if (st == TR_TOPO_ROW_UNSELECTED) continue;
        rows->selected++;
        if (st == TR_TOPO_ROW_NULL) { rows->null_rows++; continue; }
        if (st == TR_TOPO_ROW_OUTSIDE) { rows->outside_rows++; continue; }
        rows->valid_rows++;
        occ[p >> 6] |= (uint64_t)1 << (p & 63);
        if (g->value >= 0) {
            if (tr_scope_present(s, g->value, r)) valued[p >> 6] |= (uint64_t)1 << (p & 63);
            else rows->value_nulls++;
        }
    }
    rows->occupied = popcount_words(occ, v->words);
    rows->duplicates = rows->valid_rows - rows->occupied;
    rows->valued = valued ? popcount_words(valued, v->words) : 0;
}
/* the cells and boundary cells per dimension: every candidate of the Khalimsky grid, in increasing L */
static void scan_cells(const tr_topo_view *v, uint64_t *n, uint64_t *boundary) {
    for (int k = 0; k < 4; k++) { n[k] = 0; boundary[k] = 0; }
    uint64_t c[3];
    for (c[2] = 0; c[2] < v->kdim[2]; c[2]++)
        for (c[1] = 0; c[1] < v->kdim[1]; c[1]++)
            for (c[0] = 0; c[0] < v->kdim[0]; c[0]++) {
                int32_t m;
                uint32_t inside, pat = tr_topo_pattern(v, c, &m, &inside);
                if (!pat) continue;
                int32_t k = (int32_t)((c[0] & 1) + (c[1] & 1) + (c[2] & 1));
                n[k]++;
                if (tr_topo_is_boundary(m, pat)) boundary[k]++;
            }
}
static void counts_finish(const tr_topo_grid *g, const tr_topo_view *v, tr_topo_counts *c) {
    c->positions = v->positions;
    c->candidates = v->candidates;
    c->euler = (int64_t)c->n[0] - (int64_t)c->n[1] + (int64_t)c->n[2] - (int64_t)c->n[3];
    c->boundary_euler = (int64_t)c->boundary[0] - (int64_t)c->boundary[1] + (int64_t)c->boundary[2] -
                        (int64_t)c->boundary[3];
    c->dim = g->dim;
    c->adjacency = g->adjacency;
}

int32_t tr_topo_count(const tr_scope *scope, const tr_topo_grid *g, const tr_math_budget *budget, tr_topo_counts *out,
                      tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    tr_scope_view s;
    tr_topo_view v;
    if (!scope || !out || tr_scope_get_view(scope, &s)) return TR_MATH_INVALID;
    int32_t rc = grid_check_view(&s, g, &v);
    if (rc) return rc;
    tr_math_need need;
    tr_math_budget dflt;
    scan_need(&s, g, &v, 0, &need);
    rc = tr_budget_check(budget_or(budget, &dflt), &need, refusal);
    if (rc) return rc;
    uint64_t *occ = zalloc(v.words * 8), *valued = g->value >= 0 ? zalloc(v.words * 8) : NULL;
    if (!occ || (g->value >= 0 && !valued)) { free(occ); free(valued); return TR_MATH_MEMORY; }
    tr_topo_counts c;
    memset(&c, 0, sizeof c);
    scan_rows(&s, g, &v, occ, valued, &c.rows);
    v.occupancy = occ;
    scan_cells(&v, c.n, c.boundary);
    counts_finish(g, &v, &c);
    free(occ); free(valued);
    *out = c;
    return TR_MATH_OK;
}

/* the arrays of a complex (cells, incidences) from its occupancy and counts */
static int32_t complex_arrays(tr_topo_complex *cx) {
    tr_topo_view *v = &cx->v;
    for (int32_t k = 0; k <= v->dim; k++) {
        cx->cells[k] = zalloc(cx->counts.n[k] * 8);
        if (!cx->cells[k]) return TR_MATH_MEMORY;
        if (k >= 1) {
            cx->face[k] = zalloc(cx->counts.n[k] * 2 * (uint64_t)k * 4);
            cx->sign[k] = zalloc(cx->counts.n[k] * 2 * (uint64_t)k);
            if (!cx->face[k] || !cx->sign[k]) return TR_MATH_MEMORY;
        }
    }
    return TR_MATH_OK;
}
static void complex_bind(tr_topo_complex *cx) {
    tr_topo_view *v = &cx->v;
    v->occupancy = cx->occupancy;
    for (int32_t k = 0; k < 4; k++) {
        v->n[k] = cx->counts.n[k];
        v->cells[k] = cx->cells[k];
        v->face[k] = cx->face[k];
        v->sign[k] = cx->sign[k];
    }
}
static void fill_cells(tr_topo_complex *cx) {
    const tr_topo_view *v = &cx->v;
    uint64_t at[4] = {0, 0, 0, 0}, c[3], L = 0;
    for (c[2] = 0; c[2] < v->kdim[2]; c[2]++)
        for (c[1] = 0; c[1] < v->kdim[1]; c[1]++)
            for (c[0] = 0; c[0] < v->kdim[0]; c[0]++, L++) {
                int32_t m;
                uint32_t inside;
                if (!tr_topo_pattern(v, c, &m, &inside)) continue;
                int32_t k = (int32_t)((c[0] & 1) + (c[1] & 1) + (c[2] & 1));
                cx->cells[k][at[k]++] = L;
            }
}
static void fill_incidences(tr_topo_complex *cx) {
    const tr_topo_view *v = &cx->v;
    for (int32_t k = 1; k <= v->dim; k++)
        for (uint64_t j = 0; j < cx->counts.n[k]; j++) {
            uint64_t c[3], face[6];
            int8_t sign[6];
            uint64_t L = cx->cells[k][j];
            tr_topo_decode(v, L, c);
            tr_topo_faces(v, L, c, face, sign);
            for (int32_t s = 0; s < 2 * k; s++) {
                uint64_t f = tr_topo_find(cx->cells[k - 1], cx->counts.n[k - 1], face[s]);
                cx->face[k][(uint64_t)2 * (uint64_t)k * j + (uint64_t)s] = (uint32_t)f;   /* always found (closure) */
                cx->sign[k][(uint64_t)2 * (uint64_t)k * j + (uint64_t)s] = sign[s];
            }
        }
}
static void fill_rankdir(tr_topo_complex *cx) {
    uint64_t acc = 0;
    for (uint64_t w = 0; w < cx->v.words; w++) { cx->rankdir[w] = acc; acc += (uint64_t)__builtin_popcountll(cx->occupancy[w]); }
}
static uint64_t top_index(const tr_topo_complex *cx, uint64_t p) {
    uint64_t w = p >> 6, bit = p & 63;
    uint64_t below = bit ? (cx->occupancy[w] & (((uint64_t)1 << bit) - 1)) : 0;
    return cx->rankdir[w] + (uint64_t)__builtin_popcountll(below);
}
static int32_t fill_values(tr_topo_complex *cx, const tr_scope_view *s) {
    uint64_t n = cx->counts.n[cx->grid.dim];
    cx->vmin = zalloc(n * 8);
    cx->vmax = zalloc(n * 8);
    if (!cx->vmin || !cx->vmax) return TR_MATH_MEMORY;
    for (uint64_t i = 0; i < n; i++) { cx->vmin[i] = INT64_MAX; cx->vmax[i] = INT64_MIN; }
    for (uint64_t r = 0; r < s->rows; r++) {
        uint64_t p = 0;
        if (tr_topo_row(s, &cx->grid, &cx->v, r, &p) != TR_TOPO_ROW_INSIDE) continue;
        if (!tr_scope_present(s, cx->grid.value, r)) continue;
        int64_t x = tr_scope_value(s, cx->grid.value, r);
        uint64_t i = top_index(cx, p);
        if (x < cx->vmin[i]) cx->vmin[i] = x;
        if (x > cx->vmax[i]) cx->vmax[i] = x;
    }
    return TR_MATH_OK;
}
tr_topo_complex *tr_topo_build(const tr_scope *scope, const tr_topo_grid *g, const tr_math_budget *budget,
                               tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    tr_scope_view s;
    tr_topo_view v;
    if (!scope || tr_scope_get_view(scope, &s)) return complex_fail(NULL, error, TR_MATH_INVALID);
    int32_t rc = grid_check_view(&s, g, &v);
    if (rc) return complex_fail(NULL, error, rc);
    tr_math_budget dflt;
    const tr_math_budget *b = budget_or(budget, &dflt);
    tr_math_need need;
    scan_need(&s, g, &v, 0, &need);
    rc = tr_budget_check(b, &need, refusal);
    if (rc) return complex_fail(NULL, error, rc);
    tr_topo_complex *cx = zalloc(sizeof *cx);
    if (!cx) return complex_fail(NULL, error, TR_MATH_MEMORY);
    cx->record = scope_record(scope, g, b, refusal, &rc);
    if (!cx->record) return complex_fail(cx, error, rc);
    cx->grid = *g;
    cx->v = v;
    cx->has_value = g->value >= 0;
    cx->backend = TR_TOPO_BY_CPU;
    cx->occupancy = zalloc(v.words * 8);
    cx->rankdir = zalloc(v.words * 8);
    uint64_t *valued = cx->has_value ? zalloc(v.words * 8) : NULL;
    if (!cx->occupancy || !cx->rankdir || (cx->has_value && !valued)) { free(valued); return complex_fail(cx, error, TR_MATH_MEMORY); }
    scan_rows(&s, g, &cx->v, cx->occupancy, valued, &cx->counts.rows);
    free(valued);
    cx->v.occupancy = cx->occupancy;
    scan_cells(&cx->v, cx->counts.n, cx->counts.boundary);
    counts_finish(g, &cx->v, &cx->counts);
    complex_need(g, &cx->counts, tr_cert_record_bytes(cx->record), &need);   /* data-dependent: after the scan */
    rc = tr_budget_check(b, &need, refusal);
    if (rc) return complex_fail(cx, error, rc);
    rc = complex_arrays(cx);
    if (rc) return complex_fail(cx, error, rc);
    fill_cells(cx);
    complex_bind(cx);
    fill_incidences(cx);
    fill_rankdir(cx);
    if (cx->has_value) {
        rc = fill_values(cx, &s);
        if (rc) return complex_fail(cx, error, rc);
    }
    cx->bytes = tr_budget_add(tr_topo_complex_bytes(&cx->counts, cx->has_value), tr_cert_record_bytes(cx->record));
    cx->magic = COMPLEX_MAGIC;
    set_error(error, TR_MATH_OK);
    return cx;
}

/* ------------------------------------------------------------------ adoption of a device build */
static int32_t adopt_check(const tr_scope_view *s, const tr_topo_grid *g, const tr_topo_view *v, const tr_topo_states *st,
                           uint64_t *boundary) {
    const int32_t dim = g->dim;
    if (st->dim != dim || st->has_value != (g->value >= 0) || st->positions != v->positions || st->words != v->words ||
        (v->words && !st->occupancy)) return TR_MATH_INVALID;
    for (int32_t k = 0; k < 4; k++) {
        if (k > dim && st->n[k]) return TR_MATH_INVALID;
        if (k <= dim && st->n[k] && !st->cells[k]) return TR_MATH_INVALID;
        if (k >= 1 && k <= dim && st->n[k] && (!st->face[k] || !st->sign[k])) return TR_MATH_INVALID;
        if (st->n[k] > UINT32_MAX) return TR_MATH_INVALID;
    }
    if (st->has_value && st->n[dim] && (!st->vmin || !st->vmax)) return TR_MATH_INVALID;
    /* the occupancy: canonical, as many bits as top cells */
    if (v->words && (v->positions & 63) && (st->occupancy[v->words - 1] >> (v->positions & 63))) return TR_MATH_INVALID;
    if (popcount_words(st->occupancy, v->words) != st->n[dim]) return TR_MATH_INVALID;
    /* the row accounting */
    const tr_topo_rows *r = &st->rows;
    if (r->selected != s->selected || r->null_rows + r->outside_rows + r->valid_rows != r->selected ||
        r->occupied != st->n[dim] || r->valid_rows < r->occupied || r->duplicates != r->valid_rows - r->occupied ||
        r->value_nulls > r->valid_rows || (!st->has_value && (r->value_nulls || r->valued)) || r->valued > r->occupied)
        return TR_MATH_INVALID;
    tr_topo_view w = *v;
    w.occupancy = st->occupancy;
    for (int32_t k = 0; k < 4; k++) { w.n[k] = st->n[k]; w.cells[k] = st->cells[k]; w.face[k] = st->face[k]; w.sign[k] = st->sign[k]; }
    for (int32_t k = 0; k < 4; k++) boundary[k] = 0;
    /* the cells: increasing, of their dimension, in the complex (top cells: occupied); boundary counts */
    for (int32_t k = 0; k <= dim; k++)
        for (uint64_t j = 0; j < st->n[k]; j++) {
            uint64_t L = st->cells[k][j], c[3];
            if (L >= v->candidates || (j && L <= st->cells[k][j - 1])) return TR_MATH_INVALID;
            if (tr_topo_decode(&w, L, c) != k) return TR_MATH_INVALID;
            int32_t m;
            uint32_t inside, pat = tr_topo_pattern(&w, c, &m, &inside);
            if (!pat) return TR_MATH_INVALID;
            if (tr_topo_is_boundary(m, pat)) boundary[k]++;
        }
    /* the incidences: every slot holds the face of its formula with its sign */
    for (int32_t k = 1; k <= dim; k++)
        for (uint64_t j = 0; j < st->n[k]; j++) {
            uint64_t L = st->cells[k][j], c[3], face[6];
            int8_t sign[6];
            tr_topo_decode(&w, L, c);
            tr_topo_faces(&w, L, c, face, sign);
            for (int32_t q = 0; q < 2 * k; q++) {
                uint64_t at = (uint64_t)2 * (uint64_t)k * j + (uint64_t)q;
                uint64_t f = st->face[k][at];
                if (f >= st->n[k - 1] || st->cells[k - 1][f] != face[q] || st->sign[k][at] != sign[q]) return TR_MATH_INVALID;
            }
        }
    /* every cell below the top is a face of a cell one dimension up: the lists are the closure, nothing more */
    for (int32_t k = 0; k < dim; k++) {
        uint64_t words = (st->n[k] + 63) / 64;
        uint64_t *seen = zalloc(words * 8);
        if (!seen) return TR_MATH_MEMORY;
        for (uint64_t j = 0; j < st->n[k + 1] * 2 * (uint64_t)(k + 1); j++) {
            uint64_t f = st->face[k + 1][j];
            seen[f >> 6] |= (uint64_t)1 << (f & 63);
        }
        uint64_t hit = popcount_words(seen, words);
        free(seen);
        if (hit != st->n[k]) return TR_MATH_INVALID;
    }
    if (st->has_value) {
        uint64_t valued = 0;
        for (uint64_t i = 0; i < st->n[dim]; i++) {
            if (st->vmin[i] <= st->vmax[i]) valued++;
            else if (st->vmin[i] != INT64_MAX || st->vmax[i] != INT64_MIN) return TR_MATH_INVALID;
        }
        if (valued != r->valued) return TR_MATH_INVALID;
    }
    return TR_MATH_OK;
}

tr_topo_complex *tr_topo_adopt(const tr_scope *scope, const tr_topo_grid *g, const tr_topo_states *st,
                               const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    tr_scope_view s;
    tr_topo_view v;
    if (!scope || !st || tr_scope_get_view(scope, &s)) return complex_fail(NULL, error, TR_MATH_INVALID);
    int32_t rc = grid_check_view(&s, g, &v);
    if (rc) return complex_fail(NULL, error, rc);
    tr_math_budget dflt;
    const tr_math_budget *b = budget_or(budget, &dflt);
    tr_topo_counts c;
    memset(&c, 0, sizeof c);
    c.rows = st->rows;
    for (int32_t k = 0; k < 4; k++) c.n[k] = st->n[k];
    counts_finish(g, &v, &c);
    tr_cert_record *record = scope_record(scope, g, b, refusal, &rc);
    if (!record) return complex_fail(NULL, error, rc);
    tr_math_need need;
    complex_need(g, &c, tr_cert_record_bytes(record), &need);
    rc = tr_budget_check(b, &need, refusal);
    if (!rc) rc = adopt_check(&s, g, &v, st, c.boundary);
    if (rc) { tr_cert_record_free(record); return complex_fail(NULL, error, rc); }
    counts_finish(g, &v, &c);
    tr_topo_complex *cx = zalloc(sizeof *cx);
    if (!cx) { tr_cert_record_free(record); return complex_fail(NULL, error, TR_MATH_MEMORY); }
    cx->record = record;
    cx->grid = *g;
    cx->v = v;
    cx->counts = c;
    cx->has_value = g->value >= 0;
    cx->backend = TR_TOPO_BY_DEVICE;
    cx->occupancy = zalloc(v.words * 8);
    cx->rankdir = zalloc(v.words * 8);
    if (!cx->occupancy || !cx->rankdir) return complex_fail(cx, error, TR_MATH_MEMORY);
    if (v.words) memcpy(cx->occupancy, st->occupancy, (size_t)v.words * 8);
    rc = complex_arrays(cx);
    if (rc) return complex_fail(cx, error, rc);
    for (int32_t k = 0; k <= g->dim; k++) {
        if (c.n[k]) memcpy(cx->cells[k], st->cells[k], (size_t)c.n[k] * 8);
        if (k >= 1 && c.n[k]) {
            memcpy(cx->face[k], st->face[k], (size_t)(c.n[k] * 2 * (uint64_t)k * 4));
            memcpy(cx->sign[k], st->sign[k], (size_t)(c.n[k] * 2 * (uint64_t)k));
        }
    }
    complex_bind(cx);
    fill_rankdir(cx);
    if (cx->has_value) {
        uint64_t n = c.n[g->dim];
        cx->vmin = zalloc(n * 8);
        cx->vmax = zalloc(n * 8);
        if (!cx->vmin || !cx->vmax) return complex_fail(cx, error, TR_MATH_MEMORY);
        if (n) { memcpy(cx->vmin, st->vmin, (size_t)n * 8); memcpy(cx->vmax, st->vmax, (size_t)n * 8); }
    }
    cx->bytes = tr_budget_add(tr_topo_complex_bytes(&cx->counts, cx->has_value), tr_cert_record_bytes(cx->record));
    cx->magic = COMPLEX_MAGIC;
    set_error(error, TR_MATH_OK);
    return cx;
}

/* ------------------------------------------------------------------ reading a complex */
int32_t tr_topo_get_info(const tr_topo_complex *cx, tr_topo_info *out) {
    if (!complex_ok(cx) || !out) return TR_MATH_INVALID;
    memset(out, 0, sizeof *out);
    out->counts = cx->counts;
    out->cells = tr_topo_cells_total(cx->counts.n);
    out->incidences = tr_topo_incidences(cx->counts.n);
    out->bytes = cx->bytes;
    for (int32_t a = 0; a < 3; a++) { out->lo[a] = cx->grid.lo[a]; out->hi[a] = cx->grid.hi[a]; out->col[a] = cx->grid.col[a]; }
    out->value = cx->grid.value;
    out->backend = cx->backend;
    out->has_value = cx->has_value;
    return TR_MATH_OK;
}
int32_t tr_topo_get_view(const tr_topo_complex *cx, tr_topo_view *out) {
    if (!complex_ok(cx) || !out) return TR_MATH_INVALID;
    *out = cx->v;
    return TR_MATH_OK;
}
int32_t tr_topo_get_grid(const tr_topo_complex *cx, tr_topo_grid *out) {
    if (!complex_ok(cx) || !out) return TR_MATH_INVALID;
    *out = cx->grid;
    return TR_MATH_OK;
}
int32_t tr_topo_cells(const tr_topo_complex *cx, int32_t k, int64_t *anchors, uint8_t *axes, uint64_t *linear,
                      uint64_t capacity) {
    if (!complex_ok(cx) || k < 0 || k > cx->grid.dim) return TR_MATH_INVALID;
    uint64_t n = cx->counts.n[k];
    if (capacity < n) return TR_MATH_INVALID;
    for (uint64_t j = 0; j < n; j++) {
        uint64_t c[3], L = cx->cells[k][j];
        tr_topo_decode(&cx->v, L, c);
        uint8_t mask = 0;
        for (int32_t a = 0; a < cx->grid.dim; a++) {
            if (anchors) anchors[j * (uint64_t)cx->grid.dim + (uint64_t)a] = (int64_t)((uint64_t)cx->grid.lo[a] + c[a] / 2);
            if (c[a] & 1) mask |= (uint8_t)(1u << a);
        }
        if (axes) axes[j] = mask;
        if (linear) linear[j] = L;
    }
    return TR_MATH_OK;
}
int32_t tr_topo_values(const tr_topo_complex *cx, int64_t *vmin, int64_t *vmax, uint64_t capacity) {
    if (!complex_ok(cx) || !cx->has_value) return TR_MATH_INVALID;
    uint64_t n = cx->counts.n[cx->grid.dim];
    if (capacity < n || (n && (!vmin || !vmax))) return TR_MATH_INVALID;
    if (n) { memcpy(vmin, cx->vmin, (size_t)n * 8); memcpy(vmax, cx->vmax, (size_t)n * 8); }
    return TR_MATH_OK;
}
int32_t tr_topo_local_view(const tr_topo_view *v, tr_topo_local *out, uint64_t capacity, tr_topo_local_totals *tot) {
    int32_t rc = tr_topo_view_check(v);
    if (rc) return rc;
    uint64_t cells = tr_topo_cells_total(v->n);
    if (out && capacity < cells) return TR_MATH_INVALID;
    tr_topo_local_totals t;
    memset(&t, 0, sizeof t);
    uint64_t at = 0, deg = 0;
    for (int32_t k = 0; k <= v->dim; k++)
        for (uint64_t j = 0; j < v->n[k]; j++, at++) {
            tr_topo_local l;
            tr_topo_local_cell(v, v->cells[k][j], &l);
            if (out) out[at] = l;
            t.boundary[k] += l.boundary;
            t.link_euler_sum[k] += l.link_euler;
            deg += l.degree;
        }
    t.cells = cells;
    t.dual_edges = deg / 2;
    if (tot) *tot = t;
    return TR_MATH_OK;
}
int32_t tr_topo_local_get(const tr_topo_complex *cx, const tr_math_budget *budget, tr_topo_local *out, uint64_t capacity,
                      tr_topo_local_totals *tot, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!complex_ok(cx)) return TR_MATH_INVALID;
    tr_math_budget dflt;
    int32_t rc = tr_budget_check_one(budget_or(budget, &dflt), TR_BUDGET_OUTPUT, out ? tr_topo_cells_total(cx->counts.n) : 0,
                                     refusal);
    if (rc) return rc;
    return tr_topo_local_view(&cx->v, out, capacity, tot);
}

/* the dual graph by union-find (the root of a component is its smallest top index) */
static uint32_t uf_find(uint32_t *parent, uint32_t i) {
    uint32_t r = i;
    while (parent[r] != r) r = parent[r];
    while (parent[i] != r) { uint32_t n = parent[i]; parent[i] = r; i = n; }
    return r;
}
static int pair_cmp(const void *a, const void *b) {
    uint32_t x = *(const uint32_t *)a, y = *(const uint32_t *)b;
    return x < y ? -1 : x > y;
}
int32_t tr_topo_adjacency_get(const tr_topo_complex *cx, int32_t adjacency, const tr_math_budget *budget,
                              uint32_t *labels, uint32_t *pairs, uint64_t pair_capacity, tr_topo_adjacency *out,
                              tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!complex_ok(cx) || !out) return TR_MATH_INVALID;
    const int32_t dim = cx->grid.dim;
    int32_t adj = adjacency ? adjacency : cx->grid.adjacency;
    if (!tr_topo_adjacency_ok(dim, adj)) return TR_MATH_INVALID;
    const uint64_t n = cx->counts.n[dim];
    tr_math_need need;
    need_zero(&need);
    need.need[TR_BUDGET_OUTPUT] = tr_budget_add(labels ? n : 0, pairs ? pair_capacity : 0);
    need.need[TR_BUDGET_BYTES] = tr_budget_add(tr_budget_mul(n, 4), 512);
    tr_math_budget dflt;
    int32_t rc = tr_budget_check(budget_or(budget, &dflt), &need, refusal);
    if (rc) return rc;
    uint32_t *parent = zalloc(n * 4);
    if (!parent) return TR_MATH_MEMORY;
    for (uint64_t i = 0; i < n; i++) parent[i] = (uint32_t)i;
    tr_topo_view v = cx->v;
    v.adjacency = adj;
    uint64_t edges = 0;
    const uint32_t combos = dim == 3 ? 27u : 9u;
    for (uint64_t i = 0; i < n; i++) {
        uint64_t c[3];
        tr_topo_decode(&v, cx->cells[dim][i], c);
        uint32_t mine[26];
        int32_t nmine = 0;
        for (uint32_t o = 0; o < combos; o++) {
            int32_t off[3] = {(int32_t)(o % 3) - 1, (int32_t)((o / 3) % 3) - 1, dim == 3 ? (int32_t)(o / 9) - 1 : 0};
            int32_t nz = (off[0] != 0) + (off[1] != 0) + (off[2] != 0);
            if (!nz) continue;
            if ((adj == 4 || adj == 6) && nz != 1) continue;
            if (adj == 18 && nz > 2) continue;
            int32_t hi = dim - 1;
            while (hi >= 0 && off[hi] == 0) hi--;
            if (off[hi] < 0) continue;                         /* each pair once: the later position */
            uint64_t p = 0;
            int ok = 1;
            for (int32_t a = 0; a < dim && ok; a++) {
                int64_t x = (int64_t)((c[a] - 1) / 2) + off[a];
                if (x < 0 || (uint64_t)x >= v.ext[a]) ok = 0;
                else p += (uint64_t)x * v.pstride[a];
            }
            if (!ok || !tr_topo_bit(cx->occupancy, p)) continue;
            uint32_t j = (uint32_t)top_index(cx, p);
            uint32_t ri = uf_find(parent, (uint32_t)i), rj = uf_find(parent, j);
            if (ri != rj) { if (ri < rj) parent[rj] = ri; else parent[ri] = rj; }
            mine[nmine++] = j;
        }
        if (pairs && nmine) {
            qsort(mine, (size_t)nmine, sizeof mine[0], pair_cmp);
            for (int32_t q = 0; q < nmine; q++) {
                if (edges + (uint64_t)q >= pair_capacity) { free(parent); return TR_MATH_INVALID; }
                pairs[2 * (edges + (uint64_t)q)] = (uint32_t)i;
                pairs[2 * (edges + (uint64_t)q) + 1] = mine[q];
            }
        }
        edges += (uint64_t)nmine;
    }
    uint64_t comps = 0;
    for (uint64_t i = 0; i < n; i++) {
        uint32_t r = uf_find(parent, (uint32_t)i);
        if (r == i) comps++;
        if (labels) labels[i] = r;
    }
    free(parent);
    memset(out, 0, sizeof *out);
    out->tops = n; out->edges = edges; out->components = comps; out->adjacency = adj;
    return TR_MATH_OK;
}

int32_t tr_topo_fresh(const tr_topo_complex *cx, const tr_scope *current, tr_cert_staleness *out) {
    if (!complex_ok(cx) || !current || !out) return TR_MATH_INVALID;
    tr_scope_info info;
    if (tr_scope_get_info(current, &info)) return TR_MATH_INVALID;
    tr_cert_source src;
    if (tr_cert_record_get(cx->record, &src)) return TR_MATH_INVALID;
    for (int32_t i = 0; i < src.ncols; i++)
        if (src.col[i].index >= info.ncols) {          /* the current scope has no such column: another layout */
            memset(out, 0, sizeof *out);
            out->reasons = TR_CERT_STALE_COLUMNS; out->first = TR_CERT_STALE_COLUMNS;
            out->column = src.col[i].index; out->dim = -1;
            return TR_MATH_STALE;
        }
    return tr_cert_record_fresh(cx->record, current, NULL, 0, TR_CERT_MATCH_EXACT, out);
}
const tr_cert_record *tr_topo_get_record(const tr_topo_complex *cx) { return complex_ok(cx) ? cx->record : NULL; }

int32_t tr_topo_equal(const tr_topo_complex *a, const tr_topo_complex *b) {
    if (!complex_ok(a) || !complex_ok(b)) return TR_MATH_INVALID;
    if (memcmp(&a->grid, &b->grid, sizeof a->grid) || memcmp(&a->counts, &b->counts, sizeof a->counts) ||
        a->has_value != b->has_value) return 0;
    if (a->v.words && memcmp(a->occupancy, b->occupancy, (size_t)a->v.words * 8)) return 0;
    for (int32_t k = 0; k <= a->grid.dim; k++) {
        uint64_t n = a->counts.n[k];
        if (n && memcmp(a->cells[k], b->cells[k], (size_t)n * 8)) return 0;
        if (k >= 1 && n && (memcmp(a->face[k], b->face[k], (size_t)(n * 2 * (uint64_t)k * 4)) ||
                            memcmp(a->sign[k], b->sign[k], (size_t)(n * 2 * (uint64_t)k)))) return 0;
    }
    uint64_t n = a->counts.n[a->grid.dim];
    if (a->has_value && n && (memcmp(a->vmin, b->vmin, (size_t)n * 8) || memcmp(a->vmax, b->vmax, (size_t)n * 8))) return 0;
    return 1;
}

/* ------------------------------------------------------------------ TOPO-002: d o d */
static uint64_t dd_terms(const uint64_t *n) {
    return tr_budget_add(tr_budget_mul(n[2], 8), tr_budget_mul(n[3], 24));
}
int32_t tr_topo_check_view(const tr_topo_view *v, const tr_math_budget *budget, tr_topo_dd *out,
                           tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    int32_t rc = tr_topo_view_check(v);
    if (rc || !out) return TR_MATH_INVALID;
    tr_math_budget dflt;
    rc = tr_budget_check_one(budget_or(budget, &dflt), TR_BUDGET_TENSOR, dd_terms(v->n), refusal);
    if (rc) return rc;
    tr_topo_dd e;
    tr_topo_dd_init(&e);
    for (int32_t k = 2; k <= v->dim; k++)
        for (uint64_t j = 0; j < v->n[k]; j++) {
            int zz, z2, bad;
            e.terms += tr_topo_dd_cell(v, k, j, &zz, &z2, &bad);
            e.checked[k]++;
            if (bad) e.invalid++;
            if (!zz || bad) { e.violations_z[k]++; if (j < e.witness_z[k]) e.witness_z[k] = j; }
            if (!z2 || bad) { e.violations_gf2[k]++; if (j < e.witness_gf2[k]) e.witness_gf2[k] = j; }
        }
    tr_topo_dd_finish(&e);
    *out = e;
    return TR_MATH_OK;
}
int32_t tr_topo_check(const tr_topo_complex *cx, const tr_math_budget *budget, tr_topo_dd *out, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!complex_ok(cx)) return TR_MATH_INVALID;
    return tr_topo_check_view(&cx->v, budget, out, refusal);
}

int32_t tr_topo_boundary(const tr_topo_complex *cx, int32_t k, uint64_t prime, const tr_math_budget *budget,
                         uint64_t *rows, uint64_t *cols, int64_t *vals, uint64_t capacity, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!complex_ok(cx) || k < 1 || k > cx->grid.dim || prime == 1) return TR_MATH_INVALID;
    if (prime > 2) {
        tr_gfp f;
        int32_t rc = tr_gfp_field(prime, &f);
        if (rc) return rc;
    }
    uint64_t n = cx->counts.n[k], m = n * 2 * (uint64_t)k;
    tr_math_budget dflt;
    int32_t rc = tr_budget_check_one(budget_or(budget, &dflt), TR_BUDGET_OUTPUT, m, refusal);
    if (rc) return rc;
    if (capacity < m || (m && (!rows || !cols || !vals))) return TR_MATH_INVALID;
    for (uint64_t j = 0; j < n; j++)
        for (int32_t s = 0; s < 2 * k; s++) {
            uint64_t at = j * 2 * (uint64_t)k + (uint64_t)s;
            int64_t x = cx->sign[k][at];
            rows[at] = cx->face[k][at];
            cols[at] = j;
            vals[at] = prime == 0 ? x : prime == 2 ? 1 : (x > 0 ? 1 : (int64_t)(prime - 1));
        }
    return TR_MATH_OK;
}

/* ------------------------------------------------------------------ explicit chain complexes */
struct tr_topo_chain_s {
    uint64_t magic;
    int32_t top, reserved;
    uint64_t n[MAXK], nnz[MAXK];
    uint64_t *ptr[MAXK];
    uint32_t *row[MAXK];
    int64_t *val[MAXK];
    uint64_t bytes;
};
static int chain_ok(const tr_topo_chain *c) { return c && c->magic == CHAIN_MAGIC; }
void tr_topo_chain_free(tr_topo_chain *c) {
    if (!chain_ok(c)) return;
    for (int k = 0; k < MAXK; k++) { free(c->ptr[k]); free(c->row[k]); free(c->val[k]); }
    c->magic = 0;
    free(c);
}
static tr_topo_chain *chain_fail(tr_topo_chain *c, int32_t *error, int32_t rc) {
    if (c) { c->magic = CHAIN_MAGIC; tr_topo_chain_free(c); }
    set_error(error, rc);
    return NULL;
}
tr_topo_chain *tr_topo_chain_create(int32_t top, const uint64_t *n, const uint64_t *const *ptr,
                                    const uint32_t *const *row, const int64_t *const *val, const tr_math_budget *budget,
                                    tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    if (top < 0 || !n || (top > 0 && (!ptr || !row || !val))) return chain_fail(NULL, error, TR_MATH_INVALID);
    if (top > TR_TOPO_MAX_CHAIN) return chain_fail(NULL, error, TR_MATH_UNSUPPORTED);
    for (int32_t k = 0; k <= top; k++) if (n[k] > UINT32_MAX) return chain_fail(NULL, error, TR_MATH_UNSUPPORTED);
    uint64_t nnz[MAXK] = {0}, cells = 0, entries = 0, bytes = 1024, scratch = 0;
    for (int32_t k = 0; k <= top; k++) cells = tr_budget_add(cells, n[k]);
    for (int32_t k = 1; k <= top; k++) {
        if (!ptr[k] || ptr[k][0] != 0) return chain_fail(NULL, error, TR_MATH_INVALID);
        for (uint64_t j = 0; j < n[k]; j++) if (ptr[k][j + 1] < ptr[k][j]) return chain_fail(NULL, error, TR_MATH_INVALID);
        nnz[k] = ptr[k][n[k]];
        if (nnz[k] > UINT32_MAX) return chain_fail(NULL, error, TR_MATH_UNSUPPORTED);
        if (nnz[k] && (!row[k] || !val[k])) return chain_fail(NULL, error, TR_MATH_INVALID);
        entries = tr_budget_add(entries, nnz[k]);
        bytes = tr_budget_add(bytes, tr_budget_add(tr_budget_mul(n[k] + 1, 8), tr_budget_mul(nnz[k], 12)));
        if (k >= 2) scratch = u64max(scratch, tr_budget_mul(n[k - 2], 25));   /* acc, touched, mark */
    }
    tr_math_need need;
    need_zero(&need);
    need.need[TR_BUDGET_VARIABLES] = (uint64_t)top + 1;
    need.need[TR_BUDGET_STATES] = cells;
    need.need[TR_BUDGET_TENSOR] = entries;
    need.need[TR_BUDGET_BYTES] = tr_budget_add(bytes, scratch);
    tr_math_budget dflt;
    int32_t rc = tr_budget_check(budget_or(budget, &dflt), &need, refusal);
    if (rc) return chain_fail(NULL, error, rc);
    for (int32_t k = 1; k <= top; k++)                     /* canonical CSC: increasing rows in range, nonzero values */
        for (uint64_t j = 0; j < n[k]; j++)
            for (uint64_t e = ptr[k][j]; e < ptr[k][j + 1]; e++) {
                if (row[k][e] >= n[k - 1] || val[k][e] == 0 || (e > ptr[k][j] && row[k][e] <= row[k][e - 1]))
                    return chain_fail(NULL, error, TR_MATH_INVALID);
            }
    tr_topo_chain *c = zalloc(sizeof *c);
    if (!c) return chain_fail(NULL, error, TR_MATH_MEMORY);
    c->top = top;
    for (int32_t k = 0; k <= top; k++) c->n[k] = n[k];
    for (int32_t k = 1; k <= top; k++) {
        c->nnz[k] = nnz[k];
        c->ptr[k] = zalloc((n[k] + 1) * 8);
        c->row[k] = zalloc(nnz[k] * 4);
        c->val[k] = zalloc(nnz[k] * 8);
        if (!c->ptr[k] || !c->row[k] || !c->val[k]) return chain_fail(c, error, TR_MATH_MEMORY);
        memcpy(c->ptr[k], ptr[k], (size_t)(n[k] + 1) * 8);
        if (nnz[k]) { memcpy(c->row[k], row[k], (size_t)nnz[k] * 4); memcpy(c->val[k], val[k], (size_t)nnz[k] * 8); }
    }
    /* d_{k-1} d_k = 0 over Z, exactly (128-bit accumulation, overflow refused) */
    for (int32_t k = 2; k <= top; k++) {
        __int128 *acc = zalloc(n[k - 2] * 16);
        uint64_t *touched = zalloc(n[k - 2] * 8);
        uint8_t *mark = zalloc(n[k - 2]);
        if (!acc || !touched || !mark) { free(acc); free(touched); free(mark); return chain_fail(c, error, TR_MATH_MEMORY); }
        int32_t bad = 0;
        for (uint64_t j = 0; j < n[k] && !bad; j++) {
            uint64_t nt = 0;
            for (uint64_t e = c->ptr[k][j]; e < c->ptr[k][j + 1] && !bad; e++) {
                uint64_t i = c->row[k][e];
                for (uint64_t f = c->ptr[k - 1][i]; f < c->ptr[k - 1][i + 1]; f++) {
                    uint64_t g = c->row[k - 1][f];
                    __int128 t = (__int128)c->val[k][e] * c->val[k - 1][f], s;
                    if (__builtin_add_overflow(acc[g], t, &s)) { bad = TR_MATH_OVERFLOW; break; }
                    acc[g] = s;
                    if (!mark[g]) { mark[g] = 1; touched[nt++] = g; }
                }
            }
            for (uint64_t q = 0; q < nt; q++) {
                if (acc[touched[q]] != 0 && !bad) bad = TR_MATH_DOMAIN_ERROR;
                acc[touched[q]] = 0; mark[touched[q]] = 0;
            }
        }
        free(acc); free(touched); free(mark);
        if (bad) return chain_fail(c, error, bad);
    }
    c->bytes = bytes;
    c->magic = CHAIN_MAGIC;
    set_error(error, TR_MATH_OK);
    return c;
}
int32_t tr_topo_chain_get_info(const tr_topo_chain *c, tr_topo_chain_info *out) {
    if (!chain_ok(c) || !out) return TR_MATH_INVALID;
    memset(out, 0, sizeof *out);
    out->top = c->top;
    int64_t e = 0;
    for (int32_t k = 0; k <= c->top; k++) { out->n[k] = c->n[k]; out->nnz[k] = c->nnz[k]; e += (k & 1) ? -(int64_t)c->n[k] : (int64_t)c->n[k]; }
    out->euler = e;
    out->bytes = c->bytes;
    return TR_MATH_OK;
}

/* ------------------------------------------------------------------ the source of a reduction */
typedef struct {
    int32_t top;
    uint64_t n[MAXK];
    const tr_topo_complex *cx;
    const tr_topo_chain *ch;
} source;
static int32_t source_of(const tr_topo_complex *cx, const tr_topo_chain *ch, source *s) {
    memset(s, 0, sizeof *s);
    if ((cx != NULL) == (ch != NULL)) return TR_MATH_INVALID;
    if (cx) {
        if (!complex_ok(cx)) return TR_MATH_INVALID;
        s->cx = cx; s->top = cx->grid.dim;
        for (int32_t k = 0; k <= s->top; k++) s->n[k] = cx->counts.n[k];
    } else {
        if (!chain_ok(ch)) return TR_MATH_INVALID;
        s->ch = ch; s->top = ch->top;
        for (int32_t k = 0; k <= s->top; k++) s->n[k] = ch->n[k];
    }
    return TR_MATH_OK;
}
static uint64_t src_len(const source *s, int32_t k, uint64_t j) {
    return s->cx ? 2 * (uint64_t)k : s->ch->ptr[k][j + 1] - s->ch->ptr[k][j];
}
static uint64_t src_nnz(const source *s, int32_t k) {
    return s->cx ? s->n[k] * 2 * (uint64_t)k : s->ch->nnz[k];
}
static uint64_t src_row(const source *s, int32_t k, uint64_t j, uint64_t e) {
    return s->cx ? s->cx->face[k][j * 2 * (uint64_t)k + e] : s->ch->row[k][s->ch->ptr[k][j] + e];
}
static int64_t src_val(const source *s, int32_t k, uint64_t j, uint64_t e) {
    return s->cx ? (int64_t)s->cx->sign[k][j * 2 * (uint64_t)k + e] : s->ch->val[k][s->ch->ptr[k][j] + e];
}

/* ------------------------------------------------------------------ the GF(2) path through FIELD-001 */
static uint64_t packed_bytes(uint64_t rows, uint64_t cols) {
    return tr_budget_mul(tr_budget_mul(rows, tr_field_words(cols)), 8);
}
static int32_t packed_fill(const source *s, int32_t k, tr_field_matrix *out) {
    uint64_t rows = s->n[k - 1], cols = s->n[k], W = tr_field_words(cols);
    if (out->rows != rows || out->cols != cols || out->stride < W || (rows && out->stride && !out->words)) return TR_MATH_INVALID;
    if (out->stride) for (uint64_t i = 0; i < rows; i++) memset(out->words + i * out->stride, 0, (size_t)out->stride * 8);
    for (uint64_t j = 0; j < cols; j++)
        for (uint64_t e = 0; e < src_len(s, k, j); e++) {
            if (!(src_val(s, k, j, e) & 1)) continue;                 /* even entries vanish in GF(2) */
            uint64_t i = src_row(s, k, j, e);
            out->words[i * out->stride + j / 64] ^= (uint64_t)1 << (j % 64);
        }
    return TR_MATH_OK;
}
int32_t tr_topo_boundary_packed(const tr_topo_complex *cx, const tr_topo_chain *ch, int32_t k, const tr_math_budget *budget,
                                tr_field_matrix *out, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    source s;
    int32_t rc = source_of(cx, ch, &s);
    if (rc || !out || k < 1 || k > s.top) return TR_MATH_INVALID;
    tr_math_budget dflt;
    rc = tr_budget_check_one(budget_or(budget, &dflt), TR_BUDGET_TENSOR, tr_budget_mul(s.n[k - 1], s.n[k]), refusal);
    if (rc) return rc;
    return packed_fill(&s, k, out);
}

/* ------------------------------------------------------------------ accounting of a reduction */
typedef struct {
    uint64_t limit_bytes, limit_steps, live, peak, steps;
    tr_math_refusal *refusal;
} acct;
static int32_t acct_take(acct *a, uint64_t bytes) {
    uint64_t want = tr_budget_add(a->live, bytes);
    if (want > a->limit_bytes) { refuse(a->refusal, TR_BUDGET_BYTES, want, a->limit_bytes); return TR_MATH_BUDGET; }
    a->live = want;
    if (want > a->peak) a->peak = want;
    return TR_MATH_OK;
}
static void *acct_alloc(acct *a, uint64_t bytes, int32_t *rc) {
    if (*rc) return NULL;
    *rc = acct_take(a, bytes);
    if (*rc) return NULL;
    void *p = zalloc(bytes);
    if (!p) { a->live -= bytes; *rc = TR_MATH_MEMORY; }
    return p;
}
static void acct_free(acct *a, void *p, uint64_t bytes) { if (p) { free(p); a->live -= bytes; } }
static int32_t acct_step(acct *a) {
    if (a->steps + 1 > a->limit_steps) { refuse(a->refusal, TR_BUDGET_ITERATIONS, a->steps + 1, a->limit_steps); return TR_MATH_BUDGET; }
    a->steps++;
    return TR_MATH_OK;
}
/* arena of reduced columns (bump allocation in blocks; freed as a whole) */
#define ARENA_BLOCK (UINT64_C(1) << 20)
typedef struct arena_block_s { struct arena_block_s *next; uint64_t used, cap, reserved; } arena_block;
typedef struct { arena_block *head; uint64_t bytes; } arena;
static void *arena_get(acct *a, arena *ar, uint64_t bytes, int32_t *rc) {
    bytes = (bytes + 7) & ~(uint64_t)7;
    if (*rc) return NULL;
    if (!ar->head || ar->head->cap - ar->head->used < bytes) {
        uint64_t cap = u64max(ARENA_BLOCK, bytes), total = sizeof(arena_block) + cap;
        arena_block *b = acct_alloc(a, total, rc);
        if (!b) return NULL;
        b->cap = cap; b->used = 0; b->next = ar->head;
        ar->head = b;
        ar->bytes += total;
    }
    void *p = (unsigned char *)(ar->head + 1) + ar->head->used;
    ar->head->used += bytes;
    return p;
}
static void arena_free(acct *a, arena *ar) {
    arena_block *b = ar->head;
    while (b) { arena_block *n = b->next; acct_free(a, b, sizeof(arena_block) + b->cap); b = n; }
    ar->head = NULL; ar->bytes = 0;
}
/* a working column: rows (positions), coefficients (GF(p), Q), 128-bit scratch (Q) */
typedef struct { uint32_t *rows; int64_t *coef; __int128 *wide; uint64_t cap; int has_coef, has_wide; } wcol;
static uint64_t wcol_entry(const wcol *w) { return 4 + (w->has_coef ? 8 : 0) + (w->has_wide ? 16 : 0); }
static int32_t wcol_reserve(acct *a, wcol *w, uint64_t need) {
    if (need <= w->cap) return TR_MATH_OK;
    uint64_t cap = u64max(u64max(need, 2 * w->cap), 64);
    int32_t rc = acct_take(a, (cap - w->cap) * wcol_entry(w));
    if (rc) return rc;
    uint32_t *r = realloc(w->rows, (size_t)cap * 4);
    if (r) w->rows = r;
    int64_t *c = w->has_coef ? realloc(w->coef, (size_t)cap * 8) : NULL;
    if (c) w->coef = c;
    __int128 *x = w->has_wide ? realloc(w->wide, (size_t)cap * 16) : NULL;
    if (x) w->wide = x;
    if (!r || (w->has_coef && !c) || (w->has_wide && !x)) { a->live -= (cap - w->cap) * wcol_entry(w); return TR_MATH_MEMORY; }
    w->cap = cap;
    return TR_MATH_OK;
}
static void wcol_free(acct *a, wcol *w) {
    if (w->cap) a->live -= w->cap * wcol_entry(w);
    free(w->rows); free(w->coef); free(w->wide);
    memset(w, 0, sizeof *w);
}

/* ------------------------------------------------------------------ field arithmetic of the columns */
typedef struct { int32_t field; tr_gfp gf; } fieldctx;
/* x - y (GF(2): symmetric difference) over sorted rows */
static uint64_t merge2(const uint32_t *x, uint64_t xn, const uint32_t *y, uint64_t yn, uint32_t *out) {
    uint64_t i = 0, j = 0, n = 0;
    while (i < xn && j < yn) {
        if (x[i] < y[j]) out[n++] = x[i++];
        else if (x[i] > y[j]) out[n++] = y[j++];
        else { i++; j++; }
    }
    while (i < xn) out[n++] = x[i++];
    while (j < yn) out[n++] = y[j++];
    return n;
}
/* x - f y mod p */
static uint64_t mergep(const tr_gfp *gf, const uint32_t *xr, const int64_t *xc, uint64_t xn, const uint32_t *yr,
                       const int64_t *yc, uint64_t yn, uint64_t f, uint32_t *or_, int64_t *oc) {
    uint64_t i = 0, j = 0, n = 0;
    while (i < xn || j < yn) {
        if (j >= yn || (i < xn && xr[i] < yr[j])) { or_[n] = xr[i]; oc[n++] = xc[i++]; continue; }
        uint64_t t = tr_gfp_neg_hd(gf, tr_gfp_mul_hd(gf, f, (uint64_t)yc[j]));
        if (i >= xn || yr[j] < xr[i]) { or_[n] = yr[j]; oc[n++] = (int64_t)t; j++; continue; }
        uint64_t s = tr_gfp_add_hd(gf, (uint64_t)xc[i], t);
        if (s) { or_[n] = xr[i]; oc[n++] = (int64_t)s; }
        i++; j++;
    }
    return n;
}
/* sa x - sb y exactly in 128 bits; *g accumulates the gcd of the magnitudes; 0 on overflow (returns UINT64_MAX) */
static unsigned __int128 gcd128(unsigned __int128 a, unsigned __int128 b) {
    while (b) { unsigned __int128 t = a % b; a = b; b = t; }
    return a;
}
static unsigned __int128 mag128(__int128 x) { return x < 0 ? (unsigned __int128)0 - (unsigned __int128)x : (unsigned __int128)x; }
static uint64_t mergeq(const uint32_t *xr, const int64_t *xc, uint64_t xn, const uint32_t *yr, const int64_t *yc,
                       uint64_t yn, int64_t sa, int64_t sb, uint32_t *or_, __int128 *ow, unsigned __int128 *g) {
    uint64_t i = 0, j = 0, n = 0;
    while (i < xn || j < yn) {
        __int128 v;
        uint32_t r;
        if (j >= yn || (i < xn && xr[i] < yr[j])) { r = xr[i]; v = (__int128)sa * xc[i]; i++; }
        else if (i >= xn || yr[j] < xr[i]) { r = yr[j]; v = -((__int128)sb * yc[j]); j++; }
        else {
            r = xr[i];
            if (__builtin_sub_overflow((__int128)sa * xc[i], (__int128)sb * yc[j], &v)) return UINT64_MAX;
            i++; j++;
        }
        if (!v) continue;
        or_[n] = r; ow[n++] = v;
        if (*g != 1) *g = gcd128(*g, mag128(v));
    }
    return n;
}
/* the wide entries divided by d, checked to fit int64 */
static int32_t narrow(const __int128 *w, uint64_t n, unsigned __int128 d, int64_t *out) {
    for (uint64_t i = 0; i < n; i++) {
        __int128 q = d == 1 ? w[i] : w[i] / (__int128)d;
        if (q > INT64_MAX || q < INT64_MIN) return TR_MATH_OVERFLOW;
        out[i] = (int64_t)q;
    }
    return TR_MATH_OK;
}
static uint64_t gcd64(uint64_t a, uint64_t b) { while (b) { uint64_t t = a % b; a = b; b = t; } return a; }
static uint64_t mag64(int64_t x) { return tr_abs_u64(x); }

/* ------------------------------------------------------------------ the reduction */
typedef struct {
    uint32_t *rows; int64_t *coef;      /* the reduced column (coef NULL in GF(2)) */
    uint32_t *vrows; int64_t *vcoef;    /* its V (tracked dimensions) */
    uint32_t len, vlen;
    uint64_t col;                       /* its position in the dimension's order */
    uint64_t linv;                      /* GF(p): the inverse of its lowest coefficient */
} owner_rec;                            /* 56 bytes */
typedef struct { int32_t dim; uint64_t generator, offset, length; int64_t level; } rep_rec;
typedef struct {
    const source *src;
    fieldctx fc;
    uint32_t track;
    uint64_t nlevels;
    int64_t *level[MAXK];               /* per cell */
    uint64_t *order[MAXK], *pos[MAXK], m[MAXK];
    acct a;
    /* results */
    tr_topo_pair *pairs; uint64_t npairs, cappairs;
    tr_topo_pair *ess; uint64_t ness, capess;
    rep_rec *reps; uint64_t nreps, capreps;
    uint64_t *ecell; int64_t *ecoef; uint64_t nentries, capcell, capcoef;
    uint64_t rank[MAXK + 1];
    uint64_t zero_length;
} reducer;

static int32_t grow(acct *a, void **p, uint64_t *cap, uint64_t need, uint64_t size) {
    if (need <= *cap) return TR_MATH_OK;
    uint64_t nc = u64max(u64max(need, 2 * *cap), 16);
    int32_t rc = acct_take(a, (nc - *cap) * size);
    if (rc) return rc;
    void *q = realloc(*p, (size_t)(nc * size));
    if (!q) { a->live -= (nc - *cap) * size; return TR_MATH_MEMORY; }
    *p = q; *cap = nc;
    return TR_MATH_OK;
}
static int32_t push_pair(reducer *R, tr_topo_pair *pr, int essential) {
    if (essential) {
        int32_t rc = grow(&R->a, (void **)&R->ess, &R->capess, R->ness + 1, sizeof *pr);
        if (rc) return rc;
        R->ess[R->ness++] = *pr;
    } else {
        int32_t rc = grow(&R->a, (void **)&R->pairs, &R->cappairs, R->npairs + 1, sizeof *pr);
        if (rc) return rc;
        R->pairs[R->npairs++] = *pr;
        if (pr->birth == pr->death) R->zero_length++;
    }
    return TR_MATH_OK;
}
typedef struct { uint64_t cell; int64_t coef; } entry;
static int entry_cmp(const void *a, const void *b) {
    uint64_t x = ((const entry *)a)->cell, y = ((const entry *)b)->cell;
    return x < y ? -1 : x > y;
}
/* a representative: the V of an essential column (positions in the order of dimension k), normalized and sorted */
static int32_t push_rep(reducer *R, int32_t k, uint64_t genpos, const uint32_t *vr, const int64_t *vc, uint64_t vn) {
    int32_t rc = grow(&R->a, (void **)&R->reps, &R->capreps, R->nreps + 1, sizeof(rep_rec));
    if (!rc) rc = grow(&R->a, (void **)&R->ecell, &R->capcell, R->nentries + vn, 8);
    if (!rc) rc = grow(&R->a, (void **)&R->ecoef, &R->capcoef, R->nentries + vn, 8);
    entry *tmp = acct_alloc(&R->a, tr_budget_mul(vn, sizeof(entry)), &rc);
    if (rc) return rc;
    uint64_t content = 0;
    int64_t gencoef = 1;
    for (uint64_t i = 0; i < vn; i++) {
        int64_t c = R->fc.field == TR_TOPO_GF2 ? 1 : vc[i];
        tmp[i].cell = R->order[k][vr[i]];
        tmp[i].coef = c;
        if (vr[i] == genpos) gencoef = c;
        if (R->fc.field == TR_TOPO_Q) content = gcd64(content, mag64(c));
    }
    if (R->fc.field == TR_TOPO_Q && content > 1)
        for (uint64_t i = 0; i < vn; i++) tmp[i].coef = (int64_t)(tmp[i].coef / (int64_t)content);
    if (R->fc.field == TR_TOPO_Q && gencoef < 0)
        for (uint64_t i = 0; i < vn; i++) {
            if (tmp[i].coef == INT64_MIN) { acct_free(&R->a, tmp, vn * sizeof(entry)); return TR_MATH_OVERFLOW; }
            tmp[i].coef = -tmp[i].coef;
        }
    qsort(tmp, (size_t)vn, sizeof(entry), entry_cmp);
    for (uint64_t i = 0; i < vn; i++) { R->ecell[R->nentries + i] = tmp[i].cell; R->ecoef[R->nentries + i] = tmp[i].coef; }
    acct_free(&R->a, tmp, vn * sizeof(entry));
    rep_rec *rr = &R->reps[R->nreps++];
    rr->dim = k; rr->generator = R->order[k][genpos]; rr->offset = R->nentries; rr->length = vn;
    rr->level = R->level[k][R->order[k][genpos]];
    R->nentries += vn;
    return TR_MATH_OK;
}

typedef struct { uint32_t row; int64_t coef; } load_entry;
static int load_cmp(const void *a, const void *b) {
    uint32_t x = ((const load_entry *)a)->row, y = ((const load_entry *)b)->row;
    return x < y ? -1 : x > y;
}
/* column j (position jp) of d_k into w: positions of its faces, sorted, coefficients in the field; the length */
static int32_t load_column(reducer *R, int32_t k, uint64_t cell, wcol *w, uint64_t *len, load_entry *scratch) {
    const source *s = R->src;
    uint64_t n = src_len(s, k, cell), out = 0;
    int32_t rc = wcol_reserve(&R->a, w, n);
    if (rc) return rc;
    for (uint64_t e = 0; e < n; e++) {
        uint64_t face = src_row(s, k, cell, e), p = R->pos[k - 1][face];
        int64_t v = src_val(s, k, cell, e);
        if (p == TR_TOPO_NONE) return TR_MATH_INVALID;          /* a face after its coface: never for a filtration */
        int64_t c;
        if (R->fc.field == TR_TOPO_GF2) { if (!(v & 1)) continue; c = 1; }
        else if (R->fc.field == TR_TOPO_GFP) { c = (int64_t)tr_gfp_reduce_i64_hd(&R->fc.gf, v); if (!c) continue; }
        else c = v;
        scratch[out].row = (uint32_t)p; scratch[out].coef = c; out++;
    }
    if (out > 1) {
        if (out <= 16) {
            for (uint64_t i = 1; i < out; i++) {
                load_entry x = scratch[i];
                uint64_t j = i;
                while (j > 0 && scratch[j - 1].row > x.row) { scratch[j] = scratch[j - 1]; j--; }
                scratch[j] = x;
            }
        } else qsort(scratch, (size_t)out, sizeof(load_entry), load_cmp);
    }
    for (uint64_t i = 0; i < out; i++) { w->rows[i] = scratch[i].row; if (w->has_coef) w->coef[i] = scratch[i].coef; }
    *len = out;
    return TR_MATH_OK;
}

/* reduce d_k (columns: the k-cells in order; rows: the (k-1)-positions); killed_k: the k-positions that are lows of
 * d_{k+1} (NULL: none); out_killed: the (k-1)-positions that are lows here */
static int32_t reduce_dim(reducer *R, int32_t k, const uint64_t *killed_k, uint64_t *out_killed) {
    const int32_t field = R->fc.field, track = (int32_t)((R->track >> k) & 1);
    const uint64_t mk = R->m[k], mrows = R->m[k - 1];
    int32_t rc = TR_MATH_OK;
    owner_rec *owner = acct_alloc(&R->a, tr_budget_mul(mrows, sizeof(owner_rec)), &rc);
    uint64_t maxlen = 1;
    for (uint64_t jp = 0; jp < mk; jp++) maxlen = u64max(maxlen, src_len(R->src, k, R->order[k][jp]));
    load_entry *scratch = acct_alloc(&R->a, tr_budget_mul(maxlen, sizeof(load_entry)), &rc);
    if (rc) { acct_free(&R->a, owner, mrows * sizeof(owner_rec)); acct_free(&R->a, scratch, maxlen * sizeof(load_entry)); return rc; }
    arena ar = {NULL, 0};
    wcol w[2], vw[2];
    memset(w, 0, sizeof w); memset(vw, 0, sizeof vw);
    for (int i = 0; i < 2; i++) {
        w[i].has_coef = vw[i].has_coef = field != TR_TOPO_GF2;
        w[i].has_wide = vw[i].has_wide = field == TR_TOPO_Q;
    }
    for (uint64_t jp = 0; jp < mk && !rc; jp++) {
        uint64_t cell = R->order[k][jp], len = 0, vlen = 0;
        int cur = 0;
        rc = load_column(R, k, cell, &w[0], &len, scratch);
        if (rc) break;
        if (track) {
            rc = wcol_reserve(&R->a, &vw[0], 1);
            if (rc) break;
            vw[0].rows[0] = (uint32_t)jp;
            if (vw[0].has_coef) vw[0].coef[0] = 1;
            vlen = 1;
        }
        int vcur = 0;
        while (len) {
            uint32_t low = w[cur].rows[len - 1];
            owner_rec *o = &owner[low];
            if (!o->len) break;
            rc = acct_step(&R->a);
            if (rc) break;
            int nxt = cur ^ 1, vnxt = vcur ^ 1;
            rc = wcol_reserve(&R->a, &w[nxt], len + o->len);
            if (!rc && track) rc = wcol_reserve(&R->a, &vw[vnxt], vlen + o->vlen);
            if (rc) break;
            if (field == TR_TOPO_GF2) {
                len = merge2(w[cur].rows, len, o->rows, o->len, w[nxt].rows);
                if (track) vlen = merge2(vw[vcur].rows, vlen, o->vrows, o->vlen, vw[vnxt].rows);
            } else if (field == TR_TOPO_GFP) {
                uint64_t f = tr_gfp_mul_hd(&R->fc.gf, (uint64_t)w[cur].coef[len - 1], o->linv);
                len = mergep(&R->fc.gf, w[cur].rows, w[cur].coef, len, o->rows, o->coef, o->len, f, w[nxt].rows, w[nxt].coef);
                if (track) vlen = mergep(&R->fc.gf, vw[vcur].rows, vw[vcur].coef, vlen, o->vrows, o->vcoef, o->vlen, f,
                                         vw[vnxt].rows, vw[vnxt].coef);
            } else {
                int64_t a = w[cur].coef[len - 1], b = o->coef[o->len - 1];
                uint64_t g0 = gcd64(mag64(a), mag64(b));
                int64_t sa = b / (int64_t)g0, sb = a / (int64_t)g0;
                unsigned __int128 g = 0;
                len = mergeq(w[cur].rows, w[cur].coef, len, o->rows, o->coef, o->len, sa, sb, w[nxt].rows, w[nxt].wide, &g);
                if (len == UINT64_MAX) { rc = TR_MATH_OVERFLOW; break; }
                if (track) {
                    vlen = mergeq(vw[vcur].rows, vw[vcur].coef, vlen, o->vrows, o->vcoef, o->vlen, sa, sb, vw[vnxt].rows,
                                  vw[vnxt].wide, &g);
                    if (vlen == UINT64_MAX) { rc = TR_MATH_OVERFLOW; break; }
                }
                if (!g) g = 1;
                if (g >> 127) { rc = TR_MATH_OVERFLOW; break; }
                rc = narrow(w[nxt].wide, len, g, w[nxt].coef);
                if (!rc && track) rc = narrow(vw[vnxt].wide, vlen, g, vw[vnxt].coef);
                if (rc) break;
            }
            cur = nxt;
            vcur = vnxt;
        }
        if (rc) break;
        if (len) {                                              /* negative: a pair (low, jp) */
            uint32_t low = w[cur].rows[len - 1];
            owner_rec *o = &owner[low];
            o->rows = arena_get(&R->a, &ar, (uint64_t)len * 4, &rc);
            if (field != TR_TOPO_GF2) o->coef = arena_get(&R->a, &ar, (uint64_t)len * 8, &rc);
            if (track) {
                o->vrows = arena_get(&R->a, &ar, (uint64_t)vlen * 4, &rc);
                if (field != TR_TOPO_GF2) o->vcoef = arena_get(&R->a, &ar, (uint64_t)vlen * 8, &rc);
            }
            if (rc) break;
            memcpy(o->rows, w[cur].rows, (size_t)len * 4);
            if (field != TR_TOPO_GF2) memcpy(o->coef, w[cur].coef, (size_t)len * 8);
            if (track) {
                memcpy(o->vrows, vw[vcur].rows, (size_t)vlen * 4);
                if (field != TR_TOPO_GF2) memcpy(o->vcoef, vw[vcur].coef, (size_t)vlen * 8);
            }
            o->len = (uint32_t)len; o->vlen = (uint32_t)vlen; o->col = jp;
            if (field == TR_TOPO_GFP) o->linv = tr_gfp_inv_hd(&R->fc.gf, (uint64_t)o->coef[len - 1]);
            R->rank[k]++;
            out_killed[low >> 6] |= (uint64_t)1 << (low & 63);
            uint64_t bcell = R->order[k - 1][low];
            tr_topo_pair pr = {k - 1, 0, R->level[k - 1][bcell], R->level[k][cell], bcell, cell};
            rc = push_pair(R, &pr, 0);
        } else if (!(killed_k && ((killed_k[jp >> 6] >> (jp & 63)) & 1))) {   /* positive and never killed: essential */
            tr_topo_pair pr = {k, 0, R->level[k][cell], TR_TOPO_NEVER, cell, TR_TOPO_NONE};
            rc = push_pair(R, &pr, 1);
            if (!rc && track) rc = push_rep(R, k, jp, vw[vcur].rows, vw[vcur].coef, vlen);
        }
    }
    for (int i = 0; i < 2; i++) { wcol_free(&R->a, &w[i]); wcol_free(&R->a, &vw[i]); }
    arena_free(&R->a, &ar);
    acct_free(&R->a, owner, mrows * sizeof(owner_rec));
    acct_free(&R->a, scratch, maxlen * sizeof(load_entry));
    return rc;
}
/* the essential 0-cells (every 0-cell is positive) */
static int32_t reduce_vertices(reducer *R, const uint64_t *killed0) {
    int32_t rc = TR_MATH_OK;
    for (uint64_t jp = 0; jp < R->m[0] && !rc; jp++) {
        if (killed0 && ((killed0[jp >> 6] >> (jp & 63)) & 1)) continue;
        uint64_t cell = R->order[0][jp];
        tr_topo_pair pr = {0, 0, R->level[0][cell], TR_TOPO_NEVER, cell, TR_TOPO_NONE};
        rc = push_pair(R, &pr, 1);
        if (!rc && (R->track & 1)) {
            uint32_t r = (uint32_t)jp;
            int64_t c = 1;
            rc = push_rep(R, 0, jp, &r, &c, 1);
        }
    }
    return rc;
}

static int pair_order(const void *a, const void *b) {
    const tr_topo_pair *x = a, *y = b;
    if (x->dim != y->dim) return x->dim < y->dim ? -1 : 1;
    if (x->birth != y->birth) return x->birth < y->birth ? -1 : 1;
    if (x->death != y->death) return x->death < y->death ? -1 : 1;
    if (x->birth_cell != y->birth_cell) return x->birth_cell < y->birth_cell ? -1 : 1;
    return 0;
}
static int rep_order(const void *a, const void *b) {
    const rep_rec *x = a, *y = b;
    if (x->dim != y->dim) return x->dim < y->dim ? -1 : 1;
    if (x->level != y->level) return x->level < y->level ? -1 : 1;
    return x->generator < y->generator ? -1 : x->generator > y->generator;
}

/* the field of a request: GF(2), a verified prime, or Q */
static int32_t field_of(const tr_topo_field *f, fieldctx *fc) {
    memset(fc, 0, sizeof *fc);
    if (!f || f->reserved) return TR_MATH_INVALID;
    fc->field = f->field;
    if (f->field == TR_TOPO_GF2 || f->field == TR_TOPO_Q) return f->prime ? TR_MATH_INVALID : TR_MATH_OK;
    if (f->field != TR_TOPO_GFP || f->prime < 3) return TR_MATH_INVALID;
    return tr_gfp_field(f->prime, &fc->gf);
}
/* the fixed arrays of a reduction (accounted as they are allocated; also the static bytes need) */
static uint64_t reduce_fixed_bytes(const source *s, uint64_t nlevels) {
    uint64_t cells = 0, maxn = 1;
    for (int32_t k = 0; k <= s->top; k++) { cells = tr_budget_add(cells, s->n[k]); maxn = u64max(maxn, s->n[k]); }
    uint64_t b = tr_budget_mul(cells, 24);                               /* level, pos, order */
    b = tr_budget_add(b, tr_budget_mul(tr_budget_add(nlevels, 1), 8));  /* the counting sort's buckets */
    b = tr_budget_add(b, tr_budget_mul((maxn + 63) / 64, 16));           /* killed bitmaps (two at a time) */
    b = tr_budget_add(b, tr_budget_mul(maxn, sizeof(owner_rec)));        /* one owner table at a time */
    return tr_budget_add(b, 4096);
}
int32_t tr_topo_reduce_need(const tr_topo_complex *cx, const tr_topo_chain *ch, const tr_topo_field *f,
                            const tr_topo_filtration *flt, tr_math_need *need) {
    source s;
    fieldctx fc;
    int32_t rc = source_of(cx, ch, &s);
    if (rc || !need) return TR_MATH_INVALID;
    rc = field_of(f, &fc);
    if (rc) return rc;
    if (flt && f->method == TR_TOPO_PACKED) return TR_MATH_INVALID;
    uint64_t cells = 0, nnz = 0, tensor = 0, packed = 0;
    for (int32_t k = 0; k <= s.top; k++) {
        cells = tr_budget_add(cells, s.n[k]);
        if (k >= 1) {
            nnz = tr_budget_add(nnz, src_nnz(&s, k));
            tensor = u64max(tensor, tr_budget_mul(s.n[k - 1], s.n[k]));
            packed = u64max(packed, tr_budget_mul(packed_bytes(s.n[k - 1], s.n[k]), 2));
        }
    }
    need_zero(need);
    need->need[TR_BUDGET_VARIABLES] = (uint64_t)s.top + 1;
    need->need[TR_BUDGET_STATES] = cells;
    need->need[TR_BUDGET_SUPPORT] = flt ? flt->nlevels : 1;
    if (f->method == TR_TOPO_PACKED) {
        need->need[TR_BUDGET_TENSOR] = tensor;
        need->need[TR_BUDGET_BYTES] = tr_budget_add(packed, 4096);
    } else {
        need->need[TR_BUDGET_TENSOR] = nnz;
        need->need[TR_BUDGET_BYTES] = reduce_fixed_bytes(&s, flt ? flt->nlevels : 1);
    }
    return TR_MATH_OK;
}

/* the entry levels of every cell (TR_TOPO_NEVER), validated; flt NULL: every cell at level 0 */
static int32_t levels_fill(reducer *R, const tr_topo_filtration *flt) {
    const source *s = R->src;
    if (!flt) {
        for (int32_t k = 0; k <= s->top; k++) for (uint64_t i = 0; i < s->n[k]; i++) R->level[k][i] = 0;
        return TR_MATH_OK;
    }
    const uint64_t m = flt->nlevels;
    if (s->cx) {
        const tr_topo_complex *cx = s->cx;
        const int32_t d = cx->grid.dim;
        if (flt->source == TR_TOPO_LEVELS) {
            if (s->n[d] && !flt->levels) return TR_MATH_INVALID;
            for (uint64_t i = 0; i < s->n[d]; i++) {
                int64_t l = flt->levels[i];
                if (l != TR_TOPO_NEVER && (l < 0 || (uint64_t)l >= m)) return TR_MATH_INVALID;
                R->level[d][i] = l;
            }
        } else {
            if (!cx->has_value || !flt->thresholds) return TR_MATH_INVALID;
            const int up = flt->source == TR_TOPO_SUBLEVEL;
            for (uint64_t l = 1; l < m; l++)
                if (up ? flt->thresholds[l] <= flt->thresholds[l - 1] : flt->thresholds[l] >= flt->thresholds[l - 1])
                    return TR_MATH_INVALID;
            for (uint64_t i = 0; i < s->n[d]; i++) {
                int64_t l = TR_TOPO_NEVER;
                if (cx->vmin[i] <= cx->vmax[i]) {
                    int64_t v = up ? cx->vmin[i] : cx->vmax[i];
                    uint64_t lo = 0, hi = m;                    /* the first level that admits v */
                    while (lo < hi) {
                        uint64_t mid = lo + (hi - lo) / 2;
                        if (up ? v <= flt->thresholds[mid] : v >= flt->thresholds[mid]) hi = mid; else lo = mid + 1;
                    }
                    if (lo < m) l = (int64_t)lo;
                }
                R->level[d][i] = l;
            }
        }
        for (int32_t k = d - 1; k >= 0; k--) {                 /* the closure: the earliest coface */
            for (uint64_t i = 0; i < s->n[k]; i++) R->level[k][i] = TR_TOPO_NEVER;
            for (uint64_t j = 0; j < s->n[k + 1]; j++) {
                int64_t l = R->level[k + 1][j];
                if (l == TR_TOPO_NEVER) continue;
                for (uint64_t e = 0; e < 2 * (uint64_t)(k + 1); e++) {
                    uint64_t f = cx->face[k + 1][j * 2 * (uint64_t)(k + 1) + e];
                    if (R->level[k][f] == TR_TOPO_NEVER || l < R->level[k][f]) R->level[k][f] = l;
                }
            }
        }
        return TR_MATH_OK;
    }
    if (flt->source != TR_TOPO_LEVELS || !flt->levels) return TR_MATH_INVALID;   /* a chain has no value column */
    uint64_t at = 0;
    for (int32_t k = 0; k <= s->top; k++)
        for (uint64_t i = 0; i < s->n[k]; i++, at++) {
            int64_t l = flt->levels[at];
            if (l != TR_TOPO_NEVER && (l < 0 || (uint64_t)l >= m)) return TR_MATH_INVALID;
            R->level[k][i] = l;
        }
    for (int32_t k = 1; k <= s->top; k++)                       /* a filtration: no face after its coface */
        for (uint64_t j = 0; j < s->n[k]; j++) {
            int64_t l = R->level[k][j];
            if (l == TR_TOPO_NEVER) continue;
            for (uint64_t e = 0; e < src_len(s, k, j); e++) {
                int64_t lf = R->level[k - 1][src_row(s, k, j, e)];
                if (lf == TR_TOPO_NEVER || lf > l) return TR_MATH_DOMAIN_ERROR;
            }
        }
    return TR_MATH_OK;
}
/* the order of each dimension: entering cells by level (counting sort), ties by the declared rule */
static int32_t orders_fill(reducer *R, int32_t ties) {
    const source *s = R->src;
    int32_t rc = TR_MATH_OK;
    uint64_t *bucket = acct_alloc(&R->a, (R->nlevels + 1) * 8, &rc);
    if (rc) return rc;
    for (int32_t k = 0; k <= s->top; k++) {
        const uint64_t n = s->n[k];
        memset(bucket, 0, (size_t)(R->nlevels + 1) * 8);
        uint64_t m = 0;
        for (uint64_t i = 0; i < n; i++) {
            R->pos[k][i] = TR_TOPO_NONE;
            if (R->level[k][i] != TR_TOPO_NEVER) { bucket[R->level[k][i] + 1]++; m++; }
        }
        for (uint64_t l = 1; l <= R->nlevels; l++) bucket[l] += bucket[l - 1];
        for (uint64_t q = 0; q < n; q++) {
            uint64_t i = ties == TR_TOPO_TIES_REVERSE ? n - 1 - q : q;
            int64_t l = R->level[k][i];
            if (l == TR_TOPO_NEVER) continue;
            uint64_t p = bucket[l]++;
            R->order[k][p] = i;
            R->pos[k][i] = p;
        }
        R->m[k] = m;
    }
    acct_free(&R->a, bucket, (R->nlevels + 1) * 8);
    return TR_MATH_OK;
}
static void reducer_free(reducer *R) {
    for (int32_t k = 0; k < MAXK; k++) {
        acct_free(&R->a, R->level[k], R->src ? R->src->n[k] * 8 : 0);
        acct_free(&R->a, R->order[k], R->src ? R->src->n[k] * 8 : 0);
        acct_free(&R->a, R->pos[k], R->src ? R->src->n[k] * 8 : 0);
    }
    free(R->pairs); free(R->ess); free(R->reps); free(R->ecell); free(R->ecoef);
}
/* the full reduction of a source over a field; flt NULL: homology (one level) */
static int32_t reduce_all(reducer *R, const source *s, const fieldctx *fc, uint32_t track, const tr_topo_filtration *flt,
                          const tr_math_budget *b, tr_math_refusal *refusal) {
    memset(R, 0, sizeof *R);
    R->src = s; R->fc = *fc; R->track = track;
    R->nlevels = flt ? flt->nlevels : 1;
    R->a.limit_bytes = b->limit[TR_BUDGET_BYTES];
    R->a.limit_steps = b->limit[TR_BUDGET_ITERATIONS];
    R->a.refusal = refusal;
    int32_t rc = TR_MATH_OK;
    for (int32_t k = 0; k <= s->top; k++) {
        R->level[k] = acct_alloc(&R->a, s->n[k] * 8, &rc);
        R->order[k] = acct_alloc(&R->a, s->n[k] * 8, &rc);
        R->pos[k] = acct_alloc(&R->a, s->n[k] * 8, &rc);
    }
    if (rc) return rc;
    rc = levels_fill(R, flt);
    if (!rc) rc = orders_fill(R, flt ? flt->ties : TR_TOPO_TIES_INDEX);
    if (rc) return rc;
    uint64_t maxn = 1;
    for (int32_t k = 0; k <= s->top; k++) maxn = u64max(maxn, s->n[k]);
    uint64_t kw = (maxn + 63) / 64;
    uint64_t *killed = acct_alloc(&R->a, kw * 8, &rc), *next = acct_alloc(&R->a, kw * 8, &rc);
    if (rc) { acct_free(&R->a, killed, kw * 8); acct_free(&R->a, next, kw * 8); return rc; }
    int have_killed = 0;
    for (int32_t k = s->top; k >= 1 && !rc; k--) {
        memset(next, 0, (size_t)kw * 8);
        rc = reduce_dim(R, k, have_killed ? killed : NULL, next);
        uint64_t *t = killed; killed = next; next = t;
        have_killed = 1;
    }
    if (!rc) rc = reduce_vertices(R, have_killed ? killed : NULL);
    acct_free(&R->a, killed, kw * 8);
    acct_free(&R->a, next, kw * 8);
    if (rc) return rc;
    if (R->npairs > 1) qsort(R->pairs, (size_t)R->npairs, sizeof(tr_topo_pair), pair_order);   /* NULL when empty */
    if (R->ness > 1) qsort(R->ess, (size_t)R->ness, sizeof(tr_topo_pair), pair_order);
    return TR_MATH_OK;
}

/* ------------------------------------------------------------------ homology */
struct tr_topo_homology_s {
    uint64_t magic;
    tr_topo_homology_info info;
    tr_topo_class *classes;
    uint64_t *cells;
    int64_t *coefs;
};
static int homology_ok(const tr_topo_homology *h) { return h && h->magic == HOMOLOGY_MAGIC; }
void tr_topo_homology_free(tr_topo_homology *h) {
    if (!homology_ok(h)) return;
    free(h->classes); free(h->cells); free(h->coefs);
    h->magic = 0;
    free(h);
}
static tr_topo_homology *homology_fail(int32_t *error, int32_t rc) { set_error(error, rc); return NULL; }

static int32_t request_check(const source *s, const tr_topo_field *f, fieldctx *fc) {
    int32_t rc = field_of(f, fc);
    if (rc) return rc;
    if (f->method != TR_TOPO_REDUCTION && f->method != TR_TOPO_PACKED) return TR_MATH_INVALID;
    if (f->method == TR_TOPO_PACKED && (f->field != TR_TOPO_GF2 || f->cycles)) return TR_MATH_INVALID;
    if (s->top < 31 && (f->cycles >> (s->top + 1))) return TR_MATH_INVALID;
    return TR_MATH_OK;
}
tr_topo_homology *tr_topo_homology_build(const tr_topo_complex *cx, const tr_topo_chain *ch, const tr_topo_field *f,
                                         const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    source s;
    fieldctx fc;
    int32_t rc = source_of(cx, ch, &s);
    if (rc || !f) return homology_fail(error, TR_MATH_INVALID);
    rc = request_check(&s, f, &fc);
    if (rc) return homology_fail(error, rc);
    tr_math_budget dflt;
    const tr_math_budget *b = budget_or(budget, &dflt);
    tr_math_need need;
    rc = tr_topo_reduce_need(cx, ch, f, NULL, &need);
    if (!rc) rc = tr_budget_check(b, &need, refusal);
    if (rc) return homology_fail(error, rc);
    tr_topo_homology_info info;
    memset(&info, 0, sizeof info);
    info.top = s.top; info.field = f->field; info.method = f->method; info.kind = cx ? 1 : 2;
    info.prime = f->field == TR_TOPO_GFP ? f->prime : 0; info.cycles = f->cycles;
    int64_t euler = 0;
    for (int32_t k = 0; k <= s.top; k++) { info.n[k] = s.n[k]; euler += (k & 1) ? -(int64_t)s.n[k] : (int64_t)s.n[k]; }
    info.euler = euler;
    tr_topo_homology *h = NULL;
    if (f->method == TR_TOPO_PACKED) {                         /* FIELD-001: tr_field_rank of each packed d_k */
        uint64_t peak = 0;
        for (int32_t k = 1; k <= s.top; k++) {
            uint64_t rows = s.n[k - 1], cols = s.n[k], W = tr_field_words(cols), bytes = packed_bytes(rows, cols);
            uint64_t *words = zalloc(bytes);
            if (!words) return homology_fail(error, TR_MATH_MEMORY);
            tr_field_matrix M = {rows, cols, W, words};
            rc = packed_fill(&s, k, &M);
            if (!rc) rc = tr_field_rank(&M, b, &info.rank[k], refusal);
            free(words);
            if (rc) return homology_fail(error, rc);
            peak = u64max(peak, tr_budget_mul(bytes, 2));
        }
        for (int32_t k = 0; k <= s.top; k++) info.betti[k] = s.n[k] - info.rank[k] - info.rank[k + 1];
        info.peak_bytes = tr_budget_add(peak, 4096);
        h = zalloc(sizeof *h);
        if (!h) return homology_fail(error, TR_MATH_MEMORY);
        h->info = info;
        h->magic = HOMOLOGY_MAGIC;
        set_error(error, TR_MATH_OK);
        return h;
    }
    reducer R;
    rc = reduce_all(&R, &s, &fc, f->cycles, NULL, b, refusal);
    if (!rc) {
        rc = tr_budget_check_one(b, TR_BUDGET_OUTPUT, tr_budget_add(R.nreps, R.nentries), refusal);
    }
    if (!rc) {
        for (int32_t k = 0; k <= s.top + 1 && k < MAXK + 1; k++) info.rank[k] = R.rank[k];
        for (uint64_t i = 0; i < R.ness; i++) info.betti[R.ess[i].dim]++;
        for (int32_t k = 0; k <= s.top; k++)                   /* b_k = n_k - rank_k - rank_{k+1}, as it must be */
            if (info.betti[k] != s.n[k] - info.rank[k] - (k < s.top ? info.rank[k + 1] : 0)) rc = TR_MATH_INVALID;
        if (!rc) rc = acct_take(&R.a, tr_budget_add(tr_budget_mul(R.nreps, sizeof(tr_topo_class)),
                                                    tr_budget_mul(R.nentries, 16)));   /* the result's arrays */
        info.steps = R.a.steps; info.peak_bytes = R.a.peak;
        info.classes = R.nreps; info.entries = R.nentries;
        if (!rc) {
            h = zalloc(sizeof *h);
            if (!h) rc = TR_MATH_MEMORY;
        }
    }
    if (!rc && R.nreps) {
        qsort(R.reps, (size_t)R.nreps, sizeof(rep_rec), rep_order);
        h->classes = zalloc(R.nreps * sizeof(tr_topo_class));
        h->cells = zalloc(R.nentries * 8);
        h->coefs = zalloc(R.nentries * 8);
        if (!h->classes || !h->cells || !h->coefs) rc = TR_MATH_MEMORY;
        uint64_t at = 0;
        for (uint64_t i = 0; i < R.nreps && !rc; i++) {
            const rep_rec *r = &R.reps[i];
            tr_topo_class *c = &h->classes[i];
            c->dim = r->dim; c->reserved = 0; c->generator = r->generator; c->offset = at; c->length = r->length;
            memcpy(h->cells + at, R.ecell + r->offset, (size_t)r->length * 8);
            memcpy(h->coefs + at, R.ecoef + r->offset, (size_t)r->length * 8);
            at += r->length;
        }
    }
    reducer_free(&R);
    if (rc) {
        if (h) { free(h->classes); free(h->cells); free(h->coefs); free(h); }
        return homology_fail(error, rc);
    }
    h->info = info;
    h->magic = HOMOLOGY_MAGIC;
    set_error(error, TR_MATH_OK);
    return h;
}
int32_t tr_topo_homology_get_info(const tr_topo_homology *h, tr_topo_homology_info *out) {
    if (!homology_ok(h) || !out) return TR_MATH_INVALID;
    *out = h->info;
    return TR_MATH_OK;
}
int32_t tr_topo_homology_classes(const tr_topo_homology *h, tr_topo_class *out, uint64_t capacity) {
    if (!homology_ok(h) || capacity < h->info.classes || (h->info.classes && !out)) return TR_MATH_INVALID;
    if (h->info.classes) memcpy(out, h->classes, (size_t)h->info.classes * sizeof *out);
    return TR_MATH_OK;
}
int32_t tr_topo_homology_entries(const tr_topo_homology *h, uint64_t *cells, int64_t *coefs, uint64_t capacity) {
    if (!homology_ok(h) || capacity < h->info.entries || (h->info.entries && (!cells || !coefs))) return TR_MATH_INVALID;
    if (h->info.entries) { memcpy(cells, h->cells, (size_t)h->info.entries * 8); memcpy(coefs, h->coefs, (size_t)h->info.entries * 8); }
    return TR_MATH_OK;
}

/* ------------------------------------------------------------------ persistence */
struct tr_topo_persistence_s {
    uint64_t magic;
    tr_topo_persistence_info info;
    tr_topo_pair *pairs;              /* finite pairs then essential classes */
    int64_t *levels[MAXK];
};
static int persistence_ok(const tr_topo_persistence *p) { return p && p->magic == PERSIST_MAGIC; }
void tr_topo_persistence_free(tr_topo_persistence *p) {
    if (!persistence_ok(p)) return;
    free(p->pairs);
    for (int k = 0; k < MAXK; k++) free(p->levels[k]);
    p->magic = 0;
    free(p);
}
tr_topo_persistence *tr_topo_persistence_build(const tr_topo_complex *cx, const tr_topo_chain *ch, const tr_topo_field *f,
                                               const tr_topo_filtration *flt, const tr_math_budget *budget,
                                               tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    source s;
    fieldctx fc;
    int32_t rc = source_of(cx, ch, &s);
    if (rc || !f || !flt) { set_error(error, TR_MATH_INVALID); return NULL; }
    rc = request_check(&s, f, &fc);
    if (!rc && (f->method != TR_TOPO_REDUCTION || f->cycles)) rc = TR_MATH_INVALID;
    if (!rc && (flt->nlevels < 1 || (flt->ties != TR_TOPO_TIES_INDEX && flt->ties != TR_TOPO_TIES_REVERSE) ||
                (flt->source != TR_TOPO_SUBLEVEL && flt->source != TR_TOPO_SUPERLEVEL && flt->source != TR_TOPO_LEVELS)))
        rc = TR_MATH_INVALID;
    if (rc) { set_error(error, rc); return NULL; }
    tr_math_budget dflt;
    const tr_math_budget *b = budget_or(budget, &dflt);
    tr_math_need need;
    rc = tr_topo_reduce_need(cx, ch, f, flt, &need);
    if (!rc) rc = tr_budget_check(b, &need, refusal);
    if (rc) { set_error(error, rc); return NULL; }
    reducer R;
    rc = reduce_all(&R, &s, &fc, 0, flt, b, refusal);
    if (!rc) rc = tr_budget_check_one(b, TR_BUDGET_OUTPUT, tr_budget_add(R.npairs, R.ness), refusal);
    tr_topo_persistence *p = NULL;
    if (!rc) {
        uint64_t extra = tr_budget_mul(tr_budget_add(R.npairs, R.ness), sizeof(tr_topo_pair));
        for (int32_t k = 0; k <= s.top; k++) extra = tr_budget_add(extra, s.n[k] * 8);
        rc = acct_take(&R.a, extra);                          /* the result's own arrays */
    }
    if (!rc) {
        p = zalloc(sizeof *p);
        if (!p) rc = TR_MATH_MEMORY;
    }
    if (!rc) {
        tr_topo_persistence_info *in = &p->info;
        in->top = s.top; in->field = f->field; in->source = flt->source; in->ties = flt->ties;
        in->prime = f->field == TR_TOPO_GFP ? f->prime : 0; in->nlevels = flt->nlevels;
        for (int32_t k = 0; k <= s.top; k++) { in->entered[k] = R.m[k]; in->never[k] = s.n[k] - R.m[k]; }
        for (int32_t k = 0; k <= s.top + 1 && k < MAXK + 1; k++) in->rank[k] = R.rank[k];
        for (uint64_t i = 0; i < R.ness; i++) in->betti[R.ess[i].dim]++;
        in->pairs = R.npairs; in->essential = R.ness; in->zero_length = R.zero_length;
        in->steps = R.a.steps; in->peak_bytes = R.a.peak;
        p->pairs = zalloc((R.npairs + R.ness) * sizeof(tr_topo_pair));
        if (!p->pairs) rc = TR_MATH_MEMORY;
        else {
            if (R.npairs) memcpy(p->pairs, R.pairs, (size_t)R.npairs * sizeof(tr_topo_pair));
            if (R.ness) memcpy(p->pairs + R.npairs, R.ess, (size_t)R.ness * sizeof(tr_topo_pair));
        }
        for (int32_t k = 0; k <= s.top && !rc; k++) {
            p->levels[k] = zalloc(s.n[k] * 8);
            if (!p->levels[k]) rc = TR_MATH_MEMORY;
            else if (s.n[k]) memcpy(p->levels[k], R.level[k], (size_t)s.n[k] * 8);
        }
    }
    reducer_free(&R);
    if (rc) {
        if (p) { p->magic = PERSIST_MAGIC; tr_topo_persistence_free(p); }
        set_error(error, rc);
        return NULL;
    }
    p->magic = PERSIST_MAGIC;
    set_error(error, TR_MATH_OK);
    return p;
}
int32_t tr_topo_persistence_get_info(const tr_topo_persistence *p, tr_topo_persistence_info *out) {
    if (!persistence_ok(p) || !out) return TR_MATH_INVALID;
    *out = p->info;
    return TR_MATH_OK;
}
int32_t tr_topo_persistence_pairs(const tr_topo_persistence *p, tr_topo_pair *out, uint64_t capacity) {
    if (!persistence_ok(p)) return TR_MATH_INVALID;
    uint64_t n = p->info.pairs + p->info.essential;
    if (capacity < n || (n && !out)) return TR_MATH_INVALID;
    if (n) memcpy(out, p->pairs, (size_t)n * sizeof *out);
    return TR_MATH_OK;
}
int32_t tr_topo_persistence_betti(const tr_topo_persistence *p, int64_t level, uint64_t *betti, uint64_t capacity) {
    if (!persistence_ok(p) || level < 0 || (uint64_t)level >= p->info.nlevels || !betti ||
        capacity < (uint64_t)p->info.top + 1) return TR_MATH_INVALID;
    for (int32_t k = 0; k <= p->info.top; k++) betti[k] = 0;
    uint64_t n = p->info.pairs + p->info.essential;
    for (uint64_t i = 0; i < n; i++) {
        const tr_topo_pair *q = &p->pairs[i];
        if (q->birth <= level && (q->death == TR_TOPO_NEVER || q->death > level)) betti[q->dim]++;
    }
    return TR_MATH_OK;
}
int32_t tr_topo_persistence_levels(const tr_topo_persistence *p, int32_t k, int64_t *levels, uint64_t capacity) {
    if (!persistence_ok(p) || k < 0 || k > p->info.top) return TR_MATH_INVALID;
    uint64_t n = p->info.entered[k] + p->info.never[k];
    if (capacity < n || (n && !levels)) return TR_MATH_INVALID;
    if (n) memcpy(levels, p->levels[k], (size_t)n * 8);
    return TR_MATH_OK;
}
