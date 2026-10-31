/* DELTA-001: signed linear updates of maintained FND-004 bank states; see math_delta.h.
 * SPDX-License-Identifier: Apache-2.0 */
#include "math_delta.h"
#include "mathematics.h"
#include "math_domain.h"
#include "math_handles.h"
#include <stdlib.h>
#include <string.h>

TR_MATH_ABI(tr_delta_violation, 1);      /* the ctypes boundary of DELTA-001 (math_handles.h) */
TR_MATH_ABI(tr_delta_info, 1);
TR_MATH_ABI(tr_delta_batch_info, 1);
TR_MATH_ABI(tr_delta_rows, 1);

_Static_assert(sizeof(tr_delta_violation) == 64, "tr_delta_violation ABI");
_Static_assert(sizeof(tr_delta_info) == 128, "tr_delta_info ABI");
_Static_assert(sizeof(tr_delta_batch_info) == 104, "tr_delta_batch_info ABI");
_Static_assert(sizeof(tr_delta_rows) == 48, "tr_delta_rows ABI");

#define DELTA_MAGIC UINT32_C(0x41544c44)
#define STATE_MAGIC UINT32_C(0x54534c44)
#define SCHEMA_TAG UINT64_C(0x314d484353544c44)   /* "DLTSCHM1" */
#define MAXC TR_SCOPE_MAX_COLUMNS

struct tr_delta_s {
    uint32_t magic;
    int32_t kind, ncols;
    uint64_t k, ngroups, touched, inserted, deleted, wabs_total, bytes, revision;
    int64_t wnet;
    uint64_t identity[2], new_digest[2];
    int32_t scale[MAXC];
    int64_t *values[MAXC];
    uint8_t *valid[MAXC];           /* NULL: every value present */
    int64_t min[MAXC], max[MAXC];
    uint64_t nvalid[MAXC];
    int64_t *weights;
    uint32_t *gid, *local, *tgroups;
    uint64_t *wabs, *selection;
    unsigned char *schema, *old_key, *new_key;
    uint64_t schema_len, old_len, new_len;
};

struct tr_delta_state_s {
    uint32_t magic;
    int32_t ncols, nterms, mode, degree, promise;
    uint32_t joint;
    uint64_t ngroups, step, bytes, present, revision, table;
    int64_t total;
    uint64_t identity[2], base[2], source_digest[2];
    int32_t scale[MAXC];
    uint8_t *exps;
    tr_bank_term *terms;
    uint64_t *rows, *count;         /* int64 two's complement */
    tr_math_i128 *sum;
    unsigned char *schema, *source;
    uint64_t schema_len, source_len;
};

static void set_error(int32_t *error, int32_t code) { if (error) *error = code; }
static void *zalloc(uint64_t bytes) {
    if (bytes > (uint64_t)SIZE_MAX) return NULL;
    return calloc(1, bytes ? (size_t)bytes : 1);
}
static int delta_valid(const tr_delta *d) { return d && d->magic == DELTA_MAGIC; }
static int state_valid(const tr_delta_state *s) { return s && s->magic == STATE_MAGIC; }
static const tr_math_budget *budget_or(const tr_math_budget *b, tr_math_budget *dflt) {
    if (b) return b;
    tr_budget_default(dflt);
    return dflt;
}
static void need_zero(tr_math_need *need) { for (int i = 0; i < TR_BUDGET_RESOURCES; i++) need->need[i] = 0; }

/* ------------------------------------------------------------------ digests (identities; never an equality test) */
typedef struct { uint64_t h0, h1; } hasher;
static void h_init(hasher *h, uint64_t tag) {
    h->h0 = UINT64_C(0x243F6A8885A308D3) ^ tag; h->h1 = UINT64_C(0x13198A2E03707344) + tag;
}
static void h_word(hasher *h, uint64_t w) {
    h->h0 = (h->h0 ^ w) * UINT64_C(0x9E3779B97F4A7C15); h->h0 ^= h->h0 >> 29;
    h->h1 = (h->h1 + w) * UINT64_C(0xBF58476D1CE4E5B9); h->h1 ^= h->h1 >> 31; h->h1 += UINT64_C(0x94D049BB133111EB);
}
static void h_bytes(hasher *h, const void *p, uint64_t len) {
    const unsigned char *b = (const unsigned char *)p;
    h_word(h, len);
    for (; len >= 8; len -= 8, b += 8) { uint64_t w; memcpy(&w, b, 8); h_word(h, w); }
    if (len) { uint64_t w = 0; memcpy(&w, b, (size_t)len); h_word(h, w); }
}
static void h_out(const hasher *h, uint64_t out[2]) { out[0] = h->h0 ^ (h->h1 >> 17); out[1] = h->h1 ^ (h->h0 << 13); }

/* ------------------------------------------------------------------ schemas: how a scope encodes ids and values */
static uint64_t schema_size(const tr_scope *s) {
    tr_scope_info info;
    tr_scope_get_info(s, &info);
    uint64_t n = tr_budget_add(32, tr_budget_mul((uint64_t)info.ncols, 48));
    if (info.source == TR_SCOPE_TABLE)
        n = tr_budget_add(n, tr_budget_add(32, tr_scope_key_size(s, TR_SCOPE_KEY_GROUPS)));
    return n;
}
static int32_t schema_write(const tr_scope *s, unsigned char **out, uint64_t *len) {
    tr_scope_info info;
    tr_scope_get_info(s, &info);
    uint64_t n = schema_size(s), at = 0;
    unsigned char *b = zalloc(n);
    if (!b) return TR_MATH_MEMORY;
#define PUT(v) do { uint64_t _v = (uint64_t)(v); memcpy(b + at, &_v, 8); at += 8; } while (0)
    PUT(SCHEMA_TAG); PUT(info.source); PUT(info.ncols); PUT(info.ngroups);
    for (int32_t c = 0; c < info.ncols; c++) {
        tr_scope_column_info ci;
        tr_scope_column_get(s, c, &ci);
        PUT(ci.scale);
        if (info.source == TR_SCOPE_TABLE) { PUT((int64_t)ci.col); PUT((int64_t)ci.valid); PUT((int64_t)ci.hi); PUT((int64_t)ci.shift); PUT(ci.offset); }
        else { PUT(0); PUT(0); PUT(0); PUT(0); PUT(0); }
    }
    if (info.source == TR_SCOPE_TABLE) {
        uint64_t id[2], g = tr_scope_key_size(s, TR_SCOPE_KEY_GROUPS);
        memcpy(id, info.identity, 16);
        PUT(info.table); PUT(id[0]); PUT(id[1]); PUT(g);
        int32_t rc = g ? tr_scope_key_write(s, TR_SCOPE_KEY_GROUPS, b + at, g) : TR_MATH_MEMORY;
        if (rc) { free(b); return rc; }
        at += g;
    }
#undef PUT
    *out = b; *len = n;
    return TR_MATH_OK;
}
static int bytes_equal(const unsigned char *a, uint64_t na, const unsigned char *b, uint64_t nb) {
    return na == nb && (!na || !memcmp(a, b, (size_t)na));
}

/* ------------------------------------------------------------------ deltas */
static void delta_release(tr_delta *d) {
    if (!d) return;
    for (int c = 0; c < MAXC; c++) { free(d->values[c]); free(d->valid[c]); }
    free(d->weights); free(d->gid); free(d->local); free(d->tgroups); free(d->wabs); free(d->selection);
    free(d->schema); free(d->old_key); free(d->new_key);
    d->magic = 0;
    free(d);
}
void tr_delta_free(tr_delta *d) { if (delta_valid(d)) delta_release(d); }

/* every allocation of a delta of k rows (validity arrays for the columns flagged in nullmask) */
static uint64_t delta_bytes(uint64_t k, int32_t ncols, uint32_t nullmask, uint64_t extra) {
    uint64_t b = tr_budget_add(sizeof(tr_delta), tr_budget_mul(k, 8 + 4 + 4 + 4 + 8 + 8));   /* weights .. scratch */
    b = tr_budget_add(b, tr_budget_bitmap_bytes(k));
    for (int32_t c = 0; c < ncols; c++) {
        b = tr_budget_add(b, tr_budget_mul(k, 8));
        if ((nullmask >> c) & 1) b = tr_budget_add(b, k);
    }
    return tr_budget_add(b, extra);
}
static void delta_need(uint64_t k, int32_t ncols, uint32_t nullmask, uint64_t extra, tr_math_need *need) {
    need_zero(need);
    need->need[TR_BUDGET_OUTPUT] = k;
    need->need[TR_BUDGET_STATES] = k;
    need->need[TR_BUDGET_BYTES] = delta_bytes(k, ncols, nullmask, extra);
}
/* an empty delta of k rows: values (and validity for nullmask) allocated, zero */
static tr_delta *delta_alloc(uint64_t k, int32_t ncols, const int32_t *scales, uint64_t ngroups, int32_t kind,
                             uint32_t nullmask) {
    tr_delta *d = zalloc(sizeof *d);
    if (!d) return NULL;
    d->k = k; d->ncols = ncols; d->ngroups = ngroups; d->kind = kind;
    for (int32_t c = 0; c < ncols; c++) {
        d->scale[c] = scales[c];
        d->values[c] = zalloc(k * 8);
        if ((nullmask >> c) & 1) d->valid[c] = zalloc(k);
        if (!d->values[c] || (((nullmask >> c) & 1) && !d->valid[c])) { delta_release(d); return NULL; }
    }
    d->weights = zalloc(k * 8); d->gid = zalloc(k * 4); d->local = zalloc(k * 4);
    d->tgroups = zalloc(k * 4); d->wabs = zalloc(k * 8); d->selection = zalloc(((k + 63) / 64) * 8);
    if (!d->weights || !d->gid || !d->local || !d->tgroups || !d->wabs || !d->selection) { delta_release(d); return NULL; }
    return d;
}
static int cmp_u64(const void *a, const void *b) {
    uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
    return x < y ? -1 : x > y;
}
static uint64_t sat_add(uint64_t a, uint64_t b) { return a > UINT64_MAX - b ? UINT64_MAX : a + b; }
/* the touched groups, local ids, weight totals, extremes, validity and identity of a filled delta */
static int32_t delta_finish(tr_delta *d) {
    const uint64_t k = d->k;
    uint64_t *keys = zalloc(k * 8);
    if (!keys) return TR_MATH_MEMORY;
    for (uint64_t r = 0; r < k; r++) keys[r] = ((uint64_t)d->gid[r] << 32) | r;
    if (k > 1) qsort(keys, (size_t)k, 8, cmp_u64);
    uint64_t t = 0;
    for (uint64_t i = 0; i < k; i++) {
        uint32_t g = (uint32_t)(keys[i] >> 32), r = (uint32_t)keys[i];
        if (!t || d->tgroups[t - 1] != g) { d->tgroups[t] = g; d->wabs[t] = 0; t++; }
        d->local[r] = (uint32_t)(t - 1);
        d->wabs[t - 1] = sat_add(d->wabs[t - 1], tr_delta_mag64(d->weights[r]));
    }
    free(keys);
    d->touched = t;
    int64_t net = 0;
    for (uint64_t r = 0; r < k; r++) {
        int64_t w = d->weights[r];
        d->inserted += w > 0; d->deleted += w < 0;
        d->wabs_total = sat_add(d->wabs_total, tr_delta_mag64(w));
        if (__builtin_add_overflow(net, w, &net)) net = w > 0 ? INT64_MAX : INT64_MIN;
        d->selection[r >> 6] |= UINT64_C(1) << (r & 63);
    }
    d->wnet = net;
    for (int32_t c = 0; c < d->ncols; c++) {
        d->min[c] = INT64_MAX; d->max[c] = INT64_MIN; d->nvalid[c] = 0;
        for (uint64_t r = 0; r < k; r++) {
            if (d->valid[c] && !d->valid[c][r]) { d->values[c][r] = 0; continue; }   /* normalized */
            int64_t v = d->values[c][r];
            if (v < d->min[c]) d->min[c] = v;
            if (v > d->max[c]) d->max[c] = v;
            d->nvalid[c]++;
        }
        if (d->valid[c] && d->nvalid[c] == k) { free(d->valid[c]); d->valid[c] = NULL; }
    }
    hasher h;
    h_init(&h, UINT64_C(0x4154414441544c44));
    h_word(&h, (uint64_t)d->kind); h_word(&h, (uint64_t)d->ncols); h_word(&h, d->ngroups); h_word(&h, k);
    h_bytes(&h, d->scale, sizeof(int32_t) * (uint64_t)d->ncols);
    h_bytes(&h, d->gid, k * 4); h_bytes(&h, d->weights, k * 8);
    for (int32_t c = 0; c < d->ncols; c++) {
        h_word(&h, d->valid[c] != NULL);
        h_bytes(&h, d->values[c], k * 8);
        if (d->valid[c]) h_bytes(&h, d->valid[c], k);
    }
    if (d->schema) h_bytes(&h, d->schema, d->schema_len);
    if (d->old_key) { h_bytes(&h, d->old_key, d->old_len); h_bytes(&h, d->new_key, d->new_len); }
    h_out(&h, d->identity);
    uint64_t bytes = tr_budget_add(sizeof *d, tr_budget_mul(k, 8 + 4 + 4 + 4 + 8));
    bytes = tr_budget_add(bytes, tr_budget_bitmap_bytes(k));
    for (int32_t c = 0; c < d->ncols; c++) bytes = tr_budget_add(bytes, tr_budget_add(k * 8, d->valid[c] ? k : 0));
    d->bytes = tr_budget_add(bytes, tr_budget_add(d->schema_len, tr_budget_add(d->old_len, d->new_len)));
    d->magic = DELTA_MAGIC;
    return TR_MATH_OK;
}

static int32_t arrays_check(uint64_t n, const tr_scope_array *cols, int32_t ncols, const uint32_t *groups,
                            uint64_t ngroups, const int64_t *weights) {
    if (ncols < 0 || ncols > MAXC || (ncols && !cols) || ngroups < 1 || (n && !weights) ||
        (n && ngroups > 1 && !groups)) return TR_MATH_INVALID;
    if (ngroups > TR_SCOPE_MAX_GROUPS || n >= (UINT64_C(1) << 32)) return TR_MATH_UNSUPPORTED;
    for (int32_t c = 0; c < ncols; c++)
        if ((n && !cols[c].values) || cols[c].scale < 0 || cols[c].scale > TR_SCOPE_MAX_SCALE) return TR_MATH_INVALID;
    return TR_MATH_OK;
}
static uint32_t arrays_nullmask(const tr_scope_array *cols, int32_t ncols) {
    uint32_t m = 0;
    for (int32_t c = 0; c < ncols; c++) if (cols[c].valid) m |= UINT32_C(1) << c;
    return m;
}
int32_t tr_delta_need_arrays(uint64_t n, const tr_scope_array *cols, int32_t ncols, tr_math_need *need) {
    if (!need || ncols < 0 || ncols > MAXC || (ncols && !cols)) return TR_MATH_INVALID;
    delta_need(n, ncols, arrays_nullmask(cols, ncols), tr_budget_add(32, tr_budget_mul((uint64_t)ncols, 48)), need);
    return TR_MATH_OK;
}

/* the schema of an array delta: its kind, columns, scales and groups */
static int32_t arrays_schema(tr_delta *d) {
    uint64_t n = tr_budget_add(32, tr_budget_mul((uint64_t)d->ncols, 48));
    unsigned char *b = zalloc(n);
    if (!b) return TR_MATH_MEMORY;
    uint64_t head[4] = {SCHEMA_TAG, TR_SCOPE_ARRAYS, (uint64_t)d->ncols, d->ngroups};
    memcpy(b, head, 32);
    for (int32_t c = 0; c < d->ncols; c++) { uint64_t s = (uint64_t)(int64_t)d->scale[c]; memcpy(b + 32 + 48 * (uint64_t)c, &s, 8); }
    d->schema = b; d->schema_len = n;
    return TR_MATH_OK;
}

tr_delta *tr_delta_from_arrays(uint64_t n, const tr_scope_array *cols, int32_t ncols, const uint32_t *groups,
                               uint64_t ngroups, const int64_t *weights, const tr_math_budget *budget,
                               tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    int32_t rc = arrays_check(n, cols, ncols, groups, ngroups, weights);
    if (rc) { set_error(error, rc); return NULL; }
    tr_math_need need;
    delta_need(n, ncols, arrays_nullmask(cols, ncols), tr_budget_add(32, tr_budget_mul((uint64_t)ncols, 48)), &need);
    tr_math_budget dflt;
    rc = tr_budget_check(budget_or(budget, &dflt), &need, refusal);
    if (rc) { set_error(error, rc); return NULL; }
    for (uint64_t r = 0; groups && r < n; r++) if ((uint64_t)groups[r] >= ngroups) { set_error(error, TR_MATH_INVALID); return NULL; }
    int32_t scales[MAXC] = {0};
    for (int32_t c = 0; c < ncols; c++) scales[c] = cols[c].scale;
    tr_delta *d = delta_alloc(n, ncols, scales, ngroups, TR_DELTA_ARRAYS, arrays_nullmask(cols, ncols));
    if (!d) { set_error(error, TR_MATH_MEMORY); return NULL; }
    for (int32_t c = 0; c < ncols; c++) {
        if (n) memcpy(d->values[c], cols[c].values, (size_t)n * 8);
        for (uint64_t r = 0; d->valid[c] && r < n; r++) d->valid[c][r] = cols[c].valid[r] != 0;
    }
    for (uint64_t r = 0; r < n; r++) { d->gid[r] = groups ? groups[r] : 0; d->weights[r] = weights[r]; }
    rc = arrays_schema(d);
    if (!rc) rc = delta_finish(d);
    if (rc) { delta_release(d); set_error(error, rc); return NULL; }
    set_error(error, TR_MATH_OK);
    return d;
}

/* the columns of a scope whose selected rows have NULLs */
static uint32_t view_nullmask(const tr_scope_view *v) {
    uint32_t m = 0;
    for (int32_t c = 0; c < v->ncols; c++) if (v->col[c].valid) m |= UINT32_C(1) << c;
    return m;
}
static int32_t scope_schema_need(const tr_scope *s, uint64_t *bytes) {
    *bytes = schema_size(s);
    return *bytes ? TR_MATH_OK : TR_MATH_MEMORY;
}
int32_t tr_delta_need_scope(const tr_scope *scope, tr_math_need *need) {
    tr_scope_view v;
    uint64_t sb;
    if (!need || !scope || tr_scope_get_view(scope, &v) || scope_schema_need(scope, &sb)) return TR_MATH_INVALID;
    delta_need(v.selected, v.ncols, view_nullmask(&v), sb, need);
    return TR_MATH_OK;
}
/* copies the contribution of row r of a scope into delta row i with weight w */
static void put_row(tr_delta *d, uint64_t i, const tr_scope_view *v, uint64_t r, int64_t w) {
    for (int32_t c = 0; c < v->ncols; c++) {
        int present = tr_scope_present(v, c, r);
        if (d->valid[c]) d->valid[c][i] = (uint8_t)present;
        d->values[c][i] = present ? tr_scope_value(v, c, r) : 0;
    }
    d->gid[i] = tr_scope_group(v, r);
    d->weights[i] = w;
}

tr_delta *tr_delta_from_scope(const tr_scope *scope, const int64_t *weights, int32_t sign, const tr_math_budget *budget,
                              tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    tr_scope_view v;
    tr_scope_info info;
    if (!scope || tr_scope_get_view(scope, &v) || tr_scope_get_info(scope, &info) || (!weights && sign != 1 && sign != -1)) {
        set_error(error, TR_MATH_INVALID); return NULL;
    }
    if (v.selected >= (UINT64_C(1) << 32)) { set_error(error, TR_MATH_UNSUPPORTED); return NULL; }
    tr_math_need need;
    int32_t rc = tr_delta_need_scope(scope, &need);
    tr_math_budget dflt;
    if (!rc) rc = tr_budget_check(budget_or(budget, &dflt), &need, refusal);
    if (rc) { set_error(error, rc); return NULL; }
    int32_t scales[MAXC] = {0};
    for (int32_t c = 0; c < v.ncols; c++) scales[c] = v.col[c].scale;
    tr_delta *d = delta_alloc(v.selected, v.ncols, scales, v.ngroups, TR_DELTA_SCOPE, view_nullmask(&v));
    if (!d) { set_error(error, TR_MATH_MEMORY); return NULL; }
    uint64_t i = 0;
    for (uint64_t w = 0; w < (v.rows + 63) / 64; w++) {
        uint64_t bits = v.selection[w];
        while (bits) {
            uint64_t r = (w << 6) | (uint64_t)__builtin_ctzll(bits);
            bits &= bits - 1;
            put_row(d, i++, &v, r, weights ? weights[r] : sign);
        }
    }
    d->revision = info.revision;
    rc = schema_write(scope, &d->schema, &d->schema_len);
    if (!rc) rc = delta_finish(d);
    if (rc) { delta_release(d); set_error(error, rc); return NULL; }
    set_error(error, TR_MATH_OK);
    return d;
}

/* 1 when row r contributes the same to both scopes (selected in both, same group, validity and values) */
static int same_row(const tr_scope_view *a, const tr_scope_view *b, uint64_t r) {
    if (tr_scope_group(a, r) != tr_scope_group(b, r)) return 0;
    for (int32_t c = 0; c < a->ncols; c++) {
        int pa = tr_scope_present(a, c, r), pb = tr_scope_present(b, c, r);
        if (pa != pb || (pa && tr_scope_value(a, c, r) != tr_scope_value(b, c, r))) return 0;
    }
    return 1;
}
/* 1 when two table scopes read the same table at the same version (then every row has the same codes, so a row
 * selected in both contributes the same: only the selection can differ) */
static int same_version(const tr_scope *o, const tr_scope *n) {
    tr_scope_info a, b;
    if (tr_scope_get_info(o, &a) || tr_scope_get_info(n, &b)) return 0;
    return a.source == TR_SCOPE_TABLE && b.source == TR_SCOPE_TABLE && a.table == b.table &&
           !memcmp(a.identity, b.identity, 16) && a.revision == b.revision && a.version[0] == b.version[0] &&
           a.version[1] == b.version[1];
}
/* visits the rows of the diff in order: emit(row, in old, in new) for every row whose contribution differs */
typedef struct { tr_delta *d; uint64_t i; const tr_scope_view *ov, *nv; } diff_sink;
static void diff_emit(diff_sink *k, uint64_t r, int os, int ns) {
    if (!k->d) { k->i += (uint64_t)os + (uint64_t)ns; return; }
    if (os) put_row(k->d, k->i++, k->ov, r, -1);     /* the old version leaves */
    if (ns) put_row(k->d, k->i++, k->nv, r, +1);     /* the new one arrives (a move: both) */
}
static void diff_walk(const tr_scope_view *ov, const tr_scope_view *nv, int same, diff_sink *k) {
    for (uint64_t w = 0; w < (ov->rows + 63) / 64; w++) {
        uint64_t a = ov->selection[w], b = nv->selection[w];
        uint64_t bits = same ? (a ^ b) : (a | b);    /* same version: only the selection can differ */
        while (bits) {
            uint64_t r = (w << 6) | (uint64_t)__builtin_ctzll(bits);
            bits &= bits - 1;
            int os = (int)((a >> (r & 63)) & 1), ns = (int)((b >> (r & 63)) & 1);
            if (os && ns && same_row(ov, nv, r)) continue;
            diff_emit(k, r, os, ns);
        }
    }
}
/* the rows of the diff and the schema check */
static int32_t diff_count(const tr_scope *o, const tr_scope *n, tr_scope_view *ov, tr_scope_view *nv, uint64_t *rows) {
    if (!o || !n || tr_scope_get_view(o, ov) || tr_scope_get_view(n, nv)) return TR_MATH_INVALID;
    unsigned char *a = NULL, *b = NULL;
    uint64_t la = 0, lb = 0;
    int32_t rc = schema_write(o, &a, &la);
    if (!rc) rc = schema_write(n, &b, &lb);
    int same = !rc && bytes_equal(a, la, b, lb) && ov->rows == nv->rows;
    free(a); free(b);
    if (rc) return rc;
    if (!same) return TR_MATH_DOMAIN_ERROR;            /* another table, column specs or grouping */
    diff_sink k = {NULL, 0, ov, nv};
    diff_walk(ov, nv, same_version(o, n), &k);
    *rows = k.i;
    return TR_MATH_OK;
}
int32_t tr_delta_need_between(const tr_scope *o, const tr_scope *n, tr_math_need *need) {
    tr_scope_view ov, nv;
    uint64_t k = 0;
    if (!need) return TR_MATH_INVALID;
    int32_t rc = diff_count(o, n, &ov, &nv, &k);
    if (rc) return rc;
    uint64_t extra = tr_budget_add(schema_size(o), tr_budget_add(tr_scope_key_size(o, TR_SCOPE_KEY_ALL),
                                                                 tr_scope_key_size(n, TR_SCOPE_KEY_ALL)));
    delta_need(k, ov.ncols, view_nullmask(&ov) | view_nullmask(&nv), extra, need);
    return TR_MATH_OK;
}

tr_delta *tr_delta_between(const tr_scope *o, const tr_scope *n, const tr_math_budget *budget, tr_math_refusal *refusal,
                           int32_t *error) {
    tr_budget_none(refusal);
    tr_scope_view ov, nv;
    tr_math_need need;
    int32_t rc = tr_delta_need_between(o, n, &need);
    if (rc) { set_error(error, rc); return NULL; }
    uint64_t k = need.need[TR_BUDGET_OUTPUT];
    if (k >= (UINT64_C(1) << 32)) { set_error(error, TR_MATH_UNSUPPORTED); return NULL; }
    tr_math_budget dflt;
    rc = tr_budget_check(budget_or(budget, &dflt), &need, refusal);
    if (rc) { set_error(error, rc); return NULL; }
    tr_scope_get_view(o, &ov);
    tr_scope_get_view(n, &nv);
    int32_t scales[MAXC] = {0};
    for (int32_t c = 0; c < ov.ncols; c++) scales[c] = ov.col[c].scale;
    tr_delta *d = delta_alloc(k, ov.ncols, scales, ov.ngroups, TR_DELTA_DIFF, view_nullmask(&ov) | view_nullmask(&nv));
    if (!d) { set_error(error, TR_MATH_MEMORY); return NULL; }
    diff_sink sink = {d, 0, &ov, &nv};
    diff_walk(&ov, &nv, same_version(o, n), &sink);
    tr_scope_info ni;
    tr_scope_get_info(n, &ni);
    d->revision = ni.revision; d->new_digest[0] = ni.digest[0]; d->new_digest[1] = ni.digest[1];
    d->old_len = tr_scope_key_size(o, TR_SCOPE_KEY_ALL);
    d->new_len = tr_scope_key_size(n, TR_SCOPE_KEY_ALL);
    d->old_key = zalloc(d->old_len); d->new_key = zalloc(d->new_len);
    rc = (!d->old_len || !d->new_len || !d->old_key || !d->new_key) ? TR_MATH_MEMORY : TR_MATH_OK;
    if (!rc) rc = tr_scope_key_write(o, TR_SCOPE_KEY_ALL, d->old_key, d->old_len);
    if (!rc) rc = tr_scope_key_write(n, TR_SCOPE_KEY_ALL, d->new_key, d->new_len);
    if (!rc) rc = schema_write(o, &d->schema, &d->schema_len);
    if (!rc) rc = delta_finish(d);
    if (rc) { delta_release(d); set_error(error, rc); return NULL; }
    set_error(error, TR_MATH_OK);
    return d;
}

int32_t tr_delta_get_info(const tr_delta *d, tr_delta_batch_info *out) {
    if (!delta_valid(d) || !out) return TR_MATH_INVALID;
    memset(out, 0, sizeof *out);
    out->rows = d->k; out->ngroups = d->ngroups; out->touched = d->touched; out->inserted = d->inserted;
    out->deleted = d->deleted; out->wabs = d->wabs_total; out->wnet = d->wnet;
    out->identity[0] = d->identity[0]; out->identity[1] = d->identity[1];
    out->new_digest[0] = d->new_digest[0]; out->new_digest[1] = d->new_digest[1];
    out->revision = d->revision; out->kind = d->kind; out->ncols = d->ncols;
    return TR_MATH_OK;
}
int32_t tr_delta_get_view(const tr_delta *d, tr_scope_view *v) {
    if (!delta_valid(d) || !v) return TR_MATH_INVALID;
    memset(v, 0, sizeof *v);
    v->rows = d->k; v->selected = d->k; v->ngroups = d->touched ? d->touched : 1;
    v->selection = d->selection; v->groups = d->local; v->ncols = d->ncols;
    for (int32_t c = 0; c < d->ncols; c++) {
        tr_scope_view_column *k = &v->col[c];
        k->values = d->values[c]; k->valid = d->valid[c]; k->offset = 0;
        k->min = d->min[c]; k->max = d->max[c]; k->nvalid = d->nvalid[c]; k->scale = d->scale[c];
    }
    return TR_MATH_OK;
}
int32_t tr_delta_get_rows(const tr_delta *d, tr_delta_rows *out) {
    if (!delta_valid(d) || !out) return TR_MATH_INVALID;
    memset(out, 0, sizeof *out);
    out->rows = d->k; out->touched = d->touched; out->weights = d->weights; out->local = d->local;
    out->groups = d->tgroups; out->wabs = d->wabs;
    return TR_MATH_OK;
}
int32_t tr_delta_get_groups(const tr_delta *d, uint32_t *out, uint64_t capacity) {
    if (!delta_valid(d) || (d->k && (!out || capacity < d->k))) return TR_MATH_INVALID;
    if (d->k) memcpy(out, d->gid, (size_t)d->k * 4);
    return TR_MATH_OK;
}

/* ------------------------------------------------------------------ states */
static void state_release(tr_delta_state *s) {
    if (!s) return;
    free(s->exps); free(s->terms); free(s->rows); free(s->count); free(s->sum); free(s->schema); free(s->source);
    s->magic = 0;
    free(s);
}
void tr_delta_state_free(tr_delta_state *s) { if (state_valid(s)) state_release(s); }

static uint64_t state_bytes(uint64_t ngroups, int32_t nterms, int32_t ncols, uint64_t schema, uint64_t source) {
    uint64_t b = tr_budget_add(sizeof(tr_delta_state), tr_bank_state_bytes(ngroups, nterms));
    b = tr_budget_add(b, tr_budget_add(tr_budget_mul((uint64_t)nterms, sizeof(tr_bank_term)),
                                       tr_budget_mul((uint64_t)nterms, (uint64_t)ncols)));
    return tr_budget_add(b, tr_budget_add(schema, source));
}
/* a state with the shape and basis of `like` (cells allocated, schema copied; source not set) */
static tr_delta_state *state_alloc(const tr_delta_state *like, uint64_t ngroups, int32_t ncols, int32_t nterms) {
    tr_delta_state *s = zalloc(sizeof *s);
    if (!s) return NULL;
    s->ngroups = ngroups; s->ncols = ncols; s->nterms = nterms;
    uint64_t gm = ngroups * (uint64_t)nterms;
    s->rows = zalloc(ngroups * 8); s->count = zalloc(gm * 8); s->sum = zalloc(gm * 16);
    s->exps = zalloc((uint64_t)nterms * (uint64_t)ncols); s->terms = zalloc((uint64_t)nterms * sizeof(tr_bank_term));
    if (!s->rows || !s->count || !s->sum || !s->exps || !s->terms) { state_release(s); return NULL; }
    if (like) {
        s->mode = like->mode; s->degree = like->degree; s->promise = like->promise; s->joint = like->joint;
        memcpy(s->scale, like->scale, sizeof s->scale);
        if (nterms && ncols) memcpy(s->exps, like->exps, (size_t)nterms * (size_t)ncols);
        if (nterms) memcpy(s->terms, like->terms, (size_t)nterms * sizeof(tr_bank_term));
        s->schema = zalloc(like->schema_len);
        if (!s->schema) { state_release(s); return NULL; }
        memcpy(s->schema, like->schema, (size_t)like->schema_len);
        s->schema_len = like->schema_len;
    }
    return s;
}
static void state_tally(tr_delta_state *s) {
    uint64_t present = 0;
    int64_t total = 0;
    for (uint64_t g = 0; g < s->ngroups; g++) {
        int64_t r = (int64_t)s->rows[g];
        present += r != 0;
        if (__builtin_add_overflow(total, r, &total)) total = r > 0 ? INT64_MAX : INT64_MIN;
    }
    s->present = present; s->total = total;
    s->bytes = state_bytes(s->ngroups, s->nterms, s->ncols, s->schema_len, s->source_len);
}
/* the bank's scope check: scoped, the scope's digest, groups, columns and rows */
static int bank_of(const tr_bank *b, const tr_scope *s) {
    tr_bank_info bi;
    tr_scope_info si;
    if (tr_bank_get_info(b, &bi) || tr_scope_get_info(s, &si)) return 0;
    return bi.scoped && bi.digest[0] == si.digest[0] && bi.digest[1] == si.digest[1] && bi.ngroups == si.ngroups &&
           bi.ncols == si.ncols && bi.rows == si.selected;
}

tr_delta_state *tr_delta_state_from_bank(const tr_bank *bank, const tr_scope *scope, int32_t promise,
                                         const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    tr_bank_info bi;
    tr_scope_info si;
    if (!bank || !scope || tr_bank_get_info(bank, &bi) || tr_scope_get_info(scope, &si) ||
        (promise != TR_DELTA_NONNEGATIVE && promise != TR_DELTA_SIGNED)) { set_error(error, TR_MATH_INVALID); return NULL; }
    if (!bank_of(bank, scope)) { set_error(error, TR_MATH_DOMAIN_ERROR); return NULL; }     /* another scope's bank */
    uint64_t sl = schema_size(scope), kl = tr_scope_key_size(scope, TR_SCOPE_KEY_ALL);
    tr_math_need need;
    need_zero(&need);
    need.need[TR_BUDGET_SUPPORT] = (uint64_t)bi.nterms;
    need.need[TR_BUDGET_STATES] = tr_budget_mul(bi.ngroups, (uint64_t)bi.nterms);
    need.need[TR_BUDGET_BYTES] = state_bytes(bi.ngroups, bi.nterms, bi.ncols, sl, kl);
    tr_math_budget dflt;
    int32_t rc = tr_budget_check(budget_or(budget, &dflt), &need, refusal);
    if (rc) { set_error(error, rc); return NULL; }
    tr_delta_state *s = state_alloc(NULL, bi.ngroups, bi.ncols, bi.nterms);
    if (!s) { set_error(error, TR_MATH_MEMORY); return NULL; }
    s->mode = bi.mode; s->degree = bi.degree; s->joint = bi.joint; s->promise = promise;
    tr_scope_view v;
    tr_scope_get_view(scope, &v);
    for (int32_t c = 0; c < bi.ncols; c++) s->scale[c] = v.col[c].scale;
    uint64_t ne = (uint64_t)bi.nterms * (uint64_t)bi.ncols;
    rc = tr_bank_get_exponents(bank, s->exps, ne);
    uint32_t joint = 0;
    int32_t degree = 0;
    if (!rc) rc = tr_bank_compile(s->exps, bi.nterms, bi.ncols, s->terms, &joint, &degree);
    tr_bank_states st;
    if (!rc) rc = tr_bank_get_states(bank, &st);
    if (rc) { state_release(s); set_error(error, rc); return NULL; }
    uint64_t gm = bi.ngroups * (uint64_t)bi.nterms;
    memcpy(s->rows, st.rows, (size_t)bi.ngroups * 8);
    if (gm) { memcpy(s->count, st.count, (size_t)gm * 8); memcpy(s->sum, st.sum, (size_t)gm * 16); }
    rc = schema_write(scope, &s->schema, &s->schema_len);
    s->source = zalloc(kl);
    if (!rc && (!kl || !s->source)) rc = TR_MATH_MEMORY;
    if (!rc) rc = tr_scope_key_write(scope, TR_SCOPE_KEY_ALL, s->source, kl);
    if (rc) { state_release(s); set_error(error, rc); return NULL; }
    s->source_len = kl; s->source_digest[0] = si.digest[0]; s->source_digest[1] = si.digest[1];
    s->revision = si.revision; s->table = si.table;
    hasher h;                                         /* the identity of a first state: its source and its cells */
    h_init(&h, UINT64_C(0x45534142544c4544));
    h_word(&h, si.digest[0]); h_word(&h, si.digest[1]); h_word(&h, (uint64_t)s->mode); h_word(&h, (uint64_t)promise);
    h_bytes(&h, s->exps, ne); h_bytes(&h, s->rows, bi.ngroups * 8);
    h_bytes(&h, s->count, gm * 8); h_bytes(&h, s->sum, gm * 16);
    h_out(&h, s->identity);
    s->base[0] = s->identity[0]; s->base[1] = s->identity[1];
    state_tally(s);
    s->magic = STATE_MAGIC;
    set_error(error, TR_MATH_OK);
    return s;
}

/* a delta applies to a state: same columns, scales and groups; a scope delta of the same schema; a diff from exactly
 * the state's source */
static int32_t compatible(const tr_delta_state *s, const tr_delta *d) {
    if (!state_valid(s) || !delta_valid(d)) return TR_MATH_INVALID;
    if (d->ncols != s->ncols || d->ngroups != s->ngroups) return TR_MATH_DOMAIN_ERROR;
    for (int32_t c = 0; c < s->ncols; c++) if (d->scale[c] != s->scale[c]) return TR_MATH_DOMAIN_ERROR;
    if (d->kind == TR_DELTA_SCOPE || d->kind == TR_DELTA_DIFF) {
        if (!bytes_equal(d->schema, d->schema_len, s->schema, s->schema_len)) return TR_MATH_DOMAIN_ERROR;
    }
    if (d->kind == TR_DELTA_DIFF && (!s->source || !bytes_equal(d->old_key, d->old_len, s->source, s->source_len)))
        return TR_MATH_DOMAIN_ERROR;                  /* the diff starts from another version */
    return TR_MATH_OK;
}
static uint64_t cells_bytes(uint64_t touched, int32_t nterms) { return tr_bank_state_bytes(touched, nterms); }
int32_t tr_delta_state_need(const tr_delta_state *s, const tr_delta *d, int32_t target, tr_math_need *need) {
    if (!need || (target != 0 && target != 1)) return TR_MATH_INVALID;
    int32_t rc = compatible(s, d);
    if (rc == TR_MATH_DOMAIN_ERROR) rc = TR_MATH_OK;  /* the needs do not depend on it */
    if (rc) return rc;
    uint64_t source = d->kind == TR_DELTA_DIFF ? d->new_len : 0;
    uint64_t bytes = tr_budget_add(state_bytes(s->ngroups, s->nterms, s->ncols, s->schema_len, source),
                                   cells_bytes(d->touched, s->nterms));
    need_zero(need);
    if (target == 1) {
        tr_scope_view dv;
        tr_delta_get_view(d, &dv);
        tr_delta_device_need(&dv, s->nterms, s->joint, s->degree, d->touched, need);
        bytes = tr_budget_add(bytes, cells_bytes(d->touched, s->nterms));     /* the host's gathered and read-back cells */
        if (need->need[TR_BUDGET_BYTES] < bytes) need->need[TR_BUDGET_BYTES] = bytes;
    } else {
        need->need[TR_BUDGET_VARIABLES] = (uint64_t)tr_bank_popc32(s->joint);
        need->need[TR_BUDGET_DEGREE] = (uint64_t)s->degree;
        need->need[TR_BUDGET_SUPPORT] = (uint64_t)s->nterms;
        need->need[TR_BUDGET_BYTES] = bytes;
    }
    need->need[TR_BUDGET_STATES] = tr_budget_mul(s->ngroups, (uint64_t)s->nterms);
    return TR_MATH_OK;
}
/* the old cells of the touched groups (compact; arrays owned by the caller) */
static void gather(const tr_delta_state *s, const tr_delta *d, tr_bank_states *out) {
    const uint64_t m = (uint64_t)s->nterms;
    for (uint64_t u = 0; u < d->touched; u++) {
        uint64_t g = d->tgroups[u];
        out->rows[u] = s->rows[g];
        if (m) {
            memcpy(out->count + u * m, s->count + g * m, (size_t)m * 8);
            memcpy(out->sum + u * m, s->sum + g * m, (size_t)m * 16);
        }
    }
}
static void free_cells(tr_bank_states *c) { free(c->rows); free(c->count); free(c->sum); }
/* checks, budget and proof of an apply, and the gathered old cells: alloc = 1 allocates them (freed here on any
 * failure, else the caller's), alloc = 0 fills the caller's arrays */
static int32_t prepare(const tr_delta_state *s, const tr_delta *d, const tr_math_budget *budget, tr_math_refusal *refusal,
                       int32_t target, tr_bank_states *cells, int alloc) {
    tr_budget_none(refusal);
    if (alloc) memset(cells, 0, sizeof *cells);
    int32_t rc = compatible(s, d);
    if (rc) return rc;
    tr_math_need need;
    tr_delta_state_need(s, d, target, &need);
    tr_math_budget dflt;
    rc = tr_budget_check(budget_or(budget, &dflt), &need, refusal);
    if (rc) return rc;
    const uint64_t m = (uint64_t)s->nterms, t = d->touched;
    if (alloc) {
        cells->ngroups = t; cells->nterms = s->nterms;
        cells->rows = zalloc(t * 8); cells->count = zalloc(t * m * 8); cells->sum = zalloc(t * m * 16);
        if (!cells->rows || !cells->count || !cells->sum) { free_cells(cells); return TR_MATH_MEMORY; }
    } else if (!cells || cells->ngroups != t || cells->nterms != s->nterms || (t && !cells->rows) ||
               (m && t && (!cells->count || !cells->sum)))
        return TR_MATH_INVALID;
    gather(s, d, cells);
    tr_scope_view dv;
    tr_delta_get_view(d, &dv);
    rc = tr_delta_prove(&dv, s->terms, s->nterms, s->mode, s->joint, cells, d->wabs);
    if (rc && alloc) free_cells(cells);
    return rc;
}
static void violation_none(tr_delta_violation *v) {
    if (!v) return;
    memset(v, 0, sizeof *v);
    v->group = TR_DELTA_NONE; v->term = -1;
}
/* the violation record of key (compact cell u, slot), from the old state and the new cells */
static void violation_fill(const tr_delta_state *s, const tr_delta *d, const tr_bank_states *cells, uint64_t key,
                           tr_delta_violation *v) {
    if (!v) return;
    const uint64_t m = (uint64_t)s->nterms, u = key / (m + 1), slot = key % (m + 1), g = d->tgroups[u];
    __int128 have, now;
    int32_t reason = TR_DELTA_NEGATIVE_ROWS;
    if (!slot) { have = (int64_t)s->rows[g]; now = (int64_t)cells->rows[u]; }
    else {
        uint64_t j = slot - 1;
        reason = tr_delta_reason(&s->terms[j], s->mode, (int64_t)cells->rows[u], (int64_t)cells->count[u * m + j],
                                 tr_i128_get(cells->sum[u * m + j]), (int)slot);
        int on_count = reason == TR_DELTA_NEGATIVE_COUNT || reason == TR_DELTA_COUNT_ABOVE_ROWS;
        have = on_count ? (__int128)(int64_t)s->count[g * m + j] : tr_i128_get(s->sum[g * m + j]);
        now = on_count ? (__int128)(int64_t)cells->count[u * m + j] : tr_i128_get(cells->sum[u * m + j]);
    }
    v->group = g; v->term = (int32_t)slot - 1; v->reason = reason;
    v->have = tr_i128_set(have); v->result = tr_i128_set(now);
    v->change = tr_i128_set((__int128)((unsigned __int128)now - (unsigned __int128)have));   /* |change| < 2^127: proven */
}
/* the new state: the old cells with the touched groups replaced, and its identity */
static tr_delta_state *successor(const tr_delta_state *s, const tr_delta *d, const tr_bank_states *cells, int32_t *rc) {
    tr_delta_state *n = state_alloc(s, s->ngroups, s->ncols, s->nterms);
    if (!n) { *rc = TR_MATH_MEMORY; return NULL; }
    const uint64_t m = (uint64_t)s->nterms, gm = s->ngroups * m;
    memcpy(n->rows, s->rows, (size_t)s->ngroups * 8);
    if (gm) { memcpy(n->count, s->count, (size_t)gm * 8); memcpy(n->sum, s->sum, (size_t)gm * 16); }
    for (uint64_t u = 0; u < d->touched; u++) {
        uint64_t g = d->tgroups[u];
        n->rows[g] = cells->rows[u];
        if (m) {
            memcpy(n->count + g * m, cells->count + u * m, (size_t)m * 8);
            memcpy(n->sum + g * m, cells->sum + u * m, (size_t)m * 16);
        }
    }
    n->step = s->step + 1;
    n->base[0] = s->base[0]; n->base[1] = s->base[1];
    hasher h;
    h_init(&h, UINT64_C(0x50455453544c4544));
    h_word(&h, s->identity[0]); h_word(&h, s->identity[1]); h_word(&h, d->identity[0]); h_word(&h, d->identity[1]);
    h_word(&h, n->step);
    h_out(&h, n->identity);
    n->table = s->table;
    n->revision = d->kind == TR_DELTA_ARRAYS ? s->revision : d->revision;
    if (d->kind == TR_DELTA_DIFF) {                    /* the state of exactly the new scope */
        n->source = zalloc(d->new_len);
        if (!n->source) { state_release(n); *rc = TR_MATH_MEMORY; return NULL; }
        memcpy(n->source, d->new_key, (size_t)d->new_len);
        n->source_len = d->new_len; n->source_digest[0] = d->new_digest[0]; n->source_digest[1] = d->new_digest[1];
    }
    state_tally(n);
    n->magic = STATE_MAGIC;
    *rc = TR_MATH_OK;
    return n;
}
tr_delta_state *tr_delta_state_apply(const tr_delta_state *s, const tr_delta *d, const tr_math_budget *budget,
                                     tr_math_refusal *refusal, tr_delta_violation *violation, int32_t *error) {
    violation_none(violation);
    tr_bank_states cells;
    int32_t rc = prepare(s, d, budget, refusal, 0, &cells, 1);
    if (rc) { set_error(error, rc); return NULL; }
    tr_scope_view dv;
    tr_delta_get_view(d, &dv);
    const uint64_t m = (uint64_t)s->nterms;
    for (uint64_t r = 0; r < d->k; r++) {             /* every addition proven to fit by tr_delta_prove */
        int64_t w = d->weights[r];
        if (!w) continue;
        uint64_t u = d->local[r];
        cells.rows[u] += (uint64_t)w;
        if (s->mode == TR_BANK_JOINT && !tr_bank_joint_valid(&dv, s->joint, r)) continue;
        for (uint64_t j = 0; j < m; j++) {
            const tr_bank_term *t = &s->terms[j];
            if (s->mode == TR_BANK_SQL && !tr_bank_term_valid(&dv, t, r)) continue;
            cells.count[u * m + j] += (uint64_t)w;
            cells.sum[u * m + j] = tr_i128_set(tr_i128_get(cells.sum[u * m + j]) + (__int128)w * tr_bank_term_value(&dv, t, r));
        }
    }
    if (s->promise == TR_DELTA_NONNEGATIVE) {
        uint64_t key = tr_delta_first_violation(&cells, s->terms, s->mode);
        if (key != TR_DELTA_NONE) {
            violation_fill(s, d, &cells, key, violation);
            free_cells(&cells);
            set_error(error, TR_MATH_DOMAIN_ERROR);
            return NULL;
        }
    }
    tr_delta_state *n = successor(s, d, &cells, &rc);
    free_cells(&cells);
    set_error(error, rc);
    return n;
}

int32_t tr_delta_state_gather(const tr_delta_state *s, const tr_delta *d, tr_bank_states *out,
                              const tr_math_budget *budget, tr_math_refusal *refusal) {
    return prepare(s, d, budget, refusal, 1, out, 0);
}

/* 1 when two int64 words differ by more than w (exact, without overflow) */
static int moved_more(uint64_t a, uint64_t b, uint64_t w) {
    int64_t diff;
    if (__builtin_sub_overflow((int64_t)a, (int64_t)b, &diff)) return 1;
    return tr_delta_mag64(diff) > w;
}

tr_delta_state *tr_delta_state_from_cells(const tr_delta_state *s, const tr_delta *d, const tr_bank_states *cells,
                                          uint64_t key, const tr_math_budget *budget, tr_math_refusal *refusal,
                                          tr_delta_violation *violation, int32_t *error) {
    violation_none(violation);
    tr_budget_none(refusal);
    int32_t rc = compatible(s, d);
    if (rc) { set_error(error, rc); return NULL; }
    const uint64_t m = (uint64_t)s->nterms, t = d->touched;
    if (!cells || cells->ngroups != t || cells->nterms != s->nterms || (t && !cells->rows) || (m && t && (!cells->count || !cells->sum))) {
        set_error(error, TR_MATH_INVALID); return NULL;
    }
    tr_math_need need;
    tr_delta_state_need(s, d, 0, &need);
    tr_math_budget dflt;
    rc = tr_budget_check(budget_or(budget, &dflt), &need, refusal);
    if (rc) { set_error(error, rc); return NULL; }
    /* the cells must stay within the proven bounds of the old cells (a consistency check of the device result) */
    tr_scope_view dv;
    tr_delta_get_view(d, &dv);
    for (uint64_t u = 0; u < t; u++) {
        uint64_t g = d->tgroups[u], w = d->wabs[u];
        if (moved_more(cells->rows[u], s->rows[g], w)) { set_error(error, TR_MATH_INVALID); return NULL; }
        for (uint64_t j = 0; j < m; j++) {
            unsigned __int128 lim = tr_bank_satmul(tr_bank_term_bound(&dv, &s->terms[j]), w, ((unsigned __int128)1 << 127) - 1);
            __int128 ds;
            int far = __builtin_sub_overflow(tr_i128_get(cells->sum[u * m + j]), tr_i128_get(s->sum[g * m + j]), &ds);
            unsigned __int128 mag = ds < 0 ? (unsigned __int128)0 - (unsigned __int128)ds : (unsigned __int128)ds;
            if (moved_more(cells->count[u * m + j], s->count[g * m + j], w) || far || mag > lim) {
                set_error(error, TR_MATH_INVALID); return NULL;
            }
        }
    }
    uint64_t first = s->promise == TR_DELTA_NONNEGATIVE ? tr_delta_first_violation(cells, s->terms, s->mode) : TR_DELTA_NONE;
    if (first != key) { set_error(error, TR_MATH_INVALID); return NULL; }   /* the device and the host disagree */
    if (first != TR_DELTA_NONE) {
        violation_fill(s, d, cells, first, violation);
        set_error(error, TR_MATH_DOMAIN_ERROR);
        return NULL;
    }
    tr_delta_state *n = successor(s, d, cells, &rc);
    set_error(error, rc);
    return n;
}

tr_bank *tr_delta_state_to_bank(const tr_delta_state *s, const tr_scope *scope, const tr_math_budget *budget,
                                tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    if (!state_valid(s)) { set_error(error, TR_MATH_INVALID); return NULL; }
    if (scope && tr_delta_state_bound_to(s, scope) != 1) { set_error(error, TR_MATH_DOMAIN_ERROR); return NULL; }
    tr_bank_states st;
    memset(&st, 0, sizeof st);
    st.ngroups = s->ngroups; st.nterms = s->nterms; st.rows = s->rows; st.count = s->count; st.sum = s->sum;
    if (tr_delta_first_violation(&st, s->terms, s->mode) != TR_DELTA_NONE) { set_error(error, TR_MATH_DOMAIN_ERROR); return NULL; }
    return tr_bank_from_states(scope, s->ncols, s->exps, s->nterms, s->mode, &st, budget, refusal, error);
}

int32_t tr_delta_state_bound_to(const tr_delta_state *s, const tr_scope *scope) {
    if (!state_valid(s) || !scope) return TR_MATH_INVALID;
    if (!s->source) return 0;
    int32_t same = tr_scope_key_equal(scope, TR_SCOPE_KEY_ALL, s->source, s->source_len);
    return same < 0 ? same : same == 1;
}

int32_t tr_delta_state_get_info(const tr_delta_state *s, tr_delta_info *out) {
    if (!state_valid(s) || !out) return TR_MATH_INVALID;
    memset(out, 0, sizeof *out);
    out->ngroups = s->ngroups; out->present = s->present; out->step = s->step; out->bytes = s->bytes;
    for (int i = 0; i < 2; i++) { out->identity[i] = s->identity[i]; out->base[i] = s->base[i]; out->source[i] = s->source ? s->source_digest[i] : 0; }
    out->revision = s->revision; out->table = s->table; out->rows = s->total;
    out->ncols = s->ncols; out->nterms = s->nterms; out->mode = s->mode; out->promise = s->promise;
    out->bound = s->source != NULL; out->degree = s->degree;
    return TR_MATH_OK;
}
int32_t tr_delta_state_get_states(const tr_delta_state *s, tr_bank_states *out) {
    if (!state_valid(s) || !out) return TR_MATH_INVALID;
    memset(out, 0, sizeof *out);
    out->ngroups = s->ngroups; out->nterms = s->nterms; out->rows = s->rows; out->count = s->count; out->sum = s->sum;
    return TR_MATH_OK;
}
int32_t tr_delta_state_get_exponents(const tr_delta_state *s, uint8_t *out, uint64_t capacity) {
    if (!state_valid(s)) return TR_MATH_INVALID;
    uint64_t need = (uint64_t)s->nterms * (uint64_t)s->ncols;
    if (need && (!out || capacity < need)) return TR_MATH_INVALID;
    if (need) memcpy(out, s->exps, (size_t)need);
    return TR_MATH_OK;
}
int32_t tr_delta_state_equal_bank(const tr_delta_state *s, const tr_bank *b) {
    tr_bank_info bi;
    tr_bank_states st;
    if (!state_valid(s) || !b || tr_bank_get_info(b, &bi) || tr_bank_get_states(b, &st)) return TR_MATH_INVALID;
    if (bi.ncols != s->ncols || bi.nterms != s->nterms || bi.mode != s->mode || bi.ngroups != s->ngroups) return 0;
    uint64_t ne = (uint64_t)s->nterms * (uint64_t)s->ncols, gm = s->ngroups * (uint64_t)s->nterms;
    uint8_t *e = zalloc(ne);
    if (!e) return TR_MATH_MEMORY;
    int same = !tr_bank_get_exponents(b, e, ne) && (!ne || !memcmp(e, s->exps, (size_t)ne));
    free(e);
    return same && !memcmp(st.rows, s->rows, (size_t)s->ngroups * 8) &&
           (!gm || (!memcmp(st.count, s->count, (size_t)gm * 8) && !memcmp(st.sum, s->sum, (size_t)gm * 16)));
}
int32_t tr_delta_state_equal(const tr_delta_state *a, const tr_delta_state *b) {
    if (!state_valid(a) || !state_valid(b)) return TR_MATH_INVALID;
    uint64_t ne = (uint64_t)a->nterms * (uint64_t)a->ncols, gm = a->ngroups * (uint64_t)a->nterms;
    return a->ncols == b->ncols && a->nterms == b->nterms && a->mode == b->mode && a->ngroups == b->ngroups &&
           (!ne || !memcmp(a->exps, b->exps, (size_t)ne)) && !memcmp(a->rows, b->rows, (size_t)a->ngroups * 8) &&
           (!gm || (!memcmp(a->count, b->count, (size_t)gm * 8) && !memcmp(a->sum, b->sum, (size_t)gm * 16)));
}
