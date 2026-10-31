/* FND-004: the data scope of the mathematical capabilities; see math_scope.h.
 * SPDX-License-Identifier: Apache-2.0 */
#include "math_scope.h"
#include "mathematics.h"
#include "math_domain.h"
#include "torble.h"
#include <stdlib.h>
#include <string.h>

extern void at_decode_column(const atable_t *, int, uint32_t *);

_Static_assert(sizeof(tr_scope_column) == 32, "tr_scope_column ABI");
_Static_assert(sizeof(tr_scope_key) == 48, "tr_scope_key ABI");
_Static_assert(sizeof(tr_scope_array) == 24, "tr_scope_array ABI");
_Static_assert(sizeof(tr_scope_view_column) == 64, "tr_scope_view_column ABI");
_Static_assert(sizeof(tr_scope_view) == 2096, "tr_scope_view ABI");
_Static_assert(sizeof(tr_scope_info) == 112, "tr_scope_info ABI");
_Static_assert(sizeof(tr_scope_column_info) == 56, "tr_scope_column_info ABI");

#define SCOPE_MAGIC UINT32_C(0x53434f50)
#define ROWS_TAG UINT64_C(0x31534f5257504353)     /* "SCPWROS1" */
#define GROUPS_TAG UINT64_C(0x3150524757504353)   /* "SCPWGRP1" */

typedef struct {
    tr_scope_column spec;     /* array scopes: col = valid = hi = -1, shift = offset = 0 */
    uint32_t *codes;
    int64_t *values;
    uint8_t *valid;           /* NULL: no NULL among the selected rows */
    int64_t min, max;
    uint64_t nvalid;
} scope_col;

typedef struct {
    tr_scope_key spec;        /* lookup / lookup_valid are not kept here: see the owned copies */
    uint32_t *lookup;
    uint8_t *lookup_valid;    /* normalized 0 / 1 */
} scope_key;

typedef struct { int32_t col; uint64_t addr, revision; } scope_side;

struct tr_scope_s {
    uint32_t magic;
    int32_t source, ncols, nkeys, nside;
    uint64_t n, nw, selected, ngroups;
    uint64_t *selection;
    uint32_t *groups;         /* NULL when ngroups == 1 */
    scope_col col[TR_SCOPE_MAX_COLUMNS];
    scope_key key[TR_SCOPE_MAX_KEYS];
    scope_side side[MAXCOL];
    uint64_t table, revision;
    uint8_t identity[16];
    uint64_t digest[2], version[2];
    uint64_t bytes;
};

static void set_error(int32_t *error, int32_t code) { if (error) *error = code; }
static int scope_valid(const tr_scope *s) { return s && s->magic == SCOPE_MAGIC; }
static void *zalloc(uint64_t bytes) {
    if (bytes > (uint64_t)SIZE_MAX) return NULL;
    return calloc(1, bytes ? (size_t)bytes : 1);
}
static uint64_t tail_mask(uint64_t n, uint64_t w) {
    uint64_t left = n - w * 64;
    return left >= 64 ? UINT64_MAX : (left ? (UINT64_C(1) << left) - 1 : 0);
}
static uint64_t copy_selection(uint64_t *dst, const uint64_t *rows, uint64_t n) {
    uint64_t nw = (n + 63) / 64, count = 0;
    for (uint64_t w = 0; w < nw; w++) {
        dst[w] = (rows ? rows[w] : UINT64_MAX) & tail_mask(n, w);
        count += (uint64_t)__builtin_popcountll(dst[w]);
    }
    return count;
}
static int selected(const tr_scope *s, uint64_t r) { return (int)((s->selection[r >> 6] >> (r & 63)) & 1); }

static void release(tr_scope *s) {
    if (!s) return;
    free(s->selection); free(s->groups);
    for (int i = 0; i < TR_SCOPE_MAX_COLUMNS; i++) { free(s->col[i].codes); free(s->col[i].values); free(s->col[i].valid); }
    for (int i = 0; i < TR_SCOPE_MAX_KEYS; i++) { free(s->key[i].lookup); free(s->key[i].lookup_valid); }
    s->magic = 0;
    free(s);
}
void tr_scope_free(tr_scope *s) { if (scope_valid(s)) release(s); }

static tr_scope *fail(tr_scope *s, int32_t *error, int32_t code) { release(s); set_error(error, code); return NULL; }

/* ------------------------------------------------------------------ the exact descriptor
 * A descriptor is a sequence of segments: header words (collected here) and arrays the scope owns (referenced). The
 * same sequence gives the size, the bytes, the comparison with stored bytes or with another scope, and the digest. */
#define DESC_SEGS 256
#define DESC_WORDS 768
typedef struct { const unsigned char *p; uint64_t len; int header; } desc_seg;
typedef struct { int n; int nwords; desc_seg seg[DESC_SEGS]; uint64_t word[DESC_WORDS]; } desc_t;

static void d_word(desc_t *d, uint64_t v) {
    d->word[d->nwords] = v;
    if (d->n && d->seg[d->n - 1].header) d->seg[d->n - 1].len += 8;
    else { d->seg[d->n].p = (const unsigned char *)&d->word[d->nwords]; d->seg[d->n].len = 8; d->seg[d->n].header = 1; d->n++; }
    d->nwords++;
}
static void d_bytes(desc_t *d, const void *p, uint64_t len) {
    if (!len) return;
    d->seg[d->n].p = (const unsigned char *)p; d->seg[d->n].len = len; d->seg[d->n].header = 0; d->n++;
}
static void describe(const tr_scope *s, int32_t part, desc_t *d) {
    d->n = 0; d->nwords = 0;
    if (part & TR_SCOPE_KEY_ROWS) {
        d_word(d, ROWS_TAG);
        d_word(d, (uint64_t)s->source);
        d_word(d, s->n);
        if (s->source == TR_SCOPE_TABLE) {
            uint64_t id[2];
            memcpy(id, s->identity, 16);
            d_word(d, s->table); d_word(d, id[0]); d_word(d, id[1]); d_word(d, s->revision);
            d_word(d, (uint64_t)s->nside);
            for (int i = 0; i < s->nside; i++) {
                d_word(d, (uint64_t)s->side[i].col); d_word(d, s->side[i].addr); d_word(d, s->side[i].revision);
            }
        }
        d_word(d, s->nw);
        d_bytes(d, s->selection, s->nw * 8);
        d_word(d, (uint64_t)s->ncols);
        for (int i = 0; i < s->ncols; i++) {
            const scope_col *k = &s->col[i];
            if (s->source == TR_SCOPE_TABLE) {
                d_word(d, (uint64_t)(int64_t)k->spec.col); d_word(d, (uint64_t)(int64_t)k->spec.valid);
                d_word(d, (uint64_t)(int64_t)k->spec.hi); d_word(d, (uint64_t)(int64_t)(k->spec.hi >= 0 ? k->spec.shift : 0));
                d_word(d, (uint64_t)k->spec.offset); d_word(d, (uint64_t)(int64_t)k->spec.scale);
            } else {
                d_word(d, (uint64_t)(int64_t)k->spec.scale); d_word(d, k->valid != NULL);
                d_bytes(d, k->values, s->n * 8);
                if (k->valid) d_bytes(d, k->valid, s->n);
            }
        }
    }
    if (part & TR_SCOPE_KEY_GROUPS) {
        d_word(d, GROUPS_TAG);
        d_word(d, s->ngroups);
        d_word(d, (uint64_t)s->nkeys);
        if (s->source == TR_SCOPE_TABLE) {
            for (int i = 0; i < s->nkeys; i++) {
                const scope_key *q = &s->key[i];
                d_word(d, (uint64_t)(int64_t)q->spec.col); d_word(d, (uint64_t)(int64_t)q->spec.valid);
                d_word(d, (uint64_t)q->spec.nullable); d_word(d, q->spec.nullable ? q->spec.null_code : 0);
                d_word(d, q->spec.radix); d_word(d, q->lookup != NULL); d_word(d, q->lookup_valid != NULL);
                d_word(d, q->spec.lookup_n);
                if (q->lookup) d_bytes(d, q->lookup, q->spec.lookup_n * 4);
                if (q->lookup_valid) d_bytes(d, q->lookup_valid, q->spec.lookup_n);
            }
        } else {
            d_word(d, s->groups != NULL);
            if (s->groups) d_bytes(d, s->groups, s->n * 4);
        }
    }
}
static uint64_t d_size(const desc_t *d) { uint64_t n = 0; for (int i = 0; i < d->n; i++) n += d->seg[i].len; return n; }
static int part_ok(int32_t part) { return part == TR_SCOPE_KEY_ROWS || part == TR_SCOPE_KEY_GROUPS || part == TR_SCOPE_KEY_ALL; }

/* 128-bit digest over the byte stream (two multiply-xorshift lanes; a pre-filter only) */
typedef struct { uint64_t h0, h1, word, len; int nbytes; } hasher;
static void h_mix(hasher *h, uint64_t w) {
    h->h0 = (h->h0 ^ w) * UINT64_C(0x9E3779B97F4A7C15); h->h0 ^= h->h0 >> 29;
    h->h1 = (h->h1 + w) * UINT64_C(0xBF58476D1CE4E5B9); h->h1 ^= h->h1 >> 31; h->h1 += UINT64_C(0x94D049BB133111EB);
}
static void h_bytes(hasher *h, const unsigned char *p, uint64_t len) {
    h->len += len;
    while (len && h->nbytes) {
        h->word |= (uint64_t)*p++ << (8 * h->nbytes); len--;
        if (++h->nbytes == 8) { h_mix(h, h->word); h->word = 0; h->nbytes = 0; }
    }
    for (; len >= 8; len -= 8, p += 8) { uint64_t w; memcpy(&w, p, 8); h_mix(h, w); }
    while (len) { h->word |= (uint64_t)*p++ << (8 * h->nbytes); h->nbytes++; len--; }
}
static void h_final(hasher *h, uint64_t out[2]) {
    if (h->nbytes) h_mix(h, h->word);
    h_mix(h, h->len);
    out[0] = h->h0 ^ (h->h1 >> 17); out[1] = h->h1 ^ (h->h0 << 13);
}
static void h_init(hasher *h) { h->h0 = UINT64_C(0x243F6A8885A308D3); h->h1 = UINT64_C(0x13198A2E03707344); h->word = h->len = 0; h->nbytes = 0; }

/* the digest of the whole descriptor, and the version digest of a table scope (revision, rows, sidecars) */
static int digest(tr_scope *s) {
    desc_t *d = malloc(sizeof *d);
    if (!d) return TR_MATH_MEMORY;
    describe(s, TR_SCOPE_KEY_ALL, d);
    hasher h; h_init(&h);
    for (int i = 0; i < d->n; i++) h_bytes(&h, d->seg[i].p, d->seg[i].len);
    h_final(&h, s->digest);
    free(d);
    hasher v; h_init(&v);
    uint64_t head[3] = {s->revision, s->n, (uint64_t)s->nside};
    h_bytes(&v, (const unsigned char *)head, sizeof head);
    for (int i = 0; i < s->nside; i++) {
        uint64_t e[3] = {(uint64_t)s->side[i].col, s->side[i].addr, s->side[i].revision};
        h_bytes(&v, (const unsigned char *)e, sizeof e);
    }
    h_final(&v, s->version);
    return TR_MATH_OK;
}

uint64_t tr_scope_key_size(const tr_scope *s, int32_t part) {
    if (!scope_valid(s) || !part_ok(part)) return 0;
    desc_t *d = malloc(sizeof *d);
    if (!d) return 0;
    describe(s, part, d);
    uint64_t n = d_size(d);
    free(d);
    return n;
}
int32_t tr_scope_key_write(const tr_scope *s, int32_t part, unsigned char *out, uint64_t capacity) {
    if (!scope_valid(s) || !part_ok(part) || !out) return TR_MATH_INVALID;
    desc_t *d = malloc(sizeof *d);
    if (!d) return TR_MATH_MEMORY;
    describe(s, part, d);
    if (d_size(d) > capacity) { free(d); return TR_MATH_INVALID; }
    uint64_t at = 0;
    for (int i = 0; i < d->n; i++) { memcpy(out + at, d->seg[i].p, (size_t)d->seg[i].len); at += d->seg[i].len; }
    free(d);
    return TR_MATH_OK;
}
int32_t tr_scope_key_equal(const tr_scope *s, int32_t part, const unsigned char *key, uint64_t bytes) {
    if (!scope_valid(s) || !part_ok(part) || (bytes && !key)) return TR_MATH_INVALID;
    desc_t *d = malloc(sizeof *d);
    if (!d) return TR_MATH_MEMORY;
    describe(s, part, d);
    int same = d_size(d) == bytes;
    uint64_t at = 0;
    for (int i = 0; same && i < d->n; i++) {
        same = memcmp(key + at, d->seg[i].p, (size_t)d->seg[i].len) == 0;
        at += d->seg[i].len;
    }
    free(d);
    return same;
}
static int32_t equal_parts(const tr_scope *a, const tr_scope *b, int32_t part) {
    if (!scope_valid(a) || !scope_valid(b)) return TR_MATH_INVALID;
    if (a == b) return 1;
    desc_t *x = malloc(sizeof *x), *y = malloc(sizeof *y);
    if (!x || !y) { free(x); free(y); return TR_MATH_MEMORY; }
    describe(a, part, x); describe(b, part, y);
    int same = d_size(x) == d_size(y);
    int i = 0, j = 0;
    uint64_t oi = 0, oj = 0;                              /* offsets inside the current segments */
    while (same && i < x->n && j < y->n) {
        uint64_t take = x->seg[i].len - oi < y->seg[j].len - oj ? x->seg[i].len - oi : y->seg[j].len - oj;
        same = memcmp(x->seg[i].p + oi, y->seg[j].p + oj, (size_t)take) == 0;
        oi += take; oj += take;
        if (oi == x->seg[i].len) { i++; oi = 0; }
        if (oj == y->seg[j].len) { j++; oj = 0; }
    }
    free(x); free(y);
    return same;
}
int32_t tr_scope_same(const tr_scope *a, const tr_scope *b) {
    if (scope_valid(a) && scope_valid(b) && (a->digest[0] != b->digest[0] || a->digest[1] != b->digest[1])) return 0;
    return equal_parts(a, b, TR_SCOPE_KEY_ALL);
}
int32_t tr_scope_same_rows(const tr_scope *a, const tr_scope *b) { return equal_parts(a, b, TR_SCOPE_KEY_ROWS); }

/* ------------------------------------------------------------------ engine tables */
static int col_ok(const atable_t *t, int32_t c) { return c >= 0 && c < t->k; }
static int32_t table_check(const atable_t *t, const tr_scope_column *cols, int32_t ncols, const tr_scope_key *keys,
                           int32_t nkeys, uint64_t *ngroups) {
    if (!t || t->k < 0 || t->k > MAXCOL || ncols < 0 || ncols > TR_SCOPE_MAX_COLUMNS || nkeys < 0 ||
        nkeys > TR_SCOPE_MAX_KEYS || (ncols && !cols) || (nkeys && !keys)) return TR_MATH_INVALID;
    for (int i = 0; i < ncols; i++) {
        const tr_scope_column *c = &cols[i];
        if (!col_ok(t, c->col) || (c->valid != -1 && !col_ok(t, c->valid)) || (c->hi != -1 && !col_ok(t, c->hi)) ||
            (c->hi >= 0 && (c->shift < 0 || c->shift > 32)) || c->scale < 0 || c->scale > TR_SCOPE_MAX_SCALE)
            return TR_MATH_INVALID;
    }
    uint64_t g = 1;
    for (int i = 0; i < nkeys; i++) {
        const tr_scope_key *q = &keys[i];
        if (!col_ok(t, q->col) || (q->valid != -1 && !col_ok(t, q->valid)) || (q->nullable != 0 && q->nullable != 1) ||
            q->radix < 1 || (q->nullable && q->null_code >= q->radix) || (q->lookup_valid && !q->lookup) ||
            (!q->lookup && q->lookup_n) || (q->lookup && q->lookup_n > (uint64_t)UINT32_MAX + 1))
            return TR_MATH_INVALID;
        g = tr_budget_mul(g, q->radix);
    }
    *ngroups = g;
    return TR_MATH_OK;
}
/* every allocation of tr_scope_from_table, in the order it happens */
static void table_need(const atable_t *t, const tr_scope_column *cols, int32_t ncols, const tr_scope_key *keys,
                       int32_t nkeys, uint64_t ngroups, tr_math_need *need) {
    uint64_t n = t->n, col4 = tr_budget_mul(n, 4), bytes = tr_budget_add(sizeof(tr_scope), tr_budget_bitmap_bytes(n));
    int scratch_a = nkeys > 0, scratch_b = 0;
    for (int i = 0; i < ncols; i++) {
        bytes = tr_budget_add(bytes, cols[i].hi >= 0 ? tr_budget_mul(n, 8) : col4);
        if (cols[i].valid >= 0) { bytes = tr_budget_add(bytes, n); scratch_a = 1; }
        if (cols[i].hi >= 0) scratch_a = scratch_b = 1;
    }
    for (int i = 0; i < nkeys; i++) {
        if (keys[i].valid >= 0) scratch_b = 1;
        if (keys[i].lookup) bytes = tr_budget_add(bytes, tr_budget_mul(keys[i].lookup_n, keys[i].lookup_valid ? 5 : 4));
    }
    if (nkeys && ngroups > 1) bytes = tr_budget_add(bytes, col4);
    if (scratch_a) bytes = tr_budget_add(bytes, col4);
    if (scratch_b) bytes = tr_budget_add(bytes, col4);
    for (int i = 0; i < TR_BUDGET_RESOURCES; i++) need->need[i] = 0;
    need->need[TR_BUDGET_STATES] = ngroups;
    need->need[TR_BUDGET_BYTES] = bytes;
}
int32_t tr_scope_need_table(const struct atable_s *table, const tr_scope_column *cols, int32_t ncols,
                            const tr_scope_key *keys, int32_t nkeys, tr_math_need *need) {
    uint64_t g = 0;
    if (!need) return TR_MATH_INVALID;
    int32_t rc = table_check(table, cols, ncols, keys, nkeys, &g);
    if (rc) return rc;
    table_need(table, cols, ncols, keys, nkeys, g, need);
    return TR_MATH_OK;
}

static int32_t extremes_store(scope_col *k, __int128 v, uint64_t *count) {
    if (v < INT64_MIN || v > INT64_MAX) return TR_MATH_OVERFLOW;
    int64_t x = (int64_t)v;
    if (x < k->min) k->min = x;
    if (x > k->max) k->max = x;
    (*count)++;
    return TR_MATH_OK;
}

/* the self-extending (virtual) columns a scope reads: their sidecars' revisions are part of its version */
static void mark_side(const atable_t *t, int32_t c, uint32_t *seen) {
    if (c >= 0 && t->vflag[c]) *seen |= UINT32_C(1) << c;
}

static int32_t finish(tr_scope *s) {
    for (int i = 0; i < s->ncols; i++) {
        scope_col *k = &s->col[i];
        if (k->valid && k->nvalid == s->selected) { free(k->valid); k->valid = NULL; }   /* no NULL is selected */
    }
    uint64_t bytes = tr_budget_add(sizeof(tr_scope), s->nw * 8);
    if (s->groups) bytes = tr_budget_add(bytes, s->n * 4);
    for (int i = 0; i < s->ncols; i++) {
        const scope_col *k = &s->col[i];
        bytes = tr_budget_add(bytes, k->values ? s->n * 8 : (k->codes ? s->n * 4 : 0));
        if (k->valid) bytes = tr_budget_add(bytes, s->n);
    }
    for (int i = 0; i < s->nkeys; i++) {
        if (s->key[i].lookup) bytes = tr_budget_add(bytes, s->key[i].spec.lookup_n * 4);
        if (s->key[i].lookup_valid) bytes = tr_budget_add(bytes, s->key[i].spec.lookup_n);
    }
    s->bytes = bytes;
    int32_t rc = digest(s);
    if (rc) return rc;
    s->magic = SCOPE_MAGIC;
    return TR_MATH_OK;
}

tr_scope *tr_scope_from_table(const struct atable_s *table, const uint8_t *identity, const uint64_t *rows,
                              const tr_scope_column *cols, int32_t ncols, const tr_scope_key *keys, int32_t nkeys,
                              const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    const atable_t *t = table;
    uint64_t ngroups = 0;
    int32_t rc = table_check(t, cols, ncols, keys, nkeys, &ngroups);
    if (rc) return fail(NULL, error, rc);
    tr_math_need need;
    table_need(t, cols, ncols, keys, nkeys, ngroups, &need);
    tr_math_budget dflt;
    if (!budget) { tr_budget_default(&dflt); budget = &dflt; }
    rc = tr_budget_check(budget, &need, refusal);
    if (rc) return fail(NULL, error, rc);
    if (ngroups > TR_SCOPE_MAX_GROUPS) return fail(NULL, error, TR_MATH_UNSUPPORTED);
    tr_scope *s = zalloc(sizeof *s);
    if (!s) return fail(NULL, error, TR_MATH_MEMORY);
    s->source = TR_SCOPE_TABLE; s->ncols = ncols; s->nkeys = nkeys;
    s->n = t->n; s->nw = (t->n + 63) / 64; s->ngroups = ngroups;
    s->table = (uint64_t)(uintptr_t)t; s->revision = t->revision;
    if (identity) memcpy(s->identity, identity, 16);
    uint64_t n = s->n;
    s->selection = zalloc(s->nw * 8);
    if (!s->selection) return fail(s, error, TR_MATH_MEMORY);
    s->selected = copy_selection(s->selection, rows, n);
    uint32_t seen = 0;
    int scratch_a = nkeys > 0, scratch_b = 0;
    for (int i = 0; i < ncols; i++) {
        if (cols[i].valid >= 0) scratch_a = 1;
        if (cols[i].hi >= 0) scratch_a = scratch_b = 1;
    }
    for (int i = 0; i < nkeys; i++) if (keys[i].valid >= 0) scratch_b = 1;
    uint32_t *a = scratch_a ? zalloc(n * 4) : NULL, *b = scratch_b ? zalloc(n * 4) : NULL;
    if ((scratch_a && !a) || (scratch_b && !b)) { free(a); free(b); return fail(s, error, TR_MATH_MEMORY); }
#define FAIL(code) do { free(a); free(b); return fail(s, error, (code)); } while (0)
    for (int i = 0; i < ncols; i++) {
        scope_col *k = &s->col[i];
        k->spec = cols[i];
        k->spec.reserved = 0;
        if (k->spec.hi < 0) k->spec.shift = 0;
        k->min = INT64_MAX; k->max = INT64_MIN;
        mark_side(t, k->spec.col, &seen); mark_side(t, k->spec.valid, &seen); mark_side(t, k->spec.hi, &seen);
        if (k->spec.valid >= 0) {
            k->valid = zalloc(n);
            if (!k->valid) FAIL(TR_MATH_MEMORY);
            at_decode_column(t, k->spec.valid, a);
            for (uint64_t r = 0; r < n; r++) k->valid[r] = a[r] != 0;
        }
        if (k->spec.hi >= 0) {
            k->values = zalloc(n * 8);
            if (!k->values) FAIL(TR_MATH_MEMORY);
            at_decode_column(t, k->spec.col, a);
            at_decode_column(t, k->spec.hi, b);
            for (uint64_t r = 0; r < n; r++) {
                if (!selected(s, r) || (k->valid && !k->valid[r])) continue;
                __int128 v = (__int128)a[r] + ((__int128)b[r] << k->spec.shift) + k->spec.offset;
                if (extremes_store(k, v, &k->nvalid)) FAIL(TR_MATH_OVERFLOW);
                k->values[r] = (int64_t)v;
            }
        } else {
            k->codes = zalloc(n * 4);
            if (!k->codes) FAIL(TR_MATH_MEMORY);
            at_decode_column(t, k->spec.col, k->codes);
            for (uint64_t r = 0; r < n; r++) {
                if (!selected(s, r) || (k->valid && !k->valid[r])) continue;
                if (extremes_store(k, (__int128)k->codes[r] + k->spec.offset, &k->nvalid)) FAIL(TR_MATH_OVERFLOW);
            }
        }
    }
    if (nkeys && ngroups > 1) {
        s->groups = zalloc(n * 4);
        if (!s->groups) FAIL(TR_MATH_MEMORY);
    }
    for (int i = 0; i < nkeys; i++) {
        scope_key *q = &s->key[i];
        q->spec = keys[i];
        q->spec.lookup = NULL; q->spec.lookup_valid = NULL;
        if (!q->spec.nullable) q->spec.null_code = 0;
        mark_side(t, q->spec.col, &seen); mark_side(t, q->spec.valid, &seen);
        const uint32_t *lookup = keys[i].lookup, *lookup_valid = keys[i].lookup_valid;
        uint64_t ln = keys[i].lookup_n, radix = q->spec.radix;
        if (lookup) {
            q->lookup = zalloc(ln * 4);
            if (!q->lookup) FAIL(TR_MATH_MEMORY);
            if (ln) memcpy(q->lookup, lookup, (size_t)(ln * 4));
            if (lookup_valid) {
                q->lookup_valid = zalloc(ln);
                if (!q->lookup_valid) FAIL(TR_MATH_MEMORY);
                for (uint64_t r = 0; r < ln; r++) q->lookup_valid[r] = lookup_valid[r] != 0;
            }
        }
        at_decode_column(t, q->spec.col, a);
        if (q->spec.valid >= 0) at_decode_column(t, q->spec.valid, b);
        for (uint64_t r = 0; r < n; r++) {
            if (!selected(s, r)) continue;
            uint32_t code = a[r];
            int null = q->spec.valid >= 0 && b[r] == 0;
            uint64_t digit = code;
            if (!null && q->lookup) {
                if ((uint64_t)code >= ln || (q->lookup_valid && !q->lookup_valid[code])) null = 1;
                else digit = q->lookup[code];
            }
            if (null) {
                if (!q->spec.nullable) FAIL(TR_MATH_INVALID);
                digit = q->spec.null_code;
            } else if (digit >= radix || (q->spec.nullable && digit == q->spec.null_code)) FAIL(TR_MATH_INVALID);
            if (s->groups) s->groups[r] = (uint32_t)((uint64_t)s->groups[r] * radix + digit);
        }
    }
#undef FAIL
    free(a); free(b);
    for (int c = 0; c < t->k && c < MAXCOL; c++)
        if ((seen >> c) & 1) {
            scope_side *e = &s->side[s->nside++];
            e->col = c; e->addr = (uint64_t)(uintptr_t)t->vside[c]; e->revision = t->vside[c] ? t->vside[c]->revision : 0;
        }
    rc = finish(s);
    if (rc) return fail(s, error, rc);
    set_error(error, TR_MATH_OK);
    return s;
}

/* ------------------------------------------------------------------ plain arrays */
static int32_t arrays_check(uint64_t n, const tr_scope_array *cols, int32_t ncols, const uint32_t *groups, uint64_t ngroups) {
    if (ncols < 0 || ncols > TR_SCOPE_MAX_COLUMNS || (ncols && !cols) || ngroups < 1 || (!groups && ngroups != 1) ||
        n > (UINT64_MAX >> 4)) return TR_MATH_INVALID;
    for (int i = 0; i < ncols; i++)
        if ((n && !cols[i].values) || cols[i].scale < 0 || cols[i].scale > TR_SCOPE_MAX_SCALE) return TR_MATH_INVALID;
    return TR_MATH_OK;
}
static void arrays_need(uint64_t n, const tr_scope_array *cols, int32_t ncols, uint64_t ngroups, tr_math_need *need) {
    uint64_t bytes = tr_budget_add(sizeof(tr_scope), tr_budget_bitmap_bytes(n));
    for (int i = 0; i < ncols; i++) {
        bytes = tr_budget_add(bytes, tr_budget_mul(n, 8));
        if (cols[i].valid) bytes = tr_budget_add(bytes, n);
    }
    if (ngroups > 1) bytes = tr_budget_add(bytes, tr_budget_mul(n, 4));
    for (int i = 0; i < TR_BUDGET_RESOURCES; i++) need->need[i] = 0;
    need->need[TR_BUDGET_STATES] = ngroups;
    need->need[TR_BUDGET_BYTES] = bytes;
}
int32_t tr_scope_need_arrays(uint64_t n, const tr_scope_array *cols, int32_t ncols, const uint32_t *groups,
                             uint64_t ngroups, tr_math_need *need) {
    if (!need) return TR_MATH_INVALID;
    int32_t rc = arrays_check(n, cols, ncols, groups, ngroups);
    if (rc) return rc;
    arrays_need(n, cols, ncols, ngroups, need);
    return TR_MATH_OK;
}

tr_scope *tr_scope_from_arrays(uint64_t n, const uint64_t *rows, const tr_scope_array *cols, int32_t ncols,
                               const uint32_t *groups, uint64_t ngroups, const tr_math_budget *budget,
                               tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    int32_t rc = arrays_check(n, cols, ncols, groups, ngroups);
    if (rc) return fail(NULL, error, rc);
    tr_math_need need;
    arrays_need(n, cols, ncols, ngroups, &need);
    tr_math_budget dflt;
    if (!budget) { tr_budget_default(&dflt); budget = &dflt; }
    rc = tr_budget_check(budget, &need, refusal);
    if (rc) return fail(NULL, error, rc);
    if (ngroups > TR_SCOPE_MAX_GROUPS) return fail(NULL, error, TR_MATH_UNSUPPORTED);
    tr_scope *s = zalloc(sizeof *s);
    if (!s) return fail(NULL, error, TR_MATH_MEMORY);
    s->source = TR_SCOPE_ARRAYS; s->ncols = ncols; s->nkeys = 0;
    s->n = n; s->nw = (n + 63) / 64; s->ngroups = ngroups;
    s->selection = zalloc(s->nw * 8);
    if (!s->selection) return fail(s, error, TR_MATH_MEMORY);
    s->selected = copy_selection(s->selection, rows, n);
    for (int i = 0; i < ncols; i++) {
        scope_col *k = &s->col[i];
        k->spec.col = k->spec.valid = k->spec.hi = -1;
        k->spec.scale = cols[i].scale;
        k->min = INT64_MAX; k->max = INT64_MIN;
        k->values = zalloc(n * 8);
        if (!k->values) return fail(s, error, TR_MATH_MEMORY);
        if (cols[i].valid) {
            k->valid = zalloc(n);
            if (!k->valid) return fail(s, error, TR_MATH_MEMORY);
        }
        for (uint64_t r = 0; r < n; r++) {
            if (!selected(s, r) || (cols[i].valid && !cols[i].valid[r])) continue;   /* normalized to 0 */
            if (k->valid) k->valid[r] = 1;
            k->values[r] = cols[i].values[r];
            (void)extremes_store(k, cols[i].values[r], &k->nvalid);
        }
    }
    if (groups) {
        if (ngroups > 1) {
            s->groups = zalloc(n * 4);
            if (!s->groups) return fail(s, error, TR_MATH_MEMORY);
        }
        for (uint64_t r = 0; r < n; r++) {
            if (!selected(s, r)) continue;
            if ((uint64_t)groups[r] >= ngroups) return fail(s, error, TR_MATH_INVALID);
            if (s->groups) s->groups[r] = groups[r];
        }
    }
    rc = finish(s);
    if (rc) return fail(s, error, rc);
    set_error(error, TR_MATH_OK);
    return s;
}

/* ------------------------------------------------------------------ reading */
int32_t tr_scope_get_info(const tr_scope *s, tr_scope_info *out) {
    if (!scope_valid(s) || !out) return TR_MATH_INVALID;
    memset(out, 0, sizeof *out);
    out->rows = s->n; out->selected = s->selected; out->ngroups = s->ngroups; out->bytes = s->bytes;
    out->table = s->table; out->revision = s->revision;
    out->digest[0] = s->digest[0]; out->digest[1] = s->digest[1];
    out->version[0] = s->version[0]; out->version[1] = s->version[1];
    memcpy(out->identity, s->identity, 16);
    out->source = s->source; out->ncols = s->ncols; out->nkeys = s->nkeys;
    return TR_MATH_OK;
}
int32_t tr_scope_get_view(const tr_scope *s, tr_scope_view *v) {
    if (!scope_valid(s) || !v) return TR_MATH_INVALID;
    memset(v, 0, sizeof *v);
    v->rows = s->n; v->selected = s->selected; v->ngroups = s->ngroups;
    v->selection = s->selection; v->groups = s->groups; v->ncols = s->ncols;
    for (int i = 0; i < s->ncols; i++) {
        const scope_col *k = &s->col[i];
        tr_scope_view_column *c = &v->col[i];
        c->codes = k->codes; c->values = k->values; c->valid = k->valid;
        c->offset = k->codes ? k->spec.offset : 0;
        c->min = k->min; c->max = k->max; c->nvalid = k->nvalid; c->scale = k->spec.scale;
    }
    return TR_MATH_OK;
}
int32_t tr_scope_column_get(const tr_scope *s, int32_t i, tr_scope_column_info *out) {
    if (!scope_valid(s) || !out || i < 0 || i >= s->ncols) return TR_MATH_INVALID;
    const scope_col *k = &s->col[i];
    memset(out, 0, sizeof *out);
    out->offset = k->codes ? k->spec.offset : 0; out->min = k->min; out->max = k->max; out->nvalid = k->nvalid;
    out->scale = k->spec.scale; out->storage = k->codes ? 1 : 2;
    out->col = k->spec.col; out->valid = k->spec.valid; out->hi = k->spec.hi; out->shift = k->spec.hi >= 0 ? k->spec.shift : -1;
    return TR_MATH_OK;
}

int32_t tr_scope_fresh(const tr_scope *s, const struct atable_s *table, const uint64_t *rows) {
    if (!scope_valid(s)) return TR_MATH_INVALID;
    const atable_t *t = table;
    if (s->source == TR_SCOPE_ARRAYS) {
        if (t) return TR_MATH_INVALID;
    } else {
        if (!t || (uint64_t)(uintptr_t)t != s->table || t->revision != s->revision || t->n != s->n) return TR_MATH_STALE;
        for (int i = 0; i < s->nside; i++) {
            int32_t c = s->side[i].col;
            if (c >= t->k || !t->vflag[c] || (uint64_t)(uintptr_t)t->vside[c] != s->side[i].addr ||
                (t->vside[c] ? t->vside[c]->revision : 0) != s->side[i].revision) return TR_MATH_STALE;
        }
    }
    for (uint64_t w = 0; w < s->nw; w++)
        if (((rows ? rows[w] : UINT64_MAX) & tail_mask(s->n, w)) != s->selection[w]) return TR_MATH_STALE;
    return TR_MATH_OK;
}

int32_t tr_scope_rollup_map(const tr_scope *fine, const tr_scope *coarse, uint32_t *map, uint64_t capacity) {
    if (!scope_valid(fine) || !scope_valid(coarse) || !map || capacity < fine->ngroups) return TR_MATH_INVALID;
    int32_t same = tr_scope_same_rows(fine, coarse);
    if (same < 0) return same;
    if (!same) return TR_MATH_DOMAIN_ERROR;
    for (uint64_t g = 0; g < fine->ngroups; g++) map[g] = UINT32_MAX;
    for (uint64_t w = 0; w < fine->nw; w++) {
        uint64_t bits = fine->selection[w];
        while (bits) {
            uint64_t r = w * 64 + (uint64_t)__builtin_ctzll(bits);
            bits &= bits - 1;
            uint32_t f = fine->groups ? fine->groups[r] : 0, q = coarse->groups ? coarse->groups[r] : 0;
            if (map[f] == UINT32_MAX) map[f] = q;
            else if (map[f] != q) return TR_MATH_DOMAIN_ERROR;   /* fine group f spans two coarse groups */
        }
    }
    for (uint64_t g = 0; g < fine->ngroups; g++) if (map[g] == UINT32_MAX) map[g] = 0;
    return TR_MATH_OK;
}
