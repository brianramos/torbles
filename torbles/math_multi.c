/* MULTI-001: certified refinement maps, hierarchies of exact moment banks, localized updates and exact contrasts; see
 * math_multi.h.
 * SPDX-License-Identifier: Apache-2.0 */
#include "math_multi.h"
#include "mathematics.h"
#include "math_domain.h"
#include "math_handles.h"
#include "math_carriers.h"
#include <stdlib.h>
#include <string.h>

TR_MATH_ABI(tr_multi_level, 1);          /* the ctypes boundary of MULTI-001 (math_handles.h) */
TR_MATH_ABI(tr_multi_witness, 1);
TR_MATH_ABI(tr_multi_map_info, 1);
TR_MATH_ABI(tr_multi_hier_info, 1);
TR_MATH_ABI(tr_multi_q, 1);
TR_MATH_ABI(tr_multi_cert, 1);

_Static_assert(sizeof(tr_multi_level) == 24, "tr_multi_level ABI");
_Static_assert(sizeof(tr_multi_witness) == 32, "tr_multi_witness ABI");
_Static_assert(sizeof(tr_multi_map_info) == 64, "tr_multi_map_info ABI");
_Static_assert(sizeof(tr_multi_hier_info) == 408, "tr_multi_hier_info ABI");
_Static_assert(sizeof(tr_multi_q) == 64, "tr_multi_q ABI");
_Static_assert(sizeof(tr_multi_cert) == 48, "tr_multi_cert ABI");

#define MAP_MAGIC UINT32_C(0x5050414d)
#define HIER_MAGIC UINT32_C(0x52454948)
#define GROUP_TAG UINT64_C(0x31505247544c554d)    /* "MULTGRP1" */
#define ROWS_TAG UINT64_C(0x31574f52544c554d)     /* "MULTROW1" */
#define I128_LIMIT ((((unsigned __int128)1) << 127) - 1)

struct tr_multi_map_s {
    uint32_t magic;
    int32_t status, rows_bound;
    uint64_t nfine, ncoarse, mapped, checked, bytes;
    uint64_t digest[2];
    uint32_t *map;
    unsigned char *fine_key, *coarse_key, *rows_key;
    uint64_t fine_len, coarse_len, rows_len;
};

struct tr_multi_hier_s {
    uint32_t magic;
    int32_t levels, status, ncols, nterms, mode;
    uint64_t rows, work, changed, bytes;
    tr_bank *bank[TR_MULTI_MAX_LEVELS];
    tr_multi_map *map[TR_MULTI_MAX_LEVELS - 1];
    uint8_t *exps;                                  /* the basis (nterms x ncols) */
    uint64_t level_work[TR_MULTI_MAX_LEVELS];
    int32_t scoped[TR_MULTI_MAX_LEVELS], strategy[TR_MULTI_MAX_LEVELS];
};

static void set_error(int32_t *error, int32_t code) { if (error) *error = code; }
static void *zalloc(uint64_t bytes) {
    if (bytes > (uint64_t)SIZE_MAX) return NULL;
    return calloc(1, bytes ? (size_t)bytes : 1);
}
static int map_valid(const tr_multi_map *m) { return m && m->magic == MAP_MAGIC; }
static int hier_valid(const tr_multi_hier *h) { return h && h->magic == HIER_MAGIC; }
static const tr_math_budget *budget_or(const tr_math_budget *b, tr_math_budget *dflt) {
    if (b) return b;
    tr_budget_default(dflt);
    return dflt;
}
static void need_zero(tr_math_need *need) { for (int i = 0; i < TR_BUDGET_RESOURCES; i++) need->need[i] = 0; }

/* ------------------------------------------------------------------ digests (a pre-filter only) */
typedef struct { uint64_t h0, h1; } hasher;
static void h_init(hasher *h) { h->h0 = UINT64_C(0x243F6A8885A308D3); h->h1 = UINT64_C(0x13198A2E03707344); }
static void h_word(hasher *h, uint64_t w) {
    h->h0 = (h->h0 ^ w) * UINT64_C(0x9E3779B97F4A7C15); h->h0 ^= h->h0 >> 29;
    h->h1 = (h->h1 + w) * UINT64_C(0xBF58476D1CE4E5B9); h->h1 ^= h->h1 >> 31; h->h1 += UINT64_C(0x94D049BB133111EB);
}
static void h_bytes(hasher *h, const unsigned char *p, uint64_t len) {
    h_word(h, len);
    for (; len >= 8; len -= 8, p += 8) { uint64_t w; memcpy(&w, p, 8); h_word(h, w); }
    if (len) { uint64_t w = 0; memcpy(&w, p, (size_t)len); h_word(h, w); }
}

/* ------------------------------------------------------------------ levels: bindings and rows keys */
static int level_ok(const tr_multi_level *l) {
    tr_scope_info info;
    return l && l->scope && (!l->codec_len || l->codec) && !tr_scope_get_info(l->scope, &info);
}
/* the grouping binding of a level: tag, groups, the GROUPS descriptor part, the codec descriptor */
static uint64_t binding_size(const tr_multi_level *l) {
    uint64_t g = tr_scope_key_size(l->scope, TR_SCOPE_KEY_GROUPS);
    return tr_budget_add(tr_budget_add(32, g), l->codec_len);
}
static int32_t binding_write(const tr_multi_level *l, unsigned char **out, uint64_t *len) {
    tr_scope_info info;
    tr_scope_get_info(l->scope, &info);
    uint64_t g = tr_scope_key_size(l->scope, TR_SCOPE_KEY_GROUPS);
    if (!g) return TR_MATH_MEMORY;
    uint64_t n = binding_size(l);
    unsigned char *b = zalloc(n);
    if (!b) return TR_MATH_MEMORY;
    uint64_t head[3] = {GROUP_TAG, info.ngroups, g};
    memcpy(b, head, 24);
    int32_t rc = tr_scope_key_write(l->scope, TR_SCOPE_KEY_GROUPS, b + 24, g);
    if (rc) { free(b); return rc; }
    memcpy(b + 24 + g, &l->codec_len, 8);
    if (l->codec_len) memcpy(b + 32 + g, l->codec, (size_t)l->codec_len);
    *out = b; *len = n;
    return TR_MATH_OK;
}
/* 1 when the level's binding is exactly key[0..len) (nothing allocated beyond the descriptor scratch) */
static int binding_equal(const tr_multi_level *l, const unsigned char *key, uint64_t len) {
    tr_scope_info info;
    if (tr_scope_get_info(l->scope, &info) || len < 32) return 0;
    uint64_t g = tr_scope_key_size(l->scope, TR_SCOPE_KEY_GROUPS), head[3] = {GROUP_TAG, info.ngroups, g};
    if (!g || len != binding_size(l) || memcmp(key, head, 24)) return 0;
    if (tr_scope_key_equal(l->scope, TR_SCOPE_KEY_GROUPS, key + 24, g) != 1) return 0;
    uint64_t cl;
    memcpy(&cl, key + 24 + g, 8);
    return cl == l->codec_len && (!cl || !memcmp(key + 32 + g, l->codec, (size_t)cl));
}
/* the rows key: the source identity and version, the row count and the selection words */
static uint64_t rows_size(const tr_scope_info *info) { return tr_budget_add(96, tr_budget_bitmap_bytes(info->rows)); }
static void rows_head(const tr_scope_info *info, uint64_t head[12]) {
    uint64_t id[2];
    memcpy(id, info->identity, 16);
    head[0] = ROWS_TAG; head[1] = (uint64_t)info->source; head[2] = info->rows; head[3] = info->selected;
    head[4] = info->table; head[5] = id[0]; head[6] = id[1]; head[7] = info->version[0]; head[8] = info->version[1];
    head[9] = info->revision; head[10] = (info->rows + 63) / 64; head[11] = 0;
}
static int32_t rows_write(const tr_scope *s, unsigned char **out, uint64_t *len) {
    tr_scope_info info;
    tr_scope_view v;
    tr_scope_get_info(s, &info);
    tr_scope_get_view(s, &v);
    uint64_t n = rows_size(&info), head[12], nw = (info.rows + 63) / 64;
    unsigned char *b = zalloc(n);
    if (!b) return TR_MATH_MEMORY;
    rows_head(&info, head);
    memcpy(b, head, 96);
    if (nw) memcpy(b + 96, v.selection, (size_t)nw * 8);
    *out = b; *len = n;
    return TR_MATH_OK;
}
static int rows_equal_key(const tr_scope *s, const unsigned char *key, uint64_t len) {
    tr_scope_info info;
    tr_scope_view v;
    if (tr_scope_get_info(s, &info) || tr_scope_get_view(s, &v) || len != rows_size(&info)) return 0;
    uint64_t head[12], nw = (info.rows + 63) / 64;
    rows_head(&info, head);
    return !memcmp(key, head, 96) && (!nw || !memcmp(key + 96, v.selection, (size_t)nw * 8));
}
/* the same rows: same source and version, row count and selection bits (value columns may differ) */
static int same_rows(const tr_scope *a, const tr_scope *b) {
    tr_scope_info x, y;
    tr_scope_view u, w;
    if (tr_scope_get_info(a, &x) || tr_scope_get_info(b, &y) || tr_scope_get_view(a, &u) || tr_scope_get_view(b, &w))
        return 0;
    uint64_t hx[12], hy[12], nw = (x.rows + 63) / 64;
    rows_head(&x, hx); rows_head(&y, hy);
    return !memcmp(hx, hy, 96) && (!nw || !memcmp(u.selection, w.selection, (size_t)nw * 8));
}

/* ------------------------------------------------------------------ maps */
static void map_release(tr_multi_map *m) {
    if (!m) return;
    free(m->map); free(m->fine_key); free(m->coarse_key); free(m->rows_key);
    m->magic = 0;
    free(m);
}
void tr_multi_map_free(tr_multi_map *m) { if (map_valid(m)) map_release(m); }

static void map_finish(tr_multi_map *m) {
    uint64_t mapped = 0;
    for (uint64_t f = 0; f < m->nfine; f++) mapped += m->map[f] != TR_MULTI_UNMAPPED;
    m->mapped = mapped;
    hasher h;
    h_init(&h);
    h_word(&h, (uint64_t)m->status); h_word(&h, m->nfine); h_word(&h, m->ncoarse);
    h_bytes(&h, (const unsigned char *)m->map, m->nfine * 4);
    h_bytes(&h, m->fine_key, m->fine_len); h_bytes(&h, m->coarse_key, m->coarse_len);
    if (m->rows_key) h_bytes(&h, m->rows_key, m->rows_len);
    m->digest[0] = h.h0; m->digest[1] = h.h1;
    m->bytes = tr_budget_add(tr_budget_add(sizeof *m, m->nfine * 4),
                             tr_budget_add(tr_budget_add(m->fine_len, m->coarse_len), m->rows_len));
    m->magic = MAP_MAGIC;
}
/* an empty map with the bindings of the two levels (and the rows key when rows != 0), entries UNMAPPED */
static tr_multi_map *map_alloc(const tr_multi_level *fine, const tr_multi_level *coarse, int32_t status, int rows,
                               int32_t *rc) {
    tr_scope_info fi, ci;
    tr_scope_get_info(fine->scope, &fi);
    tr_scope_get_info(coarse->scope, &ci);
    tr_multi_map *m = zalloc(sizeof *m);
    if (!m) { *rc = TR_MATH_MEMORY; return NULL; }
    m->status = status; m->nfine = fi.ngroups; m->ncoarse = ci.ngroups;
    m->map = zalloc(fi.ngroups * 4);
    if (!m->map) { map_release(m); *rc = TR_MATH_MEMORY; return NULL; }
    for (uint64_t f = 0; f < fi.ngroups; f++) m->map[f] = TR_MULTI_UNMAPPED;
    *rc = binding_write(fine, &m->fine_key, &m->fine_len);
    if (!*rc) *rc = binding_write(coarse, &m->coarse_key, &m->coarse_len);
    if (!*rc && rows) { *rc = rows_write(fine->scope, &m->rows_key, &m->rows_len); m->rows_bound = 1; }
    if (*rc) { map_release(m); return NULL; }
    return m;
}
static uint64_t map_bytes_need(const tr_multi_level *fine, const tr_multi_level *coarse, int rows) {
    tr_scope_info fi;
    tr_scope_get_info(fine->scope, &fi);
    uint64_t b = tr_budget_add(sizeof(tr_multi_map), tr_budget_mul(fi.ngroups, 4));
    b = tr_budget_add(b, tr_budget_add(binding_size(fine), binding_size(coarse)));
    return rows ? tr_budget_add(b, rows_size(&fi)) : b;
}
static void witness_none(tr_multi_witness *w) {
    if (!w) return;
    memset(w, 0, sizeof *w);
    w->row = w->first_row = TR_MULTI_NONE;
    w->fine = w->coarse = w->first_coarse = TR_MULTI_UNMAPPED;
}

int32_t tr_multi_map_need(const tr_multi_level *fine, const tr_multi_level *coarse, int32_t target, tr_math_need *need) {
    if (!level_ok(fine) || !level_ok(coarse) || !need || (target != 0 && target != 1)) return TR_MATH_INVALID;
    need_zero(need);
    if (target == 1) {
        tr_scope_view fv, cv;
        tr_scope_get_view(fine->scope, &fv);
        tr_scope_get_view(coarse->scope, &cv);
        tr_multi_certify_device_need(&fv, &cv, need);
        return TR_MATH_OK;
    }
    tr_scope_info fi;
    tr_scope_get_info(fine->scope, &fi);
    need->need[TR_BUDGET_STATES] = fi.ngroups;
    need->need[TR_BUDGET_BYTES] = tr_budget_add(map_bytes_need(fine, coarse, 1), tr_budget_mul(fi.ngroups, 8));
    return TR_MATH_OK;
}

/* the common checks of certify / adopt / verify: valid levels of the same rows, the budget */
static int32_t map_prepare(const tr_multi_level *fine, const tr_multi_level *coarse, const tr_math_budget *budget,
                           tr_math_refusal *refusal, tr_scope_view *fv, tr_scope_view *cv) {
    tr_budget_none(refusal);
    if (!level_ok(fine) || !level_ok(coarse)) return TR_MATH_INVALID;
    if (!same_rows(fine->scope, coarse->scope)) return TR_MATH_DOMAIN_ERROR;     /* levels of other rows */
    tr_math_need need;
    tr_multi_map_need(fine, coarse, 0, &need);
    tr_math_budget dflt;
    int32_t rc = tr_budget_check(budget_or(budget, &dflt), &need, refusal);
    if (rc) return rc;
    tr_scope_get_view(fine->scope, fv);
    tr_scope_get_view(coarse->scope, cv);
    return TR_MATH_OK;
}

tr_multi_map *tr_multi_map_certify(const tr_multi_level *fine, const tr_multi_level *coarse,
                                   const tr_math_budget *budget, tr_math_refusal *refusal, tr_multi_witness *witness,
                                   int32_t *error) {
    witness_none(witness);
    tr_scope_view fv, cv;
    int32_t rc = map_prepare(fine, coarse, budget, refusal, &fv, &cv);
    if (rc) { set_error(error, rc); return NULL; }
    uint64_t *first = zalloc(fv.ngroups * 8);
    if (!first) { set_error(error, TR_MATH_MEMORY); return NULL; }
    tr_multi_map *m = map_alloc(fine, coarse, TR_MULTI_PROVEN, 1, &rc);
    if (!m) { free(first); set_error(error, rc); return NULL; }
    for (uint64_t w = 0; w < (fv.rows + 63) / 64; w++) {
        uint64_t bits = fv.selection[w];
        while (bits) {
            uint64_t r = (w << 6) | (uint64_t)__builtin_ctzll(bits);
            bits &= bits - 1;
            uint32_t f = tr_scope_group(&fv, r), q = tr_scope_group(&cv, r);
            if (m->map[f] == TR_MULTI_UNMAPPED) { m->map[f] = q; first[f] = r; continue; }
            if (m->map[f] == q) continue;
            if (witness) {                               /* the smallest row whose coarse group differs */
                witness->found = 1; witness->row = r; witness->first_row = first[f];
                witness->fine = f; witness->coarse = q; witness->first_coarse = m->map[f];
            }
            free(first); map_release(m);
            set_error(error, TR_MATH_DOMAIN_ERROR);
            return NULL;
        }
    }
    free(first);
    m->checked = fv.selected;
    map_finish(m);
    set_error(error, TR_MATH_OK);
    return m;
}

tr_multi_map *tr_multi_map_from_cert(const tr_multi_level *fine, const tr_multi_level *coarse, const tr_multi_cert *cert,
                                     const tr_math_budget *budget, tr_math_refusal *refusal, tr_multi_witness *witness,
                                     int32_t *error) {
    witness_none(witness);
    tr_scope_view fv, cv;
    int32_t rc = map_prepare(fine, coarse, budget, refusal, &fv, &cv);
    if (rc) { set_error(error, rc); return NULL; }
    if (!cert || cert->nfine != fv.ngroups || (fv.ngroups && (!cert->first || !cert->lo || !cert->hi)) ||
        cert->checked != fv.selected) { set_error(error, TR_MATH_INVALID); return NULL; }
    tr_multi_map *m = map_alloc(fine, coarse, TR_MULTI_PROVEN, 1, &rc);
    if (!m) { set_error(error, rc); return NULL; }
    int split = 0;
    for (uint64_t f = 0; f < fv.ngroups; f++) {
        uint64_t r = cert->first[f];
        if (r == TR_MULTI_NONE) {
            if (cert->lo[f] != UINT32_MAX || cert->hi[f] != 0) { rc = TR_MATH_INVALID; break; }
            continue;
        }
        if (r >= fv.rows || !tr_scope_is_selected(&fv, r) || tr_scope_group(&fv, r) != (uint32_t)f) { rc = TR_MATH_INVALID; break; }
        uint32_t q = tr_scope_group(&cv, r);
        if (cert->lo[f] > q || cert->hi[f] < q) { rc = TR_MATH_INVALID; break; }
        split |= cert->lo[f] != cert->hi[f];
        m->map[f] = q;
    }
    uint64_t w = cert->witness;
    if (!rc && w == TR_MULTI_NONE && split) rc = TR_MATH_INVALID;    /* a split group without a witness */
    if (!rc && w != TR_MULTI_NONE) {
        uint32_t f = w < fv.rows ? tr_scope_group(&fv, w) : 0;
        if (w >= fv.rows || !tr_scope_is_selected(&fv, w) || cert->first[f] == TR_MULTI_NONE || cert->first[f] >= w ||
            tr_scope_group(&cv, w) == m->map[f] || !split) rc = TR_MATH_INVALID;
        else {
            if (witness) {
                witness->found = 1; witness->row = w; witness->first_row = cert->first[f];
                witness->fine = f; witness->coarse = tr_scope_group(&cv, w); witness->first_coarse = m->map[f];
            }
            rc = TR_MATH_DOMAIN_ERROR;
        }
    }
    if (rc) { map_release(m); set_error(error, rc); return NULL; }
    m->checked = fv.selected;
    map_finish(m);
    set_error(error, TR_MATH_OK);
    return m;
}

tr_multi_map *tr_multi_map_declare(const tr_multi_level *fine, const tr_multi_level *coarse, const uint32_t *map,
                                   uint64_t n, const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    if (!level_ok(fine) || !level_ok(coarse)) { set_error(error, TR_MATH_INVALID); return NULL; }
    tr_scope_info fi, ci;
    tr_scope_get_info(fine->scope, &fi);
    tr_scope_get_info(coarse->scope, &ci);
    if (n != fi.ngroups || (n && !map)) { set_error(error, TR_MATH_INVALID); return NULL; }
    tr_math_need need;
    need_zero(&need);
    need.need[TR_BUDGET_STATES] = fi.ngroups;
    need.need[TR_BUDGET_BYTES] = map_bytes_need(fine, coarse, 0);
    tr_math_budget dflt;
    int32_t rc = tr_budget_check(budget_or(budget, &dflt), &need, refusal);
    if (rc) { set_error(error, rc); return NULL; }
    for (uint64_t f = 0; f < n; f++)
        if (map[f] != TR_MULTI_UNMAPPED && (uint64_t)map[f] >= ci.ngroups) { set_error(error, TR_MATH_INVALID); return NULL; }
    tr_multi_map *m = map_alloc(fine, coarse, TR_MULTI_DECLARED, 0, &rc);
    if (!m) { set_error(error, rc); return NULL; }
    if (n) memcpy(m->map, map, (size_t)n * 4);
    map_finish(m);
    set_error(error, TR_MATH_OK);
    return m;
}

int32_t tr_multi_map_check(const tr_multi_map *m, const tr_multi_level *fine, const tr_multi_level *coarse) {
    if (!map_valid(m) || !level_ok(fine) || !level_ok(coarse)) return TR_MATH_INVALID;
    if (!binding_equal(fine, m->fine_key, m->fine_len) || !binding_equal(coarse, m->coarse_key, m->coarse_len))
        return TR_MATH_DOMAIN_ERROR;                    /* another scope, grouping or codec */
    if (m->rows_bound && (!rows_equal_key(fine->scope, m->rows_key, m->rows_len) ||
                          !rows_equal_key(coarse->scope, m->rows_key, m->rows_len)))
        return TR_MATH_DOMAIN_ERROR;                    /* proven on other rows */
    return TR_MATH_OK;
}

tr_multi_map *tr_multi_map_verify(const tr_multi_map *d, const tr_multi_level *fine, const tr_multi_level *coarse,
                                  const tr_math_budget *budget, tr_math_refusal *refusal, tr_multi_witness *witness,
                                  int32_t *error) {
    witness_none(witness);
    if (!map_valid(d)) { set_error(error, TR_MATH_INVALID); return NULL; }
    tr_scope_view fv, cv;
    int32_t rc = map_prepare(fine, coarse, budget, refusal, &fv, &cv);
    if (rc) { set_error(error, rc); return NULL; }
    if (!binding_equal(fine, d->fine_key, d->fine_len) || !binding_equal(coarse, d->coarse_key, d->coarse_len)) {
        set_error(error, TR_MATH_DOMAIN_ERROR); return NULL;
    }
    for (uint64_t w = 0; w < (fv.rows + 63) / 64; w++) {
        uint64_t bits = fv.selection[w];
        while (bits) {
            uint64_t r = (w << 6) | (uint64_t)__builtin_ctzll(bits);
            bits &= bits - 1;
            uint32_t f = tr_scope_group(&fv, r), q = tr_scope_group(&cv, r);
            if (d->map[f] == q) continue;
            if (witness) {
                witness->found = 1; witness->row = r; witness->fine = f; witness->coarse = q;
                witness->first_coarse = d->map[f];
            }
            set_error(error, TR_MATH_DOMAIN_ERROR);
            return NULL;
        }
    }
    tr_multi_map *m = map_alloc(fine, coarse, TR_MULTI_PROVEN, 1, &rc);
    if (!m) { set_error(error, rc); return NULL; }
    if (d->nfine) memcpy(m->map, d->map, (size_t)d->nfine * 4);
    m->checked = fv.selected;
    map_finish(m);
    set_error(error, TR_MATH_OK);
    return m;
}

static tr_multi_map *map_clone(const tr_multi_map *s, int32_t *rc) {
    tr_multi_map *m = zalloc(sizeof *m);
    if (!m) { *rc = TR_MATH_MEMORY; return NULL; }
    *m = *s;
    m->magic = 0;
    m->map = zalloc(s->nfine * 4);
    m->fine_key = zalloc(s->fine_len); m->coarse_key = zalloc(s->coarse_len);
    m->rows_key = s->rows_key ? zalloc(s->rows_len) : NULL;
    if (!m->map || !m->fine_key || !m->coarse_key || (s->rows_key && !m->rows_key)) {
        map_release(m); *rc = TR_MATH_MEMORY; return NULL;
    }
    if (s->nfine) memcpy(m->map, s->map, (size_t)s->nfine * 4);
    memcpy(m->fine_key, s->fine_key, (size_t)s->fine_len);
    memcpy(m->coarse_key, s->coarse_key, (size_t)s->coarse_len);
    if (s->rows_key) memcpy(m->rows_key, s->rows_key, (size_t)s->rows_len);
    m->magic = MAP_MAGIC;
    *rc = TR_MATH_OK;
    return m;
}

tr_multi_map *tr_multi_map_compose(const tr_multi_map *a, const tr_multi_map *b, const tr_math_budget *budget,
                                   tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    if (!map_valid(a) || !map_valid(b)) { set_error(error, TR_MATH_INVALID); return NULL; }
    if (a->ncoarse != b->nfine || a->coarse_len != b->fine_len || memcmp(a->coarse_key, b->fine_key, (size_t)a->coarse_len)) {
        set_error(error, TR_MATH_DOMAIN_ERROR); return NULL;           /* the levels do not chain */
    }
    int proven = a->status == TR_MULTI_PROVEN && b->status == TR_MULTI_PROVEN && a->rows_bound && b->rows_bound &&
                 a->rows_len == b->rows_len && !memcmp(a->rows_key, b->rows_key, (size_t)a->rows_len);
    tr_math_need need;
    need_zero(&need);
    need.need[TR_BUDGET_STATES] = a->nfine;
    need.need[TR_BUDGET_BYTES] = tr_budget_add(tr_budget_add(sizeof(tr_multi_map), tr_budget_mul(a->nfine, 4)),
                                               tr_budget_add(tr_budget_add(a->fine_len, b->coarse_len),
                                                             proven ? a->rows_len : 0));
    tr_math_budget dflt;
    int32_t rc = tr_budget_check(budget_or(budget, &dflt), &need, refusal);
    if (rc) { set_error(error, rc); return NULL; }
    tr_multi_map *m = zalloc(sizeof *m);
    if (!m) { set_error(error, TR_MATH_MEMORY); return NULL; }
    m->status = proven ? TR_MULTI_PROVEN : TR_MULTI_DECLARED; m->rows_bound = proven;
    m->nfine = a->nfine; m->ncoarse = b->ncoarse; m->checked = proven ? a->checked : 0;
    m->map = zalloc(a->nfine * 4);
    m->fine_key = zalloc(a->fine_len); m->coarse_key = zalloc(b->coarse_len);
    m->rows_key = proven ? zalloc(a->rows_len) : NULL;
    if (!m->map || !m->fine_key || !m->coarse_key || (proven && !m->rows_key)) {
        map_release(m); set_error(error, TR_MATH_MEMORY); return NULL;
    }
    for (uint64_t f = 0; f < a->nfine; f++) m->map[f] = a->map[f] == TR_MULTI_UNMAPPED ? TR_MULTI_UNMAPPED : b->map[a->map[f]];
    memcpy(m->fine_key, a->fine_key, (size_t)a->fine_len); m->fine_len = a->fine_len;
    memcpy(m->coarse_key, b->coarse_key, (size_t)b->coarse_len); m->coarse_len = b->coarse_len;
    if (proven) { memcpy(m->rows_key, a->rows_key, (size_t)a->rows_len); m->rows_len = a->rows_len; }
    map_finish(m);
    set_error(error, TR_MATH_OK);
    return m;
}

int32_t tr_multi_map_get_info(const tr_multi_map *m, tr_multi_map_info *out) {
    if (!map_valid(m) || !out) return TR_MATH_INVALID;
    memset(out, 0, sizeof *out);
    out->nfine = m->nfine; out->ncoarse = m->ncoarse; out->mapped = m->mapped; out->checked = m->checked;
    out->bytes = m->bytes; out->digest[0] = m->digest[0]; out->digest[1] = m->digest[1];
    out->status = m->status; out->rows_bound = m->rows_bound;
    return TR_MATH_OK;
}
int32_t tr_multi_map_get(const tr_multi_map *m, uint32_t *out, uint64_t capacity) {
    if (!map_valid(m) || (m->nfine && (!out || capacity < m->nfine))) return TR_MATH_INVALID;
    if (m->nfine) memcpy(out, m->map, (size_t)m->nfine * 4);
    return TR_MATH_OK;
}
int32_t tr_multi_map_equal(const tr_multi_map *a, const tr_multi_map *b) {
    if (!map_valid(a) || !map_valid(b)) return TR_MATH_INVALID;
    return a->status == b->status && a->rows_bound == b->rows_bound && a->nfine == b->nfine && a->ncoarse == b->ncoarse &&
           !memcmp(a->map, b->map, (size_t)a->nfine * 4) && a->fine_len == b->fine_len &&
           !memcmp(a->fine_key, b->fine_key, (size_t)a->fine_len) && a->coarse_len == b->coarse_len &&
           !memcmp(a->coarse_key, b->coarse_key, (size_t)a->coarse_len) && a->rows_len == b->rows_len &&
           (!a->rows_len || !memcmp(a->rows_key, b->rows_key, (size_t)a->rows_len));
}

/* ------------------------------------------------------------------ hierarchies */
static void hier_release(tr_multi_hier *h) {
    if (!h) return;
    for (int i = 0; i < TR_MULTI_MAX_LEVELS; i++) tr_bank_free(h->bank[i]);
    for (int i = 0; i < TR_MULTI_MAX_LEVELS - 1; i++) tr_multi_map_free(h->map[i]);
    free(h->exps);
    h->magic = 0;
    free(h);
}
void tr_multi_hier_free(tr_multi_hier *h) { if (hier_valid(h)) hier_release(h); }

/* every level: valid, the same rows as level 0; bad: TR_MATH_INVALID / TR_MATH_DOMAIN_ERROR */
static int32_t levels_check(const tr_multi_level *levels, int32_t nlevels) {
    if (!levels) return TR_MATH_INVALID;
    if (nlevels < 2) return TR_MATH_INVALID;
    if (nlevels > TR_MULTI_MAX_LEVELS) return TR_MATH_UNSUPPORTED;
    for (int32_t i = 0; i < nlevels; i++) if (!level_ok(&levels[i])) return TR_MATH_INVALID;
    for (int32_t i = 1; i < nlevels; i++) if (!same_rows(levels[0].scope, levels[i].scope)) return TR_MATH_DOMAIN_ERROR;
    return TR_MATH_OK;
}
/* fine is the bank of the scope (scoped, the same descriptor digest, groups, columns and rows) */
static int bank_of(const tr_bank *b, const tr_scope *s) {
    tr_bank_info bi;
    tr_scope_info si;
    if (tr_bank_get_info(b, &bi) || tr_scope_get_info(s, &si)) return 0;
    return bi.scoped && bi.digest[0] == si.digest[0] && bi.digest[1] == si.digest[1] && bi.ngroups == si.ngroups &&
           bi.ncols == si.ncols && bi.rows == si.selected;
}
/* the coarse levels' needs: states and bytes of every coarse bank and its rollup scratch, the map copies */
static int32_t hier_need_raw(const tr_bank *fine, const uint64_t *groups, int32_t nlevels, int32_t local,
                             tr_math_need *need) {
    tr_bank_info bi;
    if (tr_bank_get_info(fine, &bi)) return TR_MATH_INVALID;
    need_zero(need);
    need->need[TR_BUDGET_VARIABLES] = (uint64_t)tr_bank_popc32(bi.joint);
    need->need[TR_BUDGET_DEGREE] = (uint64_t)bi.degree;
    need->need[TR_BUDGET_SUPPORT] = (uint64_t)bi.nterms;
    uint64_t states = 0, bytes = tr_budget_add(sizeof(tr_multi_hier), tr_budget_add(bi.bytes,
                                               tr_budget_mul((uint64_t)bi.nterms, (uint64_t)bi.ncols)));
    for (int32_t i = 1; i < nlevels; i++) {
        tr_math_need r;
        tr_bank_rollup_need(fine, groups[i], TR_BANK_NEED_CPU, &r);
        states = tr_budget_add(states, tr_budget_mul(groups[i], (uint64_t)bi.nterms));
        bytes = tr_budget_add(bytes, r.need[TR_BUDGET_BYTES]);                  /* the coarse bank and its scratch */
        bytes = tr_budget_add(bytes, tr_budget_add(sizeof(tr_multi_map), tr_budget_mul(groups[i - 1], 8)));  /* map copy, translated map */
        if (local) {          /* an update: the staged states, the dirty lists and the wide accumulators */
            bytes = tr_budget_add(bytes, tr_bank_state_bytes(groups[i], bi.nterms));
            bytes = tr_budget_add(bytes, tr_budget_mul(groups[i - 1], 8));
            bytes = tr_budget_add(bytes, tr_budget_mul(groups[i], tr_budget_add(12, tr_budget_mul((uint64_t)bi.nterms + 1, 40))));
        }
    }
    need->need[TR_BUDGET_STATES] = states;
    need->need[TR_BUDGET_BYTES] = bytes;
    return TR_MATH_OK;
}
int32_t tr_multi_hier_need(const tr_multi_level *levels, int32_t nlevels, const tr_bank *fine, tr_math_need *need) {
    if (!need || !fine) return TR_MATH_INVALID;
    int32_t rc = levels_check(levels, nlevels);
    if (rc == TR_MATH_DOMAIN_ERROR) rc = TR_MATH_OK;    /* the needs do not depend on the rows */
    if (rc) return rc;
    uint64_t groups[TR_MULTI_MAX_LEVELS];
    for (int32_t i = 0; i < nlevels; i++) {
        tr_scope_info info;
        tr_scope_get_info(levels[i].scope, &info);
        groups[i] = info.ngroups;
    }
    return hier_need_raw(fine, groups, nlevels, 0, need);
}

/* the maps against the levels: bindings, groups, and proven maps bound to the levels' rows */
static int32_t maps_check(const tr_multi_level *levels, int32_t nlevels, const tr_multi_map *const *maps, int *proven) {
    if (!maps) return TR_MATH_INVALID;
    *proven = 1;
    for (int32_t i = 0; i + 1 < nlevels; i++) {
        const tr_multi_map *m = maps[i];
        if (!map_valid(m)) return TR_MATH_INVALID;
        int32_t rc = tr_multi_map_check(m, &levels[i], &levels[i + 1]);
        if (rc) return rc;
        if (m->status != TR_MULTI_PROVEN || !m->rows_bound) *proven = 0;
    }
    return TR_MATH_OK;
}
static int exps_equal(const tr_bank *a, const uint8_t *exps, uint64_t n) {
    uint8_t buf[TR_BANK_MAX_TERMS * 4];
    uint8_t *p = n <= sizeof buf ? buf : malloc((size_t)n);
    if (!p) return 0;
    int same = !tr_bank_get_exponents(a, p, n) && (!n || !memcmp(p, exps, (size_t)n));
    if (p != buf) free(p);
    return same;
}
/* a map with UNMAPPED entries as 0 (tr_bank_rollup needs ids < coarse; unmapped fine groups are absent) */
static void translate(const tr_multi_map *m, uint32_t *out) {
    for (uint64_t f = 0; f < m->nfine; f++) out[f] = m->map[f] == TR_MULTI_UNMAPPED ? 0 : m->map[f];
}
/* a present fine group without a parent: its id, else TR_MULTI_NONE */
static uint64_t orphan(const tr_bank *b, const tr_multi_map *m) {
    tr_bank_states st;
    tr_bank_get_states(b, &st);
    for (uint64_t g = 0; g < st.ngroups; g++) if (st.rows[g] && m->map[g] == TR_MULTI_UNMAPPED) return g;
    return TR_MULTI_NONE;
}

static tr_multi_hier *hier_alloc(const tr_bank *fine, int32_t nlevels, int32_t *rc) {
    tr_bank_info bi;
    tr_bank_get_info(fine, &bi);
    tr_multi_hier *h = zalloc(sizeof *h);
    if (!h) { *rc = TR_MATH_MEMORY; return NULL; }
    h->levels = nlevels; h->ncols = bi.ncols; h->nterms = bi.nterms; h->mode = bi.mode; h->rows = bi.rows;
    uint64_t ne = (uint64_t)bi.nterms * (uint64_t)bi.ncols;
    h->exps = zalloc(ne);
    if (!h->exps || tr_bank_get_exponents(fine, h->exps, ne)) { hier_release(h); *rc = TR_MATH_MEMORY; return NULL; }
    *rc = TR_MATH_OK;
    return h;
}
static void hier_finish(tr_multi_hier *h) {
    uint64_t bytes = tr_budget_add(sizeof *h, (uint64_t)h->nterms * (uint64_t)h->ncols), work = 0;
    for (int32_t i = 0; i < h->levels; i++) {
        tr_bank_info bi;
        tr_bank_get_info(h->bank[i], &bi);
        bytes = tr_budget_add(bytes, bi.bytes);
        work = tr_budget_add(work, h->level_work[i]);
        if (i + 1 < h->levels) bytes = tr_budget_add(bytes, h->map[i]->bytes);
    }
    h->bytes = bytes; h->work = work;
    h->magic = HIER_MAGIC;
}
/* level i + 1 rolled up completely from level i along map (origin: its scope when scoped) */
static int32_t level_rollup(tr_multi_hier *h, int32_t i, const tr_multi_map *m, const tr_scope *origin,
                            const tr_math_budget *budget, tr_math_refusal *refusal, uint64_t *bad_group) {
    uint64_t g = orphan(h->bank[i], m);
    if (g != TR_MULTI_NONE) { if (bad_group) *bad_group = g; return TR_MATH_DOMAIN_ERROR; }
    uint32_t *tmap = zalloc(m->nfine * 4);
    if (!tmap) return TR_MATH_MEMORY;
    translate(m, tmap);
    int32_t rc;
    h->bank[i + 1] = tr_bank_rollup(h->bank[i], tmap, m->ncoarse, origin, budget, refusal, &rc);
    free(tmap);
    if (!h->bank[i + 1]) return rc;
    h->level_work[i + 1] = tr_budget_mul(m->nfine, (uint64_t)h->nterms + 1);
    h->strategy[i + 1] = TR_MULTI_ROLLUP;
    h->scoped[i + 1] = origin != NULL;
    return TR_MATH_OK;
}

tr_multi_hier *tr_multi_hier_build(const tr_multi_level *levels, int32_t nlevels, const tr_bank *fine,
                                   const tr_multi_map *const *maps, const tr_math_budget *budget,
                                   tr_math_refusal *refusal, uint64_t *bad_group, int32_t *error) {
    tr_budget_none(refusal);
    if (bad_group) *bad_group = TR_MULTI_NONE;
    int32_t rc = levels_check(levels, nlevels);
    if (rc) { set_error(error, rc); return NULL; }
    tr_bank_info bi;
    if (!fine || tr_bank_get_info(fine, &bi)) { set_error(error, TR_MATH_INVALID); return NULL; }
    if (!bank_of(fine, levels[0].scope)) { set_error(error, TR_MATH_DOMAIN_ERROR); return NULL; }   /* another scope */
    int proven;
    rc = maps_check(levels, nlevels, maps, &proven);
    if (rc) { set_error(error, rc); return NULL; }
    uint64_t groups[TR_MULTI_MAX_LEVELS];
    for (int32_t i = 0; i < nlevels; i++) {
        tr_scope_info info;
        tr_scope_get_info(levels[i].scope, &info);
        groups[i] = info.ngroups;
    }
    tr_math_need need;
    hier_need_raw(fine, groups, nlevels, 0, &need);
    tr_math_budget dflt;
    const tr_math_budget *b = budget_or(budget, &dflt);
    rc = tr_budget_check(b, &need, refusal);
    if (rc) { set_error(error, rc); return NULL; }
    tr_multi_hier *h = hier_alloc(fine, nlevels, &rc);
    if (!h) { set_error(error, rc); return NULL; }
    h->bank[0] = tr_bank_clone(fine, &rc);
    if (!h->bank[0]) { hier_release(h); set_error(error, rc); return NULL; }
    h->scoped[0] = 1; h->strategy[0] = TR_MULTI_COPY;
    int chain = 1;
    for (int32_t i = 0; i + 1 < nlevels; i++) {
        h->map[i] = map_clone(maps[i], &rc);
        if (!h->map[i]) break;
        chain = chain && maps[i]->status == TR_MULTI_PROVEN && maps[i]->rows_bound;
        const tr_scope *origin = chain && tr_scope_same_rows(levels[i + 1].scope, levels[0].scope) == 1 ?
                                 levels[i + 1].scope : NULL;
        rc = level_rollup(h, i, maps[i], origin, b, refusal, bad_group);
        if (rc) break;
    }
    if (rc) { hier_release(h); set_error(error, rc); return NULL; }
    h->status = proven ? TR_MULTI_PROVEN : TR_MULTI_DECLARED;
    hier_finish(h);
    set_error(error, TR_MATH_OK);
    return h;
}

/* ---- localized update */
/* 1 when group g differs between two banks of the same shape */
static int group_differs(const tr_bank_states *a, const tr_bank_states *b, uint64_t g) {
    uint64_t m = (uint64_t)a->nterms;
    if (a->rows[g] != b->rows[g]) return 1;
    return m && (memcmp(a->count + g * m, b->count + g * m, (size_t)m * 8) || memcmp(a->sum + g * m, b->sum + g * m, (size_t)m * 16));
}
/* the wide accumulators of an update: rows and counts in __int128, sums in 192-bit words */
static void wide_from_i128(uint64_t w[3], __int128 x) { w[0] = 0; w[1] = 0; w[2] = 0; tr_wide_add128(w, x); }
static void wide_add(uint64_t w[3], __int128 x, int negate) {
    if (!negate) { tr_wide_add128(w, x); return; }
    /* w -= x: add the 192-bit negation of x (exact also for x = -2^127) */
    uint64_t n[3] = {(uint64_t)x, (uint64_t)((unsigned __int128)x >> 64), x < 0 ? UINT64_MAX : 0};
    uint64_t carry = 1;
    for (int i = 0; i < 3; i++) { uint64_t v = ~n[i] + carry; carry = carry && v == 0; n[i] = v; }
    uint64_t c = 0;
    for (int i = 0; i < 3; i++) {
        uint64_t s = w[i] + n[i], c1 = s < w[i];
        w[i] = s + c;
        c = c1 | (w[i] < s);
    }
}
/* the 192-bit value as a symmetric 128-bit sum (|x| <= 2^127 - 1), else 0 */
static int wide_to_i128(const uint64_t w[3], __int128 *out) {
    uint64_t ext = (w[1] >> 63) ? UINT64_MAX : 0;
    if (w[2] != ext) return 0;
    __int128 x = (__int128)(((unsigned __int128)w[1] << 64) | w[0]);
    if (tr_multi_mag(x) > I128_LIMIT) return 0;
    *out = x;
    return 1;
}

/* level i + 1 of h2 from level i + 1 of h (old) by the dirty children of level i: old states of h (oldc), new of h2.
 * changed (in/out): the dirty children (sorted) in, the changed coarse groups out. */
static int32_t level_local(tr_multi_hier *h2, const tr_multi_hier *h, int32_t i, const tr_multi_map *om,
                           const tr_multi_map *nm, uint32_t *dirty, uint64_t ndirty, uint32_t *changed,
                           uint64_t *nchanged, const tr_scope *origin, const tr_math_budget *budget,
                           tr_math_refusal *refusal, uint64_t *bad_group) {
    tr_bank_states oc, nc, op;                       /* old child, new child, old parent */
    tr_bank_get_states(h->bank[i], &oc);
    tr_bank_get_states(h2->bank[i], &nc);
    tr_bank_get_states(h->bank[i + 1], &op);
    const uint64_t m = (uint64_t)h->nterms, G = op.ngroups;
    for (uint64_t k = 0; k < ndirty; k++) {
        uint32_t c = dirty[k];
        if (nc.rows[c] && nm->map[c] == TR_MULTI_UNMAPPED) { if (bad_group) *bad_group = c; return TR_MATH_DOMAIN_ERROR; }
    }
    /* the touched parents, each with a local slot */
    uint32_t *slot = zalloc(G * 4);
    uint32_t *touched = zalloc(tr_budget_mul(2, ndirty) * 4 + 4);
    if (!slot || !touched) { free(slot); free(touched); return TR_MATH_MEMORY; }
    for (uint64_t q = 0; q < G; q++) slot[q] = UINT32_MAX;
    uint64_t nt = 0;
    for (uint64_t k = 0; k < ndirty; k++) {
        uint32_t c = dirty[k], ps[2] = {oc.rows[c] ? om->map[c] : TR_MULTI_UNMAPPED, nc.rows[c] ? nm->map[c] : TR_MULTI_UNMAPPED};
        for (int s = 0; s < 2; s++)
            if (ps[s] != TR_MULTI_UNMAPPED && slot[ps[s]] == UINT32_MAX) { slot[ps[s]] = (uint32_t)nt; touched[nt++] = ps[s]; }
    }
    /* accumulators: per touched parent, rows and counts (__int128) and sums (192 bits), from the old parent */
    uint64_t width = m + 1;
    __int128 *cnt = zalloc(tr_budget_mul(tr_budget_mul(nt, width), 16) + 16);
    uint64_t *wsum = zalloc(tr_budget_mul(tr_budget_mul(nt, m), 24) + 24);
    if (!cnt || !wsum) { free(slot); free(touched); free(cnt); free(wsum); return TR_MATH_MEMORY; }
    for (uint64_t t = 0; t < nt; t++) {
        uint64_t q = touched[t];
        cnt[t * width] = op.rows[q];
        for (uint64_t j = 0; j < m; j++) {
            cnt[t * width + 1 + j] = op.count[q * m + j];
            wide_from_i128(wsum + (t * m + j) * 3, tr_i128_get(op.sum[q * m + j]));
        }
    }
    for (uint64_t k = 0; k < ndirty; k++) {
        uint32_t c = dirty[k];
        for (int s = 0; s < 2; s++) {                 /* minus the old child at its old parent, plus the new one */
            const tr_bank_states *cs = s ? &nc : &oc;
            uint32_t q = cs->rows[c] ? (s ? nm->map[c] : om->map[c]) : TR_MULTI_UNMAPPED;
            if (q == TR_MULTI_UNMAPPED) continue;
            uint64_t t = slot[q];
            __int128 sign = s ? 1 : -1;
            cnt[t * width] += sign * (__int128)cs->rows[c];
            for (uint64_t j = 0; j < m; j++) {
                cnt[t * width + 1 + j] += sign * (__int128)cs->count[(uint64_t)c * m + j];
                wide_add(wsum + (t * m + j) * 3, tr_i128_get(cs->sum[(uint64_t)c * m + j]), !s);
            }
        }
    }
    /* the staged new parent level: the old states with the touched groups replaced, checked */
    uint64_t gm = G * m;
    uint64_t *rows = zalloc(G * 8), *count = zalloc(gm * 8 + 8);
    tr_math_i128 *sum = zalloc(gm * 16 + 16);
    int32_t rc = (!rows || !count || !sum) ? TR_MATH_MEMORY : TR_MATH_OK;
    if (!rc) {
        memcpy(rows, op.rows, (size_t)G * 8);
        if (gm) { memcpy(count, op.count, (size_t)gm * 8); memcpy(sum, op.sum, (size_t)gm * 16); }
        for (uint64_t t = 0; t < nt && !rc; t++) {
            uint64_t q = touched[t];
            for (uint64_t j = 0; j <= m && !rc; j++) {
                __int128 v = cnt[t * width + j];
                if (v < 0 || v > (__int128)UINT64_MAX) { rc = TR_MATH_OVERFLOW; break; }
                if (j == 0) rows[q] = (uint64_t)v; else count[q * m + j - 1] = (uint64_t)v;
            }
            for (uint64_t j = 0; j < m && !rc; j++) {
                __int128 s;
                if (!wide_to_i128(wsum + (t * m + j) * 3, &s)) rc = TR_MATH_OVERFLOW;
                else sum[q * m + j] = tr_i128_set(s);
            }
        }
    }
    free(cnt); free(wsum);
    if (!rc) {
        tr_bank_states st;
        memset(&st, 0, sizeof st);
        st.ngroups = G; st.nterms = h->nterms; st.rows = rows; st.count = count; st.sum = sum;
        h2->bank[i + 1] = tr_bank_from_states(origin, h->ncols, h->exps, h->nterms, h->mode, &st, budget, refusal, &rc);
    }
    free(rows); free(count); free(sum);
    if (!rc) {                                        /* the coarse groups that changed: the next level's dirty set */
        tr_bank_states np;
        tr_bank_get_states(h2->bank[i + 1], &np);
        uint64_t n = 0;
        for (uint64_t q = 0; q < G; q++) if (slot[q] != UINT32_MAX && group_differs(&op, &np, q)) changed[n++] = (uint32_t)q;
        *nchanged = n;
        h2->level_work[i + 1] = tr_budget_add(tr_budget_mul(tr_budget_mul(ndirty, 2), width), tr_budget_mul(nt, width));
        h2->strategy[i + 1] = TR_MULTI_LOCAL;
        h2->scoped[i + 1] = origin != NULL;
    }
    free(slot); free(touched);
    return rc;
}

/* level i + 1 of h2: the unchanged states of h's level i + 1, adopted again (scoped to origin, or unscoped) */
static int32_t level_copy(tr_multi_hier *h2, const tr_multi_hier *h, int32_t i, const tr_scope *origin,
                          const tr_math_budget *budget, tr_math_refusal *refusal) {
    tr_bank_states st;
    tr_bank_get_states(h->bank[i + 1], &st);
    int32_t rc;
    h2->bank[i + 1] = tr_bank_from_states(origin, h->ncols, h->exps, h->nterms, h->mode, &st, budget, refusal, &rc);
    if (!h2->bank[i + 1]) return rc;
    h2->strategy[i + 1] = TR_MULTI_COPY; h2->scoped[i + 1] = origin != NULL; h2->level_work[i + 1] = 0;
    return TR_MATH_OK;
}

tr_multi_hier *tr_multi_hier_update(const tr_multi_hier *h, const tr_bank *fine, const tr_multi_level *levels,
                                    const tr_multi_map *const *maps, const tr_math_budget *budget,
                                    tr_math_refusal *refusal, uint64_t *bad_group, int32_t *error) {
    tr_budget_none(refusal);
    if (bad_group) *bad_group = TR_MULTI_NONE;
    tr_bank_info bi, b0;
    if (!hier_valid(h) || !fine || tr_bank_get_info(fine, &bi) || (!levels) != (!maps)) { set_error(error, TR_MATH_INVALID); return NULL; }
    tr_bank_get_info(h->bank[0], &b0);
    if (bi.ncols != h->ncols || bi.nterms != h->nterms || bi.mode != h->mode || bi.ngroups != b0.ngroups ||
        !exps_equal(fine, h->exps, (uint64_t)h->nterms * (uint64_t)h->ncols)) { set_error(error, TR_MATH_DOMAIN_ERROR); return NULL; }
    int32_t rc, proven = 0, nlevels = h->levels;
    const tr_multi_map *nmaps[TR_MULTI_MAX_LEVELS - 1];
    if (levels) {
        rc = levels_check(levels, nlevels);
        if (!rc && !bank_of(fine, levels[0].scope)) rc = TR_MATH_DOMAIN_ERROR;
        if (!rc) rc = maps_check(levels, nlevels, maps, &proven);
        for (int32_t i = 0; !rc && i + 1 < nlevels; i++)
            if (maps[i]->nfine != h->map[i]->nfine || maps[i]->ncoarse != h->map[i]->ncoarse) rc = TR_MATH_DOMAIN_ERROR;
        if (rc) { set_error(error, rc); return NULL; }
        for (int32_t i = 0; i + 1 < nlevels; i++) nmaps[i] = maps[i];
    } else for (int32_t i = 0; i + 1 < nlevels; i++) nmaps[i] = h->map[i];
    uint64_t groups[TR_MULTI_MAX_LEVELS];
    for (int32_t i = 0; i < nlevels; i++) { tr_bank_info li; tr_bank_get_info(h->bank[i], &li); groups[i] = li.ngroups; }
    tr_math_need need;
    hier_need_raw(fine, groups, nlevels, 1, &need);
    tr_math_budget dflt;
    const tr_math_budget *b = budget_or(budget, &dflt);
    rc = tr_budget_check(b, &need, refusal);
    if (rc) { set_error(error, rc); return NULL; }
    tr_multi_hier *h2 = hier_alloc(fine, nlevels, &rc);
    if (!h2) { set_error(error, rc); return NULL; }
    h2->bank[0] = tr_bank_clone(fine, &rc);
    if (!h2->bank[0]) { hier_release(h2); set_error(error, rc); return NULL; }
    h2->scoped[0] = bi.scoped; h2->strategy[0] = TR_MULTI_COPY;
    uint64_t gmax = 0;
    for (int32_t i = 0; i < nlevels; i++) if (groups[i] > gmax) gmax = groups[i];
    uint32_t *dirty = zalloc(gmax * 4), *next = zalloc(gmax * 4);
    if (!dirty || !next) { free(dirty); free(next); hier_release(h2); set_error(error, TR_MATH_MEMORY); return NULL; }
    /* level 0: the changed groups */
    tr_bank_states o0, n0;
    tr_bank_get_states(h->bank[0], &o0);
    tr_bank_get_states(h2->bank[0], &n0);
    uint64_t nchanged = 0;
    for (uint64_t g = 0; g < groups[0]; g++) if (group_differs(&o0, &n0, g)) dirty[nchanged++] = (uint32_t)g;
    h2->changed = nchanged;
    h2->level_work[0] = tr_budget_mul(groups[0], (uint64_t)h->nterms + 1);   /* the comparison */
    int chain = levels != NULL;
    for (int32_t i = 0; !rc && i + 1 < nlevels; i++) {
        h2->map[i] = map_clone(nmaps[i], &rc);
        if (!h2->map[i]) break;
        chain = chain && nmaps[i]->status == TR_MULTI_PROVEN && nmaps[i]->rows_bound;
        const tr_scope *origin = chain && tr_scope_same_rows(levels[i + 1].scope, levels[0].scope) == 1 ?
                                 levels[i + 1].scope : NULL;
        /* the dirty children: changed, or remapped (their parent differs) */
        uint64_t nd = nchanged;
        {   /* merge the remapped children into the sorted changed list */
            uint64_t a = 0, k = 0;
            for (uint64_t g = 0; g < groups[i]; g++) {
                int ch = a < nchanged && dirty[a] == (uint32_t)g;
                if (ch) a++;
                if (ch || h->map[i]->map[g] != nmaps[i]->map[g]) next[k++] = (uint32_t)g;
            }
            nd = k;
            uint32_t *t = dirty; dirty = next; next = t;
        }
        if (!nd) {                                                  /* nothing moved: the same coarse states */
            rc = level_copy(h2, h, i, origin, b, refusal);
            nchanged = 0;
            continue;
        }
        if (2 * nd > groups[i]) {                                   /* adaptive: most children moved */
            rc = level_rollup(h2, i, nmaps[i], origin, b, refusal, bad_group);
            if (rc) break;
            tr_bank_states op, np;
            tr_bank_get_states(h->bank[i + 1], &op);
            tr_bank_get_states(h2->bank[i + 1], &np);
            nchanged = 0;
            for (uint64_t q = 0; q < groups[i + 1]; q++) if (group_differs(&op, &np, q)) next[nchanged++] = (uint32_t)q;
            uint32_t *t = dirty; dirty = next; next = t;
            continue;
        }
        rc = level_local(h2, h, i, h->map[i], nmaps[i], dirty, nd, next, &nchanged, origin, b, refusal, bad_group);
        uint32_t *t = dirty; dirty = next; next = t;
    }
    free(dirty); free(next);
    if (rc) { hier_release(h2); set_error(error, rc); return NULL; }
    if (levels) h2->status = proven ? TR_MULTI_PROVEN : TR_MULTI_DECLARED;
    else h2->status = h->status == TR_MULTI_DECLARED ? TR_MULTI_DECLARED : TR_MULTI_CARRIED;
    hier_finish(h2);
    set_error(error, TR_MATH_OK);
    return h2;
}

int32_t tr_multi_hier_get_info(const tr_multi_hier *h, tr_multi_hier_info *out) {
    if (!hier_valid(h) || !out) return TR_MATH_INVALID;
    memset(out, 0, sizeof *out);
    out->levels = h->levels; out->status = h->status; out->ncols = h->ncols; out->nterms = h->nterms; out->mode = h->mode;
    out->bytes = h->bytes; out->rows = h->rows; out->work = h->work; out->changed = h->changed;
    for (int32_t i = 0; i < h->levels; i++) {
        tr_bank_info bi;
        tr_bank_get_info(h->bank[i], &bi);
        out->groups[i] = bi.ngroups; out->present[i] = bi.present; out->level_bytes[i] = bi.bytes;
        out->level_work[i] = h->level_work[i]; out->scoped[i] = h->scoped[i]; out->strategy[i] = h->strategy[i];
        if (i + 1 < h->levels) out->map_status[i] = h->map[i]->status;
    }
    return TR_MATH_OK;
}
int32_t tr_multi_hier_states(const tr_multi_hier *h, int32_t level, tr_bank_states *out) {
    if (!hier_valid(h) || !out || level < 0 || level >= h->levels) return TR_MATH_INVALID;
    return tr_bank_get_states(h->bank[level], out);
}
tr_bank *tr_multi_hier_bank(const tr_multi_hier *h, int32_t level, int32_t *error) {
    if (!hier_valid(h) || level < 0 || level >= h->levels) { set_error(error, TR_MATH_INVALID); return NULL; }
    return tr_bank_clone(h->bank[level], error);
}
tr_multi_map *tr_multi_hier_map(const tr_multi_hier *h, int32_t level, int32_t *error) {
    if (!hier_valid(h) || level < 0 || level + 1 >= h->levels) { set_error(error, TR_MATH_INVALID); return NULL; }
    int32_t rc;
    tr_multi_map *m = map_clone(h->map[level], &rc);
    set_error(error, rc);
    return m;
}

/* ------------------------------------------------------------------ contrasts */
static int32_t links_fill(const tr_multi_hier *h, int32_t level, uint32_t *map, uint32_t *next) {
    tr_bank_states ch;
    tr_bank_get_states(h->bank[level], &ch);
    const tr_multi_map *m = h->map[level];
    uint32_t *last = zalloc(m->ncoarse * 4);
    if (!last) return TR_MATH_MEMORY;
    for (uint64_t q = 0; q < m->ncoarse; q++) last[q] = TR_MULTI_UNMAPPED;
    for (uint64_t k = ch.ngroups; k-- > 0;) {
        uint32_t p = m->map[k];
        if (!ch.rows[k] || p == TR_MULTI_UNMAPPED) { map[k] = TR_MULTI_UNMAPPED; next[k] = TR_MULTI_UNMAPPED; continue; }
        map[k] = p; next[k] = last[p]; last[p] = (uint32_t)k;
    }
    free(last);
    return TR_MATH_OK;
}
int32_t tr_multi_contrasts_need(const tr_multi_hier *h, int32_t level, int32_t nsel, uint64_t count, int32_t target,
                                tr_math_need *need) {
    if (!hier_valid(h) || !need || level < 0 || level + 1 >= h->levels || nsel < 1 || nsel > h->nterms ||
        (target != 0 && target != 1)) return TR_MATH_INVALID;
    tr_bank_info c, p;
    tr_bank_get_info(h->bank[level], &c);
    tr_bank_get_info(h->bank[level + 1], &p);
    if (target == 1) { tr_multi_contrasts_device_need(c.ngroups, p.ngroups, h->nterms, nsel, count, need); return TR_MATH_OK; }
    need_zero(need);
    need->need[TR_BUDGET_SUPPORT] = (uint64_t)nsel;
    need->need[TR_BUDGET_OUTPUT] = tr_budget_mul(count, (uint64_t)nsel);
    need->need[TR_BUDGET_BYTES] = tr_budget_add(tr_budget_mul(c.ngroups, 8), tr_budget_mul(p.ngroups, 4));
    return TR_MATH_OK;
}
int32_t tr_multi_hier_links(const tr_multi_hier *h, int32_t level, uint32_t *map, uint32_t *next, uint64_t capacity,
                            const tr_math_budget *budget, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!hier_valid(h) || level < 0 || level + 1 >= h->levels || !map || !next) return TR_MATH_INVALID;
    tr_bank_info c, p;
    tr_bank_get_info(h->bank[level], &c);
    tr_bank_get_info(h->bank[level + 1], &p);
    if (capacity < c.ngroups) return TR_MATH_INVALID;
    tr_math_budget dflt;
    int32_t rc = tr_budget_check_one(budget_or(budget, &dflt), TR_BUDGET_BYTES, tr_budget_mul(p.ngroups, 4), refusal);
    if (rc) return rc;
    return links_fill(h, level, map, next);
}
int32_t tr_multi_contrasts(const tr_multi_hier *h, int32_t level, int32_t kind, const int32_t *terms, int32_t nsel,
                           uint64_t first, uint64_t count, const tr_math_budget *budget, tr_multi_q *out,
                           tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!hier_valid(h) || level < 0 || level + 1 >= h->levels || !tr_multi_kind_ok(kind) || nsel < 1 ||
        nsel > h->nterms || !terms || (count && !out)) return TR_MATH_INVALID;
    for (int32_t a = 0; a < nsel; a++) {
        if (terms[a] < 0 || terms[a] >= h->nterms) return TR_MATH_INVALID;
        for (int32_t b = 0; b < a; b++) if (terms[a] == terms[b]) return TR_MATH_INVALID;
    }
    tr_bank_states ch, pa;
    tr_bank_get_states(h->bank[level], &ch);
    tr_bank_get_states(h->bank[level + 1], &pa);
    if (first > ch.ngroups || count > ch.ngroups - first) return TR_MATH_INVALID;
    tr_math_need need;
    tr_multi_contrasts_need(h, level, nsel, count, 0, &need);
    tr_math_budget dflt;
    int32_t rc = tr_budget_check(budget_or(budget, &dflt), &need, refusal);
    if (rc) return rc;
    uint32_t *map = zalloc(ch.ngroups * 4), *next = zalloc(ch.ngroups * 4);
    if (!map || !next) { free(map); free(next); return TR_MATH_MEMORY; }
    rc = links_fill(h, level, map, next);
    uint64_t cells = count * (uint64_t)nsel;
    for (uint64_t i = 0; !rc && i < cells; i++)
        tr_multi_contrast_at(&ch, &pa, map, next, kind, terms, nsel, first, i, &out[i]);
    free(map); free(next);
    return rc;
}
