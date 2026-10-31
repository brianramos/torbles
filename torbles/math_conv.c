/* GEN-002: convolutions with explicit independence, conditional products, joint states and bounded series; see
 * math_conv.h. Built on GEN-001 (math_gen.h: the generating polynomials, their products, powers and requests),
 * DIST-001 (math_dist.h: histograms, rollups), FND-004 (math_scope.h: sources) and FND-009 (big integers, 192-bit sums).
 * SPDX-License-Identifier: Apache-2.0 */
#include "math_conv.h"
#include "mathematics.h"
#include "math_domain.h"
#include "math_handles.h"
#include <stdlib.h>
#include <string.h>

TR_MATH_ABI(tr_conv_info, TR_CONV_ABI);              /* the ctypes boundary of GEN-002 (math_handles.h) */
TR_MATH_ABI(tr_conv_check, TR_CONV_ABI);
TR_MATH_ABI(tr_conv_decl_info, TR_CONV_ABI);
TR_MATH_ABI(tr_conv_family_view, TR_CONV_ABI);
TR_MATH_ABI(tr_conv_family_info, TR_CONV_ABI);
TR_MATH_ABI(tr_conv_batch, TR_CONV_ABI);
TR_MATH_ABI(tr_conv_series_info, TR_CONV_ABI);
TR_MATH_ABI(tr_conv_walks, TR_CONV_ABI);

_Static_assert(sizeof(tr_conv_info) == 128, "tr_conv_info ABI");
_Static_assert(sizeof(tr_conv_check) == 32, "tr_conv_check ABI");
_Static_assert(sizeof(tr_conv_decl_info) == 24, "tr_conv_decl_info ABI");
_Static_assert(sizeof(tr_conv_family_view) == 64, "tr_conv_family_view ABI");
_Static_assert(sizeof(tr_conv_family_info) == 72, "tr_conv_family_info ABI");
_Static_assert(sizeof(tr_conv_batch) == 72, "tr_conv_batch ABI");
_Static_assert(sizeof(tr_conv_series_info) == 40, "tr_conv_series_info ABI");
_Static_assert(sizeof(tr_conv_walks) == 40, "tr_conv_walks ABI");

#define DECL_MAGIC UINT32_C(0x4c434544)
#define FACTOR_MAGIC UINT32_C(0x524f5446)
#define FAMILY_MAGIC UINT32_C(0x494d4146)
#define SERIES_MAGIC UINT32_C(0x49524553)

/* ------------------------------------------------------------------ sources */
typedef struct {
    int32_t kind, keyed;        /* TR_SCOPE_TABLE / _ARRAYS; the scope it was taken from had several groups */
    uint64_t len, digest[2];
    unsigned char *bytes;       /* table: address and identity token; arrays: the row descriptor */
} srckey;
typedef struct { uint32_t n; srckey k[TR_CONV_MAX_SOURCES]; } srcset;

struct tr_conv_decl_s { uint32_t magic; srcset src; };
struct tr_conv_factor_s { uint32_t magic; tr_gen_poly *poly; srcset src; tr_conv_info info; };
struct tr_conv_family_s {
    uint32_t magic;
    int32_t mass, scale;
    uint64_t keys, ncoef, maxspan, present, rows_total, nulls_total, bytes;
    uint64_t *offset, *rows, *nulls;
    int64_t *shift, *coef;
    unsigned char *grouping;    /* the scope's TR_SCOPE_KEY_GROUPS descriptor */
    uint64_t grouping_len;
    srckey src;
    int32_t col;
};
struct tr_conv_series_s { uint32_t magic; uint32_t order; int64_t bound, maxabs, shift; int64_t *coef; srcset src; };

static void set_error(int32_t *error, int32_t code) { if (error) *error = code; }
static void *zalloc(uint64_t bytes) {
    if (bytes >= (uint64_t)SIZE_MAX) return NULL;
    return calloc(1, bytes ? (size_t)bytes : 1);
}
static const tr_math_budget *budget_of(const tr_math_budget *b, tr_math_budget *dflt) {
    if (b) return b;
    tr_budget_default(dflt);
    return dflt;
}
static void need_zero(tr_math_need *n) { for (int i = 0; i < TR_BUDGET_RESOURCES; i++) n->need[i] = 0; }
/* tr_budget_check, refusing a size past 64 bits even under an unlimited budget */
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
static int32_t check_bytes(const tr_math_budget *b, uint64_t bytes, tr_math_refusal *r) {
    tr_math_need n;
    need_zero(&n);
    n.need[TR_BUDGET_BYTES] = bytes;
    return check_need(b, &n, r);
}

static void digest_of(int32_t kind, const unsigned char *p, uint64_t n, uint64_t *d) {
    uint64_t h1 = UINT64_C(14695981039346656037) ^ (uint64_t)kind, h2 = UINT64_C(0x9e3779b97f4a7c15) + (uint64_t)kind;
    for (uint64_t i = 0; i < n; i++) {
        h1 = (h1 ^ p[i]) * UINT64_C(1099511628211);
        h2 = (h2 ^ p[i]) * UINT64_C(0xff51afd7ed558ccd);
        h2 ^= h2 >> 29;
    }
    h1 ^= n; h2 ^= n * UINT64_C(0xc4ceb9fe1a85ec53);
    d[0] = h1; d[1] = h2;
}
static void key_release(srckey *k) { free(k->bytes); memset(k, 0, sizeof *k); }
static void set_release(srcset *s) { for (uint32_t i = 0; i < s->n; i++) key_release(&s->k[i]); s->n = 0; }
static int key_equal(const srckey *a, const srckey *b) {
    return a->kind == b->kind && a->len == b->len && (!a->len || memcmp(a->bytes, b->bytes, (size_t)a->len) == 0);
}
static int32_t key_copy(const srckey *from, srckey *to) {
    *to = *from;
    to->bytes = zalloc(from->len);
    if (!to->bytes) { memset(to, 0, sizeof *to); return TR_MATH_MEMORY; }
    if (from->len) memcpy(to->bytes, from->bytes, (size_t)from->len);
    return TR_MATH_OK;
}
/* the bytes a scope's source key takes */
static uint64_t key_len(const tr_scope *s, tr_scope_info *info) {
    if (tr_scope_get_info(s, info)) return TR_BUDGET_UNKNOWN;
    return info->source == TR_SCOPE_TABLE ? 24 : tr_scope_key_size(s, TR_SCOPE_KEY_ROWS);
}
static int32_t key_of_scope(const tr_scope *s, const tr_math_budget *b, tr_math_refusal *r, srckey *k) {
    memset(k, 0, sizeof *k);
    tr_scope_info info;
    uint64_t len = key_len(s, &info);
    if (len == TR_BUDGET_UNKNOWN) return TR_MATH_INVALID;
    if (b) {                                                  /* NULL: the caller counted these bytes */
        int32_t rc = check_bytes(b, tr_budget_add(len, sizeof *k), r);
        if (rc) return rc;
    }
    k->bytes = zalloc(len);
    if (!k->bytes) return TR_MATH_MEMORY;
    k->kind = info.source; k->keyed = info.ngroups > 1; k->len = len;
    if (info.source == TR_SCOPE_TABLE) {
        memcpy(k->bytes, &info.table, 8);
        memcpy(k->bytes + 8, info.identity, 16);
    } else if (tr_scope_key_write(s, TR_SCOPE_KEY_ROWS, k->bytes, len)) {
        key_release(k);
        return TR_MATH_INVALID;
    }
    digest_of(k->kind, k->bytes, k->len, k->digest);
    return TR_MATH_OK;
}
static int32_t batch_compute(const tr_conv_batch *b, int64_t *out);
static int set_find(const srcset *s, const srckey *k) {
    for (uint32_t i = 0; i < s->n; i++) if (key_equal(&s->k[i], k)) return (int)i;
    return -1;
}
/* into = into + from (deduplicated; a merged key is keyed when either is); TR_MATH_UNSUPPORTED past the limit */
static int32_t set_merge(srcset *into, const srcset *from) {
    for (uint32_t i = 0; i < from->n; i++) {
        int at = set_find(into, &from->k[i]);
        if (at >= 0) { into->k[at].keyed |= from->k[i].keyed; continue; }
        if (into->n >= TR_CONV_MAX_SOURCES) return TR_MATH_UNSUPPORTED;
        int32_t rc = key_copy(&from->k[i], &into->k[into->n]);
        if (rc) return rc;
        into->n++;
    }
    return TR_MATH_OK;
}

/* ------------------------------------------------------------------ declarations */
static int decl_valid(const tr_conv_decl *d) { return d && d->magic == DECL_MAGIC; }

tr_conv_decl *tr_conv_declare(const tr_scope *const *scopes, uint32_t n, const tr_math_budget *budget,
                              tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    if (!scopes || n < 1) { set_error(error, TR_MATH_INVALID); return NULL; }
    tr_math_budget dflt;
    const tr_math_budget *bud = budget_of(budget, &dflt);
    uint64_t bytes = sizeof(tr_conv_decl);
    tr_scope_info info;
    for (uint32_t i = 0; i < n; i++) {
        if (!scopes[i]) { set_error(error, TR_MATH_INVALID); return NULL; }
        uint64_t len = key_len(scopes[i], &info);
        if (len == TR_BUDGET_UNKNOWN) { set_error(error, TR_MATH_INVALID); return NULL; }
        bytes = tr_budget_add(bytes, len);
    }
    int32_t rc = check_bytes(bud, bytes, refusal);
    if (rc) { set_error(error, rc); return NULL; }
    tr_conv_decl *d = zalloc(sizeof *d);
    if (!d) { set_error(error, TR_MATH_MEMORY); return NULL; }
    for (uint32_t i = 0; i < n && !rc; i++) {
        srckey k;
        rc = key_of_scope(scopes[i], NULL, NULL, &k);   /* bytes counted above */
        if (rc) break;
        srcset one; one.n = 1; one.k[0] = k;
        rc = set_merge(&d->src, &one);
        key_release(&k);
    }
    if (rc) { set_release(&d->src); free(d); set_error(error, rc); return NULL; }
    d->magic = DECL_MAGIC;
    set_error(error, TR_MATH_OK);
    return d;
}
int32_t tr_conv_decl_get_info(const tr_conv_decl *d, tr_conv_decl_info *out) {
    if (!decl_valid(d) || !out) return TR_MATH_INVALID;
    memset(out, 0, sizeof *out);
    out->sources = d->src.n;
    if (d->src.n) { out->digest[0] = d->src.k[0].digest[0]; out->digest[1] = d->src.k[0].digest[1]; }
    return TR_MATH_OK;
}
int32_t tr_conv_decl_covers(const tr_conv_decl *d, const tr_scope *s) {
    if (!decl_valid(d) || !s) return TR_MATH_INVALID;
    srckey k;
    int32_t rc = key_of_scope(s, NULL, NULL, &k);
    if (rc) return rc;
    int found = set_find(&d->src, &k) >= 0;
    key_release(&k);
    return found;
}
void tr_conv_decl_free(tr_conv_decl *d) {
    if (!decl_valid(d)) return;
    set_release(&d->src);
    d->magic = 0;
    free(d);
}
static int decl_has(const tr_conv_decl *d, const srckey *k) { return decl_valid(d) && set_find(&d->src, k) >= 0; }

/* ------------------------------------------------------------------ factors */
static int factor_valid(const tr_conv_factor *f) { return f && f->magic == FACTOR_MAGIC && f->poly; }
static void factor_release(tr_conv_factor *f) {
    if (!f) return;
    tr_gen_free(f->poly);
    set_release(&f->src);
    f->magic = 0;
    free(f);
}
void tr_conv_factor_free(tr_conv_factor *f) { if (f && f->magic == FACTOR_MAGIC) factor_release(f); }

/* a factor around an owned polynomial (consumed, also on failure) with a copy of src (may be NULL) */
static tr_conv_factor *factor_make(tr_gen_poly *poly, const srcset *src, int32_t kind, int32_t mode, int32_t *error) {
    if (!poly) return NULL;
    tr_conv_factor *f = zalloc(sizeof *f);
    if (!f) { tr_gen_free(poly); set_error(error, TR_MATH_MEMORY); return NULL; }
    f->poly = poly;
    int32_t rc = src ? set_merge(&f->src, src) : TR_MATH_OK;
    if (rc) { factor_release(f); set_error(error, rc); return NULL; }
    tr_gen_info gi;
    tr_gen_get_info(poly, &gi);
    tr_conv_info *i = &f->info;
    i->kind = kind; i->mode = mode; i->nvars = gi.nvars; i->mass = gi.mass; i->combine = 0;
    i->keys = 1; i->key = TR_CONV_ALL; i->draws = 1; i->col[0] = i->col[1] = -1;
    i->scale[0] = gi.scale[0]; i->scale[1] = gi.scale[1];
    i->sources = f->src.n;
    for (uint32_t s = 0; s < f->src.n; s++) i->keyed |= f->src.k[s].keyed;
    if (f->src.n) { i->digest[0] = f->src.k[0].digest[0]; i->digest[1] = f->src.k[0].digest[1]; }
    f->magic = FACTOR_MAGIC;
    set_error(error, TR_MATH_OK);
    return f;
}

/* a copy of p with GEN-001 scale 0 (the factor keeps the scales): its terms through tr_gen_from_terms */
static tr_gen_poly *poly_terms(const tr_gen_poly *p, int coordinate_of_1d, int32_t repr, const tr_math_budget *b,
                               tr_math_refusal *r, int32_t *error);

tr_conv_factor *tr_conv_factor_from_hist(const tr_dist_hist *h, const tr_scope *scope, uint64_t key, int32_t mass,
                                         int32_t repr, const int64_t *shift, const tr_math_budget *budget,
                                         tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    tr_dist_hist_info hi;
    tr_scope_info si;
    if (!h || !scope || tr_dist_hist_get_info(h, &hi) || tr_scope_get_info(scope, &si) || !hi.scoped ||
        hi.digest[0] != si.digest[0] || hi.digest[1] != si.digest[1] || hi.ngroups != si.ngroups ||
        hi.rows != si.selected || (key != TR_CONV_ALL && key >= hi.ngroups)) { set_error(error, TR_MATH_INVALID); return NULL; }
    if (hi.k > 2) { set_error(error, TR_MATH_UNSUPPORTED); return NULL; }
    tr_math_budget dflt;
    const tr_math_budget *bud = budget_of(budget, &dflt);
    srckey k;
    int32_t rc = key_of_scope(scope, bud, refusal, &k);
    if (rc) { set_error(error, rc); return NULL; }
    const tr_dist_hist *src = h;
    tr_dist_hist *rolled = NULL;
    uint64_t g = key == TR_CONV_ALL ? 0 : key;
    if (key == TR_CONV_ALL && hi.ngroups > 1) {             /* the marginal over every key: DIST-001's rollup */
        rc = check_bytes(bud, tr_budget_mul(hi.ngroups, 4), refusal);
        uint32_t *map = rc ? NULL : zalloc(tr_budget_mul(hi.ngroups, 4));
        if (!rc && !map) rc = TR_MATH_MEMORY;
        if (!rc) rolled = tr_dist_hist_rollup(h, map, 1, NULL, bud, refusal, &rc);
        free(map);
        src = rolled;
    }
    tr_gen_poly *p = rc ? NULL : tr_gen_from_hist(src, g, mass, repr, shift, bud, refusal, &rc);
    tr_dist_hist_free(rolled);
    if (!p) { key_release(&k); set_error(error, rc ? rc : TR_MATH_MEMORY); return NULL; }
    tr_gen_info gi;
    tr_gen_get_info(p, &gi);
    if (gi.scale[0] || gi.scale[1]) {                       /* GEN-002 keeps the scales in the factor */
        tr_gen_poly *q = poly_terms(p, 0, repr, bud, refusal, &rc);
        tr_gen_free(p);
        p = q;
        if (!p) { key_release(&k); set_error(error, rc); return NULL; }
    }
    srcset one;
    one.n = 1; one.k[0] = k;
    tr_conv_factor *f = factor_make(p, &one, TR_CONV_DATA, TR_CONV_MODE_BASE, error);
    key_release(&k);
    if (!f) return NULL;
    f->info.keys = hi.ngroups; f->info.key = key; f->info.keyed = hi.ngroups > 1;
    f->info.tuples = gi.rows; f->info.nulls = gi.nulls; f->info.counted = mass == TR_DIST_MASS_COUNT;
    f->info.mass = mass;
    for (int c = 0; c < 2; c++) { f->info.col[c] = c < hi.k ? hi.col[c] : -1; f->info.scale[c] = gi.scale[c]; }
    return f;
}

tr_conv_factor *tr_conv_factor_from_terms(int32_t nvars, uint64_t n, const int64_t *values, const tr_math_i128 *coef,
                                          int32_t repr, const int64_t *shift, const tr_math_budget *budget,
                                          tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    tr_gen_poly *p = tr_gen_from_terms(nvars, n, values, coef, repr, shift, budget, refusal, error);
    return p ? factor_make(p, NULL, TR_CONV_FREE, TR_CONV_MODE_BASE, error) : NULL;
}

tr_conv_factor *tr_conv_factor_from_gen(const tr_gen_poly *src, const tr_math_budget *budget, tr_math_refusal *refusal,
                                        int32_t *error) {
    tr_budget_none(refusal);
    tr_gen_info gi;
    if (!src || tr_gen_get_info(src, &gi)) { set_error(error, TR_MATH_INVALID); return NULL; }
    tr_math_budget dflt;
    const tr_math_budget *bud = budget_of(budget, &dflt);
    int32_t rc = TR_MATH_OK;
    tr_gen_poly *p = (gi.scale[0] || gi.scale[1]) ? poly_terms(src, 0, gi.repr, bud, refusal, &rc)
                                                  : tr_gen_convert(src, gi.repr, bud, refusal, &rc);
    if (!p) { set_error(error, rc ? rc : TR_MATH_MEMORY); return NULL; }
    tr_conv_factor *f = factor_make(p, NULL, TR_CONV_FREE, TR_CONV_MODE_BASE, error);
    if (f) { f->info.scale[0] = gi.scale[0]; f->info.scale[1] = gi.scale[1]; }
    return f;
}

int32_t tr_conv_factor_get_info(const tr_conv_factor *f, tr_conv_info *out) {
    if (!factor_valid(f) || !out) return TR_MATH_INVALID;
    *out = f->info;
    return TR_MATH_OK;
}
const tr_gen_poly *tr_conv_factor_poly(const tr_conv_factor *f) { return factor_valid(f) ? f->poly : NULL; }
tr_gen_poly *tr_conv_factor_poly_copy(const tr_conv_factor *f, int32_t repr, const tr_math_budget *budget,
                                      tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    if (!factor_valid(f)) { set_error(error, TR_MATH_INVALID); return NULL; }
    return tr_gen_convert(f->poly, repr, budget, refusal, error);
}
int32_t tr_conv_factor_source(const tr_conv_factor *f, uint64_t i, uint64_t *digest, int32_t *keyed) {
    if (!factor_valid(f) || i >= f->src.n || !digest || !keyed) return TR_MATH_INVALID;
    digest[0] = f->src.k[i].digest[0]; digest[1] = f->src.k[i].digest[1]; *keyed = f->src.k[i].keyed;
    return TR_MATH_OK;
}

/* ------------------------------------------------------------------ terms of a GEN-001 polynomial */
typedef struct { uint64_t n, nl; int64_t *values; int32_t *signs; uint64_t *loff, *limbs; } exported;
static void exported_free(exported *e) { free(e->values); free(e->signs); free(e->loff); free(e->limbs); memset(e, 0, sizeof *e); }
static uint64_t export_bytes(uint64_t n, uint64_t nl, int nvars) {
    return tr_budget_add(tr_budget_mul(n, 8 * (uint64_t)nvars + 4 + 8), tr_budget_mul(tr_budget_add(nl, 1), 8));
}
static int32_t export_terms(const tr_gen_poly *p, int nvars, const tr_math_budget *b, tr_math_refusal *r, exported *e) {
    memset(e, 0, sizeof *e);
    int32_t rc = tr_gen_export_size(p, &e->n, &e->nl);
    if (rc) return rc;
    rc = check_bytes(b, export_bytes(e->n, e->nl, nvars), r);
    if (rc) return rc;
    e->values = zalloc(tr_budget_mul(e->n, 8 * (uint64_t)nvars));
    e->signs = zalloc(tr_budget_mul(e->n, 4));
    e->loff = zalloc(tr_budget_mul(e->n + 1, 8));
    e->limbs = zalloc(tr_budget_mul(e->nl, 8));
    if (!e->values || !e->signs || !e->loff || !e->limbs) { exported_free(e); return TR_MATH_MEMORY; }
    rc = tr_gen_export(p, e->values, e->signs, e->loff, e->limbs, e->n, e->nl);
    if (rc) exported_free(e);
    return rc;
}
/* term i's coefficient as an int128: 1, or 0 when it does not fit */
static int exported_i128(const exported *e, uint64_t i, __int128 *v) {
    uint64_t a = e->loff[i], n = e->loff[i + 1] - a;
    if (n > 2) return 0;
    unsigned __int128 m = n == 0 ? 0 : n == 1 ? (unsigned __int128)e->limbs[a]
                                              : ((unsigned __int128)e->limbs[a + 1] << 64) | e->limbs[a];
    unsigned __int128 lim = (unsigned __int128)1 << 127;
    if (e->signs[i] < 0 ? m > lim : m >= lim) return 0;
    *v = e->signs[i] < 0 ? (__int128)((unsigned __int128)0 - m) : (__int128)m;
    return 1;
}
/* one-variable p embedded at coordinate c of two variables (coordinate_of_1d = c + 1), or (0) a copy of the same
 * variables, through tr_gen_from_terms: every coefficient must fit int128 (TR_MATH_OVERFLOW) */
static tr_gen_poly *poly_terms(const tr_gen_poly *p, int coordinate_of_1d, int32_t repr, const tr_math_budget *b,
                               tr_math_refusal *r, int32_t *error) {
    tr_gen_info gi;
    tr_gen_get_info(p, &gi);
    const int nv = gi.nvars, outv = coordinate_of_1d ? 2 : nv;
    exported e;
    int32_t rc = export_terms(p, nv, b, r, &e);
    if (rc) { set_error(error, rc); return NULL; }
    rc = check_bytes(b, tr_budget_add(export_bytes(e.n, e.nl, nv), tr_budget_mul(e.n, 8 * (uint64_t)outv + 16)), r);
    int64_t *vals = rc ? NULL : zalloc(tr_budget_mul(e.n, 8 * (uint64_t)outv));
    tr_math_i128 *coef = rc ? NULL : zalloc(tr_budget_mul(e.n, 16));
    if (!rc && (!vals || !coef)) rc = TR_MATH_MEMORY;
    for (uint64_t i = 0; i < e.n && !rc; i++) {
        __int128 v;
        if (!exported_i128(&e, i, &v)) { rc = TR_MATH_OVERFLOW; break; }
        coef[i] = tr_i128_set(v);
        if (coordinate_of_1d) {
            vals[2 * i] = coordinate_of_1d == 1 ? e.values[i] : 0;
            vals[2 * i + 1] = coordinate_of_1d == 2 ? e.values[i] : 0;
        } else for (int c = 0; c < nv; c++) vals[i * (uint64_t)nv + (uint64_t)c] = e.values[i * (uint64_t)nv + (uint64_t)c];
    }
    tr_gen_poly *out = NULL;
    if (!rc) {
        int64_t shift[2] = {gi.shift[0], gi.shift[1]};
        if (coordinate_of_1d == 1) shift[1] = 0;
        if (coordinate_of_1d == 2) { shift[1] = gi.shift[0]; shift[0] = 0; }
        out = tr_gen_from_terms(outv, e.n, vals, coef, repr, shift, b, r, &rc);
    }
    exported_free(&e); free(vals); free(coef);
    set_error(error, rc);
    return rc ? (tr_gen_free(out), (tr_gen_poly *)NULL) : out;
}

/* ------------------------------------------------------------------ the independence check */
static void check_fill(tr_conv_check *ck, int32_t reason, int32_t i, int32_t j, const srckey *k, int keyed) {
    if (!ck) return;
    ck->reason = reason; ck->first = i; ck->second = j; ck->keyed = keyed;
    ck->digest[0] = k ? k->digest[0] : 0; ck->digest[1] = k ? k->digest[1] : 0;
}
static void check_none(tr_conv_check *ck) { check_fill(ck, TR_CONV_OK, -1, -1, NULL, 0); }
/* the linkage rule over source sets s[0..m) with multiplicities k: 0 or TR_MATH_DOMAIN_ERROR */
static int32_t linked(const srcset *const *s, const uint32_t *k, uint32_t m, const tr_conv_decl *d, tr_conv_check *ck) {
    check_none(ck);
    for (uint32_t i = 0; i < m; i++) {
        uint32_t ki = k ? k[i] : 1;
        if (!ki) continue;
        if (ki >= 2)
            for (uint32_t a = 0; a < s[i]->n; a++)
                if (!decl_has(d, &s[i]->k[a])) {
                    check_fill(ck, d ? TR_CONV_NOT_COVERED : TR_CONV_NEEDS_DECLARATION, (int32_t)i, (int32_t)i,
                               &s[i]->k[a], s[i]->k[a].keyed);
                    return TR_MATH_DOMAIN_ERROR;
                }
        for (uint32_t j = i + 1; j < m; j++) {
            if (k && !k[j]) continue;
            for (uint32_t a = 0; a < s[i]->n; a++) {
                int at = set_find(s[j], &s[i]->k[a]);
                if (at < 0 || decl_has(d, &s[i]->k[a])) continue;
                check_fill(ck, d ? TR_CONV_NOT_COVERED : TR_CONV_NEEDS_DECLARATION, (int32_t)i, (int32_t)j,
                           &s[i]->k[a], s[i]->k[a].keyed | s[j]->k[at].keyed);
                return TR_MATH_DOMAIN_ERROR;
            }
        }
    }
    return TR_MATH_OK;
}
/* 1 when the product needs the declaration (some copies are linked) */
static int any_linked(const srcset *const *s, const uint32_t *k, uint32_t m) {
    for (uint32_t i = 0; i < m; i++) {
        uint32_t ki = k ? k[i] : 1;
        if (!ki) continue;
        if (ki >= 2 && s[i]->n) return 1;
        for (uint32_t j = i + 1; j < m; j++)
            if (!k || k[j])
                for (uint32_t a = 0; a < s[i]->n; a++) if (set_find(s[j], &s[i]->k[a]) >= 0) return 1;
    }
    return 0;
}

int32_t tr_conv_check_product(const tr_conv_factor *const *f, const uint32_t *mult, uint32_t m,
                              const tr_conv_decl *decl, tr_conv_check *check) {
    if (!f || m < 1 || m > TR_CONV_MAX_FACTORS || (decl && !decl_valid(decl))) return TR_MATH_INVALID;
    const srcset *s[TR_CONV_MAX_FACTORS];
    for (uint32_t i = 0; i < m; i++) { if (!factor_valid(f[i])) return TR_MATH_INVALID; s[i] = &f[i]->src; }
    return linked(s, mult, m, decl, check);
}

/* ------------------------------------------------------------------ products */
typedef struct { uint64_t n, span[2]; int dense; } csize;
static csize size_of(const tr_conv_factor *f, uint64_t order) {
    tr_gen_info gi;
    tr_gen_get_info(f->poly, &gi);
    csize s;
    s.n = gi.nonzero; s.dense = gi.repr == TR_GEN_DENSE;
    s.span[0] = gi.span[0]; s.span[1] = gi.nvars > 1 ? gi.span[1] : 1;
    if (order && s.span[0] > order) s.span[0] = order;
    return s;
}
/* one step a x b: its pairs and result bound (spans add, capped by order) */
static csize step_of(int nv, csize a, csize b, uint64_t order, uint64_t *pairs) {
    csize r;
    r.dense = a.dense && b.dense;
    for (int c = 0; c < 2; c++)
        r.span[c] = c < nv ? (a.span[c] && b.span[c] ? tr_budget_add(a.span[c], b.span[c] - 1) : 0) : 1;
    if (order && r.span[0] > order) r.span[0] = order;
    uint64_t box = nv > 1 ? tr_budget_mul(r.span[0], r.span[1]) : r.span[0];
    *pairs = tr_budget_mul(a.n, b.n);
    r.n = r.dense ? box : (box < *pairs ? box : *pairs);
    return r;
}
static int32_t product_validate(const tr_conv_factor *const *f, const uint32_t *mult, uint32_t m, int32_t combine,
                                uint64_t order, int *nvars) {
    if (!f || m < 1 || (combine != TR_CONV_SUM && combine != TR_CONV_OUTER)) return TR_MATH_INVALID;
    if (m > TR_CONV_MAX_FACTORS) return TR_MATH_UNSUPPORTED;
    for (uint32_t i = 0; i < m; i++) if (!factor_valid(f[i])) return TR_MATH_INVALID;
    const int nv = f[0]->info.nvars;
    if (combine == TR_CONV_OUTER) {
        if (m != 2 || f[0]->info.nvars != 1 || f[1]->info.nvars != 1) return TR_MATH_UNSUPPORTED;
        if (order) return TR_MATH_UNSUPPORTED;
        *nvars = 2;
        return TR_MATH_OK;
    }
    for (uint32_t i = 1; i < m; i++) if (f[i]->info.nvars != nv) return TR_MATH_INVALID;
    if (order && nv != 1) return TR_MATH_UNSUPPORTED;
    for (uint32_t i = 1; i < m; i++)
        if (f[i]->info.scale[0] != f[0]->info.scale[0] || (nv > 1 && f[i]->info.scale[1] != f[0]->info.scale[1]))
            return TR_MATH_DOMAIN_ERROR;
    (void)mult;
    *nvars = nv;
    return TR_MATH_OK;
}

int32_t tr_conv_product_need(const tr_conv_factor *const *f, const uint32_t *mult, uint32_t m, int32_t combine,
                             uint64_t order, const tr_math_budget *budget, tr_math_need *need, uint64_t *terms) {
    int nv = 1;
    int32_t rc = product_validate(f, mult, m, combine, order, &nv);
    if (rc || !need) return rc ? rc : TR_MATH_INVALID;
    tr_math_budget dflt;
    const tr_math_budget *bud = budget_of(budget, &dflt);
    uint32_t live = 0, only = 0;
    for (uint32_t i = 0; i < m; i++) if (!mult || mult[i]) { live++; only = i; }
    if (combine == TR_CONV_SUM && live == 1)                  /* a power: GEN-001's exact chain */
        return tr_gen_pow_need(f[only]->poly, mult ? mult[only] : 1, order, bud, need, terms);
    if (combine == TR_CONV_SUM && m == 2 && (!mult || (mult[0] == 1 && mult[1] == 1)))
        return tr_gen_mul_need(f[0]->poly, f[1]->poly, order, bud, need, terms);
    need_zero(need);
    need->need[TR_BUDGET_VARIABLES] = (uint64_t)nv;
    csize cur = {0, {0, 0}, 0};
    int have = 0;
    const int sv = combine == TR_CONV_OUTER ? 1 : nv;          /* the powers are one-variable for OUTER */
    for (uint32_t i = 0; i < m; i++) {
        uint32_t k = mult ? mult[i] : 1;
        if (!k) continue;
        csize base = size_of(f[i], order), p = base;
        for (uint32_t j = 2; j <= k; j++) {
            uint64_t pairs;
            p = step_of(sv, p, base, order, &pairs);
            if (pairs > need->need[TR_BUDGET_TENSOR]) need->need[TR_BUDGET_TENSOR] = pairs;
            if (p.n > need->need[TR_BUDGET_SUPPORT]) need->need[TR_BUDGET_SUPPORT] = p.n;
        }
        if (combine == TR_CONV_OUTER) {                        /* embedded at coordinate (number of earlier) */
            csize e = p;
            if (!have) { e.span[1] = 1; cur = e; have = 1; continue; }
            e.span[1] = p.span[0]; e.span[0] = 1;
            uint64_t pairs;
            cur = step_of(2, cur, e, 0, &pairs);
            if (pairs > need->need[TR_BUDGET_TENSOR]) need->need[TR_BUDGET_TENSOR] = pairs;
        } else if (!have) { cur = p; have = 1; continue; }
        else {
            uint64_t pairs;
            cur = step_of(nv, cur, p, order, &pairs);
            if (pairs > need->need[TR_BUDGET_TENSOR]) need->need[TR_BUDGET_TENSOR] = pairs;
        }
        if (cur.n > need->need[TR_BUDGET_SUPPORT]) need->need[TR_BUDGET_SUPPORT] = cur.n;
    }
    if (!have) cur.n = 1;                                       /* every multiplicity 0: the constant 1 */
    if (cur.n > need->need[TR_BUDGET_SUPPORT]) need->need[TR_BUDGET_SUPPORT] = cur.n;
    if (terms) *terms = cur.n;
    return TR_MATH_OK;
}

/* tuples / nulls of a product (counted factors only; saturation drops the count) */
static void product_counts(const tr_conv_factor *const *f, const uint32_t *mult, uint32_t m, tr_conv_info *out) {
    uint64_t t = 1, p = 1;
    int counted = 1;
    for (uint32_t i = 0; i < m && counted; i++) {
        uint32_t k = mult ? mult[i] : 1;
        if (!k) continue;
        if (!f[i]->info.counted) { counted = 0; break; }
        for (uint32_t j = 0; j < k; j++) {
            t = tr_budget_mul(t, f[i]->info.tuples);
            p = tr_budget_mul(p, f[i]->info.tuples - f[i]->info.nulls);
        }
        if (t == TR_BUDGET_UNKNOWN) counted = 0;
    }
    out->counted = counted;
    out->tuples = counted ? t : 0;
    out->nulls = counted ? t - p : 0;
}

tr_conv_factor *tr_conv_product(const tr_conv_factor *const *f, const uint32_t *mult, uint32_t m, int32_t combine,
                                uint64_t order, const tr_conv_decl *decl, const tr_math_budget *budget,
                                tr_math_refusal *refusal, tr_conv_check *check, int32_t *error) {
    tr_budget_none(refusal);
    check_none(check);
    int nv = 1;
    int32_t rc = product_validate(f, mult, m, combine, order, &nv);
    if (rc == TR_MATH_DOMAIN_ERROR) check_fill(check, TR_CONV_SCALES_DIFFER, 0, -1, NULL, 0);
    if (!rc && decl && !decl_valid(decl)) rc = TR_MATH_INVALID;
    if (!rc) rc = tr_conv_check_product(f, mult, m, decl, check);
    if (rc) { set_error(error, rc); return NULL; }
    const srcset *sets[TR_CONV_MAX_FACTORS];
    for (uint32_t i = 0; i < m; i++) sets[i] = &f[i]->src;
    srcset all;
    memset(&all, 0, sizeof all);
    for (uint32_t i = 0; i < m && !rc; i++) if (!mult || mult[i]) rc = set_merge(&all, sets[i]);
    tr_math_budget dflt;
    const tr_math_budget *bud = budget_of(budget, &dflt);
    tr_math_need need;
    if (!rc) rc = tr_conv_product_need(f, mult, m, combine, order, bud, &need, NULL);
    if (!rc) {                                                  /* bytes are checked by each GEN-001 step */
        uint64_t keep = need.need[TR_BUDGET_BYTES];
        if (combine == TR_CONV_OUTER || m > 2) need.need[TR_BUDGET_BYTES] = 0;
        rc = check_need(bud, &need, refusal);
        need.need[TR_BUDGET_BYTES] = keep;
    }
    if (rc) { set_release(&all); set_error(error, rc); return NULL; }
    const tr_gen_poly *cur = NULL;                              /* the running product: borrowed, or `acc` */
    tr_gen_poly *acc = NULL;
    int nlive = 0;
    for (uint32_t i = 0; i < m && !rc; i++) {
        uint32_t k = mult ? mult[i] : 1;
        if (!k) continue;
        tr_gen_poly *pw = NULL;
        const tr_gen_poly *term = f[i]->poly;
        if (k >= 2 || (order && combine == TR_CONV_SUM)) {
            pw = tr_gen_pow(f[i]->poly, k, combine == TR_CONV_SUM ? order : 0, bud, refusal, &rc);
            if (!pw) break;
            term = pw;
        }
        if (combine == TR_CONV_OUTER) {
            tr_gen_poly *e = poly_terms(term, nlive ? 2 : 1, TR_GEN_SPARSE, bud, refusal, &rc);
            tr_gen_free(pw);
            if (!e) break;
            pw = e; term = e;
        }
        nlive++;
        if (!cur) { cur = term; acc = pw; continue; }           /* the first operand is not copied */
        tr_gen_poly *next = tr_gen_mul(cur, term, combine == TR_CONV_SUM ? order : 0, bud, refusal, &rc);
        tr_gen_free(acc);
        tr_gen_free(pw);
        acc = next; cur = next;
        if (!next) break;
    }
    if (!rc && cur && !acc) {                                   /* one factor, once, complete: a copy */
        tr_gen_info gi;
        tr_gen_get_info(cur, &gi);
        acc = tr_gen_convert(cur, gi.repr, bud, refusal, &rc);
    }
    if (!rc && !acc) {                                          /* every multiplicity 0: the constant 1 */
        int64_t zero[2] = {0, 0};
        tr_math_i128 one = tr_i128_set(1);
        acc = tr_gen_from_terms(nv, 1, zero, &one, TR_GEN_SPARSE, NULL, bud, refusal, &rc);
    }
    if (rc || !acc) { tr_gen_free(acc); set_release(&all); set_error(error, rc ? rc : TR_MATH_MEMORY); return NULL; }
    tr_conv_factor *r = factor_make(acc, &all, all.n ? TR_CONV_DERIVED : TR_CONV_FREE, TR_CONV_MODE_INDEPENDENT, error);
    int declared = any_linked(sets, mult, m);
    set_release(&all);
    if (!r) return NULL;
    r->info.combine = combine;
    r->info.declared = declared;
    uint64_t draws = 0;
    int32_t mass = -1;
    for (uint32_t i = 0; i < m; i++) {
        uint32_t k = mult ? mult[i] : 1;
        if (!k) continue;
        draws += k;
        mass = mass == -1 ? f[i]->info.mass : (mass == f[i]->info.mass ? mass : 0);
    }
    r->info.draws = draws;
    r->info.mass = mass < 0 ? 0 : mass;
    product_counts(f, mult, m, &r->info);
    if (combine == TR_CONV_OUTER) { r->info.scale[0] = f[0]->info.scale[0]; r->info.scale[1] = f[1]->info.scale[0]; }
    else { r->info.scale[0] = f[0]->info.scale[0]; r->info.scale[1] = nv > 1 ? f[0]->info.scale[1] : 0; }
    return r;
}

/* ------------------------------------------------------------------ joint states */
tr_conv_factor *tr_conv_joint_sum(const tr_conv_factor *f, const tr_math_budget *budget, tr_math_refusal *refusal,
                                  int32_t *error) {
    tr_budget_none(refusal);
    if (!factor_valid(f) || f->info.nvars != 2) { set_error(error, TR_MATH_INVALID); return NULL; }
    if (f->info.scale[0] != f->info.scale[1]) { set_error(error, TR_MATH_DOMAIN_ERROR); return NULL; }
    tr_math_budget dflt;
    const tr_math_budget *bud = budget_of(budget, &dflt);
    exported e;
    int32_t rc = export_terms(f->poly, 2, bud, refusal, &e);
    if (rc) { set_error(error, rc); return NULL; }
    rc = check_bytes(bud, tr_budget_add(export_bytes(e.n, e.nl, 2), tr_budget_mul(e.n, 24)), refusal);
    int64_t *vals = rc ? NULL : zalloc(tr_budget_mul(e.n, 8));
    tr_math_i128 *coef = rc ? NULL : zalloc(tr_budget_mul(e.n, 16));
    if (!rc && (!vals || !coef)) rc = TR_MATH_MEMORY;
    for (uint64_t i = 0; i < e.n && !rc; i++) {
        __int128 v, s = (__int128)e.values[2 * i] + e.values[2 * i + 1];
        if (s < INT64_MIN || s > INT64_MAX || !exported_i128(&e, i, &v)) { rc = TR_MATH_OVERFLOW; break; }
        vals[i] = (int64_t)s; coef[i] = tr_i128_set(v);
    }
    tr_gen_info gi;
    tr_gen_get_info(f->poly, &gi);
    tr_gen_poly *p = rc ? NULL : tr_gen_from_terms(1, e.n, vals, coef, gi.repr, NULL, bud, refusal, &rc);
    exported_free(&e); free(vals); free(coef);
    if (!p) { set_error(error, rc ? rc : TR_MATH_MEMORY); return NULL; }
    tr_conv_factor *r = factor_make(p, &f->src, f->src.n ? TR_CONV_DERIVED : TR_CONV_FREE, TR_CONV_MODE_JOINT, error);
    if (!r) return NULL;
    r->info.keys = f->info.keys; r->info.key = f->info.key;
    r->info.tuples = f->info.tuples; r->info.nulls = f->info.nulls; r->info.counted = f->info.counted;
    r->info.mass = f->info.mass; r->info.draws = f->info.draws; r->info.scale[0] = f->info.scale[0];
    return r;
}

tr_conv_factor *tr_conv_marginal(const tr_conv_factor *f, int32_t keep, const tr_math_budget *budget,
                                 tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    if (!factor_valid(f) || f->info.nvars != 2 || keep < 0 || keep > 1) { set_error(error, TR_MATH_INVALID); return NULL; }
    tr_gen_poly *p = tr_gen_marginal(f->poly, keep, budget, refusal, error);
    if (!p) return NULL;
    tr_conv_factor *r = factor_make(p, &f->src, f->info.kind, TR_CONV_MODE_MARGINAL, error);
    if (!r) return NULL;
    r->info.keys = f->info.keys; r->info.key = f->info.key;
    r->info.tuples = f->info.tuples; r->info.nulls = f->info.nulls; r->info.counted = f->info.counted;
    r->info.mass = f->info.mass; r->info.draws = f->info.draws;
    r->info.col[0] = f->info.col[keep]; r->info.col[1] = -1;
    r->info.scale[0] = f->info.scale[keep]; r->info.scale[1] = 0;
    return r;
}

/* ------------------------------------------------------------------ families */
static int family_valid(const tr_conv_family *f) { return f && f->magic == FAMILY_MAGIC; }
static void family_release(tr_conv_family *f) {
    if (!f) return;
    free(f->offset); free(f->rows); free(f->nulls); free(f->shift); free(f->coef); free(f->grouping);
    key_release(&f->src);
    f->magic = 0;
    free(f);
}
void tr_conv_family_free(tr_conv_family *f) { if (family_valid(f)) family_release(f); }

tr_conv_family *tr_conv_family_from_hist(const tr_dist_hist *h, const tr_scope *scope, int32_t mass,
                                         const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    tr_dist_hist_info hi;
    tr_scope_info si;
    tr_dist_states st;
    if (!h || !scope || tr_dist_hist_get_info(h, &hi) || tr_scope_get_info(scope, &si) || tr_dist_hist_get_states(h, &st) ||
        !hi.scoped || hi.digest[0] != si.digest[0] || hi.digest[1] != si.digest[1] || hi.ngroups != si.ngroups ||
        hi.rows != si.selected || (mass != TR_DIST_MASS_COUNT && mass != TR_DIST_MASS_WEIGHT) ||
        (mass == TR_DIST_MASS_WEIGHT && !hi.weighted)) { set_error(error, TR_MATH_INVALID); return NULL; }
    if (hi.k != 1) { set_error(error, TR_MATH_UNSUPPORTED); return NULL; }
    const uint64_t G = st.ngroups;
    tr_dist_plan plan;
    memset(&plan, 0, sizeof plan);
    if (st.repr == TR_DIST_DENSE) tr_dist_states_plan(&st, &plan);
    tr_math_budget dflt;
    const tr_math_budget *bud = budget_of(budget, &dflt);
    /* pass 1: per key the extremes of the values with a nonzero mass, its rows and NULL rows (scratch: 4 words a key) */
    int32_t rc = check_bytes(bud, tr_budget_mul(G, 32), refusal);
    if (rc) { set_error(error, rc); return NULL; }
    int64_t *lo = zalloc(tr_budget_mul(G, 8)), *hiv = zalloc(tr_budget_mul(G, 8));
    uint64_t *nulls = zalloc(tr_budget_mul(G, 8)), *any = zalloc(tr_budget_mul(G, 8));
    if (!lo || !hiv || !nulls || !any) rc = TR_MATH_MEMORY;
    for (uint64_t g = 0; g < G && !rc; g++) {
        uint64_t b = st.repr == TR_DIST_DENSE ? g * st.cells : st.offset[g];
        uint64_t e = st.repr == TR_DIST_DENSE ? b + st.cells : st.offset[g + 1];
        for (uint64_t i = b; i < e; i++) {
            if (!st.count[i]) continue;
            int nl;
            int64_t v = st.repr == TR_DIST_DENSE ? tr_dist_plan_coord(&plan, i - b, 0, &nl) : st.value[0][i];
            if (st.repr != TR_DIST_DENSE) nl = st.nulls[i] & 1;
            if (nl) { nulls[g] += st.count[i]; continue; }
            if (mass == TR_DIST_MASS_WEIGHT) {
                __int128 w = tr_i128_get(st.wsum[i]);
                if (!w) continue;
                if (w < INT64_MIN || w > INT64_MAX) { rc = TR_MATH_OVERFLOW; break; }
            }
            if (!any[g] || v < lo[g]) lo[g] = v;
            if (!any[g] || v > hiv[g]) hiv[g] = v;
            any[g] = 1;
        }
    }
    uint64_t total = 0, maxspan = 0, present = 0;
    for (uint64_t g = 0; g < G && !rc; g++) {
        uint64_t span = any[g] ? (uint64_t)hiv[g] - (uint64_t)lo[g] + 1 : 0;
        if (any[g] && span == 0) { rc = TR_MATH_OVERFLOW; break; }          /* 2^64 values */
        total = tr_budget_add(total, span);
        if (span > maxspan) maxspan = span;
        present += st.rows[g] > 0;
    }
    uint64_t glen = rc ? 0 : tr_scope_key_size(scope, TR_SCOPE_KEY_GROUPS);
    tr_scope_info sinfo;
    uint64_t klen = rc ? 0 : key_len(scope, &sinfo);
    tr_math_need need;
    need_zero(&need);
    need.need[TR_BUDGET_VARIABLES] = 1;
    need.need[TR_BUDGET_STATES] = G;
    need.need[TR_BUDGET_SUPPORT] = maxspan;
    uint64_t bytes = tr_budget_add(sizeof(tr_conv_family), tr_budget_mul(total, 8));
    bytes = tr_budget_add(bytes, tr_budget_mul(tr_budget_add(G, 1), 8));
    bytes = tr_budget_add(bytes, tr_budget_mul(G, 24));
    bytes = tr_budget_add(bytes, tr_budget_add(glen, klen));
    need.need[TR_BUDGET_BYTES] = tr_budget_add(bytes, tr_budget_mul(G, 32));
    if (!rc) rc = check_need(bud, &need, refusal);
    tr_conv_family *f = rc ? NULL : zalloc(sizeof *f);
    if (!rc && !f) rc = TR_MATH_MEMORY;
    if (f) {
        f->offset = zalloc(tr_budget_mul(G + 1, 8)); f->rows = zalloc(tr_budget_mul(G, 8));
        f->nulls = nulls; nulls = NULL;
        f->shift = zalloc(tr_budget_mul(G, 8)); f->coef = zalloc(tr_budget_mul(total, 8));
        f->grouping = zalloc(glen); f->grouping_len = glen;
        if (!f->offset || !f->rows || !f->shift || !f->coef || !f->grouping) rc = TR_MATH_MEMORY;
        if (!rc && tr_scope_key_write(scope, TR_SCOPE_KEY_GROUPS, f->grouping, glen)) rc = TR_MATH_INVALID;
        if (!rc) rc = key_of_scope(scope, NULL, NULL, &f->src);
    }
    if (!rc) {
        uint64_t at = 0;
        for (uint64_t g = 0; g < G; g++) {
            f->offset[g] = at; f->shift[g] = any[g] ? lo[g] : 0; f->rows[g] = st.rows[g];
            at += any[g] ? (uint64_t)hiv[g] - (uint64_t)lo[g] + 1 : 0;
        }
        f->offset[G] = at;
        for (uint64_t g = 0; g < G; g++) {                /* pass 2: the coefficients */
            uint64_t b = st.repr == TR_DIST_DENSE ? g * st.cells : st.offset[g];
            uint64_t e = st.repr == TR_DIST_DENSE ? b + st.cells : st.offset[g + 1];
            for (uint64_t i = b; i < e; i++) {
                if (!st.count[i]) continue;
                int nl;
                int64_t v = st.repr == TR_DIST_DENSE ? tr_dist_plan_coord(&plan, i - b, 0, &nl) : st.value[0][i];
                if (st.repr != TR_DIST_DENSE) nl = st.nulls[i] & 1;
                if (nl) continue;
                int64_t c = mass == TR_DIST_MASS_COUNT ? (int64_t)st.count[i] : (int64_t)tr_i128_get(st.wsum[i]);
                if (!c) continue;
                f->coef[f->offset[g] + ((uint64_t)v - (uint64_t)f->shift[g])] = c;
            }
        }
        f->keys = G; f->ncoef = total; f->maxspan = maxspan; f->present = present; f->mass = mass;
        f->scale = hi.scale[0]; f->col = hi.col[0]; f->rows_total = hi.rows; f->bytes = need.need[TR_BUDGET_BYTES];
        for (uint64_t g = 0; g < G; g++) f->nulls_total += f->nulls[g];
        f->magic = FAMILY_MAGIC;
    }
    free(lo); free(hiv); free(nulls); free(any);
    if (rc) { family_release(f); set_error(error, rc); return NULL; }
    set_error(error, TR_MATH_OK);
    return f;
}

int32_t tr_conv_family_get_info(const tr_conv_family *f, tr_conv_family_info *out) {
    if (!family_valid(f) || !out) return TR_MATH_INVALID;
    memset(out, 0, sizeof *out);
    out->keys = f->keys; out->present = f->present; out->ncoef = f->ncoef; out->maxspan = f->maxspan;
    out->rows = f->rows_total; out->nulls = f->nulls_total; out->bytes = f->bytes;
    out->digest[0] = f->src.digest[0]; out->digest[1] = f->src.digest[1];
    return TR_MATH_OK;
}
int32_t tr_conv_family_get_view(const tr_conv_family *f, tr_conv_family_view *v) {
    if (!family_valid(f) || !v) return TR_MATH_INVALID;
    memset(v, 0, sizeof *v);
    v->keys = f->keys; v->ncoef = f->ncoef; v->maxspan = f->maxspan;
    v->offset = f->offset; v->shift = f->shift; v->coef = f->coef; v->mass = f->mass; v->scale = f->scale;
    return TR_MATH_OK;
}

tr_conv_factor *tr_conv_family_factor(const tr_conv_family *f, uint64_t key, int32_t repr, const tr_math_budget *budget,
                                      tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    if (!family_valid(f) || (key != TR_CONV_ALL && key >= f->keys) || (repr != TR_GEN_DENSE && repr != TR_GEN_SPARSE)) {
        set_error(error, TR_MATH_INVALID); return NULL;
    }
    tr_math_budget dflt;
    const tr_math_budget *bud = budget_of(budget, &dflt);
    int64_t lo = 0, hi = -1;
    int any = 0;
    for (uint64_t g = key == TR_CONV_ALL ? 0 : key; g < (key == TR_CONV_ALL ? f->keys : key + 1); g++) {
        if (f->offset[g + 1] == f->offset[g]) continue;
        int64_t a = f->shift[g], b = (int64_t)((uint64_t)f->shift[g] + (f->offset[g + 1] - f->offset[g] - 1));
        if (!any || a < lo) lo = a;
        if (!any || b > hi) hi = b;
        any = 1;
    }
    uint64_t span = any ? (uint64_t)hi - (uint64_t)lo + 1 : 0;
    if (any && !span) { set_error(error, TR_MATH_OVERFLOW); return NULL; }
    int32_t rc = check_bytes(bud, tr_budget_mul(span, 16), refusal);
    tr_math_i128 *acc = rc ? NULL : zalloc(tr_budget_mul(span, 16));
    if (!rc && !acc) rc = TR_MATH_MEMORY;
    uint64_t rows = 0, nulls = 0;
    for (uint64_t g = key == TR_CONV_ALL ? 0 : key; !rc && g < (key == TR_CONV_ALL ? f->keys : key + 1); g++) {
        rows += f->rows[g]; nulls += f->nulls[g];
        for (uint64_t i = f->offset[g]; i < f->offset[g + 1]; i++) {
            uint64_t e = (uint64_t)f->shift[g] - (uint64_t)lo + (i - f->offset[g]);
            acc[e] = tr_i128_set(tr_i128_get(acc[e]) + f->coef[i]);   /* sums of int64 per value: < 2^127 */
        }
    }
    tr_gen_poly *p = rc ? NULL : tr_gen_from_dense_i128(lo, span, acc, repr, bud, refusal, &rc);
    free(acc);
    if (!p) { set_error(error, rc ? rc : TR_MATH_MEMORY); return NULL; }
    srcset one;
    one.n = 1; one.k[0] = f->src;                             /* borrowed: factor_make copies it */
    tr_conv_factor *r = factor_make(p, &one, TR_CONV_DATA, TR_CONV_MODE_BASE, error);
    if (!r) return NULL;
    r->info.keys = f->keys; r->info.key = key; r->info.keyed = f->keys > 1;
    r->info.tuples = rows; r->info.nulls = nulls; r->info.counted = f->mass == TR_DIST_MASS_COUNT;
    r->info.mass = f->mass; r->info.col[0] = f->col; r->info.scale[0] = f->scale;
    return r;
}

/* the key domains of two families: 0, or TR_MATH_DOMAIN_ERROR with the reason */
static int32_t families_check(const tr_conv_family *a, const tr_conv_family *b, int32_t combine, tr_conv_check *ck) {
    check_none(ck);
    if (combine == TR_CONV_SUM && a->scale != b->scale) { check_fill(ck, TR_CONV_SCALES_DIFFER, 0, 1, NULL, 0); return TR_MATH_DOMAIN_ERROR; }
    if (key_equal(&a->src, &b->src)) {
        if (a->grouping_len != b->grouping_len || (a->grouping_len && memcmp(a->grouping, b->grouping, (size_t)a->grouping_len))) {
            check_fill(ck, TR_CONV_GROUPING_DIFFERS, 0, 1, &a->src, a->src.keyed);
            return TR_MATH_DOMAIN_ERROR;
        }
    } else if (a->keys != b->keys) {
        check_fill(ck, TR_CONV_KEYS_DIFFER, 0, 1, NULL, 0);
        return TR_MATH_DOMAIN_ERROR;
    }
    return TR_MATH_OK;
}

int32_t tr_conv_conditional_check(const tr_conv_family *a, const tr_conv_family *b, int32_t combine,
                                  tr_conv_check *check) {
    if (!family_valid(a) || !family_valid(b) || (combine != TR_CONV_SUM && combine != TR_CONV_OUTER)) return TR_MATH_INVALID;
    return families_check(a, b, combine, check);
}

int32_t tr_conv_conditional_plan(const tr_conv_family *a, const tr_conv_family *b, uint64_t *ooff, uint64_t *total,
                                 uint64_t *pairs) {
    if (!family_valid(a) || !family_valid(b) || a->keys != b->keys || !total || !pairs) return TR_MATH_INVALID;
    return tr_conv_batch_plan(a->keys, a->offset, b->offset, 0, ooff, total, pairs);
}

int32_t tr_conv_conditional_need(const tr_conv_family *a, const tr_conv_family *b, int32_t combine, int32_t target,
                                 tr_math_need *need) {
    if (!family_valid(a) || !family_valid(b) || !need || (combine != TR_CONV_SUM && combine != TR_CONV_OUTER) ||
        (target != TR_CONV_NEED_CPU && target != TR_CONV_NEED_DEVICE) ||
        (target == TR_CONV_NEED_DEVICE && combine != TR_CONV_SUM)) return TR_MATH_INVALID;
    if (a->keys != b->keys) return TR_MATH_DOMAIN_ERROR;
    uint64_t total = 0, pairs = 0;
    int32_t rc = tr_conv_batch_plan(a->keys, a->offset, b->offset, 0, NULL, &total, &pairs);
    if (rc) return rc;
    need_zero(need);
    const int nv = combine == TR_CONV_SUM ? 1 : 2;
    need->need[TR_BUDGET_VARIABLES] = (uint64_t)nv;
    need->need[TR_BUDGET_STATES] = a->keys;
    need->need[TR_BUDGET_TENSOR] = pairs;
    uint64_t out = combine == TR_CONV_SUM ? total : pairs;
    need->need[TR_BUDGET_SUPPORT] = out;
    /* SUM: the output offsets and the 192-bit scratch; then the terms (values and int128 coefficients); then GEN-001's
     * adoption of the terms (tr_gen_from_terms sums the keys) */
    uint64_t bytes = combine == TR_CONV_SUM ? tr_budget_add(tr_budget_mul(tr_budget_add(a->keys, 1), 8), tr_budget_mul(total, 24)) : 0;
    bytes = tr_budget_add(bytes, tr_budget_mul(out, combine == TR_CONV_SUM ? 24 : 32));
    bytes = tr_budget_add(bytes, TR_CONV_GEN_TERMS_BYTES(out, nv));
    if (target == TR_CONV_NEED_DEVICE) {                         /* the device copies are a stage of their own */
        tr_conv_batch bt = {a->keys, 0, total, pairs, a->offset, b->offset, NULL, a->coef, b->coef};
        uint64_t dev = tr_conv_batch_bytes(&bt);
        if (dev > bytes) bytes = dev;
    }
    need->need[TR_BUDGET_BYTES] = bytes;
    return TR_MATH_OK;
}

static void conditional_meta(tr_conv_factor *r, const tr_conv_family *a, const tr_conv_family *b, int32_t combine) {
    uint64_t t = 0, p = 0;
    int counted = a->mass == TR_DIST_MASS_COUNT && b->mass == TR_DIST_MASS_COUNT;
    for (uint64_t g = 0; g < a->keys && counted; g++) {
        t = tr_budget_add(t, tr_budget_mul(a->rows[g], b->rows[g]));
        p = tr_budget_add(p, tr_budget_mul(a->rows[g] - a->nulls[g], b->rows[g] - b->nulls[g]));
        if (t == TR_BUDGET_UNKNOWN) counted = 0;
    }
    r->info.counted = counted; r->info.tuples = counted ? t : 0; r->info.nulls = counted ? t - p : 0;
    r->info.keys = a->keys; r->info.key = TR_CONV_ALL; r->info.combine = combine; r->info.draws = 2;
    r->info.mass = a->mass == b->mass ? a->mass : 0;
    r->info.col[0] = a->col; r->info.col[1] = combine == TR_CONV_OUTER ? b->col : -1;
    r->info.scale[0] = a->scale; r->info.scale[1] = combine == TR_CONV_OUTER ? b->scale : 0;
    r->info.keyed = a->keys > 1;
}

tr_conv_factor *tr_conv_conditional_adopt(const tr_conv_family *a, const tr_conv_family *b, const uint64_t *ooff,
                                          const int64_t *out, int32_t repr, const tr_math_budget *budget,
                                          tr_math_refusal *refusal, tr_conv_check *check, int32_t *error) {
    tr_budget_none(refusal);
    if (!family_valid(a) || !family_valid(b) || !ooff || (repr != TR_GEN_DENSE && repr != TR_GEN_SPARSE)) {
        set_error(error, TR_MATH_INVALID); return NULL;
    }
    int32_t rc = families_check(a, b, TR_CONV_SUM, check);
    if (rc) { set_error(error, rc); return NULL; }
    tr_math_budget dflt;
    const tr_math_budget *bud = budget_of(budget, &dflt);
    uint64_t total = 0, pairs = 0;
    rc = tr_conv_batch_plan(a->keys, a->offset, b->offset, 0, NULL, &total, &pairs);
    if (!rc && (ooff[0] != 0 || ooff[a->keys] != total || (total && !out))) rc = TR_MATH_INVALID;
    for (uint64_t g = 0; g < a->keys && !rc; g++)
        if (ooff[g + 1] - ooff[g] != tr_conv_batch_len(a->offset[g + 1] - a->offset[g], b->offset[g + 1] - b->offset[g], 0))
            rc = TR_MATH_INVALID;
    if (!rc) rc = check_bytes(bud, tr_budget_mul(total, 24), refusal);
    int64_t *vals = rc ? NULL : zalloc(tr_budget_mul(total, 8));
    tr_math_i128 *coef = rc ? NULL : zalloc(tr_budget_mul(total, 16));
    if (!rc && (!vals || !coef)) rc = TR_MATH_MEMORY;
    uint64_t n = 0;
    for (uint64_t g = 0; g < a->keys && !rc; g++) {
        __int128 base = (__int128)a->shift[g] + b->shift[g];
        for (uint64_t e = 0; e < ooff[g + 1] - ooff[g]; e++) {
            const uint64_t *w = (const uint64_t *)out + 3 * (ooff[g] + e);
            if (!w[0] && !w[1] && !w[2]) continue;
            __int128 v, value = base + (__int128)e;
            if (!tr_conv_fits128(w, &v) || value < INT64_MIN || value > INT64_MAX) { rc = TR_MATH_OVERFLOW; break; }
            vals[n] = (int64_t)value; coef[n] = tr_i128_set(v); n++;
        }
    }
    tr_gen_poly *p = rc ? NULL : tr_gen_from_terms(1, n, vals, coef, repr, NULL, bud, refusal, &rc);   /* sums the keys */
    free(vals); free(coef);
    if (!p) { set_error(error, rc ? rc : TR_MATH_MEMORY); return NULL; }
    srcset s;
    memset(&s, 0, sizeof s);
    s.n = 1; s.k[0] = a->src;                                  /* borrowed keys: factor_make copies them */
    if (!key_equal(&a->src, &b->src)) { s.n = 2; s.k[1] = b->src; }
    tr_conv_factor *r = factor_make(p, &s, TR_CONV_DERIVED, TR_CONV_MODE_CONDITIONAL, error);
    if (r) conditional_meta(r, a, b, TR_CONV_SUM);
    return r;
}

tr_conv_factor *tr_conv_conditional(const tr_conv_family *a, const tr_conv_family *b, int32_t combine, int32_t repr,
                                    const tr_math_budget *budget, tr_math_refusal *refusal, tr_conv_check *check,
                                    int32_t *error) {
    tr_budget_none(refusal);
    check_none(check);
    if (!family_valid(a) || !family_valid(b) || (combine != TR_CONV_SUM && combine != TR_CONV_OUTER) ||
        (repr != TR_GEN_DENSE && repr != TR_GEN_SPARSE)) { set_error(error, TR_MATH_INVALID); return NULL; }
    int32_t rc = families_check(a, b, combine, check);
    if (rc) { set_error(error, rc); return NULL; }
    tr_math_budget dflt;
    const tr_math_budget *bud = budget_of(budget, &dflt);
    tr_math_need need;
    rc = tr_conv_conditional_need(a, b, combine, TR_CONV_NEED_CPU, &need);
    if (!rc) rc = check_need(bud, &need, refusal);
    if (rc) { set_error(error, rc); return NULL; }
    if (combine == TR_CONV_SUM) {
        uint64_t total = 0, pairs = 0;
        uint64_t *ooff = zalloc(tr_budget_mul(a->keys + 1, 8));
        if (!ooff) { set_error(error, TR_MATH_MEMORY); return NULL; }
        tr_conv_conditional_plan(a, b, ooff, &total, &pairs);
        int64_t *out = zalloc(tr_budget_mul(total, 24));
        tr_conv_batch bt = {a->keys, 0, total, pairs, a->offset, b->offset, ooff, a->coef, b->coef};
        if (!out) rc = TR_MATH_MEMORY;
        if (!rc) rc = batch_compute(&bt, out);                   /* the needs were checked above */
        tr_conv_factor *r = rc ? NULL : tr_conv_conditional_adopt(a, b, ooff, out, repr, bud, refusal, check, &rc);
        free(ooff); free(out);
        set_error(error, rc);
        return r;
    }
    /* OUTER: per key the pairs (a_i, b_j) at (shift_a + i, shift_b + j): every product of two int64 fits int128 */
    uint64_t n = need.need[TR_BUDGET_SUPPORT];
    int64_t *vals = zalloc(tr_budget_mul(n, 16));
    tr_math_i128 *coef = zalloc(tr_budget_mul(n, 16));
    if (!vals || !coef) rc = TR_MATH_MEMORY;
    uint64_t m = 0;
    for (uint64_t g = 0; g < a->keys && !rc; g++)
        for (uint64_t i = a->offset[g]; i < a->offset[g + 1]; i++) {
            if (!a->coef[i]) continue;
            for (uint64_t j = b->offset[g]; j < b->offset[g + 1]; j++) {
                if (!b->coef[j]) continue;
                vals[2 * m] = (int64_t)((uint64_t)a->shift[g] + (i - a->offset[g]));
                vals[2 * m + 1] = (int64_t)((uint64_t)b->shift[g] + (j - b->offset[g]));
                coef[m++] = tr_i128_set((__int128)a->coef[i] * b->coef[j]);
            }
        }
    tr_gen_poly *p = rc ? NULL : tr_gen_from_terms(2, m, vals, coef, repr, NULL, bud, refusal, &rc);
    free(vals); free(coef);
    if (!p) { set_error(error, rc ? rc : TR_MATH_MEMORY); return NULL; }
    srcset s;
    memset(&s, 0, sizeof s);
    s.n = 1; s.k[0] = a->src;
    if (!key_equal(&a->src, &b->src)) { s.n = 2; s.k[1] = b->src; }
    tr_conv_factor *r = factor_make(p, &s, TR_CONV_DERIVED, TR_CONV_MODE_CONDITIONAL, error);
    if (r) conditional_meta(r, a, b, TR_CONV_OUTER);
    return r;
}

/* ------------------------------------------------------------------ batches */
static int32_t batch_check(const tr_conv_batch *b) { return tr_conv_batch_check(b) ? TR_MATH_INVALID : TR_MATH_OK; }
static int32_t batch_compute(const tr_conv_batch *b, int64_t *out) {
    for (uint64_t t = 0; t < b->total; t++) tr_conv_batch_one(b, t, (uint64_t *)out + 3 * t);
    return TR_MATH_OK;
}
int32_t tr_conv_batch_run(const tr_conv_batch *b, int64_t *out, const tr_math_budget *budget, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    int32_t rc = batch_check(b);
    if (rc || (b->total && !out)) return rc ? rc : TR_MATH_INVALID;
    tr_math_budget dflt;
    const tr_math_budget *bud = budget_of(budget, &dflt);
    tr_math_need need;
    need_zero(&need);
    need.need[TR_BUDGET_TENSOR] = b->pairs;
    need.need[TR_BUDGET_OUTPUT] = b->total;
    rc = check_need(bud, &need, refusal);
    return rc ? rc : batch_compute(b, out);
}
int32_t tr_conv_batch_layout(uint64_t npairs, const uint64_t *aoff, const uint64_t *boff, uint64_t order,
                             uint64_t *ooff, uint64_t *total, uint64_t *pairs) {
    if ((npairs && (!aoff || !boff || !ooff)) || !total || !pairs) return TR_MATH_INVALID;
    if (!npairs) { if (ooff) ooff[0] = 0; *total = 0; *pairs = 0; return TR_MATH_OK; }
    return tr_conv_batch_plan(npairs, aoff, boff, order, ooff, total, pairs) ? TR_MATH_INVALID : TR_MATH_OK;
}

/* ------------------------------------------------------------------ bounded series */
static int series_valid(const tr_conv_series *s) { return s && s->magic == SERIES_MAGIC && s->coef; }
static void series_release(tr_conv_series *s) {
    if (!s) return;
    free(s->coef);
    set_release(&s->src);
    s->magic = 0;
    free(s);
}
void tr_conv_series_free(tr_conv_series *s) { if (s && s->magic == SERIES_MAGIC) series_release(s); }
static int32_t series_need(uint64_t order, int products, const tr_math_budget *b, tr_math_refusal *r) {
    tr_math_need need;
    need_zero(&need);
    need.need[TR_BUDGET_VARIABLES] = 1;
    need.need[TR_BUDGET_DEGREE] = order - 1;
    need.need[TR_BUDGET_SUPPORT] = order;
    if (products) need.need[TR_BUDGET_TENSOR] = tr_budget_mul(order, order);
    need.need[TR_BUDGET_BYTES] = tr_budget_add(sizeof(tr_conv_series), tr_budget_mul(order, 8));
    return check_need(b, &need, r);
}
/* a series of order N (coefficients zeroed), sources copied from src (may be NULL) */
static tr_conv_series *series_alloc(uint64_t order, int64_t bound, int64_t shift, const srcset *src, int32_t *rc) {
    tr_conv_series *s = zalloc(sizeof *s);
    if (s) s->coef = zalloc(tr_budget_mul(order, 8));
    if (!s || !s->coef) { series_release(s); *rc = TR_MATH_MEMORY; return NULL; }
    s->order = (uint32_t)order; s->bound = bound; s->shift = shift;
    *rc = src ? set_merge(&s->src, src) : TR_MATH_OK;
    if (*rc) { series_release(s); return NULL; }
    s->magic = SERIES_MAGIC;
    return s;
}
static void series_maxabs(tr_conv_series *s) {
    int64_t m = 0;
    for (uint32_t e = 0; e < s->order; e++) { int64_t a = s->coef[e] < 0 ? -s->coef[e] : s->coef[e]; if (a > m) m = a; }
    s->maxabs = m;
}

tr_conv_series *tr_conv_series_create(const int64_t *coef, uint32_t count, uint32_t order, int64_t bound,
                                      const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    if (order < 1 || count > order || (count && !coef) || bound < 0) { set_error(error, TR_MATH_INVALID); return NULL; }
    tr_math_budget dflt;
    int32_t rc = series_need(order, 0, budget_of(budget, &dflt), refusal);
    if (rc) { set_error(error, rc); return NULL; }
    for (uint32_t e = 0; e < count; e++)
        if (coef[e] == INT64_MIN || (coef[e] < 0 ? -coef[e] : coef[e]) > bound) { set_error(error, TR_MATH_OVERFLOW); return NULL; }
    tr_conv_series *s = series_alloc(order, bound, 0, NULL, &rc);
    if (!s) { set_error(error, rc); return NULL; }
    if (count) memcpy(s->coef, coef, (size_t)count * 8);
    series_maxabs(s);
    set_error(error, TR_MATH_OK);
    return s;
}

tr_conv_series *tr_conv_series_from_factor(const tr_conv_factor *f, uint32_t order, int64_t bound,
                                           const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    if (!factor_valid(f) || f->info.nvars != 1 || order < 1 || bound < 0) { set_error(error, TR_MATH_INVALID); return NULL; }
    tr_math_budget dflt;
    const tr_math_budget *bud = budget_of(budget, &dflt);
    int32_t rc = series_need(order, 0, bud, refusal);
    if (rc) { set_error(error, rc); return NULL; }
    tr_gen_info gi;
    tr_gen_get_info(f->poly, &gi);
    tr_conv_series *s = series_alloc(order, bound, gi.shift[0], &f->src, &rc);
    if (!s) { set_error(error, rc); return NULL; }
    for (uint32_t e = 0; e < order && e < gi.span[0] && !rc; e++) {
        int64_t v;
        __int128 val = (__int128)gi.shift[0] + e;
        if (val > INT64_MAX) break;
        v = (int64_t)val;
        tr_bigint *c = NULL;
        rc = tr_gen_coeff(f->poly, &v, bud, refusal, &c);
        int64_t x = 0;
        if (!rc && tr_bigint_to_i64(c, &x)) rc = TR_MATH_OVERFLOW;
        if (!rc && (x == INT64_MIN || (x < 0 ? -x : x) > bound)) rc = TR_MATH_OVERFLOW;
        tr_bigint_free(c);
        if (!rc) s->coef[e] = x;
    }
    if (rc) { series_release(s); set_error(error, rc); return NULL; }
    series_maxabs(s);
    set_error(error, TR_MATH_OK);
    return s;
}

int32_t tr_conv_series_get_info(const tr_conv_series *s, tr_conv_series_info *out) {
    if (!series_valid(s) || !out) return TR_MATH_INVALID;
    out->order = s->order; out->sources = s->src.n; out->bound = s->bound; out->maxabs = s->maxabs; out->shift = s->shift;
    return TR_MATH_OK;
}
int32_t tr_conv_series_coefficients(const tr_conv_series *s, int64_t *out, uint64_t capacity) {
    if (!series_valid(s) || !out || capacity < s->order) return TR_MATH_INVALID;
    memcpy(out, s->coef, (size_t)s->order * 8);
    return TR_MATH_OK;
}

tr_conv_series *tr_conv_series_add(const tr_conv_series *a, const tr_conv_series *b, int64_t bound,
                                   const tr_math_budget *budget, tr_math_refusal *refusal, uint64_t *violation,
                                   int32_t *error) {
    tr_budget_none(refusal);
    if (violation) *violation = TR_CONV_NONE;
    if (!series_valid(a) || !series_valid(b) || bound < 0) { set_error(error, TR_MATH_INVALID); return NULL; }
    if (a->order != b->order) { set_error(error, TR_MATH_CARRIER_ERROR); return NULL; }
    if (a->shift != b->shift) { set_error(error, TR_MATH_DOMAIN_ERROR); return NULL; }
    tr_math_budget dflt;
    int32_t rc = series_need(a->order, 0, budget_of(budget, &dflt), refusal);
    tr_conv_series *s = rc ? NULL : series_alloc(a->order, bound, a->shift, &a->src, &rc);
    if (s) rc = set_merge(&s->src, &b->src);
    for (uint32_t e = 0; s && e < a->order && !rc; e++) {
        __int128 v = (__int128)a->coef[e] + b->coef[e];
        if (v < -(__int128)bound || v > bound) { rc = TR_MATH_OVERFLOW; if (violation) *violation = e; break; }
        s->coef[e] = (int64_t)v;
    }
    if (rc) { series_release(s); set_error(error, rc); return NULL; }
    series_maxabs(s);
    set_error(error, TR_MATH_OK);
    return s;
}

/* out = a x b mod z^N (rows of N int64), bounded: 0, or TR_MATH_OVERFLOW with *e the first violating exponent */
static int32_t mul_rows(const int64_t *a, const int64_t *b, uint64_t N, int64_t bound, int64_t *out, uint64_t *e,
                        int64_t *maxabs) {
    for (uint64_t k = 0; k < N; k++) {
        uint64_t w[3];
        tr_conv_coef192(a, N, b, N, k, w);
        int64_t v;
        if (!tr_conv_bounded(w, bound, &v)) { *e = k; return TR_MATH_OVERFLOW; }
        out[k] = v;
        int64_t m = v < 0 ? -v : v;
        if (maxabs && m > *maxabs) *maxabs = m;
    }
    return TR_MATH_OK;
}
static int32_t series_linked(const tr_conv_series *a, const tr_conv_series *b, uint32_t k, const tr_conv_decl *d,
                             tr_conv_check *ck) {
    if (d && !decl_valid(d)) return TR_MATH_INVALID;
    const srcset *s[2] = {&a->src, b ? &b->src : NULL};
    uint32_t mult[2] = {k, 1};
    return linked(s, mult, b ? 2 : 1, d, ck);
}
static int32_t shift_add(int64_t a, int64_t b, uint64_t times, int64_t *out) {
    __int128 v = (__int128)a + (__int128)b * (__int128)times;
    if (v < INT64_MIN || v > INT64_MAX) return TR_MATH_OVERFLOW;
    *out = (int64_t)v;
    return TR_MATH_OK;
}

tr_conv_series *tr_conv_series_mul(const tr_conv_series *a, const tr_conv_series *b, int64_t bound,
                                   const tr_conv_decl *decl, const tr_math_budget *budget, tr_math_refusal *refusal,
                                   tr_conv_check *check, uint64_t *violation, int32_t *error) {
    tr_budget_none(refusal);
    check_none(check);
    if (violation) *violation = TR_CONV_NONE;
    if (!series_valid(a) || !series_valid(b) || bound < 0) { set_error(error, TR_MATH_INVALID); return NULL; }
    if (a->order != b->order) { set_error(error, TR_MATH_CARRIER_ERROR); return NULL; }
    int32_t rc = series_linked(a, b, 1, decl, check);
    int64_t shift = 0;
    if (!rc) rc = shift_add(a->shift, b->shift, 1, &shift);
    tr_math_budget dflt;
    if (!rc) rc = series_need(a->order, 1, budget_of(budget, &dflt), refusal);
    tr_conv_series *s = rc ? NULL : series_alloc(a->order, bound, shift, &a->src, &rc);
    if (s) rc = set_merge(&s->src, &b->src);
    uint64_t e = 0;
    if (s && !rc) {
        rc = mul_rows(a->coef, b->coef, a->order, bound, s->coef, &e, NULL);
        if (rc && violation) *violation = e;
    }
    if (rc) { series_release(s); set_error(error, rc); return NULL; }
    series_maxabs(s);
    set_error(error, TR_MATH_OK);
    return s;
}

tr_conv_series *tr_conv_series_pow(const tr_conv_series *a, uint32_t k, int64_t bound, const tr_conv_decl *decl,
                                   const tr_math_budget *budget, tr_math_refusal *refusal, tr_conv_check *check,
                                   uint64_t *violation, int32_t *error) {
    tr_budget_none(refusal);
    check_none(check);
    if (violation) *violation = TR_CONV_NONE;
    if (!series_valid(a) || bound < 0) { set_error(error, TR_MATH_INVALID); return NULL; }
    int32_t rc = series_linked(a, NULL, k, decl, check);
    int64_t shift = 0;
    if (!rc) rc = shift_add(0, a->shift, k, &shift);
    tr_math_budget dflt;
    const tr_math_budget *bud = budget_of(budget, &dflt);
    if (!rc) {
        tr_math_need need;
        need_zero(&need);
        need.need[TR_BUDGET_VARIABLES] = 1;
        need.need[TR_BUDGET_DEGREE] = a->order - 1;
        need.need[TR_BUDGET_SUPPORT] = a->order;
        need.need[TR_BUDGET_TENSOR] = tr_budget_mul(a->order, a->order);
        need.need[TR_BUDGET_ITERATIONS] = k;
        need.need[TR_BUDGET_BYTES] = tr_budget_add(sizeof(tr_conv_series), tr_budget_mul(a->order, 16));
        rc = check_need(bud, &need, refusal);
    }
    tr_conv_series *s = rc ? NULL : series_alloc(a->order, bound, shift, k ? &a->src : NULL, &rc);
    int64_t *tmp = s ? zalloc(tr_budget_mul(a->order, 8)) : NULL;
    if (s && !tmp) rc = TR_MATH_MEMORY;
    if (s && !rc) {
        s->coef[0] = 1;                                        /* a^0 = 1, then a^j = a^(j-1) a, each bounded */
        if (bound < 1 && k == 0) rc = TR_MATH_OVERFLOW;
        for (uint32_t j = 1; j <= k && !rc; j++) {
            uint64_t e = 0;
            rc = mul_rows(s->coef, a->coef, a->order, bound, tmp, &e, NULL);
            if (rc) { if (violation) *violation = (uint64_t)j * a->order + e; break; }
            memcpy(s->coef, tmp, (size_t)a->order * 8);
        }
        if (rc == TR_MATH_OVERFLOW && k == 0 && violation) *violation = 0;
    }
    free(tmp);
    if (rc) { series_release(s); set_error(error, rc); return NULL; }
    series_maxabs(s);
    set_error(error, TR_MATH_OK);
    return s;
}

/* the walk core: lengths rows (row 0 = 1), each the previous times the step, then the totals; rows (lengths x N) or a
 * two-row scratch; info->violation = the first violation (row-major; totals after the rows) */
static void walks_core(const int64_t *step, uint64_t N, uint64_t lengths, int64_t bound, int64_t *rows, int64_t *scratch,
                       int64_t *total, tr_conv_walks *info) {
    info->lengths = lengths; info->violation = TR_CONV_NONE; info->status = 0; info->reserved = 0;
    info->bound = bound; info->maxabs = 0;
    uint64_t *acc = (uint64_t *)scratch + (rows ? 0 : 2 * N);  /* totals: 3 words per coefficient after the rows */
    memset(acc, 0, (size_t)N * 24);
    for (uint64_t l = 0; l < lengths; l++) {
        int64_t *cur = rows ? rows + l * N : scratch + (l & 1) * N;
        const int64_t *prev = l == 0 ? NULL : rows ? rows + (l - 1) * N : scratch + ((l + 1) & 1) * N;
        if (l == 0) {
            if (bound < 1) { info->status = TR_MATH_OVERFLOW; info->violation = 0; return; }
            memset(cur, 0, (size_t)N * 8);
            cur[0] = 1;
            if (info->maxabs < 1) info->maxabs = 1;
        } else {
            uint64_t e = 0;
            if (mul_rows(prev, step, N, bound, cur, &e, &info->maxabs)) {
                info->status = TR_MATH_OVERFLOW; info->violation = l * N + e; return;
            }
        }
        for (uint64_t e = 0; e < N; e++) tr_wide_add128(acc + 3 * e, (__int128)cur[e]);
    }
    for (uint64_t e = 0; e < N; e++) {
        int64_t v;
        if (!tr_conv_bounded(acc + 3 * e, bound, &v)) { info->status = TR_MATH_OVERFLOW; info->violation = lengths * N + e; return; }
        total[e] = v;
        int64_t m = v < 0 ? -v : v;
        if (m > info->maxabs) info->maxabs = m;
    }
}

int32_t tr_conv_walks_run(const int64_t *step, uint64_t order, uint64_t lengths, int64_t bound, int64_t *rows,
                          int64_t *total, tr_conv_walks *info, const tr_math_budget *budget, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!step || !order || !lengths || !rows || !total || !info || bound < 0) return TR_MATH_INVALID;
    tr_math_budget dflt;
    const tr_math_budget *bud = budget_of(budget, &dflt);
    tr_math_need need;
    need_zero(&need);
    need.need[TR_BUDGET_DEGREE] = order - 1;
    need.need[TR_BUDGET_SUPPORT] = order;
    need.need[TR_BUDGET_ITERATIONS] = lengths - 1;
    need.need[TR_BUDGET_OUTPUT] = tr_budget_mul(lengths, order);
    need.need[TR_BUDGET_BYTES] = tr_budget_mul(order, 24);
    int32_t rc = check_need(bud, &need, refusal);
    if (rc) return rc;
    int64_t *scratch = zalloc(tr_budget_mul(order, 24));
    if (!scratch) return TR_MATH_MEMORY;
    walks_core(step, order, lengths, bound, rows, scratch, total, info);
    free(scratch);
    return info->status;
}

int32_t tr_conv_series_walks_need(const tr_conv_series *s, uint32_t max_length, int32_t keep_rows, tr_math_need *need,
                                  uint64_t *lengths) {
    if (!series_valid(s) || !need) return TR_MATH_INVALID;
    if (s->shift != 0) return TR_MATH_DOMAIN_ERROR;
    uint64_t L;
    if (max_length == TR_CONV_UNBOUNDED) {
        if (s->coef[0] != 0) return TR_MATH_NO_SOLUTION;       /* the walk sum of every total diverges */
        L = s->order - 1;
    } else L = s->coef[0] == 0 && max_length > s->order - 1 ? s->order - 1 : max_length;   /* rows past N - 1 vanish */
    need_zero(need);
    need->need[TR_BUDGET_VARIABLES] = 1;
    need->need[TR_BUDGET_DEGREE] = s->order - 1;
    need->need[TR_BUDGET_SUPPORT] = s->order;
    need->need[TR_BUDGET_TENSOR] = tr_budget_mul(s->order, s->order);
    need->need[TR_BUDGET_ITERATIONS] = L;
    if (keep_rows) need->need[TR_BUDGET_OUTPUT] = tr_budget_mul(L + 1, s->order);
    uint64_t bytes = tr_budget_add(tr_budget_mul(s->order, 40), sizeof(tr_conv_series));
    if (keep_rows) bytes = tr_budget_add(bytes, tr_budget_mul(tr_budget_mul(L + 1, s->order), 8));
    need->need[TR_BUDGET_BYTES] = bytes;
    if (lengths) *lengths = L + 1;
    return TR_MATH_OK;
}

int32_t tr_conv_series_walks(const tr_conv_series *step, uint32_t max_length, int64_t bound, const tr_conv_decl *decl,
                             int64_t *rows, uint64_t rows_capacity, tr_conv_series **total, tr_conv_walks *info,
                             const tr_math_budget *budget, tr_math_refusal *refusal, tr_conv_check *check) {
    tr_budget_none(refusal);
    check_none(check);
    if (total) *total = NULL;
    if (!series_valid(step) || !info || bound < 0) return TR_MATH_INVALID;
    tr_math_need need;
    uint64_t lengths = 0;
    int32_t rc = tr_conv_series_walks_need(step, max_length, rows != NULL, &need, &lengths);
    if (rc) return rc;
    if (rows && rows_capacity < tr_budget_mul(lengths, step->order)) return TR_MATH_INVALID;
    rc = series_linked(step, NULL, lengths > 2 ? 2 : (uint32_t)(lengths - 1), decl, check);   /* powers >= 2: draws */
    if (rc) return rc;
    tr_math_budget dflt;
    rc = check_need(budget_of(budget, &dflt), &need, refusal);
    if (rc) return rc;
    const uint64_t N = step->order;
    int64_t *scratch = zalloc(tr_budget_mul(N, 40));
    tr_conv_series *t = scratch ? series_alloc(N, bound, 0, &step->src, &rc) : NULL;
    if (!scratch) rc = TR_MATH_MEMORY;
    if (!rc) {
        walks_core(step->coef, N, lengths, bound, rows, scratch, t->coef, info);
        rc = info->status;
    }
    free(scratch);
    if (rc) { series_release(t); return rc; }
    series_maxabs(t);
    if (total) *total = t; else series_release(t);
    return TR_MATH_OK;
}

int32_t tr_conv_series_check(const tr_conv_series *a, const tr_conv_series *b, uint32_t k, const tr_conv_decl *decl,
                             tr_conv_check *check) {
    if (!series_valid(a) || (b && !series_valid(b))) return TR_MATH_INVALID;
    return series_linked(a, b, k, decl, check);
}

tr_conv_series *tr_conv_series_walks_adopt(const tr_conv_series *step, const int64_t *total, int64_t bound,
                                           const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    if (!series_valid(step) || !total || bound < 0) { set_error(error, TR_MATH_INVALID); return NULL; }
    tr_math_budget dflt;
    int32_t rc = series_need(step->order, 0, budget_of(budget, &dflt), refusal);
    for (uint32_t e = 0; !rc && e < step->order; e++)
        if (total[e] == INT64_MIN || (total[e] < 0 ? -total[e] : total[e]) > bound) rc = TR_MATH_OVERFLOW;
    tr_conv_series *t = rc ? NULL : series_alloc(step->order, bound, 0, &step->src, &rc);
    if (!t) { set_error(error, rc); return NULL; }
    memcpy(t->coef, total, (size_t)step->order * 8);
    series_maxabs(t);
    set_error(error, TR_MATH_OK);
    return t;
}

tr_conv_factor *tr_conv_series_to_factor(const tr_conv_series *s, int32_t repr, const tr_math_budget *budget,
                                         tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    if (!series_valid(s)) { set_error(error, TR_MATH_INVALID); return NULL; }
    tr_math_budget dflt;
    const tr_math_budget *bud = budget_of(budget, &dflt);
    int32_t rc = check_bytes(bud, tr_budget_mul(s->order, 16), refusal);
    tr_math_i128 *c = rc ? NULL : zalloc(tr_budget_mul(s->order, 16));
    if (!rc && !c) rc = TR_MATH_MEMORY;
    for (uint32_t e = 0; c && e < s->order; e++) c[e] = tr_i128_set(s->coef[e]);
    tr_gen_poly *p = rc ? NULL : tr_gen_from_dense_i128(s->shift, s->order, c, repr, bud, refusal, &rc);
    free(c);
    if (!p) { set_error(error, rc ? rc : TR_MATH_MEMORY); return NULL; }
    tr_conv_factor *f = factor_make(p, &s->src, s->src.n ? TR_CONV_DERIVED : TR_CONV_FREE, TR_CONV_MODE_SERIES, error);
    return f;
}
