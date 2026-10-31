/* FND-007: proof certificates and their evidence scope; see math_certificates.h.
 * SPDX-License-Identifier: Apache-2.0 */
#include "math_certificates.h"
#include "mathematics.h"
#include "math_domain.h"
#include "torble.h"
#include <stdlib.h>
#include <string.h>

TR_MATH_ABI(tr_cert_bit, TR_CERT_ABI);   /* the ctypes boundary of FND-007 (math_handles.h) */
TR_MATH_ABI(tr_cert_claim, TR_CERT_ABI);
TR_MATH_ABI(tr_cert_evidence, TR_CERT_ABI);
TR_MATH_ABI(tr_cert_sample, TR_CERT_ABI);
TR_MATH_ABI(tr_cert_staleness, TR_CERT_ABI);
TR_MATH_ABI(tr_cert_decision, TR_CERT_ABI);
TR_MATH_ABI(tr_cert_info, TR_CERT_ABI);
TR_MATH_ABI(tr_cert_claim_desc, TR_CERT_ABI);
TR_MATH_ABI(tr_cert_law_desc, TR_CERT_ABI);
TR_MATH_ABI(tr_cert_column, TR_CERT_ABI);
TR_MATH_ABI(tr_cert_dim, TR_CERT_ABI);
TR_MATH_ABI(tr_cert_source, TR_CERT_ABI);

_Static_assert(sizeof(tr_cert_bit) == 24, "tr_cert_bit ABI");
_Static_assert(sizeof(tr_cert_claim) == 192, "tr_cert_claim ABI");
_Static_assert(sizeof(tr_cert_evidence) == 64, "tr_cert_evidence ABI");
_Static_assert(sizeof(tr_cert_sample) == 40, "tr_cert_sample ABI");
_Static_assert(sizeof(tr_cert_staleness) == 32, "tr_cert_staleness ABI");
_Static_assert(sizeof(tr_cert_decision) == 64, "tr_cert_decision ABI");
_Static_assert(sizeof(tr_cert_info) == 200, "tr_cert_info ABI");
_Static_assert(sizeof(tr_cert_claim_desc) == 64, "tr_cert_claim_desc ABI");
_Static_assert(sizeof(tr_cert_law_desc) == 176, "tr_cert_law_desc ABI");
_Static_assert(sizeof(tr_cert_column) == 64, "tr_cert_column ABI");
_Static_assert(sizeof(tr_cert_dim) == 88, "tr_cert_dim ABI");
_Static_assert(sizeof(tr_cert_source) == 2888, "tr_cert_source ABI");

extern void at_decode_column(const atable_t *, int, uint32_t *);

#define CERT_MAGIC UINT32_C(0x54524543)
#define RECORD_MAGIC UINT32_C(0x44524352)
#define RECORD_TAG UINT64_C(0x3143455243545254)
#define CERT_SERIAL_TAG UINT64_C(0x3154524543425254)
#define SERIAL_VERSION 1
#define P_FORBID (1 << TR_CERT_NULL_FORBID)
#define P_SKIP (1 << TR_CERT_NULL_SKIP)
#define P_PROP (1 << TR_CERT_NULL_PROPAGATE)

/* ------------------------------------------------------------------ constant tables */
static const tr_cert_claim_desc CLAIMS[TR_CLAIM_KINDS - 1] = {
    {TR_CLAIM_AFFINE, 2, 2, TR_CERT_UNIT_ROWS, P_FORBID | P_SKIP | P_PROP, 1, 1, 0, "affine"},
    {TR_CLAIM_SUM, 3, 3, TR_CERT_UNIT_ROWS, P_FORBID | P_SKIP | P_PROP, 1, 1, 0, "sum"},
    {TR_CLAIM_PRODUCT, 3, 3, TR_CERT_UNIT_ROWS, P_FORBID | P_SKIP | P_PROP, 1, 1, 0, "product"},
    {TR_CLAIM_PARITY, 1, TR_CERT_MAX_COLS, TR_CERT_UNIT_ROWS, P_FORBID | P_SKIP, 0, 1, 2, "parity"},
    {TR_CLAIM_REFINEMENT, 2, 2, TR_CERT_UNIT_ROWS, P_FORBID | P_SKIP | P_PROP, 1, 1, 0, "refinement"},
    {TR_CLAIM_COVERAGE, 1, 1, TR_CERT_UNIT_GROUPS, P_SKIP, 0, 0, 0, "coverage"},
    {TR_CLAIM_POLYNOMIAL, 1, TR_CERT_MAX_COLS, TR_CERT_UNIT_ROWS, P_FORBID | P_SKIP, 0, 1, 0, "polynomial"},
    {TR_CLAIM_UNIQUE, 1, 1, TR_CERT_UNIT_ROWS, P_FORBID | P_SKIP, 0, 0, 0, "unique"},
    {TR_CLAIM_NOT_NULL, 1, 1, TR_CERT_UNIT_ROWS, P_FORBID, 0, 1, 0, "not null"},
    {TR_CLAIM_REFERENCE, 1, 1, TR_CERT_UNIT_ROWS, P_FORBID | P_SKIP, 0, 1, 0, "reference"},
};
static const tr_cert_law_desc LAWS[TR_CERT_LAWS - 1] = {
    {TR_CERT_LAW_RING_IDENTITY, TR_CLAIM_POLYNOMIAL, 1, 0, "ring identity",
     "the polynomial's canonical form in Z[x] is zero: it holds in every commutative ring"},
    {TR_CERT_LAW_GFP_FUNCTION, TR_CLAIM_POLYNOMIAL, 0, 0, "GF(p) function identity",
     "reduced by x^p = x and mod p the polynomial is zero: it vanishes on every point of GF(p)^n"},
};

int32_t tr_cert_claim_describe(int32_t kind, tr_cert_claim_desc *out) {
    if (!out || kind < 1 || kind >= TR_CLAIM_KINDS) return TR_MATH_INVALID;
    *out = CLAIMS[kind - 1];
    return TR_MATH_OK;
}
int32_t tr_cert_law_describe(int32_t law, tr_cert_law_desc *out) {
    if (!out || law < 1 || law >= TR_CERT_LAWS) return TR_MATH_INVALID;
    *out = LAWS[law - 1];
    return TR_MATH_OK;
}

/* ------------------------------------------------------------------ small helpers */
static void set_error(int32_t *error, int32_t code) { if (error) *error = code; }
static void *zalloc(uint64_t bytes) {
    if (bytes > (uint64_t)SIZE_MAX) return NULL;
    return calloc(1, bytes ? (size_t)bytes : 1);
}
static uint64_t tail_mask(uint64_t n, uint64_t w) {
    uint64_t left = n - w * 64;
    return left >= 64 ? UINT64_MAX : (left ? (UINT64_C(1) << left) - 1 : 0);
}
static int kind_ok(int32_t k) { return k >= TR_CERT_LAW && k <= TR_CERT_HYPOTHESIS; }
static const tr_math_budget *budget_or(const tr_math_budget *b, tr_math_budget *dflt) {
    if (b) return b;
    tr_budget_default(dflt);
    return dflt;
}
static void need_zero(tr_math_need *n) { for (int i = 0; i < TR_BUDGET_RESOURCES; i++) n->need[i] = 0; }
static void need_max(tr_math_need *a, const tr_math_need *b) {
    for (int i = 0; i < TR_BUDGET_RESOURCES; i++) if (b->need[i] > a->need[i]) a->need[i] = b->need[i];
}

/* 128-bit digest of a byte stream (two multiply-xorshift lanes; a pre-filter, not an identity) */
typedef struct { uint64_t h0, h1, word, len; int nbytes; } hasher;
static void h_mix(hasher *h, uint64_t w) {
    h->h0 = (h->h0 ^ w) * UINT64_C(0x9E3779B97F4A7C15); h->h0 ^= h->h0 >> 29;
    h->h1 = (h->h1 + w) * UINT64_C(0xBF58476D1CE4E5B9); h->h1 ^= h->h1 >> 31; h->h1 += UINT64_C(0x94D049BB133111EB);
}
static void h_init(hasher *h, uint64_t domain) {
    h->h0 = UINT64_C(0x243F6A8885A308D3) ^ domain; h->h1 = UINT64_C(0x13198A2E03707344) + domain;
    h->word = h->len = 0; h->nbytes = 0;
}
static void h_word(hasher *h, uint64_t w) { h->len += 8; h_mix(h, w); }
static void h_bytes(hasher *h, const uint8_t *p, uint64_t len) {
    for (uint64_t i = 0; i < len; i++) {
        h->word |= (uint64_t)p[i] << (8 * h->nbytes);
        if (++h->nbytes == 8) { h_mix(h, h->word); h->word = 0; h->nbytes = 0; }
    }
    h->len += len;
}
static void h_final(hasher *h, uint64_t out[2]) {
    if (h->nbytes) h_mix(h, h->word);
    h_mix(h, h->len);
    out[0] = h->h0 ^ (h->h1 >> 17); out[1] = h->h1 ^ (h->h0 << 13);
}

/* canonical little-endian bytes */
typedef struct { uint8_t *p; uint64_t cap, len; int fail; } wbuf;
static void w_u64(wbuf *w, uint64_t v) {
    if (w->p) {
        if (w->len + 8 > w->cap) w->fail = 1;
        else for (int i = 0; i < 8; i++) w->p[w->len + (uint64_t)i] = (uint8_t)(v >> (8 * i));
    }
    w->len += 8;
}
static void w_i64(wbuf *w, int64_t v) { w_u64(w, (uint64_t)v); }
static void w_raw(wbuf *w, const void *src, uint64_t len) {   /* padded to 8 bytes with zeros */
    uint64_t padded = (len + 7) & ~(uint64_t)7;
    if (w->p) {
        if (w->len + padded > w->cap) w->fail = 1;
        else {
            if (len) memcpy(w->p + w->len, src, (size_t)len);
            for (uint64_t i = len; i < padded; i++) w->p[w->len + i] = 0;
        }
    }
    w->len += padded;
}
static void w_words(wbuf *w, const uint64_t *src, uint64_t n) { for (uint64_t i = 0; i < n; i++) w_u64(w, src[i]); }
typedef struct { const uint8_t *p; uint64_t len, at; int fail; } rbuf;
static uint64_t r_u64(rbuf *r) {
    if (r->fail || r->len - r->at < 8 || r->at > r->len) { r->fail = 1; return 0; }
    uint64_t v = 0;
    for (int i = 0; i < 8; i++) v |= (uint64_t)r->p[r->at + (uint64_t)i] << (8 * i);
    r->at += 8;
    return v;
}
static int64_t r_i64(rbuf *r) { return (int64_t)r_u64(r); }
static int32_t r_i32(rbuf *r) {
    int64_t v = r_i64(r);
    if (v < INT32_MIN || v > INT32_MAX) { r->fail = 1; return 0; }
    return (int32_t)v;
}
static const uint8_t *r_raw(rbuf *r, uint64_t len) {
    uint64_t padded = (len + 7) & ~(uint64_t)7;
    if (r->fail || padded < len || r->len - r->at < padded || r->at > r->len) { r->fail = 1; return NULL; }
    const uint8_t *p = r->p + r->at;
    for (uint64_t i = len; i < padded; i++) if (p[i]) { r->fail = 1; return NULL; }   /* canonical padding */
    r->at += padded;
    return p;
}

/* ------------------------------------------------------------------ polynomials */
static int exp_cmp(const uint8_t *a, const uint8_t *b, int32_t n) {
    for (int32_t i = 0; i < n; i++) if (a[i] != b[i]) return a[i] < b[i] ? -1 : 1;
    return 0;
}
typedef struct { const uint8_t *e; int32_t n; uint64_t index; } poly_ref;
/* qsort with a context-free comparison: the row width travels in each reference */
static int poly_ref_cmp(const void *x, const void *y) {
    const poly_ref *a = (const poly_ref *)x, *b = (const poly_ref *)y;
    int c = exp_cmp(a->e, b->e, a->n);
    return c ? c : (a->index < b->index ? -1 : a->index > b->index);
}
static uint8_t fermat_exp(uint8_t e, uint64_t p) { return e ? (uint8_t)(((uint64_t)(e - 1) % (p - 1)) + 1) : 0; }

int32_t tr_cert_poly_canonical(int32_t ncols, uint64_t nterms, const uint8_t *exps, const int64_t *coefs,
                               uint64_t field, int32_t fermat, uint8_t *out_e, int64_t *out_c, uint64_t *out_terms) {
    if (ncols < 1 || ncols > TR_CERT_MAX_COLS || !out_terms || (nterms && (!exps || !coefs || !out_e || !out_c)) ||
        nterms > TR_CERT_MAX_TERMS || (fermat && field < 2)) return nterms > TR_CERT_MAX_TERMS ? TR_MATH_UNSUPPORTED : TR_MATH_INVALID;
    if (field && (!tr_gfp_is_prime(field) || field >= TR_GFP_MAX)) return TR_MATH_INVALID;
    if (!nterms) { *out_terms = 0; return TR_MATH_OK; }
    uint8_t *e = zalloc(nterms * (uint64_t)ncols);
    poly_ref *refs = zalloc(nterms * sizeof *refs);
    if (!e || !refs) { free(e); free(refs); return TR_MATH_MEMORY; }
    for (uint64_t t = 0; t < nterms; t++) {
        for (int32_t v = 0; v < ncols; v++) {
            uint8_t x = exps[t * (uint64_t)ncols + (uint64_t)v];
            e[t * (uint64_t)ncols + (uint64_t)v] = fermat ? fermat_exp(x, field) : x;
        }
        refs[t].e = e + t * (uint64_t)ncols; refs[t].n = ncols; refs[t].index = t;
    }
    qsort(refs, (size_t)nterms, sizeof *refs, poly_ref_cmp);
    uint64_t m = 0;
    int32_t rc = TR_MATH_OK;
    for (uint64_t i = 0; i < nterms && !rc;) {
        __int128 sum = 0;
        uint64_t j = i;
        for (; j < nterms && exp_cmp(refs[i].e, refs[j].e, ncols) == 0; j++) {
            int64_t c = coefs[refs[j].index];
            if (field) {
                uint64_t r = tr_abs_u64(c) % field;
                sum += c < 0 ? -(__int128)r : (__int128)r;
                sum %= (__int128)field;
            } else sum += c;
        }
        if (field) {
            sum %= (__int128)field;
            if (sum < 0) sum += field;
            if (sum > (__int128)(field / 2)) sum -= field;          /* the symmetric representative */
        }
        if (sum < INT64_MIN || sum > INT64_MAX) rc = TR_MATH_OVERFLOW;
        else if (sum) {
            memcpy(out_e + m * (uint64_t)ncols, refs[i].e, (size_t)ncols);
            out_c[m++] = (int64_t)sum;
        }
        i = j;
    }
    free(e); free(refs);
    if (!rc) *out_terms = m;
    return rc;
}

static int poly_degree(const tr_cert_claim *c) {
    int top = 0;
    for (uint64_t t = 0; t < c->nterms; t++) {
        int d = 0;
        for (int32_t v = 0; v < c->ncols; v++) d += c->exponents[t * (uint64_t)c->ncols + (uint64_t)v];
        if (d > top) top = d;
    }
    return top;
}

/* ------------------------------------------------------------------ claims */
static uint64_t claim_payload_bytes(const tr_cert_claim *c) {
    switch (c->kind) {
    case TR_CLAIM_PARITY: return tr_budget_mul((uint64_t)c->nbits, sizeof(tr_cert_bit));
    case TR_CLAIM_POLYNOMIAL: return tr_budget_mul(c->nterms, tr_budget_add((uint64_t)c->ncols, 8));
    case TR_CLAIM_REFINEMENT: return tr_budget_mul(c->nvalues, 16);
    case TR_CLAIM_COVERAGE: return tr_budget_mul(c->nvalues, 8);
    case TR_CLAIM_REFERENCE: return tr_budget_bitmap_bytes(c->nvalues);
    default: return 0;
    }
}
static int strictly_increasing(const int64_t *k, uint64_t n) {
    for (uint64_t i = 1; i < n; i++) if (k[i] <= k[i - 1]) return 0;
    return 1;
}
/* scope_cols: the scope's columns, or -1 for a law; canonical: a polynomial must be canonical */
static int32_t claim_check(tr_cert_claim *c, int32_t scope_cols, int canonical, tr_math_need *need) {
    if (need) need_zero(need);
    if (!c || c->kind < 1 || c->kind >= TR_CLAIM_KINDS) return TR_MATH_INVALID;
    const tr_cert_claim_desc *d = &CLAIMS[c->kind - 1];
    if (c->policy < TR_CERT_NULL_FORBID || c->policy > TR_CERT_NULL_PROPAGATE || !((d->policies >> c->policy) & 1) ||
        c->ncols < d->min_cols || c->ncols > d->max_cols) return TR_MATH_INVALID;
    for (int32_t i = 0; i < c->ncols; i++) {
        if (c->col[i] < 0 || (scope_cols >= 0 && c->col[i] >= scope_cols) || c->col[i] >= TR_SCOPE_MAX_COLUMNS)
            return TR_MATH_INVALID;
        for (int32_t j = 0; j < i; j++) {
            int same = c->col[j] == c->col[i];
            if (same && !((c->kind == TR_CLAIM_SUM || c->kind == TR_CLAIM_PRODUCT) && j >= 1)) return TR_MATH_INVALID;
        }
    }
    for (int32_t i = c->ncols; i < TR_CERT_MAX_COLS; i++) c->col[i] = -1;
    if (c->kind == TR_CLAIM_PARITY) { c->field = 2; c->carrier = TR_CARRIER_GFP; }
    if (c->carrier != TR_CARRIER_I128 && c->carrier != TR_CARRIER_GFP) return TR_MATH_INVALID;
    memset(&c->gf, 0, sizeof c->gf);
    if (c->field && (!tr_gfp_is_prime(c->field) || c->field >= TR_GFP_MAX)) return TR_MATH_INVALID;
    if (c->field && c->carrier != TR_CARRIER_GFP) return TR_MATH_CARRIER_ERROR;
    if (c->tolerance && (c->carrier != TR_CARRIER_I128 || c->field || !(c->kind == TR_CLAIM_AFFINE ||
        c->kind == TR_CLAIM_SUM || c->kind == TR_CLAIM_PRODUCT || c->kind == TR_CLAIM_POLYNOMIAL))) return TR_MATH_INVALID;
    if (c->carrier == TR_CARRIER_GFP && !(c->kind == TR_CLAIM_AFFINE || c->kind == TR_CLAIM_SUM ||
        c->kind == TR_CLAIM_PRODUCT || c->kind == TR_CLAIM_POLYNOMIAL || c->kind == TR_CLAIM_PARITY)) return TR_MATH_INVALID;
    if (c->kind != TR_CLAIM_AFFINE) { c->scale = 0; c->bias = 0; }
    if (c->kind != TR_CLAIM_PARITY) { c->mask = 0; c->nbits = 0; c->constant = 0; c->bits = NULL; }
    if (c->kind != TR_CLAIM_POLYNOMIAL) { c->nterms = 0; c->exponents = NULL; c->coefficients = NULL; }
    if (c->kind != TR_CLAIM_REFINEMENT && c->kind != TR_CLAIM_COVERAGE && c->kind != TR_CLAIM_REFERENCE) c->nvalues = 0;
    if (c->kind != TR_CLAIM_REFINEMENT && c->kind != TR_CLAIM_COVERAGE) c->keys = NULL;
    if (c->kind != TR_CLAIM_REFINEMENT) c->targets = NULL;
    if (c->kind != TR_CLAIM_REFERENCE) c->used = NULL;
    if (!c->nvalues) { c->keys = NULL; c->targets = NULL; c->used = NULL; }
    uint64_t support = 0;
    int degree = 0;
    switch (c->kind) {
    case TR_CLAIM_PARITY:
        if (c->nbits < 1 || c->nbits > TR_CERT_MAX_BITS || !c->bits || !c->mask || (c->constant != 0 && c->constant != 1) ||
            (c->nbits < 64 && (c->mask >> c->nbits))) return TR_MATH_INVALID;
        for (int32_t j = 0; j < c->nbits; j++) {
            const tr_cert_bit *b = &c->bits[j];
            if (b->col < 0 || b->col >= c->ncols || (b->kind != TR_CERT_BIT_NONZERO && b->kind != TR_CERT_BIT_PLANE) ||
                b->reserved || (b->kind == TR_CERT_BIT_PLANE && (b->bit < 0 || b->bit > 63)) ||
                (b->kind == TR_CERT_BIT_NONZERO && (b->bit || b->base))) return TR_MATH_INVALID;
        }
        support = (uint64_t)c->nbits;
        break;
    case TR_CLAIM_REFINEMENT:
        if (c->nvalues && (!c->keys || !c->targets || !strictly_increasing(c->keys, c->nvalues))) return TR_MATH_INVALID;
        support = c->nvalues;
        break;
    case TR_CLAIM_COVERAGE:
        if (c->nvalues > TR_CERT_MAX_REQUIRED) return TR_MATH_UNSUPPORTED;
        if (c->nvalues && (!c->keys || !strictly_increasing(c->keys, c->nvalues))) return TR_MATH_INVALID;
        support = c->nvalues;
        break;
    case TR_CLAIM_POLYNOMIAL:
        if (c->nterms > TR_CERT_MAX_TERMS) return TR_MATH_UNSUPPORTED;
        if (c->nterms && (!c->exponents || !c->coefficients)) return TR_MATH_INVALID;
        if (canonical)
            for (uint64_t t = 0; t < c->nterms; t++) {
                if (!c->coefficients[t]) return TR_MATH_INVALID;
                if (t && exp_cmp(c->exponents + (t - 1) * (uint64_t)c->ncols, c->exponents + t * (uint64_t)c->ncols,
                                 c->ncols) >= 0) return TR_MATH_INVALID;
            }
        support = c->nterms;
        degree = poly_degree(c);
        break;
    case TR_CLAIM_REFERENCE:
        if (c->nvalues > (UINT64_C(1) << 32)) return TR_MATH_UNSUPPORTED;
        if (c->nvalues && !c->used) return TR_MATH_INVALID;
        if (c->nvalues % 64 && (c->used[c->nvalues / 64] & ~tail_mask(c->nvalues, c->nvalues / 64))) return TR_MATH_INVALID;
        support = (c->nvalues + 63) / 64;
        break;
    default:
        break;
    }
    if (need) {
        need->need[TR_BUDGET_VARIABLES] = (uint64_t)c->ncols;
        need->need[TR_BUDGET_DEGREE] = (uint64_t)degree;
        need->need[TR_BUDGET_SUPPORT] = support;
        need->need[TR_BUDGET_BYTES] = claim_payload_bytes(c);
    }
    return TR_MATH_OK;
}
/* the modulus of a GFP check travels in gf.p on input (claim->field for a GF(p) claim) */
static int32_t claim_field(tr_cert_claim *c, uint64_t modulus) {
    memset(&c->gf, 0, sizeof c->gf);
    if (c->carrier != TR_CARRIER_GFP) return TR_MATH_OK;
    uint64_t p = c->kind == TR_CLAIM_PARITY ? 2 : modulus;
    if (c->field && p != c->field) return TR_MATH_CARRIER_ERROR;
    int32_t rc = tr_gfp_field(p, &c->gf);
    return rc == TR_MATH_UNSUPPORTED ? rc : (rc ? TR_MATH_INVALID : TR_MATH_OK);
}
int32_t tr_cert_claim_check(tr_cert_claim *c, int32_t ncols, tr_math_need *need) {
    if (!c) return TR_MATH_INVALID;
    uint64_t modulus = c->gf.p ? c->gf.p : c->field;
    int32_t rc = claim_check(c, ncols, ncols >= 0, need);
    if (rc) return rc;
    return claim_field(c, modulus);
}
static int claim_exact(const tr_cert_claim *c) {
    return c->tolerance == 0 && ((c->carrier == TR_CARRIER_I128 && c->field == 0) ||
                                 (c->carrier == TR_CARRIER_GFP && c->gf.p == c->field && c->field));
}
/* the distinct scope columns a claim reads, increasing */
static int32_t claim_columns(const tr_cert_claim *c, int32_t *out) {
    int32_t n = 0;
    for (int32_t i = 0; i < c->ncols; i++) {
        int seen = 0;
        for (int32_t j = 0; j < n; j++) seen |= out[j] == c->col[i];
        if (!seen) out[n++] = c->col[i];
    }
    for (int32_t i = 1; i < n; i++)
        for (int32_t j = i; j > 0 && out[j - 1] > out[j]; j--) { int32_t t = out[j]; out[j] = out[j - 1]; out[j - 1] = t; }
    return n;
}

/* ------------------------------------------------------------------ dimensions */
static int cmp_u32(const void *a, const void *b) {
    uint32_t x = *(const uint32_t *)a, y = *(const uint32_t *)b;
    return x < y ? -1 : x > y;
}
int32_t tr_cert_dim_describe(const struct atable_s *dim, const uint8_t *identity, const uint64_t *used, int32_t key,
                             int32_t key_valid, const tr_math_budget *budget, tr_math_refusal *refusal, tr_cert_dim *out) {
    tr_budget_none(refusal);
    const atable_t *t = dim;
    if (!t || !out || !identity || key < 0 || key >= t->k || key_valid < -1 || key_valid >= t->k || key == key_valid)
        return TR_MATH_INVALID;
    if (t->vflag[key] || (key_valid >= 0 && t->vflag[key_valid])) return TR_MATH_UNSUPPORTED;
    uint64_t n = t->n, nw = (n + 63) / 64;
    tr_math_need need;
    need_zero(&need);
    need.need[TR_BUDGET_BYTES] = tr_budget_add(tr_budget_mul(n, key_valid >= 0 ? 12 : 8), 64);
    tr_math_budget dflt;
    int32_t rc = tr_budget_check(budget_or(budget, &dflt), &need, refusal);
    if (rc) return rc;
    uint32_t *codes = zalloc(n * 4), *valid = key_valid >= 0 ? zalloc(n * 4) : NULL, *keys = zalloc(n * 4);
    if (!codes || !keys || (key_valid >= 0 && !valid)) { free(codes); free(valid); free(keys); return TR_MATH_MEMORY; }
    if (n) at_decode_column(t, key, codes);
    if (n && valid) at_decode_column(t, key_valid, valid);
    tr_cert_dim d;
    memset(&d, 0, sizeof d);
    memcpy(d.identity, identity, 16);
    d.revision = t->revision; d.rows = n; d.key = key; d.key_valid = key_valid;
    hasher h;
    h_init(&h, 0x44494d);
    h_word(&h, n);
    uint64_t m = 0;
    for (uint64_t w = 0; w < nw; w++) {
        uint64_t bits = (used ? used[w] : UINT64_MAX) & tail_mask(n, w);
        h_word(&h, bits);
        d.used += (uint64_t)__builtin_popcountll(bits);
        while (bits) {
            uint64_t r = w * 64 + (uint64_t)__builtin_ctzll(bits);
            bits &= bits - 1;
            if (valid && !valid[r]) { d.nulls++; continue; }
            keys[m++] = codes[r];
        }
    }
    h_final(&h, d.used_digest);
    if (m > 1) qsort(keys, (size_t)m, 4, cmp_u32);
    for (uint64_t i = 1; i < m; i++) d.duplicates += keys[i] == keys[i - 1];
    d.unique = d.duplicates == 0 && d.nulls == 0;
    free(codes); free(valid); free(keys);
    *out = d;
    return TR_MATH_OK;
}

/* ------------------------------------------------------------------ records */
struct tr_cert_record_s {
    uint32_t magic;
    tr_cert_source src;
    uint64_t *selection;        /* TR_CERT_RECORD_SELECTION: ceil(rows / 64) words */
    uint8_t *grouping;          /* TR_CERT_RECORD_GROUPS: src.grouping_bytes */
    uint64_t bytes;
};
static int record_valid(const tr_cert_record *r) { return r && r->magic == RECORD_MAGIC; }
void tr_cert_record_free(tr_cert_record *r) {
    if (!record_valid(r)) return;
    free(r->selection); free(r->grouping);
    r->magic = 0;
    free(r);
}
uint64_t tr_cert_record_bytes(const tr_cert_record *r) { return record_valid(r) ? r->bytes : 0; }

static void null_domain(const tr_scope_view *v, int32_t c, tr_cert_column *out) {
    hasher h;
    h_init(&h, 0x4e554c4c);
    out->nulls = 0;
    const uint8_t *valid = v->col[c].valid;
    if (valid)
        for (uint64_t w = 0; w < (v->rows + 63) / 64; w++) {
            uint64_t bits = v->selection[w];
            while (bits) {
                uint64_t r = w * 64 + (uint64_t)__builtin_ctzll(bits);
                bits &= bits - 1;
                if (!valid[r]) { out->nulls++; h_word(&h, r); }
            }
        }
    h_final(&h, out->null_digest);
}
/* an array scope's data: the selected present values of the recorded columns */
static void data_digest(const tr_scope_view *v, const tr_cert_source *s, uint64_t out[2]) {
    hasher h;
    h_init(&h, 0x44415441);
    for (int32_t i = 0; i < s->ncols; i++) {
        int32_t c = s->col[i].index;
        h_word(&h, (uint64_t)c);
        if (c < 0) continue;
        for (uint64_t w = 0; w < (v->rows + 63) / 64; w++) {
            uint64_t bits = v->selection[w];
            while (bits) {
                uint64_t r = w * 64 + (uint64_t)__builtin_ctzll(bits);
                bits &= bits - 1;
                if (tr_scope_present(v, c, r)) { h_word(&h, r); h_word(&h, (uint64_t)tr_scope_value(v, c, r)); }
            }
        }
    }
    h_final(&h, out);
}
/* cols: indices (an index outside the scope is recorded as missing: index -1 - original) */
static tr_cert_record *record_build(const tr_scope *scope, const int32_t *cols, int32_t ncols, int32_t flags,
                                    const tr_cert_dim *dims, int32_t ndims, const tr_math_budget *budget,
                                    tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    tr_scope_info info;
    tr_scope_view view;
    if (!scope || tr_scope_get_info(scope, &info) || tr_scope_get_view(scope, &view) || (flags & ~3) ||
        ndims < 0 || ndims > TR_CERT_MAX_DIMS || (ndims && !dims) || ncols < -1 || ncols > TR_SCOPE_MAX_COLUMNS ||
        (ncols > 0 && !cols)) { set_error(error, TR_MATH_INVALID); return NULL; }
    if (ncols == -1) ncols = info.ncols;
    uint64_t nw = (info.rows + 63) / 64;
    uint64_t gbytes = (flags & TR_CERT_RECORD_GROUPS) ? tr_scope_key_size(scope, TR_SCOPE_KEY_GROUPS) : 0;
    tr_math_need need;
    need_zero(&need);
    uint64_t bytes = sizeof(tr_cert_record);
    if (flags & TR_CERT_RECORD_SELECTION) bytes = tr_budget_add(bytes, nw * 8);
    bytes = tr_budget_add(bytes, gbytes);
    need.need[TR_BUDGET_BYTES] = bytes;
    tr_math_budget dflt;
    int32_t rc = tr_budget_check(budget_or(budget, &dflt), &need, refusal);
    if (rc) { set_error(error, rc); return NULL; }
    tr_cert_record *r = zalloc(sizeof *r);
    if (!r) { set_error(error, TR_MATH_MEMORY); return NULL; }
    r->magic = RECORD_MAGIC;
    tr_cert_source *s = &r->src;
    memcpy(s->identity, info.identity, 16);
    s->revision = info.revision; s->rows = info.rows; s->selected = info.selected; s->ngroups = info.ngroups;
    s->version[0] = info.version[0]; s->version[1] = info.version[1];
    s->source = info.source; s->ncols = ncols; s->ndims = ndims; s->flags = flags;
    hasher h;
    h_init(&h, 0x53454c);
    h_word(&h, info.rows);
    for (uint64_t w = 0; w < nw; w++) h_word(&h, view.selection[w]);
    h_final(&h, s->selection);
    if (flags & TR_CERT_RECORD_SELECTION) {
        r->selection = zalloc(nw * 8);
        if (!r->selection) { tr_cert_record_free(r); set_error(error, TR_MATH_MEMORY); return NULL; }
        if (nw) memcpy(r->selection, view.selection, (size_t)nw * 8);
    }
    if (flags & TR_CERT_RECORD_GROUPS) {
        r->grouping = zalloc(gbytes);
        if (!r->grouping || tr_scope_key_write(scope, TR_SCOPE_KEY_GROUPS, r->grouping, gbytes)) {
            tr_cert_record_free(r); set_error(error, TR_MATH_MEMORY); return NULL;
        }
        s->grouping_bytes = gbytes;
        hasher g;
        h_init(&g, 0x475250);
        h_bytes(&g, r->grouping, gbytes);
        h_final(&g, s->grouping);
    }
    for (int32_t i = 0; i < ncols; i++) {
        tr_cert_column *k = &s->col[i];
        int32_t c = cols ? cols[i] : i;
        tr_scope_column_info ci;
        if (c < 0 || c >= info.ncols || tr_scope_column_get(scope, c, &ci)) {
            k->index = c < 0 ? c : -1 - c; k->col = k->valid = k->hi = -1;   /* missing: compared as COLUMNS */
            continue;
        }
        k->index = c; k->col = ci.col; k->valid = ci.valid; k->hi = ci.hi; k->shift = ci.shift; k->scale = ci.scale;
        k->storage = ci.storage; k->offset = ci.offset;
        null_domain(&view, c, k);
    }
    if (info.source == TR_SCOPE_ARRAYS) data_digest(&view, s, s->data);
    for (int32_t i = 0; i < ndims; i++) s->dim[i] = dims[i];
    r->bytes = bytes;
    set_error(error, TR_MATH_OK);
    return r;
}
tr_cert_record *tr_cert_record_scope(const tr_scope *scope, const int32_t *cols, int32_t ncols, int32_t flags,
                                     const tr_cert_dim *dims, int32_t ndims, const tr_math_budget *budget,
                                     tr_math_refusal *refusal, int32_t *error) {
    tr_scope_info info;
    if (!scope || tr_scope_get_info(scope, &info)) { tr_budget_none(refusal); set_error(error, TR_MATH_INVALID); return NULL; }
    for (int32_t i = 0; i < ncols; i++) {
        if (!cols || cols[i] < 0 || cols[i] >= info.ncols) { tr_budget_none(refusal); set_error(error, TR_MATH_INVALID); return NULL; }
        for (int32_t j = 0; j < i; j++) if (cols[j] == cols[i]) { tr_budget_none(refusal); set_error(error, TR_MATH_INVALID); return NULL; }
    }
    return record_build(scope, cols, ncols, flags, dims, ndims, budget, refusal, error);
}
int32_t tr_cert_record_get(const tr_cert_record *r, tr_cert_source *out) {
    if (!record_valid(r) || !out) return TR_MATH_INVALID;
    *out = r->src;
    return TR_MATH_OK;
}
int32_t tr_cert_record_selection(const tr_cert_record *r, uint64_t *out, uint64_t capacity) {
    if (!record_valid(r) || (!out && capacity)) return TR_MATH_INVALID;
    if (!r->selection) return TR_MATH_UNSUPPORTED;
    uint64_t nw = (r->src.rows + 63) / 64;
    if (capacity < nw) return TR_MATH_INVALID;
    if (nw) memcpy(out, r->selection, (size_t)nw * 8);
    return TR_MATH_OK;
}

static void stale_add(tr_cert_staleness *s, uint32_t reason, int32_t column, int32_t dim, uint64_t recorded,
                      uint64_t current) {
    if (!(s->reasons & reason) && (!s->first || (int32_t)reason < s->first)) {
        s->first = (int32_t)reason; s->column = column; s->dim = dim; s->recorded = recorded; s->current = current;
    }
    s->reasons |= reason;
}
static int col_spec_equal(const tr_cert_column *a, const tr_cert_column *b) {
    return a->col == b->col && a->hi == b->hi && a->shift == b->shift && a->scale == b->scale &&
           a->storage == b->storage && a->offset == b->offset;
}
int32_t tr_cert_record_compare(const tr_cert_record *a, const tr_cert_record *b, int32_t mode, tr_cert_staleness *out) {
    if (!record_valid(a) || !record_valid(b) || !out || mode < TR_CERT_MATCH_EXACT || mode > TR_CERT_MATCH_SCHEMA)
        return TR_MATH_INVALID;
    memset(out, 0, sizeof *out);
    out->column = out->dim = -1;
    tr_cert_staleness s = *out;
    const tr_cert_source *x = &a->src, *y = &b->src;
    if (x->source != y->source || memcmp(x->identity, y->identity, 16) ||
        (x->source == TR_SCOPE_ARRAYS && (x->data[0] != y->data[0] || x->data[1] != y->data[1])))
        stale_add(&s, TR_CERT_STALE_IDENTITY, -1, -1, 0, 0);
    if (mode != TR_CERT_MATCH_SCHEMA) {
        if (x->revision != y->revision) stale_add(&s, TR_CERT_STALE_REVISION, -1, -1, x->revision, y->revision);
        else if (x->version[0] != y->version[0] || x->version[1] != y->version[1])
            stale_add(&s, TR_CERT_STALE_REVISION, -1, -1, x->revision, y->revision);
        if (x->rows != y->rows) stale_add(&s, TR_CERT_STALE_ROWS, -1, -1, x->rows, y->rows);
        if (x->rows == y->rows) {
            uint64_t nw = (x->rows + 63) / 64;
            int differ;
            if (mode == TR_CERT_MATCH_SUBSET && a->selection && b->selection) {
                differ = 0;
                for (uint64_t w = 0; w < nw && !differ; w++) differ = (b->selection[w] & ~a->selection[w]) != 0;
            } else if (a->selection && b->selection) {
                differ = memcmp(a->selection, b->selection, (size_t)nw * 8) != 0;
            } else differ = x->selection[0] != y->selection[0] || x->selection[1] != y->selection[1] ||
                            x->selected != y->selected;
            if (differ) stale_add(&s, TR_CERT_STALE_SELECTION, -1, -1, x->selected, y->selected);
        }
    }
    for (int32_t i = 0; i < x->ncols; i++) {
        const tr_cert_column *c = &x->col[i];
        const tr_cert_column *d = NULL;
        for (int32_t j = 0; j < y->ncols && !d; j++) if (y->col[j].index == c->index) d = &y->col[j];
        if (!d || c->index < 0) { stale_add(&s, TR_CERT_STALE_COLUMNS, c->index, -1, 0, 0); continue; }
        if (!col_spec_equal(c, d)) stale_add(&s, TR_CERT_STALE_COLUMNS, c->index, -1, (uint64_t)c->col, (uint64_t)d->col);
        if (c->valid != d->valid) stale_add(&s, TR_CERT_STALE_NULL_DOMAIN, c->index, -1, c->nulls, d->nulls);
        else if (mode == TR_CERT_MATCH_EXACT && (c->nulls != d->nulls || c->null_digest[0] != d->null_digest[0] ||
                                                 c->null_digest[1] != d->null_digest[1]))
            stale_add(&s, TR_CERT_STALE_NULL_DOMAIN, c->index, -1, c->nulls, d->nulls);
    }
    if (x->flags & TR_CERT_RECORD_GROUPS) {
        int same = (y->flags & TR_CERT_RECORD_GROUPS) && x->grouping_bytes == y->grouping_bytes &&
                   (!x->grouping_bytes || memcmp(a->grouping, b->grouping, (size_t)x->grouping_bytes) == 0);
        if (!same) stale_add(&s, TR_CERT_STALE_GROUPING, -1, -1, x->ngroups, y->ngroups);
    }
    for (int32_t i = 0; i < x->ndims; i++) {
        const tr_cert_dim *p = &x->dim[i], *q = NULL;
        for (int32_t j = 0; j < y->ndims && !q; j++)
            if (!memcmp(y->dim[j].identity, p->identity, 16) && y->dim[j].key == p->key) q = &y->dim[j];
        if (!q) { stale_add(&s, TR_CERT_STALE_JOIN_REVISION, -1, i, p->revision, 0); continue; }
        if (mode != TR_CERT_MATCH_SCHEMA && (p->revision != q->revision || p->rows != q->rows || p->used != q->used ||
                                             p->used_digest[0] != q->used_digest[0] || p->used_digest[1] != q->used_digest[1]))
            stale_add(&s, TR_CERT_STALE_JOIN_REVISION, -1, i, p->revision, q->revision);
        if (p->key_valid != q->key_valid || (p->unique && !q->unique) || (!p->nulls && q->nulls) ||
            (!p->duplicates && q->duplicates))
            stale_add(&s, TR_CERT_STALE_JOIN_KEY, -1, i, p->duplicates + p->nulls, q->duplicates + q->nulls);
    }
    *out = s;
    return s.reasons ? TR_MATH_STALE : TR_MATH_OK;
}
int32_t tr_cert_record_fresh(const tr_cert_record *r, const tr_scope *current, const tr_cert_dim *dims, int32_t ndims,
                             int32_t mode, tr_cert_staleness *out) {
    if (!record_valid(r) || !current || !out) return TR_MATH_INVALID;
    int32_t cols[TR_SCOPE_MAX_COLUMNS];
    for (int32_t i = 0; i < r->src.ncols; i++) cols[i] = r->src.col[i].index;
    int32_t error = 0;
    tr_cert_record *now = record_build(current, cols, r->src.ncols, r->src.flags, dims, ndims, NULL, NULL, &error);
    if (!now) return error;
    int32_t rc = tr_cert_record_compare(r, now, mode, out);
    tr_cert_record_free(now);
    return rc;
}

static void record_write(wbuf *w, const tr_cert_record *r) {
    const tr_cert_source *s = &r->src;
    uint64_t id[2];
    memcpy(id, s->identity, 16);
    w_u64(w, RECORD_TAG); w_u64(w, SERIAL_VERSION);
    w_u64(w, id[0]); w_u64(w, id[1]);
    w_u64(w, s->revision); w_u64(w, s->rows); w_u64(w, s->selected); w_u64(w, s->ngroups);
    w_words(w, s->version, 2); w_words(w, s->selection, 2); w_words(w, s->grouping, 2); w_words(w, s->data, 2);
    w_i64(w, s->source); w_i64(w, s->ncols); w_i64(w, s->ndims); w_i64(w, s->flags); w_u64(w, s->grouping_bytes);
    for (int32_t i = 0; i < s->ncols; i++) {
        const tr_cert_column *c = &s->col[i];
        w_i64(w, c->index); w_i64(w, c->col); w_i64(w, c->valid); w_i64(w, c->hi); w_i64(w, c->shift);
        w_i64(w, c->scale); w_i64(w, c->storage); w_i64(w, c->offset); w_u64(w, c->nulls); w_words(w, c->null_digest, 2);
    }
    for (int32_t i = 0; i < s->ndims; i++) {
        const tr_cert_dim *d = &s->dim[i];
        uint64_t di[2];
        memcpy(di, d->identity, 16);
        w_words(w, di, 2); w_u64(w, d->revision); w_u64(w, d->rows); w_u64(w, d->used); w_words(w, d->used_digest, 2);
        w_u64(w, d->duplicates); w_u64(w, d->nulls); w_i64(w, d->key); w_i64(w, d->key_valid); w_i64(w, d->unique);
    }
    if (r->selection) w_words(w, r->selection, (s->rows + 63) / 64);
    if (r->grouping) w_raw(w, r->grouping, s->grouping_bytes);
}
uint64_t tr_cert_record_serial_size(const tr_cert_record *r) {
    if (!record_valid(r)) return 0;
    wbuf w = {NULL, 0, 0, 0};
    record_write(&w, r);
    return w.len;
}
int32_t tr_cert_record_serialize(const tr_cert_record *r, uint8_t *out, uint64_t capacity, uint64_t *written) {
    if (!record_valid(r) || !out) return TR_MATH_INVALID;
    wbuf w = {out, capacity, 0, 0};
    record_write(&w, r);
    if (w.fail) return TR_MATH_INVALID;
    if (written) *written = w.len;
    return TR_MATH_OK;
}
static tr_cert_record *record_read(rbuf *rd, const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    if (r_u64(rd) != RECORD_TAG || r_u64(rd) != SERIAL_VERSION) { set_error(error, TR_MATH_INVALID); return NULL; }
    tr_cert_source s;
    memset(&s, 0, sizeof s);
    uint64_t id[2] = {r_u64(rd), r_u64(rd)};
    memcpy(s.identity, id, 16);
    s.revision = r_u64(rd); s.rows = r_u64(rd); s.selected = r_u64(rd); s.ngroups = r_u64(rd);
    for (int i = 0; i < 2; i++) s.version[i] = r_u64(rd);
    for (int i = 0; i < 2; i++) s.selection[i] = r_u64(rd);
    for (int i = 0; i < 2; i++) s.grouping[i] = r_u64(rd);
    for (int i = 0; i < 2; i++) s.data[i] = r_u64(rd);
    s.source = r_i32(rd); s.ncols = r_i32(rd); s.ndims = r_i32(rd); s.flags = r_i32(rd); s.grouping_bytes = r_u64(rd);
    if (rd->fail || (s.source != TR_SCOPE_TABLE && s.source != TR_SCOPE_ARRAYS) || s.ncols < 0 ||
        s.ncols > TR_SCOPE_MAX_COLUMNS || s.ndims < 0 || s.ndims > TR_CERT_MAX_DIMS || (s.flags & ~3) ||
        s.selected > s.rows || s.rows > (UINT64_MAX >> 8) || (!(s.flags & TR_CERT_RECORD_GROUPS) && s.grouping_bytes)) {
        set_error(error, TR_MATH_INVALID); return NULL;
    }
    for (int32_t i = 0; i < s.ncols; i++) {
        tr_cert_column *c = &s.col[i];
        c->index = r_i32(rd); c->col = r_i32(rd); c->valid = r_i32(rd); c->hi = r_i32(rd); c->shift = r_i32(rd);
        c->scale = r_i32(rd); c->storage = r_i32(rd); c->offset = r_i64(rd); c->nulls = r_u64(rd);
        c->null_digest[0] = r_u64(rd); c->null_digest[1] = r_u64(rd);
    }
    for (int32_t i = 0; i < s.ndims; i++) {
        tr_cert_dim *d = &s.dim[i];
        uint64_t di[2] = {r_u64(rd), r_u64(rd)};
        memcpy(d->identity, di, 16);
        d->revision = r_u64(rd); d->rows = r_u64(rd); d->used = r_u64(rd);
        d->used_digest[0] = r_u64(rd); d->used_digest[1] = r_u64(rd);
        d->duplicates = r_u64(rd); d->nulls = r_u64(rd); d->key = r_i32(rd); d->key_valid = r_i32(rd); d->unique = r_i32(rd);
        if (d->unique != (d->duplicates == 0 && d->nulls == 0)) rd->fail = 1;
    }
    if (rd->fail) { set_error(error, TR_MATH_INVALID); return NULL; }
    uint64_t nw = (s.rows + 63) / 64;
    tr_math_need need;
    need_zero(&need);
    uint64_t bytes = tr_budget_add(sizeof(tr_cert_record), tr_budget_add((s.flags & TR_CERT_RECORD_SELECTION) ? nw * 8 : 0,
                                                                         s.grouping_bytes));
    need.need[TR_BUDGET_BYTES] = bytes;
    tr_math_budget dflt;
    int32_t rc = tr_budget_check(budget_or(budget, &dflt), &need, refusal);
    if (rc) { set_error(error, rc); return NULL; }
    if (((s.flags & TR_CERT_RECORD_SELECTION) && rd->len - rd->at < nw * 8) || rd->at > rd->len) {
        set_error(error, TR_MATH_INVALID); return NULL;
    }
    tr_cert_record *r = zalloc(sizeof *r);
    if (!r) { set_error(error, TR_MATH_MEMORY); return NULL; }
    r->src = s;
    r->magic = RECORD_MAGIC;
    r->bytes = bytes;
    if (s.flags & TR_CERT_RECORD_SELECTION) {
        r->selection = zalloc(nw * 8);
        if (!r->selection) { tr_cert_record_free(r); set_error(error, TR_MATH_MEMORY); return NULL; }
        uint64_t count = 0;
        for (uint64_t w = 0; w < nw; w++) {
            r->selection[w] = r_u64(rd);
            if (r->selection[w] & ~tail_mask(s.rows, w)) rd->fail = 1;
            count += (uint64_t)__builtin_popcountll(r->selection[w]);
        }
        if (count != s.selected) rd->fail = 1;
    }
    if (s.flags & TR_CERT_RECORD_GROUPS) {
        const uint8_t *p = r_raw(rd, s.grouping_bytes);
        r->grouping = zalloc(s.grouping_bytes);
        if (!r->grouping) { tr_cert_record_free(r); set_error(error, TR_MATH_MEMORY); return NULL; }
        if (p && s.grouping_bytes) memcpy(r->grouping, p, (size_t)s.grouping_bytes);
    }
    if (rd->fail) { tr_cert_record_free(r); set_error(error, TR_MATH_INVALID); return NULL; }
    set_error(error, TR_MATH_OK);
    return r;
}
tr_cert_record *tr_cert_record_deserialize(const uint8_t *in, uint64_t bytes, uint64_t *consumed,
                                           const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error) {
    if (!in) { tr_budget_none(refusal); set_error(error, TR_MATH_INVALID); return NULL; }
    rbuf rd = {in, bytes, 0, 0};
    tr_cert_record *r = record_read(&rd, budget, refusal, error);
    if (!r) return NULL;
    uint64_t size = tr_cert_record_serial_size(r);    /* canonical bytes only */
    uint8_t *again = size == rd.at ? zalloc(size) : NULL;
    int same = again && !tr_cert_record_serialize(r, again, size, NULL) && memcmp(again, in, (size_t)size) == 0;
    free(again);
    if (!same) { tr_cert_record_free(r); set_error(error, TR_MATH_INVALID); return NULL; }
    if (consumed) *consumed = rd.at;
    return r;
}
tr_cert_record *tr_cert_record_clone(const tr_cert_record *r, int32_t *error) {
    if (!record_valid(r)) { set_error(error, TR_MATH_INVALID); return NULL; }
    tr_cert_record *c = zalloc(sizeof *c);
    if (!c) { set_error(error, TR_MATH_MEMORY); return NULL; }
    *c = *r;
    c->selection = NULL; c->grouping = NULL;
    uint64_t nw = (r->src.rows + 63) / 64;
    if (r->selection && !(c->selection = zalloc(nw * 8))) { tr_cert_record_free(c); set_error(error, TR_MATH_MEMORY); return NULL; }
    if (r->selection && nw) memcpy(c->selection, r->selection, (size_t)nw * 8);
    if (r->grouping && !(c->grouping = zalloc(r->src.grouping_bytes))) { tr_cert_record_free(c); set_error(error, TR_MATH_MEMORY); return NULL; }
    if (r->grouping && r->src.grouping_bytes) memcpy(c->grouping, r->grouping, (size_t)r->src.grouping_bytes);
    set_error(error, TR_MATH_OK);
    return c;
}

tr_cert_record *tr_cert_record_copy(const tr_cert_record *r, int32_t flags, int32_t *error) {
    if (!record_valid(r) || (flags & ~r->src.flags)) { set_error(error, TR_MATH_INVALID); return NULL; }
    tr_cert_record *c = tr_cert_record_clone(r, error);
    if (!c) return NULL;
    if (!(flags & TR_CERT_RECORD_SELECTION) && c->selection) {
        free(c->selection); c->selection = NULL;
        c->bytes -= (r->src.rows + 63) / 64 * 8;
    }
    if (!(flags & TR_CERT_RECORD_GROUPS) && c->grouping) {
        free(c->grouping); c->grouping = NULL;
        c->bytes -= c->src.grouping_bytes;
        c->src.grouping_bytes = 0;
        c->src.grouping[0] = c->src.grouping[1] = 0;
    }
    c->src.flags = flags;
    set_error(error, TR_MATH_OK);
    return c;
}

/* ------------------------------------------------------------------ verification on the CPU */
typedef struct { int64_t value; uint64_t row; } uniq_entry;
static int uniq_cmp(const void *a, const void *b) {
    const uniq_entry *x = (const uniq_entry *)a, *y = (const uniq_entry *)b;
    if (x->value != y->value) return x->value < y->value ? -1 : 1;
    return x->row < y->row ? -1 : x->row > y->row;
}
static void ev_init(tr_cert_evidence *e, const tr_scope_view *v, const tr_cert_claim *c) {
    memset(e, 0, sizeof *e);
    e->rows = v->rows; e->selected = v->selected; e->witness = TR_CERT_NONE;
    e->unit = CLAIMS[c->kind - 1].unit;
}
static void ev_violate(tr_cert_evidence *e, uint64_t at) { e->violations++; if (at < e->witness) e->witness = at; }
static void ev_finish(tr_cert_evidence *e) { e->holds = !e->overflow && !e->violations; }

static int32_t verify_one(const tr_scope_view *v, const tr_cert_claim *c, tr_cert_evidence *e) {
    ev_init(e, v, c);
    uint64_t nw = (v->rows + 63) / 64;
    uint64_t *masks = NULL, *present = NULL;
    uniq_entry *u = NULL;
    uint64_t nu = 0;
    if (c->kind == TR_CLAIM_COVERAGE) {
        masks = zalloc(v->ngroups * 8);
        present = zalloc(tr_budget_bitmap_bytes(v->ngroups));
        if (!masks || !present) { free(masks); free(present); return TR_MATH_MEMORY; }
    }
    if (c->kind == TR_CLAIM_UNIQUE) {
        u = zalloc(v->selected * sizeof *u);
        if (!u) return TR_MATH_MEMORY;
    }
    for (uint64_t w = 0; w < nw; w++) {
        uint64_t bits = v->selection[w];
        while (bits) {
            uint64_t r = w * 64 + (uint64_t)__builtin_ctzll(bits), bit = 0;
            bits &= bits - 1;
            int s = tr_cert_row(v, c, r, &bit);
            if (c->kind == TR_CLAIM_COVERAGE) {                  /* a group with a selected row is present */
                uint32_t g = tr_scope_group(v, r);
                present[g >> 6] |= UINT64_C(1) << (g & 63);
                if (s == TR_CERT_ROW_EXCLUDED) e->excluded++;
                else masks[g] |= bit;
                continue;
            }
            if (s == TR_CERT_ROW_EXCLUDED) { e->excluded++; continue; }
            if (s == TR_CERT_ROW_OVERFLOW) { e->overflow++; continue; }
            e->checked++;
            if (s == TR_CERT_ROW_VIOLATES) { ev_violate(e, r); continue; }
            if (c->kind == TR_CLAIM_UNIQUE) { u[nu].value = tr_scope_value(v, c->col[0], r); u[nu].row = r; nu++; }
        }
    }
    if (c->kind == TR_CLAIM_COVERAGE) {
        uint64_t full = tr_cert_full_mask(c->nvalues);
        for (uint64_t g = 0; g < v->ngroups; g++) {
            if (!((present[g >> 6] >> (g & 63)) & 1)) continue;
            e->checked++;
            if ((masks[g] & full) != full) ev_violate(e, g);
        }
    }
    if (c->kind == TR_CLAIM_UNIQUE && nu > 1) {
        qsort(u, (size_t)nu, sizeof *u, uniq_cmp);
        for (uint64_t i = 1; i < nu; i++) if (u[i].value == u[i - 1].value) ev_violate(e, u[i].row);
    }
    free(masks); free(present); free(u);
    ev_finish(e);
    return TR_MATH_OK;
}
static int view_ok(const tr_scope_view *v) {
    return v && v->ngroups >= 1 && (!v->rows || v->selection) && (!v->rows || v->ngroups == 1 || v->groups) &&
           v->ncols >= 0 && v->ncols <= TR_SCOPE_MAX_COLUMNS;
}
int32_t tr_cert_verify_need(const tr_scope_view *v, const tr_cert_claim *claims, int32_t n, int32_t target,
                            tr_math_need *need) {
    if (!view_ok(v) || !need || n < 0 || (n && !claims) || (target != TR_CERT_NEED_CPU && target != TR_CERT_NEED_DEVICE))
        return TR_MATH_INVALID;
    if (target == TR_CERT_NEED_DEVICE) { tr_cert_device_need(v, claims, n, need); return TR_MATH_OK; }
    need_zero(need);
    uint64_t bytes = 0, states = 0;
    for (int32_t j = 0; j < n; j++) {
        tr_cert_claim c = claims[j];
        tr_math_need cn;
        int32_t rc = claim_check(&c, v->ncols, 1, &cn);
        if (rc) return rc;
        need_max(need, &cn);
        uint64_t scratch = 0;
        if (c.kind == TR_CLAIM_COVERAGE) {
            scratch = tr_budget_add(tr_budget_mul(v->ngroups, 8), tr_budget_bitmap_bytes(v->ngroups));
            states = tr_budget_add(states, v->ngroups);
        }
        if (c.kind == TR_CLAIM_UNIQUE) scratch = tr_budget_mul(v->selected, sizeof(uniq_entry));
        if (scratch > bytes) bytes = scratch;     /* one claim's scratch lives at a time */
    }
    need->need[TR_BUDGET_BYTES] = bytes;
    need->need[TR_BUDGET_STATES] = states;
    need->need[TR_BUDGET_OUTPUT] = (uint64_t)n;
    return TR_MATH_OK;
}
int32_t tr_cert_verify_view(const tr_scope_view *v, const tr_cert_claim *claims, int32_t n, const tr_math_budget *budget,
                            tr_cert_evidence *out, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!view_ok(v) || n < 0 || (n && (!claims || !out))) return TR_MATH_INVALID;
    for (int32_t j = 0; j < n; j++) {
        tr_cert_claim c = claims[j];
        int32_t rc = claim_check(&c, v->ncols, 1, NULL);
        if (rc) return rc;
        if (c.carrier == TR_CARRIER_GFP && (claims[j].gf.p < 2 || (c.field && claims[j].gf.p != c.field) ||
                                            (c.kind == TR_CLAIM_PARITY && claims[j].gf.p != 2))) return TR_MATH_INVALID;
        for (int32_t i = 0; i < c.ncols; i++)
            if (v->rows && !v->col[c.col[i]].codes && !v->col[c.col[i]].values) return TR_MATH_INVALID;
    }
    tr_math_need need;
    int32_t rc = tr_cert_verify_need(v, claims, n, TR_CERT_NEED_CPU, &need);
    if (rc) return rc;
    tr_math_budget dflt;
    rc = tr_budget_check(budget_or(budget, &dflt), &need, refusal);
    if (rc) return rc;
    for (int32_t j = 0; j < n; j++) {
        tr_cert_claim c = claims[j];
        claim_check(&c, v->ncols, 1, NULL);
        c.gf = claims[j].gf;
        rc = verify_one(v, &c, &out[j]);
        if (rc) return rc;
    }
    return TR_MATH_OK;
}
/* checked copies of the claims (polynomials canonicalized) in owned arrays */
typedef struct { tr_cert_claim claim; uint8_t *e; int64_t *c; } claim_copy;
static void copy_free(claim_copy *k) { free(k->e); free(k->c); k->e = NULL; k->c = NULL; }
static int32_t copy_checked(const tr_cert_claim *in, int32_t ncols, claim_copy *k, tr_math_need *need) {
    memset(k, 0, sizeof *k);
    k->claim = *in;
    tr_cert_claim *c = &k->claim;
    if (c->kind == TR_CLAIM_POLYNOMIAL && c->nterms) {
        if (c->ncols < 1 || c->ncols > TR_CERT_MAX_COLS) return TR_MATH_INVALID;
        if (c->nterms > TR_CERT_MAX_TERMS) return TR_MATH_UNSUPPORTED;
        if (!c->exponents || !c->coefficients) return TR_MATH_INVALID;
        k->e = zalloc(c->nterms * (uint64_t)c->ncols);
        k->c = zalloc(c->nterms * 8);
        if (!k->e || !k->c) { copy_free(k); return TR_MATH_MEMORY; }
        uint64_t m = 0;
        int32_t rc = tr_cert_poly_canonical(c->ncols, c->nterms, c->exponents, c->coefficients,
                                            c->carrier == TR_CARRIER_GFP ? c->field : 0, 0, k->e, k->c, &m);
        if (rc) { copy_free(k); return rc; }
        c->exponents = k->e; c->coefficients = k->c; c->nterms = m;
    }
    int32_t rc = tr_cert_claim_check(c, ncols, need);
    if (rc) copy_free(k);
    return rc;
}
int32_t tr_cert_verify(const tr_scope *scope, const tr_cert_claim *claims, int32_t n, const tr_math_budget *budget,
                       tr_cert_evidence *out, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    tr_scope_view v;
    if (!scope || tr_scope_get_view(scope, &v) || n < 0 || (n && (!claims || !out))) return TR_MATH_INVALID;
    if (n > TR_SCOPE_MAX_COLUMNS * 64) return TR_MATH_UNSUPPORTED;
    claim_copy *k = zalloc((uint64_t)(n ? n : 1) * sizeof *k);
    tr_cert_claim *cs = zalloc((uint64_t)(n ? n : 1) * sizeof *cs);
    if (!k || !cs) { free(k); free(cs); return TR_MATH_MEMORY; }
    int32_t rc = TR_MATH_OK, made = 0;
    for (; made < n && !rc; made++) {
        rc = copy_checked(&claims[made], v.ncols, &k[made], NULL);
        if (!rc) cs[made] = k[made].claim;
    }
    if (rc) made--;
    if (!rc) rc = tr_cert_verify_view(&v, cs, n, budget, out, refusal);
    for (int32_t j = 0; j < made; j++) copy_free(&k[j]);
    free(k); free(cs);
    return rc;
}

/* ------------------------------------------------------------------ certificates */
struct tr_cert_s {
    uint32_t magic;
    int32_t kind, law, enforcement, backend, depth, npremises;
    tr_cert_claim claim;                 /* pointers into the arrays below */
    tr_cert_bit *bits;
    uint8_t *exponents;
    int64_t *coefficients, *keys, *targets;
    uint64_t *used;
    tr_cert_record *record;              /* NULL for a law or a derived certificate */
    tr_cert_evidence evidence;
    tr_cert_sample sample;
    tr_cert **premises;
    uint64_t bytes, digest[2];
};
static int cert_valid(const tr_cert *c) { return c && c->magic == CERT_MAGIC; }
void tr_cert_free(tr_cert *c) {
    if (!cert_valid(c)) return;
    free(c->bits); free(c->exponents); free(c->coefficients); free(c->keys); free(c->targets); free(c->used);
    tr_cert_record_free(c->record);
    for (int32_t i = 0; i < c->npremises; i++) tr_cert_free(c->premises[i]);
    free(c->premises);
    c->magic = 0;
    free(c);
}
static uint64_t cert_alloc_bytes(const tr_cert_claim *c) {
    uint64_t b = sizeof(tr_cert);
    b = tr_budget_add(b, claim_payload_bytes(c));
    return b;
}
/* a certificate owning a copy of the claim's arrays (the claim is checked) */
static tr_cert *cert_alloc(const tr_cert_claim *in, int32_t kind) {
    tr_cert *c = zalloc(sizeof *c);
    if (!c) return NULL;
    c->magic = CERT_MAGIC;
    c->kind = kind;
    c->claim = *in;
    tr_cert_claim *k = &c->claim;
    int fail = 0;
    if (k->bits) {
        c->bits = zalloc((uint64_t)k->nbits * sizeof *c->bits);
        if (c->bits) memcpy(c->bits, in->bits, (size_t)k->nbits * sizeof *c->bits); else fail = 1;
        k->bits = c->bits;
    }
    if (k->nterms) {
        c->exponents = zalloc(k->nterms * (uint64_t)k->ncols);
        c->coefficients = zalloc(k->nterms * 8);
        if (c->exponents && c->coefficients) {
            memcpy(c->exponents, in->exponents, (size_t)(k->nterms * (uint64_t)k->ncols));
            memcpy(c->coefficients, in->coefficients, (size_t)(k->nterms * 8));
        } else fail = 1;
    }
    k->exponents = c->exponents; k->coefficients = c->coefficients;
    if (k->keys) {
        c->keys = zalloc(k->nvalues * 8);
        if (c->keys) { if (k->nvalues) memcpy(c->keys, in->keys, (size_t)(k->nvalues * 8)); } else fail = 1;
        k->keys = c->keys;
    }
    if (k->targets) {
        c->targets = zalloc(k->nvalues * 8);
        if (c->targets) { if (k->nvalues) memcpy(c->targets, in->targets, (size_t)(k->nvalues * 8)); } else fail = 1;
        k->targets = c->targets;
    }
    if (k->used) {
        uint64_t nw = (k->nvalues + 63) / 64;
        c->used = zalloc(nw * 8);
        if (c->used) { if (nw) memcpy(c->used, in->used, (size_t)(nw * 8)); } else fail = 1;
        k->used = c->used;
    }
    c->evidence.witness = TR_CERT_NONE;
    c->bytes = cert_alloc_bytes(k);
    if (fail) { tr_cert_free(c); return NULL; }
    return c;
}

static void cert_write(wbuf *w, const tr_cert *c) {
    const tr_cert_claim *k = &c->claim;
    w_u64(w, CERT_SERIAL_TAG); w_u64(w, SERIAL_VERSION);
    w_i64(w, c->kind); w_i64(w, c->law); w_i64(w, c->enforcement); w_i64(w, c->backend); w_i64(w, c->depth);
    w_i64(w, k->kind); w_i64(w, k->policy); w_i64(w, k->ncols); w_i64(w, k->nbits);
    for (int i = 0; i < TR_CERT_MAX_COLS; i++) w_i64(w, k->col[i]);
    w_i64(w, k->carrier); w_i64(w, k->constant); w_u64(w, k->field); w_u64(w, k->tolerance);
    w_i64(w, k->scale); w_i64(w, k->bias); w_u64(w, k->mask); w_u64(w, k->nterms); w_u64(w, k->nvalues);
    w_u64(w, k->gf.p);
    for (int32_t j = 0; j < k->nbits && k->bits; j++) {
        w_i64(w, k->bits[j].col); w_i64(w, k->bits[j].kind); w_i64(w, k->bits[j].bit); w_i64(w, k->bits[j].base);
    }
    if (k->nterms) {
        w_raw(w, k->exponents, k->nterms * (uint64_t)k->ncols);
        for (uint64_t t = 0; t < k->nterms; t++) w_i64(w, k->coefficients[t]);
    }
    w_u64(w, k->keys != NULL); w_u64(w, k->targets != NULL); w_u64(w, k->used != NULL);
    if (k->keys) for (uint64_t i = 0; i < k->nvalues; i++) w_i64(w, k->keys[i]);
    if (k->targets) for (uint64_t i = 0; i < k->nvalues; i++) w_i64(w, k->targets[i]);
    if (k->used) w_words(w, k->used, (k->nvalues + 63) / 64);
    const tr_cert_evidence *e = &c->evidence;
    w_u64(w, e->rows); w_u64(w, e->selected); w_u64(w, e->checked); w_u64(w, e->excluded); w_u64(w, e->violations);
    w_u64(w, e->witness); w_u64(w, e->overflow); w_i64(w, e->unit); w_i64(w, e->holds);
    const tr_cert_sample *s = &c->sample;
    w_i64(w, s->method); w_u64(w, s->size); w_u64(w, s->seed); w_u64(w, s->population); w_u64(w, s->distinct);
    w_u64(w, c->record != NULL);
    if (c->record) record_write(w, c->record);
    w_i64(w, c->npremises);
    for (int32_t i = 0; i < c->npremises; i++) cert_write(w, c->premises[i]);
}
static int32_t cert_seal(tr_cert *c) {   /* the digest of the canonical bytes, and the bytes owned */
    wbuf w = {NULL, 0, 0, 0};
    cert_write(&w, c);
    uint8_t *buf = zalloc(w.len);
    if (!buf) return TR_MATH_MEMORY;
    wbuf x = {buf, w.len, 0, 0};
    cert_write(&x, c);
    hasher h;
    h_init(&h, 0x43455254);
    h_bytes(&h, buf, x.len);
    h_final(&h, c->digest);
    free(buf);
    uint64_t bytes = cert_alloc_bytes(&c->claim);
    if (c->record) bytes = tr_budget_add(bytes, c->record->bytes);
    for (int32_t i = 0; i < c->npremises; i++) bytes = tr_budget_add(bytes, c->premises[i]->bytes);
    c->bytes = tr_budget_add(bytes, tr_budget_mul((uint64_t)c->npremises, sizeof(tr_cert *)));
    return TR_MATH_OK;
}
uint64_t tr_cert_serial_size(const tr_cert *c) {
    if (!cert_valid(c)) return 0;
    wbuf w = {NULL, 0, 0, 0};
    cert_write(&w, c);
    return w.len;
}
int32_t tr_cert_serialize(const tr_cert *c, uint8_t *out, uint64_t capacity, uint64_t *written) {
    if (!cert_valid(c) || !out) return TR_MATH_INVALID;
    wbuf w = {out, capacity, 0, 0};
    cert_write(&w, c);
    if (w.fail) return TR_MATH_INVALID;
    if (written) *written = w.len;
    return TR_MATH_OK;
}
static int law_holds(int32_t law, const tr_cert_claim *c, int32_t *error);
static tr_cert *cert_read(rbuf *rd, int depth, const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error) {
    if (depth > TR_CERT_MAX_DEPTH || r_u64(rd) != CERT_SERIAL_TAG || r_u64(rd) != SERIAL_VERSION) {
        set_error(error, TR_MATH_INVALID); return NULL;
    }
    int32_t kind = r_i32(rd), law = r_i32(rd), enforcement = r_i32(rd), backend = r_i32(rd), cdepth = r_i32(rd);
    tr_cert_claim k;
    memset(&k, 0, sizeof k);
    k.kind = r_i32(rd); k.policy = r_i32(rd); k.ncols = r_i32(rd); k.nbits = r_i32(rd);
    for (int i = 0; i < TR_CERT_MAX_COLS; i++) k.col[i] = r_i32(rd);
    k.carrier = r_i32(rd); k.constant = r_i32(rd); k.field = r_u64(rd); k.tolerance = r_u64(rd);
    k.scale = r_i64(rd); k.bias = r_i64(rd); k.mask = r_u64(rd); k.nterms = r_u64(rd); k.nvalues = r_u64(rd);
    uint64_t modulus = r_u64(rd);
    if (rd->fail || !kind_ok(kind) || k.kind < 1 || k.kind >= TR_CLAIM_KINDS || k.nbits < 0 || k.nbits > TR_CERT_MAX_BITS ||
        k.ncols < 0 || k.ncols > TR_CERT_MAX_COLS || k.nterms > TR_CERT_MAX_TERMS || k.nvalues > (UINT64_C(1) << 32) ||
        backend < 0 || backend > 1 || cdepth < 0 || cdepth > TR_CERT_MAX_DEPTH) {
        set_error(error, rd->fail ? TR_MATH_INVALID : TR_MATH_INVALID); return NULL;
    }
    tr_cert_bit bits[TR_CERT_MAX_BITS];
    for (int32_t j = 0; j < k.nbits; j++) {
        bits[j].col = r_i32(rd); bits[j].kind = r_i32(rd); bits[j].bit = r_i32(rd); bits[j].reserved = 0;
        bits[j].base = r_i64(rd);
    }
    uint8_t *exps = NULL;
    int64_t *coefs = NULL, *keys = NULL, *targets = NULL;
    uint64_t *used = NULL;
    tr_cert *c = NULL;
    tr_cert_record *record = NULL;
    if (k.nterms) {
        const uint8_t *p = r_raw(rd, k.nterms * (uint64_t)k.ncols);
        exps = zalloc(k.nterms * (uint64_t)k.ncols);
        coefs = zalloc(k.nterms * 8);
        if (!exps || !coefs) { set_error(error, TR_MATH_MEMORY); goto fail; }
        if (p) memcpy(exps, p, (size_t)(k.nterms * (uint64_t)k.ncols));
        for (uint64_t t = 0; t < k.nterms; t++) coefs[t] = r_i64(rd);
    }
    uint64_t has_keys = r_u64(rd), has_targets = r_u64(rd), has_used = r_u64(rd);
    if (has_keys > 1 || has_targets > 1 || has_used > 1 || rd->fail) { set_error(error, TR_MATH_INVALID); goto fail; }
    if (has_keys) {
        if (rd->len - rd->at < k.nvalues * 8) { set_error(error, TR_MATH_INVALID); goto fail; }
        keys = zalloc(k.nvalues * 8);
        if (!keys) { set_error(error, TR_MATH_MEMORY); goto fail; }
        for (uint64_t i = 0; i < k.nvalues; i++) keys[i] = r_i64(rd);
    }
    if (has_targets) {
        if (rd->len - rd->at < k.nvalues * 8) { set_error(error, TR_MATH_INVALID); goto fail; }
        targets = zalloc(k.nvalues * 8);
        if (!targets) { set_error(error, TR_MATH_MEMORY); goto fail; }
        for (uint64_t i = 0; i < k.nvalues; i++) targets[i] = r_i64(rd);
    }
    if (has_used) {
        uint64_t nw = (k.nvalues + 63) / 64;
        if (rd->len - rd->at < nw * 8) { set_error(error, TR_MATH_INVALID); goto fail; }
        used = zalloc(nw * 8);
        if (!used) { set_error(error, TR_MATH_MEMORY); goto fail; }
        for (uint64_t i = 0; i < nw; i++) used[i] = r_u64(rd);
    }
    k.bits = k.nbits ? bits : NULL; k.exponents = exps; k.coefficients = coefs; k.keys = keys; k.targets = targets;
    k.used = used;
    tr_cert_evidence e;
    e.rows = r_u64(rd); e.selected = r_u64(rd); e.checked = r_u64(rd); e.excluded = r_u64(rd); e.violations = r_u64(rd);
    e.witness = r_u64(rd); e.overflow = r_u64(rd); e.unit = r_i32(rd); e.holds = r_i32(rd);
    tr_cert_sample s;
    memset(&s, 0, sizeof s);
    s.method = r_i32(rd); s.size = r_u64(rd); s.seed = r_u64(rd); s.population = r_u64(rd); s.distinct = r_u64(rd);
    uint64_t has_record = r_u64(rd);
    if (rd->fail || has_record > 1) { set_error(error, TR_MATH_INVALID); goto fail; }
    if (kind == TR_CERT_LAW) {
        if (claim_check(&k, -1, 0, NULL) || claim_field(&k, modulus) || has_record || !law_holds(law, &k, error)) {
            if (error && *error == TR_MATH_OK) *error = TR_MATH_INVALID;
            if (error && *error == TR_MATH_DOMAIN_ERROR) *error = TR_MATH_INVALID;
            goto fail;
        }
    } else {
        int32_t rc = claim_check(&k, TR_SCOPE_MAX_COLUMNS, 1, NULL);
        if (!rc) rc = claim_field(&k, modulus);
        if (rc || (kind == TR_CERT_SCHEMA && enforcement != TR_CERT_ENFORCE_LOAD && enforcement != TR_CERT_ENFORCE_WRITE) ||
            (kind != TR_CERT_SCHEMA && enforcement) || law ||
            (e.unit != CLAIMS[k.kind - 1].unit) || (kind != TR_CERT_HYPOTHESIS && (!e.holds || e.violations || e.overflow)) ||
            e.holds != (!e.overflow && !e.violations) || (e.violations == 0) != (e.witness == TR_CERT_NONE)) {
            set_error(error, TR_MATH_INVALID); goto fail;
        }
        if (has_record) {
            record = record_read(rd, budget, refusal, error);
            if (!record) goto fail;
        }
    }
    int32_t npremises = r_i32(rd);
    if (rd->fail || npremises < 0 || npremises > TR_CERT_MAX_PREMISES || (npremises && (k.kind != TR_CLAIM_POLYNOMIAL ||
        has_record || kind == TR_CERT_LAW)) || (!npremises && !has_record && kind != TR_CERT_LAW) ||
        (npremises && cdepth < 1) || (!npremises && cdepth)) {
        set_error(error, TR_MATH_INVALID); goto fail;
    }
    c = cert_alloc(&k, kind);
    if (!c) { set_error(error, TR_MATH_MEMORY); goto fail; }
    c->law = law; c->enforcement = enforcement; c->backend = backend; c->depth = cdepth;
    c->evidence = e; c->sample = s; c->record = record; record = NULL;
    if (npremises) {
        c->premises = zalloc((uint64_t)npremises * sizeof *c->premises);
        if (!c->premises) { set_error(error, TR_MATH_MEMORY); goto fail; }
        int32_t top = 0;
        for (int32_t i = 0; i < npremises; i++) {
            c->premises[i] = cert_read(rd, depth + 1, budget, refusal, error);
            if (!c->premises[i]) goto fail;
            c->npremises = i + 1;
            if (c->premises[i]->kind > top) top = c->premises[i]->kind;
            if (c->premises[i]->depth + 1 > cdepth) { set_error(error, TR_MATH_INVALID); goto fail; }
        }
        if (top != kind) { set_error(error, TR_MATH_INVALID); goto fail; }
    }
    free(exps); free(coefs); free(keys); free(targets); free(used);
    if (cert_seal(c)) { tr_cert_free(c); set_error(error, TR_MATH_MEMORY); return NULL; }
    set_error(error, TR_MATH_OK);
    return c;
fail:
    free(exps); free(coefs); free(keys); free(targets); free(used);
    tr_cert_record_free(record);
    tr_cert_free(c);
    return NULL;
}
tr_cert *tr_cert_deserialize(const uint8_t *in, uint64_t bytes, uint64_t *consumed, const tr_math_budget *budget,
                             tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    if (!in) { set_error(error, TR_MATH_INVALID); return NULL; }
    tr_math_need need;
    need_zero(&need);
    need.need[TR_BUDGET_BYTES] = bytes;          /* everything a certificate owns comes from these bytes (bounded) */
    tr_math_budget dflt;
    int32_t rc = tr_budget_check(budget_or(budget, &dflt), &need, refusal);
    if (rc) { set_error(error, rc); return NULL; }
    rbuf rd = {in, bytes, 0, 0};
    tr_cert *c = cert_read(&rd, 0, budget, refusal, error);
    if (!c) return NULL;
    /* only canonical bytes are accepted: writing the certificate again gives exactly the bytes read */
    uint64_t size = tr_cert_serial_size(c);
    uint8_t *again = size == rd.at ? zalloc(size) : NULL;
    int same = again && !tr_cert_serialize(c, again, size, NULL) && memcmp(again, in, (size_t)size) == 0;
    free(again);
    if (!same) { tr_cert_free(c); set_error(error, TR_MATH_INVALID); return NULL; }
    if (consumed) *consumed = rd.at;
    return c;
}
tr_cert *tr_cert_clone(const tr_cert *c, int32_t *error) {
    if (!cert_valid(c)) { set_error(error, TR_MATH_INVALID); return NULL; }
    uint64_t size = tr_cert_serial_size(c);
    uint8_t *buf = zalloc(size);
    if (!buf) { set_error(error, TR_MATH_MEMORY); return NULL; }
    tr_cert *out = NULL;
    if (!tr_cert_serialize(c, buf, size, NULL)) {
        tr_math_budget unlimited;
        for (int i = 0; i < TR_BUDGET_RESOURCES; i++) unlimited.limit[i] = UINT64_MAX;
        out = tr_cert_deserialize(buf, size, NULL, &unlimited, NULL, error);
    } else set_error(error, TR_MATH_INVALID);
    free(buf);
    return out;
}
int32_t tr_cert_equal(const tr_cert *a, const tr_cert *b) {
    if (!cert_valid(a) || !cert_valid(b)) return TR_MATH_INVALID;
    if (a == b) return 1;
    if (a->digest[0] != b->digest[0] || a->digest[1] != b->digest[1]) return 0;
    uint64_t na = tr_cert_serial_size(a), nb = tr_cert_serial_size(b);
    if (na != nb) return 0;
    uint8_t *x = zalloc(na), *y = zalloc(nb);
    int same = 0;
    if (x && y && !tr_cert_serialize(a, x, na, NULL) && !tr_cert_serialize(b, y, nb, NULL)) same = memcmp(x, y, (size_t)na) == 0;
    int32_t rc = (x && y) ? same : TR_MATH_MEMORY;
    free(x); free(y);
    return rc;
}
int32_t tr_cert_get_info(const tr_cert *c, tr_cert_info *out) {
    if (!cert_valid(c) || !out) return TR_MATH_INVALID;
    memset(out, 0, sizeof *out);
    out->kind = c->kind; out->claim = c->claim.kind; out->law = c->law; out->enforcement = c->enforcement;
    out->policy = c->claim.policy; out->carrier = c->claim.carrier; out->exact = claim_exact(&c->claim);
    out->backend = c->backend; out->field = c->claim.field; out->modulus = c->claim.gf.p;
    out->tolerance = c->claim.tolerance; out->scoped = c->record != NULL || c->npremises > 0;
    out->npremises = c->npremises; out->depth = c->depth; out->bytes = c->bytes;
    out->digest[0] = c->digest[0]; out->digest[1] = c->digest[1];
    out->evidence = c->evidence; out->sample = c->sample;
    return TR_MATH_OK;
}
int32_t tr_cert_get_claim(const tr_cert *c, tr_cert_claim *out) {
    if (!cert_valid(c) || !out) return TR_MATH_INVALID;
    *out = c->claim;
    return TR_MATH_OK;
}
const tr_cert_record *tr_cert_get_record(const tr_cert *c) { return cert_valid(c) ? c->record : NULL; }
const tr_cert *tr_cert_get_premise(const tr_cert *c, int32_t i) {
    return cert_valid(c) && i >= 0 && i < c->npremises ? c->premises[i] : NULL;
}

/* ------------------------------------------------------------------ laws */
static int law_holds(int32_t law, const tr_cert_claim *c, int32_t *error) {
    if (law < 1 || law >= TR_CERT_LAWS || c->kind != TR_CLAIM_POLYNOMIAL) { set_error(error, TR_MATH_INVALID); return 0; }
    if (law == TR_CERT_LAW_RING_IDENTITY && (c->field || c->carrier != TR_CARRIER_I128 || c->tolerance)) {
        set_error(error, TR_MATH_INVALID); return 0;
    }
    if (law == TR_CERT_LAW_GFP_FUNCTION && (!c->field || c->carrier != TR_CARRIER_GFP)) { set_error(error, TR_MATH_INVALID); return 0; }
    uint8_t *e = zalloc(c->nterms * (uint64_t)c->ncols);
    int64_t *k = zalloc(c->nterms * 8);
    if (!e || !k) { free(e); free(k); set_error(error, TR_MATH_MEMORY); return 0; }
    uint64_t m = 0;
    int32_t rc = c->nterms ? tr_cert_poly_canonical(c->ncols, c->nterms, c->exponents, c->coefficients,
                                                    law == TR_CERT_LAW_GFP_FUNCTION ? c->field : 0,
                                                    law == TR_CERT_LAW_GFP_FUNCTION, e, k, &m) : TR_MATH_OK;
    free(e); free(k);
    if (rc) { set_error(error, rc); return 0; }
    if (m) { set_error(error, TR_MATH_DOMAIN_ERROR); return 0; }
    set_error(error, TR_MATH_OK);
    return 1;
}
tr_cert *tr_cert_law(int32_t law, const tr_cert_claim *claim, const tr_math_budget *budget, tr_math_refusal *refusal,
                     int32_t *error) {
    tr_budget_none(refusal);
    if (!claim || law < 1 || law >= TR_CERT_LAWS) { set_error(error, TR_MATH_INVALID); return NULL; }
    tr_cert_claim k = *claim;
    uint64_t modulus = k.gf.p ? k.gf.p : k.field;
    tr_math_need need;
    int32_t rc = claim_check(&k, -1, 0, &need);
    if (!rc) rc = claim_field(&k, modulus);
    if (rc) { set_error(error, rc); return NULL; }
    need.need[TR_BUDGET_BYTES] = tr_budget_add(cert_alloc_bytes(&k), tr_budget_mul(k.nterms, tr_budget_add((uint64_t)k.ncols, 8)));
    tr_math_budget dflt;
    rc = tr_budget_check(budget_or(budget, &dflt), &need, refusal);
    if (rc) { set_error(error, rc); return NULL; }
    if (!law_holds(law, &k, error)) return NULL;
    tr_cert *c = cert_alloc(&k, TR_CERT_LAW);
    if (!c) { set_error(error, TR_MATH_MEMORY); return NULL; }
    c->law = law;
    c->evidence.unit = TR_CERT_UNIT_ROWS; c->evidence.holds = 1;
    if (cert_seal(c)) { tr_cert_free(c); set_error(error, TR_MATH_MEMORY); return NULL; }
    set_error(error, TR_MATH_OK);
    return c;
}

/* ------------------------------------------------------------------ certificates over scopes */
/* the needs of certifying a claim on a scope: the claim, the verification scratch, the record, the certificate */
static int32_t certify_need(const tr_scope *scope, const tr_scope_view *v, const tr_cert_claim *k, int32_t flags,
                            tr_math_need *need) {
    tr_math_need vn;
    int32_t rc = tr_cert_verify_need(v, k, 1, TR_CERT_NEED_CPU, &vn);
    if (rc) return rc;
    *need = vn;
    uint64_t bytes = tr_budget_add(vn.need[TR_BUDGET_BYTES], cert_alloc_bytes(k));
    bytes = tr_budget_add(bytes, sizeof(tr_cert_record));
    if (flags & TR_CERT_RECORD_SELECTION) bytes = tr_budget_add(bytes, tr_budget_bitmap_bytes(v->rows));
    if (flags & TR_CERT_RECORD_GROUPS) bytes = tr_budget_add(bytes, tr_scope_key_size(scope, TR_SCOPE_KEY_GROUPS));
    need->need[TR_BUDGET_BYTES] = bytes;
    return TR_MATH_OK;
}
static int dims_ok(const tr_cert_dim *dims, int32_t ndims) {
    if (ndims < 0 || ndims > TR_CERT_MAX_DIMS || (ndims && !dims)) return 0;
    for (int32_t i = 0; i < ndims; i++) if (dims[i].unique != (dims[i].duplicates == 0 && dims[i].nulls == 0)) return 0;
    return 1;
}
static tr_cert *certify(const tr_scope *scope, const tr_cert_claim *claim, int32_t kind, int32_t enforcement,
                        const tr_cert_sample *sample, const tr_cert_evidence *adopted, int32_t backend,
                        const tr_cert_dim *dims, int32_t ndims, tr_cert_evidence *evidence,
                        const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    tr_scope_view v;
    if (!scope || !claim || tr_scope_get_view(scope, &v) || !dims_ok(dims, ndims)) { set_error(error, TR_MATH_INVALID); return NULL; }
    claim_copy kc;
    int32_t rc = copy_checked(claim, v.ncols, &kc, NULL);
    if (rc) { set_error(error, rc); return NULL; }
    tr_cert_claim *k = &kc.claim;
    int32_t flags = TR_CERT_RECORD_SELECTION | (k->kind == TR_CLAIM_COVERAGE ? TR_CERT_RECORD_GROUPS : 0);
    tr_math_need need;
    rc = certify_need(scope, &v, k, flags, &need);
    uint64_t *sample_bits = NULL, *cum = NULL;
    tr_cert_sample drawn;
    memset(&drawn, 0, sizeof drawn);
    uint64_t nw = (v.rows + 63) / 64;
    if (!rc && sample) {
        drawn = *sample;
        if (drawn.method < TR_CERT_SAMPLE_UNIFORM || drawn.method > TR_CERT_SAMPLE_PREFIX || drawn.reserved ||
            (drawn.method != TR_CERT_SAMPLE_UNIFORM && drawn.size > v.selected) || !drawn.size || !v.selected)
            rc = TR_MATH_INVALID;
        if (!rc && drawn.size > (UINT64_C(1) << 40)) rc = TR_MATH_UNSUPPORTED;
        if (!rc) {
            if (drawn.size > need.need[TR_BUDGET_SUPPORT]) need.need[TR_BUDGET_SUPPORT] = drawn.size;
            need.need[TR_BUDGET_BYTES] = tr_budget_add(need.need[TR_BUDGET_BYTES], tr_budget_add(nw * 16, 8));
        }
    }
    tr_math_budget dflt;
    if (!rc) rc = tr_budget_check(budget_or(budget, &dflt), &need, refusal);
    if (rc) { copy_free(&kc); set_error(error, rc); return NULL; }
    tr_cert_evidence e;
    if (adopted) {
        e = *adopted;
        int bad = e.rows != v.rows || e.selected != v.selected || e.unit != CLAIMS[k->kind - 1].unit || !e.holds ||
                  e.violations || e.overflow || e.witness != TR_CERT_NONE ||
                  (e.unit == TR_CERT_UNIT_ROWS ? tr_budget_add(e.checked, e.excluded) != v.selected
                                               : (e.checked > v.ngroups || e.excluded > v.selected));
        if (bad) { copy_free(&kc); set_error(error, TR_MATH_INVALID); return NULL; }
    } else {
        tr_scope_view w = v;
        if (sample) {
            sample_bits = zalloc(nw * 8);
            cum = zalloc((nw + 1) * 8);
            if (!sample_bits || !cum) { free(sample_bits); free(cum); copy_free(&kc); set_error(error, TR_MATH_MEMORY); return NULL; }
            for (uint64_t i = 0; i < nw; i++) cum[i + 1] = cum[i] + (uint64_t)__builtin_popcountll(v.selection[i]);
            uint64_t N = v.selected;
            for (uint64_t i = 0; i < drawn.size && N; i++) {
                uint64_t rank;
                if (drawn.method == TR_CERT_SAMPLE_PREFIX) rank = i;
                else if (drawn.method == TR_CERT_SAMPLE_STRIDE) rank = (uint64_t)(((unsigned __int128)i * N) / drawn.size);
                else {
                    uint64_t z = drawn.seed + (i + 1) * UINT64_C(0x9E3779B97F4A7C15);
                    z = (z ^ (z >> 30)) * UINT64_C(0xBF58476D1CE4E5B9);
                    z = (z ^ (z >> 27)) * UINT64_C(0x94D049BB133111EB);
                    z ^= z >> 31;
                    rank = (uint64_t)(((unsigned __int128)z * N) >> 64);
                }
                uint64_t lo = 0, hi = nw;                           /* the word holding rank: cum[lo] <= rank < cum[lo + 1] */
                while (hi - lo > 1) { uint64_t mid = (lo + hi) / 2; if (cum[mid] <= rank) lo = mid; else hi = mid; }
                uint64_t bits = v.selection[lo], skip = rank - cum[lo];
                for (uint64_t s = 0; s < skip; s++) bits &= bits - 1;
                uint64_t row = lo * 64 + (uint64_t)__builtin_ctzll(bits);
                sample_bits[row >> 6] |= UINT64_C(1) << (row & 63);
            }
            uint64_t distinct = 0;
            for (uint64_t i = 0; i < nw; i++) distinct += (uint64_t)__builtin_popcountll(sample_bits[i]);
            drawn.population = N; drawn.distinct = distinct;
            w.selection = sample_bits; w.selected = distinct;
            free(cum); cum = NULL;
        }
        rc = tr_cert_verify_view(&w, k, 1, budget, &e, refusal);
        free(sample_bits);
        if (rc) { copy_free(&kc); set_error(error, rc); return NULL; }
    }
    if (evidence) *evidence = e;
    if (e.overflow) { copy_free(&kc); set_error(error, TR_MATH_OVERFLOW); return NULL; }
    if (e.violations) { copy_free(&kc); set_error(error, TR_MATH_DOMAIN_ERROR); return NULL; }
    int32_t cols[TR_CERT_MAX_COLS];
    int32_t ncols = claim_columns(k, cols);
    tr_cert_record *record = record_build(scope, cols, ncols, flags, dims, ndims, NULL, NULL, error);
    if (!record) { copy_free(&kc); return NULL; }
    tr_cert *c = cert_alloc(k, kind);
    copy_free(&kc);
    if (!c) { tr_cert_record_free(record); set_error(error, TR_MATH_MEMORY); return NULL; }
    c->enforcement = enforcement; c->backend = backend; c->record = record; c->evidence = e;
    if (sample) c->sample = drawn;
    if (cert_seal(c)) { tr_cert_free(c); set_error(error, TR_MATH_MEMORY); return NULL; }
    set_error(error, TR_MATH_OK);
    return c;
}
tr_cert *tr_cert_schema(const tr_scope *scope, const tr_cert_claim *claim, int32_t enforcement, const tr_cert_dim *dims,
                        int32_t ndims, tr_cert_evidence *evidence, const tr_math_budget *budget, tr_math_refusal *refusal,
                        int32_t *error) {
    if (!claim || (claim->kind != TR_CLAIM_UNIQUE && claim->kind != TR_CLAIM_NOT_NULL && claim->kind != TR_CLAIM_REFERENCE) ||
        (enforcement != TR_CERT_ENFORCE_LOAD && enforcement != TR_CERT_ENFORCE_WRITE)) {
        tr_budget_none(refusal); set_error(error, TR_MATH_INVALID); return NULL;
    }
    return certify(scope, claim, TR_CERT_SCHEMA, enforcement, NULL, NULL, TR_CERT_BY_CPU, dims, ndims, evidence,
                   budget, refusal, error);
}
tr_cert *tr_cert_snapshot(const tr_scope *scope, const tr_cert_claim *claim, const tr_cert_dim *dims, int32_t ndims,
                          tr_cert_evidence *evidence, const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error) {
    return certify(scope, claim, TR_CERT_SNAPSHOT, 0, NULL, NULL, TR_CERT_BY_CPU, dims, ndims, evidence, budget,
                   refusal, error);
}
tr_cert *tr_cert_hypothesis(const tr_scope *scope, const tr_cert_claim *claim, const tr_cert_sample *sample,
                            const tr_cert_dim *dims, int32_t ndims, tr_cert_evidence *evidence,
                            const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error) {
    if (!sample) { tr_budget_none(refusal); set_error(error, TR_MATH_INVALID); return NULL; }
    return certify(scope, claim, TR_CERT_HYPOTHESIS, 0, sample, NULL, TR_CERT_BY_CPU, dims, ndims, evidence, budget,
                   refusal, error);
}
tr_cert *tr_cert_adopt(const tr_scope *scope, const tr_cert_claim *claim, const tr_cert_evidence *evidence,
                       int32_t backend, const tr_cert_dim *dims, int32_t ndims, const tr_math_budget *budget,
                       tr_math_refusal *refusal, int32_t *error) {
    if (!evidence || (backend != TR_CERT_BY_CPU && backend != TR_CERT_BY_DEVICE)) {
        tr_budget_none(refusal); set_error(error, TR_MATH_INVALID); return NULL;
    }
    return certify(scope, claim, TR_CERT_SNAPSHOT, 0, NULL, evidence, backend, dims, ndims, NULL, budget, refusal, error);
}
tr_cert *tr_cert_reverify(const tr_cert *old, const tr_scope *current, const tr_cert_dim *dims, int32_t ndims,
                          tr_cert_evidence *evidence, const tr_math_budget *budget, tr_math_refusal *refusal,
                          int32_t *error) {
    if (!cert_valid(old) || old->kind == TR_CERT_LAW || old->npremises) {
        tr_budget_none(refusal); set_error(error, TR_MATH_INVALID); return NULL;
    }
    int32_t kind = old->kind == TR_CERT_SCHEMA ? TR_CERT_SCHEMA : TR_CERT_SNAPSHOT;
    return certify(current, &old->claim, kind, old->enforcement, NULL, NULL, TR_CERT_BY_CPU, dims, ndims, evidence,
                   budget, refusal, error);
}

/* ------------------------------------------------------------------ derivation */
typedef struct { uint8_t e[TR_CERT_MAX_COLS]; __int128 c; } dterm;
/* the premise's claim as a polynomial over the variable set vars[nv] (scope columns): terms appended to out */
static int32_t as_poly(const tr_cert_claim *k, const int32_t *vars, int32_t nv, dterm *out, uint64_t *n, uint64_t cap) {
    int32_t at[TR_CERT_MAX_COLS];
    for (int32_t i = 0; i < k->ncols; i++) {
        at[i] = -1;
        for (int32_t j = 0; j < nv; j++) if (vars[j] == k->col[i]) at[i] = j;
        if (at[i] < 0) return TR_MATH_INVALID;
    }
#define PUSH(coef) do { if (*n >= cap) return TR_MATH_UNSUPPORTED; memset(&out[*n], 0, sizeof out[*n]); out[*n].c = (coef); } while (0)
    switch (k->kind) {
    case TR_CLAIM_AFFINE:            /* out - scale x - bias */
        PUSH(1); out[*n].e[at[0]] = 1; (*n)++;
        if (k->scale) { PUSH(-(__int128)k->scale); out[*n].e[at[1]] = 1; (*n)++; }
        if (k->bias) { PUSH(-(__int128)k->bias); (*n)++; }
        return TR_MATH_OK;
    case TR_CLAIM_SUM:
        PUSH(1); out[*n].e[at[0]] = 1; (*n)++;
        PUSH(-1); out[*n].e[at[1]] += 1; (*n)++;
        PUSH(-1); out[*n].e[at[2]] += 1; (*n)++;
        return TR_MATH_OK;
    case TR_CLAIM_PRODUCT:
        PUSH(1); out[*n].e[at[0]] = 1; (*n)++;
        PUSH(-1); out[*n].e[at[1]] += 1; out[*n].e[at[2]] += 1; (*n)++;
        return TR_MATH_OK;
    case TR_CLAIM_POLYNOMIAL:
        for (uint64_t t = 0; t < k->nterms; t++) {
            PUSH(k->coefficients[t]);
            for (int32_t i = 0; i < k->ncols; i++) out[*n].e[at[i]] = k->exponents[t * (uint64_t)k->ncols + (uint64_t)i];
            (*n)++;
        }
        return TR_MATH_OK;
    default:
        return TR_MATH_INVALID;
    }
#undef PUSH
}
static int dterm_cmp(const void *x, const void *y) {
    const dterm *a = (const dterm *)x, *b = (const dterm *)y;
    return memcmp(a->e, b->e, TR_CERT_MAX_COLS);
}
tr_cert *tr_cert_derive(const tr_cert_claim *claim, const tr_cert *const *premises, const tr_cert_claim *cofactors,
                        int32_t n, const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    if (!claim || claim->kind != TR_CLAIM_POLYNOMIAL || n < 1 || n > TR_CERT_MAX_PREMISES || !premises || !cofactors) {
        set_error(error, TR_MATH_INVALID); return NULL;
    }
    if (claim->ncols < 1 || claim->ncols > TR_CERT_MAX_COLS) { set_error(error, TR_MATH_INVALID); return NULL; }
    int32_t vars[TR_CERT_MAX_COLS], nv = claim->ncols, top = 0, depth = 0;
    uint32_t premise_cols = 0, claim_cols = 0;
    for (int32_t i = 0; i < nv; i++) {
        vars[i] = claim->col[i];
        if (vars[i] < 0 || vars[i] >= TR_SCOPE_MAX_COLUMNS) { set_error(error, TR_MATH_INVALID); return NULL; }
        claim_cols |= UINT32_C(1) << vars[i];
    }
    uint64_t cap = claim->nterms;
    for (int32_t i = 0; i < n; i++) {
        const tr_cert *p = premises[i];
        const tr_cert_claim *q = &cofactors[i];
        if (!cert_valid(p) || p->kind == TR_CERT_LAW || p->kind == TR_CERT_SCHEMA || p->claim.field || p->claim.tolerance ||
            p->claim.carrier != TR_CARRIER_I128 || p->claim.policy != claim->policy ||
            !(p->claim.kind == TR_CLAIM_AFFINE || p->claim.kind == TR_CLAIM_SUM || p->claim.kind == TR_CLAIM_PRODUCT ||
              p->claim.kind == TR_CLAIM_POLYNOMIAL) || q->kind != TR_CLAIM_POLYNOMIAL || q->ncols < 1 ||
            q->ncols > TR_CERT_MAX_COLS || (q->nterms && (!q->exponents || !q->coefficients)) || q->nterms > TR_CERT_MAX_TERMS) {
            set_error(error, TR_MATH_INVALID); return NULL;
        }
        if (p->kind > top) top = p->kind;
        if (p->depth + 1 > depth) depth = p->depth + 1;
        for (int32_t j = 0; j < p->claim.ncols; j++) premise_cols |= UINT32_C(1) << p->claim.col[j];
        for (int32_t j = 0; j < q->ncols; j++) {        /* a cofactor reads claim columns only */
            if (q->col[j] < 0 || q->col[j] >= TR_SCOPE_MAX_COLUMNS || !((claim_cols >> q->col[j]) & 1)) {
                set_error(error, TR_MATH_INVALID); return NULL;
            }
        }
        uint64_t pt = p->claim.kind == TR_CLAIM_POLYNOMIAL ? p->claim.nterms : 3;
        cap = tr_budget_add(cap, tr_budget_mul(q->nterms, pt));
    }
    /* premises read claim columns only; FORBID (no NULL) is proven only for columns some premise reads */
    if ((premise_cols & ~claim_cols) || (claim->policy == TR_CERT_NULL_FORBID && (claim_cols & ~premise_cols))) {
        set_error(error, TR_MATH_DOMAIN_ERROR); return NULL;
    }
    if (depth > TR_CERT_MAX_DEPTH) { set_error(error, TR_MATH_UNSUPPORTED); return NULL; }
    if (claim->policy != TR_CERT_NULL_FORBID && claim->policy != TR_CERT_NULL_SKIP) { set_error(error, TR_MATH_INVALID); return NULL; }
    claim_copy kc;
    int32_t rc = copy_checked(claim, TR_SCOPE_MAX_COLUMNS, &kc, NULL);
    if (rc) { set_error(error, rc); return NULL; }
    if (kc.claim.field || kc.claim.tolerance || kc.claim.carrier != TR_CARRIER_I128) { copy_free(&kc); set_error(error, TR_MATH_INVALID); return NULL; }
    tr_math_need need;
    need_zero(&need);
    need.need[TR_BUDGET_VARIABLES] = (uint64_t)nv;
    need.need[TR_BUDGET_SUPPORT] = cap;
    need.need[TR_BUDGET_BYTES] = tr_budget_add(tr_budget_mul(cap, sizeof(dterm) * 2), cert_alloc_bytes(claim));
    tr_math_budget dflt;
    rc = tr_budget_check(budget_or(budget, &dflt), &need, refusal);
    if (!rc && cap > (UINT64_C(1) << 26)) rc = TR_MATH_UNSUPPORTED;
    if (rc) { copy_free(&kc); set_error(error, rc); return NULL; }
    dterm *acc = zalloc(tr_budget_add(cap, 1) * sizeof *acc), *pg = zalloc(64 * sizeof *pg);
    uint64_t na = 0;
    if (!acc || !pg) { free(acc); free(pg); copy_free(&kc); set_error(error, TR_MATH_MEMORY); return NULL; }
    /* P */
    rc = as_poly(&kc.claim, vars, nv, acc, &na, cap + 1);
    /* - sum_i q_i G_i */
    for (int32_t i = 0; i < n && !rc; i++) {
        const tr_cert *p = premises[i];
        const tr_cert_claim *q = &cofactors[i];
        uint64_t np = 0, pcap = p->claim.kind == TR_CLAIM_POLYNOMIAL ? p->claim.nterms : 3;
        dterm *gp = pcap > 64 ? zalloc(pcap * sizeof *gp) : pg;
        if (!gp) { rc = TR_MATH_MEMORY; break; }
        rc = as_poly(&p->claim, vars, nv, gp, &np, pcap);
        int32_t at[TR_CERT_MAX_COLS];
        for (int32_t j = 0; j < q->ncols && !rc; j++) {
            at[j] = -1;
            for (int32_t t = 0; t < nv; t++) if (vars[t] == q->col[j]) at[j] = t;
        }
        for (uint64_t a = 0; a < q->nterms && !rc; a++)
            for (uint64_t b = 0; b < np && !rc; b++) {
                if (na >= cap + 1) { rc = TR_MATH_UNSUPPORTED; break; }
                dterm *d = &acc[na];
                memset(d, 0, sizeof *d);
                for (int32_t t = 0; t < nv; t++) {
                    int e = gp[b].e[t];
                    for (int32_t j = 0; j < q->ncols; j++) if (at[j] == t) e += q->exponents[a * (uint64_t)q->ncols + (uint64_t)j];
                    if (e > 255) rc = TR_MATH_UNSUPPORTED;
                    d->e[t] = (uint8_t)e;
                }
                d->c = -((__int128)q->coefficients[a] * gp[b].c);   /* |q_a g_b| <= 2^126: exact */
                na++;
            }
        if (gp != pg) free(gp);
    }
    if (!rc) {
        qsort(acc, (size_t)na, sizeof *acc, dterm_cmp);
        for (uint64_t i = 0; i < na && !rc;) {
            __int128 s = 0;
            uint64_t j = i;
            for (; j < na && !memcmp(acc[i].e, acc[j].e, TR_CERT_MAX_COLS); j++)
                if (__builtin_add_overflow(s, acc[j].c, &s)) rc = TR_MATH_OVERFLOW;
            if (!rc && s) rc = TR_MATH_DOMAIN_ERROR;           /* P - sum q_i G_i is not zero */
            i = j;
        }
    }
    free(acc); free(pg);
    if (rc) { copy_free(&kc); set_error(error, rc); return NULL; }
    tr_cert *c = cert_alloc(&kc.claim, top);
    copy_free(&kc);
    if (!c) { set_error(error, TR_MATH_MEMORY); return NULL; }
    c->depth = depth;
    c->evidence.unit = TR_CERT_UNIT_ROWS; c->evidence.holds = 1;
    c->premises = zalloc((uint64_t)n * sizeof *c->premises);
    if (!c->premises) { tr_cert_free(c); set_error(error, TR_MATH_MEMORY); return NULL; }
    for (int32_t i = 0; i < n; i++) {
        c->premises[i] = tr_cert_clone(premises[i], error);
        if (!c->premises[i]) { tr_cert_free(c); return NULL; }
        c->npremises = i + 1;
    }
    if (cert_seal(c)) { tr_cert_free(c); set_error(error, TR_MATH_MEMORY); return NULL; }
    set_error(error, TR_MATH_OK);
    return c;
}

/* ------------------------------------------------------------------ freshness and authorization */
static int32_t cert_mode(const tr_cert *c) {
    if (c->kind == TR_CERT_SCHEMA) return c->enforcement == TR_CERT_ENFORCE_WRITE ? TR_CERT_MATCH_SCHEMA : TR_CERT_MATCH_SUBSET;
    return TR_CERT_MATCH_EXACT;
}
int32_t tr_cert_fresh(const tr_cert *c, const tr_scope *current, const tr_cert_dim *dims, int32_t ndims,
                      tr_cert_staleness *out) {
    if (!cert_valid(c) || !out || !dims_ok(dims, ndims)) return TR_MATH_INVALID;
    memset(out, 0, sizeof *out);
    out->column = out->dim = -1;
    if (c->kind == TR_CERT_LAW) return TR_MATH_OK;
    if (!current) return TR_MATH_INVALID;
    if (c->npremises) {
        tr_cert_staleness all = *out;
        for (int32_t i = 0; i < c->npremises; i++) {
            tr_cert_staleness s;
            int32_t rc = tr_cert_fresh(c->premises[i], current, dims, ndims, &s);
            if (rc && rc != TR_MATH_STALE) return rc;
            if (s.first && (!all.first || s.first < all.first)) {
                all.first = s.first; all.column = s.column; all.dim = s.dim; all.recorded = s.recorded; all.current = s.current;
            }
            all.reasons |= s.reasons;
        }
        *out = all;
        return all.reasons ? TR_MATH_STALE : TR_MATH_OK;
    }
    return tr_cert_record_fresh(c->record, current, dims, ndims, cert_mode(c), out);
}
int32_t tr_cert_authorize_exact(const tr_cert *c, uint64_t field, const tr_scope *current, const tr_cert_dim *dims,
                                int32_t ndims, tr_cert_decision *out) {
    if (!cert_valid(c) || !out || !dims_ok(dims, ndims) || (field && (field >= TR_GFP_MAX || !tr_gfp_is_prime(field))))
        return TR_MATH_INVALID;
    memset(out, 0, sizeof *out);
    out->kind = c->kind; out->claim = c->claim.kind; out->policy = c->claim.policy; out->premise = -1;
    out->field = field; out->stale.column = out->stale.dim = -1;
    if (c->kind == TR_CERT_HYPOTHESIS) { out->reason = TR_CERT_REFUSE_STATISTICAL; return TR_MATH_DOMAIN_ERROR; }
    if (!claim_exact(&c->claim)) { out->reason = TR_CERT_REFUSE_APPROXIMATE; return TR_MATH_DOMAIN_ERROR; }
    int holds_in_target;
    if (c->kind == TR_CERT_LAW) holds_in_target = c->law == TR_CERT_LAW_RING_IDENTITY || field == c->claim.field;
    else holds_in_target = c->claim.field == 0 || field == c->claim.field;
    if (!holds_in_target) { out->reason = TR_CERT_REFUSE_CARRIER; return TR_MATH_DOMAIN_ERROR; }
    if (c->kind == TR_CERT_LAW) { out->authorized = 1; return TR_MATH_OK; }
    if (!current) { out->reason = TR_CERT_REFUSE_NO_SCOPE; return TR_MATH_DOMAIN_ERROR; }
    if (c->npremises) {
        for (int32_t i = 0; i < c->npremises; i++) {
            tr_cert_decision d;
            int32_t rc = tr_cert_authorize_exact(c->premises[i], field, current, dims, ndims, &d);
            if (rc == TR_MATH_INVALID) return rc;
            if (!d.authorized) {
                out->reason = TR_CERT_REFUSE_PREMISE; out->premise = i; out->stale = d.stale;
                return rc ? rc : TR_MATH_DOMAIN_ERROR;
            }
        }
        out->authorized = 1;
        return TR_MATH_OK;
    }
    if (!c->evidence.holds || c->evidence.overflow || c->evidence.violations) {
        out->reason = TR_CERT_REFUSE_INCOMPLETE; return TR_MATH_DOMAIN_ERROR;
    }
    int32_t rc = tr_cert_fresh(c, current, dims, ndims, &out->stale);
    if (rc == TR_MATH_STALE) { out->reason = TR_CERT_REFUSE_STALE; return rc; }
    if (rc) return rc;
    out->authorized = 1;
    return TR_MATH_OK;
}
