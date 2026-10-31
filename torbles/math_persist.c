/* FND-011: versioned binary persistence of typed mathematical states; see math_persist.h.
 * SPDX-License-Identifier: Apache-2.0 */
#include "math_persist.h"
#include "mathematics.h"
#include "math_domain.h"
#include <stdlib.h>
#include <string.h>

TR_MATH_ABI(tr_persist_header, 1);       /* the ctypes boundary of FND-011 (math_handles.h) */
TR_MATH_ABI(tr_persist_status, 1);
TR_MATH_ABI(tr_persist_bank_basis, 1);

_Static_assert(sizeof(tr_persist_header) == 96, "tr_persist_header ABI");
_Static_assert(sizeof(tr_persist_status) == 64, "tr_persist_status ABI");
_Static_assert(sizeof(tr_persist_bank_basis) == 32, "tr_persist_bank_basis ABI");

#define BLOB_MAGIC UINT32_C(0x424f4c42)
#define RECORD_MAGIC UINT32_C(0x44524550)
#define TYPE_VERSION 1

static void set_error(int32_t *error, int32_t code) { if (error) *error = code; }
static void *zalloc(uint64_t bytes) {
    if (bytes > (uint64_t)SIZE_MAX) return NULL;
    return calloc(1, bytes ? (size_t)bytes : 1);
}
static uint64_t pad8(uint64_t n) { return n > UINT64_MAX - 7 ? UINT64_MAX : (n + 7) & ~(uint64_t)7; }
static const tr_math_budget *budget_or(const tr_math_budget *b, tr_math_budget *dflt) {
    if (b) return b;
    tr_budget_default(dflt);
    return dflt;
}
static int32_t check_bytes(const tr_math_budget *budget, uint64_t bytes, tr_math_refusal *refusal) {
    tr_math_budget dflt;
    return tr_budget_check_one(budget_or(budget, &dflt), TR_BUDGET_BYTES, bytes, refusal);
}

/* ------------------------------------------------------------------ little-endian words */
static uint64_t le64(const uint8_t *p) {
    uint64_t v = 0;
    for (int i = 0; i < 8; i++) v |= (uint64_t)p[i] << (8 * i);
    return v;
}
static uint32_t le32(const uint8_t *p) {
    uint32_t v = 0;
    for (int i = 0; i < 4; i++) v |= (uint32_t)p[i] << (8 * i);
    return v;
}
static void put64(uint8_t *p, uint64_t v) { for (int i = 0; i < 8; i++) p[i] = (uint8_t)(v >> (8 * i)); }
static void put32(uint8_t *p, uint32_t v) { for (int i = 0; i < 4; i++) p[i] = (uint8_t)(v >> (8 * i)); }

typedef struct { uint8_t *p; uint64_t cap, len; int fail; } wbuf;
static void w_u64(wbuf *w, uint64_t v) {
    if (w->p) { if (w->len + 8 > w->cap) w->fail = 1; else put64(w->p + w->len, v); }
    w->len += 8;
}
static void w_i64(wbuf *w, int64_t v) { w_u64(w, (uint64_t)v); }
static void w_words(wbuf *w, const uint64_t *src, uint64_t n) { for (uint64_t i = 0; i < n; i++) w_u64(w, src[i]); }
static void w_raw(wbuf *w, const void *src, uint64_t len) {
    uint64_t padded = pad8(len);
    if (w->p) {
        if (w->len + padded > w->cap) w->fail = 1;
        else {
            if (len) memcpy(w->p + w->len, src, (size_t)len);
            memset(w->p + w->len + len, 0, (size_t)(padded - len));
        }
    }
    w->len += padded;
}
typedef struct { const uint8_t *p; uint64_t len, at; int fail; } rbuf;
static uint64_t r_u64(rbuf *r) {
    if (r->fail || r->at > r->len || r->len - r->at < 8) { r->fail = 1; return 0; }
    uint64_t v = le64(r->p + r->at);
    r->at += 8;
    return v;
}
static int64_t r_i64(rbuf *r) { return (int64_t)r_u64(r); }
static int32_t r_i32(rbuf *r) {
    int64_t v = r_i64(r);
    if (v < INT32_MIN || v > INT32_MAX) { r->fail = 1; return 0; }
    return (int32_t)v;
}
/* n words into an owned array (TR_MATH_INVALID when they are not there) */
static uint64_t *r_words(rbuf *r, uint64_t n) {
    if (r->fail || n > (r->len - r->at) / 8) { r->fail = 1; return NULL; }
    uint64_t *out = zalloc(n * 8);
    if (!out) { r->fail = 2; return NULL; }
    for (uint64_t i = 0; i < n; i++) out[i] = r_u64(r);
    return out;
}
static uint8_t *r_bytes(rbuf *r, uint64_t len) {
    uint64_t padded = pad8(len);
    if (r->fail || padded < len || r->at > r->len || r->len - r->at < padded) { r->fail = 1; return NULL; }
    for (uint64_t i = len; i < padded; i++) if (r->p[r->at + i]) { r->fail = 1; return NULL; }
    uint8_t *out = zalloc(len);
    if (!out) { r->fail = 2; return NULL; }
    if (len) memcpy(out, r->p + r->at, (size_t)len);
    r->at += padded;
    return out;
}

/* ------------------------------------------------------------------ the checksum */
uint64_t tr_persist_checksum_need(uint64_t words) { return tr_persist_device_bytes(words); }
void tr_persist_checksum(const uint64_t *words, uint64_t n, uint64_t out[2]) {
    uint64_t a = 0, b = 0;
    for (uint64_t i = 0; i < n; i++) { a += words[i] * tr_persist_weight(i, 0); b += words[i] * tr_persist_weight(i, 1); }
    out[0] = a; out[1] = b;
}
/* the record's checksum: every little-endian word but words 12 and 13 */
static void record_sum(const uint8_t *data, uint64_t total, uint64_t out[2]) {
    uint64_t a = 0, b = 0;
    for (uint64_t i = 0; i < total / 8; i++) {
        if (i == TR_PERSIST_CHECKSUM_WORD || i == TR_PERSIST_CHECKSUM_WORD + 1) continue;
        uint64_t w = le64(data + 8 * i);
        a += w * tr_persist_weight(i, 0); b += w * tr_persist_weight(i, 1);
    }
    out[0] = a; out[1] = b;
}

/* ------------------------------------------------------------------ blobs */
struct tr_persist_blob_s { uint32_t magic; uint64_t bytes; uint8_t *data; };
static int blob_valid(const tr_persist_blob *b) { return b && b->magic == BLOB_MAGIC; }
void tr_persist_blob_free(tr_persist_blob *b) { if (blob_valid(b)) { free(b->data); b->magic = 0; free(b); } }
uint64_t tr_persist_blob_size(const tr_persist_blob *b) { return blob_valid(b) ? b->bytes : 0; }
const uint8_t *tr_persist_blob_data(const tr_persist_blob *b) { return blob_valid(b) ? b->data : NULL; }
int32_t tr_persist_blob_copy(const tr_persist_blob *b, uint8_t *out, uint64_t capacity) {
    if (!blob_valid(b) || !out || capacity < b->bytes) return TR_MATH_INVALID;
    memcpy(out, b->data, (size_t)b->bytes);
    return TR_MATH_OK;
}

uint64_t tr_persist_save_need(uint64_t record_bytes) { return tr_budget_add(record_bytes, sizeof(tr_persist_blob)); }

/* writes a type's basis and payload sections (count pass when the buffers have no storage) */
typedef void (*section_fn)(wbuf *basis, wbuf *payload, const void *state);

static tr_persist_blob *assemble(int32_t type, int32_t carrier, int32_t exact, const tr_cert_record *source,
                                 section_fn fn, const void *state, const uint8_t *meta, uint64_t meta_bytes,
                                 const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error) {
    if (meta_bytes && !meta) { set_error(error, TR_MATH_INVALID); return NULL; }
    wbuf cb = {NULL, 0, 0, 0}, cp = {NULL, 0, 0, 0};
    fn(&cb, &cp, state);
    uint64_t sbytes = source ? tr_cert_record_serial_size(source) : 0;
    uint64_t total = TR_PERSIST_HEADER;
    total = tr_budget_add(total, pad8(sbytes));
    total = tr_budget_add(total, pad8(cb.len));
    total = tr_budget_add(total, pad8(meta_bytes));
    total = tr_budget_add(total, pad8(cp.len));
    int32_t rc = check_bytes(budget, tr_persist_save_need(total), refusal);
    if (rc) { set_error(error, rc); return NULL; }
    tr_persist_blob *b = zalloc(sizeof *b);
    uint8_t *d = b ? zalloc(total) : NULL;
    if (!b || !d) { free(b); free(d); set_error(error, TR_MATH_MEMORY); return NULL; }
    put64(d, TR_PERSIST_MAGIC); put32(d + 8, TR_PERSIST_FORMAT); put32(d + 12, TR_PERSIST_HEADER);
    put64(d + 16, TR_PERSIST_ENDIAN); put32(d + 24, (uint32_t)type); put32(d + 28, TYPE_VERSION);
    put32(d + 32, (uint32_t)carrier); put32(d + 36, (uint32_t)exact);
    put32(d + 40, (uint32_t)((source ? TR_PERSIST_SOURCE : 0) | TR_PERSIST_IMMUTABLE));
    put64(d + 48, sbytes); put64(d + 56, cb.len); put64(d + 64, meta_bytes); put64(d + 72, cp.len); put64(d + 80, total);
    uint64_t at = TR_PERSIST_HEADER;
    if (source && tr_cert_record_serialize(source, d + at, sbytes, NULL)) rc = TR_MATH_INVALID;
    at += pad8(sbytes);
    wbuf wb = {d + at, cb.len, 0, 0};
    wbuf wp = {d + at + pad8(cb.len) + pad8(meta_bytes), cp.len, 0, 0};
    fn(&wb, &wp, state);
    if (wb.fail || wp.fail || wb.len != cb.len || wp.len != cp.len) rc = TR_MATH_INVALID;
    at += pad8(cb.len);
    if (meta_bytes) memcpy(d + at, meta, (size_t)meta_bytes);
    uint64_t sum[2];
    record_sum(d, total, sum);
    put64(d + 96, sum[0]); put64(d + 104, sum[1]);
    if (rc) { free(d); free(b); set_error(error, rc); return NULL; }
    b->magic = BLOB_MAGIC; b->bytes = total; b->data = d;
    set_error(error, TR_MATH_OK);
    return b;
}
/* the FND-007 record of origin (every column, bound to its grouping), checked against the state's scope digest */
static tr_cert_record *source_of(const tr_scope *origin, int scoped, const uint64_t digest[2], int32_t *error) {
    tr_scope_info info;
    if (tr_scope_get_info(origin, &info)) { set_error(error, TR_MATH_INVALID); return NULL; }
    if (!scoped || info.digest[0] != digest[0] || info.digest[1] != digest[1]) {
        set_error(error, TR_MATH_DOMAIN_ERROR); return NULL;           /* the state was not built from this scope */
    }
    tr_math_budget unlimited;
    for (int i = 0; i < TR_BUDGET_RESOURCES; i++) unlimited.limit[i] = UINT64_MAX;
    return tr_cert_record_scope(origin, NULL, -1, TR_CERT_RECORD_GROUPS, NULL, 0, &unlimited, NULL, error);
}

/* ---- banks */
typedef struct { tr_bank_info info; tr_bank_states st; const uint8_t *exps; } bank_state;
static void bank_sections(wbuf *basis, wbuf *payload, const void *state) {
    const bank_state *s = state;
    uint64_t G = s->info.ngroups, m = (uint64_t)s->info.nterms;
    w_u64(basis, (uint64_t)s->info.ncols); w_u64(basis, m); w_u64(basis, (uint64_t)s->info.mode); w_u64(basis, G);
    w_raw(basis, s->exps, m * (uint64_t)s->info.ncols);
    w_words(payload, s->st.rows, G);
    if (m) {
        w_words(payload, s->st.count, G * m);
        for (uint64_t i = 0; i < G * m; i++) { w_i64(payload, s->st.sum[i].lo); w_i64(payload, s->st.sum[i].hi); }
    }
}
tr_persist_blob *tr_persist_bank(const tr_bank *bank, const tr_scope *origin, const uint8_t *meta, uint64_t meta_bytes,
                                 const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    bank_state s;
    if (!bank || tr_bank_get_info(bank, &s.info) || tr_bank_get_states(bank, &s.st)) { set_error(error, TR_MATH_INVALID); return NULL; }
    uint64_t eb = (uint64_t)s.info.nterms * (uint64_t)s.info.ncols;
    uint8_t *exps = zalloc(eb);
    if (!exps) { set_error(error, TR_MATH_MEMORY); return NULL; }
    if (tr_bank_get_exponents(bank, exps, eb)) { free(exps); set_error(error, TR_MATH_INVALID); return NULL; }
    s.exps = exps;
    tr_cert_record *src = NULL;
    if (origin && !(src = source_of(origin, s.info.scoped, s.info.digest, error))) { free(exps); return NULL; }
    tr_persist_blob *b = assemble(TR_PERSIST_BANK, TR_CARRIER_I128, 1, src, bank_sections, &s, meta, meta_bytes,
                                  budget, refusal, error);
    tr_cert_record_free(src);
    free(exps);
    return b;
}

/* ---- histograms */
typedef struct { tr_dist_hist_info info; tr_dist_states st; } hist_state;
static void hist_sections(wbuf *basis, wbuf *payload, const void *state) {
    const hist_state *s = state;
    const tr_dist_states *st = &s->st;
    uint64_t G = st->ngroups, n = st->ncells;
    w_i64(basis, st->k);
    for (int c = 0; c < 3; c++) w_i64(basis, s->info.col[c]);
    w_i64(basis, s->info.weight); w_i64(basis, st->repr); w_i64(basis, st->weighted);
    w_u64(basis, G); w_u64(basis, n); w_u64(basis, st->cells);
    for (int c = 0; c < 3; c++) { w_i64(basis, st->lo[c]); w_u64(basis, st->span[c]); w_u64(basis, st->dim[c]); }
    w_words(payload, st->rows, G);
    if (st->repr == TR_DIST_SPARSE) {
        w_words(payload, st->offset, G + 1);
        for (int c = 0; c < st->k; c++) for (uint64_t i = 0; i < n; i++) w_i64(payload, st->value[c][i]);
        w_raw(payload, st->nulls, n);
    }
    w_words(payload, st->count, n);
    if (st->weighted) {
        w_words(payload, st->wcount, n);
        for (uint64_t i = 0; i < n; i++) { w_i64(payload, st->wsum[i].lo); w_i64(payload, st->wsum[i].hi); }
    }
}
tr_persist_blob *tr_persist_histogram(const tr_dist_hist *h, const tr_scope *origin, const uint8_t *meta,
                                      uint64_t meta_bytes, const tr_math_budget *budget, tr_math_refusal *refusal,
                                      int32_t *error) {
    tr_budget_none(refusal);
    hist_state s;
    if (!h || tr_dist_hist_get_info(h, &s.info) || tr_dist_hist_get_states(h, &s.st)) { set_error(error, TR_MATH_INVALID); return NULL; }
    tr_cert_record *src = NULL;
    if (origin && !(src = source_of(origin, s.info.scoped, s.info.digest, error))) return NULL;
    tr_persist_blob *b = assemble(TR_PERSIST_HISTOGRAM, TR_CARRIER_I128, 1, src, hist_sections, &s, meta, meta_bytes,
                                  budget, refusal, error);
    tr_cert_record_free(src);
    return b;
}

/* ---- moments */
typedef struct { tr_lin_info info; tr_lin_states st; } lin_state;
static void lin_sections(wbuf *basis, wbuf *payload, const void *state) {
    const lin_state *s = state;
    uint64_t G = s->st.ngroups, S = (uint64_t)s->st.nslots;
    w_i64(basis, s->info.nfeatures); w_i64(basis, s->info.target);
    for (int i = 0; i < TR_LIN_MAX_FEATURES; i++) w_i64(basis, i < s->info.nfeatures ? s->info.column[1 + i] : -1);
    w_i64(basis, s->info.policy); w_i64(basis, s->info.intercept); w_i64(basis, s->info.width);
    w_i64(basis, s->info.marginals); w_i64(basis, s->st.nslots); w_i64(basis, s->st.words); w_u64(basis, G);
    w_words(payload, s->st.rows, G);
    w_words(payload, s->st.count, G * S);
    w_words(payload, s->st.sum, G * S * (uint64_t)s->st.words);
}
tr_persist_blob *tr_persist_moments(const tr_lin_moments *m, const tr_scope *origin, const uint8_t *meta,
                                    uint64_t meta_bytes, const tr_math_budget *budget, tr_math_refusal *refusal,
                                    int32_t *error) {
    tr_budget_none(refusal);
    lin_state s;
    if (!m || tr_lin_moments_get_info(m, &s.info) || tr_lin_moments_get_states(m, &s.st)) { set_error(error, TR_MATH_INVALID); return NULL; }
    tr_cert_record *src = NULL;
    if (origin && !(src = source_of(origin, s.info.scoped, s.info.digest, error))) return NULL;
    tr_persist_blob *b = assemble(TR_PERSIST_MOMENTS, TR_CARRIER_I128, 1, src, lin_sections, &s, meta, meta_bytes,
                                  budget, refusal, error);
    tr_cert_record_free(src);
    return b;
}

/* ---- carriers */
typedef struct { int32_t sign; uint64_t n; const uint64_t *limbs; int32_t dsign; uint64_t dn; const uint64_t *dlimbs; } big_state;
static void bigint_sections(wbuf *basis, wbuf *payload, const void *state) {
    const big_state *s = state;
    (void)basis;
    w_i64(payload, s->sign); w_u64(payload, s->n); w_words(payload, s->limbs, s->n);
}
static void bigq_sections(wbuf *basis, wbuf *payload, const void *state) {
    const big_state *s = state;
    (void)basis;
    w_i64(payload, s->sign); w_u64(payload, s->n); w_words(payload, s->limbs, s->n);
    w_u64(payload, s->dn); w_words(payload, s->dlimbs, s->dn);
}
static int32_t export_big(const tr_bigint *x, int32_t *sign, uint64_t **limbs, uint64_t *n) {
    uint64_t count = tr_bigint_limb_count(x);
    *limbs = zalloc(count * 8);
    if (!*limbs) return TR_MATH_MEMORY;
    int32_t rc = tr_bigint_export(x, sign, *limbs, count, n);
    if (rc) { free(*limbs); *limbs = NULL; }
    return rc;
}
tr_persist_blob *tr_persist_bigint(const tr_bigint *x, const uint8_t *meta, uint64_t meta_bytes,
                                   const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    if (!x || !tr_bigint_is_handle(x)) { set_error(error, TR_MATH_INVALID); return NULL; }
    big_state s;
    memset(&s, 0, sizeof s);
    uint64_t *limbs = NULL;
    int32_t rc = export_big(x, &s.sign, &limbs, &s.n);
    if (rc) { set_error(error, rc); return NULL; }
    s.limbs = limbs;
    tr_persist_blob *b = assemble(TR_PERSIST_BIGINT, TR_CARRIER_BIGINT, 1, NULL, bigint_sections, &s, meta, meta_bytes,
                                  budget, refusal, error);
    free(limbs);
    return b;
}
tr_persist_blob *tr_persist_bigq(const tr_bigq *q, const uint8_t *meta, uint64_t meta_bytes, const tr_math_budget *budget,
                                 tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    if (!q || !tr_bigq_is_handle(q)) { set_error(error, TR_MATH_INVALID); return NULL; }
    big_state s;
    memset(&s, 0, sizeof s);
    uint64_t *nl = NULL, *dl = NULL;
    int32_t rc = export_big(tr_bigq_numerator(q), &s.sign, &nl, &s.n);
    if (!rc) rc = export_big(tr_bigq_denominator(q), &s.dsign, &dl, &s.dn);
    if (rc) { free(nl); free(dl); set_error(error, rc); return NULL; }
    s.limbs = nl; s.dlimbs = dl;
    tr_persist_blob *b = assemble(TR_PERSIST_BIGQ, TR_CARRIER_BIGQ, 1, NULL, bigq_sections, &s, meta, meta_bytes,
                                  budget, refusal, error);
    free(nl); free(dl);
    return b;
}
static void gfp_sections(wbuf *basis, wbuf *payload, const void *state) {
    const tr_gfp_elem *e = state;
    (void)basis;
    w_u64(payload, e->p); w_u64(payload, e->v);
}
tr_persist_blob *tr_persist_gfp(tr_gfp_elem e, const uint8_t *meta, uint64_t meta_bytes, const tr_math_budget *budget,
                                tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    if (e.p < 2 || e.p >= TR_GFP_MAX || !tr_gfp_is_prime(e.p) || e.v >= e.p) { set_error(error, TR_MATH_INVALID); return NULL; }
    return assemble(TR_PERSIST_GFP, TR_CARRIER_GFP, 1, NULL, gfp_sections, &e, meta, meta_bytes, budget, refusal, error);
}

/* ---- certificates */
typedef struct { const uint8_t *bytes; uint64_t n; } cert_state;
static void cert_sections(wbuf *basis, wbuf *payload, const void *state) {
    const cert_state *s = state;
    (void)basis;
    w_u64(payload, s->n); w_raw(payload, s->bytes, s->n);
}
tr_persist_blob *tr_persist_certificate(const tr_cert *c, const uint8_t *meta, uint64_t meta_bytes,
                                        const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    tr_cert_info info;
    if (!c || tr_cert_get_info(c, &info)) { set_error(error, TR_MATH_INVALID); return NULL; }
    uint64_t n = tr_cert_serial_size(c);
    uint8_t *buf = zalloc(n);
    if (!buf) { set_error(error, TR_MATH_MEMORY); return NULL; }
    if (tr_cert_serialize(c, buf, n, NULL)) { free(buf); set_error(error, TR_MATH_INVALID); return NULL; }
    cert_state s = {buf, n};
    /* the source section: the certificate's record with digests only (its exact selection stays in the certificate) */
    const tr_cert_record *rec = tr_cert_get_record(c);
    tr_cert_record *src = NULL;
    if (rec) {
        tr_cert_source fixed;
        tr_cert_record_get(rec, &fixed);
        src = tr_cert_record_copy(rec, fixed.flags & ~TR_CERT_RECORD_SELECTION, error);
        if (!src) { free(buf); return NULL; }
    }
    tr_persist_blob *b = assemble(TR_PERSIST_CERTIFICATE, info.carrier, info.exact, src, cert_sections, &s, meta,
                                  meta_bytes, budget, refusal, error);
    tr_cert_record_free(src);
    free(buf);
    return b;
}

/* ------------------------------------------------------------------ loading */
static void status_init(tr_persist_status *s, uint64_t bytes) {
    if (!s) return;
    memset(s, 0, sizeof *s);
    s->bytes = bytes;
}
static int32_t fail_status(tr_persist_status *s, int32_t reason, int32_t code) { if (s) s->reason = reason; return code; }
static int type_known(int32_t t) { return t >= TR_PERSIST_BANK && t < TR_PERSIST_TYPES; }

int32_t tr_persist_header_read(const uint8_t *d, uint64_t bytes, tr_persist_header *h, tr_persist_status *st) {
    status_init(st, bytes);
    if (!d || !h) return fail_status(st, TR_PERSIST_MALFORMED, TR_MATH_INVALID);
    memset(h, 0, sizeof *h);
    if (bytes < TR_PERSIST_HEADER) { if (st) st->needed = TR_PERSIST_HEADER; return fail_status(st, TR_PERSIST_TRUNCATED, TR_MATH_INVALID); }
    if (le64(d) != TR_PERSIST_MAGIC) return fail_status(st, TR_PERSIST_MAGIC_ERROR, TR_MATH_INVALID);
    h->endian = le64(d + 16);
    h->format = le32(d + 8); h->header_bytes = le32(d + 12);
    h->type = (int32_t)le32(d + 24); h->type_version = (int32_t)le32(d + 28);
    h->carrier = (int32_t)le32(d + 32); h->exact = (int32_t)le32(d + 36); h->flags = (int32_t)le32(d + 40);
    h->reserved = (int32_t)le32(d + 44);
    h->source_bytes = le64(d + 48); h->basis_bytes = le64(d + 56); h->metadata_bytes = le64(d + 64);
    h->payload_bytes = le64(d + 72); h->total_bytes = le64(d + 80);
    h->checksum[0] = le64(d + 96); h->checksum[1] = le64(d + 104);
    if (st) { st->type = h->type; st->format = h->format; st->type_version = (uint32_t)h->type_version; st->needed = h->total_bytes; }
    if (h->endian != TR_PERSIST_ENDIAN) return fail_status(st, TR_PERSIST_ENDIAN_ERROR, TR_MATH_UNSUPPORTED);
    if (h->format != TR_PERSIST_FORMAT || h->header_bytes != TR_PERSIST_HEADER)
        return fail_status(st, TR_PERSIST_VERSION, TR_MATH_UNSUPPORTED);
    if (bytes < h->total_bytes) return fail_status(st, TR_PERSIST_TRUNCATED, TR_MATH_INVALID);
    uint64_t sum = TR_PERSIST_HEADER;
    sum = tr_budget_add(sum, pad8(h->source_bytes)); sum = tr_budget_add(sum, pad8(h->basis_bytes));
    sum = tr_budget_add(sum, pad8(h->metadata_bytes)); sum = tr_budget_add(sum, pad8(h->payload_bytes));
    if (bytes > h->total_bytes || sum != h->total_bytes || h->total_bytes % 8 || le64(d + 88) || le64(d + 112) ||
        le64(d + 120) || h->reserved || (h->flags & ~(TR_PERSIST_SOURCE | TR_PERSIST_IMMUTABLE)) ||
        !(h->flags & TR_PERSIST_IMMUTABLE) || (!(h->flags & TR_PERSIST_SOURCE) && h->source_bytes))
        return fail_status(st, TR_PERSIST_MALFORMED, TR_MATH_INVALID);
    return TR_MATH_OK;
}
static int32_t checksum_check(const uint8_t *d, const tr_persist_header *h, const uint64_t sum[2], tr_persist_status *st) {
    if (st) { st->stored[0] = h->checksum[0]; st->stored[1] = h->checksum[1]; st->computed[0] = sum[0]; st->computed[1] = sum[1]; }
    (void)d;
    if (sum[0] != h->checksum[0] || sum[1] != h->checksum[1]) return fail_status(st, TR_PERSIST_CORRUPT, TR_MATH_INVALID);
    if (!type_known(h->type)) return fail_status(st, TR_PERSIST_MALFORMED, TR_MATH_INVALID);
    if (h->type_version != TYPE_VERSION) return fail_status(st, TR_PERSIST_VERSION, TR_MATH_UNSUPPORTED);
    return TR_MATH_OK;
}
int32_t tr_persist_validate(const uint8_t *d, uint64_t bytes, tr_persist_status *st) {
    tr_persist_header h;
    int32_t rc = tr_persist_header_read(d, bytes, &h, st);
    if (rc) return rc;
    uint64_t sum[2];
    record_sum(d, h.total_bytes, sum);
    return checksum_check(d, &h, sum, st);
}
int32_t tr_persist_validate_sum(const uint8_t *d, uint64_t bytes, const uint64_t full[2], tr_persist_status *st) {
    if (!full) { status_init(st, bytes); return fail_status(st, TR_PERSIST_MALFORMED, TR_MATH_INVALID); }
    tr_persist_header h;
    int32_t rc = tr_persist_header_read(d, bytes, &h, st);
    if (rc) return rc;
    uint64_t sum[2] = {full[0], full[1]};
    for (int i = 0; i < 2; i++) {                         /* the checksum words do not count */
        uint64_t w = le64(d + 8 * (TR_PERSIST_CHECKSUM_WORD + i));
        sum[0] -= w * tr_persist_weight(TR_PERSIST_CHECKSUM_WORD + (uint64_t)i, 0);
        sum[1] -= w * tr_persist_weight(TR_PERSIST_CHECKSUM_WORD + (uint64_t)i, 1);
    }
    return checksum_check(d, &h, sum, st);
}

struct tr_persist_record_s {
    uint32_t magic;
    tr_persist_header head;
    tr_cert_record *source;
    uint8_t *meta;
    /* typed contents */
    tr_persist_bank_basis bank;
    uint8_t *exps;
    tr_dist_spec hspec;
    tr_lin_spec lspec;
    int32_t marginals;
    uint64_t ngroups, ncells, cells, nslots, words;
    int32_t k, repr, weighted, reserved;
    int64_t lo[3];
    uint64_t span[3], dim[3];
    uint64_t *rows, *count, *sums, *offset, *wcount;
    int64_t *value[3];
    uint8_t *nulls;
    tr_math_i128 *sum128, *wsum;
    int32_t sign, dsign;
    uint64_t nlimbs, dlimbs;
    uint64_t *limbs, *dlimb;
    tr_gfp_elem elem;
    uint8_t *cert;
    uint64_t cert_bytes, bytes;
};
static int record_valid(const tr_persist_record *r) { return r && r->magic == RECORD_MAGIC; }
uint64_t tr_persist_load_need(uint64_t record_bytes) {
    return tr_budget_add(tr_budget_mul(record_bytes, 2), sizeof(tr_persist_record));
}
void tr_persist_record_free(tr_persist_record *r) {
    if (!record_valid(r)) return;
    tr_cert_record_free(r->source);
    free(r->meta); free(r->exps); free(r->rows); free(r->count); free(r->sums); free(r->offset); free(r->wcount);
    for (int c = 0; c < 3; c++) free(r->value[c]);
    free(r->nulls); free(r->sum128); free(r->wsum); free(r->limbs); free(r->dlimb); free(r->cert);
    r->magic = 0;
    free(r);
}
static tr_math_i128 *pairs_i128(rbuf *rd, uint64_t n) {
    if (rd->fail || n > (rd->len - rd->at) / 16) { rd->fail = 1; return NULL; }
    tr_math_i128 *out = zalloc(n * 16);
    if (!out) { rd->fail = 2; return NULL; }
    for (uint64_t i = 0; i < n; i++) { out[i].lo = r_i64(rd); out[i].hi = r_i64(rd); }
    return out;
}
static int32_t parse_bank(tr_persist_record *r, rbuf *b, rbuf *p) {
    int64_t ncols = r_i64(b), nterms = r_i64(b), mode = r_i64(b);
    uint64_t G = r_u64(b);
    if (b->fail || ncols < 0 || ncols > TR_SCOPE_MAX_COLUMNS || nterms < 0 || nterms > TR_BANK_MAX_TERMS ||
        (mode != TR_BANK_SQL && mode != TR_BANK_JOINT) || G < 1 || G > TR_SCOPE_MAX_GROUPS) return TR_MATH_INVALID;
    r->exps = r_bytes(b, (uint64_t)nterms * (uint64_t)ncols);
    uint64_t m = (uint64_t)nterms;
    if (tr_budget_mul(G, tr_budget_add(8, tr_budget_mul(m, 24))) != p->len) return TR_MATH_INVALID;
    r->rows = r_words(p, G);
    if (m) { r->count = r_words(p, G * m); r->sum128 = pairs_i128(p, G * m); }
    if (b->fail || p->fail || b->at != b->len || p->at != p->len) return b->fail == 2 || p->fail == 2 ? TR_MATH_MEMORY : TR_MATH_INVALID;
    r->bank.ncols = (int32_t)ncols; r->bank.nterms = (int32_t)nterms; r->bank.mode = (int32_t)mode;
    r->bank.ngroups = G; r->bank.exponents = r->exps;
    return TR_MATH_OK;
}
static int32_t parse_hist(tr_persist_record *r, rbuf *b, rbuf *p) {
    tr_dist_spec *s = &r->hspec;
    memset(s, 0, sizeof *s);
    int64_t k = r_i64(b);
    int64_t col[3];
    for (int c = 0; c < 3; c++) col[c] = r_i64(b);
    int64_t weight = r_i64(b), repr = r_i64(b), weighted = r_i64(b);
    uint64_t G = r_u64(b), n = r_u64(b), cells = r_u64(b);
    for (int c = 0; c < 3; c++) { r->lo[c] = r_i64(b); r->span[c] = r_u64(b); r->dim[c] = r_u64(b); }
    if (b->fail || b->at != b->len || k < 1 || k > TR_DIST_MAX_DIMS || (repr != TR_DIST_DENSE && repr != TR_DIST_SPARSE) ||
        (weighted != 0 && weighted != 1) || G < 1 || G > TR_SCOPE_MAX_GROUPS || n > (UINT64_C(1) << 40) ||
        weight < -1 || weight >= TR_SCOPE_MAX_COLUMNS || (weighted != (weight >= 0))) return TR_MATH_INVALID;
    for (int c = 0; c < 3; c++) {
        if ((c < k && (col[c] < -1 || col[c] >= TR_SCOPE_MAX_COLUMNS)) || (c >= k && col[c] != -1)) return TR_MATH_INVALID;
        if (repr == TR_DIST_DENSE && c < k && (r->dim[c] < r->span[c] || r->dim[c] - r->span[c] > 1)) return TR_MATH_INVALID;
    }
    if (repr == TR_DIST_DENSE) {
        uint64_t cc = 1;
        for (int c = 0; c < k; c++) cc = tr_budget_mul(cc, r->dim[c]);
        if (cc != cells || tr_budget_mul(G, cells) != n) return TR_MATH_INVALID;
    } else if (cells) return TR_MATH_INVALID;
    uint64_t need = tr_budget_mul(G, 8);
    if (repr == TR_DIST_SPARSE)
        need = tr_budget_add(need, tr_budget_add(tr_budget_mul(G + 1, 8), tr_budget_add(tr_budget_mul(n, 8 * (uint64_t)k), pad8(n))));
    need = tr_budget_add(need, tr_budget_mul(n, weighted ? 32 : 8));
    if (need != p->len) return TR_MATH_INVALID;
    r->rows = r_words(p, G);
    if (repr == TR_DIST_SPARSE) {
        r->offset = r_words(p, G + 1);
        for (int c = 0; c < k; c++) r->value[c] = (int64_t *)r_words(p, n);
        r->nulls = r_bytes(p, n);
    }
    r->count = r_words(p, n);
    if (weighted) { r->wcount = r_words(p, n); r->wsum = pairs_i128(p, n); }
    if (p->fail || p->at != p->len) return p->fail == 2 ? TR_MATH_MEMORY : TR_MATH_INVALID;
    r->k = (int32_t)k; r->repr = (int32_t)repr; r->weighted = (int32_t)weighted; r->ngroups = G; r->ncells = n; r->cells = cells;
    s->k = (int32_t)k;
    for (int c = 0; c < 3; c++) s->col[c] = (int32_t)col[c];
    s->weight = (int32_t)weight; s->repr = (int32_t)repr;
    if (repr == TR_DIST_DENSE) {                 /* the stored domain, as an explicit one */
        s->domain = TR_DIST_DOMAIN_EXPLICIT;
        for (int c = 0; c < k; c++) {
            s->lo[c] = r->lo[c];
            s->hi[c] = (int64_t)((uint64_t)r->lo[c] + r->span[c] - 1);
            s->nullslot[c] = (int32_t)(r->dim[c] - r->span[c]);
        }
    }
    s->strategy = TR_DIST_STRATEGY_AUTO;
    return TR_MATH_OK;
}
static int32_t parse_lin(tr_persist_record *r, rbuf *b, rbuf *p) {
    tr_lin_spec *s = &r->lspec;
    memset(s, 0, sizeof *s);
    s->nfeatures = r_i32(b); s->target = r_i32(b);
    for (int i = 0; i < TR_LIN_MAX_FEATURES; i++) s->feature[i] = r_i32(b);
    s->policy = r_i32(b); s->intercept = r_i32(b); s->width = r_i32(b);
    int32_t marginals = r_i32(b), nslots = r_i32(b), words = r_i32(b);
    uint64_t G = r_u64(b);
    if (b->fail || b->at != b->len || s->nfeatures < 0 || s->nfeatures > TR_LIN_MAX_FEATURES || nslots < 1 ||
        nslots > TR_LIN_MAX_SLOTS || words != TR_LIN_WORDS || G < 1 || G > TR_SCOPE_MAX_GROUPS ||
        (marginals != 0 && marginals != 1) || (s->width != 128 && s->width != 192)) return TR_MATH_INVALID;
    uint64_t S = (uint64_t)nslots;
    if (tr_budget_mul(G, tr_budget_add(8, tr_budget_mul(S, 8 + 8 * TR_LIN_WORDS))) != p->len) return TR_MATH_INVALID;
    r->rows = r_words(p, G);
    r->count = r_words(p, G * S);
    r->sums = r_words(p, G * S * TR_LIN_WORDS);
    if (p->fail || p->at != p->len) return p->fail == 2 ? TR_MATH_MEMORY : TR_MATH_INVALID;
    r->marginals = marginals; r->ngroups = G; r->nslots = S; r->words = (uint64_t)words;
    return TR_MATH_OK;
}
static int32_t parse_big(tr_persist_record *r, rbuf *p, int rational) {
    int64_t sign = r_i64(p);
    uint64_t n = r_u64(p);
    if (p->fail || sign < -1 || sign > 1 || n > TR_BIGINT_MAX_LIMBS || (n == 0) != (sign == 0)) return TR_MATH_INVALID;
    r->limbs = r_words(p, n);
    if (n && !r->limbs) return p->fail == 2 ? TR_MATH_MEMORY : TR_MATH_INVALID;
    if (n && r->limbs[n - 1] == 0) return TR_MATH_INVALID;            /* normalized */
    r->sign = (int32_t)sign; r->nlimbs = n;
    if (rational) {
        uint64_t dn = r_u64(p);
        if (p->fail || dn == 0 || dn > TR_BIGINT_MAX_LIMBS) return TR_MATH_INVALID;
        r->dlimb = r_words(p, dn);
        if (!r->dlimb) return p->fail == 2 ? TR_MATH_MEMORY : TR_MATH_INVALID;
        if (r->dlimb[dn - 1] == 0) return TR_MATH_INVALID;
        r->dsign = 1; r->dlimbs = dn;
    }
    return p->at == p->len ? TR_MATH_OK : TR_MATH_INVALID;
}

tr_persist_record *tr_persist_load(const uint8_t *d, uint64_t bytes, int32_t expected, const tr_math_budget *budget,
                                   tr_math_refusal *refusal, tr_persist_status *st, int32_t *error) {
    tr_budget_none(refusal);
    tr_persist_status local;
    if (!st) st = &local;
    int32_t rc = tr_persist_validate(d, bytes, st);
    if (rc) { set_error(error, rc); return NULL; }
    tr_persist_header h;
    tr_persist_header_read(d, bytes, &h, NULL);
    if (expected < 0 || expected >= TR_PERSIST_TYPES) { set_error(error, TR_MATH_INVALID); return NULL; }
    if (expected && h.type != expected) { st->reason = TR_PERSIST_TYPE; set_error(error, TR_MATH_DOMAIN_ERROR); return NULL; }
    rc = check_bytes(budget, tr_persist_load_need(h.total_bytes), refusal);
    if (rc) { set_error(error, rc); return NULL; }
    tr_persist_record *r = zalloc(sizeof *r);
    if (!r) { set_error(error, TR_MATH_MEMORY); return NULL; }
    r->magic = RECORD_MAGIC;
    r->head = h;
    r->bytes = tr_budget_add(h.total_bytes, sizeof *r);
    uint64_t at = TR_PERSIST_HEADER;
    const uint8_t *src = d + at;
    at += pad8(h.source_bytes);
    rbuf basis = {d + at, h.basis_bytes, 0, 0};
    at += pad8(h.basis_bytes);
    const uint8_t *meta = d + at;
    at += pad8(h.metadata_bytes);
    rbuf payload = {d + at, h.payload_bytes, 0, 0};
    rc = TR_MATH_OK;
    if (h.flags & TR_PERSIST_SOURCE) {
        tr_math_budget unlimited;
        for (int i = 0; i < TR_BUDGET_RESOURCES; i++) unlimited.limit[i] = UINT64_MAX;
        uint64_t used = 0;
        int32_t e = 0;
        r->source = tr_cert_record_deserialize(src, h.source_bytes, &used, &unlimited, NULL, &e);
        if (!r->source || used != h.source_bytes) rc = e == TR_MATH_MEMORY ? e : TR_MATH_INVALID;
    }
    if (!rc) {
        r->meta = zalloc(h.metadata_bytes);
        if (!r->meta) rc = TR_MATH_MEMORY;
        else if (h.metadata_bytes) memcpy(r->meta, meta, (size_t)h.metadata_bytes);
        for (uint64_t i = h.metadata_bytes; !rc && i < pad8(h.metadata_bytes); i++) if (meta[i]) rc = TR_MATH_INVALID;
    }
    if (!rc) {
        switch (h.type) {
        case TR_PERSIST_BANK: rc = parse_bank(r, &basis, &payload); break;
        case TR_PERSIST_HISTOGRAM: rc = parse_hist(r, &basis, &payload); break;
        case TR_PERSIST_MOMENTS: rc = parse_lin(r, &basis, &payload); break;
        case TR_PERSIST_BIGINT: rc = h.basis_bytes ? TR_MATH_INVALID : parse_big(r, &payload, 0); break;
        case TR_PERSIST_BIGQ: rc = h.basis_bytes ? TR_MATH_INVALID : parse_big(r, &payload, 1); break;
        case TR_PERSIST_GFP:
            r->elem.p = r_u64(&payload); r->elem.v = r_u64(&payload);
            rc = h.basis_bytes || payload.fail || payload.at != payload.len || r->elem.p < 2 || r->elem.p >= TR_GFP_MAX ||
                 !tr_gfp_is_prime(r->elem.p) || r->elem.v >= r->elem.p ? TR_MATH_INVALID : TR_MATH_OK;
            break;
        case TR_PERSIST_CERTIFICATE: {
            uint64_t n = r_u64(&payload);
            if (h.basis_bytes || payload.fail || n > payload.len) { rc = TR_MATH_INVALID; break; }
            r->cert = r_bytes(&payload, n);
            r->cert_bytes = n;
            rc = !r->cert ? (payload.fail == 2 ? TR_MATH_MEMORY : TR_MATH_INVALID) : (payload.at != payload.len ? TR_MATH_INVALID : TR_MATH_OK);
            break;
        }
        default: rc = TR_MATH_INVALID;
        }
    }
    if (!rc && (h.flags & TR_PERSIST_SOURCE) && h.type != TR_PERSIST_BANK && h.type != TR_PERSIST_HISTOGRAM &&
        h.type != TR_PERSIST_MOMENTS && h.type != TR_PERSIST_CERTIFICATE) rc = TR_MATH_INVALID;
    if (rc) {
        tr_persist_record_free(r);
        st->reason = rc == TR_MATH_MEMORY ? TR_PERSIST_OK : TR_PERSIST_MALFORMED;
        set_error(error, rc);
        return NULL;
    }
    set_error(error, TR_MATH_OK);
    return r;
}
int32_t tr_persist_get_header(const tr_persist_record *r, tr_persist_header *out) {
    if (!record_valid(r) || !out) return TR_MATH_INVALID;
    *out = r->head;
    return TR_MATH_OK;
}
const tr_cert_record *tr_persist_get_source(const tr_persist_record *r) { return record_valid(r) ? r->source : NULL; }
int32_t tr_persist_get_metadata(const tr_persist_record *r, uint8_t *out, uint64_t capacity, uint64_t *length) {
    if (!record_valid(r) || (!out && capacity)) return TR_MATH_INVALID;
    if (length) *length = r->head.metadata_bytes;
    if (capacity < r->head.metadata_bytes) return out ? TR_MATH_INVALID : TR_MATH_OK;
    if (r->head.metadata_bytes) memcpy(out, r->meta, (size_t)r->head.metadata_bytes);
    return TR_MATH_OK;
}
int32_t tr_persist_fresh(const tr_persist_record *r, const tr_scope *current, const tr_cert_dim *dims, int32_t ndims,
                         tr_cert_staleness *out) {
    if (!record_valid(r) || !current || !out) return TR_MATH_INVALID;
    if (!r->source) return TR_MATH_UNSUPPORTED;
    return tr_cert_record_fresh(r->source, current, dims, ndims, TR_CERT_MATCH_EXACT, out);
}
int32_t tr_persist_bank_view(const tr_persist_record *r, tr_persist_bank_basis *basis, tr_bank_states *st) {
    if (!record_valid(r) || r->head.type != TR_PERSIST_BANK) return TR_MATH_INVALID;
    if (basis) *basis = r->bank;
    if (st) {
        memset(st, 0, sizeof *st);
        st->ngroups = r->bank.ngroups; st->nterms = r->bank.nterms;
        st->rows = r->rows; st->count = r->count; st->sum = r->sum128;
    }
    return TR_MATH_OK;
}
int32_t tr_persist_histogram_view(const tr_persist_record *r, tr_dist_spec *spec, tr_dist_states *st) {
    if (!record_valid(r) || r->head.type != TR_PERSIST_HISTOGRAM) return TR_MATH_INVALID;
    if (spec) *spec = r->hspec;
    if (st) {
        memset(st, 0, sizeof *st);
        st->ngroups = r->ngroups; st->ncells = r->ncells; st->capacity = r->ncells; st->cells = r->cells;
        st->k = r->k; st->repr = r->repr; st->weighted = r->weighted;
        for (int c = 0; c < 3; c++) { st->lo[c] = r->lo[c]; st->span[c] = r->span[c]; st->dim[c] = r->dim[c]; st->value[c] = r->value[c]; }
        st->rows = r->rows; st->offset = r->offset; st->nulls = r->nulls; st->count = r->count;
        st->wcount = r->wcount; st->wsum = r->wsum;
    }
    return TR_MATH_OK;
}
int32_t tr_persist_moments_view(const tr_persist_record *r, tr_lin_spec *spec, int32_t *marginals, tr_lin_states *st) {
    if (!record_valid(r) || r->head.type != TR_PERSIST_MOMENTS) return TR_MATH_INVALID;
    if (spec) *spec = r->lspec;
    if (marginals) *marginals = r->marginals;
    if (st) {
        memset(st, 0, sizeof *st);
        st->ngroups = r->ngroups; st->nslots = (int32_t)r->nslots; st->words = (int32_t)r->words;
        st->rows = r->rows; st->count = r->count; st->sum = r->sums;
    }
    return TR_MATH_OK;
}
/* origin must be the recorded source, exactly */
static int32_t origin_ok(const tr_persist_record *r, const tr_scope *origin, tr_cert_staleness *stale) {
    tr_cert_staleness local;
    if (!stale) stale = &local;
    memset(stale, 0, sizeof *stale);
    stale->column = stale->dim = -1;
    if (!r->source) return TR_MATH_DOMAIN_ERROR;            /* an unscoped record has no source to be fresh against */
    return tr_cert_record_fresh(r->source, origin, NULL, 0, TR_CERT_MATCH_EXACT, stale);
}
tr_bank *tr_persist_to_bank(const tr_persist_record *r, const tr_scope *origin, const tr_math_budget *budget,
                            tr_math_refusal *refusal, tr_cert_staleness *stale, int32_t *error) {
    tr_budget_none(refusal);
    if (!record_valid(r) || r->head.type != TR_PERSIST_BANK) { set_error(error, TR_MATH_INVALID); return NULL; }
    if (origin) {
        int32_t rc = origin_ok(r, origin, stale);
        if (rc) { set_error(error, rc); return NULL; }
    }
    tr_bank_states st;
    tr_persist_bank_view(r, NULL, &st);
    return tr_bank_from_states(origin, r->bank.ncols, r->exps, r->bank.nterms, r->bank.mode, &st, budget, refusal, error);
}
tr_dist_hist *tr_persist_to_histogram(const tr_persist_record *r, const tr_scope *origin, const tr_math_budget *budget,
                                      tr_math_refusal *refusal, tr_cert_staleness *stale, int32_t *error) {
    tr_budget_none(refusal);
    if (!record_valid(r) || r->head.type != TR_PERSIST_HISTOGRAM || !origin) { set_error(error, TR_MATH_INVALID); return NULL; }
    int32_t rc = origin_ok(r, origin, stale);
    if (rc) { set_error(error, rc); return NULL; }
    tr_dist_states st;
    tr_dist_spec spec;
    tr_persist_histogram_view(r, &spec, &st);
    return tr_dist_hist_from_states(origin, &spec, &st, budget, refusal, error);
}
tr_lin_moments *tr_persist_to_moments(const tr_persist_record *r, const tr_scope *origin, const tr_math_budget *budget,
                                      tr_math_refusal *refusal, tr_cert_staleness *stale, int32_t *error) {
    tr_budget_none(refusal);
    if (!record_valid(r) || r->head.type != TR_PERSIST_MOMENTS || !origin) { set_error(error, TR_MATH_INVALID); return NULL; }
    int32_t rc = origin_ok(r, origin, stale);
    if (rc) { set_error(error, rc); return NULL; }
    tr_lin_states st;
    tr_lin_spec spec;
    int32_t marginals = 0;
    tr_persist_moments_view(r, &spec, &marginals, &st);
    return tr_lin_moments_from_states(origin, &spec, marginals, &st, budget, refusal, error);
}
tr_bigint *tr_persist_to_bigint(const tr_persist_record *r, const tr_math_budget *budget, tr_math_refusal *refusal,
                                int32_t *error) {
    if (!record_valid(r) || r->head.type != TR_PERSIST_BIGINT) { tr_budget_none(refusal); set_error(error, TR_MATH_INVALID); return NULL; }
    return tr_bigint_from_limbs(r->sign, r->limbs, r->nlimbs, budget, refusal, error);
}
tr_bigq *tr_persist_to_bigq(const tr_persist_record *r, const tr_math_budget *budget, tr_math_refusal *refusal,
                            int32_t *error) {
    if (!record_valid(r) || r->head.type != TR_PERSIST_BIGQ) { tr_budget_none(refusal); set_error(error, TR_MATH_INVALID); return NULL; }
    tr_bigint *n = tr_bigint_from_limbs(r->sign, r->limbs, r->nlimbs, budget, refusal, error);
    if (!n) return NULL;
    tr_bigint *d = tr_bigint_from_limbs(r->dsign, r->dlimb, r->dlimbs, budget, refusal, error);
    if (!d) { tr_bigint_free(n); return NULL; }
    tr_bigq *q = tr_bigq_from_bigints(n, d, budget, refusal, error);
    tr_bigint_free(n); tr_bigint_free(d);
    return q;
}
int32_t tr_persist_to_gfp(const tr_persist_record *r, tr_gfp_elem *out) {
    if (!record_valid(r) || r->head.type != TR_PERSIST_GFP || !out) return TR_MATH_INVALID;
    *out = r->elem;
    return TR_MATH_OK;
}
tr_cert *tr_persist_to_certificate(const tr_persist_record *r, const tr_math_budget *budget, tr_math_refusal *refusal,
                                   int32_t *error) {
    if (!record_valid(r) || r->head.type != TR_PERSIST_CERTIFICATE) { tr_budget_none(refusal); set_error(error, TR_MATH_INVALID); return NULL; }
    uint64_t used = 0;
    tr_cert *c = tr_cert_deserialize(r->cert, r->cert_bytes, &used, budget, refusal, error);
    if (c && used != r->cert_bytes) { tr_cert_free(c); set_error(error, TR_MATH_INVALID); return NULL; }
    return c;
}
