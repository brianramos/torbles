/* FIELD-002: parity relationship discovery, certification and substitution; see math_parity.h.
 * SPDX-License-Identifier: Apache-2.0 */
#include "math_parity.h"
#include "mathematics.h"
#include "math_domain.h"
#include "math_handles.h"
#include <stdlib.h>
#include <string.h>

TR_MATH_ABI(tr_parity_spec, TR_PARITY_ABI);   /* the ctypes boundary of FIELD-002 (math_handles.h) */
TR_MATH_ABI(tr_parity_relation, TR_PARITY_ABI);
TR_MATH_ABI(tr_parity_found_info, TR_PARITY_ABI);
TR_MATH_ABI(tr_parity_rewrite_info, TR_PARITY_ABI);
TR_MATH_ABI(tr_parity_check, TR_PARITY_ABI);

_Static_assert(sizeof(tr_parity_spec) == 16, "tr_parity_spec ABI");
_Static_assert(sizeof(tr_parity_relation) == 128, "tr_parity_relation ABI");
_Static_assert(sizeof(tr_parity_found_info) == 96, "tr_parity_found_info ABI");
_Static_assert(sizeof(tr_parity_rewrite_info) == 112, "tr_parity_rewrite_info ABI");
_Static_assert(sizeof(tr_parity_check) == 80, "tr_parity_check ABI");

#define FOUND_MAGIC UINT32_C(0x50464e44)
#define REWRITE_MAGIC UINT32_C(0x50525752)

struct tr_parity_found_s {
    uint32_t magic;
    int32_t nvars, nrel, nrej, ncert, policy, affine, backend;
    tr_field_discovery stats;
    tr_field_bitspec *specs;
    tr_parity_relation *rels;
    tr_field_relation *rej;
    tr_field_evidence *rej_ev;
    tr_cert **certs;
    uint64_t bytes;
};

struct tr_parity_rewrite_s {
    uint32_t magic;
    int32_t k;
    tr_parity_rewrite_info info;
    tr_bool_condition *conds;
    tr_bool_program *original, *rewritten;
    uint64_t bytes;
};

static void set_error(int32_t *error, int32_t code) { if (error) *error = code; }
static void *zalloc(uint64_t bytes) {
    if (bytes > (uint64_t)SIZE_MAX) return NULL;
    return calloc(1, bytes ? (size_t)bytes : 1);
}
static void zero_need(tr_math_need *n) { for (int i = 0; i < TR_BUDGET_RESOURCES; i++) n->need[i] = 0; }
static const tr_math_budget *budget_or(const tr_math_budget *b, tr_math_budget *dflt) {
    if (b) return b;
    tr_budget_default(dflt);
    return dflt;
}
static int popc(uint64_t x) { return __builtin_popcountll(x); }

/* ------------------------------------------------------------------ claims */
int32_t tr_parity_claim(const tr_field_bitspec *specs, int32_t nvars, const tr_field_relation *r, int32_t policy,
                        tr_cert_claim *claim, tr_cert_bit *bits, int32_t *status, int32_t *ncols) {
    if (nvars < 0 || nvars > TR_FIELD_MAX_VARS || (nvars && !specs) || !r || !claim || !bits || !status ||
        (policy != TR_CERT_NULL_FORBID && policy != TR_CERT_NULL_SKIP) || r->reserved ||
        (r->constant != 0 && r->constant != 1) || (nvars < 64 && (r->vars >> nvars))) return TR_MATH_INVALID;
    memset(claim, 0, sizeof *claim);
    for (int i = 0; i < TR_CERT_MAX_COLS; i++) claim->col[i] = -1;
    int32_t cols[64], n = 0;
    for (int32_t j = 0; j < nvars; j++) {                    /* the distinct scope columns, increasing */
        if (!((r->vars >> j) & 1)) continue;
        int seen = 0;
        for (int32_t i = 0; i < n; i++) seen |= cols[i] == specs[j].col;
        if (!seen) cols[n++] = specs[j].col;
    }
    for (int32_t i = 1; i < n; i++)
        for (int32_t j = i; j > 0 && cols[j - 1] > cols[j]; j--) { int32_t t = cols[j]; cols[j] = cols[j - 1]; cols[j - 1] = t; }
    if (ncols) *ncols = n;
    if (!r->vars) { *status = TR_PARITY_EMPTY; return TR_MATH_OK; }
    if (n > TR_CERT_MAX_COLS) { *status = TR_PARITY_TOO_WIDE; return TR_MATH_OK; }
    int32_t nb = 0;
    for (int32_t j = 0; j < nvars; j++) {
        if (!((r->vars >> j) & 1)) continue;
        int32_t at = 0;
        while (cols[at] != specs[j].col) at++;
        bits[nb].col = at; bits[nb].kind = specs[j].kind; bits[nb].bit = specs[j].bit; bits[nb].reserved = 0;
        bits[nb].base = specs[j].base;
        nb++;
    }
    claim->kind = TR_CLAIM_PARITY; claim->policy = policy; claim->ncols = n; claim->nbits = nb;
    for (int32_t i = 0; i < n; i++) claim->col[i] = cols[i];
    claim->carrier = TR_CARRIER_GFP; claim->field = 2; claim->constant = r->constant;
    claim->mask = nb >= 64 ? ~(uint64_t)0 : (((uint64_t)1 << nb) - 1);
    claim->bits = bits;
    *status = 0;
    return TR_MATH_OK;
}

/* ------------------------------------------------------------------ discovery */
static int found_ok(const tr_parity_found *f) { return f && f->magic == FOUND_MAGIC; }
static void found_release(tr_parity_found *f) {
    if (!f) return;
    if (f->certs) for (int32_t i = 0; i < f->nrel; i++) tr_cert_free(f->certs[i]);
    free(f->certs); free(f->specs); free(f->rels); free(f->rej); free(f->rej_ev);
    f->magic = 0;
    free(f);
}
void tr_parity_found_free(tr_parity_found *f) { if (found_ok(f)) found_release(f); }

static int32_t spec_check(const tr_parity_spec *s) {
    if (!s || (s->affine != 0 && s->affine != 1) || (s->policy != TR_CERT_NULL_FORBID && s->policy != TR_CERT_NULL_SKIP))
        return TR_MATH_INVALID;
    return TR_MATH_OK;
}
/* the found object of a discovery with `columns` columns */
static uint64_t found_bytes(int32_t nvars, int32_t columns) {
    uint64_t fc = tr_field_found_capacity(columns), rc = tr_field_rejected_capacity(columns);
    uint64_t b = tr_budget_add(sizeof(tr_parity_found), tr_budget_mul((uint64_t)nvars, sizeof(tr_field_bitspec)));
    b = tr_budget_add(b, tr_budget_mul(fc, sizeof(tr_parity_relation) + sizeof(tr_cert *)));
    return tr_budget_add(b, tr_budget_mul(rc, sizeof(tr_field_relation) + sizeof(tr_field_evidence)));
}
int32_t tr_parity_discover_need(const tr_scope *scope, const tr_field_bitspec *specs, int32_t nvars,
                                const tr_parity_spec *spec, tr_math_need *need) {
    if (!need || spec_check(spec)) return TR_MATH_INVALID;
    tr_math_need bn;
    int32_t rc = tr_field_bits_need(scope, specs, nvars, &bn);   /* also validates the specs against the scope */
    if (rc) return rc;
    tr_scope_view v;
    tr_scope_get_view(scope, &v);
    const int32_t columns = nvars + spec->affine;
    zero_need(need);
    need->need[TR_BUDGET_VARIABLES] = (uint64_t)columns;
    need->need[TR_BUDGET_SUPPORT] = spec->sample ? spec->sample : v.selected;
    need->need[TR_BUDGET_ITERATIONS] = (uint64_t)columns + 1;
    need->need[TR_BUDGET_OUTPUT] = tr_field_found_capacity(columns) + tr_field_rejected_capacity(columns);
    uint64_t disc = tr_budget_mul((uint64_t)(columns > 0 ? columns : 1), sizeof(tr_field_relation) + sizeof(tr_field_evidence));
    need->need[TR_BUDGET_BYTES] = tr_budget_add(tr_budget_add(bn.need[TR_BUDGET_BYTES], disc), found_bytes(nvars, columns));
    return TR_MATH_OK;
}
/* the object around a discovery's results (copied); certs / statuses / claim evidence per found relation (owned
 * certificates move into the object, also on failure) */
static tr_parity_found *found_build(const tr_field_bitspec *specs, int32_t nvars, const tr_parity_spec *spec,
                                    const tr_field_discovery *stats, const tr_field_relation *found,
                                    const tr_field_evidence *fev, int32_t nfound, const tr_field_relation *rej,
                                    const tr_field_evidence *rev, int32_t nrej, tr_cert **certs,
                                    const int32_t *status, const tr_cert_evidence *cev, const int32_t *ncols,
                                    int32_t backend, int32_t *error) {
    tr_parity_found *f = zalloc(sizeof *f);
    if (!f) {
        for (int32_t i = 0; i < nfound; i++) tr_cert_free(certs[i]);
        set_error(error, TR_MATH_MEMORY); return NULL;
    }
    f->nrel = nfound;
    f->certs = zalloc((uint64_t)nfound * sizeof(tr_cert *));
    if (f->certs) for (int32_t i = 0; i < nfound; i++) { f->certs[i] = certs[i]; certs[i] = NULL; }
    else for (int32_t i = 0; i < nfound; i++) tr_cert_free(certs[i]);
    f->specs = zalloc((uint64_t)nvars * sizeof(tr_field_bitspec));
    f->rels = zalloc((uint64_t)nfound * sizeof(tr_parity_relation));
    f->rej = zalloc((uint64_t)nrej * sizeof(tr_field_relation));
    f->rej_ev = zalloc((uint64_t)nrej * sizeof(tr_field_evidence));
    if (!f->certs || !f->specs || !f->rels || !f->rej || !f->rej_ev) {
        f->magic = FOUND_MAGIC; found_release(f); set_error(error, TR_MATH_MEMORY); return NULL;
    }
    f->nvars = nvars; f->nrej = nrej; f->policy = spec->policy; f->affine = spec->affine; f->backend = backend;
    f->stats = *stats;
    if (nvars) memcpy(f->specs, specs, (size_t)nvars * sizeof(tr_field_bitspec));
    for (int32_t i = 0; i < nfound; i++) {
        tr_parity_relation *R = &f->rels[i];
        R->relation = found[i]; R->evidence = fev[i]; R->status = status[i]; R->ncols = ncols[i];
        if (status[i] == TR_PARITY_CERTIFIED || status[i] == TR_PARITY_VIOLATED) R->certified = cev[i];
        else R->certified.witness = TR_CERT_NONE;
        f->ncert += status[i] == TR_PARITY_CERTIFIED;
    }
    if (nrej) { memcpy(f->rej, rej, (size_t)nrej * sizeof *rej); memcpy(f->rej_ev, rev, (size_t)nrej * sizeof *rev); }
    f->bytes = found_bytes(nvars, nvars + spec->affine);
    for (int32_t i = 0; i < nfound; i++) {
        tr_cert_info ci;
        if (f->certs[i] && !tr_cert_get_info(f->certs[i], &ci)) f->bytes = tr_budget_add(f->bytes, ci.bytes);
    }
    f->magic = FOUND_MAGIC;
    set_error(error, TR_MATH_OK);
    return f;
}
/* the per-relation work of both paths: the claim, then a snapshot (evidence NULL) or an adoption of device evidence */
static int32_t certify_one(const tr_scope *scope, const tr_field_bitspec *specs, int32_t nvars, int32_t policy,
                           const tr_field_relation *r, const tr_cert_evidence *device, const tr_math_budget *budget,
                           tr_math_refusal *refusal, tr_cert **cert, int32_t *status, tr_cert_evidence *ev,
                           int32_t *ncols) {
    tr_cert_claim claim;
    tr_cert_bit bits[64];
    int32_t st = 0, rc = tr_parity_claim(specs, nvars, r, policy, &claim, bits, &st, ncols);
    *cert = NULL;
    memset(ev, 0, sizeof *ev);
    ev->witness = TR_CERT_NONE;
    if (rc) return rc;
    if (st) { *status = st; return TR_MATH_OK; }
    int32_t err = 0;
    if (!device) {
        *cert = tr_cert_snapshot(scope, &claim, NULL, 0, ev, budget, refusal, &err);
    } else {
        *ev = *device;
        if (device->holds) *cert = tr_cert_adopt(scope, &claim, device, TR_CERT_BY_DEVICE, NULL, 0, budget, refusal, &err);
        else {
            tr_scope_info si;
            tr_scope_get_info(scope, &si);
            if (device->rows != si.rows || device->selected != si.selected || !device->violations ||
                device->witness >= si.rows || device->unit != TR_CERT_UNIT_ROWS) return TR_MATH_INVALID;
            err = TR_MATH_DOMAIN_ERROR;
        }
    }
    if (*cert) { *status = TR_PARITY_CERTIFIED; return TR_MATH_OK; }
    if (err == TR_MATH_DOMAIN_ERROR) { *status = TR_PARITY_VIOLATED; return TR_MATH_OK; }
    return err ? err : TR_MATH_MEMORY;
}
static tr_parity_found *discover_common(const tr_scope *scope, const tr_field_bitspec *specs, int32_t nvars,
                                        const tr_parity_spec *spec, const tr_field_discovery *stats,
                                        const tr_field_relation *found, const tr_field_evidence *fev, int32_t nfound,
                                        const tr_field_relation *rej, const tr_field_evidence *rev, int32_t nrej,
                                        const tr_cert_evidence *device, const tr_math_budget *budget,
                                        tr_math_refusal *refusal, int32_t *error) {
    int32_t n = nfound > 0 ? nfound : 0;
    tr_cert **certs = zalloc((uint64_t)n * sizeof(tr_cert *));
    int32_t *status = zalloc((uint64_t)n * 4), *ncols = zalloc((uint64_t)n * 4);
    tr_cert_evidence *cev = zalloc((uint64_t)n * sizeof(tr_cert_evidence));
    if (!certs || !status || !ncols || !cev) {
        free(certs); free(status); free(ncols); free(cev); set_error(error, TR_MATH_MEMORY); return NULL;
    }
    int32_t rc = TR_MATH_OK;
    for (int32_t i = 0; i < n && !rc; i++)
        rc = certify_one(scope, specs, nvars, spec->policy, &found[i], device ? &device[i] : NULL, budget, refusal,
                         &certs[i], &status[i], &cev[i], &ncols[i]);
    tr_parity_found *f = NULL;
    if (!rc) {
        tr_budget_none(refusal);
        f = found_build(specs, nvars, spec, stats, found, fev, n, rej, rev, nrej, certs, status, cev, ncols,
                        device ? TR_CERT_BY_DEVICE : TR_CERT_BY_CPU, error);
    } else {
        for (int32_t i = 0; i < n; i++) tr_cert_free(certs[i]);
        set_error(error, rc);
    }
    free(certs); free(status); free(ncols); free(cev);
    return f;
}
tr_parity_found *tr_parity_discover(const tr_scope *scope, const tr_field_bitspec *specs, int32_t nvars,
                                    const tr_parity_spec *spec, const tr_math_budget *budget, tr_math_refusal *refusal,
                                    int32_t *error) {
    tr_budget_none(refusal);
    tr_math_need need;
    int32_t rc = tr_parity_discover_need(scope, specs, nvars, spec, &need);
    if (rc) { set_error(error, rc); return NULL; }
    tr_math_budget dflt;
    const tr_math_budget *b = budget_or(budget, &dflt);
    rc = tr_budget_check(b, &need, refusal);
    if (rc) { set_error(error, rc); return NULL; }
    const int32_t columns = nvars + spec->affine;
    if (columns > 64) { set_error(error, TR_MATH_UNSUPPORTED); return NULL; }
    tr_field_bits *bits = tr_field_bits_build(scope, specs, nvars, b, refusal, &rc);
    if (!bits) { set_error(error, rc ? rc : TR_MATH_MEMORY); return NULL; }
    tr_field_bits_view v;
    tr_field_bits_get_view(bits, &v);
    const int32_t fcap = (int32_t)tr_field_found_capacity(columns), rcap = (int32_t)tr_field_rejected_capacity(columns);
    tr_field_relation *found = zalloc((uint64_t)(fcap > 0 ? fcap : 1) * sizeof *found);
    tr_field_evidence *fev = zalloc((uint64_t)(fcap > 0 ? fcap : 1) * sizeof *fev);
    tr_field_relation *rej = zalloc((uint64_t)(rcap > 0 ? rcap : 1) * sizeof *rej);
    tr_field_evidence *rev = zalloc((uint64_t)(rcap > 0 ? rcap : 1) * sizeof *rev);
    tr_parity_found *f = NULL;
    if (!found || !fev || !rej || !rev) rc = TR_MATH_MEMORY;
    tr_field_discovery stats;
    memset(&stats, 0, sizeof stats);
    tr_field_discover_spec fs;
    fs.sample = spec->sample; fs.affine = spec->affine; fs.reserved = 0;
    if (!rc) rc = tr_field_discover(&v, &fs, b, found, fev, fcap, rej, rev, rcap, &stats, refusal);
    if (!rc) f = discover_common(scope, specs, nvars, spec, &stats, found, fev, stats.nrelations, rej, rev,
                                 stats.nrejected, NULL, b, refusal, &rc);
    tr_field_bits_free(bits);
    free(found); free(fev); free(rej); free(rev);
    set_error(error, f ? TR_MATH_OK : (rc ? rc : TR_MATH_MEMORY));
    return f;
}
tr_parity_found *tr_parity_adopt(const tr_scope *scope, const tr_field_bitspec *specs, int32_t nvars,
                                 const tr_parity_spec *spec, const tr_field_discovery *stats,
                                 const tr_field_relation *found, const tr_field_evidence *fev, int32_t nfound,
                                 const tr_field_relation *rej, const tr_field_evidence *rev, int32_t nrej,
                                 const tr_cert_evidence *claim_ev, const tr_math_budget *budget,
                                 tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    tr_math_need need;
    int32_t rc = tr_parity_discover_need(scope, specs, nvars, spec, &need);
    if (rc) { set_error(error, rc); return NULL; }
    const int32_t columns = nvars + spec->affine;
    tr_scope_info si;
    tr_scope_get_info(scope, &si);
    int bad = !stats || stats->columns != columns || stats->nrelations != nfound || stats->nrejected != nrej ||
              nfound < 0 || nrej < 0 || (uint64_t)nfound > tr_field_found_capacity(columns) ||
              (uint64_t)nrej > tr_field_rejected_capacity(columns) || (nfound && (!found || !fev || !claim_ev)) ||
              (nrej && (!rej || !rev));
    for (int32_t i = 0; i < nfound && !bad; i++)           /* found relations hold on every complete row */
        bad = found[i].reserved || (found[i].constant != 0 && found[i].constant != 1) ||
              (nvars < 64 && (found[i].vars >> nvars)) || fev[i].violations || fev[i].witness != TR_FIELD_NO_ROW;
    for (int32_t i = 0; i < nrej && !bad; i++)             /* rejected proposals failed on a row of the scope */
        bad = !rev[i].violations || rev[i].witness >= si.rows || (nvars < 64 && (rej[i].vars >> nvars));
    if (bad) { set_error(error, TR_MATH_INVALID); return NULL; }
    tr_math_budget dflt;
    const tr_math_budget *b = budget_or(budget, &dflt);
    need.need[TR_BUDGET_SUPPORT] = 0;                          /* nothing is sampled here */
    need.need[TR_BUDGET_ITERATIONS] = 0;
    rc = tr_budget_check(b, &need, refusal);
    if (rc) { set_error(error, rc); return NULL; }
    if (columns > 64) { set_error(error, TR_MATH_UNSUPPORTED); return NULL; }
    return discover_common(scope, specs, nvars, spec, stats, found, fev, nfound, rej, rev, nrej, claim_ev, b, refusal,
                           error);
}
int32_t tr_parity_found_get_info(const tr_parity_found *f, tr_parity_found_info *out) {
    if (!found_ok(f) || !out) return TR_MATH_INVALID;
    memset(out, 0, sizeof *out);
    out->stats = f->stats; out->nrelations = f->nrel; out->nrejected = f->nrej; out->ncertified = f->ncert;
    out->policy = f->policy; out->nvars = f->nvars; out->affine = f->affine; out->backend = f->backend;
    return TR_MATH_OK;
}
int32_t tr_parity_found_relation(const tr_parity_found *f, int32_t i, tr_parity_relation *out) {
    if (!found_ok(f) || !out || i < 0 || i >= f->nrel) return TR_MATH_INVALID;
    *out = f->rels[i];
    return TR_MATH_OK;
}
int32_t tr_parity_found_rejected(const tr_parity_found *f, int32_t i, tr_field_relation *r, tr_field_evidence *e) {
    if (!found_ok(f) || !r || !e || i < 0 || i >= f->nrej) return TR_MATH_INVALID;
    *r = f->rej[i]; *e = f->rej_ev[i];
    return TR_MATH_OK;
}
const tr_cert *tr_parity_found_certificate(const tr_parity_found *f, int32_t i) {
    if (!found_ok(f) || i < 0 || i >= f->nrel) return NULL;
    return f->certs[i];
}

/* ------------------------------------------------------------------ substitution */
static int rewrite_ok(const tr_parity_rewrite *r) { return r && r->magic == REWRITE_MAGIC; }
static void rewrite_release(tr_parity_rewrite *r) {
    if (!r) return;
    tr_bool_program_free(r->original); tr_bool_program_free(r->rewritten);
    free(r->conds);
    r->magic = 0;
    free(r);
}
void tr_parity_rewrite_free(tr_parity_rewrite *r) { if (rewrite_ok(r)) rewrite_release(r); }

static int cond_equal(const tr_bool_condition *a, const tr_bool_condition *b) {
    return a->op == b->op && a->col == b->col && a->col2 == b->col2 && a->reserved == b->reserved &&
           a->num[0] == b->num[0] && a->num[1] == b->num[1] && a->den[0] == b->den[0] && a->den[1] == b->den[1];
}
/* the condition of relation bit j of a parity claim: col <> 0 (NONZERO) or bit b of (col - base) (PLANE) */
static tr_bool_condition bit_condition(const tr_cert_claim *c, int32_t j) {
    tr_bool_condition x;
    memset(&x, 0, sizeof x);
    const tr_cert_bit *b = &c->bits[j];
    x.col = c->col[b->col]; x.col2 = -1; x.den[0] = x.den[1] = 1;
    if (b->kind == TR_CERT_BIT_NONZERO) x.op = TR_BOOL_NE;
    else { x.op = TR_BOOL_BIT; x.num[0] = b->base; x.num[1] = b->bit; }
    return x;
}
/* the columns and planes the conditions in mask scan (a scanned column subsumes its planes) */
static int32_t scan_units(uint64_t mask, const tr_bool_condition *conds) {
    int32_t whole[128], nwhole = 0, nplanes = 0;
    const tr_bool_condition *planes[64];
    for (int32_t i = 0; i < 64; i++) {
        if (!((mask >> i) & 1)) continue;
        const tr_bool_condition *c = &conds[i];
        if (c->op == TR_BOOL_BIT) { planes[nplanes++] = c; continue; }
        int32_t cs[2] = {c->col, c->op >= TR_BOOL_COL_EQ ? c->col2 : -1};
        for (int t = 0; t < 2; t++) {
            if (cs[t] < 0) continue;
            int seen = 0;
            for (int32_t w = 0; w < nwhole; w++) seen |= whole[w] == cs[t];
            if (!seen) whole[nwhole++] = cs[t];
        }
    }
    int32_t n = nwhole;
    for (int32_t p = 0; p < nplanes; p++) {
        int subsumed = 0, repeat = 0;
        for (int32_t w = 0; w < nwhole; w++) subsumed |= whole[w] == planes[p]->col;
        for (int32_t q = 0; q < p; q++)
            repeat |= planes[q]->col == planes[p]->col && planes[q]->num[0] == planes[p]->num[0] &&
                      planes[q]->num[1] == planes[p]->num[1];
        n += !subsumed && !repeat;
    }
    return n;
}
/* the conditions the program reads from its root when the chosen pure forms are replaced by L + S */
static uint64_t reads(const tr_bool_program_view *p, const uint64_t *L, const uint8_t *chosen, uint64_t S,
                      int32_t *stamp, int32_t mark, uint32_t *stack) {
    uint64_t mask = 0;
    uint32_t sp = 0;
    stack[sp++] = (uint32_t)p->root; stamp[p->root] = mark;
    while (sp) {
        uint32_t i = stack[--sp];
        const tr_bool_node *n = &p->nodes[i];
        if (chosen[i]) { mask |= L[i] ^ S; continue; }
        if (n->op == TR_BOOL_VAR) { mask |= (uint64_t)1 << n->a; continue; }
        int32_t ops[2] = {n->op >= TR_BOOL_NOT ? n->a : -1, n->op >= TR_BOOL_AND ? n->b : -1};
        for (int t = 0; t < 2; t++)
            if (ops[t] >= 0 && stamp[ops[t]] != mark) { stamp[ops[t]] = mark; stack[sp++] = (uint32_t)ops[t]; }
    }
    return mask;
}
/* the rewritten nodes (out NULL: only counted); map: the node of each original node; varnode: kk entries */
static int32_t emit(const tr_bool_program_view *p, const uint8_t *chosen, const uint64_t *L, const uint8_t *Lc,
                    uint64_t S, int cbit, int32_t kk, int32_t *map, int32_t *varnode, tr_bool_node *out) {
    int32_t n = 0;
    for (int32_t v = 0; v < kk; v++) varnode[v] = -1;
#define EMIT(o, x, y) do { if (out) { out[n].op = (o); out[n].a = (x); out[n].b = (y); out[n].reserved = 0; } n++; } while (0)
    for (int32_t i = 0; i < p->nnodes; i++) {
        const tr_bool_node *nd = &p->nodes[i];
        if (chosen[i]) {
            uint64_t vars = L[i] ^ S;
            int c = Lc[i] ^ cbit;
            int32_t x = -1;
            for (int32_t v = 0; v < kk; v++) {
                if (!((vars >> v) & 1)) continue;
                int32_t vn = varnode[v];
                if (vn < 0) { vn = n; EMIT(TR_BOOL_VAR, v, -1); varnode[v] = vn; }
                if (x < 0) x = vn;
                else { int32_t at = n; EMIT(TR_BOOL_XOR, x, vn); x = at; }
            }
            if (x < 0) { x = n; EMIT(TR_BOOL_CONST, c ? TR_BOOL_TRUE : TR_BOOL_FALSE, -1); }
            else if (c) { int32_t at = n; EMIT(TR_BOOL_NOT, x, -1); x = at; }
            map[i] = x;
            continue;
        }
        int32_t a = nd->a, b = nd->b;
        if (nd->op >= TR_BOOL_NOT) a = map[nd->a];
        if (nd->op >= TR_BOOL_AND) b = map[nd->b];
        if (nd->op == TR_BOOL_VAR && varnode[nd->a] < 0) varnode[nd->a] = n;
        map[i] = n;
        EMIT(nd->op, a, b);
    }
#undef EMIT
    return n;
}

tr_parity_rewrite *tr_parity_substitute(const tr_scope *current, const tr_bool_condition *conds, int32_t k,
                                        const tr_bool_program *prog, const tr_cert *cert, int32_t mode,
                                        const tr_math_budget *budget, tr_math_refusal *refusal,
                                        tr_parity_rewrite_info *info, int32_t *error) {
    tr_budget_none(refusal);
    tr_parity_rewrite_info inf;
    memset(&inf, 0, sizeof inf);
    inf.mode = mode; inf.k_original = k;
    inf.decision.premise = -1; inf.decision.stale.column = inf.decision.stale.dim = -1;
    tr_scope_view sv;
    tr_bool_program_view pv;
    tr_cert_claim claim;
    int32_t rc = TR_MATH_OK;
    uint64_t *L = NULL;
    uint8_t *Lc = NULL, *pure = NULL, *maximal = NULL, *chosen = NULL;
    int32_t *map = NULL, *stamp = NULL;
    uint32_t *stack = NULL;
    tr_bool_node *out = NULL;
    tr_parity_rewrite *r = NULL;
    tr_bool_condition all[64];
#define FAIL(code) do { rc = (code); goto done; } while (0)
    if (!current || !prog || !cert || (mode != TR_PARITY_AUTO && mode != TR_PARITY_ALL) || k < 0 || (k && !conds) ||
        tr_scope_get_view(current, &sv) || tr_bool_program_get_view(prog, &pv) || tr_cert_get_claim(cert, &claim))
        FAIL(TR_MATH_INVALID);
    if (pv.k != k) FAIL(TR_MATH_INVALID);                  /* the program reads another number of conditions */
    if (k > 64) FAIL(TR_MATH_UNSUPPORTED);
    for (int32_t i = 0; i < k; i++) if (tr_bool_condition_check(&sv, &conds[i])) FAIL(TR_MATH_INVALID);
    if (claim.kind != TR_CLAIM_PARITY) { inf.reason = TR_PARITY_REFUSE_CLAIM; FAIL(TR_MATH_DOMAIN_ERROR); }
    rc = tr_cert_authorize_exact(cert, 2, current, NULL, 0, &inf.decision);
    if (rc) { inf.reason = TR_PARITY_REFUSE_CERTIFICATE; goto done; }
    if (inf.decision.policy != TR_CERT_NULL_FORBID) {
        tr_cert_info ci;
        tr_cert_get_info(cert, &ci);
        if (ci.evidence.excluded) { inf.reason = TR_PARITY_REFUSE_NULLS; FAIL(TR_MATH_DOMAIN_ERROR); }
    }
    /* the relation over the conditions: S (GF(2): a repeated bit cancels) and the constant */
    int32_t kk = k;
    if (k) memcpy(all, conds, (size_t)k * sizeof *all);
    uint64_t S = 0;
    for (int32_t j = 0; j < claim.nbits; j++) {
        if (!((claim.mask >> j) & 1)) continue;
        tr_bool_condition x = bit_condition(&claim, j);
        if (x.col < 0 || x.col >= sv.ncols || tr_bool_condition_check(&sv, &x)) FAIL(TR_MATH_INVALID);
        int32_t at = -1;
        for (int32_t i = 0; i < kk && at < 0; i++) if (cond_equal(&all[i], &x)) at = i;
        if (at < 0) {
            if (kk == 64) FAIL(TR_MATH_UNSUPPORTED);
            all[kk] = x; at = kk++;
        }
        S ^= (uint64_t)1 << at;
    }
    const int cbit = claim.constant;
    inf.k = kk; inf.appended = kk - k;
    const int32_t nn = pv.nnodes;
    tr_math_need need;
    zero_need(&need);
    need.need[TR_BUDGET_VARIABLES] = (uint64_t)kk;
    need.need[TR_BUDGET_BYTES] = tr_budget_mul((uint64_t)nn, 8 + 1 + 1 + 1 + 1 + 4 + 4 + 4);
    tr_math_budget dflt;
    const tr_math_budget *b = budget_or(budget, &dflt);
    rc = tr_budget_check(b, &need, refusal);
    if (rc) goto done;
    L = zalloc((uint64_t)nn * 8); Lc = zalloc((uint64_t)nn); pure = zalloc((uint64_t)nn); maximal = zalloc((uint64_t)nn);
    chosen = zalloc((uint64_t)nn); map = zalloc((uint64_t)nn * 4); stamp = zalloc((uint64_t)nn * 4);
    stack = zalloc((uint64_t)nn * 4);
    if (!L || !Lc || !pure || !maximal || !chosen || !map || !stamp || !stack) FAIL(TR_MATH_MEMORY);
    /* pure forms, maximal ones, candidates */
    for (int32_t s = 0; s < pv.nsched; s++) {
        uint32_t i = pv.schedule[s];
        const tr_bool_node *n = &pv.nodes[i];
        switch (n->op) {
        case TR_BOOL_CONST:
            if (n->a != TR_BOOL_UNKNOWN) { pure[i] = 1; Lc[i] = (uint8_t)(n->a == TR_BOOL_TRUE); }
            break;
        case TR_BOOL_VAR: pure[i] = 1; L[i] = (uint64_t)1 << n->a; break;
        case TR_BOOL_NOT: if (pure[n->a]) { pure[i] = 1; L[i] = L[n->a]; Lc[i] = (uint8_t)(Lc[n->a] ^ 1); } break;
        case TR_BOOL_XOR:
            if (pure[n->a] && pure[n->b] && !(L[n->a] & L[n->b])) {
                pure[i] = 1; L[i] = L[n->a] | L[n->b]; Lc[i] = (uint8_t)(Lc[n->a] ^ Lc[n->b]);
            }
            break;
        default: break;
        }
    }
    maximal[pv.root] = pure[pv.root];
    for (int32_t s = 0; s < pv.nsched; s++) {
        uint32_t i = pv.schedule[s];
        const tr_bool_node *n = &pv.nodes[i];
        if (pure[i] || n->op == TR_BOOL_CONST || n->op == TR_BOOL_VAR) continue;
        if (pure[n->a]) maximal[n->a] = 1;
        if (n->op >= TR_BOOL_AND && pure[n->b]) maximal[n->b] = 1;
    }
    int32_t mark = 0;
    int32_t base = scan_units(reads(&pv, L, chosen, S, stamp, ++mark, stack), all), cur = base, forms = 0;
    for (int32_t s = 0; s < pv.nsched; s++) {
        uint32_t i = pv.schedule[s];
        if (!maximal[i] || !(L[i] & S)) continue;
        inf.candidates++;
        chosen[i] = 1;
        if (mode == TR_PARITY_ALL) { forms++; continue; }
        int32_t c = scan_units(reads(&pv, L, chosen, S, stamp, ++mark, stack), all);
        if (c < cur) { cur = c; forms++; } else chosen[i] = 0;
    }
    if (mode == TR_PARITY_ALL) cur = scan_units(reads(&pv, L, chosen, S, stamp, ++mark, stack), all);
    inf.cost_original = base; inf.cost_rewritten = cur; inf.forms = forms;
    inf.chosen = forms ? TR_PARITY_REWRITTEN : TR_PARITY_KEPT;
    /* the rewritten program: counted, budgeted, written */
    int32_t varnode[64];
    int32_t total = emit(&pv, chosen, L, Lc, S, cbit, kk, map, varnode, NULL);
    need.need[TR_BUDGET_CIRCUIT] = (uint64_t)total;
    need.need[TR_BUDGET_BYTES] = tr_budget_add(need.need[TR_BUDGET_BYTES],
                                               tr_budget_add(tr_budget_mul((uint64_t)total, 16), 48 * 64));
    rc = tr_budget_check(b, &need, refusal);
    if (rc) goto done;
    out = zalloc((uint64_t)total * sizeof *out);
    r = zalloc(sizeof *r);
    if (!out || !r) FAIL(TR_MATH_MEMORY);
    emit(&pv, chosen, L, Lc, S, cbit, kk, map, varnode, out);
    inf.nnodes = total; inf.root = map[pv.root];
    r->conds = zalloc((uint64_t)kk * sizeof(tr_bool_condition));
    if (!r->conds) FAIL(TR_MATH_MEMORY);
    if (kk) memcpy(r->conds, all, (size_t)kk * sizeof *all);
    r->original = tr_bool_program_create(kk, pv.nodes, nn, pv.root, b, refusal, &rc);
    if (!r->original) goto done;
    r->rewritten = tr_bool_program_create(kk, out, total, inf.root, b, refusal, &rc);
    if (!r->rewritten) goto done;
    r->k = kk;
    r->info = inf;
    r->bytes = tr_budget_add(sizeof *r, tr_budget_mul((uint64_t)kk, sizeof(tr_bool_condition)));
    r->magic = REWRITE_MAGIC;
    rc = TR_MATH_OK;
done:
#undef FAIL
    free(L); free(Lc); free(pure); free(maximal); free(chosen); free(map); free(stamp); free(stack); free(out);
    if (rc) { rewrite_release(r); r = NULL; }
    if (info) *info = inf;
    set_error(error, rc);
    return r;
}
int32_t tr_parity_rewrite_get_info(const tr_parity_rewrite *r, tr_parity_rewrite_info *out) {
    if (!rewrite_ok(r) || !out) return TR_MATH_INVALID;
    *out = r->info;
    return TR_MATH_OK;
}
int32_t tr_parity_rewrite_conditions(const tr_parity_rewrite *r, tr_bool_condition *out, int32_t cap) {
    if (!rewrite_ok(r) || cap < r->k || (r->k && !out)) return TR_MATH_INVALID;
    if (r->k) memcpy(out, r->conds, (size_t)r->k * sizeof *out);
    return TR_MATH_OK;
}
const tr_bool_program *tr_parity_rewrite_program(const tr_parity_rewrite *r, int32_t which) {
    if (!rewrite_ok(r)) return NULL;
    return which == 0 ? r->original : which == 1 ? r->rewritten : NULL;
}

/* ------------------------------------------------------------------ verification */
static int32_t verify_check(const tr_bool_domain *d, const tr_bool_program_view *a, const tr_bool_program_view *b) {
    int32_t rc = tr_bool_domain_check(d);
    if (!rc) rc = tr_bool_program_check(a);
    if (!rc) rc = tr_bool_program_check(b);
    if (rc) return rc;
    return a->k != d->k || b->k != d->k ? TR_MATH_DOMAIN_ERROR : TR_MATH_OK;
}
int32_t tr_parity_verify_need(const tr_bool_domain *d, const tr_bool_program_view *a, const tr_bool_program_view *b,
                              int32_t target, tr_math_need *need) {
    if (!need) return TR_MATH_INVALID;
    int32_t rc = verify_check(d, a, b);
    if (rc) return rc;
    zero_need(need);
    need->need[TR_BUDGET_VARIABLES] = (uint64_t)d->k;
    need->need[TR_BUDGET_CIRCUIT] = tr_budget_add((uint64_t)a->nnodes, (uint64_t)b->nnodes);
    need->need[TR_BUDGET_SUPPORT] = d->kind == TR_BOOL_ASSIGNMENTS ? d->points : 0;
    if (target == TR_PARITY_NEED_DEVICE) need->need[TR_BUDGET_BYTES] = tr_parity_device_bytes(d, a, b);
    else if (target == TR_PARITY_NEED_CPU) need->need[TR_BUDGET_BYTES] = tr_budget_mul(need->need[TR_BUDGET_CIRCUIT], 16);
    else return TR_MATH_INVALID;
    return TR_MATH_OK;
}
int32_t tr_parity_verify_view(const tr_bool_domain *d, const tr_bool_program_view *a, const tr_bool_program_view *b,
                              const tr_math_budget *budget, tr_parity_check *out, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!out) return TR_MATH_INVALID;
    tr_math_need need;
    int32_t rc = tr_parity_verify_need(d, a, b, TR_PARITY_NEED_CPU, &need);
    if (rc) return rc;
    tr_math_budget dflt;
    rc = tr_budget_check(budget_or(budget, &dflt), &need, refusal);
    if (rc) return rc;
    uint64_t *va = zalloc((uint64_t)a->nnodes * 16), *vb = zalloc((uint64_t)b->nnodes * 16);
    if (!va || !vb) { free(va); free(vb); return TR_MATH_MEMORY; }
    memset(out, 0, sizeof *out);
    out->points = d->selected; out->witness = TR_PARITY_NONE;
    const uint64_t ra = (uint64_t)a->root, rb = (uint64_t)b->root;
    for (uint64_t w = 0; w < d->nw; w++) {
        uint64_t sel = tr_bool_selection_word(d, w);
        if (!sel) continue;
        tr_bool_eval_base(a, d, w, va);
        tr_bool_eval_base(b, d, w, vb);
        uint64_t xa = va[2 * ra] & sel, ka = va[2 * ra + 1] & sel, xb = vb[2 * rb] & sel, kb = vb[2 * rb + 1] & sel;
        out->a_true += (uint64_t)popc(xa); out->a_false += (uint64_t)popc(ka & ~xa); out->a_unknown += (uint64_t)popc(sel & ~ka);
        out->b_true += (uint64_t)popc(xb); out->b_false += (uint64_t)popc(kb & ~xb); out->b_unknown += (uint64_t)popc(sel & ~kb);
        uint64_t m = tr_parity_mismatch(xa, ka, xb, kb, sel);
        out->mismatches += (uint64_t)popc(m);
        if (m && out->witness == TR_PARITY_NONE) out->witness = w * 64 + (uint64_t)__builtin_ctzll(m);
    }
    out->equivalent = out->mismatches == 0;
    free(va); free(vb);
    return TR_MATH_OK;
}
int32_t tr_parity_verify(const tr_bool_masks *m, const tr_bool_program *a, const tr_bool_program *b,
                         const tr_math_budget *budget, tr_parity_check *out, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    tr_bool_domain d;
    tr_bool_program_view x, y;
    if (tr_bool_masks_get_domain(m, &d) || tr_bool_program_get_view(a, &x) || tr_bool_program_get_view(b, &y))
        return TR_MATH_INVALID;
    return tr_parity_verify_view(&d, &x, &y, budget, out, refusal);
}
