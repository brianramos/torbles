/* DISC-003: conservation and stability of invariants across revisions; see math_conserve.h.
 * SPDX-License-Identifier: Apache-2.0 */
#include "math_conserve.h"
#include "mathematics.h"
#include "math_domain.h"
#include "math_handles.h"
#include <stdlib.h>
#include <string.h>

TR_MATH_ABI(tr_cons_delta_evidence, TR_CONS_ABI);   /* the ctypes boundary of DISC-003 (math_handles.h) */
TR_MATH_ABI(tr_cons_delta_out, TR_CONS_ABI);
TR_MATH_ABI(tr_cons_job_view, TR_CONS_ABI);
TR_MATH_ABI(tr_cons_info, TR_CONS_ABI);
TR_MATH_ABI(tr_cons_record, TR_CONS_ABI);
TR_MATH_ABI(tr_cons_step, TR_CONS_ABI);
TR_MATH_ABI(tr_cons_classes, TR_CONS_ABI);
TR_MATH_ABI(tr_cons_decision, TR_CONS_ABI);
TR_MATH_ABI(tr_cons_law_info, TR_CONS_ABI);

_Static_assert(sizeof(tr_cons_delta_evidence) == 80, "tr_cons_delta_evidence ABI");
_Static_assert(sizeof(tr_cons_delta_out) == 40, "tr_cons_delta_out ABI");
_Static_assert(sizeof(tr_cons_job_view) == 2296, "tr_cons_job_view ABI");
_Static_assert(sizeof(tr_cons_info) == 136, "tr_cons_info ABI");
_Static_assert(sizeof(tr_cons_record) == 328, "tr_cons_record ABI");
_Static_assert(sizeof(tr_cons_step) == 104, "tr_cons_step ABI");
_Static_assert(sizeof(tr_cons_classes) == 56, "tr_cons_classes ABI");
_Static_assert(sizeof(tr_cons_decision) == 80, "tr_cons_decision ABI");
_Static_assert(sizeof(tr_cons_law_info) == 200, "tr_cons_law_info ABI");

#define TRACKER_MAGIC UINT32_C(0x4b525443)
#define JOB_MAGIC UINT32_C(0x424f4a43)
#define LAW_MAGIC UINT32_C(0x57414c43)
#define I128_LIMIT ((((unsigned __int128)1) << 127) - 1)
#define MAXV TR_CONS_MAX_VARS

static void set_error(int32_t *error, int32_t code) { if (error) *error = code; }
static void *zalloc(uint64_t bytes) {
    if (bytes > (uint64_t)SIZE_MAX) return NULL;
    return calloc(1, bytes ? (size_t)bytes : 1);
}
static const tr_math_budget *budget_or(const tr_math_budget *b, tr_math_budget *store) {
    if (b) return b;
    tr_budget_default(store);
    return store;
}
static void need_zero(tr_math_need *n) { for (int i = 0; i < TR_BUDGET_RESOURCES; i++) n->need[i] = 0; }
static int32_t check_need(const tr_math_budget *b, const tr_math_need *n, tr_math_refusal *r) {
    tr_math_budget d;
    return tr_budget_check(budget_or(b, &d), n, r);
}
static int32_t check_one(const tr_math_budget *b, int32_t res, uint64_t need, tr_math_refusal *r) {
    tr_math_budget d;
    return tr_budget_check_one(budget_or(b, &d), res, need, r);
}
static unsigned __int128 satmul(unsigned __int128 a, unsigned __int128 b) {
    if (!a || !b) return 0;
    return a > I128_LIMIT / b ? I128_LIMIT + 1 : a * b;
}

/* ================================================================ trackers */
typedef struct {
    int32_t status, declared, code, sticky;       /* sticky: an undecided deletion (evidence unknown until rebind) */
    int64_t checked, excluded, violations, overflow;
    uint64_t witness_ins, witness_del;
    uint64_t certified, verified, held, failed, undecided;
    uint64_t first_certified, last_certified, streak;
    uint64_t fail_step, fail_revision, fail_violations, fail_row;
    int32_t fail_source, fail_ncols;
    int64_t fail_weight;
    int32_t fail_col[MAXV];
    int64_t fail_value[MAXV];
    uint8_t fail_valid[MAXV];
    uint64_t nruns, capruns;
    uint64_t *runs;                               /* [first, last] pairs */
    tr_cert *snap;                                /* the current revision's certificate */
} istate;

struct tr_cons_tracker_s {
    uint32_t magic;
    int32_t k, nb, nm, nid, degree, ncols, bound, kc;   /* kc: the identities' variables */
    int32_t var[MAXV], cvar[MAXV];                /* plan variables; identity variables (scope columns) */
    uint8_t *bexp, *mexp, *spec_exp;              /* basis (nb x k), moments (nm x k), the spec's explicit basis */
    int32_t *bparent, *bvar, *mleft, *mright;
    tr_ident_spec spec;                           /* exponents -> spec_exp */
    __int128 *moment;                             /* nm exact sums */
    int64_t domain;                               /* Gram rows (multiplicity) */
    /* identities: the candidate arrays (over cvar) and the FND-007 claims (over their support columns) */
    uint64_t nterms;
    uint64_t *offset;                             /* [nid + 1] */
    uint32_t *support;
    int32_t *source;                              /* the set item of each identity */
    uint8_t *exps;
    int64_t *coefs;
    uint8_t *cexp;                                /* claims: nterms x (its ncols) packed */
    int64_t *ccoef;
    uint64_t *coff;
    int32_t *cn;                                  /* claim columns */
    int32_t *ccol;                                /* nid x MAXV claim columns */
    istate *st;
    uint64_t nsteps;
    tr_cons_step *steps;
    tr_delta_state *companion;
    uint64_t revision, digest[2], srows, ssel;    /* the bound scope: revision, digest, rows, selected rows */
    uint64_t bytes;
};
static int tracker_valid(const tr_cons_tracker *t) { return t && t->magic == TRACKER_MAGIC; }

static void istate_release(istate *s) {
    free(s->runs); s->runs = NULL;
    tr_cert_free(s->snap); s->snap = NULL;
}
void tr_cons_tracker_free(tr_cons_tracker *t) {
    if (!tracker_valid(t)) return;
    free(t->bexp); free(t->mexp); free(t->spec_exp); free(t->bparent); free(t->bvar); free(t->mleft); free(t->mright);
    free(t->moment); free(t->offset); free(t->support); free(t->source); free(t->exps); free(t->coefs);
    free(t->cexp); free(t->ccoef); free(t->coff); free(t->cn); free(t->ccol);
    if (t->st) for (int32_t i = 0; i < t->nid; i++) istate_release(&t->st[i]);
    free(t->st); free(t->steps);
    tr_delta_state_free(t->companion);
    t->magic = 0;
    free(t);
}

/* bytes of a tracker's own arrays (the certificates and the companion are added by the callers that know them) */
static uint64_t tracker_bytes(int32_t k, int32_t nb, int32_t nm, int32_t nid, uint64_t nterms, uint64_t runs,
                              uint64_t steps) {
    uint64_t b = sizeof(tr_cons_tracker);
    b = tr_budget_add(b, tr_budget_mul((uint64_t)nb, (uint64_t)k + 8 + (uint64_t)k));
    b = tr_budget_add(b, tr_budget_mul((uint64_t)nm, (uint64_t)k + 8 + 16));
    b = tr_budget_add(b, tr_budget_mul((uint64_t)nid + 1, 8 + 4 + 4 + 8 + 4 + 4 * MAXV + sizeof(istate)));
    b = tr_budget_add(b, tr_budget_mul(nterms, 2 * (MAXV + 8)));
    b = tr_budget_add(b, tr_budget_mul(runs, 16));
    return tr_budget_add(b, tr_budget_mul(steps, sizeof(tr_cons_step)));
}
static uint64_t certs_bytes(const tr_cons_tracker *t) {
    uint64_t b = 0;
    for (int32_t i = 0; i < t->nid; i++) {
        const istate *s = &t->st[i];
        tr_cert_info ci;
        if (s->snap && !tr_cert_get_info(s->snap, &ci)) b = tr_budget_add(b, ci.bytes);
    }
    tr_delta_info di;
    if (t->companion && !tr_delta_state_get_info(t->companion, &di)) b = tr_budget_add(b, di.bytes);
    return b;
}
static uint64_t total_runs(const tr_cons_tracker *t) {
    uint64_t r = 0;
    for (int32_t i = 0; i < t->nid; i++) r += t->st[i].nruns;
    return r;
}
static void retally(tr_cons_tracker *t) {
    t->bytes = tr_budget_add(tracker_bytes(t->k, t->nb, t->nm, t->nid, t->nterms, total_runs(t), t->nsteps), certs_bytes(t));
}

/* a deep copy of everything but the per-revision certificates (snap) and the companion; steps capacity + extra */
static tr_cons_tracker *tracker_copy(const tr_cons_tracker *s, uint64_t extra_steps, int32_t *rc) {
    tr_cons_tracker *t = (tr_cons_tracker *)zalloc(sizeof *t);
    if (!t) { *rc = TR_MATH_MEMORY; return NULL; }
    *t = *s;
    t->magic = TRACKER_MAGIC;
    t->bexp = NULL; t->mexp = NULL; t->spec_exp = NULL; t->bparent = NULL; t->bvar = NULL; t->mleft = NULL;
    t->mright = NULL; t->moment = NULL; t->offset = NULL; t->support = NULL; t->source = NULL; t->exps = NULL;
    t->coefs = NULL; t->cexp = NULL; t->ccoef = NULL; t->coff = NULL; t->cn = NULL; t->ccol = NULL; t->st = NULL;
    t->steps = NULL; t->companion = NULL;
#define DUP(field, bytes) do { uint64_t _b = (uint64_t)(bytes); t->field = zalloc(_b); if (!t->field) goto fail; \
                               if (_b && s->field) memcpy(t->field, s->field, (size_t)_b); } while (0)
    DUP(bexp, (uint64_t)s->nb * s->k); DUP(mexp, (uint64_t)s->nm * s->k);
    DUP(spec_exp, (uint64_t)(s->spec.nterms > 0 ? s->spec.nterms : 0) * s->k);
    DUP(bparent, (uint64_t)s->nb * 4); DUP(bvar, (uint64_t)s->nb * 4);
    DUP(mleft, (uint64_t)s->nm * 4); DUP(mright, (uint64_t)s->nm * 4);
    DUP(moment, (uint64_t)s->nm * sizeof(__int128));
    DUP(offset, ((uint64_t)s->nid + 1) * 8); DUP(support, (uint64_t)s->nid * 4); DUP(source, (uint64_t)s->nid * 4);
    DUP(exps, s->nterms * (uint64_t)s->kc); DUP(coefs, s->nterms * 8);
    DUP(cexp, s->nterms * (uint64_t)MAXV); DUP(ccoef, s->nterms * 8); DUP(coff, (uint64_t)s->nid * 8);
    DUP(cn, (uint64_t)s->nid * 4); DUP(ccol, (uint64_t)s->nid * MAXV * 4);
#undef DUP
    t->spec.exponents = t->spec.nterms ? t->spec_exp : NULL;
    t->st = (istate *)zalloc((uint64_t)(s->nid ? s->nid : 1) * sizeof(istate));
    t->steps = (tr_cons_step *)zalloc((s->nsteps + extra_steps + 1) * sizeof(tr_cons_step));
    if (!t->st || !t->steps) goto fail;
    if (s->nsteps) memcpy(t->steps, s->steps, (size_t)s->nsteps * sizeof(tr_cons_step));
    for (int32_t i = 0; i < s->nid; i++) {
        istate *d = &t->st[i];
        const istate *o = &s->st[i];
        *d = *o;
        d->runs = NULL; d->snap = NULL;
        d->capruns = o->nruns + 1;
        d->runs = (uint64_t *)zalloc(d->capruns * 16);
        if (!d->runs) goto fail;
        if (o->nruns) memcpy(d->runs, o->runs, (size_t)o->nruns * 16);
    }
    *rc = TR_MATH_OK;
    return t;
fail:
    tr_cons_tracker_free(t);
    *rc = TR_MATH_MEMORY;
    return NULL;
}

/* the candidate view of the tracked identities with the given words */
static void cand_view(const tr_cons_tracker *t, const int32_t *words, tr_ident_cand_view *v) {
    memset(v, 0, sizeof *v);
    v->ncand = t->nid; v->nvars = t->kc;
    for (int32_t i = 0; i < TR_IDENT_MAX_VARS; i++) v->var[i] = i < t->kc ? t->cvar[i] : -1;
    v->nterms = t->nterms; v->offset = t->offset; v->support = t->support; v->words = words;
    v->exponents = t->exps; v->coefficients = t->coefs;
}
/* the evaluation words of every identity under the extremes of a view */
static void words_for(const tr_cons_tracker *t, const tr_scope_view *v, int32_t *words) {
    uint64_t M[TR_IDENT_MAX_VARS];
    for (int32_t i = 0; i < t->kc; i++) M[i] = tr_cons_mag(v, t->cvar[i]);
    tr_ident_cand_view cv;
    cand_view(t, words, &cv);
    for (int32_t j = 0; j < t->nid; j++) words[j] = tr_cons_words(&cv, j, M);
}
static void moment_view(const tr_cons_tracker *t, tr_ident_moment_view *m) {
    memset(m, 0, sizeof *m);
    m->nvars = t->k; m->nbasis = t->nb; m->nmoments = t->nm; m->weight = -1;
    for (int32_t i = 0; i < TR_IDENT_MAX_VARS; i++) m->var[i] = i < t->k ? t->var[i] : -1;
    m->repr = TR_IDENT_EXACT; m->nprimes = 0;
    m->basis = t->bexp; m->bparent = t->bparent; m->bvar = t->bvar; m->mleft = t->mleft; m->mright = t->mright;
    m->primes = NULL;
}
int32_t tr_cons_tracker_claim(const tr_cons_tracker *t, int32_t i, tr_cert_claim *out) {
    if (!tracker_valid(t) || i < 0 || i >= t->nid || !out) return TR_MATH_INVALID;
    memset(out, 0, sizeof *out);
    out->kind = TR_CLAIM_POLYNOMIAL; out->policy = TR_CERT_NULL_SKIP; out->carrier = TR_CARRIER_I128;
    out->ncols = t->cn[i];
    for (int32_t q = 0; q < TR_CERT_MAX_COLS; q++) out->col[q] = q < t->cn[i] ? t->ccol[(uint64_t)i * MAXV + q] : -1;
    out->nterms = t->offset[i + 1] - t->offset[i];
    out->exponents = t->cexp + t->coff[i];
    out->coefficients = t->ccoef + t->offset[i];
    return TR_MATH_OK;
}

/* ---------------------------------------------------------------- per-step bookkeeping */
static int32_t push_run(istate *s, uint64_t step) {
    if (s->streak > 1 && s->nruns) { s->runs[2 * (s->nruns - 1) + 1] = step; return TR_MATH_OK; }
    if (s->nruns + 1 > s->capruns) {
        uint64_t cap = s->capruns ? 2 * s->capruns : 4;
        uint64_t *r = (uint64_t *)realloc(s->runs, (size_t)(cap * 16));
        if (!r) return TR_MATH_MEMORY;
        s->runs = r; s->capruns = cap;
    }
    s->runs[2 * s->nruns] = step; s->runs[2 * s->nruns + 1] = step;
    s->nruns++;
    return TR_MATH_OK;
}
/* records the status of a step */
static int32_t note_status(istate *s, uint64_t step) {
    switch (s->status) {
    case TR_CONS_CERTIFIED:
        s->certified++;
        if (s->first_certified == TR_CONS_NONE) s->first_certified = step;
        s->streak = (s->streak && s->last_certified + 1 == step) ? s->streak + 1 : 1;
        s->last_certified = step;
        return push_run(s, step);
    case TR_CONS_VERIFIED: s->verified++; break;
    case TR_CONS_HELD: s->held++; break;
    case TR_CONS_FAILED: s->failed++; break;
    case TR_CONS_UNDECIDED: s->undecided++; break;
    default: break;
    }
    s->streak = 0;
    return TR_MATH_OK;
}
static void record_witness(const tr_cons_tracker *t, istate *s, int32_t i, const tr_scope_view *v, uint64_t row,
                           int32_t source, int64_t weight) {
    s->fail_row = row; s->fail_source = source; s->fail_weight = weight;
    s->fail_ncols = t->cn[i];
    for (int32_t q = 0; q < MAXV; q++) { s->fail_col[q] = -1; s->fail_value[q] = 0; s->fail_valid[q] = 0; }
    if (row == TR_CONS_NONE || !v || row >= v->rows) return;
    for (int32_t q = 0; q < t->cn[i]; q++) {
        int32_t c = t->ccol[(uint64_t)i * MAXV + q];
        s->fail_col[q] = c;
        int present = tr_scope_present(v, c, row);
        s->fail_valid[q] = (uint8_t)present;
        s->fail_value[q] = present ? tr_scope_value(v, c, row) : 0;
    }
}
/* certificates of the holding identities on a bound scope (evidence = the scope's exhaustive evidence) */
static int32_t certify_bound(tr_cons_tracker *t, const tr_scope *scope, const tr_scope_view *v, int32_t backend,
                             const tr_math_budget *budget, tr_math_refusal *refusal) {
    int32_t *words = (int32_t *)zalloc((uint64_t)(t->nid ? t->nid : 1) * 4);
    if (!words) return TR_MATH_MEMORY;
    words_for(t, v, words);
    int32_t rc = TR_MATH_OK;
    for (int32_t i = 0; i < t->nid && !rc; i++) {
        istate *s = &t->st[i];
        s->code = 0;
        if (s->sticky || s->overflow) { s->status = TR_CONS_UNDECIDED; continue; }
        if (s->violations) { s->status = TR_CONS_FAILED; continue; }
        tr_cert_claim claim;
        tr_cons_tracker_claim(t, i, &claim);
        int32_t err = 0;
        tr_cert *c;
        if (words[i] == 2) {
            tr_cert_evidence e;
            memset(&e, 0, sizeof e);
            e.rows = v->rows; e.selected = v->selected; e.checked = (uint64_t)s->checked; e.excluded = (uint64_t)s->excluded;
            e.violations = 0; e.witness = TR_CERT_NONE; e.overflow = 0; e.unit = TR_CERT_UNIT_ROWS; e.holds = 1;
            c = tr_cert_adopt(scope, &claim, &e, backend, NULL, 0, budget, refusal, &err);
        } else {
            tr_cert_evidence fe;
            c = tr_cert_snapshot(scope, &claim, NULL, 0, &fe, budget, refusal, &err);
        }
        if (c) { s->snap = c; s->status = TR_CONS_CERTIFIED; }
        else if (err == TR_MATH_OVERFLOW || err == TR_MATH_UNSUPPORTED) { s->status = TR_CONS_VERIFIED; s->code = err; }
        else rc = err ? err : TR_MATH_MEMORY;          /* inconsistent evidence (TR_MATH_INVALID), budget, memory */
    }
    free(words);
    return rc;
}
/* exhaustive verification of every identity on a scope (DISC-002 tr_ident_verify_view) into the states */
static int32_t verify_scope(tr_cons_tracker *t, const tr_scope_view *v, uint64_t step, uint64_t revision,
                            const tr_math_budget *budget, tr_math_refusal *refusal) {
    int32_t *words = (int32_t *)zalloc((uint64_t)(t->nid ? t->nid : 1) * 4);
    tr_cert_evidence *ev = (tr_cert_evidence *)zalloc((uint64_t)(t->nid ? t->nid : 1) * sizeof *ev);
    if (!words || !ev) { free(words); free(ev); return TR_MATH_MEMORY; }
    words_for(t, v, words);
    tr_ident_cand_view cv;
    cand_view(t, words, &cv);
    int32_t rc = t->nid ? tr_ident_verify_view(v, &cv, budget, ev, refusal) : TR_MATH_OK;
    for (int32_t i = 0; i < t->nid && !rc; i++) {
        istate *s = &t->st[i];
        s->checked = (int64_t)ev[i].checked; s->excluded = (int64_t)ev[i].excluded;
        s->violations = (int64_t)ev[i].violations; s->overflow = (int64_t)ev[i].overflow; s->sticky = 0;
        s->witness_ins = s->witness_del = TR_CONS_NONE;
        if (ev[i].violations && s->fail_step == TR_CONS_NONE) {
            s->fail_step = step; s->fail_revision = revision; s->fail_violations = ev[i].violations;
            record_witness(t, s, i, v, ev[i].witness, TR_CONS_WITNESS_SCOPE, 1);
        }
    }
    free(words); free(ev);
    return rc;
}
/* the companion DELTA-001 state of a scope: a JOINT bank of the constant and the plan variables */
static tr_delta_state *companion_of(const tr_scope *scope, const tr_cons_tracker *t, const tr_math_budget *budget,
                                    tr_math_refusal *refusal, int32_t *error) {
    const int32_t nterms = t->k + 1;
    uint8_t *e = (uint8_t *)zalloc((uint64_t)nterms * t->ncols);
    if (!e) { set_error(error, TR_MATH_MEMORY); return NULL; }
    for (int32_t i = 0; i < t->k; i++) e[(uint64_t)(i + 1) * t->ncols + t->var[i]] = 1;
    tr_bank *b = tr_bank_build(scope, e, nterms, TR_BANK_JOINT, budget, refusal, error);
    free(e);
    if (!b) return NULL;
    tr_delta_state *s = tr_delta_state_from_bank(b, scope, TR_DELTA_NONNEGATIVE, budget, refusal, error);
    tr_bank_free(b);
    return s;
}
/* the Gram rows the companion holds: the pooled count of its constant term */
static int32_t companion_domain(const tr_delta_state *s, int64_t *out) {
    tr_bank_states st;
    int32_t rc = tr_delta_state_get_states(s, &st);
    if (rc) return rc;
    __int128 d = 0;
    for (uint64_t g = 0; g < st.ngroups; g++) d += (int64_t)st.count[g * (uint64_t)st.nterms];
    if (d < INT64_MIN || d > INT64_MAX) return TR_MATH_OVERFLOW;
    *out = (int64_t)d;
    return TR_MATH_OK;
}
/* the Gram of a scope through DISC-002 (plan EXACT on it), its moments copied into the tracker */
static int32_t gram_into(tr_cons_tracker *t, const tr_scope *scope, const tr_math_budget *budget,
                         tr_math_refusal *refusal) {
    int32_t rc = 0;
    tr_ident_plan *p = tr_ident_plan_make(scope, &t->spec, budget, refusal, &rc);
    if (!p) return rc;
    tr_ident_plan_info pi;
    tr_ident_plan_get_info(p, &pi);
    if (pi.repr != TR_IDENT_EXACT) { tr_ident_plan_free(p); return TR_MATH_OVERFLOW; }
    if (pi.nmoments != t->nm || pi.nbasis != t->nb) { tr_ident_plan_free(p); return TR_MATH_INVALID; }
    tr_ident_gram *g = tr_ident_gram_build(scope, p, budget, refusal, &rc);
    tr_ident_plan_free(p);
    if (!g) return rc;
    int64_t *acc = (int64_t *)zalloc((uint64_t)t->nm * 16);
    if (!acc) { tr_ident_gram_free(g); return TR_MATH_MEMORY; }
    tr_ident_gram_states st;
    rc = tr_ident_gram_get_states(g, &st, acc, (uint64_t)t->nm * 2);
    if (!rc) {
        for (int32_t m = 0; m < t->nm; m++)
            t->moment[m] = (__int128)(((unsigned __int128)(uint64_t)acc[2 * m + 1] << 64) | (uint64_t)acc[2 * m]);
        t->domain = (int64_t)st.domain;
    }
    free(acc);
    tr_ident_gram_free(g);
    return rc;
}
static void step_fill(tr_cons_tracker *t, tr_cons_step *s, uint64_t step, int32_t kind, const tr_scope *scope,
                      uint64_t rows, uint64_t ins, uint64_t del, const uint64_t delta_id[2]) {
    memset(s, 0, sizeof *s);
    s->step = step; s->kind = kind; s->rows = rows; s->inserted = ins; s->deleted = del;
    s->domain = (uint64_t)t->domain;
    if (scope) {
        tr_scope_info si;
        tr_scope_get_info(scope, &si);
        s->bound = 1; s->revision = si.revision; s->digest[0] = si.digest[0]; s->digest[1] = si.digest[1];
    }
    if (delta_id) { s->delta[0] = delta_id[0]; s->delta[1] = delta_id[1]; }
    for (int32_t i = 0; i < t->nid; i++) {
        s->certified += t->st[i].status == TR_CONS_CERTIFIED;
        s->failed += t->st[i].status == TR_CONS_FAILED;
    }
}
static void bind_to(tr_cons_tracker *t, const tr_scope *scope) {
    tr_scope_info si;
    if (scope && !tr_scope_get_info(scope, &si)) {
        t->bound = 1; t->revision = si.revision; t->digest[0] = si.digest[0]; t->digest[1] = si.digest[1];
        t->srows = si.rows; t->ssel = si.selected;
    } else {
        t->bound = 0; t->revision = 0; t->digest[0] = t->digest[1] = 0; t->srows = 0; t->ssel = 0;
    }
}

/* the identities of a set: representable, resolved items, with their candidate terms and FND-007 claims */
static int32_t take_identities(tr_cons_tracker *t, const tr_ident_set *set, const int32_t *declared) {
    tr_ident_set_info si;
    tr_ident_cand_view sv;
    int32_t rc = tr_ident_set_get_info(set, &si);
    if (!rc) rc = tr_ident_set_get_view(set, &sv);
    if (rc) return rc;
    if (si.ncand > TR_CONS_MAX_IDENTITIES) return TR_MATH_UNSUPPORTED;
    tr_ident_item *items = (tr_ident_item *)zalloc((uint64_t)(si.ncand ? si.ncand : 1) * sizeof *items);
    if (!items) return TR_MATH_MEMORY;
    if (tr_ident_set_items(set, items, si.ncand) < 0) { free(items); return TR_MATH_INVALID; }
    int32_t n = 0;
    uint64_t terms = 0;
    for (int32_t i = 0; i < si.ncand; i++)
        if (items[i].representable && items[i].resolved) { n++; terms += sv.offset[i + 1] - sv.offset[i]; }
    t->kc = sv.nvars;
    for (int32_t q = 0; q < MAXV; q++) t->cvar[q] = q < sv.nvars ? sv.var[q] : -1;
    t->nid = n; t->nterms = terms;
    t->offset = (uint64_t *)zalloc(((uint64_t)n + 1) * 8); t->support = (uint32_t *)zalloc((uint64_t)(n ? n : 1) * 4);
    t->source = (int32_t *)zalloc((uint64_t)(n ? n : 1) * 4);
    t->exps = (uint8_t *)zalloc((terms ? terms : 1) * (uint64_t)t->kc); t->coefs = (int64_t *)zalloc((terms ? terms : 1) * 8);
    t->cexp = (uint8_t *)zalloc((terms ? terms : 1) * MAXV); t->ccoef = (int64_t *)zalloc((terms ? terms : 1) * 8);
    t->coff = (uint64_t *)zalloc((uint64_t)(n ? n : 1) * 8); t->cn = (int32_t *)zalloc((uint64_t)(n ? n : 1) * 4);
    t->ccol = (int32_t *)zalloc((uint64_t)(n ? n : 1) * MAXV * 4);
    t->st = (istate *)zalloc((uint64_t)(n ? n : 1) * sizeof(istate));
    if (!t->offset || !t->support || !t->source || !t->exps || !t->coefs || !t->cexp || !t->ccoef || !t->coff || !t->cn ||
        !t->ccol || !t->st) { free(items); return TR_MATH_MEMORY; }
    int32_t j = 0;
    uint64_t at = 0, cat = 0;
    for (int32_t i = 0; i < si.ncand && !rc; i++) {
        if (!(items[i].representable && items[i].resolved)) continue;
        uint64_t t0 = sv.offset[i], nt = sv.offset[i + 1] - t0;
        t->offset[j] = at; t->support[j] = sv.support[i]; t->source[j] = i;
        memcpy(t->exps + at * t->kc, sv.exponents + t0 * sv.nvars, (size_t)(nt * (uint64_t)t->kc));
        memcpy(t->coefs + at, sv.coefficients + t0, (size_t)(nt * 8));
        tr_cert_claim c;
        rc = tr_ident_claim(set, i, &c);
        if (rc) break;
        if (c.nterms != nt || c.ncols < 1 || c.ncols > MAXV) { rc = TR_MATH_INVALID; break; }
        t->cn[j] = c.ncols; t->coff[j] = cat;
        for (int32_t q = 0; q < MAXV; q++) t->ccol[(uint64_t)j * MAXV + q] = q < c.ncols ? c.col[q] : -1;
        memcpy(t->cexp + cat, c.exponents, (size_t)(nt * (uint64_t)c.ncols));
        memcpy(t->ccoef + at, c.coefficients, (size_t)(nt * 8));
        cat += nt * (uint64_t)c.ncols;
        istate *s = &t->st[j];
        s->declared = declared ? declared[i] != 0 : (items[i].origin == TR_IDENT_ORIGIN_DECLARED || items[i].known);
        s->first_certified = s->last_certified = TR_CONS_NONE;
        s->fail_step = s->fail_revision = s->fail_row = TR_CONS_NONE;
        s->witness_ins = s->witness_del = TR_CONS_NONE;
        for (int32_t q = 0; q < MAXV; q++) s->fail_col[q] = -1;
        at += nt; j++;
    }
    t->offset[n] = at;
    free(items);
    return rc;
}

tr_cons_tracker *tr_cons_tracker_make(const tr_scope *scope, const tr_ident_spec *spec, const tr_ident_set *set,
                                      const int32_t *declared, const tr_math_budget *budget, tr_math_refusal *refusal,
                                      int32_t *error) {
    tr_budget_none(refusal);
    tr_scope_view v;
    if (!scope || !spec || tr_scope_get_view(scope, &v)) { set_error(error, TR_MATH_INVALID); return NULL; }
    if (spec->weight != -1) { set_error(error, TR_MATH_UNSUPPORTED); return NULL; }   /* multiplicities are the weights */
    int32_t rc = 0;
    tr_ident_plan *p = tr_ident_plan_make(scope, spec, budget, refusal, &rc);
    if (!p) { set_error(error, rc); return NULL; }
    tr_ident_plan_info pi;
    tr_ident_moment_view mv;
    tr_ident_plan_get_info(p, &pi);
    tr_ident_plan_get_view(p, &mv);
    if (pi.repr != TR_IDENT_EXACT) { tr_ident_plan_free(p); set_error(error, TR_MATH_OVERFLOW); return NULL; }
    if (set) {                                       /* a set of this scope's columns */
        tr_ident_cand_view sv;
        tr_ident_set_info si;
        if (tr_ident_set_get_view(set, &sv) || tr_ident_set_get_info(set, &si)) { tr_ident_plan_free(p); set_error(error, TR_MATH_INVALID); return NULL; }
        for (int32_t q = 0; q < sv.nvars; q++) if (sv.var[q] < 0 || sv.var[q] >= v.ncols) { tr_ident_plan_free(p); set_error(error, TR_MATH_INVALID); return NULL; }
        if (si.ncand > TR_CONS_MAX_IDENTITIES) { tr_ident_plan_free(p); set_error(error, TR_MATH_UNSUPPORTED); return NULL; }
    }
    tr_math_need need;
    need_zero(&need);
    need.need[TR_BUDGET_VARIABLES] = (uint64_t)pi.nvars;
    need.need[TR_BUDGET_SUPPORT] = (uint64_t)pi.nbasis;
    need.need[TR_BUDGET_STATES] = (uint64_t)pi.nmoments + TR_CONS_MAX_IDENTITIES;
    need.need[TR_BUDGET_BYTES] = tracker_bytes(pi.nvars, pi.nbasis, pi.nmoments, TR_CONS_MAX_IDENTITIES,
                                               set ? (uint64_t)TR_CERT_MAX_TERMS : 0, 0, 1);
    if (set) {
        tr_ident_cand_view sv;
        tr_ident_set_get_view(set, &sv);
        need.need[TR_BUDGET_STATES] = tr_budget_add((uint64_t)pi.nmoments, (uint64_t)sv.ncand);
        need.need[TR_BUDGET_OUTPUT] = (uint64_t)sv.ncand;
        need.need[TR_BUDGET_BYTES] = tracker_bytes(pi.nvars, pi.nbasis, pi.nmoments, sv.ncand, sv.nterms, 0, 1);
    } else need.need[TR_BUDGET_STATES] = (uint64_t)pi.nmoments;
    rc = check_need(budget, &need, refusal);
    if (rc) { tr_ident_plan_free(p); set_error(error, rc); return NULL; }
    tr_cons_tracker *t = (tr_cons_tracker *)zalloc(sizeof *t);
    if (!t) { tr_ident_plan_free(p); set_error(error, TR_MATH_MEMORY); return NULL; }
    t->magic = TRACKER_MAGIC;
    t->k = pi.nvars; t->nb = pi.nbasis; t->nm = pi.nmoments; t->degree = pi.degree; t->ncols = v.ncols;
    for (int32_t i = 0; i < MAXV; i++) t->var[i] = i < t->k ? mv.var[i] : -1;
    t->spec = *spec;
    t->spec_exp = (uint8_t *)zalloc((uint64_t)(spec->nterms > 0 ? spec->nterms : 1) * (uint64_t)spec->nvars);
    t->bexp = (uint8_t *)zalloc((uint64_t)t->nb * t->k); t->mexp = (uint8_t *)zalloc((uint64_t)t->nm * t->k);
    t->bparent = (int32_t *)zalloc((uint64_t)t->nb * 4); t->bvar = (int32_t *)zalloc((uint64_t)t->nb * 4);
    t->mleft = (int32_t *)zalloc((uint64_t)t->nm * 4); t->mright = (int32_t *)zalloc((uint64_t)t->nm * 4);
    t->moment = (__int128 *)zalloc((uint64_t)t->nm * sizeof(__int128));
    t->steps = (tr_cons_step *)zalloc(sizeof(tr_cons_step));
    if (!t->spec_exp || !t->bexp || !t->mexp || !t->bparent || !t->bvar || !t->mleft || !t->mright || !t->moment || !t->steps) {
        tr_ident_plan_free(p); tr_cons_tracker_free(t); set_error(error, TR_MATH_MEMORY); return NULL;
    }
    if (spec->nterms > 0) memcpy(t->spec_exp, spec->exponents, (size_t)spec->nterms * (size_t)spec->nvars);
    t->spec.exponents = spec->nterms > 0 ? t->spec_exp : NULL;
    tr_ident_plan_basis(p, t->bexp, (uint64_t)t->nb * t->k);
    tr_ident_plan_moments(p, t->mexp, (uint64_t)t->nm * t->k);
    memcpy(t->bparent, mv.bparent, (size_t)t->nb * 4); memcpy(t->bvar, mv.bvar, (size_t)t->nb * 4);
    memcpy(t->mleft, mv.mleft, (size_t)t->nm * 4); memcpy(t->mright, mv.mright, (size_t)t->nm * 4);
    tr_ident_plan_free(p);
    if (set) rc = take_identities(t, set, declared);
    else {
        t->kc = 1; t->cvar[0] = t->var[0];
        for (int32_t q = 1; q < MAXV; q++) t->cvar[q] = -1;
        t->offset = (uint64_t *)zalloc(8); t->support = (uint32_t *)zalloc(4); t->source = (int32_t *)zalloc(4);
        t->exps = (uint8_t *)zalloc(1); t->coefs = (int64_t *)zalloc(8); t->cexp = (uint8_t *)zalloc(1);
        t->ccoef = (int64_t *)zalloc(8); t->coff = (uint64_t *)zalloc(8); t->cn = (int32_t *)zalloc(4);
        t->ccol = (int32_t *)zalloc(MAXV * 4); t->st = (istate *)zalloc(sizeof(istate));
        if (!t->offset || !t->support || !t->source || !t->exps || !t->coefs || !t->cexp || !t->ccoef || !t->coff ||
            !t->cn || !t->ccol || !t->st) rc = TR_MATH_MEMORY;
    }
    if (!rc) rc = gram_into(t, scope, budget, refusal);
    if (!rc) {
        t->companion = companion_of(scope, t, budget, refusal, &rc);
        if (t->companion) {
            int64_t d = 0;
            rc = companion_domain(t->companion, &d);
            if (!rc && d != t->domain) rc = TR_MATH_INVALID;
        }
    }
    if (!rc) rc = verify_scope(t, &v, 0, 0, budget, refusal);
    if (!rc) {
        tr_scope_info si;
        tr_scope_get_info(scope, &si);
        for (int32_t i = 0; i < t->nid; i++) if (t->st[i].fail_step == 0) t->st[i].fail_revision = si.revision;
        rc = certify_bound(t, scope, &v, TR_CERT_BY_CPU, budget, refusal);
    }
    for (int32_t i = 0; i < t->nid && !rc; i++) rc = note_status(&t->st[i], 0);
    if (rc) { tr_cons_tracker_free(t); set_error(error, rc); return NULL; }
    bind_to(t, scope);
    step_fill(t, &t->steps[0], 0, TR_CONS_STEP_BUILD, scope, v.selected, v.selected, 0, NULL);
    t->nsteps = 1;
    retally(t);
    set_error(error, TR_MATH_OK);
    return t;
}

/* ================================================================ delta jobs */
struct tr_cons_job_s {
    uint32_t magic;
    int32_t kind;
    const tr_cons_tracker *tracker;
    const tr_delta *delta;
    uint64_t step, rows, wpos, wneg, inserted, deleted, identity[2];
    int32_t *words;
    tr_cons_job_view view;
    unsigned __int128 *bound;                     /* per moment: W B_u (the update's proven magnitude) */
};
static int job_valid(const tr_cons_job *j) { return j && j->magic == JOB_MAGIC; }
void tr_cons_job_free(tr_cons_job *j) {
    if (!job_valid(j)) return;
    free(j->words); free(j->bound);
    j->magic = 0;
    free(j);
}

tr_cons_job *tr_cons_job_make(const tr_cons_tracker *t, const tr_delta *d, const tr_math_budget *budget,
                              tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    tr_delta_batch_info di;
    tr_scope_view dv;
    tr_delta_rows dr;
    if (!tracker_valid(t) || !d || tr_delta_get_info(d, &di) || tr_delta_get_view(d, &dv) || tr_delta_get_rows(d, &dr)) {
        set_error(error, TR_MATH_INVALID); return NULL;
    }
    if (di.ncols != t->ncols) { set_error(error, TR_MATH_DOMAIN_ERROR); return NULL; }   /* another column layout */
    tr_math_need need;
    need_zero(&need);
    need.need[TR_BUDGET_VARIABLES] = (uint64_t)t->k;
    need.need[TR_BUDGET_OUTPUT] = (uint64_t)t->nid;
    need.need[TR_BUDGET_STATES] = tr_budget_add((uint64_t)t->nm, (uint64_t)t->nid);
    need.need[TR_BUDGET_BYTES] = tr_budget_add(sizeof(tr_cons_job), tr_budget_add(tr_budget_mul((uint64_t)t->nid, 4),
                                                                                  tr_budget_mul((uint64_t)t->nm, 16)));
    int32_t rc = check_need(budget, &need, refusal);
    if (rc) { set_error(error, rc); return NULL; }
    /* the weights of the delta, by sign (exact: at most 2^32 rows of int64 weights) */
    unsigned __int128 wp = 0, wn = 0;
    for (uint64_t r = 0; r < dr.rows; r++) {
        int64_t w = dr.weights[r];
        if (w > 0) wp += (uint64_t)w; else if (w < 0) wn += tr_abs_u64(w);
    }
    if (wp > UINT64_MAX || wn > UINT64_MAX || wp + wn > (unsigned __int128)INT64_MAX) { set_error(error, TR_MATH_OVERFLOW); return NULL; }
    /* the update proof: |G_u| + W B_u <= 2^127 - 1 */
    tr_cons_job *j = (tr_cons_job *)zalloc(sizeof *j);
    if (!j) { set_error(error, TR_MATH_MEMORY); return NULL; }
    j->words = (int32_t *)zalloc((uint64_t)(t->nid ? t->nid : 1) * 4);
    j->bound = (unsigned __int128 *)zalloc((uint64_t)(t->nm ? t->nm : 1) * sizeof(unsigned __int128));
    if (!j->words || !j->bound) { free(j->words); free(j->bound); free(j); set_error(error, TR_MATH_MEMORY); return NULL; }
    j->magic = JOB_MAGIC;
    const unsigned __int128 W = wp + wn;
    for (int32_t m = 0; m < t->nm && !rc; m++) {
        unsigned __int128 b = 1;
        for (int32_t i = 0; i < t->k && b <= I128_LIMIT; i++)
            for (uint8_t e = 0; e < t->mexp[(uint64_t)m * t->k + i]; e++) b = satmul(b, tr_cons_mag(&dv, t->var[i]));
        b = satmul(b, W);
        unsigned __int128 g = tr_cert_mag(t->moment[m]);
        if (b > I128_LIMIT || g > I128_LIMIT - b) rc = TR_MATH_OVERFLOW;
        j->bound[m] = b;
    }
    if (rc) { tr_cons_job_free(j); set_error(error, rc); return NULL; }
    j->tracker = t; j->delta = d; j->kind = di.kind; j->step = t->nsteps; j->rows = dr.rows;
    j->wpos = (uint64_t)wp; j->wneg = (uint64_t)wn; j->inserted = di.inserted; j->deleted = di.deleted;
    j->identity[0] = di.identity[0]; j->identity[1] = di.identity[1];
    words_for(t, &dv, j->words);
    j->view.rows = dv;
    j->view.weights = dr.weights;
    moment_view(t, &j->view.plan);
    cand_view(t, j->words, &j->view.cand);
    set_error(error, TR_MATH_OK);
    return j;
}
int32_t tr_cons_job_get_view(const tr_cons_job *j, tr_cons_job_view *out) {
    if (!job_valid(j) || !out) return TR_MATH_INVALID;
    *out = j->view;
    return TR_MATH_OK;
}
int32_t tr_cons_job_need(const tr_cons_job *j, int32_t target, tr_math_need *need) {
    if (!job_valid(j) || !need) return TR_MATH_INVALID;
    if (target == TR_CONS_NEED_DEVICE) { tr_cons_delta_device_need(&j->view, need); return TR_MATH_OK; }
    if (target != TR_CONS_NEED_CPU) return TR_MATH_INVALID;
    need_zero(need);
    const tr_cons_tracker *t = j->tracker;
    need->need[TR_BUDGET_VARIABLES] = (uint64_t)(t->k > t->kc ? t->k : t->kc);
    need->need[TR_BUDGET_SUPPORT] = (uint64_t)t->nb > t->nterms ? (uint64_t)t->nb : t->nterms;
    need->need[TR_BUDGET_STATES] = tr_budget_add((uint64_t)t->nm, (uint64_t)t->nid);
    need->need[TR_BUDGET_OUTPUT] = (uint64_t)t->nid;
    need->need[TR_BUDGET_BYTES] = tr_budget_add(tr_budget_mul((uint64_t)t->nm, 16 + 16), tr_budget_mul((uint64_t)t->nb, 16));
    return TR_MATH_OK;
}
int32_t tr_cons_job_run(const tr_cons_job *j, tr_cons_delta_out *out) {
    if (!job_valid(j) || !out) return TR_MATH_INVALID;
    const tr_cons_job_view *jv = &j->view;
    if (out->nmoments != jv->plan.nmoments || out->nident != jv->cand.ncand || (jv->plan.nmoments && !out->acc) ||
        (jv->cand.ncand && !out->ev)) return TR_MATH_INVALID;
    const int32_t nm = jv->plan.nmoments, n = jv->cand.ncand;
    __int128 *acc = (__int128 *)zalloc((uint64_t)(nm ? nm : 1) * sizeof(__int128));
    __int128 *val = (__int128 *)zalloc((uint64_t)(jv->plan.nbasis ? jv->plan.nbasis : 1) * sizeof(__int128));
    if (!acc || !val) { free(acc); free(val); return TR_MATH_MEMORY; }
    for (int32_t q = 0; q < n; q++) tr_cons_evidence_init(&out->ev[q]);
    uint64_t gi = 0, gd = 0;
    const tr_scope_view *v = &jv->rows;
    for (uint64_t r = 0; r < v->rows; r++) {
        int64_t w = jv->weights[r];
        if (!w) continue;
        if (tr_cons_gram_row(v, &jv->plan, r)) {
            if (w > 0) gi += (uint64_t)w; else gd += tr_abs_u64(w);
            tr_ident_basis_exact(v, &jv->plan, r, val);
            for (int32_t m = 0; m < nm; m++) acc[m] += (__int128)w * (val[jv->plan.mleft[m]] * val[jv->plan.mright[m]]);
        }
        for (int32_t q = 0; q < n; q++) tr_cons_eval_row(v, &jv->cand, q, r, w, &out->ev[q]);
    }
    for (int32_t m = 0; m < nm; m++) { out->acc[2 * m] = (int64_t)(uint64_t)acc[m]; out->acc[2 * m + 1] = (int64_t)(acc[m] >> 64); }
    out->gram_ins = gi; out->gram_del = gd;
    free(acc); free(val);
    return TR_MATH_OK;
}
/* necessary conditions on a device result: every row's weight is counted once per identity, witnesses consistent,
 * the Gram rows and every moment increment within the job's proven bounds */
static int32_t result_consistent(const tr_cons_job *j, const tr_cons_delta_out *o) {
    const tr_cons_tracker *t = j->tracker;
    if (o->nmoments != t->nm || o->nident != t->nid || (t->nm && !o->acc) || (t->nid && !o->ev)) return TR_MATH_INVALID;
    if (o->gram_ins > j->wpos || o->gram_del > j->wneg) return TR_MATH_INVALID;
    for (int32_t q = 0; q < t->nid; q++) {
        const tr_cons_delta_evidence *e = &o->ev[q];
        unsigned __int128 si = (unsigned __int128)e->checked_ins + e->excluded_ins + e->overflow_ins;
        unsigned __int128 sd = (unsigned __int128)e->checked_del + e->excluded_del + e->overflow_del;
        if (si != j->wpos || sd != j->wneg || e->violations_ins > e->checked_ins || e->violations_del > e->checked_del ||
            (e->violations_ins == 0) != (e->witness_ins == TR_CONS_NONE) ||
            (e->violations_del == 0) != (e->witness_del == TR_CONS_NONE) ||
            (e->witness_ins != TR_CONS_NONE && e->witness_ins >= j->rows) ||
            (e->witness_del != TR_CONS_NONE && e->witness_del >= j->rows)) return TR_MATH_INVALID;
    }
    for (int32_t m = 0; m < t->nm; m++) {
        __int128 x = (__int128)(((unsigned __int128)(uint64_t)o->acc[2 * m + 1] << 64) | (uint64_t)o->acc[2 * m]);
        if (tr_cert_mag(x) > j->bound[m]) return TR_MATH_INVALID;
    }
    return TR_MATH_OK;
}

static int32_t sub_counts(int64_t have, uint64_t add, uint64_t del, int64_t *out) {
    __int128 r = (__int128)have + (__int128)add - (__int128)del;
    if (r < 0) return TR_MATH_DOMAIN_ERROR;           /* deleting rows the evidence does not hold */
    if (r > INT64_MAX) return TR_MATH_OVERFLOW;
    *out = (int64_t)r;
    return TR_MATH_OK;
}

tr_cons_tracker *tr_cons_tracker_update(const tr_cons_tracker *t, const tr_cons_job *j, const tr_cons_delta_out *computed,
                                        int32_t backend, const tr_scope *new_scope, const tr_math_budget *budget,
                                        tr_math_refusal *refusal, tr_delta_violation *violation, int32_t *error) {
    tr_budget_none(refusal);
    if (violation) { memset(violation, 0, sizeof *violation); violation->group = TR_DELTA_NONE; violation->term = -1; }
    if (!tracker_valid(t) || !job_valid(j) || j->tracker != t || j->step != t->nsteps ||
        (backend != TR_CERT_BY_CPU && backend != TR_CERT_BY_DEVICE)) { set_error(error, TR_MATH_INVALID); return NULL; }
    if (t->nsteps >= TR_CONS_MAX_STEPS) { set_error(error, TR_MATH_UNSUPPORTED); return NULL; }
    const int diff = j->kind == TR_DELTA_DIFF;
    if (diff != (new_scope != NULL)) { set_error(error, TR_MATH_INVALID); return NULL; }   /* a scope binds a DIFF only */
    tr_math_need need;
    need_zero(&need);
    need.need[TR_BUDGET_STATES] = tr_budget_add((uint64_t)t->nm, (uint64_t)t->nid);
    need.need[TR_BUDGET_OUTPUT] = (uint64_t)t->nid;
    need.need[TR_BUDGET_BYTES] = tr_budget_add(tracker_bytes(t->k, t->nb, t->nm, t->nid, t->nterms, total_runs(t) + (uint64_t)t->nid,
                                                             t->nsteps + 1),
                                               tr_budget_add(tr_budget_mul((uint64_t)t->nm, 16),
                                                             tr_budget_mul((uint64_t)t->nid, sizeof(tr_cons_delta_evidence))));
    int32_t rc = check_need(budget, &need, refusal);
    if (rc) { set_error(error, rc); return NULL; }
    /* the job's result: computed here, or a device result checked */
    tr_cons_delta_out own;
    memset(&own, 0, sizeof own);
    int64_t *oacc = NULL;
    tr_cons_delta_evidence *oev = NULL;
    const tr_cons_delta_out *res = computed;
    if (!computed) {
        oacc = (int64_t *)zalloc((uint64_t)(t->nm ? t->nm : 1) * 16);
        oev = (tr_cons_delta_evidence *)zalloc((uint64_t)(t->nid ? t->nid : 1) * sizeof *oev);
        if (!oacc || !oev) { free(oacc); free(oev); set_error(error, TR_MATH_MEMORY); return NULL; }
        own.nmoments = t->nm; own.nident = t->nid; own.acc = oacc; own.ev = oev;
        rc = tr_cons_job_run(j, &own);
        res = &own;
    } else rc = result_consistent(j, computed);
    if (rc) { free(oacc); free(oev); set_error(error, rc); return NULL; }
    /* the companion: applicability, the nonnegative measure, the binding */
    tr_delta_state *comp = tr_delta_state_apply(t->companion, j->delta, budget, refusal, violation, &rc);
    if (!comp) { free(oacc); free(oev); set_error(error, rc); return NULL; }
    tr_scope_view nv;
    if (diff) {
        int32_t b = tr_delta_state_bound_to(comp, new_scope);
        if (b != 1 || tr_scope_get_view(new_scope, &nv)) {
            tr_delta_state_free(comp); free(oacc); free(oev); set_error(error, b < 0 ? b : TR_MATH_DOMAIN_ERROR); return NULL;
        }
    }
    tr_cons_tracker *n = tracker_copy(t, 1, &rc);
    if (!n) { tr_delta_state_free(comp); free(oacc); free(oev); set_error(error, rc); return NULL; }
    n->companion = comp;
    /* the Gram: exact additions under the job's proof */
    for (int32_t m = 0; m < t->nm && !rc; m++) {
        __int128 x = (__int128)(((unsigned __int128)(uint64_t)res->acc[2 * m + 1] << 64) | (uint64_t)res->acc[2 * m]), y;
        if (__builtin_add_overflow(n->moment[m], x, &y) || tr_cert_mag(y) > I128_LIMIT) rc = TR_MATH_OVERFLOW;
        else n->moment[m] = y;
    }
    if (!rc) rc = sub_counts(t->domain, res->gram_ins, res->gram_del, &n->domain);
    if (!rc) {
        int64_t d = 0;
        rc = companion_domain(comp, &d);
        if (!rc && d != n->domain) rc = TR_MATH_INVALID;      /* the Gram rows disagree with the companion */
    }
    const uint64_t step = t->nsteps;
    tr_scope_info si;
    memset(&si, 0, sizeof si);
    if (diff) tr_scope_get_info(new_scope, &si);
    /* the evidence of every identity */
    for (int32_t q = 0; q < t->nid && !rc; q++) {
        istate *s = &n->st[q];
        const tr_cons_delta_evidence *e = &res->ev[q];
        s->witness_ins = e->witness_ins; s->witness_del = e->witness_del;
        rc = sub_counts(s->checked, e->checked_ins, e->checked_del, &s->checked);
        if (!rc) rc = sub_counts(s->excluded, e->excluded_ins, e->excluded_del, &s->excluded);
        if (!rc) rc = sub_counts(s->violations, e->violations_ins, e->violations_del, &s->violations);
        if (rc) break;
        if (e->overflow_ins || e->overflow_del) s->sticky = 1;     /* undecided rows: unknown until a rebind */
        s->code = 0;
        if (s->sticky || s->overflow) s->status = TR_CONS_UNDECIDED;
        else if (s->violations) s->status = TR_CONS_FAILED;
        else s->status = diff ? TR_CONS_PENDING : TR_CONS_HELD;
        if (s->violations && s->fail_step == TR_CONS_NONE) {
            s->fail_step = step; s->fail_revision = diff ? si.revision : 0; s->fail_violations = (uint64_t)s->violations;
            if (e->witness_ins != TR_CONS_NONE) record_witness(n, s, q, &j->view.rows, e->witness_ins, TR_CONS_WITNESS_INSERTED, j->view.weights[e->witness_ins]);
            else record_witness(n, s, q, &j->view.rows, e->witness_del, TR_CONS_WITNESS_DELETED,
                                e->witness_del != TR_CONS_NONE ? j->view.weights[e->witness_del] : 0);
        }
    }
    free(oacc); free(oev);
    if (!rc && diff) {
        bind_to(n, new_scope);
        rc = certify_bound(n, new_scope, &nv, backend, budget, refusal);
    } else if (!rc) bind_to(n, NULL);
    for (int32_t q = 0; q < n->nid && !rc; q++) rc = note_status(&n->st[q], step);
    if (rc) { tr_cons_tracker_free(n); set_error(error, rc); return NULL; }
    int32_t kind = j->kind == TR_DELTA_DIFF ? TR_CONS_STEP_DIFF : j->kind == TR_DELTA_SCOPE ? TR_CONS_STEP_SCOPE : TR_CONS_STEP_ARRAYS;
    step_fill(n, &n->steps[step], step, kind, diff ? new_scope : NULL, j->rows, j->inserted, j->deleted, j->identity);
    n->nsteps = step + 1;
    retally(n);
    set_error(error, TR_MATH_OK);
    return n;
}

tr_cons_tracker *tr_cons_tracker_rebind(const tr_cons_tracker *t, const tr_scope *scope, const tr_math_budget *budget,
                                        tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    tr_scope_view v;
    if (!tracker_valid(t) || !scope || tr_scope_get_view(scope, &v)) { set_error(error, TR_MATH_INVALID); return NULL; }
    if (v.ncols != t->ncols) { set_error(error, TR_MATH_DOMAIN_ERROR); return NULL; }
    if (t->nsteps >= TR_CONS_MAX_STEPS) { set_error(error, TR_MATH_UNSUPPORTED); return NULL; }
    tr_math_need need;
    need_zero(&need);
    need.need[TR_BUDGET_STATES] = tr_budget_add((uint64_t)t->nm, (uint64_t)t->nid);
    need.need[TR_BUDGET_BYTES] = tracker_bytes(t->k, t->nb, t->nm, t->nid, t->nterms, total_runs(t) + (uint64_t)t->nid, t->nsteps + 1);
    int32_t rc = check_need(budget, &need, refusal);
    if (rc) { set_error(error, rc); return NULL; }
    tr_cons_tracker *n = tracker_copy(t, 1, &rc);
    if (!n) { set_error(error, rc); return NULL; }
    const uint64_t step = t->nsteps;
    tr_scope_info si;
    tr_scope_get_info(scope, &si);
    rc = gram_into(n, scope, budget, refusal);
    if (!rc) {
        n->companion = companion_of(scope, n, budget, refusal, &rc);
        if (n->companion) {
            int64_t d = 0;
            rc = companion_domain(n->companion, &d);
            if (!rc && d != n->domain) rc = TR_MATH_INVALID;
        }
    }
    if (!rc) rc = verify_scope(n, &v, step, si.revision, budget, refusal);
    if (!rc) rc = certify_bound(n, scope, &v, TR_CERT_BY_CPU, budget, refusal);
    for (int32_t q = 0; q < n->nid && !rc; q++) rc = note_status(&n->st[q], step);
    if (rc) { tr_cons_tracker_free(n); set_error(error, rc); return NULL; }
    bind_to(n, scope);
    step_fill(n, &n->steps[step], step, TR_CONS_STEP_REBIND, scope, v.selected, v.selected, 0, NULL);
    n->nsteps = step + 1;
    retally(n);
    set_error(error, TR_MATH_OK);
    return n;
}


/* ---------------------------------------------------------------- evidence classes */
/* 1 when a POLYNOMIAL claim states identity i: the same canonical terms over the same columns and NULL policy */
static int same_claim(const tr_cons_tracker *t, int32_t i, const tr_cert_claim *c) {
    tr_cert_claim mine;
    tr_cons_tracker_claim(t, i, &mine);
    if (c->kind != TR_CLAIM_POLYNOMIAL || c->ncols != mine.ncols || c->nterms != mine.nterms || c->field ||
        c->tolerance || c->carrier != mine.carrier || c->policy != mine.policy) return 0;
    for (int32_t q = 0; q < mine.ncols; q++) if (c->col[q] != mine.col[q]) return 0;
    return !memcmp(c->exponents, mine.exponents, (size_t)(mine.nterms * (uint64_t)mine.ncols)) &&
           !memcmp(c->coefficients, mine.coefficients, (size_t)(mine.nterms * 8));
}
/* the given certificates by class: hypotheses of the identity's claim, schema facts about its columns; anything else
 * is refused */
static int32_t sort_given(const tr_cons_tracker *t, int32_t i, const tr_cert *const *given, int32_t n, int32_t *hyp,
                          int32_t *schema, uint64_t *hyp_size) {
    *hyp = 0; *schema = 0; *hyp_size = 0;
    if (n < 0 || (n && !given)) return TR_MATH_INVALID;
    for (int32_t g = 0; g < n; g++) {
        tr_cert_info ci;
        tr_cert_claim c;
        if (!given[g] || tr_cert_get_info(given[g], &ci) || tr_cert_get_claim(given[g], &c)) return TR_MATH_INVALID;
        if (ci.kind == TR_CERT_HYPOTHESIS) {
            if (!same_claim(t, i, &c)) return TR_MATH_DOMAIN_ERROR;
            if (!*hyp) *hyp_size = ci.sample.size;
            (*hyp)++;
        } else if (ci.kind == TR_CERT_SCHEMA) {
            int ok = 0;
            for (int32_t q = 0; q < t->cn[i]; q++) ok |= c.col[0] == t->ccol[(uint64_t)i * MAXV + q];
            if (!ok) return TR_MATH_DOMAIN_ERROR;
            (*schema)++;
        } else return TR_MATH_DOMAIN_ERROR;           /* a snapshot (not the tracker's own) or a law */
    }
    return TR_MATH_OK;
}
int32_t tr_cons_tracker_classes(const tr_cons_tracker *t, int32_t i, const tr_cert *const *given, int32_t n,
                                tr_cons_classes *out) {
    if (!tracker_valid(t) || i < 0 || i >= t->nid || !out) return TR_MATH_INVALID;
    memset(out, 0, sizeof *out);
    int32_t hyp = 0, schema = 0;
    uint64_t size = 0;
    int32_t rc = sort_given(t, i, given, n, &hyp, &schema, &size);
    if (rc) return rc;
    const istate *s = &t->st[i];
    out->declared = s->declared || schema > 0;
    out->schema = schema;
    out->snapshot = s->snap != NULL;
    out->stability = s->certified > 0;
    out->hypothesis = hyp;
    out->status = s->status;
    out->certified = s->certified;
    out->snapshot_step = s->snap ? t->nsteps - 1 : TR_CONS_NONE;
    out->hypothesis_size = size;
    return TR_MATH_OK;
}
int32_t tr_cons_authorize(const tr_cons_tracker *t, int32_t i, const tr_cert *const *given, int32_t n,
                          const tr_scope *current, uint64_t field, tr_cons_decision *out) {
    if (!tracker_valid(t) || i < 0 || i >= t->nid || !out) return TR_MATH_INVALID;
    memset(out, 0, sizeof *out);
    out->snapshot.stale.column = out->snapshot.stale.dim = -1;
    out->snapshot.premise = -1;
    int32_t hyp = 0, schema = 0;
    uint64_t size = 0;
    int32_t rc = sort_given(t, i, given, n, &hyp, &schema, &size);
    if (rc) return rc;
    const istate *s = &t->st[i];
    out->status = s->status;
    if (s->status == TR_CONS_FAILED) { out->reason = TR_CONS_REFUSE_FAILED; out->cls = TR_CONS_CLASS_NONE; return TR_MATH_DOMAIN_ERROR; }
    if (s->snap) {
        rc = tr_cert_authorize_exact(s->snap, field, current, NULL, 0, &out->snapshot);
        if (rc == TR_MATH_INVALID) return rc;
        out->cls = TR_CONS_CLASS_SNAPSHOT;
        if (out->snapshot.authorized) { out->authorized = 1; out->reason = TR_CONS_OK; return TR_MATH_OK; }
        out->reason = TR_CONS_REFUSE_SNAPSHOT;
        return rc ? rc : TR_MATH_DOMAIN_ERROR;
    }
    if (s->certified) { out->reason = TR_CONS_REFUSE_STABILITY; out->cls = TR_CONS_CLASS_STABILITY; }
    else if (hyp) { out->reason = TR_CONS_REFUSE_STATISTICAL; out->cls = TR_CONS_CLASS_HYPOTHESIS; }
    else if (s->declared || schema) { out->reason = TR_CONS_REFUSE_DECLARED; out->cls = TR_CONS_CLASS_DECLARED; }
    else { out->reason = TR_CONS_REFUSE_NONE; out->cls = TR_CONS_CLASS_NONE; }
    return TR_MATH_DOMAIN_ERROR;
}

/* ---------------------------------------------------------------- reading */
int32_t tr_cons_tracker_get_info(const tr_cons_tracker *t, tr_cons_info *out) {
    if (!tracker_valid(t) || !out) return TR_MATH_INVALID;
    memset(out, 0, sizeof *out);
    out->nvars = t->k; out->nbasis = t->nb; out->nmoments = t->nm; out->nident = t->nid; out->tracked = t->nid;
    out->bound = t->bound; out->degree = t->degree; out->ncols = t->ncols;
    out->step = t->nsteps ? t->nsteps - 1 : 0; out->revision = t->revision; out->domain = (uint64_t)t->domain;
    out->bytes = t->bytes;
    for (int32_t i = 0; i < t->nid; i++) {
        int32_t st = t->st[i].status;
        out->certified += st == TR_CONS_CERTIFIED; out->verified += st == TR_CONS_VERIFIED; out->held += st == TR_CONS_HELD;
        out->failed += st == TR_CONS_FAILED; out->undecided += st == TR_CONS_UNDECIDED;
    }
    out->scope_digest[0] = t->digest[0]; out->scope_digest[1] = t->digest[1];
    tr_delta_info di;
    if (!tr_delta_state_get_info(t->companion, &di)) { out->state_identity[0] = di.identity[0]; out->state_identity[1] = di.identity[1]; }
    return TR_MATH_OK;
}
int32_t tr_cons_tracker_record(const tr_cons_tracker *t, int32_t i, tr_cons_record *out) {
    if (!tracker_valid(t) || i < 0 || i >= t->nid || !out) return TR_MATH_INVALID;
    memset(out, 0, sizeof *out);
    const istate *s = &t->st[i];
    out->index = t->source[i]; out->status = s->status; out->tracked = 1; out->declared = s->declared;
    out->snapshot = s->snap != NULL; out->certificate_code = s->code;
    out->certified = s->certified; out->verified = s->verified; out->held = s->held; out->failed = s->failed;
    out->undecided = s->undecided; out->first_certified = s->first_certified; out->last_certified = s->last_certified;
    out->runs = s->nruns; out->streak = s->streak;
    out->fail_step = s->fail_step; out->fail_revision = s->fail_revision; out->fail_violations = s->fail_violations;
    out->fail_row = s->fail_row; out->fail_source = s->fail_source; out->fail_ncols = s->fail_ncols;
    out->fail_weight = s->fail_weight;
    for (int32_t q = 0; q < MAXV; q++) { out->fail_col[q] = s->fail_col[q]; out->fail_value[q] = s->fail_value[q]; out->fail_valid[q] = s->fail_valid[q]; }
    out->witness_ins = s->witness_ins; out->witness_del = s->witness_del;
    tr_cert_evidence *e = &out->evidence;
    e->checked = (uint64_t)s->checked; e->excluded = (uint64_t)s->excluded; e->violations = (uint64_t)s->violations;
    e->overflow = (uint64_t)s->overflow;
    e->witness = TR_CERT_NONE; e->unit = TR_CERT_UNIT_ROWS;
    e->holds = !s->violations && !s->overflow && !s->sticky;
    e->rows = t->bound ? t->srows : 0;
    e->selected = t->bound ? t->ssel : (uint64_t)(s->checked + s->excluded + s->overflow);
    return TR_MATH_OK;
}
int64_t tr_cons_tracker_runs(const tr_cons_tracker *t, int32_t i, uint64_t *pairs, uint64_t capacity) {
    if (!tracker_valid(t) || i < 0 || i >= t->nid || (capacity && !pairs)) return TR_MATH_INVALID;
    const istate *s = &t->st[i];
    uint64_t m = s->nruns < capacity ? s->nruns : capacity;
    if (m) memcpy(pairs, s->runs, (size_t)m * 16);
    return (int64_t)s->nruns;
}
int64_t tr_cons_tracker_steps(const tr_cons_tracker *t, tr_cons_step *out, uint64_t capacity) {
    if (!tracker_valid(t) || (capacity && !out)) return TR_MATH_INVALID;
    uint64_t m = t->nsteps < capacity ? t->nsteps : capacity;
    if (m) memcpy(out, t->steps, (size_t)m * sizeof *out);
    return (int64_t)t->nsteps;
}
const tr_cert *tr_cons_tracker_certificate(const tr_cons_tracker *t, int32_t i) {
    return tracker_valid(t) && i >= 0 && i < t->nid ? t->st[i].snap : NULL;
}
const tr_delta_state *tr_cons_tracker_state(const tr_cons_tracker *t) { return tracker_valid(t) ? t->companion : NULL; }
int32_t tr_cons_tracker_moments(const tr_cons_tracker *t, uint8_t *exponents, int64_t *sums, uint64_t capacity) {
    if (!tracker_valid(t) || capacity < (uint64_t)t->nm || (t->nm && (!exponents || !sums))) return TR_MATH_INVALID;
    memcpy(exponents, t->mexp, (size_t)t->nm * (size_t)t->k);
    for (int32_t m = 0; m < t->nm; m++) { sums[2 * m] = (int64_t)(uint64_t)t->moment[m]; sums[2 * m + 1] = (int64_t)(t->moment[m] >> 64); }
    return TR_MATH_OK;
}
tr_ident_gram *tr_cons_tracker_gram(const tr_cons_tracker *t, const tr_scope *scope, const tr_math_budget *budget,
                                    tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    if (!tracker_valid(t) || !scope) { set_error(error, TR_MATH_INVALID); return NULL; }
    int32_t b = tr_delta_state_bound_to(t->companion, scope);
    if (b != 1) { set_error(error, b < 0 ? b : TR_MATH_DOMAIN_ERROR); return NULL; }   /* not this revision's scope */
    int32_t rc = 0;
    tr_ident_plan *p = tr_ident_plan_make(scope, &t->spec, budget, refusal, &rc);
    if (!p) { set_error(error, rc); return NULL; }
    tr_ident_plan_info pi;
    tr_ident_plan_get_info(p, &pi);
    uint8_t *me = (uint8_t *)zalloc((uint64_t)t->nm * t->k);
    int64_t *acc = (int64_t *)zalloc((uint64_t)(t->nm ? t->nm : 1) * 16);
    if (!me || !acc) { free(me); free(acc); tr_ident_plan_free(p); set_error(error, TR_MATH_MEMORY); return NULL; }
    if (pi.repr != TR_IDENT_EXACT) rc = TR_MATH_OVERFLOW;
    else if (pi.nmoments != t->nm || tr_ident_plan_moments(p, me, (uint64_t)t->nm * t->k) ||
             memcmp(me, t->mexp, (size_t)t->nm * (size_t)t->k)) rc = TR_MATH_INVALID;
    tr_ident_gram *g = NULL;
    if (!rc) {
        tr_ident_gram_states st;
        memset(&st, 0, sizeof st);
        st.nmoments = t->nm; st.nslots = 1; st.domain = (uint64_t)t->domain; st.positive = (uint64_t)t->domain;
        st.wmin = 1; st.wmax = 1; st.acc = acc;
        for (int32_t m = 0; m < t->nm; m++) { acc[2 * m] = (int64_t)(uint64_t)t->moment[m]; acc[2 * m + 1] = (int64_t)(t->moment[m] >> 64); }
        g = tr_ident_gram_from_states(scope, p, &st, budget, refusal, &rc);
    }
    free(me); free(acc);
    tr_ident_plan_free(p);
    set_error(error, g ? TR_MATH_OK : (rc ? rc : TR_MATH_MEMORY));
    return g;
}

/* ================================================================ conservation */
typedef struct { uint8_t e[MAXV]; __int128 c; } cterm;
typedef struct { uint64_t n; cterm *t; } cpoly;
static void cp_free(cpoly *p) { free(p->t); p->t = NULL; p->n = 0; }
static int cterm_cmp(const void *x, const void *y) {
    const cterm *a = (const cterm *)x, *b = (const cterm *)y;
    int da = 0, db = 0;
    for (int i = 0; i < MAXV; i++) { da += a->e[i]; db += b->e[i]; }
    if (da != db) return da < db ? 1 : -1;           /* decreasing graded */
    int c = memcmp(a->e, b->e, MAXV);
    return c < 0 ? 1 : c > 0 ? -1 : 0;
}
/* sorts and merges equal monomials (checked), drops zeros */
static int32_t cp_merge(cpoly *p) {
    if (p->n > 1) qsort(p->t, (size_t)p->n, sizeof *p->t, cterm_cmp);
    uint64_t m = 0;
    for (uint64_t i = 0; i < p->n;) {
        __int128 s = 0;
        uint64_t j = i;
        for (; j < p->n && !memcmp(p->t[i].e, p->t[j].e, MAXV); j++)
            if (__builtin_add_overflow(s, p->t[j].c, &s)) return TR_MATH_OVERFLOW;
        if (s) { p->t[m] = p->t[i]; p->t[m].c = s; m++; }
        i = j;
    }
    p->n = m;
    return TR_MATH_OK;
}
static int32_t cp_from(const tr_ident_poly *q, int32_t n, cpoly *out) {
    out->n = 0; out->t = NULL;
    if (q->nterms && (!q->exponents || !q->coefficients)) return TR_MATH_INVALID;
    if (q->nterms > TR_CONS_MAX_TERMS) return TR_MATH_UNSUPPORTED;
    out->t = (cterm *)zalloc((q->nterms ? q->nterms : 1) * sizeof(cterm));
    if (!out->t) return TR_MATH_MEMORY;
    for (uint64_t t = 0; t < q->nterms; t++) {
        memcpy(out->t[t].e, q->exponents + t * (uint64_t)n, (size_t)n);
        out->t[t].c = q->coefficients[t];
    }
    out->n = q->nterms;
    return cp_merge(out);
}
static int32_t cp_mul(const cpoly *a, const cpoly *b, const tr_math_budget *budget, tr_math_refusal *refusal, cpoly *out) {
    out->n = 0; out->t = NULL;
    uint64_t pairs = tr_budget_mul(a->n, b->n);
    int32_t rc = check_one(budget, TR_BUDGET_SUPPORT, pairs, refusal);
    if (rc) return rc;
    rc = check_one(budget, TR_BUDGET_BYTES, tr_budget_mul(pairs, sizeof(cterm)), refusal);
    if (rc) return rc;
    out->t = (cterm *)zalloc((pairs ? pairs : 1) * sizeof(cterm));
    if (!out->t) return TR_MATH_MEMORY;
    uint64_t k = 0;
    for (uint64_t i = 0; i < a->n; i++)
        for (uint64_t j = 0; j < b->n; j++) {
            cterm *t = &out->t[k++];
            for (int v = 0; v < MAXV; v++) {
                int e = a->t[i].e[v] + b->t[j].e[v];
                if (e > 255) { cp_free(out); return TR_MATH_UNSUPPORTED; }
                t->e[v] = (uint8_t)e;
            }
            if (__builtin_mul_overflow(a->t[i].c, b->t[j].c, &t->c)) { cp_free(out); return TR_MATH_OVERFLOW; }
        }
    out->n = k;
    rc = cp_merge(out);
    if (rc) cp_free(out);
    return rc;
}
static int32_t cp_scale(cpoly *p, __int128 c) {
    for (uint64_t i = 0; i < p->n; i++) if (__builtin_mul_overflow(p->t[i].c, c, &p->t[i].c)) return TR_MATH_OVERFLOW;
    return TR_MATH_OK;
}
static int32_t cp_append(cpoly *into, const cpoly *from, int negate) {
    cterm *t = (cterm *)realloc(into->t, (size_t)((into->n + from->n + 1) * sizeof(cterm)));
    if (!t) return TR_MATH_MEMORY;
    into->t = t;
    for (uint64_t i = 0; i < from->n; i++) {
        into->t[into->n] = from->t[i];
        if (negate) into->t[into->n].c = -from->t[i].c;
        into->n++;
    }
    return TR_MATH_OK;
}

struct tr_cons_law_s {
    uint32_t magic;
    int32_t n;
    tr_cons_law_info info;
    uint8_t *exps;                                /* P: nterms x n, decreasing graded */
    int64_t *coefs;
    tr_cert *cert;
};
static int law_valid(const tr_cons_law *l) { return l && l->magic == LAW_MAGIC; }
void tr_cons_law_free(tr_cons_law *l) {
    if (!law_valid(l)) return;
    free(l->exps); free(l->coefs); tr_cert_free(l->cert);
    l->magic = 0;
    free(l);
}
/* the value of P at a point in W words (the bound proven by the caller) */
static void law_value(const tr_cons_law *l, const int64_t *x, int32_t W, uint64_t *acc) {
    uint64_t term[TR_IDENT_MAX_WORDS];
    tr_ident_w_set(acc, W, 0);
    for (uint64_t t = 0; t < l->info.residual_terms; t++) {
        const uint8_t *e = l->exps + t * (uint64_t)l->n;
        int64_t co = l->coefs[t];
        int neg = co < 0;
        tr_ident_w_set(term, W, tr_abs_u64(co));
        for (int32_t i = 0; i < l->n; i++) {
            if (!e[i]) continue;
            if (x[i] < 0 && (e[i] & 1)) neg = !neg;
            uint64_t m = tr_abs_u64(x[i]);
            for (uint8_t k = 0; k < e[i]; k++) tr_ident_w_mul(term, W, m);
        }
        if (neg) tr_ident_w_neg(term, W);
        tr_ident_w_add(acc, term, W);
    }
}
static int64_t grid_value(uint64_t i) { return i == 0 ? 0 : (i & 1) ? (int64_t)((i + 1) / 2) : -(int64_t)(i / 2); }

tr_cons_law *tr_cons_conserve(int32_t n, const tr_ident_poly *q, const tr_ident_poly *images,
                              const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    if (n < 1 || !q || !images) { set_error(error, TR_MATH_INVALID); return NULL; }
    if (n > MAXV) { set_error(error, TR_MATH_UNSUPPORTED); return NULL; }
    tr_math_need need;
    need_zero(&need);
    need.need[TR_BUDGET_VARIABLES] = (uint64_t)n;
    need.need[TR_BUDGET_SUPPORT] = q->nterms;
    int32_t rc = check_need(budget, &need, refusal);
    if (rc) { set_error(error, rc); return NULL; }
    cpoly Q = {0, NULL}, T[MAXV], acc = {0, NULL};
    cpoly *pw[MAXV];
    int32_t maxe[MAXV] = {0};
    memset(T, 0, sizeof T);
    memset(pw, 0, sizeof pw);
    rc = cp_from(q, n, &Q);
    for (int32_t v = 0; v < n && !rc; v++) {
        if (images[v].nterms) rc = cp_from(&images[v], n, &T[v]);
        else {                                        /* the identity map x_v */
            T[v].t = (cterm *)zalloc(sizeof(cterm));
            if (!T[v].t) rc = TR_MATH_MEMORY;
            else { T[v].n = 1; T[v].t[0].e[v] = 1; T[v].t[0].c = 1; }
        }
    }
    for (uint64_t t = 0; t < Q.n && !rc; t++) for (int32_t v = 0; v < n; v++) if (Q.t[t].e[v] > maxe[v]) maxe[v] = Q.t[t].e[v];
    /* powers T_v^1 .. T_v^maxe[v] */
    for (int32_t v = 0; v < n && !rc; v++) {
        if (!maxe[v]) continue;
        pw[v] = (cpoly *)zalloc((uint64_t)(maxe[v] + 1) * sizeof(cpoly));
        if (!pw[v]) { rc = TR_MATH_MEMORY; break; }
        pw[v][1].n = T[v].n;
        pw[v][1].t = (cterm *)zalloc((T[v].n ? T[v].n : 1) * sizeof(cterm));
        if (!pw[v][1].t) { rc = TR_MATH_MEMORY; break; }
        memcpy(pw[v][1].t, T[v].t, (size_t)T[v].n * sizeof(cterm));
        for (int32_t k = 2; k <= maxe[v] && !rc; k++) rc = cp_mul(&pw[v][k - 1], &T[v], budget, refusal, &pw[v][k]);
    }
    uint64_t expanded = 0;
    int32_t degree_seen = 0;
    for (uint64_t t = 0; t < Q.n && !rc; t++) {      /* c_t prod_v T_v^{e_tv} */
        cpoly term = {1, (cterm *)zalloc(sizeof(cterm))};
        if (!term.t) { rc = TR_MATH_MEMORY; break; }
        term.t[0].c = 1;
        for (int32_t v = 0; v < n && !rc; v++) {
            uint8_t e = Q.t[t].e[v];
            if (!e) continue;
            cpoly next;
            rc = cp_mul(&term, &pw[v][e], budget, refusal, &next);
            cp_free(&term);
            if (!rc) term = next;
        }
        if (!rc) rc = cp_scale(&term, Q.t[t].c);
        if (!rc) rc = cp_append(&acc, &term, 0);
        cp_free(&term);
        if (!rc) rc = check_one(budget, TR_BUDGET_SUPPORT, acc.n, refusal);
    }
    if (!rc) rc = cp_merge(&acc);
    cpoly qT = {0, NULL};                             /* q(T(x)) merged, kept for the law claim */
    if (!rc) {
        expanded = acc.n;
        qT.t = (cterm *)zalloc((acc.n ? acc.n : 1) * sizeof(cterm));
        if (!qT.t) rc = TR_MATH_MEMORY;
        else { memcpy(qT.t, acc.t, (size_t)acc.n * sizeof(cterm)); qT.n = acc.n; }
    }
    if (!rc) rc = cp_append(&acc, &Q, 1);
    if (!rc) rc = cp_merge(&acc);                     /* P = q(T(x)) - q(x) */
    for (uint64_t t = 0; t < acc.n && !rc; t++) {
        if (acc.t[t].c < INT64_MIN || acc.t[t].c > INT64_MAX) rc = TR_MATH_OVERFLOW;
        int32_t d = 0;
        for (int32_t v = 0; v < n; v++) d += acc.t[t].e[v];
        if (d > degree_seen) degree_seen = d;
    }
    if (!rc && acc.n > TR_CONS_MAX_TERMS) rc = TR_MATH_UNSUPPORTED;
    if (!rc) rc = check_one(budget, TR_BUDGET_DEGREE, (uint64_t)degree_seen, refusal);
    tr_cons_law *l = NULL;
    if (!rc) {
        l = (tr_cons_law *)zalloc(sizeof *l);
        if (l) {
            l->exps = (uint8_t *)zalloc((acc.n ? acc.n : 1) * (uint64_t)n);
            l->coefs = (int64_t *)zalloc((acc.n ? acc.n : 1) * 8);
        }
        if (!l || !l->exps || !l->coefs) {
            if (l) { free(l->exps); free(l->coefs); free(l); }
            l = NULL; rc = TR_MATH_MEMORY;
        }
    }
    if (!rc) {
        l->magic = LAW_MAGIC; l->n = n;
        for (uint64_t t = 0; t < acc.n; t++) { memcpy(l->exps + t * (uint64_t)n, acc.t[t].e, (size_t)n); l->coefs[t] = (int64_t)acc.t[t].c; }
        tr_cons_law_info *I = &l->info;
        I->nvars = n; I->q_terms = Q.n; I->expanded_terms = expanded; I->residual_terms = acc.n; I->degree = degree_seen;
        I->conserved = acc.n == 0;
        for (int32_t v = 0; v < MAXV; v++) I->point[v] = 0;
        if (I->conserved) {                           /* the universal certificate: FND-007 RING_IDENTITY */
            uint64_t nt = qT.n + Q.n;
            uint8_t *ce = (uint8_t *)zalloc((nt ? nt : 1) * (uint64_t)n);
            int64_t *cc = (int64_t *)zalloc((nt ? nt : 1) * 8);
            if (!ce || !cc) rc = TR_MATH_MEMORY;
            for (uint64_t t = 0; t < qT.n && !rc; t++) {
                if (qT.t[t].c < INT64_MIN || qT.t[t].c > INT64_MAX) { rc = TR_MATH_OVERFLOW; break; }
                memcpy(ce + t * (uint64_t)n, qT.t[t].e, (size_t)n); cc[t] = (int64_t)qT.t[t].c;
            }
            for (uint64_t t = 0; t < Q.n && !rc; t++) {
                memcpy(ce + (qT.n + t) * (uint64_t)n, Q.t[t].e, (size_t)n);
                if (Q.t[t].c <= INT64_MIN || Q.t[t].c > INT64_MAX) { rc = TR_MATH_OVERFLOW; break; }
                cc[qT.n + t] = -(int64_t)Q.t[t].c;
            }
            if (!rc && nt) {
                tr_cert_claim c;
                memset(&c, 0, sizeof c);
                c.kind = TR_CLAIM_POLYNOMIAL; c.policy = TR_CERT_NULL_FORBID; c.carrier = TR_CARRIER_I128;
                c.ncols = n;
                for (int32_t v = 0; v < TR_CERT_MAX_COLS; v++) c.col[v] = v < n ? v : -1;
                c.nterms = nt; c.exponents = ce; c.coefficients = cc;
                int32_t e = 0;
                l->cert = tr_cert_law(TR_CERT_LAW_RING_IDENTITY, &c, budget, refusal, &e);
                if (!l->cert) rc = e ? e : TR_MATH_MEMORY;
                else I->certified = 1;
            }
            free(ce); free(cc);
        } else {                                      /* the counterexample search on the Nullstellensatz grid */
            int32_t dv[MAXV] = {0};
            uint64_t M[TR_IDENT_MAX_VARS];
            for (uint64_t t = 0; t < acc.n; t++) for (int32_t v = 0; v < n; v++) if (acc.t[t].e[v] > dv[v]) dv[v] = acc.t[t].e[v];
            uint64_t points = 1;
            for (int32_t v = 0; v < n; v++) {
                points = tr_budget_mul(points, (uint64_t)dv[v] + 1);
                M[v] = dv[v] ? (uint64_t)((dv[v] + 1) / 2) : 1;
                if (!M[v]) M[v] = 1;
            }
            tr_ident_cand_view cv;
            uint64_t off[2] = {0, acc.n};
            uint32_t sup = 1;
            int32_t w0 = 0;
            memset(&cv, 0, sizeof cv);
            cv.ncand = 1; cv.nvars = n; cv.nterms = acc.n; cv.offset = off; cv.support = &sup; cv.words = &w0;
            cv.exponents = l->exps; cv.coefficients = l->coefs;
            int32_t W = tr_cons_words(&cv, 0, M);
            tr_math_refusal tmp;
            if (W && !check_one(budget, TR_BUDGET_ITERATIONS, points, &tmp)) {
                I->searched = 1; I->words = W;
                int64_t x[MAXV] = {0};
                uint64_t val[TR_IDENT_MAX_WORDS];
                for (uint64_t p = 0; p < points && !I->found; p++) {
                    uint64_t r = p;
                    for (int32_t v = n; v-- > 0;) { uint64_t base = (uint64_t)dv[v] + 1; x[v] = grid_value(r % base); r /= base; }
                    law_value(l, x, W, val);
                    I->points = p + 1;
                    if (!tr_ident_w_zero(val, W)) {
                        I->found = 1;
                        for (int32_t v = 0; v < MAXV; v++) I->point[v] = v < n ? x[v] : 0;
                        for (int32_t w = 0; w < TR_IDENT_MAX_WORDS; w++) I->value[w] = w < W ? val[w] : 0;
                    }
                }
            }
        }
        I->bytes = sizeof *l + acc.n * ((uint64_t)n + 8);
        if (l->cert) { tr_cert_info ci; if (!tr_cert_get_info(l->cert, &ci)) I->bytes += ci.bytes; }
    }
    cp_free(&Q); cp_free(&acc); cp_free(&qT);
    for (int32_t v = 0; v < MAXV; v++) {
        cp_free(&T[v]);
        if (pw[v]) { for (int32_t k = 0; k <= maxe[v]; k++) cp_free(&pw[v][k]); free(pw[v]); }
    }
    if (rc) { tr_cons_law_free(l); set_error(error, rc); return NULL; }
    set_error(error, TR_MATH_OK);
    return l;
}
int32_t tr_cons_law_get_info(const tr_cons_law *l, tr_cons_law_info *out) {
    if (!law_valid(l) || !out) return TR_MATH_INVALID;
    *out = l->info;
    return TR_MATH_OK;
}
int32_t tr_cons_law_residual(const tr_cons_law *l, uint8_t *e, int64_t *c, uint64_t capacity, uint64_t *nterms) {
    if (!law_valid(l) || !nterms) return TR_MATH_INVALID;
    uint64_t n = l->info.residual_terms;
    *nterms = n;
    if (capacity < n || (n && (!e || !c))) return TR_MATH_INVALID;
    if (n) { memcpy(e, l->exps, (size_t)(n * (uint64_t)l->n)); memcpy(c, l->coefs, (size_t)(n * 8)); }
    return TR_MATH_OK;
}
const tr_cert *tr_cons_law_certificate(const tr_cons_law *l) { return law_valid(l) ? l->cert : NULL; }

/* P over the bound variables only (every variable it reads must be bound) */
static int32_t law_bound(const tr_cons_law *l, const int32_t *cols, int32_t *bv, int32_t *bc, int32_t *nb,
                         uint8_t **exps) {
    *nb = 0; *exps = NULL;
    for (int32_t v = 0; v < l->n; v++) {
        int reads = 0;
        for (uint64_t t = 0; t < l->info.residual_terms; t++) reads |= l->exps[t * (uint64_t)l->n + v] != 0;
        if (cols[v] < 0) { if (reads) return TR_MATH_INVALID; continue; }
        bv[*nb] = v; bc[*nb] = cols[v]; (*nb)++;
    }
    if (!*nb) return TR_MATH_INVALID;
    uint64_t nt = l->info.residual_terms;
    *exps = (uint8_t *)zalloc((nt ? nt : 1) * (uint64_t)*nb);
    if (!*exps) return TR_MATH_MEMORY;
    for (uint64_t t = 0; t < nt; t++)
        for (int32_t q = 0; q < *nb; q++) (*exps)[t * (uint64_t)*nb + q] = l->exps[t * (uint64_t)l->n + bv[q]];
    return TR_MATH_OK;
}
tr_ident_set *tr_cons_law_candidates(const tr_cons_law *l, const tr_scope *scope, const int32_t *cols,
                                     const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    if (!law_valid(l) || !scope || !cols) { set_error(error, TR_MATH_INVALID); return NULL; }
    int32_t bv[MAXV], bc[MAXV], nb = 0;
    uint8_t *e = NULL;
    int32_t rc = law_bound(l, cols, bv, bc, &nb, &e);
    if (rc) { set_error(error, rc); return NULL; }
    tr_ident_poly p;
    p.nterms = l->info.residual_terms; p.exponents = e; p.coefficients = l->coefs;
    tr_ident_set *s = tr_ident_candidates(scope, nb, bc, &p, 1, NULL, budget, refusal, error);
    free(e);
    return s;
}
int32_t tr_cons_law_witnesses(const tr_cons_law *l, const tr_scope *scope, const int32_t *cols,
                              const tr_math_budget *budget, tr_cert_evidence *evidence, uint64_t *rows,
                              uint64_t capacity, uint64_t *found, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    tr_scope_view v;
    if (!law_valid(l) || !scope || !cols || !evidence || !found || (capacity && !rows) || tr_scope_get_view(scope, &v))
        return TR_MATH_INVALID;
    int32_t rc = 0;
    tr_ident_set *s = tr_cons_law_candidates(l, scope, cols, budget, refusal, &rc);
    if (!s) return rc;
    tr_ident_cand_view cv;
    tr_ident_set_get_view(s, &cv);
    memset(evidence, 0, sizeof *evidence);
    evidence->rows = v.rows; evidence->selected = v.selected; evidence->witness = TR_CERT_NONE;
    evidence->unit = TR_CERT_UNIT_ROWS;
    *found = 0;
    for (uint64_t w = 0; w < (v.rows + 63) / 64; w++) {
        uint64_t bits = v.selection[w];
        while (bits) {
            uint64_t r = w * 64 + (uint64_t)__builtin_ctzll(bits);
            bits &= bits - 1;
            int st = tr_ident_cand_row(&v, &cv, 0, r);
            if (st == TR_CERT_ROW_EXCLUDED) { evidence->excluded++; continue; }
            if (st == TR_CERT_ROW_OVERFLOW) { evidence->overflow++; continue; }
            evidence->checked++;
            if (st == TR_CERT_ROW_VIOLATES) {
                if (!evidence->violations) evidence->witness = r;
                evidence->violations++;
                if (*found < capacity) rows[*found] = r;
                if (*found < capacity) (*found)++;
            }
        }
    }
    evidence->holds = !evidence->violations && !evidence->overflow;
    tr_ident_set_free(s);
    return TR_MATH_OK;
}
