/* FND-008: complete-plan choice and cost evidence; see math_planner.h. SPDX-License-Identifier: Apache-2.0 */
#include "math_planner.h"
#include "mathematics.h"
#include <stdlib.h>
#include <string.h>

_Static_assert(sizeof(tr_planner_selectivity) == 80, "selectivity ABI size");
_Static_assert(sizeof(tr_planner_force) == 16, "force ABI size");
_Static_assert(sizeof(tr_planner_evidence) == 6168, "evidence ABI size");
_Static_assert((int)TR_PLANNER_REQ_POLYSUM == (int)TR_REG_REQ_POLYSUM && (int)TR_PLANNER_REQ_VERIFY == (int)TR_REG_REQ_VERIFY,
               "request kinds");

int32_t tr_planner_sample(const uint64_t *selected, uint64_t rows, uint64_t sample_words, int32_t exact,
                          tr_planner_selectivity *s) {
    if (!s || (!exact && !sample_words)) return TR_MATH_INVALID;
    memset(s, 0, sizeof *s);
    uint64_t words = (rows + 63) / 64, n = exact || sample_words >= words ? words : sample_words;
    s->rows = rows; s->words = words; s->sample_words = n;
    for (uint64_t i = 0; i < n; i++) {
        uint64_t w = tr_planner_sample_word(i, n, words), mask = tr_reg_rowmask(rows, w);
        s->sample_rows += (uint64_t)__builtin_popcountll(mask);
        s->sample_selected += (uint64_t)__builtin_popcountll((selected ? selected[w] : ~(uint64_t)0) & mask);
    }
    s->exact = exact || n == words;
    s->method = exact ? TR_PLANNER_EXACT : n == words ? TR_PLANNER_COVERED : TR_PLANNER_SAMPLED;
    tr_planner_finish(s);
    return TR_MATH_OK;
}

/* ------------------------------------------------------------------ forced controls */
static const char *const repr_names[TR_REG_REPRESENTATIONS] = {
    "", "formal", "moments", "cells", "verify", "scan", "histogram", "power", "bitplane", "empty"};
static int request_has(int32_t request, int32_t repr) {
    if (request == 0) return 1;                                   /* an explicit force: the first entry, strictly */
    if (request == TR_PLANNER_REQ_VERIFY) return repr == TR_REG_R_VERIFY;
    return repr >= TR_REG_R_SCAN && repr <= TR_REG_R_EMPTY;
}
static int word_is(const char *s, size_t n, const char *w) { return strlen(w) == n && !strncmp(s, w, n); }
/* one entry "<representation>[@cpu|@cuda|@reuse][:cpu|:gpu]" */
static int32_t parse_entry(const char *s, size_t n, tr_planner_force *f) {
    f->representation = 0; f->device = -1; f->finalize = -1; f->reuse = -1;
    size_t at = n, colon = n;
    for (size_t i = 0; i < n; i++) {
        if (s[i] == '@' && at == n) at = i;
        else if (s[i] == ':' && colon == n) colon = i;
    }
    if (colon < at) return TR_MATH_INVALID;
    size_t end = at < colon ? at : colon;
    for (int32_t r = 1; r < TR_REG_REPRESENTATIONS; r++) if (word_is(s, end, repr_names[r])) f->representation = r;
    if (!f->representation) return TR_MATH_INVALID;
    if (at < n) {
        const char *w = s + at + 1;
        size_t len = colon - at - 1;
        if (word_is(w, len, "cpu")) { f->device = TR_REG_ON_CPU; f->reuse = 0; }
        else if (word_is(w, len, "cuda")) { f->device = TR_REG_ON_CUDA; f->reuse = 0; }
        else if (word_is(w, len, "reuse")) { f->device = TR_REG_ON_CPU; f->reuse = 1; }
        else return TR_MATH_INVALID;
    }
    if (colon < n) {
        const char *w = s + colon + 1;
        size_t len = n - colon - 1;
        if (word_is(w, len, "cpu")) f->finalize = TR_REG_FINAL_CPU;
        else if (word_is(w, len, "gpu")) f->finalize = TR_REG_FINAL_GPU;
        else return TR_MATH_INVALID;
    }
    return TR_MATH_OK;
}
int32_t tr_planner_force_parse(const char *text, int32_t request, tr_planner_force *out) {
    if (!out) return TR_MATH_INVALID;
    out->representation = 0; out->device = -1; out->finalize = -1; out->reuse = -1;
    if (!text) return TR_MATH_OK;
    int found = 0;
    for (const char *s = text; *s;) {
        const char *e = strchr(s, ',');
        size_t n = e ? (size_t)(e - s) : strlen(s);
        while (n && (*s == ' ')) { s++; n--; }
        while (n && s[n - 1] == ' ') n--;
        if (n) {
            tr_planner_force f;
            if (parse_entry(s, n, &f)) return TR_MATH_INVALID;
            if (!found && request_has(request, f.representation)) { *out = f; found = 1; }
        }
        if (!e) break;
        s = e + 1;
    }
    return TR_MATH_OK;
}

/* ------------------------------------------------------------------ the choice */
static int matches(const tr_planner_force *f, const tr_reg_candidate *c) {
    return f && f->representation && c->representation == f->representation && (f->device < 0 || c->device == f->device) &&
           (f->finalize < 0 || c->finalize == f->finalize) && (f->reuse < 0 || c->reuse == f->reuse);
}
static void cost(tr_reg_candidate *c, const double *coefficients) {
    double rates[TR_COST_COMPONENTS];
    int dev = c->device == TR_REG_ON_CUDA ? 1 : 0;
    if (coefficients && c->module > 0 && c->module < TR_REG_MODULES)
        memcpy(rates, coefficients + ((size_t)c->module * 2 + (size_t)dev) * TR_COST_COMPONENTS, sizeof rates);
    else {
        tr_reg_module m;
        if (tr_reg_module_by_id(c->module, &m) == TR_MATH_OK) memcpy(rates, m.cost[dev], sizeof rates);
        else memset(rates, 0, sizeof rates);
    }
    double total = 0, shared = 0;
    for (int i = 0; i < TR_COST_COMPONENTS; i++) {
        if (i == TR_COST_FUSION) continue;
        c->cost[i] = (double)c->amount[i] * rates[i];
        if ((c->shared >> i) & 1) shared += c->cost[i];
        total += c->cost[i];
    }
    c->cost[TR_COST_FUSION] = c->batch > 1 ? -rates[TR_COST_FUSION] * shared * (double)(c->batch - 1) / (double)c->batch : 0.0;
    c->total = total + c->cost[TR_COST_FUSION];
}
int32_t tr_planner_choose(tr_reg_candidate *c, int32_t n, const tr_math_budget *budget, const double *coefficients,
                          const tr_planner_force *force, tr_planner_evidence *ev) {
    if (!c || !ev || n < 0 || n > TR_REG_MAX_CANDIDATES) return TR_MATH_INVALID;
    tr_math_budget dflt;
    if (!budget) { tr_budget_default(&dflt); budget = &dflt; }
    int32_t best = -1, forced_best = -1, first_match = -1, first_budget = -1, first_other = -1;
    int any_force = force && force->representation;
    for (int32_t i = 0; i < n; i++) {
        tr_reg_candidate *x = &c[i];
        cost(x, coefficients);
        tr_budget_none(&x->budget);
        if (!x->valid) { x->status = TR_PLANNER_S_REJECTED; x->reason = TR_PLANNER_R_INVALID; }
        else if (tr_budget_check(budget, &x->need, &x->budget)) {
            x->status = TR_PLANNER_S_REJECTED; x->reason = TR_PLANNER_R_BUDGET; x->code = TR_MATH_BUDGET;
            x->refusal.kind = TR_REG_REFUSE_BUDGET; x->refusal.code = TR_MATH_BUDGET;
        } else { x->status = TR_PLANNER_S_ADMISSIBLE; x->reason = TR_PLANNER_R_COSTLIER; }
        if (x->status == TR_PLANNER_S_REJECTED) {
            if (x->reason == TR_PLANNER_R_BUDGET && first_budget < 0) first_budget = i;
            if (x->refusal.kind != TR_REG_REFUSE_PROOF && first_other < 0) first_other = i;
        }
        if (matches(force, x) && first_match < 0) first_match = i;
        if (x->status != TR_PLANNER_S_ADMISSIBLE) continue;
        if (best < 0 || x->total < c[best].total) best = i;
        if (matches(force, x) && (forced_best < 0 || x->total < c[forced_best].total)) forced_best = i;
    }
    ev->forced = any_force;
    if (force) ev->force = *force;
    int32_t chosen = any_force ? forced_best : best, code = TR_MATH_OK;
    ev->refusal = (tr_math_refusal){-1, 0, 0, 0};
    if (chosen >= 0) {
        c[chosen].status = TR_PLANNER_S_CHOSEN;
        c[chosen].reason = any_force ? TR_PLANNER_R_FORCED : TR_PLANNER_R_CHEAPEST;
        if (any_force)
            for (int32_t i = 0; i < n; i++)
                if (i != chosen && c[i].status == TR_PLANNER_S_ADMISSIBLE) c[i].reason = TR_PLANNER_R_NOT_FORCED;
        ev->reason = c[chosen].reason;
    } else if (any_force && first_match < 0) {
        code = TR_MATH_UNSUPPORTED; ev->reason = TR_PLANNER_R_FORCE_UNMATCHED;
    } else if (any_force) {
        code = c[first_match].code ? c[first_match].code : TR_MATH_UNSUPPORTED; ev->reason = TR_PLANNER_R_FORCE_REFUSED;
        if (code == TR_MATH_BUDGET) ev->refusal = c[first_match].budget;
    } else {
        ev->reason = TR_PLANNER_R_NONE_ADMISSIBLE;
        if (first_budget >= 0) { code = TR_MATH_BUDGET; ev->refusal = c[first_budget].budget; }
        else code = first_other >= 0 && c[first_other].code ? c[first_other].code : TR_MATH_UNSUPPORTED;
    }
    ev->chosen = chosen;
    ev->code = code;
    ev->ncandidates = n;
    if (c != ev->candidates) memcpy(ev->candidates, c, (size_t)n * sizeof *c);
    return code;
}

static void evidence_init(tr_planner_evidence *ev, int32_t request, int32_t devices) {
    memset(ev, 0, sizeof *ev);
    ev->abi = TR_PLANNER_ABI; ev->request = request; ev->chosen = -1; ev->devices = devices;
    ev->force.device = ev->force.finalize = ev->force.reuse = -1;
    tr_budget_none(&ev->refusal);
}
static int32_t resolve_force(const tr_planner_force *force, int32_t request, tr_planner_force *out) {
    if (force) { *out = *force; return TR_MATH_OK; }
    return tr_planner_force_parse(getenv("TORBLES_MATH_PLAN_FORCE"), request, out);
}
int32_t tr_planner_polysum(const tr_reg_request *r, const tr_reg_state *cached, int32_t devices,
                           const tr_math_budget *budget, const tr_planner_selectivity *sel,
                           const tr_planner_force *force, const double *coefficients, tr_planner_evidence *ev) {
    if (!ev) return TR_MATH_INVALID;
    evidence_init(ev, TR_PLANNER_REQ_POLYSUM, devices);
    tr_planner_force f;
    int32_t rc = r ? resolve_force(force, TR_PLANNER_REQ_POLYSUM, &f) : TR_MATH_INVALID;
    if (!rc && sel && sel->rows != r->rows) rc = TR_MATH_INVALID;
    if (!rc && !sel) rc = tr_planner_sample(r->selected, r->rows, TR_PLANNER_SAMPLE_WORDS, 0, &ev->selectivity);
    if (rc) { ev->code = rc; ev->reason = TR_PLANNER_R_SEARCH; return rc; }
    if (sel) ev->selectivity = *sel;
    ev->force = f;
    if (f.representation && f.device == TR_REG_ON_CUDA) devices |= TR_REG_ON_CUDA;
    ev->devices = devices;
    int32_t n = 0;
    const tr_planner_selectivity *s = &ev->selectivity;
    rc = tr_reg_candidates(r, cached, devices, s->estimate, s->upper, s->empty_proven, budget, ev->candidates,
                           TR_REG_MAX_CANDIDATES, &n, &ev->refusal);
    if (rc) { ev->code = rc; ev->reason = TR_PLANNER_R_SEARCH; ev->forced = f.representation != 0; return rc; }
    return tr_planner_choose(ev->candidates, n, budget, coefficients, &f, ev);
}
int32_t tr_planner_verify(const tr_disc_view *v, int32_t devices, const tr_math_budget *budget,
                          const tr_planner_force *force, const double *coefficients, tr_planner_evidence *ev) {
    if (!ev) return TR_MATH_INVALID;
    evidence_init(ev, TR_PLANNER_REQ_VERIFY, devices);
    tr_planner_force f;
    int32_t rc = v ? resolve_force(force, TR_PLANNER_REQ_VERIFY, &f) : TR_MATH_INVALID;
    if (rc) { ev->code = rc; ev->reason = TR_PLANNER_R_SEARCH; return rc; }
    ev->force = f;
    if (f.representation && f.device == TR_REG_ON_CUDA) devices |= TR_REG_ON_CUDA;
    ev->devices = devices;
    tr_planner_sample(v->used, v->rows, TR_PLANNER_SAMPLE_WORDS, 1, &ev->selectivity);
    int32_t n = 0;
    rc = tr_reg_candidates_verify(v, devices, budget, ev->candidates, TR_REG_MAX_CANDIDATES, &n, &ev->refusal);
    if (rc) { ev->code = rc; ev->reason = TR_PLANNER_R_SEARCH; ev->forced = f.representation != 0; return rc; }
    return tr_planner_choose(ev->candidates, n, budget, coefficients, &f, ev);
}
