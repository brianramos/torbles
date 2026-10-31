/* UNC-001: intervals, affine forms and congruence reasoning; see math_unc.h.
 * SPDX-License-Identifier: Apache-2.0 */
#include "math_unc.h"
#include "mathematics.h"
#include "math_domain.h"
#include "math_handles.h"
#include <stdlib.h>
#include <string.h>

TR_MATH_ABI(tr_unc_node, TR_UNC_ABI);       /* the ctypes boundary of UNC-001 (math_handles.h) */
TR_MATH_ABI(tr_unc_program, TR_UNC_ABI);
TR_MATH_ABI(tr_unc_input, TR_UNC_ABI);
TR_MATH_ABI(tr_unc_term, TR_UNC_ABI);
TR_MATH_ABI(tr_unc_value, TR_UNC_ABI);
TR_MATH_ABI(tr_unc_result, TR_UNC_ABI);
TR_MATH_ABI(tr_unc_filter_stats, TR_UNC_ABI);
TR_MATH_ABI(tr_unc_sum, TR_UNC_ABI);
TR_MATH_ABI(tr_unc_state_info, TR_UNC_ABI);

_Static_assert(sizeof(tr_unc_node) == 32, "tr_unc_node ABI");
_Static_assert(sizeof(tr_unc_program) == 56, "tr_unc_program ABI");
_Static_assert(sizeof(tr_unc_input) == 48, "tr_unc_input ABI");
_Static_assert(sizeof(tr_unc_term) == 32, "tr_unc_term ABI");
_Static_assert(sizeof(tr_unc_value) == 648, "tr_unc_value ABI");
_Static_assert(sizeof(tr_unc_result) == 200, "tr_unc_result ABI");
_Static_assert(sizeof(tr_unc_plan) == 528, "tr_unc_plan ABI");
_Static_assert(sizeof(tr_unc_filter_stats) == 88, "tr_unc_filter_stats ABI");
_Static_assert(sizeof(tr_unc_sum) == 48, "tr_unc_sum ABI");
_Static_assert(sizeof(tr_unc_state_info) == 152, "tr_unc_state_info ABI");

#define STATE_MAGIC UINT32_C(0x31434e55)

struct tr_unc_state_s {
    uint32_t magic;
    int32_t ncols, scoped, reserved;
    uint64_t ngroups, bytes;
    int32_t cols[TR_UNC_MAX_COLS];
    tr_unc_input *inputs;                   /* ngroups x ncols */
    tr_cert_record *record;                 /* the scope's record (scoped) */
};

static void set_error(int32_t *error, int32_t code) { if (error) *error = code; }
static void need_zero(tr_math_need *n) { for (int i = 0; i < TR_BUDGET_RESOURCES; i++) n->need[i] = 0; }
static const tr_math_budget *budget_or(const tr_math_budget *b, tr_math_budget *dflt) {
    if (b) return b;
    tr_budget_default(dflt);
    return dflt;
}
static int state_valid(const tr_unc_state *s) { return s && s->magic == STATE_MAGIC; }
/* ------------------------------------------------------------------ programs */
int32_t tr_unc_program_check(const tr_unc_program *p, tr_unc_plan *plan, tr_math_need *need) {
    return tr_unc_check(p, 0, plan, need);
}

/* ------------------------------------------------------------------ states */
int32_t tr_unc_state_need(uint64_t ngroups, int32_t ncols, uint64_t rows, int32_t scoped, tr_math_need *need) {
    if (!need || ncols < 1) return TR_MATH_INVALID;
    need_zero(need);
    uint64_t cells = tr_budget_mul(ngroups, (uint64_t)ncols);
    need->need[TR_BUDGET_STATES] = cells;
    need->need[TR_BUDGET_VARIABLES] = (uint64_t)ncols;
    uint64_t bytes = tr_budget_add(tr_budget_mul(cells, sizeof(tr_unc_input)), sizeof(tr_unc_state));
    if (scoped) {                            /* the accumulators (base values) and the record's selection bits */
        bytes = tr_budget_add(bytes, tr_budget_mul(cells, 8));
        bytes = tr_budget_add(bytes, tr_budget_bitmap_bytes(rows));
    }
    need->need[TR_BUDGET_BYTES] = bytes;
    return TR_MATH_OK;
}
static tr_unc_state *state_alloc(uint64_t ngroups, int32_t ncols, int32_t *error) {
    tr_unc_state *s = calloc(1, sizeof *s);
    uint64_t cells = ngroups * (uint64_t)ncols;
    if (s) s->inputs = calloc((size_t)(cells ? cells : 1), sizeof *s->inputs);
    if (!s || !s->inputs) { if (s) free(s); set_error(error, TR_MATH_MEMORY); return NULL; }
    s->magic = STATE_MAGIC; s->ncols = ncols; s->ngroups = ngroups;
    s->bytes = sizeof *s + cells * sizeof *s->inputs;
    for (int32_t j = 0; j < TR_UNC_MAX_COLS; j++) s->cols[j] = -1;
    return s;
}
tr_unc_state *tr_unc_state_from_scope(const tr_scope *scope, const int32_t *cols, int32_t ncols,
                                      const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    tr_scope_view v;
    if (!scope || tr_scope_get_view(scope, &v) || !cols || ncols < 1) { set_error(error, TR_MATH_INVALID); return NULL; }
    if (ncols > TR_UNC_MAX_COLS) { set_error(error, TR_MATH_UNSUPPORTED); return NULL; }
    for (int32_t j = 0; j < ncols; j++) if (cols[j] < 0 || cols[j] >= v.ncols) { set_error(error, TR_MATH_INVALID); return NULL; }
    tr_math_need need;
    tr_unc_state_need(v.ngroups, ncols, v.rows, 1, &need);
    tr_math_budget dflt;
    const tr_math_budget *b = budget_or(budget, &dflt);
    int32_t rc = tr_budget_check(b, &need, refusal);
    if (rc) { set_error(error, rc); return NULL; }
    tr_unc_state *s = state_alloc(v.ngroups, ncols, error);
    if (!s) return NULL;
    uint64_t cells = v.ngroups * (uint64_t)ncols;
    int64_t *base = calloc((size_t)(cells ? cells : 1), sizeof *base);
    uint64_t *rows = calloc((size_t)(v.ngroups ? v.ngroups : 1), sizeof *rows);
    if (!base || !rows) { free(base); free(rows); tr_unc_state_free(s); set_error(error, TR_MATH_MEMORY); return NULL; }
    uint64_t nw = (v.rows + 63) / 64;
    for (uint64_t w = 0; w < nw; w++) {
        uint64_t bits = v.selection[w];
        while (bits) {
            uint64_t row = w * 64 + (uint64_t)tr_ipoly_ctz(bits);
            bits &= bits - 1;
            uint32_t g = tr_scope_group(&v, row);
            rows[g]++;
            for (int32_t j = 0; j < ncols; j++) {
                int32_t c = cols[j];
                if (!tr_scope_present(&v, c, row)) continue;
                int64_t x = tr_scope_value(&v, c, row);
                uint64_t at = (uint64_t)g * (uint64_t)ncols + (uint64_t)j;
                tr_unc_input *in = &s->inputs[at];
                if (!in->present) { in->min = in->max = x; base[at] = x; }
                else {
                    if (x < in->min) in->min = x;
                    if (x > in->max) in->max = x;
                    __int128 d = (__int128)x - base[at];
                    uint64_t m = (uint64_t)(d < 0 ? -d : d);
                    in->modulus = tr_gcd_u64(in->modulus, m);
                }
                in->present++;
            }
        }
    }
    for (uint64_t g = 0; g < v.ngroups; g++)
        for (int32_t j = 0; j < ncols; j++) {
            tr_unc_input *in = &s->inputs[g * (uint64_t)ncols + (uint64_t)j];
            in->rows = rows[g];
            in->scale = v.col[cols[j]].scale;
            if (!in->present) { in->min = in->max = 0; in->modulus = 0; }
        }
    free(base); free(rows);
    for (int32_t j = 0; j < ncols; j++) s->cols[j] = cols[j];
    /* the scope's record: the distinct columns read, the selection bits and the grouping (decisions are per group) */
    int32_t distinct[TR_UNC_MAX_COLS], nd = 0;
    for (int32_t j = 0; j < ncols; j++) {
        int seen = 0;
        for (int32_t i = 0; i < nd; i++) seen |= distinct[i] == cols[j];
        if (!seen) distinct[nd++] = cols[j];
    }
    int32_t e = 0;
    s->record = tr_cert_record_scope(scope, distinct, nd, TR_CERT_RECORD_SELECTION | TR_CERT_RECORD_GROUPS, NULL, 0, b,
                                     refusal, &e);
    if (!s->record) { tr_unc_state_free(s); set_error(error, e ? e : TR_MATH_MEMORY); return NULL; }
    s->scoped = 1;
    s->bytes += tr_cert_record_bytes(s->record);
    set_error(error, TR_MATH_OK);
    return s;
}
tr_unc_state *tr_unc_state_from_inputs(uint64_t ngroups, int32_t ncols, const tr_unc_input *inputs,
                                       const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    if (ncols < 1 || ngroups < 1 || !inputs) { set_error(error, TR_MATH_INVALID); return NULL; }
    if (ncols > TR_UNC_MAX_COLS) { set_error(error, TR_MATH_UNSUPPORTED); return NULL; }
    tr_math_need need;
    tr_unc_state_need(ngroups, ncols, 0, 0, &need);
    tr_math_budget dflt;
    int32_t rc = tr_budget_check(budget_or(budget, &dflt), &need, refusal);
    if (rc) { set_error(error, rc); return NULL; }
    for (uint64_t g = 0; g < ngroups; g++)
        for (int32_t j = 0; j < ncols; j++) {
            const tr_unc_input *in = &inputs[g * (uint64_t)ncols + (uint64_t)j];
            if (!tr_unc_input_ok(in) || in->rows != inputs[g * (uint64_t)ncols].rows) { set_error(error, TR_MATH_INVALID); return NULL; }
        }
    tr_unc_state *s = state_alloc(ngroups, ncols, error);
    if (!s) return NULL;
    for (uint64_t i = 0; i < ngroups * (uint64_t)ncols; i++) {
        s->inputs[i] = inputs[i];
        if (!s->inputs[i].present) { s->inputs[i].min = s->inputs[i].max = 0; s->inputs[i].modulus = 0; }
    }
    set_error(error, TR_MATH_OK);
    return s;
}
int32_t tr_unc_state_get_info(const tr_unc_state *s, tr_unc_state_info *out) {
    if (!state_valid(s) || !out) return TR_MATH_INVALID;
    memset(out, 0, sizeof *out);
    out->ngroups = s->ngroups; out->bytes = s->bytes; out->ncols = s->ncols; out->scoped = s->scoped;
    for (int32_t j = 0; j < TR_UNC_MAX_COLS; j++) out->cols[j] = s->cols[j];
    return TR_MATH_OK;
}
const tr_unc_input *tr_unc_state_inputs(const tr_unc_state *s) { return state_valid(s) ? s->inputs : NULL; }
int32_t tr_unc_state_fresh(const tr_unc_state *s, const tr_scope *current, tr_cert_staleness *out) {
    if (!state_valid(s) || !current || !out) return TR_MATH_INVALID;
    if (!s->scoped) return TR_MATH_INVALID;
    return tr_cert_record_fresh(s->record, current, NULL, 0, TR_CERT_MATCH_EXACT, out);
}
void tr_unc_state_free(tr_unc_state *s) {
    if (!state_valid(s)) return;
    s->magic = 0;
    tr_cert_record_free(s->record);
    free(s->inputs);
    free(s);
}

/* ------------------------------------------------------------------ evaluation */
int32_t tr_unc_eval_need(const tr_unc_program *p, uint64_t ngroups, int32_t ncols, int32_t target, tr_math_need *need) {
    tr_unc_plan plan;
    if (!need) return TR_MATH_INVALID;
    int32_t rc = tr_unc_check(p, 0, &plan, need);
    if (rc) return rc;
    if (ncols != p->ncols || (target != TR_UNC_NEED_CPU && target != TR_UNC_NEED_DEVICE)) return TR_MATH_INVALID;
    need->need[TR_BUDGET_OUTPUT] = ngroups;
    need->need[TR_BUDGET_BYTES] = target == TR_UNC_NEED_CPU
        ? tr_budget_mul((uint64_t)plan.nlive + 1, sizeof(tr_unc_value))
        : tr_unc_eval_device_bytes(ngroups, ncols, p->nnodes, plan.nlive);
    return TR_MATH_OK;
}
int32_t tr_unc_eval(const tr_unc_state *s, const tr_unc_program *p, uint64_t first, uint64_t count, tr_unc_result *out,
                    const tr_math_budget *budget, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!state_valid(s) || !p || (count && !out) || first > s->ngroups || count > s->ngroups - first) return TR_MATH_INVALID;
    tr_unc_plan plan;
    tr_math_need need;
    int32_t rc = tr_unc_check(p, 0, &plan, &need);
    if (rc) return rc;
    if (p->ncols != s->ncols) return TR_MATH_INVALID;
    need.need[TR_BUDGET_OUTPUT] = count;
    need.need[TR_BUDGET_BYTES] = tr_budget_mul((uint64_t)plan.nlive + 1, sizeof(tr_unc_value));
    tr_math_budget dflt;
    rc = tr_budget_check(budget_or(budget, &dflt), &need, refusal);
    if (rc) return rc;
    tr_unc_value *slots = calloc((size_t)plan.nlive + 1, sizeof *slots);
    if (!slots) return TR_MATH_MEMORY;
    for (uint64_t i = 0; i < count; i++)
        tr_unc_eval_group(p, &plan, &s->inputs[(first + i) * (uint64_t)s->ncols], slots, &out[i]);
    free(slots);
    return TR_MATH_OK;
}
int32_t tr_unc_trace(const tr_unc_state *s, const tr_unc_program *p, uint64_t group, tr_unc_value *values, int32_t capacity) {
    if (!state_valid(s) || !p || !values || group >= s->ngroups) return TR_MATH_INVALID;
    tr_unc_plan plan;
    int32_t rc = tr_unc_check(p, 0, &plan, NULL);
    if (rc) return rc;
    if (p->ncols != s->ncols || capacity < p->nnodes) return TR_MATH_INVALID;
    tr_unc_plan_slots(p, &plan, 1);
    tr_unc_value *slots = calloc((size_t)p->nnodes + 1, sizeof *slots);
    if (!slots) return TR_MATH_MEMORY;
    const tr_unc_input *in = &s->inputs[group * (uint64_t)s->ncols];
    for (int32_t k = 0; k < p->nnodes; k++) tr_unc_node_eval(p, &plan, k, in, slots, &slots[p->nnodes]);
    memcpy(values, slots, (size_t)p->nnodes * sizeof *slots);
    free(slots);
    return TR_MATH_OK;
}
/* the symbolic forms of nodes 0..node (EXACT, every term kept, columns as unbounded symbols) */
static int32_t symbolic_eval(const tr_unc_program *p, int32_t node, tr_unc_value *slots) {
    tr_unc_program q = *p;
    q.carrier = TR_UNC_EXACT; q.precision = 0; q.rank = TR_UNC_MAX_TERMS; q.budget = tr_unc_qv(0, 0);
    q.domains = TR_UNC_DOM_AFFINE | TR_UNC_DOM_SYMBOLIC; q.root = node; q.nnodes = node + 1;
    tr_unc_plan plan;
    int32_t rc = tr_unc_check(&q, 1, &plan, NULL);
    if (rc) return rc;
    tr_unc_plan_slots(&q, &plan, 1);
    tr_unc_input dummy[TR_UNC_MAX_COLS];
    memset(dummy, 0, sizeof dummy);
    for (int32_t j = 0; j < TR_UNC_MAX_COLS; j++) dummy[j].rows = 1;
    for (int32_t k = 0; k <= node; k++) {
        const tr_unc_node *nd = &q.nodes[k];
        if (nd->op == TR_UNC_COL) {
            tr_unc_value *v = &slots[k];
            tr_unc_value_top(v, TR_UNC_MAYBE_VALUE);
            v->flags &= ~TR_UNC_AFFINE_TOP;
            tr_unc_s one = tr_unc_s_zero(); one.q = tr_q_int(1);
            tr_unc_term_set(&v->term[0], nd->a, one);
            v->nterms = 1;
            continue;
        }
        tr_unc_node_eval(&q, &plan, k, dummy, slots, &slots[node + 1]);
    }
    return TR_MATH_OK;
}
int32_t tr_unc_symbolic(const tr_unc_program *p, int32_t node, tr_unc_value *out, int32_t *exact) {
    if (!p || !out || !exact) return TR_MATH_INVALID;
    int32_t rc = tr_unc_check(p, 0, NULL, NULL);
    if (rc) return rc;
    if (node < 0 || node >= p->nnodes) return TR_MATH_INVALID;
    tr_unc_value *slots = calloc((size_t)node + 2, sizeof *slots);
    if (!slots) return TR_MATH_MEMORY;
    rc = symbolic_eval(p, node, slots);
    if (!rc) {
        *out = slots[node];
        int ex = out->kind == TR_UNC_NUMERIC && !(out->flags & TR_UNC_AFFINE_TOP);
        for (int32_t i = 0; ex && i < out->nterms; i++) if (out->term[i].sym >= TR_UNC_FRESH) ex = 0;
        *exact = ex;
    }
    free(slots);
    return rc;
}

/* ------------------------------------------------------------------ concrete rows and the filter */
static int32_t rows_check(const tr_scope_view *v, const int32_t *cols, const tr_unc_program *p, const int32_t *decisions) {
    for (int32_t j = 0; j < p->ncols; j++) if (cols[j] < 0 || cols[j] >= v->ncols) return TR_MATH_INVALID;
    if (decisions)
        for (uint64_t g = 0; g < v->ngroups; g++) if (decisions[g] < TR_UNC_UNKNOWN || decisions[g] > TR_UNC_EMPTY) return TR_MATH_INVALID;
    return TR_MATH_OK;
}
int32_t tr_unc_rows(const tr_scope *scope, const int32_t *cols, const tr_unc_program *p, const int32_t *decisions,
                    uint64_t *bitmap, tr_unc_filter_stats *stats, const tr_math_budget *budget, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    tr_scope_view v;
    if (!scope || tr_scope_get_view(scope, &v) || !cols || !p || !stats || (v.rows && !bitmap)) return TR_MATH_INVALID;
    tr_math_need need;
    int32_t rc = tr_unc_check(p, 0, NULL, &need);
    if (rc) return rc;
    rc = rows_check(&v, cols, p, decisions);
    if (rc) return rc;
    need.need[TR_BUDGET_BYTES] = tr_budget_bitmap_bytes(v.rows);
    tr_math_budget dflt;
    rc = tr_budget_check(budget_or(budget, &dflt), &need, refusal);
    if (rc) return rc;
    memset(stats, 0, sizeof *stats);
    stats->rows = v.rows; stats->selected = v.selected; stats->first_overflow = UINT64_MAX;
    tr_unc_rowval vals[TR_UNC_MAX_NODES];
    uint64_t acc[4] = {0, 0, 0, 0}, nw = (v.rows + 63) / 64;
    for (uint64_t w = 0; w < nw; w++) bitmap[w] = tr_unc_rows_word(p, &v, cols, decisions, w, vals, acc, &stats->first_overflow);
    stats->rows_proven = acc[0]; stats->rows_evaluated = acc[1]; stats->rows_true = acc[2]; stats->overflow = acc[3];
    return stats->overflow ? TR_MATH_OVERFLOW : TR_MATH_OK;
}
int32_t tr_unc_filter(const tr_unc_state *s, const tr_scope *scope, const tr_unc_program *p, uint64_t *bitmap,
                      tr_unc_filter_stats *stats, const tr_math_budget *budget, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!state_valid(s) || !s->scoped || !scope || !p || !stats) return TR_MATH_INVALID;
    tr_cert_staleness st;
    int32_t rc = tr_unc_state_fresh(s, scope, &st);
    if (rc) return rc;
    tr_math_need need;
    tr_unc_plan plan;
    rc = tr_unc_check(p, 0, &plan, &need);
    if (rc) return rc;
    if (p->ncols != s->ncols || plan.kind[p->root] != TR_UNC_BOOLEAN) return TR_MATH_INVALID;
    need.need[TR_BUDGET_OUTPUT] = s->ngroups;
    need.need[TR_BUDGET_BYTES] = tr_budget_add(tr_budget_mul(s->ngroups, sizeof(tr_unc_result) + 4),
                                               tr_budget_mul((uint64_t)plan.nlive + 1, sizeof(tr_unc_value)));
    tr_math_budget dflt;
    const tr_math_budget *b = budget_or(budget, &dflt);
    rc = tr_budget_check(b, &need, refusal);
    if (rc) return rc;
    tr_unc_result *res = calloc((size_t)s->ngroups, sizeof *res);
    int32_t *dec = calloc((size_t)s->ngroups, sizeof *dec);
    if (!res || !dec) { free(res); free(dec); return TR_MATH_MEMORY; }
    rc = tr_unc_eval(s, p, 0, s->ngroups, res, b, refusal);
    uint64_t counts[4] = {0, 0, 0, 0};
    if (!rc)
        for (uint64_t g = 0; g < s->ngroups; g++) { dec[g] = res[g].decision; counts[dec[g]]++; }
    free(res);
    if (!rc) rc = tr_unc_rows(scope, s->cols, p, dec, bitmap, stats, b, refusal);
    free(dec);
    if (rc == TR_MATH_OK || rc == TR_MATH_OVERFLOW) {
        stats->groups_unknown = counts[TR_UNC_UNKNOWN]; stats->groups_all = counts[TR_UNC_ALL];
        stats->groups_none = counts[TR_UNC_NONE]; stats->groups_empty = counts[TR_UNC_EMPTY];
    }
    return rc;
}

/* ------------------------------------------------------------------ overflow decisions and the CRT meet */
int32_t tr_unc_sum_fits(const tr_unc_result *r, int32_t bits, tr_num_rational unit, tr_unc_sum *out) {
    if (!r || !out || bits < 2 || bits > 128 || !tr_unc_canonical(unit) || unit.numerator <= 0) return TR_MATH_INVALID;
    memset(out, 0, sizeof *out);
    out->bits = bits;
    if (r->kind != TR_UNC_NUMERIC) return TR_MATH_INVALID;
    if (!(r->null & TR_UNC_MAYBE_VALUE) || !r->rows) {   /* no value: every partial sum is 0 */
        out->proven = 1; out->integral = 1; out->bounded = 1; return TR_MATH_OK;
    }
    /* integral: every value in r + m Z is a multiple of the unit */
    tr_num_rational q1, q2;
    out->integral = !(r->flags & TR_UNC_CONG_TOP) && tr_q_div(r->cr, unit, &q1) && q1.denominator == 1 &&
                    tr_q_div(r->cm, unit, &q2) && q2.denominator == 1;
    if (r->flags & (TR_UNC_LO_INF | TR_UNC_HI_INF)) return TR_MATH_OK;
    /* per value, in units: [ceil(lo / unit), floor(hi / unit)] when integral, else the enclosing integers */
    tr_num_rational ql, qh;
    __int128 vl, vh;
    int okl = tr_q_div(r->lo, unit, &ql), okh = tr_q_div(r->hi, unit, &qh);
    if (!okl || !okh) return TR_MATH_OK;               /* not decidable in int64 rationals: not proven */
    vl = out->integral ? -tr_unc_floordiv(-(__int128)ql.numerator, ql.denominator) : tr_unc_floordiv(ql.numerator, ql.denominator);
    vh = tr_unc_floordiv(qh.numerator, qh.denominator);
    if (!out->integral) vh = -tr_unc_floordiv(-(__int128)qh.numerator, qh.denominator);
    if (vl > 0) vl = 0;
    if (vh < 0) vh = 0;
    /* partial sums lie in [rows vl, rows vh]: exact in 128 bits or not proven */
    unsigned __int128 n = r->rows, ml = (unsigned __int128)(-vl), mh = (unsigned __int128)vh;
    unsigned __int128 lim = (unsigned __int128)1 << 127;
    if ((ml && n > lim / ml) || (mh && n > lim / mh)) return TR_MATH_OK;
    unsigned __int128 sl = n * ml, sh = n * mh;
    if (sl > lim || sh >= lim) return TR_MATH_OK;
    out->bounded = 1;
    out->lo = tr_i128_set(-(__int128)sl); out->hi = tr_i128_set((__int128)sh);
    unsigned __int128 half = (unsigned __int128)1 << (bits - 1);
    out->proven = out->integral && sl <= half && sh <= half - 1;
    return TR_MATH_OK;
}
int32_t tr_unc_cong_meet(tr_num_rational r1, tr_num_rational m1, tr_num_rational r2, tr_num_rational m2,
                         tr_num_rational *r, tr_num_rational *m) {
    if (!r || !m || !tr_unc_canonical(r1) || !tr_unc_canonical(m1) || !tr_unc_canonical(r2) || !tr_unc_canonical(m2) || m1.numerator < 0 || m2.numerator < 0)
        return TR_MATH_INVALID;
    tr_num_rational rr, mm;
    int k = tr_unc_cong_meet_hd(r1, m1, r2, m2, &rr, &mm);
    if (k == 0) return TR_MATH_NO_SOLUTION;
    if (k < 0 || !tr_unc_cong_norm(&rr, mm)) return TR_MATH_OVERFLOW;
    *r = rr; *m = mm;
    return TR_MATH_OK;
}

/* ------------------------------------------------------------------ polynomial expansions (certificates) */
typedef struct { uint64_t n; tr_num_rational *c; uint8_t *e; } expansion;   /* n terms, e: n x m exponents */
static void exp_free(expansion *x) { free(x->c); free(x->e); x->c = NULL; x->e = NULL; x->n = 0; }
/* the expansion of node `node` as a multiset of terms over m variables (program column j -> var[j]; scale[j]: a
 * column's value is its variable / 10^scale[j]); terms are never merged (the law's canonicalization does it) */
static int32_t expand(const tr_unc_program *p, int32_t node, const int32_t *var, const int32_t *scale, int32_t m,
                      uint64_t limit, tr_math_refusal *refusal, expansion *out) {
    if (node < 0 || node >= p->nnodes || node >= TR_UNC_MAX_NODES) return TR_MATH_INVALID;
    int32_t cone[TR_UNC_MAX_NODES];
    memset(cone, 0, sizeof cone);
    cone[node] = 1;
    for (int32_t k = node; k >= 0; k--) {
        if (!cone[k]) continue;
        int32_t a, b;
        tr_unc_operands(&p->nodes[k], &a, &b);
        if (a >= 0) cone[a] = 1;
        if (b >= 0) cone[b] = 1;
    }
    expansion *x = calloc((size_t)node + 1, sizeof *x);
    if (!x) return TR_MATH_MEMORY;
    int32_t rc = TR_MATH_OK;
    for (int32_t k = 0; k <= node && !rc; k++) {
        if (!cone[k]) continue;
        const tr_unc_node *nd = &p->nodes[k];
        const expansion *A = nd->op != TR_UNC_COL && nd->op != TR_UNC_CONST ? &x[nd->a] : NULL;
        const expansion *B = (nd->op == TR_UNC_ADD || nd->op == TR_UNC_SUB || nd->op == TR_UNC_MUL) ? &x[nd->b] : NULL;
        uint64_t n;
        switch (nd->op) {
        case TR_UNC_COL: case TR_UNC_CONST: n = 1; break;
        case TR_UNC_ADD: case TR_UNC_SUB: n = tr_budget_add(A->n, B->n); break;
        case TR_UNC_NEG: case TR_UNC_DIV: n = A->n; break;
        case TR_UNC_MUL: n = tr_budget_mul(A->n, B->n); break;
        case TR_UNC_SQR: n = tr_budget_mul(A->n, A->n); break;
        default: rc = TR_MATH_UNSUPPORTED; continue;      /* not a polynomial operation */
        }
        if (n > limit) {
            if (refusal) { refusal->resource = TR_BUDGET_SUPPORT; refusal->refused = 1; refusal->need = n; refusal->limit = limit; }
            rc = TR_MATH_BUDGET; continue;
        }
        expansion *r = &x[k];
        r->n = n;
        r->c = calloc((size_t)(n ? n : 1), sizeof *r->c);
        r->e = calloc((size_t)(n ? n : 1) * (size_t)m, 1);
        if (!r->c || !r->e) { rc = TR_MATH_MEMORY; continue; }
        if (nd->op == TR_UNC_COL) {
            if (!tr_q_make(1, tr_unc_pow10(scale ? scale[nd->a] : 0), &r->c[0])) rc = TR_MATH_OVERFLOW;
            r->e[var[nd->a]] = 1;
        } else if (nd->op == TR_UNC_CONST) r->c[0] = nd->k;
        else if (nd->op == TR_UNC_ADD || nd->op == TR_UNC_SUB) {
            memcpy(r->c, A->c, (size_t)A->n * sizeof *r->c); memcpy(r->e, A->e, (size_t)A->n * (size_t)m);
            for (uint64_t t = 0; t < B->n && !rc; t++) {
                r->c[A->n + t] = B->c[t];
                if (nd->op == TR_UNC_SUB && !tr_unc_qneg(B->c[t], &r->c[A->n + t])) rc = TR_MATH_OVERFLOW;
            }
            memcpy(r->e + A->n * (uint64_t)m, B->e, (size_t)B->n * (size_t)m);
        } else if (nd->op == TR_UNC_NEG || nd->op == TR_UNC_DIV) {
            memcpy(r->e, A->e, (size_t)A->n * (size_t)m);
            for (uint64_t t = 0; t < A->n && !rc; t++)
                if (!(nd->op == TR_UNC_NEG ? tr_unc_qneg(A->c[t], &r->c[t]) : tr_q_div(A->c[t], nd->k, &r->c[t]))) rc = TR_MATH_OVERFLOW;
        } else {
            const expansion *R = nd->op == TR_UNC_SQR ? A : B;
            uint64_t t = 0;
            for (uint64_t i = 0; i < A->n && !rc; i++)
                for (uint64_t j = 0; j < R->n && !rc; j++, t++) {
                    if (!tr_q_mul(A->c[i], R->c[j], &r->c[t])) rc = TR_MATH_OVERFLOW;
                    for (int32_t v = 0; v < m; v++) {
                        unsigned s = (unsigned)A->e[i * (uint64_t)m + (uint64_t)v] + R->e[j * (uint64_t)m + (uint64_t)v];
                        if (s > 255) rc = TR_MATH_UNSUPPORTED;
                        r->e[t * (uint64_t)m + (uint64_t)v] = (uint8_t)s;
                    }
                }
        }
    }
    if (!rc) { *out = x[node]; x[node].c = NULL; x[node].e = NULL; }
    for (int32_t k = 0; k <= node; k++) exp_free(&x[k]);
    free(x);
    return rc;
}
/* the columns of node's cone: var[j] = the variable of program column j (-1: unused); m = their count */
static int32_t cone_columns(const tr_unc_program *p, int32_t node, int32_t *var, int32_t *m) {
    int32_t used[TR_UNC_MAX_COLS], cone[TR_UNC_MAX_NODES];
    memset(used, 0, sizeof used); memset(cone, 0, sizeof cone);
    cone[node] = 1;
    for (int32_t k = node; k >= 0; k--) {
        if (!cone[k]) continue;
        int32_t a, b;
        tr_unc_operands(&p->nodes[k], &a, &b);
        if (a >= 0) cone[a] = 1;
        if (b >= 0) cone[b] = 1;
        if (p->nodes[k].op == TR_UNC_COL) used[p->nodes[k].a] = 1;
    }
    *m = 0;
    for (int32_t j = 0; j < TR_UNC_MAX_COLS; j++) var[j] = j < p->ncols && used[j] ? (*m)++ : -1;
    if (*m == 0) { var[0] = 0; *m = 1; }               /* a constant expression: one (unused) variable */
    return *m > TR_CERT_MAX_COLS ? TR_MATH_UNSUPPORTED : TR_MATH_OK;
}
/* integer coefficients D c_t (D = lcm of the denominators); 0 or TR_MATH_OVERFLOW */
static int32_t integral_coefficients(const expansion *x, int64_t *out, int64_t *D) {
    uint64_t d = 1;
    for (uint64_t t = 0; t < x->n; t++) {
        uint64_t q = (uint64_t)x->c[t].denominator, g = tr_gcd_u64(d, q);
        unsigned __int128 l = (unsigned __int128)(d / g) * q;
        if (l > (unsigned __int128)INT64_MAX) return TR_MATH_OVERFLOW;
        d = (uint64_t)l;
    }
    for (uint64_t t = 0; t < x->n; t++) {
        __int128 v = (__int128)x->c[t].numerator * (__int128)(d / (uint64_t)x->c[t].denominator);
        if (v > INT64_MAX || v < INT64_MIN) return TR_MATH_OVERFLOW;
        out[t] = (int64_t)v;
    }
    *D = (int64_t)d;
    return TR_MATH_OK;
}
/* x plus extra terms: the constant -c and (affine) -a_j x_j */
static int32_t append_terms(expansion *x, const tr_num_rational *coef, const int32_t *vars, int32_t nextra, int32_t m) {
    uint64_t n = x->n + (uint64_t)nextra;
    tr_num_rational *c = realloc(x->c, (size_t)(n ? n : 1) * sizeof *c);
    if (!c) return TR_MATH_MEMORY;
    x->c = c;
    uint8_t *e = realloc(x->e, (size_t)(n ? n : 1) * (size_t)m);
    if (!e) return TR_MATH_MEMORY;
    x->e = e;
    for (int32_t i = 0; i < nextra; i++) {
        if (!tr_unc_qneg(coef[i], &x->c[x->n + (uint64_t)i])) return TR_MATH_OVERFLOW;
        memset(x->e + (x->n + (uint64_t)i) * (uint64_t)m, 0, (size_t)m);
        if (vars[i] >= 0) x->e[(x->n + (uint64_t)i) * (uint64_t)m + (uint64_t)vars[i]] = 1;
    }
    x->n = n;
    return TR_MATH_OK;
}
tr_cert *tr_unc_identity(const tr_unc_program *p, int32_t node, const tr_math_budget *budget, tr_math_refusal *refusal,
                         int32_t *error) {
    tr_budget_none(refusal);
    tr_unc_value form;
    int32_t exact = 0;
    int32_t rc = tr_unc_symbolic(p, node, &form, &exact);
    if (rc) { set_error(error, rc); return NULL; }
    if (form.kind != TR_UNC_NUMERIC) { set_error(error, TR_MATH_INVALID); return NULL; }
    if (!exact) { set_error(error, TR_MATH_DOMAIN_ERROR); return NULL; }   /* not an exact affine identity */
    int32_t var[TR_UNC_MAX_COLS], m;
    rc = cone_columns(p, node, var, &m);
    if (rc) { set_error(error, rc); return NULL; }
    tr_math_budget dflt;
    const tr_math_budget *b = budget_or(budget, &dflt);
    uint64_t limit = b->limit[TR_BUDGET_SUPPORT] < TR_CERT_MAX_TERMS ? b->limit[TR_BUDGET_SUPPORT] : TR_CERT_MAX_TERMS;
    expansion x = {0, NULL, NULL};
    rc = expand(p, node, var, NULL, m, limit, refusal, &x);
    /* minus the affine form: the constant and one term per column symbol */
    tr_num_rational coef[TR_UNC_MAX_TERMS + 1];
    int32_t vars[TR_UNC_MAX_TERMS + 1], nextra = 0;
    if (!rc) {
        if (form.center.numerator) { coef[nextra] = form.center; vars[nextra++] = -1; }
        for (int32_t i = 0; i < form.nterms; i++) { coef[nextra] = tr_unc_qv(form.term[i].num, form.term[i].den); vars[nextra++] = var[form.term[i].sym]; }
        rc = append_terms(&x, coef, vars, nextra, m);
    }
    if (!rc && x.n > limit) {
        if (refusal) { refusal->resource = TR_BUDGET_SUPPORT; refusal->refused = 1; refusal->need = x.n; refusal->limit = limit; }
        rc = TR_MATH_BUDGET;
    }
    int64_t *ic = NULL, D = 1;
    if (!rc) { ic = calloc((size_t)(x.n ? x.n : 1), sizeof *ic); if (!ic) rc = TR_MATH_MEMORY; }
    if (!rc) rc = integral_coefficients(&x, ic, &D);
    tr_cert *cert = NULL;
    if (!rc) {
        tr_cert_claim c;
        memset(&c, 0, sizeof c);
        c.kind = TR_CLAIM_POLYNOMIAL; c.policy = TR_CERT_NULL_FORBID; c.ncols = m;
        for (int32_t i = 0; i < TR_CERT_MAX_COLS; i++) c.col[i] = i < m ? i : -1;
        c.carrier = TR_CARRIER_I128; c.field = 0; c.tolerance = 0;
        c.nterms = x.n; c.exponents = x.e; c.coefficients = ic;
        cert = tr_cert_law(TR_CERT_LAW_RING_IDENTITY, &c, b, refusal, &rc);
    }
    free(ic); exp_free(&x);
    set_error(error, cert ? TR_MATH_OK : (rc ? rc : TR_MATH_MEMORY));
    return cert;
}
tr_cert *tr_unc_certify(const tr_unc_state *s, const tr_scope *scope, const tr_unc_program *p, int32_t fact,
                        uint64_t modulus, tr_cert_evidence *evidence, const tr_math_budget *budget,
                        tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    if (!state_valid(s) || !s->scoped || !scope || !p || fact < TR_UNC_FACT_NOT_NULL || fact > TR_UNC_FACT_CONSTANT) {
        set_error(error, TR_MATH_INVALID); return NULL;
    }
    tr_cert_staleness st;
    int32_t rc = tr_unc_state_fresh(s, scope, &st);
    if (rc) { set_error(error, rc); return NULL; }
    tr_unc_plan plan;
    rc = tr_unc_check(p, 0, &plan, NULL);
    if (rc || p->ncols != s->ncols || plan.kind[p->root] != TR_UNC_NUMERIC) { set_error(error, rc ? rc : TR_MATH_INVALID); return NULL; }
    tr_gfp gf;
    if (fact == TR_UNC_FACT_CONGRUENCE && (modulus > (uint64_t)INT64_MAX || tr_gfp_field(modulus, &gf))) { set_error(error, TR_MATH_INVALID); return NULL; }
    tr_math_budget dflt;
    const tr_math_budget *b = budget_or(budget, &dflt);
    tr_math_need need;
    need_zero(&need);
    need.need[TR_BUDGET_OUTPUT] = s->ngroups;
    need.need[TR_BUDGET_BYTES] = tr_budget_mul(s->ngroups, sizeof(tr_unc_result));
    rc = tr_budget_check(b, &need, refusal);
    if (rc) { set_error(error, rc); return NULL; }
    tr_unc_result *res = calloc((size_t)s->ngroups, sizeof *res);
    if (!res) { set_error(error, TR_MATH_MEMORY); return NULL; }
    rc = tr_unc_eval(s, p, 0, s->ngroups, res, b, refusal);
    /* the fact in every group with a value (NOT NULL: in every group with rows) */
    int have = 0, proven = !rc;
    tr_num_rational r0 = tr_q_int(0);
    for (uint64_t g = 0; proven && g < s->ngroups; g++) {
        const tr_unc_result *r = &res[g];
        if (!r->rows) continue;
        if (fact == TR_UNC_FACT_NOT_NULL) { proven = !(r->null & TR_UNC_MAYBE_NULL); have = 1; continue; }
        if (!(r->null & TR_UNC_MAYBE_VALUE)) continue;
        if (fact == TR_UNC_FACT_CONSTANT) {
            if ((r->flags & (TR_UNC_LO_INF | TR_UNC_HI_INF)) || !tr_unc_qeq(r->lo, r->hi) || (have && !tr_unc_qeq(r->lo, r0))) proven = 0;
            r0 = r->lo; have = 1;
            continue;
        }
        /* congruence: m in p Z and r = r0 (mod p) */
        tr_num_rational t, d;
        if ((r->flags & TR_UNC_CONG_TOP) || !tr_q_div(r->cm, tr_q_int((int64_t)modulus), &t) || t.denominator != 1) { proven = 0; continue; }
        if (!have) { r0 = r->cr; have = 1; continue; }
        if (!tr_q_sub(r->cr, r0, &d) || !tr_q_div(d, tr_q_int((int64_t)modulus), &t) || t.denominator != 1) proven = 0;
    }
    free(res);
    if (rc) { set_error(error, rc); return NULL; }
    if (!proven || !have) { set_error(error, TR_MATH_DOMAIN_ERROR); return NULL; }   /* not proved by the analysis */
    tr_cert_claim c;
    memset(&c, 0, sizeof c);
    for (int32_t i = 0; i < TR_CERT_MAX_COLS; i++) c.col[i] = -1;
    int64_t *ic = NULL;
    expansion x = {0, NULL, NULL};
    if (fact == TR_UNC_FACT_NOT_NULL) {
        if (p->nodes[p->root].op != TR_UNC_COL) { set_error(error, TR_MATH_UNSUPPORTED); return NULL; }
        c.kind = TR_CLAIM_NOT_NULL; c.policy = TR_CERT_NULL_FORBID; c.ncols = 1; c.col[0] = s->cols[p->nodes[p->root].a];
        c.carrier = TR_CARRIER_I128;
    } else {
        int32_t var[TR_UNC_MAX_COLS], m, scale[TR_UNC_MAX_COLS];
        rc = cone_columns(p, p->root, var, &m);
        for (int32_t j = 0; j < p->ncols; j++) scale[j] = s->inputs[j].scale;
        uint64_t limit = b->limit[TR_BUDGET_SUPPORT] < TR_CERT_MAX_TERMS ? b->limit[TR_BUDGET_SUPPORT] : TR_CERT_MAX_TERMS;
        if (!rc) rc = expand(p, p->root, var, scale, m, limit, refusal, &x);
        int32_t vnone = -1;
        if (!rc) rc = append_terms(&x, &r0, &vnone, 1, m);   /* E - r0 (= 0, or = 0 mod p) */
        if (!rc) { ic = calloc((size_t)(x.n ? x.n : 1), sizeof *ic); if (!ic) rc = TR_MATH_MEMORY; }
        int64_t D = 1;
        if (!rc && fact == TR_UNC_FACT_CONSTANT) rc = integral_coefficients(&x, ic, &D);
        if (!rc && fact == TR_UNC_FACT_CONGRUENCE) {     /* coefficients mod p (the denominators must be units) */
            for (uint64_t t = 0; t < x.n && !rc; t++) {
                if ((uint64_t)x.c[t].denominator % modulus == 0) { rc = TR_MATH_UNSUPPORTED; break; }
                uint64_t n = tr_gfp_reduce_i64_hd(&gf, x.c[t].numerator), dd = tr_gfp_reduce_i64_hd(&gf, x.c[t].denominator);
                ic[t] = (int64_t)tr_gfp_mul_hd(&gf, n, tr_gfp_inv_hd(&gf, dd));
            }
        }
        if (rc) { free(ic); exp_free(&x); set_error(error, rc); return NULL; }
        c.kind = TR_CLAIM_POLYNOMIAL; c.policy = TR_CERT_NULL_SKIP; c.ncols = m;
        /* the claim's columns: the scope columns of the cone's program columns, in variable order */
        for (int32_t j = 0; j < p->ncols; j++) if (var[j] >= 0) c.col[var[j]] = s->cols[j];
        if (c.col[0] < 0) c.col[0] = s->cols[0];
        for (int32_t i = 0; i < m; i++)                  /* two program columns on one scope column: unsupported */
            for (int32_t i2 = 0; i2 < i; i2++) if (c.col[i] == c.col[i2]) { free(ic); exp_free(&x); set_error(error, TR_MATH_UNSUPPORTED); return NULL; }
        c.nterms = x.n; c.exponents = x.e; c.coefficients = ic;
        if (fact == TR_UNC_FACT_CONGRUENCE) { c.carrier = TR_CARRIER_GFP; c.field = modulus; c.gf.p = modulus; }
        else c.carrier = TR_CARRIER_I128;
    }
    tr_cert *cert = tr_cert_snapshot(scope, &c, NULL, 0, evidence, b, refusal, &rc);
    free(ic); exp_free(&x);
    set_error(error, cert ? TR_MATH_OK : (rc ? rc : TR_MATH_MEMORY));
    return cert;
}
