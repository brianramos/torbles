/* BOOL-001: predicate DAGs, cofactors, XOR derivatives and signed differences; see math_bool.h.
 * SPDX-License-Identifier: Apache-2.0 */
#include "math_bool.h"
#include "mathematics.h"
#include "math_domain.h"
#include "math_handles.h"
#include <stdlib.h>
#include <string.h>

TR_MATH_ABI(tr_bool_condition, TR_BOOL_ABI);      /* the ctypes boundary of BOOL-001 (math_handles.h) */
TR_MATH_ABI(tr_bool_node, TR_BOOL_ABI);
TR_MATH_ABI(tr_bool_program_view, TR_BOOL_ABI);
TR_MATH_ABI(tr_bool_domain, TR_BOOL_ABI);
TR_MATH_ABI(tr_bool_query, TR_BOOL_ABI);
TR_MATH_ABI(tr_bool_result, TR_BOOL_ABI);
TR_MATH_ABI(tr_bool_masks_info, TR_BOOL_ABI);
TR_MATH_ABI(tr_bool_program_info, TR_BOOL_ABI);

_Static_assert(sizeof(tr_bool_condition) == 48, "tr_bool_condition ABI");
_Static_assert(sizeof(tr_bool_node) == 16, "tr_bool_node ABI");
_Static_assert(sizeof(tr_bool_program_view) == 40, "tr_bool_program_view ABI");
_Static_assert(sizeof(tr_bool_domain) == 56, "tr_bool_domain ABI");
_Static_assert(sizeof(tr_bool_query) == 16, "tr_bool_query ABI");
_Static_assert(sizeof(tr_bool_result) == 128, "tr_bool_result ABI");
_Static_assert(sizeof(tr_bool_masks_info) == 64, "tr_bool_masks_info ABI");
_Static_assert(sizeof(tr_bool_program_info) == 32, "tr_bool_program_info ABI");
_Static_assert(sizeof(tr_bool_plan) == 56, "tr_bool_plan layout");

#define MASKS_MAGIC UINT32_C(0x424f4f4d)
#define PROGRAM_MAGIC UINT32_C(0x424f4f50)

struct tr_bool_masks_s {
    uint32_t magic;
    int32_t kind, k, reserved;
    uint64_t points, nw, selected, bytes;
    uint64_t digest[2];
    uint64_t *selection;          /* ROWS */
    uint64_t *value, *valid;      /* ROWS: k * nw words; valid NULL when every condition is known on every selected row */
};

struct tr_bool_program_s {
    uint32_t magic;
    int32_t k, nnodes, nsched, root, reserved;
    uint64_t bytes;
    tr_bool_node *nodes;
    uint64_t *deps;
    uint32_t *schedule;
};

static void set_error(int32_t *error, int32_t code) { if (error) *error = code; }
static void *zalloc(uint64_t bytes) {
    if (bytes > (uint64_t)SIZE_MAX) return NULL;
    return calloc(1, bytes ? (size_t)bytes : 1);
}
static void zero_need(tr_math_need *n) { for (int i = 0; i < TR_BUDGET_RESOURCES; i++) n->need[i] = 0; }
static int masks_ok(const tr_bool_masks *m) { return m && m->magic == MASKS_MAGIC; }
static int program_ok(const tr_bool_program *p) { return p && p->magic == PROGRAM_MAGIC; }
static const tr_math_budget *budget_or_default(const tr_math_budget *b, tr_math_budget *dflt) {
    if (b) return b;
    tr_budget_default(dflt);
    return dflt;
}

/* ------------------------------------------------------------------ masks */
static void masks_release(tr_bool_masks *m) {
    if (!m) return;
    free(m->selection); free(m->value); free(m->valid);
    m->magic = 0;
    free(m);
}
void tr_bool_masks_free(tr_bool_masks *m) { if (masks_ok(m)) masks_release(m); }
static tr_bool_masks *masks_fail(tr_bool_masks *m, int32_t *error, int32_t code) {
    masks_release(m); set_error(error, code); return NULL;
}

/* conditions against the scope's view: 0 or TR_MATH_INVALID */
static int32_t conditions_check(const tr_scope_view *v, const tr_bool_condition *c, int32_t k) {
    if (k < 0 || (k && !c)) return TR_MATH_INVALID;
    for (int32_t i = 0; i < k; i++) if (tr_bool_condition_check(v, &c[i])) return TR_MATH_INVALID;
    return TR_MATH_OK;
}
static uint64_t rows_bytes(uint64_t nw, int32_t k) {
    return tr_budget_add(sizeof(tr_bool_masks), tr_budget_mul(nw, tr_budget_add(8, tr_budget_mul((uint64_t)k, 16))));
}

int32_t tr_bool_masks_need(const tr_scope *scope, const tr_bool_condition *conds, int32_t k, int32_t target,
                           tr_math_need *need) {
    if (!need || k < 0) return TR_MATH_INVALID;
    zero_need(need);
    need->need[TR_BUDGET_VARIABLES] = (uint64_t)k;
    if (!scope) {                                             /* ASSIGNMENTS */
        if (conds) return TR_MATH_INVALID;
        need->need[TR_BUDGET_SUPPORT] = tr_budget_power2((uint32_t)k);
        need->need[TR_BUDGET_BYTES] = sizeof(tr_bool_masks);
        return TR_MATH_OK;
    }
    tr_scope_view v;
    int32_t rc = tr_scope_get_view(scope, &v);
    if (rc) return rc;
    rc = conditions_check(&v, conds, k);
    if (rc) return rc;
    if (target == TR_BOOL_NEED_DEVICE) need->need[TR_BUDGET_BYTES] = tr_bool_masks_device_bytes(&v, conds, k);
    else if (target == TR_BOOL_NEED_CPU || target == TR_BOOL_NEED_ADOPT)
        need->need[TR_BUDGET_BYTES] = rows_bytes((v.rows + 63) / 64, k);
    else return TR_MATH_INVALID;
    return TR_MATH_OK;
}

/* the common start of build and adopt: checks, budget, the object with its selection */
static tr_bool_masks *masks_start(const tr_scope *scope, const tr_bool_condition *conds, int32_t k,
                                  const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error,
                                  tr_scope_view *v) {
    tr_budget_none(refusal);
    tr_math_need need;
    int32_t rc = tr_bool_masks_need(scope, conds, k, TR_BOOL_NEED_CPU, &need);
    if (rc) return masks_fail(NULL, error, rc);
    tr_math_budget dflt;
    rc = tr_budget_check(budget_or_default(budget, &dflt), &need, refusal);
    if (rc) return masks_fail(NULL, error, rc);
    if (k > TR_BOOL_MAX_CONDITIONS) return masks_fail(NULL, error, TR_MATH_UNSUPPORTED);
    tr_scope_get_view(scope, v);
    tr_scope_info info;
    tr_scope_get_info(scope, &info);
    tr_bool_masks *m = zalloc(sizeof *m);
    if (!m) return masks_fail(NULL, error, TR_MATH_MEMORY);
    m->kind = TR_BOOL_ROWS; m->k = k;
    m->points = v->rows; m->nw = (v->rows + 63) / 64; m->selected = v->selected;
    m->digest[0] = info.digest[0]; m->digest[1] = info.digest[1];
    m->selection = zalloc(m->nw * 8);
    m->value = zalloc(tr_budget_mul((uint64_t)k, m->nw) * 8);
    m->valid = zalloc(tr_budget_mul((uint64_t)k, m->nw) * 8);
    if (!m->selection || !m->value || !m->valid) return masks_fail(m, error, TR_MATH_MEMORY);
    if (m->nw) memcpy(m->selection, v->selection, (size_t)(m->nw * 8));
    return m;
}
/* drops the valid words when they equal the selection everywhere; records the bytes kept */
static void masks_finish(tr_bool_masks *m) {
    int all_known = 1;
    for (int32_t i = 0; i < m->k && all_known; i++)
        for (uint64_t w = 0; w < m->nw; w++)
            if (m->valid[(uint64_t)i * m->nw + w] != m->selection[w]) { all_known = 0; break; }
    if (all_known) { free(m->valid); m->valid = NULL; }
    m->bytes = tr_budget_add(sizeof *m, tr_budget_mul(m->nw, 8 + (uint64_t)m->k * (m->valid ? 16 : 8)));
    m->magic = MASKS_MAGIC;
}

tr_bool_masks *tr_bool_masks_build(const tr_scope *scope, const tr_bool_condition *conds, int32_t k,
                                   const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error) {
    if (!scope) { tr_budget_none(refusal); return masks_fail(NULL, error, TR_MATH_INVALID); }
    tr_scope_view v;
    tr_bool_masks *m = masks_start(scope, conds, k, budget, refusal, error, &v);
    if (!m) return NULL;
    for (int32_t i = 0; i < k; i++)
        for (uint64_t w = 0; w < m->nw; w++) {
            uint64_t at = (uint64_t)i * m->nw + w;
            tr_bool_condition_word(&v, &conds[i], w, &m->value[at], &m->valid[at]);
        }
    masks_finish(m);
    set_error(error, TR_MATH_OK);
    return m;
}

tr_bool_masks *tr_bool_masks_adopt(const tr_scope *scope, const tr_bool_condition *conds, int32_t k,
                                   const uint64_t *value, const uint64_t *valid, const tr_math_budget *budget,
                                   tr_math_refusal *refusal, int32_t *error) {
    if (!scope) { tr_budget_none(refusal); return masks_fail(NULL, error, TR_MATH_INVALID); }
    tr_scope_view v;
    tr_bool_masks *m = masks_start(scope, conds, k, budget, refusal, error, &v);
    if (!m) return NULL;
    uint64_t words = (uint64_t)k * m->nw;
    if (words && !value) return masks_fail(m, error, TR_MATH_INVALID);
    for (uint64_t at = 0; at < words; at++) {
        uint64_t sel = m->selection[at % m->nw], ok = valid ? valid[at] : sel;
        if ((value[at] & ~ok) || (ok & ~sel)) return masks_fail(m, error, TR_MATH_INVALID);   /* not canonical */
        m->value[at] = value[at];
        m->valid[at] = ok;
    }
    masks_finish(m);
    set_error(error, TR_MATH_OK);
    return m;
}

tr_bool_masks *tr_bool_masks_assignments(int32_t k, const tr_math_budget *budget, tr_math_refusal *refusal,
                                         int32_t *error) {
    tr_budget_none(refusal);
    tr_math_need need;
    int32_t rc = tr_bool_masks_need(NULL, NULL, k, TR_BOOL_NEED_CPU, &need);
    if (rc) return masks_fail(NULL, error, rc);
    tr_math_budget dflt;
    rc = tr_budget_check(budget_or_default(budget, &dflt), &need, refusal);
    if (rc) return masks_fail(NULL, error, rc);
    if (k > TR_BOOL_MAX_ASSIGN_VARS) return masks_fail(NULL, error, TR_MATH_UNSUPPORTED);
    tr_bool_masks *m = zalloc(sizeof *m);
    if (!m) return masks_fail(NULL, error, TR_MATH_MEMORY);
    m->kind = TR_BOOL_ASSIGNMENTS; m->k = k;
    m->points = (uint64_t)1 << k; m->nw = (m->points + 63) / 64; m->selected = m->points;
    m->bytes = sizeof *m;
    m->magic = MASKS_MAGIC;
    set_error(error, TR_MATH_OK);
    return m;
}

int32_t tr_bool_masks_get_info(const tr_bool_masks *m, tr_bool_masks_info *out) {
    if (!masks_ok(m) || !out) return TR_MATH_INVALID;
    memset(out, 0, sizeof *out);
    out->points = m->points; out->nw = m->nw; out->selected = m->selected; out->bytes = m->bytes;
    out->scope_digest[0] = m->digest[0]; out->scope_digest[1] = m->digest[1];
    out->kind = m->kind; out->k = m->k; out->has_valid = m->valid != NULL;
    return TR_MATH_OK;
}
int32_t tr_bool_masks_get_domain(const tr_bool_masks *m, tr_bool_domain *d) {
    if (!masks_ok(m) || !d) return TR_MATH_INVALID;
    memset(d, 0, sizeof *d);
    d->points = m->points; d->nw = m->nw; d->selected = m->selected; d->kind = m->kind; d->k = m->k;
    d->selection = m->selection; d->value = m->value; d->valid = m->valid;
    return TR_MATH_OK;
}
int32_t tr_bool_masks_counts(const tr_bool_masks *m, int32_t i, uint64_t counts[3]) {
    if (!masks_ok(m) || !counts || i < 0 || i >= m->k) return TR_MATH_INVALID;
    tr_bool_domain d;
    tr_bool_masks_get_domain(m, &d);
    counts[0] = counts[1] = counts[2] = 0;
    for (uint64_t w = 0; w < m->nw; w++) {
        uint64_t sel = tr_bool_selection_word(&d, w), v, k;
        tr_bool_var_word(&d, i, w, &v, &k);
        v &= sel; k &= sel;
        counts[0] += (uint64_t)tr_bool_popc(v);
        counts[1] += (uint64_t)tr_bool_popc(k & ~v);
        counts[2] += (uint64_t)tr_bool_popc(sel & ~k);
    }
    return TR_MATH_OK;
}
int32_t tr_bool_masks_equal(const tr_bool_masks *a, const tr_bool_masks *b) {
    if (!masks_ok(a) || !masks_ok(b)) return TR_MATH_INVALID;
    if (a->kind != b->kind || a->k != b->k || a->points != b->points || a->selected != b->selected) return 0;
    if (a->kind == TR_BOOL_ASSIGNMENTS) return 1;
    uint64_t words = (uint64_t)a->k * a->nw;
    if (a->nw && memcmp(a->selection, b->selection, (size_t)(a->nw * 8))) return 0;
    if (words && memcmp(a->value, b->value, (size_t)(words * 8))) return 0;
    if ((a->valid == NULL) != (b->valid == NULL)) return 0;
    if (a->valid && words && memcmp(a->valid, b->valid, (size_t)(words * 8))) return 0;
    return 1;
}

/* ------------------------------------------------------------------ programs */
static uint64_t program_bytes(int32_t nnodes) {
    return tr_budget_add(sizeof(tr_bool_program), tr_budget_mul((uint64_t)(nnodes > 0 ? nnodes : 0), 16 + 8 + 4 + 1));
}
int32_t tr_bool_program_need(int32_t k, int32_t nnodes, tr_math_need *need) {
    if (!need || k < 0 || nnodes < 1) return TR_MATH_INVALID;
    zero_need(need);
    need->need[TR_BUDGET_VARIABLES] = (uint64_t)k;
    need->need[TR_BUDGET_CIRCUIT] = (uint64_t)nnodes;
    need->need[TR_BUDGET_BYTES] = program_bytes(nnodes);
    return TR_MATH_OK;
}
static void program_release(tr_bool_program *p) {
    if (!p) return;
    free(p->nodes); free(p->deps); free(p->schedule);
    p->magic = 0;
    free(p);
}
void tr_bool_program_free(tr_bool_program *p) { if (program_ok(p)) program_release(p); }
static tr_bool_program *program_fail(tr_bool_program *p, int32_t *error, int32_t code) {
    program_release(p); set_error(error, code); return NULL;
}
/* the structure of the nodes (operands below the node, ops, operand fields): 0 or TR_MATH_INVALID */
static int32_t nodes_check(int32_t k, const tr_bool_node *nodes, int32_t nnodes, int32_t root) {
    if (k < 0 || nnodes < 1 || !nodes || root < 0 || root >= nnodes) return TR_MATH_INVALID;
    for (int32_t i = 0; i < nnodes; i++) {
        const tr_bool_node *n = &nodes[i];
        if (n->reserved) return TR_MATH_INVALID;
        switch (n->op) {
        case TR_BOOL_CONST: if (n->a < 0 || n->a > 2 || n->b != -1) return TR_MATH_INVALID; break;
        case TR_BOOL_VAR: if (n->a < 0 || n->a >= k || n->b != -1) return TR_MATH_INVALID; break;
        case TR_BOOL_NOT: if (n->a < 0 || n->a >= i || n->b != -1) return TR_MATH_INVALID; break;
        case TR_BOOL_AND: case TR_BOOL_OR: case TR_BOOL_XOR:
            if (n->a < 0 || n->a >= i || n->b < 0 || n->b >= i) return TR_MATH_INVALID;
            break;
        default: return TR_MATH_INVALID;
        }
    }
    return TR_MATH_OK;
}

tr_bool_program *tr_bool_program_create(int32_t k, const tr_bool_node *nodes, int32_t nnodes, int32_t root,
                                        const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    int32_t rc = nodes_check(k, nodes, nnodes, root);
    if (rc) return program_fail(NULL, error, rc);
    tr_math_need need;
    tr_bool_program_need(k, nnodes, &need);
    tr_math_budget dflt;
    rc = tr_budget_check(budget_or_default(budget, &dflt), &need, refusal);
    if (rc) return program_fail(NULL, error, rc);
    if (k > TR_BOOL_MAX_CONDITIONS || nnodes > TR_BOOL_MAX_NODES) return program_fail(NULL, error, TR_MATH_UNSUPPORTED);
    tr_bool_program *p = zalloc(sizeof *p);
    if (!p) return program_fail(NULL, error, TR_MATH_MEMORY);
    uint8_t *reach = zalloc((uint64_t)nnodes);
    p->nodes = zalloc((uint64_t)nnodes * sizeof(tr_bool_node));
    p->deps = zalloc((uint64_t)nnodes * 8);
    p->schedule = zalloc((uint64_t)nnodes * 4);
    if (!reach || !p->nodes || !p->deps || !p->schedule) { free(reach); return program_fail(p, error, TR_MATH_MEMORY); }
    memcpy(p->nodes, nodes, (size_t)nnodes * sizeof(tr_bool_node));
    for (int32_t i = 0; i < nnodes; i++) {
        const tr_bool_node *n = &p->nodes[i];
        p->deps[i] = n->op == TR_BOOL_VAR ? (uint64_t)1 << n->a : n->op == TR_BOOL_CONST ? 0
                   : n->op == TR_BOOL_NOT ? p->deps[n->a] : p->deps[n->a] | p->deps[n->b];
    }
    reach[root] = 1;                                         /* the nodes the root reaches, from the top down */
    for (int32_t i = root; i >= 0; i--) {
        if (!reach[i]) continue;
        const tr_bool_node *n = &p->nodes[i];
        if (n->op >= TR_BOOL_NOT) reach[n->a] = 1;
        if (n->op >= TR_BOOL_AND) reach[n->b] = 1;
    }
    int32_t ns = 0;
    for (int32_t i = 0; i <= root; i++) if (reach[i]) p->schedule[ns++] = (uint32_t)i;
    free(reach);
    p->k = k; p->nnodes = nnodes; p->nsched = ns; p->root = root;
    p->bytes = tr_budget_add(sizeof *p, tr_budget_mul((uint64_t)nnodes, 16 + 8 + 4));
    tr_bool_program_view view;
    view.nodes = p->nodes; view.deps = p->deps; view.schedule = p->schedule;
    view.nnodes = nnodes; view.nsched = ns; view.root = root; view.k = k;
    if (tr_bool_program_check(&view)) return program_fail(p, error, TR_MATH_INVALID);   /* never: a self-check */
    p->magic = PROGRAM_MAGIC;
    set_error(error, TR_MATH_OK);
    return p;
}
int32_t tr_bool_program_get_view(const tr_bool_program *p, tr_bool_program_view *v) {
    if (!program_ok(p) || !v) return TR_MATH_INVALID;
    memset(v, 0, sizeof *v);
    v->nodes = p->nodes; v->deps = p->deps; v->schedule = p->schedule;
    v->nnodes = p->nnodes; v->nsched = p->nsched; v->root = p->root; v->k = p->k;
    return TR_MATH_OK;
}
int32_t tr_bool_program_get_info(const tr_bool_program *p, tr_bool_program_info *out) {
    if (!program_ok(p) || !out) return TR_MATH_INVALID;
    memset(out, 0, sizeof *out);
    out->nnodes = p->nnodes; out->nsched = p->nsched; out->root = p->root; out->k = p->k;
    out->support = p->deps[p->root]; out->bytes = p->bytes;
    return TR_MATH_OK;
}

/* ------------------------------------------------------------------ cofactors and differences */
int32_t tr_bool_layout(const tr_bool_program *p, const tr_bool_query *q, int32_t nq, uint64_t *hist_total,
                       uint64_t *cof_total) {
    tr_bool_program_view v;
    if (tr_bool_program_get_view(p, &v)) return TR_MATH_INVALID;
    return tr_bool_plan_queries(&v, q, nq, NULL, NULL, NULL, hist_total, cof_total, NULL, NULL);
}

/* checks of a request on views; the totals of its plan */
static int32_t request_check(const tr_bool_domain *d, const tr_bool_program_view *p, const tr_bool_query *q,
                             int32_t nq, uint64_t *ncones, uint64_t *hist_total, uint64_t *cof_total,
                             uint64_t *tensor, int32_t *top) {
    int32_t rc = tr_bool_domain_check(d);
    if (!rc) rc = tr_bool_program_check(p);
    if (rc) return rc;
    if (d->k != p->k) return TR_MATH_DOMAIN_ERROR;
    return tr_bool_plan_queries(p, q, nq, NULL, NULL, ncones, hist_total, cof_total, tensor, top);
}

int32_t tr_bool_differences_need(const tr_bool_masks *m, const tr_bool_program *prog, const tr_bool_query *q,
                                 int32_t nq, int32_t target, tr_math_need *need) {
    tr_bool_domain d;
    tr_bool_program_view p;
    if (!need || tr_bool_masks_get_domain(m, &d) || tr_bool_program_get_view(prog, &p)) return TR_MATH_INVALID;
    uint64_t ncones = 0, hist = 0, cof = 0, tensor = 0;
    int32_t top = 0;
    int32_t rc = request_check(&d, &p, q, nq, &ncones, &hist, &cof, &tensor, &top);
    if (rc) return rc;
    tr_bool_differences_base_need(&p, &d, nq, ncones, hist, cof, tensor, top, need);
    if (target == TR_BOOL_NEED_DEVICE) need->need[TR_BUDGET_BYTES] = tr_bool_device_bytes(&p, &d, nq, ncones, hist, cof);
    else if (target != TR_BOOL_NEED_CPU) return TR_MATH_INVALID;
    return TR_MATH_OK;
}

int32_t tr_bool_differences_view(const tr_bool_domain *d, const tr_bool_program_view *p, const tr_bool_query *q,
                                 int32_t nq, const tr_math_budget *budget, tr_bool_result *results, uint64_t *hist,
                                 uint64_t *cof, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    uint64_t ncones = 0, hist_total = 0, cof_total = 0, tensor = 0;
    int32_t top = 0;
    int32_t rc = request_check(d, p, q, nq, &ncones, &hist_total, &cof_total, &tensor, &top);
    if (rc) return rc;
    if ((nq && !results) || (hist_total && !hist) || (cof_total && !cof)) return TR_MATH_INVALID;
    tr_math_need need;
    tr_bool_differences_base_need(p, d, nq, ncones, hist_total, cof_total, tensor, top, &need);
    tr_math_budget dflt;
    rc = tr_budget_check(budget_or_default(budget, &dflt), &need, refusal);
    if (rc) return rc;
    tr_bool_plan *plans = zalloc((uint64_t)nq * sizeof *plans);
    uint32_t *cones = zalloc(ncones * 4);
    uint64_t *val = zalloc((uint64_t)p->nnodes * 16);
    if (!plans || !cones || !val) { free(plans); free(cones); free(val); return TR_MATH_MEMORY; }
    tr_bool_plan_queries(p, q, nq, plans, cones, NULL, NULL, NULL, NULL, NULL);
    for (int32_t j = 0; j < nq; j++) {
        tr_bool_result *R = &results[j];
        memset(R, 0, sizeof *R);
        R->vars = plans[j].vars; R->order = plans[j].order; R->nbins = plans[j].nbins; R->lo = plans[j].lo;
        R->points = d->selected; R->hist = plans[j].hist; R->cof = plans[j].cof;
    }
    if (hist_total) memset(hist, 0, (size_t)(hist_total * 8));
    if (cof_total) memset(cof, 0, (size_t)(cof_total * 8));
    const uint64_t root = (uint64_t)p->root;
    int32_t values[64];
    uint32_t counts[64];
    for (uint64_t w = 0; w < d->nw && nq; w++) {
        uint64_t sel = tr_bool_selection_word(d, w);
        if (!sel) continue;
        tr_bool_eval_base(p, d, w, val);
        uint64_t fv = val[2 * root] & sel, fk = val[2 * root + 1] & sel;
        uint64_t ft = (uint64_t)tr_bool_popc(fv), ff = (uint64_t)tr_bool_popc(fk & ~fv), fu = (uint64_t)tr_bool_popc(sel & ~fk);
        for (int32_t j = 0; j < nq; j++) {
            const tr_bool_plan *P = &plans[j];
            tr_bool_result *R = &results[j];
            R->f_true += ft; R->f_false += ff; R->f_unknown += fu;
            tr_bool_acc A;
            tr_bool_acc_begin(&A, P->order, sel);
            const uint64_t nt = (uint64_t)1 << P->order;
            for (uint64_t t = 0; t < nt; t++) {
                if (P->ncone) tr_bool_eval_cone(p, d, cones + P->cone, P->ncone, P->vars, t, 0, w, val);
                uint64_t rv = val[2 * root] & sel, rk = val[2 * root + 1] & sel;
                tr_bool_acc_add(&A, P->order, t, rv, rk);
                if (P->cof != TR_BOOL_NONE) {
                    uint64_t *c = cof + P->cof + 3 * t;
                    c[0] += (uint64_t)tr_bool_popc(rv);
                    c[1] += (uint64_t)tr_bool_popc(rk & ~rv);
                    c[2] += (uint64_t)tr_bool_popc(sel & ~rk);
                }
            }
            if (P->ncone) tr_bool_eval_cone(p, d, cones + P->cone, P->ncone, P->vars, 0, 1, w, val);   /* restore */
            uint64_t known = A.known;
            R->known += (uint64_t)tr_bool_popc(known);
            R->unknown += (uint64_t)tr_bool_popc(sel & ~known);
            R->xor_true += (uint64_t)tr_bool_popc(A.parity & known);
            R->xor_false += (uint64_t)tr_bool_popc(~A.parity & known);
            int n = tr_bool_acc_leaves(&A, P->order, values, counts);
            __int128 sum = tr_i128_get(R->sum);
            for (int l = 0; l < n; l++) {
                int64_t b = (int64_t)values[l] - P->lo;
                if (b < 0 || b >= P->nbins) { free(plans); free(cones); free(val); return TR_MATH_INVALID; }   /* never */
                hist[P->hist + (uint64_t)b] += counts[l];
                if (values[l]) R->nonzero += counts[l];
                sum += (__int128)values[l] * (__int128)counts[l];
            }
            R->sum = tr_i128_set(sum);
        }
    }
    free(plans); free(cones); free(val);
    return TR_MATH_OK;
}

int32_t tr_bool_differences(const tr_bool_masks *m, const tr_bool_program *prog, const tr_bool_query *q, int32_t nq,
                            const tr_math_budget *budget, tr_bool_result *results, uint64_t *hist, uint64_t *cof,
                            tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    tr_bool_domain d;
    tr_bool_program_view p;
    if (tr_bool_masks_get_domain(m, &d) || tr_bool_program_get_view(prog, &p)) return TR_MATH_INVALID;
    return tr_bool_differences_view(&d, &p, q, nq, budget, results, hist, cof, refusal);
}

/* ------------------------------------------------------------------ independent evaluation (the baseline) */
static uint8_t tv_not(uint8_t a) { return a == TR_BOOL_UNKNOWN ? TR_BOOL_UNKNOWN : (uint8_t)!a; }
static uint8_t tv_and(uint8_t a, uint8_t b) {
    if (a == TR_BOOL_FALSE || b == TR_BOOL_FALSE) return TR_BOOL_FALSE;
    return a == TR_BOOL_UNKNOWN || b == TR_BOOL_UNKNOWN ? TR_BOOL_UNKNOWN : TR_BOOL_TRUE;
}
static uint8_t tv_or(uint8_t a, uint8_t b) {
    if (a == TR_BOOL_TRUE || b == TR_BOOL_TRUE) return TR_BOOL_TRUE;
    return a == TR_BOOL_UNKNOWN || b == TR_BOOL_UNKNOWN ? TR_BOOL_UNKNOWN : TR_BOOL_FALSE;
}
static uint8_t tv_xor(uint8_t a, uint8_t b) {
    return a == TR_BOOL_UNKNOWN || b == TR_BOOL_UNKNOWN ? TR_BOOL_UNKNOWN : (uint8_t)(a ^ b);
}
/* the program at one point whose condition values are cond[0..k) (scalar three-valued logic) */
static uint8_t eval_point(const tr_bool_program *p, const uint8_t *cond, uint8_t *tv) {
    for (int32_t s = 0; s < p->nsched; s++) {
        uint32_t i = p->schedule[s];
        const tr_bool_node *n = &p->nodes[i];
        uint8_t r;
        switch (n->op) {
        case TR_BOOL_CONST: r = (uint8_t)n->a; break;
        case TR_BOOL_VAR: r = cond[n->a]; break;
        case TR_BOOL_NOT: r = tv_not(tv[n->a]); break;
        case TR_BOOL_AND: r = tv_and(tv[n->a], tv[n->b]); break;
        case TR_BOOL_OR: r = tv_or(tv[n->a], tv[n->b]); break;
        default: r = tv_xor(tv[n->a], tv[n->b]); break;
        }
        tv[i] = r;
    }
    return tv[p->root];
}

int32_t tr_bool_direct(const tr_scope *scope, const tr_bool_condition *conds, const tr_bool_program *prog,
                       uint64_t vars, uint64_t values, const tr_math_budget *budget, uint64_t counts[3],
                       tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!program_ok(prog) || !counts || (values & ~vars)) return TR_MATH_INVALID;
    const int32_t k = prog->k;
    if (k < 64 && (vars >> k)) return TR_MATH_INVALID;
    tr_scope_view v;
    if (scope) {
        int32_t rc = tr_scope_get_view(scope, &v);
        if (rc) return rc;
        rc = conditions_check(&v, conds, k);
        if (rc) return rc;
    } else if (conds) return TR_MATH_INVALID;
    tr_math_need need;
    zero_need(&need);
    need.need[TR_BUDGET_VARIABLES] = (uint64_t)k;
    need.need[TR_BUDGET_CIRCUIT] = (uint64_t)prog->nnodes;
    need.need[TR_BUDGET_SUPPORT] = scope ? 0 : tr_budget_power2((uint32_t)k);
    need.need[TR_BUDGET_BYTES] = tr_budget_add(tr_budget_mul((uint64_t)prog->nnodes, 2), (uint64_t)k);
    tr_math_budget dflt;
    int32_t rc = tr_budget_check(budget_or_default(budget, &dflt), &need, refusal);
    if (rc) return rc;
    if (!scope && k > TR_BOOL_MAX_ASSIGN_VARS) return TR_MATH_UNSUPPORTED;
    uint8_t *tv = zalloc((uint64_t)prog->nnodes * 2), *cond = zalloc((uint64_t)k);
    if (!tv || !cond) { free(tv); free(cond); return TR_MATH_MEMORY; }
    counts[0] = counts[1] = counts[2] = 0;
    const uint64_t reads = prog->deps[prog->root] & ~vars;    /* the conditions a cofactor reads at a point */
    for (int32_t i = 0; i < k; i++) if ((vars >> i) & 1) cond[i] = (uint8_t)((values >> i) & 1);
    if (scope) {
        for (uint64_t row = 0; row < v.rows; row++) {
            if (!tr_scope_is_selected(&v, row)) continue;
            for (int32_t i = 0; i < k; i++)
                if ((reads >> i) & 1) cond[i] = (uint8_t)tr_bool_condition_row(&v, &conds[i], row);
            uint8_t r = eval_point(prog, cond, tv);
            counts[r == TR_BOOL_TRUE ? 0 : r == TR_BOOL_FALSE ? 1 : 2]++;
        }
    } else {
        const uint64_t points = (uint64_t)1 << k;
        for (uint64_t a = 0; a < points; a++) {
            for (int32_t i = 0; i < k; i++) if ((reads >> i) & 1) cond[i] = (uint8_t)((a >> i) & 1);
            uint8_t r = eval_point(prog, cond, tv);
            counts[r == TR_BOOL_TRUE ? 0 : r == TR_BOOL_FALSE ? 1 : 2]++;
        }
    }
    free(tv); free(cond);
    return TR_MATH_OK;
}
