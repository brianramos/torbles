/* COUNT-001: observed-domain constraint counts and witnesses; see math_count.h.
 * SPDX-License-Identifier: Apache-2.0 */
#include "math_count.h"
#include "mathematics.h"
#include "math_domain.h"
#include "math_handles.h"
#include <stdlib.h>
#include <string.h>

TR_MATH_ABI(tr_count_constraint, TR_COUNT_ABI);   /* the ctypes boundary of COUNT-001 (math_handles.h) */
TR_MATH_ABI(tr_count_request, TR_COUNT_ABI);
TR_MATH_ABI(tr_count_result, TR_COUNT_ABI);
TR_MATH_ABI(tr_count_set_view, TR_COUNT_ABI);
TR_MATH_ABI(tr_count_set_info, TR_COUNT_ABI);
TR_MATH_ABI(tr_count_domain_info, TR_COUNT_ABI);
TR_MATH_ABI(tr_count_certificate, TR_COUNT_ABI);
TR_MATH_ABI(tr_count_answer_view, TR_COUNT_ABI);
TR_MATH_ABI(tr_count_decision, TR_COUNT_ABI);

_Static_assert(sizeof(tr_count_constraint) == 16, "tr_count_constraint ABI");
_Static_assert(sizeof(tr_count_request) == 16, "tr_count_request ABI");
_Static_assert(sizeof(tr_count_result) == 88, "tr_count_result ABI");
_Static_assert(sizeof(tr_count_set_view) == 64, "tr_count_set_view ABI");
_Static_assert(sizeof(tr_count_set_info) == 40, "tr_count_set_info ABI");
_Static_assert(sizeof(tr_count_domain_info) == 48, "tr_count_domain_info ABI");
_Static_assert(sizeof(tr_count_certificate) == 72, "tr_count_certificate ABI");
_Static_assert(sizeof(tr_count_answer_view) == 48, "tr_count_answer_view ABI");
_Static_assert(sizeof(tr_count_decision) == 56, "tr_count_decision ABI");

#define SET_MAGIC UINT32_C(0x43534554)
#define DOMAIN_MAGIC UINT32_C(0x43444f4d)
#define ANSWER_MAGIC UINT32_C(0x43414e53)

struct tr_count_set_s {
    uint32_t magic;
    int32_t k, nnodes, nsched, nc, reserved;
    uint64_t support, ncones, bytes;
    tr_bool_node *nodes;
    uint64_t *deps;
    uint32_t *schedule;
    tr_count_constraint *constraints;
    uint32_t *cones;
    uint64_t *cone_at;
};

struct tr_count_domain_s {
    uint32_t magic;
    int32_t domain, k, ncols;
    uint64_t bytes;
    tr_bool_masks *masks;
    tr_cert_record *record;           /* OBSERVED */
    tr_bool_condition *conds;         /* OBSERVED: k copies */
};

struct tr_count_answer_s {
    uint32_t magic;
    int32_t nc;
    tr_count_request request;
    tr_count_certificate cert;
    tr_count_result *results;
    uint64_t *points, *witnesses;
    uint64_t points_total, witnesses_total, bytes;
    tr_cert_record *record;           /* OBSERVED: a copy of the domain's */
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
static int cmp_u32(const void *a, const void *b) {
    uint32_t x = *(const uint32_t *)a, y = *(const uint32_t *)b;
    return x < y ? -1 : x > y;
}
static int cmp_u64(const void *a, const void *b) {
    uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
    return x < y ? -1 : x > y;
}

/* ------------------------------------------------------------------ constraint sets */
static int set_ok(const tr_count_set *s) { return s && s->magic == SET_MAGIC; }
static void set_release(tr_count_set *s) {
    if (!s) return;
    free(s->nodes); free(s->deps); free(s->schedule); free(s->constraints); free(s->cones); free(s->cone_at);
    s->magic = 0;
    free(s);
}
void tr_count_set_free(tr_count_set *s) { if (set_ok(s)) set_release(s); }
static tr_count_set *set_fail(tr_count_set *s, int32_t *error, int32_t code) {
    set_release(s); set_error(error, code); return NULL;
}
/* the structure of the nodes, as BOOL-001 checks it: 0 or TR_MATH_INVALID */
static int32_t nodes_check(int32_t k, const tr_bool_node *nodes, int32_t nnodes) {
    if (k < 0 || nnodes < 1 || !nodes) return TR_MATH_INVALID;
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
static int32_t constraints_check(const tr_count_constraint *c, int32_t nc, int32_t nnodes) {
    if (nc < 1 || !c) return TR_MATH_INVALID;
    for (int32_t j = 0; j < nc; j++) {
        if (c[j].reserved || c[j].kind < TR_COUNT_HOLDS || c[j].kind > TR_COUNT_EXCLUDES || c[j].a < 0 ||
            c[j].a >= nnodes) return TR_MATH_INVALID;
        if (c[j].kind == TR_COUNT_HOLDS ? c[j].b != -1 : (c[j].b < 0 || c[j].b >= nnodes)) return TR_MATH_INVALID;
    }
    return TR_MATH_OK;
}
/* the object, the nodes, deps and schedule, the constraints and their offsets, and the walk's scratch (stamps and
 * stack); the cones themselves (4 bytes per entry) are budgeted once they are counted */
static uint64_t set_base_bytes(int32_t nnodes, int32_t nc) {
    uint64_t b = sizeof(tr_count_set);
    b = tr_budget_add(b, tr_budget_mul((uint64_t)nnodes, 16 + 8 + 4 + 4 + 4 + 1));
    return tr_budget_add(b, tr_budget_mul((uint64_t)nc, 16 + 8) + 8);
}
int32_t tr_count_set_need(int32_t k, int32_t nnodes, int32_t nc, tr_math_need *need) {
    if (!need || k < 0 || nnodes < 1 || nc < 1) return TR_MATH_INVALID;
    zero_need(need);
    need->need[TR_BUDGET_VARIABLES] = (uint64_t)k;
    need->need[TR_BUDGET_CIRCUIT] = (uint64_t)nnodes;
    need->need[TR_BUDGET_BYTES] = set_base_bytes(nnodes, nc);
    return TR_MATH_OK;
}
/* the nodes the operands of c reach (each once: stamp), written to out when given; returns their number */
static uint64_t cone_walk(const tr_bool_node *nodes, const tr_count_constraint *c, int32_t *stamp, int32_t mark,
                          uint32_t *stack, uint32_t *out, uint8_t *reach) {
    uint64_t n = 0;
    uint32_t sp = 0;
    int32_t roots[2] = {c->a, c->b};
    for (int r = 0; r < 2; r++)
        if (roots[r] >= 0 && stamp[roots[r]] != mark) { stamp[roots[r]] = mark; stack[sp++] = (uint32_t)roots[r]; }
    while (sp) {
        uint32_t i = stack[--sp];
        if (out) out[n] = i;
        n++;
        if (reach) reach[i] = 1;
        const tr_bool_node *x = &nodes[i];
        int32_t ops[2] = {x->op >= TR_BOOL_NOT ? x->a : -1, x->op >= TR_BOOL_AND ? x->b : -1};
        for (int t = 0; t < 2; t++)
            if (ops[t] >= 0 && stamp[ops[t]] != mark) { stamp[ops[t]] = mark; stack[sp++] = (uint32_t)ops[t]; }
    }
    return n;
}

tr_count_set *tr_count_set_create(int32_t k, const tr_bool_node *nodes, int32_t nnodes, const tr_count_constraint *c,
                                  int32_t nc, const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    int32_t rc = nodes_check(k, nodes, nnodes);
    if (!rc) rc = constraints_check(c, nc, nnodes);
    if (rc) return set_fail(NULL, error, rc);
    tr_math_need need;
    tr_count_set_need(k, nnodes, nc, &need);
    tr_math_budget dflt;
    const tr_math_budget *b = budget_or(budget, &dflt);
    rc = tr_budget_check(b, &need, refusal);
    if (rc) return set_fail(NULL, error, rc);
    if (k > TR_BOOL_MAX_CONDITIONS || nnodes > TR_BOOL_MAX_NODES || nc > TR_COUNT_MAX_CONSTRAINTS)
        return set_fail(NULL, error, TR_MATH_UNSUPPORTED);
    tr_count_set *s = zalloc(sizeof *s);
    if (!s) return set_fail(NULL, error, TR_MATH_MEMORY);
    s->nodes = zalloc((uint64_t)nnodes * sizeof(tr_bool_node));
    s->deps = zalloc((uint64_t)nnodes * 8);
    s->schedule = zalloc((uint64_t)nnodes * 4);
    s->constraints = zalloc((uint64_t)nc * sizeof(tr_count_constraint));
    s->cone_at = zalloc(((uint64_t)nc + 1) * 8);
    int32_t *stamp = zalloc((uint64_t)nnodes * 4);
    uint32_t *stack = zalloc((uint64_t)nnodes * 4);
    uint8_t *reach = zalloc((uint64_t)nnodes);
    if (!s->nodes || !s->deps || !s->schedule || !s->constraints || !s->cone_at || !stamp || !stack || !reach) {
        free(stamp); free(stack); free(reach);
        return set_fail(s, error, TR_MATH_MEMORY);
    }
    memcpy(s->nodes, nodes, (size_t)nnodes * sizeof(tr_bool_node));
    memcpy(s->constraints, c, (size_t)nc * sizeof(tr_count_constraint));
    for (int32_t i = 0; i < nnodes; i++) {
        const tr_bool_node *n = &s->nodes[i];
        s->deps[i] = n->op == TR_BOOL_VAR ? (uint64_t)1 << n->a : n->op == TR_BOOL_CONST ? 0
                   : n->op == TR_BOOL_NOT ? s->deps[n->a] : s->deps[n->a] | s->deps[n->b];
    }
    uint64_t total = 0, support = 0;
    for (int32_t j = 0; j < nc; j++) {                       /* count the cones (and mark their union) */
        s->cone_at[j] = total;
        total = tr_budget_add(total, cone_walk(s->nodes, &s->constraints[j], stamp, j + 1, stack, NULL, reach));
        support |= s->deps[c[j].a] | (c[j].b >= 0 ? s->deps[c[j].b] : 0);
    }
    s->cone_at[nc] = total;
    need.need[TR_BUDGET_BYTES] = tr_budget_add(need.need[TR_BUDGET_BYTES], tr_budget_mul(total, 4));
    rc = tr_budget_check(b, &need, refusal);                 /* the cones, now that their size is known */
    if (rc) { free(stamp); free(stack); free(reach); return set_fail(s, error, rc); }
    s->cones = zalloc(total * 4);
    if (!s->cones) { free(stamp); free(stack); free(reach); return set_fail(s, error, TR_MATH_MEMORY); }
    for (int32_t j = 0; j < nc; j++) {
        uint32_t *cone = s->cones + s->cone_at[j];
        uint64_t n = cone_walk(s->nodes, &s->constraints[j], stamp, -(j + 1), stack, cone, NULL);
        qsort(cone, (size_t)n, 4, cmp_u32);
    }
    int32_t ns = 0;
    for (int32_t i = 0; i < nnodes; i++) if (reach[i]) s->schedule[ns++] = (uint32_t)i;
    free(stamp); free(stack); free(reach);
    s->k = k; s->nnodes = nnodes; s->nsched = ns; s->nc = nc; s->support = support; s->ncones = total;
    s->bytes = tr_budget_add(sizeof *s, tr_budget_add(tr_budget_mul((uint64_t)nnodes, 16 + 8 + 4),
                                                      tr_budget_add(tr_budget_mul((uint64_t)nc, 16 + 8) + 8,
                                                                    tr_budget_mul(total, 4))));
    s->magic = SET_MAGIC;
    tr_count_set_view v;
    tr_count_set_get_view(s, &v);
    if (tr_count_set_check(&v)) return set_fail(s, error, TR_MATH_INVALID);   /* never: a self-check */
    set_error(error, TR_MATH_OK);
    return s;
}
int32_t tr_count_set_get_view(const tr_count_set *s, tr_count_set_view *v) {
    if (!set_ok(s) || !v) return TR_MATH_INVALID;
    memset(v, 0, sizeof *v);
    v->nodes = s->nodes; v->deps = s->deps; v->schedule = s->schedule; v->constraints = s->constraints;
    v->cones = s->cones; v->cone_at = s->cone_at;
    v->nnodes = s->nnodes; v->nsched = s->nsched; v->k = s->k; v->nc = s->nc;
    return TR_MATH_OK;
}
int32_t tr_count_set_get_info(const tr_count_set *s, tr_count_set_info *out) {
    if (!set_ok(s) || !out) return TR_MATH_INVALID;
    memset(out, 0, sizeof *out);
    out->k = s->k; out->nnodes = s->nnodes; out->nsched = s->nsched; out->nc = s->nc;
    out->support = s->support; out->cones = s->ncones; out->bytes = s->bytes;
    return TR_MATH_OK;
}

/* ------------------------------------------------------------------ domains */
static int domain_ok(const tr_count_domain *d) { return d && d->magic == DOMAIN_MAGIC; }
static void domain_release(tr_count_domain *d) {
    if (!d) return;
    tr_bool_masks_free(d->masks);
    tr_cert_record_free(d->record);
    free(d->conds);
    d->magic = 0;
    free(d);
}
void tr_count_domain_free(tr_count_domain *d) { if (domain_ok(d)) domain_release(d); }
static tr_count_domain *domain_fail(tr_count_domain *d, int32_t *error, int32_t code) {
    domain_release(d); set_error(error, code); return NULL;
}
/* the distinct scope columns the conditions read, increasing; -1 for a malformed condition */
static int32_t condition_columns(const tr_scope_view *v, const tr_bool_condition *c, int32_t k, int32_t *cols) {
    uint64_t used = 0;
    if (k < 0 || (k && !c)) return -1;
    for (int32_t i = 0; i < k; i++) {
        if (tr_bool_condition_check(v, &c[i])) return -1;
        used |= (uint64_t)1 << c[i].col;
        if (c[i].op >= TR_BOOL_COL_EQ) used |= (uint64_t)1 << c[i].col2;
    }
    int32_t n = 0;
    for (int32_t col = 0; col < TR_SCOPE_MAX_COLUMNS; col++) if ((used >> col) & 1) cols[n++] = col;
    return n;
}
/* a bound of the record's bytes (its fixed part and the exact selection bits) */
static uint64_t record_bound(uint64_t rows) {
    return tr_budget_add(sizeof(tr_cert_source) + 64, tr_budget_bitmap_bytes(rows));
}
int32_t tr_count_domain_need(const tr_scope *scope, const tr_bool_condition *conds, int32_t k, int32_t target,
                             tr_math_need *need) {
    if (!need || k < 0) return TR_MATH_INVALID;
    if (!scope) {
        if (conds || target != TR_COUNT_NEED_CPU) return TR_MATH_INVALID;
        int32_t rc = tr_bool_masks_need(NULL, NULL, k, TR_BOOL_NEED_CPU, need);
        if (rc) return rc;
        need->need[TR_BUDGET_BYTES] = tr_budget_add(need->need[TR_BUDGET_BYTES], sizeof(tr_count_domain));
        return TR_MATH_OK;
    }
    tr_scope_view v;
    int32_t rc = tr_scope_get_view(scope, &v), cols[TR_SCOPE_MAX_COLUMNS];
    if (rc) return rc;
    if (condition_columns(&v, conds, k, cols) < 0) return TR_MATH_INVALID;
    int32_t bt = target == TR_COUNT_NEED_CPU ? TR_BOOL_NEED_CPU : target == TR_COUNT_NEED_DEVICE ? TR_BOOL_NEED_DEVICE
               : target == TR_COUNT_NEED_ADOPT ? TR_BOOL_NEED_ADOPT : -1;
    if (bt < 0) return TR_MATH_INVALID;
    rc = tr_bool_masks_need(scope, conds, k, bt, need);
    if (rc) return rc;
    if (target != TR_COUNT_NEED_DEVICE) {
        uint64_t own = tr_budget_add(sizeof(tr_count_domain), tr_budget_mul((uint64_t)k, sizeof(tr_bool_condition)));
        need->need[TR_BUDGET_BYTES] = tr_budget_add(need->need[TR_BUDGET_BYTES], tr_budget_add(own, record_bound(v.rows)));
    }
    return TR_MATH_OK;
}
static tr_count_domain *observe(const tr_scope *scope, const tr_bool_condition *conds, int32_t k, const uint64_t *value,
                                const uint64_t *valid, int adopt, const tr_math_budget *budget,
                                tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    tr_scope_view v;
    int32_t cols[TR_SCOPE_MAX_COLUMNS];
    if (!scope || tr_scope_get_view(scope, &v)) return domain_fail(NULL, error, TR_MATH_INVALID);
    int32_t ncols = condition_columns(&v, conds, k, cols);
    if (ncols < 0) return domain_fail(NULL, error, TR_MATH_INVALID);
    tr_math_need need;
    int32_t rc = tr_count_domain_need(scope, conds, k, adopt ? TR_COUNT_NEED_ADOPT : TR_COUNT_NEED_CPU, &need);
    if (rc) return domain_fail(NULL, error, rc);
    tr_math_budget dflt;
    const tr_math_budget *b = budget_or(budget, &dflt);
    rc = tr_budget_check(b, &need, refusal);
    if (rc) return domain_fail(NULL, error, rc);
    if (k > TR_BOOL_MAX_CONDITIONS) return domain_fail(NULL, error, TR_MATH_UNSUPPORTED);
    tr_count_domain *d = zalloc(sizeof *d);
    if (!d) return domain_fail(NULL, error, TR_MATH_MEMORY);
    d->domain = TR_COUNT_OBSERVED; d->k = k; d->ncols = ncols;
    d->conds = zalloc((uint64_t)k * sizeof(tr_bool_condition));
    if (!d->conds) return domain_fail(d, error, TR_MATH_MEMORY);
    if (k) memcpy(d->conds, conds, (size_t)k * sizeof(tr_bool_condition));
    d->masks = adopt ? tr_bool_masks_adopt(scope, conds, k, value, valid, b, refusal, &rc)
                     : tr_bool_masks_build(scope, conds, k, b, refusal, &rc);
    if (!d->masks) return domain_fail(d, error, rc ? rc : TR_MATH_MEMORY);
    d->record = tr_cert_record_scope(scope, cols, ncols, TR_CERT_RECORD_SELECTION, NULL, 0, b, refusal, &rc);
    if (!d->record) return domain_fail(d, error, rc ? rc : TR_MATH_MEMORY);
    tr_bool_masks_info mi;
    tr_bool_masks_get_info(d->masks, &mi);
    d->bytes = tr_budget_add(tr_budget_add(sizeof *d, tr_budget_mul((uint64_t)k, sizeof(tr_bool_condition))),
                             tr_budget_add(mi.bytes, tr_cert_record_bytes(d->record)));
    d->magic = DOMAIN_MAGIC;
    set_error(error, TR_MATH_OK);
    return d;
}
tr_count_domain *tr_count_observe(const tr_scope *scope, const tr_bool_condition *conds, int32_t k,
                                  const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error) {
    return observe(scope, conds, k, NULL, NULL, 0, budget, refusal, error);
}
tr_count_domain *tr_count_observe_adopt(const tr_scope *scope, const tr_bool_condition *conds, int32_t k,
                                        const uint64_t *value, const uint64_t *valid, const tr_math_budget *budget,
                                        tr_math_refusal *refusal, int32_t *error) {
    return observe(scope, conds, k, value, valid, 1, budget, refusal, error);
}
tr_count_domain *tr_count_universe(int32_t k, const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    tr_math_need need;
    int32_t rc = tr_count_domain_need(NULL, NULL, k, TR_COUNT_NEED_CPU, &need);
    if (rc) return domain_fail(NULL, error, rc);
    tr_math_budget dflt;
    const tr_math_budget *b = budget_or(budget, &dflt);
    rc = tr_budget_check(b, &need, refusal);
    if (rc) return domain_fail(NULL, error, rc);
    if (k > TR_BOOL_MAX_ASSIGN_VARS) return domain_fail(NULL, error, TR_MATH_UNSUPPORTED);
    tr_count_domain *d = zalloc(sizeof *d);
    if (!d) return domain_fail(NULL, error, TR_MATH_MEMORY);
    d->domain = TR_COUNT_UNIVERSE; d->k = k;
    d->masks = tr_bool_masks_assignments(k, b, refusal, &rc);
    if (!d->masks) return domain_fail(d, error, rc ? rc : TR_MATH_MEMORY);
    tr_bool_masks_info mi;
    tr_bool_masks_get_info(d->masks, &mi);
    d->bytes = tr_budget_add(sizeof *d, mi.bytes);
    d->magic = DOMAIN_MAGIC;
    set_error(error, TR_MATH_OK);
    return d;
}
int32_t tr_count_domain_get_info(const tr_count_domain *d, tr_count_domain_info *out) {
    if (!domain_ok(d) || !out) return TR_MATH_INVALID;
    tr_bool_masks_info mi;
    tr_bool_masks_get_info(d->masks, &mi);
    memset(out, 0, sizeof *out);
    out->points = mi.points; out->nw = mi.nw; out->selected = mi.selected; out->bytes = d->bytes;
    out->domain = d->domain; out->k = d->k; out->has_unknown = mi.has_valid; out->ncolumns = d->ncols;
    return TR_MATH_OK;
}
int32_t tr_count_domain_get_domain(const tr_count_domain *d, tr_bool_domain *out) {
    if (!domain_ok(d) || !out) return TR_MATH_INVALID;
    return tr_bool_masks_get_domain(d->masks, out);
}
const tr_cert_record *tr_count_domain_get_record(const tr_count_domain *d) { return domain_ok(d) ? d->record : NULL; }
static int32_t record_fresh(const tr_cert_record *r, const tr_scope *current, tr_cert_staleness *out) {
    if (!out) return TR_MATH_INVALID;
    if (!r) {                                                /* UNIVERSE: no data, always fresh */
        memset(out, 0, sizeof *out);
        out->column = out->dim = -1;
        return TR_MATH_OK;
    }
    if (!current) return TR_MATH_INVALID;
    return tr_cert_record_fresh(r, current, NULL, 0, TR_CERT_MATCH_EXACT, out);
}
int32_t tr_count_domain_fresh(const tr_count_domain *d, const tr_scope *current, tr_cert_staleness *out) {
    if (!domain_ok(d)) return TR_MATH_INVALID;
    return record_fresh(d->record, current, out);
}

/* ------------------------------------------------------------------ counting */
static void results_init(const tr_bool_domain *d, const tr_count_set_view *s, tr_count_result *out) {
    for (int32_t j = 0; j < s->nc; j++) {
        tr_count_result *R = &out[j];
        memset(R, 0, sizeof *R);
        R->kind = s->constraints[j].kind; R->a = s->constraints[j].a; R->b = s->constraints[j].b;
        R->points = d->selected; R->witness = TR_COUNT_NONE;
    }
}
/* the counts and first witnesses over every word (val: 2 words per node) */
static void counts_core(const tr_bool_domain *d, const tr_count_set_view *s, uint64_t *val, tr_count_result *out) {
    results_init(d, s, out);
    tr_bool_program_view p = tr_count_schedule_view(s);
    for (uint64_t w = 0; w < d->nw; w++) {
        uint64_t sel = tr_bool_selection_word(d, w);
        if (!sel) continue;
        tr_bool_eval_base(&p, d, w, val);
        for (int32_t j = 0; j < s->nc; j++) {
            uint64_t sat, vio, unk;
            tr_count_classes(&s->constraints[j], val, sel, &sat, &vio, &unk);
            tr_count_result *R = &out[j];
            R->satisfied += (uint64_t)popc(sat); R->violated += (uint64_t)popc(vio); R->unknown += (uint64_t)popc(unk);
            if (vio && R->witness == TR_COUNT_NONE) R->witness = w * 64 + (uint64_t)__builtin_ctzll(vio);
        }
    }
}
/* the requested points (layout done); fill: 2 counters per constraint (scratch) */
static void extract_core(const tr_bool_domain *d, const tr_count_set_view *s, uint64_t *val, const tr_count_result *r,
                         int32_t flags, uint64_t *points, uint64_t *witnesses, uint64_t *fill) {
    uint64_t want_w = 0;
    for (int32_t j = 0; j < s->nc; j++) { fill[2 * j] = fill[2 * j + 1] = 0; want_w += r[j].wit_n; }
    if (!(flags & TR_COUNT_ROWS) && !want_w) return;
    tr_bool_program_view p = tr_count_schedule_view(s);
    uint64_t got_w = 0;
    for (uint64_t w = 0; w < d->nw; w++) {
        if (!(flags & TR_COUNT_ROWS) && got_w == want_w) break;
        uint64_t sel = tr_bool_selection_word(d, w);
        if (!sel) continue;
        tr_bool_eval_base(&p, d, w, val);
        for (int32_t j = 0; j < s->nc; j++) {
            uint64_t sat, vio, unk;
            tr_count_classes(&s->constraints[j], val, sel, &sat, &vio, &unk);
            if (flags & TR_COUNT_ROWS)
                for (uint64_t bits = sat; bits; bits &= bits - 1)
                    points[r[j].rows_at + fill[2 * j]++] = w * 64 + (uint64_t)__builtin_ctzll(bits);
            for (uint64_t bits = vio; bits && fill[2 * j + 1] < r[j].wit_n; bits &= bits - 1) {
                witnesses[r[j].wit_at + fill[2 * j + 1]++] = w * 64 + (uint64_t)__builtin_ctzll(bits);
                got_w++;
            }
        }
    }
}
/* the request's checks on views; 0 or an error code */
static int32_t request_check(const tr_bool_domain *d, const tr_count_set_view *s, const tr_count_request *q) {
    int32_t rc = tr_bool_domain_check(d);
    if (!rc) rc = tr_count_set_check(s);
    if (rc) return rc;
    if (d->k != s->k) return TR_MATH_DOMAIN_ERROR;
    if (q && ((q->flags & ~TR_COUNT_ROWS) || q->reserved)) return TR_MATH_INVALID;
    return TR_MATH_OK;
}
/* the support of a set view: the conditions its constraints read */
static uint64_t view_support(const tr_count_set_view *s) {
    uint64_t sup = 0;
    for (int32_t j = 0; j < s->nc; j++) {
        const tr_count_constraint *c = &s->constraints[j];
        sup |= s->deps[c->a] | (c->b >= 0 ? s->deps[c->b] : 0);
    }
    return sup;
}
/* the coverage scratch: a 2^s bitmap (states 2^s) or 8 bytes per selected row to sort */
static void coverage_need(const tr_bool_domain *d, uint64_t support, tr_math_need *need) {
    if (d->kind != TR_BOOL_ROWS) return;
    int s = popc(support);
    if (s <= TR_COUNT_PATTERN_BITS) {
        uint64_t cells = (uint64_t)1 << s;
        need->need[TR_BUDGET_STATES] = cells;
        need->need[TR_BUDGET_BYTES] = tr_budget_add(need->need[TR_BUDGET_BYTES], tr_budget_bitmap_bytes(cells));
    } else {
        need->need[TR_BUDGET_BYTES] = tr_budget_add(need->need[TR_BUDGET_BYTES], tr_budget_mul(d->selected, 8));
    }
}
/* complete points and distinct support assignments among them */
static int32_t coverage(const tr_bool_domain *d, uint64_t support, uint64_t *complete, uint64_t *observed) {
    int s = popc(support);
    if (d->kind != TR_BOOL_ROWS) {                           /* every assignment, each once */
        *complete = d->points;
        *observed = (uint64_t)1 << s;
        return TR_MATH_OK;
    }
    int32_t idx[64];
    int n = 0;
    for (int32_t i = 0; i < 64; i++) if ((support >> i) & 1) idx[n++] = i;
    uint64_t comp = 0, distinct = 0, vw[64];
    uint64_t *bm = NULL, *pats = NULL;
    if (s <= TR_COUNT_PATTERN_BITS) bm = zalloc(tr_budget_bitmap_bytes((uint64_t)1 << s));
    else pats = zalloc(tr_budget_mul(d->selected, 8));
    if (!bm && !pats) return TR_MATH_MEMORY;
    for (uint64_t w = 0; w < d->nw; w++) {
        uint64_t ok = tr_bool_selection_word(d, w);
        for (int j = 0; j < n && ok; j++) {
            uint64_t v, k;
            tr_bool_var_word(d, idx[j], w, &v, &k);
            ok &= k;
            vw[j] = v;
        }
        if (!ok) continue;
        comp += (uint64_t)popc(ok);
        for (uint64_t bits = ok; bits; bits &= bits - 1) {
            int p = __builtin_ctzll(bits);
            uint64_t pat = 0;
            for (int j = 0; j < n; j++) pat |= ((vw[j] >> p) & 1) << j;
            if (bm) bm[pat >> 6] |= (uint64_t)1 << (pat & 63);
            else pats[distinct++] = pat;
        }
    }
    if (bm) {
        uint64_t words = tr_budget_bitmap_bytes((uint64_t)1 << s) / 8, c = 0;
        for (uint64_t i = 0; i < words; i++) c += (uint64_t)popc(bm[i]);
        distinct = c;
    } else if (distinct) {
        qsort(pats, (size_t)distinct, 8, cmp_u64);
        uint64_t c = 1;
        for (uint64_t i = 1; i < distinct; i++) c += pats[i] != pats[i - 1];
        distinct = c;
    }
    free(bm); free(pats);
    *complete = comp; *observed = distinct;
    return TR_MATH_OK;
}
static uint64_t answer_fixed_bytes(const tr_count_domain *dom, int32_t nc) {
    uint64_t b = tr_budget_add(sizeof(tr_count_answer), tr_budget_mul((uint64_t)nc, sizeof(tr_count_result)));
    return tr_budget_add(b, dom->record ? tr_cert_record_bytes(dom->record) : 0);
}
int32_t tr_count_run_need(const tr_count_domain *dom, const tr_count_set *set, const tr_count_request *q,
                          int32_t target, tr_math_need *need) {
    tr_bool_domain d;
    tr_count_set_view s;
    if (!need || tr_count_domain_get_domain(dom, &d) || tr_count_set_get_view(set, &s)) return TR_MATH_INVALID;
    int32_t rc = request_check(&d, &s, q);
    if (rc) return rc;
    zero_need(need);
    need->need[TR_BUDGET_VARIABLES] = (uint64_t)s.k;
    need->need[TR_BUDGET_CIRCUIT] = (uint64_t)s.nnodes;
    need->need[TR_BUDGET_SUPPORT] = d.kind == TR_BOOL_ASSIGNMENTS ? d.points : 0;
    need->need[TR_BUDGET_OUTPUT] = (uint64_t)s.nc;
    if (target == TR_COUNT_NEED_DEVICE) {
        need->need[TR_BUDGET_BYTES] = tr_count_device_bytes(&d, &s);
        return TR_MATH_OK;
    }
    if (target != TR_COUNT_NEED_CPU) return TR_MATH_INVALID;
    uint64_t bytes = tr_budget_add(tr_budget_mul((uint64_t)s.nnodes, 16), tr_budget_mul((uint64_t)s.nc, 16));
    need->need[TR_BUDGET_BYTES] = tr_budget_add(bytes, answer_fixed_bytes(dom, s.nc));
    coverage_need(&d, view_support(&s), need);
    return TR_MATH_OK;
}
int32_t tr_count_counts_view(const tr_bool_domain *d, const tr_count_set_view *s, const tr_math_budget *budget,
                             tr_count_result *out, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    int32_t rc = request_check(d, s, NULL);
    if (rc) return rc;
    if (!out) return TR_MATH_INVALID;
    tr_math_need need;
    zero_need(&need);
    need.need[TR_BUDGET_VARIABLES] = (uint64_t)s->k;
    need.need[TR_BUDGET_CIRCUIT] = (uint64_t)s->nnodes;
    need.need[TR_BUDGET_SUPPORT] = d->kind == TR_BOOL_ASSIGNMENTS ? d->points : 0;
    need.need[TR_BUDGET_OUTPUT] = (uint64_t)s->nc;
    need.need[TR_BUDGET_BYTES] = tr_budget_mul((uint64_t)s->nnodes, 16);
    tr_math_budget dflt;
    rc = tr_budget_check(budget_or(budget, &dflt), &need, refusal);
    if (rc) return rc;
    uint64_t *val = zalloc((uint64_t)s->nnodes * 16);
    if (!val) return TR_MATH_MEMORY;
    counts_core(d, s, val, out);
    free(val);
    return TR_MATH_OK;
}
int32_t tr_count_plan(tr_count_result *results, int32_t nc, const tr_count_request *q, uint64_t *points_total,
                      uint64_t *witnesses_total) {
    return tr_count_layout(results, nc, q, points_total, witnesses_total);
}

/* the answer around owned results and outputs: the certificate (coverage on the host) and a copy of the record */
static tr_count_answer *answer_build(const tr_count_domain *dom, const tr_bool_domain *d, const tr_count_set_view *s,
                                     const tr_count_request *q, tr_count_result *results, uint64_t *points,
                                     uint64_t *witnesses, uint64_t pt, uint64_t wt, int32_t backend, int32_t *error) {
    tr_count_answer *a = zalloc(sizeof *a);
    if (!a) { free(results); free(points); free(witnesses); set_error(error, TR_MATH_MEMORY); return NULL; }
    a->nc = s->nc; a->request = *q; a->results = results; a->points = points; a->witnesses = witnesses;
    a->points_total = pt; a->witnesses_total = wt;
    tr_count_certificate *c = &a->cert;
    c->kind = dom->domain == TR_COUNT_OBSERVED ? TR_CERT_SNAPSHOT : TR_CERT_LAW;
    c->domain = dom->domain; c->backend = backend;
    c->support = view_support(s); c->support_size = popc(c->support);
    c->points = d->selected; c->rows = d->points;
    c->universe = c->support_size >= 64 ? TR_COUNT_NONE : (uint64_t)1 << c->support_size;
    c->nconstraints = s->nc; c->exact_selection = dom->record != NULL;
    int32_t rc = coverage(d, c->support, &c->complete, &c->observed);
    if (!rc && dom->record) {
        a->record = tr_cert_record_clone(dom->record, &rc);
        if (!a->record && !rc) rc = TR_MATH_MEMORY;
    }
    if (rc) { free(results); free(points); free(witnesses); free(a); set_error(error, rc); return NULL; }
    a->bytes = tr_budget_add(answer_fixed_bytes(dom, s->nc), tr_budget_mul(tr_budget_add(pt, wt), 8));
    a->magic = ANSWER_MAGIC;
    set_error(error, TR_MATH_OK);
    return a;
}

tr_count_answer *tr_count_run(const tr_count_domain *dom, const tr_count_set *set, const tr_count_request *q,
                              const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    tr_bool_domain d;
    tr_count_set_view s;
    if (!q || tr_count_domain_get_domain(dom, &d) || tr_count_set_get_view(set, &s)) {
        set_error(error, TR_MATH_INVALID); return NULL;
    }
    tr_math_need need;
    int32_t rc = tr_count_run_need(dom, set, q, TR_COUNT_NEED_CPU, &need);
    if (rc) { set_error(error, rc); return NULL; }
    tr_math_budget dflt;
    const tr_math_budget *b = budget_or(budget, &dflt);
    rc = tr_budget_check(b, &need, refusal);
    if (rc) { set_error(error, rc); return NULL; }
    uint64_t *val = zalloc((uint64_t)s.nnodes * 16), *fill = zalloc((uint64_t)s.nc * 16);
    tr_count_result *results = zalloc((uint64_t)s.nc * sizeof *results);
    if (!val || !fill || !results) { free(val); free(fill); free(results); set_error(error, TR_MATH_MEMORY); return NULL; }
    counts_core(&d, &s, val, results);
    uint64_t pt = 0, wt = 0;
    tr_count_layout(results, s.nc, q, &pt, &wt);
    need.need[TR_BUDGET_OUTPUT] = tr_budget_add((uint64_t)s.nc, tr_budget_add(pt, wt));   /* the outputs, now known */
    need.need[TR_BUDGET_BYTES] = tr_budget_add(need.need[TR_BUDGET_BYTES], tr_budget_mul(tr_budget_add(pt, wt), 8));
    rc = tr_budget_check(b, &need, refusal);
    uint64_t *points = NULL, *witnesses = NULL;
    if (!rc) {
        points = zalloc(pt * 8); witnesses = zalloc(wt * 8);
        if (!points || !witnesses) rc = TR_MATH_MEMORY;
    }
    if (rc) { free(val); free(fill); free(results); free(points); free(witnesses); set_error(error, rc); return NULL; }
    extract_core(&d, &s, val, results, q->flags, points, witnesses, fill);
    free(val); free(fill);
    return answer_build(dom, &d, &s, q, results, points, witnesses, pt, wt, TR_CERT_BY_CPU, error);
}

/* a point that counts in the domain */
static int counts_point(const tr_bool_domain *d, uint64_t x) {
    if (x >= d->points) return 0;
    return d->kind != TR_BOOL_ROWS || ((d->selection[x >> 6] >> (x & 63)) & 1);
}
static int segment_ok(const tr_bool_domain *d, const uint64_t *p, uint64_t at, uint64_t n) {
    for (uint64_t i = 0; i < n; i++) {
        uint64_t x = p[at + i];
        if (!counts_point(d, x) || (i && x <= p[at + i - 1])) return 0;
    }
    return 1;
}
tr_count_answer *tr_count_adopt(const tr_count_domain *dom, const tr_count_set *set, const tr_count_request *q,
                                const tr_count_result *in, const uint64_t *points, const uint64_t *witnesses,
                                int32_t backend, const tr_math_budget *budget, tr_math_refusal *refusal,
                                int32_t *error) {
    tr_budget_none(refusal);
    tr_bool_domain d;
    tr_count_set_view s;
    if (!q || !in || (backend != TR_CERT_BY_CPU && backend != TR_CERT_BY_DEVICE) ||
        tr_count_domain_get_domain(dom, &d) || tr_count_set_get_view(set, &s)) {
        set_error(error, TR_MATH_INVALID); return NULL;
    }
    tr_math_need need;
    int32_t rc = tr_count_run_need(dom, set, q, TR_COUNT_NEED_CPU, &need);
    if (rc) { set_error(error, rc); return NULL; }
    tr_count_result *results = zalloc((uint64_t)s.nc * sizeof *results);
    if (!results) { set_error(error, TR_MATH_MEMORY); return NULL; }
    memcpy(results, in, (size_t)s.nc * sizeof *results);
    uint64_t pt = 0, wt = 0;
    int bad = tr_count_layout(results, s.nc, q, &pt, &wt) != 0;
    for (int32_t j = 0; j < s.nc && !bad; j++) {             /* the readback is this set's over this domain */
        const tr_count_result *R = &in[j], *L = &results[j];
        const tr_count_constraint *c = &s.constraints[j];
        bad = R->kind != c->kind || R->a != c->a || R->b != c->b || R->reserved || R->points != d.selected ||
              R->rows_at != L->rows_at || R->rows_n != L->rows_n || R->wit_at != L->wit_at || R->wit_n != L->wit_n ||
              (R->violated == 0) != (R->witness == TR_COUNT_NONE) || (R->violated && !counts_point(&d, R->witness));
    }
    if (bad || ((pt && !points) || (wt && !witnesses))) { free(results); set_error(error, TR_MATH_INVALID); return NULL; }
    for (int32_t j = 0; j < s.nc && !bad; j++) {
        const tr_count_result *R = &results[j];
        bad = !segment_ok(&d, points, R->rows_at, R->rows_n) || !segment_ok(&d, witnesses, R->wit_at, R->wit_n) ||
              (R->wit_n && witnesses[R->wit_at] != R->witness);
    }
    if (bad) { free(results); set_error(error, TR_MATH_INVALID); return NULL; }
    need.need[TR_BUDGET_OUTPUT] = tr_budget_add((uint64_t)s.nc, tr_budget_add(pt, wt));
    need.need[TR_BUDGET_BYTES] = tr_budget_add(need.need[TR_BUDGET_BYTES], tr_budget_mul(tr_budget_add(pt, wt), 8));
    tr_math_budget dflt;
    rc = tr_budget_check(budget_or(budget, &dflt), &need, refusal);
    if (rc) { free(results); set_error(error, rc); return NULL; }
    uint64_t *pc = zalloc(pt * 8), *wc = zalloc(wt * 8);
    if (!pc || !wc) { free(results); free(pc); free(wc); set_error(error, TR_MATH_MEMORY); return NULL; }
    if (pt) memcpy(pc, points, (size_t)(pt * 8));
    if (wt) memcpy(wc, witnesses, (size_t)(wt * 8));
    return answer_build(dom, &d, &s, q, results, pc, wc, pt, wt, backend, error);
}

/* ------------------------------------------------------------------ answers */
static int answer_ok(const tr_count_answer *a) { return a && a->magic == ANSWER_MAGIC; }
void tr_count_answer_free(tr_count_answer *a) {
    if (!answer_ok(a)) return;
    free(a->results); free(a->points); free(a->witnesses);
    tr_cert_record_free(a->record);
    a->magic = 0;
    free(a);
}
int32_t tr_count_answer_get_view(const tr_count_answer *a, tr_count_answer_view *v) {
    if (!answer_ok(a) || !v) return TR_MATH_INVALID;
    memset(v, 0, sizeof *v);
    v->results = a->results; v->points = a->points; v->witnesses = a->witnesses;
    v->points_total = a->points_total; v->witnesses_total = a->witnesses_total;
    v->nc = a->nc; v->flags = a->request.flags;
    return TR_MATH_OK;
}
int32_t tr_count_answer_certificate(const tr_count_answer *a, tr_count_certificate *out) {
    if (!answer_ok(a) || !out) return TR_MATH_INVALID;
    *out = a->cert;
    return TR_MATH_OK;
}
const tr_cert_record *tr_count_answer_get_record(const tr_count_answer *a) { return answer_ok(a) ? a->record : NULL; }
int32_t tr_count_answer_fresh(const tr_count_answer *a, const tr_scope *current, tr_cert_staleness *out) {
    if (!answer_ok(a)) return TR_MATH_INVALID;
    return record_fresh(a->record, current, out);
}
int32_t tr_count_answer_authorize(const tr_count_answer *a, int32_t target, const tr_scope *current,
                                  tr_count_decision *out) {
    if (!answer_ok(a) || !out || (target != TR_COUNT_TARGET_SNAPSHOT && target != TR_COUNT_TARGET_UNIVERSE))
        return TR_MATH_INVALID;
    memset(out, 0, sizeof *out);
    out->target = target; out->domain = a->cert.domain;
    out->stale.column = out->stale.dim = -1;
    const tr_count_certificate *c = &a->cert;
    if (c->domain == TR_COUNT_UNIVERSE) {
        if (target == TR_COUNT_TARGET_SNAPSHOT) { out->reason = TR_COUNT_REFUSE_DOMAIN; return TR_MATH_DOMAIN_ERROR; }
        out->authorized = 1;
        return TR_MATH_OK;
    }
    if (target == TR_COUNT_TARGET_UNIVERSE) {
        out->unobserved = c->universe == TR_COUNT_NONE ? TR_COUNT_NONE : c->universe - c->observed;
        if (out->unobserved) { out->reason = TR_COUNT_REFUSE_UNOBSERVED; return TR_MATH_DOMAIN_ERROR; }
    }
    if (!current) { out->reason = TR_COUNT_REFUSE_NO_SCOPE; return TR_MATH_DOMAIN_ERROR; }
    int32_t rc = tr_cert_record_fresh(a->record, current, NULL, 0, TR_CERT_MATCH_EXACT, &out->stale);
    if (rc == TR_MATH_STALE) { out->reason = TR_COUNT_REFUSE_STALE; return rc; }
    if (rc) return rc;
    out->authorized = 1;
    return TR_MATH_OK;
}
