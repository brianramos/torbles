/* PROV-002: explanation specialization and deletion propagation over retained PROV-001 provenance; see
 * math_explain.h.
 * SPDX-License-Identifier: Apache-2.0 */
#include "math_explain.h"
#include "mathematics.h"
#include "math_domain.h"
#include "math_carriers.h"
#include "math_handles.h"
#include <stdlib.h>
#include <string.h>

TR_MATH_ABI(tr_expl_view, TR_EXPL_ABI);             /* the ctypes boundary of PROV-002 (math_handles.h) */
TR_MATH_ABI(tr_expl_set_result, TR_EXPL_ABI);
TR_MATH_ABI(tr_expl_info, TR_EXPL_ABI);
TR_MATH_ABI(tr_expl_sets_info, TR_EXPL_ABI);

_Static_assert(sizeof(tr_expl_view) == 128, "tr_expl_view ABI");
_Static_assert(sizeof(tr_expl_set_result) == 24, "tr_expl_set_result ABI");
_Static_assert(sizeof(tr_expl_info) == 64, "tr_expl_info ABI");
_Static_assert(sizeof(tr_expl_sets_info) == 40, "tr_expl_sets_info ABI");

#define STATE_MAGIC UINT32_C(0x4c505845)
#define SETS_MAGIC UINT32_C(0x53505845)
typedef unsigned __int128 u128;

static void set_error(int32_t *error, int32_t code) { if (error) *error = code; }
static void *zalloc(uint64_t bytes) {
    if (bytes >= (uint64_t)SIZE_MAX || bytes == TR_BUDGET_UNKNOWN) return NULL;
    return calloc(1, bytes ? (size_t)bytes : 1);
}
static const tr_math_budget *budget_of(const tr_math_budget *b, tr_math_budget *dflt) {
    if (b) return b;
    tr_budget_default(dflt);
    return dflt;
}
static void need_zero(tr_math_need *n) { for (int i = 0; i < TR_BUDGET_RESOURCES; i++) n->need[i] = 0; }
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
static int32_t check_one(const tr_math_budget *b, int32_t resource, uint64_t need, tr_math_refusal *r) {
    tr_math_need n;
    need_zero(&n);
    n.need[resource] = need;
    return check_need(b, &n, r);
}
static int u32cmp(const void *a, const void *b) { uint32_t x = *(const uint32_t *)a, y = *(const uint32_t *)b; return x < y ? -1 : x > y; }

/* ================================================================== the state */
struct tr_expl_state_s {
    uint32_t magic;
    tr_expl_view v;
    tr_prov_node *node;
    uint32_t *child, *output, *rev, *outs, *var_node, *base_c;
    uint64_t *rev_at, *out_at, *base_n, *base_val;
    uint8_t *base_b;
    uint64_t bytes, cone_vars, present;
};
static int state_valid(const tr_expl_state *s) { return s && s->magic == STATE_MAGIC; }
static void state_release(tr_expl_state *s) {
    free(s->node); free(s->child); free(s->output); free(s->rev); free(s->outs); free(s->var_node); free(s->base_c);
    free(s->rev_at); free(s->out_at); free(s->base_n); free(s->base_val); free(s->base_b);
    s->magic = 0;
    free(s);
}
void tr_expl_free(tr_expl_state *s) { if (state_valid(s)) state_release(s); }

static uint64_t state_bytes(const tr_prov_plan_view *p) {
    uint64_t n = p->nodes, b = sizeof(tr_expl_state);
    b = tr_budget_add(b, tr_budget_mul(n, 16 + 16 + 1 + 4 + 16));          /* node, base_n, base_b, base_c, offsets */
    b = tr_budget_add(b, tr_budget_mul(p->nchild, 8));                      /* child, rev */
    b = tr_budget_add(b, tr_budget_mul(p->noutputs, 8));                    /* output, outs */
    b = tr_budget_add(b, tr_budget_mul(p->nvars, 12));                      /* var_node, base_val */
    return tr_budget_add(b, tr_budget_add(tr_budget_mul(n, 17), tr_budget_mul(p->noutputs, 17)));   /* eval scratch */
}
int32_t tr_expl_create_need(const tr_prov_plan_view *p, tr_math_need *need) {
    if (!p || !need || tr_prov_plan_view_check(p)) return TR_MATH_INVALID;
    need_zero(need);
    need->need[TR_BUDGET_CIRCUIT] = p->nodes;
    need->need[TR_BUDGET_SUPPORT] = p->nchild;
    need->need[TR_BUDGET_OUTPUT] = p->noutputs;
    need->need[TR_BUDGET_BYTES] = state_bytes(p);
    return TR_MATH_OK;
}

tr_expl_state *tr_expl_create(const tr_prov_plan_view *p, const tr_prov_valuations *base, const tr_math_budget *budget,
                              tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    if (!p || tr_prov_plan_view_check(p) || (base && (base->count != 1 || tr_prov_valuations_check(base, p->nvars)))) {
        set_error(error, TR_MATH_INVALID); return NULL;
    }
    if (p->nodes >= TR_EXPL_NONE || p->nvars >= TR_EXPL_NONE) { set_error(error, TR_MATH_UNSUPPORTED); return NULL; }
    tr_math_budget dflt;
    const tr_math_budget *b = budget_of(budget, &dflt);
    tr_math_need need;
    tr_expl_create_need(p, &need);
    int32_t rc = check_need(b, &need, refusal);
    if (rc) { set_error(error, rc); return NULL; }
    tr_expl_state *s = (tr_expl_state *)zalloc(sizeof *s);
    if (!s) { set_error(error, TR_MATH_MEMORY); return NULL; }
    const uint64_t n = p->nodes, nc = p->nchild, no = p->noutputs, nv = p->nvars;
    s->node = (tr_prov_node *)zalloc(tr_budget_mul(n, sizeof(tr_prov_node)));
    s->child = (uint32_t *)zalloc(tr_budget_mul(nc, 4)); s->rev = (uint32_t *)zalloc(tr_budget_mul(nc, 4));
    s->output = (uint32_t *)zalloc(tr_budget_mul(no, 4)); s->outs = (uint32_t *)zalloc(tr_budget_mul(no, 4));
    s->var_node = (uint32_t *)zalloc(tr_budget_mul(nv, 4)); s->base_c = (uint32_t *)zalloc(tr_budget_mul(n, 4));
    s->rev_at = (uint64_t *)zalloc(tr_budget_mul(n + 1, 8)); s->out_at = (uint64_t *)zalloc(tr_budget_mul(n + 1, 8));
    s->base_n = (uint64_t *)zalloc(tr_budget_mul(n, 16)); s->base_val = (uint64_t *)zalloc(tr_budget_mul(nv, 8));
    s->base_b = (uint8_t *)zalloc(n);
    unsigned __int128 *s128 = (unsigned __int128 *)zalloc(tr_budget_mul(n, 16));
    uint8_t *s8 = (uint8_t *)zalloc(n);
    uint64_t *on = (uint64_t *)zalloc(tr_budget_mul(no, 16));
    uint8_t *ob = (uint8_t *)zalloc(no);
    uint64_t *fill = (uint64_t *)zalloc(tr_budget_mul(n + 1, 8));
    if (!s->node || !s->child || !s->rev || !s->output || !s->outs || !s->var_node || !s->base_c || !s->rev_at ||
        !s->out_at || !s->base_n || !s->base_val || !s->base_b || !s128 || !s8 || !on || !ob || !fill) rc = TR_MATH_MEMORY;
    if (!rc) {
        if (n) memcpy(s->node, p->node, (size_t)n * sizeof(tr_prov_node));
        if (nc) memcpy(s->child, p->child, (size_t)nc * 4);
        if (no) memcpy(s->output, p->output, (size_t)no * 4);
        for (uint64_t i = 0; i < n; i++)                   /* reverse edges: one parent entry per child entry */
            if (s->node[i].op == TR_PROV_PLUS || s->node[i].op == TR_PROV_TIMES)
                for (uint32_t c = 0; c < s->node[i].n; c++) s->rev_at[s->child[s->node[i].arg + c] + 1]++;
        for (uint64_t i = 0; i < n; i++) s->rev_at[i + 1] += s->rev_at[i];
        memcpy(fill, s->rev_at, (size_t)(n + 1) * 8);
        for (uint64_t i = 0; i < n; i++)
            if (s->node[i].op == TR_PROV_PLUS || s->node[i].op == TR_PROV_TIMES)
                for (uint32_t c = 0; c < s->node[i].n; c++) s->rev[fill[s->child[s->node[i].arg + c]]++] = (uint32_t)i;
        for (uint64_t j = 0; j < no; j++) s->out_at[s->output[j] + 1]++;
        for (uint64_t i = 0; i < n; i++) s->out_at[i + 1] += s->out_at[i];
        memcpy(fill, s->out_at, (size_t)(n + 1) * 8);
        for (uint64_t j = 0; j < no; j++) s->outs[fill[s->output[j]]++] = (uint32_t)j;
        for (uint64_t v = 0; v < nv; v++) s->var_node[v] = TR_EXPL_NONE;
        for (uint64_t i = 0; i < n; i++) if (s->node[i].op == TR_PROV_VAR) { s->var_node[s->node[i].arg] = (uint32_t)i; s->cone_vars++; }
        uint64_t offs[2] = {0, 0};
        tr_prov_valuations all;                            /* the default: every variable 1, no override */
        all.count = 1; all.offset = offs; all.value = NULL; all.fallback = NULL; all.var = NULL;
        const tr_prov_valuations *bv = base ? base : &all;
        for (uint64_t v = 0; v < nv; v++) { s->base_val[v] = tr_prov_var_value(bv, 0, (uint32_t)v); s->present += s->base_val[v] != 0; }
        int32_t st = tr_prov_eval_count(p, bv, 0, s128, on);
        s->v.count_ok = st == 0;
        if (st == 0) for (uint64_t i = 0; i < n; i++) tr_expl_put(s->base_n, i, s128[i]);
        tr_prov_eval_bool(p, bv, 0, s8, ob);
        for (uint64_t i = 0; i < n; i++) {
            s->base_b[i] = s8[i];
            const tr_prov_node nd = s->node[i];
            if (nd.op != TR_PROV_PLUS && nd.op != TR_PROV_TIMES) continue;
            uint32_t cnt = 0;
            for (uint32_t c = 0; c < nd.n; c++) cnt += nd.op == TR_PROV_PLUS ? s8[s->child[nd.arg + c]] : !s8[s->child[nd.arg + c]];
            s->base_c[i] = cnt;
        }
    }
    free(s128); free(s8); free(on); free(ob); free(fill);
    if (rc) { state_release(s); set_error(error, rc); return NULL; }
    s->v.nodes = n; s->v.nchild = nc; s->v.noutputs = no; s->v.nvars = nv;
    s->v.node = s->node; s->v.child = s->child; s->v.output = s->output; s->v.rev_at = s->rev_at; s->v.rev = s->rev;
    s->v.out_at = s->out_at; s->v.outs = s->outs; s->v.var_node = s->var_node; s->v.base_n = s->base_n;
    s->v.base_b = s->base_b; s->v.base_c = s->base_c;
    s->bytes = need.need[TR_BUDGET_BYTES];
    s->magic = STATE_MAGIC;
    set_error(error, TR_MATH_OK);
    return s;
}
int32_t tr_expl_get_view(const tr_expl_state *s, tr_expl_view *v) {
    if (!state_valid(s) || !v) return TR_MATH_INVALID;
    *v = s->v;
    return TR_MATH_OK;
}
int32_t tr_expl_get_info(const tr_expl_state *s, tr_expl_info *out) {
    if (!state_valid(s) || !out) return TR_MATH_INVALID;
    memset(out, 0, sizeof *out);
    out->nodes = s->v.nodes; out->children = s->v.nchild; out->outputs = s->v.noutputs; out->variables = s->v.nvars;
    out->cone_variables = s->cone_vars; out->present = s->present; out->bytes = s->bytes; out->count_ok = s->v.count_ok;
    return TR_MATH_OK;
}
int32_t tr_expl_base(const tr_expl_state *s, uint64_t *out, uint64_t capacity) {
    if (!state_valid(s) || capacity < s->v.nvars || (s->v.nvars && !out)) return TR_MATH_INVALID;
    if (s->v.nvars) memcpy(out, s->base_val, (size_t)s->v.nvars * 8);
    return TR_MATH_OK;
}

/* ================================================================== deletion propagation */
static uint64_t scratch_bytes(uint64_t nodes) { return tr_budget_mul(nodes ? nodes : 1, TR_EXPL_SCRATCH_PER_NODE); }
int32_t tr_expl_propagate_need(const tr_expl_state *s, const tr_prov_valuations *v, int32_t target, tr_math_need *need) {
    if (!state_valid(s) || !need || tr_expl_sets_check(v, s->v.nvars)) return TR_MATH_INVALID;
    need_zero(need);
    need->need[TR_BUDGET_OUTPUT] = tr_budget_mul(v->count, s->v.noutputs);
    need->need[TR_BUDGET_BYTES] = target == TR_EXPL_NEED_DEVICE
        ? tr_expl_device_bytes(&s->v, v, tr_expl_device_chunk(&s->v, v->count)) : scratch_bytes(s->v.nodes);
    return TR_MATH_OK;
}
static tr_expl_scratch scratch_alloc(uint64_t n, int *ok) {
    tr_expl_scratch x;
    uint64_t m = n ? n : 1;
    x.newn = (uint64_t *)zalloc(tr_budget_mul(m, 16)); x.sub = (uint64_t *)zalloc(tr_budget_mul(m, 16));
    x.add = (uint64_t *)zalloc(tr_budget_mul(m, 16)); x.dcnt = (int64_t *)zalloc(tr_budget_mul(m, 8));
    x.heap = (uint32_t *)zalloc(tr_budget_mul(m, 4)); x.touched = (uint32_t *)zalloc(tr_budget_mul(m, 4));
    x.flag = (uint8_t *)zalloc(m);
    *ok = x.newn && x.sub && x.add && x.dcnt && x.heap && x.touched && x.flag;
    return x;
}
static void scratch_free(tr_expl_scratch *x) {
    free(x->newn); free(x->sub); free(x->add); free(x->dcnt); free(x->heap); free(x->touched); free(x->flag);
}
int32_t tr_expl_propagate(const tr_expl_state *s, const tr_prov_valuations *v, uint64_t *out_n, uint8_t *out_b,
                          tr_expl_set_result *res, const tr_math_budget *budget, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!state_valid(s) || tr_expl_sets_check(v, s->v.nvars) || (v->count && s->v.noutputs && (!out_n || !out_b)) ||
        (v->count && !res)) return TR_MATH_INVALID;
    tr_math_budget dflt;
    const tr_math_budget *b = budget_of(budget, &dflt);
    tr_math_need need;
    tr_expl_propagate_need(s, v, TR_EXPL_NEED_CPU, &need);
    int32_t rc = check_need(b, &need, refusal);
    if (rc) return rc;
    int ok = 0;
    tr_expl_scratch x = scratch_alloc(s->v.nodes, &ok);
    if (!ok) { scratch_free(&x); return TR_MATH_MEMORY; }
    for (uint64_t w = 0; w < v->count; w++) {
        res[w].reserved = 0;
        res[w].status = tr_expl_set(&s->v, v, w, &x, out_n + 2 * w * s->v.noutputs, out_b + w * s->v.noutputs,
                                    &res[w].affected, &res[w].changed);
    }
    scratch_free(&x);
    return TR_MATH_OK;
}

/* ================================================================== the lineage as a decision diagram */
/* mark the cone of node r (children are below their parents) */
static void cone_mark(const tr_expl_state *s, uint32_t r, uint8_t *mark) {
    mark[r] = 1;
    for (uint64_t i = (uint64_t)r + 1; i-- > 0;) {
        if (!mark[i]) continue;
        const tr_prov_node nd = s->node[i];
        if (nd.op == TR_PROV_PLUS || nd.op == TR_PROV_TIMES)
            for (uint32_t c = 0; c < nd.n; c++) mark[s->child[nd.arg + c]] = 1;
    }
}
/* the substitution of variable v: 0 free, 1 FALSE, 2 TRUE */
static int fixed_of(const tr_expl_state *s, uint64_t v, int32_t subst) {
    if (subst == TR_EXPL_WHY) return s->base_val[v] == 0 ? 1 : 0;
    if (subst == TR_EXPL_WHYNOT) return s->base_val[v] != 0 ? 2 : 0;
    return 0;
}
static int32_t fold(tr_symc_mgr *m, int32_t op, uint32_t *ids, uint32_t n, uint32_t *out, tr_math_refusal *r) {
    while (n > 1) {                                        /* balanced: pairs, then pairs of pairs */
        uint32_t k = 0;
        for (uint32_t i = 0; i + 1 < n; i += 2) {
            int32_t rc = tr_symc_mgr_apply(m, op, ids[i], ids[i + 1], &ids[k], r);
            if (rc) return rc;
            k++;
        }
        if (n & 1) ids[k++] = ids[n - 1];
        n = k;
    }
    *out = n ? ids[0] : (op == TR_SYMC_AND ? TR_SYMC_TRUE : TR_SYMC_FALSE);
    return TR_MATH_OK;
}
tr_symc_dd *tr_expl_lineage(const tr_expl_state *s, uint32_t output, int32_t subst, int32_t order_kind,
                            const uint32_t *given, const tr_math_budget *budget, tr_math_refusal *refusal,
                            int32_t *error) {
    tr_budget_none(refusal);
    if (!state_valid(s) || output >= s->v.noutputs || subst < TR_EXPL_FREE || subst > TR_EXPL_WHYNOT ||
        (order_kind != TR_SYMC_ORDER_DFS && order_kind != TR_SYMC_ORDER_NATURAL && order_kind != TR_SYMC_ORDER_GIVEN) ||
        (order_kind == TR_SYMC_ORDER_GIVEN && s->v.nvars && !given)) { set_error(error, TR_MATH_INVALID); return NULL; }
    tr_math_budget dflt;
    const tr_math_budget *b = budget_of(budget, &dflt);
    const uint32_t r = s->output[output];
    const uint64_t top = (uint64_t)r + 1, nv = s->v.nvars;
    uint64_t wide = 1, entries = 0;                        /* the most child entries of one node; all of them */
    for (uint64_t i = 0; i < top; i++)
        if (s->node[i].op == TR_PROV_PLUS || s->node[i].op == TR_PROV_TIMES) {
            if (s->node[i].n > wide) wide = s->node[i].n;
            entries += s->node[i].n;
        }
    /* the DFS stack: a node is expanded once, so each child entry is pushed at most once (entries + 1 slots) */
    int32_t rc = check_one(b, TR_BUDGET_BYTES, tr_budget_add(tr_budget_add(tr_budget_mul(top, 1 + 4 + 1),
                                                                            tr_budget_mul(tr_budget_add(wide, tr_budget_add(entries, 1)), 4)),
                                                              tr_budget_mul(nv, 5)), refusal);
    if (rc) { set_error(error, rc); return NULL; }
    uint8_t *mark = (uint8_t *)zalloc(top), *seen = (uint8_t *)zalloc(nv ? nv : 1);
    uint32_t *map = (uint32_t *)zalloc(tr_budget_mul(top, 4)), *stack = (uint32_t *)zalloc(tr_budget_mul(tr_budget_add(entries, 1), 4));
    uint32_t *levels = (uint32_t *)zalloc(tr_budget_mul(nv ? nv : 1, 4)), *kids = (uint32_t *)zalloc(tr_budget_mul(wide, 4));
    tr_symc_mgr *m = NULL;
    tr_symc_dd *d = NULL;
    if (!mark || !seen || !map || !stack || !levels || !kids) rc = TR_MATH_MEMORY;
    uint32_t nl = 0;
    if (!rc) {
        cone_mark(s, r, mark);
        if (order_kind == TR_SYMC_ORDER_DFS) {             /* first appearance from the answer, first child first */
            uint8_t *vis = (uint8_t *)zalloc(top);
            uint64_t sp = 0;
            if (!vis) rc = TR_MATH_MEMORY;
            else stack[sp++] = r;
            while (sp && !rc) {
                uint32_t i = stack[--sp];
                if (vis[i]) continue;
                vis[i] = 1;
                const tr_prov_node nd = s->node[i];
                if (nd.op == TR_PROV_VAR) {
                    if (!fixed_of(s, nd.arg, subst) && !seen[nd.arg]) { seen[nd.arg] = 1; levels[nl++] = (uint32_t)nd.arg; }
                } else if (nd.op == TR_PROV_PLUS || nd.op == TR_PROV_TIMES) {
                    for (uint32_t c = nd.n; c-- > 0;) if (!vis[s->child[nd.arg + c]]) stack[sp++] = s->child[nd.arg + c];
                }
            }
            free(vis);
        } else {
            for (uint64_t i = 0; i < top; i++)
                if (mark[i] && s->node[i].op == TR_PROV_VAR && !fixed_of(s, s->node[i].arg, subst)) seen[s->node[i].arg] = 1;
            if (order_kind == TR_SYMC_ORDER_NATURAL) { for (uint64_t v = 0; v < nv; v++) if (seen[v]) levels[nl++] = (uint32_t)v; }
            else {
                uint8_t *used = (uint8_t *)zalloc(nv ? nv : 1);
                if (!used) rc = TR_MATH_MEMORY;
                for (uint64_t i = 0; i < nv && !rc; i++) {
                    if (given[i] >= nv || used[given[i]]) { rc = TR_MATH_INVALID; break; }
                    used[given[i]] = 1;
                    if (seen[given[i]]) levels[nl++] = given[i];
                }
                free(used);
            }
        }
    }
    if (!rc) m = tr_symc_mgr_create((uint32_t)nv, nl, levels, b, refusal, &rc);
    for (uint64_t i = 0; i < top && !rc; i++) {            /* bottom up over the cone */
        if (!mark[i]) continue;
        const tr_prov_node nd = s->node[i];
        if (nd.op == TR_PROV_ZERO) map[i] = TR_SYMC_FALSE;
        else if (nd.op == TR_PROV_ONE) map[i] = TR_SYMC_TRUE;
        else if (nd.op == TR_PROV_VAR) {
            int f = fixed_of(s, nd.arg, subst);
            if (f) map[i] = f == 2 ? TR_SYMC_TRUE : TR_SYMC_FALSE;
            else rc = tr_symc_mgr_var(m, (uint32_t)nd.arg, &map[i], refusal);
        } else {
            for (uint32_t c = 0; c < nd.n; c++) kids[c] = map[s->child[nd.arg + c]];
            rc = fold(m, nd.op == TR_PROV_PLUS ? TR_SYMC_OR : TR_SYMC_AND, kids, nd.n, &map[i], refusal);
        }
    }
    if (!rc) d = tr_symc_mgr_finish(m, &map[r], 1, 0, b, refusal, &rc);
    tr_symc_mgr_free(m);
    free(mark); free(seen); free(map); free(stack); free(levels); free(kids);
    if (!d) { set_error(error, rc ? rc : TR_MATH_MEMORY); return NULL; }
    set_error(error, TR_MATH_OK);
    return d;
}

/* ================================================================== explanations */
struct tr_expl_sets_s {
    uint32_t magic;
    tr_expl_sets_info info;
    uint64_t *offsets;
    uint32_t *vars;
};
static int sets_valid(const tr_expl_sets *x) { return x && x->magic == SETS_MAGIC; }
void tr_expl_sets_free(tr_expl_sets *x) {
    if (!sets_valid(x)) return;
    free(x->offsets); free(x->vars);
    x->magic = 0;
    free(x);
}
int32_t tr_expl_sets_get_info(const tr_expl_sets *x, tr_expl_sets_info *out) {
    if (!sets_valid(x) || !out) return TR_MATH_INVALID;
    *out = x->info;
    return TR_MATH_OK;
}
int32_t tr_expl_sets_get(const tr_expl_sets *x, uint64_t *offsets, uint32_t *vars, uint64_t sets_cap, uint64_t vars_cap) {
    if (!sets_valid(x) || !offsets || sets_cap < x->info.nsets || vars_cap < x->info.entries ||
        (x->info.entries && !vars)) return TR_MATH_INVALID;
    memcpy(offsets, x->offsets, (size_t)(x->info.nsets + 1) * 8);
    if (x->info.entries) memcpy(vars, x->vars, (size_t)x->info.entries * 4);
    return TR_MATH_OK;
}
/* a result of nsets sets in a CSR of levels or variables (adopted) */
static tr_expl_sets *sets_new(int32_t kind, int32_t status, uint64_t nsets, uint64_t entries) {
    tr_expl_sets *x = (tr_expl_sets *)zalloc(sizeof *x);
    if (!x) return NULL;
    x->offsets = (uint64_t *)zalloc(tr_budget_mul(nsets + 1, 8));
    x->vars = (uint32_t *)zalloc(tr_budget_mul(entries ? entries : 1, 4));
    if (!x->offsets || !x->vars) { free(x->offsets); free(x->vars); free(x); return NULL; }
    x->magic = SETS_MAGIC; x->info.kind = kind; x->info.status = status; x->info.nsets = nsets; x->info.entries = entries;
    return x;
}
/* the value of the diagram from node m at the assignment "the levels of S (increasing) are 1, the others 0" */
static int eval_levels(const tr_symc_dd_view *d, uint32_t m, const uint32_t *S, uint32_t len) {
    uint32_t p = 0;
    while (m >= 2) {
        uint32_t l = d->node[m].level;
        while (p < len && S[p] < l) p++;
        m = (p < len && S[p] == l) ? d->node[m].hi : d->node[m].lo;
    }
    return m == TR_SYMC_TRUE;
}
/* the minimal models of a monotone diagram's root, as sets of levels (MM(n) = MM(lo) + {l} x {S in MM(hi): lo(S) = 0}):
 * every set once in a pool of levels, every node's list as references to sets */
typedef struct { uint64_t at; uint32_t len, reserved; } mm_set;
typedef struct { mm_set *set; uint64_t nsets, scap; uint32_t *pool; uint64_t npool, pcap; uint64_t *refs; uint64_t nrefs, rcap; } mm_store;
static void mm_free(mm_store *m) { free(m->set); free(m->pool); free(m->refs); memset(m, 0, sizeof *m); }
static int mm_grow(void **p, uint64_t *cap, uint64_t need, uint64_t width) {
    if (need <= *cap) return 1;
    uint64_t c = *cap ? *cap : 64;
    while (c < need) c *= 2;
    void *q = realloc(*p, (size_t)(c * width));
    if (!q) return 0;
    *p = q; *cap = c;
    return 1;
}
static int32_t minimal_models(const tr_symc_dd_view *d, uint64_t limit_support, uint64_t *root_at, uint64_t *root_n,
                              mm_store *st, tr_math_refusal *r) {
    uint64_t *at = (uint64_t *)zalloc(tr_budget_mul(d->nnodes, 8)), *cnt = (uint64_t *)zalloc(tr_budget_mul(d->nnodes, 8));
    if (!at || !cnt) { free(at); free(cnt); return TR_MATH_MEMORY; }
    int32_t rc = TR_MATH_OK;
    /* the empty set, the list of TRUE */
    if (!mm_grow((void **)&st->set, &st->scap, 1, sizeof(mm_set)) || !mm_grow((void **)&st->refs, &st->rcap, 1, 8))
        rc = TR_MATH_MEMORY;
    if (!rc) {
        st->set[0].at = 0; st->set[0].len = 0; st->set[0].reserved = 0; st->nsets = 1;
        st->refs[0] = 0; st->nrefs = 1;
        at[0] = 0; cnt[0] = 0;                              /* FALSE: no models */
        at[1] = 0; cnt[1] = 1;                              /* TRUE: the empty set */
    }
    for (uint64_t i = 2; i < d->nnodes && !rc; i++) {
        const tr_symc_node nd = d->node[i];
        uint64_t start = st->nrefs, lo_n = cnt[nd.lo], hi_n = cnt[nd.hi];
        if (tr_budget_add(tr_budget_add(st->nrefs, lo_n + hi_n), st->npool) > limit_support) {
            if (r) { r->resource = TR_BUDGET_SUPPORT; r->refused = 1; r->need = tr_budget_add(tr_budget_add(st->nrefs, lo_n + hi_n), st->npool); r->limit = limit_support; }
            rc = TR_MATH_BUDGET; break;
        }
        if (!mm_grow((void **)&st->refs, &st->rcap, st->nrefs + lo_n + hi_n, 8)) { rc = TR_MATH_MEMORY; break; }
        for (uint64_t k = 0; k < lo_n; k++) st->refs[st->nrefs++] = st->refs[at[nd.lo] + k];
        for (uint64_t k = 0; k < hi_n && !rc; k++) {
            uint64_t sid = st->refs[at[nd.hi] + k];
            uint32_t len = st->set[sid].len;
            if (eval_levels(d, nd.lo, st->pool + st->set[sid].at, len)) continue;   /* S already suffices without l */
            if (tr_budget_add(st->npool, (uint64_t)len + 1) + st->nrefs > limit_support) {
                if (r) { r->resource = TR_BUDGET_SUPPORT; r->refused = 1; r->need = st->npool + len + 1 + st->nrefs; r->limit = limit_support; }
                rc = TR_MATH_BUDGET; break;
            }
            if (!mm_grow((void **)&st->pool, &st->pcap, st->npool + len + 1, 4) ||
                !mm_grow((void **)&st->set, &st->scap, st->nsets + 1, sizeof(mm_set))) { rc = TR_MATH_MEMORY; break; }
            uint64_t base = st->npool, from = st->set[sid].at;
            st->pool[base] = nd.level;                       /* l is above every level of S */
            memmove(st->pool + base + 1, st->pool + from, (size_t)len * 4);
            st->npool += (uint64_t)len + 1;
            st->set[st->nsets].at = base; st->set[st->nsets].len = len + 1; st->set[st->nsets].reserved = 0;
            st->refs[st->nrefs++] = st->nsets++;
        }
        at[i] = start; cnt[i] = st->nrefs - start;
    }
    if (!rc) { uint32_t root = d->roots[0]; *root_at = at[root]; *root_n = cnt[root]; }
    free(at); free(cnt);
    return rc;
}

tr_expl_sets *tr_expl_explain(const tr_expl_state *s, uint32_t output, int32_t kind, uint64_t max_sets,
                              const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    if (!state_valid(s) || output >= s->v.noutputs || kind < TR_EXPL_LINEAGE || kind > TR_EXPL_WHY_NOT) {
        set_error(error, TR_MATH_INVALID); return NULL;
    }
    tr_math_budget dflt;
    const tr_math_budget *b = budget_of(budget, &dflt);
    const uint32_t r = s->output[output];
    int32_t rc = TR_MATH_OK;
    if (kind == TR_EXPL_LINEAGE) {                         /* the variables of the cone */
        rc = check_one(b, TR_BUDGET_BYTES, tr_budget_mul((uint64_t)r + 1, 5), refusal);
        if (rc) { set_error(error, rc); return NULL; }
        uint8_t *mark = (uint8_t *)zalloc((uint64_t)r + 1);
        if (!mark) { set_error(error, TR_MATH_MEMORY); return NULL; }
        cone_mark(s, r, mark);
        uint64_t nvar = 0;
        for (uint64_t i = 0; i <= r; i++) nvar += mark[i] && s->node[i].op == TR_PROV_VAR;
        tr_expl_sets *x = sets_new(kind, s->node[r].op == TR_PROV_ZERO ? TR_EXPL_UNDERIVABLE : TR_EXPL_FOUND, 1, nvar);
        if (!x) { free(mark); set_error(error, TR_MATH_MEMORY); return NULL; }
        uint64_t k = 0;
        for (uint64_t i = 0; i <= r; i++) if (mark[i] && s->node[i].op == TR_PROV_VAR) x->vars[k++] = (uint32_t)s->node[i].arg;
        qsort(x->vars, (size_t)nvar, 4, u32cmp);
        x->offsets[0] = 0; x->offsets[1] = nvar;
        free(mark);
        set_error(error, TR_MATH_OK);
        return x;
    }
    tr_symc_dd *dd = tr_expl_lineage(s, output, kind == TR_EXPL_WHY_NOT ? TR_EXPL_WHYNOT : TR_EXPL_WHY,
                                     TR_SYMC_ORDER_DFS, NULL, b, refusal, &rc);
    if (!dd) { set_error(error, rc); return NULL; }
    tr_symc_dd_view d;
    tr_symc_dd_get_view(dd, &d);
    const uint32_t root = d.roots[0];
    tr_expl_sets *x = NULL;
    int32_t status = TR_EXPL_FOUND;
    if (kind == TR_EXPL_WHY_NOT) {
        if (root == TR_SYMC_TRUE) status = TR_EXPL_PRESENT;
        else if (root == TR_SYMC_FALSE) status = TR_EXPL_UNDERIVABLE;
    } else if (root == TR_SYMC_FALSE) status = TR_EXPL_ABSENT;
    if (status != TR_EXPL_FOUND) {
        x = sets_new(kind, status, 0, 0);
        if (!x) rc = TR_MATH_MEMORY;
    } else if (kind == TR_EXPL_RELEVANT || kind == TR_EXPL_NECESSARY) {
        uint64_t nl = 0;
        for (uint32_t l = 0; l < d.nlevels; l++) nl += d.count[l] > 0;
        uint8_t *reach = NULL;
        if (kind == TR_EXPL_NECESSARY) {
            rc = check_one(b, TR_BUDGET_ITERATIONS, tr_budget_mul(nl, d.nnodes), refusal);
            if (!rc) rc = check_one(b, TR_BUDGET_BYTES, tr_budget_add(d.nnodes, tr_budget_mul(nl, 4)), refusal);
            if (!rc && !(reach = (uint8_t *)zalloc(d.nnodes))) rc = TR_MATH_MEMORY;
        }
        uint32_t *vars = rc ? NULL : (uint32_t *)zalloc(tr_budget_mul(nl ? nl : 1, 4));
        if (!rc && !vars) rc = TR_MATH_MEMORY;
        uint64_t k = 0;
        for (uint32_t l = 0; l < d.nlevels && !rc; l++) {
            if (!d.count[l]) continue;
            if (kind == TR_EXPL_NECESSARY) {               /* TRUE unreachable with the variable of level l at 0 */
                reach[0] = 0; reach[1] = 1;
                for (uint64_t i = 2; i < d.nnodes; i++) {
                    const tr_symc_node nd = d.node[i];
                    reach[i] = nd.level == l ? reach[nd.lo] : (uint8_t)(reach[nd.lo] | reach[nd.hi]);
                }
                if (reach[root]) continue;
            }
            vars[k++] = d.order[l];
        }
        if (!rc) {
            qsort(vars, (size_t)k, 4, u32cmp);
            x = sets_new(kind, status, 1, k);
            if (!x) rc = TR_MATH_MEMORY;
            else { memcpy(x->vars, vars, (size_t)k * 4); x->offsets[0] = 0; x->offsets[1] = k; }
        }
        free(reach); free(vars);
    } else if (kind == TR_EXPL_SMALLEST) {
        rc = check_one(b, TR_BUDGET_BYTES, tr_budget_mul(d.nnodes, 5), refusal);
        uint32_t *cost = rc ? NULL : (uint32_t *)zalloc(tr_budget_mul(d.nnodes, 4));
        uint8_t *hi = rc ? NULL : (uint8_t *)zalloc(d.nnodes);
        if (!rc && (!cost || !hi)) rc = TR_MATH_MEMORY;
        if (!rc) {
            cost[0] = UINT32_MAX; cost[1] = 0;
            for (uint64_t i = 2; i < d.nnodes; i++) {
                uint32_t cl = cost[d.node[i].lo], ch = cost[d.node[i].hi];
                ch = ch == UINT32_MAX ? UINT32_MAX : ch + 1;
                hi[i] = ch < cl;
                cost[i] = ch < cl ? ch : cl;
            }
            uint32_t n = cost[root], k = 0;
            x = sets_new(kind, status, 1, n);
            if (!x) rc = TR_MATH_MEMORY;
            for (uint32_t m = root; !rc && m >= 2;) {
                if (hi[m]) { x->vars[k++] = d.order[d.node[m].level]; m = d.node[m].hi; }
                else m = d.node[m].lo;
            }
            if (!rc) { qsort(x->vars, (size_t)k, 4, u32cmp); x->offsets[0] = 0; x->offsets[1] = k; }
        }
        free(cost); free(hi);
    } else {                                               /* SUFFICIENT, WHY_NOT: the minimal models */
        mm_store st;
        memset(&st, 0, sizeof st);
        uint64_t at = 0, n = 0;
        rc = check_one(b, TR_BUDGET_BYTES, tr_budget_mul(d.nnodes, 16), refusal);
        if (!rc) rc = minimal_models(&d, b->limit[TR_BUDGET_SUPPORT], &at, &n, &st, refusal);
        if (!rc && n > max_sets) {
            if (refusal) { refusal->resource = TR_BUDGET_OUTPUT; refusal->refused = 1; refusal->need = n; refusal->limit = max_sets; }
            rc = TR_MATH_BUDGET;
        }
        if (!rc) rc = check_one(b, TR_BUDGET_OUTPUT, n, refusal);
        if (!rc) {
            uint64_t entries = 0;
            for (uint64_t k = 0; k < n; k++) entries += st.set[st.refs[at + k]].len;
            x = sets_new(kind, status, n, entries);
            if (!x) rc = TR_MATH_MEMORY;
            uint64_t e = 0;
            for (uint64_t k = 0; k < n && !rc; k++) {
                const mm_set ms = st.set[st.refs[at + k]];
                x->offsets[k] = e;
                for (uint32_t q = 0; q < ms.len; q++) x->vars[e + q] = d.order[st.pool[ms.at + q]];
                qsort(x->vars + e, ms.len, 4, u32cmp);
                e += ms.len;
            }
            if (!rc) x->offsets[n] = e;
        }
        mm_free(&st);
    }
    if (x) { x->info.diagram_nodes = d.nnodes; x->info.levels = d.nlevels; }
    tr_symc_dd_free(dd);
    if (rc) { tr_expl_sets_free(x); set_error(error, rc); return NULL; }
    set_error(error, TR_MATH_OK);
    return x;
}

/* ================================================================== probabilities */
int32_t tr_expl_probability_weights(const tr_expl_state *s, const int64_t *num, const int64_t *den, tr_symc_weight *out) {
    if (!state_valid(s) || (s->v.nvars && (!num || !den || !out))) return TR_MATH_INVALID;
    for (uint64_t v = 0; v < s->v.nvars; v++) {
        if (den[v] < 1 || num[v] < 0 || num[v] > den[v]) return TR_MATH_INVALID;
        out[v].num1 = num[v]; out[v].den1 = den[v]; out[v].num0 = den[v] - num[v]; out[v].den0 = den[v];
    }
    return TR_MATH_OK;
}
int32_t tr_expl_probability_bounds(const tr_expl_state *s, uint32_t output, const int64_t *num, const int64_t *den,
                                   tr_math_interval *out, int32_t *read_once, const tr_math_budget *budget,
                                   tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!state_valid(s) || output >= s->v.noutputs || !out || !read_once || (s->v.nvars && (!num || !den)))
        return TR_MATH_INVALID;
    tr_math_budget dflt;
    const tr_math_budget *b = budget_of(budget, &dflt);
    const uint32_t r = s->output[output];
    const uint64_t top = (uint64_t)r + 1;
    int32_t rc = check_one(b, TR_BUDGET_BYTES, tr_budget_mul(top, 5 + 4), refusal);
    if (rc) return rc;
    uint8_t *mark = (uint8_t *)zalloc(top);
    uint32_t *bit = (uint32_t *)zalloc(tr_budget_mul(top, 4));
    if (!mark || !bit) { free(mark); free(bit); return TR_MATH_MEMORY; }
    cone_mark(s, r, mark);
    uint64_t nvar = 0;
    for (uint64_t i = 0; i < top; i++) if (mark[i] && s->node[i].op == TR_PROV_VAR) bit[i] = (uint32_t)nvar++;
    const uint64_t words = (nvar + 63) / 64;
    rc = check_one(b, TR_BUDGET_BYTES, tr_budget_add(tr_budget_mul(top, 9), tr_budget_mul(tr_budget_mul(top, words + 1), 8 + 16)), refusal);
    uint64_t *sup = rc ? NULL : (uint64_t *)zalloc(tr_budget_mul(tr_budget_mul(top, words ? words : 1), 8));
    uint64_t *uni = rc ? NULL : (uint64_t *)zalloc(tr_budget_mul(words ? words : 1, 8));
    tr_math_interval *iv = rc ? NULL : (tr_math_interval *)zalloc(tr_budget_mul(top, sizeof(tr_math_interval)));
    if (!rc && (!sup || !uni || !iv)) rc = TR_MATH_MEMORY;
    int once = 1;
    const uint64_t W = words ? words : 1;
    for (uint64_t i = 0; i < top && !rc; i++) {
        if (!mark[i]) continue;
        const tr_prov_node nd = s->node[i];
        uint64_t *si = sup + i * W;
        if (nd.op == TR_PROV_ZERO) { iv[i].lo = 0.0; iv[i].hi = 0.0; continue; }
        if (nd.op == TR_PROV_ONE) { iv[i].lo = 1.0; iv[i].hi = 1.0; continue; }
        if (nd.op == TR_PROV_VAR) {
            uint64_t v = nd.arg;
            if (den[v] < 1 || num[v] < 0 || num[v] > den[v]) { rc = TR_MATH_INVALID; break; }
            tr_interval_enclose_q64(num[v], den[v], &iv[i]);
            si[bit[i] / 64] |= (uint64_t)1 << (bit[i] % 64);
            continue;
        }
        uint64_t total = 0, joint = 0;
        memset(uni, 0, (size_t)W * 8);
        for (uint32_t c = 0; c < nd.n; c++) {
            const uint64_t *sc = sup + (uint64_t)s->child[nd.arg + c] * W;
            for (uint64_t w = 0; w < W; w++) { total += (uint64_t)__builtin_popcountll(sc[w]); uni[w] |= sc[w]; }
        }
        for (uint64_t w = 0; w < W; w++) { joint += (uint64_t)__builtin_popcountll(uni[w]); si[w] = uni[w]; }
        const int disjoint = total == joint;
        once &= disjoint;
        double lo, hi;
        if (nd.op == TR_PROV_TIMES) {
            lo = 1.0; hi = 1.0;
            double mn = 1.0;
            for (uint32_t c = 0; c < nd.n; c++) {
                const tr_math_interval ci = iv[s->child[nd.arg + c]];
                lo = tr_mul_dir(lo, ci.lo, 0);
                hi = tr_mul_dir(hi, ci.hi, 1);
                if (ci.hi < mn) mn = ci.hi;
            }
            if (!disjoint) hi = mn;                        /* Harris-FKG: prod P_c <= P(AND) <= min P_c */
        } else {
            double qlo = 1.0, qhi = 1.0, mx = 0.0;          /* prod (1 - P_c), enclosed */
            for (uint32_t c = 0; c < nd.n; c++) {
                const tr_math_interval ci = iv[s->child[nd.arg + c]];
                qhi = tr_mul_dir(qhi, tr_add_ru(1.0, -ci.lo), 1);
                qlo = tr_mul_dir(qlo, tr_add_rd(1.0, -ci.hi), 0);
                if (ci.lo > mx) mx = ci.lo;
            }
            lo = disjoint ? tr_add_rd(1.0, -qhi) : mx;     /* Harris-FKG: max P_c <= P(OR) <= 1 - prod (1 - P_c) */
            hi = tr_add_ru(1.0, -qlo);
        }
        if (lo < 0) lo = 0.0;
        if (hi > 1) hi = 1.0;
        iv[i].lo = lo; iv[i].hi = hi;
    }
    if (!rc) { *out = iv[r]; *read_once = once; }
    free(mark); free(bit); free(sup); free(uni); free(iv);
    return rc;
}
