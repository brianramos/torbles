/* FND-006: fused multi-output basis reduction; see math_reduce.h. SPDX-License-Identifier: Apache-2.0 */
#include "math_reduce.h"
#include "mathematics.h"
#include "math_domain.h"
#include "math_handles.h"
#include <stdlib.h>
#include <string.h>

TR_MATH_ABI(tr_reduce_grouping, 1);          /* the ctypes boundary of FND-006 (math_handles.h) */
TR_MATH_ABI(tr_reduce_output, 1);
TR_MATH_ABI(tr_reduce_request, 1);
TR_MATH_ABI(tr_reduce_view, 1);
TR_MATH_ABI(tr_reduce_group_view, 1);
TR_MATH_ABI(tr_reduce_arena, 1);
TR_MATH_ABI(tr_reduce_stats, 1);
TR_MATH_ABI(tr_reduce_plan_info, 1);
TR_MATH_ABI(tr_reduce_output_info, 1);
TR_MATH_ABI(tr_reduce_force, 1);
TR_MATH_ABI(tr_reduce_evidence, 1);

_Static_assert(sizeof(tr_reduce_grouping) == 24, "tr_reduce_grouping ABI");
_Static_assert(sizeof(tr_reduce_output) == 128, "tr_reduce_output ABI");
_Static_assert(sizeof(tr_reduce_request) == 64, "tr_reduce_request ABI");
_Static_assert(sizeof(tr_reduce_term) == 8, "tr_reduce_term ABI");
_Static_assert(sizeof(tr_reduce_out) == 88, "tr_reduce_out ABI");
_Static_assert(sizeof(tr_reduce_group_view) == 32, "tr_reduce_group_view ABI");
_Static_assert(sizeof(tr_reduce_view) == 2736, "tr_reduce_view ABI");
_Static_assert(sizeof(tr_reduce_arena) == 32, "tr_reduce_arena ABI");
_Static_assert(sizeof(tr_reduce_stats) == 104, "tr_reduce_stats ABI");
_Static_assert(sizeof(tr_reduce_plan_info) == 112, "tr_reduce_plan_info ABI");
_Static_assert(sizeof(tr_reduce_output_info) == 192, "tr_reduce_output_info ABI");
_Static_assert(sizeof(tr_reduce_force) == 16, "tr_reduce_force ABI");
_Static_assert(sizeof(tr_reduce_evidence) == 72 + TR_REDUCE_CANDIDATES * sizeof(tr_reg_candidate), "evidence ABI");
_Static_assert((int)TR_REDUCE_ON_CPU == (int)TR_REG_ON_CPU && (int)TR_REDUCE_ON_CUDA == (int)TR_REG_ON_CUDA, "devices");

#define PLAN_MAGIC UINT32_C(0x50444552)
#define STAGE_MAGIC UINT32_C(0x53444552)
#define RESULT_MAGIC UINT32_C(0x52444552)
#define I128_LIMIT ((((unsigned __int128)1) << 127) - 1)

typedef struct {
    tr_dist_spec spec;          /* HIST: the normalized spec (base scope columns) */
    uint32_t cols;              /* the columns the output reads (its derived scope's columns) */
    int32_t sparse;             /* HIST sparse */
} out_extra;

struct tr_reduce_plan_s {
    uint32_t magic;
    int32_t ncols, module, merged_conds, merged_nodes, reserved;
    const tr_scope *scope;
    tr_reduce_view view;        /* pointers into this plan and its scopes */
    tr_bool_condition *conds;
    tr_bool_node *nodes;
    uint32_t *schedule, *condlist;
    tr_bank_term *products;
    tr_reduce_term *terms;
    uint8_t *exps;              /* nterms x ncols: compiled term j's exponents */
    tr_reduce_out *outs;
    tr_dist_plan *hplans;
    out_extra *extra;
    uint32_t *maps[TR_REDUCE_MAX_GROUPINGS];
    uint64_t states, bytes, term_rows, dense_bytes, scratch_bytes, decoded_bytes, sparse_scratch;
    tr_math_need need;          /* the fused CPU plan */
};

typedef struct {
    uint64_t *rows, *count, *wcount, *offset;
    tr_math_i128 *sum;
    int64_t *value[3];
    uint8_t *nulls;
    uint64_t ncells;            /* entries of count (and wcount, sum) */
    int32_t filled, owned;      /* REPEATED: put once; owned: the CSR arrays belong to the slot */
} slot;

struct tr_reduce_stage_s {
    uint32_t magic;
    int32_t kind;
    const tr_reduce_plan *plan;
    uint64_t *a64;
    tr_math_i128 *a128;
    uint64_t n64, n128, bytes;
    slot *slots;
    tr_math_budget budget;
};

typedef struct {
    tr_reduce_output_info info;
    uint64_t *rows, *count, *wcount, *offset;
    tr_math_i128 *sum;
    int64_t *value[3];
    uint8_t *nulls;
    uint64_t ncells;
} rout;

struct tr_reduce_result_s {
    uint32_t magic;
    int32_t noutputs;
    rout *outs;
    tr_reduce_stats stats;
    uint64_t bytes;
};

static void set_error(int32_t *error, int32_t code) { if (error) *error = code; }
static void set_failed(int32_t *failed, int32_t o) { if (failed) *failed = o; }
static void *zalloc(uint64_t bytes) {
    if (bytes > (uint64_t)SIZE_MAX) return NULL;
    return calloc(1, bytes ? (size_t)bytes : 1);
}
static int plan_valid(const tr_reduce_plan *p) { return p && p->magic == PLAN_MAGIC; }
static int stage_valid(const tr_reduce_stage *s) { return s && s->magic == STAGE_MAGIC; }
static int result_valid(const tr_reduce_result *r) { return r && r->magic == RESULT_MAGIC; }
static int popc32(uint32_t x) { return __builtin_popcount(x); }
static uint64_t max64(uint64_t a, uint64_t b) { return a > b ? a : b; }
static const tr_math_budget *budget_or(const tr_math_budget *b, tr_math_budget *dflt) {
    if (b) return b;
    tr_budget_default(dflt);
    return dflt;
}
static void need_zero(tr_math_need *n) { for (int i = 0; i < TR_BUDGET_RESOURCES; i++) n->need[i] = 0; }

/* ------------------------------------------------------------------ the plan */
void tr_reduce_plan_free(tr_reduce_plan *p) {
    if (!plan_valid(p)) return;
    free(p->conds); free(p->nodes); free(p->schedule); free(p->condlist); free(p->products); free(p->terms);
    free(p->exps); free(p->outs); free(p->hplans); free(p->extra);
    for (int g = 0; g < TR_REDUCE_MAX_GROUPINGS; g++) free(p->maps[g]);
    p->magic = 0;
    free(p);
}

/* the registry id of this module (FND-005), or 0 before it is registered */
static int32_t module_id(void) {
    int32_t n = tr_reg_module_count();
    for (int32_t i = 0; i < n; i++) {
        tr_reg_module m;
        if (tr_reg_module_get(i, &m) == TR_MATH_OK && !strcmp(m.key, "reduce")) return m.id;
    }
    return 0;
}

/* two scopes of exactly the same rows: the same source identity and version, row count and selection bits (array
 * scopes: the same row count and selection, aligned by position) */
static int same_rows(const tr_scope *a, const tr_scope *b) {
    tr_scope_info x, y;
    tr_scope_view u, v;
    if (tr_scope_get_info(a, &x) || tr_scope_get_info(b, &y) || tr_scope_get_view(a, &u) || tr_scope_get_view(b, &v))
        return 0;
    if (x.source != y.source || x.rows != y.rows || x.selected != y.selected) return 0;
    if (x.source == TR_SCOPE_TABLE && (x.table != y.table || x.revision != y.revision || memcmp(x.identity, y.identity, 16) ||
                                       x.version[0] != y.version[0] || x.version[1] != y.version[1])) return 0;
    uint64_t nw = (x.rows + 63) / 64;
    return !nw || !memcmp(u.selection, v.selection, (size_t)nw * 8);
}

/* the product of a compiled term: an index into the products table, hash-consed by the canonical term key */
typedef struct { uint32_t key, nf, index, used; } prod_slot;
static int32_t product_of(tr_reduce_plan *p, prod_slot *table, uint32_t cap, const tr_bank_term *t) {
    uint32_t key = tr_bank_term_key(t), h = (key * UINT32_C(2654435761)) ^ t->nf;
    for (uint32_t i = 0; i < cap; i++) {
        prod_slot *s = &table[(h + i) & (cap - 1)];
        if (!s->used) {
            if (p->view.nproducts >= TR_REDUCE_MAX_PRODUCTS) return TR_MATH_UNSUPPORTED;
            s->used = 1; s->key = key; s->nf = t->nf; s->index = (uint32_t)p->view.nproducts;
            p->products[p->view.nproducts++] = *t;
            return (int32_t)s->index;
        }
        if (s->key == key && s->nf == t->nf) return (int32_t)s->index;
    }
    return TR_MATH_UNSUPPORTED;
}

static uint64_t plan_bytes(const tr_reduce_plan *p) {
    const tr_reduce_view *v = &p->view;
    uint64_t b = sizeof(tr_reduce_plan);
    b = tr_budget_add(b, tr_budget_mul((uint64_t)v->nconds, sizeof(tr_bool_condition) + 4));
    b = tr_budget_add(b, tr_budget_mul((uint64_t)v->nnodes, sizeof(tr_bool_node) + 4));
    b = tr_budget_add(b, tr_budget_mul((uint64_t)v->nproducts, sizeof(tr_bank_term)));
    b = tr_budget_add(b, tr_budget_mul((uint64_t)v->nterms, sizeof(tr_reduce_term) + (uint64_t)p->ncols));
    b = tr_budget_add(b, tr_budget_mul((uint64_t)v->noutputs, sizeof(tr_reduce_out) + sizeof(out_extra)));
    b = tr_budget_add(b, tr_budget_mul((uint64_t)v->nhist, sizeof(tr_dist_plan)));
    for (int g = 0; g < v->ngroupings; g++)
        if (p->maps[g]) b = tr_budget_add(b, tr_budget_mul(v->groups[g].source_groups, 4));
    return b;
}
/* bytes of the CPU fused pass's scratch: condition and node words, masks and the live list, products and stamps */
static uint64_t cpu_scratch_bytes(const tr_reduce_view *v) {
    uint64_t b = tr_budget_add(tr_budget_mul((uint64_t)v->nconds, 16), tr_budget_mul((uint64_t)v->nnodes, 16));
    b = tr_budget_add(b, tr_budget_mul((uint64_t)v->noutputs, 12));
    return tr_budget_add(b, tr_budget_mul((uint64_t)v->nproducts, 24));
}

static uint64_t stage_bytes(uint64_t n64, uint64_t n128, int32_t noutputs);
/* the result's bytes but for the cells of sparse histograms (their count is known after counting) */
static uint64_t result_dense_bytes(const tr_reduce_plan *p);

tr_reduce_plan *tr_reduce_compile(const tr_scope *scope, const tr_reduce_request *rq, const tr_math_budget *budget,
                                  tr_math_refusal *refusal, int32_t *failed, int32_t *error) {
    tr_budget_none(refusal);
    set_failed(failed, -1);
    tr_scope_view sv;
    if (!scope || !rq || tr_scope_get_view(scope, &sv) || rq->reserved || rq->nconds < 0 || rq->nnodes < 0 ||
        rq->ngroupings < 1 || rq->noutputs < 0 || rq->nexps < 0 || (rq->nconds && !rq->conds) ||
        (rq->nnodes && !rq->nodes) || !rq->groupings || (rq->noutputs && !rq->outputs) ||
        (rq->nexps && sv.ncols && !rq->exps)) { set_error(error, TR_MATH_INVALID); return NULL; }
    if (rq->noutputs > TR_REDUCE_MAX_OUTPUTS || rq->ngroupings > TR_REDUCE_MAX_GROUPINGS ||
        rq->nconds > TR_REDUCE_MAX_CONDITIONS || rq->nnodes > TR_REDUCE_MAX_NODES || rq->nexps > TR_REDUCE_MAX_TERMS) {
        set_error(error, TR_MATH_UNSUPPORTED); return NULL;
    }
    tr_math_budget dflt;
    budget = budget_or(budget, &dflt);
    /* the request-sized resources first: nothing is allocated before they pass */
    tr_math_need pre;
    need_zero(&pre);
    pre.need[TR_BUDGET_CIRCUIT] = (uint64_t)rq->nconds + (uint64_t)rq->nnodes;
    pre.need[TR_BUDGET_OUTPUT] = (uint64_t)rq->noutputs;
    int32_t rc = tr_budget_check(budget, &pre, refusal);
    if (rc) { set_error(error, rc); return NULL; }

    tr_reduce_plan *p = zalloc(sizeof *p);
    if (!p) { set_error(error, TR_MATH_MEMORY); return NULL; }
    p->magic = PLAN_MAGIC;
    p->scope = scope; p->ncols = sv.ncols; p->module = module_id();
    tr_reduce_view *v = &p->view;
    v->scope = sv;
    int32_t total_terms = 0;
    for (int32_t o = 0; o < rq->noutputs; o++) {
        const tr_reduce_output *q = &rq->outputs[o];
        if (q->kind == TR_REDUCE_BANK) {
            if (q->nterms < 0 || q->first < 0 || q->first > rq->nexps || q->nterms > rq->nexps - q->first) {
                set_failed(failed, o); rc = TR_MATH_INVALID; goto fail;
            }
            if (q->nterms > TR_BANK_MAX_TERMS) { set_failed(failed, o); rc = TR_MATH_UNSUPPORTED; goto fail; }
            total_terms += q->nterms;
            if (total_terms > TR_REDUCE_MAX_TERMS) { set_failed(failed, o); rc = TR_MATH_UNSUPPORTED; goto fail; }
        }
    }
    p->conds = zalloc((uint64_t)rq->nconds * sizeof(tr_bool_condition));
    p->nodes = zalloc((uint64_t)rq->nnodes * sizeof(tr_bool_node));
    p->schedule = zalloc((uint64_t)rq->nnodes * 4);
    p->condlist = zalloc((uint64_t)rq->nconds * 4);
    p->products = zalloc((uint64_t)(total_terms < TR_REDUCE_MAX_PRODUCTS ? total_terms : TR_REDUCE_MAX_PRODUCTS) *
                         sizeof(tr_bank_term));
    p->terms = zalloc((uint64_t)total_terms * sizeof(tr_reduce_term));
    p->exps = zalloc((uint64_t)total_terms * (uint64_t)sv.ncols);
    p->outs = zalloc((uint64_t)rq->noutputs * sizeof(tr_reduce_out));
    p->hplans = zalloc((uint64_t)rq->noutputs * sizeof(tr_dist_plan));
    p->extra = zalloc((uint64_t)rq->noutputs * sizeof(out_extra));
    int32_t *cmap = zalloc((uint64_t)rq->nconds * 4), *nmap = zalloc((uint64_t)rq->nnodes * 4);
    uint32_t pcap = 16;
    while (pcap < 2 * (uint32_t)(total_terms + 1)) pcap <<= 1;
    prod_slot *ptable = zalloc((uint64_t)pcap * sizeof(prod_slot));
    uint8_t *mark = zalloc((uint64_t)rq->nnodes);
    tr_bank_term *bt = zalloc((uint64_t)TR_BANK_MAX_TERMS * sizeof(tr_bank_term));
    if (!p->conds || !p->nodes || !p->schedule || !p->condlist || !p->products || !p->terms || !p->exps || !p->outs ||
        !p->hplans || !p->extra || !cmap || !nmap || !ptable || !mark || !bt) {
        free(cmap); free(nmap); free(ptable); free(mark); free(bt);
        rc = TR_MATH_MEMORY; goto fail;
    }
    /* conditions: validated, structurally equal ones merged */
    for (int32_t i = 0; i < rq->nconds && !rc; i++) {
        const tr_bool_condition *c = &rq->conds[i];
        if (tr_bool_condition_check(&sv, c)) { rc = TR_MATH_INVALID; break; }
        int32_t at = -1;
        for (int32_t j = 0; j < v->nconds && at < 0; j++) if (!memcmp(&p->conds[j], c, sizeof *c)) at = j;
        if (at < 0) { at = v->nconds++; p->conds[at] = *c; } else p->merged_conds++;
        cmap[i] = at;
    }
    /* nodes: validated against the request's indices, hash-consed on (op, merged operands) */
    for (int32_t i = 0; i < rq->nnodes && !rc; i++) {
        tr_bool_node n = rq->nodes[i];
        if (n.reserved) { rc = TR_MATH_INVALID; break; }
        switch (n.op) {
        case TR_BOOL_CONST: if (n.a < 0 || n.a > 2 || n.b != -1) rc = TR_MATH_INVALID; break;
        case TR_BOOL_VAR: if (n.a < 0 || n.a >= rq->nconds || n.b != -1) rc = TR_MATH_INVALID; else n.a = cmap[n.a]; break;
        case TR_BOOL_NOT: if (n.a < 0 || n.a >= i || n.b != -1) rc = TR_MATH_INVALID; else n.a = nmap[n.a]; break;
        case TR_BOOL_AND: case TR_BOOL_OR: case TR_BOOL_XOR:
            if (n.a < 0 || n.a >= i || n.b < 0 || n.b >= i) rc = TR_MATH_INVALID;
            else { n.a = nmap[n.a]; n.b = nmap[n.b]; }
            break;
        default: rc = TR_MATH_INVALID;
        }
        if (rc) break;
        int32_t at = -1;
        for (int32_t j = v->nnodes - 1; j >= 0 && at < 0; j--)
            if (p->nodes[j].op == n.op && p->nodes[j].a == n.a && p->nodes[j].b == n.b) at = j;
        if (at < 0) { at = v->nnodes++; p->nodes[at] = n; } else p->merged_nodes++;
        nmap[i] = at;
    }
    /* groupings */
    for (int32_t g = 0; g < rq->ngroupings && !rc; g++) {
        const tr_reduce_grouping *q = &rq->groupings[g];
        tr_reduce_group_view *gv = &v->groups[g];
        tr_scope_view gsv = sv;
        if (q->scope && q->scope != scope) {
            if (!same_rows(scope, q->scope) || tr_scope_get_view(q->scope, &gsv)) { rc = TR_MATH_INVALID; break; }
        }
        gv->ids = gsv.groups;
        gv->source_groups = gsv.ngroups;
        if (q->map) {
            if (q->ngroups < 1 || q->ngroups > TR_SCOPE_MAX_GROUPS) { rc = TR_MATH_INVALID; break; }
            tr_math_need mn;
            need_zero(&mn);
            mn.need[TR_BUDGET_BYTES] = tr_budget_mul(gv->source_groups, 4);
            if ((rc = tr_budget_check(budget, &mn, refusal))) break;
            p->maps[g] = zalloc(gv->source_groups * 4);
            if (!p->maps[g]) { rc = TR_MATH_MEMORY; break; }
            memcpy(p->maps[g], q->map, (size_t)gv->source_groups * 4);
            for (uint64_t i = 0; i < gv->source_groups; i++) if (p->maps[g][i] >= q->ngroups) rc = TR_MATH_INVALID;
            gv->map = p->maps[g];
            gv->ngroups = q->ngroups;
        } else {
            if (q->ngroups) { rc = TR_MATH_INVALID; break; }
            gv->ngroups = gsv.ngroups;
        }
    }
    v->ngroupings = rq->ngroupings;
    /* outputs */
    uint64_t a64 = 0, a128 = 0, states = 0, maxcells = 0, sparse_scratch = 0;
    for (int32_t o = 0; o < rq->noutputs && !rc; o++) {
        const tr_reduce_output *q = &rq->outputs[o];
        tr_reduce_out *out = &p->outs[o];
        out_extra *x = &p->extra[o];
        if (q->reserved[0] || q->reserved[1] || q->filter < -1 || q->filter >= rq->nnodes || q->grouping < 0 ||
            q->grouping >= rq->ngroupings) { rc = TR_MATH_INVALID; set_failed(failed, o); break; }
        out->kind = q->kind; out->filter = q->filter < 0 ? -1 : nmap[q->filter]; out->grouping = q->grouping;
        out->hist = -1;
        out->ngroups = v->groups[q->grouping].ngroups;
        uint64_t G = out->ngroups;
        out->rows_at = a64; a64 = tr_budget_add(a64, G);
        if (q->kind == TR_REDUCE_BANK) {
            if (q->mode != TR_BANK_SQL && q->mode != TR_BANK_JOINT) { rc = TR_MATH_INVALID; set_failed(failed, o); break; }
            uint32_t joint = 0;
            int32_t degree = 0;
            rc = tr_bank_compile(rq->exps ? rq->exps + (uint64_t)q->first * (uint64_t)sv.ncols : NULL, q->nterms, sv.ncols,
                                 bt, &joint, &degree);
            if (rc) { set_failed(failed, o); break; }
            if ((rc = tr_bank_prove(&sv, bt, q->nterms, q->mode, joint))) { set_failed(failed, o); break; }
            out->mode = q->mode; out->joint = joint; out->first = v->nterms; out->nterms = q->nterms;
            if (degree > v->degree) v->degree = degree;
            for (int32_t j = 0; j < q->nterms; j++) {
                int32_t pi = product_of(p, ptable, pcap, &bt[j]);
                if (pi < 0) { rc = pi; break; }
                tr_reduce_term *t = &p->terms[v->nterms];
                t->product = (uint32_t)pi; t->need = bt[j].need;
                if (sv.ncols) memcpy(p->exps + (uint64_t)v->nterms * (uint64_t)sv.ncols,
                                     rq->exps + (uint64_t)(q->first + j) * (uint64_t)sv.ncols, (size_t)sv.ncols);
                v->nterms++;
            }
            if (rc) { set_failed(failed, o); break; }
            x->cols = joint;
            uint64_t gm = tr_budget_mul(G, (uint64_t)q->nterms);
            out->count_at = a64; a64 = tr_budget_add(a64, gm);
            out->sum_at = a128; a128 = tr_budget_add(a128, gm);
            states = tr_budget_add(states, tr_budget_add(G, gm));
            p->term_rows += (uint64_t)q->nterms;
        } else if (q->kind == TR_REDUCE_HIST) {
            if (q->mode || q->first || q->nterms) { rc = TR_MATH_INVALID; set_failed(failed, o); break; }
            tr_dist_spec ns;
            tr_dist_spec_normalize(&q->hist, &ns);
            ns.strategy = TR_DIST_STRATEGY_AUTO;
            tr_dist_plan *hp = &p->hplans[v->nhist];
            if ((rc = tr_dist_plan_make(&sv, &ns, hp))) { set_failed(failed, o); break; }
            x->spec = ns; x->sparse = ns.repr == TR_DIST_SPARSE;
            x->cols = 0;
            for (int c = 0; c < hp->k; c++) x->cols |= UINT32_C(1) << hp->col[c];
            if (hp->weight >= 0) x->cols |= UINT32_C(1) << hp->weight;
            out->hist = v->nhist++; out->cells = hp->cells;
            uint64_t gc = tr_budget_mul(G, hp->cells);
            if (hp->cells > maxcells) maxcells = hp->cells;
            out->count_at = a64; a64 = tr_budget_add(a64, gc);
            if (hp->weighted) {
                out->wcount_at = a64; a64 = tr_budget_add(a64, gc);
                out->sum_at = a128; a128 = tr_budget_add(a128, gc);
            }
            states = tr_budget_add(states, tr_budget_add(G, gc));
            if (x->sparse) sparse_scratch = tr_budget_add(sparse_scratch, tr_budget_mul(gc, hp->weighted ? 32 : 8));
        } else { rc = TR_MATH_INVALID; set_failed(failed, o); break; }
    }
    v->noutputs = rq->noutputs;
    free(cmap); free(nmap); free(ptable); free(bt);
    if (rc) { free(mark); goto fail; }
    /* the schedule: the nodes some filter reaches, increasing; the conditions they read */
    for (int32_t o = 0; o < v->noutputs; o++) if (p->outs[o].filter >= 0) mark[p->outs[o].filter] = 1;
    for (int32_t i = v->nnodes - 1; i >= 0; i--) {
        if (!mark[i]) continue;
        const tr_bool_node *n = &p->nodes[i];
        if (n->op == TR_BOOL_NOT || n->op >= TR_BOOL_AND) mark[n->a] = 1;
        if (n->op >= TR_BOOL_AND) mark[n->b] = 1;
    }
    uint64_t creads = 0;
    for (int32_t i = 0; i < v->nnodes; i++) {
        if (!mark[i]) continue;
        p->schedule[v->nsched++] = (uint32_t)i;
        if (p->nodes[i].op == TR_BOOL_VAR) creads |= UINT64_C(1) << p->nodes[i].a;
    }
    free(mark);
    uint32_t used = 0;
    for (int32_t c = 0; c < v->nconds; c++) {
        if (!((creads >> c) & 1)) continue;
        p->condlist[v->ncondlist++] = (uint32_t)c;
        used |= UINT32_C(1) << p->conds[c].col;
        if (p->conds[c].op >= TR_BOOL_COL_EQ) used |= UINT32_C(1) << p->conds[c].col2;
    }
    for (int32_t o = 0; o < v->noutputs; o++) used |= p->extra[o].cols;
    v->used = used;
    for (int32_t c = 0; c < sv.ncols; c++) if (((used >> c) & 1) && sv.col[c].valid) v->nullable |= UINT32_C(1) << c;
    v->conds = p->conds; v->nodes = p->nodes; v->schedule = p->schedule; v->condlist = p->condlist;
    v->products = p->products; v->terms = p->terms; v->outs = p->outs; v->hplans = p->hplans;
    v->arena64 = a64; v->arena128 = a128;
    p->states = states; p->sparse_scratch = sparse_scratch;
    /* decoded bytes the fused pass reads */
    uint64_t n = sv.rows, dec = tr_budget_bitmap_bytes(n);
    for (int32_t c = 0; c < sv.ncols; c++) {
        if (!((used >> c) & 1)) continue;
        dec = tr_budget_add(dec, tr_budget_mul(n, sv.col[c].values ? 8 : 4));
        if (sv.col[c].valid) dec = tr_budget_add(dec, n);
    }
    for (int32_t g = 0; g < v->ngroupings; g++) {
        int seen = 0, read = 0;
        for (int32_t o = 0; o < v->noutputs; o++) read |= p->outs[o].grouping == g;
        for (int32_t h = 0; h < g; h++) seen |= v->groups[h].ids == v->groups[g].ids;
        if (read && v->groups[g].ids && !seen) dec = tr_budget_add(dec, tr_budget_mul(n, 4));
        if (read && v->groups[g].map) dec = tr_budget_add(dec, tr_budget_mul(v->groups[g].source_groups, 4));
    }
    p->decoded_bytes = dec;
    /* the fused CPU need */
    p->dense_bytes = result_dense_bytes(p);
    p->scratch_bytes = cpu_scratch_bytes(v);
    p->bytes = plan_bytes(p);
    tr_math_need *nd = &p->need;
    need_zero(nd);
    nd->need[TR_BUDGET_VARIABLES] = (uint64_t)popc32(used);
    nd->need[TR_BUDGET_DEGREE] = (uint64_t)v->degree;
    nd->need[TR_BUDGET_SUPPORT] = tr_budget_add((uint64_t)v->nterms, maxcells);
    nd->need[TR_BUDGET_CIRCUIT] = (uint64_t)v->nconds + (uint64_t)v->nsched;
    nd->need[TR_BUDGET_OUTPUT] = (uint64_t)v->noutputs;
    nd->need[TR_BUDGET_STATES] = states;
    nd->need[TR_BUDGET_BYTES] = tr_budget_add(tr_budget_add(p->bytes, stage_bytes(a64, a128, v->noutputs)),
                                              tr_budget_add(p->dense_bytes, p->scratch_bytes));
    /* the plan itself must fit; the fused pass's states are checked when it runs (the planner rejects it over budget,
     * and the repeated plan may still be admissible) */
    tr_math_need own;
    need_zero(&own);
    own.need[TR_BUDGET_BYTES] = p->bytes;
    if ((rc = tr_budget_check(budget, &own, refusal))) goto fail;
    set_error(error, TR_MATH_OK);
    return p;
fail:
    tr_reduce_plan_free(p);
    set_error(error, rc);
    return NULL;
}

int32_t tr_reduce_plan_get_info(const tr_reduce_plan *p, tr_reduce_plan_info *out) {
    if (!plan_valid(p) || !out) return TR_MATH_INVALID;
    memset(out, 0, sizeof *out);
    const tr_reduce_view *v = &p->view;
    out->noutputs = v->noutputs; out->nconds = v->nconds; out->nnodes = v->nnodes; out->nsched = v->nsched;
    out->nproducts = v->nproducts; out->nterms = v->nterms; out->ngroupings = v->ngroupings; out->nhist = v->nhist;
    out->merged_conds = p->merged_conds; out->merged_nodes = p->merged_nodes;
    out->used = v->used; out->nullable = v->nullable;
    out->rows = v->scope.rows; out->selected = v->scope.selected; out->words = (v->scope.rows + 63) / 64;
    out->arena64 = v->arena64; out->arena128 = v->arena128; out->states = p->states; out->bytes = p->bytes;
    out->term_rows = p->term_rows;
    return TR_MATH_OK;
}
int32_t tr_reduce_get_view(const tr_reduce_plan *p, tr_reduce_view *out) {
    if (!plan_valid(p) || !out) return TR_MATH_INVALID;
    *out = p->view;
    return TR_MATH_OK;
}

/* ------------------------------------------------------------------ the repeated plan's shapes */
/* the derived columns of output o in increasing base order */
static int32_t out_columns(const tr_reduce_plan *p, int32_t o, int32_t *cols) {
    int32_t n = 0;
    for (int32_t c = 0; c < p->ncols; c++) if ((p->extra[o].cols >> c) & 1) cols[n++] = c;
    return n;
}
static int32_t derived_index(const int32_t *cols, int32_t n, int32_t c) {
    for (int32_t i = 0; i < n; i++) if (cols[i] == c) return i;
    return -1;
}
/* the cone of a filter (marks), and the conditions it reads (bit mask) */
static uint64_t filter_cone(const tr_reduce_plan *p, int32_t filter, uint8_t *mark) {
    memset(mark, 0, (size_t)p->view.nnodes);
    uint64_t reads = 0;
    if (filter < 0) return 0;
    mark[filter] = 1;
    for (int32_t i = filter; i >= 0; i--) {
        if (!mark[i]) continue;
        const tr_bool_node *n = &p->nodes[i];
        if (n->op == TR_BOOL_VAR) reads |= UINT64_C(1) << n->a;
        if (n->op == TR_BOOL_NOT || n->op >= TR_BOOL_AND) mark[n->a] = 1;
        if (n->op >= TR_BOOL_AND) mark[n->b] = 1;
    }
    return reads;
}
/* The shape of output o's derived scope S_o as a view for the reductions' own need formulas (math_bank.h,
 * math_dist.h), which read only its sizes and which arrays exist, never the arrays: the base scope's rows and
 * selected rows (an upper bound of S_o's), o's groups (per-row ids iff G > 1), every column stored as int64 values
 * with a validity array where the base scope has a selected NULL (tr_reduce_derive copies them so). The pointers are
 * markers (the view itself): this view is never read through. */
static void derived_shape(const tr_reduce_plan *p, int32_t o, tr_scope_view *dv) {
    *dv = p->view.scope;
    const void *mark = dv;
    dv->ngroups = p->outs[o].ngroups;
    dv->groups = dv->ngroups > 1 ? (const uint32_t *)mark : NULL;
    for (int32_t c = 0; c < dv->ncols; c++) { dv->col[c].codes = NULL; dv->col[c].values = (const int64_t *)mark; }
}
#define TR_REDUCE_OBJECT_SLACK UINT64_C(16384)   /* the fixed objects of one reduction (builders and scopes hold views) */
/* An upper bound of every check output o's own reduction makes in the repeated plan (tr_reduce_derive, BOOL-001
 * masks, FND-004 / DIST-001 builds; device: their CUDA builds and adoptions; a sparse histogram's occupied cells
 * bounded by min(G x cells, selected rows)): variables, degree, support and states as the largest single check,
 * bytes as everything the output allocates at once. *copies: the bytes its device builds copy (device) */
static void repeated_output_need(const tr_reduce_plan *p, int32_t o, int device, tr_math_need *nd, uint64_t *copies) {
    const tr_reduce_view *v = &p->view;
    const tr_reduce_out *out = &p->outs[o];
    const uint64_t n = v->scope.rows, nw = (n + 63) / 64, G = out->ngroups, S = v->scope.selected;
    const uint64_t bm = tr_budget_bitmap_bytes(n), ids = G > 1 ? tr_budget_mul(n, 4) : 0;
    int32_t cols[TR_SCOPE_MAX_COLUMNS];
    const uint64_t nc = (uint64_t)out_columns(p, o, cols);
    need_zero(nd);
    if (copies) *copies = 0;
    /* tr_reduce_derive: its temporaries (checked) and the derived scope (selection, values, validity, ids) */
    uint64_t bytes = tr_budget_add(tr_budget_add(tr_budget_mul(nw, 8), tr_budget_mul(n, nc * 9)),
                                   tr_budget_add(ids, tr_budget_mul((uint64_t)v->nnodes, 17)));
    bytes = tr_budget_add(bytes, tr_budget_add(tr_budget_add(TR_REDUCE_OBJECT_SLACK, bm),
                                               tr_budget_add(tr_budget_mul(n, nc * 9), ids)));
    nd->need[TR_BUDGET_STATES] = G;
    if (out->filter >= 0) {                   /* BOOL-001 masks of the k conditions of its filter's cone */
        uint8_t *mark = zalloc((uint64_t)v->nnodes);
        uint64_t k = mark ? (uint64_t)__builtin_popcountll(filter_cone(p, out->filter, mark)) : 64;
        free(mark);
        nd->need[TR_BUDGET_VARIABLES] = k;
        bytes = tr_budget_add(bytes, tr_budget_add(TR_REDUCE_OBJECT_SLACK, tr_budget_mul(tr_budget_mul(nw, 8), 1 + 2 * k)));
    }
    tr_scope_view dv;
    derived_shape(p, o, &dv);
    tr_math_need r;
    if (out->kind == TR_REDUCE_BANK) {
        const uint64_t m = (uint64_t)out->nterms;
        int32_t degree = 0;
        for (uint64_t j = 0; j < m; j++) {
            int32_t d = v->products[v->terms[out->first + j].product].degree;
            if (d > degree) degree = d;
        }
        /* tr_bank_build: the builder (its states, coverage bitmap, basis) becoming the bank; device: the device copies,
         * then tr_bank_from_states adopting a bank of the same size */
        uint64_t bank = tr_budget_add(tr_budget_add(TR_REDUCE_OBJECT_SLACK, tr_bank_state_bytes(G, out->nterms)),
                                      tr_budget_add(bm, tr_budget_mul(m, 17 + nc)));
        if (device) {
            tr_bank_device_need(&dv, out->nterms, out->mode, out->joint, degree, &r);
            if (copies) *copies = r.need[TR_BUDGET_BYTES];
            bank = tr_budget_add(bank, r.need[TR_BUDGET_BYTES]);
        }
        nd->need[TR_BUDGET_VARIABLES] = max64(nd->need[TR_BUDGET_VARIABLES], (uint64_t)popc32(p->extra[o].cols));
        nd->need[TR_BUDGET_DEGREE] = (uint64_t)degree;
        nd->need[TR_BUDGET_SUPPORT] = m;
        nd->need[TR_BUDGET_STATES] = max64(G, tr_budget_mul(G, m));
        nd->need[TR_BUDGET_BYTES] = tr_budget_add(bytes, bank);
        return;
    }
    const tr_dist_plan *hp = &p->hplans[out->hist];
    const uint64_t cb = tr_dist_cell_bytes(hp->weighted), total = tr_budget_mul(G, hp->cells);
    uint64_t hist, states, support = 0;
    nd->need[TR_BUDGET_VARIABLES] = max64(nd->need[TR_BUDGET_VARIABLES], (uint64_t)(hp->k + hp->weighted));
    if (device) {
        /* tr_mgpu_dist_hist_build over S_o's shape (dense: the state; sparse: its counting scratch, the output capacity,
         * the compaction's chunk counters) and tr_dist_hist_from_states adopting at most that capacity */
        uint64_t cap = 0;
        tr_dist_device_need(&dv, hp, &r, &cap);
        if (copies) *copies = r.need[TR_BUDGET_BYTES];
        uint64_t per = p->extra[o].sparse ? tr_budget_add(cb, (uint64_t)hp->k * 8 + 1) : cb;
        hist = tr_budget_add(r.need[TR_BUDGET_BYTES],
                             tr_budget_add(tr_budget_add(TR_REDUCE_OBJECT_SLACK, tr_budget_mul(G + 1, 16)),
                                           tr_budget_mul(p->extra[o].sparse ? cap : total, per)));
        support = r.need[TR_BUDGET_SUPPORT];
        states = r.need[TR_BUDGET_STATES];
    } else if (!p->extra[o].sparse) {         /* counted in place over the explicit domain */
        hist = tr_budget_add(tr_budget_add(TR_REDUCE_OBJECT_SLACK, bm), tr_budget_add(tr_budget_mul(G, 8), tr_budget_mul(total, cb)));
        support = hp->cells;
        states = total;
    } else {
        /* DIST-001 chooses per S_o: COUNT (a dense scratch of G x cells_o <= 4 S + 65536 cells) or SORT (2 x 64 bytes per
         * selected row); then its sparse state of at most min(G x cells, S) occupied cells, min(cells, S) per group */
        uint64_t scratch = tr_budget_add(tr_budget_mul(S, TR_DIST_COUNT_FACTOR), TR_DIST_COUNT_SLACK);
        if (total < scratch) scratch = total;
        uint64_t occupied = total < S ? total : S;
        hist = tr_budget_add(tr_budget_add(TR_REDUCE_OBJECT_SLACK, bm),
                             tr_budget_add(tr_budget_mul(G, 24), max64(tr_budget_mul(scratch, cb), tr_budget_mul(S, 128))));
        hist = tr_budget_add(hist, tr_budget_add(tr_budget_mul(G + 1, 8),
                                                 tr_budget_mul(occupied, tr_budget_add(cb, (uint64_t)hp->k * 8 + 1))));
        support = hp->cells < S ? hp->cells : S;
        states = max64(scratch, occupied);
    }
    nd->need[TR_BUDGET_SUPPORT] = support;
    nd->need[TR_BUDGET_STATES] = max64(G, states);
    nd->need[TR_BUDGET_BYTES] = tr_budget_add(bytes, hist);
}
/* the repeated stage's arena: rows of every output, states of banks and dense histograms (sparse ones are put as
 * CSR copies) */
static void repeated_arena(const tr_reduce_plan *p, uint64_t *n64, uint64_t *n128, uint64_t *states) {
    uint64_t a = 0, b = 0, st = 0;
    for (int32_t o = 0; o < p->view.noutputs; o++) {
        const tr_reduce_out *out = &p->outs[o];
        uint64_t G = out->ngroups;
        a = tr_budget_add(a, G); st = tr_budget_add(st, G);
        if (out->kind == TR_REDUCE_BANK) {
            uint64_t gm = tr_budget_mul(G, (uint64_t)out->nterms);
            a = tr_budget_add(a, gm); b = tr_budget_add(b, gm); st = tr_budget_add(st, gm);
        } else if (!p->extra[o].sparse) {
            uint64_t gc = tr_budget_mul(G, out->cells);
            a = tr_budget_add(a, gc); st = tr_budget_add(st, gc);
            if (p->hplans[out->hist].weighted) { a = tr_budget_add(a, gc); b = tr_budget_add(b, gc); }
        }
    }
    *n64 = a; *n128 = b;
    if (states) *states = st;
}

static uint64_t stage_bytes(uint64_t n64, uint64_t n128, int32_t noutputs) {
    return tr_budget_add(tr_budget_add(tr_budget_mul(n64, 8), tr_budget_mul(n128, 16)),
                         tr_budget_mul((uint64_t)noutputs, sizeof(slot)));
}
static uint64_t result_dense_bytes(const tr_reduce_plan *p) {
    uint64_t b = tr_budget_add(sizeof(tr_reduce_result), tr_budget_mul((uint64_t)p->view.noutputs, sizeof(rout)));
    for (int32_t o = 0; o < p->view.noutputs; o++) {
        const tr_reduce_out *out = &p->outs[o];
        uint64_t G = out->ngroups;
        if (out->kind == TR_REDUCE_BANK) { b = tr_budget_add(b, tr_bank_state_bytes(G, out->nterms)); continue; }
        b = tr_budget_add(b, tr_budget_mul(G, 8));
        if (p->extra[o].sparse) b = tr_budget_add(b, tr_budget_mul(G + 1, 8));
        else b = tr_budget_add(b, tr_budget_mul(tr_budget_mul(G, out->cells), tr_dist_cell_bytes(p->hplans[out->hist].weighted)));
    }
    return b;
}

int32_t tr_reduce_need(const tr_reduce_plan *p, int32_t target, tr_math_need *need) {
    if (!plan_valid(p) || !need) return TR_MATH_INVALID;
    const tr_reduce_view *v = &p->view;
    if (target == TR_REDUCE_NEED_FUSED) { *need = p->need; return TR_MATH_OK; }
    if (target == TR_REDUCE_NEED_FUSED_KERNEL) { tr_reduce_device_need(v, &p->need, need); return TR_MATH_OK; }
    if (target == TR_REDUCE_NEED_FUSED_DEVICE) {
        tr_reduce_device_need(v, &p->need, need);
        /* plus the host side: the plan, the stage, the adopted result's dense part */
        need->need[TR_BUDGET_BYTES] = tr_budget_add(need->need[TR_BUDGET_BYTES],
                                                    tr_budget_add(tr_budget_add(p->bytes, stage_bytes(v->arena64, v->arena128, v->noutputs)),
                                                                  p->dense_bytes));
        return TR_MATH_OK;
    }
    if (target != TR_REDUCE_NEED_REPEATED && target != TR_REDUCE_NEED_REPEATED_DEVICE) return TR_MATH_INVALID;
    /* the stage and the result (every output's rows, bank states, dense histograms), held while each output's own
     * reduction runs: the largest of those (repeated_output_need); circuit as the fused plan (the compiled predicate
     * DAG, evaluated per word), output = outputs */
    need_zero(need);
    uint64_t a, b, st = 0, top = 0;
    repeated_arena(p, &a, &b, &st);
    tr_math_need one;
    for (int32_t o = 0; o < v->noutputs; o++) {
        repeated_output_need(p, o, target == TR_REDUCE_NEED_REPEATED_DEVICE, &one, NULL);
        for (int r = TR_BUDGET_VARIABLES; r <= TR_BUDGET_SUPPORT; r++) need->need[r] = max64(need->need[r], one.need[r]);
        top = max64(top, one.need[TR_BUDGET_STATES]);
        need->need[TR_BUDGET_BYTES] = max64(need->need[TR_BUDGET_BYTES], one.need[TR_BUDGET_BYTES]);
    }
    need->need[TR_BUDGET_CIRCUIT] = p->need.need[TR_BUDGET_CIRCUIT];
    need->need[TR_BUDGET_OUTPUT] = (uint64_t)v->noutputs;
    need->need[TR_BUDGET_STATES] = tr_budget_add(st, top);
    need->need[TR_BUDGET_BYTES] = tr_budget_add(tr_budget_add(p->bytes, stage_bytes(a, b, v->noutputs)),
                                                tr_budget_add(p->dense_bytes, need->need[TR_BUDGET_BYTES]));
    return TR_MATH_OK;
}

/* ------------------------------------------------------------------ costs and the choice (FND-008 conventions) */
static const double default_rates[2][2][TR_COST_COMPONENTS] = {
    /* words  scan  chain combine construct reuse fusion materialize transfer cpu_final gpu_final fixed */
    {{1.0,   1.0,  1.0,  1.0,    2.0,      1.0,  1.0,   0.25,       0.0,     1.0,      0.0,      4096.0},    /* fused, CPU */
     {0.02,  0.02, 0.02, 1.0,    0.1,      1.0,  1.0,   0.25,       0.25,    1.0,      0.02,     50000.0}},  /* fused, CUDA */
    {{1.0,   1.0,  1.0,  1.0,    2.0,      1.0,  1.0,   0.25,       0.0,     1.0,      0.0,      4096.0},    /* repeated, CPU */
     {0.02,  0.02, 0.02, 1.0,    0.1,      1.0,  1.0,   0.25,       0.25,    1.0,      0.02,     50000.0}}}; /* repeated, CUDA */
int32_t tr_reduce_coefficients(double *out, int32_t capacity) {
    if (!out || capacity < TR_REDUCE_COEFFICIENTS) return TR_MATH_INVALID;
    memcpy(out, default_rates, sizeof default_rates);
    return TR_MATH_OK;
}
static int word_is(const char *s, size_t n, const char *w) { return strlen(w) == n && !strncmp(s, w, n); }
int32_t tr_reduce_force_parse(const char *text, tr_reduce_force *f) {
    if (!f) return TR_MATH_INVALID;
    memset(f, 0, sizeof *f);
    f->device = -1;
    if (!text) return TR_MATH_OK;
    const char *s = text;
    while (*s == ' ') s++;
    const char *e = strchr(s, ',');
    size_t n = e ? (size_t)(e - s) : strlen(s);
    while (n && s[n - 1] == ' ') n--;
    if (!n) return TR_MATH_OK;
    size_t at = n;
    for (size_t i = 0; i < n; i++) if (s[i] == '@') { at = i; break; }
    if (word_is(s, at, "fused")) f->plan = TR_REDUCE_FUSED;
    else if (word_is(s, at, "repeated")) f->plan = TR_REDUCE_REPEATED;
    else return TR_MATH_INVALID;
    if (at < n) {
        if (word_is(s + at + 1, n - at - 1, "cpu")) f->device = TR_REDUCE_ON_CPU;
        else if (word_is(s + at + 1, n - at - 1, "cuda")) f->device = TR_REDUCE_ON_CUDA;
        else return TR_MATH_INVALID;
    }
    return TR_MATH_OK;
}
/* the exact work amounts of a candidate, per FND-008 component */
static void amounts(const tr_reduce_plan *p, int32_t plan, int32_t device, tr_reg_candidate *c) {
    const tr_reduce_view *v = &p->view;
    uint64_t n = v->scope.rows, S = v->scope.selected, W = (n + 63) / 64, O = (uint64_t)v->noutputs;
    uint64_t *a = c->amount;
    for (int i = 0; i < TR_COST_COMPONENTS; i++) a[i] = 0;
    uint64_t groupings = 0;
    for (int32_t g = 0; g < v->ngroupings; g++) groupings += v->groups[g].ids != NULL;
    uint64_t factors = 0;
    for (int32_t j = 0; j < v->nproducts; j++) factors += v->products[j].nf ? v->products[j].nf : 1;
    uint64_t hist = (uint64_t)v->nhist, arena = tr_budget_add(tr_budget_mul(v->arena64, 8), tr_budget_mul(v->arena128, 16));
    a[TR_COST_MATERIALIZE] = p->dense_bytes;
    if (plan == TR_REDUCE_FUSED) {
        uint64_t batch = O ? O : 1;
        c->batch = (int32_t)batch;
        c->shared = (UINT32_C(1) << TR_COST_WORDS) | (UINT32_C(1) << TR_COST_SCAN);
        a[TR_COST_WORDS] = tr_budget_mul(W, batch);
        a[TR_COST_SCAN] = tr_budget_mul(tr_budget_mul(S, (uint64_t)popc32(v->used) + groupings), batch);
        a[TR_COST_CHAIN] = tr_budget_add(tr_budget_mul(S, tr_budget_add((uint64_t)v->ncondlist + factors + hist,
                                                                        (uint64_t)v->nterms)),
                                         tr_budget_mul(W, (uint64_t)v->nsched));
        a[TR_COST_CONSTRUCT] = tr_budget_add(v->arena64, v->arena128);
        a[TR_COST_CPU_FINAL] = p->sparse_scratch / 8;
        if (device == TR_REDUCE_ON_CUDA) {
            tr_math_need dn;
            tr_reduce_device_need(v, &p->need, &dn);
            a[TR_COST_TRANSFER] = tr_budget_add(dn.need[TR_BUDGET_BYTES], arena);
            a[TR_COST_FIXED] = 1 + 24;                       /* one launch, the copies in and out */
        } else a[TR_COST_FIXED] = 1;
        return;
    }
    c->batch = 1; c->shared = 0;
    uint8_t *mark = zalloc((uint64_t)v->nnodes);
    for (int32_t o = 0; o < v->noutputs; o++) {
        const tr_reduce_out *out = &p->outs[o];
        int32_t cols[TR_SCOPE_MAX_COLUMNS];
        uint64_t nc = (uint64_t)out_columns(p, o, cols), passes = 2, k = 0, cone = 0;
        if (out->filter >= 0 && mark) {
            uint64_t reads = filter_cone(p, out->filter, mark);
            k = (uint64_t)__builtin_popcountll(reads);
            for (int32_t i = 0; i < v->nnodes; i++) cone += mark[i];
            passes = 3;
        }
        uint64_t terms = out->kind == TR_REDUCE_BANK ? (uint64_t)out->nterms : 1, G = out->ngroups;
        a[TR_COST_WORDS] = tr_budget_add(a[TR_COST_WORDS], tr_budget_mul(W, passes));
        a[TR_COST_SCAN] = tr_budget_add(a[TR_COST_SCAN], tr_budget_add(tr_budget_mul(S, tr_budget_add(k, nc + 1)),
                                                                       tr_budget_mul(n, nc + 1)));
        a[TR_COST_CHAIN] = tr_budget_add(a[TR_COST_CHAIN], tr_budget_add(tr_budget_mul(S, tr_budget_add(k, terms * 2)),
                                                                         tr_budget_mul(W, cone)));
        uint64_t st = out->kind == TR_REDUCE_BANK ? tr_budget_mul(G, (uint64_t)out->nterms * 2 + 1)
                                                  : tr_budget_mul(G, tr_budget_add(out->cells, 1));
        a[TR_COST_CONSTRUCT] = tr_budget_add(a[TR_COST_CONSTRUCT], tr_budget_add(st, tr_budget_mul(n, nc)));
        if (device == TR_REDUCE_ON_CUDA) {             /* the device copies of its FND-004 / DIST-001 build */
            tr_math_need one;
            uint64_t copies = 0;
            repeated_output_need(p, o, 1, &one, &copies);
            a[TR_COST_TRANSFER] = tr_budget_add(a[TR_COST_TRANSFER], copies);
            a[TR_COST_FIXED] = tr_budget_add(a[TR_COST_FIXED], 4 + 12);
        } else a[TR_COST_FIXED] = tr_budget_add(a[TR_COST_FIXED], 4);
    }
    free(mark);
}
static void cost_of(tr_reg_candidate *c, const double *coef) {
    int32_t pl = c->representation - 1, dv = c->device == TR_REDUCE_ON_CUDA ? 1 : 0;
    const double *r = coef ? coef + ((size_t)pl * 2 + (size_t)dv) * TR_COST_COMPONENTS : default_rates[pl][dv];
    double total = 0, shared = 0;
    for (int i = 0; i < TR_COST_COMPONENTS; i++) {
        if (i == TR_COST_FUSION) continue;
        c->cost[i] = (double)c->amount[i] * r[i];
        if ((c->shared >> i) & 1) shared += c->cost[i];
        total += c->cost[i];
    }
    c->cost[TR_COST_FUSION] = c->batch > 1 ? -r[TR_COST_FUSION] * shared * (double)(c->batch - 1) / (double)c->batch : 0.0;
    c->total = total + c->cost[TR_COST_FUSION];
}
int32_t tr_reduce_choose(const tr_reduce_plan *p, int32_t devices, const tr_math_budget *budget, const double *coef,
                         const tr_reduce_force *force, tr_reduce_evidence *ev) {
    if (!ev) return TR_MATH_INVALID;
    memset(ev, 0, sizeof *ev);
    ev->abi = TR_REDUCE_ABI; ev->chosen = -1; ev->force.device = -1;
    tr_budget_none(&ev->refusal);
    if (!plan_valid(p) || (devices & ~(TR_REDUCE_ON_CPU | TR_REDUCE_ON_CUDA))) {
        ev->code = TR_MATH_INVALID; ev->reason = TR_PLANNER_R_SEARCH; return TR_MATH_INVALID;
    }
    tr_reduce_force f;
    int32_t rc = force ? (f = *force, TR_MATH_OK) : tr_reduce_force_parse(getenv("TORBLES_MATH_REDUCE_FORCE"), &f);
    if (!rc && (f.plan < 0 || f.plan > TR_REDUCE_REPEATED || (f.device != -1 && f.device != TR_REDUCE_ON_CPU &&
                                                            f.device != TR_REDUCE_ON_CUDA))) rc = TR_MATH_INVALID;
    if (rc) { ev->code = rc; ev->reason = TR_PLANNER_R_SEARCH; return rc; }
    if (f.plan && f.device == TR_REDUCE_ON_CUDA) devices |= TR_REDUCE_ON_CUDA;
    if (!devices) devices = TR_REDUCE_ON_CPU;
    ev->force = f; ev->forced = f.plan != 0; ev->devices = devices; ev->module = p->module;
    tr_math_budget dflt;
    budget = budget_or(budget, &dflt);
    int32_t n = 0, best = -1, fbest = -1, first_match = -1, first_budget = -1;
    for (int32_t plan = TR_REDUCE_FUSED; plan <= TR_REDUCE_REPEATED; plan++)
        for (int32_t dev = TR_REDUCE_ON_CPU; dev <= TR_REDUCE_ON_CUDA; dev <<= 1) {
            if (!(devices & dev)) continue;
            tr_reg_candidate *c = &ev->candidates[n];
            memset(c, 0, sizeof *c);
            c->module = p->module; c->representation = plan; c->device = dev; c->finalize = 0; c->reuse = 0;
            c->valid = 1;
            tr_reg_refusal_none(&c->refusal);
            tr_reduce_need(p, plan == TR_REDUCE_FUSED ? (dev == TR_REDUCE_ON_CUDA ? TR_REDUCE_NEED_FUSED_DEVICE : TR_REDUCE_NEED_FUSED)
                                                      : (dev == TR_REDUCE_ON_CUDA ? TR_REDUCE_NEED_REPEATED_DEVICE : TR_REDUCE_NEED_REPEATED),
                           &c->need);
            amounts(p, plan, dev, c);
            cost_of(c, coef);
            if (tr_budget_check(budget, &c->need, &c->budget)) {
                c->status = TR_PLANNER_S_REJECTED; c->reason = TR_PLANNER_R_BUDGET; c->code = TR_MATH_BUDGET;
                c->refusal.kind = TR_REG_REFUSE_BUDGET; c->refusal.code = TR_MATH_BUDGET;
                if (first_budget < 0) first_budget = n;
            } else { c->status = TR_PLANNER_S_ADMISSIBLE; c->reason = TR_PLANNER_R_COSTLIER; }
            int match = f.plan && f.plan == plan && (f.device < 0 || f.device == dev);
            if (match && first_match < 0) first_match = n;
            if (c->status == TR_PLANNER_S_ADMISSIBLE) {
                if (best < 0 || c->total < ev->candidates[best].total) best = n;
                if (match && (fbest < 0 || c->total < ev->candidates[fbest].total)) fbest = n;
            }
            n++;
        }
    ev->ncandidates = n;
    int32_t chosen = f.plan ? fbest : best, code = TR_MATH_OK;
    if (chosen >= 0) {
        ev->candidates[chosen].status = TR_PLANNER_S_CHOSEN;
        ev->candidates[chosen].reason = f.plan ? TR_PLANNER_R_FORCED : TR_PLANNER_R_CHEAPEST;
        for (int32_t i = 0; f.plan && i < n; i++)
            if (i != chosen && ev->candidates[i].status == TR_PLANNER_S_ADMISSIBLE) ev->candidates[i].reason = TR_PLANNER_R_NOT_FORCED;
        ev->reason = ev->candidates[chosen].reason;
    } else if (f.plan && first_match < 0) { code = TR_MATH_UNSUPPORTED; ev->reason = TR_PLANNER_R_FORCE_UNMATCHED; }
    else if (f.plan) {
        code = ev->candidates[first_match].code ? ev->candidates[first_match].code : TR_MATH_UNSUPPORTED;
        ev->reason = TR_PLANNER_R_FORCE_REFUSED;
        if (code == TR_MATH_BUDGET) ev->refusal = ev->candidates[first_match].budget;
    } else {
        ev->reason = TR_PLANNER_R_NONE_ADMISSIBLE;
        code = first_budget >= 0 ? TR_MATH_BUDGET : TR_MATH_UNSUPPORTED;
        if (first_budget >= 0) ev->refusal = ev->candidates[first_budget].budget;
    }
    ev->chosen = chosen; ev->code = code;
    return code;
}

/* ------------------------------------------------------------------ stages */
void tr_reduce_stage_free(tr_reduce_stage *s) {
    if (!stage_valid(s)) return;
    if (s->slots)
        for (int32_t o = 0; o < s->plan->view.noutputs; o++) {
            slot *x = &s->slots[o];
            if (!x->owned) continue;
            free(x->offset); free(x->count); free(x->wcount); free(x->sum); free(x->nulls);
            for (int c = 0; c < 3; c++) free(x->value[c]);
        }
    free(s->slots); free(s->a64); free(s->a128);
    s->magic = 0;
    free(s);
}
tr_reduce_stage *tr_reduce_stage_create(const tr_reduce_plan *p, int32_t kind, const tr_math_budget *budget,
                                        tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    if (!plan_valid(p) || (kind != TR_REDUCE_FUSED && kind != TR_REDUCE_REPEATED)) { set_error(error, TR_MATH_INVALID); return NULL; }
    tr_math_budget dflt;
    budget = budget_or(budget, &dflt);
    const tr_reduce_view *v = &p->view;
    uint64_t n64 = v->arena64, n128 = v->arena128, states = p->states;
    if (kind == TR_REDUCE_REPEATED) repeated_arena(p, &n64, &n128, &states);
    tr_math_need nd;
    need_zero(&nd);
    nd.need[TR_BUDGET_STATES] = states;
    nd.need[TR_BUDGET_BYTES] = stage_bytes(n64, n128, v->noutputs);
    int32_t rc = tr_budget_check(budget, &nd, refusal);
    if (rc) { set_error(error, rc); return NULL; }
    if (n64 == TR_BUDGET_UNKNOWN || n128 == TR_BUDGET_UNKNOWN) { set_error(error, TR_MATH_BUDGET); return NULL; }
    tr_reduce_stage *s = zalloc(sizeof *s);
    if (!s) { set_error(error, TR_MATH_MEMORY); return NULL; }
    s->magic = STAGE_MAGIC; s->kind = kind; s->plan = p; s->budget = *budget;
    s->n64 = n64; s->n128 = n128;
    s->a64 = zalloc(n64 * 8); s->a128 = zalloc(n128 * 16);
    s->slots = zalloc((uint64_t)v->noutputs * sizeof(slot));
    s->bytes = nd.need[TR_BUDGET_BYTES];
    if (!s->a64 || !s->a128 || !s->slots) { tr_reduce_stage_free(s); set_error(error, TR_MATH_MEMORY); return NULL; }
    uint64_t a = 0, b = 0;
    for (int32_t o = 0; o < v->noutputs; o++) {
        const tr_reduce_out *out = &p->outs[o];
        slot *x = &s->slots[o];
        uint64_t G = out->ngroups;
        if (kind == TR_REDUCE_FUSED) {
            x->rows = s->a64 + out->rows_at;
            x->count = s->a64 + out->count_at;
            x->ncells = out->kind == TR_REDUCE_BANK ? G * (uint64_t)out->nterms : G * out->cells;
            if (out->kind == TR_REDUCE_BANK || p->hplans[out->hist].weighted) x->sum = s->a128 + out->sum_at;
            if (out->kind == TR_REDUCE_HIST && p->hplans[out->hist].weighted) x->wcount = s->a64 + out->wcount_at;
            continue;
        }
        x->rows = s->a64 + a; a += G;
        if (out->kind == TR_REDUCE_BANK) {
            x->ncells = G * (uint64_t)out->nterms;
            x->count = s->a64 + a; a += x->ncells;
            x->sum = s->a128 + b; b += x->ncells;
        } else if (!p->extra[o].sparse) {
            x->ncells = G * out->cells;
            x->count = s->a64 + a; a += x->ncells;
            if (p->hplans[out->hist].weighted) { x->wcount = s->a64 + a; a += x->ncells; x->sum = s->a128 + b; b += x->ncells; }
        }
    }
    set_error(error, TR_MATH_OK);
    return s;
}
int32_t tr_reduce_stage_arena(tr_reduce_stage *s, tr_reduce_arena *out) {
    if (!stage_valid(s) || !out || s->kind != TR_REDUCE_FUSED) return TR_MATH_INVALID;
    out->a64 = s->a64; out->a128 = s->a128; out->n64 = s->n64; out->n128 = s->n128;
    return TR_MATH_OK;
}

/* the basis of output o over its derived columns */
static void derived_exps(const tr_reduce_plan *p, int32_t o, const int32_t *cols, int32_t nc, uint8_t *out) {
    const tr_reduce_out *x = &p->outs[o];
    for (int32_t j = 0; j < x->nterms; j++)
        for (int32_t i = 0; i < nc; i++)
            out[(uint64_t)j * (uint64_t)nc + (uint64_t)i] = p->exps[(uint64_t)(x->first + j) * (uint64_t)p->ncols + (uint64_t)cols[i]];
}
int32_t tr_reduce_stage_put_bank(tr_reduce_stage *s, int32_t o, const tr_bank *bank) {
    if (!stage_valid(s) || s->kind != TR_REDUCE_REPEATED || o < 0 || o >= s->plan->view.noutputs) return TR_MATH_INVALID;
    const tr_reduce_plan *p = s->plan;
    const tr_reduce_out *out = &p->outs[o];
    slot *x = &s->slots[o];
    tr_bank_info bi;
    tr_bank_states st;
    if (out->kind != TR_REDUCE_BANK || x->filled || tr_bank_get_info(bank, &bi) || tr_bank_get_states(bank, &st) ||
        bi.ngroups != out->ngroups || bi.nterms != out->nterms || bi.mode != out->mode) return TR_MATH_INVALID;
    int32_t cols[TR_SCOPE_MAX_COLUMNS];
    int32_t nc = out_columns(p, o, cols);
    if (bi.ncols != nc) return TR_MATH_INVALID;
    uint64_t sz = (uint64_t)out->nterms * (uint64_t)nc;
    uint8_t *want = zalloc(sz), *have = zalloc(sz);
    int bad = !want || !have;
    if (!bad) {
        derived_exps(p, o, cols, nc, want);
        bad = tr_bank_get_exponents(bank, have, sz) != TR_MATH_OK || (sz && memcmp(want, have, (size_t)sz));
    }
    free(want); free(have);
    if (bad) return TR_MATH_INVALID;
    memcpy(x->rows, st.rows, (size_t)out->ngroups * 8);
    if (x->ncells) { memcpy(x->count, st.count, (size_t)x->ncells * 8); memcpy(x->sum, st.sum, (size_t)x->ncells * 16); }
    x->filled = 1;
    return TR_MATH_OK;
}
int32_t tr_reduce_stage_put_hist(tr_reduce_stage *s, int32_t o, const tr_dist_hist *h, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!stage_valid(s) || s->kind != TR_REDUCE_REPEATED || o < 0 || o >= s->plan->view.noutputs) return TR_MATH_INVALID;
    const tr_reduce_plan *p = s->plan;
    const tr_reduce_out *out = &p->outs[o];
    slot *x = &s->slots[o];
    tr_dist_states st;
    if (out->kind != TR_REDUCE_HIST || x->filled || tr_dist_hist_get_states(h, &st)) return TR_MATH_INVALID;
    const tr_dist_plan *hp = &p->hplans[out->hist];
    if (st.ngroups != out->ngroups || st.k != hp->k || st.weighted != hp->weighted ||
        st.repr != (p->extra[o].sparse ? TR_DIST_SPARSE : TR_DIST_DENSE)) return TR_MATH_INVALID;
    if (!p->extra[o].sparse) {
        if (st.cells != out->cells || st.ncells != x->ncells) return TR_MATH_INVALID;
        for (int c = 0; c < hp->k; c++) if (st.lo[c] != hp->lo[c] || st.span[c] != hp->span[c] || st.dim[c] != hp->dim[c]) return TR_MATH_INVALID;
        memcpy(x->rows, st.rows, (size_t)out->ngroups * 8);
        if (x->ncells) memcpy(x->count, st.count, (size_t)x->ncells * 8);
        if (hp->weighted && x->ncells) { memcpy(x->wcount, st.wcount, (size_t)x->ncells * 8); memcpy(x->sum, st.wsum, (size_t)x->ncells * 16); }
        x->filled = 1;
        return TR_MATH_OK;
    }
    uint64_t nc = st.ncells, G = out->ngroups;
    uint64_t add = tr_budget_add(tr_budget_mul(G + 1, 8), tr_budget_mul(nc, tr_budget_add((uint64_t)hp->k * 8 + 1,
                                                                                          tr_dist_cell_bytes(hp->weighted))));
    /* the copy of its occupied cells, beside the stage, the plan and the copies already put */
    if (tr_budget_check_one(&s->budget, TR_BUDGET_BYTES, tr_budget_add(tr_budget_add(s->bytes, add), p->bytes), refusal))
        return TR_MATH_BUDGET;
    x->offset = zalloc((G + 1) * 8); x->count = zalloc(nc * 8); x->nulls = zalloc(nc);
    x->owned = 1;
    int bad = !x->offset || !x->count || !x->nulls;
    for (int c = 0; c < hp->k && !bad; c++) bad = !(x->value[c] = zalloc(nc * 8));
    if (!bad && hp->weighted) bad = !(x->wcount = zalloc(nc * 8)) || !(x->sum = zalloc(nc * 16));
    if (bad) return TR_MATH_MEMORY;
    memcpy(x->rows, st.rows, (size_t)G * 8);
    memcpy(x->offset, st.offset, (size_t)(G + 1) * 8);
    if (nc) {
        memcpy(x->count, st.count, (size_t)nc * 8);
        memcpy(x->nulls, st.nulls, (size_t)nc);
        for (int c = 0; c < hp->k; c++) memcpy(x->value[c], st.value[c], (size_t)nc * 8);
        if (hp->weighted) { memcpy(x->wcount, st.wcount, (size_t)nc * 8); memcpy(x->sum, st.wsum, (size_t)nc * 16); }
    }
    x->ncells = nc;
    s->bytes = tr_budget_add(s->bytes, add);
    x->filled = 1;
    return TR_MATH_OK;
}

/* the consistency of one staged output (see the header comment): 0 or TR_MATH_INVALID */
static int32_t check_output(const tr_reduce_plan *p, int32_t o, const slot *x, int sparse_csr) {
    const tr_reduce_view *v = &p->view;
    const tr_reduce_out *out = &p->outs[o];
    uint64_t G = out->ngroups, total = 0;
    for (uint64_t g = 0; g < G; g++) if (__builtin_add_overflow(total, x->rows[g], &total)) return TR_MATH_INVALID;
    if (total > v->scope.selected) return TR_MATH_INVALID;
    if (out->kind == TR_REDUCE_BANK) {
        const uint64_t m = (uint64_t)out->nterms;
        for (uint64_t j = 0; j < m; j++) {
            const tr_reduce_term *t = &v->terms[out->first + j];
            const tr_bank_term *pr = &v->products[t->product];
            unsigned __int128 bound = tr_bank_term_bound(&v->scope, pr);
            for (uint64_t g = 0; g < G; g++) {
                uint64_t c = x->count[g * m + j], c0 = x->count[g * m];
                __int128 s = tr_i128_get(x->sum[g * m + j]);
                unsigned __int128 mag = s < 0 ? (unsigned __int128)0 - (unsigned __int128)s : (unsigned __int128)s;
                if (c > x->rows[g] || (!c && s) || mag > tr_bank_satmul(bound, c, I128_LIMIT)) return TR_MATH_INVALID;
                if (!pr->nf && (s != (__int128)c || (out->mode == TR_BANK_SQL && c != x->rows[g]))) return TR_MATH_INVALID;
                if (out->mode == TR_BANK_JOINT && c != c0) return TR_MATH_INVALID;
            }
        }
        return TR_MATH_OK;
    }
    const tr_dist_plan *hp = &v->hplans[out->hist];
    uint64_t wmax = 0;
    if (hp->weighted) {
        const tr_scope_view_column *w = &v->scope.col[hp->weight];
        wmax = w->nvalid ? (tr_abs_u64(w->min) > tr_abs_u64(w->max) ? tr_abs_u64(w->min) : tr_abs_u64(w->max)) : 0;
    }
    for (uint64_t g = 0; g < G; g++) {
        uint64_t b = sparse_csr ? x->offset[g] : g * out->cells, e = sparse_csr ? x->offset[g + 1] : (g + 1) * out->cells;
        if (e < b || e > x->ncells || (sparse_csr && g == 0 && b) || (sparse_csr && g == G - 1 && e != x->ncells))
            return TR_MATH_INVALID;
        uint64_t sum = 0;
        for (uint64_t i = b; i < e; i++) {
            if (__builtin_add_overflow(sum, x->count[i], &sum)) return TR_MATH_INVALID;
            if (sparse_csr && !x->count[i]) return TR_MATH_INVALID;
            if (hp->weighted) {
                __int128 s = tr_i128_get(x->sum[i]);
                if (x->wcount[i] > x->count[i] || (!x->wcount[i] && s) ||
                    tr_dist_mag128(s) > (unsigned __int128)x->wcount[i] * wmax) return TR_MATH_INVALID;
            }
        }
        if (sum != x->rows[g]) return TR_MATH_INVALID;
    }
    return TR_MATH_OK;
}

void tr_reduce_result_free(tr_reduce_result *r) {
    if (!result_valid(r)) return;
    for (int32_t o = 0; o < r->noutputs; o++) {
        rout *x = &r->outs[o];
        free(x->rows); free(x->count); free(x->wcount); free(x->offset); free(x->sum); free(x->nulls);
        for (int c = 0; c < 3; c++) free(x->value[c]);
    }
    free(r->outs);
    r->magic = 0;
    free(r);
}
static void *dup_bytes(const void *src, uint64_t bytes) {
    void *d = zalloc(bytes);
    if (d && bytes && src) memcpy(d, src, (size_t)bytes);
    return d;
}

tr_reduce_result *tr_reduce_stage_finish(tr_reduce_stage *s, int32_t device, const tr_reduce_stats *stats,
                                         const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *failed,
                                         int32_t *error) {
    tr_budget_none(refusal);
    set_failed(failed, -1);
    if (!stage_valid(s) || (device != TR_REDUCE_ON_CPU && device != TR_REDUCE_ON_CUDA)) {
        tr_reduce_stage_free(s); set_error(error, TR_MATH_INVALID); return NULL;
    }
    const tr_reduce_plan *p = s->plan;
    const tr_reduce_view *v = &p->view;
    tr_math_budget dflt;
    budget = budget_or(budget, &dflt);
    int32_t rc = TR_MATH_OK;
    /* every output's consistency, and the occupied cells of the sparse ones counted in the scratch */
    uint64_t bytes = sizeof(tr_reduce_result) + (uint64_t)v->noutputs * sizeof(rout), final_states = 0;
    for (int32_t o = 0; o < v->noutputs && !rc; o++) {
        const tr_reduce_out *out = &p->outs[o];
        slot *x = &s->slots[o];
        int csr = out->kind == TR_REDUCE_HIST && p->extra[o].sparse && s->kind == TR_REDUCE_REPEATED;
        if (s->kind == TR_REDUCE_REPEATED && !x->filled) { rc = TR_MATH_INVALID; set_failed(failed, o); break; }
        if ((rc = check_output(p, o, x, csr))) { set_failed(failed, o); break; }
        uint64_t G = out->ngroups, cb = 8;
        if (out->kind == TR_REDUCE_BANK) {
            bytes = tr_budget_add(bytes, tr_bank_state_bytes(G, out->nterms));
            final_states = tr_budget_add(final_states, tr_budget_add(G, tr_budget_mul(G, (uint64_t)out->nterms)));
            continue;
        }
        const tr_dist_plan *hp = &v->hplans[out->hist];
        cb = tr_dist_cell_bytes(hp->weighted);
        uint64_t cells = x->ncells;
        if (p->extra[o].sparse && !csr) { cells = 0; for (uint64_t i = 0; i < x->ncells; i++) cells += x->count[i] != 0; }
        final_states = tr_budget_add(final_states, tr_budget_add(G, cells));
        if (p->extra[o].sparse) {
            cb = tr_budget_add(cb, (uint64_t)hp->k * 8 + 1);
            bytes = tr_budget_add(bytes, tr_budget_mul(G + 1, 8));
        }
        bytes = tr_budget_add(bytes, tr_budget_add(tr_budget_mul(G, 8), tr_budget_mul(cells, cb)));
    }
    if (!rc) {                          /* the final states (sparse cells known only now), beside the stage */
        tr_math_need nd;
        need_zero(&nd);
        nd.need[TR_BUDGET_STATES] = final_states;
        nd.need[TR_BUDGET_BYTES] = tr_budget_add(tr_budget_add(bytes, s->bytes), p->bytes);
        rc = tr_budget_check(budget, &nd, refusal);
    }
    tr_reduce_result *r = rc ? NULL : zalloc(sizeof *r);
    if (!rc && !r) rc = TR_MATH_MEMORY;
    if (!rc) {
        r->magic = RESULT_MAGIC; r->noutputs = v->noutputs;
        r->outs = zalloc((uint64_t)v->noutputs * sizeof(rout));
        if (!r->outs) rc = TR_MATH_MEMORY;
    }
    for (int32_t o = 0; o < v->noutputs && !rc; o++) {
        const tr_reduce_out *out = &p->outs[o];
        slot *x = &s->slots[o];
        rout *y = &r->outs[o];
        tr_reduce_output_info *in = &y->info;
        uint64_t G = out->ngroups;
        in->kind = out->kind; in->mode = out->mode; in->grouping = out->grouping; in->filter = out->filter;
        in->ngroups = G; in->nterms = out->nterms;
        for (int c = 0; c < 3; c++) { in->col[c] = -1; }
        in->weight = -1;
        y->rows = dup_bytes(x->rows, G * 8);
        if (!y->rows) { rc = TR_MATH_MEMORY; break; }
        for (uint64_t g = 0; g < G; g++) { in->rows += x->rows[g]; in->present += x->rows[g] != 0; }
        if (out->kind == TR_REDUCE_BANK) {
            y->ncells = x->ncells; in->cells = (uint64_t)out->nterms;
            y->count = dup_bytes(x->count, x->ncells * 8); y->sum = dup_bytes(x->sum, x->ncells * 16);
            if (!y->count || !y->sum) { rc = TR_MATH_MEMORY; break; }
            in->state_bytes = tr_bank_state_bytes(G, out->nterms);
            continue;
        }
        const tr_dist_plan *hp = &v->hplans[out->hist];
        in->k = hp->k; in->weighted = hp->weighted; in->repr = p->extra[o].sparse ? TR_DIST_SPARSE : TR_DIST_DENSE;
        in->wscale = hp->wscale; in->weight = hp->weight;
        for (int c = 0; c < hp->k; c++) {
            in->lo[c] = hp->lo[c]; in->span[c] = hp->span[c]; in->dim[c] = hp->dim[c];
            in->nullslot[c] = hp->nullslot[c]; in->scale[c] = hp->scale[c]; in->col[c] = hp->col[c];
        }
        if (!p->extra[o].sparse) {
            y->ncells = x->ncells; in->cells = out->cells;
            y->count = dup_bytes(x->count, x->ncells * 8);
            if (!y->count) { rc = TR_MATH_MEMORY; break; }
            if (hp->weighted) {
                y->wcount = dup_bytes(x->wcount, x->ncells * 8); y->sum = dup_bytes(x->sum, x->ncells * 16);
                if (!y->wcount || !y->sum) { rc = TR_MATH_MEMORY; break; }
            }
            in->state_bytes = tr_budget_add(tr_budget_mul(G, 8), tr_budget_mul(x->ncells, tr_dist_cell_bytes(hp->weighted)));
            continue;
        }
        /* sparse: CSR, from the scratch (fused) or the copies (repeated) */
        uint64_t nc = 0;
        if (s->kind == TR_REDUCE_REPEATED) nc = x->ncells;
        else for (uint64_t i = 0; i < x->ncells; i++) nc += x->count[i] != 0;
        y->ncells = nc; in->cells = nc;
        y->offset = zalloc((G + 1) * 8); y->count = zalloc(nc * 8); y->nulls = zalloc(nc);
        if (!y->offset || !y->count || !y->nulls) { rc = TR_MATH_MEMORY; break; }
        for (int c = 0; c < hp->k && !rc; c++) if (!(y->value[c] = zalloc(nc * 8))) rc = TR_MATH_MEMORY;
        if (!rc && hp->weighted && (!(y->wcount = zalloc(nc * 8)) || !(y->sum = zalloc(nc * 16)))) rc = TR_MATH_MEMORY;
        if (rc) break;
        if (s->kind == TR_REDUCE_REPEATED) {
            memcpy(y->offset, x->offset, (size_t)(G + 1) * 8);
            if (nc) {
                memcpy(y->count, x->count, (size_t)nc * 8); memcpy(y->nulls, x->nulls, (size_t)nc);
                for (int c = 0; c < hp->k; c++) memcpy(y->value[c], x->value[c], (size_t)nc * 8);
                if (hp->weighted) { memcpy(y->wcount, x->wcount, (size_t)nc * 8); memcpy(y->sum, x->sum, (size_t)nc * 16); }
            }
        } else {
            uint64_t at = 0;
            for (uint64_t g = 0; g < G; g++) {
                y->offset[g] = at;
                for (uint64_t j = 0; j < out->cells; j++) {
                    uint64_t i = g * out->cells + j;
                    if (!x->count[i]) continue;
                    uint8_t nl = 0;
                    for (int c = 0; c < hp->k; c++) {
                        int isnull = 0;
                        y->value[c][at] = tr_dist_plan_coord(hp, j, c, &isnull);
                        if (isnull) nl |= (uint8_t)(1u << c);
                    }
                    y->nulls[at] = nl; y->count[at] = x->count[i];
                    if (hp->weighted) { y->wcount[at] = x->wcount[i]; y->sum[at] = x->sum[i]; }
                    at++;
                }
            }
            y->offset[G] = at;
        }
        in->state_bytes = tr_budget_add(tr_budget_mul(G + 1, 16),
                                        tr_budget_mul(nc, (uint64_t)hp->k * 8 + 1 + tr_dist_cell_bytes(hp->weighted)));
    }
    if (rc) {
        tr_reduce_result_free(r);
        tr_reduce_stage_free(s);
        set_error(error, rc);
        return NULL;
    }
    if (stats) r->stats = *stats;
    r->stats.device = device;
    r->stats.plan = s->kind;
    r->bytes = bytes;
    r->stats.state_bytes = 0;
    for (int32_t o = 0; o < v->noutputs; o++) r->stats.state_bytes = tr_budget_add(r->stats.state_bytes, r->outs[o].info.state_bytes);
    tr_reduce_stage_free(s);
    set_error(error, TR_MATH_OK);
    return r;
}

/* ------------------------------------------------------------------ the fused pass on the CPU */
static void fused_cpu(const tr_reduce_plan *p, tr_reduce_stage *s, tr_reduce_stats *st, uint64_t *cw, uint64_t *nwd,
                      int32_t *live, uint64_t *masks, __int128 *pval, uint64_t *pstamp) {
    const tr_reduce_view *v = &p->view;
    const uint64_t n = v->scope.rows, nw = (n + 63) / 64;
    uint64_t *A = s->a64;
    tr_math_i128 *B = s->a128;
    for (uint64_t w = 0; w < nw; w++) {
        uint64_t sel = v->scope.selection[w];
        if (!sel) continue;
        st->words_visited++;
        tr_reduce_eval_word(v, w, cw, nwd);
        int32_t nlive = 0;
        uint64_t any = 0;
        for (int32_t o = 0; o < v->noutputs; o++) {
            uint64_t m = tr_reduce_mask(v, &v->outs[o], sel, nwd);
            if (m) { live[nlive] = o; masks[nlive] = m; nlive++; any |= m; }
        }
        while (any) {
            int b = __builtin_ctzll(any);
            any &= any - 1;
            uint64_t r = (w << 6) | (uint64_t)b, stamp = r + 1;
            uint32_t pm = tr_reduce_presence(v, r);
            st->rows_visited++;
            for (int32_t i = 0; i < nlive; i++) {
                if (!((masks[i] >> b) & 1)) continue;
                const tr_reduce_out *o = &v->outs[live[i]];
                uint64_t g = tr_reduce_gid(v, o->grouping, r);
                A[o->rows_at + g]++;
                if (o->kind == TR_REDUCE_BANK) {
                    const uint64_t m = (uint64_t)o->nterms;
                    for (uint64_t j = 0; j < m; j++) {
                        const tr_reduce_term *t = &v->terms[o->first + j];
                        if (!tr_reduce_term_valid(o, t, pm)) continue;
                        if (pstamp[t->product] != stamp) {
                            pval[t->product] = tr_bank_term_value(&v->scope, &v->products[t->product], r);
                            pstamp[t->product] = stamp;
                            st->products++;
                        }
                        uint64_t k = g * m + j;
                        A[o->count_at + k]++;
                        B[o->sum_at + k] = tr_i128_set(tr_i128_get(B[o->sum_at + k]) + pval[t->product]);
                        st->accumulations++;
                    }
                } else {
                    const tr_dist_plan *hp = &v->hplans[o->hist];
                    uint64_t k = g * o->cells + tr_dist_row_cell(&v->scope, hp, r);
                    A[o->count_at + k]++;
                    st->accumulations++;
                    if (hp->weighted && ((pm >> hp->weight) & 1)) {
                        A[o->wcount_at + k]++;
                        B[o->sum_at + k] = tr_i128_set(tr_i128_get(B[o->sum_at + k]) + tr_scope_value(&v->scope, hp->weight, r));
                    }
                }
            }
        }
    }
}

static tr_reduce_result *run_fused(const tr_reduce_plan *p, const tr_math_budget *budget, tr_math_refusal *refusal,
                                   int32_t *failed, int32_t *error) {
    const tr_reduce_view *v = &p->view;
    int32_t rc = tr_budget_check(budget, &p->need, refusal);
    if (rc) { set_error(error, rc); return NULL; }
    tr_reduce_stage *s = tr_reduce_stage_create(p, TR_REDUCE_FUSED, budget, refusal, error);
    if (!s) return NULL;
    uint64_t *cw = zalloc((uint64_t)v->nconds * 16), *nwd = zalloc((uint64_t)v->nnodes * 16);
    int32_t *live = zalloc((uint64_t)v->noutputs * 4);
    uint64_t *masks = zalloc((uint64_t)v->noutputs * 8), *pstamp = zalloc((uint64_t)v->nproducts * 8);
    __int128 *pval = zalloc((uint64_t)v->nproducts * 16);
    tr_reduce_stats st;
    memset(&st, 0, sizeof st);
    if (!cw || !nwd || !live || !masks || !pstamp || !pval) rc = TR_MATH_MEMORY;
    else if (v->scope.rows) fused_cpu(p, s, &st, cw, nwd, live, masks, pval, pstamp);
    free(cw); free(nwd); free(live); free(masks); free(pstamp); free(pval);
    if (rc) { tr_reduce_stage_free(s); set_error(error, rc); return NULL; }
    st.scans = 1; st.native_calls = 1; st.decoded_bytes = p->decoded_bytes; st.workspace_bytes = p->scratch_bytes;
    return tr_reduce_stage_finish(s, TR_REDUCE_ON_CPU, &st, budget, refusal, failed, error);
}

/* ------------------------------------------------------------------ derived scopes and the repeated plan */
tr_scope *tr_reduce_derive(const tr_reduce_plan *p, int32_t o, uint8_t *exps, uint64_t capacity, tr_dist_spec *spec,
                           int32_t *ncols, const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    if (!plan_valid(p) || o < 0 || o >= p->view.noutputs) { set_error(error, TR_MATH_INVALID); return NULL; }
    const tr_reduce_view *v = &p->view;
    const tr_reduce_out *out = &p->outs[o];
    tr_math_budget dflt;
    budget = budget_or(budget, &dflt);
    int32_t cols[TR_SCOPE_MAX_COLUMNS];
    int32_t nc = out_columns(p, o, cols);
    if (out->kind == TR_REDUCE_BANK && exps && capacity < (uint64_t)out->nterms * (uint64_t)nc) {
        set_error(error, TR_MATH_INVALID); return NULL;
    }
    const uint64_t n = v->scope.rows, nw = (n + 63) / 64, G = out->ngroups;
    /* the temporaries: the derived selection, values, validity and ids (the scope copies them) */
    tr_math_need nd;
    need_zero(&nd);
    nd.need[TR_BUDGET_BYTES] = tr_budget_add(tr_budget_add(tr_budget_mul(nw, 8), tr_budget_mul(n, (uint64_t)nc * 9)),
                                             tr_budget_add(G > 1 ? tr_budget_mul(n, 4) : 0, tr_budget_mul((uint64_t)v->nnodes, 17)));
    int32_t rc = tr_budget_check(budget, &nd, refusal);
    if (rc) { set_error(error, rc); return NULL; }
    uint64_t *sel = zalloc(nw * 8), *val = zalloc((uint64_t)v->nnodes * 16);
    uint8_t *mark = zalloc((uint64_t)v->nnodes);
    if (!sel || !val || !mark) { free(sel); free(val); free(mark); set_error(error, TR_MATH_MEMORY); return NULL; }
    if (out->filter < 0) memcpy(sel, v->scope.selection, (size_t)nw * 8);
    else {
        /* the filter on BOOL-001 masks of the conditions of its cone */
        uint64_t reads = filter_cone(p, out->filter, mark);
        tr_bool_condition sub[TR_REDUCE_MAX_CONDITIONS];
        int32_t remap[TR_REDUCE_MAX_CONDITIONS], k = 0;
        for (int32_t c = 0; c < v->nconds; c++) {
            remap[c] = -1;
            if ((reads >> c) & 1) { remap[c] = k; sub[k++] = p->conds[c]; }
        }
        tr_bool_masks *mk = NULL;
        tr_bool_domain d;
        memset(&d, 0, sizeof d);
        if (k) {
            mk = tr_bool_masks_build(p->scope, sub, k, budget, refusal, &rc);
            if (!mk || (rc = tr_bool_masks_get_domain(mk, &d))) {
                if (!rc) rc = TR_MATH_MEMORY;
                tr_bool_masks_free(mk); free(sel); free(val); free(mark); set_error(error, rc); return NULL;
            }
        }
        for (uint64_t w = 0; w < nw; w++) {
            uint64_t s = v->scope.selection[w];
            if (!s) continue;
            for (int32_t i = 0; i <= out->filter; i++) {
                if (!mark[i]) continue;
                const tr_bool_node *nd2 = &p->nodes[i];
                uint64_t x, y;
                if (nd2->op == TR_BOOL_CONST) tr_bool_const_word(nd2->a, &x, &y);
                else if (nd2->op == TR_BOOL_VAR) tr_bool_var_word(&d, remap[nd2->a], w, &x, &y);
                else tr_bool_apply(nd2->op, val[2 * (uint64_t)nd2->a], val[2 * (uint64_t)nd2->a + 1],
                                   nd2->b >= 0 ? val[2 * (uint64_t)nd2->b] : 0, nd2->b >= 0 ? val[2 * (uint64_t)nd2->b + 1] : 0, &x, &y);
                val[2 * (uint64_t)i] = x; val[2 * (uint64_t)i + 1] = y;
            }
            sel[w] = s & val[2 * (uint64_t)out->filter];
        }
        tr_bool_masks_free(mk);
    }
    free(val); free(mark);
    /* the derived columns and ids */
    tr_scope_array arr[TR_SCOPE_MAX_COLUMNS];
    int64_t *vals[TR_SCOPE_MAX_COLUMNS];
    uint8_t *flags[TR_SCOPE_MAX_COLUMNS];
    uint32_t *ids = G > 1 ? zalloc(n * 4) : NULL;
    int bad = G > 1 && !ids;
    for (int32_t i = 0; i < nc; i++) {
        vals[i] = zalloc(n * 8);
        flags[i] = v->scope.col[cols[i]].valid ? zalloc(n) : NULL;
        bad |= !vals[i] || (v->scope.col[cols[i]].valid && !flags[i]);
    }
    tr_scope *d = NULL;
    if (!bad) {
        for (uint64_t w = 0; w < nw; w++) {
            uint64_t bits = sel[w];
            while (bits) {
                uint64_t r = (w << 6) | (uint64_t)__builtin_ctzll(bits);
                bits &= bits - 1;
                for (int32_t i = 0; i < nc; i++) {
                    int present = tr_scope_present(&v->scope, cols[i], r);
                    if (flags[i]) flags[i][r] = (uint8_t)present;
                    if (present) vals[i][r] = tr_scope_value(&v->scope, cols[i], r);
                }
                if (ids) ids[r] = tr_reduce_gid(v, out->grouping, r);
            }
        }
        for (int32_t i = 0; i < nc; i++) {
            arr[i].values = vals[i]; arr[i].valid = flags[i]; arr[i].scale = v->scope.col[cols[i]].scale; arr[i].reserved = 0;
        }
        d = tr_scope_from_arrays(n, sel, arr, nc, ids, G, budget, refusal, &rc);
    } else rc = TR_MATH_MEMORY;
    for (int32_t i = 0; i < nc; i++) { free(vals[i]); free(flags[i]); }
    free(ids); free(sel);
    if (!d) { set_error(error, rc ? rc : TR_MATH_MEMORY); return NULL; }
    if (ncols) *ncols = nc;
    if (out->kind == TR_REDUCE_BANK && exps) derived_exps(p, o, cols, nc, exps);
    if (spec) {
        memset(spec, 0, sizeof *spec);
        if (out->kind == TR_REDUCE_HIST) {
            const tr_dist_plan *hp = &v->hplans[out->hist];
            spec->k = hp->k; spec->weight = hp->weight >= 0 ? derived_index(cols, nc, hp->weight) : -1;
            spec->repr = p->extra[o].sparse ? TR_DIST_SPARSE : TR_DIST_DENSE;
            spec->strategy = TR_DIST_STRATEGY_AUTO;
            for (int c = 0; c < 3; c++) spec->col[c] = c < hp->k ? derived_index(cols, nc, hp->col[c]) : -1;
            if (!p->extra[o].sparse) {
                spec->domain = TR_DIST_DOMAIN_EXPLICIT;
                for (int c = 0; c < hp->k; c++) {
                    spec->lo[c] = hp->lo[c];
                    spec->hi[c] = (int64_t)((uint64_t)hp->lo[c] + hp->span[c] - 1);
                    spec->nullslot[c] = hp->nullslot[c];
                }
            }
        }
    }
    set_error(error, TR_MATH_OK);
    return d;
}

static tr_reduce_result *run_repeated(const tr_reduce_plan *p, const tr_math_budget *budget, tr_math_refusal *refusal,
                                      int32_t *failed, int32_t *error) {
    const tr_reduce_view *v = &p->view;
    tr_math_need nd;
    tr_reduce_need(p, TR_REDUCE_NEED_REPEATED, &nd);
    int32_t rc = tr_budget_check(budget, &nd, refusal);
    if (rc) { set_error(error, rc); return NULL; }
    tr_reduce_stage *s = tr_reduce_stage_create(p, TR_REDUCE_REPEATED, budget, refusal, error);
    if (!s) return NULL;
    tr_reduce_stats st;
    memset(&st, 0, sizeof st);
    uint8_t *exps = zalloc((uint64_t)TR_BANK_MAX_TERMS * TR_SCOPE_MAX_COLUMNS);
    if (!exps) rc = TR_MATH_MEMORY;
    for (int32_t o = 0; o < v->noutputs && !rc; o++) {
        const tr_reduce_out *out = &p->outs[o];
        tr_dist_spec spec;
        int32_t nc = 0;
        tr_scope *d = tr_reduce_derive(p, o, exps, (uint64_t)TR_BANK_MAX_TERMS * TR_SCOPE_MAX_COLUMNS, &spec, &nc,
                                       budget, refusal, &rc);
        if (!d) { set_failed(failed, o); break; }
        st.scans += out->filter >= 0 ? 3 : 2;
        st.native_calls += out->filter >= 0 ? 3 : 2;
        tr_scope_view dv;
        tr_scope_get_view(d, &dv);
        st.decoded_bytes = tr_budget_add(st.decoded_bytes, tr_budget_mul(dv.rows, (uint64_t)nc * 9));
        if (out->kind == TR_REDUCE_BANK) {
            tr_bank *b = tr_bank_build(d, exps, out->nterms, out->mode, budget, refusal, &rc);
            if (b) { rc = tr_reduce_stage_put_bank(s, o, b); tr_bank_free(b); }
        } else {
            tr_dist_hist *h = tr_dist_hist_build(d, &spec, budget, refusal, &rc);
            if (h) { rc = tr_reduce_stage_put_hist(s, o, h, refusal); tr_dist_hist_free(h); }
        }
        st.rows_visited = tr_budget_add(st.rows_visited, dv.selected);
        st.words_visited = tr_budget_add(st.words_visited, (dv.rows + 63) / 64 * (out->filter >= 0 ? 3 : 2));
        tr_scope_free(d);
        if (rc) set_failed(failed, o);
    }
    free(exps);
    if (rc) { tr_reduce_stage_free(s); set_error(error, rc); return NULL; }
    st.workspace_bytes = nd.need[TR_BUDGET_BYTES];
    return tr_reduce_stage_finish(s, TR_REDUCE_ON_CPU, &st, budget, refusal, failed, error);
}

tr_reduce_result *tr_reduce_run(const tr_reduce_plan *p, int32_t plan, const tr_math_budget *budget,
                                tr_math_refusal *refusal, int32_t *failed, int32_t *error) {
    tr_budget_none(refusal);
    set_failed(failed, -1);
    if (!plan_valid(p) || (plan != TR_REDUCE_FUSED && plan != TR_REDUCE_REPEATED)) { set_error(error, TR_MATH_INVALID); return NULL; }
    tr_math_budget dflt;
    budget = budget_or(budget, &dflt);
    return plan == TR_REDUCE_FUSED ? run_fused(p, budget, refusal, failed, error) : run_repeated(p, budget, refusal, failed, error);
}

/* ------------------------------------------------------------------ results */
int32_t tr_reduce_result_get_stats(const tr_reduce_result *r, tr_reduce_stats *out) {
    if (!result_valid(r) || !out) return TR_MATH_INVALID;
    *out = r->stats;
    return TR_MATH_OK;
}
int32_t tr_reduce_result_output(const tr_reduce_result *r, int32_t o, tr_reduce_output_info *out) {
    if (!result_valid(r) || !out || o < 0 || o >= r->noutputs) return TR_MATH_INVALID;
    *out = r->outs[o].info;
    return TR_MATH_OK;
}
int32_t tr_reduce_result_bank(const tr_reduce_result *r, int32_t o, tr_bank_states *out) {
    if (!result_valid(r) || !out || o < 0 || o >= r->noutputs || r->outs[o].info.kind != TR_REDUCE_BANK) return TR_MATH_INVALID;
    const rout *x = &r->outs[o];
    out->ngroups = x->info.ngroups; out->nterms = x->info.nterms; out->reserved = 0;
    out->rows = x->rows; out->count = x->count; out->sum = x->sum;
    return TR_MATH_OK;
}
int32_t tr_reduce_result_hist(const tr_reduce_result *r, int32_t o, tr_dist_states *out) {
    if (!result_valid(r) || !out || o < 0 || o >= r->noutputs || r->outs[o].info.kind != TR_REDUCE_HIST) return TR_MATH_INVALID;
    const rout *x = &r->outs[o];
    memset(out, 0, sizeof *out);
    out->ngroups = x->info.ngroups; out->ncells = x->ncells; out->capacity = x->ncells;
    out->k = x->info.k; out->repr = x->info.repr; out->weighted = x->info.weighted;
    out->cells = x->info.repr == TR_DIST_DENSE ? x->info.cells : 0;
    for (int c = 0; c < x->info.k; c++) { out->lo[c] = x->info.lo[c]; out->span[c] = x->info.span[c]; out->dim[c] = x->info.dim[c]; }
    out->rows = x->rows; out->offset = x->offset; out->nulls = x->nulls; out->count = x->count;
    out->wcount = x->wcount; out->wsum = x->sum;
    for (int c = 0; c < 3; c++) out->value[c] = x->value[c];
    return TR_MATH_OK;
}
static int same_words(const void *a, const void *b, uint64_t bytes) {
    if (!bytes) return 1;
    if (!a || !b) return 0;
    return !memcmp(a, b, (size_t)bytes);
}
int32_t tr_reduce_result_equal(const tr_reduce_result *a, const tr_reduce_result *b) {
    if (!result_valid(a) || !result_valid(b)) return TR_MATH_INVALID;
    if (a->noutputs != b->noutputs) return 0;
    for (int32_t o = 0; o < a->noutputs; o++) {
        const rout *x = &a->outs[o], *y = &b->outs[o];
        const tr_reduce_output_info *i = &x->info, *j = &y->info;
        if (i->kind != j->kind || i->ngroups != j->ngroups || i->nterms != j->nterms || i->mode != j->mode ||
            i->k != j->k || i->repr != j->repr || i->weighted != j->weighted || x->ncells != y->ncells ||
            i->cells != j->cells) return 0;
        uint64_t G = i->ngroups, nc = x->ncells;
        if (!same_words(x->rows, y->rows, G * 8) || !same_words(x->count, y->count, nc * 8)) return 0;
        if ((i->kind == TR_REDUCE_BANK || i->weighted) && !same_words(x->sum, y->sum, nc * 16)) return 0;
        if (i->weighted && !same_words(x->wcount, y->wcount, nc * 8)) return 0;
        if (i->kind == TR_REDUCE_HIST && i->repr == TR_DIST_SPARSE) {
            if (!same_words(x->offset, y->offset, (G + 1) * 8) || !same_words(x->nulls, y->nulls, nc)) return 0;
            for (int c = 0; c < i->k; c++) if (!same_words(x->value[c], y->value[c], nc * 8)) return 0;
        }
        if (i->kind == TR_REDUCE_HIST && i->repr == TR_DIST_DENSE)
            for (int c = 0; c < i->k; c++) if (i->lo[c] != j->lo[c] || i->span[c] != j->span[c] || i->dim[c] != j->dim[c]) return 0;
    }
    return 1;
}
