/* PROV-001: factorized provenance payloads (a hash-consed DAG, K-relation operators, B / N / N[X] interpretations);
 * see math_prov.h.
 * SPDX-License-Identifier: Apache-2.0 */
#include "math_prov.h"
#include "mathematics.h"
#include "math_domain.h"
#include "math_handles.h"
#include <stdlib.h>
#include <string.h>

TR_MATH_ABI(tr_prov_node, TR_PROV_ABI);              /* the ctypes boundary of PROV-001 (math_handles.h) */
TR_MATH_ABI(tr_prov_pred, TR_PROV_ABI);
TR_MATH_ABI(tr_prov_valuations, TR_PROV_ABI);
TR_MATH_ABI(tr_prov_plan_view, TR_PROV_ABI);
TR_MATH_ABI(tr_prov_store_info, TR_PROV_ABI);
TR_MATH_ABI(tr_prov_rel_info, TR_PROV_ABI);
TR_MATH_ABI(tr_prov_poly_info, TR_PROV_ABI);
TR_MATH_ABI(tr_prov_eval_info, TR_PROV_ABI);

_Static_assert(sizeof(tr_prov_node) == 16, "tr_prov_node ABI");
_Static_assert(sizeof(tr_prov_pred) == 24, "tr_prov_pred ABI");
_Static_assert(sizeof(tr_prov_valuations) == 40, "tr_prov_valuations ABI");
_Static_assert(sizeof(tr_prov_plan_view) == 56, "tr_prov_plan_view ABI");
_Static_assert(sizeof(tr_prov_store_info) == 80, "tr_prov_store_info ABI");
_Static_assert(sizeof(tr_prov_rel_info) == 48, "tr_prov_rel_info ABI");
_Static_assert(sizeof(tr_prov_poly_info) == 32, "tr_prov_poly_info ABI");
_Static_assert(sizeof(tr_prov_eval_info) == 40, "tr_prov_eval_info ABI");

#define STORE_MAGIC UINT32_C(0x56524f53)
#define REL_MAGIC UINT32_C(0x4c455250)
#define PLAN_MAGIC UINT32_C(0x4e414c50)
#define POLY_MAGIC UINT32_C(0x58504f4c)
#define NONE32 UINT32_MAX

struct tr_prov_store_s {
    uint32_t magic;
    uint64_t nnodes, ncap, nchild, ccap, hcap, nvars, vcap, nsources, shared, hits;
    tr_prov_node *node;
    uint32_t *parents, *child, *hash, *var_node, *var_source;
    uint64_t *var_row;
    uint64_t limit_nodes, limit_children, limit_bytes;
};
struct tr_prov_rel_s {
    uint32_t magic;
    int32_t ncols, global, op;
    const tr_prov_store *store;
    uint64_t store_nodes, ntuples, new_nodes, bytes;
    int64_t *values;
    uint8_t *valid;
    uint32_t *ann;
    int32_t *scale;
};
struct tr_prov_plan_s { uint32_t magic; tr_prov_plan_view v; tr_prov_node *node; uint32_t *child, *output; uint64_t bytes; };
struct tr_prov_poly_s {
    uint32_t magic, semiring;
    uint64_t n, pairs, derivations;
    uint32_t degree;
    uint64_t *coef, *off;
    uint32_t *var, *exp;
};

static void set_error(int32_t *error, int32_t code) { if (error) *error = code; }
static void *zalloc(uint64_t bytes) {
    if (bytes >= (uint64_t)SIZE_MAX) return NULL;
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
static int32_t check_bytes(const tr_math_budget *b, uint64_t bytes, tr_math_refusal *r) {
    tr_math_need n;
    need_zero(&n);
    n.need[TR_BUDGET_BYTES] = bytes;
    return check_need(b, &n, r);
}

/* ------------------------------------------------------------------ the store */
static int store_valid(const tr_prov_store *s) { return s && s->magic == STORE_MAGIC; }
static uint64_t pow2_at_least(uint64_t x) { uint64_t p = 64; while (p < x && p < (UINT64_C(1) << 62)) p <<= 1; return p; }
static uint64_t grow(uint64_t cap, uint64_t need) {
    if (need <= cap) return cap;
    uint64_t c = cap ? cap : 64;
    while (c < need) c = tr_budget_mul(c, 2);
    return c;
}
static uint64_t store_bytes(uint64_t ncap, uint64_t ccap, uint64_t hcap, uint64_t vcap) {
    uint64_t b = tr_budget_add(sizeof(tr_prov_store), tr_budget_mul(ncap, 20));
    b = tr_budget_add(b, tr_budget_mul(ccap, 4));
    b = tr_budget_add(b, tr_budget_mul(hcap, 4));
    return tr_budget_add(b, tr_budget_mul(vcap, 16));
}
static uint64_t store_bytes_now(const tr_prov_store *s) { return store_bytes(s->ncap, s->ccap, s->hcap, s->vcap); }
/* the store's needs after adding up to `nodes` nodes, `children` child entries and `vars` variables */
static void store_need(const tr_prov_store *s, uint64_t nodes, uint64_t children, uint64_t vars, tr_math_need *n,
                       uint64_t *caps) {
    need_zero(n);
    uint64_t nn = tr_budget_add(s->nnodes, nodes), nc = tr_budget_add(s->nchild, children), nv = tr_budget_add(s->nvars, vars);
    uint64_t ncap = grow(s->ncap, nn), ccap = grow(s->ccap, nc), vcap = grow(s->vcap, nv);
    uint64_t hcap = ncap == s->ncap ? s->hcap : pow2_at_least(tr_budget_mul(ncap, 2));
    n->need[TR_BUDGET_CIRCUIT] = nn;
    n->need[TR_BUDGET_SUPPORT] = nc;
    n->need[TR_BUDGET_BYTES] = store_bytes(ncap, ccap, hcap, vcap);
    if (caps) { caps[0] = ncap; caps[1] = ccap; caps[2] = hcap; caps[3] = vcap; }
}
static uint64_t node_hash(uint32_t op, const uint32_t *k, uint32_t n) {
    uint64_t h = UINT64_C(0x9e3779b97f4a7c15) ^ ((uint64_t)op << 56) ^ n;
    for (uint32_t i = 0; i < n; i++) {
        h = (h ^ k[i]) * UINT64_C(0xff51afd7ed558ccd);
        h ^= h >> 31;
    }
    return h ^ (h >> 29);
}
static void hash_put(tr_prov_store *s, uint32_t id) {
    const tr_prov_node *nd = &s->node[id];
    uint64_t mask = s->hcap - 1, slot = node_hash(nd->op, s->child + nd->arg, nd->n) & mask;
    while (s->hash[slot] != NONE32) slot = (slot + 1) & mask;
    s->hash[slot] = id;
}
/* the store's limits, then its arrays grown to hold the additions (the caller checked the needs) */
static int32_t store_reserve(tr_prov_store *s, uint64_t nodes, uint64_t children, uint64_t vars, tr_math_refusal *r) {
    tr_math_need n;
    uint64_t caps[4];
    store_need(s, nodes, children, vars, &n, caps);
    tr_math_budget lim;
    tr_budget_default(&lim);
    for (int i = 0; i < TR_BUDGET_RESOURCES; i++) lim.limit[i] = TR_BUDGET_UNKNOWN - 1;
    lim.limit[TR_BUDGET_CIRCUIT] = s->limit_nodes;
    lim.limit[TR_BUDGET_SUPPORT] = s->limit_children;
    lim.limit[TR_BUDGET_BYTES] = s->limit_bytes;
    int32_t rc = check_need(&lim, &n, r);
    if (rc) return rc;
    if (n.need[TR_BUDGET_CIRCUIT] > TR_PROV_MAX_IDS || tr_budget_add(s->nvars, vars) > TR_PROV_MAX_IDS) return TR_MATH_UNSUPPORTED;
    if (caps[0] != s->ncap) {
        tr_prov_node *nd = realloc(s->node, (size_t)caps[0] * sizeof *nd);
        if (!nd) return TR_MATH_MEMORY;
        s->node = nd;
        uint32_t *pa = realloc(s->parents, (size_t)caps[0] * 4);
        if (!pa) return TR_MATH_MEMORY;
        s->parents = pa;
        memset(s->parents + s->ncap, 0, (size_t)(caps[0] - s->ncap) * 4);
        s->ncap = caps[0];
    }
    if (caps[1] != s->ccap) {
        uint32_t *c = realloc(s->child, (size_t)caps[1] * 4);
        if (!c) return TR_MATH_MEMORY;
        s->child = c; s->ccap = caps[1];
    }
    if (caps[3] != s->vcap) {
        uint32_t *a = realloc(s->var_node, (size_t)caps[3] * 4);
        if (!a) return TR_MATH_MEMORY;
        s->var_node = a;
        uint32_t *b = realloc(s->var_source, (size_t)caps[3] * 4);
        if (!b) return TR_MATH_MEMORY;
        s->var_source = b;
        uint64_t *c = realloc(s->var_row, (size_t)caps[3] * 8);
        if (!c) return TR_MATH_MEMORY;
        s->var_row = c; s->vcap = caps[3];
    }
    if (caps[2] != s->hcap) {
        uint32_t *h = malloc((size_t)caps[2] * 4);
        if (!h) return TR_MATH_MEMORY;
        free(s->hash);
        s->hash = h; s->hcap = caps[2];
        memset(s->hash, 0xff, (size_t)s->hcap * 4);
        for (uint64_t i = 2; i < s->nnodes; i++) if (s->node[i].op != TR_PROV_VAR) hash_put(s, (uint32_t)i);
    }
    return TR_MATH_OK;
}
/* append a node (capacity reserved) */
static uint32_t store_append(tr_prov_store *s, uint32_t op, const uint32_t *k, uint32_t n, uint64_t arg) {
    uint32_t id = (uint32_t)s->nnodes++;
    tr_prov_node *nd = &s->node[id];
    nd->op = op; nd->n = n; nd->arg = n ? s->nchild : arg;
    for (uint32_t i = 0; i < n; i++) {
        s->child[s->nchild++] = k[i];
        if (i && k[i] == k[i - 1]) continue;                   /* distinct parents */
        if (s->parents[k[i]] < UINT32_MAX && ++s->parents[k[i]] == 2) s->shared++;
    }
    s->parents[id] = 0;
    return id;
}
static int u32_cmp(const void *a, const void *b) {
    uint32_t x = *(const uint32_t *)a, y = *(const uint32_t *)b;
    return x < y ? -1 : x > y;
}
/* the canonical node of op over the children k[0..n) (k is scratch: reordered); capacity for one more node and n
 * children must be reserved */
static uint32_t store_combine(tr_prov_store *s, uint32_t op, uint32_t *k, uint32_t n) {
    uint32_t m = 0;
    if (op == TR_PROV_PLUS) {
        for (uint32_t i = 0; i < n; i++) if (k[i] != TR_PROV_ZERO) k[m++] = k[i];
        if (m == 0) return TR_PROV_ZERO;
    } else {
        for (uint32_t i = 0; i < n; i++) {
            if (k[i] == TR_PROV_ZERO) return TR_PROV_ZERO;
            if (k[i] != TR_PROV_ONE) k[m++] = k[i];
        }
        if (m == 0) return TR_PROV_ONE;
    }
    if (m == 1) return k[0];
    if (m <= 16) {
        for (uint32_t i = 1; i < m; i++) {
            uint32_t x = k[i], j = i;
            while (j > 0 && k[j - 1] > x) { k[j] = k[j - 1]; j--; }
            k[j] = x;
        }
    } else qsort(k, m, 4, u32_cmp);
    uint64_t mask = s->hcap - 1, slot = node_hash(op, k, m) & mask;
    while (s->hash[slot] != NONE32) {
        const tr_prov_node *nd = &s->node[s->hash[slot]];
        if (nd->op == op && nd->n == m && !memcmp(s->child + nd->arg, k, (size_t)m * 4)) { s->hits++; return s->hash[slot]; }
        slot = (slot + 1) & mask;
    }
    uint32_t id = store_append(s, op, k, m, 0);
    s->hash[slot] = id;
    return id;
}

tr_prov_store *tr_prov_store_create(const tr_math_budget *limits, tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    tr_math_budget dflt;
    const tr_math_budget *b = budget_of(limits, &dflt);
    tr_prov_store probe;
    memset(&probe, 0, sizeof probe);
    tr_math_need n;
    uint64_t caps[4];
    store_need(&probe, 2, 0, 0, &n, caps);
    tr_math_need want;
    need_zero(&want);
    want.need[TR_BUDGET_CIRCUIT] = 2; want.need[TR_BUDGET_BYTES] = n.need[TR_BUDGET_BYTES];
    int32_t rc = check_need(b, &want, refusal);
    if (rc) { set_error(error, rc); return NULL; }
    tr_prov_store *s = zalloc(sizeof *s);
    if (!s) { set_error(error, TR_MATH_MEMORY); return NULL; }
    s->limit_nodes = b->limit[TR_BUDGET_CIRCUIT]; s->limit_children = b->limit[TR_BUDGET_SUPPORT];
    s->limit_bytes = b->limit[TR_BUDGET_BYTES];
    s->magic = STORE_MAGIC;
    rc = store_reserve(s, 2, 0, 0, refusal);
    if (rc) { s->magic = 0; free(s->node); free(s->parents); free(s->child); free(s->hash); free(s->var_node);
              free(s->var_source); free(s->var_row); free(s); set_error(error, rc); return NULL; }
    store_append(s, TR_PROV_ZERO, NULL, 0, 0);
    store_append(s, TR_PROV_ONE, NULL, 0, 0);
    set_error(error, TR_MATH_OK);
    return s;
}
void tr_prov_store_free(tr_prov_store *s) {
    if (!store_valid(s)) return;
    free(s->node); free(s->parents); free(s->child); free(s->hash); free(s->var_node); free(s->var_source);
    free(s->var_row);
    s->magic = 0;
    free(s);
}
int32_t tr_prov_store_get_info(const tr_prov_store *s, tr_prov_store_info *out) {
    if (!store_valid(s) || !out) return TR_MATH_INVALID;
    memset(out, 0, sizeof *out);
    out->nodes = s->nnodes; out->children = s->nchild; out->variables = s->nvars; out->sources = s->nsources;
    out->shared = s->shared; out->hits = s->hits; out->bytes = store_bytes_now(s);
    out->limit_nodes = s->limit_nodes; out->limit_children = s->limit_children; out->limit_bytes = s->limit_bytes;
    return TR_MATH_OK;
}
int32_t tr_prov_store_nodes(const tr_prov_store *s, uint64_t first, uint64_t n, tr_prov_node *out) {
    if (!store_valid(s) || (n && !out) || first > s->nnodes || n > s->nnodes - first) return TR_MATH_INVALID;
    if (n) memcpy(out, s->node + first, (size_t)n * sizeof *out);
    return TR_MATH_OK;
}
int32_t tr_prov_store_children(const tr_prov_store *s, uint64_t first, uint64_t n, uint32_t *out) {
    if (!store_valid(s) || (n && !out) || first > s->nchild || n > s->nchild - first) return TR_MATH_INVALID;
    if (n) memcpy(out, s->child + first, (size_t)n * 4);
    return TR_MATH_OK;
}
int32_t tr_prov_store_variables(const tr_prov_store *s, uint64_t first, uint64_t n, uint32_t *source, uint64_t *row) {
    if (!store_valid(s) || (n && (!source || !row)) || first > s->nvars || n > s->nvars - first) return TR_MATH_INVALID;
    for (uint64_t i = 0; i < n; i++) { source[i] = s->var_source[first + i]; row[i] = s->var_row[first + i]; }
    return TR_MATH_OK;
}
int32_t tr_prov_store_parents(const tr_prov_store *s, uint32_t node, uint64_t *parents) {
    if (!store_valid(s) || !parents || node >= s->nnodes) return TR_MATH_INVALID;
    *parents = s->parents[node];
    return TR_MATH_OK;
}

/* ------------------------------------------------------------------ rows and canonical relations */
static int rel_valid(const tr_prov_rel *r) { return r && r->magic == REL_MAGIC; }
static int rel_of(const tr_prov_store *s, const tr_prov_rel *r) {
    return store_valid(s) && rel_valid(r) && r->store == s && r->store_nodes <= s->nnodes;
}
void tr_prov_rel_free(tr_prov_rel *r) {
    if (!rel_valid(r)) return;
    free(r->values); free(r->valid); free(r->ann); free(r->scale);
    r->magic = 0;
    free(r);
}
static uint64_t rel_bytes(uint64_t n, int32_t ncols) {
    uint64_t b = tr_budget_add(sizeof(tr_prov_rel), tr_budget_mul(n, 9 * (uint64_t)ncols + 4));
    return tr_budget_add(b, 4 * (uint64_t)(ncols > 0 ? ncols : 1));
}
static tr_prov_rel *rel_alloc(const tr_prov_store *s, uint64_t n, int32_t ncols, const int32_t *scale) {
    tr_prov_rel *r = zalloc(sizeof *r);
    if (!r) return NULL;
    r->values = zalloc(tr_budget_mul(n, 8 * (uint64_t)ncols));
    r->valid = zalloc(tr_budget_mul(n, (uint64_t)ncols));
    r->ann = zalloc(tr_budget_mul(n, 4));
    r->scale = zalloc(4 * (uint64_t)(ncols > 0 ? ncols : 1));
    if (!r->values || !r->valid || !r->ann || !r->scale) {
        free(r->values); free(r->valid); free(r->ann); free(r->scale); free(r);
        return NULL;
    }
    r->ncols = ncols; r->ntuples = n; r->store = s; r->bytes = rel_bytes(n, ncols);
    for (int32_t c = 0; c < ncols; c++) r->scale[c] = scale ? scale[c] : 0;
    r->magic = REL_MAGIC;
    return r;
}

/* rows: n x ncols values and validity, an annotation per row; sorted order and runs of equal tuples */
typedef struct {
    uint64_t n;
    int32_t ncols, owned;
    int64_t *vals;
    uint8_t *valid;
    uint32_t *ann;
    uint64_t *idx;
    uint64_t runs, multi, members;
} rowset;
static void rowset_free(rowset *r) {
    if (r->owned) { free(r->vals); free(r->valid); free(r->ann); }
    free(r->idx);
    memset(r, 0, sizeof *r);
}
/* NULL equals NULL and comes after every value */
static int row_cmp(const rowset *r, uint64_t i, uint64_t j) {
    const uint64_t nc = (uint64_t)r->ncols;
    for (uint64_t c = 0; c < nc; c++) {
        uint8_t ki = r->valid[i * nc + c], kj = r->valid[j * nc + c];
        if (!ki || !kj) { if (ki != kj) return ki ? -1 : 1; continue; }
        int64_t a = r->vals[i * nc + c], b = r->vals[j * nc + c];
        if (a != b) return a < b ? -1 : 1;
    }
    return 0;
}
static uint64_t rowset_scratch(uint64_t n) { return tr_budget_mul(n, 16); }
/* sort (stable merge sort of indices) and count the runs */
static int32_t rowset_sort(rowset *r) {
    r->idx = zalloc(tr_budget_mul(r->n, 8));
    uint64_t *tmp = zalloc(tr_budget_mul(r->n, 8));
    if (!r->idx || !tmp) { free(tmp); return TR_MATH_MEMORY; }
    for (uint64_t i = 0; i < r->n; i++) r->idx[i] = i;
    for (uint64_t w = 1; w < r->n; w *= 2) {
        for (uint64_t st = 0; st < r->n; st += 2 * w) {
            uint64_t m = st + w < r->n ? st + w : r->n, e = st + 2 * w < r->n ? st + 2 * w : r->n, i = st, j = m, o = st;
            while (i < m && j < e) tmp[o++] = row_cmp(r, r->idx[j], r->idx[i]) < 0 ? r->idx[j++] : r->idx[i++];
            while (i < m) tmp[o++] = r->idx[i++];
            while (j < e) tmp[o++] = r->idx[j++];
        }
        memcpy(r->idx, tmp, (size_t)r->n * 8);
    }
    free(tmp);
    r->runs = r->multi = r->members = 0;
    for (uint64_t i = 0; i < r->n;) {
        uint64_t j = i + 1;
        while (j < r->n && !row_cmp(r, r->idx[i], r->idx[j])) j++;
        r->runs++;
        if (j - i > 1) { r->multi++; r->members += j - i; }
        i = j;
    }
    return TR_MATH_OK;
}
/* the canonical relation of sorted rows: one tuple per run, annotated PLUS(the run) (store capacity reserved) */
static tr_prov_rel *rowset_emit(tr_prov_store *s, const rowset *r, const int32_t *scale, int32_t *rc) {
    tr_prov_rel *out = rel_alloc(s, r->runs, r->ncols, scale);
    uint32_t *kids = zalloc(tr_budget_mul(r->n, 4));
    if (!out || !kids) { tr_prov_rel_free(out); free(kids); *rc = TR_MATH_MEMORY; return NULL; }
    const uint64_t nc = (uint64_t)r->ncols;
    uint64_t t = 0;
    for (uint64_t i = 0; i < r->n;) {
        uint64_t j = i + 1;
        while (j < r->n && !row_cmp(r, r->idx[i], r->idx[j])) j++;
        uint64_t first = r->idx[i];
        memcpy(out->values + t * nc, r->vals + first * nc, (size_t)nc * 8);
        memcpy(out->valid + t * nc, r->valid + first * nc, (size_t)nc);
        for (uint64_t c = 0; c < nc; c++) if (!out->valid[t * nc + c]) out->values[t * nc + c] = 0;
        if (j - i == 1) out->ann[t] = r->ann[first];
        else {
            for (uint64_t k = i; k < j; k++) kids[k - i] = r->ann[r->idx[k]];
            out->ann[t] = store_combine(s, TR_PROV_PLUS, kids, (uint32_t)(j - i));
        }
        t++;
        i = j;
    }
    free(kids);
    *rc = TR_MATH_OK;
    return out;
}
static void rel_finish(tr_prov_rel *r, const tr_prov_store *s, uint64_t before, int32_t op) {
    r->op = op; r->new_nodes = s->nnodes - before; r->store_nodes = s->nnodes;
}

/* ------------------------------------------------------------------ operator plans: sizes before any node */
typedef struct {
    int32_t op, ncols;
    rowset rows;                                   /* source / union / group: the rows to canonicalize */
    uint64_t out_tuples, nodes, children, vars;    /* the output and the store growth bound */
    uint64_t scratch;                              /* scratch bytes */
    int32_t scale[TR_PROV_MAX_COLUMNS];
    /* join */
    uint64_t *index, *start, *count;               /* right tuples with non-NULL keys sorted by key; per left tuple */
    int64_t *keys;                                 /* right keys [nidx x nkeys] in index order */
    uint64_t nidx;
} opplan;
static void opplan_free(opplan *p) { rowset_free(&p->rows); free(p->index); free(p->start); free(p->count); free(p->keys); memset(p, 0, sizeof *p); }

static void plan_needs(const tr_prov_store *s, const opplan *p, tr_math_need *call, tr_math_need *store) {
    need_zero(call);
    call->need[TR_BUDGET_VARIABLES] = (uint64_t)p->ncols;
    call->need[TR_BUDGET_OUTPUT] = p->out_tuples;
    call->need[TR_BUDGET_BYTES] = tr_budget_add(rel_bytes(p->out_tuples, p->ncols), p->scratch);
    store_need(s, p->nodes, p->children, p->vars, store, NULL);
}
/* the scratch is budgeted before it is allocated */
static int32_t scratch_check(const tr_math_budget *b, uint64_t bytes, tr_math_refusal *r) { return check_bytes(b, bytes, r); }

static int32_t plan_rows(opplan *p, const tr_math_budget *b, tr_math_refusal *r) {
    p->scratch = tr_budget_add(p->scratch, rowset_scratch(p->rows.n));
    int32_t rc = scratch_check(b, p->scratch, r);
    if (!rc) rc = rowset_sort(&p->rows);
    if (rc) return rc;
    p->out_tuples = p->rows.runs;
    p->nodes = tr_budget_add(p->nodes, p->rows.multi);
    p->children = tr_budget_add(p->children, p->rows.members);
    p->scratch = tr_budget_add(p->scratch, tr_budget_mul(p->rows.n, 4));   /* the PLUS children */
    return TR_MATH_OK;
}

static int32_t plan_source(const tr_prov_store *s, uint64_t n, int32_t ncols, const tr_scope_array *cols,
                           const uint64_t *rows, const tr_math_budget *b, tr_math_refusal *r, opplan *p) {
    memset(p, 0, sizeof *p);
    if ((n && (!cols || ncols < 1)) || ncols < 0) return TR_MATH_INVALID;
    if (ncols > TR_PROV_MAX_COLUMNS) return TR_MATH_UNSUPPORTED;
    for (int32_t c = 0; c < ncols; c++) if (n && !cols[c].values) return TR_MATH_INVALID;
    uint64_t sel = 0;
    for (uint64_t i = 0; i < n; i++) sel += !rows || ((rows[i >> 6] >> (i & 63)) & 1);
    p->op = TR_PROV_OP_SOURCE; p->ncols = ncols;
    for (int32_t c = 0; c < ncols; c++) p->scale[c] = cols[c].scale;
    p->scratch = tr_budget_mul(sel, 9 * (uint64_t)ncols + 4);
    int32_t rc = scratch_check(b, p->scratch, r);
    if (rc) return rc;
    rowset *rs = &p->rows;
    rs->n = sel; rs->ncols = ncols; rs->owned = 1;
    rs->vals = zalloc(tr_budget_mul(sel, 8 * (uint64_t)ncols));
    rs->valid = zalloc(tr_budget_mul(sel, (uint64_t)ncols));
    rs->ann = zalloc(tr_budget_mul(sel, 4));
    if (!rs->vals || !rs->valid || !rs->ann) return TR_MATH_MEMORY;
    uint64_t o = 0;
    for (uint64_t i = 0; i < n; i++) {
        if (rows && !((rows[i >> 6] >> (i & 63)) & 1)) continue;
        for (int32_t c = 0; c < ncols; c++) {
            uint8_t ok = !cols[c].valid || cols[c].valid[i];
            rs->valid[o * (uint64_t)ncols + (uint64_t)c] = ok;
            rs->vals[o * (uint64_t)ncols + (uint64_t)c] = ok ? cols[c].values[i] : 0;
        }
        rs->ann[o] = NONE32;                         /* the VAR node is set at build */
        o++;
    }
    p->vars = sel;
    p->nodes = sel;                                  /* one VAR per row */
    (void)s;
    return plan_rows(p, b, r);
}

static int32_t plan_union(const tr_prov_store *s, const tr_prov_rel *a, const tr_prov_rel *c, const tr_math_budget *b,
                          tr_math_refusal *r, opplan *p) {
    memset(p, 0, sizeof *p);
    if (!rel_of(s, a) || !rel_of(s, c)) return TR_MATH_INVALID;
    if (a->ncols != c->ncols) return TR_MATH_DOMAIN_ERROR;
    for (int32_t k = 0; k < a->ncols; k++) if (a->scale[k] != c->scale[k]) return TR_MATH_DOMAIN_ERROR;
    p->op = TR_PROV_OP_UNION; p->ncols = a->ncols;
    for (int32_t k = 0; k < a->ncols; k++) p->scale[k] = a->scale[k];
    uint64_t n = tr_budget_add(a->ntuples, c->ntuples), nc = (uint64_t)a->ncols;
    p->scratch = tr_budget_mul(n, 9 * nc + 4);
    int32_t rc = scratch_check(b, p->scratch, r);
    if (rc) return rc;
    rowset *rs = &p->rows;
    rs->n = n; rs->ncols = a->ncols; rs->owned = 1;
    rs->vals = zalloc(tr_budget_mul(n, 8 * nc)); rs->valid = zalloc(tr_budget_mul(n, nc)); rs->ann = zalloc(tr_budget_mul(n, 4));
    if (!rs->vals || !rs->valid || !rs->ann) return TR_MATH_MEMORY;
    if (a->ntuples) {
        memcpy(rs->vals, a->values, (size_t)(a->ntuples * nc) * 8);
        memcpy(rs->valid, a->valid, (size_t)(a->ntuples * nc));
        memcpy(rs->ann, a->ann, (size_t)a->ntuples * 4);
    }
    if (c->ntuples) {
        memcpy(rs->vals + a->ntuples * nc, c->values, (size_t)(c->ntuples * nc) * 8);
        memcpy(rs->valid + a->ntuples * nc, c->valid, (size_t)(c->ntuples * nc));
        memcpy(rs->ann + a->ntuples, c->ann, (size_t)c->ntuples * 4);
    }
    return plan_rows(p, b, r);
}

static int32_t plan_group(const tr_prov_store *s, const tr_prov_rel *a, const int32_t *cols, int32_t ncols,
                          const tr_math_budget *b, tr_math_refusal *r, opplan *p) {
    memset(p, 0, sizeof *p);
    if (!rel_of(s, a) || ncols < 0 || (ncols && !cols)) return TR_MATH_INVALID;
    if (ncols > TR_PROV_MAX_COLUMNS) return TR_MATH_UNSUPPORTED;
    for (int32_t k = 0; k < ncols; k++) if (cols[k] < 0 || cols[k] >= a->ncols) return TR_MATH_INVALID;
    p->op = TR_PROV_OP_GROUP; p->ncols = ncols;
    for (int32_t k = 0; k < ncols; k++) p->scale[k] = a->scale[cols[k]];
    if (!ncols) {                                     /* the global group: one tuple, PLUS(every tuple) */
        p->out_tuples = 1;
        p->nodes = a->ntuples > 1;
        p->children = a->ntuples > 1 ? a->ntuples : 0;
        p->scratch = tr_budget_mul(a->ntuples, 4);
        return scratch_check(b, p->scratch, r);
    }
    const uint64_t n = a->ntuples, nc = (uint64_t)ncols, ac = (uint64_t)a->ncols;
    p->scratch = tr_budget_mul(n, 9 * nc + 4);
    int32_t rc = scratch_check(b, p->scratch, r);
    if (rc) return rc;
    rowset *rs = &p->rows;
    rs->n = n; rs->ncols = ncols; rs->owned = 1;
    rs->vals = zalloc(tr_budget_mul(n, 8 * nc)); rs->valid = zalloc(tr_budget_mul(n, nc)); rs->ann = zalloc(tr_budget_mul(n, 4));
    if (!rs->vals || !rs->valid || !rs->ann) return TR_MATH_MEMORY;
    for (uint64_t i = 0; i < n; i++) {
        for (uint64_t k = 0; k < nc; k++) {
            rs->vals[i * nc + k] = a->values[i * ac + (uint64_t)cols[k]];
            rs->valid[i * nc + k] = a->valid[i * ac + (uint64_t)cols[k]];
        }
        rs->ann[i] = a->ann[i];
    }
    return plan_rows(p, b, r);
}

/* join: the right tuples whose keys are all present, sorted by key (stable: canonical order within equal keys) */
static int key_cmp(const int64_t *x, const int64_t *y, uint32_t nk) {
    for (uint32_t k = 0; k < nk; k++) if (x[k] != y[k]) return x[k] < y[k] ? -1 : 1;
    return 0;
}
static int32_t plan_join(const tr_prov_store *s, const tr_prov_rel *a, const tr_prov_rel *c, const int32_t *lcols,
                         const int32_t *rcols, uint32_t nk, const tr_math_budget *b, tr_math_refusal *r, opplan *p) {
    memset(p, 0, sizeof *p);
    if (!rel_of(s, a) || !rel_of(s, c) || (nk && (!lcols || !rcols))) return TR_MATH_INVALID;
    if ((uint64_t)a->ncols + (uint64_t)c->ncols > TR_PROV_MAX_COLUMNS || nk > TR_PROV_MAX_COLUMNS) return TR_MATH_UNSUPPORTED;
    for (uint32_t k = 0; k < nk; k++) {
        if (lcols[k] < 0 || lcols[k] >= a->ncols || rcols[k] < 0 || rcols[k] >= c->ncols) return TR_MATH_INVALID;
        if (a->scale[lcols[k]] != c->scale[rcols[k]]) return TR_MATH_DOMAIN_ERROR;
    }
    p->op = TR_PROV_OP_JOIN; p->ncols = a->ncols + c->ncols;
    for (int32_t k = 0; k < a->ncols; k++) p->scale[k] = a->scale[k];
    for (int32_t k = 0; k < c->ncols; k++) p->scale[a->ncols + k] = c->scale[k];
    const uint64_t na = a->ntuples, ncr = c->ntuples, ac = (uint64_t)a->ncols, cc = (uint64_t)c->ncols;
    p->scratch = tr_budget_add(tr_budget_mul(ncr, 24 + 16 * (uint64_t)nk), tr_budget_mul(na, 16));
    p->scratch = tr_budget_add(p->scratch, 8 * (uint64_t)(nk ? nk : 1));
    int32_t rc = scratch_check(b, p->scratch, r);
    if (rc) return rc;
    p->index = zalloc(tr_budget_mul(ncr, 8));
    uint64_t *tmp = zalloc(tr_budget_mul(ncr, 8));
    p->keys = zalloc(tr_budget_mul(ncr, 8 * (uint64_t)(nk ? nk : 1)));
    p->start = zalloc(tr_budget_mul(na, 8)); p->count = zalloc(tr_budget_mul(na, 8));
    int64_t *lkey = zalloc(8 * (uint64_t)(nk ? nk : 1));
    if (!p->index || !tmp || !p->keys || !p->start || !p->count || !lkey) { free(tmp); free(lkey); return TR_MATH_MEMORY; }
    uint64_t m = 0;
    for (uint64_t j = 0; j < ncr; j++) {
        int ok = 1;
        for (uint32_t k = 0; k < nk && ok; k++) ok = c->valid[j * cc + (uint64_t)rcols[k]];
        if (ok) tmp[m++] = j;
    }
    /* stable merge sort of tmp[0..m) by key, into index; keys gathered after */
    int64_t *kv = zalloc(tr_budget_mul(ncr, 8 * (uint64_t)(nk ? nk : 1)));   /* keys by right tuple */
    uint64_t *t2 = zalloc(tr_budget_mul(ncr, 8));
    if (!kv || !t2) { free(tmp); free(lkey); free(kv); free(t2); return TR_MATH_MEMORY; }
    for (uint64_t j = 0; j < ncr; j++)
        for (uint32_t k = 0; k < nk; k++) kv[j * nk + k] = c->values[j * cc + (uint64_t)rcols[k]];
    memcpy(p->index, tmp, (size_t)m * 8);
    for (uint64_t w = 1; w < m; w *= 2) {
        for (uint64_t st = 0; st < m; st += 2 * w) {
            uint64_t mid = st + w < m ? st + w : m, e = st + 2 * w < m ? st + 2 * w : m, i = st, j = mid, o = st;
            while (i < mid && j < e)
                t2[o++] = key_cmp(kv + p->index[j] * nk, kv + p->index[i] * nk, nk) < 0 ? p->index[j++] : p->index[i++];
            while (i < mid) t2[o++] = p->index[i++];
            while (j < e) t2[o++] = p->index[j++];
        }
        memcpy(p->index, t2, (size_t)m * 8);
    }
    for (uint64_t q = 0; q < m; q++)
        for (uint32_t k = 0; k < nk; k++) p->keys[q * nk + k] = kv[p->index[q] * nk + k];
    p->nidx = m;
    free(tmp); free(kv); free(t2);
    uint64_t pairs = 0;
    for (uint64_t i = 0; i < na; i++) {
        int ok = 1;
        for (uint32_t k = 0; k < nk && ok; k++) {
            ok = a->valid[i * ac + (uint64_t)lcols[k]];
            lkey[k] = a->values[i * ac + (uint64_t)lcols[k]];
        }
        if (!ok) { p->start[i] = 0; p->count[i] = 0; continue; }
        uint64_t lo = 0, hi = m;                       /* the first index with key >= lkey */
        while (lo < hi) { uint64_t md = lo + (hi - lo) / 2; if (key_cmp(p->keys + md * nk, lkey, nk) < 0) lo = md + 1; else hi = md; }
        uint64_t e = lo;
        while (e < m && !key_cmp(p->keys + e * nk, lkey, nk)) e++;
        p->start[i] = lo; p->count[i] = e - lo;
        pairs = tr_budget_add(pairs, e - lo);
    }
    free(lkey);
    p->out_tuples = pairs;
    p->nodes = pairs;
    p->children = tr_budget_mul(pairs, 2);
    return TR_MATH_OK;
}

static int32_t plan_select(const tr_prov_store *s, const tr_prov_rel *a, const tr_prov_pred *pr, uint32_t np,
                           const tr_math_budget *b, tr_math_refusal *r, opplan *p, uint8_t **keep) {
    memset(p, 0, sizeof *p);
    *keep = NULL;
    if (!rel_of(s, a) || (np && !pr)) return TR_MATH_INVALID;
    for (uint32_t k = 0; k < np; k++) {
        if (pr[k].op < TR_PROV_EQ || pr[k].op > TR_PROV_IS_NOT_NULL || pr[k].col < 0 || pr[k].col >= a->ncols ||
            pr[k].reserved || pr[k].col2 < -1 || pr[k].col2 >= a->ncols ||
            (pr[k].op >= TR_PROV_IS_NULL && pr[k].col2 != -1)) return TR_MATH_INVALID;
        if (pr[k].col2 >= 0 && a->scale[pr[k].col] != a->scale[pr[k].col2]) return TR_MATH_DOMAIN_ERROR;
    }
    p->op = TR_PROV_OP_SELECT; p->ncols = a->ncols;
    for (int32_t k = 0; k < a->ncols; k++) p->scale[k] = a->scale[k];
    p->scratch = a->ntuples;
    int32_t rc = scratch_check(b, p->scratch, r);
    if (rc) return rc;
    uint8_t *kp = zalloc(a->ntuples);
    if (!kp) return TR_MATH_MEMORY;
    const uint64_t ac = (uint64_t)a->ncols;
    uint64_t out = 0;
    for (uint64_t i = 0; i < a->ntuples; i++) {
        int pass = 1;
        for (uint32_t k = 0; k < np && pass; k++) {
            const tr_prov_pred *q = &pr[k];
            uint8_t ok = a->valid[i * ac + (uint64_t)q->col];
            if (q->op == TR_PROV_IS_NULL) { pass = !ok; continue; }
            if (q->op == TR_PROV_IS_NOT_NULL) { pass = ok; continue; }
            if (!ok || (q->col2 >= 0 && !a->valid[i * ac + (uint64_t)q->col2])) { pass = 0; continue; }   /* UNKNOWN */
            int64_t x = a->values[i * ac + (uint64_t)q->col], y = q->col2 >= 0 ? a->values[i * ac + (uint64_t)q->col2] : q->value;
            switch (q->op) {
            case TR_PROV_EQ: pass = x == y; break;
            case TR_PROV_NE: pass = x != y; break;
            case TR_PROV_LT: pass = x < y; break;
            case TR_PROV_LE: pass = x <= y; break;
            case TR_PROV_GT: pass = x > y; break;
            default: pass = x >= y; break;
            }
        }
        kp[i] = (uint8_t)pass;
        out += pass;
    }
    p->out_tuples = out;
    *keep = kp;
    return TR_MATH_OK;
}

/* the operator's checks: its own (call budget), then the store's limits (reserving capacity) */
static int32_t plan_check(tr_prov_store *s, const opplan *p, const tr_math_budget *b, tr_math_refusal *r) {
    tr_math_need call, store;
    plan_needs(s, p, &call, &store);
    int32_t rc = check_need(b, &call, r);
    if (!rc) rc = store_reserve(s, p->nodes, p->children, p->vars, r);
    return rc;
}

tr_prov_rel *tr_prov_source(tr_prov_store *s, uint64_t n, int32_t ncols, const tr_scope_array *columns,
                            const uint64_t *rows, uint32_t *first_var, uint32_t *source, const tr_math_budget *budget,
                            tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    if (!store_valid(s)) { set_error(error, TR_MATH_INVALID); return NULL; }
    tr_math_budget dflt;
    const tr_math_budget *b = budget_of(budget, &dflt);
    opplan p;
    int32_t rc = plan_source(s, n, ncols, columns, rows, b, refusal, &p);
    if (!rc) rc = plan_check(s, &p, b, refusal);
    if (rc) { opplan_free(&p); set_error(error, rc); return NULL; }
    uint64_t before = s->nnodes;
    uint32_t v0 = (uint32_t)s->nvars, src = (uint32_t)s->nsources;
    uint64_t o = 0;
    for (uint64_t i = 0; i < n; i++) {                 /* a variable and its VAR node per selected row, in order */
        if (rows && !((rows[i >> 6] >> (i & 63)) & 1)) continue;
        uint32_t v = (uint32_t)s->nvars++;
        s->var_source[v] = src; s->var_row[v] = i;
        s->var_node[v] = store_append(s, TR_PROV_VAR, NULL, 0, v);
        p.rows.ann[o++] = s->var_node[v];
    }
    s->nsources++;
    tr_prov_rel *out = rowset_emit(s, &p.rows, p.scale, &rc);
    opplan_free(&p);
    if (!out) { set_error(error, rc); return NULL; }
    rel_finish(out, s, before, TR_PROV_OP_SOURCE);
    if (first_var) *first_var = v0;
    if (source) *source = src;
    set_error(error, TR_MATH_OK);
    return out;
}

tr_prov_rel *tr_prov_source_scope(tr_prov_store *s, const tr_scope *scope, const int32_t *cols, int32_t ncols,
                                  uint32_t *first_var, uint32_t *source, const tr_math_budget *budget,
                                  tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    tr_scope_view v;
    if (!store_valid(s) || !scope || tr_scope_get_view(scope, &v) || ncols < 1 || !cols) { set_error(error, TR_MATH_INVALID); return NULL; }
    if (ncols > TR_PROV_MAX_COLUMNS) { set_error(error, TR_MATH_UNSUPPORTED); return NULL; }
    for (int32_t c = 0; c < ncols; c++) if (cols[c] < 0 || cols[c] >= v.ncols) { set_error(error, TR_MATH_INVALID); return NULL; }
    tr_math_budget dflt;
    const tr_math_budget *b = budget_of(budget, &dflt);
    int32_t rc = check_bytes(b, tr_budget_mul(v.rows, 9 * (uint64_t)ncols), refusal);
    if (rc) { set_error(error, rc); return NULL; }
    int64_t *vals = zalloc(tr_budget_mul(v.rows, 8 * (uint64_t)ncols));
    uint8_t *ok = zalloc(tr_budget_mul(v.rows, (uint64_t)ncols));
    tr_scope_array arr[TR_PROV_MAX_COLUMNS];
    if (!vals || !ok) { free(vals); free(ok); set_error(error, TR_MATH_MEMORY); return NULL; }
    for (int32_t c = 0; c < ncols; c++) {
        arr[c].values = vals + (uint64_t)c * v.rows; arr[c].valid = ok + (uint64_t)c * v.rows;
        arr[c].scale = v.col[cols[c]].scale; arr[c].reserved = 0;
        for (uint64_t i = 0; i < v.rows; i++) {
            if (!tr_scope_is_selected(&v, i)) continue;
            uint8_t present = (uint8_t)tr_scope_present(&v, cols[c], i);
            ok[(uint64_t)c * v.rows + i] = present;
            if (present) vals[(uint64_t)c * v.rows + i] = tr_scope_value(&v, cols[c], i);
        }
    }
    tr_prov_rel *out = tr_prov_source(s, v.rows, ncols, arr, v.selection, first_var, source, b, refusal, error);
    free(vals); free(ok);
    return out;
}

tr_prov_rel *tr_prov_select(tr_prov_store *s, const tr_prov_rel *a, const tr_prov_pred *preds, uint32_t npreds,
                            const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    tr_math_budget dflt;
    const tr_math_budget *b = budget_of(budget, &dflt);
    opplan p;
    uint8_t *keep = NULL;
    int32_t rc = plan_select(s, a, preds, npreds, b, refusal, &p, &keep);
    if (!rc) rc = plan_check(s, &p, b, refusal);
    if (rc) { free(keep); opplan_free(&p); set_error(error, rc); return NULL; }
    tr_prov_rel *out = rel_alloc(s, p.out_tuples, a->ncols, a->scale);
    if (!out) { free(keep); opplan_free(&p); set_error(error, TR_MATH_MEMORY); return NULL; }
    const uint64_t ac = (uint64_t)a->ncols;
    uint64_t t = 0;
    for (uint64_t i = 0; i < a->ntuples; i++) {
        if (!keep[i]) continue;
        memcpy(out->values + t * ac, a->values + i * ac, (size_t)ac * 8);
        memcpy(out->valid + t * ac, a->valid + i * ac, (size_t)ac);
        out->ann[t++] = a->ann[i];
    }
    free(keep);
    opplan_free(&p);
    rel_finish(out, s, s->nnodes, TR_PROV_OP_SELECT);
    set_error(error, TR_MATH_OK);
    return out;
}

tr_prov_rel *tr_prov_union(tr_prov_store *s, const tr_prov_rel *a, const tr_prov_rel *c, const tr_math_budget *budget,
                           tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    tr_math_budget dflt;
    const tr_math_budget *b = budget_of(budget, &dflt);
    opplan p;
    int32_t rc = plan_union(s, a, c, b, refusal, &p);
    if (!rc) rc = plan_check(s, &p, b, refusal);
    if (rc) { opplan_free(&p); set_error(error, rc); return NULL; }
    uint64_t before = s->nnodes;
    tr_prov_rel *out = rowset_emit(s, &p.rows, p.scale, &rc);
    opplan_free(&p);
    if (!out) { set_error(error, rc); return NULL; }
    rel_finish(out, s, before, TR_PROV_OP_UNION);
    set_error(error, TR_MATH_OK);
    return out;
}

tr_prov_rel *tr_prov_group(tr_prov_store *s, const tr_prov_rel *a, const int32_t *cols, int32_t ncols,
                           const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    tr_math_budget dflt;
    const tr_math_budget *b = budget_of(budget, &dflt);
    opplan p;
    int32_t rc = plan_group(s, a, cols, ncols, b, refusal, &p);
    if (!rc) rc = plan_check(s, &p, b, refusal);
    if (rc) { opplan_free(&p); set_error(error, rc); return NULL; }
    uint64_t before = s->nnodes;
    tr_prov_rel *out = NULL;
    if (!ncols) {
        out = rel_alloc(s, 1, 0, NULL);
        uint32_t *kids = zalloc(tr_budget_mul(a->ntuples, 4));
        if (!out || !kids) { tr_prov_rel_free(out); free(kids); opplan_free(&p); set_error(error, TR_MATH_MEMORY); return NULL; }
        if (a->ntuples) memcpy(kids, a->ann, (size_t)a->ntuples * 4);
        out->ann[0] = a->ntuples ? store_combine(s, TR_PROV_PLUS, kids, (uint32_t)a->ntuples) : TR_PROV_ZERO;
        out->global = 1;
        free(kids);
    } else out = rowset_emit(s, &p.rows, p.scale, &rc);
    opplan_free(&p);
    if (!out) { set_error(error, rc); return NULL; }
    rel_finish(out, s, before, TR_PROV_OP_GROUP);
    set_error(error, TR_MATH_OK);
    return out;
}

tr_prov_rel *tr_prov_join(tr_prov_store *s, const tr_prov_rel *a, const tr_prov_rel *c, const int32_t *lcols,
                          const int32_t *rcols, uint32_t nkeys, const tr_math_budget *budget, tr_math_refusal *refusal,
                          int32_t *error) {
    tr_budget_none(refusal);
    tr_math_budget dflt;
    const tr_math_budget *b = budget_of(budget, &dflt);
    opplan p;
    int32_t rc = plan_join(s, a, c, lcols, rcols, nkeys, b, refusal, &p);
    if (!rc) rc = plan_check(s, &p, b, refusal);
    if (rc) { opplan_free(&p); set_error(error, rc); return NULL; }
    uint64_t before = s->nnodes;
    tr_prov_rel *out = rel_alloc(s, p.out_tuples, p.ncols, p.scale);
    if (!out) { opplan_free(&p); set_error(error, TR_MATH_MEMORY); return NULL; }
    const uint64_t ac = (uint64_t)a->ncols, cc = (uint64_t)c->ncols, oc = ac + cc;
    uint64_t t = 0;
    for (uint64_t i = 0; i < a->ntuples; i++)
        for (uint64_t q = p.start[i]; q < p.start[i] + p.count[i]; q++) {
            uint64_t j = p.index[q];
            memcpy(out->values + t * oc, a->values + i * ac, (size_t)ac * 8);
            memcpy(out->valid + t * oc, a->valid + i * ac, (size_t)ac);
            memcpy(out->values + t * oc + ac, c->values + j * cc, (size_t)cc * 8);
            memcpy(out->valid + t * oc + ac, c->valid + j * cc, (size_t)cc);
            uint32_t kids[2] = {a->ann[i], c->ann[j]};
            out->ann[t++] = store_combine(s, TR_PROV_TIMES, kids, 2);
        }
    opplan_free(&p);
    rel_finish(out, s, before, TR_PROV_OP_JOIN);
    set_error(error, TR_MATH_OK);
    return out;
}

int32_t tr_prov_need(const tr_prov_store *s, int32_t op, const tr_prov_rel *a, const tr_prov_rel *c, const int32_t *lcols,
                     const int32_t *rcols, uint32_t nkeys, tr_math_need *call, tr_math_need *store) {
    if (!store_valid(s) || !call || !store) return TR_MATH_INVALID;
    tr_math_budget unlimited;
    for (int i = 0; i < TR_BUDGET_RESOURCES; i++) unlimited.limit[i] = TR_BUDGET_UNKNOWN - 1;
    opplan p;
    int32_t rc;
    switch (op) {
    case TR_PROV_OP_UNION: rc = plan_union(s, a, c, &unlimited, NULL, &p); break;
    case TR_PROV_OP_JOIN: rc = plan_join(s, a, c, lcols, rcols, nkeys, &unlimited, NULL, &p); break;
    case TR_PROV_OP_GROUP: rc = plan_group(s, a, lcols, (int32_t)nkeys, &unlimited, NULL, &p); break;
    default: memset(&p, 0, sizeof p); rc = TR_MATH_INVALID;
    }
    if (!rc) plan_needs(s, &p, call, store);
    opplan_free(&p);
    return rc;
}

int32_t tr_prov_rel_get_info(const tr_prov_rel *r, tr_prov_rel_info *out) {
    if (!rel_valid(r) || !out) return TR_MATH_INVALID;
    memset(out, 0, sizeof *out);
    out->tuples = r->ntuples; out->bytes = r->bytes; out->new_nodes = r->new_nodes; out->store_nodes = r->store_nodes;
    out->ncols = r->ncols; out->global = r->global; out->op = r->op;
    return TR_MATH_OK;
}
int32_t tr_prov_rel_get(const tr_prov_rel *r, int64_t *values, uint8_t *valid, uint32_t *ann, int32_t *scales,
                        uint64_t capacity) {
    if (!rel_valid(r) || capacity < r->ntuples || (r->ntuples && (!ann || (r->ncols && (!values || !valid)))) ||
        (r->ncols && !scales)) return TR_MATH_INVALID;
    const uint64_t nc = (uint64_t)r->ncols;
    if (r->ntuples && nc) {
        memcpy(values, r->values, (size_t)(r->ntuples * nc) * 8);
        memcpy(valid, r->valid, (size_t)(r->ntuples * nc));
    }
    if (r->ntuples) memcpy(ann, r->ann, (size_t)r->ntuples * 4);
    if (nc) memcpy(scales, r->scale, (size_t)nc * 4);
    return TR_MATH_OK;
}

/* ------------------------------------------------------------------ evaluation plans */
static int plan_valid(const tr_prov_plan *p) { return p && p->magic == PLAN_MAGIC; }
void tr_prov_plan_free(tr_prov_plan *p) {
    if (!plan_valid(p)) return;
    free(p->node); free(p->child); free(p->output);
    p->magic = 0;
    free(p);
}
/* the cone of the given nodes: mark[0..top] and its size */
static int32_t cone_mark(const tr_prov_store *s, const uint32_t *nodes, uint64_t n, uint8_t *mark, uint64_t top,
                         uint64_t *size, uint64_t *children) {
    for (uint64_t i = 0; i < n; i++) mark[nodes[i]] = 1;
    uint64_t sz = 0, ch = 0;
    for (uint64_t id = top + 1; id-- > 0;) {
        if (!mark[id]) continue;
        sz++;
        const tr_prov_node *nd = &s->node[id];
        if (nd->op == TR_PROV_PLUS || nd->op == TR_PROV_TIMES) {
            ch += nd->n;
            for (uint32_t c = 0; c < nd->n; c++) mark[s->child[nd->arg + c]] = 1;
        }
    }
    *size = sz; *children = ch;
    return TR_MATH_OK;
}

tr_prov_plan *tr_prov_plan_build(const tr_prov_store *s, const uint32_t *nodes, uint64_t n, const tr_math_budget *budget,
                                 tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    if (!store_valid(s) || (n && !nodes)) { set_error(error, TR_MATH_INVALID); return NULL; }
    uint64_t top = 0;
    for (uint64_t i = 0; i < n; i++) {
        if (nodes[i] >= s->nnodes) { set_error(error, TR_MATH_INVALID); return NULL; }
        if (nodes[i] > top) top = nodes[i];
    }
    tr_math_budget dflt;
    const tr_math_budget *b = budget_of(budget, &dflt);
    int32_t rc = check_bytes(b, tr_budget_mul(top + 1, 5), refusal);
    if (rc) { set_error(error, rc); return NULL; }
    uint8_t *mark = zalloc(top + 1);
    uint32_t *local = zalloc(tr_budget_mul(top + 1, 4));
    if (!mark || !local) { free(mark); free(local); set_error(error, TR_MATH_MEMORY); return NULL; }
    uint64_t size = 0, ch = 0;
    cone_mark(s, nodes, n, mark, top, &size, &ch);
    tr_math_need need;
    need_zero(&need);
    need.need[TR_BUDGET_CIRCUIT] = size;
    need.need[TR_BUDGET_SUPPORT] = ch;
    need.need[TR_BUDGET_OUTPUT] = n;
    need.need[TR_BUDGET_BYTES] = tr_budget_add(tr_budget_add(tr_budget_mul(size, 16), tr_budget_mul(tr_budget_add(ch, n), 4)),
                                               tr_budget_add(tr_budget_mul(top + 1, 5), sizeof(tr_prov_plan)));
    rc = check_need(b, &need, refusal);
    tr_prov_plan *p = rc ? NULL : zalloc(sizeof *p);
    if (!rc && !p) rc = TR_MATH_MEMORY;
    if (p) {
        p->node = zalloc(tr_budget_mul(size, 16)); p->child = zalloc(tr_budget_mul(ch, 4)); p->output = zalloc(tr_budget_mul(n, 4));
        if (!p->node || !p->child || !p->output) rc = TR_MATH_MEMORY;
    }
    if (!rc) {
        uint64_t k = 0, c = 0;
        for (uint64_t id = 0; id <= top; id++) {
            if (!mark[id]) continue;
            local[id] = (uint32_t)k;
            tr_prov_node nd = s->node[id];
            if (nd.op == TR_PROV_PLUS || nd.op == TR_PROV_TIMES) {
                for (uint32_t q = 0; q < nd.n; q++) p->child[c + q] = local[s->child[nd.arg + q]];
                nd.arg = c;
                c += nd.n;
            }
            p->node[k++] = nd;
        }
        for (uint64_t i = 0; i < n; i++) p->output[i] = local[nodes[i]];
        p->v.nodes = size; p->v.nchild = ch; p->v.noutputs = n; p->v.nvars = s->nvars;
        p->v.node = p->node; p->v.child = p->child; p->v.output = p->output;
        p->bytes = need.need[TR_BUDGET_BYTES];
        p->magic = PLAN_MAGIC;
    }
    free(mark); free(local);
    if (rc) {
        if (p) { free(p->node); free(p->child); free(p->output); free(p); }
        set_error(error, rc);
        return NULL;
    }
    set_error(error, TR_MATH_OK);
    return p;
}
int32_t tr_prov_plan_get_view(const tr_prov_plan *p, tr_prov_plan_view *v) {
    if (!plan_valid(p) || !v) return TR_MATH_INVALID;
    *v = p->v;
    return TR_MATH_OK;
}

int32_t tr_prov_evaluate(const tr_prov_plan *p, int32_t semiring, const tr_prov_valuations *v, void *out,
                         int32_t *status, tr_prov_eval_info *info, const tr_math_budget *budget, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!plan_valid(p) || !v || (semiring != TR_PROV_BOOL && semiring != TR_PROV_COUNT) ||
        tr_prov_valuations_check(v, p->v.nvars) || (v->count && p->v.noutputs && !out) || (v->count && !status))
        return TR_MATH_INVALID;
    tr_math_budget dflt;
    const tr_math_budget *b = budget_of(budget, &dflt);
    tr_math_need need;
    need_zero(&need);
    need.need[TR_BUDGET_OUTPUT] = tr_budget_mul(v->count, p->v.noutputs);
    need.need[TR_BUDGET_BYTES] = tr_budget_mul(p->v.nodes ? p->v.nodes : 1, tr_prov_scratch_width(semiring));
    int32_t rc = check_need(b, &need, refusal);
    if (rc) return rc;
    void *scratch = zalloc(need.need[TR_BUDGET_BYTES]);
    if (!scratch) return TR_MATH_MEMORY;
    uint64_t over = 0;
    for (uint64_t w = 0; w < v->count; w++) {
        if (semiring == TR_PROV_COUNT) {
            status[w] = tr_prov_eval_count(&p->v, v, w, (unsigned __int128 *)scratch, (uint64_t *)out + 2 * w * p->v.noutputs);
            over += status[w] != 0;
        } else {
            tr_prov_eval_bool(&p->v, v, w, (uint8_t *)scratch, (uint8_t *)out + w * p->v.noutputs);
            status[w] = 0;
        }
    }
    free(scratch);
    if (info) {
        memset(info, 0, sizeof *info);
        info->count = v->count; info->noutputs = p->v.noutputs; info->overflowed = over; info->chunk = 1;
        info->semiring = semiring;
    }
    return TR_MATH_OK;
}

/* ------------------------------------------------------------------ N[X]: expansion and witnesses */
typedef struct { uint64_t n, pairs; uint64_t *coef, *off; uint32_t *var, *exp; } xpoly;
static void xp_free(xpoly *p) { free(p->coef); free(p->off); free(p->var); free(p->exp); memset(p, 0, sizeof *p); }
static int32_t xp_alloc(xpoly *p, uint64_t n, uint64_t pairs) {
    memset(p, 0, sizeof *p);
    p->coef = zalloc(tr_budget_mul(n, 8)); p->off = zalloc(tr_budget_mul(n + 1, 8));
    p->var = zalloc(tr_budget_mul(pairs, 4)); p->exp = zalloc(tr_budget_mul(pairs, 4));
    if (!p->coef || !p->off || !p->var || !p->exp) { xp_free(p); return TR_MATH_MEMORY; }
    p->n = n; p->pairs = pairs;
    return TR_MATH_OK;
}
static int mono_cmp(const xpoly *a, uint64_t i, const xpoly *b, uint64_t j) {
    uint64_t x = a->off[i], xe = a->off[i + 1], y = b->off[j], ye = b->off[j + 1];
    for (; x < xe && y < ye; x++, y++) {
        if (a->var[x] != b->var[y]) return a->var[x] < b->var[y] ? -1 : 1;
        if (a->exp[x] != b->exp[y]) return a->exp[x] < b->exp[y] ? -1 : 1;
    }
    return x < xe ? 1 : y < ye ? -1 : 0;
}
/* sort the monomials of p and merge equal ones (coefficients add, checked) into q */
static int32_t xp_normalize(const xpoly *p, xpoly *q) {
    uint64_t *idx = zalloc(tr_budget_mul(p->n, 8)), *tmp = zalloc(tr_budget_mul(p->n, 8));
    if (!idx || !tmp) { free(idx); free(tmp); return TR_MATH_MEMORY; }
    for (uint64_t i = 0; i < p->n; i++) idx[i] = i;
    for (uint64_t w = 1; w < p->n; w *= 2) {
        for (uint64_t st = 0; st < p->n; st += 2 * w) {
            uint64_t m = st + w < p->n ? st + w : p->n, e = st + 2 * w < p->n ? st + 2 * w : p->n, i = st, j = m, o = st;
            while (i < m && j < e) tmp[o++] = mono_cmp(p, idx[j], p, idx[i]) < 0 ? idx[j++] : idx[i++];
            while (i < m) tmp[o++] = idx[i++];
            while (j < e) tmp[o++] = idx[j++];
        }
        memcpy(idx, tmp, (size_t)p->n * 8);
    }
    free(tmp);
    uint64_t terms = 0, pairs = 0;
    for (uint64_t i = 0; i < p->n;) {
        uint64_t j = i + 1;
        while (j < p->n && !mono_cmp(p, idx[i], p, idx[j])) j++;
        terms++; pairs += p->off[idx[i] + 1] - p->off[idx[i]];
        i = j;
    }
    int32_t rc = xp_alloc(q, terms, pairs);
    if (rc) { free(idx); return rc; }
    uint64_t t = 0, at = 0;
    for (uint64_t i = 0; i < p->n && !rc;) {
        uint64_t j = i, c = 0;
        while (j < p->n && !mono_cmp(p, idx[i], p, idx[j])) {
            if (c > UINT64_MAX - p->coef[idx[j]]) { rc = TR_MATH_OVERFLOW; break; }
            c += p->coef[idx[j]]; j++;
        }
        uint64_t s0 = p->off[idx[i]], len = p->off[idx[i] + 1] - s0;
        q->coef[t] = c; q->off[t] = at;
        memcpy(q->var + at, p->var + s0, (size_t)len * 4);
        memcpy(q->exp + at, p->exp + s0, (size_t)len * 4);
        at += len; t++;
        i = j;
    }
    if (!rc) q->off[t] = at;
    free(idx);
    if (rc) xp_free(q);
    return rc;
}
/* a x b (every pair of monomials: variables merged, exponents added; coefficients multiplied, checked) */
static int32_t xp_mul(const xpoly *a, const xpoly *b, xpoly *out) {
    uint64_t n = tr_budget_mul(a->n, b->n), pairs = tr_budget_add(tr_budget_mul(a->pairs, b->n), tr_budget_mul(b->pairs, a->n));
    if (n == TR_BUDGET_UNKNOWN || pairs == TR_BUDGET_UNKNOWN) return TR_MATH_OVERFLOW;
    xpoly raw;
    int32_t rc = xp_alloc(&raw, n, pairs);
    if (rc) return rc;
    uint64_t t = 0, at = 0;
    for (uint64_t i = 0; i < a->n && !rc; i++)
        for (uint64_t j = 0; j < b->n; j++) {
            if (a->coef[i] && b->coef[j] > UINT64_MAX / a->coef[i]) { rc = TR_MATH_OVERFLOW; break; }
            raw.coef[t] = a->coef[i] * b->coef[j];
            raw.off[t] = at;
            uint64_t x = a->off[i], xe = a->off[i + 1], y = b->off[j], ye = b->off[j + 1];
            while (x < xe || y < ye) {
                if (y >= ye || (x < xe && a->var[x] < b->var[y])) { raw.var[at] = a->var[x]; raw.exp[at++] = a->exp[x++]; }
                else if (x >= xe || b->var[y] < a->var[x]) { raw.var[at] = b->var[y]; raw.exp[at++] = b->exp[y++]; }
                else { raw.var[at] = a->var[x]; raw.exp[at++] = a->exp[x++] + b->exp[y++]; }
            }
            t++;
        }
    if (!rc) { raw.off[t] = at; raw.n = t; rc = xp_normalize(&raw, out); }
    xp_free(&raw);
    return rc;
}
/* the sum of children polynomials (with their repetitions) */
static int32_t xp_sum(const xpoly *const *c, uint32_t k, xpoly *out) {
    uint64_t n = 0, pairs = 0;
    for (uint32_t i = 0; i < k; i++) { n = tr_budget_add(n, c[i]->n); pairs = tr_budget_add(pairs, c[i]->pairs); }
    if (n == TR_BUDGET_UNKNOWN) return TR_MATH_OVERFLOW;
    xpoly raw;
    int32_t rc = xp_alloc(&raw, n, pairs);
    if (rc) return rc;
    uint64_t t = 0, at = 0;
    for (uint32_t i = 0; i < k; i++)
        for (uint64_t j = 0; j < c[i]->n; j++) {
            uint64_t s0 = c[i]->off[j], len = c[i]->off[j + 1] - s0;
            raw.coef[t] = c[i]->coef[j]; raw.off[t] = at;
            memcpy(raw.var + at, c[i]->var + s0, (size_t)len * 4);
            memcpy(raw.exp + at, c[i]->exp + s0, (size_t)len * 4);
            at += len; t++;
        }
    raw.off[t] = at;
    rc = xp_normalize(&raw, out);
    xp_free(&raw);
    return rc;
}

/* the expansion's first pass over the cone of node: derivations D and degrees, the needs */
static int32_t expand_plan(const tr_prov_store *s, uint32_t node, int32_t witnesses, tr_math_need *need, uint8_t **mark_out,
                           uint64_t **d_out, uint64_t **deg_out) {
    uint8_t *mark = zalloc((uint64_t)node + 1);
    uint64_t *d = zalloc(tr_budget_mul((uint64_t)node + 1, 8)), *deg = zalloc(tr_budget_mul((uint64_t)node + 1, 8));
    if (!mark || !d || !deg) { free(mark); free(d); free(deg); return TR_MATH_MEMORY; }
    uint64_t size = 0, ch = 0;
    cone_mark(s, &node, 1, mark, node, &size, &ch);
    uint64_t bytes = 0;
    for (uint64_t id = 0; id <= node; id++) {
        if (!mark[id]) continue;
        const tr_prov_node *nd = &s->node[id];
        uint64_t D = 0, g = 0;
        if (nd->op == TR_PROV_ONE) D = 1;
        else if (nd->op == TR_PROV_VAR) { D = 1; g = 1; }
        else if (nd->op == TR_PROV_PLUS) for (uint32_t c = 0; c < nd->n; c++) {
            uint32_t k = s->child[nd->arg + c];
            D = tr_budget_add(D, d[k]);
            if (deg[k] > g) g = deg[k];
        } else if (nd->op == TR_PROV_TIMES) {
            D = 1;
            for (uint32_t c = 0; c < nd->n; c++) {
                uint32_t k = s->child[nd->arg + c];
                D = tr_budget_mul(D, d[k]);
                g = tr_budget_add(g, deg[k]);
            }
        }
        d[id] = D; deg[id] = g;
        /* the node's polynomial and its raw scratch (before normalizing): 3 x D monomials of <= g pairs */
        bytes = tr_budget_add(bytes, tr_budget_mul(tr_budget_mul(D, 3), tr_budget_add(32, tr_budget_mul(g, 8))));
    }
    need_zero(need);
    need->need[TR_BUDGET_CIRCUIT] = size;
    need->need[TR_BUDGET_SUPPORT] = d[node];
    need->need[TR_BUDGET_DEGREE] = deg[node];
    if (witnesses) need->need[TR_BUDGET_TENSOR] = tr_budget_mul(d[node], d[node]);
    need->need[TR_BUDGET_BYTES] = tr_budget_add(bytes, tr_budget_mul((uint64_t)node + 1, 17 + sizeof(xpoly)));
    if (mark_out) { *mark_out = mark; *d_out = d; *deg_out = deg; }
    else { free(mark); free(d); free(deg); }
    return TR_MATH_OK;
}

int32_t tr_prov_expand_need(const tr_prov_store *s, uint32_t node, int32_t witnesses, tr_math_need *need) {
    if (!store_valid(s) || !need || node >= s->nnodes) return TR_MATH_INVALID;
    return expand_plan(s, node, witnesses, need, NULL, NULL, NULL);
}

static int poly_valid(const tr_prov_poly *p) { return p && p->magic == POLY_MAGIC; }
void tr_prov_poly_free(tr_prov_poly *p) {
    if (!poly_valid(p)) return;
    free(p->coef); free(p->off); free(p->var); free(p->exp);
    p->magic = 0;
    free(p);
}
static tr_prov_poly *poly_adopt(xpoly *x, uint32_t semiring, int32_t *rc) {
    tr_prov_poly *p = zalloc(sizeof *p);
    if (!p) { xp_free(x); *rc = TR_MATH_MEMORY; return NULL; }
    p->n = x->n; p->pairs = x->pairs; p->coef = x->coef; p->off = x->off; p->var = x->var; p->exp = x->exp;
    memset(x, 0, sizeof *x);
    p->semiring = semiring;
    uint64_t der = 0;
    uint32_t deg = 0;
    for (uint64_t i = 0; i < p->n; i++) {
        der = tr_budget_add(der, p->coef[i]);
        uint32_t g = 0;
        for (uint64_t k = p->off[i]; k < p->off[i + 1]; k++) g += p->exp[k];
        if (g > deg) deg = g;
    }
    p->derivations = der; p->degree = deg;
    p->magic = POLY_MAGIC;
    *rc = TR_MATH_OK;
    return p;
}

/* the memoized bottom-up expansion of node's cone into *out */
static int32_t expand_into(const tr_prov_store *s, uint32_t node, const uint8_t *mark, xpoly *out) {
    xpoly *memo = zalloc(tr_budget_mul((uint64_t)node + 1, sizeof(xpoly)));
    if (!memo) return TR_MATH_MEMORY;
    int32_t rc = TR_MATH_OK;
    for (uint64_t id = 0; id <= node && !rc; id++) {
        if (!mark[id]) continue;
        const tr_prov_node *nd = &s->node[id];
        xpoly *m = &memo[id];
        if (nd->op == TR_PROV_ZERO) rc = xp_alloc(m, 0, 0);
        else if (nd->op == TR_PROV_ONE) { rc = xp_alloc(m, 1, 0); if (!rc) { m->coef[0] = 1; m->off[0] = m->off[1] = 0; } }
        else if (nd->op == TR_PROV_VAR) {
            rc = xp_alloc(m, 1, 1);
            if (!rc) { m->coef[0] = 1; m->off[0] = 0; m->off[1] = 1; m->var[0] = (uint32_t)nd->arg; m->exp[0] = 1; }
        } else if (nd->op == TR_PROV_PLUS) {
            const xpoly **kids = calloc(nd->n, sizeof *kids);
            if (!kids) { rc = TR_MATH_MEMORY; break; }
            for (uint32_t c = 0; c < nd->n; c++) kids[c] = &memo[s->child[nd->arg + c]];
            rc = xp_sum(kids, nd->n, m);
            free((void *)kids);
        } else {
            xpoly acc, next;
            const xpoly *first = &memo[s->child[nd->arg]];
            rc = xp_sum(&first, 1, &acc);                         /* a copy */
            for (uint32_t c = 1; c < nd->n && !rc; c++) {
                rc = xp_mul(&acc, &memo[s->child[nd->arg + c]], &next);
                xp_free(&acc);
                if (!rc) acc = next;
            }
            if (!rc) *m = acc;
        }
    }
    if (!rc) { *out = memo[node]; memset(&memo[node], 0, sizeof memo[node]); }
    for (uint64_t id = 0; id <= node; id++) xp_free(&memo[id]);
    free(memo);
    return rc;
}

tr_prov_poly *tr_prov_expand(const tr_prov_store *s, uint32_t node, const tr_math_budget *budget, tr_math_refusal *refusal,
                             int32_t *error) {
    tr_budget_none(refusal);
    if (!store_valid(s) || node >= s->nnodes) { set_error(error, TR_MATH_INVALID); return NULL; }
    tr_math_budget dflt;
    const tr_math_budget *b = budget_of(budget, &dflt);
    int32_t rc = check_bytes(b, tr_budget_mul((uint64_t)node + 1, 17), refusal);
    if (rc) { set_error(error, rc); return NULL; }
    tr_math_need need;
    uint8_t *mark = NULL;
    uint64_t *d = NULL, *deg = NULL;
    rc = expand_plan(s, node, 0, &need, &mark, &d, &deg);
    if (!rc) rc = check_need(b, &need, refusal);
    xpoly x;
    memset(&x, 0, sizeof x);
    if (!rc) rc = expand_into(s, node, mark, &x);
    free(mark); free(d); free(deg);
    if (rc) { xp_free(&x); set_error(error, rc); return NULL; }
    tr_prov_poly *p = poly_adopt(&x, TR_PROV_POLY, &rc);
    set_error(error, rc);
    return p;
}

/* 1 when the variable set of monomial i of a is a subset of monomial j's */
static int mono_subset(const xpoly *a, uint64_t i, uint64_t j) {
    uint64_t x = a->off[i], xe = a->off[i + 1], y = a->off[j], ye = a->off[j + 1];
    for (; x < xe; x++) {
        while (y < ye && a->var[y] < a->var[x]) y++;
        if (y >= ye || a->var[y] != a->var[x]) return 0;
    }
    return 1;
}

tr_prov_poly *tr_prov_witnesses(const tr_prov_store *s, uint32_t node, const tr_math_budget *budget,
                                tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    if (!store_valid(s) || node >= s->nnodes) { set_error(error, TR_MATH_INVALID); return NULL; }
    tr_math_budget dflt;
    const tr_math_budget *b = budget_of(budget, &dflt);
    int32_t rc = check_bytes(b, tr_budget_mul((uint64_t)node + 1, 17), refusal);
    if (rc) { set_error(error, rc); return NULL; }
    tr_math_need need;
    uint8_t *mark = NULL;
    uint64_t *d = NULL, *deg = NULL;
    rc = expand_plan(s, node, 1, &need, &mark, &d, &deg);
    if (!rc) rc = check_need(b, &need, refusal);
    xpoly x, sets, out;
    memset(&x, 0, sizeof x); memset(&sets, 0, sizeof sets); memset(&out, 0, sizeof out);
    if (!rc) rc = expand_into(s, node, mark, &x);
    free(mark); free(d); free(deg);
    /* the variable sets (exponents 1, coefficient 1), deduplicated, then the minimal ones under inclusion */
    if (!rc) {
        for (uint64_t i = 0; i < x.n; i++) x.coef[i] = 1;
        for (uint64_t k = 0; k < x.pairs; k++) x.exp[k] = 1;
        rc = xp_normalize(&x, &sets);
        for (uint64_t i = 0; !rc && i < sets.n; i++) sets.coef[i] = 1;
    }
    xp_free(&x);
    uint8_t *drop = rc ? NULL : zalloc(sets.n);
    if (!rc && !drop) rc = TR_MATH_MEMORY;
    uint64_t keep = 0, pairs = 0;
    for (uint64_t i = 0; !rc && i < sets.n; i++) {
        uint64_t li = sets.off[i + 1] - sets.off[i];
        for (uint64_t j = 0; j < sets.n && !drop[i]; j++)
            if (j != i && sets.off[j + 1] - sets.off[j] < li && mono_subset(&sets, j, i)) drop[i] = 1;
        if (!drop[i]) { keep++; pairs += li; }
    }
    if (!rc) rc = xp_alloc(&out, keep, pairs);
    uint64_t t = 0, at = 0;
    for (uint64_t i = 0; !rc && i < sets.n; i++) {
        if (drop[i]) continue;
        uint64_t s0 = sets.off[i], len = sets.off[i + 1] - s0;
        out.coef[t] = 1; out.off[t] = at;
        memcpy(out.var + at, sets.var + s0, (size_t)len * 4);
        memcpy(out.exp + at, sets.exp + s0, (size_t)len * 4);
        at += len; t++;
    }
    if (!rc) out.off[t] = at;
    free(drop); xp_free(&sets);
    if (rc) { xp_free(&out); set_error(error, rc); return NULL; }
    tr_prov_poly *p = poly_adopt(&out, TR_PROV_BOOL, &rc);
    set_error(error, rc);
    return p;
}

int32_t tr_prov_poly_get_info(const tr_prov_poly *p, tr_prov_poly_info *out) {
    if (!poly_valid(p) || !out) return TR_MATH_INVALID;
    memset(out, 0, sizeof *out);
    out->terms = p->n; out->pairs = p->pairs; out->derivations = p->derivations; out->degree = p->degree;
    out->semiring = p->semiring;
    return TR_MATH_OK;
}
int32_t tr_prov_poly_get(const tr_prov_poly *p, uint64_t *coef, uint64_t *offset, uint32_t *var, uint32_t *exp,
                         uint64_t terms_capacity, uint64_t pairs_capacity) {
    if (!poly_valid(p) || terms_capacity < p->n || pairs_capacity < p->pairs || !offset || (p->n && !coef) ||
        (p->pairs && (!var || !exp))) return TR_MATH_INVALID;
    if (p->n) memcpy(coef, p->coef, (size_t)p->n * 8);
    memcpy(offset, p->off, (size_t)(p->n + 1) * 8);
    if (p->pairs) { memcpy(var, p->var, (size_t)p->pairs * 4); memcpy(exp, p->exp, (size_t)p->pairs * 4); }
    return TR_MATH_OK;
}
