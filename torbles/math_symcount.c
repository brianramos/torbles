/* COUNT-002: symbolic full-domain model counting (decision diagrams, brute force, bounded Shannon expansion,
 * variable elimination over factors); see math_symcount.h.
 * SPDX-License-Identifier: Apache-2.0 */
#include "math_symcount.h"
#include "mathematics.h"
#include "math_domain.h"
#include "math_carriers.h"
#include "math_handles.h"
#include <stdlib.h>
#include <string.h>

TR_MATH_ABI(tr_symc_node, TR_SYMC_ABI);             /* the ctypes boundary of COUNT-002 (math_handles.h) */
TR_MATH_ABI(tr_symc_weight, TR_SYMC_ABI);
TR_MATH_ABI(tr_symc_dd_view, TR_SYMC_ABI);
TR_MATH_ABI(tr_symc_dd_info, TR_SYMC_ABI);
TR_MATH_ABI(tr_symc_answer_info, TR_SYMC_ABI);
TR_MATH_ABI(tr_symc_factor, TR_SYMC_ABI);
TR_MATH_ABI(tr_symc_factors_info, TR_SYMC_ABI);

_Static_assert(sizeof(tr_symc_node) == 16, "tr_symc_node ABI");
_Static_assert(sizeof(tr_symc_weight) == 32, "tr_symc_weight ABI");
_Static_assert(sizeof(tr_symc_dd_view) == 64, "tr_symc_dd_view ABI");
_Static_assert(sizeof(tr_symc_dd_info) == 56, "tr_symc_dd_info ABI");
_Static_assert(sizeof(tr_symc_answer_info) == 80, "tr_symc_answer_info ABI");
_Static_assert(sizeof(tr_symc_factor) == 24, "tr_symc_factor ABI");
_Static_assert(sizeof(tr_symc_factors_info) == 48, "tr_symc_factors_info ABI");

#define MGR_MAGIC UINT32_C(0x4d475253)
#define DD_MAGIC UINT32_C(0x44445953)
#define ANS_MAGIC UINT32_C(0x414e5359)
#define FAC_MAGIC UINT32_C(0x46414353)
#define NONE32 UINT32_MAX
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
static uint64_t pow2_at_least(uint64_t x) { uint64_t p = 1; while (p < x && p < (UINT64_C(1) << 62)) p <<= 1; return p; }
static uint64_t mix3(uint64_t a, uint64_t b, uint64_t c) {
    uint64_t h = a * UINT64_C(0x9e3779b97f4a7c15) ^ (b + UINT64_C(0x7f4a7c159e3779b9)) * UINT64_C(0xbf58476d1ce4e5b9);
    h ^= c * UINT64_C(0x94d049bb133111eb);
    h ^= h >> 31; h *= UINT64_C(0xd6e8feb86659fd93); h ^= h >> 32;
    return h;
}
static uint32_t bitlen128(u128 x) { uint32_t n = 0; while (x) { n++; x >>= 1; } return n; }
static u128 gcd128(u128 a, u128 b) { while (b) { u128 t = a % b; a = b; b = t; } return a; }

/* ================================================================== the manager (builder) */
typedef struct { uint32_t f, g, r, op; } centry;            /* computed table (op 0: empty) */
typedef struct { uint32_t f, g, lev, r0, st; } frame;

struct tr_symc_mgr_s {
    uint32_t magic, nvars, nlevels, reserved;
    uint32_t *order, *level_of;
    tr_symc_node *node;
    uint64_t nnodes, ncap;
    uint32_t *hash;
    uint64_t hcap;
    centry *cache;
    uint64_t ccap;
    frame *stack;
    uint64_t steps;
    uint64_t limit_nodes, limit_steps, limit_bytes;
};

static int mgr_valid(const tr_symc_mgr *m) { return m && m->magic == MGR_MAGIC; }
static uint64_t mgr_bytes(uint64_t nvars, uint64_t nlevels, uint64_t ncap, uint64_t hcap, uint64_t ccap) {
    uint64_t b = tr_budget_add(sizeof(tr_symc_mgr), tr_budget_mul(ncap, sizeof(tr_symc_node)));
    b = tr_budget_add(b, tr_budget_mul(hcap, 4));
    b = tr_budget_add(b, tr_budget_mul(ccap, sizeof(centry)));
    b = tr_budget_add(b, tr_budget_mul(tr_budget_add(nlevels, 2), sizeof(frame)));
    return tr_budget_add(b, tr_budget_add(tr_budget_mul(nlevels, 4), tr_budget_mul(nvars, 4)));
}
static void mgr_release(tr_symc_mgr *m) {
    free(m->order); free(m->level_of); free(m->node); free(m->hash); free(m->cache); free(m->stack);
    m->magic = 0;
    free(m);
}
static void hash_insert(tr_symc_mgr *m, uint32_t id) {
    const tr_symc_node *n = &m->node[id];
    uint64_t mask = m->hcap - 1, slot = mix3(n->level, n->lo, n->hi) & mask;
    while (m->hash[slot] != NONE32) slot = (slot + 1) & mask;
    m->hash[slot] = id;
}
/* room for one more node: arrays doubled (the limits checked first) */
static int32_t mgr_grow(tr_symc_mgr *m, tr_math_refusal *r) {
    uint64_t ncap = tr_budget_mul(m->ncap, 2);
    if (ncap > m->limit_nodes) ncap = m->limit_nodes;
    if (ncap <= m->nnodes) ncap = m->nnodes + 1;
    uint64_t hcap = pow2_at_least(tr_budget_mul(ncap, 2)), ccap = pow2_at_least(ncap);
    uint64_t bytes = mgr_bytes(m->nvars, m->nlevels, ncap, hcap, ccap);
    if (bytes > m->limit_bytes) {
        if (r) { r->resource = TR_BUDGET_BYTES; r->refused = 1; r->need = bytes; r->limit = m->limit_bytes; }
        return TR_MATH_BUDGET;
    }
    tr_symc_node *nd = (tr_symc_node *)realloc(m->node, (size_t)ncap * sizeof *nd);
    if (!nd) return TR_MATH_MEMORY;
    m->node = nd; m->ncap = ncap;
    uint32_t *h = (uint32_t *)malloc((size_t)hcap * 4);
    centry *c = (centry *)calloc((size_t)ccap, sizeof *c);
    if (!h || !c) { free(h); free(c); return TR_MATH_MEMORY; }
    free(m->hash); free(m->cache);
    m->hash = h; m->hcap = hcap; m->cache = c; m->ccap = ccap;
    memset(m->hash, 0xff, (size_t)hcap * 4);
    for (uint64_t i = 2; i < m->nnodes; i++) hash_insert(m, (uint32_t)i);
    return TR_MATH_OK;
}
/* the unique node (level, lo, hi): lo == hi is lo itself */
static int32_t mgr_mk(tr_symc_mgr *m, uint32_t level, uint32_t lo, uint32_t hi, uint32_t *out, tr_math_refusal *r) {
    if (lo == hi) { *out = lo; return TR_MATH_OK; }
    uint64_t mask = m->hcap - 1, slot = mix3(level, lo, hi) & mask;
    while (m->hash[slot] != NONE32) {
        const tr_symc_node *n = &m->node[m->hash[slot]];
        if (n->level == level && n->lo == lo && n->hi == hi) { *out = m->hash[slot]; return TR_MATH_OK; }
        slot = (slot + 1) & mask;
    }
    if (m->nnodes + 1 > m->limit_nodes) {
        if (r) { r->resource = TR_BUDGET_CIRCUIT; r->refused = 1; r->need = m->nnodes + 1; r->limit = m->limit_nodes; }
        return TR_MATH_BUDGET;
    }
    if (m->nnodes >= (uint64_t)NONE32 - 1) return TR_MATH_UNSUPPORTED;
    if (m->nnodes == m->ncap) {
        int32_t rc = mgr_grow(m, r);
        if (rc) return rc;
        mask = m->hcap - 1; slot = mix3(level, lo, hi) & mask;
        while (m->hash[slot] != NONE32) slot = (slot + 1) & mask;
    }
    uint32_t id = (uint32_t)m->nnodes++;
    m->node[id].level = level; m->node[id].lo = lo; m->node[id].hi = hi; m->node[id].reserved = 0;
    m->hash[slot] = id;
    *out = id;
    return TR_MATH_OK;
}
static uint32_t mgr_level(const tr_symc_mgr *m, uint32_t id) { return id < 2 ? m->nlevels : m->node[id].level; }
static int apply_terminal(int32_t op, uint32_t f, uint32_t g, uint32_t *res) {
    switch (op) {
    case TR_SYMC_AND:
        if (f == TR_SYMC_FALSE || g == TR_SYMC_FALSE) { *res = TR_SYMC_FALSE; return 1; }
        if (f == TR_SYMC_TRUE) { *res = g; return 1; }
        if (g == TR_SYMC_TRUE || f == g) { *res = f; return 1; }
        return 0;
    case TR_SYMC_OR:
        if (f == TR_SYMC_TRUE || g == TR_SYMC_TRUE) { *res = TR_SYMC_TRUE; return 1; }
        if (f == TR_SYMC_FALSE) { *res = g; return 1; }
        if (g == TR_SYMC_FALSE || f == g) { *res = f; return 1; }
        return 0;
    default:
        if (f == g) { *res = TR_SYMC_FALSE; return 1; }
        if (f == TR_SYMC_FALSE) { *res = g; return 1; }
        if (g == TR_SYMC_FALSE) { *res = f; return 1; }
        return 0;
    }
}

tr_symc_mgr *tr_symc_mgr_create(uint32_t nvars, uint32_t nlevels, const uint32_t *order, const tr_math_budget *budget,
                                tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    if ((nlevels && !order) || nlevels > nvars) { set_error(error, TR_MATH_INVALID); return NULL; }
    if (nlevels > TR_SYMC_MAX_LEVELS) { set_error(error, TR_MATH_UNSUPPORTED); return NULL; }
    tr_math_budget dflt;
    const tr_math_budget *b = budget_of(budget, &dflt);
    uint64_t ncap = b->limit[TR_BUDGET_CIRCUIT] < 64 ? b->limit[TR_BUDGET_CIRCUIT] : 64;
    if (ncap < 2) ncap = 2;
    uint64_t hcap = pow2_at_least(2 * ncap), ccap = pow2_at_least(ncap);
    tr_math_need need;
    need_zero(&need);
    need.need[TR_BUDGET_CIRCUIT] = 2;
    need.need[TR_BUDGET_BYTES] = mgr_bytes(nvars, nlevels, ncap, hcap, ccap);
    int32_t rc = check_need(b, &need, refusal);
    if (rc) { set_error(error, rc); return NULL; }
    tr_symc_mgr *m = (tr_symc_mgr *)zalloc(sizeof *m);
    if (!m) { set_error(error, TR_MATH_MEMORY); return NULL; }
    m->magic = MGR_MAGIC; m->nvars = nvars; m->nlevels = nlevels;
    m->limit_nodes = b->limit[TR_BUDGET_CIRCUIT]; m->limit_steps = b->limit[TR_BUDGET_ITERATIONS];
    m->limit_bytes = b->limit[TR_BUDGET_BYTES];
    m->order = (uint32_t *)zalloc(4 * (uint64_t)(nlevels ? nlevels : 1));
    m->level_of = (uint32_t *)zalloc(4 * (uint64_t)(nvars ? nvars : 1));
    m->node = (tr_symc_node *)zalloc(ncap * sizeof(tr_symc_node));
    m->hash = (uint32_t *)malloc((size_t)hcap * 4);
    m->cache = (centry *)zalloc(ccap * sizeof(centry));
    m->stack = (frame *)zalloc(((uint64_t)nlevels + 2) * sizeof(frame));
    if (!m->order || !m->level_of || !m->node || !m->hash || !m->cache || !m->stack) {
        mgr_release(m); set_error(error, TR_MATH_MEMORY); return NULL;
    }
    m->ncap = ncap; m->hcap = hcap; m->ccap = ccap;
    memset(m->hash, 0xff, (size_t)hcap * 4);
    for (uint32_t v = 0; v < nvars; v++) m->level_of[v] = NONE32;
    for (uint32_t l = 0; l < nlevels; l++) {
        if (order[l] >= nvars || m->level_of[order[l]] != NONE32) { mgr_release(m); set_error(error, TR_MATH_INVALID); return NULL; }
        m->order[l] = order[l];
        m->level_of[order[l]] = l;
    }
    m->node[0].level = nlevels; m->node[1].level = nlevels;      /* FALSE, TRUE */
    m->nnodes = 2;
    set_error(error, TR_MATH_OK);
    return m;
}
void tr_symc_mgr_free(tr_symc_mgr *m) { if (mgr_valid(m)) mgr_release(m); }

int32_t tr_symc_mgr_var(tr_symc_mgr *m, uint32_t var, uint32_t *node, tr_math_refusal *r) {
    tr_budget_none(r);
    if (!mgr_valid(m) || !node || var >= m->nvars || m->level_of[var] == NONE32) return TR_MATH_INVALID;
    return mgr_mk(m, m->level_of[var], TR_SYMC_FALSE, TR_SYMC_TRUE, node, r);
}

int32_t tr_symc_mgr_apply(tr_symc_mgr *m, int32_t op, uint32_t f, uint32_t g, uint32_t *out, tr_math_refusal *r) {
    tr_budget_none(r);
    if (!mgr_valid(m) || !out || op < TR_SYMC_AND || op > TR_SYMC_XOR || f >= m->nnodes || g >= m->nnodes)
        return TR_MATH_INVALID;
    uint64_t sp = 0;
    uint32_t ret = 0;
    m->stack[sp].f = f; m->stack[sp].g = g; m->stack[sp].st = 0; sp++;
    while (sp) {
        frame *F = &m->stack[sp - 1];
        if (F->st == 0) {
            uint32_t res;
            if (apply_terminal(op, F->f, F->g, &res)) { ret = res; sp--; continue; }
            if (F->f > F->g) { uint32_t t = F->f; F->f = F->g; F->g = t; }
            centry *c = &m->cache[mix3((uint64_t)op, F->f, F->g) & (m->ccap - 1)];
            if (c->op == (uint32_t)op && c->f == F->f && c->g == F->g) { ret = c->r; sp--; continue; }
            if (m->steps + 1 > m->limit_steps) {
                if (r) { r->resource = TR_BUDGET_ITERATIONS; r->refused = 1; r->need = m->steps + 1; r->limit = m->limit_steps; }
                return TR_MATH_BUDGET;
            }
            m->steps++;
            uint32_t lf = mgr_level(m, F->f), lg = mgr_level(m, F->g), lev = lf < lg ? lf : lg;
            F->lev = lev; F->st = 1;
            uint32_t f0 = lf == lev ? m->node[F->f].lo : F->f, g0 = lg == lev ? m->node[F->g].lo : F->g;
            if (sp >= (uint64_t)m->nlevels + 2) return TR_MATH_INVALID;   /* unreachable: each push goes one level down */
            m->stack[sp].f = f0; m->stack[sp].g = g0; m->stack[sp].st = 0; sp++;
        } else if (F->st == 1) {
            F->r0 = ret; F->st = 2;
            uint32_t lf = mgr_level(m, F->f), lg = mgr_level(m, F->g);
            uint32_t f1 = lf == F->lev ? m->node[F->f].hi : F->f, g1 = lg == F->lev ? m->node[F->g].hi : F->g;
            if (sp >= (uint64_t)m->nlevels + 2) return TR_MATH_INVALID;
            m->stack[sp].f = f1; m->stack[sp].g = g1; m->stack[sp].st = 0; sp++;
        } else {
            uint32_t res;
            int32_t rc = mgr_mk(m, F->lev, F->r0, ret, &res, r);
            if (rc) return rc;
            F = &m->stack[sp - 1];
            centry *c = &m->cache[mix3((uint64_t)op, F->f, F->g) & (m->ccap - 1)];
            c->op = (uint32_t)op; c->f = F->f; c->g = F->g; c->r = res;
            ret = res;
            sp--;
        }
    }
    *out = ret;
    return TR_MATH_OK;
}

int32_t tr_symc_mgr_stats(const tr_symc_mgr *m, uint64_t *nodes, uint64_t *steps) {
    if (!mgr_valid(m)) return TR_MATH_INVALID;
    if (nodes) *nodes = m->nnodes;
    if (steps) *steps = m->steps;
    return TR_MATH_OK;
}

/* ================================================================== immutable diagrams */
struct tr_symc_dd_s {
    uint32_t magic, nlevels, nroots, nvars, three_valued, reserved;
    tr_symc_node *node;
    uint64_t nnodes;
    uint32_t *roots, *order;
    uint64_t *first, *count;
    uint64_t peak, steps, bytes, width;
    tr_symc_dd_view v;
};
static int dd_valid(const tr_symc_dd *d) { return d && d->magic == DD_MAGIC; }
static void dd_release(tr_symc_dd *d) {
    free(d->node); free(d->roots); free(d->order); free(d->first); free(d->count);
    d->magic = 0;
    free(d);
}
void tr_symc_dd_free(tr_symc_dd *d) { if (dd_valid(d)) dd_release(d); }

tr_symc_dd *tr_symc_mgr_finish(const tr_symc_mgr *m, const uint32_t *roots, uint32_t nroots, uint32_t three_valued,
                               const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    if (!mgr_valid(m) || (nroots && !roots)) { set_error(error, TR_MATH_INVALID); return NULL; }
    for (uint32_t i = 0; i < nroots; i++) if (roots[i] >= m->nnodes) { set_error(error, TR_MATH_INVALID); return NULL; }
    tr_math_budget dflt;
    const tr_math_budget *b = budget_of(budget, &dflt);
    const uint64_t L = m->nlevels;
    uint64_t scratch = tr_budget_add(tr_budget_mul(m->nnodes, 5), tr_budget_mul(L, 16));
    int32_t rc = check_one(b, TR_BUDGET_BYTES, scratch, refusal);
    if (rc) { set_error(error, rc); return NULL; }
    uint8_t *mark = (uint8_t *)zalloc(m->nnodes);
    uint32_t *nid = (uint32_t *)zalloc(tr_budget_mul(m->nnodes, 4));
    uint64_t *cnt = (uint64_t *)zalloc(tr_budget_mul(L ? L : 1, 8)), *at = (uint64_t *)zalloc(tr_budget_mul(L ? L : 1, 8));
    if (!mark || !nid || !cnt || !at) { free(mark); free(nid); free(cnt); free(at); set_error(error, TR_MATH_MEMORY); return NULL; }
    for (uint32_t i = 0; i < nroots; i++) mark[roots[i]] = 1;
    uint64_t live = 2;
    for (uint64_t id = m->nnodes; id-- > 2;) {                 /* children are older than their parents */
        if (!mark[id]) continue;
        mark[m->node[id].lo] = 1; mark[m->node[id].hi] = 1;
        cnt[m->node[id].level]++;
        live++;
    }
    tr_math_need need;
    need_zero(&need);
    need.need[TR_BUDGET_CIRCUIT] = live;
    need.need[TR_BUDGET_OUTPUT] = nroots;
    uint64_t bytes = tr_budget_add(tr_budget_add(sizeof(tr_symc_dd), tr_budget_mul(live, sizeof(tr_symc_node))),
                                   tr_budget_add(tr_budget_mul(nroots, 4), tr_budget_mul(L, 20)));
    need.need[TR_BUDGET_BYTES] = tr_budget_add(bytes, scratch);
    rc = check_need(b, &need, refusal);
    tr_symc_dd *d = rc ? NULL : (tr_symc_dd *)zalloc(sizeof *d);
    if (!rc && !d) rc = TR_MATH_MEMORY;
    if (d) {
        d->node = (tr_symc_node *)zalloc(tr_budget_mul(live, sizeof(tr_symc_node)));
        d->roots = (uint32_t *)zalloc(4 * (uint64_t)(nroots ? nroots : 1));
        d->order = (uint32_t *)zalloc(4 * (L ? L : 1));
        d->first = (uint64_t *)zalloc(8 * (L ? L : 1));
        d->count = (uint64_t *)zalloc(8 * (L ? L : 1));
        if (!d->node || !d->roots || !d->order || !d->first || !d->count) rc = TR_MATH_MEMORY;
    }
    if (!rc) {
        uint64_t next = 2, width = 0;
        for (uint64_t l = L; l-- > 0;) { d->first[l] = next; d->count[l] = cnt[l]; at[l] = next; next += cnt[l]; if (cnt[l] > width) width = cnt[l]; }
        nid[0] = 0; nid[1] = 1;
        d->node[0].level = (uint32_t)L; d->node[1].level = (uint32_t)L;
        for (uint64_t id = 2; id < m->nnodes; id++) {         /* by level, within a level by creation */
            if (!mark[id]) continue;
            nid[id] = (uint32_t)at[m->node[id].level]++;
        }
        for (uint64_t id = 2; id < m->nnodes; id++) {
            if (!mark[id]) continue;
            tr_symc_node n = m->node[id];
            n.lo = nid[n.lo]; n.hi = nid[n.hi]; n.reserved = 0;
            d->node[nid[id]] = n;
        }
        for (uint32_t i = 0; i < nroots; i++) d->roots[i] = nid[roots[i]];
        if (L) memcpy(d->order, m->order, (size_t)L * 4);
        d->magic = DD_MAGIC; d->nlevels = (uint32_t)L; d->nroots = nroots; d->nvars = m->nvars;
        d->three_valued = three_valued ? 1 : 0; d->nnodes = live; d->peak = m->nnodes; d->steps = m->steps;
        d->bytes = bytes; d->width = width;
        d->v.node = d->node; d->v.roots = d->roots; d->v.order = d->order; d->v.first = d->first; d->v.count = d->count;
        d->v.nnodes = live; d->v.nlevels = (uint32_t)L; d->v.nroots = nroots; d->v.nvars = m->nvars;
        d->v.three_valued = d->three_valued;
    }
    free(mark); free(nid); free(cnt); free(at);
    if (rc) { if (d) dd_release(d); set_error(error, rc); return NULL; }
    set_error(error, TR_MATH_OK);
    return d;
}

int32_t tr_symc_dd_get_view(const tr_symc_dd *d, tr_symc_dd_view *v) {
    if (!dd_valid(d) || !v) return TR_MATH_INVALID;
    *v = d->v;
    return TR_MATH_OK;
}
int32_t tr_symc_dd_get_info(const tr_symc_dd *d, tr_symc_dd_info *out) {
    if (!dd_valid(d) || !out) return TR_MATH_INVALID;
    memset(out, 0, sizeof *out);
    out->nnodes = d->nnodes; out->peak_nodes = d->peak; out->steps = d->steps; out->bytes = d->bytes;
    out->width = d->width; out->nlevels = d->nlevels; out->nroots = d->nroots; out->nvars = d->nvars;
    out->three_valued = d->three_valued;
    return TR_MATH_OK;
}

/* ================================================================== programs */
int32_t tr_symc_program_order(const tr_bool_program_view *p, int32_t kind, const uint32_t *given, uint32_t *out) {
    if (!p || !out || tr_bool_program_check(p)) return TR_MATH_INVALID;
    const int32_t k = p->k;
    if (k > TR_SYMC_MAX_PROGRAM_VARS) return TR_MATH_UNSUPPORTED;
    uint8_t seen[TR_SYMC_MAX_PROGRAM_VARS];
    memset(seen, 0, sizeof seen);
    if (kind == TR_SYMC_ORDER_NATURAL) { for (int32_t i = 0; i < k; i++) out[i] = (uint32_t)i; return TR_MATH_OK; }
    if (kind == TR_SYMC_ORDER_GIVEN) {
        if (k && !given) return TR_MATH_INVALID;
        for (int32_t i = 0; i < k; i++) {
            if (given[i] >= (uint32_t)k || seen[given[i]]) return TR_MATH_INVALID;
            seen[given[i]] = 1; out[i] = given[i];
        }
        return TR_MATH_OK;
    }
    if (kind != TR_SYMC_ORDER_DFS) return TR_MATH_INVALID;
    uint8_t *vis = (uint8_t *)zalloc((uint64_t)p->nnodes);
    uint32_t *stack = (uint32_t *)zalloc(4 * ((uint64_t)p->nnodes * 2 + 2));
    if (!vis || !stack) { free(vis); free(stack); return TR_MATH_MEMORY; }
    uint64_t sp = 0;
    int32_t n = 0;
    stack[sp++] = (uint32_t)p->root;
    while (sp) {
        uint32_t i = stack[--sp];
        if (vis[i]) continue;
        vis[i] = 1;
        const tr_bool_node *nd = &p->nodes[i];
        if (nd->op == TR_BOOL_VAR) { if (!seen[nd->a]) { seen[nd->a] = 1; out[n++] = (uint32_t)nd->a; } continue; }
        if (nd->op == TR_BOOL_CONST) continue;
        if (nd->op >= TR_BOOL_AND && !vis[nd->b]) stack[sp++] = (uint32_t)nd->b;   /* the left operand first */
        if (!vis[nd->a]) stack[sp++] = (uint32_t)nd->a;
    }
    for (int32_t i = 0; i < k; i++) if (!seen[i]) out[n++] = (uint32_t)i;
    free(vis); free(stack);
    return TR_MATH_OK;
}

tr_symc_dd *tr_symc_compile(const tr_bool_program_view *p, int32_t order_kind, const uint32_t *given,
                            const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    if (!p) { set_error(error, TR_MATH_INVALID); return NULL; }
    int32_t rc = tr_bool_program_check(p);
    if (rc) { set_error(error, rc); return NULL; }
    if (p->k > TR_SYMC_MAX_PROGRAM_VARS) { set_error(error, TR_MATH_UNSUPPORTED); return NULL; }
    uint32_t order[TR_SYMC_MAX_PROGRAM_VARS];
    rc = tr_symc_program_order(p, order_kind, given, order);
    if (rc) { set_error(error, rc); return NULL; }
    tr_math_budget dflt;
    const tr_math_budget *b = budget_of(budget, &dflt);
    rc = check_one(b, TR_BUDGET_BYTES, tr_budget_mul((uint64_t)p->nnodes, 8), refusal);
    if (rc) { set_error(error, rc); return NULL; }
    int three = 0;
    for (int32_t s = 0; s < p->nsched; s++) {
        const tr_bool_node *nd = &p->nodes[p->schedule[s]];
        if (nd->op == TR_BOOL_CONST && nd->a == TR_BOOL_UNKNOWN) three = 1;
    }
    tr_symc_mgr *m = tr_symc_mgr_create((uint32_t)p->k, (uint32_t)p->k, order, b, refusal, &rc);
    if (!m) { set_error(error, rc); return NULL; }
    uint32_t *T = (uint32_t *)zalloc(tr_budget_mul((uint64_t)p->nnodes, 4)), *F = (uint32_t *)zalloc(tr_budget_mul((uint64_t)p->nnodes, 4));
    if (!T || !F) rc = TR_MATH_MEMORY;
    for (int32_t s = 0; s < p->nsched && !rc; s++) {
        uint32_t i = p->schedule[s];
        const tr_bool_node *nd = &p->nodes[i];
        uint32_t t = 0, f = 0, x = 0, y = 0;
        switch (nd->op) {
        case TR_BOOL_CONST:
            t = nd->a == TR_BOOL_TRUE ? TR_SYMC_TRUE : TR_SYMC_FALSE;
            f = nd->a == TR_BOOL_FALSE ? TR_SYMC_TRUE : TR_SYMC_FALSE;
            break;
        case TR_BOOL_VAR:
            rc = tr_symc_mgr_var(m, (uint32_t)nd->a, &t, refusal);
            if (!rc && three) rc = tr_symc_mgr_apply(m, TR_SYMC_XOR, t, TR_SYMC_TRUE, &f, refusal);
            break;
        case TR_BOOL_NOT:
            if (three) { t = F[nd->a]; f = T[nd->a]; }
            else rc = tr_symc_mgr_apply(m, TR_SYMC_XOR, T[nd->a], TR_SYMC_TRUE, &t, refusal);
            break;
        case TR_BOOL_AND:
            rc = tr_symc_mgr_apply(m, TR_SYMC_AND, T[nd->a], T[nd->b], &t, refusal);
            if (!rc && three) rc = tr_symc_mgr_apply(m, TR_SYMC_OR, F[nd->a], F[nd->b], &f, refusal);
            break;
        case TR_BOOL_OR:
            rc = tr_symc_mgr_apply(m, TR_SYMC_OR, T[nd->a], T[nd->b], &t, refusal);
            if (!rc && three) rc = tr_symc_mgr_apply(m, TR_SYMC_AND, F[nd->a], F[nd->b], &f, refusal);
            break;
        default:                                                     /* XOR */
            if (!three) { rc = tr_symc_mgr_apply(m, TR_SYMC_XOR, T[nd->a], T[nd->b], &t, refusal); break; }
            rc = tr_symc_mgr_apply(m, TR_SYMC_AND, T[nd->a], F[nd->b], &x, refusal);
            if (!rc) rc = tr_symc_mgr_apply(m, TR_SYMC_AND, F[nd->a], T[nd->b], &y, refusal);
            if (!rc) rc = tr_symc_mgr_apply(m, TR_SYMC_OR, x, y, &t, refusal);
            if (!rc) rc = tr_symc_mgr_apply(m, TR_SYMC_AND, T[nd->a], T[nd->b], &x, refusal);
            if (!rc) rc = tr_symc_mgr_apply(m, TR_SYMC_AND, F[nd->a], F[nd->b], &y, refusal);
            if (!rc) rc = tr_symc_mgr_apply(m, TR_SYMC_OR, x, y, &f, refusal);
            break;
        }
        if (!rc) { T[i] = t; F[i] = f; }
    }
    tr_symc_dd *d = NULL;
    if (!rc) {
        uint32_t roots[2] = {T[p->root], F[p->root]};
        d = tr_symc_mgr_finish(m, roots, three ? 2 : 1, (uint32_t)three, b, refusal, &rc);
    }
    free(T); free(F);
    tr_symc_mgr_free(m);
    if (!d) { set_error(error, rc ? rc : TR_MATH_MEMORY); return NULL; }
    set_error(error, TR_MATH_OK);
    return d;
}

/* ================================================================== weights and limbs */
/* the scaled integers of one literal pair: a = w(1) d, b = w(0) d over d = lcm(den1, den0), reduced by gcd(a, b, d);
 * NULL: unit weights */
static int32_t scale_weight(const tr_symc_weight *w, u128 *a, u128 *b, u128 *d) {
    if (!w) { *a = 1; *b = 1; *d = 1; return TR_MATH_OK; }
    if (w->num1 < 0 || w->num0 < 0 || w->den1 < 1 || w->den0 < 1) return TR_MATH_INVALID;
    uint64_t d1 = (uint64_t)w->den1, d0 = (uint64_t)w->den0, g = tr_gcd_u64(d1, d0);
    u128 dd = (u128)(d1 / g) * d0;
    u128 aa = (u128)(uint64_t)w->num1 * (d0 / g), bb = (u128)(uint64_t)w->num0 * (d1 / g);
    u128 h = gcd128(gcd128(aa, bb), dd);
    if (h > 1) { aa /= h; bb /= h; dd /= h; }
    *a = aa; *b = bb; *d = dd;
    return TR_MATH_OK;
}
static void put128(uint64_t *w, u128 x) { w[0] = (uint64_t)x; w[1] = (uint64_t)(x >> 64); }
static u128 get128(const uint64_t *w) { return ((u128)w[1] << 64) | w[0]; }
/* scaled weights of n variables vars[0..n) (vars NULL: 0..n-1) into scaled[4 n] and the limbs that hold every count
 * and the denominator */
static int32_t scale_all(const tr_symc_weight *weights, const uint32_t *vars, uint64_t n, uint64_t *scaled,
                         uint32_t *limbs) {
    uint64_t nbits = 1, dbits = 1;
    for (uint64_t i = 0; i < n; i++) {
        uint64_t v = vars ? vars[i] : i;
        u128 a, b, d;
        int32_t rc = scale_weight(weights ? &weights[v] : NULL, &a, &b, &d);
        if (rc) return rc;
        if (scaled) { put128(scaled + 4 * i, a); put128(scaled + 4 * i + 2, b); }
        nbits += bitlen128(a + b);
        dbits += bitlen128(d);
    }
    uint64_t bits = nbits > dbits ? nbits : dbits, L = bits / 64 + 1;
    if (L > TR_SYMC_MAX_LIMBS) return TR_MATH_UNSUPPORTED;
    if (limbs) *limbs = (uint32_t)L;
    return TR_MATH_OK;
}
/* prod d_v and prod s_v of the variables, L limbs each */
static int32_t denominators(const tr_symc_weight *weights, const uint32_t *vars, uint64_t n, uint32_t L, uint64_t *D,
                            uint64_t *S, uint64_t *t) {
    tr_symc_w_zero(D, L); D[0] = 1;
    tr_symc_w_zero(S, L); S[0] = 1;
    int over = 0;
    for (uint64_t i = 0; i < n; i++) {
        uint64_t v = vars ? vars[i] : i;
        u128 a, b, d;
        int32_t rc = scale_weight(weights ? &weights[v] : NULL, &a, &b, &d);
        if (rc) return rc;
        over |= tr_symc_w_scale(D, t, L, (uint64_t)d, (uint64_t)(d >> 64));
        u128 s = a + b;
        over |= tr_symc_w_scale(S, t, L, (uint64_t)s, (uint64_t)(s >> 64));
    }
    return over ? TR_MATH_OVERFLOW : TR_MATH_OK;
}

/* ================================================================== answers */
struct tr_symc_answer_s {
    uint32_t magic;
    tr_symc_answer_info info;
    tr_bigq **values;
    uint64_t *witnesses;
    uint32_t *order;
    uint64_t norder;
};
static int ans_valid(const tr_symc_answer *a) { return a && a->magic == ANS_MAGIC; }
void tr_symc_answer_free(tr_symc_answer *a) {
    if (!ans_valid(a)) return;
    for (int32_t i = 0; i < a->info.nvalues; i++) tr_bigq_free(a->values[i]);
    free(a->values); free(a->witnesses); free(a->order);
    a->magic = 0;
    free(a);
}
static tr_symc_answer *ans_new(int32_t method, int32_t nvalues, uint32_t universe, const uint32_t *order,
                               uint64_t norder) {
    tr_symc_answer *a = (tr_symc_answer *)zalloc(sizeof *a);
    if (!a) return NULL;
    a->values = (tr_bigq **)zalloc(sizeof(tr_bigq *) * (uint64_t)nvalues);
    a->order = (uint32_t *)zalloc(4 * (norder ? norder : 1));
    if (!a->values || !a->order) { free(a->values); free(a->order); free(a); return NULL; }
    if (norder && order) memcpy(a->order, order, (size_t)norder * 4);
    a->norder = norder;
    a->magic = ANS_MAGIC;
    a->info.method = method; a->info.status = TR_SYMC_EXACT; a->info.nvalues = nvalues;
    a->info.universe_vars = universe; a->info.backend = TR_SYMC_BY_CPU;
    return a;
}
/* value i = num / den (L limbs each), canonical */
static int32_t ans_set(tr_symc_answer *a, int32_t i, const uint64_t *num, const uint64_t *den, uint32_t L,
                       const tr_math_budget *b, tr_math_refusal *r) {
    int32_t e = 0;
    tr_bigint *n = tr_bigint_from_limbs(1, num, L, b, r, &e);
    if (!n) return e ? e : TR_MATH_MEMORY;
    tr_bigint *d = tr_bigint_from_limbs(1, den, L, b, r, &e);
    if (!d) { tr_bigint_free(n); return e ? e : TR_MATH_MEMORY; }
    tr_bigq *q = tr_bigq_from_bigints(n, d, b, r, &e);
    tr_bigint_free(n); tr_bigint_free(d);
    if (!q) return e ? e : TR_MATH_MEMORY;
    a->values[i] = q;
    uint64_t ln = tr_bigint_limb_count(tr_bigq_numerator(q)), ld = tr_bigint_limb_count(tr_bigq_denominator(q));
    if (ln > a->info.limbs) a->info.limbs = (uint32_t)ln;
    if (ld > a->info.limbs) a->info.limbs = (uint32_t)ld;
    return TR_MATH_OK;
}
int32_t tr_symc_answer_get_info(const tr_symc_answer *a, tr_symc_answer_info *out) {
    if (!ans_valid(a) || !out) return TR_MATH_INVALID;
    *out = a->info;
    return TR_MATH_OK;
}
int32_t tr_symc_answer_value(const tr_symc_answer *a, uint32_t i, uint64_t *num, uint64_t *den, uint32_t cap,
                             uint32_t *nlimbs, uint32_t *dlimbs) {
    if (!ans_valid(a) || i >= (uint32_t)a->info.nvalues || !nlimbs || !dlimbs || (cap && (!num || !den)) ||
        cap < a->info.limbs) return TR_MATH_INVALID;
    int32_t sign = 0;
    uint64_t c1 = 0, c2 = 0;
    int32_t rc = tr_bigint_export(tr_bigq_numerator(a->values[i]), &sign, num, cap, &c1);
    if (!rc) rc = tr_bigint_export(tr_bigq_denominator(a->values[i]), &sign, den, cap, &c2);
    if (rc) return rc;
    *nlimbs = (uint32_t)c1; *dlimbs = (uint32_t)c2;
    return TR_MATH_OK;
}
int32_t tr_symc_answer_witnesses(const tr_symc_answer *a, uint64_t *out, uint64_t capacity) {
    if (!ans_valid(a)) return TR_MATH_INVALID;
    uint64_t n = tr_budget_mul(a->info.witnesses, a->info.witness_width);
    if (capacity < n || (n && !out)) return TR_MATH_INVALID;
    if (n) memcpy(out, a->witnesses, (size_t)n * 8);
    return TR_MATH_OK;
}
int32_t tr_symc_answer_order(const tr_symc_answer *a, uint32_t *out, uint64_t capacity) {
    if (!ans_valid(a) || capacity < a->norder || (a->norder && !out)) return TR_MATH_INVALID;
    if (a->norder) memcpy(out, a->order, (size_t)a->norder * 4);
    return TR_MATH_OK;
}
/* the class values [TRUE, FALSE, UNKNOWN, TOTAL] from the TRUE (and, three-valued, FALSE) numerators over D */
static int32_t ans_classes(tr_symc_answer *a, const uint64_t *tnum, const uint64_t *fnum, const uint64_t *S,
                           const uint64_t *D, uint32_t L, uint64_t *t1, uint64_t *t2, const tr_math_budget *b,
                           tr_math_refusal *r) {
    tr_symc_w_copy(t1, S, L);
    if (tr_symc_w_sub(t1, tnum, L)) return TR_MATH_INVALID;      /* TOTAL - TRUE */
    if (fnum) {
        tr_symc_w_copy(t2, t1, L);
        if (tr_symc_w_sub(t2, fnum, L)) return TR_MATH_INVALID;  /* UNKNOWN = TOTAL - TRUE - FALSE */
    } else tr_symc_w_zero(t2, L);
    int32_t rc = ans_set(a, 0, tnum, D, L, b, r);
    if (!rc) rc = ans_set(a, 1, fnum ? fnum : t1, D, L, b, r);
    if (!rc) rc = ans_set(a, 2, t2, D, L, b, r);
    if (!rc) rc = ans_set(a, 3, S, D, L, b, r);
    return rc;
}

/* ================================================================== diagram counts */
int32_t tr_symc_dd_prepare(const tr_symc_dd_view *d, const tr_symc_weight *weights, uint64_t *scaled, uint32_t *limbs) {
    if (!d || tr_symc_dd_view_check(d)) return TR_MATH_INVALID;
    return scale_all(weights, d->order, d->nlevels, scaled, limbs);
}
static void dd_count_needs(const tr_symc_dd_view *d, uint32_t L, int32_t target, tr_math_need *need) {
    need_zero(need);
    need->need[TR_BUDGET_CIRCUIT] = d->nnodes;
    need->need[TR_BUDGET_OUTPUT] = d->nroots;
    need->need[TR_BUDGET_BYTES] = target == TR_SYMC_NEED_DEVICE ? tr_symc_levels_device_bytes(d, L)
        : tr_budget_add(tr_budget_mul(tr_budget_add(d->nnodes, 6), tr_budget_mul(L, 8)), tr_budget_mul(d->nlevels, 32));
}
int32_t tr_symc_dd_count_need(const tr_symc_dd *dd, const tr_symc_weight *w, int32_t target, tr_math_need *need) {
    if (!dd_valid(dd) || !need) return TR_MATH_INVALID;
    uint32_t L = 0;
    int32_t rc = scale_all(w, dd->order, dd->nlevels, NULL, &L);
    if (rc) return rc;
    dd_count_needs(&dd->v, L, target, need);
    return TR_MATH_OK;
}
int32_t tr_symc_dd_levels(const tr_symc_dd_view *d, const uint64_t *scaled, uint32_t L, uint64_t *roots_out,
                          const tr_math_budget *budget, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!d || tr_symc_dd_view_check(d) || L < 1 || L > TR_SYMC_MAX_LIMBS || (d->nroots && !roots_out) ||
        tr_symc_scaled_check(scaled, d->nlevels)) return TR_MATH_INVALID;
    tr_math_budget dflt;
    const tr_math_budget *b = budget_of(budget, &dflt);
    tr_math_need need;
    dd_count_needs(d, L, TR_SYMC_NEED_CPU, &need);
    int32_t rc = check_need(b, &need, refusal);
    if (rc) return rc;
    uint64_t *values = (uint64_t *)zalloc(tr_budget_mul(tr_budget_mul(d->nnodes, L), 8));
    uint64_t *tmp = (uint64_t *)zalloc(tr_budget_mul(2 * (uint64_t)L, 8));
    if (!values || !tmp) { free(values); free(tmp); return TR_MATH_MEMORY; }
    tr_symc_terminals(values, L);
    int over = 0;
    for (uint64_t i = 2; i < d->nnodes; i++) over |= tr_symc_node_count(d, scaled, L, values, i, tmp);
    for (uint32_t r = 0; r < d->nroots; r++) over |= tr_symc_root_count(d, scaled, L, values, r, roots_out + (uint64_t)r * L, tmp);
    free(values); free(tmp);
    return over ? TR_MATH_OVERFLOW : TR_MATH_OK;
}
tr_symc_answer *tr_symc_dd_adopt(const tr_symc_dd *dd, const tr_symc_weight *weights, uint32_t L, const uint64_t *roots,
                                 int32_t backend, const tr_math_budget *budget, tr_math_refusal *refusal,
                                 int32_t *error) {
    tr_budget_none(refusal);
    if (!dd_valid(dd) || !roots || (backend != TR_SYMC_BY_CPU && backend != TR_SYMC_BY_DEVICE) ||
        dd->nroots != (dd->three_valued ? 2u : 1u)) { set_error(error, TR_MATH_INVALID); return NULL; }
    uint32_t want = 0;
    int32_t rc = scale_all(weights, dd->order, dd->nlevels, NULL, &want);
    if (rc || want != L) { set_error(error, rc ? rc : TR_MATH_INVALID); return NULL; }
    tr_math_budget dflt;
    const tr_math_budget *b = budget_of(budget, &dflt);
    rc = check_one(b, TR_BUDGET_BYTES, tr_budget_mul(5 * (uint64_t)L, 8), refusal);
    if (rc) { set_error(error, rc); return NULL; }
    uint64_t *w = (uint64_t *)zalloc(tr_budget_mul(5 * (uint64_t)L, 8));
    tr_symc_answer *a = ans_new(TR_SYMC_M_BDD, 4, dd->nvars, dd->order, dd->nlevels);
    if (!w || !a) { free(w); tr_symc_answer_free(a); set_error(error, TR_MATH_MEMORY); return NULL; }
    uint64_t *D = w, *S = w + L, *t = w + 2 * L, *t1 = w + 3 * L, *t2 = w + 4 * L;
    rc = denominators(weights, dd->order, dd->nlevels, L, D, S, t);
    if (!rc) rc = ans_classes(a, roots, dd->three_valued ? roots + L : NULL, S, D, L, t1, t2, b, refusal);
    free(w);
    if (rc) { tr_symc_answer_free(a); set_error(error, rc); return NULL; }
    a->info.weighted = weights != NULL; a->info.three_valued = (int32_t)dd->three_valued; a->info.backend = backend;
    a->info.nodes = dd->nnodes; a->info.width = dd->width;
    set_error(error, TR_MATH_OK);
    return a;
}
tr_symc_answer *tr_symc_dd_count(const tr_symc_dd *dd, const tr_symc_weight *weights, const tr_math_budget *budget,
                                 tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    if (!dd_valid(dd)) { set_error(error, TR_MATH_INVALID); return NULL; }
    uint32_t L = 0;
    int32_t rc = scale_all(weights, dd->order, dd->nlevels, NULL, &L);
    if (rc) { set_error(error, rc); return NULL; }
    tr_math_budget dflt;
    const tr_math_budget *b = budget_of(budget, &dflt);
    uint64_t *scaled = (uint64_t *)zalloc(tr_budget_mul(4 * (uint64_t)(dd->nlevels ? dd->nlevels : 1), 8));
    uint64_t *roots = (uint64_t *)zalloc(tr_budget_mul(tr_budget_mul(dd->nroots ? dd->nroots : 1, L), 8));
    if (!scaled || !roots) { free(scaled); free(roots); set_error(error, TR_MATH_MEMORY); return NULL; }
    rc = scale_all(weights, dd->order, dd->nlevels, scaled, &L);
    if (!rc) rc = tr_symc_dd_levels(&dd->v, scaled, L, roots, b, refusal);
    tr_symc_answer *a = rc ? NULL : tr_symc_dd_adopt(dd, weights, L, roots, TR_SYMC_BY_CPU, b, refusal, &rc);
    free(scaled); free(roots);
    set_error(error, a ? TR_MATH_OK : rc);
    return a;
}
int32_t tr_symc_dd_witnesses(const tr_symc_dd *dd, uint32_t root, uint64_t m, uint64_t *out, uint64_t *count,
                             const tr_math_budget *budget, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!dd_valid(dd) || !count || root >= dd->nroots || (m && !out)) return TR_MATH_INVALID;
    if (dd->nvars > 64) return TR_MATH_UNSUPPORTED;
    tr_math_budget dflt;
    const tr_math_budget *b = budget_of(budget, &dflt);
    tr_math_need need;
    need_zero(&need);
    need.need[TR_BUDGET_OUTPUT] = m;
    need.need[TR_BUDGET_BYTES] = tr_budget_add(tr_budget_mul((uint64_t)dd->nlevels + 2, 24), tr_budget_mul(m, 8));
    int32_t rc = check_need(b, &need, refusal);
    if (rc) return rc;
    typedef struct { uint64_t mask; uint32_t l, c, next, reserved; } wframe;
    wframe *st = (wframe *)zalloc(((uint64_t)dd->nlevels + 2) * sizeof(wframe));
    if (!st) return TR_MATH_MEMORY;
    uint64_t got = 0, sp = 0;
    uint32_t r = dd->roots[root];
    if (r != TR_SYMC_FALSE && m) { st[0].mask = 0; st[0].l = 0; st[0].c = r; st[0].next = 0; sp = 1; }
    while (sp && got < m) {
        wframe *F = &st[sp - 1];
        if (F->l == dd->nlevels) { out[got++] = F->mask; sp--; continue; }
        if (F->next == 2) { sp--; continue; }
        uint32_t bit = F->next++, lev = tr_symc_level(&dd->v, F->c), child;
        if (lev > F->l) child = F->c;                                   /* a skipped level: both values */
        else { child = bit ? dd->node[F->c].hi : dd->node[F->c].lo; if (child == TR_SYMC_FALSE) continue; }
        wframe nf;
        nf.mask = F->mask | ((uint64_t)bit << dd->order[F->l]); nf.l = F->l + 1; nf.c = child; nf.next = 0; nf.reserved = 0;
        st[sp++] = nf;
    }
    free(st);
    *count = got;
    return TR_MATH_OK;
}

/* ================================================================== brute force */
static int32_t brute_check(const tr_bool_program_view *p) {
    if (!p) return TR_MATH_INVALID;
    int32_t rc = tr_bool_program_check(p);
    if (rc) return rc;
    if (p->k > TR_SYMC_MAX_BRUTE_VARS) return TR_MATH_UNSUPPORTED;
    return TR_MATH_OK;
}
int32_t tr_symc_brute_prepare(const tr_bool_program_view *p, const tr_symc_weight *w, uint64_t *scaled, uint32_t *limbs) {
    int32_t rc = brute_check(p);
    if (rc) return rc;
    return scale_all(w, NULL, (uint64_t)p->k, scaled, limbs);
}
static void brute_needs(const tr_bool_program_view *p, uint32_t L, int weighted, int32_t target, tr_math_need *need) {
    need_zero(need);
    need->need[TR_BUDGET_VARIABLES] = (uint64_t)p->k;
    need->need[TR_BUDGET_SUPPORT] = (uint64_t)1 << p->k;
    need->need[TR_BUDGET_CIRCUIT] = (uint64_t)p->nnodes;
    need->need[TR_BUDGET_BYTES] = target == TR_SYMC_NEED_DEVICE ? tr_symc_brute_device_bytes(p, L, weighted)
        : tr_budget_add(tr_budget_mul((uint64_t)p->nnodes, 16), tr_budget_mul(8 * (uint64_t)L, 8));
}
int32_t tr_symc_brute_need(const tr_bool_program_view *p, const tr_symc_weight *w, int32_t target, tr_math_need *need) {
    uint32_t L = 0;
    int32_t rc = tr_symc_brute_prepare(p, w, NULL, &L);
    if (rc || !need) return rc ? rc : TR_MATH_INVALID;
    brute_needs(p, L, w != NULL, target, need);
    return TR_MATH_OK;
}
int32_t tr_symc_brute_raw(const tr_bool_program_view *p, const uint64_t *scaled, uint32_t L, uint64_t *counts,
                          uint64_t *wsum, const tr_math_budget *budget, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    int32_t rc = brute_check(p);
    if (rc) return rc;
    if (!counts || L < 1 || L > TR_SYMC_MAX_LIMBS || (scaled && (!wsum || tr_symc_scaled_check(scaled, (uint64_t)p->k))))
        return TR_MATH_INVALID;
    tr_math_budget dflt;
    const tr_math_budget *b = budget_of(budget, &dflt);
    tr_math_need need;
    brute_needs(p, L, scaled != NULL, TR_SYMC_NEED_CPU, &need);
    rc = check_need(b, &need, refusal);
    if (rc) return rc;
    uint64_t *val = (uint64_t *)zalloc(tr_budget_mul((uint64_t)p->nnodes, 16));
    uint64_t *scr = (uint64_t *)zalloc(tr_budget_mul(5 * (uint64_t)L, 8));
    if (!val || !scr) { free(val); free(scr); return TR_MATH_MEMORY; }
    const tr_bool_domain d = tr_symc_assignments(p->k);
    uint64_t c[3] = {0, 0, 0};
    uint64_t *st = scr, *sf = scr + L, *sc = scr + 2 * L;
    tr_symc_w_zero(st, 2 * (uint64_t)L);
    int over = 0;
    for (uint64_t w = 0; w < d.nw; w++) {
        uint64_t t, f, u;
        tr_symc_brute_classes(p, &d, w, val, &t, &f, &u);
        c[0] += (uint64_t)tr_bool_popc(t); c[1] += (uint64_t)tr_bool_popc(f); c[2] += (uint64_t)tr_bool_popc(u);
        if (scaled) {
            over |= tr_symc_brute_weigh(scaled, p->k, w, t, L, st, sc);
            over |= tr_symc_brute_weigh(scaled, p->k, w, f, L, sf, sc);
        }
    }
    counts[0] = c[0]; counts[1] = c[1]; counts[2] = c[2];
    if (scaled) tr_symc_w_copy(wsum, scr, 2 * L);
    free(val); free(scr);
    return over ? TR_MATH_OVERFLOW : TR_MATH_OK;
}
tr_symc_answer *tr_symc_brute_adopt(const tr_bool_program_view *p, const tr_symc_weight *weights, uint32_t L,
                                    const uint64_t *counts, const uint64_t *wsum, int32_t backend,
                                    const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    uint32_t want = 0;
    int32_t rc = tr_symc_brute_prepare(p, weights, NULL, &want);
    if (rc) { set_error(error, rc); return NULL; }
    if (want != L || !counts || (weights && !wsum) || (backend != TR_SYMC_BY_CPU && backend != TR_SYMC_BY_DEVICE)) {
        set_error(error, TR_MATH_INVALID); return NULL;
    }
    uint64_t total = (uint64_t)1 << p->k;
    if (counts[0] > total || counts[1] > total - counts[0] || counts[2] != total - counts[0] - counts[1]) {
        set_error(error, TR_MATH_INVALID); return NULL;                       /* not a partition of the universe */
    }
    tr_math_budget dflt;
    const tr_math_budget *b = budget_of(budget, &dflt);
    uint64_t *w = (uint64_t *)zalloc(tr_budget_mul(7 * (uint64_t)L, 8));
    tr_symc_answer *a = ans_new(TR_SYMC_M_BRUTE, 4, (uint32_t)p->k, NULL, 0);
    if (!w || !a) { free(w); tr_symc_answer_free(a); set_error(error, TR_MATH_MEMORY); return NULL; }
    uint64_t *D = w, *S = w + L, *t = w + 2 * L, *t1 = w + 3 * L, *t2 = w + 4 * L, *tn = w + 5 * L, *fn = w + 6 * L;
    if (weights) {
        rc = denominators(weights, NULL, (uint64_t)p->k, L, D, S, t);
        tr_symc_w_copy(tn, wsum, L); tr_symc_w_copy(fn, wsum + L, L);
    } else {
        tr_symc_w_zero(D, L); D[0] = 1;
        tr_symc_w_zero(S, L); S[0] = total;
        tr_symc_w_zero(tn, L); tn[0] = counts[0];
        tr_symc_w_zero(fn, L); fn[0] = counts[1];
    }
    if (!rc) rc = ans_classes(a, tn, fn, S, D, L, t1, t2, b, refusal);
    free(w);
    if (rc) { tr_symc_answer_free(a); set_error(error, rc); return NULL; }
    a->info.weighted = weights != NULL; a->info.backend = backend;
    a->info.evaluations = total;
    for (int32_t s = 0; s < p->nsched; s++)
        if (p->nodes[p->schedule[s]].op == TR_BOOL_CONST && p->nodes[p->schedule[s]].a == TR_BOOL_UNKNOWN) a->info.three_valued = 1;
    set_error(error, TR_MATH_OK);
    return a;
}
tr_symc_answer *tr_symc_brute(const tr_bool_program_view *p, const tr_symc_weight *weights, const tr_math_budget *budget,
                              tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    uint32_t L = 0;
    int32_t rc = tr_symc_brute_prepare(p, weights, NULL, &L);
    if (rc) { set_error(error, rc); return NULL; }
    uint64_t *scaled = (uint64_t *)zalloc(tr_budget_mul(4 * (uint64_t)(p->k ? p->k : 1), 8));
    uint64_t *wsum = (uint64_t *)zalloc(tr_budget_mul(2 * (uint64_t)L, 8));
    uint64_t counts[3];
    if (!scaled || !wsum) { free(scaled); free(wsum); set_error(error, TR_MATH_MEMORY); return NULL; }
    rc = tr_symc_brute_prepare(p, weights, scaled, &L);
    if (!rc) rc = tr_symc_brute_raw(p, weights ? scaled : NULL, L, counts, wsum, budget, refusal);
    tr_symc_answer *a = rc ? NULL : tr_symc_brute_adopt(p, weights, L, counts, wsum, TR_SYMC_BY_CPU, budget, refusal, &rc);
    free(scaled); free(wsum);
    set_error(error, a ? TR_MATH_OK : rc);
    return a;
}

/* ================================================================== bounded Shannon expansion */
enum { VF = 1, VT = 2, VU = 4 };                 /* possible values: FALSE, TRUE, UNKNOWN */
static uint8_t k_and(uint8_t a, uint8_t b) { if (a == VF || b == VF) return VF; if (a == VU || b == VU) return VU; return VT; }
static uint8_t k_or(uint8_t a, uint8_t b) { if (a == VT || b == VT) return VT; if (a == VU || b == VU) return VU; return VF; }
static uint8_t k_xor(uint8_t a, uint8_t b) { if (a == VU || b == VU) return VU; return a == b ? VF : VT; }
static uint8_t set_op(int32_t op, uint8_t x, uint8_t y) {
    uint8_t out = 0;
    for (uint8_t a = 1; a <= 4; a <<= 1) {
        if (!(x & a)) continue;
        if (op == TR_BOOL_NOT) { out |= a == VT ? VF : a == VF ? VT : VU; continue; }
        for (uint8_t b = 1; b <= 4; b <<= 1) {
            if (!(y & b)) continue;
            out |= op == TR_BOOL_AND ? k_and(a, b) : op == TR_BOOL_OR ? k_or(a, b) : k_xor(a, b);
        }
    }
    return out;
}
typedef struct { uint64_t assign; uint32_t depth; uint8_t parent, reserved[3]; } branch;

tr_symc_answer *tr_symc_search(const tr_bool_program_view *p, int32_t order_kind, const uint32_t *given,
                               const tr_symc_weight *weights, uint64_t max_eval, const tr_math_budget *budget,
                               tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    if (!p) { set_error(error, TR_MATH_INVALID); return NULL; }
    int32_t rc = tr_bool_program_check(p);
    if (rc) { set_error(error, rc); return NULL; }
    if (p->k > TR_SYMC_MAX_PROGRAM_VARS) { set_error(error, TR_MATH_UNSUPPORTED); return NULL; }
    const uint32_t k = (uint32_t)p->k;
    uint32_t order[TR_SYMC_MAX_PROGRAM_VARS], level_of[TR_SYMC_MAX_PROGRAM_VARS];
    rc = tr_symc_program_order(p, order_kind, given, order);
    if (rc) { set_error(error, rc); return NULL; }
    for (uint32_t l = 0; l < k; l++) level_of[order[l]] = l;
    uint32_t L = 0;
    rc = scale_all(weights, order, k, NULL, &L);
    if (rc) { set_error(error, rc); return NULL; }
    tr_math_budget dflt;
    const tr_math_budget *b = budget_of(budget, &dflt);
    uint64_t qcap = tr_budget_add(tr_budget_mul(max_eval, 2), 2);
    tr_math_need need;
    need_zero(&need);
    need.need[TR_BUDGET_VARIABLES] = k;
    need.need[TR_BUDGET_CIRCUIT] = (uint64_t)p->nnodes;
    need.need[TR_BUDGET_ITERATIONS] = max_eval;
    uint64_t bytes = tr_budget_add(tr_budget_mul(qcap, sizeof(branch)), tr_budget_mul((uint64_t)p->nnodes, 1));
    bytes = tr_budget_add(bytes, tr_budget_mul(tr_budget_add(tr_budget_mul((uint64_t)k + 1, L), tr_budget_mul(14, L)), 8));
    bytes = tr_budget_add(bytes, tr_budget_mul(4 * (uint64_t)(k ? k : 1), 8));
    need.need[TR_BUDGET_BYTES] = bytes;
    rc = check_need(b, &need, refusal);
    if (rc) { set_error(error, rc); return NULL; }
    branch *q = (branch *)zalloc(tr_budget_mul(qcap, sizeof(branch)));
    uint8_t *sets = (uint8_t *)zalloc((uint64_t)p->nnodes);
    uint64_t *scaled = (uint64_t *)zalloc(tr_budget_mul(4 * (uint64_t)(k ? k : 1), 8));
    uint64_t *suffix = (uint64_t *)zalloc(tr_budget_mul(tr_budget_mul((uint64_t)k + 1, L), 8));
    uint64_t *w = (uint64_t *)zalloc(tr_budget_mul(14 * (uint64_t)L, 8));
    tr_symc_answer *a = ans_new(TR_SYMC_M_SEARCH, 7, k, order, k);
    if (!q || !sets || !scaled || !suffix || !w || !a) rc = TR_MATH_MEMORY;
    uint64_t *lo = w, *hi = w + 3 * L, *D = w + 6 * L, *S = w + 7 * L, *t = w + 8 * L, *pre = w + 9 * L,
             *wt = w + 10 * L, *t2 = w + 11 * L;
    int over = 0;
    if (!rc) rc = scale_all(weights, order, k, scaled, &L);
    if (!rc) rc = denominators(weights, order, k, L, D, S, t);
    if (!rc) {
        tr_symc_w_zero(suffix + (uint64_t)k * L, L); suffix[(uint64_t)k * L] = 1;
        for (uint32_t l = k; l-- > 0;) {
            uint64_t slo, shi;
            tr_symc_level_sum(scaled, l, &slo, &shi);
            over |= tr_symc_w_mul128(suffix + (uint64_t)(l + 1) * L, L, slo, shi, suffix + (uint64_t)l * L);
        }
        uint64_t head = 0, tail = 0, evals = 0;
        q[tail].assign = 0; q[tail].depth = 0; q[tail].parent = VF | VT | VU; tail++;
        while (head < tail) {
            if (evals >= max_eval) break;
            branch br = q[head++];
            evals++;
            for (int32_t s = 0; s < p->nsched; s++) {         /* the possible values under the partial assignment */
                uint32_t i = p->schedule[s];
                const tr_bool_node *nd = &p->nodes[i];
                uint8_t x;
                if (nd->op == TR_BOOL_CONST) x = nd->a == TR_BOOL_TRUE ? VT : nd->a == TR_BOOL_FALSE ? VF : VU;
                else if (nd->op == TR_BOOL_VAR) {
                    uint32_t l = level_of[nd->a];
                    x = l < br.depth ? (((br.assign >> l) & 1) ? VT : VF) : (VF | VT);
                } else x = set_op(nd->op, sets[nd->a], nd->b >= 0 ? sets[nd->b] : 0);
                sets[i] = x;
            }
            uint8_t root = sets[p->root];
            if (root == VF || root == VT || root == VU) {        /* decided: the branch's whole weight */
                tr_symc_w_zero(pre, L); pre[0] = 1;
                for (uint32_t l = 0; l < br.depth; l++) {
                    int bit = (int)((br.assign >> l) & 1);
                    over |= tr_symc_w_scale(pre, t, L, scaled[4 * (uint64_t)l + (bit ? 0 : 2)], scaled[4 * (uint64_t)l + (bit ? 1 : 3)]);
                }
                over |= tr_symc_w_mul(pre, suffix + (uint64_t)br.depth * L, L, wt);
                int c = root == VT ? 0 : root == VF ? 1 : 2;
                over |= tr_symc_w_add(lo + (uint64_t)c * L, wt, L);
                over |= tr_symc_w_add(hi + (uint64_t)c * L, wt, L);
                continue;
            }
            if (br.depth >= k) { rc = TR_MATH_INVALID; break; }       /* unreachable: a full assignment decides */
            for (uint64_t bit = 0; bit < 2; bit++) {
                branch nb;
                nb.assign = br.assign | (bit << br.depth); nb.depth = br.depth + 1; nb.parent = root;
                nb.reserved[0] = nb.reserved[1] = nb.reserved[2] = 0;
                q[tail++] = nb;
            }
        }
        uint64_t open = 0;
        for (uint64_t i = head; i < tail && !rc; i++) {                /* open branches: certified upper bounds */
            branch br = q[i];
            tr_symc_w_zero(pre, L); pre[0] = 1;
            for (uint32_t l = 0; l < br.depth; l++) {
                int bit = (int)((br.assign >> l) & 1);
                over |= tr_symc_w_scale(pre, t, L, scaled[4 * (uint64_t)l + (bit ? 0 : 2)], scaled[4 * (uint64_t)l + (bit ? 1 : 3)]);
            }
            over |= tr_symc_w_mul(pre, suffix + (uint64_t)br.depth * L, L, wt);
            if (br.parent & VT) over |= tr_symc_w_add(hi, wt, L);
            if (br.parent & VF) over |= tr_symc_w_add(hi + L, wt, L);
            if (br.parent & VU) over |= tr_symc_w_add(hi + 2 * L, wt, L);
            open++;
        }
        if (!rc && over) rc = TR_MATH_OVERFLOW;
        for (int c = 0; c < 3 && !rc; c++) {
            rc = ans_set(a, 2 * c, lo + (uint64_t)c * L, D, L, b, refusal);
            if (!rc) {
                tr_symc_w_copy(t2, hi + (uint64_t)c * L, L);
                rc = ans_set(a, 2 * c + 1, t2, D, L, b, refusal);
            }
        }
        if (!rc) rc = ans_set(a, 6, S, D, L, b, refusal);
        if (!rc) {
            a->info.status = open ? TR_SYMC_INCOMPLETE : TR_SYMC_EXACT;
            a->info.evaluations = evals; a->info.open = open; a->info.weighted = weights != NULL;
            for (int32_t s = 0; s < p->nsched; s++)
                if (p->nodes[p->schedule[s]].op == TR_BOOL_CONST && p->nodes[p->schedule[s]].a == TR_BOOL_UNKNOWN) a->info.three_valued = 1;
        }
    }
    free(q); free(sets); free(scaled); free(suffix); free(w);
    if (rc) { tr_symc_answer_free(a); set_error(error, rc); return NULL; }
    set_error(error, TR_MATH_OK);
    return a;
}

/* ================================================================== factor systems */
struct tr_symc_factors_s {
    uint32_t magic, nvars, nfactors, max_scope;
    uint32_t *domains;
    uint64_t *scope_at, *table_at, *size;
    uint32_t *nscope, *scope;
    uint64_t nscope_total, nentries;
    uint64_t *entry;          /* 2 words per entry: the scaled numerator */
    uint64_t *denom;          /* 2 words per factor: D_f */
    uint64_t *maxent;         /* 2 words per factor: the largest scaled entry */
    uint64_t bytes;
    u128 assignments;
    int wide;
};
static int fac_valid(const tr_symc_factors *f) { return f && f->magic == FAC_MAGIC; }
static void fac_release(tr_symc_factors *f) {
    free(f->domains); free(f->scope_at); free(f->table_at); free(f->size); free(f->nscope); free(f->scope);
    free(f->entry); free(f->denom); free(f->maxent);
    f->magic = 0;
    free(f);
}
void tr_symc_factors_free(tr_symc_factors *f) { if (fac_valid(f)) fac_release(f); }

tr_symc_factors *tr_symc_factors_create(uint32_t nvars, const uint32_t *domains, uint32_t nfactors,
                                        const tr_symc_factor *factors, const uint32_t *scope, const int64_t *num,
                                        const int64_t *den, uint64_t nentries, const tr_math_budget *budget,
                                        tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    if (nvars < 1 || !domains || (nfactors && !factors) || (nentries && (!num || !den))) {
        set_error(error, TR_MATH_INVALID); return NULL;
    }
    if (nvars > TR_SYMC_MAX_FACTOR_VARS) { set_error(error, TR_MATH_UNSUPPORTED); return NULL; }
    for (uint32_t v = 0; v < nvars; v++) {
        if (domains[v] < 1) { set_error(error, TR_MATH_INVALID); return NULL; }
        if (domains[v] > TR_SYMC_MAX_DOMAIN) { set_error(error, TR_MATH_UNSUPPORTED); return NULL; }
    }
    tr_math_budget dflt;
    const tr_math_budget *b = budget_of(budget, &dflt);
    uint64_t total_scope = 0, maxsize = 0;
    uint32_t max_scope = 0;
    for (uint32_t f = 0; f < nfactors; f++) {
        const tr_symc_factor *F = &factors[f];
        if (F->reserved || (F->nscope && !scope)) { set_error(error, TR_MATH_INVALID); return NULL; }
        if (F->nscope > TR_SYMC_MAX_SCOPE) { set_error(error, TR_MATH_UNSUPPORTED); return NULL; }
        uint64_t size = 1;
        for (uint32_t t = 0; t < F->nscope; t++) {
            uint32_t v = scope[F->scope_at + t];
            if (v >= nvars) { set_error(error, TR_MATH_INVALID); return NULL; }
            for (uint32_t u = 0; u < t; u++) if (scope[F->scope_at + u] == v) { set_error(error, TR_MATH_INVALID); return NULL; }
            size = tr_budget_mul(size, domains[v]);
        }
        if (size == TR_BUDGET_UNKNOWN || F->table_at > nentries || size > nentries - F->table_at) {
            set_error(error, size == TR_BUDGET_UNKNOWN ? TR_MATH_BUDGET : TR_MATH_INVALID);
            if (size == TR_BUDGET_UNKNOWN && refusal) { refusal->resource = TR_BUDGET_TENSOR; refusal->refused = 1; refusal->need = size; refusal->limit = b->limit[TR_BUDGET_TENSOR]; }
            return NULL;
        }
        if (size > maxsize) maxsize = size;
        if (F->nscope > max_scope) max_scope = F->nscope;
        total_scope = tr_budget_add(total_scope, F->nscope);
    }
    tr_math_need need;
    need_zero(&need);
    need.need[TR_BUDGET_VARIABLES] = nvars;
    need.need[TR_BUDGET_SUPPORT] = nentries;
    need.need[TR_BUDGET_TENSOR] = maxsize;
    uint64_t bytes = tr_budget_add(sizeof(tr_symc_factors), tr_budget_mul(nvars, 4));
    bytes = tr_budget_add(bytes, tr_budget_mul(nfactors, 8 * 3 + 4 + 48));
    bytes = tr_budget_add(bytes, tr_budget_mul(total_scope, 4));
    bytes = tr_budget_add(bytes, tr_budget_mul(nentries, 16));
    need.need[TR_BUDGET_BYTES] = bytes;
    int32_t rc = check_need(b, &need, refusal);
    if (rc) { set_error(error, rc); return NULL; }
    tr_symc_factors *fs = (tr_symc_factors *)zalloc(sizeof *fs);
    if (!fs) { set_error(error, TR_MATH_MEMORY); return NULL; }
    uint64_t nf1 = nfactors ? nfactors : 1;
    fs->domains = (uint32_t *)zalloc(4 * (uint64_t)nvars);
    fs->scope_at = (uint64_t *)zalloc(8 * nf1); fs->table_at = (uint64_t *)zalloc(8 * nf1); fs->size = (uint64_t *)zalloc(8 * nf1);
    fs->nscope = (uint32_t *)zalloc(4 * nf1); fs->scope = (uint32_t *)zalloc(4 * (total_scope ? total_scope : 1));
    fs->entry = (uint64_t *)zalloc(tr_budget_mul(nentries ? nentries : 1, 16));
    fs->denom = (uint64_t *)zalloc(16 * nf1); fs->maxent = (uint64_t *)zalloc(16 * nf1);
    if (!fs->domains || !fs->scope_at || !fs->table_at || !fs->size || !fs->nscope || !fs->scope || !fs->entry ||
        !fs->denom || !fs->maxent) { fac_release(fs); set_error(error, TR_MATH_MEMORY); return NULL; }
    memcpy(fs->domains, domains, (size_t)nvars * 4);
    uint64_t at = 0, newent = 0;
    for (uint32_t f = 0; f < nfactors && !rc; f++) {
        const tr_symc_factor *F = &factors[f];
        uint64_t size = 1;
        for (uint32_t t = 0; t < F->nscope; t++) { fs->scope[at + t] = scope[F->scope_at + t]; size *= domains[scope[F->scope_at + t]]; }
        fs->scope_at[f] = at; fs->nscope[f] = F->nscope; fs->size[f] = size; fs->table_at[f] = newent;
        at += F->nscope;
        u128 D = 1;
        for (uint64_t e = 0; e < size; e++) {                 /* D_f = lcm of the denominators, below 2^126 */
            int64_t n = num[F->table_at + e], d = den[F->table_at + e];
            if (n < 0 || d < 1) { rc = TR_MATH_INVALID; break; }
            u128 next = D / gcd128(D, (u128)(uint64_t)d);
            if (next > (((u128)1 << 126) / (uint64_t)d)) { rc = TR_MATH_UNSUPPORTED; break; }
            D = next * (uint64_t)d;
        }
        u128 mx = 0;
        for (uint64_t e = 0; e < size && !rc; e++) {
            int64_t n = num[F->table_at + e], d = den[F->table_at + e];
            u128 mult = D / (uint64_t)d;
            if (n && mult > (((u128)1 << 126) / (uint64_t)n)) { rc = TR_MATH_UNSUPPORTED; break; }
            u128 c = (u128)(uint64_t)n * mult;
            put128(fs->entry + 2 * (newent + e), c);
            if (c > mx) mx = c;
        }
        put128(fs->denom + 2 * (uint64_t)f, D);
        put128(fs->maxent + 2 * (uint64_t)f, mx);
        newent += size;
    }
    if (rc) { fac_release(fs); set_error(error, rc); return NULL; }
    fs->magic = FAC_MAGIC; fs->nvars = nvars; fs->nfactors = nfactors; fs->max_scope = max_scope;
    fs->nscope_total = total_scope; fs->nentries = newent; fs->bytes = bytes;
    u128 asg = 1;
    for (uint32_t v = 0; v < nvars && !fs->wide; v++) {
        if (asg > (~(u128)0) / domains[v]) fs->wide = 1;
        else asg *= domains[v];
    }
    fs->assignments = fs->wide ? 0 : asg;
    set_error(error, TR_MATH_OK);
    return fs;
}
int32_t tr_symc_factors_get_info(const tr_symc_factors *f, tr_symc_factors_info *out) {
    if (!fac_valid(f) || !out) return TR_MATH_INVALID;
    memset(out, 0, sizeof *out);
    out->entries = f->nentries; out->bytes = f->bytes;
    out->assignments_lo = (uint64_t)f->assignments; out->assignments_hi = (uint64_t)(f->assignments >> 64);
    out->nvars = f->nvars; out->nfactors = f->nfactors; out->max_scope = f->max_scope; out->assignments_wide = (uint32_t)f->wide;
    return TR_MATH_OK;
}

/* the elimination order */
static int32_t fac_order(const tr_symc_factors *fs, int32_t kind, const uint32_t *given, uint32_t *out,
                         const tr_math_budget *b, tr_math_refusal *r) {
    const uint32_t n = fs->nvars;
    if (kind == TR_SYMC_ORDER_GIVEN || kind == TR_SYMC_ORDER_NATURAL) {
        if (kind == TR_SYMC_ORDER_GIVEN && !given) return TR_MATH_INVALID;
        uint8_t *seen = (uint8_t *)zalloc(n);
        if (!seen) return TR_MATH_MEMORY;
        for (uint32_t i = 0; i < n; i++) {
            uint32_t v = kind == TR_SYMC_ORDER_GIVEN ? given[i] : i;
            if (v >= n || seen[v]) { free(seen); return TR_MATH_INVALID; }
            seen[v] = 1; out[i] = v;
        }
        free(seen);
        return TR_MATH_OK;
    }
    if (kind != TR_SYMC_ORDER_MIN_FILL && kind != TR_SYMC_ORDER_MIN_DEGREE) return TR_MATH_INVALID;
    if (n > TR_SYMC_MAX_HEURISTIC) return TR_MATH_UNSUPPORTED;
    const uint64_t words = ((uint64_t)n + 63) / 64;
    int32_t rc = check_one(b, TR_BUDGET_BYTES, tr_budget_add(tr_budget_mul(tr_budget_mul(n, words), 8), 2 * (uint64_t)n), r);
    if (rc) return rc;
    uint64_t *adj = (uint64_t *)zalloc(tr_budget_mul(tr_budget_mul(n, words), 8));
    uint8_t *done = (uint8_t *)zalloc(n);
    if (!adj || !done) { free(adj); free(done); return TR_MATH_MEMORY; }
    for (uint32_t f = 0; f < fs->nfactors; f++)
        for (uint32_t s = 0; s < fs->nscope[f]; s++)
            for (uint32_t t = 0; t < fs->nscope[f]; t++) {
                uint32_t u = fs->scope[fs->scope_at[f] + s], v = fs->scope[fs->scope_at[f] + t];
                if (u != v) adj[(uint64_t)u * words + v / 64] |= (uint64_t)1 << (v % 64);
            }
    for (uint32_t step = 0; step < n; step++) {
        uint32_t best = NONE32;
        uint64_t bestc = UINT64_MAX;
        for (uint32_t v = 0; v < n; v++) {
            if (done[v]) continue;
            uint64_t c = 0;
            const uint64_t *av = adj + (uint64_t)v * words;
            if (kind == TR_SYMC_ORDER_MIN_DEGREE) {
                for (uint64_t w = 0; w < words; w++) c += (uint64_t)__builtin_popcountll(av[w]);
            } else {
                for (uint32_t u = 0; u < n && c < bestc; u++) {               /* fill edges among v's neighbours */
                    if (!((av[u / 64] >> (u % 64)) & 1)) continue;
                    const uint64_t *au = adj + (uint64_t)u * words;
                    for (uint64_t w = 0; w < words; w++) c += (uint64_t)__builtin_popcountll(av[w] & ~au[w]);
                    c -= 1;                                                  /* u itself */
                }
            }
            if (c < bestc) { bestc = c; best = v; }
        }
        out[step] = best;
        done[best] = 1;
        const uint64_t *ab = adj + (uint64_t)best * words;
        for (uint32_t u = 0; u < n; u++) {                                    /* connect the neighbours; drop best */
            if (!((ab[u / 64] >> (u % 64)) & 1)) continue;
            uint64_t *au = adj + (uint64_t)u * words;
            for (uint64_t w = 0; w < words; w++) au[w] |= ab[w];
            au[u / 64] &= ~((uint64_t)1 << (u % 64));
            au[best / 64] &= ~((uint64_t)1 << (best % 64));
        }
    }
    free(adj); free(done);
    return TR_MATH_OK;
}

/* the simulated elimination: buckets, message scopes and sizes */
typedef struct {
    uint32_t ntables;                 /* factors + messages (one per step) */
    uint32_t *tscope_n; uint64_t *tscope_at; uint32_t *tscope;   /* scopes of every table, sorted by variable */
    uint64_t *tsize;
    uint32_t *bucket_n; uint64_t *bucket_at; uint32_t *bucket;   /* the tables of each step */
    uint64_t *joint;                  /* the joint table of each step */
    uint64_t width, products, msg_entries;
    int32_t *const_table;             /* per step: 1 when its message is a constant */
} elim_plan;
static void plan_free(elim_plan *p) {
    free(p->tscope_n); free(p->tscope_at); free(p->tscope); free(p->tsize); free(p->bucket_n); free(p->bucket_at);
    free(p->bucket); free(p->joint); free(p->const_table);
    memset(p, 0, sizeof *p);
}
static int u32cmp(const void *a, const void *b) { uint32_t x = *(const uint32_t *)a, y = *(const uint32_t *)b; return x < y ? -1 : x > y; }
static int32_t plan_elim(const tr_symc_factors *fs, const uint32_t *order, elim_plan *p) {
    memset(p, 0, sizeof *p);
    const uint32_t n = fs->nvars, nf = fs->nfactors, nt = nf + n;
    uint32_t *pos = (uint32_t *)zalloc(4 * (uint64_t)n), *tb = (uint32_t *)zalloc(4 * (uint64_t)nt);
    uint8_t *inu = (uint8_t *)zalloc(n);
    uint32_t *tmp = (uint32_t *)zalloc(4 * (uint64_t)n + 4);
    p->tscope_n = (uint32_t *)zalloc(4 * (uint64_t)nt); p->tscope_at = (uint64_t *)zalloc(8 * (uint64_t)nt);
    p->tsize = (uint64_t *)zalloc(8 * (uint64_t)nt);
    p->bucket_n = (uint32_t *)zalloc(4 * (uint64_t)n); p->bucket_at = (uint64_t *)zalloc(8 * (uint64_t)n + 8);
    p->bucket = (uint32_t *)zalloc(4 * (uint64_t)nt); p->joint = (uint64_t *)zalloc(8 * (uint64_t)n);
    p->const_table = (int32_t *)zalloc(4 * (uint64_t)n);
    uint64_t scap = fs->nscope_total + n + 1;                 /* grown as messages are made */
    p->tscope = (uint32_t *)zalloc(4 * scap);
    int32_t rc = TR_MATH_OK;
    if (!pos || !tb || !inu || !tmp || !p->tscope_n || !p->tscope_at || !p->tsize || !p->bucket_n || !p->bucket_at ||
        !p->bucket || !p->joint || !p->const_table || !p->tscope) rc = TR_MATH_MEMORY;
    for (uint32_t i = 0; i < n && !rc; i++) pos[order[i]] = i;
    uint64_t sat = 0;
    for (uint32_t f = 0; f < nf && !rc; f++) {                 /* the factors' scopes, sorted */
        p->tscope_at[f] = sat; p->tscope_n[f] = fs->nscope[f];
        memcpy(p->tscope + sat, fs->scope + fs->scope_at[f], (size_t)fs->nscope[f] * 4);
        qsort(p->tscope + sat, fs->nscope[f], 4, u32cmp);
        sat += fs->nscope[f];
        p->tsize[f] = fs->size[f];
    }
    /* the bucket of a table: its first-eliminated variable (NONE32: a constant); messages get theirs when made */
    for (uint32_t t = 0; t < nf && !rc; t++) {
        uint32_t bst = NONE32;
        for (uint32_t s = 0; s < p->tscope_n[t]; s++) { uint32_t q = pos[p->tscope[p->tscope_at[t] + s]]; if (q < bst) bst = q; }
        tb[t] = bst;
    }
    /* step by step: the bucket's tables are those with tb == step (messages of earlier steps included) */
    uint64_t bat = 0;
    for (uint32_t step = 0; step < n && !rc; step++) {
        uint32_t x = order[step], cnt = 0, nu = 0;
        p->bucket_at[step] = bat;
        for (uint32_t t = 0; t < nf + step; t++) {
            if (tb[t] != step) continue;
            p->bucket[bat + cnt++] = t;
            for (uint32_t s = 0; s < p->tscope_n[t]; s++) {
                uint32_t v = p->tscope[p->tscope_at[t] + s];
                if (v != x && !inu[v]) { inu[v] = 1; tmp[nu++] = v; }
            }
        }
        p->bucket_n[step] = cnt;
        bat += cnt;
        qsort(tmp, nu, 4, u32cmp);
        uint64_t msize = 1;
        uint32_t mb = NONE32;
        for (uint32_t s = 0; s < nu; s++) {
            inu[tmp[s]] = 0;
            msize = tr_budget_mul(msize, fs->domains[tmp[s]]);
            if (pos[tmp[s]] < mb) mb = pos[tmp[s]];
        }
        uint64_t joint = tr_budget_mul(msize, fs->domains[x]);
        p->joint[step] = joint;
        if (joint > p->width) p->width = joint;
        p->products = tr_budget_add(p->products, tr_budget_mul(joint, cnt ? cnt : 1));
        uint32_t m = nf + step;
        if (sat + nu > scap) {
            uint64_t ncap = scap * 2 > sat + nu ? scap * 2 : sat + nu;
            uint32_t *g = (uint32_t *)realloc(p->tscope, (size_t)ncap * 4);
            if (!g) { rc = TR_MATH_MEMORY; break; }
            p->tscope = g; scap = ncap;
        }
        p->tscope_at[m] = sat; p->tscope_n[m] = nu;
        if (nu) memcpy(p->tscope + sat, tmp, (size_t)nu * 4);
        sat += nu;
        p->tsize[m] = msize;
        tb[m] = mb;
        p->const_table[step] = nu == 0;
        p->msg_entries = tr_budget_add(p->msg_entries, msize);
    }
    p->ntables = nt;
    free(pos); free(tb); free(inu); free(tmp);
    if (rc) plan_free(p);
    return rc;
}
/* limbs of every exact value of an elimination: prod of the factors' largest entries times prod d_i, and prod D_f */
static uint32_t fac_limbs(const tr_symc_factors *fs, int32_t mode) {
    uint64_t nbits = 1, dbits = 1;
    for (uint32_t f = 0; f < fs->nfactors; f++) {
        nbits += mode == TR_SYMC_INDICATOR ? 1 : bitlen128(get128(fs->maxent + 2 * (uint64_t)f));
        if (mode != TR_SYMC_INDICATOR) dbits += bitlen128(get128(fs->denom + 2 * (uint64_t)f));
    }
    for (uint32_t v = 0; v < fs->nvars; v++) nbits += bitlen128(fs->domains[v]);
    uint64_t bits = nbits > dbits ? nbits : dbits, L = bits / 64 + 1;
    return L > TR_SYMC_MAX_LIMBS ? 0 : (uint32_t)L;
}
static void fac_needs(const tr_symc_factors *fs, const elim_plan *p, uint32_t L, uint64_t witnesses, tr_math_need *need) {
    need_zero(need);
    need->need[TR_BUDGET_VARIABLES] = fs->nvars;
    need->need[TR_BUDGET_TENSOR] = p->width;
    need->need[TR_BUDGET_ITERATIONS] = p->products;
    need->need[TR_BUDGET_OUTPUT] = tr_budget_add(2, witnesses);
    uint64_t bytes = tr_budget_mul(tr_budget_mul(tr_budget_add(p->msg_entries, fs->nentries), L), 8);
    bytes = tr_budget_add(bytes, tr_budget_mul(tr_budget_add(fs->nscope_total, fs->nvars), 8));
    bytes = tr_budget_add(bytes, tr_budget_mul(tr_budget_add(fs->nfactors, fs->nvars), 64));
    bytes = tr_budget_add(bytes, tr_budget_mul(tr_budget_mul(witnesses, fs->nvars), 8));
    need->need[TR_BUDGET_BYTES] = tr_budget_add(bytes, tr_budget_mul(16 * (uint64_t)L, 8));
}
int32_t tr_symc_factors_order(const tr_symc_factors *fs, int32_t kind, const uint32_t *given, uint32_t *out,
                              uint64_t *width, const tr_math_budget *budget, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!fac_valid(fs) || !out) return TR_MATH_INVALID;
    tr_math_budget dflt;
    const tr_math_budget *b = budget_of(budget, &dflt);
    int32_t rc = fac_order(fs, kind, given, out, b, refusal);
    if (rc || !width) return rc;
    elim_plan p;
    rc = plan_elim(fs, out, &p);
    if (!rc) *width = p.width;
    plan_free(&p);
    return rc;
}
int32_t tr_symc_factors_need(const tr_symc_factors *fs, int32_t kind, const uint32_t *given, int32_t mode,
                             uint64_t witnesses, tr_math_need *need) {
    if (!fac_valid(fs) || !need || (mode != TR_SYMC_WEIGHTED && mode != TR_SYMC_INDICATOR)) return TR_MATH_INVALID;
    uint32_t *order = (uint32_t *)zalloc(4 * (uint64_t)fs->nvars);
    if (!order) return TR_MATH_MEMORY;
    tr_math_budget unl;
    for (int i = 0; i < TR_BUDGET_RESOURCES; i++) unl.limit[i] = TR_BUDGET_UNKNOWN - 1;
    int32_t rc = fac_order(fs, kind, given, order, &unl, NULL);
    elim_plan p;
    if (!rc) rc = plan_elim(fs, order, &p);
    if (!rc) {
        uint32_t L = fac_limbs(fs, mode);
        if (!L) rc = TR_MATH_UNSUPPORTED;
        else fac_needs(fs, &p, L, witnesses, need);
        plan_free(&p);
    }
    free(order);
    return rc;
}
/* the C-order index of table t (scope sorted) at the full assignment val[] */
static uint64_t table_index(const tr_symc_factors *fs, const elim_plan *p, uint32_t t, const uint32_t *val,
                            const uint32_t *fscope, uint32_t fn) {
    uint64_t idx = 0;
    if (t < fs->nfactors) {                                    /* a factor: its own scope order */
        for (uint32_t s = 0; s < fn; s++) idx = idx * fs->domains[fscope[s]] + val[fscope[s]];
        return idx;
    }
    for (uint32_t s = 0; s < p->tscope_n[t]; s++) {
        uint32_t v = p->tscope[p->tscope_at[t] + s];
        idx = idx * fs->domains[v] + val[v];
    }
    return idx;
}
/* the L-limb entry of table t at index e (factors: their scaled u128 entries; messages: stored) */
static const uint64_t *table_entry(const tr_symc_factors *fs, uint32_t t, uint64_t e, uint32_t L, uint64_t *const *msg,
                                   uint64_t *tmp, int32_t mode) {
    if (t < fs->nfactors) {
        u128 c = get128(fs->entry + 2 * (fs->table_at[t] + e));
        if (mode == TR_SYMC_INDICATOR) c = c != 0;
        tr_symc_w_set128(tmp, L, (uint64_t)c, (uint64_t)(c >> 64));
        return tmp;
    }
    return msg[t - fs->nfactors] + e * L;
}
static int entry_nonzero(const tr_symc_factors *fs, uint32_t t, uint64_t e, uint32_t L, uint64_t *const *msg) {
    if (t < fs->nfactors) return get128(fs->entry + 2 * (fs->table_at[t] + e)) != 0;
    return !tr_symc_w_is_zero(msg[t - fs->nfactors] + e * L, L);
}

tr_symc_answer *tr_symc_factors_count(const tr_symc_factors *fs, int32_t order_kind, const uint32_t *given, int32_t mode,
                                      uint64_t witnesses, const tr_math_budget *budget, tr_math_refusal *refusal,
                                      int32_t *error) {
    tr_budget_none(refusal);
    if (!fac_valid(fs) || (mode != TR_SYMC_WEIGHTED && mode != TR_SYMC_INDICATOR)) { set_error(error, TR_MATH_INVALID); return NULL; }
    tr_math_budget dflt;
    const tr_math_budget *b = budget_of(budget, &dflt);
    const uint32_t n = fs->nvars, nf = fs->nfactors;
    uint32_t *order = (uint32_t *)zalloc(4 * (uint64_t)n);
    if (!order) { set_error(error, TR_MATH_MEMORY); return NULL; }
    int32_t rc = fac_order(fs, order_kind, given, order, b, refusal);
    elim_plan p;
    memset(&p, 0, sizeof p);
    if (!rc) rc = plan_elim(fs, order, &p);
    uint32_t L = rc ? 0 : fac_limbs(fs, mode);
    if (!rc && !L) rc = TR_MATH_UNSUPPORTED;
    if (!rc) {
        tr_math_need need;
        fac_needs(fs, &p, L, witnesses, &need);
        rc = check_need(b, &need, refusal);
    }
    uint64_t **msg = NULL;
    uint32_t *val = NULL, *uvars = NULL;
    uint64_t *w = NULL, *wit = NULL;
    tr_symc_answer *a = NULL;
    if (!rc) {
        msg = (uint64_t **)zalloc(sizeof(uint64_t *) * (uint64_t)n);
        val = (uint32_t *)zalloc(4 * (uint64_t)n);
        uvars = (uint32_t *)zalloc(4 * (uint64_t)n + 4);
        w = (uint64_t *)zalloc(tr_budget_mul(8 * (uint64_t)L, 8));
        wit = (uint64_t *)zalloc(tr_budget_mul(tr_budget_mul(witnesses ? witnesses : 1, n), 8));
        a = ans_new(TR_SYMC_M_ELIM, 2, n, order, n);
        if (!msg || !val || !uvars || !w || !wit || !a) rc = TR_MATH_MEMORY;
    }
    uint64_t *Z = w, *prod = w + L, *sum = w + 2 * L, *t1 = w + 3 * L, *e1 = w + 4 * L, *D = w + 5 * L, *tmp = w + 6 * L;
    int over = 0;
    if (!rc) {
        tr_symc_w_zero(Z, L); Z[0] = 1;
        for (uint32_t f = 0; f < nf; f++) {                    /* constant factors (empty scope) multiply Z */
            if (p.tscope_n[f]) continue;
            const uint64_t *e = table_entry(fs, f, 0, L, msg, e1, mode);
            over |= tr_symc_w_mul(Z, e, L, t1);
            tr_symc_w_copy(Z, t1, L);
        }
    }
    for (uint32_t step = 0; step < n && !rc; step++) {
        const uint32_t x = order[step], m = nf + step, nu = p.tscope_n[m];
        const uint32_t *us = p.tscope + p.tscope_at[m];
        const uint64_t msize = p.tsize[m];
        uint64_t *out = (uint64_t *)zalloc(tr_budget_mul(tr_budget_mul(msize, L), 8));
        if (!out) { rc = TR_MATH_MEMORY; break; }
        msg[step] = out;
        for (uint32_t s = 0; s < nu; s++) val[us[s]] = 0;
        for (uint64_t e = 0; e < msize; e++) {                 /* the message entry at the assignment val[us] */
            tr_symc_w_zero(sum, L);
            for (uint32_t v = 0; v < fs->domains[x]; v++) {
                val[x] = v;
                tr_symc_w_zero(prod, L); prod[0] = 1;
                for (uint32_t j = 0; j < p.bucket_n[step]; j++) {
                    uint32_t t = p.bucket[p.bucket_at[step] + j];
                    uint64_t idx = table_index(fs, &p, t, val, t < nf ? fs->scope + fs->scope_at[t] : NULL,
                                               t < nf ? fs->nscope[t] : 0);
                    const uint64_t *ent = table_entry(fs, t, idx, L, msg, e1, mode);
                    over |= tr_symc_w_mul(prod, ent, L, t1);
                    tr_symc_w_copy(prod, t1, L);
                }
                over |= tr_symc_w_add(sum, prod, L);
            }
            tr_symc_w_copy(out + e * L, sum, L);
            for (uint32_t s = nu; s-- > 0;) {                   /* the next assignment (the last variable fastest) */
                if (++val[us[s]] < fs->domains[us[s]]) break;
                val[us[s]] = 0;
            }
        }
        if (!nu) { over |= tr_symc_w_mul(Z, out, L, t1); tr_symc_w_copy(Z, t1, L); }
    }
    if (!rc && over) rc = TR_MATH_OVERFLOW;
    uint64_t got = 0;
    if (!rc && witnesses && !tr_symc_w_is_zero(Z, L)) {       /* decode in reverse elimination order */
        uint32_t *nextv = uvars;                                /* per step: the next value to try */
        int64_t s = (int64_t)n - 1;
        nextv[n - 1] = 0;
        while (s < (int64_t)n && got < witnesses) {
            if (s < 0) {                                         /* a full model */
                for (uint32_t v = 0; v < n; v++) wit[got * n + v] = val[v];
                got++;
                s = 0;
                continue;
            }
            uint32_t step = (uint32_t)s, x = order[step], v = nextv[step];
            int found = 0;
            for (; v < fs->domains[x]; v++) {
                val[x] = v;
                int ok = 1;
                for (uint32_t j = 0; j < p.bucket_n[step] && ok; j++) {
                    uint32_t t = p.bucket[p.bucket_at[step] + j];
                    uint64_t idx = table_index(fs, &p, t, val, t < nf ? fs->scope + fs->scope_at[t] : NULL,
                                               t < nf ? fs->nscope[t] : 0);
                    ok = entry_nonzero(fs, t, idx, L, msg);
                }
                if (ok) { found = 1; break; }
            }
            if (found) { nextv[step] = v + 1; s--; if (s >= 0) nextv[s] = 0; }
            else s++;                                            /* exhausted: back to the later variable */
        }
    }
    if (!rc) {
        tr_symc_w_zero(D, L); D[0] = 1;
        if (mode == TR_SYMC_WEIGHTED)
            for (uint32_t f = 0; f < nf; f++) {
                u128 d = get128(fs->denom + 2 * (uint64_t)f);
                over |= tr_symc_w_scale(D, tmp, L, (uint64_t)d, (uint64_t)(d >> 64));
            }
        uint64_t *asg = sum, *one = prod;
        tr_symc_w_zero(asg, L); asg[0] = 1;
        for (uint32_t v = 0; v < n; v++) over |= tr_symc_w_scale(asg, tmp, L, fs->domains[v], 0);
        tr_symc_w_zero(one, L); one[0] = 1;
        if (over) rc = TR_MATH_OVERFLOW;
        if (!rc) rc = ans_set(a, 0, Z, D, L, b, refusal);
        if (!rc) rc = ans_set(a, 1, asg, one, L, b, refusal);
    }
    if (!rc) {
        a->info.weighted = mode == TR_SYMC_WEIGHTED; a->info.width = p.width; a->info.evaluations = p.products;
        a->info.witnesses = got; a->info.witness_width = n;
        if (got) {
            a->witnesses = (uint64_t *)zalloc(tr_budget_mul(tr_budget_mul(got, n), 8));
            if (!a->witnesses) rc = TR_MATH_MEMORY;
            else memcpy(a->witnesses, wit, (size_t)(got * n) * 8);
        }
    }
    if (msg) for (uint32_t s = 0; s < n; s++) free(msg[s]);
    free(msg); free(val); free(uvars); free(w); free(wit); free(order);
    plan_free(&p);
    if (rc) { tr_symc_answer_free(a); set_error(error, rc); return NULL; }
    set_error(error, TR_MATH_OK);
    return a;
}

tr_symc_answer *tr_symc_factors_brute(const tr_symc_factors *fs, int32_t mode, uint64_t witnesses,
                                      const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    if (!fac_valid(fs) || (mode != TR_SYMC_WEIGHTED && mode != TR_SYMC_INDICATOR)) { set_error(error, TR_MATH_INVALID); return NULL; }
    tr_math_budget dflt;
    const tr_math_budget *b = budget_of(budget, &dflt);
    const uint32_t n = fs->nvars, nf = fs->nfactors;
    uint32_t L = fac_limbs(fs, mode);
    if (!L) { set_error(error, TR_MATH_UNSUPPORTED); return NULL; }
    tr_math_need need;
    need_zero(&need);
    need.need[TR_BUDGET_VARIABLES] = n;
    need.need[TR_BUDGET_SUPPORT] = fs->wide || fs->assignments > (u128)UINT64_MAX ? TR_BUDGET_UNKNOWN : (uint64_t)fs->assignments;
    need.need[TR_BUDGET_OUTPUT] = tr_budget_add(2, witnesses);
    need.need[TR_BUDGET_BYTES] = tr_budget_add(tr_budget_mul(8 * (uint64_t)L, 8),
                                               tr_budget_add(tr_budget_mul(4, n), tr_budget_mul(tr_budget_mul(witnesses, n), 8)));
    int32_t rc = check_need(b, &need, refusal);
    if (rc) { set_error(error, rc); return NULL; }
    uint32_t *val = (uint32_t *)zalloc(4 * (uint64_t)n);
    uint64_t *w = (uint64_t *)zalloc(tr_budget_mul(8 * (uint64_t)L, 8));
    uint64_t *wit = (uint64_t *)zalloc(tr_budget_mul(tr_budget_mul(witnesses ? witnesses : 1, n), 8));
    tr_symc_answer *a = ans_new(TR_SYMC_M_FBRUTE, 2, n, NULL, 0);
    if (!val || !w || !wit || !a) rc = TR_MATH_MEMORY;
    uint64_t *Z = w, *prod = w + L, *t1 = w + 2 * L, *e1 = w + 3 * L, *D = w + 4 * L, *tmp = w + 5 * L, *asg = w + 6 * L,
             *one = w + 7 * L;
    int over = 0;
    uint64_t got = 0, total = (uint64_t)fs->assignments;
    if (!rc) {
        tr_symc_w_zero(Z, L);
        for (uint64_t i = 0; i < total; i++) {
            tr_symc_w_zero(prod, L); prod[0] = 1;
            int nz = 1;
            for (uint32_t f = 0; f < nf; f++) {
                uint64_t idx = 0;
                for (uint32_t s = 0; s < fs->nscope[f]; s++) {
                    uint32_t v = fs->scope[fs->scope_at[f] + s];
                    idx = idx * fs->domains[v] + val[v];
                }
                u128 c = get128(fs->entry + 2 * (fs->table_at[f] + idx));
                if (mode == TR_SYMC_INDICATOR) c = c != 0;
                if (!c) { nz = 0; break; }
                tr_symc_w_set128(e1, L, (uint64_t)c, (uint64_t)(c >> 64));
                over |= tr_symc_w_mul(prod, e1, L, t1);
                tr_symc_w_copy(prod, t1, L);
            }
            if (nz) {
                over |= tr_symc_w_add(Z, prod, L);
                if (got < witnesses) { for (uint32_t v = 0; v < n; v++) wit[got * n + v] = val[v]; got++; }
            }
            for (uint32_t v = n; v-- > 0;) {                    /* variable 0 most significant */
                if (++val[v] < fs->domains[v]) break;
                val[v] = 0;
            }
        }
        tr_symc_w_zero(D, L); D[0] = 1;
        if (mode == TR_SYMC_WEIGHTED)
            for (uint32_t f = 0; f < nf; f++) {
                u128 d = get128(fs->denom + 2 * (uint64_t)f);
                over |= tr_symc_w_scale(D, tmp, L, (uint64_t)d, (uint64_t)(d >> 64));
            }
        tr_symc_w_zero(asg, L); asg[0] = 1;
        for (uint32_t v = 0; v < n; v++) over |= tr_symc_w_scale(asg, tmp, L, fs->domains[v], 0);
        tr_symc_w_zero(one, L); one[0] = 1;
        if (over) rc = TR_MATH_OVERFLOW;
        if (!rc) rc = ans_set(a, 0, Z, D, L, b, refusal);
        if (!rc) rc = ans_set(a, 1, asg, one, L, b, refusal);
    }
    if (!rc) {
        a->info.weighted = mode == TR_SYMC_WEIGHTED; a->info.evaluations = total;
        a->info.witnesses = got; a->info.witness_width = n;
        if (got) {
            a->witnesses = (uint64_t *)zalloc(tr_budget_mul(tr_budget_mul(got, n), 8));
            if (!a->witnesses) rc = TR_MATH_MEMORY;
            else memcpy(a->witnesses, wit, (size_t)(got * n) * 8);
        }
    }
    free(val); free(w); free(wit);
    if (rc) { tr_symc_answer_free(a); set_error(error, rc); return NULL; }
    set_error(error, TR_MATH_OK);
    return a;
}
