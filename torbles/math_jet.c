/* JET-001 / JET-002: circuits, jet plans, aggregates over scopes and jet values in a carrier; see math_jet.h for every
 * contract. SPDX-License-Identifier: Apache-2.0 */
#include "math_jet.h"
#include "mathematics.h"
#include "math_domain.h"
#include "math_handles.h"
#include <stdlib.h>
#include <string.h>

TR_MATH_ABI(tr_jet_node, 1);             /* the ctypes boundary of JET-001 / JET-002 (math_handles.h) */
TR_MATH_ABI(tr_jet_circuit_info, 1);
TR_MATH_ABI(tr_jet_step, 1);
TR_MATH_ABI(tr_jet_view, 1);
TR_MATH_ABI(tr_jet_plan_info, 1);
TR_MATH_ABI(tr_jet_states, 1);
TR_MATH_ABI(tr_jet_agg_info, 1);
TR_MATH_ABI(tr_jet_values_info, 1);

_Static_assert(sizeof(tr_jet_node) == 32, "tr_jet_node ABI");
_Static_assert(sizeof(tr_jet_circuit_info) == 48, "tr_jet_circuit_info ABI");
_Static_assert(sizeof(tr_jet_step) == 72, "tr_jet_step ABI");
_Static_assert(sizeof(tr_jet_view) == 144, "tr_jet_view ABI");
_Static_assert(sizeof(tr_jet_plan_info) == 144, "tr_jet_plan_info ABI");
_Static_assert(sizeof(tr_jet_states) == 40, "tr_jet_states ABI");
_Static_assert(sizeof(tr_jet_agg_info) == 104, "tr_jet_agg_info ABI");
_Static_assert(sizeof(tr_jet_values_info) == 56, "tr_jet_values_info ABI");
_Static_assert(TR_JET_MAX_SUPPORT <= 65535, "pair indices are 16 bits");

#define CIRCUIT_MAGIC UINT32_C(0x3143544a)   /* "JTC1" */
#define PLAN_MAGIC UINT32_C(0x3150544a)
#define BUILDER_MAGIC UINT32_C(0x3142544a)
#define AGG_MAGIC UINT32_C(0x3141544a)
#define VALUES_MAGIC UINT32_C(0x3156544a)
#define REACH 1                              /* circuit node flags: reached through every edge (validity) */
#define EVAL 2                               /* reached through evaluated edges (not the operand of a POW 0) */

typedef __int128 i128;
typedef unsigned __int128 u128;

static void set_error(int32_t *error, int32_t code) { if (error) *error = code; }
static void *zalloc(uint64_t bytes) {
    if (bytes > (uint64_t)SIZE_MAX) return NULL;
    return calloc(1, bytes ? (size_t)bytes : 1);
}
static const tr_math_budget *budget_of(const tr_math_budget *b, tr_math_budget *store) {
    if (b) return b;
    tr_budget_default(store);
    return store;
}
static void need_zero(tr_math_need *n) { for (int i = 0; i < TR_BUDGET_RESOURCES; i++) n->need[i] = 0; }
/* lcm of positive int64 values, checked */
static int lcm64(int64_t a, int64_t b, int64_t *out) {
    uint64_t g = tr_gcd_u64((uint64_t)a, (uint64_t)b);
    return tr_mul_i64(a / (int64_t)g, b, out);
}
/* 128-bit digest of int64 words (two FNV-1a streams; a pre-filter only) */
static void digest_words(const int64_t *w, uint64_t n, uint64_t out[2]) {
    uint64_t h0 = UINT64_C(14695981039346656037), h1 = UINT64_C(0x6c62272e07bb0142);
    for (uint64_t i = 0; i < n; i++) {
        uint64_t x = (uint64_t)w[i];
        for (int k = 0; k < 8; k++) {
            uint8_t byte = (uint8_t)(x >> (8 * k));
            h0 = (h0 ^ byte) * UINT64_C(1099511628211);
            h1 = (h1 ^ byte) * UINT64_C(0x100000001b3) + UINT64_C(0x9e3779b97f4a7c15);
        }
    }
    out[0] = h0; out[1] = h1;
}

/* ================================================================== circuits */
struct tr_jet_circuit_s {
    uint32_t magic;
    int32_t nnodes, nparams, ncols, live, evaluated, degree, param_degree;
    uint32_t columns;
    uint64_t params, bytes;
    tr_jet_node *nodes;
    uint8_t *flags;             /* REACH | EVAL per node */
};
static int circuit_valid(const tr_jet_circuit *c) { return c && c->magic == CIRCUIT_MAGIC; }

uint64_t tr_jet_circuit_bytes(int32_t nnodes) {
    uint64_t n = nnodes > 0 ? (uint64_t)nnodes : 0;
    /* the object, the canonical nodes, one flag byte per node, and the degree scratch (two int32 per node) */
    return tr_budget_add(sizeof(tr_jet_circuit), tr_budget_mul(n, sizeof(tr_jet_node) + 1 + 8));
}

static int unary(int32_t op) { return op == TR_JET_POW || op == TR_JET_SCALE || op == TR_JET_NEG; }
static int binary(int32_t op) { return op == TR_JET_ADD || op == TR_JET_SUB || op == TR_JET_MUL; }
static int leaf(int32_t op) { return op == TR_JET_CONST || op == TR_JET_PARAM || op == TR_JET_DATA; }

/* validates node i; canonical copy into out (may be NULL). INVALID / OVERFLOW / UNSUPPORTED. */
static int32_t check_node(const tr_jet_node *x, int32_t i, int32_t nparams, int32_t ncols, tr_jet_node *out) {
    tr_jet_node c;
    memset(&c, 0, sizeof c);
    c.op = x->op; c.a = -1; c.b = -1; c.value.numerator = 0; c.value.denominator = 1;
    if (leaf(x->op)) {
        if (x->a != -1 || x->b != -1) return TR_MATH_INVALID;
    } else if (unary(x->op)) {
        if (x->a < 0 || x->a >= i || x->b != -1) return TR_MATH_INVALID;
        c.a = x->a;
    } else if (binary(x->op)) {
        if (x->a < 0 || x->a >= i || x->b < 0 || x->b >= i) return TR_MATH_INVALID;
        c.a = x->a; c.b = x->b;
    } else return TR_MATH_INVALID;
    switch (x->op) {
    case TR_JET_PARAM: if (x->index < 0 || x->index >= nparams) return TR_MATH_INVALID; c.index = x->index; break;
    case TR_JET_DATA: if (x->index < 0 || x->index >= ncols) return TR_MATH_INVALID; c.index = x->index; break;
    case TR_JET_POW:
        if (x->index < 0) return TR_MATH_INVALID;
        if (x->index > TR_JET_MAX_EXPONENT) return TR_MATH_UNSUPPORTED;
        c.index = x->index;
        break;
    case TR_JET_CONST: case TR_JET_SCALE: {
        tr_num_rational q;
        if (x->value.denominator == 0) return TR_MATH_INVALID;
        if (!tr_q_make(x->value.numerator, x->value.denominator, &q)) return TR_MATH_OVERFLOW;
        c.value = q;
        break;
    }
    default: break;
    }
    if (out) *out = c;
    return TR_MATH_OK;
}

static int32_t check_shape(const tr_jet_node *nodes, int32_t nnodes, int32_t nparams, int32_t ncols) {
    if (nnodes < 1 || !nodes || nparams < 0 || ncols < 0) return TR_MATH_INVALID;
    return TR_MATH_OK;
}
static int32_t check_caps(int32_t nnodes, int32_t nparams, int32_t ncols) {
    if (nnodes > TR_JET_MAX_NODES || nparams > TR_JET_MAX_PARAMS || ncols > TR_JET_MAX_COLUMNS) return TR_MATH_UNSUPPORTED;
    return TR_MATH_OK;
}

int32_t tr_jet_circuit_need(const tr_jet_node *nodes, int32_t nnodes, int32_t nparams, int32_t ncols, tr_math_need *need) {
    if (!need) return TR_MATH_INVALID;
    int32_t rc = check_shape(nodes, nnodes, nparams, ncols);
    if (rc) return rc;
    for (int32_t i = 0; i < nnodes; i++)
        if ((rc = check_node(&nodes[i], i, nparams, ncols, NULL)) && rc != TR_MATH_UNSUPPORTED) return rc;
    need_zero(need);
    need->need[TR_BUDGET_VARIABLES] = (uint64_t)nparams + (uint64_t)ncols;
    need->need[TR_BUDGET_CIRCUIT] = (uint64_t)nnodes;
    need->need[TR_BUDGET_BYTES] = tr_jet_circuit_bytes(nnodes);
    return TR_MATH_OK;
}

static int32_t sat_add(int32_t a, int32_t b) { int64_t s = (int64_t)a + b; return s > TR_JET_DEGREE_CAP ? TR_JET_DEGREE_CAP : (int32_t)s; }
static int32_t sat_mul(int32_t a, int32_t b) { int64_t s = (int64_t)a * b; return s > TR_JET_DEGREE_CAP ? TR_JET_DEGREE_CAP : (int32_t)s; }

tr_jet_circuit *tr_jet_circuit_create(const tr_jet_node *nodes, int32_t nnodes, int32_t nparams, int32_t ncols,
                                      const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    int32_t rc = check_shape(nodes, nnodes, nparams, ncols);
    if (rc) { set_error(error, rc); return NULL; }
    for (int32_t i = 0; i < nnodes; i++) {           /* malformed input first (the caps come after the budget) */
        rc = check_node(&nodes[i], i, nparams, ncols, NULL);
        if (rc && rc != TR_MATH_UNSUPPORTED) { set_error(error, rc); return NULL; }
    }
    tr_math_need need;
    tr_math_budget dflt;
    tr_jet_circuit_need(nodes, nnodes, nparams, ncols, &need);
    rc = tr_budget_check(budget_of(budget, &dflt), &need, refusal);
    if (!rc) rc = check_caps(nnodes, nparams, ncols);
    for (int32_t i = 0; !rc && i < nnodes; i++) rc = check_node(&nodes[i], i, nparams, ncols, NULL);
    if (rc) { set_error(error, rc); return NULL; }
    tr_jet_circuit *c = (tr_jet_circuit *)zalloc(sizeof *c);
    int32_t *deg = (int32_t *)zalloc((uint64_t)nnodes * 8);
    if (c) { c->nodes = (tr_jet_node *)zalloc((uint64_t)nnodes * sizeof(tr_jet_node)); c->flags = (uint8_t *)zalloc((uint64_t)nnodes); }
    if (!c || !deg || !c->nodes || !c->flags) {
        if (c) { free(c->nodes); free(c->flags); free(c); }
        free(deg);
        set_error(error, TR_MATH_MEMORY);
        return NULL;
    }
    int32_t *pdeg = deg + nnodes;
    for (int32_t i = 0; i < nnodes; i++) {
        check_node(&nodes[i], i, nparams, ncols, &c->nodes[i]);
        const tr_jet_node *x = &c->nodes[i];
        int32_t da = x->a >= 0 ? deg[x->a] : 0, db = x->b >= 0 ? deg[x->b] : 0;
        int32_t pa = x->a >= 0 ? pdeg[x->a] : 0, pb = x->b >= 0 ? pdeg[x->b] : 0;
        switch (x->op) {
        case TR_JET_CONST: deg[i] = 0; pdeg[i] = 0; break;
        case TR_JET_PARAM: deg[i] = 1; pdeg[i] = 1; break;
        case TR_JET_DATA: deg[i] = 1; pdeg[i] = 0; break;
        case TR_JET_ADD: case TR_JET_SUB: deg[i] = da > db ? da : db; pdeg[i] = pa > pb ? pa : pb; break;
        case TR_JET_MUL: deg[i] = sat_add(da, db); pdeg[i] = sat_add(pa, pb); break;
        case TR_JET_POW: deg[i] = sat_mul(da, x->index); pdeg[i] = sat_mul(pa, x->index); break;
        default: deg[i] = da; pdeg[i] = pa; break;
        }
    }
    c->flags[nnodes - 1] = REACH | EVAL;
    for (int32_t i = nnodes - 1; i >= 0; i--) {
        const tr_jet_node *x = &c->nodes[i];
        uint8_t f = c->flags[i];
        if (!f) continue;
        uint8_t down = (x->op == TR_JET_POW && x->index == 0) ? (uint8_t)(f & REACH) : f;   /* a^0: not evaluated */
        if (x->a >= 0) c->flags[x->a] |= down;
        if (x->b >= 0) c->flags[x->b] |= down;
        if (f & REACH) {
            c->live++;
            if (x->op == TR_JET_DATA) c->columns |= UINT32_C(1) << x->index;
            if (x->op == TR_JET_PARAM) c->params |= UINT64_C(1) << x->index;
        }
        if (f & EVAL) c->evaluated++;
    }
    c->degree = deg[nnodes - 1]; c->param_degree = pdeg[nnodes - 1];
    free(deg);
    c->nnodes = nnodes; c->nparams = nparams; c->ncols = ncols;
    c->bytes = tr_jet_circuit_bytes(nnodes);
    c->magic = CIRCUIT_MAGIC;
    set_error(error, TR_MATH_OK);
    return c;
}

int32_t tr_jet_circuit_get_info(const tr_jet_circuit *c, tr_jet_circuit_info *out) {
    if (!circuit_valid(c) || !out) return TR_MATH_INVALID;
    memset(out, 0, sizeof *out);
    out->nodes = c->nnodes; out->nparams = c->nparams; out->ncols = c->ncols; out->live = c->live;
    out->degree = c->degree; out->param_degree = c->param_degree; out->columns = c->columns;
    out->params = c->params; out->bytes = c->bytes;
    return TR_MATH_OK;
}
int32_t tr_jet_circuit_get_nodes(const tr_jet_circuit *c, tr_jet_node *out, int32_t capacity) {
    if (!circuit_valid(c) || !out || capacity < c->nnodes) return TR_MATH_INVALID;
    memcpy(out, c->nodes, (size_t)c->nnodes * sizeof(tr_jet_node));
    return TR_MATH_OK;
}
void tr_jet_circuit_free(tr_jet_circuit *c) {
    if (!circuit_valid(c)) return;
    free(c->nodes); free(c->flags);
    c->magic = 0;
    free(c);
}

/* ================================================================== plans */
struct tr_jet_plan_s {
    uint32_t magic;
    int32_t order, directions, nparams, ncols, nnodes, jets, out_jet, out_pdeg;
    uint32_t support, nsteps, nregs, nslots, columns, out;
    uint64_t out_dep, pairs, workspace, bytes, mults, adds, key_words;
    int64_t denominator;
    uint64_t digest[2];
    tr_jet_step *steps;
    int64_t *leaf;              /* nparams x (m + 1) */
    uint8_t *exps;              /* S x m */
    uint8_t *degree;            /* S */
    uint64_t *mask;             /* S */
    int64_t *fact;              /* S: alpha! */
    uint32_t *conv_start;       /* S + 1 */
    uint32_t *conv_pairs;       /* pairs */
    uint32_t *upto;             /* order + 1: monomials of degree <= k */
    int64_t *denoms;            /* nnodes */
    int32_t *scales;            /* ncols */
    int64_t *key;               /* the exact descriptor */
};
static int plan_valid(const tr_jet_plan *p) { return p && p->magic == PLAN_MAGIC; }

static uint64_t key_words(int32_t nnodes, int32_t nparams, int32_t ncols, int32_t m) {
    return 6 + 6 * (uint64_t)nnodes + 2 * (uint64_t)nparams + 2 * (uint64_t)m * (uint64_t)nparams + (uint64_t)ncols;
}
/* every byte the plan allocates (kept) and its compile scratch (freed), from the circuit and the jet shape */
static uint64_t plan_bytes(const tr_jet_circuit *c, int32_t m, int32_t d, uint64_t S, uint64_t pairs) {
    uint64_t b = sizeof(tr_jet_plan);
    b = tr_budget_add(b, tr_budget_mul((uint64_t)c->evaluated, sizeof(tr_jet_step)));
    b = tr_budget_add(b, tr_budget_mul(tr_budget_mul((uint64_t)c->nparams, (uint64_t)m + 1), 8));
    b = tr_budget_add(b, tr_budget_mul(S, (uint64_t)m + 1 + 8 + 8));          /* exps, degree, mask, fact */
    b = tr_budget_add(b, tr_budget_mul(tr_budget_add(S, 1), 4));
    b = tr_budget_add(b, tr_budget_mul(pairs, 4));
    b = tr_budget_add(b, tr_budget_mul((uint64_t)d + 1, 4));                 /* upto */
    b = tr_budget_add(b, tr_budget_mul((uint64_t)c->nnodes, 8));             /* denominators */
    b = tr_budget_add(b, tr_budget_mul((uint64_t)c->ncols, 4));
    b = tr_budget_add(b, tr_budget_mul(key_words(c->nnodes, c->nparams, c->ncols, m), 8));
    /* compile scratch: per node the direction mask, last use, register / slot, kind and degree, the free lists; the
     * canonical center and seeds, the parameters' denominators and masks; the ranking tables */
    b = tr_budget_add(b, tr_budget_add(tr_budget_mul((uint64_t)c->nnodes, 8 + 4 + 4 + 1 + 1 + 8), 24));
    b = tr_budget_add(b, tr_budget_add(tr_budget_mul((uint64_t)c->nparams, 16 + 16), 48));
    b = tr_budget_add(b, tr_budget_mul(tr_budget_mul((uint64_t)m, (uint64_t)c->nparams), 16));
    b = tr_budget_add(b, tr_budget_mul(tr_budget_add(tr_budget_mul((uint64_t)m + 1, (uint64_t)d + 1), (uint64_t)d + 2), 8));
    return b;
}

int32_t tr_jet_plan_need(const tr_jet_circuit *c, int32_t order, int32_t m, tr_math_need *need) {
    if (!circuit_valid(c) || !need || order < 0 || m < 0) return TR_MATH_INVALID;
    need_zero(need);
    uint64_t S = tr_jet_support((uint32_t)m, (uint32_t)order), pairs = tr_jet_pairs((uint32_t)m, (uint32_t)order);
    need->need[TR_BUDGET_VARIABLES] = (uint64_t)m;
    need->need[TR_BUDGET_DEGREE] = (uint64_t)order;
    need->need[TR_BUDGET_SUPPORT] = S;
    need->need[TR_BUDGET_TENSOR] = pairs;
    need->need[TR_BUDGET_CIRCUIT] = (uint64_t)c->evaluated;
    need->need[TR_BUDGET_BYTES] = plan_bytes(c, m, order, S, pairs);
    return TR_MATH_OK;
}

/* the sorted index tuple of exponent row e (degree k) -> its index (graded, lexicographic tuples) */
static uint32_t rank_of(const uint8_t *e, int32_t m, int k, const uint64_t *pm, const uint64_t *offset) {
    uint64_t r = offset[k];
    int p = 0, prev = 0;
    for (int32_t i = 0; i < m; i++)
        for (int t = 0; t < e[i]; t++) {                     /* tuple position p (0-based) takes value i */
            int rest = k - p - 1;                            /* multisets after this position */
            r += pm[(uint64_t)rest * (uint64_t)(m + 1) + (uint64_t)i] - pm[(uint64_t)rest * (uint64_t)(m + 1) + (uint64_t)prev];
            prev = i; p++;
        }
    return (uint32_t)r;
}

/* builds exps, degree, mask, fact, conv_start, conv_pairs (sizes checked by the caller) */
static int32_t build_tables(tr_jet_plan *p) {
    const int32_t m = p->directions, d = p->order;
    const uint32_t S = p->support;
    uint8_t *ex = p->exps;
    p->degree[0] = 0; p->mask[0] = 0; p->fact[0] = 1;
    uint32_t idx = 1;
    int t[TR_JET_MAX_ORDER];
    for (int k = 1; k <= d && m > 0; k++) {
        for (int i = 0; i < k; i++) t[i] = 0;
        for (;;) {
            uint8_t *row = ex + (uint64_t)idx * (uint64_t)m;
            uint64_t mk = 0;
            for (int i = 0; i < k; i++) { row[t[i]]++; mk |= UINT64_C(1) << t[i]; }
            int64_t f = 1;
            for (int32_t j = 0; j < m; j++) for (int q = 2; q <= row[j]; q++) f *= q;   /* <= 16!: fits */
            p->degree[idx] = (uint8_t)k; p->mask[idx] = mk; p->fact[idx] = f;
            idx++;
            int i = k - 1;
            while (i >= 0 && t[i] == m - 1) i--;
            if (i < 0) break;
            t[i]++;
            for (int j = i + 1; j < k; j++) t[j] = t[i];
        }
    }
    if (idx != S) return TR_MATH_INVALID;                    /* unreachable: S = C(m + d, d) */
    for (int k = 0; k <= d; k++) p->upto[k] = (uint32_t)tr_budget_monomials((uint32_t)m, (uint32_t)k);
    /* ranking: M(n, r) = C(n + r - 1, r) multisets of size r from n values; pm[r][v] = sum_{u < v} M(m - u, r) */
    const uint64_t W = (uint64_t)m + 1;
    uint64_t *pm = (uint64_t *)zalloc(W * ((uint64_t)d + 1) * 8), *offset = (uint64_t *)zalloc(((uint64_t)d + 2) * 8);
    if (!pm || !offset) { free(pm); free(offset); return TR_MATH_MEMORY; }
    for (int r = 0; r <= d; r++)
        for (int32_t v = 1; v <= m; v++) {
            uint64_t n = (uint64_t)(m - (v - 1));
            uint64_t mult = r == 0 ? 1 : tr_budget_binomial(n + (uint64_t)r - 1, (uint64_t)r);
            pm[(uint64_t)r * W + (uint64_t)v] = pm[(uint64_t)r * W + (uint64_t)v - 1] + mult;
        }
    offset[0] = 0;
    for (int k = 1; k <= d; k++) offset[k] = tr_budget_binomial((uint64_t)m + (uint64_t)k - 1, (uint64_t)k - 1);
    uint64_t e = 0;
    uint8_t al[TR_JET_MAX_DIRECTIONS], be[TR_JET_MAX_DIRECTIONS];
    int pos[TR_JET_MAX_ORDER], np;
    for (uint32_t g = 0; g < S; g++) {
        p->conv_start[g] = (uint32_t)e;
        const uint8_t *gr = ex + (uint64_t)g * (uint64_t)m;
        np = 0;
        for (int32_t i = 0; i < m; i++) if (gr[i]) pos[np++] = i;
        uint8_t cnt[TR_JET_MAX_ORDER];
        for (int i = 0; i < np; i++) cnt[i] = 0;
        for (;;) {                                           /* alpha <= gamma, odometer over gamma's support */
            int ka = 0;
            for (int32_t i = 0; i < m; i++) { al[i] = 0; be[i] = gr[i]; }
            for (int i = 0; i < np; i++) { al[pos[i]] = cnt[i]; be[pos[i]] = (uint8_t)(gr[pos[i]] - cnt[i]); ka += cnt[i]; }
            uint32_t ra = rank_of(al, m, ka, pm, offset), rb = rank_of(be, m, p->degree[g] - ka, pm, offset);
            p->conv_pairs[e++] = ra | (rb << 16);
            int i = np - 1;
            while (i >= 0 && cnt[i] == gr[pos[i]]) { cnt[i] = 0; i--; }
            if (i < 0) break;
            cnt[i]++;
        }
    }
    p->conv_start[S] = (uint32_t)e;
    free(pm); free(offset);
    return e == p->pairs ? TR_MATH_OK : TR_MATH_INVALID;     /* pairs = C(2m + d, d) */
}

/* static operation counts of tr_jet_eval (the same loops, the same support tests) */
static uint64_t count_conv(const tr_jet_view *v, uint64_t dep, int lo, int hi, uint64_t da, int alo, int ahi,
                           uint64_t db, int blo, int bhi) {
    uint64_t n = 0;
    for (uint32_t g = 0; g < v->support; g++) {
        if (!tr_jet_in(v, g, dep, lo, hi)) continue;
        for (uint32_t e = v->conv_start[g]; e < v->conv_start[g + 1]; e++) {
            uint32_t pr = v->conv_pairs[e];
            if (tr_jet_in(v, pr & 0xffffu, da, alo, ahi) && tr_jet_in(v, pr >> 16, db, blo, bhi)) n++;
        }
    }
    return n;
}
static uint64_t count_support(const tr_jet_view *v, uint64_t dep, int lo, int hi) {
    uint64_t n = 0;
    for (uint32_t g = 0; g < v->support; g++) n += (uint64_t)tr_jet_in(v, g, dep, lo, hi);
    return n;
}
static uint64_t count_pow64(uint32_t n) {                   /* multiplications of tr_jet_pow64 */
    uint64_t c = 0;
    while (n) { if (n & 1) c++; n >>= 1; if (n) c++; }
    return c;
}
static void count_step(const tr_jet_view *v, const tr_jet_step *st, uint64_t *mul, uint64_t *add) {
    const int rr = st->jet & TR_JET_R, ra = st->jet & TR_JET_A, rb = st->jet & TR_JET_B;
    switch (st->op) {
    case TR_JET_ADD: case TR_JET_SUB: {
        uint64_t n = rr ? count_support(v, st->dep, 0, st->pdeg) : 1;
        *mul += 2 * n; *add += n;
        break;
    }
    case TR_JET_MUL:
        if (!rr) { *mul += 1; break; }
        if (ra && rb) {
            uint64_t n = count_conv(v, st->dep, 0, st->pdeg, st->dep_a, 0, st->pdeg_a, st->dep_b, 0, st->pdeg_b);
            *mul += n; *add += n;
        } else *mul += count_support(v, st->dep, 0, st->pdeg);
        break;
    case TR_JET_SCALE: case TR_JET_NEG: *mul += rr ? count_support(v, st->dep, 0, st->pdeg) : 1; break;
    case TR_JET_POW: {
        uint32_t n = (uint32_t)st->index;
        if (n == 0) break;
        if (!rr) { *mul += count_pow64(n); break; }
        uint32_t K = n < (uint32_t)v->order ? n : (uint32_t)v->order;
        int d = v->order, pa = st->pdeg_a;
        *mul += count_pow64(n - K) + K + 2 * (uint64_t)K;
        for (uint32_t k = 1; k <= K; k++) {
            int hi = (int)k * pa < d ? (int)k * pa : d;
            if (k >= 2) {
                uint64_t c = count_conv(v, st->dep_a, (int)k, hi, st->dep_a, (int)k - 1,
                                        (int)(k - 1) * pa < d ? (int)(k - 1) * pa : d, st->dep_a, 1, pa);
                *mul += c; *add += c;
            }
            uint64_t s = count_support(v, st->dep_a, (int)k, hi);
            *mul += s; *add += s;
        }
        break;
    }
    default: break;
    }
}

/* the free lists of the register / slot allocation */
typedef struct { uint32_t *free; uint32_t nfree, used; } alloc_list;
static uint32_t take(alloc_list *l) { return l->nfree ? l->free[--l->nfree] : l->used++; }
static void give(alloc_list *l, uint32_t r) { l->free[l->nfree++] = r; }

static void plan_view(const tr_jet_plan *p, tr_jet_view *v);
static void plan_release(tr_jet_plan *p) {
    if (!p) return;
    free(p->steps); free(p->leaf); free(p->exps); free(p->degree); free(p->mask); free(p->fact);
    free(p->conv_start); free(p->conv_pairs); free(p->upto); free(p->denoms); free(p->scales); free(p->key);
    p->magic = 0;
    free(p);
}
void tr_jet_plan_free(tr_jet_plan *p) { if (plan_valid(p)) plan_release(p); }

static int32_t canonical(tr_num_rational x, tr_num_rational *out) {
    if (x.denominator == 0) return TR_MATH_INVALID;
    return tr_q_make(x.numerator, x.denominator, out) ? TR_MATH_OK : TR_MATH_OVERFLOW;
}

tr_jet_plan *tr_jet_plan_create(const tr_jet_circuit *c, int32_t order, int32_t m, const tr_num_rational *center,
                                const tr_num_rational *seeds, const int32_t *scales, const tr_math_budget *budget,
                                tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    if (!circuit_valid(c) || order < 0 || m < 0 || (!seeds && m != c->nparams)) { set_error(error, TR_MATH_INVALID); return NULL; }
    const int32_t np = c->nparams, nc = c->ncols, nn = c->nnodes;
    for (int32_t i = 0; scales && i < nc; i++)
        if (scales[i] < 0 || scales[i] > TR_SCOPE_MAX_SCALE) { set_error(error, TR_MATH_INVALID); return NULL; }
    for (int32_t i = 0; center && i < np; i++)
        if (center[i].denominator == 0) { set_error(error, TR_MATH_INVALID); return NULL; }
    for (uint64_t i = 0; seeds && i < (uint64_t)m * (uint64_t)np; i++)
        if (seeds[i].denominator == 0) { set_error(error, TR_MATH_INVALID); return NULL; }
    tr_math_need need;
    tr_math_budget dflt;
    tr_jet_plan_need(c, order, m, &need);
    int32_t rc = tr_budget_check(budget_of(budget, &dflt), &need, refusal);
    if (!rc && (m > TR_JET_MAX_DIRECTIONS || order > TR_JET_MAX_ORDER || need.need[TR_BUDGET_SUPPORT] > TR_JET_MAX_SUPPORT ||
                need.need[TR_BUDGET_TENSOR] >= UINT32_MAX)) rc = TR_MATH_UNSUPPORTED;
    if (rc) { set_error(error, rc); return NULL; }
    /* canonical center and seeds (checked) */
    tr_num_rational *cen = (tr_num_rational *)zalloc((uint64_t)np * sizeof *cen + 16);
    tr_num_rational *sd = (tr_num_rational *)zalloc((uint64_t)m * (uint64_t)np * sizeof *sd);
    if (!cen || !sd) { free(cen); free(sd); set_error(error, TR_MATH_MEMORY); return NULL; }
    for (int32_t i = 0; i < np && !rc; i++) {
        tr_num_rational z = {0, 1};
        rc = canonical(center ? center[i] : z, &cen[i]);
    }
    for (int32_t j = 0; j < m && !rc; j++)
        for (int32_t i = 0; i < np && !rc; i++) {
            tr_num_rational z = {seeds ? 0 : (i == j), 1};
            rc = canonical(seeds ? seeds[(uint64_t)j * (uint64_t)np + (uint64_t)i] : z, &sd[(uint64_t)j * (uint64_t)np + (uint64_t)i]);
        }
    if (rc) { free(cen); free(sd); set_error(error, rc); return NULL; }

    const uint32_t S = (uint32_t)need.need[TR_BUDGET_SUPPORT];
    tr_jet_plan *p = (tr_jet_plan *)zalloc(sizeof *p);
    uint64_t *dep = (uint64_t *)zalloc((uint64_t)nn * 8);
    uint32_t *last = (uint32_t *)zalloc((uint64_t)nn * 4), *where = (uint32_t *)zalloc((uint64_t)nn * 4);
    uint8_t *isjet = (uint8_t *)zalloc((uint64_t)nn), *pdeg = (uint8_t *)zalloc((uint64_t)nn);
    uint32_t *fr = (uint32_t *)zalloc(((uint64_t)nn + 3) * 8);    /* the two free lists: nn + 3 entries each */
    if (p) {
        p->steps = (tr_jet_step *)zalloc((uint64_t)c->evaluated * sizeof(tr_jet_step));
        p->leaf = (int64_t *)zalloc((uint64_t)np * ((uint64_t)m + 1) * 8);
        p->exps = (uint8_t *)zalloc((uint64_t)S * (uint64_t)m);
        p->degree = (uint8_t *)zalloc(S); p->mask = (uint64_t *)zalloc((uint64_t)S * 8); p->fact = (int64_t *)zalloc((uint64_t)S * 8);
        p->conv_start = (uint32_t *)zalloc(((uint64_t)S + 1) * 4);
        p->conv_pairs = (uint32_t *)zalloc(need.need[TR_BUDGET_TENSOR] * 4);
        p->upto = (uint32_t *)zalloc(((uint64_t)order + 1) * 4);
        p->denoms = (int64_t *)zalloc((uint64_t)nn * 8); p->scales = (int32_t *)zalloc((uint64_t)nc * 4);
        p->key_words = key_words(nn, np, nc, m);
        p->key = (int64_t *)zalloc(p->key_words * 8);
    }
    if (!p || !dep || !last || !where || !isjet || !pdeg || !fr || !p->steps || !p->leaf || !p->exps || !p->degree ||
        !p->mask || !p->fact || !p->conv_start || !p->conv_pairs || !p->upto || !p->denoms || !p->scales || !p->key) {
        rc = TR_MATH_MEMORY;
        goto done;
    }
    p->order = order; p->directions = m; p->nparams = np; p->ncols = nc; p->nnodes = nn;
    p->support = S; p->pairs = need.need[TR_BUDGET_TENSOR]; p->columns = c->columns;
    for (int32_t i = 0; i < nc; i++) p->scales[i] = scales ? scales[i] : 0;
    if ((rc = build_tables(p))) goto done;

    /* per parameter: its denominator lcm(center, seeds), its leaf numerators, its direction mask */
    int64_t *pden = (int64_t *)zalloc((uint64_t)np * 8 + 16);
    uint64_t *pdep = (uint64_t *)zalloc((uint64_t)np * 8 + 16);
    if (!pden || !pdep) { free(pden); free(pdep); rc = TR_MATH_MEMORY; goto done; }
    for (int32_t i = 0; i < np && !rc; i++) {
        int64_t D = cen[i].denominator;
        for (int32_t j = 0; j < m && !rc; j++) {
            tr_num_rational u = sd[(uint64_t)j * (uint64_t)np + (uint64_t)i];
            if (!u.numerator) continue;
            if (order > 0) pdep[i] |= UINT64_C(1) << j;
            if (!lcm64(D, u.denominator, &D)) rc = TR_MATH_OVERFLOW;
        }
        if (rc) break;
        pden[i] = D;
        int64_t *L = p->leaf + (uint64_t)i * ((uint64_t)m + 1);
        if (!tr_mul_i64(cen[i].numerator, D / cen[i].denominator, &L[0])) rc = TR_MATH_OVERFLOW;
        for (int32_t j = 0; j < m && !rc; j++) {
            tr_num_rational u = sd[(uint64_t)j * (uint64_t)np + (uint64_t)i];
            if (u.numerator && !tr_mul_i64(u.numerator, D / u.denominator, &L[1 + j])) rc = TR_MATH_OVERFLOW;
        }
    }
    /* denominators, masks, degrees, kinds, last uses */
    const tr_jet_node *nd = c->nodes;
    for (int32_t i = 0; i < nn && !rc; i++) {
        if (!(c->flags[i] & EVAL)) continue;
        const tr_jet_node *x = &nd[i];
        int64_t D = 1, Da = x->a >= 0 ? p->denoms[x->a] : 1, Db = x->b >= 0 ? p->denoms[x->b] : 1;
        uint64_t da = x->a >= 0 ? dep[x->a] : 0, db = x->b >= 0 ? dep[x->b] : 0, dv = 0;
        int pa = x->a >= 0 ? pdeg[x->a] : 0, pb = x->b >= 0 ? pdeg[x->b] : 0, pv = 0;
        switch (x->op) {
        case TR_JET_CONST: D = x->value.denominator; break;
        case TR_JET_PARAM: D = pden[x->index]; dv = pdep[x->index]; pv = dv ? 1 : 0; break;
        case TR_JET_DATA:
            for (int32_t k = 0; k < p->scales[x->index]; k++) D *= 10;      /* <= 10^18 */
            break;
        case TR_JET_ADD: case TR_JET_SUB:
            if (!lcm64(Da, Db, &D)) rc = TR_MATH_OVERFLOW;
            dv = da | db; pv = pa > pb ? pa : pb;
            break;
        case TR_JET_MUL:
            if (!tr_mul_i64(Da, Db, &D)) rc = TR_MATH_OVERFLOW;
            dv = da | db; pv = pa + pb < order ? pa + pb : order;
            break;
        case TR_JET_POW:
            if (x->index == 0) { D = 1; break; }
            if (!tr_jet_pow64(Da, (uint32_t)x->index, &D)) rc = TR_MATH_OVERFLOW;
            dv = da; pv = (int64_t)pa * x->index < order ? pa * x->index : order;
            break;
        case TR_JET_SCALE: {
            int64_t g = (int64_t)tr_gcd_u64(tr_abs_u64(x->value.numerator), (uint64_t)Da);
            if (!tr_mul_i64(x->value.denominator, Da / g, &D)) rc = TR_MATH_OVERFLOW;
            dv = da; pv = pa;
            break;
        }
        default: D = Da; dv = da; pv = pa; break;          /* NEG */
        }
        if (rc) break;
        p->denoms[i] = D; dep[i] = dv; pdeg[i] = (uint8_t)pv; isjet[i] = dv != 0;
        last[i] = UINT32_MAX;
        if (x->op == TR_JET_POW && x->index == 0) continue;  /* the operand is not evaluated here */
        if (x->a >= 0) last[x->a] = (uint32_t)i;
        if (x->b >= 0) last[x->b] = (uint32_t)i;
    }
    if (!rc) {
        int32_t o = nn - 1;
        last[o] = UINT32_MAX;
        for (int32_t i = 0; i < nn; i++)                       /* a value no evaluated node reads dies at once */
            if ((c->flags[i] & EVAL) && i != o && last[i] == UINT32_MAX) last[i] = (uint32_t)i;
    }
    free(pden); free(pdep);
    if (rc) goto done;

    /* steps, with registers and slots allocated by last use */
    alloc_list regs = {fr, 0, 0}, slots = {fr + nn + 3, 0, 0};
    uint32_t ns = 0;
    for (int32_t i = 0; i < nn; i++) {
        if (!(c->flags[i] & EVAL)) continue;
        const tr_jet_node *x = &nd[i];
        tr_jet_step *st = &p->steps[ns++];
        memset(st, 0, sizeof *st);
        int op = x->op, n = x->index;
        if (op == TR_JET_POW && n == 1) { op = TR_JET_SCALE; }   /* a^1 = 1 * a */
        st->op = (uint8_t)op;
        st->index = x->index;
        st->pdeg = pdeg[i];
        st->dep = dep[i];
        if (x->a >= 0 && !(x->op == TR_JET_POW && n == 0)) {
            st->a = where[x->a]; st->dep_a = dep[x->a]; st->pdeg_a = pdeg[x->a];
            if (isjet[x->a]) st->jet |= TR_JET_A;
        }
        if (x->b >= 0) {
            st->b = where[x->b]; st->dep_b = dep[x->b]; st->pdeg_b = pdeg[x->b];
            if (isjet[x->b]) st->jet |= TR_JET_B;
        }
        switch (op) {
        case TR_JET_CONST: st->ma = x->value.numerator; break;
        case TR_JET_ADD: case TR_JET_SUB:
            st->ma = p->denoms[i] / p->denoms[x->a]; st->mb = p->denoms[i] / p->denoms[x->b];
            break;
        case TR_JET_SCALE:
            if (x->op == TR_JET_POW) st->ma = 1;
            else st->ma = x->value.numerator / (int64_t)tr_gcd_u64(tr_abs_u64(x->value.numerator), (uint64_t)p->denoms[x->a]);
            break;
        default: break;
        }
        if (x->op == TR_JET_POW && n == 0) { st->op = TR_JET_POW; st->index = 0; }
        if (isjet[i]) {
            st->jet |= TR_JET_R;
            st->dst = take(&slots);
            p->jets++;
            if (op == TR_JET_POW && n >= 2) {
                st->t0 = take(&slots);
                st->t1 = (n >= 2 && order >= 2) ? take(&slots) : st->t0;
                give(&slots, st->t0);
                if (st->t1 != st->t0) give(&slots, st->t1);
            }
        } else st->dst = take(&regs);
        where[i] = st->dst;
        if (x->a >= 0 && last[x->a] == (uint32_t)i) give(isjet[x->a] ? &slots : &regs, where[x->a]);
        if (x->b >= 0 && x->b != x->a && last[x->b] == (uint32_t)i) give(isjet[x->b] ? &slots : &regs, where[x->b]);
        if (last[i] == (uint32_t)i) give(isjet[i] ? &slots : &regs, st->dst);   /* never read (not the output) */
    }
    p->nsteps = ns; p->nregs = regs.used; p->nslots = slots.used;
    p->out = where[nn - 1]; p->out_jet = isjet[nn - 1]; p->out_dep = dep[nn - 1]; p->out_pdeg = pdeg[nn - 1];
    p->denominator = p->denoms[nn - 1];
    p->workspace = (uint64_t)p->nregs + (uint64_t)p->nslots * S;
    {
        tr_jet_view v;
        plan_view(p, &v);
        for (uint32_t s = 0; s < ns; s++) count_step(&v, &p->steps[s], &p->mults, &p->adds);
    }
    /* the exact descriptor */
    {
        int64_t *k = p->key;
        uint64_t w = 0;
        k[w++] = 0x4a455431; k[w++] = order; k[w++] = m; k[w++] = np; k[w++] = nc; k[w++] = nn;
        for (int32_t i = 0; i < nn; i++) {
            k[w++] = nd[i].op; k[w++] = nd[i].a; k[w++] = nd[i].b; k[w++] = nd[i].index;
            k[w++] = nd[i].value.numerator; k[w++] = nd[i].value.denominator;
        }
        for (int32_t i = 0; i < np; i++) { k[w++] = cen[i].numerator; k[w++] = cen[i].denominator; }
        for (uint64_t i = 0; i < (uint64_t)m * (uint64_t)np; i++) { k[w++] = sd[i].numerator; k[w++] = sd[i].denominator; }
        for (int32_t i = 0; i < nc; i++) k[w++] = p->scales[i];
        digest_words(k, w, p->digest);
    }
    p->bytes = plan_bytes(c, m, order, S, p->pairs);
    p->magic = PLAN_MAGIC;
done:
    free(cen); free(sd); free(dep); free(last); free(where); free(isjet); free(pdeg); free(fr);
    if (rc) { plan_release(p); set_error(error, rc); return NULL; }
    set_error(error, TR_MATH_OK);
    return p;
}

static void plan_view(const tr_jet_plan *p, tr_jet_view *v) {
    memset(v, 0, sizeof *v);
    v->support = p->support; v->nsteps = p->nsteps; v->nregs = p->nregs; v->nslots = p->nslots;
    v->order = p->order; v->directions = p->directions; v->nparams = p->nparams; v->ncols = p->ncols;
    v->columns = p->columns; v->out = p->out; v->out_jet = p->out_jet;
    v->denominator = p->denominator; v->pairs = p->pairs; v->workspace = p->workspace;
    v->steps = p->steps; v->leaf = p->leaf; v->degree = p->degree; v->mask = p->mask;
    v->conv_start = p->conv_start; v->conv_pairs = p->conv_pairs; v->scales = p->scales; v->upto = p->upto;
    v->out_pdeg = p->out_pdeg; v->out_dep = p->out_dep;
}
int32_t tr_jet_plan_get_view(const tr_jet_plan *p, tr_jet_view *v) {
    if (!plan_valid(p) || !v) return TR_MATH_INVALID;
    plan_view(p, v);
    return TR_MATH_OK;
}
int32_t tr_jet_plan_get_info(const tr_jet_plan *p, tr_jet_plan_info *out) {
    if (!plan_valid(p) || !out) return TR_MATH_INVALID;
    memset(out, 0, sizeof *out);
    out->support = p->support; out->pairs = p->pairs; out->steps = p->nsteps; out->registers = p->nregs;
    out->slots = p->nslots; out->workspace = p->workspace; out->bytes = p->bytes;
    out->multiplications = p->mults; out->additions = p->adds;
    out->digest[0] = p->digest[0]; out->digest[1] = p->digest[1]; out->denominator = p->denominator;
    out->order = p->order; out->directions = p->directions; out->nparams = p->nparams; out->ncols = p->ncols;
    out->nodes = p->nnodes; out->jets = p->jets; out->out_jet = p->out_jet; out->out_pdeg = p->out_pdeg;
    out->columns = p->columns; out->out_dep = p->out_dep;
    return TR_MATH_OK;
}
int32_t tr_jet_plan_monomials(const tr_jet_plan *p, uint8_t *exps, uint64_t capacity) {
    if (!plan_valid(p)) return TR_MATH_INVALID;
    uint64_t n = (uint64_t)p->support * (uint64_t)p->directions;
    if (n && (!exps || capacity < n)) return TR_MATH_INVALID;
    if (n) memcpy(exps, p->exps, (size_t)n);
    return TR_MATH_OK;
}
int32_t tr_jet_plan_steps(const tr_jet_plan *p, tr_jet_step *out, uint64_t capacity) {
    if (!plan_valid(p) || (p->nsteps && (!out || capacity < p->nsteps))) return TR_MATH_INVALID;
    if (p->nsteps) memcpy(out, p->steps, (size_t)p->nsteps * sizeof(tr_jet_step));
    return TR_MATH_OK;
}
int32_t tr_jet_plan_denominators(const tr_jet_plan *p, int64_t *out, uint64_t capacity) {
    if (!plan_valid(p) || !out || capacity < (uint64_t)p->nnodes) return TR_MATH_INVALID;
    memcpy(out, p->denoms, (size_t)p->nnodes * 8);
    return TR_MATH_OK;
}
static int same_key(const int64_t *a, uint64_t na, const int64_t *b, uint64_t nb) {
    return na == nb && !memcmp(a, b, (size_t)na * 8);
}
int32_t tr_jet_plan_equal(const tr_jet_plan *a, const tr_jet_plan *b) {
    if (!plan_valid(a) || !plan_valid(b)) return TR_MATH_INVALID;
    return a == b || same_key(a->key, a->key_words, b->key, b->key_words);
}

/* ================================================================== aggregates */
struct tr_jet_builder_s {
    uint32_t magic;
    int32_t spent;
    const tr_jet_plan *plan;
    const tr_scope *scope;
    tr_jet_view pv;
    tr_scope_view sv;
    uint64_t ngroups, support, ncovered;
    uint64_t *rows, *count, *covered;
    tr_math_i128 *sum;
    int64_t *ws, *out;
};
struct tr_jet_agg_s {
    uint32_t magic;
    int32_t order, directions, scoped;
    uint64_t ngroups, support, rows, count, present, bytes, key_words;
    int64_t denominator;
    uint64_t digest[2], scope_digest[2];
    uint64_t *grows, *gcount;
    tr_math_i128 *sum;
    uint8_t *exps;              /* S x m */
    int64_t *fact;              /* S */
    int64_t *key;
};
static int builder_valid(const tr_jet_builder *b) { return b && b->magic == BUILDER_MAGIC; }
static int agg_valid(const tr_jet_agg *a) { return a && a->magic == AGG_MAGIC; }

/* the scope's columns are the circuit's data columns with the plan's scales */
static int32_t scope_matches(const tr_jet_plan *p, const tr_scope_view *v) {
    if (v->ncols != p->ncols) return TR_MATH_DOMAIN_ERROR;
    for (int32_t c = 0; c < v->ncols; c++) if (v->col[c].scale != p->scales[c]) return TR_MATH_DOMAIN_ERROR;
    if (v->selected >= TR_JET_MAX_ROWS) return TR_MATH_UNSUPPORTED;
    return TR_MATH_OK;
}
static uint64_t state_bytes(uint64_t G, uint64_t S) {
    return tr_budget_add(tr_budget_mul(G, 16), tr_budget_mul(tr_budget_mul(G, S), 16));
}
/* an aggregate of G groups: the object, its states, its monomials, factorials and the plan's descriptor */
static uint64_t agg_bytes(const tr_jet_plan *p, uint64_t G) {
    uint64_t b = tr_budget_add(sizeof(tr_jet_agg), state_bytes(G, p->support));
    b = tr_budget_add(b, tr_budget_mul((uint64_t)p->support, (uint64_t)p->directions + 8));
    return tr_budget_add(b, tr_budget_mul(p->key_words, 8));
}
static void agg_need(const tr_jet_plan *p, uint64_t G, tr_math_need *need) {
    need_zero(need);
    need->need[TR_BUDGET_SUPPORT] = p->support;
    need->need[TR_BUDGET_STATES] = tr_budget_mul(G, p->support);
    need->need[TR_BUDGET_BYTES] = agg_bytes(p, G);
}
static void cpu_need(const tr_jet_plan *p, const tr_scope_view *v, tr_math_need *need) {
    agg_need(p, v->ngroups, need);
    uint64_t b = tr_budget_add(need->need[TR_BUDGET_BYTES], sizeof(tr_jet_builder));
    b = tr_budget_add(b, state_bytes(v->ngroups, p->support));
    b = tr_budget_add(b, tr_budget_bitmap_bytes(v->rows));
    b = tr_budget_add(b, tr_budget_mul(tr_budget_add(p->workspace, p->support), 8));
    need->need[TR_BUDGET_BYTES] = b;
}

int32_t tr_jet_agg_need(const tr_jet_plan *p, const tr_scope *scope, int32_t target, tr_math_need *need) {
    tr_scope_view v;
    if (!plan_valid(p) || !scope || !need || tr_scope_get_view(scope, &v)) return TR_MATH_INVALID;
    int32_t rc = scope_matches(p, &v);
    if (rc) return rc;
    if (target == TR_JET_NEED_CPU) cpu_need(p, &v, need);
    else if (target == TR_JET_NEED_ADOPT) agg_need(p, v.ngroups, need);
    else if (target == TR_JET_NEED_DEVICE) {
        tr_jet_view pv;
        tr_jet_plan_get_view(p, &pv);
        tr_jet_device_need(&pv, &v, need);
    } else return TR_MATH_INVALID;
    return TR_MATH_OK;
}

static void builder_release(tr_jet_builder *b) {
    if (!b) return;
    free(b->rows); free(b->count); free(b->covered); free(b->sum); free(b->ws); free(b->out);
    b->magic = 0;
    free(b);
}
void tr_jet_builder_free(tr_jet_builder *b) { if (builder_valid(b)) builder_release(b); }

tr_jet_builder *tr_jet_agg_init(const tr_jet_plan *p, const tr_scope *scope, const tr_math_budget *budget,
                                tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    tr_scope_view v;
    if (!plan_valid(p) || !scope || tr_scope_get_view(scope, &v)) { set_error(error, TR_MATH_INVALID); return NULL; }
    int32_t rc = scope_matches(p, &v);
    if (rc) { set_error(error, rc); return NULL; }
    tr_math_need need;
    tr_math_budget dflt;
    cpu_need(p, &v, &need);
    rc = tr_budget_check(budget_of(budget, &dflt), &need, refusal);
    if (rc) { set_error(error, rc); return NULL; }
    tr_jet_builder *b = (tr_jet_builder *)zalloc(sizeof *b);
    if (!b) { set_error(error, TR_MATH_MEMORY); return NULL; }
    b->plan = p; b->scope = scope; b->sv = v; b->ngroups = v.ngroups; b->support = p->support;
    tr_jet_plan_get_view(p, &b->pv);
    b->rows = (uint64_t *)zalloc(v.ngroups * 8);
    b->count = (uint64_t *)zalloc(v.ngroups * 8);
    b->sum = (tr_math_i128 *)zalloc(v.ngroups * p->support * 16);
    b->covered = (uint64_t *)zalloc(((v.rows + 63) / 64) * 8);
    b->ws = (int64_t *)zalloc(p->workspace * 8);
    b->out = (int64_t *)zalloc((uint64_t)p->support * 8);
    if (!b->rows || !b->count || !b->sum || !b->covered || !b->ws || !b->out) {
        builder_release(b); set_error(error, TR_MATH_MEMORY); return NULL;
    }
    b->magic = BUILDER_MAGIC;
    set_error(error, TR_MATH_OK);
    return b;
}

static uint64_t range_mask(uint64_t w, uint64_t begin, uint64_t end) {   /* the bits of word w inside [begin, end) */
    uint64_t lo = w * 64, a = begin > lo ? begin - lo : 0, z = end - lo < 64 ? end - lo : 64;
    uint64_t m = z >= 64 ? UINT64_MAX : (UINT64_C(1) << z) - 1;
    return a ? m & ~((UINT64_C(1) << a) - 1) : m;
}

/* the selected rows of [begin, end) */
static int32_t accumulate_range(tr_jet_builder *b, uint64_t begin, uint64_t end) {
    const tr_scope_view *v = &b->sv;
    const tr_jet_view *pv = &b->pv;
    const uint64_t S = b->support, Sout = pv->out_jet ? pv->upto[pv->out_pdeg] : 1;
    const uint32_t cols = pv->columns;
    int64_t data[TR_SCOPE_MAX_COLUMNS];
    for (uint64_t w = begin >> 6; w <= (end - 1) >> 6; w++) {
        uint64_t bits = v->selection[w] & range_mask(w, begin, end);
        while (bits) {
            uint64_t r = (w << 6) | (uint64_t)__builtin_ctzll(bits);
            bits &= bits - 1;
            uint32_t g = tr_scope_group(v, r);
            b->rows[g]++;
            int ok = 1;
            for (int32_t c = 0; c < v->ncols && ok; c++) {
                if (!((cols >> c) & 1)) continue;
                if (!tr_scope_present(v, c, r)) ok = 0;
                else data[c] = tr_scope_value(v, c, r);
            }
            if (!ok) continue;                                 /* f is NULL on this row */
            b->count[g]++;
            if (!tr_jet_eval(pv, data, b->ws, 1, b->out, 1)) return TR_MATH_OVERFLOW;
            tr_math_i128 *sm = b->sum + (uint64_t)g * S;
            for (uint64_t k = 0; k < Sout; k++)                    /* the output's degrees (zero past them) */
                if (b->out[k]) sm[k] = tr_i128_set(tr_i128_get(sm[k]) + b->out[k]);   /* < 2^63 rows: fits */
        }
    }
    return TR_MATH_OK;
}

int32_t tr_jet_agg_accumulate(tr_jet_builder *b, uint64_t begin, uint64_t end) {
    if (!builder_valid(b) || b->spent || begin > end || end > b->sv.rows) return TR_MATH_INVALID;
    if (begin == end) return TR_MATH_OK;
    for (uint64_t w = begin >> 6; w <= (end - 1) >> 6; w++)
        if (b->covered[w] & range_mask(w, begin, end)) return TR_MATH_INVALID;   /* a row is counted once */
    int32_t rc = accumulate_range(b, begin, end);
    if (rc) { b->spent = 1; return rc; }                       /* partial states: the builder is spent */
    for (uint64_t w = begin >> 6; w <= (end - 1) >> 6; w++) b->covered[w] |= range_mask(w, begin, end);
    b->ncovered += end - begin;
    return TR_MATH_OK;
}

int32_t tr_jet_agg_merge(tr_jet_builder *into, const tr_jet_builder *from) {
    if (!builder_valid(into) || !builder_valid(from) || into == from || into->spent || from->spent) return TR_MATH_INVALID;
    if (into->plan != from->plan && tr_jet_plan_equal(into->plan, from->plan) != 1) return TR_MATH_DOMAIN_ERROR;
    if (into->scope != from->scope && tr_scope_same(into->scope, from->scope) != 1) return TR_MATH_DOMAIN_ERROR;
    const uint64_t nw = (into->sv.rows + 63) / 64, G = into->ngroups, S = into->support;
    for (uint64_t w = 0; w < nw; w++) if (into->covered[w] & from->covered[w]) return TR_MATH_INVALID;
    for (uint64_t g = 0; g < G; g++) { into->rows[g] += from->rows[g]; into->count[g] += from->count[g]; }
    for (uint64_t i = 0; i < G * S; i++)
        into->sum[i] = tr_i128_set(tr_i128_get(into->sum[i]) + tr_i128_get(from->sum[i]));   /* disjoint rows: fits */
    for (uint64_t w = 0; w < nw; w++) into->covered[w] |= from->covered[w];
    into->ncovered += from->ncovered;
    return TR_MATH_OK;
}

static void agg_release(tr_jet_agg *a) {
    if (!a) return;
    free(a->grows); free(a->gcount); free(a->sum); free(a->exps); free(a->fact); free(a->key);
    a->magic = 0;
    free(a);
}
void tr_jet_agg_free(tr_jet_agg *a) { if (agg_valid(a)) agg_release(a); }

/* an empty aggregate shell of plan p (its tables copied; states allocated when with_states) */
static tr_jet_agg *agg_alloc(const tr_jet_plan *p, uint64_t G, int with_states) {
    tr_jet_agg *a = (tr_jet_agg *)zalloc(sizeof *a);
    if (!a) return NULL;
    a->order = p->order; a->directions = p->directions; a->ngroups = G; a->support = p->support;
    a->denominator = p->denominator; a->digest[0] = p->digest[0]; a->digest[1] = p->digest[1];
    a->key_words = p->key_words;
    a->exps = (uint8_t *)zalloc((uint64_t)p->support * (uint64_t)p->directions);
    a->fact = (int64_t *)zalloc((uint64_t)p->support * 8);
    a->key = (int64_t *)zalloc(p->key_words * 8);
    if (with_states) {
        a->grows = (uint64_t *)zalloc(G * 8); a->gcount = (uint64_t *)zalloc(G * 8);
        a->sum = (tr_math_i128 *)zalloc(G * p->support * 16);
    }
    if (!a->exps || !a->fact || !a->key || (with_states && (!a->grows || !a->gcount || !a->sum))) { agg_release(a); return NULL; }
    memcpy(a->exps, p->exps, (size_t)p->support * (size_t)p->directions);
    memcpy(a->fact, p->fact, (size_t)p->support * 8);
    memcpy(a->key, p->key, (size_t)p->key_words * 8);
    a->bytes = agg_bytes(p, G);
    a->magic = AGG_MAGIC;
    return a;
}
static void agg_tally(tr_jet_agg *a) {
    uint64_t rows = 0, count = 0, present = 0;
    for (uint64_t g = 0; g < a->ngroups; g++) { rows += a->grows[g]; count += a->gcount[g]; present += a->grows[g] != 0; }
    a->rows = rows; a->count = count; a->present = present;
}

tr_jet_agg *tr_jet_agg_finish(tr_jet_builder *b, int32_t *error) {
    if (!builder_valid(b) || b->spent || b->ncovered != b->sv.rows) { set_error(error, TR_MATH_INVALID); return NULL; }
    tr_jet_agg *a = agg_alloc(b->plan, b->ngroups, 0);
    if (!a) { set_error(error, TR_MATH_MEMORY); return NULL; }
    a->grows = b->rows; a->gcount = b->count; a->sum = b->sum;
    b->rows = NULL; b->count = NULL; b->sum = NULL;
    b->spent = 1;
    tr_scope_info info;
    tr_scope_get_info(b->scope, &info);
    a->scoped = 1; a->scope_digest[0] = info.digest[0]; a->scope_digest[1] = info.digest[1];
    agg_tally(a);
    set_error(error, TR_MATH_OK);
    return a;
}

tr_jet_agg *tr_jet_agg_build(const tr_jet_plan *p, const tr_scope *scope, const tr_math_budget *budget,
                             tr_math_refusal *refusal, int32_t *error) {
    tr_jet_builder *b = tr_jet_agg_init(p, scope, budget, refusal, error);
    if (!b) return NULL;
    int32_t rc = tr_jet_agg_accumulate(b, 0, b->sv.rows);
    tr_jet_agg *a = rc ? NULL : tr_jet_agg_finish(b, error);
    if (rc) set_error(error, rc);
    builder_release(b);
    return a;
}

tr_jet_agg *tr_jet_agg_from_states(const tr_jet_plan *p, const tr_scope *scope, const tr_jet_states *st,
                                   const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    tr_scope_view v;
    if (!plan_valid(p) || !scope || !st || tr_scope_get_view(scope, &v) || st->ngroups != v.ngroups ||
        st->support != p->support || !st->rows || !st->count || !st->sum) { set_error(error, TR_MATH_INVALID); return NULL; }
    int32_t rc = scope_matches(p, &v);
    if (rc) { set_error(error, rc); return NULL; }
    tr_math_need need;
    tr_math_budget dflt;
    agg_need(p, v.ngroups, &need);
    rc = tr_budget_check(budget_of(budget, &dflt), &need, refusal);
    if (rc) { set_error(error, rc); return NULL; }
    tr_jet_view pv;
    tr_jet_plan_get_view(p, &pv);
    const uint64_t S = p->support;
    uint64_t total = 0;
    for (uint64_t g = 0; g < v.ngroups; g++) {
        uint64_t rows = st->rows[g], cnt = st->count[g];
        if (cnt > rows || __builtin_add_overflow(total, rows, &total)) { set_error(error, TR_MATH_INVALID); return NULL; }
        for (uint64_t k = 0; k < S; k++) {
            i128 x = tr_i128_get(st->sum[g * S + k]);
            u128 mag = x < 0 ? (u128)0 - (u128)x : (u128)x;
            int outside = p->out_jet ? !tr_jet_in(&pv, (uint32_t)k, p->out_dep, 0, p->out_pdeg) : k > 0;
            if ((!cnt && x) || mag > ((u128)cnt << 63) || (outside && x)) { set_error(error, TR_MATH_INVALID); return NULL; }
        }
    }
    if (total != v.selected) { set_error(error, TR_MATH_INVALID); return NULL; }
    tr_jet_agg *a = agg_alloc(p, v.ngroups, 1);
    if (!a) { set_error(error, TR_MATH_MEMORY); return NULL; }
    memcpy(a->grows, st->rows, (size_t)v.ngroups * 8);
    memcpy(a->gcount, st->count, (size_t)v.ngroups * 8);
    memcpy(a->sum, st->sum, (size_t)(v.ngroups * S) * 16);
    tr_scope_info info;
    tr_scope_get_info(scope, &info);
    a->scoped = 1; a->scope_digest[0] = info.digest[0]; a->scope_digest[1] = info.digest[1];
    agg_tally(a);
    set_error(error, TR_MATH_OK);
    return a;
}

tr_jet_agg *tr_jet_agg_add(const tr_jet_agg *x, const tr_jet_agg *y, const tr_math_budget *budget,
                           tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    if (!agg_valid(x) || !agg_valid(y)) { set_error(error, TR_MATH_INVALID); return NULL; }
    if (x->ngroups != y->ngroups || !same_key(x->key, x->key_words, y->key, y->key_words)) {
        set_error(error, TR_MATH_DOMAIN_ERROR); return NULL;
    }
    if (x->count + y->count >= TR_JET_MAX_ROWS || x->rows + y->rows >= TR_JET_MAX_ROWS ||
        x->rows + y->rows < x->rows) { set_error(error, TR_MATH_UNSUPPORTED); return NULL; }
    tr_math_need need;
    tr_math_budget dflt;
    need_zero(&need);
    need.need[TR_BUDGET_SUPPORT] = x->support;
    need.need[TR_BUDGET_STATES] = tr_budget_mul(x->ngroups, x->support);
    need.need[TR_BUDGET_BYTES] = x->bytes;
    int32_t rc = tr_budget_check(budget_of(budget, &dflt), &need, refusal);
    if (rc) { set_error(error, rc); return NULL; }
    tr_jet_agg *a = (tr_jet_agg *)zalloc(sizeof *a);
    if (!a) { set_error(error, TR_MATH_MEMORY); return NULL; }
    *a = *x;
    a->magic = 0; a->scoped = 0; a->scope_digest[0] = a->scope_digest[1] = 0;
    a->grows = (uint64_t *)zalloc(x->ngroups * 8); a->gcount = (uint64_t *)zalloc(x->ngroups * 8);
    a->sum = (tr_math_i128 *)zalloc(x->ngroups * x->support * 16);
    a->exps = (uint8_t *)zalloc(x->support * (uint64_t)x->directions);
    a->fact = (int64_t *)zalloc(x->support * 8);
    a->key = (int64_t *)zalloc(x->key_words * 8);
    if (!a->grows || !a->gcount || !a->sum || !a->exps || !a->fact || !a->key) { agg_release(a); set_error(error, TR_MATH_MEMORY); return NULL; }
    memcpy(a->exps, x->exps, (size_t)(x->support * (uint64_t)x->directions));
    memcpy(a->fact, x->fact, (size_t)x->support * 8);
    memcpy(a->key, x->key, (size_t)x->key_words * 8);
    for (uint64_t g = 0; g < x->ngroups; g++) { a->grows[g] = x->grows[g] + y->grows[g]; a->gcount[g] = x->gcount[g] + y->gcount[g]; }
    for (uint64_t i = 0; i < x->ngroups * x->support; i++)
        a->sum[i] = tr_i128_set(tr_i128_get(x->sum[i]) + tr_i128_get(y->sum[i]));   /* |sum| <= count 2^63 < 2^126 */
    a->magic = AGG_MAGIC;
    agg_tally(a);
    set_error(error, TR_MATH_OK);
    return a;
}

int32_t tr_jet_agg_get_info(const tr_jet_agg *a, tr_jet_agg_info *out) {
    if (!agg_valid(a) || !out) return TR_MATH_INVALID;
    memset(out, 0, sizeof *out);
    out->ngroups = a->ngroups; out->present = a->present; out->rows = a->rows; out->count = a->count;
    out->support = a->support; out->bytes = a->bytes;
    out->digest[0] = a->digest[0]; out->digest[1] = a->digest[1];
    out->scope_digest[0] = a->scope_digest[0]; out->scope_digest[1] = a->scope_digest[1];
    out->denominator = a->denominator; out->order = a->order; out->directions = a->directions; out->scoped = a->scoped;
    return TR_MATH_OK;
}
int32_t tr_jet_agg_get_states(const tr_jet_agg *a, tr_jet_states *out) {
    if (!agg_valid(a) || !out) return TR_MATH_INVALID;
    memset(out, 0, sizeof *out);
    out->ngroups = a->ngroups; out->support = a->support;
    out->rows = a->grows; out->count = a->gcount; out->sum = a->sum;
    return TR_MATH_OK;
}
int32_t tr_jet_agg_monomials(const tr_jet_agg *a, uint8_t *exps, uint64_t capacity) {
    if (!agg_valid(a)) return TR_MATH_INVALID;
    uint64_t n = a->support * (uint64_t)a->directions;
    if (n && (!exps || capacity < n)) return TR_MATH_INVALID;
    if (n) memcpy(exps, a->exps, (size_t)n);
    return TR_MATH_OK;
}
int32_t tr_jet_agg_equal(const tr_jet_agg *a, const tr_jet_agg *b) {
    if (!agg_valid(a) || !agg_valid(b)) return TR_MATH_INVALID;
    if (a->ngroups != b->ngroups || !same_key(a->key, a->key_words, b->key, b->key_words)) return 0;
    return !memcmp(a->grows, b->grows, (size_t)a->ngroups * 8) && !memcmp(a->gcount, b->gcount, (size_t)a->ngroups * 8) &&
           !memcmp(a->sum, b->sum, (size_t)(a->ngroups * a->support) * 16);
}

/* (x / D) in lowest terms; with fact > 1 multiplied by fact (TR_MATH_OVERFLOW past int128) */
static int32_t reduce(i128 x, int64_t D, int64_t fact, i128 *num, int64_t *den) {
    u128 mag = x < 0 ? (u128)0 - (u128)x : (u128)x;
    uint64_t g = tr_gcd_u64((uint64_t)D, (uint64_t)(mag % (u128)(uint64_t)D));
    i128 n = x / (i128)g;
    int64_t d = D / (int64_t)g;
    if (fact > 1) {
        int64_t h = (int64_t)tr_gcd_u64((uint64_t)fact, (uint64_t)d);
        if (__builtin_mul_overflow(n, (i128)(fact / h), &n)) return TR_MATH_OVERFLOW;
        d /= h;
    }
    *num = n; *den = d;
    return TR_MATH_OK;
}
int32_t tr_jet_agg_read(const tr_jet_agg *a, uint64_t first, uint64_t count, int32_t convention, tr_math_i128 *num,
                        int64_t *den) {
    if (!agg_valid(a) || first > a->ngroups || count > a->ngroups - first ||
        (convention != TR_JET_TAYLOR && convention != TR_JET_DERIVATIVE) || (count && (!num || !den))) return TR_MATH_INVALID;
    const uint64_t S = a->support;
    for (uint64_t i = 0; i < count * S; i++) {
        uint64_t k = i % S;
        i128 n;
        int32_t rc = reduce(tr_i128_get(a->sum[first * S + i]), a->denominator,
                            convention == TR_JET_DERIVATIVE ? a->fact[k] : 1, &n, &den[i]);
        if (rc) return rc;
        num[i] = tr_i128_set(n);
    }
    return TR_MATH_OK;
}

/* ================================================================== values in a carrier */
struct tr_jet_values_s {
    uint32_t magic;
    int32_t carrier, convention, order, directions;
    uint64_t njets, support, bytes, key_words;
    tr_gfp field;
    int64_t *key;
    tr_num_rational *q;         /* Q64 */
    tr_bigq **big;              /* BIGQ (owned) */
    uint64_t *res;              /* GFP */
    tr_math_interval *iv;       /* INTERVAL */
};
static int values_valid(const tr_jet_values *v) { return v && v->magic == VALUES_MAGIC; }
static int value_carrier(int32_t c) {
    return c == TR_CARRIER_Q64 || c == TR_CARRIER_BIGQ || c == TR_CARRIER_GFP || c == TR_CARRIER_INTERVAL;
}
static uint64_t component_bytes(int32_t carrier) {
    return carrier == TR_CARRIER_GFP || carrier == TR_CARRIER_BIGQ ? 8 : 16;
}
static uint64_t values_bytes(int32_t carrier, uint64_t n, uint64_t key_words) {
    return tr_budget_add(tr_budget_add(sizeof(tr_jet_values), tr_budget_mul(n, component_bytes(carrier))),
                         tr_budget_mul(key_words, 8));
}
static void values_release(tr_jet_values *v) {
    if (!v) return;
    if (v->big) for (uint64_t i = 0; i < v->njets * v->support; i++) tr_bigq_free(v->big[i]);
    free(v->big); free(v->q); free(v->res); free(v->iv); free(v->key);
    v->magic = 0;
    free(v);
}
void tr_jet_values_free(tr_jet_values *v) { if (values_valid(v)) values_release(v); }

/* a values shell: the budget (output = components, bytes) checked, the components' array allocated */
static tr_jet_values *values_alloc(int32_t carrier, int32_t convention, int32_t order, int32_t m, uint64_t njets,
                                   uint64_t S, const int64_t *key, uint64_t kw, const tr_math_budget *budget,
                                   tr_math_refusal *refusal, int32_t *error) {
    tr_math_need need;
    tr_math_budget dflt;
    need_zero(&need);
    uint64_t n = tr_budget_mul(njets, S);
    need.need[TR_BUDGET_OUTPUT] = n;
    need.need[TR_BUDGET_BYTES] = values_bytes(carrier, n, kw);
    int32_t rc = tr_budget_check(budget_of(budget, &dflt), &need, refusal);
    if (rc) { set_error(error, rc); return NULL; }
    tr_jet_values *v = (tr_jet_values *)zalloc(sizeof *v);
    if (!v) { set_error(error, TR_MATH_MEMORY); return NULL; }
    v->carrier = carrier; v->convention = convention; v->order = order; v->directions = m;
    v->njets = njets; v->support = S; v->key_words = kw; v->bytes = need.need[TR_BUDGET_BYTES];
    v->key = (int64_t *)zalloc(kw * 8);
    void *arr = NULL;
    switch (carrier) {
    case TR_CARRIER_Q64: arr = v->q = (tr_num_rational *)zalloc(n * 16); break;
    case TR_CARRIER_BIGQ: arr = v->big = (tr_bigq **)zalloc(n * 8); break;
    case TR_CARRIER_GFP: arr = v->res = (uint64_t *)zalloc(n * 8); break;
    default: arr = v->iv = (tr_math_interval *)zalloc(n * 16); break;
    }
    if (!v->key || !arr) { values_release(v); set_error(error, TR_MATH_MEMORY); return NULL; }
    memcpy(v->key, key, (size_t)kw * 8);
    return v;
}
static int32_t field_for(int32_t carrier, uint64_t p, tr_gfp *f) {
    if (carrier != TR_CARRIER_GFP) { memset(f, 0, sizeof *f); return TR_MATH_OK; }
    return tr_gfp_field(p, f);
}

tr_jet_values *tr_jet_values_from_agg(const tr_jet_agg *a, int32_t carrier, int32_t convention, uint64_t p,
                                      const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    if (!agg_valid(a) || (convention != TR_JET_TAYLOR && convention != TR_JET_DERIVATIVE) ||
        carrier < 1 || carrier > TR_CARRIER_LAST) { set_error(error, TR_MATH_INVALID); return NULL; }
    if (!value_carrier(carrier)) { set_error(error, TR_MATH_CARRIER_ERROR); return NULL; }
    tr_gfp f;
    int32_t rc = field_for(carrier, p, &f);
    if (rc) { set_error(error, rc); return NULL; }
    tr_jet_values *v = values_alloc(carrier, convention, a->order, a->directions, a->ngroups, a->support, a->key,
                                    a->key_words, budget, refusal, error);
    if (!v) return NULL;
    v->field = f;
    const uint64_t S = a->support, n = a->ngroups * S;
    uint64_t dinv = 0;
    if (carrier == TR_CARRIER_GFP) {
        uint64_t dr = tr_gfp_reduce_i64_hd(&f, a->denominator);
        if (!dr) rc = TR_MATH_NO_SOLUTION;
        else dinv = tr_gfp_inv_hd(&f, dr);
    }
    for (uint64_t i = 0; i < n && !rc; i++) {
        const int64_t fact = convention == TR_JET_DERIVATIVE ? a->fact[i % S] : 1;
        const i128 x = tr_i128_get(a->sum[i]);
        switch (carrier) {
        case TR_CARRIER_GFP: {
            tr_gfp_elem e;
            if ((rc = tr_gfp_from_i128(&f, a->sum[i], &e))) break;
            v->res[i] = tr_gfp_mul_hd(&f, tr_gfp_mul_hd(&f, e.v, dinv), tr_gfp_reduce_i64_hd(&f, fact));
            break;
        }
        case TR_CARRIER_BIGQ: {
            int32_t err = 0;
            tr_bigint *nb = tr_bigint_from_i128(a->sum[i], budget, refusal, &err), *fb = NULL, *pb = NULL, *db = NULL;
            if (nb && fact > 1) { fb = tr_bigint_from_i64(fact, budget, refusal, &err); if (fb) pb = tr_bigint_mul(nb, fb, budget, refusal, &err); }
            if (nb && (fact <= 1 || pb)) db = tr_bigint_from_i64(a->denominator, budget, refusal, &err);
            if (db) v->big[i] = tr_bigq_from_bigints(fact > 1 ? pb : nb, db, budget, refusal, &err);
            if (!v->big[i]) rc = err ? err : TR_MATH_MEMORY;
            tr_bigint_free(nb); tr_bigint_free(fb); tr_bigint_free(pb); tr_bigint_free(db);
            break;
        }
        default: {
            i128 num;
            int64_t den;
            rc = reduce(x, a->denominator, fact, &num, &den);
            if (carrier == TR_CARRIER_Q64) {
                if (!rc && (num > INT64_MAX || num < INT64_MIN)) rc = TR_MATH_OVERFLOW;
                if (!rc) { v->q[i].numerator = (int64_t)num; v->q[i].denominator = den; }
                break;
            }
            /* INTERVAL: the tight enclosure, through a big rational when the numerator is wider than int64 */
            if (rc == TR_MATH_OVERFLOW || (!rc && (num > INT64_MAX || num < INT64_MIN))) {
                int32_t err = 0;
                tr_bigint *nb = tr_bigint_from_i128(a->sum[i], budget, refusal, &err), *fb = NULL, *pb = NULL, *db = NULL;
                tr_bigq *q = NULL;
                if (nb) { fb = tr_bigint_from_i64(fact, budget, refusal, &err); if (fb) pb = tr_bigint_mul(nb, fb, budget, refusal, &err); }
                if (pb) db = tr_bigint_from_i64(a->denominator, budget, refusal, &err);
                if (db) q = tr_bigq_from_bigints(pb, db, budget, refusal, &err);
                rc = q ? tr_bigq_enclose(q, &v->iv[i]) : (err ? err : TR_MATH_MEMORY);
                tr_bigint_free(nb); tr_bigint_free(fb); tr_bigint_free(pb); tr_bigint_free(db); tr_bigq_free(q);
            } else if (!rc) rc = tr_carrier_enclose_q64((tr_num_rational){(int64_t)num, den}, &v->iv[i]);
            break;
        }
        }
    }
    if (rc) { values_release(v); set_error(error, rc); return NULL; }
    v->magic = VALUES_MAGIC;
    set_error(error, TR_MATH_OK);
    return v;
}

tr_jet_values *tr_jet_values_convert(const tr_jet_values *s, int32_t carrier, uint64_t p, const tr_math_budget *budget,
                                     tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    if (!values_valid(s) || carrier < 1 || carrier > TR_CARRIER_LAST) { set_error(error, TR_MATH_INVALID); return NULL; }
    int32_t kind = tr_carrier_conversion(s->carrier, carrier);
    if (!value_carrier(carrier) || kind == TR_CONVERT_FORBIDDEN) { set_error(error, TR_MATH_CARRIER_ERROR); return NULL; }
    tr_gfp f;
    int32_t rc = field_for(carrier, p, &f);
    if (!rc && carrier == TR_CARRIER_GFP && s->carrier == TR_CARRIER_GFP && f.p != s->field.p) rc = TR_MATH_CARRIER_ERROR;
    if (rc) { set_error(error, rc); return NULL; }
    tr_jet_values *v = values_alloc(carrier, s->convention, s->order, s->directions, s->njets, s->support, s->key,
                                    s->key_words, budget, refusal, error);
    if (!v) return NULL;
    v->field = f;
    const uint64_t n = s->njets * s->support;
    for (uint64_t i = 0; i < n && !rc; i++) {
        int32_t err = 0;
        tr_gfp_elem e;
        switch (s->carrier * 16 + carrier) {
        case TR_CARRIER_Q64 * 16 + TR_CARRIER_Q64: v->q[i] = s->q[i]; break;
        case TR_CARRIER_Q64 * 16 + TR_CARRIER_BIGQ:
            if (!(v->big[i] = tr_bigq_from_q64(s->q[i], budget, refusal, &err))) rc = err ? err : TR_MATH_MEMORY;
            break;
        case TR_CARRIER_Q64 * 16 + TR_CARRIER_GFP:
            if (!(rc = tr_gfp_from_q64(&f, s->q[i], &e))) v->res[i] = e.v;
            break;
        case TR_CARRIER_Q64 * 16 + TR_CARRIER_INTERVAL: rc = tr_carrier_enclose_q64(s->q[i], &v->iv[i]); break;
        case TR_CARRIER_BIGQ * 16 + TR_CARRIER_Q64: rc = tr_bigq_to_q64(s->big[i], &v->q[i]); break;
        case TR_CARRIER_BIGQ * 16 + TR_CARRIER_BIGQ:
            if (!(v->big[i] = tr_bigq_copy(s->big[i], budget, refusal, &err))) rc = err ? err : TR_MATH_MEMORY;
            break;
        case TR_CARRIER_BIGQ * 16 + TR_CARRIER_GFP:
            if (!(rc = tr_gfp_from_bigq(&f, s->big[i], &e))) v->res[i] = e.v;
            break;
        case TR_CARRIER_BIGQ * 16 + TR_CARRIER_INTERVAL: rc = tr_bigq_enclose(s->big[i], &v->iv[i]); break;
        case TR_CARRIER_GFP * 16 + TR_CARRIER_GFP: v->res[i] = s->res[i]; break;
        case TR_CARRIER_INTERVAL * 16 + TR_CARRIER_INTERVAL: v->iv[i] = s->iv[i]; break;
        default: rc = TR_MATH_CARRIER_ERROR; break;
        }
    }
    if (rc) { values_release(v); set_error(error, rc); return NULL; }
    v->magic = VALUES_MAGIC;
    set_error(error, TR_MATH_OK);
    return v;
}

tr_jet_values *tr_jet_values_add(const tr_jet_values *x, const tr_jet_values *y, const tr_math_budget *budget,
                                 tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    if (!values_valid(x) || !values_valid(y)) { set_error(error, TR_MATH_INVALID); return NULL; }
    if (x->carrier != y->carrier || (x->carrier == TR_CARRIER_GFP && x->field.p != y->field.p)) {
        set_error(error, TR_MATH_CARRIER_ERROR); return NULL;
    }
    if (x->convention != y->convention || x->njets != y->njets || x->support != y->support ||
        !same_key(x->key, x->key_words, y->key, y->key_words)) { set_error(error, TR_MATH_DOMAIN_ERROR); return NULL; }
    tr_jet_values *v = values_alloc(x->carrier, x->convention, x->order, x->directions, x->njets, x->support, x->key,
                                    x->key_words, budget, refusal, error);
    if (!v) return NULL;
    v->field = x->field;
    const uint64_t n = x->njets * x->support;
    int32_t rc = TR_MATH_OK;
    for (uint64_t i = 0; i < n && !rc; i++) {
        int32_t err = 0;
        switch (x->carrier) {
        case TR_CARRIER_Q64: if (!tr_q_add(x->q[i], y->q[i], &v->q[i])) rc = TR_MATH_OVERFLOW; break;
        case TR_CARRIER_BIGQ:
            if (!(v->big[i] = tr_bigq_add(x->big[i], y->big[i], budget, refusal, &err))) rc = err ? err : TR_MATH_MEMORY;
            break;
        case TR_CARRIER_GFP: v->res[i] = tr_gfp_add_hd(&x->field, x->res[i], y->res[i]); break;
        default: v->iv[i] = tr_interval_add(x->iv[i], y->iv[i]); break;
        }
    }
    if (rc) { values_release(v); set_error(error, rc); return NULL; }
    v->magic = VALUES_MAGIC;
    set_error(error, TR_MATH_OK);
    return v;
}

int32_t tr_jet_values_get_info(const tr_jet_values *v, tr_jet_values_info *out) {
    if (!values_valid(v) || !out) return TR_MATH_INVALID;
    memset(out, 0, sizeof *out);
    out->njets = v->njets; out->support = v->support; out->bytes = v->bytes;
    out->p = v->carrier == TR_CARRIER_GFP ? v->field.p : 0;
    out->carrier = v->carrier; out->convention = v->convention; out->order = v->order; out->directions = v->directions;
    out->exact = v->carrier != TR_CARRIER_INTERVAL;
    return TR_MATH_OK;
}
static int32_t read_check(const tr_jet_values *v, int32_t carrier, const void *out, uint64_t capacity) {
    if (!values_valid(v)) return TR_MATH_INVALID;
    if (v->carrier != carrier) return TR_MATH_CARRIER_ERROR;
    uint64_t n = v->njets * v->support;
    return n && (!out || capacity < n) ? TR_MATH_INVALID : TR_MATH_OK;
}
int32_t tr_jet_values_read_q64(const tr_jet_values *v, tr_num_rational *out, uint64_t capacity) {
    int32_t rc = read_check(v, TR_CARRIER_Q64, out, capacity);
    if (!rc && v->njets && v->support) memcpy(out, v->q, (size_t)(v->njets * v->support) * sizeof *out);
    return rc;
}
int32_t tr_jet_values_read_gfp(const tr_jet_values *v, uint64_t *out, uint64_t capacity) {
    int32_t rc = read_check(v, TR_CARRIER_GFP, out, capacity);
    if (!rc && v->njets && v->support) memcpy(out, v->res, (size_t)(v->njets * v->support) * 8);
    return rc;
}
int32_t tr_jet_values_read_interval(const tr_jet_values *v, tr_math_interval *out, uint64_t capacity) {
    int32_t rc = read_check(v, TR_CARRIER_INTERVAL, out, capacity);
    if (!rc && v->njets && v->support) memcpy(out, v->iv, (size_t)(v->njets * v->support) * sizeof *out);
    return rc;
}
const tr_bigq *tr_jet_values_bigq(const tr_jet_values *v, uint64_t k) {
    if (!values_valid(v) || v->carrier != TR_CARRIER_BIGQ || k >= v->njets * v->support) return NULL;
    return v->big[k];
}
int32_t tr_jet_values_equal(const tr_jet_values *a, const tr_jet_values *b) {
    if (!values_valid(a) || !values_valid(b)) return TR_MATH_INVALID;
    if (a->carrier != b->carrier || a->convention != b->convention || a->njets != b->njets || a->support != b->support ||
        (a->carrier == TR_CARRIER_GFP && a->field.p != b->field.p) || !same_key(a->key, a->key_words, b->key, b->key_words))
        return 0;
    const uint64_t n = a->njets * a->support;
    for (uint64_t i = 0; i < n; i++) {
        switch (a->carrier) {
        case TR_CARRIER_Q64:
            if (a->q[i].numerator != b->q[i].numerator || a->q[i].denominator != b->q[i].denominator) return 0;
            break;
        case TR_CARRIER_BIGQ: if (tr_bigq_cmp(a->big[i], b->big[i]) != 0) return 0; break;
        case TR_CARRIER_GFP: if (a->res[i] != b->res[i]) return 0; break;
        default: if (a->iv[i].lo != b->iv[i].lo || a->iv[i].hi != b->iv[i].hi) return 0; break;
        }
    }
    return 1;
}
