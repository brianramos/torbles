/* ATTR-001: sparse interaction and contribution attribution (games, Moebius supports, exact indices, bounded
 * exploration and minimum explanation / repair search); see math_attr.h.
 * SPDX-License-Identifier: Apache-2.0 */
#include "math_attr.h"
#include "mathematics.h"
#include "math_domain.h"
#include "math_handles.h"
#include <stdlib.h>
#include <string.h>

TR_MATH_ABI(tr_attr_q, TR_ATTR_ABI);                 /* the ctypes boundary of ATTR-001 (math_handles.h) */
TR_MATH_ABI(tr_attr_identity, TR_ATTR_ABI);
TR_MATH_ABI(tr_attr_game_view, TR_ATTR_ABI);
TR_MATH_ABI(tr_attr_game_info, TR_ATTR_ABI);
TR_MATH_ABI(tr_attr_support_info, TR_ATTR_ABI);
TR_MATH_ABI(tr_attr_table_info, TR_ATTR_ABI);
TR_MATH_ABI(tr_attr_exhaustive_info, TR_ATTR_ABI);
TR_MATH_ABI(tr_attr_explore_info, TR_ATTR_ABI);
TR_MATH_ABI(tr_attr_search_query, TR_ATTR_ABI);
TR_MATH_ABI(tr_attr_search_info, TR_ATTR_ABI);

_Static_assert(sizeof(tr_attr_q) == 32, "tr_attr_q ABI");
_Static_assert(sizeof(tr_attr_identity) == 112, "tr_attr_identity ABI");
_Static_assert(sizeof(tr_attr_game_view) == 200, "tr_attr_game_view ABI");
_Static_assert(sizeof(tr_attr_game_info) == 176, "tr_attr_game_info ABI");
_Static_assert(sizeof(tr_attr_support_info) == 120, "tr_attr_support_info ABI");
_Static_assert(sizeof(tr_attr_table_info) == 160, "tr_attr_table_info ABI");
_Static_assert(sizeof(tr_attr_exhaustive_info) == 160, "tr_attr_exhaustive_info ABI");
_Static_assert(sizeof(tr_attr_explore_info) == 240, "tr_attr_explore_info ABI");
_Static_assert(sizeof(tr_attr_search_query) == 48, "tr_attr_search_query ABI");
_Static_assert(sizeof(tr_attr_search_info) == 112, "tr_attr_search_info ABI");

#define GAME_MAGIC UINT32_C(0x4d414741)
#define SUPP_MAGIC UINT32_C(0x50505553)
#define TABL_MAGIC UINT32_C(0x4c424154)

struct tr_attr_game_s {
    uint32_t magic;
    tr_attr_game_view v;
    int32_t projection, signed_weights, players_used, reserved;
    uint64_t baseline, present, absent, bytes;
    tr_math_i128 *table;
    tr_prov_node *node;
    uint32_t *child, *output, *var_player;
    int64_t *weights;
};

static void set_error(int32_t *error, int32_t code) { if (error) *error = code; }
static void *zalloc(uint64_t bytes) {
    if (bytes >= (uint64_t)SIZE_MAX) return NULL;
    return calloc(1, bytes ? (size_t)bytes : 1);
}
static int game_ok(const tr_attr_game *g) { return g && g->magic == GAME_MAGIC; }
static int popc(uint64_t x) { return __builtin_popcountll(x); }
static const __int128 I128_MAX = (__int128)(((unsigned __int128)1 << 127) - 1);

/* ------------------------------------------------------------------ games */
static void game_release(tr_attr_game *g) {
    if (!g) return;
    free(g->table); free(g->node); free(g->child); free(g->output); free(g->var_player); free(g->weights);
    g->magic = 0;
    free(g);
}
void tr_attr_game_free(tr_attr_game *g) { if (game_ok(g)) game_release(g); }
static tr_attr_game *game_fail(tr_attr_game *g, int32_t *error, int32_t code) {
    game_release(g); set_error(error, code); return NULL;
}
static uint64_t table_bytes(int32_t n) { return tr_budget_add(sizeof(tr_attr_game), tr_budget_mul(tr_budget_power2((uint32_t)n), 16)); }
static uint64_t prov_bytes(const tr_prov_plan_view *p) {
    uint64_t b = tr_budget_add(sizeof(tr_attr_game), tr_budget_mul(p->nodes, sizeof(tr_prov_node)));
    b = tr_budget_add(b, tr_budget_mul(tr_budget_add(p->nchild, p->noutputs), 4));
    b = tr_budget_add(b, tr_budget_mul(p->noutputs, 8));
    return tr_budget_add(b, tr_budget_mul(p->nvars, 4));
}

/* the certified facts of a table game: extremes, v(empty), v(P), monotonicity on every edge of the lattice */
static void table_finish(tr_attr_game *g) {
    const int32_t n = g->v.n;
    const uint64_t size = (uint64_t)1 << n;
    __int128 lo = tr_i128_get(g->table[0]), hi = lo, bound = 0;
    int up = 1, down = 1;
    for (uint64_t S = 0; S < size; S++) {
        __int128 x = tr_i128_get(g->table[S]);
        if (x < lo) lo = x;
        if (x > hi) hi = x;
        for (int32_t i = 0; i < n && (up || down); i++) {
            if ((S >> i) & 1) continue;
            __int128 y = tr_i128_get(g->table[S | ((uint64_t)1 << i)]);
            if (y < x) up = 0;
            if (y > x) down = 0;
        }
    }
    __int128 alo = lo < 0 ? -lo : lo, ahi = hi < 0 ? -hi : hi;   /* |x| <= 2^127 - 1 for stored int128 inputs */
    bound = alo > ahi ? alo : ahi;
    g->v.lo = tr_i128_set(lo); g->v.hi = tr_i128_set(hi); g->v.bound = tr_i128_set(bound);
    g->v.empty = g->table[0]; g->v.full = g->table[size - 1];
    g->v.monotone = up ? 1 : down ? -1 : 0;
    g->v.sign = 0; g->v.degree = n;
    g->v.table = g->table;
    g->magic = GAME_MAGIC;
}
static tr_attr_game *table_alloc(int32_t n, int32_t source) {
    tr_attr_game *g = (tr_attr_game *)zalloc(sizeof *g);
    if (!g) return NULL;
    g->table = (tr_math_i128 *)zalloc(tr_budget_mul(tr_budget_power2((uint32_t)n), 16));
    if (!g->table) { free(g); return NULL; }
    g->v.kind = TR_ATTR_GAME_TABLE; g->v.n = n; g->v.source = source;
    g->bytes = table_bytes(n);
    return g;
}
static int32_t table_need(int32_t n, tr_math_need *need) {
    if (n < 1) return TR_MATH_INVALID;
    tr_attr_need_zero(need);
    need->need[TR_BUDGET_VARIABLES] = (uint64_t)n;
    need->need[TR_BUDGET_TENSOR] = tr_budget_power2((uint32_t)n);
    need->need[TR_BUDGET_BYTES] = table_bytes(n);
    return TR_MATH_OK;
}

tr_attr_game *tr_attr_game_table(int32_t n, const tr_math_i128 *values, const tr_math_budget *budget,
                                 tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    tr_math_need need;
    if (!values || table_need(n, &need)) return game_fail(NULL, error, TR_MATH_INVALID);
    int32_t rc = tr_attr_check(budget, &need, refusal);
    if (rc) return game_fail(NULL, error, rc);
    if (n > TR_ATTR_MAX_TABLE) return game_fail(NULL, error, TR_MATH_UNSUPPORTED);
    for (uint64_t S = 0; S < ((uint64_t)1 << n); S++)
        if (tr_i128_get(values[S]) == -I128_MAX - 1) return game_fail(NULL, error, TR_MATH_OVERFLOW);   /* |x| fits */
    tr_attr_game *g = table_alloc(n, TR_ATTR_SRC_VALUES);
    if (!g) return game_fail(NULL, error, TR_MATH_MEMORY);
    memcpy(g->table, values, (size_t)(((uint64_t)1 << n) * 16));
    table_finish(g);
    set_error(error, TR_MATH_OK);
    return g;
}

tr_attr_game *tr_attr_game_lattice(const tr_lattice_hist *h, int32_t projection, int32_t rule, const tr_math_budget *budget,
                                   tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    tr_lattice_hist_info hi;
    if (!h || tr_lattice_hist_get_info(h, &hi) || (rule != TR_ATTR_SRC_CONJUNCTION && rule != TR_ATTR_SRC_DISJUNCTION) ||
        (projection != TR_LATTICE_COMPLETE && projection != TR_LATTICE_TRUE_SETS))
        return game_fail(NULL, error, TR_MATH_INVALID);
    const int32_t k = hi.k;
    tr_math_need need;
    if (table_need(k, &need)) return game_fail(NULL, error, TR_MATH_INVALID);
    need.need[TR_BUDGET_ITERATIONS] = (uint64_t)k;
    need.need[TR_BUDGET_BYTES] = tr_budget_add(need.need[TR_BUDGET_BYTES], tr_budget_mul(tr_budget_power2((uint32_t)k), 32));
    int32_t rc = tr_attr_check(budget, &need, refusal);
    if (rc) return game_fail(NULL, error, rc);
    if (k > TR_ATTR_MAX_TABLE) return game_fail(NULL, error, TR_MATH_UNSUPPORTED);
    const uint64_t size = (uint64_t)1 << k, full = size - 1;
    tr_math_i128 *proj = (tr_math_i128 *)zalloc(size * 16), *z = (tr_math_i128 *)zalloc(size * 16);
    tr_attr_game *g = table_alloc(k, rule);
    if (!proj || !z || !g) { free(proj); free(z); return game_fail(g, error, TR_MATH_MEMORY); }
    rc = tr_lattice_hist_vector_i128(h, projection, proj, size);
    if (!rc) rc = tr_lattice_transform(k, rule == TR_ATTR_SRC_CONJUNCTION ? TR_LATTICE_ZETA_UP : TR_LATTICE_ZETA_DOWN,
                                       TR_LATTICE_I128, proj, z, budget, refusal);
    if (rc) { free(proj); free(z); return game_fail(g, error, rc); }
    if (rule == TR_ATTR_SRC_CONJUNCTION) memcpy(g->table, z, (size_t)(size * 16));
    else {
        const __int128 total = tr_i128_get(z[full]);                /* zeta-down at P: every point */
        for (uint64_t S = 0; S < size; S++) g->table[S] = tr_i128_set(total - tr_i128_get(z[full & ~S]));
    }
    free(proj); free(z);
    g->projection = projection;
    table_finish(g);
    set_error(error, TR_MATH_OK);
    return g;
}

tr_attr_game *tr_attr_game_predicate(const tr_bool_domain *d, const tr_bool_program_view *p, uint64_t baseline,
                                     const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    if (!d || !p) return game_fail(NULL, error, TR_MATH_INVALID);
    int32_t rc = tr_bool_domain_check(d);
    if (!rc) rc = tr_bool_program_check(p);
    if (rc) return game_fail(NULL, error, rc);
    if (p->k != d->k) return game_fail(NULL, error, TR_MATH_DOMAIN_ERROR);
    const int32_t k = p->k;
    if (k < 1 || (k < 64 && (baseline >> k))) return game_fail(NULL, error, TR_MATH_INVALID);
    tr_math_need need;
    if (table_need(k, &need)) return game_fail(NULL, error, TR_MATH_INVALID);
    need.need[TR_BUDGET_TENSOR] = tr_budget_mul(tr_budget_power2((uint32_t)k), d->nw ? d->nw : 1);
    need.need[TR_BUDGET_CIRCUIT] = (uint64_t)p->nsched;
    need.need[TR_BUDGET_BYTES] = tr_budget_add(need.need[TR_BUDGET_BYTES], tr_budget_mul((uint64_t)p->nnodes, 16));
    rc = tr_attr_check(budget, &need, refusal);
    if (rc) return game_fail(NULL, error, rc);
    if (k > TR_ATTR_MAX_TABLE) return game_fail(NULL, error, TR_MATH_UNSUPPORTED);
    tr_attr_game *g = table_alloc(k, TR_ATTR_SRC_PREDICATE);
    uint64_t *val = (uint64_t *)zalloc(tr_budget_mul((uint64_t)p->nnodes, 16));
    if (!g || !val) { free(val); return game_fail(g, error, TR_MATH_MEMORY); }
    const uint64_t size = (uint64_t)1 << k, all = size - 1;
    for (uint64_t S = 0; S < size; S++) {
        const uint64_t absent = all & ~S;
        uint64_t t = 0;
        int32_t j = 0;
        for (uint64_t b = absent; b; b &= b - 1, j++)          /* bit j of t: the j-th absent condition's constant */
            if ((baseline >> __builtin_ctzll(b)) & 1) t |= (uint64_t)1 << j;
        uint64_t count = 0;
        for (uint64_t w = 0; w < d->nw; w++) {
            const uint64_t sel = tr_bool_selection_word(d, w);
            if (!sel) continue;
            tr_bool_eval_cone(p, d, p->schedule, (uint32_t)p->nsched, absent, t, 0, w, val);
            count += (uint64_t)popc(val[2 * (uint64_t)p->root] & sel);
        }
        g->table[S] = tr_i128_set((__int128)count);
    }
    free(val);
    g->baseline = baseline;
    table_finish(g);
    set_error(error, TR_MATH_OK);
    return g;
}

/* a host evaluation of every output with every player present (all = 1) or absent (all = 0): node values in
 * unsigned 128 bits (0, or TR_MATH_OVERFLOW when one leaves them) */
static int32_t prov_eval_all(const tr_attr_game *g, int all, unsigned __int128 *s, unsigned __int128 *out) {
    const tr_prov_plan_view *p = &g->v.plan;
    const unsigned __int128 top = ~(unsigned __int128)0;
    const int boolean = g->v.semiring == TR_PROV_BOOL;
    for (uint64_t i = 0; i < p->nodes; i++) {
        const tr_prov_node nd = p->node[i];
        unsigned __int128 x;
        if (nd.op == TR_PROV_ZERO) x = 0;
        else if (nd.op == TR_PROV_ONE) x = 1;
        else if (nd.op == TR_PROV_VAR) {
            uint32_t q = g->var_player[nd.arg];
            x = q == TR_ATTR_PRESENT ? 1 : q == TR_ATTR_ABSENT ? 0 : (unsigned __int128)(all != 0);
        } else if (nd.op == TR_PROV_PLUS) {
            x = 0;
            for (uint32_t c = 0; c < nd.n; c++) {
                unsigned __int128 y = s[p->child[nd.arg + c]];
                if (boolean) { if (y) { x = 1; break; } continue; }
                if (y > top - x) return TR_MATH_OVERFLOW;
                x += y;
            }
        } else {
            x = 1;
            for (uint32_t c = 0; c < nd.n && x; c++) {
                unsigned __int128 y = s[p->child[nd.arg + c]];
                if (boolean) { x = y ? 1 : 0; continue; }
                if (y && x > top / y) return TR_MATH_OVERFLOW;
                x *= y;
            }
        }
        s[i] = x;
    }
    for (uint64_t j = 0; j < p->noutputs; j++) out[j] = s[p->output[j]];
    return TR_MATH_OK;
}

int32_t tr_attr_game_need(int32_t kind, int32_t n, const tr_prov_plan_view *plan, tr_math_need *need) {
    if (!need) return TR_MATH_INVALID;
    if (kind == TR_ATTR_GAME_TABLE) return table_need(n, need);
    if (kind != TR_ATTR_GAME_PROV || !plan || n < 1) return TR_MATH_INVALID;
    tr_attr_need_zero(need);
    need->need[TR_BUDGET_VARIABLES] = (uint64_t)n;
    need->need[TR_BUDGET_CIRCUIT] = plan->nodes;
    need->need[TR_BUDGET_SUPPORT] = plan->nchild;
    need->need[TR_BUDGET_OUTPUT] = plan->noutputs;
    need->need[TR_BUDGET_BYTES] = tr_budget_add(prov_bytes(plan), tr_budget_mul(tr_budget_add(plan->nodes, plan->noutputs), 32));
    return TR_MATH_OK;
}

tr_attr_game *tr_attr_game_prov(const tr_prov_plan_view *plan, int32_t semiring, const int64_t *weights,
                                const uint32_t *var_player, int32_t n, const tr_math_budget *budget,
                                tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    if (!plan || tr_prov_plan_view_check(plan) || (semiring != TR_PROV_COUNT && semiring != TR_PROV_BOOL) || n < 1 ||
        (plan->noutputs && !weights) || (plan->nvars && !var_player)) return game_fail(NULL, error, TR_MATH_INVALID);
    for (uint64_t x = 0; x < plan->nvars; x++)
        if (var_player[x] < TR_ATTR_ABSENT && var_player[x] >= (uint32_t)n) return game_fail(NULL, error, TR_MATH_INVALID);
    tr_math_need need;
    tr_attr_game_need(TR_ATTR_GAME_PROV, n, plan, &need);
    int32_t rc = tr_attr_check(budget, &need, refusal);
    if (rc) return game_fail(NULL, error, rc);
    if ((uint32_t)n > TR_ATTR_MAX_PLAYERS) return game_fail(NULL, error, TR_MATH_UNSUPPORTED);
    tr_attr_game *g = (tr_attr_game *)zalloc(sizeof *g);
    if (!g) return game_fail(NULL, error, TR_MATH_MEMORY);
    g->node = (tr_prov_node *)zalloc(tr_budget_mul(plan->nodes, sizeof(tr_prov_node)));
    g->child = (uint32_t *)zalloc(tr_budget_mul(plan->nchild, 4));
    g->output = (uint32_t *)zalloc(tr_budget_mul(plan->noutputs, 4));
    g->weights = (int64_t *)zalloc(tr_budget_mul(plan->noutputs, 8));
    g->var_player = (uint32_t *)zalloc(tr_budget_mul(plan->nvars, 4));
    unsigned __int128 *s = (unsigned __int128 *)zalloc(tr_budget_mul(plan->nodes ? plan->nodes : 1, 16));
    unsigned __int128 *o0 = (unsigned __int128 *)zalloc(tr_budget_mul(plan->noutputs, 16));
    unsigned __int128 *o1 = (unsigned __int128 *)zalloc(tr_budget_mul(plan->noutputs, 16));
    uint8_t *seen = (uint8_t *)zalloc((uint64_t)n);
    uint32_t *deg = (uint32_t *)zalloc(tr_budget_mul(plan->nodes, 4));
    if (!g->node || !g->child || !g->output || !g->weights || !g->var_player || !s || !o0 || !o1 || !seen || !deg) {
        free(s); free(o0); free(o1); free(seen); free(deg);
        return game_fail(g, error, TR_MATH_MEMORY);
    }
    if (plan->nodes) memcpy(g->node, plan->node, (size_t)plan->nodes * sizeof(tr_prov_node));
    if (plan->nchild) memcpy(g->child, plan->child, (size_t)plan->nchild * 4);
    if (plan->noutputs) { memcpy(g->output, plan->output, (size_t)plan->noutputs * 4); memcpy(g->weights, weights, (size_t)plan->noutputs * 8); }
    if (plan->nvars) memcpy(g->var_player, var_player, (size_t)plan->nvars * 4);
    g->v.kind = TR_ATTR_GAME_PROV; g->v.semiring = semiring; g->v.n = n; g->v.source = TR_ATTR_SRC_PROVENANCE;
    g->v.nvars = plan->nvars;
    g->v.plan = *plan;
    g->v.plan.node = g->node; g->v.plan.child = g->child; g->v.plan.output = g->output;
    g->v.weights = g->weights; g->v.var_player = g->var_player;
    for (uint64_t x = 0; x < plan->nvars; x++) {
        if (var_player[x] == TR_ATTR_PRESENT) g->present++;
        else if (var_player[x] == TR_ATTR_ABSENT) g->absent++;
    }
    /* the degree bound (player variables per derivation) and the players the cone reaches */
    uint32_t used = 0, dmax = 0;
    for (uint64_t i = 0; i < plan->nodes; i++) {
        const tr_prov_node nd = plan->node[i];
        uint64_t dg = 0;
        if (nd.op == TR_PROV_VAR) {
            uint32_t q = var_player[nd.arg];
            if (q < TR_ATTR_ABSENT) { dg = 1; if (!seen[q]) { seen[q] = 1; used++; } }
        } else if (nd.op == TR_PROV_PLUS) {
            for (uint32_t c = 0; c < nd.n; c++) if (deg[plan->child[nd.arg + c]] > dg) dg = deg[plan->child[nd.arg + c]];
        } else if (nd.op == TR_PROV_TIMES) {
            for (uint32_t c = 0; c < nd.n; c++) dg += deg[plan->child[nd.arg + c]];
        }
        deg[i] = (uint32_t)(dg < (uint64_t)n ? dg : (uint64_t)n);
    }
    for (uint64_t j = 0; j < plan->noutputs; j++) if (deg[plan->output[j]] > dmax) dmax = deg[plan->output[j]];
    g->players_used = (int32_t)used;
    g->v.degree = semiring == TR_PROV_COUNT ? (int32_t)(dmax < used ? dmax : used) : (int32_t)used;
    /* v(empty), v(P), the certified range and bound (each annotation lies between its values at empty and P) */
    rc = prov_eval_all(g, 0, s, o0);
    if (!rc) rc = prov_eval_all(g, 1, s, o1);
    __int128 lo = 0, hi = 0, bound = 0, e = 0, f = 0;
    int neg = 0, pos = 0;
    for (uint64_t j = 0; j < plan->noutputs && !rc; j++) {
        const int64_t w = weights[j];
        if (w < 0) neg = 1;
        if (w > 0) pos = 1;
        if (!w) continue;
        if (o1[j] > (unsigned __int128)I128_MAX) { rc = TR_MATH_OVERFLOW; break; }
        __int128 a, b, aw = w < 0 ? -(__int128)w : (__int128)w, m;
        if (!tr_attr_mul128((__int128)o0[j], w, &a) || !tr_attr_mul128((__int128)o1[j], w, &b) ||
            !tr_attr_mul128((__int128)o1[j], aw, &m) || !tr_attr_add128(bound, m, &bound) ||
            !tr_attr_add128(e, a, &e) || !tr_attr_add128(f, b, &f) ||
            !tr_attr_add128(lo, a < b ? a : b, &lo) || !tr_attr_add128(hi, a < b ? b : a, &hi)) rc = TR_MATH_OVERFLOW;
    }
    free(s); free(o0); free(o1); free(seen); free(deg);
    if (rc) return game_fail(g, error, rc);
    g->v.lo = tr_i128_set(lo); g->v.hi = tr_i128_set(hi); g->v.bound = tr_i128_set(bound);
    g->v.empty = tr_i128_set(e); g->v.full = tr_i128_set(f);
    g->signed_weights = neg;
    g->v.monotone = !neg ? 1 : !pos ? -1 : 0;
    g->v.sign = semiring == TR_PROV_COUNT && !neg;
    g->bytes = prov_bytes(plan);
    g->magic = GAME_MAGIC;
    set_error(error, TR_MATH_OK);
    return g;
}

int32_t tr_attr_game_get_info(const tr_attr_game *g, tr_attr_game_info *out) {
    if (!game_ok(g) || !out) return TR_MATH_INVALID;
    memset(out, 0, sizeof *out);
    out->kind = g->v.kind; out->semiring = g->v.semiring; out->n = g->v.n; out->source = g->v.source;
    out->monotone = g->v.monotone; out->sign = g->v.sign; out->degree = g->v.degree; out->projection = g->projection;
    out->baseline = g->baseline; out->nvars = g->v.nvars; out->present = g->present; out->absent = g->absent;
    out->noutputs = g->v.plan.noutputs; out->nodes = g->v.plan.nodes; out->bytes = g->bytes;
    out->signed_weights = g->signed_weights; out->players_used = g->players_used;
    out->lo = g->v.lo; out->hi = g->v.hi; out->bound = g->v.bound; out->empty = g->v.empty; out->full = g->v.full;
    return TR_MATH_OK;
}
int32_t tr_attr_game_get_view(const tr_attr_game *g, tr_attr_game_view *out) {
    if (!game_ok(g) || !out) return TR_MATH_INVALID;
    *out = g->v;
    return TR_MATH_OK;
}
int32_t tr_attr_plan_variables(const tr_prov_plan_view *p, uint32_t *out, uint64_t capacity, uint64_t *count) {
    if (!p || !count || tr_prov_plan_view_check(p)) return TR_MATH_INVALID;
    uint8_t *mark = (uint8_t *)zalloc(p->nvars ? p->nvars : 1);
    if (!mark) return TR_MATH_MEMORY;
    uint64_t k = 0;
    for (uint64_t i = 0; i < p->nodes; i++)
        if (p->node[i].op == TR_PROV_VAR && !mark[p->node[i].arg]) { mark[p->node[i].arg] = 1; k++; }
    if (out && capacity < k) { free(mark); return TR_MATH_INVALID; }
    if (out) for (uint64_t x = 0, j = 0; x < p->nvars; x++) if (mark[x]) out[j++] = (uint32_t)x;
    free(mark);
    *count = k;
    return TR_MATH_OK;
}

/* ------------------------------------------------------------------ enumeration */
typedef struct { const tr_attr_game_view *g; unsigned __int128 *scratch; } cpu_ctx;
static int32_t cpu_eval(void *ctx, const uint64_t *masks, uint64_t count, tr_math_i128 *values) {
    cpu_ctx *c = (cpu_ctx *)ctx;
    for (uint64_t i = 0; i < count; i++) {
        __int128 v;
        if (!tr_attr_eval(c->g, masks[i], c->scratch, &v)) return TR_MATH_OVERFLOW;
        values[i] = tr_i128_set(v);
    }
    return TR_MATH_OK;
}
static int32_t enum_check(const tr_attr_game *g, const uint64_t *masks, uint64_t count) {
    if (!game_ok(g)) return TR_MATH_INVALID;
    int32_t rc = tr_attr_view_check(&g->v, 1);
    if (rc) return rc;
    const uint64_t all = tr_attr_mask_all(g->v.n);
    for (uint64_t i = 0; masks && i < count; i++) if (masks[i] & ~all) return TR_MATH_INVALID;
    return TR_MATH_OK;
}

int32_t tr_attr_enum_need(const tr_attr_game *g, int32_t op, uint64_t arg, int32_t target, tr_math_need *need) {
    if (!game_ok(g)) return TR_MATH_INVALID;
    return tr_attr_view_need(&g->v, op, arg, target, need);
}

int32_t tr_attr_values(const tr_attr_game *g, const uint64_t *masks, uint64_t count, tr_math_i128 *out,
                       const tr_math_budget *budget, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    int32_t rc = enum_check(g, masks, count);
    if (rc) return rc;
    if (count && (!masks || !out)) return TR_MATH_INVALID;
    tr_math_need need;
    tr_attr_enum_need(g, TR_ATTR_OP_VALUES, count, TR_ATTR_NEED_CPU, &need);
    rc = tr_attr_check(budget, &need, refusal);
    if (rc) return rc;
    cpu_ctx c = { &g->v, (unsigned __int128 *)zalloc(tr_budget_mul(g->v.plan.nodes ? g->v.plan.nodes : 1, 16)) };
    tr_math_i128 *tmp = (tr_math_i128 *)zalloc(tr_budget_mul(count, 16));
    if (!c.scratch || !tmp) { free(c.scratch); free(tmp); return TR_MATH_MEMORY; }
    rc = cpu_eval(&c, masks, count, tmp);
    if (!rc && count) memcpy(out, tmp, (size_t)count * 16);
    free(c.scratch); free(tmp);
    return rc;
}

static int32_t exhaustive_request(const tr_attr_game *g, const tr_math_budget *budget, tr_math_refusal *refusal) {
    int32_t rc = enum_check(g, NULL, 0);
    if (rc) return rc;
    tr_math_need need;
    tr_attr_enum_need(g, TR_ATTR_OP_EXHAUSTIVE, 0, TR_ATTR_NEED_CPU, &need);
    rc = tr_attr_check(budget, &need, refusal);
    if (rc) return rc;
    if (g->v.n > TR_ATTR_MAX_EXHAUSTIVE) return TR_MATH_UNSUPPORTED;
    return tr_attr_exhaustive_bound(&g->v);
}

int32_t tr_attr_exhaustive(const tr_attr_game *g, tr_math_i128 *shapley_num, tr_math_i128 *banzhaf_num,
                           tr_attr_q *shapley, tr_attr_q *banzhaf, tr_attr_exhaustive_info *info,
                           const tr_math_budget *budget, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    int32_t rc = exhaustive_request(g, budget, refusal);
    if (rc) return rc;
    const tr_attr_game_view *v = &g->v;
    const int32_t n = v->n;
    __int128 pos[TR_ATTR_MAX_EXHAUSTIVE + 1], neg[TR_ATTR_MAX_EXHAUSTIVE + 1], fact;
    tr_attr_shapley_weights(n, pos, neg, &fact);
    unsigned __int128 *sh = (unsigned __int128 *)zalloc((uint64_t)n * 16), *bz = (unsigned __int128 *)zalloc((uint64_t)n * 16);
    unsigned __int128 *scratch = (unsigned __int128 *)zalloc(tr_budget_mul(v->plan.nodes ? v->plan.nodes : 1, 16));
    if (!sh || !bz || !scratch) { free(sh); free(bz); free(scratch); return TR_MATH_MEMORY; }
    /* every coalition once: player i receives +pos(s) v (i in S) or -neg(s) v, Banzhaf +-v; exact modulo 2^128, and
     * exact because the bound was proven */
    const uint64_t total = (uint64_t)1 << n;
    for (uint64_t S = 0; S < total && !rc; S++) {
        __int128 x;
        if (!tr_attr_eval(v, S, scratch, &x)) { rc = TR_MATH_OVERFLOW; break; }
        const int s = popc(S);
        const unsigned __int128 vv = (unsigned __int128)x, p = (unsigned __int128)pos[s] * vv, q = (unsigned __int128)neg[s] * vv;
        for (int32_t i = 0; i < n; i++) {
            if ((S >> i) & 1) { sh[i] += p; bz[i] += vv; }
            else { sh[i] -= q; bz[i] -= vv; }
        }
    }
    if (!rc) {
        tr_math_i128 *a = (tr_math_i128 *)zalloc((uint64_t)n * 16), *b = (tr_math_i128 *)zalloc((uint64_t)n * 16);
        if (!a || !b) rc = TR_MATH_MEMORY;
        else {
            for (int32_t i = 0; i < n; i++) { a[i] = tr_i128_set((__int128)sh[i]); b[i] = tr_i128_set((__int128)bz[i]); }
            if (shapley_num) memcpy(shapley_num, a, (size_t)n * 16);
            if (banzhaf_num) memcpy(banzhaf_num, b, (size_t)n * 16);
            tr_attr_exhaustive_finish(v, a, b, shapley, banzhaf, info);
        }
        free(a); free(b);
    }
    free(sh); free(bz); free(scratch);
    return rc;
}

int32_t tr_attr_exhaustive_adopt(const tr_attr_game *g, const tr_math_i128 *shapley_num, const tr_math_i128 *banzhaf_num,
                                 tr_attr_q *shapley, tr_attr_q *banzhaf, tr_attr_exhaustive_info *info) {
    int32_t rc = enum_check(g, NULL, 0);
    if (rc) return rc;
    if (!shapley_num || !banzhaf_num || g->v.n > TR_ATTR_MAX_EXHAUSTIVE) return TR_MATH_INVALID;
    rc = tr_attr_exhaustive_bound(&g->v);
    if (rc) return rc;
    tr_attr_exhaustive_finish(&g->v, shapley_num, banzhaf_num, shapley, banzhaf, info);
    return TR_MATH_OK;
}

int32_t tr_attr_explore(const tr_attr_game *g, uint64_t max_coalitions, tr_attr_q *shapley_lo, tr_attr_q *shapley_hi,
                        tr_attr_q *banzhaf_lo, tr_attr_q *banzhaf_hi, tr_attr_explore_info *info,
                        const tr_math_budget *budget, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    int32_t rc = enum_check(g, NULL, 0);
    if (rc) return rc;
    tr_math_need need;
    tr_attr_enum_need(g, TR_ATTR_OP_EXPLORE, max_coalitions, TR_ATTR_NEED_CPU, &need);
    rc = tr_attr_check(budget, &need, refusal);
    if (rc) return rc;
    cpu_ctx c = { &g->v, (unsigned __int128 *)zalloc(tr_budget_mul(g->v.plan.nodes ? g->v.plan.nodes : 1, 16)) };
    if (!c.scratch) return TR_MATH_MEMORY;
    rc = tr_attr_explore_run(&g->v, max_coalitions, TR_ATTR_BATCH, cpu_eval, &c, shapley_lo, shapley_hi, banzhaf_lo,
                             banzhaf_hi, info, budget, refusal);
    free(c.scratch);
    return rc;
}

int32_t tr_attr_search(const tr_attr_game *g, const tr_attr_search_query *q, uint64_t *solutions,
                       tr_attr_search_info *info, const tr_math_budget *budget, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    int32_t rc = enum_check(g, NULL, 0);
    if (rc) return rc;
    if (!q) return TR_MATH_INVALID;
    tr_math_need need;
    tr_attr_enum_need(g, TR_ATTR_OP_SEARCH, q->max_coalitions, TR_ATTR_NEED_CPU, &need);
    rc = tr_attr_check(budget, &need, refusal);
    if (rc) return rc;
    cpu_ctx c = { &g->v, (unsigned __int128 *)zalloc(tr_budget_mul(g->v.plan.nodes ? g->v.plan.nodes : 1, 16)) };
    if (!c.scratch) return TR_MATH_MEMORY;
    rc = tr_attr_search_run(&g->v, q, TR_ATTR_BATCH, cpu_eval, &c, solutions, info, budget, refusal);
    free(c.scratch);
    return rc;
}

/* ------------------------------------------------------------------ multilinear polynomials over players */
/* terms: sorted player lists (canonical order: by size, then colex) with int128 coefficients */
typedef struct { uint64_t n, pairs; uint64_t *off; uint32_t *pl; __int128 *coef; } mpoly;
static void mp_free(mpoly *p) { free(p->off); free(p->pl); free(p->coef); memset(p, 0, sizeof *p); }
static int32_t mp_alloc(mpoly *p, uint64_t n, uint64_t pairs) {
    memset(p, 0, sizeof *p);
    p->off = (uint64_t *)zalloc(tr_budget_mul(tr_budget_add(n, 1), 8));
    p->pl = (uint32_t *)zalloc(tr_budget_mul(pairs, 4));
    p->coef = (__int128 *)zalloc(tr_budget_mul(n, 16));
    if (!p->off || !p->pl || !p->coef) { mp_free(p); return TR_MATH_MEMORY; }
    p->n = n; p->pairs = pairs;
    return TR_MATH_OK;
}
static uint64_t mp_len(const mpoly *p, uint64_t i) { return p->off[i + 1] - p->off[i]; }
static int set_cmp(const uint32_t *a, uint64_t la, const uint32_t *b, uint64_t lb) {
    if (la != lb) return la < lb ? -1 : 1;
    for (uint64_t k = la; k-- > 0;) if (a[k] != b[k]) return a[k] < b[k] ? -1 : 1;
    return 0;
}
static int mp_cmp(const mpoly *p, uint64_t i, uint64_t j) {
    return set_cmp(p->pl + p->off[i], mp_len(p, i), p->pl + p->off[j], mp_len(p, j));
}
/* q = the canonical form of p (sorted, equal sets merged exactly, zero coefficients dropped) */
static int32_t mp_normalize(const mpoly *p, mpoly *q) {
    uint64_t *idx = (uint64_t *)zalloc(tr_budget_mul(p->n, 8)), *tmp = (uint64_t *)zalloc(tr_budget_mul(p->n, 8));
    if (!idx || !tmp) { free(idx); free(tmp); return TR_MATH_MEMORY; }
    for (uint64_t i = 0; i < p->n; i++) idx[i] = i;
    for (uint64_t w = 1; w < p->n; w *= 2) {            /* stable merge sort of the term indices */
        for (uint64_t st = 0; st < p->n; st += 2 * w) {
            uint64_t m = st + w < p->n ? st + w : p->n, e = st + 2 * w < p->n ? st + 2 * w : p->n, i = st, j = m, o = st;
            while (i < m && j < e) tmp[o++] = mp_cmp(p, idx[j], idx[i]) < 0 ? idx[j++] : idx[i++];
            while (i < m) tmp[o++] = idx[i++];
            while (j < e) tmp[o++] = idx[j++];
        }
        memcpy(idx, tmp, (size_t)p->n * 8);
    }
    free(tmp);
    uint64_t terms = 0, pairs = 0;
    int32_t rc = TR_MATH_OK;
    __int128 *sum = (__int128 *)zalloc(tr_budget_mul(p->n, 16));
    if (!sum) { free(idx); return TR_MATH_MEMORY; }
    for (uint64_t i = 0; i < p->n && !rc;) {            /* runs of equal sets: their exact sums */
        uint64_t j = i;
        __int128 c = 0;
        while (j < p->n && !mp_cmp(p, idx[i], idx[j])) {
            if (!tr_attr_add128(c, p->coef[idx[j]], &c)) { rc = TR_MATH_OVERFLOW; break; }
            j++;
        }
        sum[i] = c;
        if (c) { terms++; pairs += mp_len(p, idx[i]); }
        i = j;
    }
    if (!rc) rc = mp_alloc(q, terms, pairs);
    uint64_t t = 0, at = 0;
    for (uint64_t i = 0; i < p->n && !rc;) {
        uint64_t j = i;
        while (j < p->n && !mp_cmp(p, idx[i], idx[j])) j++;
        if (sum[i]) {
            uint64_t s0 = p->off[idx[i]], len = mp_len(p, idx[i]);
            q->coef[t] = sum[i]; q->off[t] = at;
            if (len) memcpy(q->pl + at, p->pl + s0, (size_t)len * 4);
            at += len; t++;
        }
        i = j;
    }
    if (!rc) q->off[t] = at;
    free(idx); free(sum);
    return rc;
}
/* the union of two sorted lists into out (sorted, distinct): its length */
static uint64_t set_union(const uint32_t *a, uint64_t la, const uint32_t *b, uint64_t lb, uint32_t *out) {
    uint64_t i = 0, j = 0, k = 0;
    while (i < la || j < lb) {
        if (j >= lb || (i < la && a[i] < b[j])) out[k++] = a[i++];
        else if (i >= la || b[j] < a[i]) out[k++] = b[j++];
        else { out[k++] = a[i++]; j++; }
    }
    return k;
}
/* raw concatenation of polynomials with integer multipliers (each term's coefficient times its multiplier) */
static int32_t mp_concat(const mpoly *const *ps, const __int128 *mult, uint64_t k, mpoly *raw) {
    uint64_t n = 0, pairs = 0;
    for (uint64_t i = 0; i < k; i++) { n = tr_budget_add(n, ps[i]->n); pairs = tr_budget_add(pairs, ps[i]->pairs); }
    if (n == TR_BUDGET_UNKNOWN || pairs == TR_BUDGET_UNKNOWN) return TR_MATH_OVERFLOW;
    int32_t rc = mp_alloc(raw, n, pairs);
    if (rc) return rc;
    uint64_t t = 0, at = 0;
    for (uint64_t i = 0; i < k && !rc; i++)
        for (uint64_t j = 0; j < ps[i]->n; j++) {
            uint64_t s0 = ps[i]->off[j], len = mp_len(ps[i], j);
            raw->off[t] = at;
            if (!tr_attr_mul128(ps[i]->coef[j], mult ? mult[i] : 1, &raw->coef[t])) { rc = TR_MATH_OVERFLOW; break; }
            if (len) memcpy(raw->pl + at, ps[i]->pl + s0, (size_t)len * 4);
            at += len; t++;
        }
    if (!rc) raw->off[t] = at;
    else mp_free(raw);
    return rc;
}
static int32_t mp_sum(const mpoly *const *ps, const __int128 *mult, uint64_t k, mpoly *out) {
    mpoly raw;
    int32_t rc = mp_concat(ps, mult, k, &raw);
    if (rc) return rc;
    rc = mp_normalize(&raw, out);
    mp_free(&raw);
    return rc;
}
/* a x b (sets united, coefficients multiplied exactly), canonical */
static int32_t mp_mul(const mpoly *a, const mpoly *b, mpoly *out) {
    uint64_t n = tr_budget_mul(a->n, b->n), pairs = tr_budget_add(tr_budget_mul(a->pairs, b->n), tr_budget_mul(b->pairs, a->n));
    if (n == TR_BUDGET_UNKNOWN || pairs == TR_BUDGET_UNKNOWN) return TR_MATH_OVERFLOW;
    mpoly raw;
    int32_t rc = mp_alloc(&raw, n, pairs);
    if (rc) return rc;
    uint64_t t = 0, at = 0;
    for (uint64_t i = 0; i < a->n && !rc; i++)
        for (uint64_t j = 0; j < b->n; j++) {
            if (!tr_attr_mul128(a->coef[i], b->coef[j], &raw.coef[t])) { rc = TR_MATH_OVERFLOW; break; }
            raw.off[t] = at;
            at += set_union(a->pl + a->off[i], mp_len(a, i), b->pl + b->off[j], mp_len(b, j), raw.pl + at);
            t++;
        }
    if (!rc) { raw.off[t] = at; raw.n = t; rc = mp_normalize(&raw, out); }
    mp_free(&raw);
    return rc;
}
static int32_t mp_const(mpoly *p, __int128 c) {
    int32_t rc = mp_alloc(p, c ? 1 : 0, 0);
    if (!rc && c) { p->coef[0] = c; p->off[0] = p->off[1] = 0; }
    return rc;
}
static int32_t mp_copy(const mpoly *a, mpoly *out) {
    const mpoly *one[1] = {a};
    return mp_sum(one, NULL, 1, out);
}

/* ------------------------------------------------------------------ the expansion of a provenance game's cone */
typedef struct { uint64_t bound, raw; uint32_t deg; uint64_t reach; } xnode;
static uint64_t term_bytes(uint64_t terms, uint32_t deg) { return tr_budget_mul(terms, 24 + 4 * (uint64_t)deg); }
/* the first pass: per node the term bound (sum / product of the children's, at most 2^(players reached) when n <= 63),
 * the degree and the raw sizes; the needs of the expansion */
static int32_t expand_plan(const tr_attr_game *g, xnode *x, tr_math_need *need) {
    const tr_prov_plan_view *p = &g->v.plan;
    const int32_t n = g->v.n;
    const int small = n <= 63;
    uint64_t memo = 0, rawmax = 0;
    for (uint64_t i = 0; i < p->nodes; i++) {
        const tr_prov_node nd = p->node[i];
        xnode *o = &x[i];
        memset(o, 0, sizeof *o);
        if (nd.op == TR_PROV_ONE) o->bound = 1;
        else if (nd.op == TR_PROV_VAR) {
            uint32_t q = g->var_player[nd.arg];
            if (q == TR_ATTR_PRESENT) o->bound = 1;
            else if (q != TR_ATTR_ABSENT) { o->bound = 1; o->deg = 1; o->reach = small ? (uint64_t)1 << q : 0; }
        } else if (nd.op == TR_PROV_PLUS) {
            uint64_t raw = 0;
            for (uint32_t c = 0; c < nd.n; c++) {
                const xnode *k = &x[p->child[nd.arg + c]];
                raw = tr_budget_add(raw, k->bound);
                if (k->deg > o->deg) o->deg = k->deg;
                o->reach |= k->reach;
            }
            o->raw = raw;
            o->bound = raw;
        } else if (nd.op == TR_PROV_TIMES) {
            uint64_t acc = x[p->child[nd.arg]].bound, deg = x[p->child[nd.arg]].deg;
            o->reach = x[p->child[nd.arg]].reach;
            for (uint32_t c = 1; c < nd.n; c++) {
                const xnode *k = &x[p->child[nd.arg + c]];
                uint64_t raw = tr_budget_mul(acc, k->bound);
                if (raw > o->raw) o->raw = raw;
                o->reach |= k->reach;
                deg += k->deg;
                acc = raw;
                if (small) { uint64_t cap = tr_budget_power2((uint32_t)popc(o->reach)); if (acc > cap) acc = cap; }
            }
            o->bound = acc;
            o->deg = (uint32_t)(deg < (uint64_t)n ? deg : (uint64_t)n);
        }
        if (small && nd.op != TR_PROV_ZERO) {
            uint64_t cap = tr_budget_power2((uint32_t)popc(o->reach));
            if (o->bound > cap) o->bound = cap;
        }
        memo = tr_budget_add(memo, term_bytes(o->bound, o->deg));
        uint64_t rb = tr_budget_add(term_bytes(o->raw, o->deg), tr_budget_mul(o->raw, 32));
        if (rb > rawmax) rawmax = rb;
    }
    uint64_t support = 0;
    uint32_t degree = 0;
    for (uint64_t j = 0; j < p->noutputs; j++) {
        const xnode *o = &x[p->output[j]];
        support = tr_budget_add(support, o->bound);
        if (o->deg > degree) degree = o->deg;
    }
    tr_attr_need_zero(need);
    need->need[TR_BUDGET_VARIABLES] = (uint64_t)n;
    need->need[TR_BUDGET_CIRCUIT] = p->nodes;
    need->need[TR_BUDGET_SUPPORT] = support;
    need->need[TR_BUDGET_DEGREE] = degree;
    uint64_t bytes = tr_budget_add(memo, rawmax);
    bytes = tr_budget_add(bytes, tr_budget_mul(tr_budget_add(term_bytes(support, degree), tr_budget_mul(support, 32)), 2));
    need->need[TR_BUDGET_BYTES] = tr_budget_add(bytes, tr_budget_mul(p->nodes, sizeof(xnode) + sizeof(mpoly) + 9));
    return TR_MATH_OK;
}
/* the bottom-up expansion: memo[output nodes] on success (the other memos freed after their last parent) */
static int32_t expand_cone(const tr_attr_game *g, mpoly *memo) {
    const tr_prov_plan_view *p = &g->v.plan;
    uint64_t *last = (uint64_t *)zalloc(tr_budget_mul(p->nodes, 8));
    uint8_t *keep = (uint8_t *)zalloc(p->nodes ? p->nodes : 1);
    if (!last || !keep) { free(last); free(keep); return TR_MATH_MEMORY; }
    for (uint64_t i = 0; i < p->nodes; i++) {
        const tr_prov_node nd = p->node[i];
        if (nd.op == TR_PROV_PLUS || nd.op == TR_PROV_TIMES)
            for (uint32_t c = 0; c < nd.n; c++) last[p->child[nd.arg + c]] = i;
    }
    for (uint64_t j = 0; j < p->noutputs; j++) keep[p->output[j]] = 1;
    int32_t rc = TR_MATH_OK;
    for (uint64_t i = 0; i < p->nodes && !rc; i++) {
        const tr_prov_node nd = p->node[i];
        mpoly *m = &memo[i];
        if (nd.op == TR_PROV_ZERO) rc = mp_const(m, 0);
        else if (nd.op == TR_PROV_ONE) rc = mp_const(m, 1);
        else if (nd.op == TR_PROV_VAR) {
            uint32_t q = g->var_player[nd.arg];
            if (q == TR_ATTR_PRESENT) rc = mp_const(m, 1);
            else if (q == TR_ATTR_ABSENT) rc = mp_const(m, 0);
            else { rc = mp_alloc(m, 1, 1); if (!rc) { m->coef[0] = 1; m->off[0] = 0; m->off[1] = 1; m->pl[0] = q; } }
        } else if (nd.op == TR_PROV_PLUS) {
            const mpoly **kids = (const mpoly **)calloc(nd.n, sizeof *kids);
            if (!kids) { rc = TR_MATH_MEMORY; break; }
            for (uint32_t c = 0; c < nd.n; c++) kids[c] = &memo[p->child[nd.arg + c]];
            rc = mp_sum(kids, NULL, nd.n, m);
            free((void *)kids);
        } else {
            mpoly acc, next;
            rc = mp_copy(&memo[p->child[nd.arg]], &acc);
            for (uint32_t c = 1; c < nd.n && !rc; c++) {
                rc = mp_mul(&acc, &memo[p->child[nd.arg + c]], &next);
                mp_free(&acc);
                if (!rc) acc = next;
            }
            if (!rc) *m = acc;
        }
        if (rc) break;
        if (nd.op == TR_PROV_PLUS || nd.op == TR_PROV_TIMES)       /* children whose last parent this is */
            for (uint32_t c = 0; c < nd.n; c++) {
                uint32_t k = p->child[nd.arg + c];
                if (last[k] == i && !keep[k]) mp_free(&memo[k]);
            }
    }
    free(last); free(keep);
    return rc;
}

/* ------------------------------------------------------------------ supports */
struct tr_attr_support_s {
    uint32_t magic;
    int32_t n, method, degree, nonneg, from_game, reconstructs;
    uint64_t candidates, bytes;
    mpoly p;
    __int128 sum, empty, game_full, game_empty;
};
static int supp_ok(const tr_attr_support *s) { return s && s->magic == SUPP_MAGIC; }
void tr_attr_support_free(tr_attr_support *s) {
    if (!supp_ok(s)) return;
    mp_free(&s->p);
    s->magic = 0;
    free(s);
}
/* adopt a canonical polynomial */
static tr_attr_support *supp_adopt(mpoly *p, int32_t n, int32_t method, const tr_attr_game *g, uint64_t candidates,
                                   int32_t *error) {
    tr_attr_support *s = (tr_attr_support *)zalloc(sizeof *s);
    if (!s) { mp_free(p); set_error(error, TR_MATH_MEMORY); return NULL; }
    s->p = *p;
    memset(p, 0, sizeof *p);
    s->n = n; s->method = method; s->candidates = candidates;
    s->nonneg = 1;
    __int128 sum = 0;
    int ok = 1;
    for (uint64_t i = 0; i < s->p.n; i++) {
        const __int128 c = s->p.coef[i];
        if (c < 0) s->nonneg = 0;
        if (!tr_attr_add128(sum, c, &sum)) ok = 0;
        uint64_t len = mp_len(&s->p, i);
        if ((int32_t)len > s->degree) s->degree = (int32_t)len;
        if (!len) s->empty = c;
    }
    if (!ok) { mp_free(&s->p); free(s); set_error(error, TR_MATH_OVERFLOW); return NULL; }
    s->sum = sum;
    if (g) { s->from_game = 1; s->game_full = tr_i128_get(g->v.full); s->game_empty = tr_i128_get(g->v.empty); }
    else { s->game_full = sum; s->game_empty = s->empty; }
    s->reconstructs = s->sum == s->game_full && s->empty == s->game_empty;
    s->bytes = tr_budget_add(sizeof *s, term_bytes(s->p.n, 0));
    s->bytes = tr_budget_add(s->bytes, tr_budget_mul(s->p.pairs, 4));
    s->magic = SUPP_MAGIC;
    set_error(error, TR_MATH_OK);
    return s;
}
/* the sets of a polynomial that contain no other set of it (the minimal ones), sorted */
static int32_t mp_minimal(const mpoly *p, mpoly *out) {
    uint8_t *drop = (uint8_t *)zalloc(p->n ? p->n : 1);
    if (!drop) return TR_MATH_MEMORY;
    uint64_t keep = 0, pairs = 0;
    for (uint64_t i = 0; i < p->n; i++) {
        const uint64_t li = mp_len(p, i);
        for (uint64_t j = 0; j < i && !drop[i]; j++) {        /* canonical order: smaller sets first */
            if (drop[j] || mp_len(p, j) >= li) continue;
            const uint32_t *a = p->pl + p->off[j], *b = p->pl + p->off[i];
            uint64_t x = 0, y = 0, lj = mp_len(p, j);
            while (x < lj && y < li) { if (a[x] == b[y]) { x++; y++; } else if (a[x] > b[y]) y++; else break; }
            if (x == lj) drop[i] = 1;
        }
        if (!drop[i]) { keep++; pairs += li; }
    }
    int32_t rc = mp_alloc(out, keep, pairs);
    uint64_t t = 0, at = 0;
    for (uint64_t i = 0; i < p->n && !rc; i++) {
        if (drop[i]) continue;
        uint64_t len = mp_len(p, i);
        out->coef[t] = 1; out->off[t] = at;
        if (len) memcpy(out->pl + at, p->pl + p->off[i], (size_t)len * 4);
        at += len; t++;
    }
    if (!rc) out->off[t] = at;
    free(drop);
    return rc;
}
/* v = 1 - prod_i (1 - x_{W_i}) over the witnesses (inclusion-exclusion), canonical */
static int32_t mp_witness_game(const mpoly *w, mpoly *out) {
    mpoly P, one;
    int32_t rc = mp_const(&P, 1);
    for (uint64_t i = 0; i < w->n && !rc; i++) {
        mpoly f, next;
        rc = mp_alloc(&f, 2, mp_len(w, i));                   /* 1 - x_W */
        if (rc) break;
        f.coef[0] = 1; f.off[0] = 0; f.off[1] = 0;
        f.coef[1] = -1; f.off[2] = mp_len(w, i);
        if (f.off[2]) memcpy(f.pl, w->pl + w->off[i], (size_t)f.off[2] * 4);
        if (!f.off[2]) { f.n = 1; f.coef[0] = 0; }            /* an empty witness: the factor is 0 */
        rc = mp_mul(&P, &f, &next);
        mp_free(&f); mp_free(&P);
        if (!rc) P = next;
    }
    if (rc) { mp_free(&P); return rc; }
    rc = mp_const(&one, 1);
    if (!rc) {
        const mpoly *ps[2] = {&one, &P};
        const __int128 mult[2] = {1, -1};
        rc = mp_sum(ps, mult, 2, out);
    }
    mp_free(&one); mp_free(&P);
    return rc;
}

static int32_t support_method(const tr_attr_game *g, int32_t method) {
    if (method == TR_ATTR_SUPPORT_AUTO)
        return g->v.kind == TR_ATTR_GAME_TABLE ? TR_ATTR_SUPPORT_TRANSFORM
             : g->v.semiring == TR_PROV_COUNT ? TR_ATTR_SUPPORT_EXPANSION : TR_ATTR_SUPPORT_WITNESSES;
    if (method == TR_ATTR_SUPPORT_TRANSFORM) return method;
    if (g->v.kind == TR_ATTR_GAME_PROV &&
        ((method == TR_ATTR_SUPPORT_EXPANSION && g->v.semiring == TR_PROV_COUNT) ||
         (method == TR_ATTR_SUPPORT_WITNESSES && g->v.semiring == TR_PROV_BOOL))) return method;
    return -1;
}
static uint64_t transform_extra(int32_t n) { return tr_budget_mul(tr_budget_power2((uint32_t)n), 48); }
int32_t tr_attr_support_need(const tr_attr_game *g, int32_t method, tr_math_need *need) {
    if (!game_ok(g) || !need) return TR_MATH_INVALID;
    int32_t m = support_method(g, method);
    if (m < 0) return TR_MATH_INVALID;
    if (m == TR_ATTR_SUPPORT_TRANSFORM) {
        tr_attr_need_zero(need);
        need->need[TR_BUDGET_VARIABLES] = (uint64_t)g->v.n;
        need->need[TR_BUDGET_TENSOR] = tr_budget_power2((uint32_t)g->v.n);
        need->need[TR_BUDGET_ITERATIONS] = g->v.kind == TR_ATTR_GAME_PROV ? tr_budget_power2((uint32_t)g->v.n) : (uint64_t)g->v.n;
        need->need[TR_BUDGET_BYTES] = transform_extra(g->v.n);
        return TR_MATH_OK;
    }
    xnode *x = (xnode *)zalloc(tr_budget_mul(g->v.plan.nodes, sizeof(xnode)));
    if (!x) return TR_MATH_MEMORY;
    int32_t rc = expand_plan(g, x, need);
    free(x);
    return rc;
}

tr_attr_support *tr_attr_support_build(const tr_attr_game *g, int32_t method, const tr_math_budget *budget,
                                       tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    if (!game_ok(g)) { set_error(error, TR_MATH_INVALID); return NULL; }
    const int32_t m = support_method(g, method), n = g->v.n;
    if (m < 0) { set_error(error, TR_MATH_INVALID); return NULL; }
    tr_math_need need;
    int32_t rc;
    if (m == TR_ATTR_SUPPORT_TRANSFORM) {
        tr_attr_support_need(g, m, &need);
        rc = tr_attr_check(budget, &need, refusal);
        if (rc) { set_error(error, rc); return NULL; }
        if (n > TR_ATTR_MAX_TABLE) { set_error(error, TR_MATH_UNSUPPORTED); return NULL; }
        const uint64_t size = (uint64_t)1 << n;
        tr_math_i128 *vals = (tr_math_i128 *)zalloc(size * 16), *mob = (tr_math_i128 *)zalloc(size * 16);
        if (!vals || !mob) { free(vals); free(mob); set_error(error, TR_MATH_MEMORY); return NULL; }
        if (g->v.kind == TR_ATTR_GAME_TABLE) memcpy(vals, g->table, (size_t)(size * 16));
        else {
            cpu_ctx c = { &g->v, (unsigned __int128 *)zalloc(tr_budget_mul(g->v.plan.nodes ? g->v.plan.nodes : 1, 16)) };
            uint64_t *masks = (uint64_t *)zalloc(TR_ATTR_BATCH * 8);
            rc = c.scratch && masks ? TR_MATH_OK : TR_MATH_MEMORY;
            for (uint64_t S = 0; S < size && !rc; S += TR_ATTR_BATCH) {
                uint64_t k = size - S < TR_ATTR_BATCH ? size - S : TR_ATTR_BATCH;
                for (uint64_t j = 0; j < k; j++) masks[j] = S + j;
                rc = cpu_eval(&c, masks, k, vals + S);
            }
            free(c.scratch); free(masks);
            if (rc) { free(vals); free(mob); set_error(error, rc); return NULL; }
        }
        rc = tr_lattice_transform(n, TR_LATTICE_MOBIUS_DOWN, TR_LATTICE_I128, vals, mob, budget, refusal);
        free(vals);
        if (rc) { free(mob); set_error(error, rc); return NULL; }
        uint64_t terms = 0, pairs = 0;
        for (uint64_t S = 0; S < size; S++) if (tr_i128_get(mob[S])) { terms++; pairs += (uint64_t)popc(S); }
        mpoly p;
        rc = mp_alloc(&p, terms, pairs);
        if (rc) { free(mob); set_error(error, rc); return NULL; }
        uint64_t t = 0, at = 0;
        for (int32_t s = 0; s <= n; s++) {                 /* by size, then increasing mask: canonical */
            uint64_t S = tr_attr_mask_all(s), total = tr_attr_binom(n, s);
            for (uint64_t r = 0; r < total; r++) {
                __int128 c = tr_i128_get(mob[S]);
                if (c) {
                    p.coef[t] = c; p.off[t] = at;
                    for (uint64_t b = S; b; b &= b - 1) p.pl[at++] = (uint32_t)__builtin_ctzll(b);
                    t++;
                }
                if (r + 1 < total) S = tr_attr_next_mask(S);
            }
        }
        p.off[t] = at;
        free(mob);
        return supp_adopt(&p, n, m, g, size, error);
    }
    /* EXPANSION / WITNESSES: the cone's multilinear expansion over players */
    const tr_prov_plan_view *pv = &g->v.plan;
    xnode *x = (xnode *)zalloc(tr_budget_mul(pv->nodes, sizeof(xnode)));
    if (!x) { set_error(error, TR_MATH_MEMORY); return NULL; }
    expand_plan(g, x, &need);
    free(x);
    rc = tr_attr_check(budget, &need, refusal);
    if (rc) { set_error(error, rc); return NULL; }
    mpoly *memo = (mpoly *)zalloc(tr_budget_mul(pv->nodes, sizeof(mpoly)));
    if (!memo) { set_error(error, TR_MATH_MEMORY); return NULL; }
    rc = expand_cone(g, memo);
    mpoly result;
    memset(&result, 0, sizeof result);
    uint64_t candidates = 0;
    if (!rc && m == TR_ATTR_SUPPORT_EXPANSION) {
        const mpoly **ps = (const mpoly **)calloc(pv->noutputs ? pv->noutputs : 1, sizeof *ps);
        __int128 *mult = (__int128 *)calloc(pv->noutputs ? pv->noutputs : 1, sizeof *mult);
        if (!ps || !mult) rc = TR_MATH_MEMORY;
        for (uint64_t j = 0; j < pv->noutputs && !rc; j++) {
            ps[j] = &memo[pv->output[j]]; mult[j] = g->weights[j];
            candidates += ps[j]->n;
        }
        if (!rc) rc = mp_sum(ps, mult, pv->noutputs, &result);
        free((void *)ps); free(mult);
    } else if (!rc) {
        /* WITNESSES: per output its minimal player sets, then inclusion-exclusion; checked before the products */
        mpoly *wit = (mpoly *)zalloc(tr_budget_mul(pv->noutputs, sizeof(mpoly)));
        if (!wit) rc = TR_MATH_MEMORY;
        tr_math_need n2;
        tr_attr_need_zero(&n2);
        uint64_t quad = 0;
        for (uint64_t j = 0; j < pv->noutputs; j++) {
            const uint64_t t = memo[pv->output[j]].n;
            quad = tr_budget_add(quad, tr_budget_mul(t, t));
        }
        n2.need[TR_BUDGET_TENSOR] = quad;
        if (!rc) rc = tr_attr_check(budget, &n2, refusal);
        for (uint64_t j = 0; j < pv->noutputs && !rc; j++) {
            uint64_t o = pv->output[j];
            uint64_t prev = 0;
            for (uint64_t i = 0; i < j; i++) if (pv->output[i] == o) { prev = i + 1; break; }
            if (prev) rc = mp_copy(&wit[prev - 1], &wit[j]);
            else rc = mp_minimal(&memo[o], &wit[j]);
        }
        uint64_t terms = 0, bytes = 0;
        for (uint64_t j = 0; j < pv->noutputs && !rc; j++) {
            if (wit[j].n > TR_ATTR_MAX_WITNESSES) { rc = TR_MATH_UNSUPPORTED; break; }
            const uint64_t t = tr_budget_power2((uint32_t)wit[j].n);
            terms = tr_budget_add(terms, t);
            const uint64_t b = tr_budget_mul(term_bytes(tr_budget_mul(t, 2), (uint32_t)(wit[j].pairs < (uint64_t)n ? wit[j].pairs : (uint64_t)n)), 3);
            if (b > bytes) bytes = b;
        }
        if (!rc) {
            tr_attr_need_zero(&n2);
            n2.need[TR_BUDGET_TENSOR] = tr_budget_add(quad, terms);
            n2.need[TR_BUDGET_BYTES] = tr_budget_add(bytes, need.need[TR_BUDGET_BYTES]);
            rc = tr_attr_check(budget, &n2, refusal);
        }
        mpoly *games = rc ? NULL : (mpoly *)zalloc(tr_budget_mul(pv->noutputs, sizeof(mpoly)));
        if (!rc && !games) rc = TR_MATH_MEMORY;
        for (uint64_t j = 0; j < pv->noutputs && !rc; j++) { rc = mp_witness_game(&wit[j], &games[j]); candidates += games[j].n; }
        if (!rc) {
            const mpoly **ps = (const mpoly **)calloc(pv->noutputs ? pv->noutputs : 1, sizeof *ps);
            __int128 *mult = (__int128 *)calloc(pv->noutputs ? pv->noutputs : 1, sizeof *mult);
            if (!ps || !mult) rc = TR_MATH_MEMORY;
            for (uint64_t j = 0; j < pv->noutputs && !rc; j++) { ps[j] = &games[j]; mult[j] = g->weights[j]; }
            if (!rc) rc = mp_sum(ps, mult, pv->noutputs, &result);
            free((void *)ps); free(mult);
        }
        for (uint64_t j = 0; games && j < pv->noutputs; j++) mp_free(&games[j]);
        for (uint64_t j = 0; wit && j < pv->noutputs; j++) mp_free(&wit[j]);
        free(games); free(wit);
    }
    for (uint64_t i = 0; i < pv->nodes; i++) mp_free(&memo[i]);
    free(memo);
    if (rc) { mp_free(&result); set_error(error, rc); return NULL; }
    return supp_adopt(&result, n, m, g, candidates, error);
}

tr_attr_support *tr_attr_support_terms(int32_t n, uint64_t nterms, const uint64_t *offset, const uint32_t *players,
                                       const tr_math_i128 *coef, const tr_math_budget *budget, tr_math_refusal *refusal,
                                       int32_t *error) {
    tr_budget_none(refusal);
    if (n < 1 || !offset || (nterms && !coef) || offset[0] != 0) { set_error(error, TR_MATH_INVALID); return NULL; }
    for (uint64_t i = 0; i < nterms; i++) if (offset[i + 1] < offset[i]) { set_error(error, TR_MATH_INVALID); return NULL; }
    const uint64_t pairs = offset[nterms];
    if (pairs && !players) { set_error(error, TR_MATH_INVALID); return NULL; }
    tr_math_need need;
    tr_attr_need_zero(&need);
    need.need[TR_BUDGET_VARIABLES] = (uint64_t)n;
    need.need[TR_BUDGET_SUPPORT] = nterms;
    need.need[TR_BUDGET_BYTES] = tr_budget_mul(tr_budget_add(term_bytes(nterms, 0), tr_budget_mul(pairs, 4)), 3);
    int32_t rc = tr_attr_check(budget, &need, refusal);
    if (rc) { set_error(error, rc); return NULL; }
    if ((uint32_t)n > TR_ATTR_MAX_PLAYERS) { set_error(error, TR_MATH_UNSUPPORTED); return NULL; }
    for (uint64_t i = 0; i < nterms; i++)
        for (uint64_t k = offset[i]; k < offset[i + 1]; k++)
            if (players[k] >= (uint32_t)n || (k > offset[i] && players[k] <= players[k - 1])) { set_error(error, TR_MATH_INVALID); return NULL; }
    mpoly raw, p;
    rc = mp_alloc(&raw, nterms, pairs);
    if (rc) { set_error(error, rc); return NULL; }
    memcpy(raw.off, offset, (size_t)(nterms + 1) * 8);
    if (pairs) memcpy(raw.pl, players, (size_t)pairs * 4);
    for (uint64_t i = 0; i < nterms; i++) raw.coef[i] = tr_i128_get(coef[i]);
    rc = mp_normalize(&raw, &p);
    mp_free(&raw);
    if (rc) { set_error(error, rc); return NULL; }
    return supp_adopt(&p, n, TR_ATTR_SUPPORT_TERMS, NULL, nterms, error);
}

int32_t tr_attr_support_get_info(const tr_attr_support *s, tr_attr_support_info *out) {
    if (!supp_ok(s) || !out) return TR_MATH_INVALID;
    memset(out, 0, sizeof *out);
    out->n = s->n; out->method = s->method; out->degree = s->degree; out->nonnegative = s->nonneg;
    out->terms = s->p.n; out->pairs = s->p.pairs; out->bytes = s->bytes; out->candidates = s->candidates;
    out->from_game = s->from_game; out->reconstructs = s->reconstructs;
    out->sum = tr_i128_set(s->sum); out->empty = tr_i128_set(s->empty);
    out->game_full = tr_i128_set(s->game_full); out->game_empty = tr_i128_set(s->game_empty);
    return TR_MATH_OK;
}
int32_t tr_attr_support_get(const tr_attr_support *s, uint64_t *offset, uint32_t *players, tr_math_i128 *coef,
                            uint64_t terms_capacity, uint64_t pairs_capacity) {
    if (!supp_ok(s) || !offset || terms_capacity < s->p.n || pairs_capacity < s->p.pairs || (s->p.n && !coef) ||
        (s->p.pairs && !players)) return TR_MATH_INVALID;
    memcpy(offset, s->p.off, (size_t)(s->p.n + 1) * 8);
    if (s->p.pairs) memcpy(players, s->p.pl, (size_t)s->p.pairs * 4);
    for (uint64_t i = 0; i < s->p.n; i++) coef[i] = tr_i128_set(s->p.coef[i]);
    return TR_MATH_OK;
}

/* ------------------------------------------------------------------ indices from a support */
int32_t tr_attr_support_values(const tr_attr_support *s, int32_t index, tr_attr_q *out, tr_attr_identity *id,
                               const tr_math_budget *budget, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!supp_ok(s) || !out || (index != TR_ATTR_SHAPLEY && index != TR_ATTR_BANZHAF)) return TR_MATH_INVALID;
    const int32_t n = s->n;
    tr_math_need need;
    tr_attr_need_zero(&need);
    need.need[TR_BUDGET_VARIABLES] = (uint64_t)n;
    need.need[TR_BUDGET_SUPPORT] = s->p.n;
    need.need[TR_BUDGET_OUTPUT] = (uint64_t)n;
    need.need[TR_BUDGET_BYTES] = tr_budget_mul((uint64_t)n, sizeof(tr_attr_qacc) + sizeof(tr_attr_q));
    int32_t rc = tr_attr_check(budget, &need, refusal);
    if (rc) return rc;
    tr_attr_qacc *acc = (tr_attr_qacc *)zalloc(tr_budget_mul((uint64_t)n, sizeof(tr_attr_qacc)));
    if (!acc) return TR_MATH_MEMORY;
    for (int32_t i = 0; i < n; i++) acc[i] = tr_attr_qzero();
    for (uint64_t k = 0; k < s->p.n && !rc; k++) {
        const uint64_t t = mp_len(&s->p, k);
        if (!t) continue;
        if (index == TR_ATTR_BANZHAF && t > 126) { rc = TR_MATH_OVERFLOW; break; }
        const __int128 d = index == TR_ATTR_SHAPLEY ? (__int128)t : (__int128)1 << (t - 1);
        for (uint64_t q = s->p.off[k]; q < s->p.off[k + 1]; q++)
            if (!tr_attr_qadd(&acc[s->p.pl[q]], s->p.coef[k], d)) { rc = TR_MATH_OVERFLOW; break; }
    }
    tr_attr_qacc total = tr_attr_qzero(), e;
    for (int32_t i = 0; i < n && !rc; i++) if (!tr_attr_qadd_q(&total, acc[i])) rc = TR_MATH_OVERFLOW;
    __int128 diff = 0;
    if (!rc && !tr_attr_add128(s->game_full, -s->game_empty, &diff)) rc = TR_MATH_OVERFLOW;
    if (!rc) {
        for (int32_t i = 0; i < n; i++) out[i] = tr_attr_qout(acc[i]);
        if (id) {
            e.num = diff; e.den = 1;
            *id = tr_attr_identity_make(index == TR_ATTR_SHAPLEY ? TR_ATTR_ID_EFFICIENCY : TR_ATTR_ID_NONE, total, total, e);
            if (index != TR_ATTR_SHAPLEY) id->holds = 0;
        }
    }
    free(acc);
    return rc;
}

struct tr_attr_table_s {
    uint32_t magic;
    int32_t index, min_order, max_order, n;
    uint64_t sets, pairs, candidates, bytes;
    uint64_t *off;
    uint32_t *pl;
    tr_attr_q *val;
    tr_attr_identity id;
};
static int table_ok(const tr_attr_table *t) { return t && t->magic == TABL_MAGIC; }
void tr_attr_table_free(tr_attr_table *t) {
    if (!table_ok(t)) return;
    free(t->off); free(t->pl); free(t->val);
    t->magic = 0;
    free(t);
}
/* C(t, k) exactly in 128 bits (0 when it does not fit) */
static __int128 binom128(uint64_t t, uint64_t k) {
    if (k > t) return 0;
    if (k > t - k) k = t - k;
    __int128 r = 1;
    for (uint64_t i = 1; i <= k; i++) {
        __int128 x;
        if (!tr_attr_mul128(r, (__int128)(t - k + i), &x)) return 0;
        r = x / (__int128)i;
    }
    return r;
}
/* the candidate subsets S of a term of size t: sizes lo..hi (lo > hi: none) */
static void cand_sizes(int32_t index, int32_t min_order, int32_t max_order, uint64_t t, int64_t *lo, int64_t *hi) {
    *lo = 1; *hi = 0;
    if (index == TR_ATTR_MOBIUS) { if ((int64_t)t >= min_order && (int64_t)t <= max_order) *lo = *hi = (int64_t)t; return; }
    if (index == TR_ATTR_SHAPLEY_TAYLOR) {
        if ((int64_t)t < max_order) { if ((int64_t)t >= min_order) *lo = *hi = (int64_t)t; }
        else if (max_order >= min_order) *lo = *hi = max_order;
        return;
    }
    *lo = min_order; *hi = (int64_t)t < max_order ? (int64_t)t : max_order;
}
static int32_t inter_request(const tr_attr_support *s, int32_t index, int32_t min_order, int32_t max_order, tr_math_need *need) {
    if (!supp_ok(s) || index < TR_ATTR_MOBIUS || index > TR_ATTR_SHAPLEY_TAYLOR || min_order < 0 || max_order < min_order ||
        (index == TR_ATTR_SHAPLEY_TAYLOR && max_order < 1)) return TR_MATH_INVALID;
    uint64_t cand = 0;
    for (uint64_t k = 0; k < s->p.n; k++) {
        int64_t lo, hi;
        const uint64_t t = mp_len(&s->p, k);
        cand_sizes(index, min_order, max_order, t, &lo, &hi);
        for (int64_t j = lo; j <= hi; j++) cand = tr_budget_add(cand, tr_budget_binomial(t, (uint64_t)j));
    }
    tr_attr_need_zero(need);
    need->need[TR_BUDGET_VARIABLES] = (uint64_t)s->n;
    need->need[TR_BUDGET_DEGREE] = (uint64_t)max_order;
    need->need[TR_BUDGET_SUPPORT] = s->p.n;
    need->need[TR_BUDGET_TENSOR] = cand;
    need->need[TR_BUDGET_OUTPUT] = cand;
    uint64_t per = 8 + 32 + 16 + 16 + 4 * (uint64_t)max_order;
    need->need[TR_BUDGET_BYTES] = tr_budget_mul(tr_budget_add(cand, 1), per);
    return TR_MATH_OK;
}
int32_t tr_attr_interactions_need(const tr_attr_support *s, int32_t index, int32_t min_order, int32_t max_order,
                                  tr_math_need *need) {
    if (!need) return TR_MATH_INVALID;
    return inter_request(s, index, min_order, max_order, need);
}

tr_attr_table *tr_attr_interactions(const tr_attr_support *s, int32_t index, int32_t min_order, int32_t max_order,
                                    const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    tr_math_need need;
    int32_t rc = inter_request(s, index, min_order, max_order, &need);
    if (rc) { set_error(error, rc); return NULL; }
    rc = tr_attr_check(budget, &need, refusal);
    if (rc) { set_error(error, rc); return NULL; }
    const uint64_t cand = need.need[TR_BUDGET_TENSOR];
    uint64_t pairs = 0;
    for (uint64_t k = 0; k < s->p.n; k++) {
        int64_t lo, hi;
        const uint64_t t = mp_len(&s->p, k);
        cand_sizes(index, min_order, max_order, t, &lo, &hi);
        for (int64_t j = lo; j <= hi; j++) pairs = tr_budget_add(pairs, tr_budget_mul(tr_budget_binomial(t, (uint64_t)j), (uint64_t)j));
    }
    /* the (set, contribution) candidates: a polynomial whose coefficients index the contributions */
    mpoly raw;
    rc = mp_alloc(&raw, cand, pairs);
    tr_attr_qacc *contrib = rc ? NULL : (tr_attr_qacc *)zalloc(tr_budget_mul(cand, sizeof(tr_attr_qacc)));
    if (!rc && !contrib) rc = TR_MATH_MEMORY;
    uint64_t c = 0, at = 0;
    uint64_t idx[TR_ATTR_MAX_ENUM + 2];
    for (uint64_t k = 0; k < s->p.n && !rc; k++) {
        int64_t lo, hi;
        const uint64_t t = mp_len(&s->p, k);
        const uint32_t *T = s->p.pl + s->p.off[k];
        const __int128 mt = s->p.coef[k];
        cand_sizes(index, min_order, max_order, t, &lo, &hi);
        for (int64_t j = lo; j <= hi && !rc; j++) {
            __int128 den = 1;
            switch (index) {
            case TR_ATTR_MOBIUS: den = 1; break;
            case TR_ATTR_SHAPLEY_INTERACTION: den = (__int128)(t - (uint64_t)j + 1); break;
            case TR_ATTR_BANZHAF_INTERACTION:
                if (t - (uint64_t)j > 126) rc = TR_MATH_OVERFLOW;
                else den = (__int128)1 << (t - (uint64_t)j);
                break;
            default:
                den = (int64_t)t < max_order ? 1 : binom128(t, (uint64_t)max_order);
                if (!den) rc = TR_MATH_OVERFLOW;
            }
            if (rc) break;
            if (j > TR_ATTR_MAX_ENUM) { rc = TR_MATH_UNSUPPORTED; break; }
            for (int64_t q = 0; q < j; q++) idx[q] = (uint64_t)q;   /* the j-combinations of the t players of T */
            for (;;) {
                raw.off[c] = at;
                for (int64_t q = 0; q < j; q++) raw.pl[at++] = T[idx[q]];
                raw.coef[c] = (__int128)c;                          /* the candidate's own index */
                contrib[c].num = mt; contrib[c].den = den;
                c++;
                int64_t q = j - 1;
                while (q >= 0 && idx[q] == t - (uint64_t)(j - q)) q--;
                if (q < 0) break;
                idx[q]++;
                for (int64_t r = q + 1; r < j; r++) idx[r] = idx[r - 1] + 1;
            }
        }
    }
    if (!rc) raw.off[c] = at;
    /* sort the candidates by set (stable), then sum each run of equal sets exactly */
    uint64_t *ord = rc ? NULL : (uint64_t *)zalloc(tr_budget_mul(cand, 8)), *tmp = rc ? NULL : (uint64_t *)zalloc(tr_budget_mul(cand, 8));
    if (!rc && (!ord || !tmp)) rc = TR_MATH_MEMORY;
    if (!rc) {
        for (uint64_t i = 0; i < cand; i++) ord[i] = i;
        for (uint64_t w = 1; w < cand; w *= 2) {
            for (uint64_t st = 0; st < cand; st += 2 * w) {
                uint64_t m = st + w < cand ? st + w : cand, e = st + 2 * w < cand ? st + 2 * w : cand, i = st, j = m, o = st;
                while (i < m && j < e) tmp[o++] = mp_cmp(&raw, ord[j], ord[i]) < 0 ? ord[j++] : ord[i++];
                while (i < m) tmp[o++] = ord[i++];
                while (j < e) tmp[o++] = ord[j++];
            }
            memcpy(ord, tmp, (size_t)cand * 8);
        }
    }
    free(tmp);
    tr_attr_table *out = rc ? NULL : (tr_attr_table *)zalloc(sizeof *out);
    if (!rc && !out) rc = TR_MATH_MEMORY;
    tr_attr_qacc total = tr_attr_qzero();
    if (!rc) {
        out->off = (uint64_t *)zalloc(tr_budget_mul(cand + 1, 8));
        out->pl = (uint32_t *)zalloc(tr_budget_mul(pairs, 4));
        out->val = (tr_attr_q *)zalloc(tr_budget_mul(cand, sizeof(tr_attr_q)));
        if (!out->off || !out->pl || !out->val) rc = TR_MATH_MEMORY;
    }
    uint64_t sets = 0, op = 0;
    for (uint64_t i = 0; i < cand && !rc;) {
        uint64_t j = i;
        tr_attr_qacc sum = tr_attr_qzero();
        while (j < cand && !mp_cmp(&raw, ord[i], ord[j])) {
            if (!tr_attr_qadd_q(&sum, contrib[ord[j]])) { rc = TR_MATH_OVERFLOW; break; }
            j++;
        }
        if (rc) break;
        if (sum.num) {
            const uint64_t len = mp_len(&raw, ord[i]);
            out->off[sets] = op;
            if (len) memcpy(out->pl + op, raw.pl + raw.off[ord[i]], (size_t)len * 4);
            op += len;
            out->val[sets++] = tr_attr_qout(sum);
            if (index == TR_ATTR_SHAPLEY_TAYLOR ? len >= 1 : 1)
                if (!tr_attr_qadd_q(&total, sum)) rc = TR_MATH_OVERFLOW;
        }
        i = j;
    }
    free(ord); free(contrib); mp_free(&raw);
    if (rc) {                                           /* the table is not adopted yet: free its arrays */
        if (out) { free(out->off); free(out->pl); free(out->val); free(out); }
        set_error(error, rc);
        return NULL;
    }
    out->off[sets] = op;
    out->index = index; out->min_order = min_order; out->max_order = max_order; out->n = s->n;
    out->sets = sets; out->pairs = op; out->candidates = cand;
    out->bytes = tr_budget_add(sizeof *out, tr_budget_add(tr_budget_mul(sets + 1, 8 + sizeof(tr_attr_q)), tr_budget_mul(op, 4)));
    tr_attr_qacc e;
    e.den = 1;
    memset(&out->id, 0, sizeof out->id);
    if (index == TR_ATTR_SHAPLEY_TAYLOR && min_order <= 1) {
        e.num = s->game_full - s->game_empty;
        out->id = tr_attr_identity_make(TR_ATTR_ID_TAYLOR_EFFICIENCY, total, total, e);
    } else if (index == TR_ATTR_MOBIUS && min_order == 0 && max_order >= s->degree) {
        e.num = s->game_full;
        out->id = tr_attr_identity_make(TR_ATTR_ID_MOBIUS_SUM, total, total, e);
    }
    out->magic = TABL_MAGIC;
    set_error(error, TR_MATH_OK);
    return out;
}

int32_t tr_attr_table_get_info(const tr_attr_table *t, tr_attr_table_info *out) {
    if (!table_ok(t) || !out) return TR_MATH_INVALID;
    memset(out, 0, sizeof *out);
    out->index = t->index; out->min_order = t->min_order; out->max_order = t->max_order; out->n = t->n;
    out->sets = t->sets; out->pairs = t->pairs; out->candidates = t->candidates; out->bytes = t->bytes;
    out->identity = t->id;
    return TR_MATH_OK;
}
int32_t tr_attr_table_get(const tr_attr_table *t, uint64_t *offset, uint32_t *players, tr_attr_q *values,
                          uint64_t sets_capacity, uint64_t pairs_capacity) {
    if (!table_ok(t) || !offset || sets_capacity < t->sets || pairs_capacity < t->pairs || (t->sets && !values) ||
        (t->pairs && !players)) return TR_MATH_INVALID;
    memcpy(offset, t->off, (size_t)(t->sets + 1) * 8);
    if (t->pairs) memcpy(players, t->pl, (size_t)t->pairs * 4);
    if (t->sets) memcpy(values, t->val, (size_t)t->sets * sizeof(tr_attr_q));
    return TR_MATH_OK;
}
