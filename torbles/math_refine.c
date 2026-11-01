/* MULTI-002: adaptive refinement and change localization over MULTI-001 hierarchies; see math_refine.h.
 * SPDX-License-Identifier: Apache-2.0 */
#include "math_refine.h"
#include "mathematics.h"
#include "math_domain.h"
#include "math_handles.h"
#include <stdlib.h>
#include <string.h>

TR_MATH_ABI(tr_refine_atom, TR_REFINE_ABI);          /* the ctypes boundary of MULTI-002 (math_handles.h) */
TR_MATH_ABI(tr_refine_request, TR_REFINE_ABI);
TR_MATH_ABI(tr_refine_class, TR_REFINE_ABI);
TR_MATH_ABI(tr_refine_plan, TR_REFINE_ABI);
TR_MATH_ABI(tr_refine_level_view, TR_REFINE_ABI);
TR_MATH_ABI(tr_refine_bound, TR_REFINE_ABI);
TR_MATH_ABI(tr_refine_entry, TR_REFINE_ABI);
TR_MATH_ABI(tr_refine_stats, TR_REFINE_ABI);
TR_MATH_ABI(tr_refine_info, TR_REFINE_ABI);
TR_MATH_ABI(tr_refine_change, TR_REFINE_ABI);
TR_MATH_ABI(tr_refine_ins, TR_REFINE_ABI);
TR_MATH_ABI(tr_refine_update_view, TR_REFINE_ABI);

_Static_assert(sizeof(tr_refine_atom) == 16, "tr_refine_atom ABI");
_Static_assert(sizeof(tr_refine_request) == 184, "tr_refine_request ABI");
_Static_assert(sizeof(tr_refine_class) == 32, "tr_refine_class ABI");
_Static_assert(sizeof(tr_refine_plan) == 80, "tr_refine_plan ABI");
_Static_assert(sizeof(tr_refine_level_view) == 48, "tr_refine_level_view ABI");
_Static_assert(sizeof(tr_refine_bound) == 144, "tr_refine_bound ABI");
_Static_assert(sizeof(tr_refine_entry) == 160, "tr_refine_entry ABI");
_Static_assert(sizeof(tr_refine_stats) == 200, "tr_refine_stats ABI");
_Static_assert(sizeof(tr_refine_info) == 416, "tr_refine_info ABI");
_Static_assert(sizeof(tr_refine_change) == 24, "tr_refine_change ABI");
_Static_assert(sizeof(tr_refine_ins) == 40, "tr_refine_ins ABI");
_Static_assert(sizeof(tr_refine_update_view) == 208, "tr_refine_update_view ABI");

#define STATE_MAGIC UINT32_C(0x4e494652)
#define QUERY_MAGIC UINT32_C(0x51494652)
#define UPDATE_MAGIC UINT32_C(0x55494652)
#define L TR_MULTI_MAX_LEVELS
#define I128_LIMIT ((((unsigned __int128)1) << 127) - 1)

struct tr_refine_state_s {
    uint32_t magic;
    int32_t levels, ncols, nterms, mode, status, promise, strategy, reason;
    uint64_t groups[L], rows, step, touched, work, rebuild_work, bytes;
    uint64_t *nrows[L], *count[L];
    tr_math_i128 *sum[L];
    tr_unc_input *inputs[L];
    uint8_t *tight[L];
    uint32_t *parent[L];                       /* levels - 1 maps */
    uint64_t *child_off[L];                    /* level >= 1: groups + 1 */
    uint32_t *child_ids[L];
    uint8_t *exps;                             /* nterms x ncols */
    tr_bank_term *terms;
    uint32_t joint;
    tr_delta_state *leaves;
    uint64_t nchanges, changed[L], loose[L];
    tr_refine_change *changes;
    int64_t *ch_count;
    uint64_t *ch_sum;
};

struct tr_refine_query_s {
    uint32_t magic;
    const tr_refine_state *s;
    tr_refine_request req;
    tr_refine_plan plan;
    tr_refine_atom atoms[TR_REFINE_MAX_PLAN_ATOMS];
    tr_refine_class classes[TR_REFINE_MAX_CLASSES];
    tr_unc_node *nodes;
    int32_t level, exact, done, supplied;
    __int128 thr;                              /* THRESHOLD: the integer threshold of A (form ge: A >= thr, else A <= thr) */
    int32_t ge, reserved;
    uint32_t *wave, *next;
    uint64_t nwave, nnext, capwave;
    tr_refine_bound *bounds;
    tr_refine_entry *entries;
    uint64_t nentries, capentries, expansions;
    tr_refine_stats stats;
    tr_unc_value *scratch;
    uint64_t *dcnt[L];                         /* TOPK: present target descendants of each region (levels >= target) */
    uint64_t counted;                          /* links the counts followed */
};

struct tr_refine_update_s {
    uint32_t magic;
    const tr_refine_state *s;
    tr_delta_state *leaves;                    /* the new leaves (owned until finish) */
    tr_refine_update_view view;
    uint64_t total, nchild;
    uint32_t *ids;
    uint64_t *child_off;
    uint32_t *child_idx;
    tr_math_i128 *drows, *dcount, *dpresent;
    uint64_t *dsum;
    tr_refine_ins *ins;
    uint64_t touched, work;
    int32_t rolled;
};

static void set_error(int32_t *error, int32_t code) { if (error) *error = code; }
static const tr_math_budget *budget_or(const tr_math_budget *b, tr_math_budget *dflt) {
    if (b) return b;
    tr_budget_default(dflt);
    return dflt;
}
static void need_zero(tr_math_need *n) { for (int i = 0; i < TR_BUDGET_RESOURCES; i++) n->need[i] = 0; }
static void *zalloc(uint64_t bytes) {
    if (bytes > (uint64_t)SIZE_MAX) return NULL;
    return calloc(1, bytes ? (size_t)bytes : 1);
}
static int state_valid(const tr_refine_state *s) { return s && s->magic == STATE_MAGIC; }
static int query_valid(const tr_refine_query *q) { return q && q->magic == QUERY_MAGIC; }
static int update_valid(const tr_refine_update *u) { return u && u->magic == UPDATE_MAGIC; }

/* ------------------------------------------------------------------ states */
static void state_release(tr_refine_state *s) {
    if (!s) return;
    for (int i = 0; i < L; i++) {
        free(s->nrows[i]); free(s->count[i]); free(s->sum[i]); free(s->inputs[i]); free(s->tight[i]);
        free(s->parent[i]); free(s->child_off[i]); free(s->child_ids[i]);
    }
    free(s->exps); free(s->terms); free(s->changes); free(s->ch_count); free(s->ch_sum);
    tr_delta_state_free(s->leaves);
    s->magic = 0;
    free(s);
}
void tr_refine_free(tr_refine_state *s) { if (state_valid(s)) state_release(s); }

/* the union of two groups' values of one column */
static void input_merge(tr_unc_input *dst, const tr_unc_input *src) {
    dst->rows += src->rows;
    if (!src->present) return;
    if (!dst->present) { uint64_t r = dst->rows; *dst = *src; dst->rows = r; return; }
    uint64_t d = src->min > dst->min ? (uint64_t)src->min - (uint64_t)dst->min : (uint64_t)dst->min - (uint64_t)src->min;
    dst->modulus = tr_gcd_u64(tr_gcd_u64(dst->modulus, src->modulus), d);
    if (src->min < dst->min) dst->min = src->min;
    if (src->max > dst->max) dst->max = src->max;
    dst->present += src->present;
}
static uint64_t level_bytes(uint64_t G, int32_t m, int32_t c, int coarse, int has_parent) {
    uint64_t b = tr_budget_add(tr_budget_mul(G, 8), tr_budget_mul(tr_budget_mul(G, (uint64_t)m), 24));
    b = tr_budget_add(b, tr_budget_mul(tr_budget_mul(G, (uint64_t)c), sizeof(tr_unc_input)));
    b = tr_budget_add(b, G);                                       /* tight flags */
    if (has_parent) b = tr_budget_add(b, tr_budget_mul(G, 4));
    if (coarse) b = tr_budget_add(b, tr_budget_mul(G, 8) + 8);     /* child offsets (children counted below) */
    return b;
}
/* the build checks this need first; it covers, resource by resource, the checks of the steps it runs with the same
 * budget (the UNC-001 state of scope0, the DELTA-001 state of the leaves: an upper bound of its schema and source
 * descriptors) and the temporary copy of the leaves' bank */
int32_t tr_refine_build_need(const tr_multi_hier *h, const tr_scope *scope0, tr_math_need *need) {
    tr_multi_hier_info hi;
    tr_scope_info si;
    if (!need || tr_multi_hier_get_info(h, &hi) || !scope0 || tr_scope_get_info(scope0, &si)) return TR_MATH_INVALID;
    need_zero(need);
    uint64_t states = 0, bytes = sizeof(tr_refine_state);
    const uint64_t m = (uint64_t)hi.nterms, c = (uint64_t)hi.ncols, G0 = hi.groups[0];
    for (int32_t l = 0; l < hi.levels; l++) {
        states = tr_budget_add(states, tr_budget_mul(hi.groups[l], m + c));
        bytes = tr_budget_add(bytes, level_bytes(hi.groups[l], hi.nterms, hi.ncols, l > 0, l + 1 < hi.levels));
        if (l > 0) bytes = tr_budget_add(bytes, tr_budget_mul(hi.groups[l - 1], 4));
    }
    bytes = tr_budget_add(bytes, tr_budget_add(m * c, tr_budget_mul(m, sizeof(tr_bank_term))));   /* the basis */
    tr_math_need un;
    if (hi.ncols >= 1 && !tr_unc_state_need(G0, hi.ncols, si.rows, 1, &un)) {
        bytes = tr_budget_add(bytes, un.need[TR_BUDGET_BYTES]);
        if (un.need[TR_BUDGET_STATES] > states) states = un.need[TR_BUDGET_STATES];
        need->need[TR_BUDGET_VARIABLES] = un.need[TR_BUDGET_VARIABLES];
    }
    uint64_t cells = tr_budget_add(tr_bank_state_bytes(G0, hi.nterms), tr_budget_add(m * c, tr_budget_mul(m, 16)));
    uint64_t desc = tr_budget_add(tr_budget_add(64, tr_budget_mul(c, 48)),
                                  tr_budget_add(tr_scope_key_size(scope0, TR_SCOPE_KEY_GROUPS),
                                                tr_scope_key_size(scope0, TR_SCOPE_KEY_ALL)));
    bytes = tr_budget_add(bytes, tr_budget_add(tr_budget_add(cells, desc), 4096));      /* the leaves' DELTA-001 state */
    bytes = tr_budget_add(bytes, tr_budget_add(cells, 1024));                           /* the bank copy */
    need->need[TR_BUDGET_SUPPORT] = m;
    need->need[TR_BUDGET_STATES] = states;
    need->need[TR_BUDGET_BYTES] = bytes;
    return TR_MATH_OK;
}
/* children CSR of level l (>= 1) from the parents of level l - 1 */
static int32_t build_children(tr_refine_state *s, int32_t l) {
    uint64_t G = s->groups[l], F = s->groups[l - 1];
    s->child_off[l] = zalloc((G + 1) * 8);
    s->child_ids[l] = zalloc(F * 4);
    if (!s->child_off[l] || !s->child_ids[l]) return TR_MATH_MEMORY;
    for (uint64_t f = 0; f < F; f++) { uint32_t p = s->parent[l - 1][f]; if (p != TR_MULTI_UNMAPPED) s->child_off[l][p + 1]++; }
    for (uint64_t q = 0; q < G; q++) s->child_off[l][q + 1] += s->child_off[l][q];
    uint64_t *pos = zalloc(G * 8);
    if (!pos) return TR_MATH_MEMORY;
    for (uint64_t q = 0; q < G; q++) pos[q] = s->child_off[l][q];
    for (uint64_t f = 0; f < F; f++) { uint32_t p = s->parent[l - 1][f]; if (p != TR_MULTI_UNMAPPED) s->child_ids[l][pos[p]++] = (uint32_t)f; }
    free(pos);
    return TR_MATH_OK;
}
static void state_finish(tr_refine_state *s) {
    uint64_t bytes = sizeof *s + (uint64_t)s->nterms * (uint64_t)s->ncols;
    for (int32_t l = 0; l < s->levels; l++) {
        bytes += level_bytes(s->groups[l], s->nterms, s->ncols, l > 0, l + 1 < s->levels);
        if (l > 0) bytes += s->groups[l - 1] * 4;
        uint64_t loose = 0;
        for (uint64_t g = 0; g < s->groups[l]; g++) loose += !s->tight[l][g];
        s->loose[l] = loose;
    }
    tr_delta_info di;
    if (s->leaves && !tr_delta_state_get_info(s->leaves, &di)) bytes += di.bytes;
    bytes += s->nchanges * (sizeof(tr_refine_change) + (uint64_t)s->nterms * 32);
    s->bytes = bytes;
    s->magic = STATE_MAGIC;
}
static tr_refine_state *state_alloc(int32_t levels, const uint64_t *groups, int32_t m, int32_t c, int32_t *rc) {
    tr_refine_state *s = zalloc(sizeof *s);
    if (!s) { *rc = TR_MATH_MEMORY; return NULL; }
    s->levels = levels; s->nterms = m; s->ncols = c;
    for (int32_t l = 0; l < levels; l++) {
        uint64_t G = groups[l];
        s->groups[l] = G;
        s->nrows[l] = zalloc(G * 8); s->count[l] = zalloc(G * (uint64_t)m * 8 + 8); s->sum[l] = zalloc(G * (uint64_t)m * 16 + 16);
        s->inputs[l] = zalloc(G * (uint64_t)c * sizeof(tr_unc_input) + 8); s->tight[l] = zalloc(G + 1);
        if (l + 1 < levels) s->parent[l] = zalloc(G * 4 + 4);
        if (!s->nrows[l] || !s->count[l] || !s->sum[l] || !s->inputs[l] || !s->tight[l] || (l + 1 < levels && !s->parent[l])) {
            state_release(s); *rc = TR_MATH_MEMORY; return NULL;
        }
    }
    *rc = TR_MATH_OK;
    return s;
}

tr_refine_state *tr_refine_build(const tr_multi_hier *h, const tr_scope *scope0, int32_t promise, const tr_math_budget *budget,
                                 tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    tr_multi_hier_info hi;
    tr_scope_info si;
    if (!h || tr_multi_hier_get_info(h, &hi) || !scope0 || tr_scope_get_info(scope0, &si) ||
        (promise != TR_DELTA_NONNEGATIVE && promise != TR_DELTA_SIGNED)) { set_error(error, TR_MATH_INVALID); return NULL; }
    if (hi.ncols < 1 || hi.ncols > TR_UNC_MAX_COLS || promise != TR_DELTA_NONNEGATIVE) { set_error(error, TR_MATH_UNSUPPORTED); return NULL; }
    if (si.ngroups != hi.groups[0] || si.ncols != hi.ncols) { set_error(error, TR_MATH_DOMAIN_ERROR); return NULL; }
    tr_math_need need;
    tr_refine_build_need(h, scope0, &need);
    tr_math_budget dflt;
    const tr_math_budget *b = budget_or(budget, &dflt);
    int32_t rc = tr_budget_check(b, &need, refusal);
    if (rc) { set_error(error, rc); return NULL; }
    tr_refine_state *s = state_alloc(hi.levels, hi.groups, hi.nterms, hi.ncols, &rc);
    if (!s) { set_error(error, rc); return NULL; }
    s->mode = hi.mode; s->status = hi.status; s->promise = promise; s->rows = hi.rows;
    /* the leaves' maintained state and the basis */
    tr_bank *b0 = tr_multi_hier_bank(h, 0, &rc);
    if (b0) {
        uint64_t ne = (uint64_t)hi.nterms * (uint64_t)hi.ncols;
        s->exps = zalloc(ne + 1);
        s->terms = zalloc((uint64_t)(hi.nterms > 0 ? hi.nterms : 1) * sizeof(tr_bank_term));
        int32_t deg = 0;
        if (!s->exps || !s->terms) rc = TR_MATH_MEMORY;
        if (!rc) rc = tr_bank_get_exponents(b0, s->exps, ne);
        if (!rc && tr_bank_compile(s->exps, hi.nterms, hi.ncols, s->terms, &s->joint, &deg)) rc = TR_MATH_INVALID;
        if (!rc) s->leaves = tr_delta_state_from_bank(b0, scope0, promise, b, refusal, &rc);
        tr_bank_free(b0);
    }
    if (rc) { state_release(s); set_error(error, rc); return NULL; }
    /* the cells of every level and the maps */
    for (int32_t l = 0; l < hi.levels && !rc; l++) {
        tr_bank_states st;
        rc = tr_multi_hier_states(h, l, &st);
        if (rc) break;
        uint64_t G = st.ngroups, gm = G * (uint64_t)hi.nterms;
        memcpy(s->nrows[l], st.rows, (size_t)G * 8);
        if (gm) { memcpy(s->count[l], st.count, (size_t)gm * 8); memcpy(s->sum[l], st.sum, (size_t)gm * 16); }
        memset(s->tight[l], 1, (size_t)G);
        if (l + 1 < hi.levels) {
            tr_multi_map *m = tr_multi_hier_map(h, l, &rc);
            if (m) { rc = tr_multi_map_get(m, s->parent[l], G); tr_multi_map_free(m); }
        }
    }
    /* the UNC-001 inputs of the leaves (from the scope; logical units), checked against the cells, rolled up */
    if (!rc) {
        int32_t cols[TR_UNC_MAX_COLS];
        for (int32_t j = 0; j < hi.ncols; j++) cols[j] = j;
        tr_unc_state *us = tr_unc_state_from_scope(scope0, cols, hi.ncols, b, refusal, &rc);
        if (us) {
            const tr_unc_input *in = tr_unc_state_inputs(us);
            uint64_t n = hi.groups[0] * (uint64_t)hi.ncols;
            memcpy(s->inputs[0], in, (size_t)n * sizeof *in);
            for (uint64_t i = 0; i < n; i++) s->inputs[0][i].scale = 0;
            for (uint64_t g = 0; g < hi.groups[0] && !rc; g++)
                if (s->inputs[0][g * (uint64_t)hi.ncols].rows != s->nrows[0][g]) rc = TR_MATH_DOMAIN_ERROR;   /* another scope */
            tr_unc_state_free(us);
        }
    }
    for (int32_t l = 1; l < hi.levels && !rc; l++) {
        uint64_t c = (uint64_t)hi.ncols;
        for (uint64_t f = 0; f < hi.groups[l - 1]; f++) {
            uint32_t p = s->parent[l - 1][f];
            if (p == TR_MULTI_UNMAPPED) continue;
            for (uint64_t j = 0; j < c; j++) input_merge(&s->inputs[l][p * c + j], &s->inputs[l - 1][f * c + j]);
        }
        rc = build_children(s, l);
    }
    if (rc) { state_release(s); set_error(error, rc); return NULL; }
    s->strategy = 0; s->reason = TR_REFINE_REASON_NONE;
    state_finish(s);
    set_error(error, TR_MATH_OK);
    return s;
}

int32_t tr_refine_get_info(const tr_refine_state *s, tr_refine_info *out) {
    if (!state_valid(s) || !out) return TR_MATH_INVALID;
    memset(out, 0, sizeof *out);
    out->levels = s->levels; out->ncols = s->ncols; out->nterms = s->nterms; out->mode = s->mode;
    out->status = s->status; out->promise = s->promise; out->strategy = s->strategy; out->reason = s->reason;
    for (int32_t l = 0; l < s->levels; l++) {
        out->groups[l] = s->groups[l];
        uint64_t p = 0;
        for (uint64_t g = 0; g < s->groups[l]; g++) p += s->nrows[l][g] != 0;
        out->present[l] = p;
        out->level_bytes[l] = level_bytes(s->groups[l], s->nterms, s->ncols, l > 0, l + 1 < s->levels) + (l > 0 ? s->groups[l - 1] * 4 : 0);
        out->changed[l] = s->changed[l]; out->loose[l] = s->loose[l];
    }
    out->bytes = s->bytes; out->rows = s->rows; out->step = s->step; out->touched = s->touched;
    out->work = s->work; out->rebuild_work = s->rebuild_work;
    tr_delta_info di;
    if (s->leaves && !tr_delta_state_get_info(s->leaves, &di)) { out->identity[0] = di.identity[0]; out->identity[1] = di.identity[1]; }
    return TR_MATH_OK;
}
int32_t tr_refine_get_level(const tr_refine_state *s, int32_t level, tr_refine_level_view *out) {
    if (!state_valid(s) || !out || level < 0 || level >= s->levels) return TR_MATH_INVALID;
    out->ngroups = s->groups[level]; out->nterms = s->nterms; out->ncols = s->ncols;
    out->rows = s->nrows[level]; out->count = s->count[level]; out->sum = s->sum[level]; out->inputs = s->inputs[level];
    return TR_MATH_OK;
}
int32_t tr_refine_get_links(const tr_refine_state *s, int32_t level, const uint32_t **parent, const uint64_t **child_off,
                            const uint32_t **child_ids) {
    if (!state_valid(s) || level < 0 || level >= s->levels) return TR_MATH_INVALID;
    if (parent) *parent = level + 1 < s->levels ? s->parent[level] : NULL;
    if (child_off) *child_off = level > 0 ? s->child_off[level] : NULL;
    if (child_ids) *child_ids = level > 0 ? s->child_ids[level] : NULL;
    return TR_MATH_OK;
}
int32_t tr_refine_tight(const tr_refine_state *s, int32_t level, uint64_t g) {
    if (!state_valid(s) || level < 0 || level >= s->levels || g >= s->groups[level]) return TR_MATH_INVALID;
    return s->tight[level][g] ? 1 : 0;
}

/* ------------------------------------------------------------------ plans */
static int pnode(tr_unc_node *nodes, int32_t *n, int32_t cap, int32_t op, int32_t a, int32_t b, tr_num_rational k) {
    if (*n >= cap) return -1;
    tr_unc_node *nd = &nodes[*n];
    nd->op = op; nd->a = a; nd->b = b; nd->reserved = 0; nd->k = k;
    return (*n)++;
}
/* the UNC-001 program of a class: sum_i c_i term_i (SUM atoms) + sum of the COUNT / ROWS coefficients */
static int32_t class_program(const tr_refine_state *s, const tr_refine_atom *atoms, int32_t natoms, tr_unc_node *nodes,
                             int32_t cap,
                             int32_t *nnodes, int32_t *root) {
    int32_t n = 0, colnode[TR_UNC_MAX_COLS], acc = -1;
    tr_num_rational z = tr_q_int(0);
    __int128 k = 0;
    for (int32_t j = 0; j < TR_UNC_MAX_COLS; j++) colnode[j] = -1;
    for (int32_t i = 0; i < natoms; i++) {
        const tr_refine_atom *a = &atoms[i];
        if (a->kind != TR_REFINE_SUM || !s->terms[a->term].nf) { k += a->coef; continue; }   /* 1 per valid row */
        const tr_bank_term *t = &s->terms[a->term];
        int32_t mono = -1;
        for (int f = 0; f < t->nf; f++) {
            int32_t c = t->col[f];
            if (colnode[c] < 0) colnode[c] = pnode(nodes, &n, cap, TR_UNC_COL, c, -1, z);
            for (int e = 0; e < t->exp[f]; e++) mono = mono < 0 ? colnode[c] : pnode(nodes, &n, cap, TR_UNC_MUL, mono, colnode[c], z);
            if (mono < 0 || colnode[c] < 0) return TR_MATH_UNSUPPORTED;
        }
        int32_t term = mono;
        if (a->coef != 1) {
            int32_t cn = pnode(nodes, &n, cap, TR_UNC_CONST, -1, -1, tr_q_int(a->coef));
            term = cn < 0 ? -1 : pnode(nodes, &n, cap, TR_UNC_MUL, mono, cn, z);
        }
        acc = term < 0 ? -1 : (acc < 0 ? term : pnode(nodes, &n, cap, TR_UNC_ADD, acc, term, z));
        if (acc < 0) return TR_MATH_UNSUPPORTED;
    }
    if (k > INT64_MAX || k < INT64_MIN) return TR_MATH_UNSUPPORTED;
    if (k || acc < 0) {
        int32_t cn = pnode(nodes, &n, cap, TR_UNC_CONST, -1, -1, tr_q_int((int64_t)k));
        acc = cn < 0 ? -1 : (acc < 0 ? cn : pnode(nodes, &n, cap, TR_UNC_ADD, acc, cn, z));
        if (acc < 0) return TR_MATH_UNSUPPORTED;
    }
    *nnodes = n; *root = acc;
    return TR_MATH_OK;
}
/* the validity class of an atom: SQL: its term's column set (ROWS and the constant term: every row); listwise: one
 * class for the terms, one for ROWS */
static int64_t atom_class(const tr_refine_state *s, const tr_refine_atom *a) {
    if (a->kind == TR_REFINE_ROWS) return s->mode == TR_BANK_JOINT ? -1 : 0;
    if (s->mode == TR_BANK_JOINT) return -2;
    return (int64_t)s->terms[a->term].need;
}
static int32_t plan_aggregate(tr_refine_query *q, int32_t agg, const tr_refine_atom *atoms, int32_t natoms,
                              tr_unc_node *nodes, int32_t *nodes_used) {
    const tr_refine_state *s = q->s;
    tr_refine_plan *pl = &q->plan;
    pl->first[agg] = pl->nclass; pl->nclasses[agg] = 0; pl->merged[agg] = -1;
    int64_t keys[2 * TR_REFINE_MAX_ATOMS];
    int32_t nk = 0;
    for (int32_t i = 0; i < natoms; i++) {
        int64_t key = atom_class(s, &atoms[i]);
        int seen = 0;
        for (int32_t j = 0; j < nk; j++) seen |= keys[j] == key;
        if (!seen) keys[nk++] = key;
    }
    for (int32_t pass = 0; pass <= nk; pass++) {       /* the classes, then the merged class (every atom) */
        if (pl->nclass >= TR_REFINE_MAX_CLASSES) return TR_MATH_UNSUPPORTED;
        tr_refine_class *cl = &q->classes[pl->nclass];
        memset(cl, 0, sizeof *cl);
        cl->first_atom = pl->natoms; cl->count_term = -1; cl->merged = pass == nk;
        for (int32_t i = 0; i < natoms; i++) {
            if (pass < nk && atom_class(s, &atoms[i]) != keys[pass]) continue;
            if (pl->natoms >= TR_REFINE_MAX_PLAN_ATOMS) return TR_MATH_UNSUPPORTED;
            q->atoms[pl->natoms++] = atoms[i];
            cl->natoms++;
            if (pass < nk && atoms[i].kind != TR_REFINE_ROWS && cl->count_term < 0) cl->count_term = atoms[i].term;
        }
        int32_t nn = 0, root = -1;
        int32_t room = TR_REFINE_MAX_NODES - *nodes_used;         /* never past the plan's node array */
        int32_t rc = class_program(s, &q->atoms[cl->first_atom], cl->natoms, nodes + *nodes_used,
                                   room < TR_UNC_MAX_NODES ? room : TR_UNC_MAX_NODES, &nn, &root);
        if (rc) return rc;
        if (*nodes_used + nn > TR_REFINE_MAX_NODES) return TR_MATH_UNSUPPORTED;
        tr_unc_program p;
        memset(&p, 0, sizeof p);
        p.nnodes = nn; p.ncols = s->ncols; p.carrier = TR_UNC_EXACT; p.precision = 0; p.rank = TR_UNC_MAX_TERMS;
        p.domains = TR_UNC_DOM_ALL; p.root = root; p.budget = tr_unc_qv(0, 0); p.nodes = nodes + *nodes_used;
        tr_unc_plan plan;
        rc = tr_unc_check(&p, 0, &plan, NULL);
        if (rc) return rc;
        cl->first_node = *nodes_used; cl->nnodes = nn; cl->root = root; cl->nlive = plan.nlive;
        if (plan.nlive > pl->nlive) pl->nlive = plan.nlive;
        *nodes_used += nn;
        if (cl->merged) pl->merged[agg] = pl->nclass;
        pl->nclass++; pl->nclasses[agg]++;
    }
    return TR_MATH_OK;
}
static int32_t request_check(const tr_refine_state *s, const tr_refine_request *r) {
    if (r->kind < TR_REFINE_THRESHOLD || r->kind > TR_REFINE_TOPK || r->level < 0 || r->level >= s->levels || r->reserved ||
        r->natoms < 1 || (r->kind == TR_REFINE_RATIO) != (r->ndatoms > 0)) return TR_MATH_INVALID;
    if (r->natoms > TR_REFINE_MAX_ATOMS || r->ndatoms > TR_REFINE_MAX_ATOMS) return TR_MATH_UNSUPPORTED;
    if (r->kind != TR_REFINE_TOPK && (r->op < TR_REFINE_GT || r->op > TR_REFINE_LE || !tr_unc_canonical(r->threshold)))
        return TR_MATH_INVALID;
    if (r->kind == TR_REFINE_TOPK && (r->k < 1 || (r->descending != 0 && r->descending != 1))) return TR_MATH_INVALID;
    for (int32_t i = 0; i < r->natoms + r->ndatoms; i++) {
        const tr_refine_atom *a = i < r->natoms ? &r->atoms[i] : &r->datoms[i - r->natoms];
        if (a->kind < TR_REFINE_SUM || a->kind > TR_REFINE_ROWS || !a->coef ||
            (a->kind != TR_REFINE_ROWS && (a->term < 0 || a->term >= s->nterms))) return TR_MATH_INVALID;
    }
    return TR_MATH_OK;
}
static void query_release(tr_refine_query *q) {
    if (!q) return;
    for (int32_t l = 0; l < L; l++) free(q->dcnt[l]);
    free(q->nodes); free(q->wave); free(q->next); free(q->bounds); free(q->entries); free(q->scratch);
    q->magic = 0;
    free(q);
}
void tr_refine_query_free(tr_refine_query *q) { if (query_valid(q)) query_release(q); }
/* floor / ceil of an int64 rational */
static __int128 qfloor(tr_num_rational t) { return tr_unc_floordiv(t.numerator, t.denominator); }
static __int128 qceil(tr_num_rational t) { return -tr_unc_floordiv(-(__int128)t.numerator, t.denominator); }
/* the compiled plan of a request (shared by queries and the fine baseline) */
static int32_t compile(tr_refine_query *q, const tr_refine_state *s, const tr_refine_request *r) {
    q->s = s; q->req = *r;
    int32_t rc = request_check(s, r);
    if (rc) return rc;
    q->nodes = zalloc(TR_REFINE_MAX_NODES * sizeof(tr_unc_node));
    if (!q->nodes) return TR_MATH_MEMORY;
    tr_refine_plan *pl = &q->plan;
    memset(pl, 0, sizeof *pl);
    pl->ncols = s->ncols; pl->nterms = s->nterms; pl->mode = s->mode;
    tr_refine_atom a0[2 * TR_REFINE_MAX_ATOMS];
    int32_t n0 = 0, used = 0;
    if (r->kind == TR_REFINE_RATIO) {                  /* C = q A - p D with T = p / q */
        for (int32_t i = 0; i < r->natoms; i++) {
            a0[n0] = r->atoms[i];
            __int128 v = (__int128)r->atoms[i].coef * r->threshold.denominator;
            if (v > INT64_MAX || v < INT64_MIN) return TR_MATH_UNSUPPORTED;
            a0[n0++].coef = (int64_t)v;
        }
        for (int32_t i = 0; i < r->ndatoms; i++) {
            a0[n0] = r->datoms[i];
            __int128 v = -(__int128)r->datoms[i].coef * r->threshold.numerator;
            if (v > INT64_MAX || v < INT64_MIN) return TR_MATH_UNSUPPORTED;
            if (!v) continue;                          /* T = 0: C = q A */
            a0[n0++].coef = (int64_t)v;
        }
    } else for (int32_t i = 0; i < r->natoms; i++) a0[n0++] = r->atoms[i];
    pl->naggs = r->kind == TR_REFINE_RATIO ? 2 : 1;
    rc = plan_aggregate(q, 0, a0, n0, q->nodes, &used);
    if (!rc && r->kind == TR_REFINE_RATIO) rc = plan_aggregate(q, 1, r->datoms, r->ndatoms, q->nodes, &used);
    if (rc) return rc;
    pl->nnodes = used; pl->atoms = q->atoms; pl->classes = q->classes; pl->nodes = q->nodes;
    /* the integer form of the predicate on A (THRESHOLD) or C (RATIO, D > 0) */
    int32_t op = r->op;
    if (r->kind == TR_REFINE_THRESHOLD) {
        q->ge = op == TR_REFINE_GT || op == TR_REFINE_GE;
        q->thr = op == TR_REFINE_GT ? qfloor(r->threshold) + 1 : op == TR_REFINE_GE ? qceil(r->threshold) :
                 op == TR_REFINE_LT ? qceil(r->threshold) - 1 : qfloor(r->threshold);
    }
    return TR_MATH_OK;
}
static int32_t query_need(const tr_refine_state *s, const tr_refine_query *q, tr_math_need *need) {
    need_zero(need);
    uint64_t entries = 0;
    for (int32_t l = q->req.level; l < s->levels; l++) entries = tr_budget_add(entries, s->groups[l]);
    uint64_t gmax = 0;
    for (int32_t l = 0; l < s->levels; l++) if (s->groups[l] > gmax) gmax = s->groups[l];
    need->need[TR_BUDGET_SUPPORT] = (uint64_t)q->plan.natoms;
    need->need[TR_BUDGET_CIRCUIT] = (uint64_t)q->plan.nnodes;
    need->need[TR_BUDGET_OUTPUT] = entries;
    uint64_t most = 0;                             /* every present region above the target expanded */
    for (int32_t l = q->req.level + 1; l < s->levels; l++)
        for (uint64_t g = 0; g < s->groups[l]; g++) most += s->nrows[l][g] != 0;
    need->need[TR_BUDGET_ITERATIONS] = q->req.expand_limit && q->req.expand_limit < most ? q->req.expand_limit : most;
    uint64_t bytes = tr_budget_add(sizeof *q, TR_REFINE_MAX_NODES * sizeof(tr_unc_node));
    bytes = tr_budget_add(bytes, tr_budget_mul(entries, sizeof(tr_refine_entry)));
    bytes = tr_budget_add(bytes, tr_budget_mul(gmax, 8 + sizeof(tr_refine_bound)));
    bytes = tr_budget_add(bytes, tr_budget_mul((uint64_t)q->plan.nlive + 1, sizeof(tr_unc_value)));
    if (q->req.kind == TR_REFINE_TOPK) bytes = tr_budget_add(bytes, tr_budget_mul(entries, 8));   /* descendant counts */
    need->need[TR_BUDGET_BYTES] = bytes;
    return TR_MATH_OK;
}
static int32_t wave_reserve(tr_refine_query *q, uint64_t n) {
    if (n <= q->capwave) return TR_MATH_OK;
    uint32_t *w = realloc(q->wave, (size_t)n * 4), *x = NULL;
    if (w) q->wave = w;
    x = realloc(q->next, (size_t)n * 4);
    if (x) q->next = x;
    tr_refine_bound *b = realloc(q->bounds, (size_t)n * sizeof *b);
    if (b) q->bounds = b;
    if (!w || !x || !b) return TR_MATH_MEMORY;
    q->capwave = n;
    return TR_MATH_OK;
}
static int32_t entry_push(tr_refine_query *q, const tr_refine_bound *b, int32_t level, uint32_t g, int32_t decision, uint32_t rank) {
    if (q->nentries == q->capentries) {
        uint64_t cap = q->capentries ? 2 * q->capentries : 64;
        tr_refine_entry *e = realloc(q->entries, (size_t)cap * sizeof *e);
        if (!e) return TR_MATH_MEMORY;
        q->entries = e; q->capentries = cap;
    }
    tr_refine_entry *e = &q->entries[q->nentries++];
    memset(e, 0, sizeof *e);
    e->level = level; e->decision = decision; e->group = g; e->rank = rank; e->rows = q->s->nrows[level][g];
    if (b) {
        e->flags = b->flags;
        memcpy(e->lo, b->lo, 32); memcpy(e->hi, b->hi, 32); memcpy(e->dlo, b->dlo, 32); memcpy(e->dhi, b->dhi, 32);
    }
    if (decision == TR_REFINE_UNRESOLVED) q->stats.unresolved++;
    return TR_MATH_OK;
}
tr_refine_query *tr_refine_begin(const tr_refine_state *s, const tr_refine_request *r, const tr_math_budget *budget,
                                 tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    if (!state_valid(s) || !r) { set_error(error, TR_MATH_INVALID); return NULL; }
    tr_refine_query *q = zalloc(sizeof *q);
    if (!q) { set_error(error, TR_MATH_MEMORY); return NULL; }
    int32_t rc = compile(q, s, r);
    tr_math_need need;
    tr_math_budget dflt;
    if (!rc) { query_need(s, q, &need); rc = tr_budget_check(budget_or(budget, &dflt), &need, refusal); }
    if (!rc) { q->scratch = zalloc(((uint64_t)q->plan.nlive + 1) * sizeof(tr_unc_value)); if (!q->scratch) rc = TR_MATH_MEMORY; }
    if (!rc && r->kind == TR_REFINE_TOPK) {            /* the present target descendants of every region, bottom up */
        for (int32_t l = r->level; l < s->levels && !rc; l++) {
            q->dcnt[l] = zalloc(s->groups[l] * 8 + 8);
            if (!q->dcnt[l]) { rc = TR_MATH_MEMORY; break; }
            if (l == r->level) for (uint64_t g = 0; g < s->groups[l]; g++) q->dcnt[l][g] = s->nrows[l][g] != 0;
            else for (uint64_t f = 0; f < s->groups[l - 1]; f++) {
                uint32_t pg = s->parent[l - 1][f];
                if (pg != TR_MULTI_UNMAPPED) q->dcnt[l][pg] += q->dcnt[l - 1][f];
            }
        }
        for (int32_t l = r->level; l < s->levels && !rc; l++) q->counted += s->groups[l];
        q->stats.enumerated = q->counted;
    }
    /* the first wave: the present groups of the coarsest level (the target's own groups when it is the coarsest) */
    int32_t top = s->levels - 1;
    uint64_t gmax = 1;                                 /* a wave holds at most one level's groups */
    for (int32_t l = 0; l < s->levels; l++) if (s->groups[l] > gmax) gmax = s->groups[l];
    if (!rc) rc = wave_reserve(q, gmax);
    if (!rc) {
        q->level = top; q->exact = top == r->level; q->nwave = 0;
        for (uint64_t g = 0; g < s->groups[top]; g++) if (s->nrows[top][g]) q->wave[q->nwave++] = (uint32_t)g;
        q->stats.target = r->level; q->stats.levels = s->levels;
        for (uint64_t g = 0; g < s->groups[r->level]; g++) q->stats.fine_work += s->nrows[r->level][g] ? (uint64_t)s->nterms + 1 : 0;
        if (!q->nwave) q->done = 1;
    }
    if (rc) { query_release(q); set_error(error, rc); return NULL; }
    q->magic = QUERY_MAGIC;
    set_error(error, TR_MATH_OK);
    return q;
}
int32_t tr_refine_wave(const tr_refine_query *q, int32_t *level, int32_t *exact, const uint32_t **groups, uint64_t *n) {
    if (!query_valid(q) || !level || !exact || !groups || !n) return TR_MATH_INVALID;
    *level = q->level; *exact = q->exact; *groups = q->wave; *n = q->done ? 0 : q->nwave;
    return q->done ? 0 : 1;
}
int32_t tr_refine_get_plan(const tr_refine_query *q, tr_refine_plan *out) {
    if (!query_valid(q) || !out) return TR_MATH_INVALID;
    *out = q->plan;
    return TR_MATH_OK;
}

/* ---- decisions */
static tr_refine_x bx(const uint64_t w[4], int32_t inf) { tr_refine_x r; memcpy(r.v.w, w, 32); r.inf = inf; return r; }
static tr_refine_x xint(__int128 v) { return tr_refine_x_of(tr_refine_w_i128(v), 0); }
static int ge_thr(tr_refine_x a, __int128 t) { return tr_refine_x_cmp(a, xint(t)) >= 0; }
static int le_thr(tr_refine_x a, __int128 t) { return tr_refine_x_cmp(a, xint(t)) <= 0; }
/* the integer form of "C op 0" (RATIO): ge and threshold */
static void op_form(int32_t op, int flip, int *ge, __int128 *t) {
    if (flip) op = op == TR_REFINE_GT ? TR_REFINE_LT : op == TR_REFINE_GE ? TR_REFINE_LE : op == TR_REFINE_LT ? TR_REFINE_GT : TR_REFINE_GE;
    *ge = op == TR_REFINE_GT || op == TR_REFINE_GE;
    *t = op == TR_REFINE_GT ? 1 : op == TR_REFINE_GE ? 0 : op == TR_REFINE_LT ? -1 : 0;
}
/* a region's decision: ALL, NONE, or 0 (undecided) */
static int32_t decide_region(const tr_refine_query *q, const tr_refine_bound *b) {
    tr_refine_x lo = bx(b->lo, (b->flags & TR_REFINE_LO_INF) ? -1 : 0), hi = bx(b->hi, (b->flags & TR_REFINE_HI_INF) ? 1 : 0);
    int ge = q->ge;
    __int128 t = q->thr;
    if (q->req.kind == TR_REFINE_RATIO) {
        tr_refine_x dlo = bx(b->dlo, (b->flags & TR_REFINE_DLO_INF) ? -1 : 0), dhi = bx(b->dhi, (b->flags & TR_REFINE_DHI_INF) ? 1 : 0);
        if (tr_refine_x_cmp(dlo, xint(0)) > 0) op_form(q->req.op, 0, &ge, &t);
        else if (tr_refine_x_cmp(dhi, xint(0)) < 0) op_form(q->req.op, 1, &ge, &t);
        else if (!dlo.inf && !dhi.inf && tr_refine_x_cmp(dlo, xint(0)) == 0 && tr_refine_x_cmp(dhi, xint(0)) == 0) return TR_REFINE_NONE;
        else return 0;                                 /* the sign of D is not fixed in the region */
    }
    if (ge) { if (ge_thr(lo, t)) return TR_REFINE_ALL; if (!ge_thr(hi, t)) return TR_REFINE_NONE; }
    else { if (le_thr(hi, t)) return TR_REFINE_ALL; if (!le_thr(lo, t)) return TR_REFINE_NONE; }
    return 0;
}
/* an exact target group: TRUE, FALSE or UNDEFINED */
static int32_t decide_exact(const tr_refine_query *q, const tr_refine_bound *b) {
    tr_refine_x v = bx(b->lo, 0);
    int ge = q->ge;
    __int128 t = q->thr;
    if (q->req.kind == TR_REFINE_RATIO) {
        tr_refine_x d = bx(b->dlo, 0);
        int sd = tr_refine_x_cmp(d, xint(0));
        if (!sd) return TR_REFINE_UNDEFINED;
        op_form(q->req.op, sd < 0, &ge, &t);
    }
    return (ge ? ge_thr(v, t) : le_thr(v, t)) ? TR_REFINE_TRUE : TR_REFINE_FALSE;
}
/* TOPK scores: the bound of A (descending) or of -A (ascending) */
static void score(const tr_refine_query *q, const tr_refine_bound *b, tr_refine_x *slo, tr_refine_x *shi) {
    tr_refine_x lo = bx(b->lo, (b->flags & TR_REFINE_LO_INF) ? -1 : 0), hi = bx(b->hi, (b->flags & TR_REFINE_HI_INF) ? 1 : 0);
    if (q->req.descending) { *slo = lo; *shi = hi; return; }
    slo->v = tr_refine_w_neg(hi.v); slo->inf = -hi.inf;
    shi->v = tr_refine_w_neg(lo.v); shi->inf = -lo.inf;
}
static int cmp_x(const void *a, const void *b) { return -tr_refine_x_cmp(*(const tr_refine_x *)a, *(const tr_refine_x *)b); }
/* floor(a / d) of a 256-bit value and d >= 1 */
static tr_refine_w wfloordiv(tr_refine_w a, uint64_t d) {
    int neg = tr_refine_w_neg_p(a);
    tr_refine_w m = neg ? tr_refine_w_neg(a) : a, q;
    unsigned __int128 r = 0;
    for (int i = 3; i >= 0; i--) { unsigned __int128 cur = (r << 64) | m.w[i]; q.w[i] = (uint64_t)(cur / d); r = cur % d; }
    if (!neg) return q;
    if (r) q = tr_refine_w_add(q, tr_refine_w_i128(1));      /* -floor(-x) = ceil: floor of a negative quotient */
    return tr_refine_w_neg(q);
}
/* TOPK: a sound lower bound of the best score among region g's present target descendants: A is linear, so the
 * region's own exact aggregate is the sum of its c descendants' values and the largest is at least its mean (the
 * smallest at most): floor(A(R) / c) descending, floor(-A(R) / c) ascending; -inf at the target or without counts */
static tr_refine_x witness(const tr_refine_query *q, const tr_refine_level_view *lv, uint32_t g) {
    tr_refine_x none = tr_refine_x_of(tr_refine_w_i128(0), -1);
    if (!q->dcnt[q->level] || !q->dcnt[q->level][g]) return none;
    tr_refine_x a, b;
    int32_t merged;
    tr_refine_agg_bound(&q->plan, 0, lv, g, 1, NULL, &a, &b, &merged);
    tr_refine_w v = q->req.descending ? a.v : tr_refine_w_neg(a.v);
    return tr_refine_x_of(wfloordiv(v, q->dcnt[q->level][g]), 0);
}
/* the k-th largest of the score lower bounds of the wave and of the unresolved entries (-inf with fewer) */
static int32_t topk_tau(tr_refine_query *q, tr_refine_x *tau) {
    uint64_t n = q->nwave;
    for (uint64_t e = 0; e < q->nentries; e++) n += q->entries[e].decision == TR_REFINE_UNRESOLVED;
    tau->inf = -1; tau->v = tr_refine_w_i128(0);
    if (n < q->req.k) return TR_MATH_OK;
    tr_refine_x *v = malloc((size_t)n * sizeof *v);
    if (!v) return TR_MATH_MEMORY;
    uint64_t m = 0;
    tr_refine_level_view lv;
    tr_refine_get_level(q->s, q->level, &lv);
    for (uint64_t i = 0; i < q->nwave; i++) {
        tr_refine_x lo, hi;
        score(q, &q->bounds[i], &lo, &hi);
        tr_refine_x w = witness(q, &lv, q->wave[i]);
        v[m++] = tr_refine_x_cmp(w, lo) > 0 ? w : lo;
    }
    for (uint64_t e = 0; e < q->nentries; e++) {
        if (q->entries[e].decision != TR_REFINE_UNRESOLVED) continue;
        tr_refine_bound b;
        memset(&b, 0, sizeof b);
        b.flags = q->entries[e].flags; memcpy(b.lo, q->entries[e].lo, 32); memcpy(b.hi, q->entries[e].hi, 32);
        tr_refine_x lo, hi;
        score(q, &b, &lo, &hi);
        v[m++] = lo;
    }
    qsort(v, (size_t)m, sizeof *v, cmp_x);
    *tau = v[q->req.k - 1];
    free(v);
    return TR_MATH_OK;
}
typedef struct { tr_refine_x s; uint32_t g; uint32_t i; } exact_cand;
static int cmp_cand(const void *a, const void *b) {
    const exact_cand *x = a, *y = b;
    int c = tr_refine_x_cmp(x->s, y->s);
    if (c) return -c;
    return x->g < y->g ? -1 : x->g > y->g;
}
static int cmp_u32(const void *a, const void *b) { uint32_t x = *(const uint32_t *)a, y = *(const uint32_t *)b; return x < y ? -1 : x > y; }

static int32_t topk_final(tr_refine_query *q) {
    uint64_t n = q->nwave;
    exact_cand *c = malloc((size_t)(n ? n : 1) * sizeof *c);
    if (!c) return TR_MATH_MEMORY;
    for (uint64_t i = 0; i < n; i++) { tr_refine_x hi; score(q, &q->bounds[i], &c[i].s, &hi); c[i].g = q->wave[i]; c[i].i = (uint32_t)i; }
    qsort(c, (size_t)n, sizeof *c, cmp_cand);
    int32_t rc = TR_MATH_OK;
    uint64_t nent = q->nentries;
    for (uint64_t i = 0; i < n && !rc; i++) {
        uint64_t certain = 0, possible = 0;          /* better descendants of unresolved regions: at least / at most */
        for (uint64_t e = 0; e < nent; e++) {
            const tr_refine_entry *u = &q->entries[e];
            if (u->decision != TR_REFINE_UNRESOLVED) continue;
            tr_refine_bound b;
            memset(&b, 0, sizeof b);
            b.flags = u->flags; memcpy(b.lo, u->lo, 32); memcpy(b.hi, u->hi, 32);
            tr_refine_x lo, hi;
            score(q, &b, &lo, &hi);
            if (tr_refine_x_cmp(lo, c[i].s) > 0) certain++;
            if (tr_refine_x_cmp(hi, c[i].s) >= 0)     /* at most its present target descendants */
                possible += q->dcnt[u->level] ? q->dcnt[u->level][u->group] : u->rows;
        }
        uint64_t rmin = i + certain + 1, rmax = i + possible + 1;
        int32_t d = rmax <= q->req.k ? TR_REFINE_IN : (rmin > q->req.k ? TR_REFINE_OUT : TR_REFINE_UNRESOLVED);
        rc = entry_push(q, &q->bounds[c[i].i], q->level, c[i].g, d, d == TR_REFINE_IN ? (uint32_t)rmin : 0);
    }
    free(c);
    return rc;
}

int32_t tr_refine_supply(tr_refine_query *q, const tr_refine_bound *bounds, uint64_t n) {
    if (!query_valid(q) || q->done || n != q->nwave || (n && !bounds)) return TR_MATH_INVALID;
    const tr_refine_state *s = q->s;
    for (uint64_t i = 0; i < n; i++)                    /* the bounds of exactly this wave */
        if (bounds[i].group != q->wave[i] || bounds[i].level != (uint32_t)q->level ||
            ((bounds[i].flags & TR_REFINE_EXACT) != 0) != (q->exact != 0)) return TR_MATH_INVALID;
    if (bounds != q->bounds) memcpy(q->bounds, bounds, (size_t)n * sizeof *bounds);
    q->stats.evaluated[q->level] += n;
    q->stats.work += n * ((uint64_t)s->nterms + 1 + (q->exact ? 0 : (uint64_t)s->ncols));
    q->stats.waves++;
    int32_t rc = TR_MATH_OK;
    tr_refine_x tau;
    tau.inf = -1; tau.v = tr_refine_w_i128(0);
    if (q->req.kind == TR_REFINE_TOPK && !q->exact) rc = topk_tau(q, &tau);
    if (!rc && q->exact) {                             /* the target level: exact groups */
        if (q->req.kind == TR_REFINE_TOPK) rc = topk_final(q);
        else for (uint64_t i = 0; i < n && !rc; i++) rc = entry_push(q, &q->bounds[i], q->level, q->wave[i], decide_exact(q, &q->bounds[i]), 0);
        q->done = 1;
    } else if (!rc) {
        q->nnext = 0;
        for (uint64_t i = 0; i < n && !rc; i++) {
            const tr_refine_bound *b = &q->bounds[i];
            uint32_t g = q->wave[i];
            if (q->req.kind == TR_REFINE_TOPK) {
                tr_refine_x lo, hi;
                score(q, b, &lo, &hi);
                if (tr_refine_x_cmp(hi, tau) < 0) { rc = entry_push(q, b, q->level, g, TR_REFINE_OUT, 0); continue; }
            } else {
                int32_t d = decide_region(q, b);
                if (d) { rc = entry_push(q, b, q->level, g, d, 0); continue; }
            }
            if (q->req.expand_limit && q->expansions >= q->req.expand_limit) {
                rc = entry_push(q, b, q->level, g, TR_REFINE_UNRESOLVED, 0); continue;
            }
            q->expansions++;
            q->stats.expanded[q->level]++;
            const uint64_t *off = s->child_off[q->level];
            const uint32_t *ids = s->child_ids[q->level];
            for (uint64_t k = off[g]; k < off[g + 1]; k++)
                if (s->nrows[q->level - 1][ids[k]]) q->next[q->nnext++] = ids[k];
        }
        if (!rc) {
            qsort(q->next, (size_t)q->nnext, 4, cmp_u32);
            uint32_t *t = q->wave; q->wave = q->next; q->next = t;
            q->nwave = q->nnext; q->level--; q->exact = q->level == q->req.level;
            if (!q->nwave) q->done = 1;
        }
    }
    if (!rc && q->done) {
        uint64_t unres = 0;
        for (uint64_t e = 0; e < q->nentries; e++) unres += q->entries[e].decision == TR_REFINE_UNRESOLVED;
        q->stats.complete = unres == 0;
        q->stats.entries = q->nentries;
    }
    return rc;
}
int32_t tr_refine_query_need(const tr_refine_state *s, const tr_refine_request *r, tr_math_need *need) {
    if (!state_valid(s) || !r || !need) return TR_MATH_INVALID;
    tr_refine_query q;
    memset(&q, 0, sizeof q);
    int32_t rc = compile(&q, s, r);
    if (!rc) query_need(s, &q, need);
    free(q.nodes);
    return rc;
}
int32_t tr_refine_bounds_need(const tr_refine_level_view *lv, const tr_refine_plan *pl, uint64_t n, tr_math_need *need) {
    if (!lv || !pl || !need) return TR_MATH_INVALID;
    need_zero(need);
    need->need[TR_BUDGET_OUTPUT] = n;
    need->need[TR_BUDGET_BYTES] = tr_refine_bounds_device_bytes(lv, pl, n);
    return TR_MATH_OK;
}
int32_t tr_refine_rollup_need(const tr_refine_update_view *u, uint64_t nchild, tr_math_need *need) {
    if (!u || !need || u->levels < 1 || u->levels > L) return TR_MATH_INVALID;
    need_zero(need);
    uint64_t total = 0;
    for (int32_t l = 0; l < u->levels; l++) total = tr_budget_add(total, u->naff[l]);
    need->need[TR_BUDGET_STATES] = tr_budget_mul(total, (uint64_t)u->nterms + (uint64_t)u->ncols);
    need->need[TR_BUDGET_BYTES] = tr_refine_rollup_device_bytes(u, nchild);
    return TR_MATH_OK;
}
int32_t tr_refine_eval_bounds(const tr_refine_level_view *lv, const tr_refine_plan *pl, int32_t level, int32_t exact,
                              const uint32_t *groups, uint64_t n, tr_refine_bound *out) {
    if (!lv || !pl || (n && (!groups || !out)) || pl->nlive < 0) return TR_MATH_INVALID;
    for (uint64_t i = 0; i < n; i++) if (groups[i] >= lv->ngroups) return TR_MATH_INVALID;
    tr_unc_value *scratch = zalloc(((uint64_t)pl->nlive + 1) * sizeof *scratch);
    if (!scratch) return TR_MATH_MEMORY;
    for (uint64_t i = 0; i < n; i++) tr_refine_region_bound(pl, lv, (uint32_t)level, groups[i], exact, scratch, &out[i]);
    free(scratch);
    return TR_MATH_OK;
}
int32_t tr_refine_run(tr_refine_query *q) {
    if (!query_valid(q)) return TR_MATH_INVALID;
    int32_t rc = TR_MATH_OK;
    while (!q->done && !rc) {
        tr_refine_level_view lv;
        tr_refine_get_level(q->s, q->level, &lv);
        rc = tr_refine_eval_bounds(&lv, &q->plan, q->level, q->exact, q->wave, q->nwave, q->bounds);
        if (!rc) rc = tr_refine_supply(q, q->bounds, q->nwave);
    }
    return rc;
}
int32_t tr_refine_get_stats(const tr_refine_query *q, tr_refine_stats *out) {
    if (!query_valid(q) || !out) return TR_MATH_INVALID;
    *out = q->stats;
    out->entries = q->nentries;
    return TR_MATH_OK;
}
int32_t tr_refine_get_entries(const tr_refine_query *q, tr_refine_entry *out, uint64_t capacity, uint64_t *count) {
    if (!query_valid(q) || !count) return TR_MATH_INVALID;
    *count = q->nentries;
    if (!out && !capacity) return TR_MATH_OK;          /* the count only */
    if (q->nentries && (!out || capacity < q->nentries)) return TR_MATH_INVALID;
    if (q->nentries) memcpy(out, q->entries, (size_t)q->nentries * sizeof *out);
    return TR_MATH_OK;
}
/* every present descendant of (level, g) at the target level into bits (enumerated counts the groups visited) */
static void descend(const tr_refine_state *s, int32_t level, uint32_t g, int32_t target, uint64_t *bits, uint64_t *visited,
                    uint64_t *present) {
    if (level == target) {
        if (s->nrows[level][g]) { bits[g >> 6] |= (uint64_t)1 << (g & 63); (*present)++; }
        (*visited)++;
        return;
    }
    for (uint64_t k = s->child_off[level][g]; k < s->child_off[level][g + 1]; k++) {
        (*visited)++;
        descend(s, level - 1, s->child_ids[level][k], target, bits, visited, present);
    }
}
int32_t tr_refine_get_answer(tr_refine_query *q, uint64_t *answer, uint64_t *unresolved, uint64_t capacity) {
    if (!query_valid(q) || !q->done) return TR_MATH_INVALID;
    const tr_refine_state *s = q->s;
    int32_t t = q->req.level;
    uint64_t nw = (s->groups[t] + 63) / 64;
    if ((answer || unresolved) && capacity < nw) return TR_MATH_INVALID;
    uint64_t *tmp = zalloc(nw * 8 + 8), visited = 0, present = 0, upresent = 0;
    if (!tmp) return TR_MATH_MEMORY;
    if (answer) memset(answer, 0, (size_t)nw * 8);
    if (unresolved) memset(unresolved, 0, (size_t)nw * 8);
    for (uint64_t e = 0; e < q->nentries; e++) {
        const tr_refine_entry *u = &q->entries[e];
        int in = u->decision == TR_REFINE_ALL || u->decision == TR_REFINE_TRUE || u->decision == TR_REFINE_IN;
        int un = u->decision == TR_REFINE_UNRESOLVED;
        if (!in && !un) continue;
        uint64_t *dst = in ? answer : unresolved;
        descend(s, u->level, u->group, t, dst ? dst : tmp, &visited, in ? &present : &upresent);
    }
    free(tmp);
    q->stats.answer = present; q->stats.unresolved_groups = upresent; q->stats.enumerated = q->counted + visited;
    return TR_MATH_OK;
}
int32_t tr_refine_fine(const tr_refine_state *s, const tr_refine_request *r, uint64_t *answer, uint64_t capacity,
                       tr_refine_bound *values, tr_refine_stats *stats, const tr_math_budget *budget,
                       tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!state_valid(s) || !r || !answer) return TR_MATH_INVALID;
    tr_refine_query q;
    memset(&q, 0, sizeof q);
    int32_t rc = compile(&q, s, r);
    if (rc) { free(q.nodes); return rc; }
    int32_t t = r->level;
    uint64_t G = s->groups[t], nw = (G + 63) / 64;
    if (capacity < nw) { free(q.nodes); return TR_MATH_INVALID; }
    tr_math_need need;
    need_zero(&need);
    need.need[TR_BUDGET_SUPPORT] = (uint64_t)q.plan.natoms;
    need.need[TR_BUDGET_OUTPUT] = G;
    need.need[TR_BUDGET_BYTES] = tr_budget_mul(G, sizeof(tr_refine_bound) + 8);
    tr_math_budget dflt;
    rc = tr_budget_check(budget_or(budget, &dflt), &need, refusal);
    if (rc) { free(q.nodes); return rc; }
    tr_refine_bound *v = values ? values : zalloc(G * sizeof *v + 1);
    uint32_t *ids = zalloc(G * 4 + 4);
    if (!v || !ids) { if (!values) free(v); free(ids); free(q.nodes); return TR_MATH_MEMORY; }
    uint64_t np = 0;
    for (uint64_t g = 0; g < G; g++) if (s->nrows[t][g]) ids[np++] = (uint32_t)g;
    tr_refine_level_view lv;
    tr_refine_get_level(s, t, &lv);
    tr_refine_bound *pv = zalloc(np * sizeof *pv + 1);
    if (!pv) rc = TR_MATH_MEMORY;
    if (!rc) rc = tr_refine_eval_bounds(&lv, &q.plan, t, 1, ids, np, pv);
    memset(answer, 0, (size_t)nw * 8);
    if (values) memset(values, 0, (size_t)G * sizeof *values);
    if (!rc) {
        for (uint64_t i = 0; i < np; i++) v[ids[i]] = pv[i];
        if (r->kind == TR_REFINE_TOPK) {
            exact_cand *c = malloc((size_t)(np ? np : 1) * sizeof *c);
            if (!c) rc = TR_MATH_MEMORY;
            for (uint64_t i = 0; !rc && i < np; i++) { tr_refine_x hi; score(&q, &pv[i], &c[i].s, &hi); c[i].g = ids[i]; c[i].i = (uint32_t)i; }
            if (!rc) qsort(c, (size_t)np, sizeof *c, cmp_cand);
            for (uint64_t i = 0; !rc && i < np && i < r->k; i++) answer[c[i].g >> 6] |= (uint64_t)1 << (c[i].g & 63);
            free(c);
        } else for (uint64_t i = 0; i < np; i++) if (decide_exact(&q, &pv[i]) == TR_REFINE_TRUE) answer[ids[i] >> 6] |= (uint64_t)1 << (ids[i] & 63);
    }
    if (stats && !rc) {
        memset(stats, 0, sizeof *stats);
        stats->evaluated[t] = np; stats->fine_work = np * ((uint64_t)s->nterms + 1); stats->work = stats->fine_work;
        stats->complete = 1; stats->target = t; stats->levels = s->levels;
        for (uint64_t w = 0; w < nw; w++) stats->answer += (uint64_t)__builtin_popcountll(answer[w]);
    }
    if (!values) free(v);
    free(ids); free(pv); free(q.nodes);
    return rc;
}

/* ------------------------------------------------------------------ change localization */
static void update_release(tr_refine_update *u) {
    if (!u) return;
    tr_delta_state_free(u->leaves);
    free(u->ids); free(u->child_off); free(u->child_idx); free(u->drows); free(u->dcount); free(u->dpresent);
    free(u->dsum); free(u->ins);
    u->magic = 0;
    free(u);
}
void tr_refine_update_free(tr_refine_update *u) { if (update_valid(u)) update_release(u); }
/* w (192 bits) = x - y of two signed 128-bit values */
static void sub192(__int128 x, __int128 y, uint64_t *w) {
    uint64_t a[3] = {(uint64_t)x, (uint64_t)((unsigned __int128)x >> 64), x < 0 ? UINT64_MAX : 0};
    uint64_t b[3] = {(uint64_t)y, (uint64_t)((unsigned __int128)y >> 64), y < 0 ? UINT64_MAX : 0};
    uint64_t borrow = 0;
    for (int i = 0; i < 3; i++) { uint64_t d = a[i] - b[i], b1 = a[i] < b[i]; w[i] = d - borrow; borrow = b1 | (d < borrow); }
}
/* the 192-bit value as a signed 128-bit one with |v| <= 2^127 - 1, else 0 */
static int fit128(const uint64_t *w, __int128 *out) {
    uint64_t ext = (w[1] >> 63) ? UINT64_MAX : 0;
    if (w[2] != ext) return 0;
    __int128 v = (__int128)(((unsigned __int128)w[1] << 64) | w[0]);
    if (v < -(__int128)I128_LIMIT) return 0;
    *out = v;
    return 1;
}
typedef struct { uint32_t parent, child; } pair_t;
static int cmp_pair(const void *a, const void *b) {
    const pair_t *x = a, *y = b;
    if (x->parent != y->parent) return x->parent < y->parent ? -1 : 1;
    return x->child < y->child ? -1 : x->child > y->child;
}
/* the affected regions of every level (ids increasing: the touched leaves, then the parents of the level below),
 * and per level >= 1 each region's affected children as indices into the level below's list (CSR coffs / cidx) */
static void lists_free(uint32_t **lists, uint64_t **coffs, uint32_t **cidx) {
    for (int32_t l = 0; l < L; l++) { free(lists[l]); free(coffs[l]); free(cidx[l]); lists[l] = NULL; coffs[l] = NULL; cidx[l] = NULL; }
}
static int32_t affected_lists(const tr_refine_state *s, const tr_delta_rows *dr, uint32_t **lists, uint64_t *naff,
                              uint64_t **coffs, uint32_t **cidx) {
    for (int32_t l = 0; l < L; l++) { lists[l] = NULL; coffs[l] = NULL; cidx[l] = NULL; naff[l] = 0; }
    const uint64_t T = dr->touched;
    naff[0] = T;
    lists[0] = zalloc(T * 4 + 4);
    if (!lists[0]) return TR_MATH_MEMORY;
    for (uint64_t u = 0; u < T; u++) {
        if (dr->groups[u] >= s->groups[0]) { lists_free(lists, coffs, cidx); return TR_MATH_INVALID; }
        lists[0][u] = dr->groups[u];
    }
    for (int32_t l = 1; l < s->levels; l++) {
        uint64_t n = naff[l - 1], np = 0;
        pair_t *pp = zalloc(n * sizeof *pp + 8);
        if (!pp) { lists_free(lists, coffs, cidx); return TR_MATH_MEMORY; }
        for (uint64_t i = 0; i < n; i++) {
            uint32_t p = s->parent[l - 1][lists[l - 1][i]];
            if (p == TR_MULTI_UNMAPPED) continue;        /* a leaf absent before and after (prepare checks it) */
            pp[np].parent = p; pp[np].child = (uint32_t)i; np++;
        }
        qsort(pp, (size_t)np, sizeof *pp, cmp_pair);
        uint64_t k = 0;
        for (uint64_t i = 0; i < np; i++) if (!i || pp[i].parent != pp[i - 1].parent) k++;
        naff[l] = k;
        lists[l] = zalloc(k * 4 + 4); coffs[l] = zalloc((k + 1) * 8); cidx[l] = zalloc(np * 4 + 4);
        if (!lists[l] || !coffs[l] || !cidx[l]) { free(pp); lists_free(lists, coffs, cidx); return TR_MATH_MEMORY; }
        uint64_t j = 0;
        for (uint64_t i = 0; i < np; i++) {
            if (!i || pp[i].parent != pp[i - 1].parent) { lists[l][j] = pp[i].parent; coffs[l][j] = i; j++; }
            cidx[l][i] = pp[i].child;
        }
        coffs[l][k] = np;
        free(pp);
    }
    return TR_MATH_OK;
}
/* prepare's own need (the update's arrays) and finish's (the new state and its changes) */
static void update_needs(const tr_refine_state *s, uint64_t total, uint64_t nchild, tr_math_need *own, tr_math_need *fin) {
    const uint64_t m = (uint64_t)s->nterms, nc = (uint64_t)s->ncols;
    need_zero(own); need_zero(fin);
    own->need[TR_BUDGET_STATES] = tr_budget_mul(total, m + nc);
    own->need[TR_BUDGET_BYTES] = tr_budget_add(tr_budget_mul(total, 4 + 8 + 16 + m * (16 + 24) + nc * (16 + sizeof(tr_refine_ins))),
                                               tr_budget_mul(nchild, 4));
    fin->need[TR_BUDGET_BYTES] = tr_budget_add(s->bytes, tr_budget_mul(total, sizeof(tr_refine_change) + m * 32));
}
int32_t tr_refine_update_need(const tr_refine_state *s, const tr_delta *delta, tr_math_need *need) {
    tr_delta_rows dr;
    if (!state_valid(s) || !delta || !need || tr_delta_get_rows(delta, &dr)) return TR_MATH_INVALID;
    tr_math_need dn, own, fin;
    int32_t rc = tr_delta_state_need(s->leaves, delta, 0, &dn);
    if (rc) return rc;
    uint64_t naff[L], total = 0, nchild = 0;
    uint32_t *lists[L];
    uint64_t *coffs[L];
    uint32_t *cidx[L];
    rc = affected_lists(s, &dr, lists, naff, coffs, cidx);
    if (rc) return rc;
    for (int32_t l = 0; l < s->levels; l++) { total += naff[l]; if (l > 0) nchild += coffs[l][naff[l]]; }
    lists_free(lists, coffs, cidx);
    update_needs(s, total, nchild, &own, &fin);
    for (int i = 0; i < TR_BUDGET_RESOURCES; i++) {
        uint64_t v = dn.need[i];
        if (own.need[i] > v) v = own.need[i];
        if (fin.need[i] > v) v = fin.need[i];
        need->need[i] = v;
    }
    return TR_MATH_OK;
}
tr_refine_update *tr_refine_update_prepare(const tr_refine_state *s, const tr_delta *delta, const tr_math_budget *budget,
                                           tr_math_refusal *refusal, tr_delta_violation *violation, int32_t *reason,
                                           int32_t *error) {
    tr_budget_none(refusal);
    if (reason) *reason = TR_REFINE_REASON_NONE;
    if (violation) { memset(violation, 0, sizeof *violation); violation->group = TR_DELTA_NONE; }
    tr_delta_rows dr;
    tr_scope_view dv;
    if (!state_valid(s) || !delta || tr_delta_get_rows(delta, &dr) || tr_delta_get_view(delta, &dv)) { set_error(error, TR_MATH_INVALID); return NULL; }
    tr_math_budget dflt;
    const tr_math_budget *b = budget_or(budget, &dflt);
    /* the leaves: DELTA-001 (compatibility, the overflow proof, the promise) */
    int32_t rc = TR_MATH_OK;
    tr_delta_violation viol;
    memset(&viol, 0, sizeof viol);
    viol.group = TR_DELTA_NONE;
    tr_delta_state *nl = tr_delta_state_apply(s->leaves, delta, b, refusal, &viol, &rc);
    if (!nl) {
        if (violation) *violation = viol;
        if (reason) *reason = rc == TR_MATH_DOMAIN_ERROR && viol.group == TR_DELTA_NONE ? TR_REFINE_REASON_SCHEMA :
                              rc == TR_MATH_OVERFLOW ? TR_REFINE_REASON_OVERFLOW : TR_REFINE_REASON_NONE;
        set_error(error, rc);
        return NULL;
    }
    tr_bank_states ns;
    tr_delta_state_get_states(nl, &ns);
    const uint64_t T = dr.touched, m = (uint64_t)s->nterms, nc = (uint64_t)s->ncols;
    /* a touched leaf that has rows now must have a path */
    for (uint64_t u = 0; u < T; u++) {
        uint32_t g = dr.groups[u];
        if ((int64_t)ns.rows[g] > 0 && s->levels > 1 && s->parent[0][g] == TR_MULTI_UNMAPPED) {
            tr_delta_state_free(nl);
            if (reason) *reason = TR_REFINE_REASON_NEW_GROUP;
            set_error(error, TR_MATH_DOMAIN_ERROR);
            return NULL;
        }
    }
    /* the affected regions of every level, and their children (CSR) */
    uint64_t naff[L], off[L], total = 0;
    uint32_t *lists[L];
    uint64_t *coffs[L];
    uint32_t *cidx[L];
    rc = affected_lists(s, &dr, lists, naff, coffs, cidx);
    uint64_t nchild = 0;
    for (int32_t l = 0; l < s->levels; l++) { off[l] = total; if (!rc) { total += naff[l]; if (l > 0) nchild += coffs[l][naff[l]]; } }
    if (!rc) {
        tr_math_need own, fin;
        update_needs(s, total, nchild, &own, &fin);
        rc = tr_budget_check(b, &own, refusal);
    }
    tr_refine_update *up = rc ? NULL : zalloc(sizeof *up);
    if (!rc && !up) rc = TR_MATH_MEMORY;
    if (!rc) {
        up->ids = zalloc(total * 4 + 4); up->child_off = zalloc((total + (uint64_t)s->levels + 1) * 8);
        up->child_idx = zalloc(nchild * 4 + 4); up->drows = zalloc(total * 16 + 16); up->dcount = zalloc(total * m * 16 + 16);
        up->dsum = zalloc(total * m * 24 + 24); up->dpresent = zalloc(total * nc * 16 + 16);
        up->ins = zalloc(total * nc * sizeof(tr_refine_ins) + 8);
        if (!up->ids || !up->child_off || !up->child_idx || !up->drows || !up->dcount || !up->dsum || !up->dpresent || !up->ins) rc = TR_MATH_MEMORY;
    }
    if (!rc) {
        uint64_t ci = 0;
        for (int32_t l = 0; l < s->levels; l++) {
            memcpy(up->ids + off[l], lists[l], (size_t)naff[l] * 4);
            /* child_off has naff[l] + 1 entries per level at off[l] + l (level 0: zeros) */
            for (uint64_t i = 0; i <= naff[l]; i++) up->child_off[off[l] + (uint64_t)l + i] = l > 0 ? ci + coffs[l][i] : 0;
            if (l > 0) { memcpy(up->child_idx + ci, cidx[l], (size_t)coffs[l][naff[l]] * 4); ci += coffs[l][naff[l]]; }
        }
        /* level 0: the leaves' exact changes from the new and old cells, the present values and the inserted ones */
        for (uint64_t u = 0; u < T; u++) {
            uint32_t g = dr.groups[u];
            up->drows[u] = tr_i128_set((__int128)(int64_t)ns.rows[g] - (__int128)s->nrows[0][g]);
            for (uint64_t j = 0; j < m; j++) {
                up->dcount[u * m + j] = tr_i128_set((__int128)(int64_t)ns.count[g * m + j] - (__int128)s->count[0][g * m + j]);
                sub192(tr_i128_get(ns.sum[g * m + j]), tr_i128_get(s->sum[0][g * m + j]), &up->dsum[(u * m + j) * 3]);
            }
        }
        for (uint64_t r = 0; r < dr.rows; r++) {
            uint64_t u = dr.local[r];
            int64_t w = dr.weights[r];
            for (uint64_t c = 0; c < nc; c++) {
                if (!tr_scope_present(&dv, (int32_t)c, r) || !w) continue;
                up->dpresent[u * nc + c] = tr_i128_set(tr_i128_get(up->dpresent[u * nc + c]) + w);
                tr_refine_ins one;
                memset(&one, 0, sizeof one);
                if (w > 0) { int64_t x = tr_scope_value(&dv, (int32_t)c, r); one.count = (uint64_t)w; one.min = one.max = x; one.mod = 0; }
                else one.dels = tr_delta_mag64(w);
                tr_refine_ins_merge(&up->ins[u * nc + c], &one);
            }
        }
        up->magic = UPDATE_MAGIC; up->s = s; up->leaves = nl; nl = NULL;
        up->total = total; up->nchild = nchild; up->touched = T;
        up->work = T * (m + 1 + nc) + nchild * (m + 1 + nc);
        tr_refine_update_view *v = &up->view;
        memset(v, 0, sizeof *v);
        v->levels = s->levels; v->nterms = s->nterms; v->ncols = s->ncols;
        for (int32_t l = 0; l < s->levels; l++) { v->naff[l] = naff[l]; v->off[l] = off[l]; }
        v->ids = up->ids; v->child_off = up->child_off; v->child_idx = up->child_idx; v->drows = up->drows;
        v->dcount = up->dcount; v->dsum = up->dsum; v->dpresent = up->dpresent; v->ins = up->ins;
    }
    lists_free(lists, coffs, cidx);
    tr_delta_state_free(nl);
    if (rc) { if (up) { up->magic = UPDATE_MAGIC; update_release(up); } set_error(error, rc); return NULL; }
    set_error(error, TR_MATH_OK);
    return up;
}
int32_t tr_refine_update_view_get(tr_refine_update *u, tr_refine_update_view *out, uint64_t *nchild) {
    if (!update_valid(u) || !out) return TR_MATH_INVALID;
    *out = u->view;
    if (nchild) *nchild = u->nchild;
    return TR_MATH_OK;
}
int32_t tr_refine_update_rollup(tr_refine_update *u) {
    if (!update_valid(u)) return TR_MATH_INVALID;
    for (int32_t l = 1; l < u->view.levels; l++)
        for (uint64_t i = 0; i < u->view.naff[l]; i++) tr_refine_rollup_region(&u->view, l, i);
    u->rolled = 1;
    return TR_MATH_OK;
}
/* every level's totals of the view's changes equal the totals of the leaves with a parent */
int32_t tr_refine_update_adopt(tr_refine_update *u) {
    if (!update_valid(u)) return TR_MATH_INVALID;
    const tr_refine_update_view *v = &u->view;
    const tr_refine_state *s = u->s;
    const uint64_t m = (uint64_t)v->nterms, nc = (uint64_t)v->ncols, width = 1 + m * 4 + nc;
    uint64_t *base = zalloc(width * 8 * 2), *tot = base + width;   /* rows, counts, sums (192 bits), present */
    if (!base) return TR_MATH_MEMORY;
    int32_t rc = TR_MATH_OK;
    for (int32_t l = 0; l < v->levels && !rc; l++) {
        uint64_t *acc = l ? tot : base;
        memset(acc, 0, (size_t)width * 8);
        for (uint64_t i = 0; i < v->naff[l]; i++) {
            uint64_t me = v->off[l] + i;
            if (!l && (s->levels < 2 || s->parent[0][v->ids[me]] == TR_MULTI_UNMAPPED)) continue;
            acc[0] += (uint64_t)tr_i128_get(v->drows[me]);
            for (uint64_t j = 0; j < m; j++) {
                acc[1 + j] += (uint64_t)tr_i128_get(v->dcount[me * m + j]);
                tr_refine_add192(&acc[1 + m + 3 * j], &v->dsum[(me * m + j) * 3]);
            }
            for (uint64_t c = 0; c < nc; c++) acc[1 + 4 * m + c] += (uint64_t)tr_i128_get(v->dpresent[me * nc + c]);
        }
        if (l && memcmp(base, tot, (size_t)width * 8)) rc = TR_MATH_INVALID;
    }
    free(base);
    if (!rc) u->rolled = 1;
    return rc;
}
/* the copy of a state's arrays (the levels, the links, the basis) */
static tr_refine_state *state_copy(const tr_refine_state *s, int32_t *rc) {
    tr_refine_state *n = state_alloc(s->levels, s->groups, s->nterms, s->ncols, rc);
    if (!n) return NULL;
    n->mode = s->mode; n->status = s->status; n->promise = s->promise; n->rows = s->rows; n->step = s->step;
    n->joint = s->joint;
    uint64_t m = (uint64_t)s->nterms, c = (uint64_t)s->ncols;
    n->exps = zalloc(m * c + 1); n->terms = zalloc((m ? m : 1) * sizeof(tr_bank_term));
    if (!n->exps || !n->terms) { state_release(n); *rc = TR_MATH_MEMORY; return NULL; }
    memcpy(n->exps, s->exps, (size_t)(m * c)); memcpy(n->terms, s->terms, (size_t)m * sizeof(tr_bank_term));
    for (int32_t l = 0; l < s->levels; l++) {
        uint64_t G = s->groups[l];
        memcpy(n->nrows[l], s->nrows[l], (size_t)G * 8);
        memcpy(n->count[l], s->count[l], (size_t)(G * m) * 8); memcpy(n->sum[l], s->sum[l], (size_t)(G * m) * 16);
        memcpy(n->inputs[l], s->inputs[l], (size_t)(G * c) * sizeof(tr_unc_input)); memcpy(n->tight[l], s->tight[l], (size_t)G);
        if (l + 1 < s->levels) memcpy(n->parent[l], s->parent[l], (size_t)G * 4);
        if (l > 0) {
            n->child_off[l] = zalloc((G + 1) * 8); n->child_ids[l] = zalloc(s->groups[l - 1] * 4 + 4);
            if (!n->child_off[l] || !n->child_ids[l]) { state_release(n); *rc = TR_MATH_MEMORY; return NULL; }
            memcpy(n->child_off[l], s->child_off[l], (size_t)(G + 1) * 8);
            memcpy(n->child_ids[l], s->child_ids[l], (size_t)s->groups[l - 1] * 4);
        }
    }
    *rc = TR_MATH_OK;
    return n;
}
tr_refine_state *tr_refine_update_finish(tr_refine_update *u, const tr_math_budget *budget, tr_math_refusal *refusal,
                                         int32_t *error) {
    tr_budget_none(refusal);
    if (!update_valid(u) || !u->rolled) { set_error(error, TR_MATH_INVALID); return NULL; }
    const tr_refine_state *s = u->s;
    const tr_refine_update_view *v = &u->view;
    tr_math_need own, fin;
    tr_math_budget dflt;
    update_needs(s, u->total, u->nchild, &own, &fin);
    int32_t rc = tr_budget_check(budget_or(budget, &dflt), &fin, refusal);
    if (rc) { set_error(error, rc); return NULL; }
    tr_refine_state *n = state_copy(s, &rc);
    if (!n) { set_error(error, rc); return NULL; }
    const uint64_t m = (uint64_t)s->nterms, nc = (uint64_t)s->ncols;
    n->changes = zalloc(u->total * sizeof(tr_refine_change) + 8);
    n->ch_count = zalloc(u->total * m * 8 + 8); n->ch_sum = zalloc(u->total * m * 24 + 24);
    if (!n->changes || !n->ch_count || !n->ch_sum) rc = TR_MATH_MEMORY;
    for (int32_t l = 0; l < s->levels && !rc; l++) {
        uint64_t changed = 0;
        for (uint64_t i = 0; i < v->naff[l] && !rc; i++) {
            uint64_t me = v->off[l] + i, g = v->ids[me];
            __int128 dr = tr_i128_get(v->drows[me]), nr = (__int128)n->nrows[l][g] + dr;
            if (nr < 0 || nr > (__int128)INT64_MAX || dr > (__int128)INT64_MAX || dr < -(__int128)INT64_MAX) { rc = s->promise == TR_DELTA_SIGNED ? TR_MATH_DOMAIN_ERROR : TR_MATH_OVERFLOW; break; }
            n->nrows[l][g] = (uint64_t)nr;
            int nonzero = dr != 0;
            for (uint64_t j = 0; j < m && !rc; j++) {
                __int128 dc = tr_i128_get(v->dcount[me * m + j]), c2 = (__int128)n->count[l][g * m + j] + dc;
                if (c2 < 0 || c2 > nr || dc > (__int128)INT64_MAX || dc < -(__int128)INT64_MAX) { rc = s->promise == TR_DELTA_SIGNED ? TR_MATH_DOMAIN_ERROR : TR_MATH_OVERFLOW; break; }
                n->count[l][g * m + j] = (uint64_t)c2;
                uint64_t w[3];
                __int128 old = tr_i128_get(n->sum[l][g * m + j]), sm;
                w[0] = (uint64_t)old; w[1] = (uint64_t)((unsigned __int128)old >> 64); w[2] = old < 0 ? UINT64_MAX : 0;
                tr_refine_add192(w, &v->dsum[(me * m + j) * 3]);
                if (!fit128(w, &sm)) { rc = TR_MATH_OVERFLOW; break; }
                n->sum[l][g * m + j] = tr_i128_set(sm);
                nonzero |= dc != 0 || sm != old;
            }
            uint8_t tight = n->tight[l][g];
            int any = 0;
            for (uint64_t c = 0; c < nc && !rc; c++) {
                tr_unc_input *in = &n->inputs[l][g * nc + c];
                __int128 np = (__int128)in->present + tr_i128_get(v->dpresent[me * nc + c]);
                const tr_refine_ins *z = &v->ins[me * nc + c];
                if (np < 0 || np > nr) { rc = TR_MATH_DOMAIN_ERROR; break; }   /* the inputs would contradict the cells */
                uint64_t oldp = in->present;
                in->rows = (uint64_t)nr;
                if (z->count) {
                    tr_unc_input add;
                    memset(&add, 0, sizeof add);
                    add.present = z->count; add.min = z->min; add.max = z->max; add.modulus = z->mod;
                    if (!oldp) { in->min = add.min; in->max = add.max; in->modulus = add.modulus; }
                    else {
                        uint64_t keep = in->rows;
                        tr_unc_input cur = *in;
                        cur.rows = 0; add.rows = 0;
                        input_merge(&cur, &add);
                        in->min = cur.min; in->max = cur.max; in->modulus = cur.modulus; in->rows = keep;
                    }
                }
                in->present = (uint64_t)np;
                if (z->dels && in->present) tight = 0;     /* an enclosure of the remaining values */
                if (!in->present) { in->min = in->max = 0; in->modulus = 0; }
                any |= in->present != 0;
                nonzero |= np != (__int128)oldp;
            }
            n->tight[l][g] = any ? tight : 1;
            if (!rc && nonzero) {                      /* an exact change of this region */
                tr_refine_change *ch = &n->changes[n->nchanges];
                ch->level = l; ch->reserved = 0; ch->group = (uint32_t)g; ch->reserved2 = 0; ch->drows = (int64_t)dr;
                for (uint64_t j = 0; j < m; j++) {
                    n->ch_count[n->nchanges * m + j] = (int64_t)tr_i128_get(v->dcount[me * m + j]);
                    memcpy(&n->ch_sum[(n->nchanges * m + j) * 3], &v->dsum[(me * m + j) * 3], 24);
                }
                n->nchanges++; changed++;
            }
        }
        n->changed[l] = changed;
    }
    if (!rc) {                                        /* level 0 equals the new leaves' cells, bit for bit */
        tr_bank_states ls;
        tr_delta_state_get_states(u->leaves, &ls);
        if (ls.ngroups != n->groups[0] || memcmp(ls.rows, n->nrows[0], (size_t)ls.ngroups * 8) ||
            (m && (memcmp(ls.count, n->count[0], (size_t)(ls.ngroups * m) * 8) || memcmp(ls.sum, n->sum[0], (size_t)(ls.ngroups * m) * 16))))
            rc = TR_MATH_INVALID;
    }
    if (rc) { state_release(n); set_error(error, rc); return NULL; }
    n->leaves = u->leaves; u->leaves = NULL;
    n->step = s->step + 1; n->touched = u->touched; n->work = u->work; n->strategy = TR_REFINE_LOCAL;
    n->reason = TR_REFINE_REASON_NONE;
    n->status = s->status == TR_MULTI_DECLARED ? TR_MULTI_DECLARED : TR_MULTI_CARRIED;
    uint64_t rw = 0, rows = 0;
    for (int32_t l = 1; l < s->levels; l++) rw += s->groups[l - 1] * (m + 1 + nc);
    for (uint64_t g = 0; g < n->groups[0]; g++) rows += n->nrows[0][g];
    n->rebuild_work = rw; n->rows = rows;
    state_finish(n);
    set_error(error, TR_MATH_OK);
    return n;
}

/* the complete rebuild from new levels (the bank of level 0, maps certified on the new rows, the hierarchy) */
static tr_refine_state *rebuild(const tr_refine_state *s, const tr_delta *delta, const tr_multi_level *levels,
                                int32_t nlevels, int32_t reason, const tr_math_budget *b, tr_math_refusal *refusal,
                                int32_t *error) {
    if (nlevels != s->levels) { set_error(error, TR_MATH_INVALID); return NULL; }
    int32_t rc = TR_MATH_OK;
    tr_bank *bank = tr_bank_build(levels[0].scope, s->exps, s->nterms, s->mode, b, refusal, &rc);
    if (!bank) { set_error(error, rc); return NULL; }
    tr_multi_map *maps[L];
    memset(maps, 0, sizeof maps);
    for (int32_t l = 0; l + 1 < nlevels && !rc; l++) {
        tr_multi_witness w;
        maps[l] = tr_multi_map_certify(&levels[l], &levels[l + 1], b, refusal, &w, &rc);
    }
    tr_multi_hier *h = NULL;
    if (!rc) {
        uint64_t bad = 0;
        h = tr_multi_hier_build(levels, nlevels, bank, (const tr_multi_map *const *)maps, b, refusal, &bad, &rc);
    }
    tr_refine_state *n = h ? tr_refine_build(h, levels[0].scope, s->promise, b, refusal, &rc) : NULL;
    tr_multi_hier_free(h); tr_bank_free(bank);
    for (int32_t l = 0; l < L; l++) tr_multi_map_free(maps[l]);
    if (!n) { set_error(error, rc ? rc : TR_MATH_MEMORY); return NULL; }
    if (reason == TR_REFINE_REASON_NEW_GROUP) {      /* the delta applies: the new levels must be the data after it */
        tr_delta_violation viol;
        tr_delta_state *nl = tr_delta_state_apply(s->leaves, delta, b, refusal, &viol, &rc);
        int same_cells = nl && tr_delta_state_equal(nl, n->leaves) == 1;
        tr_delta_state_free(nl);
        if (!same_cells) { tr_refine_free(n); set_error(error, rc ? rc : TR_MATH_DOMAIN_ERROR); return NULL; }
    }
    n->strategy = TR_REFINE_REBUILD; n->reason = reason; n->step = s->step + 1;
    uint64_t rw = 0;
    for (int32_t l = 1; l < n->levels; l++) rw += n->groups[l - 1] * ((uint64_t)n->nterms + 1 + (uint64_t)n->ncols);
    n->work = rw + n->rows; n->rebuild_work = rw;
    /* the changes, region by region, when the shapes agree */
    int same = n->levels == s->levels && n->nterms == s->nterms;
    for (int32_t l = 0; same && l < n->levels; l++) same = n->groups[l] == s->groups[l];
    uint64_t total = 0;
    for (int32_t l = 0; same && l < n->levels; l++) total += n->groups[l];
    if (same) {
        uint64_t m = (uint64_t)n->nterms;
        n->changes = zalloc(total * sizeof(tr_refine_change) + 8);
        n->ch_count = zalloc(total * m * 8 + 8); n->ch_sum = zalloc(total * m * 24 + 24);
        if (!n->changes || !n->ch_count || !n->ch_sum) { tr_refine_free(n); set_error(error, TR_MATH_MEMORY); return NULL; }
        for (int32_t l = 0; l < n->levels; l++) {
            uint64_t changed = 0;
            for (uint64_t g = 0; g < n->groups[l]; g++) {
                int diff = n->nrows[l][g] != s->nrows[l][g] || (m && (memcmp(n->count[l] + g * m, s->count[l] + g * m, (size_t)m * 8) ||
                                                                       memcmp(n->sum[l] + g * m, s->sum[l] + g * m, (size_t)m * 16)));
                if (!diff) continue;
                tr_refine_change *ch = &n->changes[n->nchanges];
                ch->level = l; ch->reserved = 0; ch->group = (uint32_t)g; ch->reserved2 = 0;
                ch->drows = (int64_t)((__int128)n->nrows[l][g] - (__int128)s->nrows[l][g]);
                for (uint64_t j = 0; j < m; j++) {
                    n->ch_count[n->nchanges * m + j] = (int64_t)((__int128)n->count[l][g * m + j] - (__int128)s->count[l][g * m + j]);
                    sub192(tr_i128_get(n->sum[l][g * m + j]), tr_i128_get(s->sum[l][g * m + j]), &n->ch_sum[(n->nchanges * m + j) * 3]);
                }
                n->nchanges++; changed++;
            }
            n->changed[l] = changed;
        }
    }
    n->magic = 0;
    state_finish(n);
    set_error(error, TR_MATH_OK);
    return n;
}
tr_refine_state *tr_refine_update_state(const tr_refine_state *s, const tr_delta *delta, const tr_multi_level *levels,
                                        int32_t nlevels, const tr_math_budget *budget, tr_math_refusal *refusal,
                                        tr_delta_violation *violation, int32_t *error) {
    tr_budget_none(refusal);
    if (!state_valid(s) || !delta || (levels && nlevels != s->levels)) { set_error(error, TR_MATH_INVALID); return NULL; }
    tr_math_budget dflt;
    const tr_math_budget *b = budget_or(budget, &dflt);
    int32_t rc = TR_MATH_OK, reason = TR_REFINE_REASON_NONE;
    tr_refine_update *u = tr_refine_update_prepare(s, delta, b, refusal, violation, &reason, &rc);
    if (!u) {
        if (reason != TR_REFINE_REASON_NONE && levels) return rebuild(s, delta, levels, nlevels, reason, b, refusal, error);
        set_error(error, rc);
        return NULL;
    }
    tr_refine_update_rollup(u);
    tr_refine_state *n = tr_refine_update_finish(u, b, refusal, &rc);
    update_release(u);
    set_error(error, n ? TR_MATH_OK : rc);
    return n;
}
int32_t tr_refine_get_changes(const tr_refine_state *s, tr_refine_change *out, int64_t *dcount, uint64_t *dsum,
                              uint64_t capacity, uint64_t *count) {
    if (!state_valid(s) || !count) return TR_MATH_INVALID;
    *count = s->nchanges;
    if (!s->nchanges || (!out && !dcount && !dsum && !capacity)) return TR_MATH_OK;   /* none, or the count only */
    if (!out || (s->nterms && (!dcount || !dsum)) || capacity < s->nchanges) return TR_MATH_INVALID;
    uint64_t m = (uint64_t)s->nterms;
    memcpy(out, s->changes, (size_t)s->nchanges * sizeof *out);
    if (m) { memcpy(dcount, s->ch_count, (size_t)(s->nchanges * m) * 8); memcpy(dsum, s->ch_sum, (size_t)(s->nchanges * m) * 24); }
    return TR_MATH_OK;
}
