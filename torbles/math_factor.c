/* PROB-002: factorized probabilistic inference over bounded factor graphs (graphs, costed elimination plans, programs,
 * the CPU elimination, exact answers, k best assignments and the alternative backends); see math_factor.h.
 * Reuses PROB-001 (math_prob.h: supports, categorical models, retained statistics), TENSOR-001 / TENSOR-002
 * (math_tensor.h: count tensors, contraction planning), COUNT-002 (math_symcount.h: exact limbs, factor elimination,
 * brute force, decision diagrams), GEN-002 (math_conv.h: independence declarations), FND-008 (math_planner.h) and
 * FND-009 (math_carriers.h: canonical big rationals).
 * SPDX-License-Identifier: Apache-2.0 */
#include "math_factor.h"
#include "mathematics.h"
#include "math_domain.h"
#include "math_carriers.h"
#include "math_planner.h"
#include "math_scope.h"
#include "math_tensor.h"
#include "math_prob.h"
#include "math_conv.h"
#include "math_bool.h"
#include "math_handles.h"
#include <stdlib.h>
#include <string.h>

TR_MATH_ABI(tr_factor_spec, TR_FACTOR_ABI);          /* the ctypes boundary of PROB-002 (math_handles.h) */
TR_MATH_ABI(tr_factor_outside, TR_FACTOR_ABI);
TR_MATH_ABI(tr_factor_check, TR_FACTOR_ABI);
TR_MATH_ABI(tr_factor_graph_info, TR_FACTOR_ABI);
TR_MATH_ABI(tr_factor_info, TR_FACTOR_ABI);
TR_MATH_ABI(tr_factor_shape, TR_FACTOR_ABI);
TR_MATH_ABI(tr_factor_plan_options, TR_FACTOR_ABI);
TR_MATH_ABI(tr_factor_candidate, TR_FACTOR_ABI);
TR_MATH_ABI(tr_factor_plan_info, TR_FACTOR_ABI);
TR_MATH_ABI(tr_factor_step, TR_FACTOR_ABI);
TR_MATH_ABI(tr_factor_operand, TR_FACTOR_ABI);
TR_MATH_ABI(tr_factor_program, TR_FACTOR_ABI);
TR_MATH_ABI(tr_factor_answer_info, TR_FACTOR_ABI);

_Static_assert(sizeof(tr_factor_spec) == 56, "tr_factor_spec ABI");
_Static_assert(sizeof(tr_factor_outside) == 32, "tr_factor_outside ABI");
_Static_assert(sizeof(tr_factor_check) == 32, "tr_factor_check ABI");
_Static_assert(sizeof(tr_factor_graph_info) == 104, "tr_factor_graph_info ABI");
_Static_assert(sizeof(tr_factor_info) == 96, "tr_factor_info ABI");
_Static_assert(sizeof(tr_factor_shape) == 48, "tr_factor_shape ABI");
_Static_assert(sizeof(tr_factor_plan_options) == 224, "tr_factor_plan_options ABI");
_Static_assert(sizeof(tr_factor_candidate) == 456, "tr_factor_candidate ABI");
_Static_assert(sizeof(tr_factor_plan_info) == 152, "tr_factor_plan_info ABI");
_Static_assert(sizeof(tr_factor_step) == 56, "tr_factor_step ABI");
_Static_assert(sizeof(tr_factor_operand) == 16, "tr_factor_operand ABI");
_Static_assert(sizeof(tr_factor_program) == 96, "tr_factor_program ABI");
_Static_assert(sizeof(tr_factor_answer_info) == 64, "tr_factor_answer_info ABI");
_Static_assert(TR_FACTOR_MAX_ORDERS * 2 <= TR_REG_MAX_CANDIDATES, "candidates per plan");
_Static_assert(TR_FACTOR_COST_ROW > 0 && TR_FACTOR_COST_ROW < TR_REG_MODULES, "the cost row");

#define GRAPH_MAGIC UINT32_C(0x46475248)
#define PLAN_MAGIC UINT32_C(0x46504c4e)
#define PROG_MAGIC UINT32_C(0x46505247)
#define ANS_MAGIC UINT32_C(0x46414e53)
#define NONE32 UINT32_MAX
#define INF64 UINT64_MAX
typedef unsigned __int128 u128;
static const u128 LIM126 = (u128)1 << 126;

/* ================================================================== helpers */
static void set_error(int32_t *error, int32_t code) { if (error) *error = code; }
static void *zalloc(uint64_t bytes) {
    if (bytes == TR_BUDGET_UNKNOWN || bytes > (uint64_t)SIZE_MAX / 2) return NULL;
    return calloc(1, bytes ? (size_t)bytes : 1);
}
static const tr_math_budget *budget_of(const tr_math_budget *b, tr_math_budget *dflt) {
    if (b) return b;
    tr_budget_default(dflt);
    return dflt;
}
static void need_zero(tr_math_need *n) { for (int i = 0; i < TR_BUDGET_RESOURCES; i++) n->need[i] = 0; }
/* tr_budget_check, refusing a size past 64 bits even under an unlimited budget */
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
static void put128(uint64_t *w, u128 x) { w[0] = (uint64_t)x; w[1] = (uint64_t)(x >> 64); }
static u128 get128(const uint64_t *w) { return ((u128)w[1] << 64) | w[0]; }
static uint32_t bitlen128(u128 x) { uint32_t n = 0; while (x) { n++; x >>= 1; } return n; }
static u128 gcd128(u128 a, u128 b) { while (b) { u128 t = a % b; a = b; b = t; } return a; }
static uint64_t popc64(uint64_t x) { return (uint64_t)__builtin_popcountll(x); }
static void digest_mix(uint64_t *d, const void *p, uint64_t n) {
    const unsigned char *c = (const unsigned char *)p;
    uint64_t h1 = d[0], h2 = d[1];
    for (uint64_t i = 0; i < n; i++) {
        h1 = (h1 ^ c[i]) * UINT64_C(1099511628211);
        h2 = (h2 ^ c[i]) * UINT64_C(0xff51afd7ed558ccd);
        h2 ^= h2 >> 29;
    }
    d[0] = h1; d[1] = h2;
}
/* GEN-002's source digest (math_conv.c: digest_of), so a refusal names the source as GEN-002 does */
static void source_digest(int32_t kind, const unsigned char *p, uint64_t n, uint64_t *d) {
    uint64_t h1 = UINT64_C(14695981039346656037) ^ (uint64_t)kind, h2 = UINT64_C(0x9e3779b97f4a7c15) + (uint64_t)kind;
    for (uint64_t i = 0; i < n; i++) {
        h1 = (h1 ^ p[i]) * UINT64_C(1099511628211);
        h2 = (h2 ^ p[i]) * UINT64_C(0xff51afd7ed558ccd);
        h2 ^= h2 >> 29;
    }
    h1 ^= n; h2 ^= n * UINT64_C(0xc4ceb9fe1a85ec53);
    d[0] = h1; d[1] = h2;
}
/* a stable merge sort of idx[0..n) by cmp(ctx, a, b) (< 0: a first); tmp: n entries */
typedef int (*idx_cmp)(const void *ctx, uint64_t a, uint64_t b);
static void msort(uint64_t *idx, uint64_t n, idx_cmp cmp, const void *ctx, uint64_t *tmp) {
    for (uint64_t w = 1; w < n; w *= 2) {
        for (uint64_t lo = 0; lo < n; lo += 2 * w) {
            uint64_t mid = lo + w < n ? lo + w : n, hi = lo + 2 * w < n ? lo + 2 * w : n, i = lo, j = mid, k = lo;
            while (i < mid && j < hi) tmp[k++] = cmp(ctx, idx[j], idx[i]) < 0 ? idx[j++] : idx[i++];
            while (i < mid) tmp[k++] = idx[i++];
            while (j < hi) tmp[k++] = idx[j++];
        }
        memcpy(idx, tmp, (size_t)n * 8);
    }
}

/* ================================================================== graphs */
typedef struct { int64_t value; uint32_t index, reserved; } vkey;
struct tr_factor_graph_s {
    uint32_t magic, nvars, nfactors, max_scope;
    uint32_t *domain;                 /* [n] */
    uint64_t *vat;                    /* [n + 1] offsets of the declared values */
    int64_t *value;                   /* [sum d] logical values (indices when undeclared) */
    uint8_t *null;                    /* [sum d] 1 = the NULL category */
    int32_t *scale;                   /* [n] */
    vkey *sorted;                     /* [sum d] per variable: its non-NULL values sorted (nonnull[v] of them) */
    uint32_t *nonnull, *nullidx;      /* [n]; nullidx NONE32 without a NULL value */
    int32_t *kind, *mode;             /* [nf] */
    uint32_t *nscope;                 /* [nf] */
    uint64_t *scope_at;               /* [nf] into scope */
    uint32_t *scope;                  /* [sum nscope] */
    uint64_t *table_at, *size, *nonzero, *group;   /* [nf] */
    uint64_t *entry;                  /* [entries x 2] scaled numerators (< 2^126) */
    uint64_t *denom, *maxent;         /* [nf x 2] */
    uint64_t *src_at, *src_len;       /* [nf] data sources (len 0: none) */
    unsigned char *src;
    int32_t *src_kind, *keyed, *linked;
    uint64_t *src_digest;             /* [nf x 2] */
    int32_t declared, nlinked, nsources, ndata;
    uint64_t entries, bytes, max_table, nscope_total;
    u128 assignments;
    int wide;
    uint32_t limbs;
    uint64_t digest[2];
};
static int graph_ok(const tr_factor_graph *g) { return g && g->magic == GRAPH_MAGIC; }
static void graph_release(tr_factor_graph *g) {
    if (!g) return;
    free(g->domain); free(g->vat); free(g->value); free(g->null); free(g->scale); free(g->sorted); free(g->nonnull);
    free(g->nullidx); free(g->kind); free(g->mode); free(g->nscope); free(g->scope_at); free(g->scope);
    free(g->table_at); free(g->size); free(g->nonzero); free(g->group); free(g->entry); free(g->denom); free(g->maxent);
    free(g->src_at); free(g->src_len); free(g->src); free(g->src_kind); free(g->keyed); free(g->linked);
    free(g->src_digest);
    g->magic = 0;
    free(g);
}
void tr_factor_graph_free(tr_factor_graph *g) { if (graph_ok(g)) graph_release(g); }

static int vkey_cmp(const void *a, const void *b) {
    const vkey *x = (const vkey *)a, *y = (const vkey *)b;
    return x->value < y->value ? -1 : x->value > y->value ? 1 : (x->index < y->index ? -1 : x->index > y->index);
}
/* the index of a declared value of variable v (null: the NULL category), or NONE32 */
static uint32_t var_find(const tr_factor_graph *g, uint32_t v, int64_t value, int null) {
    if (null) return g->nullidx[v];
    const vkey *s = g->sorted + g->vat[v];
    uint64_t lo = 0, hi = g->nonnull[v];
    while (lo < hi) {
        uint64_t mid = lo + (hi - lo) / 2;
        if (s[mid].value < value) lo = mid + 1;
        else hi = mid;
    }
    return lo < g->nonnull[v] && s[lo].value == value ? s[lo].index : NONE32;
}
static void out_set(tr_factor_outside *o, uint32_t f, uint32_t axis, int32_t reason, int null, int64_t value, u128 mass) {
    if (!o) return;
    o->factor = f; o->axis = axis; o->reason = reason; o->null = null; o->value = value;
    o->mass = (mass >> 64) ? UINT64_MAX : (uint64_t)mass;
}
/* the row-major index of a factor's table at value indices idx[nscope] */
static uint64_t table_index(const tr_factor_graph *g, uint32_t f, const uint32_t *idx) {
    uint64_t e = 0;
    const uint32_t *sc = g->scope + g->scope_at[f];
    for (uint32_t p = 0; p < g->nscope[f]; p++) e = e * g->domain[sc[p]] + idx[p];
    return e;
}

/* explicit rationals: D = lcm of the reduced denominators, entries scaled (each < 2^126) */
static int32_t fill_table(tr_factor_graph *g, uint32_t f, const int64_t *num, const int64_t *den, tr_factor_outside *out) {
    const uint64_t n = g->size[f];
    u128 D = 1;
    for (uint64_t e = 0; e < n; e++) {
        if (den[e] < 1) return TR_MATH_INVALID;
        if (num[e] < 0) { out_set(out, f, 0, TR_FACTOR_OUT_NEGATIVE, 0, num[e], 0); return TR_MATH_DOMAIN_ERROR; }
        uint64_t a = (uint64_t)num[e], b = (uint64_t)den[e], h = tr_gcd_u64(a, b);
        if (!a) continue;
        b /= h;
        u128 next = D / gcd128(D, b);
        if (next > LIM126 / b) return TR_MATH_UNSUPPORTED;
        D = next * b;
    }
    uint64_t *E = g->entry + 2 * g->table_at[f];
    for (uint64_t e = 0; e < n; e++) {
        uint64_t a = (uint64_t)num[e], b = (uint64_t)den[e], h = tr_gcd_u64(a, b);
        if (!a) continue;
        a /= h; b /= h;
        u128 mult = D / b;
        if (mult > (LIM126 - 1) / a) return TR_MATH_UNSUPPORTED;
        put128(E + 2 * e, (u128)a * mult);
    }
    put128(g->denom + 2 * (uint64_t)f, D);
    return TR_MATH_OK;
}

/* a TENSOR-001 tensor's group as the table (see The model) */
static int32_t fill_tensor(tr_factor_graph *g, uint32_t f, const tr_factor_spec *sp, tr_factor_outside *out) {
    const tr_tensor *T = (const tr_tensor *)sp->object;
    tr_tensor_info ti;
    tr_tensor_states st;
    if (tr_tensor_get_info(T, &ti) || tr_tensor_get_states(T, &st)) return TR_MATH_INVALID;
    if (ti.kind == TR_TENSOR_MOMENT || ti.repr == TR_TENSOR_FACTORED || ti.has_counts) return TR_MATH_UNSUPPORTED;
    if ((uint32_t)ti.order != g->nscope[f] || sp->group >= ti.ngroups) return TR_MATH_INVALID;
    if (ti.scoped) {                                       /* the tensor's own scope: its source */
        tr_scope_info si;
        if (!sp->scope || tr_scope_get_info((const tr_scope *)sp->scope, &si) || si.digest[0] != ti.digest[0] ||
            si.digest[1] != ti.digest[1] || si.ngroups != ti.ngroups) return TR_MATH_INVALID;
    } else if (sp->scope) return TR_MATH_INVALID;
    const int32_t m = ti.order;
    const uint32_t *sc = g->scope + g->scope_at[f];
    tr_tensor_axis ax[TR_TENSOR_MAX_ORDER];
    for (int32_t k = 0; k < m; k++) {
        if (tr_tensor_get_axis(T, k, &ax[k])) return TR_MATH_INVALID;
        uint32_t v = sc[k];
        if (ax[k].kind == TR_TENSOR_AXIS_FEATURE) return TR_MATH_UNSUPPORTED;
        if (ax[k].kind == TR_TENSOR_AXIS_CATEGORY) {
            if (ax[k].scale != g->scale[v]) { out_set(out, f, (uint32_t)k, TR_FACTOR_OUT_SCALE, 0, ax[k].scale, 0); return TR_MATH_DOMAIN_ERROR; }
        } else if (ax[k].dim != g->domain[v]) {
            out_set(out, f, (uint32_t)k, TR_FACTOR_OUT_DIMS, 0, (int64_t)ax[k].dim, 0);
            return TR_MATH_DOMAIN_ERROR;
        }
    }
    uint64_t *E = g->entry + 2 * g->table_at[f];
    uint64_t b, e;
    if (st.repr == TR_TENSOR_DENSE) { b = sp->group * st.cells; e = b + st.cells; }
    else { b = st.offset[sp->group]; e = st.offset[sp->group + 1]; }
    uint32_t idx[TR_FACTOR_MAX_SCOPE];
    for (uint64_t i = b; i < e; i++) {
        __int128 x = tr_i128_get(st.value[i]);
        if (!x) continue;
        uint64_t coord[TR_TENSOR_MAX_ORDER];
        if (st.repr == TR_TENSOR_DENSE) {
            uint64_t rem = i - b;
            for (int32_t k = m; k-- > 0;) { coord[k] = rem % st.dims[k]; rem /= st.dims[k]; }
        } else for (int32_t k = 0; k < m; k++) coord[k] = st.coord[i * (uint64_t)m + (uint64_t)k];
        for (int32_t k = 0; k < m; k++) {
            uint32_t v = sc[k], at;
            if (ax[k].kind == TR_TENSOR_AXIS_CATEGORY) {
                int null = ax[k].nullslot && coord[k] == ax[k].span;
                int64_t val = null ? 0 : (int64_t)((uint64_t)ax[k].lo + coord[k]);
                at = var_find(g, v, val, null);
                if (at == NONE32) {
                    out_set(out, f, (uint32_t)k, TR_FACTOR_OUT_VALUE, null, val, x < 0 ? 0 : (u128)x);
                    return TR_MATH_DOMAIN_ERROR;           /* an observed value outside the declared support */
                }
            } else at = (uint32_t)coord[k];
            idx[k] = at;
        }
        if (x < 0) { out_set(out, f, 0, TR_FACTOR_OUT_NEGATIVE, 0, 0, 0); return TR_MATH_DOMAIN_ERROR; }
        if ((u128)x >= LIM126) return TR_MATH_UNSUPPORTED;
        put128(E + 2 * table_index(g, f, idx), (u128)x);
    }
    put128(g->denom + 2 * (uint64_t)f, 1);
    return TR_MATH_OK;
}

/* the cell of support point i of a PROB-001 support in factor f (NONE: a coordinate value not declared, *bad = c) */
static uint64_t prob_cell(const tr_factor_graph *g, uint32_t f, const tr_prob_support_view *sv, uint64_t i,
                          uint32_t *bad, int64_t *badval, int *badnull) {
    const uint32_t *sc = g->scope + g->scope_at[f];
    uint32_t idx[3];
    for (int32_t c = 0; c < sv->k; c++) {
        int null = (sv->nulls[i] >> c) & 1;
        int64_t val = null ? 0 : sv->value[i * (uint64_t)sv->k + (uint64_t)c];
        uint32_t at = var_find(g, sc[c], val, null);
        if (at == NONE32) { *bad = (uint32_t)c; *badval = val; *badnull = null; return UINT64_MAX; }
        idx[c] = at;
    }
    return table_index(g, f, idx);
}
static int32_t prob_scales(tr_factor_graph *g, uint32_t f, const tr_prob_support_view *sv, tr_factor_outside *out) {
    const uint32_t *sc = g->scope + g->scope_at[f];
    if ((uint32_t)sv->k != g->nscope[f]) return TR_MATH_INVALID;
    for (int32_t c = 0; c < sv->k; c++)
        if (sv->scale[c] != g->scale[sc[c]]) { out_set(out, f, (uint32_t)c, TR_FACTOR_OUT_SCALE, 0, sv->scale[c], 0); return TR_MATH_DOMAIN_ERROR; }
    return TR_MATH_OK;
}
/* a PROB-001 categorical model with one request's weights (mode WEIGHTS / PROBABILITIES) */
static int32_t fill_model(tr_factor_graph *g, uint32_t f, const tr_factor_spec *sp, const int64_t *num,
                          const int64_t *den, tr_factor_outside *out) {
    const tr_prob_model *M = (const tr_prob_model *)sp->object;
    tr_prob_model_view mv;
    tr_prob_support_view sv;
    if (tr_prob_model_get_view(M, &mv) || tr_prob_support_get_view(tr_prob_model_support(M), &sv)) return TR_MATH_INVALID;
    if (mv.kind != TR_PROB_CATEGORICAL) return TR_MATH_UNSUPPORTED;    /* exponential families: enclosures only */
    if (sp->mode != TR_PROB_WEIGHTS && sp->mode != TR_PROB_PROBABILITIES) return TR_MATH_INVALID;
    int32_t rc = prob_scales(g, f, &sv, out);
    if (rc) return rc;
    const uint64_t d = sv.d;
    u128 D = 1;
    for (uint64_t i = 0; i < d; i++) {
        if (den[i] < 1) return TR_MATH_INVALID;
        if (num[i] < 0) { out_set(out, f, 0, TR_FACTOR_OUT_NEGATIVE, 0, 0, 0); return TR_MATH_DOMAIN_ERROR; }
        uint64_t a = (uint64_t)num[i], b = (uint64_t)den[i], h = tr_gcd_u64(a, b);
        if (!a) continue;
        b /= h;
        u128 next = D / gcd128(D, b);
        if (next > LIM126 / b) return TR_MATH_UNSUPPORTED;
        D = next * b;
    }
    uint64_t *E = g->entry + 2 * g->table_at[f];
    u128 total = 0;
    for (uint64_t i = 0; i < d; i++) {
        uint64_t a = (uint64_t)num[i], b = (uint64_t)den[i], h = tr_gcd_u64(a, b);
        if (!a) continue;
        a /= h; b /= h;
        u128 mult = D / b;
        if (mult > (LIM126 - 1) / a) return TR_MATH_UNSUPPORTED;
        u128 s = (u128)a * mult;
        total += s;                                           /* each s <= 2^126, d <= 2^32: checked against D */
        if (sp->mode == TR_PROB_PROBABILITIES && total > D) return TR_MATH_DOMAIN_ERROR;
        uint32_t bad = 0; int64_t bv = 0; int bn = 0;
        uint64_t cell = prob_cell(g, f, &sv, i, &bad, &bv, &bn);
        if (cell == UINT64_MAX) { out_set(out, f, bad, TR_FACTOR_OUT_VALUE, bn, bv, s); return TR_MATH_DOMAIN_ERROR; }
        put128(E + 2 * cell, s);
    }
    if (sp->mode == TR_PROB_PROBABILITIES && total != D) return TR_MATH_DOMAIN_ERROR;   /* not summing to 1 */
    put128(g->denom + 2 * (uint64_t)f, D);
    return TR_MATH_OK;
}
/* PROB-001 retained statistics: one group's masses */
static int32_t fill_stats(tr_factor_graph *g, uint32_t f, const tr_factor_spec *sp, tr_factor_outside *out) {
    const tr_prob_stats *S = (const tr_prob_stats *)sp->object;
    tr_prob_stats_info info;
    tr_prob_stats_view v;
    tr_prob_support_view sv;
    if (tr_prob_stats_get_info(S, &info) || tr_prob_stats_get_view(S, &v) ||
        tr_prob_support_get_view(tr_prob_stats_support(S), &sv)) return TR_MATH_INVALID;
    if (info.product) return TR_MATH_UNSUPPORTED;           /* a product of statistics: already an assumption */
    tr_scope_info si;
    if (!info.scoped || !sp->scope || tr_scope_get_info((const tr_scope *)sp->scope, &si) ||
        si.digest[0] != info.digest[0] || si.digest[1] != info.digest[1] || si.ngroups != info.ngroups ||
        sp->group >= v.ngroups) return TR_MATH_INVALID;
    int32_t rc = prob_scales(g, f, &sv, out);
    if (rc) return rc;
    uint64_t *E = g->entry + 2 * g->table_at[f];
    for (uint64_t i = 0; i < sv.d; i++) {
        uint64_t mass = v.counts[sp->group * v.d + i];
        if (!mass) continue;
        uint32_t bad = 0; int64_t bv = 0; int bn = 0;
        uint64_t cell = prob_cell(g, f, &sv, i, &bad, &bv, &bn);
        if (cell == UINT64_MAX) { out_set(out, f, bad, TR_FACTOR_OUT_VALUE, bn, bv, mass); return TR_MATH_DOMAIN_ERROR; }
        put128(E + 2 * cell, mass);
    }
    put128(g->denom + 2 * (uint64_t)f, 1);
    return TR_MATH_OK;
}

/* the source bytes of a scope (GEN-002's rule: table address + identity, or the rows descriptor) */
static uint64_t scope_source_len(const tr_scope *s, tr_scope_info *si) {
    if (!s || tr_scope_get_info(s, si)) return TR_BUDGET_UNKNOWN;
    return si->source == TR_SCOPE_TABLE ? 24 : tr_scope_key_size(s, TR_SCOPE_KEY_ROWS);
}
static int32_t scope_source_write(const tr_scope *s, const tr_scope_info *si, unsigned char *out, uint64_t len) {
    if (si->source == TR_SCOPE_TABLE) {
        memcpy(out, &si->table, 8);
        memcpy(out + 8, si->identity, 16);
        return TR_MATH_OK;
    }
    return tr_scope_key_write(s, TR_SCOPE_KEY_ROWS, out, len);
}
static int is_data(const tr_factor_spec *sp) {
    return (sp->kind == TR_FACTOR_TENSOR && sp->scope) || sp->kind == TR_FACTOR_PROB_STATS;
}

tr_factor_graph *tr_factor_graph_create(uint32_t nvars, const uint32_t *domains, const int64_t *values,
                                        const uint8_t *nulls, const int32_t *scales, uint32_t nfactors,
                                        const tr_factor_spec *specs, const uint32_t *vars, const int64_t *num,
                                        const int64_t *den, uint64_t nentries, const void *decl,
                                        const tr_math_budget *budget, tr_math_refusal *refusal,
                                        tr_factor_outside *outside, tr_factor_check *check, int32_t *error) {
    tr_budget_none(refusal);
    if (outside) memset(outside, 0, sizeof *outside);
    if (check) { memset(check, 0, sizeof *check); check->first = check->second = -1; }
    if (nvars < 1 || !domains || (nfactors && (!specs || !vars))) { set_error(error, TR_MATH_INVALID); return NULL; }
    if (nvars > TR_FACTOR_MAX_VARS) { set_error(error, TR_MATH_UNSUPPORTED); return NULL; }
    uint64_t nvalues = 0;
    for (uint32_t v = 0; v < nvars; v++) {
        if (domains[v] < 1) { set_error(error, TR_MATH_INVALID); return NULL; }
        if (domains[v] > TR_FACTOR_MAX_DOMAIN) { set_error(error, TR_MATH_UNSUPPORTED); return NULL; }
        if (values && scales && (scales[v] < 0 || scales[v] > TR_SCOPE_MAX_SCALE)) { set_error(error, TR_MATH_INVALID); return NULL; }
        nvalues += domains[v];
    }
    tr_math_budget dflt;
    const tr_math_budget *b = budget_of(budget, &dflt);
    uint64_t total = 0, maxsize = 0, nscope_total = 0, srcbytes = 0;
    uint32_t max_scope = 0;
    for (uint32_t f = 0; f < nfactors; f++) {
        const tr_factor_spec *sp = &specs[f];
        if (sp->reserved || sp->kind < TR_FACTOR_TABLE || sp->kind > TR_FACTOR_PROB_STATS ||
            (sp->kind != TR_FACTOR_TABLE && !sp->object) || (sp->kind != TR_FACTOR_PROB_MODEL && sp->mode)) {
            set_error(error, TR_MATH_INVALID); return NULL;
        }
        if (sp->nscope > TR_FACTOR_MAX_SCOPE) { set_error(error, TR_MATH_UNSUPPORTED); return NULL; }
        uint64_t size = 1;
        for (uint32_t t = 0; t < sp->nscope; t++) {
            uint32_t v = vars[sp->scope_at + t];
            if (v >= nvars) { set_error(error, TR_MATH_INVALID); return NULL; }
            for (uint32_t u = 0; u < t; u++) if (vars[sp->scope_at + u] == v) { set_error(error, TR_MATH_INVALID); return NULL; }
            size = tr_budget_mul(size, domains[v]);
        }
        if (sp->kind == TR_FACTOR_TABLE && (!num || !den || sp->table_at > nentries || size > nentries - sp->table_at)) {
            set_error(error, size == TR_BUDGET_UNKNOWN ? TR_MATH_UNSUPPORTED : TR_MATH_INVALID); return NULL;
        }
        if (sp->kind == TR_FACTOR_PROB_MODEL) {
            tr_prob_support_view sv;
            if (!num || !den || tr_prob_support_get_view(tr_prob_model_support((const tr_prob_model *)sp->object), &sv) ||
                sp->table_at > nentries || sv.d > nentries - sp->table_at) { set_error(error, TR_MATH_INVALID); return NULL; }
        }
        if (is_data(sp)) {
            tr_scope_info si;
            uint64_t len = scope_source_len((const tr_scope *)sp->scope, &si);
            if (len == TR_BUDGET_UNKNOWN) { set_error(error, TR_MATH_INVALID); return NULL; }
            srcbytes = tr_budget_add(srcbytes, len);
        }
        if (size > maxsize) maxsize = size;
        if (sp->nscope > max_scope) max_scope = sp->nscope;
        total = tr_budget_add(total, size);
        nscope_total += sp->nscope;
    }
    tr_math_need need;
    need_zero(&need);
    need.need[TR_BUDGET_VARIABLES] = nvars;
    need.need[TR_BUDGET_SUPPORT] = total;
    need.need[TR_BUDGET_TENSOR] = maxsize;
    uint64_t bytes = tr_budget_add(sizeof(tr_factor_graph), tr_budget_mul(nvars, 4 + 8 + 4 + 4 + 4));
    bytes = tr_budget_add(bytes, tr_budget_mul(nvalues, 8 + 1 + 16));
    bytes = tr_budget_add(bytes, tr_budget_mul(nfactors, 4 * 4 + 8 * 7 + 16 * 3 + 4 * 3));
    bytes = tr_budget_add(bytes, tr_budget_mul(nscope_total, 4));
    bytes = tr_budget_add(bytes, tr_budget_mul(total, 16));
    bytes = tr_budget_add(bytes, srcbytes);
    need.need[TR_BUDGET_BYTES] = bytes;
    int32_t rc = check_need(b, &need, refusal);
    if (rc) { set_error(error, rc); return NULL; }
    tr_factor_graph *g = (tr_factor_graph *)zalloc(sizeof *g);
    if (!g) { set_error(error, TR_MATH_MEMORY); return NULL; }
    const uint64_t nf1 = nfactors ? nfactors : 1;
    g->domain = (uint32_t *)zalloc(4 * (uint64_t)nvars); g->vat = (uint64_t *)zalloc(8 * ((uint64_t)nvars + 1));
    g->value = (int64_t *)zalloc(8 * nvalues); g->null = (uint8_t *)zalloc(nvalues);
    g->scale = (int32_t *)zalloc(4 * (uint64_t)nvars); g->sorted = (vkey *)zalloc(sizeof(vkey) * nvalues);
    g->nonnull = (uint32_t *)zalloc(4 * (uint64_t)nvars); g->nullidx = (uint32_t *)zalloc(4 * (uint64_t)nvars);
    g->kind = (int32_t *)zalloc(4 * nf1); g->mode = (int32_t *)zalloc(4 * nf1); g->nscope = (uint32_t *)zalloc(4 * nf1);
    g->scope_at = (uint64_t *)zalloc(8 * nf1); g->scope = (uint32_t *)zalloc(4 * (nscope_total ? nscope_total : 1));
    g->table_at = (uint64_t *)zalloc(8 * nf1); g->size = (uint64_t *)zalloc(8 * nf1);
    g->nonzero = (uint64_t *)zalloc(8 * nf1); g->group = (uint64_t *)zalloc(8 * nf1);
    g->entry = (uint64_t *)zalloc(tr_budget_mul(total ? total : 1, 16));
    g->denom = (uint64_t *)zalloc(16 * nf1); g->maxent = (uint64_t *)zalloc(16 * nf1);
    g->src_at = (uint64_t *)zalloc(8 * nf1); g->src_len = (uint64_t *)zalloc(8 * nf1);
    g->src = (unsigned char *)zalloc(srcbytes ? srcbytes : 1);
    g->src_kind = (int32_t *)zalloc(4 * nf1); g->keyed = (int32_t *)zalloc(4 * nf1); g->linked = (int32_t *)zalloc(4 * nf1);
    g->src_digest = (uint64_t *)zalloc(16 * nf1);
    if (!g->domain || !g->vat || !g->value || !g->null || !g->scale || !g->sorted || !g->nonnull || !g->nullidx ||
        !g->kind || !g->mode || !g->nscope || !g->scope_at || !g->scope || !g->table_at || !g->size || !g->nonzero ||
        !g->group || !g->entry || !g->denom || !g->maxent || !g->src_at || !g->src_len || !g->src || !g->src_kind ||
        !g->keyed || !g->linked || !g->src_digest) { graph_release(g); set_error(error, TR_MATH_MEMORY); return NULL; }
    g->nvars = nvars; g->nfactors = nfactors; g->max_scope = max_scope; g->nscope_total = nscope_total;
    memcpy(g->domain, domains, 4 * (size_t)nvars);
    /* the declared values and their sorted lookups (distinct; at most one NULL) */
    uint64_t at = 0;
    for (uint32_t v = 0; v < nvars && !rc; v++) {
        g->vat[v] = at;
        g->scale[v] = values && scales ? scales[v] : 0;
        g->nullidx[v] = NONE32;
        uint32_t nn = 0;
        for (uint32_t i = 0; i < domains[v]; i++) {
            int null = values && nulls ? nulls[at + i] != 0 : 0;
            int64_t x = values ? (null ? 0 : values[at + i]) : (int64_t)i;
            g->value[at + i] = x; g->null[at + i] = (uint8_t)null;
            if (null) { if (g->nullidx[v] != NONE32) { rc = TR_MATH_INVALID; break; } g->nullidx[v] = i; continue; }
            g->sorted[at + nn].value = x; g->sorted[at + nn].index = i; nn++;
        }
        g->nonnull[v] = nn;
        if (rc) break;
        qsort(g->sorted + at, nn, sizeof(vkey), vkey_cmp);
        for (uint32_t i = 1; i < nn; i++) if (g->sorted[at + i].value == g->sorted[at + i - 1].value) { rc = TR_MATH_INVALID; break; }
        at += domains[v];
    }
    g->vat[nvars] = at;
    /* the factors' tables */
    uint64_t sat = 0, tat = 0;
    for (uint32_t f = 0; f < nfactors && !rc; f++) {
        const tr_factor_spec *sp = &specs[f];
        g->kind[f] = sp->kind; g->mode[f] = sp->mode; g->nscope[f] = sp->nscope; g->scope_at[f] = sat;
        g->group[f] = sp->group;
        memcpy(g->scope + sat, vars + sp->scope_at, 4 * (size_t)sp->nscope);
        sat += sp->nscope;
        uint64_t size = 1;
        for (uint32_t t = 0; t < sp->nscope; t++) size *= domains[g->scope[g->scope_at[f] + t]];
        g->size[f] = size; g->table_at[f] = tat;
        tat += size;
        if (sp->kind == TR_FACTOR_TABLE) rc = fill_table(g, f, num + sp->table_at, den + sp->table_at, outside);
        else if (sp->kind == TR_FACTOR_TENSOR) rc = fill_tensor(g, f, sp, outside);
        else if (sp->kind == TR_FACTOR_PROB_MODEL) rc = fill_model(g, f, sp, num + sp->table_at, den + sp->table_at, outside);
        else rc = fill_stats(g, f, sp, outside);
        if (rc && outside && rc == TR_MATH_DOMAIN_ERROR && !outside->reason) out_set(outside, f, 0, 0, 0, 0, 0);
        if (rc) break;
        u128 mx = 0;
        uint64_t nz = 0;
        const uint64_t *E = g->entry + 2 * g->table_at[f];
        for (uint64_t e = 0; e < size; e++) { u128 x = get128(E + 2 * e); if (x) nz++; if (x > mx) mx = x; }
        put128(g->maxent + 2 * (uint64_t)f, mx);
        g->nonzero[f] = nz;
        if (size > g->max_table) g->max_table = size;
    }
    g->entries = tat;
    /* the data sources and GEN-002's rule for linked factors */
    uint64_t sb = 0;
    for (uint32_t f = 0; f < nfactors && !rc; f++) {
        const tr_factor_spec *sp = &specs[f];
        if (!is_data(sp)) continue;
        tr_scope_info si;
        uint64_t len = scope_source_len((const tr_scope *)sp->scope, &si);
        g->src_at[f] = sb; g->src_len[f] = len; g->src_kind[f] = si.source; g->keyed[f] = si.ngroups > 1;
        rc = scope_source_write((const tr_scope *)sp->scope, &si, g->src + sb, len);
        source_digest(si.source, g->src + sb, len, g->src_digest + 2 * (uint64_t)f);
        sb += len;
        g->ndata++;
    }
    for (uint32_t i = 0; i < nfactors && !rc; i++) {
        if (!g->src_len[i]) continue;
        int first = 1;
        for (uint32_t j = 0; j < i; j++)
            if (g->src_len[j] == g->src_len[i] && g->src_kind[j] == g->src_kind[i] &&
                !memcmp(g->src + g->src_at[j], g->src + g->src_at[i], (size_t)g->src_len[i])) { first = 0; break; }
        if (first) g->nsources++;
        for (uint32_t j = i + 1; j < nfactors && !rc; j++) {
            if (g->src_len[j] != g->src_len[i] || g->src_kind[j] != g->src_kind[i] ||
                memcmp(g->src + g->src_at[j], g->src + g->src_at[i], (size_t)g->src_len[i])) continue;
            int covered = 0;
            if (decl) {
                int32_t c = tr_conv_decl_covers((const tr_conv_decl *)decl, (const tr_scope *)specs[i].scope);
                if (c < 0) { rc = c; break; }
                covered = c == 1;
            }
            if (!covered) {
                if (check) {
                    check->reason = decl ? TR_CONV_NOT_COVERED : TR_CONV_NEEDS_DECLARATION;
                    check->first = (int32_t)i; check->second = (int32_t)j;
                    check->keyed = g->keyed[i] | g->keyed[j];
                    check->digest[0] = g->src_digest[2 * (uint64_t)i]; check->digest[1] = g->src_digest[2 * (uint64_t)i + 1];
                }
                rc = TR_MATH_DOMAIN_ERROR;
                break;
            }
            g->linked[i] = g->linked[j] = 1;
            g->declared = 1;
        }
    }
    if (rc) { graph_release(g); set_error(error, rc); return NULL; }
    for (uint32_t f = 0; f < nfactors; f++) g->nlinked += g->linked[f];
    /* L: every value of an elimination is below prod max E_f x prod d_i < 2^(64 L) */
    uint64_t nbits = 1;
    for (uint32_t f = 0; f < nfactors; f++) nbits += bitlen128(get128(g->maxent + 2 * (uint64_t)f));
    for (uint32_t v = 0; v < nvars; v++) nbits += bitlen128(domains[v]);
    uint64_t L = nbits / 64 + 1;
    g->limbs = L > TR_FACTOR_MAX_LIMBS ? 0 : (uint32_t)L;   /* 0: past the representation (plans refuse) */
    u128 asg = 1;
    for (uint32_t v = 0; v < nvars && !g->wide; v++) {
        if (asg > (~(u128)0) / domains[v]) g->wide = 1;
        else asg *= domains[v];
    }
    g->assignments = g->wide ? 0 : asg;
    g->digest[0] = UINT64_C(14695981039346656037); g->digest[1] = UINT64_C(0x9e3779b97f4a7c15);
    digest_mix(g->digest, &nvars, 4); digest_mix(g->digest, domains, 4 * (uint64_t)nvars);
    digest_mix(g->digest, &nfactors, 4);
    for (uint32_t f = 0; f < nfactors; f++) {
        digest_mix(g->digest, &g->nscope[f], 4);
        digest_mix(g->digest, g->scope + g->scope_at[f], 4 * (uint64_t)g->nscope[f]);
        digest_mix(g->digest, g->entry + 2 * g->table_at[f], 16 * g->size[f]);
        digest_mix(g->digest, g->denom + 2 * (uint64_t)f, 16);
    }
    g->bytes = bytes;
    g->magic = GRAPH_MAGIC;
    set_error(error, TR_MATH_OK);
    return g;
}
int32_t tr_factor_graph_get_info(const tr_factor_graph *g, tr_factor_graph_info *out) {
    if (!graph_ok(g) || !out) return TR_MATH_INVALID;
    memset(out, 0, sizeof *out);
    out->entries = g->entries; out->bytes = g->bytes;
    out->assignments_lo = (uint64_t)g->assignments; out->assignments_hi = (uint64_t)(g->assignments >> 64);
    out->digest[0] = g->digest[0]; out->digest[1] = g->digest[1];
    out->nvars = g->nvars; out->nfactors = g->nfactors; out->max_scope = g->max_scope; out->wide = (uint32_t)g->wide;
    out->declared = g->declared; out->linked = g->nlinked; out->sources = g->nsources; out->data = g->ndata;
    out->limbs = g->limbs; out->max_table = g->max_table;
    return TR_MATH_OK;
}
int32_t tr_factor_graph_factor(const tr_factor_graph *g, uint32_t f, tr_factor_info *out) {
    if (!graph_ok(g) || !out || f >= g->nfactors) return TR_MATH_INVALID;
    memset(out, 0, sizeof *out);
    out->kind = g->kind[f]; out->mode = g->mode[f]; out->nscope = g->nscope[f]; out->linked = (uint32_t)g->linked[f];
    out->entries = g->size[f]; out->group = g->group[f]; out->nonzero = g->nonzero[f];
    memcpy(out->denom, g->denom + 2 * (uint64_t)f, 16); memcpy(out->maxent, g->maxent + 2 * (uint64_t)f, 16);
    memcpy(out->source_digest, g->src_digest + 2 * (uint64_t)f, 16);
    out->source_kind = g->src_kind[f]; out->keyed = g->keyed[f];
    return TR_MATH_OK;
}
int32_t tr_factor_graph_table(const tr_factor_graph *g, uint32_t f, uint32_t *scope, uint64_t *entries,
                              uint64_t capacity, uint64_t *denom) {
    if (!graph_ok(g) || f >= g->nfactors || capacity < g->size[f] || (g->nscope[f] && !scope) || !entries || !denom)
        return TR_MATH_INVALID;
    memcpy(scope, g->scope + g->scope_at[f], 4 * (size_t)g->nscope[f]);
    memcpy(entries, g->entry + 2 * g->table_at[f], 16 * (size_t)g->size[f]);
    memcpy(denom, g->denom + 2 * (uint64_t)f, 16);
    return TR_MATH_OK;
}

/* ================================================================== shapes and the interaction graph */
typedef struct {
    int32_t mode;
    uint32_t nkeep, nfixed;
    uint32_t *keep, *fixed;
    uint64_t R, k;
    uint8_t *role;                    /* [n] 0 eliminated, 1 kept, 2 fixed */
    uint32_t *fixpos;                 /* [n] position in fixed (NONE32) */
    uint32_t nelim;
} shape_t;
static void shape_release(shape_t *s) { free(s->keep); free(s->fixed); free(s->role); free(s->fixpos); memset(s, 0, sizeof *s); }
static int32_t shape_make(const tr_factor_graph *g, const tr_factor_shape *in, shape_t *s) {
    memset(s, 0, sizeof *s);
    if (!in || in->reserved || (in->mode != TR_FACTOR_SUM && in->mode != TR_FACTOR_MAX && in->mode != TR_FACTOR_KBEST) ||
        in->requests < 1 || (in->nkeep && !in->keep) || (in->nfixed && !in->fixed) ||
        (in->mode != TR_FACTOR_SUM && in->nkeep) || (in->mode == TR_FACTOR_KBEST && in->kbest < 1) ||
        in->nfixed > g->nvars || in->nkeep > g->nvars) return TR_MATH_INVALID;
    if (in->nkeep > TR_FACTOR_MAX_SCOPE) return TR_MATH_UNSUPPORTED;
    const uint32_t n = g->nvars;
    s->role = (uint8_t *)zalloc(n); s->fixpos = (uint32_t *)zalloc(4 * (uint64_t)n);
    s->keep = (uint32_t *)zalloc(4 * ((uint64_t)in->nkeep + 1)); s->fixed = (uint32_t *)zalloc(4 * ((uint64_t)in->nfixed + 1));
    if (!s->role || !s->fixpos || !s->keep || !s->fixed) { shape_release(s); return TR_MATH_MEMORY; }
    for (uint32_t v = 0; v < n; v++) s->fixpos[v] = NONE32;
    for (uint32_t i = 0; i < in->nfixed; i++) {
        uint32_t v = in->fixed[i];
        if (v >= n || s->role[v]) { shape_release(s); return TR_MATH_INVALID; }
        s->role[v] = 2; s->fixpos[v] = i; s->fixed[i] = v;
    }
    for (uint32_t i = 0; i < in->nkeep; i++) {
        uint32_t v = in->keep[i];
        if (v >= n || s->role[v]) { shape_release(s); return TR_MATH_INVALID; }
        s->role[v] = 1; s->keep[i] = v;
    }
    s->mode = in->mode; s->nkeep = in->nkeep; s->nfixed = in->nfixed; s->R = in->requests; s->k = in->kbest;
    for (uint32_t v = 0; v < n; v++) s->nelim += s->role[v] == 0;
    return TR_MATH_OK;
}
/* the free variables (not fixed) as bitset vertices, in increasing variable order */
typedef struct {
    uint32_t n;                       /* free variables */
    uint64_t words;
    uint32_t *var;                    /* [n] */
    uint32_t *fid;                    /* [nvars] free index (NONE32: fixed) */
    uint64_t *adj;                    /* [n x words] */
    uint64_t *dom;                    /* [n] */
} igraph;
static void ig_release(igraph *G) { free(G->var); free(G->fid); free(G->adj); free(G->dom); memset(G, 0, sizeof *G); }
static int ig_has(const igraph *G, uint32_t a, uint32_t b) { return (G->adj[(uint64_t)a * G->words + b / 64] >> (b % 64)) & 1; }
static void ig_set(igraph *G, uint32_t a, uint32_t b) { G->adj[(uint64_t)a * G->words + b / 64] |= (uint64_t)1 << (b % 64); }
static void ig_clear(igraph *G, uint32_t a, uint32_t b) { G->adj[(uint64_t)a * G->words + b / 64] &= ~((uint64_t)1 << (b % 64)); }
static int32_t ig_make(const tr_factor_graph *g, const shape_t *s, igraph *G) {
    memset(G, 0, sizeof *G);
    const uint32_t n = g->nvars;
    G->fid = (uint32_t *)zalloc(4 * (uint64_t)n);
    G->var = (uint32_t *)zalloc(4 * (uint64_t)n);
    if (!G->fid || !G->var) { ig_release(G); return TR_MATH_MEMORY; }
    for (uint32_t v = 0; v < n; v++) {
        if (s->role[v] == 2) { G->fid[v] = NONE32; continue; }
        G->fid[v] = G->n; G->var[G->n++] = v;
    }
    G->words = ((uint64_t)G->n + 63) / 64;
    if (!G->words) G->words = 1;
    G->adj = (uint64_t *)zalloc(tr_budget_mul(tr_budget_mul(G->n ? G->n : 1, G->words), 8));
    G->dom = (uint64_t *)zalloc(8 * ((uint64_t)G->n + 1));
    if (!G->adj || !G->dom) { ig_release(G); return TR_MATH_MEMORY; }
    for (uint32_t i = 0; i < G->n; i++) G->dom[i] = g->domain[G->var[i]];
    for (uint32_t f = 0; f < g->nfactors; f++) {
        const uint32_t *sc = g->scope + g->scope_at[f];
        for (uint32_t a = 0; a < g->nscope[f]; a++)
            for (uint32_t c = 0; c < g->nscope[f]; c++) {
                uint32_t u = G->fid[sc[a]], w = G->fid[sc[c]];
                if (u != NONE32 && w != NONE32 && u != w) ig_set(G, u, w);
            }
    }
    return TR_MATH_OK;
}
static int32_t ig_copy(const igraph *G, uint64_t **adj) {
    *adj = (uint64_t *)zalloc(tr_budget_mul(tr_budget_mul(G->n ? G->n : 1, G->words), 8));
    if (!*adj) return TR_MATH_MEMORY;
    memcpy(*adj, G->adj, (size_t)(G->n * G->words * 8));
    return TR_MATH_OK;
}

/* ================================================================== candidate orders */
/* greedy orders of the eliminated free vertices on a working copy of the adjacency */
/* the neighbours of v (set bits of its row) into nb; their number */
static uint32_t ig_neighbours(const igraph *G, uint32_t v, uint32_t *nb) {
    const uint64_t *a = G->adj + (uint64_t)v * G->words;
    uint32_t k = 0;
    for (uint64_t w = 0; w < G->words; w++) {
        uint64_t bits = a[w];
        while (bits) { nb[k++] = (uint32_t)(w * 64 + (uint64_t)__builtin_ctzll(bits)); bits &= bits - 1; }
    }
    return k;
}
static int32_t greedy_order(const igraph *G0, const shape_t *s, int32_t h, uint32_t *out) {
    igraph G = *G0;
    int32_t rc = ig_copy(G0, &G.adj);
    if (rc) return rc;
    uint8_t *done = (uint8_t *)zalloc(G.n + 1);
    uint32_t *nb = (uint32_t *)zalloc(4 * ((uint64_t)G.n + 1));
    if (!done || !nb) { free(G.adj); free(done); free(nb); return TR_MATH_MEMORY; }
    uint32_t m = 0;
    for (uint32_t step = 0; step < s->nelim; step++) {
        uint32_t best = NONE32;
        uint64_t bc = INF64;
        for (uint32_t v = 0; v < G.n; v++) {
            if (done[v] || s->role[G.var[v]] != 0) continue;
            uint32_t k = ig_neighbours(&G, v, nb);
            uint64_t c = 0;
            if (h == TR_FACTOR_H_MIN_DEGREE) c = k;
            else if (h == TR_FACTOR_H_MIN_WEIGHT) {
                c = G.dom[v];
                for (uint32_t i = 0; i < k; i++) c = tr_budget_mul(c, G.dom[nb[i]]);
            } else {                                        /* fill edges among the neighbours (weighted or not) */
                for (uint32_t i = 0; i < k && c < bc; i++)
                    for (uint32_t j = i + 1; j < k; j++) {
                        if (ig_has(&G, nb[i], nb[j])) continue;
                        c = tr_budget_add(c, h == TR_FACTOR_H_WEIGHTED_FILL ? tr_budget_mul(G.dom[nb[i]], G.dom[nb[j]]) : 1);
                    }
            }
            if (c < bc) { bc = c; best = v; }
        }
        if (best == NONE32) break;
        out[m++] = G.var[best];
        done[best] = 1;
        uint64_t *ab = G.adj + (uint64_t)best * G.words;
        uint32_t k = ig_neighbours(&G, best, nb);
        for (uint32_t i = 0; i < k; i++) {                  /* connect the neighbours; drop the vertex */
            uint64_t *au = G.adj + (uint64_t)nb[i] * G.words;
            for (uint64_t w = 0; w < G.words; w++) au[w] |= ab[w];
            ig_clear(&G, nb[i], nb[i]);
            ig_clear(&G, nb[i], best);
        }
        memset(ab, 0, (size_t)G.words * 8);
    }
    free(G.adj); free(done); free(nb);
    return m == s->nelim ? TR_MATH_OK : TR_MATH_INVALID;
}
/* the reach of v through the eliminated set S (bit masks over at most 64 free vertices, E first) */
static uint64_t dp_reach(const uint64_t *adj, uint64_t S, uint32_t v) {
    uint64_t R = adj[v], comp = adj[v] & S, seen = comp | ((uint64_t)1 << v);
    while (comp) {
        uint32_t u = (uint32_t)__builtin_ctzll(comp);
        comp &= comp - 1;
        uint64_t nb = adj[u];
        R |= nb;
        uint64_t nw = nb & S & ~seen;
        seen |= nw; comp |= nw;
    }
    return R & ~(S | ((uint64_t)1 << v));
}
static uint64_t dp_cells(const uint64_t *dom, uint64_t R) {
    uint64_t c = 1;
    while (R) { uint32_t u = (uint32_t)__builtin_ctzll(R); R &= R - 1; c = tr_budget_mul(c, dom[u]); }
    return c;
}
typedef struct {
    int ran, feasible;
    uint64_t states;
    uint64_t min_width, min_joint, min_support;   /* the minima over every order (the final step included) */
    int minima;
} dp_result;
/* The exact DP over the subsets of E (E <= 64 free vertices remapped first): among the orders whose steps satisfy the
 * per-step limits, the least total joint cells (order out[m]); with minima, the min-max of each per-step metric over
 * every order. */
static int32_t dp_order(const igraph *G, const shape_t *s, const tr_math_budget *b, uint64_t fin_cells, uint32_t fin_width,
                        int minima, uint32_t *out, dp_result *res) {
    memset(res, 0, sizeof *res);
    const uint32_t m = s->nelim, nf = G->n;
    uint32_t loc[64];                               /* local vertex (E first) -> free index */
    uint32_t k = 0;
    for (uint32_t i = 0; i < nf; i++) if (s->role[G->var[i]] == 0) loc[k++] = i;
    for (uint32_t i = 0; i < nf; i++) if (s->role[G->var[i]] != 0) loc[k++] = i;
    uint64_t adj[64], dom[64];
    for (uint32_t i = 0; i < nf; i++) {
        uint64_t a = 0;
        for (uint32_t j = 0; j < nf; j++) if (ig_has(G, loc[i], loc[j])) a |= (uint64_t)1 << j;
        adj[i] = a; dom[i] = G->dom[loc[i]];
    }
    const uint64_t N = (uint64_t)1 << m;
    uint64_t *best = (uint64_t *)zalloc(8 * N);
    uint8_t *par = (uint8_t *)zalloc(N);
    uint64_t *mw = minima ? (uint64_t *)zalloc(8 * N) : NULL, *mj = minima ? (uint64_t *)zalloc(8 * N) : NULL,
             *ms = minima ? (uint64_t *)zalloc(8 * N) : NULL;
    if (!best || !par || (minima && (!mw || !mj || !ms))) { free(best); free(par); free(mw); free(mj); free(ms); return TR_MATH_MEMORY; }
    const uint64_t lw = b->limit[TR_BUDGET_DEGREE], lt = b->limit[TR_BUDGET_TENSOR], ls = b->limit[TR_BUDGET_SUPPORT];
    for (uint64_t S = 1; S < N; S++) { best[S] = INF64; if (minima) mw[S] = mj[S] = ms[S] = INF64; }
    best[0] = 0;
    if (minima) mw[0] = mj[0] = ms[0] = 0;
    for (uint64_t S = 1; S < N; S++) {
        uint64_t bits = S;
        while (bits) {
            uint32_t v = (uint32_t)__builtin_ctzll(bits);
            bits &= bits - 1;
            uint64_t prev = S & ~((uint64_t)1 << v);
            res->states++;
            if (best[prev] == INF64 && !minima) continue;
            uint64_t R = dp_reach(adj, prev, v), msg = dp_cells(dom, R), joint = tr_budget_mul(msg, dom[v]);
            uint64_t w = popc64(R);
            if (best[prev] != INF64 && w <= lw && joint <= lt && msg <= ls) {
                uint64_t c = tr_budget_add(best[prev], joint);
                if (c < best[S]) { best[S] = c; par[S] = (uint8_t)v; }
            }
            if (minima) {
                uint64_t a = mw[prev] > w ? mw[prev] : w, j = mj[prev] > joint ? mj[prev] : joint,
                         q = ms[prev] > msg ? ms[prev] : msg;
                if (a < mw[S]) mw[S] = a;
                if (j < mj[S]) mj[S] = j;
                if (q < ms[S]) ms[S] = q;
            }
        }
    }
    res->ran = 1;
    res->feasible = best[N - 1] != INF64;
    if (res->feasible) {
        uint64_t S = N - 1;
        for (uint32_t i = m; i-- > 0;) { uint32_t v = par[S]; out[i] = G->var[loc[v]]; S &= ~((uint64_t)1 << v); }
    }
    if (minima) {
        res->minima = 1;
        res->min_width = mw[N - 1] > fin_width ? mw[N - 1] : fin_width;
        res->min_joint = mj[N - 1] > fin_cells ? mj[N - 1] : fin_cells;
        res->min_support = ms[N - 1] > fin_cells ? ms[N - 1] : fin_cells;
    }
    free(best); free(par); free(mw); free(mj); free(ms);
    return TR_MATH_OK;
}
/* the minor-min-width lower bound of the treewidth of the free graph (contract a min-degree vertex into its
 * min-degree neighbour; the largest min degree seen) */
static int32_t mmw_bound(const igraph *G0, uint64_t *lb) {
    igraph G = *G0;
    int32_t rc = ig_copy(G0, &G.adj);
    if (rc) return rc;
    uint8_t *gone = (uint8_t *)zalloc(G.n + 1);
    uint64_t *deg = (uint64_t *)zalloc(8 * ((uint64_t)G.n + 1));
    if (!gone || !deg) { free(G.adj); free(gone); free(deg); return TR_MATH_MEMORY; }
    for (uint32_t v = 0; v < G.n; v++) {
        const uint64_t *a = G.adj + (uint64_t)v * G.words;
        for (uint64_t w = 0; w < G.words; w++) deg[v] += popc64(a[w]);
    }
    uint64_t best = 0;
    for (uint32_t left = G.n; left > 1; left--) {
        uint32_t v = NONE32;
        for (uint32_t i = 0; i < G.n; i++) if (!gone[i] && (v == NONE32 || deg[i] < deg[v])) v = i;
        if (deg[v] > best) best = deg[v];
        uint32_t u = NONE32;
        for (uint32_t i = 0; i < G.n; i++) if (!gone[i] && i != v && ig_has(&G, v, i) && (u == NONE32 || deg[i] < deg[u])) u = i;
        gone[v] = 1;
        if (u == NONE32) continue;                         /* an isolated vertex: removed */
        for (uint32_t i = 0; i < G.n; i++) {               /* contract v into u */
            if (gone[i] || !ig_has(&G, v, i)) continue;
            ig_clear(&G, i, v);
            if (i != u && !ig_has(&G, u, i)) { ig_set(&G, u, i); ig_set(&G, i, u); deg[u]++; }
            else deg[i]--;
        }
        for (uint32_t i = 0; i < G.n; i++) if (!gone[i] && ig_has(&G, u, i) && i != u && !ig_has(&G, i, u)) ig_set(&G, i, u);
        memset(G.adj + (uint64_t)v * G.words, 0, (size_t)G.words * 8);
        deg[u] = 0;
        const uint64_t *a = G.adj + (uint64_t)u * G.words;
        for (uint64_t w = 0; w < G.words; w++) deg[u] += popc64(a[w]);
    }
    free(G.adj); free(gone); free(deg);
    *lb = best;
    return TR_MATH_OK;
}

/* ================================================================== bucket simulation of an order */
typedef struct {
    uint32_t nsteps;                  /* m + 1 (the final step) */
    uint32_t *var;                    /* [m] eliminated variable of each step */
    uint64_t *cells, *nterms;         /* [nsteps] */
    uint32_t *nout; uint64_t *out_at; uint32_t *outv;   /* output variables per step (sorted; final: Q order) */
    uint32_t *nopd; uint64_t *opd_at; uint32_t *okind, *oidx;   /* operands per step */
    uint64_t width, joint, support, products, arena, nopd_total, nout_total, nstrides;
    int32_t code;                     /* 0, or TR_MATH_UNSUPPORTED (a scope past TR_FACTOR_MAX_SCOPE) */
} sim_t;
static void sim_release(sim_t *x) {
    free(x->var); free(x->cells); free(x->nterms); free(x->nout); free(x->out_at); free(x->outv); free(x->nopd);
    free(x->opd_at); free(x->okind); free(x->oidx);
    memset(x, 0, sizeof *x);
}
static int u32cmp(const void *a, const void *b) { uint32_t x = *(const uint32_t *)a, y = *(const uint32_t *)b; return x < y ? -1 : x > y; }
static int32_t simulate(const tr_factor_graph *g, const shape_t *s, const uint32_t *order, sim_t *x) {
    memset(x, 0, sizeof *x);
    const uint32_t n = g->nvars, nf = g->nfactors, m = s->nelim, ns = m + 1;
    uint32_t *pos = (uint32_t *)zalloc(4 * (uint64_t)n), *tb = (uint32_t *)zalloc(4 * ((uint64_t)nf + m + 1));
    uint8_t *inu = (uint8_t *)zalloc(n);
    uint32_t *tmp = (uint32_t *)zalloc(4 * ((uint64_t)n + 1));
    x->var = (uint32_t *)zalloc(4 * ((uint64_t)m + 1)); x->cells = (uint64_t *)zalloc(8 * (uint64_t)ns);
    x->nterms = (uint64_t *)zalloc(8 * (uint64_t)ns); x->nout = (uint32_t *)zalloc(4 * (uint64_t)ns);
    x->out_at = (uint64_t *)zalloc(8 * (uint64_t)ns); x->nopd = (uint32_t *)zalloc(4 * (uint64_t)ns);
    x->opd_at = (uint64_t *)zalloc(8 * (uint64_t)ns);
    uint64_t ocap = (uint64_t)nf + ns + 1, vcap = g->nscope_total + (uint64_t)s->nkeep + ns + 1;   /* outv grows */
    x->okind = (uint32_t *)zalloc(4 * ocap); x->oidx = (uint32_t *)zalloc(4 * ocap);
    x->outv = (uint32_t *)zalloc(4 * vcap);
    /* scopes of the messages (sorted), to union the next buckets */
    uint64_t *msc_at = (uint64_t *)zalloc(8 * (uint64_t)ns);
    uint32_t *msc_n = (uint32_t *)zalloc(4 * (uint64_t)ns);
    int32_t rc = TR_MATH_OK;
    if (!pos || !tb || !inu || !tmp || !x->var || !x->cells || !x->nterms || !x->nout || !x->out_at || !x->nopd ||
        !x->opd_at || !x->okind || !x->oidx || !x->outv || !msc_at || !msc_n) rc = TR_MATH_MEMORY;
    for (uint32_t v = 0; v < n && !rc; v++) pos[v] = NONE32;
    for (uint32_t i = 0; i < m && !rc; i++) {
        uint32_t v = order[i];
        if (v >= n || s->role[v] != 0 || pos[v] != NONE32) { rc = TR_MATH_INVALID; break; }
        pos[v] = i; x->var[i] = v;
    }
    for (uint32_t f = 0; f < nf && !rc; f++) {             /* bucket: the first-eliminated free scope variable */
        uint32_t bst = m;
        for (uint32_t t = 0; t < g->nscope[f]; t++) { uint32_t q = pos[g->scope[g->scope_at[f] + t]]; if (q < bst) bst = q; }
        tb[f] = bst;
    }
    uint64_t oat = 0, vat = 0, arena = 0;
    for (uint32_t st = 0; st < ns && !rc; st++) {
        const int fin = st == m;
        uint32_t xv = fin ? NONE32 : order[st], cnt = 0, nu = 0;
        x->opd_at[st] = oat;
        for (uint32_t f = 0; f < nf; f++) {
            if (tb[f] != st) continue;
            x->okind[oat + cnt] = TR_FACTOR_OPD_INPUT; x->oidx[oat + cnt] = f; cnt++;
            for (uint32_t t = 0; t < g->nscope[f]; t++) {
                uint32_t v = g->scope[g->scope_at[f] + t];
                if (v != xv && s->role[v] != 2 && !inu[v]) { inu[v] = 1; tmp[nu++] = v; }
            }
        }
        for (uint32_t q = 0; q < st; q++) {
            if (tb[nf + q] != st) continue;
            x->okind[oat + cnt] = TR_FACTOR_OPD_MESSAGE; x->oidx[oat + cnt] = q; cnt++;
            for (uint32_t t = 0; t < msc_n[q]; t++) {
                uint32_t v = x->outv[msc_at[q] + t];
                if (v != xv && !inu[v]) { inu[v] = 1; tmp[nu++] = v; }
            }
        }
        x->nopd[st] = cnt;
        oat += cnt;
        for (uint32_t t = 0; t < nu; t++) inu[tmp[t]] = 0;
        x->out_at[st] = vat;
        if (vat + (fin ? s->nkeep : nu) > vcap) {           /* the message scopes can total up to m n variables */
            uint64_t cap = vcap * 2 > vat + nu + s->nkeep ? vcap * 2 : vat + nu + s->nkeep;
            uint32_t *gr = (uint32_t *)realloc(x->outv, (size_t)(cap * 4));
            if (!gr) { rc = TR_MATH_MEMORY; break; }
            x->outv = gr; vcap = cap;
        }
        if (fin) {                                          /* the output over Q, in Q's order */
            for (uint32_t t = 0; t < nu; t++) if (s->role[tmp[t]] != 1) { rc = TR_MATH_INVALID; break; }
            memcpy(x->outv + vat, s->keep, 4 * (size_t)s->nkeep);
            nu = s->nkeep;
        } else {
            qsort(tmp, nu, 4, u32cmp);
            memcpy(x->outv + vat, tmp, 4 * (size_t)nu);
        }
        if (nu > TR_FACTOR_MAX_SCOPE) x->code = TR_MATH_UNSUPPORTED;
        x->nout[st] = nu;
        msc_at[st] = vat; msc_n[st] = nu;
        vat += nu;
        uint64_t cells = 1, mb = m;
        for (uint32_t t = 0; t < nu; t++) {
            uint32_t v = x->outv[x->out_at[st] + t];
            cells = tr_budget_mul(cells, g->domain[v]);
            if (pos[v] < mb) mb = pos[v];
        }
        x->cells[st] = cells;
        x->nterms[st] = fin ? 1 : g->domain[xv];
        if (!fin) tb[nf + st] = (uint32_t)mb;
        uint64_t joint = tr_budget_mul(cells, x->nterms[st]);
        uint64_t w = fin ? (nu ? nu - 1 : 0) : nu;
        if (w > x->width) x->width = w;
        if (joint > x->joint) x->joint = joint;
        if (cells > x->support) x->support = cells;
        x->products = tr_budget_add(x->products, tr_budget_mul(joint, cnt ? cnt : 1));
        x->nstrides = tr_budget_add(x->nstrides, tr_budget_mul(cnt, (uint64_t)nu + 1));
        arena = tr_budget_add(arena, cells);
    }
    x->nsteps = ns; x->arena = arena; x->nopd_total = oat; x->nout_total = vat;
    free(pos); free(tb); free(inu); free(tmp); free(msc_at); free(msc_n);
    if (rc) sim_release(x);
    return rc;
}

/* ================================================================== plans */
void tr_factor_default_rates(double rates[2][TR_COST_COMPONENTS]) {
    /* WORDS, SCAN, CHAIN, COMBINE, CONSTRUCT, REUSE, FUSION, MATERIALIZE, TRANSFER, CPU_FINAL, GPU_FINAL, FIXED:
     * hand-set relative estimates (TENSOR-002's scale: about a nanosecond of the 2-core development machine per unit
     * on the CPU; the CUDA rates are uncalibrated estimates for a discrete device), not times */
    static const double cpu[TR_COST_COMPONENTS] = {0, 0.5, 2.0, 0, 1.0, 0, 0, 0.05, 0, 0, 0, 2000.0};
    static const double cuda[TR_COST_COMPONENTS] = {0, 0.01, 0.02, 0, 0.01, 0, 0, 0.05, 0.12, 0, 0, 25000.0};
    if (!rates) return;
    memcpy(rates[0], cpu, sizeof cpu);
    memcpy(rates[1], cuda, sizeof cuda);
}
struct tr_factor_plan_s {
    uint32_t magic;
    const tr_factor_graph *g;
    uint64_t gdigest[2];
    shape_t s;
    int32_t ncand, chosen, code, reason, forced, devices;
    uint32_t norders;
    uint32_t *orders;                 /* [norders x nelim] */
    int32_t heur[TR_FACTOR_MAX_ORDERS];
    tr_factor_candidate cand[TR_REG_MAX_CANDIDATES];
    tr_factor_plan_info info;
};
static int plan_ok(const tr_factor_plan *p) { return p && p->magic == PLAN_MAGIC; }
void tr_factor_plan_free(tr_factor_plan *p) {
    if (!plan_ok(p)) return;
    shape_release(&p->s);
    free(p->orders);
    p->magic = 0;
    free(p);
}
/* values per request of a shape's answers */
static uint64_t shape_values(const shape_t *s, uint64_t cells_final) {
    if (s->mode == TR_FACTOR_MAX) return 5;
    if (s->mode == TR_FACTOR_KBEST) return tr_budget_add(2, tr_budget_mul(2, s->k));
    return s->nkeep ? tr_budget_add(2, cells_final) : 2;
}
/* the exact amounts, needs and device sizes of a candidate (order simulated in x) */
static void make_candidate(const tr_factor_graph *g, const shape_t *s, const sim_t *x, int32_t heuristic, int32_t order,
                           int32_t device, uint32_t free_vars, tr_factor_candidate *c) {
    memset(c, 0, sizeof *c);
    tr_reg_candidate *r = &c->reg;
    r->module = 0; r->representation = 0; r->device = device; r->finalize = 0; r->reuse = 0; r->batch = 1; r->valid = 1;
    tr_budget_none(&r->budget);
    c->heuristic = heuristic; c->device = device; c->nsteps = (int32_t)x->nsteps; c->order = order;
    c->width = x->width; c->joint = x->joint; c->support = x->support; c->products = x->products; c->arena = x->arena;
    const uint32_t L = g->limbs;
    c->limbs = L;
    c->device_ok = L >= 1 && L <= TR_FACTOR_DEVICE_LIMBS;
    const uint64_t R = s->R;
    const int mx = s->mode != TR_FACTOR_SUM;
    uint64_t entries = tr_budget_mul(R, x->arena), words = tr_budget_mul(entries, L);
    uint64_t arena_bytes = tr_budget_add(tr_budget_mul(words, mx ? 24 : 8), mx ? tr_budget_mul(entries, 4) : 0);
    uint64_t prog_bytes = tr_budget_add(tr_budget_mul(x->nsteps, sizeof(tr_factor_step)),
                                        tr_budget_mul(x->nopd_total, sizeof(tr_factor_operand)));
    prog_bytes = tr_budget_add(prog_bytes, tr_budget_mul(x->nstrides, 8));
    prog_bytes = tr_budget_add(prog_bytes, tr_budget_mul(x->nout_total, 12));
    uint64_t bases = tr_budget_mul(tr_budget_mul(R, g->nfactors), 8);
    uint64_t vals = shape_values(s, x->cells[x->nsteps - 1]);
    uint64_t out_values = tr_budget_mul(R, vals);
    uint64_t answer = tr_budget_mul(out_values, tr_budget_add(tr_budget_mul(4 * (uint64_t)L, 8), 128));
    uint64_t host = tr_budget_add(tr_budget_add(arena_bytes, prog_bytes), tr_budget_add(bases, answer));
    host = tr_budget_add(host, tr_budget_mul(TR_FACTOR_SCRATCH * (uint64_t)L, 8));
    tr_math_need *nd = &r->need;
    need_zero(nd);
    nd->need[TR_BUDGET_VARIABLES] = free_vars;
    nd->need[TR_BUDGET_DEGREE] = x->width;
    nd->need[TR_BUDGET_SUPPORT] = x->support;
    nd->need[TR_BUDGET_TENSOR] = x->joint;
    nd->need[TR_BUDGET_STATES] = entries;
    nd->need[TR_BUDGET_OUTPUT] = out_values;
    uint64_t *a = r->amount;
    uint64_t reads = 0;
    for (uint32_t st = 0; st < x->nsteps; st++)
        reads = tr_budget_add(reads, tr_budget_mul(tr_budget_mul(x->cells[st], x->nterms[st]), x->nopd[st]));
    a[TR_COST_SCAN] = tr_budget_mul(R, reads);
    a[TR_COST_CHAIN] = tr_budget_mul(tr_budget_mul(R, x->products), tr_budget_mul(L, mx ? 2 : 1));
    a[TR_COST_CONSTRUCT] = words;
    a[TR_COST_MATERIALIZE] = answer;
    if (device == TR_REG_ON_CUDA) {
        tr_factor_program pv;
        memset(&pv, 0, sizeof pv);
        pv.nsteps = x->nsteps; pv.limbs = L; pv.mode = mx ? TR_FACTOR_MAX : TR_FACTOR_SUM; pv.nfactors = g->nfactors;
        pv.arena = x->arena; pv.nopd = x->nopd_total; pv.nstrides = x->nstrides;
        pv.ndims = x->nout_total; pv.inputs = g->entries;
        c->device_bytes = tr_factor_device_bytes(&pv, R);
        a[TR_COST_TRANSFER] = tr_budget_add(c->device_bytes, arena_bytes);
        a[TR_COST_FIXED] = (uint64_t)x->nsteps + 1;
        nd->need[TR_BUDGET_BYTES] = tr_budget_add(c->device_bytes, host);
        if (!c->device_ok || s->mode == TR_FACTOR_KBEST) {
            r->valid = 0; r->code = TR_MATH_UNSUPPORTED;
            r->refusal.kind = TR_REG_REFUSE_UNSUPPORTED; r->refusal.code = TR_MATH_UNSUPPORTED; r->refusal.step = -1;
        }
    } else {
        a[TR_COST_FIXED] = x->nsteps;
        nd->need[TR_BUDGET_BYTES] = host;
    }
    c->bytes = host;
    if (x->code || !L) {
        r->valid = 0; r->code = TR_MATH_UNSUPPORTED;
        r->refusal.kind = TR_REG_REFUSE_UNSUPPORTED; r->refusal.code = TR_MATH_UNSUPPORTED; r->refusal.step = -1;
    }
}

int32_t tr_factor_plan_create(const tr_factor_graph *g, const tr_factor_shape *shape, const tr_factor_plan_options *opt,
                              const tr_math_budget *budget, tr_factor_plan **out, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!out) return TR_MATH_INVALID;
    *out = NULL;
    if (!graph_ok(g)) return TR_MATH_INVALID;
    tr_factor_plan_options o;
    if (opt) o = *opt;
    else { memset(&o, 0, sizeof o); o.force = -1; }
    if (o.devices & ~TR_REG_ON_BOTH || o.heuristics < 0 || o.heuristics >= (1 << TR_FACTOR_HEURISTICS) ||
        (o.norder && !o.order) || o.dp_limit > TR_FACTOR_DP_MAX || o.force < -1) return TR_MATH_INVALID;
    if (!o.devices) o.devices = TR_REG_ON_CPU;
    double rates[2][TR_COST_COMPONENTS];
    if (o.custom_rates) memcpy(rates, o.rates, sizeof rates);
    else tr_factor_default_rates(rates);
    for (int d = 0; d < 2; d++)
        for (int k = 0; k < TR_COST_COMPONENTS; k++)
            if (!(rates[d][k] >= 0) || rates[d][k] > 1e300) return TR_MATH_INVALID;
    tr_math_budget dflt;
    const tr_math_budget *b = budget_of(budget, &dflt);
    shape_t s;
    int32_t rc = shape_make(g, shape, &s);
    if (rc) return rc;
    if (o.norder && o.norder != s.nelim) { shape_release(&s); return TR_MATH_INVALID; }
    igraph G;
    rc = ig_make(g, &s, &G);
    if (rc) { shape_release(&s); return rc; }
    const uint32_t m = s.nelim;
    uint32_t dp_limit = o.dp_limit ? o.dp_limit : TR_FACTOR_DP_DEFAULT;
    int heur = o.heuristics ? o.heuristics : ((1 << TR_FACTOR_HEURISTICS) - 1) & ~(1 << TR_FACTOR_H_GIVEN);
    if (o.norder) heur |= 1 << TR_FACTOR_H_GIVEN;
    int dp_ok = (heur & (1 << TR_FACTOR_H_EXACT)) && G.n <= 64 && m <= dp_limit;
    uint64_t dp_states = dp_ok ? tr_budget_mul((uint64_t)1 << m, m ? m : 1) : 0;
    uint64_t dp_bytes = dp_ok ? tr_budget_mul((uint64_t)1 << m, 9 + 24) : 0;
    uint32_t ngen = 0;
    for (int h = 0; h < TR_FACTOR_HEURISTICS; h++) if (((heur >> h) & 1) && h != TR_FACTOR_H_EXACT) ngen++;
    if (dp_ok && (tr_budget_add(dp_states, ngen) > b->limit[TR_BUDGET_ITERATIONS] ||
                  tr_budget_add(dp_bytes, sizeof(tr_factor_plan)) > b->limit[TR_BUDGET_BYTES])) {
        dp_ok = 0; dp_states = 0; dp_bytes = 0;            /* skipped (not refused): the heuristics remain */
    }
    tr_math_need need;
    need_zero(&need);
    need.need[TR_BUDGET_ITERATIONS] = tr_budget_add(dp_states, ngen ? ngen : 1);
    need.need[TR_BUDGET_BYTES] = tr_budget_add(tr_budget_add(sizeof(tr_factor_plan), dp_bytes),
                                               tr_budget_mul(TR_FACTOR_MAX_ORDERS + 1, 4 * ((uint64_t)m + 1)));
    rc = check_need(b, &need, refusal);
    if (rc) { ig_release(&G); shape_release(&s); return rc; }
    tr_factor_plan *p = (tr_factor_plan *)zalloc(sizeof *p);
    uint32_t *ord = (uint32_t *)zalloc(4 * ((uint64_t)TR_FACTOR_MAX_ORDERS + 1) * ((uint64_t)m + 1));
    if (!p || !ord) { free(p); free(ord); ig_release(&G); shape_release(&s); return TR_MATH_MEMORY; }
    p->s = s; p->g = g; p->gdigest[0] = g->digest[0]; p->gdigest[1] = g->digest[1]; p->orders = ord; p->chosen = -1;
    p->devices = o.devices;
    p->info.mode = s.mode; p->info.eliminate = m; p->info.keep = s.nkeep; p->info.fixed = s.nfixed;
    p->info.free_vars = G.n; p->info.bytes = need.need[TR_BUDGET_BYTES];
    tr_budget_none(&p->info.refusal);
    /* the final step's sizes (order independent) */
    uint64_t fin_cells = 1;
    for (uint32_t i = 0; i < s.nkeep; i++) fin_cells = tr_budget_mul(fin_cells, g->domain[s.keep[i]]);
    uint32_t fin_width = s.nkeep ? s.nkeep - 1 : 0;
    dp_result dpr;
    memset(&dpr, 0, sizeof dpr);
    uint32_t *cur = ord + (uint64_t)TR_FACTOR_MAX_ORDERS * ((uint64_t)m + 1);   /* scratch order */
    for (int h = 0; h < TR_FACTOR_HEURISTICS && !rc; h++) {
        if (!((heur >> h) & 1)) continue;
        int have = 0;
        if (h == TR_FACTOR_H_EXACT) {
            if (!dp_ok) continue;
            rc = dp_order(&G, &s, b, fin_cells, fin_width, 0, cur, &dpr);
            have = !rc && dpr.feasible;
        } else if (h == TR_FACTOR_H_NATURAL || h == TR_FACTOR_H_REVERSE) {
            uint32_t k = 0;
            for (uint32_t v = 0; v < g->nvars; v++) {
                uint32_t vv = h == TR_FACTOR_H_NATURAL ? v : g->nvars - 1 - v;
                if (s.role[vv] == 0) cur[k++] = vv;
            }
            have = 1;
        } else if (h == TR_FACTOR_H_GIVEN) {
            memcpy(cur, o.order, 4 * (size_t)m);
            have = 1;
        } else {
            rc = greedy_order(&G, &s, h, cur);
            have = !rc;
        }
        p->info.evaluated += h == TR_FACTOR_H_EXACT ? 0 : 1;
        if (!have || rc) continue;
        int dup = -1;
        for (uint32_t i = 0; i < p->norders; i++)
            if (!memcmp(ord + (uint64_t)i * ((uint64_t)m + 1), cur, 4 * (size_t)m)) { dup = (int)i; break; }
        if (dup >= 0 || p->norders >= TR_FACTOR_MAX_ORDERS) continue;
        memcpy(ord + (uint64_t)p->norders * ((uint64_t)m + 1), cur, 4 * (size_t)m);
        p->heur[p->norders++] = h;
    }
    p->info.dp_states = dpr.states;
    p->info.complete = dpr.ran;
    if (rc) { ig_release(&G); tr_factor_plan_free((p->magic = PLAN_MAGIC, p)); return rc; }
    /* the candidates: each order on each device */
    int32_t k = 0;
    for (uint32_t i = 0; i < p->norders && !rc; i++) {
        sim_t x;
        rc = simulate(g, &s, ord + (uint64_t)i * ((uint64_t)m + 1), &x);
        if (rc) break;
        if (o.devices & TR_REG_ON_CPU) make_candidate(g, &s, &x, p->heur[i], (int32_t)i, TR_REG_ON_CPU, G.n, &p->cand[k++]);
        if (o.devices & TR_REG_ON_CUDA) make_candidate(g, &s, &x, p->heur[i], (int32_t)i, TR_REG_ON_CUDA, G.n, &p->cand[k++]);
        sim_release(&x);
    }
    if (rc) { ig_release(&G); tr_factor_plan_free((p->magic = PLAN_MAGIC, p)); return rc; }
    p->ncand = k;
    p->info.norders = (int32_t)p->norders;
    if (o.force >= k) { ig_release(&G); tr_factor_plan_free((p->magic = PLAN_MAGIC, p)); return TR_MATH_INVALID; }
    /* the FND-008 choice over the candidates (one round: at most TR_REG_MAX_CANDIDATES) */
    double coef[TR_REG_MODULES][2][TR_COST_COMPONENTS];
    memset(coef, 0, sizeof coef);
    memcpy(coef[TR_FACTOR_COST_ROW], rates, sizeof rates);
    tr_reg_candidate chunk[TR_REG_MAX_CANDIDATES];
    tr_planner_evidence *ev = (tr_planner_evidence *)zalloc(sizeof *ev);
    if (!ev) { ig_release(&G); tr_factor_plan_free((p->magic = PLAN_MAGIC, p)); return TR_MATH_MEMORY; }
    for (int32_t i = 0; i < k; i++) { chunk[i] = p->cand[i].reg; chunk[i].module = TR_FACTOR_COST_ROW; }
    int32_t code = k ? tr_planner_choose(chunk, k, b, &coef[0][0][0], NULL, ev) : TR_MATH_UNSUPPORTED;
    int32_t chosen = k ? ev->chosen : -1, reason = k ? ev->reason : TR_PLANNER_R_NONE_ADMISSIBLE;
    tr_math_refusal plan_refusal = k ? ev->refusal : (tr_math_refusal){-1, 0, 0, 0};
    for (int32_t i = 0; i < k; i++) { chunk[i].module = 0; p->cand[i].reg = chunk[i]; }
    free(ev);
    if (o.force >= 0) {
        p->forced = 1;
        tr_reg_candidate *f = &p->cand[o.force].reg;
        if (f->status == TR_PLANNER_S_REJECTED) {
            chosen = -1; reason = TR_PLANNER_R_FORCE_REFUSED;
            code = f->code ? f->code : TR_MATH_UNSUPPORTED;
            plan_refusal = f->budget;
        } else { chosen = o.force; reason = TR_PLANNER_R_FORCED; code = TR_MATH_OK; }
        for (int32_t i = 0; i < k; i++) {
            tr_reg_candidate *c = &p->cand[i].reg;
            if (i == chosen) { c->status = TR_PLANNER_S_CHOSEN; c->reason = reason; }
            else if (c->status != TR_PLANNER_S_REJECTED) { c->status = TR_PLANNER_S_ADMISSIBLE; c->reason = TR_PLANNER_R_NOT_FORCED; }
        }
    }
    /* a refusal: is it proven for every order? the exact minima (DP) or lower bounds */
    if (chosen < 0) {
        uint64_t lw = fin_width, lj = fin_cells, ls = fin_cells;
        int exact = 0;
        if (dp_ok) {
            dp_result mm;
            uint32_t *tmp = (uint32_t *)zalloc(4 * ((uint64_t)m + 1));
            if (tmp && !dp_order(&G, &s, b, fin_cells, fin_width, 1, tmp, &mm)) {
                lw = mm.min_width; lj = mm.min_joint; ls = mm.min_support; exact = 1;
                p->info.dp_states = tr_budget_add(p->info.dp_states, mm.states);
            }
            free(tmp);
        }
        if (!exact) {
            uint64_t mmw = 0;
            if (!mmw_bound(&G, &mmw) && mmw > lw) lw = mmw;
            uint64_t first_j = INF64, first_s = INF64;
            uint32_t *nb = (uint32_t *)zalloc(4 * ((uint64_t)G.n + 1));
            for (uint32_t i = 0; nb && i < G.n; i++) {
                if (s.role[G.var[i]] != 0) continue;
                uint64_t c = 1;
                uint32_t kk = ig_neighbours(&G, i, nb);
                for (uint32_t u = 0; u < kk; u++) c = tr_budget_mul(c, G.dom[nb[u]]);
                if (c < first_s) first_s = c;
                uint64_t j = tr_budget_mul(c, G.dom[i]);
                if (j < first_j) first_j = j;
            }
            free(nb);
            if (first_j != INF64 && first_j > lj) lj = first_j;
            if (first_s != INF64 && first_s > ls) ls = first_s;
        }
        p->info.min_width = lw; p->info.min_joint = lj; p->info.min_support = ls; p->info.bounds_exact = exact;
        int proven = (dp_ok && !dpr.feasible) || lw > b->limit[TR_BUDGET_DEGREE] || lj > b->limit[TR_BUDGET_TENSOR] ||
                     ls > b->limit[TR_BUDGET_SUPPORT] || (uint64_t)G.n > b->limit[TR_BUDGET_VARIABLES];
        if (code == TR_MATH_BUDGET && p->ncand) {           /* order-independent needs past their limits */
            const tr_math_need *nd = &p->cand[0].reg.need;
            if (nd->need[TR_BUDGET_OUTPUT] > b->limit[TR_BUDGET_OUTPUT]) proven = 1;
        }
        p->info.every_order = proven && code == TR_MATH_BUDGET;
    }
    if (chosen >= 0 && p->cand[chosen].reg.status != TR_PLANNER_S_CHOSEN) {
        p->cand[chosen].reg.status = TR_PLANNER_S_CHOSEN; p->cand[chosen].reg.reason = reason;
    }
    p->chosen = chosen; p->code = chosen >= 0 ? TR_MATH_OK : code; p->reason = reason;
    p->info.ncandidates = k; p->info.chosen = chosen; p->info.code = p->code; p->info.reason = reason;
    p->info.forced = p->forced; p->info.devices = o.devices;
    p->info.refusal = p->code == TR_MATH_BUDGET ? plan_refusal : (tr_math_refusal){-1, 0, 0, 0};
    if (p->code == TR_MATH_BUDGET && refusal) *refusal = p->info.refusal;
    ig_release(&G);
    p->magic = PLAN_MAGIC;
    *out = p;
    return p->code;
}
int32_t tr_factor_plan_get_info(const tr_factor_plan *p, tr_factor_plan_info *out) {
    if (!plan_ok(p) || !out) return TR_MATH_INVALID;
    *out = p->info;
    return TR_MATH_OK;
}
int32_t tr_factor_plan_get(const tr_factor_plan *p, int32_t i, tr_factor_candidate *out) {
    if (!plan_ok(p) || !out || i < 0 || i >= p->ncand) return TR_MATH_INVALID;
    *out = p->cand[i];
    return TR_MATH_OK;
}
int32_t tr_factor_plan_order(const tr_factor_plan *p, int32_t i, uint32_t *out, uint64_t capacity, uint64_t *n) {
    if (!plan_ok(p) || !n || i < 0 || i >= p->ncand) return TR_MATH_INVALID;
    const uint32_t m = p->s.nelim;
    *n = m;
    if (capacity < m || (m && !out)) return TR_MATH_INVALID;
    memcpy(out, p->orders + (uint64_t)p->cand[i].order * ((uint64_t)m + 1), 4 * (size_t)m);
    return TR_MATH_OK;
}

/* ================================================================== programs */
struct tr_factor_prog_s {
    uint32_t magic;
    const tr_factor_graph *g;
    const tr_factor_plan *plan;
    uint64_t gdigest[2];
    int32_t index, device, mode;
    shape_t s;                        /* a copy of the plan's shape */
    tr_factor_step *step;
    tr_factor_operand *opd;
    uint64_t *stride, *dims;
    uint32_t *outv;                   /* [ndims] the output variables of every step (dims order) */
    uint64_t *fstride;                /* [nf x nfixed] the strides of the fixed variables in each factor */
    uint32_t nsteps;
    uint64_t nopd, nstrides, ndims, arena, bytes;
    uint32_t limbs;
};
static int prog_ok(const tr_factor_prog *p) { return p && p->magic == PROG_MAGIC; }
void tr_factor_prog_free(tr_factor_prog *p) {
    if (!prog_ok(p)) return;
    free(p->step); free(p->opd); free(p->stride); free(p->dims); free(p->outv); free(p->fstride);
    shape_release(&p->s);
    p->magic = 0;
    free(p);
}
static int32_t shape_copy(const shape_t *a, uint32_t n, shape_t *b) {
    memset(b, 0, sizeof *b);
    b->role = (uint8_t *)zalloc(n); b->fixpos = (uint32_t *)zalloc(4 * (uint64_t)n);
    b->keep = (uint32_t *)zalloc(4 * ((uint64_t)a->nkeep + 1)); b->fixed = (uint32_t *)zalloc(4 * ((uint64_t)a->nfixed + 1));
    if (!b->role || !b->fixpos || !b->keep || !b->fixed) { shape_release(b); return TR_MATH_MEMORY; }
    memcpy(b->role, a->role, n); memcpy(b->fixpos, a->fixpos, 4 * (size_t)n);
    memcpy(b->keep, a->keep, 4 * (size_t)a->nkeep); memcpy(b->fixed, a->fixed, 4 * (size_t)a->nfixed);
    b->mode = a->mode; b->nkeep = a->nkeep; b->nfixed = a->nfixed; b->R = a->R; b->k = a->k; b->nelim = a->nelim;
    return TR_MATH_OK;
}
/* the row-major stride of variable v in factor f's full table (0 when absent) */
static uint64_t factor_stride(const tr_factor_graph *g, uint32_t f, uint32_t v) {
    const uint32_t *sc = g->scope + g->scope_at[f];
    uint64_t st = 1;
    for (uint32_t p = g->nscope[f]; p-- > 0;) {
        if (sc[p] == v) return st;
        st *= g->domain[sc[p]];
    }
    return 0;
}
static uint64_t message_stride(const tr_factor_graph *g, const uint32_t *mv, uint32_t n, uint32_t v) {
    uint64_t st = 1;
    for (uint32_t p = n; p-- > 0;) {
        if (mv[p] == v) return st;
        st *= g->domain[mv[p]];
    }
    return 0;
}
tr_factor_prog *tr_factor_prog_create(const tr_factor_graph *g, const tr_factor_plan *plan, int32_t index,
                                      const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    if (!graph_ok(g) || !plan_ok(plan) || plan->g != g || plan->gdigest[0] != g->digest[0] ||
        plan->gdigest[1] != g->digest[1] || index < -1 || index >= plan->ncand) { set_error(error, TR_MATH_INVALID); return NULL; }
    if (index < 0) {
        if (plan->chosen < 0) { set_error(error, plan->code ? plan->code : TR_MATH_INVALID); return NULL; }
        index = plan->chosen;
    }
    const tr_factor_candidate *c = &plan->cand[index];
    if (c->reg.status == TR_PLANNER_S_REJECTED) {
        if (refusal && c->reg.code == TR_MATH_BUDGET) *refusal = c->reg.budget;
        set_error(error, c->reg.code ? c->reg.code : TR_MATH_UNSUPPORTED);
        return NULL;
    }
    const shape_t *s = &plan->s;
    const uint32_t m = s->nelim;
    sim_t x;
    int32_t rc = simulate(g, s, plan->orders + (uint64_t)c->order * ((uint64_t)m + 1), &x);
    if (rc) { set_error(error, rc); return NULL; }
    tr_math_budget dflt;
    const tr_math_budget *b = budget_of(budget, &dflt);
    uint64_t ns_total = 0;
    for (uint32_t st = 0; st < x.nsteps; st++) ns_total = tr_budget_add(ns_total, tr_budget_mul(x.nopd[st], (uint64_t)x.nout[st] + 1));
    uint64_t bytes = tr_budget_add(sizeof(tr_factor_prog), tr_budget_mul(x.nsteps, sizeof(tr_factor_step)));
    bytes = tr_budget_add(bytes, tr_budget_mul(x.nopd_total, sizeof(tr_factor_operand)));
    bytes = tr_budget_add(bytes, tr_budget_mul(ns_total, 8));
    bytes = tr_budget_add(bytes, tr_budget_mul(x.nout_total, 12));
    bytes = tr_budget_add(bytes, tr_budget_mul(tr_budget_mul(g->nfactors, s->nfixed), 8));
    rc = check_one(b, TR_BUDGET_BYTES, bytes, refusal);
    tr_factor_prog *p = rc ? NULL : (tr_factor_prog *)zalloc(sizeof *p);
    if (rc || !p) { sim_release(&x); set_error(error, rc ? rc : TR_MATH_MEMORY); return NULL; }
    p->g = g; p->plan = plan; p->gdigest[0] = g->digest[0]; p->gdigest[1] = g->digest[1]; p->index = index;
    p->device = c->device; p->mode = s->mode == TR_FACTOR_SUM ? TR_FACTOR_SUM : TR_FACTOR_MAX; p->limbs = g->limbs;
    rc = shape_copy(s, g->nvars, &p->s);
    p->step = (tr_factor_step *)zalloc(tr_budget_mul(x.nsteps, sizeof(tr_factor_step)));
    p->opd = (tr_factor_operand *)zalloc(tr_budget_mul(x.nopd_total ? x.nopd_total : 1, sizeof(tr_factor_operand)));
    p->stride = (uint64_t *)zalloc(tr_budget_mul(ns_total ? ns_total : 1, 8));
    p->dims = (uint64_t *)zalloc(tr_budget_mul(x.nout_total ? x.nout_total : 1, 8));
    p->outv = (uint32_t *)zalloc(tr_budget_mul(x.nout_total ? x.nout_total : 1, 4));
    p->fstride = (uint64_t *)zalloc(tr_budget_mul(tr_budget_mul(g->nfactors ? g->nfactors : 1, s->nfixed ? s->nfixed : 1), 8));
    if (!rc && (!p->step || !p->opd || !p->stride || !p->dims || !p->outv || !p->fstride)) rc = TR_MATH_MEMORY;
    uint64_t sat = 0;
    for (uint32_t st = 0; st < x.nsteps && !rc; st++) {
        tr_factor_step *S = &p->step[st];
        S->final = st + 1 == x.nsteps; S->var = S->final ? NONE32 : x.var[st];
        S->nout = x.nout[st]; S->nopd = x.nopd[st]; S->cells = x.cells[st]; S->nterms = x.nterms[st];
        S->out_at = 0; S->opd_at = x.opd_at[st]; S->dim_at = x.out_at[st];
        const uint32_t *ov = x.outv + x.out_at[st];
        for (uint32_t k = 0; k < S->nout; k++) { p->dims[S->dim_at + k] = g->domain[ov[k]]; p->outv[S->dim_at + k] = ov[k]; }
        for (uint32_t j = 0; j < S->nopd; j++) {
            tr_factor_operand *op = &p->opd[S->opd_at + j];
            op->kind = x.okind[S->opd_at + j]; op->index = x.oidx[S->opd_at + j]; op->stride_at = sat;
            for (uint32_t k = 0; k <= S->nout; k++) {
                uint32_t v = k < S->nout ? ov[k] : S->var;
                uint64_t sv = 0;
                if (v != NONE32) {
                    if (op->kind == TR_FACTOR_OPD_INPUT) sv = factor_stride(g, op->index, v);
                    else sv = message_stride(g, x.outv + x.out_at[op->index], x.nout[op->index], v);
                }
                p->stride[sat + k] = sv;
            }
            sat += (uint64_t)S->nout + 1;
        }
    }
    uint64_t at = 0;
    for (uint32_t st = 0; st < x.nsteps && !rc; st++) { p->step[st].out_at = at; at += p->step[st].cells; }
    for (uint32_t f = 0; f < g->nfactors && !rc; f++)
        for (uint32_t i = 0; i < s->nfixed; i++) p->fstride[(uint64_t)f * s->nfixed + i] = factor_stride(g, f, s->fixed[i]);
    p->nsteps = x.nsteps; p->nopd = x.nopd_total; p->nstrides = sat; p->ndims = x.nout_total; p->arena = at; p->bytes = bytes;
    sim_release(&x);
    if (rc) { p->magic = PROG_MAGIC; tr_factor_prog_free(p); set_error(error, rc); return NULL; }
    p->magic = PROG_MAGIC;
    tr_factor_program v;
    if (tr_factor_prog_get_view(p, &v) || tr_factor_program_check(&v)) {   /* a construction invariant */
        tr_factor_prog_free(p); set_error(error, TR_MATH_INVALID); return NULL;
    }
    set_error(error, TR_MATH_OK);
    return p;
}
int32_t tr_factor_prog_get_view(const tr_factor_prog *p, tr_factor_program *v) {
    if (!prog_ok(p) || !v) return TR_MATH_INVALID;
    memset(v, 0, sizeof *v);
    v->nsteps = p->nsteps; v->limbs = p->limbs; v->mode = (uint32_t)p->mode; v->nfactors = p->g->nfactors;
    v->arena = p->arena; v->nopd = p->nopd; v->nstrides = p->nstrides; v->ndims = p->ndims; v->inputs = p->g->entries;
    v->step = p->step; v->opd = p->opd; v->stride = p->stride; v->dims = p->dims; v->input = p->g->entry;
    return TR_MATH_OK;
}
int32_t tr_factor_prog_candidate(const tr_factor_prog *p, int32_t *index, int32_t *device) {
    if (!prog_ok(p) || !index || !device) return TR_MATH_INVALID;
    *index = p->index; *device = p->device;
    return TR_MATH_OK;
}
int32_t tr_factor_prog_bases(const tr_factor_prog *p, uint64_t R, const uint32_t *ev, uint64_t *bases) {
    if (!prog_ok(p) || (R && p->g->nfactors && !bases) || (R && p->s.nfixed && !ev)) return TR_MATH_INVALID;
    const tr_factor_graph *g = p->g;
    const uint32_t nf = g->nfactors, nx = p->s.nfixed;
    for (uint64_t r = 0; r < R; r++) {
        const uint32_t *e = ev + r * nx;
        for (uint32_t i = 0; i < nx; i++) if (e[i] >= g->domain[p->s.fixed[i]]) return TR_MATH_INVALID;
        for (uint32_t f = 0; f < nf; f++) {
            uint64_t base = g->table_at[f];
            for (uint32_t i = 0; i < nx; i++) base += (uint64_t)e[i] * p->fstride[(uint64_t)f * nx + i];
            bases[r * nf + f] = base;
        }
    }
    return TR_MATH_OK;
}
int32_t tr_factor_prog_need(const tr_factor_prog *p, uint64_t R, int32_t target, tr_math_need *need) {
    if (!prog_ok(p) || !need || R < 1 || (target != TR_FACTOR_BY_CPU && target != TR_FACTOR_BY_DEVICE)) return TR_MATH_INVALID;
    need_zero(need);
    const uint32_t L = p->limbs;
    const int mx = p->mode == TR_FACTOR_MAX;
    uint64_t entries = tr_budget_mul(R, p->arena), words = tr_budget_mul(entries, L);
    uint64_t arena = tr_budget_add(tr_budget_mul(words, mx ? 24 : 8), mx ? tr_budget_mul(entries, 4) : 0);
    uint64_t host = tr_budget_add(arena, tr_budget_mul(tr_budget_mul(R, p->g->nfactors), 8));
    host = tr_budget_add(host, tr_budget_mul(TR_FACTOR_SCRATCH * (uint64_t)L, 8));
    need->need[TR_BUDGET_STATES] = entries;
    need->need[TR_BUDGET_BYTES] = host;
    if (target == TR_FACTOR_BY_DEVICE) {
        tr_factor_program v;
        tr_factor_prog_get_view(p, &v);
        need->need[TR_BUDGET_BYTES] = tr_budget_add(host, tr_factor_device_bytes(&v, R));
    }
    return TR_MATH_OK;
}
int32_t tr_factor_eliminate(const tr_factor_program *p, uint64_t R, const uint64_t *bases, uint64_t *vals, uint64_t *maxv,
                            uint64_t *cnts, uint32_t *arg, const tr_math_budget *budget, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (tr_factor_program_check(p) || tr_factor_bases_check(p, R, bases) || (R && !vals) ||
        (R && p->mode == TR_FACTOR_MAX && (!maxv || !cnts || !arg))) return TR_MATH_INVALID;
    tr_math_budget dflt;
    const tr_math_budget *b = budget_of(budget, &dflt);
    tr_math_need need;
    need_zero(&need);
    need.need[TR_BUDGET_STATES] = tr_budget_mul(R, p->arena);
    need.need[TR_BUDGET_BYTES] = tr_budget_mul(TR_FACTOR_SCRATCH * (uint64_t)p->limbs, 8);
    int32_t rc = check_need(b, &need, refusal);
    if (rc) return rc;
    uint64_t *scratch = (uint64_t *)zalloc(8 * TR_FACTOR_SCRATCH * (uint64_t)p->limbs);
    if (!scratch) return TR_MATH_MEMORY;
    int over = 0;
    for (uint64_t r = 0; r < R; r++)
        for (uint32_t s = 0; s < p->nsteps; s++)
            for (uint64_t o = 0; o < p->step[s].cells; o++) over |= tr_factor_cell(p, s, r, o, bases, vals, maxv, cnts, arg, scratch);
    free(scratch);
    return over ? TR_MATH_OVERFLOW : TR_MATH_OK;
}

/* ================================================================== answers */
struct tr_factor_answer_s {
    uint32_t magic;
    tr_factor_answer_info info;
    const tr_factor_graph *g;
    uint64_t gdigest[2];
    uint32_t nfixed;                  /* evidence variables (a normalizer has none) */
    int32_t *status;                  /* [R] */
    tr_bigq **values;                 /* [R x nvalues] (NULL: undefined) */
    uint64_t *asg_off;                /* [R + 1] */
    uint32_t *asg;                    /* [assignments x nvars] */
    uint64_t *tied;                   /* KBEST [R] */
    uint64_t *zscaled;                /* [R x L] the scaled Z_e */
    uint32_t L;
};
static int ans_ok(const tr_factor_answer *a) { return a && a->magic == ANS_MAGIC; }
void tr_factor_answer_free(tr_factor_answer *a) {
    if (!ans_ok(a)) return;
    uint64_t n = a->info.requests * a->info.nvalues;
    if (a->values) for (uint64_t i = 0; i < n; i++) tr_bigq_free(a->values[i]);
    free(a->values); free(a->status); free(a->asg_off); free(a->asg); free(a->tied); free(a->zscaled);
    a->magic = 0;
    free(a);
}
static tr_factor_answer *ans_new(const tr_factor_graph *g, int32_t kind, uint64_t R, uint64_t nvalues, uint32_t L) {
    tr_factor_answer *a = (tr_factor_answer *)zalloc(sizeof *a);
    if (!a) return NULL;
    a->values = (tr_bigq **)zalloc(tr_budget_mul(tr_budget_mul(R ? R : 1, nvalues ? nvalues : 1), sizeof(tr_bigq *)));
    a->status = (int32_t *)zalloc(4 * (R ? R : 1));
    a->asg_off = (uint64_t *)zalloc(8 * (R + 1));
    a->tied = (uint64_t *)zalloc(8 * (R ? R : 1));
    a->zscaled = (uint64_t *)zalloc(tr_budget_mul(tr_budget_mul(R ? R : 1, L ? L : 1), 8));
    if (!a->values || !a->status || !a->asg_off || !a->tied || !a->zscaled) {
        free(a->values); free(a->status); free(a->asg_off); free(a->tied); free(a->zscaled); free(a); return NULL;
    }
    a->g = g; a->gdigest[0] = g->digest[0]; a->gdigest[1] = g->digest[1]; a->L = L;
    a->info.kind = kind; a->info.requests = R; a->info.nvalues = nvalues; a->info.nvars = g->nvars;
    a->info.backend = TR_FACTOR_BY_CPU;
    a->magic = ANS_MAGIC;
    return a;
}
/* the big integer of L limbs */
static tr_bigint *big_of(const uint64_t *w, uint64_t L, const tr_math_budget *b, tr_math_refusal *r, int32_t *rc) {
    tr_bigint *x = tr_bigint_from_limbs(1, w, L, b, r, rc);
    if (!x && !*rc) *rc = TR_MATH_MEMORY;
    return x;
}
/* *out = num / den (canonical); the answer's limbs record grows */
static int32_t ans_q(tr_factor_answer *a, uint64_t slot, const tr_bigint *num, const tr_bigint *den,
                     const tr_math_budget *b, tr_math_refusal *r) {
    int32_t e = 0;
    tr_bigq *q = tr_bigq_from_bigints(num, den, b, r, &e);
    if (!q) return e ? e : TR_MATH_MEMORY;
    a->values[slot] = q;
    uint64_t ln = tr_bigint_limb_count(tr_bigq_numerator(q)), ld = tr_bigint_limb_count(tr_bigq_denominator(q));
    if (ln > a->info.limbs) a->info.limbs = (uint32_t)ln;
    if (ld > a->info.limbs) a->info.limbs = (uint32_t)ld;
    return TR_MATH_OK;
}
static int32_t ans_q_limbs(tr_factor_answer *a, uint64_t slot, const uint64_t *num, uint32_t L, const tr_bigint *den,
                           const tr_math_budget *b, tr_math_refusal *r) {
    int32_t rc = 0;
    tr_bigint *n = big_of(num, L, b, r, &rc);
    if (!n) return rc;
    rc = ans_q(a, slot, n, den, b, r);
    tr_bigint_free(n);
    return rc;
}
/* D = prod_f D_f */
static tr_bigint *graph_denominator(const tr_factor_graph *g, const tr_math_budget *b, tr_math_refusal *r, int32_t *rc) {
    *rc = 0;
    uint64_t one = 1;
    tr_bigint *D = tr_bigint_from_limbs(1, &one, 1, b, r, rc);
    for (uint32_t f = 0; f < g->nfactors && D; f++) {
        tr_bigint *d = tr_bigint_from_limbs(1, g->denom + 2 * (uint64_t)f, 2, b, r, rc);
        if (!d) { tr_bigint_free(D); D = NULL; break; }
        tr_bigint *x = tr_bigint_mul(D, d, b, r, rc);
        tr_bigint_free(D); tr_bigint_free(d);
        D = x;
    }
    if (!D && !*rc) *rc = TR_MATH_MEMORY;
    return D;
}
static int ans_graph(const tr_factor_answer *a, const tr_factor_graph *g) {
    return ans_ok(a) && a->g == g && a->gdigest[0] == g->digest[0] && a->gdigest[1] == g->digest[1];
}

/* the tied maximizers of request r: a depth-first walk of the steps in reverse, each branch over the tight values */
typedef struct {
    const tr_factor_program *v;
    const tr_factor_prog *p;
    uint64_t r;
    const uint64_t *bases, *vals, *maxv, *cnts;
    uint32_t *val;                    /* [n] the assignment being decoded */
    uint64_t *coords;                 /* [MAX_SCOPE] */
    uint64_t *w;                      /* scratch: sp, mp, cp, tmp (4 L) */
} decode_t;
/* the cell index of step s at the current assignment */
static uint64_t step_cell(const decode_t *d, uint32_t s, uint64_t *c) {
    const tr_factor_step *st = &d->v->step[s];
    uint64_t o = 0;
    for (uint32_t k = 0; k < st->nout; k++) {
        c[k] = d->val[d->p->outv[st->dim_at + k]];
        o = o * d->v->dims[st->dim_at + k] + c[k];
    }
    return o;
}
/* 1 when term t of step s at its current cell reaches the cell's maximum (positive) */
static int tight(decode_t *d, uint32_t s, uint64_t o, const uint64_t *c, uint64_t t) {
    const uint32_t L = d->v->limbs;
    const tr_factor_step *st = &d->v->step[s];
    int over = 0;
    if (!tr_factor_term(d->v, st, d->r, d->bases, c, t, d->vals, d->maxv, d->cnts, d->w, d->w + L, d->w + 2 * L,
                        d->w + 3 * L, &over)) return 0;
    const uint64_t *m = d->maxv + (d->r * d->v->arena + st->out_at + o) * L;
    return tr_factor_w_cmp(d->w + L, m, L) == 0 && !tr_symc_w_is_zero(m, L);
}
static int asg_cmp(const void *ctx, uint64_t a, uint64_t b) {
    const uint32_t *A = (const uint32_t *)ctx;
    const uint32_t n = A[0];
    const uint32_t *x = A + 1 + a * n, *y = A + 1 + b * n;
    for (uint32_t i = 0; i < n; i++) if (x[i] != y[i]) return x[i] < y[i] ? -1 : 1;
    return 0;
}
/* every maximizer of request r into rows (each n values, sorted lexicographically); *got */
static int32_t decode_ties(decode_t *d, uint64_t expect, uint32_t *rows, uint64_t *got) {
    const tr_factor_program *v = d->v;
    const uint32_t n = d->p->g->nvars, m = v->nsteps - 1;
    uint64_t *next = (uint64_t *)zalloc(8 * ((uint64_t)m + 1));
    if (!next) return TR_MATH_MEMORY;
    uint64_t cnt = 0;
    if (m == 0) {                                          /* nothing eliminated: the evidence is the assignment */
        memcpy(rows, d->val, 4 * (size_t)n);
        cnt = 1;
    } else {
        int64_t s = (int64_t)m - 1;                        /* steps from the last eliminated down to 0 */
        next[s] = 0;
        while (s < (int64_t)m) {
            if (s < 0) {
                if (cnt >= expect) { free(next); return TR_MATH_INVALID; }    /* more than the count: inconsistent */
                memcpy(rows + cnt * n, d->val, 4 * (size_t)n);
                cnt++;
                s = 0;
                continue;
            }
            const uint32_t st = (uint32_t)s;
            uint64_t o = step_cell(d, st, d->coords), t = next[st];
            int found = 0;
            for (; t < v->step[st].nterms; t++) if (tight(d, st, o, d->coords, t)) { found = 1; break; }
            if (found) {
                d->val[v->step[st].var] = (uint32_t)t;
                next[st] = t + 1;
                s--;
                if (s >= 0) next[s] = 0;
            } else s++;
        }
    }
    free(next);
    if (cnt != expect) return TR_MATH_INVALID;
    *got = cnt;
    return TR_MATH_OK;
}
static uint64_t limbs_u64(const uint64_t *w, uint32_t L) {
    for (uint32_t i = 1; i < L; i++) if (w[i]) return UINT64_MAX;
    return w[0];
}

tr_factor_answer *tr_factor_answer_create(const tr_factor_prog *p, int32_t kind, uint32_t given, uint64_t R,
                                          const uint32_t *evidence, const uint64_t *vals, const uint64_t *maxv,
                                          const uint64_t *cnts, const uint32_t *arg, const tr_factor_answer *norm,
                                          int32_t backend, const tr_math_budget *budget, tr_math_refusal *refusal,
                                          int32_t *error) {
    tr_budget_none(refusal);
    if (!prog_ok(p) || R < 1 || !vals || (backend != TR_FACTOR_BY_CPU && backend != TR_FACTOR_BY_DEVICE)) {
        set_error(error, TR_MATH_INVALID); return NULL;
    }
    const tr_factor_graph *g = p->g;
    if (p->gdigest[0] != g->digest[0] || p->gdigest[1] != g->digest[1]) { set_error(error, TR_MATH_INVALID); return NULL; }
    const shape_t *s = &p->s;
    const int sum = p->mode == TR_FACTOR_SUM;
    if ((sum && kind != TR_FACTOR_A_PARTITION && kind != TR_FACTOR_A_MARGINAL && kind != TR_FACTOR_A_CONDITIONAL) ||
        (!sum && kind != TR_FACTOR_A_MAX) || (kind == TR_FACTOR_A_PARTITION && s->nkeep) ||
        ((kind == TR_FACTOR_A_MARGINAL || kind == TR_FACTOR_A_CONDITIONAL) && !s->nkeep) ||
        (kind == TR_FACTOR_A_CONDITIONAL ? (given < 1 || given >= s->nkeep) : given != 0) ||
        (!sum && (!maxv || !cnts || !arg)) || (s->nfixed && !evidence)) { set_error(error, TR_MATH_INVALID); return NULL; }
    if (norm && (!ans_graph(norm, g) || norm->info.kind != TR_FACTOR_A_PARTITION || norm->nfixed || norm->info.requests != 1)) {
        set_error(error, TR_MATH_INVALID); return NULL;
    }
    tr_math_budget dflt;
    const tr_math_budget *b = budget_of(budget, &dflt);
    const uint32_t L = p->limbs, n = g->nvars;
    const tr_factor_step *fin = &p->step[p->nsteps - 1];
    const uint64_t cells = fin->cells;
    const uint64_t nv = kind == TR_FACTOR_A_MAX ? 5 : (s->nkeep ? 2 + cells : 2);
    int32_t rc = check_one(b, TR_BUDGET_OUTPUT, tr_budget_mul(R, nv), refusal);
    tr_factor_program v;
    tr_factor_prog_get_view(p, &v);
    uint64_t *bases = NULL;
    if (!rc) {
        rc = check_one(b, TR_BUDGET_BYTES, tr_budget_mul(tr_budget_mul(R, g->nfactors), 8), refusal);
        if (!rc) {
            bases = (uint64_t *)zalloc(tr_budget_mul(tr_budget_mul(R, g->nfactors ? g->nfactors : 1), 8));
            if (!bases) rc = TR_MATH_MEMORY;
        }
        if (!rc) rc = tr_factor_prog_bases(p, R, evidence, bases);
    }
    /* MAX: every maximizer is enumerated: the output budget counts them first (exactly) */
    uint64_t total_asg = 0;
    if (!rc && !sum) {
        for (uint64_t r = 0; r < R; r++) {
            const uint64_t e = r * p->arena + fin->out_at;
            if (tr_symc_w_is_zero(maxv + e * L, L)) continue;
            total_asg = tr_budget_add(total_asg, limbs_u64(cnts + e * L, L));
        }
        rc = check_one(b, TR_BUDGET_OUTPUT, tr_budget_add(tr_budget_mul(R, nv), total_asg), refusal);
        if (!rc) rc = check_one(b, TR_BUDGET_BYTES, tr_budget_mul(tr_budget_mul(total_asg, n), 8), refusal);
    }
    tr_factor_answer *a = rc ? NULL : ans_new(g, kind, R, nv, L);
    if (!rc && !a) rc = TR_MATH_MEMORY;
    if (!rc && !sum) {
        a->asg = (uint32_t *)zalloc(tr_budget_mul(tr_budget_add(tr_budget_mul(total_asg, n), 1), 4));
        if (!a->asg) rc = TR_MATH_MEMORY;
    }
    tr_bigint *D = rc ? NULL : graph_denominator(g, b, refusal, &rc);
    tr_bigint *Z = NULL;
    if (!rc && norm) Z = big_of(norm->zscaled, norm->L, b, refusal, &rc);
    const int zpos = Z && tr_bigint_sign(Z) > 0;
    decode_t dc;
    memset(&dc, 0, sizeof dc);
    uint32_t *val = NULL;
    uint64_t *coords = NULL, *w = NULL;
    if (!rc && !sum) {
        val = (uint32_t *)zalloc(4 * (uint64_t)n); coords = (uint64_t *)zalloc(8 * TR_FACTOR_MAX_SCOPE);
        w = (uint64_t *)zalloc(8 * 4 * (uint64_t)L);
        if (!val || !coords || !w) rc = TR_MATH_MEMORY;
        dc.v = &v; dc.p = p; dc.bases = bases; dc.vals = vals; dc.maxv = maxv; dc.cnts = cnts; dc.val = val;
        dc.coords = coords; dc.w = w;
    }
    uint64_t asg_at = 0;
    for (uint64_t r = 0; r < R && !rc; r++) {
        const uint64_t e = r * p->arena + fin->out_at;
        /* Z_e: the final table's sum (one cell without kept variables) */
        uint64_t *ze = a->zscaled + r * L;
        if (sum) {
            tr_symc_w_zero(ze, L);
            for (uint64_t o = 0; o < cells; o++) if (tr_symc_w_add(ze, vals + (e + o) * L, L)) rc = TR_MATH_OVERFLOW;
        } else tr_symc_w_copy(ze, vals + e * L, L);
        if (rc) break;
        tr_bigint *Ze = big_of(ze, L, b, refusal, &rc);
        if (!Ze) break;
        rc = ans_q(a, r * nv, Ze, D, b, refusal);                       /* Z_e = scaled / D */
        if (!rc && zpos) rc = ans_q(a, r * nv + 1, Ze, Z, b, refusal);  /* P(e) = Z_e / Z */
        const int zero = tr_bigint_sign(Ze) == 0;
        if (!rc && norm && !zpos) a->status[r] = TR_MATH_DOMAIN_ERROR;  /* Z = 0: the model has no mass */
        if (!rc && kind != TR_FACTOR_A_PARTITION && zero) a->status[r] = TR_MATH_DOMAIN_ERROR;   /* P(e) = 0 */
        if (!rc && !zero && (kind == TR_FACTOR_A_MARGINAL)) {
            for (uint64_t o = 0; o < cells && !rc; o++) rc = ans_q_limbs(a, r * nv + 2 + o, vals + (e + o) * L, L, Ze, b, refusal);
        }
        if (!rc && !zero && kind == TR_FACTOR_A_CONDITIONAL) {
            /* the condition: the last `given` kept variables; P(q | r) = Z(q, r) / sum_q Z(q, r) */
            uint64_t inner = 1;
            for (uint32_t i = s->nkeep - given; i < s->nkeep; i++) inner *= g->domain[s->keep[i]];
            const uint64_t outer = cells / inner;
            uint64_t *acc = (uint64_t *)zalloc(8 * (uint64_t)L);
            if (!acc) rc = TR_MATH_MEMORY;
            for (uint64_t c = 0; c < inner && !rc; c++) {
                tr_symc_w_zero(acc, L);
                for (uint64_t q = 0; q < outer; q++) if (tr_symc_w_add(acc, vals + (e + q * inner + c) * L, L)) rc = TR_MATH_OVERFLOW;
                if (rc || tr_symc_w_is_zero(acc, L)) continue;            /* undefined: the condition has probability 0 */
                tr_bigint *den = big_of(acc, L, b, refusal, &rc);
                for (uint64_t q = 0; q < outer && !rc && den; q++)
                    rc = ans_q_limbs(a, r * nv + 2 + q * inner + c, vals + (e + q * inner + c) * L, L, den, b, refusal);
                tr_bigint_free(den);
            }
            free(acc);
        }
        if (!rc && !sum && !zero) {
            const uint64_t *mx = maxv + e * L, *ct = cnts + e * L;
            rc = ans_q_limbs(a, r * nv + 2, mx, L, D, b, refusal);         /* w* */
            if (!rc) rc = ans_q_limbs(a, r * nv + 3, mx, L, Ze, b, refusal);   /* P(x* | e) */
            uint64_t one = 1;
            tr_bigint *u = rc ? NULL : tr_bigint_from_limbs(1, &one, 1, b, refusal, &rc);
            tr_bigint *cb = (rc || !u) ? NULL : big_of(ct, L, b, refusal, &rc);
            if (!rc && cb && u) rc = ans_q(a, r * nv + 4, cb, u, b, refusal);   /* the number of maximizers */
            tr_bigint_free(u); tr_bigint_free(cb);
            if (!rc) {
                for (uint32_t i = 0; i < n; i++) val[i] = 0;
                for (uint32_t i = 0; i < s->nfixed; i++) val[s->fixed[i]] = evidence[r * s->nfixed + i];
                dc.r = r;
                uint64_t got = 0, expect = limbs_u64(ct, L);
                rc = decode_ties(&dc, expect, a->asg + asg_at * n, &got);
                if (!rc && got > 1) {                                       /* sort lexicographically */
                    uint64_t *idx = (uint64_t *)zalloc(16 * got);
                    uint32_t *copy = (uint32_t *)zalloc(4 * (got * n + 1));
                    if (!idx || !copy) rc = TR_MATH_MEMORY;
                    else {
                        copy[0] = n;
                        memcpy(copy + 1, a->asg + asg_at * n, 4 * (size_t)(got * n));
                        for (uint64_t i = 0; i < got; i++) idx[i] = i;
                        msort(idx, got, asg_cmp, copy, idx + got);
                        for (uint64_t i = 0; i < got; i++) memcpy(a->asg + (asg_at + i) * n, copy + 1 + idx[i] * n, 4 * (size_t)n);
                    }
                    free(idx); free(copy);
                }
                asg_at += got;
            }
        }
        a->asg_off[r + 1] = asg_at;
        tr_bigint_free(Ze);
    }
    (void)arg;
    free(bases); free(val); free(coords); free(w);
    tr_bigint_free(D); tr_bigint_free(Z);
    if (rc) { tr_factor_answer_free(a); set_error(error, rc); return NULL; }
    a->nfixed = s->nfixed;
    a->info.backend = backend; a->info.mode = p->mode; a->info.cells = s->nkeep ? cells : 0; a->info.assignments = asg_at;
    a->info.nkeep = s->nkeep; a->info.ngiven = given;
    set_error(error, TR_MATH_OK);
    return a;
}

/* ================================================================== the k best assignments (CPU) */
/* A list holds the assignments of largest weight of a message cell: every element whose weight is at least the K-th
 * largest (the top K with ties: closed under products and maxima of nonnegative weights, so the final list is exactly
 * {x : weight(x) >= the K-th weight}, whatever the tie order); zero weights are never held. */
typedef struct {
    uint64_t n, cap;
    uint32_t L, nmsg;
    uint64_t *val;                    /* [cap x L] */
    uint32_t *t;                      /* [cap] */
    uint32_t *child;                  /* [cap x nmsg] list positions of the message operands' elements */
} pool_t;
typedef struct {
    const tr_math_budget *b;
    tr_math_refusal *r;
    uint64_t bytes;                   /* held by lists and pools */
} kb_budget;
static void pool_release(kb_budget *kb, pool_t *p) {
    if (kb) kb->bytes -= p->cap * (8 * (uint64_t)p->L + 4 + 4 * (uint64_t)p->nmsg);
    free(p->val); free(p->t); free(p->child);
    p->val = NULL; p->t = NULL; p->child = NULL; p->n = p->cap = 0;
}
static int32_t pool_reserve(kb_budget *kb, pool_t *p, uint64_t need) {
    if (need <= p->cap) return TR_MATH_OK;
    uint64_t cap = p->cap ? p->cap : 8;
    while (cap < need) cap = tr_budget_mul(cap, 2);
    const uint64_t per = 8 * (uint64_t)p->L + 4 + 4 * (uint64_t)p->nmsg;
    uint64_t grow = tr_budget_mul(cap - p->cap, per);
    int32_t rc = check_one(kb->b, TR_BUDGET_BYTES, tr_budget_add(kb->bytes, grow), kb->r);
    if (rc) return rc;
    uint64_t *v = (uint64_t *)realloc(p->val, (size_t)(cap * p->L * 8));
    if (!v) return TR_MATH_MEMORY;
    p->val = v;
    uint32_t *t = (uint32_t *)realloc(p->t, (size_t)(cap * 4));
    if (!t) return TR_MATH_MEMORY;
    p->t = t;
    uint32_t *c = (uint32_t *)realloc(p->child, (size_t)(cap * (p->nmsg ? p->nmsg : 1) * 4));
    if (!c) return TR_MATH_MEMORY;
    p->child = c;
    kb->bytes = tr_budget_add(kb->bytes, grow);
    p->cap = cap;
    return TR_MATH_OK;
}
static int32_t pool_push(kb_budget *kb, pool_t *p, const uint64_t *val, uint32_t t, const uint32_t *child) {
    int32_t rc = pool_reserve(kb, p, p->n + 1);
    if (rc) return rc;
    tr_symc_w_copy(p->val + p->n * p->L, val, p->L);
    p->t[p->n] = t;
    if (p->nmsg) memcpy(p->child + p->n * p->nmsg, child, 4 * (size_t)p->nmsg);
    p->n++;
    return TR_MATH_OK;
}
static int pool_cmp(const void *ctx, uint64_t a, uint64_t b) {
    const pool_t *p = (const pool_t *)ctx;
    int c = tr_factor_w_cmp(p->val + a * p->L, p->val + b * p->L, p->L);
    if (c) return -c;                                     /* larger weights first */
    if (p->t[a] != p->t[b]) return p->t[a] < p->t[b] ? -1 : 1;
    for (uint32_t j = 0; j < p->nmsg; j++)
        if (p->child[a * p->nmsg + j] != p->child[b * p->nmsg + j]) return p->child[a * p->nmsg + j] < p->child[b * p->nmsg + j] ? -1 : 1;
    return 0;
}
/* sorts the pool and keeps the top K with ties (in order) */
static int32_t pool_cut(kb_budget *kb, pool_t *p, uint64_t K) {
    if (!p->n) return TR_MATH_OK;
    uint64_t *idx = (uint64_t *)zalloc(16 * p->n);
    if (!idx) return TR_MATH_MEMORY;
    for (uint64_t i = 0; i < p->n; i++) idx[i] = i;
    msort(idx, p->n, pool_cmp, p, idx + p->n);
    uint64_t keep = p->n;
    if (p->n > K) {
        keep = K;
        const uint64_t *kth = p->val + idx[K - 1] * p->L;
        while (keep < p->n && tr_factor_w_cmp(p->val + idx[keep] * p->L, kth, p->L) == 0) keep++;
    }
    pool_t q;
    memset(&q, 0, sizeof q);
    q.L = p->L; q.nmsg = p->nmsg;
    int32_t rc = pool_reserve(kb, &q, keep);
    for (uint64_t i = 0; i < keep && !rc; i++) rc = pool_push(kb, &q, p->val + idx[i] * p->L, p->t[idx[i]], p->child + idx[i] * p->nmsg);
    free(idx);
    if (rc) { pool_release(kb, &q); return rc; }
    pool_release(kb, p);
    *p = q;
    return TR_MATH_OK;
}
typedef struct {
    uint64_t *off;                    /* [cells + 1] */
    pool_t e;                         /* the cells' elements, in cell order */
} klist_t;
static int32_t kb_step(const tr_factor_program *v, uint32_t s, const uint64_t *bases, klist_t *lists, uint64_t K,
                       kb_budget *kb, uint64_t *maxlen) {
    const tr_factor_step *st = &v->step[s];
    const uint32_t L = v->limbs;
    uint32_t nmsg = 0;
    for (uint32_t j = 0; j < st->nopd; j++) nmsg += v->opd[st->opd_at + j].kind == TR_FACTOR_OPD_MESSAGE;
    klist_t *out = &lists[s];
    out->e.L = L; out->e.nmsg = nmsg;
    int32_t rc = check_one(kb->b, TR_BUDGET_BYTES, tr_budget_add(kb->bytes, tr_budget_mul(st->cells + 1, 8)), kb->r);
    if (rc) return rc;
    out->off = (uint64_t *)zalloc(8 * (st->cells + 1));
    if (!out->off) return TR_MATH_MEMORY;
    kb->bytes = tr_budget_add(kb->bytes, 8 * (st->cells + 1));
    pool_t cell, P, Q;
    memset(&cell, 0, sizeof cell); memset(&P, 0, sizeof P); memset(&Q, 0, sizeof Q);
    cell.L = P.L = Q.L = L; cell.nmsg = P.nmsg = Q.nmsg = nmsg;
    uint64_t *scal = (uint64_t *)zalloc(8 * 3 * (uint64_t)L), *tmp = scal + L, *prod = scal + 2 * L;
    uint32_t *ch = (uint32_t *)zalloc(4 * ((uint64_t)nmsg + 1));
    uint64_t c[TR_FACTOR_MAX_SCOPE];
    if (!scal || !ch) rc = TR_MATH_MEMORY;
    int over = 0;
    for (uint64_t o = 0; o < st->cells && !rc; o++) {
        tr_factor_coords(v, st, o, c);
        cell.n = 0;
        for (uint64_t t = 0; t < st->nterms && !rc; t++) {
            tr_factor_w_one(scal, L);
            int zero = 0;
            for (uint32_t j = 0; j < st->nopd && !zero; j++) {
                const tr_factor_operand *op = &v->opd[st->opd_at + j];
                if (op->kind != TR_FACTOR_OPD_INPUT) continue;
                uint64_t e = tr_factor_opd_entry(v, st, op, 0, bases, c, t);
                uint64_t lo = v->input[2 * e], hi = v->input[2 * e + 1];
                if (!lo && !hi) { zero = 1; break; }
                over |= tr_symc_w_mul128(scal, L, lo, hi, tmp); tr_symc_w_copy(scal, tmp, L);
            }
            if (zero) continue;
            P.n = 0;
            for (uint32_t j = 0; j < nmsg; j++) ch[j] = 0;
            rc = pool_push(kb, &P, scal, (uint32_t)t, ch);
            uint32_t mi = 0;
            for (uint32_t j = 0; j < st->nopd && !rc && P.n; j++) {
                const tr_factor_operand *op = &v->opd[st->opd_at + j];
                if (op->kind != TR_FACTOR_OPD_MESSAGE) continue;
                uint64_t cc = tr_factor_opd_entry(v, st, op, 0, bases, c, t) - v->step[op->index].out_at;
                const klist_t *C = &lists[op->index];
                Q.n = 0;
                for (uint64_t a = 0; a < P.n && !rc; a++)
                    for (uint64_t q = C->off[cc]; q < C->off[cc + 1] && !rc; q++) {
                        over |= tr_symc_w_mul(P.val + a * L, C->e.val + q * L, L, prod);
                        memcpy(ch, P.child + a * nmsg, 4 * (size_t)nmsg);
                        ch[mi] = (uint32_t)(q - C->off[cc]);
                        rc = pool_push(kb, &Q, prod, (uint32_t)t, ch);
                    }
                if (!rc) rc = pool_cut(kb, &Q, K);
                pool_t sw = P; P = Q; Q = sw;
                mi++;
            }
            for (uint64_t a = 0; a < P.n && !rc; a++) rc = pool_push(kb, &cell, P.val + a * L, (uint32_t)t, P.child + a * nmsg);
        }
        if (!rc) rc = pool_cut(kb, &cell, K);
        if (!rc && cell.n > *maxlen) *maxlen = cell.n;
        if (!rc) rc = check_one(kb->b, TR_BUDGET_SUPPORT, cell.n, kb->r);
        out->off[o] = out->e.n;
        for (uint64_t a = 0; a < cell.n && !rc; a++) rc = pool_push(kb, &out->e, cell.val + a * L, cell.t[a], cell.child + a * nmsg);
        out->off[o + 1] = out->e.n;
    }
    pool_release(kb, &cell); pool_release(kb, &P); pool_release(kb, &Q);
    free(scal); free(ch);
    if (!rc && over) rc = TR_MATH_OVERFLOW;
    return rc;
}
/* the assignment of element q of the final list (the evidence already in val) */
static int32_t kb_decode(const tr_factor_program *v, const tr_factor_prog *p, const uint64_t *bases, const klist_t *lists,
                         uint64_t q, uint32_t *val, uint64_t *stack) {
    /* stack entries: (step, cell, element position) */
    uint64_t sp = 0;
    stack[0] = v->nsteps - 1; stack[1] = 0; stack[2] = q; sp = 1;
    uint64_t c[TR_FACTOR_MAX_SCOPE];
    while (sp) {
        sp--;
        uint32_t s = (uint32_t)stack[3 * sp];
        uint64_t o = stack[3 * sp + 1], e = stack[3 * sp + 2];
        const tr_factor_step *st = &v->step[s];
        const klist_t *Ls = &lists[s];
        uint32_t t = Ls->e.t[e];
        if (!st->final) val[st->var] = t;
        tr_factor_coords(v, st, o, c);
        uint32_t mi = 0;
        for (uint32_t j = 0; j < st->nopd; j++) {
            const tr_factor_operand *op = &v->opd[st->opd_at + j];
            if (op->kind != TR_FACTOR_OPD_MESSAGE) continue;
            uint64_t cc = tr_factor_opd_entry(v, st, op, 0, bases, c, t) - v->step[op->index].out_at;
            uint64_t pos = lists[op->index].off[cc] + Ls->e.child[e * Ls->e.nmsg + mi];
            if (sp >= (uint64_t)v->nsteps) return TR_MATH_INVALID;
            stack[3 * sp] = op->index; stack[3 * sp + 1] = cc; stack[3 * sp + 2] = pos; sp++;
            mi++;
        }
    }
    (void)p;
    return TR_MATH_OK;
}
typedef struct { const uint32_t *rows; const uint64_t *w; uint32_t n, L; } kbsort_t;
static int kbsort_cmp(const void *ctx, uint64_t a, uint64_t b) {
    const kbsort_t *k = (const kbsort_t *)ctx;
    int c = tr_factor_w_cmp(k->w + a * k->L, k->w + b * k->L, k->L);
    if (c) return -c;
    for (uint32_t i = 0; i < k->n; i++) if (k->rows[a * k->n + i] != k->rows[b * k->n + i]) return k->rows[a * k->n + i] < k->rows[b * k->n + i] ? -1 : 1;
    return 0;
}
tr_factor_answer *tr_factor_kbest(const tr_factor_prog *p, uint64_t R, const uint32_t *evidence, uint64_t k,
                                  const tr_factor_answer *norm, const tr_math_budget *budget, tr_math_refusal *refusal,
                                  int32_t *error) {
    tr_budget_none(refusal);
    if (!prog_ok(p) || R < 1 || k < 1 || p->mode != TR_FACTOR_MAX || (p->s.nfixed && !evidence)) {
        set_error(error, TR_MATH_INVALID); return NULL;
    }
    const tr_factor_graph *g = p->g;
    if (p->gdigest[0] != g->digest[0] || p->gdigest[1] != g->digest[1] ||
        (norm && (!ans_graph(norm, g) || norm->info.kind != TR_FACTOR_A_PARTITION || norm->nfixed || norm->info.requests != 1))) {
        set_error(error, TR_MATH_INVALID); return NULL;
    }
    tr_math_budget dflt;
    const tr_math_budget *b = budget_of(budget, &dflt);
    const uint32_t L = p->limbs, n = g->nvars;
    const uint64_t nv = tr_budget_add(2, tr_budget_mul(2, k));
    int32_t rc = check_one(b, TR_BUDGET_OUTPUT, tr_budget_mul(R, nv), refusal);
    tr_factor_program v;
    tr_factor_prog_get_view(p, &v);
    tr_math_need need;
    if (!rc) rc = tr_factor_prog_need(p, R, TR_FACTOR_BY_CPU, &need);
    if (!rc) rc = check_need(b, &need, refusal);
    uint64_t entries = tr_budget_mul(R, p->arena), words = tr_budget_mul(entries, L);
    uint64_t *bases = NULL, *vals = NULL, *maxv = NULL, *cnts = NULL;
    uint32_t *arg = NULL;
    if (!rc) {
        bases = (uint64_t *)zalloc(tr_budget_mul(tr_budget_mul(R, g->nfactors ? g->nfactors : 1), 8));
        vals = (uint64_t *)zalloc(tr_budget_mul(words, 8)); maxv = (uint64_t *)zalloc(tr_budget_mul(words, 8));
        cnts = (uint64_t *)zalloc(tr_budget_mul(words, 8)); arg = (uint32_t *)zalloc(tr_budget_mul(entries, 4));
        if (!bases || !vals || !maxv || !cnts || !arg) rc = TR_MATH_MEMORY;
    }
    if (!rc) rc = tr_factor_prog_bases(p, R, evidence, bases);
    if (!rc) rc = tr_factor_eliminate(&v, R, bases, vals, maxv, cnts, arg, b, refusal);    /* Z_e (the sum chain) */
    tr_factor_answer *a = rc ? NULL : ans_new(g, TR_FACTOR_A_KBEST, R, nv, L);
    if (!rc && !a) rc = TR_MATH_MEMORY;
    tr_bigint *D = rc ? NULL : graph_denominator(g, b, refusal, &rc);
    tr_bigint *Z = NULL;
    if (!rc && norm) Z = big_of(norm->zscaled, norm->L, b, refusal, &rc);
    const int zpos = Z && tr_bigint_sign(Z) > 0;
    klist_t *lists = rc ? NULL : (klist_t *)zalloc(sizeof(klist_t) * (uint64_t)v.nsteps);
    uint64_t *stack = rc ? NULL : (uint64_t *)zalloc(24 * ((uint64_t)v.nsteps + 1));
    if (!rc && (!lists || !stack)) rc = TR_MATH_MEMORY;
    kb_budget kb = {b, refusal, 0};
    uint32_t *rows = NULL;
    uint64_t asg_total = 0, asg_cap = 0;
    for (uint64_t r = 0; r < R && !rc; r++) {
        const uint64_t e0 = r * p->arena + p->step[p->nsteps - 1].out_at;
        uint64_t *ze = a->zscaled + r * L;
        tr_symc_w_copy(ze, vals + e0 * L, L);
        tr_bigint *Ze = big_of(ze, L, b, refusal, &rc);
        if (!Ze) break;
        rc = ans_q(a, r * nv, Ze, D, b, refusal);
        if (!rc && zpos) rc = ans_q(a, r * nv + 1, Ze, Z, b, refusal);
        if (!rc && norm && !zpos) a->status[r] = TR_MATH_DOMAIN_ERROR;
        if (!rc && tr_bigint_sign(Ze) == 0) a->status[r] = TR_MATH_DOMAIN_ERROR;
        if (!rc && !a->status[r]) {
            uint64_t maxlen = 0;
            const uint64_t *rb = bases + r * g->nfactors;
            for (uint32_t s = 0; s < v.nsteps && !rc; s++) rc = kb_step(&v, s, rb, lists, k, &kb, &maxlen);
            const klist_t *F = rc ? NULL : &lists[v.nsteps - 1];
            uint64_t S = F ? F->off[1] : 0;
            if (!rc) rc = check_one(b, TR_BUDGET_OUTPUT, tr_budget_add(tr_budget_mul(R, nv), tr_budget_add(asg_total, S)), refusal);
            uint32_t *row = NULL;
            uint64_t *idx = NULL, *w = NULL;
            if (!rc) {
                row = (uint32_t *)zalloc(4 * (S * (uint64_t)n + 1)); idx = (uint64_t *)zalloc(16 * (S + 1));
                w = (uint64_t *)zalloc(8 * (S * (uint64_t)L + 1));
                if (!row || !idx || !w) rc = TR_MATH_MEMORY;
            }
            for (uint64_t q = 0; q < S && !rc; q++) {
                uint32_t *val = row + q * n;
                for (uint32_t i = 0; i < n; i++) val[i] = 0;
                for (uint32_t i = 0; i < p->s.nfixed; i++) val[p->s.fixed[i]] = evidence[r * p->s.nfixed + i];
                rc = kb_decode(&v, p, rb, lists, q, val, stack);
                tr_symc_w_copy(w + q * L, F->e.val + q * L, L);
                idx[q] = q;
            }
            if (!rc && S > 1) { kbsort_t ks = {row, w, n, L}; msort(idx, S, kbsort_cmp, &ks, idx + S); }
            uint64_t got = S < k ? S : k, tied = 0;
            if (!rc && got)
                for (uint64_t q = got; q < S; q++) if (tr_factor_w_cmp(w + idx[q] * L, w + idx[got - 1] * L, L) == 0) tied++;
            if (!rc && asg_total + got > asg_cap) {
                uint64_t cap = asg_cap ? asg_cap : 16;
                while (cap < asg_total + got) cap *= 2;
                uint32_t *nr = (uint32_t *)realloc(rows, (size_t)(cap * n * 4 + 4));
                if (!nr) rc = TR_MATH_MEMORY;
                else { rows = nr; asg_cap = cap; }
            }
            for (uint64_t q = 0; q < got && !rc; q++) {
                memcpy(rows + (asg_total + q) * n, row + idx[q] * n, 4 * (size_t)n);
                rc = ans_q_limbs(a, r * nv + 2 + 2 * q, w + idx[q] * L, L, D, b, refusal);
                if (!rc) rc = ans_q_limbs(a, r * nv + 3 + 2 * q, w + idx[q] * L, L, Ze, b, refusal);
            }
            if (!rc) { asg_total += got; a->tied[r] = tied; }
            free(row); free(idx); free(w);
            for (uint32_t s = 0; s < v.nsteps; s++) {
                if (lists[s].off) { kb.bytes -= 8 * (p->step[s].cells + 1); free(lists[s].off); lists[s].off = NULL; }
                pool_release(&kb, &lists[s].e);
            }
        }
        a->asg_off[r + 1] = asg_total;
        tr_bigint_free(Ze);
    }
    if (lists) for (uint32_t s = 0; s < v.nsteps; s++) { free(lists[s].off); pool_release(NULL, &lists[s].e); }
    free(lists); free(stack); free(bases); free(vals); free(maxv); free(cnts); free(arg);
    tr_bigint_free(D); tr_bigint_free(Z);
    if (rc) { free(rows); tr_factor_answer_free(a); set_error(error, rc); return NULL; }
    a->asg = rows ? rows : (uint32_t *)zalloc(4);
    if (!a->asg) { tr_factor_answer_free(a); set_error(error, TR_MATH_MEMORY); return NULL; }
    a->nfixed = p->s.nfixed;
    a->info.mode = TR_FACTOR_KBEST; a->info.assignments = asg_total;
    set_error(error, TR_MATH_OK);
    return a;
}

/* ================================================================== alternative backends (COUNT-002, TENSOR-002) */
/* factor f restricted by the evidence: its free scope (in its order) and entries (C order over the free scope) */
typedef struct {
    uint32_t nfree;                   /* free variables of the graph */
    uint32_t *fid;                    /* [n] free index (NONE32: fixed) */
    uint32_t *val;                    /* [n] the evidence value of a fixed variable */
    uint32_t *fscope;                 /* [nf x MAX_SCOPE] free scope (free indices) */
    uint32_t *fn;                     /* [nf] */
    uint64_t *fat, *fsize;            /* [nf] entry offsets, sizes */
    u128 *ent;                        /* restricted scaled entries */
    uint64_t total;
} restricted_t;
static void restricted_release(restricted_t *x) {
    free(x->fid); free(x->val); free(x->fscope); free(x->fn); free(x->fat); free(x->fsize); free(x->ent);
    memset(x, 0, sizeof *x);
}
static int32_t restrict_graph(const tr_factor_graph *g, uint32_t nfixed, const uint32_t *fixed, const uint32_t *values,
                              const tr_math_budget *b, tr_math_refusal *r, restricted_t *x) {
    memset(x, 0, sizeof *x);
    const uint32_t n = g->nvars, nf = g->nfactors;
    if (nfixed > n || (nfixed && (!fixed || !values))) return TR_MATH_INVALID;
    x->fid = (uint32_t *)zalloc(4 * (uint64_t)n); x->val = (uint32_t *)zalloc(4 * (uint64_t)n);
    x->fscope = (uint32_t *)zalloc(4 * ((uint64_t)nf + 1) * TR_FACTOR_MAX_SCOPE); x->fn = (uint32_t *)zalloc(4 * ((uint64_t)nf + 1));
    x->fat = (uint64_t *)zalloc(8 * ((uint64_t)nf + 1)); x->fsize = (uint64_t *)zalloc(8 * ((uint64_t)nf + 1));
    if (!x->fid || !x->val || !x->fscope || !x->fn || !x->fat || !x->fsize) { restricted_release(x); return TR_MATH_MEMORY; }
    for (uint32_t v = 0; v < n; v++) x->fid[v] = 0;
    for (uint32_t i = 0; i < nfixed; i++) {
        if (fixed[i] >= n || x->fid[fixed[i]] == NONE32 || values[i] >= g->domain[fixed[i]]) { restricted_release(x); return TR_MATH_INVALID; }
        x->fid[fixed[i]] = NONE32; x->val[fixed[i]] = values[i];
    }
    for (uint32_t v = 0; v < n; v++) if (x->fid[v] != NONE32) x->fid[v] = x->nfree++;
    uint64_t total = 0;
    for (uint32_t f = 0; f < nf; f++) {
        uint64_t size = 1;
        for (uint32_t t = 0; t < g->nscope[f]; t++) {
            uint32_t v = g->scope[g->scope_at[f] + t];
            if (x->fid[v] == NONE32) continue;
            x->fscope[(uint64_t)f * TR_FACTOR_MAX_SCOPE + x->fn[f]++] = v;
            size *= g->domain[v];
        }
        x->fat[f] = total; x->fsize[f] = size;
        total = tr_budget_add(total, size);
    }
    int32_t rc = check_one(b, TR_BUDGET_BYTES, tr_budget_mul(total, 16), r);
    if (rc) { restricted_release(x); return rc; }
    x->ent = (u128 *)zalloc(tr_budget_mul(total ? total : 1, 16));
    if (!x->ent) { restricted_release(x); return TR_MATH_MEMORY; }
    x->total = total;
    for (uint32_t f = 0; f < nf; f++) {
        uint64_t base = g->table_at[f];
        for (uint32_t t = 0; t < g->nscope[f]; t++) {
            uint32_t v = g->scope[g->scope_at[f] + t];
            if (x->fid[v] == NONE32) base += (uint64_t)x->val[v] * factor_stride(g, f, v);
        }
        uint32_t c[TR_FACTOR_MAX_SCOPE];
        for (uint32_t t = 0; t < x->fn[f]; t++) c[t] = 0;
        for (uint64_t e = 0; e < x->fsize[f]; e++) {
            uint64_t off = base;
            for (uint32_t t = 0; t < x->fn[f]; t++) off += (uint64_t)c[t] * factor_stride(g, f, x->fscope[(uint64_t)f * TR_FACTOR_MAX_SCOPE + t]);
            x->ent[x->fat[f] + e] = get128(g->entry + 2 * off);
            for (uint32_t t = x->fn[f]; t-- > 0;) {
                if (++c[t] < g->domain[x->fscope[(uint64_t)f * TR_FACTOR_MAX_SCOPE + t]]) break;
                c[t] = 0;
            }
        }
    }
    return TR_MATH_OK;
}
/* an entry over D_f as a canonical int64 rational, or 0 when it does not fit */
static int entry_q64(u128 E, u128 D, int64_t *num, int64_t *den) {
    if (!E) { *num = 0; *den = 1; return 1; }
    u128 h = gcd128(E, D);
    E /= h; D /= h;
    if (E > (u128)INT64_MAX || D > (u128)INT64_MAX) return 0;
    *num = (int64_t)E; *den = (int64_t)D;
    return 1;
}
/* a COUNT-002 answer's value i as a big rational into a's slot */
static int32_t adopt_symc(tr_factor_answer *a, uint64_t slot, const tr_symc_answer *s, uint32_t i, const tr_bigint *mul,
                          const tr_math_budget *b, tr_math_refusal *r) {
    tr_symc_answer_info si;
    int32_t rc = tr_symc_answer_get_info(s, &si);
    if (rc) return rc;
    uint32_t cap = si.limbs ? si.limbs : 1, nl = 0, dl = 0;
    uint64_t *num = (uint64_t *)zalloc(8 * (uint64_t)cap), *den = (uint64_t *)zalloc(8 * (uint64_t)cap);
    if (!num || !den) { free(num); free(den); return TR_MATH_MEMORY; }
    rc = tr_symc_answer_value(s, i, num, den, cap, &nl, &dl);
    tr_bigint *N = NULL, *Dd = NULL, *M = NULL;
    if (!rc) N = tr_bigint_from_limbs(1, num, nl, b, r, &rc);
    if (!rc && N) Dd = tr_bigint_from_limbs(1, den, dl ? dl : 1, b, r, &rc);
    if (!rc && N && Dd && mul) { M = tr_bigint_mul(N, mul, b, r, &rc); }
    if (!rc && N && Dd) rc = ans_q(a, slot, M ? M : N, Dd, b, r);
    if (!rc && (!N || !Dd)) rc = TR_MATH_MEMORY;
    tr_bigint_free(N); tr_bigint_free(Dd); tr_bigint_free(M);
    free(num); free(den);
    return rc;
}
static int32_t alt_symcount(const tr_factor_graph *g, const restricted_t *x, int32_t method, const tr_math_budget *b,
                            tr_math_refusal *r, tr_factor_answer *a) {
    const uint32_t nf = g->nfactors, nv = x->nfree ? x->nfree : 1;   /* COUNT-002 needs one variable at least */
    uint32_t *dom = (uint32_t *)zalloc(4 * (uint64_t)nv), *scope = (uint32_t *)zalloc(4 * ((uint64_t)nf * TR_FACTOR_MAX_SCOPE + 1));
    tr_symc_factor *fs = (tr_symc_factor *)zalloc(sizeof(tr_symc_factor) * ((uint64_t)nf + 1));
    int64_t *num = (int64_t *)zalloc(8 * (x->total + 1)), *den = (int64_t *)zalloc(8 * (x->total + 1));
    int32_t rc = (!dom || !scope || !fs || !num || !den) ? TR_MATH_MEMORY : TR_MATH_OK;
    if (!rc) {
        dom[0] = 1;
        for (uint32_t v = 0; v < g->nvars; v++) if (x->fid[v] != NONE32) dom[x->fid[v]] = g->domain[v];
        uint64_t sat = 0;
        for (uint32_t f = 0; f < nf && !rc; f++) {
            fs[f].scope_at = sat; fs[f].table_at = x->fat[f]; fs[f].nscope = x->fn[f]; fs[f].reserved = 0;
            for (uint32_t t = 0; t < x->fn[f]; t++) scope[sat++] = x->fid[x->fscope[(uint64_t)f * TR_FACTOR_MAX_SCOPE + t]];
            u128 D = get128(g->denom + 2 * (uint64_t)f);
            for (uint64_t e = 0; e < x->fsize[f] && !rc; e++)
                if (!entry_q64(x->ent[x->fat[f] + e], D, &num[x->fat[f] + e], &den[x->fat[f] + e])) rc = TR_MATH_UNSUPPORTED;
        }
    }
    tr_symc_factors *F = NULL;
    tr_symc_answer *S = NULL;
    if (!rc) F = tr_symc_factors_create(nv, dom, nf, fs, scope, num, den, x->total, b, r, &rc);
    if (!rc && F) S = method == TR_FACTOR_ALT_SYMBRUTE ? tr_symc_factors_brute(F, TR_SYMC_WEIGHTED, 0, b, r, &rc)
                                                       : tr_symc_factors_count(F, TR_SYMC_ORDER_MIN_FILL, NULL,
                                                                               TR_SYMC_WEIGHTED, 0, b, r, &rc);
    if (!rc && S) rc = adopt_symc(a, 0, S, 0, NULL, b, r);
    tr_symc_answer_free(S); tr_symc_factors_free(F);
    free(dom); free(scope); free(fs); free(num); free(den);
    return rc;
}
/* COUNT-002's decision diagram of the parameter encoding: binary free variables x; per factor entry w != 1 a
 * constraint NOT [scope = a] (w = 0) or theta XNOR [scope = a] with literal weights (w, 1); Z_e = the weighted count
 * of the conjunction's TRUE class */
static int32_t alt_diagram(const tr_factor_graph *g, const restricted_t *x, const tr_math_budget *b, tr_math_refusal *r,
                           tr_factor_answer *a) {
    const uint32_t nf = g->nfactors, m = x->nfree;
    for (uint32_t v = 0; v < g->nvars; v++) if (x->fid[v] != NONE32 && g->domain[v] != 2) return TR_MATH_UNSUPPORTED;
    uint64_t nparams = 0, nodes_cap = 4 + 2 * (uint64_t)m;
    for (uint32_t f = 0; f < nf; f++) {
        u128 D = get128(g->denom + 2 * (uint64_t)f);
        for (uint64_t e = 0; e < x->fsize[f]; e++) {
            u128 E = x->ent[x->fat[f] + e];
            if (E == D) continue;
            if (E) nparams++;
            nodes_cap += (uint64_t)x->fn[f] + 4;
        }
    }
    uint64_t k = m + nparams;
    if (k > TR_SYMC_MAX_PROGRAM_VARS) return TR_MATH_UNSUPPORTED;
    if (k == 0) k = 1;                                   /* a dummy variable of weights (1, 0) */
    tr_bool_node *nodes = (tr_bool_node *)zalloc(sizeof(tr_bool_node) * nodes_cap);
    tr_symc_weight *w = (tr_symc_weight *)zalloc(sizeof(tr_symc_weight) * k);
    if (!nodes || !w) { free(nodes); free(w); return TR_MATH_MEMORY; }
    int32_t nn = 0, rc = TR_MATH_OK;
#define NODE(o, A, B) (nodes[nn].op = (o), nodes[nn].a = (A), nodes[nn].b = (B), nodes[nn].reserved = 0, nn++)
    int32_t *pos = (int32_t *)zalloc(4 * ((uint64_t)m + 1)), *neg = (int32_t *)zalloc(4 * ((uint64_t)m + 1));
    if (!pos || !neg) rc = TR_MATH_MEMORY;
    for (uint32_t i = 0; i < m && !rc; i++) { pos[i] = NODE(TR_BOOL_VAR, (int32_t)i, -1); neg[i] = NODE(TR_BOOL_NOT, pos[i], -1); }
    for (uint64_t i = 0; i < k; i++) { w[i].num1 = 1; w[i].den1 = 1; w[i].num0 = 1; w[i].den0 = 1; }
    if (m + nparams == 0) w[0].num0 = 0;
    int32_t truth = rc ? 0 : NODE(TR_BOOL_CONST, 1, -1), root = truth;
    uint32_t theta = m;
    for (uint32_t f = 0; f < nf && !rc; f++) {
        u128 D = get128(g->denom + 2 * (uint64_t)f);
        uint32_t c[TR_FACTOR_MAX_SCOPE];
        for (uint32_t t = 0; t < x->fn[f]; t++) c[t] = 0;
        for (uint64_t e = 0; e < x->fsize[f] && !rc; e++) {
            u128 E = x->ent[x->fat[f] + e];
            if (E != D) {
                int32_t match = truth;
                for (uint32_t t = 0; t < x->fn[f]; t++) {
                    uint32_t fv = x->fid[x->fscope[(uint64_t)f * TR_FACTOR_MAX_SCOPE + t]];
                    int32_t lit = c[t] ? pos[fv] : neg[fv];
                    match = match == truth ? lit : NODE(TR_BOOL_AND, match, lit);
                }
                int32_t cons;
                if (!E) cons = NODE(TR_BOOL_NOT, match, -1);
                else {
                    int64_t qn, qd;
                    if (!entry_q64(E, D, &qn, &qd)) { rc = TR_MATH_UNSUPPORTED; break; }
                    w[theta].num1 = qn; w[theta].den1 = qd;
                    int32_t th = NODE(TR_BOOL_VAR, (int32_t)theta, -1);
                    theta++;
                    int32_t xr = NODE(TR_BOOL_XOR, th, match);
                    cons = NODE(TR_BOOL_NOT, xr, -1);
                }
                root = root == truth ? cons : NODE(TR_BOOL_AND, root, cons);
            }
            for (uint32_t t = x->fn[f]; t-- > 0;) {
                if (++c[t] < 2) break;
                c[t] = 0;
            }
        }
    }
#undef NODE
    tr_bool_program *prog = NULL;
    tr_symc_dd *dd = NULL;
    tr_symc_answer *S = NULL;
    if (!rc) prog = tr_bool_program_create((int32_t)k, nodes, nn, root, b, r, &rc);
    tr_bool_program_view pv;
    if (!rc && prog && tr_bool_program_get_view(prog, &pv)) rc = TR_MATH_INVALID;
    if (!rc && prog) dd = tr_symc_compile(&pv, TR_SYMC_ORDER_DFS, NULL, b, r, &rc);
    if (!rc && dd) S = tr_symc_dd_count(dd, w, b, r, &rc);
    if (!rc && S) rc = adopt_symc(a, 0, S, 0, NULL, b, r);           /* the TRUE class */
    tr_symc_answer_free(S); tr_symc_dd_free(dd); tr_bool_program_free(prog);
    free(nodes); free(w); free(pos); free(neg);
    return rc;
}
/* TENSOR-002: the dense scaled tables (free scopes) contracted to a scalar along its own costed plan */
static int32_t alt_tensor(const tr_factor_graph *g, const restricted_t *x, const tr_math_budget *b, tr_math_refusal *r,
                          tr_factor_answer *a) {
    const uint32_t nf = g->nfactors;
    uint8_t *inany = (uint8_t *)zalloc((uint64_t)x->nfree + 1);
    int32_t *label = (int32_t *)zalloc(4 * ((uint64_t)x->nfree + 1));
    if (!inany || !label) { free(inany); free(label); return TR_MATH_MEMORY; }
    uint32_t nt = 0, nl = 0;
    int32_t rc = TR_MATH_OK;
    for (uint32_t f = 0; f < nf; f++) {
        if (!x->fn[f]) continue;
        nt++;
        if (x->fn[f] > TR_TENSOR_MAX_ORDER) rc = TR_MATH_UNSUPPORTED;
        for (uint32_t t = 0; t < x->fn[f]; t++) inany[x->fid[x->fscope[(uint64_t)f * TR_FACTOR_MAX_SCOPE + t]]] = 1;
    }
    for (uint32_t i = 0; i < x->nfree; i++) label[i] = inany[i] ? (int32_t)nl++ : -1;
    if (nt > TR_TENSOR_NET_MAX || nl > TR_TENSOR_MAX_LABELS) rc = TR_MATH_UNSUPPORTED;
    uint64_t one = 1;
    tr_bigint *mul = rc ? NULL : tr_bigint_from_limbs(1, &one, 1, b, r, &rc);
    /* constant factors and the variables no table reads multiply natively */
    for (uint32_t f = 0; f < nf && !rc && mul; f++) {
        if (x->fn[f]) continue;
        uint64_t wv[2];
        put128(wv, x->ent[x->fat[f]]);
        tr_bigint *c = tr_bigint_from_limbs(1, wv, 2, b, r, &rc), *m2 = c ? tr_bigint_mul(mul, c, b, r, &rc) : NULL;
        tr_bigint_free(c);
        if (m2) { tr_bigint_free(mul); mul = m2; }
    }
    for (uint32_t v = 0; v < g->nvars && !rc && mul; v++) {
        if (x->fid[v] == NONE32 || inany[x->fid[v]]) continue;
        uint64_t d = g->domain[v];
        tr_bigint *c = tr_bigint_from_limbs(1, &d, 1, b, r, &rc), *m2 = c ? tr_bigint_mul(mul, c, b, r, &rc) : NULL;
        tr_bigint_free(c);
        if (m2) { tr_bigint_free(mul); mul = m2; }
    }
    tr_tensor *T[TR_TENSOR_NET_MAX];
    memset(T, 0, sizeof T);
    tr_tensor_net_spec spec;
    memset(&spec, 0, sizeof spec);
    uint32_t ti = 0;
    for (uint32_t f = 0; f < nf && !rc && nt; f++) {
        if (!x->fn[f]) continue;
        uint64_t dims[TR_TENSOR_MAX_ORDER];
        for (uint32_t t = 0; t < x->fn[f]; t++) {
            uint32_t v = x->fscope[(uint64_t)f * TR_FACTOR_MAX_SCOPE + t];
            dims[t] = g->domain[v];
            spec.labels[ti][t] = label[x->fid[v]];
        }
        spec.nlabels[ti] = (int32_t)x->fn[f];
        tr_math_i128 *vals = (tr_math_i128 *)zalloc(16 * x->fsize[f]);
        if (!vals) { rc = TR_MATH_MEMORY; break; }
        for (uint64_t e = 0; e < x->fsize[f]; e++) vals[e] = tr_i128_set((__int128)x->ent[x->fat[f] + e]);
        T[ti] = tr_tensor_from_dense((int32_t)x->fn[f], dims, 1, vals, NULL, b, r, &rc);
        free(vals);
        if (!T[ti]) break;
        ti++;
    }
    spec.ntensors = (int32_t)ti; spec.noutput = 0;
    tr_tensor_net *net = NULL;
    tr_tensor_plan *plan = NULL;
    tr_tensor *out = NULL;
    tr_bigint *val = NULL;
    if (!rc && nt) net = tr_tensor_net_create((const tr_tensor *const *)T, &spec, b, r, &rc);
    if (!rc && net) { int32_t pc = tr_tensor_net_plan(net, NULL, b, &plan, r); if (pc) rc = pc; }
    if (!rc && plan) out = tr_tensor_net_execute(net, plan, -1, b, NULL, r, &rc);
    if (!rc && nt && out) {
        tr_math_i128 v;
        uint64_t cnt = 0;
        rc = tr_tensor_get(out, 0, NULL, &v, &cnt);
        __int128 z = tr_i128_get(v);
        if (!rc && z < 0) rc = TR_MATH_INVALID;
        uint64_t wv[2];
        put128(wv, (u128)z);
        if (!rc) val = tr_bigint_from_limbs(1, wv, 2, b, r, &rc);
    }
    tr_bigint *D = rc ? NULL : graph_denominator(g, b, r, &rc);
    if (!rc && mul && D) {
        tr_bigint *num = val ? tr_bigint_mul(val, mul, b, r, &rc) : tr_bigint_copy(mul, b, r, &rc);
        if (num) rc = ans_q(a, 0, num, D, b, r);
        tr_bigint_free(num);
    }
    tr_bigint_free(D); tr_bigint_free(val); tr_bigint_free(mul);
    tr_tensor_free(out); tr_tensor_plan_free(plan); tr_tensor_net_free(net);
    for (uint32_t i = 0; i < TR_TENSOR_NET_MAX; i++) tr_tensor_free(T[i]);
    free(inany); free(label);
    return rc;
}
tr_factor_answer *tr_factor_alternative(const tr_factor_graph *g, int32_t method, uint32_t nfixed, const uint32_t *fixed,
                                        const uint32_t *values, const tr_math_budget *budget, tr_math_refusal *refusal,
                                        int32_t *error) {
    tr_budget_none(refusal);
    if (!graph_ok(g) || method < TR_FACTOR_ALT_SYMCOUNT || method > TR_FACTOR_ALT_TENSOR) { set_error(error, TR_MATH_INVALID); return NULL; }
    tr_math_budget dflt;
    const tr_math_budget *b = budget_of(budget, &dflt);
    restricted_t x;
    int32_t rc = restrict_graph(g, nfixed, fixed, values, b, refusal, &x);
    if (rc) { set_error(error, rc); return NULL; }
    tr_factor_answer *a = ans_new(g, TR_FACTOR_A_COUNT, 1, 1, 1);
    if (!a) rc = TR_MATH_MEMORY;
    if (!rc && method == TR_FACTOR_ALT_DIAGRAM) rc = alt_diagram(g, &x, b, refusal, a);
    else if (!rc && method == TR_FACTOR_ALT_TENSOR) rc = alt_tensor(g, &x, b, refusal, a);
    else if (!rc) rc = alt_symcount(g, &x, method, b, refusal, a);
    restricted_release(&x);
    if (rc) { tr_factor_answer_free(a); set_error(error, rc); return NULL; }
    a->info.method = method; a->nfixed = nfixed;
    a->asg = (uint32_t *)zalloc(4);
    if (!a->asg) { tr_factor_answer_free(a); set_error(error, TR_MATH_MEMORY); return NULL; }
    set_error(error, TR_MATH_OK);
    return a;
}

/* ================================================================== answer accessors */
int32_t tr_factor_answer_get_info(const tr_factor_answer *a, tr_factor_answer_info *out) {
    if (!ans_ok(a) || !out) return TR_MATH_INVALID;
    *out = a->info;
    return TR_MATH_OK;
}
int32_t tr_factor_answer_status(const tr_factor_answer *a, uint64_t r) {
    if (!ans_ok(a) || r >= a->info.requests) return TR_MATH_INVALID;
    return a->status[r];
}
int32_t tr_factor_answer_value(const tr_factor_answer *a, uint64_t r, uint64_t i, uint64_t *num, uint64_t *den,
                               uint32_t cap, uint32_t *nlimbs, uint32_t *dlimbs) {
    if (!ans_ok(a) || r >= a->info.requests || i >= a->info.nvalues || !nlimbs || !dlimbs || (cap && (!num || !den)))
        return TR_MATH_INVALID;
    const tr_bigq *q = a->values[r * a->info.nvalues + i];
    *nlimbs = 0; *dlimbs = 0;
    if (!q) return 0;
    int32_t sign = 0;
    uint64_t c1 = 0, c2 = 0;
    if (tr_bigint_limb_count(tr_bigq_numerator(q)) > cap || tr_bigint_limb_count(tr_bigq_denominator(q)) > cap)
        return TR_MATH_INVALID;
    int32_t rc = tr_bigint_export(tr_bigq_numerator(q), &sign, num, cap, &c1);
    if (!rc) rc = tr_bigint_export(tr_bigq_denominator(q), &sign, den, cap, &c2);
    if (rc) return rc;
    *nlimbs = (uint32_t)c1; *dlimbs = (uint32_t)c2;
    return 1;
}
int32_t tr_factor_answer_column(const tr_factor_answer *a, uint64_t i, uint64_t r0, uint64_t n, uint64_t *num,
                                uint64_t *den, uint32_t cap, uint32_t *nlimbs, uint32_t *dlimbs) {
    if (!ans_ok(a) || i >= a->info.nvalues || r0 > a->info.requests || n > a->info.requests - r0 ||
        (n && (!num || !den || !nlimbs || !dlimbs)) || cap < (a->info.limbs ? a->info.limbs : 1)) return TR_MATH_INVALID;
    for (uint64_t k = 0; k < n; k++) {
        int32_t rc = tr_factor_answer_value(a, r0 + k, i, num + k * cap, den + k * cap, cap, nlimbs + k, dlimbs + k);
        if (rc < 0) return rc;
        if (rc == 0) { nlimbs[k] = 0; dlimbs[k] = 0; }
    }
    return TR_MATH_OK;
}
int32_t tr_factor_answer_assignments(const tr_factor_answer *a, uint64_t r, uint32_t *out, uint64_t capacity,
                                     uint64_t *count) {
    if (!ans_ok(a) || r >= a->info.requests || !count) return TR_MATH_INVALID;
    uint64_t b = a->asg_off[r], e = a->asg_off[r + 1], n = (e - b) * a->info.nvars;
    *count = e - b;
    if (capacity < n || (n && !out)) return TR_MATH_INVALID;
    if (n) memcpy(out, a->asg + b * a->info.nvars, 4 * (size_t)n);
    return TR_MATH_OK;
}
int32_t tr_factor_answer_tied(const tr_factor_answer *a, uint64_t r, uint64_t *tied) {
    if (!ans_ok(a) || r >= a->info.requests || !tied) return TR_MATH_INVALID;
    *tied = a->tied[r];
    return TR_MATH_OK;
}
