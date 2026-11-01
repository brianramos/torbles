/* PAIR-002: structured pairwise query planning (support aggregation, finite-feature factorization through KERNEL-001,
 * sorted-prefix evaluation through PAIR-001, bounded direct pairs), costed through FND-008; see math_pairplan.h.
 * SPDX-License-Identifier: Apache-2.0 */
#include "math_pairplan.h"
#include "math_planner.h"
#include "mathematics.h"
#include "math_domain.h"
#include "math_handles.h"
#include <stdlib.h>
#include <string.h>

TR_MATH_ABI(tr_pairplan_spec, TR_PAIRPLAN_ABI);         /* the ctypes boundary of PAIR-002 (math_handles.h) */
TR_MATH_ABI(tr_pairplan_request, TR_PAIRPLAN_ABI);
TR_MATH_ABI(tr_pairplan_result, TR_PAIRPLAN_ABI);
TR_MATH_ABI(tr_pairplan_totals, TR_PAIRPLAN_ABI);
TR_MATH_ABI(tr_pairplan_table_view, TR_PAIRPLAN_ABI);
TR_MATH_ABI(tr_pairplan_kernel_view, TR_PAIRPLAN_ABI);
TR_MATH_ABI(tr_pairplan_options, TR_PAIRPLAN_ABI);
TR_MATH_ABI(tr_pairplan_candidate, TR_PAIRPLAN_ABI);
TR_MATH_ABI(tr_pairplan_plan_info, TR_PAIRPLAN_ABI);
TR_MATH_ABI(tr_pairplan_exec, TR_PAIRPLAN_ABI);
TR_MATH_ABI(tr_pairplan_state_info, TR_PAIRPLAN_ABI);

_Static_assert(sizeof(tr_pairplan_spec) == 2808, "tr_pairplan_spec ABI");
_Static_assert(sizeof(tr_pairplan_request) == 24, "tr_pairplan_request ABI");
_Static_assert(sizeof(tr_pairplan_result) == TR_PAIRPLAN_RESULT_BYTES, "tr_pairplan_result ABI");
_Static_assert(sizeof(tr_pairplan_totals) == 80, "tr_pairplan_totals ABI");
_Static_assert(sizeof(tr_pairplan_table_view) == 216, "tr_pairplan_table_view ABI");
_Static_assert(sizeof(tr_pairplan_kernel_view) == 1096, "tr_pairplan_kernel_view ABI");
_Static_assert(sizeof(tr_pairplan_options) == 224, "tr_pairplan_options ABI");
_Static_assert(sizeof(tr_pairplan_candidate) == 472, "tr_pairplan_candidate ABI");
_Static_assert(sizeof(tr_pairplan_plan_info) == 104, "tr_pairplan_plan_info ABI");
_Static_assert(sizeof(tr_pairplan_exec) == 64, "tr_pairplan_exec ABI");
_Static_assert(sizeof(tr_pairplan_state_info) == 136, "tr_pairplan_state_info ABI");
_Static_assert(TR_PAIRPLAN_MAX_COLUMNS <= 8, "null masks are bytes");

#define STATE_MAGIC UINT32_C(0x54535050)
#define PLAN_MAGIC UINT32_C(0x4c505050)

struct tr_pairplan_state_s {
    uint32_t magic;
    int32_t strategy, form, policy, ncols, weight, repr, mass, kind;
    int32_t cols[TR_PAIRPLAN_MAX_COLUMNS];
    const tr_scope *scope;                 /* borrowed identity: the scope must outlive the state */
    uint64_t ngroups, bytes, elements, nentries, largest;
    /* SUPPORT / ROWS */
    uint64_t *offset;
    int64_t *value;
    uint8_t *nulls;
    tr_math_i128 *W;
    tr_dist_i256 *Q;
    uint64_t *count;
    tr_pairplan_totals *totals;
    int64_t lo[TR_PAIRPLAN_MAX_COLUMNS], hi[TR_PAIRPLAN_MAX_COLUMNS];
    /* PREFIX */
    tr_dist_hist *hist;
    tr_dist_prefix *prefix;
    /* FEATURES */
    tr_kernel_spec kernel;
    tr_kernel_state *features;
    tr_kernel_eval *eval;
};

/* a validated spec: the policy, the kernel ready for evaluation (TABLE: an owned canonical copy) */
typedef struct {
    int32_t policy, kdeg;
    uint64_t ops;                          /* ops(k) of the cost model */
    tr_kernel_spec kernel;                 /* FEATURE: canonical (tr_kernel_check) */
    tr_pairplan_kernel_view kv;
    int64_t *table;
    uint64_t table_bytes;
} prepared;

struct tr_pairplan_plan_s {
    uint32_t magic;
    const tr_scope *scope;
    tr_pairplan_spec spec;
    prepared pr;
    uint64_t n, G, elements, selected, rows, bytes;
    tr_pairplan_request *req;
    int32_t *status;                       /* per request: 0, -1 invalid, -3 unproven */
    tr_pairplan_totals *tot;               /* the profile, per group */
    uint64_t *dsup, *dpre;                 /* per group: support entries, prefix entries (exact or bounds) */
    int64_t lo[TR_PAIRPLAN_MAX_COLUMNS], hi[TR_PAIRPLAN_MAX_COLUMNS];
    int32_t nullseen[TR_PAIRPLAN_MAX_COLUMNS];
    int32_t profile_reused, nretained, devices, forced;
    const tr_pairplan_state *use[TR_PAIRPLAN_STRATEGIES + 1][2];   /* [strategy][dense 0 / sparse 1] */
    int32_t use_index[TR_PAIRPLAN_STRATEGIES + 1][2];
    int32_t ncand, chosen, code, reason;
    tr_math_refusal refusal;
    tr_pairplan_candidate cand[TR_PAIRPLAN_MAX_CANDIDATES];
};

static void set_error(int32_t *error, int32_t code) { if (error) *error = code; }
static void *zalloc(uint64_t bytes) {
    if (bytes > (uint64_t)SIZE_MAX / 2) return NULL;
    return calloc(1, bytes ? (size_t)bytes : 1);
}
static void need_zero(tr_math_need *n) { for (int i = 0; i < TR_BUDGET_RESOURCES; i++) n->need[i] = 0; }
static const tr_math_budget *budget_or(const tr_math_budget *b, tr_math_budget *dflt) {
    if (b) return b;
    tr_budget_default(dflt);
    return dflt;
}
static int state_ok(const tr_pairplan_state *s) { return s && s->magic == STATE_MAGIC; }
static int plan_ok(const tr_pairplan_plan *p) { return p && p->magic == PLAN_MAGIC; }
static uint64_t umin(uint64_t a, uint64_t b) { return a < b ? a : b; }
static uint64_t umax(uint64_t a, uint64_t b) { return a > b ? a : b; }
static uint64_t bits64(uint64_t x) { return tr_pairplan_bitlen((unsigned __int128)x); }
static uint64_t sat_add(uint64_t a, uint64_t b) { return tr_budget_add(a, b); }
static uint64_t sat_mul(uint64_t a, uint64_t b) { return tr_budget_mul(a, b); }

void tr_pairplan_default_rates(double rates[2][TR_COST_COMPONENTS]) {
    /* WORDS, SCAN, CHAIN, COMBINE, CONSTRUCT, REUSE, FUSION, MATERIALIZE, TRANSFER, CPU_FINAL, GPU_FINAL, FIXED: hand-set
     * relative estimates (about a nanosecond of the 2-core development machine per unit on the CPU; the CUDA rates
     * are uncalibrated estimates for a discrete device, with the host's rates for the host-side table build), not
     * times */
    static const double cpu[TR_COST_COMPONENTS] = {0.1, 0.5, 1.0, 1.0, 2.0, 10.0, 0, 0.05, 0, 1.0, 0, 2000.0};
    static const double cuda[TR_COST_COMPONENTS] = {0.1, 0.5, 0.02, 0.02, 2.0, 10.0, 0, 0.05, 0.12, 1.0, 0.02, 25000.0};
    if (!rates) return;
    memcpy(rates[0], cpu, sizeof cpu);
    memcpy(rates[1], cuda, sizeof cuda);
}

/* ------------------------------------------------------------------ a merge sort of indices with a context */
typedef int (*index_cmp)(const void *ctx, uint64_t a, uint64_t b);
static void index_sort(uint64_t *idx, uint64_t *tmp, uint64_t n, const void *ctx, index_cmp cmp) {
    uint64_t *src = idx, *dst = tmp;
    for (uint64_t w = 1; w < n; w *= 2) {
        for (uint64_t lo = 0; lo < n; lo += 2 * w) {
            uint64_t mid = umin(lo + w, n), hi = umin(lo + 2 * w, n), i = lo, j = mid, k = lo;
            while (i < mid && j < hi) dst[k++] = cmp(ctx, src[j], src[i]) < 0 ? src[j++] : src[i++];   /* stable */
            while (i < mid) dst[k++] = src[i++];
            while (j < hi) dst[k++] = src[j++];
        }
        uint64_t *t = src; src = dst; dst = t;
        if (w > n / 2) break;
    }
    if (src != idx) memcpy(idx, src, (size_t)n * 8);
}

/* ------------------------------------------------------------------ the spec */
typedef struct { const int64_t *rows; uint64_t width; int32_t n; } rows_ctx;
static int cmp_table_rows(const void *vctx, uint64_t a, uint64_t b) {
    const rows_ctx *c = (const rows_ctx *)vctx;
    const int64_t *x = c->rows + a * c->width, *y = c->rows + b * c->width;
    int r = tr_pairplan_tuple_cmp(x, y, c->n);
    return r ? r : tr_pairplan_tuple_cmp(x + c->n, y + c->n, c->n);
}
static void prepared_free(prepared *p) { if (p) { free(p->table); p->table = NULL; } }
static int scale_kind(int32_t kind) {
    return kind == TR_PAIRPLAN_K_ABS || kind == TR_PAIRPLAN_K_MANN_WHITNEY || kind == TR_PAIRPLAN_K_BAND ||
           kind == TR_PAIRPLAN_K_SQDIST;
}
/* validates the spec over the scope and prepares the kernel (see tr_pairplan_make's codes); with_table: a TABLE
 * kernel's canonical copy (a state build reads only the policy and skips it) */
static int32_t prepare(const tr_scope *scope, const tr_pairplan_spec *s, const tr_math_budget *budget,
                       tr_math_refusal *refusal, int with_table, prepared *p) {
    memset(p, 0, sizeof *p);
    tr_scope_view v;
    if (!scope || !s || tr_scope_get_view(scope, &v)) return TR_MATH_INVALID;
    if (s->kind >= TR_PAIRPLAN_K_RBF && s->kind <= TR_PAIRPLAN_K_OTHER) return TR_MATH_UNSUPPORTED;   /* no exact value */
    if (s->kind < TR_PAIRPLAN_K_ABS || s->kind > TR_PAIRPLAN_K_SQDIST || s->ncols < 1) return TR_MATH_INVALID;
    if (s->ncols > TR_PAIRPLAN_MAX_COLUMNS) return TR_MATH_UNSUPPORTED;
    const int32_t n = s->ncols;
    for (int32_t j = 0; j < n; j++) {
        if (s->cols[j] < 0 || s->cols[j] >= v.ncols) return TR_MATH_INVALID;
        for (int32_t i = 0; i < j; i++) if (s->cols[i] == s->cols[j]) return TR_MATH_INVALID;
    }
    if (s->weight != -1 && (s->weight < 0 || s->weight >= v.ncols)) return TR_MATH_INVALID;
    if (s->kind != TR_PAIRPLAN_K_TABLE && (s->symmetric || s->ntable || s->table || s->dflt)) return TR_MATH_INVALID;
    if ((s->kind != TR_PAIRPLAN_K_BAND && s->h) || s->h < 0) return TR_MATH_INVALID;
    if (s->kind == TR_PAIRPLAN_K_MANN_WHITNEY && n != 1) return TR_MATH_INVALID;
    if (s->kind != TR_PAIRPLAN_K_FEATURE && s->kernel.kind != 0) return TR_MATH_INVALID;
    if (scale_kind(s->kind))
        for (int32_t j = 1; j < n; j++) if (v.col[s->cols[j]].scale != v.col[s->cols[0]].scale) return TR_MATH_DOMAIN_ERROR;
    tr_pairplan_kernel_view *kv = &p->kv;
    memset(kv, 0, sizeof *kv);
    kv->kind = s->kind; kv->ncols = n; kv->symmetric = 1; kv->h = s->h;
    p->policy = TR_PAIRPLAN_LISTWISE;
    switch (s->kind) {
    case TR_PAIRPLAN_K_ABS: p->ops = (uint64_t)n; break;
    case TR_PAIRPLAN_K_MANN_WHITNEY: kv->symmetric = 0; p->ops = 1; break;
    case TR_PAIRPLAN_K_BAND: p->ops = (uint64_t)n; break;
    case TR_PAIRPLAN_K_SQDIST: p->ops = 2 * (uint64_t)n; break;
    case TR_PAIRPLAN_K_FEATURE: {
        tr_kernel_spec k = s->kernel;
        if (k.kind >= TR_KERNEL_RBF && k.kind <= TR_KERNEL_OTHER) return TR_MATH_UNSUPPORTED;   /* never approximated */
        if (k.ncols != n) return TR_MATH_INVALID;
        for (int32_t j = 0; j < n; j++) if (k.cols[j] != s->cols[j]) return TR_MATH_INVALID;
        int32_t rc = tr_kernel_check(scope, &k);
        if (rc) return rc;
        p->kernel = k;
        kv->fkind = k.kind; kv->degree = k.degree; kv->c = k.c; kv->nterms = k.nterms;
        if (k.kind == TR_KERNEL_DELTA) { p->policy = TR_PAIRPLAN_CATEGORY; p->ops = (uint64_t)n; }
        else if (k.kind == TR_KERNEL_POLY) { p->ops = (uint64_t)n + 4 * (uint64_t)k.degree; p->kdeg = k.degree; }
        else {
            p->kdeg = k.degree;
            for (int32_t t = 0; t < k.nterms; t++) {
                uint64_t d = 0;
                kv->weight[t] = k.weight[t];
                for (int32_t j = 0; j < n; j++) { kv->exps[t][j] = k.exps[t][j]; d += k.exps[t][j]; }
                p->ops += 1 + 4 * d;
            }
        }
        break;
    }
    case TR_PAIRPLAN_K_TABLE: {
        if ((s->symmetric != 0 && s->symmetric != 1) || (s->ntable && !s->table)) return TR_MATH_INVALID;
        if (!with_table) { kv->symmetric = s->symmetric; break; }
        const uint64_t width = 2 * (uint64_t)n + 1, T = s->ntable;
        tr_math_need need;
        need_zero(&need);
        need.need[TR_BUDGET_VARIABLES] = 2 * (uint64_t)n;
        need.need[TR_BUDGET_SUPPORT] = T;
        p->table_bytes = sat_mul(T, sat_mul(width, 8));
        need.need[TR_BUDGET_BYTES] = sat_add(sat_mul(p->table_bytes, 2), sat_mul(T, 16));
        tr_math_budget dflt;
        int32_t rc = tr_budget_check(budget_or(budget, &dflt), &need, refusal);
        if (rc) return rc;
        int64_t *raw = (int64_t *)zalloc(p->table_bytes);
        uint64_t *idx = (uint64_t *)zalloc(T * 8), *tmp = (uint64_t *)zalloc(T * 8);
        p->table = (int64_t *)zalloc(p->table_bytes);
        if (!raw || !idx || !tmp || !p->table) { free(raw); free(idx); free(tmp); prepared_free(p); return TR_MATH_MEMORY; }
        if (T) memcpy(raw, s->table, (size_t)p->table_bytes);
        uint32_t top = tr_pairplan_bitlen(tr_abs_u64(s->dflt));
        for (uint64_t r = 0; r < T; r++) {
            int64_t *row = raw + r * width;
            if (s->symmetric && tr_pairplan_tuple_cmp(row + n, row, n) < 0)
                for (int32_t j = 0; j < n; j++) { int64_t t = row[j]; row[j] = row[n + j]; row[n + j] = t; }
            uint32_t b = tr_pairplan_bitlen(tr_abs_u64(row[2 * n]));
            if (b > top) top = b;
            idx[r] = r;
        }
        rows_ctx ctx = {raw, width, n};
        index_sort(idx, tmp, T, &ctx, cmp_table_rows);
        int dup = 0;
        for (uint64_t r = 0; r < T; r++) {
            memcpy(p->table + r * width, raw + idx[r] * width, (size_t)(width * 8));
            if (r && !cmp_table_rows(&ctx, idx[r - 1], idx[r])) dup = 1;
        }
        free(raw); free(idx); free(tmp);
        if (dup) { prepared_free(p); return TR_MATH_INVALID; }          /* a pair declared twice */
        kv->symmetric = s->symmetric; kv->dflt = s->dflt; kv->ntable = T; kv->table = p->table;
        kv->tbits = (int32_t)top + 1;
        p->ops = 1 + 2 * (uint64_t)n * bits64(T);
        break;
    }
    default:
        return TR_MATH_INVALID;
    }
    kv->policy = p->policy;
    return TR_MATH_OK;
}

/* ------------------------------------------------------------------ elements of a scope */
/* whether selected row r is an element (weight present; LISTWISE: every feature present), its weight and NULL mask */
static int element(const tr_scope_view *v, const tr_pairplan_spec *s, int32_t policy, uint64_t r, int64_t *w,
                   uint8_t *nulls) {
    if (s->weight >= 0) { if (!tr_scope_present(v, s->weight, r)) return 0; *w = tr_scope_value(v, s->weight, r); }
    else *w = 1;
    uint8_t m = 0;
    for (int32_t j = 0; j < s->ncols; j++)
        if (!tr_scope_present(v, s->cols[j], r)) { if (policy == TR_PAIRPLAN_LISTWISE) return 0; m |= (uint8_t)(1u << j); }
    *nulls = m;
    return 1;
}
#define FOR_SELECTED(v, r) for (uint64_t w_ = 0, nw_ = ((v)->rows + 63) / 64; w_ < nw_; w_++) \
    for (uint64_t b_ = (v)->selection[w_], r = 0; b_ && ((r = (w_ << 6) | (uint64_t)__builtin_ctzll(b_)), 1); b_ &= b_ - 1)
static void totals_add(tr_pairplan_totals *t, int64_t w) {
    __int128 x = w;
    t->rows++;
    t->W = tr_i128_set(tr_i128_get(t->W) + x);                       /* |sum| < rows 2^63 < 2^127 */
    t->A = tr_i128_set(tr_i128_get(t->A) + (x < 0 ? -x : x));
    tr_pair_add(&t->Q, tr_pair_u((unsigned __int128)(x * x)));       /* < rows 2^126 < 2^190 */
}
static void extremes_init(int64_t *lo, int64_t *hi) {
    for (int j = 0; j < TR_PAIRPLAN_MAX_COLUMNS; j++) { lo[j] = INT64_MAX; hi[j] = INT64_MIN; }
}
/* the profile (per group: elements, W, A, Q) and the elements' value extremes, by one pass over the scope */
static void profile_scope(const tr_scope_view *v, const tr_pairplan_spec *s, int32_t policy, tr_pairplan_totals *tot,
                          int64_t *lo, int64_t *hi, int32_t *nullseen) {
    extremes_init(lo, hi);
    for (int j = 0; j < TR_PAIRPLAN_MAX_COLUMNS; j++) nullseen[j] = 0;
    if (!v->rows) return;
    FOR_SELECTED(v, r) {
        int64_t w;
        uint8_t m;
        if (!element(v, s, policy, r, &w, &m)) continue;
        tr_pairplan_totals *t = &tot[tr_scope_group(v, r)];
        totals_add(t, w);
        t->entries = t->rows;
        for (int32_t j = 0; j < s->ncols; j++) {
            if ((m >> j) & 1) { nullseen[j] = 1; continue; }
            int64_t x = tr_scope_value(v, s->cols[j], r);
            if (x < lo[j]) lo[j] = x;
            if (x > hi[j]) hi[j] = x;
        }
    }
}

/* ------------------------------------------------------------------ SUPPORT and ROWS tables */
static void state_release(tr_pairplan_state *s) {
    if (!s) return;
    free(s->offset); free(s->value); free(s->nulls); free(s->W); free(s->Q); free(s->count); free(s->totals);
    tr_dist_prefix_free(s->prefix); tr_dist_hist_free(s->hist);
    tr_kernel_eval_free(s->eval); tr_kernel_free(s->features);
    s->magic = 0;
    free(s);
}
void tr_pairplan_state_free(tr_pairplan_state *s) { if (state_ok(s)) state_release(s); }
static tr_pairplan_state *state_new(const tr_scope *scope, const tr_pairplan_spec *spec, int32_t strategy, int32_t policy) {
    tr_pairplan_state *s = (tr_pairplan_state *)zalloc(sizeof *s);
    if (!s) return NULL;
    s->strategy = strategy; s->policy = policy; s->ncols = spec->ncols; s->weight = spec->weight; s->scope = scope;
    s->kind = spec->kind;
    for (int32_t j = 0; j < spec->ncols; j++) s->cols[j] = spec->cols[j];
    extremes_init(s->lo, s->hi);
    s->magic = STATE_MAGIC;
    return s;
}
static int table_alloc(tr_pairplan_state *s, int32_t form, uint64_t G, uint64_t N) {
    s->form = form; s->ngroups = G; s->nentries = N;
    s->offset = (uint64_t *)zalloc((G + 1) * 8);
    s->value = (int64_t *)zalloc(N * 8 * (uint64_t)s->ncols);
    s->W = (tr_math_i128 *)zalloc(N * 16);
    s->totals = (tr_pairplan_totals *)zalloc(G * sizeof(tr_pairplan_totals));
    int ok = s->offset && s->value && s->W && s->totals;
    if (s->policy == TR_PAIRPLAN_CATEGORY) { s->nulls = (uint8_t *)zalloc(N); ok = ok && s->nulls; }
    if (form == TR_PAIRPLAN_FORM_SUPPORT) {
        s->Q = (tr_dist_i256 *)zalloc(N * sizeof(tr_dist_i256));
        s->count = (uint64_t *)zalloc(N * 8);
        ok = ok && s->Q && s->count;
    }
    s->bytes = sat_add(sizeof *s, tr_pairplan_table_bytes(form, s->policy, s->ncols, G, N));
    return ok;
}
static void table_extremes(tr_pairplan_state *s) {
    for (uint64_t e = 0; e < s->nentries; e++)
        for (int32_t j = 0; j < s->ncols; j++) {
            if (s->nulls && ((s->nulls[e] >> j) & 1)) continue;
            int64_t x = s->value[e * (uint64_t)s->ncols + (uint64_t)j];
            if (x < s->lo[j]) s->lo[j] = x;
            if (x > s->hi[j]) s->hi[j] = x;
        }
    s->largest = 0;
    for (uint64_t g = 0; g < s->ngroups; g++) s->largest = umax(s->largest, s->offset[g + 1] - s->offset[g]);
}
/* records of the support sort: [0] group, [1] NULL mask, [2] w, [3..) the values (0 where NULL) */
typedef struct { const int64_t *rec; uint64_t width; int32_t n; } rec_ctx;
static int cmp_records(const void *vctx, uint64_t a, uint64_t b) {
    const rec_ctx *c = (const rec_ctx *)vctx;
    const int64_t *x = c->rec + a * c->width, *y = c->rec + b * c->width;
    if (x[0] != y[0]) return (uint64_t)x[0] < (uint64_t)y[0] ? -1 : 1;
    uint64_t nx = (uint64_t)x[1], ny = (uint64_t)y[1];
    for (int32_t j = 0; j < c->n; j++) {
        int xn = (int)((nx >> j) & 1), yn = (int)((ny >> j) & 1);
        if (xn != yn) return xn ? 1 : -1;                               /* NULL after every value */
        if (!xn && x[3 + j] != y[3 + j]) return x[3 + j] < y[3 + j] ? -1 : 1;
    }
    return 0;
}
/* the bytes a SUPPORT build's first stage holds: group counts, records, the sort's two index arrays */
static uint64_t support_scratch_bytes(uint64_t G, uint64_t N, int32_t ncols) {
    return sat_add(sat_mul(G, 8), sat_mul(N, sat_add(sat_mul(3 + (uint64_t)ncols, 8), 16)));
}
static uint64_t rows_build_bytes(int32_t policy, int32_t ncols, uint64_t G, uint64_t N) {
    return sat_add(sat_add(sat_mul(G, 8), sizeof(tr_pairplan_state)),
                   tr_pairplan_table_bytes(TR_PAIRPLAN_FORM_ROWS, policy, ncols, G, N));
}
static tr_pairplan_state *table_build(const tr_scope *scope, const tr_pairplan_spec *spec, int32_t form, int32_t policy,
                                      const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error) {
    tr_scope_view v;
    tr_scope_get_view(scope, &v);
    const uint64_t G = v.ngroups;
    const int32_t n = spec->ncols;
    tr_math_budget dflt;
    budget = budget_or(budget, &dflt);
    tr_math_need need;
    need_zero(&need);
    need.need[TR_BUDGET_BYTES] = sat_mul(G, 8);
    int32_t rc = tr_budget_check(budget, &need, refusal);
    if (rc) { set_error(error, rc); return NULL; }
    uint64_t *cnt = (uint64_t *)zalloc(G * 8), N = 0;
    if (!cnt) { set_error(error, TR_MATH_MEMORY); return NULL; }
    if (v.rows) FOR_SELECTED(&v, r) {
        int64_t w;
        uint8_t m;
        if (element(&v, spec, policy, r, &w, &m)) cnt[tr_scope_group(&v, r)]++;
    }
    for (uint64_t g = 0; g < G; g++) N += cnt[g];
    need_zero(&need);
    need.need[TR_BUDGET_VARIABLES] = (uint64_t)n + (spec->weight >= 0);
    need.need[TR_BUDGET_STATES] = N;
    need.need[TR_BUDGET_BYTES] = form == TR_PAIRPLAN_FORM_ROWS ? rows_build_bytes(policy, n, G, N)
                                                               : support_scratch_bytes(G, N, n);
    rc = tr_budget_check(budget, &need, refusal);
    if (rc) { free(cnt); set_error(error, rc); return NULL; }
    tr_pairplan_state *s = state_new(scope, spec, form == TR_PAIRPLAN_FORM_ROWS ? TR_PAIRPLAN_S_DIRECT : TR_PAIRPLAN_S_SUPPORT,
                                     policy);
    if (!s) { free(cnt); set_error(error, TR_MATH_MEMORY); return NULL; }
    s->elements = N;
    if (form == TR_PAIRPLAN_FORM_ROWS) {
        if (!table_alloc(s, form, G, N)) { free(cnt); state_release(s); set_error(error, TR_MATH_MEMORY); return NULL; }
        for (uint64_t g = 0; g < G; g++) s->offset[g + 1] = s->offset[g] + cnt[g];
        memcpy(cnt, s->offset, (size_t)G * 8);                          /* the write positions */
        if (v.rows) FOR_SELECTED(&v, r) {
            int64_t w;
            uint8_t m;
            if (!element(&v, spec, policy, r, &w, &m)) continue;
            uint32_t g = tr_scope_group(&v, r);
            uint64_t e = cnt[g]++;
            for (int32_t j = 0; j < n; j++)
                s->value[e * (uint64_t)n + (uint64_t)j] = ((m >> j) & 1) ? 0 : tr_scope_value(&v, spec->cols[j], r);
            if (s->nulls) s->nulls[e] = m;
            s->W[e] = tr_i128_set(w);
            totals_add(&s->totals[g], w);
        }
        for (uint64_t g = 0; g < G; g++) s->totals[g].entries = s->totals[g].rows;
        free(cnt);
        table_extremes(s);
        set_error(error, TR_MATH_OK);
        return s;
    }
    /* SUPPORT: sort the elements' records by (group, tuple), then reduce equal tuples */
    const uint64_t width = 3 + (uint64_t)n;
    int64_t *rec = (int64_t *)zalloc(N * width * 8);
    uint64_t *idx = (uint64_t *)zalloc(N * 8), *tmp = (uint64_t *)zalloc(N * 8);
    if (!rec || !idx || !tmp) {
        free(cnt); free(rec); free(idx); free(tmp); state_release(s); set_error(error, TR_MATH_MEMORY); return NULL;
    }
    uint64_t i = 0;
    if (v.rows) FOR_SELECTED(&v, r) {
        int64_t w;
        uint8_t m;
        if (!element(&v, spec, policy, r, &w, &m)) continue;
        int64_t *x = rec + i * width;
        x[0] = (int64_t)tr_scope_group(&v, r); x[1] = m; x[2] = w;
        for (int32_t j = 0; j < n; j++) x[3 + j] = ((m >> j) & 1) ? 0 : tr_scope_value(&v, spec->cols[j], r);
        idx[i] = i;
        i++;
    }
    rec_ctx ctx = {rec, width, n};
    index_sort(idx, tmp, N, &ctx, cmp_records);
    uint64_t D = 0, largest = 0, run = 0;                             /* entries, and of the largest group */
    for (uint64_t k = 0; k < N; k++) {
        if (k && !cmp_records(&ctx, idx[k - 1], idx[k])) continue;        /* the same entry */
        run = k && rec[idx[k - 1] * width] == rec[idx[k] * width] ? run + 1 : 1;
        D++;
        if (run > largest) largest = run;
    }
    need.need[TR_BUDGET_SUPPORT] = largest;
    need.need[TR_BUDGET_BYTES] = sat_add(need.need[TR_BUDGET_BYTES],
                                         sat_add(sizeof *s, tr_pairplan_table_bytes(form, policy, n, G, D)));
    rc = tr_budget_check(budget, &need, refusal);
    if (rc || !table_alloc(s, form, G, D)) {
        free(cnt); free(rec); free(idx); free(tmp); state_release(s); set_error(error, rc ? rc : TR_MATH_MEMORY); return NULL;
    }
    uint64_t e = 0;
    for (uint64_t k = 0; k < N; k++) {
        const int64_t *x = rec + idx[k] * width;
        uint64_t g = (uint64_t)x[0];
        if (!k || cmp_records(&ctx, idx[k - 1], idx[k])) {                /* a new entry */
            e = k ? e + 1 : 0;
            for (int32_t j = 0; j < n; j++) s->value[e * (uint64_t)n + (uint64_t)j] = x[3 + j];
            if (s->nulls) s->nulls[e] = (uint8_t)x[1];
            s->totals[g].entries++;
        }
        __int128 w = x[2];
        s->count[e]++;
        s->W[e] = tr_i128_set(tr_i128_get(s->W[e]) + w);
        tr_pair_add(&s->Q[e], tr_pair_u((unsigned __int128)(w * w)));
        totals_add(&s->totals[g], x[2]);
    }
    for (uint64_t g = 0; g < G; g++) s->offset[g + 1] = s->offset[g] + s->totals[g].entries;
    free(cnt); free(rec); free(idx); free(tmp);
    table_extremes(s);
    set_error(error, TR_MATH_OK);
    return s;
}

/* ------------------------------------------------------------------ retained states */
static tr_dist_spec prefix_spec(const tr_pairplan_spec *s) {
    tr_dist_spec d;
    memset(&d, 0, sizeof d);
    d.k = 1; d.col[0] = s->cols[0]; d.col[1] = d.col[2] = -1; d.weight = s->weight;
    d.repr = TR_DIST_SPARSE; d.domain = TR_DIST_DOMAIN_SCOPE; d.strategy = TR_DIST_STRATEGY_AUTO;
    return d;
}
tr_pairplan_state *tr_pairplan_retain(const tr_scope *scope, const tr_pairplan_spec *spec, int32_t strategy,
                                      int32_t repr, const tr_math_budget *budget, tr_math_refusal *refusal,
                                      int32_t *error) {
    tr_budget_none(refusal);
    prepared pr;
    int32_t rc = prepare(scope, spec, budget, refusal, 0, &pr);
    if (rc) { set_error(error, rc); return NULL; }
    tr_pairplan_state *s = NULL;
    if (strategy == TR_PAIRPLAN_S_SUPPORT || strategy == TR_PAIRPLAN_S_DIRECT) {
        if (repr) { prepared_free(&pr); set_error(error, TR_MATH_INVALID); return NULL; }
        s = table_build(scope, spec, strategy == TR_PAIRPLAN_S_SUPPORT ? TR_PAIRPLAN_FORM_SUPPORT : TR_PAIRPLAN_FORM_ROWS,
                        pr.policy, budget, refusal, error);
        if (s) s->strategy = strategy;
        prepared_free(&pr);
        return s;
    }
    if (strategy == TR_PAIRPLAN_S_PREFIX) {
        if (spec->ncols != 1 || pr.policy != TR_PAIRPLAN_LISTWISE || repr) {
            prepared_free(&pr); set_error(error, TR_MATH_INVALID); return NULL;
        }
        prepared_free(&pr);
        s = state_new(scope, spec, strategy, TR_PAIRPLAN_LISTWISE);
        if (!s) { set_error(error, TR_MATH_MEMORY); return NULL; }
        tr_dist_spec ds = prefix_spec(spec);
        s->mass = spec->weight >= 0 ? TR_DIST_MASS_WEIGHT : TR_DIST_MASS_COUNT;
        s->hist = tr_dist_hist_build(scope, &ds, budget, refusal, error);
        if (s->hist) s->prefix = tr_dist_prefix_build(s->hist, s->mass, 0, budget, refusal, error);
        if (!s->prefix) { state_release(s); return NULL; }
        tr_dist_hist_info hi;
        tr_dist_prefix_info pi;
        tr_dist_hist_get_info(s->hist, &hi);
        tr_dist_prefix_get_info(s->prefix, &pi);
        s->ngroups = pi.ngroups; s->nentries = pi.nsupport; s->largest = pi.max_support; s->elements = pi.rows - pi.nulls;
        s->bytes = sat_add(sizeof *s, sat_add(hi.bytes, pi.bytes));
        s->magic = STATE_MAGIC;
        set_error(error, TR_MATH_OK);
        return s;
    }
    if (strategy == TR_PAIRPLAN_S_FEATURES) {
        if (spec->kind != TR_PAIRPLAN_K_FEATURE || (repr != TR_TENSOR_DENSE && repr != TR_TENSOR_SPARSE)) {
            prepared_free(&pr); set_error(error, TR_MATH_INVALID); return NULL;
        }
        s = state_new(scope, spec, strategy, pr.policy);
        if (!s) { prepared_free(&pr); set_error(error, TR_MATH_MEMORY); return NULL; }
        s->kernel = pr.kernel; s->repr = repr;
        prepared_free(&pr);
        s->features = tr_kernel_build(scope, &s->kernel, spec->weight, 0, repr, budget, refusal, error);
        if (s->features) s->eval = tr_kernel_eval_make(s->features, NULL, budget, refusal, error);
        if (!s->eval) { state_release(s); return NULL; }
        tr_kernel_info ki;
        tr_kernel_view kv;
        tr_kernel_get_info(s->features, &ki);
        tr_kernel_eval_get_view(s->eval, &kv);
        s->ngroups = ki.ngroups; s->nentries = ki.nnz; s->largest = kv.nitems ? kv.nitems : kv.cells;
        s->elements = ki.used;
        s->bytes = sat_add(sizeof *s, sat_add(ki.bytes, sat_mul(kv.nitems, TR_KERNEL_ITEM_BYTES)));
        s->magic = STATE_MAGIC;
        set_error(error, TR_MATH_OK);
        return s;
    }
    prepared_free(&pr);
    set_error(error, TR_MATH_INVALID);
    return NULL;
}
int32_t tr_pairplan_state_get_info(const tr_pairplan_state *s, tr_pairplan_state_info *out) {
    if (!state_ok(s) || !out) return TR_MATH_INVALID;
    memset(out, 0, sizeof *out);
    out->strategy = s->strategy; out->form = s->form; out->policy = s->policy; out->ncols = s->ncols;
    out->weight = s->weight; out->repr = s->repr; out->mass = s->mass; out->kind = s->kind;
    for (int32_t j = 0; j < s->ncols; j++) out->cols[j] = s->cols[j];
    out->ngroups = s->ngroups; out->entries = s->nentries; out->largest = s->largest; out->elements = s->elements;
    out->bytes = s->bytes;
    return TR_MATH_OK;
}
static void table_view_of(const tr_pairplan_state *s, tr_pairplan_table_view *v) {
    memset(v, 0, sizeof *v);
    v->form = s->form; v->policy = s->policy; v->ncols = s->ncols; v->ngroups = s->ngroups; v->nentries = s->nentries;
    v->offset = s->offset; v->value = s->value; v->nulls = s->nulls; v->W = s->W; v->Q = s->Q; v->count = s->count;
    v->totals = s->totals;
    for (int j = 0; j < TR_PAIRPLAN_MAX_COLUMNS; j++) { v->lo[j] = s->lo[j]; v->hi[j] = s->hi[j]; }
}
int32_t tr_pairplan_state_table_view(const tr_pairplan_state *s, tr_pairplan_table_view *out) {
    if (!state_ok(s) || !out || !s->form) return TR_MATH_INVALID;
    table_view_of(s, out);
    return TR_MATH_OK;
}

/* ------------------------------------------------------------------ the CPU evaluation of a table */
int32_t tr_pairplan_eval_table(const tr_pairplan_table_view *t, const tr_pairplan_kernel_view *k,
                               const tr_pairplan_request *req, uint64_t n, tr_pairplan_result *out,
                               const tr_math_budget *budget, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!t || !k || (n && (!req || !out)) || !t->offset || !t->totals || t->policy != k->policy ||
        t->ncols != k->ncols || (t->nentries && (!t->value || !t->W))) return TR_MATH_INVALID;
    tr_math_need need;
    need_zero(&need);
    need.need[TR_BUDGET_OUTPUT] = n;
    for (uint64_t i = 0; i < n; i++)
        if (!tr_pairplan_check(t, k, &req[i]))
            need.need[TR_BUDGET_ITERATIONS] = sat_add(need.need[TR_BUDGET_ITERATIONS], tr_pairplan_visits(
                tr_pairplan_group_entries(t, req[i].a), tr_pairplan_group_entries(t, req[i].b), req[i].a == req[i].b,
                (req[i].flags & TR_PAIRPLAN_ORDERED) != 0));
    tr_math_budget dflt;
    int32_t rc = tr_budget_check(budget_or(budget, &dflt), &need, refusal);
    if (rc) return rc;
    for (uint64_t i = 0; i < n; i++) {
        const tr_pairplan_request *q = &req[i];
        int32_t status = tr_pairplan_check(t, k, q);
        tr_kernel_wide sum, local;
        tr_kernel_w_zero(&sum);
        uint64_t visits = 0;
        int overflow = 0;
        if (!status) {
            const uint64_t tiles = tr_pairplan_request_tiles(t, q);
            for (uint64_t tile = 0; tile < tiles && !overflow; tile++) {
                uint64_t v = 0;
                if (!tr_pairplan_tile(t, k, q, tile, &local, &v) || !tr_kernel_w_add(&sum, &local, &sum)) overflow = 1;
                visits += v;
            }
        }
        tr_pairplan_finish(t, k, q, status, &sum, overflow, visits, &out[i]);
    }
    return TR_MATH_OK;
}

/* ------------------------------------------------------------------ planning */
static int same_source(const tr_pairplan_state *s, const tr_scope *scope, const tr_pairplan_spec *spec) {
    if (!state_ok(s) || s->scope != scope || s->ncols != spec->ncols || s->weight != spec->weight) return 0;
    for (int32_t j = 0; j < spec->ncols; j++) if (s->cols[j] != spec->cols[j]) return 0;
    return 1;
}
/* the per-request statuses against the plan's profile (0, -1 invalid, -3 unproven) */
static void plan_view(const tr_pairplan_plan *p, tr_pairplan_table_view *v) {
    memset(v, 0, sizeof *v);
    v->policy = p->pr.policy; v->ncols = p->spec.ncols; v->ngroups = p->G; v->totals = p->tot;
    for (int j = 0; j < TR_PAIRPLAN_MAX_COLUMNS; j++) { v->lo[j] = p->lo[j]; v->hi[j] = p->hi[j]; }
}
static uint64_t ceil_log2p1(uint64_t x) { return bits64(x); }   /* ceil(log2(x + 1)) */
/* the cells of the elements' value box (CATEGORY: a NULL slot per coordinate holding a NULL), saturating */
static uint64_t box_cells(const tr_pairplan_plan *p) {
    uint64_t c = 1;
    for (int32_t j = 0; j < p->spec.ncols; j++) {
        uint64_t span = p->lo[j] <= p->hi[j] ? (uint64_t)p->hi[j] - (uint64_t)p->lo[j] + 1 : 0;
        if (p->lo[j] <= p->hi[j] && span == 0) span = TR_BUDGET_UNKNOWN;
        span = sat_add(span, (uint64_t)(p->pr.policy == TR_PAIRPLAN_CATEGORY && p->nullseen[j]));
        c = sat_mul(c, span);
    }
    return c;
}
static void cand_init(tr_pairplan_candidate *c, int32_t strategy, int32_t device, int32_t repr, int32_t policy) {
    memset(c, 0, sizeof *c);
    c->strategy = strategy; c->device = device; c->repr = repr; c->reuse = -1; c->policy = policy;
    c->reg.module = 0; c->reg.representation = 0; c->reg.device = device;
    c->reg.finalize = device == TR_REG_ON_CUDA ? TR_REG_FINAL_GPU : TR_REG_FINAL_CPU;
    c->reg.batch = 1; c->reg.valid = 1;
    tr_reg_refusal_none(&c->reg.refusal);
    tr_budget_none(&c->reg.budget);
}
static void cand_reject(tr_pairplan_candidate *c, int32_t kind, int32_t code, int32_t op, int32_t carrier) {
    c->reg.valid = 0; c->reg.code = code;
    tr_reg_refuse(&c->reg.refusal, kind, code, -1, op, carrier, 0, 0, 0);
}
static void fixed_amounts(const tr_pairplan_plan *p, tr_pairplan_candidate *c, int built, uint64_t read_cols,
                          uint64_t construct, uint64_t chain, uint64_t calls) {
    uint64_t *a = c->reg.amount;
    if (built) {
        a[TR_COST_WORDS] = (p->rows + 63) / 64;
        a[TR_COST_SCAN] = sat_mul(p->selected, read_cols);
        a[TR_COST_CONSTRUCT] = construct;
    } else a[TR_COST_REUSE] = 1;
    a[TR_COST_CHAIN] = chain;
    a[TR_COST_MATERIALIZE] = sat_mul(p->n, TR_PAIRPLAN_RESULT_BYTES);
    a[c->device == TR_REG_ON_CUDA ? TR_COST_GPU_FINAL : TR_COST_CPU_FINAL] = sat_mul(p->n, 16);
    a[TR_COST_FIXED] = calls;
}
/* SUPPORT or DIRECT (a table strategy) on a device */
static void cand_table(tr_pairplan_plan *p, int32_t strategy, int32_t device, tr_pairplan_candidate *c) {
    const int support = strategy == TR_PAIRPLAN_S_SUPPORT;
    const int32_t form = support ? TR_PAIRPLAN_FORM_SUPPORT : TR_PAIRPLAN_FORM_ROWS, ncols = p->spec.ncols;
    cand_init(c, strategy, device, 0, p->pr.policy);
    c->form = form;
    const tr_pairplan_state *st = p->use[strategy][0];
    const int built = st == NULL;
    c->reuse = built ? -1 : p->use_index[strategy][0];
    c->exact = !support || !built;
    uint64_t entries = 0, largest = 0, evals = 0, tiles = 0, bound = 0, chain = 0;
    for (uint64_t g = 0; g < p->G; g++) {
        uint64_t d = support ? p->dsup[g] : p->tot[g].rows;
        entries = sat_add(entries, d); largest = umax(largest, d);
    }
    tr_pairplan_table_view v;
    plan_view(p, &v);
    for (uint64_t i = 0; i < p->n; i++) {
        const tr_pairplan_request *q = &p->req[i];
        if (p->status[i] == -1) continue;
        uint64_t da = support ? p->dsup[q->a] : p->tot[q->a].rows, db = support ? p->dsup[q->b] : p->tot[q->b].rows;
        uint64_t bits = tr_pairplan_request_bits(&v, &p->pr.kv, q);
        bound = umax(bound, bits);
        const int within = q->a == q->b, ordered = (q->flags & TR_PAIRPLAN_ORDERED) != 0;
        uint64_t visits = tr_pairplan_visits(da, db, within, ordered);
        evals = sat_add(evals, visits);
        /* ops(k) + 3 per visit, 8 times that for a request summed in 1024-bit words */
        chain = sat_add(chain, sat_mul(visits, sat_mul(p->pr.ops + 3, bits <= TR_PAIRPLAN_NARROW_BITS ? 1 : 8)));
        uint64_t ta = tr_pairplan_ceil_tiles(da), tb = tr_pairplan_ceil_tiles(db);
        tiles = sat_add(tiles, within ? (ordered ? sat_mul(ta, ta) : sat_mul(ta, ta + 1) / 2) : sat_mul(ta, tb));
    }
    c->bound_bits = (int32_t)umin(bound, INT32_MAX);
    c->evaluations = evals; c->native_requests = p->n; c->entries = entries; c->largest = largest;
    uint64_t N = p->elements, G = p->G;
    uint64_t state = sat_add(sizeof(tr_pairplan_state), tr_pairplan_table_bytes(form, p->pr.policy, ncols, G, entries));
    c->state_bytes = built ? state : st->bytes;
    uint64_t host = 0;
    if (built) host = support ? sat_add(support_scratch_bytes(G, N, ncols), state) : rows_build_bytes(p->pr.policy, ncols, G, N);
    uint64_t construct = support ? sat_add(sat_add(N, sat_mul(N, ceil_log2p1(N))), sat_add(entries, G)) : sat_add(N, G);
    fixed_amounts(p, c, built, (uint64_t)ncols + (p->spec.weight >= 0), construct, chain, (uint64_t)(1 + built));
    tr_math_need *need = &c->reg.need;
    need->need[TR_BUDGET_VARIABLES] = (uint64_t)ncols + (p->spec.weight >= 0);
    need->need[TR_BUDGET_DEGREE] = (uint64_t)p->pr.kdeg;
    need->need[TR_BUDGET_SUPPORT] = umax(largest, p->pr.kv.ntable);
    need->need[TR_BUDGET_STATES] = built ? (support ? umax(N, entries) : N) : entries;
    need->need[TR_BUDGET_ITERATIONS] = evals;
    need->need[TR_BUDGET_OUTPUT] = p->n;
    need->need[TR_BUDGET_BYTES] = host;
    if (device == TR_REG_ON_CUDA) {
        uint64_t up = tr_pairplan_upload_bytes(form, p->pr.policy, ncols, G, entries);
        uint64_t batch = tr_pairplan_batch_device_bytes(p->n, ncols, p->pr.kv.ntable);
        c->device_bytes = sat_add(up, batch); c->work_items = tiles;
        need->need[TR_BUDGET_BYTES] = sat_add(host, c->device_bytes);
        c->reg.amount[TR_COST_TRANSFER] = sat_add(tr_pairplan_table_bytes(form, p->pr.policy, ncols, G, entries), batch);
        c->reg.amount[TR_COST_FIXED] = (uint64_t)(3 + built);                /* upload, the tile and finish launches */
    }
    if (bound > TR_PAIRPLAN_BITS_LIMIT)
        cand_reject(c, TR_REG_REFUSE_BOUND, TR_MATH_OVERFLOW, TR_OP_MULTIPLY, TR_REG_C_BIGINT);
}
/* PAIR-001 requests issued for request q (ABS: one; MANN_WHITNEY between, ordered: two) */
static uint64_t prefix_issued(const tr_pairplan_plan *p, const tr_pairplan_request *q) {
    return p->spec.kind == TR_PAIRPLAN_K_MANN_WHITNEY && q->a != q->b && (q->flags & TR_PAIRPLAN_ORDERED) ? 2 : 1;
}
static void cand_prefix(tr_pairplan_plan *p, tr_pairplan_candidate *c) {
    cand_init(c, TR_PAIRPLAN_S_PREFIX, TR_REG_ON_CPU, 0, p->pr.policy);
    const tr_pairplan_spec *s = &p->spec;
    const tr_pairplan_state *st = p->use[TR_PAIRPLAN_S_PREFIX][0];
    const int built = st == NULL;
    c->reuse = built ? -1 : p->use_index[TR_PAIRPLAN_S_PREFIX][0];
    c->exact = !built;
    tr_scope_view v;
    tr_scope_get_view(p->scope, &v);
    if ((s->kind != TR_PAIRPLAN_K_ABS && s->kind != TR_PAIRPLAN_K_MANN_WHITNEY) || s->ncols != 1 ||
        v.rows > TR_DIST_MAX_ROWS) {
        cand_reject(c, TR_REG_REFUSE_UNSUPPORTED, TR_MATH_UNSUPPORTED, TR_OP_ORDER, TR_REG_C_INTEGER);
        return;
    }
    if (s->weight >= 0 && v.col[s->weight].nvalid && v.col[s->weight].min < 0) {   /* weight masses are >= 0 */
        cand_reject(c, TR_REG_REFUSE_UNSUPPORTED, TR_MATH_DOMAIN_ERROR, TR_OP_ORDER, TR_REG_C_COUNT);
        return;
    }
    uint64_t issued = 0, evals = 0, entries = 0, largest = 0;
    int heavy = 0;
    for (uint64_t g = 0; g < p->G; g++) { entries = sat_add(entries, p->dpre[g]); largest = umax(largest, p->dpre[g]); }
    for (uint64_t i = 0; i < p->n; i++) {
        const tr_pairplan_request *q = &p->req[i];
        if (p->status[i] == -1) continue;
        for (int side = 0; side < 2; side++)                              /* every class mass below 2^64 */
            if ((unsigned __int128)tr_i128_get(p->tot[side ? q->b : q->a].W) >> 64) heavy = 1;
        uint64_t m = prefix_issued(p, q);
        issued += m;
        evals = sat_add(evals, sat_mul(m, q->a == q->b ? p->dpre[q->a] : sat_add(p->dpre[q->a], p->dpre[q->b])));
    }
    c->evaluations = evals; c->native_requests = issued; c->entries = entries; c->largest = largest;
    /* the DIST-001 sparse build (the counting scratch when it fits its rule, else the sort) and the prefix state */
    tr_dist_spec ds = prefix_spec(s);
    tr_math_need hn;
    need_zero(&hn);
    int32_t rc = built ? tr_dist_hist_need(p->scope, &ds, TR_DIST_NEED_CPU, &hn, NULL) : TR_MATH_OK;
    if (rc) { cand_reject(c, TR_REG_REFUSE_INVALID, rc, TR_OP_HISTOGRAM, TR_REG_C_CODE); return; }
    const tr_scope_view_column *xc = &v.col[s->cols[0]];
    uint64_t span = xc->nvalid ? (uint64_t)xc->max - (uint64_t)xc->min + 1 : 0, cells = sat_add(span, xc->valid != NULL);
    if (xc->nvalid && !span) cells = TR_BUDGET_UNKNOWN;
    uint64_t total = sat_mul(p->G, cells), sel = v.selected, construct;
    if (cells != TR_BUDGET_UNKNOWN && total <= sat_add(sat_mul(sel, TR_DIST_COUNT_FACTOR), TR_DIST_COUNT_SLACK))
        construct = sat_add(total, sel);
    else construct = sat_mul(sel, 1 + ceil_log2p1(sel));
    construct = sat_add(construct, sat_add(entries, p->G));
    /* bounds of the final histogram (one cell per selected row or per scratch cell) and of the prefix state */
    uint64_t hcells = umin(sel, total), weighted = s->weight >= 0;
    uint64_t hist_bytes = sat_add(sat_add(512, sat_mul(p->G, 16)), sat_mul(hcells, 8 + 8 + 1 + (weighted ? 24 : 0)));
    uint64_t pre_bytes = sat_add(sat_add(512, sat_mul(p->G, 24)), sat_mul(entries, 24));
    c->state_bytes = built ? sat_add(hist_bytes, pre_bytes) : st->bytes;
    fixed_amounts(p, c, built, 1 + weighted, construct, sat_mul(evals, 4), (uint64_t)(1 + 2 * built));
    tr_math_need *need = &c->reg.need;
    *need = hn;
    need->need[TR_BUDGET_VARIABLES] = umax(need->need[TR_BUDGET_VARIABLES], 1 + weighted);
    need->need[TR_BUDGET_SUPPORT] = umax(need->need[TR_BUDGET_SUPPORT], built ? sat_add(largest, 1) : largest);   /* + NULL */
    need->need[TR_BUDGET_STATES] = umax(need->need[TR_BUDGET_STATES], entries);
    need->need[TR_BUDGET_ITERATIONS] = evals;
    need->need[TR_BUDGET_OUTPUT] = umax(p->n, issued);
    need->need[TR_BUDGET_BYTES] = sat_add(built ? sat_add(hn.need[TR_BUDGET_BYTES], c->state_bytes) : 0,
                                          sat_mul(issued, sizeof(tr_pair_request) + sizeof(tr_pair_result)));
    if (heavy) cand_reject(c, TR_REG_REFUSE_BOUND, TR_MATH_OVERFLOW, TR_OP_ORDER, TR_REG_C_COUNT);
}
static uint64_t features_items(const tr_pairplan_plan *p, const tr_scope_view *v) {
    const tr_kernel_spec *k = &p->pr.kernel;
    if (k->kind == TR_KERNEL_POLY) return tr_budget_binomial((uint64_t)k->ncols + (uint64_t)k->degree, (uint64_t)k->degree);
    if (k->kind == TR_KERNEL_MONOMIAL) return (uint64_t)k->nterms;
    uint64_t cells = 1;                                                   /* DELTA: the cells of the count tensor */
    for (int32_t j = 0; j < k->ncols; j++) {
        const tr_scope_view_column *c = &v->col[k->cols[j]];
        uint64_t span = c->nvalid ? (uint64_t)c->max - (uint64_t)c->min + 1 : 0;
        cells = sat_mul(cells, sat_add(span, c->valid != NULL));
    }
    return cells;
}
static void cand_features(tr_pairplan_plan *p, int32_t repr, tr_pairplan_candidate *c) {
    cand_init(c, TR_PAIRPLAN_S_FEATURES, TR_REG_ON_CPU, repr, p->pr.policy);
    const tr_pairplan_spec *s = &p->spec;
    const int ri = repr == TR_TENSOR_SPARSE;
    const tr_pairplan_state *st = p->use[TR_PAIRPLAN_S_FEATURES][ri];
    const int built = st == NULL;
    c->reuse = built ? -1 : p->use_index[TR_PAIRPLAN_S_FEATURES][ri];
    c->exact = 1;
    if (s->kind != TR_PAIRPLAN_K_FEATURE) {                               /* no exact finite feature map retained */
        cand_reject(c, TR_REG_REFUSE_UNSUPPORTED, TR_MATH_UNSUPPORTED, TR_OP_CONTRACT, TR_REG_C_BIGINT);
        return;
    }
    tr_scope_view v;
    tr_scope_get_view(p->scope, &v);
    const tr_kernel_spec *k = &p->pr.kernel;
    uint64_t items = features_items(p, &v), issued = 0;
    for (uint64_t i = 0; i < p->n; i++)
        if (p->status[i] != -1) issued += p->req[i].a != p->req[i].b && (p->req[i].flags & TR_PAIRPLAN_ORDERED) ? 2 : 1;
    c->evaluations = sat_mul(issued, items); c->native_requests = issued; c->largest = items;
    tr_math_need kn;
    need_zero(&kn);
    uint64_t cap = 0;
    int32_t D = k->kind == TR_KERNEL_DELTA ? 0 : (k->degree ? k->degree : 1);
    if (built) {
        /* KERNEL-001's own proofs (the self-pair bits, the tensor bound) and its exact needs */
        if (tr_kernel_self_bits(&v, k, s->weight) > TR_KERNEL_BITS_LIMIT) {
            cand_reject(c, TR_REG_REFUSE_BOUND, TR_MATH_OVERFLOW, TR_OP_CONTRACT, TR_REG_C_BIGINT); return;
        }
        tr_tensor_spec ts;
        tr_tensor_build_plan bp;
        int32_t rc = tr_kernel_tensor_spec(k, D, s->weight, repr, &ts);
        if (!rc) rc = tr_tensor_plan_make(&v, &ts, &bp);
        if (!rc && tr_tensor_plan_bound(&v, &bp) > TR_TENSOR_I128_MAX) rc = TR_MATH_OVERFLOW;
        if (!rc) rc = tr_kernel_need(p->scope, k, s->weight, 0, repr, TR_TENSOR_NEED_CPU, &kn, &cap);
        if (rc) {
            cand_reject(c, rc == TR_MATH_OVERFLOW ? TR_REG_REFUSE_BOUND : TR_REG_REFUSE_UNSUPPORTED, rc, TR_OP_CONTRACT,
                        TR_REG_C_BIGINT);
            return;
        }
        uint64_t construct;
        if (repr == TR_TENSOR_DENSE) {
            uint64_t per_row = k->kind == TR_KERNEL_DELTA ? 1 : tr_budget_binomial((uint64_t)k->ncols + (uint64_t)D, (uint64_t)D);
            construct = sat_add(sat_mul(p->G, bp.cells), sat_mul(p->elements, per_row));
        } else construct = sat_mul(cap, 1 + ceil_log2p1(cap));
        c->entries = repr == TR_TENSOR_DENSE ? sat_mul(p->G, bp.cells) : cap;
        c->state_bytes = sat_add(kn.need[TR_BUDGET_BYTES], sat_mul(items, TR_KERNEL_ITEM_BYTES));
        fixed_amounts(p, c, 1, (uint64_t)s->ncols + (s->weight >= 0), sat_add(construct, p->G), sat_mul(c->evaluations, 2), 3);
    } else {
        c->entries = st->nentries; c->state_bytes = st->bytes;
        fixed_amounts(p, c, 0, 0, 0, sat_mul(c->evaluations, 2), 1);
    }
    tr_math_need *need = &c->reg.need;
    *need = kn;
    need->need[TR_BUDGET_DEGREE] = umax(need->need[TR_BUDGET_DEGREE], (uint64_t)k->degree);
    need->need[TR_BUDGET_SUPPORT] = umax(need->need[TR_BUDGET_SUPPORT], k->kind == TR_KERNEL_DELTA ? 0 : items);
    need->need[TR_BUDGET_ITERATIONS] = c->evaluations;
    need->need[TR_BUDGET_OUTPUT] = umax(p->n, issued);
    need->need[TR_BUDGET_BYTES] = sat_add(built ? c->state_bytes : 0,
                                          sat_mul(issued, sizeof(tr_kernel_request) + sizeof(tr_kernel_result)));
}

void tr_pairplan_plan_free(tr_pairplan_plan *p) {
    if (!plan_ok(p)) return;
    prepared_free(&p->pr);
    free(p->req); free(p->status); free(p->tot); free(p->dsup); free(p->dpre);
    p->magic = 0;
    free(p);
}
static int force_match(const tr_pairplan_options *o, const tr_pairplan_candidate *c) {
    return o->force_strategy && c->strategy == o->force_strategy && (!o->force_device || c->device == o->force_device) &&
           (!o->force_repr || c->repr == o->force_repr);
}
int32_t tr_pairplan_make(const tr_scope *scope, const tr_pairplan_spec *spec, const tr_pairplan_request *requests,
                         uint64_t n, const tr_pairplan_state *const *retained, int32_t nretained,
                         const tr_pairplan_options *options, const tr_math_budget *budget, tr_pairplan_plan **out,
                         tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!out) return TR_MATH_INVALID;
    *out = NULL;
    if ((n && !requests) || nretained < 0 || (nretained && !retained)) return TR_MATH_INVALID;
    tr_pairplan_options o;
    if (options) o = *options;
    else memset(&o, 0, sizeof o);
    if (!o.devices) o.devices = TR_REG_ON_CPU;
    if (!o.strategies) o.strategies = (1 << (TR_PAIRPLAN_STRATEGIES + 1)) - 2;
    if ((o.devices & ~TR_REG_ON_BOTH) || (o.strategies & ~((1 << (TR_PAIRPLAN_STRATEGIES + 1)) - 2)) || o.reserved[0] ||
        o.reserved[1] || o.force_strategy < 0 || o.force_strategy > TR_PAIRPLAN_STRATEGIES ||
        (o.force_device && o.force_device != TR_REG_ON_CPU && o.force_device != TR_REG_ON_CUDA) ||
        (o.force_repr && o.force_repr != TR_TENSOR_DENSE && o.force_repr != TR_TENSOR_SPARSE) ||
        (o.custom_rates != 0 && o.custom_rates != 1) || (!o.force_strategy && (o.force_device || o.force_repr)))
        return TR_MATH_INVALID;
    if (o.force_device == TR_REG_ON_CUDA) o.devices |= TR_REG_ON_CUDA;   /* a force naming CUDA adds its candidates */
    if (o.force_strategy) o.strategies |= 1 << o.force_strategy;
    double rates[2][TR_COST_COMPONENTS];
    if (o.custom_rates) memcpy(rates, o.rates, sizeof rates);
    else tr_pairplan_default_rates(rates);
    for (int d = 0; d < 2; d++)
        for (int k = 0; k < TR_COST_COMPONENTS; k++)
            if (!(rates[d][k] >= 0) || rates[d][k] > 1e300) return TR_MATH_INVALID;
    tr_math_budget dflt;
    budget = budget_or(budget, &dflt);
    prepared pr;
    int32_t rc = prepare(scope, spec, budget, refusal, 1, &pr);
    if (rc) return rc;
    tr_scope_view v;
    tr_scope_get_view(scope, &v);
    const uint64_t G = v.ngroups;
    tr_math_need need;
    need_zero(&need);
    need.need[TR_BUDGET_OUTPUT] = n;
    uint64_t bytes = sat_add(sizeof(tr_pairplan_plan), sat_mul(n, sizeof(tr_pairplan_request) + 4));
    bytes = sat_add(bytes, sat_mul(G, sizeof(tr_pairplan_totals) + 16));
    need.need[TR_BUDGET_BYTES] = sat_add(bytes, pr.table_bytes);
    rc = tr_budget_check(budget, &need, refusal);
    if (rc) { prepared_free(&pr); return rc; }
    tr_pairplan_plan *p = (tr_pairplan_plan *)zalloc(sizeof *p);
    if (!p) { prepared_free(&pr); return TR_MATH_MEMORY; }
    p->magic = PLAN_MAGIC;
    p->scope = scope; p->spec = *spec; p->pr = pr; p->n = n; p->G = G; p->rows = v.rows; p->selected = v.selected;
    p->spec.table = pr.table; p->pr.kv.table = pr.table;
    p->bytes = need.need[TR_BUDGET_BYTES]; p->nretained = nretained; p->devices = o.devices;
    p->chosen = -1;
    tr_budget_none(&p->refusal);
    p->req = (tr_pairplan_request *)zalloc(n * sizeof(tr_pairplan_request));
    p->status = (int32_t *)zalloc(n * 4);
    p->tot = (tr_pairplan_totals *)zalloc(G * sizeof(tr_pairplan_totals));
    p->dsup = (uint64_t *)zalloc(G * 8); p->dpre = (uint64_t *)zalloc(G * 8);
    if (!p->req || !p->status || !p->tot || !p->dsup || !p->dpre) { tr_pairplan_plan_free(p); return TR_MATH_MEMORY; }
    if (n) memcpy(p->req, requests, (size_t)n * sizeof *requests);
    for (int s = 0; s <= TR_PAIRPLAN_STRATEGIES; s++) { p->use_index[s][0] = p->use_index[s][1] = -1; }
    /* the retained states that match the source (the scope by identity, the columns, the weight, the policy) */
    for (int32_t i = 0; i < nretained; i++) {
        const tr_pairplan_state *st = retained[i];
        if (!st || !same_source(st, scope, spec)) continue;
        int32_t s = st->strategy, ri = st->repr == TR_TENSOR_SPARSE;
        int ok = (s == TR_PAIRPLAN_S_SUPPORT || s == TR_PAIRPLAN_S_DIRECT) ? st->policy == pr.policy
               : s == TR_PAIRPLAN_S_PREFIX ? 1
               : s == TR_PAIRPLAN_S_FEATURES ? spec->kind == TR_PAIRPLAN_K_FEATURE && !memcmp(&st->kernel, &pr.kernel, sizeof pr.kernel)
               : 0;
        if (ok && !p->use[s][ri]) { p->use[s][ri] = st; p->use_index[s][ri] = i; }
    }
    /* the profile: a retained table's totals (exact), or one pass over the scope */
    const tr_pairplan_state *tab = p->use[TR_PAIRPLAN_S_SUPPORT][0] ? p->use[TR_PAIRPLAN_S_SUPPORT][0]
                                                                     : p->use[TR_PAIRPLAN_S_DIRECT][0];
    if (tab) {
        p->profile_reused = 1;
        memcpy(p->tot, tab->totals, (size_t)G * sizeof(tr_pairplan_totals));
        for (int j = 0; j < TR_PAIRPLAN_MAX_COLUMNS; j++) { p->lo[j] = tab->lo[j]; p->hi[j] = tab->hi[j]; p->nullseen[j] = 0; }
        for (uint64_t e = 0; tab->nulls && e < tab->nentries; e++)
            for (int32_t j = 0; j < spec->ncols; j++) if ((tab->nulls[e] >> j) & 1) p->nullseen[j] = 1;
        for (uint64_t g = 0; g < G; g++) p->tot[g].entries = p->tot[g].rows;
    } else profile_scope(&v, spec, pr.policy, p->tot, p->lo, p->hi, p->nullseen);
    for (uint64_t g = 0; g < G; g++) p->elements = sat_add(p->elements, p->tot[g].rows);
    /* per group: support entries (exact from a retained SUPPORT table, else the bound min(n, box cells)) and prefix
     * entries (exact from a retained PREFIX state, else min(n, the value span)) */
    const tr_pairplan_state *sup = p->use[TR_PAIRPLAN_S_SUPPORT][0], *pre = p->use[TR_PAIRPLAN_S_PREFIX][0];
    uint64_t cells = box_cells(p), span0 = p->lo[0] <= p->hi[0] ? (uint64_t)p->hi[0] - (uint64_t)p->lo[0] + 1 : 0;
    if (p->lo[0] <= p->hi[0] && !span0) span0 = TR_BUDGET_UNKNOWN;
    tr_dist_prefix_view pv;
    if (pre) tr_dist_prefix_get_view(pre->prefix, &pv);
    for (uint64_t g = 0; g < G; g++) {
        p->dsup[g] = sup ? sup->offset[g + 1] - sup->offset[g] : umin(p->tot[g].rows, cells);
        p->dpre[g] = pre ? pv.offset[g + 1] - pv.offset[g] : umin(p->tot[g].rows, span0);
    }
    tr_pairplan_table_view pvw;
    plan_view(p, &pvw);
    for (uint64_t i = 0; i < n; i++) p->status[i] = tr_pairplan_check(&pvw, &p->pr.kv, &p->req[i]);
    /* the candidates, in a fixed order */
    int32_t k = 0;
    const int cpu = (o.devices & TR_REG_ON_CPU) != 0, cuda = (o.devices & TR_REG_ON_CUDA) != 0;
    if (o.strategies & (1 << TR_PAIRPLAN_S_SUPPORT)) {
        if (cpu) cand_table(p, TR_PAIRPLAN_S_SUPPORT, TR_REG_ON_CPU, &p->cand[k++]);
        if (cuda) cand_table(p, TR_PAIRPLAN_S_SUPPORT, TR_REG_ON_CUDA, &p->cand[k++]);
    }
    if ((o.strategies & (1 << TR_PAIRPLAN_S_FEATURES)) && cpu) {
        cand_features(p, TR_TENSOR_DENSE, &p->cand[k++]);
        cand_features(p, TR_TENSOR_SPARSE, &p->cand[k++]);
    }
    if ((o.strategies & (1 << TR_PAIRPLAN_S_PREFIX)) && cpu) cand_prefix(p, &p->cand[k++]);
    if (o.strategies & (1 << TR_PAIRPLAN_S_DIRECT)) {
        if (cpu) cand_table(p, TR_PAIRPLAN_S_DIRECT, TR_REG_ON_CPU, &p->cand[k++]);
        if (cuda) cand_table(p, TR_PAIRPLAN_S_DIRECT, TR_REG_ON_CUDA, &p->cand[k++]);
    }
    p->ncand = k;
    /* the FND-008 choice with the family's rates in the override table */
    double coef[TR_REG_MODULES][2][TR_COST_COMPONENTS];
    memset(coef, 0, sizeof coef);
    memcpy(coef[TR_PAIRPLAN_COST_ROW], rates, sizeof rates);
    tr_reg_candidate chunk[TR_REG_MAX_CANDIDATES];
    for (int32_t i = 0; i < k; i++) { chunk[i] = p->cand[i].reg; chunk[i].module = TR_PAIRPLAN_COST_ROW; }
    tr_planner_evidence *ev = (tr_planner_evidence *)zalloc(sizeof *ev);
    if (!ev) { tr_pairplan_plan_free(p); return TR_MATH_MEMORY; }
    int32_t code = tr_planner_choose(chunk, k, budget, &coef[0][0][0], NULL, ev);
    int32_t chosen = ev->chosen, reason = ev->reason;
    if (code == TR_MATH_BUDGET) p->refusal = ev->refusal;
    if (k == 0) { code = TR_MATH_UNSUPPORTED; reason = TR_PLANNER_R_NONE_ADMISSIBLE; }
    free(ev);
    for (int32_t i = 0; i < k; i++) { chunk[i].module = 0; p->cand[i].reg = chunk[i]; }
    if (o.force_strategy) {
        p->forced = 1;
        int32_t best = -1, first = -1;
        for (int32_t i = 0; i < k; i++) {
            if (!force_match(&o, &p->cand[i])) continue;
            if (first < 0) first = i;
            if (p->cand[i].reg.status != TR_PLANNER_S_REJECTED && (best < 0 || p->cand[i].reg.total < p->cand[best].reg.total))
                best = i;
        }
        tr_budget_none(&p->refusal);
        code = TR_MATH_OK;
        if (best >= 0) { chosen = best; reason = TR_PLANNER_R_FORCED; }
        else if (first < 0) { chosen = -1; code = TR_MATH_UNSUPPORTED; reason = TR_PLANNER_R_FORCE_UNMATCHED; }
        else {
            chosen = -1; reason = TR_PLANNER_R_FORCE_REFUSED;
            code = p->cand[first].reg.code ? p->cand[first].reg.code : TR_MATH_UNSUPPORTED;
            if (code == TR_MATH_BUDGET) p->refusal = p->cand[first].reg.budget;
        }
        for (int32_t i = 0; i < k; i++) {
            tr_reg_candidate *x = &p->cand[i].reg;
            if (x->status != TR_PLANNER_S_REJECTED) { x->status = TR_PLANNER_S_ADMISSIBLE; x->reason = TR_PLANNER_R_NOT_FORCED; }
        }
        if (chosen >= 0) { p->cand[chosen].reg.status = TR_PLANNER_S_CHOSEN; p->cand[chosen].reg.reason = TR_PLANNER_R_FORCED; }
    }
    p->chosen = chosen; p->code = code; p->reason = reason;
    if (code == TR_MATH_BUDGET && refusal) *refusal = p->refusal;
    *out = p;
    return code;
}
int32_t tr_pairplan_plan_get_info(const tr_pairplan_plan *p, tr_pairplan_plan_info *out) {
    if (!plan_ok(p) || !out) return TR_MATH_INVALID;
    memset(out, 0, sizeof *out);
    out->ncandidates = p->ncand; out->chosen = p->chosen; out->code = p->code; out->reason = p->reason;
    out->forced = p->forced; out->devices = p->devices; out->kind = p->spec.kind; out->policy = p->pr.policy;
    out->requests = p->n; out->groups = p->G; out->elements = p->elements; out->bytes = p->bytes;
    out->refusal = p->refusal; out->profile_reused = p->profile_reused; out->nretained = p->nretained;
    return TR_MATH_OK;
}
int32_t tr_pairplan_plan_get(const tr_pairplan_plan *p, int32_t i, tr_pairplan_candidate *out) {
    if (!plan_ok(p) || !out || i < 0 || i >= p->ncand) return TR_MATH_INVALID;
    *out = p->cand[i];
    return TR_MATH_OK;
}
int32_t tr_pairplan_plan_kernel_view(const tr_pairplan_plan *p, tr_pairplan_kernel_view *out) {
    if (!plan_ok(p) || !out) return TR_MATH_INVALID;
    *out = p->pr.kv;
    return TR_MATH_OK;
}
int32_t tr_pairplan_plan_requests(const tr_pairplan_plan *p, const tr_pairplan_request **req, uint64_t *n) {
    if (!plan_ok(p) || !req || !n) return TR_MATH_INVALID;
    *req = p->req; *n = p->n;
    return TR_MATH_OK;
}

/* ------------------------------------------------------------------ execution */
static void result_clear(tr_pairplan_result *r, int32_t status) {
    memset(r, 0, sizeof *r);
    r->status = status;
    tr_kernel_w_from_i128(1, &r->den);
}
/* S of a native sum (PAIR-001's 256-bit num, KERNEL-001's 1024-bit num) completed into the request's answer: P from the
 * profile, MEAN as tr_pairplan_finish */
static void complete(const tr_pairplan_plan *p, const tr_pairplan_request *q, const tr_kernel_wide *S, uint64_t evals,
                     tr_pairplan_result *r) {
    tr_pairplan_table_view v;
    plan_view(p, &v);
    tr_pairplan_finish(&v, &p->pr.kv, q, 0, S, 0, evals, r);
}
static int32_t run_prefix(const tr_pairplan_plan *p, const tr_pairplan_state *st, const tr_math_budget *budget,
                          tr_pairplan_result *out, tr_pairplan_exec *ex, tr_math_refusal *refusal) {
    const int mw = p->spec.kind == TR_PAIRPLAN_K_MANN_WHITNEY;
    uint64_t m = 0;
    for (uint64_t i = 0; i < p->n; i++) if (p->status[i] != -1) m += prefix_issued(p, &p->req[i]);
    tr_pair_request *pq = (tr_pair_request *)zalloc(m * sizeof *pq);
    tr_pair_result *pr = (tr_pair_result *)zalloc(m * sizeof *pr);
    if (!pq || !pr) { free(pq); free(pr); return TR_MATH_MEMORY; }
    uint64_t j = 0;
    for (uint64_t i = 0; i < p->n; i++) {
        const tr_pairplan_request *q = &p->req[i];
        if (p->status[i] == -1) continue;
        tr_pair_request *x = &pq[j++];
        x->a = q->a; x->b = q->b;
        if (!mw) x->op = q->a == q->b ? TR_PAIR_OP_ABS_WITHIN : TR_PAIR_OP_ABS_CROSS;   /* unordered, no self-pair */
        else { x->op = TR_PAIR_OP_MANN_WHITNEY; x->flags = q->a == q->b ? TR_PAIR_SELF : 0; }
        if (prefix_issued(p, q) == 2) { tr_pair_request *y = &pq[j++]; *y = *x; y->a = q->b; y->b = q->a; }
    }
    int32_t rc = tr_pair_query(st->prefix, NULL, pq, m, pr, budget, refusal);
    if (rc) { free(pq); free(pr); return rc; }
    tr_dist_prefix_view pv;
    tr_dist_prefix_get_view(st->prefix, &pv);
    j = 0;
    for (uint64_t i = 0; i < p->n; i++) {
        const tr_pairplan_request *q = &p->req[i];
        if (p->status[i] == -1) { result_clear(&out[i], -1); continue; }
        const uint64_t issued = prefix_issued(p, q);
        const int within = q->a == q->b, ordered = (q->flags & TR_PAIRPLAN_ORDERED) != 0;
        const uint64_t da = pv.offset[q->a + 1] - pv.offset[q->a], db = pv.offset[q->b + 1] - pv.offset[q->b];
        int32_t status = 0;
        for (uint64_t t = 0; t < issued; t++) if (pr[j + t].status) status = pr[j + t].status;
        if (status) { result_clear(&out[i], status); j += issued; continue; }
        tr_kernel_wide S, t2;
        tr_pairplan_w_from_i256(pr[j].num, &S);                            /* < 2^255: nonnegative */
        if (issued == 2) { tr_pairplan_w_from_i256(pr[j + 1].num, &t2); tr_kernel_w_add(&S, &t2, &S); }
        if (!mw && ordered) { t2 = S; tr_kernel_w_add(&S, &t2, &S); }     /* ABS: both orientations, or both (i, j) */
        if (mw && within && !(q->flags & TR_PAIRPLAN_SELF)) {             /* 2U over every pair minus the diagonal Q */
            tr_pairplan_w_from_i256(p->tot[q->a].Q, &t2);
            tr_kernel_w_sub(&S, &t2, &S);
        }
        complete(p, q, &S, sat_mul(issued, within ? da : sat_add(da, db)), &out[i]);
        ex->evaluations = sat_add(ex->evaluations, out[i].evaluations);
        j += issued;
    }
    ex->native_requests = m;
    free(pq); free(pr);
    return TR_MATH_OK;
}
static int32_t run_features(const tr_pairplan_plan *p, const tr_pairplan_state *st, const tr_math_budget *budget,
                            tr_pairplan_result *out, tr_pairplan_exec *ex, tr_math_refusal *refusal) {
    uint64_t m = 0;
    for (uint64_t i = 0; i < p->n; i++)
        if (p->status[i] != -1) m += p->req[i].a != p->req[i].b && (p->req[i].flags & TR_PAIRPLAN_ORDERED) ? 2 : 1;
    tr_kernel_request *kq = (tr_kernel_request *)zalloc(m * sizeof *kq);
    tr_kernel_result *kr = (tr_kernel_result *)zalloc(m * sizeof *kr);
    if (!kq || !kr) { free(kq); free(kr); return TR_MATH_MEMORY; }
    uint64_t j = 0;
    for (uint64_t i = 0; i < p->n; i++) {
        const tr_pairplan_request *q = &p->req[i];
        if (p->status[i] == -1) continue;
        tr_kernel_request *x = &kq[j++];
        x->op = TR_KERNEL_OP_SUM; x->a = q->a; x->b = q->b;
        if (q->a == q->b) x->flags = ((q->flags & TR_PAIRPLAN_SELF) ? TR_KERNEL_SELF : 0) |
                                     ((q->flags & TR_PAIRPLAN_ORDERED) ? 0 : TR_KERNEL_UNORDERED);
        else if (q->flags & TR_PAIRPLAN_ORDERED) { tr_kernel_request *y = &kq[j++]; *y = *x; y->a = q->b; y->b = q->a; }
    }
    int32_t rc = tr_kernel_query(st->eval, kq, m, NULL, NULL, 0, kr, budget, refusal);
    if (rc) { free(kq); free(kr); return rc; }
    tr_kernel_view kv;
    tr_kernel_eval_get_view(st->eval, &kv);
    const uint64_t items = kv.nitems ? kv.nitems : kv.cells;
    j = 0;
    for (uint64_t i = 0; i < p->n; i++) {
        const tr_pairplan_request *q = &p->req[i];
        if (p->status[i] == -1) { result_clear(&out[i], -1); continue; }
        const uint64_t issued = q->a != q->b && (q->flags & TR_PAIRPLAN_ORDERED) ? 2 : 1;
        int32_t status = 0;
        for (uint64_t t = 0; t < issued; t++) if (kr[j + t].status) status = kr[j + t].status;
        tr_kernel_wide S = kr[j].num, P = kr[j].pairs;
        if (!status && issued == 2 && (!tr_kernel_w_add(&S, &kr[j + 1].num, &S) || !tr_kernel_w_add(&P, &kr[j + 1].pairs, &P)))
            status = -3;
        if (status) { result_clear(&out[i], status); j += issued; continue; }
        complete(p, q, &S, sat_mul(issued, items), &out[i]);
        if (memcmp(&out[i].pairs, &P, sizeof P)) result_clear(&out[i], -1);   /* KERNEL-001's P differs: never */
        ex->evaluations = sat_add(ex->evaluations, out[i].evaluations);
        j += issued;
    }
    ex->native_requests = m;
    free(kq); free(kr);
    return TR_MATH_OK;
}
int32_t tr_pairplan_execute(const tr_pairplan_plan *p, int32_t index, const tr_math_budget *budget,
                            tr_pairplan_result *out, tr_pairplan_exec *exec, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!plan_ok(p) || index < -1 || index >= p->ncand || (p->n && !out)) return TR_MATH_INVALID;
    if (index < 0) { if (p->chosen < 0) return p->code ? p->code : TR_MATH_INVALID; index = p->chosen; }
    const tr_pairplan_candidate *c = &p->cand[index];
    if (c->reg.status == TR_PLANNER_S_REJECTED) return c->reg.code ? c->reg.code : TR_MATH_UNSUPPORTED;
    if (c->device != TR_REG_ON_CPU) return TR_MATH_INVALID;              /* a CUDA candidate: tr_mgpu_pairplan_query */
    tr_pairplan_exec ex;
    memset(&ex, 0, sizeof ex);
    ex.candidate = index; ex.strategy = c->strategy; ex.device = c->device; ex.requests = p->n;
    int ri = c->repr == TR_TENSOR_SPARSE;
    const tr_pairplan_state *st = p->use[c->strategy][ri];
    tr_pairplan_state *own = NULL;
    int32_t rc = TR_MATH_OK, error = 0;
    if (!st) {                                                            /* a transient state for this call */
        own = tr_pairplan_retain(p->scope, &p->spec, c->strategy, c->repr, budget, refusal, &error);
        if (!own) return error ? error : TR_MATH_MEMORY;
        st = own; ex.built = 1;
    }
    ex.state_bytes = st->bytes;
    if (c->strategy == TR_PAIRPLAN_S_SUPPORT || c->strategy == TR_PAIRPLAN_S_DIRECT) {
        tr_pairplan_table_view v;
        table_view_of(st, &v);
        rc = tr_pairplan_eval_table(&v, &p->pr.kv, p->req, p->n, out, budget, refusal);
        ex.native_requests = p->n;
        for (uint64_t i = 0; !rc && i < p->n; i++) ex.evaluations = sat_add(ex.evaluations, out[i].evaluations);
    } else if (c->strategy == TR_PAIRPLAN_S_PREFIX) rc = run_prefix(p, st, budget, out, &ex, refusal);
    else rc = run_features(p, st, budget, out, &ex, refusal);
    tr_pairplan_state_free(own);
    if (exec) *exec = ex;
    return rc;
}
