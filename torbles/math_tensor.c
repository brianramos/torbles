/* TENSOR-001 / TENSOR-002: exact multi-index states and costed contraction planning; see math_tensor.h.
 * SPDX-License-Identifier: Apache-2.0 */
#include "math_tensor.h"
#include "mathematics.h"
#include "math_domain.h"
#include "math_handles.h"
#include "math_planner.h"
#include <stdlib.h>
#include <string.h>

TR_MATH_ABI(tr_tensor_spec, 1);          /* the ctypes boundary of TENSOR-001 / TENSOR-002 (math_handles.h) */
TR_MATH_ABI(tr_tensor_axis, 1);
TR_MATH_ABI(tr_tensor_info, 1);
TR_MATH_ABI(tr_tensor_states, 1);
TR_MATH_ABI(tr_tensor_net_spec, 1);
TR_MATH_ABI(tr_tensor_plan_options, 1);
TR_MATH_ABI(tr_tensor_step_info, 1);
TR_MATH_ABI(tr_tensor_candidate, 1);
TR_MATH_ABI(tr_tensor_plan_info, 1);
TR_MATH_ABI(tr_tensor_exec, 1);
TR_MATH_ABI(tr_tensor_dense_step, 1);
TR_MATH_ABI(tr_tensor_program, 1);
TR_MATH_ABI(tr_tensor_dense_view, 1);

_Static_assert(sizeof(tr_tensor_spec) == 736, "tr_tensor_spec ABI");
_Static_assert(sizeof(tr_tensor_axis) == 112, "tr_tensor_axis ABI");
_Static_assert(sizeof(tr_tensor_info) == 216, "tr_tensor_info ABI");
_Static_assert(sizeof(tr_tensor_states) == 192, "tr_tensor_states ABI");
_Static_assert(sizeof(tr_tensor_net_spec) == 392, "tr_tensor_net_spec ABI");
_Static_assert(sizeof(tr_tensor_plan_options) == 216, "tr_tensor_plan_options ABI");
_Static_assert(sizeof(tr_tensor_step_info) == 128, "tr_tensor_step_info ABI");
_Static_assert(sizeof(tr_tensor_candidate) == 1480, "tr_tensor_candidate ABI");
_Static_assert(sizeof(tr_tensor_plan_info) == 80, "tr_tensor_plan_info ABI");
_Static_assert(sizeof(tr_tensor_exec) == 240, "tr_tensor_exec ABI");
_Static_assert(sizeof(tr_tensor_dense_step) == 840, "tr_tensor_dense_step ABI");
_Static_assert(sizeof(tr_tensor_program) == 7064, "tr_tensor_program ABI");
_Static_assert(sizeof(tr_tensor_dense_view) == 88, "tr_tensor_dense_view ABI");

#define TENSOR_MAGIC UINT32_C(0x534e4554)
#define BUILDER_MAGIC UINT32_C(0x424e4554)
#define NET_MAGIC UINT32_C(0x4e4e4554)
#define PLAN_MAGIC UINT32_C(0x504e4554)
#define LIMIT TR_TENSOR_I128_MAX
typedef tr_tensor_u128 u128;

struct tr_tensor_s {
    uint32_t magic;
    int32_t kind, repr, order, policy, has_counts, weighted, scoped, strategy, symmetric, wcol, wscale;
    uint64_t ngroups, cells, nnz, nterms, fdim, bytes;
    uint64_t dims[TR_TENSOR_MAX_ORDER], stride[TR_TENSOR_MAX_ORDER];
    tr_tensor_axis axis[TR_TENSOR_MAX_ORDER];
    uint64_t digest[2];
    u128 bound;
    uint64_t *rows, *used;
    tr_math_i128 *value;          /* dense [G x cells], sparse [nnz] */
    uint64_t *count;              /* has_counts: like value */
    uint64_t *offset;             /* sparse [G + 1]; factored term offsets [G + 1] */
    uint32_t *coord;              /* sparse [nnz x order] */
    tr_math_i128 *weight;         /* factored [nterms] */
    int64_t *factor;              /* factored [nterms x fdim] */
};

static void set_error(int32_t *error, int32_t code) { if (error) *error = code; }
static void *zalloc(uint64_t bytes) {
    if (bytes > (uint64_t)SIZE_MAX / 2) return NULL;
    return calloc(1, bytes ? (size_t)bytes : 1);
}
static int tensor_ok(const tr_tensor *t) { return t && t->magic == TENSOR_MAGIC; }
static __int128 get128(tr_math_i128 v) { return tr_i128_get(v); }
static tr_math_i128 set128(__int128 v) { return tr_i128_set(v); }
static int add_ok(__int128 a, __int128 b, __int128 *r) { return !__builtin_add_overflow(a, b, r) && tr_tensor_mag(*r) <= LIMIT; }
static int mul_ok(__int128 a, __int128 b, __int128 *r) { return !__builtin_mul_overflow(a, b, r) && tr_tensor_mag(*r) <= LIMIT; }
static void need_zero(tr_math_need *n) { for (int i = 0; i < TR_BUDGET_RESOURCES; i++) n->need[i] = 0; }
static const tr_math_budget *budget_or(const tr_math_budget *b, tr_math_budget *dflt) {
    if (b) return b;
    tr_budget_default(dflt);
    return dflt;
}
uint64_t tr_tensor_header_bytes(void) { return sizeof(struct tr_tensor_s); }

void tr_tensor_default_rates(double rates[2][TR_COST_COMPONENTS]) {
    /* WORDS, SCAN, CHAIN, COMBINE, CONSTRUCT, REUSE, FUSION, MATERIALIZE, TRANSFER, CPU_FINAL, GPU_FINAL, FIXED:
     * hand-set relative estimates (about a nanosecond of the 2-core development machine per unit on the CPU; the
     * CUDA rates are uncalibrated estimates for a discrete device), not times */
    static const double cpu[TR_COST_COMPONENTS] = {0, 0.5, 2.0, 0, 1.0, 0, 0, 0.05, 0, 0, 0, 2000.0};
    static const double cuda[TR_COST_COMPONENTS] = {0, 0.01, 0.02, 0, 0.01, 0, 0, 0.05, 0.12, 0, 0, 25000.0};
    if (!rates) return;
    memcpy(rates[0], cpu, sizeof cpu);
    memcpy(rates[1], cuda, sizeof cuda);
}

/* ------------------------------------------------------------------ allocation */
static uint64_t data_bytes(int32_t repr, uint64_t G, uint64_t cells, uint64_t nnz, int32_t order, int has_counts,
                           uint64_t nterms, uint64_t fdim) {
    if (repr == TR_TENSOR_DENSE) return tr_tensor_dense_data_bytes(G, cells, has_counts);
    if (repr == TR_TENSOR_SPARSE) return tr_tensor_sparse_data_bytes(G, nnz, order, has_counts);
    return tr_tensor_factored_data_bytes(G, nterms, fdim);
}
static uint64_t object_bytes(int32_t repr, uint64_t G, uint64_t cells, uint64_t nnz, int32_t order, int has_counts,
                             uint64_t nterms, uint64_t fdim) {
    return tr_budget_add(sizeof(struct tr_tensor_s), data_bytes(repr, G, cells, nnz, order, has_counts, nterms, fdim));
}
static void tensor_release(tr_tensor *t) {
    if (!t) return;
    free(t->rows); free(t->used); free(t->value); free(t->count); free(t->offset); free(t->coord); free(t->weight);
    free(t->factor);
    t->magic = 0;
    free(t);
}
void tr_tensor_free(tr_tensor *t) { if (tensor_ok(t)) tensor_release(t); }
static void set_strides(tr_tensor *t) {
    uint64_t c = 1;
    for (int32_t k = t->order - 1; k >= 0; k--) { t->stride[k] = c; c = tr_budget_mul(c, t->dims[k]); }
    t->cells = tr_tensor_cells(t->dims, t->order);
    t->fdim = 0;
    for (int32_t k = 0; k < t->order; k++) t->fdim += t->dims[k];
}
/* a tensor of this shape with its arrays allocated (zeroed); axes INDEX unless set later */
static tr_tensor *tensor_alloc(int32_t repr, int32_t order, const uint64_t *dims, uint64_t G, uint64_t nnz,
                               uint64_t nterms, int has_counts) {
    tr_tensor *t = zalloc(sizeof *t);
    if (!t) return NULL;
    t->repr = repr; t->order = order; t->ngroups = G; t->has_counts = has_counts; t->wcol = -1;
    for (int32_t k = 0; k < order; k++) {
        t->dims[k] = dims[k];
        t->axis[k].kind = TR_TENSOR_AXIS_INDEX; t->axis[k].col = -1; t->axis[k].dim = dims[k];
    }
    set_strides(t);
    t->rows = zalloc(G * 8); t->used = zalloc(G * 8);
    int ok = t->rows && t->used;
    if (repr == TR_TENSOR_DENSE) {
        uint64_t e = G * t->cells;
        t->nnz = e;
        t->value = zalloc(e * 16);
        if (has_counts) t->count = zalloc(e * 8);
        ok = ok && t->value && (!has_counts || t->count);
    } else if (repr == TR_TENSOR_SPARSE) {
        t->nnz = nnz;
        t->offset = zalloc((G + 1) * 8);
        t->coord = zalloc(nnz * (uint64_t)order * 4);
        t->value = zalloc(nnz * 16);
        if (has_counts) t->count = zalloc(nnz * 8);
        ok = ok && t->offset && t->coord && t->value && (!has_counts || t->count);
    } else {
        t->nterms = nterms;
        t->offset = zalloc((G + 1) * 8);
        t->weight = zalloc(nterms * 16);
        t->factor = zalloc(nterms * t->fdim * 8);
        ok = ok && t->offset && t->weight && t->factor;
    }
    if (!ok) { tensor_release(t); return NULL; }
    t->bytes = object_bytes(repr, G, t->cells, nnz, order, has_counts, nterms, t->fdim);
    t->magic = TENSOR_MAGIC;
    return t;
}
/* metadata (kind, policy, axes, rows, used, scope) of src onto t (same order) */
static void copy_meta(tr_tensor *t, const tr_tensor *src) {
    t->kind = src->kind; t->policy = src->policy; t->weighted = src->weighted; t->scoped = src->scoped;
    t->symmetric = src->symmetric; t->wcol = src->wcol; t->wscale = src->wscale;
    t->digest[0] = src->digest[0]; t->digest[1] = src->digest[1];
    for (int32_t k = 0; k < t->order && k < src->order; k++) t->axis[k] = src->axis[k];
    memcpy(t->rows, src->rows, (size_t)src->ngroups * 8);
    memcpy(t->used, src->used, (size_t)src->ngroups * 8);
}
static u128 scan_bound(const tr_tensor *t) {
    u128 b = 0;
    uint64_t n = t->nnz;
    for (uint64_t i = 0; i < n; i++) { u128 m = tr_tensor_mag(get128(t->value[i])); if (m > b) b = m; }
    return b;
}
/* sum_t |w_t| prod_k max_i |f_{t,k}[i]| per group, the largest (saturated past LIMIT) */
static u128 factored_bound(const tr_tensor *t) {
    u128 best = 0;
    for (uint64_t g = 0; g < t->ngroups; g++) {
        u128 s = 0;
        for (uint64_t j = t->offset[g]; j < t->offset[g + 1]; j++) {
            u128 b = tr_tensor_mag(get128(t->weight[j]));
            const int64_t *f = t->factor + j * t->fdim;
            for (int32_t k = 0; k < t->order && b; k++) {
                uint64_t m = 0;
                for (uint64_t i = 0; i < t->dims[k]; i++) { uint64_t x = tr_abs_u64(f[i]); if (x > m) m = x; }
                b = tr_tensor_satmul(b, m, LIMIT);
                f += t->dims[k];
            }
            s = tr_tensor_satadd(s, b, LIMIT);
        }
        if (s > best) best = s;
    }
    return best;
}
static uint64_t linear(const tr_tensor *t, const uint32_t *c) {
    uint64_t idx = 0;
    for (int32_t k = 0; k < t->order; k++) idx += (uint64_t)c[k] * t->stride[k];
    return idx;
}
static void decode(const uint64_t *dims, const uint64_t *stride, int32_t order, uint64_t idx, uint32_t *c) {
    for (int32_t k = 0; k < order; k++) c[k] = (uint32_t)((idx / stride[k]) % dims[k]);
}
static int coord_cmp(const uint32_t *a, const uint32_t *b, int32_t m) {
    for (int32_t k = 0; k < m; k++) if (a[k] != b[k]) return a[k] < b[k] ? -1 : 1;
    return 0;
}
static uint64_t max_group(const uint64_t *offset, uint64_t G) {
    uint64_t m = 0;
    for (uint64_t g = 0; g < G; g++) if (offset[g + 1] - offset[g] > m) m = offset[g + 1] - offset[g];
    return m;
}

/* ------------------------------------------------------------------ canonical sparse states from records */
/* an emitted entry: (group, linear index, value, count); 48 bytes */
typedef struct { uint64_t g, idx; __int128 v; uint64_t c, pad; } rec_t;
#define REC_BYTES ((uint64_t)sizeof(rec_t))
static int rec_cmp(const void *x, const void *y) {
    const rec_t *a = x, *b = y;
    if (a->g != b->g) return a->g < b->g ? -1 : 1;
    if (a->idx != b->idx) return a->idx < b->idx ? -1 : 1;
    return 0;
}
/* Sorts n records, sums equal (group, index) pairs (checked), drops zeros (counts: count 0) and builds the sparse
 * tensor of this shape. budget: the final state (support = the largest group, bytes = the records + the state). */
static tr_tensor *records_to_sparse(rec_t *r, uint64_t n, int32_t order, const uint64_t *dims, uint64_t G,
                                    int has_counts, const tr_math_budget *budget, uint64_t extra_bytes,
                                    tr_math_refusal *refusal, int32_t *error) {
    tr_math_budget dflt;
    budget = budget_or(budget, &dflt);
    if (n > 1) qsort(r, (size_t)n, sizeof *r, rec_cmp);
    uint64_t out = 0;
    for (uint64_t i = 0; i < n;) {
        uint64_t j = i + 1;
        __int128 v = r[i].v;
        uint64_t c = r[i].c;
        for (; j < n && r[j].g == r[i].g && r[j].idx == r[i].idx; j++) {
            if (!add_ok(v, r[j].v, &v) || __builtin_add_overflow(c, r[j].c, &c)) { set_error(error, TR_MATH_OVERFLOW); return NULL; }
        }
        if (has_counts ? c != 0 : v != 0) { r[out] = r[i]; r[out].v = v; r[out].c = c; out++; }
        i = j;
    }
    uint64_t *per = zalloc((G + 1) * 8);
    if (!per) { set_error(error, TR_MATH_MEMORY); return NULL; }
    for (uint64_t i = 0; i < out; i++) per[r[i].g + 1]++;
    uint64_t largest = 0;
    for (uint64_t g = 0; g < G; g++) { if (per[g + 1] > largest) largest = per[g + 1]; per[g + 1] += per[g]; }
    tr_math_need need;
    need_zero(&need);
    need.need[TR_BUDGET_DEGREE] = (uint64_t)order;
    need.need[TR_BUDGET_SUPPORT] = largest;
    need.need[TR_BUDGET_BYTES] = tr_budget_add(extra_bytes, object_bytes(TR_TENSOR_SPARSE, G, 0, out, order, has_counts, 0, 0));
    int32_t rc = tr_budget_check(budget, &need, refusal);
    if (rc) { free(per); set_error(error, rc); return NULL; }
    tr_tensor *t = tensor_alloc(TR_TENSOR_SPARSE, order, dims, G, out, 0, has_counts);
    if (!t) { free(per); set_error(error, TR_MATH_MEMORY); return NULL; }
    memcpy(t->offset, per, (size_t)(G + 1) * 8);
    free(per);
    for (uint64_t i = 0; i < out; i++) {
        decode(t->dims, t->stride, order, r[i].idx, t->coord + i * (uint64_t)order);
        t->value[i] = set128(r[i].v);
        if (has_counts) t->count[i] = r[i].c;
    }
    t->bound = scan_bound(t);
    set_error(error, TR_MATH_OK);
    return t;
}
/* the sparse image of a dense state (entries with a nonzero value, or a nonzero count) */
static tr_tensor *dense_to_sparse(const tr_tensor *d, const tr_math_budget *budget, uint64_t extra_bytes,
                                  tr_math_refusal *refusal, int32_t *error) {
    tr_math_budget dflt;
    budget = budget_or(budget, &dflt);
    const uint64_t G = d->ngroups, C = d->cells;
    uint64_t *per = zalloc((G + 1) * 8);
    if (!per) { set_error(error, TR_MATH_MEMORY); return NULL; }
    uint64_t largest = 0, total = 0;
    for (uint64_t g = 0; g < G; g++) {
        uint64_t n = 0;
        for (uint64_t e = 0; e < C; e++) {
            uint64_t i = g * C + e;
            n += d->has_counts ? d->count[i] != 0 : get128(d->value[i]) != 0;
        }
        per[g + 1] = per[g] + n; total += n;
        if (n > largest) largest = n;
    }
    tr_math_need need;
    need_zero(&need);
    need.need[TR_BUDGET_DEGREE] = (uint64_t)d->order;
    need.need[TR_BUDGET_SUPPORT] = largest;
    need.need[TR_BUDGET_BYTES] = tr_budget_add(extra_bytes, object_bytes(TR_TENSOR_SPARSE, G, 0, total, d->order, d->has_counts, 0, 0));
    int32_t rc = tr_budget_check(budget, &need, refusal);
    if (rc) { free(per); set_error(error, rc); return NULL; }
    tr_tensor *t = tensor_alloc(TR_TENSOR_SPARSE, d->order, d->dims, G, total, 0, d->has_counts);
    if (!t) { free(per); set_error(error, TR_MATH_MEMORY); return NULL; }
    memcpy(t->offset, per, (size_t)(G + 1) * 8);
    free(per);
    uint64_t o = 0;
    for (uint64_t g = 0; g < G; g++)
        for (uint64_t e = 0; e < C; e++) {
            uint64_t i = g * C + e;
            if (d->has_counts ? d->count[i] == 0 : get128(d->value[i]) == 0) continue;
            decode(d->dims, d->stride, d->order, e, t->coord + o * (uint64_t)d->order);
            t->value[o] = d->value[i];
            if (d->has_counts) t->count[o] = d->count[i];
            o++;
        }
    copy_meta(t, d);
    t->bound = d->bound;
    set_error(error, TR_MATH_OK);
    return t;
}

/* ------------------------------------------------------------------ building from a scope */
enum { B_DENSE = 1, B_SORT = 2, B_FACTORED = 3 };
typedef struct { uint64_t g, row; } term_key;
struct tr_tensor_builder_s {
    uint32_t magic;
    int32_t mode, spent, canonical;      /* canonical: a symmetric dense state accumulated on sorted indices */
    const tr_scope *scope;
    tr_scope_view view;
    tr_tensor_build_plan plan;
    tr_tensor_spec spec;
    uint64_t digest[2];
    u128 bound;                          /* the proven bound */
    uint64_t *rows, *used;
    tr_math_i128 *value; uint64_t *count;   /* B_DENSE: G x cells */
    rec_t *rec; uint64_t nrec, caprec;      /* B_SORT */
    term_key *tkey; tr_math_i128 *tweight; int64_t *tfactor; uint64_t nterms, capterms;   /* B_FACTORED */
    uint64_t *covered, ncovered;
    uint64_t emissions;                  /* exact (B_SORT and AUTO), else 0 */
    uint64_t scratch_bytes;              /* what the builder holds besides the final state (for finish) */
};
static int builder_ok(const tr_tensor_builder *b) { return b && b->magic == BUILDER_MAGIC; }
static void builder_release(tr_tensor_builder *b) {
    if (!b) return;
    free(b->rows); free(b->used); free(b->value); free(b->count); free(b->rec); free(b->tkey); free(b->tweight);
    free(b->tfactor); free(b->covered);
    b->magic = 0;
    free(b);
}
void tr_tensor_builder_free(tr_tensor_builder *b) { if (builder_ok(b)) builder_release(b); }
static uint64_t range_mask(uint64_t w, uint64_t begin, uint64_t end) {
    uint64_t lo = w * 64, a = begin > lo ? begin - lo : 0, z = end - lo < 64 ? end - lo : 64;
    uint64_t m = z >= 64 ? UINT64_MAX : (UINT64_C(1) << z) - 1;
    return a ? m & ~((UINT64_C(1) << a) - 1) : m;
}
/* the distinct scope columns a plan reads */
static int32_t columns_read(const tr_scope_view *v, const tr_tensor_build_plan *p) {
    uint32_t read = p->refmask;
    if (p->kind == TR_TENSOR_COUNT) for (int32_t k = 0; k < p->order; k++) read |= UINT32_C(1) << p->cols[k][0];
    if (p->weight >= 0) read |= UINT32_C(1) << p->weight;
    int32_t n = 0;
    for (int32_t c = 0; c < v->ncols; c++) n += (int32_t)((read >> c) & 1);
    return n;
}
/* the exact emissions and contributing rows per group of the selected rows */
static uint64_t count_emissions(const tr_scope_view *v, const tr_tensor_build_plan *p, uint64_t *terms_per_group,
                                uint64_t *largest_group_terms) {
    uint64_t total = 0, nw = (v->rows + 63) / 64;
    for (uint64_t w = 0; w < nw; w++) {
        uint64_t bits = v->selection[w];
        while (bits) {
            uint64_t r = (w << 6) | (uint64_t)__builtin_ctzll(bits);
            bits &= bits - 1;
            __int128 wt;
            if (!tr_tensor_row_ok(v, p, r, &wt)) continue;
            if (terms_per_group) terms_per_group[tr_scope_group(v, r)]++;
            total = tr_budget_add(total, tr_tensor_emissions(v, p, r));
        }
    }
    if (terms_per_group && largest_group_terms) {
        uint64_t m = 0;
        for (uint64_t g = 0; g < v->ngroups; g++) if (terms_per_group[g] > m) m = terms_per_group[g];
        *largest_group_terms = m;
    }
    return total;
}
static int32_t choose_mode(const tr_tensor_build_plan *p, uint64_t G, uint64_t emissions) {
    if (p->repr == TR_TENSOR_DENSE) return B_DENSE;
    if (p->repr == TR_TENSOR_FACTORED) return B_FACTORED;
    if (p->strategy == TR_TENSOR_STRATEGY_COUNT) return B_DENSE;
    if (p->strategy == TR_TENSOR_STRATEGY_SORT) return B_SORT;
    uint64_t total = tr_budget_mul(G, p->cells);
    uint64_t room = tr_budget_add(tr_budget_mul(emissions, TR_TENSOR_COUNT_FACTOR), TR_TENSOR_COUNT_SLACK);
    return total <= room ? B_DENSE : B_SORT;
}
/* what init checks: the builder and its scratch (dense state or counting scratch, records, terms), the bitmap */
static void builder_need(const tr_scope_view *v, const tr_tensor_build_plan *p, int32_t mode, uint64_t emissions,
                         uint64_t nterms, uint64_t largest_terms, tr_math_need *need) {
    need_zero(need);
    uint64_t G = v->ngroups, total = tr_budget_mul(G, p->cells);
    need->need[TR_BUDGET_VARIABLES] = (uint64_t)columns_read(v, p);
    need->need[TR_BUDGET_DEGREE] = (uint64_t)p->order;
    uint64_t bytes = tr_budget_add(sizeof(tr_tensor_builder), tr_budget_add(tr_budget_bitmap_bytes(v->rows), tr_budget_mul(G, 16)));
    if (mode == B_DENSE) {
        need->need[TR_BUDGET_TENSOR] = total;
        need->need[TR_BUDGET_STATES] = total;
        bytes = tr_budget_add(bytes, tr_budget_mul(total, p->has_counts ? 24 : 16));
        bytes = tr_budget_add(bytes, sizeof(struct tr_tensor_s));
    } else if (mode == B_SORT) {
        need->need[TR_BUDGET_STATES] = emissions;
        bytes = tr_budget_add(bytes, tr_budget_mul(emissions, REC_BYTES));
    } else {
        uint64_t fdim = 0;
        for (int32_t k = 0; k < p->order; k++) fdim += p->dims[k];
        need->need[TR_BUDGET_SUPPORT] = largest_terms;
        bytes = tr_budget_add(bytes, tr_budget_mul(nterms, tr_budget_add(sizeof(term_key) + 16, tr_budget_mul(fdim, 8))));
        bytes = tr_budget_add(bytes, object_bytes(TR_TENSOR_FACTORED, G, 0, 0, p->order, 0, nterms, fdim));
    }
    need->need[TR_BUDGET_BYTES] = bytes;
}

tr_tensor_builder *tr_tensor_init(const tr_scope *scope, const tr_tensor_spec *spec, const tr_math_budget *budget,
                                  tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    tr_scope_view view;
    if (!scope || !spec || tr_scope_get_view(scope, &view)) { set_error(error, TR_MATH_INVALID); return NULL; }
    tr_tensor_build_plan plan;
    int32_t rc = tr_tensor_plan_make(&view, spec, &plan);
    if (rc) { set_error(error, rc); return NULL; }
    if (tr_tensor_plan_bound(&view, &plan) > LIMIT) { set_error(error, TR_MATH_OVERFLOW); return NULL; }
    const uint64_t G = view.ngroups;
    uint64_t emissions = 0, largest = 0, nterms = 0;
    uint64_t *per = NULL;
    int32_t mode;
    if (plan.repr == TR_TENSOR_FACTORED) {
        per = zalloc(G * 8);
        if (!per) { set_error(error, TR_MATH_MEMORY); return NULL; }
        count_emissions(&view, &plan, per, &largest);
        for (uint64_t g = 0; g < G; g++) nterms += per[g];
        free(per); per = NULL;
        mode = B_FACTORED;
    } else {
        if (plan.repr == TR_TENSOR_SPARSE && plan.strategy != TR_TENSOR_STRATEGY_COUNT)
            emissions = count_emissions(&view, &plan, NULL, NULL);
        mode = choose_mode(&plan, G, emissions);
    }
    tr_math_need need;
    builder_need(&view, &plan, mode, emissions, nterms, largest, &need);
    tr_math_budget dflt;
    budget = budget_or(budget, &dflt);
    rc = tr_budget_check(budget, &need, refusal);
    if (rc) { set_error(error, rc); return NULL; }
    tr_tensor_builder *b = zalloc(sizeof *b);
    if (!b) { set_error(error, TR_MATH_MEMORY); return NULL; }
    tr_scope_info info;
    tr_scope_get_info(scope, &info);
    b->mode = mode; b->scope = scope; b->view = view; b->plan = plan; b->spec = *spec;
    b->digest[0] = info.digest[0]; b->digest[1] = info.digest[1];
    b->bound = tr_tensor_plan_bound(&view, &plan);
    b->emissions = emissions;
    b->canonical = mode == B_DENSE && plan.symmetric;
    b->rows = zalloc(G * 8); b->used = zalloc(G * 8);
    b->covered = zalloc(((view.rows + 63) / 64) * 8);
    int ok = b->rows && b->used && b->covered;
    if (mode == B_DENSE) {
        uint64_t e = G * plan.cells;
        b->value = zalloc(e * 16);
        if (plan.has_counts) b->count = zalloc(e * 8);
        ok = ok && b->value && (!plan.has_counts || b->count);
        b->scratch_bytes = plan.repr == TR_TENSOR_SPARSE ? need.need[TR_BUDGET_BYTES] : 0;
    } else if (mode == B_SORT) {
        b->caprec = emissions;
        b->rec = zalloc(emissions * REC_BYTES);
        ok = ok && b->rec;
        b->scratch_bytes = need.need[TR_BUDGET_BYTES];
    } else {
        uint64_t fdim = 0;
        for (int32_t k = 0; k < plan.order; k++) fdim += plan.dims[k];
        b->capterms = nterms;
        b->tkey = zalloc(nterms * sizeof(term_key));
        b->tweight = zalloc(nterms * 16);
        b->tfactor = zalloc(nterms * fdim * 8);
        ok = ok && b->tkey && b->tweight && b->tfactor;
    }
    if (!ok) { builder_release(b); set_error(error, TR_MATH_MEMORY); return NULL; }
    b->magic = BUILDER_MAGIC;
    set_error(error, TR_MATH_OK);
    return b;
}

static void accumulate_range(tr_tensor_builder *b, uint64_t begin, uint64_t end) {
    const tr_scope_view *v = &b->view;
    const tr_tensor_build_plan *p = &b->plan;
    const uint64_t C = p->cells;
    const int32_t m = p->order;
    uint64_t fdim = 0;
    for (int32_t k = 0; k < m; k++) fdim += p->dims[k];
    for (uint64_t w = begin >> 6; w <= (end - 1) >> 6; w++) {
        uint64_t bits = v->selection[w] & range_mask(w, begin, end);
        while (bits) {
            uint64_t r = (w << 6) | (uint64_t)__builtin_ctzll(bits);
            bits &= bits - 1;
            uint32_t g = tr_scope_group(v, r);
            b->rows[g]++;
            __int128 wt;
            if (!tr_tensor_row_ok(v, p, r, &wt)) continue;
            b->used[g]++;
            if (b->mode == B_FACTORED) {
                uint64_t j = b->nterms++;
                b->tkey[j].g = g; b->tkey[j].row = r;
                b->tweight[j] = set128(wt);
                int64_t *f = b->tfactor + j * fdim;
                for (int32_t k = 0; k < m; k++)
                    for (int32_t i = 0; i < p->ncols[k]; i++) {
                        int32_t c = p->cols[k][i];
                        *f++ = c < 0 ? 1 : tr_scope_value(v, c, r);
                    }
                continue;
            }
            tr_tensor_emitter e;
            int more = tr_tensor_emit_first(v, p, r, wt, b->canonical, &e);
            while (more) {
                uint64_t idx = e.idx[m];
                __int128 x = e.pre[m];
                if (b->mode == B_DENSE) {
                    uint64_t i = (uint64_t)g * C + idx;
                    b->value[i] = set128(get128(b->value[i]) + x);
                    if (b->count) b->count[i]++;
                } else {
                    rec_t *q = &b->rec[b->nrec++];
                    q->g = g; q->idx = idx; q->v = x; q->c = 1; q->pad = 0;
                }
                more = tr_tensor_emit_next(v, p, &e);
            }
        }
    }
}

int32_t tr_tensor_accumulate(tr_tensor_builder *b, uint64_t begin, uint64_t end) {
    if (!builder_ok(b) || b->spent || begin > end || end > b->view.rows) return TR_MATH_INVALID;
    if (begin == end) return TR_MATH_OK;
    for (uint64_t w = begin >> 6; w <= (end - 1) >> 6; w++)
        if (b->covered[w] & range_mask(w, begin, end)) return TR_MATH_INVALID;   /* a row is counted once */
    accumulate_range(b, begin, end);
    for (uint64_t w = begin >> 6; w <= (end - 1) >> 6; w++) b->covered[w] |= range_mask(w, begin, end);
    b->ncovered += end - begin;
    return TR_MATH_OK;
}

int32_t tr_tensor_merge_builder(tr_tensor_builder *into, const tr_tensor_builder *from) {
    if (!builder_ok(into) || !builder_ok(from) || into == from || into->spent || from->spent ||
        into->mode != from->mode || memcmp(&into->spec, &from->spec, sizeof into->spec)) return TR_MATH_INVALID;
    if (into->scope != from->scope) {
        int32_t same = tr_scope_same(into->scope, from->scope);
        if (same != 1) return same < 0 ? same : TR_MATH_INVALID;
    }
    uint64_t nw = (into->view.rows + 63) / 64;
    for (uint64_t w = 0; w < nw; w++) if (into->covered[w] & from->covered[w]) return TR_MATH_INVALID;
    const uint64_t G = into->view.ngroups;
    if (into->mode == B_SORT && from->nrec > into->caprec - into->nrec) return TR_MATH_INVALID;
    if (into->mode == B_FACTORED && from->nterms > into->capterms - into->nterms) return TR_MATH_INVALID;
    for (uint64_t g = 0; g < G; g++) { into->rows[g] += from->rows[g]; into->used[g] += from->used[g]; }
    if (into->mode == B_DENSE) {
        uint64_t e = G * into->plan.cells;
        for (uint64_t i = 0; i < e; i++) {          /* disjoint rows: proven to fit */
            into->value[i] = set128(get128(into->value[i]) + get128(from->value[i]));
            if (into->count) into->count[i] += from->count[i];
        }
    } else if (into->mode == B_SORT) {
        if (from->nrec) memcpy(into->rec + into->nrec, from->rec, (size_t)(from->nrec * REC_BYTES));
        into->nrec += from->nrec;
    } else {
        uint64_t fdim = 0;
        for (int32_t k = 0; k < into->plan.order; k++) fdim += into->plan.dims[k];
        if (from->nterms) {
            memcpy(into->tkey + into->nterms, from->tkey, (size_t)from->nterms * sizeof(term_key));
            memcpy(into->tweight + into->nterms, from->tweight, (size_t)from->nterms * 16);
            memcpy(into->tfactor + into->nterms * fdim, from->tfactor, (size_t)(from->nterms * fdim * 8));
        }
        into->nterms += from->nterms;
    }
    for (uint64_t w = 0; w < nw; w++) into->covered[w] |= from->covered[w];
    into->ncovered += from->ncovered;
    return TR_MATH_OK;
}

/* axis metadata of a scoped tensor from its plan */
static void scoped_meta(tr_tensor *t, const tr_scope_view *v, const tr_tensor_build_plan *p, const uint64_t *digest) {
    t->kind = p->kind; t->policy = p->policy; t->weighted = p->weight >= 0; t->wcol = p->weight;
    t->wscale = p->weight >= 0 ? v->col[p->weight].scale : 0;
    t->scoped = 1; t->symmetric = p->symmetric;
    t->digest[0] = digest[0]; t->digest[1] = digest[1];
    for (int32_t k = 0; k < p->order; k++) {
        tr_tensor_axis *a = &t->axis[k];
        memset(a, 0, sizeof *a);
        a->dim = p->dims[k]; a->col = -1;
        if (p->kind == TR_TENSOR_MOMENT) {
            a->kind = TR_TENSOR_AXIS_FEATURE; a->ncols = p->ncols[k];
            for (int32_t i = 0; i < TR_TENSOR_MAX_FEATURES; i++) a->cols[i] = i < p->ncols[k] ? p->cols[k][i] : 0;
        } else {
            a->kind = TR_TENSOR_AXIS_CATEGORY; a->ncols = 1; a->col = p->cols[k][0]; a->cols[0] = p->cols[k][0];
            a->lo = p->lo[k]; a->span = p->span[k]; a->nullslot = p->nullslot[k]; a->scale = v->col[a->col].scale;
        }
    }
}
/* completes a symmetric dense state accumulated on sorted indices: every entry from its canonical one */
static void symmetrize(tr_math_i128 *value, uint64_t *count, uint64_t G, const tr_tensor_build_plan *p) {
    const uint64_t C = p->cells;
    for (uint64_t e = 0; e < C; e++) {
        uint64_t c = tr_tensor_canonical(p->dims, p->stride, p->order, e);
        if (c == e) continue;
        for (uint64_t g = 0; g < G; g++) {
            value[g * C + e] = value[g * C + c];
            if (count) count[g * C + e] = count[g * C + c];
        }
    }
}
typedef struct { uint64_t g, row, i; } term_sort;
static int term_cmp(const void *x, const void *y) {
    const term_sort *a = x, *b = y;
    if (a->g != b->g) return a->g < b->g ? -1 : 1;
    return a->row < b->row ? -1 : a->row > b->row;
}

tr_tensor *tr_tensor_finish(tr_tensor_builder *b, const tr_math_budget *budget, tr_math_refusal *refusal,
                            int32_t *error) {
    tr_budget_none(refusal);
    if (!builder_ok(b) || b->spent || b->ncovered != b->view.rows) { set_error(error, TR_MATH_INVALID); return NULL; }
    tr_math_budget dflt;
    budget = budget_or(budget, &dflt);
    const tr_tensor_build_plan *p = &b->plan;
    const uint64_t G = b->view.ngroups;
    tr_tensor *t = NULL;
    if (b->mode == B_DENSE) {
        if (b->canonical) symmetrize(b->value, b->count, G, p);
        if (p->repr == TR_TENSOR_DENSE) {
            t = zalloc(sizeof *t);
            if (!t) { set_error(error, TR_MATH_MEMORY); return NULL; }
            t->repr = TR_TENSOR_DENSE; t->order = p->order; t->ngroups = G; t->has_counts = p->has_counts;
            for (int32_t k = 0; k < p->order; k++) t->dims[k] = p->dims[k];
            set_strides(t);
            t->nnz = G * t->cells;
            t->value = b->value; t->count = b->count; t->rows = b->rows; t->used = b->used;
            b->value = NULL; b->count = NULL; b->rows = NULL; b->used = NULL;
            t->bytes = object_bytes(TR_TENSOR_DENSE, G, t->cells, 0, p->order, p->has_counts, 0, 0);
            t->magic = TENSOR_MAGIC;
            scoped_meta(t, &b->view, p, b->digest);
            t->bound = scan_bound(t);
            t->strategy = TR_TENSOR_STRATEGY_DIRECT;
        } else {                                  /* a sparse build counted in a dense scratch */
            tr_tensor tmp;
            memset(&tmp, 0, sizeof tmp);
            tmp.magic = TENSOR_MAGIC; tmp.repr = TR_TENSOR_DENSE; tmp.order = p->order; tmp.ngroups = G;
            tmp.has_counts = p->has_counts;
            for (int32_t k = 0; k < p->order; k++) tmp.dims[k] = p->dims[k];
            set_strides(&tmp);
            tmp.nnz = G * tmp.cells; tmp.value = b->value; tmp.count = b->count; tmp.rows = b->rows; tmp.used = b->used;
            t = dense_to_sparse(&tmp, budget, b->scratch_bytes, refusal, error);
            if (!t) return NULL;
            memcpy(t->rows, b->rows, (size_t)G * 8); memcpy(t->used, b->used, (size_t)G * 8);
            scoped_meta(t, &b->view, p, b->digest);
            t->bound = scan_bound(t);
            t->strategy = TR_TENSOR_STRATEGY_COUNT;
        }
    } else if (b->mode == B_SORT) {
        t = records_to_sparse(b->rec, b->nrec, p->order, p->dims, G, p->has_counts, budget, b->scratch_bytes, refusal, error);
        if (!t) return NULL;
        memcpy(t->rows, b->rows, (size_t)G * 8); memcpy(t->used, b->used, (size_t)G * 8);
        scoped_meta(t, &b->view, p, b->digest);
        t->strategy = TR_TENSOR_STRATEGY_SORT;
    } else {
        uint64_t n = b->nterms, fdim = 0;
        for (int32_t k = 0; k < p->order; k++) fdim += p->dims[k];
        t = tensor_alloc(TR_TENSOR_FACTORED, p->order, p->dims, G, 0, n, 0);
        if (!t) { set_error(error, TR_MATH_MEMORY); return NULL; }
        /* the terms in (group, physical row) order: any chunking gives the same state */
        term_sort *order = zalloc(n * sizeof(term_sort));
        if (!order) { tensor_release(t); set_error(error, TR_MATH_MEMORY); return NULL; }
        for (uint64_t j = 0; j < n; j++) { order[j].g = b->tkey[j].g; order[j].row = b->tkey[j].row; order[j].i = j; }
        if (n > 1) qsort(order, (size_t)n, sizeof *order, term_cmp);
        for (uint64_t j = 0; j < n; j++) {
            uint64_t s = order[j].i;
            t->offset[order[j].g + 1]++;
            t->weight[j] = b->tweight[s];
            if (fdim) memcpy(t->factor + j * fdim, b->tfactor + s * fdim, (size_t)fdim * 8);
        }
        free(order);
        for (uint64_t g = 0; g < G; g++) t->offset[g + 1] += t->offset[g];
        memcpy(t->rows, b->rows, (size_t)G * 8); memcpy(t->used, b->used, (size_t)G * 8);
        scoped_meta(t, &b->view, p, b->digest);
        t->bound = factored_bound(t);
        t->strategy = TR_TENSOR_STRATEGY_DIRECT;
    }
    b->spent = 1;
    set_error(error, TR_MATH_OK);
    return t;
}

tr_tensor *tr_tensor_build(const tr_scope *scope, const tr_tensor_spec *spec, const tr_math_budget *budget,
                           tr_math_refusal *refusal, int32_t *error) {
    tr_tensor_builder *b = tr_tensor_init(scope, spec, budget, refusal, error);
    if (!b) return NULL;
    int32_t rc = tr_tensor_accumulate(b, 0, b->view.rows);
    tr_tensor *t = rc ? NULL : tr_tensor_finish(b, budget, refusal, error);
    if (rc) set_error(error, rc);
    builder_release(b);
    return t;
}

int32_t tr_tensor_need(const tr_scope *scope, const tr_tensor_spec *spec, int32_t target, tr_math_need *need,
                       uint64_t *capacity) {
    tr_scope_view view;
    if (!scope || !spec || !need || tr_scope_get_view(scope, &view) ||
        (target != TR_TENSOR_NEED_CPU && target != TR_TENSOR_NEED_DEVICE && target != TR_TENSOR_NEED_ADOPT))
        return TR_MATH_INVALID;
    tr_tensor_build_plan plan;
    int32_t rc = tr_tensor_plan_make(&view, spec, &plan);
    if (rc) return rc;
    if (target == TR_TENSOR_NEED_DEVICE) {
        if (plan.repr == TR_TENSOR_FACTORED) return TR_MATH_UNSUPPORTED;
        tr_tensor_device_need(&view, &plan, need, capacity);
        return TR_MATH_OK;
    }
    uint64_t G = view.ngroups, cap = 0;
    if (target == TR_TENSOR_NEED_ADOPT) {
        if (plan.repr == TR_TENSOR_FACTORED) return TR_MATH_UNSUPPORTED;
        tr_math_need dev;
        tr_tensor_device_need(&view, &plan, &dev, &cap);
        need_zero(need);
        need->need[TR_BUDGET_VARIABLES] = dev.need[TR_BUDGET_VARIABLES];
        need->need[TR_BUDGET_DEGREE] = (uint64_t)plan.order;
        if (plan.repr == TR_TENSOR_DENSE) {
            need->need[TR_BUDGET_TENSOR] = tr_budget_mul(G, plan.cells);
            need->need[TR_BUDGET_BYTES] = object_bytes(TR_TENSOR_DENSE, G, plan.cells, 0, plan.order, plan.has_counts, 0, 0);
        } else {
            need->need[TR_BUDGET_SUPPORT] = cap < plan.cells ? cap : plan.cells;
            need->need[TR_BUDGET_BYTES] = object_bytes(TR_TENSOR_SPARSE, G, 0, cap, plan.order, plan.has_counts, 0, 0);
        }
        if (capacity) *capacity = cap;
        return TR_MATH_OK;
    }
    uint64_t emissions = 0, largest = 0, nterms = 0;
    if (plan.repr == TR_TENSOR_FACTORED) {
        uint64_t *per = zalloc(G * 8);
        if (!per) return TR_MATH_MEMORY;
        count_emissions(&view, &plan, per, &largest);
        for (uint64_t g = 0; g < G; g++) nterms += per[g];
        free(per);
    } else if (plan.repr == TR_TENSOR_SPARSE && plan.strategy != TR_TENSOR_STRATEGY_COUNT) {
        emissions = count_emissions(&view, &plan, NULL, NULL);
    }
    builder_need(&view, &plan, choose_mode(&plan, G, emissions), emissions, nterms, largest, need);
    if (capacity) *capacity = plan.repr == TR_TENSOR_FACTORED ? nterms : emissions;
    return TR_MATH_OK;
}

/* ------------------------------------------------------------------ adoption of device states */
tr_tensor *tr_tensor_from_states(const tr_scope *origin, const tr_tensor_spec *spec, const tr_tensor_states *st,
                                 const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    tr_scope_view view;
    tr_tensor_build_plan p;
    if (!origin || !spec || !st || tr_scope_get_view(origin, &view)) { set_error(error, TR_MATH_INVALID); return NULL; }
    int32_t rc = tr_tensor_plan_make(&view, spec, &p);
    if (rc) { set_error(error, rc); return NULL; }
    if (p.repr == TR_TENSOR_FACTORED) { set_error(error, TR_MATH_UNSUPPORTED); return NULL; }
    const uint64_t G = view.ngroups;
    u128 bound = tr_tensor_plan_bound(&view, &p);
    if (bound > LIMIT) { set_error(error, TR_MATH_OVERFLOW); return NULL; }
    int bad = st->ngroups != G || st->order != p.order || st->repr != p.repr || st->has_counts != p.has_counts ||
              !st->rows || !st->used || !st->value || (p.has_counts && !st->count);
    for (int32_t k = 0; !bad && k < p.order; k++) bad = st->dims[k] != p.dims[k];
    if (!bad && p.repr == TR_TENSOR_DENSE) bad = st->cells != p.cells || st->nnz != G * p.cells;
    if (!bad && p.repr == TR_TENSOR_SPARSE) bad = !st->offset || (p.order && !st->coord) || st->nnz > st->capacity ||
                                                 st->offset[0] != 0 || st->offset[G] != st->nnz;
    uint64_t total = 0;
    for (uint64_t g = 0; !bad && g < G; g++) {
        bad = st->used[g] > st->rows[g] || __builtin_add_overflow(total, st->rows[g], &total);
        if (!bad && p.repr == TR_TENSOR_SPARSE) bad = st->offset[g + 1] < st->offset[g];
    }
    if (!bad) bad = total != view.selected;
    if (bad) { set_error(error, TR_MATH_INVALID); return NULL; }
    tr_math_need need;
    need_zero(&need);
    need.need[TR_BUDGET_DEGREE] = (uint64_t)p.order;
    if (p.repr == TR_TENSOR_DENSE) need.need[TR_BUDGET_TENSOR] = tr_budget_mul(G, p.cells);
    else need.need[TR_BUDGET_SUPPORT] = max_group(st->offset, G);
    need.need[TR_BUDGET_BYTES] = object_bytes(p.repr, G, p.cells, st->nnz, p.order, p.has_counts, 0, 0);
    tr_math_budget dflt;
    rc = tr_budget_check(budget_or(budget, &dflt), &need, refusal);
    if (rc) { set_error(error, rc); return NULL; }
    tr_tensor *t = tensor_alloc(p.repr, p.order, p.dims, G, st->nnz, 0, p.has_counts);
    if (!t) { set_error(error, TR_MATH_MEMORY); return NULL; }
    memcpy(t->rows, st->rows, (size_t)G * 8); memcpy(t->used, st->used, (size_t)G * 8);
    uint64_t n = t->nnz;
    memcpy(t->value, st->value, (size_t)n * 16);
    if (p.has_counts) memcpy(t->count, st->count, (size_t)n * 8);
    if (p.repr == TR_TENSOR_SPARSE) {
        memcpy(t->offset, st->offset, (size_t)(G + 1) * 8);
        if (p.order) memcpy(t->coord, st->coord, (size_t)(n * (uint64_t)p.order * 4));
        for (uint64_t g = 0; !bad && g < G; g++)            /* canonical: strictly increasing within the group */
            for (uint64_t e = t->offset[g]; !bad && e < t->offset[g + 1]; e++) {
                const uint32_t *c = t->coord + e * (uint64_t)p.order;
                for (int32_t k = 0; k < p.order; k++) bad |= c[k] >= p.dims[k];
                if (e > t->offset[g]) bad |= coord_cmp(c - p.order, c, p.order) >= 0;
                bad |= p.has_counts ? t->count[e] == 0 : get128(t->value[e]) == 0;
            }
    }
    for (uint64_t i = 0; !bad && i < n; i++) {
        bad = tr_tensor_mag(get128(t->value[i])) > bound;
        if (!bad && p.has_counts) bad = t->count[i] > view.selected || (!t->count[i] && get128(t->value[i]));
    }
    if (bad) { tensor_release(t); set_error(error, TR_MATH_INVALID); return NULL; }
    tr_scope_info info;
    tr_scope_get_info(origin, &info);
    scoped_meta(t, &view, &p, info.digest);
    t->bound = scan_bound(t);
    t->strategy = TR_TENSOR_STRATEGY_DEVICE;
    set_error(error, TR_MATH_OK);
    return t;
}

/* ------------------------------------------------------------------ GENERIC tensors from arrays */
static int32_t shape_ok(int32_t order, const uint64_t *dims, uint64_t G) {
    if (order < 0 || order > TR_TENSOR_MAX_ORDER || (order && !dims) || G < 1 || G > TR_SCOPE_MAX_GROUPS) {
        return order > TR_TENSOR_MAX_ORDER ? TR_MATH_UNSUPPORTED : TR_MATH_INVALID;
    }
    for (int32_t k = 0; k < order; k++) if (dims[k] > TR_TENSOR_MAX_DIM) return TR_MATH_UNSUPPORTED;
    if (tr_tensor_cells(dims, order) == TR_BUDGET_UNKNOWN) return TR_MATH_UNSUPPORTED;
    return TR_MATH_OK;
}
tr_tensor *tr_tensor_from_dense(int32_t order, const uint64_t *dims, uint64_t G, const tr_math_i128 *values,
                                const uint64_t *counts, const tr_math_budget *budget, tr_math_refusal *refusal,
                                int32_t *error) {
    tr_budget_none(refusal);
    int32_t rc = shape_ok(order, dims, G);
    if (rc) { set_error(error, rc); return NULL; }
    uint64_t cells = tr_tensor_cells(dims, order), e = tr_budget_mul(G, cells);
    if (e && !values) { set_error(error, TR_MATH_INVALID); return NULL; }
    tr_math_need need;
    need_zero(&need);
    need.need[TR_BUDGET_DEGREE] = (uint64_t)order;
    need.need[TR_BUDGET_TENSOR] = e;
    need.need[TR_BUDGET_BYTES] = object_bytes(TR_TENSOR_DENSE, G, cells, 0, order, counts != NULL, 0, 0);
    tr_math_budget dflt;
    rc = tr_budget_check(budget_or(budget, &dflt), &need, refusal);
    if (rc) { set_error(error, rc); return NULL; }
    for (uint64_t i = 0; i < e; i++)
        if (tr_tensor_mag(get128(values[i])) > LIMIT || (counts && !counts[i] && get128(values[i]))) {
            set_error(error, tr_tensor_mag(get128(values[i])) > LIMIT ? TR_MATH_OVERFLOW : TR_MATH_INVALID); return NULL;
        }
    tr_tensor *t = tensor_alloc(TR_TENSOR_DENSE, order, dims, G, 0, 0, counts != NULL);
    if (!t) { set_error(error, TR_MATH_MEMORY); return NULL; }
    if (e) memcpy(t->value, values, (size_t)e * 16);
    if (counts && e) memcpy(t->count, counts, (size_t)e * 8);
    t->bound = scan_bound(t);
    t->strategy = TR_TENSOR_STRATEGY_ARRAYS;
    set_error(error, TR_MATH_OK);
    return t;
}
tr_tensor *tr_tensor_from_sparse(int32_t order, const uint64_t *dims, uint64_t G, uint64_t nnz, const uint32_t *groups,
                                 const uint32_t *coords, const tr_math_i128 *values, const uint64_t *counts,
                                 const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    int32_t rc = shape_ok(order, dims, G);
    if (rc) { set_error(error, rc); return NULL; }
    if (nnz && (!values || (order && !coords))) { set_error(error, TR_MATH_INVALID); return NULL; }
    tr_math_need need;
    need_zero(&need);
    need.need[TR_BUDGET_DEGREE] = (uint64_t)order;
    need.need[TR_BUDGET_STATES] = nnz;
    need.need[TR_BUDGET_BYTES] = tr_budget_mul(nnz, REC_BYTES);
    tr_math_budget dflt;
    budget = budget_or(budget, &dflt);
    rc = tr_budget_check(budget, &need, refusal);
    if (rc) { set_error(error, rc); return NULL; }
    rec_t *r = zalloc(nnz * REC_BYTES);
    if (!r) { set_error(error, TR_MATH_MEMORY); return NULL; }
    uint64_t stride[TR_TENSOR_MAX_ORDER], c = 1;
    for (int32_t k = order - 1; k >= 0; k--) { stride[k] = c; c *= dims[k] ? dims[k] : 1; }
    for (uint64_t i = 0; i < nnz; i++) {
        uint64_t g = groups ? groups[i] : 0, idx = 0;
        int bad = g >= G || tr_tensor_mag(get128(values[i])) > LIMIT || (counts && !counts[i] && get128(values[i]));
        for (int32_t k = 0; !bad && k < order; k++) {
            uint32_t x = coords[i * (uint64_t)order + (uint64_t)k];
            bad = x >= dims[k];
            idx += (uint64_t)x * stride[k];
        }
        if (bad) { free(r); set_error(error, tr_tensor_mag(get128(values[i])) > LIMIT ? TR_MATH_OVERFLOW : TR_MATH_INVALID); return NULL; }
        r[i].g = g; r[i].idx = idx; r[i].v = get128(values[i]); r[i].c = counts ? counts[i] : 0;
    }
    tr_tensor *t = records_to_sparse(r, nnz, order, dims, G, counts != NULL, budget, tr_budget_mul(nnz, REC_BYTES), refusal, error);
    free(r);
    if (t) t->strategy = TR_TENSOR_STRATEGY_ARRAYS;
    return t;
}
tr_tensor *tr_tensor_from_factors(int32_t order, const uint64_t *dims, uint64_t G, uint64_t nterms,
                                  const uint32_t *groups, const tr_math_i128 *weights, const int64_t *factors,
                                  const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    int32_t rc = shape_ok(order, dims, G);
    if (rc) { set_error(error, rc); return NULL; }
    uint64_t fdim = 0;
    for (int32_t k = 0; k < order; k++) fdim += dims[k];
    if (nterms && (!weights || (fdim && !factors))) { set_error(error, TR_MATH_INVALID); return NULL; }
    uint64_t *per = zalloc((G + 1) * 8);
    if (!per) { set_error(error, TR_MATH_MEMORY); return NULL; }
    for (uint64_t j = 0; j < nterms; j++) {
        uint64_t g = groups ? groups[j] : 0;
        if (g >= G || tr_tensor_mag(get128(weights[j])) > LIMIT) { free(per); set_error(error, TR_MATH_INVALID); return NULL; }
        per[g + 1]++;
    }
    uint64_t largest = 0;
    for (uint64_t g = 0; g < G; g++) { if (per[g + 1] > largest) largest = per[g + 1]; per[g + 1] += per[g]; }
    tr_math_need need;
    need_zero(&need);
    need.need[TR_BUDGET_DEGREE] = (uint64_t)order;
    need.need[TR_BUDGET_SUPPORT] = largest;
    need.need[TR_BUDGET_BYTES] = object_bytes(TR_TENSOR_FACTORED, G, 0, 0, order, 0, nterms, fdim);
    tr_math_budget dflt;
    rc = tr_budget_check(budget_or(budget, &dflt), &need, refusal);
    if (rc) { free(per); set_error(error, rc); return NULL; }
    tr_tensor *t = tensor_alloc(TR_TENSOR_FACTORED, order, dims, G, 0, nterms, 0);
    if (!t) { free(per); set_error(error, TR_MATH_MEMORY); return NULL; }
    memcpy(t->offset, per, (size_t)(G + 1) * 8);
    for (uint64_t j = 0; j < nterms; j++) {          /* stable by group */
        uint64_t g = groups ? groups[j] : 0, at = per[g]++;
        t->weight[at] = weights[j];
        if (fdim) memcpy(t->factor + at * fdim, factors + j * fdim, (size_t)fdim * 8);
    }
    free(per);
    t->bound = factored_bound(t);
    if (t->bound > LIMIT) { tensor_release(t); set_error(error, TR_MATH_OVERFLOW); return NULL; }
    t->strategy = TR_TENSOR_STRATEGY_ARRAYS;
    set_error(error, TR_MATH_OK);
    return t;
}

/* ------------------------------------------------------------------ factored terms */
/* the nonzero entries of term j of a factored tensor: (linear index, weight x product) by an odometer over the
 * nonzero factor coordinates (exact: the tensor's bound covers every product) */
typedef struct { int64_t i[TR_TENSOR_MAX_ORDER]; __int128 pre[TR_TENSOR_MAX_ORDER + 1]; uint64_t idx[TR_TENSOR_MAX_ORDER + 1]; } term_iter;
static int term_step(const tr_tensor *t, const int64_t *f, term_iter *it, int32_t level) {
    const int32_t m = t->order;
    for (;;) {
        const int64_t *fk = f;
        for (int32_t k = 0; k < level; k++) fk += t->dims[k];
        uint64_t c = (uint64_t)(it->i[level] + 1);
        while (c < t->dims[level] && fk[c] == 0) c++;
        if (c >= t->dims[level]) {
            if (level == 0) return 0;
            level--;
            continue;
        }
        it->i[level] = (int64_t)c;
        it->pre[level + 1] = it->pre[level] * (__int128)fk[c];
        it->idx[level + 1] = it->idx[level] + c * t->stride[level];
        if (level == m - 1) return 1;
        level++;
        it->i[level] = -1;
    }
}
static int term_first(const tr_tensor *t, uint64_t j, term_iter *it) {
    it->pre[0] = get128(t->weight[j]); it->idx[0] = 0;
    if (it->pre[0] == 0) return 0;
    if (t->order == 0) return 1;
    it->i[0] = -1;
    return term_step(t, t->factor + j * t->fdim, it, 0);
}
static int term_next(const tr_tensor *t, uint64_t j, term_iter *it) {
    if (t->order == 0) return 0;
    return term_step(t, t->factor + j * t->fdim, it, t->order - 1);
}
/* prod_k (nonzero entries of f_{j,k}) for a term of nonzero weight (saturating) */
static uint64_t term_emissions(const tr_tensor *t, uint64_t j) {
    if (get128(t->weight[j]) == 0) return 0;
    const int64_t *f = t->factor + j * t->fdim;
    uint64_t n = 1;
    for (int32_t k = 0; k < t->order; k++) {
        uint64_t a = 0;
        for (uint64_t i = 0; i < t->dims[k]; i++) a += f[i] != 0;
        n = tr_budget_mul(n, a);
        f += t->dims[k];
    }
    return n;
}

/* ------------------------------------------------------------------ conversions */
static int32_t check_one(const tr_math_budget *budget, tr_math_need *need, tr_math_refusal *refusal) {
    tr_math_budget dflt;
    return tr_budget_check(budget_or(budget, &dflt), need, refusal);
}
static tr_tensor *factored_to_dense(const tr_tensor *f, const tr_math_budget *budget, uint64_t extra,
                                    tr_math_refusal *refusal, int32_t *error) {
    tr_math_need need;
    need_zero(&need);
    need.need[TR_BUDGET_DEGREE] = (uint64_t)f->order;
    need.need[TR_BUDGET_TENSOR] = tr_budget_mul(f->ngroups, f->cells);
    need.need[TR_BUDGET_BYTES] = tr_budget_add(extra, object_bytes(TR_TENSOR_DENSE, f->ngroups, f->cells, 0, f->order, 0, 0, 0));
    int32_t rc = check_one(budget, &need, refusal);
    if (rc) { set_error(error, rc); return NULL; }
    tr_tensor *t = tensor_alloc(TR_TENSOR_DENSE, f->order, f->dims, f->ngroups, 0, 0, 0);
    if (!t) { set_error(error, TR_MATH_MEMORY); return NULL; }
    for (uint64_t g = 0; g < f->ngroups; g++)
        for (uint64_t j = f->offset[g]; j < f->offset[g + 1]; j++) {
            term_iter it;
            for (int more = term_first(f, j, &it); more; more = term_next(f, j, &it)) {
                uint64_t i = g * f->cells + it.idx[f->order];
                t->value[i] = set128(get128(t->value[i]) + it.pre[f->order]);
            }
        }
    copy_meta(t, f);
    t->bound = scan_bound(t);
    set_error(error, TR_MATH_OK);
    return t;
}
static tr_tensor *factored_to_sparse(const tr_tensor *f, const tr_math_budget *budget, tr_math_refusal *refusal,
                                     int32_t *error) {
    uint64_t emissions = 0;
    for (uint64_t j = 0; j < f->nterms; j++) emissions = tr_budget_add(emissions, term_emissions(f, j));
    uint64_t total = tr_budget_mul(f->ngroups, f->cells);
    if (total <= tr_budget_add(tr_budget_mul(emissions, TR_TENSOR_COUNT_FACTOR), TR_TENSOR_COUNT_SLACK)) {
        uint64_t scratch = object_bytes(TR_TENSOR_DENSE, f->ngroups, f->cells, 0, f->order, 0, 0, 0);
        tr_tensor *d = factored_to_dense(f, budget, 0, refusal, error);
        if (!d) return NULL;
        tr_tensor *t = dense_to_sparse(d, budget, scratch, refusal, error);
        tensor_release(d);
        return t;
    }
    tr_math_need need;
    need_zero(&need);
    need.need[TR_BUDGET_DEGREE] = (uint64_t)f->order;
    need.need[TR_BUDGET_STATES] = emissions;
    need.need[TR_BUDGET_BYTES] = tr_budget_mul(emissions, REC_BYTES);
    int32_t rc = check_one(budget, &need, refusal);
    if (rc) { set_error(error, rc); return NULL; }
    rec_t *r = zalloc(emissions * REC_BYTES);
    if (!r) { set_error(error, TR_MATH_MEMORY); return NULL; }
    uint64_t n = 0;
    for (uint64_t g = 0; g < f->ngroups; g++)
        for (uint64_t j = f->offset[g]; j < f->offset[g + 1]; j++) {
            term_iter it;
            for (int more = term_first(f, j, &it); more; more = term_next(f, j, &it)) {
                r[n].g = g; r[n].idx = it.idx[f->order]; r[n].v = it.pre[f->order]; r[n].c = 0; n++;
            }
        }
    tr_math_budget dflt;
    tr_tensor *t = records_to_sparse(r, n, f->order, f->dims, f->ngroups, 0, budget_or(budget, &dflt),
                                     tr_budget_mul(emissions, REC_BYTES), refusal, error);
    free(r);
    if (t) copy_meta(t, f);
    return t;
}
static tr_tensor *sparse_to_dense(const tr_tensor *s, const tr_math_budget *budget, tr_math_refusal *refusal,
                                  int32_t *error) {
    tr_math_need need;
    need_zero(&need);
    need.need[TR_BUDGET_DEGREE] = (uint64_t)s->order;
    need.need[TR_BUDGET_TENSOR] = tr_budget_mul(s->ngroups, s->cells);
    need.need[TR_BUDGET_BYTES] = object_bytes(TR_TENSOR_DENSE, s->ngroups, s->cells, 0, s->order, s->has_counts, 0, 0);
    int32_t rc = check_one(budget, &need, refusal);
    if (rc) { set_error(error, rc); return NULL; }
    tr_tensor *t = tensor_alloc(TR_TENSOR_DENSE, s->order, s->dims, s->ngroups, 0, 0, s->has_counts);
    if (!t) { set_error(error, TR_MATH_MEMORY); return NULL; }
    for (uint64_t g = 0; g < s->ngroups; g++)
        for (uint64_t e = s->offset[g]; e < s->offset[g + 1]; e++) {
            uint64_t i = g * s->cells + linear(s, s->coord + e * (uint64_t)s->order);
            t->value[i] = s->value[e];
            if (s->has_counts) t->count[i] = s->count[e];
        }
    copy_meta(t, s);
    t->bound = s->bound;
    set_error(error, TR_MATH_OK);
    return t;
}
/* one unit rank-one term per stored nonzero entry (weight = the value) */
static tr_tensor *to_unit_terms(const tr_tensor *s, const tr_math_budget *budget, tr_math_refusal *refusal,
                                int32_t *error) {
    if (s->has_counts) { set_error(error, TR_MATH_UNSUPPORTED); return NULL; }
    const uint64_t G = s->ngroups;
    uint64_t *per = zalloc((G + 1) * 8);
    if (!per) { set_error(error, TR_MATH_MEMORY); return NULL; }
    for (uint64_t g = 0; g < G; g++) {
        uint64_t b = s->repr == TR_TENSOR_DENSE ? g * s->cells : s->offset[g];
        uint64_t e = s->repr == TR_TENSOR_DENSE ? (g + 1) * s->cells : s->offset[g + 1];
        for (uint64_t i = b; i < e; i++) per[g + 1] += get128(s->value[i]) != 0;
    }
    uint64_t largest = 0;
    for (uint64_t g = 0; g < G; g++) { if (per[g + 1] > largest) largest = per[g + 1]; per[g + 1] += per[g]; }
    uint64_t n = per[G];
    tr_math_need need;
    need_zero(&need);
    need.need[TR_BUDGET_DEGREE] = (uint64_t)s->order;
    need.need[TR_BUDGET_SUPPORT] = largest;
    need.need[TR_BUDGET_BYTES] = object_bytes(TR_TENSOR_FACTORED, G, 0, 0, s->order, 0, n, s->fdim);
    int32_t rc = check_one(budget, &need, refusal);
    if (rc) { free(per); set_error(error, rc); return NULL; }
    tr_tensor *t = tensor_alloc(TR_TENSOR_FACTORED, s->order, s->dims, G, 0, n, 0);
    if (!t) { free(per); set_error(error, TR_MATH_MEMORY); return NULL; }
    memcpy(t->offset, per, (size_t)(G + 1) * 8);
    free(per);
    uint64_t j = 0;
    uint32_t c[TR_TENSOR_MAX_ORDER];
    for (uint64_t g = 0; g < G; g++) {
        uint64_t b = s->repr == TR_TENSOR_DENSE ? g * s->cells : s->offset[g];
        uint64_t e = s->repr == TR_TENSOR_DENSE ? (g + 1) * s->cells : s->offset[g + 1];
        for (uint64_t i = b; i < e; i++) {
            if (!get128(s->value[i])) continue;
            if (s->repr == TR_TENSOR_DENSE) decode(s->dims, s->stride, s->order, i - b, c);
            else memcpy(c, s->coord + i * (uint64_t)s->order, (size_t)s->order * 4);
            t->weight[j] = s->value[i];
            int64_t *f = t->factor + j * t->fdim;
            for (int32_t k = 0; k < s->order; k++) { f[c[k]] = 1; f += s->dims[k]; }
            j++;
        }
    }
    copy_meta(t, s);
    t->bound = s->bound;           /* one term per entry: every partial sum is one stored value */
    set_error(error, TR_MATH_OK);
    return t;
}
static tr_tensor *clone_tensor(const tr_tensor *s, int32_t *error) {
    tr_tensor *t = tensor_alloc(s->repr, s->order, s->dims, s->ngroups, s->nnz, s->nterms, s->has_counts);
    if (!t) { set_error(error, TR_MATH_MEMORY); return NULL; }
    copy_meta(t, s);
    if (s->repr == TR_TENSOR_FACTORED) {
        memcpy(t->offset, s->offset, (size_t)(s->ngroups + 1) * 8);
        memcpy(t->weight, s->weight, (size_t)s->nterms * 16);
        memcpy(t->factor, s->factor, (size_t)(s->nterms * s->fdim * 8));
    } else {
        memcpy(t->value, s->value, (size_t)s->nnz * 16);
        if (s->has_counts) memcpy(t->count, s->count, (size_t)s->nnz * 8);
        if (s->repr == TR_TENSOR_SPARSE) {
            memcpy(t->offset, s->offset, (size_t)(s->ngroups + 1) * 8);
            memcpy(t->coord, s->coord, (size_t)(s->nnz * (uint64_t)s->order * 4));
        }
    }
    t->bound = s->bound; t->strategy = s->strategy;
    set_error(error, TR_MATH_OK);
    return t;
}
tr_tensor *tr_tensor_clone(const tr_tensor *s, int32_t *error) {
    if (!tensor_ok(s)) { set_error(error, TR_MATH_INVALID); return NULL; }
    return clone_tensor(s, error);
}
tr_tensor *tr_tensor_convert(const tr_tensor *s, int32_t repr, const tr_math_budget *budget, tr_math_refusal *refusal,
                             int32_t *error) {
    tr_budget_none(refusal);
    if (!tensor_ok(s) || (repr != TR_TENSOR_DENSE && repr != TR_TENSOR_SPARSE && repr != TR_TENSOR_FACTORED)) {
        set_error(error, TR_MATH_INVALID); return NULL;
    }
    tr_tensor *t = NULL;
    if (repr == s->repr) {
        tr_math_need need;
        need_zero(&need);
        need.need[TR_BUDGET_BYTES] = s->bytes;
        int32_t rc = check_one(budget, &need, refusal);
        if (rc) { set_error(error, rc); return NULL; }
        t = clone_tensor(s, error);
    } else if (repr == TR_TENSOR_DENSE) {
        t = s->repr == TR_TENSOR_SPARSE ? sparse_to_dense(s, budget, refusal, error)
                                        : factored_to_dense(s, budget, 0, refusal, error);
    } else if (repr == TR_TENSOR_SPARSE) {
        if (s->repr == TR_TENSOR_DENSE) {
            tr_math_budget dflt;
            t = dense_to_sparse(s, budget_or(budget, &dflt), 0, refusal, error);
        } else t = factored_to_sparse(s, budget, refusal, error);
    } else {
        t = to_unit_terms(s, budget, refusal, error);
    }
    if (t) t->strategy = TR_TENSOR_STRATEGY_DERIVED;
    return t;
}

/* ------------------------------------------------------------------ marginal, slice, permute */
static int popc(uint32_t x) { int n = 0; while (x) { x &= x - 1; n++; } return n; }
static void refresh_symmetric(tr_tensor *t) {
    int sym = t->kind == TR_TENSOR_MOMENT && t->order >= 2;
    for (int32_t k = 1; sym && k < t->order; k++)
        sym = t->axis[k].kind == TR_TENSOR_AXIS_FEATURE && t->axis[k].ncols == t->axis[0].ncols &&
              !memcmp(t->axis[k].cols, t->axis[0].cols, (size_t)t->axis[0].ncols);
    t->symmetric = sym;
}
tr_tensor *tr_tensor_marginal(const tr_tensor *s, uint32_t keep, const tr_math_budget *budget,
                              tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    if (!tensor_ok(s) || (s->order < 32 && (keep >> s->order))) { set_error(error, TR_MATH_INVALID); return NULL; }
    int32_t m = popc(keep), map[TR_TENSOR_MAX_ORDER];
    uint64_t dims[TR_TENSOR_MAX_ORDER];
    for (int32_t k = 0, j = 0; k < s->order; k++) if ((keep >> k) & 1) { dims[j] = s->dims[k]; map[j++] = k; }
    uint64_t cells = tr_tensor_cells(dims, m);
    tr_math_need need;
    need_zero(&need);
    need.need[TR_BUDGET_DEGREE] = (uint64_t)m;
    tr_tensor *t = NULL;
    int32_t rc;
    if (s->repr == TR_TENSOR_DENSE) {
        need.need[TR_BUDGET_TENSOR] = tr_budget_mul(s->ngroups, cells);
        need.need[TR_BUDGET_BYTES] = object_bytes(TR_TENSOR_DENSE, s->ngroups, cells, 0, m, s->has_counts, 0, 0);
        if ((rc = check_one(budget, &need, refusal))) { set_error(error, rc); return NULL; }
        t = tensor_alloc(TR_TENSOR_DENSE, m, dims, s->ngroups, 0, 0, s->has_counts);
        if (!t) { set_error(error, TR_MATH_MEMORY); return NULL; }
        uint32_t c[TR_TENSOR_MAX_ORDER];
        for (uint64_t g = 0; g < s->ngroups; g++)
            for (uint64_t e = 0; e < s->cells; e++) {
                uint64_t i = g * s->cells + e;
                __int128 v = get128(s->value[i]);
                if (!v && (!s->has_counts || !s->count[i])) continue;
                decode(s->dims, s->stride, s->order, e, c);
                uint64_t idx = 0;
                for (int32_t j = 0; j < m; j++) idx += (uint64_t)c[map[j]] * t->stride[j];
                uint64_t o = g * cells + idx;
                __int128 x;
                if (!add_ok(get128(t->value[o]), v, &x) ||
                    (s->has_counts && __builtin_add_overflow(t->count[o], s->count[i], &t->count[o]))) {
                    tensor_release(t); set_error(error, TR_MATH_OVERFLOW); return NULL;
                }
                t->value[o] = set128(x);
            }
        t->bound = scan_bound(t);
    } else if (s->repr == TR_TENSOR_SPARSE) {
        need.need[TR_BUDGET_STATES] = s->nnz;
        need.need[TR_BUDGET_BYTES] = tr_budget_mul(s->nnz, REC_BYTES);
        if ((rc = check_one(budget, &need, refusal))) { set_error(error, rc); return NULL; }
        rec_t *r = zalloc(s->nnz * REC_BYTES);
        if (!r) { set_error(error, TR_MATH_MEMORY); return NULL; }
        uint64_t stride[TR_TENSOR_MAX_ORDER], c1 = 1;
        for (int32_t j = m - 1; j >= 0; j--) { stride[j] = c1; c1 *= dims[j] ? dims[j] : 1; }
        for (uint64_t g = 0; g < s->ngroups; g++)
            for (uint64_t e = s->offset[g]; e < s->offset[g + 1]; e++) {
                const uint32_t *c = s->coord + e * (uint64_t)s->order;
                uint64_t idx = 0;
                for (int32_t j = 0; j < m; j++) idx += (uint64_t)c[map[j]] * stride[j];
                r[e].g = g; r[e].idx = idx; r[e].v = get128(s->value[e]); r[e].c = s->has_counts ? s->count[e] : 0;
            }
        tr_math_budget dflt;
        t = records_to_sparse(r, s->nnz, m, dims, s->ngroups, s->has_counts, budget_or(budget, &dflt),
                              tr_budget_mul(s->nnz, REC_BYTES), refusal, error);
        free(r);
        if (!t) return NULL;
    } else {
        uint64_t fdim = 0;
        for (int32_t j = 0; j < m; j++) fdim += dims[j];
        need.need[TR_BUDGET_SUPPORT] = max_group(s->offset, s->ngroups);
        need.need[TR_BUDGET_BYTES] = object_bytes(TR_TENSOR_FACTORED, s->ngroups, 0, 0, m, 0, s->nterms, fdim);
        if ((rc = check_one(budget, &need, refusal))) { set_error(error, rc); return NULL; }
        t = tensor_alloc(TR_TENSOR_FACTORED, m, dims, s->ngroups, 0, s->nterms, 0);
        if (!t) { set_error(error, TR_MATH_MEMORY); return NULL; }
        memcpy(t->offset, s->offset, (size_t)(s->ngroups + 1) * 8);
        for (uint64_t j = 0; j < s->nterms; j++) {
            __int128 w = get128(s->weight[j]);
            const int64_t *f = s->factor + j * s->fdim;
            int64_t *o = t->factor + j * fdim;
            for (int32_t k = 0; k < s->order; k++) {
                if ((keep >> k) & 1) { memcpy(o, f, (size_t)s->dims[k] * 8); o += s->dims[k]; }
                else {
                    __int128 sum = 0;
                    for (uint64_t i = 0; i < s->dims[k]; i++) sum += f[i];      /* < 2^96: exact */
                    if (!mul_ok(w, sum, &w)) { tensor_release(t); set_error(error, TR_MATH_OVERFLOW); return NULL; }
                }
                f += s->dims[k];
            }
            t->weight[j] = set128(w);
        }
        t->bound = factored_bound(t);
        if (t->bound > LIMIT) { tensor_release(t); set_error(error, TR_MATH_OVERFLOW); return NULL; }
    }
    int32_t kind = s->kind == TR_TENSOR_COUNT ? TR_TENSOR_COUNT : TR_TENSOR_GENERIC;
    u128 bound = t->bound;
    copy_meta(t, s);
    for (int32_t j = 0; j < m; j++) t->axis[j] = s->axis[map[j]];
    t->kind = kind; t->bound = bound; t->strategy = TR_TENSOR_STRATEGY_DERIVED;
    refresh_symmetric(t);
    set_error(error, TR_MATH_OK);
    return t;
}

/* the axis metadata of [lo, hi) of axis a */
static tr_tensor_axis slice_axis(const tr_tensor_axis *a, uint64_t lo, uint64_t hi) {
    tr_tensor_axis r = *a;
    r.dim = hi - lo;
    if (a->kind == TR_TENSOR_AXIS_FEATURE) {
        r.ncols = (int32_t)(hi - lo);
        memset(r.cols, 0, sizeof r.cols);
        for (uint64_t i = lo; i < hi; i++) r.cols[i - lo] = a->cols[i];
    } else if (a->kind == TR_TENSOR_AXIS_CATEGORY) {
        uint64_t top = hi < a->span ? hi : a->span;
        r.span = top > lo ? top - lo : 0;
        r.lo = (int64_t)((uint64_t)a->lo + (lo < a->span ? lo : a->span));
        r.nullslot = a->nullslot && lo <= a->span && a->span < hi;
    }
    return r;
}
tr_tensor *tr_tensor_slice(const tr_tensor *s, uint32_t fix, const uint64_t *lo, const uint64_t *hi,
                           const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    if (!tensor_ok(s) || (s->order && !lo) || (s->order < 32 && (fix >> s->order))) { set_error(error, TR_MATH_INVALID); return NULL; }
    int32_t m = 0, map[TR_TENSOR_MAX_ORDER];
    uint64_t dims[TR_TENSOR_MAX_ORDER];
    for (int32_t k = 0; k < s->order; k++) {
        if ((fix >> k) & 1) { if (lo[k] >= s->dims[k]) { set_error(error, TR_MATH_INVALID); return NULL; } continue; }
        if (!hi || lo[k] > hi[k] || hi[k] > s->dims[k]) { set_error(error, TR_MATH_INVALID); return NULL; }
        dims[m] = hi[k] - lo[k]; map[m++] = k;
    }
    uint64_t cells = tr_tensor_cells(dims, m);
    tr_math_need need;
    need_zero(&need);
    need.need[TR_BUDGET_DEGREE] = (uint64_t)m;
    tr_tensor *t = NULL;
    int32_t rc;
    uint32_t c[TR_TENSOR_MAX_ORDER];
    if (s->repr == TR_TENSOR_DENSE) {
        need.need[TR_BUDGET_TENSOR] = tr_budget_mul(s->ngroups, cells);
        need.need[TR_BUDGET_BYTES] = object_bytes(TR_TENSOR_DENSE, s->ngroups, cells, 0, m, s->has_counts, 0, 0);
        if ((rc = check_one(budget, &need, refusal))) { set_error(error, rc); return NULL; }
        t = tensor_alloc(TR_TENSOR_DENSE, m, dims, s->ngroups, 0, 0, s->has_counts);
        if (!t) { set_error(error, TR_MATH_MEMORY); return NULL; }
        for (uint64_t e = 0; e < cells; e++) {
            decode(t->dims, t->stride, m, e, c);
            uint64_t src = 0;
            for (int32_t k = 0, j = 0; k < s->order; k++)
                src += (((fix >> k) & 1) ? lo[k] : lo[k] + c[j++]) * s->stride[k];
            for (uint64_t g = 0; g < s->ngroups; g++) {
                t->value[g * cells + e] = s->value[g * s->cells + src];
                if (s->has_counts) t->count[g * cells + e] = s->count[g * s->cells + src];
            }
        }
        t->bound = scan_bound(t);
    } else if (s->repr == TR_TENSOR_SPARSE) {
        uint64_t *per = zalloc((s->ngroups + 1) * 8);
        if (!per) { set_error(error, TR_MATH_MEMORY); return NULL; }
        for (uint64_t g = 0; g < s->ngroups; g++)
            for (uint64_t e = s->offset[g]; e < s->offset[g + 1]; e++) {
                const uint32_t *x = s->coord + e * (uint64_t)s->order;
                int in = 1;
                for (int32_t k = 0; k < s->order && in; k++)
                    in = ((fix >> k) & 1) ? x[k] == lo[k] : (x[k] >= lo[k] && x[k] < hi[k]);
                per[g + 1] += (uint64_t)in;
            }
        for (uint64_t g = 0; g < s->ngroups; g++) per[g + 1] += per[g];
        need.need[TR_BUDGET_SUPPORT] = max_group(per, s->ngroups);
        need.need[TR_BUDGET_BYTES] = object_bytes(TR_TENSOR_SPARSE, s->ngroups, 0, per[s->ngroups], m, s->has_counts, 0, 0);
        if ((rc = check_one(budget, &need, refusal))) { free(per); set_error(error, rc); return NULL; }
        t = tensor_alloc(TR_TENSOR_SPARSE, m, dims, s->ngroups, per[s->ngroups], 0, s->has_counts);
        if (!t) { free(per); set_error(error, TR_MATH_MEMORY); return NULL; }
        memcpy(t->offset, per, (size_t)(s->ngroups + 1) * 8);
        free(per);
        uint64_t o = 0;
        for (uint64_t e = 0; e < s->nnz; e++) {                  /* the order is kept: a filter of sorted entries */
            const uint32_t *x = s->coord + e * (uint64_t)s->order;
            int in = 1;
            for (int32_t k = 0; k < s->order && in; k++)
                in = ((fix >> k) & 1) ? x[k] == lo[k] : (x[k] >= lo[k] && x[k] < hi[k]);
            if (!in) continue;
            for (int32_t j = 0; j < m; j++) t->coord[o * (uint64_t)m + (uint64_t)j] = x[map[j]] - (uint32_t)lo[map[j]];
            t->value[o] = s->value[e];
            if (s->has_counts) t->count[o] = s->count[e];
            o++;
        }
        t->bound = scan_bound(t);
    } else {
        uint64_t fdim = 0;
        for (int32_t j = 0; j < m; j++) fdim += dims[j];
        need.need[TR_BUDGET_SUPPORT] = max_group(s->offset, s->ngroups);
        need.need[TR_BUDGET_BYTES] = object_bytes(TR_TENSOR_FACTORED, s->ngroups, 0, 0, m, 0, s->nterms, fdim);
        if ((rc = check_one(budget, &need, refusal))) { set_error(error, rc); return NULL; }
        t = tensor_alloc(TR_TENSOR_FACTORED, m, dims, s->ngroups, 0, s->nterms, 0);
        if (!t) { set_error(error, TR_MATH_MEMORY); return NULL; }
        memcpy(t->offset, s->offset, (size_t)(s->ngroups + 1) * 8);
        for (uint64_t j = 0; j < s->nterms; j++) {
            __int128 w = get128(s->weight[j]);
            const int64_t *f = s->factor + j * s->fdim;
            int64_t *o = t->factor + j * fdim;
            for (int32_t k = 0; k < s->order; k++) {
                if ((fix >> k) & 1) {
                    if (!mul_ok(w, f[lo[k]], &w)) { tensor_release(t); set_error(error, TR_MATH_OVERFLOW); return NULL; }
                } else { memcpy(o, f + lo[k], (size_t)(hi[k] - lo[k]) * 8); o += hi[k] - lo[k]; }
                f += s->dims[k];
            }
            t->weight[j] = set128(w);
        }
        t->bound = factored_bound(t);
        if (t->bound > LIMIT) { tensor_release(t); set_error(error, TR_MATH_OVERFLOW); return NULL; }
    }
    u128 bound = t->bound;
    copy_meta(t, s);
    for (int32_t j = 0; j < m; j++) t->axis[j] = slice_axis(&s->axis[map[j]], lo[map[j]], hi[map[j]]);
    if (fix) t->kind = TR_TENSOR_GENERIC;
    t->bound = bound; t->strategy = TR_TENSOR_STRATEGY_DERIVED;
    refresh_symmetric(t);
    set_error(error, TR_MATH_OK);
    return t;
}

tr_tensor *tr_tensor_permute(const tr_tensor *s, const int32_t *perm, const tr_math_budget *budget,
                             tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    if (!tensor_ok(s) || (s->order && !perm)) { set_error(error, TR_MATH_INVALID); return NULL; }
    uint32_t seen = 0;
    uint64_t dims[TR_TENSOR_MAX_ORDER];
    for (int32_t k = 0; k < s->order; k++) {
        if (perm[k] < 0 || perm[k] >= s->order || ((seen >> perm[k]) & 1)) { set_error(error, TR_MATH_INVALID); return NULL; }
        seen |= UINT32_C(1) << perm[k];
        dims[k] = s->dims[perm[k]];
    }
    const int32_t m = s->order;
    tr_math_need need;
    need_zero(&need);
    need.need[TR_BUDGET_DEGREE] = (uint64_t)m;
    tr_tensor *t = NULL;
    int32_t rc;
    uint32_t c[TR_TENSOR_MAX_ORDER];
    if (s->repr == TR_TENSOR_DENSE) {
        need.need[TR_BUDGET_TENSOR] = tr_budget_mul(s->ngroups, s->cells);
        need.need[TR_BUDGET_BYTES] = s->bytes;
        if ((rc = check_one(budget, &need, refusal))) { set_error(error, rc); return NULL; }
        t = tensor_alloc(TR_TENSOR_DENSE, m, dims, s->ngroups, 0, 0, s->has_counts);
        if (!t) { set_error(error, TR_MATH_MEMORY); return NULL; }
        for (uint64_t e = 0; e < s->cells; e++) {
            decode(t->dims, t->stride, m, e, c);
            uint64_t src = 0;
            for (int32_t k = 0; k < m; k++) src += (uint64_t)c[k] * s->stride[perm[k]];
            for (uint64_t g = 0; g < s->ngroups; g++) {
                t->value[g * s->cells + e] = s->value[g * s->cells + src];
                if (s->has_counts) t->count[g * s->cells + e] = s->count[g * s->cells + src];
            }
        }
    } else if (s->repr == TR_TENSOR_SPARSE) {
        need.need[TR_BUDGET_STATES] = s->nnz;
        need.need[TR_BUDGET_BYTES] = tr_budget_mul(s->nnz, REC_BYTES);
        if ((rc = check_one(budget, &need, refusal))) { set_error(error, rc); return NULL; }
        rec_t *r = zalloc(s->nnz * REC_BYTES);
        if (!r) { set_error(error, TR_MATH_MEMORY); return NULL; }
        uint64_t stride[TR_TENSOR_MAX_ORDER], c1 = 1;
        for (int32_t k = m - 1; k >= 0; k--) { stride[k] = c1; c1 *= dims[k] ? dims[k] : 1; }
        for (uint64_t g = 0; g < s->ngroups; g++)
            for (uint64_t e = s->offset[g]; e < s->offset[g + 1]; e++) {
                const uint32_t *x = s->coord + e * (uint64_t)m;
                uint64_t idx = 0;
                for (int32_t k = 0; k < m; k++) idx += (uint64_t)x[perm[k]] * stride[k];
                r[e].g = g; r[e].idx = idx; r[e].v = get128(s->value[e]); r[e].c = s->has_counts ? s->count[e] : 0;
            }
        tr_math_budget dflt;
        t = records_to_sparse(r, s->nnz, m, dims, s->ngroups, s->has_counts, budget_or(budget, &dflt),
                              tr_budget_mul(s->nnz, REC_BYTES), refusal, error);
        free(r);
        if (!t) return NULL;
    } else {
        need.need[TR_BUDGET_SUPPORT] = max_group(s->offset, s->ngroups);
        need.need[TR_BUDGET_BYTES] = s->bytes;
        if ((rc = check_one(budget, &need, refusal))) { set_error(error, rc); return NULL; }
        t = tensor_alloc(TR_TENSOR_FACTORED, m, dims, s->ngroups, 0, s->nterms, 0);
        if (!t) { set_error(error, TR_MATH_MEMORY); return NULL; }
        memcpy(t->offset, s->offset, (size_t)(s->ngroups + 1) * 8);
        memcpy(t->weight, s->weight, (size_t)s->nterms * 16);
        uint64_t foff[TR_TENSOR_MAX_ORDER], acc = 0;
        for (int32_t k = 0; k < m; k++) { foff[k] = acc; acc += s->dims[k]; }
        for (uint64_t j = 0; j < s->nterms; j++) {
            int64_t *o = t->factor + j * t->fdim;
            for (int32_t k = 0; k < m; k++) {
                memcpy(o, s->factor + j * s->fdim + foff[perm[k]], (size_t)dims[k] * 8);
                o += dims[k];
            }
        }
    }
    copy_meta(t, s);
    for (int32_t k = 0; k < m; k++) t->axis[k] = s->axis[perm[k]];
    t->bound = s->bound; t->strategy = TR_TENSOR_STRATEGY_DERIVED;
    refresh_symmetric(t);
    set_error(error, TR_MATH_OK);
    return t;
}

/* ------------------------------------------------------------------ outer product, ungroup, values */
static tr_tensor *as_repr(const tr_tensor *s, int32_t repr, const tr_math_budget *budget, tr_math_refusal *refusal,
                          int32_t *error, int *owned) {
    *owned = 0;
    if (s->repr == repr) return (tr_tensor *)s;
    *owned = 1;
    return tr_tensor_convert(s, repr, budget, refusal, error);
}
tr_tensor *tr_tensor_outer(const tr_tensor *a, const tr_tensor *b, int32_t repr, const tr_math_budget *budget,
                           tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    if (!tensor_ok(a) || !tensor_ok(b) || a->ngroups != b->ngroups ||
        (repr != TR_TENSOR_DENSE && repr != TR_TENSOR_SPARSE && repr != TR_TENSOR_FACTORED)) {
        set_error(error, TR_MATH_INVALID); return NULL;
    }
    if (a->has_counts || b->has_counts) { set_error(error, TR_MATH_DOMAIN_ERROR); return NULL; }
    const int32_t m = a->order + b->order;
    if (m > TR_TENSOR_MAX_ORDER) { set_error(error, TR_MATH_UNSUPPORTED); return NULL; }
    if (tr_tensor_satmul(a->bound, b->bound, LIMIT) > LIMIT) { set_error(error, TR_MATH_OVERFLOW); return NULL; }
    uint64_t dims[TR_TENSOR_MAX_ORDER];
    for (int32_t k = 0; k < a->order; k++) dims[k] = a->dims[k];
    for (int32_t k = 0; k < b->order; k++) dims[a->order + k] = b->dims[k];
    if (tr_tensor_cells(dims, m) == TR_BUDGET_UNKNOWN) { set_error(error, TR_MATH_UNSUPPORTED); return NULL; }
    int oa, ob;
    tr_tensor *A = as_repr(a, repr, budget, refusal, error, &oa);
    if (!A) return NULL;
    tr_tensor *B = as_repr(b, repr, budget, refusal, error, &ob);
    if (!B) { if (oa) tensor_release(A); return NULL; }
    const uint64_t G = a->ngroups;
    tr_math_need need;
    need_zero(&need);
    need.need[TR_BUDGET_DEGREE] = (uint64_t)m;
    tr_tensor *t = NULL;
    int32_t rc = 0;
    if (repr == TR_TENSOR_DENSE) {
        uint64_t cells = tr_budget_mul(A->cells, B->cells);
        need.need[TR_BUDGET_TENSOR] = tr_budget_mul(G, cells);
        need.need[TR_BUDGET_BYTES] = object_bytes(TR_TENSOR_DENSE, G, cells, 0, m, 0, 0, 0);
        if (!(rc = check_one(budget, &need, refusal)) && !(t = tensor_alloc(TR_TENSOR_DENSE, m, dims, G, 0, 0, 0))) rc = TR_MATH_MEMORY;
        for (uint64_t g = 0; !rc && g < G; g++)
            for (uint64_t i = 0; i < A->cells; i++) {
                __int128 x = get128(A->value[g * A->cells + i]);
                for (uint64_t j = 0; j < B->cells; j++)
                    t->value[g * cells + i * B->cells + j] = set128(x * get128(B->value[g * B->cells + j]));
            }
    } else if (repr == TR_TENSOR_SPARSE) {
        uint64_t total = 0, largest = 0;
        for (uint64_t g = 0; g < G; g++) {
            uint64_t n = tr_budget_mul(A->offset[g + 1] - A->offset[g], B->offset[g + 1] - B->offset[g]);
            total = tr_budget_add(total, n);
            if (n > largest) largest = n;
        }
        need.need[TR_BUDGET_SUPPORT] = largest;
        need.need[TR_BUDGET_BYTES] = object_bytes(TR_TENSOR_SPARSE, G, 0, total, m, 0, 0, 0);
        if (!(rc = check_one(budget, &need, refusal)) && !(t = tensor_alloc(TR_TENSOR_SPARSE, m, dims, G, total, 0, 0))) rc = TR_MATH_MEMORY;
        uint64_t o = 0;
        for (uint64_t g = 0; !rc && g < G; g++) {
            for (uint64_t i = A->offset[g]; i < A->offset[g + 1]; i++)
                for (uint64_t j = B->offset[g]; j < B->offset[g + 1]; j++) {
                    uint32_t *c = t->coord + o * (uint64_t)m;
                    memcpy(c, A->coord + i * (uint64_t)A->order, (size_t)A->order * 4);
                    memcpy(c + A->order, B->coord + j * (uint64_t)B->order, (size_t)B->order * 4);
                    t->value[o++] = set128(get128(A->value[i]) * get128(B->value[j]));
                }
            t->offset[g + 1] = o;
        }
    } else {
        uint64_t total = 0, largest = 0;
        for (uint64_t g = 0; g < G; g++) {
            uint64_t n = tr_budget_mul(A->offset[g + 1] - A->offset[g], B->offset[g + 1] - B->offset[g]);
            total = tr_budget_add(total, n);
            if (n > largest) largest = n;
        }
        need.need[TR_BUDGET_SUPPORT] = largest;
        need.need[TR_BUDGET_BYTES] = object_bytes(TR_TENSOR_FACTORED, G, 0, 0, m, 0, total, A->fdim + B->fdim);
        if (!(rc = check_one(budget, &need, refusal)) && !(t = tensor_alloc(TR_TENSOR_FACTORED, m, dims, G, 0, total, 0))) rc = TR_MATH_MEMORY;
        uint64_t o = 0;
        for (uint64_t g = 0; !rc && g < G; g++) {
            for (uint64_t i = A->offset[g]; i < A->offset[g + 1] && !rc; i++)
                for (uint64_t j = B->offset[g]; j < B->offset[g + 1]; j++) {
                    __int128 w;
                    if (!mul_ok(get128(A->weight[i]), get128(B->weight[j]), &w)) { rc = TR_MATH_OVERFLOW; break; }
                    t->weight[o] = set128(w);
                    memcpy(t->factor + o * t->fdim, A->factor + i * A->fdim, (size_t)A->fdim * 8);
                    memcpy(t->factor + o * t->fdim + A->fdim, B->factor + j * B->fdim, (size_t)B->fdim * 8);
                    o++;
                }
            t->offset[g + 1] = o;
        }
        if (!rc) { t->bound = factored_bound(t); if (t->bound > LIMIT) rc = TR_MATH_OVERFLOW; }
    }
    if (!rc && repr != TR_TENSOR_FACTORED) t->bound = scan_bound(t);
    if (!rc) {
        for (int32_t k = 0; k < a->order; k++) t->axis[k] = a->axis[k];
        for (int32_t k = 0; k < b->order; k++) t->axis[a->order + k] = b->axis[k];
        t->kind = TR_TENSOR_GENERIC; t->strategy = TR_TENSOR_STRATEGY_DERIVED;
    }
    if (oa) tensor_release(A);
    if (ob) tensor_release(B);
    if (rc) { if (t) tensor_release(t); set_error(error, rc); return NULL; }
    set_error(error, TR_MATH_OK);
    return t;
}

tr_tensor *tr_tensor_ungroup(const tr_tensor *s, const tr_math_budget *budget, tr_math_refusal *refusal,
                             int32_t *error) {
    tr_budget_none(refusal);
    if (!tensor_ok(s)) { set_error(error, TR_MATH_INVALID); return NULL; }
    const int32_t m = s->order + 1;
    if (m > TR_TENSOR_MAX_ORDER || s->ngroups > TR_TENSOR_MAX_DIM) { set_error(error, TR_MATH_UNSUPPORTED); return NULL; }
    uint64_t dims[TR_TENSOR_MAX_ORDER];
    dims[0] = s->ngroups;
    for (int32_t k = 0; k < s->order; k++) dims[k + 1] = s->dims[k];
    if (tr_tensor_cells(dims, m) == TR_BUDGET_UNKNOWN) { set_error(error, TR_MATH_UNSUPPORTED); return NULL; }
    uint64_t fdim = s->fdim + s->ngroups;
    tr_math_need need;
    need_zero(&need);
    need.need[TR_BUDGET_DEGREE] = (uint64_t)m;
    if (s->repr == TR_TENSOR_DENSE) need.need[TR_BUDGET_TENSOR] = tr_budget_mul(s->ngroups, s->cells);
    else need.need[TR_BUDGET_SUPPORT] = s->repr == TR_TENSOR_SPARSE ? s->nnz : s->nterms;
    need.need[TR_BUDGET_BYTES] = object_bytes(s->repr, 1, tr_budget_mul(s->ngroups, s->cells), s->nnz, m, s->has_counts,
                                              s->nterms, fdim);
    int32_t rc = check_one(budget, &need, refusal);
    if (rc) { set_error(error, rc); return NULL; }
    tr_tensor *t = tensor_alloc(s->repr, m, dims, 1, s->nnz, s->nterms, s->has_counts);
    if (!t) { set_error(error, TR_MATH_MEMORY); return NULL; }
    if (s->repr == TR_TENSOR_DENSE) {
        memcpy(t->value, s->value, (size_t)s->nnz * 16);
        if (s->has_counts) memcpy(t->count, s->count, (size_t)s->nnz * 8);
    } else if (s->repr == TR_TENSOR_SPARSE) {
        t->offset[1] = s->nnz;
        for (uint64_t g = 0; g < s->ngroups; g++)
            for (uint64_t e = s->offset[g]; e < s->offset[g + 1]; e++) {
                t->coord[e * (uint64_t)m] = (uint32_t)g;
                memcpy(t->coord + e * (uint64_t)m + 1, s->coord + e * (uint64_t)s->order, (size_t)s->order * 4);
            }
        memcpy(t->value, s->value, (size_t)s->nnz * 16);
        if (s->has_counts) memcpy(t->count, s->count, (size_t)s->nnz * 8);
    } else {
        t->offset[1] = s->nterms;
        memcpy(t->weight, s->weight, (size_t)s->nterms * 16);
        for (uint64_t g = 0; g < s->ngroups; g++)
            for (uint64_t j = s->offset[g]; j < s->offset[g + 1]; j++) {
                int64_t *f = t->factor + j * fdim;
                f[g] = 1;
                memcpy(f + s->ngroups, s->factor + j * s->fdim, (size_t)s->fdim * 8);
            }
    }
    t->kind = s->kind == TR_TENSOR_COUNT ? TR_TENSOR_COUNT : TR_TENSOR_GENERIC;
    t->policy = s->policy; t->weighted = s->weighted; t->wcol = s->wcol; t->wscale = s->wscale;
    t->axis[0].kind = TR_TENSOR_AXIS_GROUP; t->axis[0].dim = s->ngroups; t->axis[0].col = -1;
    for (int32_t k = 0; k < s->order; k++) t->axis[k + 1] = s->axis[k];
    t->bound = s->bound; t->strategy = TR_TENSOR_STRATEGY_DERIVED;
    set_error(error, TR_MATH_OK);
    return t;
}

tr_tensor *tr_tensor_values(const tr_tensor *s, const tr_math_budget *budget, tr_math_refusal *refusal,
                            int32_t *error) {
    tr_budget_none(refusal);
    if (!tensor_ok(s)) { set_error(error, TR_MATH_INVALID); return NULL; }
    tr_tensor *t = NULL;
    if (s->repr == TR_TENSOR_SPARSE && s->has_counts) {      /* entries kept only for their counts leave */
        uint64_t *per = zalloc((s->ngroups + 1) * 8);
        if (!per) { set_error(error, TR_MATH_MEMORY); return NULL; }
        for (uint64_t g = 0; g < s->ngroups; g++) {
            per[g + 1] = per[g];
            for (uint64_t e = s->offset[g]; e < s->offset[g + 1]; e++) per[g + 1] += get128(s->value[e]) != 0;
        }
        tr_math_need need;
        need_zero(&need);
        need.need[TR_BUDGET_SUPPORT] = max_group(per, s->ngroups);
        need.need[TR_BUDGET_BYTES] = object_bytes(TR_TENSOR_SPARSE, s->ngroups, 0, per[s->ngroups], s->order, 0, 0, 0);
        int32_t rc = check_one(budget, &need, refusal);
        if (rc) { free(per); set_error(error, rc); return NULL; }
        t = tensor_alloc(TR_TENSOR_SPARSE, s->order, s->dims, s->ngroups, per[s->ngroups], 0, 0);
        if (!t) { free(per); set_error(error, TR_MATH_MEMORY); return NULL; }
        memcpy(t->offset, per, (size_t)(s->ngroups + 1) * 8);
        free(per);
        uint64_t o = 0;
        for (uint64_t e = 0; e < s->nnz; e++) {
            if (!get128(s->value[e])) continue;
            memcpy(t->coord + o * (uint64_t)s->order, s->coord + e * (uint64_t)s->order, (size_t)s->order * 4);
            t->value[o++] = s->value[e];
        }
    } else {
        tr_math_need need;
        need_zero(&need);
        need.need[TR_BUDGET_BYTES] = s->bytes;
        int32_t rc = check_one(budget, &need, refusal);
        if (rc) { set_error(error, rc); return NULL; }
        t = tensor_alloc(s->repr, s->order, s->dims, s->ngroups, s->nnz, s->nterms, 0);
        if (!t) { set_error(error, TR_MATH_MEMORY); return NULL; }
        if (s->repr == TR_TENSOR_FACTORED) {
            memcpy(t->offset, s->offset, (size_t)(s->ngroups + 1) * 8);
            memcpy(t->weight, s->weight, (size_t)s->nterms * 16);
            memcpy(t->factor, s->factor, (size_t)(s->nterms * s->fdim * 8));
        } else {
            memcpy(t->value, s->value, (size_t)s->nnz * 16);
            if (s->repr == TR_TENSOR_SPARSE) {
                memcpy(t->offset, s->offset, (size_t)(s->ngroups + 1) * 8);
                memcpy(t->coord, s->coord, (size_t)(s->nnz * (uint64_t)s->order * 4));
            }
        }
    }
    copy_meta(t, s);
    t->kind = TR_TENSOR_GENERIC; t->policy = 0; t->symmetric = 0;
    t->bound = s->bound; t->strategy = TR_TENSOR_STRATEGY_DERIVED;
    set_error(error, TR_MATH_OK);
    return t;
}

/* ------------------------------------------------------------------ reading */
int32_t tr_tensor_get(const tr_tensor *t, uint64_t g, const uint32_t *coords, tr_math_i128 *value, uint64_t *count) {
    if (!tensor_ok(t) || g >= t->ngroups || (t->order && !coords) || !value) return TR_MATH_INVALID;
    for (int32_t k = 0; k < t->order; k++) if (coords[k] >= t->dims[k]) return TR_MATH_INVALID;
    if (count) *count = 0;
    *value = set128(0);
    if (t->repr == TR_TENSOR_DENSE) {
        uint64_t i = g * t->cells + linear(t, coords);
        *value = t->value[i];
        if (count && t->has_counts) *count = t->count[i];
        return TR_MATH_OK;
    }
    if (t->repr == TR_TENSOR_SPARSE) {
        uint64_t lo = t->offset[g], hi = t->offset[g + 1];
        while (lo < hi) {
            uint64_t mid = lo + (hi - lo) / 2;
            int c = coord_cmp(t->coord + mid * (uint64_t)t->order, coords, t->order);
            if (c == 0) { *value = t->value[mid]; if (count && t->has_counts) *count = t->count[mid]; return TR_MATH_OK; }
            if (c < 0) lo = mid + 1; else hi = mid;
        }
        return TR_MATH_OK;
    }
    __int128 sum = 0;
    for (uint64_t j = t->offset[g]; j < t->offset[g + 1]; j++) {
        __int128 p = get128(t->weight[j]);
        const int64_t *f = t->factor + j * t->fdim;
        for (int32_t k = 0; k < t->order && p; k++) { p *= f[coords[k]]; f += t->dims[k]; }
        sum += p;                                        /* within the factored bound */
    }
    *value = set128(sum);
    return TR_MATH_OK;
}

static int meta_equal(const tr_tensor *a, const tr_tensor *b) {
    if (a->kind != b->kind || a->order != b->order || a->policy != b->policy || a->has_counts != b->has_counts ||
        a->ngroups != b->ngroups || a->weighted != b->weighted || a->scoped != b->scoped) return 0;
    for (int32_t k = 0; k < a->order; k++)
        if (a->dims[k] != b->dims[k] || memcmp(&a->axis[k], &b->axis[k], sizeof a->axis[k])) return 0;
    return !memcmp(a->rows, b->rows, (size_t)a->ngroups * 8) && !memcmp(a->used, b->used, (size_t)a->ngroups * 8);
}
int32_t tr_tensor_equal(const tr_tensor *a, const tr_tensor *b) {
    if (!tensor_ok(a) || !tensor_ok(b)) return TR_MATH_INVALID;
    if (a->repr != b->repr || !meta_equal(a, b) || a->nnz != b->nnz || a->nterms != b->nterms) return 0;
    if (a->repr == TR_TENSOR_FACTORED)
        return !memcmp(a->offset, b->offset, (size_t)(a->ngroups + 1) * 8) &&
               !memcmp(a->weight, b->weight, (size_t)a->nterms * 16) &&
               !memcmp(a->factor, b->factor, (size_t)(a->nterms * a->fdim * 8));
    if (memcmp(a->value, b->value, (size_t)a->nnz * 16)) return 0;
    if (a->has_counts && memcmp(a->count, b->count, (size_t)a->nnz * 8)) return 0;
    if (a->repr == TR_TENSOR_SPARSE)
        return !memcmp(a->offset, b->offset, (size_t)(a->ngroups + 1) * 8) &&
               !memcmp(a->coord, b->coord, (size_t)(a->nnz * (uint64_t)a->order * 4));
    return 1;
}
int32_t tr_tensor_same(const tr_tensor *a, const tr_tensor *b) {
    if (!tensor_ok(a) || !tensor_ok(b)) return TR_MATH_INVALID;
    if (!meta_equal(a, b)) return 0;
    if (a->repr == TR_TENSOR_DENSE && b->repr == TR_TENSOR_DENSE) {
        if (memcmp(a->value, b->value, (size_t)a->nnz * 16)) return 0;
        return !a->has_counts || !memcmp(a->count, b->count, (size_t)a->nnz * 8);
    }
    int32_t err = 0;
    tr_tensor *sa = a->repr == TR_TENSOR_SPARSE ? (tr_tensor *)a : tr_tensor_convert(a, TR_TENSOR_SPARSE, NULL, NULL, &err);
    if (!sa) return err ? err : TR_MATH_MEMORY;
    tr_tensor *sb = b->repr == TR_TENSOR_SPARSE ? (tr_tensor *)b : tr_tensor_convert(b, TR_TENSOR_SPARSE, NULL, NULL, &err);
    if (!sb) { if (sa != a) tensor_release(sa); return err ? err : TR_MATH_MEMORY; }
    int32_t same = sa->nnz == sb->nnz && !memcmp(sa->offset, sb->offset, (size_t)(sa->ngroups + 1) * 8) &&
                   !memcmp(sa->coord, sb->coord, (size_t)(sa->nnz * (uint64_t)sa->order * 4)) &&
                   !memcmp(sa->value, sb->value, (size_t)sa->nnz * 16) &&
                   (!sa->has_counts || !memcmp(sa->count, sb->count, (size_t)sa->nnz * 8));
    if (sa != a) tensor_release(sa);
    if (sb != b) tensor_release(sb);
    return same;
}
int32_t tr_tensor_get_info(const tr_tensor *t, tr_tensor_info *out) {
    if (!tensor_ok(t) || !out) return TR_MATH_INVALID;
    memset(out, 0, sizeof *out);
    out->ngroups = t->ngroups; out->cells = t->cells; out->nnz = t->nnz; out->nterms = t->nterms; out->bytes = t->bytes;
    for (uint64_t g = 0; g < t->ngroups; g++) { out->present += t->rows[g] > 0; out->rows += t->rows[g]; out->used += t->used[g]; }
    for (int32_t k = 0; k < t->order; k++) out->dims[k] = t->dims[k];
    out->fdim = t->fdim;
    out->bound = set128((__int128)t->bound);
    out->digest[0] = t->digest[0]; out->digest[1] = t->digest[1];
    out->kind = t->kind; out->repr = t->repr; out->order = t->order; out->policy = t->policy;
    out->has_counts = t->has_counts; out->weighted = t->weighted; out->scoped = t->scoped; out->strategy = t->strategy;
    out->symmetric = t->symmetric; out->weight = t->wcol; out->wscale = t->wscale;
    return TR_MATH_OK;
}
int32_t tr_tensor_get_axis(const tr_tensor *t, int32_t k, tr_tensor_axis *out) {
    if (!tensor_ok(t) || !out || k < 0 || k >= t->order) return TR_MATH_INVALID;
    *out = t->axis[k];
    return TR_MATH_OK;
}
int32_t tr_tensor_get_states(const tr_tensor *t, tr_tensor_states *out) {
    if (!tensor_ok(t) || !out) return TR_MATH_INVALID;
    memset(out, 0, sizeof *out);
    out->ngroups = t->ngroups; out->cells = t->cells; out->nnz = t->nnz; out->capacity = t->nnz;
    out->order = t->order; out->repr = t->repr; out->has_counts = t->has_counts;
    for (int32_t k = 0; k < t->order; k++) out->dims[k] = t->dims[k];
    out->rows = t->rows; out->used = t->used; out->offset = t->offset; out->coord = t->coord; out->value = t->value;
    out->count = t->count; out->nterms = t->nterms; out->fdim = t->fdim; out->weight = t->weight; out->factor = t->factor;
    return TR_MATH_OK;
}

/* ================================================================== TENSOR-002: networks */
_Static_assert(TR_TENSOR_PAIR_BYTES == sizeof(rec_t), "the sparse kernel's record");
struct tr_tensor_net_s {
    uint32_t magic;
    int32_t n, noutput, nlabels;
    const tr_tensor *t[TR_TENSOR_NET_MAX];       /* the operands (an ungrouped copy for a grouped tensor) */
    tr_tensor *owned[TR_TENSOR_NET_MAX];
    int32_t nl[TR_TENSOR_NET_MAX];
    int8_t lab[TR_TENSOR_NET_MAX][TR_TENSOR_MAX_ORDER];
    int8_t out[TR_TENSOR_MAX_LABELS];
    uint32_t outmask, labmask;
    uint64_t ldim[TR_TENSOR_MAX_LABELS];
    uint64_t nnz[TR_TENSOR_NET_MAX];             /* entries of the operand's sparse form (factored: an upper bound) */
    u128 bound[TR_TENSOR_NET_MAX];
    uint64_t bytes;
};
static int net_ok(const tr_tensor_net *n) { return n && n->magic == NET_MAGIC; }
void tr_tensor_net_free(tr_tensor_net *n) {
    if (!net_ok(n)) return;
    for (int32_t i = 0; i < n->n; i++) tensor_release(n->owned[i]);
    n->magic = 0;
    free(n);
}
static uint64_t nonzeros(const tr_tensor *t) {
    if (t->repr == TR_TENSOR_SPARSE) return t->nnz;
    if (t->repr == TR_TENSOR_DENSE) {
        uint64_t n = 0;
        for (uint64_t i = 0; i < t->nnz; i++) n += get128(t->value[i]) != 0;
        return n;
    }
    uint64_t e = 0;
    for (uint64_t j = 0; j < t->nterms; j++) e = tr_budget_add(e, term_emissions(t, j));
    uint64_t c = tr_budget_mul(t->ngroups, t->cells);
    return e < c ? e : c;
}
tr_tensor_net *tr_tensor_net_create(const tr_tensor *const *tensors, const tr_tensor_net_spec *spec,
                                    const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    if (!tensors || !spec || spec->ntensors < 1 || spec->noutput < 0 || spec->noutput > TR_TENSOR_MAX_LABELS) {
        set_error(error, spec && spec->ntensors > TR_TENSOR_NET_MAX ? TR_MATH_UNSUPPORTED : TR_MATH_INVALID); return NULL;
    }
    if (spec->ntensors > TR_TENSOR_NET_MAX) { set_error(error, TR_MATH_UNSUPPORTED); return NULL; }
    uint64_t ldim[TR_TENSOR_MAX_LABELS];
    uint32_t labmask = 0, outmask = 0;
    uint64_t copies = 0;
    for (int32_t i = 0; i < spec->ntensors; i++) {
        const tr_tensor *t = tensors[i];
        if (!tensor_ok(t)) { set_error(error, TR_MATH_INVALID); return NULL; }
        if (t->has_counts) { set_error(error, TR_MATH_DOMAIN_ERROR); return NULL; }
        int32_t nl = spec->nlabels[i], grouped = nl == t->order + 1;
        if (nl != t->order && !grouped) { set_error(error, TR_MATH_INVALID); return NULL; }
        if (!grouped && t->ngroups != 1) { set_error(error, TR_MATH_INVALID); return NULL; }
        if (grouped && nl > TR_TENSOR_MAX_ORDER) { set_error(error, TR_MATH_UNSUPPORTED); return NULL; }
        uint32_t mine = 0;
        for (int32_t k = 0; k < nl; k++) {
            int32_t l = spec->labels[i][k];
            if (l < 0 || l >= TR_TENSOR_MAX_LABELS) { set_error(error, TR_MATH_INVALID); return NULL; }
            if ((mine >> l) & 1) { set_error(error, TR_MATH_UNSUPPORTED); return NULL; }   /* a trace */
            mine |= UINT32_C(1) << l;
            uint64_t d = grouped ? (k == 0 ? t->ngroups : t->dims[k - 1]) : t->dims[k];
            if ((labmask >> l) & 1) { if (ldim[l] != d) { set_error(error, TR_MATH_INVALID); return NULL; } }
            else { labmask |= UINT32_C(1) << l; ldim[l] = d; }
        }
        if (grouped) copies = tr_budget_add(copies, object_bytes(t->repr, 1, tr_budget_mul(t->ngroups, t->cells), t->nnz,
                                                                 nl, 0, t->nterms, t->fdim + t->ngroups));
    }
    for (int32_t j = 0; j < spec->noutput; j++) {
        int32_t l = spec->output[j];
        if (l < 0 || l >= TR_TENSOR_MAX_LABELS || !((labmask >> l) & 1) || ((outmask >> l) & 1)) {
            set_error(error, TR_MATH_INVALID); return NULL;
        }
        outmask |= UINT32_C(1) << l;
    }
    tr_math_need need;
    need_zero(&need);
    need.need[TR_BUDGET_VARIABLES] = (uint64_t)popc(labmask);
    need.need[TR_BUDGET_BYTES] = tr_budget_add(sizeof(tr_tensor_net), copies);
    int32_t rc = check_one(budget, &need, refusal);
    if (rc) { set_error(error, rc); return NULL; }
    tr_tensor_net *n = zalloc(sizeof *n);
    if (!n) { set_error(error, TR_MATH_MEMORY); return NULL; }
    n->n = spec->ntensors; n->noutput = spec->noutput; n->labmask = labmask; n->outmask = outmask;
    n->nlabels = popc(labmask); n->bytes = need.need[TR_BUDGET_BYTES];
    memcpy(n->ldim, ldim, sizeof ldim);
    for (int32_t j = 0; j < spec->noutput; j++) n->out[j] = (int8_t)spec->output[j];
    for (int32_t i = 0; i < spec->ntensors; i++) {
        const tr_tensor *t = tensors[i];
        n->nl[i] = spec->nlabels[i];
        for (int32_t k = 0; k < n->nl[i]; k++) n->lab[i][k] = (int8_t)spec->labels[i][k];
        if (n->nl[i] == t->order + 1) {
            n->owned[i] = tr_tensor_ungroup(t, budget, refusal, error);
            if (!n->owned[i]) { tr_tensor_net_free((n->magic = NET_MAGIC, n)); return NULL; }
            n->t[i] = n->owned[i];
        } else n->t[i] = t;
        n->nnz[i] = nonzeros(n->t[i]);
        n->bound[i] = n->t[i]->bound;
    }
    n->magic = NET_MAGIC;
    set_error(error, TR_MATH_OK);
    return n;
}
tr_tensor *tr_tensor_net_input_dense(const tr_tensor_net *n, int32_t i, const tr_math_budget *budget,
                                     tr_math_refusal *refusal, int32_t *error) {
    if (!net_ok(n) || i < 0 || i >= n->n) { set_error(error, TR_MATH_INVALID); return NULL; }
    return tr_tensor_convert(n->t[i], TR_TENSOR_DENSE, budget, refusal, error);
}

/* ------------------------------------------------------------------ orders */
typedef struct { int32_t nsteps; int32_t a[TR_TENSOR_MAX_STEPS], b[TR_TENSOR_MAX_STEPS]; } order_t;
typedef struct { uint32_t mask; int32_t n; int8_t lab[TR_TENSOR_MAX_LABELS]; } slot_t;
static uint64_t mask_cells(const tr_tensor_net *n, uint32_t mask) {
    uint64_t c = 1;
    for (int32_t l = 0; l < TR_TENSOR_MAX_LABELS; l++) if ((mask >> l) & 1) c = tr_budget_mul(c, n->ldim[l]);
    return c;
}
static void input_slots(const tr_tensor_net *n, slot_t *s) {
    for (int32_t i = 0; i < n->n; i++) {
        s[i].mask = 0; s[i].n = n->nl[i];
        for (int32_t k = 0; k < n->nl[i]; k++) { s[i].lab[k] = n->lab[i][k]; s[i].mask |= UINT32_C(1) << n->lab[i][k]; }
    }
}
/* the result slot of contracting a and b (b < 0: a reduction) among the live slots; last: the output's order */
static void result_slot(const tr_tensor_net *n, const slot_t *slots, const int32_t *live, int32_t nlive, int32_t a,
                        int32_t b, slot_t *r, uint32_t *summed) {
    uint32_t both = slots[a].mask | (b >= 0 ? slots[b].mask : 0), needed = n->outmask;
    for (int32_t i = 0; i < nlive; i++) if (live[i] != a && live[i] != b) needed |= slots[live[i]].mask;
    int last = nlive <= (b >= 0 ? 2 : 1);
    r->n = 0; r->mask = both & needed;
    if (last) {
        for (int32_t j = 0; j < n->noutput; j++) r->lab[r->n++] = n->out[j];
        r->mask = n->outmask;
    } else {
        for (int32_t l = 0; l < TR_TENSOR_MAX_LABELS; l++) if ((r->mask >> l) & 1) r->lab[r->n++] = (int8_t)l;
    }
    *summed = both & ~r->mask;
}
static uint64_t binom2(uint64_t k) { return k * (k - 1) / 2; }
static uint64_t total_orders(int32_t n) {
    uint64_t t = 1;
    for (int32_t k = 2; k <= n; k++) t = tr_budget_mul(t, binom2((uint64_t)k));
    return t;
}
typedef struct { order_t *o; int32_t n, cap, truncated; } order_list;
static int order_same(const order_t *x, const order_t *y) {
    if (x->nsteps != y->nsteps) return 0;
    for (int32_t s = 0; s < x->nsteps; s++) if (x->a[s] != y->a[s] || x->b[s] != y->b[s]) return 0;
    return 1;
}
static void order_add(order_list *L, const order_t *o) {
    for (int32_t i = 0; i < L->n; i++) if (order_same(&L->o[i], o)) return;
    if (L->n >= L->cap) { L->truncated = 1; return; }
    L->o[L->n++] = *o;
}
static void dfs_orders(const tr_tensor_net *n, slot_t *slots, int32_t *live, int32_t nlive, order_t *cur, order_list *L) {
    if (L->truncated) return;
    if (nlive == 1) { order_add(L, cur); return; }
    int32_t step = cur->nsteps;
    for (int32_t i = 0; i < nlive && !L->truncated; i++)
        for (int32_t j = i + 1; j < nlive && !L->truncated; j++) {
            int32_t a = live[i], b = live[j], next[TR_TENSOR_NET_MAX], m = 0;
            uint32_t summed;
            result_slot(n, slots, live, nlive, a, b, &slots[n->n + step], &summed);
            for (int32_t k = 0; k < nlive; k++) if (k != i && k != j) next[m++] = live[k];
            next[m++] = n->n + step;
            cur->a[step] = a; cur->b[step] = b; cur->nsteps = step + 1;
            dfs_orders(n, slots, next, m, cur, L);
            cur->nsteps = step;
        }
}
/* greedy: at each step the pair with the smallest result (by = 0) or the fewest products (by = 1) */
static void greedy_order(const tr_tensor_net *n, int by, order_t *o) {
    slot_t slots[2 * TR_TENSOR_NET_MAX];
    input_slots(n, slots);
    int32_t live[TR_TENSOR_NET_MAX], nlive = n->n;
    for (int32_t i = 0; i < n->n; i++) live[i] = i;
    o->nsteps = 0;
    while (nlive > 1) {
        int32_t bi = 0, bj = 1;
        uint64_t best = UINT64_MAX;
        for (int32_t i = 0; i < nlive; i++)
            for (int32_t j = i + 1; j < nlive; j++) {
                slot_t r;
                uint32_t summed;
                result_slot(n, slots, live, nlive, live[i], live[j], &r, &summed);
                uint64_t c = mask_cells(n, r.mask);
                if (by) c = tr_budget_mul(c, mask_cells(n, summed));
                if (c < best) { best = c; bi = i; bj = j; }
            }
        int32_t s = o->nsteps, a = live[bi], b = live[bj], next[TR_TENSOR_NET_MAX], m = 0;
        uint32_t summed;
        result_slot(n, slots, live, nlive, a, b, &slots[n->n + s], &summed);
        for (int32_t k = 0; k < nlive; k++) if (k != bi && k != bj) next[m++] = live[k];
        next[m++] = n->n + s;
        o->a[s] = a; o->b[s] = b; o->nsteps = s + 1;
        memcpy(live, next, sizeof(int32_t) * (size_t)m);
        nlive = m;
    }
}

/* ------------------------------------------------------------------ candidates */
struct tr_tensor_plan_s {
    uint32_t magic;
    int32_t ncand, chosen, code, reason, complete, norders, forced, devices;
    uint64_t total_orders, bytes;
    tr_math_refusal refusal;
    const tr_tensor_net *net;
    order_t *orders;
    tr_tensor_candidate cand[];
};
static int plan_ok(const tr_tensor_plan *p) { return p && p->magic == PLAN_MAGIC; }
void tr_tensor_plan_free(tr_tensor_plan *p) {
    if (!plan_ok(p)) return;
    free(p->orders);
    p->magic = 0;
    free(p);
}
static uint64_t log2_ceil(uint64_t x) { uint64_t l = 0; while (l < 64 && (UINT64_C(1) << l) < x) l++; return l; }
static uint64_t dense_obj(uint64_t cells, int32_t order) { return object_bytes(TR_TENSOR_DENSE, 1, cells, 0, order, 0, 0, 0); }
static uint64_t sparse_obj(uint64_t nnz, int32_t order) { return object_bytes(TR_TENSOR_SPARSE, 1, 0, nnz, order, 0, 0, 0); }
/* element strides of the labels of a slot (row major over its label order) */
static void slot_strides(const tr_tensor_net *n, const slot_t *s, uint64_t *stride) {
    uint64_t c = 1;
    for (int32_t k = s->n - 1; k >= 0; k--) { stride[s->lab[k]] = c; c = tr_budget_mul(c, n->ldim[(int)s->lab[k]]); }
}
static void fill_dense_step(const tr_tensor_net *n, const slot_t *slots, const tr_tensor_step_info *si,
                            tr_tensor_dense_step *d) {
    memset(d, 0, sizeof *d);
    uint64_t sa[TR_TENSOR_MAX_LABELS] = {0}, sb[TR_TENSOR_MAX_LABELS] = {0};
    slot_strides(n, &slots[si->a], sa);
    if (si->b >= 0) slot_strides(n, &slots[si->b], sb);
    d->a = si->a; d->b = si->b; d->nout = si->nout; d->nsum = si->nsum; d->cells = si->cells; d->terms = si->terms;
    for (int32_t k = 0; k < si->nout; k++) {
        int l = si->out[k];
        d->dim_out[k] = n->ldim[l];
        d->sa_out[k] = (slots[si->a].mask >> l) & 1 ? sa[l] : 0;
        d->sb_out[k] = si->b >= 0 && ((slots[si->b].mask >> l) & 1) ? sb[l] : 0;
    }
    for (int32_t k = 0; k < si->nsum; k++) {
        int l = si->sum[k];
        d->dim_sum[k] = n->ldim[l];
        d->sa_sum[k] = (slots[si->a].mask >> l) & 1 ? sa[l] : 0;
        d->sb_sum[k] = si->b >= 0 && ((slots[si->b].mask >> l) & 1) ? sb[l] : 0;
    }
    d->bound = si->bound; d->free_a = si->free_a; d->free_b = si->free_b;
    d->a_cells = mask_cells(n, slots[si->a].mask);
    d->b_cells = si->b >= 0 ? mask_cells(n, slots[si->b].mask) : 0;
}
/* the steps, shapes, bounds, bytes, needs and amounts of one (order, kernel, device) */
static void make_candidate(const tr_tensor_net *n, const order_t *o, int32_t order_index, int32_t kernel,
                           int32_t device, tr_tensor_candidate *c) {
    memset(c, 0, sizeof *c);
    c->kernel = kernel; c->device = device; c->order = order_index;
    tr_reg_candidate *g = &c->reg;
    g->module = 0; g->representation = 0; g->device = device; g->finalize = TR_REG_FINAL_CPU; g->batch = 1;
    g->valid = 1; tr_reg_refusal_none(&g->refusal); tr_budget_none(&g->budget);
    slot_t slots[2 * TR_TENSOR_NET_MAX];
    input_slots(n, slots);
    const int dense = kernel == TR_TENSOR_KERNEL_DENSE;
    uint64_t nnz[2 * TR_TENSOR_NET_MAX], bytes[2 * TR_TENSOR_NET_MAX];
    u128 bound[2 * TR_TENSOR_NET_MAX];
    uint64_t *amount = g->amount, max_tensor = 0, max_support = 0, live = 0, peak = 0;
    int32_t maxdeg = 0;
    for (int32_t i = 0; i < n->n; i++) {
        const tr_tensor *t = n->t[i];
        bound[i] = n->bound[i];
        nnz[i] = dense ? t->cells : n->nnz[i];
        bytes[i] = 0;
        if (t->order > maxdeg) maxdeg = t->order;
        int convert = dense ? t->repr != TR_TENSOR_DENSE : t->repr != TR_TENSOR_SPARSE;
        if (dense && t->cells > max_tensor) max_tensor = t->cells;
        if (!dense && nnz[i] > max_support) max_support = nnz[i];
        if (!convert) continue;
        c->converted++;
        bytes[i] = dense ? dense_obj(t->cells, t->order) : sparse_obj(nnz[i], t->order);
        c->input_bytes = tr_budget_add(c->input_bytes, bytes[i]);
        uint64_t work = t->repr == TR_TENSOR_FACTORED ? (dense ? n->nnz[i] : nnz[i]) : (t->repr == TR_TENSOR_DENSE ? t->cells : t->nnz);
        amount[TR_COST_SCAN] = tr_budget_add(amount[TR_COST_SCAN], work);
        amount[TR_COST_CONSTRUCT] = tr_budget_add(amount[TR_COST_CONSTRUCT], dense ? t->cells : nnz[i]);
        if (t->repr == TR_TENSOR_FACTORED) amount[TR_COST_CHAIN] = tr_budget_add(amount[TR_COST_CHAIN], work);
    }
    live = c->input_bytes; peak = live;
    int32_t live_slots[TR_TENSOR_NET_MAX], nlive = n->n;
    for (int32_t i = 0; i < n->n; i++) live_slots[i] = i;
    int32_t steps = n->n == 1 ? 1 : o->nsteps;
    c->nsteps = steps;
    for (int32_t s = 0; s < steps; s++) {
        tr_tensor_step_info *si = &c->step[s];
        int32_t a = n->n == 1 ? 0 : o->a[s], b = n->n == 1 ? -1 : o->b[s];
        slot_t *r = &slots[n->n + s];
        uint32_t summed;
        result_slot(n, slots, live_slots, nlive, a, b, r, &summed);
        si->a = a; si->b = b; si->nout = r->n;
        memcpy(si->out, r->lab, sizeof si->out);
        si->nsum = 0;
        for (int32_t l = 0; l < TR_TENSOR_MAX_LABELS; l++) if ((summed >> l) & 1) si->sum[si->nsum++] = (int8_t)l;
        si->cells = mask_cells(n, r->mask);
        si->terms = mask_cells(n, summed);
        if (r->n > maxdeg) maxdeg = r->n;
        u128 bb = b >= 0 ? bound[b] : 1;
        u128 rb = tr_tensor_satmul(tr_tensor_satmul(si->terms, bound[a], LIMIT), bb, LIMIT);
        if (si->terms == 0 || bound[a] == 0 || bb == 0) rb = 0;
        bound[n->n + s] = rb;
        si->bound = set128((__int128)(rb > LIMIT ? LIMIT : rb));
        if (rb > LIMIT && g->valid) {
            g->valid = 0; g->code = TR_MATH_OVERFLOW;
            tr_reg_refuse(&g->refusal, TR_REG_REFUSE_BOUND, TR_MATH_OVERFLOW, s, TR_OP_CONTRACT, TR_REG_C_INT128, 0, 0, 0);
        }
        if (dense) {
            si->nnz = si->cells;
            si->pairs = tr_budget_mul(si->cells, si->terms);
            si->bytes = dense_obj(si->cells, r->n);
            si->scratch = 0;
            if (si->cells > max_tensor) max_tensor = si->cells;
            amount[TR_COST_SCAN] = tr_budget_add(amount[TR_COST_SCAN], tr_budget_mul(si->pairs, b >= 0 ? 2 : 1));
            amount[TR_COST_CHAIN] = tr_budget_add(amount[TR_COST_CHAIN], si->pairs);
            amount[TR_COST_CONSTRUCT] = tr_budget_add(amount[TR_COST_CONSTRUCT], si->cells);
        } else {
            uint64_t na = nnz[a], nb = b >= 0 ? nnz[b] : 1, pairs;
            if (b >= 0) {
                uint32_t shared = slots[a].mask & slots[b].mask;
                uint64_t x = tr_budget_mul(na, mask_cells(n, slots[b].mask & ~shared));
                uint64_t y = tr_budget_mul(nb, mask_cells(n, slots[a].mask & ~shared));
                pairs = x < y ? x : y;
                uint64_t full = tr_budget_mul(na, nb);
                if (full < pairs) pairs = full;
            } else pairs = na;
            si->pairs = pairs;
            si->nnz = pairs < si->cells ? pairs : si->cells;
            si->bytes = sparse_obj(si->nnz, r->n);
            si->scratch = tr_budget_add(tr_budget_mul(pairs, TR_TENSOR_PAIR_BYTES),
                                        tr_budget_mul(tr_budget_add(na, b >= 0 ? nb : 0), TR_TENSOR_KEY_BYTES));
            if (si->nnz > max_support) max_support = si->nnz;
            amount[TR_COST_SCAN] = tr_budget_add(amount[TR_COST_SCAN], tr_budget_add(na, b >= 0 ? nb : 0));
            amount[TR_COST_CHAIN] = tr_budget_add(amount[TR_COST_CHAIN], pairs);
            amount[TR_COST_CONSTRUCT] = tr_budget_add(amount[TR_COST_CONSTRUCT],
                                                      tr_budget_add(tr_budget_mul(pairs, log2_ceil(pairs) + 1), si->nnz));
        }
        nnz[n->n + s] = si->nnz;
        bytes[n->n + s] = si->bytes;
        live = tr_budget_add(live, tr_budget_add(si->bytes, si->scratch));
        si->live = live;
        if (live > peak) peak = live;
        live -= si->scratch;
        /* every slot is consumed once: an intermediate or a converted input is freed right after its consumer */
        si->free_a = a >= n->n || bytes[a] > 0;
        si->free_b = b >= 0 && (b >= n->n || bytes[b] > 0);
        if (si->free_a) live -= bytes[a];
        if (si->free_b) live -= bytes[b];
        int32_t next[TR_TENSOR_NET_MAX], m = 0;
        for (int32_t k = 0; k < nlive; k++) if (live_slots[k] != a && live_slots[k] != b) next[m++] = live_slots[k];
        next[m++] = n->n + s;
        memcpy(live_slots, next, sizeof(int32_t) * (size_t)m);
        nlive = m;
    }
    tr_tensor_step_info *last = &c->step[steps - 1];
    c->output_cells = last->cells; c->output_nnz = last->nnz; c->output_bytes = last->bytes;
    c->bound = last->bound;
    c->peak_bytes = peak;
    amount[TR_COST_MATERIALIZE] = c->output_bytes;
    amount[TR_COST_FIXED] = (uint64_t)steps;
    tr_math_need *need = &g->need;
    need_zero(need);
    need->need[TR_BUDGET_VARIABLES] = (uint64_t)n->nlabels;
    need->need[TR_BUDGET_DEGREE] = (uint64_t)maxdeg;
    need->need[TR_BUDGET_TENSOR] = max_tensor;
    need->need[TR_BUDGET_SUPPORT] = max_support;
    need->need[TR_BUDGET_OUTPUT] = dense ? c->output_cells : c->output_nnz;
    need->need[TR_BUDGET_BYTES] = peak;
    if (device == TR_REG_ON_CUDA) {
        tr_tensor_program prog;
        memset(&prog, 0, sizeof prog);
        prog.ninputs = n->n; prog.nsteps = steps; prog.noutput = n->noutput; prog.output_cells = c->output_cells;
        uint64_t transfer = 0;
        for (int32_t i = 0; i < n->n; i++) { prog.input_cells[i] = n->t[i]->cells; transfer = tr_budget_add(transfer, tr_budget_mul(n->t[i]->cells, 16)); }
        for (int32_t s = 0; s < steps; s++) fill_dense_step(n, slots, &c->step[s], &prog.step[s]);
        uint64_t dev = tr_tensor_program_device_bytes(&prog);
        need->need[TR_BUDGET_BYTES] = tr_budget_add(dev, c->input_bytes);
        c->peak_bytes = need->need[TR_BUDGET_BYTES];
        amount[TR_COST_TRANSFER] = tr_budget_add(transfer, tr_budget_mul(c->output_cells, 16));
        amount[TR_COST_FIXED] = (uint64_t)steps + 1;
    }
}

int32_t tr_tensor_net_plan(const tr_tensor_net *n, const tr_tensor_plan_options *opt, const tr_math_budget *budget,
                           tr_tensor_plan **out, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!out) return TR_MATH_INVALID;
    *out = NULL;
    if (!net_ok(n)) return TR_MATH_INVALID;
    tr_tensor_plan_options o;
    if (opt) o = *opt;
    else { memset(&o, 0, sizeof o); o.force = -1; }
    if (o.max_orders < 0 || o.devices & ~TR_REG_ON_BOTH || o.kernels & ~(TR_TENSOR_KERNEL_DENSE | TR_TENSOR_KERNEL_SPARSE) ||
        o.reserved) return TR_MATH_INVALID;
    if (!o.devices) o.devices = TR_REG_ON_CPU;
    if (!o.kernels) o.kernels = TR_TENSOR_KERNEL_DENSE | TR_TENSOR_KERNEL_SPARSE;
    double rates[2][TR_COST_COMPONENTS];
    if (o.custom_rates) memcpy(rates, o.rates, sizeof rates);
    else tr_tensor_default_rates(rates);
    for (int d = 0; d < 2; d++)
        for (int k = 0; k < TR_COST_COMPONENTS; k++)
            if (!(rates[d][k] >= 0) || rates[d][k] > 1e300) return TR_MATH_INVALID;
    tr_math_budget dflt;
    budget = budget_or(budget, &dflt);
    uint64_t limit = budget->limit[TR_BUDGET_ITERATIONS];
    int32_t max_orders = o.max_orders ? o.max_orders : TR_TENSOR_PLAN_ORDERS;
    if ((uint64_t)max_orders > limit) max_orders = (int32_t)(limit > INT32_MAX ? INT32_MAX : limit);
    int32_t per_order = ((o.devices & TR_REG_ON_CPU) ? popc((uint32_t)o.kernels) : 0) +
                        ((o.devices & TR_REG_ON_CUDA) && (o.kernels & TR_TENSOR_KERNEL_DENSE) ? 1 : 0);
    if (per_order == 0) return TR_MATH_INVALID;
    if (max_orders > TR_TENSOR_MAX_CANDIDATES / per_order) max_orders = TR_TENSOR_MAX_CANDIDATES / per_order;
    if (max_orders < 1) {
        if (refusal) { refusal->resource = TR_BUDGET_ITERATIONS; refusal->refused = 1; refusal->need = 1; refusal->limit = limit; }
        return TR_MATH_BUDGET;
    }
    uint64_t total = total_orders(n->n);
    int32_t norders_cap = total < (uint64_t)max_orders ? (int32_t)total : max_orders;
    uint64_t plan_bytes = tr_budget_add(sizeof(tr_tensor_plan), tr_budget_mul((uint64_t)norders_cap * (uint64_t)per_order,
                                                                             sizeof(tr_tensor_candidate)));
    plan_bytes = tr_budget_add(plan_bytes, tr_budget_mul((uint64_t)norders_cap, sizeof(order_t)));
    tr_math_need need;
    need_zero(&need);
    need.need[TR_BUDGET_VARIABLES] = (uint64_t)n->nlabels;
    need.need[TR_BUDGET_ITERATIONS] = (uint64_t)norders_cap;
    need.need[TR_BUDGET_BYTES] = plan_bytes;
    int32_t rc = tr_budget_check(budget, &need, refusal);
    if (rc) return rc;
    order_list L = {zalloc((uint64_t)norders_cap * sizeof(order_t)), 0, norders_cap, 0};
    if (!L.o) return TR_MATH_MEMORY;
    int complete = total <= (uint64_t)norders_cap;
    if (n->n == 1) { L.o[0].nsteps = 0; L.n = 1; }
    else {
        if (!complete) {
            order_t g;
            greedy_order(n, 0, &g); order_add(&L, &g);
            greedy_order(n, 1, &g); order_add(&L, &g);
        }
        slot_t slots[2 * TR_TENSOR_NET_MAX];
        input_slots(n, slots);
        int32_t live[TR_TENSOR_NET_MAX];
        for (int32_t i = 0; i < n->n; i++) live[i] = i;
        order_t cur;
        memset(&cur, 0, sizeof cur);
        L.truncated = 0;
        dfs_orders(n, slots, live, n->n, &cur, &L);
    }
    int32_t ncand = L.n * per_order;
    tr_tensor_plan *p = zalloc(sizeof(tr_tensor_plan) + (uint64_t)ncand * sizeof(tr_tensor_candidate));
    if (!p) { free(L.o); return TR_MATH_MEMORY; }
    p->magic = PLAN_MAGIC; p->net = n; p->orders = L.o; p->norders = L.n; p->complete = complete;
    p->total_orders = total; p->bytes = plan_bytes; p->devices = o.devices; p->chosen = -1;
    tr_budget_none(&p->refusal);
    int32_t k = 0;
    for (int32_t i = 0; i < L.n; i++) {
        if ((o.devices & TR_REG_ON_CPU) && (o.kernels & TR_TENSOR_KERNEL_DENSE))
            make_candidate(n, &L.o[i], i, TR_TENSOR_KERNEL_DENSE, TR_REG_ON_CPU, &p->cand[k++]);
        if ((o.devices & TR_REG_ON_CPU) && (o.kernels & TR_TENSOR_KERNEL_SPARSE))
            make_candidate(n, &L.o[i], i, TR_TENSOR_KERNEL_SPARSE, TR_REG_ON_CPU, &p->cand[k++]);
        if ((o.devices & TR_REG_ON_CUDA) && (o.kernels & TR_TENSOR_KERNEL_DENSE))
            make_candidate(n, &L.o[i], i, TR_TENSOR_KERNEL_DENSE, TR_REG_ON_CUDA, &p->cand[k++]);
    }
    p->ncand = k;
    if (o.force >= k) { tr_tensor_plan_free(p); return TR_MATH_INVALID; }
    /* the FND-008 choice: rounds of TR_REG_MAX_CANDIDATES candidates, then one over the rounds' winners */
    double coef[TR_REG_MODULES][2][TR_COST_COMPONENTS];
    memset(coef, 0, sizeof coef);
    memcpy(coef[TR_TENSOR_COST_ROW], rates, sizeof rates);
    tr_reg_candidate chunk[TR_REG_MAX_CANDIDATES], winners[TR_REG_MAX_CANDIDATES];
    int32_t winner_index[TR_REG_MAX_CANDIDATES], nwin = 0, first_budget = -1, first_other = -1;
    tr_planner_evidence *ev = zalloc(sizeof *ev);
    if (!ev) { tr_tensor_plan_free(p); return TR_MATH_MEMORY; }
    for (int32_t base = 0; base < k; base += TR_REG_MAX_CANDIDATES) {
        int32_t m = k - base < TR_REG_MAX_CANDIDATES ? k - base : TR_REG_MAX_CANDIDATES;
        for (int32_t i = 0; i < m; i++) { chunk[i] = p->cand[base + i].reg; chunk[i].module = TR_TENSOR_COST_ROW; }
        tr_planner_choose(chunk, m, budget, &coef[0][0][0], NULL, ev);
        for (int32_t i = 0; i < m; i++) {
            chunk[i].module = 0;
            p->cand[base + i].reg = chunk[i];
            if (chunk[i].status == TR_PLANNER_S_REJECTED) {
                if (chunk[i].reason == TR_PLANNER_R_BUDGET && first_budget < 0) first_budget = base + i;
                if (first_other < 0) first_other = base + i;
            }
        }
        if (ev->chosen >= 0) { winners[nwin] = chunk[ev->chosen]; winner_index[nwin++] = base + ev->chosen; }
    }
    int32_t chosen = -1;
    if (nwin) {
        for (int32_t i = 0; i < nwin; i++) winners[i].module = TR_TENSOR_COST_ROW;
        tr_planner_choose(winners, nwin, budget, &coef[0][0][0], NULL, ev);
        chosen = ev->chosen >= 0 ? winner_index[ev->chosen] : -1;
        for (int32_t i = 0; i < nwin; i++) {
            tr_reg_candidate *x = &p->cand[winner_index[i]].reg;
            x->status = TR_PLANNER_S_ADMISSIBLE; x->reason = TR_PLANNER_R_COSTLIER;
        }
    }
    free(ev);
    int32_t code = TR_MATH_OK, reason = TR_PLANNER_R_CHEAPEST;
    if (o.force >= 0) {
        p->forced = 1;
        tr_reg_candidate *f = &p->cand[o.force].reg;
        if (f->status == TR_PLANNER_S_REJECTED) {
            chosen = -1; reason = TR_PLANNER_R_FORCE_REFUSED;
            code = f->code ? f->code : TR_MATH_UNSUPPORTED;
            if (code == TR_MATH_BUDGET) p->refusal = f->budget;
        } else { chosen = o.force; reason = TR_PLANNER_R_FORCED; }
        for (int32_t i = 0; i < k; i++)
            if (i != chosen && p->cand[i].reg.status != TR_PLANNER_S_REJECTED) p->cand[i].reg.reason = TR_PLANNER_R_NOT_FORCED;
    } else if (chosen < 0) {
        reason = TR_PLANNER_R_NONE_ADMISSIBLE;
        if (first_budget >= 0) { code = TR_MATH_BUDGET; p->refusal = p->cand[first_budget].reg.budget; }
        else code = first_other >= 0 && p->cand[first_other].reg.code ? p->cand[first_other].reg.code : TR_MATH_UNSUPPORTED;
    }
    if (chosen >= 0) { p->cand[chosen].reg.status = TR_PLANNER_S_CHOSEN; p->cand[chosen].reg.reason = reason; }
    p->chosen = chosen; p->code = code; p->reason = reason;
    if (code == TR_MATH_BUDGET && refusal) *refusal = p->refusal;
    *out = p;
    return code;
}
int32_t tr_tensor_plan_get_info(const tr_tensor_plan *p, tr_tensor_plan_info *out) {
    if (!plan_ok(p) || !out) return TR_MATH_INVALID;
    memset(out, 0, sizeof *out);
    out->ncandidates = p->ncand; out->chosen = p->chosen; out->code = p->code; out->reason = p->reason;
    out->complete = p->complete; out->norders = p->norders; out->forced = p->forced; out->devices = p->devices;
    out->total_orders = p->total_orders; out->bytes = p->bytes; out->refusal = p->refusal;
    out->ntensors = p->net->n; out->nlabels = p->net->nlabels;
    return TR_MATH_OK;
}
int32_t tr_tensor_plan_get(const tr_tensor_plan *p, int32_t i, tr_tensor_candidate *out) {
    if (!plan_ok(p) || !out || i < 0 || i >= p->ncand) return TR_MATH_INVALID;
    *out = p->cand[i];
    return TR_MATH_OK;
}

/* ------------------------------------------------------------------ execution */
static int32_t resolve(const tr_tensor_net *n, const tr_tensor_plan *p, int32_t index, const tr_tensor_candidate **c) {
    if (!net_ok(n) || !plan_ok(p) || p->net != n || index < -1 || index >= p->ncand) return TR_MATH_INVALID;
    if (index < 0) { if (p->chosen < 0) return p->code ? p->code : TR_MATH_INVALID; index = p->chosen; }
    *c = &p->cand[index];
    if ((*c)->reg.status == TR_PLANNER_S_REJECTED) return (*c)->reg.code ? (*c)->reg.code : TR_MATH_UNSUPPORTED;
    return TR_MATH_OK;
}
int32_t tr_tensor_net_program(const tr_tensor_net *n, const tr_tensor_plan *p, int32_t index, tr_tensor_program *out) {
    const tr_tensor_candidate *c = NULL;
    if (!out) return TR_MATH_INVALID;
    int32_t rc = resolve(n, p, index, &c);
    if (rc) return rc;
    if (c->kernel != TR_TENSOR_KERNEL_DENSE) return TR_MATH_INVALID;
    memset(out, 0, sizeof *out);
    slot_t slots[2 * TR_TENSOR_NET_MAX];
    input_slots(n, slots);
    out->ninputs = n->n; out->nsteps = c->nsteps; out->noutput = n->noutput; out->output_cells = c->output_cells;
    for (int32_t i = 0; i < n->n; i++) { out->input_cells[i] = n->t[i]->cells; out->input_bound[i] = set128((__int128)n->bound[i]); }
    for (int32_t j = 0; j < n->noutput; j++) out->output_dims[j] = n->ldim[(int)n->out[j]];
    for (int32_t s = 0; s < c->nsteps; s++) {
        const tr_tensor_step_info *si = &c->step[s];
        slot_t *r = &slots[n->n + s];
        r->n = si->nout; r->mask = 0;
        memcpy(r->lab, si->out, sizeof r->lab);
        for (int32_t k = 0; k < si->nout; k++) r->mask |= UINT32_C(1) << si->out[k];
        fill_dense_step(n, slots, si, &out->step[s]);
    }
    return TR_MATH_OK;
}
/* the result of a dense step as a one-group GENERIC dense tensor */
static tr_tensor *run_dense_step(const tr_tensor_dense_step *d, const tr_math_i128 *A, const tr_math_i128 *B,
                                 const uint64_t *dims, int32_t order, int32_t *error) {
    tr_tensor *t = tensor_alloc(TR_TENSOR_DENSE, order, dims, 1, 0, 0, 0);
    if (!t) { set_error(error, TR_MATH_MEMORY); return NULL; }
    for (uint64_t o = 0; o < d->cells; o++) t->value[o] = set128(tr_tensor_dense_entry(d, A, B, o));
    t->bound = scan_bound(t);
    return t;
}
typedef struct { uint64_t key, i; } skey;
static int skey_cmp(const void *x, const void *y) {
    const skey *a = x, *b = y;
    if (a->key != b->key) return a->key < b->key ? -1 : 1;
    return a->i < b->i ? -1 : a->i > b->i;
}
/* a sparse step: entries of X and Y matched on their shared labels, products emitted, sorted and reduced */
static tr_tensor *run_sparse_step(const tr_tensor_net *n, const slot_t *sx, const tr_tensor *X, const slot_t *sy,
                                  const tr_tensor *Y, const tr_tensor_step_info *si, const tr_math_budget *budget,
                                  uint64_t *pairs_out, tr_math_refusal *refusal, int32_t *error) {
    uint32_t shared = Y ? sx->mask & sy->mask : 0;
    int32_t px[TR_TENSOR_MAX_LABELS], py[TR_TENSOR_MAX_LABELS];    /* position of label l in X / Y, -1 */
    for (int32_t l = 0; l < TR_TENSOR_MAX_LABELS; l++) { px[l] = -1; py[l] = -1; }
    for (int32_t k = 0; k < sx->n; k++) px[(int)sx->lab[k]] = k;
    if (Y) for (int32_t k = 0; k < sy->n; k++) py[(int)sy->lab[k]] = k;
    uint64_t ostride[TR_TENSOR_MAX_LABELS], odims[TR_TENSOR_MAX_LABELS], c1 = 1;
    for (int32_t k = si->nout - 1; k >= 0; k--) { ostride[k] = c1; odims[k] = n->ldim[(int)si->out[k]]; c1 *= odims[k] ? odims[k] : 1; }
    uint64_t nx = X->nnz, ny = Y ? Y->nnz : 0;
    skey *kx = zalloc((nx + 1) * sizeof(skey)), *ky = zalloc((ny + 1) * sizeof(skey));
    if (!kx || !ky) { free(kx); free(ky); set_error(error, TR_MATH_MEMORY); return NULL; }
    for (uint64_t e = 0; e < nx; e++) {
        uint64_t key = 0;
        for (int32_t l = 0; l < TR_TENSOR_MAX_LABELS; l++)
            if ((shared >> l) & 1) key = key * n->ldim[l] + X->coord[e * (uint64_t)X->order + (uint64_t)px[l]];
        kx[e].key = key; kx[e].i = e;
    }
    for (uint64_t e = 0; e < ny; e++) {
        uint64_t key = 0;
        for (int32_t l = 0; l < TR_TENSOR_MAX_LABELS; l++)
            if ((shared >> l) & 1) key = key * n->ldim[l] + Y->coord[e * (uint64_t)Y->order + (uint64_t)py[l]];
        ky[e].key = key; ky[e].i = e;
    }
    if (Y) { qsort(kx, (size_t)nx, sizeof *kx, skey_cmp); qsort(ky, (size_t)ny, sizeof *ky, skey_cmp); }
    uint64_t pairs = 0;
    if (!Y) pairs = nx;
    else
        for (uint64_t i = 0, j = 0; i < nx && j < ny;) {
            if (kx[i].key < ky[j].key) { i++; continue; }
            if (kx[i].key > ky[j].key) { j++; continue; }
            uint64_t i2 = i, j2 = j;
            while (i2 < nx && kx[i2].key == kx[i].key) i2++;
            while (j2 < ny && ky[j2].key == ky[j].key) j2++;
            pairs = tr_budget_add(pairs, tr_budget_mul(i2 - i, j2 - j));
            i = i2; j = j2;
        }
    tr_math_need need;
    need_zero(&need);
    need.need[TR_BUDGET_STATES] = pairs;
    need.need[TR_BUDGET_BYTES] = tr_budget_mul(pairs, REC_BYTES);
    int32_t rc = check_one(budget, &need, refusal);
    rec_t *r = rc ? NULL : zalloc(pairs * REC_BYTES);
    if (rc || !r) { free(kx); free(ky); set_error(error, rc ? rc : TR_MATH_MEMORY); return NULL; }
    uint64_t w = 0;
    if (!Y)
        for (uint64_t e = 0; e < nx; e++) {
            uint64_t idx = 0;
            for (int32_t k = 0; k < si->nout; k++) idx += (uint64_t)X->coord[e * (uint64_t)X->order + (uint64_t)px[(int)si->out[k]]] * ostride[k];
            r[w].idx = idx; r[w].v = get128(X->value[e]); w++;
        }
    else
        for (uint64_t i = 0, j = 0; i < nx && j < ny;) {
            if (kx[i].key < ky[j].key) { i++; continue; }
            if (kx[i].key > ky[j].key) { j++; continue; }
            uint64_t i2 = i, j2 = j;
            while (i2 < nx && kx[i2].key == kx[i].key) i2++;
            while (j2 < ny && ky[j2].key == ky[j].key) j2++;
            for (uint64_t a = i; a < i2; a++)
                for (uint64_t b = j; b < j2; b++) {
                    const uint32_t *ca = X->coord + kx[a].i * (uint64_t)X->order, *cb = Y->coord + ky[b].i * (uint64_t)Y->order;
                    uint64_t idx = 0;
                    for (int32_t k = 0; k < si->nout; k++) {
                        int l = si->out[k];
                        idx += (uint64_t)(px[l] >= 0 ? ca[px[l]] : cb[py[l]]) * ostride[k];
                    }
                    r[w].idx = idx;
                    r[w].v = get128(X->value[kx[a].i]) * get128(Y->value[ky[b].i]);     /* within the step's bound */
                    w++;
                }
            i = i2; j = j2;
        }
    free(kx); free(ky);
    tr_tensor *t = records_to_sparse(r, w, si->nout, odims, 1, 0, budget, tr_budget_mul(pairs, REC_BYTES), refusal, error);
    free(r);
    if (pairs_out) *pairs_out = pairs;
    return t;
}
tr_tensor *tr_tensor_net_execute(const tr_tensor_net *n, const tr_tensor_plan *p, int32_t index,
                                 const tr_math_budget *budget, tr_tensor_exec *rec, tr_math_refusal *refusal,
                                 int32_t *error) {
    tr_budget_none(refusal);
    const tr_tensor_candidate *c = NULL;
    int32_t rc = resolve(n, p, index, &c);
    if (rc) { if (rc == TR_MATH_BUDGET && refusal) *refusal = index < 0 ? p->refusal : c->reg.budget; set_error(error, rc); return NULL; }
    if (c->device != TR_REG_ON_CPU) { set_error(error, TR_MATH_UNSUPPORTED); return NULL; }
    tr_math_budget dflt;
    budget = budget_or(budget, &dflt);
    rc = tr_budget_check(budget, &c->reg.need, refusal);
    if (rc) { set_error(error, rc); return NULL; }
    const int dense = c->kernel == TR_TENSOR_KERNEL_DENSE;
    tr_tensor_exec local;
    if (!rec) rec = &local;
    memset(rec, 0, sizeof *rec);
    rec->candidate = (int32_t)(c - p->cand); rec->kernel = c->kernel; rec->device = c->device; rec->nsteps = c->nsteps;
    const tr_tensor *op[2 * TR_TENSOR_NET_MAX];
    tr_tensor *own[2 * TR_TENSOR_NET_MAX];
    memset(own, 0, sizeof own);
    uint64_t live = 0;
    for (int32_t i = 0; i < n->n; i++) {
        op[i] = n->t[i];
        int convert = dense ? op[i]->repr != TR_TENSOR_DENSE : op[i]->repr != TR_TENSOR_SPARSE;
        if (!convert) continue;
        own[i] = tr_tensor_convert(op[i], dense ? TR_TENSOR_DENSE : TR_TENSOR_SPARSE, budget, refusal, error);
        if (!own[i]) { for (int32_t j = 0; j < i; j++) tensor_release(own[j]); return NULL; }
        op[i] = own[i];
        live += own[i]->bytes;
        rec->converted_bytes += own[i]->bytes;
    }
    rec->peak_bytes = live;
    slot_t slots[2 * TR_TENSOR_NET_MAX];
    input_slots(n, slots);
    tr_tensor *result = NULL;
    for (int32_t s = 0; s < c->nsteps && !rc; s++) {
        const tr_tensor_step_info *si = &c->step[s];
        slot_t *r = &slots[n->n + s];
        r->n = si->nout; r->mask = 0;
        memcpy(r->lab, si->out, sizeof r->lab);
        for (int32_t k = 0; k < si->nout; k++) r->mask |= UINT32_C(1) << si->out[k];
        uint64_t dims[TR_TENSOR_MAX_LABELS];
        for (int32_t k = 0; k < si->nout; k++) dims[k] = n->ldim[(int)si->out[k]];
        tr_tensor *t;
        uint64_t pairs = 0;
        if (dense) {
            tr_tensor_dense_step d;
            fill_dense_step(n, slots, si, &d);
            t = run_dense_step(&d, op[si->a]->value, si->b >= 0 ? op[si->b]->value : NULL, dims, si->nout, error);
            pairs = si->pairs;
        } else {
            t = run_sparse_step(n, &slots[si->a], op[si->a], si->b >= 0 ? &slots[si->b] : NULL,
                                si->b >= 0 ? op[si->b] : NULL, si, budget, &pairs, refusal, error);
        }
        if (!t) { rc = error ? *error : TR_MATH_MEMORY; break; }
        rec->nnz[s] = t->nnz; rec->bytes[s] = t->bytes; rec->pairs[s] = pairs;
        live += t->bytes;
        uint64_t scratch = dense ? 0 : tr_budget_add(tr_budget_mul(pairs, TR_TENSOR_PAIR_BYTES),
                                                    tr_budget_mul(tr_budget_add(op[si->a]->nnz, si->b >= 0 ? op[si->b]->nnz : 0), TR_TENSOR_KEY_BYTES));
        if (tr_budget_add(live, scratch) > rec->peak_bytes) rec->peak_bytes = tr_budget_add(live, scratch);
        op[n->n + s] = own[n->n + s] = t;
        if (si->free_a && own[si->a]) { live -= own[si->a]->bytes; tensor_release(own[si->a]); own[si->a] = NULL; }
        if (si->b >= 0 && si->free_b && own[si->b]) { live -= own[si->b]->bytes; tensor_release(own[si->b]); own[si->b] = NULL; }
        result = t;
    }
    for (int32_t i = 0; i < 2 * TR_TENSOR_NET_MAX; i++) if (own[i] && own[i] != result) tensor_release(own[i]);
    if (rc) { if (result) tensor_release(result); set_error(error, rc); return NULL; }
    result->kind = TR_TENSOR_GENERIC; result->strategy = TR_TENSOR_STRATEGY_CONTRACTED;
    for (int32_t k = 0; k < result->order; k++) {
        memset(&result->axis[k], 0, sizeof result->axis[k]);
        result->axis[k].kind = TR_TENSOR_AXIS_INDEX; result->axis[k].col = -1; result->axis[k].dim = result->dims[k];
    }
    rec->output_nnz = result->nnz; rec->output_bytes = result->bytes;
    set_error(error, TR_MATH_OK);
    return result;
}
