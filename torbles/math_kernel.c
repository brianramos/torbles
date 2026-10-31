/* KERNEL-001: exact finite-feature kernel embeddings over TENSOR-001 feature states; see math_kernel.h.
 * SPDX-License-Identifier: Apache-2.0 */
#include "math_kernel.h"
#include "mathematics.h"
#include "math_domain.h"
#include "math_handles.h"
#include <stdlib.h>
#include <string.h>

TR_MATH_ABI(tr_kernel_spec, 1);          /* the ctypes boundary of KERNEL-001 (math_handles.h) */
TR_MATH_ABI(tr_kernel_wide, 1);
TR_MATH_ABI(tr_kernel_request, 1);
TR_MATH_ABI(tr_kernel_result, 1);
TR_MATH_ABI(tr_kernel_view, 1);
TR_MATH_ABI(tr_kernel_info, 1);

_Static_assert(sizeof(tr_kernel_spec) == 2728, "tr_kernel_spec ABI");
_Static_assert(sizeof(tr_kernel_wide) == 128, "tr_kernel_wide ABI");
_Static_assert(sizeof(tr_kernel_request) == 40, "tr_kernel_request ABI");
_Static_assert(sizeof(tr_kernel_result) == 392, "tr_kernel_result ABI");
_Static_assert(sizeof(tr_kernel_view) == 408, "tr_kernel_view ABI");
_Static_assert(sizeof(tr_kernel_info) == 104, "tr_kernel_info ABI");

#define STATE_MAGIC UINT32_C(0x534e524b)
#define EVAL_MAGIC UINT32_C(0x454e524b)

struct tr_kernel_state_s {
    uint32_t magic;
    int32_t degree, repr, weight, wscale;
    tr_kernel_spec kernel;              /* the declared kernel (validated, canonical) */
    uint64_t ngroups, bytes;
    uint64_t digest[2];
    tr_tensor *tensor;                  /* owned */
    tr_math_i128 *W;
    tr_kernel_wide *Q, *Dg;
};
struct tr_kernel_eval_s {
    uint32_t magic;
    const tr_kernel_state *state;
    tr_kernel_spec kernel;
    tr_kernel_view view;
    uint64_t *item_index;
    int32_t *item_class, *pos_of_axis;
    uint32_t *item_mult;
    tr_math_i128 *class_coef;
    uint64_t bytes;
};

static void set_error(int32_t *error, int32_t code) { if (error) *error = code; }
static void *zalloc(uint64_t bytes) {
    if (bytes > (uint64_t)SIZE_MAX / 2) return NULL;
    return calloc(1, bytes ? (size_t)bytes : 1);
}
static int state_ok(const tr_kernel_state *s) { return s && s->magic == STATE_MAGIC; }
static int eval_ok(const tr_kernel_eval *e) { return e && e->magic == EVAL_MAGIC; }
static const tr_math_budget *budget_or(const tr_math_budget *b, tr_math_budget *dflt) {
    if (b) return b;
    tr_budget_default(dflt);
    return dflt;
}
static void need_zero(tr_math_need *n) { for (int i = 0; i < TR_BUDGET_RESOURCES; i++) n->need[i] = 0; }

/* ------------------------------------------------------------------ kernels */
static int known_kind(int32_t k) { return k == TR_KERNEL_POLY || k == TR_KERNEL_MONOMIAL || k == TR_KERNEL_DELTA; }
static int32_t term_degree(const tr_kernel_spec *k, int32_t t) {
    int32_t d = 0;
    for (int32_t j = 0; j < k->ncols; j++) d += k->exps[t][j];
    return d;
}
/* the scope-independent checks; MONOMIAL terms merged (equal exponents), zero weights dropped, degree set */
static int32_t check_structure(tr_kernel_spec *k) {
    if (!k) return TR_MATH_INVALID;
    if (!known_kind(k->kind)) return (k->kind >= TR_KERNEL_RBF && k->kind <= TR_KERNEL_OTHER) ? TR_MATH_UNSUPPORTED : TR_MATH_INVALID;
    for (int i = 0; i < 4; i++) if (k->reserved[i]) return TR_MATH_INVALID;
    if (k->ncols < 1) return TR_MATH_INVALID;
    if (k->ncols > TR_KERNEL_MAX_COLUMNS || (k->kind == TR_KERNEL_DELTA && k->ncols > TR_TENSOR_MAX_ORDER)) return TR_MATH_UNSUPPORTED;
    for (int32_t j = k->ncols; j < TR_KERNEL_MAX_COLUMNS; j++) if (k->cols[j]) return TR_MATH_INVALID;
    for (int32_t j = 0; j < k->ncols; j++)
        for (int32_t i = 0; i < j; i++) if (k->cols[i] == k->cols[j]) return TR_MATH_INVALID;
    if (k->kind == TR_KERNEL_POLY) {
        if (k->nterms) return TR_MATH_INVALID;
        if (k->degree < 0) return TR_MATH_INVALID;
        if (k->degree > TR_KERNEL_MAX_DEGREE) return TR_MATH_UNSUPPORTED;
        __int128 pw[TR_KERNEL_MAX_DEGREE + 1];
        if (!tr_kernel_powers(k->c, k->degree, pw)) return TR_MATH_OVERFLOW;
    } else if (k->kind == TR_KERNEL_DELTA) {
        if (k->nterms || k->c || k->degree) return TR_MATH_INVALID;
    } else {
        if (k->c || k->nterms < 1) return TR_MATH_INVALID;
        if (k->nterms > TR_KERNEL_MAX_TERMS) return TR_MATH_UNSUPPORTED;
        int32_t out = 0, top = 0;
        for (int32_t t = 0; t < k->nterms; t++) {
            for (int32_t j = k->ncols; j < TR_KERNEL_MAX_COLUMNS; j++) if (k->exps[t][j]) return TR_MATH_INVALID;
            if (term_degree(k, t) > TR_KERNEL_MAX_DEGREE) return TR_MATH_UNSUPPORTED;
            int32_t same = -1;
            for (int32_t u = 0; u < out; u++) if (!memcmp(k->exps[u], k->exps[t], TR_KERNEL_MAX_COLUMNS)) same = u;
            if (same >= 0) {
                int64_t s;
                if (__builtin_add_overflow(k->weight[same], k->weight[t], &s)) return TR_MATH_OVERFLOW;
                k->weight[same] = s;
            } else {
                if (out != t) { memcpy(k->exps[out], k->exps[t], TR_KERNEL_MAX_COLUMNS); k->weight[out] = k->weight[t]; }
                out++;
            }
        }
        int32_t kept = 0;
        for (int32_t t = 0; t < out; t++) {
            if (!k->weight[t]) continue;
            if (kept != t) { memcpy(k->exps[kept], k->exps[t], TR_KERNEL_MAX_COLUMNS); k->weight[kept] = k->weight[t]; }
            kept++;
        }
        for (int32_t t = kept; t < TR_KERNEL_MAX_TERMS; t++) { memset(k->exps[t], 0, TR_KERNEL_MAX_COLUMNS); k->weight[t] = 0; }
        if (!kept) return TR_MATH_INVALID;                        /* every weight cancelled: no kernel */
        k->nterms = kept;
        for (int32_t t = 0; t < kept; t++) { int32_t d = term_degree(k, t); if (d > top) top = d; }
        k->degree = top;
    }
    return TR_MATH_OK;
}
int32_t tr_kernel_check(const tr_scope *scope, tr_kernel_spec *k) {
    tr_scope_view v;
    if (!scope || tr_scope_get_view(scope, &v)) return TR_MATH_INVALID;
    int32_t rc = check_structure(k);
    if (rc) return rc;
    for (int32_t j = 0; j < k->ncols; j++) if (k->cols[j] < 0 || k->cols[j] >= v.ncols) return TR_MATH_INVALID;
    if (k->kind != TR_KERNEL_DELTA)
        for (int32_t j = 1; j < k->ncols; j++) if (v.col[k->cols[j]].scale != v.col[k->cols[0]].scale) return TR_MATH_DOMAIN_ERROR;
    return TR_MATH_OK;
}

/* ------------------------------------------------------------------ the feature state */
static void state_release(tr_kernel_state *s) {
    if (!s) return;
    tr_tensor_free(s->tensor); free(s->W); free(s->Q); free(s->Dg);
    s->magic = 0;
    free(s);
}
void tr_kernel_free(tr_kernel_state *s) { if (state_ok(s)) state_release(s); }
/* resolves the build's arguments: the checked kernel, the order D and the tensor spec */
static int32_t prepare(const tr_scope *scope, const tr_kernel_spec *kernel, int32_t weight, int32_t degree, int32_t repr,
                       tr_kernel_spec *k, int32_t *D, tr_tensor_spec *spec, tr_scope_view *view) {
    if (!scope || !kernel || tr_scope_get_view(scope, view) || (repr != TR_TENSOR_DENSE && repr != TR_TENSOR_SPARSE) ||
        (weight != -1 && (weight < 0 || weight >= view->ncols)) || degree < 0) return TR_MATH_INVALID;
    *k = *kernel;
    int32_t rc = tr_kernel_check(scope, k);
    if (rc) return rc;
    if (k->kind == TR_KERNEL_DELTA) { if (degree) return TR_MATH_INVALID; *D = 0; }
    else {
        *D = degree ? degree : k->degree;
        if (*D < k->degree) return TR_MATH_INVALID;
        if (*D > TR_KERNEL_MAX_DEGREE) return TR_MATH_UNSUPPORTED;
        if (*D == 0) *D = 1;            /* order >= 1: the axes reference the kernel's columns (complete rows) */
    }
    return tr_kernel_tensor_spec(k, *D, weight, repr, spec);
}
static uint64_t extra_bytes(uint64_t G) { return tr_budget_add(sizeof(tr_kernel_state), tr_budget_mul(G, TR_KERNEL_GROUP_BYTES)); }

int32_t tr_kernel_need(const tr_scope *scope, const tr_kernel_spec *kernel, int32_t weight, int32_t degree, int32_t repr,
                       int32_t target, tr_math_need *need, uint64_t *capacity) {
    tr_kernel_spec k;
    tr_tensor_spec spec;
    tr_scope_view view;
    int32_t D;
    if (!need) return TR_MATH_INVALID;
    int32_t rc = prepare(scope, kernel, weight, degree, repr, &k, &D, &spec, &view);
    if (rc) return rc;
    rc = tr_tensor_need(scope, &spec, target, need, capacity);
    if (rc) return rc;
    uint64_t extra = extra_bytes(view.ngroups);
    if (target == TR_TENSOR_NEED_DEVICE)                 /* the device's W, Q, Dg and the kernel's copy */
        extra = tr_budget_add(tr_budget_mul(view.ngroups, TR_KERNEL_GROUP_BYTES), sizeof(tr_kernel_spec));
    need->need[TR_BUDGET_BYTES] = tr_budget_add(need->need[TR_BUDGET_BYTES], extra);
    if ((uint64_t)k.ncols + (weight >= 0) > need->need[TR_BUDGET_VARIABLES])
        need->need[TR_BUDGET_VARIABLES] = (uint64_t)k.ncols + (weight >= 0);
    return TR_MATH_OK;
}

static tr_kernel_state *state_alloc(const tr_kernel_spec *k, int32_t D, int32_t repr, int32_t weight,
                                    const tr_scope_view *view, uint64_t G) {
    tr_kernel_state *s = zalloc(sizeof *s);
    if (!s) return NULL;
    s->kernel = *k; s->degree = D; s->repr = repr; s->weight = weight; s->ngroups = G;
    s->wscale = weight >= 0 ? view->col[weight].scale : 0;
    s->W = zalloc(G * 16); s->Q = zalloc(G * sizeof(tr_kernel_wide)); s->Dg = zalloc(G * sizeof(tr_kernel_wide));
    if (!s->W || !s->Q || !s->Dg) { state_release(s); return NULL; }
    s->magic = STATE_MAGIC;
    return s;
}
tr_kernel_state *tr_kernel_build(const tr_scope *scope, const tr_kernel_spec *kernel, int32_t weight, int32_t degree,
                                 int32_t repr, const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    tr_kernel_spec k;
    tr_tensor_spec spec;
    tr_scope_view view;
    int32_t D;
    int32_t rc = prepare(scope, kernel, weight, degree, repr, &k, &D, &spec, &view);
    if (rc) { set_error(error, rc); return NULL; }
    if (tr_kernel_self_bits(&view, &k, weight) > TR_KERNEL_BITS_LIMIT) { set_error(error, TR_MATH_OVERFLOW); return NULL; }
    tr_math_need need;
    rc = tr_kernel_need(scope, kernel, weight, degree, repr, TR_TENSOR_NEED_CPU, &need, NULL);
    tr_math_budget dflt;
    budget = budget_or(budget, &dflt);
    if (!rc) rc = tr_budget_check(budget, &need, refusal);
    if (rc) { set_error(error, rc); return NULL; }
    const uint64_t G = view.ngroups;
    tr_kernel_state *s = state_alloc(&k, D, repr, weight, &view, G);
    if (!s) { set_error(error, TR_MATH_MEMORY); return NULL; }
    s->tensor = tr_tensor_build(scope, &spec, budget, refusal, error);
    if (!s->tensor) { state_release(s); return NULL; }
    /* the per-group weight sums and self-pair terms of the declared kernel (proven to fit 1024 bits) */
    uint64_t nw = (view.rows + 63) / 64;
    for (uint64_t w = 0; w < nw; w++) {
        uint64_t bits = view.selection[w];
        while (bits) {
            uint64_t r = (w << 6) | (uint64_t)__builtin_ctzll(bits);
            bits &= bits - 1;
            __int128 wt;
            if (!tr_kernel_row_ok(&view, &k, weight, r, &wt)) continue;
            uint32_t g = tr_scope_group(&view, r);
            tr_kernel_wide t, q;
            s->W[g] = tr_i128_set(tr_i128_get(s->W[g]) + wt);
            tr_kernel_w_from_i128(wt * wt, &q);
            if (!tr_kernel_w_add(&s->Q[g], &q, &s->Q[g]) || !tr_kernel_self_term(&view, &k, r, wt, &t) ||
                !tr_kernel_w_add(&s->Dg[g], &t, &s->Dg[g])) { state_release(s); set_error(error, TR_MATH_OVERFLOW); return NULL; }
        }
    }
    tr_tensor_info ti;
    tr_tensor_get_info(s->tensor, &ti);
    s->bytes = tr_budget_add(ti.bytes, extra_bytes(G));
    s->digest[0] = ti.digest[0]; s->digest[1] = ti.digest[1];
    set_error(error, TR_MATH_OK);
    return s;
}

/* W of a group from the tensor: the all-constant entry (moments) or the sum of the cells (counts) */
static int tensor_weight(const tr_tensor *t, int32_t kind, uint64_t g, __int128 *W) {
    tr_tensor_info info;
    tr_tensor_get_info(t, &info);
    if (kind != TR_KERNEL_DELTA) {
        uint32_t zero[TR_TENSOR_MAX_ORDER] = {0};
        tr_math_i128 v;
        if (tr_tensor_get(t, g, zero, &v, NULL)) return 0;
        *W = tr_i128_get(v);
        return 1;
    }
    tr_tensor_states st;
    tr_tensor_get_states(t, &st);
    __int128 sum = 0;
    uint64_t b = st.repr == TR_TENSOR_DENSE ? g * st.cells : st.offset[g], e = st.repr == TR_TENSOR_DENSE ? (g + 1) * st.cells : st.offset[g + 1];
    for (uint64_t i = b; i < e; i++) if (__builtin_add_overflow(sum, tr_i128_get(st.value[i]), &sum)) return 0;
    *W = sum;
    return 1;
}
tr_kernel_state *tr_kernel_from_states(const tr_scope *origin, const tr_kernel_spec *kernel, int32_t weight,
                                       int32_t degree, int32_t repr, const tr_tensor_states *tensor,
                                       const tr_math_i128 *W, const tr_kernel_wide *Q, const tr_kernel_wide *Dg,
                                       const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    tr_kernel_spec k;
    tr_tensor_spec spec;
    tr_scope_view view;
    int32_t D;
    int32_t rc = prepare(origin, kernel, weight, degree, repr, &k, &D, &spec, &view);
    if (rc) { set_error(error, rc); return NULL; }
    if (!tensor || !W || !Q || !Dg) { set_error(error, TR_MATH_INVALID); return NULL; }
    tr_math_need need;
    need_zero(&need);
    need.need[TR_BUDGET_BYTES] = extra_bytes(view.ngroups);
    tr_math_budget dflt;
    budget = budget_or(budget, &dflt);
    rc = tr_budget_check(budget, &need, refusal);
    if (rc) { set_error(error, rc); return NULL; }
    const uint64_t G = view.ngroups;
    tr_kernel_state *s = state_alloc(&k, D, repr, weight, &view, G);
    if (!s) { set_error(error, TR_MATH_MEMORY); return NULL; }
    s->tensor = tr_tensor_from_states(origin, &spec, tensor, budget, refusal, error);
    if (!s->tensor) { state_release(s); return NULL; }
    for (uint64_t g = 0; g < G; g++) {               /* consistency: W against the tensor, Q >= 0, DELTA: Dg = Q */
        __int128 tw;
        int bad = !tensor_weight(s->tensor, k.kind, g, &tw) || tw != tr_i128_get(W[g]) || tr_kernel_w_neg(&Q[g]) ||
                  (k.kind == TR_KERNEL_DELTA && memcmp(&Q[g], &Dg[g], sizeof Q[g]));
        if (bad) { state_release(s); set_error(error, TR_MATH_INVALID); return NULL; }
    }
    memcpy(s->W, W, (size_t)G * 16);
    memcpy(s->Q, Q, (size_t)G * sizeof(tr_kernel_wide));
    memcpy(s->Dg, Dg, (size_t)G * sizeof(tr_kernel_wide));
    tr_tensor_info ti;
    tr_tensor_get_info(s->tensor, &ti);
    s->bytes = tr_budget_add(ti.bytes, extra_bytes(G));
    s->digest[0] = ti.digest[0]; s->digest[1] = ti.digest[1];
    set_error(error, TR_MATH_OK);
    return s;
}
int32_t tr_kernel_get_info(const tr_kernel_state *s, tr_kernel_info *out) {
    if (!state_ok(s) || !out) return TR_MATH_INVALID;
    memset(out, 0, sizeof *out);
    tr_tensor_info ti;
    tr_tensor_get_info(s->tensor, &ti);
    out->ngroups = s->ngroups; out->present = ti.present; out->rows = ti.rows; out->used = ti.used;
    out->bytes = s->bytes; out->cells = ti.cells; out->nnz = ti.nnz;
    out->kind = s->kernel.kind; out->degree = s->kernel.degree; out->ncols = s->kernel.ncols; out->order = ti.order;
    out->repr = s->repr; out->weighted = s->weight >= 0;
    out->digest[0] = s->digest[0]; out->digest[1] = s->digest[1];
    return TR_MATH_OK;
}
int32_t tr_kernel_get_group(const tr_kernel_state *s, uint64_t g, tr_math_i128 *W, tr_kernel_wide *Q, tr_kernel_wide *Dg) {
    if (!state_ok(s) || g >= s->ngroups) return TR_MATH_INVALID;
    if (W) *W = s->W[g];
    if (Q) *Q = s->Q[g];
    if (Dg) *Dg = s->Dg[g];
    return TR_MATH_OK;
}
const tr_tensor *tr_kernel_get_tensor(const tr_kernel_state *s) { return state_ok(s) ? s->tensor : NULL; }
int32_t tr_kernel_get_spec(const tr_kernel_state *s, tr_kernel_spec *out) {
    if (!state_ok(s) || !out) return TR_MATH_INVALID;
    *out = s->kernel;
    return TR_MATH_OK;
}

/* ------------------------------------------------------------------ evaluations */
static void eval_release(tr_kernel_eval *e) {
    if (!e) return;
    free(e->item_index); free(e->item_class); free(e->item_mult); free(e->pos_of_axis); free(e->class_coef);
    e->magic = 0;
    free(e);
}
void tr_kernel_eval_free(tr_kernel_eval *e) { if (eval_ok(e)) eval_release(e); }
typedef struct { uint64_t index; int32_t cls; uint32_t mult; } item_t;
static int item_cmp(const void *x, const void *y) {
    const item_t *a = x, *b = y;
    return a->index < b->index ? -1 : a->index > b->index;
}
tr_kernel_eval *tr_kernel_eval_make(const tr_kernel_state *s, const tr_kernel_spec *kernel, const tr_math_budget *budget,
                                    tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    if (!state_ok(s)) { set_error(error, TR_MATH_INVALID); return NULL; }
    tr_kernel_spec k = kernel ? *kernel : s->kernel;
    int32_t rc = check_structure(&k);
    if (rc) { set_error(error, rc); return NULL; }
    tr_tensor_info ti;
    tr_tensor_states st;
    tr_tensor_get_info(s->tensor, &ti);
    tr_tensor_get_states(s->tensor, &st);
    const int32_t D = ti.order;
    /* the state's columns: moment positions 1..n of axis 0 (0 is the constant); counts: one column per axis */
    tr_tensor_axis ax[TR_TENSOR_MAX_ORDER];
    for (int32_t a = 0; a < D; a++) tr_tensor_get_axis(s->tensor, a, &ax[a]);
    int moments = s->kernel.kind != TR_KERNEL_DELTA;
    if ((k.kind == TR_KERNEL_DELTA) == moments) { set_error(error, TR_MATH_UNSUPPORTED); return NULL; }   /* family */
    int32_t pos[TR_KERNEL_MAX_COLUMNS];
    if (moments) {
        if (k.degree > s->degree) { set_error(error, TR_MATH_UNSUPPORTED); return NULL; }        /* degree absent */
        for (int32_t j = 0; j < k.ncols; j++) {
            pos[j] = -1;
            for (int32_t i = 0; i < s->kernel.ncols; i++) if (s->kernel.cols[i] == k.cols[j]) pos[j] = i + 1;
            if (pos[j] < 0) { set_error(error, TR_MATH_UNSUPPORTED); return NULL; }              /* dimension absent */
        }
    } else {
        if (k.ncols != s->kernel.ncols || memcmp(k.cols, s->kernel.cols, sizeof k.cols)) {
            set_error(error, TR_MATH_UNSUPPORTED); return NULL;
        }
    }
    /* the items: POLY every sorted multi-index (0^{D-d}, J), J non-decreasing over {0} u positions, class = the zeros
     * of J, multiplicity d! / prod_c r_c! (its orderings: the tensor is symmetric); MONOMIAL one item per term (its
     * positions sorted, leading zeros), multiplicity 1 */
    uint64_t nitems = 0, choices = (uint64_t)k.ncols + 1;
    if (k.kind == TR_KERNEL_POLY) {                                    /* C(choices + d - 1, d), exact at every step */
        nitems = 1;
        for (int32_t a = 1; a <= k.degree; a++) nitems = nitems * (choices - 1 + (uint64_t)a) / (uint64_t)a;
    } else if (k.kind == TR_KERNEL_MONOMIAL) nitems = (uint64_t)k.nterms;
    tr_math_need need;
    need_zero(&need);
    need.need[TR_BUDGET_DEGREE] = (uint64_t)k.degree;
    need.need[TR_BUDGET_SUPPORT] = nitems;
    need.need[TR_BUDGET_BYTES] = tr_budget_add(sizeof(tr_kernel_eval), tr_budget_mul(nitems, TR_KERNEL_ITEM_BYTES));
    tr_math_budget dflt;
    rc = tr_budget_check(budget_or(budget, &dflt), &need, refusal);
    if (rc) { set_error(error, rc); return NULL; }
    tr_kernel_eval *e = zalloc(sizeof *e);
    if (!e) { set_error(error, TR_MATH_MEMORY); return NULL; }
    e->state = s; e->kernel = k; e->bytes = need.need[TR_BUDGET_BYTES];
    int32_t nclass = k.kind == TR_KERNEL_POLY ? k.degree + 1 : k.kind == TR_KERNEL_MONOMIAL ? k.nterms : 1;
    e->item_index = zalloc(nitems * 8);
    e->item_class = zalloc(nitems * 4);
    e->item_mult = zalloc(nitems * 4);
    e->class_coef = zalloc((uint64_t)nclass * 16);
    e->pos_of_axis = zalloc((uint64_t)(TR_TENSOR_MAX_FEATURES + 1) * 4);
    item_t *items = zalloc(nitems * sizeof(item_t));
    if (!e->item_index || !e->item_class || !e->item_mult || !e->class_coef || !e->pos_of_axis || !items) {
        free(items); eval_release(e); set_error(error, TR_MATH_MEMORY); return NULL;
    }
    uint64_t stride[TR_TENSOR_MAX_ORDER], c = 1;
    for (int32_t a = D - 1; a >= 0; a--) { stride[a] = c; c *= ti.dims[a] ? ti.dims[a] : 1; }
    for (int32_t i = 0; i <= TR_TENSOR_MAX_FEATURES; i++) e->pos_of_axis[i] = -1;
    if (moments) for (int32_t j = 0; j < k.ncols; j++) e->pos_of_axis[pos[j]] = j;
    if (k.kind == TR_KERNEL_POLY) {
        __int128 pw[TR_KERNEL_MAX_DEGREE + 1];
        tr_kernel_powers(k.c, k.degree, pw);
        for (int32_t z = 0; z <= k.degree; z++) e->class_coef[z] = tr_i128_set(pw[z]);
        int32_t sorted[TR_KERNEL_MAX_COLUMNS + 1], m = 0;
        sorted[m++] = 0;
        for (int32_t j = 0; j < k.ncols; j++) sorted[m++] = pos[j];
        for (int32_t i = 1; i < m; i++) { int32_t x = sorted[i], j = i - 1; while (j >= 1 && sorted[j] > x) { sorted[j + 1] = sorted[j]; j--; } sorted[j + 1] = x; }
        static const uint32_t fact[TR_KERNEL_MAX_DEGREE + 1] = {1, 1, 2, 6, 24, 120, 720, 5040, 40320};
        int32_t ch[TR_KERNEL_MAX_DEGREE] = {0};                        /* J: non-decreasing choices, odometer */
        for (uint64_t t = 0; t < nitems; t++) {
            uint64_t idx = 0;
            int32_t zeros = 0;
            uint32_t mult = fact[k.degree];
            for (int32_t q = 0, run = 0; q < k.degree; q++) {          /* the trailing d axes; the leading ones 0 */
                zeros += ch[q] == 0;
                idx += (uint64_t)sorted[ch[q]] * stride[D - k.degree + q];
                run = q > 0 && ch[q] == ch[q - 1] ? run + 1 : 1;
                mult /= (uint32_t)run;                                 /* d! / prod r_c!, exact at every step */
            }
            items[t].index = idx; items[t].cls = zeros; items[t].mult = mult;   /* coefficient c^(constants of J) */
            int32_t q = k.degree - 1;
            while (q >= 0 && ch[q] == m - 1) q--;
            if (q < 0) break;
            ch[q]++;
            for (int32_t r = q + 1; r < k.degree; r++) ch[r] = ch[q];
        }
    } else if (k.kind == TR_KERNEL_MONOMIAL) {
        for (int32_t t = 0; t < k.nterms; t++) {
            int32_t p[TR_KERNEL_MAX_DEGREE], n = 0;
            for (int32_t j = 0; j < k.ncols; j++) for (int32_t q = 0; q < k.exps[t][j]; q++) p[n++] = pos[j];
            for (int32_t i = 1; i < n; i++) { int32_t x = p[i], j = i - 1; while (j >= 0 && p[j] > x) { p[j + 1] = p[j]; j--; } p[j + 1] = x; }
            uint64_t idx = 0;
            for (int32_t q = 0; q < n; q++) idx += (uint64_t)p[q] * stride[D - n + q];
            items[t].index = idx; items[t].cls = t; items[t].mult = 1;
            e->class_coef[t] = tr_i128_set(k.weight[t]);
        }
    } else e->class_coef[0] = tr_i128_set(1);
    if (nitems > 1) qsort(items, (size_t)nitems, sizeof *items, item_cmp);
    for (uint64_t t = 0; t < nitems; t++) {
        e->item_index[t] = items[t].index; e->item_class[t] = items[t].cls; e->item_mult[t] = items[t].mult;
    }
    free(items);
    tr_kernel_view *v = &e->view;
    memset(v, 0, sizeof *v);
    v->kind = k.kind; v->repr = s->repr; v->order = D; v->nclass = nclass; v->npos = k.ncols;
    v->has_self = !memcmp(&k, &s->kernel, sizeof k);
    v->ngroups = s->ngroups; v->cells = ti.cells; v->nitems = nitems;
    for (int32_t a = 0; a < D; a++) {
        v->dims[a] = ti.dims[a]; v->stride[a] = stride[a];
        if (!moments) { v->lo[a] = ax[a].lo; v->span[a] = ax[a].span; }
    }
    v->item_index = e->item_index; v->item_class = e->item_class; v->item_mult = e->item_mult;
    v->class_coef = e->class_coef;
    v->pos_of_axis = e->pos_of_axis;
    v->value = st.value; v->offset = st.offset; v->coord = st.coord;
    v->W = s->W; v->Q = s->Q; v->Dg = s->Dg;
    e->magic = EVAL_MAGIC;
    set_error(error, TR_MATH_OK);
    return e;
}
int32_t tr_kernel_eval_get_view(const tr_kernel_eval *e, tr_kernel_view *out) {
    if (!eval_ok(e) || !out) return TR_MATH_INVALID;
    *out = e->view;
    return TR_MATH_OK;
}
int32_t tr_kernel_query(const tr_kernel_eval *e, const tr_kernel_request *requests, uint64_t n, const int64_t *points,
                        const uint8_t *valid, uint64_t npoints, tr_kernel_result *out, const tr_math_budget *budget,
                        tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!eval_ok(e) || (n && (!requests || !out)) || (npoints && !points)) return TR_MATH_INVALID;
    tr_math_need need;
    need_zero(&need);
    need.need[TR_BUDGET_OUTPUT] = n;
    tr_math_budget dflt;
    int32_t rc = tr_budget_check(budget_or(budget, &dflt), &need, refusal);
    if (rc) return rc;
    tr_kernel_view v = e->view;
    v.npoints = npoints;
    for (uint64_t i = 0; i < n; i++) tr_kernel_eval_request(&v, &requests[i], points, valid, &out[i]);
    return TR_MATH_OK;
}
uint64_t tr_kernel_view_device_bytes(const tr_kernel_view *v) {
    if (!v) return TR_BUDGET_UNKNOWN;
    uint64_t nnz = v->repr == TR_TENSOR_SPARSE && v->offset ? v->offset[v->ngroups] : 0;
    return tr_kernel_view_bytes(v, nnz);
}
