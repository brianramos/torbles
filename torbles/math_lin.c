/* LIN-001 / LIN-002: grouped exact moment matrices and regression systems, and their exact and numerical
 * finalization (solves, inertia, covariance, fits, eigensystems, modular batches); see math_lin.h.
 * SPDX-License-Identifier: Apache-2.0 */
#include "math_lin.h"
#include "mathematics.h"
#include "math_domain.h"
#include "math_handles.h"
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

TR_MATH_ABI(tr_lin_spec, 1);             /* the ctypes boundary of LIN-001 / LIN-002 (math_handles.h) */
TR_MATH_ABI(tr_lin_slot, 1);
TR_MATH_ABI(tr_lin_states, 1);
TR_MATH_ABI(tr_lin_info, 1);
TR_MATH_ABI(tr_lin_qbuf, 1);
TR_MATH_ABI(tr_lin_fit_spec, 1);
TR_MATH_ABI(tr_lin_result_info, 1);
TR_MATH_ABI(tr_lin_batch_view, 1);

_Static_assert(sizeof(tr_lin_spec) == 88, "tr_lin_spec ABI");
_Static_assert(sizeof(tr_lin_slot) == 8, "tr_lin_slot ABI");
_Static_assert(sizeof(tr_lin_states) == 40, "tr_lin_states ABI");
_Static_assert(sizeof(tr_lin_info) == 240, "tr_lin_info ABI");
_Static_assert(sizeof(tr_lin_qbuf) == 48, "tr_lin_qbuf ABI");
_Static_assert(sizeof(tr_lin_fit_spec) == 40, "tr_lin_fit_spec ABI");
_Static_assert(sizeof(tr_lin_result_info) == 80, "tr_lin_result_info ABI");
_Static_assert(sizeof(tr_lin_batch_view) == 56, "tr_lin_batch_view ABI");

#define MOM_MAGIC UINT32_C(0x4d4e494c)
#define RES_MAGIC UINT32_C(0x524e494c)
#define BAT_MAGIC UINT32_C(0x424e494c)
#define I128_LIMIT ((((unsigned __int128)1) << 127) - 1)
#define MAX_PRIMES 4096

static void set_error(int32_t *error, int32_t code) { if (error) *error = code; }
static void *zalloc(uint64_t bytes) {
    if (bytes > (uint64_t)SIZE_MAX) return NULL;
    return calloc(1, bytes ? (size_t)bytes : 1);
}
static const tr_math_budget *budget_or(const tr_math_budget *b, tr_math_budget *store) {
    if (b) return b;
    tr_budget_default(store);
    return store;
}

/* ================================================================ exact arithmetic with a sticky error */
typedef struct { const tr_math_budget *b; tr_math_refusal *r; int32_t rc; } lctx;
static void lfail(lctx *x, int32_t rc) { if (!x->rc) x->rc = rc ? rc : TR_MATH_MEMORY; }
static tr_bigint *zk(lctx *x, tr_bigint *v, int32_t e) { if (!v) lfail(x, e); return v; }
static tr_bigq *qk(lctx *x, tr_bigq *v, int32_t e) { if (!v) lfail(x, e); return v; }
#define LIVE(x) (!(x)->rc)
static tr_bigint *Z_i64(lctx *x, int64_t v) { int32_t e = 0; return x->rc ? NULL : zk(x, tr_bigint_from_i64(v, x->b, x->r, &e), e); }
static tr_bigint *Z_copy(lctx *x, const tr_bigint *a) { int32_t e = 0; return x->rc ? NULL : zk(x, tr_bigint_copy(a, x->b, x->r, &e), e); }
static tr_bigint *Z_neg(lctx *x, const tr_bigint *a) { int32_t e = 0; return x->rc ? NULL : zk(x, tr_bigint_neg(a, x->b, x->r, &e), e); }
static tr_bigint *Z_add(lctx *x, const tr_bigint *a, const tr_bigint *c) { int32_t e = 0; return x->rc ? NULL : zk(x, tr_bigint_add(a, c, x->b, x->r, &e), e); }
static tr_bigint *Z_sub(lctx *x, const tr_bigint *a, const tr_bigint *c) { int32_t e = 0; return x->rc ? NULL : zk(x, tr_bigint_sub(a, c, x->b, x->r, &e), e); }
static tr_bigint *Z_mul(lctx *x, const tr_bigint *a, const tr_bigint *c) { int32_t e = 0; return x->rc ? NULL : zk(x, tr_bigint_mul(a, c, x->b, x->r, &e), e); }
static tr_bigint *Z_div(lctx *x, const tr_bigint *a, const tr_bigint *c) { int32_t e = 0; return x->rc ? NULL : zk(x, tr_bigint_divexact(a, c, x->b, x->r, &e), e); }
static tr_bigint *Z_gcd(lctx *x, const tr_bigint *a, const tr_bigint *c) { int32_t e = 0; return x->rc ? NULL : zk(x, tr_bigint_gcd(a, c, x->b, x->r, &e), e); }
static tr_bigint *Z_twos(lctx *x, const uint64_t *w, uint64_t n) { int32_t e = 0; return x->rc ? NULL : zk(x, tr_bigint_from_twos(w, n, x->b, x->r, &e), e); }
static tr_bigint *Z_pow10(lctx *x, int32_t k) {
    tr_bigint *ten = Z_i64(x, 10);
    int32_t e = 0;
    tr_bigint *p = x->rc ? NULL : zk(x, tr_bigint_pow(ten, (uint64_t)k, x->b, x->r, &e), e);
    tr_bigint_free(ten);
    return p;
}
static int Z_sign(const tr_bigint *a) { return a ? tr_bigint_sign(a) : 0; }
static int Z_is_one(const tr_bigint *a) { return a && tr_bigint_sign(a) == 1 && tr_bigint_bit_length(a) == 1; }
/* a * b - c * d (any of the products may be skipped when a factor is zero) */
static tr_bigint *Z_cross(lctx *x, const tr_bigint *a, const tr_bigint *b, const tr_bigint *c, const tr_bigint *d) {
    int p = Z_sign(a) && Z_sign(b), q = Z_sign(c) && Z_sign(d);
    if (!p && !q) return Z_i64(x, 0);
    if (!q) return Z_mul(x, a, b);
    tr_bigint *t2 = Z_mul(x, c, d);
    if (!p) { tr_bigint *r = Z_neg(x, t2); tr_bigint_free(t2); return r; }
    tr_bigint *t1 = Z_mul(x, a, b), *r = Z_sub(x, t1, t2);
    tr_bigint_free(t1); tr_bigint_free(t2);
    return r;
}
static tr_bigint *Z_dot(lctx *x, const tr_bigint *const *a, const tr_bigint *const *b, uint64_t n) {
    int32_t e = 0;
    return x->rc ? NULL : zk(x, tr_bigint_dot(a, b, n, x->b, x->r, &e), e);
}

static tr_bigq *Q_int(lctx *x, int64_t v) {
    tr_num_rational q; q.numerator = v; q.denominator = 1;
    int32_t e = 0;
    return x->rc ? NULL : qk(x, tr_bigq_from_q64(q, x->b, x->r, &e), e);
}
static tr_bigq *Q_q64(lctx *x, int64_t n, int64_t d) {
    tr_num_rational q; q.numerator = n; q.denominator = d;
    int32_t e = 0;
    return x->rc ? NULL : qk(x, tr_bigq_from_q64(q, x->b, x->r, &e), e);
}
static tr_bigq *Q_frac(lctx *x, const tr_bigint *n, const tr_bigint *d) { int32_t e = 0; return x->rc ? NULL : qk(x, tr_bigq_from_bigints(n, d, x->b, x->r, &e), e); }
static tr_bigq *Q_z(lctx *x, const tr_bigint *n) { int32_t e = 0; return x->rc ? NULL : qk(x, tr_bigq_from_bigint(n, x->b, x->r, &e), e); }
static tr_bigq *Q_copy(lctx *x, const tr_bigq *a) { int32_t e = 0; return x->rc ? NULL : qk(x, tr_bigq_copy(a, x->b, x->r, &e), e); }
static tr_bigq *Q_add(lctx *x, const tr_bigq *a, const tr_bigq *c) { int32_t e = 0; return x->rc ? NULL : qk(x, tr_bigq_add(a, c, x->b, x->r, &e), e); }
static tr_bigq *Q_sub(lctx *x, const tr_bigq *a, const tr_bigq *c) { int32_t e = 0; return x->rc ? NULL : qk(x, tr_bigq_sub(a, c, x->b, x->r, &e), e); }
static tr_bigq *Q_mul(lctx *x, const tr_bigq *a, const tr_bigq *c) { int32_t e = 0; return x->rc ? NULL : qk(x, tr_bigq_mul(a, c, x->b, x->r, &e), e); }
static tr_bigq *Q_div(lctx *x, const tr_bigq *a, const tr_bigq *c) { int32_t e = 0; return x->rc ? NULL : qk(x, tr_bigq_div(a, c, x->b, x->r, &e), e); }
/* *acc += a * c (acc owned; replaced) */
static void Q_fma(lctx *x, tr_bigq **acc, const tr_bigq *a, const tr_bigq *c) {
    if (x->rc) return;
    if (!tr_bigq_sign(a) || !tr_bigq_sign(c)) return;
    tr_bigq *p = Q_mul(x, a, c), *s = Q_add(x, *acc, p);
    tr_bigq_free(p);
    if (s) { tr_bigq_free(*acc); *acc = s; }
}
static tr_bigq *Q_scale10(lctx *x, const tr_bigint *n, int32_t e) {     /* n / 10^e */
    if (!e) return Q_z(x, n);
    tr_bigint *d = Z_pow10(x, e);
    tr_bigq *q = Q_frac(x, n, d);
    tr_bigint_free(d);
    return q;
}
void tr_lin_q_free(tr_bigq **v, uint64_t n) {
    if (!v) return;
    for (uint64_t i = 0; i < n; i++) tr_bigq_free(v[i]);
    free(v);
}
static void Z_free_all(tr_bigint **v, uint64_t n) {
    if (!v) return;
    for (uint64_t i = 0; i < n; i++) tr_bigint_free(v[i]);
    free(v);
}
static tr_bigq **Q_arr(lctx *x, uint64_t n) {
    if (x->rc) return NULL;
    tr_bigq **v = (tr_bigq **)zalloc(n * sizeof *v);
    if (!v) lfail(x, TR_MATH_MEMORY);
    return v;
}
static tr_bigint **Z_arr(lctx *x, uint64_t n) {
    if (x->rc) return NULL;
    tr_bigint **v = (tr_bigint **)zalloc(n * sizeof *v);
    if (!v) lfail(x, TR_MATH_MEMORY);
    return v;
}

/* ================================================================ packed rationals */
uint64_t tr_lin_q_limbs(const tr_bigq *const *v, uint64_t n) {
    uint64_t total = 0;
    for (uint64_t i = 0; v && i < n; i++) {
        if (!v[i]) continue;
        const tr_bigint *d = tr_bigq_denominator(v[i]);
        total += tr_bigint_limb_count(tr_bigq_numerator(v[i])) + (Z_is_one(d) ? 0 : tr_bigint_limb_count(d));
    }
    return total;
}
int32_t tr_lin_q_export(const tr_bigq *const *v, uint64_t n, tr_lin_qbuf *out) {
    if (!out || (n && (!v || !out->sign || !out->nlen || !out->dlen)) || out->count < n) return TR_MATH_INVALID;
    uint64_t need = tr_lin_q_limbs(v, n);
    if (need > out->nlimbs || (need && !out->limbs)) return TR_MATH_INVALID;
    uint64_t off = 0;
    for (uint64_t i = 0; i < n; i++) {
        if (!v[i]) { out->sign[i] = TR_LIN_UNDEFINED; out->nlen[i] = 0; out->dlen[i] = 0; continue; }
        const tr_bigint *num = tr_bigq_numerator(v[i]), *den = tr_bigq_denominator(v[i]);
        int32_t sign = 0;
        uint64_t count = 0;
        int32_t rc = tr_bigint_export(num, &sign, out->limbs ? out->limbs + off : NULL, need - off, &count);
        if (rc) return rc;
        out->sign[i] = (int8_t)sign; out->nlen[i] = (uint32_t)count; off += count;
        if (Z_is_one(den)) { out->dlen[i] = 0; continue; }
        rc = tr_bigint_export(den, &sign, out->limbs + off, need - off, &count);
        if (rc) return rc;
        out->dlen[i] = (uint32_t)count; off += count;
    }
    out->count = n; out->nlimbs = off;
    return TR_MATH_OK;
}
int32_t tr_lin_q_import(const tr_lin_qbuf *in, uint64_t n, tr_bigq **out, const tr_math_budget *budget,
                        tr_math_refusal *refusal) {
    if (!in || (n && (!out || !in->sign || !in->nlen || !in->dlen)) || in->count != n) return TR_MATH_INVALID;
    for (uint64_t i = 0; i < n; i++) out[i] = NULL;
    lctx x = {budget, refusal, 0};
    uint64_t off = 0;
    for (uint64_t i = 0; i < n && LIVE(&x); i++) {
        int s = in->sign[i];
        uint64_t nl = in->nlen[i], dl = in->dlen[i];
        if (s < -1 || s > 1 || nl > in->nlimbs || dl > in->nlimbs - nl || off > in->nlimbs - nl - dl ||
            ((nl || dl) && !in->limbs)) { lfail(&x, TR_MATH_INVALID); break; }
        int32_t e = 0;
        tr_bigint *num = zk(&x, tr_bigint_from_limbs(s, nl ? in->limbs + off : NULL, nl, budget, refusal, &e), e);
        tr_bigint *den = dl ? (LIVE(&x) ? zk(&x, tr_bigint_from_limbs(1, in->limbs + off + nl, dl, budget, refusal, &e), e) : NULL)
                            : Z_i64(&x, 1);
        if (LIVE(&x) && !Z_sign(den)) lfail(&x, TR_MATH_INVALID);
        out[i] = Q_frac(&x, num, den);
        tr_bigint_free(num); tr_bigint_free(den);
        off += nl + dl;
    }
    if (x.rc) { for (uint64_t i = 0; i < n; i++) { tr_bigq_free(out[i]); out[i] = NULL; } }
    return x.rc;
}

/* ================================================================ results */
struct tr_lin_result_s {
    uint32_t magic;
    tr_lin_result_info info;
    tr_bigq **q[TR_LIN_PARTS];
    uint64_t nq[TR_LIN_PARTS];
    double *d[TR_LIN_PARTS];
    uint64_t nd[TR_LIN_PARTS];
};
static int res_valid(const tr_lin_result *r) { return r && r->magic == RES_MAGIC; }
static tr_lin_result *res_new(lctx *x, int32_t what) {
    if (x->rc) return NULL;
    tr_lin_result *r = (tr_lin_result *)zalloc(sizeof *r);
    if (!r) { lfail(x, TR_MATH_MEMORY); return NULL; }
    r->magic = RES_MAGIC; r->info.what = what;
    return r;
}
void tr_lin_result_free(tr_lin_result *r) {
    if (!res_valid(r)) return;
    for (int p = 0; p < TR_LIN_PARTS; p++) { tr_lin_q_free(r->q[p], r->nq[p]); free(r->d[p]); }
    r->magic = 0;
    free(r);
}
static void res_q(tr_lin_result *r, int32_t part, tr_bigq **v, uint64_t n) {   /* takes v */
    tr_lin_q_free(r->q[part], r->nq[part]);
    r->q[part] = v; r->nq[part] = n;
}
static int res_d(lctx *x, tr_lin_result *r, int32_t part, uint64_t n) {        /* allocates n doubles */
    if (x->rc) return 0;
    free(r->d[part]);
    r->d[part] = (double *)zalloc(n * sizeof(double));
    r->nd[part] = r->d[part] ? n : 0;
    if (!r->d[part]) lfail(x, TR_MATH_MEMORY);
    return r->d[part] != NULL;
}
int32_t tr_lin_result_get_info(const tr_lin_result *r, tr_lin_result_info *out) {
    if (!res_valid(r) || !out) return TR_MATH_INVALID;
    *out = r->info;
    return TR_MATH_OK;
}
uint64_t tr_lin_result_count(const tr_lin_result *r, int32_t part) {
    if (!res_valid(r) || part < 0 || part >= TR_LIN_PARTS) return 0;
    return r->q[part] ? r->nq[part] : r->nd[part];
}
uint64_t tr_lin_result_limbs(const tr_lin_result *r, int32_t part) {
    if (!res_valid(r) || part < 0 || part >= TR_LIN_PARTS) return 0;
    return tr_lin_q_limbs((const tr_bigq *const *)r->q[part], r->nq[part]);
}
int32_t tr_lin_result_export(const tr_lin_result *r, int32_t part, tr_lin_qbuf *out) {
    if (!res_valid(r) || part < 0 || part >= TR_LIN_PARTS || !r->q[part]) return TR_MATH_INVALID;
    return tr_lin_q_export((const tr_bigq *const *)r->q[part], r->nq[part], out);
}
int32_t tr_lin_result_doubles(const tr_lin_result *r, int32_t part, double *out, uint64_t capacity, uint64_t *count) {
    if (!res_valid(r) || part < 0 || part >= TR_LIN_PARTS || !r->d[part] || capacity < r->nd[part] || (r->nd[part] && !out))
        return TR_MATH_INVALID;
    memcpy(out, r->d[part], (size_t)r->nd[part] * sizeof(double));
    if (count) *count = r->nd[part];
    return TR_MATH_OK;
}
const tr_bigq *const *tr_lin_result_part(const tr_lin_result *r, int32_t part, uint64_t *count) {
    if (!res_valid(r) || part < 0 || part >= TR_LIN_PARTS || !r->q[part]) { if (count) *count = 0; return NULL; }
    if (count) *count = r->nq[part];
    return (const tr_bigq *const *)r->q[part];
}
static tr_lin_result *res_done(lctx *x, tr_lin_result *r, int32_t *error) {
    if (x->rc) { tr_lin_result_free(r); set_error(error, x->rc); return NULL; }
    set_error(error, TR_MATH_OK);
    return r;
}

/* ================================================================ LIN-001: moment states */
struct tr_lin_moments_s {
    uint32_t magic;
    int32_t k, nslots, ngram, marginals, source, words, scoped;
    tr_lin_spec spec;
    int32_t col[TR_LIN_MAX_COLS], scale[TR_LIN_MAX_COLS];
    uint64_t mag[TR_LIN_MAX_COLS];
    uint64_t ngroups, rows, present, bytes;
    uint64_t digest[2];
    tr_lin_slot *slots;
    uint64_t *grows, *count, *sum;
};
static int mom_valid(const tr_lin_moments *s) { return s && s->magic == MOM_MAGIC; }
static uint64_t mom_bytes(uint64_t G, int32_t S) {
    return tr_budget_add(tr_budget_add(sizeof(tr_lin_moments), tr_budget_mul((uint64_t)S, sizeof(tr_lin_slot))),
                         tr_lin_state_bytes(G, S));
}
void tr_lin_moments_free(tr_lin_moments *s) {
    if (!mom_valid(s)) return;
    free(s->slots); free(s->grows); free(s->count); free(s->sum);
    s->magic = 0;
    free(s);
}
static tr_lin_moments *mom_alloc(uint64_t G, const tr_lin_spec *spec, const int32_t *col, int32_t k, int32_t marginals) {
    tr_lin_moments *s = (tr_lin_moments *)zalloc(sizeof *s);
    if (!s) return NULL;
    int32_t S = tr_lin_slot_count(k, spec->policy, marginals);
    s->spec = *spec; s->k = k; s->nslots = S; s->ngram = k * (k + 1) / 2; s->marginals = marginals;
    s->ngroups = G;
    for (int32_t i = 0; i < k; i++) s->col[i] = col[i];
    s->slots = (tr_lin_slot *)zalloc((uint64_t)S * sizeof(tr_lin_slot));
    s->grows = (uint64_t *)zalloc(G * 8);
    s->count = (uint64_t *)zalloc(G * (uint64_t)S * 8);
    s->sum = (uint64_t *)zalloc(G * (uint64_t)S * 8 * TR_LIN_WORDS);
    if (!s->slots || !s->grows || !s->count || !s->sum) { free(s->slots); free(s->grows); free(s->count); free(s->sum); free(s); return NULL; }
    tr_lin_layout(k, spec->policy, marginals, s->slots);
    s->bytes = mom_bytes(G, S);
    s->magic = MOM_MAGIC;
    return s;
}
static void mom_scope(tr_lin_moments *s, const tr_scope *scope, const tr_scope_view *v) {
    for (int32_t i = 0; i < s->k; i++) {
        s->scale[i] = i ? v->col[s->col[i]].scale : 0;
        s->mag[i] = tr_lin_magnitude(v, s->col, i);
    }
    tr_scope_info info;
    tr_scope_get_info(scope, &info);
    s->scoped = 1; s->digest[0] = info.digest[0]; s->digest[1] = info.digest[1];
}
static void mom_tally(tr_lin_moments *s) {
    uint64_t rows = 0, present = 0;
    for (uint64_t g = 0; g < s->ngroups; g++) { rows += s->grows[g]; present += s->grows[g] != 0; }
    s->rows = rows; s->present = present;
}
/* listwise: every slot's count is the complete rows (the constant slot's sum) */
static void mom_listwise_counts(tr_lin_moments *s) {
    if (s->spec.policy != TR_LIN_LISTWISE) return;
    const uint64_t S = (uint64_t)s->nslots;
    for (uint64_t g = 0; g < s->ngroups; g++) {
        uint64_t n = s->sum[(g * S) * TR_LIN_WORDS];
        for (uint64_t j = 0; j < S; j++) s->count[g * S + j] = n;
    }
}

/* the CPU needs: the state, the accumulators (16 bytes per slot and group with 128-bit words), the object */
static void cpu_need(const tr_scope_view *v, const int32_t *col, int32_t k, int32_t S, int32_t words, tr_math_need *need) {
    tr_lin_device_need(v, col, k, S, words, need);
    uint64_t acc = words == 2 ? tr_budget_mul(tr_budget_mul(v->ngroups, (uint64_t)S), 16) : 0;
    need->need[TR_BUDGET_BYTES] = tr_budget_add(mom_bytes(v->ngroups, S), acc);
}
static void adopt_need(const tr_scope_view *v, const int32_t *col, int32_t k, int32_t S, tr_math_need *need) {
    tr_lin_device_need(v, col, k, S, TR_LIN_WORDS, need);
    need->need[TR_BUDGET_BYTES] = mom_bytes(v->ngroups, S);
}

int32_t tr_lin_moments_need(const tr_scope *scope, const tr_lin_spec *spec, int32_t target, tr_math_need *need) {
    tr_scope_view v;
    if (!scope || !need || tr_scope_get_view(scope, &v) ||
        (target != TR_LIN_NEED_CPU && target != TR_LIN_NEED_DEVICE && target != TR_LIN_NEED_ADOPT)) return TR_MATH_INVALID;
    int32_t col[TR_LIN_MAX_COLS], k = tr_lin_columns(spec, v.ncols, col);
    if (k < 0) return k;
    int32_t marginals = spec->policy == TR_LIN_PAIRWISE, S = tr_lin_slot_count(k, spec->policy, marginals);
    tr_lin_slot slots[TR_LIN_MAX_SLOTS];
    tr_lin_layout(k, spec->policy, marginals, slots);
    int32_t words = tr_lin_words(spec->width, tr_lin_needs_wide(&v, col, k, slots, S));
    if (words < 0) words = 3;
    if (target == TR_LIN_NEED_DEVICE) tr_lin_device_need(&v, col, k, S, words, need);
    else if (target == TR_LIN_NEED_ADOPT) adopt_need(&v, col, k, S, need);
    else cpu_need(&v, col, k, S, words, need);
    return TR_MATH_OK;
}

/* every selected row: bag multiplicity, then the valid slots (sums proven to fit their words) */
static void accumulate(tr_lin_moments *s, const tr_scope_view *v, __int128 *acc2) {
    const int32_t k = s->k, S = s->nslots, words = s->words, listwise = s->spec.policy == TR_LIN_LISTWISE;
    const tr_lin_slot *slots = s->slots;
    const uint8_t *valid[TR_LIN_MAX_COLS];
    uint32_t full = 1;
    for (int32_t i = 0; i < k; i++) valid[i] = i ? v->col[s->col[i]].valid : NULL;
    for (int32_t i = 1; i < k; i++) full |= (uint32_t)1 << i;
    int64_t z[TR_LIN_MAX_COLS];
    z[0] = 1;
    const uint64_t nw = (v->rows + 63) / 64;
    for (uint64_t w = 0; w < nw; w++) {
        uint64_t bits = v->selection[w];
        while (bits) {
            uint64_t r = (w << 6) | (uint64_t)__builtin_ctzll(bits);
            bits &= bits - 1;
            uint64_t g = tr_scope_group(v, r);
            s->grows[g]++;
            uint32_t mask = 1;
            for (int32_t i = 1; i < k; i++) if (!valid[i] || valid[i][r]) mask |= (uint32_t)1 << i;
            if (listwise && mask != full) continue;
            for (int32_t i = 1; i < k; i++) if ((mask >> i) & 1) z[i] = tr_scope_value(v, s->col[i], r);
            const uint64_t base = g * (uint64_t)S;
            if (words == 2) {
                __int128 *a = acc2 + base;
                if (listwise) for (int32_t j = 0; j < S; j++) a[j] += (__int128)z[slots[j].a] * z[slots[j].b];
                else
                    for (int32_t j = 0; j < S; j++) {
                        if ((mask & (slots[j].need | 1)) != (slots[j].need | 1)) continue;
                        a[j] += (__int128)z[slots[j].a] * z[slots[j].b];
                        s->count[base + (uint64_t)j]++;
                    }
            } else {
                uint64_t *a = s->sum + base * TR_LIN_WORDS;
                for (int32_t j = 0; j < S; j++) {
                    if (!listwise) {
                        if ((mask & (slots[j].need | 1)) != (slots[j].need | 1)) continue;
                        s->count[base + (uint64_t)j]++;
                    }
                    tr_lin_w3_add(a + (uint64_t)j * TR_LIN_WORDS, (__int128)z[slots[j].a] * z[slots[j].b]);
                }
            }
        }
    }
}

tr_lin_moments *tr_lin_moments_build(const tr_scope *scope, const tr_lin_spec *spec, const tr_math_budget *budget,
                                     tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    tr_scope_view v;
    if (!scope || tr_scope_get_view(scope, &v)) { set_error(error, TR_MATH_INVALID); return NULL; }
    int32_t col[TR_LIN_MAX_COLS], k = tr_lin_columns(spec, v.ncols, col);
    if (k < 0) { set_error(error, k); return NULL; }
    int32_t marginals = spec->policy == TR_LIN_PAIRWISE, S = tr_lin_slot_count(k, spec->policy, marginals);
    tr_lin_slot slots[TR_LIN_MAX_SLOTS];
    tr_lin_layout(k, spec->policy, marginals, slots);
    int32_t words = tr_lin_words(spec->width, tr_lin_needs_wide(&v, col, k, slots, S));
    tr_math_need need;
    cpu_need(&v, col, k, S, words < 0 ? 3 : words, &need);
    tr_math_budget dflt;
    int32_t rc = tr_budget_check(budget_or(budget, &dflt), &need, refusal);
    if (!rc && words < 0) rc = words;                                     /* width 128 without a proof */
    if (rc) { set_error(error, rc); return NULL; }
    tr_lin_moments *s = mom_alloc(v.ngroups, spec, col, k, marginals);
    __int128 *acc2 = s && words == 2 ? (__int128 *)zalloc(v.ngroups * (uint64_t)S * 16) : NULL;
    if (!s || (words == 2 && !acc2)) { tr_lin_moments_free(s); set_error(error, TR_MATH_MEMORY); return NULL; }
    s->words = words; s->source = TR_LIN_SOURCE_FUSED;
    mom_scope(s, scope, &v);
    if (v.selected) accumulate(s, &v, acc2);
    if (acc2) {
        for (uint64_t i = 0; i < v.ngroups * (uint64_t)S; i++) tr_lin_w3_set(s->sum + i * TR_LIN_WORDS, acc2[i]);
        free(acc2);
    }
    mom_listwise_counts(s);
    mom_tally(s);
    set_error(error, TR_MATH_OK);
    return s;
}

/* |w| (192-bit two's complement) <= bound (unsigned 128-bit, UINT128 max = no check) */
static int w3_within(const uint64_t *w, unsigned __int128 bound) {
    int neg = (w[2] >> 63) != 0;
    uint64_t m0 = w[0], m1 = w[1], m2 = w[2];
    if (neg) {                                     /* magnitude = -w */
        m0 = ~m0; m1 = ~m1; m2 = ~m2;
        m0 += 1; if (!m0) { m1 += 1; if (!m1) m2 += 1; }
    }
    if (bound == ~(unsigned __int128)0) return 1;
    if (m2) return 0;
    return (((unsigned __int128)m1 << 64) | m0) <= bound;
}
static int w3_is_zero(const uint64_t *w) { return !w[0] && !w[1] && !w[2]; }
static int w3_fits128(const uint64_t *w) { return w[2] == ((w[1] >> 63) ? ~(uint64_t)0 : 0); }
static void w3_addsub(uint64_t *r, const uint64_t *a, const uint64_t *b, int sub) {
    uint64_t c = sub ? 1 : 0;
    for (int i = 0; i < 3; i++) {
        uint64_t y = sub ? ~b[i] : b[i];
        unsigned __int128 t = (unsigned __int128)a[i] + y + c;
        r[i] = (uint64_t)t; c = (uint64_t)(t >> 64);
    }
}

tr_lin_moments *tr_lin_moments_from_states(const tr_scope *origin, const tr_lin_spec *spec, int32_t marginals,
                                           const tr_lin_states *st, const tr_math_budget *budget, tr_math_refusal *refusal,
                                           int32_t *error) {
    tr_budget_none(refusal);
    tr_scope_view v;
    if (!origin || !st || tr_scope_get_view(origin, &v)) { set_error(error, TR_MATH_INVALID); return NULL; }
    int32_t col[TR_LIN_MAX_COLS], k = tr_lin_columns(spec, v.ncols, col);
    if (k < 0) { set_error(error, k); return NULL; }
    marginals = spec->policy == TR_LIN_PAIRWISE && marginals;
    int32_t S = tr_lin_slot_count(k, spec->policy, marginals);
    if (st->ngroups != v.ngroups || st->nslots != S || st->words != TR_LIN_WORDS || !st->rows || !st->count || !st->sum) {
        set_error(error, TR_MATH_INVALID); return NULL;
    }
    tr_lin_slot slots[TR_LIN_MAX_SLOTS];
    tr_lin_layout(k, spec->policy, marginals, slots);
    tr_math_need need;
    adopt_need(&v, col, k, S, &need);
    tr_math_budget dflt;
    int32_t rc = tr_budget_check(budget_or(budget, &dflt), &need, refusal);
    if (rc) { set_error(error, rc); return NULL; }
    int32_t wide = tr_lin_needs_wide(&v, col, k, slots, S);
    if (tr_lin_words(spec->width, wide) < 0) { set_error(error, TR_MATH_OVERFLOW); return NULL; }
    unsigned __int128 bound[TR_LIN_MAX_SLOTS];
    for (int32_t j = 0; j < S; j++)
        bound[j] = tr_bank_satmul(tr_lin_magnitude(&v, col, slots[j].a), tr_lin_magnitude(&v, col, slots[j].b),
                                  ~(unsigned __int128)0 - 1);
    uint64_t total = 0;
    for (uint64_t g = 0; g < st->ngroups; g++) {
        uint64_t rows = st->rows[g];
        if (__builtin_add_overflow(total, rows, &total)) { set_error(error, TR_MATH_INVALID); return NULL; }
        const uint64_t *cnt = st->count + g * (uint64_t)S, *sm = st->sum + g * (uint64_t)S * TR_LIN_WORDS;
        for (int32_t j = 0; j < S; j++) {
            const uint64_t *w = sm + (uint64_t)j * TR_LIN_WORDS;
            uint64_t c = cnt[j];
            unsigned __int128 b = bound[j] > (~(unsigned __int128)0 - 1) / (c ? c : 1) ? ~(unsigned __int128)0 : bound[j] * c;
            int bad = c > rows || (!c && !w3_is_zero(w)) || !w3_within(w, b);
            if (spec->policy == TR_LIN_LISTWISE) bad = bad || c != cnt[0];
            else if (j == 0) bad = bad || c != rows;
            else if (slots[j].kind) {                                  /* a marginal: the count of its pair */
                int32_t pa = __builtin_ctz(slots[j].need), pb = 31 - __builtin_clz(slots[j].need);
                bad = bad || c != cnt[tr_lin_gram_slot(k, pa, pb)];
            }
            if (j == 0) bad = bad || w[0] != c || w[1] || w[2];
            if (bad) { set_error(error, TR_MATH_INVALID); return NULL; }
        }
    }
    if (total != v.selected) { set_error(error, TR_MATH_INVALID); return NULL; }
    tr_lin_moments *s = mom_alloc(v.ngroups, spec, col, k, marginals);
    if (!s) { set_error(error, TR_MATH_MEMORY); return NULL; }
    memcpy(s->grows, st->rows, (size_t)v.ngroups * 8);
    memcpy(s->count, st->count, (size_t)(v.ngroups * (uint64_t)S) * 8);
    memcpy(s->sum, st->sum, (size_t)(v.ngroups * (uint64_t)S) * 8 * TR_LIN_WORDS);
    s->words = tr_lin_words(spec->width, wide); s->source = TR_LIN_SOURCE_ADOPTED;
    mom_scope(s, origin, &v);
    mom_tally(s);
    set_error(error, TR_MATH_OK);
    return s;
}

tr_lin_moments *tr_lin_moments_from_bank(const tr_scope *scope, const tr_bank *bank, const tr_lin_spec *spec,
                                         const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    tr_scope_view v;
    tr_scope_info si;
    tr_bank_info bi;
    tr_bank_states bs;
    if (!scope || !bank || tr_scope_get_view(scope, &v) || tr_scope_get_info(scope, &si) || tr_bank_get_info(bank, &bi) ||
        tr_bank_get_states(bank, &bs)) { set_error(error, TR_MATH_INVALID); return NULL; }
    int32_t col[TR_LIN_MAX_COLS], k = tr_lin_columns(spec, v.ncols, col);
    if (k < 0) { set_error(error, k); return NULL; }
    if (!bi.scoped || bi.digest[0] != si.digest[0] || bi.digest[1] != si.digest[1] || bi.ncols != v.ncols ||
        bi.ngroups != v.ngroups) { set_error(error, TR_MATH_DOMAIN_ERROR); return NULL; }
    uint32_t used = 0;
    for (int32_t i = 1; i < k; i++) used |= (uint32_t)1 << col[i];
    if ((spec->policy == TR_LIN_LISTWISE && (bi.mode != TR_BANK_JOINT || bi.joint != used)) ||
        (spec->policy == TR_LIN_PAIRWISE && bi.mode != TR_BANK_SQL)) { set_error(error, TR_MATH_DOMAIN_ERROR); return NULL; }
    int32_t S = tr_lin_slot_count(k, spec->policy, 0);
    tr_math_need need;
    adopt_need(&v, col, k, S, &need);
    tr_math_budget dflt;
    int32_t rc = tr_budget_check(budget_or(budget, &dflt), &need, refusal);
    if (rc) { set_error(error, rc); return NULL; }
    uint64_t nexp = (uint64_t)bi.nterms * (uint64_t)bi.ncols;
    uint8_t *exps = (uint8_t *)zalloc(nexp);
    if (!exps) { set_error(error, TR_MATH_MEMORY); return NULL; }
    tr_bank_get_exponents(bank, exps, nexp);
    tr_lin_slot slots[TR_LIN_MAX_SLOTS];
    tr_lin_layout(k, spec->policy, 0, slots);
    int32_t term[TR_LIN_MAX_SLOTS];
    for (int32_t j = 0; j < S; j++) {
        uint8_t want[TR_SCOPE_MAX_COLUMNS];
        memset(want, 0, sizeof want);
        if (slots[j].a) want[col[slots[j].a]]++;
        if (slots[j].b) want[col[slots[j].b]]++;
        term[j] = -1;
        for (int32_t t = 0; t < bi.nterms && term[j] < 0; t++)
            if (!memcmp(exps + (uint64_t)t * (uint64_t)bi.ncols, want, (size_t)bi.ncols)) term[j] = t;
        if (term[j] < 0) { free(exps); set_error(error, TR_MATH_INVALID); return NULL; }
    }
    free(exps);
    tr_lin_moments *s = mom_alloc(v.ngroups, spec, col, k, 0);
    if (!s) { set_error(error, TR_MATH_MEMORY); return NULL; }
    const uint64_t m = (uint64_t)bi.nterms;
    for (uint64_t g = 0; g < v.ngroups; g++) {
        s->grows[g] = bs.rows[g];
        for (int32_t j = 0; j < S; j++) {
            uint64_t i = g * (uint64_t)S + (uint64_t)j, t = g * m + (uint64_t)term[j];
            s->count[i] = bs.count[t];
            tr_lin_w3_set(s->sum + i * TR_LIN_WORDS, tr_i128_get(bs.sum[t]));
        }
    }
    s->words = 2; s->source = TR_LIN_SOURCE_BANK;
    mom_scope(s, scope, &v);
    mom_tally(s);
    set_error(error, TR_MATH_OK);
    return s;
}

tr_lin_moments *tr_lin_moments_combine(const tr_lin_moments *a, const tr_lin_moments *b, int32_t sign,
                                       const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    if (!mom_valid(a) || !mom_valid(b) || (sign != 1 && sign != -1)) { set_error(error, TR_MATH_INVALID); return NULL; }
    if (memcmp(&a->spec, &b->spec, sizeof a->spec) || a->k != b->k || a->nslots != b->nslots ||
        a->marginals != b->marginals || a->ngroups != b->ngroups || memcmp(a->col, b->col, sizeof a->col) ||
        memcmp(a->scale, b->scale, sizeof a->scale)) { set_error(error, TR_MATH_DOMAIN_ERROR); return NULL; }
    tr_math_need need;
    memset(&need, 0, sizeof need);
    need.need[TR_BUDGET_STATES] = tr_budget_mul(a->ngroups, (uint64_t)a->nslots);
    need.need[TR_BUDGET_OUTPUT] = a->ngroups;
    need.need[TR_BUDGET_BYTES] = mom_bytes(a->ngroups, a->nslots);
    tr_math_budget dflt;
    int32_t rc = tr_budget_check(budget_or(budget, &dflt), &need, refusal);
    if (rc) { set_error(error, rc); return NULL; }
    tr_lin_moments *s = mom_alloc(a->ngroups, &a->spec, a->col, a->k, a->marginals);
    if (!s) { set_error(error, TR_MATH_MEMORY); return NULL; }
    const uint64_t S = (uint64_t)a->nslots;
    int words = a->words > b->words ? a->words : b->words;
    for (uint64_t g = 0; g < a->ngroups && !rc; g++) {
        if (sign > 0 ? __builtin_add_overflow(a->grows[g], b->grows[g], &s->grows[g]) : b->grows[g] > a->grows[g]) rc = TR_MATH_INVALID;
        if (sign < 0 && !rc) s->grows[g] = a->grows[g] - b->grows[g];
        for (uint64_t j = 0; j < S && !rc; j++) {
            uint64_t i = g * S + j;
            if (sign > 0 ? __builtin_add_overflow(a->count[i], b->count[i], &s->count[i]) : b->count[i] > a->count[i]) rc = TR_MATH_INVALID;
            if (sign < 0 && !rc) s->count[i] = a->count[i] - b->count[i];
            w3_addsub(s->sum + i * TR_LIN_WORDS, a->sum + i * TR_LIN_WORDS, b->sum + i * TR_LIN_WORDS, sign < 0);
            if (!w3_fits128(s->sum + i * TR_LIN_WORDS)) words = 3;
            if (!s->count[i] && !w3_is_zero(s->sum + i * TR_LIN_WORDS)) rc = TR_MATH_INVALID;
        }
    }
    if (rc) { tr_lin_moments_free(s); set_error(error, rc); return NULL; }
    for (int32_t i = 0; i < a->k; i++) { s->scale[i] = a->scale[i]; s->mag[i] = 0; }
    s->words = words; s->source = TR_LIN_SOURCE_COMBINED;
    mom_tally(s);
    set_error(error, TR_MATH_OK);
    return s;
}

int32_t tr_lin_moments_get_info(const tr_lin_moments *s, tr_lin_info *out) {
    if (!mom_valid(s) || !out) return TR_MATH_INVALID;
    memset(out, 0, sizeof *out);
    out->ngroups = s->ngroups; out->present = s->present; out->rows = s->rows; out->bytes = s->bytes;
    out->digest[0] = s->digest[0]; out->digest[1] = s->digest[1];
    out->k = s->k; out->nslots = s->nslots; out->ngram = s->ngram; out->nfeatures = s->spec.nfeatures;
    out->target = s->spec.target; out->policy = s->spec.policy; out->intercept = s->spec.intercept;
    out->width = s->words == 2 ? 128 : 192;
    out->marginals = s->marginals; out->source = s->source; out->scoped = s->scoped;
    for (int32_t i = 0; i < TR_LIN_MAX_COLS; i++) { out->column[i] = i < s->k ? s->col[i] : -1; out->scale[i] = i < s->k ? s->scale[i] : 0; }
    return TR_MATH_OK;
}
int32_t tr_lin_moments_get_states(const tr_lin_moments *s, tr_lin_states *out) {
    if (!mom_valid(s) || !out) return TR_MATH_INVALID;
    memset(out, 0, sizeof *out);
    out->ngroups = s->ngroups; out->nslots = s->nslots; out->words = TR_LIN_WORDS;
    out->rows = s->grows; out->count = s->count; out->sum = s->sum;
    return TR_MATH_OK;
}
int32_t tr_lin_moments_get_slots(const tr_lin_moments *s, tr_lin_slot *out, int32_t capacity) {
    if (!mom_valid(s) || !out || capacity < s->nslots) return TR_MATH_INVALID;
    memcpy(out, s->slots, (size_t)s->nslots * sizeof *out);
    return s->nslots;
}
int32_t tr_lin_moments_equal(const tr_lin_moments *a, const tr_lin_moments *b) {
    if (!mom_valid(a) || !mom_valid(b)) return TR_MATH_INVALID;
    if (memcmp(&a->spec, &b->spec, sizeof a->spec) || a->nslots != b->nslots || a->ngroups != b->ngroups ||
        a->marginals != b->marginals) return 0;
    uint64_t n = a->ngroups * (uint64_t)a->nslots;
    return !memcmp(a->grows, b->grows, (size_t)a->ngroups * 8) && !memcmp(a->count, b->count, (size_t)n * 8) &&
           !memcmp(a->sum, b->sum, (size_t)n * 8 * TR_LIN_WORDS);
}

/* the sum of slot j of group g as a big integer */
static tr_bigint *mom_sum(lctx *x, const tr_lin_moments *s, uint64_t g, int32_t j) {
    return Z_twos(x, s->sum + (g * (uint64_t)s->nslots + (uint64_t)j) * TR_LIN_WORDS, TR_LIN_WORDS);
}
static uint64_t mom_count(const tr_lin_moments *s, uint64_t g, int32_t j) { return s->count[g * (uint64_t)s->nslots + (uint64_t)j]; }
static tr_bigint *gram_z(lctx *x, const tr_lin_moments *s, uint64_t g, int32_t a, int32_t b) {
    return mom_sum(x, s, g, a <= b ? tr_lin_gram_slot(s->k, a, b) : tr_lin_gram_slot(s->k, b, a));
}
static tr_bigq *gram_q(lctx *x, const tr_lin_moments *s, uint64_t g, int32_t a, int32_t b) {   /* value units */
    tr_bigint *z = gram_z(x, s, g, a, b);
    tr_bigq *q = Q_scale10(x, z, s->scale[a] + s->scale[b]);
    tr_bigint_free(z);
    return q;
}

/* ================================================================ LIN-002: exact systems */
/* the solution set in canonical form (owned) */
typedef struct {
    int32_t kind, rank, nfree;
    uint64_t pivots;
    tr_bigq **x, **null;           /* n; nfree x n */
} zsol;
static void zsol_free(zsol *z, int32_t n) {
    tr_lin_q_free(z->x, z->x ? (uint64_t)n : 0);
    tr_lin_q_free(z->null, z->null ? (uint64_t)z->nfree * (uint64_t)n : 0);
    z->x = NULL; z->null = NULL;
}

/* row i scaled by the lcm of its denominators: M (m x (n+1)) and the scales L (m); b may be NULL (zero) */
static void scale_rows(lctx *x, int32_t m, int32_t n, const tr_bigq *const *A, const tr_bigq *const *b, tr_bigint **M,
                       tr_bigint **L) {
    const int32_t c = n + 1;
    for (int32_t i = 0; i < m && LIVE(x); i++) {
        tr_bigint *l = Z_i64(x, 1);
        for (int32_t j = 0; j < c && LIVE(x); j++) {
            const tr_bigq *e = j < n ? A[(uint64_t)i * n + j] : b ? b[i] : NULL;
            if (!e) continue;
            const tr_bigint *d = tr_bigq_denominator(e);
            if (Z_is_one(d)) continue;
            tr_bigint *g = Z_gcd(x, l, d), *t = Z_div(x, l, g), *nl = Z_mul(x, t, d);
            tr_bigint_free(g); tr_bigint_free(t);
            if (nl) { tr_bigint_free(l); l = nl; }
        }
        for (int32_t j = 0; j < c && LIVE(x); j++) {
            const tr_bigq *e = j < n ? A[(uint64_t)i * n + j] : b ? b[i] : NULL;
            if (!e || !tr_bigq_sign(e)) { M[(uint64_t)i * c + j] = Z_i64(x, 0); continue; }
            const tr_bigint *d = tr_bigq_denominator(e);
            if (Z_is_one(d)) { M[(uint64_t)i * c + j] = Z_mul(x, tr_bigq_numerator(e), l); continue; }
            tr_bigint *t = Z_div(x, l, d);
            M[(uint64_t)i * c + j] = Z_mul(x, tr_bigq_numerator(e), t);
            tr_bigint_free(t);
        }
        L[i] = l;
    }
}

/* fraction-free Gauss-Jordan elimination (Bareiss) in place: returns the rank r (pivcol[0..r)), *det = the last pivot;
 * afterwards pivot rows 0..r-1 hold det * RREF and the other rows are zero */
static int32_t bareiss(lctx *x, tr_bigint **M, int32_t m, int32_t c, int32_t *pivcol, tr_bigint **det) {
    tr_bigint *prev = Z_i64(x, 1);
    int32_t r = 0;
    for (int32_t col = 0; col < c && r < m && LIVE(x); col++) {
        int32_t i = r;
        while (i < m && !Z_sign(M[(uint64_t)i * c + col])) i++;
        if (i == m) continue;
        if (i != r)
            for (int32_t j = 0; j < c; j++) {
                tr_bigint *t = M[(uint64_t)i * c + j];
                M[(uint64_t)i * c + j] = M[(uint64_t)r * c + j];
                M[(uint64_t)r * c + j] = t;
            }
        const tr_bigint *piv = M[(uint64_t)r * c + col];
        int one = Z_is_one(prev);
        for (int32_t i2 = 0; i2 < m && LIVE(x); i2++) {
            if (i2 == r) continue;
            tr_bigint **row = M + (uint64_t)i2 * c, **prow = M + (uint64_t)r * c;
            const tr_bigint *f = row[col];
            for (int32_t j = 0; j < c && LIVE(x); j++) {
                if (j == col) continue;
                if (!Z_sign(row[j]) && (!Z_sign(f) || !Z_sign(prow[j]))) continue;     /* stays 0 */
                tr_bigint *t = Z_cross(x, piv, row[j], f, prow[j]);
                tr_bigint *nv = one ? t : Z_div(x, t, prev);
                if (!one) tr_bigint_free(t);
                if (!nv) break;
                tr_bigint_free(row[j]); row[j] = nv;
            }
            tr_bigint *zero = Z_i64(x, 0);
            if (zero) { tr_bigint_free(row[col]); row[col] = zero; }
        }
        tr_bigint *np = Z_copy(x, piv);
        if (np) { tr_bigint_free(prev); prev = np; }
        pivcol[r++] = col;
    }
    *det = prev;
    return r;
}

/* the canonical solution set from D and the det-scaled RREF rows DR[t * c + j] (t < r; only non-pivot columns are
 * read) of the integer system Mint (m x c), checked: RREF form (zeros left of each pivot), A (D x0) = D b and
 * A (D N_f) = 0. TR_MATH_INVALID when a check fails. */
static int32_t assemble(lctx *x, tr_bigint *const *Mint, int32_t m, int32_t c, int32_t r, const int32_t *pivcol,
                        const tr_bigint *D, tr_bigint *const *DR, zsol *out) {
    const int32_t n = c - 1;
    memset(out, 0, sizeof *out);
    uint64_t pivots = 0;
    int ispiv[64] = {0};
    for (int32_t t = 0; t < r; t++) { pivots |= (uint64_t)1 << pivcol[t]; ispiv[pivcol[t]] = 1; }
    out->pivots = pivots & ~((uint64_t)1 << n);
    if (r && pivcol[r - 1] == n) {                         /* the right-hand side is a pivot: inconsistent */
        out->kind = TR_LIN_NONE; out->rank = r - 1; out->nfree = n - (r - 1);
        return TR_MATH_OK;
    }
    out->kind = r == n ? TR_LIN_UNIQUE : TR_LIN_FAMILY; out->rank = r; out->nfree = n - r;
    for (int32_t t = 0; t < r; t++)                          /* the RREF form: nothing left of the pivot */
        for (int32_t j = 0; j < pivcol[t]; j++) if (!ispiv[j] && Z_sign(DR[(uint64_t)t * c + j])) return TR_MATH_INVALID;
    /* verification: per equation, sum_t A[i][p_t] DR[t][j] == D * M[i][j] for every non-pivot column j (j = n: b) */
    const tr_bigint *a[64], *y[64];
    for (int32_t j = 0; j < c && LIVE(x); j++) {
        if (ispiv[j]) continue;
        for (int32_t i = 0; i < m && LIVE(x); i++) {
            for (int32_t t = 0; t < r; t++) { a[t] = Mint[(uint64_t)i * c + pivcol[t]]; y[t] = DR[(uint64_t)t * c + j]; }
            tr_bigint *lhs = r ? Z_dot(x, a, y, (uint64_t)r) : Z_i64(x, 0), *rhs = Z_mul(x, D, Mint[(uint64_t)i * c + j]);
            int ok = LIVE(x) && tr_bigint_cmp(lhs, rhs) == 0;
            tr_bigint_free(lhs); tr_bigint_free(rhs);
            if (LIVE(x) && !ok) return TR_MATH_INVALID;
        }
    }
    if (x->rc) return x->rc;
    out->x = Q_arr(x, (uint64_t)n);
    out->null = out->nfree ? Q_arr(x, (uint64_t)out->nfree * (uint64_t)n) : NULL;
    for (int32_t j = 0; j < n && LIVE(x); j++) out->x[j] = Q_int(x, 0);
    for (int32_t t = 0; t < r && LIVE(x); t++) {
        tr_bigq_free(out->x[pivcol[t]]);
        out->x[pivcol[t]] = Q_frac(x, DR[(uint64_t)t * c + n], D);
    }
    int32_t f = 0;
    for (int32_t j = 0; j < n && LIVE(x); j++) {
        if (ispiv[j]) continue;
        tr_bigq **v = out->null + (uint64_t)f * n;
        for (int32_t i = 0; i < n && LIVE(x); i++) v[i] = Q_int(x, i == j);
        for (int32_t t = 0; t < r && LIVE(x); t++) {
            tr_bigint *neg = Z_neg(x, DR[(uint64_t)t * c + j]);
            tr_bigq_free(v[pivcol[t]]);
            v[pivcol[t]] = Q_frac(x, neg, D);
            tr_bigint_free(neg);
        }
        f++;
    }
    if (x->rc) { zsol_free(out, n); return x->rc; }
    return TR_MATH_OK;
}

/* the exact solution set of the integer system Mint (m x (n+1)) by Bareiss */
static int32_t int_solve(lctx *x, tr_bigint *const *Mint, int32_t m, int32_t n, zsol *out) {
    const int32_t c = n + 1;
    uint64_t cells = (uint64_t)m * (uint64_t)c;
    tr_bigint **W = Z_arr(x, cells);
    for (uint64_t i = 0; W && i < cells && LIVE(x); i++) W[i] = Z_copy(x, Mint[i]);
    int32_t pivcol[64], r = 0;
    tr_bigint *D = NULL;
    if (LIVE(x)) r = bareiss(x, W, m, c, pivcol, &D);
    int32_t rc = x->rc ? x->rc : assemble(x, Mint, m, c, r, pivcol, D, W, out);
    tr_bigint_free(D);
    Z_free_all(W, W ? cells : 0);
    if (rc && !x->rc) lfail(x, rc);
    return rc;
}

/* y with A'^T y = 0, b'^T y = 1 for an inconsistent integer system (the Fredholm certificate), mapped to the
 * original rows: y_i = L_i y'_i */
static tr_bigq **certificate(lctx *x, tr_bigint *const *Mint, tr_bigint *const *L, int32_t m, int32_t n) {
    const int32_t c = n + 1, tc = m + 1;
    tr_bigint **T = Z_arr(x, (uint64_t)c * (uint64_t)tc);
    for (int32_t j = 0; T && j < c && LIVE(x); j++) {
        for (int32_t i = 0; i < m && LIVE(x); i++) T[(uint64_t)j * tc + i] = Z_copy(x, Mint[(uint64_t)i * c + j]);
        T[(uint64_t)j * tc + m] = Z_i64(x, j == n);
    }
    zsol z;
    memset(&z, 0, sizeof z);
    if (LIVE(x)) int_solve(x, T, c, m, &z);
    Z_free_all(T, T ? (uint64_t)c * (uint64_t)tc : 0);
    if (LIVE(x) && z.kind == TR_LIN_NONE) lfail(x, TR_MATH_INVALID);      /* the system was consistent after all */
    tr_bigq **y = Q_arr(x, (uint64_t)m);
    for (int32_t i = 0; y && i < m && LIVE(x); i++) {
        tr_bigq *l = Q_z(x, L[i]);
        y[i] = Q_mul(x, z.x[i], l);
        tr_bigq_free(l);
    }
    zsol_free(&z, m);
    if (x->rc) { tr_lin_q_free(y, y ? (uint64_t)m : 0); return NULL; }
    return y;
}

static void sol_result(lctx *x, tr_lin_result *res, zsol *z, int32_t m, int32_t n, int32_t method) {
    if (x->rc) return;
    res->info.status = z->kind; res->info.rows = m; res->info.cols = n; res->info.rank = z->rank;
    res->info.nullity = n - z->rank; res->info.pivots = z->pivots; res->info.method = method; res->info.certified = 1;
    if (z->x) { res_q(res, TR_LIN_PART_X, z->x, (uint64_t)n); z->x = NULL; }
    if (z->null) { res_q(res, TR_LIN_PART_NULL, z->null, (uint64_t)z->nfree * (uint64_t)n); z->null = NULL; }
}

static int dims_ok(int32_t m, int32_t n) { return m >= 1 && m <= TR_LIN_MAX_ROWS && n >= 1 && n <= TR_LIN_MAX_UNKNOWNS; }
static int32_t solve_need(int32_t m, int32_t n, const tr_math_budget *budget, tr_math_refusal *refusal) {
    tr_math_need need;
    memset(&need, 0, sizeof need);
    need.need[TR_BUDGET_VARIABLES] = (uint64_t)n;
    need.need[TR_BUDGET_TENSOR] = (uint64_t)m * (uint64_t)(n + 1);
    need.need[TR_BUDGET_OUTPUT] = (uint64_t)n * (uint64_t)(n + 1);
    tr_math_budget dflt;
    return tr_budget_check(budget_or(budget, &dflt), &need, refusal);
}

tr_lin_result *tr_lin_q_solve(int32_t m, int32_t n, const tr_bigq *const *A, const tr_bigq *const *b,
                              const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    if (!dims_ok(m, n) || !A || !b) { set_error(error, m > TR_LIN_MAX_ROWS || n > TR_LIN_MAX_UNKNOWNS ? TR_MATH_UNSUPPORTED : TR_MATH_INVALID); return NULL; }
    int32_t rc = solve_need(m, n, budget, refusal);
    if (rc) { set_error(error, rc); return NULL; }
    lctx x = {budget, refusal, 0};
    const int32_t c = n + 1;
    tr_bigint **M = Z_arr(&x, (uint64_t)m * c), **L = Z_arr(&x, (uint64_t)m);
    if (LIVE(&x)) scale_rows(&x, m, n, A, b, M, L);
    zsol z;
    memset(&z, 0, sizeof z);
    if (LIVE(&x)) int_solve(&x, M, m, n, &z);
    tr_lin_result *res = res_new(&x, TR_LIN_RESULT_SOLVE);
    if (LIVE(&x) && z.kind == TR_LIN_NONE) res_q(res, TR_LIN_PART_CERT, certificate(&x, M, L, m, n), (uint64_t)m);
    sol_result(&x, res, &z, m, n, TR_LIN_METHOD_BAREISS);
    zsol_free(&z, n);
    Z_free_all(M, M ? (uint64_t)m * c : 0); Z_free_all(L, L ? (uint64_t)m : 0);
    return res_done(&x, res, error);
}

tr_lin_result *tr_lin_solve(int32_t m, int32_t n, const tr_lin_qbuf *A, const tr_lin_qbuf *b, const tr_math_budget *budget,
                            tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    if (!dims_ok(m, n) || !A || !b) { set_error(error, m > TR_LIN_MAX_ROWS || n > TR_LIN_MAX_UNKNOWNS ? TR_MATH_UNSUPPORTED : TR_MATH_INVALID); return NULL; }
    int32_t rc = solve_need(m, n, budget, refusal);
    if (rc) { set_error(error, rc); return NULL; }
    uint64_t na = (uint64_t)m * (uint64_t)n;
    tr_bigq **qa = (tr_bigq **)zalloc(na * sizeof *qa), **qb = (tr_bigq **)zalloc((uint64_t)m * sizeof *qb);
    if (!qa || !qb) { free(qa); free(qb); set_error(error, TR_MATH_MEMORY); return NULL; }
    rc = tr_lin_q_import(A, na, qa, budget, refusal);
    if (!rc) rc = tr_lin_q_import(b, (uint64_t)m, qb, budget, refusal);
    tr_lin_result *res = NULL;
    if (!rc) res = tr_lin_q_solve(m, n, (const tr_bigq *const *)qa, (const tr_bigq *const *)qb, budget, refusal, &rc);
    tr_lin_q_free(qa, na); tr_lin_q_free(qb, (uint64_t)m);
    set_error(error, rc);
    return rc ? NULL : res;
}

/* ================================================================ inertia */
/* v^T M v for an integer symmetric matrix */
static tr_bigq *quad_form(lctx *x, tr_bigint *const *M, const tr_bigq *const *v, int32_t n) {
    tr_bigq *acc = Q_int(x, 0);
    for (int32_t i = 0; i < n && LIVE(x); i++) {
        if (!tr_bigq_sign(v[i])) continue;
        for (int32_t j = 0; j < n && LIVE(x); j++) {
            if (!tr_bigq_sign(v[j]) || !Z_sign(M[(uint64_t)i * n + j])) continue;
            tr_bigq *mij = Q_z(x, M[(uint64_t)i * n + j]), *t = Q_mul(x, v[i], mij);
            Q_fma(x, &acc, t, v[j]);
            tr_bigq_free(mij); tr_bigq_free(t);
        }
    }
    return acc;
}
/* the certificate of pivot t: v (B coordinates) = (-B[P,P]^-1 B[P,p], 1 at p), mapped by T (v_A = T v) */
static tr_bigq **pivot_vector(lctx *x, tr_bigint *const *B, tr_bigint *const *T, int32_t n, const int32_t *P, int32_t t) {
    tr_bigq **v = Q_arr(x, (uint64_t)n), **w = Q_arr(x, (uint64_t)n);
    for (int32_t i = 0; v && w && i < n && LIVE(x); i++) w[i] = Q_int(x, 0);
    int32_t p = P[t];
    if (LIVE(x)) { tr_bigq_free(w[p]); w[p] = Q_int(x, 1); }
    if (t > 0 && LIVE(x)) {
        tr_bigint **S = Z_arr(x, (uint64_t)t * (uint64_t)(t + 1));
        for (int32_t i = 0; S && i < t && LIVE(x); i++) {
            for (int32_t j = 0; j < t && LIVE(x); j++) S[(uint64_t)i * (t + 1) + j] = Z_copy(x, B[(uint64_t)P[i] * n + P[j]]);
            S[(uint64_t)i * (t + 1) + t] = Z_neg(x, B[(uint64_t)P[i] * n + p]);
        }
        zsol z;
        memset(&z, 0, sizeof z);
        if (LIVE(x)) int_solve(x, S, t, t, &z);
        if (LIVE(x) && z.kind != TR_LIN_UNIQUE) lfail(x, TR_MATH_INVALID);     /* the leading block is nonsingular */
        for (int32_t i = 0; i < t && LIVE(x); i++) { tr_bigq_free(w[P[i]]); w[P[i]] = Q_copy(x, z.x[i]); }
        zsol_free(&z, t);
        Z_free_all(S, S ? (uint64_t)t * (uint64_t)(t + 1) : 0);
    }
    for (int32_t i = 0; v && i < n && LIVE(x); i++) {           /* v = T w */
        v[i] = Q_int(x, 0);
        for (int32_t j = 0; j < n && LIVE(x); j++) {
            if (!Z_sign(T[(uint64_t)i * n + j])) continue;
            tr_bigq *tij = Q_z(x, T[(uint64_t)i * n + j]);
            Q_fma(x, &v[i], tij, w[j]);
            tr_bigq_free(tij);
        }
    }
    tr_lin_q_free(w, w ? (uint64_t)n : 0);
    if (x->rc) { tr_lin_q_free(v, v ? (uint64_t)n : 0); return NULL; }
    return v;
}

tr_lin_result *tr_lin_q_inertia(int32_t n, const tr_bigq *const *A, const tr_math_budget *budget, tr_math_refusal *refusal,
                                int32_t *error) {
    tr_budget_none(refusal);
    if (n < 1 || n > TR_LIN_MAX_UNKNOWNS || !A) { set_error(error, n > TR_LIN_MAX_UNKNOWNS ? TR_MATH_UNSUPPORTED : TR_MATH_INVALID); return NULL; }
    int32_t rc = solve_need(n, n, budget, refusal);
    if (rc) { set_error(error, rc); return NULL; }
    for (int32_t i = 0; i < n; i++)
        for (int32_t j = i + 1; j < n; j++)
            if (tr_bigq_cmp(A[(uint64_t)i * n + j], A[(uint64_t)j * n + i]) != 0) { set_error(error, TR_MATH_INVALID); return NULL; }
    lctx x = {budget, refusal, 0};
    const uint64_t nn = (uint64_t)n * (uint64_t)n;
    /* A' = L A with L > 0 the lcm of every denominator (inertia unchanged) */
    tr_bigint *L = Z_i64(&x, 1);
    for (uint64_t i = 0; i < nn && LIVE(&x); i++) {
        const tr_bigint *d = tr_bigq_denominator(A[i]);
        if (Z_is_one(d)) continue;
        tr_bigint *g = Z_gcd(&x, L, d), *t = Z_div(&x, L, g), *nl = Z_mul(&x, t, d);
        tr_bigint_free(g); tr_bigint_free(t);
        if (nl) { tr_bigint_free(L); L = nl; }
    }
    tr_bigint **B = Z_arr(&x, nn), **S = Z_arr(&x, nn), **T = Z_arr(&x, nn);
    for (uint64_t i = 0; B && S && T && i < nn && LIVE(&x); i++) {
        const tr_bigint *d = tr_bigq_denominator(A[i]);
        tr_bigint *t = Z_div(&x, L, d);
        B[i] = Z_mul(&x, tr_bigq_numerator(A[i]), t);
        tr_bigint_free(t);
        S[i] = Z_copy(&x, B[i]);
        T[i] = Z_i64(&x, i / (uint64_t)n == i % (uint64_t)n);
    }
    int32_t R[64], nr = n, P[64], np = 0, dsign[64], pos = 0, neg = 0;
    for (int32_t i = 0; i < n; i++) R[i] = i;
    tr_bigint *prev = Z_i64(&x, 1);
    while (nr && LIVE(&x)) {
        int32_t at = -1;
        for (int32_t a = 0; a < nr && at < 0; a++) if (Z_sign(S[(uint64_t)R[a] * n + R[a]])) at = a;
        if (at < 0) {
            int32_t ia = -1, ja = -1;
            for (int32_t a = 0; a < nr && ia < 0; a++)
                for (int32_t b = a + 1; b < nr && ia < 0; b++) if (Z_sign(S[(uint64_t)R[a] * n + R[b]])) { ia = a; ja = b; }
            if (ia < 0) break;                                            /* the remaining block is zero */
            int32_t i = R[ia], j = R[ja];
            /* the congruence e_i <- e_i + e_j on S (remaining block), B (all) and T (columns) */
            for (int32_t b = 0; b < nr && LIVE(&x); b++) {
                tr_bigint *t = Z_add(&x, S[(uint64_t)i * n + R[b]], S[(uint64_t)j * n + R[b]]);
                if (t) { tr_bigint_free(S[(uint64_t)i * n + R[b]]); S[(uint64_t)i * n + R[b]] = t; }
            }
            for (int32_t a = 0; a < nr && LIVE(&x); a++) {
                tr_bigint *t = Z_add(&x, S[(uint64_t)R[a] * n + i], S[(uint64_t)R[a] * n + j]);
                if (t) { tr_bigint_free(S[(uint64_t)R[a] * n + i]); S[(uint64_t)R[a] * n + i] = t; }
            }
            for (int32_t b = 0; b < n && LIVE(&x); b++) {
                tr_bigint *t = Z_add(&x, B[(uint64_t)i * n + b], B[(uint64_t)j * n + b]);
                if (t) { tr_bigint_free(B[(uint64_t)i * n + b]); B[(uint64_t)i * n + b] = t; }
            }
            for (int32_t a = 0; a < n && LIVE(&x); a++) {
                tr_bigint *t = Z_add(&x, B[(uint64_t)a * n + i], B[(uint64_t)a * n + j]);
                if (t) { tr_bigint_free(B[(uint64_t)a * n + i]); B[(uint64_t)a * n + i] = t; }
                tr_bigint *u = Z_add(&x, T[(uint64_t)a * n + i], T[(uint64_t)a * n + j]);
                if (u) { tr_bigint_free(T[(uint64_t)a * n + i]); T[(uint64_t)a * n + i] = u; }
            }
            continue;
        }
        int32_t i = R[at];
        const tr_bigint *piv = S[(uint64_t)i * n + i];
        int sg = Z_sign(piv) * Z_sign(prev);
        dsign[np] = sg; P[np++] = i;
        if (sg > 0) pos++; else neg++;
        for (int32_t a = at; a + 1 < nr; a++) R[a] = R[a + 1];          /* drop i, keep the order of the rest */
        nr--;
        int one = Z_is_one(prev);
        for (int32_t a = 0; a < nr && LIVE(&x); a++)
            for (int32_t b = a; b < nr && LIVE(&x); b++) {
                uint64_t ab = (uint64_t)R[a] * n + R[b], ba = (uint64_t)R[b] * n + R[a];
                tr_bigint *t = Z_cross(&x, piv, S[ab], S[(uint64_t)R[a] * n + i], S[(uint64_t)i * n + R[b]]);
                tr_bigint *nv = one ? t : Z_div(&x, t, prev);
                if (!one) tr_bigint_free(t);
                if (!nv) break;
                tr_bigint_free(S[ab]); S[ab] = nv;
                if (ab != ba) { tr_bigint *cp = Z_copy(&x, nv); if (cp) { tr_bigint_free(S[ba]); S[ba] = cp; } }
            }
        tr_bigint *cp = Z_copy(&x, piv);
        if (cp) { tr_bigint_free(prev); prev = cp; }
    }
    int32_t zero = n - np;
    tr_lin_result *res = res_new(&x, TR_LIN_RESULT_INERTIA);
    if (LIVE(&x)) {
        res->info.rows = n; res->info.cols = n; res->info.positive = pos; res->info.negative = neg; res->info.zero = zero;
        res->info.rank = np; res->info.nullity = zero; res->info.method = TR_LIN_METHOD_BAREISS;
        res->info.status = !pos && !neg ? TR_LIN_ZERO : pos && neg ? TR_LIN_INDEFINITE
                         : pos ? (zero ? TR_LIN_PSD : TR_LIN_PD) : (zero ? TR_LIN_NSD : TR_LIN_ND);
    }
    /* certificates: the first negative and the first positive pivot, verified exactly */
    for (int pass = 0; pass < 2 && LIVE(&x); pass++) {
        int32_t want = pass ? 1 : -1, t = -1;
        for (int32_t q = 0; q < np && t < 0; q++) if (dsign[q] == want) t = q;
        if (t < 0) continue;
        tr_bigq **v = pivot_vector(&x, B, T, n, P, t);
        tr_bigint **Ain = Z_arr(&x, nn);                     /* the scaled input A' = L A for the check */
        for (uint64_t i = 0; Ain && i < nn && LIVE(&x); i++) {
            tr_bigint *t2 = Z_div(&x, L, tr_bigq_denominator(A[i]));
            Ain[i] = Z_mul(&x, tr_bigq_numerator(A[i]), t2);
            tr_bigint_free(t2);
        }
        tr_bigq *val = NULL;
        if (LIVE(&x)) val = quad_form(&x, Ain, (const tr_bigq *const *)v, n);
        if (LIVE(&x) && tr_bigq_sign(val) != want) lfail(&x, TR_MATH_INVALID);
        tr_bigq_free(val);
        Z_free_all(Ain, Ain ? nn : 0);
        if (LIVE(&x)) res_q(res, pass ? TR_LIN_PART_POS : TR_LIN_PART_NEG, v, (uint64_t)n);
        else tr_lin_q_free(v, v ? (uint64_t)n : 0);
    }
    /* the kernel: the canonical basis of A x = 0 (its size must be the zero count) */
    if (LIVE(&x)) {
        tr_bigq **zb = Q_arr(&x, (uint64_t)n);
        for (int32_t i = 0; zb && i < n && LIVE(&x); i++) zb[i] = Q_int(&x, 0);
        int32_t e2 = 0;
        tr_lin_result *ker = LIVE(&x) ? tr_lin_q_solve(n, n, A, (const tr_bigq *const *)zb, x.b, x.r, &e2) : NULL;
        if (!ker && LIVE(&x)) lfail(&x, e2);
        if (ker && ker->info.nullity != zero) lfail(&x, TR_MATH_INVALID);
        if (ker && LIVE(&x) && ker->q[TR_LIN_PART_NULL]) {
            res_q(res, TR_LIN_PART_KERNEL, ker->q[TR_LIN_PART_NULL], ker->nq[TR_LIN_PART_NULL]);
            ker->q[TR_LIN_PART_NULL] = NULL; ker->nq[TR_LIN_PART_NULL] = 0;
        }
        tr_lin_result_free(ker);
        tr_lin_q_free(zb, zb ? (uint64_t)n : 0);
        if (LIVE(&x)) res->info.certified = 1;
    }
    tr_bigint_free(prev); tr_bigint_free(L);
    Z_free_all(B, B ? nn : 0); Z_free_all(S, S ? nn : 0); Z_free_all(T, T ? nn : 0);
    return res_done(&x, res, error);
}

tr_lin_result *tr_lin_inertia(int32_t n, const tr_lin_qbuf *A, const tr_math_budget *budget, tr_math_refusal *refusal,
                              int32_t *error) {
    tr_budget_none(refusal);
    if (n < 1 || n > TR_LIN_MAX_UNKNOWNS || !A) { set_error(error, n > TR_LIN_MAX_UNKNOWNS ? TR_MATH_UNSUPPORTED : TR_MATH_INVALID); return NULL; }
    int32_t rc = solve_need(n, n, budget, refusal);
    if (rc) { set_error(error, rc); return NULL; }
    uint64_t nn = (uint64_t)n * (uint64_t)n;
    tr_bigq **qa = (tr_bigq **)zalloc(nn * sizeof *qa);
    if (!qa) { set_error(error, TR_MATH_MEMORY); return NULL; }
    rc = tr_lin_q_import(A, nn, qa, budget, refusal);
    tr_lin_result *res = rc ? NULL : tr_lin_q_inertia(n, (const tr_bigq *const *)qa, budget, refusal, &rc);
    tr_lin_q_free(qa, nn);
    set_error(error, rc);
    return rc ? NULL : res;
}

/* ================================================================ moment operators */
typedef struct {
    int32_t dim, intercept, listwise, ridged;
    int32_t design[TR_LIN_MAX_COLS];
    uint64_t n, group;
    tr_bigq **A, **A0, **b;          /* dim x dim with the ridge, without it, dim */
    tr_bigq *yty, *sy;
} fitctx;
static void fit_free(fitctx *f) {
    uint64_t dd = (uint64_t)f->dim * (uint64_t)f->dim;
    tr_lin_q_free(f->A, f->A ? dd : 0); tr_lin_q_free(f->A0, f->A0 ? dd : 0); tr_lin_q_free(f->b, f->b ? (uint64_t)f->dim : 0);
    tr_bigq_free(f->yty); tr_bigq_free(f->sy);
    memset(f, 0, sizeof *f);
}
/* the normal equations of a fit (value units); no target: TR_MATH_INVALID */
static int32_t fit_build(lctx *x, const tr_lin_moments *s, const tr_lin_fit_spec *fs, fitctx *f) {
    memset(f, 0, sizeof *f);
    if (!mom_valid(s) || !fs || fs->group >= s->ngroups || fs->intercept < -1 || fs->intercept > 1 ||
        (fs->masked != 0 && fs->masked != 1) || fs->reserved || fs->ridge_den <= 0 || fs->ridge_num < 0 || s->spec.target < 0)
        return TR_MATH_INVALID;
    const int32_t d = s->spec.nfeatures, y = s->k - 1;
    if (fs->masked && (fs->features >> d)) return TR_MATH_INVALID;
    f->intercept = fs->intercept < 0 ? s->spec.intercept : fs->intercept;
    f->listwise = s->spec.policy == TR_LIN_LISTWISE;
    f->ridged = fs->ridge_num != 0;
    f->group = fs->group;
    if (f->intercept) f->design[f->dim++] = 0;
    for (int32_t i = 0; i < d; i++) if (!fs->masked || ((fs->features >> i) & 1)) f->design[f->dim++] = i + 1;
    if (!f->dim) return TR_MATH_INVALID;                                  /* an empty design */
    const uint64_t g = fs->group, dd = (uint64_t)f->dim * (uint64_t)f->dim;
    f->A = Q_arr(x, dd); f->A0 = Q_arr(x, dd); f->b = Q_arr(x, (uint64_t)f->dim);
    tr_bigq *lam = f->ridged ? Q_q64(x, fs->ridge_num, fs->ridge_den) : NULL;
    for (int32_t u = 0; u < f->dim && LIVE(x); u++) {
        for (int32_t w = 0; w < f->dim && LIVE(x); w++) {
            uint64_t i = (uint64_t)u * f->dim + w;
            f->A0[i] = gram_q(x, s, g, f->design[u], f->design[w]);
            f->A[i] = (u == w && lam && f->design[u]) ? Q_add(x, f->A0[i], lam) : Q_copy(x, f->A0[i]);
        }
        f->b[u] = gram_q(x, s, g, f->design[u], y);
    }
    tr_bigq_free(lam);
    f->yty = gram_q(x, s, g, y, y);
    f->sy = gram_q(x, s, g, 0, y);
    f->n = mom_count(s, g, 0);
    if (!f->listwise) f->n = mom_count(s, g, tr_lin_gram_slot(s->k, y, y));
    return x->rc;
}
/* rss = yty - 2 x.b + x^T A0 x, tss, r2 = 1 - rss / tss, sigma2 = rss / (n - rank), yty, n, and the squared norm of
 * the projection of y onto the design's column space, ||X x||^2 = x^T A0 x */
static tr_bigq **fit_stats(lctx *x, const fitctx *f, const zsol *z) {
    tr_bigq **st = Q_arr(x, TR_LIN_STATS);
    if (!st) return NULL;
    st[4] = Q_copy(x, f->yty);
    if (!f->listwise || !z->x) return st;
    st[5] = Q_q64(x, (int64_t)(f->n > (uint64_t)INT64_MAX ? INT64_MAX : f->n), 1);
    tr_bigq *xb = Q_int(x, 0), *xax = Q_int(x, 0);
    for (int32_t u = 0; u < f->dim && LIVE(x); u++) {
        Q_fma(x, &xb, z->x[u], f->b[u]);
        for (int32_t w = 0; w < f->dim && LIVE(x); w++) {
            tr_bigq *t = Q_mul(x, z->x[u], f->A0[(uint64_t)u * f->dim + w]);
            Q_fma(x, &xax, t, z->x[w]);
            tr_bigq_free(t);
        }
    }
    tr_bigq *two = Q_int(x, 2), *t2 = Q_mul(x, two, xb), *t3 = Q_sub(x, f->yty, t2);
    st[0] = Q_add(x, t3, xax);                                               /* rss */
    st[6] = Q_copy(x, xax);                                                  /* ||X x||^2 */
    tr_bigq_free(two); tr_bigq_free(t2); tr_bigq_free(t3); tr_bigq_free(xb); tr_bigq_free(xax);
    if (f->intercept) {
        if (f->n) {
            tr_bigq *nn = Q_q64(x, (int64_t)f->n, 1), *s2 = Q_mul(x, f->sy, f->sy), *q = Q_div(x, s2, nn);
            st[1] = Q_sub(x, f->yty, q);
            tr_bigq_free(nn); tr_bigq_free(s2); tr_bigq_free(q);
        }
    } else st[1] = Q_copy(x, f->yty);
    if (st[1] && LIVE(x) && tr_bigq_sign(st[1])) {
        tr_bigq *one = Q_int(x, 1), *q = Q_div(x, st[0], st[1]);
        st[2] = Q_sub(x, one, q);
        tr_bigq_free(one); tr_bigq_free(q);
    }
    if (!f->ridged && f->n > (uint64_t)z->rank && LIVE(x)) {
        tr_bigq *df = Q_q64(x, (int64_t)(f->n - (uint64_t)z->rank), 1);
        st[3] = Q_div(x, st[0], df);
        tr_bigq_free(df);
    }
    return st;
}
static tr_lin_result *fit_result(lctx *x, const fitctx *f, zsol *z, int32_t method, uint64_t primes) {
    tr_lin_result *res = res_new(x, TR_LIN_RESULT_FIT);
    if (!res) return NULL;
    tr_bigq **st = fit_stats(x, f, z);
    sol_result(x, res, z, f->dim, f->dim, method);
    if (x->rc) { tr_lin_q_free(st, st ? TR_LIN_STATS : 0); return res; }
    res->info.what = TR_LIN_RESULT_FIT; res->info.primes = primes; res->info.used = f->n;
    res_q(res, TR_LIN_PART_STATS, st, TR_LIN_STATS);
    uint64_t dd = (uint64_t)f->dim * (uint64_t)f->dim;
    tr_bigq **a = Q_arr(x, dd), **b = Q_arr(x, (uint64_t)f->dim), **ds = Q_arr(x, (uint64_t)f->dim);
    for (uint64_t i = 0; a && i < dd && LIVE(x); i++) a[i] = Q_copy(x, f->A[i]);
    for (int32_t i = 0; b && ds && i < f->dim && LIVE(x); i++) { b[i] = Q_copy(x, f->b[i]); ds[i] = Q_int(x, f->design[i]); }
    if (x->rc) { tr_lin_q_free(a, a ? dd : 0); tr_lin_q_free(b, b ? (uint64_t)f->dim : 0); tr_lin_q_free(ds, ds ? (uint64_t)f->dim : 0); return res; }
    res_q(res, TR_LIN_PART_A, a, dd); res_q(res, TR_LIN_PART_B, b, (uint64_t)f->dim); res_q(res, TR_LIN_PART_DESIGN, ds, (uint64_t)f->dim);
    return res;
}
static int32_t fit_need(const tr_lin_moments *s, const tr_math_budget *budget, tr_math_refusal *refusal) {
    return solve_need(s->k, s->k, budget, refusal);
}

tr_lin_result *tr_lin_system(const tr_lin_moments *s, const tr_lin_fit_spec *fs, const tr_math_budget *budget,
                             tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    if (!mom_valid(s)) { set_error(error, TR_MATH_INVALID); return NULL; }
    int32_t rc = fit_need(s, budget, refusal);
    if (rc) { set_error(error, rc); return NULL; }
    lctx x = {budget, refusal, 0};
    fitctx f;
    rc = fit_build(&x, s, fs, &f);
    if (rc) { fit_free(&f); set_error(error, rc); return NULL; }
    tr_lin_result *res = res_new(&x, TR_LIN_RESULT_SYSTEM);
    uint64_t dd = (uint64_t)f.dim * (uint64_t)f.dim;
    if (LIVE(&x)) {
        res->info.rows = f.dim; res->info.cols = f.dim; res->info.used = f.n;
        res_q(res, TR_LIN_PART_A, f.A, dd); f.A = NULL;
        res_q(res, TR_LIN_PART_B, f.b, (uint64_t)f.dim); f.b = NULL;
        tr_bigq **ds = Q_arr(&x, (uint64_t)f.dim), **st = Q_arr(&x, TR_LIN_STATS);
        for (int32_t i = 0; ds && i < f.dim && LIVE(&x); i++) ds[i] = Q_int(&x, f.design[i]);
        if (st && LIVE(&x)) { st[4] = Q_copy(&x, f.yty); st[5] = f.listwise ? Q_q64(&x, (int64_t)f.n, 1) : NULL; }
        res_q(res, TR_LIN_PART_DESIGN, ds, ds ? (uint64_t)f.dim : 0);
        res_q(res, TR_LIN_PART_STATS, st, st ? TR_LIN_STATS : 0);
    }
    fit_free(&f);
    return res_done(&x, res, error);
}

/* the system of a fit as integers (rows scaled) */
static void fit_integers(lctx *x, const fitctx *f, tr_bigint **M, tr_bigint **L) {
    scale_rows(x, f->dim, f->dim, (const tr_bigq *const *)f->A, (const tr_bigq *const *)f->b, M, L);
}

tr_lin_result *tr_lin_fit(const tr_lin_moments *s, const tr_lin_fit_spec *fs, const tr_math_budget *budget,
                          tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    if (!mom_valid(s)) { set_error(error, TR_MATH_INVALID); return NULL; }
    int32_t rc = fit_need(s, budget, refusal);
    if (rc) { set_error(error, rc); return NULL; }
    lctx x = {budget, refusal, 0};
    fitctx f;
    rc = fit_build(&x, s, fs, &f);
    if (rc) { fit_free(&f); set_error(error, rc); return NULL; }
    const int32_t m = f.dim, c = f.dim + 1;
    tr_bigint **M = Z_arr(&x, (uint64_t)m * c), **L = Z_arr(&x, (uint64_t)m);
    if (LIVE(&x)) fit_integers(&x, &f, M, L);
    zsol z;
    memset(&z, 0, sizeof z);
    if (LIVE(&x)) int_solve(&x, M, m, m, &z);
    tr_lin_result *res = LIVE(&x) ? fit_result(&x, &f, &z, TR_LIN_METHOD_BAREISS, 0) : NULL;
    if (LIVE(&x) && z.kind == TR_LIN_NONE) res_q(res, TR_LIN_PART_CERT, certificate(&x, M, L, m, m), (uint64_t)m);
    zsol_free(&z, m);
    Z_free_all(M, M ? (uint64_t)m * c : 0); Z_free_all(L, L ? (uint64_t)m : 0);
    fit_free(&f);
    return res_done(&x, res, error);
}

int32_t tr_lin_q_normal(const tr_lin_moments *s, const tr_lin_fit_spec *fs, int32_t *dim, tr_bigq ***A, tr_bigq ***A0,
                        tr_bigq ***b, tr_bigq **yty, tr_bigq **sy, uint64_t *n, int32_t *design,
                        const tr_math_budget *budget, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!mom_valid(s) || !dim || !A || !b || !yty) return TR_MATH_INVALID;
    int32_t rc = fit_need(s, budget, refusal);
    if (rc) return rc;
    lctx x = {budget, refusal, 0};
    fitctx f;
    rc = fit_build(&x, s, fs, &f);
    if (rc) { fit_free(&f); return rc; }
    *dim = f.dim; *A = f.A; *b = f.b; *yty = f.yty; f.A = NULL; f.b = NULL; f.yty = NULL;
    if (A0) { *A0 = f.A0; f.A0 = NULL; }
    if (sy) { *sy = f.sy; f.sy = NULL; }
    if (n) *n = f.n;
    if (design) memcpy(design, f.design, (size_t)f.dim * sizeof(int32_t));
    fit_free(&f);
    return TR_MATH_OK;
}

/* covariance, squared correlation, its sign and counts of the p = k - 1 non-constant columns */
static int32_t cov_parts(lctx *x, const tr_lin_moments *s, uint64_t g, int32_t ddof, tr_bigq ***cov, tr_bigq ***r2,
                         tr_bigq ***sign, tr_bigq ***count) {
    const int32_t p = s->k - 1, k = s->k;
    const uint64_t pp = (uint64_t)p * (uint64_t)p;
    *cov = Q_arr(x, pp); *r2 = Q_arr(x, pp); *sign = Q_arr(x, pp); *count = Q_arr(x, pp);
    for (int32_t ia = 0; ia < p && LIVE(x); ia++)
        for (int32_t ib = ia; ib < p && LIVE(x); ib++) {
            int32_t a = ia + 1, b = ib + 1;
            uint64_t n;
            tr_bigint *Sa, *Sb, *Saa, *Sbb, *Sab;
            if (s->spec.policy == TR_LIN_LISTWISE || a == b) {
                n = s->spec.policy == TR_LIN_LISTWISE ? mom_count(s, g, 0) : mom_count(s, g, tr_lin_gram_slot(k, a, a));
                Sa = gram_z(x, s, g, 0, a); Sb = gram_z(x, s, g, 0, b);
                Saa = gram_z(x, s, g, a, a); Sbb = gram_z(x, s, g, b, b); Sab = gram_z(x, s, g, a, b);
            } else {
                int32_t q = tr_lin_pair_slot(k, a, b);
                n = mom_count(s, g, tr_lin_gram_slot(k, a, b));
                Sa = mom_sum(x, s, g, q); Sb = mom_sum(x, s, g, q + 1);
                Saa = mom_sum(x, s, g, q + 2); Sbb = mom_sum(x, s, g, q + 3); Sab = gram_z(x, s, g, a, b);
            }
            tr_bigint *N = Z_i64(x, (int64_t)n);
            tr_bigint *num = Z_cross(x, N, Sab, Sa, Sb);                       /* n Sab - Sa Sb */
            tr_bigint *va = Z_cross(x, N, Saa, Sa, Sa), *vb = Z_cross(x, N, Sbb, Sb, Sb);
            uint64_t ij = (uint64_t)ia * p + ib, ji = (uint64_t)ib * p + ia;
            tr_bigq *cv = NULL, *rr = NULL, *sg = NULL, *ct = NULL;
            if (LIVE(x)) {
                ct = Q_q64(x, (int64_t)n, 1);
                sg = Q_int(x, Z_sign(num));
                if (n > (uint64_t)ddof) {
                    tr_bigint *den = Z_i64(x, (int64_t)n), *n2 = Z_i64(x, (int64_t)(n - (uint64_t)ddof)), *dd = Z_mul(x, den, n2);
                    tr_bigint *p10 = Z_pow10(x, s->scale[a] + s->scale[b]), *full = Z_mul(x, dd, p10);
                    cv = Q_frac(x, num, full);
                    tr_bigint_free(den); tr_bigint_free(n2); tr_bigint_free(dd); tr_bigint_free(p10); tr_bigint_free(full);
                }
                if (Z_sign(va) && Z_sign(vb)) {
                    tr_bigint *n2 = Z_mul(x, num, num), *d2 = Z_mul(x, va, vb);
                    rr = Q_frac(x, n2, d2);
                    tr_bigint_free(n2); tr_bigint_free(d2);
                }
            }
            tr_bigint_free(N); tr_bigint_free(num); tr_bigint_free(va); tr_bigint_free(vb);
            tr_bigint_free(Sa); tr_bigint_free(Sb); tr_bigint_free(Saa); tr_bigint_free(Sbb); tr_bigint_free(Sab);
            if (!LIVE(x)) { tr_bigq_free(cv); tr_bigq_free(rr); tr_bigq_free(sg); tr_bigq_free(ct); break; }
            (*cov)[ij] = cv; (*r2)[ij] = rr; (*sign)[ij] = sg; (*count)[ij] = ct;
            if (ij != ji) {
                (*cov)[ji] = cv ? Q_copy(x, cv) : NULL; (*r2)[ji] = rr ? Q_copy(x, rr) : NULL;
                (*sign)[ji] = Q_copy(x, sg); (*count)[ji] = Q_copy(x, ct);
            }
        }
    return x->rc;
}

tr_lin_result *tr_lin_covariance(const tr_lin_moments *s, uint64_t group, int32_t ddof, const tr_math_budget *budget,
                                 tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    if (!mom_valid(s) || group >= s->ngroups || (ddof != 0 && ddof != 1) || s->k < 2) { set_error(error, TR_MATH_INVALID); return NULL; }
    if (s->spec.policy == TR_LIN_PAIRWISE && !s->marginals && s->k > 2) { set_error(error, TR_MATH_DOMAIN_ERROR); return NULL; }
    int32_t rc = solve_need(s->k, s->k, budget, refusal);
    if (rc) { set_error(error, rc); return NULL; }
    lctx x = {budget, refusal, 0};
    tr_bigq **cov = NULL, **r2 = NULL, **sg = NULL, **ct = NULL;
    cov_parts(&x, s, group, ddof, &cov, &r2, &sg, &ct);
    const uint64_t pp = (uint64_t)(s->k - 1) * (uint64_t)(s->k - 1);
    tr_lin_result *res = res_new(&x, TR_LIN_RESULT_COVARIANCE);
    if (!LIVE(&x)) {
        tr_lin_q_free(cov, cov ? pp : 0); tr_lin_q_free(r2, r2 ? pp : 0); tr_lin_q_free(sg, sg ? pp : 0); tr_lin_q_free(ct, ct ? pp : 0);
        return res_done(&x, res, error);
    }
    res->info.rows = s->k - 1; res->info.cols = s->k - 1; res->info.status = s->spec.policy;
    res->info.used = s->spec.policy == TR_LIN_LISTWISE ? mom_count(s, group, 0) : s->grows[group];
    res->info.certified = 1;
    res_q(res, TR_LIN_PART_COV, cov, pp); res_q(res, TR_LIN_PART_R2, r2, pp);
    res_q(res, TR_LIN_PART_SIGN, sg, pp); res_q(res, TR_LIN_PART_COUNT, ct, pp);
    return res_done(&x, res, error);
}

/* ================================================================ numerical eigensystems (APPROXIMATE) */
/* cyclic Jacobi (the threshold strategy of Numerical Recipes' jacobi): a (n x n, destroyed), d eigenvalues, v columns
 * eigenvectors; 0 or TR_MATH_NOT_CONVERGED after cap sweeps */
static int32_t jacobi(int32_t n, double *a, double *d, double *v, uint64_t cap, uint64_t *sweeps) {
    double *bb = (double *)malloc((size_t)n * sizeof(double)), *z = (double *)malloc((size_t)n * sizeof(double));
    if (!bb || !z) { free(bb); free(z); return TR_MATH_MEMORY; }
    for (int32_t i = 0; i < n; i++) {
        for (int32_t j = 0; j < n; j++) v[i * n + j] = i == j ? 1.0 : 0.0;
        bb[i] = d[i] = a[i * n + i]; z[i] = 0.0;
    }
#define A_(i, j) a[(i) * n + (j)]
    for (uint64_t sw = 0; sw <= cap; sw++) {
        double sm = 0.0;
        for (int32_t p = 0; p < n - 1; p++) for (int32_t q = p + 1; q < n; q++) sm += fabs(A_(p, q));
        if (sm == 0.0) { *sweeps = sw; free(bb); free(z); return TR_MATH_OK; }
        if (sw == cap) break;
        double tresh = sw < 3 ? 0.2 * sm / ((double)n * n) : 0.0;
        for (int32_t p = 0; p < n - 1; p++)
            for (int32_t q = p + 1; q < n; q++) {
                double g = 100.0 * fabs(A_(p, q));
                if (sw > 3 && fabs(d[p]) + g == fabs(d[p]) && fabs(d[q]) + g == fabs(d[q])) A_(p, q) = 0.0;
                else if (fabs(A_(p, q)) > tresh) {
                    double h = d[q] - d[p], t;
                    if (fabs(h) + g == fabs(h)) t = A_(p, q) / h;
                    else {
                        double theta = 0.5 * h / A_(p, q);
                        t = 1.0 / (fabs(theta) + sqrt(1.0 + theta * theta));
                        if (theta < 0.0) t = -t;
                    }
                    double c = 1.0 / sqrt(1 + t * t), s = t * c, tau = s / (1.0 + c);
                    h = t * A_(p, q);
                    z[p] -= h; z[q] += h; d[p] -= h; d[q] += h;
                    A_(p, q) = 0.0;
#define ROT(M, i, j, k, l) do { double gg = M[(i) * n + (j)], hh = M[(k) * n + (l)]; \
                                M[(i) * n + (j)] = gg - s * (hh + gg * tau); M[(k) * n + (l)] = hh + s * (gg - hh * tau); } while (0)
                    for (int32_t j = 0; j < p; j++) ROT(a, j, p, j, q);
                    for (int32_t j = p + 1; j < q; j++) ROT(a, p, j, j, q);
                    for (int32_t j = q + 1; j < n; j++) ROT(a, p, j, q, j);
                    for (int32_t j = 0; j < n; j++) ROT(v, j, p, j, q);
#undef ROT
                }
            }
        for (int32_t p = 0; p < n; p++) { bb[p] += z[p]; d[p] = bb[p]; z[p] = 0.0; }
    }
#undef A_
    *sweeps = cap;
    free(bb); free(z);
    return TR_MATH_NOT_CONVERGED;
}

static double up_abs(tr_math_interval x) { double a = fabs(x.lo), b = fabs(x.hi); return a > b ? a : b; }

/* eigenvalues sorted descending with their vectors and the rigorous bounds against the enclosure [lo, hi] */
static tr_lin_result *eigen_core(lctx *x, int32_t n, const double *lo, const double *hi, uint64_t cap) {
    const uint64_t nn = (uint64_t)n * (uint64_t)n;
    double *a = (double *)malloc((size_t)nn * sizeof(double)), *v = (double *)malloc((size_t)nn * sizeof(double));
    double *d = (double *)malloc((size_t)n * sizeof(double));
    int32_t *ord = (int32_t *)malloc((size_t)n * sizeof(int32_t));
    if (!a || !v || !d || !ord) { free(a); free(v); free(d); free(ord); lfail(x, TR_MATH_MEMORY); return NULL; }
    double width = 0.0;
    for (uint64_t i = 0; i < nn; i++) {
        a[i] = lo[i] == hi[i] ? lo[i] : 0.5 * lo[i] + 0.5 * hi[i];
        double w = tr_add_ru(hi[i], -lo[i]);
        if (w > width) width = w;
    }
    uint64_t sweeps = 0;
    int32_t rc = jacobi(n, a, d, v, cap, &sweeps);
    tr_lin_result *res = NULL;
    if (rc) lfail(x, rc);
    else res = res_new(x, TR_LIN_RESULT_EIGEN);
    if (res) {
        for (int32_t i = 0; i < n; i++) ord[i] = i;
        for (int32_t i = 1; i < n; i++)                          /* stable insertion sort, descending */
            for (int32_t j = i; j > 0 && d[ord[j - 1]] < d[ord[j]]; j--) { int32_t t = ord[j]; ord[j] = ord[j - 1]; ord[j - 1] = t; }
        res->info.rows = n; res->info.cols = n; res->info.method = TR_LIN_METHOD_JACOBI; res->info.iterations = sweeps;
        res->info.converged = 1; res->info.status = 1;
        if (res_d(x, res, TR_LIN_PART_VALUES, (uint64_t)n) && res_d(x, res, TR_LIN_PART_VECTORS, nn) &&
            res_d(x, res, TR_LIN_PART_RESIDUAL, (uint64_t)n) && res_d(x, res, TR_LIN_PART_BOUND, (uint64_t)n) &&
            res_d(x, res, TR_LIN_PART_ORTHO, 1) && res_d(x, res, TR_LIN_PART_WIDTH, 1)) {
            double *val = res->d[TR_LIN_PART_VALUES], *vec = res->d[TR_LIN_PART_VECTORS];
            for (int32_t i = 0; i < n; i++) {
                int32_t c = ord[i], big = 0;
                val[i] = d[c];
                for (int32_t j = 0; j < n; j++) if (fabs(v[j * n + c]) > fabs(v[big * n + c])) big = j;
                double sgn = v[big * n + c] < 0 ? -1.0 : 1.0;           /* the largest component positive */
                for (int32_t j = 0; j < n; j++) vec[(uint64_t)i * n + j] = sgn * v[j * n + c];
            }
            for (int32_t i = 0; i < n; i++) {
                const double *u = vec + (uint64_t)i * n;
                tr_math_interval lam = tr_interval_point(val[i]);
                double r2 = 0.0, v2 = 0.0;
                for (int32_t j = 0; j < n; j++) {
                    tr_math_interval acc = tr_interval_mul(lam, tr_interval_point(u[j]));
                    acc = tr_interval_neg(acc);
                    for (int32_t k = 0; k < n; k++) {
                        tr_math_interval e; e.lo = lo[(uint64_t)j * n + k]; e.hi = hi[(uint64_t)j * n + k];
                        acc = tr_interval_add(acc, tr_interval_mul(e, tr_interval_point(u[k])));
                    }
                    double m = up_abs(acc);
                    r2 = tr_add_ru(r2, tr_mul_dir(m, m, 1));
                    v2 = tr_add_rd(v2, tr_mul_dir(u[j], u[j], 0));
                }
                double res2 = tr_sqrt_dir(r2, 1), vn = tr_sqrt_dir(v2, 0);
                res->d[TR_LIN_PART_RESIDUAL][i] = res2;
                res->d[TR_LIN_PART_BOUND][i] = vn > 0 ? tr_div_dir(res2, vn, 1) : INFINITY;
            }
            double ortho = 0.0;
            for (int32_t i = 0; i < n; i++)
                for (int32_t j = i; j < n; j++) {
                    tr_math_interval acc = tr_interval_point(i == j ? -1.0 : 0.0);
                    for (int32_t k = 0; k < n; k++)
                        acc = tr_interval_add(acc, tr_interval_mul(tr_interval_point(vec[(uint64_t)i * n + k]),
                                                                   tr_interval_point(vec[(uint64_t)j * n + k])));
                    double m = up_abs(acc);
                    if (m > ortho) ortho = m;
                }
            res->d[TR_LIN_PART_ORTHO][0] = ortho;
            res->d[TR_LIN_PART_WIDTH][0] = width;
        }
    }
    free(a); free(v); free(d); free(ord);
    return res;
}
static int32_t eigen_need(int32_t n, uint64_t *cap, const tr_math_budget *budget, tr_math_refusal *refusal) {
    tr_math_budget dflt;
    const tr_math_budget *b = budget_or(budget, &dflt);
    if (!*cap) *cap = TR_LIN_JACOBI_SWEEPS;
    tr_math_need need;
    memset(&need, 0, sizeof need);
    need.need[TR_BUDGET_TENSOR] = (uint64_t)n * (uint64_t)n;
    need.need[TR_BUDGET_ITERATIONS] = *cap;
    need.need[TR_BUDGET_OUTPUT] = (uint64_t)n * (uint64_t)(n + 3);
    need.need[TR_BUDGET_BYTES] = tr_budget_mul((uint64_t)n * (uint64_t)n, 8 * 5);
    return tr_budget_check(b, &need, refusal);
}

tr_lin_result *tr_lin_eigen(int32_t n, const double *lo, const double *hi, uint64_t sweeps, const tr_math_budget *budget,
                            tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    if (n < 1 || n > TR_LIN_MAX_ROWS || !lo || !hi) { set_error(error, n > TR_LIN_MAX_ROWS ? TR_MATH_UNSUPPORTED : TR_MATH_INVALID); return NULL; }
    for (int32_t i = 0; i < n; i++)
        for (int32_t j = 0; j < n; j++) {
            double l = lo[i * n + j], h = hi[i * n + j];
            if (!(l <= h) || !isfinite(l) || !isfinite(h) || l != lo[j * n + i] || h != hi[j * n + i]) {
                set_error(error, TR_MATH_INVALID); return NULL;
            }
        }
    int32_t rc = eigen_need(n, &sweeps, budget, refusal);
    if (rc) { set_error(error, rc); return NULL; }
    lctx x = {budget, refusal, 0};
    tr_lin_result *res = eigen_core(&x, n, lo, hi, sweeps);
    return res_done(&x, res, error);
}

/* the tight enclosure of an exact matrix entry by entry; TR_MATH_INVALID past the double range */
static int32_t enclose_all(const tr_bigq *const *A, uint64_t nn, double *lo, double *hi) {
    for (uint64_t i = 0; i < nn; i++) {
        tr_math_interval iv;
        int32_t rc = tr_bigq_enclose(A[i], &iv);
        if (rc) return rc;
        if (!isfinite(iv.lo) || !isfinite(iv.hi)) return TR_MATH_OVERFLOW;
        lo[i] = iv.lo; hi[i] = iv.hi;
    }
    return TR_MATH_OK;
}

tr_lin_result *tr_lin_eigen_exact(int32_t n, const tr_lin_qbuf *A, uint64_t sweeps, const tr_math_budget *budget,
                                  tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    if (n < 1 || n > TR_LIN_MAX_ROWS || !A) { set_error(error, n > TR_LIN_MAX_ROWS ? TR_MATH_UNSUPPORTED : TR_MATH_INVALID); return NULL; }
    int32_t rc = eigen_need(n, &sweeps, budget, refusal);
    if (rc) { set_error(error, rc); return NULL; }
    uint64_t nn = (uint64_t)n * (uint64_t)n;
    tr_bigq **qa = (tr_bigq **)zalloc(nn * sizeof *qa);
    double *lo = (double *)malloc((size_t)nn * sizeof(double)), *hi = (double *)malloc((size_t)nn * sizeof(double));
    if (!qa || !lo || !hi) { free(qa); free(lo); free(hi); set_error(error, TR_MATH_MEMORY); return NULL; }
    rc = tr_lin_q_import(A, nn, qa, budget, refusal);
    for (int32_t i = 0; !rc && i < n; i++)
        for (int32_t j = i + 1; j < n && !rc; j++) if (tr_bigq_cmp(qa[(uint64_t)i * n + j], qa[(uint64_t)j * n + i])) rc = TR_MATH_INVALID;
    if (!rc) rc = enclose_all((const tr_bigq *const *)qa, nn, lo, hi);
    tr_lin_q_free(qa, nn);
    tr_lin_result *res = NULL;
    if (!rc) {
        lctx x = {budget, refusal, 0};
        res = eigen_core(&x, n, lo, hi, sweeps);
        res = res_done(&x, res, &rc);
    }
    free(lo); free(hi);
    set_error(error, rc);
    return rc ? NULL : res;
}

tr_lin_result *tr_lin_pca(const tr_lin_moments *s, uint64_t group, int32_t kind, int32_t ddof, uint64_t sweeps,
                          const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    if (!mom_valid(s) || group >= s->ngroups || (kind != 0 && kind != 1) || (ddof != 0 && ddof != 1) || s->spec.nfeatures < 1) {
        set_error(error, TR_MATH_INVALID); return NULL;
    }
    if (s->spec.policy == TR_LIN_PAIRWISE && !s->marginals && s->k > 2) { set_error(error, TR_MATH_DOMAIN_ERROR); return NULL; }
    const int32_t d = s->spec.nfeatures, p = s->k - 1;
    int32_t rc = eigen_need(d, &sweeps, budget, refusal);
    if (rc) { set_error(error, rc); return NULL; }
    lctx x = {budget, refusal, 0};
    tr_bigq **cov = NULL, **r2 = NULL, **sg = NULL, **ct = NULL;
    cov_parts(&x, s, group, ddof, &cov, &r2, &sg, &ct);
    const uint64_t pp = (uint64_t)p * (uint64_t)p, dd = (uint64_t)d * (uint64_t)d;
    double *lo = (double *)malloc((size_t)dd * sizeof(double)), *hi = (double *)malloc((size_t)dd * sizeof(double));
    if (!lo || !hi) lfail(&x, TR_MATH_MEMORY);
    for (int32_t i = 0; i < d && LIVE(&x); i++)
        for (int32_t j = 0; j < d && LIVE(&x); j++) {
            const tr_bigq *c = cov[(uint64_t)i * p + j];
            if (!c) { lfail(&x, TR_MATH_DOMAIN_ERROR); break; }               /* undefined: n <= ddof */
            tr_math_interval iv;
            int32_t e = tr_bigq_enclose(c, &iv);
            if (e || !isfinite(iv.lo) || !isfinite(iv.hi)) { lfail(&x, e ? e : TR_MATH_OVERFLOW); break; }
            lo[(uint64_t)i * d + j] = iv.lo; hi[(uint64_t)i * d + j] = iv.hi;
        }
    if (LIVE(&x) && kind == 1) {                                             /* correlation: c_ij / sqrt(c_ii c_jj) */
        double *clo = (double *)malloc((size_t)dd * sizeof(double)), *chi = (double *)malloc((size_t)dd * sizeof(double));
        if (!clo || !chi) lfail(&x, TR_MATH_MEMORY);
        for (int32_t i = 0; i < d && LIVE(&x); i++) if (!tr_bigq_sign(cov[(uint64_t)i * p + i]) || tr_bigq_sign(cov[(uint64_t)i * p + i]) < 0) lfail(&x, TR_MATH_DOMAIN_ERROR);
        for (int32_t i = 0; i < d && LIVE(&x); i++)
            for (int32_t j = 0; j < d; j++) {
                if (i == j) { clo[(uint64_t)i * d + j] = chi[(uint64_t)i * d + j] = 1.0; continue; }
                tr_math_interval cij, cii, cjj;
                cij.lo = lo[(uint64_t)i * d + j]; cij.hi = hi[(uint64_t)i * d + j];
                cii.lo = lo[(uint64_t)i * d + i]; cii.hi = hi[(uint64_t)i * d + i];
                cjj.lo = lo[(uint64_t)j * d + j]; cjj.hi = hi[(uint64_t)j * d + j];
                tr_math_interval den = tr_interval_sqrt(tr_interval_mul(cii, cjj));
                tr_math_interval r = tr_interval_div(cij, den);
                if (r.lo < -1.0) r.lo = -1.0;                                  /* |r| <= 1 exactly (Cauchy-Schwarz) */
                if (r.hi > 1.0) r.hi = 1.0;
                clo[(uint64_t)i * d + j] = r.lo; chi[(uint64_t)i * d + j] = r.hi;
            }
        if (LIVE(&x)) { memcpy(lo, clo, (size_t)dd * sizeof(double)); memcpy(hi, chi, (size_t)dd * sizeof(double)); }
        free(clo); free(chi);
    }
    if (LIVE(&x))                                                            /* symmetric by construction */
        for (int32_t i = 0; i < d; i++)
            for (int32_t j = i + 1; j < d; j++) { lo[(uint64_t)j * d + i] = lo[(uint64_t)i * d + j]; hi[(uint64_t)j * d + i] = hi[(uint64_t)i * d + j]; }
    tr_lin_result *res = LIVE(&x) ? eigen_core(&x, d, lo, hi, sweeps) : NULL;
    if (res && LIVE(&x)) {
        res->info.used = s->spec.policy == TR_LIN_LISTWISE ? mom_count(s, group, 0) : s->grows[group];
        tr_bigq **c2 = Q_arr(&x, dd);
        for (int32_t i = 0; c2 && i < d && LIVE(&x); i++)
            for (int32_t j = 0; j < d && LIVE(&x); j++) c2[(uint64_t)i * d + j] = Q_copy(&x, cov[(uint64_t)i * p + j]);
        if (LIVE(&x)) res_q(res, TR_LIN_PART_COV, c2, dd); else tr_lin_q_free(c2, c2 ? dd : 0);
    }
    tr_lin_q_free(cov, cov ? pp : 0); tr_lin_q_free(r2, r2 ? pp : 0); tr_lin_q_free(sg, sg ? pp : 0); tr_lin_q_free(ct, ct ? pp : 0);
    free(lo); free(hi);
    return res_done(&x, res, error);
}

/* ================================================================ modular batches */
struct tr_lin_batch_s {
    uint32_t magic;
    int32_t fits;
    uint64_t nsys, words, entries, max_need;
    uint64_t *data, *offset, *need, *rowoff;
    int32_t *rows, *cols;
    tr_bigint **ints, **scale;
    fitctx *fc;
};
static int bat_valid(const tr_lin_batch *b) { return b && b->magic == BAT_MAGIC; }
void tr_lin_batch_free(tr_lin_batch *b) {
    if (!bat_valid(b)) return;
    Z_free_all(b->ints, b->ints ? b->entries : 0);
    uint64_t nrows = 0;
    for (uint64_t s = 0; b->rows && s < b->nsys; s++) nrows += (uint64_t)b->rows[s];
    Z_free_all(b->scale, b->scale ? nrows : 0);
    if (b->fc) for (uint64_t s = 0; s < b->nsys; s++) fit_free(&b->fc[s]);
    free(b->fc); free(b->data); free(b->offset); free(b->need); free(b->rowoff); free(b->rows); free(b->cols);
    b->magic = 0;
    free(b);
}
static uint32_t ceil_log2(uint64_t m) { uint32_t k = 0; while (((uint64_t)1 << k) < m && k < 63) k++; return k; }
/* encode the integer systems for the device and bound the primes each needs (Hadamard) */
static int32_t batch_encode(lctx *x, tr_lin_batch *B) {
    uint64_t limbs = 1;
    for (uint64_t i = 0; i < B->entries; i++) { uint64_t l = tr_bigint_limb_count(B->ints[i]); if (l > limbs) limbs = l; }
    B->words = limbs + 1;
    tr_math_need need;
    memset(&need, 0, sizeof need);
    need.need[TR_BUDGET_BYTES] = tr_budget_mul(tr_budget_mul(B->entries, B->words), 8);
    tr_math_budget dflt;
    int32_t rc = tr_budget_check(budget_or(x->b, &dflt), &need, x->r);
    if (rc) { lfail(x, rc); return rc; }
    B->data = (uint64_t *)zalloc(B->entries * B->words * 8);
    if (!B->data) { lfail(x, TR_MATH_MEMORY); return x->rc; }
    uint64_t *mag = (uint64_t *)malloc((size_t)limbs * 8);
    if (!mag) { lfail(x, TR_MATH_MEMORY); return x->rc; }
    for (uint64_t i = 0; i < B->entries; i++) {
        int32_t sg = 0;
        uint64_t cnt = 0, *w = B->data + i * B->words;
        tr_bigint_export(B->ints[i], &sg, mag, limbs, &cnt);
        for (uint64_t j = 0; j < cnt; j++) w[j] = mag[j];
        if (sg < 0) {                                                        /* two's complement: ~m + 1 */
            uint64_t c = 1;
            for (uint64_t j = 0; j < B->words; j++) { uint64_t t = ~w[j] + c; c = c && t == 0; w[j] = t; }
        }
    }
    free(mag);
    B->max_need = 1;
    for (uint64_t s = 0; s < B->nsys; s++) {
        int32_t m = B->rows[s], c = B->cols[s];
        uint64_t bits[64], half = (ceil_log2((uint64_t)m) + 1) / 2 + 1;
        for (int32_t j = 0; j < c; j++) {
            uint64_t mb = 0;
            for (int32_t i = 0; i < m; i++) {
                uint64_t b = tr_bigint_bit_length(B->ints[B->offset[s] + (uint64_t)i * c + j]);
                if (b > mb) mb = b;
            }
            bits[j] = mb ? mb + half : 0;
        }
        for (int32_t i = 1; i < c; i++) for (int32_t j = i; j > 0 && bits[j - 1] < bits[j]; j--) { uint64_t t = bits[j]; bits[j] = bits[j - 1]; bits[j - 1] = t; }
        uint64_t hb = 0;
        for (int32_t j = 0; j < (m < c ? m : c); j++) hb += bits[j];
        B->need[s] = (hb + 2 + TR_LIN_PRIME_BITS - 1) / TR_LIN_PRIME_BITS;
        if (!B->need[s]) B->need[s] = 1;
        if (B->need[s] > B->max_need) B->max_need = B->need[s];
    }
    if (B->max_need + TR_LIN_PRIME_MARGIN > MAX_PRIMES) { lfail(x, TR_MATH_UNSUPPORTED); return x->rc; }
    return TR_MATH_OK;
}
static tr_lin_batch *batch_alloc(lctx *x, uint64_t nsys) {
    if (x->rc) return NULL;
    tr_lin_batch *B = (tr_lin_batch *)zalloc(sizeof *B);
    if (B) {
        B->nsys = nsys;
        B->offset = (uint64_t *)zalloc(nsys * 8); B->need = (uint64_t *)zalloc(nsys * 8); B->rowoff = (uint64_t *)zalloc(nsys * 8);
        B->rows = (int32_t *)zalloc(nsys * 4); B->cols = (int32_t *)zalloc(nsys * 4);
        B->magic = BAT_MAGIC;
    }
    if (!B || !B->offset || !B->need || !B->rowoff || !B->rows || !B->cols) { tr_lin_batch_free(B); lfail(x, TR_MATH_MEMORY); return NULL; }
    return B;
}
static tr_lin_batch *batch_done(lctx *x, tr_lin_batch *B, int32_t *error) {
    if (x->rc) { tr_lin_batch_free(B); set_error(error, x->rc); return NULL; }
    set_error(error, TR_MATH_OK);
    return B;
}

tr_lin_batch *tr_lin_batch_systems(int32_t nsys, const int32_t *rows, const int32_t *unknowns, const tr_lin_qbuf *A,
                                   const tr_lin_qbuf *b, const tr_math_budget *budget, tr_math_refusal *refusal,
                                   int32_t *error) {
    tr_budget_none(refusal);
    if (nsys < 1 || !rows || !unknowns || !A || !b) { set_error(error, TR_MATH_INVALID); return NULL; }
    uint64_t na = 0, nb = 0, entries = 0;
    for (int32_t s = 0; s < nsys; s++) {
        if (!dims_ok(rows[s], unknowns[s])) { set_error(error, rows[s] > TR_LIN_MAX_ROWS || unknowns[s] > TR_LIN_MAX_UNKNOWNS ? TR_MATH_UNSUPPORTED : TR_MATH_INVALID); return NULL; }
        na += (uint64_t)rows[s] * (uint64_t)unknowns[s]; nb += (uint64_t)rows[s];
        entries += (uint64_t)rows[s] * (uint64_t)(unknowns[s] + 1);
    }
    tr_math_need need;
    memset(&need, 0, sizeof need);
    need.need[TR_BUDGET_TENSOR] = entries;
    need.need[TR_BUDGET_OUTPUT] = (uint64_t)nsys;
    tr_math_budget dflt;
    int32_t rc = tr_budget_check(budget_or(budget, &dflt), &need, refusal);
    if (rc) { set_error(error, rc); return NULL; }
    lctx x = {budget, refusal, 0};
    tr_bigq **qa = Q_arr(&x, na), **qb = Q_arr(&x, nb);
    if (LIVE(&x)) { rc = tr_lin_q_import(A, na, qa, budget, refusal); if (!rc) rc = tr_lin_q_import(b, nb, qb, budget, refusal); if (rc) lfail(&x, rc); }
    tr_lin_batch *B = batch_alloc(&x, (uint64_t)nsys);
    if (B) {
        B->entries = entries;
        B->ints = Z_arr(&x, entries); B->scale = Z_arr(&x, nb);
        uint64_t oa = 0, ob = 0, oe = 0;
        for (int32_t s = 0; s < nsys && LIVE(&x); s++) {
            B->rows[s] = rows[s]; B->cols[s] = unknowns[s] + 1; B->offset[s] = oe; B->rowoff[s] = ob;
            scale_rows(&x, rows[s], unknowns[s], (const tr_bigq *const *)qa + oa, (const tr_bigq *const *)qb + ob,
                       B->ints + oe, B->scale + ob);
            oa += (uint64_t)rows[s] * (uint64_t)unknowns[s]; ob += (uint64_t)rows[s];
            oe += (uint64_t)rows[s] * (uint64_t)(unknowns[s] + 1);
        }
        if (LIVE(&x)) batch_encode(&x, B);
    }
    tr_lin_q_free(qa, qa ? na : 0); tr_lin_q_free(qb, qb ? nb : 0);
    return batch_done(&x, B, error);
}

tr_lin_batch *tr_lin_batch_fits(const tr_lin_moments *s, int32_t nfits, const tr_lin_fit_spec *fs, const tr_math_budget *budget,
                                tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    if (!mom_valid(s) || nfits < 1 || !fs) { set_error(error, TR_MATH_INVALID); return NULL; }
    tr_math_need need;
    memset(&need, 0, sizeof need);
    need.need[TR_BUDGET_TENSOR] = tr_budget_mul((uint64_t)nfits, (uint64_t)s->k * (uint64_t)(s->k + 1));
    need.need[TR_BUDGET_OUTPUT] = (uint64_t)nfits;
    tr_math_budget dflt;
    int32_t rc = tr_budget_check(budget_or(budget, &dflt), &need, refusal);
    if (rc) { set_error(error, rc); return NULL; }
    lctx x = {budget, refusal, 0};
    tr_lin_batch *B = batch_alloc(&x, (uint64_t)nfits);
    if (B) {
        B->fits = 1;
        B->fc = (fitctx *)zalloc((uint64_t)nfits * sizeof(fitctx));
        if (!B->fc) lfail(&x, TR_MATH_MEMORY);
        uint64_t entries = 0, nrows = 0;
        for (int32_t i = 0; i < nfits && LIVE(&x); i++) {
            rc = fit_build(&x, s, &fs[i], &B->fc[i]);
            if (rc) { lfail(&x, rc); break; }
            B->rows[i] = B->fc[i].dim; B->cols[i] = B->fc[i].dim + 1; B->offset[i] = entries; B->rowoff[i] = nrows;
            entries += (uint64_t)B->rows[i] * (uint64_t)B->cols[i]; nrows += (uint64_t)B->rows[i];
        }
        B->entries = entries;
        B->ints = Z_arr(&x, entries); B->scale = Z_arr(&x, nrows);
        for (int32_t i = 0; i < nfits && LIVE(&x); i++) fit_integers(&x, &B->fc[i], B->ints + B->offset[i], B->scale + B->rowoff[i]);
        if (LIVE(&x)) batch_encode(&x, B);
    }
    return batch_done(&x, B, error);
}

int32_t tr_lin_batch_get_view(const tr_lin_batch *B, tr_lin_batch_view *v) {
    if (!bat_valid(B) || !v) return TR_MATH_INVALID;
    v->nsys = B->nsys; v->words = B->words; v->entries = B->entries;
    v->data = B->data; v->offset = B->offset; v->rows = B->rows; v->cols = B->cols;
    return TR_MATH_OK;
}
uint64_t tr_lin_batch_primes_needed(const tr_lin_batch *B) { return bat_valid(B) ? B->max_need + TR_LIN_PRIME_MARGIN : 0; }
uint64_t tr_lin_batch_bytes(const tr_lin_batch *B, uint64_t nprimes) {
    tr_lin_batch_view v;
    return tr_lin_batch_get_view(B, &v) ? TR_BUDGET_UNKNOWN : tr_lin_batch_device_bytes(&v, nprimes);
}

int32_t tr_lin_primes(uint64_t count, tr_gfp *out) {
    if (!out || !count) return TR_MATH_INVALID;
    if (count > MAX_PRIMES) return TR_MATH_UNSUPPORTED;
    uint64_t p = ((uint64_t)1 << 62) - 1, found = 0;
    while (found < count) {
        if (p < ((uint64_t)1 << 61)) return TR_MATH_UNSUPPORTED;
        if (tr_gfp_is_prime(p)) { int32_t rc = tr_gfp_field(p, &out[found]); if (rc) return rc; found++; }
        p -= 2;
    }
    return TR_MATH_OK;
}

static int32_t primes_ok(const tr_gfp *primes, uint64_t np, int prove) {
    if (!primes || !np || np > MAX_PRIMES) return 0;
    for (uint64_t i = 0; i < np; i++) {
        if (!tr_gfp_consistent(&primes[i]) || primes[i].p < ((uint64_t)1 << 61) || (prove && !tr_gfp_is_prime(primes[i].p))) return 0;
        if (i && primes[i].p >= primes[i - 1].p) return 0;          /* strictly descending: distinct */
    }
    return 1;
}
static uint64_t batch_roff(const tr_lin_batch *B, uint64_t np, uint64_t *roff) {
    uint64_t total = 0;
    for (uint64_t s = 0; s < B->nsys; s++) {
        roff[s] = total;
        total += np * (4 + (uint64_t)B->rows[s] * (uint64_t)B->cols[s]);
    }
    return total;
}

int32_t tr_lin_batch_residues(const tr_lin_batch *B, const tr_gfp *primes, uint64_t np, uint64_t *residues,
                              const tr_math_budget *budget, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!bat_valid(B) || !residues || !primes_ok(primes, np, 0)) return TR_MATH_INVALID;
    tr_math_need need;
    memset(&need, 0, sizeof need);
    need.need[TR_BUDGET_BYTES] = tr_budget_mul(B->nsys, 8);
    tr_math_budget dflt;
    int32_t rc = tr_budget_check(budget_or(budget, &dflt), &need, refusal);
    if (rc) return rc;
    uint64_t *roff = (uint64_t *)zalloc(B->nsys * 8);
    if (!roff) return TR_MATH_MEMORY;
    batch_roff(B, np, roff);
    for (uint64_t s = 0; s < B->nsys; s++) {
        const int32_t m = B->rows[s], c = B->cols[s];
        const uint64_t mc = (uint64_t)m * (uint64_t)c;
        for (uint64_t q = 0; q < np; q++) {
            uint64_t *blk = residues + roff[s] + q * (4 + mc);
            const uint64_t *src = B->data + B->offset[s] * B->words;
            for (uint64_t e = 0; e < mc; e++) blk[4 + e] = tr_lin_reduce_words(&primes[q], src + e * B->words, B->words);
            tr_lin_modular_rref(&primes[q], blk + 4, m, c, blk);
        }
    }
    free(roff);
    return TR_MATH_OK;
}

/* mixed-radix lexicographic order of pivot masks: the mask whose lowest differing bit is set comes first */
static int mask_less(uint64_t a, uint64_t b) { uint64_t d = a ^ b; return d && (a & d & (~d + 1)); }

/* Garner's CRT with constants shared by the values of one system: x in (-M/2, M/2] for residues mod k primes */
typedef struct { uint64_t k; const tr_gfp *f[MAX_PRIMES]; uint64_t inv[MAX_PRIMES]; uint64_t *M; uint64_t mn; } garner;
static int32_t garner_init(garner *G, const tr_gfp *const *f, uint64_t k) {
    G->k = k;
    for (uint64_t i = 0; i < k; i++) G->f[i] = f[i];
    for (uint64_t i = 0; i < k; i++) {
        uint64_t prod = 1 % f[i]->p;
        for (uint64_t j = 0; j < i; j++) prod = tr_gfp_mul_hd(f[i], prod, f[j]->p % f[i]->p);
        G->inv[i] = i ? tr_gfp_inv_hd(f[i], prod) : 1;
    }
    G->M = (uint64_t *)zalloc((k + 1) * 8);
    if (!G->M) return TR_MATH_MEMORY;
    G->M[0] = 1; G->mn = 1;
    for (uint64_t i = 0; i < k; i++) {                         /* M *= p_i */
        unsigned __int128 c = 0;
        for (uint64_t j = 0; j < G->mn; j++) { c += (unsigned __int128)G->M[j] * f[i]->p; G->M[j] = (uint64_t)c; c >>= 64; }
        if (c) G->M[G->mn++] = (uint64_t)c;
    }
    return TR_MATH_OK;
}
static tr_bigint *garner_value(lctx *x, const garner *G, const uint64_t *res, uint64_t *v, uint64_t *limbs) {
    const uint64_t k = G->k;
    for (uint64_t i = 0; i < k; i++) {                         /* mixed-radix digits */
        const tr_gfp *f = G->f[i];
        uint64_t acc = 0;
        for (uint64_t j = i; j-- > 0;) acc = tr_gfp_add_hd(f, tr_gfp_mul_hd(f, acc, G->f[j]->p % f->p), v[j] % f->p);
        v[i] = i ? tr_gfp_mul_hd(f, tr_gfp_sub_hd(f, res[i], acc), G->inv[i]) : res[0];
    }
    uint64_t n = 1;                                           /* x = v0 + p0 (v1 + p1 (...)) */
    limbs[0] = v[k - 1];
    for (uint64_t i = k - 1; i-- > 0;) {
        unsigned __int128 c = v[i];
        for (uint64_t j = 0; j < n; j++) { c += (unsigned __int128)limbs[j] * G->f[i]->p; limbs[j] = (uint64_t)c; c >>= 64; }
        if (c) limbs[n++] = (uint64_t)c;
    }
    while (n > 1 && !limbs[n - 1]) n--;
    int sign = 1;
    /* symmetric: 2 x > M -> x - M */
    int gt = 0;
    {
        uint64_t top = n > G->mn ? n : G->mn;
        for (uint64_t j = top + 1; j-- > 0;) {
            uint64_t a = (j < n ? limbs[j] << 1 : 0) | (j && j - 1 < n ? limbs[j - 1] >> 63 : 0);
            uint64_t b = j < G->mn ? G->M[j] : 0;
            if (a != b) { gt = a > b; break; }
        }
    }
    if (gt) {                                                  /* |x - M| = M - x */
        unsigned __int128 br = 0;
        for (uint64_t j = 0; j < G->mn; j++) {
            unsigned __int128 t = (unsigned __int128)G->M[j] - (j < n ? limbs[j] : 0) - br;
            limbs[j] = (uint64_t)t; br = (t >> 64) ? 1 : 0;
        }
        n = G->mn;
        while (n > 1 && !limbs[n - 1]) n--;
        sign = -1;
    }
    int32_t e = 0;
    return x->rc ? NULL : zk(x, tr_bigint_from_limbs(sign, limbs, n, x->b, x->r, &e), e);
}

/* one system: choose the agreeing primes, CRT, assemble and verify; NOT_CONVERGED with *deficit when short */
static int32_t finish_one(lctx *x, const tr_lin_batch *B, uint64_t s, const tr_gfp *primes, uint64_t np,
                          const uint64_t *res, zsol *out, uint64_t *deficit, uint64_t *used) {
    const int32_t m = B->rows[s], c = B->cols[s];
    const uint64_t mc = (uint64_t)m * (uint64_t)c, stride = 4 + mc;
    /* the best key: rank, then pivot columns, then the most primes, then pivot rows */
    uint64_t best = UINT64_MAX, bestcount = 0;
    for (uint64_t q = 0; q < np; q++) {
        const uint64_t *h = res + q * stride;
        uint64_t cnt = 0;
        for (uint64_t q2 = 0; q2 < np; q2++) {
            const uint64_t *h2 = res + q2 * stride;
            cnt += h2[0] == h[0] && h2[1] == h[1] && h2[2] == h[2];
        }
        if (best == UINT64_MAX) { best = q; bestcount = cnt; continue; }
        const uint64_t *hb = res + best * stride;
        int better = h[0] > hb[0] || (h[0] == hb[0] && (mask_less(h[1], hb[1]) || (h[1] == hb[1] &&
                     (cnt > bestcount || (cnt == bestcount && mask_less(h[2], hb[2]))))));
        if (better) { best = q; bestcount = cnt; }
    }
    const uint64_t *hb = res + best * stride;
    if (bestcount < B->need[s]) { *deficit = B->need[s] - bestcount; return TR_MATH_NOT_CONVERGED; }
    const uint64_t k = B->need[s];
    uint64_t idx[MAX_PRIMES], kk = 0;
    for (uint64_t q = 0; q < np && kk < k; q++) {
        const uint64_t *h = res + q * stride;
        if (h[0] == hb[0] && h[1] == hb[1] && h[2] == hb[2]) idx[kk++] = q;
    }
    const int32_t r = (int32_t)hb[0];
    int32_t pivcol[64], np_ = 0;
    for (int32_t j = 0; j < c; j++) if ((hb[1] >> j) & 1) pivcol[np_++] = j;
    garner *G = (garner *)malloc(sizeof *G);
    uint64_t *v = (uint64_t *)malloc((size_t)k * 8), *rv = (uint64_t *)malloc((size_t)k * 8), *limbs = (uint64_t *)malloc((size_t)(k + 2) * 8);
    const tr_gfp *fs[MAX_PRIMES];
    for (uint64_t i = 0; i < k; i++) fs[i] = &primes[idx[i]];
    int32_t rc = !G || !v || !rv || !limbs ? TR_MATH_MEMORY : garner_init(G, fs, k);
    if (G && rc) G->M = NULL;
    tr_bigint *D = NULL, **DR = rc ? NULL : Z_arr(x, (uint64_t)r * (uint64_t)c);
    if (!rc && LIVE(x)) {
        for (uint64_t i = 0; i < k; i++) rv[i] = res[idx[i] * stride + 3];
        D = garner_value(x, G, rv, v, limbs);
        for (int32_t t = 0; t < r && LIVE(x); t++)
            for (int32_t j = 0; j < c && LIVE(x); j++) {
                if ((hb[1] >> j) & 1) continue;
                for (uint64_t i = 0; i < k; i++) rv[i] = res[idx[i] * stride + 4 + (uint64_t)t * c + j];
                DR[(uint64_t)t * c + j] = garner_value(x, G, rv, v, limbs);
            }
    }
    if (!rc && LIVE(x)) {
        int32_t arc = assemble(x, B->ints + B->offset[s], m, c, r, pivcol, D, DR, out);
        if (arc == TR_MATH_INVALID) { *deficit = k; rc = TR_MATH_NOT_CONVERGED; }
        else if (arc) rc = arc;
    }
    if (G) free(G->M);
    free(G); free(v); free(rv); free(limbs);
    tr_bigint_free(D);
    Z_free_all(DR, DR ? (uint64_t)r * (uint64_t)c : 0);
    *used = k;
    if (rc == TR_MATH_MEMORY) lfail(x, rc);
    return x->rc ? x->rc : rc;
}

int32_t tr_lin_batch_finish(const tr_lin_batch *B, const tr_gfp *primes, uint64_t np, const uint64_t *residues,
                            tr_lin_result **out, uint64_t *need, const tr_math_budget *budget, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!bat_valid(B) || !residues || !out || !need || !primes_ok(primes, np, 1)) return TR_MATH_INVALID;
    for (uint64_t s = 0; s < B->nsys; s++) out[s] = NULL;
    *need = 0;
    uint64_t *roff = (uint64_t *)zalloc(B->nsys * 8);
    if (!roff) return TR_MATH_MEMORY;
    batch_roff(B, np, roff);
    lctx x = {budget, refusal, 0};
    uint64_t deficit = 0;
    for (uint64_t s = 0; s < B->nsys && LIVE(&x); s++) {
        zsol z;
        memset(&z, 0, sizeof z);
        uint64_t d = 0, used = 0;
        int32_t rc = finish_one(&x, B, s, primes, np, residues + roff[s], &z, &d, &used);
        if (rc == TR_MATH_NOT_CONVERGED) { if (d > deficit) deficit = d; zsol_free(&z, B->cols[s] - 1); continue; }
        if (rc) { zsol_free(&z, B->cols[s] - 1); break; }
        const int32_t m = B->rows[s], n = B->cols[s] - 1;
        tr_bigq **cert = NULL;
        if (z.kind == TR_LIN_NONE) {
            lctx y = {budget, refusal, 0};
            cert = certificate(&y, B->ints + B->offset[s], B->scale + B->rowoff[s], m, n);
            if (y.rc == TR_MATH_INVALID) { deficit = deficit > used ? deficit : used; zsol_free(&z, n); continue; }
            if (y.rc) { lfail(&x, y.rc); zsol_free(&z, n); break; }
        }
        tr_lin_result *res = B->fits ? fit_result(&x, &B->fc[s], &z, TR_LIN_METHOD_MODULAR, used) : res_new(&x, TR_LIN_RESULT_SOLVE);
        if (!B->fits) { sol_result(&x, res, &z, m, n, TR_LIN_METHOD_MODULAR); if (res) res->info.primes = used; }
        if (res && cert) { res_q(res, TR_LIN_PART_CERT, cert, (uint64_t)m); cert = NULL; }
        tr_lin_q_free(cert, cert ? (uint64_t)m : 0);
        zsol_free(&z, n);
        out[s] = res;
    }
    free(roff);
    int32_t rc = x.rc;
    if (!rc && deficit) { rc = TR_MATH_NOT_CONVERGED; *need = np + deficit + TR_LIN_PRIME_MARGIN; }
    if (rc) for (uint64_t s = 0; s < B->nsys; s++) { tr_lin_result_free(out[s]); out[s] = NULL; }
    return rc;
}
