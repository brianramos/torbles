/* DYN-001 / DYN-002: finite closed-observable dynamics, operator powers, semigroups and recurrences (every contract:
 * math_dyn.h).
 * SPDX-License-Identifier: Apache-2.0 */
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "math_dyn.h"
#include "math_handles.h"
#include "math_lin.h"
#include "mathematics.h"
#include "math_domain.h"

TR_MATH_ABI(tr_dyn_term, TR_DYN_ABI);
TR_MATH_ABI(tr_dyn_map_info, TR_DYN_ABI);
TR_MATH_ABI(tr_dyn_map_term, TR_DYN_ABI);
TR_MATH_ABI(tr_dyn_box, TR_DYN_ABI);
TR_MATH_ABI(tr_dyn_op_info, TR_DYN_ABI);
TR_MATH_ABI(tr_dyn_power_info, TR_DYN_ABI);
TR_MATH_ABI(tr_dyn_evolve_info, TR_DYN_ABI);
TR_MATH_ABI(tr_dyn_float_info, TR_DYN_ABI);
TR_MATH_ABI(tr_dyn_rec_info, TR_DYN_ABI);
TR_MATH_ABI(tr_dyn_semigroup_evidence, TR_DYN_ABI);
TR_MATH_ABI(tr_dyn_apply_view, TR_DYN_ABI);
TR_MATH_ABI(tr_dyn_mod_view, TR_DYN_ABI);

_Static_assert(sizeof(tr_dyn_term) == 56, "tr_dyn_term");
_Static_assert(sizeof(tr_dyn_map_info) == 56, "tr_dyn_map_info");
_Static_assert(sizeof(tr_dyn_map_term) == 72, "tr_dyn_map_term");
_Static_assert(sizeof(tr_dyn_box) == 1040, "tr_dyn_box");
_Static_assert(sizeof(tr_dyn_op_info) == 104, "tr_dyn_op_info");
_Static_assert(sizeof(tr_dyn_power_info) == 112, "tr_dyn_power_info");
_Static_assert(sizeof(tr_dyn_evolve_info) == 120, "tr_dyn_evolve_info");
_Static_assert(sizeof(tr_dyn_float_info) == 56, "tr_dyn_float_info");
_Static_assert(sizeof(tr_dyn_rec_info) == 48, "tr_dyn_rec_info");
_Static_assert(sizeof(tr_dyn_semigroup_evidence) == 48, "tr_dyn_semigroup_evidence");
_Static_assert(sizeof(tr_dyn_apply_view) == 32, "tr_dyn_apply_view");
_Static_assert(sizeof(tr_dyn_mod_view) == 40, "tr_dyn_mod_view");

#define MAP_MAGIC 0x44594e4du
#define OP_MAGIC 0x44594e4fu
#define POW_MAGIC 0x44594e50u
#define EV_MAGIC 0x44594e45u
#define REC_MAGIC 0x44594e52u
#define NC TR_DYN_MAX_COORDS

typedef __int128 i128;
typedef unsigned __int128 u128;
#define I128_MAX ((i128)(((u128)1 << 127) - 1))

/* ------------------------------------------------------------------ small helpers */
static void set_error(int32_t *e, int32_t v) { if (e) *e = v; }
static void *zalloc(uint64_t bytes) {
    if (bytes > (uint64_t)SIZE_MAX / 2) return NULL;
    return calloc(1, bytes ? (size_t)bytes : 1);
}
static const tr_math_budget *budget_or(const tr_math_budget *b, tr_math_budget *d) {
    if (b) return b;
    tr_budget_default(d);
    return d;
}
static void need_zero(tr_math_need *n) { memset(n, 0, sizeof *n); }
static int32_t check_need(const tr_math_budget *budget, const tr_math_need *need, tr_math_refusal *refusal) {
    tr_math_budget dflt;
    return tr_budget_check(budget_or(budget, &dflt), need, refusal);
}
static void unlimited(tr_math_budget *b) { for (int i = 0; i < TR_BUDGET_RESOURCES; i++) b->limit[i] = UINT64_MAX; }
static u128 mag128(i128 x) { return x < 0 ? (u128)0 - (u128)x : (u128)x; }
static u128 gcd128(u128 a, u128 b) {
    while (b) { u128 t = a % b; a = b; b = t; }
    return a;
}
static int add128(i128 a, i128 b, i128 *r) { return !__builtin_add_overflow(a, b, r); }
static int mul128(i128 a, i128 b, i128 *r) { return !__builtin_mul_overflow(a, b, r); }
static int in_range(i128 x) { return x != -I128_MAX - 1; }     /* exact values stay in |x| <= 2^127 - 1 */
static i128 pow10_128(int k) { i128 r = 1; while (k-- > 0) r *= 10; return r; }
static int lcm128(i128 a, i128 b, i128 *out) {                 /* a, b > 0 */
    u128 g = gcd128((u128)a, (u128)b);
    return mul128(a / (i128)g, b, out);
}
static uint64_t bitlen64(uint64_t x) { return x ? (uint64_t)(64 - __builtin_clzll(x)) : 0; }
static uint64_t bitlen128(u128 x) { return (uint64_t)(x >> 64) ? 64 + bitlen64((uint64_t)(x >> 64)) : bitlen64((uint64_t)x); }
static uint64_t limbs_for(uint64_t bits) { return bits > TR_BUDGET_UNKNOWN - 64 ? TR_BUDGET_UNKNOWN : (bits + 63) / 64 + 1; }
static uint64_t popcount64(uint64_t x) { return (uint64_t)__builtin_popcountll(x); }
static uint64_t floorlog2(uint64_t x) { return x ? bitlen64(x) - 1 : 0; }
static int geo_equal(const tr_sym_geometry *a, const tr_sym_geometry *b) {
    if (a->n != b->n || a->ncols != b->ncols || a->digest[0] != b->digest[0] || a->digest[1] != b->digest[1]) return 0;
    for (int32_t i = 0; i < a->n; i++)
        if (a->col[i] != b->col[i] || a->kind[i] != b->kind[i] || a->unit[i] != b->unit[i] ||
            a->scale[i] != b->scale[i]) return 0;
    return 1;
}

/* exact rationals with int128 parts (d > 0, reduced) */
typedef struct { i128 n, d; } q128;
static int q_make(i128 n, i128 d, q128 *out) {
    if (d == 0 || !in_range(n) || !in_range(d)) return 0;
    if (d < 0) { n = -n; d = -d; }
    u128 g = gcd128(mag128(n), (u128)d);
    if (g > 1) { n /= (i128)g; d /= (i128)g; }
    out->n = n; out->d = d;
    return 1;
}
static int q_mul(q128 a, q128 b, q128 *out) {
    u128 g1 = gcd128(mag128(a.n), (u128)b.d), g2 = gcd128(mag128(b.n), (u128)a.d);
    if (!g1) g1 = 1;
    if (!g2) g2 = 1;
    i128 n, d;
    if (!mul128(a.n / (i128)g1, b.n / (i128)g2, &n) || !mul128(a.d / (i128)g2, b.d / (i128)g1, &d)) return 0;
    return q_make(n, d, out);
}
static int q_add(q128 a, q128 b, q128 *out) {
    i128 x, y, n, d;
    if (!mul128(a.n, b.d, &x) || !mul128(b.n, a.d, &y) || !add128(x, y, &n) || !mul128(a.d, b.d, &d)) return 0;
    return q_make(n, d, out);
}
static int q_from(tr_num_rational r, q128 *out) { return q_make(r.numerator, r.denominator, out); }
static int q_scale10(q128 v, int k, q128 *out) {               /* v 10^k */
    q128 f;
    if (k >= 0) { f.n = pow10_128(k); f.d = 1; } else { f.n = 1; f.d = pow10_128(-k); }
    return q_mul(v, f, out);
}
static int q_cmp(q128 a, q128 b, int *out) {                    /* -1, 0, 1 */
    i128 x, y;
    if (!mul128(a.n, b.d, &x) || !mul128(b.n, a.d, &y)) return 0;
    *out = x < y ? -1 : x > y;
    return 1;
}

/* ================================================================== big integer vectors
 * count numbers of fixed capacity cap limbs each, sign and magnitude (len significant limbs, neg only when nonzero).
 * Products are accumulated in two unsigned magnitude accumulators (positive and negative parts), so carries stop at
 * the top of what was written; the difference is stored at the end. The capacities come from proven bounds; a value
 * that does not fit is reported (TR_MATH_OVERFLOW), never truncated. */
typedef struct { uint64_t count, cap; uint64_t *limb; uint32_t *len; uint8_t *neg; } bvec;
static uint64_t bvec_bytes(uint64_t count, uint64_t cap) {
    return tr_budget_add(tr_budget_mul(count, tr_budget_add(tr_budget_mul(cap, 8), 5)), 64);
}
static int bvec_init(bvec *v, uint64_t count, uint64_t cap) {
    memset(v, 0, sizeof *v);
    if (!cap) cap = 1;
    if (!count) count = 1;
    uint64_t limbs = tr_budget_mul(count, cap);
    if (limbs == TR_BUDGET_UNKNOWN || limbs > ((uint64_t)1 << 40)) return 0;
    v->limb = (uint64_t *)zalloc(limbs * 8);
    v->len = (uint32_t *)zalloc(count * 4);
    v->neg = (uint8_t *)zalloc(count);
    if (!v->limb || !v->len || !v->neg) { free(v->limb); free(v->len); free(v->neg); memset(v, 0, sizeof *v); return 0; }
    v->count = count; v->cap = cap;
    return 1;
}
static void bvec_free(bvec *v) { if (v) { free(v->limb); free(v->len); free(v->neg); memset(v, 0, sizeof *v); } }
#define BV(v, i) ((v)->limb + (uint64_t)(i) * (v)->cap)
static void bv_set_mag(bvec *v, uint64_t i, int neg, const uint64_t *l, uint64_t n) {
    while (n && !l[n - 1]) n--;
    uint64_t *d = BV(v, i);
    for (uint64_t k = 0; k < n; k++) d[k] = l[k];
    v->len[i] = (uint32_t)n; v->neg[i] = (uint8_t)(n && neg);
}
static void bv_set_i128(bvec *v, uint64_t i, i128 x) {
    u128 m = mag128(x);
    uint64_t l[2] = {(uint64_t)m, (uint64_t)(m >> 64)};
    bv_set_mag(v, i, x < 0, l, v->cap >= 2 ? 2 : 1);
}
static int bv_copy(bvec *d, uint64_t i, const bvec *s, uint64_t j) {
    if (s->len[j] > d->cap) return 0;
    bv_set_mag(d, i, s->neg[j], BV(s, j), s->len[j]);
    return 1;
}
static uint64_t bv_bitlen(const bvec *v, uint64_t i) {
    uint32_t n = v->len[i];
    return n ? 64 * (uint64_t)(n - 1) + bitlen64(BV(v, i)[n - 1]) : 0;
}
static uint64_t i128_mag(i128 x, uint64_t l[2], int *neg) {
    u128 m = mag128(x);
    l[0] = (uint64_t)m; l[1] = (uint64_t)(m >> 64); *neg = x < 0;
    return l[1] ? 2 : (l[0] ? 1 : 0);
}
static int bv_from_bigint(bvec *v, uint64_t i, const tr_bigint *x) {
    int32_t sign = 0;
    uint64_t count = 0;
    if (tr_bigint_export(x, &sign, BV(v, i), v->cap, &count)) return 0;
    v->len[i] = (uint32_t)count; v->neg[i] = (uint8_t)(count && sign < 0);
    return 1;
}
static tr_bigint *bv_to_bigint(const bvec *v, uint64_t i, int32_t *err) {
    tr_math_budget u;
    unlimited(&u);
    return tr_bigint_from_limbs(v->len[i] ? (v->neg[i] ? -1 : 1) : 0, BV(v, i), v->len[i], &u, NULL, err);
}

/* acc[0..W) += a b (magnitudes); returns the index past the highest limb written; *over when a carry leaves W */
static uint64_t mag_addmul(uint64_t *acc, uint64_t W, const uint64_t *a, uint64_t na, const uint64_t *b, uint64_t nb,
                           int *over) {
    uint64_t top = 0;
    for (uint64_t i = 0; i < na; i++) {
        uint64_t ai = a[i];
        if (!ai) continue;
        if (i + nb > W) { *over = 1; return W; }
        u128 carry = 0;
        for (uint64_t j = 0; j < nb; j++) {
            u128 t = (u128)ai * b[j] + acc[i + j] + carry;
            acc[i + j] = (uint64_t)t; carry = t >> 64;
        }
        uint64_t k = i + nb;
        while (carry) {
            if (k >= W) { *over = 1; return W; }
            u128 t = (u128)acc[k] + carry;
            acc[k] = (uint64_t)t; carry = t >> 64; k++;
        }
        if (k > top) top = k;
    }
    return top;
}
typedef struct { uint64_t W, tp, tn; uint64_t *pos, *neg; int over; } acc_t;
static int acc_init(acc_t *a, uint64_t W) {
    memset(a, 0, sizeof *a);
    a->W = W ? W : 1;
    a->pos = (uint64_t *)zalloc(a->W * 8); a->neg = (uint64_t *)zalloc(a->W * 8);
    if (!a->pos || !a->neg) { free(a->pos); free(a->neg); memset(a, 0, sizeof *a); return 0; }
    return 1;
}
static void acc_free(acc_t *a) { free(a->pos); free(a->neg); memset(a, 0, sizeof *a); }
static void acc_clear(acc_t *a) {
    memset(a->pos, 0, (size_t)a->tp * 8); memset(a->neg, 0, (size_t)a->tn * 8);
    a->tp = a->tn = 0; a->over = 0;
}
static void acc_mul(acc_t *a, int neg, const uint64_t *x, uint64_t nx, const uint64_t *y, uint64_t ny) {
    if (!nx || !ny) return;
    uint64_t top = mag_addmul(neg ? a->neg : a->pos, a->W, x, nx, y, ny, &a->over);
    if (neg) { if (top > a->tn) a->tn = top; } else if (top > a->tp) a->tp = top;
}
static void acc_bb(acc_t *a, const bvec *x, uint64_t i, const bvec *y, uint64_t j) {
    acc_mul(a, x->neg[i] ^ y->neg[j], BV(x, i), x->len[i], BV(y, j), y->len[j]);
}
static void acc_ib(acc_t *a, i128 c, const bvec *y, uint64_t j) {
    uint64_t l[2];
    int neg;
    uint64_t n = i128_mag(c, l, &neg);
    acc_mul(a, neg ^ y->neg[j], l, n, BV(y, j), y->len[j]);
}
/* dst[i] = pos - neg; 0 when it does not fit dst (or the accumulator overflowed) */
static int acc_store(acc_t *a, bvec *d, uint64_t i) {
    if (a->over) { acc_clear(a); return 0; }
    uint64_t n = a->tp > a->tn ? a->tp : a->tn;
    int cmp = 0;
    for (uint64_t k = n; k-- > 0;) if (a->pos[k] != a->neg[k]) { cmp = a->pos[k] > a->neg[k] ? 1 : -1; break; }
    uint64_t *big = cmp >= 0 ? a->pos : a->neg, *small = cmp >= 0 ? a->neg : a->pos;
    uint64_t borrow = 0;
    for (uint64_t k = 0; k < n; k++) {
        uint64_t x = big[k], y = small[k];
        uint64_t r = x - y - borrow;
        borrow = (x < y) || (x - y < borrow);
        big[k] = r;
    }
    while (n && !big[n - 1]) n--;
    int ok = n <= d->cap;
    if (ok) bv_set_mag(d, i, cmp < 0, big, n);
    /* both accumulators back to zero (big was overwritten by the difference) */
    a->tp = a->tp > a->tn ? a->tp : a->tn; a->tn = a->tp;
    acc_clear(a);
    return ok;
}

/* ================================================================== maps */
typedef struct { uint8_t e[NC]; int32_t p, coord; i128 c; } mterm;   /* c over q[coord] (logical units) */
struct tr_dyn_map_s {
    uint32_t magic;
    int32_t n, kind, degree, pdegree, parametric, nterms;
    uint32_t touched;
    tr_sym_geometry geo;
    tr_sym_map *sym;                       /* AFFINE: a private copy of the SYM-001 map */
    mterm *terms;                          /* sorted by coordinate, parameter exponent, exponents */
    int32_t first[NC + 1];                 /* the terms of coordinate i: [first[i], first[i + 1]) */
    i128 q[NC];
    uint64_t bytes;
};
static int map_ok(const tr_dyn_map *m) { return m && m->magic == MAP_MAGIC; }
static tr_dyn_map *map_fail(tr_dyn_map *m, int32_t *error, int32_t code) {
    if (m) { tr_sym_map_free(m->sym); free(m->terms); free(m); }
    set_error(error, code);
    return NULL;
}
static int mterm_cmp(const void *a, const void *b) {
    const mterm *x = (const mterm *)a, *y = (const mterm *)b;
    if (x->coord != y->coord) return x->coord < y->coord ? -1 : 1;
    if (x->p != y->p) return x->p < y->p ? -1 : 1;
    return memcmp(x->e, y->e, NC);
}
typedef struct { mterm t; q128 q; } mq;
static int mq_cmp(const void *a, const void *b) { return mterm_cmp(&((const mq *)a)->t, &((const mq *)b)->t); }
static int32_t mterm_degree(const mterm *t) { int32_t d = 0; for (int32_t i = 0; i < NC; i++) d += t->e[i]; return d; }
/* the coordinates a term list reads */
static uint32_t map_reads(const tr_dyn_map *m, int32_t i) {
    uint32_t r = 0;
    for (int32_t t = m->first[i]; t < m->first[i + 1]; t++)
        for (int32_t j = 0; j < m->n; j++) if (m->terms[t].e[j]) r |= UINT32_C(1) << j;
    return r;
}
static void map_finish(tr_dyn_map *m) {
    qsort(m->terms, (size_t)m->nterms, sizeof(mterm), mterm_cmp);
    int32_t at = 0;
    m->degree = 0; m->pdegree = 0;
    for (int32_t i = 0; i <= m->n; i++) {
        while (at < m->nterms && m->terms[at].coord < i) at++;
        m->first[i] = at;
    }
    for (int32_t t = 0; t < m->nterms; t++) {
        int32_t d = mterm_degree(&m->terms[t]);
        if (d > m->degree) m->degree = d;
        if (m->terms[t].p > m->pdegree) m->pdegree = m->terms[t].p;
    }
    m->touched = 0;
    for (int32_t i = 0; i < m->n; i++) {
        int id = m->first[i + 1] - m->first[i] == 1 && m->q[i] == 1;
        if (id) {
            const mterm *t = &m->terms[m->first[i]];
            id = t->c == 1 && t->p == 0 && mterm_degree(t) == 1 && t->e[i] == 1;
        }
        if (!id) m->touched |= UINT32_C(1) << i;
    }
    m->kind = (m->degree <= 1 && !m->parametric) ? (m->sym ? TR_DYN_MAP_AFFINE : TR_DYN_MAP_POLYNOMIAL) : TR_DYN_MAP_POLYNOMIAL;
    m->bytes = sizeof *m + (uint64_t)m->nterms * sizeof(mterm);
}
tr_dyn_map *tr_dyn_map_from_sym(const tr_sym_map *s, const tr_sym_geometry *g, const tr_math_budget *budget,
                                tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    tr_sym_map_info mi;
    if (!s || tr_sym_map_get_info(s, &mi) || tr_sym_geometry_check(g)) return map_fail(NULL, error, TR_MATH_INVALID);
    if (mi.n != g->n) return map_fail(NULL, error, TR_MATH_INVALID);
    if (mi.kind != TR_SYM_MAP_AFFINE || mi.carrier != TR_SYM_EXACT) return map_fail(NULL, error, TR_MATH_UNSUPPORTED);
    const int32_t n = g->n;
    tr_math_need need;
    need_zero(&need);
    need.need[TR_BUDGET_VARIABLES] = (uint64_t)n;
    need.need[TR_BUDGET_DEGREE] = 1;
    need.need[TR_BUDGET_SUPPORT] = (uint64_t)n * (uint64_t)(n + 1);
    need.need[TR_BUDGET_BYTES] = sizeof(tr_dyn_map) + (uint64_t)n * (uint64_t)(n + 1) * sizeof(mterm) + 2 * sizeof(tr_sym_geometry);
    int32_t rc = check_need(budget, &need, refusal);
    if (rc) return map_fail(NULL, error, rc);
    tr_math_i128 *a = (tr_math_i128 *)zalloc((uint64_t)n * n * sizeof(tr_math_i128));
    tr_math_i128 b[NC], q[NC];
    tr_dyn_map *m = (tr_dyn_map *)zalloc(sizeof *m);
    if (m) m->terms = (mterm *)zalloc((uint64_t)n * (uint64_t)(n + 1) * sizeof(mterm));
    if (!a || !m || !m->terms) { free(a); return map_fail(m, error, TR_MATH_MEMORY); }
    rc = tr_sym_map_get_exact(s, a, b, q);
    if (rc) { free(a); return map_fail(m, error, rc); }
    m->n = n; m->geo = *g;
    tr_sym_map *id = tr_sym_map_identity(n, &rc);
    m->sym = id ? tr_sym_map_compose(id, s, NULL, NULL, &rc) : NULL;   /* a private copy (the identity is geometry-free) */
    tr_sym_map_free(id);
    if (!m->sym) { free(a); return map_fail(m, error, rc ? rc : TR_MATH_MEMORY); }
    for (int32_t i = 0; i < n; i++) {
        m->q[i] = tr_i128_get(q[i]);
        for (int32_t j = 0; j <= n; j++) {
            i128 c = j < n ? tr_i128_get(a[(uint64_t)i * n + j]) : tr_i128_get(b[i]);
            if (!c) continue;
            mterm *t = &m->terms[m->nterms++];
            memset(t, 0, sizeof *t);
            t->coord = i; t->c = c;
            if (j < n) t->e[j] = 1;
        }
    }
    free(a);
    map_finish(m);
    m->kind = TR_DYN_MAP_AFFINE;
    m->magic = MAP_MAGIC;
    set_error(error, TR_MATH_OK);
    return m;
}
tr_dyn_map *tr_dyn_map_polynomial(const tr_sym_geometry *g, const tr_dyn_term *terms, int32_t nterms, int32_t parametric,
                                  const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    int32_t rc = tr_sym_geometry_check(g);
    if (rc) return map_fail(NULL, error, rc);
    if (nterms < 0 || (nterms && !terms) || (parametric != 0 && parametric != 1)) return map_fail(NULL, error, TR_MATH_INVALID);
    if (nterms > TR_DYN_MAX_TERMS) return map_fail(NULL, error, TR_MATH_UNSUPPORTED);
    const int32_t n = g->n;
    int32_t deg = 0;
    uint32_t declared = 0;
    for (int32_t t = 0; t < nterms; t++) {                       /* the declared geometry the terms need */
        const tr_dyn_term *x = &terms[t];
        if (x->coord < 0 || x->coord >= n || x->pexp < 0 || (x->pexp && !parametric) || x->coef.denominator == 0)
            return map_fail(NULL, error, TR_MATH_INVALID);
        int32_t d = 0;
        for (int32_t j = 0; j < NC; j++) {
            if (x->exps[j] && j >= n) return map_fail(NULL, error, TR_MATH_INVALID);
            d += x->exps[j];
        }
        if (d > TR_DYN_MAX_DEGREE || x->pexp > TR_DYN_MAX_PDEGREE) return map_fail(NULL, error, TR_MATH_UNSUPPORTED);
        if (g->kind[x->coord] != TR_SYM_NUMERIC) return map_fail(NULL, error, TR_MATH_DOMAIN_ERROR);
        for (int32_t j = 0; j < n; j++)
            if (x->exps[j] && g->kind[j] != TR_SYM_NUMERIC) return map_fail(NULL, error, TR_MATH_DOMAIN_ERROR);
        if (d > deg) deg = d;
        declared |= UINT32_C(1) << x->coord;
    }
    uint64_t cap = (uint64_t)nterms + (uint64_t)n;
    tr_math_need need;
    need_zero(&need);
    need.need[TR_BUDGET_VARIABLES] = (uint64_t)n;
    need.need[TR_BUDGET_DEGREE] = (uint64_t)deg;
    need.need[TR_BUDGET_SUPPORT] = cap;
    need.need[TR_BUDGET_BYTES] = sizeof(tr_dyn_map) + tr_budget_mul(cap, sizeof(mterm) + sizeof(q128));
    rc = check_need(budget, &need, refusal);
    if (rc) return map_fail(NULL, error, rc);
    tr_dyn_map *m = (tr_dyn_map *)zalloc(sizeof *m);
    q128 *qc = (q128 *)zalloc(cap * sizeof(q128));
    if (m) m->terms = (mterm *)zalloc(cap * sizeof(mterm));
    if (!m || !m->terms || !qc) { free(qc); return map_fail(m, error, TR_MATH_MEMORY); }
    m->n = n; m->geo = *g; m->parametric = parametric;
    int32_t k = 0;
    for (int32_t t = 0; t < nterms && !rc; t++) {                /* value units -> logical: c 10^(s_i - sum s_j e_j) */
        const tr_dyn_term *x = &terms[t];
        int32_t sh = g->scale[x->coord];
        for (int32_t j = 0; j < n; j++) sh -= g->scale[j] * (int32_t)x->exps[j];
        q128 c;
        if (!q_from(x->coef, &c)) { rc = TR_MATH_INVALID; break; }
        if (!q_scale10(c, sh, &qc[k])) { rc = TR_MATH_OVERFLOW; break; }
        mterm *mt = &m->terms[k++];
        memset(mt, 0, sizeof *mt);
        mt->coord = x->coord; mt->p = x->pexp;
        memcpy(mt->e, x->exps, NC);
    }
    for (int32_t i = 0; i < n && !rc; i++) {                     /* undeclared coordinates are fixed: x_i' = x_i */
        if ((declared >> i) & 1) continue;
        mterm *mt = &m->terms[k];
        memset(mt, 0, sizeof *mt);
        mt->coord = i; mt->e[i] = 1;
        qc[k].n = 1; qc[k].d = 1;
        k++;
    }
    if (rc) { free(qc); return map_fail(m, error, rc); }
    /* merge equal monomials (the pairs sorted by the term order), drop zeros, one denominator per coordinate */
    mq *pairs = (mq *)zalloc((uint64_t)(k ? k : 1) * sizeof(mq));
    mterm *sorted = (mterm *)zalloc((uint64_t)(k ? k : 1) * sizeof(mterm));
    q128 *sq = (q128 *)zalloc((uint64_t)(k ? k : 1) * sizeof(q128));
    if (!pairs || !sorted || !sq) { free(pairs); free(sorted); free(sq); free(qc); return map_fail(m, error, TR_MATH_MEMORY); }
    for (int32_t t = 0; t < k; t++) { pairs[t].t = m->terms[t]; pairs[t].q = qc[t]; }
    qsort(pairs, (size_t)k, sizeof(mq), mq_cmp);
    int32_t out = 0;
    for (int32_t a = 0; a < k && !rc;) {
        q128 s = pairs[a].q;
        int32_t b = a + 1;
        for (; b < k && !mterm_cmp(&pairs[a].t, &pairs[b].t); b++)
            if (!q_add(s, pairs[b].q, &s)) { rc = TR_MATH_OVERFLOW; break; }
        if (!rc && s.n) { sorted[out] = pairs[a].t; sq[out] = s; out++; }
        a = b;
    }
    free(pairs); free(qc);
    if (rc) { free(sorted); free(sq); return map_fail(m, error, rc); }
    for (int32_t i = 0; i < n; i++) m->q[i] = 1;
    for (int32_t t = 0; t < out && !rc; t++) if (!lcm128(m->q[sorted[t].coord], sq[t].d, &m->q[sorted[t].coord])) rc = TR_MATH_OVERFLOW;
    for (int32_t t = 0; t < out && !rc; t++) {
        i128 v;
        if (!mul128(sq[t].n, m->q[sorted[t].coord] / sq[t].d, &v) || !in_range(v)) rc = TR_MATH_OVERFLOW;
        sorted[t].c = v;
    }
    if (!rc) {
        u128 g2[NC];
        for (int32_t i = 0; i < n; i++) g2[i] = (u128)m->q[i];
        for (int32_t t = 0; t < out; t++) g2[sorted[t].coord] = gcd128(g2[sorted[t].coord], mag128(sorted[t].c));
        for (int32_t t = 0; t < out; t++) if (g2[sorted[t].coord] > 1) sorted[t].c /= (i128)g2[sorted[t].coord];
        for (int32_t i = 0; i < n; i++) if (g2[i] > 1) m->q[i] /= (i128)g2[i];
    }
    free(sq);
    if (rc) { free(sorted); return map_fail(m, error, rc); }
    free(m->terms);
    m->terms = sorted; m->nterms = out;
    map_finish(m);
    m->magic = MAP_MAGIC;
    set_error(error, TR_MATH_OK);
    return m;
}
int32_t tr_dyn_map_get_info(const tr_dyn_map *m, tr_dyn_map_info *out) {
    if (!map_ok(m) || !out) return TR_MATH_INVALID;
    memset(out, 0, sizeof *out);
    out->n = m->n; out->kind = m->kind; out->degree = m->degree; out->pdegree = m->pdegree;
    out->nterms = m->nterms; out->parametric = m->parametric; out->touched = m->touched;
    out->digest[0] = m->geo.digest[0]; out->digest[1] = m->geo.digest[1]; out->bytes = m->bytes;
    return TR_MATH_OK;
}
int32_t tr_dyn_map_get_terms(const tr_dyn_map *m, tr_dyn_map_term *out, uint64_t capacity) {
    if (!map_ok(m) || (m->nterms && !out) || capacity < (uint64_t)m->nterms) return TR_MATH_INVALID;
    for (int32_t t = 0; t < m->nterms; t++) {
        memset(&out[t], 0, sizeof out[t]);
        out[t].coord = m->terms[t].coord; out[t].pexp = m->terms[t].p;
        memcpy(out[t].exps, m->terms[t].e, NC);
        out[t].num = tr_i128_set(m->terms[t].c); out[t].den = tr_i128_set(m->q[m->terms[t].coord]);
    }
    return TR_MATH_OK;
}
void tr_dyn_map_free(tr_dyn_map *m) {
    if (!map_ok(m)) return;
    m->magic = 0;
    map_fail(m, NULL, 0);
}

/* ================================================================== expansion of phi_a(F(x)) */
typedef struct { uint64_t w[4]; uint32_t p, pad; } ekey;        /* 8-bit exponents of 32 coordinates; theta's */
typedef struct { ekey k; i128 c; } eterm;
static int ekey_cmp(const ekey *a, const ekey *b) {
    if (a->p != b->p) return a->p < b->p ? -1 : 1;
    for (int i = 3; i >= 0; i--) if (a->w[i] != b->w[i]) return a->w[i] < b->w[i] ? -1 : 1;
    return 0;
}
static int eterm_cmp(const void *a, const void *b) { return ekey_cmp(&((const eterm *)a)->k, &((const eterm *)b)->k); }
static void ekey_set(ekey *k, const uint8_t *e, uint32_t p) {
    memset(k, 0, sizeof *k);
    for (int32_t i = 0; i < NC; i++) k->w[i >> 3] |= (uint64_t)e[i] << (8 * (i & 7));
    k->p = p;
}
static void ekey_exps(const ekey *k, uint8_t *e) { for (int32_t i = 0; i < NC; i++) e[i] = (uint8_t)(k->w[i >> 3] >> (8 * (i & 7))); }
/* the factor lists F_i as expansion terms */
typedef struct { eterm *t; int32_t nt, degree, pdegree; uint32_t reads; } factor;
/* P (np terms) times F into out: products, sorted, merged, zeros dropped; TR_MATH_OVERFLOW past int128 */
static int32_t exp_mul(const eterm *P, uint64_t np, const factor *F, eterm *out, uint64_t *nout) {
    uint64_t k = 0;
    for (uint64_t a = 0; a < np; a++)
        for (int32_t b = 0; b < F->nt; b++) {
            eterm *o = &out[k++];
            for (int w = 0; w < 4; w++) o->k.w[w] = P[a].k.w[w] + F->t[b].k.w[w];   /* bytes never carry (<= 64) */
            o->k.p = P[a].k.p + F->t[b].k.p; o->k.pad = 0;
            if (!mul128(P[a].c, F->t[b].c, &o->c) || !in_range(o->c)) return TR_MATH_OVERFLOW;
        }
    if (F->nt > 1) qsort(out, (size_t)k, sizeof *out, eterm_cmp);
    uint64_t m = 0;
    for (uint64_t i = 0; i < k;) {
        i128 s = out[i].c;
        uint64_t j = i + 1;
        for (; j < k && !ekey_cmp(&out[j].k, &out[i].k); j++)
            if (!add128(s, out[j].c, &s) || !in_range(s)) return TR_MATH_OVERFLOW;
        if (s) { out[m].k = out[i].k; out[m].c = s; m++; }
        i = j;
    }
    *nout = m;
    return TR_MATH_OK;
}
/* upper bounds of one row's expansion: the largest product step and the terms (tb), for exponents e */
static uint64_t exp_cap(const factor *F, const uint8_t *e, int32_t n) {
    uint32_t vars = 0;
    for (int32_t i = 0; i < n; i++) if (e[i]) vars |= F[i].reads;
    uint64_t v = popcount64(vars), cap = 1, tb = 1, deg = 0, pdeg = 0;
    for (int32_t i = 0; i < n; i++)
        for (int32_t r = 0; r < e[i]; r++) {
            uint64_t prod = tr_budget_mul(tb, (uint64_t)(F[i].nt ? F[i].nt : 1));
            if (prod > cap) cap = prod;
            deg += (uint64_t)F[i].degree; pdeg += (uint64_t)F[i].pdegree;
            uint64_t lim = tr_budget_mul(tr_budget_monomials((uint32_t)v, (uint32_t)deg), pdeg + 1);
            tb = prod < lim ? prod : lim;
            if (!tb) tb = 1;
        }
    return cap > tb ? cap : tb;
}
static int32_t exp_row(const factor *F, const uint8_t *e, int32_t n, eterm *P, eterm *tmp, uint64_t *np) {
    memset(&P[0], 0, sizeof P[0]);
    P[0].c = 1;
    uint64_t cnt = 1;
    for (int32_t i = 0; i < n; i++)
        for (int32_t r = 0; r < e[i]; r++) {
            uint64_t m = 0;
            int32_t rc = exp_mul(P, cnt, &F[i], tmp, &m);
            if (rc) return rc;
            memcpy(P, tmp, (size_t)m * sizeof *P);
            cnt = m;
            if (!cnt) { *np = 0; return TR_MATH_OK; }
        }
    *np = cnt;
    return TR_MATH_OK;
}
static int32_t factors_make(const tr_dyn_map *m, factor *F, eterm **store) {
    eterm *s = (eterm *)zalloc((uint64_t)(m->nterms ? m->nterms : 1) * sizeof(eterm));
    if (!s) return TR_MATH_MEMORY;
    for (int32_t i = 0; i < m->n; i++) {
        F[i].t = s + m->first[i]; F[i].nt = m->first[i + 1] - m->first[i];
        F[i].degree = 0; F[i].pdegree = 0; F[i].reads = map_reads(m, i);
        for (int32_t t = m->first[i]; t < m->first[i + 1]; t++) {
            ekey_set(&s[t].k, m->terms[t].e, (uint32_t)m->terms[t].p);
            s[t].c = m->terms[t].c;
            int32_t d = mterm_degree(&m->terms[t]);
            if (d > F[i].degree) F[i].degree = d;
            if (m->terms[t].p > F[i].pdegree) F[i].pdegree = m->terms[t].p;
        }
    }
    *store = s;
    return TR_MATH_OK;
}
/* the coordinates the evolution of the basis observables reads: the basis coordinates, closed under "F_i reads x_j" */
static uint32_t reach_set(const tr_dyn_map *m, uint32_t coords) {
    uint32_t r = coords, prev = 0;
    while (r != prev) {
        prev = r;
        for (int32_t i = 0; i < m->n; i++) if ((r >> i) & 1) r |= map_reads(m, i);
    }
    return r;
}

/* ================================================================== operators */
struct tr_dyn_operator_s {
    uint32_t magic;
    int32_t nb, n, kind, label, source, pdegree, sym_action, invariant, closed;
    tr_sym_geometry geo;
    tr_sym_basis *basis;                   /* an owned clone */
    i128 den;
    tr_math_i128 *num;                     /* (pdegree + 1) x nb x nb */
    tr_dyn_box box;                        /* TRUNCATED: the contract (value units) */
    q128 blo[NC], bhi[NC];                 /* TRUNCATED: the box in logical units */
    uint32_t bounded;                      /* TRUNCATED: the coordinates the contract covers (the reach set) */
    tr_bigq **eps;                         /* TRUNCATED: nb certified bounds of the dropped terms (logical units) */
    uint64_t dropped, nonzeros, norm_bits, bytes;
    uint64_t nnz0, pairs;                  /* N_0's nonzeros and nonzero products sum_k colnnz_k rownnz_k (AUTO) */
};
static int op_ok(const tr_dyn_operator *o) { return o && o->magic == OP_MAGIC; }
static void op_release(tr_dyn_operator *o) {
    if (!o) return;
    tr_sym_basis_free(o->basis);
    free(o->num);
    if (o->eps) { for (int32_t a = 0; a < o->nb; a++) tr_bigq_free(o->eps[a]); free(o->eps); }
    free(o);
}
static tr_dyn_operator *op_fail(tr_dyn_operator *o, int32_t *error, int32_t code) {
    op_release(o);
    set_error(error, code);
    return NULL;
}
static uint64_t op_bytes(int32_t nb, int32_t pdeg, uint64_t basis_bytes) {
    return tr_budget_add(tr_budget_add(sizeof(tr_dyn_operator), basis_bytes),
                         tr_budget_mul(tr_budget_mul(tr_budget_mul((uint64_t)nb, (uint64_t)nb), (uint64_t)pdeg + 1), 16));
}
static uint64_t basis_bytes_of(const tr_sym_basis *b) {
    tr_sym_basis_info bi;
    return tr_sym_basis_get_info(b, &bi) ? 0 : bi.bytes;
}
/* nonzeros and bitlen(||N||_inf) (parametric: of sum_j |N_j|) */
static void op_stats(tr_dyn_operator *o) {
    const uint64_t nb = (uint64_t)o->nb, e = nb * nb;
    o->nonzeros = 0; o->norm_bits = 0; o->nnz0 = 0; o->pairs = 0;
    for (uint64_t i = 0; i < (uint64_t)(o->pdegree + 1) * e; i++) if (tr_i128_get(o->num[i])) o->nonzeros++;
    for (uint64_t k = 0; k < nb; k++) {                          /* the nonzero products of N_0 N_0 */
        uint64_t col = 0, row = 0;
        for (uint64_t i = 0; i < nb; i++) { col += tr_i128_get(o->num[i * nb + k]) != 0; row += tr_i128_get(o->num[k * nb + i]) != 0; }
        o->nnz0 += row;
        o->pairs += col * row;
    }
    for (uint64_t i = 0; i < nb; i++) {
        u128 s = 0;
        uint64_t carry = 0;
        for (int32_t j = 0; j <= o->pdegree; j++)
            for (uint64_t c = 0; c < nb; c++) {
                u128 v = mag128(tr_i128_get(o->num[(uint64_t)j * e + i * nb + c]));
                u128 t = s + v;
                if (t < s) carry++;
                s = t;
            }
        uint64_t bits = carry ? 128 + bitlen64(carry) : bitlen128(s);
        if (bits > o->norm_bits) o->norm_bits = bits;
    }
}
/* gcd-reduces D and every numerator */
static void op_reduce(tr_dyn_operator *o) {
    u128 g = (u128)o->den;
    uint64_t e = (uint64_t)(o->pdegree + 1) * (uint64_t)o->nb * (uint64_t)o->nb;
    for (uint64_t i = 0; i < e && g > 1; i++) g = gcd128(g, mag128(tr_i128_get(o->num[i])));
    if (g <= 1) return;
    for (uint64_t i = 0; i < e; i++) o->num[i] = tr_i128_set(tr_i128_get(o->num[i]) / (i128)g);
    o->den /= (i128)g;
}
static tr_dyn_operator *op_alloc(const tr_sym_basis *b, const tr_sym_geometry *g, int32_t pdeg, int32_t *rc) {
    tr_sym_basis_info bi;
    if (tr_sym_basis_get_info(b, &bi)) { *rc = TR_MATH_INVALID; return NULL; }
    tr_dyn_operator *o = (tr_dyn_operator *)zalloc(sizeof *o);
    if (!o) { *rc = TR_MATH_MEMORY; return NULL; }
    o->nb = bi.nb; o->n = bi.n; o->pdegree = pdeg; o->geo = *g; o->den = 1;
    o->basis = tr_sym_basis_clone(b, rc);
    o->num = (tr_math_i128 *)zalloc(tr_budget_mul(tr_budget_mul((uint64_t)bi.nb * (uint64_t)bi.nb, (uint64_t)pdeg + 1), 16));
    if (!o->basis || !o->num) { op_release(o); *rc = TR_MATH_MEMORY; return NULL; }
    o->bytes = op_bytes(bi.nb, pdeg, bi.bytes);
    return o;
}
static tr_dyn_operator *op_done(tr_dyn_operator *o, int32_t *error) {
    op_stats(o);
    o->magic = OP_MAGIC;
    set_error(error, TR_MATH_OK);
    return o;
}

/* the box contract in logical units over the reach set: blo, bhi; TR_MATH_DOMAIN_ERROR when a coordinate it must bound
 * is not bounded, TR_MATH_INVALID for lo > hi */
static int32_t box_logical(const tr_dyn_box *box, const tr_sym_geometry *g, uint32_t reach, q128 *lo, q128 *hi) {
    if (box->n != g->n || box->reserved || box->reserved2) return TR_MATH_INVALID;
    for (int32_t i = 0; i < g->n; i++) {
        if (!((reach >> i) & 1)) continue;
        if (!((box->declared >> i) & 1)) return TR_MATH_DOMAIN_ERROR;
        q128 a, b;
        if (!q_from(box->lo[i], &a) || !q_from(box->hi[i], &b)) return TR_MATH_INVALID;
        if (!q_scale10(a, g->scale[i], &lo[i]) || !q_scale10(b, g->scale[i], &hi[i])) return TR_MATH_OVERFLOW;
        int c;
        if (!q_cmp(lo[i], hi[i], &c)) return TR_MATH_OVERFLOW;
        if (c > 0) return TR_MATH_INVALID;
    }
    return TR_MATH_OK;
}
static tr_bigq *bq_of_q(q128 v, const tr_math_budget *b, int32_t *err) {
    tr_bigint *n = tr_bigint_from_i128(tr_i128_set(v.n), b, NULL, err), *d = n ? tr_bigint_from_i128(tr_i128_set(v.d), b, NULL, err) : NULL;
    tr_bigq *r = d ? tr_bigq_from_bigints(n, d, b, NULL, err) : NULL;
    tr_bigint_free(n); tr_bigint_free(d);
    return r;
}
static tr_bigq *bq_of_i128(i128 v, const tr_math_budget *b, int32_t *err) { q128 q; q.n = v; q.d = 1; return bq_of_q(q, b, err); }
/* r = r OP x, freeing the old r (OP 0 add, 1 mul) */
static int bq_acc(tr_bigq **r, const tr_bigq *x, int op, const tr_math_budget *b, int32_t *err) {
    tr_bigq *t = op ? tr_bigq_mul(*r, x, b, NULL, err) : tr_bigq_add(*r, x, b, NULL, err);
    if (!t) return 0;
    tr_bigq_free(*r);
    *r = t;
    return 1;
}
/* the exact range of x^e over [lo, hi] (rationals): [mn, mx] */
static int bq_pow_range(const tr_bigq *lo, const tr_bigq *hi, int64_t e, const tr_math_budget *b, tr_bigq **mn, tr_bigq **mx,
                        int32_t *err) {
    tr_bigq *a = tr_bigq_pow(lo, e, b, NULL, err), *c = a ? tr_bigq_pow(hi, e, b, NULL, err) : NULL;
    if (!a || !c) { tr_bigq_free(a); tr_bigq_free(c); return 0; }
    int ac = tr_bigq_cmp(a, c);
    tr_bigq *small = ac <= 0 ? a : c, *large = ac <= 0 ? c : a;
    if ((e % 2) == 0 && tr_bigq_sign(lo) <= 0 && tr_bigq_sign(hi) >= 0) {      /* an even power over an interval with 0 */
        tr_bigq_free(small);
        tr_bigint *z = tr_bigint_from_i64(0, b, NULL, err);
        small = z ? tr_bigq_from_bigint(z, b, NULL, err) : NULL;
        tr_bigint_free(z);
        if (!small) { tr_bigq_free(large); return 0; }
    }
    *mn = small; *mx = large;
    return 1;
}
/* interval product [a, b] [c, d] -> [lo, hi] (frees nothing it is given) */
static int bq_iv_mul(const tr_bigq *a, const tr_bigq *b, const tr_bigq *c, const tr_bigq *d, const tr_math_budget *bud,
                     tr_bigq **lo, tr_bigq **hi, int32_t *err) {
    tr_bigq *p[4] = {tr_bigq_mul(a, c, bud, NULL, err), tr_bigq_mul(a, d, bud, NULL, err), tr_bigq_mul(b, c, bud, NULL, err),
                     tr_bigq_mul(b, d, bud, NULL, err)};
    if (!p[0] || !p[1] || !p[2] || !p[3]) { for (int i = 0; i < 4; i++) tr_bigq_free(p[i]); return 0; }
    int mn = 0, mx = 0;
    for (int i = 1; i < 4; i++) { if (tr_bigq_cmp(p[i], p[mn]) < 0) mn = i; if (tr_bigq_cmp(p[i], p[mx]) > 0) mx = i; }
    *lo = tr_bigq_copy(p[mn], bud, NULL, err); *hi = tr_bigq_copy(p[mx], bud, NULL, err);
    for (int i = 0; i < 4; i++) tr_bigq_free(p[i]);
    if (!*lo || !*hi) { tr_bigq_free(*lo); tr_bigq_free(*hi); *lo = *hi = NULL; return 0; }
    return 1;
}
/* 1 when F(box) lies in the box (exact rational interval arithmetic over the reach set), 0 when not proven, < 0 an error */
static int32_t box_invariant(const tr_dyn_map *m, uint32_t reach, const q128 *lo, const q128 *hi, const tr_math_budget *bud) {
    int32_t err = 0, inv = 1;
    tr_bigq *L[NC], *H[NC];
    memset(L, 0, sizeof L); memset(H, 0, sizeof H);
    for (int32_t i = 0; i < m->n && !err; i++) {
        if (!((reach >> i) & 1)) continue;
        L[i] = bq_of_q(lo[i], bud, &err);
        H[i] = L[i] ? bq_of_q(hi[i], bud, &err) : NULL;
        if (!H[i] && !err) err = TR_MATH_MEMORY;
    }
    for (int32_t i = 0; i < m->n && !err && inv; i++) {
        if (!((reach >> i) & 1)) continue;
        tr_bigq *slo = bq_of_i128(0, bud, &err), *shi = slo ? bq_of_i128(0, bud, &err) : NULL;
        for (int32_t t = m->first[i]; t < m->first[i + 1] && !err && shi; t++) {
            const mterm *x = &m->terms[t];
            tr_bigq *rl = bq_of_i128(1, bud, &err), *rh = rl ? bq_of_i128(1, bud, &err) : NULL;
            for (int32_t j = 0; j < m->n && rh && !err; j++) {
                if (!x->e[j]) continue;
                tr_bigq *pl = NULL, *ph = NULL, *nl = NULL, *nh = NULL;
                if (!bq_pow_range(L[j], H[j], x->e[j], bud, &pl, &ph, &err)) { if (!err) err = TR_MATH_MEMORY; break; }
                if (!bq_iv_mul(rl, rh, pl, ph, bud, &nl, &nh, &err)) { if (!err) err = TR_MATH_MEMORY; }
                tr_bigq_free(pl); tr_bigq_free(ph);
                if (err) break;
                tr_bigq_free(rl); tr_bigq_free(rh);
                rl = nl; rh = nh;
            }
            if (!err && rh) {                                    /* c / q times [rl, rh] */
                q128 cq;
                if (!q_make(x->c, m->q[i], &cq)) err = TR_MATH_OVERFLOW;
                tr_bigq *c = err ? NULL : bq_of_q(cq, bud, &err);
                tr_bigq *a = c ? tr_bigq_mul(c, rl, bud, NULL, &err) : NULL, *b = a ? tr_bigq_mul(c, rh, bud, NULL, &err) : NULL;
                if (b) {
                    int pos = tr_bigq_sign(c) >= 0;
                    if (!bq_acc(&slo, pos ? a : b, 0, bud, &err) || !bq_acc(&shi, pos ? b : a, 0, bud, &err)) if (!err) err = TR_MATH_MEMORY;
                } else if (!err) err = TR_MATH_MEMORY;
                tr_bigq_free(c); tr_bigq_free(a); tr_bigq_free(b);
            }
            tr_bigq_free(rl); tr_bigq_free(rh);
        }
        if (!err && shi && (tr_bigq_cmp(slo, L[i]) < 0 || tr_bigq_cmp(shi, H[i]) > 0)) inv = 0;
        if (!shi && !err) err = TR_MATH_MEMORY;
        tr_bigq_free(slo); tr_bigq_free(shi);
    }
    for (int32_t i = 0; i < NC; i++) { tr_bigq_free(L[i]); tr_bigq_free(H[i]); }
    return err ? err : inv;
}

int32_t tr_dyn_operator_need(const tr_sym_basis *b, const tr_dyn_map *m, const tr_dyn_box *box, tr_math_need *need) {
    tr_sym_basis_info bi;
    if (!need || !map_ok(m) || tr_sym_basis_get_info(b, &bi) || bi.n != m->n) return TR_MATH_INVALID;
    need_zero(need);
    const int32_t n = bi.n, nb = bi.nb;
    uint8_t *ex = (uint8_t *)zalloc((uint64_t)nb * (uint64_t)n);
    eterm *store = NULL;
    factor F[NC];
    if (!ex) return TR_MATH_MEMORY;
    int32_t rc = tr_sym_basis_get_exponents(b, ex, (uint64_t)nb * (uint64_t)n);
    if (!rc) rc = factors_make(m, F, &store);
    if (rc) { free(ex); return rc; }
    uint64_t cap = 1;
    for (int32_t r = 0; r < nb; r++) {
        uint64_t c = exp_cap(F, ex + (uint64_t)r * n, n);
        if (c > cap) cap = c;
    }
    free(ex); free(store);
    const int32_t pdeg = bi.degree * m->pdegree;
    need->need[TR_BUDGET_VARIABLES] = (uint64_t)n;
    need->need[TR_BUDGET_DEGREE] = (uint64_t)bi.degree * (uint64_t)(m->degree ? m->degree : 1);
    need->need[TR_BUDGET_SUPPORT] = (uint64_t)nb;
    need->need[TR_BUDGET_TENSOR] = tr_budget_mul(tr_budget_mul((uint64_t)nb, (uint64_t)nb), (uint64_t)pdeg + 1);
    need->need[TR_BUDGET_STATES] = cap;
    uint64_t bytes = op_bytes(nb, pdeg, bi.bytes);
    bytes = tr_budget_add(bytes, tr_budget_mul(cap, 2 * sizeof(eterm)));
    bytes = tr_budget_add(bytes, tr_budget_add(tr_budget_mul((uint64_t)nb, 32), (uint64_t)m->nterms * sizeof(eterm)));
    if (box) bytes = tr_budget_add(bytes, tr_budget_mul((uint64_t)nb, 4096));   /* the bounds (FND-009 rationals) */
    need->need[TR_BUDGET_BYTES] = bytes;
    return TR_MATH_OK;
}

/* SYM-001's action matrix of an exact affine map */
static tr_dyn_operator *op_from_sym(const tr_sym_basis *b, const tr_dyn_map *m, const tr_math_budget *budget,
                                    tr_math_refusal *refusal, tr_sym_closure *closure, int32_t *error) {
    tr_math_need need;
    int32_t rc = tr_sym_action_need(b, m->sym, &need);
    if (rc) return op_fail(NULL, error, rc);
    tr_sym_basis_info bi;
    tr_sym_basis_get_info(b, &bi);
    need.need[TR_BUDGET_BYTES] = tr_budget_add(need.need[TR_BUDGET_BYTES], op_bytes(bi.nb, 0, bi.bytes));
    if ((rc = check_need(budget, &need, refusal))) return op_fail(NULL, error, rc);
    tr_sym_action *a = tr_sym_action_make(b, m->sym, budget, refusal, closure, &rc);
    if (!a) return op_fail(NULL, error, rc ? rc : TR_MATH_MEMORY);
    tr_sym_action_view v;
    tr_sym_action_get_view(a, &v);
    tr_dyn_operator *o = op_alloc(b, &m->geo, 0, &rc);
    if (!o) { tr_sym_action_free(a); return op_fail(NULL, error, rc); }
    memcpy(o->num, v.num, (size_t)o->nb * (size_t)o->nb * sizeof(tr_math_i128));
    o->den = tr_i128_get(v.den);
    tr_sym_action_free(a);
    o->kind = TR_DYN_OP_EXACT; o->label = TR_DYN_LABEL_EXACT; o->source = TR_DYN_FROM_MAP; o->sym_action = 1;
    o->closed = 1;
    return op_done(o, error);
}
tr_dyn_operator *tr_dyn_operator_make(const tr_sym_basis *b, const tr_dyn_map *m, const tr_dyn_box *box,
                                      const tr_math_budget *budget, tr_math_refusal *refusal, tr_sym_closure *closure,
                                      int32_t *error) {
    tr_budget_none(refusal);
    if (closure) { memset(closure, 0, sizeof *closure); closure->term = -1; }
    tr_sym_basis_info bi;
    if (!map_ok(m) || tr_sym_basis_get_info(b, &bi) || bi.n != m->n) return op_fail(NULL, error, TR_MATH_INVALID);
    int32_t rc = tr_sym_basis_check(b, &m->geo);
    if (rc) return op_fail(NULL, error, rc);
    if (box && m->parametric) return op_fail(NULL, error, TR_MATH_UNSUPPORTED);
    if (!box && m->kind == TR_DYN_MAP_AFFINE && m->sym) return op_from_sym(b, m, budget, refusal, closure, error);
    tr_math_need need;
    if ((rc = tr_dyn_operator_need(b, m, box, &need))) return op_fail(NULL, error, rc);
    if ((rc = check_need(budget, &need, refusal))) return op_fail(NULL, error, rc);
    const int32_t n = bi.n, nb = bi.nb, pdeg = bi.degree * m->pdegree;
    const uint64_t e2 = (uint64_t)nb * (uint64_t)nb;
    const tr_math_budget *bud = budget;
    q128 blo[NC], bhi[NC];
    memset(blo, 0, sizeof blo); memset(bhi, 0, sizeof bhi);
    uint32_t reach = reach_set(m, bi.coords);
    if (box && (rc = box_logical(box, &m->geo, reach, blo, bhi))) return op_fail(NULL, error, rc);
    tr_dyn_operator *o = op_alloc(b, &m->geo, pdeg, &rc);
    if (!o) return op_fail(NULL, error, rc);
    uint8_t *ex = (uint8_t *)zalloc((uint64_t)nb * (uint64_t)n);
    i128 *rden = (i128 *)zalloc((uint64_t)nb * sizeof(i128));
    eterm *store = NULL;
    factor F[NC];
    if (!ex || !rden) { free(ex); free(rden); return op_fail(o, error, TR_MATH_MEMORY); }
    rc = tr_sym_basis_get_exponents(b, ex, (uint64_t)nb * (uint64_t)n);
    if (!rc) rc = factors_make(m, F, &store);
    uint64_t cap = 1;
    for (int32_t r = 0; r < nb && !rc; r++) { uint64_t c = exp_cap(F, ex + (uint64_t)r * n, n); if (c > cap) cap = c; }
    eterm *P = rc ? NULL : (eterm *)zalloc(tr_budget_mul(cap, sizeof(eterm)));
    eterm *tmp = P ? (eterm *)zalloc(tr_budget_mul(cap, sizeof(eterm))) : NULL;
    if (!rc && (!P || !tmp)) rc = TR_MATH_MEMORY;
    tr_bigq *Mpow[NC][TR_SYM_MAX_DEGREE * TR_DYN_MAX_DEGREE + 1];
    memset(Mpow, 0, sizeof Mpow);
    if (box && !rc) {
        o->eps = (tr_bigq **)zalloc((uint64_t)nb * sizeof(tr_bigq *));
        if (!o->eps) rc = TR_MATH_MEMORY;
    }
    int32_t bad_row = -1;
    uint8_t bad_mono[NC];
    for (int32_t r = 0; r < nb && !rc; r++) {
        const uint8_t *e = ex + (uint64_t)r * n;
        uint64_t np = 0;
        if ((rc = exp_row(F, e, n, P, tmp, &np))) break;
        tr_bigq *eps = NULL;
        for (uint64_t t = 0; t < np && !rc; t++) {
            uint8_t ee[NC];
            ekey_exps(&P[t].k, ee);
            int32_t col = tr_sym_basis_find(b, ee);
            if (col >= 0) { o->num[(uint64_t)P[t].k.p * e2 + (uint64_t)r * nb + (uint64_t)col] = tr_i128_set(P[t].c); continue; }
            if (!box) { bad_row = r; memcpy(bad_mono, ee, NC); rc = TR_MATH_DOMAIN_ERROR; break; }
            /* a dropped term: |c| prod_j M_j^e_j with M_j = max(|lo_j|, |hi_j|) (logical units) */
            o->dropped++;
            int32_t err = 0;
            tr_bigq *term = bq_of_i128(P[t].c < 0 ? -P[t].c : P[t].c, bud, &err);
            for (int32_t j = 0; j < n && term && !err; j++) {
                if (!ee[j]) continue;
                if (!Mpow[j][ee[j]]) {
                    q128 a = blo[j], c = bhi[j];
                    a.n = (i128)mag128(a.n); c.n = (i128)mag128(c.n);
                    int cmpv;
                    if (!q_cmp(a, c, &cmpv)) { err = TR_MATH_OVERFLOW; break; }
                    tr_bigq *M = bq_of_q(cmpv >= 0 ? a : c, bud, &err);
                    Mpow[j][ee[j]] = M ? tr_bigq_pow(M, ee[j], bud, NULL, &err) : NULL;
                    tr_bigq_free(M);
                    if (!Mpow[j][ee[j]]) { if (!err) err = TR_MATH_MEMORY; break; }
                }
                if (!bq_acc(&term, Mpow[j][ee[j]], 1, bud, &err) && !err) err = TR_MATH_MEMORY;
            }
            if (!term && !err) err = TR_MATH_MEMORY;
            if (!err) {
                if (!eps) { eps = term; term = NULL; }
                else if (!bq_acc(&eps, term, 0, bud, &err) && !err) err = TR_MATH_MEMORY;
            }
            tr_bigq_free(term);
            if (err) rc = err;
        }
        if (rc) { tr_bigq_free(eps); break; }
        i128 d = 1;                                              /* the row's denominator prod q_i^a_i */
        for (int32_t i = 0; i < n && !rc; i++)
            for (int32_t k = 0; k < e[i]; k++) if (!mul128(d, m->q[i], &d)) { rc = TR_MATH_OVERFLOW; break; }
        if (rc) { tr_bigq_free(eps); break; }
        if (box) {                                               /* eps_a = (sum of the dropped |c| M^b) / rowden */
            int32_t err = 0;
            if (!eps) eps = bq_of_i128(0, bud, &err);
            tr_bigq *dq = eps ? bq_of_i128(d, bud, &err) : NULL;
            tr_bigq *v = dq ? tr_bigq_div(eps, dq, bud, NULL, &err) : NULL;
            tr_bigq_free(eps); tr_bigq_free(dq);
            if (!v) { rc = err ? err : TR_MATH_MEMORY; break; }
            o->eps[r] = v;
        }
        u128 g = (u128)d;                                        /* reduced by the row's gcd (every N_j) */
        for (int32_t j = 0; j <= pdeg; j++)
            for (int32_t c = 0; c < nb; c++) g = gcd128(g, mag128(tr_i128_get(o->num[(uint64_t)j * e2 + (uint64_t)r * nb + c])));
        for (int32_t j = 0; j <= pdeg && g > 1; j++)
            for (int32_t c = 0; c < nb; c++) {
                uint64_t at = (uint64_t)j * e2 + (uint64_t)r * nb + c;
                o->num[at] = tr_i128_set(tr_i128_get(o->num[at]) / (i128)g);
            }
        rden[r] = d / (i128)g;
    }
    for (int32_t j = 0; j < NC; j++) for (int32_t k = 0; k <= TR_SYM_MAX_DEGREE * TR_DYN_MAX_DEGREE; k++) tr_bigq_free(Mpow[j][k]);
    free(P); free(tmp); free(store); free(ex);
    if (rc == TR_MATH_DOMAIN_ERROR && bad_row >= 0 && closure) { closure->term = bad_row; memcpy(closure->missing, bad_mono, NC); }
    if (!rc) {                                                   /* one common denominator D = lcm of the rows' */
        i128 D = 1;
        for (int32_t r = 0; r < nb && !rc; r++) if (!lcm128(D, rden[r], &D)) rc = TR_MATH_OVERFLOW;
        for (int32_t r = 0; r < nb && !rc; r++) {
            i128 f = D / rden[r];
            if (f == 1) continue;
            for (int32_t j = 0; j <= pdeg && !rc; j++)
                for (int32_t c = 0; c < nb; c++) {
                    uint64_t at = (uint64_t)j * e2 + (uint64_t)r * nb + c;
                    i128 v;
                    if (!mul128(tr_i128_get(o->num[at]), f, &v) || !in_range(v)) { rc = TR_MATH_OVERFLOW; break; }
                    o->num[at] = tr_i128_set(v);
                }
        }
        o->den = D;
        if (!rc) op_reduce(o);
    }
    free(rden);
    if (rc) return op_fail(o, error, rc);
    o->source = TR_DYN_FROM_MAP;
    if (box && o->dropped) {
        o->kind = TR_DYN_OP_TRUNCATED; o->label = TR_DYN_LABEL_CERTIFIED; o->closed = 0;
        o->box = *box; o->bounded = reach;
        for (int32_t i = 0; i < NC; i++) { o->blo[i] = blo[i]; o->bhi[i] = bhi[i]; }
        int32_t inv = box_invariant(m, reach, blo, bhi, bud);
        if (inv < 0) return op_fail(o, error, inv);
        o->invariant = inv;
        o->bytes = tr_budget_add(o->bytes, tr_budget_mul((uint64_t)nb, 64));
    } else {                                                     /* nothing dropped: the basis is closed, exact */
        if (o->eps) { for (int32_t a = 0; a < nb; a++) tr_bigq_free(o->eps[a]); free(o->eps); o->eps = NULL; }
        o->kind = TR_DYN_OP_EXACT; o->label = TR_DYN_LABEL_EXACT; o->closed = 1;
    }
    if (closure) closure->closed = o->closed;
    return op_done(o, error);
}

static int same_frame(const tr_dyn_operator *a, const tr_dyn_operator *b) {
    return tr_sym_basis_equal(a->basis, b->basis) == 1 && geo_equal(&a->geo, &b->geo);
}
tr_dyn_operator *tr_dyn_operator_mixture(const tr_dyn_operator *const *ops, const tr_num_rational *w, int32_t n,
                                         const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    if (!ops || !w || n < 1) return op_fail(NULL, error, TR_MATH_INVALID);
    if (n > TR_DYN_MAX_OPERATORS) return op_fail(NULL, error, TR_MATH_UNSUPPORTED);
    for (int32_t j = 0; j < n; j++) {
        if (!op_ok(ops[j]) || w[j].denominator == 0) return op_fail(NULL, error, TR_MATH_INVALID);
        if (ops[j]->kind != TR_DYN_OP_EXACT || !same_frame(ops[0], ops[j])) return op_fail(NULL, error, TR_MATH_DOMAIN_ERROR);
        if (ops[j]->pdegree) return op_fail(NULL, error, TR_MATH_UNSUPPORTED);
    }
    q128 sum = {0, 1}, wq[TR_DYN_MAX_OPERATORS];
    for (int32_t j = 0; j < n; j++) {
        if (!q_from(w[j], &wq[j])) return op_fail(NULL, error, TR_MATH_INVALID);
        if (wq[j].n < 0) return op_fail(NULL, error, TR_MATH_DOMAIN_ERROR);      /* convex: weights >= 0 */
        if (!q_add(sum, wq[j], &sum)) return op_fail(NULL, error, TR_MATH_OVERFLOW);
    }
    if (sum.n != 1 || sum.d != 1) return op_fail(NULL, error, TR_MATH_DOMAIN_ERROR); /* convex: sum = 1 */
    const uint64_t nb = (uint64_t)ops[0]->nb, e = nb * nb;
    tr_math_need need;
    need_zero(&need);
    need.need[TR_BUDGET_SUPPORT] = nb;
    need.need[TR_BUDGET_TENSOR] = e;
    need.need[TR_BUDGET_BYTES] = op_bytes((int32_t)nb, 0, basis_bytes_of(ops[0]->basis));
    int32_t rc = check_need(budget, &need, refusal);
    if (rc) return op_fail(NULL, error, rc);
    i128 L = 1, f[TR_DYN_MAX_OPERATORS];
    for (int32_t j = 0; j < n && !rc; j++) {                     /* L = lcm_j (wd_j D_j) */
        i128 d;
        if (!mul128(wq[j].d, ops[j]->den, &d) || !lcm128(L, d, &L)) rc = TR_MATH_OVERFLOW;
    }
    for (int32_t j = 0; j < n && !rc; j++) {
        i128 d;
        mul128(wq[j].d, ops[j]->den, &d);
        if (!mul128(wq[j].n, L / d, &f[j])) rc = TR_MATH_OVERFLOW;
    }
    if (rc) return op_fail(NULL, error, rc);
    tr_dyn_operator *o = op_alloc(ops[0]->basis, &ops[0]->geo, 0, &rc);
    if (!o) return op_fail(NULL, error, rc);
    for (uint64_t x = 0; x < e && !rc; x++) {
        i128 s = 0;
        for (int32_t j = 0; j < n; j++) {
            i128 p;
            if (!mul128(tr_i128_get(ops[j]->num[x]), f[j], &p) || !add128(s, p, &s) || !in_range(s)) { rc = TR_MATH_OVERFLOW; break; }
        }
        o->num[x] = tr_i128_set(s);
    }
    if (rc) return op_fail(o, error, rc);
    o->den = L;
    op_reduce(o);
    o->kind = TR_DYN_OP_EXACT; o->label = TR_DYN_LABEL_EXACT; o->source = TR_DYN_FROM_MIXTURE; o->closed = 1;
    return op_done(o, error);
}
tr_dyn_operator *tr_dyn_operator_substitute(const tr_dyn_operator *s, tr_num_rational theta, const tr_math_budget *budget,
                                            tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    q128 t;
    if (!op_ok(s) || !q_from(theta, &t)) return op_fail(NULL, error, TR_MATH_INVALID);
    if (s->kind != TR_DYN_OP_EXACT) return op_fail(NULL, error, TR_MATH_DOMAIN_ERROR);
    const uint64_t nb = (uint64_t)s->nb, e = nb * nb;
    tr_math_need need;
    need_zero(&need);
    need.need[TR_BUDGET_SUPPORT] = nb;
    need.need[TR_BUDGET_TENSOR] = e;
    need.need[TR_BUDGET_BYTES] = op_bytes((int32_t)nb, 0, basis_bytes_of(s->basis));
    int32_t rc = check_need(budget, &need, refusal);
    if (rc) return op_fail(NULL, error, rc);
    const int32_t P = s->pdegree;
    i128 ap[TR_DYN_MAX_PDEGREE * TR_SYM_MAX_DEGREE + 1], bp[TR_DYN_MAX_PDEGREE * TR_SYM_MAX_DEGREE + 1];
    ap[0] = 1; bp[0] = 1;
    for (int32_t j = 1; j <= P && !rc; j++)
        if (!mul128(ap[j - 1], t.n, &ap[j]) || !mul128(bp[j - 1], t.d, &bp[j])) rc = TR_MATH_OVERFLOW;
    if (rc) return op_fail(NULL, error, rc);
    tr_dyn_operator *o = op_alloc(s->basis, &s->geo, 0, &rc);
    if (!o) return op_fail(NULL, error, rc);
    for (uint64_t x = 0; x < e && !rc; x++) {                    /* sum_j a^j b^(P-j) N_j / (b^P D) */
        i128 acc = 0;
        for (int32_t j = 0; j <= P; j++) {
            i128 c = tr_i128_get(s->num[(uint64_t)j * e + x]), f, v;
            if (!c) continue;
            if (!mul128(ap[j], bp[P - j], &f) || !mul128(c, f, &v) || !add128(acc, v, &acc) || !in_range(acc)) { rc = TR_MATH_OVERFLOW; break; }
        }
        o->num[x] = tr_i128_set(acc);
    }
    if (!rc && !mul128(bp[P], s->den, &o->den)) rc = TR_MATH_OVERFLOW;
    if (rc) return op_fail(o, error, rc);
    op_reduce(o);
    o->kind = TR_DYN_OP_EXACT; o->label = TR_DYN_LABEL_EXACT; o->closed = s->closed;
    o->source = P ? TR_DYN_FROM_SUBSTITUTION : s->source; o->sym_action = P ? 0 : s->sym_action;
    return op_done(o, error);
}
int32_t tr_dyn_operator_get_info(const tr_dyn_operator *o, tr_dyn_op_info *out) {
    if (!op_ok(o) || !out) return TR_MATH_INVALID;
    memset(out, 0, sizeof *out);
    out->nb = o->nb; out->n = o->n; out->kind = o->kind; out->label = o->label; out->source = o->source;
    out->closed = o->closed; out->pdegree = o->pdegree; out->invariant = o->invariant; out->sym_action = o->sym_action;
    out->den = tr_i128_set(o->den); out->nonzeros = o->nonzeros; out->dropped = o->dropped; out->norm_bits = o->norm_bits;
    out->digest[0] = o->geo.digest[0]; out->digest[1] = o->geo.digest[1]; out->bytes = o->bytes;
    return TR_MATH_OK;
}
int32_t tr_dyn_operator_get_view(const tr_dyn_operator *o, int32_t j, tr_sym_action_view *v) {
    if (!op_ok(o) || !v || j < 0 || j > o->pdegree) return TR_MATH_INVALID;
    memset(v, 0, sizeof *v);
    v->carrier = TR_SYM_EXACT; v->nb = o->nb; v->den = tr_i128_set(o->den);
    v->num = o->num + (uint64_t)j * (uint64_t)o->nb * (uint64_t)o->nb;
    return TR_MATH_OK;
}
const tr_sym_basis *tr_dyn_operator_get_basis(const tr_dyn_operator *o) { return op_ok(o) ? o->basis : NULL; }
int32_t tr_dyn_operator_check_geometry(const tr_dyn_operator *o, const tr_sym_geometry *g) {
    if (!op_ok(o) || tr_sym_geometry_check(g)) return TR_MATH_INVALID;
    return geo_equal(&o->geo, g) ? TR_MATH_OK : TR_MATH_DOMAIN_ERROR;
}
int32_t tr_dyn_operator_get_box(const tr_dyn_operator *o, tr_dyn_box *out) {
    if (!op_ok(o) || !out || o->kind != TR_DYN_OP_TRUNCATED) return TR_MATH_INVALID;
    *out = o->box;
    return TR_MATH_OK;
}
static void decision_base(tr_cert_decision *d, int exact) {
    memset(d, 0, sizeof *d);
    d->kind = exact ? TR_CERT_LAW : TR_DYN_LABEL_CERTIFIED; d->claim = TR_CLAIM_POLYNOMIAL;
    d->premise = -1; d->stale.column = d->stale.dim = -1;
    d->authorized = exact; d->reason = exact ? TR_CERT_OK : TR_CERT_REFUSE_APPROXIMATE;
}
int32_t tr_dyn_authorize_exact(const tr_dyn_operator *o, tr_cert_decision *d) {
    if (!op_ok(o) || !d) return TR_MATH_INVALID;
    decision_base(d, o->kind == TR_DYN_OP_EXACT);
    return o->kind == TR_DYN_OP_EXACT ? TR_MATH_OK : TR_MATH_DOMAIN_ERROR;
}
void tr_dyn_operator_free(tr_dyn_operator *o) {
    if (!op_ok(o)) return;
    o->magic = 0;
    op_release(o);
}

/* ================================================================== modular arithmetic: primes, residues, CRT */
int32_t tr_dyn_primes(uint64_t count, tr_gfp *out) {
    if (!out || !count) return TR_MATH_INVALID;
    if (count > TR_DYN_MAX_PRIMES) return TR_MATH_UNSUPPORTED;
    return tr_lin_primes(count, out);
}
/* reconstruction primes for numerators of at most bits bits (symmetric range: sum log2 p >= 61 k > bits + 1), plus
 * the check primes */
static uint64_t primes_for(uint64_t bits) {
    if (bits == TR_BUDGET_UNKNOWN) return TR_BUDGET_UNKNOWN;
    return (bits + 2 + TR_DYN_PRIME_BITS - 1) / TR_DYN_PRIME_BITS + TR_DYN_CHECK_PRIMES;
}
/* caller primes: consistent, distinct (strictly descending) in [2^61, 2^62), and proven prime (prove 0: primes this
 * module just took from tr_dyn_primes, already proven there) */
static int primes_valid(const tr_gfp *p, uint64_t np, int prove) {
    if (!p || !np || np > TR_DYN_MAX_PRIMES) return 0;
    for (uint64_t i = 0; i < np; i++) {
        if (!tr_gfp_consistent(&p[i]) || p[i].p < ((uint64_t)1 << 61) || (prove && !tr_gfp_is_prime(p[i].p))) return 0;
        if (i && p[i].p >= p[i - 1].p) return 0;                 /* strictly descending: distinct */
    }
    return 1;
}
/* x mod p without a division in the usual case x < 2p (every prime here lies in [2^61, 2^62), so a residue or a prime
 * of another modulus is below 2p) */
static inline uint64_t red_small(uint64_t x, const tr_gfp *f) {
    if (x < f->p) return x;
    x -= f->p;
    return x < f->p ? x : x % f->p;
}
/* Garner's mixed-radix CRT with constants shared by every value (the first k primes) */
typedef struct { uint64_t k, mn; const tr_gfp *f; uint64_t *inv, *M, *v; } garner;
static int garner_init(garner *G, const tr_gfp *f, uint64_t k) {
    memset(G, 0, sizeof *G);
    G->k = k; G->f = f;
    G->inv = (uint64_t *)zalloc(k * 8); G->M = (uint64_t *)zalloc((k + 1) * 8); G->v = (uint64_t *)zalloc(k * 8);
    if (!G->inv || !G->M || !G->v) { free(G->inv); free(G->M); free(G->v); return 0; }
    for (uint64_t i = 0; i < k; i++) {
        uint64_t prod = 1 % f[i].p;
        for (uint64_t j = 0; j < i; j++) prod = tr_gfp_mul_hd(&f[i], prod, red_small(f[j].p, &f[i]));
        G->inv[i] = i ? tr_gfp_inv_hd(&f[i], prod) : 1;
    }
    G->M[0] = 1; G->mn = 1;
    for (uint64_t i = 0; i < k; i++) {
        u128 c = 0;
        for (uint64_t j = 0; j < G->mn; j++) { c += (u128)G->M[j] * f[i].p; G->M[j] = (uint64_t)c; c >>= 64; }
        if (c) G->M[G->mn++] = (uint64_t)c;
    }
    return 1;
}
static void garner_free(garner *G) { free(G->inv); free(G->M); free(G->v); }
/* the symmetric value (-M/2, M/2] of residues res[i * stride], i < k, into limbs (capacity k + 1): magnitude, sign */
static uint64_t garner_value(const garner *G, const uint64_t *res, uint64_t stride, uint64_t *limbs, int *neg) {
    const uint64_t k = G->k;
    uint64_t *v = G->v;
    for (uint64_t i = 0; i < k; i++) {
        const tr_gfp *f = &G->f[i];
        uint64_t acc = 0;
        for (uint64_t j = i; j-- > 0;) acc = tr_gfp_add_hd(f, tr_gfp_mul_hd(f, acc, red_small(G->f[j].p, f)), red_small(v[j], f));
        v[i] = i ? tr_gfp_mul_hd(f, tr_gfp_sub_hd(f, res[i * stride], acc), G->inv[i]) : res[0];
    }
    uint64_t n = 1;
    limbs[0] = v[k - 1];
    for (uint64_t i = k - 1; i-- > 0;) {
        u128 c = v[i];
        for (uint64_t j = 0; j < n; j++) { c += (u128)limbs[j] * G->f[i].p; limbs[j] = (uint64_t)c; c >>= 64; }
        if (c) limbs[n++] = (uint64_t)c;
    }
    for (uint64_t j = n; j < G->mn; j++) limbs[j] = 0;
    while (n > 1 && !limbs[n - 1]) n--;
    int gt = 0;                                                  /* 2 x > M: x - M */
    uint64_t top = n > G->mn ? n : G->mn;
    for (uint64_t j = top + 1; j-- > 0;) {
        uint64_t a = (j < n ? limbs[j] << 1 : 0) | (j && j - 1 < n ? limbs[j - 1] >> 63 : 0);
        uint64_t b = j < G->mn ? G->M[j] : 0;
        if (a != b) { gt = a > b; break; }
    }
    *neg = 0;
    if (gt) {
        uint64_t br = 0;
        for (uint64_t j = 0; j < G->mn; j++) {
            uint64_t x = G->M[j], y = j < n ? limbs[j] : 0;
            uint64_t r = x - y - br;
            br = (x < y) || (x - y < br);
            limbs[j] = r;
        }
        n = G->mn;
        while (n > 1 && !limbs[n - 1]) n--;
        *neg = 1;
    }
    if (n == 1 && !limbs[0]) { n = 0; *neg = 0; }
    return n;
}
/* a magnitude modulo p, with its sign */
static uint64_t limbs_mod(const uint64_t *l, uint64_t n, int neg, const tr_gfp *f) {
    u128 r = 0;
    for (uint64_t i = n; i-- > 0;) r = ((r << 64) | l[i]) % f->p;
    uint64_t x = (uint64_t)r;
    return neg ? tr_gfp_neg_hd(f, x) : x;
}
/* reconstruct count values (residue of value i modulo prime q at res[q * qstride + i]) into dst[i]; the first
 * np - TR_DYN_CHECK_PRIMES primes reconstruct, the others check; |value| < 2^bits. 0, TR_MATH_INVALID (a check or the
 * bound failed: the residues are inconsistent), TR_MATH_MEMORY, TR_MATH_OVERFLOW (dst too small) */
static int32_t crt_values(const tr_gfp *p, uint64_t np, const uint64_t *res, uint64_t qstride, uint64_t count,
                          uint64_t bits, bvec *dst, uint64_t at) {
    if (np <= TR_DYN_CHECK_PRIMES) return TR_MATH_INVALID;
    const uint64_t k = np - TR_DYN_CHECK_PRIMES;
    if (61 * k < bits + 2) return TR_MATH_INVALID;              /* too few primes for the bound */
    garner G;
    uint64_t *limbs = (uint64_t *)zalloc((k + 2) * 8);
    if (!limbs || !garner_init(&G, p, k)) { free(limbs); return TR_MATH_MEMORY; }
    int32_t rc = TR_MATH_OK;
    for (uint64_t i = 0; i < count && !rc; i++) {
        for (uint64_t q = 0; q < np && !rc; q++) if (res[q * qstride + i] >= p[q].p) rc = TR_MATH_INVALID;   /* not a residue */
        if (rc) break;
        int neg = 0;
        uint64_t n = garner_value(&G, res + i, qstride, limbs, &neg);
        for (uint64_t c = k; c < np && !rc; c++) if (limbs_mod(limbs, n, neg, &p[c]) != res[c * qstride + i]) rc = TR_MATH_INVALID;
        uint64_t bl = n ? 64 * (n - 1) + bitlen64(limbs[n - 1]) : 0;
        if (!rc && bl > bits) rc = TR_MATH_INVALID;
        if (!rc && n > dst->cap) rc = TR_MATH_OVERFLOW;
        if (!rc) bv_set_mag(dst, at + i, neg, limbs, n);
    }
    garner_free(&G);
    free(limbs);
    return rc;
}

/* ------------------------------------------------------------------ the CPU references of the kernels */
int32_t tr_dyn_apply_view_eval(const tr_dyn_apply_view *v, tr_sym_wide *out, const tr_math_budget *budget,
                               tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (tr_dyn_apply_view_check(v) || !out) return TR_MATH_INVALID;
    const uint64_t total = (uint64_t)v->nops * v->ngroups * (uint64_t)v->nb;
    tr_math_need need;
    need_zero(&need);
    need.need[TR_BUDGET_OUTPUT] = total;
    need.need[TR_BUDGET_BYTES] = tr_budget_mul(total, sizeof(tr_sym_wide));
    int32_t rc = check_need(budget, &need, refusal);
    if (rc) return rc;
    for (uint64_t w = 0; w < total; w++) if (!tr_dyn_apply_entry(v, w, &out[w])) return TR_MATH_OVERFLOW;
    return TR_MATH_OK;
}
/* N_op^e mod p for one (operator, prime) block (the order of the device: base reduced, binary method) */
static void powmod_block(const tr_dyn_mod_view *v, uint64_t op, uint64_t q, uint64_t *R, uint64_t *B, uint64_t *T) {
    const int32_t nb = v->nb;
    const uint64_t e2 = (uint64_t)nb * (uint64_t)nb;
    const tr_gfp *f = &v->primes[q];
    for (uint64_t x = 0; x < e2; x++) { B[x] = tr_dyn_reduce_i128(f, v->num[op * e2 + x]); R[x] = (x % (uint64_t)(nb + 1)) == 0 ? 1 % f->p : 0; }
    uint64_t e = v->exponent;
    while (e) {
        if (e & 1) {
            for (int32_t i = 0; i < nb; i++) for (int32_t j = 0; j < nb; j++) T[(uint64_t)i * nb + j] = tr_dyn_mulmod_entry(f, R, B, nb, i, j);
            memcpy(R, T, (size_t)e2 * 8);
        }
        e >>= 1;
        if (e) {
            for (int32_t i = 0; i < nb; i++) for (int32_t j = 0; j < nb; j++) T[(uint64_t)i * nb + j] = tr_dyn_mulmod_entry(f, B, B, nb, i, j);
            memcpy(B, T, (size_t)e2 * 8);
        }
    }
}
static int32_t mod_need(const tr_dyn_mod_view *v, uint64_t ngroups, tr_math_need *need) {
    need_zero(need);
    const uint64_t nb = (uint64_t)v->nb, nm = tr_budget_mul((uint64_t)v->nops, v->nprimes);
    need->need[TR_BUDGET_TENSOR] = tr_budget_mul(nb, nb);
    need->need[TR_BUDGET_ITERATIONS] = v->exponent;
    /* output: the values of the request (every operator's power or evolved moments); the residues are bytes */
    need->need[TR_BUDGET_OUTPUT] = ngroups ? tr_budget_mul(tr_budget_mul((uint64_t)v->nops, ngroups), nb)
                                           : tr_budget_mul((uint64_t)v->nops, nb * nb);
    uint64_t res = ngroups ? tr_budget_mul(tr_budget_mul(nm, ngroups), nb) : tr_budget_mul(nm, nb * nb);
    need->need[TR_BUDGET_BYTES] = tr_budget_add(tr_budget_mul(res, 8), tr_budget_mul(nb * nb, 24));
    return TR_MATH_OK;
}
int32_t tr_dyn_powmod_view_eval(const tr_dyn_mod_view *v, uint64_t *out, const tr_math_budget *budget,
                                tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (tr_dyn_mod_view_check(v) || !out) return TR_MATH_INVALID;
    tr_math_need need;
    mod_need(v, 0, &need);
    int32_t rc = check_need(budget, &need, refusal);
    if (rc) return rc;
    const uint64_t e2 = (uint64_t)v->nb * (uint64_t)v->nb;
    uint64_t *B = (uint64_t *)zalloc(e2 * 8), *T = (uint64_t *)zalloc(e2 * 8);
    if (!B || !T) { free(B); free(T); return TR_MATH_MEMORY; }
    for (uint64_t op = 0; op < (uint64_t)v->nops; op++)
        for (uint64_t q = 0; q < v->nprimes; q++) powmod_block(v, op, q, out + (op * v->nprimes + q) * e2, B, T);
    free(B); free(T);
    return TR_MATH_OK;
}
int32_t tr_dyn_evolvemod_view_eval(const tr_dyn_mod_view *v, const tr_math_i128 *moments, uint64_t ngroups,
                                   uint64_t *out, const tr_math_budget *budget, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (tr_dyn_mod_view_check(v) || !out || !moments || !ngroups) return TR_MATH_INVALID;
    tr_math_need need;
    mod_need(v, ngroups, &need);
    int32_t rc = check_need(budget, &need, refusal);
    if (rc) return rc;
    const int32_t nb = v->nb;
    const uint64_t e2 = (uint64_t)nb * (uint64_t)nb;
    uint64_t *R = (uint64_t *)zalloc(e2 * 8), *B = (uint64_t *)zalloc(e2 * 8), *T = (uint64_t *)zalloc(e2 * 8);
    uint64_t *mm = (uint64_t *)zalloc((uint64_t)nb * 8);
    if (!R || !B || !T || !mm) { free(R); free(B); free(T); free(mm); return TR_MATH_MEMORY; }
    for (uint64_t op = 0; op < (uint64_t)v->nops; op++)
        for (uint64_t q = 0; q < v->nprimes; q++) {
            powmod_block(v, op, q, R, B, T);
            const tr_gfp *f = &v->primes[q];
            for (uint64_t g = 0; g < ngroups; g++) {
                for (int32_t j = 0; j < nb; j++) mm[j] = tr_dyn_reduce_i128(f, moments[g * (uint64_t)nb + j]);
                uint64_t *o = out + ((op * v->nprimes + q) * ngroups + g) * (uint64_t)nb;
                for (int32_t i = 0; i < nb; i++) o[i] = tr_dyn_applymod_entry(f, R, mm, nb, i);
            }
        }
    free(R); free(B); free(T); free(mm);
    return TR_MATH_OK;
}

/* ================================================================== bounds, costs and needs */
static uint64_t mom_bits(const tr_math_i128 *m, uint64_t count) {
    uint64_t b = 0;
    for (uint64_t i = 0; i < count; i++) { uint64_t x = bitlen128(mag128(tr_i128_get(m[i]))); if (x > b) b = x; }
    return b;
}
/* the proven bound (bits) of N^n's entries: n bitlen(||N||_inf) (I: 1) */
static uint64_t pow_bits(const tr_dyn_operator *o, uint64_t n) {
    if (n == 0) return 1;
    uint64_t b = tr_budget_mul(n, o->norm_bits);
    return b ? b : 1;
}
static uint64_t den_bits(const tr_dyn_operator *o, uint64_t n) {
    uint64_t b = tr_budget_mul(n, bitlen128((u128)o->den));
    return b ? b : 1;
}
/* AUTO's cost estimates, in 1/16 ns of the reference machine (one 2.1 GHz Xeon core). Only AUTO's choice reads them;
 * they bound nothing. Each method's structural counts -- loop steps, nonzero products (N's pattern: pairs = sum_k
 * colnnz_k rownnz_k, those of N^2; the powers' patterns are taken as N's, exact for patterns closed under products such
 * as the degree-triangular ones of polynomial dynamics), limb products from the proven sizes, residues, Garner steps,
 * primes -- times unit costs fitted to timed runs of every method (the benchmark's operators and two dense ones). */
#define CU_IT 27          /* a step of a big-integer matrix product's loop (the zero tests): 1.7 ns */
#define CU_PAIR 174       /* a nonzero big-integer product's overhead: 10.9 ns */
#define CU_LIMB 16        /* a limb product of the schoolbook multiplication (large dense operands): 1 ns */
#define CU_SIT 45         /* stepping: a step of the int128 x big-integer loop: 2.8 ns */
#define CU_SPAIR 60       /* stepping: a nonzero product's overhead: 3.8 ns */
#define CU_SLIMB 14       /* stepping: a limb product: 0.87 ns */
#define CU_RED 480        /* an int128 reduced modulo p (two divisions): 30 ns */
#define CU_ITM 26         /* a step of a residue matrix product's loop: 1.6 ns */
#define CU_MUL 65         /* a Barrett product and sum: 4 ns */
#define CU_GARNER 130     /* one (value, prime x prime) step of Garner's reconstruction: 8.1 ns */
#define CU_PRIME 374600   /* a prime of LIN-002's table (candidates tested downward from 2^62): 23 us */
static uint64_t cu_mul3(uint64_t a, uint64_t b, uint64_t c) { return tr_budget_mul(tr_budget_mul(a, b), c); }
/* one product N^ea N^eb of the binary method (bv_matmul) */
static uint64_t product_cost(const tr_dyn_operator *o, uint64_t ea, uint64_t eb) {
    const uint64_t nb = (uint64_t)o->nb;
    const uint64_t ll = tr_budget_mul(limbs_for(pow_bits(o, ea)), limbs_for(pow_bits(o, eb)));
    return tr_budget_add(cu_mul3(tr_budget_mul(nb, nb), nb, CU_IT),
                         tr_budget_mul(o->pairs, tr_budget_add(CU_PAIR, tr_budget_mul(ll, CU_LIMB))));
}
/* the binary method's products (squaring) and their number */
static uint64_t squaring_cost(const tr_dyn_operator *o, uint64_t n, uint64_t *products) {
    uint64_t cost = 0, prods = 0, eR = 0, eB = 1, e = n;
    while (e) {
        if (e & 1) {
            if (eR) { cost = tr_budget_add(cost, product_cost(o, eR, eB)); prods++; }
            eR += eB;
        }
        e >>= 1;
        if (e) { cost = tr_budget_add(cost, product_cost(o, eB, eB)); prods++; eB *= 2; }
    }
    if (products) *products = prods;
    return cost;
}
/* y_g = P m_g for G groups, P's entries of L limbs (bv_matvec_big: squaring's last step) */
static uint64_t matvec_cost(const tr_dyn_operator *o, uint64_t G, uint64_t L) {
    const uint64_t nb = (uint64_t)o->nb;
    return tr_budget_mul(G, tr_budget_add(tr_budget_mul(tr_budget_mul(nb, nb), CU_IT),
                                          tr_budget_mul(o->nnz0, tr_budget_add(CU_PAIR, cu_mul3(2, L, CU_LIMB)))));
}
static uint64_t modular_products(uint64_t n) { return n ? popcount64(n) + floorlog2(n) : 0; }
/* k primes (the check included): the primes, per prime the reduced base and the binary method's residue products
 * (and with groups the reduced moments and their products), then Garner's O(k^2) reconstruction of every value */
static uint64_t modular_cost(const tr_dyn_operator *o, uint64_t n, uint64_t k, uint64_t groups) {
    const uint64_t nb = (uint64_t)o->nb, n2 = tr_budget_mul(nb, nb), n3 = tr_budget_mul(n2, nb);
    const uint64_t prods = modular_products(n);
    uint64_t per = tr_budget_add(tr_budget_mul(n2, CU_RED),
                                 tr_budget_mul(prods, tr_budget_add(tr_budget_mul(n3, CU_ITM), tr_budget_mul(o->pairs, CU_MUL))));
    if (groups)
        per = tr_budget_add(per, tr_budget_mul(groups, tr_budget_add(tr_budget_mul(nb, CU_RED),
                                                                    tr_budget_add(tr_budget_mul(n2, CU_ITM), tr_budget_mul(o->nnz0, CU_MUL)))));
    const uint64_t vals = groups ? tr_budget_mul(groups, nb) : n2;
    uint64_t c = tr_budget_add(tr_budget_mul(k, CU_PRIME), tr_budget_mul(k, per));
    return tr_budget_add(c, cu_mul3(vals, tr_budget_mul(k, k), CU_GARNER));
}
/* t steps v <- N v for G groups: step s multiplies int128 entries by values of about s bitlen(norm) + mbits bits */
static uint64_t stepping_cost(const tr_dyn_operator *o, uint64_t t, uint64_t groups, uint64_t mbits) {
    const uint64_t nb = (uint64_t)o->nb;
    /* sum_(s=1..t) limbs_for(s norm_bits + mbits) <= t (mbits / 64 + 2) + norm_bits t (t + 1) / 128 */
    uint64_t limbs = tr_budget_add(tr_budget_mul(t, tr_budget_add(mbits / 64, 2)),
                                   tr_budget_mul(o->norm_bits, tr_budget_mul(t, tr_budget_add(t, 1)) / 128));
    uint64_t fixed = tr_budget_mul(t, tr_budget_add(tr_budget_mul(tr_budget_mul(nb, nb), CU_SIT), tr_budget_mul(o->nnz0, CU_SPAIR)));
    return tr_budget_mul(groups, tr_budget_add(fixed, cu_mul3(o->nnz0, tr_budget_mul(limbs, 2), CU_SLIMB)));
}
static int32_t state_frame(const tr_sym_state *s, const tr_dyn_operator *o, tr_sym_state_info *si, tr_sym_query_view *qv) {
    if (!s || tr_sym_state_get_info(s, si) || tr_sym_state_get_view(s, qv)) return TR_MATH_INVALID;
    if (!o->geo.digest[0] && !o->geo.digest[1]) return TR_MATH_DOMAIN_ERROR;    /* an abstract geometry has no state */
    if (si->digest[0] != o->geo.digest[0] || si->digest[1] != o->geo.digest[1] || si->n != o->n) return TR_MATH_DOMAIN_ERROR;
    if (tr_sym_basis_equal(tr_sym_state_get_basis(s), o->basis) != 1) return TR_MATH_DOMAIN_ERROR;
    return TR_MATH_OK;
}
/* the resolved method and the needs of a power (s NULL) or an evolution */
static int32_t need_core(int32_t request, const tr_dyn_operator *o, const tr_sym_state *s, uint64_t n, int32_t method,
                         int32_t target, tr_math_need *need, int32_t *chosen, uint64_t costs[3]) {
    need_zero(need);
    const uint64_t nb = (uint64_t)o->nb, e2 = nb * nb;
    uint64_t G = 0, mbits = 0;
    tr_sym_state_info si;
    tr_sym_query_view qv;
    if (request == TR_DYN_REQ_EVOLVE || request == TR_DYN_REQ_APPLY || (request == TR_DYN_REQ_FLOAT && s)) {
        int32_t rc = state_frame(s, o, &si, &qv);
        if (rc) return rc;
        G = qv.ngroups;
        mbits = mom_bits(qv.moments, G * nb);
    }
    need->need[TR_BUDGET_VARIABLES] = (uint64_t)o->n;
    need->need[TR_BUDGET_SUPPORT] = nb;
    if (request == TR_DYN_REQ_APPLY) {
        need->need[TR_BUDGET_TENSOR] = e2;
        need->need[TR_BUDGET_OUTPUT] = tr_budget_mul(G, nb);
        need->need[TR_BUDGET_STATES] = tr_budget_mul(G, nb);
        uint64_t bytes = tr_budget_mul(tr_budget_mul(G, nb), sizeof(tr_sym_wide));
        if (target == TR_DYN_NEED_DEVICE) {
            tr_dyn_apply_view v;
            v.nb = (int32_t)nb; v.nops = 1; v.ngroups = G; v.num = o->num; v.moments = qv.moments;
            bytes = tr_dyn_apply_device_bytes(&v);
        }
        need->need[TR_BUDGET_BYTES] = bytes;
        if (chosen) *chosen = TR_DYN_STEPPING;
        return TR_MATH_OK;
    }
    if (request == TR_DYN_REQ_FLOAT) {
        need->need[TR_BUDGET_TENSOR] = e2;
        need->need[TR_BUDGET_ITERATIONS] = n;
        need->need[TR_BUDGET_OUTPUT] = s ? tr_budget_mul(G, nb) : e2;
        need->need[TR_BUDGET_BYTES] = tr_budget_add(tr_budget_mul(e2, 3 * sizeof(tr_math_interval)),
                                                    tr_budget_mul(tr_budget_mul(G, nb), 2 * sizeof(tr_math_interval)));
        if (chosen) *chosen = TR_DYN_FLOAT;
        return TR_MATH_OK;
    }
    const int evolve = request == TR_DYN_REQ_EVOLVE;
    const int param = o->pdegree > 0;
    uint64_t pb = pow_bits(o, n), bound = evolve ? tr_budget_add(pb, mbits) : pb;
    uint64_t k = primes_for(bound), prods = 0;
    uint64_t c_sq = squaring_cost(o, n, &prods);
    if (evolve) c_sq = tr_budget_add(c_sq, matvec_cost(o, G, limbs_for(pb)));
    uint64_t c_mod = modular_cost(o, n, k == TR_BUDGET_UNKNOWN ? TR_BUDGET_UNKNOWN : k, evolve ? G : 0);
    uint64_t c_st = evolve ? stepping_cost(o, n, G, mbits) : TR_BUDGET_UNKNOWN;
    if (costs) { costs[0] = c_st; costs[1] = c_sq; costs[2] = c_mod; }
    int32_t m = method;
    const int truncated = o->kind == TR_DYN_OP_TRUNCATED;
    if (truncated && !evolve) return TR_MATH_DOMAIN_ERROR;      /* an approximate operator has no exact powers */
    if (param) {                                                 /* parametric moments: stepping only */
        if (!evolve || (m != TR_DYN_AUTO && m != TR_DYN_STEPPING)) return TR_MATH_UNSUPPORTED;
        if (n > ((uint64_t)1 << 24) || n * (uint64_t)o->pdegree > ((uint64_t)1 << 30)) return TR_MATH_UNSUPPORTED;
        m = TR_DYN_STEPPING;
    }
    const int modular_ok = !truncated && k <= TR_DYN_MAX_PRIMES;
    if (m == TR_DYN_AUTO) {
        if (target == TR_DYN_NEED_DEVICE) m = TR_DYN_MODULAR;
        else if (evolve) m = (c_st <= c_sq && (c_st <= c_mod || !modular_ok)) ? TR_DYN_STEPPING
                           : ((c_sq <= c_mod || !modular_ok) ? TR_DYN_SQUARING : TR_DYN_MODULAR);
        else m = (c_sq <= c_mod || n <= 1 || !modular_ok) ? TR_DYN_SQUARING : TR_DYN_MODULAR;
    }
    if (m != TR_DYN_SQUARING && m != TR_DYN_MODULAR && !(evolve && m == TR_DYN_STEPPING)) return TR_MATH_INVALID;
    if (target == TR_DYN_NEED_DEVICE && m != TR_DYN_MODULAR) return TR_MATH_INVALID;
    if (m == TR_DYN_MODULAR && truncated) return TR_MATH_UNSUPPORTED;   /* truncated evolutions: stepping or squaring */
    if (m == TR_DYN_MODULAR && k > TR_DYN_MAX_PRIMES) return TR_MATH_UNSUPPORTED;   /* past LIN-002's prime table */
    if (chosen) *chosen = m;
    need->need[TR_BUDGET_TENSOR] = tr_budget_mul(e2, (uint64_t)o->pdegree + 1);
    need->need[TR_BUDGET_ITERATIONS] = n;
    const uint64_t vals = evolve ? tr_budget_mul(tr_budget_mul(G, nb), param ? tr_budget_add(tr_budget_mul(n, (uint64_t)o->pdegree), 1) : 1) : e2;
    uint64_t out = vals;
    if (evolve && o->kind == TR_DYN_OP_TRUNCATED) out = tr_budget_mul(out, 2);
    need->need[TR_BUDGET_OUTPUT] = out;
    if (evolve) {
        need->need[TR_BUDGET_STATES] = tr_budget_mul(G, nb);
        need->need[TR_BUDGET_DEGREE] = param ? tr_budget_mul(n, (uint64_t)o->pdegree) : 0;
    }
    const uint64_t cap = limbs_for(bound), dcap = limbs_for(den_bits(o, n));
    uint64_t bytes = tr_budget_add(bvec_bytes(vals, cap), bvec_bytes(evolve ? nb : 1, dcap));
    if (m == TR_DYN_STEPPING) bytes = tr_budget_add(bytes, bvec_bytes(vals, cap));
    if (m == TR_DYN_SQUARING) bytes = tr_budget_add(bytes, tr_budget_mul(bvec_bytes(e2, limbs_for(pb)), 3));
    if (m == TR_DYN_MODULAR) {
        uint64_t res = evolve ? tr_budget_mul(tr_budget_mul(k, G), nb) : tr_budget_mul(k, e2);
        bytes = tr_budget_add(bytes, tr_budget_add(tr_budget_mul(res, 8), tr_budget_mul(e2, 24)));
        bytes = tr_budget_add(bytes, tr_budget_mul(k, sizeof(tr_gfp) + 24));
        if (target == TR_DYN_NEED_DEVICE) {
            tr_dyn_mod_view v;
            v.nb = (int32_t)nb; v.nops = 1; v.nprimes = k; v.exponent = n; v.num = o->num; v.primes = NULL;
            bytes = tr_budget_add(bytes, tr_dyn_mod_device_bytes(&v, evolve ? G : 0));
        }
    }
    bytes = tr_budget_add(bytes, tr_budget_mul(tr_budget_add(tr_budget_mul(cap, 2), 4), 16));   /* accumulators */
    if (evolve && o->kind == TR_DYN_OP_TRUNCATED) {              /* the error recursion */
        uint64_t eb = tr_budget_add(tr_budget_mul(n, bitlen128((u128)o->den) + o->norm_bits + 1), 8192);
        bytes = tr_budget_add(bytes, tr_budget_mul(bvec_bytes(tr_budget_mul(G, nb), limbs_for(eb)), 3));
    }
    need->need[TR_BUDGET_BYTES] = bytes;
    return TR_MATH_OK;
}
int32_t tr_dyn_need(int32_t request, const tr_dyn_operator *o, const tr_sym_state *s, uint64_t n, int32_t method,
                    int32_t target, tr_math_need *need, int32_t *chosen) {
    if (!op_ok(o) || !need || (target != TR_DYN_NEED_CPU && target != TR_DYN_NEED_DEVICE)) return TR_MATH_INVALID;
    const uint64_t nb = (uint64_t)o->nb;
    switch (request) {
    case TR_DYN_REQ_POWER: case TR_DYN_REQ_EVOLVE: case TR_DYN_REQ_FLOAT: case TR_DYN_REQ_APPLY:
        return need_core(request, o, s, n, method, target, need, chosen, NULL);
    case TR_DYN_REQ_CHARPOLY: case TR_DYN_REQ_RECURRENCE: {
        need_zero(need);
        uint64_t bound = tr_budget_add(tr_budget_mul(nb, o->norm_bits + 1), nb + 1);
        uint64_t k = primes_for(bound);
        need->need[TR_BUDGET_VARIABLES] = (uint64_t)o->n;
        need->need[TR_BUDGET_SUPPORT] = nb;
        need->need[TR_BUDGET_TENSOR] = nb * nb;
        need->need[TR_BUDGET_ITERATIONS] = request == TR_DYN_REQ_RECURRENCE ? 2 * nb : nb;
        need->need[TR_BUDGET_OUTPUT] = request == TR_DYN_REQ_RECURRENCE ? 2 * nb : nb + 1;
        uint64_t bytes = tr_budget_add(tr_budget_mul(k, tr_budget_add(tr_budget_mul(nb, 8), nb * nb * 8)),
                                       tr_budget_mul(nb + 1, tr_budget_mul(limbs_for(tr_budget_add(bound, tr_budget_mul(nb, bitlen128((u128)o->den)))), 16)));
        if (request == TR_DYN_REQ_RECURRENCE) {
            uint64_t sb = tr_budget_add(tr_budget_mul(2 * nb, o->norm_bits), 256);
            bytes = tr_budget_add(bytes, tr_budget_mul(tr_budget_mul(4 * nb, limbs_for(sb)), tr_budget_mul(nb, 16)));
        }
        need->need[TR_BUDGET_BYTES] = bytes;
        if (chosen) *chosen = request == TR_DYN_REQ_CHARPOLY ? TR_DYN_MODULAR : TR_DYN_STEPPING;
        return TR_MATH_OK;
    }
    default:
        return TR_MATH_INVALID;
    }
}
int32_t tr_dyn_mod_plan(const tr_dyn_operator *o, const tr_sym_state *s, uint64_t n, uint64_t *nprimes, uint64_t *bound) {
    if (!op_ok(o) || !nprimes) return TR_MATH_INVALID;
    if (o->kind != TR_DYN_OP_EXACT) return TR_MATH_DOMAIN_ERROR;
    if (o->pdegree) return TR_MATH_UNSUPPORTED;
    uint64_t b = pow_bits(o, n);
    if (s) {
        tr_sym_state_info si;
        tr_sym_query_view qv;
        int32_t rc = state_frame(s, o, &si, &qv);
        if (rc) return rc;
        b = tr_budget_add(b, mom_bits(qv.moments, qv.ngroups * (uint64_t)o->nb));
    }
    uint64_t k = primes_for(b);
    if (k > TR_DYN_MAX_PRIMES) return TR_MATH_UNSUPPORTED;
    *nprimes = k;
    if (bound) *bound = b;
    return TR_MATH_OK;
}
int32_t tr_dyn_mod_bind(const tr_dyn_operator *o, uint64_t n, const tr_gfp *primes, uint64_t np, tr_dyn_mod_view *v) {
    if (!op_ok(o) || !v || !primes || !np) return TR_MATH_INVALID;
    if (o->kind != TR_DYN_OP_EXACT) return TR_MATH_DOMAIN_ERROR;
    if (o->pdegree) return TR_MATH_UNSUPPORTED;
    memset(v, 0, sizeof *v);
    v->nb = o->nb; v->nops = 1; v->nprimes = np; v->exponent = n; v->num = o->num; v->primes = primes;
    return tr_dyn_mod_view_check(v) ? TR_MATH_INVALID : TR_MATH_OK;
}
int32_t tr_dyn_apply_bind(const tr_sym_state *s, const tr_dyn_operator *o, tr_dyn_apply_view *v) {
    if (!op_ok(o) || !v) return TR_MATH_INVALID;
    if (o->kind != TR_DYN_OP_EXACT) return TR_MATH_DOMAIN_ERROR;
    if (o->pdegree) return TR_MATH_UNSUPPORTED;
    tr_sym_state_info si;
    tr_sym_query_view qv;
    int32_t rc = state_frame(s, o, &si, &qv);
    if (rc) return rc;
    memset(v, 0, sizeof *v);
    v->nb = o->nb; v->nops = 1; v->ngroups = qv.ngroups; v->num = o->num; v->moments = qv.moments;
    return TR_MATH_OK;
}

/* ================================================================== powers */
struct tr_dyn_power_s {
    uint32_t magic;
    int32_t nb, method, primes, checked, identity, backend;
    uint64_t exponent, products, bound_bits, cost_sq, cost_mod, bytes;
    tr_sym_geometry geo;
    tr_sym_basis *basis;
    bvec num, den;                         /* nb x nb numerators; one denominator */
};
static int pow_ok(const tr_dyn_power *p) { return p && p->magic == POW_MAGIC; }
static tr_dyn_power *pow_fail(tr_dyn_power *p, int32_t *error, int32_t code) {
    if (p) { tr_sym_basis_free(p->basis); bvec_free(&p->num); bvec_free(&p->den); free(p); }
    set_error(error, code);
    return NULL;
}
static tr_dyn_power *pow_new(const tr_sym_basis *b, const tr_sym_geometry *g, int32_t nb, uint64_t cap, uint64_t dcap, int32_t *rc) {
    tr_dyn_power *p = (tr_dyn_power *)zalloc(sizeof *p);
    if (!p) { *rc = TR_MATH_MEMORY; return NULL; }
    p->nb = nb; p->geo = *g;
    p->basis = tr_sym_basis_clone(b, rc);
    if (!p->basis || !bvec_init(&p->num, (uint64_t)nb * (uint64_t)nb, cap) || !bvec_init(&p->den, 1, dcap)) {
        pow_fail(p, NULL, 0); *rc = TR_MATH_MEMORY; return NULL;
    }
    p->bytes = sizeof *p + bvec_bytes((uint64_t)nb * nb, cap) + bvec_bytes(1, dcap) + basis_bytes_of(b);
    return p;
}
/* *out = D^n */
static int32_t den_power(const tr_dyn_operator *o, uint64_t n, bvec *d, uint64_t at) {
    tr_math_budget u;
    unlimited(&u);
    int32_t err = 0;
    tr_bigint *D = tr_bigint_from_i128(tr_i128_set(o->den), &u, NULL, &err);
    tr_bigint *P = D ? tr_bigint_pow(D, n, &u, NULL, &err) : NULL;
    int ok = P && bv_from_bigint(d, at, P);
    tr_bigint_free(D); tr_bigint_free(P);
    return ok ? TR_MATH_OK : (err ? err : TR_MATH_OVERFLOW);
}
static int32_t bv_matmul(const bvec *A, const bvec *B, int32_t nb, bvec *C, acc_t *acc) {
    for (int32_t i = 0; i < nb; i++)
        for (int32_t j = 0; j < nb; j++) {
            for (int32_t k = 0; k < nb; k++) {
                uint64_t a = (uint64_t)i * nb + k, b = (uint64_t)k * nb + j;
                if (A->len[a] && B->len[b]) acc_bb(acc, A, a, B, b);
            }
            if (!acc_store(acc, C, (uint64_t)i * nb + j)) return TR_MATH_OVERFLOW;
        }
    return TR_MATH_OK;
}
/* N^n by the binary method into P (cap from the bound); *products */
static int32_t power_squaring(const tr_dyn_operator *o, uint64_t n, bvec *P, uint64_t *products) {
    const int32_t nb = o->nb;
    const uint64_t e2 = (uint64_t)nb * (uint64_t)nb;
    *products = 0;
    if (n == 0) {
        for (uint64_t x = 0; x < e2; x++) { if (x % (uint64_t)(nb + 1) == 0) bv_set_i128(P, x, 1); else bv_set_i128(P, x, 0); }
        return TR_MATH_OK;
    }
    bvec B, R, T;
    int haveR = 0;
    uint64_t eB = 1, eR = 0, e = n;
    if (!bvec_init(&B, e2, limbs_for(pow_bits(o, 1)))) return TR_MATH_MEMORY;
    for (uint64_t x = 0; x < e2; x++) bv_set_i128(&B, x, tr_i128_get(o->num[x]));
    acc_t acc;
    if (!acc_init(&acc, 2 * P->cap + 4)) { bvec_free(&B); return TR_MATH_MEMORY; }
    int32_t rc = TR_MATH_OK;
    memset(&R, 0, sizeof R);
    while (e && !rc) {
        if (e & 1) {
            if (!haveR) {
                if (!bvec_init(&R, e2, limbs_for(pow_bits(o, eR + eB)))) { rc = TR_MATH_MEMORY; break; }
                for (uint64_t x = 0; x < e2 && !rc; x++) if (!bv_copy(&R, x, &B, x)) rc = TR_MATH_OVERFLOW;
                haveR = 1;
            } else {
                if (!bvec_init(&T, e2, limbs_for(pow_bits(o, eR + eB)))) { rc = TR_MATH_MEMORY; break; }
                rc = bv_matmul(&R, &B, nb, &T, &acc);
                bvec_free(&R); R = T;
                (*products)++;
            }
            eR += eB;
        }
        e >>= 1;
        if (e && !rc) {
            if (!bvec_init(&T, e2, limbs_for(pow_bits(o, 2 * eB)))) { rc = TR_MATH_MEMORY; break; }
            rc = bv_matmul(&B, &B, nb, &T, &acc);
            bvec_free(&B); B = T;
            eB *= 2;
            (*products)++;
        }
    }
    for (uint64_t x = 0; x < e2 && !rc; x++) if (!bv_copy(P, x, &R, x)) rc = TR_MATH_OVERFLOW;
    bvec_free(&B); bvec_free(&R); acc_free(&acc);
    return rc;
}
static void pow_finish(tr_dyn_power *p) {
    const uint64_t e2 = (uint64_t)p->nb * (uint64_t)p->nb;
    int id = 1;
    for (uint64_t x = 0; x < e2 && id; x++) {
        int diag = x % (uint64_t)(p->nb + 1) == 0;
        if (diag) id = p->num.len[x] == p->den.len[0] && p->num.neg[x] == p->den.neg[0] &&
                      !memcmp(BV(&p->num, x), BV(&p->den, 0), (size_t)p->den.len[0] * 8);
        else id = p->num.len[x] == 0;
    }
    p->identity = id;
    p->magic = POW_MAGIC;
}
static int32_t power_check(const tr_dyn_operator *o) {
    if (!op_ok(o)) return TR_MATH_INVALID;
    if (o->kind != TR_DYN_OP_EXACT) return TR_MATH_DOMAIN_ERROR;   /* an approximate operator has no exact powers */
    if (o->pdegree) return TR_MATH_UNSUPPORTED;
    return TR_MATH_OK;
}
static tr_dyn_power *power_adopt(const tr_dyn_operator *o, uint64_t n, const tr_gfp *primes, uint64_t np,
                                 const uint64_t *res, int32_t backend, int prove, const tr_math_budget *budget,
                                 tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    int32_t rc = power_check(o);
    if (rc) return pow_fail(NULL, error, rc);
    if (!res || !primes_valid(primes, np, prove) || (backend != TR_CERT_BY_CPU && backend != TR_CERT_BY_DEVICE))
        return pow_fail(NULL, error, TR_MATH_INVALID);
    tr_math_need need;
    if ((rc = need_core(TR_DYN_REQ_POWER, o, NULL, n, TR_DYN_MODULAR, TR_DYN_NEED_CPU, &need, NULL, NULL))) return pow_fail(NULL, error, rc);
    if ((rc = check_need(budget, &need, refusal))) return pow_fail(NULL, error, rc);
    const int32_t nb = o->nb;
    const uint64_t e2 = (uint64_t)nb * (uint64_t)nb, bits = pow_bits(o, n);
    if (np < primes_for(bits)) return pow_fail(NULL, error, TR_MATH_INVALID);
    tr_dyn_power *p = pow_new(o->basis, &o->geo, nb, limbs_for(bits), limbs_for(den_bits(o, n)), &rc);
    if (!p) return pow_fail(NULL, error, rc);
    rc = crt_values(primes, np, res, e2, e2, bits, &p->num, 0);
    if (!rc) rc = den_power(o, n, &p->den, 0);
    if (rc) return pow_fail(p, error, rc);
    p->exponent = n; p->method = TR_DYN_MODULAR; p->primes = (int32_t)np; p->checked = 1; p->backend = backend;
    p->bound_bits = bits; p->products = modular_products(n);
    p->cost_sq = squaring_cost(o, n, NULL); p->cost_mod = modular_cost(o, n, np, 0);
    pow_finish(p);
    set_error(error, TR_MATH_OK);
    return p;
}
tr_dyn_power *tr_dyn_power_adopt(const tr_dyn_operator *o, uint64_t n, const tr_gfp *primes, uint64_t np,
                                 const uint64_t *res, int32_t backend, const tr_math_budget *budget,
                                 tr_math_refusal *refusal, int32_t *error) {
    return power_adopt(o, n, primes, np, res, backend, 1, budget, refusal, error);
}
tr_dyn_power *tr_dyn_power_make(const tr_dyn_operator *o, uint64_t n, int32_t method, const tr_math_budget *budget,
                                tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    int32_t rc = power_check(o);
    if (rc) return pow_fail(NULL, error, rc);
    tr_math_need need;
    int32_t m = 0;
    if ((rc = need_core(TR_DYN_REQ_POWER, o, NULL, n, method, TR_DYN_NEED_CPU, &need, &m, NULL))) return pow_fail(NULL, error, rc);
    if ((rc = check_need(budget, &need, refusal))) return pow_fail(NULL, error, rc);
    if (m == TR_DYN_MODULAR) {
        uint64_t np = 0;
        if ((rc = tr_dyn_mod_plan(o, NULL, n, &np, NULL))) return pow_fail(NULL, error, rc);
        const uint64_t e2 = (uint64_t)o->nb * (uint64_t)o->nb;
        tr_gfp *pr = (tr_gfp *)zalloc(np * sizeof(tr_gfp));
        uint64_t *res = (uint64_t *)zalloc(tr_budget_mul(np, e2) * 8);
        tr_dyn_mod_view v;
        if (!pr || !res) rc = TR_MATH_MEMORY;
        if (!rc) rc = tr_dyn_primes(np, pr);
        if (!rc) rc = tr_dyn_mod_bind(o, n, pr, np, &v);
        tr_math_budget u;
        unlimited(&u);
        if (!rc) rc = tr_dyn_powmod_view_eval(&v, res, &u, NULL);
        tr_dyn_power *p = rc ? NULL : power_adopt(o, n, pr, np, res, TR_CERT_BY_CPU, 0, budget, refusal, &rc);
        free(pr); free(res);
        if (!p) return pow_fail(NULL, error, rc);
        set_error(error, TR_MATH_OK);
        return p;
    }
    const uint64_t bits = pow_bits(o, n);
    tr_dyn_power *p = pow_new(o->basis, &o->geo, o->nb, limbs_for(bits), limbs_for(den_bits(o, n)), &rc);
    if (!p) return pow_fail(NULL, error, rc);
    rc = power_squaring(o, n, &p->num, &p->products);
    if (!rc) rc = den_power(o, n, &p->den, 0);
    if (rc) return pow_fail(p, error, rc);
    p->exponent = n; p->method = TR_DYN_SQUARING; p->bound_bits = bits; p->backend = TR_CERT_BY_CPU;
    p->cost_sq = squaring_cost(o, n, NULL); p->cost_mod = modular_cost(o, n, primes_for(bits), 0);
    pow_finish(p);
    set_error(error, TR_MATH_OK);
    return p;
}
tr_dyn_power *tr_dyn_power_compose(const tr_dyn_power *A, const tr_dyn_power *B, const tr_math_budget *budget,
                                   tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    if (!pow_ok(A) || !pow_ok(B)) return pow_fail(NULL, error, TR_MATH_INVALID);
    if (tr_sym_basis_equal(A->basis, B->basis) != 1 || !geo_equal(&A->geo, &B->geo)) return pow_fail(NULL, error, TR_MATH_DOMAIN_ERROR);
    const int32_t nb = A->nb;
    const uint64_t e2 = (uint64_t)nb * (uint64_t)nb;
    uint64_t ma = 0, mb = 0;
    for (uint64_t x = 0; x < e2; x++) { if (A->num.len[x] > ma) ma = A->num.len[x]; if (B->num.len[x] > mb) mb = B->num.len[x]; }
    uint64_t cap = ma + mb + 1 + (bitlen64((uint64_t)nb) + 63) / 64, dcap = (uint64_t)A->den.len[0] + B->den.len[0] + 1;
    tr_math_need need;
    need_zero(&need);
    need.need[TR_BUDGET_SUPPORT] = (uint64_t)nb;
    need.need[TR_BUDGET_TENSOR] = e2;
    need.need[TR_BUDGET_OUTPUT] = e2;
    need.need[TR_BUDGET_BYTES] = tr_budget_add(bvec_bytes(e2, cap), bvec_bytes(1, dcap));
    int32_t rc = check_need(budget, &need, refusal);
    if (rc) return pow_fail(NULL, error, rc);
    tr_dyn_power *p = pow_new(A->basis, &A->geo, nb, cap, dcap, &rc);
    acc_t acc;
    if (!p) return pow_fail(NULL, error, rc);
    if (!acc_init(&acc, 2 * cap + 4)) return pow_fail(p, error, TR_MATH_MEMORY);
    rc = bv_matmul(&A->num, &B->num, nb, &p->num, &acc);
    if (!rc) { acc_bb(&acc, &A->den, 0, &B->den, 0); if (!acc_store(&acc, &p->den, 0)) rc = TR_MATH_OVERFLOW; }
    acc_free(&acc);
    if (rc) return pow_fail(p, error, rc);
    p->exponent = A->exponent + B->exponent; p->method = A->method == B->method ? A->method : TR_DYN_AUTO;
    p->products = 1; p->backend = TR_CERT_BY_CPU;
    p->bound_bits = A->bound_bits + B->bound_bits + bitlen64((uint64_t)nb);
    pow_finish(p);
    set_error(error, TR_MATH_OK);
    return p;
}
/* entry x of A over its denominator equals entry x of B over its: A_x B_d == B_x A_d */
static int cross_equal(const bvec *an, uint64_t ai, const bvec *ad, const bvec *bn, uint64_t bi, const bvec *bd, acc_t *acc,
                       bvec *t1, bvec *t2) {
    acc_bb(acc, an, ai, bd, 0);
    if (!acc_store(acc, t1, 0)) return -1;
    acc_bb(acc, bn, bi, ad, 0);
    if (!acc_store(acc, t2, 0)) return -1;
    return t1->len[0] == t2->len[0] && t1->neg[0] == t2->neg[0] && !memcmp(BV(t1, 0), BV(t2, 0), (size_t)t1->len[0] * 8);
}
int32_t tr_dyn_power_equal(const tr_dyn_power *A, const tr_dyn_power *B) {
    if (!pow_ok(A) || !pow_ok(B)) return TR_MATH_INVALID;
    if (tr_sym_basis_equal(A->basis, B->basis) != 1 || A->nb != B->nb) return 0;
    const uint64_t e2 = (uint64_t)A->nb * (uint64_t)A->nb;
    uint64_t m = 0;
    for (uint64_t x = 0; x < e2; x++) { if (A->num.len[x] > m) m = A->num.len[x]; if (B->num.len[x] > m) m = B->num.len[x]; }
    uint64_t dm = A->den.len[0] > B->den.len[0] ? A->den.len[0] : B->den.len[0];
    uint64_t cap = m + dm + 2;
    acc_t acc;
    bvec t1, t2;
    if (!acc_init(&acc, 2 * cap + 4)) return TR_MATH_MEMORY;
    if (!bvec_init(&t1, 1, cap) || !bvec_init(&t2, 1, cap)) { acc_free(&acc); bvec_free(&t1); return TR_MATH_MEMORY; }
    int32_t r = 1;
    for (uint64_t x = 0; x < e2 && r == 1; x++) {
        int c = cross_equal(&A->num, x, &A->den, &B->num, x, &B->den, &acc, &t1, &t2);
        r = c < 0 ? TR_MATH_OVERFLOW : c;
    }
    acc_free(&acc); bvec_free(&t1); bvec_free(&t2);
    return r;
}
int32_t tr_dyn_power_get_info(const tr_dyn_power *p, tr_dyn_power_info *out) {
    if (!pow_ok(p) || !out) return TR_MATH_INVALID;
    memset(out, 0, sizeof *out);
    out->nb = (uint64_t)p->nb; out->exponent = p->exponent; out->method = p->method; out->primes = p->primes;
    out->identity = p->identity; out->checked = p->checked; out->backend = p->backend;
    out->bound_bits = p->bound_bits; out->den_bits = bv_bitlen(&p->den, 0);
    const uint64_t e2 = (uint64_t)p->nb * (uint64_t)p->nb;
    for (uint64_t x = 0; x < e2; x++) { uint64_t b = bv_bitlen(&p->num, x); if (b > out->max_bits) out->max_bits = b; }
    out->products = p->products; out->bytes = p->bytes; out->cost_squaring = p->cost_sq; out->cost_modular = p->cost_mod;
    out->digest[0] = p->geo.digest[0]; out->digest[1] = p->geo.digest[1];
    return TR_MATH_OK;
}
void tr_dyn_power_free(tr_dyn_power *p) {
    if (!pow_ok(p)) return;
    p->magic = 0;
    pow_fail(p, NULL, 0);
}
static tr_dyn_power *pow_of_op(const tr_dyn_operator *o, int32_t *rc) {      /* K itself (exponent 1) */
    tr_dyn_power *p = pow_new(o->basis, &o->geo, o->nb, 3, 3, rc);
    if (!p) return NULL;
    const uint64_t e2 = (uint64_t)o->nb * (uint64_t)o->nb;
    for (uint64_t x = 0; x < e2; x++) bv_set_i128(&p->num, x, tr_i128_get(o->num[x]));
    bv_set_i128(&p->den, 0, o->den);
    p->exponent = 1; p->method = TR_DYN_SQUARING; p->bound_bits = o->norm_bits;
    pow_finish(p);
    return p;
}
int32_t tr_dyn_semigroup(const tr_dyn_operator *o, uint64_t m, uint64_t n, int32_t method, tr_dyn_semigroup_evidence *ev,
                         const tr_math_budget *budget, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    int32_t rc = power_check(o);
    if (rc) return rc;
    if (!ev || m > UINT64_MAX - n) return TR_MATH_INVALID;
    memset(ev, 0, sizeof *ev);
    ev->m = m; ev->n = n; ev->method = method;
    tr_dyn_power *P0 = tr_dyn_power_make(o, 0, method, budget, refusal, &rc), *P1 = NULL, *Pm = NULL, *Pn = NULL, *Pmn = NULL,
                 *Pc = NULL, *K = NULL;
    if (P0) P1 = tr_dyn_power_make(o, 1, method, budget, refusal, &rc);
    if (P1) Pm = tr_dyn_power_make(o, m, method, budget, refusal, &rc);
    if (Pm) Pn = tr_dyn_power_make(o, n, method, budget, refusal, &rc);
    if (Pn) Pmn = tr_dyn_power_make(o, m + n, method, budget, refusal, &rc);
    if (Pmn) Pc = tr_dyn_power_compose(Pm, Pn, budget, refusal, &rc);
    if (Pc) K = pow_of_op(o, &rc);
    if (K) {
        ev->identity = P0->identity;
        int32_t e1 = tr_dyn_power_equal(P1, K), e2 = tr_dyn_power_equal(Pc, Pmn);
        if (e1 < 0 || e2 < 0) rc = e1 < 0 ? e1 : e2;
        ev->one = e1 == 1; ev->product = e2 == 1;
        ev->holds = ev->identity && ev->one && ev->product;
        ev->products = P0->products + P1->products + Pm->products + Pn->products + Pmn->products + 1;
    }
    tr_dyn_power_free(P0); tr_dyn_power_free(P1); tr_dyn_power_free(Pm); tr_dyn_power_free(Pn); tr_dyn_power_free(Pmn);
    tr_dyn_power_free(Pc); tr_dyn_power_free(K);
    return rc;
}

/* ================================================================== evolution */
struct tr_dyn_evolution_s {
    uint32_t magic;
    int32_t nb, method, label, kind, primes, invariant, pdegree, checked, backend, weighted;
    uint64_t ngroups, steps, bound_bits, rows_total, used_total, cost_st, cost_sq, cost_mod, bytes;
    uint64_t *rows, *used;
    bvec num;                              /* ngroups x nb x (pdegree + 1) */
    bvec den;                              /* nb (per basis term) */
    bvec err, errden;                      /* truncated: ngroups x nb, 1 */
    tr_sym_basis *basis;
    tr_sym_geometry geo;
};
static int ev_ok(const tr_dyn_evolution *e) { return e && e->magic == EV_MAGIC; }
static tr_dyn_evolution *ev_fail(tr_dyn_evolution *e, int32_t *error, int32_t code) {
    if (e) {
        free(e->rows); free(e->used); bvec_free(&e->num); bvec_free(&e->den); bvec_free(&e->err); bvec_free(&e->errden);
        tr_sym_basis_free(e->basis); free(e);
    }
    set_error(error, code);
    return NULL;
}
static tr_dyn_evolution *ev_new(const tr_sym_basis *b, const tr_sym_geometry *g, uint64_t G, int32_t nb, int32_t pdeg,
                                uint64_t cap, uint64_t dcap, int32_t *rc) {
    tr_dyn_evolution *e = (tr_dyn_evolution *)zalloc(sizeof *e);
    if (!e) { *rc = TR_MATH_MEMORY; return NULL; }
    e->nb = nb; e->ngroups = G; e->pdegree = pdeg; e->geo = *g;
    e->basis = tr_sym_basis_clone(b, rc);
    e->rows = (uint64_t *)zalloc(G * 8); e->used = (uint64_t *)zalloc(G * 8);
    uint64_t vals = tr_budget_mul(tr_budget_mul(G, (uint64_t)nb), (uint64_t)pdeg + 1);
    if (!e->basis || !e->rows || !e->used || !bvec_init(&e->num, vals, cap) || !bvec_init(&e->den, (uint64_t)nb, dcap)) {
        ev_fail(e, NULL, 0); *rc = TR_MATH_MEMORY; return NULL;
    }
    e->bytes = sizeof *e + G * 16 + bvec_bytes(vals, cap) + bvec_bytes((uint64_t)nb, dcap) + basis_bytes_of(b);
    return e;
}
static tr_dyn_evolution *ev_done(tr_dyn_evolution *e, int32_t *error) {
    e->rows_total = e->used_total = 0;
    for (uint64_t g = 0; g < e->ngroups; g++) { e->rows_total += e->rows[g]; e->used_total += e->used[g]; }
    e->magic = EV_MAGIC;
    set_error(error, TR_MATH_OK);
    return e;
}
static int32_t ev_state_rows(const tr_sym_state *s, tr_dyn_evolution *e) {
    return tr_sym_state_get_rows(s, e->rows, e->used, e->ngroups);
}
/* y_g = N x_g (int128 N; x, y: ngroups x nb) */
static int32_t bv_matvec_i128(const tr_math_i128 *N, int32_t nb, const bvec *x, uint64_t G, bvec *y, acc_t *acc) {
    for (uint64_t g = 0; g < G; g++)
        for (int32_t i = 0; i < nb; i++) {
            for (int32_t j = 0; j < nb; j++) {
                i128 c = tr_i128_get(N[(uint64_t)i * nb + j]);
                uint64_t at = g * (uint64_t)nb + (uint64_t)j;
                if (c && x->len[at]) acc_ib(acc, c, x, at);
            }
            if (!acc_store(acc, y, g * (uint64_t)nb + (uint64_t)i)) return TR_MATH_OVERFLOW;
        }
    return TR_MATH_OK;
}
/* y_g = P m_g (P: nb x nb big; m: int128 moments) */
static int32_t bv_matvec_big(const bvec *P, int32_t nb, const tr_math_i128 *m, uint64_t G, bvec *y, acc_t *acc) {
    for (uint64_t g = 0; g < G; g++)
        for (int32_t i = 0; i < nb; i++) {
            for (int32_t j = 0; j < nb; j++) {
                i128 c = tr_i128_get(m[g * (uint64_t)nb + (uint64_t)j]);
                uint64_t at = (uint64_t)i * nb + (uint64_t)j;
                if (c && P->len[at]) acc_ib(acc, c, P, at);
            }
            if (!acc_store(acc, y, g * (uint64_t)nb + (uint64_t)i)) return TR_MATH_OVERFLOW;
        }
    return TR_MATH_OK;
}
/* the moments' scope checks of a truncated evolution: unweighted, the state's scope, extremes inside the box */
static int32_t truncation_premises(const tr_dyn_operator *o, const tr_sym_state_info *si, const tr_scope *scope, uint64_t steps) {
    if (si->weighted) return TR_MATH_UNSUPPORTED;
    tr_scope_info sc;
    if (!scope || tr_scope_get_info(scope, &sc)) return TR_MATH_INVALID;
    if (sc.digest[0] != si->digest[0] || sc.digest[1] != si->digest[1]) return TR_MATH_DOMAIN_ERROR;
    for (int32_t i = 0; i < o->n; i++) {
        if (!((o->bounded >> i) & 1)) continue;
        tr_scope_column_info ci;
        if (tr_scope_column_get(scope, o->geo.col[i], &ci)) return TR_MATH_INVALID;
        if (!ci.nvalid) continue;
        q128 mn = {ci.min, 1}, mx = {ci.max, 1};
        int a, b;
        if (!q_cmp(o->blo[i], mn, &a) || !q_cmp(mx, o->bhi[i], &b)) return TR_MATH_OVERFLOW;
        if (a > 0 || b > 0) return TR_MATH_DOMAIN_ERROR;          /* a selected row lies outside the box */
    }
    if (steps > 1 && !o->invariant) return TR_MATH_DOMAIN_ERROR;   /* rows may leave the box after a step */
    return TR_MATH_OK;
}
/* E_t = W_g e_t / (Deps D^(t-1)): e_1 = epst, e_s = |N| e_(s-1) + D^(s-1) epst */
static int32_t truncation_errors(const tr_dyn_operator *o, uint64_t t, tr_dyn_evolution *e) {
    const int32_t nb = o->nb;
    const uint64_t G = e->ngroups;
    tr_math_budget u;
    unlimited(&u);
    int32_t err = 0, rc = TR_MATH_OK;
    tr_bigint *Deps = tr_bigint_from_i64(1, &u, NULL, &err);
    for (int32_t a = 0; a < nb && Deps; a++) {                   /* Deps = lcm of the denominators */
        const tr_bigint *d = tr_bigq_denominator(o->eps[a]);
        tr_bigint *g = tr_bigint_gcd(Deps, d, &u, NULL, &err), *q = g ? tr_bigint_divexact(d, g, &u, NULL, &err) : NULL;
        tr_bigint *l = q ? tr_bigint_mul(Deps, q, &u, NULL, &err) : NULL;
        tr_bigint_free(g); tr_bigint_free(q); tr_bigint_free(Deps);
        Deps = l;
    }
    if (!Deps) return err ? err : TR_MATH_MEMORY;
    tr_bigint **et = (tr_bigint **)zalloc((uint64_t)nb * sizeof(tr_bigint *));
    uint64_t mbits = 0;
    for (int32_t a = 0; a < nb && et && !rc; a++) {
        tr_bigint *q = tr_bigint_divexact(Deps, tr_bigq_denominator(o->eps[a]), &u, NULL, &err);
        et[a] = q ? tr_bigint_mul(tr_bigq_numerator(o->eps[a]), q, &u, NULL, &err) : NULL;
        tr_bigint_free(q);
        if (!et[a]) rc = err ? err : TR_MATH_MEMORY;
        else if (tr_bigint_bit_length(et[a]) > mbits) mbits = tr_bigint_bit_length(et[a]);
    }
    if (!et) rc = TR_MATH_MEMORY;
    /* |e_s| <= (||N||_inf + D)^(s-1) max epst */
    uint64_t nu = bitlen128((u128)o->den) > o->norm_bits ? bitlen128((u128)o->den) : o->norm_bits;
    uint64_t bits = tr_budget_add(tr_budget_add(tr_budget_mul(t ? t - 1 : 0, nu + 1), mbits), 64 + 8);
    uint64_t cap = limbs_for(bits);
    bvec cur, nxt, ep, Dp;
    memset(&cur, 0, sizeof cur); memset(&nxt, 0, sizeof nxt); memset(&ep, 0, sizeof ep); memset(&Dp, 0, sizeof Dp);
    acc_t acc;
    memset(&acc, 0, sizeof acc);
    if (!rc && (!bvec_init(&cur, (uint64_t)nb, cap) || !bvec_init(&nxt, (uint64_t)nb, cap) || !bvec_init(&ep, (uint64_t)nb, limbs_for(mbits)) ||
                !bvec_init(&Dp, 1, cap) || !acc_init(&acc, 2 * cap + 4) || !bvec_init(&e->err, G * (uint64_t)nb, cap) ||
                !bvec_init(&e->errden, 1, limbs_for(tr_budget_add(tr_budget_mul(t ? t - 1 : 0, bitlen128((u128)o->den)), tr_bigint_bit_length(Deps))))))
        rc = TR_MATH_MEMORY;
    for (int32_t a = 0; a < nb && !rc; a++) if (!bv_from_bigint(&ep, (uint64_t)a, et[a]) || !bv_from_bigint(&cur, (uint64_t)a, et[a])) rc = TR_MATH_OVERFLOW;
    if (!rc) bv_set_i128(&Dp, 0, 1);
    for (uint64_t s = 2; s <= t && !rc; s++) {
        acc_ib(&acc, o->den, &Dp, 0);                             /* Dp = D^(s-1) */
        if (!acc_store(&acc, &Dp, 0)) { rc = TR_MATH_OVERFLOW; break; }
        for (int32_t i = 0; i < nb && !rc; i++) {
            for (int32_t j = 0; j < nb; j++) {
                i128 c = tr_i128_get(o->num[(uint64_t)i * nb + j]);
                if (c && cur.len[j]) acc_ib(&acc, (i128)mag128(c), &cur, (uint64_t)j);
            }
            if (ep.len[i]) acc_bb(&acc, &Dp, 0, &ep, (uint64_t)i);
            if (!acc_store(&acc, &nxt, (uint64_t)i)) rc = TR_MATH_OVERFLOW;
        }
        bvec sw = cur; cur = nxt; nxt = sw;
    }
    for (uint64_t g = 0; g < G && !rc; g++)                      /* W_g e_t: the group's contributing rows */
        for (int32_t i = 0; i < nb && !rc; i++) {
            uint64_t at = g * (uint64_t)nb + (uint64_t)i;
            if (t == 0) { bv_set_i128(&e->err, at, 0); continue; }
            uint64_t w = e->used[g];
            if (w && cur.len[i]) acc_mul(&acc, cur.neg[i], &w, 1, BV(&cur, i), cur.len[i]);
            if (!acc_store(&acc, &e->err, at)) rc = TR_MATH_OVERFLOW;
        }
    if (!rc) {                                                   /* the denominator Deps D^(t-1) */
        bvec dd;
        if (!bvec_init(&dd, 1, limbs_for(tr_bigint_bit_length(Deps)))) rc = TR_MATH_MEMORY;
        else {
            if (!bv_from_bigint(&dd, 0, Deps)) rc = TR_MATH_OVERFLOW;
            if (!rc) {
                if (t <= 1) { if (!bv_copy(&e->errden, 0, &dd, 0)) rc = TR_MATH_OVERFLOW; }
                else { acc_bb(&acc, &dd, 0, &Dp, 0); if (!acc_store(&acc, &e->errden, 0)) rc = TR_MATH_OVERFLOW; }
            }
            bvec_free(&dd);
        }
    }
    if (et) for (int32_t a = 0; a < nb; a++) tr_bigint_free(et[a]);
    free(et);
    tr_bigint_free(Deps);
    bvec_free(&cur); bvec_free(&nxt); bvec_free(&ep); bvec_free(&Dp); acc_free(&acc);
    e->bytes = tr_budget_add(e->bytes, tr_budget_add(bvec_bytes(G * (uint64_t)nb, e->err.cap), bvec_bytes(1, e->errden.cap)));
    return rc;
}
static tr_dyn_evolution *evolution_adopt(const tr_sym_state *s, const tr_dyn_operator *o, uint64_t t,
                                         const tr_gfp *primes, uint64_t np, const uint64_t *res, int32_t backend,
                                         int prove, const tr_math_budget *budget, tr_math_refusal *refusal,
                                         int32_t *error) {
    tr_budget_none(refusal);
    int32_t rc = power_check(o);
    if (rc) return ev_fail(NULL, error, rc);
    tr_sym_state_info si;
    tr_sym_query_view qv;
    if ((rc = state_frame(s, o, &si, &qv))) return ev_fail(NULL, error, rc);
    if (!res || !primes_valid(primes, np, prove) || (backend != TR_CERT_BY_CPU && backend != TR_CERT_BY_DEVICE))
        return ev_fail(NULL, error, TR_MATH_INVALID);
    tr_math_need need;
    uint64_t costs[3];
    if ((rc = need_core(TR_DYN_REQ_EVOLVE, o, s, t, TR_DYN_MODULAR, TR_DYN_NEED_CPU, &need, NULL, costs))) return ev_fail(NULL, error, rc);
    if ((rc = check_need(budget, &need, refusal))) return ev_fail(NULL, error, rc);
    const int32_t nb = o->nb;
    const uint64_t G = qv.ngroups, bits = tr_budget_add(pow_bits(o, t), mom_bits(qv.moments, G * (uint64_t)nb));
    if (np < primes_for(bits)) return ev_fail(NULL, error, TR_MATH_INVALID);
    tr_dyn_evolution *e = ev_new(o->basis, &o->geo, G, nb, 0, limbs_for(bits), limbs_for(den_bits(o, t)), &rc);
    if (!e) return ev_fail(NULL, error, rc);
    if ((rc = ev_state_rows(s, e))) return ev_fail(e, error, rc);
    rc = crt_values(primes, np, res, G * (uint64_t)nb, G * (uint64_t)nb, bits, &e->num, 0);
    if (!rc) rc = den_power(o, t, &e->den, 0);
    for (int32_t a = 1; a < nb && !rc; a++) if (!bv_copy(&e->den, (uint64_t)a, &e->den, 0)) rc = TR_MATH_OVERFLOW;
    if (rc) return ev_fail(e, error, rc);
    e->steps = t; e->method = TR_DYN_MODULAR; e->label = TR_DYN_LABEL_EXACT; e->kind = TR_DYN_EV_OPERATOR;
    e->primes = (int32_t)np; e->checked = 1; e->backend = backend; e->weighted = si.weighted; e->bound_bits = bits;
    e->cost_st = costs[0]; e->cost_sq = costs[1]; e->cost_mod = costs[2];
    return ev_done(e, error);
}
tr_dyn_evolution *tr_dyn_evolution_adopt(const tr_sym_state *s, const tr_dyn_operator *o, uint64_t t,
                                         const tr_gfp *primes, uint64_t np, const uint64_t *res, int32_t backend,
                                         const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error) {
    return evolution_adopt(s, o, t, primes, np, res, backend, 1, budget, refusal, error);
}
/* parametric stepping: v_(s+1)[d] = sum_j N_j v_s[d - j] */
static int32_t param_stepping(const tr_dyn_operator *o, const tr_math_i128 *m, uint64_t G, uint64_t t, bvec *out, acc_t *acc,
                              uint64_t cap) {
    const int32_t nb = o->nb, P = o->pdegree;
    const uint64_t T = t * (uint64_t)P, W = T + 1, e2 = (uint64_t)nb * (uint64_t)nb;
    bvec cur, nxt;
    if (!bvec_init(&cur, G * (uint64_t)nb * W, cap)) return TR_MATH_MEMORY;
    if (!bvec_init(&nxt, G * (uint64_t)nb * W, cap)) { bvec_free(&cur); return TR_MATH_MEMORY; }
    for (uint64_t g = 0; g < G; g++)
        for (int32_t i = 0; i < nb; i++) bv_set_i128(&cur, (g * (uint64_t)nb + (uint64_t)i) * W, tr_i128_get(m[g * (uint64_t)nb + (uint64_t)i]));
    int32_t rc = TR_MATH_OK;
    for (uint64_t s = 0; s < t && !rc; s++) {
        const uint64_t dmax = (s + 1) * (uint64_t)P;
        for (uint64_t g = 0; g < G && !rc; g++)
            for (int32_t i = 0; i < nb && !rc; i++)
                for (uint64_t d = 0; d <= dmax && !rc; d++) {
                    for (int32_t j = 0; j <= P; j++) {
                        if ((uint64_t)j > d || d - (uint64_t)j > s * (uint64_t)P) continue;
                        for (int32_t c = 0; c < nb; c++) {
                            i128 k = tr_i128_get(o->num[(uint64_t)j * e2 + (uint64_t)i * nb + c]);
                            uint64_t at = (g * (uint64_t)nb + (uint64_t)c) * W + d - (uint64_t)j;
                            if (k && cur.len[at]) acc_ib(acc, k, &cur, at);
                        }
                    }
                    if (!acc_store(acc, &nxt, (g * (uint64_t)nb + (uint64_t)i) * W + d)) rc = TR_MATH_OVERFLOW;
                }
        bvec sw = cur; cur = nxt; nxt = sw;
    }
    for (uint64_t x = 0; x < G * (uint64_t)nb * W && !rc; x++) if (!bv_copy(out, x, &cur, x)) rc = TR_MATH_OVERFLOW;
    bvec_free(&cur); bvec_free(&nxt);
    return rc;
}
tr_dyn_evolution *tr_dyn_evolve(const tr_sym_state *s, const tr_dyn_operator *o, uint64_t t, int32_t method,
                                const tr_scope *scope, const tr_math_budget *budget, tr_math_refusal *refusal,
                                int32_t *error) {
    tr_budget_none(refusal);
    if (!op_ok(o)) return ev_fail(NULL, error, TR_MATH_INVALID);
    tr_sym_state_info si;
    tr_sym_query_view qv;
    int32_t rc = state_frame(s, o, &si, &qv);
    if (rc) return ev_fail(NULL, error, rc);
    if (o->kind == TR_DYN_OP_TRUNCATED && (rc = truncation_premises(o, &si, scope, t))) return ev_fail(NULL, error, rc);
    tr_math_need need;
    uint64_t costs[3];
    int32_t m = 0;
    if ((rc = need_core(TR_DYN_REQ_EVOLVE, o, s, t, method, TR_DYN_NEED_CPU, &need, &m, costs))) return ev_fail(NULL, error, rc);
    if ((rc = check_need(budget, &need, refusal))) return ev_fail(NULL, error, rc);
    const int32_t nb = o->nb;
    const uint64_t G = qv.ngroups, mbits = mom_bits(qv.moments, G * (uint64_t)nb);
    const uint64_t bits = tr_budget_add(pow_bits(o, t), mbits), cap = limbs_for(bits);
    if (m == TR_DYN_MODULAR && o->kind == TR_DYN_OP_EXACT) {
        uint64_t np = 0;
        if ((rc = tr_dyn_mod_plan(o, s, t, &np, NULL))) return ev_fail(NULL, error, rc);
        tr_gfp *pr = (tr_gfp *)zalloc(np * sizeof(tr_gfp));
        uint64_t *res = (uint64_t *)zalloc(tr_budget_mul(tr_budget_mul(np, G), (uint64_t)nb) * 8);
        tr_dyn_mod_view v;
        if (!pr || !res) rc = TR_MATH_MEMORY;
        if (!rc) rc = tr_dyn_primes(np, pr);
        if (!rc) rc = tr_dyn_mod_bind(o, t, pr, np, &v);
        tr_math_budget u;
        unlimited(&u);
        if (!rc) rc = tr_dyn_evolvemod_view_eval(&v, qv.moments, G, res, &u, NULL);
        tr_dyn_evolution *e = rc ? NULL : evolution_adopt(s, o, t, pr, np, res, TR_CERT_BY_CPU, 0, budget, refusal, &rc);
        free(pr); free(res);
        if (!e) return ev_fail(NULL, error, rc);
        set_error(error, TR_MATH_OK);
        return e;
    }
    tr_dyn_evolution *e = ev_new(o->basis, &o->geo, G, nb, (int32_t)(t * (uint64_t)o->pdegree), cap, limbs_for(den_bits(o, t)), &rc);
    if (!e) return ev_fail(NULL, error, rc);
    if ((rc = ev_state_rows(s, e))) return ev_fail(e, error, rc);
    acc_t acc;
    if (!acc_init(&acc, 2 * cap + 4)) return ev_fail(e, error, TR_MATH_MEMORY);
    if (o->pdegree) rc = param_stepping(o, qv.moments, G, t, &e->num, &acc, cap);
    else if (m == TR_DYN_SQUARING) {
        bvec P;
        if (!bvec_init(&P, (uint64_t)nb * nb, limbs_for(pow_bits(o, t)))) rc = TR_MATH_MEMORY;
        uint64_t prods = 0;
        if (!rc) rc = power_squaring(o, t, &P, &prods);
        if (!rc) rc = bv_matvec_big(&P, nb, qv.moments, G, &e->num, &acc);
        bvec_free(&P);
    } else {                                                     /* stepping */
        bvec cur, nxt;
        if (!bvec_init(&cur, G * (uint64_t)nb, cap)) rc = TR_MATH_MEMORY;
        else if (!bvec_init(&nxt, G * (uint64_t)nb, cap)) { bvec_free(&cur); rc = TR_MATH_MEMORY; }
        if (!rc) {
            for (uint64_t x = 0; x < G * (uint64_t)nb; x++) bv_set_i128(&cur, x, tr_i128_get(qv.moments[x]));
            for (uint64_t k = 0; k < t && !rc; k++) {
                rc = bv_matvec_i128(o->num, nb, &cur, G, &nxt, &acc);
                bvec sw = cur; cur = nxt; nxt = sw;
            }
            for (uint64_t x = 0; x < G * (uint64_t)nb && !rc; x++) if (!bv_copy(&e->num, x, &cur, x)) rc = TR_MATH_OVERFLOW;
            bvec_free(&cur); bvec_free(&nxt);
        }
        m = TR_DYN_STEPPING;
    }
    acc_free(&acc);
    if (!rc) rc = den_power(o, t, &e->den, 0);
    for (int32_t a = 1; a < nb && !rc; a++) if (!bv_copy(&e->den, (uint64_t)a, &e->den, 0)) rc = TR_MATH_OVERFLOW;
    if (!rc && o->kind == TR_DYN_OP_TRUNCATED) rc = truncation_errors(o, t, e);
    if (rc) return ev_fail(e, error, rc);
    e->steps = t; e->method = m; e->label = o->label; e->kind = TR_DYN_EV_OPERATOR; e->invariant = o->invariant;
    e->backend = TR_CERT_BY_CPU; e->weighted = si.weighted; e->bound_bits = bits;
    e->cost_st = costs[0]; e->cost_sq = costs[1]; e->cost_mod = costs[2];
    return ev_done(e, error);
}
tr_dyn_evolution *tr_dyn_evolution_at(const tr_dyn_evolution *s, tr_num_rational theta, const tr_math_budget *budget,
                                      tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    q128 th;
    if (!ev_ok(s) || !q_from(theta, &th)) return ev_fail(NULL, error, TR_MATH_INVALID);
    const uint64_t T = (uint64_t)s->pdegree, W = T + 1, G = s->ngroups, nb = (uint64_t)s->nb;
    uint64_t mx = 0;
    for (uint64_t x = 0; x < s->num.count; x++) if (s->num.len[x] > mx) mx = s->num.len[x];
    uint64_t pbits = tr_budget_mul(T, bitlen128(mag128(th.n)) > bitlen128((u128)th.d) ? bitlen128(mag128(th.n)) : bitlen128((u128)th.d));
    uint64_t cap = tr_budget_add(tr_budget_add(mx, limbs_for(pbits)), limbs_for(bitlen64(W)) + 1);
    uint64_t dcap = tr_budget_add(s->den.cap, limbs_for(tr_budget_mul(T, bitlen128((u128)th.d))));
    tr_math_need need;
    need_zero(&need);
    need.need[TR_BUDGET_OUTPUT] = G * nb;
    need.need[TR_BUDGET_DEGREE] = T;
    need.need[TR_BUDGET_BYTES] = tr_budget_add(tr_budget_add(bvec_bytes(G * nb, cap), bvec_bytes(nb, dcap)),
                                               tr_budget_mul(tr_budget_mul(W, 2), tr_budget_mul(limbs_for(pbits), 8)));
    int32_t rc = check_need(budget, &need, refusal);
    if (rc) return ev_fail(NULL, error, rc);
    tr_dyn_evolution *e = ev_new(s->basis, &s->geo, G, (int32_t)nb, 0, cap, dcap, &rc);
    if (!e) return ev_fail(NULL, error, rc);
    memcpy(e->rows, s->rows, G * 8); memcpy(e->used, s->used, G * 8);
    tr_math_budget u;
    unlimited(&u);
    int32_t err = 0;
    /* value = sum_d a^d b^(T-d) v[d] / (b^T den) */
    tr_bigint *a = tr_bigint_from_i128(tr_i128_set(th.n), &u, NULL, &err), *b = a ? tr_bigint_from_i128(tr_i128_set(th.d), &u, NULL, &err) : NULL;
    tr_bigint **f = (tr_bigint **)zalloc(W * sizeof(tr_bigint *));
    if (!b || !f) rc = err ? err : TR_MATH_MEMORY;
    for (uint64_t d = 0; d < W && !rc; d++) {
        tr_bigint *ad = tr_bigint_pow(a, d, &u, NULL, &err), *bd = ad ? tr_bigint_pow(b, T - d, &u, NULL, &err) : NULL;
        f[d] = bd ? tr_bigint_mul(ad, bd, &u, NULL, &err) : NULL;
        tr_bigint_free(ad); tr_bigint_free(bd);
        if (!f[d]) rc = err ? err : TR_MATH_MEMORY;
    }
    bvec fv;
    acc_t acc;
    memset(&fv, 0, sizeof fv); memset(&acc, 0, sizeof acc);
    if (!rc && (!bvec_init(&fv, W, limbs_for(pbits) + 1) || !acc_init(&acc, 2 * cap + 4))) rc = TR_MATH_MEMORY;
    for (uint64_t d = 0; d < W && !rc; d++) if (!bv_from_bigint(&fv, d, f[d])) rc = TR_MATH_OVERFLOW;
    for (uint64_t x = 0; x < G * nb && !rc; x++) {
        for (uint64_t d = 0; d < W; d++) if (s->num.len[x * W + d] && fv.len[d]) acc_bb(&acc, &s->num, x * W + d, &fv, d);
        if (!acc_store(&acc, &e->num, x)) rc = TR_MATH_OVERFLOW;
    }
    for (uint64_t i = 0; i < nb && !rc; i++) {                   /* b^T den */
        tr_bigint *bt = tr_bigint_pow(b, T, &u, NULL, &err), *dn = bt ? bv_to_bigint(&s->den, i, &err) : NULL;
        tr_bigint *pr = dn ? tr_bigint_mul(bt, dn, &u, NULL, &err) : NULL;
        if (!pr || !bv_from_bigint(&e->den, i, pr)) rc = err ? err : TR_MATH_OVERFLOW;
        tr_bigint_free(bt); tr_bigint_free(dn); tr_bigint_free(pr);
    }
    if (f) for (uint64_t d = 0; d < W; d++) tr_bigint_free(f[d]);
    free(f); tr_bigint_free(a); tr_bigint_free(b); bvec_free(&fv); acc_free(&acc);
    if (rc) return ev_fail(e, error, rc);
    e->steps = s->steps; e->method = s->method; e->label = s->label; e->kind = s->kind; e->backend = s->backend;
    e->weighted = s->weighted; e->bound_bits = s->bound_bits + pbits;
    return ev_done(e, error);
}

/* ------------------------------------------------------------------ explicit row evolution (the reference) */
typedef struct {
    uint32_t reach;                        /* the coordinates evolved (read by the basis observables' evolution) */
    int32_t d;                             /* the common degree d >= 1 */
    i128 L;                                /* lcm of the reach set's q_i */
    uint64_t xbits0, coefbits;             /* initial bound; bitlen(max_i (L / q_i) sum_k |c_ik|) */
} rowplan;
static int32_t row_plan(const tr_scope *scope, const tr_dyn_map *m, const tr_sym_basis *b, int32_t weight, rowplan *rp) {
    tr_sym_basis_info bi;
    tr_scope_info sc;
    if (!map_ok(m) || tr_sym_basis_get_info(b, &bi) || bi.n != m->n || !scope || tr_scope_get_info(scope, &sc))
        return TR_MATH_INVALID;
    if (m->parametric) return TR_MATH_UNSUPPORTED;
    int32_t rc = tr_sym_basis_check(b, &m->geo);
    if (rc) return rc;
    if ((!m->geo.digest[0] && !m->geo.digest[1]) || sc.digest[0] != m->geo.digest[0] || sc.digest[1] != m->geo.digest[1])
        return TR_MATH_DOMAIN_ERROR;                             /* the map's geometry belongs to another scope */
    if (weight < -1 || weight >= sc.ncols) return TR_MATH_INVALID;
    memset(rp, 0, sizeof *rp);
    rp->reach = reach_set(m, bi.coords);
    rp->d = 1; rp->L = 1;
    for (int32_t i = 0; i < m->n; i++) {
        if (!((rp->reach >> i) & 1)) continue;
        if (!lcm128(rp->L, m->q[i], &rp->L)) return TR_MATH_OVERFLOW;
        for (int32_t t = m->first[i]; t < m->first[i + 1]; t++) if (mterm_degree(&m->terms[t]) > rp->d) rp->d = mterm_degree(&m->terms[t]);
        tr_scope_column_info ci;
        if (tr_scope_column_get(scope, m->geo.col[i], &ci)) return TR_MATH_INVALID;
        if (ci.nvalid) {
            uint64_t a = bitlen128(mag128(ci.min)), c = bitlen128(mag128(ci.max));
            if (a > rp->xbits0) rp->xbits0 = a;
            if (c > rp->xbits0) rp->xbits0 = c;
        }
    }
    for (int32_t i = 0; i < m->n; i++) {
        if (!((rp->reach >> i) & 1)) continue;
        u128 s = 0;
        uint64_t carry = 0;
        for (int32_t t = m->first[i]; t < m->first[i + 1]; t++) {
            u128 v = mag128(m->terms[t].c), x = s + v;
            if (x < s) carry++;
            s = x;
        }
        uint64_t bits = (carry ? 128 + bitlen64(carry) : bitlen128(s)) + bitlen128((u128)(rp->L / m->q[i]));
        if (bits > rp->coefbits) rp->coefbits = bits;
    }
    return TR_MATH_OK;
}
/* bits of X_t and Q_t after t steps (a priori: |X'| <= C max(B, Q)^d, Q' = L Q^d) */
static void row_bounds(const rowplan *rp, uint64_t t, uint64_t *xb, uint64_t *qb) {
    uint64_t x = rp->xbits0, q = 0, lb = bitlen128((u128)rp->L);
    for (uint64_t s = 0; s < t; s++) {
        uint64_t mx = x > q ? x : q;
        x = tr_budget_add(tr_budget_add(rp->coefbits, tr_budget_mul((uint64_t)rp->d, mx)), 1);
        q = tr_budget_add(lb, tr_budget_mul((uint64_t)rp->d, q));
        if (x == TR_BUDGET_UNKNOWN || q == TR_BUDGET_UNKNOWN) { x = q = TR_BUDGET_UNKNOWN; break; }
    }
    *xb = x; *qb = q;
}
int32_t tr_dyn_rows_need(const tr_scope *scope, const tr_dyn_map *m, const tr_sym_basis *b, uint64_t t, int32_t weight,
                         tr_math_need *need) {
    if (!need) return TR_MATH_INVALID;
    rowplan rp;
    int32_t rc = row_plan(scope, m, b, weight, &rp);
    if (rc) return rc;
    tr_sym_basis_info bi;
    tr_scope_info sc;
    tr_sym_basis_get_info(b, &bi);
    tr_scope_get_info(scope, &sc);
    uint64_t xb, qb;
    row_bounds(&rp, t, &xb, &qb);
    need_zero(need);
    const uint64_t n = (uint64_t)m->n, G = sc.ngroups, nb = (uint64_t)bi.nb;
    uint64_t mb = tr_budget_add(tr_budget_mul((uint64_t)bi.degree, xb), 128);
    need->need[TR_BUDGET_VARIABLES] = n;
    need->need[TR_BUDGET_DEGREE] = (uint64_t)bi.degree;
    need->need[TR_BUDGET_SUPPORT] = nb;
    need->need[TR_BUDGET_ITERATIONS] = t;
    need->need[TR_BUDGET_OUTPUT] = tr_budget_mul(G, nb);
    need->need[TR_BUDGET_STATES] = tr_budget_mul(G, nb);
    uint64_t bytes = tr_budget_mul(bvec_bytes(tr_budget_mul(sc.selected, n), limbs_for(xb)), 2);
    bytes = tr_budget_add(bytes, bvec_bytes(tr_budget_mul(G, nb), limbs_for(mb)));
    bytes = tr_budget_add(bytes, tr_budget_mul(bvec_bytes(nb, limbs_for(tr_budget_mul((uint64_t)bi.degree, qb))), 2));
    bytes = tr_budget_add(bytes, tr_budget_mul(tr_budget_mul(n, (uint64_t)rp.d + 1), tr_budget_mul(limbs_for(tr_budget_mul((uint64_t)rp.d, xb > qb ? xb : qb)), 16)));
    bytes = tr_budget_add(bytes, tr_budget_mul(sc.selected, 8));
    need->need[TR_BUDGET_BYTES] = bytes;
    return TR_MATH_OK;
}
tr_dyn_evolution *tr_dyn_rows_evolve(const tr_scope *scope, const tr_dyn_map *m, const tr_sym_basis *b, uint64_t t,
                                     int32_t weight, const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    tr_math_need need;
    int32_t rc = tr_dyn_rows_need(scope, m, b, t, weight, &need);
    if (rc) return ev_fail(NULL, error, rc);
    if ((rc = check_need(budget, &need, refusal))) return ev_fail(NULL, error, rc);
    rowplan rp;
    row_plan(scope, m, b, weight, &rp);
    tr_sym_basis_info bi;
    tr_sym_basis_get_info(b, &bi);
    tr_scope_view v;
    if (tr_scope_get_view(scope, &v)) return ev_fail(NULL, error, TR_MATH_INVALID);
    const int32_t n = m->n, nb = bi.nb;
    const uint64_t G = v.ngroups;
    uint64_t xb, qb;
    row_bounds(&rp, t, &xb, &qb);
    const uint64_t xcap = limbs_for(xb), mcap = limbs_for(tr_budget_add(tr_budget_mul((uint64_t)bi.degree, xb), 128));
    tr_dyn_evolution *e = ev_new(b, &m->geo, G, nb, 0, mcap, limbs_for(tr_budget_add(tr_budget_mul((uint64_t)bi.degree, qb), 1)), &rc);
    if (!e) return ev_fail(NULL, error, rc);
    uint8_t *ex = (uint8_t *)zalloc((uint64_t)nb * (uint64_t)n);
    uint64_t *rowsel = (uint64_t *)zalloc((v.selected ? v.selected : 1) * 8);
    if (!ex || !rowsel) { free(ex); free(rowsel); return ev_fail(e, error, TR_MATH_MEMORY); }
    tr_sym_basis_get_exponents(b, ex, (uint64_t)nb * (uint64_t)n);
    uint64_t nr = 0;                                             /* the rows: selected, reach (and weight) present */
    for (uint64_t r = 0; r < v.rows; r++) {
        if (!tr_scope_is_selected(&v, r)) continue;
        uint32_t g = tr_scope_group(&v, r);
        e->rows[g]++;
        int ok = weight < 0 || tr_scope_present(&v, weight, r);
        for (int32_t i = 0; i < n && ok; i++) if (((rp.reach >> i) & 1) && !tr_scope_present(&v, m->geo.col[i], r)) ok = 0;
        if (!ok) continue;
        e->used[g]++;
        rowsel[nr++] = r;
    }
    bvec X, Y;
    acc_t acc;
    memset(&X, 0, sizeof X); memset(&Y, 0, sizeof Y); memset(&acc, 0, sizeof acc);
    if (!bvec_init(&X, (nr ? nr : 1) * (uint64_t)n, xcap) || !bvec_init(&Y, (nr ? nr : 1) * (uint64_t)n, xcap) ||
        !acc_init(&acc, 2 * (xcap > mcap ? xcap : mcap) + 8)) rc = TR_MATH_MEMORY;
    for (uint64_t k = 0; k < nr && !rc; k++)
        for (int32_t i = 0; i < n; i++)
            if ((rp.reach >> i) & 1) bv_set_i128(&X, k * (uint64_t)n + (uint64_t)i, (i128)tr_scope_value(&v, m->geo.col[i], rowsel[k]));
    tr_math_budget u;
    unlimited(&u);
    int32_t err = 0;
    tr_bigint *Q = tr_bigint_from_i64(1, &u, NULL, &err);
    if (!Q) rc = err ? err : TR_MATH_MEMORY;
    const int affine = rp.d == 1;
    i128 scale[NC];
    for (int32_t i = 0; i < n; i++) scale[i] = ((rp.reach >> i) & 1) ? rp.L / m->q[i] : 1;
    for (uint64_t s = 0; s < t && !rc; s++) {
        if (affine) {                                            /* X'_i = sum_j (L/q_i) a_ij X_j + (L/q_i) b_i Q */
            bvec cb;
            if (!bvec_init(&cb, (uint64_t)n, limbs_for(tr_budget_add(tr_bigint_bit_length(Q), 256)))) { rc = TR_MATH_MEMORY; break; }
            bvec qv;
            if (!bvec_init(&qv, 1, limbs_for(tr_bigint_bit_length(Q)))) { bvec_free(&cb); rc = TR_MATH_MEMORY; break; }
            if (!bv_from_bigint(&qv, 0, Q)) rc = TR_MATH_OVERFLOW;
            for (int32_t i = 0; i < n && !rc; i++) {
                if (!((rp.reach >> i) & 1)) continue;
                for (int32_t k = m->first[i]; k < m->first[i + 1]; k++) {
                    if (mterm_degree(&m->terms[k])) continue;
                    i128 c;
                    if (!mul128(m->terms[k].c, scale[i], &c) || !in_range(c)) { rc = TR_MATH_OVERFLOW; break; }
                    acc_ib(&acc, c, &qv, 0);
                }
                if (!rc && !acc_store(&acc, &cb, (uint64_t)i)) rc = TR_MATH_OVERFLOW;
            }
            for (uint64_t r = 0; r < nr && !rc; r++)
                for (int32_t i = 0; i < n && !rc; i++) {
                    if (!((rp.reach >> i) & 1)) continue;
                    for (int32_t k = m->first[i]; k < m->first[i + 1]; k++) {
                        const mterm *x = &m->terms[k];
                        int32_t j = -1;
                        for (int32_t c = 0; c < n; c++) if (x->e[c]) { j = c; break; }
                        if (j < 0) continue;
                        i128 c;
                        if (!mul128(x->c, scale[i], &c) || !in_range(c)) { rc = TR_MATH_OVERFLOW; break; }
                        uint64_t at = r * (uint64_t)n + (uint64_t)j;
                        if (X.len[at]) acc_ib(&acc, c, &X, at);
                    }
                    if (!rc && cb.len[i]) { uint64_t one = 1; acc_mul(&acc, cb.neg[i], &one, 1, BV(&cb, i), cb.len[i]); }
                    if (!rc && !acc_store(&acc, &Y, r * (uint64_t)n + (uint64_t)i)) rc = TR_MATH_OVERFLOW;
                }
            bvec_free(&cb); bvec_free(&qv);
        } else {                                                 /* X'_i = (L/q_i) sum_k c_k X^e_k Q^(d - |e_k|) */
            const int32_t d = rp.d;
            tr_bigint *Qp[TR_DYN_MAX_DEGREE + 1];
            memset(Qp, 0, sizeof Qp);
            for (int32_t k = 0; k <= d && !rc; k++) { Qp[k] = tr_bigint_pow(Q, (uint64_t)k, &u, NULL, &err); if (!Qp[k]) rc = err ? err : TR_MATH_MEMORY; }
            for (uint64_t r = 0; r < nr && !rc; r++) {
                tr_bigint *Xp[NC][TR_DYN_MAX_DEGREE + 1];
                memset(Xp, 0, sizeof Xp);
                for (int32_t j = 0; j < n && !rc; j++) {
                    if (!((rp.reach >> j) & 1)) continue;
                    Xp[j][1] = bv_to_bigint(&X, r * (uint64_t)n + (uint64_t)j, &err);
                    if (!Xp[j][1]) { rc = err ? err : TR_MATH_MEMORY; break; }
                    for (int32_t k = 2; k <= d && !rc; k++) { Xp[j][k] = tr_bigint_mul(Xp[j][k - 1], Xp[j][1], &u, NULL, &err); if (!Xp[j][k]) rc = err ? err : TR_MATH_MEMORY; }
                }
                for (int32_t i = 0; i < n && !rc; i++) {
                    if (!((rp.reach >> i) & 1)) continue;
                    tr_bigint *sum = tr_bigint_from_i64(0, &u, NULL, &err);
                    for (int32_t k = m->first[i]; k < m->first[i + 1] && sum && !rc; k++) {
                        const mterm *x = &m->terms[k];
                        i128 c;
                        if (!mul128(x->c, scale[i], &c) || !in_range(c)) { rc = TR_MATH_OVERFLOW; break; }
                        tr_bigint *term = tr_bigint_from_i128(tr_i128_set(c), &u, NULL, &err);
                        for (int32_t j = 0; j < n && term; j++) {
                            if (!x->e[j]) continue;
                            tr_bigint *nt = tr_bigint_mul(term, Xp[j][x->e[j]], &u, NULL, &err);
                            tr_bigint_free(term); term = nt;
                        }
                        tr_bigint *nt = term ? tr_bigint_mul(term, Qp[d - mterm_degree(x)], &u, NULL, &err) : NULL;
                        tr_bigint_free(term);
                        tr_bigint *ns = nt ? tr_bigint_add(sum, nt, &u, NULL, &err) : NULL;
                        tr_bigint_free(nt); tr_bigint_free(sum);
                        sum = ns;
                    }
                    if (!sum && !rc) rc = err ? err : TR_MATH_MEMORY;
                    if (!rc && !bv_from_bigint(&Y, r * (uint64_t)n + (uint64_t)i, sum)) rc = TR_MATH_OVERFLOW;
                    tr_bigint_free(sum);
                }
                for (int32_t j = 0; j < n; j++) for (int32_t k = 0; k <= TR_DYN_MAX_DEGREE; k++) tr_bigint_free(Xp[j][k]);
            }
            for (int32_t k = 0; k <= d; k++) tr_bigint_free(Qp[k]);
        }
        if (rc) break;
        bvec sw = X; X = Y; Y = sw;                              /* Q_(s+1) = L Q_s^d */
        tr_bigint *Lb = tr_bigint_from_i128(tr_i128_set(rp.L), &u, NULL, &err);
        tr_bigint *Qd = Lb ? tr_bigint_pow(Q, (uint64_t)rp.d, &u, NULL, &err) : NULL;
        tr_bigint *Qn = Qd ? tr_bigint_mul(Lb, Qd, &u, NULL, &err) : NULL;
        tr_bigint_free(Lb); tr_bigint_free(Qd);
        if (!Qn) { rc = err ? err : TR_MATH_MEMORY; break; }
        tr_bigint_free(Q); Q = Qn;
    }
    /* the moments: sum_r w_r X_r^a over Q^|a| */
    tr_bigint **S = rc ? NULL : (tr_bigint **)zalloc(G * (uint64_t)nb * sizeof(tr_bigint *));
    if (!rc && !S) rc = TR_MATH_MEMORY;
    for (uint64_t x = 0; x < G * (uint64_t)nb && !rc; x++) { S[x] = tr_bigint_from_i64(0, &u, NULL, &err); if (!S[x]) rc = err ? err : TR_MATH_MEMORY; }
    for (uint64_t r = 0; r < nr && !rc; r++) {
        uint32_t g = tr_scope_group(&v, rowsel[r]);
        tr_bigint *Xp[NC][TR_SYM_MAX_DEGREE + 1];
        memset(Xp, 0, sizeof Xp);
        for (int32_t j = 0; j < n && !rc; j++) {
            if (!((rp.reach >> j) & 1)) continue;
            Xp[j][0] = tr_bigint_from_i64(1, &u, NULL, &err);
            for (int32_t k = 1; k <= bi.degree && Xp[j][k - 1]; k++) {
                tr_bigint *xj = bv_to_bigint(&X, r * (uint64_t)n + (uint64_t)j, &err);
                Xp[j][k] = xj ? tr_bigint_mul(Xp[j][k - 1], xj, &u, NULL, &err) : NULL;
                tr_bigint_free(xj);
            }
            if (!Xp[j][bi.degree]) rc = err ? err : TR_MATH_MEMORY;
        }
        tr_bigint *w = NULL;
        if (!rc) w = tr_bigint_from_i64(weight >= 0 ? tr_scope_value(&v, weight, rowsel[r]) : 1, &u, NULL, &err);
        if (!w && !rc) rc = err ? err : TR_MATH_MEMORY;
        for (int32_t a = 0; a < nb && !rc; a++) {
            const uint8_t *ea = ex + (uint64_t)a * n;
            tr_bigint *p = tr_bigint_copy(w, &u, NULL, &err);
            for (int32_t j = 0; j < n && p; j++) {
                if (!ea[j]) continue;
                tr_bigint *np2 = tr_bigint_mul(p, Xp[j][ea[j]], &u, NULL, &err);
                tr_bigint_free(p); p = np2;
            }
            tr_bigint *ns = p ? tr_bigint_add(S[g * (uint64_t)nb + a], p, &u, NULL, &err) : NULL;
            tr_bigint_free(p);
            if (!ns) { rc = err ? err : TR_MATH_MEMORY; break; }
            tr_bigint_free(S[g * (uint64_t)nb + a]); S[g * (uint64_t)nb + a] = ns;
        }
        tr_bigint_free(w);
        for (int32_t j = 0; j < n; j++) for (int32_t k = 0; k <= TR_SYM_MAX_DEGREE; k++) tr_bigint_free(Xp[j][k]);
    }
    for (uint64_t x = 0; x < G * (uint64_t)nb && !rc; x++) if (!bv_from_bigint(&e->num, x, S[x])) rc = TR_MATH_OVERFLOW;
    for (int32_t a = 0; a < nb && !rc; a++) {                    /* Q_t^|a| */
        int32_t deg = 0;
        for (int32_t j = 0; j < n; j++) deg += ex[(uint64_t)a * n + j];
        tr_bigint *qa = tr_bigint_pow(Q, (uint64_t)deg, &u, NULL, &err);
        if (!qa || !bv_from_bigint(&e->den, (uint64_t)a, qa)) rc = err ? err : TR_MATH_OVERFLOW;
        tr_bigint_free(qa);
    }
    if (S) for (uint64_t x = 0; x < G * (uint64_t)nb; x++) tr_bigint_free(S[x]);
    free(S);
    tr_bigint_free(Q);
    bvec_free(&X); bvec_free(&Y); acc_free(&acc); free(ex); free(rowsel);
    if (rc) return ev_fail(e, error, rc);
    e->steps = t; e->method = TR_DYN_STEPPING; e->label = TR_DYN_LABEL_EXACT; e->kind = TR_DYN_EV_ROWS;
    e->backend = TR_CERT_BY_CPU; e->weighted = weight >= 0; e->bound_bits = xb;
    return ev_done(e, error);
}
int32_t tr_dyn_evolution_get_info(const tr_dyn_evolution *e, tr_dyn_evolve_info *out) {
    if (!ev_ok(e) || !out) return TR_MATH_INVALID;
    memset(out, 0, sizeof *out);
    out->ngroups = e->ngroups; out->steps = e->steps; out->nb = e->nb; out->method = e->method; out->label = e->label;
    out->kind = e->kind; out->primes = e->primes; out->invariant = e->invariant; out->pdegree = e->pdegree;
    out->checked = e->checked; out->backend = e->backend; out->weighted = e->weighted; out->bound_bits = e->bound_bits;
    for (uint64_t x = 0; x < e->num.count; x++) { uint64_t b = bv_bitlen(&e->num, x); if (b > out->max_bits) out->max_bits = b; }
    out->bytes = e->bytes; out->rows = e->rows_total; out->used = e->used_total;
    out->cost_stepping = e->cost_st; out->cost_squaring = e->cost_sq; out->cost_modular = e->cost_mod;
    return TR_MATH_OK;
}
int32_t tr_dyn_evolution_get_rows(const tr_dyn_evolution *e, uint64_t *rows, uint64_t *used, uint64_t capacity) {
    if (!ev_ok(e) || !rows || !used || capacity < e->ngroups) return TR_MATH_INVALID;
    memcpy(rows, e->rows, (size_t)e->ngroups * 8); memcpy(used, e->used, (size_t)e->ngroups * 8);
    return TR_MATH_OK;
}
int32_t tr_dyn_evolution_authorize_exact(const tr_dyn_evolution *e, tr_cert_decision *d) {
    if (!ev_ok(e) || !d) return TR_MATH_INVALID;
    decision_base(d, e->label == TR_DYN_LABEL_EXACT);
    return e->label == TR_DYN_LABEL_EXACT ? TR_MATH_OK : TR_MATH_DOMAIN_ERROR;
}
void tr_dyn_evolution_free(tr_dyn_evolution *e) {
    if (!ev_ok(e)) return;
    e->magic = 0;
    ev_fail(e, NULL, 0);
}

/* ================================================================== characteristic polynomials and recurrences */
struct tr_dyn_recurrence_s {
    uint32_t magic;
    int32_t kind, degree, nb, primes, sequence, checked;
    uint64_t terms, bound_bits, bytes;
    tr_bigint **r;                         /* d integer coefficients: u_(t+d) = sum_k r_k u_(t+k) */
    tr_bigint **u;                         /* d initial terms (sequence) */
    tr_bigint *D;                          /* the operator's denominator: s_t = u_t / D^t */
    tr_bigq **coef;                        /* charpoly: nb + 1 coefficients of chi_K; recurrence: d of s */
    tr_bigq **init;                        /* d: s_0 .. s_(d-1) */
    int32_t ncoef;
};
static int rec_ok(const tr_dyn_recurrence *r) { return r && r->magic == REC_MAGIC; }
static void rec_release(tr_dyn_recurrence *r) {
    if (!r) return;
    if (r->r) for (int32_t k = 0; k < r->degree; k++) tr_bigint_free(r->r[k]);
    if (r->u) for (int32_t k = 0; k < r->degree; k++) tr_bigint_free(r->u[k]);
    if (r->coef) for (int32_t k = 0; k < r->ncoef; k++) tr_bigq_free(r->coef[k]);
    if (r->init) for (int32_t k = 0; k < r->degree; k++) tr_bigq_free(r->init[k]);
    free(r->r); free(r->u); free(r->coef); free(r->init); tr_bigint_free(r->D);
    free(r);
}
static tr_dyn_recurrence *rec_fail(tr_dyn_recurrence *r, int32_t *error, int32_t code) {
    rec_release(r);
    set_error(error, code);
    return NULL;
}
/* the characteristic polynomial of an n x n residue matrix H (destroyed) mod p: c[0..n], c[n] = 1 (Hessenberg) */
static int32_t charpoly_mod(const tr_gfp *f, uint64_t *H, int32_t n, uint64_t *c, uint64_t *poly) {
#define HH(i, j) H[(uint64_t)(i) * (uint64_t)n + (uint64_t)(j)]
    for (int32_t m = 1; m < n - 1; m++) {                        /* reduce column m - 1 below the subdiagonal */
        int32_t i = m;
        while (i < n && !HH(i, m - 1)) i++;
        if (i == n) continue;
        if (i != m) {
            for (int32_t j = 0; j < n; j++) { uint64_t t = HH(i, j); HH(i, j) = HH(m, j); HH(m, j) = t; }
            for (int32_t j = 0; j < n; j++) { uint64_t t = HH(j, i); HH(j, i) = HH(j, m); HH(j, m) = t; }
        }
        uint64_t inv = tr_gfp_inv_hd(f, HH(m, m - 1));
        for (int32_t r = m + 1; r < n; r++) {
            uint64_t u = tr_gfp_mul_hd(f, HH(r, m - 1), inv);
            if (!u) continue;
            for (int32_t j = 0; j < n; j++) HH(r, j) = tr_gfp_sub_hd(f, HH(r, j), tr_gfp_mul_hd(f, u, HH(m, j)));
            for (int32_t j = 0; j < n; j++) HH(j, m) = tr_gfp_add_hd(f, HH(j, m), tr_gfp_mul_hd(f, u, HH(j, r)));
        }
    }
    /* p_0 = 1; p_m = (x - h_mm) p_(m-1) - sum_i (prod) h_(m-i, m) p_(m-i-1) (1-based, Cohen 2.2.9); poly: (n+1)^2 */
#define PP(m, k) poly[(uint64_t)(m) * (uint64_t)(n + 1) + (uint64_t)(k)]
    for (uint64_t x = 0; x < (uint64_t)(n + 1) * (uint64_t)(n + 1); x++) poly[x] = 0;
    PP(0, 0) = 1 % f->p;
    for (int32_t m = 1; m <= n; m++) {
        uint64_t h = HH(m - 1, m - 1);
        for (int32_t k = 0; k <= m; k++) {                       /* (x - h) p_(m-1) */
            uint64_t a = k ? PP(m - 1, k - 1) : 0, b = k < m ? PP(m - 1, k) : 0;
            PP(m, k) = tr_gfp_sub_hd(f, a, tr_gfp_mul_hd(f, h, b));
        }
        uint64_t t = 1 % f->p;
        for (int32_t i = 1; i <= m - 1; i++) {
            t = tr_gfp_mul_hd(f, t, HH(m - i, m - i - 1));
            uint64_t s = tr_gfp_mul_hd(f, t, HH(m - i - 1, m - 1));
            if (!s) continue;
            for (int32_t k = 0; k <= m - i - 1; k++) PP(m, k) = tr_gfp_sub_hd(f, PP(m, k), tr_gfp_mul_hd(f, s, PP(m - i - 1, k)));
        }
    }
    for (int32_t k = 0; k <= n; k++) c[k] = PP(n, k);
#undef PP
#undef HH
    return TR_MATH_OK;
}
static tr_bigq *bq_of_bigints(const tr_bigint *n, const tr_bigint *d, int32_t *err) {
    tr_math_budget u;
    unlimited(&u);
    return tr_bigq_from_bigints(n, d, &u, NULL, err);
}
/* chi_N's integer coefficients a_0..a_nb (a_nb = 1) by the modular path */
static int32_t charpoly_int(const tr_dyn_operator *o, tr_bigint **a, int32_t *nprimes, uint64_t *bound) {
    const int32_t n = o->nb;
    const uint64_t e2 = (uint64_t)n * (uint64_t)n;
    uint64_t bits = tr_budget_add(tr_budget_mul((uint64_t)n, o->norm_bits), (uint64_t)n + 1);  /* C(n,k) R^k */
    uint64_t np = primes_for(bits);
    if (np > TR_DYN_MAX_PRIMES) return TR_MATH_UNSUPPORTED;
    tr_gfp *pr = (tr_gfp *)zalloc(np * sizeof(tr_gfp));
    uint64_t *res = (uint64_t *)zalloc(np * (uint64_t)(n + 1) * 8), *H = (uint64_t *)zalloc(e2 * 8);
    uint64_t *poly = (uint64_t *)zalloc((uint64_t)(n + 1) * (uint64_t)(n + 1) * 8);
    int32_t rc = !pr || !res || !H || !poly ? TR_MATH_MEMORY : tr_dyn_primes(np, pr);
    for (uint64_t q = 0; q < np && !rc; q++) {
        for (uint64_t x = 0; x < e2; x++) H[x] = tr_dyn_reduce_i128(&pr[q], o->num[x]);
        rc = charpoly_mod(&pr[q], H, n, res + q * (uint64_t)(n + 1), poly);
    }
    bvec out;
    if (!rc && !bvec_init(&out, (uint64_t)n + 1, limbs_for(bits))) rc = TR_MATH_MEMORY;
    if (!rc) {
        /* res is laid out per prime: value k of prime q at q (n + 1) + k */
        rc = crt_values(pr, np, res, (uint64_t)n + 1, (uint64_t)n + 1, bits, &out, 0);
        for (int32_t k = 0; k <= n && !rc; k++) { int32_t err = 0; a[k] = bv_to_bigint(&out, (uint64_t)k, &err); if (!a[k]) rc = err ? err : TR_MATH_MEMORY; }
        bvec_free(&out);
    }
    free(pr); free(res); free(H); free(poly);
    *nprimes = (int32_t)np; *bound = bits;
    return rc;
}
tr_dyn_recurrence *tr_dyn_charpoly(const tr_dyn_operator *o, const tr_math_budget *budget, tr_math_refusal *refusal,
                                   int32_t *error) {
    tr_budget_none(refusal);
    int32_t rc = power_check(o);
    if (rc) return rec_fail(NULL, error, rc);
    tr_math_need need;
    if ((rc = tr_dyn_need(TR_DYN_REQ_CHARPOLY, o, NULL, 0, TR_DYN_MODULAR, TR_DYN_NEED_CPU, &need, NULL))) return rec_fail(NULL, error, rc);
    if ((rc = check_need(budget, &need, refusal))) return rec_fail(NULL, error, rc);
    const int32_t n = o->nb;
    tr_dyn_recurrence *r = (tr_dyn_recurrence *)zalloc(sizeof *r);
    tr_bigint **a = (tr_bigint **)zalloc((uint64_t)(n + 1) * sizeof(tr_bigint *));
    if (!r || !a) { free(a); return rec_fail(r, error, TR_MATH_MEMORY); }
    r->kind = TR_DYN_CHARPOLY; r->nb = n; r->degree = 0; r->ncoef = n + 1;
    r->coef = (tr_bigq **)zalloc((uint64_t)(n + 1) * sizeof(tr_bigq *));
    if (!r->coef) rc = TR_MATH_MEMORY;
    if (!rc) rc = charpoly_int(o, a, &r->primes, &r->bound_bits);
    tr_math_budget u;
    unlimited(&u);
    int32_t err = 0;
    tr_bigint *D = rc ? NULL : tr_bigint_from_i128(tr_i128_set(o->den), &u, NULL, &err);
    for (int32_t k = 0; k <= n && !rc; k++) {                    /* a_k / D^(n - k) */
        tr_bigint *dp = D ? tr_bigint_pow(D, (uint64_t)(n - k), &u, NULL, &err) : NULL;
        r->coef[k] = dp ? bq_of_bigints(a[k], dp, &err) : NULL;
        tr_bigint_free(dp);
        if (!r->coef[k]) rc = err ? err : TR_MATH_MEMORY;
    }
    tr_bigint_free(D);
    for (int32_t k = 0; k <= n; k++) tr_bigint_free(a[k]);
    free(a);
    if (rc) return rec_fail(r, error, rc);
    r->checked = 1;
    r->bytes = sizeof *r + (uint64_t)(n + 1) * 64;
    r->magic = REC_MAGIC;
    set_error(error, TR_MATH_OK);
    return r;
}
/* Berlekamp-Massey over Q on u[0..N): L and C[0..L] (C[0] = 1): u_i = -sum_(j=1..L) C_j u_(i-j) */
static int32_t berlekamp_massey(tr_bigint *const *u, int32_t N, tr_bigq **Cout, int32_t *Lout) {
    tr_math_budget bud;
    unlimited(&bud);
    int32_t err = 0, rc = TR_MATH_OK, L = 0, mm = 1;
    tr_bigq **C = (tr_bigq **)zalloc((uint64_t)(N + 2) * sizeof(tr_bigq *)), **B = (tr_bigq **)zalloc((uint64_t)(N + 2) * sizeof(tr_bigq *));
    tr_bigq **T = (tr_bigq **)zalloc((uint64_t)(N + 2) * sizeof(tr_bigq *)), **s = (tr_bigq **)zalloc((uint64_t)(N + 1) * sizeof(tr_bigq *));
    tr_bigq *b = NULL;
    if (!C || !B || !T || !s) rc = TR_MATH_MEMORY;
    for (int32_t i = 0; i <= N + 1 && !rc; i++) {
        C[i] = bq_of_i128(i == 0, &bud, &err); B[i] = bq_of_i128(i == 0, &bud, &err);
        if (!C[i] || !B[i]) rc = err ? err : TR_MATH_MEMORY;
    }
    for (int32_t i = 0; i < N && !rc; i++) { tr_bigint *one = tr_bigint_from_i64(1, &bud, NULL, &err); s[i] = one ? bq_of_bigints(u[i], one, &err) : NULL; tr_bigint_free(one); if (!s[i]) rc = err ? err : TR_MATH_MEMORY; }
    if (!rc) { b = bq_of_i128(1, &bud, &err); if (!b) rc = err ? err : TR_MATH_MEMORY; }
    for (int32_t i = 0; i < N && !rc; i++) {
        tr_bigq *d = tr_bigq_copy(s[i], &bud, NULL, &err);       /* the discrepancy */
        for (int32_t j = 1; j <= L && d; j++) {
            tr_bigq *p = tr_bigq_mul(C[j], s[i - j], &bud, NULL, &err), *nd = p ? tr_bigq_add(d, p, &bud, NULL, &err) : NULL;
            tr_bigq_free(p); tr_bigq_free(d); d = nd;
        }
        if (!d) { rc = err ? err : TR_MATH_MEMORY; break; }
        if (tr_bigq_sign(d) == 0) { mm++; tr_bigq_free(d); continue; }
        tr_bigq *coef = tr_bigq_div(d, b, &bud, NULL, &err);     /* C -= (d / b) x^m B */
        if (!coef) { tr_bigq_free(d); rc = err ? err : TR_MATH_MEMORY; break; }
        int lengthen = 2 * L <= i;
        for (int32_t j = 0; j <= N + 1 && !rc; j++) { T[j] = tr_bigq_copy(C[j], &bud, NULL, &err); if (!T[j]) rc = err ? err : TR_MATH_MEMORY; }
        for (int32_t j = 0; j + mm <= N + 1 && !rc; j++) {
            if (tr_bigq_sign(B[j]) == 0) continue;
            tr_bigq *p = tr_bigq_mul(coef, B[j], &bud, NULL, &err), *nc = p ? tr_bigq_sub(C[j + mm], p, &bud, NULL, &err) : NULL;
            tr_bigq_free(p);
            if (!nc) { rc = err ? err : TR_MATH_MEMORY; break; }
            tr_bigq_free(C[j + mm]); C[j + mm] = nc;
        }
        tr_bigq_free(coef);
        if (!rc && lengthen) {
            L = i + 1 - L;
            for (int32_t j = 0; j <= N + 1; j++) { tr_bigq_free(B[j]); B[j] = T[j]; T[j] = NULL; }
            tr_bigq_free(b); b = d; d = NULL; mm = 1;
        } else mm++;
        for (int32_t j = 0; j <= N + 1 && T; j++) { tr_bigq_free(T[j]); T[j] = NULL; }
        tr_bigq_free(d);
    }
    if (!rc) { for (int32_t j = 0; j <= L; j++) { Cout[j] = C[j]; C[j] = NULL; } *Lout = L; }
    for (int32_t i = 0; i <= N + 1; i++) { if (C) tr_bigq_free(C[i]); if (B) tr_bigq_free(B[i]); if (T) tr_bigq_free(T[i]); }
    for (int32_t i = 0; i < N && s; i++) tr_bigq_free(s[i]);
    free(C); free(B); free(T); free(s); tr_bigq_free(b);
    return rc;
}
tr_dyn_recurrence *tr_dyn_recurrence_make(const tr_dyn_operator *o, const int64_t *c, const tr_sym_state *s,
                                          uint64_t group, const tr_math_i128 *vec, int32_t kind,
                                          const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    int32_t rc = power_check(o);
    if (rc) return rec_fail(NULL, error, rc);
    if (!c || (kind != TR_DYN_CHARPOLY && kind != TR_DYN_MINIMAL) || (!s && !vec)) return rec_fail(NULL, error, TR_MATH_INVALID);
    const tr_math_i128 *v = vec;
    if (s) {
        tr_sym_state_info si;
        tr_sym_query_view qv;
        if ((rc = state_frame(s, o, &si, &qv))) return rec_fail(NULL, error, rc);
        if (group >= qv.ngroups) return rec_fail(NULL, error, TR_MATH_INVALID);
        v = qv.moments + group * (uint64_t)o->nb;
    }
    tr_math_need need;
    if ((rc = tr_dyn_need(TR_DYN_REQ_RECURRENCE, o, NULL, 0, TR_DYN_STEPPING, TR_DYN_NEED_CPU, &need, NULL))) return rec_fail(NULL, error, rc);
    if ((rc = check_need(budget, &need, refusal))) return rec_fail(NULL, error, rc);
    const int32_t nb = o->nb, N = kind == TR_DYN_MINIMAL ? 2 * nb : nb;
    tr_math_budget u;
    unlimited(&u);
    int32_t err = 0;
    /* u_t = c^T N^t v for t < N (stepping, exact) */
    uint64_t vbits = mom_bits(v, (uint64_t)nb), cbits = 0;
    for (int32_t i = 0; i < nb; i++) { uint64_t b = bitlen64(c[i] < 0 ? (uint64_t)0 - (uint64_t)c[i] : (uint64_t)c[i]); if (b > cbits) cbits = b; }
    const uint64_t cap = limbs_for(tr_budget_add(pow_bits(o, (uint64_t)N), vbits)), ucap = cap + limbs_for(cbits + 13);
    bvec cur, nxt, uu;
    acc_t acc;
    memset(&cur, 0, sizeof cur); memset(&nxt, 0, sizeof nxt); memset(&uu, 0, sizeof uu); memset(&acc, 0, sizeof acc);
    if (!bvec_init(&cur, (uint64_t)nb, cap) || !bvec_init(&nxt, (uint64_t)nb, cap) || !bvec_init(&uu, (uint64_t)N + 1, ucap) ||
        !acc_init(&acc, 2 * ucap + 8)) rc = TR_MATH_MEMORY;
    for (int32_t i = 0; i < nb && !rc; i++) bv_set_i128(&cur, (uint64_t)i, tr_i128_get(v[i]));
    for (int32_t t = 0; t <= N && !rc; t++) {
        for (int32_t i = 0; i < nb; i++) if (c[i] && cur.len[i]) acc_ib(&acc, (i128)c[i], &cur, (uint64_t)i);
        if (!acc_store(&acc, &uu, (uint64_t)t)) { rc = TR_MATH_OVERFLOW; break; }
        if (t == N) break;
        rc = bv_matvec_i128(o->num, nb, &cur, 1, &nxt, &acc);
        bvec sw = cur; cur = nxt; nxt = sw;
    }
    tr_bigint **ub = (tr_bigint **)zalloc((uint64_t)(N + 1) * sizeof(tr_bigint *));
    if (!ub && !rc) rc = TR_MATH_MEMORY;
    for (int32_t t = 0; t <= N && !rc; t++) { ub[t] = bv_to_bigint(&uu, (uint64_t)t, &err); if (!ub[t]) rc = err ? err : TR_MATH_MEMORY; }
    bvec_free(&cur); bvec_free(&nxt); bvec_free(&uu); acc_free(&acc);
    tr_dyn_recurrence *r = rc ? NULL : (tr_dyn_recurrence *)zalloc(sizeof *r);
    if (!rc && !r) rc = TR_MATH_MEMORY;
    if (!rc) { r->kind = kind; r->nb = nb; r->sequence = 1; r->terms = (uint64_t)N + 1; }
    int32_t d = 0;
    tr_bigint **rr = NULL;
    if (!rc && kind == TR_DYN_CHARPOLY) {                        /* r_k = -a_k */
        tr_bigint **a = (tr_bigint **)zalloc((uint64_t)(nb + 1) * sizeof(tr_bigint *));
        rr = (tr_bigint **)zalloc((uint64_t)(nb ? nb : 1) * sizeof(tr_bigint *));
        if (!a || !rr) rc = TR_MATH_MEMORY;
        if (!rc) rc = charpoly_int(o, a, &r->primes, &r->bound_bits);
        for (int32_t k = 0; k < nb && !rc; k++) { rr[k] = tr_bigint_neg(a[k], &u, NULL, &err); if (!rr[k]) rc = err ? err : TR_MATH_MEMORY; }
        if (a) for (int32_t k = 0; k <= nb; k++) tr_bigint_free(a[k]);
        free(a);
        d = nb;
    } else if (!rc) {                                            /* Berlekamp-Massey on u_0 .. u_(2 nb - 1) */
        tr_bigq **C = (tr_bigq **)zalloc((uint64_t)(N + 2) * sizeof(tr_bigq *));
        if (!C) rc = TR_MATH_MEMORY;
        if (!rc) rc = berlekamp_massey(ub, N, C, &d);
        rr = (tr_bigint **)zalloc((uint64_t)(d ? d : 1) * sizeof(tr_bigint *));
        if (!rr && !rc) rc = TR_MATH_MEMORY;
        for (int32_t k = 0; k < d && !rc; k++) {                 /* r_k = -C_(d-k), integral (Gauss) */
            const tr_bigq *ck = C[d - k];
            tr_bigint *one = tr_bigint_from_i64(1, &u, NULL, &err);
            if (!one) { rc = TR_MATH_MEMORY; break; }
            if (tr_bigint_cmp(tr_bigq_denominator(ck), one) != 0) rc = TR_MATH_INVALID;   /* not integral: impossible */
            tr_bigint_free(one);
            if (!rc) { rr[k] = tr_bigint_neg(tr_bigq_numerator(ck), &u, NULL, &err); if (!rr[k]) rc = err ? err : TR_MATH_MEMORY; }
        }
        if (C) for (int32_t k = 0; k <= N + 1; k++) tr_bigq_free(C[k]);
        free(C);
    }
    if (!rc) {                                                   /* the recurrence on every computed term */
        for (int32_t t = d; t <= N && !rc; t++) {
            tr_bigint *sum = tr_bigint_from_i64(0, &u, NULL, &err);
            for (int32_t k = 0; k < d && sum; k++) {
                tr_bigint *p = tr_bigint_mul(rr[k], ub[t - d + k], &u, NULL, &err), *ns = p ? tr_bigint_add(sum, p, &u, NULL, &err) : NULL;
                tr_bigint_free(p); tr_bigint_free(sum); sum = ns;
            }
            if (!sum) { rc = err ? err : TR_MATH_MEMORY; break; }
            if (tr_bigint_cmp(sum, ub[t]) != 0) rc = TR_MATH_INVALID;
            tr_bigint_free(sum);
        }
    }
    if (!rc) {
        r->degree = d; r->r = rr; rr = NULL; r->checked = 1;
        r->u = (tr_bigint **)zalloc((uint64_t)(d ? d : 1) * sizeof(tr_bigint *));
        r->coef = (tr_bigq **)zalloc((uint64_t)(d ? d : 1) * sizeof(tr_bigq *));
        r->init = (tr_bigq **)zalloc((uint64_t)(d ? d : 1) * sizeof(tr_bigq *));
        r->ncoef = d;
        r->D = tr_bigint_from_i128(tr_i128_set(o->den), &u, NULL, &err);
        if (!r->u || !r->coef || !r->init || !r->D) rc = TR_MATH_MEMORY;
        for (int32_t k = 0; k < d && !rc; k++) {                 /* s-coefficients r_k D^(k - d), s_k = u_k / D^k */
            r->u[k] = tr_bigint_copy(ub[k], &u, NULL, &err);
            tr_bigint *dp = tr_bigint_pow(r->D, (uint64_t)(d - k), &u, NULL, &err), *dk = tr_bigint_pow(r->D, (uint64_t)k, &u, NULL, &err);
            r->coef[k] = dp ? bq_of_bigints(r->r[k], dp, &err) : NULL;
            r->init[k] = dk ? bq_of_bigints(ub[k], dk, &err) : NULL;
            tr_bigint_free(dp); tr_bigint_free(dk);
            if (!r->u[k] || !r->coef[k] || !r->init[k]) rc = err ? err : TR_MATH_MEMORY;
        }
    }
    if (rr) { for (int32_t k = 0; k < d; k++) tr_bigint_free(rr[k]); free(rr); }
    if (ub) for (int32_t t = 0; t <= N; t++) tr_bigint_free(ub[t]);
    free(ub);
    if (rc) return rec_fail(r, error, rc);
    r->bytes = sizeof *r + (uint64_t)(d + 1) * 128;
    r->magic = REC_MAGIC;
    set_error(error, TR_MATH_OK);
    return r;
}
int32_t tr_dyn_recurrence_get_info(const tr_dyn_recurrence *r, tr_dyn_rec_info *out) {
    if (!rec_ok(r) || !out) return TR_MATH_INVALID;
    memset(out, 0, sizeof *out);
    out->kind = r->kind; out->degree = r->kind == TR_DYN_CHARPOLY && !r->sequence ? r->nb : r->degree; out->nb = r->nb;
    out->primes = r->primes; out->sequence = r->sequence; out->checked = r->checked;
    out->terms = r->terms; out->bound_bits = r->bound_bits; out->bytes = r->bytes;
    return TR_MATH_OK;
}
static uint64_t rec_maxbits(const tr_dyn_recurrence *r) {
    uint64_t b = 0;
    for (int32_t k = 0; k < r->degree; k++) { uint64_t x = tr_bigint_bit_length(r->r[k]); if (x > b) b = x; }
    return b;
}
int32_t tr_dyn_recurrence_term_need(const tr_dyn_recurrence *r, uint64_t T, tr_math_need *need) {
    if (!rec_ok(r) || !need || !r->sequence) return TR_MATH_INVALID;
    need_zero(need);
    uint64_t wb = tr_budget_mul(T, rec_maxbits(r) + 1), ub = 0;   /* |x^T mod p| <= (1 + max|r|)^T */
    for (int32_t k = 0; k < r->degree; k++) { uint64_t x = tr_bigint_bit_length(r->u[k]); if (x > ub) ub = x; }
    uint64_t d = (uint64_t)(r->degree ? r->degree : 1);
    uint64_t bits = tr_budget_add(tr_budget_add(wb, ub), bitlen64(d) + tr_budget_mul(T, tr_bigint_bit_length(r->D)));
    need->need[TR_BUDGET_SUPPORT] = d;
    need->need[TR_BUDGET_ITERATIONS] = T;
    need->need[TR_BUDGET_OUTPUT] = 1;
    need->need[TR_BUDGET_BYTES] = tr_budget_mul(tr_budget_mul(6 * d, limbs_for(tr_budget_add(tr_budget_mul(wb, 2), 64))), 8);
    need->need[TR_BUDGET_BYTES] = tr_budget_add(need->need[TR_BUDGET_BYTES], tr_budget_mul(limbs_for(bits), 24));
    return TR_MATH_OK;
}
/* a (degree < d) times b mod p(x) = x^d - sum r_k x^k, into out (d coefficients); big integers */
static int32_t polymulmod(tr_bigint *const *a, tr_bigint *const *b, tr_bigint *const *r, int32_t d, tr_bigint **out) {
    tr_math_budget u;
    unlimited(&u);
    int32_t err = 0, rc = TR_MATH_OK;
    tr_bigint **p = (tr_bigint **)zalloc((uint64_t)(2 * d) * sizeof(tr_bigint *));
    if (!p) return TR_MATH_MEMORY;
    for (int32_t k = 0; k < 2 * d && !rc; k++) { p[k] = tr_bigint_from_i64(0, &u, NULL, &err); if (!p[k]) rc = TR_MATH_MEMORY; }
    for (int32_t i = 0; i < d && !rc; i++) {
        if (!tr_bigint_sign(a[i])) continue;
        for (int32_t j = 0; j < d && !rc; j++) {
            if (!tr_bigint_sign(b[j])) continue;
            tr_bigint *m = tr_bigint_mul(a[i], b[j], &u, NULL, &err), *s = m ? tr_bigint_add(p[i + j], m, &u, NULL, &err) : NULL;
            tr_bigint_free(m);
            if (!s) { rc = err ? err : TR_MATH_MEMORY; break; }
            tr_bigint_free(p[i + j]); p[i + j] = s;
        }
    }
    for (int32_t m = 2 * d - 2; m >= d && !rc; m--) {            /* x^m = x^(m-d) sum_k r_k x^k */
        if (!tr_bigint_sign(p[m])) continue;
        for (int32_t k = 0; k < d && !rc; k++) {
            if (!tr_bigint_sign(r[k])) continue;
            tr_bigint *x = tr_bigint_mul(p[m], r[k], &u, NULL, &err), *s = x ? tr_bigint_add(p[m - d + k], x, &u, NULL, &err) : NULL;
            tr_bigint_free(x);
            if (!s) { rc = err ? err : TR_MATH_MEMORY; break; }
            tr_bigint_free(p[m - d + k]); p[m - d + k] = s;
        }
    }
    for (int32_t k = 0; k < d && !rc; k++) { out[k] = p[k]; p[k] = NULL; }
    for (int32_t k = 0; k < 2 * d; k++) tr_bigint_free(p[k]);
    free(p);
    return rc;
}
static int32_t export_bigint(const tr_bigint *x, uint64_t *out, uint64_t capacity, uint64_t *w) {
    uint64_t L = tr_bigint_limb_count(x);
    if (out) {
        if (*w + 1 + L > capacity) return TR_MATH_INVALID;
        int32_t sign = 0;
        uint64_t count = 0;
        if (tr_bigint_export(x, &sign, out + *w + 1, L ? L : 1, &count)) return TR_MATH_INVALID;
        out[*w] = count | (sign < 0 ? (uint64_t)1 << 63 : 0);
    }
    *w += 1 + L;
    return TR_MATH_OK;
}
static int32_t export_bigq(const tr_bigq *q, uint64_t *out, uint64_t capacity, uint64_t *w) {
    int32_t rc = export_bigint(tr_bigq_numerator(q), out, capacity, w);
    return rc ? rc : export_bigint(tr_bigq_denominator(q), out, capacity, w);
}
int32_t tr_dyn_recurrence_term(const tr_dyn_recurrence *r, uint64_t T, uint64_t *out, uint64_t capacity, uint64_t *words,
                               const tr_math_budget *budget, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!rec_ok(r) || !r->sequence || !words) return TR_MATH_INVALID;
    tr_math_need need;
    tr_dyn_recurrence_term_need(r, T, &need);
    int32_t rc = check_need(budget, &need, refusal);
    if (rc) return rc;
    tr_math_budget u;
    unlimited(&u);
    int32_t err = 0;
    const int32_t d = r->degree;
    tr_bigint *uT = NULL;
    if (d == 0) uT = tr_bigint_from_i64(0, &u, NULL, &err);       /* the zero sequence */
    else if (T < (uint64_t)d) uT = tr_bigint_copy(r->u[T], &u, NULL, &err);
    else {                                                       /* w = x^T mod p by square and multiply */
        tr_bigint **w = (tr_bigint **)zalloc((uint64_t)d * sizeof(tr_bigint *)), **x = (tr_bigint **)zalloc((uint64_t)d * sizeof(tr_bigint *));
        tr_bigint **t = (tr_bigint **)zalloc((uint64_t)d * sizeof(tr_bigint *));
        if (!w || !x || !t) rc = TR_MATH_MEMORY;
        for (int32_t k = 0; k < d && !rc; k++) {
            w[k] = tr_bigint_from_i64(k == 0, &u, NULL, &err);
            x[k] = tr_bigint_from_i64(d == 1 ? 0 : (k == 1), &u, NULL, &err);
            if (!w[k] || !x[k]) rc = TR_MATH_MEMORY;
        }
        if (!rc && d == 1) { tr_bigint_free(x[0]); x[0] = tr_bigint_copy(r->r[0], &u, NULL, &err); if (!x[0]) rc = TR_MATH_MEMORY; }   /* x mod (x - r0) = r0 */
        uint64_t e = T;
        while (e && !rc) {
            if (e & 1) {
                rc = polymulmod(w, x, r->r, d, t);
                for (int32_t k = 0; k < d && !rc; k++) { tr_bigint_free(w[k]); w[k] = t[k]; t[k] = NULL; }
            }
            e >>= 1;
            if (e && !rc) {
                rc = polymulmod(x, x, r->r, d, t);
                for (int32_t k = 0; k < d && !rc; k++) { tr_bigint_free(x[k]); x[k] = t[k]; t[k] = NULL; }
            }
        }
        if (!rc) {                                               /* u_T = sum_k w_k u_k */
            uT = tr_bigint_from_i64(0, &u, NULL, &err);
            for (int32_t k = 0; k < d && uT; k++) {
                tr_bigint *p = tr_bigint_mul(w[k], r->u[k], &u, NULL, &err), *s = p ? tr_bigint_add(uT, p, &u, NULL, &err) : NULL;
                tr_bigint_free(p); tr_bigint_free(uT); uT = s;
            }
        }
        for (int32_t k = 0; k < d; k++) { if (w) tr_bigint_free(w[k]); if (x) tr_bigint_free(x[k]); if (t) tr_bigint_free(t[k]); }
        free(w); free(x); free(t);
    }
    if (!uT && !rc) rc = err ? err : TR_MATH_MEMORY;
    tr_bigint *DT = rc ? NULL : tr_bigint_pow(r->D, T, &u, NULL, &err);
    tr_bigq *sT = DT ? bq_of_bigints(uT, DT, &err) : NULL;
    if (!sT && !rc) rc = err ? err : TR_MATH_MEMORY;
    uint64_t w = 0;
    if (!rc) rc = export_bigq(sT, out, out ? capacity : 0, &w);
    if (!rc) *words = w;
    tr_bigint_free(uT); tr_bigint_free(DT); tr_bigq_free(sT);
    return rc;
}
void tr_dyn_recurrence_free(tr_dyn_recurrence *r) {
    if (!rec_ok(r)) return;
    r->magic = 0;
    rec_release(r);
}

/* ================================================================== floating powers (approximate, certified) */
static tr_math_interval iv_entry(const tr_dyn_operator *o, uint64_t x) {
    i128 v = tr_i128_get(o->num[x]);
    if (!v) { tr_math_interval z; z.lo = 0.0; z.hi = 0.0; return z; }
    tr_math_interval a = tr_sym_iv_i128(v);
    return o->den == 1 ? a : tr_interval_div(a, tr_sym_iv_i128(o->den));
}
static int iv_isz(tr_math_interval a) { return a.lo == 0.0 && a.hi == 0.0; }
static void iv_matmul(const tr_math_interval *A, const tr_math_interval *B, int32_t nb, tr_math_interval *C) {
    for (int32_t i = 0; i < nb; i++)
        for (int32_t j = 0; j < nb; j++) {
            tr_math_interval s;
            s.lo = 0.0; s.hi = 0.0;
            for (int32_t k = 0; k < nb; k++) {
                tr_math_interval a = A[(uint64_t)i * nb + k], b = B[(uint64_t)k * nb + j];
                if (iv_isz(a) || iv_isz(b)) continue;
                s = tr_interval_add(s, tr_interval_mul(a, b));
            }
            C[(uint64_t)i * nb + j] = s;
        }
}
/* the error and magnitude of a set of enclosures (ABS: max radius; REL: max radius / max |mid|), RU */
static void iv_error(const tr_math_interval *v, uint64_t n, int32_t norm, double *mid, double *rad, double *error, double *mag) {
    double er = 0.0, mg = 0.0, mgd = 0.0;
    int finite = 1;
    for (uint64_t i = 0; i < n; i++) {
        tr_math_interval a = v[i];
        if (!(a.lo > -INFINITY && a.hi < INFINITY)) { finite = 0; break; }
        double m = a.lo * 0.5 + a.hi * 0.5;
        if (m < a.lo) m = a.lo;
        if (m > a.hi) m = a.hi;
        double r1 = tr_add_ru(m, -a.lo), r2 = tr_add_ru(a.hi, -m), r = r1 > r2 ? r1 : r2;
        if (mid) mid[i] = m;
        if (rad) rad[i] = r;
        if (r > er) er = r;
        double am = fabs(a.lo) > fabs(a.hi) ? fabs(a.lo) : fabs(a.hi);
        if (am > mg) mg = am;
        double dm = fabs(m);
        if (dm > mgd) mgd = dm;
    }
    if (!finite) { *error = INFINITY; *mag = INFINITY; return; }
    *mag = mg;
    if (norm == TR_DYN_NORM_REL) *error = er == 0.0 ? 0.0 : (mgd > 0.0 ? tr_div_dir(er, mgd, 1) : INFINITY);
    else *error = er;
}
int32_t tr_dyn_float_power(const tr_dyn_operator *o, uint64_t n, const tr_sym_state *s, double tol, int32_t norm,
                           double *mid, double *rad, tr_dyn_float_info *info, const tr_math_budget *budget,
                           tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    int32_t rc = power_check(o);
    if (rc) return rc;
    if (!mid || !rad || !info || !(tol > 0.0) || !(tol < INFINITY) || (norm != TR_DYN_NORM_ABS && norm != TR_DYN_NORM_REL))
        return TR_MATH_INVALID;
    tr_math_need need;
    if ((rc = need_core(TR_DYN_REQ_FLOAT, o, s, n, TR_DYN_FLOAT, TR_DYN_NEED_CPU, &need, NULL, NULL))) return rc;
    if ((rc = check_need(budget, &need, refusal))) return rc;
    memset(info, 0, sizeof *info);
    info->exponent = n; info->norm = norm; info->tolerance = tol;
    const int32_t nb = o->nb;
    const uint64_t e2 = (uint64_t)nb * (uint64_t)nb;
    tr_sym_state_info si;
    tr_sym_query_view qv;
    uint64_t G = 0;
    if (s) { if ((rc = state_frame(s, o, &si, &qv))) return rc; G = qv.ngroups; }
    tr_math_interval *R = (tr_math_interval *)zalloc(e2 * sizeof *R), *B = (tr_math_interval *)zalloc(e2 * sizeof *B);
    tr_math_interval *T = (tr_math_interval *)zalloc(e2 * sizeof *T);
    tr_math_interval *Mv = s ? (tr_math_interval *)zalloc(tr_budget_mul(G, (uint64_t)nb) * sizeof *Mv) : NULL;
    if (!R || !B || !T || (s && !Mv)) { free(R); free(B); free(T); free(Mv); return TR_MATH_MEMORY; }
    for (uint64_t x = 0; x < e2; x++) {
        B[x] = iv_entry(o, x);
        R[x].lo = R[x].hi = (x % (uint64_t)(nb + 1) == 0) ? 1.0 : 0.0;
    }
    uint64_t e = n, eR = 0, eB = 1, reached = 0;
    int finite = 1;
    double er = 0.0, mg = 0.0;
    while (e && finite) {
        if (e & 1) {
            iv_matmul(R, B, nb, T);
            memcpy(R, T, (size_t)e2 * sizeof *R);
            eR += eB;
            info->products++;
            iv_error(R, e2, norm, NULL, NULL, &er, &mg);
            if (!(er < INFINITY)) finite = 0;
            else if (er <= tol) reached = eR;
        }
        e >>= 1;
        if (e && finite) {
            iv_matmul(B, B, nb, T);
            memcpy(B, T, (size_t)e2 * sizeof *B);
            eB *= 2;
            info->products++;
            iv_error(B, e2, norm, NULL, NULL, &er, &mg);
            if (!(er < INFINITY)) finite = 0;
        }
    }
    tr_math_interval *outv = R;
    uint64_t nout = e2;
    if (finite && s) {                                           /* K^n m_g with the moments enclosed */
        for (uint64_t g = 0; g < G; g++)
            for (int32_t i = 0; i < nb; i++) {
                tr_math_interval acc;
                acc.lo = 0.0; acc.hi = 0.0;
                for (int32_t j = 0; j < nb; j++) {
                    i128 m = tr_i128_get(qv.moments[g * (uint64_t)nb + (uint64_t)j]);
                    tr_math_interval a = R[(uint64_t)i * nb + j];
                    if (!m || iv_isz(a)) continue;
                    acc = tr_interval_add(acc, tr_interval_mul(a, tr_sym_iv_i128(m)));
                }
                Mv[g * (uint64_t)nb + (uint64_t)i] = acc;
            }
        outv = Mv; nout = G * (uint64_t)nb;
        info->groups = (int32_t)G;
    }
    double error = INFINITY, mag = INFINITY;
    if (finite) iv_error(outv, nout, norm, NULL, NULL, &error, &mag);
    info->error = error; info->magnitude = mag;
    if (n == 0) reached = 0;
    info->reached = finite && error <= tol ? n : reached;
    info->stable = finite && error <= tol;
    if (info->stable) iv_error(outv, nout, norm, mid, rad, &error, &mag);
    free(R); free(B); free(T); free(Mv);
    return info->stable ? TR_MATH_OK : TR_MATH_NOT_CONVERGED;
}

/* ================================================================== exports */
static int32_t export_bv(const bvec *v, uint64_t from, uint64_t count, uint64_t *out, uint64_t capacity, uint64_t *w) {
    for (uint64_t i = from; i < from + count; i++) {
        uint64_t L = v->len[i];
        if (out) {
            if (*w + 1 + L > capacity) return TR_MATH_INVALID;
            out[*w] = L | (v->neg[i] ? (uint64_t)1 << 63 : 0);
            memcpy(out + *w + 1, BV(v, i), (size_t)L * 8);
        }
        *w += 1 + L;
    }
    return TR_MATH_OK;
}
int32_t tr_dyn_export(const void *object, int32_t part, uint64_t *out, uint64_t capacity, uint64_t *words) {
    if (!object || !words) return TR_MATH_INVALID;
    uint32_t magic = *(const uint32_t *)object;
    uint64_t w = 0;
    int32_t rc = TR_MATH_INVALID;
    if (magic == OP_MAGIC) {
        const tr_dyn_operator *o = (const tr_dyn_operator *)object;
        if (part == TR_DYN_PART_EPSILON && o->kind == TR_DYN_OP_TRUNCATED) {
            rc = TR_MATH_OK;
            for (int32_t a = 0; a < o->nb && !rc; a++) rc = export_bigq(o->eps[a], out, capacity, &w);
        }
    } else if (magic == POW_MAGIC) {
        const tr_dyn_power *p = (const tr_dyn_power *)object;
        if (part == TR_DYN_PART_NUMERATORS) rc = export_bv(&p->num, 0, p->num.count, out, capacity, &w);
        else if (part == TR_DYN_PART_DENOMINATOR) rc = export_bv(&p->den, 0, 1, out, capacity, &w);
    } else if (magic == EV_MAGIC) {
        const tr_dyn_evolution *e = (const tr_dyn_evolution *)object;
        const uint64_t vals = e->ngroups * (uint64_t)e->nb * (uint64_t)(e->pdegree + 1);
        if (part == TR_DYN_PART_MOMENTS) rc = export_bv(&e->num, 0, vals, out, capacity, &w);
        else if (part == TR_DYN_PART_DENOMINATORS) rc = export_bv(&e->den, 0, (uint64_t)e->nb, out, capacity, &w);
        else if (part == TR_DYN_PART_ERRORS && e->err.count) rc = export_bv(&e->err, 0, e->ngroups * (uint64_t)e->nb, out, capacity, &w);
        else if (part == TR_DYN_PART_ERROR_DEN && e->errden.count) rc = export_bv(&e->errden, 0, 1, out, capacity, &w);
    } else if (magic == REC_MAGIC) {
        const tr_dyn_recurrence *r = (const tr_dyn_recurrence *)object;
        if (part == TR_DYN_PART_COEFFICIENTS) { rc = TR_MATH_OK; for (int32_t k = 0; k < r->ncoef && !rc; k++) rc = export_bigq(r->coef[k], out, capacity, &w); }
        else if (part == TR_DYN_PART_INITIAL && r->sequence) { rc = TR_MATH_OK; for (int32_t k = 0; k < r->degree && !rc; k++) rc = export_bigq(r->init[k], out, capacity, &w); }
    }
    if (!rc) *words = w;
    return rc;
}
