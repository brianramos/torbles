/* PRE-002: finite roots, residues and periodic preimages; see math_preimg.h.
 * SPDX-License-Identifier: Apache-2.0 */
#include "math_preimg.h"
#include "mathematics.h"
#include "math_domain.h"
#include "math_handles.h"
#include "math_carriers.h"
#include "math_certificates.h"
#include "math_inv.h"
#include <stdlib.h>
#include <string.h>

TR_MATH_ABI(tr_preimg_map, TR_PREIMG_ABI);     /* the ctypes boundary of PRE-002 (math_handles.h) */
TR_MATH_ABI(tr_preimg_spec, TR_PREIMG_ABI);
TR_MATH_ABI(tr_preimg_branch, TR_PREIMG_ABI);
TR_MATH_ABI(tr_preimg_info, TR_PREIMG_ABI);
TR_MATH_ABI(tr_preimg_view, TR_PREIMG_ABI);
TR_MATH_ABI(tr_preimg_verification, TR_PREIMG_ABI);

_Static_assert(sizeof(tr_preimg_map) == 56, "tr_preimg_map ABI");
_Static_assert(sizeof(tr_preimg_spec) == 8, "tr_preimg_spec ABI");
_Static_assert(sizeof(tr_preimg_branch) == 64, "tr_preimg_branch ABI");
_Static_assert(sizeof(tr_preimg_info) == 136, "tr_preimg_info ABI");
_Static_assert(sizeof(tr_preimg_view) == 192, "tr_preimg_view ABI");
_Static_assert(sizeof(tr_preimg_verification) == 64, "tr_preimg_verification ABI");

#define SET_MAGIC UINT32_C(0x47494d50)
#define I128_MAX ((__int128)(((unsigned __int128)1 << 127) - 1))
#define I128_MIN (-I128_MAX - 1)

struct tr_preimg_set_s {
    uint32_t magic;
    tr_preimg_info info;
    tr_preimg_view view;              /* points into the arrays below */
    tr_preimg_branch *branch;
    uint64_t *lookup;
    int64_t *icoef;
    uint64_t *wcoef, *wthreshold;
    tr_num_rational *values;
    tr_cert *cert;
};

static void set_error(int32_t *error, int32_t code) { if (error) *error = code; }
static void *zalloc(uint64_t bytes) {
    if (bytes > (uint64_t)SIZE_MAX) return NULL;
    return calloc(1, bytes ? (size_t)bytes : 1);
}
static int set_valid(const tr_preimg_set *s) { return s && s->magic == SET_MAGIC; }
static uint64_t bitlen(uint64_t x) { return x ? (uint64_t)(64 - __builtin_clzll(x)) : 0; }
static int64_t fmod64(__int128 x, int64_t m) { return tr_preimg_fmod128(x, m); }
static int canonical(tr_num_rational q) {
    if (q.denominator <= 0) return 0;
    uint64_t g = tr_gcd_u64(tr_abs_u64(q.numerator), (uint64_t)q.denominator);
    return g == 1 || (q.numerator == 0 && q.denominator == 1);
}
static __int128 floor_q(tr_num_rational q) {                /* floor of an int64 rational */
    __int128 n = q.numerator, d = q.denominator, f = n / d;
    if (n % d && n < 0) f--;
    return f;
}
static __int128 ceil_q(tr_num_rational q) {
    __int128 n = q.numerator, d = q.denominator, f = n / d;
    if (n % d && n > 0) f++;
    return f;
}

/* ================================================================ the build context */
typedef struct {
    const tr_math_budget *b;
    tr_math_refusal *r;
    int32_t rc;                        /* a hard error (memory, invalid): the request fails */
    int32_t why;                       /* the union could not be built within the budget: TR_PREIMG_WHY_* */
    int32_t why_resource;              /* the refused resource (the refusal reported when nothing falls back) */
    uint64_t why_need, why_limit;
    uint64_t iter, max_iter;
} ctx;
static void hard(ctx *x, int32_t rc) { if (!x->rc) x->rc = rc ? rc : TR_MATH_MEMORY; }
static void soft(ctx *x, int32_t why, int32_t resource, uint64_t need, uint64_t limit) {
    if (x->why) return;
    x->why = why; x->why_resource = resource; x->why_need = need; x->why_limit = limit;
}
static int step(ctx *x, uint64_t k) {                  /* k exact evaluations; 0 when past the budget */
    if (x->rc || x->why) return 0;
    x->iter += k;
    if (x->iter > x->max_iter) { soft(x, TR_PREIMG_WHY_ITERATIONS, TR_BUDGET_ITERATIONS, x->iter, x->max_iter); return 0; }
    return 1;
}
/* FND-009 operations: a budget refusal stops the union (soft), anything else is hard */
static void big_fail(ctx *x, int32_t e) {
    if (e == TR_MATH_BUDGET && x->r) soft(x, TR_PREIMG_WHY_BYTES, x->r->resource, x->r->need, x->r->limit);
    else if (e == TR_MATH_BUDGET) soft(x, TR_PREIMG_WHY_BYTES, TR_BUDGET_BYTES, 0, 0);
    else hard(x, e);
}
static tr_bigint *bi64(ctx *x, int64_t v) {
    if (x->rc || x->why) return NULL;
    int32_t e = 0; tr_bigint *r = tr_bigint_from_i64(v, x->b, x->r, &e);
    if (!r) big_fail(x, e);
    return r;
}
static tr_bigint *bi128(ctx *x, __int128 v) {
    if (x->rc || x->why) return NULL;
    int32_t e = 0; tr_bigint *r = tr_bigint_from_i128(tr_i128_set(v), x->b, x->r, &e);
    if (!r) big_fail(x, e);
    return r;
}
typedef tr_bigint *(*bigop_fn)(const tr_bigint *, const tr_bigint *, const tr_math_budget *, tr_math_refusal *, int32_t *);
static tr_bigint *bigop(ctx *x, bigop_fn f, const tr_bigint *a, const tr_bigint *b) {
    if (x->rc || x->why || !a || !b) return NULL;
    int32_t e = 0;
    tr_bigint *r = f(a, b, x->b, x->r, &e);
    if (!r) big_fail(x, e);
    return r;
}
static tr_bigint *z_add(ctx *x, const tr_bigint *a, const tr_bigint *b) { return bigop(x, tr_bigint_add, a, b); }
static tr_bigint *z_sub(ctx *x, const tr_bigint *a, const tr_bigint *b) { return bigop(x, tr_bigint_sub, a, b); }
static tr_bigint *z_mul(ctx *x, const tr_bigint *a, const tr_bigint *b) { return bigop(x, tr_bigint_mul, a, b); }
static tr_bigint *z_gcd(ctx *x, const tr_bigint *a, const tr_bigint *b) { return bigop(x, tr_bigint_gcd, a, b); }
static tr_bigint *z_divexact(ctx *x, const tr_bigint *a, const tr_bigint *b) {
    if (x->rc || x->why || !a || !b) return NULL;
    int32_t e = 0; tr_bigint *r = tr_bigint_divexact(a, b, x->b, x->r, &e);
    if (!r) big_fail(x, e);
    return r;
}
static tr_bigint *z_pow(ctx *x, const tr_bigint *a, uint64_t k) {
    if (x->rc || x->why || !a) return NULL;
    int32_t e = 0; tr_bigint *r = tr_bigint_pow(a, k, x->b, x->r, &e);
    if (!r) big_fail(x, e);
    return r;
}
static tr_bigint *z_horner(ctx *x, tr_bigint *const *c, uint32_t n, const tr_bigint *v) {
    if (x->rc || x->why || !v) return NULL;
    int32_t e = 0; tr_bigint *r = tr_bigint_horner((const tr_bigint *const *)c, n, v, x->b, x->r, &e);
    if (!r) big_fail(x, e);
    return r;
}
static tr_bigq *q_of(ctx *x, const tr_bigint *a) {
    if (x->rc || x->why || !a) return NULL;
    int32_t e = 0; tr_bigq *r = tr_bigq_from_bigint(a, x->b, x->r, &e);
    if (!r) big_fail(x, e);
    return r;
}
/* floor (up = 0) or ceil (up = 1) of a big rational, as int128 clamped to [lo - 1, hi + 1] */
static __int128 q_round(ctx *x, const tr_bigq *t, int up, __int128 lo, __int128 hi) {
    if (x->rc || x->why || !t) return lo - 1;
    tr_bigint *q = NULL, *r = NULL;
    int32_t e = tr_bigint_divmod(tr_bigq_numerator(t), tr_bigq_denominator(t), TR_DIV_FLOOR, &q, &r, x->b, x->r);
    if (e) { big_fail(x, e); return lo - 1; }
    tr_math_i128 w;
    __int128 v;
    if (tr_bigint_to_i128(q, &w)) v = tr_bigint_sign(q) < 0 ? lo - 1 : hi + 1;
    else {
        v = tr_i128_get(w);
        if (up && tr_bigint_sign(r) != 0) v++;
        if (v < lo - 1) v = lo - 1;
        if (v > hi + 1) v = hi + 1;
    }
    tr_bigint_free(q); tr_bigint_free(r);
    return v;
}
/* x as W words of two's complement (|x| < 2^(64 W - 1) proven by the caller) */
static int32_t to_words(const tr_bigint *a, int32_t W, uint64_t *out) {
    int32_t sign = 0;
    uint64_t limbs[TR_PREIMG_MAX_WORDS + 1], count = 0;
    for (int32_t i = 0; i < W; i++) out[i] = 0;
    if (tr_bigint_limb_count(a) > (uint64_t)W) return TR_MATH_OVERFLOW;
    int32_t rc = tr_bigint_export(a, &sign, limbs, TR_PREIMG_MAX_WORDS + 1, &count);
    if (rc) return rc;
    for (uint64_t i = 0; i < count && i < (uint64_t)W; i++) out[i] = limbs[i];
    if (sign < 0) tr_preimg_w_neg(out, W);
    return TR_MATH_OK;
}

/* ================================================================ integer interval lists */
typedef struct { __int128 lo, hi; } ziv;
typedef struct { ziv *v; uint64_t n, cap; } zlist;
static int zl_push(ctx *x, zlist *l, __int128 lo, __int128 hi) {
    if (x->rc) return 0;
    if (lo > hi) return 1;
    if (l->n == l->cap) {
        uint64_t cap = l->cap ? 2 * l->cap : 16;
        ziv *v = (ziv *)realloc(l->v, (size_t)cap * sizeof *v);
        if (!v) { hard(x, TR_MATH_MEMORY); return 0; }
        l->v = v; l->cap = cap;
    }
    l->v[l->n].lo = lo; l->v[l->n].hi = hi; l->n++;
    return 1;
}
static int cmp_ziv(const void *p, const void *q) {
    const ziv *a = (const ziv *)p, *b = (const ziv *)q;
    return a->lo < b->lo ? -1 : (a->lo > b->lo ? 1 : 0);
}
static void zl_merge(zlist *l) {                       /* sort, merge overlapping and adjacent intervals */
    if (!l->n) return;
    qsort(l->v, (size_t)l->n, sizeof *l->v, cmp_ziv);
    uint64_t k = 0;
    for (uint64_t i = 0; i < l->n; i++) {
        if (k && l->v[i].lo <= l->v[k - 1].hi + 1) { if (l->v[i].hi > l->v[k - 1].hi) l->v[k - 1].hi = l->v[i].hi; }
        else l->v[k++] = l->v[i];
    }
    l->n = k;
}
static int cmp_u64(const void *p, const void *q) {
    uint64_t a = *(const uint64_t *)p, b = *(const uint64_t *)q;
    return a < b ? -1 : a > b;
}

/* ================================================================ targets of integer-valued maps */
/* the values y in [0, ymax] with y op c: up to two intervals, or (EQ / IN) a sorted list of integer points */
static void int_targets(ctx *x, const tr_pre_predicate *pr, const tr_num_rational *vals, uint64_t nvals, __int128 ymax,
                        zlist *out) {
    switch (pr->op) {
    case TR_PRE_LT: zl_push(x, out, 0, ceil_q(pr->c) - 1 < ymax ? ceil_q(pr->c) - 1 : ymax); break;
    case TR_PRE_LE: zl_push(x, out, 0, floor_q(pr->c) < ymax ? floor_q(pr->c) : ymax); break;
    case TR_PRE_GT: zl_push(x, out, floor_q(pr->c) + 1 > 0 ? floor_q(pr->c) + 1 : 0, ymax); break;
    case TR_PRE_GE: zl_push(x, out, ceil_q(pr->c) > 0 ? ceil_q(pr->c) : 0, ymax); break;
    case TR_PRE_EQ:
        if (pr->c.denominator == 1 && pr->c.numerator >= 0 && pr->c.numerator <= ymax) zl_push(x, out, pr->c.numerator, pr->c.numerator);
        break;
    case TR_PRE_NE:
        if (pr->c.denominator == 1 && pr->c.numerator >= 0 && pr->c.numerator <= ymax) {
            zl_push(x, out, 0, (__int128)pr->c.numerator - 1);
            zl_push(x, out, (__int128)pr->c.numerator + 1, ymax);
        } else zl_push(x, out, 0, ymax);
        break;
    default:                                          /* IN: the integral values in range, sorted and distinct */
        for (uint64_t i = 0; i < nvals; i++)
            if (vals[i].denominator == 1 && vals[i].numerator >= 0 && vals[i].numerator <= ymax)
                zl_push(x, out, vals[i].numerator, vals[i].numerator);
        break;
    }
}

/* ================================================================ residues to branches */
/* runs of L residues mod M (sorted, disjoint) shifted to code residues (code = L - offset) and merged */
static void shift_runs(ctx *x, const zlist *runs, int64_t M, int64_t offset, zlist *out) {
    const int64_t off = fmod64(offset, M);
    for (uint64_t i = 0; i < runs->n && !x->rc; i++) {
        __int128 a = runs->v[i].lo - off, b = runs->v[i].hi - off;
        if (a < 0 && b < 0) { a += M; b += M; }
        if (a < 0) { zl_push(x, out, a + M, (__int128)M - 1); zl_push(x, out, 0, b); }
        else zl_push(x, out, a, b);
    }
    zl_merge(out);
}
/* exact count of the codes of [lo, hi] with code mod M in [rlo, rhi] */
static uint64_t count_residue(__int128 lo, __int128 hi, __int128 M, __int128 rlo, __int128 rhi) {
    __int128 N = hi - lo + 1, q = N / M, rem = N % M, s = lo % M;
    if (s < 0) s += M;
    __int128 c = q * (rhi - rlo + 1);
    for (int k = 0; k < 2; k++) {
        __int128 a = s, b = s + rem - 1, l = rlo + k * M, h = rhi + k * M;
        __int128 x0 = a > l ? a : l, x1 = b < h ? b : h;
        if (rem > 0 && x0 <= x1) c += x1 - x0 + 1;
    }
    return c > (__int128)UINT64_MAX ? UINT64_MAX : (uint64_t)c;
}
/* the codes x of [0, n] with x & mask == value (value within mask), saturating */
static uint64_t count_cube_upto(__int128 n, uint64_t mask, uint64_t value) {
    if (n < 0) return 0;
    if (n > (__int128)INT64_MAX) n = INT64_MAX;
    uint64_t N = (uint64_t)n, total = 0;
    for (int bit = 63; bit >= 0; bit--) {
        uint64_t nb = (N >> bit) & 1, mb = (mask >> bit) & 1, vb = (value >> bit) & 1;
        uint64_t below = bit ? (((uint64_t)1 << bit) - 1) : 0;
        int freeb = __builtin_popcountll(~mask & below);
        if (nb) {
            if (!mb || !vb) {                          /* this bit 0: every lower assignment is below N */
                uint64_t add = freeb >= 64 ? UINT64_MAX : ((uint64_t)1 << freeb);
                total = total > UINT64_MAX - add ? UINT64_MAX : total + add;
            }
            if (mb && !vb) return total;               /* the tight path needs a 1 the cube forbids */
        } else if (mb && vb) return total;
    }
    return total == UINT64_MAX ? total : total + 1;   /* x == N itself */
}
static uint64_t sat_add(uint64_t a, uint64_t b) { return tr_budget_add(a, b); }

typedef struct {                      /* the exact union in one representation */
    int32_t repr;                      /* TR_PREIMG_INTERVALS / _RESIDUES / _CUBES */
    int64_t modulus;
    zlist iv;                          /* INTERVALS: codes; RESIDUES: code residue runs */
    uint64_t *mask, *value;            /* CUBES */
    uint64_t ncubes, capcubes;
    int built;                         /* 1: the union is exact and complete */
} uset;
static void uset_free(uset *u) { free(u->iv.v); free(u->mask); free(u->value); memset(u, 0, sizeof *u); }
/* code residue runs mod M over the domain: INTERVALS when the domain is no larger than M, else RESIDUES */
static void residues_to_set(ctx *x, const zlist *code_runs, int64_t M, const tr_pre_codec *cd, uset *u) {
    const __int128 size = (__int128)cd->hi - cd->lo + 1;
    if (code_runs->n == 1 && code_runs->v[0].lo == 0 && code_runs->v[0].hi == (__int128)M - 1) {   /* every residue */
        u->repr = TR_PREIMG_INTERVALS;
        zl_push(x, &u->iv, cd->lo, cd->hi);
        return;
    }
    if (size <= M) {
        u->repr = TR_PREIMG_INTERVALS;
        __int128 base = (__int128)cd->lo - fmod64(cd->lo, M);
        for (uint64_t i = 0; i < code_runs->n && !x->rc; i++)
            for (int w = 0; w < 2; w++) {
                __int128 a = base + (__int128)w * M + code_runs->v[i].lo, z = base + (__int128)w * M + code_runs->v[i].hi;
                if (a < cd->lo) a = cd->lo;
                if (z > cd->hi) z = cd->hi;
                zl_push(x, &u->iv, a, z);
            }
        zl_merge(&u->iv);
        return;
    }
    u->repr = TR_PREIMG_RESIDUES; u->modulus = M;
    for (uint64_t i = 0; i < code_runs->n && !x->rc; i++) zl_push(x, &u->iv, code_runs->v[i].lo, code_runs->v[i].hi);
}
/* runs of the set bits of a bitmap of M bits */
static void bitmap_runs(ctx *x, const uint64_t *bm, uint64_t M, zlist *out) {
    uint64_t i = 0;
    while (i < M && !x->rc) {
        uint64_t w = i >> 6, bits = bm[w] >> (i & 63);
        if (!bits) { i = (w + 1) << 6; continue; }
        i += (uint64_t)__builtin_ctzll(bits);
        if (i >= M) break;
        uint64_t s = i;
        while (i < M && ((bm[i >> 6] >> (i & 63)) & 1)) {
            uint64_t ww = bm[i >> 6] >> (i & 63);
            if (ww == (UINT64_MAX >> (i & 63))) i = ((i >> 6) + 1) << 6; else i += (uint64_t)__builtin_ctzll(~ww);
        }
        if (i > M) i = M;
        zl_push(x, out, s, (__int128)i - 1);
    }
}

/* ================================================================ POLY: monotone pieces and level sets */
typedef struct {
    ctx *x;
    int32_t d, W;
    tr_bigint **n;                     /* d + 1 coefficients of N */
    tr_bigint **R;                     /* nt thresholds K num(c_j) */
    int64_t *den;                      /* nt denominators */
    uint64_t nt;
    const uint64_t *wcoef, *wthr;      /* W-word forms (W > 0) */
    tr_preimg_view *v;                 /* the view the W-word evaluation reads */
} pctx;
/* sign(N(L) den_j - R_j) (-1, 0, 1); -2 when the budget or an error stopped it */
static int poly_sign(pctx *p, __int128 L, uint64_t j) {
    if (!step(p->x, 1)) return -2;
    if (p->W) {
        uint64_t nv[TR_PREIMG_MAX_WORDS];
        tr_preimg_poly_value(p->v, (int64_t)L, nv);
        return tr_preimg_poly_cmp(nv, p->W, p->den[j], p->wthr + j * (uint64_t)p->W);
    }
    tr_bigint *Lb = bi128(p->x, L), *N = Lb ? z_horner(p->x, p->n, (uint32_t)p->d + 1, Lb) : NULL;
    tr_bigint *db = bi64(p->x, p->den[j]), *Nd = z_mul(p->x, N, db), *diff = z_sub(p->x, Nd, p->R[j]);
    int s = diff ? tr_bigint_sign(diff) : -2;
    tr_bigint_free(Lb); tr_bigint_free(N); tr_bigint_free(db); tr_bigint_free(Nd); tr_bigint_free(diff);
    return s;
}
/* N(e) > N(s) (the direction of a strictly monotone piece); -2 on a stop */
static int poly_increasing(pctx *p, __int128 s, __int128 e) {
    if (!step(p->x, 2)) return -2;
    if (p->W) {
        uint64_t a[TR_PREIMG_MAX_WORDS], b[TR_PREIMG_MAX_WORDS];
        tr_preimg_poly_value(p->v, (int64_t)s, a);
        tr_preimg_poly_value(p->v, (int64_t)e, b);
        tr_preimg_w_neg(a, p->W);
        tr_preimg_w_add(b, a, p->W);
        return tr_preimg_w_sign(b, p->W) > 0;
    }
    tr_bigint *Ls = bi128(p->x, s), *Le = bi128(p->x, e);
    tr_bigint *Ns = Ls ? z_horner(p->x, p->n, (uint32_t)p->d + 1, Ls) : NULL, *Ne = Le ? z_horner(p->x, p->n, (uint32_t)p->d + 1, Le) : NULL;
    int r = (Ns && Ne) ? tr_bigint_cmp(Ne, Ns) > 0 : -2;
    tr_bigint_free(Ls); tr_bigint_free(Le); tr_bigint_free(Ns); tr_bigint_free(Ne);
    return r;
}
/* the first L in [s, e] with (inc ? sign >= k : sign <= k), e + 1 when none (the predicate is monotone) */
static __int128 first_level(pctx *p, __int128 s, __int128 e, uint64_t j, int inc, int k) {
    __int128 lo = s, hi = e + 1;
    while (lo < hi) {
        __int128 mid = lo + (hi - lo) / 2;
        int sg = poly_sign(p, mid, j);
        if (sg == -2) return e + 1;
        int ok = inc ? sg >= k : sg <= k;
        if (ok) hi = mid; else lo = mid + 1;
    }
    return lo;
}
/* the allowed signs of an op: bit 0 negative, bit 1 zero, bit 2 positive */
static int op_signs(int32_t op) {
    switch (op) {
    case TR_PRE_LT: return 1; case TR_PRE_LE: return 3; case TR_PRE_EQ: return 2; case TR_PRE_NE: return 5;
    case TR_PRE_GE: return 6; case TR_PRE_GT: return 4; default: return 2;
    }
}
static void piece_levels(pctx *p, __int128 s, __int128 e, int32_t op, zlist *out) {
    ctx *x = p->x;
    const int allowed = op_signs(op);
    if (s == e) {
        for (uint64_t j = 0; j < p->nt && !x->rc && !x->why; j++) {
            int sg = poly_sign(p, s, j);
            if (sg == -2) return;
            if ((allowed >> (sg + 1)) & 1) zl_push(x, out, s, s);
        }
        return;
    }
    int inc = poly_increasing(p, s, e);
    if (inc == -2) return;
    for (uint64_t j = 0; j < p->nt && !x->rc && !x->why; j++) {
        __int128 a, b;
        if (inc) { a = first_level(p, s, e, j, 1, 0); b = first_level(p, a, e, j, 1, 1); }
        else { a = first_level(p, s, e, j, 0, 0); b = first_level(p, a, e, j, 0, -1); }
        if (x->rc || x->why) return;
        const int first = inc ? 0 : 2, last = inc ? 2 : 0;             /* the sign bits of [s, a), [b, e] */
        if ((allowed >> first) & 1) zl_push(x, out, s, a - 1);
        if ((allowed >> 1) & 1) zl_push(x, out, a, b - 1);
        if ((allowed >> last) & 1) zl_push(x, out, b, e);
    }
}

/* ================================================================ the set */
void tr_preimg_set_free(tr_preimg_set *s) {
    if (!set_valid(s)) return;
    s->magic = 0;
    free(s->branch); free(s->lookup); free(s->icoef); free(s->wcoef); free(s->wthreshold); free(s->values);
    tr_cert_free(s->cert);
    free(s);
}
static int cmp_q(const void *p, const void *q) { return tr_q_cmp(*(const tr_num_rational *)p, *(const tr_num_rational *)q); }
static int32_t check_input(const tr_pre_codec *cd, const tr_preimg_map *m, const tr_pre_predicate *pr, const tr_preimg_spec *sp) {
    if (!cd || !m || !pr || cd->scale < 0 || cd->scale > TR_SCOPE_MAX_SCALE || cd->lo > cd->hi || pr->reserved) return TR_MATH_INVALID;
    if (sp && (sp->fallback < TR_PREIMG_FALLBACK_AUTO || sp->fallback > TR_PREIMG_FALLBACK_NONE || sp->reserved)) return TR_MATH_INVALID;
    if (pr->op < TR_PRE_LT || pr->op > TR_PRE_IN) return TR_MATH_INVALID;
    if (pr->op == TR_PRE_IN) {
        if (pr->nvalues > TR_PREIMG_MAX_VALUES) return TR_MATH_UNSUPPORTED;
        if (pr->nvalues && !pr->values) return TR_MATH_INVALID;
        for (uint64_t i = 0; i < pr->nvalues; i++) if (!canonical(pr->values[i])) return TR_MATH_INVALID;
    } else if (!canonical(pr->c)) return TR_MATH_INVALID;
    if (m->kind == TR_PREIMG_POLY || m->kind == TR_PREIMG_POLYMOD) {
        if (m->degree < 0 || !m->coef) return TR_MATH_INVALID;
        if (m->degree > TR_PREIMG_MAX_DEGREE) return TR_MATH_UNSUPPORTED;
        for (int32_t i = 0; i <= m->degree; i++) {
            if (!canonical(m->coef[i])) return TR_MATH_INVALID;
            if (m->kind == TR_PREIMG_POLYMOD && m->coef[i].denominator != 1) return TR_MATH_INVALID;
        }
        if (m->kind == TR_PREIMG_POLY) return TR_MATH_OK;
    }
    switch (m->kind) {
    case TR_PREIMG_POLYMOD:
    case TR_PREIMG_AFFMOD:
        if (m->modulus < 0) return TR_MATH_INVALID;
        if (m->modulus == 0) return TR_MATH_DOMAIN_ERROR;          /* a zero divisor */
        if (m->modulus > TR_PREIMG_MAX_MODULUS || cd->scale) return TR_MATH_UNSUPPORTED;
        return TR_MATH_OK;
    case TR_PREIMG_XOR: case TR_PREIMG_AND: case TR_PREIMG_OR:
        if (m->width < 1 || (m->base_is_offset != 0 && m->base_is_offset != 1)) return TR_MATH_INVALID;
        if (m->width > 63 || cd->lo < 0) return TR_MATH_UNSUPPORTED;
        if (m->b < 0 || (uint64_t)m->b >> m->width) return TR_MATH_INVALID;
        return TR_MATH_OK;
    default:
        return TR_MATH_INVALID;
    }
}

/* POLY: N, K, thresholds, the W-word forms and the exact union of level sets */
typedef struct {
    tr_bigint *n[TR_PREIMG_MAX_DEGREE + 1], *K;
    tr_bigint **R;
    int64_t *den;
    uint64_t nt;
    int32_t d;
} polydata;
static void poly_free(polydata *pd) {
    for (int32_t i = 0; i <= TR_PREIMG_MAX_DEGREE; i++) tr_bigint_free(pd->n[i]);
    tr_bigint_free(pd->K);
    if (pd->R) for (uint64_t j = 0; j < pd->nt; j++) tr_bigint_free(pd->R[j]);
    free(pd->R); free(pd->den);
    memset(pd, 0, sizeof *pd);
}
/* N(L) = K p(L / T) with K = T^d lcm(den a_i); R_j = K num(c_j) */
static void poly_prepare(ctx *x, const tr_pre_codec *cd, const tr_preimg_map *m, const tr_pre_predicate *pr,
                         const tr_num_rational *vals, uint64_t nvals, polydata *pd) {
    int32_t d = m->degree;
    while (d > 0 && m->coef[d].numerator == 0) d--;
    pd->d = d;
    tr_bigint *T = bi64(x, (int64_t)tr_unc_pow10(cd->scale)), *L = bi64(x, 1);
    for (int32_t i = 0; i <= d && !x->rc && !x->why; i++) {          /* lcm of the denominators */
        tr_bigint *di = bi64(x, m->coef[i].denominator), *g = z_gcd(x, L, di), *q = z_divexact(x, di, g), *l2 = z_mul(x, L, q);
        tr_bigint_free(di); tr_bigint_free(g); tr_bigint_free(q);
        if (l2) { tr_bigint_free(L); L = l2; }
    }
    tr_bigint *Td = z_pow(x, T, (uint64_t)d);
    pd->K = z_mul(x, Td, L);
    for (int32_t i = 0; i <= d && !x->rc && !x->why; i++) {          /* n_i = a_i K / T^i = num_i (L / den_i) T^(d - i) */
        tr_bigint *num = bi64(x, m->coef[i].numerator), *di = bi64(x, m->coef[i].denominator), *ld = z_divexact(x, L, di);
        tr_bigint *Tp = z_pow(x, T, (uint64_t)(d - i)), *a = z_mul(x, num, ld);
        pd->n[i] = z_mul(x, a, Tp);
        tr_bigint_free(num); tr_bigint_free(di); tr_bigint_free(ld); tr_bigint_free(Tp); tr_bigint_free(a);
    }
    tr_bigint_free(T); tr_bigint_free(L); tr_bigint_free(Td);
    pd->nt = pr->op == TR_PRE_IN ? nvals : 1;
    pd->R = (tr_bigint **)zalloc((pd->nt ? pd->nt : 1) * sizeof *pd->R);
    pd->den = (int64_t *)zalloc((pd->nt ? pd->nt : 1) * 8);
    if (!pd->R || !pd->den) { hard(x, TR_MATH_MEMORY); return; }
    for (uint64_t j = 0; j < pd->nt && !x->rc && !x->why; j++) {
        tr_num_rational c = pr->op == TR_PRE_IN ? vals[j] : pr->c;
        tr_bigint *num = bi64(x, c.numerator);
        pd->R[j] = z_mul(x, pd->K, num);
        pd->den[j] = c.denominator;
        tr_bigint_free(num);
    }
}
/* the words of the direct evaluation over the domain's L range (0 when past TR_PREIMG_MAX_WORDS) */
static int32_t poly_words(ctx *x, const polydata *pd, __int128 Llo, __int128 Lhi) {
    __int128 Mabs = Llo < 0 ? -Llo : Llo, Mh = Lhi < 0 ? -Lhi : Lhi;
    if (Mh > Mabs) Mabs = Mh;
    if (Mabs < 1) Mabs = 1;
    tr_bigint *M = bi128(x, Mabs), *B = bi64(x, 0);
    for (int32_t i = pd->d; i >= 0 && !x->rc && !x->why; i--) {      /* B = sum |n_i| M^i by Horner on magnitudes */
        tr_bigint *t = z_mul(x, B, M);
        int32_t e = 0;
        tr_bigint *ai = pd->n[i] ? tr_bigint_abs(pd->n[i], x->b, x->r, &e) : NULL;
        if (!ai) big_fail(x, e);
        tr_bigint *s = z_add(x, t, ai);
        tr_bigint_free(t); tr_bigint_free(ai);
        if (s) { tr_bigint_free(B); B = s; }
    }
    uint64_t bits = B ? tr_bigint_bit_length(B) : 0, maxden = 1, rbits = 0;
    for (uint64_t j = 0; j < pd->nt; j++) {
        if ((uint64_t)pd->den[j] > maxden) maxden = (uint64_t)pd->den[j];
        uint64_t rb = pd->R[j] ? tr_bigint_bit_length(pd->R[j]) : 0;
        if (rb > rbits) rbits = rb;
    }
    tr_bigint_free(M); tr_bigint_free(B);
    if (x->rc || x->why) return 0;
    uint64_t need = bits + bitlen(maxden);
    if (rbits > need) need = rbits;
    need += 3;                                                       /* the products' and differences' margin */
    for (int32_t W = 1; W <= TR_PREIMG_MAX_WORDS; W++) if (need <= (uint64_t)(64 * W - 1)) return W;
    return 0;
}
/* the exact union of a POLY preimage (L space, then codes) */
static void poly_union(ctx *x, const tr_pre_codec *cd, int32_t op, polydata *pd, pctx *p, uset *u, int32_t *npieces,
                       int32_t *ncritical) {
    const __int128 Llo = (__int128)cd->lo + cd->offset, Lhi = (__int128)cd->hi + cd->offset;
    zlist lv = {NULL, 0, 0};
    u->repr = TR_PREIMG_INTERVALS;
    if (pd->d == 0) {                                     /* a constant map: the predicate on every code or none */
        int any = 0;
        for (uint64_t j = 0; j < pd->nt && !x->rc && !x->why; j++) {
            tr_bigint *db = bi64(x, pd->den[j]), *Nd = z_mul(x, pd->n[0], db), *diff = z_sub(x, Nd, pd->R[j]);
            if (diff && ((op_signs(op) >> (tr_bigint_sign(diff) + 1)) & 1)) any = 1;
            tr_bigint_free(db); tr_bigint_free(Nd); tr_bigint_free(diff);
        }
        if (any) zl_push(x, &u->iv, cd->lo, cd->hi);
        u->built = !x->rc && !x->why;
        *npieces = 1;
        return;
    }
    /* pieces: the integer runs between the critical points (INV-001's isolation of N' on [L_lo, L_hi]) */
    zlist pieces = {NULL, 0, 0};
    __int128 cur = Llo;
    tr_inv_roots *roots = NULL, *ref = NULL;
    if (pd->d >= 2) {
        tr_bigq *coef[TR_PREIMG_MAX_DEGREE];
        memset(coef, 0, sizeof coef);
        for (int32_t i = 1; i <= pd->d && !x->rc && !x->why; i++) {
            tr_bigint *ib = bi64(x, i), *c = z_mul(x, pd->n[i], ib);
            coef[i - 1] = q_of(x, c);
            tr_bigint_free(ib); tr_bigint_free(c);
        }
        tr_bigint *lob = bi128(x, Llo), *hib = bi128(x, Lhi);
        tr_bigq *lq = q_of(x, lob), *hq = q_of(x, hib);
        tr_bigint_free(lob); tr_bigint_free(hib);
        int32_t e = 0;
        if (!x->rc && !x->why) {
            roots = tr_inv_isolate(pd->d, (const tr_bigq *const *)coef, NULL, lq, hq, x->b, x->r, &e);
            if (!roots) big_fail(x, e);
        }
        if (roots) {
            tr_bigint *one = bi64(x, 1);
            tr_bigq *w = q_of(x, one);
            tr_bigint_free(one);
            if (w) { ref = tr_inv_roots_refine(roots, w, NULL, x->b, x->r, &e); if (!ref) big_fail(x, e); }
            tr_bigq_free(w);
        }
        for (int32_t i = 0; i < TR_PREIMG_MAX_DEGREE; i++) tr_bigq_free(coef[i]);
        tr_bigq_free(lq); tr_bigq_free(hq);
        tr_inv_roots_info ri;
        if (ref && !tr_inv_roots_get_info(ref, &ri)) {
            *ncritical = ri.nroots;
            for (int32_t i = 0; i < ri.nroots && !x->rc && !x->why; i++) {
                tr_inv_root_info r;
                const tr_bigq *a = NULL, *b = NULL;
                if (tr_inv_roots_root(ref, i, &r, &a, &b)) { hard(x, TR_MATH_INVALID); break; }
                if (r.exact) {
                    __int128 s = q_round(x, a, 0, Llo, Lhi);
                    if (s >= cur) { zl_push(x, &pieces, cur, s > Lhi ? Lhi : s); cur = s + 1; }
                } else {
                    __int128 fa = q_round(x, a, 0, Llo, Lhi), cb = q_round(x, b, 1, Llo, Lhi);
                    if (fa >= cur) { zl_push(x, &pieces, cur, fa > Lhi ? Lhi : fa); cur = fa + 1; }
                    for (__int128 k = cur; k <= cb - 1 && k <= Lhi; k++) zl_push(x, &pieces, k, k);   /* at most one */
                    if (cb > cur) cur = cb;
                }
            }
        }
    }
    if (!x->rc && !x->why && cur <= Lhi) zl_push(x, &pieces, cur, Lhi);
    tr_inv_roots_free(roots); tr_inv_roots_free(ref);
    *npieces = (int32_t)pieces.n;
    for (uint64_t i = 0; i < pieces.n && !x->rc && !x->why; i++) {
        __int128 s = pieces.v[i].lo < Llo ? Llo : pieces.v[i].lo, e = pieces.v[i].hi > Lhi ? Lhi : pieces.v[i].hi;
        if (s <= e) piece_levels(p, s, e, op, &lv);
    }
    free(pieces.v);
    for (uint64_t i = 0; i < lv.n && !x->rc; i++) zl_push(x, &u->iv, lv.v[i].lo - cd->offset, lv.v[i].hi - cd->offset);
    free(lv.v);
    zl_merge(&u->iv);
    u->built = !x->rc && !x->why;
}

/* POLYMOD residues: CRT over the prime powers of m (EQ / IN), else every residue enumerated */
static uint64_t mulmod(uint64_t a, uint64_t b, uint64_t m) { return (uint64_t)(((unsigned __int128)a * b) % m); }
static uint64_t inv_mod(uint64_t a, uint64_t m) {                 /* a^-1 mod m (gcd 1), m >= 1 */
    __int128 t = 0, nt = 1, r = (__int128)m, nr = (__int128)(a % m);
    while (nr) { __int128 q = r / nr, tmp = t - q * nt; t = nt; nt = tmp; tmp = r - q * nr; r = nr; nr = tmp; }
    if (t < 0) t += m;
    return m == 1 ? 0 : (uint64_t)t;
}
static uint64_t poly_mod(const int64_t *c, int32_t d, uint64_t r, uint64_t q) {   /* P(r) mod q, coefficients any sign */
    unsigned __int128 acc = (uint64_t)fmod64(c[d], (int64_t)q);
    for (int32_t i = d; i-- > 0;) acc = ((acc * r) % q + (uint64_t)fmod64(c[i], (int64_t)q)) % q;
    return (uint64_t)acc;
}
static void polymod_union(ctx *x, const tr_pre_codec *cd, const tr_pre_predicate *pr, const tr_num_rational *vals,
                          uint64_t nvals, const int64_t *coef, int32_t d, const tr_preimg_view *v, uset *u) {
    const uint64_t m = (uint64_t)v->m;
    zlist Y = {NULL, 0, 0}, runs = {NULL, 0, 0}, crun = {NULL, 0, 0};
    int_targets(x, pr, vals, nvals, (__int128)m - 1, &Y);
    const int eq = pr->op == TR_PRE_EQ || pr->op == TR_PRE_IN;
    if (!x->rc && eq) {
        /* the prime powers of m by trial division (iterations), each enumerated, the classes combined by CRT */
        uint64_t q[64], rest = m;
        int32_t nq = 0;
        for (uint64_t p = 2; p <= rest / p && !x->why; p += (p == 2 ? 1 : 2)) {
            if (!step(x, 1)) break;
            if (rest % p) continue;
            uint64_t qq = 1;
            while (rest % p == 0) { rest /= p; qq *= p; }
            q[nq++] = qq;
        }
        if (!x->why && rest > 1) q[nq++] = rest;
        for (int32_t i = 0; i < nq && !x->why; i++)
            if (q[i] > TR_PREIMG_MAX_PERIOD) soft(x, TR_PREIMG_WHY_PERIOD, TR_BUDGET_ITERATIONS, q[i], TR_PREIMG_MAX_PERIOD);
        uint64_t total = 0;
        for (int32_t i = 0; i < nq && !x->why; i++) total += q[i];
        if (!x->why && !x->rc && step(x, total)) {
            uint64_t *table = (uint64_t *)zalloc(total * 8), *e = (uint64_t *)zalloc((uint64_t)(nq ? nq : 1) * 8);
            uint64_t off[64], *res = NULL, nres = 0, capres = 0;
            if (!table || !e) hard(x, TR_MATH_MEMORY);
            for (int32_t i = 0, at = 0; i < nq && !x->rc; i++) {
                off[i] = (uint64_t)at;
                for (uint64_t r = 0; r < q[i]; r++) table[at + r] = poly_mod(coef, d, r, q[i]);
                at += (int32_t)q[i];
                uint64_t mi = m / q[i];                          /* e_i = (m / q_i) ((m / q_i)^-1 mod q_i) mod m */
                e[i] = mulmod(mi % m, inv_mod(mi % q[i], q[i]), m);
            }
            const uint64_t cap = x->b->limit[TR_BUDGET_BYTES] / 8 < TR_PREIMG_MAX_LOOKUP ? x->b->limit[TR_BUDGET_BYTES] / 8 : TR_PREIMG_MAX_LOOKUP;
            for (uint64_t t = 0; t < Y.n && !x->rc && !x->why; t++) {
                const uint64_t y = (uint64_t)Y.v[t].lo;
                /* the roots of P = y mod q_i, then every combination */
                uint64_t *roots[64], nroots[64];
                uint64_t combos = 1;
                for (int32_t i = 0; i < nq; i++) {
                    roots[i] = NULL; nroots[i] = 0;
                    for (uint64_t r = 0; r < q[i]; r++) if (table[off[i] + r] == y % q[i]) nroots[i]++;
                    combos = nroots[i] && combos <= UINT64_MAX / nroots[i] ? combos * nroots[i] : (nroots[i] ? UINT64_MAX : 0);
                }
                if (!combos) continue;
                if (combos > cap || nres + combos > cap) { soft(x, TR_PREIMG_WHY_CLASSES, TR_BUDGET_BYTES, sat_add(nres, combos) * 8, cap * 8); break; }
                if (!step(x, combos)) break;
                for (int32_t i = 0; i < nq && !x->rc; i++) {
                    roots[i] = (uint64_t *)zalloc(nroots[i] * 8);
                    if (!roots[i]) { hard(x, TR_MATH_MEMORY); break; }
                    uint64_t k = 0;
                    for (uint64_t r = 0; r < q[i]; r++) if (table[off[i] + r] == y % q[i]) roots[i][k++] = r;
                }
                if (nres + combos > capres && !x->rc) {
                    uint64_t nc = capres ? capres : 64;
                    while (nc < nres + combos) nc *= 2;
                    uint64_t *t2 = (uint64_t *)realloc(res, (size_t)nc * 8);
                    if (!t2) hard(x, TR_MATH_MEMORY); else { res = t2; capres = nc; }
                }
                uint64_t digit[64] = {0};
                for (uint64_t c = 0; c < combos && !x->rc; c++) {
                    uint64_t r = 0;
                    for (int32_t i = 0; i < nq; i++) r = (r + mulmod(roots[i][digit[i]], e[i], m)) % m;
                    res[nres++] = nq ? r : 0;
                    for (int32_t i = 0; i < nq; i++) { if (++digit[i] < nroots[i]) break; digit[i] = 0; }
                }
                for (int32_t i = 0; i < nq; i++) free(roots[i]);
            }
            if (!x->rc && !x->why) {
                if (nres) qsort(res, (size_t)nres, 8, cmp_u64);     /* res is NULL when no residue solves */
                for (uint64_t i = 0; i < nres; i++) {
                    if (runs.n && (__int128)res[i] <= runs.v[runs.n - 1].hi + 1) {
                        if ((__int128)res[i] > runs.v[runs.n - 1].hi) runs.v[runs.n - 1].hi = res[i];
                    } else zl_push(x, &runs, res[i], res[i]);
                }
            }
            free(res); free(table); free(e);
        }
    } else if (!x->rc) {
        if (m > TR_PREIMG_MAX_PERIOD) soft(x, TR_PREIMG_WHY_PERIOD, TR_BUDGET_ITERATIONS, m, TR_PREIMG_MAX_PERIOD);
        else if (step(x, m)) {
            uint64_t *bm = (uint64_t *)zalloc(((m + 63) / 64) * 8);
            if (!bm) hard(x, TR_MATH_MEMORY);
            for (uint64_t r = 0; bm && r < m; r++)
                if (tr_preimg_int_holds(v, tr_preimg_int_value(v, (int64_t)r))) bm[r >> 6] |= (uint64_t)1 << (r & 63);
            if (bm) bitmap_runs(x, bm, m, &runs);
            free(bm);
        }
    }
    if (!x->rc && !x->why) {
        shift_runs(x, &runs, (int64_t)m, cd->offset, &crun);
        if (crun.n) residues_to_set(x, &crun, (int64_t)m, cd, u); else u->repr = TR_PREIMG_INTERVALS;
        u->built = !x->rc;
    }
    free(Y.v); free(runs.v); free(crun.v);
}

/* AFFMOD residues of the period M = m / gcd(a, m): one class per solvable y */
static void affmod_union(ctx *x, const tr_pre_codec *cd, const tr_pre_predicate *pr, const tr_num_rational *vals,
                         uint64_t nvals, const tr_preimg_view *v, int64_t *period, uset *u) {
    const uint64_t m = (uint64_t)v->m, a = (uint64_t)v->a, b = (uint64_t)v->b;
    zlist Y = {NULL, 0, 0}, runs = {NULL, 0, 0}, crun = {NULL, 0, 0};
    int_targets(x, pr, vals, nvals, (__int128)m - 1, &Y);
    if (a == 0) {                                          /* a constant map */
        int any = 0;
        for (uint64_t i = 0; i < Y.n; i++) any |= (__int128)b >= Y.v[i].lo && (__int128)b <= Y.v[i].hi;
        u->repr = TR_PREIMG_INTERVALS;
        if (any) zl_push(x, &u->iv, cd->lo, cd->hi);
        *period = 1;
        u->built = !x->rc;
        free(Y.v);
        return;
    }
    const uint64_t g = tr_gcd_u64(a, m), M = m / g, inv = inv_mod(a / g, M);
    *period = (int64_t)M;
    /* the solvable y = b + g t of each target interval: residues t inv mod M */
    uint64_t count = 0;
    for (uint64_t i = 0; i < Y.n; i++) {
        __int128 y0 = Y.v[i].lo, y1 = Y.v[i].hi, first = y0 + fmod64((__int128)b - y0, (int64_t)g);
        if (first <= y1) count += (uint64_t)((y1 - first) / g + 1);
    }
    if (count >= M) {                                      /* every residue (count <= M solvable values cover all) */
        int covers = 0;
        for (uint64_t i = 0; i < Y.n; i++) covers |= Y.v[i].lo == 0 && Y.v[i].hi == (__int128)m - 1;
        if (covers) { zl_push(x, &runs, 0, (__int128)M - 1); count = 0; }
    }
    if (!runs.n && count) {
        if (M <= TR_PREIMG_MAX_PERIOD) {
            if (step(x, count)) {
                uint64_t *bm = (uint64_t *)zalloc(((M + 63) / 64) * 8);
                if (!bm) hard(x, TR_MATH_MEMORY);
                for (uint64_t i = 0; bm && i < Y.n; i++) {
                    __int128 y0 = Y.v[i].lo, y1 = Y.v[i].hi;
                    for (__int128 y = y0 + fmod64((__int128)b - y0, (int64_t)g); y <= y1; y += g) {
                        uint64_t t = (uint64_t)fmod64((y - (__int128)b) / (__int128)g, (int64_t)M), r = mulmod(t, inv, M);
                        bm[r >> 6] |= (uint64_t)1 << (r & 63);
                    }
                }
                if (bm) bitmap_runs(x, bm, M, &runs);
                free(bm);
            }
        } else if (count <= TR_PREIMG_MAX_LOOKUP && count <= x->b->limit[TR_BUDGET_BYTES] / 8) {
            if (step(x, count)) {
                uint64_t *res = (uint64_t *)zalloc(count * 8), k = 0;
                if (!res) hard(x, TR_MATH_MEMORY);
                for (uint64_t i = 0; res && i < Y.n; i++) {
                    __int128 y0 = Y.v[i].lo, y1 = Y.v[i].hi;
                    for (__int128 y = y0 + fmod64((__int128)b - y0, (int64_t)g); y <= y1 && k < count; y += g) {
                        uint64_t t = (uint64_t)fmod64((y - (__int128)b) / (__int128)g, (int64_t)M);
                        res[k++] = mulmod(t, inv, M);
                    }
                }
                if (res) {
                    qsort(res, (size_t)k, 8, cmp_u64);
                    for (uint64_t i = 0; i < k; i++) {
                        if (runs.n && (__int128)res[i] <= runs.v[runs.n - 1].hi + 1) {
                            if ((__int128)res[i] > runs.v[runs.n - 1].hi) runs.v[runs.n - 1].hi = res[i];
                        } else zl_push(x, &runs, res[i], res[i]);
                    }
                }
                free(res);
            }
        } else soft(x, TR_PREIMG_WHY_PERIOD, TR_BUDGET_ITERATIONS, count, TR_PREIMG_MAX_LOOKUP);
    }
    if (!x->rc && !x->why) {
        shift_runs(x, &runs, (int64_t)M, cd->offset, &crun);
        if (crun.n) residues_to_set(x, &crun, (int64_t)M, cd, u); else u->repr = TR_PREIMG_INTERVALS;
        u->built = !x->rc;
    }
    free(Y.v); free(runs.v); free(crun.v);
}

/* bit maps: the dyadic blocks of {y op c}, their GF(2) images (cubes of u), then cubes or residue runs of codes */
static int cube_push(ctx *x, uset *u, uint64_t mask, uint64_t value) {
    if (u->ncubes == u->capcubes) {
        uint64_t nc = u->capcubes ? 2 * u->capcubes : 16;
        uint64_t *a = (uint64_t *)realloc(u->mask, (size_t)nc * 8);
        if (a) u->mask = a;
        uint64_t *b = a ? (uint64_t *)realloc(u->value, (size_t)nc * 8) : NULL;
        if (b) u->value = b;
        if (!a || !b) { hard(x, TR_MATH_MEMORY); return 0; }
        u->capcubes = nc;
    }
    u->mask[u->ncubes] = mask; u->value[u->ncubes] = value; u->ncubes++;
    return 1;
}
static void bits_union(ctx *x, const tr_pre_codec *cd, const tr_pre_predicate *pr, const tr_num_rational *vals,
                       uint64_t nvals, const tr_preimg_view *v, int64_t *period, uset *u) {
    const int32_t w = v->width;
    const uint64_t wm = ((uint64_t)1 << w) - 1, c = (uint64_t)v->b;
    zlist Y = {NULL, 0, 0};
    int_targets(x, pr, vals, nvals, (__int128)wm, &Y);
    uset cu;
    memset(&cu, 0, sizeof cu);
    for (uint64_t i = 0; i < Y.n && !x->rc; i++) {
        uint64_t y0 = (uint64_t)Y.v[i].lo, y1 = (uint64_t)Y.v[i].hi;
        while (y0 <= y1 && !x->rc) {                       /* the largest aligned block at y0 inside [y0, y1] */
            int j = 0;
            while (j < w && !((y0 >> j) & 1) && y0 + (((uint64_t)2 << j) - 1) <= y1) j++;
            const uint64_t mk = wm & ~((((uint64_t)1) << j) - 1), V = y0 & mk;
            uint64_t um = 0, uv = 0;
            int ok = 1;
            switch (v->map) {
            case TR_PREIMG_XOR: um = mk; uv = V ^ (c & mk); break;
            case TR_PREIMG_AND: if (V & ~c & mk) ok = 0; um = mk & c; uv = V & c; break;
            default: if (~V & c & mk) ok = 0; um = mk & ~c; uv = V & ~c; break;
            }
            if (ok) cube_push(x, &cu, um, uv);
            uint64_t len = (uint64_t)1 << j;
            if (y1 - y0 < len) break;
            y0 += len;
        }
    }
    free(Y.v);
    /* codes: u = (code + d) mod 2^w */
    const uint64_t d = (uint64_t)cd->offset - (uint64_t)v->base;
    int H = -1, aligned = 1;
    for (uint64_t i = 0; i < cu.ncubes; i++) {
        if (!cu.mask[i]) { H = -2; break; }               /* every u: the whole domain */
        int h = 63 - __builtin_clzll(cu.mask[i]);
        if (h > H) H = h;
        uint64_t lowm = h >= 63 ? UINT64_MAX : (((uint64_t)2 << h) - 1);
        if (d & lowm) aligned = 0;
    }
    *period = 0;
    if (H == -2) { u->repr = TR_PREIMG_INTERVALS; zl_push(x, &u->iv, cd->lo, cd->hi); u->built = !x->rc; uset_free(&cu); return; }
    if (!cu.ncubes) { u->repr = TR_PREIMG_INTERVALS; u->built = !x->rc; uset_free(&cu); return; }
    if (aligned) {
        u->repr = TR_PREIMG_CUBES;
        for (uint64_t i = 0; i < cu.ncubes && !x->rc; i++) cube_push(x, u, cu.mask[i], cu.value[i]);
        u->built = !x->rc;
        uset_free(&cu);
        return;
    }
    /* residue runs mod 2^(H+1): count them first (the free bits above each cube's lowest fixed bit) */
    if (H + 1 >= 63) { soft(x, TR_PREIMG_WHY_PERIOD, TR_BUDGET_OUTPUT, UINT64_MAX, x->b->limit[TR_BUDGET_OUTPUT]); uset_free(&cu); return; }
    const uint64_t M = (uint64_t)1 << (H + 1), Mm = M - 1;
    *period = (int64_t)M;
    uint64_t runs_needed = 0;
    for (uint64_t i = 0; i < cu.ncubes; i++) {
        int l = __builtin_ctzll(cu.mask[i]);
        uint64_t above = Mm & ~(((uint64_t)2 << l) - 1) & ~cu.mask[i];
        int f = __builtin_popcountll(above);
        uint64_t k = f >= 63 ? UINT64_MAX : ((uint64_t)1 << f);
        runs_needed = sat_add(runs_needed, k);
    }
    if (runs_needed > x->b->limit[TR_BUDGET_OUTPUT] || runs_needed > TR_PREIMG_MAX_LOOKUP) {
        soft(x, TR_PREIMG_WHY_OUTPUT, TR_BUDGET_OUTPUT, runs_needed, x->b->limit[TR_BUDGET_OUTPUT]);
        uset_free(&cu);
        return;
    }
    zlist runs = {NULL, 0, 0}, crun = {NULL, 0, 0};
    for (uint64_t i = 0; i < cu.ncubes && !x->rc; i++) {
        const int l = __builtin_ctzll(cu.mask[i]);
        const uint64_t above = Mm & ~(((uint64_t)2 << l) - 1) & ~cu.mask[i];
        uint64_t sub = 0;
        for (;;) {                                         /* every assignment of the free bits above l */
            uint64_t start = (cu.value[i] | sub) & Mm & ~((((uint64_t)1) << l) - 1);
            zl_push(x, &runs, start, (__int128)start + (((uint64_t)1 << l) - 1));
            if (sub == above) break;
            sub = (sub - above) & above;
        }
    }
    zl_merge(&runs);
    shift_runs(x, &runs, (int64_t)M, (int64_t)d, &crun);             /* code = u - d (mod M) */
    if (!x->rc) {
        if (crun.n) residues_to_set(x, &crun, (int64_t)M, cd, u); else u->repr = TR_PREIMG_INTERVALS;
        u->built = !x->rc;
    }
    free(runs.v); free(crun.v);
    uset_free(&cu);
}

/* ================================================================ law certificates */
static int fits64(const tr_bigint *a, int64_t *out) { return a && !tr_bigint_to_i64(a, out); }
static tr_cert *law_of(int32_t ncols, uint64_t n, const uint8_t *e, const int64_t *c, const tr_math_budget *b) {
    tr_cert_claim k;
    memset(&k, 0, sizeof k);
    k.kind = TR_CLAIM_POLYNOMIAL; k.policy = TR_CERT_NULL_FORBID; k.ncols = ncols;
    for (int32_t i = 0; i < TR_CERT_MAX_COLS; i++) k.col[i] = i < ncols ? i : -1;
    k.carrier = TR_CARRIER_I128; k.nterms = n; k.exponents = e; k.coefficients = c;
    tr_math_refusal r;
    int32_t err = 0;
    return tr_cert_law(TR_CERT_LAW_RING_IDENTITY, &k, b, &r, &err);
}
/* POLY: sum_i n_i (code + offset)^i term by term - Q(code) */
static tr_cert *poly_law(ctx *x, const polydata *pd, int64_t offset) {
    const int32_t d = pd->d;
    uint64_t cap = (uint64_t)(d + 1) * (uint64_t)(d + 2) / 2 + (uint64_t)d + 1, n = 0;
    uint8_t *e = (uint8_t *)zalloc(cap);
    int64_t *c = (int64_t *)zalloc(cap * 8);
    tr_bigint *Q[TR_PREIMG_MAX_DEGREE + 1];
    memset(Q, 0, sizeof Q);
    int ok = e && c;
    tr_bigint *off = bi64(x, offset);
    for (int32_t j = 0; j <= d && ok; j++) Q[j] = bi64(x, 0);
    for (int32_t i = 0; i <= d && ok && !x->rc && !x->why; i++) {
        tr_bigint *binom = bi64(x, 1);
        for (int32_t j = 0; j <= i && ok && !x->rc && !x->why; j++) {      /* n_i C(i, j) offset^(i - j) code^j */
            if (j) {                                                       /* C(i, j) = C(i, j - 1) (i - j + 1) / j */
                tr_bigint *f = bi64(x, i - j + 1), *t = z_mul(x, binom, f), *jj = bi64(x, j), *q = z_divexact(x, t, jj);
                tr_bigint_free(f); tr_bigint_free(t); tr_bigint_free(jj);
                if (q) { tr_bigint_free(binom); binom = q; }
            }
            tr_bigint *op = z_pow(x, off, (uint64_t)(i - j)), *t1 = z_mul(x, pd->n[i], binom), *t2 = z_mul(x, t1, op);
            int64_t v = 0;
            if (!fits64(t2, &v)) ok = 0;
            else if (v) { e[n] = (uint8_t)j; c[n] = v; n++; }
            tr_bigint *s = z_add(x, Q[j], t2);
            if (s) { tr_bigint_free(Q[j]); Q[j] = s; }
            tr_bigint_free(op); tr_bigint_free(t1); tr_bigint_free(t2);
        }
        tr_bigint_free(binom);
    }
    for (int32_t j = 0; j <= d && ok && !x->rc && !x->why; j++) {
        int64_t v = 0;
        if (!fits64(Q[j], &v) || v == INT64_MIN) ok = 0;
        else if (v) { e[n] = (uint8_t)j; c[n] = -v; n++; }
    }
    tr_cert *cert = ok && !x->rc && !x->why && n ? law_of(1, n, e, c, x->b) : NULL;
    for (int32_t j = 0; j <= d; j++) tr_bigint_free(Q[j]);
    tr_bigint_free(off); free(e); free(c);
    return cert;
}
/* POLYMOD: P(r + m t) expanded term by term - P(r) - m Q(r, t, m) = 0 over the variables (r, t, m): every term of
 * P(r + m t) - P(r) carries the factor m, so P(x) mod m depends on x mod m only */
static tr_cert *polymod_law(const int64_t *coef, int32_t d, const tr_math_budget *b) {
    uint64_t cap = (uint64_t)(d + 1) * (uint64_t)(d + 2) + (uint64_t)d + 1, n = 0;
    uint8_t *e = (uint8_t *)zalloc(cap * 3);
    int64_t *c = (int64_t *)zalloc(cap * 8);
    int ok = e && c;
    for (int32_t i = 0; i <= d && ok; i++) {
        __int128 binom = 1;                                /* C(i, j) for j = 0, 1, .. */
        for (int32_t j = 0; j <= i && ok; j++) {
            if (j) binom = binom * (i - j + 1) / j;
            __int128 v = (__int128)coef[i] * binom;        /* c_i C(i, j) r^j t^(i - j) m^(i - j) */
            if (v < -(__int128)INT64_MAX || v > INT64_MAX) { ok = 0; break; }
            if (!v) continue;
            e[3 * n] = (uint8_t)j; e[3 * n + 1] = e[3 * n + 2] = (uint8_t)(i - j); c[n++] = (int64_t)v;
            /* j < i: the same monomial in - m Q; j = i: - P(r) */
            e[3 * n] = (uint8_t)j; e[3 * n + 1] = e[3 * n + 2] = (uint8_t)(i - j); c[n++] = (int64_t)-v;
        }
    }
    tr_cert *cert = ok && n ? law_of(3, n, e, c, b) : NULL;
    free(e); free(c);
    return cert;
}
/* AFFMOD: a (r + M t) + b - (a r + b) - (a / g) m t over (r, t) */
static tr_cert *affmod_law(int64_t a, int64_t b, int64_t m, int64_t M, const tr_math_budget *bud) {
    if (!a || !M) return NULL;
    __int128 aM = (__int128)a * M, am = (__int128)(a / (m / M)) * m;
    if (aM > INT64_MAX || am > INT64_MAX) return NULL;
    uint8_t e[12] = {1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1};
    int64_t c[6] = {a, (int64_t)aM, b, -a, -b, -(int64_t)am};
    return law_of(2, 6, e, c, bud);
}

/* ================================================================ the preimage */
static void info_counts(tr_preimg_set *s, const uset *u) {
    tr_preimg_info *in = &s->info;
    const tr_pre_codec *cd = &in->codec;
    uint64_t codes = 0;
    if (u->repr == TR_PREIMG_CUBES) {
        for (uint64_t i = 0; i < u->ncubes; i++) {
            uint64_t k = count_cube_upto(cd->hi, u->mask[i], u->value[i]) - count_cube_upto((__int128)cd->lo - 1, u->mask[i], u->value[i]);
            codes = sat_add(codes, k);
        }
        in->branches = u->ncubes;
    } else {
        for (uint64_t i = 0; i < u->iv.n; i++) {
            uint64_t k = u->repr == TR_PREIMG_RESIDUES ? count_residue(cd->lo, cd->hi, u->modulus, u->iv.v[i].lo, u->iv.v[i].hi)
                                                       : (u->iv.v[i].hi - u->iv.v[i].lo + 1 > (__int128)UINT64_MAX ? UINT64_MAX :
                                                          (uint64_t)(u->iv.v[i].hi - u->iv.v[i].lo + 1));
            codes = sat_add(codes, k);
        }
        in->branches = u->iv.n;
    }
    in->codes = codes;
    __int128 size = (__int128)cd->hi - cd->lo + 1;
    in->empty = codes == 0;
    in->all = size <= (__int128)UINT64_MAX && codes == (uint64_t)size;
}
static int32_t store_branches(tr_preimg_set *s, const uset *u) {
    const uint64_t n = u->repr == TR_PREIMG_CUBES ? u->ncubes : u->iv.n;
    s->branch = (tr_preimg_branch *)zalloc((n ? n : 1) * sizeof *s->branch);
    if (!s->branch) return TR_MATH_MEMORY;
    const tr_pre_codec *cd = &s->info.codec;
    for (uint64_t i = 0; i < n; i++) {
        tr_preimg_branch *b = &s->branch[i];
        b->modulus = 1;
        if (u->repr == TR_PREIMG_CUBES) { b->lo = cd->lo; b->hi = cd->hi; b->mask = u->mask[i]; b->value = u->value[i]; }
        else if (u->repr == TR_PREIMG_RESIDUES) {
            b->lo = cd->lo; b->hi = cd->hi; b->modulus = u->modulus; b->rlo = (int64_t)u->iv.v[i].lo; b->rhi = (int64_t)u->iv.v[i].hi;
        } else { b->lo = (int64_t)u->iv.v[i].lo; b->hi = (int64_t)u->iv.v[i].hi; }
    }
    s->info.nbranch = n; s->info.mode = TR_PREIMG_BRANCHES; s->info.repr = u->repr;
    s->info.modulus = u->repr == TR_PREIMG_RESIDUES ? u->modulus : 0;
    s->view.nbranch = n; s->view.branch = s->branch; s->view.mode = TR_PREIMG_BRANCHES; s->view.repr = u->repr;
    s->view.modulus = s->info.modulus;
    return TR_MATH_OK;
}
/* the lookup bitmap: residues mod period (periodic maps, period <= the limit) or the domain's codes; from the exact
 * union when it was built, else by the direct evaluation of each key */
static int32_t build_lookup(tr_preimg_set *s, const uset *u, int64_t period, const tr_math_budget *b, uint64_t base,
                            int *fits) {
    const tr_pre_codec *cd = &s->info.codec;
    const __int128 size = (__int128)cd->hi - cd->lo + 1;
    uint64_t bits = 0, mod = 0;
    *fits = 0;
    if (period > 0 && (uint64_t)period <= TR_PREIMG_MAX_LOOKUP && (__int128)period < size) { bits = (uint64_t)period; mod = (uint64_t)period; }
    else if (size <= (__int128)TR_PREIMG_MAX_LOOKUP) bits = (uint64_t)size;
    else return TR_MATH_OK;
    if (tr_budget_add(base, tr_budget_bitmap_bytes(bits)) > b->limit[TR_BUDGET_BYTES]) return TR_MATH_OK;
    if (!u->built) {                                         /* the direct evaluation decides each key */
        if (s->view.map == TR_PREIMG_POLY && !s->view.words) return TR_MATH_OK;
        __int128 L0 = (__int128)cd->lo + cd->offset, L1 = L0 + (__int128)bits - 1;
        if (L0 < INT64_MIN || L1 > INT64_MAX) return TR_MATH_OK;
    }
    uint64_t *bm = (uint64_t *)zalloc(tr_budget_bitmap_bytes(bits));
    if (!bm) return TR_MATH_MEMORY;
    tr_preimg_view w = s->view;
    if (u->built) {                                          /* membership through the union's branches */
        w.mode = TR_PREIMG_BRANCHES; w.repr = u->repr;
        tr_preimg_branch *tmp = (tr_preimg_branch *)zalloc(((u->repr == TR_PREIMG_CUBES ? u->ncubes : u->iv.n) + 1) * sizeof *tmp);
        if (!tmp) { free(bm); return TR_MATH_MEMORY; }
        uint64_t n = u->repr == TR_PREIMG_CUBES ? u->ncubes : u->iv.n;
        for (uint64_t i = 0; i < n; i++) {
            tr_preimg_branch *br = &tmp[i];
            br->modulus = 1;
            if (u->repr == TR_PREIMG_CUBES) { br->lo = cd->lo; br->hi = cd->hi; br->mask = u->mask[i]; br->value = u->value[i]; }
            else if (u->repr == TR_PREIMG_RESIDUES) { br->lo = cd->lo; br->hi = cd->hi; br->modulus = u->modulus; br->rlo = (int64_t)u->iv.v[i].lo; br->rhi = (int64_t)u->iv.v[i].hi; }
            else { br->lo = (int64_t)u->iv.v[i].lo; br->hi = (int64_t)u->iv.v[i].hi; }
        }
        w.nbranch = n; w.branch = tmp;
        for (uint64_t k = 0; k < bits; k++) {
            __int128 code = mod ? (__int128)cd->lo - fmod64(cd->lo, (int64_t)mod) + k : (__int128)cd->lo + k;
            if (mod && code < cd->lo) code += mod;           /* a representative code of residue k inside the domain */
            if (tr_preimg_member(&w, (int64_t)code)) bm[k >> 6] |= (uint64_t)1 << (k & 63);
        }
        free(tmp);
    } else {
        for (uint64_t k = 0; k < bits; k++) {
            __int128 code = mod ? (__int128)cd->lo - fmod64(cd->lo, (int64_t)mod) + k : (__int128)cd->lo + k;
            if (mod && code < cd->lo) code += mod;
            __int128 L = code + cd->offset;
            if (L < INT64_MIN || L > INT64_MAX) { free(bm); return TR_MATH_OK; }
            if (tr_preimg_direct(&s->view, (int64_t)L) == 1) bm[k >> 6] |= (uint64_t)1 << (k & 63);
        }
    }
    s->lookup = bm;
    s->info.mode = TR_PREIMG_LOOKUP; s->info.nlookup = bits; s->info.modulus = (int64_t)mod; s->info.nbranch = 0;
    s->view.mode = TR_PREIMG_LOOKUP; s->view.nlookup = bits; s->view.lookup = bm; s->view.modulus = (int64_t)mod;
    s->view.nbranch = 0; s->view.branch = NULL;
    *fits = 1;
    return TR_MATH_OK;
}
tr_preimg_set *tr_preimg_preimage(const tr_pre_codec *cd, const tr_preimg_map *m, const tr_pre_predicate *pr,
                                  const tr_preimg_spec *spec, const tr_math_budget *budget, tr_math_refusal *refusal,
                                  int32_t *error) {
    tr_budget_none(refusal);
    int32_t rc = check_input(cd, m, pr, spec);
    if (rc) { set_error(error, rc); return NULL; }
    tr_math_budget dflt;
    const tr_math_budget *b = budget;
    if (!b) { tr_budget_default(&dflt); b = &dflt; }
    const int32_t fallback = spec ? spec->fallback : TR_PREIMG_FALLBACK_AUTO;
    const uint64_t nvalues0 = pr->op == TR_PRE_IN ? pr->nvalues : 0;
    tr_math_need need;
    for (int i = 0; i < TR_BUDGET_RESOURCES; i++) need.need[i] = 0;
    need.need[TR_BUDGET_SUPPORT] = nvalues0 > (uint64_t)(m->degree + 1) ? nvalues0 : (uint64_t)(m->degree + 1);
    need.need[TR_BUDGET_DEGREE] = (m->kind == TR_PREIMG_POLY || m->kind == TR_PREIMG_POLYMOD) ? (uint64_t)m->degree : 1;
    /* the base need: the set and its values with their W-word thresholds; the representation adds its branches or
     * its lookup bitmap (checked when it is chosen) */
    const uint64_t base_bytes = tr_budget_add(sizeof(tr_preimg_set),
                                              tr_budget_mul(nvalues0, sizeof(tr_num_rational) + 8 * TR_PREIMG_MAX_WORDS));
    need.need[TR_BUDGET_BYTES] = base_bytes;
    rc = tr_budget_check(b, &need, refusal);
    if (rc) { set_error(error, rc); return NULL; }
    tr_preimg_set *s = (tr_preimg_set *)zalloc(sizeof *s);
    if (!s) { set_error(error, TR_MATH_MEMORY); return NULL; }
    s->magic = SET_MAGIC;
    tr_preimg_info *in = &s->info;
    tr_preimg_view *v = &s->view;
    in->map = m->kind; in->op = pr->op; in->codec = *cd;
    v->codec = *cd; v->map = m->kind; v->op = pr->op; v->c = pr->op == TR_PRE_IN ? tr_q_int(0) : pr->c;
    v->width = m->width;
    uint64_t nvals = 0;
    if (nvalues0) {                                          /* IN: sorted distinct values */
        s->values = (tr_num_rational *)zalloc(nvalues0 * sizeof *s->values);
        if (!s->values) { tr_preimg_set_free(s); set_error(error, TR_MATH_MEMORY); return NULL; }
        memcpy(s->values, pr->values, (size_t)nvalues0 * sizeof *s->values);
        qsort(s->values, (size_t)nvalues0, sizeof *s->values, cmp_q);
        for (uint64_t i = 0; i < nvalues0; i++) if (!nvals || tr_q_cmp(s->values[nvals - 1], s->values[i]) != 0) s->values[nvals++] = s->values[i];
    }
    v->values = s->values; v->nvalues = nvals;
    ctx x;
    memset(&x, 0, sizeof x);
    x.b = b; x.r = refusal; x.max_iter = b->limit[TR_BUDGET_ITERATIONS];
    uset u;
    memset(&u, 0, sizeof u);
    polydata pd;
    memset(&pd, 0, sizeof pd);
    int64_t period = 0;
    const __int128 Llo = (__int128)cd->lo + cd->offset, Lhi = (__int128)cd->hi + cd->offset;
    switch (m->kind) {
    case TR_PREIMG_POLY: {
        poly_prepare(&x, cd, m, pr, s->values, nvals, &pd);
        v->degree = pd.d;
        int32_t W = (!x.rc && !x.why && Llo >= INT64_MIN && Lhi <= INT64_MAX) ? poly_words(&x, &pd, Llo, Lhi) : 0;
        if (W) {
            s->wcoef = (uint64_t *)zalloc((uint64_t)(pd.d + 1) * (uint64_t)W * 8);
            s->wthreshold = (uint64_t *)zalloc((pd.nt ? pd.nt : 1) * (uint64_t)W * 8);
            if (!s->wcoef || !s->wthreshold) hard(&x, TR_MATH_MEMORY);
            for (int32_t i = 0; i <= pd.d && !x.rc; i++) { int32_t e = to_words(pd.n[i], W, s->wcoef + (uint64_t)i * W); if (e) hard(&x, e); }
            for (uint64_t j = 0; j < pd.nt && !x.rc; j++) { int32_t e = to_words(pd.R[j], W, s->wthreshold + j * (uint64_t)W); if (e) hard(&x, e); }
            v->words = W; v->wcoef = s->wcoef; v->wthreshold = s->wthreshold;
        }
        in->words = v->words;
        if (!x.rc && !x.why) {
            pctx p;
            memset(&p, 0, sizeof p);
            p.x = &x; p.d = pd.d; p.W = v->words; p.n = pd.n; p.R = pd.R; p.den = pd.den; p.nt = pd.nt;
            p.wcoef = s->wcoef; p.wthr = s->wthreshold; p.v = v;
            poly_union(&x, cd, pr->op, &pd, &p, &u, &in->pieces, &in->critical);
        }
        if (!x.rc && pd.K) { tr_cert *c = poly_law(&x, &pd, cd->offset); if (c) { s->cert = c; in->certified = 1; } }
        break;
    }
    case TR_PREIMG_POLYMOD: {
        int32_t d = m->degree;
        while (d > 0 && fmod64(m->coef[d].numerator, m->modulus) == 0) d--;
        s->icoef = (int64_t *)zalloc((uint64_t)(d + 1) * 8);
        int64_t *raw = (int64_t *)zalloc((uint64_t)(m->degree + 1) * 8);
        if (!s->icoef || !raw) { free(raw); hard(&x, TR_MATH_MEMORY); break; }
        for (int32_t i = 0; i <= m->degree; i++) raw[i] = m->coef[i].numerator;
        for (int32_t i = 0; i <= d; i++) s->icoef[i] = fmod64(m->coef[i].numerator, m->modulus);
        v->icoef = s->icoef; v->degree = d; v->m = m->modulus;
        period = m->modulus;
        polymod_union(&x, cd, pr, s->values, nvals, raw, m->degree, v, &u);
        tr_cert *c = polymod_law(raw, m->degree, b);
        if (c) { s->cert = c; in->certified = 1; }
        free(raw);
        break;
    }
    case TR_PREIMG_AFFMOD: {
        v->m = m->modulus; v->a = fmod64(m->a, m->modulus); v->b = fmod64(m->b, m->modulus);
        affmod_union(&x, cd, pr, s->values, nvals, v, &period, &u);
        if (v->a) {
            tr_cert *c = affmod_law(v->a, v->b, v->m, period, b);
            if (c) { s->cert = c; in->certified = 1; }
        }
        break;
    }
    default:
        v->base = m->base_is_offset ? cd->offset : m->base; v->b = m->b;
        bits_union(&x, cd, pr, s->values, nvals, v, &period, &u);
        break;
    }
    in->iterations = x.iter;
    rc = x.rc;
    if (!rc) {
        if (u.built) info_counts(s, &u);
        else { in->branches = UINT64_MAX; in->codes = UINT64_MAX; }
        const uint64_t nb = u.built ? in->branches : UINT64_MAX;
        const uint64_t bytes = tr_budget_add(base_bytes, tr_budget_mul(nb, sizeof(tr_preimg_branch)));
        if (u.built && nb <= b->limit[TR_BUDGET_OUTPUT] && bytes <= b->limit[TR_BUDGET_BYTES]) rc = store_branches(s, &u);
        else {
            if (u.built && !x.why) {
                if (nb > b->limit[TR_BUDGET_OUTPUT]) soft(&x, TR_PREIMG_WHY_OUTPUT, TR_BUDGET_OUTPUT, nb, b->limit[TR_BUDGET_OUTPUT]);
                else soft(&x, TR_PREIMG_WHY_BYTES, TR_BUDGET_BYTES, bytes, b->limit[TR_BUDGET_BYTES]);
            }
            in->why = x.why;
            int done = 0;
            if (fallback == TR_PREIMG_FALLBACK_AUTO || fallback == TR_PREIMG_FALLBACK_LOOKUP) {
                rc = build_lookup(s, &u, period, b, base_bytes, &done);
                if (done) in->fallback = TR_PREIMG_LOOKUP;
            }
            if (!rc && !done && (fallback == TR_PREIMG_FALLBACK_AUTO || fallback == TR_PREIMG_FALLBACK_DIRECT) &&
                (m->kind != TR_PREIMG_POLY || v->words)) {
                in->mode = v->mode = TR_PREIMG_DIRECT; in->fallback = TR_PREIMG_DIRECT; done = 1;
            }
            if (!rc && !done) {                                /* nothing falls back: the budget refusal */
                if (refusal) { refusal->resource = x.why_resource; refusal->refused = 1; refusal->need = x.why_need; refusal->limit = x.why_limit; }
                rc = TR_MATH_BUDGET;
            }
        }
    }
    poly_free(&pd);
    uset_free(&u);
    if (rc) { tr_preimg_set_free(s); set_error(error, rc); return NULL; }
    if (in->mode != TR_PREIMG_BRANCHES) { in->empty = in->codes == 0; }
    in->bytes = tr_budget_add(base_bytes, tr_budget_add(tr_budget_mul(in->nbranch, sizeof(tr_preimg_branch)),
                                                         tr_budget_bitmap_bytes(in->nlookup)));
    set_error(error, TR_MATH_OK);
    return s;
}

/* ================================================================ reading */
int32_t tr_preimg_set_get_info(const tr_preimg_set *s, tr_preimg_info *out) {
    if (!set_valid(s) || !out) return TR_MATH_INVALID;
    *out = s->info;
    return TR_MATH_OK;
}
int32_t tr_preimg_set_branches(const tr_preimg_set *s, tr_preimg_branch *out, uint64_t capacity) {
    if (!set_valid(s) || (s->info.nbranch && !out) || capacity < s->info.nbranch) return TR_MATH_INVALID;
    if (s->info.nbranch) memcpy(out, s->branch, (size_t)s->info.nbranch * sizeof *out);
    return TR_MATH_OK;
}
int32_t tr_preimg_set_contains(const tr_preimg_set *s, int64_t code) {
    if (!set_valid(s)) return TR_MATH_INVALID;
    return tr_preimg_member(&s->view, code);
}
int32_t tr_preimg_set_get_view(const tr_preimg_set *s, tr_preimg_view *out) {
    if (!set_valid(s) || !out) return TR_MATH_INVALID;
    *out = s->view;
    return TR_MATH_OK;
}
int32_t tr_preimg_set_lookup(const tr_preimg_set *s, uint64_t *out, uint64_t capacity_words) {
    if (!set_valid(s) || s->info.mode != TR_PREIMG_LOOKUP || !out) return TR_MATH_INVALID;
    uint64_t nw = (s->info.nlookup + 63) / 64;
    if (capacity_words < nw) return TR_MATH_INVALID;
    memcpy(out, s->lookup, (size_t)nw * 8);
    return TR_MATH_OK;
}
const tr_cert *tr_preimg_set_certificate(const tr_preimg_set *s) { return set_valid(s) ? s->cert : NULL; }
int32_t tr_preimg_set_cubes(const tr_preimg_set *s, int32_t width, tr_preimg_branch *out, uint64_t capacity, uint64_t *count) {
    if (!set_valid(s) || !count || width < 1 || width > 63 || (capacity && !out)) return TR_MATH_INVALID;
    if (s->info.mode != TR_PREIMG_BRANCHES || s->info.repr == TR_PREIMG_INTERVALS) return TR_MATH_UNSUPPORTED;
    const uint64_t wm = ((uint64_t)1 << width) - 1;
    uint64_t n = 0;
    const tr_pre_codec *cd = &s->info.codec;
#define EMIT(mk, val) do { if (n < capacity) { tr_preimg_branch *o = &out[n]; memset(o, 0, sizeof *o); o->lo = cd->lo; o->hi = cd->hi; \
        o->modulus = 1; o->mask = (mk); o->value = (val); } n++; } while (0)
    if (s->info.repr == TR_PREIMG_CUBES) {
        for (uint64_t i = 0; i < s->info.nbranch; i++) {
            const tr_preimg_branch *b = &s->branch[i];
            if ((b->value & ~wm) || (b->mask & ~wm & b->value)) continue;       /* a bit above the codes' width set: empty */
            EMIT(b->mask & wm, b->value & wm);
        }
    } else {
        const int64_t M = s->info.modulus;
        if (M < 1 || (M & (M - 1))) return TR_MATH_UNSUPPORTED;               /* not a power of two: no cubes */
        const int k = __builtin_ctzll((uint64_t)M);
        if (k > width) return TR_MATH_UNSUPPORTED;
        for (uint64_t i = 0; i < s->info.nbranch; i++) {                       /* aligned dyadic blocks of each run */
            uint64_t r0 = (uint64_t)s->branch[i].rlo, r1 = (uint64_t)s->branch[i].rhi;
            while (r0 <= r1) {
                int j = 0;
                while (j < k && !((r0 >> j) & 1) && r0 + (((uint64_t)2 << j) - 1) <= r1) j++;
                const uint64_t mk = ((uint64_t)M - 1) & ~((((uint64_t)1) << j) - 1);
                EMIT(mk, r0 & mk);
                r0 += (uint64_t)1 << j;
                if (!r0) break;
            }
        }
    }
#undef EMIT
    *count = n;
    return n > capacity ? TR_MATH_INVALID : TR_MATH_OK;
}

/* ================================================================ scan and verification */
int32_t tr_preimg_scan(const tr_preimg_set *s, const tr_scope *scope, int32_t column, int32_t direct, uint64_t *bitmap,
                       tr_pre_scan_stats *stats, const tr_math_budget *budget, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    tr_scope_view sv;
    if (!set_valid(s) || !scope || tr_scope_get_view(scope, &sv) || !stats || column < 0 || column >= sv.ncols ||
        (sv.rows && !bitmap) || (direct != 0 && direct != 1)) return TR_MATH_INVALID;
    if (sv.col[column].scale != s->info.codec.scale) return TR_MATH_INVALID;          /* another value unit */
    if (direct && s->view.map == TR_PREIMG_POLY && !s->view.words) return TR_MATH_UNSUPPORTED;
    tr_math_budget dflt;
    if (!budget) { tr_budget_default(&dflt); budget = &dflt; }
    int32_t rc = tr_budget_check_one(budget, TR_BUDGET_BYTES, tr_budget_bitmap_bytes(sv.rows), refusal);
    if (rc) return rc;
    memset(stats, 0, sizeof *stats);
    stats->rows = sv.rows; stats->selected = sv.selected;
    stats->first_mismatch = UINT64_MAX; stats->first_overflow = UINT64_MAX;
    uint64_t acc[6] = {0, 0, 0, 0, 0, 0}, nw = (sv.rows + 63) / 64;
    for (uint64_t w = 0; w < nw; w++)
        bitmap[w] = tr_preimg_word(&s->view, &sv, column, direct, w, acc, &stats->first_mismatch, &stats->first_overflow);
    stats->nulls = acc[0]; stats->in_set = acc[1]; stats->direct_true = acc[2]; stats->mismatches = acc[3];
    stats->overflow = acc[4]; stats->outside = acc[5];
    return stats->overflow ? TR_MATH_OVERFLOW : TR_MATH_OK;
}
int32_t tr_preimg_verify(const tr_preimg_set *s, const tr_math_budget *budget, tr_preimg_verification *out,
                         tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!set_valid(s) || !out) return TR_MATH_INVALID;
    const tr_pre_codec *cd = &s->info.codec;
    const __int128 size = (__int128)cd->hi - cd->lo + 1, L0 = (__int128)cd->lo + cd->offset, L1 = (__int128)cd->hi + cd->offset;
    if (size > (__int128)TR_PREIMG_MAX_VERIFY || L0 < INT64_MIN || L1 > INT64_MAX) return TR_MATH_UNSUPPORTED;
    if (s->view.map == TR_PREIMG_POLY && !s->view.words) return TR_MATH_UNSUPPORTED;   /* no direct evaluation */
    tr_math_budget dflt;
    if (!budget) { tr_budget_default(&dflt); budget = &dflt; }
    int32_t rc = tr_budget_check_one(budget, TR_BUDGET_ITERATIONS, (uint64_t)size, refusal);
    if (rc) return rc;
    memset(out, 0, sizeof *out);
    out->lo = cd->lo; out->hi = cd->hi;
    uint64_t acc[4] = {0, 0, 0, 0}, first = UINT64_MAX, total = (uint64_t)size, nw = (total + 63) / 64;
    for (uint64_t w = 0; w < nw; w++) tr_preimg_verify_word(&s->view, w, total, acc, &first);
    out->codes = acc[0]; out->members = acc[1]; out->truths = acc[2]; out->mismatches = acc[3];
    out->first = first == UINT64_MAX ? 0 : (int64_t)((uint64_t)cd->lo + first);
    return TR_MATH_OK;
}
