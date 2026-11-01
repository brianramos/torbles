/* PROB-001: small-support partition and likelihood functions; every contract is in math_prob.h.
 * Built on DIST-001 (math_dist.h: the histogram states), CALC-003 (math_calc.h: the certified exponential), DIST-003
 * (math_info.h: the certified logarithm), FND-009 (math_carriers.h: big integers, tight intervals), FND-004
 * (math_scope.h: sources and descriptors) and GEN-002 (math_conv.h: the independence declaration).
 * SPDX-License-Identifier: Apache-2.0 */
#include "math_prob.h"
#include "mathematics.h"
#include "math_domain.h"
#include "math_handles.h"
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

TR_MATH_ABI(tr_prob_support_view, TR_PROB_ABI);     /* the ctypes boundary of PROB-001 (math_handles.h) */
TR_MATH_ABI(tr_prob_model_view, TR_PROB_ABI);
TR_MATH_ABI(tr_prob_stats_view, TR_PROB_ABI);
TR_MATH_ABI(tr_prob_stats_info, TR_PROB_ABI);
TR_MATH_ABI(tr_prob_outside, TR_PROB_ABI);
TR_MATH_ABI(tr_prob_plan_view, TR_PROB_ABI);
TR_MATH_ABI(tr_prob_answer, TR_PROB_ABI);

_Static_assert(sizeof(tr_prob_support_view) == 64, "tr_prob_support_view ABI");
_Static_assert(sizeof(tr_prob_model_view) == 136, "tr_prob_model_view ABI");
_Static_assert(sizeof(tr_prob_stats_view) == 64, "tr_prob_stats_view ABI");
_Static_assert(sizeof(tr_prob_stats_info) == 104, "tr_prob_stats_info ABI");
_Static_assert(sizeof(tr_prob_outside) == 48, "tr_prob_outside ABI");
_Static_assert(sizeof(tr_prob_plan_view) == 208, "tr_prob_plan_view ABI");
_Static_assert(sizeof(tr_prob_answer) == 296, "tr_prob_answer ABI");

#define SUPPORT_MAGIC UINT32_C(0x50505553)
#define MODEL_MAGIC UINT32_C(0x4c444f4d)
#define STATS_MAGIC UINT32_C(0x54415453)
#define PLAN_MAGIC UINT32_C(0x4e414c50)

/* floor(2^512 / ln 2), little-endian limbs: CALC-003's reduction constant (math_calc.c, LOG2E_512) */
static const uint64_t PROB_LOG2E_512[9] = {0x897f5e06a7be7366ULL, 0xd52173cc1895213fULL, 0x49b25eeb82d7c167ULL,
                                           0xbc3887eeaa2ed9acULL, 0x164a2cd9a342648fULL, 0xd6aef551bad2b4b1ULL,
                                           0x7d0ffda0d23a7d11ULL, 0x71547652b82fe177ULL, 0x0000000000000001ULL};

/* ================================================================ small helpers */
static void set_error(int32_t *e, int32_t v) { if (e) *e = v; }
static void *zalloc(uint64_t bytes) {
    if (bytes >= (uint64_t)SIZE_MAX / 2) return NULL;
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
static void digest_bytes(uint64_t *d, const void *p, uint64_t n) {
    const unsigned char *c = (const unsigned char *)p;
    uint64_t h1 = d[0], h2 = d[1];
    for (uint64_t i = 0; i < n; i++) {
        h1 = (h1 ^ c[i]) * UINT64_C(1099511628211);
        h2 = (h2 ^ c[i]) * UINT64_C(0xff51afd7ed558ccd);
        h2 ^= h2 >> 29;
    }
    d[0] = h1; d[1] = h2;
}
static uint64_t gcd64(uint64_t a, uint64_t b) { return tr_gcd_u64(a, b); }
/* lcm(a, b) of positive int64 values into *out: 0 on overflow */
static int lcm64(int64_t a, int64_t b, int64_t *out) {
    if (a % b == 0) { *out = a; return 1; }
    if (b % a == 0) { *out = b; return 1; }
    uint64_t g = gcd64((uint64_t)a, (uint64_t)b);
    return tr_mul_i64(a / (int64_t)g, b, out);
}
/* RD / RU of v 2^e for a finite double v (ldexp is exact unless the result overflows or is subnormal) */
static double ldexp_dir(double v, int64_t e, int up) {
    if (v == 0) return 0.0;
    if (e > 4000) return up ? (v > 0 ? INFINITY : -DBL_MAX) : (v > 0 ? DBL_MAX : -INFINITY);
    if (e < -4000) {
        if (v > 0) return up ? 0x1p-1074 : 0.0;
        return up ? -0.0 : -0x1p-1074;
    }
    double r = ldexp(v, (int)e);
    if (r == INFINITY) return up ? r : DBL_MAX;
    if (r == -INFINITY) return up ? -DBL_MAX : r;
    if (fabs(r) >= DBL_MIN || (r != 0 && ldexp(r, (int)-e) == v)) return r;   /* normal results are exact */
    if (r == 0) return (up == (v > 0)) ? (v > 0 ? 0x1p-1074 : -0x1p-1074) : 0.0;
    return tr_step(r, up);
}
static tr_math_interval iv_make(double lo, double hi) {
    tr_math_interval r;
    r.lo = lo == 0 ? 0.0 : lo; r.hi = hi == 0 ? 0.0 : hi;
    return r;
}
static tr_math_interval iv_nan(void) { tr_math_interval r; r.lo = NAN; r.hi = NAN; return r; }
/* the bit length of a 256-bit magnitude */
static int32_t w_bitlen(tr_dist_i256 m) {
    for (int i = 3; i >= 0; i--) if (m.w[i]) return 64 * i + tr_prob_bitlen64(m.w[i]);
    return 0;
}
/* [RD(v 2^e), RU(v 2^e)] of a signed 256-bit v */
static tr_math_interval enclose_w(tr_dist_i256 v, int64_t e) {
    int neg = tr_dist_w_negative(v);
    tr_dist_i256 m = neg ? tr_dist_w_negate_bits(v) : v;
    int32_t B = w_bitlen(m);
    if (B == 0) return iv_make(0.0, 0.0);
    double lo, hi;
    if (B <= 53) { lo = hi = (double)m.w[0]; }
    else {
        int32_t sh = B - 53;
        uint64_t t = 0;
        int rem = 0;
        __int128 q = 0;
        /* t = floor(m / 2^sh) < 2^53, rem: any bit below */
        tr_prob_shift_round(m, sh, 0, &q);
        t = (uint64_t)q;
        for (int32_t i = 0; i < 4 && !rem; i++) {
            int32_t lo_bit = 64 * i;
            if (lo_bit >= sh) break;
            uint64_t word = m.w[i];
            if (lo_bit + 64 > sh) word &= (((uint64_t)1 << (sh - lo_bit)) - 1);
            rem = word != 0;
        }
        lo = ldexp((double)t, sh);
        hi = ldexp((double)(t + (uint64_t)rem), sh);
    }
    if (neg) { double x = lo; lo = -hi; hi = -x; }
    return iv_make(ldexp_dir(lo, e, 0), ldexp_dir(hi, e, 1));
}
static tr_math_interval enclose_i128(__int128 v, int64_t e) { return enclose_w(tr_dist_w_from_i128(v), e); }
static tr_math_interval enclose_u64(uint64_t v) { return enclose_w(tr_dist_w_from_i128((__int128)v), 0); }
static tr_dist_i256 w_from(__int128 v) { return tr_dist_w_from_i128(v); }
/* a + b (checked: the bounds of the header keep every value below 2^255) */
static int w_add(tr_dist_i256 a, tr_dist_i256 b, tr_dist_i256 *r) { return tr_dist_w_add(a, b, r); }
static int w_sub(tr_dist_i256 a, tr_dist_i256 b, tr_dist_i256 *r) { return tr_dist_w_sub(a, b, r); }
/* a * u for a nonnegative 64-bit u */
static int w_mul_u64(tr_dist_i256 a, uint64_t u, tr_dist_i256 *r) {
    int na = tr_dist_w_negative(a);
    tr_dist_i256 m = na ? tr_dist_w_negate_bits(a) : a, p;
    uint64_t carry = 0;
    for (int i = 0; i < 4; i++) {
        unsigned __int128 t = (unsigned __int128)m.w[i] * u + carry;
        p.w[i] = (uint64_t)t; carry = (uint64_t)(t >> 64);
    }
    if (carry || (p.w[3] >> 63)) return 0;
    *r = na ? tr_dist_w_negate_bits(p) : p;
    return 1;
}
static __int128 i128_of(tr_math_i128 v) { return tr_i128_get(v); }

/* ================================================================ supports */
struct tr_prob_support_s {
    uint32_t magic;
    int32_t k;
    uint64_t d, bytes;
    int32_t scale[3], reserved;
    int64_t *value;            /* [d x k] */
    uint8_t *nulls;            /* [d] */
    uint64_t digest[2];
};
static int support_valid(const tr_prob_support *s) { return s && s->magic == SUPPORT_MAGIC; }
static int point_cmp(int32_t k, const int64_t *va, uint8_t na, const int64_t *vb, uint8_t nb) {
    for (int32_t c = 0; c < k; c++) {
        int a = (na >> c) & 1, b = (nb >> c) & 1;
        if (a != b) return a - b;                                     /* a value before NULL */
        if (!a && va[c] != vb[c]) return va[c] < vb[c] ? -1 : 1;
    }
    return 0;
}
static uint64_t support_bytes(int32_t k, uint64_t d) {
    return tr_budget_add(sizeof(tr_prob_support), tr_budget_mul(d, tr_budget_add(tr_budget_mul((uint64_t)k, 8), 1)));
}
static void support_digest(tr_prob_support *s) {
    s->digest[0] = UINT64_C(14695981039346656037); s->digest[1] = UINT64_C(0x9e3779b97f4a7c15);
    digest_bytes(s->digest, &s->k, sizeof s->k);
    digest_bytes(s->digest, &s->d, sizeof s->d);
    digest_bytes(s->digest, s->scale, sizeof s->scale);
    digest_bytes(s->digest, s->value, s->d * (uint64_t)s->k * 8);
    digest_bytes(s->digest, s->nulls, s->d);
}
static tr_prob_support *support_alloc(int32_t k, uint64_t d, const int32_t *scales, int32_t *error) {
    tr_prob_support *s = (tr_prob_support *)zalloc(sizeof *s);
    if (!s) { set_error(error, TR_MATH_MEMORY); return NULL; }
    s->k = k; s->d = d;
    for (int32_t c = 0; c < 3; c++) s->scale[c] = c < k ? scales[c] : 0;
    s->value = (int64_t *)zalloc(tr_budget_mul(tr_budget_mul(d, (uint64_t)k), 8));
    s->nulls = (uint8_t *)zalloc(d);
    if (!s->value || !s->nulls) { free(s->value); free(s->nulls); free(s); set_error(error, TR_MATH_MEMORY); return NULL; }
    s->bytes = support_bytes(k, d);
    return s;
}
void tr_prob_support_free(tr_prob_support *s) {
    if (!support_valid(s)) return;
    s->magic = 0;
    free(s->value); free(s->nulls); free(s);
}
/* a stable merge sort of the indices by point (no recursion, buffer given) */
static void sort_points(int32_t k, const int64_t *val, const uint8_t *nul, uint64_t *idx, uint64_t *tmp, uint64_t n) {
    for (uint64_t w = 1; w < n; w *= 2) {
        for (uint64_t s = 0; s < n; s += 2 * w) {
            uint64_t m = s + w < n ? s + w : n, e = s + 2 * w < n ? s + 2 * w : n, i = s, j = m, o = s;
            while (i < m && j < e) {
                uint64_t a = idx[i], b = idx[j];
                tmp[o++] = point_cmp(k, val + b * (uint64_t)k, nul[b], val + a * (uint64_t)k, nul[a]) < 0 ? idx[j++] : idx[i++];
            }
            while (i < m) tmp[o++] = idx[i++];
            while (j < e) tmp[o++] = idx[j++];
        }
        memcpy(idx, tmp, (size_t)n * 8);
    }
}
tr_prob_support *tr_prob_support_create(int32_t k, uint64_t d, const int64_t *values, const uint8_t *nulls,
                                        const int32_t *scales, uint64_t *perm, const tr_math_budget *budget,
                                        tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    if (k > TR_PROB_MAX_DIMS) { set_error(error, TR_MATH_UNSUPPORTED); return NULL; }
    if (k < 1 || d < 1 || !values || !scales || d > TR_PROB_MAX_POINTS) { set_error(error, TR_MATH_INVALID); return NULL; }
    for (int32_t c = 0; c < k; c++)
        if (scales[c] < 0 || scales[c] > TR_SCOPE_MAX_SCALE) { set_error(error, TR_MATH_INVALID); return NULL; }
    if (nulls) for (uint64_t i = 0; i < d; i++) if (nulls[i] >> k) { set_error(error, TR_MATH_INVALID); return NULL; }
    tr_math_budget dflt;
    const tr_math_budget *bud = budget_of(budget, &dflt);
    tr_math_need need;
    need_zero(&need);
    need.need[TR_BUDGET_VARIABLES] = (uint64_t)k;
    need.need[TR_BUDGET_SUPPORT] = d;
    need.need[TR_BUDGET_BYTES] = tr_budget_add(support_bytes(k, d), tr_budget_mul(d, 16));   /* + the sort's indices */
    int32_t rc = check_need(bud, &need, refusal);
    if (rc) { set_error(error, rc); return NULL; }
    uint64_t *idx = (uint64_t *)zalloc(tr_budget_mul(d, 8)), *tmp = (uint64_t *)zalloc(tr_budget_mul(d, 8));
    uint8_t *nz = (uint8_t *)zalloc(d);
    if (!idx || !tmp || !nz) { free(idx); free(tmp); free(nz); set_error(error, TR_MATH_MEMORY); return NULL; }
    for (uint64_t i = 0; i < d; i++) { idx[i] = i; nz[i] = nulls ? nulls[i] : 0; }
    sort_points(k, values, nz, idx, tmp, d);
    tr_prob_support *s = support_alloc(k, d, scales, &rc);
    if (!s) { free(idx); free(tmp); free(nz); set_error(error, rc); return NULL; }
    for (uint64_t i = 0; i < d; i++) {
        uint64_t j = idx[i];
        s->nulls[i] = nz[j];
        for (int32_t c = 0; c < k; c++) s->value[i * (uint64_t)k + (uint64_t)c] = (nz[j] >> c) & 1 ? 0 : values[j * (uint64_t)k + (uint64_t)c];
        if (i && point_cmp(k, s->value + (i - 1) * (uint64_t)k, s->nulls[i - 1], s->value + i * (uint64_t)k, s->nulls[i]) == 0)
            rc = TR_MATH_INVALID;                                      /* a duplicate point */
    }
    if (!rc && perm) memcpy(perm, idx, (size_t)d * 8);
    free(idx); free(tmp); free(nz);
    if (rc) { s->magic = SUPPORT_MAGIC; tr_prob_support_free(s); set_error(error, rc); return NULL; }
    support_digest(s);
    s->magic = SUPPORT_MAGIC;
    set_error(error, TR_MATH_OK);
    return s;
}
static tr_prob_support *support_copy(const tr_prob_support *a, int32_t *error) {
    tr_prob_support *s = support_alloc(a->k, a->d, a->scale, error);
    if (!s) return NULL;
    memcpy(s->value, a->value, (size_t)(a->d * (uint64_t)a->k * 8));
    memcpy(s->nulls, a->nulls, (size_t)a->d);
    s->digest[0] = a->digest[0]; s->digest[1] = a->digest[1];
    s->magic = SUPPORT_MAGIC;
    return s;
}
tr_prob_support *tr_prob_support_product(const tr_prob_support *a, const tr_prob_support *b,
                                         const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    if (!support_valid(a) || !support_valid(b)) { set_error(error, TR_MATH_INVALID); return NULL; }
    int32_t k = a->k + b->k;
    if (k > TR_PROB_MAX_DIMS) { set_error(error, TR_MATH_UNSUPPORTED); return NULL; }
    uint64_t d = tr_budget_mul(a->d, b->d);
    tr_math_budget dflt;
    const tr_math_budget *bud = budget_of(budget, &dflt);
    tr_math_need need;
    need_zero(&need);
    need.need[TR_BUDGET_VARIABLES] = (uint64_t)k;
    need.need[TR_BUDGET_SUPPORT] = d;
    need.need[TR_BUDGET_BYTES] = support_bytes(k, d);
    int32_t rc = check_need(bud, &need, refusal);
    if (!rc && d > TR_PROB_MAX_POINTS) rc = TR_MATH_UNSUPPORTED;
    if (rc) { set_error(error, rc); return NULL; }
    int32_t scales[3] = {0, 0, 0};
    for (int32_t c = 0; c < k; c++) scales[c] = c < a->k ? a->scale[c] : b->scale[c - a->k];
    tr_prob_support *s = support_alloc(k, d, scales, &rc);
    if (!s) { set_error(error, rc); return NULL; }
    for (uint64_t i = 0; i < a->d; i++)
        for (uint64_t j = 0; j < b->d; j++) {
            uint64_t p = i * b->d + j;
            for (int32_t c = 0; c < a->k; c++) s->value[p * (uint64_t)k + (uint64_t)c] = a->value[i * (uint64_t)a->k + (uint64_t)c];
            for (int32_t c = 0; c < b->k; c++) s->value[p * (uint64_t)k + (uint64_t)(a->k + c)] = b->value[j * (uint64_t)b->k + (uint64_t)c];
            s->nulls[p] = (uint8_t)(a->nulls[i] | (b->nulls[j] << a->k));
        }
    support_digest(s);
    s->magic = SUPPORT_MAGIC;
    set_error(error, TR_MATH_OK);
    return s;
}
int32_t tr_prob_support_get_view(const tr_prob_support *s, tr_prob_support_view *out) {
    if (!support_valid(s) || !out) return TR_MATH_INVALID;
    memset(out, 0, sizeof *out);
    out->k = s->k; out->d = s->d;
    for (int32_t c = 0; c < 3; c++) out->scale[c] = s->scale[c];
    out->value = s->value; out->nulls = s->nulls;
    out->digest[0] = s->digest[0]; out->digest[1] = s->digest[1];
    return TR_MATH_OK;
}
/* the index of a point by binary search (cell order); 1 found, 0 not */
static int support_search(const tr_prob_support *s, const int64_t *v, uint8_t nulls, uint64_t *index) {
    uint64_t lo = 0, hi = s->d;
    while (lo < hi) {
        uint64_t m = lo + (hi - lo) / 2;
        int c = point_cmp(s->k, s->value + m * (uint64_t)s->k, s->nulls[m], v, nulls);
        if (c == 0) { *index = m; return 1; }
        if (c < 0) lo = m + 1; else hi = m;
    }
    return 0;
}
int32_t tr_prob_support_find(const tr_prob_support *s, const int64_t *values, uint8_t nulls, uint64_t *index) {
    if (!support_valid(s) || !values || !index || (nulls >> s->k)) return TR_MATH_INVALID;
    int64_t v[3] = {0, 0, 0};
    for (int32_t c = 0; c < s->k; c++) v[c] = (nulls >> c) & 1 ? 0 : values[c];
    return support_search(s, v, nulls, index);
}
static int support_same(const tr_prob_support *a, const tr_prob_support *b) {
    if (a == b) return 1;
    if (a->k != b->k || a->d != b->d || a->digest[0] != b->digest[0] || a->digest[1] != b->digest[1]) return 0;
    for (int32_t c = 0; c < 3; c++) if (a->scale[c] != b->scale[c]) return 0;
    return memcmp(a->value, b->value, (size_t)(a->d * (uint64_t)a->k * 8)) == 0 && memcmp(a->nulls, b->nulls, (size_t)a->d) == 0;
}
int32_t tr_prob_support_equal(const tr_prob_support *a, const tr_prob_support *b) {
    if (!support_valid(a) || !support_valid(b)) return TR_MATH_INVALID;
    return support_same(a, b);
}

/* ================================================================ models */
struct tr_prob_model_s {
    uint32_t magic;
    int32_t kind, m, scale;
    uint64_t d, bytes;
    tr_prob_support *support;  /* owned copy */
    int64_t *feat, *fden, *base;
    int32_t *hbits;
    tr_math_i128 *lnh;
    int64_t hden;
    int32_t bits[16];          /* bitlen(max_i |N_ij|) */
    int32_t pbits[136];        /* bitlen(max_i |N_ij N_il|), packed j <= l */
};
static int model_valid(const tr_prob_model *m) { return m && m->magic == MODEL_MAGIC; }
void tr_prob_model_free(tr_prob_model *m) {
    if (!model_valid(m)) return;
    m->magic = 0;
    tr_prob_support_free(m->support);
    free(m->feat); free(m->fden); free(m->base); free(m->hbits); free(m->lnh); free(m);
}
static uint64_t model_bytes(uint64_t d, int32_t m, int32_t kind) {
    uint64_t b = tr_budget_add(sizeof(tr_prob_model), tr_budget_mul(tr_budget_mul(d, (uint64_t)m), 8));
    b = tr_budget_add(b, (uint64_t)m * 8);
    if (kind == TR_PROB_EXPFAM) b = tr_budget_add(b, tr_budget_mul(d, 8 + 4 + 32));
    return b;
}
tr_prob_model *tr_prob_model_create(const tr_prob_support *support, int32_t kind, int32_t m, const int64_t *fnum,
                                    const int64_t *fden, const int64_t *bnum, const int64_t *bden,
                                    const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    if (!support_valid(support) || (kind != TR_PROB_CATEGORICAL && kind != TR_PROB_EXPFAM) || m < 0) {
        set_error(error, TR_MATH_INVALID); return NULL;
    }
    if (m > TR_PROB_MAX_FEATURES) { set_error(error, TR_MATH_UNSUPPORTED); return NULL; }
    if ((kind == TR_PROB_EXPFAM && m < 1) || (m && !fnum) || (kind == TR_PROB_CATEGORICAL && (bnum || bden))) {
        set_error(error, TR_MATH_INVALID); return NULL;
    }
    const uint64_t d = support->d;
    tr_math_budget dflt;
    const tr_math_budget *bud = budget_of(budget, &dflt);
    tr_math_need need;
    need_zero(&need);
    need.need[TR_BUDGET_VARIABLES] = (uint64_t)m;
    need.need[TR_BUDGET_SUPPORT] = d;
    need.need[TR_BUDGET_TENSOR] = tr_budget_mul(d, (uint64_t)m);
    need.need[TR_BUDGET_BYTES] = tr_budget_add(model_bytes(d, m, kind), support->bytes);
    int32_t rc = check_need(bud, &need, refusal);
    if (rc) { set_error(error, rc); return NULL; }
    tr_prob_model *M = (tr_prob_model *)zalloc(sizeof *M);
    if (!M) { set_error(error, TR_MATH_MEMORY); return NULL; }
    M->magic = MODEL_MAGIC; M->kind = kind; M->m = m; M->d = d;
    M->bytes = need.need[TR_BUDGET_BYTES];
    M->support = support_copy(support, &rc);
    if (m) {
        M->feat = (int64_t *)zalloc(tr_budget_mul(tr_budget_mul(d, (uint64_t)m), 8));
        M->fden = (int64_t *)zalloc((uint64_t)m * 8);
        if (!M->feat || !M->fden) rc = TR_MATH_MEMORY;
    }
    if (!rc && kind == TR_PROB_EXPFAM) {
        M->base = (int64_t *)zalloc(tr_budget_mul(d, 8));
        M->hbits = (int32_t *)zalloc(tr_budget_mul(d, 4));
        M->lnh = (tr_math_i128 *)zalloc(tr_budget_mul(d, 32));
        if (!M->base || !M->hbits || !M->lnh) rc = TR_MATH_MEMORY;
    }
    if (!M->support && !rc) rc = TR_MATH_MEMORY;
    /* features: each over the lcm of its (reduced) denominators */
    for (int32_t j = 0; j < m && !rc; j++) {
        int64_t D = 1;
        for (uint64_t i = 0; i < d && !rc; i++) {
            int64_t den = fden ? fden[i * (uint64_t)m + (uint64_t)j] : 1, num = fnum[i * (uint64_t)m + (uint64_t)j];
            if (den < 1) { rc = TR_MATH_INVALID; break; }
            uint64_t g = gcd64(tr_abs_u64(num), (uint64_t)den);
            if (g > 1) den /= (int64_t)g;
            if (!lcm64(D, den, &D)) rc = TR_MATH_OVERFLOW;
        }
        if (rc) break;
        M->fden[j] = D;
        uint64_t fmax = 0;
        for (uint64_t i = 0; i < d; i++) {
            int64_t den = fden ? fden[i * (uint64_t)m + (uint64_t)j] : 1, num = fnum[i * (uint64_t)m + (uint64_t)j];
            uint64_t g = gcd64(tr_abs_u64(num), (uint64_t)den);
            if (g > 1) { den /= (int64_t)g; num /= (int64_t)g; }
            int64_t v;
            if (!tr_mul_i64(num, D / den, &v) || v == INT64_MIN) { rc = TR_MATH_OVERFLOW; break; }
            M->feat[i * (uint64_t)m + (uint64_t)j] = v;
            if (tr_abs_u64(v) > fmax) fmax = tr_abs_u64(v);
        }
        M->bits[j] = tr_prob_bitlen64(fmax);
    }
    for (int32_t j = 0, p = 0; j < m && !rc; j++)
        for (int32_t l = j; l < m; l++, p++) {
            unsigned __int128 pm = 0;
            for (uint64_t i = 0; i < d; i++) {
                unsigned __int128 x = (unsigned __int128)tr_abs_u64(M->feat[i * (uint64_t)m + (uint64_t)j]) *
                                      tr_abs_u64(M->feat[i * (uint64_t)m + (uint64_t)l]);
                if (x > pm) pm = x;
            }
            M->pbits[p] = tr_prob_bitlen128(pm);
        }
    /* the base measure */
    if (!rc && kind == TR_PROB_EXPFAM) {
        int64_t D = 1;
        for (uint64_t i = 0; i < d && !rc; i++) {
            int64_t num = bnum ? bnum[i] : 1, den = bden ? bden[i] : 1;
            if (den < 1) { rc = TR_MATH_INVALID; break; }
            if (num < 0) { rc = TR_MATH_DOMAIN_ERROR; break; }
            uint64_t g = gcd64((uint64_t)num, (uint64_t)den);
            if (num && g > 1) den /= (int64_t)g;
            if (num && !lcm64(D, den, &D)) rc = TR_MATH_OVERFLOW;
        }
        int any = 0;
        for (uint64_t i = 0; i < d && !rc; i++) {
            int64_t num = bnum ? bnum[i] : 1, den = bden ? bden[i] : 1, v = 0;
            if (num) {
                uint64_t g = gcd64((uint64_t)num, (uint64_t)den);
                if (g > 1) { den /= (int64_t)g; num /= (int64_t)g; }
                if (!tr_mul_i64(num, D / den, &v)) { rc = TR_MATH_OVERFLOW; break; }
            }
            M->base[i] = v;
            M->hbits[i] = tr_prob_bitlen64((uint64_t)v);
            any |= v > 0;
            if (v > 1) {
                unsigned __int128 lo, hi;
                tr_info_ln((uint64_t)v, &lo, &hi);
                M->lnh[2 * i] = tr_i128_set((__int128)lo); M->lnh[2 * i + 1] = tr_i128_set((__int128)hi);
            }
        }
        if (!rc && !any) rc = TR_MATH_DOMAIN_ERROR;                    /* Z = 0 for every theta */
        M->hden = D;
        int32_t bl = tr_prob_bitlen64(d);
        M->scale = 124 - bl < TR_PROB_F ? 124 - bl : TR_PROB_F;
    }
    if (rc) { tr_prob_model_free(M); set_error(error, rc); return NULL; }
    set_error(error, TR_MATH_OK);
    return M;
}
int32_t tr_prob_model_get_view(const tr_prob_model *M, tr_prob_model_view *out) {
    if (!model_valid(M) || !out) return TR_MATH_INVALID;
    memset(out, 0, sizeof *out);
    out->kind = M->kind; out->m = M->m; out->d = M->d; out->scale = M->scale;
    for (int32_t j = 0; j < 16; j++) out->bits[j] = M->bits[j];
    out->feat = M->feat; out->fden = M->fden; out->base = M->base; out->hbits = M->hbits; out->lnh = M->lnh;
    out->hden = M->hden;
    return TR_MATH_OK;
}
const tr_prob_support *tr_prob_model_support(const tr_prob_model *M) { return model_valid(M) ? M->support : NULL; }

/* ================================================================ retained statistics */
struct tr_prob_stats_s {
    uint32_t magic;
    int32_t k, mass, declared, product, scoped, source_kind, keyed;
    uint32_t exclude, reserved;
    uint64_t G, d, bytes, present;
    tr_prob_support *support;  /* owned copy */
    uint64_t *counts, *total, *rows, *dropped;
    uint64_t hdigest[2];
    unsigned char *source;     /* the data source (GEN-002's rule): table address + token, or the rows descriptor */
    uint64_t source_len;
    unsigned char *grouping;   /* TR_SCOPE_KEY_GROUPS */
    uint64_t grouping_len;
};
static int stats_valid(const tr_prob_stats *s) { return s && s->magic == STATS_MAGIC; }
void tr_prob_stats_free(tr_prob_stats *s) {
    if (!stats_valid(s)) return;
    s->magic = 0;
    tr_prob_support_free(s->support);
    free(s->counts); free(s->total); free(s->rows); free(s->dropped); free(s->source); free(s->grouping); free(s);
}
static uint64_t stats_bytes(uint64_t G, uint64_t d, uint64_t extra) {
    uint64_t b = tr_budget_add(sizeof(tr_prob_stats), tr_budget_mul(tr_budget_mul(G, d), 8));
    return tr_budget_add(tr_budget_add(b, tr_budget_mul(G, 24)), extra);
}
/* the source key and grouping descriptor of a scope (sizes; table: 24 bytes) */
static int32_t scope_keys(const tr_scope *scope, tr_scope_info *si, uint64_t *slen, uint64_t *glen) {
    if (tr_scope_get_info(scope, si)) return TR_MATH_INVALID;
    *slen = si->source == TR_SCOPE_TABLE ? 24 : tr_scope_key_size(scope, TR_SCOPE_KEY_ROWS);
    *glen = tr_scope_key_size(scope, TR_SCOPE_KEY_GROUPS);
    return TR_MATH_OK;
}
static int32_t write_source(const tr_scope *scope, const tr_scope_info *si, unsigned char *out, uint64_t len) {
    if (si->source == TR_SCOPE_TABLE) {
        memcpy(out, &si->table, 8);
        memcpy(out + 8, si->identity, 16);
        return TR_MATH_OK;
    }
    return tr_scope_key_write(scope, TR_SCOPE_KEY_ROWS, out, len);
}
typedef struct {
    tr_dist_states st;
    tr_dist_plan plan;
    tr_dist_hist_info hi;
} hist_src;
static int32_t hist_open(const tr_dist_hist *h, const tr_scope *scope, const tr_prob_support *sp, int32_t mass,
                         uint32_t exclude, hist_src *s, tr_scope_info *si) {
    memset(s, 0, sizeof *s);
    if (!h || !support_valid(sp) || tr_dist_hist_get_states(h, &s->st) || tr_dist_hist_get_info(h, &s->hi))
        return TR_MATH_INVALID;
    if ((mass != TR_DIST_MASS_COUNT && mass != TR_DIST_MASS_WEIGHT) || (exclude >> s->hi.k)) return TR_MATH_INVALID;
    if (mass == TR_DIST_MASS_WEIGHT && !s->hi.weighted) return TR_MATH_INVALID;
    if (scope) {
        if (tr_scope_get_info(scope, si) || !s->hi.scoped || s->hi.digest[0] != si->digest[0] ||
            s->hi.digest[1] != si->digest[1] || s->hi.ngroups != si->ngroups || s->hi.rows != si->selected)
            return TR_MATH_INVALID;
    }
    if (s->hi.k != sp->k) return TR_MATH_DOMAIN_ERROR;
    for (int32_t c = 0; c < sp->k; c++) if (s->hi.scale[c] != sp->scale[c]) return TR_MATH_DOMAIN_ERROR;
    if (mass == TR_DIST_MASS_WEIGHT && !s->hi.wnonneg) return TR_MATH_DOMAIN_ERROR;
    if (s->st.repr == TR_DIST_DENSE) tr_dist_states_plan(&s->st, &s->plan);
    return TR_MATH_OK;
}
static void hist_group(const hist_src *s, uint64_t g, uint64_t *b, uint64_t *e) {
    if (s->st.repr == TR_DIST_DENSE) { *b = g * s->st.cells; *e = *b + s->st.cells; }
    else { *b = s->st.offset[g]; *e = s->st.offset[g + 1]; }
}
static void hist_cell(const hist_src *s, uint64_t g, uint64_t i, int64_t *v, uint8_t *nulls) {
    *nulls = 0;
    v[0] = v[1] = v[2] = 0;
    if (s->st.repr == TR_DIST_DENSE) {
        uint64_t idx = i - g * s->st.cells;
        for (int c = 0; c < s->st.k; c++) {
            int null;
            v[c] = tr_dist_plan_coord(&s->plan, idx, c, &null);
            if (null) { *nulls |= (uint8_t)(1u << c); v[c] = 0; }
        }
    } else {
        *nulls = s->st.nulls[i];
        for (int c = 0; c < s->st.k; c++) v[c] = (*nulls >> c) & 1 ? 0 : s->st.value[c][i];
    }
}
static void stats_need_of(uint64_t G, uint64_t d, uint64_t extra, tr_math_need *need) {
    need_zero(need);
    need->need[TR_BUDGET_SUPPORT] = d;
    need->need[TR_BUDGET_STATES] = tr_budget_mul(G, d);
    need->need[TR_BUDGET_BYTES] = stats_bytes(G, d, extra);
}
int32_t tr_prob_stats_need(const tr_dist_hist *h, const tr_scope *scope, const tr_prob_support *sp, int32_t mass,
                           uint32_t exclude, tr_math_need *need) {
    hist_src s;
    tr_scope_info si;
    if (!need) return TR_MATH_INVALID;
    int32_t rc = hist_open(h, scope, sp, mass, exclude, &s, &si);
    if (rc) return rc;
    uint64_t slen = 0, glen = 0;
    if (scope && scope_keys(scope, &si, &slen, &glen)) return TR_MATH_INVALID;
    stats_need_of(s.st.ngroups, sp->d, tr_budget_add(tr_budget_add(slen, glen), sp->bytes), need);
    return TR_MATH_OK;
}
tr_prob_stats *tr_prob_stats_build(const tr_dist_hist *h, const tr_scope *scope, const tr_prob_support *sp,
                                   int32_t mass, uint32_t exclude, const tr_math_budget *budget,
                                   tr_math_refusal *refusal, tr_prob_outside *outside, int32_t *error) {
    tr_budget_none(refusal);
    hist_src src;
    tr_scope_info si;
    int32_t rc = hist_open(h, scope, sp, mass, exclude, &src, &si);
    if (rc) { set_error(error, rc); return NULL; }
    uint64_t slen = 0, glen = 0;
    if (scope && scope_keys(scope, &si, &slen, &glen)) { set_error(error, TR_MATH_INVALID); return NULL; }
    tr_math_budget dflt;
    const tr_math_budget *bud = budget_of(budget, &dflt);
    const uint64_t G = src.st.ngroups, d = sp->d;
    tr_math_need need;
    stats_need_of(G, d, tr_budget_add(tr_budget_add(slen, glen), sp->bytes), &need);
    rc = check_need(bud, &need, refusal);
    if (rc) { set_error(error, rc); return NULL; }
    tr_prob_stats *S = (tr_prob_stats *)zalloc(sizeof *S);
    if (!S) { set_error(error, TR_MATH_MEMORY); return NULL; }
    S->magic = STATS_MAGIC; S->k = sp->k; S->mass = mass; S->exclude = exclude; S->G = G; S->d = d;
    S->bytes = need.need[TR_BUDGET_BYTES];
    S->support = support_copy(sp, &rc);
    S->counts = (uint64_t *)zalloc(tr_budget_mul(tr_budget_mul(G, d), 8));
    S->total = (uint64_t *)zalloc(tr_budget_mul(G, 8));
    S->rows = (uint64_t *)zalloc(tr_budget_mul(G, 8));
    S->dropped = (uint64_t *)zalloc(tr_budget_mul(G, 8));
    if (!S->support || !S->counts || !S->total || !S->rows || !S->dropped) rc = TR_MATH_MEMORY;
    if (!rc && scope) {
        S->source = (unsigned char *)zalloc(slen); S->grouping = (unsigned char *)zalloc(glen);
        if (!S->source || !S->grouping) rc = TR_MATH_MEMORY;
        if (!rc) rc = write_source(scope, &si, S->source, slen);
        if (!rc) rc = tr_scope_key_write(scope, TR_SCOPE_KEY_GROUPS, S->grouping, glen);
        S->source_len = slen; S->grouping_len = glen; S->scoped = 1; S->source_kind = si.source;
        S->keyed = si.ngroups > 1;
    }
    if (!rc && src.hi.scoped) { S->hdigest[0] = src.hi.digest[0]; S->hdigest[1] = src.hi.digest[1]; }
    for (uint64_t g = 0; g < G && !rc; g++) {
        uint64_t b, e;
        hist_group(&src, g, &b, &e);
        S->rows[g] = src.st.rows[g];
        if (src.st.rows[g]) S->present++;
        unsigned __int128 t = 0, dropped = 0;
        for (uint64_t i = b; i < e; i++) {
            if (!src.st.count[i]) continue;                            /* a structural zero: not observed */
            unsigned __int128 m = mass == TR_DIST_MASS_COUNT ? (unsigned __int128)src.st.count[i]
                                                             : (unsigned __int128)tr_i128_get(src.st.wsum[i]);
            int64_t v[3];
            uint8_t nl;
            hist_cell(&src, g, i, v, &nl);
            if (nl & exclude) { dropped += m; if (dropped >> 64) dropped = UINT64_MAX; continue; }
            uint64_t at;
            if (!support_search(sp, v, nl, &at)) {
                if (outside) {
                    memset(outside, 0, sizeof *outside);
                    outside->group = g; outside->mass = (uint64_t)src.st.count[i]; outside->nulls = nl;
                    for (int c = 0; c < 3; c++) outside->value[c] = v[c];
                }
                rc = TR_MATH_DOMAIN_ERROR;                             /* an observed value outside the support */
                break;
            }
            t += m;                                                    /* each m < 2^127: no wrap before the check */
            if (t >> 64) { rc = TR_MATH_UNSUPPORTED; break; }
            S->counts[g * d + at] += (uint64_t)m;
        }
        S->total[g] = (uint64_t)t;
        S->dropped[g] = (uint64_t)dropped;
    }
    if (rc) { tr_prob_stats_free(S); set_error(error, rc); return NULL; }
    set_error(error, TR_MATH_OK);
    return S;
}
static void check_set(tr_conv_check *ck, int32_t reason, int32_t keyed, const unsigned char *src, uint64_t len) {
    if (!ck) return;
    memset(ck, 0, sizeof *ck);
    ck->reason = reason; ck->first = reason ? 0 : -1; ck->second = reason ? 1 : -1; ck->keyed = keyed;
    if (src) {
        uint64_t dg[2] = {UINT64_C(14695981039346656037), UINT64_C(0x9e3779b97f4a7c15)};
        digest_bytes(dg, src, len);
        ck->digest[0] = dg[0]; ck->digest[1] = dg[1];
    }
}
tr_prob_stats *tr_prob_stats_product(const tr_prob_stats *a, const tr_scope *sa, const tr_prob_stats *b,
                                     const tr_scope *sb, const tr_conv_decl *decl, const tr_math_budget *budget,
                                     tr_math_refusal *refusal, tr_conv_check *check, int32_t *error) {
    tr_budget_none(refusal);
    check_set(check, TR_CONV_OK, 0, NULL, 0);
    if (!stats_valid(a) || !stats_valid(b) || !sa || !sb || !a->scoped || !b->scoped || a->product || b->product) {
        set_error(error, TR_MATH_INVALID); return NULL;
    }
    /* the scopes given are the ones the states recorded (their sources and groupings, exactly) */
    tr_scope_info ia, ib;
    uint64_t la, ga, lb, gb;
    if (scope_keys(sa, &ia, &la, &ga) || scope_keys(sb, &ib, &lb, &gb) || la != a->source_len ||
        lb != b->source_len || ga != a->grouping_len || gb != b->grouping_len ||
        ia.ngroups != a->G || ib.ngroups != b->G || (int32_t)ia.source != a->source_kind ||
        (int32_t)ib.source != b->source_kind) { set_error(error, TR_MATH_INVALID); return NULL; }
    {
        uint64_t big = la > lb ? la : lb;
        if (ga > big) big = ga;
        if (gb > big) big = gb;
        unsigned char *t1 = (unsigned char *)zalloc(big);
        if (!t1) { set_error(error, TR_MATH_MEMORY); return NULL; }
        int ok = write_source(sa, &ia, t1, la) == 0 && memcmp(t1, a->source, (size_t)la) == 0;
        ok = ok && write_source(sb, &ib, t1, lb) == 0 && memcmp(t1, b->source, (size_t)lb) == 0;
        ok = ok && tr_scope_key_write(sa, TR_SCOPE_KEY_GROUPS, t1, ga) == 0 && memcmp(t1, a->grouping, (size_t)ga) == 0;
        ok = ok && tr_scope_key_write(sb, TR_SCOPE_KEY_GROUPS, t1, gb) == 0 && memcmp(t1, b->grouping, (size_t)gb) == 0;
        free(t1);
        if (!ok) { set_error(error, TR_MATH_INVALID); return NULL; }
    }
    if (a->mass != b->mass) { set_error(error, TR_MATH_DOMAIN_ERROR); return NULL; }
    int same = a->source_kind == b->source_kind && a->source_len == b->source_len &&
               memcmp(a->source, b->source, (size_t)a->source_len) == 0;
    int keyed = a->keyed | b->keyed;
    if (same) {
        /* GEN-002's linkage rule: rows of one source can be paired, so the product of the marginals is an
         * independence assumption: only under a declaration covering that source */
        int covered = 0;
        if (decl) {
            int32_t c = tr_conv_decl_covers(decl, sa);
            if (c < 0) { set_error(error, c); return NULL; }
            covered = c == 1;
        }
        if (!covered) {
            check_set(check, decl ? TR_CONV_NOT_COVERED : TR_CONV_NEEDS_DECLARATION, keyed, a->source, a->source_len);
            set_error(error, TR_MATH_DOMAIN_ERROR); return NULL;
        }
        if (a->grouping_len != b->grouping_len || memcmp(a->grouping, b->grouping, (size_t)a->grouping_len) != 0) {
            check_set(check, TR_CONV_GROUPING_DIFFERS, keyed, a->source, a->source_len);
            set_error(error, TR_MATH_DOMAIN_ERROR); return NULL;
        }
    } else if (a->G != b->G) {
        check_set(check, TR_CONV_KEYS_DIFFER, keyed, NULL, 0);
        set_error(error, TR_MATH_DOMAIN_ERROR); return NULL;
    }
    int32_t rc = TR_MATH_OK;
    tr_prob_support *sp = tr_prob_support_product(a->support, b->support, budget, refusal, &rc);
    if (!sp) { set_error(error, rc); return NULL; }
    tr_math_budget dflt;
    const tr_math_budget *bud = budget_of(budget, &dflt);
    const uint64_t G = a->G, d = sp->d;
    tr_math_need need;
    stats_need_of(G, d, tr_budget_add(sp->bytes, sp->bytes), &need);
    rc = check_need(bud, &need, refusal);
    if (rc) { tr_prob_support_free(sp); set_error(error, rc); return NULL; }
    tr_prob_stats *S = (tr_prob_stats *)zalloc(sizeof *S);
    if (!S) { tr_prob_support_free(sp); set_error(error, TR_MATH_MEMORY); return NULL; }
    S->magic = STATS_MAGIC; S->k = sp->k; S->mass = a->mass; S->G = G; S->d = d; S->support = sp;
    S->exclude = a->exclude | (b->exclude << a->k);
    S->declared = same; S->product = 1; S->keyed = keyed;
    S->bytes = need.need[TR_BUDGET_BYTES];
    S->counts = (uint64_t *)zalloc(tr_budget_mul(tr_budget_mul(G, d), 8));
    S->total = (uint64_t *)zalloc(tr_budget_mul(G, 8));
    S->rows = (uint64_t *)zalloc(tr_budget_mul(G, 8));
    S->dropped = (uint64_t *)zalloc(tr_budget_mul(G, 8));
    if (!S->counts || !S->total || !S->rows || !S->dropped) rc = TR_MATH_MEMORY;
    for (uint64_t g = 0; g < G && !rc; g++) {
        unsigned __int128 t = (unsigned __int128)a->total[g] * b->total[g];
        unsigned __int128 rws = (unsigned __int128)a->rows[g] * b->rows[g];
        if (t >> 64) { rc = TR_MATH_OVERFLOW; break; }
        S->total[g] = (uint64_t)t;
        S->rows[g] = (rws >> 64) ? UINT64_MAX : (uint64_t)rws;
        if (S->rows[g]) S->present++;
        for (uint64_t i = 0; i < a->d; i++)
            for (uint64_t j = 0; j < b->d; j++)
                S->counts[g * d + i * b->d + j] = a->counts[g * a->d + i] * b->counts[g * b->d + j];   /* <= t */
    }
    if (rc) { tr_prob_stats_free(S); set_error(error, rc); return NULL; }
    set_error(error, TR_MATH_OK);
    return S;
}
int32_t tr_prob_stats_get_view(const tr_prob_stats *S, tr_prob_stats_view *out) {
    if (!stats_valid(S) || !out) return TR_MATH_INVALID;
    memset(out, 0, sizeof *out);
    out->ngroups = S->G; out->d = S->d; out->k = S->k; out->mass = S->mass; out->exclude = S->exclude;
    out->declared = S->declared;
    out->counts = S->counts; out->total = S->total; out->rows = S->rows; out->dropped = S->dropped;
    return TR_MATH_OK;
}
int32_t tr_prob_stats_get_info(const tr_prob_stats *S, tr_prob_stats_info *out) {
    if (!stats_valid(S) || !out) return TR_MATH_INVALID;
    memset(out, 0, sizeof *out);
    out->ngroups = S->G; out->present = S->present; out->d = S->d; out->bytes = S->bytes;
    uint64_t t = 0;
    for (uint64_t g = 0; g < S->G; g++) t = tr_budget_add(t, S->total[g]);
    out->total = t;
    out->digest[0] = S->hdigest[0]; out->digest[1] = S->hdigest[1];
    out->support_digest[0] = S->support->digest[0]; out->support_digest[1] = S->support->digest[1];
    out->k = S->k; out->mass = S->mass; out->exclude = (int32_t)S->exclude; out->scoped = S->scoped;
    out->declared = S->declared; out->product = S->product; out->source_kind = S->source_kind; out->keyed = S->keyed;
    return TR_MATH_OK;
}
const tr_prob_support *tr_prob_stats_support(const tr_prob_stats *S) { return stats_valid(S) ? S->support : NULL; }
int32_t tr_prob_stats_require(const tr_prob_stats *S, const tr_prob_model *M) {
    if (!stats_valid(S) || !model_valid(M)) return TR_MATH_INVALID;
    return support_same(S->support, M->support) ? TR_MATH_OK : TR_MATH_DOMAIN_ERROR;
}
int32_t tr_prob_stats_equal(const tr_prob_stats *a, const tr_prob_stats *b) {
    if (!stats_valid(a) || !stats_valid(b)) return TR_MATH_INVALID;
    if (a->G != b->G || a->d != b->d || a->mass != b->mass || a->exclude != b->exclude ||
        !support_same(a->support, b->support)) return 0;
    size_t G = (size_t)a->G;
    return memcmp(a->counts, b->counts, G * (size_t)a->d * 8) == 0 && memcmp(a->total, b->total, G * 8) == 0 &&
           memcmp(a->rows, b->rows, G * 8) == 0 && memcmp(a->dropped, b->dropped, G * 8) == 0;
}

/* ================================================================ plans */
struct tr_prob_plan_s {
    uint32_t magic;
    int32_t kind, m, mm, what, mode;
    uint64_t n, d, G, nsums, bytes;
    const tr_prob_stats *stats;    /* borrowed: they outlive the plan */
    const tr_prob_model *model;
    tr_prob_plan_view v;
    uint64_t *group;
    int32_t *status;
    tr_calc_plan_view *exp;        /* EXPFAM [n] */
    int64_t *tnum, *tden;          /* EXPFAM [n x m] theta */
    int64_t *weight;               /* CATEGORICAL [n x d] */
    int64_t *wden;                 /* CATEGORICAL [n] Dr */
    int32_t *lscale;               /* CATEGORICAL [n] */
};
static int plan_valid(const tr_prob_plan *p) { return p && p->magic == PLAN_MAGIC; }
void tr_prob_plan_free(tr_prob_plan *p) {
    if (!plan_valid(p)) return;
    p->magic = 0;
    free(p->group); free(p->status); free(p->exp); free(p->tnum); free(p->tden); free(p->weight); free(p->wden);
    free(p->lscale); free(p);
}
static uint64_t plan_mm(const tr_prob_model *M) { return M->kind == TR_PROB_CATEGORICAL && M->m == 0 ? M->d : (uint64_t)M->m; }
/* bytes of a plan, its sums and the finish scratch */
static uint64_t plan_bytes(const tr_prob_stats *S, const tr_prob_model *M, uint64_t n, uint64_t nsums) {
    uint64_t b = tr_budget_add(sizeof(tr_prob_plan), tr_budget_mul(n, 12));
    if (M->kind == TR_PROB_EXPFAM)
        b = tr_budget_add(b, tr_budget_mul(n, tr_budget_add(sizeof(tr_calc_plan_view), 16 * (uint64_t)M->m)));
    else b = tr_budget_add(b, tr_budget_add(tr_budget_mul(tr_budget_mul(n, M->d), 8), tr_budget_mul(n, 12)));
    b = tr_budget_add(b, tr_budget_add(tr_budget_mul(tr_budget_mul(n, nsums), 16), tr_budget_mul(n, 16)));
    uint64_t slots = n < S->G ? n : S->G;
    b = tr_budget_add(b, tr_budget_add(tr_budget_mul(S->G, 4), tr_budget_mul(slots, 32 * (uint64_t)M->m + 96)));
    return b;
}
static int32_t plan_needs(const tr_prob_stats *S, const tr_prob_model *M, uint64_t n, int32_t what, int32_t target,
                          tr_math_need *need) {
    if (!stats_valid(S) || !model_valid(M) || !need || (what & ~(TR_PROB_WANT_GRAD | TR_PROB_WANT_HESS)) ||
        (target != TR_PROB_NEED_CPU && target != TR_PROB_NEED_DEVICE)) return TR_MATH_INVALID;
    const uint64_t mm = plan_mm(M), nsums = tr_prob_nsums(M->kind, M->m, what);
    need_zero(need);
    uint64_t per = 3;
    if (what & (TR_PROB_WANT_GRAD | TR_PROB_WANT_HESS)) per = tr_budget_add(per, mm);
    if (what & TR_PROB_WANT_HESS) per = tr_budget_add(per, mm == TR_BUDGET_UNKNOWN ? mm : tr_budget_mul(mm, mm + 1) / 2);
    need->need[TR_BUDGET_VARIABLES] = (uint64_t)M->m;
    need->need[TR_BUDGET_SUPPORT] = M->d;
    need->need[TR_BUDGET_OUTPUT] = tr_budget_mul(n, per);
    need->need[TR_BUDGET_TENSOR] = tr_budget_mul(n, M->d);
    need->need[TR_BUDGET_STATES] = tr_budget_mul(n, nsums);
    if (target == TR_PROB_NEED_CPU) need->need[TR_BUDGET_BYTES] = plan_bytes(S, M, n, nsums);
    else {
        tr_prob_plan_view v;
        memset(&v, 0, sizeof v);
        v.kind = M->kind; v.m = M->m; v.n = n; v.d = M->d; v.ngroups = S->G; v.nsums = nsums;
        need->need[TR_BUDGET_BYTES] = tr_prob_device_bytes(&v);
    }
    return TR_MATH_OK;
}
int32_t tr_prob_plan_need(const tr_prob_stats *S, const tr_prob_model *M, uint64_t n, int32_t what, int32_t target,
                          tr_math_need *need) {
    int32_t rc = plan_needs(S, M, n, what, target, need);
    if (rc) return rc;
    return tr_prob_stats_require(S, M);
}
/* K = floor(|tn| C / (td D 2^320)) (C = floor(2^512 / ln 2)) into k[4]: 0, or TR_MATH_UNSUPPORTED when K >= 2^256 */
static int32_t make_k(int64_t tn, int64_t td, int64_t D, uint64_t *k, int32_t *ksign) {
    memset(k, 0, 4 * sizeof *k);
    *ksign = tn > 0 ? 1 : tn < 0 ? -1 : 0;
    if (!tn) return TR_MATH_OK;
    uint64_t a = tr_abs_u64(tn), p[10];
    unsigned __int128 carry = 0;
    for (int i = 0; i < 9; i++) {
        unsigned __int128 t = (unsigned __int128)PROB_LOG2E_512[i] * a + carry;
        p[i] = (uint64_t)t; carry = t >> 64;
    }
    p[9] = (uint64_t)carry;
    unsigned __int128 div = (unsigned __int128)(uint64_t)td * (uint64_t)D;
    uint64_t q[10];
    if ((div >> 64) == 0) {                                            /* limb by limb */
        uint64_t dv = (uint64_t)div, r = 0;
        for (int i = 9; i >= 0; i--) {
            unsigned __int128 cur = ((unsigned __int128)r << 64) | p[i];
            q[i] = (uint64_t)(cur / dv); r = (uint64_t)(cur % dv);
        }
    } else {                                                           /* restoring division by a 128-bit divisor */
        unsigned __int128 r = 0;
        memset(q, 0, sizeof q);
        for (int bit = 639; bit >= 0; bit--) {
            int top = (int)(r >> 127);
            r = (r << 1) | ((p[bit / 64] >> (bit % 64)) & 1);
            if (top || r >= div) { r -= div; q[bit / 64] |= (uint64_t)1 << (bit % 64); }
        }
    }
    if (q[9]) return TR_MATH_UNSUPPORTED;                              /* K >= 2^256: impossible for |tn| < 2^63 */
    for (int i = 0; i < 4; i++) k[i] = q[5 + i];                       /* >> 320 */
    if (!(k[0] | k[1] | k[2] | k[3])) *ksign = 0;                     /* below 2^-192 */
    return TR_MATH_OK;
}
static int32_t lscale_of(uint64_t N) { int32_t s = 119 - tr_prob_bitlen64(N); return s > TR_PROB_F ? TR_PROB_F : s; }
tr_prob_plan *tr_prob_plan_create(const tr_prob_stats *S, const tr_prob_model *M, uint64_t n, const uint64_t *groups,
                                  const int64_t *pnum, const int64_t *pden, int32_t mode, int32_t what,
                                  const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    if (!stats_valid(S) || !model_valid(M) || (n && (!groups || !pnum || !pden)) ||
        (mode != TR_PROB_WEIGHTS && mode != TR_PROB_PROBABILITIES) ||
        (M->kind == TR_PROB_EXPFAM && mode != TR_PROB_WEIGHTS)) { set_error(error, TR_MATH_INVALID); return NULL; }
    int32_t rc = tr_prob_stats_require(S, M);
    if (rc) { set_error(error, rc); return NULL; }
    if (plan_mm(M) > INT32_MAX && (what & (TR_PROB_WANT_GRAD | TR_PROB_WANT_HESS))) {
        set_error(error, TR_MATH_UNSUPPORTED); return NULL;
    }
    tr_math_need need;
    rc = plan_needs(S, M, n, what, TR_PROB_NEED_CPU, &need);
    if (rc) { set_error(error, rc); return NULL; }
    tr_math_budget dflt;
    const tr_math_budget *bud = budget_of(budget, &dflt);
    rc = check_need(bud, &need, refusal);
    if (rc) { set_error(error, rc); return NULL; }
    tr_prob_plan *p = (tr_prob_plan *)zalloc(sizeof *p);
    if (!p) { set_error(error, TR_MATH_MEMORY); return NULL; }
    p->magic = PLAN_MAGIC; p->kind = M->kind; p->m = M->m; p->mm = (int32_t)(plan_mm(M) > INT32_MAX ? INT32_MAX : plan_mm(M));
    p->what = what; p->mode = mode; p->n = n; p->d = M->d; p->G = S->G;
    p->nsums = tr_prob_nsums(M->kind, M->m, what); p->bytes = need.need[TR_BUDGET_BYTES];
    p->stats = S; p->model = M;
    const uint64_t d = M->d, m = (uint64_t)M->m;
    p->group = (uint64_t *)zalloc(tr_budget_mul(n, 8));
    p->status = (int32_t *)zalloc(tr_budget_mul(n, 4));
    if (!p->group || !p->status) rc = TR_MATH_MEMORY;
    if (!rc && M->kind == TR_PROB_EXPFAM) {
        p->exp = (tr_calc_plan_view *)zalloc(tr_budget_mul(n, sizeof(tr_calc_plan_view)));
        p->tnum = (int64_t *)zalloc(tr_budget_mul(n, 8 * m));
        p->tden = (int64_t *)zalloc(tr_budget_mul(n, 8 * m));
        if (!p->exp || !p->tnum || !p->tden) rc = TR_MATH_MEMORY;
    } else if (!rc) {
        p->weight = (int64_t *)zalloc(tr_budget_mul(tr_budget_mul(n, d), 8));
        p->wden = (int64_t *)zalloc(tr_budget_mul(n, 8));
        p->lscale = (int32_t *)zalloc(tr_budget_mul(n, 4));
        if (!p->weight || !p->wden || !p->lscale) rc = TR_MATH_MEMORY;
    }
    const int32_t bl_d = tr_prob_bitlen64(d);
    for (uint64_t r = 0; r < n && !rc; r++) {
        uint64_t g = groups[r];
        p->group[r] = g;
        int32_t st = g < S->G ? TR_MATH_OK : TR_MATH_INVALID;
        if (M->kind == TR_PROB_EXPFAM) {
            tr_calc_plan_view *e = &p->exp[r];
            e->family = TR_CALC_EXP; e->nfactors = M->m; e->weight = -1; e->nsubsets = 1; e->ngroups = 1;
            for (uint64_t j = 0; j < m; j++) {
                int64_t tn = pnum[r * m + j], td = pden[r * m + j];
                p->tnum[r * m + j] = tn; p->tden[r * m + j] = td;
                if (st) continue;
                if (td < 1) { st = TR_MATH_INVALID; continue; }
                e->column[j] = (int32_t)j;
                int32_t k = make_k(tn, td, M->fden[j], e->k[j], &e->ksign[j]);
                if (k) st = k;
            }
        } else {
            /* the weights over their lcm: W_i = w_i Dr (int64), Zw > 0; the proofs of the exact sums */
            int64_t Dr = 1;
            const int64_t *wn = pnum + r * d, *wd = pden + r * d;
            for (uint64_t i = 0; i < d && !st; i++) {
                if (wd[i] < 1) { st = TR_MATH_INVALID; break; }
                if (wn[i] < 0) { st = TR_MATH_DOMAIN_ERROR; break; }
                if (!wn[i] || wd[i] == 1 || Dr % wd[i] == 0) continue;
                int64_t den = wd[i];
                uint64_t gg = gcd64((uint64_t)wn[i], (uint64_t)den);
                if (gg > 1) den /= (int64_t)gg;
                if (!lcm64(Dr, den, &Dr)) st = TR_MATH_OVERFLOW;
            }
            unsigned __int128 Zw = 0;
            uint64_t wmax = 0;
            int64_t *W = p->weight + r * d;
            for (uint64_t i = 0; i < d && !st; i++) {
                int64_t num = wn[i], den = wd[i], v = 0;
                if (num) {
                    uint64_t gg = gcd64((uint64_t)num, (uint64_t)den);
                    if (gg > 1) { num /= (int64_t)gg; den /= (int64_t)gg; }
                    if (!tr_mul_i64(num, Dr / den, &v)) { st = TR_MATH_OVERFLOW; break; }
                }
                W[i] = v; Zw += (unsigned __int128)v;
                if ((uint64_t)v > wmax) wmax = (uint64_t)v;
            }
            if (!st && Zw == 0) st = TR_MATH_DOMAIN_ERROR;              /* Z = 0 */
            if (!st && mode == TR_PROB_PROBABILITIES && Zw != (unsigned __int128)Dr) st = TR_MATH_DOMAIN_ERROR;
            if (!st && m && (what & (TR_PROB_WANT_GRAD | TR_PROB_WANT_HESS))) {
                int32_t bw = tr_prob_bitlen64(wmax);
                for (uint64_t j = 0; j < m; j++) if (bl_d + bw + M->bits[j] > 126) st = TR_MATH_OVERFLOW;
                if (what & TR_PROB_WANT_HESS)
                    for (uint64_t q = 0; q < tr_prob_tri(m); q++) if (bl_d + bw + M->pbits[q] > 126) st = TR_MATH_OVERFLOW;
            }
            p->wden[r] = Dr;
            p->lscale[r] = st ? 0 : lscale_of(S->total[g]);
            if (st) memset(W, 0, (size_t)d * 8);
        }
        p->status[r] = st;
    }
    if (rc) { tr_prob_plan_free(p); set_error(error, rc); return NULL; }
    tr_prob_plan_view *v = &p->v;
    memset(v, 0, sizeof *v);
    v->kind = p->kind; v->what = what; v->m = M->m; v->mm = p->mm; v->n = n; v->d = d; v->ngroups = S->G;
    v->nsums = p->nsums; v->scale = M->scale;
    for (int32_t j = 0; j < 16; j++) v->bits[j] = M->bits[j];
    v->feat = M->feat; v->base = M->base; v->hbits = M->hbits; v->counts = S->counts;
    v->group = p->group; v->status = p->status; v->exp = p->exp; v->weight = p->weight; v->lscale = p->lscale;
    set_error(error, TR_MATH_OK);
    return p;
}
int32_t tr_prob_plan_get_view(const tr_prob_plan *p, tr_prob_plan_view *out) {
    if (!plan_valid(p) || !out) return TR_MATH_INVALID;
    *out = p->v;
    return TR_MATH_OK;
}
int32_t tr_prob_plan_status(const tr_prob_plan *p, uint64_t r) {
    if (!plan_valid(p) || r >= p->n) return TR_MATH_INVALID;
    return p->status[r];
}
uint64_t tr_prob_device_need(const tr_prob_plan_view *v) { return v ? tr_prob_device_bytes(v) : TR_BUDGET_UNKNOWN; }

/* ================================================================ the sums (CPU reference of the kernels) */
/* the shared per-point terms of math_prob.h (tr_prob_exp_ref, tr_prob_exp_point, tr_prob_cat_point) in the natural
 * order; every sum is exact and proven below 2^126, so the totals do not depend on the order of the additions */
int32_t tr_prob_sums(const tr_prob_plan_view *v, tr_math_i128 *sums, int64_t *nref, uint64_t *flags) {
    if (!tr_prob_plan_view_valid(v) || (v->n && (!sums || !nref || !flags))) return TR_MATH_INVALID;
    for (uint64_t r = 0; r < v->n; r++) if (v->group[r] >= v->ngroups && !v->status[r]) return TR_MATH_INVALID;
    const uint64_t nq = tr_prob_nq(v), ns = v->nsums;
    __int128 acc[2 * TR_PROB_MAX_SUMS];
    for (uint64_t r = 0; r < v->n; r++) {
        nref[r] = INT64_MIN; flags[r] = 0;
        for (uint64_t q = 0; q < ns; q++) sums[r * ns + q] = tr_i128_set(0);
        if (v->status[r]) continue;
        memset(acc, 0, sizeof(__int128) * (size_t)ns);
        if (v->kind == TR_PROB_EXPFAM) {
            int64_t best = INT64_MIN;                                   /* the points that overflow are skipped */
            for (uint64_t i = 0; i < v->d; i++) {
                if (!v->base[i]) continue;
                int64_t e;
                if (!tr_prob_exp_ref(v, r, i, &e)) { flags[r] |= TR_PROB_EXP_OVERFLOW; continue; }
                if (e > best) best = e;
            }
            nref[r] = best;
            if (flags[r]) continue;
            for (uint64_t i = 0; i < v->d; i++) {
                if (!v->base[i]) continue;
                int ok = tr_prob_exp_point(v, r, i, best, acc);
                if (ok < 0) flags[r] |= TR_PROB_EXP_OVERFLOW;
                else if (!ok) flags[r] |= TR_PROB_INCONSISTENT;
            }
        } else {
            for (uint64_t i = 0; i < v->d; i++)
                if (!tr_prob_cat_point(v, r, i, acc, acc + nq)) flags[r] |= TR_PROB_NEG_INF;
        }
        for (uint64_t q = 0; q < ns; q++) sums[r * ns + q] = tr_i128_set(acc[q]);
    }
    return TR_MATH_OK;
}

/* ================================================================ finishing the answers */
typedef struct {
    uint64_t N, rows;
    int neg;                    /* EXPFAM: a point with H_i = 0 has mass */
    tr_dist_i256 alo, ahi;      /* EXPFAM: sum n ln H bounds at 2^-120 */
    tr_dist_i256 *tobs;         /* [m] TobsInt_j */
} group_vals;
static void group_compute(const tr_prob_plan *p, uint64_t g, group_vals *gv) {
    const tr_prob_stats *S = p->stats;
    const tr_prob_model *M = p->model;
    const uint64_t d = p->d, m = (uint64_t)p->m, *cnt = S->counts + g * d;
    gv->N = S->total[g]; gv->rows = S->rows[g]; gv->neg = 0;
    gv->alo = gv->ahi = tr_dist_w_zero();
    for (uint64_t j = 0; j < m; j++) gv->tobs[j] = tr_dist_w_zero();
    for (uint64_t i = 0; i < d; i++) {
        uint64_t c = cnt[i];
        if (!c) continue;
        for (uint64_t j = 0; j < m; j++)
            w_add(gv->tobs[j], tr_dist_w_mul128((__int128)c, (__int128)M->feat[i * m + j]), &gv->tobs[j]);
        if (p->kind == TR_PROB_EXPFAM) {
            if (!M->base[i]) { gv->neg = 1; continue; }
            w_add(gv->alo, tr_dist_w_mul128((__int128)c, i128_of(M->lnh[2 * i])), &gv->alo);
            w_add(gv->ahi, tr_dist_w_mul128((__int128)c, i128_of(M->lnh[2 * i + 1])), &gv->ahi);
        }
    }
}
/* [lo, hi] (2^-120) of e ln 2 */
static void ln2_times(int64_t e, tr_dist_i256 *lo, tr_dist_i256 *hi) {
    tr_dist_i256 a = tr_dist_w_mul128((__int128)e, (__int128)TR_INFO_LN2), b = tr_dist_w_zero();
    w_add(a, w_from((__int128)e), &b);                                 /* e (LN2 + 1) */
    if (e >= 0) { *lo = a; *hi = b; } else { *lo = b; *hi = a; }
}
static double ru_width(tr_math_interval iv) { return iv.hi != iv.hi ? NAN : tr_add_ru(iv.hi, -iv.lo); }
/* the double enclosure of num / den (den > 0, both exact 256-bit) */
static tr_math_interval quotient(tr_dist_i256 num, tr_dist_i256 den) {
    return tr_interval_div(enclose_w(num, 0), enclose_w(den, 0));
}
/* box helpers: [a, b] of int128 sums */
typedef struct { __int128 lo, hi; } box;
static box box_of(const tr_math_i128 *s, uint64_t q) { box b; b.lo = i128_of(s[2 * q]); b.hi = i128_of(s[2 * q + 1]); return b; }
/* min and max of x y over two boxes (the corners) */
static void prod_range(box x, box y, tr_dist_i256 *mn, tr_dist_i256 *mx) {
    tr_dist_i256 c[4] = {tr_dist_w_mul128(x.lo, y.lo), tr_dist_w_mul128(x.lo, y.hi), tr_dist_w_mul128(x.hi, y.lo),
                         tr_dist_w_mul128(x.hi, y.hi)};
    *mn = *mx = c[0];
    for (int i = 1; i < 4; i++) {
        if (tr_dist_w_cmp(c[i], *mn) < 0) *mn = c[i];
        if (tr_dist_w_cmp(c[i], *mx) > 0) *mx = c[i];
    }
}
static void square_range(box x, tr_dist_i256 *mn, tr_dist_i256 *mx) {
    tr_dist_i256 a = tr_dist_w_mul128(x.lo, x.lo), b = tr_dist_w_mul128(x.hi, x.hi);
    *mx = tr_dist_w_cmp(a, b) > 0 ? a : b;
    if (x.lo <= 0 && x.hi >= 0) *mn = tr_dist_w_zero();
    else *mn = tr_dist_w_cmp(a, b) < 0 ? a : b;
}
/* the corner bounds of an EXPFAM gradient entry: numerators and denominators (see the header comment) */
typedef struct { tr_dist_i256 num[2], den[2]; } bounds2;
static int exp_grad(const tr_prob_plan *p, const tr_math_i128 *s, const group_vals *gv, int32_t j, bounds2 *out) {
    box S0 = box_of(s, 0), Sj = box_of(s, tr_prob_q_feature(j));
    const int32_t b = p->model->bits[j];
    const int64_t D = p->model->fden[j];
    /* lower bound: the corner of max Sj / S0; upper: of min Sj / S0 */
    __int128 sj[2] = {Sj.hi, Sj.lo}, s0[2] = {Sj.hi >= 0 ? S0.lo : S0.hi, Sj.lo >= 0 ? S0.hi : S0.lo};
    for (int t = 0; t < 2; t++) {
        tr_dist_i256 a, c, x;
        if (!w_mul_u64(tr_prob_w_shl(w_from(sj[t]), b), gv->N, &c)) return 0;   /* N 2^b Sj */
        /* TobsInt S0: |TobsInt| < 2^(64 + b) <= 2^127, 0 < S0 < 2^126 */
        tr_dist_i256 T = gv->tobs[j];
        int nt = tr_dist_w_negative(T);
        tr_dist_i256 mt = nt ? tr_dist_w_negate_bits(T) : T;
        if (mt.w[2] || mt.w[3]) return 0;
        tr_dist_i256 pr = tr_dist_w_mul_u128(((unsigned __int128)mt.w[1] << 64) | mt.w[0], (unsigned __int128)s0[t]);
        a = nt ? tr_dist_w_negate_bits(pr) : pr;
        if (!w_sub(a, c, &x)) return 0;
        out->num[t] = x;
        out->den[t] = tr_dist_w_mul128(s0[t], (__int128)D);
    }
    return 1;
}
/* the bounds of an EXPFAM Hessian entry as -c P / S0^2 (c = N 2^(b_j + b_l) / (D_j D_l)): P's range and S0's */
typedef struct { tr_dist_i256 plo, phi; } prange;
static void exp_hess_p(const tr_math_i128 *s, int32_t m, int32_t j, int32_t l, prange *out) {
    box S0 = box_of(s, 0), Sj = box_of(s, tr_prob_q_feature(j)), Sl = box_of(s, tr_prob_q_feature(l));
    box Sjl = box_of(s, 1 + (uint64_t)m + tr_prob_pack((uint64_t)m, (uint64_t)j, (uint64_t)l));
    tr_dist_i256 amn, amx, bmn, bmx;
    amn = tr_dist_w_mul128(Sjl.lo, Sjl.lo >= 0 ? S0.lo : S0.hi);       /* min, max of Sjl S0 (S0 > 0) */
    amx = tr_dist_w_mul128(Sjl.hi, Sjl.hi >= 0 ? S0.hi : S0.lo);
    if (j == l) square_range(Sj, &bmn, &bmx); else prod_range(Sj, Sl, &bmn, &bmx);
    w_sub(amn, bmx, &out->plo);
    w_sub(amx, bmn, &out->phi);
}
static int32_t answer_fail(tr_prob_answer *a, int32_t status, tr_math_interval *grad, tr_math_interval *hess,
                           uint64_t mm, uint64_t nh) {
    a->status = status;
    a->z = a->logz = a->loglik = iv_nan();
    for (int i = 0; i < 5; i++) a->width[i] = NAN;
    if (grad) for (uint64_t j = 0; j < mm; j++) grad[j] = iv_nan();
    if (hess) for (uint64_t j = 0; j < nh; j++) hess[j] = iv_nan();
    return status;
}
/* consistency of a request's sums with their proofs (device sums are checked like CPU sums) */
static int sums_ok(const tr_prob_plan *p, const tr_math_i128 *s) {
    const __int128 lim = (__int128)1 << 126;
    if (p->kind == TR_PROB_EXPFAM) {
        for (uint64_t q = 0; q < p->nsums / 2; q++) {
            __int128 lo = i128_of(s[2 * q]), hi = i128_of(s[2 * q + 1]);
            if (lo > hi || lo <= -lim || hi >= lim) return 0;
        }
        return i128_of(s[0]) >= 1;
    }
    for (uint64_t q = 0; q < p->nsums; q++) { __int128 x = i128_of(s[q]); if (x <= -lim || x >= lim) return 0; }
    return i128_of(s[0]) >= 1 && i128_of(s[p->nsums - 2]) <= i128_of(s[p->nsums - 1]);
}
/* group values of a request (a slot per distinct group, computed once) */
typedef struct {
    uint32_t *slot;             /* [G] slot + 1, 0 = not computed */
    group_vals *vals;
    tr_dist_i256 *tobs;
    uint64_t used, cap;
} gcache;
static const group_vals *gcache_get(const tr_prob_plan *p, gcache *c, uint64_t g) {
    if (c->slot[g]) return &c->vals[c->slot[g] - 1];
    if (c->used >= c->cap) return NULL;
    group_vals *gv = &c->vals[c->used];
    gv->tobs = c->tobs + c->used * (uint64_t)p->m;
    group_compute(p, g, gv);
    c->slot[g] = (uint32_t)(++c->used);
    return gv;
}
static int32_t finish_one(const tr_prob_plan *p, uint64_t r, const tr_math_i128 *s, int64_t nref, uint64_t fl,
                          const group_vals *gv, tr_prob_answer *a, tr_math_interval *grad, tr_math_interval *hess) {
    const tr_prob_model *M = p->model;
    const uint64_t mm = (uint64_t)p->mm, nh = tr_prob_tri(mm), m = (uint64_t)p->m;
    const int wg = (p->what & (TR_PROB_WANT_GRAD | TR_PROB_WANT_HESS)) != 0, wh = (p->what & TR_PROB_WANT_HESS) != 0;
    a->mass = gv->N; a->rows = gv->rows;
    if (!gv->N) a->flags |= TR_PROB_EMPTY;
    double gw = 0, hw = 0;
    if (p->kind == TR_PROB_EXPFAM) {
        if (fl & TR_PROB_EXP_OVERFLOW) return answer_fail(a, TR_MATH_OVERFLOW, wg ? grad : NULL, wh ? hess : NULL, mm, nh);
        if ((fl & TR_PROB_INCONSISTENT) || nref == INT64_MIN || !sums_ok(p, s))
            return answer_fail(a, TR_MATH_INVALID, wg ? grad : NULL, wh ? hess : NULL, mm, nh);
        const int32_t sc = M->scale;
        box S0 = box_of(s, 0);
        const int64_t zexp = nref - sc;
        a->zexp = zexp; a->zden = M->hden; a->zlo = tr_i128_set(S0.lo); a->zhi = tr_i128_set(S0.hi);
        tr_math_interval dh = enclose_u64((uint64_t)M->hden);
        tr_math_interval z0 = enclose_i128(S0.lo, 0), z1 = enclose_i128(S0.hi, 0);
        tr_math_interval zz = iv_make(z0.lo, z1.hi);
        zz = tr_interval_div(zz, dh);
        a->z = iv_make(ldexp_dir(zz.lo, zexp, 0), ldexp_dir(zz.hi, zexp, 1));
        a->width[0] = tr_div_dir(enclose_i128(S0.hi - S0.lo, 0).hi, z0.lo, 1);
        /* ln of the sum, of Z */
        __int128 l0, l1, t0, t1;
        tr_prob_ln128((unsigned __int128)S0.lo, &l0, &t0);
        tr_prob_ln128((unsigned __int128)S0.hi, &t1, &l1);
        tr_dist_i256 e0, e1, ls0, ls1;
        ln2_times(zexp, &e0, &e1);
        w_add(e0, w_from(l0), &ls0); w_add(e1, w_from(l1), &ls1);
        unsigned __int128 dlo, dhi;
        tr_info_ln((uint64_t)M->hden, &dlo, &dhi);
        w_sub(ls0, w_from((__int128)dhi), &a->logz_lo);
        w_sub(ls1, w_from((__int128)dlo), &a->logz_hi);
        a->logz = iv_make(enclose_w(a->logz_lo, -TR_PROB_F).lo, enclose_w(a->logz_hi, -TR_PROB_F).hi);
        /* the log-likelihood */
        if (gv->neg) {
            a->flags |= TR_PROB_NEG_INF;
            a->loglik = iv_make(-INFINITY, -INFINITY);
            a->width[2] = 0;
        } else {
            tr_dist_i256 n1, n0;
            if (!w_mul_u64(ls1, gv->N, &n1) || !w_mul_u64(ls0, gv->N, &n0) || !w_sub(gv->alo, n1, &a->ll_lo) ||
                !w_sub(gv->ahi, n0, &a->ll_hi)) return answer_fail(a, TR_MATH_OVERFLOW, wg ? grad : NULL, wh ? hess : NULL, mm, nh);
            tr_math_interval ll = iv_make(enclose_w(a->ll_lo, -TR_PROB_F).lo, enclose_w(a->ll_hi, -TR_PROB_F).hi);
            tr_math_interval th = iv_make(0.0, 0.0);
            for (uint64_t j = 0; j < m; j++) {
                tr_math_interval t = tr_interval_div(enclose_w(tr_dist_w_from_i128((__int128)p->tnum[r * m + j]), 0),
                                                     enclose_u64((uint64_t)p->tden[r * m + j]));
                t = tr_interval_mul(t, tr_interval_div(enclose_w(gv->tobs[j], 0), enclose_u64((uint64_t)M->fden[j])));
                th = tr_interval_add(th, t);
            }
            a->loglik = tr_interval_add(ll, th);
            a->width[2] = ru_width(a->loglik);
        }
        a->width[1] = ru_width(a->logz);
        if (wg) {
            for (uint64_t j = 0; j < m; j++) {
                if (gv->neg) { grad[j] = iv_nan(); continue; }
                bounds2 b;
                if (!exp_grad(p, s, gv, (int32_t)j, &b)) return answer_fail(a, TR_MATH_OVERFLOW, grad, wh ? hess : NULL, mm, nh);
                grad[j] = iv_make(quotient(b.num[0], b.den[0]).lo, quotient(b.num[1], b.den[1]).hi);
                double w = ru_width(grad[j]);
                if (w > gw) gw = w;
            }
            a->width[3] = gv->neg ? NAN : gw;
        }
        if (wh) {
            for (uint64_t j = 0; j < m; j++)
                for (uint64_t l = j; l < m; l++) {
                    uint64_t at = tr_prob_pack(m, j, l);
                    if (gv->neg) { hess[at] = iv_nan(); continue; }
                    prange pr;
                    exp_hess_p(s, (int32_t)m, (int32_t)j, (int32_t)l, &pr);
                    /* P / S0^2 range, then times -c */
                    tr_dist_i256 q0 = tr_dist_w_mul128(S0.lo, S0.lo), q1 = tr_dist_w_mul128(S0.hi, S0.hi);
                    tr_math_interval hiq = quotient(pr.phi, tr_dist_w_negative(pr.phi) ? q1 : q0);
                    tr_math_interval loq = quotient(pr.plo, tr_dist_w_negative(pr.plo) ? q0 : q1);
                    tr_math_interval c = tr_interval_div(enclose_u64(gv->N), tr_interval_mul(enclose_u64((uint64_t)M->fden[j]),
                                                         enclose_u64((uint64_t)M->fden[l])));
                    c = iv_make(ldexp_dir(c.lo, M->bits[j] + M->bits[l], 0), ldexp_dir(c.hi, M->bits[j] + M->bits[l], 1));
                    tr_math_interval ratio = iv_make(loq.lo, hiq.hi);
                    tr_math_interval h = tr_interval_neg(tr_interval_mul(c, ratio));
                    if (j == l && h.hi > 0) h.hi = 0.0;                    /* -N Var <= 0 */
                    if (j == l && h.lo > 0) h.lo = 0.0;
                    hess[at] = iv_make(h.lo, h.hi);
                    double w = ru_width(hess[at]);
                    if (w > hw) hw = w;
                }
            a->width[4] = gv->neg ? NAN : hw;
        }
        return TR_MATH_OK;
    }
    /* ---- CATEGORICAL */
    if (fl & TR_PROB_INCONSISTENT) return answer_fail(a, TR_MATH_INVALID, wg ? grad : NULL, wh ? hess : NULL, mm, nh);
    if (!sums_ok(p, s)) return answer_fail(a, TR_MATH_INVALID, wg ? grad : NULL, wh ? hess : NULL, mm, nh);
    const uint64_t nq = p->nsums - 2;
    const __int128 Zw = i128_of(s[0]);
    const int64_t Dr = p->wden[r];
    const int neg = (fl & TR_PROB_NEG_INF) != 0;
    a->zexp = 0; a->zden = Dr; a->zlo = a->zhi = tr_i128_set(Zw);
    a->z = tr_interval_div(enclose_i128(Zw, 0), enclose_u64((uint64_t)Dr));
    a->width[0] = 0;
    __int128 lz0, lz1, tmp;
    tr_prob_ln128((unsigned __int128)Zw, &lz0, &lz1);
    unsigned __int128 dlo, dhi;
    tr_info_ln((uint64_t)Dr, &dlo, &dhi);
    w_sub(w_from(lz0), w_from((__int128)dhi), &a->logz_lo);
    w_sub(w_from(lz1), w_from((__int128)dlo), &a->logz_hi);
    a->logz = iv_make(enclose_w(a->logz_lo, -TR_PROB_F).lo, enclose_w(a->logz_hi, -TR_PROB_F).hi);
    a->width[1] = ru_width(a->logz);
    (void)tmp;
    if (neg) {
        a->flags |= TR_PROB_NEG_INF;
        a->loglik = iv_make(-INFINITY, -INFINITY);
        a->width[2] = 0;
    } else {
        int32_t sL = p->lscale[r];
        tr_dist_i256 L0 = tr_prob_w_shl(w_from(i128_of(s[nq])), TR_PROB_F - sL);
        tr_dist_i256 L1 = tr_prob_w_shl(w_from(i128_of(s[nq + 1])), TR_PROB_F - sL);
        tr_dist_i256 n0, n1;
        if (!w_mul_u64(w_from(lz1), gv->N, &n1) || !w_mul_u64(w_from(lz0), gv->N, &n0) || !w_sub(L0, n1, &a->ll_lo) ||
            !w_sub(L1, n0, &a->ll_hi)) return answer_fail(a, TR_MATH_OVERFLOW, wg ? grad : NULL, wh ? hess : NULL, mm, nh);
        a->loglik = iv_make(enclose_w(a->ll_lo, -TR_PROB_F).lo, enclose_w(a->ll_hi, -TR_PROB_F).hi);
        a->width[2] = ru_width(a->loglik);
    }
    const int64_t *W = p->weight + r * p->d;
    const uint64_t *cnt = p->stats->counts + p->group[r] * p->d;
    if (wg) {
        for (uint64_t j = 0; j < mm; j++) {
            if (neg) { grad[j] = iv_nan(); continue; }
            tr_dist_i256 num, den, x, y;
            if (m) {   /* (TobsInt_j Zw - N S_j) / (D_j Zw) */
                tr_dist_i256 T = gv->tobs[j];
                int nt = tr_dist_w_negative(T);
                tr_dist_i256 mt = nt ? tr_dist_w_negate_bits(T) : T;
                unsigned __int128 lo = ((unsigned __int128)mt.w[1] << 64) | mt.w[0];
                x = tr_dist_w_mul_u128(lo, (unsigned __int128)Zw);
                if (nt) x = tr_dist_w_negate_bits(x);
                if (!w_mul_u64(w_from(i128_of(s[tr_prob_q_feature((int32_t)j)])), gv->N, &y) || !w_sub(x, y, &num))
                    return answer_fail(a, TR_MATH_OVERFLOW, grad, wh ? hess : NULL, mm, nh);
                den = tr_dist_w_mul128(Zw, (__int128)M->fden[j]);
            } else {   /* (n_j Zw - N W_j) / Zw */
                x = tr_dist_w_mul128((__int128)cnt[j], Zw);
                y = tr_dist_w_mul128((__int128)gv->N, (__int128)W[j]);
                w_sub(x, y, &num);
                den = w_from(Zw);
            }
            grad[j] = quotient(num, den);
            double w = ru_width(grad[j]);
            if (w > gw) gw = w;
        }
        a->width[3] = neg ? NAN : gw;
    }
    if (wh) {
        tr_dist_i256 Z2 = tr_dist_w_mul128(Zw, Zw);
        tr_math_interval cN = enclose_u64(gv->N);
        for (uint64_t j = 0; j < mm; j++)
            for (uint64_t l = j; l < mm; l++) {
                uint64_t at = tr_prob_pack(mm, j, l);
                if (neg) { hess[at] = iv_nan(); continue; }
                tr_dist_i256 P, x, y;
                tr_math_interval den;
                if (m) {   /* -N (S_jl Zw - S_j S_l) / (D_j D_l Zw^2) */
                    __int128 Sjl = i128_of(s[1 + m + tr_prob_pack(m, j, l)]);
                    x = tr_dist_w_mul128(Sjl, Zw);
                    y = tr_dist_w_mul128(i128_of(s[tr_prob_q_feature((int32_t)j)]), i128_of(s[tr_prob_q_feature((int32_t)l)]));
                    if (!w_sub(x, y, &P)) return answer_fail(a, TR_MATH_OVERFLOW, grad, hess, mm, nh);
                    den = tr_interval_mul(enclose_w(Z2, 0), enclose_i128((__int128)M->fden[j] * M->fden[l], 0));
                } else {   /* -N (delta W_j Zw - W_j W_l) / Zw^2 */
                    x = j == l ? tr_dist_w_mul128((__int128)W[j], Zw) : tr_dist_w_zero();
                    y = tr_dist_w_mul128((__int128)W[j], (__int128)W[l]);
                    w_sub(x, y, &P);
                    den = enclose_w(Z2, 0);
                }
                tr_math_interval h = tr_interval_neg(tr_interval_div(tr_interval_mul(cN, enclose_w(P, 0)), den));
                hess[at] = iv_make(h.lo, h.hi);
                double w = ru_width(hess[at]);
                if (w > hw) hw = w;
            }
        a->width[4] = neg ? NAN : hw;
    }
    return TR_MATH_OK;
}
static int32_t gcache_open(const tr_prob_plan *p, const tr_math_budget *bud, tr_math_refusal *refusal, gcache *c) {
    memset(c, 0, sizeof *c);
    uint64_t slots = p->n < p->G ? p->n : p->G, m = (uint64_t)p->m;
    if (slots < 1) slots = 1;
    uint64_t bytes = tr_budget_add(tr_budget_mul(p->G, 4), tr_budget_mul(slots, sizeof(group_vals) + 32 * m));
    tr_math_need need;
    need_zero(&need);
    need.need[TR_BUDGET_BYTES] = bytes;
    int32_t rc = check_need(bud, &need, refusal);
    if (rc) return rc;
    c->slot = (uint32_t *)zalloc(tr_budget_mul(p->G, 4));
    c->vals = (group_vals *)zalloc(tr_budget_mul(slots, sizeof(group_vals)));
    c->tobs = (tr_dist_i256 *)zalloc(tr_budget_mul(slots, 32 * (m ? m : 1)));
    c->cap = slots;
    if (!c->slot || !c->vals || !c->tobs) { free(c->slot); free(c->vals); free(c->tobs); return TR_MATH_MEMORY; }
    return TR_MATH_OK;
}
static void gcache_close(gcache *c) { free(c->slot); free(c->vals); free(c->tobs); memset(c, 0, sizeof *c); }
int32_t tr_prob_finish(const tr_prob_plan *p, const tr_math_i128 *sums, const int64_t *nref, const uint64_t *flags,
                       tr_prob_answer *out, tr_math_interval *grad, tr_math_interval *hess,
                       const tr_math_budget *budget, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!plan_valid(p) || (p->n && (!sums || !nref || !flags || !out))) return TR_MATH_INVALID;
    const int wg = (p->what & (TR_PROB_WANT_GRAD | TR_PROB_WANT_HESS)) != 0, wh = (p->what & TR_PROB_WANT_HESS) != 0;
    if (p->n && ((wg && !grad) || (wh && !hess))) return TR_MATH_INVALID;
    tr_math_budget dflt;
    const tr_math_budget *bud = budget_of(budget, &dflt);
    gcache c;
    int32_t rc = gcache_open(p, bud, refusal, &c);
    if (rc) return rc;
    const uint64_t mm = (uint64_t)p->mm, nh = tr_prob_tri(mm);
    for (uint64_t r = 0; r < p->n; r++) {
        tr_prob_answer *a = &out[r];
        memset(a, 0, sizeof *a);
        a->group = p->group[r];
        tr_math_interval *gr = wg ? grad + r * mm : NULL, *hs = wh ? hess + r * nh : NULL;
        if (p->status[r]) { answer_fail(a, p->status[r], gr, hs, mm, nh); continue; }
        const group_vals *gv = gcache_get(p, &c, p->group[r]);
        if (!gv) { gcache_close(&c); return TR_MATH_INVALID; }
        finish_one(p, r, sums + r * p->nsums, nref[r], flags[r], gv, a, gr, hs);
    }
    gcache_close(&c);
    return TR_MATH_OK;
}

/* ================================================================ exact bounds (FND-009 big integers) */
typedef struct {
    const tr_math_budget *b;
    tr_math_refusal *r;
    int32_t err;
    tr_bigint *tmp[96];
    int nt;
} bctx;
static tr_bigint *keep(bctx *c, tr_bigint *x, int32_t e) {
    if (!x) { if (!c->err) c->err = e ? e : TR_MATH_MEMORY; return NULL; }
    if (c->nt < 96) c->tmp[c->nt++] = x;
    else { tr_bigint_free(x); if (!c->err) c->err = TR_MATH_MEMORY; return NULL; }
    return x;
}
static tr_bigint *B_w(bctx *c, tr_dist_i256 v) {
    if (c->err) return NULL;
    int32_t e = 0;
    return keep(c, tr_bigint_from_twos(v.w, 4, c->b, c->r, &e), e);
}
static tr_bigint *B_i(bctx *c, __int128 v) { return B_w(c, w_from(v)); }
static tr_bigint *B_mul(bctx *c, const tr_bigint *x, const tr_bigint *y) {
    if (c->err || !x || !y) return NULL;
    int32_t e = 0;
    return keep(c, tr_bigint_mul(x, y, c->b, c->r, &e), e);
}
static tr_bigint *B_add(bctx *c, const tr_bigint *x, const tr_bigint *y) {
    if (c->err || !x || !y) return NULL;
    int32_t e = 0;
    return keep(c, tr_bigint_add(x, y, c->b, c->r, &e), e);
}
static tr_bigint *B_neg(bctx *c, const tr_bigint *x) {
    if (c->err || !x) return NULL;
    int32_t e = 0;
    return keep(c, tr_bigint_neg(x, c->b, c->r, &e), e);
}
static tr_bigint *B_shl(bctx *c, const tr_bigint *x, uint64_t k) {
    if (c->err || !x) return NULL;
    int32_t e = 0;
    return keep(c, tr_bigint_shl(x, k, c->b, c->r, &e), e);
}
static tr_bigint *B_copy(bctx *c, const tr_bigint *x) {   /* an owned copy for the caller (not kept) */
    if (c->err || !x) return NULL;
    int32_t e = 0;
    tr_bigint *y = tr_bigint_copy(x, c->b, c->r, &e);
    if (!y && !c->err) c->err = e ? e : TR_MATH_MEMORY;
    return y;
}
static void bctx_close(bctx *c) { for (int i = 0; i < c->nt; i++) tr_bigint_free(c->tmp[i]); c->nt = 0; }
int32_t tr_prob_exact(const tr_prob_plan *p, uint64_t r, const tr_math_i128 *sums, const int64_t *nref,
                      const uint64_t *flags, uint64_t entry, const tr_math_budget *budget, tr_math_refusal *refusal,
                      tr_bigint **lo_num, tr_bigint **lo_den, tr_bigint **hi_num, tr_bigint **hi_den, int32_t *exact,
                      int32_t *eflags) {
    tr_budget_none(refusal);
    if (!plan_valid(p) || r >= p->n || !lo_num || !lo_den || !hi_num || !hi_den || !exact || !eflags)
        return TR_MATH_INVALID;
    *lo_num = *lo_den = *hi_num = *hi_den = NULL; *exact = 0; *eflags = 0;
    const uint64_t mm = (uint64_t)p->mm, nh = tr_prob_tri(mm), m = (uint64_t)p->m;
    const int wg = (p->what & (TR_PROB_WANT_GRAD | TR_PROB_WANT_HESS)) != 0, wh = (p->what & TR_PROB_WANT_HESS) != 0;
    if (entry >= 3 + (wg ? mm : 0) + (wh ? nh : 0)) return TR_MATH_INVALID;
    if (p->status[r]) return p->status[r];
    tr_math_budget dflt;
    const tr_math_budget *bud = budget_of(budget, &dflt);
    /* the sums: given, or recomputed for this request alone */
    tr_math_i128 own[TR_PROB_MAX_SUMS * 2];
    int64_t nr;
    uint64_t fl;
    const tr_math_i128 *s;
    if (sums) {
        if (!nref || !flags) return TR_MATH_INVALID;
        s = sums + r * p->nsums; nr = nref[r]; fl = flags[r];
    } else {
        tr_prob_plan_view v = p->v;
        v.n = 1; v.group = p->group + r; v.status = p->status + r;
        if (v.exp) v.exp = p->exp + r;
        if (v.weight) v.weight = p->weight + r * p->d;
        if (v.lscale) v.lscale = p->lscale + r;
        int32_t rc = tr_prob_sums(&v, own, &nr, &fl);
        if (rc) return rc;
        s = own;
    }
    /* the answer (its exact 2^-120 bounds and flags) and the group values */
    gcache c;
    int32_t rc = gcache_open(p, bud, refusal, &c);
    if (rc) return rc;
    const group_vals *gv = gcache_get(p, &c, p->group[r]);
    tr_prob_answer a;
    memset(&a, 0, sizeof a);
    tr_math_interval gtmp[TR_PROB_MAX_FEATURES], htmp[136];
    int one_hot = p->kind == TR_PROB_CATEGORICAL && m == 0;
    rc = gv ? TR_MATH_OK : TR_MATH_INVALID;
    if (!rc) {
        /* finish_one with small scratch (one-hot: the gradient / Hessian entries are computed below directly) */
        tr_prob_plan q = *p;
        if (one_hot) q.what = 0;
        rc = finish_one(&q, r, s, nr, fl, gv, &a, gtmp, htmp);
    }
    *eflags = a.flags;
    bctx B;
    memset(&B, 0, sizeof B);
    B.b = bud; B.r = refusal;
    tr_bigint *n0 = NULL, *d0 = NULL, *n1 = NULL, *d1 = NULL;
    const tr_prob_model *M = p->model;
    if (!rc && entry == TR_PROB_ENTRY_Z) {
        if (p->kind == TR_PROB_EXPFAM) {
            tr_bigint *D = B_i(&B, (__int128)M->hden);
            n0 = B_i(&B, i128_of(a.zlo)); n1 = B_i(&B, i128_of(a.zhi)); d0 = d1 = D;
            if (a.zexp >= 0) { n0 = B_shl(&B, n0, (uint64_t)a.zexp); n1 = B_shl(&B, n1, (uint64_t)a.zexp); }
            else { d0 = d1 = B_shl(&B, D, (uint64_t)(-a.zexp)); }
        } else {
            n0 = n1 = B_i(&B, i128_of(a.zlo)); d0 = d1 = B_i(&B, (__int128)a.zden);
            *exact = 1;
        }
    } else if (!rc && entry == TR_PROB_ENTRY_LOGZ) {
        tr_bigint *D = B_shl(&B, B_i(&B, 1), TR_PROB_F);
        n0 = B_w(&B, a.logz_lo); n1 = B_w(&B, a.logz_hi); d0 = d1 = D;
    } else if (!rc && entry == TR_PROB_ENTRY_LOGLIK) {
        if (a.flags & TR_PROB_NEG_INF) { bctx_close(&B); gcache_close(&c); return TR_MATH_OK; }
        tr_bigint *D = B_shl(&B, B_i(&B, 1), TR_PROB_F);
        n0 = B_w(&B, a.ll_lo); n1 = B_w(&B, a.ll_hi); d0 = d1 = D;
        if (p->kind == TR_PROB_EXPFAM) {
            /* + theta . Tobs = sum_j tn_j TobsInt_j / (td_j D_j), over the product of the denominators */
            tr_bigint *num = B_i(&B, 0), *den = B_i(&B, 1);
            for (uint64_t j = 0; j < m && !B.err; j++) {
                tr_bigint *dj = B_i(&B, (__int128)p->tden[r * m + j] * M->fden[j]);
                tr_bigint *tj = B_mul(&B, B_i(&B, (__int128)p->tnum[r * m + j]), B_w(&B, gv->tobs[j]));
                num = B_add(&B, B_mul(&B, num, dj), B_mul(&B, tj, den));
                den = B_mul(&B, den, dj);
            }
            /* ll / 2^120 + num / den = (ll den + num 2^120) / (2^120 den) */
            tr_bigint *num2 = B_shl(&B, num, TR_PROB_F);
            n0 = B_add(&B, B_mul(&B, n0, den), num2); n1 = B_add(&B, B_mul(&B, n1, den), num2);
            d0 = d1 = B_mul(&B, D, den);
        }
    } else if (!rc && entry < 3 + mm) {
        uint64_t j = entry - 3;
        if (a.flags & TR_PROB_NEG_INF) { bctx_close(&B); gcache_close(&c); return TR_MATH_OK; }
        if (p->kind == TR_PROB_EXPFAM) {
            bounds2 b;
            if (!exp_grad(p, s, gv, (int32_t)j, &b)) rc = TR_MATH_OVERFLOW;
            else { n0 = B_w(&B, b.num[0]); d0 = B_w(&B, b.den[0]); n1 = B_w(&B, b.num[1]); d1 = B_w(&B, b.den[1]); }
        } else {
            const __int128 Zw = i128_of(s[0]);
            tr_bigint *N = B_i(&B, (__int128)gv->N), *Z = B_i(&B, Zw);
            if (m) {
                n0 = B_add(&B, B_mul(&B, B_w(&B, gv->tobs[j]), Z), B_neg(&B, B_mul(&B, N, B_i(&B, i128_of(s[tr_prob_q_feature((int32_t)j)])))));
                d0 = B_mul(&B, Z, B_i(&B, (__int128)M->fden[j]));
            } else {
                const uint64_t *cnt = p->stats->counts + p->group[r] * p->d;
                n0 = B_add(&B, B_mul(&B, B_i(&B, (__int128)cnt[j]), Z), B_neg(&B, B_mul(&B, N, B_i(&B, (__int128)p->weight[r * p->d + j]))));
                d0 = Z;
            }
            n1 = n0; d1 = d0; *exact = 1;
        }
    } else if (!rc) {
        uint64_t at = entry - 3 - mm, j = 0, l;
        while (at >= mm - j) { at -= mm - j; j++; }
        l = j + at;
        if (a.flags & TR_PROB_NEG_INF) { bctx_close(&B); gcache_close(&c); return TR_MATH_OK; }
        tr_bigint *N = B_i(&B, (__int128)gv->N);
        if (p->kind == TR_PROB_EXPFAM) {
            prange pr;
            exp_hess_p(s, (int32_t)m, (int32_t)j, (int32_t)l, &pr);
            box S0 = box_of(s, 0);
            /* hess = -c P / S0^2, c = N 2^(b_j + b_l) / (D_j D_l): lo from max(P / S0^2), hi from min */
            tr_bigint *c0 = B_shl(&B, N, (uint64_t)(M->bits[j] + M->bits[l]));
            tr_bigint *DD = B_i(&B, (__int128)M->fden[j] * M->fden[l]);
            __int128 s0hi = tr_dist_w_negative(pr.phi) ? S0.hi : S0.lo, s0lo = tr_dist_w_negative(pr.plo) ? S0.lo : S0.hi;
            n0 = B_neg(&B, B_mul(&B, c0, B_w(&B, pr.phi)));
            d0 = B_mul(&B, DD, B_w(&B, tr_dist_w_mul128(s0hi, s0hi)));
            n1 = B_neg(&B, B_mul(&B, c0, B_w(&B, pr.plo)));
            d1 = B_mul(&B, DD, B_w(&B, tr_dist_w_mul128(s0lo, s0lo)));
            if (j == l && !B.err && tr_bigint_sign(n1) > 0) { n1 = B_i(&B, 0); d1 = B_i(&B, 1); }   /* -N Var <= 0 */
            if (j == l && !B.err && tr_bigint_sign(n0) > 0) { n0 = B_i(&B, 0); d0 = B_i(&B, 1); }
        } else {
            const __int128 Zw = i128_of(s[0]);
            tr_bigint *Z = B_i(&B, Zw), *P;
            if (m) {
                P = B_add(&B, B_mul(&B, B_i(&B, i128_of(s[1 + m + tr_prob_pack(m, j, l)])), Z),
                          B_neg(&B, B_mul(&B, B_i(&B, i128_of(s[tr_prob_q_feature((int32_t)j)])),
                                          B_i(&B, i128_of(s[tr_prob_q_feature((int32_t)l)])))));
                d0 = B_mul(&B, B_mul(&B, Z, Z), B_i(&B, (__int128)M->fden[j] * M->fden[l]));
            } else {
                const int64_t *W = p->weight + r * p->d;
                P = B_add(&B, j == l ? B_mul(&B, B_i(&B, (__int128)W[j]), Z) : B_i(&B, 0),
                          B_neg(&B, B_mul(&B, B_i(&B, (__int128)W[j]), B_i(&B, (__int128)W[l]))));
                d0 = B_mul(&B, Z, Z);
            }
            n0 = B_neg(&B, B_mul(&B, N, P));
            n1 = n0; d1 = d0; *exact = 1;
        }
    }
    if (!rc && !B.err) {
        *lo_num = B_copy(&B, n0); *lo_den = B_copy(&B, d0); *hi_num = B_copy(&B, n1); *hi_den = B_copy(&B, d1);
    }
    if (!rc) rc = B.err;
    if (rc) {
        tr_bigint_free(*lo_num); tr_bigint_free(*lo_den); tr_bigint_free(*hi_num); tr_bigint_free(*hi_den);
        *lo_num = *lo_den = *hi_num = *hi_den = NULL;
    }
    bctx_close(&B);
    gcache_close(&c);
    return rc;
}

int32_t tr_prob_ln_bounds(uint64_t lo, uint64_t hi, int64_t out[4]) {
    unsigned __int128 M = ((unsigned __int128)hi << 64) | lo;
    if (!out || M == 0) return TR_MATH_INVALID;
    __int128 a, b;
    tr_prob_ln128(M, &a, &b);
    out[0] = (int64_t)(uint64_t)a; out[1] = (int64_t)(a >> 64); out[2] = (int64_t)(uint64_t)b; out[3] = (int64_t)(b >> 64);
    return TR_MATH_OK;
}
