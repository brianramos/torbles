/* Numeric carriers and exactness conversions (FND-009); see math_carriers.h for every contract.
 * SPDX-License-Identifier: Apache-2.0 */
#include "math_carriers.h"
#include "mathematics.h"
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

_Static_assert(sizeof(tr_math_rational) == sizeof(tr_num_rational), "tr_num_rational has the layout of tr_math_rational");
_Static_assert(offsetof(tr_math_rational, denominator) == offsetof(tr_num_rational, denominator), "rational layout");
_Static_assert(sizeof(tr_gfp) == 32 && sizeof(tr_gfp_elem) == 16 && sizeof(tr_carrier_cinterval) == 32 &&
               sizeof(tr_carrier_desc) == 48, "carrier ABI");

typedef unsigned __int128 u128;
typedef __int128 i128;

#define BIGINT_MAGIC 0x31474942u   /* "BIG1" */
#define BIGQ_MAGIC 0x31514742u     /* "BGQ1" */
#define SERIES_MAGIC 0x31524553u   /* "SER1" */

struct tr_bigint_s { uint32_t magic; int32_t sign; uint64_t n, cap; uint64_t limb[]; };
struct tr_bigq_s { uint32_t magic; uint32_t reserved; tr_bigint *num, *den; };
struct tr_series_s { uint32_t magic; uint32_t order; tr_bigq *coef[]; };

static void seterr(int32_t *e, int32_t v) { if (e) *e = v; }
static int is_bi(const tr_bigint *x) { return x && x->magic == BIGINT_MAGIC; }
static int is_bq(const tr_bigq *q) { return q && q->magic == BIGQ_MAGIC && is_bi(q->num) && is_bi(q->den); }
static int is_ser(const tr_series *s) { return s && s->magic == SERIES_MAGIC; }
/* 0 for a big integer; CARRIER_ERROR for another carrier's handle (a rational or a series); INVALID otherwise */
static int32_t want_bi(const tr_bigint *x) {
    if (is_bi(x)) return TR_MATH_OK;
    if (x && (x->magic == BIGQ_MAGIC || x->magic == SERIES_MAGIC)) return TR_MATH_CARRIER_ERROR;
    return TR_MATH_INVALID;
}
static int32_t want_bq(const tr_bigq *q) {
    if (is_bq(q)) return TR_MATH_OK;
    if (q && (q->magic == BIGINT_MAGIC || q->magic == SERIES_MAGIC)) return TR_MATH_CARRIER_ERROR;
    return TR_MATH_INVALID;
}
static int32_t want_ser(const tr_series *s) {
    if (is_ser(s)) return TR_MATH_OK;
    if (s && (s->magic == BIGINT_MAGIC || s->magic == BIGQ_MAGIC)) return TR_MATH_CARRIER_ERROR;
    return TR_MATH_INVALID;
}

/* ================================================================ budgets */
static const tr_math_budget *budget_of(const tr_math_budget *b, tr_math_budget *store) {
    if (b) return b;
    tr_budget_default(store);
    return store;
}
/* one big number of `limbs` limbs against the limb budget (bytes / 8) and the hard cap */
static int32_t check_limbs(const tr_math_budget *b, uint64_t limbs, tr_math_refusal *r) {
    tr_math_budget d;
    b = budget_of(b, &d);
    uint64_t need = tr_budget_mul(limbs, 8), limit = b->limit[TR_BUDGET_BYTES], cap = TR_BIGINT_MAX_LIMBS * 8;
    if (need > limit || limbs > TR_BIGINT_MAX_LIMBS) {
        if (r) { r->resource = TR_BUDGET_BYTES; r->refused = 1; r->need = need; r->limit = limit < cap ? limit : cap; }
        return TR_MATH_BUDGET;
    }
    return TR_MATH_OK;
}
/* the order of a series: degree order - 1 and support order */
static int32_t check_order(const tr_math_budget *b, uint64_t order, tr_math_refusal *r) {
    tr_math_budget d;
    tr_math_need need;
    b = budget_of(b, &d);
    memset(&need, 0, sizeof need);
    need.need[TR_BUDGET_DEGREE] = order ? order - 1 : 0;
    need.need[TR_BUDGET_SUPPORT] = order;
    return tr_budget_check(b, &need, r);
}

/* ================================================================ carriers and the conversion matrix */
static const tr_carrier_desc DESCS[TR_CARRIER_LAST + 1] = {
    {0, 0, 0, 0, 0, 0, ""},
    {TR_CARRIER_I64, TR_LAW_INTEGER, 1, 1, 64, 0, "i64"},
    {TR_CARRIER_I128, TR_LAW_INTEGER, 1, 1, 128, 0, "i128"},
    {TR_CARRIER_BIGINT, TR_LAW_INTEGER, 1, 1, 0, 0, "bigint"},
    {TR_CARRIER_Q64, TR_LAW_RATIONAL, 1, 1, 64, 0, "q64"},
    {TR_CARRIER_BIGQ, TR_LAW_RATIONAL, 1, 1, 0, 0, "bigq"},
    {TR_CARRIER_GFP, TR_LAW_PRIME_FIELD, 1, 1, 62, 0, "gfp"},
    {TR_CARRIER_BINARY64, TR_LAW_DYADIC, 1, 1, 53, 0, "binary64"},
    {TR_CARRIER_INTERVAL, TR_LAW_REAL_ENCLOSURE, 0, 1, 53, 0, "interval"},
    {TR_CARRIER_CINTERVAL, TR_LAW_COMPLEX_ENCLOSURE, 0, 1, 53, 0, "cinterval"},
    {TR_CARRIER_SERIES, TR_LAW_TRUNCATED_SERIES, 1, 1, 0, 0, "series"},
};
enum { K_F = TR_CONVERT_FORBIDDEN, K_ID = TR_CONVERT_IDENTITY, K_LL = TR_CONVERT_LOSSLESS, K_NA = TR_CONVERT_NARROWING,
       K_EN = TR_CONVERT_ENCLOSING, K_RE = TR_CONVERT_REDUCTION, K_LI = TR_CONVERT_LIFT, K_TT = TR_CONVERT_TRUNCATION };
/* MATRIX[from][to]; columns: I64 I128 BIGINT Q64 BIGQ GFP BINARY64 INTERVAL CINTERVAL SERIES */
static const signed char MATRIX[TR_CARRIER_LAST + 1][TR_CARRIER_LAST + 1] = {
    {0},
    /* I64       */ {0, K_ID, K_LL, K_LL, K_LL, K_LL, K_RE, K_F, K_EN, K_EN, K_LL},
    /* I128      */ {0, K_NA, K_ID, K_LL, K_NA, K_LL, K_RE, K_F, K_EN, K_EN, K_LL},
    /* BIGINT    */ {0, K_NA, K_NA, K_ID, K_NA, K_LL, K_RE, K_F, K_EN, K_EN, K_LL},
    /* Q64       */ {0, K_NA, K_NA, K_NA, K_ID, K_LL, K_RE, K_F, K_EN, K_EN, K_LL},
    /* BIGQ      */ {0, K_NA, K_NA, K_NA, K_NA, K_ID, K_RE, K_F, K_EN, K_EN, K_LL},
    /* GFP       */ {0, K_LI, K_F, K_LI, K_F, K_F, K_ID, K_F, K_F, K_F, K_F},
    /* BINARY64  */ {0, K_F, K_F, K_F, K_F, K_LL, K_F, K_ID, K_LL, K_LL, K_F},
    /* INTERVAL  */ {0, K_F, K_F, K_F, K_F, K_F, K_F, K_F, K_ID, K_LL, K_F},
    /* CINTERVAL */ {0, K_F, K_F, K_F, K_F, K_F, K_F, K_F, K_NA, K_ID, K_F},
    /* SERIES    */ {0, K_F, K_F, K_F, K_F, K_F, K_F, K_F, K_F, K_F, K_TT},
};
static int known(int32_t c) { return c >= 1 && c <= TR_CARRIER_LAST; }
int32_t tr_carrier_describe(int32_t carrier, tr_carrier_desc *out) {
    if (!known(carrier) || !out) return TR_MATH_INVALID;
    *out = DESCS[carrier];
    return TR_MATH_OK;
}
int32_t tr_carrier_conversion(int32_t from, int32_t to) {
    if (!known(from) || !known(to)) return TR_MATH_INVALID;
    return MATRIX[from][to];
}
int32_t tr_carrier_check_conversion(int32_t from, int32_t to) {
    int32_t k = tr_carrier_conversion(from, to);
    if (k < 0) return k;
    return k == TR_CONVERT_FORBIDDEN ? TR_MATH_CARRIER_ERROR : TR_MATH_OK;
}
int32_t tr_carrier_abi(uint64_t *sizes, int32_t n) {
    const uint64_t v[7] = {sizeof(tr_gfp), sizeof(tr_gfp_elem), sizeof(tr_carrier_cinterval), sizeof(tr_carrier_desc),
                           sizeof(tr_math_interval), sizeof(tr_math_i128), sizeof(tr_num_rational)};
    if (!sizes || n < 0) return TR_MATH_INVALID;
    for (int32_t i = 0; i < n && i < 7; i++) sizes[i] = v[i];
    return TR_CARRIERS_ABI;
}
uint64_t tr_carrier_device_bytes(int32_t kernel, uint64_t n, uint64_t m, int32_t flag) {
    return tr_carrier_kernel_bytes(kernel, n, m, flag);
}
int32_t tr_carrier_combine(int32_t a, int32_t b) {
    if (!known(a) || !known(b)) return TR_MATH_INVALID;
    return a == b ? a : TR_MATH_CARRIER_ERROR;
}

/* ================================================================ checked fixed-width carriers */
int32_t tr_carrier_i64_op(int32_t op, int64_t a, int64_t b, int64_t *out) {
    if (!out) return TR_MATH_INVALID;
    int64_t r;
    switch (op) {
    case TR_OP_ADD: if (!tr_add_i64(a, b, &r)) return TR_MATH_OVERFLOW; break;
    case TR_OP_SUB: if (!tr_sub_i64(a, b, &r)) return TR_MATH_OVERFLOW; break;
    case TR_OP_MUL: if (!tr_mul_i64(a, b, &r)) return TR_MATH_OVERFLOW; break;
    case TR_OP_NEG: if (a == INT64_MIN) return TR_MATH_OVERFLOW; r = -a; break;
    case TR_OP_DIV:
        if (b == 0) return TR_MATH_NO_SOLUTION;
        if (a == INT64_MIN && b == -1) return TR_MATH_OVERFLOW;
        r = a / b;
        if ((a % b != 0) && ((a < 0) != (b < 0))) r -= 1;
        break;
    default: return TR_MATH_INVALID;
    }
    *out = r;
    return TR_MATH_OK;
}

int32_t tr_carrier_i128_op(int32_t op, tr_math_i128 a, tr_math_i128 b, tr_math_i128 *out) {
    if (!out) return TR_MATH_INVALID;
    i128 x = tr_i128_get(a), y = tr_i128_get(b), r;
    const i128 MIN = (i128)((u128)1 << 127);
    switch (op) {
    case TR_OP_ADD: if (__builtin_add_overflow(x, y, &r)) return TR_MATH_OVERFLOW; break;
    case TR_OP_SUB: if (__builtin_sub_overflow(x, y, &r)) return TR_MATH_OVERFLOW; break;
    case TR_OP_MUL: if (__builtin_mul_overflow(x, y, &r)) return TR_MATH_OVERFLOW; break;
    case TR_OP_NEG: if (x == MIN) return TR_MATH_OVERFLOW; r = -x; break;
    case TR_OP_DIV:
        if (y == 0) return TR_MATH_NO_SOLUTION;
        if (x == MIN && y == -1) return TR_MATH_OVERFLOW;
        r = x / y;
        if ((x % y != 0) && ((x < 0) != (y < 0))) r -= 1;
        break;
    default: return TR_MATH_INVALID;
    }
    *out = tr_i128_set(r);
    return TR_MATH_OK;
}
int32_t tr_carrier_i128_cmp(tr_math_i128 a, tr_math_i128 b) {
    i128 x = tr_i128_get(a), y = tr_i128_get(b);
    return x < y ? -1 : x > y;
}

int32_t tr_carrier_q64_op(int32_t op, tr_num_rational a, tr_num_rational b, tr_num_rational *out) {
    if (!out || a.denominator == 0) return TR_MATH_INVALID;
    int binary = op == TR_OP_ADD || op == TR_OP_SUB || op == TR_OP_MUL || op == TR_OP_DIV;
    if (binary && b.denominator == 0) return TR_MATH_INVALID;
    tr_num_rational r;
    int ok;
    switch (op) {
    case TR_OP_ADD: ok = tr_q_add(a, b, &r); break;
    case TR_OP_SUB: ok = tr_q_sub(a, b, &r); break;
    case TR_OP_MUL: ok = tr_q_mul(a, b, &r); break;
    case TR_OP_DIV: if (b.numerator == 0) return TR_MATH_NO_SOLUTION; ok = tr_q_div(a, b, &r); break;
    case TR_OP_NEG: ok = tr_q_make128(-(i128)a.numerator, a.denominator, &r); break;
    case TR_OP_INV: if (a.numerator == 0) return TR_MATH_NO_SOLUTION; ok = tr_q_make128(a.denominator, a.numerator, &r); break;
    default: return TR_MATH_INVALID;
    }
    if (!ok) return TR_MATH_OVERFLOW;
    *out = r;
    return TR_MATH_OK;
}
int32_t tr_carrier_q64_cmp(tr_num_rational a, tr_num_rational b, int32_t *out) {
    if (!out || a.denominator == 0 || b.denominator == 0) return TR_MATH_INVALID;
    i128 l = (i128)a.numerator * b.denominator, r = (i128)b.numerator * a.denominator;
    int c = l < r ? -1 : l > r;
    if ((a.denominator < 0) != (b.denominator < 0)) c = -c;
    *out = c;
    return TR_MATH_OK;
}
int32_t tr_carrier_i128_to_i64(tr_math_i128 v, int64_t *out) {
    if (!out) return TR_MATH_INVALID;
    if (!tr_i128_fits_i64(v)) return TR_MATH_OVERFLOW;
    *out = (int64_t)tr_i128_get(v);
    return TR_MATH_OK;
}
int32_t tr_carrier_q64_to_i64(tr_num_rational q, int64_t *out) {
    if (!out || q.denominator == 0) return TR_MATH_INVALID;
    i128 n = q.numerator, d = q.denominator;
    if (n % d != 0) return TR_MATH_NO_SOLUTION;
    i128 v = n / d;
    if (v < INT64_MIN || v > INT64_MAX) return TR_MATH_OVERFLOW;
    *out = (int64_t)v;
    return TR_MATH_OK;
}

/* ================================================================ limb arrays (little endian, 64-bit limbs) */
static uint64_t mpn_norm(const uint64_t *x, uint64_t n) { while (n && !x[n - 1]) n--; return n; }
static uint64_t mpn_bitlen(const uint64_t *x, uint64_t n) {
    return n ? 64 * (n - 1) + (uint64_t)(64 - __builtin_clzll(x[n - 1])) : 0;
}
static int mpn_cmp(const uint64_t *a, uint64_t na, const uint64_t *b, uint64_t nb) {
    if (na != nb) return na < nb ? -1 : 1;
    for (uint64_t i = na; i-- > 0;)
        if (a[i] != b[i]) return a[i] < b[i] ? -1 : 1;
    return 0;
}
/* r[0..na] = a + b for na >= nb (r has na + 1 limbs, may alias a); returns na + 1 */
static uint64_t mpn_add(uint64_t *r, const uint64_t *a, uint64_t na, const uint64_t *b, uint64_t nb) {
    u128 c = 0;
    uint64_t i = 0;
    for (; i < nb; i++) { c += (u128)a[i] + b[i]; r[i] = (uint64_t)c; c >>= 64; }
    for (; i < na; i++) { c += a[i]; r[i] = (uint64_t)c; c >>= 64; }
    r[na] = (uint64_t)c;
    return na + 1;
}
/* r = a - b for a >= b as numbers (na >= nb; r may alias a); returns na */
static uint64_t mpn_sub(uint64_t *r, const uint64_t *a, uint64_t na, const uint64_t *b, uint64_t nb) {
    uint64_t borrow = 0, i = 0;
    for (; i < nb; i++) {
        uint64_t x = a[i], y = b[i], d = x - y - borrow;
        borrow = (x < y) || (x - y < borrow);
        r[i] = d;
    }
    for (; i < na; i++) { uint64_t x = a[i]; r[i] = x - borrow; borrow = x < borrow; }
    return na;
}
/* r[0..na+nb) = a * b (schoolbook; r must not alias a or b) */
static void mpn_mul(uint64_t *r, const uint64_t *a, uint64_t na, const uint64_t *b, uint64_t nb) {
    if (na < nb) { const uint64_t *t = a; a = b; b = t; uint64_t tn = na; na = nb; nb = tn; }
    memset(r, 0, (size_t)(na + nb) * sizeof *r);
    for (uint64_t j = 0; j < nb; j++) {
        uint64_t bj = b[j];
        if (!bj) continue;
        u128 c = 0;
        for (uint64_t i = 0; i < na; i++) { c += (u128)a[i] * bj + r[i + j]; r[i + j] = (uint64_t)c; c >>= 64; }
        r[j + na] = (uint64_t)c;
    }
}
/* x = x * m + add in place over n limbs; returns the carry limb */
static uint64_t mpn_mul1_add(uint64_t *x, uint64_t n, uint64_t m, uint64_t add) {
    u128 c = add;
    for (uint64_t i = 0; i < n; i++) { c += (u128)x[i] * m; x[i] = (uint64_t)c; c >>= 64; }
    return (uint64_t)c;
}
/* q = x / d (q may alias x or be NULL), returns x mod d; d != 0 */
static uint64_t mpn_divrem_1(uint64_t *q, const uint64_t *x, uint64_t n, uint64_t d) {
    u128 rem = 0;
    for (uint64_t i = n; i-- > 0;) {
        u128 cur = (rem << 64) | x[i];
        if (q) q[i] = (uint64_t)(cur / d);
        rem = cur % d;
    }
    return (uint64_t)rem;
}
/* division by a normalized d (top bit set) with v = floor((2^128 - 1) / d) - 2^64 (Moller and Granlund 2011) */
static uint64_t mpn_divrem_1_norm(uint64_t *q, const uint64_t *x, uint64_t n, uint64_t d, uint64_t v) {
    uint64_t r = 0;
    for (uint64_t i = n; i-- > 0;) {
        u128 p = (u128)v * r;
        p += ((u128)r << 64) | x[i];
        uint64_t q1 = (uint64_t)(p >> 64) + 1, q0 = (uint64_t)p, rr = x[i] - q1 * d;
        if (rr > q0) { q1--; rr += d; }
        if (rr >= d) { q1++; rr -= d; }
        q[i] = q1;
        r = rr;
    }
    return r;
}
/* the 64 bits of x starting at bit s (bits past the top are 0) */
static uint64_t mpn_bits_at(const uint64_t *x, uint64_t n, uint64_t s) {
    uint64_t w = s / 64;
    unsigned o = (unsigned)(s % 64);
    uint64_t lo = w < n ? x[w] : 0, hi = w + 1 < n ? x[w + 1] : 0;
    return o ? (lo >> o) | (hi << (64 - o)) : lo;
}
/* r = a << bits; r has na + bits / 64 + 1 limbs (r must not alias a); returns that count */
static uint64_t mpn_lshift(uint64_t *r, const uint64_t *a, uint64_t na, uint64_t bits) {
    uint64_t w = bits / 64;
    unsigned s = (unsigned)(bits % 64);
    memset(r, 0, (size_t)w * sizeof *r);
    if (!s) { memcpy(r + w, a, (size_t)na * sizeof *r); r[w + na] = 0; }
    else {
        uint64_t carry = 0;
        for (uint64_t i = 0; i < na; i++) { r[w + i] = (a[i] << s) | carry; carry = a[i] >> (64 - s); }
        r[w + na] = carry;
    }
    return w + na + 1;
}
/* Knuth's algorithm D: q[0..na-nb] = a / b, r[0..nb) = a mod b, for na >= nb >= 2 and b[nb-1] != 0; un (na + 1
 * limbs) and vn (nb limbs) are scratch. q or r may be NULL. */
static void mpn_divrem(uint64_t *q, uint64_t *r, const uint64_t *a, uint64_t na, const uint64_t *b, uint64_t nb,
                       uint64_t *un, uint64_t *vn) {
    unsigned s = (unsigned)__builtin_clzll(b[nb - 1]);
    if (s) {
        for (uint64_t i = nb - 1; i > 0; i--) vn[i] = (b[i] << s) | (b[i - 1] >> (64 - s));
        vn[0] = b[0] << s;
        un[na] = a[na - 1] >> (64 - s);
        for (uint64_t i = na - 1; i > 0; i--) un[i] = (a[i] << s) | (a[i - 1] >> (64 - s));
        un[0] = a[0] << s;
    } else {
        memcpy(vn, b, (size_t)nb * sizeof *vn);
        memcpy(un, a, (size_t)na * sizeof *un);
        un[na] = 0;
    }
    uint64_t v1 = vn[nb - 1], v2 = vn[nb - 2];
    for (uint64_t j = na - nb + 1; j-- > 0;) {
        u128 num = ((u128)un[j + nb] << 64) | un[j + nb - 1];
        u128 qhat = num / v1, rhat = num - qhat * v1;
        while ((qhat >> 64) || qhat * v2 > ((rhat << 64) | un[j + nb - 2])) {
            qhat--;
            rhat += v1;
            if (rhat >> 64) break;
        }
        i128 k = 0, t;
        for (uint64_t i = 0; i < nb; i++) {
            u128 p = qhat * vn[i];
            t = (i128)un[i + j] - k - (i128)(uint64_t)p;
            un[i + j] = (uint64_t)t;
            k = (i128)(uint64_t)(p >> 64) - (t >> 64);
        }
        t = (i128)un[j + nb] - k;
        un[j + nb] = (uint64_t)t;
        if (t < 0) {
            qhat--;
            u128 c = 0;
            for (uint64_t i = 0; i < nb; i++) { c += (u128)un[i + j] + vn[i]; un[i + j] = (uint64_t)c; c >>= 64; }
            un[j + nb] += (uint64_t)c;
        }
        if (q) q[j] = (uint64_t)qhat;
    }
    if (r) {
        if (s) {
            for (uint64_t i = 0; i + 1 < nb; i++) r[i] = (un[i] >> s) | (un[i + 1] << (64 - s));
            r[nb - 1] = (un[nb - 1] >> s) | (un[nb] << (64 - s));
        } else memcpy(r, un, (size_t)nb * sizeof *r);
    }
}

/* ================================================================ big integers */
static tr_bigint *bi_new(uint64_t cap, const tr_math_budget *b, tr_math_refusal *r, int32_t *err) {
    int32_t rc = check_limbs(b, cap, r);
    if (rc) { seterr(err, rc); return NULL; }
    tr_bigint *x = (tr_bigint *)malloc(sizeof(tr_bigint) + (size_t)(cap ? cap : 1) * sizeof(uint64_t));
    if (!x) { seterr(err, TR_MATH_MEMORY); return NULL; }
    x->magic = BIGINT_MAGIC; x->sign = 0; x->n = 0; x->cap = cap;
    return x;
}
static tr_bigint *bi_done(tr_bigint *x, uint64_t n, int sign, int32_t *err) {
    n = mpn_norm(x->limb, n);
    x->n = n;
    x->sign = n ? (sign < 0 ? -1 : 1) : 0;
    seterr(err, TR_MATH_OK);
    return x;
}
static tr_bigint *bi_from_mpn(const uint64_t *limbs, uint64_t n, int sign, const tr_math_budget *b,
                              tr_math_refusal *r, int32_t *err) {
    n = mpn_norm(limbs, n);
    tr_bigint *x = bi_new(n, b, r, err);
    if (!x) return NULL;
    if (n) memcpy(x->limb, limbs, (size_t)n * sizeof *limbs);
    return bi_done(x, n, sign, err);
}
static tr_bigint *bi_u64(uint64_t v, int sign, const tr_math_budget *b, tr_math_refusal *r, int32_t *err) {
    tr_bigint *x = bi_new(1, b, r, err);
    if (!x) return NULL;
    x->limb[0] = v;
    return bi_done(x, 1, sign, err);
}
void tr_bigint_free(tr_bigint *x) {
    if (!is_bi(x)) return;
    x->magic = 0;
    free(x);
}
void tr_bigint_free_array(tr_bigint **x, uint64_t n) {
    if (!x) return;
    for (uint64_t i = 0; i < n; i++) { tr_bigint_free(x[i]); x[i] = NULL; }
}

tr_bigint *tr_bigint_from_i64(int64_t v, const tr_math_budget *b, tr_math_refusal *r, int32_t *err) {
    return bi_u64(tr_abs_u64(v), v < 0 ? -1 : 1, b, r, err);
}
tr_bigint *tr_bigint_from_i128(tr_math_i128 v, const tr_math_budget *b, tr_math_refusal *r, int32_t *err) {
    i128 x = tr_i128_get(v);
    u128 m = x < 0 ? (u128)0 - (u128)x : (u128)x;
    tr_bigint *y = bi_new(2, b, r, err);
    if (!y) return NULL;
    y->limb[0] = (uint64_t)m; y->limb[1] = (uint64_t)(m >> 64);
    return bi_done(y, 2, x < 0 ? -1 : 1, err);
}
tr_bigint *tr_bigint_from_limbs(int32_t sign, const uint64_t *limbs, uint64_t n, const tr_math_budget *b,
                                tr_math_refusal *r, int32_t *err) {
    if (sign < -1 || sign > 1 || (n && !limbs)) { seterr(err, TR_MATH_INVALID); return NULL; }
    int32_t rc = check_limbs(b, n, r);
    if (rc) { seterr(err, rc); return NULL; }
    if (sign == 0) n = 0;
    return bi_from_mpn(limbs, n, sign, b, r, err);
}
tr_bigint *tr_bigint_from_twos(const uint64_t *words, uint64_t n, const tr_math_budget *b, tr_math_refusal *r,
                               int32_t *err) {
    if (!words || !n) { seterr(err, TR_MATH_INVALID); return NULL; }
    tr_bigint *x = bi_new(n, b, r, err);
    if (!x) return NULL;
    int neg = (words[n - 1] >> 63) != 0;
    if (!neg) { memcpy(x->limb, words, (size_t)n * sizeof *words); return bi_done(x, n, 1, err); }
    u128 c = 1;
    for (uint64_t i = 0; i < n; i++) { c += (uint64_t)~words[i]; x->limb[i] = (uint64_t)c; c >>= 64; }
    return bi_done(x, n, -1, err);
}
int32_t tr_bigint_to_i64(const tr_bigint *x, int64_t *out) {
    int32_t rc = want_bi(x);
    if (rc || !out) return rc ? rc : TR_MATH_INVALID;
    if (x->n == 0) { *out = 0; return TR_MATH_OK; }
    if (x->n > 1) return TR_MATH_OVERFLOW;
    uint64_t m = x->limb[0];
    if (x->sign > 0) { if (m > (uint64_t)INT64_MAX) return TR_MATH_OVERFLOW; *out = (int64_t)m; }
    else { if (m > (uint64_t)1 << 63) return TR_MATH_OVERFLOW; *out = (int64_t)((uint64_t)0 - m); }
    return TR_MATH_OK;
}
int32_t tr_bigint_to_i128(const tr_bigint *x, tr_math_i128 *out) {
    int32_t rc = want_bi(x);
    if (rc || !out) return rc ? rc : TR_MATH_INVALID;
    if (x->n > 2) return TR_MATH_OVERFLOW;
    u128 m = x->n == 0 ? 0 : x->n == 1 ? x->limb[0] : ((u128)x->limb[1] << 64) | x->limb[0];
    u128 top = (u128)1 << 127;
    if (x->sign >= 0) { if (m >= top) return TR_MATH_OVERFLOW; *out = tr_i128_set((i128)m); }
    else { if (m > top) return TR_MATH_OVERFLOW; *out = tr_i128_set((i128)((u128)0 - m)); }
    return TR_MATH_OK;
}
int32_t tr_bigint_export(const tr_bigint *x, int32_t *sign, uint64_t *limbs, uint64_t capacity, uint64_t *count) {
    int32_t rc = want_bi(x);
    if (rc) return rc;
    if ((x->n && (!limbs || capacity < x->n))) return TR_MATH_INVALID;
    if (x->n) memcpy(limbs, x->limb, (size_t)x->n * sizeof *limbs);
    if (sign) *sign = x->sign;
    if (count) *count = x->n;
    return TR_MATH_OK;
}
int32_t tr_bigint_sign(const tr_bigint *x) { int32_t rc = want_bi(x); return rc ? rc : x->sign; }
uint64_t tr_bigint_limb_count(const tr_bigint *x) { return is_bi(x) ? x->n : 0; }
uint64_t tr_bigint_bit_length(const tr_bigint *x) { return is_bi(x) ? mpn_bitlen(x->limb, x->n) : 0; }
int32_t tr_bigint_is_handle(const void *p) { return is_bi((const tr_bigint *)p); }
static int bi_cmp(const tr_bigint *a, const tr_bigint *b) {
    if (a->sign != b->sign) return a->sign < b->sign ? -1 : 1;
    int c = mpn_cmp(a->limb, a->n, b->limb, b->n);
    return a->sign < 0 ? -c : c;
}
int32_t tr_bigint_cmp(const tr_bigint *a, const tr_bigint *b) {
    int32_t rc = want_bi(a);
    if (!rc) rc = want_bi(b);
    return rc ? rc : bi_cmp(a, b);
}
static tr_bigint *bi_signed_copy(const tr_bigint *a, int sign, const tr_math_budget *b, tr_math_refusal *r, int32_t *err) {
    int32_t rc = want_bi(a);
    if (rc) { seterr(err, rc); return NULL; }
    return bi_from_mpn(a->limb, a->n, sign, b, r, err);
}
tr_bigint *tr_bigint_copy(const tr_bigint *a, const tr_math_budget *b, tr_math_refusal *r, int32_t *err) {
    return bi_signed_copy(a, a && is_bi(a) ? a->sign : 1, b, r, err);
}
tr_bigint *tr_bigint_neg(const tr_bigint *a, const tr_math_budget *b, tr_math_refusal *r, int32_t *err) {
    return bi_signed_copy(a, a && is_bi(a) ? -a->sign : 1, b, r, err);
}
tr_bigint *tr_bigint_abs(const tr_bigint *a, const tr_math_budget *b, tr_math_refusal *r, int32_t *err) {
    return bi_signed_copy(a, 1, b, r, err);
}

static tr_bigint *bi_addsub(const tr_bigint *a, const tr_bigint *c, int negate, const tr_math_budget *b,
                            tr_math_refusal *r, int32_t *err) {
    int32_t rc = want_bi(a);
    if (!rc) rc = want_bi(c);
    if (rc) { seterr(err, rc); return NULL; }
    int sa = a->sign, sc = negate ? -c->sign : c->sign;
    tr_bigint *x = bi_new((a->n > c->n ? a->n : c->n) + 1, b, r, err);
    if (!x) return NULL;
    if (sc == 0) { if (a->n) memcpy(x->limb, a->limb, (size_t)a->n * 8); return bi_done(x, a->n, sa, err); }
    if (sa == 0) { memcpy(x->limb, c->limb, (size_t)c->n * 8); return bi_done(x, c->n, sc, err); }
    if (sa == sc) {
        uint64_t n = a->n >= c->n ? mpn_add(x->limb, a->limb, a->n, c->limb, c->n)
                                  : mpn_add(x->limb, c->limb, c->n, a->limb, a->n);
        return bi_done(x, n, sa, err);
    }
    int cmp = mpn_cmp(a->limb, a->n, c->limb, c->n);
    if (cmp == 0) return bi_done(x, 0, 0, err);
    if (cmp > 0) return bi_done(x, mpn_sub(x->limb, a->limb, a->n, c->limb, c->n), sa, err);
    return bi_done(x, mpn_sub(x->limb, c->limb, c->n, a->limb, a->n), sc, err);
}
tr_bigint *tr_bigint_add(const tr_bigint *a, const tr_bigint *c, const tr_math_budget *b, tr_math_refusal *r, int32_t *err) {
    return bi_addsub(a, c, 0, b, r, err);
}
tr_bigint *tr_bigint_sub(const tr_bigint *a, const tr_bigint *c, const tr_math_budget *b, tr_math_refusal *r, int32_t *err) {
    return bi_addsub(a, c, 1, b, r, err);
}
tr_bigint *tr_bigint_mul(const tr_bigint *a, const tr_bigint *c, const tr_math_budget *b, tr_math_refusal *r, int32_t *err) {
    int32_t rc = want_bi(a);
    if (!rc) rc = want_bi(c);
    if (rc) { seterr(err, rc); return NULL; }
    tr_bigint *x = bi_new(a->n + c->n, b, r, err);
    if (!x) return NULL;
    if (!a->n || !c->n) return bi_done(x, 0, 0, err);
    mpn_mul(x->limb, a->limb, a->n, c->limb, c->n);
    return bi_done(x, a->n + c->n, a->sign * c->sign, err);
}

int32_t tr_bigint_divmod(const tr_bigint *a, const tr_bigint *c, int32_t rounding, tr_bigint **qo, tr_bigint **ro,
                         const tr_math_budget *b, tr_math_refusal *r) {
    int32_t rc = want_bi(a);
    if (!rc) rc = want_bi(c);
    if (rc) return rc;
    if (rounding != TR_DIV_FLOOR && rounding != TR_DIV_TRUNC) return TR_MATH_INVALID;
    if (c->n == 0) return TR_MATH_NO_SOLUTION;
    uint64_t na = a->n, nc = c->n, big = (na > nc ? na : nc) + 2;
    rc = check_limbs(b, big, r);
    if (rc) return rc;
    uint64_t qcap = na >= nc ? na - nc + 2 : 1, rcap = nc;
    int32_t err = 0;
    tr_bigint *q = bi_new(qcap, b, r, &err), *m = q ? bi_new(rcap, b, r, &err) : NULL;
    uint64_t *scratch = NULL;
    if (!q || !m) { tr_bigint_free(q); tr_bigint_free(m); return err; }
    memset(q->limb, 0, (size_t)qcap * 8);
    memset(m->limb, 0, (size_t)rcap * 8);
    uint64_t qn = 0, rn = 0;
    if (mpn_cmp(a->limb, na, c->limb, nc) < 0) {
        if (na) memcpy(m->limb, a->limb, (size_t)na * 8);
        rn = na;
    } else if (nc == 1) {
        m->limb[0] = mpn_divrem_1(q->limb, a->limb, na, c->limb[0]);
        qn = na; rn = 1;
    } else {
        scratch = (uint64_t *)malloc((size_t)(na + 1 + nc) * sizeof *scratch);
        if (!scratch) { tr_bigint_free(q); tr_bigint_free(m); return TR_MATH_MEMORY; }
        mpn_divrem(q->limb, m->limb, a->limb, na, c->limb, nc, scratch, scratch + na + 1);
        free(scratch);
        qn = na - nc + 1; rn = nc;
    }
    qn = mpn_norm(q->limb, qn); rn = mpn_norm(m->limb, rn);
    int qs = a->sign * c->sign, rs = a->sign;
    if (rounding == TR_DIV_FLOOR && rn && a->sign != c->sign) {
        /* floor: q - 1 (magnitude + 1) and r + c (magnitude |c| - |r|, the sign of c) */
        uint64_t one = 1;
        qn = mpn_norm(q->limb, mpn_add(q->limb, q->limb, qn, &one, qn ? 1 : 0));
        if (!qn) { q->limb[0] = 1; qn = 1; }
        uint64_t *tmp = (uint64_t *)malloc((size_t)nc * sizeof *tmp);
        if (!tmp) { tr_bigint_free(q); tr_bigint_free(m); return TR_MATH_MEMORY; }
        mpn_sub(tmp, c->limb, nc, m->limb, rn);
        memcpy(m->limb, tmp, (size_t)nc * 8);
        free(tmp);
        rn = mpn_norm(m->limb, nc);
        qs = -1;
        rs = c->sign;
    }
    int32_t e2;
    bi_done(q, qn, qs, &e2);
    bi_done(m, rn, rs, &e2);
    if (qo) *qo = q; else tr_bigint_free(q);
    if (ro) *ro = m; else tr_bigint_free(m);
    return TR_MATH_OK;
}
tr_bigint *tr_bigint_divexact(const tr_bigint *a, const tr_bigint *c, const tr_math_budget *b, tr_math_refusal *r,
                              int32_t *err) {
    tr_bigint *q = NULL, *m = NULL;
    int32_t rc = tr_bigint_divmod(a, c, TR_DIV_TRUNC, &q, &m, b, r);
    if (rc) { seterr(err, rc); return NULL; }
    int exact = m->n == 0;
    tr_bigint_free(m);
    if (!exact) { tr_bigint_free(q); seterr(err, TR_MATH_NO_SOLUTION); return NULL; }
    seterr(err, TR_MATH_OK);
    return q;
}

/* r = A x + B y over n limbs (y has ny <= n limbs, zero above), |A|, |B| < 2^62; 0, or -1 when the result is
 * negative or does not fit n limbs */
static int mpn_lincomb(uint64_t *r, const uint64_t *x, uint64_t nx, const uint64_t *y, uint64_t ny, int64_t A, int64_t B) {
    i128 acc = 0;
    for (uint64_t i = 0; i < nx; i++) {
        acc += (i128)A * (i128)x[i] + (i128)B * (i128)(i < ny ? y[i] : 0);
        r[i] = (uint64_t)acc;
        acc >>= 64;
    }
    return acc == 0 ? 0 : -1;
}
/* Lehmer's gcd of x (nx limbs) and y (ny limbs) in buffers of cap limbs (t1, t2 also cap; un cap + 1, vn cap);
 * returns the limb count of the gcd, left in *res (one of the buffers) */
static uint64_t mpn_gcd(uint64_t *x, uint64_t nx, uint64_t *y, uint64_t ny, uint64_t *t1, uint64_t *t2,
                        uint64_t *un, uint64_t *vn, uint64_t **res) {
    for (;;) {
        nx = mpn_norm(x, nx); ny = mpn_norm(y, ny);
        if (mpn_cmp(x, nx, y, ny) < 0) { uint64_t *t = x; x = y; y = t; uint64_t tn = nx; nx = ny; ny = tn; }
        if (ny == 0) { *res = x; return nx; }
        if (nx == 1) {
            uint64_t X = x[0], Y = y[0];
            while (Y) { uint64_t t = X % Y; X = Y; Y = t; }
            x[0] = X; *res = x; return 1;
        }
        if (nx == 2) {
            u128 X = ((u128)x[1] << 64) | x[0], Y = ny > 1 ? ((u128)y[1] << 64) | y[0] : y[0];
            while (Y) { u128 t = X % Y; X = Y; Y = t; }
            x[0] = (uint64_t)X; x[1] = (uint64_t)(X >> 64); *res = x;
            return mpn_norm(x, 2);
        }
        uint64_t s = mpn_bitlen(x, nx) - 62;
        int64_t ah = (int64_t)mpn_bits_at(x, nx, s), bh = (int64_t)mpn_bits_at(y, ny, s);
        int64_t A = 1, B = 0, C = 0, D = 1;
        for (;;) {
            int64_t yc = bh + C, yd = bh + D, xa = ah + A, xb = ah + B;
            if (yc <= 0 || yd <= 0 || xa < 0 || xb < 0) break;
            int64_t q = xa / yc;
            if (q != xb / yd) break;
            int64_t t = A - q * C; A = C; C = t;
            t = B - q * D; B = D; D = t;
            t = ah - q * bh; ah = bh; bh = t;
        }
        if (B != 0 && !mpn_lincomb(t1, x, nx, y, ny, A, B) && !mpn_lincomb(t2, x, nx, y, ny, C, D)) {
            uint64_t *t = x; x = t1; t1 = t;
            t = y; y = t2; t2 = t;
            ny = nx;
            continue;
        }
        /* one Euclidean division step: (x, y) <- (y, x mod y) */
        uint64_t nr;
        if (ny == 1) { t1[0] = mpn_divrem_1(NULL, x, nx, y[0]); nr = 1; }
        else { mpn_divrem(NULL, t1, x, nx, y, ny, un, vn); nr = ny; }
        uint64_t *t = x; x = y; y = t1; t1 = t;
        nx = ny; ny = nr;
    }
}
tr_bigint *tr_bigint_gcd(const tr_bigint *a, const tr_bigint *c, const tr_math_budget *b, tr_math_refusal *r, int32_t *err) {
    int32_t rc = want_bi(a);
    if (!rc) rc = want_bi(c);
    if (rc) { seterr(err, rc); return NULL; }
    uint64_t cap = a->n > c->n ? a->n : c->n;
    rc = check_limbs(b, cap + 1, r);
    if (rc) { seterr(err, rc); return NULL; }
    if (!a->n || !c->n) return bi_from_mpn(a->n ? a->limb : c->limb, a->n ? a->n : c->n, 1, b, r, err);
    uint64_t *buf = (uint64_t *)calloc((size_t)(6 * cap + 1), sizeof *buf);
    if (!buf) { seterr(err, TR_MATH_MEMORY); return NULL; }
    uint64_t *x = buf, *y = buf + cap, *t1 = buf + 2 * cap, *t2 = buf + 3 * cap, *un = buf + 4 * cap,
             *vn = buf + 5 * cap + 1, *res = NULL;
    memcpy(x, a->limb, (size_t)a->n * 8);
    memcpy(y, c->limb, (size_t)c->n * 8);
    uint64_t n = mpn_gcd(x, a->n, y, c->n, t1, t2, un, vn, &res);
    tr_bigint *g = bi_from_mpn(res, n, 1, b, r, err);
    free(buf);
    return g;
}

tr_bigint *tr_bigint_shl(const tr_bigint *a, uint64_t bits, const tr_math_budget *b, tr_math_refusal *r, int32_t *err) {
    int32_t rc = want_bi(a);
    if (rc) { seterr(err, rc); return NULL; }
    if (!a->n) { tr_bigint *z = bi_new(0, b, r, err); return z ? bi_done(z, 0, 0, err) : NULL; }
    uint64_t cap = tr_budget_add(a->n, bits / 64 + 1);
    tr_bigint *x = bi_new(cap, b, r, err);
    if (!x) return NULL;
    uint64_t n = mpn_lshift(x->limb, a->limb, a->n, bits);
    return bi_done(x, n, a->sign, err);
}
tr_bigint *tr_bigint_shr(const tr_bigint *a, uint64_t bits, const tr_math_budget *b, tr_math_refusal *r, int32_t *err) {
    int32_t rc = want_bi(a);
    if (rc) { seterr(err, rc); return NULL; }
    tr_bigint *x = bi_new(a->n + 1, b, r, err);
    if (!x) return NULL;
    uint64_t w = bits / 64;
    unsigned s = (unsigned)(bits % 64);
    if (w >= a->n) {
        if (a->sign < 0) { x->limb[0] = 1; return bi_done(x, 1, -1, err); }   /* floor of a negative fraction */
        return bi_done(x, 0, 0, err);
    }
    uint64_t n = a->n - w;
    int lost = 0;
    for (uint64_t i = 0; i < w && !lost; i++) lost = a->limb[i] != 0;
    if (s && (a->limb[w] & (((uint64_t)1 << s) - 1))) lost = 1;
    for (uint64_t i = 0; i < n; i++) {
        uint64_t lo = a->limb[w + i] >> s, hi = (s && w + i + 1 < a->n) ? a->limb[w + i + 1] << (64 - s) : 0;
        x->limb[i] = lo | hi;
    }
    x->limb[n] = 0;
    if (a->sign < 0 && lost) {
        uint64_t one = 1;
        n = mpn_add(x->limb, x->limb, n, &one, 1);
    }
    return bi_done(x, n, a->sign, err);
}
tr_bigint *tr_bigint_pow(const tr_bigint *a, uint64_t e, const tr_math_budget *b, tr_math_refusal *r, int32_t *err) {
    int32_t rc = want_bi(a);
    if (rc) { seterr(err, rc); return NULL; }
    if (e == 0) return bi_u64(1, 1, b, r, err);
    if (a->n == 0 || (a->n == 1 && a->limb[0] == 1)) {
        int sign = a->sign < 0 && (e & 1) ? -1 : 1;
        return a->n ? bi_u64(1, sign, b, r, err) : bi_u64(0, 1, b, r, err);
    }
    uint64_t bits = mpn_bitlen(a->limb, a->n), total = tr_budget_mul(e, bits);
    uint64_t cap = total == TR_BUDGET_UNKNOWN ? TR_BUDGET_UNKNOWN : tr_budget_add(total / 64 + (total % 64 != 0), 1);
    tr_bigint *x = bi_new(cap, b, r, err);
    if (!x) return NULL;
    uint64_t *tmp = (uint64_t *)malloc((size_t)cap * sizeof *tmp);
    if (!tmp) { tr_bigint_free(x); seterr(err, TR_MATH_MEMORY); return NULL; }
    uint64_t *cur = x->limb, *nxt = tmp, n = a->n;
    memcpy(cur, a->limb, (size_t)n * 8);
    int top = 63 - __builtin_clzll(e);
    for (int i = top - 1; i >= 0; i--) {
        mpn_mul(nxt, cur, n, cur, n);
        n = mpn_norm(nxt, 2 * n);
        uint64_t *t = cur; cur = nxt; nxt = t;
        if ((e >> i) & 1) {
            mpn_mul(nxt, cur, n, a->limb, a->n);
            n = mpn_norm(nxt, n + a->n);
            t = cur; cur = nxt; nxt = t;
        }
    }
    if (cur != x->limb) memcpy(x->limb, cur, (size_t)n * 8);
    free(tmp);
    return bi_done(x, n, a->sign < 0 && (e & 1) ? -1 : 1, err);
}
tr_bigint *tr_bigint_isqrt(const tr_bigint *a, const tr_math_budget *b, tr_math_refusal *r, int32_t *err) {
    int32_t rc = want_bi(a);
    if (rc) { seterr(err, rc); return NULL; }
    if (a->sign < 0) { seterr(err, TR_MATH_NO_SOLUTION); return NULL; }
    if (!a->n) return bi_u64(0, 1, b, r, err);
    /* Newton from x0 = 2^ceil(bits / 2) >= sqrt(a): x <- (x + a / x) / 2 decreases to floor(sqrt(a)) */
    uint64_t bits = mpn_bitlen(a->limb, a->n);
    tr_bigint *one = bi_u64(1, 1, b, r, err), *x = one ? tr_bigint_shl(one, (bits + 1) / 2, b, r, err) : NULL;
    tr_bigint_free(one);
    while (x) {
        tr_bigint *q = NULL, *sum, *y;
        rc = tr_bigint_divmod(a, x, TR_DIV_FLOOR, &q, NULL, b, r);
        if (rc) { tr_bigint_free(x); seterr(err, rc); return NULL; }
        sum = tr_bigint_add(x, q, b, r, err);
        tr_bigint_free(q);
        y = sum ? tr_bigint_shr(sum, 1, b, r, err) : NULL;
        tr_bigint_free(sum);
        if (!y) { tr_bigint_free(x); return NULL; }
        if (bi_cmp(y, x) >= 0) { tr_bigint_free(y); seterr(err, TR_MATH_OK); return x; }
        tr_bigint_free(x);
        x = y;
    }
    return NULL;
}
int32_t tr_bigint_mod_u64(const tr_bigint *a, uint64_t m, uint64_t *out) {
    int32_t rc = want_bi(a);
    if (rc) return rc;
    if (!m || !out) return TR_MATH_INVALID;
    uint64_t v = mpn_divrem_1(NULL, a->limb, a->n, m);
    *out = a->sign < 0 && v ? m - v : v;
    return TR_MATH_OK;
}

/* ---------------------------------------------------------------- decimal text */
#define TEN19 10000000000000000000ULL
uint64_t tr_bigint_decimal_size(const tr_bigint *x) {
    if (!is_bi(x)) return 0;
    uint64_t bits = mpn_bitlen(x->limb, x->n);
    return (uint64_t)((u128)bits * 30103 / 100000) + 3;
}
int32_t tr_bigint_to_decimal(const tr_bigint *x, char *out, uint64_t capacity, uint64_t *length,
                             const tr_math_budget *b, tr_math_refusal *r) {
    int32_t rc = want_bi(x);
    if (rc) return rc;
    if (!out || capacity < tr_bigint_decimal_size(x)) return TR_MATH_INVALID;
    if (!x->n) { out[0] = '0'; out[1] = 0; if (length) *length = 1; return TR_MATH_OK; }
    rc = check_limbs(b, x->n, r);
    if (rc) return rc;
    uint64_t *t = (uint64_t *)malloc((size_t)x->n * sizeof *t), n = x->n;
    if (!t) return TR_MATH_MEMORY;
    memcpy(t, x->limb, (size_t)n * 8);
    uint64_t v = (uint64_t)((((u128)~TEN19) << 64 | ~(uint64_t)0) / TEN19), pos = capacity - 1;
    out[pos] = 0;
    while (n) {
        uint64_t rem = mpn_divrem_1_norm(t, t, n, TEN19, v);
        n = mpn_norm(t, n);
        if (n) for (int k = 0; k < 19; k++) { out[--pos] = (char)('0' + rem % 10); rem /= 10; }
        else do { out[--pos] = (char)('0' + rem % 10); rem /= 10; } while (rem);
    }
    free(t);
    if (x->sign < 0) out[--pos] = '-';
    uint64_t len = capacity - 1 - pos;
    memmove(out, out + pos, (size_t)len + 1);
    if (length) *length = len;
    return TR_MATH_OK;
}
tr_bigint *tr_bigint_from_decimal(const char *text, uint64_t length, const tr_math_budget *b, tr_math_refusal *r,
                                  int32_t *err) {
    if (!text || !length) { seterr(err, TR_MATH_INVALID); return NULL; }
    uint64_t i = 0;
    int sign = 1;
    if (text[0] == '+' || text[0] == '-') { sign = text[0] == '-' ? -1 : 1; i = 1; }
    if (i == length) { seterr(err, TR_MATH_INVALID); return NULL; }
    for (uint64_t k = i; k < length; k++)
        if (text[k] < '0' || text[k] > '9') { seterr(err, TR_MATH_INVALID); return NULL; }
    uint64_t digits = length - i;
    uint64_t cap = tr_budget_add(tr_budget_add(tr_budget_mul(digits, 3322), 63999) / 64000, 1);
    tr_bigint *x = bi_new(cap, b, r, err);
    if (!x) return NULL;
    uint64_t n = 0, first = digits % 19 ? digits % 19 : 19;
    static const uint64_t P10[20] = {1ULL, 10ULL, 100ULL, 1000ULL, 10000ULL, 100000ULL, 1000000ULL, 10000000ULL,
        100000000ULL, 1000000000ULL, 10000000000ULL, 100000000000ULL, 1000000000000ULL, 10000000000000ULL,
        100000000000000ULL, 1000000000000000ULL, 10000000000000000ULL, 100000000000000000ULL,
        1000000000000000000ULL, 10000000000000000000ULL};
    for (uint64_t k = i; k < length;) {
        uint64_t take = k == i ? first : 19, chunk = 0;
        for (uint64_t j = 0; j < take; j++) chunk = chunk * 10 + (uint64_t)(text[k + j] - '0');
        uint64_t carry = mpn_mul1_add(x->limb, n, P10[take], chunk);
        if (carry) x->limb[n++] = carry;
        k += take;
    }
    return bi_done(x, n, sign, err);
}

/* ================================================================ big rationals */
static tr_bigq *bq_wrap(tr_bigint *num, tr_bigint *den, int32_t *err) {
    tr_bigq *q = num && den ? (tr_bigq *)malloc(sizeof *q) : NULL;
    if (!q) {
        int32_t e = (num && den) ? TR_MATH_MEMORY : (err ? *err : TR_MATH_MEMORY);
        tr_bigint_free(num); tr_bigint_free(den);
        seterr(err, e ? e : TR_MATH_MEMORY);
        return NULL;
    }
    q->magic = BIGQ_MAGIC; q->reserved = 0; q->num = num; q->den = den;
    seterr(err, TR_MATH_OK);
    return q;
}
void tr_bigq_free(tr_bigq *q) {
    if (!is_bq(q)) return;
    q->magic = 0;
    tr_bigint_free(q->num); tr_bigint_free(q->den);
    free(q);
}
int32_t tr_bigq_is_handle(const void *p) { return is_bq((const tr_bigq *)p); }
static tr_bigq *bq_int(int64_t v, const tr_math_budget *b, tr_math_refusal *r, int32_t *err) {
    tr_bigint *n = tr_bigint_from_i64(v, b, r, err), *d = n ? bi_u64(1, 1, b, r, err) : NULL;
    return bq_wrap(n, d, err);
}
/* n / d for d != 0, canonicalized (copies) */
static tr_bigq *bq_make(const tr_bigint *n, const tr_bigint *d, const tr_math_budget *b, tr_math_refusal *r, int32_t *err) {
    tr_bigint *g = tr_bigint_gcd(n, d, b, r, err);
    if (!g) return NULL;
    tr_bigint *nn, *dd;
    if (g->n == 1 && g->limb[0] == 1) {
        nn = tr_bigint_copy(n, b, r, err);
        dd = nn ? tr_bigint_copy(d, b, r, err) : NULL;
    } else {
        nn = tr_bigint_divexact(n, g, b, r, err);
        dd = nn ? tr_bigint_divexact(d, g, b, r, err) : NULL;
    }
    tr_bigint_free(g);
    if (nn && dd && dd->sign < 0) { nn->sign = -nn->sign; dd->sign = 1; }
    return bq_wrap(nn, dd, err);
}
tr_bigq *tr_bigq_from_bigints(const tr_bigint *n, const tr_bigint *d, const tr_math_budget *b, tr_math_refusal *r,
                              int32_t *err) {
    int32_t rc = want_bi(n);
    if (!rc) rc = want_bi(d);
    if (rc) { seterr(err, rc); return NULL; }
    if (!d->n) { seterr(err, TR_MATH_NO_SOLUTION); return NULL; }
    return bq_make(n, d, b, r, err);
}
tr_bigq *tr_bigq_from_bigint(const tr_bigint *x, const tr_math_budget *b, tr_math_refusal *r, int32_t *err) {
    int32_t rc = want_bi(x);
    if (rc) { seterr(err, rc); return NULL; }
    tr_bigint *n = tr_bigint_copy(x, b, r, err), *d = n ? bi_u64(1, 1, b, r, err) : NULL;
    return bq_wrap(n, d, err);
}
tr_bigq *tr_bigq_from_q64(tr_num_rational q, const tr_math_budget *b, tr_math_refusal *r, int32_t *err) {
    if (q.denominator == 0) { seterr(err, TR_MATH_INVALID); return NULL; }
    uint64_t an = tr_abs_u64(q.numerator), ad = tr_abs_u64(q.denominator), g = tr_gcd_u64(an, ad);
    if (g > 1) { an /= g; ad /= g; }
    if (!an) ad = 1;
    int sign = (q.numerator < 0) != (q.denominator < 0) ? -1 : 1;
    tr_bigint *n = bi_u64(an, sign, b, r, err), *d = n ? bi_u64(ad, 1, b, r, err) : NULL;
    return bq_wrap(n, d, err);
}
tr_bigq *tr_bigq_from_double(double x, const tr_math_budget *b, tr_math_refusal *r, int32_t *err) {
    if (x != x || x == INFINITY || x == -INFINITY) { seterr(err, TR_MATH_INVALID); return NULL; }
    if (x == 0) return bq_int(0, b, r, err);
    union { double d; uint64_t u; } u;
    u.d = x;
    int sign = (u.u >> 63) ? -1 : 1;
    int64_t ex = (int64_t)((u.u >> 52) & 0x7ff);
    uint64_t m = u.u & (((uint64_t)1 << 52) - 1);
    if (ex) { m |= (uint64_t)1 << 52; ex -= 1075; } else ex = -1074;
    int tz = __builtin_ctzll(m);
    m >>= tz; ex += tz;
    tr_bigint *one = bi_u64(m, sign, b, r, err), *n = NULL, *d = NULL;
    if (!one) return NULL;
    if (ex >= 0) { n = tr_bigint_shl(one, (uint64_t)ex, b, r, err); d = n ? bi_u64(1, 1, b, r, err) : NULL; }
    else {
        tr_bigint *u1 = bi_u64(1, 1, b, r, err);
        n = u1 ? tr_bigint_copy(one, b, r, err) : NULL;
        d = n ? tr_bigint_shl(u1, (uint64_t)(-ex), b, r, err) : NULL;
        tr_bigint_free(u1);
    }
    tr_bigint_free(one);
    if (!n || !d) { tr_bigint_free(n); tr_bigint_free(d); return NULL; }
    return bq_wrap(n, d, err);
}
int32_t tr_bigq_to_q64(const tr_bigq *q, tr_num_rational *out) {
    int32_t rc = want_bq(q);
    if (rc || !out) return rc ? rc : TR_MATH_INVALID;
    int64_t n, d;
    if (tr_bigint_to_i64(q->num, &n) || tr_bigint_to_i64(q->den, &d)) return TR_MATH_OVERFLOW;
    out->numerator = n; out->denominator = d;
    return TR_MATH_OK;
}
tr_bigint *tr_bigq_to_bigint(const tr_bigq *q, const tr_math_budget *b, tr_math_refusal *r, int32_t *err) {
    int32_t rc = want_bq(q);
    if (rc) { seterr(err, rc); return NULL; }
    if (!(q->den->n == 1 && q->den->limb[0] == 1)) { seterr(err, TR_MATH_NO_SOLUTION); return NULL; }
    return tr_bigint_copy(q->num, b, r, err);
}
const tr_bigint *tr_bigq_numerator(const tr_bigq *q) { return is_bq(q) ? q->num : NULL; }
const tr_bigint *tr_bigq_denominator(const tr_bigq *q) { return is_bq(q) ? q->den : NULL; }
int32_t tr_bigq_sign(const tr_bigq *q) { int32_t rc = want_bq(q); return rc ? rc : q->num->sign; }
static int bq_is_one(const tr_bigint *x) { return x->n == 1 && x->limb[0] == 1 && x->sign > 0; }
/* exact comparison; products of the inputs' sizes are scratch (bounded by the inputs, no budget) */
static int32_t bq_cmp(const tr_bigq *a, const tr_bigq *c) {
    if (a->num->sign != c->num->sign) return a->num->sign < c->num->sign ? -1 : 1;
    if (!a->num->sign) return 0;
    if (mpn_cmp(a->den->limb, a->den->n, c->den->limb, c->den->n) == 0) return bi_cmp(a->num, c->num);
    tr_math_budget wide;
    tr_budget_default(&wide);
    wide.limit[TR_BUDGET_BYTES] = TR_BUDGET_UNKNOWN;
    int32_t e1 = 0, e2 = 0;
    tr_bigint *l = tr_bigint_mul(a->num, c->den, &wide, NULL, &e1), *rr = l ? tr_bigint_mul(c->num, a->den, &wide, NULL, &e2) : NULL;
    if (!l || !rr) { tr_bigint_free(l); tr_bigint_free(rr); return TR_MATH_MEMORY; }
    int res = bi_cmp(l, rr);
    tr_bigint_free(l); tr_bigint_free(rr);
    return res;
}
int32_t tr_bigq_cmp(const tr_bigq *a, const tr_bigq *c) {
    int32_t rc = want_bq(a);
    if (!rc) rc = want_bq(c);
    return rc ? rc : bq_cmp(a, c);
}
int32_t tr_bigq_cmp_double(const tr_bigq *q, double x, int32_t *out) {
    int32_t rc = want_bq(q);
    if (rc || !out || x != x) return rc ? rc : TR_MATH_INVALID;
    if (x == INFINITY) { *out = -1; return TR_MATH_OK; }
    if (x == -INFINITY) { *out = 1; return TR_MATH_OK; }
    tr_math_budget wide;
    tr_budget_default(&wide);
    wide.limit[TR_BUDGET_BYTES] = TR_BUDGET_UNKNOWN;
    int32_t e = 0;
    tr_bigq *y = tr_bigq_from_double(x, &wide, NULL, &e);
    if (!y) return e;
    int32_t c = bq_cmp(q, y);
    tr_bigq_free(y);
    if (c < -1 || c > 1) return c;
    *out = c;
    return TR_MATH_OK;
}
int32_t tr_bigq_in_interval(const tr_bigq *q, tr_math_interval iv) {
    int32_t rc = want_bq(q);
    if (rc) return rc;
    if (!tr_interval_valid(iv)) return TR_MATH_INVALID;
    int32_t lo, hi;
    rc = tr_bigq_cmp_double(q, iv.lo, &lo);
    if (!rc) rc = tr_bigq_cmp_double(q, iv.hi, &hi);
    if (rc) return rc;
    return lo >= 0 && hi <= 0;
}
tr_bigq *tr_bigq_copy(const tr_bigq *q, const tr_math_budget *b, tr_math_refusal *r, int32_t *err) {
    int32_t rc = want_bq(q);
    if (rc) { seterr(err, rc); return NULL; }
    tr_bigint *n = tr_bigint_copy(q->num, b, r, err), *d = n ? tr_bigint_copy(q->den, b, r, err) : NULL;
    return bq_wrap(n, d, err);
}
tr_bigq *tr_bigq_neg(const tr_bigq *q, const tr_math_budget *b, tr_math_refusal *r, int32_t *err) {
    int32_t rc = want_bq(q);
    if (rc) { seterr(err, rc); return NULL; }
    tr_bigint *n = tr_bigint_neg(q->num, b, r, err), *d = n ? tr_bigint_copy(q->den, b, r, err) : NULL;
    return bq_wrap(n, d, err);
}
tr_bigq *tr_bigq_inv(const tr_bigq *q, const tr_math_budget *b, tr_math_refusal *r, int32_t *err) {
    int32_t rc = want_bq(q);
    if (rc) { seterr(err, rc); return NULL; }
    if (!q->num->sign) { seterr(err, TR_MATH_NO_SOLUTION); return NULL; }
    tr_bigint *n = bi_signed_copy(q->den, q->num->sign, b, r, err), *d = n ? tr_bigint_abs(q->num, b, r, err) : NULL;
    return bq_wrap(n, d, err);
}
/* Henrici: a/b +- c/d with g = gcd(b, d); canonical without a gcd of the full products */
static tr_bigq *bq_addsub(const tr_bigq *x, const tr_bigq *y, int sub, const tr_math_budget *b, tr_math_refusal *r,
                          int32_t *err) {
    int32_t rc = want_bq(x);
    if (!rc) rc = want_bq(y);
    if (rc) { seterr(err, rc); return NULL; }
    tr_bigint *t[8] = {0}, *num = NULL, *den = NULL;
    tr_bigint *(*op)(const tr_bigint *, const tr_bigint *, const tr_math_budget *, tr_math_refusal *, int32_t *) =
        sub ? tr_bigint_sub : tr_bigint_add;
    if (bq_is_one(x->den) && bq_is_one(y->den)) {
        num = op(x->num, y->num, b, r, err);
        den = num ? bi_u64(1, 1, b, r, err) : NULL;
        return bq_wrap(num, den, err);
    }
    t[0] = tr_bigint_gcd(x->den, y->den, b, r, err);
    if (!t[0]) goto fail;
    if (bq_is_one(t[0])) {
        if (!(t[1] = tr_bigint_mul(x->num, y->den, b, r, err))) goto fail;
        if (!(t[2] = tr_bigint_mul(y->num, x->den, b, r, err))) goto fail;
        if (!(num = op(t[1], t[2], b, r, err))) goto fail;
        if (!(den = tr_bigint_mul(x->den, y->den, b, r, err))) goto fail;
    } else {
        if (!(t[1] = tr_bigint_divexact(x->den, t[0], b, r, err))) goto fail;      /* b' = b / g */
        if (!(t[2] = tr_bigint_divexact(y->den, t[0], b, r, err))) goto fail;      /* d' = d / g */
        if (!(t[3] = tr_bigint_mul(x->num, t[2], b, r, err))) goto fail;
        if (!(t[4] = tr_bigint_mul(y->num, t[1], b, r, err))) goto fail;
        if (!(t[5] = op(t[3], t[4], b, r, err))) goto fail;                        /* t = a d' +- c b' */
        if (!(t[6] = tr_bigint_gcd(t[5], t[0], b, r, err))) goto fail;             /* g2 = gcd(t, g) */
        if (!(num = tr_bigint_divexact(t[5], t[6], b, r, err))) goto fail;
        if (!(t[7] = tr_bigint_divexact(y->den, t[6], b, r, err))) goto fail;
        if (!(den = tr_bigint_mul(t[1], t[7], b, r, err))) goto fail;              /* b' (d / g2) */
    }
    for (int i = 0; i < 8; i++) tr_bigint_free(t[i]);
    if (!num->sign) { tr_bigint_free(den); den = bi_u64(1, 1, b, r, err); }
    return bq_wrap(num, den, err);
fail:
    for (int i = 0; i < 8; i++) tr_bigint_free(t[i]);
    tr_bigint_free(num); tr_bigint_free(den);
    return NULL;
}
tr_bigq *tr_bigq_add(const tr_bigq *x, const tr_bigq *y, const tr_math_budget *b, tr_math_refusal *r, int32_t *err) {
    return bq_addsub(x, y, 0, b, r, err);
}
tr_bigq *tr_bigq_sub(const tr_bigq *x, const tr_bigq *y, const tr_math_budget *b, tr_math_refusal *r, int32_t *err) {
    return bq_addsub(x, y, 1, b, r, err);
}
/* (a/b)(c/d) = (a/g1)(c/g2) / ((b/g2)(d/g1)), g1 = gcd(a, d), g2 = gcd(c, b) */
static tr_bigq *bq_mul_parts(const tr_bigint *a, const tr_bigint *bb, const tr_bigint *c, const tr_bigint *d,
                             const tr_math_budget *b, tr_math_refusal *r, int32_t *err) {
    if (!a->sign || !c->sign) return bq_int(0, b, r, err);
    tr_bigint *t[6] = {0}, *num = NULL, *den = NULL;
    if (!(t[0] = tr_bigint_gcd(a, d, b, r, err))) goto fail;
    if (!(t[1] = tr_bigint_gcd(c, bb, b, r, err))) goto fail;
    if (!(t[2] = tr_bigint_divexact(a, t[0], b, r, err))) goto fail;
    if (!(t[3] = tr_bigint_divexact(c, t[1], b, r, err))) goto fail;
    if (!(t[4] = tr_bigint_divexact(bb, t[1], b, r, err))) goto fail;
    if (!(t[5] = tr_bigint_divexact(d, t[0], b, r, err))) goto fail;
    if (!(num = tr_bigint_mul(t[2], t[3], b, r, err))) goto fail;
    if (!(den = tr_bigint_mul(t[4], t[5], b, r, err))) goto fail;
    for (int i = 0; i < 6; i++) tr_bigint_free(t[i]);
    if (den->sign < 0) { den->sign = 1; num->sign = -num->sign; }
    return bq_wrap(num, den, err);
fail:
    for (int i = 0; i < 6; i++) tr_bigint_free(t[i]);
    tr_bigint_free(num); tr_bigint_free(den);
    return NULL;
}
tr_bigq *tr_bigq_mul(const tr_bigq *x, const tr_bigq *y, const tr_math_budget *b, tr_math_refusal *r, int32_t *err) {
    int32_t rc = want_bq(x);
    if (!rc) rc = want_bq(y);
    if (rc) { seterr(err, rc); return NULL; }
    return bq_mul_parts(x->num, x->den, y->num, y->den, b, r, err);
}
tr_bigq *tr_bigq_div(const tr_bigq *x, const tr_bigq *y, const tr_math_budget *b, tr_math_refusal *r, int32_t *err) {
    int32_t rc = want_bq(x);
    if (!rc) rc = want_bq(y);
    if (rc) { seterr(err, rc); return NULL; }
    if (!y->num->sign) { seterr(err, TR_MATH_NO_SOLUTION); return NULL; }
    /* x / y = (a/b)(d/c): the denominator c may be negative; bq_mul_parts moves the sign to the numerator */
    return bq_mul_parts(x->num, x->den, y->den, y->num, b, r, err);
}
tr_bigq *tr_bigq_pow(const tr_bigq *q, int64_t e, const tr_math_budget *b, tr_math_refusal *r, int32_t *err) {
    int32_t rc = want_bq(q);
    if (rc) { seterr(err, rc); return NULL; }
    if (e < 0 && !q->num->sign) { seterr(err, TR_MATH_NO_SOLUTION); return NULL; }
    uint64_t m = tr_abs_u64(e);
    tr_bigint *n = tr_bigint_pow(e < 0 ? q->den : q->num, m, b, r, err);
    tr_bigint *d = n ? tr_bigint_pow(e < 0 ? q->num : q->den, m, b, r, err) : NULL;
    if (n && d && d->sign < 0) { d->sign = 1; n->sign = -n->sign; }
    return bq_wrap(n, d, err);
}

/* ---------------------------------------------------------------- tight enclosures of exact values */
/* [RD(n / d), RU(n / d)] for magnitudes n (nn limbs), d (nd >= 1 limbs) and a sign. With 2^E <= q < 2^(E+1):
 * s = 52 - E (normal range) or 1074 (subnormal range), t = floor(n 2^s / d) < 2^53, lo = t 2^-s and hi = lo when the
 * division is exact, else (t + 1) 2^-s; both are exact doubles (ldexp of an integer below 2^53 into range). Past the
 * double range: [DBL_MAX, inf]; below 2^-1099: [0, 2^-1074]. Scratch is bounded by the inputs (no budget). */
static int32_t enclose_mpn(const uint64_t *n, uint64_t nn, const uint64_t *d, uint64_t nd, int sign, tr_math_interval *out) {
    nn = mpn_norm(n, nn); nd = mpn_norm(d, nd);
    if (!nd) return TR_MATH_INVALID;
    double lo, hi;
    int64_t e0 = nn ? (int64_t)mpn_bitlen(n, nn) - (int64_t)mpn_bitlen(d, nd) : 0;
    if (!nn) { lo = 0.0; hi = 0.0; }
    else if (e0 > 1025) { lo = DBL_MAX; hi = INFINITY; }
    else if (e0 < -1100) { lo = 0.0; hi = 4.9406564584124654e-324; }
    else {
        uint64_t S = nn + nd + 64;
        uint64_t *buf = (uint64_t *)malloc((size_t)(4 * S) * sizeof *buf);
        if (!buf) return TR_MATH_MEMORY;
        uint64_t *sh = buf, *un = buf + S, *vn = buf + 2 * S, *rm = buf + 3 * S, k;
        int ge;
        if (e0 >= 0) { k = mpn_norm(sh, mpn_lshift(sh, d, nd, (uint64_t)e0)); ge = mpn_cmp(n, nn, sh, k) >= 0; }
        else { k = mpn_norm(sh, mpn_lshift(sh, n, nn, (uint64_t)-e0)); ge = mpn_cmp(sh, k, d, nd) >= 0; }
        int64_t E = ge ? e0 : e0 - 1;
        if (E >= 1024) { lo = DBL_MAX; hi = INFINITY; }
        else {
            int64_t s = E >= -1022 ? 52 - E : 1074;
            const uint64_t *N = n, *D = d;
            uint64_t nN = nn, nD = nd, t;
            int inexact;
            if (s >= 0) { nN = mpn_norm(sh, mpn_lshift(sh, n, nn, (uint64_t)s)); N = sh; }
            else { nD = mpn_norm(sh, mpn_lshift(sh, d, nd, (uint64_t)-s)); D = sh; }
            /* N / D < 2^53, so nN <= nD + 1 and the quotient has at most two limbs (the second zero) */
            if (mpn_cmp(N, nN, D, nD) < 0) { t = 0; inexact = 1; }
            else if (nD == 1) {
                u128 x = nN == 2 ? ((u128)N[1] << 64) | N[0] : N[0];
                t = (uint64_t)(x / D[0]);
                inexact = x % D[0] != 0;
            } else {
                uint64_t q[2] = {0, 0};
                mpn_divrem(q, rm, N, nN, D, nD, un, vn);
                t = q[0];
                inexact = mpn_norm(rm, nD) != 0;
            }
            lo = ldexp((double)t, (int)-s);
            hi = inexact ? ldexp((double)(t + 1), (int)-s) : lo;
        }
        free(buf);
    }
    if (sign < 0) { out->lo = -hi; out->hi = -lo; } else { out->lo = lo; out->hi = hi; }
    return TR_MATH_OK;
}
int32_t tr_bigint_enclose(const tr_bigint *x, tr_math_interval *out) {
    int32_t rc = want_bi(x);
    if (rc || !out) return rc ? rc : TR_MATH_INVALID;
    uint64_t one = 1;
    return enclose_mpn(x->limb, x->n, &one, 1, x->sign, out);
}
int32_t tr_bigq_enclose(const tr_bigq *q, tr_math_interval *out) {
    int32_t rc = want_bq(q);
    if (rc || !out) return rc ? rc : TR_MATH_INVALID;
    return enclose_mpn(q->num->limb, q->num->n, q->den->limb, q->den->n, q->num->sign, out);
}
int32_t tr_carrier_enclose_i64(int64_t v, tr_math_interval *out) {
    if (!out) return TR_MATH_INVALID;
    return tr_interval_enclose_q64(v, 1, out) ? TR_MATH_INVALID : TR_MATH_OK;
}
int32_t tr_carrier_enclose_i128(tr_math_i128 v, tr_math_interval *out) {
    if (!out) return TR_MATH_INVALID;
    i128 x = tr_i128_get(v);
    u128 m = x < 0 ? (u128)0 - (u128)x : (u128)x;
    uint64_t limbs[2] = {(uint64_t)m, (uint64_t)(m >> 64)}, one = 1;
    return enclose_mpn(limbs, 2, &one, 1, x < 0 ? -1 : 1, out);
}
int32_t tr_carrier_enclose_q64(tr_num_rational q, tr_math_interval *out) {
    if (!out || q.denominator == 0) return TR_MATH_INVALID;
    tr_interval_enclose_q64(q.numerator, q.denominator, out);
    return TR_MATH_OK;
}
int32_t tr_carrier_enclose_q64_vec(const tr_num_rational *q, uint64_t n, tr_math_interval *out) {
    if (n && (!q || !out)) return TR_MATH_INVALID;
    for (uint64_t i = 0; i < n; i++) if (q[i].denominator == 0) return TR_MATH_INVALID;
    for (uint64_t i = 0; i < n; i++) tr_interval_enclose_q64(q[i].numerator, q[i].denominator, &out[i]);
    return TR_MATH_OK;
}

/* ================================================================ GF(p) */
static uint64_t mulmod(uint64_t a, uint64_t b, uint64_t m) { return (uint64_t)(((u128)a * b) % m); }
static uint64_t powmod(uint64_t a, uint64_t e, uint64_t m) {
    uint64_t r = 1 % m;
    a %= m;
    while (e) { if (e & 1) r = mulmod(r, a, m); a = mulmod(a, a, m); e >>= 1; }
    return r;
}
int32_t tr_gfp_is_prime(uint64_t n) {
    static const uint64_t small[] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37};
    if (n < 2) return 0;
    for (unsigned i = 0; i < sizeof small / sizeof small[0]; i++) {
        if (n == small[i]) return 1;
        if (n % small[i] == 0) return 0;
    }
    uint64_t d = n - 1;
    int s = 0;
    while (!(d & 1)) { d >>= 1; s++; }
    for (unsigned i = 0; i < sizeof small / sizeof small[0]; i++) {
        uint64_t x = powmod(small[i], d, n);
        if (x == 1 || x == n - 1) continue;
        int composite = 1;
        for (int k = 1; k < s; k++) {
            x = mulmod(x, x, n);
            if (x == n - 1) { composite = 0; break; }
        }
        if (composite) return 0;
    }
    return 1;
}
int32_t tr_gfp_field(uint64_t p, tr_gfp *out) {
    if (!out) return TR_MATH_INVALID;
    if (!tr_gfp_is_prime(p)) return TR_MATH_INVALID;
    if (p >= TR_GFP_MAX) return TR_MATH_UNSUPPORTED;
    tr_gfp_fill(p, out);
    return TR_MATH_OK;
}
static int field_ok(const tr_gfp *f) { return tr_gfp_consistent(f); }
static int elem_ok(tr_gfp_elem a) { return a.p >= 2 && a.p < TR_GFP_MAX && a.v < a.p; }
static int32_t elem_pair(tr_gfp_elem a, tr_gfp_elem b) {
    if (!elem_ok(a) || !elem_ok(b)) return TR_MATH_INVALID;
    return a.p == b.p ? TR_MATH_OK : TR_MATH_CARRIER_ERROR;
}
static uint64_t inv_mod(uint64_t a, uint64_t p) { tr_gfp f; f.p = p; return tr_gfp_inv_hd(&f, a); }
static int32_t make_elem(const tr_gfp *f, uint64_t v, tr_gfp_elem *out) {
    if (!out) return TR_MATH_INVALID;
    out->p = f->p; out->v = v;
    return TR_MATH_OK;
}
int32_t tr_gfp_from_u64(const tr_gfp *f, uint64_t x, tr_gfp_elem *out) {
    if (!field_ok(f)) return TR_MATH_INVALID;
    return make_elem(f, x % f->p, out);
}
int32_t tr_gfp_from_i64(const tr_gfp *f, int64_t x, tr_gfp_elem *out) {
    if (!field_ok(f)) return TR_MATH_INVALID;
    return make_elem(f, tr_gfp_reduce_i64_hd(f, x), out);
}
int32_t tr_gfp_from_i128(const tr_gfp *f, tr_math_i128 v, tr_gfp_elem *out) {
    if (!field_ok(f)) return TR_MATH_INVALID;
    i128 x = tr_i128_get(v);
    u128 m = x < 0 ? (u128)0 - (u128)x : (u128)x;
    uint64_t r = (uint64_t)(m % f->p);
    return make_elem(f, x < 0 && r ? f->p - r : r, out);
}
int32_t tr_gfp_from_q64(const tr_gfp *f, tr_num_rational q, tr_gfp_elem *out) {
    if (!field_ok(f) || q.denominator == 0) return TR_MATH_INVALID;
    uint64_t d = tr_gfp_reduce_i64_hd(f, q.denominator);
    if (!d) return TR_MATH_NO_SOLUTION;
    return make_elem(f, tr_gfp_mul_hd(f, tr_gfp_reduce_i64_hd(f, q.numerator), tr_gfp_inv_hd(f, d)), out);
}
int32_t tr_gfp_from_bigint(const tr_gfp *f, const tr_bigint *x, tr_gfp_elem *out) {
    if (!field_ok(f)) return TR_MATH_INVALID;
    uint64_t v;
    int32_t rc = tr_bigint_mod_u64(x, f->p, &v);
    return rc ? rc : make_elem(f, v, out);
}
int32_t tr_gfp_from_bigq(const tr_gfp *f, const tr_bigq *q, tr_gfp_elem *out) {
    int32_t rc = want_bq(q);
    if (rc) return rc;
    if (!field_ok(f)) return TR_MATH_INVALID;
    uint64_t n, d;
    tr_bigint_mod_u64(q->num, f->p, &n);
    tr_bigint_mod_u64(q->den, f->p, &d);
    if (!d) return TR_MATH_NO_SOLUTION;
    return make_elem(f, tr_gfp_mul_hd(f, n, tr_gfp_inv_hd(f, d)), out);
}
int32_t tr_gfp_add(tr_gfp_elem a, tr_gfp_elem b, tr_gfp_elem *out) {
    int32_t rc = elem_pair(a, b);
    if (rc || !out) return rc ? rc : TR_MATH_INVALID;
    uint64_t s = a.v + b.v;
    out->p = a.p; out->v = s >= a.p ? s - a.p : s;
    return TR_MATH_OK;
}
int32_t tr_gfp_sub(tr_gfp_elem a, tr_gfp_elem b, tr_gfp_elem *out) {
    int32_t rc = elem_pair(a, b);
    if (rc || !out) return rc ? rc : TR_MATH_INVALID;
    out->p = a.p; out->v = a.v >= b.v ? a.v - b.v : a.v + (a.p - b.v);
    return TR_MATH_OK;
}
int32_t tr_gfp_mul(tr_gfp_elem a, tr_gfp_elem b, tr_gfp_elem *out) {
    int32_t rc = elem_pair(a, b);
    if (rc || !out) return rc ? rc : TR_MATH_INVALID;
    out->p = a.p; out->v = mulmod(a.v, b.v, a.p);
    return TR_MATH_OK;
}
int32_t tr_gfp_div(tr_gfp_elem a, tr_gfp_elem b, tr_gfp_elem *out) {
    int32_t rc = elem_pair(a, b);
    if (rc || !out) return rc ? rc : TR_MATH_INVALID;
    if (!b.v) return TR_MATH_NO_SOLUTION;
    out->p = a.p; out->v = mulmod(a.v, inv_mod(b.v, a.p), a.p);
    return TR_MATH_OK;
}
int32_t tr_gfp_neg(tr_gfp_elem a, tr_gfp_elem *out) {
    if (!elem_ok(a) || !out) return TR_MATH_INVALID;
    out->p = a.p; out->v = a.v ? a.p - a.v : 0;
    return TR_MATH_OK;
}
int32_t tr_gfp_inv(tr_gfp_elem a, tr_gfp_elem *out) {
    if (!elem_ok(a) || !out) return TR_MATH_INVALID;
    if (!a.v) return TR_MATH_NO_SOLUTION;
    out->p = a.p; out->v = inv_mod(a.v, a.p);
    return TR_MATH_OK;
}
int32_t tr_gfp_pow(tr_gfp_elem a, int64_t e, tr_gfp_elem *out) {
    if (!elem_ok(a) || !out) return TR_MATH_INVALID;
    uint64_t base = a.v;
    if (e < 0) {
        if (!a.v) return TR_MATH_NO_SOLUTION;
        base = inv_mod(a.v, a.p);
    }
    out->p = a.p; out->v = powmod(base, tr_abs_u64(e), a.p);
    return TR_MATH_OK;
}
int32_t tr_gfp_equal(tr_gfp_elem a, tr_gfp_elem b) {
    int32_t rc = elem_pair(a, b);
    return rc ? rc : a.v == b.v;
}
int32_t tr_gfp_lift_i64(tr_gfp_elem a, int32_t symmetric, int64_t *out) {
    if (!elem_ok(a) || !out) return TR_MATH_INVALID;
    *out = symmetric && a.v > a.p / 2 ? (int64_t)a.v - (int64_t)a.p : (int64_t)a.v;
    return TR_MATH_OK;
}
tr_bigint *tr_gfp_lift(tr_gfp_elem a, int32_t symmetric, const tr_math_budget *b, tr_math_refusal *r, int32_t *err) {
    int64_t v;
    int32_t rc = tr_gfp_lift_i64(a, symmetric, &v);
    if (rc) { seterr(err, rc); return NULL; }
    return tr_bigint_from_i64(v, b, r, err);
}
int32_t tr_gfp_vec(const tr_gfp *f, int32_t op, const uint64_t *a, const uint64_t *b, uint64_t n, uint64_t *out) {
    if (!field_ok(f)) return TR_MATH_INVALID;
    int unary = op == TR_OP_NEG || op == TR_OP_INV;
    if (!unary && op != TR_OP_ADD && op != TR_OP_SUB && op != TR_OP_MUL && op != TR_OP_DIV) return TR_MATH_INVALID;
    if (n && (!a || !out || (!unary && !b))) return TR_MATH_INVALID;
    uint64_t p = f->p;
    for (uint64_t i = 0; i < n; i++) if (a[i] >= p || (!unary && b[i] >= p)) return TR_MATH_INVALID;
    if (op == TR_OP_DIV) for (uint64_t i = 0; i < n; i++) if (!b[i]) return TR_MATH_NO_SOLUTION;
    if (op == TR_OP_INV) for (uint64_t i = 0; i < n; i++) if (!a[i]) return TR_MATH_NO_SOLUTION;
    for (uint64_t i = 0; i < n; i++) {
        uint64_t x = a[i], y = unary ? 0 : b[i], v;
        switch (op) {
        case TR_OP_ADD: v = tr_gfp_add_hd(f, x, y); break;
        case TR_OP_SUB: v = tr_gfp_sub_hd(f, x, y); break;
        case TR_OP_MUL: v = tr_gfp_mul_hd(f, x, y); break;
        case TR_OP_DIV: v = tr_gfp_mul_hd(f, x, tr_gfp_inv_hd(f, y)); break;
        case TR_OP_NEG: v = tr_gfp_neg_hd(f, x); break;
        default: v = tr_gfp_inv_hd(f, x); break;
        }
        out[i] = v;
    }
    return TR_MATH_OK;
}
int32_t tr_gfp_dot(const tr_gfp *f, const uint64_t *a, const uint64_t *b, uint64_t n, uint64_t *out) {
    if (!field_ok(f) || !out || (n && (!a || !b))) return TR_MATH_INVALID;
    for (uint64_t i = 0; i < n; i++) if (a[i] >= f->p || b[i] >= f->p) return TR_MATH_INVALID;
    uint64_t s = 0;
    for (uint64_t i = 0; i < n; i++) s = tr_gfp_add_hd(f, s, tr_gfp_mul_hd(f, a[i], b[i]));
    *out = s;
    return TR_MATH_OK;
}
int32_t tr_gfp_matvec(const tr_gfp *f, const uint64_t *A, uint64_t rows, uint64_t cols, const uint64_t *x, uint64_t *y) {
    if (!field_ok(f) || (rows && !y) || (rows && cols && (!A || !x))) return TR_MATH_INVALID;
    uint64_t cells;
    if (__builtin_mul_overflow(rows, cols, &cells)) return TR_MATH_INVALID;
    for (uint64_t i = 0; i < cells; i++) if (A[i] >= f->p) return TR_MATH_INVALID;
    for (uint64_t j = 0; j < cols && rows; j++) if (x[j] >= f->p) return TR_MATH_INVALID;
    for (uint64_t i = 0; i < rows; i++) {
        uint64_t s = 0;
        const uint64_t *row = A + i * cols;
        for (uint64_t j = 0; j < cols; j++) s = tr_gfp_add_hd(f, s, tr_gfp_mul_hd(f, row[j], x[j]));
        y[i] = s;
    }
    return TR_MATH_OK;
}
int32_t tr_gfp_reduce_i64(const tr_gfp *f, const int64_t *x, uint64_t n, uint64_t *out) {
    if (!field_ok(f) || (n && (!x || !out))) return TR_MATH_INVALID;
    for (uint64_t i = 0; i < n; i++) out[i] = tr_gfp_reduce_i64_hd(f, x[i]);
    return TR_MATH_OK;
}

/* ---------------------------------------------------------------- CRT and rational reconstruction */
int32_t tr_carrier_crt(const uint64_t *res, const tr_gfp *fields, uint32_t k, int32_t symmetric, tr_bigint **xo,
                       tr_bigint **mo, const tr_math_budget *b, tr_math_refusal *r) {
    if (!k || !res || !fields) return TR_MATH_INVALID;
    for (uint32_t i = 0; i < k; i++) {
        if (!field_ok(&fields[i]) || !tr_gfp_is_prime(fields[i].p) || res[i] >= fields[i].p) return TR_MATH_INVALID;
        for (uint32_t j = 0; j < i; j++) if (fields[j].p == fields[i].p) return TR_MATH_INVALID;
    }
    uint64_t cap = (uint64_t)k + 1;
    int32_t rc = check_limbs(b, cap, r), err = 0;
    if (rc) return rc;
    uint64_t *v = (uint64_t *)malloc((size_t)k * sizeof *v);
    if (!v) return TR_MATH_MEMORY;
    v[0] = res[0];
    for (uint32_t i = 1; i < k; i++) {
        const tr_gfp *f = &fields[i];
        uint64_t acc = 0, prod = 1;
        for (uint32_t j = 0; j < i; j++) {
            acc = tr_gfp_add_hd(f, acc, tr_gfp_mul_hd(f, v[j] % f->p, prod));
            prod = tr_gfp_mul_hd(f, prod, fields[j].p % f->p);
        }
        v[i] = tr_gfp_mul_hd(f, tr_gfp_sub_hd(f, res[i], acc), tr_gfp_inv_hd(f, prod));
    }
    tr_bigint *x = bi_new(cap, b, r, &err), *m = x ? bi_new(cap, b, r, &err) : NULL;
    if (!x || !m) { free(v); tr_bigint_free(x); tr_bigint_free(m); return err; }
    /* x = v0 + p0 (v1 + p1 (v2 + ...)) and M = p0 p1 ... */
    uint64_t nx = 1, nm = 1;
    x->limb[0] = v[k - 1];
    for (uint32_t i = k - 1; i-- > 0;) {
        uint64_t c = mpn_mul1_add(x->limb, nx, fields[i].p, v[i]);
        if (c) x->limb[nx++] = c;
    }
    m->limb[0] = 1;
    for (uint32_t i = 0; i < k; i++) {
        uint64_t c = mpn_mul1_add(m->limb, nm, fields[i].p, 0);
        if (c) m->limb[nm++] = c;
    }
    free(v);
    nx = mpn_norm(x->limb, nx);
    int sign = 1;
    if (symmetric && nx) {
        /* x > M / 2 exactly when 2 x > M: then x - M, of magnitude M - x */
        uint64_t *two = (uint64_t *)malloc((size_t)(nx + 2) * sizeof *two);
        if (!two) { tr_bigint_free(x); tr_bigint_free(m); return TR_MATH_MEMORY; }
        uint64_t n2 = mpn_norm(two, mpn_lshift(two, x->limb, nx, 1));
        if (mpn_cmp(two, n2, m->limb, nm) > 0) {
            mpn_sub(two, m->limb, nm, x->limb, nx);
            memcpy(x->limb, two, (size_t)nm * 8);
            nx = nm;
            sign = -1;
        }
        free(two);
    }
    bi_done(x, nx, sign, &err);
    bi_done(m, nm, 1, &err);
    if (xo) *xo = x; else tr_bigint_free(x);
    if (mo) *mo = m; else tr_bigint_free(m);
    return TR_MATH_OK;
}
int32_t tr_carrier_crt_combine(const tr_bigint *x, const tr_bigint *m, uint64_t residue, const tr_gfp *f,
                               tr_bigint **xo, tr_bigint **mo, const tr_math_budget *b, tr_math_refusal *r) {
    int32_t rc = want_bi(x);
    if (!rc) rc = want_bi(m);
    if (rc) return rc;
    if (!field_ok(f) || !tr_gfp_is_prime(f->p) || residue >= f->p || m->sign <= 0 || x->sign < 0 ||
        mpn_cmp(x->limb, x->n, m->limb, m->n) >= 0) return TR_MATH_INVALID;
    uint64_t xm, mm;
    tr_bigint_mod_u64(x, f->p, &xm);
    tr_bigint_mod_u64(m, f->p, &mm);
    if (!mm) return TR_MATH_INVALID;
    uint64_t t = tr_gfp_mul_hd(f, tr_gfp_sub_hd(f, residue, xm), tr_gfp_inv_hd(f, mm));
    int32_t err = 0;
    uint64_t cap = m->n + 1;
    rc = check_limbs(b, cap, r);
    if (rc) return rc;
    tr_bigint *nx = bi_new(cap, b, r, &err), *nm = nx ? bi_new(cap, b, r, &err) : NULL;
    if (!nx || !nm) { tr_bigint_free(nx); tr_bigint_free(nm); return err; }
    /* x' = x + m t (< m p), m' = m p */
    memcpy(nm->limb, m->limb, (size_t)m->n * 8);
    nm->limb[m->n] = mpn_mul1_add(nm->limb, m->n, f->p, 0);
    memcpy(nx->limb, m->limb, (size_t)m->n * 8);
    nx->limb[m->n] = mpn_mul1_add(nx->limb, m->n, t, 0);
    uint64_t n = mpn_norm(nx->limb, m->n + 1);
    if (x->n) {
        uint64_t *s = (uint64_t *)malloc((size_t)(cap + 1) * sizeof *s);
        if (!s) { tr_bigint_free(nx); tr_bigint_free(nm); return TR_MATH_MEMORY; }
        uint64_t ns = n >= x->n ? mpn_add(s, nx->limb, n, x->limb, x->n) : mpn_add(s, x->limb, x->n, nx->limb, n);
        ns = mpn_norm(s, ns);
        memcpy(nx->limb, s, (size_t)ns * 8);
        n = ns;
        free(s);
    }
    bi_done(nx, n, 1, &err);
    bi_done(nm, m->n + 1, 1, &err);
    if (xo) *xo = nx; else tr_bigint_free(nx);
    if (mo) *mo = nm; else tr_bigint_free(nm);
    return TR_MATH_OK;
}
tr_bigq *tr_carrier_ratrecon(const tr_bigint *u, const tr_bigint *m, const tr_bigint *N, const tr_bigint *D,
                             const tr_math_budget *b, tr_math_refusal *r, int32_t *err) {
    int32_t rc = want_bi(u);
    if (!rc) rc = want_bi(m);
    if (!rc && N) rc = want_bi(N);
    if (!rc && D) rc = want_bi(D);
    if (rc) { seterr(err, rc); return NULL; }
    if (m->sign <= 0 || (m->n == 1 && m->limb[0] < 2) || (N && N->sign < 0) || (D && D->sign <= 0)) {
        seterr(err, TR_MATH_INVALID);
        return NULL;
    }
    tr_bigint *bound = NULL, *r0 = NULL, *r1 = NULL, *t0 = NULL, *t1 = NULL;
    tr_bigq *out = NULL;
    if (!N || !D) {   /* isqrt((m - 1) / 2) */
        tr_bigint *one = bi_u64(1, 1, b, r, err), *m1 = one ? tr_bigint_sub(m, one, b, r, err) : NULL;
        tr_bigint *h = m1 ? tr_bigint_shr(m1, 1, b, r, err) : NULL;
        bound = h ? tr_bigint_isqrt(h, b, r, err) : NULL;
        tr_bigint_free(one); tr_bigint_free(m1); tr_bigint_free(h);
        if (!bound) return NULL;
    }
    const tr_bigint *NN = N ? N : bound, *DD = D ? D : bound;
    /* the extended Euclidean algorithm on (m, u mod m), keeping r_i = t_i u (mod m), until r_i <= N */
    rc = tr_bigint_divmod(u, m, TR_DIV_FLOOR, NULL, &r1, b, r);
    if (rc) { seterr(err, rc); goto done; }
    if (!(r0 = tr_bigint_copy(m, b, r, err)) || !(t0 = tr_bigint_from_i64(0, b, r, err)) ||
        !(t1 = tr_bigint_from_i64(1, b, r, err))) goto done;
    while (bi_cmp(r1, NN) > 0) {
        tr_bigint *q = NULL, *rem = NULL, *qt, *nt;
        rc = tr_bigint_divmod(r0, r1, TR_DIV_FLOOR, &q, &rem, b, r);
        if (rc) { seterr(err, rc); goto done; }
        qt = tr_bigint_mul(q, t1, b, r, err);
        tr_bigint_free(q);
        nt = qt ? tr_bigint_sub(t0, qt, b, r, err) : NULL;
        tr_bigint_free(qt);
        if (!nt) { tr_bigint_free(rem); goto done; }
        tr_bigint_free(r0); r0 = r1; r1 = rem;
        tr_bigint_free(t0); t0 = t1; t1 = nt;
    }
    /* the candidate r1 / t1: a denominator within D, coprime to the numerator and to m */
    if (!t1->n || mpn_cmp(t1->limb, t1->n, DD->limb, DD->n) > 0) { seterr(err, TR_MATH_NO_SOLUTION); goto done; }
    {
        tr_bigint *g1 = tr_bigint_gcd(r1, t1, b, r, err), *g2 = g1 ? tr_bigint_gcd(t1, m, b, r, err) : NULL;
        int failed = !g1 || !g2, ok = !failed && bq_is_one(g1) && bq_is_one(g2);
        tr_bigint_free(g1); tr_bigint_free(g2);
        if (failed) goto done;
        if (!ok) { seterr(err, TR_MATH_NO_SOLUTION); goto done; }
        tr_bigint *n = bi_signed_copy(r1, t1->sign, b, r, err), *d = n ? tr_bigint_abs(t1, b, r, err) : NULL;
        out = bq_wrap(n, d, err);
    }
done:
    tr_bigint_free(bound); tr_bigint_free(r0); tr_bigint_free(r1); tr_bigint_free(t0); tr_bigint_free(t1);
    return out;
}

/* ================================================================ matched expressions */
int32_t tr_carrier_dot_i64(const int64_t *a, const int64_t *b, uint64_t n, int64_t *out) {
    if (!out || (n && (!a || !b))) return TR_MATH_INVALID;
    int64_t s = 0, p;
    for (uint64_t i = 0; i < n; i++)
        if (!tr_mul_i64(a[i], b[i], &p) || !tr_add_i64(s, p, &s)) return TR_MATH_OVERFLOW;
    *out = s;
    return TR_MATH_OK;
}
int32_t tr_carrier_dot_i128(const int64_t *a, const int64_t *b, uint64_t n, tr_math_i128 *out) {
    if (!out || (n && (!a || !b))) return TR_MATH_INVALID;
    i128 s = 0;
    for (uint64_t i = 0; i < n; i++)
        if (__builtin_add_overflow(s, (i128)a[i] * b[i], &s)) return TR_MATH_OVERFLOW;
    *out = tr_i128_set(s);
    return TR_MATH_OK;
}
int32_t tr_carrier_dot_wide(const int64_t *a, const int64_t *b, uint64_t n, uint64_t out[3]) {
    if (!out || (n && (!a || !b))) return TR_MATH_INVALID;
    uint64_t w[3] = {0, 0, 0};
    for (uint64_t i = 0; i < n; i++) tr_wide_add128(w, (i128)a[i] * b[i]);
    out[0] = w[0]; out[1] = w[1]; out[2] = w[2];
    return TR_MATH_OK;
}
int32_t tr_carrier_horner_i64(const int64_t *c, uint32_t nc, const int64_t *x, uint64_t n, int64_t *out) {
    if (!nc || !c || (n && (!x || !out))) return TR_MATH_INVALID;
    for (uint64_t i = 0; i < n; i++) {
        int64_t acc = c[nc - 1];
        for (uint32_t j = nc - 1; j-- > 0;)
            if (!tr_mul_i64(acc, x[i], &acc) || !tr_add_i64(acc, c[j], &acc)) return TR_MATH_OVERFLOW;
        out[i] = acc;
    }
    return TR_MATH_OK;
}
int32_t tr_carrier_horner_i128(const int64_t *c, uint32_t nc, const int64_t *x, uint64_t n, tr_math_i128 *out) {
    if (!nc || !c || (n && (!x || !out))) return TR_MATH_INVALID;
    for (uint64_t i = 0; i < n; i++) {
        i128 acc = c[nc - 1];
        for (uint32_t j = nc - 1; j-- > 0;)
            if (__builtin_mul_overflow(acc, (i128)x[i], &acc) || __builtin_add_overflow(acc, (i128)c[j], &acc))
                return TR_MATH_OVERFLOW;
        out[i] = tr_i128_set(acc);
    }
    return TR_MATH_OK;
}
int32_t tr_carrier_sum_i64(const int64_t *x, uint64_t n, const uint32_t *groups, uint64_t ngroups, tr_math_i128 *out) {
    if (!out || !ngroups || (n && !x) || (!groups && ngroups != 1)) return TR_MATH_INVALID;
    if (groups) for (uint64_t i = 0; i < n; i++) if (groups[i] >= ngroups) return TR_MATH_INVALID;
    for (uint64_t g = 0; g < ngroups; g++) out[g] = tr_i128_set(0);
    for (uint64_t i = 0; i < n; i++) {
        uint64_t g = groups ? groups[i] : 0;
        out[g] = tr_i128_set(tr_i128_get(out[g]) + x[i]);
    }
    return TR_MATH_OK;
}
tr_bigint *tr_bigint_dot(const tr_bigint *const *a, const tr_bigint *const *c, uint64_t n, const tr_math_budget *b,
                         tr_math_refusal *r, int32_t *err) {
    if (n && (!a || !c)) { seterr(err, TR_MATH_INVALID); return NULL; }
    tr_bigint *s = tr_bigint_from_i64(0, b, r, err);
    for (uint64_t i = 0; s && i < n; i++) {
        tr_bigint *p = tr_bigint_mul(a[i], c[i], b, r, err), *t = p ? tr_bigint_add(s, p, b, r, err) : NULL;
        tr_bigint_free(p);
        tr_bigint_free(s);
        s = t;
    }
    return s;
}
tr_bigint *tr_bigint_horner(const tr_bigint *const *c, uint32_t nc, const tr_bigint *x, const tr_math_budget *b,
                            tr_math_refusal *r, int32_t *err) {
    if (!nc || !c) { seterr(err, TR_MATH_INVALID); return NULL; }
    tr_bigint *acc = tr_bigint_copy(c[nc - 1], b, r, err);
    for (uint32_t j = nc - 1; acc && j-- > 0;) {
        tr_bigint *p = tr_bigint_mul(acc, x, b, r, err), *t = p ? tr_bigint_add(p, c[j], b, r, err) : NULL;
        tr_bigint_free(p);
        tr_bigint_free(acc);
        acc = t;
    }
    return acc;
}
int32_t tr_bigint_batch(int32_t op, const tr_bigint *const *a, const tr_bigint *const *c, uint64_t n, tr_bigint **out,
                        const tr_math_budget *b, tr_math_refusal *r) {
    if (n && (!a || !c || !out)) return TR_MATH_INVALID;
    if (op != TR_OP_ADD && op != TR_OP_SUB && op != TR_OP_MUL && op != TR_OP_DIV && op != TR_OP_MOD && op != TR_OP_GCD)
        return TR_MATH_INVALID;
    tr_bigint **tmp = (tr_bigint **)calloc((size_t)(n ? n : 1), sizeof *tmp);
    if (!tmp) return TR_MATH_MEMORY;
    int32_t rc = TR_MATH_OK;
    for (uint64_t i = 0; i < n && !rc; i++) {
        int32_t e = TR_MATH_OK;
        switch (op) {
        case TR_OP_ADD: tmp[i] = tr_bigint_add(a[i], c[i], b, r, &e); break;
        case TR_OP_SUB: tmp[i] = tr_bigint_sub(a[i], c[i], b, r, &e); break;
        case TR_OP_MUL: tmp[i] = tr_bigint_mul(a[i], c[i], b, r, &e); break;
        case TR_OP_GCD: tmp[i] = tr_bigint_gcd(a[i], c[i], b, r, &e); break;
        case TR_OP_DIV: e = tr_bigint_divmod(a[i], c[i], TR_DIV_FLOOR, &tmp[i], NULL, b, r); break;
        default: e = tr_bigint_divmod(a[i], c[i], TR_DIV_FLOOR, NULL, &tmp[i], b, r); break;
        }
        if (e) rc = e;
    }
    if (rc) { for (uint64_t i = 0; i < n; i++) tr_bigint_free(tmp[i]); }
    else if (n) memcpy(out, tmp, (size_t)n * sizeof *out);
    free(tmp);
    return rc;
}

/* ================================================================ intervals and complex intervals */
int32_t tr_carrier_interval_op(int32_t op, const tr_math_interval *a, const tr_math_interval *b, int64_t n,
                               tr_math_interval *out) {
    if (!a || !out || !tr_interval_valid(*a)) return TR_MATH_INVALID;
    int binary = op == TR_OP_ADD || op == TR_OP_SUB || op == TR_OP_MUL || op == TR_OP_DIV || op == TR_OP_HULL ||
                 op == TR_OP_INTERSECT;
    if (binary && (!b || !tr_interval_valid(*b))) return TR_MATH_INVALID;
    tr_math_interval one = tr_interval_point(1.0), res;
    switch (op) {
    case TR_OP_ADD: res = tr_interval_add(*a, *b); break;
    case TR_OP_SUB: res = tr_interval_sub(*a, *b); break;
    case TR_OP_MUL: res = tr_interval_mul(*a, *b); break;
    case TR_OP_DIV: if (tr_iv_contains_zero(*b)) return TR_MATH_NO_SOLUTION; res = tr_interval_div(*a, *b); break;
    case TR_OP_NEG: res = tr_interval_neg(*a); break;
    case TR_OP_SQR: res = tr_interval_sqr(*a); break;
    case TR_OP_ABS: res = tr_interval_abs(*a); break;
    case TR_OP_SQRT: if (a->lo < 0) return TR_MATH_NO_SOLUTION; res = tr_interval_sqrt(*a); break;
    case TR_OP_INV: if (tr_iv_contains_zero(*a)) return TR_MATH_NO_SOLUTION; res = tr_interval_div(one, *a); break;
    case TR_OP_POW:
        if (n >= 0) { res = tr_interval_pown(*a, (uint64_t)n); break; }
        if (tr_iv_contains_zero(*a)) return TR_MATH_NO_SOLUTION;
        res = tr_interval_div(one, tr_interval_pown(*a, tr_abs_u64(n)));
        break;
    case TR_OP_HULL: res = tr_interval_hull(*a, *b); break;
    case TR_OP_INTERSECT:
        res.lo = a->lo > b->lo ? a->lo : b->lo; res.hi = a->hi < b->hi ? a->hi : b->hi;
        if (res.lo > res.hi) return TR_MATH_NO_SOLUTION;
        break;
    default: return TR_MATH_INVALID;
    }
    *out = res;
    return TR_MATH_OK;
}
int32_t tr_carrier_interval_cmp(const tr_math_interval *a, const tr_math_interval *b) {
    if (!a || !b || !tr_interval_valid(*a) || !tr_interval_valid(*b)) return TR_MATH_INVALID;
    if (a->hi < b->lo) return TR_CMP_LESS;
    if (a->lo > b->hi) return TR_CMP_GREATER;
    if (a->lo == a->hi && b->lo == b->hi && a->lo == b->lo) return TR_CMP_EQUAL;
    return TR_CMP_UNKNOWN;
}
static int civ_valid(const tr_carrier_cinterval *z) { return z && tr_interval_valid(z->re) && tr_interval_valid(z->im); }
int32_t tr_carrier_cinterval_op(int32_t op, const tr_carrier_cinterval *a, const tr_carrier_cinterval *b,
                                tr_carrier_cinterval *out) {
    if (!out || !civ_valid(a)) return TR_MATH_INVALID;
    int binary = op == TR_OP_ADD || op == TR_OP_SUB || op == TR_OP_MUL || op == TR_OP_DIV;
    if (binary && !civ_valid(b)) return TR_MATH_INVALID;
    tr_carrier_cinterval z;
    tr_math_interval two = tr_interval_point(2.0), den;
    switch (op) {
    case TR_OP_ADD: z.re = tr_interval_add(a->re, b->re); z.im = tr_interval_add(a->im, b->im); break;
    case TR_OP_SUB: z.re = tr_interval_sub(a->re, b->re); z.im = tr_interval_sub(a->im, b->im); break;
    case TR_OP_MUL:
        z.re = tr_interval_sub(tr_interval_mul(a->re, b->re), tr_interval_mul(a->im, b->im));
        z.im = tr_interval_add(tr_interval_mul(a->re, b->im), tr_interval_mul(a->im, b->re));
        break;
    case TR_OP_SQR:
        z.re = tr_interval_sub(tr_interval_sqr(a->re), tr_interval_sqr(a->im));
        z.im = tr_interval_mul(two, tr_interval_mul(a->re, a->im));
        break;
    case TR_OP_NEG: z.re = tr_interval_neg(a->re); z.im = tr_interval_neg(a->im); break;
    case TR_OP_CONJ: z.re = a->re; z.im = tr_interval_neg(a->im); break;
    case TR_OP_DIV:
    case TR_OP_INV: {
        const tr_carrier_cinterval *w = op == TR_OP_DIV ? b : a;
        den = tr_interval_add(tr_interval_sqr(w->re), tr_interval_sqr(w->im));
        if (!(den.lo > 0)) return TR_MATH_NO_SOLUTION;
        tr_math_interval nre, nim;
        if (op == TR_OP_DIV) {   /* a conj(w) / |w|^2 */
            nre = tr_interval_add(tr_interval_mul(a->re, w->re), tr_interval_mul(a->im, w->im));
            nim = tr_interval_sub(tr_interval_mul(a->im, w->re), tr_interval_mul(a->re, w->im));
        } else { nre = w->re; nim = tr_interval_neg(w->im); }
        z.re = tr_interval_div(nre, den); z.im = tr_interval_div(nim, den);
        break;
    }
    default: return TR_MATH_INVALID;
    }
    *out = z;
    return TR_MATH_OK;
}
int32_t tr_carrier_cinterval_abs(const tr_carrier_cinterval *a, tr_math_interval *out) {
    if (!out || !civ_valid(a)) return TR_MATH_INVALID;
    *out = tr_interval_sqrt(tr_interval_add(tr_interval_sqr(a->re), tr_interval_sqr(a->im)));
    return TR_MATH_OK;
}
int32_t tr_carrier_cinterval_real(const tr_carrier_cinterval *a, tr_math_interval *out) {
    if (!out || !civ_valid(a)) return TR_MATH_INVALID;
    if (!(a->im.lo == 0 && a->im.hi == 0)) return TR_MATH_NO_SOLUTION;
    *out = a->re;
    return TR_MATH_OK;
}

/* ================================================================ truncated power series over Q */
static tr_series *ser_new(uint32_t order, const tr_math_budget *b, tr_math_refusal *r, int32_t *err) {
    if (order < 1) { seterr(err, TR_MATH_INVALID); return NULL; }
    int32_t rc = check_order(b, order, r);
    if (rc) { seterr(err, rc); return NULL; }
    tr_series *s = (tr_series *)calloc(1, sizeof(tr_series) + (size_t)order * sizeof(tr_bigq *));
    if (!s) { seterr(err, TR_MATH_MEMORY); return NULL; }
    s->magic = SERIES_MAGIC; s->order = order;
    return s;
}
void tr_series_free(tr_series *s) {
    if (!is_ser(s)) return;
    s->magic = 0;
    for (uint32_t i = 0; i < s->order; i++) tr_bigq_free(s->coef[i]);
    free(s);
}
/* a finished series, or NULL (and s freed) when a coefficient failed */
static tr_series *ser_done(tr_series *s, int32_t *err) {
    for (uint32_t i = 0; i < s->order; i++)
        if (!s->coef[i]) { int32_t e = err && *err ? *err : TR_MATH_MEMORY; tr_series_free(s); seterr(err, e); return NULL; }
    seterr(err, TR_MATH_OK);
    return s;
}
tr_series *tr_series_create(const tr_bigq *const *coef, uint32_t count, uint32_t order, const tr_math_budget *b,
                            tr_math_refusal *r, int32_t *err) {
    if (count > order || (count && !coef)) { seterr(err, TR_MATH_INVALID); return NULL; }
    for (uint32_t i = 0; i < count; i++) {
        int32_t rc = want_bq(coef[i]);
        if (rc) { seterr(err, rc); return NULL; }
    }
    tr_series *s = ser_new(order, b, r, err);
    if (!s) return NULL;
    int32_t e = TR_MATH_OK;
    for (uint32_t i = 0; i < order && !e; i++)
        s->coef[i] = i < count ? tr_bigq_copy(coef[i], b, r, &e) : bq_int(0, b, r, &e);
    seterr(err, e);
    return ser_done(s, err);
}
tr_series *tr_series_constant(const tr_bigq *c, uint32_t order, const tr_math_budget *b, tr_math_refusal *r, int32_t *err) {
    return tr_series_create(&c, 1, order, b, r, err);
}
uint32_t tr_series_order(const tr_series *s) { return is_ser(s) ? s->order : 0; }
const tr_bigq *tr_series_coeff(const tr_series *s, uint32_t k) { return is_ser(s) && k < s->order ? s->coef[k] : NULL; }
int32_t tr_series_is_handle(const void *p) { return is_ser((const tr_series *)p); }
static int32_t ser_pair(const tr_series *a, const tr_series *c) {
    int32_t rc = want_ser(a);
    if (!rc) rc = want_ser(c);
    if (rc) return rc;
    return a->order == c->order ? TR_MATH_OK : TR_MATH_CARRIER_ERROR;
}
int32_t tr_series_equal(const tr_series *a, const tr_series *c) {
    int32_t rc = ser_pair(a, c);
    if (rc) return rc;
    for (uint32_t i = 0; i < a->order; i++) {
        int32_t k = bq_cmp(a->coef[i], c->coef[i]);
        if (k < -1 || k > 1) return k;
        if (k) return 0;
    }
    return 1;
}
tr_series *tr_series_truncate(const tr_series *a, uint32_t order, const tr_math_budget *b, tr_math_refusal *r, int32_t *err) {
    int32_t rc = want_ser(a);
    if (rc) { seterr(err, rc); return NULL; }
    if (order < 1) { seterr(err, TR_MATH_INVALID); return NULL; }
    if (order > a->order) { seterr(err, TR_MATH_NO_SOLUTION); return NULL; }
    return tr_series_create((const tr_bigq *const *)a->coef, order, order, b, r, err);
}
static tr_series *ser_addsub(const tr_series *a, const tr_series *c, int sub, const tr_math_budget *b, tr_math_refusal *r,
                             int32_t *err) {
    int32_t rc = ser_pair(a, c);
    if (rc) { seterr(err, rc); return NULL; }
    tr_series *s = ser_new(a->order, b, r, err);
    if (!s) return NULL;
    int32_t e = TR_MATH_OK;
    for (uint32_t i = 0; i < a->order && !e; i++)
        s->coef[i] = sub ? tr_bigq_sub(a->coef[i], c->coef[i], b, r, &e) : tr_bigq_add(a->coef[i], c->coef[i], b, r, &e);
    seterr(err, e);
    return ser_done(s, err);
}
tr_series *tr_series_add(const tr_series *a, const tr_series *c, const tr_math_budget *b, tr_math_refusal *r, int32_t *err) {
    return ser_addsub(a, c, 0, b, r, err);
}
tr_series *tr_series_sub(const tr_series *a, const tr_series *c, const tr_math_budget *b, tr_math_refusal *r, int32_t *err) {
    return ser_addsub(a, c, 1, b, r, err);
}
/* acc += x * y (acc replaced); 0 or an error */
static int32_t bq_fma(tr_bigq **acc, const tr_bigq *x, const tr_bigq *y, const tr_math_budget *b, tr_math_refusal *r) {
    if (!x->num->sign || !y->num->sign) return TR_MATH_OK;
    int32_t e = TR_MATH_OK;
    tr_bigq *p = tr_bigq_mul(x, y, b, r, &e);
    if (!p) return e;
    tr_bigq *s = tr_bigq_add(*acc, p, b, r, &e);
    tr_bigq_free(p);
    if (!s) return e;
    tr_bigq_free(*acc);
    *acc = s;
    return TR_MATH_OK;
}
/* the truncated product of coefficient arrays x, y of `order` terms into s (s->coef filled) */
static int32_t conv(tr_series *s, tr_bigq *const *x, tr_bigq *const *y, const tr_math_budget *b, tr_math_refusal *r) {
    int32_t e = TR_MATH_OK;
    for (uint32_t n = 0; n < s->order && !e; n++) {
        tr_bigq *acc = bq_int(0, b, r, &e);
        for (uint32_t k = 0; acc && k <= n && !e; k++) e = bq_fma(&acc, x[k], y[n - k], b, r);
        if (e) { tr_bigq_free(acc); break; }
        s->coef[n] = acc;
    }
    return e;
}
tr_series *tr_series_mul(const tr_series *a, const tr_series *c, const tr_math_budget *b, tr_math_refusal *r, int32_t *err) {
    int32_t rc = ser_pair(a, c);
    if (rc) { seterr(err, rc); return NULL; }
    tr_series *s = ser_new(a->order, b, r, err);
    if (!s) return NULL;
    seterr(err, conv(s, a->coef, c->coef, b, r));
    return ser_done(s, err);
}
tr_series *tr_series_scale(const tr_series *a, const tr_bigq *q, const tr_math_budget *b, tr_math_refusal *r, int32_t *err) {
    int32_t rc = want_ser(a);
    if (!rc) rc = want_bq(q);
    if (rc) { seterr(err, rc); return NULL; }
    tr_series *s = ser_new(a->order, b, r, err);
    if (!s) return NULL;
    int32_t e = TR_MATH_OK;
    for (uint32_t i = 0; i < a->order && !e; i++) s->coef[i] = tr_bigq_mul(a->coef[i], q, b, r, &e);
    seterr(err, e);
    return ser_done(s, err);
}
tr_series *tr_series_derivative(const tr_series *a, const tr_math_budget *b, tr_math_refusal *r, int32_t *err) {
    int32_t rc = want_ser(a);
    if (rc) { seterr(err, rc); return NULL; }
    if (a->order < 2) { seterr(err, TR_MATH_INVALID); return NULL; }
    tr_series *s = ser_new(a->order - 1, b, r, err);
    if (!s) return NULL;
    int32_t e = TR_MATH_OK;
    for (uint32_t k = 0; k + 1 < a->order && !e; k++) {
        tr_bigq *f = bq_int((int64_t)k + 1, b, r, &e);
        s->coef[k] = f ? tr_bigq_mul(a->coef[k + 1], f, b, r, &e) : NULL;
        tr_bigq_free(f);
    }
    seterr(err, e);
    return ser_done(s, err);
}
tr_series *tr_series_integral(const tr_series *a, const tr_math_budget *b, tr_math_refusal *r, int32_t *err) {
    int32_t rc = want_ser(a);
    if (rc) { seterr(err, rc); return NULL; }
    if (a->order == UINT32_MAX) { seterr(err, TR_MATH_INVALID); return NULL; }
    tr_series *s = ser_new(a->order + 1, b, r, err);
    if (!s) return NULL;
    int32_t e = TR_MATH_OK;
    s->coef[0] = bq_int(0, b, r, &e);
    for (uint32_t k = 0; k < a->order && !e; k++) {
        tr_bigq *f = bq_int((int64_t)k + 1, b, r, &e);
        s->coef[k + 1] = f ? tr_bigq_div(a->coef[k], f, b, r, &e) : NULL;
        tr_bigq_free(f);
    }
    seterr(err, e);
    return ser_done(s, err);
}
tr_series *tr_series_reciprocal(const tr_series *a, const tr_math_budget *b, tr_math_refusal *r, int32_t *err) {
    int32_t rc = want_ser(a);
    if (rc) { seterr(err, rc); return NULL; }
    if (!a->coef[0]->num->sign) { seterr(err, TR_MATH_NO_SOLUTION); return NULL; }
    tr_series *s = ser_new(a->order, b, r, err);
    if (!s) return NULL;
    int32_t e = TR_MATH_OK;
    tr_bigq *inv0 = tr_bigq_inv(a->coef[0], b, r, &e);
    if (!inv0) { tr_series_free(s); seterr(err, e); return NULL; }
    s->coef[0] = tr_bigq_copy(inv0, b, r, &e);
    /* h_n = -(1/c_0) sum_{k=1..n} c_k h_{n-k} */
    for (uint32_t n = 1; n < a->order && !e; n++) {
        tr_bigq *acc = bq_int(0, b, r, &e);
        for (uint32_t k = 1; acc && k <= n && !e; k++) e = bq_fma(&acc, a->coef[k], s->coef[n - k], b, r);
        tr_bigq *t = acc && !e ? tr_bigq_mul(acc, inv0, b, r, &e) : NULL;
        tr_bigq_free(acc);
        s->coef[n] = t ? tr_bigq_neg(t, b, r, &e) : NULL;
        tr_bigq_free(t);
    }
    tr_bigq_free(inv0);
    seterr(err, e);
    return ser_done(s, err);
}
tr_series *tr_series_compose(const tr_series *f, const tr_series *g, const tr_math_budget *b, tr_math_refusal *r,
                             int32_t *err) {
    int32_t rc = ser_pair(f, g);
    if (rc) { seterr(err, rc); return NULL; }
    if (g->coef[0]->num->sign) { seterr(err, TR_MATH_NO_SOLUTION); return NULL; }
    /* Horner: acc = f_{N-1}; acc = acc g + f_k for k = N-2 .. 0 */
    tr_series *acc = tr_series_constant(f->coef[f->order - 1], f->order, b, r, err);
    for (uint32_t k = f->order - 1; acc && k-- > 0;) {
        tr_series *p = tr_series_mul(acc, g, b, r, err);
        tr_series_free(acc);
        acc = NULL;
        if (!p) break;
        int32_t e = TR_MATH_OK;
        tr_bigq *c = tr_bigq_add(p->coef[0], f->coef[k], b, r, &e);
        if (!c) { tr_series_free(p); seterr(err, e); break; }
        tr_bigq_free(p->coef[0]);
        p->coef[0] = c;
        acc = p;
    }
    if (acc) seterr(err, TR_MATH_OK);
    return acc;
}
tr_series *tr_series_exp(const tr_series *a, const tr_math_budget *b, tr_math_refusal *r, int32_t *err) {
    int32_t rc = want_ser(a);
    if (rc) { seterr(err, rc); return NULL; }
    if (a->coef[0]->num->sign) { seterr(err, TR_MATH_NO_SOLUTION); return NULL; }
    tr_series *s = ser_new(a->order, b, r, err);
    if (!s) return NULL;
    int32_t e = TR_MATH_OK;
    s->coef[0] = bq_int(1, b, r, &e);
    /* g_n = (1/n) sum_{k=1..n} k f_k g_{n-k} */
    for (uint32_t n = 1; n < a->order && !e; n++) {
        tr_bigq *acc = bq_int(0, b, r, &e);
        for (uint32_t k = 1; acc && k <= n && !e; k++) {
            if (!a->coef[k]->num->sign) continue;
            tr_bigq *kk = bq_int(k, b, r, &e), *kf = kk ? tr_bigq_mul(kk, a->coef[k], b, r, &e) : NULL;
            tr_bigq_free(kk);
            if (kf) e = bq_fma(&acc, kf, s->coef[n - k], b, r);
            tr_bigq_free(kf);
        }
        tr_bigq *nn = acc && !e ? bq_int(n, b, r, &e) : NULL;
        s->coef[n] = nn ? tr_bigq_div(acc, nn, b, r, &e) : NULL;
        tr_bigq_free(nn);
        tr_bigq_free(acc);
    }
    seterr(err, e);
    return ser_done(s, err);
}
tr_series *tr_series_log(const tr_series *a, const tr_math_budget *b, tr_math_refusal *r, int32_t *err) {
    int32_t rc = want_ser(a);
    if (rc) { seterr(err, rc); return NULL; }
    if (!bq_is_one(a->coef[0]->num) || !bq_is_one(a->coef[0]->den)) { seterr(err, TR_MATH_NO_SOLUTION); return NULL; }
    tr_series *s = ser_new(a->order, b, r, err);
    if (!s) return NULL;
    int32_t e = TR_MATH_OK;
    s->coef[0] = bq_int(0, b, r, &e);
    /* g_n = f_n - (1/n) sum_{k=1..n-1} k g_k f_{n-k} */
    for (uint32_t n = 1; n < a->order && !e; n++) {
        tr_bigq *acc = bq_int(0, b, r, &e);
        for (uint32_t k = 1; acc && k < n && !e; k++) {
            if (!s->coef[k]->num->sign) continue;
            tr_bigq *kk = bq_int(k, b, r, &e), *kg = kk ? tr_bigq_mul(kk, s->coef[k], b, r, &e) : NULL;
            tr_bigq_free(kk);
            if (kg) e = bq_fma(&acc, kg, a->coef[n - k], b, r);
            tr_bigq_free(kg);
        }
        tr_bigq *nn = acc && !e ? bq_int(n, b, r, &e) : NULL, *t = nn ? tr_bigq_div(acc, nn, b, r, &e) : NULL;
        s->coef[n] = t ? tr_bigq_sub(a->coef[n], t, b, r, &e) : NULL;
        tr_bigq_free(nn); tr_bigq_free(t); tr_bigq_free(acc);
    }
    seterr(err, e);
    return ser_done(s, err);
}
