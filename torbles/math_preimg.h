/* PRE-002: finite roots, residues and periodic preimages: every bounded preimage branch of a predicate on a derived
 * value as intervals, residue classes, finite sets or Boolean cubes of stored codes.
 * SPDX-License-Identifier: Apache-2.0
 *
 * The generator column's codec and the predicate are PRE-001's (math_pre.h): logical L = code + offset, x = L / T with
 * T = 10^scale, codes over the declared domain [lo, hi]; y op c (op LT .. GT, c an int64 rational) or y IN {values}.
 * Maps beyond PRE-001's (tr_preimg_map):
 *   POLY     y = sum_i a_i x^i, rational a_i, degree 1..TR_PREIMG_MAX_DEGREE (a constant map decides ALL or NONE). With
 *            K = T^d lcm(den a_i) > 0 the integer polynomial N(L) = K p(L / T) (FND-009 big integers) and thresholds
 *            R = K num(c): y op c iff N(L) den(c) op R. The critical points (the real roots of N') are isolated by
 *            INV-001 (Sturm sequences, tr_inv_isolate, refined to width 1 by tr_inv_roots_refine) in [L_lo, L_hi];
 *            between them N is strictly monotone on the integers: pieces [.., floor(rho)] and [floor(rho) + 1, ..] around
 *            an exact critical point, the integers <= floor(alpha) and >= ceil(beta) around an isolating interval
 *            (alpha, beta), the integers strictly inside it as single points. On each piece the sign of N(L) den - R is
 *            monotone and two binary searches (exact big-integer evaluations) give its level sets: an interval for
 *            each op, a point or nothing for EQ (the exact integer roots), every value of IN through the same pieces.
 *   POLYMOD  y = (sum_i c_i x^i) mod m, integer c_i, x an integer (scale 0), 1 <= m <= 2^62 (power residues x^k mod m):
 *            y depends on L mod m only. EQ and IN: the residues r with P(r) = y (mod m) by the CRT over the prime powers
 *            q of m (trial division; each q enumerated, q <= TR_PREIMG_MAX_PERIOD; the classes combined by CRT); other
 *            ops: every residue r < m enumerated (m <= TR_PREIMG_MAX_PERIOD).
 *   AFFMOD   y = (a x + b) mod m, integers, scale 0: a x = y - b (mod m) is solvable iff g = gcd(a, m) divides y - b,
 *            then x = ((y - b) / g) (a / g)^-1 (mod m / g): residue classes of the period m / g, one per solvable y.
 *   XOR, AND, OR  y = u XOR c, u AND c, u OR c on the w low bits u = (L - base) mod 2^w (FIELD-001's bit variables of
 *            the logical integer; base_is_offset: base = the codec offset, so u is the stored code's bits); the GF(2)
 *            images of the dyadic blocks of {y op c} are Boolean cubes {u : u & mask = value} (XOR translates the
 *            fixed bits, AND fixes the bits of c, OR the bits outside c; a block whose fixed bits the map cannot
 *            produce is empty). Codes: u = (code + d) mod 2^w with d = offset - base: a cube whose highest fixed bit is
 *            h is a code cube when d = 0 mod 2^(h+1), else residue runs of codes mod 2^(H+1) (H the highest over the cubes).
 * Every result is a union of branches of one representation:
 *   INTERVALS  codes lo..hi (sorted, disjoint, not adjacent; single points for finite sets);
 *   RESIDUES   codes of the domain with code mod M in rlo..rhi (one period M; sorted, disjoint runs); a domain no
 *              larger than M becomes INTERVALS;
 *   CUBES      codes of the domain with (uint64) code & mask == value (disjoint).
 * Budgeted fallback (never a dropped solution): when the exact union has more branches than the output budget, or
 * its construction would exceed the iterations or bytes budget or a representation limit (a period past
 * TR_PREIMG_MAX_PERIOD, too many CRT classes), the set is represented by
 *   LOOKUP     a bitmap of the residues mod M (periodic maps, M <= TR_PREIMG_MAX_LOOKUP bits) or of the domain's codes
 *              (code - lo < TR_PREIMG_MAX_LOOKUP), computed exactly on the host, or
 *   DIRECT     the original predicate evaluated exactly on each value (POLY: N(L) in W 64-bit words under a bound proven
 *              over the domain, W <= TR_PREIMG_MAX_WORDS; the periodic maps: 128-bit modular arithmetic; bit maps).
 * spec.fallback: AUTO (LOOKUP when it fits the bytes budget, else DIRECT), LOOKUP or DIRECT only, NONE (refused:
 * TR_MATH_BUDGET with the refusal of the exceeded resource). The info says which and why.
 *
 * Exactness: thresholds, critical points and level sets are exact (FND-009 big integers and rationals, INV-001 Sturm
 * isolation); residues and cubes are exact integer and bit arithmetic. The set is exact on the codec's domain. NULL
 * guard: a NULL value never satisfies. Certificates (FND-007 LAW, ring identities decided by FND-007): POLY the codec
 * substitution sum_i n_i (code + offset)^i expanded term by term minus its code-space form Q(code); POLYMOD the
 * periodicity P(r + m t) - P(r) - m Q(r, t) = 0 over (r, t); AFFMOD a (r + M t) + b - (a r + b) - (a / g) m t = 0 with
 * M = m / g. One per set when every coefficient fits int64 (info.certified), else none (the set stays exact).
 *
 * Scan (tr_preimg_scan; tr_mgpu_preimg_scan, bit for bit): bit r for a selected row whose value is present and whose
 * code is in the set (any representation); with direct = 1 the original predicate is also evaluated on every row
 * (the DIRECT evaluation; refused for a POLY without a proven width) and compared: PRE-001's tr_pre_scan_stats.
 * Verification (tr_preimg_verify; tr_mgpu_preimg_verify, bit for bit): every code of the domain (at most
 * TR_PREIMG_MAX_VERIFY codes, L within int64): membership against the direct predicate: members, true codes,
 * mismatches and the smallest mismatching code.
 *
 * Budgets (math_budgets.h): output = branches (exceeded: the fallback), iterations = exact evaluations (bisection
 * steps, residues enumerated, trial divisions, CRT classes; exceeded: the fallback), support = IN values and the
 * polynomial's terms, degree, bytes (the set, the scratch, the lookup; exceeded: the fallback or a refusal), INV-001's
 * own budget for the isolation. Errors: TR_MATH_INVALID, _MEMORY, _OVERFLOW, _BUDGET, _UNSUPPORTED (a scale for the
 * periodic maps, a width past 63, a modulus past 2^62, a domain below 0 for bit maps, a direct check without a proven
 * width), _DOMAIN_ERROR (a zero modulus). Objects are immutable and individually owned. */
#ifndef TORBLES_MATH_PREIMG_H
#define TORBLES_MATH_PREIMG_H
#include <stdint.h>
#include "math_num.h"
#include "math_budgets.h"
#include "math_scope.h"
#include "math_pre.h"
#ifdef __cplusplus
extern "C" {
#endif

#define TR_PREIMG_ABI 1
#define TR_PREIMG_MAX_DEGREE 16
#define TR_PREIMG_MAX_WORDS 8
#define TR_PREIMG_MAX_VALUES 4096
#define TR_PREIMG_MAX_PERIOD (UINT64_C(1) << 24)       /* residues enumerated (a prime power, or the whole period) */
#define TR_PREIMG_MAX_LOOKUP (UINT64_C(1) << 26)       /* bits of a lookup table */
#define TR_PREIMG_MAX_VERIFY (UINT64_C(1) << 26)       /* codes of an exhaustive verification */
#define TR_PREIMG_MAX_MODULUS (INT64_C(1) << 62)

enum { TR_PREIMG_POLY = 1, TR_PREIMG_POLYMOD = 2, TR_PREIMG_AFFMOD = 3, TR_PREIMG_XOR = 4, TR_PREIMG_AND = 5,
       TR_PREIMG_OR = 6 };                                                      /* maps */
enum { TR_PREIMG_BRANCHES = 1, TR_PREIMG_LOOKUP = 2, TR_PREIMG_DIRECT = 3 };      /* modes */
enum { TR_PREIMG_INTERVALS = 1, TR_PREIMG_RESIDUES = 2, TR_PREIMG_CUBES = 3 };    /* branch representations */
enum { TR_PREIMG_FALLBACK_AUTO = 0, TR_PREIMG_FALLBACK_LOOKUP = 1, TR_PREIMG_FALLBACK_DIRECT = 2,
       TR_PREIMG_FALLBACK_NONE = 3 };
enum { TR_PREIMG_WHY_NONE = 0, TR_PREIMG_WHY_OUTPUT = 1, TR_PREIMG_WHY_ITERATIONS = 2, TR_PREIMG_WHY_BYTES = 3,
       TR_PREIMG_WHY_PERIOD = 4, TR_PREIMG_WHY_CLASSES = 5 };                    /* why a set fell back */

/* ------------------------------------------------------------------ structs crossing the ctypes boundary */
typedef struct {
    int32_t kind, degree;                /* TR_PREIMG_*; POLY / POLYMOD: the degree (degree + 1 coefficients) */
    int32_t width, base_is_offset;       /* bit maps: w (1..63); 1: base = the codec offset (the stored code's bits) */
    int64_t modulus;                     /* POLYMOD / AFFMOD: m >= 1 */
    int64_t a, b;                        /* AFFMOD: (a x + b) mod m; bit maps: b = the constant c (0 <= c < 2^w) */
    int64_t base;                        /* bit maps: u = (L - base) mod 2^w (unless base_is_offset) */
    const tr_num_rational *coef;         /* POLY: degree + 1 rationals, low degree first; POLYMOD: integers */
} tr_preimg_map;                         /* 56 bytes */

typedef struct {
    int32_t fallback, reserved;          /* TR_PREIMG_FALLBACK_* */
} tr_preimg_spec;                        /* 8 bytes */

typedef struct {
    int64_t lo, hi;                      /* codes lo..hi (RESIDUES and CUBES: the domain) */
    int64_t modulus, rlo, rhi;           /* RESIDUES: code mod modulus in rlo..rhi (1, 0, 0 otherwise) */
    uint64_t mask, value;                /* CUBES: (uint64) code & mask == value */
    int64_t reserved;
} tr_preimg_branch;                      /* 64 bytes */

typedef struct {
    int32_t map, op, mode, repr;         /* TR_PREIMG_* map, TR_PRE_* op, mode, representation (BRANCHES) */
    int32_t fallback, why;               /* the fallback taken (0: none) and why (TR_PREIMG_WHY_*) */
    int32_t words, certified;            /* POLY: direct-evaluation words (0: none); law certificates */
    int32_t pieces, critical;            /* POLY: monotone integer pieces; critical points of N' isolated */
    int32_t empty, all;
    uint64_t nbranch;                    /* branches stored (BRANCHES) */
    uint64_t branches;                   /* branches of the exact union (UINT64_MAX: not counted) */
    uint64_t codes;                      /* codes of the domain in the set (UINT64_MAX: past 2^64 or not counted) */
    int64_t modulus;                     /* RESIDUES / a residue LOOKUP: the period M (0 otherwise) */
    uint64_t nlookup;                    /* LOOKUP: bits */
    uint64_t iterations;                 /* exact evaluations performed */
    uint64_t bytes;
    tr_pre_codec codec;
} tr_preimg_info;                        /* 136 bytes */

/* the plain view the scan, the verification and their kernels read (pointers into the set; swapped for device copies) */
typedef struct {
    tr_pre_codec codec;
    int32_t map, op, mode, repr;
    int32_t words, degree, width, reserved;
    int64_t modulus;                     /* RESIDUES / residue LOOKUP: M; a domain LOOKUP: 0 */
    int64_t m, a, b, base;               /* POLYMOD / AFFMOD: m, a mod m, b mod m; bit maps: base, b = c */
    uint64_t nbranch;
    const tr_preimg_branch *branch;
    uint64_t nlookup;
    const uint64_t *lookup;              /* LOOKUP: ceil(nlookup / 64) words */
    const int64_t *icoef;                /* POLYMOD: degree + 1 coefficients mod m (low first) */
    const uint64_t *wcoef;               /* POLY (words > 0): (degree + 1) x words, N's coefficients (two's complement) */
    tr_num_rational c;                   /* the threshold (not IN) */
    uint64_t nvalues;
    const tr_num_rational *values;       /* IN: sorted, distinct */
    const uint64_t *wthreshold;          /* POLY (words > 0): (IN: nvalues, else 1) x words: K num(c_j) */
} tr_preimg_view;                        /* 192 bytes */

typedef struct {
    uint64_t codes, members, truths;     /* domain codes checked; in the set; the predicate true */
    uint64_t mismatches;                 /* codes where membership and the predicate differ */
    int64_t first;                       /* the smallest mismatching code (valid when mismatches > 0) */
    int64_t lo, hi;                      /* the domain verified */
    uint64_t reserved;
} tr_preimg_verification;                /* 64 bytes */

typedef struct tr_preimg_set_s tr_preimg_set;

/* ------------------------------------------------------------------ shared inline functions (CPU and kernels) */
/* W-word two's complement integers modulo 2^(64 W), least significant word first */
TR_NUM_HD void tr_preimg_w_copy(uint64_t *d, const uint64_t *s, int32_t W) { for (int32_t i = 0; i < W; i++) d[i] = s[i]; }
TR_NUM_HD void tr_preimg_w_neg(uint64_t *a, int32_t W) {
    uint64_t c = 1;
    for (int32_t i = 0; i < W; i++) { uint64_t t = ~a[i] + c; c = (c && t == 0) ? 1 : 0; a[i] = t; }
}
TR_NUM_HD void tr_preimg_w_add(uint64_t *acc, const uint64_t *a, int32_t W) {
    uint64_t c = 0;
    for (int32_t i = 0; i < W; i++) {
        uint64_t s = acc[i] + a[i], c1 = s < a[i] ? 1 : 0, s2 = s + c, c2 = s2 < c ? 1 : 0;
        acc[i] = s2; c = c1 | c2;
    }
}
/* acc *= m (an unsigned 64-bit factor) */
TR_NUM_HD void tr_preimg_w_mul(uint64_t *a, int32_t W, uint64_t m) {
    unsigned __int128 carry = 0;
    for (int32_t i = 0; i < W; i++) {
        unsigned __int128 t = (unsigned __int128)a[i] * m + carry;
        a[i] = (uint64_t)t; carry = t >> 64;
    }
}
/* acc *= x (signed) */
TR_NUM_HD void tr_preimg_w_mul_signed(uint64_t *a, int32_t W, int64_t x) {
    tr_preimg_w_mul(a, W, tr_abs_u64(x));
    if (x < 0) tr_preimg_w_neg(a, W);
}
/* the sign of a (exact by the proven bound): -1, 0, 1 */
TR_NUM_HD int tr_preimg_w_sign(const uint64_t *a, int32_t W) {
    if (a[W - 1] >> 63) return -1;
    for (int32_t i = 0; i < W; i++) if (a[i]) return 1;
    return 0;
}
/* N(L) in W words by Horner's rule (the bound proves |N(L)| < 2^(64 W - 1) on the domain) */
TR_NUM_HD void tr_preimg_poly_value(const tr_preimg_view *v, int64_t L, uint64_t *acc) {
    const int32_t W = v->words, d = v->degree;
    tr_preimg_w_copy(acc, v->wcoef + (uint64_t)d * (uint64_t)W, W);
    for (int32_t i = d; i-- > 0;) {
        tr_preimg_w_mul_signed(acc, W, L);
        tr_preimg_w_add(acc, v->wcoef + (uint64_t)i * (uint64_t)W, W);
    }
}
/* the sign of N den - R (W words) */
TR_NUM_HD int tr_preimg_poly_cmp(const uint64_t *n, int32_t W, int64_t den, const uint64_t *R) {
    uint64_t t[TR_PREIMG_MAX_WORDS] = {0}, r[TR_PREIMG_MAX_WORDS] = {0};
    tr_preimg_w_copy(t, n, W);
    tr_preimg_w_mul(t, W, (uint64_t)den);
    tr_preimg_w_copy(r, R, W);
    tr_preimg_w_neg(r, W);
    tr_preimg_w_add(t, r, W);
    return tr_preimg_w_sign(t, W);
}
TR_NUM_HD int64_t tr_preimg_fmod128(__int128 x, int64_t m) {        /* m >= 1 */
    __int128 r = x % m;
    return (int64_t)(r < 0 ? r + m : r);
}
/* y of the periodic and bit maps at L (an exact integer in [0, m) or [0, 2^w)) */
TR_NUM_HD int64_t tr_preimg_int_value(const tr_preimg_view *v, int64_t L) {
    switch (v->map) {
    case TR_PREIMG_POLYMOD: {
        const uint64_t m = (uint64_t)v->m, x = (uint64_t)tr_preimg_fmod128(L, v->m);
        unsigned __int128 r = (unsigned __int128)(uint64_t)v->icoef[v->degree];
        for (int32_t i = v->degree; i-- > 0;) r = (r * x + (uint64_t)v->icoef[i]) % m;
        return (int64_t)(r % m);
    }
    case TR_PREIMG_AFFMOD:
        return tr_preimg_fmod128((__int128)v->a * L + v->b, v->m);
    default: {
        const uint64_t wm = v->width >= 64 ? UINT64_MAX : (((uint64_t)1 << v->width) - 1);
        const uint64_t u = ((uint64_t)L - (uint64_t)v->base) & wm, c = (uint64_t)v->b;
        uint64_t y = v->map == TR_PREIMG_XOR ? (u ^ c) : v->map == TR_PREIMG_AND ? (u & c) : (u | c);
        return (int64_t)(y & wm);
    }
    }
}
/* y op c for an integer y */
TR_NUM_HD int tr_preimg_int_holds(const tr_preimg_view *v, int64_t y) {
    if (v->op == TR_PRE_IN) {
        uint64_t lo = 0, hi = v->nvalues;
        while (lo < hi) {
            uint64_t mid = lo + (hi - lo) / 2;
            __int128 l = (__int128)v->values[mid].numerator, r = (__int128)y * v->values[mid].denominator;
            if (l < r) lo = mid + 1; else hi = mid;
        }
        return lo < v->nvalues && (__int128)v->values[lo].numerator == (__int128)y * v->values[lo].denominator;
    }
    __int128 l = (__int128)y * v->c.denominator, r = v->c.numerator;
    int s = l < r ? -1 : l > r ? 1 : 0;
    switch (v->op) {
    case TR_PRE_LT: return s < 0;
    case TR_PRE_LE: return s <= 0;
    case TR_PRE_EQ: return s == 0;
    case TR_PRE_NE: return s != 0;
    case TR_PRE_GE: return s >= 0;
    default: return s > 0;
    }
}
/* the original predicate at the value of a code (L = code + offset within int64): 1 / 0, or -1 when there is no
 * direct evaluation (a POLY without a proven width) */
TR_NUM_HD int tr_preimg_direct(const tr_preimg_view *v, int64_t L) {
    if (v->map != TR_PREIMG_POLY) return tr_preimg_int_holds(v, tr_preimg_int_value(v, L));
    const int32_t W = v->words;
    if (W < 1 || W > TR_PREIMG_MAX_WORDS) return -1;
    uint64_t n[TR_PREIMG_MAX_WORDS];
    tr_preimg_poly_value(v, L, n);
    if (v->op == TR_PRE_IN) {
        uint64_t lo = 0, hi = v->nvalues;
        while (lo < hi) {                                /* the first value >= y = N / K */
            uint64_t mid = lo + (hi - lo) / 2;
            if (tr_preimg_poly_cmp(n, W, v->values[mid].denominator, v->wthreshold + mid * (uint64_t)W) > 0) lo = mid + 1;
            else hi = mid;
        }
        return lo < v->nvalues && tr_preimg_poly_cmp(n, W, v->values[lo].denominator, v->wthreshold + lo * (uint64_t)W) == 0;
    }
    int s = tr_preimg_poly_cmp(n, W, v->c.denominator, v->wthreshold);
    switch (v->op) {
    case TR_PRE_LT: return s < 0;
    case TR_PRE_LE: return s <= 0;
    case TR_PRE_EQ: return s == 0;
    case TR_PRE_NE: return s != 0;
    case TR_PRE_GE: return s >= 0;
    default: return s > 0;
    }
}
/* membership of a code (any mode); a DIRECT set without an evaluation answers 0 */
TR_NUM_HD int tr_preimg_member(const tr_preimg_view *v, int64_t code) {
    if (code < v->codec.lo || code > v->codec.hi) return 0;
    if (v->mode == TR_PREIMG_DIRECT) {
        __int128 L = (__int128)code + v->codec.offset;
        if (L < INT64_MIN || L > INT64_MAX) return 0;
        return tr_preimg_direct(v, (int64_t)L) == 1;
    }
    if (v->mode == TR_PREIMG_LOOKUP) {
        uint64_t key = v->modulus ? (uint64_t)tr_preimg_fmod128(code, v->modulus) : (uint64_t)code - (uint64_t)v->codec.lo;
        return key < v->nlookup && ((v->lookup[key >> 6] >> (key & 63)) & 1);
    }
    if (!v->nbranch) return 0;
    if (v->repr == TR_PREIMG_CUBES) {
        for (uint64_t i = 0; i < v->nbranch; i++) {
            const tr_preimg_branch *b = &v->branch[i];
            if (code >= b->lo && code <= b->hi && ((uint64_t)code & b->mask) == b->value) return 1;
        }
        return 0;
    }
    uint64_t lo = 0, hi = v->nbranch;
    if (v->repr == TR_PREIMG_INTERVALS) {
        while (lo < hi) { uint64_t mid = lo + (hi - lo) / 2; if (v->branch[mid].lo <= code) lo = mid + 1; else hi = mid; }
        return lo > 0 && code <= v->branch[lo - 1].hi;
    }
    const int64_t r = tr_preimg_fmod128(code, v->branch[0].modulus);
    while (lo < hi) { uint64_t mid = lo + (hi - lo) / 2; if (v->branch[mid].rlo <= r) lo = mid + 1; else hi = mid; }
    return lo > 0 && r <= v->branch[lo - 1].rhi;
}
/* one 64-row word of the scan of scope column `col`: the bitmap word; acc[0..5] += nulls, in_set, direct_true,
 * mismatches, undecided, outside; *first_mismatch, *first_undecided = the smallest such rows */
TR_NUM_HD uint64_t tr_preimg_word(const tr_preimg_view *v, const tr_scope_view *sv, int32_t col, int direct, uint64_t w,
                                  uint64_t acc[6], uint64_t *first_mismatch, uint64_t *first_undecided) {
    uint64_t bits = sv->selection[w], out = 0;
    while (bits) {
        int bit = tr_ipoly_ctz(bits);
        bits &= bits - 1;
        uint64_t row = w * 64 + (uint64_t)bit;
        if (!tr_scope_present(sv, col, row)) { acc[0]++; continue; }        /* NULL never satisfies */
        int64_t logical = tr_scope_value(sv, col, row);
        __int128 exact = (__int128)logical - v->codec.offset;
        int inside = exact >= v->codec.lo && exact <= v->codec.hi;
        int m = inside && tr_preimg_member(v, (int64_t)exact);
        if (!inside) acc[5]++;
        if (m) { out |= (uint64_t)1 << bit; acc[1]++; }
        if (!direct || !inside) continue;                 /* outside the domain: the set is not stated there */
        int truth = tr_preimg_direct(v, logical);
        if (truth < 0) { acc[4]++; if (row < *first_undecided) *first_undecided = row; continue; }
        if (truth) acc[2]++;
        if (truth != m) { acc[3]++; if (row < *first_mismatch) *first_mismatch = row; }
    }
    return out;
}
/* one 64-code word of the exhaustive verification (codes lo + 64 w ..): acc[0..3] += codes, members, truths,
 * mismatches; *first = the smallest mismatching offset code - lo */
TR_NUM_HD void tr_preimg_verify_word(const tr_preimg_view *v, uint64_t w, uint64_t total, uint64_t acc[4], uint64_t *first) {
    for (uint64_t k = w * 64; k < w * 64 + 64 && k < total; k++) {
        int64_t code = (int64_t)((uint64_t)v->codec.lo + k);
        int64_t L = (int64_t)((uint64_t)code + (uint64_t)v->codec.offset);
        int m = tr_preimg_member(v, code), t = tr_preimg_direct(v, L) == 1;
        acc[0]++;
        if (m) acc[1]++;
        if (t) acc[2]++;
        if (m != t) { acc[3]++; if (k < *first) *first = k; }
    }
}
/* the device copies of tr_mgpu_preimg_scan and tr_mgpu_preimg_verify */
static inline uint64_t tr_preimg_set_device_bytes(const tr_preimg_view *v) {
    uint64_t b = tr_budget_mul(v->nbranch, sizeof(tr_preimg_branch));
    b = tr_budget_add(b, tr_budget_bitmap_bytes(v->nlookup));
    if (v->map == TR_PREIMG_POLYMOD) b = tr_budget_add(b, tr_budget_mul((uint64_t)v->degree + 1, 8));
    if (v->map == TR_PREIMG_POLY && v->words > 0) {
        b = tr_budget_add(b, tr_budget_mul(tr_budget_mul((uint64_t)v->degree + 1, (uint64_t)v->words), 8));
        b = tr_budget_add(b, tr_budget_mul(tr_budget_mul(v->nvalues ? v->nvalues : 1, (uint64_t)v->words), 8));
    }
    return tr_budget_add(tr_budget_add(b, tr_budget_mul(v->nvalues, sizeof(tr_num_rational))), 64);
}
static inline uint64_t tr_preimg_scan_device_bytes(const tr_preimg_view *v, const tr_scope_view *sv, int32_t col) {
    uint64_t rows = sv->rows, b = tr_budget_mul(tr_budget_bitmap_bytes(rows), 2);
    b = tr_budget_add(b, tr_budget_mul(rows, sv->col[col].values ? 8 : 4));
    if (sv->col[col].valid) b = tr_budget_add(b, rows);
    return tr_budget_add(b, tr_preimg_set_device_bytes(v));
}
/* the structure the scan and the verification rely on (host code of both libraries): 1 when well formed */
static inline int tr_preimg_view_ok(const tr_preimg_view *v) {
    if (!v || v->codec.lo > v->codec.hi || v->codec.scale < 0 || v->codec.scale > TR_SCOPE_MAX_SCALE) return 0;
    if (v->map < TR_PREIMG_POLY || v->map > TR_PREIMG_OR || v->op < TR_PRE_LT || v->op > TR_PRE_IN) return 0;
    if (v->op == TR_PRE_IN && v->nvalues && !v->values) return 0;
    if (v->op != TR_PRE_IN && v->c.denominator <= 0) return 0;
    for (uint64_t i = 0; v->op == TR_PRE_IN && i < v->nvalues; i++) if (v->values[i].denominator <= 0) return 0;
    if (v->degree < 0 || v->degree > TR_PREIMG_MAX_DEGREE || v->words < 0 || v->words > TR_PREIMG_MAX_WORDS) return 0;
    if (v->map == TR_PREIMG_POLY && v->words && (!v->wcoef || !v->wthreshold)) return 0;
    if ((v->map == TR_PREIMG_POLYMOD || v->map == TR_PREIMG_AFFMOD) && (v->m < 1 || v->m > TR_PREIMG_MAX_MODULUS)) return 0;
    if (v->map == TR_PREIMG_POLYMOD && !v->icoef) return 0;
    if (v->map >= TR_PREIMG_XOR && (v->width < 1 || v->width > 63)) return 0;
    if (v->mode == TR_PREIMG_LOOKUP) return v->lookup && v->nlookup && v->nlookup <= TR_PREIMG_MAX_LOOKUP && v->modulus >= 0;
    if (v->mode == TR_PREIMG_DIRECT) return v->map != TR_PREIMG_POLY || v->words > 0;
    if (v->mode != TR_PREIMG_BRANCHES || (v->nbranch && !v->branch)) return 0;
    for (uint64_t i = 0; i < v->nbranch; i++) {
        const tr_preimg_branch *b = &v->branch[i];
        if (b->lo > b->hi) return 0;
        if (v->repr == TR_PREIMG_RESIDUES && (b->modulus < 1 || b->modulus != v->branch[0].modulus || b->rlo > b->rhi ||
                                              b->rlo < 0 || b->rhi >= b->modulus || (i && b->rlo <= v->branch[i - 1].rhi))) return 0;
        if (v->repr == TR_PREIMG_INTERVALS && i && b->lo <= v->branch[i - 1].hi) return 0;
        if (v->repr == TR_PREIMG_CUBES && (b->value & ~b->mask)) return 0;
    }
    return v->repr >= TR_PREIMG_INTERVALS && v->repr <= TR_PREIMG_CUBES;
}

/* ------------------------------------------------------------------ exported (math_preimg.c) */
tr_preimg_set *tr_preimg_preimage(const tr_pre_codec *, const tr_preimg_map *, const tr_pre_predicate *,
                                  const tr_preimg_spec *, const tr_math_budget *, tr_math_refusal *, int32_t *error);
int32_t tr_preimg_set_get_info(const tr_preimg_set *, tr_preimg_info *);
int32_t tr_preimg_set_branches(const tr_preimg_set *, tr_preimg_branch *out, uint64_t capacity);
int32_t tr_preimg_set_contains(const tr_preimg_set *, int64_t code);   /* 1 / 0 (TR_MATH_INVALID: a bad handle) */
int32_t tr_preimg_set_get_view(const tr_preimg_set *, tr_preimg_view *);
/* the lookup bits (ceil(nlookup / 64) words) of a LOOKUP set */
int32_t tr_preimg_set_lookup(const tr_preimg_set *, uint64_t *out, uint64_t capacity_words);
/* the set as cubes on the code bits below `width` (CUBES; RESIDUES of a power-of-two period decomposed into low-bit
 * cubes), each restricted to the domain; TR_MATH_UNSUPPORTED for another representation; *count the cubes */
int32_t tr_preimg_set_cubes(const tr_preimg_set *, int32_t width, tr_preimg_branch *out, uint64_t capacity,
                            uint64_t *count);
/* the law certificate of the transformation (borrowed; NULL when there is none) */
const tr_cert *tr_preimg_set_certificate(const tr_preimg_set *);
void tr_preimg_set_free(tr_preimg_set *);
/* bitmap: ceil(rows / 64) words; stats: PRE-001's (overflow = rows the direct check could not decide) */
int32_t tr_preimg_scan(const tr_preimg_set *, const tr_scope *, int32_t column, int32_t direct, uint64_t *bitmap,
                       tr_pre_scan_stats *, const tr_math_budget *, tr_math_refusal *);
int32_t tr_preimg_verify(const tr_preimg_set *, const tr_math_budget *, tr_preimg_verification *, tr_math_refusal *);

/* ------------------------------------------------------------------ the CUDA library (math_preimg_gpu.cu) */
int tr_mgpu_preimg_scan(const tr_preimg_view *, const tr_scope_view *, int32_t column, int32_t direct, uint64_t *bitmap,
                        tr_pre_scan_stats *, const tr_math_budget *, tr_math_refusal *);
int tr_mgpu_preimg_verify(const tr_preimg_view *, const tr_math_budget *, tr_preimg_verification *, tr_math_refusal *);
#ifdef __cplusplus
}
#endif
#endif
