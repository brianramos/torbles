/* DIST-003: entropy, mutual information and contingency statistics of the retained count support of a DIST-001 joint
 * histogram (math_dist.h), with certified logarithms and exact integer accumulation, on the CPU and in CUDA.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Source. A histogram (tr_dist_hist) with k = 1..3 coordinates and G groups; the mass of a cell is its count (rows,
 * TR_DIST_MASS_COUNT) or its weight sum (TR_DIST_MASS_WEIGHT: every present weight >= 0, else TR_MATH_DOMAIN_ERROR).
 * Cells absent from the support and cells of zero mass are structural zeros: they contribute 0 log 0 = 0 and are never
 * missing data. Masses are exact integers; per group the total N must stay below 2^64 (counts always do; a weight total
 * at or past 2^64 is TR_MATH_UNSUPPORTED).
 *
 * NULL. The NULL category of a coordinate is a category of its own (as in the histogram), unless the coordinate is in
 * the declared exclusion mask: then every cell with a NULL in an excluded coordinate is dropped before any statistic
 * (listwise over the excluded coordinates; N is the remaining mass). The mask is fixed when the support is built.
 *
 * The support (tr_info_support, built once on the CPU from the histogram's states, no row scan): per group and for every
 * subset S of the coordinates the marginal table T(S) of the kept cells (S = 0: one cell of mass N; S = all: the joint),
 * and for every unordered pair of disjoint nonempty coordinate sets (A, B), A < B as masks, the cells of T(A | B) with
 * their masses O and the masses m_a, m_b of their A- and B-projections. Marginals are computed exactly from the joint
 * cells; nothing is ever inferred from marginals or moments: a statistic of coordinates the state does not hold jointly
 * (e.g. mutual information of a 1-D state) is refused (TR_MATH_UNSUPPORTED).
 *
 * Sums (the data-parallel part, tr_info_sums on the CPU and tr_mgpu_info_sums in CUDA, the same inline terms):
 *   E_S  = sum over the cells of T(S) of m ln m, enclosed: each cell adds floor(m lo / 2^8) to the lower and
 *          ceil(m hi / 2^8) to the upper sum, where lo <= ln(m) 2^120 <= hi (tr_info_ln), so the lower and upper sums
 *          bound E_S 2^112 (TR_INFO_SCALE)
 *   C_AB = sum over the cells of T(A | B) of O^2 / (m_a m_b), each cell as floor / ceil(O^2 2^128 / (m_a m_b)) (exact
 *          integer division): the bounds of C_AB 2^128 (TR_INFO_CHI_SCALE)
 * Every bound is an exact nonnegative integer summed in 256 bits (below 2^190), so the sums do not depend on the order
 * of the additions: the CUDA kernels add them with exact 256-bit atomics and equal the CPU's bit for bit.
 *
 * The logarithm (tr_info_ln, 1 <= m < 2^64): m = 2^e f with f in [sqrt(2)/2, sqrt(2)] (exact), ln m = e ln 2 + 2 atanh(z),
 * z = (f - 1) / (f + 1), |z| < 0.1716, in 120-bit fixed point: z by exact restoring division (floor), the series
 * sum_{j <= 23} z^(2j+1) / (2j+1) with every product and quotient truncated downward. Proof: each truncated power is
 * within 1.27 ulps below the exact one and each term within 1.43 ulps (the first within 1), the tail after 24 terms is
 * below 0.001 ulp, so the series is at most 34 ulps (2^-120) below its exact value; ln 2 = floor(ln 2 2^120) + [0, 1).
 * The result [lo, hi] (lo <= ln m 2^120 <= hi) is 2 TR_INFO_LOG_RAD + e <= 144 ulps wide (TR_INFO_LOG_RAD = 40).
 *
 * Requests (tr_info_query, CPU, per group g; a, b, c are disjoint coordinate masks; N = the group's kept mass):
 *   ENTROPY              H(A)      = (N ln N - E_A) / N                (A of several bits: the joint entropy)
 *   CONDITIONAL_ENTROPY  H(A | B)  = (E_B - E_AB) / N
 *   MUTUAL_INFORMATION   I(A; B)   = (N ln N + E_AB - E_A - E_B) / N
 *   CONDITIONAL_MI       I(A; B | C) = (E_ABC + E_C - E_AC - E_BC) / N
 *   G_TEST               G = 2 sum O ln(O / E) = 2 N I(A; B)           (the likelihood-ratio statistic, nats)
 *   CHI_SQUARE           X^2 = sum over T(A) x T(B) of (O - E)^2 / E, E = m_a m_b / N, structural zeros included,
 *                        = N (C_AB - 1): an exact rational (tr_info_exact) and its enclosure
 *   CRAMERS_V            V = sqrt(X^2 / (N (min(r, c) - 1))), r, c = the categories (cells) of T(A), T(B); V^2 is rational
 * with r, c and df = (r - 1)(c - 1) reported. Units: nats, or bits (TR_INFO_BITS: H and I divided by an enclosure of
 * ln 2); G, X^2 and V take no unit. With weight masses, G and X^2 are in logical weight units (N is the logical total).
 * Answers: the exact rational bounds num_lo / (den 2^shift) <= value <= num_hi / (den 2^shift) (nats; V: of V^2;
 * den = N for H and I, 1 for G and X^2, min(r, c) - 1 for V^2; shift 112, or 128 for X^2 and V^2), lower bounds clamped
 * at 0 (every statistic is >= 0), and the double enclosure [lo, hi] = [RD(lower), RU(upper)] (then divided by ln 2 in
 * bits, square-rooted for V, outward). Precision: the exact bounds of H, I are at most 2^-104 apart in nats (four sums of
 * width <= (N 144 / 2^8 + 2 cells) 2^-112 each, over N), of G at most 2^-101 N; of X^2 at most N cells 2^-127; the double
 * enclosure adds the outward rounding (at most 2 ulps, 4 in bits).
 * Empty answers (no value, SQL NULL): an absent group, N = 0 (everything excluded, zero weights), V with min(r, c) < 2.
 * Each request has its own status: 0, TR_MATH_INVALID (malformed: an unknown op or unit, empty or overlapping masks),
 * TR_MATH_UNSUPPORTED (a mask outside the state's coordinates: no joint support for it).
 *
 * Exact contingency values (tr_info_exact, CPU): X^2 and V^2 as canonical big rationals (FND-009 carriers): with
 * D = lcm of the B-masses, L = lcm of the A-masses, X^2 = N (T - D L) / (D L), T = sum O^2 (L / m_a)(D / m_b). Budget:
 * support = the cells of T(A | B), bytes = every big number held (bounded by the bit lengths of the masses) before any
 * is computed.
 *
 * Budgets (math_budgets.h), checked before allocation: support build, stage 1 (before the scratch): variables = k,
 * states = the kept joint cells, bytes = the sorting scratch; stage 2 (before the support, once the marginal tables are
 * counted): states = log + chi entries, support = the largest table of a group, bytes = scratch + support. State:
 * states = G x slots, bytes = the sums and group tables. CUDA: bytes = the device copies (tr_info_device_bytes), states
 * = G x slots. Requests: output = n. Errors: TR_MATH_INVALID, _MEMORY, _BUDGET (with the tr_math_refusal), _UNSUPPORTED,
 * _DOMAIN_ERROR, TR_MATH_DEVICE. Constructors return NULL and set *error; objects are immutable. */
#ifndef TORBLES_MATH_INFO_H
#define TORBLES_MATH_INFO_H
#include <stdint.h>
#include "math_num.h"
#include "math_budgets.h"
#include "math_dist.h"
#ifdef __cplusplus
extern "C" {
#endif

#define TR_INFO_ABI 1
#define TR_INFO_F 120                 /* fraction bits of the fixed-point logarithm */
#define TR_INFO_SCALE 112             /* entropy sums: m ln m 2^112 */
#define TR_INFO_CHI_SCALE 128         /* chi-square sums: O^2 2^128 / (m_a m_b) */
#define TR_INFO_LOG_TERMS 23          /* atanh series terms after the first */
#define TR_INFO_LOG_RAD 40            /* the series' error bound in ulps of 2^-120 (proof: below 34) */
#define TR_INFO_CHUNK 64              /* entries per CUDA work item (local accumulation, then atomics) */
enum { TR_INFO_OP_ENTROPY = 1, TR_INFO_OP_CONDITIONAL_ENTROPY = 2, TR_INFO_OP_MUTUAL_INFORMATION = 3,
       TR_INFO_OP_CONDITIONAL_MI = 4, TR_INFO_OP_CHI_SQUARE = 5, TR_INFO_OP_G_TEST = 6, TR_INFO_OP_CRAMERS_V = 7,
       TR_INFO_OPS = 7 };
enum { TR_INFO_NATS = 0, TR_INFO_BITS = 1 };

/* the support as the sums read it (80 bytes): entry keys are g x nslots + slot; slots 0 .. 2^k - 1 are the tables
 * T(S), slots 2^k + p the chi pairs p (tr_info_pair) */
typedef struct {
    uint64_t ngroups, nslots;
    uint64_t nlog, nchi;
    const uint64_t *lkey, *lmass;           /* [nlog] the cells of every table */
    const uint64_t *ckey, *co, *cma, *cmb;  /* [nchi] the cells of T(A | B) of every pair */
} tr_info_view;

/* a support's description (104 bytes) */
typedef struct {
    uint64_t ngroups, present, joint_cells, nlog, nchi, nslots, max_cells, bytes;
    uint64_t total;                         /* the kept mass over all groups (saturating) */
    int32_t k, mass, exclude, npairs;
    int32_t scale[3], wscale;
} tr_info_support_info;

/* one request (32 bytes) and its answer (136 bytes); see the header comment */
typedef struct {
    int32_t op, units;                      /* TR_INFO_OP_*, TR_INFO_NATS / _BITS */
    uint64_t group;
    uint32_t a, b, c, reserved;             /* coordinate masks (bit i: coordinate i); reserved 0 */
} tr_info_request;

typedef struct {
    int32_t status, empty;
    double lo, hi;                          /* the enclosure of the value (bits: the bits value; V: V) */
    tr_dist_i256 num_lo, num_hi;            /* exact bounds num / (den 2^shift), nats (V: of V^2) */
    uint64_t den;
    int32_t shift, squared;                 /* squared: the exact bounds are those of V^2 */
    uint64_t total, rows, cols, df;         /* N; categories of T(a), T(b); (rows - 1)(cols - 1) */
} tr_info_result;

/* ---------------------------------------------------------------- slots and pairs */
TR_NUM_HD uint32_t tr_info_npairs(int32_t k) { return k == 2 ? 1u : k == 3 ? 6u : 0u; }
TR_NUM_HD uint64_t tr_info_nslots(int32_t k) { return ((uint64_t)1 << k) + tr_info_npairs(k); }
/* the p-th unordered pair (A, B) of disjoint nonempty masks, A < B, in the order of U = A | B then A; -1 past the end */
TR_NUM_HD int tr_info_pair_masks(int32_t k, uint32_t p, uint32_t *A, uint32_t *B) {
    uint32_t n = 0, full = (1u << k) - 1;
    for (uint32_t U = 1; U <= full; U++)
        for (uint32_t a = 1; a < U; a++) {
            if ((a & U) != a) continue;
            uint32_t b = U ^ a;
            if (a >= b) continue;
            if (n == p) { *A = a; *B = b; return 0; }
            n++;
        }
    return -1;
}
/* the pair index of {a, b} (either order), or -1 */
TR_NUM_HD int tr_info_pair_index(int32_t k, uint32_t a, uint32_t b) {
    if (a > b) { uint32_t t = a; a = b; b = t; }
    for (uint32_t p = 0; p < tr_info_npairs(k); p++) {
        uint32_t x = 0, y = 0;
        tr_info_pair_masks(k, p, &x, &y);
        if (x == a && y == b) return (int)p;
    }
    return -1;
}

/* ---------------------------------------------------------------- 120-bit fixed point */
#define TR_INFO_ONE ((unsigned __int128)1 << TR_INFO_F)
#define TR_INFO_LN2 ((((unsigned __int128)0x00b17217f7d1cf79ULL) << 64) | (unsigned __int128)0xabc9e3b39803f2f6ULL)
#define TR_INFO_SQRT2 ((((unsigned __int128)0x016a09e667f3bcc9ULL) << 64) | (unsigned __int128)0x08b2fb1366ea957dULL)

/* floor(a b / 2^120) for a, b < 2^124 */
TR_NUM_HD unsigned __int128 tr_info_fmul(unsigned __int128 a, unsigned __int128 b) {
    tr_dist_i256 p = tr_dist_w_mul_u128(a, b);
    return ((unsigned __int128)p.w[3] << 72) | ((unsigned __int128)p.w[2] << 8) | (unsigned __int128)(p.w[1] >> 56);
}
/* floor(x / k) for 1 <= k < 2^32 by 32-bit digits (no 128-bit division) */
TR_NUM_HD unsigned __int128 tr_info_divsmall(unsigned __int128 x, uint32_t k) {
    uint64_t hi = (uint64_t)(x >> 64), lo = (uint64_t)x;
    uint64_t qh = hi / k, r = hi % k;
    uint64_t m = (r << 32) | (lo >> 32), q1 = m / k;
    r = m % k;
    uint64_t l = (r << 32) | (lo & 0xffffffffULL), q0 = l / k;
    return ((unsigned __int128)qh << 64) | (unsigned __int128)((q1 << 32) | q0);
}
/* floor(num 2^120 / den) for num < den < 2^123 (restoring division, exact) */
TR_NUM_HD unsigned __int128 tr_info_fdiv(unsigned __int128 num, unsigned __int128 den) {
    unsigned __int128 q = 0, r = num;
    for (int i = 0; i < TR_INFO_F; i++) {
        r <<= 1; q <<= 1;
        if (r >= den) { r -= den; q |= 1; }
    }
    return q;
}
/* lo <= ln(m) 2^120 <= hi for 1 <= m < 2^64 (m = 0 is treated as 1: 0 ln 0 = 0); see the header comment */
TR_NUM_HD void tr_info_ln(uint64_t m, unsigned __int128 *lo, unsigned __int128 *hi) {
    if (m <= 1) { *lo = 0; *hi = 0; return; }
    int e = 63;
    while (!((m >> e) & 1)) e--;
    unsigned __int128 F = (unsigned __int128)m << (TR_INFO_F - e);          /* f 2^120, exact: f in [1, 2) */
    if (F > TR_INFO_SQRT2) { F = (unsigned __int128)m << (TR_INFO_F - 1 - e); e += 1; }   /* f / 2, exact */
    const int neg = F < TR_INFO_ONE;
    unsigned __int128 num = neg ? TR_INFO_ONE - F : F - TR_INFO_ONE, den = F + TR_INFO_ONE;
    unsigned __int128 Z = tr_info_fdiv(num, den), W = tr_info_fmul(Z, Z), T = Z, S = Z;
    for (uint32_t j = 1; j <= TR_INFO_LOG_TERMS; j++) {
        T = tr_info_fmul(T, W);
        S += tr_info_divsmall(T, 2 * j + 1);
    }
    unsigned __int128 blo = (unsigned __int128)(uint64_t)e * TR_INFO_LN2, bhi = blo + (unsigned __int128)(uint64_t)e;
    if (!neg) { *lo = blo + 2 * S; *hi = bhi + 2 * S + 2 * TR_INFO_LOG_RAD; }
    else { *lo = blo - (2 * S + 2 * TR_INFO_LOG_RAD); *hi = bhi - 2 * S; }
}

/* ---------------------------------------------------------------- the per-cell terms (CPU loop and CUDA kernels) */
/* floor (up = 0) or ceil (up = 1) of p / 2^8 for a nonnegative 256-bit p */
TR_NUM_HD tr_dist_i256 tr_info_shr8(tr_dist_i256 p, int up) {
    tr_dist_i256 r;
    int rem = (p.w[0] & 0xff) != 0;
    for (int i = 0; i < 4; i++) r.w[i] = (p.w[i] >> 8) | (i < 3 ? p.w[i + 1] << 56 : 0);
    if (up && rem) {
        for (int i = 0; i < 4; i++) { r.w[i] += 1; if (r.w[i]) break; }
    }
    return r;
}
/* the bounds L <= m ln(m) 2^112 <= H of one cell (m < 2^64) */
TR_NUM_HD void tr_info_log_term(uint64_t m, tr_dist_i256 *L, tr_dist_i256 *H) {
    unsigned __int128 lo, hi;
    tr_info_ln(m, &lo, &hi);
    *L = tr_info_shr8(tr_dist_w_mul_u128((unsigned __int128)m, lo), 0);
    *H = tr_info_shr8(tr_dist_w_mul_u128((unsigned __int128)m, hi), 1);
}
TR_NUM_HD int tr_info_clz64(uint64_t x) {
#if defined(__CUDA_ARCH__)
    return __clzll((long long)x);
#else
    return x ? __builtin_clzll(x) : 64;
#endif
}
/* floor((u1 2^64 + u0) / v) and the remainder for u1 < v (v > 0), with 64-bit operations only (no 128-bit division:
 * Knuth's algorithm D with 32-bit digits, Hacker's Delight divlu; the wrapping products are exact modulo 2^64) */
TR_NUM_HD uint64_t tr_info_divlu(uint64_t u1, uint64_t u0, uint64_t v, uint64_t *r) {
    const uint64_t b = UINT64_C(1) << 32;
    int s = tr_info_clz64(v);
    v <<= s;
    uint64_t vn1 = v >> 32, vn0 = v & 0xffffffffULL;
    uint64_t un32 = (u1 << s) | (s ? u0 >> (64 - s) : 0), un10 = u0 << s;
    uint64_t un1 = un10 >> 32, un0 = un10 & 0xffffffffULL;
    uint64_t q1 = un32 / vn1, rhat = un32 - q1 * vn1;
    while (q1 >= b || q1 * vn0 > b * rhat + un1) { q1--; rhat += vn1; if (rhat >= b) break; }
    uint64_t un21 = un32 * b + un1 - q1 * v;
    uint64_t q0 = un21 / vn1;
    rhat = un21 - q0 * vn1;
    while (q0 >= b || q0 * vn0 > b * rhat + un0) { q0--; rhat += vn1; if (rhat >= b) break; }
    *r = (un21 * b + un0 - q0 * v) >> s;
    return q1 * b + q0;
}
/* floor(x 2^128 / d) and whether it is inexact, for x < d < 2^128 */
TR_NUM_HD unsigned __int128 tr_info_frac128(unsigned __int128 x, unsigned __int128 d, int *inexact) {
    if ((d >> 64) == 0) {                                                /* x < d < 2^64: two digits */
        uint64_t dd = (uint64_t)d, r1, r0;
        uint64_t q1 = tr_info_divlu((uint64_t)x, 0, dd, &r1);
        uint64_t q0 = tr_info_divlu(r1, 0, dd, &r0);
        *inexact = r0 != 0;
        return ((unsigned __int128)q1 << 64) | q0;
    }
    unsigned __int128 q = 0, r = x;
    for (int i = 0; i < 128; i++) {
        int top = (int)(r >> 127);
        r <<= 1; q <<= 1;
        if (top || r >= d) { r -= d; q |= 1; }                           /* 2r - d, exact modulo 2^128 */
    }
    *inexact = r != 0;
    return q;
}
/* the bounds L <= O^2 2^128 / (m_a m_b) <= H of one cell (1 <= O <= m_a, m_b < 2^64) */
TR_NUM_HD void tr_info_chi_term(uint64_t o, uint64_t ma, uint64_t mb, tr_dist_i256 *L, tr_dist_i256 *H) {
    unsigned __int128 x = (unsigned __int128)o * o, d = (unsigned __int128)ma * mb;
    tr_dist_i256 z = tr_dist_w_zero();
    if (x >= d) { z.w[2] = 1; *L = z; *H = z; return; }                    /* O = m_a = m_b: exactly 1 */
    int inexact = 0;
    unsigned __int128 q = tr_info_frac128(x, d, &inexact);
    L->w[0] = (uint64_t)q; L->w[1] = (uint64_t)(q >> 64); L->w[2] = L->w[3] = 0;
    unsigned __int128 q1 = q + (unsigned __int128)inexact;
    H->w[0] = (uint64_t)q1; H->w[1] = (uint64_t)(q1 >> 64); H->w[2] = (inexact && q1 == 0) ? 1 : 0; H->w[3] = 0;
}
/* *acc += x on 4 words (exact: every sum stays below 2^255) */
TR_NUM_HD void tr_info_add256(uint64_t *acc, tr_dist_i256 x) {
    uint64_t c = 0;
    for (int i = 0; i < 4; i++) {
        uint64_t t = acc[i] + c, c1 = t < c, y = t + x.w[i], c2 = y < t;
        acc[i] = y; c = c1 | c2;
    }
}
/* the bytes of tr_mgpu_info_sums' device copies: the entries and the sums */
static inline uint64_t tr_info_device_bytes(const tr_info_view *v) {
    uint64_t b = tr_budget_add(tr_budget_mul(v->nlog, 16), tr_budget_mul(v->nchi, 32));
    return tr_budget_add(b, tr_budget_mul(tr_budget_mul(v->ngroups, v->nslots), 64));
}
/* a view whose sizes and pointers are consistent (host code of both libraries) */
static inline int tr_info_view_valid(const tr_info_view *v) {
    if (!v || v->ngroups < 1 || v->nslots < 2 || v->nslots > 14) return 0;
    if (v->nlog && (!v->lkey || !v->lmass)) return 0;
    if (v->nchi && (!v->ckey || !v->co || !v->cma || !v->cmb)) return 0;
    return tr_budget_mul(v->ngroups, v->nslots) != TR_BUDGET_UNKNOWN;
}

/* ---------------------------------------------------------------- objects (libalg) */
typedef struct tr_info_support_s tr_info_support;
typedef struct tr_info_state_s tr_info_state;

/* the support of histogram h with mass TR_DIST_MASS_COUNT / _WEIGHT and the NULL exclusion mask */
tr_info_support *tr_info_support_build(const tr_dist_hist *h, int32_t mass, uint32_t exclude,
                                       const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error);
/* the first-stage need of that build (variables, states, bytes) */
int32_t tr_info_support_need(const tr_dist_hist *h, int32_t mass, uint32_t exclude, tr_math_need *need);
int32_t tr_info_support_get_view(const tr_info_support *, tr_info_view *);
int32_t tr_info_support_get_info(const tr_info_support *, tr_info_support_info *);
/* group g: its selected rows, its kept mass N and the cells of each table T(S), S = 0 .. 2^k - 1 (cells[2^k]) */
int32_t tr_info_support_group(const tr_info_support *, uint64_t g, uint64_t *rows, uint64_t *total, uint64_t *cells);
void tr_info_support_free(tr_info_support *);
/* the CPU sums of a view: sums[(g nslots + slot) 8 + 0..3] the lower, + 4..7 the upper bound (256-bit, zeroed first) */
int32_t tr_info_sums(const tr_info_view *, int64_t *sums);
/* the state from a support and its sums (NULL: computed here; else the CUDA readback, checked) */
tr_info_state *tr_info_state_build(const tr_info_support *, const int64_t *sums, const tr_math_budget *budget,
                                   tr_math_refusal *refusal, int32_t *error);
int32_t tr_info_state_equal(const tr_info_state *, const tr_info_state *);   /* 1 / 0, INVALID */
uint64_t tr_info_state_bytes(const tr_info_state *);
void tr_info_state_free(tr_info_state *);
/* n requests (output = n); 0 or TR_MATH_BUDGET / _INVALID for the batch, each answer with its own status */
int32_t tr_info_query(const tr_info_state *, const tr_info_request *requests, uint64_t n, tr_info_result *out,
                      const tr_math_budget *budget, tr_math_refusal *refusal);
/* the exact X^2 (op TR_INFO_OP_CHI_SQUARE) or V^2 (TR_INFO_OP_CRAMERS_V) of masks a, b in group g as a canonical
 * fraction *num / *den (FND-009 big integers, owned by the caller); an empty answer leaves both NULL */
struct tr_bigint_s;
int32_t tr_info_exact(const tr_info_support *, uint64_t g, int32_t op, uint32_t a, uint32_t b,
                      const tr_math_budget *budget, tr_math_refusal *refusal, struct tr_bigint_s **num,
                      struct tr_bigint_s **den);
uint64_t tr_info_device_need(const tr_info_view *);   /* tr_info_device_bytes, exported */
/* the bounds of one cell's term for tests: kind 1 = m ln(m) 2^112 (a = m), kind 2 = O^2 2^128 / (m_a m_b) (a, b, c =
 * O, m_a, m_b); out[0..3] the lower, out[4..7] the upper bound (256-bit, least significant first) */
int32_t tr_info_term_bounds(int32_t kind, uint64_t a, uint64_t b, uint64_t c, uint64_t out[8]);
/* the certified bounds of tr_info_ln (out: lo low word, lo high word, hi low, hi high), for tests and bindings */
int32_t tr_info_ln_bounds(uint64_t m, uint64_t out[4]);

/* ---------------------------------------------------------------- the CUDA library (math_info_gpu.cu) */
/* the sums of tr_info_sums on the device: one work item per TR_INFO_CHUNK entries (local 256-bit sums while the key
 * stays the same, then exact 256-bit atomics). Budget: states = G x slots, bytes = tr_info_device_bytes. */
int tr_mgpu_info_sums(const tr_info_view *view, const tr_math_budget *budget, int64_t *sums, tr_math_refusal *refusal);
#ifdef __cplusplus
}
#endif
#endif
