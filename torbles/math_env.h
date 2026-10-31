/* ENV-001: univariate affine / quadratic group envelopes: the winning group on every piece of a parameter interval
 * and the exact breakpoints, with a declared tie policy.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Input. One function of a parameter t per group g < ngroups, f_g(t) = a_g + b_g t + c_g t^2 with exact rational
 * coefficients (FND-009 tr_bigq; e.g. a group's RESP-001 response function from retained moments, or first-slice
 * polynomials in one variable), or none: an empty group (absent: it never wins and appears nowhere). A parameter
 * interval [lo, hi], each end an exact rational or unbounded (lo = hi: one point). Lower envelope E(t) = min_g f_g(t)
 * over the present groups, or the upper envelope (max). A coefficient of degree 3 or more, or a polynomial in another
 * variable (a parameter dimension above 1), is TR_MATH_UNSUPPORTED.
 *
 * Classes and ties. Identical functions (equal canonical forms (A + B t + C t^2) / d, gcd 1, d > 0) form one class
 * whose members are coincident on every piece; distinct classes agree only at finitely many points. The tie policy
 * picks the winning group among the groups attaining the envelope: TR_ENV_TIE_SMALLEST (the smallest group id) or
 * TR_ENV_TIE_LARGEST; every region also reports all of its classes and member groups.
 *
 * Regions. The interval is partitioned, in increasing order, into points and open pieces:
 *   [point lo] piece point piece ... point piece [point hi]
 * (a bounded end is a boundary point; lo = hi is a single point). On a piece exactly one class attains the envelope
 * (its winner). A point is a boundary or a breakpoint: a t where two or more classes attain the envelope (where the
 * winning class changes, three or more functions meet, or a function touches the envelope); its classes are exactly
 * those attaining it there. Breakpoints are exact: a rational (an affine difference, a quadratic difference with a
 * square discriminant or a double root), or an algebraic number: the smaller or larger root of a primitive integer
 * quadratic q0 + q1 t + q2 t^2 (q2 > 0, non-square discriminant: irreducible) together with a certified isolating
 * interval (lo, hi) with rational ends from INV-001 (tr_inv_isolate: Sturm-certified, exactly one root of q inside,
 * a sign change of q; refinable, tr_env_point_refine). An algebraic breakpoint is never given as a number.
 *
 * Construction (exact, FND-009 big integers). The classes are sorted and deduplicated; a left-to-right sweep keeps the
 * class w attaining the envelope just right of the current position x and finds the next point as the smallest root
 * greater than x (and below hi) of the differences f_c - f_w over all classes c != w (exact roots: rational, or
 * algebraic compared exactly by sign rules for p + s sqrt(D) and r + u sqrt(D1) + v sqrt(D2), never by rounding);
 * the classes attaining it are those whose difference has that root; the next winner is the one smallest just to its
 * right (by the sign of the derivative of the difference, then of its leading coefficient). At -inf the winner is the
 * class smallest as t -> -inf (asymptotic order of the coefficients). Each formed difference is one iteration.
 *
 * Lookups (CPU and CUDA, the same shared functions below): at query points t = u / v (exact rationals), mode
 * TR_ENV_SEARCH answers the region containing t by a binary search over the sorted points, each comparison of t with a
 * point being the exact sign of an integer polynomial at t (t - r for a rational point; for an algebraic one t - lo,
 * t - hi of its isolating interval and then the sign of q(t), compared with q's sign at lo); mode TR_ENV_ALL
 * evaluates every class at t exactly (signs of d_b S_c - d_c S_b, S the homogenized numerators) for verification.
 * Signs come from residues modulo k primes in (2^61, 2^62) (INV-001's verified table) with Garner's mixed-radix digits,
 * k from a proven bound |S| < 2^B, 61 k >= B + 2 (more than TR_ENV_PRIMES_MAX moduli: TR_MATH_OVERFLOW). A query
 * outside [lo, hi] answers TR_ENV_Q_OUTSIDE, any query of an empty envelope TR_ENV_Q_EMPTY.
 *
 * Budgets (math_budgets.h): a build checks variables = 1, degree (<= 2), support = present groups, states = groups and
 * bytes = the class tables before anything is allocated; output = regions (counted exactly by the sweep: an envelope
 * with more regions than the limit is refused with that exact count), iterations = the differences formed and
 * INV-001's exact evaluations when it isolates the algebraic points (within the iterations the sweep left; refused when
 * the work reaches limit + 1, reported as that need), bytes again as the region tables grow (tr_env_info.bytes is the
 * largest amount checked). Lookups: output = queries, bytes = the job (CUDA: its device copies,
 * tr_env_view_device_bytes). Every big number is budgeted by FND-009 (bytes / 8 limbs).
 *
 * Errors: TR_MATH_INVALID, _MEMORY, _OVERFLOW, _BUDGET (with the tr_math_refusal), _UNSUPPORTED, TR_MATH_DOMAIN_ERROR
 * (a polynomial of another context). Objects are immutable and individually owned; constructors return NULL and set
 * *error, inputs unchanged. */
#ifndef TORBLES_MATH_ENV_H
#define TORBLES_MATH_ENV_H
#include <stdint.h>
#include <stddef.h>
#include "math_num.h"
#include "math_budgets.h"
#include "math_carriers.h"
#include "math_inv.h"
#include "mathematics.h"
#include "math_bank.h"
#ifdef __cplusplus
extern "C" {
#endif

#define TR_ENV_ABI 1
#define TR_ENV_PRIMES_MAX 64            /* moduli of the lookups: values below 2^3900 */
#define TR_ENV_NONE UINT64_MAX
#define TR_ENV_SLOTS 7                  /* residues per point: rational (c0, c1); algebraic (l0, l1, h0, h1, q0, q1, q2) */
enum { TR_ENV_LOWER = 0, TR_ENV_UPPER = 1 };
enum { TR_ENV_TIE_SMALLEST = 0, TR_ENV_TIE_LARGEST = 1 };
enum { TR_ENV_EMPTY = 0, TR_ENV_FINITE = 1 };               /* envelope kinds */
enum { TR_ENV_PIECE = 1, TR_ENV_POINT = 2 };                /* region kinds */
enum { TR_ENV_RATIONAL = 1, TR_ENV_ALGEBRAIC = 2 };         /* point kinds */
enum { TR_ENV_AT_LO = 1, TR_ENV_AT_HI = 2 };                /* boundary flags */
enum { TR_ENV_SEARCH = 1, TR_ENV_ALL = 2 };                 /* lookup modes */
enum { TR_ENV_Q_OK = 0, TR_ENV_Q_OUTSIDE = 1, TR_ENV_Q_EMPTY = 2 };   /* answer status (else a negative error) */
enum { TR_ENV_PER_NONE = -1, TR_ENV_PER_ROWS = -2 };       /* tr_env_from_bank divisors */

/* ------------------------------------------------------------------ structs crossing the ctypes boundary */
typedef struct {
    uint64_t groups, present, classes;   /* groups given, present groups, distinct functions */
    uint64_t regions, pieces, points;    /* regions = pieces + points */
    uint64_t iterations;                 /* differences formed by the sweep */
    uint64_t bytes;
    int32_t kind, upper, tie, degree;    /* TR_ENV_EMPTY / FINITE; lower 0 / upper 1; tie policy; largest degree */
    int32_t lo_bounded, hi_bounded;
    int32_t algebraic, reserved;         /* points that are algebraic (irrational) */
} tr_env_info;                           /* 96 bytes */

typedef struct {
    int32_t kind, boundary;      /* TR_ENV_PIECE / POINT; a POINT's TR_ENV_AT_LO | TR_ENV_AT_HI */
    uint64_t point;              /* POINT: its point; PIECE: the point at its left end (TR_ENV_NONE: unbounded) */
    uint64_t right;              /* PIECE: the point at its right end (TR_ENV_NONE: unbounded) */
    uint64_t classes;            /* classes attaining the envelope here (1 on a piece) */
    uint64_t cls;                /* PIECE: its class; POINT: the first of its classes */
    uint64_t winner;             /* the tie policy's group */
    uint64_t members;            /* groups attaining the envelope here */
} tr_env_region;                 /* 56 bytes */

typedef struct {
    int32_t kind;                /* TR_ENV_RATIONAL / ALGEBRAIC */
    int32_t root;                /* ALGEBRAIC: 0 the smaller, 1 the larger root of its polynomial */
    int32_t boundary;            /* TR_ENV_AT_LO | TR_ENV_AT_HI */
    int32_t sign_lo;             /* ALGEBRAIC: the sign of its polynomial at the isolating interval's lower end */
    uint64_t region;             /* its region */
} tr_env_point_info;             /* 24 bytes */

typedef struct {
    uint64_t members, winner;    /* member groups; the tie policy's member */
    int32_t degree, reserved;
} tr_env_class_info;             /* 24 bytes */

/* one lookup answer */
typedef struct {
    int32_t status;              /* TR_ENV_Q_OK / OUTSIDE / EMPTY, or a negative error code */
    int32_t kind;                /* TR_ENV_PIECE / POINT (ALL mode: POINT when two or more classes attain the minimum) */
    uint64_t region;             /* SEARCH: the region containing t; ALL: TR_ENV_NONE */
    uint64_t winner;             /* the winning group (tie policy) */
    uint64_t tied;               /* classes attaining the envelope at t */
} tr_env_answer;                 /* 32 bytes */

/* the lookup job as the CPU evaluator and the CUDA kernel read it (pointers into a tr_env_lookup) */
typedef struct {
    uint64_t npoints, nbreaks, nregions, nclasses;
    uint32_t nprimes, limbs;     /* moduli; limbs of every query numerator / denominator magnitude (>= 1) */
    int32_t mode, kind;          /* TR_ENV_SEARCH / ALL; TR_ENV_EMPTY / FINITE */
    int32_t lo_bounded, hi_bounded, tie, reserved;
    const tr_gfp *fields;        /* k */
    const uint64_t *garner;      /* k x k: garner[j * k + t] = p_t^-1 mod p_j (t < j) */
    const uint64_t *half;        /* k mixed-radix digits of (M - 1) / 2 */
    const int32_t *bkind;        /* nbreaks: TR_ENV_RATIONAL / ALGEBRAIC */
    const int8_t *bsign;         /* nbreaks: ALGEBRAIC: the sign of q at the isolating interval's lower end */
    const uint64_t *bres;        /* k x nbreaks x TR_ENV_SLOTS residues */
    const uint64_t *point_region;   /* nbreaks */
    const uint64_t *piece_region;   /* nbreaks + 1: the piece before point i (i < nbreaks) or after the last; NONE */
    const uint64_t *rwinner;     /* nregions */
    const uint64_t *rclasses;    /* nregions */
    const uint64_t *cres;        /* ALL: k x nclasses x 4 residues (A, B, C, d) of the internal forms */
    const uint64_t *cwinner;     /* ALL: nclasses: the tie policy's member */
    const int8_t *qsign;         /* npoints: the sign of u */
    const uint64_t *qnum;        /* npoints x limbs: |u| */
    const uint64_t *qden;        /* npoints x limbs: v > 0 */
} tr_env_view;                   /* 184 bytes */

typedef struct tr_env_s tr_env;
typedef struct tr_env_lookup_s tr_env_lookup;

/* ------------------------------------------------------------------ shared per-query functions (CPU and kernels) */
/* the sign (-1, 0, 1) of the integer S whose residues are r[0..k-1] (|S| < M / 2): Garner's digits compared with
 * those of (M - 1) / 2 from the most significant one */
TR_NUM_HD int tr_env_sign_residues(const tr_env_view *v, const uint64_t *r) {
    uint64_t dig[TR_ENV_PRIMES_MAX];
    const uint32_t k = v->nprimes;
    int zero = 1;
    for (uint32_t j = 0; j < k; j++) {
        const tr_gfp *F = &v->fields[j];
        const uint64_t *g = v->garner + (uint64_t)j * k;
        uint64_t x = r[j];
        if (x) zero = 0;
        for (uint32_t t = 0; t < j; t++) {
            uint64_t dt = dig[t] >= F->p ? dig[t] - F->p : dig[t];   /* p_t < 2 p_j: one subtraction reduces */
            x = tr_gfp_mul_hd(F, tr_gfp_sub_hd(F, x, dt), g[t]);
        }
        dig[j] = x;
    }
    if (zero) return 0;
    for (uint32_t j = k; j-- > 0;)
        if (dig[j] != v->half[j]) return dig[j] < v->half[j] ? 1 : -1;
    return 1;
}
/* the sign of c0 v + c1 u: residues of (c0, c1) for prime j at base + j * stride, of u and v in ur, vr */
TR_NUM_HD int tr_env_sign_lin(const tr_env_view *v, const uint64_t *base, uint64_t stride, const uint64_t *ur,
                              const uint64_t *vr) {
    uint64_t r[TR_ENV_PRIMES_MAX];
    for (uint32_t j = 0; j < v->nprimes; j++) {
        const tr_gfp *F = &v->fields[j];
        const uint64_t *c = base + (uint64_t)j * stride;
        r[j] = tr_gfp_add_hd(F, tr_gfp_mul_hd(F, c[0], vr[j]), tr_gfp_mul_hd(F, c[1], ur[j]));
    }
    return tr_env_sign_residues(v, r);
}
/* the sign of q0 v^2 + q1 u v + q2 u^2 */
TR_NUM_HD int tr_env_sign_quad(const tr_env_view *v, const uint64_t *base, uint64_t stride, const uint64_t *ur,
                               const uint64_t *vr) {
    uint64_t r[TR_ENV_PRIMES_MAX];
    for (uint32_t j = 0; j < v->nprimes; j++) {
        const tr_gfp *F = &v->fields[j];
        const uint64_t *c = base + (uint64_t)j * stride;
        uint64_t vv = tr_gfp_mul_hd(F, vr[j], vr[j]), uv = tr_gfp_mul_hd(F, ur[j], vr[j]), uu = tr_gfp_mul_hd(F, ur[j], ur[j]);
        r[j] = tr_gfp_add_hd(F, tr_gfp_add_hd(F, tr_gfp_mul_hd(F, c[0], vv), tr_gfp_mul_hd(F, c[1], uv)),
                             tr_gfp_mul_hd(F, c[2], uu));
    }
    return tr_env_sign_residues(v, r);
}
/* sign(t - point b): -1, 0, 1, or 2 when an algebraic point's polynomial vanishes at t (impossible for a valid job) */
TR_NUM_HD int tr_env_cmp_break(const tr_env_view *v, uint64_t b, const uint64_t *ur, const uint64_t *vr) {
    const uint64_t stride = v->nbreaks * TR_ENV_SLOTS, *base = v->bres + b * TR_ENV_SLOTS;
    if (v->bkind[b] == TR_ENV_RATIONAL) return tr_env_sign_lin(v, base, stride, ur, vr);
    if (tr_env_sign_lin(v, base, stride, ur, vr) <= 0) return -1;            /* t <= lo < the root */
    if (tr_env_sign_lin(v, base + 2, stride, ur, vr) >= 0) return 1;         /* t >= hi > the root */
    int s = tr_env_sign_quad(v, base + 4, stride, ur, vr);
    if (s == 0) return 2;
    return s == v->bsign[b] ? -1 : 1;                                       /* on lo's side of the sign change */
}
/* one query (u = (-1)^neg |u|, v > 0): the answer of mode SEARCH or ALL */
TR_NUM_HD void tr_env_answer_one(const tr_env_view *v, uint64_t i, tr_env_answer *out) {
    out->status = 0; out->kind = 0; out->region = TR_ENV_NONE; out->winner = TR_ENV_NONE; out->tied = 0;
    if (v->kind == TR_ENV_EMPTY) { out->status = TR_ENV_Q_EMPTY; return; }
    uint64_t ur[TR_ENV_PRIMES_MAX], vr[TR_ENV_PRIMES_MAX];
    const uint64_t *num = v->qnum + i * (uint64_t)v->limbs, *den = v->qden + i * (uint64_t)v->limbs;
    for (uint32_t j = 0; j < v->nprimes; j++) {
        const tr_gfp *F = &v->fields[j];
        ur[j] = tr_inv_limbs_mod(F, num, v->limbs);
        vr[j] = tr_inv_limbs_mod(F, den, v->limbs);
        if (v->qsign[i] < 0) ur[j] = tr_gfp_neg_hd(F, ur[j]);
    }
    const uint64_t m = v->nbreaks;
    if (v->lo_bounded && m && tr_env_cmp_break(v, 0, ur, vr) < 0) { out->status = TR_ENV_Q_OUTSIDE; return; }
    if (v->hi_bounded && m && tr_env_cmp_break(v, m - 1, ur, vr) > 0) { out->status = TR_ENV_Q_OUTSIDE; return; }
    if (v->mode == TR_ENV_SEARCH) {
        uint64_t lo = 0, hi = m;
        int at = 0;                                                          /* t equals point hi (when hi < m) */
        while (lo < hi) {                                                    /* the first point >= t */
            uint64_t mid = lo + (hi - lo) / 2;
            int c = tr_env_cmp_break(v, mid, ur, vr);
            if (c == 2) { out->status = -1; return; }
            if (c <= 0) { hi = mid; at = c == 0; } else lo = mid + 1;
        }
        uint64_t r = at ? v->point_region[lo] : v->piece_region[lo];
        if (r == TR_ENV_NONE) { out->status = TR_ENV_Q_OUTSIDE; return; }
        out->kind = at ? TR_ENV_POINT : TR_ENV_PIECE; out->region = r;
        out->winner = v->rwinner[r]; out->tied = v->rclasses[r];
        return;
    }
    /* ALL: every class at t. S_c = A v^2 + B u v + C u^2, so f_c(t) = S_c / (d_c v^2) and f_c < f_b iff
     * d_b S_c - d_c S_b < 0 (the internal forms: an upper envelope's are negated, so the minimum is always taken) */
    uint64_t sb[TR_ENV_PRIMES_MAX], db[TR_ENV_PRIMES_MAX], sc[TR_ENV_PRIMES_MAX], dc[TR_ENV_PRIMES_MAX];
    uint64_t r[TR_ENV_PRIMES_MAX];
    const uint64_t cstride = v->nclasses * 4;
    uint64_t tied = 0, win = TR_ENV_NONE;
    for (uint64_t c = 0; c < v->nclasses; c++) {
        for (uint32_t j = 0; j < v->nprimes; j++) {
            const tr_gfp *F = &v->fields[j];
            const uint64_t *cr = v->cres + (uint64_t)j * cstride + c * 4;
            uint64_t vv = tr_gfp_mul_hd(F, vr[j], vr[j]), uv = tr_gfp_mul_hd(F, ur[j], vr[j]), uu = tr_gfp_mul_hd(F, ur[j], ur[j]);
            sc[j] = tr_gfp_add_hd(F, tr_gfp_add_hd(F, tr_gfp_mul_hd(F, cr[0], vv), tr_gfp_mul_hd(F, cr[1], uv)),
                                  tr_gfp_mul_hd(F, cr[2], uu));
            dc[j] = cr[3];
            if (c) r[j] = tr_gfp_sub_hd(F, tr_gfp_mul_hd(F, db[j], sc[j]), tr_gfp_mul_hd(F, dc[j], sb[j]));
        }
        int sg = c ? tr_env_sign_residues(v, r) : -1;
        if (sg < 0) {                                                       /* a new minimum */
            for (uint32_t j = 0; j < v->nprimes; j++) { sb[j] = sc[j]; db[j] = dc[j]; }
            tied = 1; win = v->cwinner[c];
        } else if (sg == 0) {                                               /* tied with the minimum */
            tied++;
            if (v->tie == TR_ENV_TIE_SMALLEST ? v->cwinner[c] < win : v->cwinner[c] > win) win = v->cwinner[c];
        }
    }
    out->kind = tied >= 2 ? TR_ENV_POINT : TR_ENV_PIECE;
    out->winner = win; out->tied = tied;
}

/* bytes the CUDA lookup copies to / allocates on the device (saturating) */
static inline uint64_t tr_env_view_device_bytes(const tr_env_view *v) {
    uint64_t k = v->nprimes, b = tr_budget_mul(k, sizeof(tr_gfp));
    b = tr_budget_add(b, tr_budget_mul(tr_budget_mul(k, k), 8));
    b = tr_budget_add(b, tr_budget_mul(k, 8));
    b = tr_budget_add(b, tr_budget_mul(v->nbreaks, 4 + 1 + 8 + 8));
    b = tr_budget_add(b, tr_budget_mul(tr_budget_mul(k, v->nbreaks), 8 * TR_ENV_SLOTS));
    b = tr_budget_add(b, tr_budget_add(8, tr_budget_mul(v->nregions, 16)));
    if (v->mode == TR_ENV_ALL) b = tr_budget_add(b, tr_budget_add(tr_budget_mul(tr_budget_mul(k, v->nclasses), 32),
                                                                  tr_budget_mul(v->nclasses, 8)));
    b = tr_budget_add(b, tr_budget_mul(v->npoints, 1 + sizeof(tr_env_answer)));
    return tr_budget_add(b, tr_budget_mul(tr_budget_mul(v->npoints, v->limbs), 16));
}
/* a well-formed view (checked by the CPU evaluator and the CUDA entry point alike) */
static inline int tr_env_view_valid(const tr_env_view *v) {
    if (!v || (v->mode != TR_ENV_SEARCH && v->mode != TR_ENV_ALL) || (v->tie != TR_ENV_TIE_SMALLEST && v->tie != TR_ENV_TIE_LARGEST))
        return 0;
    if (v->kind == TR_ENV_EMPTY) return v->npoints == 0 || (v->qsign != NULL);
    if (v->kind != TR_ENV_FINITE || v->nprimes < 1 || v->nprimes > TR_ENV_PRIMES_MAX || v->limbs < 1 ||
        v->limbs > TR_INV_POINT_LIMBS_MAX || !v->fields || !v->garner || !v->half || !v->rwinner || !v->rclasses ||
        !v->piece_region || (v->nbreaks && (!v->bkind || !v->bsign || !v->bres || !v->point_region)) ||
        (v->npoints && (!v->qsign || !v->qnum || !v->qden)) || v->nclasses < 1 ||
        (v->mode == TR_ENV_ALL && (!v->cres || !v->cwinner))) return 0;
    for (uint32_t j = 0; j < v->nprimes; j++) if (!tr_gfp_consistent(&v->fields[j])) return 0;
    for (uint64_t i = 0; i <= v->nbreaks; i++) if (v->piece_region[i] != TR_ENV_NONE && v->piece_region[i] >= v->nregions) return 0;
    for (uint64_t i = 0; i < v->nbreaks; i++) if (v->point_region[i] >= v->nregions) return 0;
    return 1;
}

/* ------------------------------------------------------------------ envelopes (libalg) */
/* the a-priori needs of a build: variables 1, degree, support = present groups, states = groups, bytes */
int32_t tr_env_need(uint64_t ngroups, uint64_t present, int32_t degree, tr_math_need *need);
/* ngroups functions: coef[g * ncoef + k] the coefficient of t^k (NULL = 0; ncoef <= 8, a nonzero coefficient of
 * degree >= 3 is TR_MATH_UNSUPPORTED); present[g] (NULL: every group) 0 = an empty group; lo / hi NULL = unbounded */
tr_env *tr_env_build(uint64_t ngroups, int32_t ncoef, const tr_bigq *const *coef, const uint8_t *present,
                     const tr_bigq *lo, const tr_bigq *hi, int32_t upper, int32_t tie, const tr_math_budget *budget,
                     tr_math_refusal *refusal, int32_t *error);
/* first-slice polynomials in one variable slot (NULL = an empty group; a term in another variable: UNSUPPORTED) */
tr_env *tr_env_from_polys(uint64_t ngroups, const tr_math_poly *const *polys, int32_t variable, const tr_bigq *lo,
                          const tr_bigq *hi, int32_t upper, int32_t tie, const tr_math_budget *budget,
                          tr_math_refusal *refusal, int32_t *error);
/* response functions from retained moments: f_g(t) = sum_k mult[k] sum[g][term[k]] t^k / den_g (term[k] < 0: the
 * coefficient is 0), den_g = 1 (per TR_ENV_PER_NONE), rows[g] (TR_ENV_PER_ROWS) or count[g][per]; a group without
 * rows, or with den_g = 0, is empty */
tr_env *tr_env_from_bank(const tr_bank *bank, const int32_t *term, const int64_t *mult, int32_t per,
                         const tr_bigq *lo, const tr_bigq *hi, int32_t upper, int32_t tie, const tr_math_budget *budget,
                         tr_math_refusal *refusal, int32_t *error);
int32_t tr_env_get_info(const tr_env *, tr_env_info *);
int32_t tr_env_region_get(const tr_env *, uint64_t region, tr_env_region *);
/* the classes of a region (PIECE: its class), increasing; capacity >= the region's classes */
int32_t tr_env_region_classes(const tr_env *, uint64_t region, uint64_t *out, uint64_t capacity);
int32_t tr_env_class_get(const tr_env *, uint64_t cls, tr_env_class_info *);
int32_t tr_env_class_members(const tr_env *, uint64_t cls, uint64_t *out, uint64_t capacity);
const tr_bigq *tr_env_class_coef(const tr_env *, uint64_t cls, int32_t k);   /* the given coefficient of t^k (borrowed) */
int32_t tr_env_point_get(const tr_env *, uint64_t point, tr_env_point_info *);
const tr_bigq *tr_env_point_value(const tr_env *, uint64_t point);            /* RATIONAL (borrowed) */
const tr_bigint *tr_env_point_poly(const tr_env *, uint64_t point, int32_t k); /* ALGEBRAIC: q_k (borrowed) */
int32_t tr_env_point_interval(const tr_env *, uint64_t point, const tr_bigq **lo, const tr_bigq **hi);  /* borrowed */
/* an ALGEBRAIC point's isolating interval refined to a width <= width (INV-001; owned results) */
int32_t tr_env_point_refine(const tr_env *, uint64_t point, const tr_bigq *width, tr_bigq **lo, tr_bigq **hi,
                            const tr_math_budget *budget, tr_math_refusal *refusal);
const tr_bigq *tr_env_bound(const tr_env *, int32_t hi);                      /* NULL: unbounded */
/* the envelope value at a rational t (owned), its region; *status TR_ENV_Q_OUTSIDE / EMPTY (then *out NULL) */
int32_t tr_env_value(const tr_env *, const tr_bigq *t, tr_bigq **out, uint64_t *region, int32_t *status,
                     const tr_math_budget *budget, tr_math_refusal *refusal);
void tr_env_free(tr_env *);

/* ------------------------------------------------------------------ lookups (libalg) */
int32_t tr_env_lookup_need(const tr_env *, uint64_t npoints, const tr_bigq *const *points, int32_t mode,
                           int32_t target, tr_math_need *need);
tr_env_lookup *tr_env_lookup_create(const tr_env *, uint64_t npoints, const tr_bigq *const *points, int32_t mode,
                                    const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error);
/* the same with int64 rationals num[i] / den[i] (den >= 1): no big-number marshaling */
int32_t tr_env_lookup_need_q64(const tr_env *, uint64_t npoints, const int64_t *num, const int64_t *den, int32_t mode,
                               int32_t target, tr_math_need *need);
tr_env_lookup *tr_env_lookup_create_q64(const tr_env *, uint64_t npoints, const int64_t *num, const int64_t *den,
                                        int32_t mode, const tr_math_budget *budget, tr_math_refusal *refusal,
                                        int32_t *error);
int32_t tr_env_lookup_get_view(const tr_env_lookup *, tr_env_view *);
uint64_t tr_env_lookup_bound_bits(const tr_env_lookup *);   /* B of the proof (|S| < 2^B) */
/* the CPU evaluator (output = npoints) */
int32_t tr_env_lookup_eval(const tr_env_view *, tr_env_answer *out, const tr_math_budget *budget, tr_math_refusal *refusal);
void tr_env_lookup_free(tr_env_lookup *);

/* ------------------------------------------------------------------ the CUDA library (math_env_gpu.cu) */
int tr_mgpu_env_lookup(const tr_env_view *view, tr_env_answer *out, const tr_math_budget *budget,
                       tr_math_refusal *refusal);
#ifdef __cplusplus
}
#endif
#endif
