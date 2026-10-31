/* PAIR-001: sorted-distribution pairwise summaries from the sorted support of DIST-002 prefix states (math_dist.h),
 * without enumerating pairs, on the CPU and in CUDA.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Data. A class is one group of a prefix state: its distinct present values x_1 < ... < x_d (logical integers) with
 * masses m_i > 0 (rows, or nonnegative weight sums: TR_DIST_MASS_COUNT / _WEIGHT) and M = sum m_i. As in DIST-002, a
 * class is the expanded multiset y_1 <= ... <= y_M (x_i repeated m_i times; a weight replicates its value in units of
 * 10^-wscale), so an element is a row (counts) or a unit of weight; NULL values are in no class (they form no pair).
 * Every summary below is a sum over pairs of elements, computed exactly from the sorted support in one pass (two-class
 * summaries: one merge of the two supports), never by enumerating the pairs. Every result is an exact rational
 * num / den (num >= 0, den >= 1, 256-bit, not necessarily in lowest terms).
 *
 * Conventions (each one explicit in the request):
 *   self-pairs   a self-pair is an element paired with itself; it exists only within one class (TR_PAIR_OP_ABS_WITHIN,
 *                or a two-class summary given the same class twice: the same state and group). Excluded unless
 *                TR_PAIR_SELF. (For weights: a self-pair is a unit paired with itself, so there are M of them.)
 *   ordering     TR_PAIR_OP_ABS_WITHIN counts unordered pairs {i, j} (i < j, or i <= j with self-pairs) unless
 *                TR_PAIR_ORDERED (pairs (i, j): each unordered pair twice). Two-class summaries pair (i in A, j in B):
 *                ordered by class (A first); TR_PAIR_ORDERED does not apply to them.
 *   normalization TR_PAIR_MEAN divides by the number of pairs P (ABS: the mean absolute difference; Mann-Whitney: the
 *                AUC); without it the raw sum. KS, Wasserstein and Cramer-von Mises compare the normalized empirical
 *                distributions F_A = c_A / M_A, F_B = c_B / M_B (c: the mass at or below t) and take no flag.
 *   empty class  M_A = 0 or M_B = 0 (an absent group, only NULL values, only zero weights): sums over no pairs are 0
 *                (ABS, U), every ratio is empty (no answer, SQL NULL): the means, the AUC, KS, W1, CvM. A pair count
 *                P = 0 (one element within one class without self-pairs) also empties the means.
 *
 * Summaries (P = the number of pairs; within one class of mass M: unordered M (M - 1) / 2, with self-pairs
 * M (M + 1) / 2, ordered M (M - 1), with self-pairs M^2; two classes: M_A M_B, minus M without self-pairs when the
 * classes coincide):
 *   ABS_WITHIN       sum over the pairs of |y_i - y_j| = sum_k (x_{k+1} - x_k) c_k (M - c_k) (unordered; ordered: twice)
 *                    (the Gini mean difference: MEAN with self-pairs excluded, ordered or not)
 *   ABS_CROSS        sum_{i in A, j in B} |x_i - y_j| = sum over consecutive pooled values t_k < t_{k+1} of
 *                    (t_{k+1} - t_k) (c_A (M_B - c_B) + c_B (M_A - c_A)) (c at t_k; self-pairs add 0)
 *   MANN_WHITNEY     U = sum_{i in A, j in B} ([x_i > y_j] + 1/2 [x_i = y_j]) (ties count half; num = 2U, den = 2);
 *                    MEAN: the AUC U / P. A self-pair is a tie (it adds 1/2 when TR_PAIR_SELF).
 *   KS               sup_t |F_A(t) - F_B(t)| = max_k |c_A M_B - c_B M_A| / (M_A M_B) over the pooled values (right-
 *                    continuous empirical CDFs); `at` = the smallest pooled value attaining it (none when it is 0)
 *   WASSERSTEIN      the 1-Wasserstein (earth mover's) distance of the normalized distributions,
 *                    integral |F_A - F_B| dt = sum_k (t_{k+1} - t_k) |c_A M_B - c_B M_A| / (M_A M_B), logical units
 *                    (times 10^-scale for the real values)
 *   CRAMER_VON_MISES the two-sample statistic T = M_A M_B / (M_A + M_B)^2 sum over the pooled elements u of
 *                    (F_A(u) - F_B(u))^2 = sum_k (m_A + m_B)(t_k) (c_A M_B - c_B M_A)^2 / (M_A M_B (M_A + M_B)^2)
 *                    (Anderson 1962, with ties handled by the right-continuous empirical CDFs at the pooled values)
 *
 * Exactness. Every class mass must be below 2^64 (counts always are; a weight class at or past 2^64 is
 * TR_MATH_OVERFLOW for that request). Then every sum is exact in 256 bits (ABS and W1 below M_A M_B (x_max - x_min) <
 * 2^192, 2U below 2^129); Cramer-von Mises needs (M_A M_B)^2 (M_A + M_B) < 2^255 and its denominator below 2^255 (else
 * TR_MATH_OVERFLOW; always true for counts). Two classes from two states must have the same value scale (else
 * TR_MATH_DOMAIN_ERROR). Each request has its own status (0, TR_MATH_INVALID, TR_MATH_OVERFLOW, TR_MATH_DOMAIN_ERROR),
 * so a batch never fails as a whole.
 *
 * Evaluation (tr_pair_eval) is shared by the CPU loop (tr_pair_query) and the CUDA kernel (math_pair_gpu.cu: one work
 * item per request over prefix states uploaded once with tr_mgpu_dist_prefix_upload, one output slot each), so both
 * answer bit for bit alike. Cost per request: O(d) within a class, O(d_A + d_B) for two classes.
 *
 * Budgets: a batch's output = its requests (CPU and CUDA); CUDA bytes = the device copies of the requests and answers. */
#ifndef TORBLES_MATH_PAIR_H
#define TORBLES_MATH_PAIR_H
#include <stdint.h>
#include "math_num.h"
#include "math_budgets.h"
#include "math_dist.h"
#ifdef __cplusplus
extern "C" {
#endif

#define TR_PAIR_ABI 1
enum { TR_PAIR_OP_ABS_WITHIN = 1, TR_PAIR_OP_ABS_CROSS = 2, TR_PAIR_OP_MANN_WHITNEY = 3, TR_PAIR_OP_KS = 4,
       TR_PAIR_OP_WASSERSTEIN = 5, TR_PAIR_OP_CRAMER_VON_MISES = 6, TR_PAIR_OPS = 6 };
enum { TR_PAIR_ORDERED = 1, TR_PAIR_SELF = 2, TR_PAIR_MEAN = 4 };

/* one request (32 bytes): class A = group a of the first state, class B = group b of the second (two-class ops) */
typedef struct {
    int32_t op, flags;                 /* TR_PAIR_OP_*; TR_PAIR_ORDERED | TR_PAIR_SELF | TR_PAIR_MEAN where they apply */
    uint64_t a, b;                     /* the groups (b ignored by ABS_WITHIN) */
    int64_t reserved;                  /* 0 */
} tr_pair_request;

/* its answer (152 bytes) */
typedef struct {
    int32_t status, empty;             /* 0 or this request's error code; 1: no answer (SQL NULL) */
    tr_dist_i256 num, den;             /* the value num / den (num >= 0, den >= 1) */
    tr_dist_i256 pairs;                /* P, the number of pairs (elements: rows or weight units) */
    tr_math_i128 mass_a, mass_b;       /* M_A, M_B (mass_b = 0 for ABS_WITHIN) */
    int64_t at;                        /* KS: the smallest pooled value where the supremum is attained */
    int32_t has_at, reserved;          /* 1 when `at` is set */
} tr_pair_result;

/* ---------------------------------------------------------------- unsigned 256-bit helpers (values below 2^255) */
TR_NUM_HD tr_dist_i256 tr_pair_u(unsigned __int128 v) {
    tr_dist_i256 r; r.w[0] = (uint64_t)v; r.w[1] = (uint64_t)(v >> 64); r.w[2] = r.w[3] = 0; return r;
}
/* *acc += x: 1, or 0 when the sum reaches 2^255 (*acc unchanged) */
TR_NUM_HD int tr_pair_add(tr_dist_i256 *acc, tr_dist_i256 x) {
    tr_dist_i256 s;
    uint64_t c = 0;
    for (int i = 0; i < 4; i++) {
        uint64_t t = acc->w[i] + c, c1 = t < c, y = t + x.w[i], c2 = y < t;
        s.w[i] = y; c = c1 | c2;
    }
    if (c || (s.w[3] >> 63)) return 0;
    *acc = s;
    return 1;
}
/* a * b for a < 2^255 and b < 2^128: 1, or 0 when the product reaches 2^255 */
TR_NUM_HD int tr_pair_mul(tr_dist_i256 a, unsigned __int128 b, tr_dist_i256 *r) {
    uint64_t b0 = (uint64_t)b, b1 = (uint64_t)(b >> 64), p[6] = {0, 0, 0, 0, 0, 0};
    for (int i = 0; i < 4; i++) {
        unsigned __int128 t = (unsigned __int128)a.w[i] * b0 + p[i];
        p[i] = (uint64_t)t;
        unsigned __int128 u = (unsigned __int128)a.w[i] * b1 + p[i + 1] + (uint64_t)(t >> 64);
        p[i + 1] = (uint64_t)u;
        p[i + 2] += (uint64_t)(u >> 64);
    }
    if (p[4] || p[5] || (p[3] >> 63)) return 0;
    for (int i = 0; i < 4; i++) r->w[i] = p[i];
    return 1;
}
TR_NUM_HD int tr_pair_bits128(unsigned __int128 v) {
    int n = 0;
    while (v) { n++; v >>= 1; }
    return n;
}
/* the mass of support entry i of the group starting at b */
TR_NUM_HD unsigned __int128 tr_pair_mass(const tr_dist_prefix_view *v, uint64_t b, uint64_t i) {
    return (unsigned __int128)tr_dist_px(v, i, 0) - tr_dist_ex(v, b, i, 0);
}
TR_NUM_HD unsigned __int128 tr_pair_absdiff(unsigned __int128 x, unsigned __int128 y) { return x > y ? x - y : y - x; }
TR_NUM_HD void tr_pair_result_init(tr_pair_result *r) {
    r->status = 0; r->empty = 0;
    r->num = tr_pair_u(0); r->den = tr_pair_u(1); r->pairs = tr_pair_u(0);
    r->mass_a = tr_i128_set(0); r->mass_b = tr_i128_set(0);
    r->at = 0; r->has_at = 0; r->reserved = 0;
}

/* ---------------------------------------------------------------- one request (CPU loop and CUDA kernel) */
/* A: the first state, B: the second (the same view for one state); see the header comment */
TR_NUM_HD void tr_pair_eval(const tr_dist_prefix_view *A, const tr_dist_prefix_view *B, const tr_pair_request *q,
                            tr_pair_result *r) {
    tr_pair_result_init(r);
    const int32_t op = q->op, fl = q->flags;
    const int two = op != TR_PAIR_OP_ABS_WITHIN;
    int allowed = op == TR_PAIR_OP_ABS_WITHIN ? (TR_PAIR_ORDERED | TR_PAIR_SELF | TR_PAIR_MEAN)
                : (op == TR_PAIR_OP_ABS_CROSS || op == TR_PAIR_OP_MANN_WHITNEY) ? (TR_PAIR_SELF | TR_PAIR_MEAN) : 0;
    if (op < 1 || op > TR_PAIR_OPS || (fl & ~allowed) || q->reserved || q->a >= A->ngroups ||
        (two && q->b >= B->ngroups)) { r->status = -1; return; }                         /* TR_MATH_INVALID */
    const unsigned __int128 LIM64 = (unsigned __int128)1 << 64;
    const uint64_t ba = A->offset[q->a], ea = A->offset[q->a + 1];
    const unsigned __int128 MA = ea > ba ? (unsigned __int128)tr_dist_px(A, ea - 1, 0) : 0;
    r->mass_a = tr_i128_set((__int128)MA);
    if (MA >= LIM64) { r->status = -3; return; }                                         /* TR_MATH_OVERFLOW */
    if (!two) {
        /* D = sum_k gap_k c_k (M - c_k): each unordered pair of elements adds the length between them */
        tr_dist_i256 D = tr_pair_u(0);
        unsigned __int128 c = 0;
        for (uint64_t i = ba; i + 1 < ea; i++) {
            c += tr_pair_mass(A, ba, i);
            uint64_t gap = (uint64_t)A->value[i + 1] - (uint64_t)A->value[i];
            tr_dist_i256 t;
            tr_pair_mul(tr_pair_u(c * (MA - c)), (unsigned __int128)gap, &t);              /* < 2^192 */
            tr_pair_add(&D, t);
        }
        int ordered = (fl & TR_PAIR_ORDERED) != 0, self = (fl & TR_PAIR_SELF) != 0;
        tr_dist_i256 P = tr_pair_u(0);
        if (MA) {
            unsigned __int128 sq = MA * MA;
            if (ordered) P = tr_pair_u(self ? sq : sq - MA);
            else P = tr_pair_u(self ? (sq + MA) >> 1 : (sq - MA) >> 1);     /* M (M +- 1) is even */
        }
        if (ordered) tr_pair_add(&D, D);
        r->num = D; r->pairs = P;
        if (fl & TR_PAIR_MEAN) {
            if (P.w[0] == 0 && P.w[1] == 0) { r->empty = 1; r->num = tr_pair_u(0); return; }
            r->den = P;
        }
        return;
    }
    if (A->scale != B->scale) { r->status = -6; return; }                                /* TR_MATH_DOMAIN_ERROR */
    const uint64_t bb = B->offset[q->b], eb = B->offset[q->b + 1];
    const unsigned __int128 MB = eb > bb ? (unsigned __int128)tr_dist_px(B, eb - 1, 0) : 0;
    r->mass_b = tr_i128_set((__int128)MB);
    if (MB >= LIM64) { r->status = -3; return; }
    const int same = A->value == B->value && A->offset == B->offset && q->a == q->b;
    const unsigned __int128 MAB = MA * MB;                                              /* < 2^128 */
    tr_dist_i256 P = tr_pair_u(MAB);
    if (same && !(fl & TR_PAIR_SELF) && (op == TR_PAIR_OP_ABS_CROSS || op == TR_PAIR_OP_MANN_WHITNEY))
        P = tr_pair_u(MAB - MA);
    r->pairs = P;
    tr_dist_i256 den = tr_pair_u(MAB);
    if (op == TR_PAIR_OP_CRAMER_VON_MISES && MAB) {
        /* (M_A M_B)^2 (M_A + M_B) and M_A M_B (M_A + M_B)^2 below 2^255 */
        unsigned __int128 S = MA + MB;
        tr_dist_i256 sq = tr_dist_w_mul_u128(MAB, MAB), bound, d2;
        if ((sq.w[3] >> 63) || !tr_pair_mul(sq, S, &bound) || !tr_pair_mul(tr_dist_w_mul_u128(S, S), MAB, &d2)) {
            r->status = -3; return;
        }
        den = d2;
    }
    const int ratio = op == TR_PAIR_OP_KS || op == TR_PAIR_OP_WASSERSTEIN || op == TR_PAIR_OP_CRAMER_VON_MISES;
    if (ratio && (MA == 0 || MB == 0)) { r->empty = 1; return; }
    tr_dist_i256 acc = tr_pair_u(0);
    unsigned __int128 cA = 0, cB = 0, best = 0;
    int64_t prev = 0, at = 0;
    int have = 0;
    uint64_t i = ba, j = bb;
    while (i < ea || j < eb) {
        int64_t t = (j >= eb || (i < ea && A->value[i] <= B->value[j])) ? A->value[i] : B->value[j];
        unsigned __int128 mA = (i < ea && A->value[i] == t) ? tr_pair_mass(A, ba, i) : 0;
        unsigned __int128 mB = (j < eb && B->value[j] == t) ? tr_pair_mass(B, bb, j) : 0;
        tr_dist_i256 term;
        if (have && (op == TR_PAIR_OP_ABS_CROSS || op == TR_PAIR_OP_WASSERSTEIN)) {
            uint64_t gap = (uint64_t)t - (uint64_t)prev;                               /* over [prev, t) */
            unsigned __int128 w = op == TR_PAIR_OP_ABS_CROSS ? cA * (MB - cB) : tr_pair_absdiff(cA * MB, cB * MA);
            tr_dist_i256 wide = tr_pair_u(w);
            if (op == TR_PAIR_OP_ABS_CROSS) tr_pair_add(&wide, tr_pair_u(cB * (MA - cA)));   /* < 2^129 */
            tr_pair_mul(wide, (unsigned __int128)gap, &term);                              /* < 2^193 */
            tr_pair_add(&acc, term);
        }
        if (op == TR_PAIR_OP_MANN_WHITNEY && mA) {                                       /* mA (2 c_B< + m_B) */
            tr_pair_mul(tr_pair_u(mA), 2 * cB + mB, &term);
            tr_pair_add(&acc, term);
        }
        cA += mA; cB += mB;
        if (op == TR_PAIR_OP_KS || op == TR_PAIR_OP_CRAMER_VON_MISES) {
            unsigned __int128 diff = tr_pair_absdiff(cA * MB, cB * MA);
            if (op == TR_PAIR_OP_KS && diff > best) { best = diff; at = t; }
            if (op == TR_PAIR_OP_CRAMER_VON_MISES && diff) {
                tr_pair_mul(tr_dist_w_mul_u128(diff, diff), mA + mB, &term);              /* within the proof */
                tr_pair_add(&acc, term);
            }
        }
        prev = t; have = 1;
        if (i < ea && A->value[i] == t) i++;
        if (j < eb && B->value[j] == t) j++;
    }
    if (op == TR_PAIR_OP_MANN_WHITNEY && same && !(fl & TR_PAIR_SELF)) {               /* the M self-ties, 1 each in 2U */
        tr_dist_i256 d;
        tr_dist_w_sub(acc, tr_pair_u(MA), &d);                                           /* 2U >= M: every tie counted */
        acc = d;
    }
    if (op == TR_PAIR_OP_KS) { acc = tr_pair_u(best); r->has_at = best != 0; r->at = best ? at : 0; }
    r->num = acc;
    if (op == TR_PAIR_OP_ABS_CROSS || op == TR_PAIR_OP_MANN_WHITNEY) {
        int zero = P.w[0] == 0 && P.w[1] == 0;
        if (op == TR_PAIR_OP_MANN_WHITNEY) r->den = tr_pair_u(2);
        if (fl & TR_PAIR_MEAN) {
            if (zero) { r->empty = 1; r->num = tr_pair_u(0); r->den = tr_pair_u(1); return; }
            if (op == TR_PAIR_OP_MANN_WHITNEY) tr_pair_mul(P, 2, &r->den);                   /* AUC = 2U / 2P */
            else r->den = P;
        }
        return;
    }
    r->den = den;
}

/* ---------------------------------------------------------------- the CPU entry point (libalg) */
/* n requests against the prefix states a and b (b NULL: a; class B of a two-class request is a group of b).
 * Budget: output = n. 0, or TR_MATH_BUDGET / TR_MATH_INVALID for the batch; each answer has its own status. */
int32_t tr_pair_query(const tr_dist_prefix *a, const tr_dist_prefix *b, const tr_pair_request *requests, uint64_t n,
                      tr_pair_result *out, const tr_math_budget *budget, tr_math_refusal *refusal);

/* ---------------------------------------------------------------- the CUDA library (math_pair_gpu.cu) */
/* the same batch against device copies made with tr_mgpu_dist_prefix_upload (b NULL: a). Budget: output = n, bytes
 * = the device copies of the requests and answers. 0 or a native error code (TR_MATH_DEVICE when the device fails). */
int tr_mgpu_pair_query(const tr_mgpu_dist_prefix *a, const tr_mgpu_dist_prefix *b, const tr_pair_request *requests,
                       uint64_t n, tr_pair_result *out, const tr_math_budget *budget, tr_math_refusal *refusal);
#ifdef __cplusplus
}
#endif
#endif
