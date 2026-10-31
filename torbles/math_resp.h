/* RESP-002: complete piecewise absolute, hinge and squared-hinge loss functions of a parameter theta, from the sorted
 * support and prefix moments of a DIST-002 prefix state (math_dist.h), on the CPU and in CUDA.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Data. A group of a prefix state holds the distinct support values x_1 < ... < x_d (logical integers) with masses
 * m_i > 0 (rows, or nonnegative weight sums) and the prefix sums of m x^k (k <= degree). M = sum m_i. Rows whose value
 * is NULL are not part of the support (SQL aggregate semantics: they contribute nothing to any loss).
 *
 * Losses of theta (logical units of x), summed over the support:
 *   TR_RESP_ABS           sum m |x - theta|
 *   TR_RESP_HINGE_UP      sum m max(0, x - theta)            (the hinge; decreasing)
 *   TR_RESP_HINGE_DOWN    sum m max(0, theta - x)            (its mirror; increasing)
 *   TR_RESP_SQHINGE_UP    sum m max(0, x - theta)^2
 *   TR_RESP_SQHINGE_DOWN  sum m max(0, theta - x)^2
 *   TR_RESP_SQUARED       sum m (x - theta)^2
 *   TR_RESP_PINBALL       sum m (tau max(0, x - theta) + (1 - tau) max(0, theta - x)), 0 <= tau <= 1 (the check loss;
 *                         ABS = 2 PINBALL(1/2))
 * ABS, the hinges and PINBALL need degree >= 1, the squared forms degree >= 2 (else TR_MATH_UNSUPPORTED).
 *
 * Evaluation (tr_resp_eval; batched). theta = num / den (den >= 1, int64 parts) is exact. With M_<, M_<=, M_>, M_>=
 * the masses of the values below, at most, above, at least theta and S1, S2 the matching moment sums, every value is
 * an exact rational (numerator 256-bit two's complement, denominator > 0, not necessarily in lowest terms) and so are
 * the one-sided derivatives (slopes) left = d/dtheta from below, right = from above:
 *   ABS          value (S1_> - S1_<) - theta (M_> - M_<)      left M_< - M_>=, right M_<= - M_>
 *   HINGE_UP     value S1_> - theta M_>                        left -M_>=,      right -M_>
 *   HINGE_DOWN   value theta M_< - S1_<                        left M_<,        right M_<=
 *   SQHINGE_UP   value S2_> - 2 theta S1_> + theta^2 M_>       left = right = -2 (S1_> - theta M_>)
 *   SQHINGE_DOWN value S2_< - 2 theta S1_< + theta^2 M_<       left = right = 2 (theta M_< - S1_<)
 *   SQUARED      value S2 - 2 theta S1 + theta^2 M             left = right = 2 (theta M - S1)
 *   PINBALL      tau HINGE_UP + (1 - tau) HINGE_DOWN; left (1 - tau) M_< - tau M_>=, right (1 - tau) M_<= - tau M_>
 * Between breakpoints left = right; at a breakpoint x_j (theta equal to a support value) ABS, the hinges and PINBALL
 * have a kink of size m_j (times 2, 1 or tau / 1 - tau); the squared forms are continuously differentiable. A value
 * that does not fit 256 bits (only near the int64 limits of theta, den and the moments) is TR_MATH_OVERFLOW for that
 * request; every request has its own status.
 *
 * The complete function (tr_resp_pieces): the d + 1 pieces (-inf, x_1], [x_1, x_2], ..., [x_d, +inf) (one piece
 * (-inf, +inf) for SQUARED or an empty support), each with loss(theta) = (c0 + c1 theta + c2 theta^2) / den exactly on
 * the closed piece (den = 1, or tau's denominator for PINBALL). Output budget: the pieces.
 *
 * Minimizers (tr_resp_minimize) with the tie convention: the whole minimizer set is reported as a closed interval
 * [lo, hi] (ends possibly unbounded); its canonical point is lo when bounded below, else hi. With the expanded order
 * statistics y_1 <= ... <= y_M (each x_i repeated m_i times):
 *   ABS          [y_ceil(M/2), y_floor(M/2)+1]: the median interval (a point when M is odd or the two middle values tie)
 *   PINBALL      [y_ceil(tau M), y_floor(tau M)+1], y_0 = -inf, y_{M+1} = +inf: the tau-quantile interval
 *   HINGE_UP, SQHINGE_UP      [x_d, +inf);   HINGE_DOWN, SQHINGE_DOWN   (-inf, x_1]   (minimum 0)
 *   SQUARED      the point S1 / M (the mean), minimum (M S2 - S1^2) / M
 *   empty (M = 0: an absent group, only NULL values, only zero weights): the loss is 0 everywhere, the set is
 *   (-inf, +inf) (TR_RESP_MIN_ALL) and there is no canonical point.
 * Duplicate observations are one support value with their mass; degenerate minima (a flat bottom between two
 * support values) are intervals. The minimum value is the loss at the canonical point.
 *
 * CUDA (math_resp_gpu.cu): tr_mgpu_resp_eval answers a batch against a prefix state uploaded once with
 * tr_mgpu_dist_prefix_upload: one work item per request running the shared tr_resp_eval_one, one output slot each, so
 * the answers equal the CPU's bit for bit. Budget of a batch: output = requests, bytes = the device copies. */
#ifndef TORBLES_MATH_RESP_H
#define TORBLES_MATH_RESP_H
#include <stdint.h>
#include "math_num.h"
#include "math_budgets.h"
#include "math_dist.h"
#ifdef __cplusplus
extern "C" {
#endif

#define TR_RESP_ABI 1
enum { TR_RESP_ABS = 1, TR_RESP_HINGE_UP = 2, TR_RESP_HINGE_DOWN = 3, TR_RESP_SQHINGE_UP = 4,
       TR_RESP_SQHINGE_DOWN = 5, TR_RESP_SQUARED = 6, TR_RESP_PINBALL = 7, TR_RESP_LOSSES = 7 };
enum { TR_RESP_MIN_ALL = 1, TR_RESP_MIN_POINT = 2, TR_RESP_MIN_INTERVAL = 3, TR_RESP_MIN_UP = 4, TR_RESP_MIN_DOWN = 5 };

/* one evaluation (48 bytes) and its answer (144 bytes) */
typedef struct {
    int32_t loss, reserved;            /* TR_RESP_*; 0 */
    uint64_t group;
    int64_t num, den;                  /* theta = num / den, den >= 1 */
    int64_t tau_num, tau_den;          /* PINBALL: tau = tau_num / tau_den in [0, 1]; ignored otherwise */
} tr_resp_request;

typedef struct {
    int32_t status, at_break;          /* 0 or this request's error code; 1 when theta is a support value */
    uint64_t index;                    /* the support values below theta (at a breakpoint: its position) */
    tr_dist_i256 value, left, right;   /* value / value_den, left / slope_den, right / slope_den */
    tr_math_i128 value_den, slope_den; /* > 0 */
} tr_resp_result;

/* one piece of the complete function (136 bytes): (c0 + c1 theta + c2 theta^2) / den on [lo, hi] */
typedef struct {
    int32_t lo_inf, hi_inf;            /* the piece is unbounded below / above */
    int64_t lo, hi;                    /* its breakpoints when bounded */
    tr_dist_i256 c[3];
    tr_math_i128 den;
} tr_resp_piece;

/* the minimizer set and minimum (176 bytes) */
typedef struct {
    int32_t status, kind;              /* 0 or an error code; TR_RESP_MIN_* */
    int32_t lo_inf, hi_inf;            /* an end is unbounded */
    tr_dist_i256 lo, hi;               /* lo / lo_den and hi / hi_den: the closed set's ends */
    tr_math_i128 lo_den, hi_den;
    tr_dist_i256 value;                /* the minimum: value / value_den (the loss at the canonical point) */
    tr_math_i128 value_den;
    uint64_t lo_index, hi_index;       /* positions of the ends in the group's support (support values only) */
} tr_resp_minimum;

/* ---------------------------------------------------------------- shared evaluation (CPU code and kernels) */
TR_NUM_HD int tr_resp_needs_degree(int32_t loss) {
    return loss == TR_RESP_SQHINGE_UP || loss == TR_RESP_SQHINGE_DOWN || loss == TR_RESP_SQUARED ? 2 : 1;
}
/* the exact quadratic form d^2 S2 - 2 p d S1 + p^2 M0 (theta = p / d): 1, or 0 on overflow */
TR_NUM_HD int tr_resp_quad(int64_t p, int64_t d, __int128 S2, __int128 S1, __int128 M0, tr_dist_i256 *out) {
    __int128 dd = (__int128)d * d, pd2 = 2 * ((__int128)p * d), pp = (__int128)p * p;
    tr_dist_i256 a = tr_dist_w_mul128(dd, S2), b = tr_dist_w_mul128(pd2, S1), c = tr_dist_w_mul128(pp, M0), r;
    return tr_dist_w_sub(a, b, &r) && tr_dist_w_add(r, c, out);
}
/* d S1 - p M (theta = p / d): exact */
TR_NUM_HD int tr_resp_lin(int64_t p, int64_t d, __int128 S1, __int128 M, tr_dist_i256 *out) {
    return tr_dist_w_sub(tr_dist_w_mul128(d, S1), tr_dist_w_mul128(p, M), out);
}
TR_NUM_HD void tr_resp_result_init(tr_resp_result *r) {
    r->status = 0; r->at_break = 0; r->index = 0;
    r->value = tr_dist_w_zero(); r->left = tr_dist_w_zero(); r->right = tr_dist_w_zero();
    r->value_den = tr_i128_set(1); r->slope_den = tr_i128_set(1);
}
TR_NUM_HD __int128 tr_resp_s(unsigned __int128 x) { return (__int128)x; }

TR_NUM_HD void tr_resp_eval_one(const tr_dist_prefix_view *v, const tr_resp_request *q, tr_resp_result *r) {
    tr_resp_result_init(r);
    if (q->group >= v->ngroups || q->den < 1 || q->reserved || q->loss < 1 || q->loss > TR_RESP_LOSSES ||
        (q->loss == TR_RESP_PINBALL && (q->tau_den < 1 || q->tau_num < 0 || q->tau_num > q->tau_den))) {
        r->status = -1; return;
    }
    if (v->degree < tr_resp_needs_degree(q->loss)) { r->status = -12; return; }
    const uint64_t b = v->offset[q->group], e = v->offset[q->group + 1];
    const int64_t p = q->num, d = q->den;
    uint64_t lt = tr_dist_upper_bound(v, b, e, tr_dist_ceil_div(p, d) - 1);    /* first entry >= theta */
    uint64_t le = tr_dist_upper_bound(v, b, e, tr_dist_floor_div(p, d));       /* first entry > theta */
    r->at_break = le > lt; r->index = lt - b;
    const int sq = v->degree >= 2;
    unsigned __int128 M = tr_dist_ex(v, b, e, 0), S1 = tr_dist_ex(v, b, e, 1), S2 = sq ? tr_dist_ex(v, b, e, 2) : 0;
    unsigned __int128 Mlt = tr_dist_ex(v, b, lt, 0), Mle = tr_dist_ex(v, b, le, 0);
    unsigned __int128 S1lt = tr_dist_ex(v, b, lt, 1), S1le = tr_dist_ex(v, b, le, 1);
    unsigned __int128 S2lt = sq ? tr_dist_ex(v, b, lt, 2) : 0, S2le = sq ? tr_dist_ex(v, b, le, 2) : 0;
    /* masses in [0, 2^127); a moment over a subset of the support is bounded by the prefix proof */
    __int128 Mgt = tr_resp_s(M - Mle), Mge = tr_resp_s(M - Mlt), mlt = tr_resp_s(Mlt), mle = tr_resp_s(Mle);
    __int128 S1gt = tr_resp_s(S1 - S1le), s1lt = tr_resp_s(S1lt), S2gt = tr_resp_s(S2 - S2le), s2lt = tr_resp_s(S2lt);
    int ok = 1;
    switch (q->loss) {
    case TR_RESP_ABS:
        ok = tr_dist_w_add(tr_dist_w_mul128(d, tr_resp_s(S1 - S1le - S1lt)), tr_dist_w_mul128(p, mlt - Mgt), &r->value);
        r->value_den = tr_i128_set(d);
        r->left = tr_dist_w_from_i128(mlt - Mge); r->right = tr_dist_w_from_i128(mle - Mgt);
        break;
    case TR_RESP_HINGE_UP:
        ok = tr_resp_lin(p, d, S1gt, Mgt, &r->value);
        r->value_den = tr_i128_set(d);
        r->left = tr_dist_w_from_i128(-Mge); r->right = tr_dist_w_from_i128(-Mgt);
        break;
    case TR_RESP_HINGE_DOWN: {
        tr_dist_i256 t;
        ok = tr_resp_lin(p, d, s1lt, mlt, &t);                     /* d S1_< - p M_< = -(value numerator) */
        r->value = tr_dist_w_negate_bits(t);
        r->value_den = tr_i128_set(d);
        r->left = tr_dist_w_from_i128(mlt); r->right = tr_dist_w_from_i128(mle);
        break;
    }
    case TR_RESP_SQHINGE_UP:
    case TR_RESP_SQHINGE_DOWN:
    case TR_RESP_SQUARED: {
        int up = q->loss == TR_RESP_SQHINGE_UP, all = q->loss == TR_RESP_SQUARED;
        __int128 s2 = all ? tr_resp_s(S2) : up ? S2gt : s2lt, s1 = all ? tr_resp_s(S1) : up ? S1gt : s1lt;
        __int128 m0 = all ? tr_resp_s(M) : up ? Mgt : mlt;
        tr_dist_i256 lin;                                           /* d S1 - p M0: the slope is -2 lin / d */
        ok = tr_resp_quad(p, d, s2, s1, m0, &r->value) && tr_resp_lin(p, d, s1, m0, &lin) &&
             tr_dist_w_mul64(lin, -2, &r->left);
        r->right = r->left;
        r->value_den = tr_i128_set((__int128)d * d); r->slope_den = tr_i128_set(d);
        break;
    }
    case TR_RESP_PINBALL: {
        const int64_t a = q->tau_num, bt = q->tau_den, c = bt - a;
        tr_dist_i256 up, down, x, y, nd;
        ok = tr_resp_lin(p, d, S1gt, Mgt, &up) && tr_resp_lin(p, d, s1lt, mlt, &nd);
        down = tr_dist_w_negate_bits(nd);                           /* p M_< - d S1_< */
        ok = ok && tr_dist_w_mul64(up, a, &x) && tr_dist_w_mul64(down, c, &y) && tr_dist_w_add(x, y, &r->value);
        r->value_den = tr_i128_set((__int128)bt * d);
        ok = ok && tr_dist_w_sub(tr_dist_w_mul128(c, mlt), tr_dist_w_mul128(a, Mge), &r->left) &&
             tr_dist_w_sub(tr_dist_w_mul128(c, mle), tr_dist_w_mul128(a, Mgt), &r->right);
        r->slope_den = tr_i128_set(bt);
        break;
    }
    }
    if (!ok) { tr_resp_result_init(r); r->status = -3; }
}

/* ---------------------------------------------------------------- libalg */
/* n requests (output = n); 0 or TR_MATH_BUDGET / _INVALID; each answer has its own status */
int32_t tr_resp_eval(const tr_dist_prefix *, const tr_resp_request *requests, uint64_t n, tr_resp_result *out,
                     const tr_math_budget *budget, tr_math_refusal *refusal);
/* the pieces of group's loss: *count is set; out may be NULL (count only); capacity < count is TR_MATH_INVALID */
int32_t tr_resp_pieces(const tr_dist_prefix *, uint64_t group, int32_t loss, int64_t tau_num, int64_t tau_den,
                       tr_resp_piece *out, uint64_t capacity, uint64_t *count, const tr_math_budget *budget,
                       tr_math_refusal *refusal);
int32_t tr_resp_minimize(const tr_dist_prefix *, uint64_t group, int32_t loss, int64_t tau_num, int64_t tau_den,
                         tr_resp_minimum *out);

/* ---------------------------------------------------------------- the CUDA library (math_resp_gpu.cu) */
int tr_mgpu_resp_eval(const tr_mgpu_dist_prefix *, const tr_resp_request *requests, uint64_t n, tr_resp_result *out,
                      const tr_math_budget *budget, tr_math_refusal *refusal);
#ifdef __cplusplus
}
#endif
#endif
