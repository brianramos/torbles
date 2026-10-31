/* RESP-002: piecewise absolute, hinge and squared-hinge loss curves of a prefix state; see math_resp.h.
 * SPDX-License-Identifier: Apache-2.0 */
#include "math_resp.h"
#include "mathematics.h"
#include "math_domain.h"
#include "math_handles.h"
#include <string.h>

TR_MATH_ABI(tr_resp_request, 1);           /* the ctypes boundary of RESP-002 (math_handles.h) */
TR_MATH_ABI(tr_resp_result, 1);
TR_MATH_ABI(tr_resp_piece, 1);
TR_MATH_ABI(tr_resp_minimum, 1);

_Static_assert(sizeof(tr_resp_request) == 48, "tr_resp_request ABI");
_Static_assert(sizeof(tr_resp_result) == 144, "tr_resp_result ABI");
_Static_assert(sizeof(tr_resp_piece) == 136, "tr_resp_piece ABI");
_Static_assert(sizeof(tr_resp_minimum) == 176, "tr_resp_minimum ABI");

static int loss_ok(int32_t loss, int64_t tau_num, int64_t tau_den) {
    if (loss < 1 || loss > TR_RESP_LOSSES) return 0;
    return loss != TR_RESP_PINBALL || (tau_den >= 1 && tau_num >= 0 && tau_num <= tau_den);
}

int32_t tr_resp_eval(const tr_dist_prefix *p, const tr_resp_request *req, uint64_t n, tr_resp_result *out,
                     const tr_math_budget *budget, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    tr_dist_prefix_view v;
    if (tr_dist_prefix_get_view(p, &v) || (n && (!req || !out))) return TR_MATH_INVALID;
    tr_math_budget dflt;
    if (!budget) { tr_budget_default(&dflt); budget = &dflt; }
    int32_t rc = tr_budget_check_one(budget, TR_BUDGET_OUTPUT, n, refusal);
    if (rc) return rc;
    for (uint64_t i = 0; i < n; i++) tr_resp_eval_one(&v, &req[i], &out[i]);
    return TR_MATH_OK;
}

int32_t tr_resp_pieces(const tr_dist_prefix *p, uint64_t group, int32_t loss, int64_t tau_num, int64_t tau_den,
                       tr_resp_piece *out, uint64_t capacity, uint64_t *count, const tr_math_budget *budget,
                       tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    tr_dist_prefix_view v;
    if (tr_dist_prefix_get_view(p, &v) || !count || group >= v.ngroups || !loss_ok(loss, tau_num, tau_den))
        return TR_MATH_INVALID;
    if (v.degree < tr_resp_needs_degree(loss)) return TR_MATH_UNSUPPORTED;
    const uint64_t b = v.offset[group], e = v.offset[group + 1], d = e - b;
    const uint64_t n = loss == TR_RESP_SQUARED || d == 0 ? 1 : d + 1;
    tr_math_budget dflt;
    if (!budget) { tr_budget_default(&dflt); budget = &dflt; }
    int32_t rc = tr_budget_check_one(budget, TR_BUDGET_OUTPUT, n, refusal);
    if (rc) return rc;
    *count = n;
    if (!out) return TR_MATH_OK;
    if (capacity < n) return TR_MATH_INVALID;
    const int sq = v.degree >= 2;
    const unsigned __int128 M = tr_dist_ex(&v, b, e, 0), S1 = tr_dist_ex(&v, b, e, 1), S2 = sq ? tr_dist_ex(&v, b, e, 2) : 0;
    const int64_t a = tau_num, c = tau_den - tau_num;
    for (uint64_t s = 0; s < n; s++) {
        tr_resp_piece *pc = &out[s];
        memset(pc, 0, sizeof *pc);
        pc->lo_inf = s == 0; pc->hi_inf = s == n - 1;
        if (!pc->lo_inf) pc->lo = v.value[b + s - 1];
        if (!pc->hi_inf) pc->hi = v.value[b + s];
        pc->den = tr_i128_set(1);
        /* on the piece the values below theta are the entries before b + s */
        const uint64_t at = loss == TR_RESP_SQUARED ? b : b + s;
        __int128 mlo = (__int128)tr_dist_ex(&v, b, at, 0), mhi = (__int128)(M - tr_dist_ex(&v, b, at, 0));
        __int128 s1lo = (__int128)tr_dist_ex(&v, b, at, 1), s1hi = (__int128)(S1 - tr_dist_ex(&v, b, at, 1));
        __int128 s2lo = sq ? (__int128)tr_dist_ex(&v, b, at, 2) : 0, s2hi = sq ? (__int128)(S2 - tr_dist_ex(&v, b, at, 2)) : 0;
        int ok = 1;
        switch (loss) {
        case TR_RESP_ABS:
            pc->c[0] = tr_dist_w_from_i128((__int128)(S1 - 2 * tr_dist_ex(&v, b, at, 1)));   /* S1_hi - S1_lo */
            pc->c[1] = tr_dist_w_from_i128(mlo - mhi);
            break;
        case TR_RESP_HINGE_UP:
            pc->c[0] = tr_dist_w_from_i128(s1hi); pc->c[1] = tr_dist_w_from_i128(-mhi);
            break;
        case TR_RESP_HINGE_DOWN:
            pc->c[0] = tr_dist_w_from_i128(-s1lo); pc->c[1] = tr_dist_w_from_i128(mlo);
            break;
        case TR_RESP_SQHINGE_UP:
            pc->c[0] = tr_dist_w_from_i128(s2hi); pc->c[2] = tr_dist_w_from_i128(mhi);
            ok = tr_dist_w_mul64(tr_dist_w_from_i128(s1hi), -2, &pc->c[1]);
            break;
        case TR_RESP_SQHINGE_DOWN:
            pc->c[0] = tr_dist_w_from_i128(s2lo); pc->c[2] = tr_dist_w_from_i128(mlo);
            ok = tr_dist_w_mul64(tr_dist_w_from_i128(s1lo), -2, &pc->c[1]);
            break;
        case TR_RESP_SQUARED:
            pc->c[0] = tr_dist_w_from_i128((__int128)S2); pc->c[2] = tr_dist_w_from_i128((__int128)M);
            ok = tr_dist_w_mul64(tr_dist_w_from_i128((__int128)S1), -2, &pc->c[1]);
            break;
        case TR_RESP_PINBALL:
            ok = tr_dist_w_sub(tr_dist_w_mul128(a, s1hi), tr_dist_w_mul128(c, s1lo), &pc->c[0]) &&
                 tr_dist_w_sub(tr_dist_w_mul128(c, mlo), tr_dist_w_mul128(a, mhi), &pc->c[1]);
            pc->den = tr_i128_set(tau_den);
            break;
        }
        if (!ok) return TR_MATH_OVERFLOW;
    }
    return TR_MATH_OK;
}

/* the loss at the integer theta x (the canonical point of a minimizer set) */
static int32_t loss_at(const tr_dist_prefix_view *v, uint64_t group, int32_t loss, int64_t tau_num, int64_t tau_den,
                       int64_t x, tr_resp_minimum *out) {
    tr_resp_request q;
    memset(&q, 0, sizeof q);
    q.loss = loss; q.group = group; q.num = x; q.den = 1; q.tau_num = tau_num; q.tau_den = tau_den;
    tr_resp_result r;
    tr_resp_eval_one(v, &q, &r);
    if (r.status) return r.status;
    out->value = r.value; out->value_den = r.value_den;
    return TR_MATH_OK;
}

int32_t tr_resp_minimize(const tr_dist_prefix *p, uint64_t group, int32_t loss, int64_t tau_num, int64_t tau_den,
                         tr_resp_minimum *out) {
    tr_dist_prefix_view v;
    if (!out || tr_dist_prefix_get_view(p, &v) || group >= v.ngroups || !loss_ok(loss, tau_num, tau_den))
        return TR_MATH_INVALID;
    if (v.degree < tr_resp_needs_degree(loss)) return TR_MATH_UNSUPPORTED;
    memset(out, 0, sizeof *out);
    out->lo_den = out->hi_den = out->value_den = tr_i128_set(1);
    const uint64_t b = v.offset[group], e = v.offset[group + 1];
    const unsigned __int128 M = tr_dist_ex(&v, b, e, 0);
    if (M == 0) {                                   /* the loss is 0 everywhere */
        out->kind = TR_RESP_MIN_ALL; out->lo_inf = out->hi_inf = 1;
        return TR_MATH_OK;
    }
    uint64_t ilo = b, ihi = b;
    switch (loss) {
    case TR_RESP_ABS:
    case TR_RESP_PINBALL: {
        unsigned __int128 lo_rank, hi_rank;
        if (loss == TR_RESP_ABS) { lo_rank = (M + 1) / 2; hi_rank = M / 2 + 1; }
        else {
            unsigned __int128 q;
            uint64_t rem;
            tr_dist_muldiv(M, (uint64_t)tau_num, (uint64_t)tau_den, &q, &rem);   /* tau M = q + rem / tau_den */
            lo_rank = q + (rem != 0); hi_rank = q + 1;
        }
        out->lo_inf = lo_rank == 0; out->hi_inf = hi_rank > M;
        if (!out->lo_inf) ilo = tr_dist_rank_entry(&v, b, e, lo_rank);
        if (!out->hi_inf) ihi = tr_dist_rank_entry(&v, b, e, hi_rank);
        break;
    }
    case TR_RESP_HINGE_UP:
    case TR_RESP_SQHINGE_UP:
        ilo = e - 1; out->hi_inf = 1;
        break;
    case TR_RESP_HINGE_DOWN:
    case TR_RESP_SQHINGE_DOWN:
        ihi = b; out->lo_inf = 1;
        break;
    case TR_RESP_SQUARED: {
        __int128 S1 = (__int128)tr_dist_ex(&v, b, e, 1), S2 = (__int128)tr_dist_ex(&v, b, e, 2), m = (__int128)M;
        out->kind = TR_RESP_MIN_POINT;
        out->lo = out->hi = tr_dist_w_from_i128(S1);
        out->lo_den = out->hi_den = tr_i128_set(m);
        if (!tr_dist_w_sub(tr_dist_w_mul128(m, S2), tr_dist_w_mul128(S1, S1), &out->value)) return TR_MATH_OVERFLOW;
        out->value_den = tr_i128_set(m);
        return TR_MATH_OK;
    }
    }
    if (!out->lo_inf) { out->lo = tr_dist_w_from_i128(v.value[ilo]); out->lo_index = ilo - b; }
    if (!out->hi_inf) { out->hi = tr_dist_w_from_i128(v.value[ihi]); out->hi_index = ihi - b; }
    out->kind = out->lo_inf ? TR_RESP_MIN_DOWN : out->hi_inf ? TR_RESP_MIN_UP
              : (v.value[ilo] == v.value[ihi] ? TR_RESP_MIN_POINT : TR_RESP_MIN_INTERVAL);
    return loss_at(&v, group, loss, tau_num, tau_den, out->lo_inf ? v.value[ihi] : v.value[ilo], out);
}
