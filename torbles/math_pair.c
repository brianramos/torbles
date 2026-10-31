/* PAIR-001: sorted-distribution pairwise summaries of DIST-002 prefix states; see math_pair.h.
 * SPDX-License-Identifier: Apache-2.0 */
#include "math_pair.h"
#include "mathematics.h"
#include "math_domain.h"
#include "math_handles.h"

TR_MATH_ABI(tr_pair_request, TR_PAIR_ABI);      /* the ctypes boundary of PAIR-001 (math_handles.h) */
TR_MATH_ABI(tr_pair_result, TR_PAIR_ABI);

_Static_assert(sizeof(tr_pair_request) == 32, "tr_pair_request ABI");
_Static_assert(sizeof(tr_pair_result) == 152, "tr_pair_result ABI");

int32_t tr_pair_query(const tr_dist_prefix *a, const tr_dist_prefix *b, const tr_pair_request *req, uint64_t n,
                      tr_pair_result *out, const tr_math_budget *budget, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    tr_dist_prefix_view va, vb;
    if (tr_dist_prefix_get_view(a, &va) || (b && tr_dist_prefix_get_view(b, &vb)) || (n && (!req || !out)))
        return TR_MATH_INVALID;
    if (!b) vb = va;
    tr_math_budget dflt;
    if (!budget) { tr_budget_default(&dflt); budget = &dflt; }
    int32_t rc = tr_budget_check_one(budget, TR_BUDGET_OUTPUT, n, refusal);
    if (rc) return rc;
    for (uint64_t i = 0; i < n; i++) tr_pair_eval(&va, &vb, &req[i], &out[i]);
    return TR_MATH_OK;
}
