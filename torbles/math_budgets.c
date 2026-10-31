/* Search, representation and output budgets of the mathematical modules (FND-010); see math_budgets.h.
 * SPDX-License-Identifier: Apache-2.0 */
#include "math_budgets.h"
#include "mathematics.h"
#include <math.h>
#include <stdlib.h>

void tr_math_budget_default(tr_math_budget *b) { if (b) tr_budget_default(b); }

void tr_math_budget_polynomial_kernel(tr_math_budget *b) {
    if (!b) return;
    tr_budget_default(b);
    b->limit[TR_BUDGET_VARIABLES] = TR_MATH_VARIABLES;
    b->limit[TR_BUDGET_DEGREE] = TR_MATH_DEGREE;
    b->limit[TR_BUDGET_SUPPORT] = TR_MATH_TERMS;
}

uint64_t tr_math_budget_binomial(uint64_t n, uint64_t k) { return tr_budget_binomial(n, k); }
uint64_t tr_math_budget_monomials(uint32_t n, uint32_t d) { return tr_budget_monomials(n, d); }
uint64_t tr_math_budget_homogeneous(uint32_t n, uint32_t d) { return tr_budget_homogeneous(n, d); }
uint64_t tr_math_budget_product(const uint64_t *dims, uint32_t k) { return k && !dims ? TR_BUDGET_UNKNOWN : tr_budget_product(dims, k); }
uint64_t tr_math_budget_power2(uint32_t k) { return tr_budget_power2(k); }
uint64_t tr_math_budget_add(uint64_t a, uint64_t b) { return tr_budget_add(a, b); }
uint64_t tr_math_budget_mul(uint64_t a, uint64_t b) { return tr_budget_mul(a, b); }

int32_t tr_math_budget_check(const tr_math_budget *b, const tr_math_need *n, tr_math_refusal *r) {
    if (!b || !n) { tr_budget_none(r); return TR_MATH_INVALID; }
    return tr_budget_check(b, n, r);
}

int32_t tr_math_budget_choose(const tr_math_budget *b, const tr_math_need *expanded, double expanded_cost,
                              const tr_math_need *factored, double factored_cost, int32_t allow_factored,
                              tr_math_choice *out) {
    if (!b || !expanded || !out || (allow_factored && !factored) || !isfinite(expanded_cost) || expanded_cost < 0 ||
        (allow_factored && (!isfinite(factored_cost) || factored_cost < 0))) return TR_MATH_INVALID;
    out->expanded_cost = expanded_cost;
    out->factored_cost = allow_factored ? factored_cost : 0;
    int e = tr_budget_check(b, expanded, &out->expanded) == 0;
    int f = 0;
    if (allow_factored) f = tr_budget_check(b, factored, &out->factored) == 0;
    else tr_budget_none(&out->factored);
    out->choice = -1; out->reason = -1;
    if (e && f) {
        out->choice = factored_cost < expanded_cost ? TR_REPR_FACTORED : TR_REPR_EXPANDED;
        out->reason = TR_CHOICE_CHEAPER;
        return TR_MATH_OK;
    }
    if (e || f) {
        out->choice = e ? TR_REPR_EXPANDED : TR_REPR_FACTORED;
        out->reason = TR_CHOICE_ONLY_FIT;
        return TR_MATH_OK;
    }
    return TR_MATH_BUDGET;
}

int32_t tr_math_budget_count_cells(const tr_math_budget *b, const uint32_t *cells, const uint64_t *rows, uint64_t n,
                                   uint64_t ncells, uint64_t *distinct, tr_math_refusal *r) {
    tr_budget_none(r);
    if (!b || !distinct || (n && !cells)) return TR_MATH_INVALID;
    tr_math_need need;
    tr_budget_count_cells_need(n, ncells, rows != NULL, 0, &need);
    int32_t rc = tr_budget_check(b, &need, r);
    if (rc) return rc;
    size_t words = (size_t)((ncells + 63) / 64);
    uint64_t *bits = calloc(words ? words : 1, sizeof *bits);
    if (!bits) return TR_MATH_MEMORY;
    for (uint64_t i = 0; i < n; i++) {
        if (rows && !((rows[i >> 6] >> (i & 63)) & 1)) continue;
        uint32_t c = cells[i];
        if (c >= ncells) { free(bits); return TR_MATH_INVALID; }
        bits[c >> 6] |= (uint64_t)1 << (c & 63);
    }
    uint64_t count = 0;
    for (size_t w = 0; w < words; w++) count += (uint64_t)__builtin_popcountll(bits[w]);
    free(bits);
    rc = tr_budget_check_one(b, TR_BUDGET_OUTPUT, count, r);
    if (rc) return rc;
    *distinct = count;
    return TR_MATH_OK;
}
