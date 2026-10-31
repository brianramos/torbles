/* FND-008: complete-plan choice with its cost evidence, over the candidates of the registry (math_registry.h).
 * SPDX-License-Identifier: Apache-2.0
 *
 * Selectivity. tr_planner_sample counts the selected rows of a deterministic sample of the selection's 64-row words
 * (word i * (words - 1) / (n - 1) for i < n; every word when exact or when n >= words), on the CPU or with the CUDA
 * kernel (tr_mgpu_planner_sample, equal counts). The sampled words are counted exactly, so lower = sampled selected and
 * upper = lower + the rows outside the sample are exact bounds; the costs use the estimate rows * (k + 1) / (m + 2)
 * (k selected among m sampled rows, rounded half up, clamped to the bounds). A sample with no selected row
 * (zero_sample) is no evidence of emptiness: only an exact count of zero (exact = 1) sets empty_proven, which is what
 * the EMPTY candidate requires; otherwise the plan still scans. The planner's exactness bound for POWER / BITPLANE
 * uses `upper`, never the estimate.
 *
 * Choice. Every candidate's cost is its amounts times the coefficients of its module and device (the registry's
 * descriptors, or an override table [TR_REG_MODULES][2][TR_COST_COMPONENTS]), component by component; the fusion
 * component is -coef * (batch - 1) / batch times the cost of the components shared by the batch (counted for batch
 * unfused passes), so total = the fused execution's cost. Costs are relative work units, NOT time. A candidate the
 * registry refused (law, conversion, proof, bound, stale), or whose needs exceed the budget, is rejected with its
 * reason and keeps its amounts and costs; the cheapest admissible candidate is chosen (the first on a tie). With no
 * admissible candidate the plan is refused: TR_MATH_BUDGET when one was over budget, else the first refusal's code.
 *
 * Forced controls. A force (a parameter, or the environment variable TORBLES_MATH_PLAN_FORCE) names a representation
 * and optionally where it runs: "<representation>[@cpu|@cuda|@reuse][:cpu|:gpu]" (finalization), e.g.
 * "histogram@cuda:cpu"; the variable may list one entry per request kind, comma-separated ("power,verify@cuda"), and
 * a request uses the entry naming one of its representations. The cheapest admissible candidate matching the force
 * runs; when every match is rejected the plan is refused with the first match's code, and when nothing matches it is
 * refused with TR_MATH_UNSUPPORTED: a forced representation is never replaced. A force naming @cuda adds the CUDA
 * candidates. Evidence and candidates are fixed-layout structs (sizes checked in math_planner.c). */
#ifndef TORBLES_MATH_PLANNER_H
#define TORBLES_MATH_PLANNER_H
#include "math_registry.h"
#ifdef __cplusplus
extern "C" {
#endif

#define TR_PLANNER_ABI 1
#define TR_PLANNER_SAMPLE_WORDS 32
enum { TR_PLANNER_SAMPLED = 0, TR_PLANNER_EXACT = 1, TR_PLANNER_COVERED = 2 };   /* method */
/* 80 bytes */
typedef struct {
    uint64_t rows, words, sample_words, sample_rows, sample_selected;
    uint64_t estimate, lower, upper;          /* selected rows: the costs' estimate, exact lower and upper bounds */
    int32_t exact, empty_proven, zero_sample, method;
} tr_planner_selectivity;

/* the i-th sampled word of n among words (n >= words: word i) */
TR_NUM_HD uint64_t tr_planner_sample_word(uint64_t i, uint64_t n, uint64_t words) {
    if (n >= words) return i;
    if (n <= 1) return 0;
    return (uint64_t)(((unsigned __int128)i * (words - 1)) / (n - 1));
}
/* the bounds, estimate and proofs of counted sample words (rows, words, sample_* and exact set) */
static inline void tr_planner_finish(tr_planner_selectivity *s) {
    if (s->exact) {
        s->estimate = s->lower = s->upper = s->sample_selected;
        s->empty_proven = s->sample_selected == 0; s->zero_sample = 0;
        return;
    }
    uint64_t m = s->sample_rows, k = s->sample_selected, d = m + 2;
    s->lower = k; s->upper = k + (s->rows - m);
    unsigned __int128 est = (unsigned __int128)(s->rows / d) * (k + 1) +
                            ((unsigned __int128)(s->rows % d) * (k + 1) * 2 + d) / ((unsigned __int128)d * 2);
    uint64_t e = (uint64_t)est;
    s->estimate = e < s->lower ? s->lower : e > s->upper ? s->upper : e;
    s->zero_sample = k == 0; s->empty_proven = 0;
}

enum { TR_PLANNER_S_CHOSEN = 1, TR_PLANNER_S_ADMISSIBLE = 2, TR_PLANNER_S_REJECTED = 3 };
enum { TR_PLANNER_R_CHEAPEST = 1,        /* chosen: the cheapest admissible candidate */
       TR_PLANNER_R_FORCED = 2,          /* chosen: the force's (cheapest admissible) match */
       TR_PLANNER_R_COSTLIER = 3,        /* admissible, not chosen: costlier */
       TR_PLANNER_R_NOT_FORCED = 4,      /* admissible, not chosen: another representation was forced */
       TR_PLANNER_R_INVALID = 5,         /* rejected by the registry: refusal says why */
       TR_PLANNER_R_BUDGET = 6,          /* rejected: needs over the budget (budget says which) */
       TR_PLANNER_R_FORCE_REFUSED = 7,   /* plan refused: every match of the force is rejected */
       TR_PLANNER_R_FORCE_UNMATCHED = 8, /* plan refused: no candidate matches the force */
       TR_PLANNER_R_NONE_ADMISSIBLE = 9, /* plan refused: no admissible candidate */
       TR_PLANNER_R_SEARCH = 10 };       /* plan refused before the choice: the request or the candidate search */
enum { TR_PLANNER_REQ_POLYSUM = 16, TR_PLANNER_REQ_VERIFY = 8 };   /* = TR_REG_REQ_* */
/* representation 0: no force; device / finalize / reuse -1: any */
typedef struct { int32_t representation, device, finalize, reuse; } tr_planner_force;
/* the plan: every candidate with its status, reason, amounts and costs. 6168 bytes. */
typedef struct {
    int32_t abi, request, ncandidates, chosen;     /* chosen: candidate index, -1 when refused */
    int32_t forced, reason, code, devices;         /* forced: a force was in effect; code: the plan's native code */
    tr_planner_force force;
    tr_math_refusal refusal;                       /* a plan-level budget refusal (search, request kernel limits) */
    tr_planner_selectivity selectivity;
    tr_reg_candidate candidates[TR_REG_MAX_CANDIDATES];
} tr_planner_evidence;

/* exported (math_planner.c) */
int32_t tr_planner_sample(const uint64_t *selected, uint64_t rows, uint64_t sample_words, int32_t exact,
                          tr_planner_selectivity *);
/* text NULL or "": no force; request: TR_PLANNER_REQ_* (the entry naming one of its representations), or 0 for an
 * explicit force (its first entry, whatever it names: a representation the request lacks is then refused) */
int32_t tr_planner_force_parse(const char *text, int32_t request, tr_planner_force *);
/* costs, budgets and the choice over given candidates (statuses and costs written in place, copied to ev) */
int32_t tr_planner_choose(tr_reg_candidate *, int32_t n, const tr_math_budget *, const double *coefficients,
                          const tr_planner_force *, tr_planner_evidence *);
/* a polynomial aggregate request: sel NULL samples on the CPU (TR_PLANNER_SAMPLE_WORDS); force NULL reads
 * TORBLES_MATH_PLAN_FORCE; coefficients NULL uses the registry's */
int32_t tr_planner_polysum(const tr_reg_request *, const tr_reg_state *cached, int32_t devices,
                           const tr_math_budget *, const tr_planner_selectivity *sel, const tr_planner_force *,
                           const double *coefficients, tr_planner_evidence *);
/* a DISC-001 verification: the CPU verifier or the CUDA kernel */
int32_t tr_planner_verify(const tr_disc_view *, int32_t devices, const tr_math_budget *, const tr_planner_force *,
                          const double *coefficients, tr_planner_evidence *);
#ifdef __cplusplus
}
#endif
#endif
