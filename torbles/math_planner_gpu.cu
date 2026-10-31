/* FND-008 on a CUDA device: the selectivity sample of the planner (math_planner.h). One thread per sampled word
 * (or every word when exact) counts the word's rows and selected rows; each thread adds its counts with one atomic,
 * so the counts equal tr_planner_sample's whatever the thread order. SPDX-License-Identifier: Apache-2.0 */
#include "math_cuda.h"
#include "math_planner.h"
#include "math_handles.h"

TR_MGPU_ABI(tr_planner_selectivity, 1);          /* the ctypes boundary (math_handles.h) */

TR_KERNEL void tr_planner_sample_kernel(const uint64_t *selected, uint64_t rows, uint64_t n, uint64_t words,
                                        uint64_t *acc) {
    uint64_t sampled_rows = 0, sampled_selected = 0;
    TR_GRID_LOOP(i, n) {
        uint64_t w = tr_planner_sample_word(i, n, words), mask = tr_reg_rowmask(rows, w);
        sampled_rows += (uint64_t)tr_popc64(mask);
        sampled_selected += (uint64_t)tr_popc64((selected ? selected[w] : ~(uint64_t)0) & mask);
    }
    if (sampled_rows) tr_atomic_add_u64(&acc[0], sampled_rows);
    if (sampled_selected) tr_atomic_add_u64(&acc[1], sampled_selected);
}

#ifdef __cplusplus
extern "C" {
#endif
/* Same contract as tr_planner_sample; the device copy of the selection (8 bytes per word, plus the two counters) is
 * budgeted (bytes) before anything is allocated (budget NULL: the defaults). */
int tr_mgpu_planner_sample(const uint64_t *selected, uint64_t rows, uint64_t sample_words, int32_t exact,
                           const tr_math_budget *budget, tr_planner_selectivity *s, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!s || (!exact && !sample_words)) return TR_REG_E_INVALID;
    tr_math_budget dflt;
    if (!budget) { tr_budget_default(&dflt); budget = &dflt; }
    uint64_t words = (rows + 63) / 64, n = exact || sample_words >= words ? words : sample_words;
    uint64_t bytes = tr_budget_add(selected ? tr_budget_bitmap_bytes(rows) : 0, 16);
    int rc = tr_budget_check_one(budget, TR_BUDGET_BYTES, bytes, refusal);
    if (rc) return rc;
    memset(s, 0, sizeof *s);
    s->rows = rows; s->words = words; s->sample_words = n;
    uint64_t acc[2] = {0, 0}, *dacc = NULL, *dsel = NULL;
    rc = tr_cuda_alloc((void **)&dacc, sizeof acc);
    if (!rc && selected && words) {
        rc = tr_cuda_alloc((void **)&dsel, (size_t)words * 8);
        if (!rc) rc = tr_cuda_copy_in(dsel, selected, (size_t)words * 8);
    }
    if (!rc && n) { TR_LAUNCH(tr_planner_sample_kernel, n, dsel, rows, n, words, dacc); rc = tr_cuda_sync(); }
    if (!rc) rc = tr_cuda_copy_out(acc, dacc, sizeof acc);
    tr_cuda_free(dacc); tr_cuda_free(dsel);
    if (rc) return rc;
    s->sample_rows = acc[0]; s->sample_selected = acc[1];
    s->exact = exact || n == words;
    s->method = exact ? TR_PLANNER_EXACT : n == words ? TR_PLANNER_COVERED : TR_PLANNER_SAMPLED;
    tr_planner_finish(s);
    return 0;
}
#ifdef __cplusplus
}
#endif
