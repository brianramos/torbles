/* Copyright 2026 Brian Richard RAMOS
 * SPDX-License-Identifier: Apache-2.0 */
#ifndef TORBLES_PERIODIC_H
#define TORBLES_PERIODIC_H
#include "torble.h"
#ifdef __cplusplus
extern "C" {
#endif

enum {
    TP_MAX_KEYS = 3, TP_MAX_MEASURES = 2, TP_MAX_GROUPS = 4096,
    TP_MAX_CYCLE_WORDS = 4096, TP_MAX_PROGRAM = 1024, TP_MAX_STACK = 48,
    TP_SECTOR_WORDS = 4096, TP_MAX_SECTOR_WORDS = 1 << 20, TP_MIN_REPEATS = 8
};
enum { TP_SUM = 1, TP_MIN = 2, TP_MAX = 4 };

/* valid = -1 means all rows have a value. Otherwise it is a one-bit
 * physical validity column. A nullable key reserves radix-1 for NULL. */
typedef struct { int32_t col, valid; uint32_t radix; } tp_key_t;
typedef struct { int32_t col, valid; uint32_t flags; } tp_measure_t;
typedef struct {
    uint32_t nkeys, nmeasures, groups;
    tp_key_t keys[TP_MAX_KEYS];
    tp_measure_t measures[TP_MAX_MEASURES];
} tp_plan_t;
typedef struct {
    uint64_t count, valid[TP_MAX_MEASURES], sums[TP_MAX_MEASURES];
    uint32_t mins[TP_MAX_MEASURES], maxs[TP_MAX_MEASURES];
} tp_state_t;

/* The caller owns a current word-period certificate for every plane read
 * by the plan. prog already incorporates any SQL validity and visibility
 * predicates. No result preserves physical row identities or order.
 * out holds plan->groups entries, indexed by mixed-radix codes in key
 * order. Empty extrema are UINT32_MAX/0 and have valid count zero.
 * A measure with flags=0 needs only its validity column and may use col=-1.
 * Returns 0 on success and -1 for an unsupported or invalid plan. */
int at_periodic_reduce(const atable_t *t, uint32_t cycle_words, const pinstr_t *prog,
                        int32_t nprog, const tp_plan_t *plan, tp_state_t *out);

/* Physical word ranges. cycle_words=0 scans all complete words. The
 * final descriptor alone may carry tail_rows (1..63). */
typedef struct { uint32_t first_word, full_words, cycle_words, tail_rows; } tp_sector_t;
typedef struct {
    uint64_t periodic_sectors, ordinary_sectors, representative_words;
} tp_sector_metrics_t;

/* Prove a common period only for selected physical columns. A sector
 * with no proof has cycle_words=0. sector_words=0 requests one global
 * sector; otherwise output ceil(table words / sector_words) descriptors.
 * Callers serialize against writes and anchor the source revision(s).
 * max_words bounds word-period discovery for unindexed dependencies.
 * Both functions accept a NULL metrics pointer. */
int at_periodic_plan_sectors(const atable_t *t, uint32_t columns_mask,
                             uint32_t sector_words, uint32_t max_words,
                             tp_sector_t *sectors, uint32_t capacity,
                             tp_sector_metrics_t *metrics);
/* A current, validated sector index is available for at least one selected
 * physical column. Its entries may be motifs or ordinary sectors. The
 * extended planner counts each index decision it uses. Composed indexed
 * periods are bounded by the execution limit, independently of max_words. */
int at_periodic_has_persisted(const atable_t *t, uint32_t columns_mask);
/* Stitch indexed local motifs into one global cycle by comparing only
 * bounded motif words and a bounded final literal. cycle=0 is a miss. */
int at_periodic_ingested_global(const atable_t *t, uint32_t columns_mask,
                                uint32_t *cycle, uint64_t *uses);
int at_periodic_plan_sectors_ex(const atable_t *t, uint32_t columns_mask,
                                uint32_t sector_words, uint32_t max_words,
                                tp_sector_t *sectors, uint32_t capacity,
                                tp_sector_metrics_t *metrics, uint64_t *persisted_uses);
int at_periodic_reduce_sectors(const atable_t *t, const tp_sector_t *sectors,
                               uint32_t nsectors, const pinstr_t *prog, int32_t nprog,
                               const tp_plan_t *plan, tp_state_t *out,
                               tp_sector_metrics_t *metrics);
/* Execute the same plan directly against checked V-edge blocks. cycle is
 * nonzero for a whole-table motif; otherwise use native sector descriptors. */
int at_periodic_reduce_encoded(const atable_t *t, uint32_t cycle, const tp_sector_t *sectors,
                               uint32_t nsectors, uint32_t mask, const pinstr_t *prog, int32_t nprog,
                               const tp_plan_t *plan, tp_state_t *out, uint64_t *payload);
/* Materialize only the requested row bitmap. Its predicate motifs can be
 * shorter than the table's value cycles. Metrics are count, referenced
 * encoded bytes, and evaluated predicate words. */
int at_cycle_filter_bitmap(const atable_t *t, const pinstr_t *prog, int32_t nprog,
                           const uint64_t *visible, uint64_t *out, uint64_t metrics[3]);

#ifdef __cplusplus
}
#endif
#endif
