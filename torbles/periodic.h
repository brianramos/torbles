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
    TP_MAX_CYCLE_WORDS = 4096, TP_MAX_PROGRAM = 1024, TP_MAX_STACK = 48
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

#ifdef __cplusplus
}
#endif
#endif
