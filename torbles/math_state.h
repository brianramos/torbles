/* Bounded immutable mathematical state reuse. SPDX-License-Identifier: Apache-2.0 */
#ifndef TORBLES_MATH_STATE_H
#define TORBLES_MATH_STATE_H
#include "math_domain.h"
#ifdef __cplusplus
extern "C" {
#endif

#define TR_MATH_STATE_CAPACITY_MAX 128
#define TR_MATH_STATE_BASIS_KEY_MAX 65536
#define TR_MATH_STATE_BYTES_DEFAULT UINT64_C(8388608)

typedef struct tr_math_state_cache_s tr_math_state_cache;

/* Six uint64 fields followed by two int32 fields; sizeof==56. bytes includes
 * the fixed cache allocation and owned domain/basis descriptors, not clones
 * returned to callers. Counters saturate rather than wrap. */
typedef struct {
    uint64_t hits, misses, puts, evictions, bytes, max_bytes;
    int32_t capacity, entries;
} tr_math_state_stats;

/* capacity is 1..128. max_bytes bounds retained cache allocations, including
 * its fixed entry array; it must accommodate that array. One additional domain
 * and basis key may be staged during put, before eviction, to preserve existing
 * entries on an allocation/validation failure. Each semantic key is <=64KiB.
 * Calls on one mutable cache must be externally serialized. */
tr_math_state_cache *tr_math_state_cache_create(int32_t capacity, uint64_t max_bytes,
                                                int32_t *error);

/* Domain and ordered basis identity are separate exact descriptors. Supported
 * domains: PHYSICAL/SNAPSHOT, INTEGER, NONNULL, SCALAR, flags==0, ndata 0..4.
 * The basis key includes ordered logical coordinate/codec/degree information.
 * ndata must match the domain; a valid absent entry returns NULL with error=OK.
 * A hit returns an individually owned immutable bank, independent of cache,
 * domain, input-bank and later eviction lifetimes. */
tr_math_bank *tr_math_state_cache_get(tr_math_state_cache *, const tr_math_domain *,
                                      const void *basis_key, uint32_t basis_bytes,
                                      int32_t ndata, int32_t *error);

/* Copies the input bank's complete canonical moment values and clones domain
 * and basis bytes. The bank's count cannot exceed domain.universe_rows. No
 * input is retained or modified. Failed puts leave existing entries/stats
 * unchanged. Replacing an existing identity is a put, not an eviction. */
int32_t tr_math_state_cache_put(tr_math_state_cache *, const tr_math_domain *,
                                const void *basis_key, uint32_t basis_bytes,
                                const tr_math_bank *);

/* Clear releases all entry payloads and resets counters/clock; capacity and
 * the fixed allocation remain. Free accepts NULL. Neither function invalidates
 * banks previously returned by get. */
int32_t tr_math_state_cache_clear(tr_math_state_cache *);
int32_t tr_math_state_cache_stats(const tr_math_state_cache *, tr_math_state_stats *);
void tr_math_state_cache_free(tr_math_state_cache *);

#ifdef __cplusplus
}
#endif
#endif
