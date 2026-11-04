/* Copyright 2026 Brian Richard RAMOS
 * SPDX-License-Identifier: Apache-2.0 */
#ifndef TORBLES_CYCLE_STORAGE_H
#define TORBLES_CYCLE_STORAGE_H
#include <stddef.h>
#include "torble.h"

typedef struct {
    const uint64_t *data;           /* owned, aligned words in native byte order */
    uint16_t period, words;         /* period zero is literal */
} tb_cycle_v_t;

typedef struct tb_cycle_store_s {
    uint16_t first[MAXCOL];         /* first encoded value plane, per physical column */
    uint16_t nplanes;
    size_t planebytes, payload_bytes;
    uint8_t (*ids)[4];              /* independent physical edges in file order */
    tb_cycle_v_t *blocks;           /* physical plane-major by owner cycle_sectors */
    uint64_t *data;
} tb_cycle_store_t;

/* Borrow an immutable sector's value-carrying XOR edges. The caller keeps
 * the owning table alive and serializes queries against mutations. All
 * complete words are exact. Callers mask unoccupied bits of a final partial
 * motif word, just as they mask the final word of an ordinary query. */
static inline int at_cycle_v_planes(const atable_t *t, uint32_t sector, uint32_t col,
                                    tb_cycle_v_t out[NB]) {
    if (!t || col >= MAXCOL || col >= (uint32_t)t->k || t->vflag[col] ||
        __atomic_load_n(&t->revision, __ATOMIC_ACQUIRE) != t->cycle_revision) return -1;
    const atable_t *owner = t->cycle_sources[col];
    uint32_t src = col;
    if (owner) src = t->cycle_columns[col]; else owner = t;
    int width = t->width[col];
    if (width < 1 || width >= NB || !owner->cycle_store || sector >= owner->cycle_sectors ||
        src >= MAXCOL || src >= (uint32_t)owner->k || owner->vflag[src] || owner->width[src] != width ||
        owner->cycle_store->first[src] == UINT16_MAX || owner->n != t->n || owner->nw != t->nw ||
        __atomic_load_n(&owner->revision, __ATOMIC_ACQUIRE) != owner->cycle_revision) return -1;
    if (out) {
        size_t first = (size_t)owner->cycle_store->first[src] * owner->cycle_sectors + sector;
        for (int j = 0; j < width; ++j)
            out[j] = owner->cycle_store->blocks[first + (size_t)j * owner->cycle_sectors];
    }
    return width;
}

static inline int tb_cycle_has_mask(const atable_t *t, uint32_t mask) {
    if (!t || t->k < 1 || t->k > MAXCOL || !mask || (mask >> t->k)) return 0;
    for (; mask; mask &= mask - 1)
        if (at_cycle_v_planes(t, 0, (uint32_t)__builtin_ctz(mask), NULL) < 0) return 0;
    return 1;
}

static inline uint64_t tb_cycle_v_word(const tb_cycle_v_t *plane, uint32_t local_word) {
    unsigned period = plane->period;
    if (period && local_word >= period) local_word = period == 1 ? 0 : local_word % period;
    return plane->data[local_word];
}
#ifdef __cplusplus
extern "C" {
#endif
atable_t *at_load_cycles_lazy(const char *path, int *err);
int at_materialize(atable_t *t);
int at_has_dense(const atable_t *t);
uint64_t at_cycle_packed_bytes(const atable_t *t);
int at_cycle_validate(const atable_t *t, uint64_t *bad, uint64_t maxbad, uint64_t *nbad);
#ifdef __cplusplus
}
#endif
#endif
