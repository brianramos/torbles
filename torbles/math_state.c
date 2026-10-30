/* Bounded immutable mathematical state reuse. SPDX-License-Identifier: Apache-2.0 */
#include "math_state.h"
#include <limits.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#define STATE_MAGIC UINT32_C(0x544d5354)
#define STATE_TERMS (1+TR_MATH_DATA_VARIABLES+TR_MATH_DATA_VARIABLES*(TR_MATH_DATA_VARIABLES+1)/2)

_Static_assert(sizeof(tr_math_state_stats)==56, "math state stats ABI size");
_Static_assert(offsetof(tr_math_state_stats,capacity)==48, "math state stats ABI offset");

typedef struct {
    tr_math_domain *domain;
    unsigned char *basis;
    uint32_t basis_bytes;
    int32_t ndata;
    int64_t moments[STATE_TERMS];
    uint64_t dynamic_bytes, touched;
} state_entry;

struct tr_math_state_cache_s {
    uint32_t magic;
    int32_t capacity, entries;
    uint64_t hits, misses, puts, evictions, bytes, max_bytes, clock;
    state_entry entry[];
};

static int cache_valid(const tr_math_state_cache *cache) {
    return cache && cache->magic==STATE_MAGIC;
}
static void error_set(int32_t *error,int code) { if (error) *error=code; }
static void increment(uint64_t *counter) { if (*counter<UINT64_MAX) ++*counter; }
static uint64_t fixed_bytes(int32_t capacity) {
    return sizeof(tr_math_state_cache)+(uint64_t)capacity*sizeof(state_entry);
}
static int basis_check(const void *key,uint32_t bytes) {
    if (!key || !bytes) return TR_MATH_INVALID;
    return bytes<=TR_MATH_STATE_BASIS_KEY_MAX ? TR_MATH_OK : TR_MATH_BUDGET;
}
static int domain_valid(const tr_math_domain *domain,int32_t ndata,tr_math_domain_spec *spec) {
    int rc=tr_math_domain_get_spec(domain,spec);
    if (rc) return rc;
    if ((spec->kind!=TR_MATH_PHYSICAL && spec->kind!=TR_MATH_SNAPSHOT) ||
        spec->validity!=TR_MATH_NONNULL || spec->grouping!=TR_MATH_SCALAR ||
        spec->flags || ndata<0 || ndata>TR_MATH_DATA_VARIABLES || spec->ndata!=ndata)
        return TR_MATH_DOMAIN_ERROR;
    return spec->carrier==TR_MATH_INTEGER ? TR_MATH_OK : TR_MATH_CARRIER_ERROR;
}

/* Canonical count/linear/square/cross basis, row stride exactly ndata. */
static int basis_powers(int ndata,uint8_t *powers) {
    memset(powers,0,STATE_TERMS*TR_MATH_DATA_VARIABLES);
    int n=1;
    for (int i=0;i<ndata;++i) powers[n++*ndata+i]=1;
    for (int i=0;i<ndata;++i) for (int j=i;j<ndata;++j) {
        powers[n*ndata+i]++;
        powers[n*ndata+j]++;
        ++n;
    }
    return n;
}
static int copy_moments(const tr_math_bank *bank,int ndata,int64_t *moments) {
    uint8_t powers[STATE_TERMS*TR_MATH_DATA_VARIABLES];
    int n=basis_powers(ndata,powers);
    for (int i=0;i<n;++i) {
        int rc=tr_math_bank_moment(bank,ndata ? powers+i*ndata : NULL,&moments[i]);
        if (rc) return rc;
    }
    return TR_MATH_OK;
}
static tr_math_bank *clone_moments(const state_entry *entry,int32_t *error) {
    uint8_t powers[STATE_TERMS*TR_MATH_DATA_VARIABLES];
    int n=basis_powers(entry->ndata,powers);
    return tr_math_bank_create(entry->ndata,n,entry->ndata ? powers : NULL,entry->moments,error);
}
static void entry_clear(tr_math_state_cache *cache,int index) {
    state_entry *entry=&cache->entry[index];
    if (!entry->domain) return;
    tr_math_domain_free(entry->domain);
    free(entry->basis);
    cache->bytes-=entry->dynamic_bytes;
    --cache->entries;
    memset(entry,0,sizeof *entry);
}
static int oldest(const tr_math_state_cache *cache,int except) {
    int found=-1;
    for (int i=0;i<cache->capacity;++i) if (i!=except && cache->entry[i].domain &&
        (found<0 || cache->entry[i].touched<cache->entry[found].touched)) found=i;
    return found;
}
static uint64_t touch(tr_math_state_cache *cache) {
    if (cache->clock==UINT64_MAX) {
        /* Preserve relative LRU order without relying on counter wraparound. */
        int order[TR_MATH_STATE_CAPACITY_MAX],n=0;
        for (int i=0;i<cache->capacity;++i) if (cache->entry[i].domain) {
            int at=n;
            while (at && cache->entry[order[at-1]].touched>cache->entry[i].touched) {
                order[at]=order[at-1]; --at;
            }
            order[at]=i; ++n;
        }
        for (int i=0;i<n;++i) cache->entry[order[i]].touched=(uint64_t)i+1;
        cache->clock=(uint64_t)n;
    }
    return ++cache->clock;
}
static int find(const tr_math_state_cache *cache,const tr_math_domain *domain,
                 const void *basis,uint32_t basis_bytes,int ndata,int *index) {
    *index=-1;
    for (int i=0;i<cache->capacity;++i) {
        const state_entry *entry=&cache->entry[i];
        if (!entry->domain || entry->ndata!=ndata || entry->basis_bytes!=basis_bytes ||
            memcmp(entry->basis,basis,basis_bytes)) continue;
        int same=tr_math_domain_equal(entry->domain,domain);
        if (same<0) return same;
        if (same) { *index=i; break; }
    }
    return TR_MATH_OK;
}

tr_math_state_cache *tr_math_state_cache_create(int32_t capacity,uint64_t max_bytes,int32_t *error) {
    if (capacity<1 || capacity>TR_MATH_STATE_CAPACITY_MAX) {
        error_set(error,TR_MATH_INVALID); return NULL;
    }
    uint64_t bytes=fixed_bytes(capacity);
    if (max_bytes<bytes) { error_set(error,TR_MATH_BUDGET); return NULL; }
    tr_math_state_cache *cache=calloc(1,(size_t)bytes);
    if (!cache) { error_set(error,TR_MATH_MEMORY); return NULL; }
    cache->magic=STATE_MAGIC; cache->capacity=capacity;
    cache->bytes=bytes; cache->max_bytes=max_bytes;
    error_set(error,TR_MATH_OK); return cache;
}

tr_math_bank *tr_math_state_cache_get(tr_math_state_cache *cache,const tr_math_domain *domain,
                                      const void *basis,uint32_t basis_bytes,int32_t ndata,int32_t *error) {
    if (!cache_valid(cache)) {
        error_set(error,TR_MATH_INVALID); return NULL;
    }
    int rc=basis_check(basis,basis_bytes);
    if (rc) { error_set(error,rc); return NULL; }
    tr_math_domain_spec spec;
    rc=domain_valid(domain,ndata,&spec);
    if (rc) { error_set(error,rc); return NULL; }
    int index;
    rc=find(cache,domain,basis,basis_bytes,ndata,&index);
    if (rc) { error_set(error,rc); return NULL; }
    if (index==-1) { increment(&cache->misses); error_set(error,TR_MATH_OK); return NULL; }
    tr_math_bank *copy=clone_moments(&cache->entry[index],error);
    if (!copy) return NULL;
    increment(&cache->hits);
    cache->entry[index].touched=touch(cache);
    return copy;
}

int32_t tr_math_state_cache_put(tr_math_state_cache *cache,const tr_math_domain *domain,
                                const void *basis,uint32_t basis_bytes,const tr_math_bank *bank) {
    if (!cache_valid(cache)) return TR_MATH_INVALID;
    int rc=basis_check(basis,basis_bytes);
    if (rc) return rc;
    int ndata=tr_math_bank_variables(bank);
    if (ndata<0) return ndata;
    tr_math_domain_spec spec;
    rc=domain_valid(domain,ndata,&spec);
    if (rc) return rc;
    int64_t count;
    if ((rc=tr_math_bank_count(bank,&count))) return rc;
    if (count<0 || (uint64_t)count>spec.universe_rows) return TR_MATH_DOMAIN_ERROR;
    int64_t domain_bytes=tr_math_domain_bytes(domain);
    if (domain_bytes<0) return (int32_t)domain_bytes;
    uint64_t dynamic=(uint64_t)domain_bytes+basis_bytes;
    if (dynamic>cache->max_bytes-fixed_bytes(cache->capacity)) return TR_MATH_BUDGET;
    state_entry next; memset(&next,0,sizeof next);
    next.ndata=ndata; next.basis_bytes=basis_bytes; next.dynamic_bytes=dynamic;
    if ((rc=copy_moments(bank,ndata,next.moments))) return rc;
    int index;
    if ((rc=find(cache,domain,basis,basis_bytes,ndata,&index))) return rc;
    int replacement=index>=0;
    if (!replacement) {
        for (int i=0;i<cache->capacity;++i) if (!cache->entry[i].domain) { index=i; break; }
        if (index<0) index=oldest(cache,-1);
    }
    /* All fallible work happens before changing existing entries or counters. */
    int32_t error;
    next.domain=tr_math_domain_clone(domain,&error);
    if (!next.domain) return error;
    next.basis=malloc(basis_bytes);
    if (!next.basis) { tr_math_domain_free(next.domain); return TR_MATH_MEMORY; }
    memcpy(next.basis,basis,basis_bytes);
    uint64_t retained=cache->bytes-cache->entry[index].dynamic_bytes+dynamic;
    while (retained>cache->max_bytes) {
        int victim=oldest(cache,index);
        if (victim<0) { /* Guarded above; unreachable for valid cache state. */
            tr_math_domain_free(next.domain); free(next.basis); return TR_MATH_BUDGET;
        }
        retained-=cache->entry[victim].dynamic_bytes;
        entry_clear(cache,victim); increment(&cache->evictions);
    }
    if (cache->entry[index].domain && !replacement) increment(&cache->evictions);
    entry_clear(cache,index);
    next.touched=touch(cache);
    cache->entry[index]=next;
    ++cache->entries; cache->bytes+=dynamic; increment(&cache->puts);
    return TR_MATH_OK;
}

int32_t tr_math_state_cache_clear(tr_math_state_cache *cache) {
    if (!cache_valid(cache)) return TR_MATH_INVALID;
    for (int i=0;i<cache->capacity;++i) entry_clear(cache,i);
    cache->hits=cache->misses=cache->puts=cache->evictions=cache->clock=0;
    return TR_MATH_OK;
}
int32_t tr_math_state_cache_stats(const tr_math_state_cache *cache,tr_math_state_stats *output) {
    if (!cache_valid(cache) || !output) return TR_MATH_INVALID;
    *output=(tr_math_state_stats){cache->hits,cache->misses,cache->puts,cache->evictions,
                                  cache->bytes,cache->max_bytes,cache->capacity,cache->entries};
    return TR_MATH_OK;
}
void tr_math_state_cache_free(tr_math_state_cache *cache) {
    if (!cache) return;
    if (cache_valid(cache)) {
        (void)tr_math_state_cache_clear(cache);
        cache->magic=0;
    }
    free(cache);
}
