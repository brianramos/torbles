
/* Independent native state ownership/failure harness. SPDX-License-Identifier: Apache-2.0
 * Reproduce with tests/validate_math_state_native.py; linked only in /tmp. */
#include "math_state.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static int fail_allocation=-1;
void *__real_malloc(size_t n);
void *__real_calloc(size_t n,size_t k);
void *__wrap_malloc(size_t n) {
    if (fail_allocation==0) { fail_allocation=-1; return NULL; }
    if (fail_allocation>0) --fail_allocation;
    return __real_malloc(n);
}
void *__wrap_calloc(size_t n,size_t k) {
    if (fail_allocation==0) { fail_allocation=-1; return NULL; }
    if (fail_allocation>0) --fail_allocation;
    return __real_calloc(n,k);
}
static tr_math_domain *domain(const char *key,int carrier,int validity,uint64_t rows) {
    tr_math_domain_spec spec={1,TR_MATH_SNAPSHOT,carrier,1,1,validity,TR_MATH_SCALAR,0,1,rows};
    int32_t error;
    tr_math_domain *out=tr_math_domain_create(&spec,key,(uint32_t)strlen(key),NULL,&error);
    assert(out && !error); return out;
}
static tr_math_bank *bank(int64_t count,int64_t sum,int64_t squares) {
    uint8_t e[]={0,1,2}; int64_t values[]={count,sum,squares}; int32_t error;
    tr_math_bank *out=tr_math_bank_create(1,3,e,values,&error); assert(out && !error); return out;
}
static tr_math_state_stats stats(tr_math_state_cache *cache) {
    tr_math_state_stats out; assert(!tr_math_state_cache_stats(cache,&out)); return out;
}
static void equalstats(tr_math_state_cache *cache,tr_math_state_stats before) {
    tr_math_state_stats after=stats(cache); assert(!memcmp(&before,&after,sizeof before));
}
static tr_math_bank *get(tr_math_state_cache *cache,tr_math_domain *d,const char *key) {
    int32_t error=123; tr_math_bank *out=tr_math_state_cache_get(cache,d,key,(uint32_t)strlen(key),1,&error);
    assert(!error); return out;
}
static void value(tr_math_bank *b,int64_t expected) {
    uint8_t e[]={2}; int64_t out; assert(b); assert(!tr_math_bank_moment(b,e,&out)); assert(out==expected);
}
int main(void) {
    int32_t error;
    assert(!tr_math_state_cache_create(0,100000,&error) && error==TR_MATH_INVALID);
    assert(!tr_math_state_cache_create(129,100000,&error) && error==TR_MATH_INVALID);
    assert(!tr_math_state_cache_create(2,1,&error) && error==TR_MATH_BUDGET);
    tr_math_state_cache *cache=tr_math_state_cache_create(2,1000000,&error); assert(cache && !error);
    uint64_t base=stats(cache).bytes;
    tr_math_domain *a=domain("A",TR_MATH_INTEGER,TR_MATH_NONNULL,10);
    tr_math_domain *b=domain("B",TR_MATH_INTEGER,TR_MATH_NONNULL,10);
    tr_math_domain *c=domain("C",TR_MATH_INTEGER,TR_MATH_NONNULL,10);
    tr_math_domain *wrong=domain("A",TR_MATH_RATIONAL,TR_MATH_NONNULL,10);
    tr_math_domain *nullable=domain("A",TR_MATH_INTEGER,TR_MATH_SQL_NULLS,10);
    tr_math_domain *short_domain=domain("short",TR_MATH_INTEGER,TR_MATH_NONNULL,1);
    tr_math_bank *input=bank(3,6,14);
    assert(!get(cache,a,"x"));
    assert(!tr_math_state_cache_put(cache,a,"x",1,input));
    tr_math_domain_spec changed;
    assert(!tr_math_domain_get_spec(a,&changed)); ++changed.revision;
    tr_math_domain *revision=tr_math_domain_create(&changed,"A",1,NULL,&error);
    assert(revision && !error && !get(cache,revision,"x"));
    tr_math_domain_free(revision);
    tr_math_bank *owned=get(cache,a,"x"); value(owned,14);
    assert(!tr_math_state_cache_put(cache,b,"x",1,input));
    tr_math_bank *temporary=get(cache,a,"x"); value(temporary,14); tr_math_bank_free(temporary);
    assert(!tr_math_state_cache_put(cache,c,"x",1,input));
    assert(!get(cache,b,"x"));
    temporary=get(cache,a,"x"); value(temporary,14); tr_math_bank_free(temporary);
    assert(stats(cache).entries==2 && stats(cache).evictions==1);
    tr_math_state_stats before=stats(cache);
    assert(tr_math_state_cache_put(cache,wrong,"x",1,input)==TR_MATH_CARRIER_ERROR); equalstats(cache,before);
    assert(tr_math_state_cache_put(cache,nullable,"x",1,input)==TR_MATH_DOMAIN_ERROR); equalstats(cache,before);
    assert(tr_math_state_cache_put(cache,short_domain,"x",1,input)==TR_MATH_DOMAIN_ERROR); equalstats(cache,before);
    assert(tr_math_state_cache_put(cache,a,"x",65537,input)==TR_MATH_BUDGET); equalstats(cache,before);
    for (int i=0;i<3;++i) {
        fail_allocation=i;
        assert(tr_math_state_cache_put(cache,b,"x",1,input)==TR_MATH_MEMORY);
        fail_allocation=-1; equalstats(cache,before);
    }
    fail_allocation=0;
    assert(!tr_math_state_cache_get(cache,a,"x",1,1,&error) && error==TR_MATH_MEMORY);
    fail_allocation=-1; equalstats(cache,before);
    tr_math_bank *replacement=bank(4,10,30);
    assert(!tr_math_state_cache_put(cache,a,"x",1,replacement));
    temporary=get(cache,a,"x"); value(temporary,30); tr_math_bank_free(temporary);
    value(owned,14);
    tr_math_bank_free(replacement);
    assert(!tr_math_state_cache_clear(cache));
    assert(stats(cache).entries==0 && stats(cache).bytes==base && stats(cache).hits==0);
    const unsigned char nul_a[]={'x',0,'a'},nul_b[]={'x',0,'b'};
    assert(!tr_math_state_cache_put(cache,a,nul_a,sizeof nul_a,input));
    temporary=tr_math_state_cache_get(cache,a,nul_b,sizeof nul_b,1,&error);
    assert(!temporary && !error);
    temporary=tr_math_state_cache_get(cache,a,nul_a,sizeof nul_a,1,&error);
    value(temporary,14); tr_math_bank_free(temporary);
    assert(!tr_math_state_cache_put(cache,a,"x",1,input));
    tr_math_bank_free(input);
    tr_math_state_cache_free(cache);
    value(owned,14); tr_math_bank_free(owned);
    cache=tr_math_state_cache_create(2,base+(uint64_t)tr_math_domain_bytes(a)+1,&error);
    assert(cache && !error);
    input=bank(3,6,14);
    assert(!tr_math_state_cache_put(cache,a,"x",1,input));
    before=stats(cache);
    assert(tr_math_state_cache_put(cache,b,"long",4,input)==TR_MATH_BUDGET); equalstats(cache,before);
    assert(!tr_math_state_cache_put(cache,b,"x",1,input));
    assert(stats(cache).entries==1 && stats(cache).evictions==1 && stats(cache).bytes<=stats(cache).max_bytes);
    assert(!get(cache,a,"x"));
    temporary=get(cache,b,"x"); value(temporary,14); tr_math_bank_free(temporary);
    tr_math_bank_free(input); tr_math_state_cache_free(cache);
    tr_math_domain_free(a);tr_math_domain_free(b);tr_math_domain_free(c);
    tr_math_domain_free(wrong);tr_math_domain_free(nullable);tr_math_domain_free(short_domain);
    puts("ASAN/UBSAN state ownership, allocation-failure atomicity, LRU and budget checks passed.");
    return 0;
}
