/* Immutable typed mathematical domains. SPDX-License-Identifier: Apache-2.0 */
#include "math_domain.h"
#include <stdlib.h>
#include <string.h>

#define DOMAIN_MAGIC UINT32_C(0x544d444f)
struct tr_math_domain_s {
    uint32_t magic, key_bytes;
    tr_math_domain_spec spec;
    tr_math_domain *origin;
    unsigned char *key;
};
_Static_assert(sizeof(tr_math_domain_spec)==48,"domain spec ABI size");
_Static_assert(offsetof(tr_math_domain_spec,revision)==32,"domain spec ABI alignment");

static void error_set(int32_t *error,int code) { if (error) *error=code; }
static int valid(const tr_math_domain *domain) { return domain && domain->magic==DOMAIN_MAGIC; }
static int valid_spec(const tr_math_domain_spec *spec) {
    return spec && spec->abi_version==TR_MATH_DOMAIN_ABI &&
        spec->kind>=TR_MATH_PHYSICAL && spec->kind<=TR_MATH_GEOMETRY &&
        spec->carrier>=TR_MATH_INTEGER && spec->carrier<=TR_MATH_GF2 &&
        spec->nvars>=0 && spec->nvars<=TR_MATH_VARIABLES &&
        spec->ndata>=0 && spec->ndata<=TR_MATH_DATA_VARIABLES && spec->ndata<=spec->nvars &&
        spec->validity>=TR_MATH_NONNULL && spec->validity<=TR_MATH_SQL_NULLS &&
        spec->grouping>=TR_MATH_SCALAR && spec->grouping<=TR_MATH_SPARSE_GROUPS &&
        !(spec->flags & ~TR_MATH_AGGREGATE) &&
        (!spec->flags || spec->kind==TR_MATH_FORMAL);
}

tr_math_domain *tr_math_domain_create(const tr_math_domain_spec *spec,const void *key,
                                      uint32_t key_bytes,const tr_math_domain *origin,int32_t *error) {
    error_set(error,TR_MATH_OK);
    if (!valid_spec(spec) || !key || !key_bytes) {
        error_set(error,TR_MATH_INVALID); return NULL;
    }
    if (key_bytes>TR_MATH_DOMAIN_KEY_MAX) { error_set(error,TR_MATH_BUDGET); return NULL; }
    if (origin && (!valid(origin) || spec->kind!=TR_MATH_FORMAL ||
            (origin->spec.kind!=TR_MATH_PHYSICAL && origin->spec.kind!=TR_MATH_SNAPSHOT) ||
            origin->spec.ndata!=spec->ndata)) {
        error_set(error,TR_MATH_DOMAIN_ERROR); return NULL;
    }
    if (spec->kind==TR_MATH_FORMAL && ((spec->ndata || spec->flags) && !origin)) {
        error_set(error,TR_MATH_DOMAIN_ERROR); return NULL;
    }
    tr_math_domain *out=calloc(1,sizeof *out);
    if (!out) { error_set(error,TR_MATH_MEMORY); return NULL; }
    out->key=malloc(key_bytes);
    if (!out->key) { free(out); error_set(error,TR_MATH_MEMORY); return NULL; }
    memcpy(out->key,key,key_bytes); out->key_bytes=key_bytes; out->spec=*spec;
    if (origin) {
        out->origin=tr_math_domain_clone(origin,error);
        if (!out->origin) { free(out->key); free(out); return NULL; }
    }
    out->magic=DOMAIN_MAGIC;
    return out;
}

tr_math_domain *tr_math_domain_clone(const tr_math_domain *domain,int32_t *error) {
    if (!valid(domain)) { error_set(error,TR_MATH_INVALID); return NULL; }
    return tr_math_domain_create(&domain->spec,domain->key,domain->key_bytes,domain->origin,error);
}
tr_math_domain *tr_math_domain_with_aggregate(const tr_math_domain *domain,int32_t aggregate,int32_t *error) {
    if (!valid(domain) || (aggregate!=0 && aggregate!=1)) {
        error_set(error,TR_MATH_INVALID); return NULL;
    }
    if (domain->spec.kind!=TR_MATH_FORMAL || (aggregate && !domain->origin)) {
        error_set(error,TR_MATH_DOMAIN_ERROR); return NULL;
    }
    tr_math_domain_spec spec=domain->spec;
    spec.flags=aggregate ? TR_MATH_AGGREGATE : 0;
    return tr_math_domain_create(&spec,domain->key,domain->key_bytes,domain->origin,error);
}
tr_math_domain *tr_math_domain_origin(const tr_math_domain *domain,int32_t *error) {
    if (!valid(domain)) { error_set(error,TR_MATH_INVALID); return NULL; }
    if (!domain->origin) { error_set(error,TR_MATH_OK); return NULL; }
    return tr_math_domain_clone(domain->origin,error);
}
int32_t tr_math_domain_get_spec(const tr_math_domain *domain,tr_math_domain_spec *out) {
    if (!valid(domain) || !out) return TR_MATH_INVALID;
    *out=domain->spec; return TR_MATH_OK;
}
int32_t tr_math_domain_key_size(const tr_math_domain *domain) {
    return valid(domain) ? (int32_t)domain->key_bytes : TR_MATH_INVALID;
}
int32_t tr_math_domain_key_copy(const tr_math_domain *domain,void *out,uint32_t capacity) {
    if (!valid(domain) || !out) return TR_MATH_INVALID;
    if (capacity<domain->key_bytes) return TR_MATH_BUDGET;
    memcpy(out,domain->key,domain->key_bytes); return TR_MATH_OK;
}
int64_t tr_math_domain_bytes(const tr_math_domain *domain) {
    if (!valid(domain)) return TR_MATH_INVALID;
    return (int64_t)sizeof *domain+domain->key_bytes+
        (domain->origin ? tr_math_domain_bytes(domain->origin) : 0);
}
static int same(const tr_math_domain *left,const tr_math_domain *right,int context) {
    if (!valid(left) || !valid(right)) return TR_MATH_INVALID;
    tr_math_domain_spec a=left->spec,b=right->spec;
    if (context) {
        if (a.kind!=TR_MATH_FORMAL || b.kind!=TR_MATH_FORMAL) return 0;
        a.flags=b.flags=0;
    }
    if (memcmp(&a,&b,sizeof a) || left->key_bytes!=right->key_bytes ||
        memcmp(left->key,right->key,left->key_bytes) || !!left->origin!=!!right->origin) return 0;
    return !left->origin || same(left->origin,right->origin,0)==1;
}
int32_t tr_math_domain_equal(const tr_math_domain *left,const tr_math_domain *right) { return same(left,right,0); }
int32_t tr_math_domain_context_equal(const tr_math_domain *left,const tr_math_domain *right) { return same(left,right,1); }
void tr_math_domain_free(tr_math_domain *domain) {
    if (!domain) return;
    tr_math_domain_free(domain->origin);
    free(domain->key); domain->magic=0; free(domain);
}
