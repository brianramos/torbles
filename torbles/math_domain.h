/* Immutable typed mathematical domains. SPDX-License-Identifier: Apache-2.0 */
#ifndef TORBLES_MATH_DOMAIN_H
#define TORBLES_MATH_DOMAIN_H
#include "mathematics.h"
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif

#define TR_MATH_DOMAIN_ABI 1
#define TR_MATH_DOMAIN_KEY_MAX 65536
enum { TR_MATH_DOMAIN_ERROR=-6, TR_MATH_CARRIER_ERROR=-7, TR_MATH_OPERATOR_ERROR=-8 };
enum { TR_MATH_PHYSICAL=1, TR_MATH_SNAPSHOT=2, TR_MATH_ASSIGNMENTS=3,
       TR_MATH_FORMAL=4, TR_MATH_GEOMETRY=5 };
enum { TR_MATH_INTEGER=1, TR_MATH_RATIONAL=2, TR_MATH_BOOLEAN=3, TR_MATH_GF2=4 };
enum { TR_MATH_NONNULL=0, TR_MATH_JOINT_VALID=1, TR_MATH_SQL_NULLS=2 };
enum { TR_MATH_SCALAR=0, TR_MATH_DENSE_GROUPS=1, TR_MATH_SPARSE_GROUPS=2 };
enum { TR_MATH_AGGREGATE=1 };

/* Fixed ABI: eight int32 fields, then two uint64 fields; sizeof==48.
 * nvars counts fixed formal slots (normally eight) or source coordinates;
 * the first ndata slots denote observed data, remaining slots free parameters.
 * universe_rows is the physical universe size, NOT selected-row cardinality.
 * flags currently accepts only AGGREGATE on FORMAL domains. */
typedef struct {
    int32_t abi_version, kind, carrier, nvars, ndata, validity, grouping, flags;
    uint64_t revision, universe_rows;
} tr_math_domain_spec;
typedef struct tr_math_domain_s tr_math_domain;

/* key is an exact bounded semantic descriptor, not merely a hash lookup.
 * A FORMAL context uses a stable unique token independent of appended names.
 * Origin is allowed only on FORMAL and must be PHYSICAL/SNAPSHOT with matching
 * ndata. It is deep-copied. Formal observed coordinates require an origin.
 * Descriptors label domains; constructing one does not prove a data identity,
 * causal law, geometric model, or implement operators for every declared type. */
tr_math_domain *tr_math_domain_create(const tr_math_domain_spec *, const void *key,
                                      uint32_t key_bytes, const tr_math_domain *origin, int32_t *error);
tr_math_domain *tr_math_domain_clone(const tr_math_domain *, int32_t *error);
tr_math_domain *tr_math_domain_with_aggregate(const tr_math_domain *, int32_t aggregate, int32_t *error);
tr_math_domain *tr_math_domain_origin(const tr_math_domain *, int32_t *error); /* owned clone, or NULL/OK */
int32_t tr_math_domain_get_spec(const tr_math_domain *, tr_math_domain_spec *);
int32_t tr_math_domain_key_size(const tr_math_domain *);
int32_t tr_math_domain_key_copy(const tr_math_domain *, void *output, uint32_t capacity);
int64_t tr_math_domain_bytes(const tr_math_domain *);
int32_t tr_math_domain_equal(const tr_math_domain *, const tr_math_domain *); /* exact 0/1, or error */
/* Same formal context/origin/type/layout, permitting differing aggregate flag. */
int32_t tr_math_domain_context_equal(const tr_math_domain *, const tr_math_domain *);
void tr_math_domain_free(tr_math_domain *);

#ifdef __cplusplus
}
#endif
#endif
