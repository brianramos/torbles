/* Bounded exact mathematical query objects. SPDX-License-Identifier: Apache-2.0 */
#ifndef TORBLES_MATHEMATICS_H
#define TORBLES_MATHEMATICS_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

#define TR_MATH_VARIABLES 8
#define TR_MATH_DEGREE 8
#define TR_MATH_TERMS 128
#define TR_MATH_DATA_VARIABLES 4
enum { TR_MATH_OK=0, TR_MATH_INVALID=-1, TR_MATH_MEMORY=-2,
       TR_MATH_OVERFLOW=-3, TR_MATH_BUDGET=-5 };

typedef struct { int64_t numerator, denominator; } tr_math_rational;
/* ABI: exponents at offset 0, coefficient at offset 8; sizeof(term)==24. */
typedef struct { uint8_t exponents[TR_MATH_VARIABLES]; tr_math_rational coefficient; } tr_math_term;
typedef struct tr_math_poly_s tr_math_poly;
typedef struct tr_math_bank_s tr_math_bank;

/* Objects are immutable and individually owned. error may be NULL. A failed
 * constructor/operator returns NULL and sets error without changing inputs.
 * Coefficients/results are canonical exact int64 rationals, positive denominator.
 * Overflow is reported; it never wraps or silently approximates. Intermediate
 * coefficient limits may refuse an expression before later cancellation. */
tr_math_poly *tr_math_poly_constant(int64_t numerator, int64_t denominator, int32_t *error);
tr_math_poly *tr_math_poly_variable(int32_t variable, int32_t *error);
tr_math_poly *tr_math_poly_add(const tr_math_poly *, const tr_math_poly *, int32_t *error);
tr_math_poly *tr_math_poly_sub(const tr_math_poly *, const tr_math_poly *, int32_t *error);
tr_math_poly *tr_math_poly_mul(const tr_math_poly *, const tr_math_poly *, int32_t *error);
tr_math_poly *tr_math_poly_pow(const tr_math_poly *, uint32_t exponent, int32_t *error);
tr_math_poly *tr_math_poly_derivative(const tr_math_poly *, int32_t variable, int32_t *error);
/* The antiderivative has zero integration constant. */
tr_math_poly *tr_math_poly_integrate(const tr_math_poly *, int32_t variable, int32_t *error);
tr_math_poly *tr_math_poly_substitute(const tr_math_poly *, int32_t variable,
                                    int64_t numerator, int64_t denominator, int32_t *error);
/* values contains eight valid rationals, including (0,1) for unused variables.
 * Output is written only on success. */
int32_t tr_math_poly_evaluate(const tr_math_poly *, const tr_math_rational *values,
                             tr_math_rational *output);
int32_t tr_math_poly_equal(const tr_math_poly *, const tr_math_poly *); /* 0/1 or INVALID */
int32_t tr_math_poly_term_count(const tr_math_poly *);
int32_t tr_math_poly_get_term(const tr_math_poly *, int32_t index, tr_math_term *output);
int32_t tr_math_poly_degree(const tr_math_poly *);
int32_t tr_math_poly_support(const tr_math_poly *); /* variable bitmask or INVALID */
void tr_math_poly_free(tr_math_poly *);

/* A bank is an immutable scalar, fixed-scope degree-two moment snapshot.
 * ndata is 0..4. The required complete basis has 1+ndata+ndata*(ndata+1)/2
 * entries (constant, each linear term, all squares and cross-products).
 * exponents is row-major nterms*ndata bytes, in any order; NULL is allowed
 * only when ndata==0. moments are exact int64 sums. Constant count must be
 * nonnegative; an empty bank must have every moment zero. No table is retained.
 * Building these values currently uses existing native aggregate queries;
 * this constructor is not a fused single-pass data executor. */
tr_math_bank *tr_math_bank_create(int32_t ndata, int32_t nterms,
                                 const uint8_t *exponents, const int64_t *moments, int32_t *error);
int32_t tr_math_bank_variables(const tr_math_bank *);
int32_t tr_math_bank_term_count(const tr_math_bank *);
int32_t tr_math_bank_count(const tr_math_bank *, int64_t *output);
int32_t tr_math_bank_moment(const tr_math_bank *, const uint8_t *exponents, int64_t *output);
/* Replaces each first-ndata-variable monomial by its stored moment and keeps
 * all remaining variable indices. Data degree >2 is refused with BUDGET. */
tr_math_poly *tr_math_bank_reduce(const tr_math_bank *, const tr_math_poly *, int32_t *error);
void tr_math_bank_free(tr_math_bank *);

struct atable_s;
uint64_t tr_math_table_revision(const struct atable_s *);
#ifdef __cplusplus
}
#endif
#endif
