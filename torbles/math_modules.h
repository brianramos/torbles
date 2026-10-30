/* Executable native mathematical operator registry. SPDX-License-Identifier: Apache-2.0 */
#ifndef TORBLES_MATH_MODULES_H
#define TORBLES_MATH_MODULES_H
#include "math_domain.h"
#ifdef __cplusplus
extern "C" {
#endif
enum { TR_MATH_MODULE_POLYNOMIAL=1, TR_MATH_MODULE_MOMENTS=2 };
enum { TR_MATH_OP_ADD=1, TR_MATH_OP_SUB=2, TR_MATH_OP_MUL=3, TR_MATH_OP_POW=4,
       TR_MATH_OP_DERIVATIVE=5, TR_MATH_OP_INTEGRATE=6, TR_MATH_OP_SUBSTITUTE=7,
       TR_MATH_OP_EVALUATE=8, TR_MATH_OP_REDUCE=9 };
enum { TR_MATH_CPU=0, TR_MATH_CUDA=1 };
typedef struct {
    int32_t abi_version, id, carrier, backend, variable_limit, degree_limit, term_limit, data_limit;
    uint64_t operators;
    char name[32];
} tr_math_module_info;
typedef struct {
    int32_t module_id, operation, backend, reason, native_calls, output_terms, left_kind, right_kind;
    double estimated_work; /* relative shape work, NOT nanoseconds */
} tr_math_module_trace;

int32_t tr_math_module_count(void);
int32_t tr_math_module_get(int32_t index, tr_math_module_info *);
/* Actual native polynomial dispatch. Binary ops require right/domain; unary ops
 * ignore them. Only declared formal-rational contracts are executed. An
 * aggregate function may combine with a parameter-only expression from the same
 * context, but never with unaggregated observed variables. Output_domain is set
 * to NULL on failure and owns an independent descriptor on success. */
tr_math_poly *tr_math_module_apply(int32_t module_id, int32_t operation,
                                  const tr_math_domain *left_domain, const tr_math_poly *left,
                                  const tr_math_domain *right_domain, const tr_math_poly *right,
                                  int32_t variable, uint32_t exponent, int64_t numerator, int64_t denominator,
                                  tr_math_domain **output_domain, tr_math_module_trace *, int32_t *error);
int32_t tr_math_module_evaluate(int32_t module_id, const tr_math_domain *, const tr_math_poly *,
                                const tr_math_rational values[TR_MATH_VARIABLES],
                                tr_math_rational *output, tr_math_module_trace *);
tr_math_poly *tr_math_module_reduce(int32_t module_id, const tr_math_domain *source_domain,
                                   const tr_math_bank *, const tr_math_domain *expression_domain,
                                   const tr_math_poly *, tr_math_domain **output_domain,
                                   tr_math_module_trace *, int32_t *error);
#ifdef __cplusplus
}
#endif
#endif
