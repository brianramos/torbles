/* Actual checked native operator dispatch. SPDX-License-Identifier: Apache-2.0 */
#include "math_modules.h"
#include <stddef.h>
#include <string.h>

#define OPBIT(op) (UINT64_C(1)<<(op))
_Static_assert(sizeof(tr_math_module_info)==72,"module info ABI size");
_Static_assert(sizeof(tr_math_module_trace)==40,"module trace ABI size");
static const tr_math_module_info modules[]={
    {TR_MATH_DOMAIN_ABI,TR_MATH_MODULE_POLYNOMIAL,TR_MATH_RATIONAL,TR_MATH_CPU,
     TR_MATH_VARIABLES,TR_MATH_DEGREE,TR_MATH_TERMS,TR_MATH_DATA_VARIABLES,
     OPBIT(TR_MATH_OP_ADD)|OPBIT(TR_MATH_OP_SUB)|OPBIT(TR_MATH_OP_MUL)|OPBIT(TR_MATH_OP_POW)|
     OPBIT(TR_MATH_OP_DERIVATIVE)|OPBIT(TR_MATH_OP_INTEGRATE)|OPBIT(TR_MATH_OP_SUBSTITUTE)|
     OPBIT(TR_MATH_OP_EVALUATE),"checked rational polynomials"},
    {TR_MATH_DOMAIN_ABI,TR_MATH_MODULE_MOMENTS,TR_MATH_INTEGER,TR_MATH_CPU,
     TR_MATH_VARIABLES,TR_MATH_DEGREE,TR_MATH_TERMS,TR_MATH_DATA_VARIABLES,
     OPBIT(TR_MATH_OP_REDUCE),"scalar integer moments"}
};
int32_t tr_math_module_count(void) { return (int32_t)(sizeof modules/sizeof modules[0]); }
int32_t tr_math_module_get(int32_t index,tr_math_module_info *out) {
    if (!out || index<0 || index>=tr_math_module_count()) return TR_MATH_INVALID;
    *out=modules[index]; return TR_MATH_OK;
}
static void trace_init(tr_math_module_trace *trace,int module,int operation,
                       const tr_math_domain *left,const tr_math_domain *right) {
    if (!trace) return;
    memset(trace,0,sizeof *trace);
    trace->module_id=module; trace->operation=operation; trace->backend=TR_MATH_CPU;
    tr_math_domain_spec spec;
    if (!tr_math_domain_get_spec(left,&spec)) trace->left_kind=spec.kind;
    if (!tr_math_domain_get_spec(right,&spec)) trace->right_kind=spec.kind;
}
static void error_set(int32_t *error,tr_math_module_trace *trace,int code) {
    if (error) *error=code;
    if (trace) trace->reason=code;
}
static int formal(const tr_math_domain *domain,const tr_math_poly *poly,tr_math_domain_spec *spec) {
    if (tr_math_domain_get_spec(domain,spec)) return TR_MATH_INVALID;
    if (spec->kind!=TR_MATH_FORMAL || spec->validity!=TR_MATH_NONNULL || spec->grouping!=TR_MATH_SCALAR)
        return TR_MATH_DOMAIN_ERROR;
    if (spec->carrier!=TR_MATH_RATIONAL) return TR_MATH_CARRIER_ERROR;
    int support=tr_math_poly_support(poly);
    if (support<0) return support;
    if (support>>spec->nvars) return TR_MATH_DOMAIN_ERROR;
    if ((spec->flags & TR_MATH_AGGREGATE) && (support & ((1<<spec->ndata)-1))) return TR_MATH_DOMAIN_ERROR;
    return TR_MATH_OK;
}
static tr_math_poly *finish(tr_math_poly *poly,const tr_math_domain *domain,int aggregate,
                            tr_math_domain **output,tr_math_module_trace *trace,int32_t *error) {
    if (!poly) return NULL;
    *output=tr_math_domain_with_aggregate(domain,aggregate,error);
    if (!*output) {
        tr_math_poly_free(poly);
        if (trace) trace->reason=error ? *error : TR_MATH_MEMORY;
        return NULL;
    }
    if (trace) trace->output_terms=tr_math_poly_term_count(poly);
    return poly;
}
tr_math_poly *tr_math_module_apply(int32_t module,int32_t operation,
                                  const tr_math_domain *left_domain,const tr_math_poly *left,
                                  const tr_math_domain *right_domain,const tr_math_poly *right,
                                  int32_t variable,uint32_t exponent,int64_t numerator,int64_t denominator,
                                  tr_math_domain **output,tr_math_module_trace *trace,int32_t *error) {
    trace_init(trace,module,operation,left_domain,right_domain);
    error_set(error,trace,TR_MATH_OK);
    if (output) *output=NULL;
    if (!output) { error_set(error,trace,TR_MATH_INVALID); return NULL; }
    if (module!=TR_MATH_MODULE_POLYNOMIAL || operation<TR_MATH_OP_ADD || operation>TR_MATH_OP_SUBSTITUTE) {
        error_set(error,trace,TR_MATH_OPERATOR_ERROR); return NULL;
    }
    tr_math_domain_spec a,b;
    int rc=formal(left_domain,left,&a);
    if (rc) { error_set(error,trace,rc); return NULL; }
    int aggregate=!!(a.flags & TR_MATH_AGGREGATE);
    int left_terms=tr_math_poly_term_count(left),right_terms=0;
    if (operation<=TR_MATH_OP_MUL) {
        rc=formal(right_domain,right,&b);
        if (rc) { error_set(error,trace,rc); return NULL; }
        if (tr_math_domain_context_equal(left_domain,right_domain)!=1) {
            error_set(error,trace,TR_MATH_DOMAIN_ERROR); return NULL;
        }
        aggregate|=!!(b.flags & TR_MATH_AGGREGATE);
        int data_mask=(1<<a.ndata)-1;
        if (aggregate && ((tr_math_poly_support(left)|tr_math_poly_support(right)) & data_mask)) {
            error_set(error,trace,TR_MATH_DOMAIN_ERROR); return NULL;
        }
        right_terms=tr_math_poly_term_count(right);
    } else if (operation!=TR_MATH_OP_POW &&
               (variable<0 || variable>=a.nvars || (aggregate && variable<a.ndata))) {
        error_set(error,trace,TR_MATH_DOMAIN_ERROR); return NULL;
    }
    if (trace) {
        trace->native_calls=1;
        trace->estimated_work=operation==TR_MATH_OP_MUL ? (double)left_terms*right_terms :
            operation==TR_MATH_OP_POW ? (double)left_terms*(1.0+exponent) : left_terms+right_terms;
    }
    int32_t code=0;
    tr_math_poly *poly=NULL;
    switch (operation) {
    case TR_MATH_OP_ADD: poly=tr_math_poly_add(left,right,&code); break;
    case TR_MATH_OP_SUB: poly=tr_math_poly_sub(left,right,&code); break;
    case TR_MATH_OP_MUL: poly=tr_math_poly_mul(left,right,&code); break;
    case TR_MATH_OP_POW: poly=tr_math_poly_pow(left,exponent,&code); break;
    case TR_MATH_OP_DERIVATIVE: poly=tr_math_poly_derivative(left,variable,&code); break;
    case TR_MATH_OP_INTEGRATE: poly=tr_math_poly_integrate(left,variable,&code); break;
    case TR_MATH_OP_SUBSTITUTE: poly=tr_math_poly_substitute(left,variable,numerator,denominator,&code); break;
    }
    if (!poly) { error_set(error,trace,code ? code : TR_MATH_MEMORY); return NULL; }
    return finish(poly,left_domain,aggregate,output,trace,error);
}
int32_t tr_math_module_evaluate(int32_t module,const tr_math_domain *domain,const tr_math_poly *poly,
                                const tr_math_rational values[TR_MATH_VARIABLES],
                                tr_math_rational *output,tr_math_module_trace *trace) {
    trace_init(trace,module,TR_MATH_OP_EVALUATE,domain,NULL);
    int rc=TR_MATH_OPERATOR_ERROR;
    tr_math_domain_spec spec;
    if (module==TR_MATH_MODULE_POLYNOMIAL) rc=formal(domain,poly,&spec);
    if (!rc) {
        if (trace) { trace->native_calls=1; trace->estimated_work=tr_math_poly_term_count(poly)*(1.0+tr_math_poly_degree(poly)); }
        rc=tr_math_poly_evaluate(poly,values,output);
    }
    if (trace) trace->reason=rc;
    return rc;
}
tr_math_poly *tr_math_module_reduce(int32_t module,const tr_math_domain *source,
                                   const tr_math_bank *bank,const tr_math_domain *expression_domain,
                                   const tr_math_poly *expression,tr_math_domain **output,
                                   tr_math_module_trace *trace,int32_t *error) {
    trace_init(trace,module,TR_MATH_OP_REDUCE,source,expression_domain);
    error_set(error,trace,TR_MATH_OK);
    if (output) *output=NULL;
    if (!output) { error_set(error,trace,TR_MATH_INVALID); return NULL; }
    if (module!=TR_MATH_MODULE_MOMENTS) { error_set(error,trace,TR_MATH_OPERATOR_ERROR); return NULL; }
    tr_math_domain_spec row,formal_spec;
    int rc=tr_math_domain_get_spec(source,&row);
    if (rc) { error_set(error,trace,rc); return NULL; }
    if ((row.kind!=TR_MATH_PHYSICAL && row.kind!=TR_MATH_SNAPSHOT) || row.validity!=TR_MATH_NONNULL ||
        row.grouping!=TR_MATH_SCALAR || row.flags) {
        error_set(error,trace,TR_MATH_DOMAIN_ERROR); return NULL;
    }
    if (row.carrier!=TR_MATH_INTEGER) { error_set(error,trace,TR_MATH_CARRIER_ERROR); return NULL; }
    rc=formal(expression_domain,expression,&formal_spec);
    if (rc) { error_set(error,trace,rc); return NULL; }
    if (tr_math_bank_variables(bank)!=row.ndata || formal_spec.ndata!=row.ndata ||
        (formal_spec.flags & TR_MATH_AGGREGATE)) {
        error_set(error,trace,TR_MATH_DOMAIN_ERROR); return NULL;
    }
    int32_t code=0;
    tr_math_domain *origin=tr_math_domain_origin(expression_domain,&code);
    int same=origin && tr_math_domain_equal(source,origin)==1;
    tr_math_domain_free(origin);
    if (!same) { error_set(error,trace,code ? code : TR_MATH_DOMAIN_ERROR); return NULL; }
    if (trace) { trace->native_calls=1; trace->estimated_work=tr_math_poly_term_count(expression)*tr_math_bank_term_count(bank); }
    tr_math_poly *poly=tr_math_bank_reduce(bank,expression,&code);
    if (!poly) { error_set(error,trace,code ? code : TR_MATH_MEMORY); return NULL; }
    return finish(poly,expression_domain,1,output,trace,error);
}
