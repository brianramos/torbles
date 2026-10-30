/* Standalone typed-domain/registry ownership and refusal checks.
 * gcc -std=c11 -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer
 *   -Itorbles tests/math_domain_native_edges.c torbles/mathematics.c
 *   torbles/math_domain.c torbles/math_modules.c -o /tmp/math-domain-edges
 * ASAN_OPTIONS=detect_leaks=1 /tmp/math-domain-edges
 * SPDX-License-Identifier: Apache-2.0 */
#include "math_modules.h"
#include <assert.h>
#include <limits.h>
#include <stdio.h>
#include <string.h>

static tr_math_domain *domain(int kind,int carrier,int ndata,const char *key,
                              const tr_math_domain *origin) {
    tr_math_domain_spec spec={1,kind,carrier,kind==TR_MATH_FORMAL?8:ndata,ndata,0,0,0,0,0};
    int32_t error=99;
    tr_math_domain *out=tr_math_domain_create(&spec,key,(uint32_t)strlen(key),origin,&error);
    assert(out && !error); return out;
}
static tr_math_poly *apply(int op,const tr_math_domain *ld,const tr_math_poly *left,
                           const tr_math_domain *rd,const tr_math_poly *right,int var,
                           tr_math_domain **out) {
    int32_t error=99; tr_math_module_trace trace;
    tr_math_poly *value=tr_math_module_apply(1,op,ld,left,rd,right,var,0,0,1,out,&trace,&error);
    assert(value && *out && !error && !trace.reason && trace.native_calls==1 && trace.backend==0);
    return value;
}
static void refuse(int expected,int op,const tr_math_domain *ld,const tr_math_poly *left,
                    const tr_math_domain *rd,const tr_math_poly *right,int var) {
    int32_t error=99; tr_math_module_trace trace; tr_math_domain *out=NULL;
    tr_math_poly *value=tr_math_module_apply(1,op,ld,left,rd,right,var,0,0,1,&out,&trace,&error);
    assert(!value && !out && error==expected && trace.reason==expected && trace.native_calls==0);
}
int main(void) {
    int32_t error=0;
    char key[]="source-A";
    tr_math_domain_spec spec={1,TR_MATH_SNAPSHOT,TR_MATH_INTEGER,2,2,0,0,0,7,9};
    tr_math_domain *rows=tr_math_domain_create(&spec,key,sizeof key,NULL,&error);
    assert(rows && !error);
    tr_math_domain *copy=tr_math_domain_clone(rows,&error);
    assert(copy && !error && tr_math_domain_equal(rows,copy)==1);
    key[0]='X';
    char saved[sizeof key]; memset(saved,0,sizeof saved);
    assert(!tr_math_domain_key_copy(rows,saved,sizeof saved) && !strcmp(saved,"source-A"));
    char short_key[1]={'!'};
    assert(tr_math_domain_key_copy(rows,short_key,1)==TR_MATH_BUDGET && short_key[0]=='!');
    assert(tr_math_domain_bytes(rows)>(int64_t)sizeof spec);
    for (int field=0;field<5;++field) {
        tr_math_domain_spec changed=spec;
        if (field==0) changed.revision++;
        if (field==1) changed.validity=TR_MATH_SQL_NULLS;
        if (field==2) changed.grouping=TR_MATH_DENSE_GROUPS;
        if (field==3) changed.kind=TR_MATH_PHYSICAL;
        if (field==4) changed.universe_rows++;
        tr_math_domain *other=tr_math_domain_create(&changed,"source-A",sizeof key,NULL,&error);
        assert(other && !error && tr_math_domain_equal(rows,other)==0);
        tr_math_domain_free(other);
    }
    assert(!tr_math_domain_create(&spec,"a",0,NULL,&error) && error==TR_MATH_INVALID);
    assert(!tr_math_domain_create(&spec,"a",TR_MATH_DOMAIN_KEY_MAX+1,NULL,&error) && error==TR_MATH_BUDGET);
    tr_math_domain *formal=domain(TR_MATH_FORMAL,TR_MATH_RATIONAL,2,"context-A",rows);
    tr_math_domain *formal_other=domain(TR_MATH_FORMAL,TR_MATH_RATIONAL,2,"context-B",rows);
    tr_math_domain *other_rows=domain(TR_MATH_SNAPSHOT,TR_MATH_INTEGER,2,"source-B",NULL);
    tr_math_domain *wrong_origin=domain(TR_MATH_FORMAL,TR_MATH_RATIONAL,2,"context-A",other_rows);
    tr_math_domain *boolean=domain(TR_MATH_FORMAL,TR_MATH_GF2,2,"context-A",rows);
    tr_math_domain *geometry=domain(TR_MATH_GEOMETRY,TR_MATH_RATIONAL,0,"grid-A",NULL);
    tr_math_domain *detached=tr_math_domain_origin(geometry,&error);
    assert(!detached && !error);
    tr_math_domain_spec bad={1,TR_MATH_FORMAL,TR_MATH_RATIONAL,8,2,0,0,0,0,0};
    assert(!tr_math_domain_create(&bad,"x",1,NULL,&error) && error==TR_MATH_DOMAIN_ERROR);
    bad.ndata=0; bad.flags=TR_MATH_AGGREGATE;
    assert(!tr_math_domain_create(&bad,"x",1,NULL,&error) && error==TR_MATH_DOMAIN_ERROR);

    tr_math_poly *x=tr_math_poly_variable(0,&error),*y=tr_math_poly_variable(1,&error);
    tr_math_poly *theta=tr_math_poly_variable(2,&error);
    tr_math_domain *dtx=NULL,*ddifference=NULL,*dsquare=NULL;
    tr_math_poly *tx=apply(TR_MATH_OP_MUL,formal,theta,formal,x,0,&dtx);
    tr_math_poly *difference=apply(TR_MATH_OP_SUB,formal,y,dtx,tx,0,&ddifference);
    tr_math_poly *square=apply(TR_MATH_OP_MUL,ddifference,difference,ddifference,difference,0,&dsquare);
    refuse(TR_MATH_DOMAIN_ERROR,TR_MATH_OP_ADD,formal,x,formal_other,y,0);
    refuse(TR_MATH_DOMAIN_ERROR,TR_MATH_OP_ADD,formal,x,wrong_origin,y,0);
    refuse(TR_MATH_CARRIER_ERROR,TR_MATH_OP_ADD,boolean,x,boolean,y,0);
    refuse(TR_MATH_DOMAIN_ERROR,TR_MATH_OP_ADD,geometry,x,geometry,y,0);
    refuse(TR_MATH_DOMAIN_ERROR,TR_MATH_OP_DERIVATIVE,formal,x,NULL,NULL,8);

    const uint8_t exponents[]={0,0,1,0,0,1,2,0,1,1,0,2};
    const int64_t values[]={2,3,4,5,6,8}; /* rows (1,2), (2,2) */
    tr_math_bank *bank=tr_math_bank_create(2,6,exponents,values,&error);
    assert(bank && !error);
    tr_math_domain *function_domain=NULL; tr_math_module_trace trace;
    tr_math_poly *function=tr_math_module_reduce(2,rows,bank,dsquare,square,&function_domain,&trace,&error);
    assert(function && function_domain && !error && trace.native_calls==1 && trace.backend==0);
    tr_math_domain_spec function_spec;
    assert(!tr_math_domain_get_spec(function_domain,&function_spec) && function_spec.flags==TR_MATH_AGGREGATE);
    assert(tr_math_domain_equal(function_domain,formal)==0 && tr_math_domain_context_equal(function_domain,formal)==1);
    tr_math_rational inputs[8],answer={111,1};
    for (int i=0;i<8;++i) inputs[i]=(tr_math_rational){0,1};
    inputs[2]=(tr_math_rational){3,1};
    assert(!tr_math_module_evaluate(1,function_domain,function,inputs,&answer,&trace));
    assert(answer.numerator==17 && answer.denominator==1);
    refuse(TR_MATH_DOMAIN_ERROR,TR_MATH_OP_ADD,function_domain,function,formal,x,0);
    refuse(TR_MATH_DOMAIN_ERROR,TR_MATH_OP_DERIVATIVE,function_domain,function,NULL,NULL,0);
    refuse(TR_MATH_DOMAIN_ERROR,TR_MATH_OP_SUBSTITUTE,function_domain,function,NULL,NULL,0);
    tr_math_domain *sum_domain=NULL;
    tr_math_poly *sum=apply(TR_MATH_OP_ADD,function_domain,function,formal,theta,0,&sum_domain);
    assert(!tr_math_module_evaluate(1,sum_domain,sum,inputs,&answer,&trace) && answer.numerator==20);
    tr_math_domain *integral_domain=NULL;
    tr_math_poly *integral=apply(TR_MATH_OP_INTEGRATE,function_domain,function,NULL,NULL,2,&integral_domain);
    inputs[2]=(tr_math_rational){1,1};
    assert(!tr_math_module_evaluate(1,integral_domain,integral,inputs,&answer,&trace));
    assert(answer.numerator==11 && answer.denominator==3);
    tr_math_domain *failed=NULL;
    tr_math_poly *invalid=tr_math_module_reduce(2,other_rows,bank,formal,x,&failed,&trace,&error);
    assert(!invalid && !failed && error==TR_MATH_DOMAIN_ERROR && trace.native_calls==0);
    invalid=tr_math_module_reduce(2,rows,bank,function_domain,function,&failed,&trace,&error);
    assert(!invalid && !failed && error==TR_MATH_DOMAIN_ERROR);
    answer=(tr_math_rational){111,1}; inputs[2].denominator=0;
    assert(tr_math_module_evaluate(1,function_domain,function,inputs,&answer,&trace)==TR_MATH_INVALID);
    assert(answer.numerator==111 && answer.denominator==1);

    for (int i=0;i<tr_math_module_count();++i) {
        tr_math_module_info info;
        assert(!tr_math_module_get(i,&info) && info.backend==TR_MATH_CPU && info.operators && info.name[0]);
    }
    tr_math_module_info info;
    assert(tr_math_module_get(2,&info)==TR_MATH_INVALID);
    tr_math_poly *one=tr_math_poly_constant(1,1,&error);
    tr_math_domain *power_domain=NULL;
    tr_math_poly *power=tr_math_module_apply(1,TR_MATH_OP_POW,formal,one,NULL,NULL,0,UINT32_MAX,0,1,&power_domain,&trace,&error);
    assert(power && !error && trace.estimated_work>0);
    tr_math_poly_free(power); tr_math_domain_free(power_domain); tr_math_poly_free(one);

    tr_math_domain_free(rows); tr_math_domain_free(copy);
    tr_math_domain_free(formal); tr_math_domain_free(formal_other);
    tr_math_domain_free(other_rows); tr_math_domain_free(wrong_origin);
    tr_math_domain_free(boolean); tr_math_domain_free(geometry);
    tr_math_domain_free(dtx); tr_math_domain_free(ddifference); tr_math_domain_free(dsquare);
    tr_math_bank_free(bank);
    /* Function owns its origin: original source descriptors/bank are gone. */
    inputs[2]=(tr_math_rational){3,1};
    assert(!tr_math_module_evaluate(1,function_domain,function,inputs,&answer,&trace) && answer.numerator==17);
    tr_math_poly_free(x); tr_math_poly_free(y); tr_math_poly_free(theta);
    tr_math_poly_free(tx); tr_math_poly_free(difference); tr_math_poly_free(square);
    tr_math_poly_free(function); tr_math_poly_free(sum); tr_math_poly_free(integral);
    tr_math_domain_free(function_domain); tr_math_domain_free(sum_domain); tr_math_domain_free(integral_domain);
    puts("Typed domains and executable registry edge checks passed.");
    return 0;
}
