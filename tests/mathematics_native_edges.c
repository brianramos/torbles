/* Independent native lifetime/undefined-behavior edge sweep.
 * SPDX-License-Identifier: Apache-2.0
 * Reproduce, including the Fraction oracle and receipt:
 *   python tests/validate_mathematics_native.py
 * The runner builds this executable with ASAN+UBSAN and leak detection in an
 * isolated temporary directory. It never rebuilds the engine's shared library.
 */
#include "mathematics.h"
#include <assert.h>
#include <limits.h>
#include <stdio.h>

static void check_result(tr_math_poly *p, int32_t rc) {
    if (!p) { assert(rc==TR_MATH_OVERFLOW); return; }
    assert(rc==TR_MATH_OK);
    int n=tr_math_poly_term_count(p);
    assert(n==0 || n==1);
    if (n) {
        tr_math_term term;
        assert(!tr_math_poly_get_term(p,0,&term));
        assert(term.coefficient.denominator>0 && term.coefficient.numerator);
        for (int v=0;v<TR_MATH_VARIABLES;v++) assert(!term.exponents[v]);
    }
    tr_math_poly_free(p);
}

int main(void) {
    const int64_t ns[]={INT64_MIN,INT64_MIN+1,-INT64_MAX,-2,-1,0,1,2,INT64_MAX-1,INT64_MAX};
    const int64_t ds[]={INT64_MIN,-INT64_MAX,-2,-1,1,2,INT64_MAX};
    int tested=0;
    for (unsigned i=0;i<sizeof ns/sizeof *ns;i++) for (unsigned j=0;j<sizeof ds/sizeof *ds;j++) {
        int32_t rc=0;
        tr_math_poly *a=tr_math_poly_constant(ns[i],ds[j],&rc);
        if (!a) { assert(rc==TR_MATH_OVERFLOW); continue; }
        for (unsigned k=0;k<sizeof ns/sizeof *ns;k++) for (unsigned l=0;l<sizeof ds/sizeof *ds;l++) {
            tr_math_poly *b=tr_math_poly_constant(ns[k],ds[l],&rc);
            if (!b) { assert(rc==TR_MATH_OVERFLOW); continue; }
            tr_math_poly *result=tr_math_poly_add(a,b,&rc); check_result(result,rc);
            result=tr_math_poly_sub(a,b,&rc); check_result(result,rc);
            result=tr_math_poly_mul(a,b,&rc); check_result(result,rc);
            tr_math_poly *zero=tr_math_poly_sub(b,b,&rc);
            assert(zero && !rc && !tr_math_poly_term_count(zero));
            tr_math_poly_free(zero); tr_math_poly_free(b); tested++;
        }
        tr_math_poly_free(a);
    }
    assert(tested==3969);
    for (int v=-1;v<=TR_MATH_VARIABLES;v++) {
        int32_t rc;
        tr_math_poly *x=tr_math_poly_variable(v,&rc);
        if (v<0 || v==TR_MATH_VARIABLES) { assert(!x && rc==TR_MATH_INVALID); continue; }
        tr_math_poly *p=tr_math_poly_pow(x,TR_MATH_DEGREE,&rc);
        assert(p && !rc);
        tr_math_poly *bad=tr_math_poly_integrate(p,v,&rc);
        assert(!bad && rc==TR_MATH_BUDGET);
        tr_math_rational values[TR_MATH_VARIABLES],out={87,19};
        for (int k=0;k<TR_MATH_VARIABLES;k++) values[k]=(tr_math_rational){2,1};
        assert(!tr_math_poly_evaluate(p,values,&out) && out.numerator==256 && out.denominator==1);
        values[v]=(tr_math_rational){INT64_MAX,1}; out=(tr_math_rational){87,19};
        assert(tr_math_poly_evaluate(p,values,&out)==TR_MATH_OVERFLOW);
        assert(out.numerator==87 && out.denominator==19);
        tr_math_poly_free(p); tr_math_poly_free(x);
    }
    puts("ASAN/UBSAN native edge sweep passed.");
    printf("canonical-rational pairs checked: %d\n",tested);
    return 0;
}
