/* Bounded exact mathematical query objects. SPDX-License-Identifier: Apache-2.0 */
#include "mathematics.h"
#include "torble.h"
#include <limits.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#define POLY_MAGIC UINT32_C(0x544d504c)
#define BANK_MAGIC UINT32_C(0x544d424b)
#define BANK_TERMS (1+TR_MATH_DATA_VARIABLES+TR_MATH_DATA_VARIABLES*(TR_MATH_DATA_VARIABLES+1)/2)
_Static_assert(sizeof(tr_math_rational)==16,"rational ABI size");
_Static_assert(sizeof(tr_math_term)==24 && offsetof(tr_math_term,coefficient)==8,"term ABI layout");

struct tr_math_poly_s {
    uint32_t magic;
    int32_t n;
    tr_math_term terms[TR_MATH_TERMS];
};
typedef struct { uint8_t exponents[TR_MATH_DATA_VARIABLES]; int64_t value; } moment;
struct tr_math_bank_s {
    uint32_t magic;
    int32_t ndata, n;
    int64_t count;
    moment terms[BANK_TERMS];
};

static void error_set(int32_t *error, int32_t code) { if (error) *error=code; }
static int poly_valid(const tr_math_poly *p) { return p && p->magic==POLY_MAGIC; }
static int bank_valid(const tr_math_bank *p) { return p && p->magic==BANK_MAGIC; }
static int variable_valid(int32_t v) { return v>=0 && v<TR_MATH_VARIABLES; }
static unsigned __int128 absolute128(__int128 n) {
    return n<0 ? -(unsigned __int128)n : (unsigned __int128)n;
}
static unsigned __int128 gcd128(unsigned __int128 a, unsigned __int128 b) {
    while (b) { unsigned __int128 r=a%b; a=b; b=r; }
    return a;
}
/* Every caller supplies an int64 or products/sums of two canonical int64
 * rationals. Those intermediates fit signed __int128 before normalization. */
static int rational(__int128 n, __int128 d, tr_math_rational *out) {
    if (!d) return TR_MATH_INVALID;
    if (!n) { *out=(tr_math_rational){0,1}; return TR_MATH_OK; }
    if (d<0) { n=-n; d=-d; }
    unsigned __int128 gcd=gcd128(absolute128(n),(unsigned __int128)d);
    n/=(__int128)gcd; d/=(__int128)gcd;
    if (n<INT64_MIN || n>INT64_MAX || d>INT64_MAX) return TR_MATH_OVERFLOW;
    *out=(tr_math_rational){(int64_t)n,(int64_t)d};
    return TR_MATH_OK;
}
static int rational_add(tr_math_rational a, tr_math_rational b, int sign,
                        tr_math_rational *out) {
    return rational((__int128)a.numerator*b.denominator+
                    (__int128)sign*b.numerator*a.denominator,
                    (__int128)a.denominator*b.denominator,out);
}
static int rational_mul(tr_math_rational a, tr_math_rational b, tr_math_rational *out) {
    return rational((__int128)a.numerator*b.numerator,
                    (__int128)a.denominator*b.denominator,out);
}
static int rational_pow(tr_math_rational value, unsigned power, tr_math_rational *out) {
    tr_math_rational result={1,1};
    while (power) {
        int rc;
        if ((power&1) && (rc=rational_mul(result,value,&result))) return rc;
        power>>=1;
        if (power && (rc=rational_mul(value,value,&value))) return rc;
    }
    *out=result; return TR_MATH_OK;
}

static tr_math_poly *poly_new(int32_t *error) {
    tr_math_poly *p=calloc(1,sizeof *p);
    if (!p) { error_set(error,TR_MATH_MEMORY); return NULL; }
    p->magic=POLY_MAGIC; error_set(error,TR_MATH_OK); return p;
}
static tr_math_poly *fail(tr_math_poly *p, int code, int32_t *error) {
    free(p); error_set(error,code); return NULL;
}
static int term_order(const void *a, const void *b) {
    return memcmp(((const tr_math_term *)a)->exponents,
                  ((const tr_math_term *)b)->exponents,TR_MATH_VARIABLES);
}
static void sort(tr_math_poly *p) {
    qsort(p->terms,(size_t)p->n,sizeof p->terms[0],term_order);
}
static int put(tr_math_poly *p, const uint8_t *exponents, tr_math_rational c, int sign) {
    if (!c.numerator) return TR_MATH_OK;
    int degree=0;
    for (int i=0;i<TR_MATH_VARIABLES;++i) degree+=exponents[i];
    if (degree>TR_MATH_DEGREE) return TR_MATH_BUDGET;
    for (int i=0;i<p->n;++i) if (!memcmp(p->terms[i].exponents,exponents,TR_MATH_VARIABLES)) {
        tr_math_rational value;
        int rc=rational_add(p->terms[i].coefficient,c,sign,&value);
        if (rc) return rc;
        if (!value.numerator) p->terms[i]=p->terms[--p->n];
        else p->terms[i].coefficient=value;
        return TR_MATH_OK;
    }
    if (p->n==TR_MATH_TERMS) return TR_MATH_BUDGET;
    if (sign<0) {
        int rc=rational(-(__int128)c.numerator,c.denominator,&c);
        if (rc) return rc;
    }
    memcpy(p->terms[p->n].exponents,exponents,TR_MATH_VARIABLES);
    p->terms[p->n++].coefficient=c;
    return TR_MATH_OK;
}

tr_math_poly *tr_math_poly_constant(int64_t n, int64_t d, int32_t *error) {
    tr_math_rational c;
    int rc=rational(n,d,&c);
    if (rc) return fail(NULL,rc,error);
    tr_math_poly *p=poly_new(error);
    if (p && c.numerator) { p->n=1; p->terms[0].coefficient=c; }
    return p;
}
tr_math_poly *tr_math_poly_variable(int32_t variable, int32_t *error) {
    if (!variable_valid(variable)) return fail(NULL,TR_MATH_INVALID,error);
    tr_math_poly *p=poly_new(error);
    if (p) {
        p->n=1; p->terms[0].exponents[variable]=1;
        p->terms[0].coefficient=(tr_math_rational){1,1};
    }
    return p;
}
static tr_math_poly *add(const tr_math_poly *a, const tr_math_poly *b, int sign, int32_t *error) {
    if (!poly_valid(a) || !poly_valid(b)) return fail(NULL,TR_MATH_INVALID,error);
    tr_math_poly *out=poly_new(error);
    if (!out) return NULL;
    *out=*a;
    for (int i=0;i<b->n;++i) {
        int rc=put(out,b->terms[i].exponents,b->terms[i].coefficient,sign);
        if (rc) return fail(out,rc,error);
    }
    sort(out); return out;
}
tr_math_poly *tr_math_poly_add(const tr_math_poly *a,const tr_math_poly *b,int32_t *error) {
    return add(a,b,1,error);
}
tr_math_poly *tr_math_poly_sub(const tr_math_poly *a,const tr_math_poly *b,int32_t *error) {
    return add(a,b,-1,error);
}
tr_math_poly *tr_math_poly_mul(const tr_math_poly *a,const tr_math_poly *b,int32_t *error) {
    if (!poly_valid(a) || !poly_valid(b)) return fail(NULL,TR_MATH_INVALID,error);
    tr_math_poly *out=poly_new(error);
    if (!out) return NULL;
    for (int i=0;i<a->n;++i) for (int j=0;j<b->n;++j) {
        uint8_t exponents[TR_MATH_VARIABLES]; int degree=0;
        for (int v=0;v<TR_MATH_VARIABLES;++v) {
            exponents[v]=(uint8_t)(a->terms[i].exponents[v]+b->terms[j].exponents[v]);
            degree+=exponents[v];
        }
        if (degree>TR_MATH_DEGREE) return fail(out,TR_MATH_BUDGET,error);
        tr_math_rational c;
        int rc=rational_mul(a->terms[i].coefficient,b->terms[j].coefficient,&c);
        if (!rc) rc=put(out,exponents,c,1);
        if (rc) return fail(out,rc,error);
    }
    sort(out); return out;
}
tr_math_poly *tr_math_poly_pow(const tr_math_poly *p,uint32_t exponent,int32_t *error) {
    if (!poly_valid(p)) return fail(NULL,TR_MATH_INVALID,error);
    if ((uint64_t)tr_math_poly_degree(p)*exponent>TR_MATH_DEGREE)
        return fail(NULL,TR_MATH_BUDGET,error);
    tr_math_poly *result=tr_math_poly_constant(1,1,error);
    if (!result) return NULL;
    tr_math_poly *base=poly_new(error);
    if (!base) { free(result); return NULL; }
    *base=*p;
    while (exponent) {
        tr_math_poly *next;
        if (exponent&1) {
            next=tr_math_poly_mul(result,base,error); free(result);
            if (!next) { free(base); return NULL; }
            result=next;
        }
        exponent>>=1;
        if (exponent) {
            next=tr_math_poly_mul(base,base,error); free(base);
            if (!next) { free(result); return NULL; }
            base=next;
        }
    }
    free(base); return result;
}
static tr_math_poly *calculus(const tr_math_poly *p,int32_t variable,int integrate,int32_t *error) {
    if (!poly_valid(p) || !variable_valid(variable)) return fail(NULL,TR_MATH_INVALID,error);
    tr_math_poly *out=poly_new(error);
    if (!out) return NULL;
    for (int i=0;i<p->n;++i) {
        const tr_math_term *source=&p->terms[i];
        unsigned power=source->exponents[variable];
        if (!integrate && !power) continue;
        uint8_t exponents[TR_MATH_VARIABLES]; memcpy(exponents,source->exponents,sizeof exponents);
        exponents[variable]=(uint8_t)(integrate ? power+1 : power-1);
        tr_math_rational c;
        int rc=integrate ? rational(source->coefficient.numerator,
                                   (__int128)source->coefficient.denominator*(power+1),&c) :
                           rational((__int128)source->coefficient.numerator*power,
                                    source->coefficient.denominator,&c);
        if (!rc) rc=put(out,exponents,c,1);
        if (rc) return fail(out,rc,error);
    }
    sort(out); return out;
}
tr_math_poly *tr_math_poly_derivative(const tr_math_poly *p,int32_t variable,int32_t *error) {
    return calculus(p,variable,0,error);
}
tr_math_poly *tr_math_poly_integrate(const tr_math_poly *p,int32_t variable,int32_t *error) {
    return calculus(p,variable,1,error);
}
tr_math_poly *tr_math_poly_substitute(const tr_math_poly *p,int32_t variable,
                                    int64_t n,int64_t d,int32_t *error) {
    if (!poly_valid(p) || !variable_valid(variable)) return fail(NULL,TR_MATH_INVALID,error);
    tr_math_rational value;
    int rc=rational(n,d,&value);
    if (rc) return fail(NULL,rc,error);
    tr_math_poly *out=poly_new(error);
    if (!out) return NULL;
    for (int i=0;i<p->n;++i) {
        uint8_t exponents[TR_MATH_VARIABLES]; memcpy(exponents,p->terms[i].exponents,sizeof exponents);
        tr_math_rational factor,c;
        rc=rational_pow(value,exponents[variable],&factor);
        if (!rc) rc=rational_mul(p->terms[i].coefficient,factor,&c);
        exponents[variable]=0;
        if (!rc) rc=put(out,exponents,c,1);
        if (rc) return fail(out,rc,error);
    }
    sort(out); return out;
}
int32_t tr_math_poly_evaluate(const tr_math_poly *p,const tr_math_rational *values,
                             tr_math_rational *output) {
    if (!poly_valid(p) || !values || !output) return TR_MATH_INVALID;
    tr_math_rational canonical[TR_MATH_VARIABLES],sum={0,1};
    for (int v=0;v<TR_MATH_VARIABLES;++v) {
        int rc=rational(values[v].numerator,values[v].denominator,&canonical[v]);
        if (rc) return rc;
    }
    for (int i=0;i<p->n;++i) {
        tr_math_rational term=p->terms[i].coefficient;
        for (int v=0;v<TR_MATH_VARIABLES;++v) if (p->terms[i].exponents[v]) {
            tr_math_rational factor;
            int rc=rational_pow(canonical[v],p->terms[i].exponents[v],&factor);
            if (!rc) rc=rational_mul(term,factor,&term);
            if (rc) return rc;
        }
        int rc=rational_add(sum,term,1,&sum);
        if (rc) return rc;
    }
    *output=sum; return TR_MATH_OK;
}
int32_t tr_math_poly_equal(const tr_math_poly *a,const tr_math_poly *b) {
    if (!poly_valid(a) || !poly_valid(b)) return TR_MATH_INVALID;
    if (a->n!=b->n) return 0;
    for (int i=0;i<a->n;++i)
        if (memcmp(a->terms[i].exponents,b->terms[i].exponents,TR_MATH_VARIABLES) ||
            a->terms[i].coefficient.numerator!=b->terms[i].coefficient.numerator ||
            a->terms[i].coefficient.denominator!=b->terms[i].coefficient.denominator) return 0;
    return 1;
}
int32_t tr_math_poly_term_count(const tr_math_poly *p) { return poly_valid(p) ? p->n : TR_MATH_INVALID; }
int32_t tr_math_poly_get_term(const tr_math_poly *p,int32_t index,tr_math_term *output) {
    if (!poly_valid(p) || !output || index<0 || index>=p->n) return TR_MATH_INVALID;
    *output=p->terms[index]; return TR_MATH_OK;
}
int32_t tr_math_poly_degree(const tr_math_poly *p) {
    if (!poly_valid(p)) return TR_MATH_INVALID;
    int maximum=0;
    for (int i=0;i<p->n;++i) {
        int degree=0;
        for (int v=0;v<TR_MATH_VARIABLES;++v) degree+=p->terms[i].exponents[v];
        if (degree>maximum) maximum=degree;
    }
    return maximum;
}
int32_t tr_math_poly_support(const tr_math_poly *p) {
    if (!poly_valid(p)) return TR_MATH_INVALID;
    int mask=0;
    for (int i=0;i<p->n;++i) for (int v=0;v<TR_MATH_VARIABLES;++v)
        if (p->terms[i].exponents[v]) mask|=1<<v;
    return mask;
}
void tr_math_poly_free(tr_math_poly *p) { if (p) { p->magic=0; free(p); } }

static int bank_find(const tr_math_bank *bank,const uint8_t *exponents) {
    for (int i=0;i<bank->n;++i)
        if (!bank->ndata || !memcmp(bank->terms[i].exponents,exponents,(size_t)bank->ndata)) return i;
    return -1;
}
tr_math_bank *tr_math_bank_create(int32_t ndata,int32_t nterms,
                                 const uint8_t *exponents,const int64_t *moments,int32_t *error) {
    if (ndata<0 || ndata>TR_MATH_DATA_VARIABLES ||
        nterms!=1+ndata+ndata*(ndata+1)/2 || !moments || (ndata && !exponents)) {
        error_set(error,TR_MATH_INVALID); return NULL;
    }
    tr_math_bank *bank=calloc(1,sizeof *bank);
    if (!bank) { error_set(error,TR_MATH_MEMORY); return NULL; }
    bank->magic=BANK_MAGIC; bank->ndata=ndata;
    int has_count=0;
    for (int i=0;i<nterms;++i) {
        const uint8_t *powers=ndata ? exponents+(size_t)i*ndata : NULL;
        int degree=0;
        for (int j=0;j<ndata;++j) degree+=powers[j];
        if (degree>2 || bank_find(bank,powers)>=0) {
            free(bank); error_set(error,TR_MATH_INVALID); return NULL;
        }
        if (!degree) { bank->count=moments[i]; has_count=1; }
        if (ndata) memcpy(bank->terms[i].exponents,powers,(size_t)ndata);
        bank->terms[i].value=moments[i]; ++bank->n;
    }
    if (!has_count || bank->count<0) {
        free(bank); error_set(error,TR_MATH_INVALID); return NULL;
    }
    if (!bank->count) for (int i=0;i<nterms;++i) if (bank->terms[i].value) {
        free(bank); error_set(error,TR_MATH_INVALID); return NULL;
    }
    error_set(error,TR_MATH_OK); return bank;
}
int32_t tr_math_bank_variables(const tr_math_bank *bank) { return bank_valid(bank) ? bank->ndata : TR_MATH_INVALID; }
int32_t tr_math_bank_term_count(const tr_math_bank *bank) { return bank_valid(bank) ? bank->n : TR_MATH_INVALID; }
int32_t tr_math_bank_count(const tr_math_bank *bank,int64_t *output) {
    if (!bank_valid(bank) || !output) return TR_MATH_INVALID;
    *output=bank->count; return TR_MATH_OK;
}
int32_t tr_math_bank_moment(const tr_math_bank *bank,const uint8_t *exponents,int64_t *output) {
    if (!bank_valid(bank) || !output || (bank->ndata && !exponents)) return TR_MATH_INVALID;
    int index=bank_find(bank,exponents);
    if (index<0) return TR_MATH_BUDGET;
    *output=bank->terms[index].value; return TR_MATH_OK;
}
tr_math_poly *tr_math_bank_reduce(const tr_math_bank *bank,const tr_math_poly *p,int32_t *error) {
    if (!bank_valid(bank) || !poly_valid(p)) return fail(NULL,TR_MATH_INVALID,error);
    tr_math_poly *out=poly_new(error);
    if (!out) return NULL;
    for (int i=0;i<p->n;++i) {
        int index=bank_find(bank,p->terms[i].exponents);
        if (index<0) return fail(out,TR_MATH_BUDGET,error);
        uint8_t exponents[TR_MATH_VARIABLES]; memcpy(exponents,p->terms[i].exponents,sizeof exponents);
        memset(exponents,0,(size_t)bank->ndata);
        tr_math_rational c;
        int rc=rational_mul(p->terms[i].coefficient,(tr_math_rational){bank->terms[index].value,1},&c);
        if (!rc) rc=put(out,exponents,c,1);
        if (rc) return fail(out,rc,error);
    }
    sort(out); return out;
}
void tr_math_bank_free(tr_math_bank *bank) { if (bank) { bank->magic=0; free(bank); } }
uint64_t tr_math_table_revision(const struct atable_s *table) { return table ? table->revision : 0; }
