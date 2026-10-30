/* Bounded whole-stage integer algebra. SPDX-License-Identifier: Apache-2.0
 *
 * A degree <=4 measure is separated into polynomials in a full group key and
 * degree <=2 residual polynomials. A low-cardinality hidden key can refine an
 * ungrouped reduction. Existing executors evaluate the residuals; this module
 * owns exact native finalization and small buffers. No formula names are matched.
 */
#include "relationships.h"
#include <limits.h>
#include <stdlib.h>
#include <string.h>

#define ST_TERMS 128
#define ST_DEGREE 4
#define ST_SCANS 2
#define ST_GROUPS 64
/* A small CPU pilot adjustment, fitted on mixed SF1 on the Core Ultra 9 290HX
 * Plus: retain C3's useful stage choice without selecting the slower C4 stage.
 * This is not a globally calibrated latency model; held-out scales must report
 * its misses as well as its wins. GPU relative work is unchanged. */
#define ST_CPU_WORD_FACTOR 0.9

typedef struct { int32_t degree, col[ST_DEGREE]; int64_t c; } st_term;
typedef struct { int32_t n; st_term term[ST_TERMS]; } st_poly;
typedef struct {
    uintptr_t table;
    uint64_t revision, rows, ngroups, nranges;
    int32_t ncols; tr_column columns[TR_MAX_COLUMNS];
    int32_t k, width[MAXCOL], msb[MAXCOL], backend, mode, row_policy;
    int32_t agg, nprog, nmeas, nkey, nlin, bil_a, bil_b, npl, npb;
    int32_t nsemi, nattr, xmode, wide, widekey, ext_on, sparse;
    pinstr_t prog[1024]; minstr_t meas[48];
    int32_t lin_col[8]; int64_t lin_coef[8];
    int32_t pl_col[8], pb_a[8], pb_b[8];
    int64_t pl_coef[8], pb_coef[8], pl_hi[8], pb_hi[8], p_const, p_hi;
    int32_t key_kind[MAXKEY], key_arg[MAXKEY], key_nbits[MAXKEY];
    uint32_t key_mask[MAXKEY]; uint64_t key_radix[MAXKEY];
} st_key;
typedef struct {
    st_key key;
    const atable_t *table;
    tr_stage_trace trace;
    qplan_t scan[ST_SCANS];
    tr_poly residual[ST_SCANS];
    int64_t weight[ST_SCANS][ST_DEGREE+1], constant[ST_DEGREE+1];
    int64_t sums[ST_SCANS][ST_GROUPS];
    uint64_t counts[ST_SCANS][ST_GROUPS];
} st_handle;

static int fits64(__int128 n) { return n >= INT64_MIN && n <= INT64_MAX; }
static uint64_t magnitude(int64_t n) { return n < 0 ? -(uint64_t)n : (uint64_t)n; }
static uint64_t gcd64(uint64_t a, uint64_t b) { while (b) { uint64_t r=a%b; a=b; b=r; } return a; }

static int add_term(st_poly *p, const int32_t *cols, int degree, __int128 c) {
    if (!c) return 1;
    if (degree < 0 || degree > ST_DEGREE || !fits64(c)) return 0;
    for (int i=0; i<p->n; ++i) if (p->term[i].degree==degree &&
        (!degree || !memcmp(p->term[i].col,cols,(size_t)degree*sizeof *cols))) {
        c+=p->term[i].c;
        if (!fits64(c)) return 0;
        if (!c) p->term[i]=p->term[--p->n]; else p->term[i].c=(int64_t)c;
        return 1;
    }
    if (p->n==ST_TERMS) return 0;
    st_term *t=&p->term[p->n++]; memset(t,0,sizeof *t);
    t->degree=degree; t->c=(int64_t)c;
    if (degree) memcpy(t->col,cols,(size_t)degree*sizeof *cols);
    return 1;
}

static int combine(const st_poly *a,const st_poly *b,int op,st_poly *out) {
    memset(out,0,sizeof *out);
    if (op==M_ADD || op==M_SUB) {
        for (int i=0;i<a->n;++i) if (!add_term(out,a->term[i].col,a->term[i].degree,a->term[i].c)) return 0;
        for (int i=0;i<b->n;++i) if (!add_term(out,b->term[i].col,b->term[i].degree,
            (__int128)b->term[i].c*(op==M_SUB?-1:1))) return 0;
        return 1;
    }
    for (int i=0;i<a->n;++i) for (int j=0;j<b->n;++j) {
        const st_term *x=&a->term[i],*y=&b->term[j]; int d=x->degree+y->degree, ia=0,ib=0; int32_t cols[ST_DEGREE];
        if (d>ST_DEGREE) return 0;
        for (int k=0;k<d;++k) cols[k]=ib==y->degree || (ia<x->degree && x->col[ia]<y->col[ib]) ? x->col[ia++] : y->col[ib++];
        if (!add_term(out,cols,d,(__int128)x->c*y->c)) return 0;
    }
    return 1;
}

/* Original dependency tracking deliberately precedes polynomial cancellation. */
static int analyze(const atable_t *t,const qplan_t *q,st_poly *out,uint64_t *dependencies,int *ops,int *degree) {
    if (!q->nmeas || !tr_measure_valid(q,t->k)) return 0;
    st_poly *work=calloc(73,sizeof *work);
    if (!work) return 0;
    st_poly *stack=work,*reg=work+24,*next=work+72; int sp=0,ok=1;
    *dependencies=0; *ops=0; *degree=0;
    for (int i=0;i<q->nmeas && ok;++i) {
        const minstr_t *m=&q->meas[i];
        if (m->op==M_SAVE) { reg[m->arg]=stack[sp-1]; continue; }
        if (m->op==M_LOAD) { stack[sp++]=reg[m->arg]; continue; }
        if (m->op==M_FCOL || m->op==M_CONST) {
            memset(&stack[sp],0,sizeof stack[sp]);
            if (m->op==M_FCOL) {
                int32_t c=m->arg; *dependencies|=UINT64_C(1)<<c;
                ok=add_term(&stack[sp],&c,1,1);
            } else ok=add_term(&stack[sp],NULL,0,m->c);
            ++sp;
        } else if (m->op==M_ADD || m->op==M_SUB || m->op==M_MUL) {
            ok=combine(&stack[sp-2],&stack[sp-1],m->op,next); --sp;
            if (ok) stack[sp-1]=*next;
            ++*ops;
        } else ok=0;
    }
    if (ok && sp==1) {
        *out=stack[0];
        for (int i=0;i<out->n;++i) if (out->term[i].degree>*degree) *degree=out->term[i].degree;
    } else ok=0;
    free(work); return ok;
}

static int known_nonnull(const tr_catalog *cat,const atable_t *t,int col) {
    if (col<0 || col>=t->k || t->width[col]<1 || t->width[col]>32 || t->vflag[col]) return 0;
    for (int i=0;i<cat->ncols;++i) if (cat->columns[i].col==col) return cat->columns[i].valid<0;
    return 0;
}

static int key_of(const atable_t *t,const qplan_t *q,int backend,int mode,st_key *key) {
    if (t->k<1 || t->k>MAXCOL || q->nprog<0 || q->nprog>1024 || q->nmeas<0 || q->nmeas>48 ||
        q->nkey<0 || q->nkey>MAXKEY || q->nlin<0 || q->nlin>8 || q->npl<0 || q->npl>8 || q->npb<0 || q->npb>8) return 0;
    memset(key,0,sizeof *key);
    key->table=(uintptr_t)t; key->revision=t->revision; key->rows=t->n; key->k=t->k;
    memcpy(key->width,t->width,sizeof key->width); memcpy(key->msb,t->msb,sizeof key->msb);
    key->backend=backend; key->mode=mode;
    const char *env=getenv("TORBLES_MM_ROWS"); key->row_policy=env && *env ? atoi(env) : INT_MIN;
    key->agg=q->agg; key->nprog=q->nprog; key->nmeas=q->nmeas; key->nkey=q->nkey;
    key->nlin=q->nlin; key->bil_a=q->bil_a; key->bil_b=q->bil_b; key->npl=q->npl; key->npb=q->npb;
    key->nsemi=q->nsemi; key->nattr=q->nattr; key->nranges=q->nranges; key->xmode=q->xmode;
    key->wide=q->wide; key->widekey=q->widekey; key->ext_on=q->ext_on; key->sparse=!!q->out_key; key->ngroups=q->ngroups;
    memcpy(key->prog,q->prog,(size_t)q->nprog*sizeof q->prog[0]); memcpy(key->meas,q->meas,(size_t)q->nmeas*sizeof q->meas[0]);
    memcpy(key->lin_col,q->lin_col,sizeof key->lin_col); memcpy(key->lin_coef,q->lin_coef,sizeof key->lin_coef);
    memcpy(key->pl_col,q->pl_col,sizeof key->pl_col); memcpy(key->pl_coef,q->pl_coef,sizeof key->pl_coef); memcpy(key->pl_hi,q->pl_coef_hi,sizeof key->pl_hi);
    memcpy(key->pb_a,q->pb_a,sizeof key->pb_a); memcpy(key->pb_b,q->pb_b,sizeof key->pb_b); memcpy(key->pb_coef,q->pb_coef,sizeof key->pb_coef); memcpy(key->pb_hi,q->pb_coef_hi,sizeof key->pb_hi);
    key->p_const=q->p_const; key->p_hi=q->p_const_hi;
    for (int i=0;i<q->nkey;++i) {
        key->key_kind[i]=q->key_kind[i]; key->key_arg[i]=q->key_arg[i]; key->key_nbits[i]=q->key_nbits[i];
        key->key_mask[i]=q->key_mask[i]; key->key_radix[i]=q->key_radix[i];
    }
    return 1;
}

typedef struct { int degree,col[2]; int64_t v[ST_DEGREE+1]; } st_row;

static int add_residual(tr_poly *p,const st_row *row,int64_t coefficient) {
    if (!coefficient) return 1;
    if (p->n==TR_MAX_TERMS) return 0;
    p->terms[p->n++]=(tr_term){row->col[0],row->degree==2?row->col[1]:-1,coefficient}; return 1;
}

/* Separate key powers, then merge proportional coefficient vectors. This keeps
 * an affine-offset square as one residual instead of unrelated x*x and x scans. */
static int decompose(const st_poly *p,int col,st_handle *h) {
    st_row rows[ST_TERMS]; int nr=0; memset(rows,0,sizeof rows);
    memset(h->residual,0,sizeof h->residual); memset(h->weight,0,sizeof h->weight); memset(h->constant,0,sizeof h->constant);
    int has_key=0;
    for (int i=0;i<p->n;++i) {
        const st_term *term=&p->term[i]; int power=0,rd=0,cols[2];
        for (int j=0;j<term->degree;++j) {
            if (term->col[j]==col) ++power;
            else { if (rd==2) return 0; cols[rd++]=term->col[j]; }
        }
        has_key|=power>0;
        if (!rd) { h->constant[power]=term->c; continue; }
        int r=0;
        while (r<nr && (rows[r].degree!=rd || rows[r].col[0]!=cols[0] || (rd==2 && rows[r].col[1]!=cols[1]))) ++r;
        if (r==nr) { rows[nr].degree=rd; rows[nr].col[0]=cols[0]; if (rd==2) rows[nr].col[1]=cols[1]; ++nr; }
        rows[r].v[power]=term->c;
    }
    if (!has_key) return 0;
    int scans=0,merge_ok=1;
    for (int r=0;r<nr;++r) {
        uint64_t gcd=0; int first=0; int64_t primitive[ST_DEGREE+1];
        for (int a=0;a<=ST_DEGREE;++a) gcd=gcd64(gcd,magnitude(rows[r].v[a]));
        if (!gcd) continue;
        while (first<=ST_DEGREE && !rows[r].v[first]) ++first;
        int sign=rows[r].v[first]<0?-1:1; __int128 scale=(__int128)gcd*sign;
        if (!fits64(scale)) { merge_ok=0; break; }
        for (int a=0;a<=ST_DEGREE;++a) {
            __int128 v=(__int128)rows[r].v[a]/(__int128)gcd*sign;
            if (!fits64(v)) { merge_ok=0; break; }
            primitive[a]=(int64_t)v;
        }
        if (!merge_ok) break;
        int s=0; while (s<scans && memcmp(h->weight[s],primitive,sizeof primitive)) ++s;
        if (s==scans) {
            if (scans==ST_SCANS) { merge_ok=0; break; }
            memcpy(h->weight[scans++],primitive,sizeof primitive);
        }
        if (!add_residual(&h->residual[s],&rows[r],(int64_t)scale)) { merge_ok=0; break; }
    }
    if (!merge_ok) {
        if (nr>ST_SCANS) return 0;
        scans=nr; memset(h->residual,0,sizeof h->residual); memset(h->weight,0,sizeof h->weight);
        for (int r=0;r<nr;++r) {
            memcpy(h->weight[r],rows[r].v,sizeof rows[r].v);
            if (!add_residual(&h->residual[r],&rows[r],1)) return 0;
        }
    }
    h->trace.scan_count=scans?scans:1;
    h->trace.partition_kind=1;
    h->trace.residual_degree=h->trace.residual_terms=h->trace.coefficient_terms=0;
    for (int s=0;s<scans;++s) for (int i=0;i<h->residual[s].n;++i) {
        int degree=h->residual[s].terms[i].b>=0?2:1;
        if (degree>h->trace.residual_degree) h->trace.residual_degree=degree;
        ++h->trace.residual_terms;
    }
    for (int a=0;a<=ST_DEGREE;++a) {
        h->trace.coefficient_terms+=h->constant[a]!=0;
        for (int s=0;s<scans;++s) h->trace.coefficient_terms+=h->weight[s][a]!=0;
    }
    return 1;
}

/* Exact synthetic division by the group-invariant affine factor (key+shift).
 * Coefficients are grouped by the remaining row monomial. No sampled equality
 * or literal expression recognition can authorize this transformation. */
static int divide_key_factor(const st_poly *p,int col,int64_t shift,st_poly *out) {
    st_row rows[ST_TERMS]; int nr=0; memset(rows,0,sizeof rows); memset(out,0,sizeof *out);
    for (int i=0;i<p->n;++i) {
        const st_term *term=&p->term[i]; int power=0,rd=0,cols[2];
        for (int j=0;j<term->degree;++j) {
            if (term->col[j]==col) ++power;
            else { if (rd==2) return 0; cols[rd++]=term->col[j]; }
        }
        int r=0;
        while (r<nr && (rows[r].degree!=rd || (rd && rows[r].col[0]!=cols[0]) || (rd==2 && rows[r].col[1]!=cols[1]))) ++r;
        if (r==nr) { rows[nr].degree=rd; if (rd) rows[nr].col[0]=cols[0]; if (rd==2) rows[nr].col[1]=cols[1]; ++nr; }
        rows[r].v[power]=term->c;
    }
    for (int r=0;r<nr;++r) {
        int top=ST_DEGREE; while (top && !rows[r].v[top]) --top;
        if (!top) return 0;
        int64_t quotient[ST_DEGREE]={0}; quotient[top-1]=rows[r].v[top];
        for (int a=top-1;a>=1;--a) {
            __int128 value=(__int128)rows[r].v[a]-(__int128)shift*quotient[a];
            if (!fits64(value)) return 0;
            quotient[a-1]=(int64_t)value;
        }
        if ((__int128)rows[r].v[0]-(__int128)shift*quotient[0]) return 0;
        for (int a=0;a<top;++a) if (quotient[a]) {
            int32_t cols[ST_DEGREE]; int rd=rows[r].degree, at=0,keys=a,pos=0;
            while (at<rd || keys) {
                if (keys && (at==rd || col<rows[r].col[at])) { cols[pos++]=col; --keys; }
                else cols[pos++]=rows[r].col[at++];
            }
            if (!add_term(out,cols,pos,quotient[a])) return 0;
        }
    }
    return 1;
}

static int factor_weight(int64_t *w,int64_t shift) {
    int64_t next[ST_DEGREE+1];
    for (int a=0;a<=ST_DEGREE;++a) {
        __int128 value=(__int128)w[a]*shift+(a?w[a-1]:0);
        if (!fits64(value)) return 0;
        next[a]=(int64_t)value;
    }
    memcpy(w,next,sizeof next); return 1;
}

static int residual_from_poly(const st_poly *p,const int64_t *weight,st_handle *h) {
    memset(h->residual,0,sizeof h->residual); memset(h->weight,0,sizeof h->weight); memset(h->constant,0,sizeof h->constant);
    h->trace.scan_count=1; h->trace.partition_kind=2;
    h->trace.residual_degree=h->trace.residual_terms=h->trace.coefficient_terms=0;
    for (int i=0;i<p->n;++i) {
        const st_term *term=&p->term[i];
        if (term->degree>2 || h->residual[0].n==TR_MAX_TERMS) return 0;
        h->residual[0].terms[h->residual[0].n++]=(tr_term){term->degree?term->col[0]:-1,term->degree==2?term->col[1]:-1,term->c};
        if (term->degree>h->trace.residual_degree) h->trace.residual_degree=term->degree;
        ++h->trace.residual_terms;
    }
    memcpy(h->weight[0],weight,sizeof h->weight[0]);
    for (int a=0;a<=ST_DEGREE;++a) h->trace.coefficient_terms+=weight[a]!=0;
    return 1;
}

/* Conservative absolute bounds deliberately decline cancellation-dependent
 * overflow cases. All intermediate residual sums and final folds must fit. */
static int bounded_mul(__int128 *n,uint64_t v) {
    if (v && *n>INT64_MAX/(__int128)v) return 0;
    *n*=v; return 1;
}

static int bound_weights(const int64_t *w,int maxkey,__int128 *out) {
    __int128 sum=0;
    for (int a=0;a<=ST_DEGREE;++a) {
        __int128 value=magnitude(w[a]);
        if (value>INT64_MAX) return 0;
        for (int p=0;p<a;++p) if (!bounded_mul(&value,(uint64_t)maxkey)) return 0;
        sum+=value; if (sum>INT64_MAX) return 0;
    }
    *out=sum; return 1;
}

static int bounds(const atable_t *t,const st_handle *h) {
    __int128 total=0,weight;
    if (!bound_weights(h->constant,h->trace.key_domain-1,&weight) || !bounded_mul(&weight,t->n)) return 0;
    total=weight;
    for (int s=0;s<h->trace.scan_count;++s) {
        __int128 residual=0;
        for (int i=0;i<h->residual[s].n;++i) {
            const tr_term *term=&h->residual[s].terms[i]; __int128 v=magnitude(term->coefficient);
            if (v>INT64_MAX || (term->a>=0 && !bounded_mul(&v,(UINT64_C(1)<<t->width[term->a])-1)) ||
                (term->b>=0 && !bounded_mul(&v,(UINT64_C(1)<<t->width[term->b])-1))) return 0;
            residual+=v; if (residual>INT64_MAX) return 0;
        }
        if (!bounded_mul(&residual,t->n) || !bound_weights(h->weight[s],h->trace.key_domain-1,&weight)) return 0;
        if (residual && weight>INT64_MAX/residual) return 0;
        total+=residual*weight; if (total>INT64_MAX) return 0;
    }
    return 1;
}

static int make_scans(st_handle *h,const qplan_t *source) {
    for (int s=0;s<h->trace.scan_count;++s) {
        qplan_t *q=&h->scan[s]; *q=*source;
        /* Grouped COUNT is represented by the polynomial group executor with
         * a zero measure. A_COUNT is an ungrouped fast path and does not fill
         * per-group counters. The existing out_cnt state supplies COUNT here. */
        q->agg=A_POLYG; q->nmeas=q->nlin=q->npl=q->npb=0;
        memset(q->meas,0,sizeof q->meas); q->p_const=q->p_const_hi=0;
        memset(q->pl_coef_hi,0,sizeof q->pl_coef_hi); memset(q->pb_coef_hi,0,sizeof q->pb_coef_hi);
        for (int i=0;i<h->residual[s].n;++i) {
            const tr_term *term=&h->residual[s].terms[i];
            if (term->a<0) q->p_const=term->coefficient;
            else if (term->b<0) {
                int j=q->npl++; if (j==8) return 0;
                q->pl_col[j]=term->a; q->pl_coef[j]=term->coefficient;
            } else {
                int j=q->npb++; if (j==8) return 0;
                q->pb_a[j]=term->a; q->pb_b[j]=term->b; q->pb_coef[j]=term->coefficient;
            }
        }
        q->nkey=1; q->key_kind[0]=0; q->key_arg[0]=h->trace.key_col; q->key_radix[0]=(uint64_t)h->trace.key_domain;
        q->key_nbits[0]=0; q->key_mask[0]=0; q->ngroups=(uint64_t)h->trace.key_domain;
        q->out_sum=h->sums[s]; q->out_cnt=h->counts[s]; q->out_sum_hi=NULL; q->out_key=NULL; q->out_key_hi=NULL;
        q->out_cap=q->out_n=0; q->out_rows=NULL; q->wide=q->widekey=0;
        /* Eligibility excludes these scopes; do not retain even inactive
         * addresses from the source query in the owned residual programs. */
        memset(q->semi_bm,0,sizeof q->semi_bm); memset(q->attr_vals,0,sizeof q->attr_vals);
        q->ranges=NULL; q->xrows=NULL; q->dom_t=NULL;
        q->total_sum=q->total_sum_hi=0; q->total_cnt=0; memset(q->mm_stats,0,sizeof q->mm_stats);
    }
    return 1;
}

static double best_existing(const atable_t *t,const qplan_t *source) {
    qplan_t q=*source; tr_system_trace trace;
    if (tr_plan_system(t,&q,TR_BACKEND_CPU,TR_SYSTEM_AUTO,&trace)!=TR_OK) return -1;
    double best=-1;
    double values[3]={trace.chain_cost,trace.combine_cost,trace.hybrid_cost};
    for (int i=0;i<3;++i) if (values[i]>=0 && (best<0 || values[i]<best)) best=values[i];
    return best;
}

static int candidate_cost(st_handle *h,const atable_t *t,int backend,double chain) {
    tr_system_sample sample;
    if (at_relationship_sample(t,&h->scan[0],&sample)!=TR_OK) return 0;
    double sample_cost=0;
    for (int s=0;s<h->trace.scan_count;++s) {
        uint64_t inputs=UINT64_C(1)<<h->trace.key_col; double bits=0;
        for (int i=0;i<h->residual[s].n;++i) {
            if (h->residual[s].terms[i].a>=0) inputs|=UINT64_C(1)<<h->residual[s].terms[i].a;
            if (h->residual[s].terms[i].b>=0) inputs|=UINT64_C(1)<<h->residual[s].terms[i].b;
        }
        for (int c=0;c<t->k;++c) if (inputs>>c&1) bits+=t->width[c];
        double plane=tr_poly_cost(t,&h->residual[s]);
        sample_cost+=sample.nonempty*bits*0.3+sample.groups*(plane+t->width[h->trace.key_col]*0.5+2.0);
    }
    double scale=sample.words ? (double)t->nw/sample.words : 0;
    h->trace.chain_cost=chain>=0?chain*scale:-1;
    /* GPU factor and launch/finalization charges are explicit relative work
     * heuristics, not nanoseconds or a claim of calibrated hardware throughput. */
    h->trace.compose_cost=sample_cost*scale*(backend==TR_BACKEND_GPU?1.0/16.0:ST_CPU_WORD_FACTOR)+
        h->trace.key_domain*(16.0+8.0*h->trace.scan_count)+(backend==TR_BACKEND_GPU?1024.0*h->trace.scan_count:0);
    return 1;
}

static void consider(st_handle *best,st_handle *candidate,const atable_t *t,const qplan_t *q,
                     int backend,double chain,double *cost,int *count,int *refusal) {
    if (!bounds(t,candidate) || !make_scans(candidate,q)) { *refusal=1; return; }
    if (!candidate_cost(candidate,t,backend,chain)) return;
    ++*count;
    if (*cost<0 || candidate->trace.compose_cost<*cost) {
        *cost=candidate->trace.compose_cost; best->trace=candidate->trace;
        memcpy(best->residual,candidate->residual,sizeof best->residual);
        memcpy(best->weight,candidate->weight,sizeof best->weight);
        memcpy(best->constant,candidate->constant,sizeof best->constant);
    }
}

void tr_stage_free(void *opaque) { free(opaque); }
int32_t tr_stage_scan_count(const void *opaque) {
    const st_handle *h=opaque; return h && h->trace.choice ? h->trace.scan_count : 0;
}
qplan_t *tr_stage_scan(void *opaque,int32_t index) {
    st_handle *h=opaque;
    return h && h->trace.choice && index>=0 && index<h->trace.scan_count ? &h->scan[index] : NULL;
}
int32_t tr_stage_weights(const void *opaque,int32_t index,int64_t *out) {
    const st_handle *h=opaque;
    if (!h || !out || !h->trace.choice || index < -1 || index>=h->trace.scan_count) return TR_INVALID;
    memcpy(out,index<0?h->constant:h->weight[index],(ST_DEGREE+1)*sizeof *out);
    return TR_OK;
}
int32_t tr_stage_reset(void *opaque) {
    st_handle *h=opaque;
    if (!h || !h->trace.choice) return TR_INVALID;
    memset(h->sums,0,sizeof h->sums); memset(h->counts,0,sizeof h->counts);
    for (int s=0;s<h->trace.scan_count;++s) {
        h->scan[s].total_sum=h->scan[s].total_sum_hi=0; h->scan[s].total_cnt=0; h->scan[s].out_n=0;
    }
    return TR_OK;
}

static int weight_at(const int64_t *w,int key,__int128 *out) {
    __int128 sum=0;
    for (int a=ST_DEGREE;a>=0;--a)
        if (__builtin_mul_overflow(sum,(__int128)key,&sum) || __builtin_add_overflow(sum,(__int128)w[a],&sum)) return 0;
    *out=sum; return 1;
}

int32_t tr_stage_finish(void *opaque,qplan_t *live) {
    st_handle *h=opaque;
    if (!h || !live || !h->trace.choice) return TR_INVALID;
    if (h->table->revision!=h->key.revision) return TR_STALE;
    if (live->nkey!=h->key.nkey || (live->nkey &&
        (live->ngroups!=h->key.ngroups || live->key_kind[0]!=0 || live->key_arg[0]!=h->trace.key_col ||
         live->key_radix[0]!=(uint64_t)h->trace.key_domain || !live->out_sum || !live->out_cnt))) return TR_INVALID;
    int64_t result[ST_GROUPS]; uint64_t counts[ST_GROUPS]; __int128 total=0; uint64_t total_count=0;
    memset(result,0,sizeof result); memset(counts,0,sizeof counts);
    for (int g=0;g<h->trace.key_domain;++g) {
        uint64_t count=h->counts[0][g]; __int128 sum,weight,part;
        if (count>h->table->n || total_count>h->table->n-count) return TR_INVALID;
        for (int s=1;s<h->trace.scan_count;++s) if (h->counts[s][g]!=count) return TR_INVALID;
        if (!weight_at(h->constant,g,&weight) || __builtin_mul_overflow(weight,(__int128)count,&sum)) return TR_OVERFLOW;
        for (int s=0;s<h->trace.scan_count;++s) {
            if (!weight_at(h->weight[s],g,&weight) || __builtin_mul_overflow(weight,(__int128)h->sums[s][g],&part) ||
                __builtin_add_overflow(sum,part,&sum)) return TR_OVERFLOW;
        }
        if (!fits64(sum) || __builtin_add_overflow(total,sum,&total) || !fits64(total)) return TR_OVERFLOW;
        result[g]=(int64_t)sum; counts[g]=count; total_count+=count;
    }
    if (live->nkey) {
        memset(live->out_sum,0,(size_t)live->ngroups*sizeof *live->out_sum);
        memset(live->out_cnt,0,(size_t)live->ngroups*sizeof *live->out_cnt);
        memcpy(live->out_sum,result,(size_t)h->trace.key_domain*sizeof *result);
        memcpy(live->out_cnt,counts,(size_t)h->trace.key_domain*sizeof *counts);
    }
    live->total_sum=(int64_t)total; live->total_sum_hi=0; live->total_cnt=total_count;
    return TR_OK;
}

int32_t tr_stage_prepare(void **opaque,tr_catalog *cat,const atable_t *t,qplan_t *live,const void *source,
                         int32_t backend,int32_t mode,tr_stage_trace *trace) {
    if (!opaque || !t || !live || !trace || mode<TR_STAGE_OFF || mode>TR_STAGE_COMPOSE ||
        backend<TR_BACKEND_CPU || backend>TR_BACKEND_PLAIN) return TR_INVALID;
    memset(trace,0,sizeof *trace); trace->key_col=-1; trace->chain_cost=trace->compose_cost=-1;
    if (mode==TR_STAGE_OFF) { trace->reason=5; return 0; }
    if (!cat || cat->table!=t || cat->revision!=t->revision) { trace->reason=6; return 0; }
    qplan_t q=*live;
    if (source && tr_system_source_restore(&q,source)!=TR_OK) return TR_INVALID;
    st_key key;
    if (!key_of(t,&q,backend,mode,&key)) { trace->reason=2; return 0; }
    if (cat->ncols<1 || cat->ncols>TR_MAX_COLUMNS) { trace->reason=6; return 0; }
    key.ncols=cat->ncols; memcpy(key.columns,cat->columns,(size_t)cat->ncols*sizeof cat->columns[0]);
    st_handle *old=*opaque;
    if (old && !memcmp(&old->key,&key,sizeof key)) {
        *trace=old->trace; trace->cache_hit=1; return trace->choice?1:0;
    }
    st_handle *h=calloc(1,sizeof *h);
    if (!h) { trace->reason=4; return 0; }
    h->key=key; h->table=t; h->trace=*trace; h->trace.workspace_bytes=sizeof *h;
    if (backend==TR_BACKEND_PLAIN) { h->trace.reason=7; goto done; }
    if (q.agg!=A_GENERAL || q.wide || q.widekey || q.ext_on || q.nattr || q.nsemi || q.nranges || q.xmode || q.out_key ||
        q.nkey>1 || (q.nkey && (q.key_kind[0]!=0 || q.key_radix[0]<1 || q.key_radix[0]>ST_GROUPS ||
        q.ngroups<q.key_radix[0] || q.ngroups>ST_GROUPS || !q.out_sum || !q.out_cnt))) { h->trace.reason=2; goto done; }
    if (!q.nmeas) { h->trace.reason=5; goto done; }
    st_poly *poly=malloc(sizeof *poly); uint64_t deps=0; int ops=0,degree=0;
    if (!poly) { h->trace.reason=4; goto done; }
    if (!analyze(t,&q,poly,&deps,&ops,&degree)) { free(poly); h->trace.reason=4; goto done; }
    h->trace.input_degree=degree; h->trace.source_ops=ops;
    for (int c=0;c<t->k;++c) if ((deps>>c&1) && !known_nonnull(cat,t,c)) {
        free(poly); h->trace.reason=3; goto done;
    }
    if (degree<=2) { free(poly); h->trace.reason=5; goto done; }
    double chain=best_existing(t,&q),best=-1; st_handle *candidate=calloc(1,sizeof *candidate);
    if (!candidate) { free(poly); h->trace.reason=4; goto done; }
    int bounds_refusal=0,candidate_count=0;
    for (int c=0;c<t->k;++c) {
        if (q.nkey ? c!=q.key_arg[0] : !(deps>>c&1)) continue;
        if (!known_nonnull(cat,t,c)) continue;
        if (!q.nkey && (t->width[c]>4 || (1<<t->width[c])>16)) continue;
        memset(candidate,0,sizeof *candidate); candidate->trace=h->trace;
        candidate->trace.key_col=c; candidate->trace.hidden_key=!q.nkey;
        candidate->trace.key_domain=q.nkey?(int)q.key_radix[0]:(1<<t->width[c]);
        if (decompose(poly,c,candidate))
            consider(h,candidate,t,&q,backend,chain,&best,&candidate_count,&bounds_refusal);
        int64_t offset=0;
        for (int i=0;i<cat->ncols;++i) if (cat->columns[i].col==c) offset=cat->columns[i].offset;
        int64_t shifts[2]={0,offset};
        for (int sh=0;sh<(offset?2:1);++sh) {
            st_poly current=*poly,next; int64_t weight[ST_DEGREE+1]={1,0,0,0,0};
            for (int repeat=0;repeat<2;++repeat) {
                if (!divide_key_factor(&current,c,shifts[sh],&next) || !factor_weight(weight,shifts[sh])) break;
                current=next;
                memset(candidate,0,sizeof *candidate); candidate->trace=h->trace;
                candidate->trace.key_col=c; candidate->trace.hidden_key=!q.nkey;
                candidate->trace.key_domain=q.nkey?(int)q.key_radix[0]:(1<<t->width[c]);
                if (residual_from_poly(&current,weight,candidate))
                    consider(h,candidate,t,&q,backend,chain,&best,&candidate_count,&bounds_refusal);
            }
        }
    }
    free(candidate); free(poly);
    h->trace.candidate_count=candidate_count;
    if (best<0) { h->trace.reason=bounds_refusal?4:5; goto done; }
    if (mode==TR_STAGE_AUTO && (h->trace.chain_cost<0 || h->trace.compose_cost>=h->trace.chain_cost)) { h->trace.reason=0; goto done; }
    h->trace.choice=1; h->trace.reason=mode==TR_STAGE_COMPOSE?1:0;
    if (!make_scans(h,&q)) { h->trace.choice=0; h->trace.reason=4; }
done:
    free(old); *opaque=h; *trace=h->trace;
    return trace->choice?1:0;
}
