/* META-002: query families as one multi-output program; see math_program.h. SPDX-License-Identifier: Apache-2.0 */
#include "math_program.h"
#include "mathematics.h"
#include "math_domain.h"
#include "math_handles.h"
#include <stdlib.h>
#include <string.h>

TR_MATH_ABI(tr_prog_measure, 1);           /* the ctypes boundary of META-002 (math_handles.h) */
TR_MATH_ABI(tr_prog_query, 1);
TR_MATH_ABI(tr_prog_family, 1);
TR_MATH_ABI(tr_prog_fmeasure, 1);
TR_MATH_ABI(tr_prog_fterm, 1);
TR_MATH_ABI(tr_prog_fout, 1);
TR_MATH_ABI(tr_prog_final_view, 1);
TR_MATH_ABI(tr_prog_value, 1);
TR_MATH_ABI(tr_prog_info, 1);
TR_MATH_ABI(tr_prog_query_info, 1);
TR_MATH_ABI(tr_prog_measure_info, 1);
TR_MATH_ABI(tr_prog_compare_result, 1);
TR_MATH_ABI(tr_prog_snapshot_result, 1);
TR_MATH_ABI(tr_prog_output_info, 1);

_Static_assert(sizeof(tr_prog_measure) == 112, "tr_prog_measure ABI");
_Static_assert(sizeof(tr_prog_query) == 32, "tr_prog_query ABI");
_Static_assert(sizeof(tr_prog_family) == 88, "tr_prog_family ABI");
_Static_assert(sizeof(tr_prog_fmeasure) == 40, "tr_prog_fmeasure ABI");
_Static_assert(sizeof(tr_prog_fterm) == 16, "tr_prog_fterm ABI");
_Static_assert(sizeof(tr_prog_fout) == 40, "tr_prog_fout ABI");
_Static_assert(sizeof(tr_prog_final_view) == 48, "tr_prog_final_view ABI");
_Static_assert(sizeof(tr_prog_value) == 72, "tr_prog_value ABI");
_Static_assert(sizeof(tr_prog_info) == 88, "tr_prog_info ABI");
_Static_assert(sizeof(tr_prog_query_info) == 64, "tr_prog_query_info ABI");
_Static_assert(sizeof(tr_prog_measure_info) == 40, "tr_prog_measure_info ABI");
_Static_assert(sizeof(tr_prog_compare_result) == 328, "tr_prog_compare_result ABI");
_Static_assert(sizeof(tr_prog_snapshot_result) == 56, "tr_prog_snapshot_result ABI");
_Static_assert(sizeof(tr_prog_output_info) == 128, "tr_prog_output_info ABI");

#define PROG_MAGIC UINT32_C(0x474f5250)
#define NCOND TR_REDUCE_MAX_CONDITIONS
enum { K_F = 0, K_T = 1, K_U = 2 };            /* scalar Kleene values */

/* one canonical measure */
typedef struct {
    int32_t kind, query, nterms, first;   /* first: its canonical terms in the program's term arrays */
    uint32_t colset;                      /* the columns of its polynomial as written (SQL: NULL where one is NULL),
                                             also those of terms that cancel or have a zero coefficient */
    int32_t homogeneous;                  /* its validity is a SQL-mode term's: no column, or every canonical
                                             monomial over exactly colset */
    int32_t output, count_term;
    tr_dist_spec spec;                    /* HIST, normalized */
} cmeasure;

/* one output under construction (its key and basis) */
typedef struct {
    int32_t kind, filter, grouping, mode;  /* TR_REDUCE_BANK / _HIST, canonical filter, grouping, bank mode */
    uint32_t colset;                       /* JOINT: the column set */
    int32_t nterms, cap, hist_measure;     /* basis monomials; HIST: the measure whose spec it holds */
    uint8_t *exps;                         /* nterms x ncols */
} pout;

struct tr_prog_s {
    uint32_t magic;
    int32_t ncols, nqueries, nmeasures, flags, lifetime;
    const tr_scope *scope;
    tr_scope_view sv;
    uint32_t nullable;                     /* columns that may be NULL in the declared universe */
    tr_bool_condition conds[NCOND];
    int32_t comp[NCOND];                   /* complement condition (-1 none, -2 not computed) */
    int32_t nconds, conds_in;
    tr_bool_node *nodes;
    uint64_t *deps;
    int32_t *neg;                          /* the canonical negation of each node once computed (-1: not yet) */
    int32_t nnodes, capnodes, nodes_in;
    uint32_t *htab;
    uint32_t hcap;
    int32_t *qfilter, *qclass, *qproof;    /* per query */
    tr_prog_query *queries;
    cmeasure *cm;
    uint8_t *texps;                        /* canonical polynomial terms: ncols bytes each */
    int64_t *tcoef;
    int32_t ntterms;
    pout *outs;
    int32_t nouts;
    tr_reduce_plan *plan;
    tr_prog_fmeasure *fm;
    tr_prog_fterm *ft;
    int32_t nfm, nft;
    uint64_t nslots;
    tr_prog_info info;
    tr_prog_query_info *qinfo;
    tr_prog_measure_info *minfo;
};

static void set_error(int32_t *error, int32_t code) { if (error) *error = code; }
static void *zalloc(uint64_t bytes) {
    if (bytes > (uint64_t)SIZE_MAX) return NULL;
    return calloc(1, bytes ? (size_t)bytes : 1);
}
static int prog_valid(const tr_prog *p) { return p && p->magic == PROG_MAGIC; }
static const tr_math_budget *budget_or(const tr_math_budget *b, tr_math_budget *dflt) {
    if (b) return b;
    tr_budget_default(dflt);
    return dflt;
}
static void need_zero(tr_math_need *n) { for (int i = 0; i < TR_BUDGET_RESOURCES; i++) n->need[i] = 0; }

void tr_prog_free(tr_prog *p) {
    if (!prog_valid(p)) return;
    free(p->nodes); free(p->deps); free(p->neg); free(p->htab); free(p->qfilter); free(p->qclass); free(p->qproof); free(p->queries);
    free(p->cm); free(p->texps); free(p->tcoef);
    if (p->outs) for (int32_t o = 0; o < p->nouts; o++) free(p->outs[o].exps);
    free(p->outs); free(p->fm); free(p->ft); free(p->qinfo); free(p->minfo);
    tr_reduce_plan_free(p->plan);
    p->magic = 0;
    free(p);
}

/* ------------------------------------------------------------------ canonical conditions */
static int64_t pow10s(int32_t s) { int64_t p = 1; for (int32_t i = 0; i < s; i++) p *= 10; return p; }
static __int128 floor_div(__int128 n, __int128 d) { __int128 q = n / d; if (n % d && (n < 0) != (d < 0)) q--; return q; }
static __int128 ceil_div(__int128 n, __int128 d) { __int128 q = n / d; if (n % d && (n < 0) == (d < 0)) q++; return q; }
static void make_interval(tr_bool_condition *c, int32_t col, __int128 lo, __int128 hi, int64_t p) {
    memset(c, 0, sizeof *c);
    c->op = TR_BOOL_BETWEEN; c->col = col; c->col2 = -1; c->den[0] = p; c->den[1] = p;
    if (lo < (__int128)INT64_MIN) lo = INT64_MIN;
    if (hi > (__int128)INT64_MAX) hi = INT64_MAX;
    if (lo > hi) { lo = 1; hi = 0; }                     /* the canonical empty interval */
    c->num[0] = (int64_t)lo; c->num[1] = (int64_t)hi;
}
/* the canonical form of a (checked) condition over the scope view */
static void canon_condition(const tr_scope_view *v, const tr_bool_condition *in, tr_bool_condition *out) {
    *out = *in;
    if (in->op == TR_BOOL_IS_NULL || in->op == TR_BOOL_IS_NOT_NULL || in->op == TR_BOOL_BIT) return;
    if (in->op >= TR_BOOL_COL_EQ) {
        if (in->col > in->col2) {
            static const int32_t mirror[7] = {0, TR_BOOL_EQ, TR_BOOL_NE, TR_BOOL_GT, TR_BOOL_GE, TR_BOOL_LT, TR_BOOL_LE};
            int32_t base = in->op - TR_BOOL_COL_EQ + TR_BOOL_EQ;
            out->op = mirror[base] - TR_BOOL_EQ + TR_BOOL_COL_EQ;
            out->col = in->col2; out->col2 = in->col;
        }
        return;
    }
    const int64_t p = pow10s(v->col[in->col].scale);
    const __int128 big = (__int128)1 << 100;
    __int128 n0 = (__int128)in->num[0] * p, d0 = in->den[0];
    switch (in->op) {
    case TR_BOOL_GT: make_interval(out, in->col, floor_div(n0, d0) + 1, big, p); return;
    case TR_BOOL_GE: make_interval(out, in->col, ceil_div(n0, d0), big, p); return;
    case TR_BOOL_LT: make_interval(out, in->col, -big, ceil_div(n0, d0) - 1, p); return;
    case TR_BOOL_LE: make_interval(out, in->col, -big, floor_div(n0, d0), p); return;
    case TR_BOOL_EQ:
        if (n0 % d0) make_interval(out, in->col, 1, 0, p);
        else make_interval(out, in->col, n0 / d0, n0 / d0, p);
        return;
    case TR_BOOL_BETWEEN: {
        __int128 n1 = (__int128)in->num[1] * p, d1 = in->den[1];
        make_interval(out, in->col, ceil_div(n0, d0), floor_div(n1, d1), p);
        return;
    }
    case TR_BOOL_NE:
        if (n0 % d0 || n0 / d0 > (__int128)INT64_MAX || n0 / d0 < (__int128)INT64_MIN)
            make_interval(out, in->col, -big, big, p);    /* never equal: TRUE on every present value */
        else {
            memset(out, 0, sizeof *out);
            out->op = TR_BOOL_NE; out->col = in->col; out->col2 = -1;
            out->num[0] = (int64_t)(n0 / d0); out->den[0] = p; out->den[1] = 1;
        }
        return;
    default: return;
    }
}
static int32_t intern_condition(tr_prog *p, const tr_bool_condition *c) {
    for (int32_t i = 0; i < p->nconds; i++) if (!memcmp(&p->conds[i], c, sizeof *c)) return i;
    if (p->nconds >= NCOND) return -1;
    p->conds[p->nconds] = *c;
    p->comp[p->nconds] = -2;
    return p->nconds++;
}
/* the complement of canonical condition i as a canonical condition (or -1: none) */
static int32_t complement(tr_prog *p, int32_t i) {
    if (p->comp[i] != -2) return p->comp[i];
    const tr_bool_condition *c = &p->conds[i];
    tr_bool_condition x = *c;
    int ok = 1;
    if (c->op == TR_BOOL_IS_NULL) x.op = TR_BOOL_IS_NOT_NULL;
    else if (c->op == TR_BOOL_IS_NOT_NULL) x.op = TR_BOOL_IS_NULL;
    else if (c->op == TR_BOOL_NE) make_interval(&x, c->col, c->num[0], c->num[0], c->den[0]);
    else if (c->op == TR_BOOL_BETWEEN) {
        int64_t lo = c->num[0], hi = c->num[1];
        if (lo > hi) make_interval(&x, c->col, INT64_MIN, INT64_MAX, c->den[0]);            /* empty <-> full */
        else if (lo == INT64_MIN && hi == INT64_MAX) make_interval(&x, c->col, 1, 0, c->den[0]);
        else if (lo == INT64_MIN) make_interval(&x, c->col, (__int128)hi + 1, INT64_MAX, c->den[0]);
        else if (hi == INT64_MAX) make_interval(&x, c->col, INT64_MIN, (__int128)lo - 1, c->den[0]);
        else if (lo == hi) { memset(&x, 0, sizeof x); x.op = TR_BOOL_NE; x.col = c->col; x.col2 = -1; x.num[0] = lo;
                             x.den[0] = c->den[0]; x.den[1] = 1; }
        else ok = 0;
    } else if (c->op >= TR_BOOL_COL_EQ) {
        static const int32_t neg[7] = {0, TR_BOOL_NE, TR_BOOL_EQ, TR_BOOL_GE, TR_BOOL_GT, TR_BOOL_LE, TR_BOOL_LT};
        x.op = neg[c->op - TR_BOOL_COL_EQ + TR_BOOL_EQ] - TR_BOOL_EQ + TR_BOOL_COL_EQ;
    } else ok = 0;
    int32_t j = ok ? intern_condition(p, &x) : -1;
    p->comp[i] = j;
    if (j >= 0 && p->comp[j] == -2) p->comp[j] = i;
    return j;
}

/* ------------------------------------------------------------------ the canonical DAG (hash-consed) */
static uint32_t node_hash(int32_t op, int32_t a, int32_t b) {
    uint64_t h = (uint64_t)(uint32_t)op * UINT64_C(0x9E3779B97F4A7C15) ^ (uint64_t)(uint32_t)a * UINT64_C(0xC2B2AE3D27D4EB4F) ^
                 (uint64_t)(uint32_t)b * UINT64_C(0x165667B19E3779F9);
    return (uint32_t)(h ^ (h >> 29));
}
static int grow_table(tr_prog *p) {
    uint32_t cap = p->hcap ? p->hcap * 2 : 1024;
    uint32_t *t = zalloc((uint64_t)cap * 4);
    if (!t) return 0;
    for (int32_t i = 0; i < p->nnodes; i++) {
        const tr_bool_node *n = &p->nodes[i];
        uint32_t h = node_hash(n->op, n->a, n->b);
        for (uint32_t k = 0;; k++) { uint32_t s = (h + k) & (cap - 1); if (!t[s]) { t[s] = (uint32_t)i + 1; break; } }
    }
    free(p->htab);
    p->htab = t; p->hcap = cap;
    return 1;
}
/* the node (op, a, b), created when absent; -1 on memory or representation failure */
static int32_t intern(tr_prog *p, int32_t op, int32_t a, int32_t b) {
    if (p->hcap == 0 || (uint64_t)(p->nnodes + 1) * 2 > p->hcap) if (!grow_table(p)) return -1;
    uint32_t h = node_hash(op, a, b);
    for (uint32_t k = 0;; k++) {
        uint32_t s = (h + k) & (p->hcap - 1), e = p->htab[s];
        if (!e) {
            if (p->nnodes >= TR_REDUCE_MAX_NODES) return -1;
            if (p->nnodes == p->capnodes) {
                int32_t cap = p->capnodes ? p->capnodes * 2 : 256;
                tr_bool_node *nn = realloc(p->nodes, (size_t)cap * sizeof *nn);
                if (!nn) return -1;
                p->nodes = nn;
                uint64_t *nd = realloc(p->deps, (size_t)cap * 8);
                if (!nd) return -1;
                p->deps = nd;
                int32_t *ng = realloc(p->neg, (size_t)cap * 4);
                if (!ng) return -1;
                p->neg = ng; p->capnodes = cap;
            }
            int32_t i = p->nnodes++;
            tr_bool_node *n = &p->nodes[i];
            n->op = op; n->a = a; n->b = b; n->reserved = 0;
            uint64_t d = 0;
            if (op == TR_BOOL_VAR) d = UINT64_C(1) << a;
            else if (op == TR_BOOL_NOT) d = p->deps[a];
            else if (op >= TR_BOOL_AND) d = p->deps[a] | p->deps[b];
            p->deps[i] = d;
            p->neg[i] = -1;
            p->htab[s] = (uint32_t)i + 1;
            return i;
        }
        const tr_bool_node *n = &p->nodes[e - 1];
        if (n->op == op && n->a == a && n->b == b) return (int32_t)e - 1;
    }
}
static int32_t mk_const(tr_prog *p, int32_t t) { return intern(p, TR_BOOL_CONST, t, -1); }
static int is_const(const tr_prog *p, int32_t x, int32_t t) {
    return p->nodes[x].op == TR_BOOL_CONST && p->nodes[x].a == t;
}
static int cmp_i32(const void *a, const void *b) { int32_t x = *(const int32_t *)a, y = *(const int32_t *)b; return (x > y) - (x < y); }
/* the leaves of an op-chain under x (pushed onto leaves[]) */
typedef struct { int32_t *v; int32_t n, cap; } ivec;
static int push(ivec *s, int32_t x) {
    if (s->n == s->cap) {
        int32_t cap = s->cap ? s->cap * 2 : 16;
        int32_t *nv = realloc(s->v, (size_t)cap * 4);
        if (!nv) return 0;
        s->v = nv; s->cap = cap;
    }
    s->v[s->n++] = x;
    return 1;
}
static int flatten(const tr_prog *p, int32_t op, int32_t x, ivec *leaves, ivec *stack, int *parity) {
    stack->n = 0;
    if (!push(stack, x)) return 0;
    while (stack->n) {
        int32_t y = stack->v[--stack->n];
        const tr_bool_node *n = &p->nodes[y];
        if (n->op == op) { if (!push(stack, n->b) || !push(stack, n->a)) return 0; continue; }
        if (op == TR_BOOL_XOR && n->op == TR_BOOL_NOT && p->nodes[n->a].op == TR_BOOL_XOR) {
            *parity ^= 1;
            if (!push(stack, n->a)) return 0;
            continue;
        }
        if (!push(leaves, y)) return 0;
    }
    return 1;
}
static int32_t mk_not(tr_prog *p, int32_t x);
/* AND / OR of the operands xs[0..n): flattened, constants folded, idempotent, sorted, left-deep */
static int32_t mk_assoc(tr_prog *p, int32_t op, const int32_t *xs, int32_t n) {
    ivec leaves = {0}, stack = {0};
    int parity = 0;
    int32_t r = -1;
    for (int32_t i = 0; i < n; i++) if (!flatten(p, op, xs[i], &leaves, &stack, &parity)) goto done;
    {
        const int32_t absorb = op == TR_BOOL_AND ? K_F : K_T, unit = op == TR_BOOL_AND ? K_T : K_F;
        int32_t m = 0;
        for (int32_t i = 0; i < leaves.n; i++) {
            if (is_const(p, leaves.v[i], absorb)) { r = mk_const(p, absorb); goto done; }
            if (!is_const(p, leaves.v[i], unit)) leaves.v[m++] = leaves.v[i];
        }
        qsort(leaves.v, (size_t)m, 4, cmp_i32);
        int32_t u = 0;
        for (int32_t i = 0; i < m; i++) if (!u || leaves.v[u - 1] != leaves.v[i]) leaves.v[u++] = leaves.v[i];
        if (!u) { r = mk_const(p, unit); goto done; }
        r = leaves.v[0];
        for (int32_t i = 1; i < u && r >= 0; i++) r = intern(p, op, r, leaves.v[i]);
    }
done:
    free(leaves.v); free(stack.v);
    return r;
}
/* XOR of xs[0..n): flattened with multiplicity, FALSE dropped, TRUE negates, UNKNOWN absorbs; NOT above the chain */
static int32_t mk_xor(tr_prog *p, const int32_t *xs, int32_t n) {
    ivec leaves = {0}, stack = {0};
    int parity = 0;
    int32_t r = -1;
    for (int32_t i = 0; i < n; i++) if (!flatten(p, TR_BOOL_XOR, xs[i], &leaves, &stack, &parity)) goto done;
    {
        int32_t m = 0;
        for (int32_t i = 0; i < leaves.n; i++) {
            int32_t y = leaves.v[i];
            if (is_const(p, y, K_U)) { r = mk_const(p, K_U); goto done; }
            if (is_const(p, y, K_T)) { parity ^= 1; continue; }
            if (is_const(p, y, K_F)) continue;
            leaves.v[m++] = y;                     /* a negated leaf stays a leaf */
        }
        qsort(leaves.v, (size_t)m, 4, cmp_i32);
        if (!m) { r = mk_const(p, parity ? K_T : K_F); goto done; }
        r = leaves.v[0];
        for (int32_t i = 1; i < m && r >= 0; i++) r = intern(p, TR_BOOL_XOR, r, leaves.v[i]);
        if (r >= 0 && parity) r = m == 1 ? mk_not(p, r) : intern(p, TR_BOOL_NOT, r, -1);
    }
done:
    free(leaves.v); free(stack.v);
    return r;
}
/* NOT x, pushed to the leaves (De Morgan), folded into complementary conditions where they exist; each node's
 * negation is computed once (a shared subexpression is negated once, so the work stays linear in the DAG) */
static int32_t mk_not_new(tr_prog *p, int32_t x);
static int32_t mk_not(tr_prog *p, int32_t x) {
    if (p->neg[x] >= 0) return p->neg[x];
    int32_t r = mk_not_new(p, x);
    if (r >= 0) p->neg[x] = r;
    return r;
}
static int32_t mk_not_new(tr_prog *p, int32_t x) {
    const tr_bool_node n = p->nodes[x];
    switch (n.op) {
    case TR_BOOL_CONST: return mk_const(p, n.a == K_U ? K_U : (n.a == K_T ? K_F : K_T));
    case TR_BOOL_NOT: return n.a;
    case TR_BOOL_VAR: {
        int32_t c = complement(p, n.a);
        return c >= 0 ? intern(p, TR_BOOL_VAR, c, -1) : intern(p, TR_BOOL_NOT, x, -1);
    }
    case TR_BOOL_AND: case TR_BOOL_OR: {
        ivec leaves = {0}, stack = {0};
        int parity = 0;
        int32_t r = -1;
        if (flatten(p, n.op, x, &leaves, &stack, &parity)) {
            int ok = 1;
            for (int32_t i = 0; i < leaves.n && ok; i++) ok = (leaves.v[i] = mk_not(p, leaves.v[i])) >= 0;
            if (ok) r = mk_assoc(p, n.op == TR_BOOL_AND ? TR_BOOL_OR : TR_BOOL_AND, leaves.v, leaves.n);
        }
        free(leaves.v); free(stack.v);
        return r;
    }
    default: return intern(p, TR_BOOL_NOT, x, -1);        /* NOT above an XOR chain */
    }
}

/* ------------------------------------------------------------------ the universal search */
static int k_not(int a) { return a == K_U ? K_U : !a; }
static int k_and(int a, int b) { return (a == K_F || b == K_F) ? K_F : (a == K_U || b == K_U) ? K_U : K_T; }
static int k_or(int a, int b) { return (a == K_T || b == K_T) ? K_T : (a == K_U || b == K_U) ? K_U : K_F; }
static int k_xor(int a, int b) { return (a == K_U || b == K_U) ? K_U : (a != b); }
/* the nodes the roots reach, increasing (marks) */
static void cone(const tr_prog *p, int32_t ra, int32_t rb, uint8_t *mark, int32_t *list, int32_t *n) {
    memset(mark, 0, (size_t)p->nnodes);
    if (ra >= 0) mark[ra] = 1;
    if (rb >= 0) mark[rb] = 1;
    int32_t top = ra > rb ? ra : rb;
    for (int32_t i = top; i >= 0; i--) {
        if (!mark[i]) continue;
        const tr_bool_node *x = &p->nodes[i];
        if (x->op == TR_BOOL_NOT || x->op >= TR_BOOL_AND) mark[x->a] = 1;
        if (x->op >= TR_BOOL_AND) mark[x->b] = 1;
    }
    *n = 0;
    for (int32_t i = 0; i <= top; i++) if (mark[i]) list[(*n)++] = i;
}
static int eval_cone(const tr_prog *p, const int32_t *list, int32_t n, const int *val, int8_t *nv) {
    for (int32_t k = 0; k < n; k++) {
        int32_t i = list[k];
        const tr_bool_node *x = &p->nodes[i];
        int r;
        switch (x->op) {
        case TR_BOOL_CONST: r = x->a; break;
        case TR_BOOL_VAR: r = val[x->a]; break;
        case TR_BOOL_NOT: r = k_not(nv[x->a]); break;
        case TR_BOOL_AND: r = k_and(nv[x->a], nv[x->b]); break;
        case TR_BOOL_OR: r = k_or(nv[x->a], nv[x->b]); break;
        default: r = k_xor(nv[x->a], nv[x->b]); break;
        }
        nv[i] = (int8_t)r;
    }
    return 0;
}
typedef struct { int64_t a, b; } span;
static int cmp_span(const void *x, const void *y) {
    const span *s = x, *t = y;
    return (s->a > t->a) - (s->a < t->a);
}
/* is there a row giving the support's conditions the values val? 1 (and the witness), 0, or -1 (undecided) */
static int feasible(const tr_prog *p, uint64_t support, const int *val, tr_prog_compare_result *w) {
    uint64_t nullm = 0, setm = 0;
    int undecided = 0;
    for (int32_t col = 0; col < p->ncols; col++) {
        int need_null = 0, need_present = 0;
        __int128 lo = INT64_MIN, hi = INT64_MAX;
        span ex[2 * NCOND];
        int nex = 0, touched = 0;
        for (int32_t c = 0; c < p->nconds; c++) {
            if (!((support >> c) & 1)) continue;
            const tr_bool_condition *k = &p->conds[c];
            int on = k->col == col || (k->op >= TR_BOOL_COL_EQ && k->col2 == col);
            if (!on) continue;
            touched = 1;
            int v = val[c];
            if ((k->op == TR_BOOL_IS_NULL || k->op == TR_BOOL_IS_NOT_NULL) && v == K_U) return 0;   /* never UNKNOWN */
            if (k->op == TR_BOOL_IS_NULL) { if (v == K_T) need_null = 1; else need_present = 1; continue; }
            if (k->op == TR_BOOL_IS_NOT_NULL) { if (v == K_T) need_present = 1; else need_null = 1; continue; }
            if (k->op >= TR_BOOL_COL_EQ) { if (v != K_U) { need_present = 1; undecided = 1; } else undecided = 1; continue; }
            if (v == K_U) { need_null = 1; continue; }
            need_present = 1;
            if (k->op == TR_BOOL_BIT) { undecided = 1; continue; }
            if (k->op == TR_BOOL_BETWEEN) {
                if (v == K_T) { if (k->num[0] > lo) lo = k->num[0]; if (k->num[1] < hi) hi = k->num[1]; }
                else if (k->num[0] <= k->num[1]) { ex[nex].a = k->num[0]; ex[nex].b = k->num[1]; nex++; }
            } else if (k->op == TR_BOOL_NE) {
                if (v == K_T) { ex[nex].a = k->num[0]; ex[nex].b = k->num[0]; nex++; }
                else { if (k->num[0] > lo) lo = k->num[0]; if (k->num[0] < hi) hi = k->num[0]; }
            } else undecided = 1;
        }
        if (!touched) continue;
        if (need_null && need_present) return 0;
        if (need_null) {
            if (!((p->nullable >> col) & 1)) return 0;
            nullm |= UINT64_C(1) << col;
            continue;
        }
        if (lo > hi) return 0;
        qsort(ex, (size_t)nex, sizeof ex[0], cmp_span);
        __int128 at = lo;
        for (int i = 0; i < nex && at <= hi; i++) {
            if ((__int128)ex[i].a > at) break;
            if ((__int128)ex[i].b + 1 > at) at = (__int128)ex[i].b + 1;
        }
        if (at > hi) return 0;
        setm |= UINT64_C(1) << col;
        if (w) w->value[col] = (int64_t)at;
    }
    if (undecided) return -1;
    if (w) { w->null_mask = nullm; w->set_mask = setm; }
    return 1;
}
static uint64_t pow3(int k) { uint64_t r = 1; for (int i = 0; i < k; i++) r = tr_budget_mul(r, 3); return r; }
/* the universal outcome for two canonical predicate nodes (-1: no predicate, every row passes) */
static int32_t search(const tr_prog *p, int32_t fa, int32_t fb, uint64_t limit, tr_prog_compare_result *r) {
    r->iterations = 0;
    if (fa == fb) { r->outcome = TR_PROG_EQUIVALENT; r->how = TR_PROG_BY_CANONICAL; return TR_MATH_OK; }
    uint64_t support = (fa >= 0 ? p->deps[fa] : 0) | (fb >= 0 ? p->deps[fb] : 0);
    r->support = support;
    int32_t idx[NCOND], k = 0;
    for (int32_t c = 0; c < p->nconds; c++) if ((support >> c) & 1) idx[k++] = c;
    uint64_t total = pow3(k);
    if (total > limit) { r->outcome = TR_PROG_BUDGET; r->how = 0; return TR_MATH_OK; }
    uint8_t *mark = zalloc((uint64_t)p->nnodes);
    int32_t *list = zalloc((uint64_t)p->nnodes * 4);
    int8_t *nv = zalloc((uint64_t)p->nnodes);
    if (!mark || !list || !nv) { free(mark); free(list); free(nv); return TR_MATH_MEMORY; }
    int32_t n = 0;
    cone(p, fa, fb, mark, list, &n);
    int val[NCOND];
    for (int32_t c = 0; c < NCOND; c++) val[c] = K_F;
    int unknown = 0;
    r->outcome = TR_PROG_EQUIVALENT; r->how = TR_PROG_BY_SEARCH;
    for (uint64_t a = 0; a < total; a++) {
        uint64_t x = a;
        for (int32_t i = 0; i < k; i++) { val[idx[i]] = (int)(x % 3); x /= 3; }
        eval_cone(p, list, n, val, nv);
        int ta = fa < 0 ? 1 : nv[fa] == K_T, tb = fb < 0 ? 1 : nv[fb] == K_T;
        r->iterations++;
        if (ta == tb) continue;
        tr_prog_compare_result w;
        memset(&w, 0, sizeof w);
        int f = feasible(p, support, val, &w);
        if (f == 0) continue;
        if (f < 0) { unknown = 1; continue; }
        r->outcome = TR_PROG_DIFFERENT; r->how = TR_PROG_BY_WITNESS;
        r->accepts = ta ? 1 : 2;
        r->assign_true = r->assign_unknown = 0;
        for (int32_t i = 0; i < k; i++) {
            if (val[idx[i]] == K_T) r->assign_true |= UINT64_C(1) << idx[i];
            if (val[idx[i]] == K_U) r->assign_unknown |= UINT64_C(1) << idx[i];
        }
        r->null_mask = w.null_mask; r->set_mask = w.set_mask;
        memcpy(r->value, w.value, sizeof r->value);
        break;
    }
    if (r->outcome == TR_PROG_EQUIVALENT && unknown) { r->outcome = TR_PROG_UNKNOWN; r->how = 0; }
    free(mark); free(list); free(nv);
    return TR_MATH_OK;
}

/* ------------------------------------------------------------------ measures and outputs */
static int texp_cmp(const uint8_t *a, const uint8_t *b, int32_t n) { return memcmp(a, b, (size_t)n); }
/* the index of monomial e in output o's basis, appended when absent (-1: memory, -12: too many) */
static int32_t basis_term(tr_prog *p, pout *o, const uint8_t *e) {
    const int32_t nc = p->ncols;
    for (int32_t j = 0; j < o->nterms; j++) if (!nc || !texp_cmp(o->exps + (uint64_t)j * (uint64_t)nc, e, nc)) return j;
    if (o->nterms >= TR_BANK_MAX_TERMS) return TR_MATH_UNSUPPORTED;
    if (o->nterms == o->cap) {
        int32_t cap = o->cap ? o->cap * 2 : 8;
        uint8_t *x = realloc(o->exps, (size_t)cap * (size_t)(nc ? nc : 1));
        if (!x) return TR_MATH_MEMORY;
        o->exps = x; o->cap = cap;
    }
    if (nc) memcpy(o->exps + (uint64_t)o->nterms * (uint64_t)nc, e, (size_t)nc);
    return o->nterms++;
}
static int32_t find_output(tr_prog *p, int32_t kind, int32_t filter, int32_t grouping, int32_t mode, uint32_t colset,
                           const tr_dist_spec *spec, int32_t measure) {
    for (int32_t o = 0; o < p->nouts; o++) {
        const pout *x = &p->outs[o];
        if (x->kind != kind || x->filter != filter || x->grouping != grouping) continue;
        if (kind == TR_REDUCE_BANK && x->mode == mode && (mode == TR_BANK_SQL || x->colset == colset)) return o;
        if (kind == TR_REDUCE_HIST && !memcmp(&p->cm[x->hist_measure].spec, spec, sizeof *spec)) return o;
    }
    if (p->nouts >= TR_REDUCE_MAX_OUTPUTS) return TR_MATH_UNSUPPORTED;
    pout *x = &p->outs[p->nouts];
    memset(x, 0, sizeof *x);
    x->kind = kind; x->filter = filter; x->grouping = grouping; x->mode = mode; x->colset = colset;
    x->hist_measure = measure;
    return p->nouts++;
}

tr_prog *tr_prog_compile(const tr_scope *scope, const tr_prog_family *fam, const tr_math_budget *budget,
                         tr_math_refusal *refusal, int32_t *failed, int32_t *error) {
    tr_budget_none(refusal);
    if (failed) *failed = -1;
    tr_scope_view sv;
    if (!scope || !fam || tr_scope_get_view(scope, &sv) || fam->nconds < 0 || fam->nnodes < 0 || fam->ngroupings < 1 ||
        fam->nqueries < 0 || fam->nmeasures < 0 || fam->nterms < 0 || (fam->flags & ~TR_PROG_PROVE) ||
        (fam->lifetime != TR_PROG_LIFETIME_RUN && fam->lifetime != TR_PROG_LIFETIME_SNAPSHOT) ||
        (fam->nconds && !fam->conds) || (fam->nnodes && !fam->nodes) || !fam->groupings || (fam->nqueries && !fam->queries) ||
        (fam->nmeasures && !fam->measures) || (fam->nterms && (!fam->coefs || (sv.ncols && !fam->exps)))) {
        set_error(error, TR_MATH_INVALID); return NULL;
    }
    if (fam->nqueries > TR_PROG_MAX_QUERIES || fam->nmeasures > TR_PROG_MAX_MEASURES || fam->nterms > TR_PROG_MAX_TERMS ||
        fam->nconds > NCOND || fam->nnodes > TR_REDUCE_MAX_NODES || fam->ngroupings > TR_REDUCE_MAX_GROUPINGS) {
        set_error(error, TR_MATH_UNSUPPORTED); return NULL;
    }
    tr_math_budget dflt;
    budget = budget_or(budget, &dflt);
    tr_math_need pre;
    need_zero(&pre);
    pre.need[TR_BUDGET_CIRCUIT] = (uint64_t)fam->nconds + (uint64_t)fam->nnodes;
    pre.need[TR_BUDGET_OUTPUT] = (uint64_t)fam->nqueries;
    pre.need[TR_BUDGET_SUPPORT] = (uint64_t)fam->nterms;
    int32_t rc = tr_budget_check(budget, &pre, refusal);
    if (rc) { set_error(error, rc); return NULL; }
    tr_prog *p = zalloc(sizeof *p);
    if (!p) { set_error(error, TR_MATH_MEMORY); return NULL; }
    p->magic = PROG_MAGIC;
    p->scope = scope; p->sv = sv; p->ncols = sv.ncols; p->nqueries = fam->nqueries; p->nmeasures = fam->nmeasures;
    p->flags = fam->flags; p->lifetime = fam->lifetime; p->conds_in = fam->nconds; p->nodes_in = fam->nnodes;
    for (int32_t c = 0; c < sv.ncols; c++) {             /* the declared universe's NULLs (math_program.h) */
        tr_scope_column_info ci;
        if (!tr_scope_column_get(scope, c, &ci) && (ci.valid >= 0 || (ci.col < 0 && sv.col[c].valid)))
            p->nullable |= UINT32_C(1) << c;
    }
    int32_t *cmap = zalloc((uint64_t)fam->nconds * 4), *nmap = zalloc((uint64_t)fam->nnodes * 4);
    p->qfilter = zalloc((uint64_t)fam->nqueries * 4); p->qclass = zalloc((uint64_t)fam->nqueries * 4);
    p->qproof = zalloc((uint64_t)fam->nqueries * 4);
    p->queries = zalloc((uint64_t)fam->nqueries * sizeof(tr_prog_query));
    p->cm = zalloc((uint64_t)fam->nmeasures * sizeof(cmeasure));
    p->texps = zalloc((uint64_t)fam->nterms * (uint64_t)sv.ncols); p->tcoef = zalloc((uint64_t)fam->nterms * 8);
    p->outs = zalloc((uint64_t)(fam->nmeasures + fam->nqueries + 1) * sizeof(pout));
    if (!cmap || !nmap || !p->qfilter || !p->qclass || !p->qproof || !p->queries || !p->cm || !p->texps || !p->tcoef ||
        !p->outs) { rc = TR_MATH_MEMORY; goto fail; }
    /* 1. canonical conditions */
    for (int32_t i = 0; i < fam->nconds; i++) {
        if (tr_bool_condition_check(&sv, &fam->conds[i])) { rc = TR_MATH_INVALID; goto fail; }
        tr_bool_condition c;
        canon_condition(&sv, &fam->conds[i], &c);
        if ((cmap[i] = intern_condition(p, &c)) < 0) { rc = TR_MATH_UNSUPPORTED; goto fail; }
    }
    /* 2. canonical predicates */
    for (int32_t i = 0; i < fam->nnodes; i++) {
        const tr_bool_node *n = &fam->nodes[i];
        int32_t r = -1;
        if (n->reserved) { rc = TR_MATH_INVALID; goto fail; }
        switch (n->op) {
        case TR_BOOL_CONST: if (n->a < 0 || n->a > 2 || n->b != -1) { rc = TR_MATH_INVALID; goto fail; } r = mk_const(p, n->a); break;
        case TR_BOOL_VAR:
            if (n->a < 0 || n->a >= fam->nconds || n->b != -1) { rc = TR_MATH_INVALID; goto fail; }
            r = intern(p, TR_BOOL_VAR, cmap[n->a], -1); break;
        case TR_BOOL_NOT:
            if (n->a < 0 || n->a >= i || n->b != -1) { rc = TR_MATH_INVALID; goto fail; }
            r = mk_not(p, nmap[n->a]); break;
        case TR_BOOL_AND: case TR_BOOL_OR: case TR_BOOL_XOR: {
            if (n->a < 0 || n->a >= i || n->b < 0 || n->b >= i) { rc = TR_MATH_INVALID; goto fail; }
            int32_t xs[2] = {nmap[n->a], nmap[n->b]};
            r = n->op == TR_BOOL_XOR ? mk_xor(p, xs, 2) : mk_assoc(p, n->op, xs, 2);
            break;
        }
        default: rc = TR_MATH_INVALID; goto fail;
        }
        if (r < 0) { rc = p->nnodes >= TR_REDUCE_MAX_NODES ? TR_MATH_UNSUPPORTED : TR_MATH_MEMORY; goto fail; }
        nmap[i] = r;
    }
    {
        tr_math_need cn;
        need_zero(&cn);
        cn.need[TR_BUDGET_CIRCUIT] = (uint64_t)p->nconds + (uint64_t)p->nnodes;
        if ((rc = tr_budget_check(budget, &cn, refusal))) goto fail;
    }
    /* queries and their canonical measures */
    int32_t at = 0;
    for (int32_t q = 0; q < fam->nqueries; q++) {
        const tr_prog_query *x = &fam->queries[q];
        if (x->filter < -1 || x->filter >= fam->nnodes || x->grouping < 0 || x->grouping >= fam->ngroupings ||
            x->first < 0 || x->nmeasures < 0 || x->first > fam->nmeasures || x->nmeasures > fam->nmeasures - x->first ||
            (x->scalar != 0 && x->scalar != 1) || x->reserved[0] || x->reserved[1] || x->reserved[2]) {
            if (failed) *failed = q;
            rc = TR_MATH_INVALID; goto fail;
        }
        p->queries[q] = *x;
        p->qfilter[q] = x->filter < 0 ? -1 : nmap[x->filter];
        if (p->qfilter[q] >= 0 && is_const(p, p->qfilter[q], K_T)) p->qfilter[q] = -1;   /* WHERE TRUE: every row */
        p->qclass[q] = q; p->qproof[q] = 0;
        for (int32_t k = 0; k < x->nmeasures; k++) {
            const tr_prog_measure *m = &fam->measures[x->first + k];
            cmeasure *c = &p->cm[x->first + k];
            c->kind = m->kind; c->query = q; c->output = -1; c->count_term = -1;
            if (m->reserved || m->kind < TR_PROG_COUNT_STAR || m->kind > TR_PROG_HIST) { rc = TR_MATH_INVALID; break; }
            if (m->kind == TR_PROG_HIST) {
                if (m->first || m->nterms) { rc = TR_MATH_INVALID; break; }
                tr_dist_spec_normalize(&m->hist, &c->spec);
                c->spec.strategy = TR_DIST_STRATEGY_AUTO;
                continue;
            }
            if (m->kind == TR_PROG_COUNT_STAR) { if (m->first || m->nterms) rc = TR_MATH_INVALID; continue; }
            if (m->first < 0 || m->nterms < 0 || m->first > fam->nterms || m->nterms > fam->nterms - m->first) {
                rc = TR_MATH_INVALID; break;
            }
            if (m->nterms > TR_PROG_MAX_MEASURE_TERMS) { rc = TR_MATH_UNSUPPORTED; break; }
            /* the canonical polynomial: monomials sorted, equal ones merged, zero coefficients dropped; its validity
             * stays that of the written expression (x - x is NULL where x is) */
            const int32_t nc = sv.ncols, base = at;
            uint32_t written = 0;
            for (int32_t t = 0; t < m->nterms; t++) {
                const uint8_t *e = fam->exps ? fam->exps + (uint64_t)(m->first + t) * (uint64_t)nc : NULL;
                int deg = 0;
                for (int32_t col = 0; col < nc; col++) { deg += e[col]; if (e[col]) written |= UINT32_C(1) << col; }
                if (deg > TR_BANK_MAX_DEGREE) { rc = TR_MATH_UNSUPPORTED; break; }
                if (nc) memcpy(p->texps + (uint64_t)at * (uint64_t)nc, e, (size_t)nc);
                p->tcoef[at] = fam->coefs[m->first + t];
                at++;
            }
            if (rc) break;
            /* sort the (monomial, coefficient) pairs by monomial: a small insertion sort keeps them together */
            for (int32_t i = base + 1; i < at; i++)
                for (int32_t j = i; j > base && nc && memcmp(p->texps + (uint64_t)(j - 1) * nc, p->texps + (uint64_t)j * nc, (size_t)nc) > 0; j--) {
                    uint8_t tmp[TR_SCOPE_MAX_COLUMNS];
                    memcpy(tmp, p->texps + (uint64_t)(j - 1) * nc, (size_t)nc);
                    memcpy(p->texps + (uint64_t)(j - 1) * nc, p->texps + (uint64_t)j * nc, (size_t)nc);
                    memcpy(p->texps + (uint64_t)j * nc, tmp, (size_t)nc);
                    int64_t tc = p->tcoef[j - 1]; p->tcoef[j - 1] = p->tcoef[j]; p->tcoef[j] = tc;
                }
            /* equal monomials: their coefficients added exactly (at most 1024 int64 terms fit 128 bits), the sum must
             * fit int64 whatever the order of the terms */
            int32_t w = base;
            for (int32_t i = base; i < at;) {
                __int128 acc = p->tcoef[i];
                int32_t j = i + 1;
                while (j < at && (!nc || !memcmp(p->texps + (uint64_t)i * nc, p->texps + (uint64_t)j * nc, (size_t)nc)))
                    acc += p->tcoef[j++];
                if (acc < (__int128)INT64_MIN || acc > (__int128)INT64_MAX) { rc = TR_MATH_OVERFLOW; break; }
                if (nc && w != i) memcpy(p->texps + (uint64_t)w * nc, p->texps + (uint64_t)i * nc, (size_t)nc);
                p->tcoef[w++] = (int64_t)acc;
                i = j;
            }
            if (rc) break;
            int32_t z = base;
            for (int32_t i = base; i < w; i++) {
                if (!p->tcoef[i]) continue;
                if (nc && z != i) memcpy(p->texps + (uint64_t)z * nc, p->texps + (uint64_t)i * nc, (size_t)nc);
                p->tcoef[z++] = p->tcoef[i];
            }
            at = z;
            c->first = base; c->nterms = z - base;
            c->colset = written;
            c->homogeneous = !written || z > base;
            for (int32_t t = base; t < z && written; t++) {
                uint32_t s = 0;
                for (int32_t col = 0; col < nc; col++) if (p->texps[(uint64_t)t * nc + col]) s |= UINT32_C(1) << col;
                if (s != written) c->homogeneous = 0;
            }
        }
        if (rc) { if (failed) *failed = q; goto fail; }
    }
    p->ntterms = at;
    /* 3. the bounded universal search merges equivalent predicates of equal support */
    {
        uint64_t limit = budget->limit[TR_BUDGET_ITERATIONS], used = 0;
        for (int32_t q = 0; q < fam->nqueries; q++) {
            for (int32_t r = 0; r < q; r++) {
                if (p->qclass[r] != r) continue;
                if (p->qfilter[r] == p->qfilter[q]) {
                    p->qclass[q] = r; p->qproof[q] = TR_PROG_BY_CANONICAL; p->info.merged_canonical++;
                    break;
                }
            }
            if (p->qclass[q] != q || !(p->flags & TR_PROG_PROVE)) continue;
            for (int32_t r = 0; r < q; r++) {
                if (p->qclass[r] != r || p->qfilter[r] < 0 || p->qfilter[q] < 0 ||
                    p->deps[p->qfilter[r]] != p->deps[p->qfilter[q]]) continue;
                tr_prog_compare_result res;
                memset(&res, 0, sizeof res);
                rc = search(p, p->qfilter[r], p->qfilter[q], limit > used ? limit - used : 0, &res);
                if (rc) goto fail;
                used = tr_budget_add(used, res.iterations);
                if (res.outcome == TR_PROG_EQUIVALENT) {
                    p->qclass[q] = r; p->qproof[q] = TR_PROG_BY_SEARCH; p->info.merged_search++;
                    break;
                }
                if (res.outcome == TR_PROG_UNKNOWN) p->info.search_unknown++;
                if (res.outcome == TR_PROG_BUDGET) p->info.search_budget++;
            }
        }
        p->info.search_iterations = used;
    }
    /* 4. the states: one output per (class, grouping[, column set | spec]) */
    for (int32_t q = 0; q < fam->nqueries && !rc; q++) {
        const tr_prog_query *x = &p->queries[q];
        const int32_t f = p->qfilter[p->qclass[q]], g = x->grouping;
        for (int32_t k = 0; k < x->nmeasures && !rc; k++) {
            cmeasure *c = &p->cm[x->first + k];
            int32_t o;
            if (c->kind == TR_PROG_HIST) o = find_output(p, TR_REDUCE_HIST, f, g, 0, 0, &c->spec, x->first + k);
            else if (c->kind == TR_PROG_COUNT_STAR || c->homogeneous) o = find_output(p, TR_REDUCE_BANK, f, g, TR_BANK_SQL, 0, NULL, -1);
            else o = find_output(p, TR_REDUCE_BANK, f, g, TR_BANK_JOINT, c->colset, NULL, -1);
            if (o < 0) { rc = o; break; }
            c->output = o;
            if (c->kind == TR_PROG_HIST || c->kind == TR_PROG_COUNT_STAR) continue;
            uint32_t covered = 0;
            for (int32_t t = 0; t < c->nterms; t++) {
                const uint8_t *e = p->texps + (uint64_t)(c->first + t) * (uint64_t)sv.ncols;
                int32_t j = basis_term(p, &p->outs[o], e);
                if (j < 0) { rc = j; break; }
                if (t == 0) c->count_term = j;
                for (int32_t col = 0; col < sv.ncols; col++) if (e[col]) covered |= UINT32_C(1) << col;
            }
            /* a listwise state over the written columns: a degree-1 term for each column no monomial of the measure
             * reads (a column whose terms cancelled), so that the basis's columns are exactly the written ones */
            for (int32_t col = 0; !rc && !c->homogeneous && col < sv.ncols; col++) {
                if (!((c->colset & ~covered) >> col & 1)) continue;
                uint8_t e[TR_SCOPE_MAX_COLUMNS];
                memset(e, 0, sizeof e);
                e[col] = 1;
                int32_t j = basis_term(p, &p->outs[o], e);
                if (j < 0) rc = j;
                else if (c->count_term < 0) c->count_term = j;
            }
        }
        if (rc && failed) *failed = q;
    }
    if (rc) goto fail;
    /* the FND-006 request and plan */
    {
        int32_t total = 0;
        for (int32_t o = 0; o < p->nouts; o++) total += p->outs[o].nterms;
        tr_reduce_output *ro = zalloc((uint64_t)p->nouts * sizeof *ro);
        uint8_t *ex = zalloc((uint64_t)total * (uint64_t)sv.ncols);
        if (!ro || !ex) { free(ro); free(ex); rc = TR_MATH_MEMORY; goto fail; }
        int32_t first = 0;
        for (int32_t o = 0; o < p->nouts; o++) {
            const pout *x = &p->outs[o];
            ro[o].kind = x->kind; ro[o].filter = x->filter; ro[o].grouping = x->grouping;
            if (x->kind == TR_REDUCE_BANK) {
                ro[o].mode = x->mode; ro[o].first = first; ro[o].nterms = x->nterms;
                if (x->nterms && sv.ncols) memcpy(ex + (uint64_t)first * sv.ncols, x->exps, (size_t)x->nterms * (size_t)sv.ncols);
                first += x->nterms;
            } else ro[o].hist = p->cm[x->hist_measure].spec;
        }
        tr_reduce_request rq;
        memset(&rq, 0, sizeof rq);
        rq.nconds = p->nconds; rq.nnodes = p->nnodes; rq.ngroupings = fam->ngroupings; rq.noutputs = p->nouts;
        rq.nexps = total; rq.conds = p->conds; rq.nodes = p->nodes; rq.groupings = fam->groupings; rq.outputs = ro;
        rq.exps = ex;
        int32_t of = -1;
        p->plan = tr_reduce_compile(scope, &rq, budget, refusal, &of, &rc);
        free(ro); free(ex);
        if (!p->plan) {
            if (failed && of >= 0)
                for (int32_t m = 0; m < fam->nmeasures; m++) if (p->cm[m].output == of) { *failed = p->cm[m].query; break; }
            goto fail;
        }
    }
    /* 5. the finalization tables, query and measure descriptions */
    {
        tr_reduce_view rv;
        tr_reduce_get_view(p->plan, &rv);
        p->fm = zalloc((uint64_t)fam->nmeasures * sizeof(tr_prog_fmeasure));
        p->ft = zalloc((uint64_t)p->ntterms * sizeof(tr_prog_fterm));
        p->qinfo = zalloc((uint64_t)fam->nqueries * sizeof(tr_prog_query_info));
        p->minfo = zalloc((uint64_t)fam->nmeasures * sizeof(tr_prog_measure_info));
        if (!p->fm || !p->ft || !p->qinfo || !p->minfo) { rc = TR_MATH_MEMORY; goto fail; }
        uint64_t slot = 0;
        for (int32_t m = 0; m < fam->nmeasures; m++) {
            cmeasure *c = &p->cm[m];
            tr_prog_measure_info *mi = &p->minfo[m];
            mi->kind = c->kind; mi->query = c->query; mi->output = c->output; mi->nterms = c->nterms;
            mi->count_term = c->count_term < 0 ? UINT32_MAX : (uint32_t)c->count_term;
            const tr_prog_query *x = &p->queries[c->query];
            uint64_t G = rv.groups[x->grouping].ngroups;
            if (x->scalar && G != 1) { if (failed) *failed = c->query; rc = TR_MATH_INVALID; goto fail; }
            mi->groups = G; mi->slot = UINT64_MAX;
            if (c->kind == TR_PROG_HIST) continue;
            tr_prog_fmeasure *f = &p->fm[p->nfm++];
            f->kind = c->kind; f->output = c->output; f->first = p->nft; f->nterms = c->kind == TR_PROG_COUNT_STAR ? 0 : c->nterms;
            f->count_term = mi->count_term; f->query = c->query; f->groups = G; f->slot = slot;
            mi->slot = slot;
            for (int32_t t = 0; t < f->nterms; t++) {
                int32_t j = basis_term(p, &p->outs[c->output], p->texps + (uint64_t)(c->first + t) * (uint64_t)sv.ncols);
                p->ft[p->nft].term = (uint32_t)j; p->ft[p->nft].reserved = 0; p->ft[p->nft].coef = p->tcoef[c->first + t];
                p->nft++;
            }
            slot = tr_budget_add(slot, G);
        }
        p->nslots = slot;
        /* the outputs each query reads, their sharers and bytes */
        uint8_t *reads = zalloc((uint64_t)p->nouts * (uint64_t)(fam->nqueries ? fam->nqueries : 1));
        int32_t *readers = zalloc((uint64_t)p->nouts * 4), *csize = zalloc((uint64_t)fam->nqueries * 4);
        if (!reads || !readers || !csize) { free(reads); free(readers); free(csize); rc = TR_MATH_MEMORY; goto fail; }
        for (int32_t m = 0; m < fam->nmeasures; m++) reads[(uint64_t)p->cm[m].query * p->nouts + p->cm[m].output] = 1;
        for (int32_t q = 0; q < fam->nqueries; q++) {
            csize[p->qclass[q]]++;
            for (int32_t o = 0; o < p->nouts; o++) readers[o] += reads[(uint64_t)q * p->nouts + o];
        }
        for (int32_t q = 0; q < fam->nqueries; q++) {
            const tr_prog_query *x = &p->queries[q];
            tr_prog_query_info *qi = &p->qinfo[q];
            qi->filter = p->qfilter[p->qclass[q]]; qi->klass = p->qclass[q]; qi->proof = p->qproof[q];
            qi->grouping = x->grouping; qi->nmeasures = x->nmeasures; qi->scalar = x->scalar;
            qi->conditions = qi->filter >= 0 ? p->deps[qi->filter] : 0;
            for (int32_t k = 0; k < x->nmeasures; k++) if (p->minfo[x->first + k].slot != UINT64_MAX)
                qi->slots = tr_budget_add(qi->slots, p->minfo[x->first + k].groups);
            for (int32_t o = 0; o < p->nouts; o++) {
                if (!reads[(uint64_t)q * p->nouts + o]) continue;
                qi->outputs++;
                if (readers[o] > qi->sharers) qi->sharers = readers[o];
                const tr_reduce_out *ro = &rv.outs[o];
                uint64_t G = ro->ngroups;
                qi->state_bytes = tr_budget_add(qi->state_bytes, ro->kind == TR_REDUCE_BANK ? tr_bank_state_bytes(G, ro->nterms)
                                                                                           : tr_budget_mul(G, 8));
            }
            qi->class_size = csize[p->qclass[q]];
        }
        free(reads); free(readers); free(csize);
        /* the program's own bytes (its tables, the FND-006 plan) and its value slots, before any run can need them */
        tr_reduce_plan_info pi;
        tr_reduce_plan_get_info(p->plan, &pi);
        uint64_t own = tr_budget_add(sizeof(tr_prog), tr_budget_add(tr_budget_mul((uint64_t)p->capnodes, 28),
                                                                     tr_budget_mul((uint64_t)p->hcap, 4)));
        own = tr_budget_add(own, tr_budget_mul((uint64_t)fam->nqueries, 12 + sizeof(tr_prog_query) + sizeof(tr_prog_query_info)));
        own = tr_budget_add(own, tr_budget_mul((uint64_t)fam->nmeasures,
                                               sizeof(cmeasure) + sizeof(tr_prog_fmeasure) + sizeof(tr_prog_measure_info)));
        own = tr_budget_add(own, tr_budget_add(tr_budget_mul((uint64_t)fam->nterms, (uint64_t)sv.ncols + 8),
                                               tr_budget_mul((uint64_t)p->ntterms, sizeof(tr_prog_fterm))));
        own = tr_budget_add(own, tr_budget_mul((uint64_t)(fam->nmeasures + fam->nqueries + 1), sizeof(pout)));
        for (int32_t o = 0; o < p->nouts; o++)
            own = tr_budget_add(own, tr_budget_mul((uint64_t)p->outs[o].cap, (uint64_t)(sv.ncols ? sv.ncols : 1)));
        p->info.bytes = tr_budget_add(own, pi.bytes);
        tr_math_need fn;
        need_zero(&fn);
        fn.need[TR_BUDGET_OUTPUT] = tr_budget_add((uint64_t)fam->nqueries, p->nslots);
        fn.need[TR_BUDGET_BYTES] = tr_budget_add(p->info.bytes, tr_budget_mul(p->nslots, sizeof(tr_prog_value)));
        if ((rc = tr_budget_check(budget, &fn, refusal))) goto fail;
    }
    {
        tr_prog_info *in = &p->info;
        in->nqueries = fam->nqueries; in->nmeasures = fam->nmeasures; in->noutputs = p->nouts;
        for (int32_t q = 0; q < fam->nqueries; q++) in->nclasses += p->qclass[q] == q;
        in->conds_in = fam->nconds; in->conds = p->nconds; in->nodes_in = fam->nnodes; in->nodes = p->nnodes;
        in->lifetime = p->lifetime; in->flags = p->flags; in->nvalue_measures = p->nfm;
        in->nhist = fam->nmeasures - p->nfm; in->slots = p->nslots;
    }
    free(cmap); free(nmap);
    set_error(error, TR_MATH_OK);
    return p;
fail:
    free(cmap); free(nmap);
    tr_prog_free(p);
    set_error(error, rc ? rc : TR_MATH_INVALID);
    return NULL;
}

int32_t tr_prog_get_info(const tr_prog *p, tr_prog_info *out) {
    if (!prog_valid(p) || !out) return TR_MATH_INVALID;
    *out = p->info;
    return TR_MATH_OK;
}
int32_t tr_prog_query_get(const tr_prog *p, int32_t q, tr_prog_query_info *out) {
    if (!prog_valid(p) || !out || q < 0 || q >= p->nqueries) return TR_MATH_INVALID;
    *out = p->qinfo[q];
    return TR_MATH_OK;
}
int32_t tr_prog_measure_get(const tr_prog *p, int32_t m, tr_prog_measure_info *out) {
    if (!prog_valid(p) || !out || m < 0 || m >= p->nmeasures) return TR_MATH_INVALID;
    *out = p->minfo[m];
    return TR_MATH_OK;
}
const tr_reduce_plan *tr_prog_reduce_plan(const tr_prog *p) { return prog_valid(p) ? p->plan : NULL; }
int32_t tr_prog_output_get(const tr_prog *p, int32_t o, tr_prog_output_info *out, uint8_t *exps, uint64_t capacity) {
    if (!prog_valid(p) || !out || o < 0 || o >= p->nouts) return TR_MATH_INVALID;
    const pout *x = &p->outs[o];
    memset(out, 0, sizeof *out);
    out->kind = x->kind; out->mode = x->mode; out->filter = x->filter; out->grouping = x->grouping;
    out->nterms = x->kind == TR_REDUCE_BANK ? x->nterms : 0; out->colset = x->colset;
    if (x->kind == TR_REDUCE_HIST) out->hist = p->cm[x->hist_measure].spec;
    for (int32_t m = 0; m < p->nmeasures; m++) {
        int seen = 0;
        for (int32_t k = 0; k < m && !seen; k++) seen = p->cm[k].output == o && p->cm[k].query == p->cm[m].query;
        if (p->cm[m].output == o && !seen) out->readers++;
    }
    uint64_t need = (uint64_t)out->nterms * (uint64_t)p->ncols;
    if (exps) {
        if (capacity < need) return TR_MATH_INVALID;
        if (need) memcpy(exps, x->exps, (size_t)need);
    }
    return TR_MATH_OK;
}

int32_t tr_prog_final_view_get(const tr_prog *p, const tr_reduce_result *r, tr_prog_fout *outs, int32_t capacity,
                               tr_prog_final_view *fv) {
    if (!prog_valid(p) || !fv || (p->nouts && (!outs || capacity < p->nouts))) return TR_MATH_INVALID;
    memset(fv, 0, sizeof *fv);
    for (int32_t o = 0; o < p->nouts; o++) {
        memset(&outs[o], 0, sizeof outs[o]);
        if (p->outs[o].kind != TR_REDUCE_BANK) continue;
        if (!r) continue;
        tr_bank_states st;
        int32_t rc = tr_reduce_result_bank(r, o, &st);
        if (rc) return rc;
        outs[o].rows = st.rows; outs[o].count = st.count; outs[o].sum = st.sum;
        outs[o].ngroups = st.ngroups; outs[o].nterms = st.nterms;
    }
    fv->measures = p->fm; fv->terms = p->ft; fv->outs = outs;
    fv->nmeasures = p->nfm; fv->nterms = p->nft; fv->noutputs = p->nouts; fv->nslots = p->nslots;
    return TR_MATH_OK;
}
/* the finalization view of the program's plan before any run: the bank outputs' shapes (their rows marked present,
 * never read): what the finalization's need formulas read */
static void final_shape(const tr_prog *p, tr_prog_fout *outs, tr_prog_final_view *fv) {
    tr_reduce_view rv;
    tr_reduce_get_view(p->plan, &rv);
    memset(fv, 0, sizeof *fv);
    for (int32_t o = 0; o < p->nouts; o++) {
        memset(&outs[o], 0, sizeof outs[o]);
        if (rv.outs[o].kind != TR_REDUCE_BANK) continue;
        outs[o].rows = (const uint64_t *)(const void *)outs;      /* a marker: the output has states */
        outs[o].ngroups = rv.outs[o].ngroups; outs[o].nterms = rv.outs[o].nterms;
    }
    fv->measures = p->fm; fv->terms = p->ft; fv->outs = outs;
    fv->nmeasures = p->nfm; fv->nterms = p->nft; fv->noutputs = p->nouts; fv->nslots = p->nslots;
}
int32_t tr_prog_final_need(const tr_prog *p, int32_t finalize, int32_t states, tr_math_need *need) {
    if (!prog_valid(p) || !need || (finalize != TR_REG_FINAL_CPU && finalize != TR_REG_FINAL_GPU)) return TR_MATH_INVALID;
    need_zero(need);
    need->need[TR_BUDGET_OUTPUT] = p->nslots;
    if (finalize == TR_REG_FINAL_CPU) {         /* the views of the result's states and the value slots */
        need->need[TR_BUDGET_BYTES] = tr_budget_add(tr_budget_mul((uint64_t)p->nouts, sizeof(tr_prog_fout)),
                                                    tr_budget_mul(p->nslots, sizeof(tr_prog_value)));
        return TR_MATH_OK;
    }
    tr_prog_fout *outs = zalloc((uint64_t)p->nouts * sizeof *outs);
    if (!outs) return TR_MATH_MEMORY;
    tr_prog_final_view fv;
    final_shape(p, outs, &fv);
    need->need[TR_BUDGET_BYTES] = tr_prog_final_bytes(&fv, states);
    free(outs);
    return TR_MATH_OK;
}
int32_t tr_prog_need(const tr_prog *p, int32_t plan, int32_t device, int32_t finalize, tr_math_need *need) {
    if (!prog_valid(p) || !need || (plan != TR_REDUCE_FUSED && plan != TR_REDUCE_REPEATED) ||
        (device != TR_REDUCE_ON_CPU && device != TR_REDUCE_ON_CUDA) ||
        (finalize != TR_REG_FINAL_CPU && finalize != TR_REG_FINAL_GPU)) return TR_MATH_INVALID;
    int32_t target = plan == TR_REDUCE_FUSED ? (device == TR_REDUCE_ON_CUDA ? TR_REDUCE_NEED_FUSED_DEVICE : TR_REDUCE_NEED_FUSED)
                                             : (device == TR_REDUCE_ON_CUDA ? TR_REDUCE_NEED_REPEATED_DEVICE : TR_REDUCE_NEED_REPEATED);
    int32_t rc = tr_reduce_need(p->plan, target, need);
    if (rc) return rc;
    /* the fused pass on the device with a device finalization keeps its states there (tr_mgpu_program_run) */
    int keep = plan == TR_REDUCE_FUSED && device == TR_REDUCE_ON_CUDA && finalize == TR_REG_FINAL_GPU;
    tr_math_need fin;
    if ((rc = tr_prog_final_need(p, finalize, !keep, &fin))) return rc;
    if (finalize == TR_REG_FINAL_GPU)             /* the host's copy of the values */
        fin.need[TR_BUDGET_BYTES] = tr_budget_add(fin.need[TR_BUDGET_BYTES], tr_budget_mul(p->nslots, sizeof(tr_prog_value)));
    if (fin.need[TR_BUDGET_OUTPUT] > need->need[TR_BUDGET_OUTPUT]) need->need[TR_BUDGET_OUTPUT] = fin.need[TR_BUDGET_OUTPUT];
    need->need[TR_BUDGET_BYTES] = tr_budget_add(need->need[TR_BUDGET_BYTES], fin.need[TR_BUDGET_BYTES]);
    return TR_MATH_OK;
}
int32_t tr_prog_finalize(const tr_prog *p, const tr_reduce_result *r, tr_prog_value *out, uint64_t capacity,
                         const tr_math_budget *budget, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!prog_valid(p) || !r || (p->nslots && (!out || capacity < p->nslots))) return TR_MATH_INVALID;
    tr_math_budget dflt;
    budget = budget_or(budget, &dflt);
    tr_math_need nd;
    int32_t rc = tr_prog_final_need(p, TR_REG_FINAL_CPU, 0, &nd);
    if (!rc) rc = tr_budget_check(budget, &nd, refusal);
    if (rc) return rc;
    tr_prog_fout *outs = zalloc((uint64_t)p->nouts * sizeof *outs);
    if (!outs) return TR_MATH_MEMORY;
    tr_prog_final_view fv;
    rc = tr_prog_final_view_get(p, r, outs, p->nouts, &fv);
    for (uint64_t i = 0; !rc && i < p->nslots; i++) tr_prog_eval_slot(&fv, i, &out[i]);
    free(outs);
    return rc;
}

/* ------------------------------------------------------------------ comparisons */
/* two canonical measures with equal values on every database: COUNT(p) counts the rows where p's written columns are
 * present (its coefficients do not matter; over no column it is COUNT(*)); SUM / AVG need the same written columns and
 * the same canonical polynomial; histograms the same normalized spec */
static int measure_equal(const tr_prog *p, const cmeasure *x, const cmeasure *y) {
    int32_t kx = x->kind == TR_PROG_COUNT && !x->colset ? TR_PROG_COUNT_STAR : x->kind;
    int32_t ky = y->kind == TR_PROG_COUNT && !y->colset ? TR_PROG_COUNT_STAR : y->kind;
    if (kx != ky) return 0;
    if (kx == TR_PROG_COUNT_STAR) return 1;
    if (kx == TR_PROG_HIST) return !memcmp(&x->spec, &y->spec, sizeof x->spec);
    if (x->colset != y->colset) return 0;
    if (kx == TR_PROG_COUNT) return 1;
    if (x->nterms != y->nterms) return 0;
    for (int32_t t = 0; t < x->nterms; t++) {
        if (p->tcoef[x->first + t] != p->tcoef[y->first + t]) return 0;
        if (p->ncols && memcmp(p->texps + (uint64_t)(x->first + t) * p->ncols, p->texps + (uint64_t)(y->first + t) * p->ncols,
                               (size_t)p->ncols)) return 0;
    }
    return 1;
}
static int measures_equal(const tr_prog *p, int32_t qa, int32_t qb) {
    const tr_prog_query *a = &p->queries[qa], *b = &p->queries[qb];
    if (a->nmeasures != b->nmeasures) return 0;
    for (int32_t k = 0; k < a->nmeasures; k++)
        if (!measure_equal(p, &p->cm[a->first + k], &p->cm[b->first + k])) return 0;
    return 1;
}
int32_t tr_prog_compare(const tr_prog *p, int32_t qa, int32_t qb, const tr_math_budget *budget,
                        tr_prog_compare_result *r, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!prog_valid(p) || !r || qa < 0 || qb < 0 || qa >= p->nqueries || qb >= p->nqueries) return TR_MATH_INVALID;
    memset(r, 0, sizeof *r);
    tr_math_budget dflt;
    budget = budget_or(budget, &dflt);
    r->ncols = p->ncols;
    r->grouping_equal = p->queries[qa].grouping == p->queries[qb].grouping && p->queries[qa].scalar == p->queries[qb].scalar;
    r->measures_equal = measures_equal(p, qa, qb);
    return search(p, p->qfilter[qa], p->qfilter[qb], budget->limit[TR_BUDGET_ITERATIONS], r);
}
int32_t tr_prog_snapshot_compare(const tr_prog *p, int32_t qa, int32_t qb, tr_prog_snapshot_result *r) {
    if (!prog_valid(p) || !r || qa < 0 || qb < 0 || qa >= p->nqueries || qb >= p->nqueries) return TR_MATH_INVALID;
    memset(r, 0, sizeof *r);
    r->first = UINT64_MAX;
    r->grouping_equal = p->queries[qa].grouping == p->queries[qb].grouping && p->queries[qa].scalar == p->queries[qb].scalar;
    r->measures_equal = measures_equal(p, qa, qb);
    const int32_t fa = p->qfilter[qa], fb = p->qfilter[qb];
    uint8_t *mark = zalloc((uint64_t)p->nnodes);
    int32_t *list = zalloc((uint64_t)p->nnodes * 4);
    uint64_t *val = zalloc((uint64_t)p->nnodes * 16), cw[2 * NCOND];
    if (!mark || !list || !val) { free(mark); free(list); free(val); return TR_MATH_MEMORY; }
    int32_t n = 0;
    cone(p, fa, fb, mark, list, &n);
    uint64_t reads = (fa >= 0 ? p->deps[fa] : 0) | (fb >= 0 ? p->deps[fb] : 0);
    const tr_scope_view *v = &p->sv;
    const uint64_t nw = (v->rows + 63) / 64;
    for (uint64_t w = 0; w < nw; w++) {
        uint64_t sel = v->selection[w];
        if (!sel) continue;
        r->words++;
        for (int32_t c = 0; c < p->nconds; c++)
            if ((reads >> c) & 1) tr_bool_condition_word(v, &p->conds[c], w, &cw[2 * c], &cw[2 * c + 1]);
        for (int32_t k = 0; k < n; k++) {
            int32_t i = list[k];
            const tr_bool_node *x = &p->nodes[i];
            uint64_t a, b;
            if (x->op == TR_BOOL_CONST) tr_bool_const_word(x->a, &a, &b);
            else if (x->op == TR_BOOL_VAR) { a = cw[2 * x->a]; b = cw[2 * x->a + 1]; }
            else tr_bool_apply(x->op, val[2 * (uint64_t)x->a], val[2 * (uint64_t)x->a + 1],
                               x->b >= 0 ? val[2 * (uint64_t)x->b] : 0, x->b >= 0 ? val[2 * (uint64_t)x->b + 1] : 0, &a, &b);
            val[2 * (uint64_t)i] = a; val[2 * (uint64_t)i + 1] = b;
        }
        uint64_t ma = fa < 0 ? sel : sel & val[2 * (uint64_t)fa], mb = fb < 0 ? sel : sel & val[2 * (uint64_t)fb];
        r->rows_a += (uint64_t)__builtin_popcountll(ma);
        r->rows_b += (uint64_t)__builtin_popcountll(mb);
        uint64_t d = ma ^ mb;
        r->differ += (uint64_t)__builtin_popcountll(d);
        if (d && r->first == UINT64_MAX) r->first = (w << 6) | (uint64_t)__builtin_ctzll(d);
    }
    r->equal = r->differ == 0 && r->grouping_equal && r->measures_equal;
    free(mark); free(list); free(val);
    return TR_MATH_OK;
}
