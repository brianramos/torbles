/* ENV-002: bounded multivariate parameter regions; see math_region.h.
 * SPDX-License-Identifier: Apache-2.0 */
#include "math_region.h"
#include "math_polyopt.h"
#include "math_domain.h"
#include "math_handles.h"
#include <stdlib.h>
#include <string.h>

TR_MATH_ABI(tr_region_problem, 1);      /* the ctypes boundary of ENV-002 (math_handles.h) */
TR_MATH_ABI(tr_region_info, 1);
TR_MATH_ABI(tr_region_class_info, 1);
TR_MATH_ABI(tr_region_region_info, 1);
TR_MATH_ABI(tr_region_vertex_info, 1);
TR_MATH_ABI(tr_region_boundary_info, 1);
TR_MATH_ABI(tr_region_answer, 1);
TR_MATH_ABI(tr_region_view, 1);

_Static_assert(sizeof(tr_region_problem) == 96, "tr_region_problem ABI");
_Static_assert(sizeof(tr_region_info) == 184, "tr_region_info ABI");
_Static_assert(sizeof(tr_region_class_info) == 48, "tr_region_class_info ABI");
_Static_assert(sizeof(tr_region_region_info) == 80, "tr_region_region_info ABI");
_Static_assert(sizeof(tr_region_vertex_info) == 32, "tr_region_vertex_info ABI");
_Static_assert(sizeof(tr_region_boundary_info) == 48, "tr_region_boundary_info ABI");
_Static_assert(sizeof(tr_region_answer) == 48, "tr_region_answer ABI");
_Static_assert(sizeof(tr_region_view) == 272, "tr_region_view ABI");

#define SET_MAGIC UINT32_C(0x32564e45)
#define JOB_MAGIC UINT32_C(0x4a474552)
#define NONE TR_REGION_NONE
#define MAXW TR_REGION_FORM

static void set_error(int32_t *error, int32_t code) { if (error) *error = code; }
static void *zalloc(uint64_t bytes) {
    if (bytes > (uint64_t)SIZE_MAX / 2) return NULL;
    return calloc(1, bytes ? (size_t)bytes : 1);
}
static const tr_math_budget *budget_or(const tr_math_budget *b, tr_math_budget *dflt) {
    if (b) return b;
    tr_budget_default(dflt);
    return dflt;
}
static void need_zero(tr_math_need *n) { for (int i = 0; i < TR_BUDGET_RESOURCES; i++) n->need[i] = 0; }

/* ================================================================== the build context: sticky error, counted work */
typedef struct {
    int32_t rc;
    const tr_math_budget *b;
    tr_math_refusal *r;
    uint64_t iters, bytes, maxbytes, regions, incid, nodes;
    uint64_t lps, pivots, clips, unverified;
} rctx;
static void fail(rctx *c, int32_t e) { if (!c->rc) c->rc = e ? e : TR_MATH_MEMORY; }
static void refuse(rctx *c, int32_t resource, uint64_t need) {
    if (c->rc) return;
    if (c->r) { c->r->resource = resource; c->r->refused = 1; c->r->need = need; c->r->limit = c->b->limit[resource]; }
    c->rc = TR_MATH_BUDGET;
}
/* a counted resource: refused when the count passes its limit (the need reported is limit + 1) */
static void count(rctx *c, int32_t resource, uint64_t *counter, uint64_t add) {
    if (c->rc) return;
    *counter = tr_budget_add(*counter, add);
    if (*counter > c->b->limit[resource]) {
        uint64_t lim = c->b->limit[resource];
        refuse(c, resource, lim == UINT64_MAX ? lim : lim + 1);
    }
}
static void work(rctx *c, uint64_t add) { count(c, TR_BUDGET_ITERATIONS, &c->iters, add); }
static void charge(rctx *c, uint64_t add) {
    if (c->rc) return;
    c->bytes = tr_budget_add(c->bytes, add);
    if (c->bytes > c->maxbytes) c->maxbytes = c->bytes;
    if (c->bytes > c->b->limit[TR_BUDGET_BYTES]) refuse(c, TR_BUDGET_BYTES, c->bytes);
}
#define LIVE(c) (!(c)->rc)

/* ------------------------------------------------------------------ big integers with the sticky error */
static tr_bigint *zk(rctx *c, tr_bigint *x, int32_t e) { if (!x) fail(c, e); return x; }
static tr_bigint *zi(rctx *c, int64_t v) {
    if (c->rc) return NULL;
    int32_t e = 0;
    return zk(c, tr_bigint_from_i64(v, c->b, c->r, &e), e);
}
#define ZOP2(name, fn) static tr_bigint *name(rctx *c, const tr_bigint *a, const tr_bigint *b) { \
        int32_t e = 0;                                                                          \
        return c->rc ? NULL : zk(c, fn(a, b, c->b, c->r, &e), e); }
#define ZOP1(name, fn) static tr_bigint *name(rctx *c, const tr_bigint *a) { \
        int32_t e = 0;                                                      \
        return c->rc ? NULL : zk(c, fn(a, c->b, c->r, &e), e); }
ZOP2(zmul, tr_bigint_mul)
ZOP2(zadd, tr_bigint_add)
ZOP2(zsub, tr_bigint_sub)
ZOP2(zgcd, tr_bigint_gcd)
ZOP2(zdivx, tr_bigint_divexact)
ZOP1(zneg, tr_bigint_neg)
ZOP1(zcp, tr_bigint_copy)
ZOP1(zabs, tr_bigint_abs)
static void zset(tr_bigint **dst, tr_bigint *v) { tr_bigint_free(*dst); *dst = v; }
static int zsg(const tr_bigint *x) { return x ? tr_bigint_sign(x) : 0; }
static tr_bigint *zshl(rctx *c, const tr_bigint *a, uint64_t k) {
    if (c->rc) return NULL;
    int32_t e = 0;
    return zk(c, tr_bigint_shl(a, k, c->b, c->r, &e), e);
}
static tr_bigint *zu64(rctx *c, uint64_t v) {
    if (c->rc) return NULL;
    int32_t e = 0;
    return zk(c, tr_bigint_from_limbs(1, &v, 1, c->b, c->r, &e), e);
}
/* a + b * m (m small, >= 0) */
static tr_bigint *zaddmul_u(rctx *c, const tr_bigint *a, const tr_bigint *b, uint64_t m) {
    tr_bigint *t = zu64(c, m), *p = zmul(c, b, t), *s = zadd(c, a, p);
    tr_bigint_free(t); tr_bigint_free(p);
    return s;
}
static uint64_t zbytes(const tr_bigint *x) { return 24 + 8 * (x ? tr_bigint_limb_count(x) : 0); }

/* ------------------------------------------------------------------ rationals with the sticky error */
static tr_bigq *qk(rctx *c, tr_bigq *x, int32_t e) { if (!x) fail(c, e); return x; }
#define QOP2(name, fn) static tr_bigq *name(rctx *c, const tr_bigq *a, const tr_bigq *b) { \
        int32_t e = 0;                                                                    \
        return c->rc ? NULL : qk(c, fn(a, b, c->b, c->r, &e), e); }
QOP2(qadd, tr_bigq_add)
QOP2(qsub, tr_bigq_sub)
QOP2(qmul, tr_bigq_mul)
QOP2(qdiv, tr_bigq_div)
static tr_bigq *qcp(rctx *c, const tr_bigq *a) {
    if (c->rc) return NULL;
    int32_t e = 0;
    return qk(c, tr_bigq_copy(a, c->b, c->r, &e), e);
}
static tr_bigq *qneg(rctx *c, const tr_bigq *a) {
    if (c->rc) return NULL;
    int32_t e = 0;
    return qk(c, tr_bigq_neg(a, c->b, c->r, &e), e);
}
static tr_bigq *qint(rctx *c, int64_t v) {
    if (c->rc) return NULL;
    int32_t e = 0;
    tr_num_rational q = {v, 1};
    return qk(c, tr_bigq_from_q64(q, c->b, c->r, &e), e);
}
static tr_bigq *qfrac(rctx *c, int64_t n, int64_t d) {
    if (c->rc) return NULL;
    int32_t e = 0;
    tr_num_rational q = {n, d};
    return qk(c, tr_bigq_from_q64(q, c->b, c->r, &e), e);
}
static tr_bigq *qz(rctx *c, const tr_bigint *n, const tr_bigint *d) {
    if (c->rc) return NULL;
    int32_t e = 0;
    return qk(c, tr_bigq_from_bigints(n, d, c->b, c->r, &e), e);
}
static tr_bigq *qzi(rctx *c, const tr_bigint *n) {
    if (c->rc) return NULL;
    int32_t e = 0;
    return qk(c, tr_bigq_from_bigint(n, c->b, c->r, &e), e);
}
static void qset(tr_bigq **dst, tr_bigq *v) { tr_bigq_free(*dst); *dst = v; }
static int qsg(const tr_bigq *q) { return q ? tr_bigq_sign(q) : 0; }
static int q_ok(const tr_bigq *q) { return q && tr_bigq_is_handle(q); }
/* lcm(a, b) of positive integers */
static tr_bigint *zlcm(rctx *c, const tr_bigint *a, const tr_bigint *b) {
    tr_bigint *g = zgcd(c, a, b), *q = g ? zdivx(c, a, g) : NULL, *l = zmul(c, q, b);
    tr_bigint_free(g); tr_bigint_free(q);
    return l;
}

/* ------------------------------------------------------------------ linear forms in homogeneous coordinates */
/* a form f: f[0..d-1] the coefficients of x_1..x_d, f[d] that of w; a point x: x[0..d-1], x[d] = w > 0 */
static tr_bigint *form_eval(rctx *c, tr_bigint *const *f, tr_bigint *const *x, int32_t W) {
    tr_bigint *s = zi(c, 0);
    for (int32_t t = 0; t < W && LIVE(c); t++) {
        if (!zsg(f[t]) || !zsg(x[t])) continue;
        tr_bigint *p = zmul(c, f[t], x[t]);
        zset(&s, zadd(c, s, p));
        tr_bigint_free(p);
    }
    if (c->rc) { tr_bigint_free(s); return NULL; }
    return s;
}
/* divide every entry by the gcd of their absolute values (> 0); *allzero when every entry is 0 */
static void primitive(rctx *c, tr_bigint **f, int32_t W, int *allzero) {
    tr_bigint *g = NULL;
    for (int32_t t = 0; t < W && LIVE(c); t++) {
        if (!zsg(f[t])) continue;
        if (!g) g = zabs(c, f[t]);
        else zset(&g, zgcd(c, g, f[t]));
    }
    if (allzero) *allzero = g == NULL;
    if (g && LIVE(c) && !(tr_bigint_bit_length(g) == 1)) {
        for (int32_t t = 0; t < W && LIVE(c); t++) if (zsg(f[t])) zset(&f[t], zdivx(c, f[t], g));
    }
    tr_bigint_free(g);
}
static int forms_cmp(tr_bigint *const *a, tr_bigint *const *b, int32_t W) {
    for (int32_t t = 0; t < W; t++) {
        int r = tr_bigint_cmp(a[t], b[t]);
        if (r) return r;
    }
    return 0;
}
static uint64_t bitlen(const tr_bigint *x) { return x ? tr_bigint_bit_length(x) : 0; }

/* ------------------------------------------------------------------ merge sort of indices with a context */
typedef int (*cmp_fn)(const void *ctx, uint64_t a, uint64_t b);
static void msort_rec(uint64_t *a, uint64_t *tmp, uint64_t n, cmp_fn cmp, const void *ctx) {
    if (n < 2) return;
    uint64_t h = n / 2;
    msort_rec(a, tmp, h, cmp, ctx); msort_rec(a + h, tmp, n - h, cmp, ctx);
    uint64_t i = 0, j = h, k = 0;
    while (i < h && j < n) tmp[k++] = cmp(ctx, a[j], a[i]) < 0 ? a[j++] : a[i++];
    while (i < h) tmp[k++] = a[i++];
    while (j < n) tmp[k++] = a[j++];
    memcpy(a, tmp, (size_t)n * 8);
}
static int msort(uint64_t *a, uint64_t n, cmp_fn cmp, const void *ctx) {
    uint64_t *tmp = zalloc(n * 8);
    if (!tmp) return TR_MATH_MEMORY;
    msort_rec(a, tmp, n, cmp, ctx);
    free(tmp);
    return 0;
}

/* ================================================================== objects */
typedef struct { int32_t kind, reserved; uint64_t index; } src_t;
typedef struct {
    tr_bigint *a[MAXW];                 /* the form (<= 0 inside) */
    uint32_t ns, cap;
    src_t *src;
} cons_t;
static void cons_free(cons_t *k) {
    for (int t = 0; t < MAXW; t++) tr_bigint_free(k->a[t]);
    free(k->src);
    memset(k, 0, sizeof *k);
}
static void cons_addsrc(rctx *c, cons_t *k, int32_t kind, uint64_t index) {
    if (c->rc) return;
    for (uint32_t i = 0; i < k->ns; i++) if (k->src[i].kind == kind && k->src[i].index == index) return;
    if (k->ns == k->cap) {
        uint32_t nc = k->cap ? 2 * k->cap : 2;
        src_t *n = realloc(k->src, (size_t)nc * sizeof *n);
        if (!n) { fail(c, TR_MATH_MEMORY); return; }
        k->src = n; k->cap = nc;
    }
    k->src[k->ns].kind = kind; k->src[k->ns].reserved = 0; k->src[k->ns].index = index;
    k->ns++;
}

typedef struct { tr_bigint *x[MAXW]; uint64_t *z; } vert_t;     /* homogeneous primitive (x, w), w > 0; tight bits */
typedef struct { uint32_t n, cap, words, W; vert_t *v; } poly_t;
static void vert_free(vert_t *v) { for (int t = 0; t < MAXW; t++) tr_bigint_free(v->x[t]); free(v->z); memset(v, 0, sizeof *v); }
static void poly_free(poly_t *P) {
    for (uint32_t i = 0; i < P->n; i++) vert_free(&P->v[i]);
    free(P->v);
    memset(P, 0, sizeof *P);
}
static int poly_push(rctx *c, poly_t *P, vert_t *v) {    /* takes v */
    if (c->rc) { vert_free(v); return 0; }
    if (P->n == P->cap) {
        uint32_t nc = P->cap ? 2 * P->cap : 16;
        vert_t *n = realloc(P->v, (size_t)nc * sizeof *n);
        if (!n) { vert_free(v); fail(c, TR_MATH_MEMORY); return 0; }
        P->v = n; P->cap = nc;
    }
    P->v[P->n++] = *v;
    memset(v, 0, sizeof *v);
    return 1;
}
static void point_primitive(rctx *c, tr_bigint **x, int32_t W) {
    int z = 0;
    primitive(c, x, W, &z);
    if (LIVE(c) && (z || zsg(x[W - 1]) <= 0)) fail(c, TR_MATH_INVALID);   /* w > 0 always (a bounded polytope) */
}
/* a copy of P with tight bitsets of `words` words */
static void poly_copy(rctx *c, poly_t *dst, const poly_t *src, uint32_t words) {
    memset(dst, 0, sizeof *dst);
    dst->W = src->W; dst->words = words;
    for (uint32_t i = 0; i < src->n && LIVE(c); i++) {
        vert_t v;
        memset(&v, 0, sizeof v);
        for (uint32_t t = 0; t < src->W; t++) v.x[t] = zcp(c, src->v[i].x[t]);
        v.z = zalloc((uint64_t)words * 8);
        if (!v.z) fail(c, TR_MATH_MEMORY);
        else memcpy(v.z, src->v[i].z, (size_t)(src->words < words ? src->words : words) * 8);
        poly_push(c, dst, &v);
    }
}
static int bit(const uint64_t *z, uint32_t i) { return (int)((z[i >> 6] >> (i & 63)) & 1); }
static void setbit(uint64_t *z, uint32_t i) { z[i >> 6] |= (uint64_t)1 << (i & 63); }

/* clips P by the halfspace of form h (constraint index hidx): the double description step. Vertices with slack > 0
 * stay, those on the hyperplane gain the bit, those outside go; every edge from a kept to a removed vertex
 * (adjacency by the combinatorial test: no third vertex is tight on every constraint both are) gives a new vertex */
static void poly_clip(rctx *c, poly_t *P, tr_bigint *const *h, uint32_t hidx, int32_t d) {
    if (c->rc || !P->n) return;
    work(c, 1);
    c->clips++;
    const uint32_t n = P->n, W = P->W, words = P->words;
    tr_bigint **s = zalloc((uint64_t)n * sizeof *s);
    int8_t *sg = zalloc(n);
    uint64_t *and = zalloc((uint64_t)words * 8);
    if (!s || !sg || !and) { free(s); free(sg); free(and); fail(c, TR_MATH_MEMORY); return; }
    uint32_t npos = 0, nneg = 0;
    for (uint32_t i = 0; i < n && LIVE(c); i++) {
        tr_bigint *e = form_eval(c, h, P->v[i].x, (int32_t)W);
        s[i] = zneg(c, e);                       /* the slack: >= 0 inside */
        tr_bigint_free(e);
        sg[i] = (int8_t)zsg(s[i]);
        if (sg[i] > 0) npos++; else if (sg[i] < 0) nneg++;
    }
    poly_t Q;
    memset(&Q, 0, sizeof Q);
    Q.W = W; Q.words = words;
    if (LIVE(c) && nneg) {
        for (uint32_t u = 0; u < n && LIVE(c); u++) {
            if (sg[u] <= 0) continue;
            for (uint32_t v = 0; v < n && LIVE(c); v++) {
                if (sg[v] >= 0) continue;
                int pc = 0;
                for (uint32_t q = 0; q < words; q++) { and[q] = P->v[u].z[q] & P->v[v].z[q]; pc += __builtin_popcountll(and[q]); }
                if (pc < d - 1) continue;
                int adjacent = 1;
                for (uint32_t w = 0; w < n && adjacent; w++) {
                    if (w == u || w == v) continue;
                    int sub = 1;
                    for (uint32_t q = 0; q < words && sub; q++) if (and[q] & ~P->v[w].z[q]) sub = 0;
                    if (sub) adjacent = 0;
                }
                if (!adjacent) continue;
                vert_t nv;                       /* s_u x_v - s_v x_u: on the hyperplane, w > 0 */
                memset(&nv, 0, sizeof nv);
                for (uint32_t t = 0; t < W && LIVE(c); t++) {
                    tr_bigint *a = zmul(c, s[u], P->v[v].x[t]), *b = zmul(c, s[v], P->v[u].x[t]);
                    nv.x[t] = zsub(c, a, b);
                    tr_bigint_free(a); tr_bigint_free(b);
                }
                point_primitive(c, nv.x, (int32_t)W);
                nv.z = zalloc((uint64_t)words * 8);
                if (!nv.z) { fail(c, TR_MATH_MEMORY); vert_free(&nv); break; }
                memcpy(nv.z, and, (size_t)words * 8);
                setbit(nv.z, hidx);
                poly_push(c, &Q, &nv);
            }
        }
    }
    if (LIVE(c)) {
        /* the kept vertices first (in their order), then the new ones */
        poly_t R;
        memset(&R, 0, sizeof R);
        R.W = W; R.words = words;
        for (uint32_t i = 0; i < n; i++) {
            if (sg[i] < 0) { vert_free(&P->v[i]); continue; }
            if (sg[i] == 0) setbit(P->v[i].z, hidx);
            poly_push(c, &R, &P->v[i]);
        }
        for (uint32_t i = 0; i < Q.n; i++) poly_push(c, &R, &Q.v[i]);
        free(Q.v); Q.v = NULL; Q.n = 0;
        free(P->v);
        *P = R;
    } else poly_free(&Q);
    for (uint32_t i = 0; i < n; i++) tr_bigint_free(s[i]);
    free(s); free(sg); free(and);
}

/* the exact rank of the rows (homogeneous vectors of W entries), stopping once it reaches `target` */
static int32_t rank_rows(rctx *c, tr_bigint *const *const *rows, uint32_t nrows, int32_t W, int32_t target) {
    if (c->rc || !nrows) return 0;
    tr_bigint **M = zalloc((uint64_t)nrows * W * sizeof *M);
    if (!M) { fail(c, TR_MATH_MEMORY); return 0; }
    for (uint32_t i = 0; i < nrows && LIVE(c); i++)
        for (int32_t t = 0; t < W; t++) M[(uint64_t)i * W + t] = zcp(c, rows[i][t]);
    int32_t rank = 0;
    for (int32_t col = 0; col < W && LIVE(c) && (uint32_t)rank < nrows && rank < target; col++) {
        uint32_t piv = nrows;
        for (uint32_t i = (uint32_t)rank; i < nrows; i++) if (zsg(M[(uint64_t)i * W + col])) { piv = i; break; }
        if (piv == nrows) continue;
        if (piv != (uint32_t)rank)
            for (int32_t t = 0; t < W; t++) {
                tr_bigint *x = M[(uint64_t)piv * W + t];
                M[(uint64_t)piv * W + t] = M[(uint64_t)rank * W + t]; M[(uint64_t)rank * W + t] = x;
            }
        tr_bigint *const *pr = &M[(uint64_t)rank * W];
        for (uint32_t i = (uint32_t)rank + 1; i < nrows && LIVE(c); i++) {
            tr_bigint **ri = &M[(uint64_t)i * W];
            if (!zsg(ri[col])) continue;
            tr_bigint *f = zcp(c, ri[col]);
            for (int32_t t = col; t < W && LIVE(c); t++) {
                tr_bigint *a = zmul(c, ri[t], pr[col]), *b = zmul(c, pr[t], f);
                zset(&ri[t], zsub(c, a, b));
                tr_bigint_free(a); tr_bigint_free(b);
            }
            tr_bigint_free(f);
            int z = 0;
            primitive(c, ri, W, &z);
        }
        rank++;
    }
    for (uint64_t i = 0; i < (uint64_t)nrows * W; i++) tr_bigint_free(M[i]);
    free(M);
    return rank;
}

/* ------------------------------------------------------------------ the region set */
typedef struct {
    uint64_t cell, nmem, winner, region;
    uint64_t *mem;
    tr_bigint *A[MAXW];                 /* the internal form A_1..A_d, A_0 (negated for upper regions) */
    tr_bigint *D;
    tr_bigq *coef[TR_REGION_MAX_DIMS + 1];   /* a_0, a_1..a_d as given (or the enclosure), not negated */
    tr_bigq *eps;
    int32_t degree;
} class_t;
typedef struct {
    tr_bigint *a[MAXW];                 /* the form */
    tr_bigint *rhs;                     /* c of n theta <= c: -a[d] */
    int32_t eq, reserved;
    uint32_t ns;
    src_t *src;
} rcons_t;
typedef struct {
    uint64_t cls, cell;
    int32_t dim, lp_sign;
    uint64_t lp_rows;
    tr_bigq *pt[TR_REGION_MAX_DIMS], *margin;
    uint64_t nv;
    uint64_t *vid;                      /* global vertex ids */
    vert_t *lv;                         /* the local vertices until the global merge (then freed) */
    uint32_t nineq, neq;
    rcons_t *k;                         /* inequalities then equalities */
    uint64_t **kvert;                   /* per inequality (full regions): its local vertex bitset (scratch) */
} region_t;
typedef struct {
    uint64_t cell;
    tr_bigint *x[MAXW];
    tr_bigq *pt[TR_REGION_MAX_DIMS];
    uint64_t ncls, *cls, winner, nwalls, *walls;
} gvert_t;
typedef struct {
    uint64_t cell, region, nv, *vid, ncls, *cls, winner;
    int32_t kind, reserved;
    tr_bigint *a[MAXW], *rhs;
} bnd_t;
typedef struct {
    int32_t axis, reserved;
    uint64_t link[2];                   /* children, or (candidate offset, count) */
    uint32_t tlo[TR_REGION_MAX_DIMS], thi[TR_REGION_MAX_DIMS];
    tr_bigint *split;                   /* the split value's numerator p over the cell's W */
} node_t;
typedef struct {
    tr_bigq *lo[TR_REGION_MAX_DIMS], *hi[TR_REGION_MAX_DIMS];
    uint64_t digit[TR_REGION_MAX_DIMS];
    tr_bigint *W, *base[TR_REGION_MAX_DIMS], *step[TR_REGION_MAX_DIMS];
    uint64_t class_begin, class_end, region_begin, region_end, root;
} cell_t;

struct tr_region_set_s {
    uint32_t magic;
    int32_t d, kind, upper, tie, claim, contract, pieces, nrows, degree;
    uint64_t ngroups, present;
    tr_bigq *lo[TR_REGION_MAX_DIMS], *hi[TR_REGION_MAX_DIMS];
    tr_bigq **A, **b;
    tr_bigq *eps, *env_err, *budget_err;
    cons_t walls[2 * TR_REGION_MAX_DIMS + TR_REGION_MAX_ROWS];   /* declared box faces (2k, 2k + 1) and rows */
    uint64_t nwalls;
    uint64_t ncells; cell_t *cells;
    uint64_t nclasses; class_t *cls;
    uint64_t nregions, regcap; region_t *reg;
    uint64_t nvert; gvert_t *vert;
    uint64_t nbnd; bnd_t *bnd;
    uint64_t nnodes, nodecap; node_t *node;
    uint64_t ncand, candcap; uint64_t *cand;
    tr_region_info info;
};
static int set_ok(const tr_region_set *s) { return s && s->magic == SET_MAGIC; }

static void region_free(region_t *r) {
    for (int t = 0; t < TR_REGION_MAX_DIMS; t++) tr_bigq_free(r->pt[t]);
    tr_bigq_free(r->margin);
    free(r->vid);
    if (r->lv) { for (uint64_t i = 0; i < r->nv; i++) vert_free(&r->lv[i]); free(r->lv); }
    for (uint32_t i = 0; r->k && i < r->nineq + r->neq; i++) {
        for (int t = 0; t < MAXW; t++) tr_bigint_free(r->k[i].a[t]);
        tr_bigint_free(r->k[i].rhs);
        free(r->k[i].src);
    }
    free(r->k);
    if (r->kvert) { for (uint32_t i = 0; i < r->nineq; i++) free(r->kvert[i]); free(r->kvert); }
    memset(r, 0, sizeof *r);
}
void tr_region_free(tr_region_set *s) {
    if (!s || s->magic != SET_MAGIC) return;
    s->magic = 0;
    for (int k = 0; k < TR_REGION_MAX_DIMS; k++) { tr_bigq_free(s->lo[k]); tr_bigq_free(s->hi[k]); }
    for (int64_t i = 0; s->A && i < (int64_t)s->nrows * s->d; i++) tr_bigq_free(s->A[i]);
    for (int32_t i = 0; s->b && i < s->nrows; i++) tr_bigq_free(s->b[i]);
    free(s->A); free(s->b);
    tr_bigq_free(s->eps); tr_bigq_free(s->env_err); tr_bigq_free(s->budget_err);
    for (uint64_t i = 0; i < s->nwalls; i++) cons_free(&s->walls[i]);
    for (uint64_t i = 0; s->cells && i < s->ncells; i++) {
        cell_t *C = &s->cells[i];
        for (int k = 0; k < TR_REGION_MAX_DIMS; k++) {
            tr_bigq_free(C->lo[k]); tr_bigq_free(C->hi[k]); tr_bigint_free(C->base[k]); tr_bigint_free(C->step[k]);
        }
        tr_bigint_free(C->W);
    }
    free(s->cells);
    for (uint64_t i = 0; s->cls && i < s->nclasses; i++) {
        class_t *K = &s->cls[i];
        free(K->mem);
        for (int t = 0; t < MAXW; t++) { tr_bigint_free(K->A[t]); }
        for (int t = 0; t <= TR_REGION_MAX_DIMS; t++) tr_bigq_free(K->coef[t]);
        tr_bigint_free(K->D); tr_bigq_free(K->eps);
    }
    free(s->cls);
    for (uint64_t i = 0; s->reg && i < s->nregions; i++) region_free(&s->reg[i]);
    free(s->reg);
    for (uint64_t i = 0; s->vert && i < s->nvert; i++) {
        gvert_t *v = &s->vert[i];
        for (int t = 0; t < MAXW; t++) tr_bigint_free(v->x[t]);
        for (int t = 0; t < TR_REGION_MAX_DIMS; t++) tr_bigq_free(v->pt[t]);
        free(v->cls); free(v->walls);
    }
    free(s->vert);
    for (uint64_t i = 0; s->bnd && i < s->nbnd; i++) {
        bnd_t *b = &s->bnd[i];
        free(b->vid); free(b->cls);
        for (int t = 0; t < MAXW; t++) tr_bigint_free(b->a[t]);
        tr_bigint_free(b->rhs);
    }
    free(s->bnd);
    for (uint64_t i = 0; s->node && i < s->nnodes; i++) tr_bigint_free(s->node[i].split);
    free(s->node); free(s->cand);
    free(s);
}

/* ================================================================== problem checks and needs */
static int32_t problem_check(const tr_region_problem *p, int *quadratic) {
    if (!p || p->reserved) return TR_MATH_INVALID;
    if (p->d < 1) return TR_MATH_INVALID;
    if (p->d > TR_REGION_MAX_DIMS) return TR_MATH_UNSUPPORTED;
    const int32_t d = p->d, aff = 1 + d, full = 1 + d + d * (d + 1) / 2;
    if ((p->ncoef != aff && p->ncoef != full) || (p->ngroups && !p->coef && p->ncoef)) return TR_MATH_INVALID;
    if (p->nrows < 0) return TR_MATH_INVALID;
    if (p->nrows > TR_REGION_MAX_ROWS) return TR_MATH_UNSUPPORTED;
    if (p->nrows && (!p->A || !p->b)) return TR_MATH_INVALID;
    if (p->upper != 0 && p->upper != 1) return TR_MATH_INVALID;
    if (p->tie != TR_REGION_TIE_SMALLEST && p->tie != TR_REGION_TIE_LARGEST) return TR_MATH_INVALID;
    if (p->contract != TR_REGION_EXACT && p->contract != TR_REGION_PIECEWISE_LINEAR) return TR_MATH_INVALID;
    if (p->contract == TR_REGION_EXACT && p->pieces != 0 && p->pieces != 1) return TR_MATH_INVALID;
    if (p->contract == TR_REGION_PIECEWISE_LINEAR) {
        if (p->pieces < 1 || !q_ok(p->error_budget) || tr_bigq_sign(p->error_budget) < 0) return TR_MATH_INVALID;
        if (p->pieces > TR_REGION_MAX_PIECES) return TR_MATH_UNSUPPORTED;
        uint64_t cells = 1;
        for (int32_t k = 0; k < d; k++) cells *= (uint64_t)p->pieces;
        if (cells > TR_REGION_MAX_CELLS) return TR_MATH_UNSUPPORTED;
    }
    if (!p->lo || !p->hi) return TR_MATH_INVALID;
    for (int32_t k = 0; k < d; k++) {
        if (!q_ok(p->lo[k]) || !q_ok(p->hi[k])) return TR_MATH_INVALID;
        if (tr_bigq_cmp(p->lo[k], p->hi[k]) >= 0) return TR_MATH_INVALID;
    }
    for (int64_t i = 0; i < (int64_t)p->nrows * d; i++) if (p->A[i] && !q_ok(p->A[i])) return TR_MATH_INVALID;
    for (int32_t i = 0; i < p->nrows; i++) if (!q_ok(p->b[i])) return TR_MATH_INVALID;
    int quad = 0;
    for (uint64_t g = 0; g < p->ngroups; g++) {
        if (p->present && !p->present[g]) continue;
        for (int32_t t = 0; t < p->ncoef; t++) {
            const tr_bigq *q = p->coef[g * (uint64_t)p->ncoef + (uint64_t)t];
            if (q && !q_ok(q)) return TR_MATH_INVALID;
            if (t >= aff && q && tr_bigq_sign(q)) quad = 1;
        }
    }
    *quadratic = quad;
    return 0;
}
static uint64_t cells_of(const tr_region_problem *p) {
    if (p->contract != TR_REGION_PIECEWISE_LINEAR) return 1;
    uint64_t cells = 1;
    for (int32_t k = 0; k < p->d; k++) cells *= (uint64_t)p->pieces;
    return cells;
}
static uint64_t present_of(const tr_region_problem *p) {
    uint64_t n = 0;
    for (uint64_t g = 0; g < p->ngroups; g++) if (!p->present || p->present[g]) n++;
    return n;
}
int32_t tr_region_need(const tr_region_problem *p, tr_math_need *need) {
    int quad = 0;
    if (!need) return TR_MATH_INVALID;
    int32_t rc = problem_check(p, &quad);
    if (rc) return rc;
    need_zero(need);
    const uint64_t cells = cells_of(p), present = present_of(p);
    need->need[TR_BUDGET_VARIABLES] = (uint64_t)p->d;
    need->need[TR_BUDGET_DEGREE] = quad ? 2 : 1;
    need->need[TR_BUDGET_STATES] = tr_budget_mul(cells, p->ngroups);
    /* the class tables: per cell and present group its form (d + 2 numbers) and coefficients */
    uint64_t per = (uint64_t)(2 * p->d + 4) * 48 + sizeof(class_t) + 16;
    need->need[TR_BUDGET_BYTES] = tr_budget_add(tr_budget_add(sizeof(tr_region_set), tr_budget_mul(cells, sizeof(cell_t))),
                                                tr_budget_mul(tr_budget_mul(cells, present), per));
    return 0;
}

/* ================================================================== building blocks */
/* the primitive form of n theta <= c, n and c rationals (n NULL entries 0) */
static void form_from_q(rctx *c, const tr_bigq *const *n, const tr_bigq *rhs, int32_t d, tr_bigint **out) {
    tr_bigint *L = zi(c, 1);
    for (int32_t k = 0; k < d && LIVE(c); k++) if (n[k] && qsg(n[k])) zset(&L, zlcm(c, L, tr_bigq_denominator(n[k])));
    if (LIVE(c)) zset(&L, zlcm(c, L, tr_bigq_denominator(rhs)));
    for (int32_t k = 0; k <= d && LIVE(c); k++) {
        const tr_bigq *q = k < d ? n[k] : rhs;
        if (!q || !qsg(q)) { out[k] = zi(c, 0); continue; }
        tr_bigint *t = zdivx(c, L, tr_bigq_denominator(q)), *v = zmul(c, t, tr_bigq_numerator(q));
        out[k] = k < d ? v : zneg(c, v);
        if (k == d) tr_bigint_free(v);
        tr_bigint_free(t);
    }
    tr_bigint_free(L);
    int z = 0;
    primitive(c, out, d + 1, &z);
}
/* the box face forms of [lo, hi]: 2 k: theta_k >= lo_k (-q x_k + p w <= 0), 2 k + 1: theta_k <= hi_k */
static void box_face(rctx *c, const tr_bigq *bound, int32_t k, int upper, int32_t d, tr_bigint **out) {
    for (int32_t t = 0; t <= d; t++) out[t] = zi(c, 0);
    if (c->rc) return;
    if (upper) {
        zset(&out[k], zcp(c, tr_bigq_denominator(bound)));
        zset(&out[d], zneg(c, tr_bigq_numerator(bound)));
    } else {
        zset(&out[k], zneg(c, tr_bigq_denominator(bound)));
        zset(&out[d], zcp(c, tr_bigq_numerator(bound)));
    }
}
/* the homogeneous primitive point of rationals q[0..d) */
static void point_of_q(rctx *c, tr_bigq *const *q, int32_t d, tr_bigint **x) {
    tr_bigint *L = zi(c, 1);
    for (int32_t k = 0; k < d && LIVE(c); k++) zset(&L, zlcm(c, L, tr_bigq_denominator(q[k])));
    for (int32_t k = 0; k < d && LIVE(c); k++) {
        tr_bigint *t = zdivx(c, L, tr_bigq_denominator(q[k]));
        x[k] = zmul(c, t, tr_bigq_numerator(q[k]));
        tr_bigint_free(t);
    }
    x[d] = zcp(c, L);
    tr_bigint_free(L);
    point_primitive(c, x, d + 1);
}
/* the box polytope of [lo, hi]: 2^d corners, tight on their faces (constraint 2 k: lower, 2 k + 1: upper) */
static void box_poly(rctx *c, tr_bigq *const *lo, tr_bigq *const *hi, int32_t d, uint32_t words, poly_t *P) {
    memset(P, 0, sizeof *P);
    P->W = (uint32_t)d + 1; P->words = words;
    for (uint32_t m = 0; m < (1u << d) && LIVE(c); m++) {
        tr_bigq *q[TR_REGION_MAX_DIMS];
        vert_t v;
        memset(&v, 0, sizeof v);
        v.z = zalloc((uint64_t)words * 8);
        if (!v.z) { fail(c, TR_MATH_MEMORY); break; }
        for (int32_t k = 0; k < d; k++) {
            q[k] = ((m >> k) & 1) ? hi[k] : lo[k];
            setbit(v.z, (uint32_t)(2 * k + ((m >> k) & 1)));
        }
        point_of_q(c, q, d, v.x);
        poly_push(c, P, &v);
    }
}
static uint32_t words_of(uint64_t m) { return (uint32_t)((m + 63) / 64); }

/* ------------------------------------------------------------------ primes and Garner constants (INV-001's table) */
typedef struct { uint32_t k; tr_gfp F[TR_REGION_PRIMES_MAX]; uint64_t garner[TR_REGION_PRIMES_MAX * TR_REGION_PRIMES_MAX];
                 uint64_t half[TR_REGION_PRIMES_MAX]; } primes_t;
static int32_t primes_setup(rctx *c, uint32_t k, primes_t *P) {
    if (k < 1 || k > TR_REGION_PRIMES_MAX) return TR_MATH_OVERFLOW;
    memset(P, 0, sizeof *P);
    P->k = k;
    for (uint32_t j = 0; j < k; j++) {
        uint64_t p = 0;
        if (tr_inv_prime((int32_t)j, &p) || tr_gfp_field(p, &P->F[j])) return TR_MATH_INVALID;
    }
    for (uint32_t j = 0; j < k; j++)
        for (uint32_t t = 0; t < j; t++) P->garner[j * k + t] = tr_gfp_inv_hd(&P->F[j], P->F[t].p % P->F[j].p);
    tr_bigint *M = zi(c, 1);
    for (uint32_t j = 0; j < k && LIVE(c); j++) { tr_bigint *pj = zu64(c, P->F[j].p); zset(&M, zmul(c, M, pj)); tr_bigint_free(pj); }
    tr_bigint *H = NULL;
    if (LIVE(c)) { int32_t e = 0; H = zk(c, tr_bigint_shr(M, 1, c->b, c->r, &e), e); }
    tr_bigint_free(M);
    for (uint32_t j = 0; j < k && H && LIVE(c); j++) {
        tr_bigint *pj = zu64(c, P->F[j].p), *q = NULL, *r = NULL;
        if (pj && !tr_bigint_divmod(H, pj, TR_DIV_FLOOR, &q, &r, c->b, c->r)) {
            uint64_t cnt = 0, w = 0;
            int32_t sg = 0;
            if (tr_bigint_export(r, &sg, &w, 1, &cnt)) w = 0;
            P->half[j] = cnt ? w : 0;
            tr_bigint_free(H); H = q;
        } else { tr_bigint_free(H); H = NULL; fail(c, TR_MATH_MEMORY); }
        tr_bigint_free(pj); tr_bigint_free(r);
    }
    tr_bigint_free(H);
    return c->rc;
}
static uint64_t zres(const tr_bigint *x, const tr_gfp *F) { uint64_t r = 0; if (x) tr_bigint_mod_u64(x, F->p, &r); return r; }

/* ================================================================== classes of one cell */
/* the affine function of group g on the cell (exact: as given; contract: the enclosure), its eps and degree */
static void cell_function(rctx *c, const tr_region_problem *p, uint64_t g, const cell_t *C, tr_bigq **a, tr_bigq **eps,
                          int32_t *degree) {
    const int32_t d = p->d, aff = 1 + d;
    const tr_bigq *const *in = p->coef ? p->coef + g * (uint64_t)p->ncoef : NULL;
    int quad = 0, lin = 0;
    for (int32_t t = 0; t < p->ncoef; t++) {
        const tr_bigq *q = in ? in[t] : NULL;
        if (q && qsg(q)) { if (t >= aff) quad = 1; else if (t >= 1) lin = 1; }
    }
    *degree = quad ? 2 : lin ? 1 : 0;
    for (int32_t t = 0; t < aff; t++) a[t] = (in && in[t]) ? qcp(c, in[t]) : qint(c, 0);
    *eps = qint(c, 0);
    if (!quad || c->rc) return;
    /* the enclosure at the midpoint m with half widths r: q = q(m) + grad . y + R(y), R = sum h_kl y_k y_l */
    tr_bigq *m[TR_REGION_MAX_DIMS], *r[TR_REGION_MAX_DIMS], *grad[TR_REGION_MAX_DIMS];
    tr_bigq *half = qfrac(c, 1, 2);
    for (int32_t k = 0; k < d; k++) {
        tr_bigq *s = qadd(c, C->lo[k], C->hi[k]), *w = qsub(c, C->hi[k], C->lo[k]);
        m[k] = qmul(c, s, half); r[k] = qmul(c, w, half);
        tr_bigq_free(s); tr_bigq_free(w);
        grad[k] = qcp(c, a[1 + k]);
    }
    tr_bigq *qm = qcp(c, a[0]), *lo = qint(c, 0), *hi = qint(c, 0);
    for (int32_t k = 0; k < d; k++) { tr_bigq *t = qmul(c, a[1 + k], m[k]); qset(&qm, qadd(c, qm, t)); tr_bigq_free(t); }
    int32_t idx = aff;
    for (int32_t k = 0; k < d && LIVE(c); k++)
        for (int32_t l = k; l < d && LIVE(c); l++, idx++) {
            const tr_bigq *h = in[idx];
            if (!h || !qsg(h)) continue;
            tr_bigq *mm = qmul(c, m[k], m[l]), *t = qmul(c, h, mm);
            qset(&qm, qadd(c, qm, t));
            tr_bigq_free(mm); tr_bigq_free(t);
            if (k == l) {
                tr_bigq *two = qint(c, 2), *t2 = qmul(c, two, h), *g2 = qmul(c, t2, m[k]);
                qset(&grad[k], qadd(c, grad[k], g2));
                tr_bigq *rr = qmul(c, r[k], r[k]), *e = qmul(c, h, rr);
                if (qsg(e) < 0) qset(&lo, qadd(c, lo, e)); else qset(&hi, qadd(c, hi, e));
                tr_bigq_free(two); tr_bigq_free(t2); tr_bigq_free(g2); tr_bigq_free(rr); tr_bigq_free(e);
            } else {
                tr_bigq *g1 = qmul(c, h, m[l]), *g2 = qmul(c, h, m[k]);
                qset(&grad[k], qadd(c, grad[k], g1)); qset(&grad[l], qadd(c, grad[l], g2));
                tr_bigq *rr = qmul(c, r[k], r[l]), *e = qmul(c, h, rr);
                if (qsg(e) < 0) qset(&e, qneg(c, e));
                qset(&hi, qadd(c, hi, e)); qset(&lo, qsub(c, lo, e));
                tr_bigq_free(g1); tr_bigq_free(g2); tr_bigq_free(rr); tr_bigq_free(e);
            }
        }
    /* L = q(m) - grad . m + (lo + hi) / 2 + grad . theta; eps = (hi - lo) / 2 */
    tr_bigq *mid = qadd(c, lo, hi), *spread = qsub(c, hi, lo);
    qset(&mid, qmul(c, mid, half)); qset(eps, qmul(c, spread, half));
    tr_bigq *a0 = qadd(c, qm, mid);
    for (int32_t k = 0; k < d; k++) {
        tr_bigq *t = qmul(c, grad[k], m[k]);
        qset(&a0, qsub(c, a0, t));
        tr_bigq_free(t);
        qset(&a[1 + k], qcp(c, grad[k]));
    }
    qset(&a[0], a0);
    for (int32_t k = 0; k < d; k++) { tr_bigq_free(m[k]); tr_bigq_free(r[k]); tr_bigq_free(grad[k]); }
    tr_bigq_free(half); tr_bigq_free(qm); tr_bigq_free(lo); tr_bigq_free(hi); tr_bigq_free(mid); tr_bigq_free(spread);
}
/* canonical integer form of the affine function a[0..d] ((A_1..A_d, A_0) / D, gcd 1, D > 0); negated for upper */
static void canonical(rctx *c, tr_bigq *const *a, int32_t d, int upper, tr_bigint **A, tr_bigint **D) {
    tr_bigint *L = zi(c, 1);
    for (int32_t t = 0; t <= d && LIVE(c); t++) zset(&L, zlcm(c, L, tr_bigq_denominator(a[t])));
    tr_bigint *f[MAXW + 1];
    memset(f, 0, sizeof f);
    for (int32_t t = 0; t <= d && LIVE(c); t++) {
        tr_bigint *q = zdivx(c, L, tr_bigq_denominator(a[t])), *v = zmul(c, q, tr_bigq_numerator(a[t]));
        f[t == 0 ? d : t - 1] = upper ? zneg(c, v) : zcp(c, v);
        tr_bigint_free(q); tr_bigint_free(v);
    }
    f[d + 1] = zcp(c, L);
    int z = 0;
    primitive(c, f, d + 2, &z);
    for (int32_t t = 0; t <= d; t++) A[t] = f[t];
    *D = f[d + 1];
    tr_bigint_free(L);
}
typedef struct { const tr_bigint *const *A; const tr_bigint *D; } formref_t;
typedef struct { formref_t *f; int32_t W; const uint64_t *grp; } gsort_ctx;
static int gsort_cmp(const void *vctx, uint64_t a, uint64_t b) {
    const gsort_ctx *x = (const gsort_ctx *)vctx;
    int r = forms_cmp((tr_bigint *const *)x->f[a].A, (tr_bigint *const *)x->f[b].A, x->W);
    if (r) return r;
    r = tr_bigint_cmp(x->f[a].D, x->f[b].D);
    if (r) return r;
    return x->grp[a] < x->grp[b] ? -1 : x->grp[a] > x->grp[b];
}
typedef struct { const class_t *k; } csort_ctx;
static int csort_cmp(const void *vctx, uint64_t a, uint64_t b) {
    const class_t *k = ((const csort_ctx *)vctx)->k;
    return k[a].mem[0] < k[b].mem[0] ? -1 : k[a].mem[0] > k[b].mem[0];
}

/* ================================================================== the candidate index of one cell */
typedef struct {
    uint32_t nc, nr, kk, W;
    const primes_t *P;
    uint64_t *G, *Hr, *Dr;              /* residues: class i, prime j: G[(i kk + j) W + t]; rows likewise; D[i kk + j] */
    int8_t *Gs, *Hs;                    /* signs of the theta coefficients: [i d + t] */
} kd_t;
static void kd_free(kd_t *K) { free(K->G); free(K->Hr); free(K->Dr); free(K->Gs); free(K->Hs); memset(K, 0, sizeof *K); }
/* the residues (k primes) of S over the box [tlo, thi] at its min (want_max 0) or max corner */
static void kd_corner(const kd_t *K, const uint64_t *R, const int8_t *sg, uint64_t i, const uint32_t *tlo,
                      const uint32_t *thi, int want_max, int32_t d, uint64_t *out) {
    for (uint32_t j = 0; j < K->kk; j++) {
        const tr_gfp *F = &K->P->F[j];
        const uint64_t *f = R + ((uint64_t)i * K->kk + j) * K->W;
        uint64_t s = f[d];
        for (int32_t t = 0; t < d; t++) {
            int8_t g = sg[(uint64_t)i * d + t];
            if (!g) continue;
            uint32_t at = (g > 0) == (want_max != 0) ? thi[t] : tlo[t];
            s = tr_gfp_add_hd(F, s, tr_gfp_mul_hd(F, f[t], (uint64_t)at % F->p));
        }
        out[j] = s;
    }
}
static int kd_sign(const kd_t *K, const uint64_t *r) { return tr_region_garner_sign(K->P->F, K->P->garner, K->P->half, K->kk, r); }
/* the sign of min over the box of D_b S_c - D_c S_b (the difference f_c - f_b times D_b D_c W > 0) */
static int kd_diff_min_sign(const kd_t *K, uint64_t ci, uint64_t bi, const uint32_t *tlo, const uint32_t *thi,
                            int32_t d) {
    const uint32_t kk = K->kk, W = K->W;
    uint64_t E[TR_REGION_PRIMES_MAX * TR_REGION_FORM], r[TR_REGION_PRIMES_MAX];
    const uint64_t *Dc = K->Dr + ci * kk, *Db = K->Dr + bi * kk;
    for (uint32_t j = 0; j < kk; j++) {
        const tr_gfp *F = &K->P->F[j];
        const uint64_t *gc = K->G + (ci * kk + j) * W, *gb = K->G + (bi * kk + j) * W;
        for (uint32_t t = 0; t < W; t++)
            E[j * W + t] = tr_gfp_sub_hd(F, tr_gfp_mul_hd(F, Db[j], gc[t]), tr_gfp_mul_hd(F, Dc[j], gb[t]));
    }
    int sg[TR_REGION_MAX_DIMS];
    for (int32_t t = 0; t < d; t++) {
        for (uint32_t j = 0; j < kk; j++) r[j] = E[j * W + (uint32_t)t];
        sg[t] = kd_sign(K, r);
    }
    for (uint32_t j = 0; j < kk; j++) {
        const tr_gfp *F = &K->P->F[j];
        uint64_t v = E[j * W + (uint32_t)d];
        for (int32_t t = 0; t < d; t++) {
            if (!sg[t]) continue;
            uint32_t at = sg[t] > 0 ? tlo[t] : thi[t];
            v = tr_gfp_add_hd(F, v, tr_gfp_mul_hd(F, E[j * W + (uint32_t)t], (uint64_t)at % F->p));
        }
        r[j] = v;
    }
    return kd_sign(K, r);
}
/* the candidates of the box among `in` (class ids local to the cell): out (may alias in), *nout. A class c is kept
 * unless f_c > f_b on the whole closed box for a reference class b: b1 minimizing max_B f, b2 minimizing f at a
 * point of the box (exact signs of affine differences at the box's corners): every class attaining the minimum
 * somewhere in B is kept */
static void kd_prune(rctx *c, const kd_t *K, int32_t d, const uint32_t *tlo, const uint32_t *thi, const uint64_t *in,
                     uint64_t nin, uint64_t *out, uint64_t *nout, uint64_t *mins, uint64_t *maxs) {
    *nout = 0;
    if (c->rc) return;
    uint64_t r[TR_REGION_PRIMES_MAX], hv[TR_REGION_PRIMES_MAX];
    const uint32_t kk = K->kk;
    for (uint32_t i = 0; i < K->nr; i++) {                /* a row of P excludes the whole box */
        kd_corner(K, K->Hr, K->Hs, i, tlo, thi, 0, d, hv);
        if (kd_sign(K, hv) > 0) return;
    }
    if (!nin) return;
    uint32_t mid[TR_REGION_MAX_DIMS];
    for (int32_t t = 0; t < d; t++) mid[t] = tlo[t] + (thi[t] - tlo[t]) / 2;
    uint64_t b1 = 0, b2 = 0;
    for (uint64_t q = 0; q < nin; q++) {
        kd_corner(K, K->G, K->Gs, in[q], tlo, thi, 1, d, maxs + q * kk);      /* max over the box */
        kd_corner(K, K->G, K->Gs, in[q], mid, mid, 0, d, mins + q * kk);      /* the value at the point */
        if (!q) continue;
        const uint64_t *Dq = K->Dr + in[q] * kk, *D1 = K->Dr + in[b1] * kk, *D2 = K->Dr + in[b2] * kk;
        for (uint32_t j = 0; j < kk; j++)
            r[j] = tr_gfp_sub_hd(&K->P->F[j], tr_gfp_mul_hd(&K->P->F[j], maxs[q * kk + j], D1[j]),
                                 tr_gfp_mul_hd(&K->P->F[j], maxs[b1 * kk + j], Dq[j]));
        if (kd_sign(K, r) < 0) b1 = q;
        for (uint32_t j = 0; j < kk; j++)
            r[j] = tr_gfp_sub_hd(&K->P->F[j], tr_gfp_mul_hd(&K->P->F[j], mins[q * kk + j], D2[j]),
                                 tr_gfp_mul_hd(&K->P->F[j], mins[b2 * kk + j], Dq[j]));
        if (kd_sign(K, r) < 0) b2 = q;
    }
    uint64_t n = 0;
    for (uint64_t q = 0; q < nin; q++) {
        if (q != b1 && kd_diff_min_sign(K, in[q], in[b1], tlo, thi, d) > 0) continue;
        if (q != b2 && b2 != b1 && kd_diff_min_sign(K, in[q], in[b2], tlo, thi, d) > 0) continue;
        out[n++] = in[q];
    }
    *nout = n;
}
static uint64_t node_new(rctx *c, tr_region_set *s) {
    count(c, TR_BUDGET_CIRCUIT, &c->nodes, 1);
    work(c, 1);
    charge(c, sizeof(node_t) + 16);
    if (c->rc) return NONE;
    if (s->nnodes == s->nodecap) {
        uint64_t nc = s->nodecap ? 2 * s->nodecap : 256;
        node_t *n = realloc(s->node, (size_t)nc * sizeof *n);
        if (!n) { fail(c, TR_MATH_MEMORY); return NONE; }
        s->node = n; s->nodecap = nc;
    }
    memset(&s->node[s->nnodes], 0, sizeof(node_t));
    return s->nnodes++;
}
static uint64_t cand_push(rctx *c, tr_region_set *s, const uint64_t *list, uint64_t n) {
    charge(c, n * 8);
    if (c->rc) return 0;
    if (s->ncand + n > s->candcap) {
        uint64_t nc = s->candcap ? s->candcap : 1024;
        while (nc < s->ncand + n) nc *= 2;
        uint64_t *x = realloc(s->cand, (size_t)nc * 8);
        if (!x) { fail(c, TR_MATH_MEMORY); return 0; }
        s->cand = x; s->candcap = nc;
    }
    uint64_t off = s->ncand;
    memcpy(s->cand + off, list, (size_t)n * 8);
    s->ncand += n;
    return off;
}
/* builds the subtree of the box with the parent's candidates `in` (cell-local), returns its node */
static uint64_t kd_build(rctx *c, tr_region_set *s, const kd_t *K, const cell_t *C, uint64_t cbase,
                         const uint32_t *tlo, const uint32_t *thi, const uint64_t *in, uint64_t nin, int stall,
                         uint64_t *scratch_min, uint64_t *scratch_max) {
    const int32_t d = s->d;
    uint64_t id = node_new(c, s);
    if (c->rc) return NONE;
    uint64_t *cand = zalloc((nin ? nin : 1) * 8);
    if (!cand) { fail(c, TR_MATH_MEMORY); return NONE; }
    uint64_t n = 0;
    kd_prune(c, K, d, tlo, thi, in, nin, cand, &n, scratch_min, scratch_max);
    node_t *nd = &s->node[id];
    for (int32_t k = 0; k < d; k++) { nd->tlo[k] = tlo[k]; nd->thi[k] = thi[k]; }
    int32_t axis = 0;
    uint32_t side = 0;
    for (int32_t k = 0; k < d; k++) if (thi[k] - tlo[k] > side) { side = thi[k] - tlo[k]; axis = k; }
    const int nstall = n < nin ? 0 : stall + 1;
    if (c->rc || n <= TR_REGION_LEAF || side <= 1 || nstall > d + 1) {
        if (LIVE(c)) {
            uint64_t *glob = zalloc((n ? n : 1) * 8);
            if (!glob) fail(c, TR_MATH_MEMORY);
            for (uint64_t q = 0; glob && q < n; q++) glob[q] = cbase + cand[q];
            uint64_t off = glob ? cand_push(c, s, glob, n) : 0;
            free(glob);
            nd = &s->node[id];
            nd->axis = -1; nd->link[0] = off; nd->link[1] = n;
        }
        free(cand);
        return id;
    }
    const uint32_t mid = tlo[axis] + side / 2;
    uint32_t a_hi[TR_REGION_MAX_DIMS], b_lo[TR_REGION_MAX_DIMS];
    for (int32_t k = 0; k < d; k++) { a_hi[k] = thi[k]; b_lo[k] = tlo[k]; }
    a_hi[axis] = mid; b_lo[axis] = mid;
    tr_bigint *split = zaddmul_u(c, C->base[axis], C->step[axis], mid);
    uint64_t left = kd_build(c, s, K, C, cbase, tlo, a_hi, cand, n, nstall, scratch_min, scratch_max);
    uint64_t right = kd_build(c, s, K, C, cbase, b_lo, thi, cand, n, nstall, scratch_min, scratch_max);
    free(cand);
    if (c->rc) { tr_bigint_free(split); return NONE; }
    nd = &s->node[id];
    nd->axis = axis; nd->link[0] = left; nd->link[1] = right; nd->split = split;
    return id;
}
/* the leaf of the cell's index containing the homogeneous point x (exact comparisons: theta_axis <= split: left) */
static uint64_t kd_locate(rctx *c, const tr_region_set *s, const cell_t *C, tr_bigint *const *x) {
    uint64_t node = C->root;
    const int32_t d = s->d;
    while (LIVE(c) && s->node[node].axis >= 0) {
        const node_t *nd = &s->node[node];
        tr_bigint *a = zmul(c, C->W, x[nd->axis]), *b = zmul(c, nd->split, x[d]), *e = zsub(c, a, b);
        int sg = zsg(e);
        tr_bigint_free(a); tr_bigint_free(b); tr_bigint_free(e);
        node = nd->link[sg <= 0 ? 0 : 1];
    }
    return node;
}

/* ================================================================== the certifying LP (OPT-002) */
/* max s subject to n_j theta + s <= c_j (rows: forms), s <= 1: the optimum s* (owned) and the point (d, owned) */
static void lp_margin(rctx *c, int32_t d, cons_t *const *rows, uint32_t m, tr_bigq **pt, tr_bigq **margin, int *sign) {
    *margin = NULL;
    for (int32_t k = 0; k < d; k++) pt[k] = NULL;
    if (c->rc) return;
    const int32_t n = d + 1;
    tr_bigq **A = zalloc((uint64_t)m * n * sizeof *A), **b = zalloc((uint64_t)(m ? m : 1) * sizeof *b);
    tr_bigq **g = zalloc((uint64_t)n * sizeof *g), **hi = zalloc((uint64_t)n * sizeof *hi);
    if (!A || !b || !g || !hi) { free(A); free(b); free(g); free(hi); fail(c, TR_MATH_MEMORY); return; }
    for (uint32_t j = 0; j < m && LIVE(c); j++) {
        for (int32_t k = 0; k < d; k++) A[(uint64_t)j * n + k] = zsg(rows[j]->a[k]) ? qzi(c, rows[j]->a[k]) : NULL;
        A[(uint64_t)j * n + d] = qint(c, 1);
        tr_bigint *cj = zneg(c, rows[j]->a[d]);
        b[j] = qzi(c, cj);
        tr_bigint_free(cj);
    }
    g[d] = qint(c, -1);
    hi[d] = qint(c, 1);
    tr_polyopt_qp *qp = NULL;
    tr_polyopt_result *res = NULL;
    /* the LP has d + 1 variables (the margin s): the declared dimension d is what the variables budget bounds */
    tr_math_budget lb = *c->b;
    if (lb.limit[TR_BUDGET_VARIABLES] < (uint64_t)n) lb.limit[TR_BUDGET_VARIABLES] = (uint64_t)n;
    if (LIVE(c)) {
        int32_t e = 0;
        qp = tr_polyopt_qp_create(n, NULL, (const tr_bigq *const *)g, NULL, (int32_t)m, (const tr_bigq *const *)A,
                                  (const tr_bigq *const *)b, 0, NULL, NULL, NULL, (const tr_bigq *const *)hi, &lb, c->r, &e);
        if (!qp) fail(c, e);
        else {
            res = tr_polyopt_qp_solve(qp, TR_POLYOPT_NO_UNIQUENESS, NULL, 0, NULL, 0, &lb, c->r, &e);
            if (!res) fail(c, e);
        }
    }
    if (res && LIVE(c)) {
        tr_polyopt_info info;
        tr_polyopt_result_get_info(res, &info);
        c->lps++; c->pivots += info.iterations;
        work(c, info.iterations);
        uint64_t cnt = 0;
        const tr_bigq *const *x = tr_polyopt_result_part(res, TR_POLYOPT_PART_POINT, &cnt);
        if (info.status != TR_POLYOPT_OPTIMAL || !x || cnt != (uint64_t)n) fail(c, TR_MATH_INVALID);
        else {
            for (int32_t k = 0; k < d; k++) pt[k] = qcp(c, x[k]);
            *margin = qcp(c, x[d]);
            *sign = qsg(x[d]);
        }
    }
    tr_polyopt_result_free(res);
    tr_polyopt_qp_free(qp);
    for (uint64_t i = 0; i < (uint64_t)m * n; i++) tr_bigq_free(A[i]);
    for (uint32_t j = 0; j < m; j++) tr_bigq_free(b[j]);
    for (int32_t k = 0; k < n; k++) { tr_bigq_free(g[k]); tr_bigq_free(hi[k]); }
    free(A); free(b); free(g); free(hi);
}

/* ================================================================== one class's region */
typedef struct { cons_t *k; uint32_t n; int32_t W; } csrt_ctx;
static int csrt_cmp(const void *vctx, uint64_t a, uint64_t b) {
    const csrt_ctx *x = (const csrt_ctx *)vctx;
    int r = forms_cmp(x->k[a].a, x->k[b].a, x->W);
    if (r) return r;
    return a < b ? -1 : a > b;
}
static region_t *region_new(rctx *c, tr_region_set *s) {
    count(c, TR_BUDGET_OUTPUT, &c->regions, 1);
    charge(c, sizeof(region_t));
    if (c->rc) return NULL;
    if (s->nregions == s->regcap) {
        uint64_t nc = s->regcap ? 2 * s->regcap : 64;
        region_t *n = realloc(s->reg, (size_t)nc * sizeof *n);
        if (!n) { fail(c, TR_MATH_MEMORY); return NULL; }
        s->reg = n; s->regcap = nc;
    }
    region_t *r = &s->reg[s->nregions++];
    memset(r, 0, sizeof *r);
    return r;
}
static void rcons_from(rctx *c, rcons_t *out, const cons_t *k, int32_t W, int eq) {
    memset(out, 0, sizeof *out);
    for (int32_t t = 0; t < W; t++) out->a[t] = zcp(c, k->a[t]);
    out->rhs = zneg(c, k->a[W - 1]);
    out->eq = eq;
    out->src = zalloc((uint64_t)(k->ns ? k->ns : 1) * sizeof(src_t));
    if (!out->src) { fail(c, TR_MATH_MEMORY); return; }
    memcpy(out->src, k->src, (size_t)k->ns * sizeof(src_t));
    out->ns = k->ns;
}
/* the region of class ci (global id), given the cell's constraints pc[0..mp) and polytope Pp */
static void class_region(rctx *c, tr_region_set *s, cell_t *C, uint64_t ci, const uint64_t *nb, uint64_t nnb,
                         cons_t *pc, uint32_t mp, const poly_t *Pp) {
    if (c->rc) return;
    const int32_t d = s->d, W = d + 1;
    const class_t *K = &s->cls[ci];
    /* the class constraints f_c <= f_c': D_c' A_c - D_c A_c' (primitive), deduplicated, merged with the walls */
    cons_t *kc = zalloc((nnb ? nnb : 1) * sizeof *kc);
    if (!kc) { fail(c, TR_MATH_MEMORY); return; }
    uint32_t nk = 0;
    int empty = 0;
    for (uint64_t q = 0; q < nnb && LIVE(c) && !empty; q++) {
        const class_t *O = &s->cls[nb[q]];
        cons_t *k = &kc[nk];
        for (int32_t t = 0; t < W && LIVE(c); t++) {
            tr_bigint *x = zmul(c, O->D, K->A[t]), *y = zmul(c, K->D, O->A[t]);
            k->a[t] = zsub(c, x, y);
            tr_bigint_free(x); tr_bigint_free(y);
        }
        int z = 0;
        primitive(c, k->a, W, &z);
        int zero_normal = 1;
        for (int32_t t = 0; t < d; t++) if (zsg(k->a[t])) zero_normal = 0;
        if (zero_normal) {                                 /* 0 <= -a_d: always (drop) or never (empty) */
            if (zsg(k->a[d]) > 0) empty = 1;
            cons_free(k);
            continue;
        }
        cons_addsrc(c, k, TR_REGION_SRC_CLASS, nb[q]);
        nk++;
    }
    uint64_t *ord = zalloc((uint64_t)(nk ? nk : 1) * 8);
    if (!ord) fail(c, TR_MATH_MEMORY);
    uint32_t m = mp;
    cons_t *all = NULL;
    if (LIVE(c) && !empty) {
        for (uint32_t i = 0; i < nk; i++) ord[i] = i;
        csrt_ctx sc = {kc, nk, W};
        if (nk > 1 && msort(ord, nk, csrt_cmp, &sc)) fail(c, TR_MATH_MEMORY);
        all = zalloc((uint64_t)(mp + nk) * sizeof *all);
        if (!all) fail(c, TR_MATH_MEMORY);
        for (uint32_t i = 0; all && i < mp && LIVE(c); i++) {          /* the walls: copies */
            for (int32_t t = 0; t < W; t++) all[i].a[t] = zcp(c, pc[i].a[t]);
            for (uint32_t q = 0; q < pc[i].ns; q++) cons_addsrc(c, &all[i], pc[i].src[q].kind, pc[i].src[q].index);
        }
        for (uint32_t i = 0; all && i < nk && LIVE(c); i++) {
            cons_t *k = &kc[ord[i]];
            if (m > mp && !forms_cmp(all[m - 1].a, k->a, W) && m - 1 >= mp) {   /* equal to the previous one */
                for (uint32_t q = 0; q < k->ns; q++) cons_addsrc(c, &all[m - 1], k->src[q].kind, k->src[q].index);
                continue;
            }
            uint32_t w = mp;
            for (uint32_t j = 0; j < mp; j++) if (!forms_cmp(all[j].a, k->a, W)) { w = j; break; }
            if (w < mp) { for (uint32_t q = 0; q < k->ns; q++) cons_addsrc(c, &all[w], k->src[q].kind, k->src[q].index); continue; }
            all[m] = *k;                                     /* moved */
            memset(k, 0, sizeof *k);
            m++;
        }
    }
    for (uint32_t i = 0; i < nk; i++) cons_free(&kc[i]);
    free(kc); free(ord);
    if (c->rc || empty) { for (uint32_t i = 0; all && i < m; i++) cons_free(&all[i]); free(all); return; }
    /* the vertices of Q_c */
    const uint32_t words = words_of(m);
    poly_t Q;
    poly_copy(c, &Q, Pp, words);
    for (uint32_t j = mp; j < m && LIVE(c) && Q.n; j++) poly_clip(c, &Q, all[j].a, j, d);
    charge(c, (uint64_t)Q.n * ((uint64_t)words * 8 + (uint64_t)W * 32));
    /* the certifying LP over every constraint (or, past TR_REGION_LP_ROWS, over those tight at a vertex) */
    tr_bigq *pt[TR_REGION_MAX_DIMS], *margin = NULL;
    int lsign = 0;
    uint32_t lrows = 0;
    memset(pt, 0, sizeof pt);
    if (LIVE(c)) {
        cons_t **rows = zalloc((uint64_t)m * sizeof *rows);
        if (!rows) fail(c, TR_MATH_MEMORY);
        if (rows && m <= TR_REGION_LP_ROWS) { for (uint32_t j = 0; j < m; j++) rows[j] = &all[j]; lrows = m; }
        else if (rows && Q.n) {
            for (uint32_t j = 0; j < m; j++) {
                int any = 0;
                for (uint32_t i = 0; i < Q.n && !any; i++) if (bit(Q.v[i].z, j)) any = 1;
                if (any) rows[lrows++] = &all[j];
            }
            if (lrows > TR_REGION_LP_ROWS) lrows = 0;
        }
        if (rows && lrows) lp_margin(c, d, rows, lrows, pt, &margin, &lsign);
        else if (LIVE(c)) c->unverified++;
        free(rows);
        if (LIVE(c) && lrows) {                            /* the LP and the vertex enumeration agree */
            if (!Q.n ? lsign >= 0 : lsign < 0) fail(c, TR_MATH_INVALID);
        }
    }
    int inside = 0;
    if (LIVE(c) && Q.n) {                                 /* one point of Q_c decides R_c = Q_c */
        uint64_t leaf = kd_locate(c, s, C, Q.v[0].x);
        const node_t *nd = &s->node[leaf];
        for (uint64_t q = 0; q < nd->link[1] && LIVE(c); q++) if (s->cand[nd->link[0] + q] == ci) inside = 1;
    }
    if (c->rc || !inside) {
        poly_free(&Q);
        for (int32_t k = 0; k < d; k++) tr_bigq_free(pt[k]);
        tr_bigq_free(margin);
        for (uint32_t i = 0; i < m; i++) cons_free(&all[i]);
        free(all);
        return;
    }
    /* the region: dimension, inequalities (facets / relative facets) and equalities */
    region_t *R = region_new(c, s);
    if (R) {
        R->cls = ci; R->cell = (uint64_t)(C - s->cells);
        R->lp_rows = lrows; R->lp_sign = lsign;
        for (int32_t k = 0; k < d; k++) { R->pt[k] = pt[k]; pt[k] = NULL; }
        R->margin = margin; margin = NULL;
        count(c, TR_BUDGET_SUPPORT, &c->incid, Q.n);
        R->nv = Q.n;
        R->lv = zalloc((uint64_t)Q.n * sizeof(vert_t));
        if (!R->lv) fail(c, TR_MATH_MEMORY);
        tr_bigint *const **vr = zalloc((uint64_t)Q.n * sizeof *vr);
        if (!vr) fail(c, TR_MATH_MEMORY);
        for (uint32_t i = 0; vr && i < Q.n; i++) vr[i] = (tr_bigint *const *)Q.v[i].x;
        int32_t rk = vr ? rank_rows(c, vr, Q.n, W, W) : 0;
        R->dim = rk - 1;
        R->k = zalloc((uint64_t)(m ? m : 1) * sizeof(rcons_t));
        R->kvert = zalloc((uint64_t)(m ? m : 1) * sizeof(uint64_t *));
        if (!R->k || !R->kvert) fail(c, TR_MATH_MEMORY);
        const uint32_t vw = words_of(Q.n);
        uint64_t *tb = zalloc((uint64_t)vw * 8);
        tr_bigint *const **sub = zalloc((uint64_t)Q.n * sizeof *sub);
        if (!tb || !sub) fail(c, TR_MATH_MEMORY);
        /* inequalities: the constraints tight on a face of dimension dim - 1 (one per distinct face) */
        for (uint32_t j = 0; j < m && LIVE(c); j++) {
            uint32_t nt = 0;
            memset(tb, 0, (size_t)vw * 8);
            for (uint32_t i = 0; i < Q.n; i++) if (bit(Q.v[i].z, j)) { sub[nt++] = vr[i]; setbit(tb, i); }
            if (nt == Q.n) continue;                      /* an implicit equality (or nothing): later */
            if (nt < (uint32_t)(R->dim > 0 ? R->dim : 1) || R->dim < 1) continue;
            if (rank_rows(c, sub, nt, W, R->dim) < R->dim) continue;
            uint32_t dup = R->nineq;
            for (uint32_t q = 0; q < R->nineq; q++) if (!memcmp(R->kvert[q], tb, (size_t)vw * 8)) { dup = q; break; }
            if (dup < R->nineq) {                          /* the same face: merge the sources */
                rcons_t *rk2 = &R->k[dup];
                for (uint32_t q = 0; q < all[j].ns; q++) {
                    int have = 0;
                    for (uint32_t t = 0; t < rk2->ns; t++) if (rk2->src[t].kind == all[j].src[q].kind && rk2->src[t].index == all[j].src[q].index) have = 1;
                    if (have) continue;
                    src_t *ns = realloc(rk2->src, (size_t)(rk2->ns + 1) * sizeof(src_t));
                    if (!ns) { fail(c, TR_MATH_MEMORY); break; }
                    rk2->src = ns; rk2->src[rk2->ns++] = all[j].src[q];
                }
                continue;
            }
            rcons_from(c, &R->k[R->nineq], &all[j], W, 0);
            R->kvert[R->nineq] = zalloc((uint64_t)vw * 8);
            if (!R->kvert[R->nineq]) { fail(c, TR_MATH_MEMORY); break; }
            memcpy(R->kvert[R->nineq], tb, (size_t)vw * 8);
            R->nineq++;
        }
        /* equalities: a basis of the constraints tight at every vertex (d - dim of them) */
        if (LIVE(c) && R->dim < d) {
            tr_bigint *const **eqs = zalloc((uint64_t)(m ? m : 1) * sizeof *eqs);
            if (!eqs) fail(c, TR_MATH_MEMORY);
            int32_t have = 0;
            for (uint32_t j = 0; eqs && j < m && LIVE(c) && have < d - R->dim; j++) {
                int all_tight = 1;
                for (uint32_t i = 0; i < Q.n && all_tight; i++) if (!bit(Q.v[i].z, j)) all_tight = 0;
                if (!all_tight) continue;
                eqs[have] = (tr_bigint *const *)all[j].a;
                if (rank_rows(c, eqs, (uint32_t)have + 1, W, have + 1) <= have) continue;
                rcons_from(c, &R->k[R->nineq + R->neq], &all[j], W, 1);
                R->neq++; have++;
            }
            free(eqs);
            if (LIVE(c) && have != d - R->dim) fail(c, TR_MATH_INVALID);
        }
        /* keep the vertices (moved) for the global merge */
        for (uint32_t i = 0; R->lv && i < Q.n; i++) { R->lv[i] = Q.v[i]; memset(&Q.v[i], 0, sizeof(vert_t)); }
        if (LIVE(c) && lrows && (R->dim == d) != (lsign > 0)) fail(c, TR_MATH_INVALID);   /* LP and vertices agree */
        free(tb); free(sub); free((void *)vr);
        uint64_t cb = (uint64_t)Q.n * ((uint64_t)W * 32 + 8) + (uint64_t)(R->nineq + R->neq) * ((uint64_t)W * 32 + 32);
        charge(c, cb);
    }
    poly_free(&Q);
    for (int32_t k = 0; k < d; k++) tr_bigq_free(pt[k]);
    tr_bigq_free(margin);
    for (uint32_t i = 0; i < m; i++) cons_free(&all[i]);
    free(all);
    if (R && LIVE(c)) s->cls[ci].region = s->nregions - 1;
}

/* ================================================================== one cell */
static void build_cell(rctx *c, tr_region_set *s, const tr_region_problem *p, uint64_t cell) {
    cell_t *C = &s->cells[cell];
    const int32_t d = s->d, W = d + 1;
    const uint64_t nc = C->class_end - C->class_begin;
    C->region_begin = s->nregions;
    /* the cell's constraints: its box faces and the rows; its polytope */
    const uint32_t mp = (uint32_t)(2 * d + s->nrows);
    cons_t *pc = zalloc((uint64_t)mp * sizeof *pc);
    if (!pc) { fail(c, TR_MATH_MEMORY); return; }
    for (int32_t k = 0; k < d && LIVE(c); k++) {
        box_face(c, C->lo[k], k, 0, d, pc[2 * k].a);
        cons_addsrc(c, &pc[2 * k], TR_REGION_SRC_LOWER, (uint64_t)k);
        box_face(c, C->hi[k], k, 1, d, pc[2 * k + 1].a);
        cons_addsrc(c, &pc[2 * k + 1], TR_REGION_SRC_UPPER, (uint64_t)k);
    }
    for (int32_t i = 0; i < s->nrows && LIVE(c); i++) {
        for (int32_t t = 0; t < W; t++) pc[2 * d + i].a[t] = zcp(c, s->walls[2 * d + i].a[t]);
        cons_addsrc(c, &pc[2 * d + i], TR_REGION_SRC_ROW, (uint64_t)i);
    }
    poly_t Pp;
    box_poly(c, C->lo, C->hi, d, words_of(mp), &Pp);
    for (int32_t i = 0; i < s->nrows && LIVE(c) && Pp.n; i++) poly_clip(c, &Pp, pc[2 * d + i].a, (uint32_t)(2 * d + i), d);
    /* the lattice of the index: x_k = base_k + step_k t_k over W = 2^L Qb */
    tr_bigint *Qb = zi(c, 1);
    tr_bigq *delta[TR_REGION_MAX_DIMS];
    for (int32_t k = 0; k < d; k++) {
        delta[k] = qsub(c, C->hi[k], C->lo[k]);
        if (LIVE(c)) zset(&Qb, zlcm(c, Qb, tr_bigq_denominator(C->lo[k])));
        if (LIVE(c)) zset(&Qb, zlcm(c, Qb, tr_bigq_denominator(delta[k])));
    }
    C->W = zshl(c, Qb, TR_REGION_LATTICE);
    for (int32_t k = 0; k < d && LIVE(c); k++) {
        tr_bigint *t = zdivx(c, C->W, tr_bigq_denominator(C->lo[k]));
        C->base[k] = zmul(c, t, tr_bigq_numerator(C->lo[k]));
        tr_bigint_free(t);
        t = zdivx(c, Qb, tr_bigq_denominator(delta[k]));
        C->step[k] = zmul(c, t, tr_bigq_numerator(delta[k]));
        tr_bigint_free(t);
    }
    for (int32_t k = 0; k < d; k++) tr_bigq_free(delta[k]);
    tr_bigint_free(Qb);
    /* lattice forms of the classes (G) and rows (H), their bound, the primes */
    kd_t K;
    memset(&K, 0, sizeof K);
    tr_bigint **G = zalloc((nc ? nc : 1) * (uint64_t)W * sizeof *G), **H = zalloc((uint64_t)(s->nrows ? s->nrows : 1) * W * sizeof *H);
    if (!G || !H) fail(c, TR_MATH_MEMORY);
    uint64_t bitsG = 0, bitsD = 0, bitsH = 0;
    for (int pass = 0; pass < 2 && LIVE(c); pass++) {
        const uint64_t cnt = pass == 0 ? nc : (uint64_t)s->nrows;
        for (uint64_t i = 0; i < cnt && LIVE(c); i++) {
            tr_bigint *const *f = pass == 0 ? s->cls[C->class_begin + i].A : pc[2 * d + i].a;
            tr_bigint **out = (pass == 0 ? G : H) + i * (uint64_t)W;
            tr_bigint *cst = zmul(c, f[d], C->W), *mag = NULL;
            for (int32_t k = 0; k < d && LIVE(c); k++) {
                out[k] = zmul(c, f[k], C->step[k]);
                tr_bigint *t = zmul(c, f[k], C->base[k]);
                zset(&cst, zadd(c, cst, t));
                tr_bigint_free(t);
            }
            out[d] = cst;
            mag = zabs(c, out[d]);
            for (int32_t k = 0; k < d && LIVE(c); k++) {
                tr_bigint *a = zabs(c, out[k]), *t = zshl(c, a, TR_REGION_LATTICE);
                zset(&mag, zadd(c, mag, t));
                tr_bigint_free(a); tr_bigint_free(t);
            }
            uint64_t bl = bitlen(mag);
            tr_bigint_free(mag);
            if (pass == 0) {
                if (bl > bitsG) bitsG = bl;
                if (bitlen(s->cls[C->class_begin + i].D) > bitsD) bitsD = bitlen(s->cls[C->class_begin + i].D);
            } else if (bl > bitsH) bitsH = bl;
        }
    }
    uint64_t B = bitsG + bitsD + 1;
    if (bitsH > B) B = bitsH;
    primes_t *P = zalloc(sizeof *P);
    if (!P) fail(c, TR_MATH_MEMORY);
    if (LIVE(c)) {
        uint64_t kk = (B + 2 + 60) / 61;
        int32_t e = kk > TR_REGION_PRIMES_MAX ? TR_MATH_OVERFLOW : primes_setup(c, (uint32_t)kk, P);
        if (e) fail(c, e);
    }
    if (LIVE(c)) {
        K.nc = (uint32_t)nc; K.nr = (uint32_t)s->nrows; K.kk = P->k; K.W = (uint32_t)W; K.P = P;
        K.G = zalloc(nc * K.kk * W * 8 + 8); K.Dr = zalloc(nc * K.kk * 8 + 8); K.Gs = zalloc(nc * d + 1);
        K.Hr = zalloc((uint64_t)K.nr * K.kk * W * 8 + 8); K.Hs = zalloc((uint64_t)K.nr * d + 1);
        if (!K.G || !K.Dr || !K.Gs || !K.Hr || !K.Hs) fail(c, TR_MATH_MEMORY);
        charge(c, (nc + K.nr) * (uint64_t)K.kk * (W + 1) * 8);
        for (uint64_t i = 0; LIVE(c) && i < nc; i++) {
            for (uint32_t j = 0; j < K.kk; j++) {
                for (int32_t t = 0; t < W; t++) K.G[(i * K.kk + j) * W + t] = zres(G[i * W + t], &P->F[j]);
                K.Dr[i * K.kk + j] = zres(s->cls[C->class_begin + i].D, &P->F[j]);
            }
            for (int32_t t = 0; t < d; t++) K.Gs[i * d + t] = (int8_t)zsg(G[i * W + t]);
        }
        for (uint64_t i = 0; LIVE(c) && i < K.nr; i++) {
            for (uint32_t j = 0; j < K.kk; j++)
                for (int32_t t = 0; t < W; t++) K.Hr[(i * K.kk + j) * W + t] = zres(H[i * W + t], &P->F[j]);
            for (int32_t t = 0; t < d; t++) K.Hs[i * d + t] = (int8_t)zsg(H[i * W + t]);
        }
    }
    for (uint64_t i = 0; G && i < nc * W; i++) tr_bigint_free(G[i]);
    for (uint64_t i = 0; H && i < (uint64_t)s->nrows * W; i++) tr_bigint_free(H[i]);
    free(G); free(H);
    /* the index */
    uint64_t *in = zalloc((nc ? nc : 1) * 8), *mins = zalloc((nc ? nc : 1) * TR_REGION_PRIMES_MAX * 8),
             *maxs = zalloc((nc ? nc : 1) * TR_REGION_PRIMES_MAX * 8);
    if (!in || !mins || !maxs) fail(c, TR_MATH_MEMORY);
    for (uint64_t i = 0; in && i < nc; i++) in[i] = i;
    uint32_t tlo[TR_REGION_MAX_DIMS], thi[TR_REGION_MAX_DIMS];
    for (int32_t k = 0; k < d; k++) { tlo[k] = 0; thi[k] = (uint32_t)1 << TR_REGION_LATTICE; }
    const uint64_t first_node = s->nnodes;
    if (LIVE(c)) C->root = kd_build(c, s, &K, C, C->class_begin, tlo, thi, in, nc, 0, mins, maxs);
    free(in); free(mins); free(maxs);
    kd_free(&K); free(P);
    /* the neighbors of each class: the co-candidates of the leaves where it is one */
    uint64_t **nb = zalloc((nc ? nc : 1) * sizeof *nb), *nnb = zalloc((nc ? nc : 1) * 8), *cap = zalloc((nc ? nc : 1) * 8);
    uint8_t *cand_any = zalloc(nc ? nc : 1);
    if (!nb || !nnb || !cap || !cand_any) fail(c, TR_MATH_MEMORY);
    for (uint64_t node = first_node; LIVE(c) && node < s->nnodes; node++) {
        const node_t *nd = &s->node[node];
        if (nd->axis >= 0) continue;
        const uint64_t *L = s->cand + nd->link[0], n = nd->link[1];
        charge(c, n * n * 8);
        for (uint64_t a = 0; a < n && LIVE(c); a++) {
            const uint64_t ci = L[a] - C->class_begin;
            cand_any[ci] = 1;
            if (nnb[ci] + n > cap[ci]) {
                uint64_t ncap = cap[ci] ? cap[ci] : 16;
                while (ncap < nnb[ci] + n) ncap *= 2;
                uint64_t *x = realloc(nb[ci], (size_t)ncap * 8);
                if (!x) { fail(c, TR_MATH_MEMORY); break; }
                nb[ci] = x; cap[ci] = ncap;
            }
            for (uint64_t q = 0; q < n; q++) if (q != a) nb[ci][nnb[ci]++] = L[q];
        }
    }
    for (uint64_t ci = 0; LIVE(c) && ci < nc; ci++) {        /* sort and deduplicate (insertion-free: counting by mark) */
        if (!nnb[ci]) continue;
        uint64_t *x = nb[ci], n = nnb[ci], w = 0;
        for (uint64_t i = 1; i < n; i++) {                   /* shell sort of class ids */
            uint64_t v = x[i], j = i;
            while (j > 0 && x[j - 1] > v) { x[j] = x[j - 1]; j--; }
            x[j] = v;
        }
        for (uint64_t i = 0; i < n; i++) if (!w || x[w - 1] != x[i]) x[w++] = x[i];
        nnb[ci] = w;
    }
    /* the regions, class by class */
    for (uint64_t ci = 0; LIVE(c) && ci < nc; ci++) {
        if (!cand_any[ci] || !Pp.n) continue;
        class_region(c, s, C, C->class_begin + ci, nb[ci], nnb[ci], pc, mp, &Pp);
    }
    C->region_end = s->nregions;
    for (uint64_t ci = 0; nb && ci < nc; ci++) free(nb[ci]);
    free(nb); free(nnb); free(cap); free(cand_any);
    poly_free(&Pp);
    for (uint32_t i = 0; i < mp; i++) cons_free(&pc[i]);
    free(pc);
}

/* ================================================================== global vertices and boundaries */
typedef struct { uint64_t reg, loc; } vref_t;
typedef struct { const tr_region_set *s; const vref_t *v; int32_t W; } vsort_ctx;
static int vsort_cmp(const void *vctx, uint64_t a, uint64_t b) {
    const vsort_ctx *x = (const vsort_ctx *)vctx;
    const region_t *ra = &x->s->reg[x->v[a].reg], *rb = &x->s->reg[x->v[b].reg];
    if (ra->cell != rb->cell) return ra->cell < rb->cell ? -1 : 1;
    int r = forms_cmp(ra->lv[x->v[a].loc].x, rb->lv[x->v[b].loc].x, x->W);
    if (r) return r;
    return x->v[a].reg < x->v[b].reg ? -1 : x->v[a].reg > x->v[b].reg ? 1 : (x->v[a].loc < x->v[b].loc ? -1 : x->v[a].loc > x->v[b].loc);
}
typedef struct { uint64_t cell, reg, ineq, nv; const uint64_t *vid; } fref_t;
static int fsort_cmp(const void *vctx, uint64_t a, uint64_t b) {
    const fref_t *f = (const fref_t *)vctx;
    if (f[a].cell != f[b].cell) return f[a].cell < f[b].cell ? -1 : 1;
    if (f[a].nv != f[b].nv) return f[a].nv < f[b].nv ? -1 : 1;
    for (uint64_t i = 0; i < f[a].nv; i++) if (f[a].vid[i] != f[b].vid[i]) return f[a].vid[i] < f[b].vid[i] ? -1 : 1;
    return f[a].reg < f[b].reg ? -1 : f[a].reg > f[b].reg ? 1 : (f[a].ineq < f[b].ineq ? -1 : f[a].ineq > f[b].ineq);
}
static void sort_u64(uint64_t *x, uint64_t n) {
    for (uint64_t i = 1; i < n; i++) { uint64_t v = x[i], j = i; while (j > 0 && x[j - 1] > v) { x[j] = x[j - 1]; j--; } x[j] = v; }
}
static void finish_global(rctx *c, tr_region_set *s) {
    const int32_t d = s->d, W = d + 1;
    uint64_t total = 0;
    for (uint64_t r = 0; r < s->nregions; r++) total += s->reg[r].nv;
    vref_t *v = zalloc((total ? total : 1) * sizeof *v);
    uint64_t *ord = zalloc((total ? total : 1) * 8);
    if (!v || !ord) { free(v); free(ord); fail(c, TR_MATH_MEMORY); return; }
    uint64_t at = 0;
    for (uint64_t r = 0; r < s->nregions; r++)
        for (uint64_t i = 0; i < s->reg[r].nv; i++) { v[at].reg = r; v[at].loc = i; ord[at] = at; at++; }
    vsort_ctx vc = {s, v, W};
    if (total > 1 && msort(ord, total, vsort_cmp, &vc)) fail(c, TR_MATH_MEMORY);
    /* distinct vertices */
    uint64_t nd = 0;
    for (uint64_t i = 0; i < total; i++) {
        const region_t *ra = &s->reg[v[ord[i]].reg];
        if (i) {
            const region_t *rb = &s->reg[v[ord[i - 1]].reg];
            if (ra->cell == rb->cell && !forms_cmp(ra->lv[v[ord[i]].loc].x, rb->lv[v[ord[i - 1]].loc].x, W)) continue;
        }
        nd++;
    }
    s->vert = zalloc((nd ? nd : 1) * sizeof(gvert_t));
    if (!s->vert) fail(c, TR_MATH_MEMORY);
    for (uint64_t r = 0; LIVE(c) && r < s->nregions; r++) {
        s->reg[r].vid = zalloc((s->reg[r].nv ? s->reg[r].nv : 1) * 8);
        if (!s->reg[r].vid) fail(c, TR_MATH_MEMORY);
    }
    uint64_t g = NONE;
    for (uint64_t i = 0; LIVE(c) && i < total; i++) {
        const vref_t *vi = &v[ord[i]];
        region_t *ra = &s->reg[vi->reg];
        int fresh = 1;
        if (i) {
            const region_t *rb = &s->reg[v[ord[i - 1]].reg];
            if (ra->cell == rb->cell && !forms_cmp(ra->lv[vi->loc].x, rb->lv[v[ord[i - 1]].loc].x, W)) fresh = 0;
        }
        if (fresh) {
            g = g == NONE ? 0 : g + 1;
            gvert_t *gv = &s->vert[g];
            gv->cell = ra->cell;
            for (int32_t t = 0; t < W; t++) gv->x[t] = zcp(c, ra->lv[vi->loc].x[t]);
            for (int32_t k = 0; k < d; k++) gv->pt[k] = qz(c, gv->x[k], gv->x[d]);
            s->nvert = g + 1;
        }
        ra->vid[vi->loc] = g;
    }
    /* tied classes per vertex: the regions having it (sorted by class) */
    uint64_t *cnt = zalloc((s->nvert ? s->nvert : 1) * 8);
    if (!cnt) fail(c, TR_MATH_MEMORY);
    for (uint64_t r = 0; LIVE(c) && r < s->nregions; r++) for (uint64_t i = 0; i < s->reg[r].nv; i++) cnt[s->reg[r].vid[i]]++;
    for (uint64_t q = 0; LIVE(c) && q < s->nvert; q++) {
        s->vert[q].cls = zalloc((cnt[q] ? cnt[q] : 1) * 8);
        if (!s->vert[q].cls) fail(c, TR_MATH_MEMORY);
    }
    for (uint64_t r = 0; LIVE(c) && r < s->nregions; r++)
        for (uint64_t i = 0; i < s->reg[r].nv; i++) {
            gvert_t *gv = &s->vert[s->reg[r].vid[i]];
            gv->cls[gv->ncls++] = s->reg[r].cls;
        }
    for (uint64_t q = 0; LIVE(c) && q < s->nvert; q++) {
        gvert_t *gv = &s->vert[q];
        sort_u64(gv->cls, gv->ncls);
        uint64_t w = 0;
        for (uint64_t i = 0; i < gv->ncls; i++) if (!w || gv->cls[w - 1] != gv->cls[i]) gv->cls[w++] = gv->cls[i];
        gv->ncls = w;
        gv->winner = NONE;
        for (uint64_t i = 0; i < w; i++) {
            uint64_t cw = s->cls[gv->cls[i]].winner;
            if (gv->winner == NONE || (s->tie == TR_REGION_TIE_SMALLEST ? cw < gv->winner : cw > gv->winner)) gv->winner = cw;
        }
        /* the walls of P through it */
        gv->walls = zalloc((s->nwalls ? s->nwalls : 1) * 8);
        if (!gv->walls) { fail(c, TR_MATH_MEMORY); break; }
        for (uint64_t wl = 0; wl < s->nwalls && LIVE(c); wl++) {
            tr_bigint *e = form_eval(c, s->walls[wl].a, gv->x, W);
            if (LIVE(c) && !zsg(e)) gv->walls[gv->nwalls++] = wl;
            tr_bigint_free(e);
        }
    }
    free(cnt);
    charge(c, s->nvert * ((uint64_t)W * 64 + 64));
    /* boundaries: the facets of the full-dimensional regions, each once */
    uint64_t nf = 0;
    for (uint64_t r = 0; r < s->nregions; r++) if (s->reg[r].dim == d) nf += s->reg[r].nineq;
    fref_t *fr = zalloc((nf ? nf : 1) * sizeof *fr);
    uint64_t *ford = zalloc((nf ? nf : 1) * 8);
    uint64_t **fv = zalloc((nf ? nf : 1) * sizeof *fv);
    if (!fr || !ford || !fv) fail(c, TR_MATH_MEMORY);
    uint64_t q = 0;
    for (uint64_t r = 0; LIVE(c) && r < s->nregions; r++) {
        const region_t *R = &s->reg[r];
        if (R->dim != d) continue;
        for (uint32_t j = 0; j < R->nineq && LIVE(c); j++) {
            uint64_t n = 0;
            for (uint64_t i = 0; i < R->nv; i++) if (bit(R->kvert[j], (uint32_t)i)) n++;
            fv[q] = zalloc((n ? n : 1) * 8);
            if (!fv[q]) { fail(c, TR_MATH_MEMORY); break; }
            n = 0;
            for (uint64_t i = 0; i < R->nv; i++) if (bit(R->kvert[j], (uint32_t)i)) fv[q][n++] = R->vid[i];
            sort_u64(fv[q], n);
            fr[q].cell = R->cell; fr[q].reg = r; fr[q].ineq = j; fr[q].nv = n; fr[q].vid = fv[q];
            ford[q] = q;
            q++;
        }
    }
    if (LIVE(c) && nf > 1 && msort(ford, nf, fsort_cmp, fr)) fail(c, TR_MATH_MEMORY);
    uint64_t nb = 0;
    for (uint64_t i = 0; LIVE(c) && i < nf; i++) {
        if (i) {
            const fref_t *a = &fr[ford[i - 1]], *b = &fr[ford[i]];
            if (a->cell == b->cell && a->nv == b->nv && !memcmp(a->vid, b->vid, (size_t)a->nv * 8)) continue;
        }
        nb++;
    }
    s->bnd = zalloc((nb ? nb : 1) * sizeof(bnd_t));
    if (!s->bnd) fail(c, TR_MATH_MEMORY);
    for (uint64_t i = 0; LIVE(c) && i < nf; i++) {
        const fref_t *f = &fr[ford[i]];
        if (i) {
            const fref_t *a = &fr[ford[i - 1]];
            if (a->cell == f->cell && a->nv == f->nv && !memcmp(a->vid, f->vid, (size_t)a->nv * 8)) continue;
        }
        bnd_t *B = &s->bnd[s->nbnd++];
        const region_t *R = &s->reg[f->reg];
        const rcons_t *k = &R->k[f->ineq];
        B->cell = f->cell; B->region = f->reg; B->nv = f->nv;
        B->vid = zalloc((f->nv ? f->nv : 1) * 8);
        if (!B->vid) { fail(c, TR_MATH_MEMORY); break; }
        memcpy(B->vid, f->vid, (size_t)f->nv * 8);
        for (int32_t t = 0; t < W; t++) B->a[t] = zcp(c, k->a[t]);
        B->rhs = zcp(c, k->rhs);
        /* the classes tied on the whole facet: those at every one of its vertices */
        const gvert_t *v0 = &s->vert[f->vid[0]];
        B->cls = zalloc((v0->ncls ? v0->ncls : 1) * 8);
        if (!B->cls) { fail(c, TR_MATH_MEMORY); break; }
        for (uint64_t a = 0; a < v0->ncls; a++) {
            int every = 1;
            for (uint64_t e = 1; e < f->nv && every; e++) {
                const gvert_t *ve = &s->vert[f->vid[e]];
                int found = 0;
                for (uint64_t t = 0; t < ve->ncls && !found; t++) if (ve->cls[t] == v0->cls[a]) found = 1;
                if (!found) every = 0;
            }
            if (every) B->cls[B->ncls++] = v0->cls[a];
        }
        B->winner = NONE;
        for (uint64_t a = 0; a < B->ncls; a++) {
            uint64_t cw = s->cls[B->cls[a]].winner;
            if (B->winner == NONE || (s->tie == TR_REGION_TIE_SMALLEST ? cw < B->winner : cw > B->winner)) B->winner = cw;
        }
        /* the kind: a wall of P (a face of the declared box, or a row), an internal cell face, or between regions */
        B->kind = TR_REGION_INNER;
        const cell_t *C = &s->cells[f->cell];
        for (uint32_t t = 0; t < k->ns; t++) {
            const src_t *sr = &k->src[t];
            if (sr->kind == TR_REGION_SRC_ROW) B->kind = TR_REGION_WALL;
            else if (sr->kind == TR_REGION_SRC_LOWER || sr->kind == TR_REGION_SRC_UPPER) {
                const int up = sr->kind == TR_REGION_SRC_UPPER;
                const uint64_t K = (uint64_t)(s->pieces > 1 ? s->pieces : 1);
                const int outer = up ? C->digit[sr->index] == K - 1 : C->digit[sr->index] == 0;
                if (outer) B->kind = TR_REGION_WALL;
                else if (B->kind != TR_REGION_WALL) B->kind = TR_REGION_CELL;
            }
        }
    }
    charge(c, s->nbnd * ((uint64_t)W * 32 + 96));
    for (uint64_t i = 0; fv && i < nf; i++) free(fv[i]);
    free(fv); free(fr); free(ford);
    free(v); free(ord);
    /* the local vertices and scratch are no longer needed */
    for (uint64_t r = 0; r < s->nregions; r++) {
        region_t *R = &s->reg[r];
        if (R->lv) { for (uint64_t i = 0; i < R->nv; i++) vert_free(&R->lv[i]); free(R->lv); R->lv = NULL; }
        if (R->kvert) { for (uint32_t i = 0; i < R->nineq; i++) free(R->kvert[i]); free(R->kvert); R->kvert = NULL; }
    }
}

/* ================================================================== the build */
static tr_region_set *build(const tr_region_problem *p, const tr_math_budget *budget, tr_math_refusal *refusal,
                            int32_t *error) {
    tr_budget_none(refusal);
    int quad = 0;
    int32_t rc = problem_check(p, &quad);
    if (rc) { set_error(error, rc); return NULL; }
    tr_math_budget dflt;
    budget = budget_or(budget, &dflt);
    tr_math_need need;
    tr_region_need(p, &need);
    if ((rc = tr_budget_check(budget, &need, refusal))) { set_error(error, rc); return NULL; }
    if (quad && p->contract != TR_REGION_PIECEWISE_LINEAR) { set_error(error, TR_MATH_UNSUPPORTED); return NULL; }
    rctx cx, *c = &cx;
    memset(c, 0, sizeof cx);
    c->b = budget; c->r = refusal;
    charge(c, need.need[TR_BUDGET_BYTES]);
    tr_region_set *s = zalloc(sizeof *s);
    if (!s) { set_error(error, TR_MATH_MEMORY); return NULL; }
    s->magic = SET_MAGIC;
    const int32_t d = p->d, W = d + 1;
    s->d = d; s->upper = p->upper; s->tie = p->tie; s->contract = p->contract; s->nrows = p->nrows;
    s->pieces = p->contract == TR_REGION_PIECEWISE_LINEAR ? p->pieces : 1;
    s->claim = TR_REGION_CLAIM_EXACT;                 /* APPROXIMATE once an enclosure error is positive */
    s->ngroups = p->ngroups; s->present = present_of(p);
    s->degree = 0;
    for (int32_t k = 0; k < d; k++) { s->lo[k] = qcp(c, p->lo[k]); s->hi[k] = qcp(c, p->hi[k]); }
    if (p->nrows) {
        s->A = zalloc((uint64_t)p->nrows * d * sizeof *s->A); s->b = zalloc((uint64_t)p->nrows * sizeof *s->b);
        if (!s->A || !s->b) fail(c, TR_MATH_MEMORY);
        for (int64_t i = 0; LIVE(c) && i < (int64_t)p->nrows * d; i++) s->A[i] = p->A[i] ? qcp(c, p->A[i]) : qint(c, 0);
        for (int32_t i = 0; LIVE(c) && i < p->nrows; i++) s->b[i] = qcp(c, p->b[i]);
    }
    if (p->contract == TR_REGION_PIECEWISE_LINEAR) s->budget_err = qcp(c, p->error_budget);
    /* the walls of P */
    s->nwalls = (uint64_t)(2 * d + p->nrows);
    for (int32_t k = 0; k < d && LIVE(c); k++) {
        box_face(c, s->lo[k], k, 0, d, s->walls[2 * k].a);
        cons_addsrc(c, &s->walls[2 * k], TR_REGION_SRC_LOWER, (uint64_t)k);
        box_face(c, s->hi[k], k, 1, d, s->walls[2 * k + 1].a);
        cons_addsrc(c, &s->walls[2 * k + 1], TR_REGION_SRC_UPPER, (uint64_t)k);
    }
    for (int32_t i = 0; i < p->nrows && LIVE(c); i++) {
        form_from_q(c, (const tr_bigq *const *)(s->A + (uint64_t)i * d), s->b[i], d, s->walls[2 * d + i].a);
        cons_addsrc(c, &s->walls[2 * d + i], TR_REGION_SRC_ROW, (uint64_t)i);
        int zero_normal = 1;
        for (int32_t k = 0; LIVE(c) && k < d; k++) if (zsg(s->walls[2 * d + i].a[k])) zero_normal = 0;
        if (LIVE(c) && zero_normal) fail(c, TR_MATH_INVALID);           /* a row without coefficients */
    }
    /* P has an interior point: max s with every wall + s <= its bound (exact LP) */
    if (LIVE(c)) {
        cons_t *rows[2 * TR_REGION_MAX_DIMS + TR_REGION_MAX_ROWS];
        uint32_t m = 0;
        for (uint64_t i = 0; i < s->nwalls; i++) {
            int zero_normal = 1;
            for (int32_t k = 0; k < d; k++) if (zsg(s->walls[i].a[k])) zero_normal = 0;
            if (!zero_normal) rows[m++] = &s->walls[i];
        }
        tr_bigq *pt[TR_REGION_MAX_DIMS], *mg = NULL;
        int sg = 0;
        lp_margin(c, d, rows, m, pt, &mg, &sg);
        for (int32_t k = 0; k < d; k++) tr_bigq_free(pt[k]);
        tr_bigq_free(mg);
        if (LIVE(c) && sg < 0) fail(c, TR_MATH_INFEASIBLE);
        else if (LIVE(c) && sg == 0) fail(c, TR_MATH_UNSUPPORTED);
    }
    /* cells */
    const uint64_t K = (uint64_t)s->pieces;
    s->ncells = cells_of(p);
    s->cells = zalloc(s->ncells * sizeof(cell_t));
    if (!s->cells) fail(c, TR_MATH_MEMORY);
    for (uint64_t ce = 0; LIVE(c) && ce < s->ncells; ce++) {
        cell_t *C = &s->cells[ce];
        uint64_t rem = ce;
        for (int32_t k = d; k-- > 0;) { C->digit[k] = rem % K; rem /= K; }
        for (int32_t k = 0; k < d && LIVE(c); k++) {
            tr_bigq *w = qsub(c, s->hi[k], s->lo[k]), *ka = qint(c, (int64_t)K), *step = qdiv(c, w, ka);
            tr_bigq *j0 = qint(c, (int64_t)C->digit[k]), *j1 = qint(c, (int64_t)C->digit[k] + 1);
            tr_bigq *o0 = qmul(c, step, j0), *o1 = qmul(c, step, j1);
            C->lo[k] = C->digit[k] == 0 ? qcp(c, s->lo[k]) : qadd(c, s->lo[k], o0);
            C->hi[k] = C->digit[k] == K - 1 ? qcp(c, s->hi[k]) : qadd(c, s->lo[k], o1);
            tr_bigq_free(w); tr_bigq_free(ka); tr_bigq_free(step); tr_bigq_free(j0); tr_bigq_free(j1);
            tr_bigq_free(o0); tr_bigq_free(o1);
        }
    }
    /* the classes of every cell */
    const uint64_t G = p->ngroups;
    tr_bigq **fa = zalloc((G ? G : 1) * (uint64_t)W * sizeof *fa), **feps = zalloc((G ? G : 1) * sizeof *feps);
    tr_bigint **fA = zalloc((G ? G : 1) * (uint64_t)W * sizeof *fA), **fD = zalloc((G ? G : 1) * sizeof *fD);
    int32_t *fdeg = zalloc((G ? G : 1) * 4);
    uint64_t *grp = zalloc((G ? G : 1) * 8), *ord = zalloc((G ? G : 1) * 8);
    formref_t *fr = zalloc((G ? G : 1) * sizeof *fr);
    if (!fa || !feps || !fA || !fD || !fdeg || !grp || !ord || !fr) fail(c, TR_MATH_MEMORY);
    uint64_t total_classes = 0, cap_classes = 0;
    for (uint64_t ce = 0; LIVE(c) && ce < s->ncells; ce++) {
        cell_t *C = &s->cells[ce];
        uint64_t n = 0;
        for (uint64_t g = 0; LIVE(c) && g < G; g++) {
            if (p->present && !p->present[g]) continue;
            cell_function(c, p, g, C, fa + n * W, &feps[n], &fdeg[n]);
            canonical(c, fa + n * W, d, s->upper, fA + n * W, &fD[n]);
            if (fdeg[n] > s->degree) s->degree = fdeg[n];
            if (LIVE(c) && s->budget_err && tr_bigq_cmp(feps[n], s->budget_err) > 0) fail(c, TR_MATH_NOT_CONVERGED);
            if (LIVE(c) && (!s->eps || tr_bigq_cmp(feps[n], s->eps) > 0)) qset(&s->eps, qcp(c, feps[n]));
            grp[n] = g;
            fr[n].A = (const tr_bigint *const *)(fA + n * W); fr[n].D = fD[n];
            ord[n] = n;
            n++;
        }
        gsort_ctx gc = {fr, W, grp};
        if (LIVE(c) && n > 1 && msort(ord, n, gsort_cmp, &gc)) fail(c, TR_MATH_MEMORY);
        /* count the classes of this cell */
        uint64_t ncls = 0;
        for (uint64_t i = 0; i < n; i++)
            if (!i || forms_cmp(fA + ord[i - 1] * W, fA + ord[i] * W, W) || tr_bigint_cmp(fD[ord[i - 1]], fD[ord[i]])) ncls++;
        if (total_classes + ncls > cap_classes && LIVE(c)) {
            uint64_t nc = cap_classes ? cap_classes : 64;
            while (nc < total_classes + ncls) nc *= 2;
            class_t *x = realloc(s->cls, (size_t)nc * sizeof *x);
            if (!x) fail(c, TR_MATH_MEMORY);
            else { s->cls = x; cap_classes = nc; }
        }
        C->class_begin = total_classes;
        uint64_t first = total_classes;
        for (uint64_t i = 0; LIVE(c) && i < n; i++) {
            const uint64_t a = ord[i];
            int fresh = !i || forms_cmp(s->cls[total_classes - 1].A, fA + a * W, W) ||
                        tr_bigint_cmp(s->cls[total_classes - 1].D, fD[a]);
            if (fresh) {
                class_t *Kc = &s->cls[total_classes++];
                memset(Kc, 0, sizeof *Kc);
                s->nclasses = total_classes;
                Kc->cell = ce; Kc->region = NONE; Kc->degree = fdeg[a];
                for (int32_t t = 0; t < W; t++) { Kc->A[t] = fA[a * W + t]; fA[a * W + t] = NULL; Kc->coef[t] = fa[a * W + t]; fa[a * W + t] = NULL; }
                Kc->D = fD[a]; fD[a] = NULL;
                Kc->eps = feps[a]; feps[a] = NULL;
                Kc->mem = zalloc(8 * 8);
            }
            class_t *Kc = &s->cls[total_classes - 1];
            if (!Kc->mem) { fail(c, TR_MATH_MEMORY); break; }
            if (Kc->nmem && (Kc->nmem & (Kc->nmem - 1)) == 0 && Kc->nmem >= 8) {
                uint64_t *x = realloc(Kc->mem, (size_t)(2 * Kc->nmem) * 8);
                if (!x) { fail(c, TR_MATH_MEMORY); break; }
                Kc->mem = x;
            }
            Kc->mem[Kc->nmem++] = grp[a];
            if (Kc->degree < fdeg[a]) Kc->degree = fdeg[a];
        }
        for (uint64_t i = 0; i < n; i++) {
            for (int32_t t = 0; t < W; t++) { tr_bigint_free(fA[i * W + t]); fA[i * W + t] = NULL; tr_bigq_free(fa[i * W + t]); fa[i * W + t] = NULL; }
            tr_bigint_free(fD[i]); fD[i] = NULL; tr_bigq_free(feps[i]); feps[i] = NULL;
        }
        /* members sorted, the tie policy's member, classes ordered by their smallest member */
        for (uint64_t q = first; LIVE(c) && q < total_classes; q++) {
            class_t *Kc = &s->cls[q];
            sort_u64(Kc->mem, Kc->nmem);
            Kc->winner = s->tie == TR_REGION_TIE_SMALLEST ? Kc->mem[0] : Kc->mem[Kc->nmem - 1];
        }
        if (LIVE(c) && total_classes - first > 1) {
            uint64_t m = total_classes - first;
            uint64_t *co = zalloc(m * 8);
            class_t *tmp = zalloc(m * sizeof *tmp);
            if (!co || !tmp) fail(c, TR_MATH_MEMORY);
            else {
                for (uint64_t q = 0; q < m; q++) co[q] = q;
                csort_ctx cc = {s->cls + first};
                if (msort(co, m, csort_cmp, &cc)) fail(c, TR_MATH_MEMORY);
                for (uint64_t q = 0; q < m; q++) tmp[q] = s->cls[first + co[q]];
                memcpy(s->cls + first, tmp, (size_t)m * sizeof *tmp);
            }
            free(co); free(tmp);
        }
        C->class_end = total_classes;
    }
    free(fa); free(feps); free(fA); free(fD); free(fdeg); free(grp); free(ord); free(fr);
    if (LIVE(c) && s->contract == TR_REGION_PIECEWISE_LINEAR && !s->eps) s->eps = qint(c, 0);
    if (LIVE(c) && s->eps && qsg(s->eps) > 0) s->claim = TR_REGION_CLAIM_APPROXIMATE;
    if (s->eps) { tr_bigq *two = qint(c, 2); s->env_err = qmul(c, s->eps, two); tr_bigq_free(two); }
    s->kind = s->present ? TR_REGION_FINITE : TR_REGION_EMPTY;
    /* the regions of every cell */
    for (uint64_t ce = 0; LIVE(c) && s->kind == TR_REGION_FINITE && ce < s->ncells; ce++) build_cell(c, s, p, ce);
    if (s->kind == TR_REGION_EMPTY) for (uint64_t ce = 0; ce < s->ncells; ce++) s->cells[ce].root = NONE;
    if (LIVE(c) && s->kind == TR_REGION_FINITE) finish_global(c, s);
    if (c->rc) { tr_region_free(s); set_error(error, c->rc); return NULL; }
    tr_region_info *in = &s->info;
    memset(in, 0, sizeof *in);
    in->groups = s->ngroups; in->present = s->present; in->classes = s->nclasses; in->cells = s->ncells;
    in->regions = s->nregions;
    for (uint64_t r = 0; r < s->nregions; r++) { if (s->reg[r].dim == d) in->full++; else in->lower++; in->incidences += s->reg[r].nv; }
    in->vertices = s->nvert; in->boundaries = s->nbnd; in->nodes = s->nnodes; in->candidates = s->ncand;
    for (uint64_t q = 0; q < s->nnodes; q++) if (s->node[q].axis < 0) in->leaves++;
    in->lps = c->lps; in->pivots = c->pivots; in->clips = c->clips; in->iterations = c->iters; in->bytes = c->maxbytes;
    in->d = d; in->kind = s->kind; in->upper = s->upper; in->tie = s->tie; in->claim = s->claim; in->contract = s->contract;
    in->pieces = s->pieces; in->degree = s->degree; in->nrows = s->nrows; in->unverified = (int32_t)c->unverified;
    set_error(error, TR_MATH_OK);
    return s;
}
tr_region_set *tr_region_build(const tr_region_problem *p, const tr_math_budget *budget, tr_math_refusal *refusal,
                               int32_t *error) {
    return build(p, budget, refusal, error);
}
tr_region_set *tr_region_from_polys(const tr_region_problem *pr, const tr_math_poly *const *polys, const int32_t *vars,
                                    const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    if (!pr || (pr->ngroups && !polys) || !vars || pr->d < 1) { set_error(error, TR_MATH_INVALID); return NULL; }
    if (pr->d > TR_REGION_MAX_DIMS) { set_error(error, TR_MATH_UNSUPPORTED); return NULL; }
    const int32_t d = pr->d, ncoef = 1 + d + d * (d + 1) / 2;
    for (int32_t k = 0; k < d; k++) {
        if (vars[k] < 0 || vars[k] >= TR_MATH_VARIABLES) { set_error(error, TR_MATH_INVALID); return NULL; }
        for (int32_t l = 0; l < k; l++) if (vars[l] == vars[k]) { set_error(error, TR_MATH_INVALID); return NULL; }
    }
    tr_math_budget dflt;
    budget = budget_or(budget, &dflt);
    const uint64_t G = pr->ngroups;
    tr_bigq **coef = zalloc(tr_budget_mul(tr_budget_mul(G ? G : 1, (uint64_t)ncoef), sizeof(void *)));
    uint8_t *present = zalloc(G ? G : 1);
    int32_t rc = coef && present ? TR_MATH_OK : TR_MATH_MEMORY;
    for (uint64_t g = 0; g < G && !rc; g++) {
        if (!polys[g]) continue;
        present[g] = 1;
        int32_t terms = tr_math_poly_term_count(polys[g]);
        if (terms < 0) { rc = TR_MATH_INVALID; break; }
        for (int32_t i = 0; i < terms && !rc; i++) {
            tr_math_term term;
            if (tr_math_poly_get_term(polys[g], i, &term)) { rc = TR_MATH_INVALID; break; }
            int32_t deg = 0, pos[2] = {-1, -1};
            for (int32_t v = 0; v < TR_MATH_VARIABLES && !rc; v++) {
                if (!term.exponents[v]) continue;
                int32_t k = -1;
                for (int32_t q = 0; q < d; q++) if (vars[q] == v) k = q;
                if (k < 0) { rc = TR_MATH_DOMAIN_ERROR; break; }   /* a variable outside the parameters */
                for (uint32_t e = 0; e < term.exponents[v]; e++) { if (deg < 2) pos[deg] = k; deg++; }
            }
            if (rc) break;
            if (deg > 2) {
                rc = tr_budget_check_one(budget, TR_BUDGET_DEGREE, (uint64_t)deg, refusal);
                if (!rc) rc = TR_MATH_UNSUPPORTED;
                break;
            }
            int32_t slot = 0;
            if (deg == 1) slot = 1 + pos[0];
            else if (deg == 2) {
                int32_t k = pos[0] < pos[1] ? pos[0] : pos[1], l = pos[0] < pos[1] ? pos[1] : pos[0];
                slot = 1 + d;
                for (int32_t a = 0; a < k; a++) slot += d - a;
                slot += l - k;
            }
            int32_t e = 0;
            tr_num_rational q = {term.coefficient.numerator, term.coefficient.denominator};
            tr_bigq *v = tr_bigq_from_q64(q, budget, refusal, &e);
            if (!v) { rc = e ? e : TR_MATH_MEMORY; break; }
            uint64_t at = g * (uint64_t)ncoef + (uint64_t)slot;
            if (coef[at]) {
                tr_bigq *sum = tr_bigq_add(coef[at], v, budget, refusal, &e);
                tr_bigq_free(v);
                if (!sum) { rc = e ? e : TR_MATH_MEMORY; break; }
                v = sum;
            }
            tr_bigq_free(coef[at]); coef[at] = v;
        }
    }
    tr_region_set *s = NULL;
    if (!rc) {
        tr_region_problem q = *pr;
        q.ncoef = ncoef; q.coef = (const tr_bigq *const *)coef; q.present = present;
        s = build(&q, budget, refusal, error);
    } else set_error(error, rc);
    for (uint64_t i = 0; coef && i < G * (uint64_t)ncoef; i++) tr_bigq_free(coef[i]);
    free(coef); free(present);
    return s;
}

/* ================================================================== accessors */
int32_t tr_region_get_info(const tr_region_set *s, tr_region_info *out) {
    if (!set_ok(s) || !out) return TR_MATH_INVALID;
    *out = s->info;
    return 0;
}
const tr_bigq *tr_region_bound(const tr_region_set *s, int32_t hi, int32_t k) {
    if (!set_ok(s) || k < 0 || k >= s->d) return NULL;
    return hi ? s->hi[k] : s->lo[k];
}
const tr_bigq *tr_region_error(const tr_region_set *s, int32_t which) {
    if (!set_ok(s)) return NULL;
    return which == 0 ? s->eps : which == 1 ? s->env_err : which == 2 ? s->budget_err : NULL;
}
int32_t tr_region_class_get(const tr_region_set *s, uint64_t cls, tr_region_class_info *out) {
    if (!set_ok(s) || !out || cls >= s->nclasses) return TR_MATH_INVALID;
    const class_t *K = &s->cls[cls];
    memset(out, 0, sizeof *out);
    out->cell = K->cell; out->members = K->nmem; out->winner = K->winner; out->region = K->region;
    out->degree = K->degree; out->exact = !qsg(K->eps);
    return 0;
}
int32_t tr_region_class_members(const tr_region_set *s, uint64_t cls, uint64_t *out, uint64_t capacity) {
    if (!set_ok(s) || cls >= s->nclasses || (!out && s->cls[cls].nmem) || capacity < s->cls[cls].nmem) return TR_MATH_INVALID;
    memcpy(out, s->cls[cls].mem, (size_t)s->cls[cls].nmem * 8);
    return 0;
}
const tr_bigq *tr_region_class_coef(const tr_region_set *s, uint64_t cls, int32_t k) {
    if (!set_ok(s) || cls >= s->nclasses || k < 0 || k > s->d) return NULL;
    return s->cls[cls].coef[k];
}
const tr_bigq *tr_region_class_eps(const tr_region_set *s, uint64_t cls) {
    if (!set_ok(s) || cls >= s->nclasses) return NULL;
    return s->cls[cls].eps;
}
int32_t tr_region_region_get(const tr_region_set *s, uint64_t region, tr_region_region_info *out) {
    if (!set_ok(s) || !out || region >= s->nregions) return TR_MATH_INVALID;
    const region_t *R = &s->reg[region];
    const class_t *K = &s->cls[R->cls];
    memset(out, 0, sizeof *out);
    out->cls = R->cls; out->cell = R->cell; out->winner = K->winner; out->members = K->nmem;
    out->vertices = R->nv; out->inequalities = R->nineq; out->equalities = R->neq; out->lp_rows = R->lp_rows;
    out->dim = R->dim; out->full = R->dim == s->d; out->lp_sign = R->lp_sign;
    return 0;
}
int32_t tr_region_region_vertices(const tr_region_set *s, uint64_t region, uint64_t *out, uint64_t capacity) {
    if (!set_ok(s) || region >= s->nregions || capacity < s->reg[region].nv || (!out && s->reg[region].nv)) return TR_MATH_INVALID;
    memcpy(out, s->reg[region].vid, (size_t)s->reg[region].nv * 8);
    return 0;
}
int32_t tr_region_region_constraint(const tr_region_set *s, uint64_t region, uint64_t i, const tr_bigint **normal,
                                    const tr_bigint **rhs, int32_t *equality, uint64_t *nsources) {
    if (!set_ok(s) || region >= s->nregions) return TR_MATH_INVALID;
    const region_t *R = &s->reg[region];
    if (i >= (uint64_t)R->nineq + R->neq) return TR_MATH_INVALID;
    const rcons_t *k = &R->k[i];
    for (int32_t t = 0; normal && t < s->d; t++) normal[t] = k->a[t];
    if (rhs) *rhs = k->rhs;
    if (equality) *equality = k->eq;
    if (nsources) *nsources = k->ns;
    return 0;
}
int32_t tr_region_region_sources(const tr_region_set *s, uint64_t region, uint64_t i, int32_t *kinds, uint64_t *index,
                                 uint64_t capacity) {
    if (!set_ok(s) || region >= s->nregions) return TR_MATH_INVALID;
    const region_t *R = &s->reg[region];
    if (i >= (uint64_t)R->nineq + R->neq || capacity < R->k[i].ns || ((!kinds || !index) && R->k[i].ns)) return TR_MATH_INVALID;
    for (uint32_t q = 0; q < R->k[i].ns; q++) { kinds[q] = R->k[i].src[q].kind; index[q] = R->k[i].src[q].index; }
    return 0;
}
int32_t tr_region_region_lp(const tr_region_set *s, uint64_t region, const tr_bigq **point, const tr_bigq **margin) {
    if (!set_ok(s) || region >= s->nregions) return TR_MATH_INVALID;
    const region_t *R = &s->reg[region];
    for (int32_t k = 0; point && k < s->d; k++) point[k] = R->pt[k];
    if (margin) *margin = R->margin;
    return 0;
}
int32_t tr_region_vertex_get(const tr_region_set *s, uint64_t vertex, tr_region_vertex_info *out) {
    if (!set_ok(s) || !out || vertex >= s->nvert) return TR_MATH_INVALID;
    const gvert_t *v = &s->vert[vertex];
    out->cell = v->cell; out->classes = v->ncls; out->winner = v->winner; out->walls = v->nwalls;
    return 0;
}
int32_t tr_region_vertex_point(const tr_region_set *s, uint64_t vertex, const tr_bigq **coords) {
    if (!set_ok(s) || !coords || vertex >= s->nvert) return TR_MATH_INVALID;
    for (int32_t k = 0; k < s->d; k++) coords[k] = s->vert[vertex].pt[k];
    return 0;
}
int32_t tr_region_vertex_classes(const tr_region_set *s, uint64_t vertex, uint64_t *out, uint64_t capacity) {
    if (!set_ok(s) || vertex >= s->nvert || capacity < s->vert[vertex].ncls || !out) return TR_MATH_INVALID;
    memcpy(out, s->vert[vertex].cls, (size_t)s->vert[vertex].ncls * 8);
    return 0;
}
int32_t tr_region_vertex_walls(const tr_region_set *s, uint64_t vertex, uint64_t *out, uint64_t capacity) {
    if (!set_ok(s) || vertex >= s->nvert || capacity < s->vert[vertex].nwalls || (!out && s->vert[vertex].nwalls)) return TR_MATH_INVALID;
    memcpy(out, s->vert[vertex].walls, (size_t)s->vert[vertex].nwalls * 8);
    return 0;
}
int32_t tr_region_boundary_get(const tr_region_set *s, uint64_t boundary, tr_region_boundary_info *out) {
    if (!set_ok(s) || !out || boundary >= s->nbnd) return TR_MATH_INVALID;
    const bnd_t *b = &s->bnd[boundary];
    memset(out, 0, sizeof *out);
    out->cell = b->cell; out->classes = b->ncls; out->winner = b->winner; out->vertices = b->nv; out->region = b->region;
    out->kind = b->kind;
    return 0;
}
int32_t tr_region_boundary_plane(const tr_region_set *s, uint64_t boundary, const tr_bigint **normal, const tr_bigint **rhs) {
    if (!set_ok(s) || boundary >= s->nbnd) return TR_MATH_INVALID;
    for (int32_t t = 0; normal && t < s->d; t++) normal[t] = s->bnd[boundary].a[t];
    if (rhs) *rhs = s->bnd[boundary].rhs;
    return 0;
}
int32_t tr_region_boundary_vertices(const tr_region_set *s, uint64_t boundary, uint64_t *out, uint64_t capacity) {
    if (!set_ok(s) || boundary >= s->nbnd || capacity < s->bnd[boundary].nv || !out) return TR_MATH_INVALID;
    memcpy(out, s->bnd[boundary].vid, (size_t)s->bnd[boundary].nv * 8);
    return 0;
}
int32_t tr_region_boundary_classes(const tr_region_set *s, uint64_t boundary, uint64_t *out, uint64_t capacity) {
    if (!set_ok(s) || boundary >= s->nbnd || capacity < s->bnd[boundary].ncls || (!out && s->bnd[boundary].ncls)) return TR_MATH_INVALID;
    memcpy(out, s->bnd[boundary].cls, (size_t)s->bnd[boundary].ncls * 8);
    return 0;
}
int32_t tr_region_authorize(const tr_region_set *s, tr_cert_decision *out) {
    if (!set_ok(s) || !out) return TR_MATH_INVALID;
    memset(out, 0, sizeof *out);
    out->premise = -1; out->stale.column = -1; out->stale.dim = -1;
    if (s->claim == TR_REGION_CLAIM_APPROXIMATE) {
        out->authorized = 0; out->reason = TR_CERT_REFUSE_APPROXIMATE; out->kind = s->claim;
        return TR_MATH_DOMAIN_ERROR;
    }
    out->authorized = 1; out->reason = TR_CERT_OK; out->kind = s->claim;
    return 0;
}

/* ================================================================== lookups */
struct tr_region_lookup_s {
    uint32_t magic;
    tr_region_view view;
    uint64_t bound_bits;
    tr_gfp *fields;
    uint64_t *garner, *half, *wres, *gres, *cell_root, *cell_class, *cell_region, *nlink, *nres, *cand, *cres, *cwinner,
             *cregion, *rclass, *rcons, *kres, *qnum, *qden;
    int32_t *naxis;
    int8_t *ceq, *qsign;
};
void tr_region_lookup_free(tr_region_lookup *l) {
    if (!l || l->magic != JOB_MAGIC) return;
    l->magic = 0;
    free(l->fields); free(l->garner); free(l->half); free(l->wres); free(l->gres); free(l->cell_root); free(l->cell_class);
    free(l->cell_region); free(l->nlink); free(l->nres); free(l->cand); free(l->cres); free(l->cwinner); free(l->cregion);
    free(l->rclass); free(l->rcons); free(l->kres); free(l->qnum); free(l->qden); free(l->naxis); free(l->ceq); free(l->qsign);
    free(l);
}
typedef struct { uint64_t n; const tr_bigq *const *q; const int64_t *num, *den; } qsrc_t;
/* limbs of the largest magnitude, and Q = max over values of sum_k bits(max(|u_k|, v_k)) */
static int32_t query_sizes(const tr_region_set *s, const qsrc_t *q, uint32_t *limbs, uint64_t *qbits) {
    const int32_t d = s->d;
    uint64_t L = 1, Qm = 1;
    for (uint64_t i = 0; i < q->n; i++) {
        uint64_t sum = 0;
        for (int32_t k = 0; k < d; k++) {
            uint64_t bn, bd;
            if (q->q) {
                const tr_bigq *v = q->q[i * d + k];
                if (!q_ok(v)) return TR_MATH_INVALID;
                bn = tr_bigint_bit_length(tr_bigq_numerator(v)); bd = tr_bigint_bit_length(tr_bigq_denominator(v));
            } else {
                int64_t n = q->num[i * d + k], dd = q->den[i * d + k];
                if (dd < 1) return TR_MATH_INVALID;
                bn = (uint64_t)(64 - (n ? __builtin_clzll(tr_abs_u64(n)) : 64)); bd = (uint64_t)(64 - __builtin_clzll((uint64_t)dd));
            }
            uint64_t b = bn > bd ? bn : bd;
            if ((b + 63) / 64 > L) L = (b + 63) / 64;
            sum += b;
        }
        if (sum > Qm) Qm = sum;
    }
    if (L > TR_REGION_LIMBS_MAX) return TR_MATH_UNSUPPORTED;
    *limbs = (uint32_t)L; *qbits = Qm;
    return 0;
}
static uint64_t form_sum_bits(rctx *c, tr_bigint *const *f, int32_t n) {
    tr_bigint *s = zi(c, 0);
    for (int32_t t = 0; t < n && LIVE(c); t++) { tr_bigint *a = zabs(c, f[t]); zset(&s, zadd(c, s, a)); tr_bigint_free(a); }
    uint64_t b = bitlen(s);
    tr_bigint_free(s);
    return b;
}
static int32_t lookup_need(const tr_region_set *s, const qsrc_t *q, int32_t mode, uint64_t max_ties, int32_t target,
                           tr_math_need *need) {
    if (!set_ok(s) || !need || mode < TR_REGION_INDEX || mode > TR_REGION_ALL ||
        (target != TR_REGION_NEED_CPU && target != TR_REGION_NEED_DEVICE)) return TR_MATH_INVALID;
    if (max_ties > ((uint64_t)1 << 32)) return TR_MATH_INVALID;
    uint32_t limbs = 1;
    uint64_t qbits = 1;
    int32_t rc = query_sizes(s, q, &limbs, &qbits);
    if (rc) return rc;
    need_zero(need);
    need->need[TR_BUDGET_OUTPUT] = q->n;
    return 0;
}
static void fill_residues(const tr_bigint *x, const tr_gfp *F, uint32_t k, uint64_t *out, uint64_t n, uint64_t e,
                          uint64_t stride, uint64_t t) {
    for (uint32_t j = 0; j < k; j++) out[((uint64_t)j * n + e) * stride + t] = zres(x, &F[j]);
}
static tr_region_lookup *lookup_create(const tr_region_set *s, const qsrc_t *q, int32_t mode, uint64_t max_ties,
                                       const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    tr_math_need need;
    int32_t rc = lookup_need(s, q, mode, max_ties, TR_REGION_NEED_CPU, &need);
    tr_math_budget dflt;
    budget = budget_or(budget, &dflt);
    if (rc) { set_error(error, rc); return NULL; }
    rctx cx, *c = &cx;
    memset(c, 0, sizeof cx);
    c->b = budget; c->r = refusal;
    const int32_t d = s->d, W = d + 1;
    uint32_t limbs = 1;
    uint64_t qbits = 1;
    query_sizes(s, q, &limbs, &qbits);
    /* the bound: forms (walls, region constraints), splits, cell grid, and the class comparisons */
    uint64_t fb = 0, cb = 0, db = 0;
    for (uint64_t i = 0; i < s->nwalls; i++) { uint64_t b = form_sum_bits(c, s->walls[i].a, W); if (b > fb) fb = b; }
    for (uint64_t r = 0; r < s->nregions; r++)
        for (uint32_t i = 0; i < s->reg[r].nineq + s->reg[r].neq; i++) {
            uint64_t b = form_sum_bits(c, s->reg[r].k[i].a, W);
            if (b > fb) fb = b;
        }
    for (uint64_t ci = 0; ci < s->nclasses; ci++) {
        uint64_t b = form_sum_bits(c, s->cls[ci].A, W);
        if (b > cb) cb = b;
        if (bitlen(s->cls[ci].D) > db) db = bitlen(s->cls[ci].D);
    }
    for (uint64_t n = 0; n < s->nnodes; n++) {
        if (s->node[n].axis < 0) continue;
        uint64_t b = bitlen(s->node[n].split) + 1;
        if (b > fb) fb = b;
    }
    for (uint64_t ce = 0; ce < s->ncells; ce++) { uint64_t b = bitlen(s->cells[ce].W) + 1; if (b > fb) fb = b; }
    tr_bigq **gb = NULL;                               /* the interior cell grid boundaries */
    const uint64_t K = (uint64_t)s->pieces, nb = (uint64_t)d * (K > 1 ? K - 1 : 0);
    if (nb) {
        gb = zalloc(nb * sizeof *gb);
        if (!gb) fail(c, TR_MATH_MEMORY);
        for (int32_t a = 0; LIVE(c) && a < d; a++)
            for (uint64_t j = 1; LIVE(c) && j < K; j++) {
                tr_bigq *w = qsub(c, s->hi[a], s->lo[a]), *kk = qint(c, (int64_t)K), *st = qdiv(c, w, kk), *jj = qint(c, (int64_t)j);
                tr_bigq *o = qmul(c, st, jj);
                gb[(uint64_t)a * (K - 1) + j - 1] = qadd(c, s->lo[a], o);
                tr_bigq_free(w); tr_bigq_free(kk); tr_bigq_free(st); tr_bigq_free(jj); tr_bigq_free(o);
                tr_bigq *g = gb[(uint64_t)a * (K - 1) + j - 1];
                uint64_t b = g ? bitlen(tr_bigq_numerator(g)) + bitlen(tr_bigq_denominator(g)) + 1 : 0;
                if (b > fb) fb = b;
            }
    }
    uint64_t B = qbits + 3 + (fb > cb + db + 1 ? fb : cb + db + 1);
    const uint64_t kk = (B + 2 + 60) / 61;
    if (LIVE(c) && kk > TR_REGION_PRIMES_MAX) fail(c, TR_MATH_OVERFLOW);
    uint64_t ncons = 0;
    for (uint64_t r = 0; r < s->nregions; r++) ncons += (uint64_t)s->reg[r].nineq + s->reg[r].neq;
    /* the job's bytes (host) */
    tr_region_view tv;
    memset(&tv, 0, sizeof tv);
    tv.npoints = q->n; tv.nclasses = s->nclasses; tv.nregions = s->nregions; tv.nnodes = s->nnodes; tv.ncells = s->ncells;
    tv.ncons = ncons; tv.ncand = s->ncand; tv.max_ties = max_ties; tv.d = d; tv.nprimes = (int32_t)kk;
    tv.nwalls = (int32_t)s->nwalls; tv.pieces = s->pieces; tv.limbs = (int32_t)limbs;
    uint64_t bytes = tr_budget_add(sizeof(tr_region_lookup), tr_region_view_device_bytes(&tv));
    if (LIVE(c)) { need.need[TR_BUDGET_BYTES] = bytes; rc = tr_budget_check(budget, &need, refusal); if (rc) fail(c, rc); }
    tr_region_lookup *l = NULL;
    if (LIVE(c)) { l = zalloc(sizeof *l); if (!l) fail(c, TR_MATH_MEMORY); else l->magic = JOB_MAGIC; }
    primes_t *P = LIVE(c) ? zalloc(sizeof *P) : NULL;
    if (LIVE(c) && !P) fail(c, TR_MATH_MEMORY);
    if (LIVE(c) && s->kind == TR_REGION_FINITE) { int32_t e = primes_setup(c, (uint32_t)kk, P); if (e) fail(c, e); }
    if (LIVE(c)) {
        tr_region_view *v = &l->view;
        *v = tv;
        v->mode = mode; v->tie = s->tie; v->kind = s->kind;
        l->bound_bits = B;
        const uint64_t n = q->n;
        l->qsign = zalloc(n * d + 1); l->qnum = zalloc(n * d * limbs * 8 + 8); l->qden = zalloc(n * d * limbs * 8 + 8);
        if (!l->qsign || !l->qnum || !l->qden) fail(c, TR_MATH_MEMORY);
        for (uint64_t i = 0; LIVE(c) && i < n * d; i++) {
            if (q->q) {
                const tr_bigq *x = q->q[i];
                int32_t sg = 0;
                uint64_t cnt = 0;
                if (tr_bigint_export(tr_bigq_numerator(x), &sg, l->qnum + i * limbs, limbs, &cnt)) fail(c, TR_MATH_INVALID);
                l->qsign[i] = (int8_t)sg;
                if (tr_bigint_export(tr_bigq_denominator(x), &sg, l->qden + i * limbs, limbs, &cnt)) fail(c, TR_MATH_INVALID);
            } else {
                int64_t x = q->num[i];
                l->qsign[i] = (int8_t)(x < 0 ? -1 : x > 0);
                l->qnum[i * limbs] = tr_abs_u64(x);
                l->qden[i * limbs] = (uint64_t)q->den[i];
            }
        }
        v->qsign = l->qsign; v->qnum = l->qnum; v->qden = l->qden;
        if (s->kind == TR_REGION_FINITE && LIVE(c)) {
            const uint32_t k = (uint32_t)kk;
            l->fields = zalloc((uint64_t)k * sizeof(tr_gfp)); l->garner = zalloc((uint64_t)k * k * 8); l->half = zalloc((uint64_t)k * 8);
            if (!l->fields || !l->garner || !l->half) fail(c, TR_MATH_MEMORY);
            else {
                memcpy(l->fields, P->F, (size_t)k * sizeof(tr_gfp));
                memcpy(l->garner, P->garner, (size_t)k * k * 8);
                memcpy(l->half, P->half, (size_t)k * 8);
            }
            const uint64_t nw = s->nwalls;
            l->wres = zalloc((uint64_t)k * nw * W * 8 + 8);
            l->gres = zalloc((uint64_t)k * (nb ? nb : 1) * 16);
            l->cell_root = zalloc(s->ncells * 8 + 8); l->cell_class = zalloc((s->ncells + 1) * 8); l->cell_region = zalloc((s->ncells + 1) * 8);
            l->naxis = zalloc(s->nnodes * 4 + 4); l->nlink = zalloc(s->nnodes * 16 + 16); l->nres = zalloc((uint64_t)k * s->nnodes * 16 + 16);
            l->cand = zalloc(s->ncand * 8 + 8);
            l->cres = zalloc((uint64_t)k * s->nclasses * (W + 1) * 8 + 8);
            l->cwinner = zalloc(s->nclasses * 8 + 8); l->cregion = zalloc(s->nclasses * 8 + 8);
            l->rclass = zalloc(s->nregions * 8 + 8); l->rcons = zalloc((s->nregions + 1) * 8);
            l->ceq = zalloc(ncons + 1); l->kres = zalloc((uint64_t)k * ncons * W * 8 + 8);
            if (!l->wres || !l->gres || !l->cell_root || !l->cell_class || !l->cell_region || !l->naxis || !l->nlink ||
                !l->nres || !l->cand || !l->cres || !l->cwinner || !l->cregion || !l->rclass || !l->rcons || !l->ceq || !l->kres)
                fail(c, TR_MATH_MEMORY);
            for (uint64_t i = 0; LIVE(c) && i < nw; i++)
                for (int32_t t = 0; t < W; t++) fill_residues(s->walls[i].a[t], P->F, k, l->wres, nw, i, (uint64_t)W, (uint64_t)t);
            for (uint64_t i = 0; LIVE(c) && i < nb; i++) {
                fill_residues(tr_bigq_denominator(gb[i]), P->F, k, l->gres, nb, i, 2, 0);
                fill_residues(tr_bigq_numerator(gb[i]), P->F, k, l->gres, nb, i, 2, 1);
            }
            for (uint64_t ce = 0; LIVE(c) && ce < s->ncells; ce++) {
                l->cell_root[ce] = s->cells[ce].root;
                l->cell_class[ce] = s->cells[ce].class_begin; l->cell_class[ce + 1] = s->cells[ce].class_end;
                l->cell_region[ce] = s->cells[ce].region_begin; l->cell_region[ce + 1] = s->cells[ce].region_end;
            }
            for (uint64_t ce = 0; LIVE(c) && ce < s->ncells; ce++) {   /* the nodes of each cell: q = the cell's W */
                const uint64_t first = s->cells[ce].root, last = ce + 1 < s->ncells ? s->cells[ce + 1].root : s->nnodes;
                for (uint64_t nd = first; nd < last && LIVE(c); nd++) {
                    const node_t *N = &s->node[nd];
                    l->naxis[nd] = N->axis;
                    l->nlink[2 * nd] = N->link[0]; l->nlink[2 * nd + 1] = N->link[1];
                    if (N->axis >= 0) {
                        fill_residues(s->cells[ce].W, P->F, k, l->nres, s->nnodes, nd, 2, 0);
                        fill_residues(N->split, P->F, k, l->nres, s->nnodes, nd, 2, 1);
                    }
                }
            }
            if (LIVE(c) && s->ncand) memcpy(l->cand, s->cand, (size_t)s->ncand * 8);
            for (uint64_t ci = 0; LIVE(c) && ci < s->nclasses; ci++) {
                for (int32_t t = 0; t < W; t++) fill_residues(s->cls[ci].A[t], P->F, k, l->cres, s->nclasses, ci, (uint64_t)W + 1, (uint64_t)t);
                fill_residues(s->cls[ci].D, P->F, k, l->cres, s->nclasses, ci, (uint64_t)W + 1, (uint64_t)W);
                l->cwinner[ci] = s->cls[ci].winner; l->cregion[ci] = s->cls[ci].region;
            }
            uint64_t e = 0;
            for (uint64_t r = 0; LIVE(c) && r < s->nregions; r++) {
                l->rclass[r] = s->reg[r].cls; l->rcons[r] = e;
                for (uint32_t i = 0; i < s->reg[r].nineq + s->reg[r].neq; i++, e++) {
                    l->ceq[e] = (int8_t)s->reg[r].k[i].eq;
                    for (int32_t t = 0; t < W; t++) fill_residues(s->reg[r].k[i].a[t], P->F, k, l->kres, ncons, e, (uint64_t)W, (uint64_t)t);
                }
            }
            l->rcons[s->nregions] = e;
            v->fields = l->fields; v->garner = l->garner; v->half = l->half; v->wres = l->wres; v->gres = nb ? l->gres : NULL;
            v->cell_root = l->cell_root; v->cell_class = l->cell_class; v->cell_region = l->cell_region;
            v->naxis = l->naxis; v->nlink = l->nlink; v->nres = l->nres; v->cand = l->cand; v->cres = l->cres;
            v->cwinner = l->cwinner; v->cregion = l->cregion; v->rclass = l->rclass; v->rcons = l->rcons;
            v->ceq = l->ceq; v->kres = l->kres;
        } else if (LIVE(c)) {
            v->nprimes = 0;
        }
    }
    for (uint64_t i = 0; gb && i < nb; i++) tr_bigq_free(gb[i]);
    free(gb); free(P);
    if (c->rc) { tr_region_lookup_free(l); set_error(error, c->rc); return NULL; }
    set_error(error, TR_MATH_OK);
    return l;
}
int32_t tr_region_lookup_need(const tr_region_set *s, uint64_t npoints, const tr_bigq *const *values, int32_t mode,
                              uint64_t max_ties, int32_t target, tr_math_need *need) {
    if (npoints && !values) return TR_MATH_INVALID;
    qsrc_t q = {npoints, values, NULL, NULL};
    tr_region_lookup *l = NULL;
    int32_t rc = lookup_need(s, &q, mode, max_ties, target, need);
    if (rc) return rc;
    tr_math_budget big;
    for (int i = 0; i < TR_BUDGET_RESOURCES; i++) big.limit[i] = UINT64_MAX;
    int32_t e = 0;
    l = lookup_create(s, &q, mode, max_ties, &big, NULL, &e);   /* the exact job size needs the job's k */
    if (!l) return e;
    need->need[TR_BUDGET_BYTES] = tr_budget_add(sizeof(tr_region_lookup), tr_region_view_device_bytes(&l->view));
    if (target == TR_REGION_NEED_DEVICE)
        need->need[TR_BUDGET_BYTES] = tr_budget_add(need->need[TR_BUDGET_BYTES], tr_region_view_device_bytes(&l->view));
    tr_region_lookup_free(l);
    return 0;
}
tr_region_lookup *tr_region_lookup_create(const tr_region_set *s, uint64_t npoints, const tr_bigq *const *values,
                                          int32_t mode, uint64_t max_ties, const tr_math_budget *budget,
                                          tr_math_refusal *refusal, int32_t *error) {
    if (npoints && !values) { set_error(error, TR_MATH_INVALID); return NULL; }
    qsrc_t q = {npoints, values, NULL, NULL};
    return lookup_create(s, &q, mode, max_ties, budget, refusal, error);
}
int32_t tr_region_lookup_need_q64(const tr_region_set *s, uint64_t npoints, const int64_t *num, const int64_t *den,
                                  int32_t mode, uint64_t max_ties, int32_t target, tr_math_need *need) {
    if (npoints && (!num || !den)) return TR_MATH_INVALID;
    qsrc_t q = {npoints, NULL, num, den};
    int32_t rc = lookup_need(s, &q, mode, max_ties, target, need);
    if (rc) return rc;
    tr_math_budget big;
    for (int i = 0; i < TR_BUDGET_RESOURCES; i++) big.limit[i] = UINT64_MAX;
    int32_t e = 0;
    tr_region_lookup *l = lookup_create(s, &q, mode, max_ties, &big, NULL, &e);
    if (!l) return e;
    need->need[TR_BUDGET_BYTES] = tr_budget_add(sizeof(tr_region_lookup), tr_region_view_device_bytes(&l->view));
    if (target == TR_REGION_NEED_DEVICE)
        need->need[TR_BUDGET_BYTES] = tr_budget_add(need->need[TR_BUDGET_BYTES], tr_region_view_device_bytes(&l->view));
    tr_region_lookup_free(l);
    return 0;
}
tr_region_lookup *tr_region_lookup_create_q64(const tr_region_set *s, uint64_t npoints, const int64_t *num,
                                              const int64_t *den, int32_t mode, uint64_t max_ties,
                                              const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error) {
    if (npoints && (!num || !den)) { set_error(error, TR_MATH_INVALID); return NULL; }
    qsrc_t q = {npoints, NULL, num, den};
    return lookup_create(s, &q, mode, max_ties, budget, refusal, error);
}
int32_t tr_region_lookup_get_view(const tr_region_lookup *l, tr_region_view *out) {
    if (!l || l->magic != JOB_MAGIC || !out) return TR_MATH_INVALID;
    *out = l->view;
    return 0;
}
uint64_t tr_region_lookup_bound_bits(const tr_region_lookup *l) { return l && l->magic == JOB_MAGIC ? l->bound_bits : 0; }
int32_t tr_region_lookup_eval(const tr_region_view *v, tr_region_answer *out, uint64_t *ties, const tr_math_budget *budget,
                              tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!tr_region_view_valid(v) || (v->npoints && !out) || (v->max_ties && v->npoints && !ties)) return TR_MATH_INVALID;
    tr_math_budget dflt;
    budget = budget_or(budget, &dflt);
    int32_t rc = tr_budget_check_one(budget, TR_BUDGET_OUTPUT, v->npoints, refusal);
    if (rc) return rc;
    for (uint64_t i = 0; i < v->npoints; i++) tr_region_answer_one(v, i, &out[i], v->max_ties ? ties + i * v->max_ties : NULL);
    return 0;
}
