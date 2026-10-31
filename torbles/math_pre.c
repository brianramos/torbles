/* PRE-001: affine and monotone-piece preimages; see math_pre.h.
 * SPDX-License-Identifier: Apache-2.0 */
#include "math_pre.h"
#include "mathematics.h"
#include "math_domain.h"
#include "math_handles.h"
#include "math_carriers.h"
#include "math_certificates.h"
#include "torble.h"
#include <stdlib.h>
#include <string.h>

TR_MATH_ABI(tr_pre_codec, TR_PRE_ABI);      /* the ctypes boundary of PRE-001 (math_handles.h) */
TR_MATH_ABI(tr_pre_piece, TR_PRE_ABI);
TR_MATH_ABI(tr_pre_map, TR_PRE_ABI);
TR_MATH_ABI(tr_pre_predicate, TR_PRE_ABI);
TR_MATH_ABI(tr_pre_span, TR_PRE_ABI);
TR_MATH_ABI(tr_pre_set_info, TR_PRE_ABI);
TR_MATH_ABI(tr_pre_scan_stats, TR_PRE_ABI);
TR_MATH_ABI(tr_pre_view, TR_PRE_ABI);

_Static_assert(sizeof(tr_pre_codec) == 32, "tr_pre_codec ABI");
_Static_assert(sizeof(tr_pre_piece) == 80, "tr_pre_piece ABI");
_Static_assert(sizeof(tr_pre_map) == 64, "tr_pre_map ABI");
_Static_assert(sizeof(tr_pre_predicate) == 40, "tr_pre_predicate ABI");
_Static_assert(sizeof(tr_pre_span) == 40, "tr_pre_span ABI");
_Static_assert(sizeof(tr_pre_set_info) == 80, "tr_pre_set_info ABI");
_Static_assert(sizeof(tr_pre_scan_stats) == 80, "tr_pre_scan_stats ABI");
_Static_assert(sizeof(tr_pre_view) == 136, "tr_pre_view ABI");

#define SET_MAGIC UINT32_C(0x31455250)
#define BRANCH_NODES 6                       /* nodes of one branch program (y = a x^2 + b) */

struct tr_pre_set_s {
    uint32_t magic;
    int32_t flags, kind, op, branches, decided, certified, residue;
    uint64_t nspans, codes, bytes, nvalues, nnodes;
    int64_t modulus;
    int32_t nbranch, reserved;
    tr_pre_codec codec;
    tr_num_rational c;
    tr_pre_span *spans;
    tr_num_rational *values;
    int64_t *branch_lo, *branch_hi;
    int32_t *node_off;
    tr_unc_node *nodes;
    tr_cert **certs;                         /* nbranch entries (NULL: none) */
};

static void set_error(int32_t *error, int32_t code) { if (error) *error = code; }
static int set_valid(const tr_pre_set *s) { return s && s->magic == SET_MAGIC; }
static int canonical(tr_num_rational q) { return tr_unc_canonical(q); }
static int flip(int op) {
    switch (op) { case TR_PRE_LT: return TR_PRE_GT; case TR_PRE_LE: return TR_PRE_GE; case TR_PRE_GT: return TR_PRE_LT;
                  case TR_PRE_GE: return TR_PRE_LE; default: return op; }
}

int32_t tr_pre_codec_for_column(const struct atable_s *t, int32_t col, int64_t offset, int32_t scale, int32_t nullable,
                                tr_pre_codec *out) {
    if (!t || !out || col < 0 || col >= MAXCOL || scale < 0 || scale > TR_SCOPE_MAX_SCALE) return TR_MATH_INVALID;
    int32_t w = t->width[col];
    if (w < 1 || w > 32) return TR_MATH_UNSUPPORTED;
    out->offset = offset; out->scale = scale; out->nullable = nullable ? 1 : 0;
    out->lo = 0; out->hi = (int64_t)((UINT64_C(1) << w) - 1);
    return TR_MATH_OK;
}

/* ------------------------------------------------------------------ exact integers (FND-009) */
typedef struct { const tr_math_budget *b; tr_math_refusal *r; int32_t rc; } ctx;
static tr_bigint *bi(ctx *x, int64_t v) { int32_t e = 0; tr_bigint *r = tr_bigint_from_i64(v, x->b, x->r, &e); if (!r && !x->rc) x->rc = e ? e : TR_MATH_MEMORY; return r; }
static tr_bigq *bq(ctx *x, tr_num_rational q) { int32_t e = 0; tr_bigq *r = tr_bigq_from_q64(q, x->b, x->r, &e); if (!r && !x->rc) x->rc = e ? e : TR_MATH_MEMORY; return r; }
static void keep_rc(ctx *x, int32_t e) { if (!x->rc) x->rc = e ? e : TR_MATH_MEMORY; }
static tr_bigq *q_sub(ctx *x, const tr_bigq *a, const tr_bigq *b) { if (x->rc) return NULL; int32_t e = 0; tr_bigq *r = tr_bigq_sub(a, b, x->b, x->r, &e); if (!r) keep_rc(x, e); return r; }
static tr_bigq *q_mul(ctx *x, const tr_bigq *a, const tr_bigq *b) { if (x->rc) return NULL; int32_t e = 0; tr_bigq *r = tr_bigq_mul(a, b, x->b, x->r, &e); if (!r) keep_rc(x, e); return r; }
static tr_bigq *q_div(ctx *x, const tr_bigq *a, const tr_bigq *b) { if (x->rc) return NULL; int32_t e = 0; tr_bigq *r = tr_bigq_div(a, b, x->b, x->r, &e); if (!r) keep_rc(x, e); return r; }
/* floor (up = 0) or ceil (up = 1) of a rational */
static tr_bigint *q_round(ctx *x, const tr_bigq *t, int up) {
    if (x->rc) return NULL;
    tr_bigint *q = NULL, *r = NULL;
    int32_t e = tr_bigint_divmod(tr_bigq_numerator(t), tr_bigq_denominator(t), TR_DIV_FLOOR, &q, &r, x->b, x->r);
    if (e) { keep_rc(x, e); return NULL; }
    if (up && tr_bigint_sign(r) != 0) {
        tr_bigint *one = bi(x, 1), *s = NULL;
        if (one) { s = tr_bigint_add(q, one, x->b, x->r, &e); if (!s) keep_rc(x, e); }
        tr_bigint_free(one); tr_bigint_free(q); q = s;
    }
    tr_bigint_free(r);
    return q;
}
static tr_bigint *z_add(ctx *x, const tr_bigint *a, int64_t k) {
    if (x->rc || !a) return NULL;
    tr_bigint *kk = bi(x, k), *r = NULL;
    int32_t e = 0;
    if (kk) { r = tr_bigint_add(a, kk, x->b, x->r, &e); if (!r) keep_rc(x, e); }
    tr_bigint_free(kk);
    return r;
}
static tr_bigint *z_neg(ctx *x, const tr_bigint *a) {
    if (x->rc || !a) return NULL;
    int32_t e = 0; tr_bigint *r = tr_bigint_neg(a, x->b, x->r, &e);
    if (!r) keep_rc(x, e);
    return r;
}
static tr_bigint *z_isqrt(ctx *x, const tr_bigint *a) {
    if (x->rc || !a) return NULL;
    int32_t e = 0; tr_bigint *r = tr_bigint_isqrt(a, x->b, x->r, &e);
    if (!r) keep_rc(x, e);
    return r;
}
/* the integer value of an integral rational, or NULL (not integral: no error) */
static tr_bigint *q_integral(ctx *x, const tr_bigq *t) {
    if (x->rc) return NULL;
    int32_t e = 0;
    tr_bigint *r = tr_bigq_to_bigint(t, x->b, x->r, &e);
    if (!r && e != TR_MATH_NO_SOLUTION) keep_rc(x, e);
    return r;
}

/* ------------------------------------------------------------------ integer intervals, clipped */
typedef struct { __int128 lo, hi; } ziv;
typedef struct { ziv *v; uint64_t n, cap; } zlist;
static int zl_push(zlist *l, __int128 lo, __int128 hi) {
    if (l->n == l->cap) {
        uint64_t cap = l->cap ? 2 * l->cap : 16;
        ziv *v = realloc(l->v, (size_t)cap * sizeof *v);
        if (!v) return 0;
        l->v = v; l->cap = cap;
    }
    l->v[l->n].lo = lo; l->v[l->n].hi = hi; l->n++;
    return 1;
}
/* a bigint bound against [lo, hi]: -1 below lo, 1 above hi, 0 inside (*v its value) */
static int z_where(const tr_bigint *b, __int128 lo, __int128 hi, __int128 *v) {
    tr_math_i128 w;
    if (tr_bigint_to_i128(b, &w)) return tr_bigint_sign(b) < 0 ? -1 : 1;
    __int128 x = tr_i128_get(w);
    if (x < lo) return -1;
    if (x > hi) return 1;
    *v = x;
    return 0;
}
/* [u, v] (NULL: unbounded) intersected with [lo, hi], appended when not empty */
static void emit(ctx *x, zlist *out, const tr_bigint *u, const tr_bigint *v, __int128 lo, __int128 hi) {
    if (x->rc) return;
    __int128 a = lo, b = hi, t;
    if (u) { int w = z_where(u, lo, hi, &t); if (w > 0) return; if (w == 0) a = t; }
    if (v) { int w = z_where(v, lo, hi, &t); if (w < 0) return; if (w == 0) b = t; }
    if (a > b) return;
    if (!zl_push(out, a, b)) x->rc = TR_MATH_MEMORY;
}
/* the integers X with X op t, within [lo, hi] */
static void solve_linear(ctx *x, int op, const tr_bigq *t, __int128 lo, __int128 hi, zlist *out) {
    tr_bigint *f = NULL, *g = NULL, *h = NULL;
    switch (op) {
    case TR_PRE_LT: f = q_round(x, t, 1); g = z_add(x, f, -1); emit(x, out, NULL, g, lo, hi); break;
    case TR_PRE_LE: f = q_round(x, t, 0); emit(x, out, NULL, f, lo, hi); break;
    case TR_PRE_GT: f = q_round(x, t, 0); g = z_add(x, f, 1); emit(x, out, g, NULL, lo, hi); break;
    case TR_PRE_GE: f = q_round(x, t, 1); emit(x, out, f, NULL, lo, hi); break;
    case TR_PRE_EQ: f = q_integral(x, t); if (f) emit(x, out, f, f, lo, hi); break;
    default:                                              /* NE */
        f = q_integral(x, t);
        if (!f) { if (!x->rc) emit(x, out, NULL, NULL, lo, hi); break; }
        g = z_add(x, f, -1); h = z_add(x, f, 1);
        emit(x, out, NULL, g, lo, hi); emit(x, out, h, NULL, lo, hi);
        break;
    }
    tr_bigint_free(f); tr_bigint_free(g); tr_bigint_free(h);
}
/* the integers X with X^2 op C, within [lo, hi] */
static void solve_square(ctx *x, int op, const tr_bigq *C, __int128 lo, __int128 hi, zlist *out) {
    int sc = tr_bigq_sign(C);
    tr_bigint *f = NULL, *r = NULL, *nr = NULL, *a1 = NULL, *b1 = NULL, *m1 = NULL, *p1 = NULL;
    switch (op) {
    case TR_PRE_LE: case TR_PRE_GT:                       /* X^2 <= C: |X| <= isqrt(floor(C)) */
        if (sc < 0) { if (op == TR_PRE_GT) emit(x, out, NULL, NULL, lo, hi); break; }
        f = q_round(x, C, 0); r = z_isqrt(x, f); nr = z_neg(x, r);
        if (op == TR_PRE_LE) emit(x, out, nr, r, lo, hi);
        else { a1 = z_add(x, nr, -1); b1 = z_add(x, r, 1); emit(x, out, NULL, a1, lo, hi); emit(x, out, b1, NULL, lo, hi); }
        break;
    case TR_PRE_LT: case TR_PRE_GE:                       /* X^2 < C: |X| <= isqrt(ceil(C) - 1) */
        if (sc <= 0) { if (op == TR_PRE_GE) emit(x, out, NULL, NULL, lo, hi); break; }
        f = q_round(x, C, 1); m1 = z_add(x, f, -1); r = z_isqrt(x, m1); nr = z_neg(x, r);
        if (op == TR_PRE_LT) emit(x, out, nr, r, lo, hi);
        else { a1 = z_add(x, nr, -1); b1 = z_add(x, r, 1); emit(x, out, NULL, a1, lo, hi); emit(x, out, b1, NULL, lo, hi); }
        break;
    default: {                                            /* EQ / NE: C a square r^2 */
        int square = 0;
        if (sc >= 0) {
            f = q_integral(x, C);
            if (f) {
                r = z_isqrt(x, f);
                int32_t e = 0;
                tr_bigint *rr = r ? tr_bigint_mul(r, r, x->b, x->r, &e) : NULL;
                if (r && !rr) keep_rc(x, e);
                square = rr && tr_bigint_cmp(rr, f) == 0;
                tr_bigint_free(rr);
            }
        }
        if (x->rc) break;
        if (op == TR_PRE_EQ) {
            if (!square) break;
            nr = z_neg(x, r);
            if (tr_bigint_sign(r) == 0) emit(x, out, r, r, lo, hi);
            else { emit(x, out, nr, nr, lo, hi); emit(x, out, r, r, lo, hi); }
            break;
        }
        if (!square) { emit(x, out, NULL, NULL, lo, hi); break; }
        nr = z_neg(x, r);
        a1 = z_add(x, nr, -1); b1 = z_add(x, r, 1);
        emit(x, out, NULL, a1, lo, hi);
        if (tr_bigint_sign(r) != 0) { p1 = z_add(x, nr, 1); m1 = z_add(x, r, -1); emit(x, out, p1, m1, lo, hi); }
        emit(x, out, b1, NULL, lo, hi);
        break;
    }
    }
    tr_bigint_free(f); tr_bigint_free(r); tr_bigint_free(nr); tr_bigint_free(a1); tr_bigint_free(b1);
    tr_bigint_free(m1); tr_bigint_free(p1);
}

/* ------------------------------------------------------------------ branches */
typedef struct {
    int32_t kind;                    /* AFFINE / SQUARE / RESIDUE */
    tr_num_rational a, b;
    int64_t m;
    __int128 llo, lhi;               /* its logical range */
} branch;
static int cmp_ziv(const void *p, const void *q) {
    const ziv *a = p, *b = q;
    return a->lo < b->lo ? -1 : (a->lo > b->lo ? 1 : 0);
}
static int cmp_q(const void *p, const void *q) { return tr_q_cmp(*(const tr_num_rational *)p, *(const tr_num_rational *)q); }
/* the y = f(x) program of a branch (UNC-001 nodes), and with `cmp` the predicate f(x) op c; returns the node count */
static int32_t branch_program(const branch *br, int32_t cmp, tr_num_rational c, tr_unc_node *nd) {
    int32_t n = 0;
    memset(nd, 0, sizeof(tr_unc_node) * (BRANCH_NODES + 2));
    for (int32_t i = 0; i < BRANCH_NODES + 2; i++) { nd[i].k = tr_q_int(0); nd[i].a = nd[i].b = -1; }
    nd[n].op = TR_UNC_COL; nd[n].a = 0; n++;
    if (br->kind == TR_PRE_RESIDUE) { nd[n].op = TR_UNC_MOD; nd[n].a = 0; nd[n].k = tr_q_int(br->m); n++; }
    else {
        int32_t xin = 0;
        if (br->kind == TR_PRE_SQUARE) { nd[n].op = TR_UNC_SQR; nd[n].a = 0; xin = n; n++; }
        nd[n].op = TR_UNC_CONST; nd[n].k = br->a; n++;
        nd[n].op = TR_UNC_MUL; nd[n].a = xin; nd[n].b = n - 1; n++;
        nd[n].op = TR_UNC_CONST; nd[n].k = br->b; n++;
        nd[n].op = TR_UNC_ADD; nd[n].a = n - 2; nd[n].b = n - 1; n++;
    }
    if (cmp) {
        static const int32_t ops[7] = {0, TR_UNC_LT, TR_UNC_LE, TR_UNC_EQ, TR_UNC_NE, TR_UNC_GE, TR_UNC_GT};
        nd[n].op = TR_UNC_CONST; nd[n].k = c; n++;
        nd[n].op = ops[cmp]; nd[n].a = n - 2; nd[n].b = n - 1; n++;
    }
    return n;
}
/* UNC-001's decision of f(x) op c on the whole branch: TR_UNC_ALL / _NONE / _UNKNOWN */
static int32_t classify(const branch *br, int32_t op, tr_num_rational c, int32_t scale) {
    if (op == TR_PRE_IN || br->llo < INT64_MIN || br->lhi > INT64_MAX) return TR_UNC_UNKNOWN;
    tr_unc_node nd[BRANCH_NODES + 2];
    tr_unc_program p;
    memset(&p, 0, sizeof p);
    p.nnodes = branch_program(br, op, c, nd); p.ncols = 1; p.carrier = TR_UNC_EXACT; p.precision = 0;
    p.rank = TR_UNC_MAX_TERMS; p.domains = TR_UNC_DOM_ALL; p.root = p.nnodes - 1; p.budget = tr_unc_qv(0, 0);
    p.nodes = nd;
    tr_unc_plan plan;
    if (tr_unc_check(&p, 0, &plan, NULL)) return TR_UNC_UNKNOWN;
    tr_unc_input in;
    memset(&in, 0, sizeof in);
    in.rows = 1; in.present = 1; in.min = (int64_t)br->llo; in.max = (int64_t)br->lhi; in.scale = scale;
    in.modulus = br->llo == br->lhi ? 0 : 1;
    tr_unc_value *slots = calloc((size_t)plan.nlive + 1, sizeof *slots);
    if (!slots) return TR_UNC_UNKNOWN;
    tr_unc_result res;
    tr_unc_eval_group(&p, &plan, &in, slots, &res);
    free(slots);
    return res.decision;
}
/* the exact preimage of one AFFINE / SQUARE branch on its range */
static void solve_branch(ctx *x, const branch *br, int32_t op, tr_num_rational c, const tr_num_rational *values,
                         uint64_t nvalues, int32_t scale, zlist *out) {
    tr_bigq *T = bq(x, tr_q_int(tr_unc_pow10(scale))), *A = bq(x, br->a), *B = bq(x, br->b);
    tr_bigq *TT = br->kind == TR_PRE_SQUARE ? q_mul(x, T, T) : NULL;
    int neg = tr_unc_qsign(br->a) < 0;
    uint64_t n = op == TR_PRE_IN ? nvalues : 1;
    for (uint64_t i = 0; i < n && !x->rc; i++) {
        tr_bigq *Cq = bq(x, op == TR_PRE_IN ? values[i] : c);
        tr_bigq *d = q_sub(x, Cq, B);
        tr_bigq *dt = q_mul(x, d, br->kind == TR_PRE_SQUARE ? TT : T);
        tr_bigq *t = q_div(x, dt, A);
        int o = op == TR_PRE_IN ? TR_PRE_EQ : op;
        if (neg) o = flip(o);
        if (!x->rc) {
            if (br->kind == TR_PRE_SQUARE) solve_square(x, o, t, br->llo, br->lhi, out);
            else solve_linear(x, o, t, br->llo, br->lhi, out);
        }
        tr_bigq_free(Cq); tr_bigq_free(d); tr_bigq_free(dt); tr_bigq_free(t);
    }
    tr_bigq_free(T); tr_bigq_free(A); tr_bigq_free(B); tr_bigq_free(TT);
}

/* ------------------------------------------------------------------ law certificates */
static int fits(__int128 v) { return v >= INT64_MIN && v <= INT64_MAX; }
/* the law of a branch's algebraic transformation (NULL when a coefficient leaves int64; see the header comment) */
static tr_cert *branch_law(const branch *br, const tr_pre_codec *cd, const tr_math_budget *b, tr_math_refusal *r) {
    __int128 T = tr_unc_pow10(cd->scale), off = cd->offset, k[8];
    uint8_t e[8];
    int32_t n = 0;
    if (br->kind == TR_PRE_RESIDUE) {                /* (code + offset) - (code + R0) - M k0, offset = M k0 + R0 */
        __int128 M = (__int128)br->m * T, R0 = off % M;
        if (R0 < 0) R0 += M;
        __int128 Mk0 = off - R0;
        k[0] = 1; e[0] = 1; k[1] = off; e[1] = 0; k[2] = -1; e[2] = 1; k[3] = -R0; e[3] = 0; k[4] = -Mk0; e[4] = 0;
        n = 5;
    } else {
        __int128 p = br->a.numerator, q = br->a.denominator, rr = br->b.numerator, t = br->b.denominator;
        __int128 pt = p * t, qr = q * rr;            /* each < 2^126 */
        if (!fits(pt) || !fits(qr)) return NULL;
        __int128 pto = pt * off;
        if (!fits(pto)) return NULL;
        if (br->kind == TR_PRE_AFFINE) {             /* D = q t T: p t code + p t offset + q r T - (P code + Q) */
            __int128 qrT = qr * T, Q = pto + qrT;    /* |qr T| < 2^123 */
            if (!fits(qrT) || !fits(Q)) return NULL;
            k[0] = pt; e[0] = 1; k[1] = pto; e[1] = 0; k[2] = qrT; e[2] = 0; k[3] = -pt; e[3] = 1; k[4] = -Q; e[4] = 0;
            n = 5;
        } else {                                       /* D = q t T^2: p t (code + offset)(code + offset) + q r T^2 */
            __int128 TT = T * T;                        /* < 2^120 */
            if (!fits(TT)) return NULL;
            __int128 qrT = qr * TT, pto2 = pto * off, two = 2 * pto;
            if (!fits(qrT) || !fits(pto2) || !fits(two) || !fits(pto2 + qrT)) return NULL;
            k[0] = pt; e[0] = 2; k[1] = pto; e[1] = 1; k[2] = pto; e[2] = 1; k[3] = pto2; e[3] = 0; k[4] = qrT; e[4] = 0;
            k[5] = -pt; e[5] = 2; k[6] = -two; e[6] = 1; k[7] = -(pto2 + qrT); e[7] = 0;   /* - (P2 code^2 + P1 code + P0) */
            n = 8;
        }
    }
    int64_t co[8];
    uint8_t ex[8];
    for (int32_t i = 0; i < n; i++) {
        if (!fits(k[i])) return NULL;
        co[i] = (int64_t)k[i]; ex[i] = e[i];
    }
    tr_cert_claim c;
    memset(&c, 0, sizeof c);
    c.kind = TR_CLAIM_POLYNOMIAL; c.policy = TR_CERT_NULL_FORBID; c.ncols = 1; c.col[0] = 0;
    for (int32_t i = 1; i < TR_CERT_MAX_COLS; i++) c.col[i] = -1;
    c.carrier = TR_CARRIER_I128; c.nterms = (uint64_t)n; c.exponents = ex; c.coefficients = co;
    int32_t err = 0;
    return tr_cert_law(TR_CERT_LAW_RING_IDENTITY, &c, b, r, &err);
}

/* ------------------------------------------------------------------ the preimage */
static int32_t check_map(const tr_pre_codec *cd, const tr_pre_map *m, const tr_pre_predicate *pr) {
    if (!cd || !m || !pr || cd->scale < 0 || cd->scale > TR_SCOPE_MAX_SCALE || cd->lo > cd->hi || pr->reserved || m->reserved)
        return TR_MATH_INVALID;
    if (pr->op < TR_PRE_LT || pr->op > TR_PRE_IN) return TR_MATH_INVALID;
    if (pr->op == TR_PRE_IN) {
        if (pr->nvalues > TR_PRE_MAX_VALUES) return TR_MATH_UNSUPPORTED;
        if (pr->nvalues && !pr->values) return TR_MATH_INVALID;
        for (uint64_t i = 0; i < pr->nvalues; i++) if (!canonical(pr->values[i])) return TR_MATH_INVALID;
    } else if (!canonical(pr->c)) return TR_MATH_INVALID;
    switch (m->kind) {
    case TR_PRE_AFFINE: case TR_PRE_SQUARE:
        if (!canonical(m->a) || !canonical(m->b)) return TR_MATH_INVALID;
        return m->a.numerator == 0 ? TR_MATH_DOMAIN_ERROR : TR_MATH_OK;   /* a zero divisor */
    case TR_PRE_RESIDUE:
        if (m->modulus < 0) return TR_MATH_INVALID;
        if (m->modulus == 0) return TR_MATH_DOMAIN_ERROR;
        return (__int128)m->modulus * tr_unc_pow10(cd->scale) > INT64_MAX ? TR_MATH_UNSUPPORTED : TR_MATH_OK;
    case TR_PRE_PIECES:
        if (m->npieces < 1 || !m->pieces) return TR_MATH_INVALID;
        if (m->npieces > TR_PRE_MAX_PIECES) return TR_MATH_UNSUPPORTED;
        for (int32_t i = 0; i < m->npieces; i++) {
            const tr_pre_piece *p = &m->pieces[i];
            if ((p->kind != TR_PRE_AFFINE && p->kind != TR_PRE_SQUARE) || p->reserved || !canonical(p->a) || !canonical(p->b) ||
                (!p->lo_inf && !canonical(p->lo)) || (!p->hi_inf && !canonical(p->hi)) ||
                (p->lo_inf != 0 && p->lo_inf != 1) || (p->hi_inf != 0 && p->hi_inf != 1) ||
                (!p->lo_inf && !p->hi_inf && tr_q_cmp(p->lo, p->hi) > 0)) return TR_MATH_INVALID;
        }
        for (int32_t i = 0; i < m->npieces; i++) if (m->pieces[i].a.numerator == 0) return TR_MATH_DOMAIN_ERROR;
        return TR_MATH_OK;
    default:
        return TR_MATH_INVALID;
    }
}
/* the logical range of a piece: [ceil(lo T), floor(hi T)] within [llo, lhi]; 0 when it holds no code */
static int piece_range(ctx *x, const tr_pre_piece *p, int32_t scale, __int128 llo, __int128 lhi, __int128 *lo, __int128 *hi) {
    *lo = llo; *hi = lhi;
    tr_bigq *T = bq(x, tr_q_int(tr_unc_pow10(scale)));
    for (int side = 0; side < 2 && !x->rc; side++) {
        if (side == 0 ? p->lo_inf : p->hi_inf) continue;
        tr_bigq *v = bq(x, side == 0 ? p->lo : p->hi), *vt = q_mul(x, v, T);
        tr_bigint *r = vt ? q_round(x, vt, side == 0) : NULL;
        __int128 w;
        if (r) {
            int where = z_where(r, llo, lhi, &w);
            if (side == 0) { if (where > 0) *lo = lhi + 1; else if (where == 0) *lo = w; }
            else { if (where < 0) *hi = llo - 1; else if (where == 0) *hi = w; }
        }
        tr_bigint_free(r); tr_bigq_free(v); tr_bigq_free(vt);
    }
    tr_bigq_free(T);
    return !x->rc && *lo <= *hi;
}
static uint64_t count_residue(__int128 lo, __int128 hi, __int128 M, __int128 rlo, __int128 rhi) {
    __int128 N = hi - lo + 1, q = N / M, rem = N % M, s = lo % M;
    if (s < 0) s += M;
    __int128 c = q * (rhi - rlo + 1);
    /* the remaining rem codes have residues s .. s + rem - 1 (mod M) */
    for (int k = 0; k < 2; k++) {
        __int128 a = s, b = s + rem - 1, l = rlo + k * M, h = rhi + k * M;
        __int128 x0 = a > l ? a : l, x1 = b < h ? b : h;
        if (rem > 0 && x0 <= x1) c += x1 - x0 + 1;
    }
    return c > (__int128)UINT64_MAX ? UINT64_MAX : (uint64_t)c;
}
tr_pre_set *tr_pre_preimage(const tr_pre_codec *cd, const tr_pre_map *m, const tr_pre_predicate *pr,
                            const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    int32_t rc = check_map(cd, m, pr);
    if (rc) { set_error(error, rc); return NULL; }
    tr_math_budget dflt;
    const tr_math_budget *b = budget;
    if (!b) { tr_budget_default(&dflt); b = &dflt; }
    int32_t npieces = m->kind == TR_PRE_PIECES ? m->npieces : 1;
    uint64_t nvalues = pr->op == TR_PRE_IN ? pr->nvalues : 0;
    /* needs before allocating: support = pieces / values; bytes = the set and the intermediate intervals (at most
     * 3 per branch and value, residue maps 2 per value) */
    tr_math_need need;
    for (int i = 0; i < TR_BUDGET_RESOURCES; i++) need.need[i] = 0;
    need.need[TR_BUDGET_SUPPORT] = (uint64_t)npieces > nvalues ? (uint64_t)npieces : nvalues;
    uint64_t ivals = tr_budget_mul(tr_budget_mul(3, (uint64_t)npieces), nvalues ? nvalues : 1);
    uint64_t bytes = tr_budget_add(sizeof(tr_pre_set), tr_budget_mul(ivals, sizeof(ziv) + 2 * sizeof(tr_pre_span)));
    bytes = tr_budget_add(bytes, tr_budget_mul((uint64_t)npieces, 20 + (BRANCH_NODES) * sizeof(tr_unc_node) + 8 + 64));
    bytes = tr_budget_add(bytes, tr_budget_mul(nvalues, sizeof(tr_num_rational)));
    need.need[TR_BUDGET_BYTES] = bytes;
    rc = tr_budget_check(b, &need, refusal);
    if (rc) { set_error(error, rc); return NULL; }
    ctx x = {b, refusal, 0};
    __int128 llo = (__int128)cd->lo + cd->offset, lhi = (__int128)cd->hi + cd->offset;
    /* the branches: the pieces' integer ranges must cover the domain exactly */
    branch *br = calloc((size_t)npieces, sizeof *br);
    tr_num_rational *vals = nvalues ? calloc((size_t)nvalues, sizeof *vals) : NULL;
    zlist out = {NULL, 0, 0};
    if (!br || (nvalues && !vals)) { free(br); free(vals); set_error(error, TR_MATH_MEMORY); return NULL; }
    int32_t nb = 0;
    if (m->kind == TR_PRE_PIECES) {
        for (int32_t i = 0; i < m->npieces && !x.rc; i++) {
            __int128 lo, hi;
            if (!piece_range(&x, &m->pieces[i], cd->scale, llo, lhi, &lo, &hi)) continue;
            br[nb].kind = m->pieces[i].kind; br[nb].a = m->pieces[i].a; br[nb].b = m->pieces[i].b;
            br[nb].llo = lo; br[nb].lhi = hi; nb++;
        }
        /* sort by range, then coverage */
        for (int32_t i = 1; i < nb; i++) {
            branch t = br[i];
            int32_t j = i;
            while (j > 0 && br[j - 1].llo > t.llo) { br[j] = br[j - 1]; j--; }
            br[j] = t;
        }
        if (!x.rc) {
            __int128 next = llo;
            for (int32_t i = 0; i < nb && !rc; i++) {
                if (br[i].llo < next) rc = TR_MATH_INVALID;            /* overlapping pieces: not a function */
                else if (br[i].llo > next) rc = TR_MATH_DOMAIN_ERROR;  /* a gap: a branch is missing */
                next = br[i].lhi + 1;
            }
            if (!rc && next != lhi + 1) rc = TR_MATH_DOMAIN_ERROR;
        }
    } else {
        br[0].kind = m->kind; br[0].a = m->a; br[0].b = m->b; br[0].m = m->modulus; br[0].llo = llo; br[0].lhi = lhi;
        nb = 1;
    }
    if (!rc) rc = x.rc;
    if (rc) { free(br); free(vals); set_error(error, rc); return NULL; }
    if (nvalues) {                                   /* IN: sorted distinct values */
        memcpy(vals, pr->values, (size_t)nvalues * sizeof *vals);
        qsort(vals, (size_t)nvalues, sizeof *vals, cmp_q);
        uint64_t k = 0;
        for (uint64_t i = 0; i < nvalues; i++) if (!k || tr_q_cmp(vals[k - 1], vals[i]) != 0) vals[k++] = vals[i];
        nvalues = k;
    }
    int32_t decided = 0, residue = 0;
    int64_t M = 1;
    tr_pre_span *spans = NULL;
    uint64_t nspans = 0;
    if (m->kind == TR_PRE_RESIDUE) {
        /* residues R = L mod M with R / T op c, then code residues (R - offset) mod M */
        M = (int64_t)((__int128)m->modulus * tr_unc_pow10(cd->scale));
        zlist rr = {NULL, 0, 0};
        tr_bigq *T = bq(&x, tr_q_int(tr_unc_pow10(cd->scale)));
        uint64_t n = pr->op == TR_PRE_IN ? nvalues : 1;
        for (uint64_t i = 0; i < n && !x.rc; i++) {
            tr_bigq *C = bq(&x, pr->op == TR_PRE_IN ? vals[i] : pr->c), *CT = q_mul(&x, C, T);
            if (!x.rc) solve_linear(&x, pr->op == TR_PRE_IN ? TR_PRE_EQ : pr->op, CT, 0, (__int128)M - 1, &rr);
            tr_bigq_free(C); tr_bigq_free(CT);
        }
        tr_bigq_free(T);
        __int128 off = tr_pre_fmod(cd->offset, M);
        for (uint64_t i = 0; i < rr.n && !x.rc; i++) {
            __int128 u1 = rr.v[i].lo - off, u2 = rr.v[i].hi - off;
            u1 %= M; if (u1 < 0) u1 += M;
            u2 %= M; if (u2 < 0) u2 += M;
            if (u1 <= u2) { if (!zl_push(&out, u1, u2)) x.rc = TR_MATH_MEMORY; }
            else if (!zl_push(&out, u1, (__int128)M - 1) || !zl_push(&out, 0, u2)) x.rc = TR_MATH_MEMORY;
        }
        free(rr.v);
        residue = 1;
    } else {
        for (int32_t i = 0; i < nb && !x.rc; i++) {
            int32_t d = classify(&br[i], pr->op, pr->c, cd->scale);
            if (d == TR_UNC_ALL) { decided++; if (!zl_push(&out, br[i].llo, br[i].lhi)) x.rc = TR_MATH_MEMORY; continue; }
            if (d == TR_UNC_NONE) { decided++; continue; }
            solve_branch(&x, &br[i], pr->op, pr->c, vals, nvalues, cd->scale, &out);
        }
    }
    rc = x.rc;
    /* sort and merge (adjacent intervals join) */
    if (!rc && out.n) {
        qsort(out.v, (size_t)out.n, sizeof *out.v, cmp_ziv);
        uint64_t k = 0;
        for (uint64_t i = 0; i < out.n; i++) {
            if (k && out.v[i].lo <= out.v[k - 1].hi + 1) { if (out.v[i].hi > out.v[k - 1].hi) out.v[k - 1].hi = out.v[i].hi; }
            else out.v[k++] = out.v[i];
        }
        out.n = k;
    }
    uint64_t domain_span = (uint64_t)((__int128)cd->hi - cd->lo);   /* domain size - 1 */
    if (!rc && residue) {
        if (out.n == 1 && out.v[0].lo == 0 && out.v[0].hi == (__int128)M - 1) {   /* every residue */
            out.v[0].lo = llo; out.v[0].hi = lhi; residue = 0;
        } else if (out.n && domain_span < (uint64_t)M) {                          /* to plain code intervals */
            zlist pl = {NULL, 0, 0};
            __int128 base = (__int128)cd->lo - tr_pre_fmod(cd->lo, M);
            for (uint64_t i = 0; i < out.n && !rc; i++)
                for (int w = 0; w < 2; w++) {
                    __int128 a = base + (__int128)w * M + out.v[i].lo, z = base + (__int128)w * M + out.v[i].hi;
                    if (a < cd->lo) a = cd->lo;
                    if (z > cd->hi) z = cd->hi;
                    if (a <= z && !zl_push(&pl, a + cd->offset, z + cd->offset)) rc = TR_MATH_MEMORY;
                }
            free(out.v); out = pl; residue = 0;
            if (!rc && out.n) {
                qsort(out.v, (size_t)out.n, sizeof *out.v, cmp_ziv);
                uint64_t k = 0;
                for (uint64_t i = 0; i < out.n; i++) {
                    if (k && out.v[i].lo <= out.v[k - 1].hi + 1) { if (out.v[i].hi > out.v[k - 1].hi) out.v[k - 1].hi = out.v[i].hi; }
                    else out.v[k++] = out.v[i];
                }
                out.n = k;
            }
        }
    }
    if (!rc) rc = tr_budget_check_one(b, TR_BUDGET_OUTPUT, out.n, refusal);
    if (!rc && out.n) {
        spans = calloc((size_t)out.n, sizeof *spans);
        if (!spans) rc = TR_MATH_MEMORY;
    }
    uint64_t codes = 0;
    for (uint64_t i = 0; !rc && i < out.n; i++) {
        tr_pre_span *s = &spans[i];
        if (residue) {
            s->lo = cd->lo; s->hi = cd->hi; s->modulus = M; s->rlo = (int64_t)out.v[i].lo; s->rhi = (int64_t)out.v[i].hi;
            uint64_t c = count_residue(cd->lo, cd->hi, M, out.v[i].lo, out.v[i].hi);
            codes = codes > UINT64_MAX - c ? UINT64_MAX : codes + c;
        } else {
            s->lo = (int64_t)(out.v[i].lo - cd->offset); s->hi = (int64_t)(out.v[i].hi - cd->offset);
            s->modulus = 1; s->rlo = 0; s->rhi = 0;
            uint64_t c = (uint64_t)(s->hi - s->lo);
            codes = (c == UINT64_MAX || codes > UINT64_MAX - c - 1) ? UINT64_MAX : codes + c + 1;
        }
    }
    nspans = rc ? 0 : out.n;
    free(out.v);
    tr_pre_set *s = NULL;
    if (!rc) {
        s = calloc(1, sizeof *s);
        if (!s) rc = TR_MATH_MEMORY;
    }
    if (!rc) {
        s->magic = SET_MAGIC; s->kind = m->kind; s->op = pr->op; s->codec = *cd; s->c = pr->op == TR_PRE_IN ? tr_q_int(0) : pr->c;
        s->spans = spans; s->nspans = nspans; s->codes = codes; s->residue = residue; s->modulus = residue ? M : 1;
        s->branches = nb; s->decided = decided; s->values = vals; s->nvalues = nvalues; vals = NULL; spans = NULL;
        s->flags = (nspans == 0 ? TR_PRE_SET_EMPTY : 0) | (residue ? TR_PRE_SET_RESIDUE : 0);
        if (!residue && nspans == 1 && s->spans[0].lo == cd->lo && s->spans[0].hi == cd->hi) s->flags |= TR_PRE_SET_ALL;
        /* the branches of the direct check (code ranges, programs y = f(x)) and their laws */
        s->nbranch = nb;
        s->branch_lo = calloc((size_t)nb, 8); s->branch_hi = calloc((size_t)nb, 8);
        s->node_off = calloc((size_t)nb + 1, 4); s->nodes = calloc((size_t)nb * BRANCH_NODES, sizeof *s->nodes);
        s->certs = calloc((size_t)nb, sizeof *s->certs);
        if (!s->branch_lo || !s->branch_hi || !s->node_off || !s->nodes || !s->certs) rc = TR_MATH_MEMORY;
        int32_t off = 0;
        for (int32_t i = 0; i < nb && !rc; i++) {
            s->branch_lo[i] = (int64_t)(br[i].llo - cd->offset); s->branch_hi[i] = (int64_t)(br[i].lhi - cd->offset);
            tr_unc_node nd[BRANCH_NODES + 2];
            int32_t n = branch_program(&br[i], 0, tr_q_int(0), nd);
            memcpy(s->nodes + off, nd, (size_t)n * sizeof *nd);
            s->node_off[i] = off; off += n;
            s->certs[i] = branch_law(&br[i], cd, b, refusal);
            tr_budget_none(refusal);
            if (s->certs[i]) s->certified++;
        }
        s->node_off[nb] = off; s->nnodes = (uint64_t)off;
        s->bytes = sizeof *s + nspans * sizeof(tr_pre_span) + s->nvalues * sizeof(tr_num_rational) + (uint64_t)nb * 20 +
                   (uint64_t)off * sizeof(tr_unc_node);
    }
    free(br); free(vals); free(spans);
    if (rc) { tr_pre_set_free(s); set_error(error, rc); return NULL; }
    set_error(error, TR_MATH_OK);
    return s;
}
int32_t tr_pre_set_get_info(const tr_pre_set *s, tr_pre_set_info *out) {
    if (!set_valid(s) || !out) return TR_MATH_INVALID;
    memset(out, 0, sizeof *out);
    out->nspans = s->nspans; out->codes = s->codes; out->flags = s->flags; out->nullable = s->codec.nullable;
    out->branches = s->branches; out->decided = s->decided; out->certified = s->certified; out->kind = s->kind;
    out->modulus = s->modulus; out->codec = s->codec;
    return TR_MATH_OK;
}
int32_t tr_pre_set_spans(const tr_pre_set *s, tr_pre_span *out, uint64_t capacity) {
    if (!set_valid(s) || (s->nspans && !out) || capacity < s->nspans) return TR_MATH_INVALID;
    if (s->nspans) memcpy(out, s->spans, (size_t)s->nspans * sizeof *out);
    return TR_MATH_OK;
}
int32_t tr_pre_set_get_view(const tr_pre_set *s, tr_pre_view *v) {
    if (!set_valid(s) || !v) return TR_MATH_INVALID;
    memset(v, 0, sizeof *v);
    v->codec = s->codec; v->op = s->op; v->residue = s->residue; v->nbranch = s->nbranch;
    v->nspans = s->nspans; v->spans = s->spans; v->c = s->c; v->nvalues = s->nvalues; v->values = s->values;
    v->branch_lo = s->branch_lo; v->branch_hi = s->branch_hi; v->node_off = s->node_off; v->nodes = s->nodes;
    v->nnodes = s->nnodes;
    return TR_MATH_OK;
}
int32_t tr_pre_set_contains(const tr_pre_set *s, int64_t code) {
    tr_pre_view v;
    if (tr_pre_set_get_view(s, &v)) return TR_MATH_INVALID;
    return tr_pre_member(&v, code);
}
const tr_cert *tr_pre_set_certificate(const tr_pre_set *s, int32_t i) {
    return set_valid(s) && i >= 0 && i < s->nbranch ? s->certs[i] : NULL;
}
void tr_pre_set_free(tr_pre_set *s) {
    if (!set_valid(s)) return;
    s->magic = 0;
    if (s->certs) for (int32_t i = 0; i < s->nbranch; i++) tr_cert_free(s->certs[i]);
    free(s->certs); free(s->spans); free(s->values); free(s->branch_lo); free(s->branch_hi); free(s->node_off);
    free(s->nodes);
    free(s);
}

/* ------------------------------------------------------------------ scanning */
int32_t tr_pre_scan(const tr_pre_set *s, const tr_scope *scope, int32_t column, int32_t direct, uint64_t *bitmap,
                    tr_pre_scan_stats *stats, const tr_math_budget *budget, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    tr_scope_view sv;
    tr_pre_view v;
    if (tr_pre_set_get_view(s, &v) || !scope || tr_scope_get_view(scope, &sv) || !stats || column < 0 ||
        column >= sv.ncols || (sv.rows && !bitmap) || (direct != 0 && direct != 1)) return TR_MATH_INVALID;
    if (sv.col[column].scale != v.codec.scale) return TR_MATH_INVALID;       /* another value unit */
    tr_math_budget dflt;
    if (!budget) { tr_budget_default(&dflt); budget = &dflt; }
    int32_t rc = tr_budget_check_one(budget, TR_BUDGET_BYTES, tr_budget_bitmap_bytes(sv.rows), refusal);
    if (rc) return rc;
    memset(stats, 0, sizeof *stats);
    stats->rows = sv.rows; stats->selected = sv.selected;
    stats->first_mismatch = UINT64_MAX; stats->first_overflow = UINT64_MAX;
    uint64_t acc[6] = {0, 0, 0, 0, 0, 0}, nw = (sv.rows + 63) / 64;
    tr_unc_rowval vals[BRANCH_NODES + 2];
    for (uint64_t w = 0; w < nw; w++)
        bitmap[w] = tr_pre_word(&v, &sv, column, direct, w, vals, acc, &stats->first_mismatch, &stats->first_overflow);
    stats->nulls = acc[0]; stats->in_set = acc[1]; stats->direct_true = acc[2]; stats->mismatches = acc[3];
    stats->overflow = acc[4]; stats->outside = acc[5];
    return stats->overflow ? TR_MATH_OVERFLOW : TR_MATH_OK;
}
