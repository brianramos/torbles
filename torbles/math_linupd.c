/* LIN-003: leave-group-out and low-rank model updates on LIN-001 / LIN-002 states; every contract is in
 * math_linupd.h.
 * SPDX-License-Identifier: Apache-2.0 */
#include "math_linupd.h"
#include "math_handles.h"
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

TR_MATH_ABI(tr_linupd_info, TR_LINUPD_ABI);         /* the ctypes boundary of LIN-003 (math_handles.h) */
TR_MATH_ABI(tr_linupd_logo_view, TR_LINUPD_ABI);
TR_MATH_ABI(tr_linupd_float_info, TR_LINUPD_ABI);

_Static_assert(sizeof(tr_linupd_info) == 152 && sizeof(tr_linupd_logo_view) == 296 &&
               sizeof(tr_linupd_float_info) == 48, "LIN-003 ABI");

#define MODEL_MAGIC 0x31444d4cu   /* "LMD1" */
#define LOGO_MAGIC 0x31474c4cu    /* "LLG1" */

static void seterr(int32_t *e, int32_t v) { if (e) *e = v; }
static void *zalloc(uint64_t bytes) {
    if (bytes > (uint64_t)SIZE_MAX) return NULL;
    return calloc(1, bytes ? (size_t)bytes : 1);
}
static const tr_math_budget *budget_or(const tr_math_budget *b, tr_math_budget *store) {
    if (b) return b;
    tr_budget_default(store);
    return store;
}

/* ================================================================ exact arithmetic with a sticky error */
typedef struct { const tr_math_budget *b; tr_math_refusal *r; int32_t rc; } lctx;
#define LIVE(x) (!(x)->rc)
static void lfail(lctx *x, int32_t rc) { if (!x->rc) x->rc = rc ? rc : TR_MATH_MEMORY; }
static tr_bigint *zk(lctx *x, tr_bigint *v, int32_t e) { if (!v) lfail(x, e); return v; }
static tr_bigq *qk(lctx *x, tr_bigq *v, int32_t e) { if (!v) lfail(x, e); return v; }
static tr_bigint *Z_i64(lctx *x, int64_t v) { int32_t e = 0; return x->rc ? NULL : zk(x, tr_bigint_from_i64(v, x->b, x->r, &e), e); }
static tr_bigint *Z_copy(lctx *x, const tr_bigint *a) { int32_t e = 0; return x->rc ? NULL : zk(x, tr_bigint_copy(a, x->b, x->r, &e), e); }
static tr_bigint *Z_sub(lctx *x, const tr_bigint *a, const tr_bigint *c) { int32_t e = 0; return x->rc ? NULL : zk(x, tr_bigint_sub(a, c, x->b, x->r, &e), e); }
static tr_bigint *Z_add(lctx *x, const tr_bigint *a, const tr_bigint *c) { int32_t e = 0; return x->rc ? NULL : zk(x, tr_bigint_add(a, c, x->b, x->r, &e), e); }
static tr_bigint *Z_mul(lctx *x, const tr_bigint *a, const tr_bigint *c) { int32_t e = 0; return x->rc ? NULL : zk(x, tr_bigint_mul(a, c, x->b, x->r, &e), e); }
static tr_bigint *Z_div(lctx *x, const tr_bigint *a, const tr_bigint *c) { int32_t e = 0; return x->rc ? NULL : zk(x, tr_bigint_divexact(a, c, x->b, x->r, &e), e); }
static tr_bigint *Z_gcd(lctx *x, const tr_bigint *a, const tr_bigint *c) { int32_t e = 0; return x->rc ? NULL : zk(x, tr_bigint_gcd(a, c, x->b, x->r, &e), e); }
static tr_bigint *Z_twos(lctx *x, const uint64_t *w, uint64_t n) { int32_t e = 0; return x->rc ? NULL : zk(x, tr_bigint_from_twos(w, n, x->b, x->r, &e), e); }
static tr_bigint *Z_dot(lctx *x, const tr_bigint *const *a, const tr_bigint *const *b, uint64_t n) {
    int32_t e = 0;
    return x->rc ? NULL : zk(x, tr_bigint_dot(a, b, n, x->b, x->r, &e), e);
}
static tr_bigint *Z_pow10(lctx *x, int32_t k) {
    tr_bigint *ten = Z_i64(x, 10);
    int32_t e = 0;
    tr_bigint *p = x->rc ? NULL : zk(x, tr_bigint_pow(ten, (uint64_t)k, x->b, x->r, &e), e);
    tr_bigint_free(ten);
    return p;
}
static int Z_sign(const tr_bigint *a) { return a ? tr_bigint_sign(a) : 0; }
static int Z_is_one(const tr_bigint *a) { return a && tr_bigint_sign(a) == 1 && tr_bigint_bit_length(a) == 1; }
/* a * b - c * d */
static tr_bigint *Z_cross(lctx *x, const tr_bigint *a, const tr_bigint *b, const tr_bigint *c, const tr_bigint *d) {
    int p = Z_sign(a) && Z_sign(b), q = Z_sign(c) && Z_sign(d);
    if (!p && !q) return Z_i64(x, 0);
    if (!q) return Z_mul(x, a, b);
    tr_bigint *t2 = Z_mul(x, c, d);
    if (!p) { int32_t e = 0; tr_bigint *r = x->rc ? NULL : zk(x, tr_bigint_neg(t2, x->b, x->r, &e), e); tr_bigint_free(t2); return r; }
    tr_bigint *t1 = Z_mul(x, a, b), *r = Z_sub(x, t1, t2);
    tr_bigint_free(t1); tr_bigint_free(t2);
    return r;
}
static tr_bigq *Q_q64(lctx *x, int64_t n, int64_t d) {
    tr_num_rational q; q.numerator = n; q.denominator = d;
    int32_t e = 0;
    return x->rc ? NULL : qk(x, tr_bigq_from_q64(q, x->b, x->r, &e), e);
}
static tr_bigq *Q_int(lctx *x, int64_t v) { return Q_q64(x, v, 1); }
static tr_bigq *Q_frac(lctx *x, const tr_bigint *n, const tr_bigint *d) { int32_t e = 0; return x->rc ? NULL : qk(x, tr_bigq_from_bigints(n, d, x->b, x->r, &e), e); }
static tr_bigq *Q_z(lctx *x, const tr_bigint *n) { int32_t e = 0; return x->rc ? NULL : qk(x, tr_bigq_from_bigint(n, x->b, x->r, &e), e); }
static tr_bigq *Q_copy(lctx *x, const tr_bigq *a) { int32_t e = 0; return x->rc ? NULL : qk(x, tr_bigq_copy(a, x->b, x->r, &e), e); }
static tr_bigq *Q_add(lctx *x, const tr_bigq *a, const tr_bigq *c) { int32_t e = 0; return x->rc ? NULL : qk(x, tr_bigq_add(a, c, x->b, x->r, &e), e); }
static tr_bigq *Q_sub(lctx *x, const tr_bigq *a, const tr_bigq *c) { int32_t e = 0; return x->rc ? NULL : qk(x, tr_bigq_sub(a, c, x->b, x->r, &e), e); }
static tr_bigq *Q_mul(lctx *x, const tr_bigq *a, const tr_bigq *c) { int32_t e = 0; return x->rc ? NULL : qk(x, tr_bigq_mul(a, c, x->b, x->r, &e), e); }
static tr_bigq *Q_div(lctx *x, const tr_bigq *a, const tr_bigq *c) { int32_t e = 0; return x->rc ? NULL : qk(x, tr_bigq_div(a, c, x->b, x->r, &e), e); }
static tr_bigq *Q_neg(lctx *x, const tr_bigq *a) { int32_t e = 0; return x->rc ? NULL : qk(x, tr_bigq_neg(a, x->b, x->r, &e), e); }
/* *acc += a * c (acc owned, replaced) */
static void Q_fma(lctx *x, tr_bigq **acc, const tr_bigq *a, const tr_bigq *c) {
    if (x->rc || !tr_bigq_sign(a) || !tr_bigq_sign(c)) return;
    tr_bigq *p = Q_mul(x, a, c), *s = Q_add(x, *acc, p);
    tr_bigq_free(p);
    if (s) { tr_bigq_free(*acc); *acc = s; }
}
/* *acc += a (acc owned, replaced) */
static void Q_acc(lctx *x, tr_bigq **acc, const tr_bigq *a) {
    if (x->rc || !tr_bigq_sign(a)) return;
    tr_bigq *s = Q_add(x, *acc, a);
    if (s) { tr_bigq_free(*acc); *acc = s; }
}
static tr_bigq *Q_scale10(lctx *x, const tr_bigint *n, int32_t e) {   /* n / 10^e */
    if (!e) return Q_z(x, n);
    tr_bigint *d = Z_pow10(x, e);
    tr_bigq *q = Q_frac(x, n, d);
    tr_bigint_free(d);
    return q;
}
static tr_bigq **Q_arr(lctx *x, uint64_t n) {
    if (x->rc) return NULL;
    tr_bigq **v = (tr_bigq **)zalloc(n * sizeof *v);
    if (!v) lfail(x, TR_MATH_MEMORY);
    return v;
}
static tr_bigint **Z_arr(lctx *x, uint64_t n) {
    if (x->rc) return NULL;
    tr_bigint **v = (tr_bigint **)zalloc(n * sizeof *v);
    if (!v) lfail(x, TR_MATH_MEMORY);
    return v;
}
static void Q_free_all(tr_bigq **v, uint64_t n) { if (v) { for (uint64_t i = 0; i < n; i++) tr_bigq_free(v[i]); free(v); } }
static void Z_free_all(tr_bigint **v, uint64_t n) { if (v) { for (uint64_t i = 0; i < n; i++) tr_bigint_free(v[i]); free(v); } }
static tr_bigq **Q_copy_all(lctx *x, tr_bigq *const *v, uint64_t n) {
    tr_bigq **o = Q_arr(x, n);
    for (uint64_t i = 0; o && i < n && LIVE(x); i++) o[i] = v[i] ? Q_copy(x, v[i]) : NULL;
    if (x->rc) { Q_free_all(o, o ? n : 0); return NULL; }
    return o;
}
static int Q_eq(const tr_bigq *a, const tr_bigq *b) { return (!a && !b) || (a && b && tr_bigq_cmp(a, b) == 0); }

/* ================================================================ fraction-free Gauss-Jordan (Bareiss) */
/* In place on the m x c integer matrix M, pivots searched in the columns < pc only: returns the rank r (pivcol[0..r)),
 * *det = the last pivot and *swaps the parity of the row exchanges; the pivot rows then hold det times the reduced
 * row echelon form (the pivot columns det * I) and the other rows of the pivot columns are zero. */
static int32_t bareiss(lctx *x, tr_bigint **M, int32_t m, int32_t c, int32_t pc, int32_t *pivcol, tr_bigint **det,
                       int *swaps) {
    tr_bigint *prev = Z_i64(x, 1);
    int32_t r = 0;
    *swaps = 0;
    for (int32_t col = 0; col < pc && r < m && LIVE(x); col++) {
        int32_t i = r;
        while (i < m && !Z_sign(M[(uint64_t)i * c + col])) i++;
        if (i == m) continue;
        if (i != r) {
            for (int32_t j = 0; j < c; j++) {
                tr_bigint *t = M[(uint64_t)i * c + j];
                M[(uint64_t)i * c + j] = M[(uint64_t)r * c + j];
                M[(uint64_t)r * c + j] = t;
            }
            *swaps ^= 1;
        }
        const tr_bigint *piv = M[(uint64_t)r * c + col];
        int one = Z_is_one(prev);
        for (int32_t i2 = 0; i2 < m && LIVE(x); i2++) {
            if (i2 == r) continue;
            tr_bigint **row = M + (uint64_t)i2 * c, **prow = M + (uint64_t)r * c;
            const tr_bigint *f = row[col];
            for (int32_t j = 0; j < c && LIVE(x); j++) {
                if (j == col) continue;
                if (!Z_sign(row[j]) && (!Z_sign(f) || !Z_sign(prow[j]))) continue;
                tr_bigint *t = Z_cross(x, piv, row[j], f, prow[j]);
                tr_bigint *nv = one ? t : Z_div(x, t, prev);
                if (!one) tr_bigint_free(t);
                if (!nv) break;
                tr_bigint_free(row[j]); row[j] = nv;
            }
            tr_bigint *zero = Z_i64(x, 0);
            if (zero) { tr_bigint_free(row[col]); row[col] = zero; }
        }
        tr_bigint *np = Z_copy(x, piv);
        if (np) { tr_bigint_free(prev); prev = np; }
        pivcol[r++] = col;
    }
    *det = prev;
    return r;
}

/* ================================================================ exact inverses and solves of small rational systems */
/* Fraction-free Gauss-Jordan elimination of [M | diag(L) | C] with M = diag(L) A and C = diag(L) B the row-scaled
 * integer forms of the n x n rational A and the n x nrhs rational B (L_i: the lcm of the denominators of row i):
 * A^-1 = block / D (when inv), A^-1 B = C block / D (when sol), det(A) = (-1)^swaps D / prod L, every block verified
 * exactly (M times it equals D times the scaled identity / C). Returns 1 for a nonsingular A, 0 for a singular one
 * (nothing set), with the sticky error on failure. */
static int rat_solve(lctx *x, int32_t n, const tr_bigq *const *A, const tr_bigq *const *B, int32_t nrhs,
                     tr_bigq ***inv, tr_bigq ***sol, tr_bigq **det) {
    const int32_t ni = inv ? n : 0, c = n + ni + nrhs;
    const uint64_t cells = (uint64_t)n * (uint64_t)c;
    tr_bigint **W = Z_arr(x, cells), **M = Z_arr(x, (uint64_t)n * (uint64_t)(n + nrhs)), **L = Z_arr(x, (uint64_t)n);
    for (int32_t i = 0; W && M && L && i < n && LIVE(x); i++) {
        tr_bigint *l = Z_i64(x, 1);
        for (int32_t j = 0; j < n + nrhs && LIVE(x); j++) {
            const tr_bigq *e = j < n ? A[(uint64_t)i * n + j] : B[(uint64_t)i * nrhs + (j - n)];
            const tr_bigint *d = tr_bigq_denominator(e);
            if (Z_is_one(d)) continue;
            tr_bigint *g = Z_gcd(x, l, d), *t = Z_div(x, l, g), *nl = Z_mul(x, t, d);
            tr_bigint_free(g); tr_bigint_free(t);
            if (nl) { tr_bigint_free(l); l = nl; }
        }
        L[i] = l;
        for (int32_t j = 0; j < n + nrhs && LIVE(x); j++) {
            const tr_bigq *e = j < n ? A[(uint64_t)i * n + j] : B[(uint64_t)i * nrhs + (j - n)];
            tr_bigint *t = Z_div(x, l, tr_bigq_denominator(e));
            M[(uint64_t)i * (n + nrhs) + j] = Z_mul(x, tr_bigq_numerator(e), t);
            tr_bigint_free(t);
        }
        for (int32_t j = 0; j < c && LIVE(x); j++) {
            uint64_t at = (uint64_t)i * c + j;
            if (j < n) W[at] = Z_copy(x, M[(uint64_t)i * (n + nrhs) + j]);
            else if (j < n + ni) W[at] = (j - n == i) ? Z_copy(x, l) : Z_i64(x, 0);
            else W[at] = Z_copy(x, M[(uint64_t)i * (n + nrhs) + n + (j - n - ni)]);
        }
    }
    int32_t pivcol[TR_LIN_MAX_ROWS], r = 0;
    int swaps = 0;
    tr_bigint *D = NULL;
    if (LIVE(x)) r = bareiss(x, W, n, c, n, pivcol, &D, &swaps);
    int ok = LIVE(x) && r == n;
    /* verification: M times each right block column equals D times the scaled identity column / C column */
    const tr_bigint *row[TR_LIN_MAX_ROWS], *col[TR_LIN_MAX_ROWS];
    for (int32_t j = n; ok && j < c && LIVE(x); j++) {
        for (int32_t t = 0; t < n; t++) col[t] = W[(uint64_t)t * c + j];
        for (int32_t i = 0; i < n && LIVE(x); i++) {
            for (int32_t t = 0; t < n; t++) row[t] = M[(uint64_t)i * (n + nrhs) + t];
            tr_bigint *lhs = Z_dot(x, row, col, (uint64_t)n), *rhs = NULL;
            if (j < n + ni) rhs = (j - n == i) ? Z_mul(x, D, L[i]) : Z_i64(x, 0);
            else rhs = Z_mul(x, D, M[(uint64_t)i * (n + nrhs) + n + (j - n - ni)]);
            if (LIVE(x) && tr_bigint_cmp(lhs, rhs) != 0) lfail(x, TR_MATH_INVALID);   /* impossible: exact */
            tr_bigint_free(lhs); tr_bigint_free(rhs);
        }
    }
    if (ok && LIVE(x)) {
        if (inv) {
            *inv = Q_arr(x, (uint64_t)n * n);
            for (int32_t i = 0; *inv && i < n && LIVE(x); i++)
                for (int32_t j = 0; j < n && LIVE(x); j++) (*inv)[(uint64_t)i * n + j] = Q_frac(x, W[(uint64_t)i * c + n + j], D);
        }
        if (sol) {
            *sol = Q_arr(x, (uint64_t)n * nrhs);
            for (int32_t i = 0; *sol && i < n && LIVE(x); i++)
                for (int32_t j = 0; j < nrhs && LIVE(x); j++)
                    (*sol)[(uint64_t)i * nrhs + j] = Q_frac(x, W[(uint64_t)i * c + n + ni + j], D);
        }
        if (det) {
            tr_bigint *p = Z_i64(x, swaps ? -1 : 1);
            for (int32_t i = 0; i < n && LIVE(x); i++) { tr_bigint *t = Z_mul(x, p, L[i]); tr_bigint_free(p); p = t; }
            *det = Q_frac(x, D, p);
            tr_bigint_free(p);
        }
    }
    tr_bigint_free(D);
    Z_free_all(W, W ? cells : 0); Z_free_all(M, M ? (uint64_t)n * (uint64_t)(n + nrhs) : 0); Z_free_all(L, L ? (uint64_t)n : 0);
    if (x->rc) {
        if (inv && *inv) { Q_free_all(*inv, (uint64_t)n * n); *inv = NULL; }
        if (sol && *sol) { Q_free_all(*sol, (uint64_t)n * nrhs); *sol = NULL; }
        if (det && *det) { tr_bigq_free(*det); *det = NULL; }
        return 0;
    }
    return ok;
}

/* ================================================================ models */
struct tr_linupd_model_s {
    uint32_t magic;
    int32_t flags;                             /* TR_LINUPD_INVERSE requested (kept by updates) */
    tr_linupd_info info;
    int32_t pen[TR_LIN_MAX_COLS];              /* 1: unknown u carries the ridge (a feature) */
    int32_t k, policy, nfeatures, target;      /* the origin moment state (regression models) */
    int32_t col[TR_LIN_MAX_COLS], scale[TR_LIN_MAX_COLS];
    int64_t ridge_num, ridge_den;
    tr_bigq **part[TR_LINUPD_PARTS];
    uint64_t npart[TR_LINUPD_PARTS];
    tr_bigq *yty, *sy;                         /* regression: y^T y and sum y (value units) */
};
static int model_ok(const tr_linupd_model *m) { return m && m->magic == MODEL_MAGIC; }
void tr_linupd_free(tr_linupd_model *m) {
    if (!model_ok(m)) return;
    m->magic = 0;
    for (int p = 0; p < TR_LINUPD_PARTS; p++) Q_free_all(m->part[p], m->npart[p]);
    tr_bigq_free(m->yty); tr_bigq_free(m->sy);
    free(m);
}
static tr_linupd_model *model_new(lctx *x) {
    if (x->rc) return NULL;
    tr_linupd_model *m = (tr_linupd_model *)zalloc(sizeof *m);
    if (!m) { lfail(x, TR_MATH_MEMORY); return NULL; }
    m->magic = MODEL_MAGIC;
    m->info.denominator_sign = 2;
    return m;
}
static void set_part(tr_linupd_model *m, int32_t p, tr_bigq **v, uint64_t n) {     /* takes v */
    Q_free_all(m->part[p], m->npart[p]);
    m->part[p] = v; m->npart[p] = v ? n : 0;
}
static tr_bigq **Q_from(lctx *x, const tr_bigq *const *v, uint64_t n) {
    tr_bigq **o = Q_arr(x, n);
    for (uint64_t i = 0; o && i < n && LIVE(x); i++) o[i] = Q_copy(x, v[i]);
    if (x->rc) { Q_free_all(o, o ? n : 0); return NULL; }
    return o;
}
/* everything but the system and its solution (the design part included) */
static void model_identity(lctx *x, tr_linupd_model *m, const tr_linupd_model *src) {
    m->flags = src->flags;
    m->info = src->info;
    m->info.inverse = 0; m->info.fallback = 0; m->info.update_rank = 0; m->info.denominator_sign = 2;
    m->info.primes = 0;
    memcpy(m->pen, src->pen, sizeof m->pen);
    m->k = src->k; m->policy = src->policy; m->nfeatures = src->nfeatures; m->target = src->target;
    memcpy(m->col, src->col, sizeof m->col); memcpy(m->scale, src->scale, sizeof m->scale);
    m->ridge_num = src->ridge_num; m->ridge_den = src->ridge_den;
    if (src->part[TR_LINUPD_PART_DESIGN])
        set_part(m, TR_LINUPD_PART_DESIGN, Q_copy_all(x, src->part[TR_LINUPD_PART_DESIGN], src->npart[TR_LINUPD_PART_DESIGN]),
                 src->npart[TR_LINUPD_PART_DESIGN]);
}
static void set_design(lctx *x, tr_linupd_model *m, const int32_t *design, int32_t n) {
    tr_bigq **ds = Q_arr(x, (uint64_t)n);
    for (int32_t u = 0; ds && u < n && LIVE(x); u++) { ds[u] = Q_int(x, design[u]); m->info.design[u] = design[u]; }
    if (x->rc) { Q_free_all(ds, ds ? (uint64_t)n : 0); return; }
    set_part(m, TR_LINUPD_PART_DESIGN, ds, (uint64_t)n);
    for (int32_t u = 0; u < n; u++) m->pen[u] = design[u] != 0;
}

/* the exact solution of the model's system (parts A, B): X, NULL, CERT, kind, rank, pivots; with want_inverse and a
 * nonsingular A also INVERSE and DET (verified), else LIN-002's canonical solve */
static void solve_core(lctx *x, tr_linupd_model *m, int want_inverse) {
    const int32_t n = m->info.n;
    const tr_bigq *const *A = (const tr_bigq *const *)m->part[TR_LINUPD_PART_A];
    const tr_bigq *const *b = (const tr_bigq *const *)m->part[TR_LINUPD_PART_B];
    m->info.inverse = 0;
    if (want_inverse && LIVE(x)) {
        tr_bigq **inv = NULL, **sol = NULL, *det = NULL;
        if (rat_solve(x, n, A, b, 1, &inv, &sol, &det)) {
            set_part(m, TR_LINUPD_PART_INVERSE, inv, (uint64_t)n * n);
            set_part(m, TR_LINUPD_PART_X, sol, (uint64_t)n);
            tr_bigq **dd = Q_arr(x, 1);
            if (dd) { dd[0] = det; set_part(m, TR_LINUPD_PART_DET, dd, 1); } else tr_bigq_free(det);
            m->info.kind = TR_LIN_UNIQUE; m->info.rank = n;
            m->info.pivots = n >= 64 ? ~(uint64_t)0 : (((uint64_t)1 << n) - 1);
            m->info.inverse = LIVE(x);
            return;
        }
    }
    if (!LIVE(x)) return;
    int32_t e = 0;
    tr_lin_result *res = tr_lin_q_solve(n, n, A, b, x->b, x->r, &e);
    if (!res) { lfail(x, e); return; }
    tr_lin_result_info ri;
    tr_lin_result_get_info(res, &ri);
    m->info.kind = ri.status; m->info.rank = ri.rank; m->info.pivots = ri.pivots;
    static const int32_t from[3] = {TR_LIN_PART_X, TR_LIN_PART_NULL, TR_LIN_PART_CERT};
    static const int32_t to[3] = {TR_LINUPD_PART_X, TR_LINUPD_PART_NULL, TR_LINUPD_PART_CERT};
    for (int i = 0; i < 3 && LIVE(x); i++) {
        uint64_t cnt = 0;
        const tr_bigq *const *p = tr_lin_result_part(res, from[i], &cnt);
        set_part(m, to[i], p ? Q_from(x, p, cnt) : NULL, cnt);
    }
    tr_lin_result_free(res);
}

/* LIN-002's statistics of a regression model (common-listwise; pairwise: y^T y only) */
static void model_stats(lctx *x, tr_linupd_model *m) {
    if (!m->info.regression || x->rc) return;
    tr_bigq **st = Q_arr(x, TR_LIN_STATS);
    if (!st) return;
    st[4] = Q_copy(x, m->yty);
    tr_bigq **xs = m->part[TR_LINUPD_PART_X], **A0 = m->part[TR_LINUPD_PART_A0], **b = m->part[TR_LINUPD_PART_B];
    const int32_t n = m->info.n;
    const uint64_t rows = m->info.rows;
    if (m->info.listwise && xs && LIVE(x)) {
        st[5] = Q_q64(x, (int64_t)(rows > (uint64_t)INT64_MAX ? INT64_MAX : rows), 1);
        tr_bigq *xb = Q_int(x, 0), *xax = Q_int(x, 0);
        for (int32_t u = 0; u < n && LIVE(x); u++) {
            Q_fma(x, &xb, xs[u], b[u]);
            for (int32_t w = 0; w < n && LIVE(x); w++) {
                if (!tr_bigq_sign(xs[u]) || !tr_bigq_sign(xs[w])) continue;
                tr_bigq *t = Q_mul(x, xs[u], A0[(uint64_t)u * n + w]);
                Q_fma(x, &xax, t, xs[w]);
                tr_bigq_free(t);
            }
        }
        tr_bigq *two = Q_int(x, 2), *t2 = Q_mul(x, two, xb), *t3 = Q_sub(x, m->yty, t2);
        st[0] = Q_add(x, t3, xax);                                         /* rss */
        st[6] = Q_copy(x, xax);                                            /* ||X x||^2 */
        tr_bigq_free(two); tr_bigq_free(t2); tr_bigq_free(t3); tr_bigq_free(xb); tr_bigq_free(xax);
        if (m->info.intercept) {
            if (rows) {
                tr_bigq *nn = Q_q64(x, (int64_t)rows, 1), *s2 = Q_mul(x, m->sy, m->sy), *q = Q_div(x, s2, nn);
                st[1] = Q_sub(x, m->yty, q);
                tr_bigq_free(nn); tr_bigq_free(s2); tr_bigq_free(q);
            }
        } else st[1] = Q_copy(x, m->yty);
        if (st[1] && LIVE(x) && tr_bigq_sign(st[1])) {
            tr_bigq *one = Q_int(x, 1), *q = Q_div(x, st[0], st[1]);
            st[2] = Q_sub(x, one, q);
            tr_bigq_free(one); tr_bigq_free(q);
        }
        if (!m->info.ridged && rows > (uint64_t)m->info.rank && LIVE(x)) {
            tr_bigq *df = Q_q64(x, (int64_t)(rows - (uint64_t)m->info.rank), 1);
            st[3] = Q_div(x, st[0], df);
            tr_bigq_free(df);
        }
    }
    if (x->rc) { Q_free_all(st, TR_LIN_STATS); return; }
    set_part(m, TR_LINUPD_PART_STATS, st, TR_LIN_STATS);
}
static tr_linupd_model *model_done(lctx *x, tr_linupd_model *m, int32_t *error) {
    if (x->rc) { tr_linupd_free(m); seterr(error, x->rc); return NULL; }
    seterr(error, TR_MATH_OK);
    return m;
}
/* a model's needs: variables n; tensor n (2n + 1) (the elimination) or n r + r^2 + n^2 (an update of rank r);
 * output n (n + 1) (the solution and the inverse; LIN-002's need of one solve) */
static int32_t model_need(int32_t n, uint64_t r, const tr_math_budget *budget, tr_math_refusal *refusal) {
    tr_math_need need;
    memset(&need, 0, sizeof need);
    need.need[TR_BUDGET_VARIABLES] = (uint64_t)n;
    uint64_t nn = (uint64_t)n * (uint64_t)n;
    need.need[TR_BUDGET_TENSOR] = r ? tr_budget_add(tr_budget_add((uint64_t)n * r, r * r), nn) : (uint64_t)n * (uint64_t)(2 * n + 1);
    need.need[TR_BUDGET_OUTPUT] = (uint64_t)n * (uint64_t)(n + 1);
    tr_math_budget dflt;
    return tr_budget_check(budget_or(budget, &dflt), &need, refusal);
}

tr_linupd_model *tr_linupd_fit(const tr_lin_moments *st, const tr_lin_fit_spec *fs, int32_t flags,
                               const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    tr_lin_info li;
    if (!st || !fs || (flags & ~TR_LINUPD_INVERSE) || tr_lin_moments_get_info(st, &li)) { seterr(error, TR_MATH_INVALID); return NULL; }
    int32_t dim = 0, design[TR_LIN_MAX_COLS];
    uint64_t n = 0;
    tr_bigq **A = NULL, **A0 = NULL, **b = NULL, *yty = NULL, *sy = NULL;
    int32_t rc = tr_lin_q_normal(st, fs, &dim, &A, &A0, &b, &yty, &sy, &n, design, budget, refusal);
    if (!rc) rc = model_need(dim, 0, budget, refusal);
    if (rc) {
        Q_free_all(A, A ? (uint64_t)dim * dim : 0); Q_free_all(A0, A0 ? (uint64_t)dim * dim : 0);
        Q_free_all(b, b ? (uint64_t)dim : 0); tr_bigq_free(yty); tr_bigq_free(sy);
        seterr(error, rc); return NULL;
    }
    lctx x = {budget, refusal, 0};
    tr_linupd_model *m = model_new(&x);
    if (!m) {
        Q_free_all(A, (uint64_t)dim * dim); Q_free_all(A0, (uint64_t)dim * dim); Q_free_all(b, (uint64_t)dim);
        tr_bigq_free(yty); tr_bigq_free(sy);
        seterr(error, TR_MATH_MEMORY); return NULL;
    }
    m->flags = flags;
    set_part(m, TR_LINUPD_PART_A, A, (uint64_t)dim * dim);
    set_part(m, TR_LINUPD_PART_A0, A0, (uint64_t)dim * dim);
    set_part(m, TR_LINUPD_PART_B, b, (uint64_t)dim);
    m->yty = yty; m->sy = sy;
    m->info.n = dim; m->info.regression = 1; m->info.listwise = li.policy == TR_LIN_LISTWISE;
    m->info.intercept = design[0] == 0; m->info.ridged = fs->ridge_num != 0;
    m->info.rows = n; m->info.group = fs->group;
    set_design(&x, m, design, dim);
    m->k = li.k; m->policy = li.policy; m->nfeatures = li.nfeatures; m->target = li.target;
    for (int32_t i = 0; i < TR_LIN_MAX_COLS; i++) { m->col[i] = li.column[i]; m->scale[i] = li.scale[i]; }
    m->ridge_num = fs->ridge_num; m->ridge_den = fs->ridge_den;
    solve_core(&x, m, flags & TR_LINUPD_INVERSE);
    m->info.method = TR_LINUPD_BUILT;
    model_stats(&x, m);
    return model_done(&x, m, error);
}

tr_linupd_model *tr_linupd_system(int32_t n, const tr_lin_qbuf *A, const tr_lin_qbuf *b, int32_t flags,
                                  const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    if (n > TR_LIN_MAX_UNKNOWNS) { seterr(error, TR_MATH_UNSUPPORTED); return NULL; }
    if (n < 1 || !A || !b || (flags & ~TR_LINUPD_INVERSE) || A->count != (uint64_t)n * n || b->count != (uint64_t)n) {
        seterr(error, TR_MATH_INVALID); return NULL;
    }
    int32_t rc = model_need(n, 0, budget, refusal);
    if (rc) { seterr(error, rc); return NULL; }
    lctx x = {budget, refusal, 0};
    tr_bigq **qa = Q_arr(&x, (uint64_t)n * n), **qb = Q_arr(&x, (uint64_t)n);
    if (LIVE(&x)) { rc = tr_lin_q_import(A, (uint64_t)n * n, qa, budget, refusal); if (!rc) rc = tr_lin_q_import(b, (uint64_t)n, qb, budget, refusal); if (rc) lfail(&x, rc); }
    tr_linupd_model *m = model_new(&x);
    if (!m) { Q_free_all(qa, qa ? (uint64_t)n * n : 0); Q_free_all(qb, qb ? (uint64_t)n : 0); seterr(error, x.rc); return NULL; }
    m->flags = flags;
    set_part(m, TR_LINUPD_PART_A, qa, (uint64_t)n * n);
    set_part(m, TR_LINUPD_PART_B, qb, (uint64_t)n);
    m->info.n = n;
    solve_core(&x, m, flags & TR_LINUPD_INVERSE);
    m->info.method = TR_LINUPD_BUILT;
    return model_done(&x, m, error);
}

/* ================================================================ low-rank updates */
/* the model of A' = A + U V^T (A0 too when gram), b' = b + db (db may be NULL), with the regression extras yty, sy
 * (taken) and rows: Woodbury from the kept inverse when det(K) != 0, else the declared contract */
static tr_linupd_model *update_core(lctx *x, const tr_linupd_model *base, int32_t r, tr_bigq *const *U,
                                    tr_bigq *const *V, tr_bigq *const *db, int gram, int32_t fallback, tr_bigq *yty,
                                    tr_bigq *sy, uint64_t rows) {
    const int32_t n = base->info.n;
    const uint64_t nn = (uint64_t)n * n;
    tr_linupd_model *m = model_new(x);
    if (!m) { tr_bigq_free(yty); tr_bigq_free(sy); return NULL; }
    model_identity(x, m, base);
    m->yty = yty; m->sy = sy;
    m->info.rows = rows; m->info.update_rank = r;
    tr_bigq **A = Q_arr(x, nn), **A0 = base->part[TR_LINUPD_PART_A0] ? Q_arr(x, nn) : NULL, **b = Q_arr(x, (uint64_t)n);
    tr_bigq *const *bA = base->part[TR_LINUPD_PART_A], *const *bA0 = base->part[TR_LINUPD_PART_A0];
    for (int32_t i = 0; A && i < n && LIVE(x); i++)
        for (int32_t j = 0; j < n && LIVE(x); j++) {
            uint64_t at = (uint64_t)i * n + j;
            tr_bigq *s = Q_copy(x, bA[at]);
            for (int32_t t = 0; t < r && s; t++) Q_fma(x, &s, U[(uint64_t)i * r + t], V[(uint64_t)j * r + t]);
            A[at] = s;
            if (A0) {
                tr_bigq *s0 = Q_copy(x, bA0[at]);
                for (int32_t t = 0; gram && t < r && s0; t++) Q_fma(x, &s0, U[(uint64_t)i * r + t], V[(uint64_t)j * r + t]);
                A0[at] = s0;
            }
        }
    for (int32_t i = 0; b && i < n && LIVE(x); i++) {
        b[i] = Q_copy(x, base->part[TR_LINUPD_PART_B][i]);
        if (db && b[i]) Q_acc(x, &b[i], db[i]);
    }
    set_part(m, TR_LINUPD_PART_A, A, A ? nn : 0);
    set_part(m, TR_LINUPD_PART_A0, A0, A0 ? nn : 0);
    set_part(m, TR_LINUPD_PART_B, b, b ? (uint64_t)n : 0);
    if (x->rc) return m;
    if (base->info.inverse) {
        tr_bigq *const *Ai = base->part[TR_LINUPD_PART_INVERSE];
        const uint64_t nr = (uint64_t)n * r;
        tr_bigq **W = Q_arr(x, nr), **T = Q_arr(x, nr), **K = Q_arr(x, (uint64_t)r * r);
        for (int32_t i = 0; W && i < n && LIVE(x); i++)                   /* W = A^-1 U */
            for (int32_t t = 0; t < r && LIVE(x); t++) {
                tr_bigq *s = Q_int(x, 0);
                for (int32_t j = 0; j < n && s; j++) Q_fma(x, &s, Ai[(uint64_t)i * n + j], U[(uint64_t)j * r + t]);
                W[(uint64_t)i * r + t] = s;
            }
        for (int32_t t = 0; T && t < r && LIVE(x); t++)                   /* T = V^T A^-1 */
            for (int32_t j = 0; j < n && LIVE(x); j++) {
                tr_bigq *s = Q_int(x, 0);
                for (int32_t i = 0; i < n && s; i++) Q_fma(x, &s, V[(uint64_t)i * r + t], Ai[(uint64_t)i * n + j]);
                T[(uint64_t)t * n + j] = s;
            }
        for (int32_t s1 = 0; K && s1 < r && LIVE(x); s1++)                /* K = I + V^T W */
            for (int32_t t = 0; t < r && LIVE(x); t++) {
                tr_bigq *s = Q_int(x, s1 == t);
                for (int32_t i = 0; i < n && s; i++) Q_fma(x, &s, V[(uint64_t)i * r + s1], W[(uint64_t)i * r + t]);
                K[(uint64_t)s1 * r + t] = s;
            }
        tr_bigq **Y = NULL, *detK = NULL;
        int nonsingular = LIVE(x) ? rat_solve(x, r, (const tr_bigq *const *)K, (const tr_bigq *const *)T, n, NULL, &Y,
                                              &detK) : 0;
        /* the denominator det(K): for a singular K, 0 exactly (rat_solve found rank < r) */
        tr_bigq **dn = Q_arr(x, 1);
        if (dn) { dn[0] = nonsingular ? detK : Q_int(x, 0); set_part(m, TR_LINUPD_PART_DENOMINATOR, dn, 1); }
        else tr_bigq_free(detK);
        m->info.denominator_sign = nonsingular && dn ? tr_bigq_sign(dn[0]) : 0;
        if (nonsingular && LIVE(x)) {
            tr_bigq **Ainv = Q_arr(x, nn), **xs = Q_arr(x, (uint64_t)n);
            for (int32_t i = 0; Ainv && i < n && LIVE(x); i++)            /* A'^-1 = A^-1 - W (K^-1 T) */
                for (int32_t j = 0; j < n && LIVE(x); j++) {
                    tr_bigq *s = Q_int(x, 0);
                    for (int32_t t = 0; t < r && s; t++) Q_fma(x, &s, W[(uint64_t)i * r + t], Y[(uint64_t)t * n + j]);
                    Ainv[(uint64_t)i * n + j] = s ? Q_sub(x, Ai[(uint64_t)i * n + j], s) : NULL;
                    tr_bigq_free(s);
                }
            for (int32_t i = 0; xs && i < n && LIVE(x); i++) {           /* x' = A'^-1 b' */
                tr_bigq *s = Q_int(x, 0);
                for (int32_t j = 0; j < n && s; j++) Q_fma(x, &s, Ainv[(uint64_t)i * n + j], b[j]);
                xs[i] = s;
            }
            for (int32_t i = 0; i < n && LIVE(x); i++) {                  /* verified: A' x' = b' exactly */
                tr_bigq *s = Q_int(x, 0);
                for (int32_t j = 0; j < n && s; j++) Q_fma(x, &s, A[(uint64_t)i * n + j], xs[j]);
                if (LIVE(x) && tr_bigq_cmp(s, b[i]) != 0) lfail(x, TR_MATH_INVALID);
                tr_bigq_free(s);
            }
            tr_bigq **dt = Q_arr(x, 1);
            if (dt && base->part[TR_LINUPD_PART_DET]) dt[0] = Q_mul(x, base->part[TR_LINUPD_PART_DET][0], m->part[TR_LINUPD_PART_DENOMINATOR][0]);
            set_part(m, TR_LINUPD_PART_INVERSE, Ainv, Ainv ? nn : 0);
            set_part(m, TR_LINUPD_PART_X, xs, xs ? (uint64_t)n : 0);
            if (dt && dt[0]) set_part(m, TR_LINUPD_PART_DET, dt, 1); else Q_free_all(dt, dt ? 1 : 0);
            m->info.kind = TR_LIN_UNIQUE; m->info.rank = n;
            m->info.pivots = n >= 64 ? ~(uint64_t)0 : (((uint64_t)1 << n) - 1);
            m->info.inverse = LIVE(x); m->info.method = TR_LINUPD_WOODBURY;
        }
        Q_free_all(W, W ? nr : 0); Q_free_all(T, T ? nr : 0); Q_free_all(K, K ? (uint64_t)r * r : 0);
        Q_free_all(Y, Y ? nr : 0);
        if (nonsingular || x->rc) { model_stats(x, m); return m; }
    }
    /* det(K) = 0 (A' singular) or no inverse kept: the declared contract */
    if (fallback == TR_LINUPD_FAIL) { lfail(x, base->info.inverse || base->info.kind != TR_LIN_UNIQUE ? TR_MATH_SINGULAR : TR_MATH_DOMAIN_ERROR); return m; }
    solve_core(x, m, m->flags & TR_LINUPD_INVERSE);
    m->info.method = TR_LINUPD_REBUILT; m->info.fallback = 1;
    model_stats(x, m);
    return m;
}

tr_linupd_model *tr_linupd_update(const tr_linupd_model *base, int32_t r, const tr_lin_qbuf *U, const tr_lin_qbuf *V,
                                  const tr_lin_qbuf *db, int32_t fallback, const tr_math_budget *budget,
                                  tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    if (!model_ok(base) || (fallback != TR_LINUPD_FAIL && fallback != TR_LINUPD_REBUILD)) { seterr(error, TR_MATH_INVALID); return NULL; }
    if (r > TR_LINUPD_MAX_RANK) { seterr(error, TR_MATH_UNSUPPORTED); return NULL; }
    const int32_t n = base->info.n;
    const uint64_t nr = (uint64_t)n * (uint64_t)(r > 0 ? r : 0);
    if (r < 1 || !U || !V || U->count != nr || V->count != nr || (db && db->count != (uint64_t)n)) { seterr(error, TR_MATH_INVALID); return NULL; }
    int32_t rc = model_need(n, (uint64_t)r, budget, refusal);
    if (rc) { seterr(error, rc); return NULL; }
    lctx x = {budget, refusal, 0};
    tr_bigq **qu = Q_arr(&x, nr), **qv = Q_arr(&x, nr), **qd = db ? Q_arr(&x, (uint64_t)n) : NULL;
    if (LIVE(&x)) {
        rc = tr_lin_q_import(U, nr, qu, budget, refusal);
        if (!rc) rc = tr_lin_q_import(V, nr, qv, budget, refusal);
        if (!rc && db) rc = tr_lin_q_import(db, (uint64_t)n, qd, budget, refusal);
        if (rc) lfail(&x, rc);
    }
    tr_bigq *yty = base->yty ? Q_copy(&x, base->yty) : NULL, *sy = base->sy ? Q_copy(&x, base->sy) : NULL;
    tr_linupd_model *m = NULL;                                        /* update_core takes yty and sy */
    if (LIVE(&x)) m = update_core(&x, base, r, qu, qv, qd, 1, fallback, yty, sy, base->info.rows);
    else { tr_bigq_free(yty); tr_bigq_free(sy); }
    Q_free_all(qu, qu ? nr : 0); Q_free_all(qv, qv ? nr : 0); Q_free_all(qd, qd ? (uint64_t)n : 0);
    if (!m) { seterr(error, x.rc ? x.rc : TR_MATH_MEMORY); return NULL; }
    return model_done(&x, m, error);
}

tr_linupd_model *tr_linupd_rows(const tr_linupd_model *base, int32_t sign, int32_t r, const tr_lin_qbuf *Z,
                                const tr_lin_qbuf *y, int32_t fallback, const tr_math_budget *budget,
                                tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    if (!model_ok(base) || (sign != 1 && sign != -1) || (fallback != TR_LINUPD_FAIL && fallback != TR_LINUPD_REBUILD)) {
        seterr(error, TR_MATH_INVALID); return NULL;
    }
    if (!base->info.regression || !base->info.listwise) { seterr(error, TR_MATH_DOMAIN_ERROR); return NULL; }
    if (r > TR_LINUPD_MAX_RANK) { seterr(error, TR_MATH_UNSUPPORTED); return NULL; }
    const int32_t n = base->info.n;
    int32_t f = 0;
    for (int32_t u = 0; u < n; u++) f += base->info.design[u] != 0;
    if (r < 1 || !Z || !y || Z->count != (uint64_t)r * (uint64_t)f || y->count != (uint64_t)r ||
        (sign < 0 && (uint64_t)r > base->info.rows)) { seterr(error, TR_MATH_INVALID); return NULL; }
    int32_t rc = model_need(n, (uint64_t)r, budget, refusal);
    if (rc) { seterr(error, rc); return NULL; }
    lctx x = {budget, refusal, 0};
    const uint64_t nr = (uint64_t)n * r, zf = (uint64_t)r * f;
    tr_bigq **qz = Q_arr(&x, zf ? zf : 1), **qy = Q_arr(&x, (uint64_t)r);
    if (LIVE(&x)) { rc = tr_lin_q_import(Z, zf, qz, budget, refusal); if (!rc) rc = tr_lin_q_import(y, (uint64_t)r, qy, budget, refusal); if (rc) lfail(&x, rc); }
    tr_bigq **U = Q_arr(&x, nr), **V = Q_arr(&x, nr), **db = Q_arr(&x, (uint64_t)n);
    tr_bigq *yty = Q_copy(&x, base->yty), *sy = Q_copy(&x, base->sy);
    for (int32_t t = 0; U && V && t < r && LIVE(&x); t++) {             /* z = the design row of row t */
        int32_t fi = 0;
        for (int32_t u = 0; u < n && LIVE(&x); u++) {
            tr_bigq *z = base->info.design[u] == 0 ? Q_int(&x, 1) : Q_copy(&x, qz[(uint64_t)t * f + fi++]);
            V[(uint64_t)u * r + t] = z;
            U[(uint64_t)u * r + t] = sign > 0 ? Q_copy(&x, z) : Q_neg(&x, z);
        }
        tr_bigq *y2 = Q_mul(&x, qy[t], qy[t]);
        if (sign > 0) { Q_acc(&x, &yty, y2); Q_acc(&x, &sy, qy[t]); }
        else { tr_bigq *a = Q_sub(&x, yty, y2), *c = Q_sub(&x, sy, qy[t]); if (a && c) { tr_bigq_free(yty); tr_bigq_free(sy); yty = a; sy = c; } else { tr_bigq_free(a); tr_bigq_free(c); } }
        tr_bigq_free(y2);
    }
    for (int32_t u = 0; db && u < n && LIVE(&x); u++) {                 /* db = sign sum_t z_t y_t */
        tr_bigq *s = Q_int(&x, 0);
        for (int32_t t = 0; t < r && s; t++) Q_fma(&x, &s, U[(uint64_t)u * r + t], qy[t]);
        db[u] = s;
    }
    uint64_t rows = sign > 0 ? base->info.rows + (uint64_t)r : base->info.rows - (uint64_t)r;
    tr_linupd_model *m = NULL;                                        /* update_core takes yty and sy */
    if (LIVE(&x)) m = update_core(&x, base, r, U, V, db, 1, fallback, yty, sy, rows);
    else { tr_bigq_free(yty); tr_bigq_free(sy); }
    Q_free_all(qz, zf ? zf : 1); Q_free_all(qy, (uint64_t)r); Q_free_all(U, U ? nr : 0); Q_free_all(V, V ? nr : 0);
    Q_free_all(db, db ? (uint64_t)n : 0);
    if (!m) { seterr(error, x.rc ? x.rc : TR_MATH_MEMORY); return NULL; }
    return model_done(&x, m, error);
}

static tr_linupd_model *model_copy(lctx *x, const tr_linupd_model *src) {
    tr_linupd_model *m = model_new(x);
    if (!m) return NULL;
    model_identity(x, m, src);
    m->info = src->info;
    for (int p = 0; p < TR_LINUPD_PARTS && LIVE(x); p++)
        if (src->part[p]) set_part(m, p, Q_copy_all(x, src->part[p], src->npart[p]), src->npart[p]);
    m->yty = src->yty ? Q_copy(x, src->yty) : NULL;
    m->sy = src->sy ? Q_copy(x, src->sy) : NULL;
    return m;
}

tr_linupd_model *tr_linupd_ridge(const tr_linupd_model *base, int64_t num, int64_t den, int32_t method, int32_t fallback,
                                 const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    if (!model_ok(base) || num < 0 || den < 1 || (method != TR_LINUPD_METHOD_WOODBURY && method != TR_LINUPD_METHOD_REBUILD) ||
        (fallback != TR_LINUPD_FAIL && fallback != TR_LINUPD_REBUILD)) { seterr(error, TR_MATH_INVALID); return NULL; }
    if (!base->info.regression) { seterr(error, TR_MATH_DOMAIN_ERROR); return NULL; }
    const int32_t n = base->info.n;
    int32_t p = 0;
    for (int32_t u = 0; u < n; u++) p += base->pen[u];
    int32_t rc = model_need(n, (uint64_t)(p ? p : 1), budget, refusal);
    if (rc) { seterr(error, rc); return NULL; }
    lctx x = {budget, refusal, 0};
    tr_bigq *lam = Q_q64(&x, num, den), *old = Q_q64(&x, base->ridge_num, base->ridge_den), *delta = Q_sub(&x, lam, old);
    tr_linupd_model *m = NULL;
    if (LIVE(&x) && (!p || !tr_bigq_sign(delta))) m = model_copy(&x, base);          /* nothing changes */
    else if (LIVE(&x) && method == TR_LINUPD_METHOD_WOODBURY) {
        const uint64_t np = (uint64_t)n * p;
        tr_bigq **U = Q_arr(&x, np), **V = Q_arr(&x, np);
        for (int32_t u = 0; U && V && u < n && LIVE(&x); u++)        /* U = delta E, V = E: column tt of E is e_j */
            for (int32_t j = 0, tt = 0; j < n && LIVE(&x); j++) {
                if (!base->pen[j]) continue;
                int on = u == j;
                V[(uint64_t)u * p + tt] = Q_int(&x, on);
                U[(uint64_t)u * p + tt] = on ? Q_copy(&x, delta) : Q_int(&x, 0);
                tt++;
            }
        tr_bigq *yty = Q_copy(&x, base->yty), *sy = Q_copy(&x, base->sy);
        if (LIVE(&x)) m = update_core(&x, base, p, U, V, NULL, 0, fallback, yty, sy, base->info.rows);   /* takes them */
        else { tr_bigq_free(yty); tr_bigq_free(sy); }
        Q_free_all(U, U ? np : 0); Q_free_all(V, V ? np : 0);
    } else if (LIVE(&x)) {                                             /* rebuilt from A0 + lambda' D */
        m = model_new(&x);
        if (m) {
            model_identity(&x, m, base);
            const uint64_t nn = (uint64_t)n * n;
            tr_bigq **A = Q_arr(&x, nn);
            for (int32_t u = 0; A && u < n && LIVE(&x); u++)
                for (int32_t w = 0; w < n && LIVE(&x); w++) {
                    const tr_bigq *a0 = base->part[TR_LINUPD_PART_A0][(uint64_t)u * n + w];
                    A[(uint64_t)u * n + w] = (u == w && base->pen[u]) ? Q_add(&x, a0, lam) : Q_copy(&x, a0);
                }
            set_part(m, TR_LINUPD_PART_A, A, A ? nn : 0);
            set_part(m, TR_LINUPD_PART_A0, Q_copy_all(&x, base->part[TR_LINUPD_PART_A0], nn), nn);
            set_part(m, TR_LINUPD_PART_B, Q_copy_all(&x, base->part[TR_LINUPD_PART_B], (uint64_t)n), (uint64_t)n);
            m->yty = Q_copy(&x, base->yty); m->sy = Q_copy(&x, base->sy);
            solve_core(&x, m, m->flags & TR_LINUPD_INVERSE);
            m->info.method = TR_LINUPD_REBUILT;
        }
    }
    if (m && LIVE(&x)) {
        m->ridge_num = num; m->ridge_den = den; m->info.ridged = num != 0;
        model_stats(&x, m);                                           /* sigma2 depends on the ridge */
    }
    tr_bigq_free(lam); tr_bigq_free(old); tr_bigq_free(delta);
    if (!m) { seterr(error, x.rc ? x.rc : TR_MATH_MEMORY); return NULL; }
    return model_done(&x, m, error);
}

/* the compatibility of a moment state with a regression model's origin */
static int compatible(const tr_linupd_model *m, const tr_lin_info *li) {
    if (li->k != m->k || li->policy != m->policy || li->nfeatures != m->nfeatures || li->target != m->target) return 0;
    for (int32_t i = 0; i < m->k; i++) if (li->column[i] != m->col[i] || li->scale[i] != m->scale[i]) return 0;
    return 1;
}
static tr_bigq *slot_q(lctx *x, const tr_lin_states *s, uint64_t g, int32_t k, int32_t a, int32_t b, const int32_t *scale) {
    int32_t slot = a <= b ? tr_lin_gram_slot(k, a, b) : tr_lin_gram_slot(k, b, a);
    tr_bigint *z = Z_twos(x, s->sum + (g * (uint64_t)s->nslots + (uint64_t)slot) * TR_LIN_WORDS, TR_LIN_WORDS);
    tr_bigq *q = z ? Q_scale10(x, z, scale[a] + scale[b]) : NULL;
    tr_bigint_free(z);
    return q;
}

tr_linupd_model *tr_linupd_subtract(const tr_linupd_model *base, const tr_lin_moments *st, uint64_t group, int32_t sign,
                                    int32_t flags, const tr_math_budget *budget, tr_math_refusal *refusal,
                                    int32_t *error) {
    tr_budget_none(refusal);
    tr_lin_info li;
    tr_lin_states ss;
    if (!model_ok(base) || !st || (sign != 1 && sign != -1) || (flags & ~TR_LINUPD_INVERSE) ||
        tr_lin_moments_get_info(st, &li) || tr_lin_moments_get_states(st, &ss)) { seterr(error, TR_MATH_INVALID); return NULL; }
    if (!base->info.regression || !compatible(base, &li)) { seterr(error, TR_MATH_DOMAIN_ERROR); return NULL; }
    if (group >= li.ngroups) { seterr(error, TR_MATH_INVALID); return NULL; }
    const int32_t n = base->info.n, k = base->k, y = k - 1;
    int32_t rc = model_need(n, 0, budget, refusal);
    if (rc) { seterr(error, rc); return NULL; }
    uint64_t cnt = ss.count[group * (uint64_t)ss.nslots + (uint64_t)(base->policy == TR_LIN_LISTWISE ? 0 : tr_lin_gram_slot(k, y, y))];
    if (sign < 0 && cnt > base->info.rows) { seterr(error, TR_MATH_INVALID); return NULL; }
    lctx x = {budget, refusal, 0};
    tr_linupd_model *m = model_new(&x);
    if (!m) { seterr(error, TR_MATH_MEMORY); return NULL; }
    model_identity(&x, m, base);
    m->flags = flags;
    const uint64_t nn = (uint64_t)n * n;
    tr_bigq **A = Q_arr(&x, nn), **A0 = Q_arr(&x, nn), **b = Q_arr(&x, (uint64_t)n);
    const int32_t *d = base->info.design;
    for (int32_t u = 0; A && A0 && b && u < n && LIVE(&x); u++) {
        for (int32_t w = 0; w < n && LIVE(&x); w++) {
            tr_bigq *gq = slot_q(&x, &ss, group, k, d[u], d[w], base->scale);
            tr_bigq *t = gq && sign < 0 ? Q_neg(&x, gq) : gq;
            uint64_t at = (uint64_t)u * n + w;
            A[at] = t ? Q_add(&x, base->part[TR_LINUPD_PART_A][at], t) : NULL;
            A0[at] = t ? Q_add(&x, base->part[TR_LINUPD_PART_A0][at], t) : NULL;
            if (t != gq) tr_bigq_free(t);
            tr_bigq_free(gq);
        }
        tr_bigq *gq = slot_q(&x, &ss, group, k, d[u], y, base->scale);
        tr_bigq *t = gq && sign < 0 ? Q_neg(&x, gq) : gq;
        b[u] = t ? Q_add(&x, base->part[TR_LINUPD_PART_B][u], t) : NULL;
        if (t != gq) tr_bigq_free(t);
        tr_bigq_free(gq);
    }
    set_part(m, TR_LINUPD_PART_A, A, A ? nn : 0);
    set_part(m, TR_LINUPD_PART_A0, A0, A0 ? nn : 0);
    set_part(m, TR_LINUPD_PART_B, b, b ? (uint64_t)n : 0);
    tr_bigq *gyy = slot_q(&x, &ss, group, k, y, y, base->scale), *gsy = slot_q(&x, &ss, group, k, 0, y, base->scale);
    if (LIVE(&x)) {
        m->yty = sign > 0 ? Q_add(&x, base->yty, gyy) : Q_sub(&x, base->yty, gyy);
        m->sy = sign > 0 ? Q_add(&x, base->sy, gsy) : Q_sub(&x, base->sy, gsy);
    }
    tr_bigq_free(gyy); tr_bigq_free(gsy);
    m->info.rows = sign > 0 ? base->info.rows + cnt : base->info.rows - cnt;
    solve_core(&x, m, flags & TR_LINUPD_INVERSE);
    m->info.method = TR_LINUPD_SUBTRACTED;
    model_stats(&x, m);
    return model_done(&x, m, error);
}

/* ================================================================ reading */
int32_t tr_linupd_get_info(const tr_linupd_model *m, tr_linupd_info *out) {
    if (!model_ok(m) || !out) return TR_MATH_INVALID;
    *out = m->info;
    return TR_MATH_OK;
}
uint64_t tr_linupd_count(const tr_linupd_model *m, int32_t part) {
    if (!model_ok(m) || part < 0 || part >= TR_LINUPD_PARTS) return 0;
    return m->part[part] ? m->npart[part] : 0;
}
uint64_t tr_linupd_limbs(const tr_linupd_model *m, int32_t part) {
    if (!model_ok(m) || part < 0 || part >= TR_LINUPD_PARTS || !m->part[part]) return 0;
    return tr_lin_q_limbs((const tr_bigq *const *)m->part[part], m->npart[part]);
}
int32_t tr_linupd_export(const tr_linupd_model *m, int32_t part, tr_lin_qbuf *out) {
    if (!model_ok(m) || part < 0 || part >= TR_LINUPD_PARTS || !m->part[part]) return TR_MATH_INVALID;
    return tr_lin_q_export((const tr_bigq *const *)m->part[part], m->npart[part], out);
}
static int part_eq(const tr_linupd_model *a, const tr_linupd_model *b, int32_t p) {
    if ((a->part[p] == NULL) != (b->part[p] == NULL) || a->npart[p] != b->npart[p]) return 0;
    for (uint64_t i = 0; i < a->npart[p]; i++) if (!Q_eq(a->part[p][i], b->part[p][i])) return 0;
    return 1;
}
int32_t tr_linupd_equal(const tr_linupd_model *a, const tr_linupd_model *b) {
    if (!model_ok(a) || !model_ok(b)) return TR_MATH_INVALID;
    if (a->info.n != b->info.n || a->info.kind != b->info.kind || a->info.rank != b->info.rank ||
        a->info.pivots != b->info.pivots || a->info.regression != b->info.regression || a->info.rows != b->info.rows)
        return 0;
    static const int32_t parts[] = {TR_LINUPD_PART_X, TR_LINUPD_PART_NULL, TR_LINUPD_PART_CERT, TR_LINUPD_PART_A,
                                    TR_LINUPD_PART_A0, TR_LINUPD_PART_B, TR_LINUPD_PART_STATS, TR_LINUPD_PART_DESIGN,
                                    TR_LINUPD_PART_INVERSE};
    for (unsigned i = 0; i < sizeof parts / sizeof parts[0]; i++) if (!part_eq(a, b, parts[i])) return 0;
    if (a->part[TR_LINUPD_PART_DET] && b->part[TR_LINUPD_PART_DET] && !part_eq(a, b, TR_LINUPD_PART_DET)) return 0;
    return 1;
}

/* ================================================================ leave-one-group-out */
struct tr_linupd_logo_s {
    uint32_t magic;
    tr_linupd_logo_view v;
    uint64_t *sum, *total, *groups, *count, *tcount;   /* owned copies: G S 3, S 3, nsys, G S, S */
    int32_t rowslot;                                     /* the slot whose count is the rows: 0 or (y, y) */
    tr_linupd_model proto;                               /* the identity of every model (no parts) */
    tr_bigint **ints;                                    /* nsys x n x (n + 1): the integer systems */
    uint64_t *need, max_need;
};
static int logo_ok(const tr_linupd_logo *L) { return L && L->magic == LOGO_MAGIC; }
void tr_linupd_logo_free(tr_linupd_logo *L) {
    if (!logo_ok(L)) return;
    L->magic = 0;
    uint64_t c = (uint64_t)L->v.n * (uint64_t)(L->v.n + 1);
    Z_free_all(L->ints, L->ints ? L->v.nsys * c : 0);
    free(L->sum); free(L->total); free(L->groups); free(L->count); free(L->tcount); free(L->need);
    free(L);
}
static void w3_add(uint64_t *r, const uint64_t *a) {
    unsigned __int128 c = 0;
    for (int i = 0; i < 3; i++) { c += (unsigned __int128)r[i] + a[i]; r[i] = (uint64_t)c; c >>= 64; }
}
static void w3_sub(uint64_t *r, const uint64_t *a, const uint64_t *b) {
    uint64_t c = 1;
    for (int i = 0; i < 3; i++) { unsigned __int128 t = (unsigned __int128)a[i] + (uint64_t)~b[i] + c; r[i] = (uint64_t)t; c = (uint64_t)(t >> 64); }
}
static uint32_t ceil_log2(uint64_t m) { uint32_t k = 0; while (((uint64_t)1 << k) < m && k < 63) k++; return k; }
/* the complement sum (total - group g) of slot (a, b) */
static void logo_slot(const tr_linupd_logo *L, uint64_t g, int32_t a, int32_t b, uint64_t *w) {
    const int32_t k = L->v.k;
    int32_t slot = a <= b ? tr_lin_gram_slot(k, a, b) : tr_lin_gram_slot(k, b, a);
    w3_sub(w, L->total + 3 * (uint64_t)slot, L->sum + 3 * (g * (uint64_t)L->v.nslots + (uint64_t)slot));
}
static tr_bigq *logo_q(lctx *x, const tr_linupd_logo *L, uint64_t g, int32_t a, int32_t b) {   /* value units */
    uint64_t w[3];
    logo_slot(L, g, a, b, w);
    tr_bigint *z = Z_twos(x, w, 3);
    tr_bigq *q = z ? Q_scale10(x, z, L->v.scale[a] + L->v.scale[b]) : NULL;
    tr_bigint_free(z);
    return q;
}
/* the normal equations of system s (value units) into m: A, A0, b, y^T y, sum y, rows, design, identity */
static void logo_system(lctx *x, const tr_linupd_logo *L, uint64_t s, tr_linupd_model *m) {
    const int32_t n = L->v.n, y = L->v.ycol;
    const uint64_t g = L->groups[s], nn = (uint64_t)n * n;
    model_identity(x, m, &L->proto);
    tr_bigq **A = Q_arr(x, nn), **A0 = Q_arr(x, nn), **b = Q_arr(x, (uint64_t)n);
    tr_bigq *lam = L->v.ridge_num ? Q_q64(x, L->v.ridge_num, L->v.ridge_den) : NULL;
    for (int32_t u = 0; A && A0 && b && u < n && LIVE(x); u++) {
        for (int32_t w = 0; w < n && LIVE(x); w++) {
            uint64_t at = (uint64_t)u * n + w;
            A0[at] = logo_q(x, L, g, L->v.design[u], L->v.design[w]);
            A[at] = (u == w && lam && L->v.pen[u]) ? Q_add(x, A0[at], lam) : Q_copy(x, A0[at]);
        }
        b[u] = logo_q(x, L, g, L->v.design[u], y);
    }
    tr_bigq_free(lam);
    set_part(m, TR_LINUPD_PART_A, A, A ? nn : 0);
    set_part(m, TR_LINUPD_PART_A0, A0, A0 ? nn : 0);
    set_part(m, TR_LINUPD_PART_B, b, b ? (uint64_t)n : 0);
    set_design(x, m, L->v.design, n);
    m->yty = logo_q(x, L, g, y, y);
    m->sy = logo_q(x, L, g, 0, y);
    m->info.rows = L->tcount[L->rowslot] - L->count[g * (uint64_t)L->v.nslots + (uint64_t)L->rowslot];
    m->info.group = g;
}

tr_linupd_logo *tr_linupd_logo_plan(const tr_lin_moments *st, const tr_lin_fit_spec *fs, uint64_t nsel,
                                    const uint64_t *groups, const tr_math_budget *budget, tr_math_refusal *refusal,
                                    int32_t *error) {
    tr_budget_none(refusal);
    tr_lin_info li;
    tr_lin_states ss;
    if (!st || !fs || tr_lin_moments_get_info(st, &li) || tr_lin_moments_get_states(st, &ss)) { seterr(error, TR_MATH_INVALID); return NULL; }
    const int32_t d = li.nfeatures, k = li.k;
    if (li.target < 0 || fs->intercept < -1 || fs->intercept > 1 || (fs->masked != 0 && fs->masked != 1) || fs->reserved ||
        fs->ridge_den <= 0 || fs->ridge_num < 0 || (fs->masked && (fs->features >> d)) || (groups && !nsel)) {
        seterr(error, TR_MATH_INVALID); return NULL;
    }
    const uint64_t G = li.ngroups, S = (uint64_t)li.nslots, nsys = groups ? nsel : G;
    for (uint64_t i = 0; groups && i < nsel; i++) if (groups[i] >= G) { seterr(error, TR_MATH_INVALID); return NULL; }
    int32_t design[TR_LIN_MAX_COLS], n = 0;
    int intercept = fs->intercept < 0 ? li.intercept : fs->intercept;
    if (intercept) design[n++] = 0;
    for (int32_t i = 0; i < d; i++) if (!fs->masked || ((fs->features >> i) & 1)) design[n++] = i + 1;
    if (!n) { seterr(error, TR_MATH_INVALID); return NULL; }
    const uint64_t c = (uint64_t)n * (uint64_t)(n + 1);
    tr_math_need need;
    memset(&need, 0, sizeof need);
    need.need[TR_BUDGET_VARIABLES] = (uint64_t)n;
    need.need[TR_BUDGET_STATES] = tr_budget_mul(G, S);
    need.need[TR_BUDGET_TENSOR] = tr_budget_mul(nsys, c);
    need.need[TR_BUDGET_OUTPUT] = nsys;
    need.need[TR_BUDGET_BYTES] = tr_budget_add(tr_budget_add(tr_budget_mul(tr_budget_mul(G, S), 32), tr_budget_mul(S, 32)),
                                               tr_budget_mul(nsys, tr_budget_add(16, tr_budget_mul(c, 8))));
    tr_math_budget dflt;
    int32_t rc = tr_budget_check(budget_or(budget, &dflt), &need, refusal);
    if (rc) { seterr(error, rc); return NULL; }
    tr_linupd_logo *L = (tr_linupd_logo *)zalloc(sizeof *L);
    if (!L) { seterr(error, TR_MATH_MEMORY); return NULL; }
    L->magic = LOGO_MAGIC;
    L->sum = (uint64_t *)zalloc(G * S * 24); L->count = (uint64_t *)zalloc(G * S * 8);
    L->total = (uint64_t *)zalloc(S * 24); L->tcount = (uint64_t *)zalloc(S * 8);
    L->groups = (uint64_t *)zalloc(nsys * 8); L->need = (uint64_t *)zalloc(nsys * 8);
    lctx x = {budget, refusal, 0};
    if (!L->sum || !L->count || !L->total || !L->tcount || !L->groups || !L->need) lfail(&x, TR_MATH_MEMORY);
    if (LIVE(&x)) {
        memcpy(L->sum, ss.sum, (size_t)(G * S) * 24);
        memcpy(L->count, ss.count, (size_t)(G * S) * 8);
        for (uint64_t g = 0; g < G; g++)
            for (uint64_t j = 0; j < S; j++) { w3_add(L->total + 3 * j, L->sum + 3 * (g * S + j)); L->tcount[j] += L->count[g * S + j]; }
        for (uint64_t i = 0; i < nsys; i++) L->groups[i] = groups ? groups[i] : i;
    }
    tr_linupd_logo_view *v = &L->v;
    v->ngroups = G; v->nsys = nsys; v->k = k; v->nslots = (int32_t)S; v->n = n; v->ycol = k - 1;
    int32_t T = li.scale[k - 1];
    for (int32_t u = 0; u < n; u++) {
        v->design[u] = design[u]; v->pen[u] = design[u] != 0;
        if (li.scale[design[u]] > T) T = li.scale[design[u]];
    }
    for (int32_t i = 0; i < k; i++) v->scale[i] = li.scale[i];
    v->T = T; v->ridge_num = fs->ridge_num; v->ridge_den = fs->ridge_den;
    v->sum = L->sum; v->total = L->total; v->groups = L->groups;
    L->rowslot = li.policy == TR_LIN_LISTWISE ? 0 : tr_lin_gram_slot(k, k - 1, k - 1);
    /* the identity of the models */
    tr_linupd_model *P = &L->proto;
    P->magic = 0; P->flags = 0; P->info.n = n; P->info.regression = 1; P->info.listwise = li.policy == TR_LIN_LISTWISE;
    P->info.intercept = intercept; P->info.ridged = fs->ridge_num != 0; P->info.denominator_sign = 2;
    for (int32_t u = 0; u < n; u++) { P->info.design[u] = design[u]; P->pen[u] = design[u] != 0; }
    P->k = k; P->policy = li.policy; P->nfeatures = d; P->target = li.target;
    for (int32_t i = 0; i < TR_LIN_MAX_COLS; i++) { P->col[i] = li.column[i]; P->scale[i] = li.scale[i]; }
    P->ridge_num = fs->ridge_num; P->ridge_den = fs->ridge_den;
    /* the integer systems (verification and the Hadamard bounds) */
    L->ints = Z_arr(&x, nsys * c);
    tr_bigint *den = Z_i64(&x, fs->ridge_den), *num = Z_i64(&x, fs->ridge_num);
    tr_bigint *p10[2 * TR_SCOPE_MAX_SCALE + 1];
    memset(p10, 0, sizeof p10);
    for (int32_t e = 0; e <= 2 * TR_SCOPE_MAX_SCALE && LIVE(&x); e++) p10[e] = Z_pow10(&x, e);
    L->max_need = 1;
    for (uint64_t s = 0; L->ints && s < nsys && LIVE(&x); s++) {
        tr_bigint **M = L->ints + s * c;
        for (int32_t u = 0; u < n && LIVE(&x); u++)
            for (int32_t w = 0; w <= n && LIVE(&x); w++) {
                int32_t a = design[u], b = w < n ? design[w] : k - 1;
                uint64_t ww[3];
                logo_slot(L, L->groups[s], a, b, ww);
                tr_bigint *z = Z_twos(&x, ww, 3), *t1 = Z_mul(&x, z, den), *e = Z_mul(&x, t1, p10[T - v->scale[b]]);
                tr_bigint_free(z); tr_bigint_free(t1);
                if (w == u && v->pen[u] && fs->ridge_num && e) {
                    tr_bigint *r2 = Z_mul(&x, num, p10[v->scale[a] + T]), *e2 = Z_add(&x, e, r2);
                    tr_bigint_free(r2); tr_bigint_free(e); e = e2;
                }
                M[(uint64_t)u * (n + 1) + w] = e;
            }
        /* primes: the Hadamard bound of every r x r minor (LIN-002's) in 61-bit primes */
        uint64_t bits[TR_LIN_MAX_COLS + 1], half = (ceil_log2((uint64_t)n) + 1) / 2 + 1;
        for (int32_t w = 0; w <= n && LIVE(&x); w++) {
            uint64_t mb = 0;
            for (int32_t u = 0; u < n; u++) { uint64_t bl = tr_bigint_bit_length(M[(uint64_t)u * (n + 1) + w]); if (bl > mb) mb = bl; }
            bits[w] = mb ? mb + half : 0;
        }
        for (int32_t i = 1; i <= n; i++) for (int32_t j = i; j > 0 && bits[j - 1] < bits[j]; j--) { uint64_t t = bits[j]; bits[j] = bits[j - 1]; bits[j - 1] = t; }
        uint64_t hb = 0;
        for (int32_t j = 0; j < n; j++) hb += bits[j];
        L->need[s] = (hb + 2 + TR_LIN_PRIME_BITS - 1) / TR_LIN_PRIME_BITS;
        if (!L->need[s]) L->need[s] = 1;
        if (L->need[s] > L->max_need) L->max_need = L->need[s];
    }
    tr_bigint_free(den); tr_bigint_free(num);
    for (int32_t e = 0; e <= 2 * TR_SCOPE_MAX_SCALE; e++) tr_bigint_free(p10[e]);
    if (LIVE(&x) && L->max_need + TR_LIN_PRIME_MARGIN > TR_LINUPD_MAX_PRIMES) lfail(&x, TR_MATH_UNSUPPORTED);
    if (x.rc) { tr_linupd_logo_free(L); seterr(error, x.rc); return NULL; }
    seterr(error, TR_MATH_OK);
    return L;
}
int32_t tr_linupd_logo_get_view(const tr_linupd_logo *L, tr_linupd_logo_view *v) {
    if (!logo_ok(L) || !v) return TR_MATH_INVALID;
    *v = L->v;
    return TR_MATH_OK;
}
uint64_t tr_linupd_logo_primes_needed(const tr_linupd_logo *L) { return logo_ok(L) ? L->max_need + TR_LIN_PRIME_MARGIN : 0; }
uint64_t tr_linupd_logo_bytes(const tr_linupd_logo *L, uint64_t np) { return logo_ok(L) ? tr_linupd_logo_device_bytes(&L->v, np) : TR_BUDGET_UNKNOWN; }

static void free_models(tr_linupd_model **out, uint64_t n) { for (uint64_t i = 0; i < n; i++) { tr_linupd_free(out[i]); out[i] = NULL; } }
int32_t tr_linupd_logo_solve(const tr_linupd_logo *L, int32_t flags, tr_linupd_model **out, const tr_math_budget *budget,
                             tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!logo_ok(L) || !out || (flags & ~TR_LINUPD_INVERSE)) return TR_MATH_INVALID;
    for (uint64_t s = 0; s < L->v.nsys; s++) out[s] = NULL;
    int32_t rc = model_need(L->v.n, 0, budget, refusal);
    if (rc) return rc;
    lctx x = {budget, refusal, 0};
    for (uint64_t s = 0; s < L->v.nsys && LIVE(&x); s++) {
        tr_linupd_model *m = model_new(&x);
        if (!m) break;
        logo_system(&x, L, s, m);
        m->flags = flags;
        solve_core(&x, m, flags & TR_LINUPD_INVERSE);
        m->info.method = TR_LINUPD_SUBTRACTED;
        model_stats(&x, m);
        out[s] = m;
    }
    if (x.rc) free_models(out, L->v.nsys);
    return x.rc;
}

static int primes_ok(const tr_gfp *primes, uint64_t np, int prove) {
    if (!primes || !np || np > TR_LINUPD_MAX_PRIMES) return 0;
    for (uint64_t i = 0; i < np; i++) {
        if (!tr_gfp_consistent(&primes[i]) || primes[i].p < ((uint64_t)1 << 61) || (prove && !tr_gfp_is_prime(primes[i].p))) return 0;
        if (i && primes[i].p >= primes[i - 1].p) return 0;                 /* strictly descending: distinct */
    }
    return 1;
}
int32_t tr_linupd_logo_residues(const tr_linupd_logo_view *v, const tr_gfp *primes, uint64_t np, uint64_t *residues,
                                const tr_math_budget *budget, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!tr_linupd_logo_view_valid(v) || !residues || !primes_ok(primes, np, 0)) return TR_MATH_INVALID;
    tr_math_need need;
    memset(&need, 0, sizeof need);
    need.need[TR_BUDGET_TENSOR] = tr_budget_mul(tr_budget_mul(v->nsys, np), tr_linupd_block(v));
    tr_math_budget dflt;
    int32_t rc = tr_budget_check(budget_or(budget, &dflt), &need, refusal);
    if (rc) return rc;
    const uint64_t blk = tr_linupd_block(v);
    for (uint64_t s = 0; s < v->nsys; s++)
        for (uint64_t q = 0; q < np; q++) tr_linupd_logo_system(v, &primes[q], s, residues + (s * np + q) * blk);
    return TR_MATH_OK;
}

/* Garner's CRT with constants shared by the values of one system: x in (-M/2, M/2] */
typedef struct { uint64_t k, mn; const tr_gfp *f[TR_LINUPD_MAX_PRIMES]; uint64_t *inv, *M, *v, *limbs; } garner;
static void garner_free(garner *G) { if (G) { free(G->inv); free(G->M); free(G->v); free(G->limbs); free(G); } }
static garner *garner_new(const tr_gfp *const *f, uint64_t k) {
    garner *G = (garner *)zalloc(sizeof *G);
    if (!G) return NULL;
    G->k = k;
    G->inv = (uint64_t *)zalloc(k * 8); G->M = (uint64_t *)zalloc((k + 1) * 8);
    G->v = (uint64_t *)zalloc(k * 8); G->limbs = (uint64_t *)zalloc((k + 2) * 8);
    if (!G->inv || !G->M || !G->v || !G->limbs) { garner_free(G); return NULL; }
    for (uint64_t i = 0; i < k; i++) {
        G->f[i] = f[i];
        uint64_t prod = 1 % f[i]->p;
        for (uint64_t j = 0; j < i; j++) prod = tr_gfp_mul_hd(f[i], prod, f[j]->p % f[i]->p);
        G->inv[i] = i ? tr_gfp_inv_hd(f[i], prod) : 1;
    }
    G->M[0] = 1; G->mn = 1;
    for (uint64_t i = 0; i < k; i++) {
        unsigned __int128 c = 0;
        for (uint64_t j = 0; j < G->mn; j++) { c += (unsigned __int128)G->M[j] * f[i]->p; G->M[j] = (uint64_t)c; c >>= 64; }
        if (c) G->M[G->mn++] = (uint64_t)c;
    }
    return G;
}
static tr_bigint *garner_value(lctx *x, garner *G, const uint64_t *res) {
    const uint64_t k = G->k;
    uint64_t *v = G->v, *limbs = G->limbs;
    for (uint64_t i = 0; i < k; i++) {                                    /* mixed-radix digits */
        const tr_gfp *f = G->f[i];
        uint64_t acc = 0;
        for (uint64_t j = i; j-- > 0;) acc = tr_gfp_add_hd(f, tr_gfp_mul_hd(f, acc, G->f[j]->p % f->p), v[j] % f->p);
        v[i] = i ? tr_gfp_mul_hd(f, tr_gfp_sub_hd(f, res[i], acc), G->inv[i]) : res[0];
    }
    uint64_t n = 1;                                                       /* x = v0 + p0 (v1 + p1 (...)) */
    limbs[0] = v[k - 1];
    for (uint64_t i = k - 1; i-- > 0;) {
        unsigned __int128 c = v[i];
        for (uint64_t j = 0; j < n; j++) { c += (unsigned __int128)limbs[j] * G->f[i]->p; limbs[j] = (uint64_t)c; c >>= 64; }
        if (c) limbs[n++] = (uint64_t)c;
    }
    while (n > 1 && !limbs[n - 1]) n--;
    int gt = 0;                                                           /* 2 x > M: the symmetric x - M */
    uint64_t top = n > G->mn ? n : G->mn;
    for (uint64_t j = top + 1; j-- > 0;) {
        uint64_t a = (j < n ? limbs[j] << 1 : 0) | (j && j - 1 < n ? limbs[j - 1] >> 63 : 0);
        uint64_t b = j < G->mn ? G->M[j] : 0;
        if (a != b) { gt = a > b; break; }
    }
    int sign = 1;
    if (gt) {
        unsigned __int128 br = 0;
        for (uint64_t j = 0; j < G->mn; j++) {
            unsigned __int128 t = (unsigned __int128)G->M[j] - (j < n ? limbs[j] : 0) - br;
            limbs[j] = (uint64_t)t; br = (t >> 64) ? 1 : 0;
        }
        n = G->mn;
        while (n > 1 && !limbs[n - 1]) n--;
        sign = -1;
    }
    int32_t e = 0;
    return x->rc ? NULL : zk(x, tr_bigint_from_limbs(sign, limbs, n, x->b, x->r, &e), e);
}
static int mask_less(uint64_t a, uint64_t b) { uint64_t d = a ^ b; return d && (a & d & (~d + 1)); }

/* one system: the agreeing primes, CRT, the canonical solution verified against the integer system; *deficit > 0
 * when more primes are needed */
static tr_linupd_model *finish_one(lctx *x, const tr_linupd_logo *L, uint64_t s, const tr_gfp *primes, uint64_t np,
                                   const uint64_t *residues, uint64_t *deficit) {
    const int32_t n = L->v.n, c = n + 1;
    const uint64_t blk = tr_linupd_block(&L->v);
    const uint64_t *res = residues + s * np * blk;
    uint64_t best = UINT64_MAX, bestcount = 0;
    for (uint64_t q = 0; q < np; q++) {
        const uint64_t *h = res + q * blk;
        uint64_t cnt = 0;
        for (uint64_t q2 = 0; q2 < np; q2++) { const uint64_t *h2 = res + q2 * blk; cnt += h2[0] == h[0] && h2[1] == h[1] && h2[2] == h[2]; }
        if (best == UINT64_MAX) { best = q; bestcount = cnt; continue; }
        const uint64_t *hb = res + best * blk;
        int better = h[0] > hb[0] || (h[0] == hb[0] && (mask_less(h[1], hb[1]) || (h[1] == hb[1] &&
                     (cnt > bestcount || (cnt == bestcount && mask_less(h[2], hb[2]))))));
        if (better) { best = q; bestcount = cnt; }
    }
    const uint64_t *hb = res + best * blk;
    const uint64_t k = L->need[s];
    if (bestcount < k) { *deficit = k - bestcount; return NULL; }
    const tr_gfp *fs[TR_LINUPD_MAX_PRIMES];
    uint64_t idx[TR_LINUPD_MAX_PRIMES], kk = 0;
    for (uint64_t q = 0; q < np && kk < k; q++) {
        const uint64_t *h = res + q * blk;
        if (h[0] == hb[0] && h[1] == hb[1] && h[2] == hb[2]) { idx[kk] = q; fs[kk] = &primes[q]; kk++; }
    }
    const int32_t r = (int32_t)hb[0];
    int32_t pivcol[TR_LIN_MAX_COLS + 1], npiv = 0;
    for (int32_t j = 0; j < c; j++) if ((hb[1] >> j) & 1) pivcol[npiv++] = j;
    tr_linupd_model *m = model_new(x);
    if (!m) return NULL;
    logo_system(x, L, s, m);
    if (npiv != r || (r && pivcol[r - 1] == n)) {                         /* inconsistent: LIN-002's certificate */
        solve_core(x, m, 0);
        m->info.method = TR_LINUPD_SUBTRACTED;
        model_stats(x, m);
        return m;
    }
    garner *G = garner_new(fs, k);
    if (!G) { lfail(x, TR_MATH_MEMORY); return m; }
    uint64_t rv[TR_LINUPD_MAX_PRIMES];
    for (uint64_t i = 0; i < k; i++) rv[i] = res[idx[i] * blk + 3];
    tr_bigint *D = garner_value(x, G, rv), **DR = Z_arr(x, (uint64_t)(r ? r : 1) * c);
    int ispiv[TR_LIN_MAX_COLS + 1] = {0};
    for (int32_t t = 0; t < r; t++) ispiv[pivcol[t]] = 1;
    for (int32_t t = 0; DR && t < r && LIVE(x); t++)
        for (int32_t j = 0; j < c && LIVE(x); j++) {
            if (ispiv[j]) continue;
            for (uint64_t i = 0; i < k; i++) rv[i] = res[idx[i] * blk + 4 + (uint64_t)t * c + j];
            DR[(uint64_t)t * c + j] = garner_value(x, G, rv);
        }
    garner_free(G);
    /* the RREF form and A (D x0) = D b, A (D N_f) = 0 against the exact integer system */
    tr_bigint *const *M = L->ints + s * (uint64_t)n * c;
    int bad = !D || !Z_sign(D);
    for (int32_t t = 0; !bad && t < r && LIVE(x); t++)
        for (int32_t j = 0; j < pivcol[t]; j++) if (!ispiv[j] && Z_sign(DR[(uint64_t)t * c + j])) bad = 1;
    const tr_bigint *a[TR_LIN_MAX_COLS + 1], *yv[TR_LIN_MAX_COLS + 1];
    for (int32_t j = 0; !bad && j < c && LIVE(x); j++) {
        if (ispiv[j]) continue;
        for (int32_t i = 0; !bad && i < n && LIVE(x); i++) {
            for (int32_t t = 0; t < r; t++) { a[t] = M[(uint64_t)i * c + pivcol[t]]; yv[t] = DR[(uint64_t)t * c + j]; }
            tr_bigint *lhs = r ? Z_dot(x, a, yv, (uint64_t)r) : Z_i64(x, 0), *rhs = Z_mul(x, D, M[(uint64_t)i * c + j]);
            if (LIVE(x) && tr_bigint_cmp(lhs, rhs) != 0) bad = 1;
            tr_bigint_free(lhs); tr_bigint_free(rhs);
        }
    }
    if (bad && LIVE(x)) {                                                 /* an unlucky set: rerun with more */
        *deficit = k;
        tr_bigint_free(D); Z_free_all(DR, DR ? (uint64_t)(r ? r : 1) * c : 0);
        tr_linupd_free(m);
        return NULL;
    }
    tr_bigq **xs = Q_arr(x, (uint64_t)n), **null = n > r ? Q_arr(x, (uint64_t)(n - r) * n) : NULL;
    for (int32_t j = 0; xs && j < n && LIVE(x); j++) xs[j] = Q_int(x, 0);
    for (int32_t t = 0; xs && t < r && LIVE(x); t++) { tr_bigq_free(xs[pivcol[t]]); xs[pivcol[t]] = Q_frac(x, DR[(uint64_t)t * c + n], D); }
    for (int32_t j = 0, f = 0; null && j < n && LIVE(x); j++) {
        if (ispiv[j]) continue;
        tr_bigq **vv = null + (uint64_t)f * n;
        for (int32_t i = 0; i < n && LIVE(x); i++) vv[i] = Q_int(x, i == j);
        for (int32_t t = 0; t < r && LIVE(x); t++) {
            int32_t e = 0;
            tr_bigint *neg = zk(x, tr_bigint_neg(DR[(uint64_t)t * c + j], x->b, x->r, &e), e);
            tr_bigq_free(vv[pivcol[t]]);
            vv[pivcol[t]] = Q_frac(x, neg, D);
            tr_bigint_free(neg);
        }
        f++;
    }
    tr_bigint_free(D); Z_free_all(DR, DR ? (uint64_t)(r ? r : 1) * c : 0);
    set_part(m, TR_LINUPD_PART_X, xs, xs ? (uint64_t)n : 0);
    set_part(m, TR_LINUPD_PART_NULL, null, null ? (uint64_t)(n - r) * n : 0);
    m->info.kind = r == n ? TR_LIN_UNIQUE : TR_LIN_FAMILY; m->info.rank = r;
    m->info.pivots = hb[1] & ~((uint64_t)1 << n);
    m->info.method = TR_LINUPD_MODULAR; m->info.primes = k;
    model_stats(x, m);
    return m;
}
int32_t tr_linupd_logo_finish(const tr_linupd_logo *L, const tr_gfp *primes, uint64_t np, const uint64_t *residues,
                              tr_linupd_model **out, uint64_t *need, const tr_math_budget *budget,
                              tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!logo_ok(L) || !residues || !out || !need || !primes_ok(primes, np, 1)) return TR_MATH_INVALID;
    for (uint64_t s = 0; s < L->v.nsys; s++) out[s] = NULL;
    *need = 0;
    int32_t rc = model_need(L->v.n, 0, budget, refusal);
    if (rc) return rc;
    lctx x = {budget, refusal, 0};
    uint64_t deficit = 0;
    for (uint64_t s = 0; s < L->v.nsys && LIVE(&x); s++) {
        uint64_t dd = 0;
        out[s] = finish_one(&x, L, s, primes, np, residues, &dd);
        if (dd > deficit) deficit = dd;
    }
    rc = x.rc;
    if (!rc && deficit) { rc = TR_MATH_NOT_CONVERGED; *need = np + deficit + TR_LIN_PRIME_MARGIN; }
    if (rc) free_models(out, L->v.nsys);
    return rc;
}

/* ================================================================ the floating variant */
int32_t tr_linupd_float_solve(const tr_linupd_model *m, double tol, double *xout, tr_linupd_float_info *info,
                              const tr_math_budget *budget, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!model_ok(m) || !xout || !info || !(tol >= 0) || !isfinite(tol)) return TR_MATH_INVALID;
    const int32_t n = m->info.n;
    const uint64_t nn = (uint64_t)n * n;
    tr_math_need need;
    memset(&need, 0, sizeof need);
    need.need[TR_BUDGET_TENSOR] = nn;
    need.need[TR_BUDGET_BYTES] = tr_budget_mul(nn, 6 * 8);
    tr_math_budget dflt;
    int32_t rc = tr_budget_check(budget_or(budget, &dflt), &need, refusal);
    if (rc) return rc;
    memset(info, 0, sizeof *info);
    info->tol = tol;
    if (m->info.kind != TR_LIN_UNIQUE) return TR_MATH_SINGULAR;           /* the exact system is singular */
    tr_math_interval *Ai = (tr_math_interval *)zalloc(nn * sizeof *Ai), *bi = (tr_math_interval *)zalloc((uint64_t)n * sizeof *bi);
    double *Am = (double *)zalloc(nn * 8), *R = (double *)zalloc(nn * 8), *work = (double *)zalloc(nn * 8);
    double *bm = (double *)zalloc((uint64_t)n * 8), *xt = (double *)zalloc((uint64_t)n * 8), *rr = (double *)zalloc((uint64_t)n * 8);
    rc = Ai && bi && Am && R && work && bm && xt && rr ? TR_MATH_OK : TR_MATH_MEMORY;
    for (uint64_t i = 0; !rc && i < nn; i++) {
        rc = tr_bigq_enclose(m->part[TR_LINUPD_PART_A][i], &Ai[i]);
        Am[i] = Ai[i].lo / 2 + Ai[i].hi / 2;
    }
    for (int32_t i = 0; !rc && i < n; i++) { rc = tr_bigq_enclose(m->part[TR_LINUPD_PART_B][i], &bi[i]); bm[i] = bi[i].lo / 2 + bi[i].hi / 2; }
    int ok = !rc;
    /* R ~ Am^-1: Gauss-Jordan with partial pivoting in binary64 */
    if (ok) {
        memcpy(work, Am, (size_t)nn * 8);
        for (int32_t i = 0; i < n; i++) for (int32_t j = 0; j < n; j++) R[(uint64_t)i * n + j] = i == j;
        for (int32_t col = 0; col < n && ok; col++) {
            int32_t piv = col;
            for (int32_t i = col + 1; i < n; i++) if (fabs(work[(uint64_t)i * n + col]) > fabs(work[(uint64_t)piv * n + col])) piv = i;
            if (work[(uint64_t)piv * n + col] == 0 || !isfinite(work[(uint64_t)piv * n + col])) { ok = 0; info->reason = 1; break; }
            if (piv != col)
                for (int32_t j = 0; j < n; j++) {
                    double t = work[(uint64_t)piv * n + j]; work[(uint64_t)piv * n + j] = work[(uint64_t)col * n + j]; work[(uint64_t)col * n + j] = t;
                    t = R[(uint64_t)piv * n + j]; R[(uint64_t)piv * n + j] = R[(uint64_t)col * n + j]; R[(uint64_t)col * n + j] = t;
                }
            double p = work[(uint64_t)col * n + col];
            for (int32_t j = 0; j < n; j++) { work[(uint64_t)col * n + j] /= p; R[(uint64_t)col * n + j] /= p; }
            for (int32_t i = 0; i < n; i++) {
                if (i == col) continue;
                double f = work[(uint64_t)i * n + col];
                if (f == 0) continue;
                for (int32_t j = 0; j < n; j++) { work[(uint64_t)i * n + j] -= f * work[(uint64_t)col * n + j]; R[(uint64_t)i * n + j] -= f * R[(uint64_t)col * n + j]; }
            }
        }
    }
    if (ok) {                                                              /* x~ = R b, one refinement step */
        for (int32_t i = 0; i < n; i++) { double s = 0; for (int32_t j = 0; j < n; j++) s += R[(uint64_t)i * n + j] * bm[j]; xt[i] = s; }
        for (int32_t i = 0; i < n; i++) { double s = bm[i]; for (int32_t j = 0; j < n; j++) s -= Am[(uint64_t)i * n + j] * xt[j]; rr[i] = s; }
        for (int32_t i = 0; i < n; i++) { double s = 0; for (int32_t j = 0; j < n; j++) s += R[(uint64_t)i * n + j] * rr[j]; xt[i] += s; }
        for (int32_t i = 0; i < n; i++) if (!isfinite(xt[i])) { ok = 0; info->reason = 1; }
    }
    if (ok) {
        /* alpha >= ||I - R A||_inf, kappa >= ||A||_inf ||R||_inf (outward) */
        double alpha = 0, an = 0, rn = 0;
        for (int32_t i = 0; i < n; i++) {
            double rowc = 0, rowa = 0, rowr = 0;
            for (int32_t j = 0; j < n; j++) {
                tr_math_interval acc = tr_interval_point(i == j ? 1.0 : 0.0);
                for (int32_t t = 0; t < n; t++)
                    acc = tr_interval_sub(acc, tr_interval_mul(tr_interval_point(R[(uint64_t)i * n + t]), Ai[(uint64_t)t * n + j]));
                rowc = tr_add_ru(rowc, tr_interval_abs(acc).hi);
                rowa = tr_add_ru(rowa, tr_interval_abs(Ai[(uint64_t)i * n + j]).hi);
                rowr = tr_add_ru(rowr, fabs(R[(uint64_t)i * n + j]));
            }
            if (rowc > alpha) alpha = rowc;
            if (rowa > an) an = rowa;
            if (rowr > rn) rn = rowr;
        }
        double xn = 0;
        for (int32_t i = 0; i < n; i++) if (fabs(xt[i]) > xn) xn = fabs(xt[i]);
        info->alpha = alpha; info->kappa = tr_mul_dir(an, rn, 1); info->xnorm = xn;
        if (!(alpha < 1)) { ok = 0; info->reason = 2; }
        if (ok) {
            /* ||x* - x~|| <= ||R (b - A x~)|| / (1 - alpha) */
            double zn = 0;
            tr_math_interval *res = (tr_math_interval *)zalloc((uint64_t)n * sizeof *res);
            if (!res) rc = TR_MATH_MEMORY;
            for (int32_t i = 0; res && i < n; i++) {
                tr_math_interval acc = bi[i];
                for (int32_t j = 0; j < n; j++) acc = tr_interval_sub(acc, tr_interval_mul(Ai[(uint64_t)i * n + j], tr_interval_point(xt[j])));
                res[i] = acc;
            }
            for (int32_t i = 0; res && i < n; i++) {
                tr_math_interval acc = tr_interval_point(0.0);
                for (int32_t j = 0; j < n; j++) acc = tr_interval_add(acc, tr_interval_mul(tr_interval_point(R[(uint64_t)i * n + j]), res[j]));
                double a = tr_interval_abs(acc).hi;
                if (a > zn) zn = a;
            }
            free(res);
            double bound = tr_div_dir(zn, tr_add_rd(1.0, -alpha), 1);
            info->bound = bound;
            if (!rc && !(bound <= tol)) { ok = 0; info->reason = 3; }
        }
    }
    if (!rc && ok) { memcpy(xout, xt, (size_t)n * 8); info->certified = 1; }
    free(Ai); free(bi); free(Am); free(R); free(work); free(bm); free(xt); free(rr);
    if (rc) return rc;
    return ok ? TR_MATH_OK : TR_MATH_NOT_CONVERGED;
}
