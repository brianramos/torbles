/* Copyright 2026 Brian Richard RAMOS
 * SPDX-License-Identifier: Apache-2.0
 * Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0). */
/* torbles: the algebraic engine (CPU).
 * A torble is stored as torus bit planes: plane j of torus lane L holds bit j of L for every row (64 rows per word).
 * Column c is torus row c; with value width b the potential phi_c has bits 1..b, so only these planes can be nonzero:
 *   V_c[j] = phi_c(j) ^ phi_c(j+1),         j = 0..b_c
 *   H_i[j] = phi_i(j) ^ phi_(i+1)(j),       i < k-1;   H_(k-1)[j] = phi_(k-1)(j);   wrap H_15[j] = phi_0(j)
 * Every other plane of the 16 x 32 torus is zero and is not stored; faces through it are still checked.
 *   decode   : value bit j = XOR of V_c[0..j]                     (prefix over planes, 64 rows per op)
 *   validate : face(i,j) = H_i[j] ^ H_i[j+1] ^ V_i[j] ^ V_(i+1)[j] (XOR of planes)
 *   filter   : any predicate is a Boolean polynomial evaluated on value planes
 *   SUM(x)   : sum_j 2^j popcount(x_j & F);   SUM(x*y): sum_jk 2^(j+k) popcount(x_j & y_k & F)
 * A plain columnar engine (uint32 columns, vectorized chunks) runs the same plans for comparison.
 * Build: gcc -O2 -march=native -shared -fPIC alg.c -o libalg.so */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <stdatomic.h>

#include "torble.h"
#include <unistd.h>

const char *at_version(void) { return TORBLES_VERSION; }

/* ---------------------------------------------------------------- persistent worker pool
 * Every parallel call (queries, filters, validation, the build) runs on one pool of threads created once. Woken
 * workers spin for a short while before sleeping, so back-to-back calls do not pay for thread creation or a wake-up.
 * pool_jobs(T, worker, jobs, size): worker(jobs + i * size) for i in [0, T), the caller taking i = 0. Nested calls
 * (from inside a pool task) run serially. */
#include <time.h>
#include <immintrin.h>
typedef void *(*worker_fn)(void *);
typedef struct { worker_fn fn; char *jobs; size_t size; _Atomic int remaining; } pdesc_t;
typedef struct { _Atomic uint64_t gen; pdesc_t *desc; pthread_mutex_t mu; pthread_cond_t cv; char pad[64]; } pslot_t;
static pslot_t SLOT[256]; static int NWORKERS;                 /* worker i (1..) watches only its own slot */
static pthread_mutex_t POOL_RUN = PTHREAD_MUTEX_INITIALIZER, DONE_MU = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t DONE_CV = PTHREAD_COND_INITIALIZER;
static __thread int IN_POOL;
static double now_us(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec * 1e6 + ts.tv_nsec * 1e-3; }
#define POOL_SPIN_US 300.0
static void *pool_main(void *p) {
    int id = (int)(intptr_t)p; pslot_t *S = &SLOT[id]; uint64_t seen = 0; IN_POOL = 1;
    for (;;) {
        uint64_t g = atomic_load(&S->gen);
        if (g == seen) {                                   /* spin briefly, then sleep */
            double t0 = now_us();
            for (int i = 0; (g = atomic_load(&S->gen)) == seen; i++) { _mm_pause(); if ((i & 255) == 255 && now_us() - t0 > POOL_SPIN_US) break; }
            if (g == seen) { pthread_mutex_lock(&S->mu); while ((g = atomic_load(&S->gen)) == seen) pthread_cond_wait(&S->cv, &S->mu); pthread_mutex_unlock(&S->mu); }
        }
        seen = g;
        pdesc_t *d = S->desc;
        d->fn(d->jobs + (size_t)id * d->size);
        if (atomic_fetch_sub(&d->remaining, 1) == 1) {      /* d is not touched after this: the caller may return */
            pthread_mutex_lock(&DONE_MU); pthread_cond_broadcast(&DONE_CV); pthread_mutex_unlock(&DONE_MU);
        }
    }
    return NULL;
}
static void pool_jobs(int T, worker_fn fn, void *jobs, size_t size) {
    if (T <= 1 || IN_POOL) { for (int i = 0; i < T; i++) fn((char *)jobs + (size_t)i * size); return; }
    if (T > 256) T = 256;
    pthread_mutex_lock(&POOL_RUN);
    while (NWORKERS < T - 1) {
        int id = ++NWORKERS; pthread_mutex_init(&SLOT[id].mu, NULL); pthread_cond_init(&SLOT[id].cv, NULL); atomic_init(&SLOT[id].gen, 0);
        pthread_t th; pthread_create(&th, NULL, pool_main, (void *)(intptr_t)id); pthread_detach(th);
    }
    pdesc_t d; d.fn = fn; d.jobs = jobs; d.size = size; atomic_init(&d.remaining, T - 1);
    for (int i = 1; i < T; i++) {                          /* wake exactly the T - 1 workers needed */
        pslot_t *S = &SLOT[i]; S->desc = &d;
        pthread_mutex_lock(&S->mu); atomic_fetch_add(&S->gen, 1); pthread_cond_signal(&S->cv); pthread_mutex_unlock(&S->mu);
    }
    IN_POOL = 1; fn(jobs); IN_POOL = 0;
    double t0 = now_us();
    for (int i = 0; atomic_load(&d.remaining) > 0; i++) { _mm_pause(); if ((i & 255) == 255 && now_us() - t0 > POOL_SPIN_US) break; }
    if (atomic_load(&d.remaining) > 0) { pthread_mutex_lock(&DONE_MU); while (atomic_load(&d.remaining) > 0) pthread_cond_wait(&DONE_CV, &DONE_MU); pthread_mutex_unlock(&DONE_MU); }
    pthread_mutex_unlock(&POOL_RUN);
}

/* ---------------------------------------------------------------- parallel loops (ingestion, decoding)
 * par_for(n, grain, fn, ctx): fn(ctx, a, b) over [0, n) in chunks of `grain`, on BUILD_THREADS threads (default: all
 * online CPUs; at_set_build_threads). Chunks are handed out by an atomic counter. */
static int BUILD_THREADS = 0;
void at_set_build_threads(int n) { BUILD_THREADS = n < 1 ? 1 : n > 256 ? 256 : n; }
int at_get_build_threads(void) {
    if (!BUILD_THREADS) { long c = sysconf(_SC_NPROCESSORS_ONLN); BUILD_THREADS = c < 1 ? 1 : c > 256 ? 256 : (int)c; }
    return BUILD_THREADS;
}
typedef void (*range_fn)(void *ctx, uint64_t a, uint64_t b);
typedef struct { range_fn fn; void *ctx; uint64_t n, grain; _Atomic uint64_t *next; } pjob_t;
static void *pworker(void *arg) {
    pjob_t *j = arg;
    for (;;) { uint64_t a = atomic_fetch_add(j->next, j->grain); if (a >= j->n) break;
               j->fn(j->ctx, a, a + j->grain < j->n ? a + j->grain : j->n); }
    return NULL;
}
static void par_for(uint64_t n, uint64_t grain, range_fn fn, void *ctx) {
    if (!n) return;
    int T = at_get_build_threads(); if ((uint64_t)T > (n + grain - 1) / grain) T = (int)((n + grain - 1) / grain);
    if (T <= 1) { fn(ctx, 0, n); return; }
    _Atomic uint64_t next = 0; pjob_t j[256];
    for (int i = 0; i < T; i++) j[i] = (pjob_t){fn, ctx, n, grain, &next};
    pool_jobs(T, pworker, j, sizeof j[0]);
}
/* column maxima, in parallel (per-chunk maxima merged with an atomic max) */
typedef struct { const uint32_t **cols; int ncols; _Atomic uint32_t mx[64]; } maxctx_t;
static void max_range(void *v, uint64_t a, uint64_t b) {
    maxctx_t *m = v;
    for (int c = 0; c < m->ncols; c++) { uint32_t x = 0; const uint32_t *col = m->cols[c];
        for (uint64_t r = a; r < b; r++) x = col[r] > x ? col[r] : x;
        uint32_t cur = atomic_load(&m->mx[c]); while (x > cur && !atomic_compare_exchange_weak(&m->mx[c], &cur, x)) {} }
}
static void col_widths(uint64_t n, int ncols, const uint32_t **cols, int32_t *width) {
    maxctx_t m; m.cols = cols; m.ncols = ncols; for (int c = 0; c < 64; c++) atomic_init(&m.mx[c], 0);
    par_for(n, 1 << 20, max_range, &m);
    for (int c = 0; c < ncols; c++) { uint32_t mx = atomic_load(&m.mx[c]); width[c] = mx ? 32 - __builtin_clz(mx) : 1; }
}

/* ---------------------------------------------------------------- utilities */
static inline uint32_t rev32(uint32_t x) {
    x = ((x >> 1) & 0x55555555u) | ((x & 0x55555555u) << 1); x = ((x >> 2) & 0x33333333u) | ((x & 0x33333333u) << 2);
    x = ((x >> 4) & 0x0F0F0F0Fu) | ((x & 0x0F0F0F0Fu) << 4); return __builtin_bswap32(x);
}
static inline uint32_t revb(uint32_t x, int b) { return rev32(x) >> (32 - b); }
static void transpose64(uint64_t A[64]) {               /* A[r] bit j <- A[j] bit r */
    uint64_t m = 0x00000000FFFFFFFFULL;
    for (int j = 32; j; j >>= 1, m ^= m << j)
        for (int k = 0; k < 64; k = ((k | j) + 1) & ~j) {
            uint64_t t = ((A[k] >> j) ^ A[k | j]) & m;
            A[k] ^= t << j; A[k | j] ^= t;
        }
}
static uint64_t *plane_alloc(atable_t *t) {          /* not cleared: the build writes every word of every plane */
    uint64_t *p = aligned_alloc(64, ((t->nw * 8) + 63) & ~(size_t)63);
    t->bytes += t->nw * 8; return p;
}
static inline uint64_t rowmask(const atable_t *t, uint64_t w) {
    uint64_t left = t->n - w * 64; return left >= 64 ? ~0ULL : ((1ULL << left) - 1);
}

/* ---------------------------------------------------------------- build */
/* the planes of words [w0, w1): one 64 x 64 transpose per column per word, then the V and H planes (independent per
 * word, so the build runs on all cores) */
typedef struct { atable_t *t; const uint32_t **cols; } bctx_t;
static void build_range(void *v, uint64_t w0, uint64_t w1) {
    bctx_t *b = v; atable_t *t = b->t; const uint32_t **cols = b->cols; uint64_t n = t->n; int k = t->k;
    uint64_t A[64], val[MAXCOL][NB];
    for (uint64_t w = w0; w < w1; w++) {
        for (int c = 0; c < k; c++) {                    /* value planes of this word via one 64x64 transpose */
            if (w * 64 + 64 <= n) { const uint32_t *src = cols[c] + w * 64;
                if (t->msb[c]) for (int r = 0; r < 64; r++) A[r] = revb(src[r], t->width[c]); else for (int r = 0; r < 64; r++) A[r] = src[r]; }
            else for (int r = 0; r < 64; r++) { uint32_t x = w * 64 + r < n ? cols[c][w * 64 + r] : 0; A[r] = t->msb[c] ? revb(x, t->width[c]) : x; }
            transpose64(A);
            for (int j = 0; j < NB; j++) val[c][j] = j < 32 ? A[j] : 0;
        }
        for (int c = 0; c < k; c++) {                    /* phi(j) = val(j-1): V[j] = val[j-1] ^ val[j] */
            int bw = t->width[c];
            for (int j = 0; j <= bw; j++) t->V[c][j][w] = (j ? val[c][j - 1] : 0) ^ (j < bw ? val[c][j] : 0);
        }
        for (int i = 0; i < k; i++)
            for (int j = 1; j < NB; j++) if (t->H[i][j])
                t->H[i][j][w] = (j - 1 < t->width[i] ? val[i][j - 1] : 0) ^ (i + 1 < k && j - 1 < t->width[i + 1] ? val[i + 1][j - 1] : 0);
        for (int j = 1; j < NB; j++) if (t->H[k][j]) t->H[k][j][w] = val[0][j - 1];
    }
}
atable_t *at_create(uint64_t n, int ncols, const uint32_t **cols, const int32_t *msb) {
    atable_t *t = calloc(1, sizeof *t);
    t->n = n; t->nw = (n + 63) / 64; t->k = ncols;
    for (int c = 0; c < ncols; c++) t->msb[c] = msb ? msb[c] : 0;
    col_widths(n, ncols, cols, t->width);
    int k = ncols;
    for (int c = 0; c < k; c++) for (int j = 0; j <= t->width[c]; j++) t->V[c][j] = plane_alloc(t);
    for (int i = 0; i < k; i++) {
        int hi = i + 1 < k ? (t->width[i] > t->width[i + 1] ? t->width[i] : t->width[i + 1]) : t->width[i];
        for (int j = 1; j <= hi; j++) t->H[i][j] = plane_alloc(t);
    }
    for (int j = 1; j <= t->width[0]; j++) t->H[k][j] = plane_alloc(t);
    bctx_t b = {t, cols}; par_for(t->nw, 2048, build_range, &b);
    /* face list: every face of the 16 x 32 torus that touches a stored plane */
    for (int ii = 0; ii <= k; ii++) {
        int hrow = ii, vrow = ii < k ? ii : -1, vnext = ii < k ? (ii + 1 < k ? ii + 1 : -1) : 0;   /* ii == k is torus row 15 */
        if (k == 16) break;
        for (int j = 0; j < NB; j++) {                 /* lanes are wide enough for 32-bit values (potentials need 33 bits) */
            uint64_t *terms[4]; int nt = 0;
            if (t->H[hrow][j]) terms[nt++] = t->H[hrow][j];
            if (j + 1 < NB && t->H[hrow][j + 1]) terms[nt++] = t->H[hrow][j + 1];
            if (vrow >= 0 && j <= t->width[vrow] && t->V[vrow][j]) terms[nt++] = t->V[vrow][j];
            if (vnext >= 0 && j <= t->width[vnext] && t->V[vnext][j]) terms[nt++] = t->V[vnext][j];
            if (!nt) continue;
            for (int q = 0; q < nt; q++) t->face[t->nfaces][q] = terms[q];
            t->fterms[t->nfaces++] = nt;
        }
    }
    return t;
}
/* make column c virtual: its values are in sidecar `side` for the rows flagged in column flagcol (rank directory) */
void at_set_virtual(atable_t *t, int c, int flagcol, const atable_t *side, const uint32_t *rank) {
    t->vflag[c] = flagcol + 1; t->vside[c] = side; t->vrank[c] = rank;
    t->width[c] = side->width[0]; t->msb[c] = side->msb[0];
}
/* delta plane: the rows whose foreign key column c (masked by kmask) points at a dimension row set in bm */
uint64_t at_rows_semi(const atable_t *t, int c, uint32_t kmask, const uint64_t *bm, uint64_t bmn, uint64_t *out) {
    uint64_t cnt = 0, A[64]; int b = t->width[c];
    for (uint64_t w = 0; w < t->nw; w++) {
        uint64_t x = 0, o = 0; memset(A, 0, sizeof A);
        for (int j = 0; j < b; j++) { x ^= t->V[c][j][w]; A[j] = x; }
        transpose64(A);
        uint64_t rm = rowmask(t, w);
        for (int r = 0; r < 64; r++) {
            if (!(rm >> r & 1)) break;
            uint64_t v = t->msb[c] ? revb((uint32_t)A[r], b) : A[r]; v = KM(kmask, v);
            if (v < bmn && (bm[v >> 6] >> (v & 63) & 1)) o |= 1ULL << r;
        }
        out[w] = o; cnt += __builtin_popcountll(o);
    }
    return cnt;
}
uint64_t at_bytes(const atable_t *t) { return t->bytes; }
int at_width(const atable_t *t, int c) { return t->width[c]; }
int at_nplanes(const atable_t *t) { return (int)(t->bytes / (t->nw * 8)); }

/* fault injection: flip row r in the p-th stored plane */
void at_flip(atable_t *t, int p, uint64_t r) {
    int q = 0;
    for (int c = 0; c < t->k; c++) for (int j = 0; j < NB; j++) if (t->V[c][j]) { if (q++ == p) { t->V[c][j][r >> 6] ^= 1ULL << (r & 63); return; } }
    for (int i = 0; i <= t->k; i++) for (int j = 0; j < NB; j++) if (t->H[i][j]) { if (q++ == p) { t->H[i][j][r >> 6] ^= 1ULL << (r & 63); return; } }
}
/* integrity: rows with any violated face (counted per row), first bad row ids */
static uint64_t validate_part(const atable_t *t, uint64_t wa, uint64_t wb, uint64_t *bad, uint64_t maxbad) {
    uint64_t nb = 0; enum { BW = 256 };
    uint64_t acc[BW];
    for (uint64_t w0 = wa; w0 < wb; w0 += BW) {
        uint64_t w1 = w0 + BW < wb ? w0 + BW : wb;
        memset(acc, 0, sizeof acc);
        for (int f = 0; f < t->nfaces; f++) {
            uint64_t *const *P = t->face[f];
            switch (t->fterms[f]) {
            case 4: for (uint64_t w = w0; w < w1; w++) acc[w - w0] |= P[0][w] ^ P[1][w] ^ P[2][w] ^ P[3][w]; break;
            case 3: for (uint64_t w = w0; w < w1; w++) acc[w - w0] |= P[0][w] ^ P[1][w] ^ P[2][w]; break;
            case 2: for (uint64_t w = w0; w < w1; w++) acc[w - w0] |= P[0][w] ^ P[1][w]; break;
            default: for (uint64_t w = w0; w < w1; w++) acc[w - w0] |= P[0][w]; break;
            }
        }
        for (uint64_t w = w0; w < w1; w++) {
            uint64_t x = acc[w - w0];
            while (x) { if (bad && nb < maxbad) bad[nb] = w * 64 + __builtin_ctzll(x); nb++; x &= x - 1; }
        }
    }
    return nb;
}
typedef struct { const atable_t *t; uint64_t nw; _Atomic uint64_t *next; uint64_t nb; } vjob_t;
static void *vworker(void *arg) {
    vjob_t *j = arg;
    for (;;) {
        uint64_t w0 = atomic_fetch_add(j->next, 2048);
        if (w0 >= j->nw) break;
        j->nb += validate_part(j->t, w0, w0 + 2048 < j->nw ? w0 + 2048 : j->nw, NULL, 0);
    }
    return NULL;
}
static int nthreads(void);
uint64_t at_validate(const atable_t *t, uint64_t *bad, uint64_t maxbad) {
    int T = nthreads(); vjob_t jobs[64]; pthread_t th[64]; _Atomic uint64_t next = 0;
    if ((uint64_t)T > (t->nw + 2047) / 2048) T = (int)((t->nw + 2047) / 2048); if (T < 1) T = 1;
    for (int i = 0; i < T; i++) jobs[i] = (vjob_t){t, t->nw, &next, 0};
    pool_jobs(T, vworker, jobs, sizeof jobs[0]); (void)th;
    uint64_t nb = 0; for (int i = 0; i < T; i++) nb += jobs[i].nb;
    if (nb && bad && maxbad) validate_part(t, 0, t->nw, bad, maxbad);      /* list the corrupt rows, in order */
    return nb;
}

/* per-thread group accumulator for worker threads: small open-addressing hash, merged into the result at the end */
typedef struct { uint64_t *key; int64_t *sum; uint64_t *cnt; uint64_t cap, n; } touch_t;
#define TEMPTY (~0ULL)
static inline uint64_t thash(uint64_t k) { return (k * 0x9E3779B97F4A7C15ULL) ^ ((k * 0x9E3779B97F4A7C15ULL) >> 29); }
static void hgrow(touch_t *h) {
    uint64_t oc = h->cap, nc = oc ? 2 * oc : 4096; uint64_t *ok = h->key; int64_t *os = h->sum; uint64_t *on = h->cnt;
    h->key = malloc(nc * 8); h->sum = calloc(nc, 8); h->cnt = calloc(nc, 8); h->cap = nc;
    memset(h->key, 0xFF, nc * 8);
    for (uint64_t i = 0; i < oc; i++) if (ok[i] != TEMPTY) {
        uint64_t j = thash(ok[i]) & (nc - 1);
        while (h->key[j] != TEMPTY) j = (j + 1) & (nc - 1);
        h->key[j] = ok[i]; h->sum[j] = os[i]; h->cnt[j] = on[i];
    }
    free(ok); free(os); free(on);
}
static inline void hadd_n(touch_t *h, uint64_t key, int64_t v, uint64_t n);
static inline void hadd(touch_t *h, uint64_t key, int64_t v) { hadd_n(h, key, v, 1); }
static inline void hadd_n(touch_t *h, uint64_t key, int64_t v, uint64_t nadd) {
    if (2 * (h->n + 1) > h->cap) hgrow(h);
    uint64_t j = thash(key) & (h->cap - 1);
    while (h->key[j] != key && h->key[j] != TEMPTY) j = (j + 1) & (h->cap - 1);
    if (h->key[j] == TEMPTY) { h->key[j] = key; h->n++; }
    h->sum[j] += v; h->cnt[j] += nadd;
}
/* ---------------------------------------------------------------- algebraic evaluation, one 64-row word at a time */
typedef struct { int8_t nd[MAXCOL]; uint64_t p[MAXCOL][32]; } dec_t;   /* nd[c]: planes of column c decoded so far */
#define DEC_RESET(D) memset((D).nd, 0, sizeof (D).nd)
static inline const uint64_t *decode_n(const atable_t *t, dec_t *D, int c, uint64_t w, int n) {
    int d = D->nd[c];
    if (t->vflag[c]) {                                   /* self-extending column: planes from the sidecar */
        if (d < t->width[c]) {
            int f = t->vflag[c] - 1; uint64_t fw = t->V[f][0][w];   /* the flag column's value bit 0 = its V plane 0 */
            tb_virtual_planes(t, c, fw, w, D->p[c], t->width[c]); D->nd[c] = (int8_t)t->width[c];
        }
        return D->p[c];
    }
    if (d < n) {                                     /* region read: prefix XOR over planes d..n-1 only */
        uint64_t x = d ? D->p[c][d - 1] : 0;
        for (int j = d; j < n; j++) { x ^= t->V[c][j][w]; D->p[c][j] = x; }
        D->nd[c] = (int8_t)n;
    }
    return D->p[c];
}
static inline const uint64_t *decode(const atable_t *t, dec_t *D, int c, uint64_t w) { return decode_n(t, D, c, w, t->width[c]); }
static inline uint64_t cmp_planes(const uint64_t *x, int b, int op, uint32_t c);
static inline uint64_t cmp_planes_msb(const uint64_t *x, int b, int op, uint32_t c) {   /* plane j holds bit b-1-j */
    if (b < 32 && (c >> b)) return (op == P_LT || op == P_LE || op == P_NE) ? ~0ULL : 0;
    uint64_t lt = 0, eq = ~0ULL;
    for (int j = 0; j < b; j++) {
        if (c >> (b - 1 - j) & 1) { lt |= eq & ~x[j]; eq &= x[j]; } else eq &= ~x[j];
    }
    switch (op) {
    case P_EQ: return eq; case P_NE: return ~eq; case P_LT: return lt; case P_LE: return lt | eq;
    case P_GT: return ~(lt | eq); default: return ~lt;
    }
}
static inline uint64_t cmp_any(const atable_t *t, const uint64_t *x, int col, int op, uint32_t c) {
    return t->msb[col] ? cmp_planes_msb(x, t->width[col], op, c) : cmp_planes(x, t->width[col], op, c);
}
static inline uint64_t cmp_planes(const uint64_t *x, int b, int op, uint32_t c) {
    if (b < 32 && (c >> b)) return (op == P_LT || op == P_LE || op == P_NE) ? ~0ULL : 0;   /* constant beyond the column's range */
    uint64_t lt = 0, eq = ~0ULL;
    for (int j = b - 1; j >= 0; j--) {
        if (c >> j & 1) { lt |= eq & ~x[j]; eq &= x[j]; } else eq &= ~x[j];
    }
    switch (op) {
    case P_EQ: return eq; case P_NE: return ~eq; case P_LT: return lt; case P_LE: return lt | eq;
    case P_GT: return ~(lt | eq); default: return ~lt;
    }
}
static inline uint64_t eval_prog(const atable_t *t, const pinstr_t *prog, int n, dec_t *D, uint64_t w) {
    uint64_t st[48]; int sp = 0;
    for (int i = 0; i < n; i++) {
        const pinstr_t *I = &prog[i];
        switch (I->op) {
        case P_TRUE: st[sp++] = ~0ULL; break;
        case P_AND: sp--; st[sp - 1] &= st[sp]; break;
        case P_OR: sp--; st[sp - 1] |= st[sp]; break;
        case P_XOR: sp--; st[sp - 1] ^= st[sp]; break;
        case P_NOT: st[sp - 1] = ~st[sp - 1]; break;
        case P_BETWEEN: { const uint64_t *x = decode(t, D, I->col, w); int b = t->width[I->col];
            st[sp++] = cmp_any(t, x, I->col, P_GE, I->a) & (I->b >> b && b < 32 ? ~0ULL : cmp_any(t, x, I->col, P_LE, I->b)); break; }
        case P_CUBE: {                               /* cube: AND of literals on the planes in mask b (an id coordinate field) */
            uint32_t mk = I->b; int hi = 31 - __builtin_clz(mk); const uint64_t *x = decode_n(t, D, I->col, w, hi + 1); uint64_t m = ~0ULL;
            while (mk) { int j = __builtin_ctz(mk); mk &= mk - 1; m &= (I->a >> j & 1) ? x[j] : ~x[j]; }
            st[sp++] = m; break; }
        case P_CLT: case P_CGE: {                    /* factored comparator on the planes in mask b, MSB of the code first */
            uint32_t mk = I->b; int hi = 31 - __builtin_clz(mk), k = __builtin_popcount(mk) - 1, ms = t->msb[I->col];
            const uint64_t *x = decode_n(t, D, I->col, w, hi + 1); uint64_t lt = 0, eq = ~0ULL;
            while (mk) {                             /* msb layout: ascending planes; lsb layout: descending planes */
                int j = ms ? __builtin_ctz(mk) : 31 - __builtin_clz(mk); mk &= ~(1u << j);
                if (I->a >> k & 1) { lt |= eq & ~x[j]; eq &= x[j]; } else eq &= ~x[j];
                k--;
            }
            st[sp++] = I->op == P_CLT ? lt : ~lt; break; }
        case P_BIT: {                                /* one plane: the column's value bit a */
            int b = t->width[I->col], j = t->msb[I->col] ? b - 1 - (int)I->a : (int)I->a;
            st[sp++] = (int)I->a < b ? decode_n(t, D, I->col, w, j + 1)[j] : 0; break; }
        case P_PREFIX: {                             /* monomial: AND of nbits literals on the first planes */
            int nb = (int)I->b; const uint64_t *x = decode_n(t, D, I->col, w, nb); uint64_t m = ~0ULL;
            for (int j = 0; j < nb; j++) m &= (I->a >> (nb - 1 - j) & 1) ? x[j] : ~x[j];
            st[sp++] = m; break; }
        default: st[sp++] = cmp_any(t, decode(t, D, I->col, w), I->col, I->op, I->a); break;
        }
    }
    return sp ? st[0] : ~0ULL;
}
static inline uint32_t extract(const uint64_t *x, int b, int r) {    /* integer value of row r from value planes */
    uint32_t v = 0; for (int j = 0; j < b; j++) v |= (uint32_t)((x[j] >> r) & 1) << j; return v;
}
static inline uint32_t extract_msb(const uint64_t *x, int nb, int r) {  /* first nb planes, plane 0 = most significant */
    uint32_t v = 0; for (int j = 0; j < nb; j++) v = (v << 1) | (uint32_t)((x[j] >> r) & 1); return v;
}
static inline uint32_t extract_col(const atable_t *t, const uint64_t *x, int c, int r) {
    return t->msb[c] ? extract_msb(x, t->width[c], r) : extract(x, t->width[c], r);
}
/* filter a (dimension) torble into a row bitmap */
typedef struct { const atable_t *t; const pinstr_t *prog; int n; uint64_t *out; const uint64_t *mask; _Atomic uint64_t *next; uint64_t cnt; } fjob_t;
static void *fworker(void *arg) {
    fjob_t *j = arg; dec_t D;
    for (;;) {
        uint64_t w0 = atomic_fetch_add(j->next, 1024);
        if (w0 >= j->t->nw) break;
        uint64_t w1 = w0 + 1024 < j->t->nw ? w0 + 1024 : j->t->nw;
        for (uint64_t w = w0; w < w1; w++) {
            DEC_RESET(D);
            uint64_t x = eval_prog(j->t, j->prog, j->n, &D, w) & rowmask(j->t, w);
            if (j->mask) x &= j->mask[w];
            j->out[w] = x; j->cnt += __builtin_popcountll(x);
        }
    }
    return NULL;
}
static int nthreads(void);
/* filter a (dimension) torble into a row bitmap, optionally ANDed with a mask (e.g. ids in use) */
uint64_t at_filter_bitmap_masked(const atable_t *t, const pinstr_t *prog, int n, uint64_t *out, const uint64_t *mask) {
    int T = nthreads(); fjob_t jobs[64]; pthread_t th[64]; _Atomic uint64_t next = 0;
    if ((uint64_t)T > (t->nw + 1023) / 1024) T = (int)((t->nw + 1023) / 1024); if (T < 1) T = 1;   /* one chunk = 1024 words */
    for (int i = 0; i < T; i++) jobs[i] = (fjob_t){t, prog, n, out, mask, &next, 0};
    pool_jobs(T, fworker, jobs, sizeof jobs[0]); (void)th;
    uint64_t c = 0; for (int i = 0; i < T; i++) c += jobs[i].cnt;
    return c;
}
uint64_t at_filter_bitmap(const atable_t *t, const pinstr_t *prog, int n, uint64_t *out) { return at_filter_bitmap_masked(t, prog, n, out, NULL); }
/* set bits of a bitmap -> sorted ids */
uint64_t bm_ids(const uint64_t *bm, uint64_t nw, uint32_t *out) {
    uint64_t k = 0;
    for (uint64_t w = 0; w < nw; w++) { uint64_t x = bm[w]; while (x) { out[k++] = (uint32_t)(w * 64 + __builtin_ctzll(x)); x &= x - 1; } }
    return k;
}
/* decode a full column to integers (dimension attributes) */
typedef struct { const atable_t *t; int c; uint32_t *out; } dcctx_t;
static void decode_range(void *v, uint64_t w0, uint64_t w1);
void at_decode_column(const atable_t *t, int c, uint32_t *out) { dcctx_t d = {t, c, out}; par_for(t->nw, 4096, decode_range, &d); }
static void decode_range(void *v, uint64_t w0, uint64_t w1) {
    const dcctx_t *d = v; const atable_t *t = d->t; int c = d->c; uint32_t *out = d->out;
    uint64_t A[64]; dec_t D;
    for (uint64_t w = w0; w < w1; w++) {
        DEC_RESET(D); const uint64_t *x = decode(t, &D, c, w);
        memset(A, 0, sizeof A); for (int j = 0; j < t->width[c]; j++) A[j] = x[j];
        transpose64(A);
        for (int r = 0; r < 64 && w * 64 + r < t->n; r++) out[w * 64 + r] = t->msb[c] ? revb((uint32_t)A[r], t->width[c]) : (uint32_t)A[r];
    }
}
/* values of column c at the given rows (any column, virtual ones included): each row's word is decoded once */
void at_get_rows(const atable_t *t, int c, const uint64_t *rows, uint64_t n, uint32_t *out) {
    dec_t D; uint64_t cur = ~0ULL; const uint64_t *x = 0;
    for (uint64_t i = 0; i < n; i++) {
        uint64_t w = rows[i] >> 6;
        if (w != cur) { DEC_RESET(D); x = decode(t, &D, c, w); cur = w; }
        out[i] = extract_col(t, x, c, (int)(rows[i] & 63));
    }
}
/* payload offsets: the tail of row r starts at dir[w] + sum over the rows before r in its word of max(0, len - H).
 * Only one number per 64 rows is stored (dir); the rest is derived from the length planes. */
void at_tail_offsets(const atable_t *t, int lc, uint32_t H, const uint64_t *dir, const uint64_t *rows, uint64_t n, uint64_t *out) {
    dec_t D; uint64_t cur = ~0ULL; uint64_t pre[65];
    for (uint64_t i = 0; i < n; i++) {
        uint64_t w = rows[i] >> 6;
        if (w != cur) {
            DEC_RESET(D); const uint64_t *x = decode(t, &D, lc, w); pre[0] = 0;
            for (int r = 0; r < 64; r++) { uint32_t L = w * 64 + r < t->n ? extract_col(t, x, lc, r) : 0; pre[r + 1] = pre[r] + (L > H ? L - H : 0); }
            cur = w;
        }
        out[i] = dir[w] + pre[rows[i] & 63];
    }
}
static inline int64_t eval_meas(const qplan_t *q, const uint32_t *fv, const uint32_t *av) {
    int64_t st[24]; int sp = 0;
    for (int i = 0; i < q->nmeas; i++) {
        const minstr_t *I = &q->meas[i];
        switch (I->op) {
        case M_FCOL: st[sp++] = fv[I->arg]; break;
        case M_ATTR: st[sp++] = av[I->arg]; break;
        case M_CONST: st[sp++] = I->c; break;
        case M_ADD: sp--; st[sp - 1] += st[sp]; break;
        case M_SUB: sp--; st[sp - 1] -= st[sp]; break;
        default: sp--; st[sp - 1] *= st[sp]; break;
        }
    }
    return sp ? st[0] : 0;
}
/* plane weight: bit position of plane j of column c */
static inline int pw(const atable_t *t, int c, int j) { return t->msb[c] ? t->width[c] - 1 - j : j; }
/* sum of the degree-2 measure over the rows in mask M: popcount algebra, no per-row arithmetic */
static inline int64_t poly_sum(const atable_t *t, const qplan_t *q, dec_t *D, uint64_t w, uint64_t M) {
    int64_t s = q->p_const * __builtin_popcountll(M);
    for (int i = 0; i < q->npl; i++) {
        int c = q->pl_col[i]; const uint64_t *x = decode(t, D, c, w); int64_t a = 0;
        for (int j = 0; j < t->width[c]; j++) a += (int64_t)__builtin_popcountll(x[j] & M) << pw(t, c, j);
        s += q->pl_coef[i] * a;
    }
    for (int i = 0; i < q->npb; i++) {
        int ca = q->pb_a[i], cb = q->pb_b[i];
        const uint64_t *x = decode(t, D, ca, w), *y = decode(t, D, cb, w); int64_t a = 0;
        for (int k = 0; k < t->width[cb]; k++) {
            uint64_t yk = y[k] & M; if (!yk) continue;
            int64_t r = 0; for (int j = 0; j < t->width[ca]; j++) r += (int64_t)__builtin_popcountll(x[j] & yk) << pw(t, ca, j);
            a += r << pw(t, cb, k);
        }
        s += q->pb_coef[i] * a;
    }
    return s;
}
/* ---------------------------------------------------------------- GROUP BY as one matrix product (A_MM)
 * C = M^T X over the rows (see torble.h), evaluated tile by tile. A tile is TW = 64 words (4096 rows). In a tile the
 * filter F and the key planes give each group present its row of M^T: one mask word per tile word (found by mask
 * splitting each word: the rows sharing a key are an AND of key-plane literals). The tile's block of the product,
 * C_tile[g, p] = sum_w popcount(M[g][w] & X_p[w]), is then a dense AND-popcount GEMM of (groups x 64 words) by
 * (64 words x planes), on wide registers: VPOPCNTDQ (8 words per instruction) when the CPU has it, AVX2 nibble-table
 * popcounts (4 words) otherwise, scalar POPCNT last. A group present in only a few words of the tile is evaluated
 * sparsely (only its nonzero words). C_tile is folded into SUM and COUNT per group (weights 2^j, coefficients) and
 * added to a small per-thread hash table: the output has one entry per group present, never a dense key domain. */
#include <immintrin.h>
#define TW 64
#define TSB 8
#define TS (1 << TSB)
typedef struct {
    uint64_t code[TS], nz[TS], cnt[TS]; uint8_t used[TS]; int occ[TS], nocc;
    uint64_t M[TS][TW];
    uint64_t xt[MAXCOL][32][TW];          /* decoded measure planes of the tile, plane-major: X^T tile */
    uint64_t ones[TW];
    mplane_t P[MAXMP]; int np, npair; uint32_t mcols; int64_t st[4];
} mtile_t;
typedef void (*mmrow_fn)(const uint64_t *m, const uint64_t *const *xa, const uint64_t *const *xb, int np, uint64_t *out);
static void mm_row_scalar(const uint64_t *m, const uint64_t *const *xa, const uint64_t *const *xb, int np, uint64_t *out) {
    for (int p = 0; p < np; p++) { const uint64_t *a = xa[p], *b = xb[p]; uint64_t s = 0;
        for (int i = 0; i < TW; i++) s += __builtin_popcountll(m[i] & a[i] & b[i]);
        out[p] = s; }
}
__attribute__((target("avx2"))) static void mm_row_avx2(const uint64_t *m, const uint64_t *const *xa, const uint64_t *const *xb, int np, uint64_t *out) {
    const __m256i lut = _mm256_setr_epi8(0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4, 0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4);
    const __m256i lo = _mm256_set1_epi8(0x0f), z = _mm256_setzero_si256();
    for (int p = 0; p < np; p++) {
        const uint64_t *a = xa[p], *b = xb[p]; __m256i acc = z;   /* byte counters: 16 steps x 8 <= 128, no overflow */
        for (int i = 0; i < TW; i += 4) {
            __m256i v = _mm256_and_si256(_mm256_loadu_si256((const __m256i *)(m + i)),
                        _mm256_and_si256(_mm256_loadu_si256((const __m256i *)(a + i)), _mm256_loadu_si256((const __m256i *)(b + i))));
            acc = _mm256_add_epi8(acc, _mm256_add_epi8(_mm256_shuffle_epi8(lut, _mm256_and_si256(v, lo)),
                                                       _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16(v, 4), lo))));
        }
        __m256i s = _mm256_sad_epu8(acc, z);
        out[p] = (uint64_t)_mm256_extract_epi64(s, 0) + (uint64_t)_mm256_extract_epi64(s, 1) + (uint64_t)_mm256_extract_epi64(s, 2) + (uint64_t)_mm256_extract_epi64(s, 3);
    }
}
__attribute__((target("avx512f,avx512vpopcntdq"))) static void mm_row_vpop(const uint64_t *m, const uint64_t *const *xa, const uint64_t *const *xb, int np, uint64_t *out) {
    __m512i m0 = _mm512_loadu_si512(m), m1 = _mm512_loadu_si512(m + 8), m2 = _mm512_loadu_si512(m + 16), m3 = _mm512_loadu_si512(m + 24),
            m4 = _mm512_loadu_si512(m + 32), m5 = _mm512_loadu_si512(m + 40), m6 = _mm512_loadu_si512(m + 48), m7 = _mm512_loadu_si512(m + 56);
#define VP(k, mk) _mm512_popcnt_epi64(_mm512_and_si512(mk, _mm512_and_si512(_mm512_loadu_si512(a + 8 * k), _mm512_loadu_si512(b + 8 * k))))
    for (int p = 0; p < np; p++) {                       /* the group's 64 mask words stay in 8 registers */
        const uint64_t *a = xa[p], *b = xb[p];
        __m512i acc = _mm512_add_epi64(_mm512_add_epi64(_mm512_add_epi64(VP(0, m0), VP(1, m1)), _mm512_add_epi64(VP(2, m2), VP(3, m3))),
                                       _mm512_add_epi64(_mm512_add_epi64(VP(4, m4), VP(5, m5)), _mm512_add_epi64(VP(6, m6), VP(7, m7))));
        out[p] = (uint64_t)_mm512_reduce_add_epi64(acc);
    }
#undef VP
}
static mmrow_fn MMROW; static const char *MMNAME = "unset"; static int MMSPARSE = 12;
static void mm_init(void) {
    if (MMROW) return;
    const char *e = getenv("TORBLES_POPCNT"), *sp = getenv("TORBLES_MM_SPARSE");
    __builtin_cpu_init();
    int vpop = __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512vpopcntdq"), avx2 = __builtin_cpu_supports("avx2");
    if (e && !strcmp(e, "scalar")) vpop = avx2 = 0;
    if (e && !strcmp(e, "avx2")) vpop = 0;
    if (sp) MMSPARSE = atoi(sp);
    if (vpop) { MMNAME = "AVX-512 VPOPCNTDQ"; MMSPARSE = sp ? MMSPARSE : 6; MMROW = mm_row_vpop; }
    else if (avx2) { MMNAME = "AVX2 (vpshufb popcount)"; MMROW = mm_row_avx2; }
    else { MMNAME = "scalar POPCNT"; MMSPARSE = 64; MMROW = mm_row_scalar; }
}
const char *at_mm_kernel(void) { mm_init(); return MMNAME; }

/* per-thread tiles, allocated and cleared once: every flush leaves its slots, masks and counts cleared again */
static mtile_t *MTILES[64]; static pthread_mutex_t MTILE_MU = PTHREAD_MUTEX_INITIALIZER;
static mtile_t *mm_tile(int i) {
    pthread_mutex_lock(&MTILE_MU);
    if (!MTILES[i]) {
        mtile_t *T = aligned_alloc(64, (sizeof(mtile_t) + 63) & ~(size_t)63);
        memset(T->used, 0, sizeof T->used); T->nocc = 0;
        memset(T->nz, 0, sizeof T->nz); memset(T->cnt, 0, sizeof T->cnt); memset(T->M, 0, sizeof T->M);
        for (int k = 0; k < TW; k++) T->ones[k] = ~0ULL;
        MTILES[i] = T;
    }
    pthread_mutex_unlock(&MTILE_MU);
    return MTILES[i];
}
static void mm_prepare(const atable_t *t, const qplan_t *q, mtile_t *T) {
    T->np = tb_mm_planes(t->width, t->msb, q, T->P); T->mcols = 0;
    T->npair = 0;
    for (int p = 0; p < T->np; p++) { T->mcols |= 1u << T->P[p].ca; if (T->P[p].cb >= 0) { T->mcols |= 1u << T->P[p].cb; T->npair++; } }
    memset(T->st, 0, sizeof T->st);
}
/* fold the tile: decode the measure planes of the tile's words (only words holding rows of some group), the product
 * C_tile, then SUM/COUNT per group into the thread's table */
static void mm_flush(const atable_t *t, const qplan_t *q, mtile_t *T, uint64_t wb, uint64_t nwt, touch_t *tl, int64_t *total) {
    if (!T->nocc) return;
    T->st[3]++;
    uint64_t U = 0; for (int i = 0; i < T->nocc; i++) U |= T->nz[T->occ[i]];
    uint32_t mc = T->mcols;
    while (mc) {
        int c = __builtin_ctz(mc); mc &= mc - 1; int b = t->width[c];
        if (t->vflag[c]) {                                       /* self-extending column: from its sidecar, per word */
            uint64_t tmp[32], u = U; int f = t->vflag[c] - 1;
            for (int j = 0; j < b; j++) for (uint64_t i = 0; i < TW; i++) if (!(U >> i & 1)) T->xt[c][j][i] = 0;
            while (u) { int i = __builtin_ctzll(u); u &= u - 1;
                tb_virtual_planes(t, c, t->V[f][0][wb + i], wb + i, tmp, b);
                for (int j = 0; j < b; j++) T->xt[c][j][i] = tmp[j]; }
            continue;
        }
        if (__builtin_popcountll(U) > 24) {                   /* dense: prefix XOR across planes, vectorized over words */
            for (uint64_t i = 0; i < TW; i++) T->xt[c][0][i] = i < nwt ? t->V[c][0][wb + i] : 0;
            for (int j = 1; j < b; j++) { const uint64_t *v = t->V[c][j] + wb; uint64_t *o = T->xt[c][j], *pr = T->xt[c][j - 1];
                for (uint64_t i = 0; i < TW; i++) o[i] = pr[i] ^ (i < nwt ? v[i] : 0); }
        } else {
            for (int j = 0; j < b; j++) for (uint64_t i = 0; i < TW; i++) T->xt[c][j][i] = 0;   /* words without rows: zero */
            uint64_t u = U;
            while (u) { int i = __builtin_ctzll(u); u &= u - 1; uint64_t x = 0;
                for (int j = 0; j < b; j++) { x ^= t->V[c][j][wb + i]; T->xt[c][j][i] = x; } }
        }
    }
    const uint64_t *xa[MAXMP], *xb[MAXMP]; uint64_t C[MAXMP];
    for (int p = 0; p < T->np; p++) { xa[p] = T->xt[T->P[p].ca][T->P[p].ja]; xb[p] = T->P[p].cb >= 0 ? T->xt[T->P[p].cb][T->P[p].jb] : T->ones; }
    const uint64_t *pa[MAXMP], *pb[MAXMP]; int act[MAXMP];
    for (int k = 0; k < T->nocc; k++) {
        int s = T->occ[k]; uint64_t nzm = T->nz[s];
        if (__builtin_popcountll(nzm) > MMSPARSE) {
            if (T->npair) {
                /* products of plane pairs: drop the columns of M^T X whose factor planes are zero on this group
                 * (a key that fixes y fixes its planes: half of the x_j y_k vanish), the rest as one dense product */
                uint32_t live[MAXCOL]; uint32_t mc = T->mcols;
                while (mc) { int c = __builtin_ctz(mc); mc &= mc - 1; live[c] = 0;
                    for (int j = 0; j < t->width[c]; j++) { uint64_t o = 0, u = nzm;
                        while (u) { int i = __builtin_ctzll(u); u &= u - 1; o |= T->M[s][i] & T->xt[c][j][i]; }
                        if (o) live[c] |= 1u << j; } }
                int na = 0;
                for (int p = 0; p < T->np; p++) { const mplane_t *P = &T->P[p];
                    if (!(live[P->ca] >> P->ja & 1) || (P->cb >= 0 && !(live[P->cb] >> P->jb & 1))) continue;
                    act[na] = p; pa[na] = xa[p]; pb[na] = xb[p]; na++; }
                uint64_t C2[MAXMP]; MMROW(T->M[s], pa, pb, na, C2);
                memset(C, 0, (size_t)T->np * 8); for (int i = 0; i < na; i++) C[act[i]] = C2[i];
            } else MMROW(T->M[s], xa, xb, T->np, C);
            T->st[0]++;
        }
        else {                                                 /* sparse: only the group's nonzero words */
            memset(C, 0, (size_t)T->np * 8); T->st[1]++;
            while (nzm) { int i = __builtin_ctzll(nzm); nzm &= nzm - 1; uint64_t m = T->M[s][i];
                for (int p = 0; p < T->np; p++) C[p] += __builtin_popcountll(m & xa[p][i] & xb[p][i]); }
        }
        int64_t v = q->p_const * (int64_t)T->cnt[s];
        for (int p = 0; p < T->np; p++) v += T->P[p].w * (int64_t)C[p];
        hadd_n(tl, T->code[s], v, T->cnt[s]); *total += v;
        nzm = T->nz[s]; while (nzm) { int i = __builtin_ctzll(nzm); nzm &= nzm - 1; T->M[s][i] = 0; }
        T->nz[s] = 0; T->cnt[s] = 0; T->used[s] = 0;
    }
    T->nocc = 0;
}
/* the groups of one word: split F into the masks of its distinct keys, each placed in its tile slot (row of M^T) */
static inline void mm_word(const atable_t *t, const qplan_t *q, dec_t *D, uint64_t w, int wi, uint64_t F, mtile_t *T, touch_t *tl, int64_t *total) {
    uint32_t kl[6][64]; int slot[6], ns = 0;
    for (int i = 0; i < q->nkey; i++) if (q->key_kind[i] == 1) {        /* looked-up keys: one gather per row */
        int a = q->key_arg[i], sl = slot[i] = ns++, c = q->attr_fkcol[a]; const uint64_t *x = decode(t, D, c, w); uint64_t G = F;
        while (G) { int r = __builtin_ctzll(G); G &= G - 1;
            uint32_t fk = KM(q->attr_kmask[a], extract_col(t, x, c, r));
            if (fk >= q->attr_n[a]) F &= ~(1ULL << r); else kl[sl][r] = q->attr_vals[a][fk]; }
    }
    const uint64_t *kp[6][32]; int kb[6], km[6];
    for (int i = 0; i < q->nkey; i++) {
        int c = q->key_arg[i]; const uint64_t *x; kb[i] = 0;
        if (q->key_kind[i] == 1) continue;
        if (q->key_kind[i] == 3) { uint32_t mk = q->key_mask[i]; x = decode_n(t, D, c, w, 32 - __builtin_clz(mk)); km[i] = 1;
            while (mk) { int j = __builtin_ctz(mk); mk &= mk - 1; kp[i][kb[i]++] = &x[j]; } }
        else { if (q->key_kind[i] == 2) { kb[i] = q->key_nbits[i]; x = decode_n(t, D, c, w, kb[i]); km[i] = 1; }
               else { kb[i] = t->width[c]; x = decode(t, D, c, w); km[i] = t->msb[c]; }
               for (int j = 0; j < kb[i]; j++) kp[i][j] = &x[j]; }
    }
    uint64_t R = F;
    while (R) {
        int r = __builtin_ctzll(R); uint64_t M = R, key = 0;
        for (int i = 0; i < q->nkey; i++) {
            uint64_t kv = 0;
            if (q->key_kind[i] == 1) { kv = kl[slot[i]][r]; uint64_t G = M;
                while (G) { int rr = __builtin_ctzll(G); G &= G - 1; if (kl[slot[i]][rr] != kv) M &= ~(1ULL << rr); } }
            else for (int j = 0; j < kb[i]; j++) { uint64_t pl = *kp[i][j], bit = (pl >> r) & 1; M &= bit ? pl : ~pl; kv |= bit << (km[i] ? kb[i] - 1 - j : j); }
            key = key * q->key_radix[i] + kv;
        }
        R &= ~M;
        uint32_t h = (uint32_t)((key * 0x9E3779B97F4A7C15ULL) >> (64 - TSB));    /* tile slot: open addressing */
        while (T->used[h] && T->code[h] != key) h = (h + 1) & (TS - 1);
        if (!T->used[h]) {
            if (T->nocc >= TS * 3 / 4) {                       /* tile table full: this pair is evaluated directly */
                int64_t v = poly_sum(t, q, D, w, M); hadd_n(tl, key, v, __builtin_popcountll(M)); *total += v; T->st[2]++; continue;
            }
            T->used[h] = 1; T->code[h] = key; T->occ[T->nocc++] = h;
        }
        T->M[h][wi] |= M; T->nz[h] |= 1ULL << wi; T->cnt[h] += __builtin_popcountll(M);
    }
}
static void at_query_part(const atable_t *t, const qplan_t *q, uint64_t w0, uint64_t w1,
                          int64_t *osum, uint64_t *ocnt, int64_t *ptotal, uint64_t *pcnt, touch_t *tl, mtile_t *T) {
    dec_t D; int64_t total = 0; uint64_t cnt = 0;
    uint64_t needf = 0;                                  /* fact columns needed as integers (general path) */
    for (int i = 0; i < q->nmeas; i++) if (q->meas[i].op == M_FCOL) needf |= 1ULL << q->meas[i].arg;
    for (int i = 0; i < q->nkey; i++) if (q->key_kind[i] == 0) needf |= 1ULL << q->key_arg[i];
    for (int i = 0; i < q->nattr; i++) needf |= 1ULL << q->attr_fkcol[i];
    uint32_t fv[MAXCOL], av[8];
    for (uint64_t w = w0; w < w1; w++) {
        if (T && w > w0 && (w - w0) % TW == 0) mm_flush(t, q, T, w - TW, TW, tl, &total);   /* tile boundary */
        DEC_RESET(D);
        uint64_t F = eval_prog(t, q->prog, q->nprog, &D, w) & rowmask(t, w);
        if (q->xmode) F &= q->xmode == 1 ? q->xrows[w] : ~q->xrows[w];
        for (int s = 0; s < q->nsemi && F; s++) {       /* semi-joins: keep rows whose dimension row passed */
            const uint64_t *x = decode(t, &D, q->semi_col[s], w); int b = t->width[q->semi_col[s]];
            const uint64_t *bm = q->semi_bm[s];
            uint64_t G = F;
            if (__builtin_popcountll(F) > 12) {
                uint64_t A[64]; memset(A, 0, sizeof A); for (int j = 0; j < b; j++) A[j] = x[j]; transpose64(A);
                int ms = t->msb[q->semi_col[s]];
                while (G) { int r = __builtin_ctzll(G); G &= G - 1; uint64_t fk = KM(q->semi_kmask[s], ms ? revb((uint32_t)A[r], b) : A[r]); if (fk >= q->semi_n[s] || !(bm[fk >> 6] >> (fk & 63) & 1)) F &= ~(1ULL << r); }
            } else {
                while (G) { int r = __builtin_ctzll(G); G &= G - 1; uint32_t fk = KM(q->semi_kmask[s], extract_col(t, x, q->semi_col[s], r)); if (fk >= q->semi_n[s] || !(bm[fk >> 6] >> (fk & 63) & 1)) F &= ~(1ULL << r); }
            }
        }
        if (!F) continue;
        cnt += __builtin_popcountll(F);
        if (q->agg == A_COUNT) continue;
        if (T) { mm_word(t, q, &D, w, (int)((w - w0) % TW), F, T, tl, &total); continue; }
        if (q->agg == A_LIN) {                           /* SUM = sum_j 2^j popcount(x_j & F) */
            for (int i = 0; i < q->nlin; i++) {
                const uint64_t *x = decode(t, &D, q->lin_col[i], w); int64_t s = 0;
                for (int j = 0; j < t->width[q->lin_col[i]]; j++) s += (int64_t)__builtin_popcountll(x[j] & F) << j;
                total += q->lin_coef[i] * s;
            }
            continue;
        }
        if (q->agg == A_POLYG) {
            /* GROUP BY by mask splitting: the rows sharing the first row's key are an AND of key-plane literals
             * (a monomial); the measure over them is popcount algebra. Remove them and repeat. On a table
             * clustered by the key this is almost always a single step per 64-row word. */
            const uint64_t *kp[6][32]; int kb[6], km[6];
            for (int i = 0; i < q->nkey; i++) {
                int c = q->key_arg[i]; const uint64_t *x;
                if (q->key_kind[i] == 3) {
                    uint32_t mk = q->key_mask[i]; x = decode_n(t, &D, c, w, 32 - __builtin_clz(mk)); kb[i] = 0; km[i] = 1;
                    while (mk) { int j = __builtin_ctz(mk); mk &= mk - 1; kp[i][kb[i]++] = &x[j]; }
                } else {
                    if (q->key_kind[i] == 2) { kb[i] = q->key_nbits[i]; x = decode_n(t, &D, c, w, kb[i]); km[i] = 1; }
                    else { kb[i] = t->width[c]; x = decode(t, &D, c, w); km[i] = t->msb[c]; }
                    for (int j = 0; j < kb[i]; j++) kp[i][j] = &x[j];
                }
            }
            uint64_t R = F;
            while (R) {
                int r = __builtin_ctzll(R); uint64_t M = R, key = 0;
                for (int i = 0; i < q->nkey; i++) {
                    uint64_t kv = 0;
                    for (int j = 0; j < kb[i]; j++) {
                        uint64_t pl = *kp[i][j], bit = (pl >> r) & 1; M &= bit ? pl : ~pl;
                        kv |= bit << (km[i] ? kb[i] - 1 - j : j);
                    }
                    key = key * q->key_radix[i] + kv;
                }
                int64_t v = poly_sum(t, q, &D, w, M); uint64_t n = __builtin_popcountll(M);
                if (tl) hadd_n(tl, key, v, n); else { osum[key] += v; ocnt[key] += n; }
                total += v; R &= ~M;
            }
            continue;
        }
        if (q->agg == A_DOM) {
            /* dominance sums for a range / inequality join: slice k holds the rows with t_(k-1) <= x < t_k, the
             * difference of two comparator polynomials [x < t] on x's planes (sum_j (1 + x_j) t_j E_j); the measure
             * over each slice is popcount algebra. D(t_k) is the prefix sum of the slices (done by the caller). */
            const uint64_t *x = decode(t, &D, q->dom_col, w); uint64_t below = 0;
            for (int k = 0; k < q->ndom; k++) {
                uint64_t lt = cmp_any(t, x, q->dom_col, P_LT, q->dom_t[k]) & F, M = lt & ~below; below = lt;
                if (!M) continue;
                int64_t v = poly_sum(t, q, &D, w, M); uint64_t n = __builtin_popcountll(M);
                if (tl) hadd_n(tl, k, v, n); else { osum[k] += v; ocnt[k] += n; }
            }
            total += poly_sum(t, q, &D, w, F);
            continue;
        }
        if (q->agg == A_BIL) {                           /* SUM(x*y) = sum_jk 2^(j+k) popcount(x_j & y_k & F) */
            const uint64_t *x = decode(t, &D, q->bil_a, w), *y = decode(t, &D, q->bil_b, w);
            int bx = t->width[q->bil_a], by = t->width[q->bil_b];
            for (int j = 0; j < bx; j++) { uint64_t xf = x[j] & F; if (!xf) continue;
                int64_t s = 0; for (int k = 0; k < by; k++) s += (int64_t)__builtin_popcountll(xf & y[k]) << k;
                total += s << j; }
            continue;
        }
        /* general path: integers for surviving rows only */
        uint64_t G = F; int dense = __builtin_popcountll(F) > 12;
        uint64_t TA[MAXCOL][64];
        if (dense) for (int c = 0; c < t->k; c++) if (needf >> c & 1) {
            const uint64_t *x = decode(t, &D, c, w); memset(TA[c], 0, sizeof TA[c]);
            for (int j = 0; j < t->width[c]; j++) TA[c][j] = x[j];
            transpose64(TA[c]);
            if (t->msb[c]) for (int r = 0; r < 64; r++) TA[c][r] = revb((uint32_t)TA[c][r], t->width[c]);
        }
        while (G) {
            int r = __builtin_ctzll(G); G &= G - 1;
            for (int c = 0; c < t->k; c++) if (needf >> c & 1) fv[c] = dense ? (uint32_t)TA[c][r] : extract_col(t, decode(t, &D, c, w), c, r);
            int orphan = 0;
            for (int a = 0; a < q->nattr; a++) { uint32_t fk = KM(q->attr_kmask[a], fv[q->attr_fkcol[a]]); if (fk >= q->attr_n[a]) { orphan = 1; break; } av[a] = q->attr_vals[a][fk]; }
            if (orphan) { cnt--; continue; }                 /* foreign key outside the dimension: no join partner */
            int64_t m = q->nmeas ? eval_meas(q, fv, av) : 0;
            uint64_t key = 0;
            for (int i = 0; i < q->nkey; i++) {
                uint64_t kv;
                if (q->key_kind[i] == 3) {               /* coordinate projection: read the field's planes */
                    uint32_t mk = q->key_mask[i]; const uint64_t *x = decode_n(t, &D, q->key_arg[i], w, 32 - __builtin_clz(mk)); kv = 0;
                    while (mk) { int j = __builtin_ctz(mk); mk &= mk - 1; kv = (kv << 1) | ((x[j] >> r) & 1); }
                } else kv = q->key_kind[i] == 0 ? fv[q->key_arg[i]] : q->key_kind[i] == 1 ? av[q->key_arg[i]]
                            : extract_msb(decode_n(t, &D, q->key_arg[i], w, q->key_nbits[i]), q->key_nbits[i], r);   /* field read from the fk planes */
                key = key * q->key_radix[i] + kv;
            }
            if (q->nkey) { if (tl) hadd(tl, key, m); else { osum[key] += m; ocnt[key]++; } }
            total += m;
        }
    }
    if (T && w1 > w0) { uint64_t wb = w0 + (w1 - 1 - w0) / TW * TW; mm_flush(t, q, T, wb, w1 - wb, tl, &total); }
    *ptotal += total; *pcnt += cnt;
}

/* ---------------------------------------------------------------- plain columnar baseline: same plans */
typedef struct { uint64_t n; int32_t k, width[MAXCOL]; const uint32_t *col[MAXCOL]; } ptable_t;
ptable_t *pt_create(uint64_t n, int ncols, const uint32_t **cols) {     /* references caller's arrays (no copy) */
    ptable_t *t = calloc(1, sizeof *t); t->n = n; t->k = ncols;
    for (int c = 0; c < ncols; c++) t->col[c] = cols[c];
    col_widths(n, ncols, cols, t->width);
    return t;
}

/* ---------------------------------------------------------------- ingestion helpers (native, all cores)
 * tb_map: out[i] = lut[in[i]] (in[i] < lutn, else 0): foreign keys rewritten through a key map (derived surrogates).
 * tb_date_surrogate: yyyymmdd -> (y - 1992) << 9 | m << 5 | d.
 * tb_refok: bit `bit` of ok[i] = the foreign key points at an existing row: k = v & mask (0 = no mask), k < dim_n,
 *   used[k], and (derived surrogate) s_of_row[k] == v. Returns the rows whose surrogate disagrees (k in range, used). */
typedef struct { const uint32_t *in; uint64_t n; const int64_t *lut; uint64_t lutn; uint32_t *out; } mapctx_t;
static void map_range(void *v, uint64_t a, uint64_t b) {
    const mapctx_t *m = v;
    for (uint64_t i = a; i < b; i++) { uint32_t x = m->in[i]; m->out[i] = x < m->lutn ? (uint32_t)m->lut[x] : 0; }
}
void tb_map(const uint32_t *in, uint64_t n, const int64_t *lut, uint64_t lutn, uint32_t *out) {
    mapctx_t m = {in, n, lut, lutn, out}; par_for(n, 1 << 18, map_range, &m);
}
typedef struct { const uint32_t *in; uint32_t *out; } dsctx_t;
static void ds_range(void *v, uint64_t a, uint64_t b) {
    const dsctx_t *d = v;
    for (uint64_t i = a; i < b; i++) { uint32_t k = d->in[i]; d->out[i] = ((k / 10000 - 1992) << 9) | ((k / 100 % 100) << 5) | (k % 100); }
}
void tb_date_surrogate(const uint32_t *in, uint64_t n, uint32_t *out) { dsctx_t d = {in, out}; par_for(n, 1 << 18, ds_range, &d); }
typedef struct { const uint32_t *v; uint32_t mask; uint64_t dim_n; const uint64_t *used; const int64_t *s_of_row; uint32_t bit; uint32_t *ok; _Atomic uint64_t bad; } rfctx_t;
static void refok_range(void *vv, uint64_t a, uint64_t b) {
    rfctx_t *c = vv; uint64_t bad = 0;
    for (uint64_t i = a; i < b; i++) {
        uint32_t v = c->v[i], k = c->mask ? v & c->mask : v; uint32_t ok = 0;
        if (k < c->dim_n && (c->used[k >> 6] >> (k & 63) & 1)) {
            if (!c->s_of_row || (uint32_t)c->s_of_row[k] == v) ok = 1; else bad++;
        }
        c->ok[i] |= ok << c->bit;
    }
    if (bad) atomic_fetch_add(&c->bad, bad);
}
uint64_t tb_refok(const uint32_t *v, uint64_t n, uint32_t mask, uint64_t dim_n, const uint64_t *used, const int64_t *s_of_row, uint32_t bit, uint32_t *ok) {
    rfctx_t c = {v, mask, dim_n, used, s_of_row, bit, ok, 0}; par_for(n, 1 << 18, refok_range, &c); return atomic_load(&c.bad);
}
enum { CH = 2048 };
static void eval_prog_chunk(const ptable_t *t, const pinstr_t *prog, int n, uint64_t r0, int len, uint8_t *out) {
    static __thread uint8_t st[48][CH]; int sp = 0;
    for (int i = 0; i < n; i++) {
        const pinstr_t *I = &prog[i];
        const uint32_t *x = (I->op >= P_EQ && I->op <= P_BETWEEN) || I->op == P_PREFIX || I->op == P_BIT || I->op == P_CUBE || I->op == P_CLT || I->op == P_CGE ? t->col[I->col] + r0 : 0; uint8_t *d = st[sp];
        switch (I->op) {
        case P_TRUE: memset(d, 1, len); sp++; break;
        case P_EQ: for (int r = 0; r < len; r++) d[r] = x[r] == I->a; sp++; break;
        case P_NE: for (int r = 0; r < len; r++) d[r] = x[r] != I->a; sp++; break;
        case P_LT: for (int r = 0; r < len; r++) d[r] = x[r] < I->a; sp++; break;
        case P_LE: for (int r = 0; r < len; r++) d[r] = x[r] <= I->a; sp++; break;
        case P_GT: for (int r = 0; r < len; r++) d[r] = x[r] > I->a; sp++; break;
        case P_GE: for (int r = 0; r < len; r++) d[r] = x[r] >= I->a; sp++; break;
        case P_BETWEEN: for (int r = 0; r < len; r++) d[r] = (x[r] >= I->a) & (x[r] <= I->b); sp++; break;
        case P_PREFIX: { int sh = t->width[I->col] - (int)I->b; for (int r = 0; r < len; r++) d[r] = (x[r] >> sh) == I->a; sp++; break; }
        case P_BIT: for (int r = 0; r < len; r++) d[r] = (x[r] >> I->a) & 1; sp++; break;
        case P_CUBE: { uint32_t im = 0, iv = 0, mk = I->b; int wd = t->width[I->col];
            while (mk) { int j = __builtin_ctz(mk); mk &= mk - 1; im |= 1u << (wd - 1 - j); if (I->a >> j & 1) iv |= 1u << (wd - 1 - j); }
            for (int r = 0; r < len; r++) d[r] = (x[r] & im) == iv; sp++; break; }
        case P_CLT: case P_CGE: { uint32_t im = 0, mk = I->b; int wd = t->width[I->col];   /* code = the masked value bits */
            while (mk) { int j = __builtin_ctz(mk); mk &= mk - 1; im |= 1u << (wd - 1 - j); }
            for (int r = 0; r < len; r++) { uint32_t v = x[r] & im, c = 0; uint32_t m = im;
                while (m) { int hb = 31 - __builtin_clz(m); m &= ~(1u << hb); c = (c << 1) | (v >> hb & 1); }
                d[r] = I->op == P_CLT ? c < I->a : c >= I->a; }
            sp++; break; }
        case P_AND: sp--; for (int r = 0; r < len; r++) st[sp - 1][r] &= st[sp][r]; break;
        case P_OR: sp--; for (int r = 0; r < len; r++) st[sp - 1][r] |= st[sp][r]; break;
        case P_XOR: sp--; for (int r = 0; r < len; r++) st[sp - 1][r] ^= st[sp][r]; break;
        default: for (int r = 0; r < len; r++) st[sp - 1][r] ^= 1; break;
        }
    }
    if (sp) memcpy(out, st[0], len); else memset(out, 1, len);
}
uint64_t pt_filter_bitmap(const ptable_t *t, const pinstr_t *prog, int n, uint64_t *out) {
    uint8_t m[CH]; uint64_t cnt = 0; memset(out, 0, ((t->n + 63) / 64) * 8);
    for (uint64_t r0 = 0; r0 < t->n; r0 += CH) {
        int len = (int)(t->n - r0 < CH ? t->n - r0 : CH);
        eval_prog_chunk(t, prog, n, r0, len, m);
        for (int r = 0; r < len; r++) if (m[r]) { out[(r0 + r) >> 6] |= 1ULL << ((r0 + r) & 63); cnt++; }
    }
    return cnt;
}
static void pt_query_part(const ptable_t *t, const qplan_t *q, uint64_t ra, uint64_t rb,
                          int64_t *osum, uint64_t *ocnt, int64_t *ptotal, uint64_t *pcnt, touch_t *tl) {
    uint8_t m[CH]; int64_t total = 0; uint64_t cnt = 0; uint32_t fv[MAXCOL], av[8];
    for (uint64_t r0 = ra; r0 < rb; r0 += CH) {
        int len = (int)(rb - r0 < CH ? rb - r0 : CH);
        eval_prog_chunk(t, q->prog, q->nprog, r0, len, m);
        if (q->xmode) for (int r = 0; r < len; r++) { int bit = (int)(q->xrows[(r0 + r) >> 6] >> ((r0 + r) & 63) & 1); m[r] &= q->xmode == 1 ? bit : !bit; }
        for (int s = 0; s < q->nsemi; s++) {
            const uint32_t *fk = t->col[q->semi_col[s]] + r0; const uint64_t *bm = q->semi_bm[s];
            uint32_t km = q->semi_kmask[s];
            for (int r = 0; r < len; r++) if (m[r]) { uint32_t v = KM(km, fk[r]); m[r] = v < q->semi_n[s] && (bm[v >> 6] >> (v & 63) & 1); }
        }
        if (q->agg == A_LIN) {
            for (int i = 0; i < q->nlin; i++) { const uint32_t *x = t->col[q->lin_col[i]] + r0; int64_t s = 0;
                for (int r = 0; r < len; r++) s += m[r] ? x[r] : 0; total += q->lin_coef[i] * s; }
            for (int r = 0; r < len; r++) cnt += m[r];
            continue;
        }
        if (q->agg == A_BIL) {
            const uint32_t *x = t->col[q->bil_a] + r0, *y = t->col[q->bil_b] + r0; int64_t s = 0;
            for (int r = 0; r < len; r++) { s += m[r] ? (int64_t)x[r] * y[r] : 0; cnt += m[r]; }
            total += s; continue;
        }
        if (q->agg == A_DOM) {                           /* plain engine: the same slices, row by row */
            const uint32_t *x = t->col[q->dom_col] + r0;
            for (int r = 0; r < len; r++) {
                if (!m[r]) continue;
                cnt++;
                int64_t v = q->p_const;
                for (int i = 0; i < q->npl; i++) v += q->pl_coef[i] * (int64_t)t->col[q->pl_col[i]][r0 + r];
                for (int i = 0; i < q->npb; i++) v += q->pb_coef[i] * (int64_t)t->col[q->pb_a[i]][r0 + r] * t->col[q->pb_b[i]][r0 + r];
                total += v;
                int k = 0; while (k < q->ndom && x[r] >= q->dom_t[k]) k++;
                if (k < q->ndom) { if (tl) hadd(tl, k, v); else { osum[k] += v; ocnt[k]++; } }
            }
            continue;
        }
        if (q->agg == A_POLYG) {
            for (int r = 0; r < len; r++) {
                if (!m[r]) continue;
                cnt++;
                int64_t v = q->p_const; uint64_t key = 0;
                for (int i = 0; i < q->npl; i++) v += q->pl_coef[i] * (int64_t)t->col[q->pl_col[i]][r0 + r];
                for (int i = 0; i < q->npb; i++) v += q->pb_coef[i] * (int64_t)t->col[q->pb_a[i]][r0 + r] * t->col[q->pb_b[i]][r0 + r];
                for (int i = 0; i < q->nkey; i++) {
                    uint32_t x = t->col[q->key_arg[i]][r0 + r];
                    uint64_t kv = x;
                    if (q->key_kind[i] == 2) kv = x >> (t->width[q->key_arg[i]] - q->key_nbits[i]);
                    else if (q->key_kind[i] == 3) { uint32_t mk = q->key_mask[i]; int wd = t->width[q->key_arg[i]]; kv = 0;
                        while (mk) { int j = __builtin_ctz(mk); mk &= mk - 1; kv = (kv << 1) | ((x >> (wd - 1 - j)) & 1); } }
                    key = key * q->key_radix[i] + kv;
                }
                if (tl) hadd(tl, key, v); else { osum[key] += v; ocnt[key]++; }
                total += v;
            }
            continue;
        }
        for (int r = 0; r < len; r++) {
            if (!m[r]) continue;
            cnt++;
            if (q->agg == A_COUNT) continue;
            for (int c = 0; c < t->k; c++) fv[c] = t->col[c][r0 + r];   /* all columns (incl. prefix-key fks) */
            int orphan = 0;
            for (int a = 0; a < q->nattr; a++) { uint32_t fk = KM(q->attr_kmask[a], fv[q->attr_fkcol[a]]); if (fk >= q->attr_n[a]) { orphan = 1; break; } av[a] = q->attr_vals[a][fk]; }
            if (orphan) { cnt--; continue; }
            int64_t mv = q->nmeas ? eval_meas(q, fv, av) : 0;
            uint64_t key = 0;
            for (int i = 0; i < q->nkey; i++) {
                uint64_t kv = q->key_kind[i] == 0 ? fv[q->key_arg[i]] : q->key_kind[i] == 1 ? av[q->key_arg[i]]
                            : q->key_kind[i] == 2 ? fv[q->key_arg[i]] >> (t->width[q->key_arg[i]] - q->key_nbits[i]) : 0;
                if (q->key_kind[i] == 3) { uint32_t mk = q->key_mask[i], x = fv[q->key_arg[i]]; int wd = t->width[q->key_arg[i]]; kv = 0;
                    while (mk) { int j = __builtin_ctz(mk); mk &= mk - 1; kv = (kv << 1) | ((x >> (wd - 1 - j)) & 1); } }
                key = key * q->key_radix[i] + kv;
            }
            if (q->nkey) { if (tl) hadd(tl, key, mv); else { osum[key] += mv; ocnt[key]++; } }
            total += mv;
        }
    }
    *ptotal += total; *pcnt += cnt;
}

/* ---------------------------------------------------------------- multithreading
 * Every 64-row word is independent, so work is handed out in chunks from an atomic counter. Each thread keeps its
 * own totals and group arrays; integer sums merge exactly, so results do not depend on the thread count. */
static int NTHREADS = 1;
static int nthreads(void) { return NTHREADS; }
void at_set_threads(int n) { NTHREADS = n < 1 ? 1 : n > 64 ? 64 : n; }
int at_get_threads(void) { return NTHREADS; }
typedef struct {
    int kind; const void *t; const qplan_t *q; const uint64_t *chunks; uint64_t nchunks, limit; _Atomic uint64_t *next;
    int64_t *osum; uint64_t *ocnt; int64_t total; uint64_t cnt; touch_t tl; int extra; mtile_t *mt;
} job_t;
static void *qworker(void *arg) {
    job_t *j = arg;
    for (;;) {
        uint64_t i = atomic_fetch_add(j->next, 1);
        if (i >= j->nchunks) break;
        uint64_t u0 = j->chunks[2 * i], u1 = j->chunks[2 * i + 1];
        touch_t *tl = j->extra ? &j->tl : NULL;
        if (j->kind == 0) at_query_part(j->t, j->q, u0, u1, j->osum, j->ocnt, &j->total, &j->cnt, tl, j->mt);
        else pt_query_part(j->t, j->q, u0 * CH, u1 * CH < j->limit ? u1 * CH : j->limit, j->osum, j->ocnt, &j->total, &j->cnt, tl);
    }
    return NULL;
}
/* split the units to scan (all, or only the given ranges) into chunks handed out by an atomic counter */
static uint64_t make_chunks(const qplan_t *q, uint64_t nunits, uint64_t chunk, uint64_t div, uint64_t **out) {
    uint64_t cap = 1024, n = 0; uint64_t *c = malloc(cap * 16);
    uint64_t nr = q->nranges ? q->nranges : 1, last = 0;
    for (uint64_t k = 0; k < nr; k++) {
        uint64_t a = q->nranges ? q->ranges[2 * k] / div : 0, b = q->nranges ? (q->ranges[2 * k + 1] + div - 1) / div : nunits;
        if (b > nunits) b = nunits;
        if (a < last) a = last;                        /* ranges are ascending; never scan a unit twice */
        if (a >= b) continue;
        last = b;
        for (uint64_t u = a; u < b; u += chunk) {
            if (n == cap) { cap *= 2; c = realloc(c, cap * 16); }
            c[2 * n] = u; c[2 * n + 1] = u + chunk < b ? u + chunk : b; n++;
        }
    }
    *out = c; return n;
}
static void run_parallel(int kind, const void *t, qplan_t *q, uint64_t nunits, uint64_t chunk, uint64_t limit, uint64_t div) {
    int T = NTHREADS; job_t jobs[64]; pthread_t th[64]; _Atomic uint64_t next = 0;
    uint64_t *chunks; uint64_t nchunks = make_chunks(q, nunits, chunk, div, &chunks);
    if ((uint64_t)T > nchunks) T = nchunks ? (int)nchunks : 1;      /* no more threads than chunks */
    int mm = kind == 0 && q->agg == A_MM;
    for (int i = 0; i < T; i++) {
        jobs[i] = (job_t){kind, t, q, chunks, nchunks, limit, &next, q->out_sum, q->out_cnt, 0, 0, {0}, 0, NULL};
        if ((i || mm) && q->nkey) jobs[i].extra = 1;
        if (mm) { jobs[i].mt = mm_tile(i); mm_prepare(t, q, jobs[i].mt); }
    }
    pool_jobs(T, qworker, jobs, sizeof jobs[0]); (void)th;
    q->total_sum = jobs[0].total; q->total_cnt = jobs[0].cnt;
    if (mm) {                                            /* sparse output: the groups present, merged across threads */
        touch_t *h0 = &jobs[0].tl; memset(q->mm_stats, 0, sizeof q->mm_stats);
        for (int i = 0; i < T; i++) {
            for (int k = 0; k < 4; k++) q->mm_stats[k] += jobs[i].mt->st[k];
            if (i) { q->total_sum += jobs[i].total; q->total_cnt += jobs[i].cnt;
                     touch_t *h = &jobs[i].tl;
                     for (uint64_t x = 0; x < h->cap; x++) if (h->key[x] != TEMPTY) hadd_n(h0, h->key[x], h->sum[x], h->cnt[x]);
                     free(h->key); free(h->sum); free(h->cnt); }
        }
        uint64_t n = 0;
        for (uint64_t x = 0; x < h0->cap; x++) if (h0->key[x] != TEMPTY) {
            if (n < q->out_cap) { q->out_key[n] = h0->key[x]; q->out_sum[n] = h0->sum[x]; q->out_cnt[n] = h0->cnt[x]; }
            n++;
        }
        q->out_n = n; free(h0->key); free(h0->sum); free(h0->cnt); free(chunks); return;
    }
    for (int i = 1; i < T; i++) {
        q->total_sum += jobs[i].total; q->total_cnt += jobs[i].cnt;
        if (q->nkey) {
            touch_t *h = &jobs[i].tl;
            for (uint64_t x = 0; x < h->cap; x++) if (h->key[x] != TEMPTY) { q->out_sum[h->key[x]] += h->sum[x]; q->out_cnt[h->key[x]] += h->cnt[x]; }
            free(h->key); free(h->sum); free(h->cnt);
        }
    }
    free(chunks);
}
int64_t at_query(const atable_t *t, qplan_t *q) {
    if (q->agg == A_MM) { static mplane_t tmp[MAXMP]; mm_init(); if (!q->nkey || tb_mm_planes(t->width, t->msb, q, tmp) < 0) return -1; }
    run_parallel(0, t, q, t->nw, 512, 0, 1); return 0; }      /* ranges in words */      /* ranges in words */
int64_t pt_query(const ptable_t *t, qplan_t *q) { run_parallel(1, t, q, (t->n + CH - 1) / CH, 16, t->n, CH / 64); return 0; }   /* same word ranges, in CH-row chunks */

/* repair single-bit faults: for each corrupted row, find the stored plane whose flip clears all its faces */
static int row_ok(const atable_t *t, uint64_t r) {
    uint64_t w = r >> 6, bit = 1ULL << (r & 63), acc = 0;
    for (int f = 0; f < t->nfaces; f++) {
        uint64_t x = 0; for (int q = 0; q < t->fterms[f]; q++) x ^= t->face[f][q][w];
        acc |= x;
    }
    return !(acc & bit);
}
uint64_t at_repair(atable_t *t) {
    uint64_t nbad = at_validate(t, NULL, 0), fixed = 0;
    if (!nbad) return 0;
    uint64_t *bad = malloc(nbad * 8); at_validate(t, bad, nbad);
    uint64_t *planes[MAXCOL * 2 * NB]; int np = 0;
    for (int c = 0; c < t->k; c++) for (int j = 0; j < NB; j++) if (t->V[c][j]) planes[np++] = t->V[c][j];
    for (int i = 0; i <= t->k; i++) for (int j = 0; j < NB; j++) if (t->H[i][j]) planes[np++] = t->H[i][j];
    for (uint64_t b = 0; b < nbad; b++) {
        uint64_t r = bad[b], w = r >> 6, m = 1ULL << (r & 63);
        for (int p = 0; p < np; p++) { planes[p][w] ^= m; if (row_ok(t, r)) { fixed++; break; } planes[p][w] ^= m; }
    }
    free(bad); return fixed;
}

/* ---------------------------------------------------------------- join polynomial planning (native)
 * The interval terms of a dimension filter over the codes that occur (see JoinPoly in algq.py): runs of selected
 * codes widen into the unused codes around them to the bounds with the most trailing zeros, and each run is costed in
 * its factored (two comparators) and expanded (aligned blocks, one cube each) form. Output, per term:
 * [form (0 factored, 1 expanded), nparts, then nparts pairs: (op 0 = CGE / 1 = CLT, K) or (block start, k)].
 * Returns the int64s written, -1 over budget, -2 if out is too small. */
static inline int64_t jp_bitlen(int64_t v) { return v ? 64 - __builtin_clzll((uint64_t)v) : 0; }
static inline int64_t jp_best_tz(int64_t lo, int64_t hi) {
    if (lo <= 0) return 0;
    for (int64_t t = jp_bitlen(hi); t >= 0; t--) { int64_t c = (hi >> t) << t; if (c >= lo) return c; }
    return hi;
}
static inline int jp_tz(int64_t v, int nb) { return v == 0 ? nb : __builtin_ctzll((uint64_t)v); }
int64_t jp_terms(const uint8_t *sel, int invert, const int64_t *codes, int64_t n, int nb, int form, double budget,
                 double c_lit, double c_instr, int64_t *out, int64_t cap, double *pcost) {
    int64_t full = (int64_t)1 << nb, nruns = 0, w = 0;
#define SEL(i) ((sel[i] != 0) ^ invert)
    for (int64_t i = 0; i < n; i++) if (SEL(i) && (i == 0 || !SEL(i - 1))) nruns++;
    if (nruns * c_instr > budget) return -1;
    double cost = c_instr * (nruns > 0 ? nruns - 1 : 0);
    for (int64_t i = 0; i < n; i++) {
        if (!SEL(i) || (i > 0 && SEL(i - 1))) continue;
        int64_t j = i; while (j + 1 < n && SEL(j + 1)) j++;
        int64_t lo_min = i > 0 ? codes[i - 1] + 1 : 0, hi_max = j + 1 < n ? codes[j + 1] : full;
        int64_t A = jp_best_tz(lo_min, codes[i]), B = jp_best_tz(codes[j] + 1, hi_max);
        int nbounds = (A > 0) + (B < full);
        double cf = 0;                                   /* same summation order as the Python planner (ties) */
        if (A > 0) cf += c_lit * (nb - jp_tz(A, nb)) + c_instr;
        if (B < full) cf += c_lit * (nb - jp_tz(B, nb)) + c_instr;
        cf += c_instr * (nbounds == 2);
        int64_t bs[130][2]; int nbl = 0; double ce = 0;
        for (int64_t cur = A; cur < B; ) {
            int64_t k = cur ? __builtin_ctzll((uint64_t)cur) : jp_bitlen(B) - 1;
            while (cur + ((int64_t)1 << k) > B) k--;
            if (nbl < 130) { bs[nbl][0] = cur; bs[nbl][1] = k; } nbl++;
            ce += c_lit * (nb - k) + c_instr; cur += (int64_t)1 << k;
        }
        ce += c_instr * (nbl - 1);
        int pick = form == 0 ? 0 : form == 1 ? 1 : (cf < ce ? 0 : 1);
        if (pick == 1 && nbl > 130) return -2;
        int np = pick ? nbl : nbounds;
        if (w + 2 + 2 * np > cap) return -2;
        out[w++] = pick; out[w++] = np;
        if (pick) for (int b = 0; b < nbl; b++) { out[w++] = bs[b][0]; out[w++] = bs[b][1]; }
        else { if (A > 0) { out[w++] = 0; out[w++] = A; } if (B < full) { out[w++] = 1; out[w++] = B; } }
        cost += pick ? ce : cf;
        if (cost > budget) return -1;
        i = j;
    }
#undef SEL
    *pcost = cost; return w;
}
