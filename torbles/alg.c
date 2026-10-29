/* Copyright 2026 Brian Richard RAMOS
 * SPDX-License-Identifier: Apache-2.0
 * Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0). */
/* torbles: the algebraic engine (CPU).
 * A torble is stored as torus bit planes: plane j of torus lane L holds bit j of L for every row (64 rows per word).
 * Column c is torus row c; with value width b the potential phi_c has bits 1..b, so only these planes can be nonzero:
 *   V_c[j] = phi_c(j) ^ phi_c(j+1),         j = 0..b_c
 *   H_i[j] = phi_i(j) ^ phi_(i+1)(j),       i < k-1;   H_(k-1)[j] = phi_(k-1)(j);   wrap H_15[j] = phi_0(j)
 * Every other plane of the torus (k + 1 rows, NB lanes) is zero and is not stored; faces through it are still checked.
 *   decode   : value bit j = XOR of V_c[0..j]                     (prefix over planes, 64 rows per op)
 *   validate : face(i,j) = H_i[j] ^ H_i[j+1] ^ V_i[j] ^ V_(i+1)[j] (XOR of planes)
 *   filter   : any predicate is a Boolean polynomial evaluated on value planes
 *   SUM(x)   : sum_j 2^j popcount(x_j & F);   SUM(x*y): sum_jk 2^(j+k) popcount(x_j & y_k & F)
 * A plain columnar engine (uint32 columns, vectorized chunks) runs the same plans for comparison.
 * Build: gcc -O2 -march=native -shared -fPIC alg.c -o libalg.so */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
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
/* A[r] bit j <- A[j] bit r: the butterfly stage by stage, each stage over contiguous blocks of words so the compiler
 * vectorizes it (about 4x faster than one loop over k with a computed stride) */
#define TB_STAGE(J, MASK) do { const uint64_t m_ = (MASK); \
    for (int b_ = 0; b_ < 64; b_ += 2 * (J)) for (int k = b_; k < b_ + (J); k++) { \
        uint64_t t_ = ((A[k] >> (J)) ^ A[k + (J)]) & m_; A[k] ^= t_ << (J); A[k + (J)] ^= t_; } } while (0)
static inline void transpose64(uint64_t *restrict A) {
    TB_STAGE(32, 0x00000000FFFFFFFFULL); TB_STAGE(16, 0x0000FFFF0000FFFFULL); TB_STAGE(8, 0x00FF00FF00FF00FFULL);
    TB_STAGE(4, 0x0F0F0F0F0F0F0F0FULL); TB_STAGE(2, 0x3333333333333333ULL); TB_STAGE(1, 0x5555555555555555ULL);
}
#undef TB_STAGE_DOC
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
/* face list: every face of the torus that touches a stored plane (which planes exist decides the faces) */
static void at_faces(atable_t *t) {
    int k = t->k; t->nfaces = 0;
    for (int ii = 0; ii <= k; ii++) {
        int hrow = ii, vrow = ii < k ? ii : -1, vnext = ii < k ? (ii + 1 < k ? ii + 1 : -1) : 0;   /* ii == k is the wrap row */
        for (int j = 0; j < NB; j++) {                 /* lanes are wide enough for 32-bit values (potentials need 33 bits) */
            uint64_t *terms[4]; int nt = 0;
            if (t->H[hrow][j]) terms[nt++] = t->H[hrow][j];
            if (j + 1 < NB && t->H[hrow][j + 1]) terms[nt++] = t->H[hrow][j + 1];
            if (vrow >= 0 && t->V[vrow][j]) terms[nt++] = t->V[vrow][j];
            if (vnext >= 0 && t->V[vnext][j]) terms[nt++] = t->V[vnext][j];
            if (!nt) continue;
            for (int q = 0; q < nt; q++) t->face[t->nfaces][q] = terms[q];
            t->fterms[t->nfaces++] = nt;
        }
    }
}
atable_t *at_create(uint64_t n, int ncols, const uint32_t **cols, const int32_t *msb) {
    if (ncols < 1 || ncols > MAXCOL) return NULL;
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
    at_faces(t);
    return t;
}
/* ---------------------------------------------------------------- persistence
 * A torble file: a header (magic, version, shape, widths, the directory of stored planes, a hash of the header), then
 * every stored plane, each 64-byte aligned. at_load maps the file privately (copy-on-write: repairs and fault
 * injection never touch the file) and points the planes into the mapping; nothing is copied or decoded. The planes
 * carry their own integrity check: the torus faces are recomputed from them, so a loaded torble is validated (and
 * single-bit faults repaired) by the same at_validate / at_repair as one just built. */
#include <stdio.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#define TB_MAGIC "TORBLE\x01\x00"
#define TB_FORMAT 1
#define TB_SAVED_COLS 64
typedef struct { char magic[8]; uint32_t format, endian; uint64_t n, nw; int32_t k, nplanes, maxcol, pad;
                 int32_t width[TB_SAVED_COLS], msb[TB_SAVED_COLS]; uint64_t hdrlen, planebytes, hash; } tbhdr_t;
static uint64_t tb_fnv(const void *p, size_t len, uint64_t h) {
    const unsigned char *c = p; for (size_t i = 0; i < len; i++) { h ^= c[i]; h *= 0x100000001b3ULL; } return h;
}
static uint64_t tb_planebytes(uint64_t nw) { return ((nw * 8) + 63) & ~(uint64_t)63; }
/* returns 0, or -1 if the file cannot be written */
int at_save(const atable_t *t, const char *path) {
    char tmp[4096]; if (snprintf(tmp, sizeof tmp, "%s.tmp", path) >= (int)sizeof tmp) return -1;
    uint8_t dir[(2 * MAXCOL + 2) * NB][4]; int np = 0;
    for (int c = 0; c < t->k; c++) for (int j = 0; j < NB; j++) if (t->V[c][j]) { dir[np][0] = 0; dir[np][1] = (uint8_t)c; dir[np][2] = (uint8_t)j; dir[np][3] = 0; np++; }
    for (int i = 0; i <= t->k; i++) for (int j = 0; j < NB; j++) if (t->H[i][j]) { dir[np][0] = 1; dir[np][1] = (uint8_t)i; dir[np][2] = (uint8_t)j; dir[np][3] = 0; np++; }
    tbhdr_t h; memset(&h, 0, sizeof h); memcpy(h.magic, TB_MAGIC, 8); h.format = TB_FORMAT; h.endian = 0x01020304u;
    h.n = t->n; h.nw = t->nw; h.k = t->k; h.nplanes = np; h.maxcol = MAXCOL;
    for (int c = 0; c < t->k; c++) { h.width[c] = t->vflag[c] ? -1 : t->width[c]; h.msb[c] = t->msb[c]; }   /* -1: virtual (restored by the caller) */
    h.planebytes = tb_planebytes(t->nw);
    h.hdrlen = (sizeof h + (uint64_t)np * 4 + 4095) & ~(uint64_t)4095;
    h.hash = tb_fnv(dir, (size_t)np * 4, tb_fnv(&h, offsetof(tbhdr_t, hash), 0xcbf29ce484222325ULL));
    FILE *f = fopen(tmp, "wb"); if (!f) return -1;
    int ok = fwrite(&h, sizeof h, 1, f) == 1 && fwrite(dir, 4, np, f) == (size_t)np;
    static const char zeros[4096];
    uint64_t at = sizeof h + (uint64_t)np * 4;
    if (ok && h.hdrlen > at) ok = fwrite(zeros, 1, h.hdrlen - at, f) == h.hdrlen - at;
    for (int p = 0; ok && p < np; p++) {
        const uint64_t *pl = dir[p][0] ? t->H[dir[p][1]][dir[p][2]] : t->V[dir[p][1]][dir[p][2]];
        ok = fwrite(pl, 8, t->nw, f) == t->nw;
        if (ok && h.planebytes > t->nw * 8) ok = fwrite(zeros, 1, h.planebytes - t->nw * 8, f) == h.planebytes - t->nw * 8;
    }
    if (fflush(f) != 0) ok = 0;
    if (ok) fsync(fileno(f));
    if (fclose(f) != 0) ok = 0;
    if (!ok || rename(tmp, path) != 0) { remove(tmp); return -1; }
    return 0;
}
/* err: 0 ok, 1 cannot open/map, 2 not a torble file, 3 format/version or build mismatch, 4 header damaged, 5 truncated */
atable_t *at_load(const char *path, int *err) {
    int fd = open(path, O_RDONLY); if (fd < 0) { *err = 1; return NULL; }
    struct stat st; if (fstat(fd, &st) != 0 || (uint64_t)st.st_size < sizeof(tbhdr_t)) { close(fd); *err = st.st_size < (off_t)sizeof(tbhdr_t) ? 5 : 1; return NULL; }
    void *m = mmap(NULL, (size_t)st.st_size, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0); close(fd);
    if (m == MAP_FAILED) { *err = 1; return NULL; }
    tbhdr_t *h = m; *err = 0;
    if (memcmp(h->magic, TB_MAGIC, 8)) *err = 2;
    else if (h->format != TB_FORMAT || h->endian != 0x01020304u || h->k < 1 || h->k > MAXCOL) *err = 3;
    else if (h->nplanes < 1 || h->nplanes > (2 * MAXCOL + 2) * NB || sizeof *h + (uint64_t)h->nplanes * 4 > h->hdrlen || h->hdrlen > (uint64_t)st.st_size) *err = 4;
    else if (tb_fnv((char *)m + sizeof *h, (size_t)h->nplanes * 4, tb_fnv(h, offsetof(tbhdr_t, hash), 0xcbf29ce484222325ULL)) != h->hash) *err = 4;
    else if (h->planebytes != tb_planebytes(h->nw) || h->nw != (h->n + 63) / 64) *err = 4;
    else if (h->hdrlen + (uint64_t)h->nplanes * h->planebytes > (uint64_t)st.st_size) *err = 5;
    if (*err) { munmap(m, (size_t)st.st_size); return NULL; }
    atable_t *t = calloc(1, sizeof *t);
    t->n = h->n; t->nw = h->nw; t->k = h->k; t->map = m; t->maplen = (uint64_t)st.st_size;
    for (int c = 0; c < t->k; c++) { t->width[c] = h->width[c] < 0 ? 1 : h->width[c]; t->msb[c] = h->msb[c]; }
    const uint8_t (*dir)[4] = (const void *)((char *)m + sizeof *h);
    for (int p = 0; p < h->nplanes; p++) {
        int kind = dir[p][0], row = dir[p][1], j = dir[p][2];
        if (j >= NB || (kind == 0 && row >= t->k) || (kind == 1 && row > t->k) || kind > 1) { *err = 4; munmap(m, (size_t)st.st_size); free(t); return NULL; }
        uint64_t *pl = (uint64_t *)((char *)m + h->hdrlen + (uint64_t)p * h->planebytes);
        if (kind) t->H[row][j] = pl; else t->V[row][j] = pl;
    }
    t->bytes = (uint64_t)h->nplanes * t->nw * 8;
    at_faces(t);
    return t;
}
/* release a torble: its planes (built) or its mapping (loaded) */
void at_free(atable_t *t) {
    if (!t) return;
    if (t->map) munmap(t->map, (size_t)t->maplen);
    else {
        for (int c = 0; c < t->k; c++) for (int j = 0; j < NB; j++) free(t->V[c][j]);
        for (int i = 0; i <= t->k; i++) for (int j = 0; j < NB; j++) free(t->H[i][j]);
    }
    free(t);
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

/* per-thread group accumulator for worker threads: small open-addressing hash, merged into the result at the end.
 * Group codes are 128-bit (a GROUP BY whose key domain passes 2^64: widekey plans); sum holds each group's sum modulo
 * 2^64, shi the high word of a 128-bit sum (wide plans) */
typedef unsigned __int128 u128;                     /* 128-bit codes and sums: modular (two's complement) arithmetic */
#define U128(hi, lo) (((u128)(uint64_t)(hi) << 64) | (uint64_t)(lo))
typedef struct { u128 *key; int64_t *sum; uint64_t *cnt; uint64_t cap, n; int64_t *shi; } touch_t;
#define TEMPTY (~(u128)0)                           /* codes stay below 2^128 - 2: the planner bounds the key domain */
static inline uint64_t thash(u128 k) {
    uint64_t x = (uint64_t)k ^ ((uint64_t)(k >> 64) * 0xC2B2AE3D27D4EB4FULL);
    return (x * 0x9E3779B97F4A7C15ULL) ^ ((x * 0x9E3779B97F4A7C15ULL) >> 29);
}
static void hgrow(touch_t *h) {
    uint64_t oc = h->cap, nc = oc ? 2 * oc : 4096; u128 *ok = h->key; int64_t *os = h->sum, *oh = h->shi; uint64_t *on = h->cnt;
    h->key = malloc(nc * sizeof(u128)); h->sum = calloc(nc, 8); h->cnt = calloc(nc, 8); h->shi = calloc(nc, 8); h->cap = nc;
    memset(h->key, 0xFF, nc * sizeof(u128));
    for (uint64_t i = 0; i < oc; i++) if (ok[i] != TEMPTY) {
        uint64_t j = thash(ok[i]) & (nc - 1);
        while (h->key[j] != TEMPTY) j = (j + 1) & (nc - 1);
        h->key[j] = ok[i]; h->sum[j] = os[i]; h->cnt[j] = on[i]; h->shi[j] = oh[i];
    }
    free(ok); free(os); free(on); free(oh);
}
static void hfree(touch_t *h) { free(h->key); free(h->sum); free(h->cnt); free(h->shi); }
static inline uint64_t hslot(touch_t *h, u128 key) {
    if (2 * (h->n + 1) > h->cap) hgrow(h);
    uint64_t j = thash(key) & (h->cap - 1);
    while (h->key[j] != key && h->key[j] != TEMPTY) j = (j + 1) & (h->cap - 1);
    if (h->key[j] == TEMPTY) { h->key[j] = key; h->n++; }
    return j;
}
/* a 128-bit sum kept as (lo, hi) words */
static inline void add128(int64_t *lo, int64_t *hi, u128 v) { u128 x = U128(*hi, *lo) + v; *lo = (int64_t)(uint64_t)x; *hi = (int64_t)(uint64_t)(x >> 64); }
static inline void hadd_n(touch_t *h, u128 key, uint64_t v, uint64_t nadd) {
    uint64_t j = hslot(h, key); h->sum[j] = (int64_t)((uint64_t)h->sum[j] + v); h->cnt[j] += nadd;
}
static inline void hadd(touch_t *h, u128 key, uint64_t v) { hadd_n(h, key, v, 1); }
static inline void hadd_w(touch_t *h, u128 key, u128 v, uint64_t nadd) {
    uint64_t j = hslot(h, key); add128(&h->sum[j], &h->shi[j], v); h->cnt[j] += nadd;
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
static inline uint64_t eval_meas(const qplan_t *q, const uint32_t *fv, const uint32_t *av) {   /* modular: exact mod 2^64 */
    uint64_t st[24]; int sp = 0;
    for (int i = 0; i < q->nmeas; i++) {
        const minstr_t *I = &q->meas[i];
        switch (I->op) {
        case M_FCOL: st[sp++] = fv[I->arg]; break;
        case M_ATTR: st[sp++] = av[I->arg]; break;
        case M_CONST: st[sp++] = (uint64_t)I->c; break;
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
static inline uint64_t poly_sum(const atable_t *t, const qplan_t *q, dec_t *D, uint64_t w, uint64_t M) {   /* modular */
    uint64_t s = (uint64_t)q->p_const * (uint64_t)__builtin_popcountll(M);
    for (int i = 0; i < q->npl; i++) {
        int c = q->pl_col[i]; const uint64_t *x = decode(t, D, c, w); uint64_t a = 0;
        for (int j = 0; j < t->width[c]; j++) a += (uint64_t)__builtin_popcountll(x[j] & M) << pw(t, c, j);
        s += (uint64_t)q->pl_coef[i] * a;
    }
    for (int i = 0; i < q->npb; i++) {
        int ca = q->pb_a[i], cb = q->pb_b[i];
        const uint64_t *x = decode(t, D, ca, w), *y = decode(t, D, cb, w); uint64_t a = 0;
        for (int k = 0; k < t->width[cb]; k++) {
            uint64_t yk = y[k] & M; if (!yk) continue;
            uint64_t r = 0; for (int j = 0; j < t->width[ca]; j++) r += (uint64_t)__builtin_popcountll(x[j] & yk) << pw(t, ca, j);
            a += r << pw(t, cb, k);
        }
        s += (uint64_t)q->pb_coef[i] * a;
    }
    return s;
}
/* the same sum with 128-bit coefficients and accumulators (wide plans). A product of two 32-bit columns is the
 * 32 x 32 matrix of plane pairs; a 64-bit value is two such columns (low and high words), so the product of two
 * 64-bit values is the 64 x 64 matrix of plane pairs, weights 2^(j+k) up to 2^126. The popcounts are exact counts;
 * only the fold is 128-bit. */
static inline u128 poly_sum_w(const atable_t *t, const qplan_t *q, dec_t *D, uint64_t w, uint64_t M) {
    u128 s = U128(q->p_const_hi, q->p_const) * (u128)__builtin_popcountll(M);
    for (int i = 0; i < q->npl; i++) {
        int c = q->pl_col[i]; const uint64_t *x = decode(t, D, c, w); uint64_t a = 0;
        for (int j = 0; j < t->width[c]; j++) a += (uint64_t)__builtin_popcountll(x[j] & M) << pw(t, c, j);
        s += U128(q->pl_coef_hi[i], q->pl_coef[i]) * (u128)a;
    }
    for (int i = 0; i < q->npb; i++) {
        int ca = q->pb_a[i], cb = q->pb_b[i];
        const uint64_t *x = decode(t, D, ca, w), *y = decode(t, D, cb, w); u128 a = 0;
        for (int k = 0; k < t->width[cb]; k++) {
            uint64_t yk = y[k] & M; if (!yk) continue;
            uint64_t r = 0; for (int j = 0; j < t->width[ca]; j++) r += (uint64_t)__builtin_popcountll(x[j] & yk) << pw(t, ca, j);
            a += (u128)r << pw(t, cb, k);
        }
        s += U128(q->pb_coef_hi[i], q->pb_coef[i]) * a;
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
    u128 code[TS]; uint64_t nz[TS], cnt[TS]; uint8_t used[TS]; int occ[TS], nocc;
    uint64_t M[TS][TW];
    uint64_t xt[MAXCOL][32][TW];          /* decoded measure planes of the tile, plane-major: X^T tile */
    uint64_t ones[TW];
    uint64_t rowv[TW * 64];                /* per-row measure values of the tile (row mode of a flush) */
    mplane_t P[MAXMP]; int np, npair; uint32_t mcols; int64_t st[4];
    u128 W[MAXMP], WC, tw;                /* wide plans: plane weights (coefficient x 2^(j+k)), constant term, total */
    /* row kernel (dense tiles): every row's key fields and measure columns packed into one 64-bit word by one bit
     * matrix transpose per 64-row word; groups accumulated in a small array indexed by group code (or the hash) */
    int rk_ok, rk_nfill, rk_off[MAXKEY], rk_bits[MAXKEY], rm_off[MAXCOL], rm_bits[MAXCOL];
    int16_t rf_col[64], rf_plane[64], rf_slot[64]; int8_t rf_key[64];
    uint64_t rk_R; int rk_dense, rk_kbits, use_rows, use_mob, tile_rows, since; double rk_ratio;
    uint64_t *asum; uint64_t *acnt; uint32_t *touched; uint32_t ntouched; u128 *asumw;   /* asumw: 128-bit sums (wide plans) */
    int kt_ok, kt_n, kt_direct; int16_t kt_col[64], kt_plane[64];   /* splitting tree: the key bits, least significant first */
    /* Möbius bridge: masks of the monomials over the key bits (positive literals only) for the tile's words, their sums
     * and counts accumulated in the monomial basis, turned into group sums by inclusion-exclusion at the end */
    int mob_ok, tile_mob, mob_dirty; uint64_t mobU; uint64_t *mob; uint64_t *msum; uint64_t *mcnt;
} mtile_t;
#define MOB_MAXBITS 10
#define RK_DENSE (1 << 16)
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
/* cost model of a tile (rough cycles), for the choice between plane popcounts and per-row values:
 * one (group, measure plane) over a dense tile row of M^T, one (word, plane) of a sparse group, a 64x64 transpose of
 * one column for one word, one term per row, one selected row gathered into its group's sum.
 * TORBLES_MM_ROWS=0 / 1 forces the plane popcounts / the per-row values. */
static double MMC_DENSE = 64, MMC_SPARSE = 1.5, MMC_TRANS = 200, MMC_TERM = 1.5, MMC_GATHER = 3, MMC_SPLIT_KEY = 24, MMC_SPLIT_BIT = 2,
              MMC_ROW = 6, MMC_MOBAND = 1.5; static int MMROWS = -1;   /* TORBLES_MM_ROWS: 0 popcounts only, 1 per-row measure in flushes, 2 row kernel */
static int MMTREE = 0;                     /* TORBLES_SPLIT: "scan" (default) or "tree": how a word is split into its groups */
static int MMDIRECT = 0;                   /* TORBLES_DIRECT=1: the group code is the tile slot when it fits (no hashing) */
static int MMMOB = 0;                      /* TORBLES_MOBIUS: 0 off (default), 1 every tile, -1 chosen by cost */
/* the row kernel replaces the popcount product only when the model puts it below RK_MARGIN of the product's cost:
 * measured on SSB SF1 (ablation in docs/ENGINE.md), the model is right about dense, many-group tiles (M2, M3: the
 * row kernel 3x faster at a model ratio of 0.33) and optimistic about the rest (M1 at 0.70, Q3.1 at 0.88: slower) */
static double RK_MARGIN = 0.5;
static void mm_init(void) {
    if (MMROW) return;
    const char *e = getenv("TORBLES_POPCNT"), *sp = getenv("TORBLES_MM_SPARSE");
    __builtin_cpu_init();
    int vpop = __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512vpopcntdq"), avx2 = __builtin_cpu_supports("avx2");
    if (e && !strcmp(e, "scalar")) vpop = avx2 = 0;
    if (e && !strcmp(e, "avx2")) vpop = 0;
    if (sp) MMSPARSE = atoi(sp);
    if (vpop) { MMNAME = "AVX-512 VPOPCNTDQ"; MMSPARSE = sp ? MMSPARSE : 6; MMROW = mm_row_vpop; MMC_DENSE = 24; }
    else if (avx2) { MMNAME = "AVX2 (vpshufb popcount)"; MMROW = mm_row_avx2; }
    else { MMNAME = "scalar POPCNT"; MMSPARSE = 64; MMROW = mm_row_scalar; MMC_DENSE = 128; }
    const char *rw = getenv("TORBLES_MM_ROWS"); if (rw && *rw) MMROWS = atoi(rw);
    const char *sp_ = getenv("TORBLES_SPLIT"); if (sp_ && *sp_) MMTREE = strcmp(sp_, "tree") == 0;
    const char *rm_ = getenv("TORBLES_RK_MARGIN"); if (rm_ && *rm_) RK_MARGIN = atof(rm_);
    const char *dr_ = getenv("TORBLES_DIRECT"); if (dr_ && *dr_) MMDIRECT = atoi(dr_) != 0;
    const char *mb_ = getenv("TORBLES_MOBIUS"); if (mb_ && *mb_) MMMOB = atoi(mb_);
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
        T->asum = calloc(RK_DENSE, 8); T->acnt = calloc(RK_DENSE, 8); T->touched = malloc(RK_DENSE * 4); T->ntouched = 0;
        T->asumw = calloc(RK_DENSE, sizeof(u128));
        T->mob = calloc((size_t)TW << MOB_MAXBITS, 8); T->msum = calloc((size_t)1 << MOB_MAXBITS, 8); T->mcnt = calloc((size_t)1 << MOB_MAXBITS, 8);
        T->mob_dirty = 0; T->mobU = 0;
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
    /* row kernel layout: key fields first (value bits in order), then each measure column; at most 64 bits in all */
    int off = 0, n = 0, ok = q->nkey > 0; uint64_t R = 1, kbt = 0;
    for (int i = 0; i < q->nkey; i++) { int kind = q->key_kind[i], c = q->key_arg[i];
        kbt += kind == 3 ? __builtin_popcount(q->key_mask[i]) : kind == 2 ? q->key_nbits[i] : kind == 0 ? t->width[c] : 0; }
    int kpos = (int)kbt;                               /* key i's field ends where key i-1's starts */
#define RK_FILL(c, j, slot, k) do { if (n < 64 && (slot) < 64) { T->rf_col[n] = (int16_t)(c); T->rf_plane[n] = (int16_t)(j); T->rf_slot[n] = (int16_t)(slot); T->rf_key[n] = (int8_t)(k); n++; } else ok = 0; } while (0)
    for (int i = 0; i < q->nkey; i++) R = (q->key_radix[i] && R <= UINT64_MAX / q->key_radix[i]) ? R * q->key_radix[i] : UINT64_MAX;
    memset(T->rk_off, 0, sizeof T->rk_off); memset(T->rk_bits, 0, sizeof T->rk_bits);   /* no layout left from another query */
    int laid = R != UINT64_MAX;                        /* the fields below are laid out only when the code fits 64 bits */
    if (!laid) ok = 0;
    for (int i = 0; i < q->nkey && ok; i++) {
        int kind = q->key_kind[i], c = q->key_arg[i];
        if (kind == 1) { T->rk_off[i] = -1; T->rk_bits[i] = 0; continue; }       /* looked up per row */
        int nb = kind == 3 ? __builtin_popcount(q->key_mask[i]) : kind == 2 ? q->key_nbits[i] : t->width[c];
        kpos -= nb; int o = kpos; T->rk_off[i] = o; T->rk_bits[i] = nb;
        if (kind == 3) { uint32_t mk = q->key_mask[i]; int idx = 0;
            while (mk) { int j = __builtin_ctz(mk); mk &= mk - 1; RK_FILL(c, j, o + nb - 1 - idx, 1); idx++; } }
        else if (kind == 2) { for (int j = 0; j < nb; j++) RK_FILL(c, j, o + nb - 1 - j, 1); }
        else { for (int j = 0; j < nb; j++) RK_FILL(c, j, o + (t->msb[c] ? nb - 1 - j : j), 1); }
    }
    off = (int)kbt;
    int lookups = 0; for (int i = 0; i < q->nkey; i++) lookups += q->key_kind[i] == 1;
    T->rk_kbits = lookups ? -1 : (int)kbt;            /* concatenated key bits (no looked-up key): the code is a mask */
    uint32_t mc = T->mcols;
    while (mc && ok) { int c = __builtin_ctz(mc); mc &= mc - 1; int wd = t->width[c];
        T->rm_off[c] = off; T->rm_bits[c] = wd;
        for (int j = 0; j < wd; j++) RK_FILL(c, j, off + (t->msb[c] ? wd - 1 - j : j), 0);
        off += wd; }
#undef RK_FILL
    T->rk_ok = ok && off <= 64; T->rk_nfill = n; T->rk_R = R;
    T->rk_dense = T->rk_kbits >= 0 ? T->rk_kbits <= 16 : R <= RK_DENSE;   /* dense array indexed by the code */
    T->use_rows = MMROWS == 2 && T->rk_ok; T->use_mob = 0; T->tile_rows = 0; T->since = 0; T->ntouched = 0; T->rk_ratio = -1;
    /* splitting tree: bit pos of the concatenated key code (key 0 most significant) comes from plane kt_plane[pos] */
    T->kt_ok = laid && T->rk_kbits >= 0 && T->rk_kbits <= 64 && q->nkey > 0; T->kt_n = 0;
    if (T->kt_ok) {
        for (int i = 0; i < q->nkey; i++) {
            int kind = q->key_kind[i], c = q->key_arg[i], nb = T->rk_bits[i], o = T->rk_off[i];
            if (kind == 3) { uint32_t mk = q->key_mask[i]; int idx = 0;
                while (mk) { int j = __builtin_ctz(mk); mk &= mk - 1; T->kt_col[o + nb - 1 - idx] = (int16_t)c; T->kt_plane[o + nb - 1 - idx] = (int16_t)j; idx++; } }
            else if (kind == 2) for (int j = 0; j < nb; j++) { T->kt_col[o + nb - 1 - j] = (int16_t)c; T->kt_plane[o + nb - 1 - j] = (int16_t)j; }
            else for (int j = 0; j < nb; j++) { int pos = o + (t->msb[c] ? nb - 1 - j : j); T->kt_col[pos] = (int16_t)c; T->kt_plane[pos] = (int16_t)j; }
        }
        T->kt_n = T->rk_kbits;
    }
    T->kt_direct = T->rk_R <= TS;                      /* the group code is the tile slot: no hashing */
    T->mob_ok = T->kt_ok && T->kt_n <= MOB_MAXBITS && MMMOB != 0; T->tile_mob = 0; T->mob_dirty = 0; T->mobU = 0;
    T->tw = 0;
    if (q->wide) {                                  /* 128-bit weights, in the order of tb_mm_planes */
        int n = 0; T->WC = U128(q->p_const_hi, q->p_const);
        for (int i = 0; i < q->npl; i++) { int c = q->pl_col[i]; u128 cf = U128(q->pl_coef_hi[i], q->pl_coef[i]);
            for (int j = 0; j < t->width[c]; j++) T->W[n++] = cf << pw(t, c, j); }
        for (int i = 0; i < q->npb; i++) { int a = q->pb_a[i], b = q->pb_b[i]; u128 cf = U128(q->pb_coef_hi[i], q->pb_coef[i]);
            for (int j = 0; j < t->width[a]; j++) for (int k = 0; k < t->width[b]; k++) T->W[n++] = cf << (pw(t, a, j) + pw(t, b, k)); }
        T->mob_ok = 0; T->use_mob = 0;
    }
    if (q->widekey) { T->rk_ok = 0; T->kt_ok = 0; T->kt_direct = 0; T->mob_ok = 0; T->use_rows = 0; T->use_mob = 0; }
    T->use_mob = MMMOB == 1 && T->mob_ok;
}
/* the row kernel for one word: keys and measure of every passing row from one transpose, accumulated by group code */
static inline void mm_rows_word(const atable_t *t, const qplan_t *q, dec_t *D, uint64_t w, uint64_t F, mtile_t *T, touch_t *tl, uint64_t *total) {
    uint32_t kl[MAXKEY][64];
    for (int i = 0; i < q->nkey; i++) if (q->key_kind[i] == 1) {        /* looked-up keys: one gather per row */
        int a = q->key_arg[i], c = q->attr_fkcol[a]; const uint64_t *x = decode(t, D, c, w); uint64_t G = F;
        while (G) { int r = __builtin_ctzll(G); G &= G - 1;
            uint32_t fk = KM(q->attr_kmask[a], extract_col(t, x, c, r));
            if (fk >= q->attr_n[a]) F &= ~(1ULL << r); else kl[i][r] = q->attr_vals[a][fk]; }
    }
    if (!F) return;
    uint64_t A[64]; memset(A, 0, sizeof A);
    for (int k = 0; k < T->rk_nfill; k++) {
        int c = T->rf_col[k], j = T->rf_plane[k];
        const uint64_t *x = decode_n(t, D, c, w, t->vflag[c] ? t->width[c] : j + 1);
        A[T->rf_slot[k]] = x[j];
    }
    transpose64(A);
    uint64_t G = F;
    if (q->wide) {                                      /* 128-bit sums: each row's polynomial with 128-bit coefficients */
        int concat = T->rk_kbits >= 0 && T->rk_dense && q->npb <= 2 && q->npl <= 4;   /* the index mm_rows_flush expects */
        const uint64_t km = T->rk_kbits >= 0 && T->rk_kbits < 64 ? (1ULL << T->rk_kbits) - 1 : ~0ULL;
        const u128 c0 = U128(q->p_const_hi, q->p_const);
#define FLDW(x, c) (((x) >> T->rm_off[c]) & ((1ULL << T->rm_bits[c]) - 1))
        while (G) {
            int r = __builtin_ctzll(G); G &= G - 1; uint64_t x = A[r], code = 0;
            if (concat) code = x & km;
            else for (int i = 0; i < q->nkey; i++) {
                uint64_t kv = q->key_kind[i] == 1 ? kl[i][r] : (x >> T->rk_off[i]) & ((T->rk_bits[i] >= 64) ? ~0ULL : ((1ULL << T->rk_bits[i]) - 1));
                code = code * q->key_radix[i] + kv; }
            u128 v = c0;
            for (int l = 0; l < q->npl; l++) v += U128(q->pl_coef_hi[l], q->pl_coef[l]) * (u128)FLDW(x, q->pl_col[l]);
            for (int l = 0; l < q->npb; l++) v += U128(q->pb_coef_hi[l], q->pb_coef[l]) * (u128)(FLDW(x, q->pb_a[l]) * FLDW(x, q->pb_b[l]));
            if (T->rk_dense) { if (!T->acnt[code]) T->touched[T->ntouched++] = (uint32_t)code; T->asumw[code] += v; T->acnt[code]++; }
            else hadd_w(tl, code, v, 1);
            T->tw += v;
        }
#undef FLDW
        return;
    }
    if (T->rk_kbits >= 0 && T->rk_dense && q->npb <= 2 && q->npl <= 4) {
        /* the common case: the code is the low key bits; at most 4 linear and 2 product terms, unrolled */
        const uint64_t km = (1ULL << T->rk_kbits) - 1; uint64_t loc = 0;
        int lo_[4], lw_[4], ao_[2], aw_[2], bo_[2], bw_[2]; uint64_t lc_[4] = {0}, bc_[2] = {0};
        for (int l = 0; l < 4; l++) { int c = l < q->npl ? q->pl_col[l] : -1; lo_[l] = c >= 0 ? T->rm_off[c] : 0; lw_[l] = c >= 0 ? T->rm_bits[c] : 0; lc_[l] = c >= 0 ? q->pl_coef[l] : 0; }
        for (int l = 0; l < 2; l++) { int a = l < q->npb ? q->pb_a[l] : -1, b = l < q->npb ? q->pb_b[l] : -1;
            ao_[l] = a >= 0 ? T->rm_off[a] : 0; aw_[l] = a >= 0 ? T->rm_bits[a] : 0; bo_[l] = b >= 0 ? T->rm_off[b] : 0; bw_[l] = b >= 0 ? T->rm_bits[b] : 0; bc_[l] = a >= 0 ? q->pb_coef[l] : 0; }
#define FLD(x, o, w) (((x) >> (o)) & ((1ULL << (w)) - 1))
        while (G) {
            int r = __builtin_ctzll(G); G &= G - 1; uint64_t x = A[r], code = x & km;
            uint64_t v = (uint64_t)q->p_const + lc_[0] * FLD(x, lo_[0], lw_[0]) + lc_[1] * FLD(x, lo_[1], lw_[1]) + lc_[2] * FLD(x, lo_[2], lw_[2])
                      + lc_[3] * FLD(x, lo_[3], lw_[3]) + bc_[0] * FLD(x, ao_[0], aw_[0]) * FLD(x, bo_[0], bw_[0])
                      + bc_[1] * FLD(x, ao_[1], aw_[1]) * FLD(x, bo_[1], bw_[1]);
            if (!T->acnt[code]) T->touched[T->ntouched++] = (uint32_t)code;
            T->asum[code] += v; T->acnt[code]++; loc += v;
        }
#undef FLD
        *total += loc; return;
    }
    while (G) {
        int r = __builtin_ctzll(G); G &= G - 1; uint64_t x = A[r], code = 0;
        for (int i = 0; i < q->nkey; i++) {
            uint64_t kv = q->key_kind[i] == 1 ? kl[i][r] : (x >> T->rk_off[i]) & ((T->rk_bits[i] >= 64) ? ~0ULL : ((1ULL << T->rk_bits[i]) - 1));
            code = code * q->key_radix[i] + kv;
        }
        uint64_t v = (uint64_t)q->p_const;
        for (int l = 0; l < q->npl; l++) { int c = q->pl_col[l]; v += (uint64_t)q->pl_coef[l] * ((x >> T->rm_off[c]) & ((1ULL << T->rm_bits[c]) - 1)); }
        for (int l = 0; l < q->npb; l++) { int a = q->pb_a[l], b = q->pb_b[l];
            v += (uint64_t)q->pb_coef[l] * ((x >> T->rm_off[a]) & ((1ULL << T->rm_bits[a]) - 1)) * ((x >> T->rm_off[b]) & ((1ULL << T->rm_bits[b]) - 1)); }
        if (T->rk_dense) { if (!T->acnt[code]) T->touched[T->ntouched++] = (uint32_t)code; T->asum[code] += v; T->acnt[code]++; }
        else hadd_n(tl, code, v, 1);
        *total += v;
    }
}
static void mm_rows_flush(const qplan_t *q, mtile_t *T, touch_t *tl) {
    int concat = T->rk_kbits >= 0 && T->rk_dense && q->npb <= 2 && q->npl <= 4;
    for (uint32_t k = 0; k < T->ntouched; k++) {
        uint32_t c = T->touched[k]; uint64_t code = c;
        if (concat) { code = 0;                          /* fields (key 0 most significant) -> mixed radix */
            for (int i = 0; i < q->nkey; i++) code = code * q->key_radix[i] + ((c >> T->rk_off[i]) & ((1ULL << T->rk_bits[i]) - 1)); }
        if (q->wide) { hadd_w(tl, code, T->asumw[c], T->acnt[c]); T->asumw[c] = 0; }
        else hadd_n(tl, code, T->asum[c], T->acnt[c]);
        T->asum[c] = 0; T->acnt[c] = 0;
    }
    T->ntouched = 0;
}
/* the monomial masks of one word: m_S = F & AND_{i in S} (key bit i), every S over the kt_n key bits, by adding the
 * highest bit of S to the mask of S without it (one AND per monomial) */
static inline void mm_mob_word(const atable_t *t, dec_t *D, uint64_t w, int wi, uint64_t F, mtile_t *T) {
    int nb = T->kt_n; uint64_t P[MOB_MAXBITS];
    for (int pos = 0; pos < nb; pos++) { int c = T->kt_col[pos], j = T->kt_plane[pos];
        P[pos] = decode_n(t, D, c, w, t->vflag[c] ? t->width[c] : j + 1)[j]; }
    uint64_t *m = T->mob; m[wi] = F;                     /* m[(S << 6) | wi]: monomial S, word wi of the tile */
    for (uint32_t S = 1; S < (1u << nb); S++) {
        int hb = 31 - __builtin_clz(S); m[((size_t)S << 6) | wi] = m[((size_t)(S ^ (1u << hb)) << 6) | wi] & P[hb];
    }
    T->mobU |= 1ULL << wi;
}
static void mm_decode_tile(const atable_t *t, mtile_t *T, uint64_t wb, uint64_t nwt, uint64_t U);
/* the tile in the monomial basis: per monomial, its count and the measure's popcount product (the same AND-popcount
 * rows as the group product, on monomial masks instead of group masks) */
static void mm_flush_mob(const atable_t *t, const qplan_t *q, mtile_t *T, uint64_t wb, uint64_t nwt) {
    uint64_t U = T->mobU; if (!U) return;
    int nb = T->kt_n; uint32_t NS = 1u << nb;
    mm_decode_tile(t, T, wb, nwt, U);
    const uint64_t *xa[MAXMP], *xb[MAXMP]; uint64_t C[MAXMP];
    for (int p = 0; p < T->np; p++) { xa[p] = T->xt[T->P[p].ca][T->P[p].ja]; xb[p] = T->P[p].cb >= 0 ? T->xt[T->P[p].cb][T->P[p].jb] : T->ones; }
    int dense = __builtin_popcountll(U) > MMSPARSE;
    for (uint32_t S = 0; S < NS; S++) {
        uint64_t *ms = T->mob + ((size_t)S << 6), any = 0, n = 0;
        for (uint64_t u = U; u; u &= u - 1) { int i = __builtin_ctzll(u); any |= ms[i]; n += __builtin_popcountll(ms[i]); }
        if (!any) { for (uint64_t u = U; u; u &= u - 1) ms[__builtin_ctzll(u)] = 0; continue; }
        uint64_t v = (uint64_t)q->p_const * n;
        if (T->np) {
            if (dense) { for (int i = 0; i < TW; i++) if (!(U >> i & 1)) ms[i] = 0; MMROW(ms, xa, xb, T->np, C); }
            else { memset(C, 0, (size_t)T->np * 8);
                for (uint64_t u = U; u; u &= u - 1) { int i = __builtin_ctzll(u); uint64_t mm_ = ms[i];
                    for (int p = 0; p < T->np; p++) C[p] += __builtin_popcountll(mm_ & xa[p][i] & xb[p][i]); } }
            for (int p = 0; p < T->np; p++) v += (uint64_t)T->P[p].w * C[p];
        }
        T->msum[S] += v; T->mcnt[S] += n;
        for (uint64_t u = U; u; u &= u - 1) ms[__builtin_ctzll(u)] = 0;
    }
    T->mobU = 0; T->mob_dirty = 1;
}
/* inclusion-exclusion (the Möbius transform of the superset lattice): the sum over the rows whose key bits are
 * exactly g is sum over S containing g of (-1)^|S - g| (sum over the monomial S). Then the groups go to the table. */
static void mm_mob_finish(const qplan_t *q, mtile_t *T, touch_t *tl, uint64_t *total) {
    int nb = T->kt_n; uint32_t NS = 1u << nb;
    for (int i = 0; i < nb; i++) for (uint32_t S = 0; S < NS; S++) if (!(S >> i & 1)) {
        T->msum[S] -= T->msum[S | (1u << i)]; T->mcnt[S] -= T->mcnt[S | (1u << i)]; }
    for (uint32_t g = 0; g < NS; g++) {
        if (T->mcnt[g]) {
            uint64_t key = 0;
            for (int i = 0; i < q->nkey; i++) key = key * q->key_radix[i] + ((g >> T->rk_off[i]) & ((1ULL << T->rk_bits[i]) - 1));
            hadd_n(tl, key, T->msum[g], T->mcnt[g]); *total += T->msum[g];
        }
        T->msum[g] = 0; T->mcnt[g] = 0;
    }
    T->mob_dirty = 0;
}
/* fold the tile: decode the measure planes of the tile's words (only words holding rows of some group), the product
 * C_tile, then SUM/COUNT per group into the thread's table */
/* row mode of a tile: the measure's integer value for every row of the words holding groups (each measure column's
 * planes transposed once per word: 64 rows at a time), then each group's sum gathered from its mask bits */
static void mm_flush_rows(const atable_t *t, const qplan_t *q, mtile_t *T, uint64_t U, touch_t *tl, uint64_t *total) {
    uint64_t v64[MAXCOL][64], A[64]; uint32_t mc0 = T->mcols;
    for (uint64_t u = U; u; u &= u - 1) {
        int i = __builtin_ctzll(u); uint32_t mc = mc0;
        while (mc) { int c = __builtin_ctz(mc); mc &= mc - 1; int b = t->width[c];
            for (int j = 0; j < 64; j++) A[j] = j < b ? T->xt[c][j][i] : 0;
            transpose64(A);
            if (t->msb[c]) for (int r = 0; r < 64; r++) v64[c][r] = revb((uint32_t)A[r], b);
            else for (int r = 0; r < 64; r++) v64[c][r] = A[r]; }
        uint64_t *o = T->rowv + (size_t)i * 64;
        for (int r = 0; r < 64; r++) {
            uint64_t v = (uint64_t)q->p_const;
            for (int l = 0; l < q->npl; l++) v += (uint64_t)q->pl_coef[l] * v64[q->pl_col[l]][r];
            for (int l = 0; l < q->npb; l++) v += (uint64_t)q->pb_coef[l] * v64[q->pb_a[l]][r] * v64[q->pb_b[l]][r];
            o[r] = v;
        }
    }
    for (int k = 0; k < T->nocc; k++) {
        int s = T->occ[k]; uint64_t v = 0; uint64_t nzm = T->nz[s];
        while (nzm) { int i = __builtin_ctzll(nzm); nzm &= nzm - 1; uint64_t m = T->M[s][i]; const uint64_t *o = T->rowv + (size_t)i * 64;
            while (m) { v += o[__builtin_ctzll(m)]; m &= m - 1; }
            T->M[s][i] = 0; }
        hadd_n(tl, T->code[s], v, T->cnt[s]); *total += v;
        T->nz[s] = 0; T->cnt[s] = 0; T->used[s] = 0;
    }
    T->nocc = 0;
}
static void mm_flush_choose_and_fold(const atable_t *t, const qplan_t *q, mtile_t *T, uint64_t U, touch_t *tl, uint64_t *total);
static void mm_flush(const atable_t *t, const qplan_t *q, mtile_t *T, uint64_t wb, uint64_t nwt, touch_t *tl, uint64_t *total) {
    if (!T->nocc) return;
    T->st[3]++;
    uint64_t U = 0; for (int i = 0; i < T->nocc; i++) U |= T->nz[T->occ[i]];
    mm_decode_tile(t, T, wb, nwt, U);
    mm_flush_choose_and_fold(t, q, T, U, tl, total);
}
static void mm_decode_tile(const atable_t *t, mtile_t *T, uint64_t wb, uint64_t nwt, uint64_t U) {
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
}
static void mm_flush_choose_and_fold(const atable_t *t, const qplan_t *q, mtile_t *T, uint64_t U, touch_t *tl, uint64_t *total) {
    /* Plane popcounts cost (groups x measure planes) per tile; per-row values cost a transpose per measure column and
     * word plus a few operations per row, whatever the number of groups. Dense GROUP BYs with wide or product measures
     * (many groups in every tile) are cheaper per row; selective ones keep the popcount algebra. */
    {
        double cpop = 0, rows = 0, csplit = 0; int kbits = 0;
        for (int i = 0; i < q->nkey; i++) kbits += q->key_kind[i] == 1 ? 8 : T->rk_bits[i];
        for (int k = 0; k < T->nocc; k++) { int s = T->occ[k], nzw = __builtin_popcountll(T->nz[s]);
            cpop += nzw > MMSPARSE ? T->np * MMC_DENSE : (double)nzw * T->np * MMC_SPARSE; rows += (double)T->cnt[s];
            csplit += nzw * (MMC_SPLIT_KEY + MMC_SPLIT_BIT * kbits); }
        int nu = __builtin_popcountll(U), ncol = __builtin_popcount(T->mcols);
        double crow = nu * (ncol * MMC_TRANS + 64.0 * (q->npl + 2 * q->npb) * MMC_TERM) + rows * MMC_GATHER;
        /* the next tiles: the row kernel if it would have cost less than this tile's splitting and product */
        double ckern = nu * MMC_TRANS + rows * (MMC_ROW + MMC_TERM * (q->npl + 2 * q->npb));
        int frows = !q->wide && T->np && MMROWS != 0 && (MMROWS == 1 || crow < RK_MARGIN * cpop);   /* this tile's fold per row */
        double cmm = csplit + (frows ? crow : cpop);
        double cmob = T->mob_ok ? nu * (double)(1u << T->kt_n) * MMC_MOBAND + (double)(1u << T->kt_n) * (T->np + 1) * MMC_DENSE : 1e300;
        T->use_rows = 0; T->use_mob = 0;
        if (MMMOB == 1 && T->mob_ok) T->use_mob = 1;
        else if (T->rk_ok && MMROWS >= 2) T->use_rows = 1;
        else {
            /* the ratio of the estimates, smoothed over the popcount tiles (one tile does not switch the kernel) */
            double r = cmm > 0 ? ckern / cmm : 1, best = cmm;
            T->rk_ratio = T->rk_ratio < 0 ? r : 0.75 * T->rk_ratio + 0.25 * r;
            if (T->rk_ok && MMROWS != 0 && T->rk_ratio < RK_MARGIN) { best = ckern; T->use_rows = 1; }
            if (T->mob_ok && MMMOB != 0 && cmob < best) { T->use_rows = 0; T->use_mob = 1; }
        }
        if (frows) { mm_flush_rows(t, q, T, U, tl, total); return; }
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
        if (q->wide) {                                         /* exact counts, 128-bit weights */
            u128 v = T->WC * (u128)T->cnt[s];
            for (int p = 0; p < T->np; p++) v += T->W[p] * (u128)C[p];
            hadd_w(tl, T->code[s], v, T->cnt[s]); T->tw += v;
        } else {
            uint64_t v = (uint64_t)q->p_const * T->cnt[s];
            for (int p = 0; p < T->np; p++) v += (uint64_t)T->P[p].w * C[p];
            hadd_n(tl, T->code[s], v, T->cnt[s]); *total += v;
        }
        nzm = T->nz[s]; while (nzm) { int i = __builtin_ctzll(nzm); nzm &= nzm - 1; T->M[s][i] = 0; }
        T->nz[s] = 0; T->cnt[s] = 0; T->used[s] = 0;
    }
    T->nocc = 0;
}
/* the groups of one word: split F into the masks of its distinct keys, each placed in its tile slot (row of M^T) */
/* wk: 128-bit group codes (a compile-time constant in each of the two copies below) */
static inline __attribute__((always_inline)) void mm_word_k(const atable_t *t, const qplan_t *q, dec_t *D, uint64_t w, int wi, uint64_t F, mtile_t *T, touch_t *tl, uint64_t *total, const int wk) {
    uint32_t kl[MAXKEY][64]; int slot[MAXKEY], ns = 0;
    for (int i = 0; i < q->nkey; i++) if (q->key_kind[i] == 1) {        /* looked-up keys: one gather per row */
        int a = q->key_arg[i], sl = slot[i] = ns++, c = q->attr_fkcol[a]; const uint64_t *x = decode(t, D, c, w); uint64_t G = F;
        while (G) { int r = __builtin_ctzll(G); G &= G - 1;
            uint32_t fk = KM(q->attr_kmask[a], extract_col(t, x, c, r));
            if (fk >= q->attr_n[a]) F &= ~(1ULL << r); else kl[sl][r] = q->attr_vals[a][fk]; }
    }
    const uint64_t *kp[MAXKEY][32]; int kb[MAXKEY], km[MAXKEY];
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
        int r = __builtin_ctzll(R); uint64_t M = R, k64 = 0; u128 kw = 0;
        for (int i = 0; i < q->nkey; i++) {
            uint64_t kv = 0;
            if (q->key_kind[i] == 1) { kv = kl[slot[i]][r]; uint64_t G = M;
                while (G) { int rr = __builtin_ctzll(G); G &= G - 1; if (kl[slot[i]][rr] != kv) M &= ~(1ULL << rr); } }
            else for (int j = 0; j < kb[i]; j++) { uint64_t pl = *kp[i][j], bit = (pl >> r) & 1; M &= bit ? pl : ~pl; kv |= bit << (km[i] ? kb[i] - 1 - j : j); }
            if (wk) kw = kw * q->key_radix[i] + kv; else k64 = k64 * q->key_radix[i] + kv;
        }
        R &= ~M;
        const u128 key = wk ? kw : (u128)k64;
        uint32_t h;
        if (MMDIRECT && T->kt_direct) h = (uint32_t)k64;                           /* the code is the slot */
        else { h = wk ? (uint32_t)(((uint64_t)kw ^ (uint64_t)(kw >> 64) * 0xC2B2AE3D27D4EB4FULL) * 0x9E3779B97F4A7C15ULL >> (64 - TSB))
                      : (uint32_t)((k64 * 0x9E3779B97F4A7C15ULL) >> (64 - TSB));        /* tile slot: open addressing */
               while (T->used[h] && T->code[h] != key) h = (h + 1) & (TS - 1); }
        if (!T->used[h]) {
            if (!(MMDIRECT && T->kt_direct) && T->nocc >= TS * 3 / 4) {             /* tile table full: this pair is evaluated directly */
                if (q->wide) { u128 v = poly_sum_w(t, q, D, w, M); hadd_w(tl, key, v, __builtin_popcountll(M)); T->tw += v; }
                else { uint64_t v = poly_sum(t, q, D, w, M); hadd_n(tl, key, v, __builtin_popcountll(M)); *total += v; }
                T->st[2]++; continue;
            }
            T->used[h] = 1; T->code[h] = key; T->occ[T->nocc++] = h;
        }
        T->M[h][wi] |= M; T->nz[h] |= 1ULL << wi; T->cnt[h] += __builtin_popcountll(M);
    }
}
static void mm_word_64(const atable_t *t, const qplan_t *q, dec_t *D, uint64_t w, int wi, uint64_t F, mtile_t *T, touch_t *tl, uint64_t *total) { mm_word_k(t, q, D, w, wi, F, T, tl, total, 0); }
static void mm_word_128(const atable_t *t, const qplan_t *q, dec_t *D, uint64_t w, int wi, uint64_t F, mtile_t *T, touch_t *tl, uint64_t *total) { mm_word_k(t, q, D, w, wi, F, T, tl, total, 1); }
/* the groups of one word by Shannon expansion over the key bits: a mask is split by the next key bit's plane into
 * its 0 and 1 halves (two ANDs), empty halves are dropped, and a leaf's path is its group code. Every group present
 * costs at most two ANDs per key bit, shared along common prefixes; when the codes fit the tile, the code is the slot. */
static inline void mm_word_tree(const atable_t *t, const qplan_t *q, dec_t *D, uint64_t w, int wi, uint64_t F, mtile_t *T, touch_t *tl, uint64_t *total) {
    uint64_t P[64]; int nb = T->kt_n;
    for (int pos = 0; pos < nb; pos++) { int c = T->kt_col[pos], j = T->kt_plane[pos];
        P[pos] = decode_n(t, D, c, w, t->vflag[c] ? t->width[c] : j + 1)[j]; }
    uint64_t sm[130], sc[130]; int sd[130], sp = 1; sm[0] = F; sc[0] = 0; sd[0] = nb;
    while (sp) {
        sp--; uint64_t m = sm[sp], cb = sc[sp]; int d = sd[sp];
        if (d) {
            uint64_t p = P[d - 1], m1 = m & p, m0 = m & ~p;
            if (m0) { sm[sp] = m0; sc[sp] = cb << 1; sd[sp] = d - 1; sp++; }
            if (m1) { sm[sp] = m1; sc[sp] = (cb << 1) | 1; sd[sp] = d - 1; sp++; }
            continue;
        }
        u128 key = 0;                                   /* concatenated fields -> mixed-radix group code */
        for (int i = 0; i < q->nkey; i++) key = key * q->key_radix[i] + ((cb >> T->rk_off[i]) & ((T->rk_bits[i] >= 64) ? ~0ULL : ((1ULL << T->rk_bits[i]) - 1)));
        uint32_t h;
        if (MMDIRECT && T->kt_direct) h = (uint32_t)key;
        else {
            h = (uint32_t)(((uint64_t)key * 0x9E3779B97F4A7C15ULL) >> (64 - TSB));
            while (T->used[h] && T->code[h] != key) h = (h + 1) & (TS - 1);
        }
        if (!T->used[h]) {
            if (!(MMDIRECT && T->kt_direct) && T->nocc >= TS * 3 / 4) {
                if (q->wide) { u128 v = poly_sum_w(t, q, D, w, m); hadd_w(tl, key, v, __builtin_popcountll(m)); T->tw += v; }
                else { uint64_t v = poly_sum(t, q, D, w, m); hadd_n(tl, key, v, __builtin_popcountll(m)); *total += v; }
                T->st[2]++; continue;
            }
            T->used[h] = 1; T->code[h] = key; T->occ[T->nocc++] = h;
        }
        T->M[h][wi] |= m; T->nz[h] |= 1ULL << wi; T->cnt[h] += __builtin_popcountll(m);
    }
}
static void at_query_part(const atable_t *t, const qplan_t *q, uint64_t w0, uint64_t w1,
                          int64_t *osum, uint64_t *ocnt, uint64_t *ptotal, uint64_t *pcnt, touch_t *tl, mtile_t *T, u128 *ptw) {
    dec_t D; uint64_t total = 0; uint64_t cnt = 0; u128 tw = 0;
    uint64_t needf = 0;                                  /* fact columns needed as integers (general path) */
    for (int i = 0; i < q->nmeas; i++) if (q->meas[i].op == M_FCOL) needf |= 1ULL << q->meas[i].arg;
    for (int i = 0; i < q->nkey; i++) if (q->key_kind[i] == 0) needf |= 1ULL << q->key_arg[i];
    for (int i = 0; i < q->nattr; i++) needf |= 1ULL << q->attr_fkcol[i];
    uint32_t fv[MAXCOL], av[8];
    for (uint64_t w = w0; w < w1; w++) {
        if (T && (w - w0) % TW == 0) {                   /* tile boundary: fold the last tile, choose the next one's kernel */
            if (w > w0) { if (T->tile_mob) mm_flush_mob(t, q, T, w - TW, TW); else mm_flush(t, q, T, w - TW, TW, tl, &total); }
            int forced = (MMMOB == 1 && T->mob_ok) || (MMROWS >= 2 && T->rk_ok);
            /* a popcount tile measures the model again: every 32nd tile, or every 256th when the row kernel is far ahead
             * (a popcount tile with more groups than the tile table holds evaluates the rest word by word) */
            if ((T->use_rows || T->use_mob) && (forced || T->since < (T->use_rows && T->rk_ratio < RK_MARGIN / 4 ? 255 : 31))) {
                T->tile_rows = T->use_rows; T->tile_mob = T->use_mob; T->since++; }
            else { T->tile_rows = 0; T->tile_mob = 0; T->since = 0; }
        }
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
        if (T) {
            if (T->tile_mob) mm_mob_word(t, &D, w, (int)((w - w0) % TW), F, T);
            else if (T->tile_rows) mm_rows_word(t, q, &D, w, F, T, tl, &total);
            else if (MMTREE && T->kt_ok) mm_word_tree(t, q, &D, w, (int)((w - w0) % TW), F, T, tl, &total);
            else if (q->widekey) mm_word_128(t, q, &D, w, (int)((w - w0) % TW), F, T, tl, &total);
            else mm_word_64(t, q, &D, w, (int)((w - w0) % TW), F, T, tl, &total);
            continue; }
        if (q->agg == A_LIN) {                           /* SUM = sum_j 2^j popcount(x_j & F) */
            for (int i = 0; i < q->nlin; i++) {
                const uint64_t *x = decode(t, &D, q->lin_col[i], w); uint64_t s = 0;
                for (int j = 0; j < t->width[q->lin_col[i]]; j++) s += (uint64_t)__builtin_popcountll(x[j] & F) << j;
                total += (uint64_t)q->lin_coef[i] * s;
            }
            continue;
        }
        if (q->agg == A_POLYG) {
            /* GROUP BY by mask splitting: the rows sharing the first row's key are an AND of key-plane literals
             * (a monomial); the measure over them is popcount algebra. Remove them and repeat. On a table
             * clustered by the key this is almost always a single step per 64-row word. */
            const uint64_t *kp[MAXKEY][32]; int kb[MAXKEY], km[MAXKEY];
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
                int r = __builtin_ctzll(R); uint64_t M = R; u128 key = 0;
                for (int i = 0; i < q->nkey; i++) {
                    uint64_t kv = 0;
                    for (int j = 0; j < kb[i]; j++) {
                        uint64_t pl = *kp[i][j], bit = (pl >> r) & 1; M &= bit ? pl : ~pl;
                        kv |= bit << (km[i] ? kb[i] - 1 - j : j);
                    }
                    key = key * q->key_radix[i] + kv;
                }
                uint64_t n = __builtin_popcountll(M);
                if (q->wide) { u128 v = poly_sum_w(t, q, &D, w, M); tw += v;
                    if (q->nkey) { if (tl) hadd_w(tl, key, v, n); else { add128(&osum[(uint64_t)key], &q->out_sum_hi[(uint64_t)key], v); ocnt[(uint64_t)key] += n; } } }
                else { uint64_t v = poly_sum(t, q, &D, w, M); total += v;
                    if (q->nkey) { if (tl) hadd_n(tl, key, v, n); else { osum[(uint64_t)key] += v; ocnt[(uint64_t)key] += n; } } }
                R &= ~M;
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
                uint64_t n = __builtin_popcountll(M);
                if (q->wide) { u128 v = poly_sum_w(t, q, &D, w, M); if (tl) hadd_w(tl, k, v, n); else { add128(&osum[k], &q->out_sum_hi[k], v); ocnt[k] += n; } }
                else { uint64_t v = poly_sum(t, q, &D, w, M); if (tl) hadd_n(tl, k, v, n); else { osum[k] += v; ocnt[k] += n; } }
            }
            if (q->wide) tw += poly_sum_w(t, q, &D, w, F); else total += poly_sum(t, q, &D, w, F);
            continue;
        }
        if (q->agg == A_BIL) {                           /* SUM(x*y) = sum_jk 2^(j+k) popcount(x_j & y_k & F) */
            const uint64_t *x = decode(t, &D, q->bil_a, w), *y = decode(t, &D, q->bil_b, w);
            int bx = t->width[q->bil_a], by = t->width[q->bil_b];
            for (int j = 0; j < bx; j++) { uint64_t xf = x[j] & F; if (!xf) continue;
                uint64_t s = 0; for (int k = 0; k < by; k++) s += (uint64_t)__builtin_popcountll(xf & y[k]) << k;
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
            uint64_t m = q->nmeas && !q->wide ? eval_meas(q, fv, av) : 0;
            u128 key = 0;
            for (int i = 0; i < q->nkey; i++) {
                uint64_t kv;
                if (q->key_kind[i] == 3) {               /* coordinate projection: read the field's planes */
                    uint32_t mk = q->key_mask[i]; const uint64_t *x = decode_n(t, &D, q->key_arg[i], w, 32 - __builtin_clz(mk)); kv = 0;
                    while (mk) { int j = __builtin_ctz(mk); mk &= mk - 1; kv = (kv << 1) | ((x[j] >> r) & 1); }
                } else kv = q->key_kind[i] == 0 ? fv[q->key_arg[i]] : q->key_kind[i] == 1 ? av[q->key_arg[i]]
                            : extract_msb(decode_n(t, &D, q->key_arg[i], w, q->key_nbits[i]), q->key_nbits[i], r);   /* field read from the fk planes */
                key = key * q->key_radix[i] + kv;
            }
            if (q->wide) {                               /* 128 bits: the measure as its polynomial, row by row */
                u128 v = U128(q->p_const_hi, q->p_const);
                for (int i = 0; i < q->npl; i++) v += U128(q->pl_coef_hi[i], q->pl_coef[i]) * (u128)fv[q->pl_col[i]];
                for (int i = 0; i < q->npb; i++) v += U128(q->pb_coef_hi[i], q->pb_coef[i]) * (u128)((uint64_t)fv[q->pb_a[i]] * fv[q->pb_b[i]]);
                tw += v;
                if (q->nkey) { if (tl) hadd_w(tl, key, v, 1); else { add128(&osum[(uint64_t)key], &q->out_sum_hi[(uint64_t)key], v); ocnt[(uint64_t)key]++; } }
                continue;
            }
            if (q->nkey) { if (tl) hadd(tl, key, m); else { osum[(uint64_t)key] += m; ocnt[(uint64_t)key]++; } }
            total += m;
        }
    }
    if (T && w1 > w0) { uint64_t wb = w0 + (w1 - 1 - w0) / TW * TW;
        if (T->tile_mob) mm_flush_mob(t, q, T, wb, w1 - wb); else mm_flush(t, q, T, wb, w1 - wb, tl, &total); }
    if (T && T->ntouched) mm_rows_flush(q, T, tl);
    if (T && T->mob_dirty) mm_mob_finish(q, T, tl, &total);
    if (T) { tw += T->tw; T->tw = 0; }
    *ptotal += total; *pcnt += cnt; *ptw += tw;
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
/* the degree-2 measure of one row in 128 bits (wide plans on the plain baseline: row-wise int128, as a columnar engine
 * would do it) */
static inline u128 pt_row_w(const ptable_t *t, const qplan_t *q, uint64_t row) {
    u128 v = U128(q->p_const_hi, q->p_const);
    for (int i = 0; i < q->npl; i++) v += U128(q->pl_coef_hi[i], q->pl_coef[i]) * (u128)t->col[q->pl_col[i]][row];
    for (int i = 0; i < q->npb; i++) v += U128(q->pb_coef_hi[i], q->pb_coef[i]) * (u128)((uint64_t)t->col[q->pb_a[i]][row] * t->col[q->pb_b[i]][row]);
    return v;
}
static void pt_query_part(const ptable_t *t, const qplan_t *q, uint64_t ra, uint64_t rb,
                          int64_t *osum, uint64_t *ocnt, uint64_t *ptotal, uint64_t *pcnt, touch_t *tl, u128 *ptw) {
    uint8_t m[CH]; uint64_t total = 0; uint64_t cnt = 0; uint32_t fv[MAXCOL], av[8]; u128 tw = 0;
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
            for (int i = 0; i < q->nlin; i++) { const uint32_t *x = t->col[q->lin_col[i]] + r0; uint64_t s = 0;
                for (int r = 0; r < len; r++) s += m[r] ? x[r] : 0; total += (uint64_t)q->lin_coef[i] * s; }
            for (int r = 0; r < len; r++) cnt += m[r];
            continue;
        }
        if (q->agg == A_BIL) {
            const uint32_t *x = t->col[q->bil_a] + r0, *y = t->col[q->bil_b] + r0; uint64_t s = 0;
            for (int r = 0; r < len; r++) { s += m[r] ? (uint64_t)x[r] * y[r] : 0; cnt += m[r]; }
            total += s; continue;
        }
        if (q->agg == A_DOM) {                           /* plain engine: the same slices, row by row */
            const uint32_t *x = t->col[q->dom_col] + r0;
            for (int r = 0; r < len; r++) {
                if (!m[r]) continue;
                cnt++;
                int k = 0; while (k < q->ndom && x[r] >= q->dom_t[k]) k++;
                if (q->wide) { u128 v = pt_row_w(t, q, r0 + r); tw += v;
                    if (k < q->ndom) { if (tl) hadd_w(tl, k, v, 1); else { add128(&osum[k], &q->out_sum_hi[k], v); ocnt[k]++; } }
                    continue; }
                uint64_t v = (uint64_t)q->p_const;
                for (int i = 0; i < q->npl; i++) v += (uint64_t)q->pl_coef[i] * t->col[q->pl_col[i]][r0 + r];
                for (int i = 0; i < q->npb; i++) v += (uint64_t)q->pb_coef[i] * t->col[q->pb_a[i]][r0 + r] * t->col[q->pb_b[i]][r0 + r];
                total += v;
                if (k < q->ndom) { if (tl) hadd(tl, k, v); else { osum[k] += v; ocnt[k]++; } }
            }
            continue;
        }
        if (q->agg == A_POLYG) {
            for (int r = 0; r < len; r++) {
                if (!m[r]) continue;
                cnt++;
                uint64_t v = (uint64_t)q->p_const; u128 key = 0;
                for (int i = 0; i < q->npl; i++) v += (uint64_t)q->pl_coef[i] * t->col[q->pl_col[i]][r0 + r];
                for (int i = 0; i < q->npb; i++) v += (uint64_t)q->pb_coef[i] * t->col[q->pb_a[i]][r0 + r] * t->col[q->pb_b[i]][r0 + r];
                for (int i = 0; i < q->nkey; i++) {
                    uint32_t x = t->col[q->key_arg[i]][r0 + r];
                    uint64_t kv = x;
                    if (q->key_kind[i] == 2) kv = x >> (t->width[q->key_arg[i]] - q->key_nbits[i]);
                    else if (q->key_kind[i] == 3) { uint32_t mk = q->key_mask[i]; int wd = t->width[q->key_arg[i]]; kv = 0;
                        while (mk) { int j = __builtin_ctz(mk); mk &= mk - 1; kv = (kv << 1) | ((x >> (wd - 1 - j)) & 1); } }
                    key = key * q->key_radix[i] + kv;
                }
                if (q->wide) { u128 vw = pt_row_w(t, q, r0 + r); tw += vw;
                    if (q->nkey) { if (tl) hadd_w(tl, key, vw, 1); else { add128(&osum[(uint64_t)key], &q->out_sum_hi[(uint64_t)key], vw); ocnt[(uint64_t)key]++; } }
                    continue; }
                if (q->nkey) { if (tl) hadd(tl, key, v); else { osum[(uint64_t)key] += v; ocnt[(uint64_t)key]++; } }
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
            uint64_t mv = q->nmeas && !q->wide ? eval_meas(q, fv, av) : 0;
            u128 key = 0;
            for (int i = 0; i < q->nkey; i++) {
                uint64_t kv = q->key_kind[i] == 0 ? fv[q->key_arg[i]] : q->key_kind[i] == 1 ? av[q->key_arg[i]]
                            : q->key_kind[i] == 2 ? fv[q->key_arg[i]] >> (t->width[q->key_arg[i]] - q->key_nbits[i]) : 0;
                if (q->key_kind[i] == 3) { uint32_t mk = q->key_mask[i], x = fv[q->key_arg[i]]; int wd = t->width[q->key_arg[i]]; kv = 0;
                    while (mk) { int j = __builtin_ctz(mk); mk &= mk - 1; kv = (kv << 1) | ((x >> (wd - 1 - j)) & 1); } }
                key = key * q->key_radix[i] + kv;
            }
            if (q->wide) { u128 v = pt_row_w(t, q, r0 + r); tw += v;       /* the measure as its degree-2 polynomial */
                if (q->nkey) { if (tl) hadd_w(tl, key, v, 1); else { add128(&osum[(uint64_t)key], &q->out_sum_hi[(uint64_t)key], v); ocnt[(uint64_t)key]++; } }
                continue; }
            if (q->nkey) { if (tl) hadd(tl, key, mv); else { osum[(uint64_t)key] += mv; ocnt[(uint64_t)key]++; } }
            total += mv;
        }
    }
    *ptotal += total; *pcnt += cnt; *ptw += tw;
}

/* ---------------------------------------------------------------- multithreading
 * Every 64-row word is independent, so work is handed out in chunks from an atomic counter. Each thread keeps its
 * own totals and group arrays; integer sums merge exactly, so results do not depend on the thread count. */
static int NTHREADS = 0;                               /* 0: every online CPU (at most 64), decided at first use */
static int nthreads(void) {
    if (!NTHREADS) { long c = sysconf(_SC_NPROCESSORS_ONLN); NTHREADS = c < 1 ? 1 : c > 64 ? 64 : (int)c; }
    return NTHREADS;
}
void at_set_threads(int n) { NTHREADS = n < 1 ? 1 : n > 64 ? 64 : n; }
int at_get_threads(void) { return nthreads(); }
typedef struct {
    int kind; const void *t; const qplan_t *q; const uint64_t *chunks; uint64_t nchunks, limit; _Atomic uint64_t *next;
    int64_t *osum; uint64_t *ocnt; uint64_t total; uint64_t cnt; touch_t tl; int extra; mtile_t *mt; u128 totalw;
} job_t;
static void *qworker(void *arg) {
    job_t *j = arg;
    for (;;) {
        uint64_t i = atomic_fetch_add(j->next, 1);
        if (i >= j->nchunks) break;
        uint64_t u0 = j->chunks[2 * i], u1 = j->chunks[2 * i + 1];
        touch_t *tl = j->extra ? &j->tl : NULL;
        if (j->kind == 0) at_query_part(j->t, j->q, u0, u1, j->osum, j->ocnt, &j->total, &j->cnt, tl, j->mt, &j->totalw);
        else pt_query_part(j->t, j->q, u0 * CH, u1 * CH < j->limit ? u1 * CH : j->limit, j->osum, j->ocnt, &j->total, &j->cnt, tl, &j->totalw);
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
    int T = nthreads(); job_t jobs[64]; pthread_t th[64]; _Atomic uint64_t next = 0;
    uint64_t *chunks; uint64_t nchunks = make_chunks(q, nunits, chunk, div, &chunks);
    if ((uint64_t)T > nchunks) T = nchunks ? (int)nchunks : 1;      /* no more threads than chunks */
    int mm = kind == 0 && q->agg == A_MM;
    /* sparse output (out_key set): the matrix product always; the general path when the key domain is too large
     * for dense arrays. Every thread then accumulates into its own hash table. */
    int sparse = mm || ((q->agg == A_GENERAL || q->agg == A_POLYG) && q->out_key && q->nkey);
    for (int i = 0; i < T; i++) {
        jobs[i] = (job_t){kind, t, q, chunks, nchunks, limit, &next, q->out_sum, q->out_cnt, 0, 0, {0}, 0, NULL, 0};
        if ((i || sparse) && q->nkey) jobs[i].extra = 1;
        if (mm) { jobs[i].mt = mm_tile(i); mm_prepare(t, q, jobs[i].mt); }
    }
    pool_jobs(T, qworker, jobs, sizeof jobs[0]); (void)th;
    q->total_sum = jobs[0].total; q->total_cnt = jobs[0].cnt;
    if (q->wide) { u128 tw = 0; for (int i = 0; i < T; i++) tw += jobs[i].totalw;
                   q->total_sum_hi = (int64_t)(uint64_t)(tw >> 64); q->total_sum = (int64_t)(uint64_t)tw; }
    if (sparse) {                                        /* sparse output: the groups present, merged across threads */
        touch_t *h0 = &jobs[0].tl; memset(q->mm_stats, 0, sizeof q->mm_stats);
        for (int i = 0; i < T; i++) {
            if (mm) for (int k = 0; k < 4; k++) q->mm_stats[k] += jobs[i].mt->st[k];
            if (i) { if (!q->wide) q->total_sum += jobs[i].total;
                     q->total_cnt += jobs[i].cnt;
                     touch_t *h = &jobs[i].tl;
                     for (uint64_t x = 0; x < h->cap; x++) if (h->key[x] != TEMPTY) hadd_w(h0, h->key[x], U128(h->shi[x], h->sum[x]), h->cnt[x]);
                     hfree(h); }
        }
        uint64_t n = 0;
        for (uint64_t x = 0; x < h0->cap; x++) if (h0->key[x] != TEMPTY) {
            if (n < q->out_cap) { q->out_key[n] = (uint64_t)h0->key[x]; q->out_sum[n] = h0->sum[x]; q->out_cnt[n] = h0->cnt[x];
                                  if (q->out_key_hi) q->out_key_hi[n] = (uint64_t)(h0->key[x] >> 64);
                                  if (q->wide) q->out_sum_hi[n] = h0->shi[x]; }
            n++;
        }
        q->out_n = n; hfree(h0); free(chunks); return;
    }
    for (int i = 1; i < T; i++) {
        if (!q->wide) q->total_sum += jobs[i].total;
        q->total_cnt += jobs[i].cnt;
        if (q->nkey) {
            touch_t *h = &jobs[i].tl;
            for (uint64_t x = 0; x < h->cap; x++) if (h->key[x] != TEMPTY) {
                uint64_t k = (uint64_t)h->key[x]; q->out_cnt[k] += h->cnt[x];
                if (q->wide) add128(&q->out_sum[k], &q->out_sum_hi[k], U128(h->shi[x], h->sum[x])); else q->out_sum[k] = (int64_t)((uint64_t)q->out_sum[k] + (uint64_t)h->sum[x]); }
            hfree(h);
        }
    }
    free(chunks);
}
int64_t at_query(const atable_t *t, qplan_t *q) {
    if (q->wide && q->agg != A_MM && q->agg != A_POLYG && q->agg != A_DOM && q->agg != A_COUNT && q->agg != A_GENERAL) return -2;
    if (q->agg == A_MM) { static mplane_t tmp[MAXMP]; mm_init(); if (!q->nkey || tb_mm_planes(t->width, t->msb, q, tmp) < 0) return -1; }
    run_parallel(0, t, q, t->nw, 512, 0, 1); return 0; }      /* ranges in words */      /* ranges in words */
int64_t pt_query(const ptable_t *t, qplan_t *q) { if (q->wide && q->agg != A_POLYG && q->agg != A_DOM && q->agg != A_COUNT && q->agg != A_GENERAL) return -2; run_parallel(1, t, q, (t->n + CH - 1) / CH, 16, t->n, CH / 64); return 0; }   /* same word ranges, in CH-row chunks */

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
    uint64_t *planes[(2 * MAXCOL + 2) * NB]; int np = 0;
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
