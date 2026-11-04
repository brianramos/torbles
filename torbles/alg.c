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
#include "version.h"
#include "relationships.h"
#include "periodic.h"
#include "cycle_storage.h"
#include <unistd.h>

const char *at_version(void) { return TORBLES_RELEASE; }

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

/* Stable radix partition of compact SQL order keys. A token carries the
 * packed key above its original row id. Each worker counts the masked digit
 * on a contiguous range. Prefix sums assign disjoint, stable bucket ranges
 * in worker/input order. The same order yields ROW_NUMBER via partition
 * starts, without returning a permutation to Python first. */
enum { ORDER_RADIX = 256, ORDER_WORKERS = 32 };
typedef struct {
    uint64_t begin, end, key_mask, row_mask;
    const void *codes;
    uint64_t *src, *dst, *out;
    unsigned key_bytes, row_bits, shift, mask, partition_shift;
    uint32_t offsets[ORDER_RADIX];
    int bad;
} order_job_t;

static void *order_histogram(void *arg) {
    order_job_t *j = arg;
    memset(j->offsets, 0, (j->mask + 1) * sizeof *j->offsets);
    if (j->codes) {
        uint64_t seen = 0;
#define ORDER_INIT(T) do {                                                       \
        const T *codes = j->codes;                                               \
        for (uint64_t i = j->begin; i < j->end; ++i) {                            \
            uint64_t key = codes[i];                                             \
            seen |= key;                                                         \
            j->src[i] = (key << j->row_bits) | i;                                \
            ++j->offsets[key & j->mask];                                         \
        }                                                                        \
    } while (0)
        if (j->key_bytes == 2) ORDER_INIT(uint16_t);
        else if (j->key_bytes == 4) ORDER_INIT(uint32_t);
        else ORDER_INIT(uint64_t);
#undef ORDER_INIT
        j->bad = (seen & ~j->key_mask) != 0;
        j->codes = NULL;
    } else {
        for (uint64_t i = j->begin; i < j->end; ++i)
            ++j->offsets[(j->src[i] >> j->shift) & j->mask];
    }
    return NULL;
}

static void *order_scatter(void *arg) {
    order_job_t *j = arg;
    for (uint64_t i = j->begin; i < j->end; ++i) {
        uint64_t token = j->src[i];
        j->dst[j->offsets[(token >> j->shift) & j->mask]++] = token;
    }
    return NULL;
}

static uint64_t order_partition(uint64_t token, unsigned shift) {
    return shift == 64 ? 0 : token >> shift;
}

static void *order_finish(void *arg) {
    order_job_t *j = arg;
    if (j->partition_shift > 64) {
        for (uint64_t i = j->begin; i < j->end; ++i) j->out[i] = j->src[i] & j->row_mask;
        return NULL;
    }
    unsigned shift = j->partition_shift;
    uint64_t partition = order_partition(j->src[j->begin], shift);
    uint64_t start = j->begin;
    if (start && order_partition(j->src[start - 1], shift) == partition) {
        /* The partition crosses a worker boundary. Its start is the lower
         * bound of this partition in the already sorted key prefix. */
        uint64_t lo = 0;
        while (lo < start) {
            uint64_t mid = lo + (start - lo) / 2;
            if (order_partition(j->src[mid], shift) < partition) lo = mid + 1;
            else start = mid;
        }
    }
    for (uint64_t i = j->begin; i < j->end; ++i) {
        uint64_t token = j->src[i], p = order_partition(token, shift);
        if (p != partition) { partition = p; start = i; }
        j->out[token & j->row_mask] = i - start + 1;
    }
    return NULL;
}

int at_order_codes(const void *codes, unsigned key_bytes, uint64_t n, unsigned key_bits,
                   unsigned order_bits, uint64_t *out, int row_numbers) {
    if (!n) return 0;
    if (!codes || !out || n > UINT32_MAX || n > SIZE_MAX / sizeof(uint64_t) ||
        (key_bytes != 2 && key_bytes != 4 && key_bytes != 8) ||
        !key_bits || key_bits > 63 || key_bits > key_bytes * 8 || order_bits > key_bits) return -1;
    unsigned row_bits = n == 1 ? 0 : 64 - (unsigned)__builtin_clzll(n - 1);
    if (key_bits + row_bits > 64) return -1;
    uint64_t *src = malloc(n * sizeof *src), *dst = malloc(n * sizeof *dst);
    if (!src || !dst) { free(src); free(dst); return -1; }
    extern int at_get_threads(void);
    int workers = at_get_threads();
    int useful = (int)((n + 65535) / 65536);
    if (workers > useful) workers = useful;
    if (workers > ORDER_WORKERS) workers = ORDER_WORKERS;
    if (workers < 1) workers = 1;
    order_job_t jobs[ORDER_WORKERS];
    for (int t = 0; t < workers; ++t) {
        jobs[t] = (order_job_t){.begin = n * t / workers, .end = n * (t + 1) / workers,
            .key_mask = (UINT64_C(1) << key_bits) - 1, .row_mask = (UINT64_C(1) << row_bits) - 1,
            .codes = codes, .out = out, .key_bytes = key_bytes, .row_bits = row_bits,
            .partition_shift = row_numbers ? row_bits + order_bits : 65};
    }
    int rc = 0;
    for (unsigned bit = 0; bit < key_bits; bit += 8) {
        unsigned digit_bits = key_bits - bit < 8 ? key_bits - bit : 8;
        unsigned buckets = 1u << digit_bits;
        for (int t = 0; t < workers; ++t) {
            jobs[t].src = src; jobs[t].dst = dst;
            jobs[t].shift = row_bits + bit; jobs[t].mask = buckets - 1;
        }
        pool_jobs(workers, order_histogram, jobs, sizeof *jobs);
        if (!bit) for (int t = 0; t < workers; ++t) if (jobs[t].bad) { rc = -1; goto done; }
        uint32_t prefix = 0;
        unsigned nonempty = 0;
        for (unsigned b = 0; b < buckets; ++b) {
            uint32_t before = prefix;
            for (int t = 0; t < workers; ++t) {
                uint32_t count = jobs[t].offsets[b];
                jobs[t].offsets[b] = prefix;
                prefix += count;
            }
            nonempty += prefix != before;
        }
        if (nonempty > 1) {
            pool_jobs(workers, order_scatter, jobs, sizeof *jobs);
            uint64_t *old = src; src = dst; dst = old;
        }
    }
    for (int t = 0; t < workers; ++t) jobs[t].src = src;
    pool_jobs(workers, order_finish, jobs, sizeof *jobs);
done:
    free(src); free(dst);
    return rc;
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
    /* an empty torble's planes have no words but must still exist (a NULL plane means an implicit zero plane, which
     * would change its faces and its file directory): aligned_alloc of 0 bytes may return NULL, so never ask for 0 */
    size_t sz = ((t->nw * 8) + 63) & ~(size_t)63;
    uint64_t *p = aligned_alloc(64, sz ? sz : 64);
    t->bytes += t->nw * 8; return p;
}
static inline uint64_t rowmask(const atable_t *t, uint64_t w) {
    uint64_t left = t->n - w * 64; return left >= 64 ? ~0ULL : ((1ULL << left) - 1);
}

static const uint64_t *periodic_plane(const atable_t *t, unsigned col, unsigned bit) {
    const atable_t *owner = t->cycle_sources[col];
    return owner ? owner->V[t->cycle_columns[col]][bit] : t->V[col][bit];
}

static int periodic_column(const atable_t *t, int32_t col) {
    if (col < 0 || col >= t->k || t->vflag[col] || t->width[col] < 1 || t->width[col] > 32) return 0;
    if (at_cycle_v_planes(t, 0, (uint32_t)col, NULL) > 0) return 1;
    for (int bit = 0; bit < t->width[col]; ++bit) if (!periodic_plane(t, col, bit)) return 0;
    return 1;
}

/* A prefix of length twice the bound is sufficient. By Fine-Wilf its
 * least period divides every other bounded prefix period. Any extension
 * with an allowed period therefore repeats that same least motif. */
static unsigned periodic_prefix(const atable_t *t, unsigned col, size_t first,
                                unsigned max_words) {
    size_t prefix = 2 * (size_t)max_words;
    for (unsigned period = 1; period <= max_words; ++period) {
        int bit;
        for (bit = 0; bit < t->width[col]; ++bit) {
            const uint64_t *v = periodic_plane(t, col, bit);
            if (v && memcmp(v + first + period, v + first, (prefix - period) * sizeof *v)) break;
        }
        if (bit == t->width[col]) return period;
    }
    return 0;
}

static int periodic_verify(const atable_t *t, unsigned col, size_t first,
                           size_t full, uint64_t tail, size_t prefix,
                           unsigned period) {
    for (int bit = 0; bit < t->width[col]; ++bit) {
        const uint64_t *v = periodic_plane(t, col, bit);
        if (!v) continue;
        v += first;
        if (memcmp(v + prefix, v + prefix - period, (full - prefix) * sizeof *v) ||
            (tail && ((v[full] ^ v[full % period]) & tail))) return 0;
    }
    return 1;
}

static int periodic_layout(const atable_t *t) {
    return t && t->k >= 1 && t->k <= MAXCOL &&
           t->nw == t->n / 64 + (t->n % 64 != 0) &&
           t->nw <= SIZE_MAX / sizeof(uint64_t);
}

/* The word periods of the stored XOR planes are also periods of their
 * logical codes because XOR across planes is invertible. */
int at_periodic_periods(const atable_t *t, uint32_t max_words,
                        uint16_t *periods, uint32_t capacity) {
    if (!periodic_layout(t) || !at_has_dense(t) || !periods || !max_words || max_words > 128 ||
        capacity < (uint32_t)t->k) return -1;

    uint64_t revision = __atomic_load_n(&t->revision, __ATOMIC_ACQUIRE);
    uint16_t found[MAXCOL] = {0};
    size_t full = (size_t)(t->n / 64);
    if (max_words > full / 2) max_words = (uint32_t)(full / 2);
    uint64_t tail = t->n % 64 ? rowmask(t, full) : 0;
    for (int col = 0; col < t->k && max_words; ++col) {
        if (t->vflag[col] || t->width[col] < 1 || t->width[col] > 32) continue;
        unsigned period = periodic_prefix(t, (unsigned)col, 0, max_words);
        if (period && periodic_verify(t, (unsigned)col, 0, full, tail, 2 * max_words, period))
            found[col] = (uint16_t)period;
    }
    if (revision != __atomic_load_n(&t->revision, __ATOMIC_ACQUIRE)) return -1;
    memcpy(periods, found, (size_t)t->k * sizeof *periods);
    return 0;
}

static unsigned periodic_lcm(unsigned a, unsigned b) {
    unsigned x = a, y = b;
    while (y) { unsigned r = x % y; x = y; y = r; }
    return (a / x) * b;
}

typedef struct {
    const atable_t *owner;
    const uint16_t *periods;
    uint64_t revision;
} periodic_index_t;

static uint32_t periodic_index_columns(const atable_t *t, uint32_t columns,
                                       periodic_index_t *indices) {
    if (__atomic_load_n(&t->revision, __ATOMIC_ACQUIRE) != t->cycle_revision) return 0;
    uint32_t nsectors = (uint32_t)((t->nw + TP_SECTOR_WORDS - 1) / TP_SECTOR_WORDS);
    uint32_t indexed = 0;
    for (uint32_t mask = columns; mask; mask &= mask - 1) {
        unsigned col = (unsigned)__builtin_ctz(mask), source_col = col;
        const atable_t *owner = t->cycle_sources[col];
        if (owner) source_col = t->cycle_columns[col];
        else owner = t;
        uint64_t revision = __atomic_load_n(&owner->revision, __ATOMIC_ACQUIRE);
        if (owner->n != t->n || owner->nw != t->nw || source_col >= (unsigned)owner->k ||
            !owner->cycle_periods || owner->cycle_sectors != nsectors ||
            revision != owner->cycle_revision) continue;
        if (indices) indices[col] = (periodic_index_t){owner, owner->cycle_periods + source_col, revision};
        indexed |= UINT32_C(1) << col;
    }
    return indexed;
}

int at_periodic_has_persisted(const atable_t *t, uint32_t columns_mask) {
    if (!periodic_layout(t) || !t->n || t->n > UINT32_MAX || (columns_mask >> t->k)) return 0;
    return periodic_index_columns(t, columns_mask, NULL) != 0;
}

static unsigned periodic_index_period(const periodic_index_t *index, uint32_t sector) {
    return index->periods[(size_t)sector * index->owner->k];
}

static int periodic_index_current(uint32_t indexed, const periodic_index_t *indices) {
    for (uint32_t mask = indexed; mask; mask &= mask - 1) {
        const periodic_index_t *index = indices + __builtin_ctz(mask);
        if (index->revision != __atomic_load_n(&index->owner->revision, __ATOMIC_ACQUIRE)) return 0;
    }
    return 1;
}

/* A later local motif can be shorter than the first motif. Its first p
 * words alone only prove the match if the first motif also repeats at p. */
static int periodic_reference_repeats(const atable_t *t, unsigned col, unsigned first_period,
                                      unsigned period) {
    unsigned shift = period % first_period;
    if (!shift) return 1;
    tb_cycle_v_t edges[NB];
    if (at_cycle_v_planes(t, 0, col, edges) > 0) {
        for (int bit = 0; bit < t->width[col]; ++bit) {
            const tb_cycle_v_t *edge = &edges[bit];
            if (!edge->period) return 0;
            for (uint32_t i = 0, n = edge->period; i < n; ++i)
                if (edge->data[i] != edge->data[(i + shift) % n]) return 0;
        }
        return 1;
    }
    unsigned right = first_period - shift;
    for (int bit = 0; bit < t->width[col]; ++bit) {
        const uint64_t *v = periodic_plane(t, col, bit);
        if (memcmp(v, v + shift, right * sizeof *v) ||
            memcmp(v, v + right, shift * sizeof *v)) return 0;
    }
    return 1;
}

static int periodic_global_match(const atable_t *t, unsigned col, uint32_t first,
                                  uint32_t words, uint32_t cycle) {
    tb_cycle_v_t reference[NB], local[NB];
    if (at_cycle_v_planes(t, 0, col, reference) > 0 &&
        at_cycle_v_planes(t, first / TP_SECTOR_WORDS, col, local) > 0) {
        uint32_t offset = first % TP_SECTOR_WORDS;
        for (int bit = 0; bit < t->width[col]; ++bit) {
            if (!reference[bit].period) return 0;
            for (uint32_t i = 0; i < words; ++i)
                if (tb_cycle_v_word(&local[bit], offset + i) !=
                    tb_cycle_v_word(&reference[bit], first + i)) return 0;
        }
        return 1;
    }
    for (uint32_t phase = first % cycle; words; phase = 0) {
        uint32_t span = cycle - phase;
        if (span > words) span = words;
        for (int bit = 0; bit < t->width[col]; ++bit) {
            const uint64_t *v = periodic_plane(t, col, bit);
            if (memcmp(v + first, v + phase, span * sizeof *v)) return 0;
        }
        first += span;
        words -= span;
    }
    return 1;
}

static int periodic_global_tail(const atable_t *t, unsigned col, uint32_t full,
                                uint32_t cycle, uint64_t tail) {
    tb_cycle_v_t reference[NB], local[NB];
    if (at_cycle_v_planes(t, 0, col, reference) > 0 &&
        at_cycle_v_planes(t, full / TP_SECTOR_WORDS, col, local) > 0) {
        for (int bit = 0; bit < t->width[col]; ++bit)
            if ((tb_cycle_v_word(&local[bit], full % TP_SECTOR_WORDS) ^
                 tb_cycle_v_word(&reference[bit], full % cycle)) & tail) return 0;
    } else {
        for (int bit = 0; bit < t->width[col]; ++bit) {
            const uint64_t *v = periodic_plane(t, col, bit);
            if ((v[full] ^ v[full % cycle]) & tail) return 0;
        }
    }
    return 1;
}

static unsigned periodic_global_local(const periodic_index_t *index, uint32_t sector,
                                       uint32_t full_words) {
    unsigned period = periodic_index_period(index, sector);
    return period <= TP_SECTOR_WORDS / TP_MIN_REPEATS && period <= full_words ? period : 0;
}

int at_periodic_ingested_global(const atable_t *t, uint32_t columns_mask,
                                uint32_t *cycle, uint64_t *uses) {
    if (!cycle || !periodic_layout(t) || !t->n || t->n > UINT32_MAX || !columns_mask ||
        (columns_mask >> t->k)) return -1;
    *cycle = 0;
    if (uses) *uses = 0;

    uint64_t revision = __atomic_load_n(&t->revision, __ATOMIC_ACQUIRE);
    if (revision != t->cycle_revision) return -1;
    periodic_index_t indices[MAXCOL];
    uint32_t indexed = periodic_index_columns(t, columns_mask, indices);
    for (uint32_t mask = columns_mask; mask; mask &= mask - 1) {
        unsigned col = (unsigned)__builtin_ctz(mask);
        if (!periodic_column(t, col)) return -1;
        const atable_t *owner = t->cycle_sources[col];
        if (owner && owner->cycle_periods &&
            __atomic_load_n(&owner->revision, __ATOMIC_ACQUIRE) != owner->cycle_revision) return -1;
    }
    if (indexed != columns_mask) return 0;

    uint32_t nsectors = (uint32_t)((t->nw + TP_SECTOR_WORDS - 1) / TP_SECTOR_WORDS);
    uint32_t full = (uint32_t)(t->n / 64), cap = full / TP_MIN_REPEATS;
    if (cap > TP_MAX_CYCLE_WORDS) cap = TP_MAX_CYCLE_WORDS;
    uint32_t candidate = 1, result = 0;
    uint64_t inspected = 0;
    /* The whole candidate prefix fits in the first sector. Later periods
     * are certificates to compare, so they cannot increase its length. */
    uint32_t first_words = full < TP_SECTOR_WORDS ? full : TP_SECTOR_WORDS;
    for (uint32_t mask = columns_mask; mask; mask &= mask - 1) {
        unsigned col = (unsigned)__builtin_ctz(mask);
        unsigned period = periodic_global_local(indices + col, 0, first_words);
        ++inspected;
        if (!period) goto finished;
        candidate = periodic_lcm(candidate, period);
        if (candidate > cap) goto finished;
    }

    uint32_t tail_rows = (uint32_t)(t->n % 64);
    for (uint32_t sector = 1; sector < nsectors; ++sector) {
        uint32_t first = sector * TP_SECTOR_WORDS, words = full - first;
        if (words > TP_SECTOR_WORDS) words = TP_SECTOR_WORDS;
        int final = sector + 1 == nsectors;
        for (uint32_t mask = columns_mask; mask; mask &= mask - 1) {
            unsigned col = (unsigned)__builtin_ctz(mask);
            const periodic_index_t *index = indices + col;
            unsigned period = periodic_global_local(index, sector, words);
            ++inspected;
            /* Only a bounded final literal can be matched without an index. */
            if (final && words + (tail_rows != 0) <= candidate) {
                if (!periodic_global_match(t, col, first, words, candidate)) goto finished;
                continue;
            }
            if (!period || !periodic_reference_repeats(t, col, periodic_index_period(index, 0), period) ||
                !periodic_global_match(t, col, first, period, candidate)) goto finished;
        }
    }
    if (tail_rows) {
        uint64_t tail = (UINT64_C(1) << tail_rows) - 1;
        for (uint32_t mask = columns_mask; mask; mask &= mask - 1)
            if (!periodic_global_tail(t, (unsigned)__builtin_ctz(mask), full, candidate, tail)) goto finished;
    }
    result = candidate;

finished:
    if (revision != __atomic_load_n(&t->revision, __ATOMIC_ACQUIRE) ||
        !periodic_index_current(indexed, indices)) return -1;
    *cycle = result;
    if (uses) *uses = inspected;
    return 0;
}

static unsigned periodic_sector_cycle(const atable_t *t, uint32_t columns,
                                      const tp_sector_t *sector, unsigned max_words,
                                      uint32_t indexed, const periodic_index_t *indices,
                                      uint32_t sector_id, uint64_t *persisted_uses) {
    unsigned cap = sector->full_words / TP_MIN_REPEATS;
    if (cap > TP_MAX_CYCLE_WORDS) cap = TP_MAX_CYCLE_WORDS;
    if (max_words > cap) max_words = cap;
    if (!max_words) return 0;
    unsigned periods[MAXCOL], cycle = 1;

    /* Consult the validated file index first. A zero entry keeps this sector
     * on the ordinary path, even if another dependency has no loaded index. */
    for (uint32_t mask = indexed; mask; mask &= mask - 1) {
        unsigned col = (unsigned)__builtin_ctz(mask);
        const periodic_index_t *index = indices + col;
        unsigned period = periodic_index_period(index, sector_id);
        ++*persisted_uses;
        if (!period || period > cap) return 0;
        cycle = periodic_lcm(cycle, period);
        if (cycle > cap) return 0;
    }

    /* Probe legacy or invalidated dependencies before reading beyond their
     * prefixes. Loaded dependencies have already been proved by the loader. */
    uint32_t discover = columns & ~indexed;
    for (uint32_t mask = discover; mask; mask &= mask - 1) {
        unsigned col = (unsigned)__builtin_ctz(mask);
        if (!periodic_plane(t, col, 0)) return 0;
        unsigned period = periodic_prefix(t, col, sector->first_word, max_words);
        if (!period) return 0;
        cycle = periodic_lcm(cycle, period);
        if (cycle > cap) return 0;
        periods[col] = period;
    }
    uint64_t tail = sector->tail_rows ? (UINT64_C(1) << sector->tail_rows) - 1 : 0;
    for (uint32_t mask = discover; mask; mask &= mask - 1) {
        unsigned col = (unsigned)__builtin_ctz(mask);
        if (!periodic_verify(t, col, sector->first_word, sector->full_words,
                             tail, 2 * max_words, periods[col])) return 0;
    }
    return cycle;
}

static void periodic_sector_metrics(tp_sector_metrics_t *m, const tp_sector_t *sector) {
    if (sector->cycle_words) ++m->periodic_sectors;
    else ++m->ordinary_sectors;
    m->representative_words += (sector->cycle_words ? sector->cycle_words : sector->full_words)
                               + (sector->tail_rows != 0);
}

int at_periodic_plan_sectors_ex(const atable_t *t, uint32_t columns_mask,
                                uint32_t sector_words, uint32_t max_words,
                                tp_sector_t *sectors, uint32_t capacity,
                                tp_sector_metrics_t *metrics, uint64_t *persisted_uses) {
    if (!periodic_layout(t) || !t->n || t->n > UINT32_MAX || !columns_mask ||
        (columns_mask >> t->k) || !sectors || sector_words > TP_MAX_SECTOR_WORDS ||
        !max_words || max_words > 128) return -1;
    if (!sector_words) sector_words = (uint32_t)t->nw;
    uint32_t nsectors = (uint32_t)((t->nw + sector_words - 1) / sector_words);
    if (capacity < nsectors) return -1;

    uint64_t revision = __atomic_load_n(&t->revision, __ATOMIC_ACQUIRE);
    for (uint32_t mask = columns_mask; mask; mask &= mask - 1)
        if (!periodic_column(t, __builtin_ctz(mask))) return -1;
    periodic_index_t indices[MAXCOL];
    uint32_t indexed = 0;
    if (sector_words == TP_SECTOR_WORDS || (nsectors == 1 && t->nw <= TP_SECTOR_WORDS))
        indexed = periodic_index_columns(t, columns_mask, indices);
    uint32_t full = (uint32_t)(t->n / 64);
    uint64_t uses = 0;
    tp_sector_metrics_t result = {0};
    for (uint32_t i = 0; i < nsectors; ++i) {
        uint32_t first = i * sector_words, words = full - first;
        if (words > sector_words) words = sector_words;
        tp_sector_t sector = {first, words, 0, i + 1 == nsectors ? (uint32_t)(t->n % 64) : 0};
        sector.cycle_words = periodic_sector_cycle(t, columns_mask, &sector, max_words,
                                                    indexed, indices, i, &uses);
        sectors[i] = sector;
        periodic_sector_metrics(&result, &sector);
    }
    if (revision != __atomic_load_n(&t->revision, __ATOMIC_ACQUIRE) ||
        !periodic_index_current(indexed, indices)) return -1;
    if (metrics) *metrics = result;
    if (persisted_uses) *persisted_uses = uses;
    return 0;
}

int at_periodic_plan_sectors(const atable_t *t, uint32_t columns_mask,
                             uint32_t sector_words, uint32_t max_words,
                             tp_sector_t *sectors, uint32_t capacity,
                             tp_sector_metrics_t *metrics) {
    return at_periodic_plan_sectors_ex(t, columns_mask, sector_words, max_words,
                                       sectors, capacity, metrics, NULL);
}

/* Radix ordering over the stored XOR planes. Each pass holds only the
 * current digit and two arrays of row ids. In each bucket, workers
 * scatter contiguous pieces of the prior order into disjoint ranges. */
enum { PLANE_ORDER_BITS = 8, PLANE_ORDER_KEYS = 12 };
typedef struct {
    const atable_t *table;
    const uint64_t *filter;
    const int32_t *cols;
    uint8_t *digit;
    uint64_t *ids, *rank;
    uint32_t *src, *dst;
    uint64_t begin, end, words_begin, words_end, count;
    unsigned col, bit, bits, desc, npart;
    _Atomic int *inversion;
    uint32_t offsets[1 << PLANE_ORDER_BITS];
} plane_order_job_t;

static inline uint64_t plane_order_rows(const plane_order_job_t *j, uint64_t w) {
    return rowmask(j->table, w) & (j->filter ? j->filter[w] : UINT64_MAX);
}

static void *plane_order_count(void *arg) {
    plane_order_job_t *j = arg;
    j->count = 0;
    for (uint64_t w = j->words_begin; w < j->words_end; ++w)
        j->count += __builtin_popcountll(plane_order_rows(j, w));
    return NULL;
}

static void *plane_order_init(void *arg) {
    plane_order_job_t *j = arg;
    if (!j->filter) {
        for (uint64_t i = j->begin; i < j->end; ++i) j->src[i] = (uint32_t)i;
    } else {
        uint32_t *out = j->src + j->count;
        for (uint64_t w = j->words_begin; w < j->words_end; ++w) {
            uint64_t mask = plane_order_rows(j, w);
            while (mask) {
                *out++ = (uint32_t)(64 * w + __builtin_ctzll(mask));
                mask &= mask - 1;
            }
        }
    }
    return NULL;
}

/* If a trailing ORDER BY key is globally monotone, stable physical order
 * already supplies it. Test all within-word neighbours using shifted planes,
 * and the single adjacent pair at each word/worker boundary. */
static void *plane_order_monotone(void *arg) {
    plane_order_job_t *j = arg;
    const atable_t *t = j->table;
    unsigned c = j->col, width = t->width[c], msb = t->msb[c];
    uint32_t prev = 0;
    uint64_t start = j->words_begin;
    if (start) {
        uint64_t x = 0;
        for (unsigned p = 0; p < width; ++p) {
            x ^= t->V[c][p][start - 1];
            prev |= (uint32_t)(x >> 63) << (msb ? width - p - 1 : p);
        }
    }
    for (uint64_t w = start; w < j->words_end; ++w) {
        if ((w & 63) == 0 && atomic_load_explicit(j->inversion, memory_order_relaxed)) return NULL;
        uint64_t x = 0, bad = 0, equal = rowmask(t, w) >> 1;
        uint32_t first = 0, last = 0;
        unsigned tail = w + 1 < t->nw ? 63 : (unsigned)((t->n - 1) & 63);
        for (unsigned p = 0; p < width; ++p) {
            x ^= t->V[c][p][w];
            uint64_t a = j->desc ? x >> 1 : x, b = j->desc ? x : x >> 1;
            if (msb) {
                bad |= equal & a & ~b;
                equal &= ~(a ^ b);
                if (bad) goto unordered;
            } else {
                bad = (bad & ~(a ^ b)) | (a & ~b);
            }
            unsigned bit = msb ? width - p - 1 : p;
            first |= (uint32_t)(x & 1) << bit;
            last |= (uint32_t)((x >> tail) & 1) << bit;
        }
        if ((!msb && (bad & equal)) || (w && (j->desc ? prev < first : prev > first))) goto unordered;
        prev = last;
    }
    return NULL;
unordered:
    atomic_store_explicit(j->inversion, 1, memory_order_relaxed);
    return NULL;
}

static inline uint64_t plane_order_transpose8(uint64_t x) {
    uint64_t z = (x ^ (x >> 7)) & UINT64_C(0x00aa00aa00aa00aa);
    x ^= z ^ (z << 7);
    z = (x ^ (x >> 14)) & UINT64_C(0x0000cccc0000cccc);
    x ^= z ^ (z << 14);
    z = (x ^ (x >> 28)) & UINT64_C(0x00000000f0f0f0f0);
    return x ^ z ^ (z << 28);
}

static void *plane_order_digit(void *arg) {
    plane_order_job_t *j = arg;
    const atable_t *t = j->table;
    unsigned c = j->col, width = t->width[c], msb = t->msb[c];
    unsigned first = msb ? width - j->bit - j->bits : j->bit;
    const uint64_t *const *v = (const uint64_t *const *)t->V[c];
    for (uint64_t w = j->words_begin; w < j->words_end; ++w) {
        if (j->filter && !plane_order_rows(j, w)) continue;
        uint64_t value = 0;
        if (first + 1 <= width - first) {
            for (unsigned p = 0; p <= first; ++p) value ^= v[p][w];
        } else {
            for (unsigned p = first + 1; p <= width; ++p) value ^= v[p][w];
        }
        uint64_t plane[PLANE_ORDER_BITS] = {0};
        for (unsigned p = 0; p < j->bits; ++p) {
            if (p) value ^= v[first + p][w];
            plane[msb ? j->bits - p - 1 : p] = j->desc ? ~value : value;
        }
        /* The only row-indexed key scratch is this one radix digit. Its
         * bytes make histogram/scatter random gathers a single load. */
        uint8_t *digit = j->digit + w * 64;
        for (unsigned r = 0; r < 64; r += 8) {
            uint64_t block = 0;
            for (unsigned p = 0; p < PLANE_ORDER_BITS; ++p)
                block |= ((plane[p] >> r) & 255) << (p * 8);
            block = plane_order_transpose8(block);
            memcpy(digit + r, &block, sizeof block);
        }
    }
    return NULL;
}

static void *plane_order_histogram(void *arg) {
    plane_order_job_t *j = arg;
    memset(j->offsets, 0, ((size_t)1 << j->bits) * sizeof *j->offsets);
    for (uint64_t i = j->begin; i < j->end; ++i)
        ++j->offsets[j->digit[j->src[i]]];
    return NULL;
}

static void *plane_order_scatter(void *arg) {
    plane_order_job_t *j = arg;
    for (uint64_t i = j->begin; i < j->end; ++i) {
        uint32_t id = j->src[i];
        j->dst[j->offsets[j->digit[id]]++] = id;
    }
    return NULL;
}

/* The stored XOR sequence is invertible, so partition keys compare for
 * equality without materializing their values. */
static inline int plane_order_same(const plane_order_job_t *j, uint32_t a, uint32_t b) {
    unsigned wa = a >> 6, wb = b >> 6, ra = a & 63, rb = b & 63;
    for (unsigned k = 0; k < j->npart; ++k) {
        unsigned c = j->cols[k];
        for (int p = 0; p < j->table->width[c]; ++p) {
            const uint64_t *v = j->table->V[c][p];
            if (((v[wa] >> ra) ^ (v[wb] >> rb)) & 1) return 0;
        }
    }
    return 1;
}

static void *plane_order_finish(void *arg) {
    plane_order_job_t *j = arg;
    if (!j->rank) {
        for (uint64_t i = j->begin; i < j->end; ++i) j->ids[i] = j->src[i];
        return NULL;
    }
    uint64_t start = j->npart ? j->begin : 0;
    if (start && plane_order_same(j, j->src[start - 1], j->src[start])) {
        uint32_t id = j->src[start];
        uint64_t lo = 0;
        while (lo < start) {
            uint64_t mid = lo + (start - lo) / 2;
            if (plane_order_same(j, j->src[mid], id)) start = mid;
            else lo = mid + 1;
        }
    }
    for (uint64_t i = j->begin; i < j->end; ++i) {
        uint32_t id = j->src[i];
        if (j->npart && i > j->begin && !plane_order_same(j, j->src[i - 1], id)) start = i;
        if (j->ids) j->ids[i] = id;
        j->rank[j->ids ? i : id] = i - start + 1;
    }
    return NULL;
}

/* Select a bounded lexicographic prefix of physical row IDs. A logical wide
 * key is represented by consecutive high and low columns of the same direction.
 * Only the candidate and accepted masks scale with row count. */
#define PLANE_TOPK_KEYS 12
#define PLANE_TOPK_LIMIT 1024

typedef struct {
    uint64_t id;
    uint32_t key[PLANE_TOPK_KEYS];
} plane_topk_row_t;

static int plane_topk_cmp(const void *pa, const void *pb) {
    const plane_topk_row_t *a = pa, *b = pb;
    for (unsigned c = 0; c < PLANE_TOPK_KEYS; ++c) {
        if (a->key[c] != b->key[c]) return a->key[c] < b->key[c] ? -1 : 1;
    }
    return (a->id > b->id) - (a->id < b->id);
}

static uint32_t plane_topk_code(const atable_t *t, int c, uint64_t row) {
    uint64_t w = row >> 6;
    unsigned bit = row & 63;
    uint32_t code = 0, value = 0;
    uint64_t virtual_bits[32];
    int flag = t->vflag[c];
    if (flag) tb_virtual_planes(t, c, t->V[flag - 1][0][w], w, virtual_bits, t->width[c]);
    for (int p = 0; p < t->width[c]; ++p) {
        if (flag) value = (uint32_t)((virtual_bits[p] >> bit) & 1);
        else value ^= (uint32_t)((t->V[c][p][w] >> bit) & 1);
        unsigned significance = t->msb[c] ? t->width[c] - 1 - p : p;
        code |= value << significance;
    }
    return code;
}

int at_plane_topk(const atable_t *t, const uint64_t *filter, int32_t nkeys,
                  const int32_t *cols, const int32_t *desc, uint32_t cap,
                  uint64_t *ids, uint32_t *nout, uint64_t *ninput) {
    if (!nout || !ninput) return -1;
    *nout = 0;
    *ninput = 0;
    if (!t || !cols || !desc || !ids || !cap || cap > PLANE_TOPK_LIMIT ||
        nkeys < 1 || nkeys > PLANE_TOPK_KEYS || t->n > UINT32_MAX ||
        t->nw != (t->n + 63) / 64) return -1;
    for (int i = 0; i < nkeys; ++i) {
        int c = cols[i];
        if (c < 0 || c >= t->k || t->width[c] < 0 || t->width[c] > 32 ||
            (desc[i] != 0 && desc[i] != 1)) return -1;
    }
    if (!t->n) return 0;

    uint64_t *masks = calloc(3 * t->nw, sizeof *masks);
    uint32_t *words = malloc(t->nw * sizeof *words);
    plane_topk_row_t *rows = calloc(cap, sizeof *rows);
    if (!masks || !words || !rows) {
        free(rows);
        free(words);
        free(masks);
        return -2;
    }
    uint64_t *active = masks, *chosen = active + t->nw, *value = chosen + t->nw;
    uint64_t remaining = 0;
    size_t nwords = 0;
    for (uint64_t w = 0; w < t->nw; ++w) {
        active[w] = rowmask(t, w) & (filter ? filter[w] : UINT64_MAX);
        if (active[w]) {
            words[nwords++] = (uint32_t)w;
            remaining += __builtin_popcountll(active[w]);
        }
    }
    *ninput = remaining;
    uint32_t need = remaining < cap ? (uint32_t)remaining : cap;

    for (int k = 0; k < nkeys && remaining > need; ++k) {
        int c = cols[k], width = t->width[c];
        for (int bit = 0; bit < width && remaining > need; ++bit) {
            /* High-to-low binary bits are a prefix XOR for MSB storage
             * and a suffix XOR starting at the closing plane for LSB. */
            int physical = t->msb[c] ? bit : width - bit;
            int flag = t->vflag[c];
            const uint64_t *plane = t->V[c][physical];
            uint64_t preferred_count = 0;
            for (size_t i = 0; i < nwords; ++i) {
                uint32_t w = words[i];
                if (flag) {
                    uint64_t virtual_bits[32];
                    int vp = t->msb[c] ? bit : width - 1 - bit;
                    tb_virtual_planes(t, c, t->V[flag - 1][0][w], w, virtual_bits, vp + 1);
                    value[w] = virtual_bits[vp];
                } else {
                    uint64_t x = plane[w];
                    value[w] = bit ? value[w] ^ x : x;
                }
                preferred_count += __builtin_popcountll(active[w] & (desc[k] ? value[w] : ~value[w]));
            }
            if (!preferred_count || preferred_count == remaining) continue;
            int keep_preferred = preferred_count >= need;
            size_t next_words = 0;
            for (size_t i = 0; i < nwords; ++i) {
                uint32_t w = words[i];
                uint64_t preferred = active[w] & (desc[k] ? value[w] : ~value[w]);
                if (keep_preferred) active[w] = preferred;
                else {
                    chosen[w] |= preferred;
                    active[w] ^= preferred;
                }
                if (active[w]) words[next_words++] = w;
            }
            nwords = next_words;
            if (keep_preferred) remaining = preferred_count;
            else {
                remaining -= preferred_count;
                need -= (uint32_t)preferred_count;
            }
        }
    }
    /* The remaining candidates tie on all inspected keys, or all fit.
     * Physical row order chooses deterministically at the tie boundary. */
    for (size_t i = 0; i < nwords && need; ++i) {
        uint32_t w = words[i];
        uint64_t bits = active[w];
        while (bits && need) {
            uint64_t low = bits & -bits;
            chosen[w] |= low;
            bits ^= low;
            --need;
        }
    }

    uint32_t n = 0;
    for (uint64_t w = 0; w < t->nw; ++w) {
        uint64_t mask = chosen[w];
        while (mask) {
            uint64_t row = 64 * w + __builtin_ctzll(mask);
            rows[n].id = row;
            for (int k = 0; k < nkeys; ++k) {
                uint32_t code = plane_topk_code(t, cols[k], row);
                rows[n].key[k] = desc[k] ? ~code : code;
            }
            ++n;
            mask &= mask - 1;
        }
    }
    qsort(rows, n, sizeof *rows, plane_topk_cmp);
    for (uint32_t i = 0; i < n; ++i) ids[i] = rows[i].id;
    *nout = n;
    free(rows);
    free(words);
    free(masks);
    return 0;
}

/* Outputs are aligned with the sorted ids. With no filter, ids may be NULL
 * to return only row numbers, indexed by physical row id.
 * Returns -1 for an unsupported request, -2 for allocation failure and
 * -3 when cap is too small. nout receives the number of qualifying rows. */
int at_plane_order(const atable_t *t, const uint64_t *filter, int32_t nkeys,
                   const int32_t *cols, const int32_t *desc, int32_t npart,
                   uint64_t *ids, uint64_t *rank, uint64_t cap, uint64_t *nout) {
    if (!nout) return -1;
    *nout = 0;
    if (!t || !cols || !desc || nkeys < 1 || nkeys > PLANE_ORDER_KEYS || npart < 0 || npart > nkeys ||
        t->n > UINT32_MAX || (!ids && (!rank || filter)) || (ids && ids == rank)) return -1;
    for (int k = 0; k < nkeys; ++k) {
        int c = cols[k];
        if (c < 0 || c >= t->k || (desc[k] != 0 && desc[k] != 1) || t->vflag[c] ||
            t->width[c] < 0 || t->width[c] > 32) return -1;
    }
    if (!t->n) return 0;
    extern int at_get_threads(void);
    int workers = at_get_threads();
    int useful = (int)((t->n + 65535) / 65536);
    if (workers > useful) workers = useful;
    if (workers > ORDER_WORKERS) workers = ORDER_WORKERS;
    if (workers < 1) workers = 1;

    plane_order_job_t jobs[ORDER_WORKERS];
    for (int k = 0; k < workers; ++k)
        jobs[k] = (plane_order_job_t){.table = t, .filter = filter, .cols = cols, .ids = ids, .rank = rank,
            .npart = npart, .words_begin = t->nw * k / workers, .words_end = t->nw * (k + 1) / workers};
    uint64_t n = t->n;
    if (filter) {
        pool_jobs(workers, plane_order_count, jobs, sizeof *jobs);
        n = 0;
        for (int k = 0; k < workers; ++k) {
            uint64_t count = jobs[k].count;
            jobs[k].count = n;
            n += count;
        }
    }
    *nout = n;
    if (cap < n) return -3;
    if (!n) return 0;
    while (nkeys > npart) {
        _Atomic int inversion = 0;
        for (int k = 0; k < workers; ++k) {
            jobs[k].col = cols[nkeys - 1]; jobs[k].desc = desc[nkeys - 1];
            jobs[k].inversion = &inversion;
        }
        pool_jobs(workers, plane_order_monotone, jobs, sizeof *jobs);
        if (atomic_load_explicit(&inversion, memory_order_relaxed)) break;
        --nkeys;
    }
    if (n > SIZE_MAX / sizeof(uint32_t) || t->nw > SIZE_MAX / 64) return -2;
    uint32_t *src = malloc(n * sizeof *src), *dst = nkeys ? malloc(n * sizeof *dst) : NULL;
    uint8_t *digit = nkeys ? malloc(t->nw * 64) : NULL;
    if (!src || (nkeys && (!dst || !digit))) {
        free(src); free(dst); free(digit);
        return -2;
    }
    for (int k = 0; k < workers; ++k) {
        jobs[k].begin = n * k / workers; jobs[k].end = n * (k + 1) / workers;
        jobs[k].src = src; jobs[k].digit = digit;
    }
    pool_jobs(workers, plane_order_init, jobs, sizeof *jobs);
    for (int key = nkeys - 1; key >= 0; --key) {
        unsigned c = cols[key], width = t->width[c];
        for (unsigned bit = 0; bit < width; bit += PLANE_ORDER_BITS) {
            unsigned bits = width - bit < PLANE_ORDER_BITS ? width - bit : PLANE_ORDER_BITS;
            for (int k = 0; k < workers; ++k) {
                jobs[k].col = c; jobs[k].bit = bit; jobs[k].bits = bits; jobs[k].desc = desc[key];
                jobs[k].src = src; jobs[k].dst = dst;
            }
            pool_jobs(workers, plane_order_digit, jobs, sizeof *jobs);
            pool_jobs(workers, plane_order_histogram, jobs, sizeof *jobs);
            uint32_t prefix = 0;
            unsigned occupied = 0;
            for (unsigned b = 0; b < (1u << bits); ++b) {
                uint32_t before = prefix;
                for (int k = 0; k < workers; ++k) {
                    uint32_t count = jobs[k].offsets[b];
                    jobs[k].offsets[b] = prefix;
                    prefix += count;
                }
                occupied += prefix != before;
            }
            if (occupied > 1) {
                pool_jobs(workers, plane_order_scatter, jobs, sizeof *jobs);
                uint32_t *swap = src; src = dst; dst = swap;
            }
        }
    }
    /* For a sparse mask n can be smaller than the number of workers. The
     * finish pass must avoid reading an empty worker's first row. */
    if ((uint64_t)workers > n) workers = (int)n;
    for (int k = 0; k < workers; ++k) {
        jobs[k].begin = n * k / workers; jobs[k].end = n * (k + 1) / workers;
        jobs[k].src = src;
    }
    pool_jobs(workers, plane_order_finish, jobs, sizeof *jobs);
    free(src); free(dst); free(digit);
    return 0;
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
    if (!t) return NULL;
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
#define TB_CYCLE_FORMAT 2
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
/* v2 keeps the physical V and H directory, followed by one record per plane
 * and 4096-word sector. Payloads follow in the same order without padding.
 * Period zero stores the exact native words. A positive period stores one
 * motif; the final partial word is reconstructed with its occupied-bit mask.
 * The metadata hash covers the directory and records. Each record includes
 * a CRC32C of its payload. */
#define TB_CYCLE_MAX_PERIOD 128
#define TB_MAX_PLANES ((2 * MAXCOL + 2) * NB)
#define TB_FNV_SEED UINT64_C(0xcbf29ce484222325)
typedef uint8_t tb_plane_id_t[4];
typedef struct { uint16_t period, words; uint32_t crc; } tb_cycle_block_t;
_Static_assert(sizeof(tb_cycle_block_t) == 8, "cycle block layout");

static int tb_cycle_directory(const tbhdr_t *h, tb_plane_id_t *dir) {
    int width[MAXCOL], np = 0;
    for (int c = 0; c < h->k; ++c) {
        int b = h->width[c];
        if ((b != -1 && (b < 1 || b >= NB)) || (h->msb[c] != 0 && h->msb[c] != 1))
            return 0;
        width[c] = b == -1 ? 1 : b;          /* physical virtual placeholder */
        for (int j = 0; j <= width[c]; ++j) {
            dir[np][0] = 0; dir[np][1] = c; dir[np][2] = j; dir[np++][3] = 0;
        }
    }
    for (int i = 0; i <= h->k; ++i) {
        int hi = i == h->k ? width[0] : width[i];
        if (i + 1 < h->k && width[i + 1] > hi) hi = width[i + 1];
        for (int j = 1; j <= hi; ++j) {
            dir[np][0] = 1; dir[np][1] = i; dir[np][2] = j; dir[np++][3] = 0;
        }
    }
    return np;
}

static size_t tb_cycle_sectors(uint64_t nw) {
    return (size_t)(nw / TP_SECTOR_WORDS + (nw % TP_SECTOR_WORDS != 0));
}

static size_t tb_cycle_words(uint64_t nw, size_t first) {
    size_t left = (size_t)nw - first;
    return left < TP_SECTOR_WORDS ? left : TP_SECTOR_WORDS;
}

static uint64_t tb_cycle_tail(uint64_t n, size_t first, size_t words) {
    unsigned tail = (unsigned)(n % 64);
    return tail && first + words == n / 64 + 1 ? (UINT64_C(1) << tail) - 1 : 0;
}

static unsigned tb_cycle_motif(const uint64_t *plane, size_t full, uint64_t tail) {
    unsigned bound = full / 2 < TB_CYCLE_MAX_PERIOD ? (unsigned)(full / 2) : TB_CYCLE_MAX_PERIOD;
    size_t prefix = 2 * bound;
    for (unsigned period = 1; period <= bound; ++period) {
        if (memcmp(plane + period, plane, (prefix - period) * sizeof *plane)) continue;
        /* The least period of this prefix divides every bounded period, so
         * an incompatible extension cannot be saved by a larger motif. */
        if (memcmp(plane + prefix, plane + prefix - period, (full - prefix) * sizeof *plane))
            return 0;
        if (tail && plane[full] != (plane[full % period] & tail)) return 0;
        return period;
    }
    return 0;
}

#if !defined(__SSE4_2__) || !defined(__x86_64__)
static uint32_t tb_cycle_crc_table[256];
static pthread_once_t tb_cycle_crc_once = PTHREAD_ONCE_INIT;
static void tb_cycle_crc_init(void) {
    for (unsigned i = 0; i < 256; ++i) {
        uint32_t crc = i;
        for (int j = 0; j < 8; ++j)
            crc = (crc >> 1) ^ ((crc & 1) ? UINT32_C(0x82f63b78) : 0);
        tb_cycle_crc_table[i] = crc;
    }
}
#endif

static uint32_t tb_cycle_crc(const void *payload, size_t words) {
    uint32_t crc = UINT32_MAX;
    const unsigned char *src = payload;
#if defined(__SSE4_2__) && defined(__x86_64__)
    for (size_t i = 0; i < words; ++i) {
        uint64_t word;
        memcpy(&word, src + i * sizeof word, sizeof word);
        crc = (uint32_t)_mm_crc32_u64(crc, word);
    }
#else
    pthread_once(&tb_cycle_crc_once, tb_cycle_crc_init);
    for (size_t i = 0; i < words * sizeof(uint64_t); ++i)
        crc = (crc >> 8) ^ tb_cycle_crc_table[(crc ^ src[i]) & 255];
#endif
    return ~crc;
}

static int tb_cycle_write(const char *path, const tbhdr_t *h, const tb_plane_id_t *dir,
                          const tb_cycle_block_t *blocks, const uint64_t *const *planes,
                          const atable_t *t, uint64_t revision, size_t sectors) {
    char tmp[4096];
    if (snprintf(tmp, sizeof tmp, "%s.tmp.XXXXXX", path) >= (int)sizeof tmp) return -1;
    int fd = mkstemp(tmp);
    if (fd < 0) return -1;
    FILE *f = fdopen(fd, "wb");
    if (!f) { close(fd); unlink(tmp); return -1; }
    size_t nplanes = (size_t)h->nplanes, nblocks = nplanes * sectors;
    int ok = fwrite(h, sizeof *h, 1, f) == 1 &&
             fwrite(dir, sizeof *dir, nplanes, f) == nplanes &&
             fwrite(blocks, sizeof *blocks, nblocks, f) == nblocks;
    for (size_t p = 0; ok && p < nplanes; ++p) {
        for (size_t s = 0; ok && s < sectors; ++s) {
            size_t words = blocks[p * sectors + s].words;
            ok = fwrite(planes[p] + s * TP_SECTOR_WORDS, sizeof(uint64_t), words, f) == words;
        }
    }
    if (ok && (fflush(f) != 0 || fsync(fd) != 0)) ok = 0;
    if (fclose(f) != 0) ok = 0;
    if (!ok || revision != __atomic_load_n(&t->revision, __ATOMIC_ACQUIRE) ||
        rename(tmp, path) != 0) {
        unlink(tmp);
        return -1;
    }
    return 0;
}

/* Keep v1 when no motifs are stored or the complete encoded file does not
 * beat the existing mapped representation. */
int at_save_cycles(const atable_t *t, const char *path) {
    if (!periodic_layout(t) || !path) return -1;
    if (!t->nw) return at_save(t, path);
    uint64_t revision = __atomic_load_n(&t->revision, __ATOMIC_ACQUIRE);
    tbhdr_t h = {0};
    memcpy(h.magic, TB_MAGIC, sizeof h.magic);
    h.format = TB_CYCLE_FORMAT; h.endian = 0x01020304u; h.maxcol = MAXCOL;
    h.n = t->n; h.nw = t->nw; h.k = t->k;
    for (int c = 0; c < t->k; ++c) {
        h.width[c] = t->vflag[c] ? -1 : t->width[c];
        h.msb[c] = t->msb[c];
    }
    tb_plane_id_t dir[TB_MAX_PLANES];
    int np = tb_cycle_directory(&h, dir);
    if (!np) return at_save(t, path);
    const uint64_t *planes[TB_MAX_PLANES];
    int actual = 0;
    for (int c = 0; c < t->k; ++c)
        for (int j = 0; j < NB; ++j) actual += t->V[c][j] != NULL;
    for (int i = 0; i <= t->k; ++i)
        for (int j = 0; j < NB; ++j) actual += t->H[i][j] != NULL;
    if (actual != np) return at_save(t, path);
    for (int p = 0; p < np; ++p) {
        const uint8_t *id = dir[p];
        planes[p] = id[0] ? t->H[id[1]][id[2]] : t->V[id[1]][id[2]];
        if (!planes[p]) return at_save(t, path);
    }
    h.nplanes = np;
    h.planebytes = tb_planebytes(h.nw);
    size_t sectors = tb_cycle_sectors(h.nw);
    size_t dir_bytes = (size_t)np * sizeof *dir;
    size_t old_header = (sizeof h + dir_bytes + 4095) & ~(size_t)4095;
    if (sectors > UINT32_MAX || h.planebytes > (SIZE_MAX - old_header) / (size_t)np ||
        sectors > (SIZE_MAX - sizeof h - dir_bytes) / (size_t)np / sizeof(tb_cycle_block_t))
        return -1;
    size_t blocks_bytes = (size_t)np * sectors * sizeof(tb_cycle_block_t);
    h.hdrlen = sizeof h + dir_bytes + blocks_bytes;
    tb_cycle_block_t *blocks = malloc(blocks_bytes);
    if (!blocks) return -1;
    size_t payload_words = 0;
    int motifs = 0;
    for (int p = 0; p < np; ++p) {
        for (size_t s = 0; s < sectors; ++s) {
            size_t first = s * TP_SECTOR_WORDS;
            size_t words = tb_cycle_words(h.nw, first);
            uint64_t tail = tb_cycle_tail(h.n, first, words);
            unsigned period = tb_cycle_motif(planes[p] + first, words - !!tail, tail);
            tb_cycle_block_t *b = &blocks[(size_t)p * sectors + s];
            b->period = period;
            b->words = period ? period : words;
            payload_words += b->words;
            motifs |= period != 0;
        }
    }
    if (revision != __atomic_load_n(&t->revision, __ATOMIC_ACQUIRE)) { free(blocks); return -1; }
    size_t old_bytes = old_header + (size_t)np * h.planebytes;
    if (!motifs || h.hdrlen >= old_bytes || payload_words > (old_bytes - h.hdrlen - 1) / 8) {
        free(blocks);
        return at_save(t, path);
    }
    for (int p = 0; p < np; ++p) {
        for (size_t s = 0; s < sectors; ++s) {
            tb_cycle_block_t *b = &blocks[(size_t)p * sectors + s];
            b->crc = tb_cycle_crc(planes[p] + s * TP_SECTOR_WORDS, b->words);
        }
    }
    h.hash = tb_fnv(&h, offsetof(tbhdr_t, hash), TB_FNV_SEED);
    h.hash = tb_fnv(dir, dir_bytes, h.hash);
    h.hash = tb_fnv(blocks, blocks_bytes, h.hash);
    int rc = tb_cycle_write(path, &h, dir, blocks, planes, t, revision, sectors);
    free(blocks);
    return rc;
}

static void tb_cycle_expand(uint64_t *dst, size_t words,
                            const tb_cycle_v_t *block, uint64_t tail) {
    size_t filled = block->words;
    memcpy(dst, block->data, filled * sizeof *dst);
    if (!block->period) return;
    size_t full = words - !!tail;
    while (filled < full) {
        size_t step = full - filled < filled ? full - filled : filled;
        memcpy(dst + filled, dst, step * sizeof *dst);
        filled += step;
    }
    if (tail) dst[full] = dst[full % block->period] & tail;
}

static int tb_cycle_value_plane(const tbhdr_t *h, const uint8_t *id) {
    return !id[0] && h->width[id[1]] > 0 && id[2] < h->width[id[1]];
}

static void tb_cycle_store_free(tb_cycle_store_t *store) {
    if (!store) return;
    free(store->data);
    free(store->blocks);
    free(store->ids);
    free(store);
}

static pthread_mutex_t TB_MATERIALIZE_LOCK = PTHREAD_MUTEX_INITIALIZER;

int at_has_dense(const atable_t *t) {
    if (!t) return 0;
    pthread_mutex_lock(&TB_MATERIALIZE_LOCK);
    int dense = !t->cycle_store || t->map;
    for (int c = 0; dense && c < t->k; ++c)
        if (t->cycle_sources[c] && !t->V[c][0]) dense = 0;
    pthread_mutex_unlock(&TB_MATERIALIZE_LOCK);
    return dense;
}

uint64_t at_cycle_packed_bytes(const atable_t *t) {
    return t && t->cycle_store ? t->cycle_store->payload_bytes : 0;
}

/* The caller holds the materialization lock. The immutable encoded
 * representation remains available to direct queries. */
static int tb_materialize_owner(atable_t *t) {
    if (!t->cycle_store || t->map) return 0;
    const tb_cycle_store_t *store = t->cycle_store;
    size_t maplen = (size_t)store->nplanes * store->planebytes;
    void *dense = mmap(NULL, maplen, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (dense == MAP_FAILED) return -1;
    for (size_t p = 0; p < store->nplanes; ++p) {
        const uint8_t *id = store->ids[p];
        uint64_t *plane = (uint64_t *)((char *)dense + p * store->planebytes);
        for (size_t s = 0; s < t->cycle_sectors; ++s) {
            size_t first = s * TP_SECTOR_WORDS;
            size_t words = tb_cycle_words(t->nw, first);
            const tb_cycle_v_t *block = &store->blocks[p * t->cycle_sectors + s];
            tb_cycle_expand(plane + first, words, block, tb_cycle_tail(t->n, first, words));
        }
        if (id[0]) t->H[id[1]][id[2]] = plane; else t->V[id[1]][id[2]] = plane;
    }
    t->map = dense;
    t->maplen = maplen;
    at_faces(t);
    return 0;
}

/* A view keeps its header identity on fallback. Its flattened sources own
 * both packed and dense storage, so rebuilding aliases never copies values. */
int at_materialize(atable_t *t) {
    if (!t) return -1;
    pthread_mutex_lock(&TB_MATERIALIZE_LOCK);
    int rc = tb_materialize_owner(t);
    for (int c = 0; !rc && c < t->k; ++c) {
        atable_t *owner = (atable_t *)t->cycle_sources[c];
        if (!owner || t->V[c][0]) continue;
        rc = tb_materialize_owner(owner);
        if (!rc) memcpy(t->V[c], owner->V[t->cycle_columns[c]], sizeof t->V[c]);
    }
    pthread_mutex_unlock(&TB_MATERIALIZE_LOCK);
    return rc;
}

/* Validate the immutable encoded image while loading. It owns every
 * independent V/H edge and suffices to materialize a writable torus later,
 * even after the original file has been removed. */
static atable_t *tb_cycle_load(const void *m, size_t bytes, int *err) {
    tbhdr_t header;
    memcpy(&header, m, sizeof header);
    const tbhdr_t *h = &header;
    *err = 4;
    if (memcmp(h->magic, TB_MAGIC, 8) || h->format != TB_CYCLE_FORMAT ||
        h->endian != 0x01020304u || h->k < 1 || h->k > MAXCOL ||
        !h->nw || h->nw != h->n / 64 + (h->n % 64 != 0) || h->nw > SIZE_MAX / 8 ||
        h->pad || h->maxcol < h->k || h->maxcol > TB_SAVED_COLS ||
        h->planebytes != tb_planebytes(h->nw)) return NULL;
    tb_plane_id_t dir[TB_MAX_PLANES];
    int np = tb_cycle_directory(h, dir);
    if (!np || np != h->nplanes || h->planebytes > SIZE_MAX / (size_t)np) return NULL;
    size_t sectors = tb_cycle_sectors(h->nw);
    size_t dir_bytes = (size_t)np * sizeof *dir;
    if (sectors > UINT32_MAX ||
        sectors > (SIZE_MAX - sizeof *h - dir_bytes) / (size_t)np / sizeof(tb_cycle_block_t))
        return NULL;
    size_t metadata = sizeof *h + dir_bytes + (size_t)np * sectors * sizeof(tb_cycle_block_t);
    if (h->hdrlen != metadata) return NULL;
    if (metadata > bytes) { *err = 5; return NULL; }
    const unsigned char *file = m;
    unsigned char *index = malloc(metadata - sizeof *h);
    if (!index) { *err = 1; return NULL; }
    memcpy(index, file + sizeof *h, metadata - sizeof *h);
    const unsigned char *blocks = index + dir_bytes;
    atable_t *t = NULL;
    if (memcmp(dir, index, dir_bytes) ||
        tb_fnv(index, metadata - sizeof *h,
               tb_fnv(h, offsetof(tbhdr_t, hash), TB_FNV_SEED)) != h->hash) goto fail;

    /* Bound all spans and allocations using the checksummed metadata.
     * The payload CRCs below apply to the exact owned bytes we expand. */
    size_t cursor = metadata;
    for (int p = 0; p < np; ++p) {
        for (size_t s = 0; s < sectors; ++s) {
            tb_cycle_block_t b;
            memcpy(&b, blocks + ((size_t)p * sectors + s) * sizeof b, sizeof b);
            size_t first = s * TP_SECTOR_WORDS;
            size_t words = tb_cycle_words(h->nw, first);
            uint64_t tail = tb_cycle_tail(h->n, first, words);
            if (b.period > TB_CYCLE_MAX_PERIOD ||
                (b.period && (size_t)b.period * 2 > words - !!tail) ||
                b.words != (b.period ? b.period : words)) goto fail;
            size_t len = (size_t)b.words * sizeof(uint64_t);
            if (len > bytes - cursor) { *err = 5; goto fail; }
            cursor += len;
        }
    }
    if (cursor != bytes || (size_t)np > SIZE_MAX / sizeof(tb_cycle_v_t) / sectors) goto fail;

    *err = 1;
    t = calloc(1, sizeof *t);
    if (!t) goto fail;
    t->n = h->n; t->nw = h->nw; t->k = h->k;
    t->cycle_sectors = (uint32_t)sectors;
    t->cycle_periods = calloc(sectors * (size_t)t->k, sizeof *t->cycle_periods);
    t->cycle_store = calloc(1, sizeof *t->cycle_store);
    if (!t->cycle_periods || !t->cycle_store) goto fail;
    tb_cycle_store_t *store = t->cycle_store;
    for (int c = 0; c < t->k; ++c) store->first[c] = UINT16_MAX;
    store->nplanes = (uint16_t)np;
    store->planebytes = (size_t)h->planebytes;
    store->payload_bytes = bytes - metadata;
    store->ids = malloc(dir_bytes);
    store->blocks = malloc((size_t)np * sectors * sizeof *store->blocks);
    store->data = malloc(store->payload_bytes);
    if (!store->ids || !store->blocks || !store->data) goto fail;
    memcpy(store->ids, dir, dir_bytes);
    memcpy(store->data, file + metadata, store->payload_bytes);
    for (int c = 0; c < t->k; ++c) {
        t->width[c] = h->width[c] < 0 ? 1 : h->width[c];
        t->msb[c] = h->msb[c];
    }
    size_t offset = 0;
    for (int p = 0; p < np; ++p) {
        const uint8_t *id = dir[p];
        unsigned col = id[1], bit = id[2];
        int value = tb_cycle_value_plane(h, id);
        if (value && bit == 0) store->first[col] = (uint16_t)p;
        for (size_t s = 0; s < sectors; ++s) {
            size_t entry = (size_t)p * sectors + s;
            tb_cycle_block_t b;
            memcpy(&b, blocks + entry * sizeof b, sizeof b);
            const uint64_t *payload = store->data + offset;
            if (tb_cycle_crc(payload, b.words) != b.crc) { *err = 4; goto fail; }
            store->blocks[entry] = (tb_cycle_v_t){payload, b.period, b.words};
            offset += b.words;
            if (value) {
                uint16_t *period = &t->cycle_periods[s * (size_t)t->k + col];
                unsigned next = bit == 0 ? b.period :
                                (*period && b.period ? periodic_lcm(*period, b.period) : 0);
                *period = next <= TP_MAX_CYCLE_WORDS ? (uint16_t)next : 0;
            }
        }
    }
    free(index);
    t->bytes = (uint64_t)np * t->nw * 8;
    *err = 0;
    return t;
fail:
    free(index);
    if (t) {
        free(t->cycle_periods);
        tb_cycle_store_free(t->cycle_store);
        free(t);
    }
    return NULL;
}
/* err: 0 ok, 1 cannot open/map, 2 not a torble file, 3 format/version or build mismatch, 4 header damaged, 5 truncated */
static atable_t *tb_load(const char *path, int *err, int lazy) {
    int fd = open(path, O_RDONLY); if (fd < 0) { *err = 1; return NULL; }
    struct stat st;
    if (fstat(fd, &st) != 0) { close(fd); *err = 1; return NULL; }
    if (st.st_size < (off_t)sizeof(tbhdr_t)) { close(fd); *err = 5; return NULL; }
    if ((uint64_t)st.st_size > SIZE_MAX) { close(fd); *err = 1; return NULL; }
    void *m = mmap(NULL, (size_t)st.st_size, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0); close(fd);
    if (m == MAP_FAILED) { *err = 1; return NULL; }
    tbhdr_t *h = m; *err = 0;
    if (!memcmp(h->magic, TB_MAGIC, 8) && h->format == TB_CYCLE_FORMAT &&
        h->endian == 0x01020304u && h->k >= 1 && h->k <= MAXCOL) {
        atable_t *t = tb_cycle_load(m, (size_t)st.st_size, err);
        munmap(m, (size_t)st.st_size);
        if (!t || lazy) return t;
        int status = at_materialize(t);
        tb_cycle_store_free(t->cycle_store);
        t->cycle_store = NULL;
        if (status) { free(t->cycle_periods); free(t); *err = 1; return NULL; }
        return t;
    }
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
atable_t *at_load(const char *path, int *err) { return tb_load(path, err, 0); }
atable_t *at_load_cycles_lazy(const char *path, int *err) { return tb_load(path, err, 1); }

/* release a torble's storage, its planes (built) or its mapping (loaded), and leave the header as an empty torble of a
 * new revision: an object that kept the header's address (a relationship catalog decodes through its own pointer, after
 * checking the revision) then finds a changed, empty table instead of freed memory. The header goes with at_free. */
void at_release(atable_t *t) {
    if (!t) return;
    if (t->map) munmap(t->map, (size_t)t->maplen);
    else {
        for (int c = 0; c < t->k; c++) for (int j = 0; j < NB; j++) free(t->V[c][j]);
        for (int i = 0; i <= t->k; i++) for (int j = 0; j < NB; j++) free(t->H[i][j]);
    }
    t->map = NULL; t->maplen = 0;
    free(t->cycle_periods);
    t->cycle_periods = NULL; t->cycle_sectors = 0;
    tb_cycle_store_free(t->cycle_store);
    t->cycle_store = NULL;
    memset(t->V, 0, sizeof t->V); memset(t->H, 0, sizeof t->H); t->nfaces = 0;
    t->n = t->nw = 0; t->bytes = 0;
    for (int c = 0; c < MAXCOL; c++) { t->vflag[c] = 0; t->vside[c] = NULL; t->vrank[c] = NULL; }   /* (sidecars: freed apart) */
    t->revision++;
}
/* release a torble: its storage, then its header */
void at_free(atable_t *t) {
    if (!t) return;
    at_release(t);
    free(t);
}
/* make column c virtual: its values are in sidecar `side` for the rows flagged in column flagcol (rank directory) */
void at_set_virtual(atable_t *t, int c, int flagcol, const atable_t *side, const uint32_t *rank) {
    /* Installing a sidecar does not touch physical edges. Retain any
     * current proofs for other columns; virtual columns never use them. */
    int cycles_current = t->cycle_periods && t->cycle_revision == t->revision;
    t->revision++;
    t->vflag[c] = flagcol + 1; t->vside[c] = side; t->vrank[c] = rank;
    t->width[c] = side->width[0]; t->msb[c] = side->msb[0];
    if (cycles_current) t->cycle_revision = t->revision;
}
/* A view: a torble header whose column i is column col[i] of torble src[i], the sources row-aligned (the same rows in
 * the same order: the parts of a table wider than one torble). Its value planes are the sources' own (nothing is
 * copied), so a query reads a side part's planes at the same 64-row word as the main part's: the join of the parts on
 * the row id is the identity substitution. It has no torus of its own (no H planes, no faces): the integrity checks
 * belong to the sources. The view is read-only and shares its sources' period proof owners. A self-extending
 * column needs its flag column in the view. NULL: no column or more than
 * MAXCOL, sources of different lengths, a column out of range, a flag column missing, or no memory. Freed with
 * at_view_free (the header only), before its sources. */
static atable_t *tb_view(int k, const atable_t *const *src, const int32_t *col) {
    if (k < 1 || k > MAXCOL || !src || !col || !src[0]) return NULL;
    atable_t *t = (atable_t *)calloc(1, sizeof *t);
    if (!t) return NULL;
    t->n = src[0]->n; t->nw = src[0]->nw; t->k = k;
    for (int i = 0; i < k; i++) {
        const atable_t *s = src[i]; int c = col[i];
        if (!s || s->n != t->n || c < 0 || c >= s->k) { free(t); return NULL; }
        t->width[i] = s->width[c]; t->msb[i] = s->msb[c];
        for (int j = 0; j < NB; j++) t->V[i][j] = s->V[c][j];
        if (s->cycle_sources[c]) {
            t->cycle_sources[i] = s->cycle_sources[c];
            t->cycle_columns[i] = s->cycle_columns[c];
        } else if (s->cycle_periods) {
            t->cycle_sources[i] = s;
            t->cycle_columns[i] = (uint8_t)c;
        }
        if (s->vflag[c]) {
            int f = s->vflag[c] - 1, fi = -1;
            for (int m = 0; m < k && fi < 0; m++) if (src[m] == s && col[m] == f) fi = m;
            if (fi < 0) { free(t); return NULL; }
            t->vflag[i] = fi + 1; t->vside[i] = s->vside[c]; t->vrank[i] = s->vrank[c];
        }
    }
    return t;
}
atable_t *at_view(int k, const atable_t *const *src, const int32_t *col) {
    pthread_mutex_lock(&TB_MATERIALIZE_LOCK);
    atable_t *t = tb_view(k, src, col);
    pthread_mutex_unlock(&TB_MATERIALIZE_LOCK);
    return t;
}
void at_view_free(atable_t *t) { free(t); }
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
/* the stored planes, counted (bytes / (nw * 8) divides by zero on an empty torble, whose planes hold no words) */
int at_nplanes(const atable_t *t) {
    int np = 0;
    for (int c = 0; c < t->k; c++) for (int j = 0; j < NB; j++) np += t->V[c][j] != NULL;
    for (int i = 0; i <= t->k; i++) for (int j = 0; j < NB; j++) np += t->H[i][j] != NULL;
    return np;
}

/* XOR a logical value mask into each selected physical row. Validate the
 * whole batch first. A potential delta updates its two incident H edges
 * and the difference of adjacent potentials on V. Callers serialize writes
 * and own logical key/membership constraints and any external caches. */
int at_xor_words(atable_t *t, uint64_t npatch, const int32_t *cols,
                 const uint64_t *words, const uint64_t *rows, const uint32_t *value_xor) {
    if (!t || t->k < 1 || t->k > MAXCOL || !t->nfaces ||
        t->nw != t->n / 64 + (t->n % 64 != 0)) return -1;
    if (!npatch) return 0;
    if (!cols || !words || !rows || !value_xor || npatch > SIZE_MAX / sizeof *words) return -1;

    uint32_t blocked = 0, validated = 0;
    for (int c = 0; c < t->k; ++c) {
        int flag = t->vflag[c];
        if (!flag) continue;
        if (flag < 1 || flag > t->k) return -1;
        blocked |= (UINT32_C(1) << c) | (UINT32_C(1) << (flag - 1));
    }

    int changed = 0;
    for (uint64_t i = 0; i < npatch; ++i) {
        int c = cols[i];
        uint64_t w = words[i];
        if (c < 0 || c >= t->k || w >= t->nw || (rows[i] & ~rowmask(t, w))) return -1;
        uint32_t col_bit = UINT32_C(1) << c;
        if (blocked & col_bit) return -1;
        int width = t->width[c];
        if (width < 0 || width > 32 || (width < 32 && value_xor[i] >> width)) return -1;
        if (!(validated & col_bit)) {
            int left = c ? c - 1 : t->k;
            for (int j = 0; j <= width; ++j) {
                if (!t->V[c][j] || (j && (!t->H[left][j] || !t->H[c][j]))) return -1;
            }
            validated |= col_bit;
        }
        changed |= rows[i] != 0 && value_xor[i] != 0;
    }
    if (!changed) return 0;
    if (t->revision == UINT64_MAX) return -1;

    for (uint64_t i = 0; i < npatch; ++i) {
        uint64_t mask = rows[i];
        uint32_t code = value_xor[i];
        if (!mask || !code) continue;
        int c = cols[i], left = c ? c - 1 : t->k;
        uint64_t w = words[i];
        uint64_t delta = t->msb[c] ? revb(code, t->width[c]) : code;
        uint64_t h = delta << 1;
        for (uint64_t v = delta ^ h; v; v &= v - 1)
            t->V[c][__builtin_ctzll(v)][w] ^= mask;
        for (; h; h &= h - 1) {
            int j = __builtin_ctzll(h);
            t->H[left][j][w] ^= mask;
            t->H[c][j][w] ^= mask;
        }
    }
    ++t->revision;
    return 0;
}

/* fault injection: flip row r in the p-th stored plane (a row past the table is not stored: nothing to flip) */
void at_flip(atable_t *t, int p, uint64_t r) {
    if (r >= t->n) return;
    t->revision++;
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

/* The packed directory names the same independent edges as at_faces.
 * Store one-based plane indices so an absent edge is the zero operand. */
typedef struct { uint16_t edge[4]; unsigned n; } tb_cycle_face_t;
static unsigned tb_cycle_faces(const atable_t *t, tb_cycle_face_t *faces) {
    uint16_t v[MAXCOL][NB] = {{0}}, h[MAXCOL + 1][NB] = {{0}};
    const tb_cycle_store_t *store = t->cycle_store;
    for (unsigned p = 0; p < store->nplanes; ++p) {
        const uint8_t *id = store->ids[p];
        (id[0] ? h : v)[id[1]][id[2]] = (uint16_t)(p + 1);
    }
    unsigned nfaces = 0;
    for (int row = 0; row <= t->k; ++row) {
        int next = row == t->k ? 0 : row + 1;
        for (int j = 0; j < NB; ++j) {
            uint16_t edges[4] = {
                h[row][j], j + 1 < NB ? h[row][j + 1] : 0,
                row < t->k ? v[row][j] : 0, next < t->k ? v[next][j] : 0
            };
            tb_cycle_face_t face = {{0}, 0};
            for (unsigned i = 0; i < 4; ++i)
                if (edges[i]) face.edge[face.n++] = (uint16_t)(edges[i] - 1);
            if (face.n) faces[nfaces++] = face;
        }
    }
    return nfaces;
}

typedef struct { uint64_t acc[TP_SECTOR_WORDS]; unsigned words; int dirty; } tb_cycle_faults_t;
static void tb_cycle_fault(tb_cycle_faults_t *faults, uint64_t mask, unsigned first,
                           unsigned stop, unsigned step) {
    if (!mask) return;
    if (!faults->dirty) {
        memset(faults->acc, 0, (size_t)faults->words * sizeof *faults->acc);
        faults->dirty = 1;
    }
    for (unsigned w = first; w < stop; w += step) faults->acc[w] |= mask;
}

static inline uint64_t tb_cycle_face_word(const tb_cycle_v_t *const *edges, unsigned n, unsigned w) {
    uint64_t mask = 0;
    for (unsigned i = 0; i < n; ++i) mask ^= tb_cycle_v_word(edges[i], w);
    return mask;
}

/* Literal/constant plaquettes are common in mixed snapshots. XOR/OR the
 * spans as contiguous vectors before paying to enumerate any bad word. */
static uint64_t tb_cycle_raw_syndrome(const uint64_t *const *raw, unsigned n,
                                      uint64_t seed, unsigned words) {
    uint64_t any = 0;
    switch (n) {
    case 1:
        for (unsigned w = 0; w < words; ++w) any |= seed ^ raw[0][w];
        break;
    case 2:
        for (unsigned w = 0; w < words; ++w) any |= seed ^ raw[0][w] ^ raw[1][w];
        break;
    case 3:
        for (unsigned w = 0; w < words; ++w) any |= seed ^ raw[0][w] ^ raw[1][w] ^ raw[2][w];
        break;
    default:
        for (unsigned w = 0; w < words; ++w) any |= raw[0][w] ^ raw[1][w] ^ raw[2][w] ^ raw[3][w];
        break;
    }
    return any;
}

static void tb_cycle_face_syndrome(const tb_cycle_v_t *const *edges, unsigned n,
                                  unsigned full, uint64_t tail, tb_cycle_faults_t *faults) {
    const uint64_t *raw[4];
    unsigned nraw = 0, varying = 0, cycle = 1;
    uint64_t seed = 0;
    for (unsigned i = 0; i < n; ++i) {
        unsigned period = edges[i]->period;
        if (!period) raw[nraw++] = edges[i]->data;
        else if (period == 1) seed ^= edges[i]->data[0];
        else {
            ++varying;
            if (cycle <= full) cycle = periodic_lcm(cycle, period);
        }
    }
    if (!nraw && cycle <= full) {
        for (unsigned w = 0; w < cycle; ++w)
            tb_cycle_fault(faults, tb_cycle_face_word(edges, n, w), w, full, cycle);
    } else if (full && (varying || tb_cycle_raw_syndrome(raw, nraw, seed, full))) {
        for (unsigned w = 0; w < full; ++w)
            tb_cycle_fault(faults, tb_cycle_face_word(edges, n, w), w, w + 1, 1);
    }
    if (tail) {
        uint64_t mask = tb_cycle_face_word(edges, n, full) & tail;
        tb_cycle_fault(faults, mask, full, full + 1, 1);
    }
}

static uint64_t tb_cycle_validate_sector(const atable_t *t, const tb_cycle_face_t *faces,
                                         unsigned nfaces, uint64_t sector,
                                         uint64_t *bad, uint64_t maxbad) {
    size_t first = (size_t)sector * TP_SECTOR_WORDS;
    unsigned words = (unsigned)tb_cycle_words(t->nw, first);
    uint64_t tail = tb_cycle_tail(t->n, first, words);
    unsigned full = words - !!tail;
    tb_cycle_faults_t faults;
    faults.words = words;
    faults.dirty = 0;
    const tb_cycle_store_t *store = t->cycle_store;
    for (unsigned f = 0; f < nfaces; ++f) {
        const tb_cycle_v_t *edges[4];
        for (unsigned i = 0; i < faces[f].n; ++i)
            edges[i] = &store->blocks[(size_t)faces[f].edge[i] * t->cycle_sectors + sector];
        tb_cycle_face_syndrome(edges, faces[f].n, full, tail, &faults);
    }
    if (!faults.dirty) return 0;
    uint64_t count = 0;
    for (unsigned w = 0; w < words; ++w) {
        uint64_t x = faults.acc[w];
        while (x && bad && count < maxbad) {
            bad[count++] = (first + w) * 64 + (unsigned)__builtin_ctzll(x);
            x &= x - 1;
        }
        count += (unsigned)__builtin_popcountll(x);
    }
    return count;
}

typedef struct {
    const atable_t *t;
    const tb_cycle_face_t *faces;
    unsigned nfaces;
    _Atomic uint64_t *next;
    uint64_t count;
} tb_cycle_validate_job_t;

static void *tb_cycle_validate_worker(void *arg) {
    tb_cycle_validate_job_t *job = arg;
    for (;;) {
        uint64_t s = atomic_fetch_add(job->next, 1);
        if (s >= job->t->cycle_sectors) return NULL;
        job->count += tb_cycle_validate_sector(job->t, job->faces, job->nfaces, s, NULL, 0);
    }
}

/* The packed image describes the original edges, including closing V
 * edges and H. It cannot validate later changes to a materialized torus. */
int at_cycle_validate(const atable_t *t, uint64_t *bad, uint64_t maxbad, uint64_t *nbad) {
    if (!t || !t->cycle_store || !t->cycle_sectors || !nbad) return -1;
    uint64_t revision = __atomic_load_n(&t->revision, __ATOMIC_ACQUIRE);
    if (revision != t->cycle_revision) return -1;
    tb_cycle_face_t faces[(MAXCOL + 1) * NB];
    unsigned nfaces = tb_cycle_faces(t, faces);
    int threads = nthreads();
    if ((unsigned)threads > t->cycle_sectors) threads = (int)t->cycle_sectors;
    _Atomic uint64_t next = 0;
    tb_cycle_validate_job_t jobs[64];
    for (int i = 0; i < threads; ++i)
        jobs[i] = (tb_cycle_validate_job_t){t, faces, nfaces, &next, 0};
    pool_jobs(threads, tb_cycle_validate_worker, jobs, sizeof *jobs);
    uint64_t count = 0;
    for (int i = 0; i < threads; ++i) count += jobs[i].count;
    if (count && bad && maxbad) {
        uint64_t listed = 0;
        for (uint64_t s = 0; s < t->cycle_sectors && listed < maxbad; ++s)
            listed += tb_cycle_validate_sector(t, faces, nfaces, s, bad + listed, maxbad - listed);
    }
    if (revision != __atomic_load_n(&t->revision, __ATOMIC_ACQUIRE)) return -1;
    *nbad = count;
    return 0;
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
/* Sparse top groups. Transpose only the requested planes of each word.
 * Partition compact records as they are decoded, then reduce each shard
 * independently. Only each worker's bounded top groups reach the merge. */
enum { AT_GROUP_RAW, AT_GROUP_MINUTE, AT_GROUP_EPOCH_MINUTE, AT_GROUP_PROJECT, AT_GROUP_COORDINATE };
typedef struct {
    int32_t low, high, shift, transform;
    int64_t offset;
} at_group_key_t;

enum { GROUP_SHARDS = 256, GROUP_BLOCK = 128, GROUP_LIMIT = 1024, GROUP_BUNDLES = 6, GROUP_KEYS = 5, GROUP_PREFIX_LIMIT = 16 };
typedef struct { uint64_t low, high; uint32_t sum[2]; } group_record_t;
typedef struct group_block_s {
    struct group_block_s *next;
    unsigned n;
    group_record_t records[GROUP_BLOCK];
} group_block_t;
typedef struct { group_block_t *blocks; uint64_t n; } group_shard_t;
typedef struct { uint64_t low, high, sum[2]; uint32_t count, distinct; } group_slot_t;
typedef struct {
    const atable_t *t;
    const uint64_t *filter;
    const at_group_key_t *key;
    const int32_t *sum;
    unsigned nkey, nsum, bundles, limit, rank, distinct_width;
    size_t distinct_words;
    const uint32_t *projection;
    uint64_t projection_length, projection_mask;
    int col[MAXCOL], ncol, bundle[MAXCOL], offset[MAXCOL];
    unsigned key_width[GROUP_KEYS + 1];
    int64_t minute_offset[GROUP_KEYS + 1];
    int workers;
    _Atomic unsigned next;
    _Atomic int failed;
} group_plan_t;
typedef struct group_worker_s {
    group_plan_t *plan;
    struct group_worker_s *all;
    uint64_t begin, end;
    group_shard_t shard[GROUP_SHARDS];
    group_slot_t *heap;
    unsigned nheap;
    uint64_t distinct_total, *distinct_bits;
} group_worker_t;

static inline uint64_t group_hash(uint64_t lo, uint64_t hi) {
    uint64_t x = lo ^ (hi * UINT64_C(0xc2b2ae3d27d4eb4f));
    x = (x ^ (x >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)) * UINT64_C(0x94d049bb133111eb);
    return x ^ (x >> 31);
}

static int group_add_column(group_plan_t *p, int c, unsigned used[GROUP_BUNDLES]) {
    const atable_t *t = p->t;
    if (c < 0 || c >= t->k || t->width[c] < 1 || t->width[c] > 32) return 0;
    if (p->bundle[c] >= 0) return 1;
    int flag = t->vflag[c];
    if (flag && (flag < 1 || flag > t->k || !t->V[flag - 1][0] || !t->vside[c] || !t->vrank[c])) return 0;
    if (!flag) for (int j = 0; j < t->width[c]; ++j) if (!t->V[c][j]) return 0;
    unsigned b = 0;
    while (b < GROUP_BUNDLES && used[b] + t->width[c] > 64) ++b;
    if (b == GROUP_BUNDLES) return 0;
    p->col[p->ncol++] = c;
    p->bundle[c] = b;
    p->offset[c] = used[b];
    used[b] += t->width[c];
    if (p->bundles <= b) p->bundles = b + 1;
    return 1;
}

static void group_word(const group_plan_t *p, uint64_t w, uint64_t rows[GROUP_BUNDLES][64]) {
    const atable_t *t = p->t;
    memset(rows, 0, p->bundles * sizeof rows[0]);
    for (int i = 0; i < p->ncol; ++i) {
        int c = p->col[i], width = t->width[c];
        uint64_t *planes = rows[p->bundle[c]] + p->offset[c];
        int start = t->msb[c] ? width - 1 : 0;
        int step = t->msb[c] ? -1 : 1;
        if (t->vflag[c]) {
            uint64_t values[32];
            tb_virtual_planes(t, c, t->V[t->vflag[c] - 1][0][w], w, values, width);
            for (int b = 0; b < width; ++b) planes[start + b * step] = values[b];
        } else {
            uint64_t value = 0;
            for (int b = 0; b < width; ++b) {
                value ^= t->V[c][b][w];
                planes[start + b * step] = value;
            }
        }
    }
    for (unsigned b = 0; b < p->bundles; ++b) transpose64(rows[b]);
}

static inline uint32_t group_value(const group_plan_t *p, uint64_t rows[GROUP_BUNDLES][64], int c, unsigned r) {
    return (uint32_t)((rows[p->bundle[c]][r] >> p->offset[c]) & ((UINT64_C(1) << p->t->width[c]) - 1));
}

static inline uint64_t group_minute(uint64_t code, int64_t offset) {
    const uint64_t hour = UINT64_C(3600000000);
    int64_t rem = (int64_t)(code % hour) + offset;
    if (rem < 0) rem += (int64_t)hour;
    if (rem >= (int64_t)hour) rem -= (int64_t)hour;
    return (uint64_t)rem / UINT64_C(60000000);
}

static inline u128 group_pack(group_plan_t *p, uint64_t rows[GROUP_BUNDLES][64], unsigned r) {
    u128 packed = 0;
    for (unsigned i = 0; i < p->nkey + !!p->distinct_width; ++i) {
        const at_group_key_t *k = p->key + i;
        uint64_t code = group_value(p, rows, k->low, r);
        if (k->high >= 0) code |= (uint64_t)group_value(p, rows, k->high, r) << k->shift;
        if (k->transform == AT_GROUP_MINUTE) code = group_minute(code, p->minute_offset[i]);
        else if (k->transform == AT_GROUP_EPOCH_MINUTE)
            code = code / UINT64_C(60000000)
                   + (code % UINT64_C(60000000) + p->minute_offset[i]) / UINT64_C(60000000);
        else if (k->transform == AT_GROUP_PROJECT) {
            uint64_t index = code & p->projection_mask;
            if (index >= p->projection_length || (uint64_t)p->projection[index] >= (UINT64_C(1) << p->key_width[i])) {
                atomic_store_explicit(&p->failed, 2, memory_order_relaxed);
                return 0;
            }
            code = p->projection[index];
        } else if (k->transform == AT_GROUP_COORDINATE) {
            code = (code >> k->shift) & ((UINT64_C(1) << k->offset) - 1);
        }
        packed = (packed << p->key_width[i]) | code;
    }
    return packed;
}

static void *group_partition(void *arg) {
    group_worker_t *j = arg;
    group_plan_t *p = j->plan;
    for (uint64_t w = j->begin; w < j->end; ++w) {
        if ((w & 255) == 0 && atomic_load_explicit(&p->failed, memory_order_relaxed)) break;
        uint64_t mask = rowmask(p->t, w) & (p->filter ? p->filter[w] : UINT64_MAX);
        if (!mask) continue;
        uint64_t rows[GROUP_BUNDLES][64];
        group_word(p, w, rows);
        while (mask) {
            unsigned r = __builtin_ctzll(mask);
            mask &= mask - 1;
            u128 packed = group_pack(p, rows, r);
            if (p->projection && atomic_load_explicit(&p->failed, memory_order_relaxed)) return NULL;
            uint64_t lo = (uint64_t)packed, hi = (uint64_t)(packed >> 64);
            if (j->distinct_bits) {
                j->distinct_bits[lo >> 6] |= UINT64_C(1) << (lo & 63);
                continue;
            }
            u128 shard_key = p->nkey ? packed >> p->distinct_width : packed;
            unsigned shard = group_hash((uint64_t)shard_key, (uint64_t)(shard_key >> 64)) & (GROUP_SHARDS - 1);
            group_shard_t *s = j->shard + shard;
            group_block_t *b = s->blocks;
            if (!b || b->n == GROUP_BLOCK) {
                b = malloc(sizeof *b);
                if (!b) {
                    atomic_store_explicit(&p->failed, 1, memory_order_relaxed);
                    return NULL;
                }
                b->next = s->blocks;
                b->n = 0;
                s->blocks = b;
            }
            group_record_t *out = b->records + b->n++;
            *out = (group_record_t){.low = lo, .high = hi};
            for (unsigned i = 0; i < p->nsum; ++i) out->sum[i] = group_value(p, rows, p->sum[i], r);
            ++s->n;
        }
    }
    return NULL;
}

static inline int group_better(const group_slot_t *a, const group_slot_t *b, unsigned rank) {
    if (rank == 0 && a->count != b->count) return a->count > b->count;
    if (rank == 1 && a->distinct != b->distinct) return a->distinct > b->distinct;
    return a->high == b->high ? a->low < b->low : a->high < b->high;
}

static void group_keep(group_worker_t *j, group_slot_t s) {
    group_slot_t *heap = j->heap;
    if (j->nheap < j->plan->limit) {
        unsigned i = j->nheap++;
        while (i && group_better(&heap[(i - 1) / 2], &s, j->plan->rank)) {
            heap[i] = heap[(i - 1) / 2];
            i = (i - 1) / 2;
        }
        heap[i] = s;
    } else if (group_better(&s, &heap[0], j->plan->rank)) {
        unsigned i = 0, child = 1;
        while (child < j->nheap) {
            if (child + 1 < j->nheap && group_better(&heap[child], &heap[child + 1], j->plan->rank)) ++child;
            if (!group_better(&s, &heap[child], j->plan->rank)) break;
            heap[i] = heap[child];
            i = child;
            child = 2 * i + 1;
        }
        heap[i] = s;
    }
}

static group_slot_t *group_slot(group_slot_t *slots, uint64_t capacity, uint64_t low, uint64_t high) {
    uint64_t at = (group_hash(low, high) >> 8) & (capacity - 1);
    while (slots[at].count && (slots[at].low != low || slots[at].high != high))
        at = (at + 1) & (capacity - 1);
    return slots + at;
}

static group_slot_t *group_grow(group_slot_t *old, uint64_t capacity) {
    if (capacity > SIZE_MAX / sizeof *old / 2) return NULL;
    group_slot_t *slots = calloc((size_t)capacity * 2, sizeof *slots);
    if (!slots) return NULL;
    for (uint64_t i = 0; i < capacity; ++i) if (old[i].count)
        *group_slot(slots, capacity * 2, old[i].low, old[i].high) = old[i];
    free(old);
    return slots;
}

typedef struct {
    group_slot_t *slots;
    uint64_t capacity, occupied;
} group_hash_t;

static int group_hash_init(group_hash_t *h, uint64_t rows) {
    h->capacity = 2;
    while (h->capacity < rows + rows / 2 + 1 && h->capacity < 65536) h->capacity <<= 1;
    h->slots = calloc((size_t)h->capacity, sizeof *h->slots);
    return h->slots != NULL;
}

static group_slot_t *group_hash_entry(group_hash_t *h, uint64_t low, uint64_t high) {
    group_slot_t *slot = group_slot(h->slots, h->capacity, low, high);
    if (!slot->count) {
        if (h->occupied >= h->capacity * 2 / 3) {
            group_slot_t *bigger = group_grow(h->slots, h->capacity);
            if (!bigger) return NULL;
            h->slots = bigger;
            h->capacity *= 2;
            slot = group_slot(h->slots, h->capacity, low, high);
        }
        slot->low = low;
        slot->high = high;
        ++h->occupied;
    }
    return slot;
}

static int group_reduce_shard(group_worker_t *j, unsigned part, uint64_t rows) {
    group_plan_t *p = j->plan;
    group_hash_t groups = {0}, pairs = {0};
    int ok = 0;
    if (p->nkey && !group_hash_init(&groups, p->distinct_width ? 512 : rows)) goto done;
    if (p->distinct_width && !group_hash_init(&pairs, rows)) goto done;
    for (int w = 0; w < p->workers; ++w) {
        group_shard_t *s = j->all[w].shard + part;
        while (s->blocks) {
            group_block_t *b = s->blocks;
            for (unsigned i = 0; i < b->n; ++i) {
                const group_record_t *r = b->records + i;
                unsigned fresh = 0;
                if (p->distinct_width) {
                    group_slot_t *pair = group_hash_entry(&pairs, r->low, r->high);
                    if (!pair) goto done;
                    fresh = !pair->count;
                    pair->count = 1;
                }
                if (p->nkey) {
                    u128 key = U128(r->high, r->low) >> p->distinct_width;
                    group_slot_t *slot = group_hash_entry(&groups, (uint64_t)key, (uint64_t)(key >> 64));
                    if (!slot) goto done;
                    ++slot->count;
                    slot->distinct += fresh;
                    slot->sum[0] += r->sum[0];
                    slot->sum[1] += r->sum[1];
                }
            }
            s->blocks = b->next;
            free(b);
        }
    }
    if (p->nkey) {
        for (uint64_t i = 0; i < groups.capacity; ++i)
            if (groups.slots[i].count) group_keep(j, groups.slots[i]);
    } else {
        j->distinct_total += pairs.occupied;
    }
    ok = 1;
done:
    free(groups.slots);
    free(pairs.slots);
    return ok;
}

static void *group_reduce(void *arg) {
    group_worker_t *j = arg;
    group_plan_t *p = j->plan;
    unsigned part;
    while (!atomic_load_explicit(&p->failed, memory_order_relaxed) &&
           (part = atomic_fetch_add_explicit(&p->next, 1, memory_order_relaxed)) < GROUP_SHARDS) {
        uint64_t rows = 0;
        for (int w = 0; w < p->workers; ++w) rows += j->all[w].shard[part].n;
        if (rows && !group_reduce_shard(j, part, rows)) {
            atomic_store_explicit(&p->failed, 1, memory_order_relaxed);
            break;
        }
    }
    return NULL;
}

static void *group_reduce_bits(void *arg) {
    group_worker_t *j = arg;
    size_t i = (size_t)(j - j->all), words = j->plan->distinct_words;
    int workers = j->plan->workers;
    for (size_t w = words * i / workers; w < words * (i + 1) / workers; ++w) {
        uint64_t combined = 0;
        for (int worker = 0; worker < workers; ++worker) combined |= j->all[worker].distinct_bits[w];
        j->distinct_total += __builtin_popcountll(combined);
    }
    return NULL;
}

static int group_cmp(const void *a, const void *b) {
    return group_better(a, b, 0) ? -1 : group_better(b, a, 0);
}
static int group_cmp_distinct(const void *a, const void *b) {
    return group_better(a, b, 1) ? -1 : group_better(b, a, 1);
}
static int group_cmp_key(const void *a, const void *b) {
    return group_better(a, b, 2) ? -1 : group_better(b, a, 2);
}

static uint64_t plane_equal_word(const atable_t *t, int col, uint32_t code, uint64_t word, uint64_t mask);

typedef struct {
    uint32_t value[GROUP_PREFIX_LIMIT][GROUP_KEYS][2];
    const group_plan_t *plan;
    unsigned groups;
} group_prefix_t;

typedef struct {
    const group_prefix_t *prefix;
    uint64_t begin, end;
    uint64_t count[GROUP_PREFIX_LIMIT];
} group_prefix_job_t;

static void *group_prefix_count(void *arg) {
    group_prefix_job_t *job = arg;
    const group_prefix_t *prefix = job->prefix;
    const group_plan_t *p = prefix->plan;
    for (uint64_t w = job->begin; w < job->end; ++w) {
        uint64_t scope = rowmask(p->t, w) & (p->filter ? p->filter[w] : UINT64_MAX);
        for (unsigned g = 0; g < prefix->groups && scope; ++g) {
            uint64_t mask = scope;
            for (unsigned i = 0; i < p->nkey && mask; ++i) {
                const at_group_key_t *key = p->key + i;
                mask = plane_equal_word(p->t, key->low, prefix->value[g][i][0], w, mask);
                if (mask && key->high >= 0)
                    mask = plane_equal_word(p->t, key->high, prefix->value[g][i][1], w, mask);
            }
            job->count[g] += __builtin_popcountll(mask);
            scope &= ~mask;                 /* the selected groups are disjoint */
        }
    }
    return NULL;
}

/* An unordered LIMIT may pick any groups. Discover a small prefix of distinct
 * keys, then count those groups over the entire scope with packed equality. */
static int group_prefix(group_plan_t *p, int workers, uint64_t *out_keys,
                        uint64_t *out_counts, uint32_t *out_n) {
    u128 selected[GROUP_PREFIX_LIMIT];
    group_prefix_t prefix = {.plan = p};
    uint64_t budget = p->t->nw < 4096 ? p->t->nw : 4096;
    for (uint64_t w = 0; w < budget && prefix.groups < p->limit; ++w) {
        uint64_t mask = rowmask(p->t, w) & (p->filter ? p->filter[w] : UINT64_MAX);
        if (!mask) continue;
        uint64_t rows[GROUP_BUNDLES][64];
        group_word(p, w, rows);
        while (mask && prefix.groups < p->limit) {
            u128 key = group_pack(p, rows, __builtin_ctzll(mask));
            mask &= mask - 1;
            unsigned i = 0;
            while (i < prefix.groups && key != selected[i]) ++i;
            if (i == prefix.groups) selected[prefix.groups++] = key;
        }
    }
    if (prefix.groups < p->limit && budget != p->t->nw) return 0;
    for (unsigned g = 0; g < prefix.groups; ++g) {
        u128 packed = selected[g];
        for (int i = (int)p->nkey - 1; i >= 0; --i) {
            const at_group_key_t *key = p->key + i;
            unsigned width = p->key_width[i];
            uint64_t code = (uint64_t)(packed & (((u128)1 << width) - 1));
            packed >>= width;
            out_keys[(size_t)g * p->nkey + i] = code;
            prefix.value[g][i][0] = (uint32_t)(code & ((UINT64_C(1) << p->t->width[key->low]) - 1));
            if (key->high >= 0) prefix.value[g][i][1] = (uint32_t)(code >> key->shift);
        }
    }
    if (prefix.groups) {
        group_prefix_job_t jobs[64] = {0};
        for (int i = 0; i < workers; ++i) {
            jobs[i].prefix = &prefix;
            jobs[i].begin = p->t->nw * i / workers;
            jobs[i].end = p->t->nw * (i + 1) / workers;
        }
        pool_jobs(workers, group_prefix_count, jobs, sizeof *jobs);
        for (unsigned g = 0; g < prefix.groups; ++g) {
            out_counts[g] = 0;
            for (int i = 0; i < workers; ++i) out_counts[g] += jobs[i].count[g];
        }
    }
    *out_n = prefix.groups;
    return 1;
}

/* Optional packed input mask. Output is row-major by key or measure,
 * sorted by descending count and then ascending raw key tuple. */
static int group_encoded(const atable_t *t, const uint64_t *filter, int32_t nkey,
                  const at_group_key_t *keys, int32_t nsum, const int32_t *sums,
                  uint32_t k, uint64_t *out_keys, uint64_t *out_counts,
                  uint64_t *out_sums, uint32_t *out_n,
                  int32_t distinct, uint32_t rank, uint64_t *out_distinct,
                  const uint32_t *projection, uint64_t projection_length,
                  uint64_t projection_mask, uint32_t projection_width) {
    if (!out_n) return -1;
    *out_n = 0;
    if (!t || t->k < 1 || t->k > MAXCOL || t->n > UINT32_MAX || t->nw != (t->n + 63) / 64 ||
        nkey < 0 || nkey > GROUP_KEYS || !keys || nsum < 0 || nsum > 2 || (nsum && (!sums || !out_sums)) ||
        (nkey && (!out_keys || !out_counts)) || (!nkey && (!distinct || nsum)) ||
        (distinct != 0 && distinct != 1) || (distinct && !out_distinct) || rank > 3 ||
        (rank == 3 && (!nkey || nsum || distinct || k > GROUP_PREFIX_LIMIT)) || (rank == 1 && !distinct) || !k || k > GROUP_LIMIT) return -1;
    group_plan_t p = {.t = t, .filter = filter, .key = keys, .sum = sums, .nkey = nkey, .nsum = nsum, .limit = k, .rank = rank};
    for (int c = 0; c < MAXCOL; ++c) p.bundle[c] = -1;
    p.projection = projection;
    p.projection_length = projection_length;
    p.projection_mask = projection_mask;
    if (projection && (!projection_length || projection_length > SIZE_MAX / sizeof *projection
                       || !projection_width || projection_width > 32)) return -1;
    unsigned used[GROUP_BUNDLES] = {0}, packed_width = 0, projections = 0;
    for (int i = 0; i < nkey + distinct; ++i) {
        const at_group_key_t *q = keys + i;
        if (q->high < -1 || (rank == 3 && q->transform != AT_GROUP_RAW) ||
            (q->transform < AT_GROUP_RAW || q->transform > AT_GROUP_COORDINATE) ||
            !group_add_column(&p, q->low, used)) return -1;
        unsigned width = t->width[q->low];
        if (q->high >= 0) {
            if (q->shift < (int)width || q->shift >= 64 || !group_add_column(&p, q->high, used) ||
                t->width[q->high] > 64 - q->shift) return -1;
            width = q->shift + t->width[q->high];
        }
        if (q->transform == AT_GROUP_MINUTE) {
            width = 6;
            p.minute_offset[i] = q->offset % INT64_C(3600000000);
        } else if (q->transform == AT_GROUP_EPOCH_MINUTE) {
            const int64_t minute = INT64_C(60000000);
            int64_t rem = q->offset % minute;
            if (rem < 0) rem += minute;
            p.minute_offset[i] = rem;
            uint64_t max_bucket = (uint64_t)(((((u128)1 << width) - 1) + rem) / minute);
            width = max_bucket ? 64 - __builtin_clzll(max_bucket) : 1;
        } else if (q->transform == AT_GROUP_PROJECT) {
            if (!projection || projections++ || q->high >= 0) return -1;
            width = projection_width;
        } else if (q->transform == AT_GROUP_COORDINATE) {
            if (q->high >= 0 || q->shift < 0 || (unsigned)q->shift >= width ||
                q->offset < 1 || q->offset > width - (unsigned)q->shift) return -1;
            width = (unsigned)q->offset;
        }
        p.key_width[i] = width;
        if (i == nkey) p.distinct_width = width;
        packed_width += width;
    }
    if (packed_width > 128 || (!!projection != !!projections)) return -1;
    for (int i = 0; i < nsum; ++i) if (!group_add_column(&p, sums[i], used)) return -1;
    if (!t->n) {
        if (!nkey) { *out_distinct = 0; if (out_counts) *out_counts = 0; *out_n = 1; }
        return 0;
    }
    int workers = nthreads(), useful = (int)((t->n + 16383) / 16384);
    if (workers > useful) workers = useful;
    if (workers > 64) workers = 64;
    if (workers < 1) workers = 1;
    if (rank == 3) {
        if (group_prefix(&p, workers, out_keys, out_counts, out_n)) return 0;
        rank = p.rank = 2;                   /* insufficient diversity in the discovery prefix */
    }
    p.workers = workers;
    group_worker_t *jobs = calloc(workers, sizeof *jobs);
    group_slot_t *heaps = malloc((size_t)workers * k * sizeof *heaps);
    if (!nkey && p.distinct_width <= 22)
        p.distinct_words = p.distinct_width <= 6 ? 1 : (size_t)1 << (p.distinct_width - 6);
    uint64_t *bits = p.distinct_words ? calloc((size_t)workers * p.distinct_words, sizeof *bits) : NULL;
    if (!jobs || !heaps || (p.distinct_words && !bits)) {
        free(jobs); free(heaps); free(bits);
        return -2;
    }
    for (int i = 0; i < workers; ++i) {
        jobs[i].plan = &p; jobs[i].all = jobs;
        jobs[i].begin = t->nw * i / workers; jobs[i].end = t->nw * (i + 1) / workers;
        jobs[i].heap = heaps + (size_t)i * k;
        jobs[i].distinct_bits = bits ? bits + (size_t)i * p.distinct_words : NULL;
    }
    pool_jobs(workers, group_partition, jobs, sizeof *jobs);
    if (!atomic_load_explicit(&p.failed, memory_order_relaxed))
        pool_jobs(workers, bits ? group_reduce_bits : group_reduce, jobs, sizeof *jobs);
    int failure = atomic_load_explicit(&p.failed, memory_order_relaxed);
    int rc = failure == 2 ? -1 : failure ? -2 : 0;
    if (!rc && !nkey) {
        *out_distinct = 0;
        if (out_counts) *out_counts = 0;
        for (int i = 0; i < workers; ++i) *out_distinct += jobs[i].distinct_total;
        *out_n = 1;
    } else if (!rc) {
        for (int i = 1; i < workers; ++i)
            for (unsigned h = 0; h < jobs[i].nheap; ++h) group_keep(jobs, jobs[i].heap[h]);
        qsort(heaps, jobs[0].nheap, sizeof *heaps, rank == 0 ? group_cmp : rank == 1 ? group_cmp_distinct : group_cmp_key);
        for (unsigned i = 0; i < jobs[0].nheap; ++i) {
            const group_slot_t *g = heaps + i;
            u128 packed = U128(g->high, g->low);
            for (int key = nkey - 1; key >= 0; --key) {
                unsigned width = p.key_width[key];
                out_keys[(size_t)i * nkey + key] = (uint64_t)(packed & (((u128)1 << width) - 1));
                packed >>= width;
            }
            out_counts[i] = g->count;
            if (distinct) out_distinct[i] = g->distinct;
            for (int sum = 0; sum < nsum; ++sum) out_sums[(size_t)i * nsum + sum] = g->sum[sum];
        }
        *out_n = jobs[0].nheap;
    }
    for (int i = 0; i < workers; ++i) for (unsigned part = 0; part < GROUP_SHARDS; ++part) {
        group_block_t *b = jobs[i].shard[part].blocks;
        while (b) { group_block_t *next = b->next; free(b); b = next; }
    }
    free(bits);
    free(heaps);
    free(jobs);
    return rc;
}
int at_group_encoded_projected(const atable_t *t, const uint64_t *filter, int32_t nkey,
                  const at_group_key_t *keys, int32_t nsum, const int32_t *sums,
                  uint32_t k, uint64_t *out_keys, uint64_t *out_counts,
                  uint64_t *out_sums, uint32_t *out_n, int32_t distinct,
                  uint32_t rank, uint64_t *out_distinct, const uint32_t *projection,
                  uint64_t projection_length, uint64_t projection_mask, uint32_t projection_width) {
    return group_encoded(t, filter, nkey, keys, nsum, sums, k, out_keys, out_counts, out_sums,
                         out_n, distinct, rank, out_distinct,
                         projection, projection_length, projection_mask, projection_width);
}

int at_group_encoded(const atable_t *t, const uint64_t *filter, int32_t nkey,
                  const at_group_key_t *keys, int32_t nsum, const int32_t *sums,
                  uint32_t k, uint64_t *out_keys, uint64_t *out_counts,
                  uint64_t *out_sums, uint32_t *out_n, int32_t distinct,
                  uint32_t rank, uint64_t *out_distinct) {
    return group_encoded(t, filter, nkey, keys, nsum, sums, k, out_keys, out_counts, out_sums,
                         out_n, distinct, rank, out_distinct, NULL, 0, 0, 0);
}

int at_group_topk(const atable_t *t, const uint64_t *filter, int32_t nkey,
                  const at_group_key_t *keys, int32_t nsum, const int32_t *sums,
                  uint32_t k, uint64_t *out_keys, uint64_t *out_counts,
                  uint64_t *out_sums, uint32_t *out_n) {
    return at_group_encoded(t, filter, nkey, keys, nsum, sums, k, out_keys, out_counts,
                            out_sums, out_n, 0, 0, NULL);
}
typedef struct {
    const group_plan_t *plan;
    uint64_t begin, end, count;
    u128 sums[2];
} scalar_plane_job_t;

static void *scalar_plane_moments(void *arg) {
    scalar_plane_job_t *job = arg;
    const group_plan_t *p = job->plan;
    const atable_t *t = p->t;
    for (uint64_t w = job->begin; w < job->end; ++w) {
        uint64_t mask = rowmask(t, w) & (p->filter ? p->filter[w] : UINT64_MAX);
        if (!mask) continue;
        job->count += __builtin_popcountll(mask);
        for (unsigned i = 0; i < p->nkey; ++i) {
            const at_group_key_t *key = p->key + i;
            const int columns[2] = {key->low, key->high};
            for (int j = 0; j < 2 && columns[j] >= 0; ++j) {
                int c = columns[j], width = t->width[c];
                uint64_t values[32], plane = 0;
                if (t->vflag[c])
                    tb_virtual_planes(t, c, t->V[t->vflag[c] - 1][0][w], w, values, width);
                for (int b = 0; b < width; ++b) {
                    if (t->vflag[c]) plane = values[b];
                    else plane ^= t->V[c][b][w];
                    int power = (t->msb[c] ? width - b - 1 : b) + (j ? key->shift : 0);
                    job->sums[i] += (u128)__builtin_popcountll(plane & mask) << power;
                }
            }
        }
    }
    return NULL;
}

/* Count and up to two unsigned code sums, each written as low/high limbs. */
int at_scalar_plane_moments(const atable_t *t, const uint64_t *filter, int32_t nsum,
                            const at_group_key_t *keys, uint64_t *out) {
    if (!t || !out || t->k < 1 || t->k > MAXCOL || t->n > UINT32_MAX ||
        nsum < 0 || nsum > 2 || (nsum && !keys)) return -1;
    group_plan_t p = {.t = t, .filter = filter, .key = keys, .nkey = nsum};
    for (int c = 0; c < MAXCOL; ++c) p.bundle[c] = -1;
    unsigned used[GROUP_BUNDLES] = {0};
    for (int i = 0; i < nsum; ++i) {
        const at_group_key_t *key = keys + i;
        if (key->transform || key->high < -1 || !group_add_column(&p, key->low, used)) return -1;
        if (key->high >= 0 && (key->shift < t->width[key->low] || key->shift >= 64 ||
                !group_add_column(&p, key->high, used) || t->width[key->high] > 64 - key->shift)) return -1;
    }
    int workers = nthreads(), useful = 1 + (int)(t->n / 65536);
    if (workers > useful) workers = useful;
    if (workers > 64) workers = 64;
    if (workers < 1) workers = 1;
    scalar_plane_job_t jobs[64] = {0};
    for (int i = 0; i < workers; ++i) {
        jobs[i].plan = &p;
        jobs[i].begin = t->nw * i / workers;
        jobs[i].end = t->nw * (i + 1) / workers;
    }
    pool_jobs(workers, scalar_plane_moments, jobs, sizeof *jobs);
    out[0] = 0;
    u128 sums[2] = {0, 0};
    for (int i = 0; i < workers; ++i) {
        out[0] += jobs[i].count;
        for (int j = 0; j < nsum; ++j) sums[j] += jobs[i].sums[j];
    }
    for (int i = 0; i < nsum; ++i) {
        out[1 + 2 * i] = (uint64_t)sums[i];
        out[2 + 2 * i] = (uint64_t)(sums[i] >> 64);
    }
    return 0;
}

static uint64_t plane_equal_word(const atable_t *t, int col, uint32_t code, uint64_t word, uint64_t mask) {
    int width = t->width[col], previous = 0;
    uint64_t virtual_values[32];
    if (t->vflag[col])
        tb_virtual_planes(t, col, t->V[t->vflag[col] - 1][0][word], word, virtual_values, width);
    for (int b = 0; b < width && mask; ++b) {
        int bit = (code >> (t->msb[col] ? width - b - 1 : b)) & 1;
        uint64_t value = t->vflag[col] ? virtual_values[b] : t->V[col][b][word];
        mask &= (t->vflag[col] ? bit : bit ^ previous) ? value : ~value;
        previous = bit;
    }
    return mask;
}

enum { LOOKUP_SUM_KEYS = 128, LOOKUP_SUM_NODES = 2 + 32 * LOOKUP_SUM_KEYS };

typedef struct { uint16_t child[2], key; } lookup_sum_node_t;
typedef struct { uint64_t mask; unsigned key; } lookup_sum_match_t;
typedef struct {
    const atable_t *table;
    const uint64_t *filter;
    const uint32_t *lookup;
    uint64_t lookup_len;
    uint32_t lookup_mask;
    int group_col, lookup_col;
    lookup_sum_node_t trie[LOOKUP_SUM_NODES];
    _Atomic int invalid;
} lookup_sum_plan_t;
typedef struct {
    lookup_sum_plan_t *plan;
    uint64_t begin, end, sums[LOOKUP_SUM_KEYS];
} lookup_sum_job_t;

/* A trie over the physical XOR digits shares common equality prefixes.
 * Branches terminate as soon as their intersection with the scope is empty. */
static void lookup_sum_match(const lookup_sum_plan_t *plan, uint64_t word, unsigned digit,
                             unsigned node, uint64_t scope, lookup_sum_match_t *matches,
                             unsigned *nmatch, uint64_t *selected) {
    const lookup_sum_node_t *n = plan->trie + node;
    if (n->key) {
        matches[(*nmatch)++] = (lookup_sum_match_t){scope, n->key - 1};
        *selected |= scope;
        return;
    }
    uint64_t plane = plan->table->V[plan->group_col][digit][word];
    uint64_t one = scope & plane, zero = scope & ~plane;
    if (zero && n->child[0])
        lookup_sum_match(plan, word, digit + 1, n->child[0], zero, matches, nmatch, selected);
    if (one && n->child[1])
        lookup_sum_match(plan, word, digit + 1, n->child[1], one, matches, nmatch, selected);
}

static void *lookup_sum_worker(void *arg) {
    lookup_sum_job_t *job = arg;
    lookup_sum_plan_t *p = job->plan;
    const atable_t *t = p->table;
    int col = p->lookup_col, width = t->width[col];
    for (uint64_t w = job->begin; w < job->end; ++w) {
        uint64_t scope = rowmask(t, w) & (p->filter ? p->filter[w] : UINT64_MAX);
        if (!scope) continue;
        lookup_sum_match_t matches[LOOKUP_SUM_KEYS];
        unsigned nmatch = 0;
        uint64_t selected = 0;
        lookup_sum_match(p, w, 0, 1, scope, matches, &nmatch, &selected);
        if (!selected) continue;

        uint64_t codes[64] = {0}, value = 0;
        for (int b = 0; b < width; ++b) {
            value ^= t->V[col][b][w];
            unsigned bit = t->msb[col] ? width - b - 1 : b;
            if ((p->lookup_mask >> bit) & 1) codes[bit] = value & selected;
        }
        transpose64(codes);
        for (unsigned m = 0; m < nmatch; ++m) {
            uint64_t total = 0;
            for (uint64_t rows = matches[m].mask; rows; rows &= rows - 1) {
                uint64_t code = codes[__builtin_ctzll(rows)];
                if (code >= p->lookup_len) {
                    atomic_store_explicit(&p->invalid, 1, memory_order_relaxed);
                    return NULL;
                }
                total += p->lookup[code];
            }
            job->sums[matches[m].key] += total;
        }
    }
    return NULL;
}

int at_plane_lookup_sums(const atable_t *t, const uint64_t *filter,
                        int32_t group_col, int32_t lookup_col, uint32_t lookup_mask,
                        const uint32_t *lookup, uint64_t lookup_len,
                        const uint32_t *keys, uint32_t nkeys, uint64_t *sums) {
    if (!t || t->k < 1 || t->k > MAXCOL || t->n > UINT32_MAX ||
        t->nw != (t->n + 63) / 64 || !keys || !nkeys || nkeys > LOOKUP_SUM_KEYS ||
        !lookup || !lookup_len || lookup_len > SIZE_MAX / sizeof *lookup || !sums) return -1;
    int cols[] = {group_col, lookup_col};
    for (unsigned i = 0; i < sizeof cols / sizeof cols[0]; ++i) {
        int col = cols[i];
        if (col < 0 || col >= t->k || t->vflag[col] || t->width[col] < 1 || t->width[col] > 32)
            return -1;
        for (int b = 0; b < t->width[col]; ++b) if (!t->V[col][b]) return -1;
    }
    lookup_sum_plan_t plan = {.table = t, .filter = filter, .lookup = lookup,
        .lookup_len = lookup_len, .lookup_mask = lookup_mask, .group_col = group_col, .lookup_col = lookup_col};
    int width = t->width[group_col], msb = t->msb[group_col];
    unsigned nodes = 1;
    for (unsigned k = 0; k < nkeys; ++k) {
        if ((uint64_t)keys[k] >= (UINT64_C(1) << width)) return -1;
        unsigned node = 1, previous = 0;
        for (int b = 0; b < width; ++b) {
            unsigned bit = (keys[k] >> (msb ? width - b - 1 : b)) & 1;
            uint16_t *edge = &plan.trie[node].child[bit ^ previous];
            if (!*edge) *edge = ++nodes;
            node = *edge;
            previous = bit;
        }
        if (plan.trie[node].key) return -1;
        plan.trie[node].key = k + 1;
    }

    int workers = nthreads(), useful = 1 + (int)(t->n / 65536);
    if (workers > useful) workers = useful;
    if (workers > 64) workers = 64;
    if (workers < 1) workers = 1;
    lookup_sum_job_t jobs[64] = {0};
    for (int i = 0; i < workers; ++i) {
        jobs[i].plan = &plan;
        jobs[i].begin = t->nw * i / workers;
        jobs[i].end = t->nw * (i + 1) / workers;
    }
    pool_jobs(workers, lookup_sum_worker, jobs, sizeof *jobs);
    if (atomic_load_explicit(&plan.invalid, memory_order_relaxed)) return -1;
    for (unsigned k = 0; k < nkeys; ++k) {
        sums[k] = 0;
        for (int i = 0; i < workers; ++i) sums[k] += jobs[i].sums[k];
    }
    return 0;
}

enum { PLANE_HISTOGRAM_LOCAL_BYTES = 64 * 1024 * 1024 };

typedef struct {
    const atable_t *table;
    const uint64_t *filter;
    uint32_t *locals, *counts;
    size_t stride;
    uint32_t domain;
    int col, workers;
    _Atomic int invalid;
} plane_histogram_t;

typedef struct { plane_histogram_t *hist; int worker; } plane_histogram_job_t;

static void *plane_histogram_count(void *arg) {
    const plane_histogram_job_t *job = arg;
    plane_histogram_t *h = job->hist;
    const atable_t *t = h->table;
    int col = h->col, width = t->width[col], msb = t->msb[col];
    uint32_t *counts = h->locals + (size_t)job->worker * h->stride;
    uint64_t begin = t->nw * job->worker / h->workers;
    uint64_t end = t->nw * (job->worker + 1) / h->workers;
    for (uint64_t w = begin; w < end; ++w) {
        if ((w & 255) == 0 && atomic_load_explicit(&h->invalid, memory_order_relaxed)) break;
        uint64_t mask = rowmask(t, w) & (h->filter ? h->filter[w] : UINT64_MAX);
        if (!mask) continue;
        uint64_t planes[64] = {0}, value = 0;
        for (int b = 0; b < width; ++b) {
            value ^= t->V[col][b][w];
            planes[msb ? width - b - 1 : b] = value;
        }
        int sparse = __builtin_popcountll(mask) <= 3;
        if (!sparse) transpose64(planes);
        while (mask) {
            unsigned row = __builtin_ctzll(mask);
            mask &= mask - 1;
            uint32_t code = 0;
            if (sparse) {
                for (int b = 0; b < width; ++b)
                    code |= (uint32_t)((planes[b] >> row) & 1) << b;
            } else {
                code = (uint32_t)planes[row];
            }
            if (code >= h->domain) {
                atomic_store_explicit(&h->invalid, 1, memory_order_relaxed);
                return NULL;
            }
            ++counts[code];
        }
    }
    return NULL;
}

static void *plane_histogram_merge(void *arg) {
    const plane_histogram_job_t *job = arg;
    plane_histogram_t *h = job->hist;
    size_t begin = ((h->stride / 16) * job->worker / h->workers) * 16;
    size_t end = ((h->stride / 16) * (job->worker + 1) / h->workers) * 16;
    if (begin >= h->domain) return NULL;
    if (end > h->domain) end = h->domain;
    uint32_t *restrict out = h->counts + begin;
    size_t count = end - begin;
    memcpy(out, h->locals + begin, count * sizeof *out);
    for (int worker = 1; worker < h->workers; ++worker) {
        const uint32_t *restrict local = h->locals + (size_t)worker * h->stride + begin;
        for (size_t i = 0; i < count; ++i) out[i] += local[i];
    }
    return NULL;
}

/* One encoded axis, directly from the packed planes. Exact counts in the
 * caller's domain, with a bounded worker-local histogram and no row staging. */
int at_plane_code_histogram(const atable_t *t, const uint64_t *filter,
                            int32_t col, uint32_t domain, uint32_t *counts) {
    if (!t || t->k < 1 || t->k > MAXCOL || t->n > UINT32_MAX ||
        t->nw != (t->n + 63) / 64 || col < 0 || col >= t->k || t->vflag[col] ||
        t->width[col] < 0 || t->width[col] > 32 || !domain ||
        domain > PLANE_HISTOGRAM_LOCAL_BYTES / sizeof *counts || !counts) return -1;
    for (int b = 0; b < t->width[col]; ++b) if (!t->V[col][b]) return -1;
    if (!t->n) {
        memset(counts, 0, (size_t)domain * sizeof *counts);
        return 0;
    }

    size_t stride = ((size_t)domain + 15) & ~(size_t)15;
    int workers = nthreads(), useful = 1 + (int)(t->n / 16384);
    int capacity = (int)(PLANE_HISTOGRAM_LOCAL_BYTES / sizeof *counts / stride);
    if (workers > useful) workers = useful;
    if (workers > capacity) workers = capacity;
    if (workers > 64) workers = 64;
    if (workers < 1) workers = 1;
    uint32_t *locals = calloc((size_t)workers * stride, sizeof *locals);
    if (!locals) return -2;

    plane_histogram_t hist = {.table = t, .filter = filter, .locals = locals,
        .counts = counts, .stride = stride, .domain = domain, .col = col, .workers = workers};
    plane_histogram_job_t jobs[64];
    for (int i = 0; i < workers; ++i) jobs[i] = (plane_histogram_job_t){&hist, i};
    pool_jobs(workers, plane_histogram_count, jobs, sizeof *jobs);
    int rc = atomic_load_explicit(&hist.invalid, memory_order_relaxed) ? -1 : 0;
    if (!rc) pool_jobs(workers, plane_histogram_merge, jobs, sizeof *jobs);
    free(locals);
    return rc;
}

/* Small equality result, with fallback when the match set is larger. */
int at_plane_equal_ids(const atable_t *t, const uint64_t *visible, const at_group_key_t *key,
                       uint64_t code, uint32_t limit, uint64_t *ids, uint32_t *nout) {
    if (!t || !visible || !key || !ids || !nout || !limit || t->k < 1 || t->k > MAXCOL ||
        key->transform || key->high < -1) return -1;
    group_plan_t plan = {.t = t};
    for (int c = 0; c < MAXCOL; ++c) plan.bundle[c] = -1;
    unsigned used[GROUP_BUNDLES] = {0};
    if (!group_add_column(&plan, key->low, used)) return -1;
    int low_width = t->width[key->low];
    int width = low_width;
    if (key->high >= 0) {
        if (key->shift < low_width || key->shift >= 64 || !group_add_column(&plan, key->high, used) ||
            t->width[key->high] > 64 - key->shift) return -1;
        width = key->shift + t->width[key->high];
    }
    *nout = 0;
    if (width < 64 && (code >> width)) return 0;
    uint32_t low = (uint32_t)(code & ((UINT64_C(1) << low_width) - 1));
    uint32_t high = key->high < 0 ? 0 : (uint32_t)(code >> key->shift);
    for (uint64_t word = 0; word < t->nw; ++word) {
        uint64_t mask = rowmask(t, word) & visible[word];
        if (!mask) continue;
        mask = plane_equal_word(t, key->low, low, word, mask);
        if (mask && key->high >= 0) mask = plane_equal_word(t, key->high, high, word, mask);
        while (mask) {
            if (*nout == limit) return 1;
            ids[(*nout)++] = word * 64 + __builtin_ctzll(mask);
            mask &= mask - 1;
        }
    }
    return 0;
}

/* a 128-bit sum kept as (lo, hi) words */
static inline void add128(int64_t *lo, int64_t *hi, u128 v) { u128 x = U128(*hi, *lo) + v; *lo = (int64_t)(uint64_t)x; *hi = (int64_t)(uint64_t)(x >> 64); }
static inline void hadd_n(touch_t *h, u128 key, uint64_t v, uint64_t nadd) {
    uint64_t j = hslot(h, key); h->sum[j] = (int64_t)((uint64_t)h->sum[j] + v); h->cnt[j] += nadd;
}
static inline void hadd(touch_t *h, u128 key, uint64_t v) { hadd_n(h, key, v, 1); }
static inline void hext(touch_t *h, u128 key, uint32_t lo, uint32_t hi, uint64_t nadd) {   /* min in sum, max in shi */
    uint64_t j = hslot(h, key);
    if (!h->cnt[j]) { h->sum[j] = lo; h->shi[j] = hi; }
    else { if ((int64_t)lo < h->sum[j]) h->sum[j] = lo; if ((int64_t)hi > h->shi[j]) h->shi[j] = hi; }
    h->cnt[j] += nadd;
}
/* MIN and MAX of the codes of the rows in mask M (M != 0) by plane elimination, most significant value bit first */
static inline void ext_word(const uint64_t *x, int b, int msb, uint64_t M, uint32_t *mn, uint32_t *mx) {
    uint64_t a = M, c = M; uint32_t lo = 0, hi = 0;
    for (int k = b - 1; k >= 0; k--) {
        uint64_t p = x[msb ? b - 1 - k : k];
        uint64_t z = a & ~p; if (z) a = z; else lo |= 1u << k;
        uint64_t o = c & p; if (o) { c = o; hi |= 1u << k; }
    }
    *mn = lo; *mx = hi;
}
static inline void ext_dense(int64_t *osum, int64_t *ohi, uint64_t *ocnt, uint64_t k, uint32_t lo, uint32_t hi, uint64_t n) {
    if ((int64_t)lo < osum[k]) osum[k] = lo;
    if ((int64_t)hi > ohi[k]) ohi[k] = hi;
    ocnt[k] += n;
}
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
            /* b < 32 first: a shift by the type's width (a 32-bit column) is undefined; as in gpu.cu */
            st[sp++] = cmp_any(t, x, I->col, P_GE, I->a) & (b < 32 && I->b >> b ? ~0ULL : cmp_any(t, x, I->col, P_LE, I->b)); break; }
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
static int periodic_validity(const atable_t *t, int32_t valid) {
    return valid == -1 || (periodic_column(t, valid) && t->width[valid] == 1);
}

static int periodic_program(const atable_t *t, const pinstr_t *prog, int32_t nprog) {
    if (nprog < 0 || nprog > TP_MAX_PROGRAM || (nprog && !prog)) return 0;
    int depth = 0;
    uint32_t checked = 0;
    for (int32_t i = 0; i < nprog; ++i) {
        const pinstr_t *p = prog + i;
        if (p->op == P_AND || p->op == P_OR || p->op == P_XOR) {
            if (depth < 2) return 0;
            --depth;
            continue;
        }
        if (p->op == P_NOT) {
            if (!depth) return 0;
            continue;
        }
        if (++depth > TP_MAX_STACK || p->op < P_TRUE || p->op > P_XOR) return 0;
        if (p->op == P_TRUE) continue;
        if (p->col < 0 || p->col >= t->k) return 0;
        uint32_t col = UINT32_C(1) << p->col;
        if (!(checked & col)) {
            if (!periodic_column(t, p->col)) return 0;
            checked |= col;
        }
        unsigned width = (unsigned)t->width[p->col];
        if ((p->op == P_CUBE || p->op == P_CLT || p->op == P_CGE) &&
            (!p->b || (width < 32 && (p->b >> width)))) return 0;
        if (p->op == P_PREFIX && p->b > width) return 0;
        if (p->op == P_BIT && p->a >= 32) return 0;
    }
    return !nprog || depth == 1;
}

static int periodic_plan(const atable_t *t, const tp_plan_t *plan) {
    if (plan->nkeys > TP_MAX_KEYS || plan->nmeasures > TP_MAX_MEASURES ||
        !plan->groups || plan->groups > TP_MAX_GROUPS) return 0;
    uint32_t groups = 1;
    for (uint32_t i = 0; i < plan->nkeys; ++i) {
        const tp_key_t *key = plan->keys + i;
        if (!periodic_column(t, key->col) || !periodic_validity(t, key->valid) ||
            !key->radix || groups > TP_MAX_GROUPS / key->radix) return 0;
        groups *= key->radix;
    }
    if (groups != plan->groups) return 0;
    for (uint32_t i = 0; i < plan->nmeasures; ++i) {
        const tp_measure_t *measure = plan->measures + i;
        if ((measure->flags & ~(TP_SUM | TP_MIN | TP_MAX)) ||
            !periodic_validity(t, measure->valid) ||
            ((measure->flags || measure->col != -1) && !periodic_column(t, measure->col))) return 0;
    }
    return 1;
}

typedef struct {
    tb_cycle_v_t edges[MAXCOL][NB];
    uint8_t predicate_width[MAXCOL];
    uint32_t predicate_mask;
} periodic_encoded_t;

static uint32_t periodic_predicate_widths(const atable_t *t, const pinstr_t *prog, int32_t nprog,
                                          uint8_t *widths) {
    uint32_t mask = 0;
    for (int32_t i = 0; i < nprog; ++i) {
        const pinstr_t *p = &prog[i];
        if (p->op == P_TRUE || p->op == P_NOT || p->op == P_AND ||
            p->op == P_OR || p->op == P_XOR) continue;
        unsigned width = t->width[p->col];
        if (p->op == P_BIT) width = p->a >= width ? 0 : t->msb[p->col] ? width - p->a : p->a + 1;
        else if (p->op == P_PREFIX) width = p->b;
        else if (p->op == P_CUBE || p->op == P_CLT || p->op == P_CGE) width = 32 - __builtin_clz(p->b);
        if (width > widths[p->col]) widths[p->col] = width;
        if (width) mask |= UINT32_C(1) << p->col;
    }
    return mask;
}

static uint64_t periodic_read_edges(const atable_t *t, periodic_encoded_t *source, uint32_t sector,
                                    uint32_t mask, const uint8_t *widths) {
    uint64_t bytes = 0;
    for (; mask; mask &= mask - 1) {
        unsigned col = __builtin_ctz(mask);
        int width = at_cycle_v_planes(t, sector, col, source->edges[col]);
        if (width < 0) return UINT64_MAX;
        if (widths) width = widths[col];
        for (int j = 0; j < width; ++j) bytes += (uint64_t)source->edges[col][j].words * 8;
    }
    return bytes;
}

static inline const uint64_t *periodic_decode(const atable_t *t, dec_t *d, int col, uint64_t word,
                                              int width, const periodic_encoded_t *source) {
    if (!source) return decode_n(t, d, col, word, width);
    int begin = d->nd[col];
    uint64_t bits = begin ? d->p[col][begin - 1] : 0;
    uint32_t phase = (uint32_t)(word % TP_SECTOR_WORDS);
    for (int j = begin; j < width; ++j) {
        bits ^= tb_cycle_v_word(&source->edges[col][j], phase);
        d->p[col][j] = bits;
    }
    if (begin < width) d->nd[col] = width;
    return d->p[col];
}

static uint64_t periodic_mask(const atable_t *t, dec_t *d, uint64_t word,
                              const pinstr_t *prog, int32_t nprog, const periodic_encoded_t *source) {
    if (source)
        for (uint32_t mask = source->predicate_mask; mask; mask &= mask - 1) {
            int col = __builtin_ctz(mask);
            periodic_decode(t, d, col, word, source->predicate_width[col], source);
        }
    return eval_prog(t, prog, nprog, d, word) & rowmask(t, word);
}

static uint64_t periodic_valid_word(const atable_t *t, dec_t *d, int col, uint64_t word,
                                    const periodic_encoded_t *source) {
    return col < 0 ? UINT64_MAX : periodic_decode(t, d, col, word, 1, source)[0];
}

static int periodic_reduce_word(const atable_t *t, uint64_t word, uint64_t repeats,
                                const pinstr_t *prog, int32_t nprog,
                                const tp_plan_t *plan, tp_state_t *out, const periodic_encoded_t *source) {
    dec_t d;
    DEC_RESET(d);
    uint64_t pending = periodic_mask(t, &d, word, prog, nprog, source);
    if (!pending) return 0;
    const uint64_t *keys[TP_MAX_KEYS], *measures[TP_MAX_MEASURES];
    uint64_t key_valid[TP_MAX_KEYS], measure_valid[TP_MAX_MEASURES];
    for (uint32_t i = 0; i < plan->nkeys; ++i) {
        const tp_key_t *key = plan->keys + i;
        key_valid[i] = periodic_valid_word(t, &d, key->valid, word, source);
        keys[i] = periodic_decode(t, &d, key->col, word, t->width[key->col], source);
    }
    for (uint32_t i = 0; i < plan->nmeasures; ++i) {
        const tp_measure_t *measure = plan->measures + i;
        measure_valid[i] = periodic_valid_word(t, &d, measure->valid, word, source);
        measures[i] = measure->flags
            ? periodic_decode(t, &d, measure->col, word, t->width[measure->col], source) : NULL;
    }

    /* Split the bitmap by the first remaining row's mixed-radix key.
     * Intersect decoded value planes instead of transposing every row. */
    while (pending) {
        unsigned row = (unsigned)__builtin_ctzll(pending);
        uint64_t group_mask = pending;
        uint32_t group = 0;
        for (uint32_t i = 0; i < plan->nkeys; ++i) {
            const tp_key_t *key = plan->keys + i;
            uint32_t code = 0;
            if (!(key_valid[i] >> row & 1)) {
                code = key->radix - 1;
                group_mask &= ~key_valid[i];
            } else {
                int width = t->width[key->col];
                group_mask &= key_valid[i];
                for (int bit = 0; bit < width; ++bit) {
                    uint64_t plane = keys[i][t->msb[key->col] ? width - 1 - bit : bit];
                    uint32_t set = (uint32_t)((plane >> row) & 1);
                    code |= set << bit;
                    group_mask &= set ? plane : ~plane;
                }
                if (code >= key->radix - (key->valid >= 0)) return -1;
            }
            group = group * key->radix + code;
        }
        pending &= ~group_mask;
        tp_state_t *state = out + group;
        state->count += (uint64_t)__builtin_popcountll(group_mask) * repeats;
        for (uint32_t i = 0; i < plan->nmeasures; ++i) {
            uint64_t mask = group_mask & measure_valid[i];
            if (!mask) continue;
            state->valid[i] += (uint64_t)__builtin_popcountll(mask) * repeats;
            const tp_measure_t *measure = plan->measures + i;
            if (measure->flags & TP_SUM) {
                uint64_t sum = 0;
                int width = t->width[measure->col];
                for (int bit = 0; bit < width; ++bit) {
                    uint64_t plane = measures[i][t->msb[measure->col] ? width - 1 - bit : bit];
                    sum += (uint64_t)__builtin_popcountll(plane & mask) << bit;
                }
                state->sums[i] += sum * repeats;
            }
            if (measure->flags & (TP_MIN | TP_MAX)) {
                uint32_t lo, hi;
                ext_word(measures[i], t->width[measure->col], t->msb[measure->col], mask, &lo, &hi);
                if ((measure->flags & TP_MIN) && lo < state->mins[i]) state->mins[i] = lo;
                if ((measure->flags & TP_MAX) && hi > state->maxs[i]) state->maxs[i] = hi;
            }
        }
    }
    return 0;
}

static void periodic_init_states(const tp_plan_t *plan, tp_state_t *out) {
    memset(out, 0, (size_t)plan->groups * sizeof *out);
    for (uint32_t group = 0; group < plan->groups; ++group)
        for (uint32_t i = 0; i < TP_MAX_MEASURES; ++i) out[group].mins[i] = UINT32_MAX;
}

static int periodic_reduce_range(const atable_t *t, const tp_sector_t *sector,
                                 const pinstr_t *prog, int32_t nprog,
                                 const tp_plan_t *plan, tp_state_t *out, uint32_t encoded,
                                 uint64_t *payload) {
    uint32_t full = sector->full_words, cycle = sector->cycle_words;
    uint32_t phases = cycle ? cycle : full;
    uint32_t copies = cycle ? full / cycle : 1, extra = cycle ? full % cycle : 0;
    periodic_encoded_t source;
    if (encoded) {
        memset(source.predicate_width, 0, sizeof source.predicate_width);
        source.predicate_mask = periodic_predicate_widths(t, prog, nprog, source.predicate_width);
        uint64_t bytes = periodic_read_edges(t, &source, sector->first_word / TP_SECTOR_WORDS, encoded, NULL);
        if (bytes == UINT64_MAX) return -1;
        *payload += bytes;
    }
    for (uint32_t phase = 0; phase < phases; ++phase)
        if (periodic_reduce_word(t, (uint64_t)sector->first_word + phase,
                                 copies + (phase < extra), prog, nprog, plan, out,
                                 encoded ? &source : NULL)) return -1;
    if (sector->tail_rows) {
        uint32_t tail_word = sector->first_word + full;
        if (encoded && tail_word / TP_SECTOR_WORDS != sector->first_word / TP_SECTOR_WORDS) {
            uint64_t bytes = periodic_read_edges(t, &source, tail_word / TP_SECTOR_WORDS, encoded, NULL);
            if (bytes == UINT64_MAX) return -1;
            *payload += bytes;
        }
        if (periodic_reduce_word(t, tail_word, 1, prog, nprog, plan, out, encoded ? &source : NULL)) return -1;
    }
    return 0;
}

int at_periodic_reduce(const atable_t *t, uint32_t cycle_words, const pinstr_t *prog,
                        int32_t nprog, const tp_plan_t *plan, tp_state_t *out) {
    if (!periodic_layout(t) || !at_has_dense(t) || !plan || !out || t->n > UINT32_MAX ||
        !cycle_words || cycle_words > TP_MAX_CYCLE_WORDS || cycle_words > t->n / 64) return -1;
    uint64_t revision = __atomic_load_n(&t->revision, __ATOMIC_ACQUIRE);
    if (!periodic_plan(t, plan) || !periodic_program(t, prog, nprog)) return -1;
    periodic_init_states(plan, out);
    tp_sector_t whole = {0, (uint32_t)(t->n / 64), cycle_words, (uint32_t)(t->n % 64)};
    int rc = periodic_reduce_range(t, &whole, prog, nprog, plan, out, 0, NULL);
    return revision == __atomic_load_n(&t->revision, __ATOMIC_ACQUIRE) ? rc : -1;
}

typedef struct {
    const atable_t *t;
    const tp_sector_t *sectors;
    const pinstr_t *prog;
    const tp_plan_t *plan;
    uint32_t nsectors, encoded;
    int32_t nprog;
    _Atomic uint32_t next;
    _Atomic int invalid;
} periodic_sector_work_t;

typedef struct { periodic_sector_work_t *work; tp_state_t *out; uint64_t payload; } periodic_sector_job_t;

static void *periodic_sector_worker(void *arg) {
    periodic_sector_job_t *job = arg;
    periodic_sector_work_t *work = job->work;
    periodic_init_states(work->plan, job->out);
    while (!atomic_load_explicit(&work->invalid, memory_order_relaxed)) {
        uint32_t i = atomic_fetch_add_explicit(&work->next, 1, memory_order_relaxed);
        if (i >= work->nsectors) break;
        if (periodic_reduce_range(work->t, work->sectors + i, work->prog, work->nprog,
                                  work->plan, job->out, work->encoded, &job->payload))
            atomic_store_explicit(&work->invalid, 1, memory_order_relaxed);
    }
    return NULL;
}

static void periodic_merge_states(const tp_plan_t *plan, tp_state_t *out, const tp_state_t *part) {
    for (uint32_t g = 0; g < plan->groups; ++g) {
        out[g].count += part[g].count;
        for (uint32_t i = 0; i < plan->nmeasures; ++i) {
            out[g].valid[i] += part[g].valid[i];
            out[g].sums[i] += part[g].sums[i];
            if (out[g].mins[i] > part[g].mins[i]) out[g].mins[i] = part[g].mins[i];
            if (out[g].maxs[i] < part[g].maxs[i]) out[g].maxs[i] = part[g].maxs[i];
        }
    }
}

static int periodic_reduce_sectors(const atable_t *t, const tp_sector_t *sectors,
                                   uint32_t nsectors, const pinstr_t *prog, int32_t nprog,
                                   const tp_plan_t *plan, tp_state_t *out,
                                   tp_sector_metrics_t *metrics, uint32_t encoded, uint64_t *payload) {
    if (!periodic_layout(t) || !sectors || !nsectors || nsectors > t->nw ||
        !plan || !out || t->n > UINT32_MAX) return -1;
    uint64_t revision = __atomic_load_n(&t->revision, __ATOMIC_ACQUIRE);
    if (!periodic_plan(t, plan) || !periodic_program(t, prog, nprog)) return -1;

    uint32_t full = (uint32_t)(t->n / 64), at = 0;
    tp_sector_metrics_t result = {0};
    for (uint32_t i = 0; i < nsectors; ++i) {
        const tp_sector_t *sector = sectors + i;
        uint32_t tail = i + 1 == nsectors ? (uint32_t)(t->n % 64) : 0;
        if (sector->first_word != at || sector->full_words > full - at ||
            (!sector->full_words && !tail) || sector->tail_rows != tail ||
            sector->cycle_words > TP_MAX_CYCLE_WORDS ||
            sector->cycle_words > sector->full_words / TP_MIN_REPEATS) return -1;
        if (encoded && !(nsectors == 1 && sector->cycle_words) &&
            (sector->first_word % TP_SECTOR_WORDS ||
             sector->full_words + (tail != 0) > TP_SECTOR_WORDS)) return -1;
        at += sector->full_words;
        periodic_sector_metrics(&result, sector);
    }
    if (at != full) return -1;

    int workers = nthreads(), useful = 1 + (int)(result.representative_words / 2048);
    if (workers > useful) workers = useful;
    if ((uint32_t)workers > nsectors) workers = (int)nsectors;
    periodic_sector_work_t work = {
        .t = t, .sectors = sectors, .prog = prog, .plan = plan, .nsectors = nsectors, .nprog = nprog,
        .encoded = encoded,
    };
    periodic_sector_job_t jobs[64] = {{&work, out}};
    tp_state_t *partials = NULL;
    if (workers > 1) {
        partials = malloc((size_t)(workers - 1) * plan->groups * sizeof *partials);
        if (!partials) return -1;
        for (int i = 1; i < workers; ++i)
            jobs[i] = (periodic_sector_job_t){&work, partials + (size_t)(i - 1) * plan->groups};
    }
    pool_jobs(workers, periodic_sector_worker, jobs, sizeof *jobs);
    int rc = -1;
    if (!atomic_load_explicit(&work.invalid, memory_order_relaxed) &&
        revision == __atomic_load_n(&t->revision, __ATOMIC_ACQUIRE) &&
        (!encoded || tb_cycle_has_mask(t, encoded))) {
        for (int i = 1; i < workers; ++i) periodic_merge_states(plan, out, jobs[i].out);
        if (metrics) *metrics = result;
        if (payload) {
            *payload = 0;
            for (int i = 0; i < workers; ++i) *payload += jobs[i].payload;
        }
        rc = 0;
    }
    free(partials);
    return rc;
}

int at_periodic_reduce_sectors(const atable_t *t, const tp_sector_t *sectors,
                               uint32_t nsectors, const pinstr_t *prog, int32_t nprog,
                               const tp_plan_t *plan, tp_state_t *out,
                               tp_sector_metrics_t *metrics) {
    if (!at_has_dense(t)) return -1;
    return periodic_reduce_sectors(t, sectors, nsectors, prog, nprog, plan, out, metrics, 0, NULL);
}

int at_periodic_reduce_encoded(const atable_t *t, uint32_t cycle, const tp_sector_t *sectors,
                               uint32_t nsectors, uint32_t mask, const pinstr_t *prog, int32_t nprog,
                               const tp_plan_t *plan, tp_state_t *out, uint64_t *payload) {
    if (!periodic_layout(t) || !t->n || t->n > UINT32_MAX || !plan || !out ||
        !tb_cycle_has_mask(t, mask) || !periodic_program(t, prog, nprog) || !periodic_plan(t, plan)) return -1;
    uint8_t widths[MAXCOL] = {0};
    uint32_t reads = periodic_predicate_widths(t, prog, nprog, widths);
    for (uint32_t i = 0; i < plan->nkeys; ++i) {
        const tp_key_t *key = plan->keys + i;
        reads |= UINT32_C(1) << key->col;
        if (key->valid >= 0) reads |= UINT32_C(1) << key->valid;
    }
    for (uint32_t i = 0; i < plan->nmeasures; ++i) {
        const tp_measure_t *m = plan->measures + i;
        if (m->flags) reads |= UINT32_C(1) << m->col;
        if (m->valid >= 0) reads |= UINT32_C(1) << m->valid;
    }
    if (reads & ~mask) return -1;
    tp_sector_t whole = {0, (uint32_t)(t->n / 64), cycle, (uint32_t)(t->n % 64)};
    if (cycle) {
        if (sectors || nsectors) return -1;
        sectors = &whole;
        nsectors = 1;
    }
    return periodic_reduce_sectors(t, sectors, nsectors, prog, nprog, plan, out, NULL, mask, payload);
}

typedef struct {
    const atable_t *t;
    const pinstr_t *prog;
    const uint64_t *visible;
    uint64_t *out;
    int32_t nprog;
    uint32_t sectors, mask;
    uint8_t widths[MAXCOL];
    _Atomic uint32_t next;
    _Atomic int invalid;
} cycle_bitmap_work_t;

typedef struct {
    cycle_bitmap_work_t *work;
    uint64_t count, payload, evaluated;
} cycle_bitmap_job_t;

static uint32_t cycle_bitmap_period(const periodic_encoded_t *source, uint32_t words) {
    unsigned cap = words / TP_MIN_REPEATS, cycle = 1;
    for (uint32_t mask = source->predicate_mask; mask; mask &= mask - 1) {
        unsigned col = __builtin_ctz(mask);
        for (unsigned j = 0; j < source->predicate_width[col]; ++j) {
            unsigned period = source->edges[col][j].period;
            if (!period) return 0;
            cycle = periodic_lcm(cycle, period);
            if (cycle > cap) return 0;
        }
    }
    return cycle;
}

static void *cycle_bitmap_worker(void *arg) {
    cycle_bitmap_job_t *job = arg;
    cycle_bitmap_work_t *work = job->work;
    const atable_t *t = work->t;
    periodic_encoded_t source = {.predicate_mask = work->mask};
    memcpy(source.predicate_width, work->widths, sizeof source.predicate_width);
    while (!atomic_load_explicit(&work->invalid, memory_order_relaxed)) {
        uint32_t sector = atomic_fetch_add_explicit(&work->next, 1, memory_order_relaxed);
        if (sector >= work->sectors) break;
        uint32_t first = sector * TP_SECTOR_WORDS;
        uint32_t words = t->nw - first < TP_SECTOR_WORDS ? (uint32_t)(t->nw - first) : TP_SECTOR_WORDS;
        uint32_t tail = first + words == t->nw && t->n % 64;
        uint32_t full = words - tail;
        uint64_t bytes = periodic_read_edges(t, &source, sector, work->mask, work->widths);
        if (bytes == UINT64_MAX) {
            atomic_store_explicit(&work->invalid, 1, memory_order_relaxed);
            break;
        }
        job->payload += bytes;
        uint32_t cycle = cycle_bitmap_period(&source, full);
        uint32_t phases = cycle ? cycle : words;
        job->evaluated += phases;
        uint64_t *out = work->out + first;
        for (uint32_t phase = 0; phase < phases; ++phase) {
            dec_t d;
            DEC_RESET(d);
            out[phase] = periodic_mask(t, &d, first + phase, work->prog, work->nprog, &source);
        }
        if (cycle) {
            for (uint32_t filled = cycle; filled < full;) {
                uint32_t n = full - filled < filled ? full - filled : filled;
                memcpy(out + filled, out, n * sizeof *out);
                filled += n;
            }
            if (tail) out[full] = out[full % cycle] & rowmask(t, first + full);
        }
        for (uint32_t i = 0; i < words; ++i) {
            if (work->visible) out[i] &= work->visible[first + i];
            job->count += __builtin_popcountll(out[i]);
        }
    }
    return NULL;
}

int at_cycle_filter_bitmap(const atable_t *t, const pinstr_t *prog, int32_t nprog,
                           const uint64_t *visible, uint64_t *out, uint64_t metrics[3]) {
    if (!periodic_layout(t) || !t->n || t->n > UINT32_MAX || !out || !metrics ||
        !periodic_program(t, prog, nprog)) return -1;
    cycle_bitmap_work_t work = {
        .t = t, .prog = prog, .visible = visible, .out = out, .nprog = nprog,
        .sectors = (uint32_t)((t->nw + TP_SECTOR_WORDS - 1) / TP_SECTOR_WORDS),
    };
    work.mask = periodic_predicate_widths(t, prog, nprog, work.widths);
    if (!tb_cycle_has_mask(t, work.mask)) return -1;
    uint64_t revision = __atomic_load_n(&t->revision, __ATOMIC_ACQUIRE);
    int workers = nthreads();
    if ((uint32_t)workers > work.sectors) workers = (int)work.sectors;
    cycle_bitmap_job_t jobs[64] = {0};
    for (int i = 0; i < workers; ++i) jobs[i].work = &work;
    pool_jobs(workers, cycle_bitmap_worker, jobs, sizeof *jobs);
    if (atomic_load_explicit(&work.invalid, memory_order_relaxed) ||
        revision != __atomic_load_n(&t->revision, __ATOMIC_ACQUIRE) ||
        !tb_cycle_has_mask(t, work.mask)) return -1;
    metrics[0] = metrics[1] = metrics[2] = 0;
    for (int i = 0; i < workers; ++i) {
        metrics[0] += jobs[i].count;
        metrics[1] += jobs[i].payload;
        metrics[2] += jobs[i].evaluated;
    }
    return 0;
}

/* Bounded native profiling for relationship-system physical alternatives. The
 * sampled masks have exactly the ordinary fact executor's filter/semijoin/delta
 * semantics. No result rows or column-sized intermediate arrays are materialized. */
int32_t at_relationship_sample(const atable_t *t, const qplan_t *q, tr_system_sample *out) {
    if (!t || !q || !out || q->nprog < 0 || q->nprog > 1024 || q->nsemi < 0 || q->nsemi > 8 ||
        q->nkey < 0 || q->nkey > MAXKEY || q->nattr || q->nranges > 4096 ||
        (q->nranges && !q->ranges) || q->xmode < 0 || q->xmode > 2 || (q->xmode && !q->xrows)) return TR_INVALID;
    memset(out, 0, sizeof *out);
    int sp = 0;
    for (int i = 0; i < q->nprog; ++i) {
        const pinstr_t *p = &q->prog[i];
        if (p->op == P_AND || p->op == P_OR || p->op == P_XOR) { if (sp < 2) return TR_INVALID; --sp; }
        else if (p->op == P_NOT) { if (sp < 1) return TR_INVALID; }
        else {
            if (p->op < P_TRUE || p->op > P_XOR || (p->op != P_TRUE && (p->col < 0 || p->col >= t->k))) return TR_INVALID;
            if (p->op != P_TRUE && (t->width[p->col] < 1 || t->width[p->col] > 32)) return TR_INVALID;
            if ((p->op == P_CUBE || p->op == P_CLT || p->op == P_CGE) && !p->b) return TR_INVALID;
            if ((p->op == P_CUBE || p->op == P_CLT || p->op == P_CGE) && t->width[p->col] < 32 &&
                (p->b >> t->width[p->col])) return TR_INVALID;
            if (p->op == P_PREFIX && p->b > (uint32_t)t->width[p->col]) return TR_INVALID;
            if (p->op == P_BIT && p->a > 31) return TR_INVALID;
            if (++sp > 48) return TR_INVALID;
        }
    }
    if (q->nprog && sp != 1) return TR_INVALID;
    for (int i = 0; i < q->nsemi; ++i)
        if (q->semi_col[i] < 0 || q->semi_col[i] >= t->k || !q->semi_bm[i]) return TR_INVALID;
    for (int i = 0; i < q->nkey; ++i) {
        int k = q->key_kind[i], c = q->key_arg[i];
        if ((k != 0 && k != 2 && k != 3) || c < 0 || c >= t->k ||
            t->width[c] < 1 || t->width[c] > 32 ||
            (k == 2 && (q->key_nbits[i] < 1 || q->key_nbits[i] > t->width[c])) ||
            (k == 3 && (!q->key_mask[i] || (t->width[c] < 32 && (q->key_mask[i] >> t->width[c]))))) return TR_INVALID;
    }
    uint64_t eligible = 0, nr = q->nranges ? q->nranges : 1, last = 0;
    for (uint64_t k = 0; k < nr; ++k) {
        uint64_t a = q->nranges ? q->ranges[2*k] : 0, b = q->nranges ? q->ranges[2*k+1] : t->nw;
        if (b > t->nw) b = t->nw;
        if (a < last) a = last;
        if (a >= b) continue;
        last = b; eligible += b-a;
    }
    if (!eligible) return TR_OK;
    uint64_t ns = eligible < 32 ? eligible : 32, targets[32], words[32];
    for (uint64_t i = 0; i < ns; ++i)
        targets[i] = ns == 1 ? 0 : (uint64_t)(((__uint128_t)i * (eligible-1)) / (ns-1));
    uint64_t base = 0, next = 0; last = 0;
    for (uint64_t k = 0; k < nr && next < ns; ++k) {
        uint64_t a = q->nranges ? q->ranges[2*k] : 0, b = q->nranges ? q->ranges[2*k+1] : t->nw;
        if (b > t->nw) b = t->nw;
        if (a < last) a = last;
        if (a >= b) continue;
        last = b;
        while (next < ns && targets[next] < base+b-a) { words[next] = a+targets[next]-base; ++next; }
        base += b-a;
    }
    for (uint64_t si = 0; si < ns; ++si) {
        uint64_t w = words[si]; dec_t D; DEC_RESET(D);
        uint64_t F = eval_prog(t, q->prog, q->nprog, &D, w) & rowmask(t, w);
        out->rows += (uint64_t)__builtin_popcountll(rowmask(t, w)); out->words++;
        if (q->xmode) F &= q->xmode == 1 ? q->xrows[w] : ~q->xrows[w];
        for (int s = 0; s < q->nsemi && F; ++s) {
            const uint64_t *x = decode(t, &D, q->semi_col[s], w); uint64_t G = F;
            while (G) {
                int r = __builtin_ctzll(G); G &= G-1;
                uint32_t fk = KM(q->semi_kmask[s], extract_col(t, x, q->semi_col[s], r));
                if (fk >= q->semi_n[s] || !(q->semi_bm[s][fk>>6] >> (fk&63) & 1)) F &= ~(UINT64_C(1)<<r);
            }
        }
        if (!F) continue;
        out->passing += (uint64_t)__builtin_popcountll(F); out->nonempty++;
        uint32_t keys[64][MAXKEY]; int ng = 0; uint64_t G = F;
        while (G) {
            int r = __builtin_ctzll(G); G &= G-1; uint32_t key[MAXKEY];
            for (int i = 0; i < q->nkey; ++i) {
                int c = q->key_arg[i], kind = q->key_kind[i];
                if (kind == 0) key[i] = extract_col(t, decode(t, &D, c, w), c, r);
                else if (kind == 2) key[i] = extract_msb(decode_n(t, &D, c, w, q->key_nbits[i]), q->key_nbits[i], r);
                else {
                    uint32_t mk = q->key_mask[i], v = 0;
                    const uint64_t *x = decode_n(t, &D, c, w, 32-__builtin_clz(mk));
                    while (mk) { int j = __builtin_ctz(mk); mk &= mk-1; v = (v<<1) | (uint32_t)(x[j]>>r&1); }
                    key[i] = v;
                }
            }
            int seen = 0;
            for (int j = 0; j < ng; ++j) if (!memcmp(keys[j], key, (size_t)q->nkey*sizeof(uint32_t))) { seen = 1; break; }
            if (!seen) { memcpy(keys[ng++], key, (size_t)q->nkey*sizeof(uint32_t)); }
        }
        out->groups += ng;
    }
    return TR_OK;
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
    uint64_t st[24], registers[48]; int sp = 0;
    for (int i = 0; i < q->nmeas; i++) {
        const minstr_t *I = &q->meas[i];
        switch (I->op) {
        case M_FCOL: st[sp++] = fv[I->arg]; break;
        case M_ATTR: st[sp++] = av[I->arg]; break;
        case M_CONST: st[sp++] = (uint64_t)I->c; break;
        case M_ADD: sp--; st[sp - 1] += st[sp]; break;
        case M_SUB: sp--; st[sp - 1] -= st[sp]; break;
        case M_MUL: sp--; st[sp - 1] *= st[sp]; break;
        case M_SAVE: registers[I->arg] = st[sp - 1]; break;
        case M_LOAD: st[sp++] = registers[I->arg]; break;
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
                          int64_t *osum, uint64_t *ocnt, uint64_t *ptotal, uint64_t *pcnt, touch_t *tl, mtile_t *T, u128 *ptw,
                          uint32_t *pemin, uint32_t *pemax) {
    dec_t D; uint64_t total = 0; uint64_t cnt = 0; u128 tw = 0; uint32_t emin = *pemin, emax = *pemax;
    uint64_t needf = 0;                                  /* fact columns needed as integers (general path) */
    for (int i = 0; i < q->nmeas; i++) if (q->meas[i].op == M_FCOL) needf |= 1ULL << q->meas[i].arg;
    for (int i = 0; i < q->nkey; i++) if (q->key_kind[i] == 0) needf |= 1ULL << q->key_arg[i];
    for (int i = 0; i < q->nattr; i++) needf |= 1ULL << q->attr_fkcol[i];
    if (q->ext_on) needf |= 1ULL << q->ext_col;
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
        if (q->agg == A_ROWS) { q->out_rows[w] = F; continue; }
        if (q->ext_on && !q->nkey) {                    /* MIN / MAX without GROUP BY: plane elimination per word */
            uint32_t lo, hi; ext_word(decode(t, &D, q->ext_col, w), t->width[q->ext_col], t->msb[q->ext_col], F, &lo, &hi);
            if (lo < emin) emin = lo; if (hi > emax) emax = hi;
            continue;
        }
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
                if (q->ext_on) {                         /* MIN / MAX of the group's rows in this word */
                    uint32_t lo, hi; ext_word(decode(t, &D, q->ext_col, w), t->width[q->ext_col], t->msb[q->ext_col], M, &lo, &hi);
                    if (tl) hext(tl, key, lo, hi, n); else ext_dense(osum, q->out_sum_hi, ocnt, (uint64_t)key, lo, hi, n);
                    R &= ~M; continue;
                }
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
        uint64_t KT[MAXKEY][64];                         /* plane-mask keys of a dense word: every row's code at once */
        if (dense) for (int i = 0; i < q->nkey; i++) if (q->key_kind[i] == 3) {
            uint32_t mk = q->key_mask[i]; const uint64_t *x = decode_n(t, &D, q->key_arg[i], w, 32 - __builtin_clz(mk));
            int nb = __builtin_popcount(mk), j = 0; memset(KT[i], 0, sizeof KT[i]);
            while (mk) { int p = __builtin_ctz(mk); mk &= mk - 1; KT[i][nb - 1 - j++] = x[p]; }   /* first plane: top bit */
            transpose64(KT[i]);
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
                if (q->key_kind[i] == 3 && dense) kv = KT[i][r];
                else if (q->key_kind[i] == 3) {          /* coordinate projection: read the field's planes */
                    uint32_t mk = q->key_mask[i]; const uint64_t *x = decode_n(t, &D, q->key_arg[i], w, 32 - __builtin_clz(mk)); kv = 0;
                    while (mk) { int j = __builtin_ctz(mk); mk &= mk - 1; kv = (kv << 1) | ((x[j] >> r) & 1); }
                } else kv = q->key_kind[i] == 0 ? fv[q->key_arg[i]] : q->key_kind[i] == 1 ? av[q->key_arg[i]]
                            : extract_msb(decode_n(t, &D, q->key_arg[i], w, q->key_nbits[i]), q->key_nbits[i], r);   /* field read from the fk planes */
                key = key * q->key_radix[i] + kv;
            }
            if (q->agg == A_DIST) {                      /* COUNT(DISTINCT): the bit of the row's (group, value) cell */
                uint64_t cell = (uint64_t)key;
                __atomic_fetch_or(&q->dist_bits[cell >> 6], 1ULL << (cell & 63), __ATOMIC_RELAXED);
                continue;
            }
            if (q->ext_on) {                             /* MIN / MAX: the row's code */
                uint32_t xv = fv[q->ext_col];
                if (tl) hext(tl, key, xv, xv, 1); else ext_dense(osum, q->out_sum_hi, ocnt, (uint64_t)key, xv, xv, 1);
                continue;
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
    *ptotal += total; *pcnt += cnt; *ptw += tw; *pemin = emin; *pemax = emax;
}

/* ---------------------------------------------------------------- plain columnar baseline: same plans */
typedef struct { uint64_t n; int32_t k, width[MAXCOL]; const uint32_t *col[MAXCOL]; } ptable_t;
ptable_t *pt_create(uint64_t n, int ncols, const uint32_t **cols) {     /* references caller's arrays (no copy) */
    if (ncols < 1 || ncols > MAXCOL) return NULL;
    ptable_t *t = calloc(1, sizeof *t); if (!t) return NULL;
    t->n = n; t->k = ncols;
    for (int c = 0; c < ncols; c++) t->col[c] = cols[c];
    col_widths(n, ncols, cols, t->width);
    return t;
}
void pt_free(ptable_t *t) { free(t); }                 /* the caller's arrays stay the caller's */

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
                          int64_t *osum, uint64_t *ocnt, uint64_t *ptotal, uint64_t *pcnt, touch_t *tl, u128 *ptw,
                          uint32_t *pemin, uint32_t *pemax) {
    uint8_t m[CH]; uint64_t total = 0; uint64_t cnt = 0; uint32_t fv[MAXCOL], av[8]; u128 tw = 0; uint32_t emin = *pemin, emax = *pemax;
    for (uint64_t r0 = ra; r0 < rb; r0 += CH) {
        int len = (int)(rb - r0 < CH ? rb - r0 : CH);
        eval_prog_chunk(t, q->prog, q->nprog, r0, len, m);
        if (q->xmode) for (int r = 0; r < len; r++) { int bit = (int)(q->xrows[(r0 + r) >> 6] >> ((r0 + r) & 63) & 1); m[r] &= q->xmode == 1 ? bit : !bit; }
        for (int s = 0; s < q->nsemi; s++) {
            const uint32_t *fk = t->col[q->semi_col[s]] + r0; const uint64_t *bm = q->semi_bm[s];
            uint32_t km = q->semi_kmask[s];
            for (int r = 0; r < len; r++) if (m[r]) { uint32_t v = KM(km, fk[r]); m[r] = v < q->semi_n[s] && (bm[v >> 6] >> (v & 63) & 1); }
        }
        if (q->agg == A_ROWS) {
            for (int r = 0; r < len; r++) if (m[r]) { q->out_rows[(r0 + r) >> 6] |= 1ULL << ((r0 + r) & 63); cnt++; }
            continue;
        }
        if (q->ext_on && !q->nkey) {
            const uint32_t *x = t->col[q->ext_col] + r0;
            for (int r = 0; r < len; r++) if (m[r]) { cnt++; if (x[r] < emin) emin = x[r]; if (x[r] > emax) emax = x[r]; }
            continue;
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
                if (q->ext_on) { uint32_t xv = t->col[q->ext_col][r0 + r];
                    if (tl) hext(tl, key, xv, xv, 1); else ext_dense(osum, q->out_sum_hi, ocnt, (uint64_t)key, xv, xv, 1);
                    continue; }
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
            if (q->ext_on) { uint32_t xv = fv[q->ext_col];
                if (tl) hext(tl, key, xv, xv, 1); else ext_dense(osum, q->out_sum_hi, ocnt, (uint64_t)key, xv, xv, 1);
                continue; }
            if (q->wide) { u128 v = pt_row_w(t, q, r0 + r); tw += v;       /* the measure as its degree-2 polynomial */
                if (q->nkey) { if (tl) hadd_w(tl, key, v, 1); else { add128(&osum[(uint64_t)key], &q->out_sum_hi[(uint64_t)key], v); ocnt[(uint64_t)key]++; } }
                continue; }
            if (q->nkey) { if (tl) hadd(tl, key, mv); else { osum[(uint64_t)key] += mv; ocnt[(uint64_t)key]++; } }
            total += mv;
        }
    }
    *ptotal += total; *pcnt += cnt; *ptw += tw; *pemin = emin; *pemax = emax;
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
    uint32_t emin, emax;
} job_t;
static void *qworker(void *arg) {
    job_t *j = arg;
    for (;;) {
        uint64_t i = atomic_fetch_add(j->next, 1);
        if (i >= j->nchunks) break;
        uint64_t u0 = j->chunks[2 * i], u1 = j->chunks[2 * i + 1];
        touch_t *tl = j->extra ? &j->tl : NULL;
        if (j->kind == 0) at_query_part(j->t, j->q, u0, u1, j->osum, j->ocnt, &j->total, &j->cnt, tl, j->mt, &j->totalw, &j->emin, &j->emax);
        else pt_query_part(j->t, j->q, u0 * CH, u1 * CH < j->limit ? u1 * CH : j->limit, j->osum, j->ocnt, &j->total, &j->cnt, tl, &j->totalw, &j->emin, &j->emax);
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
    if (q->agg == A_ROWS) sparse = 0;
    for (int i = 0; i < T; i++) {
        jobs[i] = (job_t){kind, t, q, chunks, nchunks, limit, &next, q->out_sum, q->out_cnt, 0, 0, {0}, 0, NULL, 0, 0xFFFFFFFFu, 0};
        if ((i || sparse) && q->nkey) jobs[i].extra = 1;
        if (mm) { jobs[i].mt = mm_tile(i); mm_prepare(t, q, jobs[i].mt); }
    }
    pool_jobs(T, qworker, jobs, sizeof jobs[0]); (void)th;
    q->total_sum = jobs[0].total; q->total_cnt = jobs[0].cnt;
    if (q->ext_on) {                                     /* MIN / MAX: the extremes over the threads */
        uint32_t lo = 0xFFFFFFFFu, hi = 0;
        for (int i = 0; i < T; i++) { if (jobs[i].emin < lo) lo = jobs[i].emin; if (jobs[i].emax > hi) hi = jobs[i].emax; }
        q->total_sum = lo; q->total_sum_hi = hi;
    }
    if (q->wide) { u128 tw = 0; for (int i = 0; i < T; i++) tw += jobs[i].totalw;
                   q->total_sum_hi = (int64_t)(uint64_t)(tw >> 64); q->total_sum = (int64_t)(uint64_t)tw; }
    if (sparse) {                                        /* sparse output: the groups present, merged across threads */
        touch_t *h0 = &jobs[0].tl; memset(q->mm_stats, 0, sizeof q->mm_stats);
        for (int i = 0; i < T; i++) {
            if (mm) for (int k = 0; k < 4; k++) q->mm_stats[k] += jobs[i].mt->st[k];
            if (i) { if (!q->wide && !q->ext_on) q->total_sum += jobs[i].total;
                     q->total_cnt += jobs[i].cnt;
                     touch_t *h = &jobs[i].tl;
                     for (uint64_t x = 0; x < h->cap; x++) if (h->key[x] != TEMPTY) {
                         if (q->ext_on) hext(h0, h->key[x], (uint32_t)h->sum[x], (uint32_t)h->shi[x], h->cnt[x]);
                         else hadd_w(h0, h->key[x], U128(h->shi[x], h->sum[x]), h->cnt[x]); }
                     hfree(h); }
        }
        uint64_t n = 0;
        for (uint64_t x = 0; x < h0->cap; x++) if (h0->key[x] != TEMPTY) {
            if (n < q->out_cap) { q->out_key[n] = (uint64_t)h0->key[x]; q->out_sum[n] = h0->sum[x]; q->out_cnt[n] = h0->cnt[x];
                                  if (q->out_key_hi) q->out_key_hi[n] = (uint64_t)(h0->key[x] >> 64);
                                  if (q->wide || q->ext_on) q->out_sum_hi[n] = h0->shi[x]; }
            n++;
        }
        q->out_n = n; hfree(h0); free(chunks); return;
    }
    for (int i = 1; i < T; i++) {
        if (!q->wide && !q->ext_on) q->total_sum += jobs[i].total;
        q->total_cnt += jobs[i].cnt;
        if (q->nkey) {
            touch_t *h = &jobs[i].tl;
            for (uint64_t x = 0; x < h->cap; x++) if (h->key[x] != TEMPTY) {
                uint64_t k = (uint64_t)h->key[x];
                if (q->ext_on) { ext_dense(q->out_sum, q->out_sum_hi, q->out_cnt, k, (uint32_t)h->sum[x], (uint32_t)h->shi[x], h->cnt[x]); continue; }
                q->out_cnt[k] += h->cnt[x];
                if (q->wide) add128(&q->out_sum[k], &q->out_sum_hi[k], U128(h->shi[x], h->sum[x])); else q->out_sum[k] = (int64_t)((uint64_t)q->out_sum[k] + (uint64_t)h->sum[x]); }
            hfree(h);
        }
    }
    free(chunks);
}
static uint64_t popcount_bits(const uint64_t *b, uint64_t s, uint64_t n) {   /* bits set in [s, s + n) */
    uint64_t c = 0, e = s + n;
    while (s < e && (s & 63)) { c += (b[s >> 6] >> (s & 63)) & 1; s++; }
    while (s + 64 <= e) { c += (uint64_t)__builtin_popcountll(b[s >> 6]); s += 64; }
    while (s < e) { c += (b[s >> 6] >> (s & 63)) & 1; s++; }
    return c;
}
int64_t at_query(const atable_t *t, qplan_t *q) {
    if (!q->wide && !tr_measure_valid(q, t->k)) return -1;
    if (q->agg == A_DIST) {                              /* dense cells only: (group keys, value) with the value last */
        if (!q->dist_bits || !q->dist_radix || q->nkey < 1 || q->widekey || q->out_key || q->wide || q->ext_on || q->nmeas) return -1;
        run_parallel(0, t, q, t->nw, 512, 0, 1);
        uint64_t R = q->dist_radix, G = q->ngroups / R;
        for (uint64_t g = 0; g < G; g++) q->out_cnt[g] = popcount_bits(q->dist_bits, g * R, R);
        return 0;
    }
    if (q->ext_on && q->agg != A_POLYG && q->agg != A_GENERAL) return -1;
    if (q->wide && q->agg != A_MM && q->agg != A_POLYG && q->agg != A_DOM && q->agg != A_COUNT && q->agg != A_GENERAL) return -2;
    if (q->agg == A_MM) { static mplane_t tmp[MAXMP]; mm_init(); if (!q->nkey || tb_mm_planes(t->width, t->msb, q, tmp) < 0) return -1; }
    run_parallel(0, t, q, t->nw, 512, 0, 1); return 0; }      /* ranges in words */      /* ranges in words */
int64_t pt_query(const ptable_t *t, qplan_t *q) { if (!q->wide && !tr_measure_valid(q, t->k)) return -1; if (q->wide && q->agg != A_POLYG && q->agg != A_DOM && q->agg != A_COUNT && q->agg != A_GENERAL) return -2; run_parallel(1, t, q, (t->n + CH - 1) / CH, 16, t->n, CH / 64); return 0; }   /* same word ranges, in CH-row chunks */

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
    t->revision++;
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
