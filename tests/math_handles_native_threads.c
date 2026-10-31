/* FND-013: native threads racing operations against close, slot reuse and serialized mutable payloads.
 * Built and run by tests/test_math_handles.py plain, with ASAN+UBSAN and with TSAN:
 *   gcc -std=c11 -O1 -g -fsanitize=thread -Itorbles tests/math_handles_native_threads.c torbles/math_handles.c \
 *       torbles/mathematics.c torbles/math_domain.c torbles/math_state.c -lpthread -o /tmp/handles_threads
 * Readers evaluate shared immutable polynomials (directly, and through a retained view as the CUDA path does) while a
 * closer closes them and a churn thread reuses the freed slots with new polynomials. Every evaluation must equal the
 * value precomputed for its own handle, or be refused with TR_MATH_CLOSED; once a close has returned, every later
 * call on that handle is refused. Lockers increment an unprotected counter inside a mutable payload under
 * tr_math_handles_lock. At the end every handle is freed, every retain released and every allocation returned.
 * Prints "native handle threads: OK" on success. SPDX-License-Identifier: Apache-2.0 */
#define _POSIX_C_SOURCE 200809L
#include "math_handles.h"
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define READERS 3
#define LOCKERS 2
#define POLYS 8
#define POINTS 48
#define CHURN 3000
#define LOCKS 4000

#define CHECK(cond, ...) do { if (!(cond)) { fprintf(stderr, "FAILED %s:%d: ", __FILE__, __LINE__); \
    fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); exit(1); } } while (0)

typedef struct {
    tr_math_handles *t;
    tr_math_handle h[POLYS], counter;
    int64_t points[POINTS];
    __int128 expect[POLYS][POINTS];
    _Atomic int closed[POLYS], started, stop;
    _Atomic uint64_t ok, refused, deferred, freed, mismatches, violations, destroyed;
} shared;

typedef struct { uint64_t count, shadow; _Atomic uint64_t *destroyed; } counter;

static void sleep_us(long us) { struct timespec ts = {0, us * 1000L}; nanosleep(&ts, NULL); }
static void destroy_counter(void *p) { atomic_fetch_add(((counter *)p)->destroyed, 1); free(p); }

/* p_i(x) = x^2 + i*x + 1000 + i, distinct per polynomial (a churn polynomial has constant 5000 + k) */
static tr_math_handle make_poly(tr_math_handles *t, int64_t linear, int64_t constant) {
    uint8_t e[3] = {2, 1, 0};
    int64_t c[3] = {1, linear, constant};
    tr_math_handle h = 0;
    int32_t rc = tr_math_handles_ipoly_create(t, 1, 3, e, c, NULL, &h, NULL);
    CHECK(rc == 0 && h, "create %d", rc);
    return h;
}

static void *reader(void *arg) {
    shared *s = arg;
    int seen_closed[POLYS] = {0};
    tr_math_i128 values[POINTS];
    uint8_t flags[POINTS];
    while (!atomic_load(&s->started)) ;
    for (unsigned round = 0; !atomic_load(&s->stop); round++) {
        for (int i = 0; i < POLYS; i++) {
            int closed_before = atomic_load(&s->closed[i]);
            uint64_t over = 0;
            int32_t rc;
            if (round & 1) {
                rc = tr_math_handles_ipoly_evaluate(s->t, s->h[i], s->points, POINTS, 1, values, flags, &over, NULL, NULL);
            } else {                                      /* the protocol of the CUDA path: view, evaluate, release */
                tr_math_ipoly_view v;
                rc = tr_math_handles_ipoly_view(s->t, s->h[i], &v);
                if (rc == 0) {
                    sleep_us(20);                         /* hold the reference while the closer may run */
                    rc = tr_math_handles_view_evaluate(&v, s->points, POINTS, values, flags, &over);
                    CHECK(tr_math_handles_release(s->t, s->h[i]) == 0, "release");
                }
            }
            if (rc == 0) {
                atomic_fetch_add(&s->ok, 1);
                if (seen_closed[i] || closed_before) atomic_fetch_add(&s->violations, 1);   /* accepted after close */
                for (int k = 0; k < POINTS; k++)
                    if (flags[k] || tr_i128_get(values[k]) != s->expect[i][k]) { atomic_fetch_add(&s->mismatches, 1); break; }
            } else if (rc == TR_MATH_CLOSED) {
                atomic_fetch_add(&s->refused, 1);
                seen_closed[i] = 1;
            } else {
                atomic_fetch_add(&s->violations, 1);       /* no other answer is possible */
            }
        }
    }
    return NULL;
}

static void *closer(void *arg) {
    shared *s = arg;
    while (!atomic_load(&s->started)) ;
    sleep_us(5000);
    for (int i = 0; i < POLYS; i++) {
        int32_t rc = tr_math_handles_close(s->t, s->h[i]);
        CHECK(rc == 0 || rc == 1, "close %d", rc);
        atomic_fetch_add(rc ? &s->deferred : &s->freed, 1);
        atomic_store(&s->closed[i], 1);
        CHECK(tr_math_handles_close(s->t, s->h[i]) == TR_MATH_CLOSED, "second close");
        sleep_us(1000);
    }
    return NULL;
}

static void *churn(void *arg) {                           /* reuses freed slots with new generations */
    shared *s = arg;
    while (!atomic_load(&s->started)) ;
    for (int k = 0; k < CHURN; k++) {
        tr_math_handle h = make_poly(s->t, 0, 5000 + k);
        int32_t rc = tr_math_handles_close(s->t, h);
        CHECK(rc == 0, "churn close %d", rc);             /* nobody else knows this handle */
    }
    return NULL;
}

static void *locker(void *arg) {
    shared *s = arg;
    while (!atomic_load(&s->started)) ;
    for (int k = 0; k < LOCKS; k++) {
        void *p = NULL;
        int32_t rc = tr_math_handles_lock(s->t, s->counter, TR_HANDLE_USER, &p);
        CHECK(rc == 0, "lock %d", rc);
        counter *c = p;
        c->count++;                                       /* unprotected: only the payload's mutex serializes it */
        c->shadow = c->count * 2;
        CHECK(tr_math_handles_unlock(s->t, s->counter) == 0, "unlock");
    }
    return NULL;
}

int main(void) {
    shared *s = calloc(1, sizeof *s);
    CHECK(s, "calloc");
    int32_t error = 0;
    s->t = tr_math_handles_create(64, 0, NULL, &error, NULL);
    CHECK(s->t && !error, "table %d", error);
    for (int k = 0; k < POINTS; k++) s->points[k] = (int64_t)k * 7919 - 150000;
    for (int i = 0; i < POLYS; i++) {
        s->h[i] = make_poly(s->t, i, 1000 + i);
        for (int k = 0; k < POINTS; k++) {
            __int128 x = s->points[k];
            s->expect[i][k] = x * x + i * x + 1000 + i;
        }
    }
    counter *c = calloc(1, sizeof *c);
    CHECK(c, "calloc");
    c->destroyed = &s->destroyed;
    CHECK(tr_math_handles_insert(s->t, TR_HANDLE_USER, TR_HANDLE_MUTABLE, c, destroy_counter, sizeof *c,
                                 &s->counter, NULL) == 0, "insert");
    void *p = NULL;
    CHECK(tr_math_handles_lock(s->t, s->h[0], TR_HANDLE_ANY, &p) == TR_MATH_INVALID, "lock of an immutable payload");

    pthread_t r[READERS], l[LOCKERS], cl, ch;
    for (int i = 0; i < READERS; i++) pthread_create(&r[i], NULL, reader, s);
    for (int i = 0; i < LOCKERS; i++) pthread_create(&l[i], NULL, locker, s);
    pthread_create(&cl, NULL, closer, s);
    pthread_create(&ch, NULL, churn, s);
    atomic_store(&s->started, 1);
    pthread_join(cl, NULL);
    pthread_join(ch, NULL);
    for (int i = 0; i < LOCKERS; i++) pthread_join(l[i], NULL);
    sleep_us(2000);
    atomic_store(&s->stop, 1);
    for (int i = 0; i < READERS; i++) pthread_join(r[i], NULL);

    CHECK(!atomic_load(&s->mismatches), "%llu evaluations differ from their handle's polynomial",
          (unsigned long long)atomic_load(&s->mismatches));
    CHECK(!atomic_load(&s->violations), "%llu calls accepted after close or failed otherwise",
          (unsigned long long)atomic_load(&s->violations));
    CHECK(c->count == (uint64_t)LOCKERS * LOCKS && c->shadow == 2 * c->count, "serialized counter %llu",
          (unsigned long long)c->count);
    uint64_t increments = c->count;                     /* c is freed by the deferred close below */

    /* a close while another thread holds the lock is deferred; the payload is freed by the unlock */
    CHECK(tr_math_handles_lock(s->t, s->counter, TR_HANDLE_USER, &p) == 0, "lock");
    CHECK(tr_math_handles_close(s->t, s->counter) == 1, "deferred close of a locked payload");
    CHECK(tr_math_handles_lock(s->t, s->counter, TR_HANDLE_USER, &p) == TR_MATH_CLOSED, "lock after close");
    CHECK(!atomic_load(&s->destroyed), "freed under a live lock");
    CHECK(tr_math_handles_unlock(s->t, s->counter) == 0, "unlock");
    CHECK(atomic_load(&s->destroyed) == 1, "the last release frees");
    CHECK(tr_math_handles_unlock(s->t, s->counter) == TR_MATH_INVALID, "unbalanced unlock");

    tr_math_handle_stats st;
    CHECK(tr_math_handles_stats(s->t, &st) == 0, "stats");
    CHECK(st.live == 0 && st.open == 0 && st.deferred == 0 && st.payload_bytes == 0, "live %llu",
          (unsigned long long)st.live);
    CHECK(st.retains == st.releases, "retains %llu releases %llu", (unsigned long long)st.retains,
          (unsigned long long)st.releases);
    CHECK(st.allocations == 1 && st.destroyed == st.inserted && st.free_slots == st.capacity, "allocations %llu",
          (unsigned long long)st.allocations);

    /* the table refuses to be freed under a live reference */
    tr_math_handle h = make_poly(s->t, 1, 1);
    CHECK(tr_math_handles_retain(s->t, h, TR_HANDLE_IPOLY, &p) == 0, "retain");
    CHECK(tr_math_handles_free(s->t) == TR_MATH_INVALID, "free under a reference");
    CHECK(tr_math_handles_release(s->t, h) == 0, "release");
    printf("native handle threads: OK (evaluations %llu, refused %llu, closes freed %llu / deferred %llu, "
           "churn %d, serialized increments %llu, retains %llu)\n",
           (unsigned long long)atomic_load(&s->ok), (unsigned long long)atomic_load(&s->refused),
           (unsigned long long)atomic_load(&s->freed), (unsigned long long)atomic_load(&s->deferred), CHURN,
           (unsigned long long)increments, (unsigned long long)st.retains);
    CHECK(tr_math_handles_free(s->t) == 0, "free");
    free(s);
    return 0;
}
