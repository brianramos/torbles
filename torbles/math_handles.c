/* State ownership, concurrency and the native ABI of the mathematical modules (FND-013); see math_handles.h.
 * SPDX-License-Identifier: Apache-2.0 */
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L          /* error-checking mutexes under -std=c11 */
#endif
#include "math_handles.h"
#include "math_disc.h"
#include "math_modules.h"
#include "mathematics.h"
#include <pthread.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>

_Static_assert(sizeof(tr_math_handle_stats) == 216, "handle stats ABI size");
_Static_assert(sizeof(tr_math_handle_info) == 40, "handle info ABI size");
_Static_assert(sizeof(tr_math_ipoly_view) == 32, "polynomial view ABI size");
_Static_assert(sizeof(tr_math_abi_entry) == 64, "ABI entry size");

/* ------------------------------------------------------------------ the ABI registry of libalg.so
 * The structs of the modules that exist now, one line each; later families register theirs in their own files. */
TR_MATH_ABI(tr_math_rational, 1);                        /* mathematics.h (first slice) */
TR_MATH_ABI(tr_math_term, 1);
TR_MATH_ABI(tr_math_domain_spec, TR_MATH_DOMAIN_ABI);    /* math_domain.h */
TR_MATH_ABI(tr_math_module_info, TR_MATH_DOMAIN_ABI);    /* math_modules.h */
TR_MATH_ABI(tr_math_module_trace, 1);
TR_MATH_ABI(tr_math_state_stats, 1);                     /* math_state.h */
TR_MATH_ABI(tr_math_budget, 1);                          /* math_budgets.h (FND-010) */
TR_MATH_ABI(tr_math_need, 1);
TR_MATH_ABI(tr_math_refusal, 1);
TR_MATH_ABI(tr_math_choice, 1);
TR_MATH_ABI(tr_disc_view, 1);                            /* math_disc.h (DISC-001) */
TR_MATH_ABI(tr_disc_evidence, 1);
TR_MATH_ABI(tr_math_i128, 1);                            /* math_num.h (shared carriers) */
TR_MATH_ABI(tr_math_interval, 1);
TR_MATH_ABI(tr_math_handle_stats, TR_HANDLES_ABI);       /* math_handles.h (FND-013) */
TR_MATH_ABI(tr_math_handle_info, TR_HANDLES_ABI);
TR_MATH_ABI(tr_math_ipoly_view, TR_HANDLES_ABI);
TR_MATH_ABI(tr_math_abi_entry, TR_HANDLES_ABI);

/* the bounds of the section, provided by the linker; hidden, so each library reads its own registry */
extern const tr_math_abi_entry __start_tr_math_abi[] __attribute__((visibility("hidden")));
extern const tr_math_abi_entry __stop_tr_math_abi[] __attribute__((visibility("hidden")));

int32_t tr_math_handles_abi(tr_math_abi_entry *out, int32_t capacity, uint32_t entry_size) {
    if (entry_size != sizeof(tr_math_abi_entry) || capacity < 0 || (capacity && !out)) return TR_MATH_INVALID;
    int32_t n = (int32_t)(__stop_tr_math_abi - __start_tr_math_abi);
    for (int32_t i = 0; i < n && i < capacity; i++) out[i] = __start_tr_math_abi[i];
    return n;
}

/* ------------------------------------------------------------------ tables
 * A slot's state word: bits 0..31 the generation of its current (LIVE) or next occupant, LIVE, CLOSED, RETIRED, and
 * from bit 35 the number of operations holding the handle. Every transition is one atomic operation on this word, so
 * exactly one thread observes "closed and unreferenced" and frees the payload. */
#define TABLE_MAGIC UINT32_C(0x544d484e)
#define IPOLY_MAGIC UINT32_C(0x544d4950)
#define S_LIVE (UINT64_C(1) << 32)
#define S_CLOSED (UINT64_C(1) << 33)
#define S_RETIRED (UINT64_C(1) << 34)
#define S_REF_SHIFT 35
#define S_REF (UINT64_C(1) << S_REF_SHIFT)
#define S_GEN(s) ((uint32_t)(s))
#define S_REFS(s) ((s) >> S_REF_SHIFT)
#define NO_SLOT UINT32_MAX
#define OWNED (1 << 30)              /* slot flag: the payload block was allocated (and counted) by this module */

typedef struct {
    _Alignas(64) _Atomic uint64_t state;   /* one cache line per slot: readers of different handles do not contend */
    int32_t kind, flags;                   /* written under the table mutex while the slot is not live */
    void *payload;                         /* the payload, or the box of a mutable payload */
    tr_math_handle_destroy destroy;        /* NULL for a mutable payload (the box holds it) */
    uint64_t bytes;
    uint32_t next_free;
} slot;

typedef struct {                           /* a mutable payload and the mutex that serializes it */
    pthread_mutex_t lock;
    void *object;
    tr_math_handle_destroy destroy;
} box;

struct tr_math_handles_s {
    uint32_t magic, capacity, generation_limit, free_head;
    pthread_mutex_t lock;
    /* under lock */
    uint64_t live, deferred, retired, free_slots, inserted, closes, deferred_closes, destroyed;
    uint64_t payload_bytes, peak_payload_bytes, table_bytes;
    uint64_t live_kind[TR_HANDLE_KINDS];
    /* updated without the lock */
    _Atomic uint64_t retains, releases, refused_closed, refused_invalid, allocations, allocations_total;
    slot slots[];
};

static int valid(const tr_math_handles *t) { return t && t->magic == TABLE_MAGIC; }
static void error_set(int32_t *error, int32_t code) { if (error) *error = code; }
static int kind_index(int32_t kind) { return kind > 0 && kind < TR_HANDLE_KINDS ? kind : 0; }

static void *counted_alloc(tr_math_handles *t, size_t bytes) {
    void *p = malloc(bytes ? bytes : 1);
    if (p) {
        atomic_fetch_add_explicit(&t->allocations, 1, memory_order_relaxed);
        atomic_fetch_add_explicit(&t->allocations_total, 1, memory_order_relaxed);
    }
    return p;
}
static void counted_free(tr_math_handles *t, void *p) {
    if (!p) return;
    free(p);
    atomic_fetch_sub_explicit(&t->allocations, 1, memory_order_relaxed);
}
static int32_t refuse(tr_math_handles *t, int32_t code) {
    atomic_fetch_add_explicit(code == TR_MATH_CLOSED ? &t->refused_closed : &t->refused_invalid, 1,
                              memory_order_relaxed);
    return code;
}

uint64_t tr_math_handles_table_bytes(uint64_t capacity) {
    uint64_t bytes = tr_budget_add(sizeof(struct tr_math_handles_s), tr_budget_mul(capacity, sizeof(slot)));
    bytes = tr_budget_add(bytes, 63);
    return bytes == TR_BUDGET_UNKNOWN ? bytes : bytes / 64 * 64;
}

tr_math_handles *tr_math_handles_create(uint32_t capacity, uint32_t generation_limit, const tr_math_budget *budget,
                                        int32_t *error, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (capacity < 1 || capacity > TR_HANDLES_CAPACITY_MAX) { error_set(error, TR_MATH_INVALID); return NULL; }
    tr_math_budget dflt;
    if (!budget) { tr_budget_default(&dflt); budget = &dflt; }
    tr_math_need need;
    memset(&need, 0, sizeof need);
    need.need[TR_BUDGET_STATES] = capacity;
    need.need[TR_BUDGET_BYTES] = tr_math_handles_table_bytes(capacity);
    int32_t rc = tr_budget_check(budget, &need, refusal);
    if (rc) { error_set(error, rc); return NULL; }
    size_t bytes = (size_t)need.need[TR_BUDGET_BYTES];
    tr_math_handles *t = aligned_alloc(64, bytes);
    if (!t) { error_set(error, TR_MATH_MEMORY); return NULL; }
    memset(t, 0, bytes);
    if (pthread_mutex_init(&t->lock, NULL)) { free(t); error_set(error, TR_MATH_MEMORY); return NULL; }
    t->capacity = capacity;
    t->generation_limit = generation_limit ? generation_limit : TR_HANDLES_GENERATIONS;
    t->free_head = 0;
    t->free_slots = capacity;
    t->table_bytes = bytes;
    for (uint32_t i = 0; i < capacity; i++) {
        atomic_init(&t->slots[i].state, 1);        /* generation 1, free */
        t->slots[i].next_free = i + 1 < capacity ? i + 1 : NO_SLOT;
    }
    atomic_init(&t->retains, 0); atomic_init(&t->releases, 0);
    atomic_init(&t->refused_closed, 0); atomic_init(&t->refused_invalid, 0);
    atomic_init(&t->allocations, 1); atomic_init(&t->allocations_total, 1);   /* the table itself */
    t->magic = TABLE_MAGIC;
    error_set(error, TR_MATH_OK);
    return t;
}

/* runs the payload's destructor; the caller holds the table mutex and has exclusive use of the slot */
static void destroy_payload(tr_math_handles *t, slot *s) {
    if (s->flags & TR_HANDLE_MUTABLE) {
        box *b = s->payload;
        b->destroy(b->object);
        pthread_mutex_destroy(&b->lock);
        counted_free(t, b);
    } else {
        s->destroy(s->payload);
        if (s->flags & OWNED) atomic_fetch_sub_explicit(&t->allocations, 1, memory_order_relaxed);
    }
}

int32_t tr_math_handles_free(tr_math_handles *t) {
    if (!t) return TR_MATH_OK;
    if (!valid(t)) return TR_MATH_INVALID;
    pthread_mutex_lock(&t->lock);
    for (uint32_t i = 0; i < t->capacity; i++)
        if (S_REFS(atomic_load_explicit(&t->slots[i].state, memory_order_acquire))) {
            pthread_mutex_unlock(&t->lock);
            return TR_MATH_INVALID;                  /* an operation still holds a handle */
        }
    for (uint32_t i = 0; i < t->capacity; i++)
        if (atomic_load_explicit(&t->slots[i].state, memory_order_acquire) & S_LIVE) destroy_payload(t, &t->slots[i]);
    t->magic = 0;
    pthread_mutex_unlock(&t->lock);
    pthread_mutex_destroy(&t->lock);
    free(t);
    return TR_MATH_OK;
}

/* frees the payload of a slot that is closed and unreferenced (observed by the caller's own atomic operation, so no
 * other thread can reach the payload) and makes the slot reusable with the next generation, or retires it */
static void reclaim(tr_math_handles *t, slot *s, int deferred, int locked) {
    if (!locked) pthread_mutex_lock(&t->lock);
    destroy_payload(t, s);
    uint64_t state = atomic_load_explicit(&s->state, memory_order_relaxed);
    uint32_t gen = S_GEN(state), index = (uint32_t)(s - t->slots);
    t->live--;
    if (deferred) t->deferred--;
    t->destroyed++;
    t->payload_bytes -= s->bytes;
    t->live_kind[kind_index(s->kind)]--;
    s->kind = 0; s->flags = 0; s->payload = NULL; s->destroy = NULL; s->bytes = 0;
    uint64_t next;
    if (gen >= t->generation_limit) {                /* every generation of this slot was issued: never reuse it */
        next = (uint64_t)gen | S_RETIRED;
        t->retired++;
    } else {
        next = (uint64_t)gen + 1;
        s->next_free = t->free_head;
        t->free_head = index;
        t->free_slots++;
    }
    atomic_store_explicit(&s->state, next, memory_order_release);
    if (!locked) pthread_mutex_unlock(&t->lock);
}

static slot *slot_of(tr_math_handles *t, tr_math_handle h, uint32_t *gen) {
    uint32_t index = (uint32_t)h;
    *gen = (uint32_t)(h >> 32);
    return index < t->capacity && *gen ? &t->slots[index] : NULL;
}

/* the handle's generation against one reading of its slot: OK (open), CLOSED (issued, then closed: freed or not),
 * INVALID (never issued) */
static int32_t classify(uint64_t state, uint32_t gen) {
    uint32_t current = S_GEN(state);
    if (state & S_RETIRED) return gen <= current ? TR_MATH_CLOSED : TR_MATH_INVALID;
    if (gen < current) return TR_MATH_CLOSED;
    if (gen > current || !(state & S_LIVE)) return TR_MATH_INVALID;
    return state & S_CLOSED ? TR_MATH_CLOSED : TR_MATH_OK;
}

/* drops one reference the caller holds (not counted as a release) */
static void drop(tr_math_handles *t, slot *s) {
    uint64_t old = atomic_fetch_sub_explicit(&s->state, S_REF, memory_order_acq_rel);
    if (S_REFS(old) == 1 && (old & S_CLOSED)) reclaim(t, s, 1, 0);
}
static void put_back(tr_math_handles *t, slot *s) {
    atomic_fetch_add_explicit(&t->releases, 1, memory_order_relaxed);
    drop(t, s);
}

/* a counted reference to an open handle of this kind (lock-free) */
static int32_t acquire(tr_math_handles *t, tr_math_handle h, int32_t kind, slot **out) {
    uint32_t gen;
    slot *s = slot_of(t, h, &gen);
    if (!s) return refuse(t, TR_MATH_INVALID);
    uint64_t state = atomic_load_explicit(&s->state, memory_order_acquire);
    for (;;) {
        int32_t rc = classify(state, gen);
        if (rc) return refuse(t, rc);
        if (S_REFS(state) >= TR_HANDLES_REFS_MAX) return TR_MATH_OVERFLOW;
        if (atomic_compare_exchange_weak_explicit(&s->state, &state, state + S_REF, memory_order_acquire,
                                                  memory_order_acquire)) break;
    }
    if (kind != TR_HANDLE_ANY && s->kind != kind) { drop(t, s); return refuse(t, TR_MATH_INVALID); }
    atomic_fetch_add_explicit(&t->retains, 1, memory_order_relaxed);
    *out = s;
    return TR_MATH_OK;
}

/* a full table: every slot is live or retired; the request needs one slot more than the capacity */
static int32_t full(tr_math_refusal *refusal, uint64_t capacity) {
    if (refusal) {
        refusal->resource = TR_BUDGET_STATES; refusal->refused = 1;
        refusal->need = capacity + 1; refusal->limit = capacity;
    }
    return TR_MATH_BUDGET;
}

static int32_t put_slot(tr_math_handles *t, int32_t kind, int32_t flags, void *payload,
                        tr_math_handle_destroy destroy, uint64_t bytes, tr_math_handle *out,
                        tr_math_refusal *refusal) {
    box *b = NULL;
    if (flags & TR_HANDLE_MUTABLE) {
        b = counted_alloc(t, sizeof *b);
        if (!b) return TR_MATH_MEMORY;
        pthread_mutexattr_t attr;
        int fail = pthread_mutexattr_init(&attr);
        if (!fail) {
            fail = pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_ERRORCHECK) || pthread_mutex_init(&b->lock, &attr);
            pthread_mutexattr_destroy(&attr);
        }
        if (fail) { counted_free(t, b); return TR_MATH_MEMORY; }
        b->object = payload;
        b->destroy = destroy;
    }
    pthread_mutex_lock(&t->lock);
    if (t->free_head == NO_SLOT) {
        pthread_mutex_unlock(&t->lock);
        if (b) { pthread_mutex_destroy(&b->lock); counted_free(t, b); }
        return full(refusal, t->capacity);
    }
    uint32_t index = t->free_head;
    slot *s = &t->slots[index];
    t->free_head = s->next_free;
    s->next_free = NO_SLOT;
    s->kind = kind; s->flags = flags;
    s->payload = b ? (void *)b : payload;
    s->destroy = b ? NULL : destroy;
    s->bytes = bytes;
    uint32_t gen = S_GEN(atomic_load_explicit(&s->state, memory_order_relaxed));
    atomic_store_explicit(&s->state, (uint64_t)gen | S_LIVE, memory_order_release);   /* publishes the fields */
    t->free_slots--; t->live++; t->inserted++;
    t->payload_bytes += bytes;
    if (t->payload_bytes > t->peak_payload_bytes) t->peak_payload_bytes = t->payload_bytes;
    t->live_kind[kind_index(kind)]++;
    pthread_mutex_unlock(&t->lock);
    *out = ((uint64_t)gen << 32) | index;
    return TR_MATH_OK;
}

/* refusal before building a payload that would not find a slot (decided again, atomically, by put_slot) */
static int32_t slot_available(tr_math_handles *t, tr_math_refusal *refusal) {
    pthread_mutex_lock(&t->lock);
    int none = t->free_head == NO_SLOT;
    pthread_mutex_unlock(&t->lock);
    return none ? full(refusal, t->capacity) : TR_MATH_OK;
}

int32_t tr_math_handles_insert(tr_math_handles *t, int32_t kind, int32_t flags, void *payload,
                               tr_math_handle_destroy destroy, uint64_t bytes, tr_math_handle *out,
                               tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (out) *out = 0;
    if (!valid(t) || !out || !payload || !destroy || kind <= 0 || (flags & ~TR_HANDLE_MUTABLE)) return TR_MATH_INVALID;
    return put_slot(t, kind, flags, payload, destroy, bytes, out, refusal);
}

int32_t tr_math_handles_retain(tr_math_handles *t, tr_math_handle h, int32_t kind, void **payload) {
    if (!valid(t) || !payload) return TR_MATH_INVALID;
    slot *s;
    int32_t rc = acquire(t, h, kind, &s);
    if (rc) return rc;
    *payload = (s->flags & TR_HANDLE_MUTABLE) ? ((box *)s->payload)->object : s->payload;
    return TR_MATH_OK;
}

int32_t tr_math_handles_release(tr_math_handles *t, tr_math_handle h) {
    if (!valid(t)) return TR_MATH_INVALID;
    uint32_t gen;
    slot *s = slot_of(t, h, &gen);
    if (!s) return refuse(t, TR_MATH_INVALID);
    uint64_t state = atomic_load_explicit(&s->state, memory_order_acquire);
    for (;;) {
        if (S_GEN(state) != gen || !(state & S_LIVE) || !S_REFS(state)) return refuse(t, TR_MATH_INVALID);
        if (atomic_compare_exchange_weak_explicit(&s->state, &state, state - S_REF, memory_order_acq_rel,
                                                  memory_order_acquire)) break;
    }
    atomic_fetch_add_explicit(&t->releases, 1, memory_order_relaxed);
    if (S_REFS(state) == 1 && (state & S_CLOSED)) reclaim(t, s, 1, 0);
    return TR_MATH_OK;
}

/* retain + lock a mutable payload */
static int32_t lock_slot(tr_math_handles *t, tr_math_handle h, int32_t kind, slot **out) {
    slot *s;
    int32_t rc = acquire(t, h, kind, &s);
    if (rc) return rc;
    if (!(s->flags & TR_HANDLE_MUTABLE) || pthread_mutex_lock(&((box *)s->payload)->lock)) {
        put_back(t, s);                              /* immutable, or already locked by this thread */
        return refuse(t, TR_MATH_INVALID);
    }
    *out = s;
    return TR_MATH_OK;
}
static void unlock_slot(tr_math_handles *t, slot *s) {
    pthread_mutex_unlock(&((box *)s->payload)->lock);
    put_back(t, s);
}

int32_t tr_math_handles_lock(tr_math_handles *t, tr_math_handle h, int32_t kind, void **payload) {
    if (!valid(t) || !payload) return TR_MATH_INVALID;
    slot *s;
    int32_t rc = lock_slot(t, h, kind, &s);
    if (!rc) *payload = ((box *)s->payload)->object;
    return rc;
}

int32_t tr_math_handles_unlock(tr_math_handles *t, tr_math_handle h) {
    if (!valid(t)) return TR_MATH_INVALID;
    uint32_t gen;
    slot *s = slot_of(t, h, &gen);
    if (!s) return refuse(t, TR_MATH_INVALID);
    uint64_t state = atomic_load_explicit(&s->state, memory_order_acquire);
    /* the caller's lock holds a reference, so the slot's fields and box are stable */
    if (S_GEN(state) != gen || !(state & S_LIVE) || !S_REFS(state) || !(s->flags & TR_HANDLE_MUTABLE))
        return refuse(t, TR_MATH_INVALID);
    if (pthread_mutex_unlock(&((box *)s->payload)->lock)) return refuse(t, TR_MATH_INVALID);   /* not this thread's */
    return tr_math_handles_release(t, h);
}

int32_t tr_math_handles_close(tr_math_handles *t, tr_math_handle h) {
    if (!valid(t)) return TR_MATH_INVALID;
    uint32_t gen;
    slot *s = slot_of(t, h, &gen);
    if (!s) return refuse(t, TR_MATH_INVALID);
    pthread_mutex_lock(&t->lock);
    uint64_t state = atomic_load_explicit(&s->state, memory_order_acquire);
    int32_t rc;
    for (;;) {
        rc = classify(state, gen);
        if (rc) { pthread_mutex_unlock(&t->lock); return refuse(t, rc); }
        if (atomic_compare_exchange_weak_explicit(&s->state, &state, state | S_CLOSED, memory_order_acq_rel,
                                                  memory_order_acquire)) break;
    }
    t->closes++;
    if (S_REFS(state)) {                             /* a live operation holds it: its last release frees it */
        t->deferred++;
        t->deferred_closes++;
        rc = 1;
    } else {
        reclaim(t, s, 0, 1);
        rc = 0;
    }
    pthread_mutex_unlock(&t->lock);
    return rc;
}

int32_t tr_math_handles_info(tr_math_handles *t, tr_math_handle h, tr_math_handle_info *info) {
    if (!valid(t) || !info) return TR_MATH_INVALID;
    uint32_t gen;
    slot *s = slot_of(t, h, &gen);
    if (!s) return refuse(t, TR_MATH_INVALID);
    pthread_mutex_lock(&t->lock);                    /* the fields change only under the mutex */
    uint64_t state = atomic_load_explicit(&s->state, memory_order_acquire);
    int32_t rc = classify(state, gen);
    if (!rc) {
        memset(info, 0, sizeof *info);
        info->kind = s->kind;
        info->flags = s->flags & TR_HANDLE_MUTABLE;
        info->refs = S_REFS(state);
        info->bytes = s->bytes;
        info->slot = (uint32_t)h;
        info->generation = gen;
    }
    pthread_mutex_unlock(&t->lock);
    return rc ? refuse(t, rc) : TR_MATH_OK;
}

int32_t tr_math_handles_stats(tr_math_handles *t, tr_math_handle_stats *out) {
    if (!valid(t) || !out) return TR_MATH_INVALID;
    memset(out, 0, sizeof *out);
    pthread_mutex_lock(&t->lock);
    out->capacity = t->capacity; out->live = t->live; out->deferred = t->deferred; out->open = t->live - t->deferred;
    out->retired = t->retired; out->free_slots = t->free_slots;
    out->inserted = t->inserted; out->closes = t->closes; out->deferred_closes = t->deferred_closes;
    out->destroyed = t->destroyed;
    out->payload_bytes = t->payload_bytes; out->peak_payload_bytes = t->peak_payload_bytes;
    out->table_bytes = t->table_bytes;
    memcpy(out->live_kind, t->live_kind, sizeof out->live_kind);
    pthread_mutex_unlock(&t->lock);
    out->retains = atomic_load_explicit(&t->retains, memory_order_relaxed);
    out->releases = atomic_load_explicit(&t->releases, memory_order_relaxed);
    out->refused_closed = atomic_load_explicit(&t->refused_closed, memory_order_relaxed);
    out->refused_invalid = atomic_load_explicit(&t->refused_invalid, memory_order_relaxed);
    out->allocations = atomic_load_explicit(&t->allocations, memory_order_relaxed);
    out->allocations_total = atomic_load_explicit(&t->allocations_total, memory_order_relaxed);
    return TR_MATH_OK;
}

/* ------------------------------------------------------------------ immutable sparse integer polynomials */
typedef struct {
    uint32_t magic;
    int32_t nvars, degree, reserved;
    uint64_t nterms, bytes;
    /* then int64 coefficients[nterms] and uint8 exponents[nterms * nvars] */
} ipoly;

static int64_t *ipoly_coefficients(ipoly *p) { return (int64_t *)(p + 1); }
static uint8_t *ipoly_exponents(ipoly *p) { return (uint8_t *)(ipoly_coefficients(p) + p->nterms); }
static void ipoly_free(void *p) { if (p) { ((ipoly *)p)->magic = 0; free(p); } }

static uint64_t ipoly_payload_bytes(uint64_t nvars, uint64_t nterms) {
    return tr_budget_add(sizeof(ipoly), tr_budget_mul(nterms, tr_budget_add(8, nvars)));
}

uint64_t tr_math_handles_ipoly_bytes(int32_t nvars, uint64_t nterms) {
    if (nvars < 0) return TR_BUDGET_UNKNOWN;
    return tr_budget_add(ipoly_payload_bytes((uint64_t)nvars, nterms), tr_budget_mul(nterms, 16));
}

static int row_less(const uint8_t *e, size_t nv, uint64_t a, uint64_t b) {
    return memcmp(e + a * nv, e + b * nv, nv) < 0;
}

/* stable bottom-up merge sort of idx[0..n) by exponent row; returns the buffer that holds the order */
static uint64_t *sort_rows(uint64_t *idx, uint64_t *tmp, uint64_t n, const uint8_t *e, size_t nv) {
    for (uint64_t width = 1; width < n; width *= 2) {
        for (uint64_t lo = 0; lo < n; lo += 2 * width) {
            uint64_t mid = lo + width < n ? lo + width : n, hi = lo + 2 * width < n ? lo + 2 * width : n;
            uint64_t i = lo, j = mid, k = lo;
            while (i < mid && j < hi) tmp[k++] = row_less(e, nv, idx[j], idx[i]) ? idx[j++] : idx[i++];
            while (i < mid) tmp[k++] = idx[i++];
            while (j < hi) tmp[k++] = idx[j++];
        }
        uint64_t *swap = idx; idx = tmp; tmp = swap;
    }
    return idx;
}

int32_t tr_math_handles_ipoly_create(tr_math_handles *t, int32_t nvars, uint64_t nterms, const uint8_t *exponents,
                                     const int64_t *coefficients, const tr_math_budget *budget, tr_math_handle *out,
                                     tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (out) *out = 0;
    if (!valid(t) || !out || nvars < 0 || (nterms && (!coefficients || (nvars && !exponents)))) return TR_MATH_INVALID;
    if (nvars > TR_IPOLY_VARIABLES_MAX) return TR_MATH_UNSUPPORTED;
    tr_math_budget dflt;
    if (!budget) { tr_budget_default(&dflt); budget = &dflt; }
    size_t nv = (size_t)nvars;
    uint64_t degree = 0;
    for (uint64_t i = 0; nv && i < nterms; i++) {
        uint64_t d = 0;
        for (size_t v = 0; v < nv; v++) d += exponents[i * nv + v];
        if (d > degree) degree = d;
    }
    tr_math_need need;
    memset(&need, 0, sizeof need);
    need.need[TR_BUDGET_VARIABLES] = (uint64_t)nvars;
    need.need[TR_BUDGET_DEGREE] = degree;
    need.need[TR_BUDGET_SUPPORT] = nterms;
    need.need[TR_BUDGET_BYTES] = tr_math_handles_ipoly_bytes(nvars, nterms);
    int32_t rc = tr_budget_check(budget, &need, refusal);
    if (rc) return rc;
    if ((rc = slot_available(t, refusal))) return rc;
    /* canonical order and merged coefficients in the scratch: idx[k] the row of term k, merged[k] its coefficient */
    uint64_t *a = NULL, *b = NULL, *idx = NULL, *merged = NULL, m = 0;
    if (nterms) {
        a = counted_alloc(t, (size_t)nterms * sizeof *a);
        b = counted_alloc(t, (size_t)nterms * sizeof *b);
        if (!a || !b) { counted_free(t, a); counted_free(t, b); return TR_MATH_MEMORY; }
        for (uint64_t i = 0; i < nterms; i++) a[i] = i;
        idx = nv ? sort_rows(a, b, nterms, exponents, nv) : a;
        merged = idx == a ? b : a;
        for (uint64_t i = 0; i < nterms;) {
            uint64_t row = idx[i], j = i;
            __int128 sum = 0;                         /* exact: at most 2^64 int64 values */
            while (j < nterms && (!nv || !memcmp(exponents + idx[j] * nv, exponents + row * nv, nv)))
                sum += coefficients[idx[j++]];
            if (sum < INT64_MIN || sum > INT64_MAX) { counted_free(t, a); counted_free(t, b); return TR_MATH_OVERFLOW; }
            if (sum) { idx[m] = row; merged[m] = (uint64_t)(int64_t)sum; m++; }   /* idx[m] was already read */
            i = j;
        }
    }
    uint64_t bytes = ipoly_payload_bytes(nv, m);
    ipoly *p = counted_alloc(t, (size_t)bytes);
    if (!p) { counted_free(t, a); counted_free(t, b); return TR_MATH_MEMORY; }
    memset(p, 0, sizeof *p);
    p->magic = IPOLY_MAGIC; p->nvars = nvars; p->nterms = m; p->bytes = bytes;
    int64_t *c = ipoly_coefficients(p);
    uint8_t *e = ipoly_exponents(p);
    uint64_t top = 0;
    for (uint64_t k = 0; k < m; k++) {
        c[k] = (int64_t)merged[k];
        uint64_t d = 0;
        for (size_t v = 0; v < nv; v++) d += (e[k * nv + v] = exponents[idx[k] * nv + v]);
        if (d > top) top = d;
    }
    p->degree = (int32_t)top;
    counted_free(t, a);
    counted_free(t, b);
    rc = put_slot(t, TR_HANDLE_IPOLY, OWNED, p, ipoly_free, bytes, out, refusal);
    if (rc) counted_free(t, p);
    return rc;
}

static void ipoly_view_of(ipoly *p, tr_math_ipoly_view *v) {
    v->nterms = p->nterms; v->nvars = p->nvars; v->degree = p->degree;
    v->exponents = ipoly_exponents(p); v->coefficients = ipoly_coefficients(p);
}

int32_t tr_math_handles_view_evaluate(const tr_math_ipoly_view *v, const int64_t *points, uint64_t npoints,
                                      tr_math_i128 *values, uint8_t *flags, uint64_t *overflowed) {
    if (!v || v->nvars < 0 || (v->nterms && (!v->exponents || !v->coefficients)) || !overflowed ||
        (npoints && (!points || !values || !flags))) return TR_MATH_INVALID;
    uint64_t count = 0;
    for (uint64_t i = 0; i < npoints; i++) {
        __int128 value = 0;
        int ok = tr_ipoly_point(v, points + i * (uint64_t)v->nvars, &value);
        values[i] = tr_i128_set(ok ? value : 0);
        flags[i] = (uint8_t)!ok;
        count += (uint64_t)!ok;
    }
    *overflowed = count;
    return TR_MATH_OK;
}

int32_t tr_math_handles_ipoly_evaluate(tr_math_handles *t, tr_math_handle h, const int64_t *points, uint64_t npoints,
                                       int32_t nvars, tr_math_i128 *values, uint8_t *flags, uint64_t *overflowed,
                                       const tr_math_budget *budget, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!valid(t) || !overflowed || (npoints && (!points || !values || !flags))) return TR_MATH_INVALID;
    slot *s;
    int32_t rc = acquire(t, h, TR_HANDLE_IPOLY, &s);
    if (rc) return rc;
    ipoly *p = s->payload;
    tr_math_budget dflt;
    if (!budget) { tr_budget_default(&dflt); budget = &dflt; }
    if (p->nvars != nvars) rc = TR_MATH_INVALID;
    else rc = tr_budget_check_one(budget, TR_BUDGET_OUTPUT, npoints, refusal);
    if (!rc) {
        tr_math_ipoly_view v;
        ipoly_view_of(p, &v);
        rc = tr_math_handles_view_evaluate(&v, points, npoints, values, flags, overflowed);
    }
    put_back(t, s);
    return rc;
}

int32_t tr_math_handles_ipoly_view(tr_math_handles *t, tr_math_handle h, tr_math_ipoly_view *v) {
    if (!valid(t) || !v) return TR_MATH_INVALID;
    slot *s;
    int32_t rc = acquire(t, h, TR_HANDLE_IPOLY, &s);   /* the caller's release ends the view */
    if (!rc) ipoly_view_of(s->payload, v);
    return rc;
}

uint64_t tr_math_handles_device_bytes(const tr_math_ipoly_view *v, uint64_t npoints) {
    return v && v->nvars >= 0 ? tr_handles_device_bytes(v, npoints) : TR_BUDGET_UNKNOWN;
}

/* ------------------------------------------------------------------ first-slice states owned by a table */
static void free_domain(void *p) { tr_math_domain_free(p); }
static void free_bank(void *p) { tr_math_bank_free(p); }
static void free_cache(void *p) { tr_math_state_cache_free(p); }

int32_t tr_math_handles_domain_create(tr_math_handles *t, const tr_math_domain_spec *spec, const void *key,
                                      uint32_t key_bytes, tr_math_handle origin, tr_math_handle *out,
                                      tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (out) *out = 0;
    if (!valid(t) || !out) return TR_MATH_INVALID;
    int32_t rc = slot_available(t, refusal);
    if (rc) return rc;
    slot *o = NULL;
    if (origin && (rc = acquire(t, origin, TR_HANDLE_DOMAIN, &o))) return rc;
    int32_t error = TR_MATH_OK;
    tr_math_domain *d = tr_math_domain_create(spec, key, key_bytes, o ? o->payload : NULL, &error);
    if (o) put_back(t, o);                           /* the origin is deep-copied */
    if (!d) return error ? error : TR_MATH_MEMORY;
    int64_t bytes = tr_math_domain_bytes(d);
    rc = put_slot(t, TR_HANDLE_DOMAIN, 0, d, free_domain, bytes > 0 ? (uint64_t)bytes : 0, out, refusal);
    if (rc) tr_math_domain_free(d);
    return rc;
}

int32_t tr_math_handles_domain_spec(tr_math_handles *t, tr_math_handle h, tr_math_domain_spec *spec) {
    if (!valid(t) || !spec) return TR_MATH_INVALID;
    slot *s;
    int32_t rc = acquire(t, h, TR_HANDLE_DOMAIN, &s);
    if (rc) return rc;
    rc = tr_math_domain_get_spec(s->payload, spec);
    put_back(t, s);
    return rc;
}

int32_t tr_math_handles_bank_create(tr_math_handles *t, int32_t ndata, int32_t nterms, const uint8_t *exponents,
                                    const int64_t *moments, tr_math_handle *out, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (out) *out = 0;
    if (!valid(t) || !out) return TR_MATH_INVALID;
    int32_t rc = slot_available(t, refusal);
    if (rc) return rc;
    int32_t error = TR_MATH_OK;
    tr_math_bank *bank = tr_math_bank_create(ndata, nterms, exponents, moments, &error);
    if (!bank) return error ? error : TR_MATH_MEMORY;
    rc = put_slot(t, TR_HANDLE_BANK, 0, bank, free_bank, 0, out, refusal);
    if (rc) tr_math_bank_free(bank);
    return rc;
}

int32_t tr_math_handles_bank_moment(tr_math_handles *t, tr_math_handle h, const uint8_t *exponents, int64_t *out) {
    if (!valid(t) || !out) return TR_MATH_INVALID;
    slot *s;
    int32_t rc = acquire(t, h, TR_HANDLE_BANK, &s);
    if (rc) return rc;
    rc = tr_math_bank_moment(s->payload, exponents, out);
    put_back(t, s);
    return rc;
}

int32_t tr_math_handles_cache_create(tr_math_handles *t, int32_t capacity, uint64_t max_bytes, tr_math_handle *out,
                                     tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (out) *out = 0;
    if (!valid(t) || !out) return TR_MATH_INVALID;
    int32_t rc = slot_available(t, refusal);
    if (rc) return rc;
    int32_t error = TR_MATH_OK;
    tr_math_state_cache *cache = tr_math_state_cache_create(capacity, max_bytes, &error);
    if (!cache) return error ? error : TR_MATH_MEMORY;
    rc = put_slot(t, TR_HANDLE_STATE_CACHE, TR_HANDLE_MUTABLE, cache, free_cache, max_bytes, out, refusal);
    if (rc) tr_math_state_cache_free(cache);
    return rc;
}

int32_t tr_math_handles_cache_put(tr_math_handles *t, tr_math_handle cache, tr_math_handle domain, const void *basis,
                                  uint32_t basis_bytes, tr_math_handle bank) {
    if (!valid(t)) return TR_MATH_INVALID;
    slot *d, *b, *c;
    int32_t rc = acquire(t, domain, TR_HANDLE_DOMAIN, &d);
    if (rc) return rc;
    if ((rc = acquire(t, bank, TR_HANDLE_BANK, &b))) { put_back(t, d); return rc; }
    if (!(rc = lock_slot(t, cache, TR_HANDLE_STATE_CACHE, &c))) {
        rc = tr_math_state_cache_put(((box *)c->payload)->object, d->payload, basis, basis_bytes, b->payload);
        unlock_slot(t, c);
    }
    put_back(t, b);
    put_back(t, d);
    return rc;
}

int32_t tr_math_handles_cache_get(tr_math_handles *t, tr_math_handle cache, tr_math_handle domain, const void *basis,
                                  uint32_t basis_bytes, int32_t ndata, tr_math_handle *bank, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (bank) *bank = 0;
    if (!valid(t) || !bank) return TR_MATH_INVALID;
    slot *d, *c;
    int32_t rc = acquire(t, domain, TR_HANDLE_DOMAIN, &d);
    if (rc) return rc;
    if ((rc = lock_slot(t, cache, TR_HANDLE_STATE_CACHE, &c))) { put_back(t, d); return rc; }
    int32_t error = TR_MATH_OK;
    tr_math_bank *copy = tr_math_state_cache_get(((box *)c->payload)->object, d->payload, basis, basis_bytes, ndata,
                                                 &error);
    unlock_slot(t, c);                               /* the clone is independent of the cache */
    put_back(t, d);
    if (!copy) return error;                         /* a miss (OK) or the cache's refusal */
    rc = put_slot(t, TR_HANDLE_BANK, 0, copy, free_bank, 0, bank, refusal);
    if (rc) tr_math_bank_free(copy);
    return rc;
}

int32_t tr_math_handles_cache_stats(tr_math_handles *t, tr_math_handle cache, tr_math_state_stats *out) {
    if (!valid(t) || !out) return TR_MATH_INVALID;
    slot *c;
    int32_t rc = lock_slot(t, cache, TR_HANDLE_STATE_CACHE, &c);
    if (rc) return rc;
    rc = tr_math_state_cache_stats(((box *)c->payload)->object, out);
    unlock_slot(t, c);
    return rc;
}
