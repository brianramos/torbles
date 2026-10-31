/* State ownership, concurrency and the native ABI of the mathematical modules (FND-013).
 * SPDX-License-Identifier: Apache-2.0
 *
 * Handle tables. A tr_math_handles table is created and owned by its caller; there is no global table. It issues
 * 64-bit handles: bits 0..31 are the slot index, bits 32..63 the slot's generation (>= 1), so 0 and every value below
 * 2^32 are never issued. Each live handle owns one payload with a kind tag and a destructor. A slot is reused with
 * the next generation after its payload is freed; a slot whose generation reaches the table's generation limit is
 * retired, never reused, so a (slot, generation) pair is issued at most once in the table's lifetime and a released
 * handle can never be accepted again. Errors are decided from one atomic read of the slot and do not depend on
 * timing: a handle that was issued and closed is TR_MATH_CLOSED (whether or not its payload was already freed), a
 * handle that was never issued (or names another kind) is TR_MATH_INVALID.
 *
 * Lifetimes. An operation retains the handles it uses for its duration (tr_math_handles_retain / _release: a C11
 * atomic reference count in the slot's state word). tr_math_handles_close marks the handle closed at once (no new
 * retain succeeds) and frees the payload only when no operation holds it; otherwise the last release frees it
 * (a deferred close). So a close never frees a payload under a live native operation.
 *
 * Locking. retain and release are lock-free (one compare-and-swap on the slot's state word); immutable payloads are
 * read concurrently by any number of retained operations. The table's mutex serializes the table's own mutations:
 * insert, close, the final free of a payload (the destructor runs with the mutex held and must not call the table),
 * stats/info snapshots and free. A mutable payload (inserted with TR_HANDLE_MUTABLE) has its own error-checking
 * mutex: tr_math_handles_lock = retain + lock, tr_math_handles_unlock = unlock + release, so every operation on one
 * mutable cache is serialized. Lock order: a payload mutex may be held while taking the table mutex, never the
 * reverse. tr_math_handles_free must not run concurrently with any other call on the same table.
 *
 * Payload kinds of this module: TR_HANDLE_IPOLY (an immutable sparse integer polynomial, evaluated exactly on the CPU
 * and by the CUDA kernel of math_handles_gpu.cu), and the first-slice states TR_HANDLE_DOMAIN (tr_math_domain),
 * TR_HANDLE_BANK (tr_math_bank) and TR_HANDLE_STATE_CACHE (tr_math_state_cache, mutable, serialized). Other modules
 * insert their own states with kinds >= TR_HANDLE_USER.
 *
 * Polynomial evaluation (the shared inline functions below are compiled into the CPU library and the kernels). A point
 * of nvars int64 coordinates has the exact value sum_t c_t * prod_v x_v^e_tv (with x^0 = 1, also for x = 0). It is
 * returned exactly as a 128-bit integer, or flagged as overflow (value 0, never wrapped) exactly when some canonical
 * term c_t * prod_v x_v^e_tv or the sum lies outside [-2^127, 2^127 - 1]. Terms are computed as unsigned magnitudes
 * with overflow detection and summed in a 192-bit accumulator, so the flag does not depend on the order of the terms.
 *
 * ABI registry. Every struct that crosses a ctypes boundary is registered once in each library it crosses into, with
 * TR_MATH_ABI(type, version) in a torbles/math*.c file (libalg.so) and with TR_MGPU_ABI(type, version) in a
 * torbles/math_*_gpu.cu file (the CUDA library), one line each, in the files of the family that defines it (the
 * structs of the modules that predate the registry are registered in math_handles.c / math_handles_gpu.cu).
 * tr_math_handles_abi / tr_mgpu_handles_abi
 * report every registered entry (name, size, alignment, version as compiled); torbles.math_handles.abi_check compares
 * them with every ctypes Structure of the torbles.math* modules and fails loudly on a mismatch. */
#ifndef TORBLES_MATH_HANDLES_H
#define TORBLES_MATH_HANDLES_H
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include "math_num.h"
#include "math_budgets.h"
#include "math_domain.h"
#include "math_state.h"
#ifdef __cplusplus
extern "C" {
#endif

#define TR_HANDLES_ABI 1
/* A closed (released) handle. Local to FND-013: not yet in math_num.h, whose next free code it takes. */
enum { TR_MATH_CLOSED = -17 };

enum { TR_HANDLE_ANY = 0,             /* retain / lock: accept any kind */
       TR_HANDLE_IPOLY = 1,           /* immutable sparse integer polynomial (this module) */
       TR_HANDLE_DOMAIN = 2,          /* immutable tr_math_domain (math_domain.h) */
       TR_HANDLE_BANK = 3,            /* immutable tr_math_bank (mathematics.h) */
       TR_HANDLE_STATE_CACHE = 4,     /* mutable tr_math_state_cache (math_state.h), every call serialized */
       TR_HANDLE_KINDS = 8,           /* stats: live_kind[k] for k = 1..7, live_kind[0] for every other kind */
       TR_HANDLE_USER = 16 };         /* the first kind other modules use for their own states */
enum { TR_HANDLE_MUTABLE = 1 };       /* insert flag: the payload is mutable (serialized by its own mutex) */

#define TR_HANDLES_CAPACITY_MAX (UINT32_C(1) << 24)   /* slots per table (representation limit) */
#define TR_HANDLES_GENERATIONS UINT32_MAX              /* default generations per slot */
#define TR_HANDLES_REFS_MAX ((UINT64_C(1) << 29) - 1) /* concurrent references per handle */
#define TR_IPOLY_EXPONENT_MAX 255                      /* exponents are bytes */
#define TR_IPOLY_VARIABLES_MAX 65535                   /* coordinates per point (representation limit) */

typedef struct tr_math_handles_s tr_math_handles;
typedef uint64_t tr_math_handle;
typedef void (*tr_math_handle_destroy)(void *payload);

/* Counters of one table (27 uint64; sizeof == 216). Slots now: capacity, live (issued and not yet freed: open plus
 * deferred), open, deferred (closed but still retained), retired, free_slots. Totals: inserted, closes,
 * deferred_closes, destroyed (payload destructors run), retains / releases (successful; equal when no operation is
 * in flight), refused_closed / refused_invalid (calls refused with TR_MATH_CLOSED / TR_MATH_INVALID). Memory:
 * payload_bytes (declared bytes of the live payloads: exact for polynomials, tr_math_domain_bytes for domains, the
 * retained-bytes bound max_bytes for state caches, 0 for banks, whose size mathematics.h does not report),
 * peak_payload_bytes, table_bytes (the fixed allocation), allocations / allocations_total (blocks this module
 * allocated that are live now / ever: the table, polynomial payloads, mutable-payload boxes, sorting scratch).
 * live_kind: live handles by kind. */
typedef struct {
    uint64_t capacity, live, open, deferred, retired, free_slots;
    uint64_t inserted, closes, deferred_closes, destroyed;
    uint64_t retains, releases, refused_closed, refused_invalid;
    uint64_t payload_bytes, peak_payload_bytes, table_bytes;
    uint64_t allocations, allocations_total;
    uint64_t live_kind[TR_HANDLE_KINDS];
} tr_math_handle_stats;

/* One open handle (sizeof == 40): kind, flags (TR_HANDLE_MUTABLE), refs (operations holding it now), declared bytes,
 * slot and generation (the two halves of the handle). */
typedef struct {
    int32_t kind, flags;
    uint64_t refs, bytes;
    uint32_t slot, generation;
    uint64_t reserved;
} tr_math_handle_info;

/* An immutable polynomial as the CPU code and the kernel read it (sizeof == 32): nterms canonical terms (distinct
 * exponent vectors in increasing lexicographic order, nonzero coefficients), nvars coordinates per point, degree =
 * the largest total degree of a term. */
typedef struct {
    uint64_t nterms;
    int32_t nvars, degree;
    const uint8_t *exponents;      /* nterms x nvars, row-major */
    const int64_t *coefficients;   /* nterms */
} tr_math_ipoly_view;

/* ------------------------------------------------------------------ the ABI registry */
#define TR_MATH_ABI_NAME 48
typedef struct {
    char name[TR_MATH_ABI_NAME];   /* the C type name */
    uint32_t size, align;          /* sizeof and alignment as compiled into this library */
    int32_t version, reserved;     /* the struct's ABI version (its header's *_ABI macro, else 1); reserved == 0 */
} tr_math_abi_entry;               /* sizeof == 64 */

#ifdef __cplusplus
#define TR_ABI_ALIGNOF(type) alignof(type)
#else
#define TR_ABI_ALIGNOF(type) _Alignof(type)
#endif
#define TR_ABI_CAT2(a, b) a##b
#define TR_ABI_CAT(a, b) TR_ABI_CAT2(a, b)
/* a constant entry in a named linker section; the section is the registry (no mutable state, no central list).
 * Sections named like C identifiers are never padded by AddressSanitizer (GCC and LLVM). */
#define TR_ABI_ENTRY(section_name, type, version)                                                          \
    static const tr_math_abi_entry TR_ABI_CAT(tr_abi_entry_, __LINE__)                                     \
        __attribute__((used, section(section_name), aligned(8))) =                                         \
        {#type, (uint32_t)sizeof(type), (uint32_t)TR_ABI_ALIGNOF(type), (int32_t)(version), 0}
/* one line per struct that crosses the ctypes boundary into libalg.so (in a torbles/math*.c file) */
#define TR_MATH_ABI(type, version) TR_ABI_ENTRY("tr_math_abi", type, version)
/* one line per struct that crosses the ctypes boundary into the CUDA library (in a torbles/math_*_gpu.cu file) */
#define TR_MGPU_ABI(type, version) TR_ABI_ENTRY("tr_mgpu_abi", type, version)

/* Every TR_MATH_ABI entry linked into libalg.so. entry_size must equal sizeof(tr_math_abi_entry) (else
 * TR_MATH_INVALID). Writes min(count, capacity) entries to out and returns count. */
int32_t tr_math_handles_abi(tr_math_abi_entry *out, int32_t capacity, uint32_t entry_size);

/* ------------------------------------------------------------------ tables */
/* the fixed allocation of a table of capacity slots (its bytes need); UNKNOWN past 64 bits */
uint64_t tr_math_handles_table_bytes(uint64_t capacity);
/* A table of capacity slots (1..TR_HANDLES_CAPACITY_MAX), each reused for at most generation_limit generations
 * (0 = TR_HANDLES_GENERATIONS). Budget (NULL = defaults): states = capacity, bytes = tr_math_handles_table_bytes.
 * NULL with *error set on failure. */
tr_math_handles *tr_math_handles_create(uint32_t capacity, uint32_t generation_limit, const tr_math_budget *,
                                        int32_t *error, tr_math_refusal *);
/* Frees every live payload and the table; NULL is OK. Refused (TR_MATH_INVALID, nothing freed) while any handle is
 * retained. No other call on the table may run concurrently with this one. */
int32_t tr_math_handles_free(tr_math_handles *);

/* Registers a payload (kind > 0, destroy != NULL) and writes its new handle. A full table (no free slot) is
 * TR_MATH_BUDGET with the refusal resource states (need = live + retired + 1, limit = capacity); on any failure *out
 * is 0 and the payload stays the caller's. */
int32_t tr_math_handles_insert(tr_math_handles *, int32_t kind, int32_t flags, void *payload,
                               tr_math_handle_destroy, uint64_t bytes, tr_math_handle *out, tr_math_refusal *);
/* A reference to an open handle of this kind (TR_HANDLE_ANY: any kind) and its payload; every successful retain is
 * matched by one release. Refused: TR_MATH_CLOSED (closed), TR_MATH_INVALID (never issued, wrong kind). */
int32_t tr_math_handles_retain(tr_math_handles *, tr_math_handle, int32_t kind, void **payload);
/* Drops one reference (the handle may be closed meanwhile); the last release of a closed handle frees its payload.
 * TR_MATH_INVALID when the handle holds no reference. */
int32_t tr_math_handles_release(tr_math_handles *, tr_math_handle);
/* retain + lock the mutex of a mutable payload (TR_MATH_INVALID for an immutable one) / unlock + release, by the
 * thread that locked it */
int32_t tr_math_handles_lock(tr_math_handles *, tr_math_handle, int32_t kind, void **payload);
int32_t tr_math_handles_unlock(tr_math_handles *, tr_math_handle);
/* Closes an open handle: 0 when its payload was freed now, 1 when the free is deferred to the last release;
 * TR_MATH_CLOSED when already closed, TR_MATH_INVALID when never issued. */
int32_t tr_math_handles_close(tr_math_handles *, tr_math_handle);
/* the description of an open handle (TR_MATH_CLOSED / TR_MATH_INVALID otherwise) */
int32_t tr_math_handles_info(tr_math_handles *, tr_math_handle, tr_math_handle_info *);
int32_t tr_math_handles_stats(tr_math_handles *, tr_math_handle_stats *);

/* ------------------------------------------------------------------ immutable sparse integer polynomials */
/* bytes a polynomial of nterms input terms needs: its payload (header, coefficients, exponents) and the sorting
 * scratch (two uint64 per term); UNKNOWN past 64 bits */
uint64_t tr_math_handles_ipoly_bytes(int32_t nvars, uint64_t nterms);
/* An immutable polynomial in nvars coordinates from nterms terms (exponents: nterms x nvars bytes, row-major;
 * coefficients: nterms int64), stored canonically: equal exponent vectors merged exactly (a merged coefficient
 * outside int64 is TR_MATH_OVERFLOW), zero terms dropped, rows sorted. Budget (NULL = defaults): variables = nvars,
 * degree = the largest total degree of an input term, support = nterms, bytes = tr_math_handles_ipoly_bytes; checked
 * before anything is allocated. nvars above TR_IPOLY_VARIABLES_MAX is TR_MATH_UNSUPPORTED. */
int32_t tr_math_handles_ipoly_create(tr_math_handles *, int32_t nvars, uint64_t nterms, const uint8_t *exponents,
                                     const int64_t *coefficients, const tr_math_budget *, tr_math_handle *out,
                                     tr_math_refusal *);
/* Exact values at npoints points (points: npoints x nvars int64, row-major; nvars must equal the polynomial's):
 * values[i] exact and flags[i] = 0, or values[i] = 0 and flags[i] = 1 (overflow, see above); *overflowed counts the
 * flagged points. Budget: output = npoints. The handle is retained for the duration of the call. */
int32_t tr_math_handles_ipoly_evaluate(tr_math_handles *, tr_math_handle, const int64_t *points, uint64_t npoints,
                                       int32_t nvars, tr_math_i128 *values, uint8_t *flags, uint64_t *overflowed,
                                       const tr_math_budget *, tr_math_refusal *);
/* Retains the handle and describes its immutable terms. The view is valid until the caller's matching
 * tr_math_handles_release, even if the handle is closed meanwhile (e.g. around the CUDA evaluation). */
int32_t tr_math_handles_ipoly_view(tr_math_handles *, tr_math_handle, tr_math_ipoly_view *);
/* the evaluation of a view without a table (the CPU reference of the kernel); no budget */
int32_t tr_math_handles_view_evaluate(const tr_math_ipoly_view *, const int64_t *points, uint64_t npoints,
                                      tr_math_i128 *values, uint8_t *flags, uint64_t *overflowed);

/* ------------------------------------------------------------------ first-slice states owned by a table */
/* A tr_math_domain (math_domain.h semantics); origin: 0 or an open domain handle, retained during the call. */
int32_t tr_math_handles_domain_create(tr_math_handles *, const tr_math_domain_spec *, const void *key,
                                      uint32_t key_bytes, tr_math_handle origin, tr_math_handle *out,
                                      tr_math_refusal *);
int32_t tr_math_handles_domain_spec(tr_math_handles *, tr_math_handle, tr_math_domain_spec *);
/* A tr_math_bank (mathematics.h semantics) / one of its moments */
int32_t tr_math_handles_bank_create(tr_math_handles *, int32_t ndata, int32_t nterms, const uint8_t *exponents,
                                    const int64_t *moments, tr_math_handle *out, tr_math_refusal *);
int32_t tr_math_handles_bank_moment(tr_math_handles *, tr_math_handle, const uint8_t *exponents, int64_t *out);
/* A mutable tr_math_state_cache (math_state.h semantics). put / get / stats retain the cache, domain and bank
 * handles for their duration and run under the cache's mutex, so concurrent callers are serialized natively.
 * get writes a new bank handle (an owned clone) on a hit and 0 on a miss. */
int32_t tr_math_handles_cache_create(tr_math_handles *, int32_t capacity, uint64_t max_bytes, tr_math_handle *out,
                                     tr_math_refusal *);
int32_t tr_math_handles_cache_put(tr_math_handles *, tr_math_handle cache, tr_math_handle domain, const void *basis,
                                  uint32_t basis_bytes, tr_math_handle bank);
int32_t tr_math_handles_cache_get(tr_math_handles *, tr_math_handle cache, tr_math_handle domain, const void *basis,
                                  uint32_t basis_bytes, int32_t ndata, tr_math_handle *bank, tr_math_refusal *);
int32_t tr_math_handles_cache_stats(tr_math_handles *, tr_math_handle cache, tr_math_state_stats *);

/* bytes the CUDA evaluation copies to the device (inline for the CUDA library; exported for the bindings) */
static inline uint64_t tr_handles_device_bytes(const tr_math_ipoly_view *v, uint64_t npoints) {
    uint64_t nv = v->nvars > 0 ? (uint64_t)v->nvars : 0;
    uint64_t bytes = tr_budget_mul(v->nterms, tr_budget_add(nv, 8));          /* exponents, coefficients */
    uint64_t per_point = tr_budget_add(tr_budget_mul(nv, 8), 17);              /* coordinates, value, flag */
    bytes = tr_budget_add(bytes, tr_budget_mul(npoints, per_point));
    return tr_budget_add(bytes, 8);                                           /* the overflow counter */
}
uint64_t tr_math_handles_device_bytes(const tr_math_ipoly_view *, uint64_t npoints);

/* ------------------------------------------------------------------ exact evaluation (CPU and CUDA) */
/* *a = *a * m for a 128-bit magnitude and a 64-bit factor; 0 (unchanged) when the product exceeds 2^128 - 1 */
TR_NUM_HD int tr_ipoly_mul(unsigned __int128 *a, uint64_t m) {
    uint64_t lo = (uint64_t)*a, hi = (uint64_t)(*a >> 64);
    if (!hi) { *a = (unsigned __int128)lo * m; return 1; }           /* below 2^64: one exact product */
    unsigned __int128 plo = (unsigned __int128)lo * m, phi = (unsigned __int128)hi * m;
    if (phi >> 64) return 0;
    unsigned __int128 r = plo + (phi << 64);
    if (r < plo) return 0;
    *a = r;
    return 1;
}

/* the index of the lowest set bit (x != 0), on the host and in device code */
TR_NUM_HD int tr_ipoly_ctz(uint64_t x) {
#if defined(__CUDA_ARCH__)
    return __ffsll((long long)x) - 1;
#else
    return __builtin_ctzll(x);
#endif
}
/* bit 8k + 7 set for each nonzero byte k of w */
TR_NUM_HD uint64_t tr_ipoly_nonzero_bytes(uint64_t w) {
    const uint64_t low7 = UINT64_C(0x7f7f7f7f7f7f7f7f);
    return (((w & low7) + low7) | w) & ~low7;
}

/* One point x[0..nvars-1]: 1 with *value, or 0 when a term or the sum lies outside the int128 range. Each exponent
 * row is read eight bytes at a time and only its nonzero exponents are visited; a zero factor anywhere makes the
 * term exactly 0, so a magnitude past 2^128 - 1 counts only once the whole row was read. */
TR_NUM_HD int tr_ipoly_point(const tr_math_ipoly_view *p, const int64_t *x, __int128 *value) {
    const unsigned __int128 top = (unsigned __int128)1 << 127;         /* |int128 min| */
    unsigned __int128 lo = 0;                                          /* the 192-bit sum: hi * 2^128 + lo */
    int64_t hi = 0;
    const int32_t n = p->nvars;
    for (uint64_t t = 0; t < p->nterms; t++) {
        const uint8_t *e = p->exponents + t * (uint64_t)n;
        int64_t c = p->coefficients[t];
        if (c == 0) continue;
        int negative = c < 0, zero = 0, wide = 0;                      /* wide: the magnitude passed 2^128 - 1 */
        unsigned __int128 magnitude = tr_abs_u64(c);
        for (int32_t v = 0; v < n && !zero; v += 8) {
            uint64_t word = 0;                                         /* byte k = exponent v + k */
#if defined(__CUDA_ARCH__) || !defined(__BYTE_ORDER__) || __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
            for (int32_t b = 0; b < 8 && v + b < n; b++) word |= (uint64_t)e[v + b] << (8 * b);   /* no stack */
#else
            memcpy(&word, e + v, n - v < 8 ? (size_t)(n - v) : 8);
#endif
            for (uint64_t mask = tr_ipoly_nonzero_bytes(word); mask; mask &= mask - 1) {
                int k = tr_ipoly_ctz(mask) >> 3;
                unsigned power = (unsigned)(word >> (8 * k)) & 0xffu;
                int64_t xv = x[v + k];
                if (xv == 0) { zero = 1; break; }
                if (xv < 0 && (power & 1)) negative = !negative;
                uint64_t base = tr_abs_u64(xv);
                if (base == 1 || wide) continue;
                for (unsigned j = 0; j < power; j++)                   /* magnitudes only grow: |x| >= 2 */
                    if (!tr_ipoly_mul(&magnitude, base)) { wide = 1; break; }
            }
        }
        if (zero) continue;                                            /* the term is exactly 0 */
        if (wide || (negative ? magnitude > top : magnitude >= top)) return 0;   /* the term is outside int128 */
        unsigned __int128 term = negative ? (unsigned __int128)0 - magnitude : magnitude;   /* two's complement */
        unsigned __int128 before = lo;
        lo += term;
        hi += (negative ? -1 : 0) + (lo < before ? 1 : 0);             /* sign extension and carry: |hi| <= nterms */
    }
    int64_t sign = (lo >> 127) ? -1 : 0;
    if (hi != sign) return 0;                                          /* the sum is outside int128 */
    *value = (__int128)lo;
    return 1;
}

#ifdef __cplusplus
}
#endif
#endif
