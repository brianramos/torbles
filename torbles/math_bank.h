/* FND-004: grouped exact moment banks over a data scope (math_scope.h), their rollup along a refinement map, and a
 * bounded cache of immutable banks keyed by an exact semantic descriptor.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Basis. A bank's basis is an ordered list of nterms distinct monomials over the scope's ncols value columns, given
 * as an exponent matrix (nterms rows of ncols uint8 exponents, row-major): term j is prod_c x_c^e[j][c], total degree
 * <= TR_BANK_MAX_DEGREE (4), nterms <= TR_BANK_MAX_TERMS (1024; all monomials of degree <= 2 over 32 columns are 561).
 * The zero row is the constant term. A basis outside these limits is TR_MATH_UNSUPPORTED; a duplicate term or a
 * malformed argument TR_MATH_INVALID.
 *
 * States, per group g < ngroups (the scope's dense groups):
 *   rows[g]                 the bag multiplicity: selected rows of group g;
 *   count[g * nterms + j]   the rows of group g where term j is valid;
 *   sum[g * nterms + j]     the exact sum of term j's logical values over those rows (signed 128 bits).
 * Values are the scope's logical integers (decimal value = logical / 10^scale per factor, so term j's value carries the
 * scale sum_c e[j][c] * scale_c). Validity: TR_BANK_SQL (per term, SQL semantics: SUM(x*y) skips a row where x or y
 * is NULL; the constant term counts every row) or TR_BANK_JOINT (listwise: a row counts for every term only when every
 * column of the basis, i.e. every column with a nonzero exponent in some term, is present). Presence is explicit: group
 * g is present iff rows[g] > 0, also when all its values are NULL (then count 0 and sum 0, but present); a group
 * without selected rows is absent (rows 0), which is not a zero group.
 *
 * Exactness. Before accumulating, the bank proves from the scope's column extremes and row counts that every product
 * and every partial sum fits: for each term, N * prod_c max(|min_c|, |max_c|, 1)^e_c <= 2^127 - 1, N the number of
 * rows that can be valid for it (the selected rows, at most the present values of each of its columns). Otherwise the
 * request is refused with TR_MATH_OVERFLOW before anything is computed; nothing ever wraps. The bound holds for every
 * subset of rows, so chunks, merges and rollups of a proven bank fit as well; rollups also check their sums.
 *
 * Accumulation. tr_bank_init creates a builder (the one mutable object: owned by one caller, calls on it serialized)
 * over a scope that must outlive it; tr_bank_accumulate adds the selected rows of a physical row range [begin, end);
 * tr_bank_merge adds another builder of the same scope (or an equal descriptor), basis and mode; ranges must be
 * disjoint (a row is never counted twice: TR_MATH_INVALID); tr_bank_finish moves the states into an immutable bank once
 * every row 0..rows-1 is covered (the builder is spent). tr_bank_build is init + accumulate(0, rows) + finish. A chunked
 * or merged accumulation is bit for bit the one-shot bank. The CUDA path (tr_mgpu_bank_build, math_bank_gpu.cu) writes
 * the same states, which tr_bank_from_states adopts (checking their consistency: shape, total rows, count <= rows,
 * sum 0 without valid rows, the constant term, the proven bound).
 *
 * Rollup. A map from the ngroups fine group ids to coarse ids < coarse_groups (tr_scope_rollup_map gives the map between
 * two scopes of the same rows) builds the coarse bank exactly: counts and sums add, a coarse group is present iff one
 * of its fine groups is. With `origin` (the coarse scope) the result is the bank of that scope and may enter a cache.
 * Before adding, the magnitudes gathered into each coarse cell are bounded by 2^127 - 1 (tr_bank_rollup_bound, in any
 * order; else TR_MATH_OVERFLOW): never refused for a proven bank, only for adopted states whose sums cannot add.
 *
 * Budgets (math_budgets.h), checked before allocation (tr_bank_need, tr_bank_rollup_need): variables = columns of the
 * basis, degree, support = nterms, states = ngroups * nterms, bytes = every allocation (builder or bank: the states
 * 8 * ngroups + 24 * ngroups * nterms, the row-coverage bitmap, the basis, the objects). The CUDA entry points check
 * the bytes of their device copies instead (tr_bank_device_need).
 *
 * Cache. capacity 1..TR_BANK_CACHE_CAPACITY_MAX entries and a retained-byte budget max_bytes (default 8 MiB) that
 * covers the cache's fixed allocation, every owned key and every owned bank; least recently used entries are evicted to
 * make room (one staged entry may exist beside the retained ones while it is installed, so a failure changes nothing).
 * The key is exact, never a hash alone: the scope's whole descriptor (tr_scope_key_*: source identity and version, row
 * count, selection bits, column specs, key specs with lookup contents, or the data of an array scope), the basis and
 * the mode. So a different predicate (selection), NULL policy (mode, validity columns), revision or group grain never
 * hits. A get returns an independently owned clone (or NULL: a miss); a put stores a clone of a bank built from that
 * scope; refresh rebuilds and replaces, dropping the entries of older versions of the same table. The cache matches
 * descriptors only: whether a scope is still fresh is the caller's check (tr_scope_fresh). Calls on one cache must be
 * externally serialized (the Python wrapper holds a lock); returned banks are independent of the cache.
 *
 * Errors: TR_MATH_INVALID, _MEMORY, _OVERFLOW, _BUDGET (with the tr_math_refusal), _UNSUPPORTED, TR_MATH_DOMAIN_ERROR
 * (a bank put under a scope it was not built from). Constructors return NULL and set *error, inputs unchanged. */
#ifndef TORBLES_MATH_BANK_H
#define TORBLES_MATH_BANK_H
#include <stdint.h>
#include <string.h>
#include "math_num.h"
#include "math_budgets.h"
#include "math_scope.h"
#ifdef __cplusplus
extern "C" {
#endif

#define TR_BANK_MAX_TERMS 1024
#define TR_BANK_MAX_DEGREE 4
#define TR_BANK_CACHE_CAPACITY_MAX 128
#define TR_BANK_CACHE_BYTES_DEFAULT UINT64_C(8388608)
enum { TR_BANK_SQL = 0, TR_BANK_JOINT = 1 };          /* validity modes */
enum { TR_BANK_PUT = 0, TR_BANK_PUT_REFRESH = 1 };    /* tr_bank_cache_put */
enum { TR_BANK_NEED_CPU = 0, TR_BANK_NEED_DEVICE = 1, TR_BANK_NEED_ADOPT = 2 };   /* tr_bank_need targets */

/* one compiled term: its support and factors (columns increasing) */
typedef struct {
    uint32_t need;            /* bit c: column c has a nonzero exponent */
    uint8_t nf, degree;       /* factors 0..4, total degree 0..4 */
    uint8_t col[4], exp[4];   /* factor columns and exponents (>= 1) */
    uint8_t reserved[2];
} tr_bank_term;               /* 16 bytes */

/* the state layout (see the header comment); sum[i] = hi * 2^64 + (uint64_t)lo */
typedef struct {
    uint64_t ngroups;
    int32_t nterms, reserved;
    uint64_t *rows;           /* [ngroups] */
    uint64_t *count;          /* [ngroups * nterms] */
    tr_math_i128 *sum;        /* [ngroups * nterms] */
} tr_bank_states;             /* 40 bytes */

typedef struct {
    uint64_t ngroups, present;  /* groups, groups with rows */
    uint64_t rows;              /* selected rows (the sum of rows[g]) */
    uint64_t bytes;             /* bytes the bank owns */
    uint64_t digest[2];         /* the descriptor digest of its scope (0 when unscoped) */
    int32_t ncols, nterms, mode, degree;
    uint32_t joint;             /* columns of the basis */
    int32_t scoped;             /* 1: the bank of a scope (built, adopted or rolled up with an origin) */
} tr_bank_info;                 /* 72 bytes */

typedef struct {
    uint64_t hits, misses, puts, evictions, refreshes, invalidations, bytes, max_bytes;
    int32_t capacity, entries;
} tr_bank_cache_stats;          /* 72 bytes; counters saturate */

typedef struct tr_bank_s tr_bank;
typedef struct tr_bank_builder_s tr_bank_builder;
typedef struct tr_bank_cache_s tr_bank_cache;

/* ------------------------------------------------------------------ shared helpers (CPU code, CUDA host and kernels) */
static inline uint32_t tr_bank_term_key(const tr_bank_term *t) {          /* canonical: distinct terms, distinct keys */
    uint32_t k = 0;
    for (int f = 0; f < t->nf; f++) k |= (uint32_t)((t->col[f] << 3) | t->exp[f]) << (8 * f);
    return k;
}
/* Compiles and validates an exponent matrix: terms[nterms], the basis's columns (*joint) and degree (*degree). */
static inline int32_t tr_bank_compile(const uint8_t *exps, int32_t nterms, int32_t ncols, tr_bank_term *terms,
                                      uint32_t *joint, int32_t *degree) {
    if (nterms < 0 || ncols < 0 || ncols > TR_SCOPE_MAX_COLUMNS || (nterms && ncols && !exps) || (nterms && !terms) ||
        !joint || !degree) return -1;                                     /* TR_MATH_INVALID */
    if (nterms > TR_BANK_MAX_TERMS) return -12;                           /* TR_MATH_UNSUPPORTED */
    uint32_t all = 0;
    int32_t top = 0;
    for (int32_t j = 0; j < nterms; j++) {
        tr_bank_term *t = &terms[j];
        memset(t, 0, sizeof *t);
        int d = 0;
        for (int32_t c = 0; c < ncols; c++) {
            int e = exps[(uint64_t)j * (uint64_t)ncols + (uint64_t)c];
            if (!e) continue;
            if (e > TR_BANK_MAX_DEGREE || d + e > TR_BANK_MAX_DEGREE) return -12;
            t->col[t->nf] = (uint8_t)c; t->exp[t->nf] = (uint8_t)e; t->nf++;
            t->need |= UINT32_C(1) << c;
            d += e;
        }
        t->degree = (uint8_t)d;
        all |= t->need;
        if (d > top) top = d;
        uint32_t key = tr_bank_term_key(t);
        for (int32_t i = 0; i < j; i++)
            if (terms[i].nf == t->nf && tr_bank_term_key(&terms[i]) == key) return -1;   /* a duplicate term */
    }
    *joint = all; *degree = top;
    return 0;
}
/* the rows term j can be valid on: the selected rows, at most the present values of each column it needs */
static inline uint64_t tr_bank_term_rows(const tr_scope_view *v, const tr_bank_term *t, int32_t mode, uint32_t joint) {
    uint32_t need = mode == TR_BANK_JOINT ? joint : t->need;
    uint64_t rows = v->selected;
    for (int32_t c = 0; c < v->ncols; c++)
        if (((need >> c) & 1) && v->col[c].nvalid < rows) rows = v->col[c].nvalid;
    return rows;
}
static inline unsigned __int128 tr_bank_satmul(unsigned __int128 a, unsigned __int128 b, unsigned __int128 limit) {
    if (!a || !b) return 0;
    return a > limit / b ? limit + 1 : a * b;
}
/* |term| <= this bound (max(|min|, |max|, 1) per factor) on every row where it is valid */
static inline unsigned __int128 tr_bank_term_bound(const tr_scope_view *v, const tr_bank_term *t) {
    const unsigned __int128 limit = ((unsigned __int128)1 << 127) - 1;
    unsigned __int128 b = 1;
    for (int f = 0; f < t->nf; f++) {
        const tr_scope_view_column *c = &v->col[t->col[f]];
        uint64_t m = c->nvalid ? (tr_abs_u64(c->min) > tr_abs_u64(c->max) ? tr_abs_u64(c->min) : tr_abs_u64(c->max)) : 0;
        if (m < 1) m = 1;
        for (int e = 0; e < t->exp[f]; e++) b = tr_bank_satmul(b, m, limit);
    }
    return b;
}
/* 0 when every product and partial sum of every term provably fits 128 bits, else TR_MATH_OVERFLOW (-3) */
static inline int32_t tr_bank_prove(const tr_scope_view *v, const tr_bank_term *terms, int32_t nterms, int32_t mode,
                                    uint32_t joint) {
    const unsigned __int128 limit = ((unsigned __int128)1 << 127) - 1;
    for (int32_t j = 0; j < nterms; j++) {
        uint64_t rows = tr_bank_term_rows(v, &terms[j], mode, joint);
        if (rows && tr_bank_satmul(tr_bank_term_bound(v, &terms[j]), rows, limit) > limit) return -3;
    }
    return 0;
}
static inline uint64_t tr_bank_state_bytes(uint64_t ngroups, int32_t nterms) {
    return tr_budget_add(tr_budget_mul(ngroups, 8), tr_budget_mul(tr_budget_mul(ngroups, (uint64_t)nterms), 24));
}
static inline int tr_bank_popc32(uint32_t x) { int n = 0; while (x) { x &= x - 1; n++; } return n; }
/* the needs of tr_mgpu_bank_build: bytes = its device copies (the selection, group ids, the basis's columns and their
 * validity, the joint-validity flags in TR_BANK_JOINT, the terms, the states) */
static inline void tr_bank_device_need(const tr_scope_view *v, int32_t nterms, int32_t mode, uint32_t joint,
                                       int32_t degree, tr_math_need *need) {
    for (int i = 0; i < TR_BUDGET_RESOURCES; i++) need->need[i] = 0;
    uint64_t n = v->rows, bytes = tr_budget_add(tr_budget_bitmap_bytes(n), tr_bank_state_bytes(v->ngroups, nterms));
    if (v->groups) bytes = tr_budget_add(bytes, tr_budget_mul(n, 4));
    for (int32_t c = 0; c < v->ncols; c++) {
        if (!((joint >> c) & 1)) continue;
        bytes = tr_budget_add(bytes, tr_budget_mul(n, v->col[c].values ? 8 : 4));
        if (v->col[c].valid) bytes = tr_budget_add(bytes, n);
    }
    if (mode == TR_BANK_JOINT) bytes = tr_budget_add(bytes, n);
    bytes = tr_budget_add(bytes, tr_budget_mul((uint64_t)nterms, sizeof(tr_bank_term)));
    need->need[TR_BUDGET_VARIABLES] = (uint64_t)tr_bank_popc32(joint);
    need->need[TR_BUDGET_DEGREE] = (uint64_t)degree;
    need->need[TR_BUDGET_SUPPORT] = (uint64_t)nterms;
    need->need[TR_BUDGET_STATES] = tr_budget_mul(v->ngroups, (uint64_t)nterms);
    need->need[TR_BUDGET_BYTES] = bytes;
}
/* the needs of tr_mgpu_bank_rollup: the fine states, the map and the coarse states on the device, and the host's
 * overflow-bound scratch (16 bytes per coarse group) */
static inline void tr_bank_rollup_device_need(uint64_t fine_groups, int32_t nterms, uint64_t coarse_groups,
                                              tr_math_need *need) {
    for (int i = 0; i < TR_BUDGET_RESOURCES; i++) need->need[i] = 0;
    need->need[TR_BUDGET_SUPPORT] = (uint64_t)nterms;
    need->need[TR_BUDGET_STATES] = tr_budget_mul(coarse_groups, (uint64_t)nterms);
    need->need[TR_BUDGET_BYTES] = tr_budget_add(tr_budget_add(tr_bank_state_bytes(fine_groups, nterms),
                                                              tr_bank_state_bytes(coarse_groups, nterms)),
                                                tr_budget_add(tr_budget_mul(fine_groups, 4), tr_budget_mul(coarse_groups, 16)));
}
/* 0 when, for every term, the magnitudes of the fine sums gathered into each coarse group add up to at most
 * 2^127 - 1 (then every partial sum of the rollup fits, in any order), else TR_MATH_OVERFLOW (-3). scratch: coarse
 * groups' worth of unsigned 128-bit words. Never refuses a rollup of a proven bank (its sums fit as a whole). */
static inline int32_t tr_bank_rollup_bound(const tr_bank_states *fine, const uint32_t *map, uint64_t coarse_groups,
                                           unsigned __int128 *scratch) {
    const unsigned __int128 limit = ((unsigned __int128)1 << 127) - 1;
    const uint64_t m = (uint64_t)fine->nterms;
    for (uint64_t j = 0; j < m; j++) {
        for (uint64_t q = 0; q < coarse_groups; q++) scratch[q] = 0;
        for (uint64_t g = 0; g < fine->ngroups; g++) {
            __int128 x = tr_i128_get(fine->sum[g * m + j]);
            unsigned __int128 mag = x < 0 ? (unsigned __int128)0 - (unsigned __int128)x : (unsigned __int128)x;
            unsigned __int128 *t = &scratch[map[g]];
            if (mag > limit - *t) return -3;
            *t += mag;
        }
    }
    return 0;
}

/* per row (CPU loops and kernels): term validity in TR_BANK_SQL, joint validity in TR_BANK_JOINT, the term's value
 * (exact when the bank's proof passed and the term is valid on the row) */
TR_NUM_HD int tr_bank_term_valid(const tr_scope_view *v, const tr_bank_term *t, uint64_t row) {
    for (int f = 0; f < t->nf; f++) if (!tr_scope_present(v, t->col[f], row)) return 0;
    return 1;
}
TR_NUM_HD int tr_bank_joint_valid(const tr_scope_view *v, uint32_t joint, uint64_t row) {
    for (int32_t c = 0; c < v->ncols; c++) if (((joint >> c) & 1) && !tr_scope_present(v, c, row)) return 0;
    return 1;
}
TR_NUM_HD __int128 tr_bank_term_value(const tr_scope_view *v, const tr_bank_term *t, uint64_t row) {
    __int128 p = 1;
    for (int f = 0; f < t->nf; f++) {
        __int128 x = tr_scope_value(v, t->col[f], row);
        for (int e = 0; e < t->exp[f]; e++) p *= x;
    }
    return p;
}

/* ------------------------------------------------------------------ building (libalg) */
tr_bank_builder *tr_bank_init(const tr_scope *, const uint8_t *exponents, int32_t nterms, int32_t mode,
                              const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error);
int32_t tr_bank_accumulate(tr_bank_builder *, uint64_t begin, uint64_t end);
int32_t tr_bank_merge(tr_bank_builder *into, const tr_bank_builder *from);
tr_bank *tr_bank_finish(tr_bank_builder *, int32_t *error);
void tr_bank_builder_free(tr_bank_builder *);
tr_bank *tr_bank_build(const tr_scope *, const uint8_t *exponents, int32_t nterms, int32_t mode,
                       const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error);
/* the needs checked by tr_bank_init / tr_bank_build (TR_BANK_NEED_CPU), by tr_mgpu_bank_build (TR_BANK_NEED_DEVICE:
 * its device copies) or by tr_bank_from_states (TR_BANK_NEED_ADOPT: the adopted bank); the CUDA path needs the last two */
int32_t tr_bank_need(const tr_scope *, const uint8_t *exponents, int32_t nterms, int32_t mode, int32_t target,
                     tr_math_need *need);
/* Adopts states computed elsewhere (the readback of tr_mgpu_bank_build or tr_mgpu_bank_rollup). origin: the scope the
 * states summarize (ncols, ngroups and total rows must match; the bank becomes that scope's), or NULL (unscoped). */
tr_bank *tr_bank_from_states(const tr_scope *origin, int32_t ncols, const uint8_t *exponents, int32_t nterms,
                             int32_t mode, const tr_bank_states *, const tr_math_budget *budget,
                             tr_math_refusal *refusal, int32_t *error);
/* The coarse bank along map[ngroups] (each < coarse_groups); origin: the coarse scope this rollup is the bank of (the
 * map must be tr_scope_rollup_map(fine scope, origin)), or NULL (unscoped). Overflow of a sum: TR_MATH_OVERFLOW. */
tr_bank *tr_bank_rollup(const tr_bank *, const uint32_t *map, uint64_t coarse_groups, const tr_scope *origin,
                        const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error);
/* the needs of tr_bank_rollup (and of adopting a rollup: TR_BANK_NEED_CPU) or of tr_mgpu_bank_rollup
 * (TR_BANK_NEED_DEVICE) */
int32_t tr_bank_rollup_need(const tr_bank *, uint64_t coarse_groups, int32_t target, tr_math_need *need);

/* ------------------------------------------------------------------ reading */
tr_bank *tr_bank_clone(const tr_bank *, int32_t *error);
/* 1 when both have the same columns, basis, mode, groups and identical states (the scope digests are not compared) */
int32_t tr_bank_equal(const tr_bank *, const tr_bank *);
int32_t tr_bank_get_info(const tr_bank *, tr_bank_info *);
/* pointers into the bank's own states (read only; valid while the bank lives) */
int32_t tr_bank_get_states(const tr_bank *, tr_bank_states *);
int32_t tr_bank_get_exponents(const tr_bank *, uint8_t *out, uint64_t capacity);
void tr_bank_free(tr_bank *);

/* ------------------------------------------------------------------ the cache */
/* capacity 1..TR_BANK_CACHE_CAPACITY_MAX (else TR_MATH_INVALID); max_bytes 0 = TR_BANK_CACHE_BYTES_DEFAULT; a
 * max_bytes below the fixed entry array is TR_MATH_BUDGET */
tr_bank_cache *tr_bank_cache_create(int32_t capacity, uint64_t max_bytes, int32_t *error);
/* an independently owned clone of the entry with exactly this scope descriptor, basis and mode, or NULL (a miss,
 * *error = TR_MATH_OK) */
tr_bank *tr_bank_cache_get(tr_bank_cache *, const tr_scope *, const uint8_t *exponents, int32_t nterms, int32_t mode,
                           int32_t *error);
/* stores a clone of `bank` (built from `scope`: TR_MATH_DOMAIN_ERROR otherwise) under the scope's descriptor, basis and
 * mode, replacing an equal entry; TR_BANK_PUT_REFRESH also drops the entries of the same table at another version.
 * An entry larger than the budget is refused (TR_MATH_BUDGET, resource bytes, need = fixed + entry bytes). */
int32_t tr_bank_cache_put(tr_bank_cache *, const tr_scope *, const tr_bank *, int32_t how, tr_math_refusal *refusal);
/* rebuilds the bank of the scope on the CPU (never a hit), puts it with TR_BANK_PUT_REFRESH and returns it */
tr_bank *tr_bank_cache_refresh(tr_bank_cache *, const tr_scope *, const uint8_t *exponents, int32_t nterms,
                               int32_t mode, const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error);
int32_t tr_bank_cache_get_stats(const tr_bank_cache *, tr_bank_cache_stats *);
int32_t tr_bank_cache_clear(tr_bank_cache *);   /* drops every entry and resets the counters; banks given out live on */
void tr_bank_cache_free(tr_bank_cache *);

/* ------------------------------------------------------------------ the CUDA library (math_bank_gpu.cu, libmathgpu)
 * view: tr_scope_get_view of the scope (host pointers; copied to the device, budgeted with tr_bank_device_need);
 * out: host arrays for the states with out->ngroups = view->ngroups and out->nterms = nterms, overwritten with exactly
 * the states tr_bank_build computes (then adopt them with tr_bank_from_states). The rollup reads `fine` (host arrays,
 * e.g. tr_bank_get_states) and writes the coarse states into out (out->ngroups = coarse_groups). budget: NULL =
 * defaults. Both return 0 or a native error code (TR_MATH_DEVICE when the device fails). */
int tr_mgpu_bank_build(const tr_scope_view *view, const uint8_t *exponents, int32_t nterms, int32_t mode,
                       const tr_math_budget *budget, tr_bank_states *out, tr_math_refusal *refusal);
int tr_mgpu_bank_rollup(const tr_bank_states *fine, const uint32_t *map, uint64_t coarse_groups,
                        const tr_math_budget *budget, tr_bank_states *out, tr_math_refusal *refusal);
#ifdef __cplusplus
}
#endif
#endif
