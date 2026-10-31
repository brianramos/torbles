/* DELTA-001: maintained FND-004 bank states (math_bank.h) under signed linear updates: insert, delete and
 * weight-change deltas merged exactly into a new immutable state whose identity records the update.
 * SPDX-License-Identifier: Apache-2.0
 *
 * States. A maintained state holds, per group g < ngroups, the FND-004 cells of a weighted bag of rows: rows[g] = the
 * sum of the multiplicities of its rows (the bag multiplicity), and per term j of the basis count[g][j] = the sum of
 * the multiplicities of the rows where term j is valid (TR_BANK_SQL or TR_BANK_JOINT validity) and sum[g][j] = the
 * exact sum of multiplicity * term value (signed 128 bits). Rows and counts are int64 (stored as their two's
 * complement in uint64 words, so the layout is tr_bank_states'), every value exact. The state promises
 *   TR_DELTA_NONNEGATIVE  an empirical measure: every multiplicity is >= 0. An update is refused (TR_MATH_DOMAIN_ERROR,
 *                         tr_delta_violation) when a touched group would contradict it: rows < 0 (deleting rows that
 *                         are not there), count < 0, count > rows (deleting NULL rows that are not there), count = 0
 *                         with sum != 0, the constant term's sum != count (in TR_BANK_SQL also count != rows), or the
 *                         sum of a term with only even exponents < 0. These are exact necessary conditions on the
 *                         aggregates; a delete of a row that is not there but that has the group, validity and values
 *                         of present rows is not visible in linear states. The first violation in (group, slot) order
 *                         is reported (slot -1: the bag multiplicity, then the terms): group, term, reason, the old
 *                         value (have), the delta's net change and the result (need = -change for a delete).
 *   TR_DELTA_SIGNED       a signed measure: no sign condition (counts may be negative).
 * A state is built from a FND-004 bank of a scope (tr_delta_state_from_bank: the bank of that scope, its source); its
 * schema is how ids and values are encoded: for a table scope the table (address and lifetime token), every column
 * spec (stored column, validity, high part, shift, codec offset, decimal scale) and the grouping descriptor
 * (tr_scope_key_write part TR_SCOPE_KEY_GROUPS: key specs, radices, NULL codes, lookup contents); for an array scope
 * the column scales and the groups.
 *
 * Deltas (immutable batches of k rows, each with its logical values, validity, group id and an int64 weight: +1 an
 * insert, -1 a delete, any other value a weight change; 0 changes nothing):
 *   ARRAYS  tr_delta_from_arrays: values, validity, group ids < ngroups and weights given; compatible with a state of
 *           the same column count, scales and groups (the ids are the caller's assertion).
 *   SCOPE   tr_delta_from_scope: the selected rows of a FND-004 scope with a sign or per-row weights; compatible with a
 *           state of exactly the same schema (same table, column specs and grouping: the ids mean the same groups).
 *   DIFF    tr_delta_between(old, new): two scopes of the same schema and row count (e.g. one table before and after
 *           a mutation, t.flip, or two selections of it): for every physical row whose contribution differs (selected
 *           in one only, or another group, validity or value), -1 for its old version (when selected in old) and +1 for
 *           its new one (when selected in new). A group move is a delete and an insert. It applies only to the state of
 *           exactly the old scope (its exact descriptor, TR_SCOPE_KEY_ALL), and the result is the state of the new one.
 * Identity. A delta has a 128-bit identity (a digest of its exact content; DIFF: and of both descriptors). Applying
 * one gives a new state: identity = digest(old identity, delta identity, step), base = the first state's identity,
 * step + 1, the table's revision of the new scope (DIFF). Its source (the scope it is the bank of) is the new scope
 * for a DIFF, none otherwise: tr_delta_state_to_bank gives a scoped FND-004 bank only of that exact scope (the bank's
 * overflow proof is then redone against that scope's extremes), so a cache keyed by descriptors never reuses the old
 * entries, and a refresh put drops them.
 *
 * Exactness. Before anything is computed, every touched cell is proven to stay inside its carrier, whatever the order
 * of the additions: |rows| + W_g <= 2^63 - 1, |count| + W_g <= 2^63 - 1 and |sum[g][j]| + W_g * B_j <= 2^127 - 1,
 * W_g = the sum of |weight| over the delta's rows of group g and B_j = prod_c max(|min_c|, |max_c|, 1)^e_jc over the
 * delta's present values (terms no delta row can be valid on are unchanged); otherwise TR_MATH_OVERFLOW (the update
 * is refused, never wrapped; a complete rebuild may still succeed: the proof is conservative). The CPU accumulation
 * and the CUDA kernels (signed 64-bit and two-word 128-bit atomic adds on the touched cells, then the exact sign check
 * on the device and again on the host before the new state exists) give the same states bit for bit.
 *
 * Budgets (math_budgets.h), checked before allocation: a delta: output = its rows, states = its rows (the touched
 * groups), bytes = its arrays, sorting scratch and descriptors; an apply: support = terms, states = groups x terms of
 * the new state, bytes = the new state (CUDA: the compact cells and the device copies, tr_delta_device_need).
 *
 * Errors: TR_MATH_INVALID, _MEMORY, _OVERFLOW, _BUDGET (with the tr_math_refusal), _UNSUPPORTED, TR_MATH_DOMAIN_ERROR
 * (an incompatible delta or state; a nonnegative measure contradicted: tr_delta_violation). Objects are immutable and
 * individually owned; constructors return NULL and set *error, inputs unchanged. */
#ifndef TORBLES_MATH_DELTA_H
#define TORBLES_MATH_DELTA_H
#include <stdint.h>
#include <string.h>
#include "math_num.h"
#include "math_budgets.h"
#include "math_scope.h"
#include "math_bank.h"
#ifdef __cplusplus
extern "C" {
#endif

enum { TR_DELTA_NONNEGATIVE = 1, TR_DELTA_SIGNED = 2 };                  /* the measure a state promises */
enum { TR_DELTA_ARRAYS = 1, TR_DELTA_SCOPE = 2, TR_DELTA_DIFF = 3 };     /* delta sources */
enum { TR_DELTA_OK = 0, TR_DELTA_NEGATIVE_ROWS = 1, TR_DELTA_NEGATIVE_COUNT = 2, TR_DELTA_COUNT_ABOVE_ROWS = 3,
       TR_DELTA_SUM_WITHOUT_ROWS = 4, TR_DELTA_CONSTANT = 5, TR_DELTA_NEGATIVE_EVEN = 6 };   /* violation reasons */
#define TR_DELTA_NONE UINT64_MAX
#define TR_DELTA_I64_LIMIT UINT64_C(9223372036854775807)

/* ------------------------------------------------------------------ structs crossing the ctypes boundary */
typedef struct {
    uint64_t group;                 /* TR_DELTA_NONE when nothing was violated */
    int32_t term;                   /* -1: the bag multiplicity, else the term index */
    int32_t reason;                 /* TR_DELTA_* reason */
    tr_math_i128 have, change, result;   /* the old value, the delta's net change, the value it would become */
} tr_delta_violation;               /* 64 bytes */

typedef struct {
    uint64_t ngroups, present;      /* groups; groups with rows != 0 */
    uint64_t step, bytes;           /* updates applied since the base; bytes the state owns */
    uint64_t identity[2], base[2];  /* this state's identity; the identity of its first state */
    uint64_t source[2];             /* the descriptor digest of the scope it is the bank of (0, 0 when none) */
    uint64_t revision, table;       /* the table revision and address of its source (0 when none) */
    int64_t rows;                   /* the sum of rows[g] */
    int32_t ncols, nterms, mode, promise;
    int32_t bound;                  /* 1: the state of exactly its source scope */
    int32_t degree;
} tr_delta_info;                    /* 128 bytes */

typedef struct {
    uint64_t rows, ngroups, touched;   /* delta rows; the groups of the state it targets; distinct groups it touches */
    uint64_t inserted, deleted;        /* rows with weight > 0 / < 0 */
    uint64_t wabs;                     /* sum of |weight| (saturating) */
    int64_t wnet;                      /* sum of weights (saturating at the int64 limits) */
    uint64_t identity[2];
    uint64_t new_digest[2];            /* DIFF: the descriptor digest of the new scope */
    uint64_t revision;                 /* DIFF: the new scope's table revision */
    int32_t kind, ncols;
} tr_delta_batch_info;                 /* 104 bytes */

/* the arrays of a delta its CUDA application reads (pointers into the delta, valid while it lives) */
typedef struct {
    uint64_t rows, touched;
    const int64_t *weights;         /* [rows] */
    const uint32_t *local;          /* [rows] index of the row's group among the touched groups */
    const uint32_t *groups;         /* [touched] the touched group ids, increasing */
    const uint64_t *wabs;           /* [touched] sum of |weight| per touched group (saturating) */
} tr_delta_rows;                    /* 48 bytes */

typedef struct tr_delta_s tr_delta;
typedef struct tr_delta_state_s tr_delta_state;

/* ------------------------------------------------------------------ shared helpers (CPU code, CUDA host and kernels) */
TR_NUM_HD uint64_t tr_delta_mag64(int64_t x) { return x < 0 ? (uint64_t)0 - (uint64_t)x : (uint64_t)x; }
/* 1 when every exponent of the term is even (its values are >= 0) */
TR_NUM_HD int tr_delta_even(const tr_bank_term *t) {
    for (int f = 0; f < t->nf; f++) if (t->exp[f] & 1) return 0;
    return 1;
}
/* the reason a cell of a nonnegative measure is contradicted (TR_DELTA_OK when it is not): slot 0 is the bag
 * multiplicity (rows), slot j + 1 term j (count, sum) */
TR_NUM_HD int32_t tr_delta_reason(const tr_bank_term *t, int32_t mode, int64_t rows, int64_t count, __int128 sum, int slot) {
    if (slot == 0) return rows < 0 ? TR_DELTA_NEGATIVE_ROWS : TR_DELTA_OK;
    if (count < 0) return TR_DELTA_NEGATIVE_COUNT;
    if (count > rows) return TR_DELTA_COUNT_ABOVE_ROWS;
    if (!count && sum) return TR_DELTA_SUM_WITHOUT_ROWS;
    if (!t->nf && (sum != (__int128)count || (mode == TR_BANK_SQL && count != rows))) return TR_DELTA_CONSTANT;
    if (t->nf && tr_delta_even(t) && sum < 0) return TR_DELTA_NEGATIVE_EVEN;
    return TR_DELTA_OK;
}
/* the first contradicted cell of compact cells (touched groups x (1 + nterms) slots, in that order): its key
 * u * (nterms + 1) + slot, or TR_DELTA_NONE */
static inline uint64_t tr_delta_first_violation(const tr_bank_states *cells, const tr_bank_term *terms, int32_t mode) {
    const uint64_t m = (uint64_t)cells->nterms;
    for (uint64_t u = 0; u < cells->ngroups; u++) {
        int64_t rows = (int64_t)cells->rows[u];
        for (uint64_t s = 0; s <= m; s++) {
            int64_t count = s ? (int64_t)cells->count[u * m + s - 1] : 0;
            __int128 sum = s ? tr_i128_get(cells->sum[u * m + s - 1]) : 0;
            if (tr_delta_reason(s ? &terms[s - 1] : terms, mode, rows, count, sum, (int)s)) return u * (m + 1) + s;
        }
    }
    return TR_DELTA_NONE;
}
/* The a-priori proof (see the header comment) for the compact old cells of the touched groups: 0 or
 * TR_MATH_OVERFLOW (-3). dv: the view over the delta's rows (extremes per column), wabs: sum of |weight| per touched
 * group. Rows and counts are read as int64. */
static inline int32_t tr_delta_prove(const tr_scope_view *dv, const tr_bank_term *terms, int32_t nterms, int32_t mode,
                                     uint32_t joint, const tr_bank_states *cells, const uint64_t *wabs) {
    const unsigned __int128 limit = ((unsigned __int128)1 << 127) - 1;
    unsigned __int128 bound[TR_BANK_MAX_TERMS];
    uint8_t live[TR_BANK_MAX_TERMS];
    for (int32_t j = 0; j < nterms; j++) {
        live[j] = tr_bank_term_rows(dv, &terms[j], mode, joint) > 0;
        bound[j] = live[j] ? tr_bank_term_bound(dv, &terms[j]) : 0;
    }
    const uint64_t m = (uint64_t)nterms;
    for (uint64_t u = 0; u < cells->ngroups; u++) {
        uint64_t w = wabs[u];
        if (w > TR_DELTA_I64_LIMIT || tr_delta_mag64((int64_t)cells->rows[u]) > TR_DELTA_I64_LIMIT - w) return -3;
        for (uint64_t j = 0; j < m; j++) {
            if (!live[j]) continue;
            if (tr_delta_mag64((int64_t)cells->count[u * m + j]) > TR_DELTA_I64_LIMIT - w) return -3;
            __int128 s = tr_i128_get(cells->sum[u * m + j]);
            unsigned __int128 mag = s < 0 ? (unsigned __int128)0 - (unsigned __int128)s : (unsigned __int128)s;
            unsigned __int128 add = tr_bank_satmul(bound[j], w, limit);
            if (mag > limit || add > limit - mag) return -3;
        }
    }
    return 0;
}
/* the needs of tr_mgpu_delta_apply: the delta's columns of the basis and their validity, its weights and local ids,
 * the terms, the compact cells (in and out) and the violation word on the device */
static inline void tr_delta_device_need(const tr_scope_view *dv, int32_t nterms, uint32_t joint, int32_t degree,
                                        uint64_t touched, tr_math_need *need) {
    for (int i = 0; i < TR_BUDGET_RESOURCES; i++) need->need[i] = 0;
    uint64_t k = dv->rows, bytes = tr_budget_add(tr_budget_mul(k, 12), tr_budget_mul((uint64_t)nterms, sizeof(tr_bank_term)));
    for (int32_t c = 0; c < dv->ncols; c++) {
        if (!((joint >> c) & 1)) continue;
        bytes = tr_budget_add(bytes, tr_budget_mul(k, 8));
        if (dv->col[c].valid) bytes = tr_budget_add(bytes, k);
    }
    bytes = tr_budget_add(bytes, tr_budget_add(tr_budget_mul(tr_bank_state_bytes(touched, nterms), 2), 8));
    need->need[TR_BUDGET_VARIABLES] = (uint64_t)tr_bank_popc32(joint);
    need->need[TR_BUDGET_DEGREE] = (uint64_t)degree;
    need->need[TR_BUDGET_SUPPORT] = (uint64_t)nterms;
    need->need[TR_BUDGET_STATES] = tr_budget_mul(touched, (uint64_t)nterms);
    need->need[TR_BUDGET_BYTES] = bytes;
}

/* ------------------------------------------------------------------ deltas (libalg) */
/* n rows: columns[ncols] (logical values and validity), groups[n] < ngroups (NULL when ngroups == 1), weights[n] */
tr_delta *tr_delta_from_arrays(uint64_t n, const tr_scope_array *columns, int32_t ncols, const uint32_t *groups,
                               uint64_t ngroups, const int64_t *weights, const tr_math_budget *budget,
                               tr_math_refusal *refusal, int32_t *error);
/* the selected rows of a scope: weights[physical row] (NULL: every row has weight `sign`, +1 or -1) */
tr_delta *tr_delta_from_scope(const tr_scope *scope, const int64_t *weights, int32_t sign, const tr_math_budget *budget,
                              tr_math_refusal *refusal, int32_t *error);
tr_delta *tr_delta_between(const tr_scope *old_scope, const tr_scope *new_scope, const tr_math_budget *budget,
                           tr_math_refusal *refusal, int32_t *error);
/* the needs the three constructors check (kind TR_DELTA_ARRAYS: n, ncols, columns' validity; SCOPE: scope; DIFF:
 * both scopes; the rows of a DIFF are counted exactly by a pre-pass over the two scopes) */
int32_t tr_delta_need_arrays(uint64_t n, const tr_scope_array *columns, int32_t ncols, tr_math_need *need);
int32_t tr_delta_need_scope(const tr_scope *scope, tr_math_need *need);
int32_t tr_delta_need_between(const tr_scope *old_scope, const tr_scope *new_scope, tr_math_need *need);
int32_t tr_delta_get_info(const tr_delta *, tr_delta_batch_info *);
/* the view over the delta's rows (rows = selected = k, groups = the local ids, ngroups = the touched groups) */
int32_t tr_delta_get_view(const tr_delta *, tr_scope_view *);
int32_t tr_delta_get_rows(const tr_delta *, tr_delta_rows *);
/* the delta's group id of each row (out[k]) */
int32_t tr_delta_get_groups(const tr_delta *, uint32_t *out, uint64_t capacity);
void tr_delta_free(tr_delta *);

/* ------------------------------------------------------------------ maintained states (libalg) */
/* the state of a FND-004 bank of `scope` (the bank must be that scope's: scoped, same digest, groups, rows) */
tr_delta_state *tr_delta_state_from_bank(const tr_bank *bank, const tr_scope *scope, int32_t promise,
                                         const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error);
/* the new state of `state` updated by `delta` on the CPU; violation (may be NULL) is filled on a refusal of the
 * promise (TR_MATH_DOMAIN_ERROR), else its group is TR_DELTA_NONE */
tr_delta_state *tr_delta_state_apply(const tr_delta_state *, const tr_delta *, const tr_math_budget *budget,
                                     tr_math_refusal *refusal, tr_delta_violation *violation, int32_t *error);
/* the CUDA path: the compatibility checks and the proof of tr_delta_state_apply, then the old cells of the touched
 * groups into out (out->ngroups = touched, out->nterms = nterms; host arrays) */
int32_t tr_delta_state_gather(const tr_delta_state *, const tr_delta *, tr_bank_states *out,
                              const tr_math_budget *budget, tr_math_refusal *refusal);
/* adopts the cells tr_mgpu_delta_apply computed for the touched groups (and its first violation key): rechecks the
 * promise exactly on the host (the keys must agree, else TR_MATH_INVALID) and builds the new state as
 * tr_delta_state_apply would */
tr_delta_state *tr_delta_state_from_cells(const tr_delta_state *, const tr_delta *, const tr_bank_states *cells,
                                          uint64_t violation_key, const tr_math_budget *budget,
                                          tr_math_refusal *refusal, tr_delta_violation *violation, int32_t *error);
/* the needs of an apply on the CPU (target 0) or on the device (target 1: tr_mgpu_delta_apply's copies) */
int32_t tr_delta_state_need(const tr_delta_state *, const tr_delta *, int32_t target, tr_math_need *need);
/* a FND-004 bank of the state's cells (every count >= 0 and consistent, else TR_MATH_DOMAIN_ERROR): scoped to `scope`
 * when the state is bound to exactly that scope (else TR_MATH_DOMAIN_ERROR), unscoped when scope is NULL */
tr_bank *tr_delta_state_to_bank(const tr_delta_state *, const tr_scope *scope, const tr_math_budget *budget,
                                tr_math_refusal *refusal, int32_t *error);
int32_t tr_delta_state_get_info(const tr_delta_state *, tr_delta_info *);
/* pointers into the state's cells (rows and counts are int64 in their uint64 words; read only) */
int32_t tr_delta_state_get_states(const tr_delta_state *, tr_bank_states *);
int32_t tr_delta_state_get_exponents(const tr_delta_state *, uint8_t *out, uint64_t capacity);
/* 1 when the cells equal a bank's bit for bit (same basis, mode, columns and groups), else 0 */
int32_t tr_delta_state_equal_bank(const tr_delta_state *, const tr_bank *);
int32_t tr_delta_state_equal(const tr_delta_state *, const tr_delta_state *);   /* the same cells and basis: 1 / 0 */
/* 1 when the state is bound to exactly this scope (its source), else 0 */
int32_t tr_delta_state_bound_to(const tr_delta_state *, const tr_scope *);
void tr_delta_state_free(tr_delta_state *);

/* ------------------------------------------------------------------ the CUDA library (math_delta_gpu.cu)
 * dv / dr: tr_delta_get_view / tr_delta_get_rows of the delta (host pointers, copied to the device); exps: the basis
 * (nterms x dv->ncols); in: the old cells of the touched groups (tr_delta_state_gather); out: host arrays of the same
 * shape, overwritten with the new cells; *violation: the first contradicted cell's key when promise is
 * TR_DELTA_NONNEGATIVE (TR_DELTA_NONE: none), computed on the device. The proof is checked before the launch. */
int tr_mgpu_delta_apply(const tr_scope_view *dv, const tr_delta_rows *dr, const uint8_t *exps, int32_t nterms,
                        int32_t mode, int32_t promise, const tr_bank_states *in, const tr_math_budget *budget,
                        tr_bank_states *out, uint64_t *violation, tr_math_refusal *refusal);
#ifdef __cplusplus
}
#endif
#endif
