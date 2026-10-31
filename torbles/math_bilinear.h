/* DELTA-002: bilinear (join) and nonlinear update laws over the maintained states of DELTA-001 (math_delta.h) and
 * the FND-004 data path (math_scope.h, math_bank.h).
 * SPDX-License-Identifier: Apache-2.0
 *
 * 1. Bilinear join states. Two sides A and B are DELTA-001 maintained states (FND-004 cells: bag multiplicity rows,
 *    per term count and exact 128-bit sum) whose dense group ids are key-major: side A's group of join key K and
 *    output group g is K * GA + g, side B's group of key K and output group h is K * GB + h (Scope group_by=[key, g]
 *    gives exactly this order). Keys are dense ids < nkeys shared by both sides; null_key (when given) never matches
 *    (SQL inner-join NULL keys). A join state holds, per output cell (g, h) (cell = g * GB + h) and per pair
 *    p = (i, j) of an A term i and a B term j:
 *        rows    = sum_K rowsA[K, g] * rowsB[K, h]          (the join's bag multiplicity: matched row pairs)
 *        count_p = sum_K countA[K, g, i] * countB[K, h, j]  (matched pairs where both terms are valid)
 *        sum_p   = sum_K sumA[K, g, i] * sumB[K, h, j]      (SUM(a_i * b_j) over the inner join)
 *    i.e. the SQL aggregates COUNT(*), COUNT(a.t_i * b.t_j), SUM(a.t_i * b.t_j) of FROM A JOIN B ON A.key = B.key
 *    GROUP BY A.g, B.h, weighted by the rows' multiplicities; for one value per cell it is the matrix product
 *    (A B)[g, h] = sum_K A[g, K] B[K, h]. The values are 320-bit two's complement (tr_bilin_wide): every value is a
 *    sum over fewer than 2^64 keys of products of two cells (|int64 * int64| < 2^126, |int128 * int128| <= 2^254), so
 *    |value| < 2^318 holds whatever the data: the representation is exact by this bound and nothing is refused for
 *    overflow in the join; each side's cells keep DELTA-001's a-priori proof (an update past it is refused there).
 *
 *    Update law (exact finite differences): a side update A -> A' = A + dA (a DELTA-001 delta applied to side A), and
 *    likewise B, change the join by
 *        d(AB) = dA B + A dB + dA dB          per key, per output cell, per pair (all finite cross terms),
 *    computed only for the keys the deltas touch. The three terms are kept (tr_bilin_join_get_terms): T1 = dA B (old
 *    B), T2 = A dB (old A), T3 = dA dB. Each is formed from the four exact products of old and new cells
 *    (T1 = A'B - AB, T2 = AB' - AB, T3 = A'B' - A'B - AB' + AB), so no difference of two 128-bit cells is ever formed
 *    in 128 bits. A key gaining or losing matches on the other side (a changing join multiplicity: a B insert at key
 *    K multiplies every A row of K once more) is exactly the T2 term of the per-key cells; nothing applies a SUM
 *    deletion rule to a joined row. This is a finite difference of exact states, NOT an infinitesimal derivative:
 *    the derivative of the bilinear map, D(AB)[dA, dB] = dA B + A dB (JET-001 computes derivatives), omits the
 *    second-order term dA dB, which is exactly the contribution of new pairs matched on both sides at once (a row
 *    inserted on each side with the same key) and is never dropped here. A join update requires the side states it
 *    summarizes (identity), the delta and the new side state of each changed side (a successor: same base, step + 1,
 *    same shape, and the cells outside the delta's touched groups unchanged; else TR_MATH_DOMAIN_ERROR); the result
 *    is then exactly the join of the new side states.
 *
 * 2. Dense matrices (tr_bilin_matrix_delta): int64 matrices A (G x K), B (K x H) and updates dA, dB (NULL = 0): the
 *    five G x H products AB, dA B, A dB, dA dB and the updated product AB + dA B + A dB + dA dB, each exact in 320 bits.
 *
 * 3. Nonlinear finalizers keep their own typed retained state; none inherits the SUM deletion rule
 *    (tr_bilin_requirement_get describes each one):
 *      SUM, COUNT                 linear cells (FND-004 / DELTA-001): d(SUM) = SUM(d); a delete subtracts.
 *      AVG, RATIO                 the sufficient sums (numerator and denominator cells), finalized exactly as a
 *                                 rational; the ratio itself is never updated.
 *      VAR_POP, VAR_SAMP          the sufficient sums n, S1 = sum x, S2 = sum x^2 (terms t and t^2):
 *                                 (n S2 - S1^2) / n^2 and / (n (n - 1)), exact rationals.
 *      COUNT_DISTINCT, SUM_DISTINCT  per-value multiplicities (a multiset per group: tr_bilin_values, kind MULTISET):
 *                                 a delete decrements the value's multiplicity; the value leaves the distinct set only
 *                                 when its last copy goes; deleting a value that is not there is refused exactly.
 *      MIN, MAX                   the multiset (exact under every update), or the EXTREMUM state (per group the
 *                                 minimum and maximum with their multiplicities and the present count): inserts and
 *                                 deletes of copies are exact while a copy of the extreme survives (or a smaller /
 *                                 larger value is inserted); deleting the last copy of an extreme is refused with
 *                                 reason TR_BILIN_EXTREME_DELETED (the next extreme is unknown): rebuild from the data.
 *      JOIN_SUM                   per-key cells of both sides (the join state above).
 *    A finalizer asked of a retained state that does not carry what it needs is refused (TR_MATH_DOMAIN_ERROR): e.g.
 *    COUNT_DISTINCT, MIN or MAX of linear cells, COUNT_DISTINCT of an EXTREMUM state, VAR of terms that are not t and
 *    t^2. Finalized values are exact FND-009 big rationals (tr_bigq) in logical units (a DECIMAL(p, s) value is
 *    logical / 10^s per factor); SQL NULL (no valid rows, a zero denominator, VAR_SAMP of one row) is is_null = 1.
 *
 *    Value states (tr_bilin_values) are built from a FND-004 scope (one value column, its groups) and updated by
 *    DELTA-001 deltas (tr_delta: the delta's rows, values, validity, groups and weights; compatible when the delta has
 *    the scope's column count, the column's scale and the groups: like DELTA-001's array deltas, the group ids are the
 *    caller's assertion). Per group: rows (bag multiplicity), nulls (multiplicity of NULL values), present (of
 *    present values), and MULTISET: the distinct present values in increasing order with multiplicities > 0;
 *    EXTREMUM: min, max and their multiplicities. Refusals (TR_MATH_DOMAIN_ERROR, tr_bilin_violation, nothing
 *    exposed): a negative multiplicity of a value (MULTISET: exact; EXTREMUM: a delete below the minimum or above the
 *    maximum, or more copies of an extreme than it has), of rows or of NULL values, present above rows, or the last
 *    copy of an extreme deleted (EXTREMUM). The first violation in (group, check) order is reported.
 *
 * CUDA (math_bilinear_gpu.cu): tr_mgpu_bilin_cross computes the three cross terms of a job (a join update, or a build:
 * old cells zero, T3 = AB) with one work item per (key, g, h, value slot) and multiword atomic adds of the 320-bit
 * terms; tr_mgpu_bilin_matrix computes the dense matrix products with one work item per output entry. Both run the
 * shared inline functions of this header, so the CPU and the device give the same words bit for bit.
 *
 * Budgets (math_budgets.h), checked before allocation, refusal TR_MATH_BUDGET with the tr_math_refusal:
 *   join build / update   support = pairs, output = output cells, states = cells x values, iterations = cell
 *                         products (keys x GA x GB x values: every key of a build, the touched keys of an update),
 *                         bytes = the new join and its three terms, the terms scratch, the job (an update: the
 *                         gathered cells and the old values) and, target 1, the device copies
 *                         (tr_bilin_cross_device_bytes)
 *   matrix delta          tensor = G K + K H (+ G K, + K H with the updates), output = 5 G H, iterations = G H K,
 *                         bytes = the outputs (target 1: and the inputs on the device)
 *   value states          build: support = the scope's present values (MULTISET: the most entries it can hold) or
 *                         2 x groups (EXTREMUM); update: support = entries + delta rows (MULTISET); states = groups,
 *                         bytes = the state and the sorting scratch
 * Errors: TR_MATH_INVALID, _MEMORY, _BUDGET, _UNSUPPORTED, _OVERFLOW (a finalizer's big number past its budget),
 * TR_MATH_DOMAIN_ERROR (incompatible states or deltas, a finalizer without its retained state, a violated value
 * state). Objects are immutable and individually owned; constructors return NULL and set *error, inputs unchanged. */
#ifndef TORBLES_MATH_BILINEAR_H
#define TORBLES_MATH_BILINEAR_H
#include <stdint.h>
#include <string.h>
#include "math_num.h"
#include "math_budgets.h"
#include "math_scope.h"
#include "math_bank.h"
#include "math_delta.h"
#include "math_carriers.h"
#ifdef __cplusplus
extern "C" {
#endif

#define TR_BILIN_ABI 1
#define TR_BILIN_WORDS 5                    /* join values: 320-bit two's complement, least significant word first */
#define TR_BILIN_MAX_PAIRS 1024
#define TR_BILIN_NONE UINT64_MAX
enum { TR_BILIN_TERM_AB = 0, TR_BILIN_TERM_DA_B = 1, TR_BILIN_TERM_A_DB = 2, TR_BILIN_TERM_DA_DB = 3,
       TR_BILIN_TERM_UPDATED = 4 };          /* the five products of tr_bilin_matrix_delta (and T1..T3 = 1..3) */

/* finalizers */
enum { TR_BILIN_SUM = 1, TR_BILIN_COUNT = 2, TR_BILIN_AVG = 3, TR_BILIN_RATIO = 4, TR_BILIN_VAR_POP = 5,
       TR_BILIN_VAR_SAMP = 6, TR_BILIN_COUNT_DISTINCT = 7, TR_BILIN_SUM_DISTINCT = 8, TR_BILIN_MIN = 9,
       TR_BILIN_MAX = 10, TR_BILIN_JOIN_SUM = 11, TR_BILIN_FINALIZERS = 11 };
/* retained states */
enum { TR_BILIN_RETAIN_LINEAR = 1, TR_BILIN_RETAIN_SUFFICIENT = 2, TR_BILIN_RETAIN_MULTIPLICITIES = 3,
       TR_BILIN_RETAIN_EXTREMUM = 4, TR_BILIN_RETAIN_PER_KEY = 5 };
/* how a delete is handled */
enum { TR_BILIN_DELETE_SUBTRACT = 1,        /* the SUM rule: subtract the deleted rows' contribution */
       TR_BILIN_DELETE_SUFFICIENT = 2,      /* update the sufficient sums, finalize again */
       TR_BILIN_DELETE_MULTIPLICITY = 3,    /* decrement the value's multiplicity (leaves at 0) */
       TR_BILIN_DELETE_REBUILD = 4,         /* exact while the extreme survives; else refused: rebuild */
       TR_BILIN_DELETE_PER_KEY = 5 };       /* through the per-key cells of both sides (the three cross terms) */
/* value-state kinds and violation reasons */
enum { TR_BILIN_MULTISET = 1, TR_BILIN_EXTREMUM = 2 };
enum { TR_BILIN_OK = 0, TR_BILIN_NEGATIVE_ROWS = 1, TR_BILIN_NEGATIVE_NULLS = 2, TR_BILIN_NEGATIVE_VALUE = 3,
       TR_BILIN_PRESENT_ABOVE_ROWS = 4, TR_BILIN_EXTREME_DELETED = 5 };

/* ------------------------------------------------------------------ structs crossing the ctypes boundary */
typedef struct { uint64_t w[TR_BILIN_WORDS]; } tr_bilin_wide;   /* 40 bytes */

typedef struct {
    uint64_t nkeys;          /* join keys K (>= 1): dense ids shared by both sides */
    uint64_t groups_a;       /* GA (>= 1): side A's state has nkeys * GA groups, id = key * GA + g */
    uint64_t groups_b;       /* GB (>= 1) */
    uint64_t null_key;       /* a key that never matches, or TR_BILIN_NONE */
    int32_t npairs;          /* 1..TR_BILIN_MAX_PAIRS pairs (A term, B term) */
    int32_t reserved;        /* 0 */
} tr_bilin_spec;             /* 40 bytes */

typedef struct {
    uint64_t ngroups;        /* output cells GA * GB */
    uint64_t nkeys, groups_a, groups_b, null_key;
    uint64_t step;           /* updates since the build */
    uint64_t bytes;          /* bytes the join owns */
    uint64_t touched;        /* keys the last update touched (a build: every key) */
    uint64_t identity[2], base[2];
    uint64_t side_a[2], side_b[2];   /* the identities of the side states it is the join of */
    int32_t npairs, values;  /* values per cell: 1 + 2 npairs (rows, then count and sum of each pair) */
    int32_t terms_a, terms_b;
} tr_bilin_info;             /* 144 bytes */

/* the three cross terms' inputs (pointers into a job, valid while it lives), the kernel argument of
 * tr_mgpu_bilin_cross. Compact cells of `keys` keys: side A rows[keys * GA], count / sum[keys * GA * terms_a]
 * (FND-004 layout, cell u * GA + g), likewise side B. [0] old, [1] new; an old NULL is zero (a build), a new NULL is
 * the old (an unchanged side). Key `skip` (compact index) contributes nothing (the null key), TR_BILIN_NONE: none. */
typedef struct {
    uint64_t keys, groups_a, groups_b, ngroups, skip;
    int32_t npairs, values, terms_a, terms_b;
    const int32_t *pairs;               /* npairs x 2 */
    const uint64_t *a_rows[2];
    const uint64_t *a_count[2];
    const tr_math_i128 *a_sum[2];
    const uint64_t *b_rows[2];
    const uint64_t *b_count[2];
    const tr_math_i128 *b_sum[2];
} tr_bilin_cross_view;                  /* 160 bytes */

typedef struct {
    int32_t finalizer, retained;   /* the retained state it requests (TR_BILIN_RETAIN_*) */
    int32_t linear;                /* 1: d(finalizer) = finalizer(d): the SUM deletion rule applies */
    int32_t deletes;               /* how that state handles a delete (TR_BILIN_DELETE_*) */
    int32_t sufficient;            /* the cells it needs: 1 (a term t), 2 (t and t^2), 0 (not cells) */
    int32_t needs_both_sides;      /* 1: per-key cells of both join sides */
    int32_t alternative;           /* a smaller retained state that refuses some deletes (MIN / MAX: EXTREMUM), or 0 */
    int32_t alternative_deletes;   /* its delete handling (TR_BILIN_DELETE_REBUILD), or 0 */
    char name[32];
} tr_bilin_requirement;            /* 64 bytes */

typedef struct {
    uint64_t ngroups, entries, step, bytes;
    uint64_t identity[2], base[2];
    int32_t kind, column, ncols, scale;
} tr_bilin_values_info;            /* 80 bytes */

typedef struct {
    int64_t rows, nulls, present;  /* bag multiplicity, of NULL values, of present values */
    uint64_t distinct;             /* MULTISET: distinct present values; EXTREMUM: 0 (not retained) */
    int64_t min, max;              /* when present > 0 */
    int64_t min_count, max_count;  /* their multiplicities */
    tr_math_i128 sum_distinct;     /* MULTISET: the sum of the distinct values */
} tr_bilin_group_values;           /* 80 bytes */

typedef struct {
    uint64_t group;                /* TR_BILIN_NONE: nothing violated */
    int32_t reason, has_value;     /* TR_BILIN_*; 1 when value is set */
    int64_t value;                 /* the value concerned */
    int64_t have, change;          /* its multiplicity before, the delta's net change */
} tr_bilin_violation;              /* 40 bytes */

typedef struct tr_bilin_join_s tr_bilin_join;
typedef struct tr_bilin_job_s tr_bilin_job;
typedef struct tr_bilin_values_s tr_bilin_values;

/* ------------------------------------------------------------------ shared 320-bit helpers (CPU and kernels) */
TR_NUM_HD void tr_bilin_w_zero(uint64_t *w) { for (int i = 0; i < TR_BILIN_WORDS; i++) w[i] = 0; }
TR_NUM_HD int tr_bilin_w_is_zero(const uint64_t *w) {
    uint64_t o = 0;
    for (int i = 0; i < TR_BILIN_WORDS; i++) o |= w[i];
    return o == 0;
}
/* the exact 256-bit product of two 128-bit magnitudes, r[0..3] */
TR_NUM_HD void tr_bilin_mul_u128(unsigned __int128 x, unsigned __int128 y, uint64_t *r) {
    uint64_t x0 = (uint64_t)x, x1 = (uint64_t)(x >> 64), y0 = (uint64_t)y, y1 = (uint64_t)(y >> 64);
    unsigned __int128 p00 = (unsigned __int128)x0 * y0, p01 = (unsigned __int128)x0 * y1;
    unsigned __int128 p10 = (unsigned __int128)x1 * y0, p11 = (unsigned __int128)x1 * y1;
    unsigned __int128 mid = (p00 >> 64) + (uint64_t)p01 + (uint64_t)p10;                  /* < 3 * 2^64 */
    unsigned __int128 hi = (mid >> 64) + (p01 >> 64) + (p10 >> 64) + (uint64_t)p11;       /* < 4 * 2^64 */
    r[0] = (uint64_t)p00; r[1] = (uint64_t)mid; r[2] = (uint64_t)hi;
    r[3] = (uint64_t)(hi >> 64) + (uint64_t)(p11 >> 64);
}
TR_NUM_HD unsigned __int128 tr_bilin_mag(__int128 a) { return a < 0 ? (unsigned __int128)0 - (unsigned __int128)a : (unsigned __int128)a; }
/* w = -w (mod 2^320) */
TR_NUM_HD void tr_bilin_w_negate(uint64_t *w) {
    uint64_t carry = 1;
    for (int i = 0; i < TR_BILIN_WORDS; i++) { uint64_t x = ~w[i] + carry; carry = carry && x == 0; w[i] = x; }
}
/* the exact signed product a * b as 320-bit two's complement words */
TR_NUM_HD void tr_bilin_w_mul(__int128 a, __int128 b, uint64_t *w) {
    tr_bilin_w_zero(w);
    if (!a || !b) return;
    tr_bilin_mul_u128(tr_bilin_mag(a), tr_bilin_mag(b), w);
    if ((a < 0) != (b < 0)) tr_bilin_w_negate(w);
}
/* w += x, w -= x (mod 2^320) */
TR_NUM_HD void tr_bilin_w_add(uint64_t *w, const uint64_t *x) {
    uint64_t carry = 0;
    for (int i = 0; i < TR_BILIN_WORDS; i++) {
        uint64_t s = w[i] + x[i], c1 = s < x[i], t = s + carry, c2 = t < carry;
        w[i] = t; carry = c1 | c2;
    }
}
TR_NUM_HD void tr_bilin_w_sub(uint64_t *w, const uint64_t *x) {
    uint64_t borrow = 0;
    for (int i = 0; i < TR_BILIN_WORDS; i++) {
        uint64_t d = w[i] - x[i], b1 = w[i] < x[i], t = d - borrow, b2 = d < borrow;
        w[i] = t; borrow = b1 | b2;
    }
}
/* a signed 128-bit value as 320-bit words */
TR_NUM_HD void tr_bilin_w_from_i128(__int128 v, uint64_t *w) {
    uint64_t fill = v < 0 ? ~(uint64_t)0 : 0;
    w[0] = (uint64_t)v; w[1] = (uint64_t)((unsigned __int128)v >> 64);
    for (int i = 2; i < TR_BILIN_WORDS; i++) w[i] = fill;
}

/* ------------------------------------------------------------------ the cross terms of one (key, g, h, slot) item */
/* slot 0: rows; slot 1 + 2p: count of pair p; slot 2 + 2p: sum of pair p. The value of side A's cell for the slot
 * (old: version 0, new: version 1) */
TR_NUM_HD __int128 tr_bilin_cell_a(const tr_bilin_cross_view *v, int version, uint64_t cell, int32_t slot) {
    const int ver = (version == 1 && !v->a_rows[1]) ? 0 : version;           /* an unchanged side: new = old */
    if (!v->a_rows[ver]) return 0;                                         /* a build: old = 0 */
    if (slot == 0) return (__int128)(int64_t)v->a_rows[ver][cell];
    const int32_t p = (slot - 1) / 2, i = v->pairs[2 * p];
    const uint64_t k = cell * (uint64_t)v->terms_a + (uint64_t)i;
    return (slot & 1) ? (__int128)(int64_t)v->a_count[ver][k] : tr_i128_get(v->a_sum[ver][k]);
}
TR_NUM_HD __int128 tr_bilin_cell_b(const tr_bilin_cross_view *v, int version, uint64_t cell, int32_t slot) {
    const int ver = (version == 1 && !v->b_rows[1]) ? 0 : version;
    if (!v->b_rows[ver]) return 0;
    if (slot == 0) return (__int128)(int64_t)v->b_rows[ver][cell];
    const int32_t p = (slot - 1) / 2, j = v->pairs[2 * p + 1];
    const uint64_t k = cell * (uint64_t)v->terms_b + (uint64_t)j;
    return (slot & 1) ? (__int128)(int64_t)v->b_count[ver][k] : tr_i128_get(v->b_sum[ver][k]);
}
/* The contributions of work item i = ((u * GA + g) * GB + h) * values + slot to T1 = A'B - AB, T2 = AB' - AB and
 * T3 = A'B' - A'B - AB' + AB (t[0..2] x 5 words); returns the output index (cell * values + slot), or TR_BILIN_NONE
 * when the item contributes nothing (both cells unchanged, or the skipped key). */
TR_NUM_HD uint64_t tr_bilin_cross_item(const tr_bilin_cross_view *v, uint64_t i, uint64_t *t) {
    const uint64_t V = (uint64_t)v->values, slot = i % V, rest = i / V, h = rest % v->groups_b,
                   rest2 = rest / v->groups_b, g = rest2 % v->groups_a, u = rest2 / v->groups_a;
    if (u == v->skip) return TR_BILIN_NONE;
    const uint64_t ca = u * v->groups_a + g, cb = u * v->groups_b + h;
    __int128 ao = tr_bilin_cell_a(v, 0, ca, (int32_t)slot), an = tr_bilin_cell_a(v, 1, ca, (int32_t)slot);
    __int128 bo = tr_bilin_cell_b(v, 0, cb, (int32_t)slot), bn = tr_bilin_cell_b(v, 1, cb, (int32_t)slot);
    if (ao == an && bo == bn) return TR_BILIN_NONE;
    uint64_t pnn[TR_BILIN_WORDS], pno[TR_BILIN_WORDS], pon[TR_BILIN_WORDS], poo[TR_BILIN_WORDS];
    tr_bilin_w_mul(an, bn, pnn); tr_bilin_w_mul(an, bo, pno); tr_bilin_w_mul(ao, bn, pon); tr_bilin_w_mul(ao, bo, poo);
    uint64_t *t1 = t, *t2 = t + TR_BILIN_WORDS, *t3 = t + 2 * TR_BILIN_WORDS;
    for (int w = 0; w < TR_BILIN_WORDS; w++) { t1[w] = pno[w]; t2[w] = pon[w]; t3[w] = pnn[w]; }
    tr_bilin_w_sub(t1, poo);
    tr_bilin_w_sub(t2, poo);
    tr_bilin_w_sub(t3, pno); tr_bilin_w_sub(t3, pon); tr_bilin_w_add(t3, poo);
    return (g * v->groups_b + h) * V + slot;
}
TR_NUM_HD uint64_t tr_bilin_cross_items(const tr_bilin_cross_view *v) {
    return v->keys * v->groups_a * v->groups_b * (uint64_t)v->values;
}

/* ------------------------------------------------------------------ one entry of the dense matrix products */
/* out[k * G * H + g * H + h] for k = AB, dA B, A dB, dA dB, updated (dA / dB NULL: zero) */
TR_NUM_HD void tr_bilin_matrix_entry(uint64_t G, uint64_t K, uint64_t H, const int64_t *A, const int64_t *B,
                                     const int64_t *dA, const int64_t *dB, uint64_t g, uint64_t h, tr_bilin_wide *out) {
    uint64_t acc[4][TR_BILIN_WORDS], p[TR_BILIN_WORDS];
    for (int k = 0; k < 4; k++) tr_bilin_w_zero(acc[k]);
    for (uint64_t q = 0; q < K; q++) {
        __int128 a = A[g * K + q], b = B[q * H + h], da = dA ? dA[g * K + q] : 0, db = dB ? dB[q * H + h] : 0;
        if (a && b) { tr_bilin_w_from_i128(a * b, p); tr_bilin_w_add(acc[0], p); }      /* |int64 * int64| < 2^126 */
        if (da && b) { tr_bilin_w_from_i128(da * b, p); tr_bilin_w_add(acc[1], p); }
        if (a && db) { tr_bilin_w_from_i128(a * db, p); tr_bilin_w_add(acc[2], p); }
        if (da && db) { tr_bilin_w_from_i128(da * db, p); tr_bilin_w_add(acc[3], p); }
    }
    const uint64_t GH = G * H, e = g * H + h;
    for (int k = 0; k < 4; k++) for (int w = 0; w < TR_BILIN_WORDS; w++) out[k * GH + e].w[w] = acc[k][w];
    for (int k = 1; k < 4; k++) tr_bilin_w_add(acc[0], acc[k]);
    for (int w = 0; w < TR_BILIN_WORDS; w++) out[4 * GH + e].w[w] = acc[0][w];
}

/* the bytes tr_mgpu_bilin_cross copies to / allocates on the device: the pairs, the non-NULL cell arrays (cell
 * bytes 8 + 24 * terms each) and the three terms (saturating) */
static inline uint64_t tr_bilin_cross_device_bytes(const tr_bilin_cross_view *v) {
    uint64_t b = tr_budget_mul((uint64_t)(v->npairs > 0 ? v->npairs : 0), 8);
    uint64_t ca = tr_budget_mul(v->keys, v->groups_a), cb = tr_budget_mul(v->keys, v->groups_b);
    for (int ver = 0; ver < 2; ver++) {
        if (v->a_rows[ver]) b = tr_budget_add(b, tr_bank_state_bytes(ca, v->terms_a));
        if (v->b_rows[ver]) b = tr_budget_add(b, tr_bank_state_bytes(cb, v->terms_b));
    }
    return tr_budget_add(b, tr_budget_mul(tr_budget_mul(v->ngroups, (uint64_t)v->values), 3 * sizeof(tr_bilin_wide)));
}
/* the needs of the dense matrix delta (target 1: the device copies of the inputs and outputs) */
static inline void tr_bilin_matrix_need_inline(uint64_t G, uint64_t K, uint64_t H, int with_da, int with_db,
                                               int32_t target, tr_math_need *need) {
    for (int i = 0; i < TR_BUDGET_RESOURCES; i++) need->need[i] = 0;
    uint64_t gk = tr_budget_mul(G, K), kh = tr_budget_mul(K, H), gh = tr_budget_mul(G, H);
    uint64_t tensor = tr_budget_add(gk, kh);
    if (with_da) tensor = tr_budget_add(tensor, gk);
    if (with_db) tensor = tr_budget_add(tensor, kh);
    need->need[TR_BUDGET_TENSOR] = tensor;
    need->need[TR_BUDGET_OUTPUT] = tr_budget_mul(gh, 5);
    need->need[TR_BUDGET_ITERATIONS] = tr_budget_mul(gh, K);
    uint64_t bytes = tr_budget_mul(tr_budget_mul(gh, 5), sizeof(tr_bilin_wide));
    if (target == 1) bytes = tr_budget_add(bytes, tr_budget_mul(tensor, 8));
    need->need[TR_BUDGET_BYTES] = bytes;
}

/* ------------------------------------------------------------------ join states (libalg) */
/* the needs of a build (target 0 CPU, 1 the device path: the job's view on the device) */
int32_t tr_bilin_join_need(const tr_delta_state *a, const tr_delta_state *b, const tr_bilin_spec *spec,
                           const int32_t *pairs, int32_t target, tr_math_need *need);
tr_bilin_join *tr_bilin_join_build(const tr_delta_state *a, const tr_delta_state *b, const tr_bilin_spec *spec,
                                   const int32_t *pairs, const tr_math_budget *budget, tr_math_refusal *refusal,
                                   int32_t *error);
/* The join of the new side states: a2 = a updated by da (DELTA-001: tr_delta_state_apply or its CUDA path), b2 = b
 * updated by db; (da, a2) both NULL when side A does not change, likewise B (at least one side changes). */
int32_t tr_bilin_update_need(const tr_bilin_join *, const tr_delta_state *a, const tr_delta *da,
                             const tr_delta_state *a2, const tr_delta_state *b, const tr_delta *db,
                             const tr_delta_state *b2, int32_t target, tr_math_need *need);
tr_bilin_join *tr_bilin_join_update(const tr_bilin_join *, const tr_delta_state *a, const tr_delta *da,
                                    const tr_delta_state *a2, const tr_delta_state *b, const tr_delta *db,
                                    const tr_delta_state *b2, const tr_math_budget *budget, tr_math_refusal *refusal,
                                    int32_t *error);
/* The device path in three steps: a job (the checks, the budget of `target` and the gathered cells; a build job
 * borrows the side states' arrays, which must outlive it), its view (tr_mgpu_bilin_cross or tr_bilin_cross_eval
 * computes the terms), the new join from the terms. */
tr_bilin_job *tr_bilin_job_build(const tr_delta_state *a, const tr_delta_state *b, const tr_bilin_spec *spec,
                                 const int32_t *pairs, int32_t target, const tr_math_budget *budget,
                                 tr_math_refusal *refusal, int32_t *error);
tr_bilin_job *tr_bilin_job_update(const tr_bilin_join *, const tr_delta_state *a, const tr_delta *da,
                                  const tr_delta_state *a2, const tr_delta_state *b, const tr_delta *db,
                                  const tr_delta_state *b2, int32_t target, const tr_math_budget *budget,
                                  tr_math_refusal *refusal, int32_t *error);
int32_t tr_bilin_job_get_view(const tr_bilin_job *, tr_bilin_cross_view *);
/* the CPU evaluator: terms (3 x ngroups x values, overwritten) */
int32_t tr_bilin_cross_eval(const tr_bilin_cross_view *, tr_bilin_wide *terms, const tr_math_budget *budget,
                            tr_math_refusal *refusal);
tr_bilin_join *tr_bilin_join_finish(const tr_bilin_job *, const tr_bilin_wide *terms, const tr_math_budget *budget,
                                    tr_math_refusal *refusal, int32_t *error);
void tr_bilin_job_free(tr_bilin_job *);

int32_t tr_bilin_join_get_info(const tr_bilin_join *, tr_bilin_info *);
/* copies of the values (ngroups x values) / of the last update's terms T1, T2, T3 (3 x ngroups x values; a build:
 * 0, 0, AB) / of the pairs (npairs x 2) */
int32_t tr_bilin_join_get_values(const tr_bilin_join *, tr_bilin_wide *out, uint64_t capacity);
int32_t tr_bilin_join_get_terms(const tr_bilin_join *, tr_bilin_wide *out, uint64_t capacity);
int32_t tr_bilin_join_get_pairs(const tr_bilin_join *, int32_t *out, uint64_t capacity);
int32_t tr_bilin_join_equal(const tr_bilin_join *, const tr_bilin_join *);       /* same spec, pairs and values */
/* 1 when a and b are exactly the side states the join is the join of (their identities), else 0 */
int32_t tr_bilin_join_summarizes(const tr_bilin_join *, const tr_delta_state *a, const tr_delta_state *b);
/* SUM, COUNT (of pair p0), AVG (sum / count of p0), RATIO (sum p0 / sum p1) of an output cell; *out an owned
 * tr_bigq or NULL with *is_null = 1; another finalizer: TR_MATH_DOMAIN_ERROR */
int32_t tr_bilin_join_finalize(const tr_bilin_join *, uint64_t cell, int32_t finalizer, int32_t p0, int32_t p1,
                               tr_bigq **out, int32_t *is_null, const tr_math_budget *budget, tr_math_refusal *refusal);
void tr_bilin_join_free(tr_bilin_join *);

/* ------------------------------------------------------------------ dense matrices (libalg) */
int32_t tr_bilin_matrix_need(uint64_t G, uint64_t K, uint64_t H, int32_t with_da, int32_t with_db, int32_t target,
                             tr_math_need *need);
/* A (G x K), B (K x H), dA, dB (NULL = 0), row-major int64; out: 5 x G x H values (AB, dA B, A dB, dA dB, updated) */
int32_t tr_bilin_matrix_delta(uint64_t G, uint64_t K, uint64_t H, const int64_t *A, const int64_t *B,
                              const int64_t *dA, const int64_t *dB, tr_bilin_wide *out, const tr_math_budget *budget,
                              tr_math_refusal *refusal);

/* ------------------------------------------------------------------ finalizers and value states (libalg) */
int32_t tr_bilin_requirement_get(int32_t finalizer, tr_bilin_requirement *out);
/* a finalizer of a linear state's group: SUM, COUNT, AVG (of term t0), RATIO (sum t0 / sum t1), VAR_POP / VAR_SAMP
 * (t0 = t, t1 = t^2); COUNT_DISTINCT, SUM_DISTINCT, MIN, MAX, JOIN_SUM: TR_MATH_DOMAIN_ERROR (not in linear cells) */
int32_t tr_bilin_finalize_state(const tr_delta_state *, uint64_t group, int32_t finalizer, int32_t t0, int32_t t1,
                                tr_bigq **out, int32_t *is_null, const tr_math_budget *budget,
                                tr_math_refusal *refusal);
int32_t tr_bilin_values_need_scope(const tr_scope *, int32_t column, int32_t kind, tr_math_need *need);
tr_bilin_values *tr_bilin_values_from_scope(const tr_scope *, int32_t column, int32_t kind,
                                            const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error);
int32_t tr_bilin_values_need_apply(const tr_bilin_values *, const tr_delta *, tr_math_need *need);
/* the state updated by a DELTA-001 delta; a violation (TR_MATH_DOMAIN_ERROR) is described in *violation */
tr_bilin_values *tr_bilin_values_apply(const tr_bilin_values *, const tr_delta *, const tr_math_budget *budget,
                                       tr_math_refusal *refusal, tr_bilin_violation *violation, int32_t *error);
int32_t tr_bilin_values_get_info(const tr_bilin_values *, tr_bilin_values_info *);
int32_t tr_bilin_values_group(const tr_bilin_values *, uint64_t group, tr_bilin_group_values *out);
/* MULTISET: the group's distinct values (increasing) and multiplicities; *count is set (capacity may be 0 to count) */
int32_t tr_bilin_values_entries(const tr_bilin_values *, uint64_t group, int64_t *values, int64_t *mult,
                                uint64_t capacity, uint64_t *count);
/* COUNT (present values), COUNT_DISTINCT, SUM_DISTINCT, MIN, MAX; another finalizer, or a MULTISET finalizer of an
 * EXTREMUM state: TR_MATH_DOMAIN_ERROR */
int32_t tr_bilin_values_finalize(const tr_bilin_values *, uint64_t group, int32_t finalizer, tr_bigq **out,
                                 int32_t *is_null, const tr_math_budget *budget, tr_math_refusal *refusal);
int32_t tr_bilin_values_equal(const tr_bilin_values *, const tr_bilin_values *);   /* same kind, groups, contents */
void tr_bilin_values_free(tr_bilin_values *);

/* ------------------------------------------------------------------ the CUDA library (math_bilinear_gpu.cu) */
/* terms: host array of 3 x ngroups x values, overwritten with exactly tr_bilin_cross_eval's words */
int tr_mgpu_bilin_cross(const tr_bilin_cross_view *view, tr_bilin_wide *terms, const tr_math_budget *budget,
                        tr_math_refusal *refusal);
int tr_mgpu_bilin_matrix(uint64_t G, uint64_t K, uint64_t H, const int64_t *A, const int64_t *B, const int64_t *dA,
                         const int64_t *dB, tr_bilin_wide *out, const tr_math_budget *budget, tr_math_refusal *refusal);
#ifdef __cplusplus
}
#endif
#endif
