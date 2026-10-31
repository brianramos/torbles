/* DISC-001: exhaustive verification of the bounded affine, sum and product relations of relationships.c, as
 * order-independent evidence that the CPU and a CUDA kernel compute identically.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Discovery itself (proposals, the catalog, rewrites) is relationships.c, and tr_probe there is the reference
 * verification. This module exposes one hypothesis's decoded columns (a tr_disc_view; the buffers are owned by the
 * catalog) and verifies the hypothesis over every used row with tr_probe's semantics:
 *   - a used row is excluded when one of its columns is NULL, else checked;
 *   - a checked row violates when the predicted logical value (scale*x + bias, x*y or x + y, exact in 128 bits)
 *     differs from the stored logical value; the witness is the smallest violating stored row (UINT64_MAX if none);
 *   - a logical value is code + offset and must fit int64, else the verification is refused (TR_MATH_OVERFLOW).
 * The evidence also carries the minimum and maximum of each input over the checked rows, so "varied" (min != max)
 * does not depend on the order the rows are visited in.
 *
 * A proof is usable as an exact rewrite only when it covers every physical row (used == rows), checks at least two
 * rows, excludes none, has no violation, varies in every input and is non-degenerate (affine scale != 0). Partial,
 * nullable and stale proofs are evidence, never rewrites: a view is refused (TR_MATH_STALE) when the table's
 * revision, row count or used rows differ from the catalog's snapshot. Verification registers nothing in the
 * catalog (tr_probe does that). */
#ifndef TORBLES_MATH_DISC_H
#define TORBLES_MATH_DISC_H
#include <stdint.h>
#include "math_num.h"
#include "math_budgets.h"
#ifdef __cplusplus
extern "C" {
#endif

enum { TR_DISC_AFFINE = 1, TR_DISC_PRODUCT = 2, TR_DISC_SUM = 4 };   /* the module ids of relationships.h */

typedef struct {
    uint64_t rows, used_count;      /* physical rows; rows set in the used bitmap */
    const uint64_t *used;           /* ceil(rows / 64) words */
    int32_t module, arity;          /* TR_DISC_*; 1 (affine) or 2 inputs */
    const uint32_t *codes[3];       /* out, left, right (right NULL for affine); stored codes */
    const uint32_t *validity[3];    /* NULL when the column has no validity column */
    int64_t offset[3];              /* logical value = code + offset */
} tr_disc_view;

typedef struct {
    uint64_t checked, excluded, violations, witness;
    int64_t min_left, max_left, min_right, max_right;   /* over checked rows (right: 0 for affine) */
    int32_t overflow, usable;
} tr_disc_evidence;

/* one row: 0 unused, 1 excluded (a NULL), 2 checked and holds, 3 checked and violates, -1 a logical value overflows */
TR_NUM_HD int tr_disc_row(const tr_disc_view *v, uint64_t row, int64_t scale, int64_t bias, int64_t *x, int64_t *y) {
    if (!((v->used[row >> 6] >> (row & 63)) & 1)) return 0;
    for (int i = 0; i <= v->arity; i++)
        if (v->validity[i] && v->validity[i][row] == 0) return 1;
    __int128 actual = (__int128)v->codes[0][row] + v->offset[0];
    __int128 left = (__int128)v->codes[1][row] + v->offset[1];
    __int128 right = v->arity == 2 ? (__int128)v->codes[2][row] + v->offset[2] : 0;
    if (actual < INT64_MIN || actual > INT64_MAX || left < INT64_MIN || left > INT64_MAX ||
        right < INT64_MIN || right > INT64_MAX) return -1;
    *x = (int64_t)left; *y = (int64_t)right;
    __int128 predicted = v->module == TR_DISC_AFFINE ? (__int128)scale * left + bias
                       : v->module == TR_DISC_PRODUCT ? left * right : left + right;
    return predicted == actual ? 2 : 3;
}

TR_NUM_HD void tr_disc_init(tr_disc_evidence *e) {
    e->checked = e->excluded = e->violations = 0; e->witness = UINT64_MAX;
    e->min_left = e->min_right = INT64_MAX; e->max_left = e->max_right = INT64_MIN;
    e->overflow = 0; e->usable = 0;
}

/* the rewrite decision from complete evidence (see the header comment) */
TR_NUM_HD int32_t tr_disc_decide(const tr_disc_view *v, int64_t scale, const tr_disc_evidence *e) {
    int varied_left = e->checked && e->min_left != e->max_left;
    int varied_right = v->arity == 1 || (e->checked && e->min_right != e->max_right);
    return v->used_count == v->rows && e->checked >= 2 && !e->excluded && !e->violations && !e->overflow &&
           varied_left && varied_right && (v->module != TR_DISC_AFFINE || scale != 0);
}

struct tr_catalog_s;
struct atable_s;
/* The decoded columns of one hypothesis (column indices of the table, as for tr_probe; right = -1 for affine),
 * decoding into the catalog's own buffers on first use. used is the table's current used bitmap (NULL = every
 * row); a mismatch with the catalog's snapshot is TR_MATH_STALE. */
int32_t tr_disc_view_get(struct tr_catalog_s *, const struct atable_s *, const uint64_t *used,
                         int32_t module, int32_t out, int32_t left, int32_t right, tr_disc_view *);
/* the affine proposal of relationships.c (the first two used complete rows with distinct inputs define an integral
 * scale and bias, else none): 1 with *scale, *bias, or 0. The proposal is only a hypothesis until verified. */
int32_t tr_disc_propose_affine(const tr_disc_view *, int64_t *scale, int64_t *bias);
/* exhaustive verification on the CPU; TR_MATH_OVERFLOW when a checked row's logical value overflows int64 */
int32_t tr_disc_verify(const tr_disc_view *, int64_t scale, int64_t bias, tr_disc_evidence *);
/* bytes the CUDA verification copies to the device (inline for the CUDA library; exported for the bindings) */
static inline uint64_t tr_disc_bytes(const tr_disc_view *v) {
    uint64_t col = tr_budget_mul(v->rows, 4), bytes = tr_budget_add(tr_budget_bitmap_bytes(v->rows), 128);
    for (int i = 0; i <= v->arity; i++) {
        bytes = tr_budget_add(bytes, col);
        if (v->validity[i]) bytes = tr_budget_add(bytes, col);
    }
    return bytes;
}
uint64_t tr_disc_device_bytes(const tr_disc_view *);
#ifdef __cplusplus
}
#endif
#endif
