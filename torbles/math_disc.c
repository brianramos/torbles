/* DISC-001: exhaustive verification of bounded affine, sum and product relations; see math_disc.h.
 * SPDX-License-Identifier: Apache-2.0 */
#include "math_disc.h"
#include "mathematics.h"
#include "relationships.h"
#include <stdlib.h>
#include <string.h>

extern void at_decode_column(const atable_t *, int, uint32_t *);

static int descriptor(const tr_catalog *cat, int col) {
    for (int i = 0; i < cat->ncols; ++i)
        if (cat->columns[i].col == col) return i;
    return -1;
}

/* decode one descriptor's codes and validity into the catalog's buffers, exactly as relationships.c does */
static int decoded(tr_catalog *cat, int i) {
    size_t n = cat->rows ? (size_t)cat->rows : 1;
    if (!cat->codes[i]) {
        cat->codes[i] = malloc(n * sizeof *cat->codes[i]);
        if (!cat->codes[i]) return TR_MATH_MEMORY;
        at_decode_column(cat->table, cat->columns[i].col, cat->codes[i]);
    }
    if (cat->columns[i].valid >= 0 && !cat->validity[i]) {
        cat->validity[i] = malloc(n * sizeof *cat->validity[i]);
        if (!cat->validity[i]) return TR_MATH_MEMORY;
        at_decode_column(cat->table, cat->columns[i].valid, cat->validity[i]);
    }
    return TR_MATH_OK;
}

int32_t tr_disc_view_get(tr_catalog *cat, const atable_t *table, const uint64_t *used,
                         int32_t module, int32_t out, int32_t left, int32_t right, tr_disc_view *v) {
    if (!cat || !table || !v) return TR_MATH_INVALID;
    memset(v, 0, sizeof *v);
    if (!tr_scope_valid(cat, table, used)) return TR_MATH_STALE;
    int arity = module == TR_DISC_AFFINE ? 1 : (module == TR_DISC_PRODUCT || module == TR_DISC_SUM) ? 2 : 0;
    if (!arity) return TR_MATH_INVALID;
    int oi = descriptor(cat, out), li = descriptor(cat, left), ri = arity == 2 ? descriptor(cat, right) : -1;
    if (oi < 0 || li < 0 || out == left || (arity == 2 && (ri < 0 || out == right)) || (arity == 1 && right != -1))
        return TR_MATH_INVALID;
    int idx[3] = {oi, li, ri};
    for (int k = 0; k <= arity; k++) {
        int rc = decoded(cat, idx[k]);
        if (rc) return rc;
        v->codes[k] = cat->codes[idx[k]];
        v->validity[k] = cat->validity[idx[k]];
        v->offset[k] = cat->columns[idx[k]].offset;
    }
    v->rows = cat->rows; v->used_count = cat->used_count; v->used = cat->used;
    v->module = module; v->arity = arity;
    return TR_MATH_OK;
}

int32_t tr_disc_verify(const tr_disc_view *v, int64_t scale, int64_t bias, tr_disc_evidence *e) {
    if (!v || !e || (v->rows && (!v->used || !v->codes[0] || !v->codes[1] || (v->arity == 2 && !v->codes[2]))) ||
        (v->arity != 1 && v->arity != 2)) return TR_MATH_INVALID;
    tr_disc_init(e);
    for (uint64_t row = 0; row < v->rows; row++) {
        int64_t x = 0, y = 0;
        int s = tr_disc_row(v, row, scale, bias, &x, &y);
        if (s == 0) continue;
        if (s == 1) { e->excluded++; continue; }
        if (s < 0) { e->overflow = 1; continue; }
        e->checked++;
        if (x < e->min_left) e->min_left = x;
        if (x > e->max_left) e->max_left = x;
        if (y < e->min_right) e->min_right = y;
        if (y > e->max_right) e->max_right = y;
        if (s == 3) { e->violations++; if (row < e->witness) e->witness = row; }
    }
    e->usable = tr_disc_decide(v, scale, e);
    return e->overflow ? TR_MATH_OVERFLOW : TR_MATH_OK;
}

int32_t tr_disc_propose_affine(const tr_disc_view *v, int64_t *scale, int64_t *bias) {
    if (!v || !scale || !bias || v->module != TR_DISC_AFFINE || (v->rows && (!v->used || !v->codes[0] || !v->codes[1])))
        return 0;
    int seen = 0;
    int64_t first_x = 0, first_y = 0;
    for (uint64_t row = 0; row < v->rows; row++) {
        if (!((v->used[row >> 6] >> (row & 63)) & 1)) continue;
        if ((v->validity[0] && !v->validity[0][row]) || (v->validity[1] && !v->validity[1][row])) continue;
        __int128 yy = (__int128)v->codes[0][row] + v->offset[0], xx = (__int128)v->codes[1][row] + v->offset[1];
        if (yy < INT64_MIN || yy > INT64_MAX || xx < INT64_MIN || xx > INT64_MAX) return 0;
        int64_t x = (int64_t)xx, y = (int64_t)yy;
        if (!seen) { first_x = x; first_y = y; seen = 1; continue; }
        if (x == first_x) continue;
        __int128 dx = (__int128)x - first_x, dy = (__int128)y - first_y;
        if (dy % dx) return 0;
        __int128 a = dy / dx, b;
        if (!a || a < INT64_MIN || a > INT64_MAX || __builtin_mul_overflow(a, (__int128)first_x, &b) ||
            __builtin_sub_overflow((__int128)first_y, b, &b) || b < INT64_MIN || b > INT64_MAX) return 0;
        *scale = (int64_t)a; *bias = (int64_t)b;
        return 1;
    }
    return 0;
}

uint64_t tr_disc_device_bytes(const tr_disc_view *v) { return v ? tr_disc_bytes(v) : TR_BUDGET_UNKNOWN; }
