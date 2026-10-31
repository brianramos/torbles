/* LATTICE-001 (occupancy histograms) and LATTICE-002 (subset-lattice transforms); see math_lattice.h.
 * SPDX-License-Identifier: Apache-2.0 */
#include "math_lattice.h"
#include "mathematics.h"
#include "math_domain.h"
#include "math_carriers.h"
#include "math_handles.h"
#include <stdlib.h>
#include <string.h>

TR_MATH_ABI(tr_lattice_axis, TR_LATTICE_ABI);      /* the ctypes boundary of LATTICE-001/002 (math_handles.h) */
TR_MATH_ABI(tr_lattice_hist_info, TR_LATTICE_ABI);
TR_MATH_ABI(tr_lattice_count, TR_LATTICE_ABI);

_Static_assert(sizeof(tr_lattice_axis) == 16, "tr_lattice_axis ABI");
_Static_assert(sizeof(tr_lattice_hist_info) == 64, "tr_lattice_hist_info ABI");
_Static_assert(sizeof(tr_lattice_count) == 104, "tr_lattice_count ABI");

#define HIST_MAGIC UINT32_C(0x4c415448)

struct tr_lattice_hist_s {
    uint32_t magic;
    int32_t k, nulls, kind;
    uint64_t ncells, points, complete, occupied, bytes;
    uint64_t *cells;              /* ncells */
    uint64_t *axis;               /* 3k: TRUE, FALSE, UNKNOWN per axis */
};

static void set_error(int32_t *error, int32_t code) { if (error) *error = code; }
static void *zalloc(uint64_t bytes) {
    if (bytes > (uint64_t)SIZE_MAX) return NULL;
    return calloc(1, bytes ? (size_t)bytes : 1);
}
static void zero_need(tr_math_need *n) { for (int i = 0; i < TR_BUDGET_RESOURCES; i++) n->need[i] = 0; }
static int hist_ok(const tr_lattice_hist *h) { return h && h->magic == HIST_MAGIC; }
static const tr_math_budget *budget_or_default(const tr_math_budget *b, tr_math_budget *dflt) {
    if (b) return b;
    tr_budget_default(dflt);
    return dflt;
}
static int popc(uint64_t x) { return __builtin_popcountll(x); }

/* ------------------------------------------------------------------ histogram objects */
static void hist_release(tr_lattice_hist *h) {
    if (!h) return;
    free(h->cells); free(h->axis);
    h->magic = 0;
    free(h);
}
void tr_lattice_hist_free(tr_lattice_hist *h) { if (hist_ok(h)) hist_release(h); }
static tr_lattice_hist *hist_fail(tr_lattice_hist *h, int32_t *error, int32_t code) {
    hist_release(h); set_error(error, code); return NULL;
}
static tr_lattice_hist *hist_alloc(int32_t k, int32_t nulls, int32_t kind) {
    tr_lattice_hist *h = zalloc(sizeof *h);
    if (!h) return NULL;
    h->k = k; h->nulls = nulls; h->kind = kind;
    h->ncells = tr_lattice_cells(k, nulls);
    h->cells = zalloc(h->ncells * 8);
    h->axis = zalloc((uint64_t)(k > 0 ? k : 1) * 24);
    if (!h->cells || !h->axis) { hist_release(h); return NULL; }
    h->bytes = tr_lattice_object_bytes(h->ncells, k);
    return h;
}
static void hist_finish(tr_lattice_hist *h) {
    uint64_t occupied = 0;
    for (uint64_t c = 0; c < h->ncells; c++) occupied += h->cells[c] != 0;
    h->occupied = occupied;
    h->magic = HIST_MAGIC;
}

/* checks, needs and budget of a build; target CPU or DEVICE */
static int32_t hist_request(const tr_bool_domain *d, const tr_lattice_axis *axes, int32_t k,
                            const tr_bool_program_view *progs, int32_t nprogs, int32_t nulls, int32_t target,
                            tr_math_need *need) {
    int32_t rc = tr_lattice_axes_check(d, axes, k, progs, nprogs, nulls);
    if (rc) return rc;
    tr_lattice_hist_base_need(d, k, progs, nprogs, nulls, need);
    if (target == TR_LATTICE_NEED_DEVICE) need->need[TR_BUDGET_BYTES] = tr_lattice_hist_device_bytes(d, k, progs, nprogs, nulls);
    else if (target != TR_LATTICE_NEED_CPU) return TR_MATH_INVALID;
    return TR_MATH_OK;
}

int32_t tr_lattice_hist_need(const tr_bool_domain *d, const tr_lattice_axis *axes, int32_t k,
                             const tr_bool_program_view *progs, int32_t nprogs, int32_t nulls, int32_t target,
                             tr_math_need *need) {
    if (!need) return TR_MATH_INVALID;
    return hist_request(d, axes, k, progs, nprogs, nulls, target, need);
}

tr_lattice_hist *tr_lattice_hist_build(const tr_bool_domain *d, const tr_lattice_axis *axes, int32_t k,
                                       const tr_bool_program_view *progs, int32_t nprogs, int32_t nulls,
                                       const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    tr_math_need need;
    int32_t rc = hist_request(d, axes, k, progs, nprogs, nulls, TR_LATTICE_NEED_CPU, &need);
    if (rc) return hist_fail(NULL, error, rc);
    tr_math_budget dflt;
    rc = tr_budget_check(budget_or_default(budget, &dflt), &need, refusal);
    if (rc) return hist_fail(NULL, error, rc);
    rc = tr_lattice_hist_limits(k, nprogs);
    if (rc) return hist_fail(NULL, error, rc);
    tr_lattice_hist *h = hist_alloc(k, nulls, d->kind);
    uint64_t *scratch = zalloc(tr_lattice_prog_nodes(progs, nprogs) * 16);
    if (!h || !scratch) { free(scratch); return hist_fail(h, error, TR_MATH_MEMORY); }
    uint64_t value[TR_LATTICE_MAX_AXES + 1], valid[TR_LATTICE_MAX_AXES + 1], complete = 0;
    for (uint64_t w = 0; w < d->nw; w++) {
        uint64_t sel = tr_bool_selection_word(d, w);
        if (!sel) continue;
        tr_lattice_word(d, axes, k, progs, nprogs, w, sel, scratch, value, valid);
        uint64_t known = sel;
        for (int32_t i = 0; i < k; i++) {
            h->axis[3 * i] += (uint64_t)popc(value[i]);
            h->axis[3 * i + 1] += (uint64_t)popc(valid[i] & ~value[i]);
            h->axis[3 * i + 2] += (uint64_t)popc(sel & ~valid[i]);
            known &= valid[i];
        }
        complete += (uint64_t)popc(known);
        uint64_t bits = nulls == TR_LATTICE_APART ? known : sel;
        while (bits) {
            int b = __builtin_ctzll(bits);
            bits &= bits - 1;
            h->cells[tr_lattice_cell_of(value, valid, k, nulls, b)]++;
        }
    }
    free(scratch);
    h->points = d->selected;
    h->complete = complete;
    hist_finish(h);
    set_error(error, TR_MATH_OK);
    return h;
}

/* the TRUE / FALSE / UNKNOWN counts of every axis implied by the cells (APART: the complete points only) */
static int32_t derive_axis(int32_t k, int32_t nulls, uint64_t ncells, const uint64_t *cells, uint64_t *axis,
                           uint64_t *total) {
    for (int32_t i = 0; i < 3 * k; i++) axis[i] = 0;
    uint64_t sum = 0;
    int32_t digits[TR_LATTICE_MAX_AXES + 1] = {0};
    for (uint64_t c = 0; c < ncells; c++) {
        uint64_t x = cells[c];
        if (__builtin_add_overflow(sum, x, &sum)) return TR_MATH_INVALID;   /* no point set has 2^64 points */
        for (int32_t i = 0; i < k; i++) {
            int32_t dgt = nulls == TR_LATTICE_APART ? (int32_t)((c >> i) & 1) : digits[i];
            uint64_t *slot = &axis[3 * i + (dgt == 1 ? 0 : dgt == 0 ? 1 : 2)];
            *slot += x;                                       /* bounded by sum: never wraps */
        }
        if (nulls == TR_LATTICE_THREE)                        /* the base-3 odometer of the next cell */
            for (int32_t i = 0; i < k; i++) { if (++digits[i] < 3) break; digits[i] = 0; }
    }
    *total = sum;
    return TR_MATH_OK;
}

tr_lattice_hist *tr_lattice_hist_from_counts(int32_t k, int32_t nulls, int32_t kind, uint64_t points,
                                             const uint64_t *cells, const uint64_t *axis, const tr_math_budget *budget,
                                             tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    if (k < 0 || (nulls != TR_LATTICE_APART && nulls != TR_LATTICE_THREE) || kind < 0 || kind > TR_BOOL_ASSIGNMENTS ||
        !cells) return hist_fail(NULL, error, TR_MATH_INVALID);
    tr_math_need need;
    zero_need(&need);
    uint64_t ncells = tr_lattice_cells(k, nulls);
    need.need[TR_BUDGET_VARIABLES] = (uint64_t)k;
    need.need[TR_BUDGET_OUTPUT] = ncells;
    need.need[TR_BUDGET_STATES] = ncells;
    need.need[TR_BUDGET_BYTES] = tr_lattice_object_bytes(ncells, k);
    tr_math_budget dflt;
    int32_t rc = tr_budget_check(budget_or_default(budget, &dflt), &need, refusal);
    if (rc) return hist_fail(NULL, error, rc);
    if (k > TR_LATTICE_MAX_AXES) return hist_fail(NULL, error, TR_MATH_UNSUPPORTED);
    tr_lattice_hist *h = hist_alloc(k, nulls, kind);
    if (!h) return hist_fail(NULL, error, TR_MATH_MEMORY);
    memcpy(h->cells, cells, (size_t)(ncells * 8));
    uint64_t sum = 0;
    rc = derive_axis(k, nulls, ncells, h->cells, h->axis, &sum);
    if (rc) return hist_fail(h, error, rc);
    if (points == TR_LATTICE_POINTS_SUM) points = sum;
    if (nulls == TR_LATTICE_THREE) {
        if (sum != points) return hist_fail(h, error, TR_MATH_INVALID);
        if (axis && memcmp(axis, h->axis, (size_t)k * 24)) return hist_fail(h, error, TR_MATH_INVALID);
        uint64_t complete = 0;                                /* the cells without an UNKNOWN digit */
        int32_t digits[TR_LATTICE_MAX_AXES + 1] = {0}, twos = 0;
        for (uint64_t c = 0; c < ncells; c++) {
            if (!twos) complete += h->cells[c];
            for (int32_t i = 0; i < k; i++) {
                if (digits[i] == 2) twos--;
                if (++digits[i] < 3) { if (digits[i] == 2) twos++; break; }
                digits[i] = 0;
            }
        }
        h->complete = complete;
    } else {
        if (sum > points) return hist_fail(h, error, TR_MATH_INVALID);
        uint64_t unknown = points - sum;
        if (!axis) {
            if (unknown) return hist_fail(h, error, TR_MATH_INVALID);   /* the unknown points cannot be attributed */
        } else {
            uint64_t umax = 0;
            unsigned __int128 usum = 0;
            for (int32_t i = 0; i < k; i++) {
                const uint64_t *a = axis + 3 * i;
                unsigned __int128 t = (unsigned __int128)a[0] + a[1] + a[2];
                if (t != points || a[0] < h->axis[3 * i] || a[1] < h->axis[3 * i + 1])
                    return hist_fail(h, error, TR_MATH_INVALID);
                if (a[2] > umax) umax = a[2];
                usum += a[2];
            }
            if (unknown < umax || (unsigned __int128)unknown > usum || (k == 0 && unknown))
                return hist_fail(h, error, TR_MATH_INVALID);
            memcpy(h->axis, axis, (size_t)k * 24);
        }
        h->complete = sum;
    }
    h->points = points;
    hist_finish(h);
    set_error(error, TR_MATH_OK);
    return h;
}

tr_lattice_hist *tr_lattice_hist_merge(const tr_lattice_hist *a, const tr_lattice_hist *b, const tr_math_budget *budget,
                                       tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    if (!hist_ok(a) || !hist_ok(b)) return hist_fail(NULL, error, TR_MATH_INVALID);
    if (a->k != b->k || a->nulls != b->nulls) return hist_fail(NULL, error, TR_MATH_DOMAIN_ERROR);
    tr_math_need need;
    zero_need(&need);
    need.need[TR_BUDGET_VARIABLES] = (uint64_t)a->k;
    need.need[TR_BUDGET_OUTPUT] = a->ncells;
    need.need[TR_BUDGET_STATES] = a->ncells;
    need.need[TR_BUDGET_BYTES] = tr_lattice_object_bytes(a->ncells, a->k);
    tr_math_budget dflt;
    int32_t rc = tr_budget_check(budget_or_default(budget, &dflt), &need, refusal);
    if (rc) return hist_fail(NULL, error, rc);
    tr_lattice_hist *h = hist_alloc(a->k, a->nulls, a->kind == b->kind ? a->kind : 0);
    if (!h) return hist_fail(NULL, error, TR_MATH_MEMORY);
    int over = __builtin_add_overflow(a->points, b->points, &h->points) |
               __builtin_add_overflow(a->complete, b->complete, &h->complete);
    for (uint64_t c = 0; c < a->ncells; c++) over |= __builtin_add_overflow(a->cells[c], b->cells[c], &h->cells[c]);
    for (int32_t i = 0; i < 3 * a->k; i++) over |= __builtin_add_overflow(a->axis[i], b->axis[i], &h->axis[i]);
    if (over) return hist_fail(h, error, TR_MATH_OVERFLOW);
    hist_finish(h);
    set_error(error, TR_MATH_OK);
    return h;
}

int32_t tr_lattice_hist_get_info(const tr_lattice_hist *h, tr_lattice_hist_info *out) {
    if (!hist_ok(h) || !out) return TR_MATH_INVALID;
    memset(out, 0, sizeof *out);
    out->ncells = h->ncells; out->points = h->points; out->complete = h->complete;
    out->unknown = h->points - h->complete; out->occupied = h->occupied; out->bytes = h->bytes;
    out->k = h->k; out->nulls = h->nulls; out->kind = h->kind;
    return TR_MATH_OK;
}
int32_t tr_lattice_hist_cells(const tr_lattice_hist *h, uint64_t *cells, uint64_t capacity) {
    if (!hist_ok(h) || !cells || capacity < h->ncells) return TR_MATH_INVALID;
    memcpy(cells, h->cells, (size_t)(h->ncells * 8));
    return TR_MATH_OK;
}
int32_t tr_lattice_hist_axis_counts(const tr_lattice_hist *h, uint64_t *axis, uint64_t capacity) {
    if (!hist_ok(h) || (h->k && !axis) || capacity < 3 * (uint64_t)h->k) return TR_MATH_INVALID;
    if (h->k) memcpy(axis, h->axis, (size_t)h->k * 24);
    return TR_MATH_OK;
}
int32_t tr_lattice_hist_equal(const tr_lattice_hist *a, const tr_lattice_hist *b) {
    if (!hist_ok(a) || !hist_ok(b)) return TR_MATH_INVALID;
    if (a->k != b->k || a->nulls != b->nulls || a->points != b->points || a->complete != b->complete) return 0;
    if (memcmp(a->cells, b->cells, (size_t)(a->ncells * 8))) return 0;
    if (a->k && memcmp(a->axis, b->axis, (size_t)a->k * 24)) return 0;
    return 1;
}

/* the 2^k projection (counts; each sums at most `points` < 2^64, so no addition wraps) into o64 or, as tr_math_i128
 * values, into o128 (the other NULL) */
static void put(uint64_t *o64, tr_math_i128 *o128, uint64_t at, uint64_t x) {
    if (o64) o64[at] += x;
    else o128[at].lo = (int64_t)((uint64_t)o128[at].lo + x);   /* hi stays 0: the value is the unsigned low word */
}
static int32_t project(const tr_lattice_hist *h, int32_t projection, uint64_t *o64, tr_math_i128 *o128) {
    const uint64_t n = (uint64_t)1 << h->k;
    if (projection < TR_LATTICE_COMPLETE || projection > TR_LATTICE_UNKNOWN_SETS) return TR_MATH_INVALID;
    if (h->nulls == TR_LATTICE_APART && projection != TR_LATTICE_COMPLETE) return TR_MATH_DOMAIN_ERROR;
    if (o64) memset(o64, 0, (size_t)(n * 8)); else memset(o128, 0, (size_t)(n * 16));
    if (h->nulls == TR_LATTICE_APART) {
        for (uint64_t a = 0; a < n; a++) put(o64, o128, a, h->cells[a]);
        return TR_MATH_OK;
    }
    int32_t digits[TR_LATTICE_MAX_AXES + 1] = {0};
    uint64_t t = 0, f = (n - 1), u = 0;                       /* the TRUE / FALSE / UNKNOWN sets of cell c */
    for (uint64_t c = 0; c < h->ncells; c++) {
        uint64_t x = h->cells[c];
        if (x) {
            switch (projection) {
            case TR_LATTICE_COMPLETE: if (!u) put(o64, o128, t, x); break;
            case TR_LATTICE_TRUE_SETS: put(o64, o128, t, x); break;
            case TR_LATTICE_FALSE_SETS: put(o64, o128, f, x); break;
            default: put(o64, o128, u, x); break;
            }
        }
        for (int32_t i = 0; i < h->k; i++) {                  /* the next cell: 0 -> 1 -> 2 -> 0 with a carry */
            uint64_t bit = (uint64_t)1 << i;
            if (digits[i] == 0) { digits[i] = 1; f &= ~bit; t |= bit; break; }
            if (digits[i] == 1) { digits[i] = 2; t &= ~bit; u |= bit; break; }
            digits[i] = 0; u &= ~bit; f |= bit;
        }
    }
    return TR_MATH_OK;
}
static int32_t vector_check(const tr_lattice_hist *h, int32_t projection, const void *out, uint64_t capacity) {
    if (!hist_ok(h) || !out || capacity < ((uint64_t)1 << h->k)) return TR_MATH_INVALID;
    if (projection < TR_LATTICE_COMPLETE || projection > TR_LATTICE_UNKNOWN_SETS) return TR_MATH_INVALID;
    if (h->nulls == TR_LATTICE_APART && projection != TR_LATTICE_COMPLETE) return TR_MATH_DOMAIN_ERROR;
    return TR_MATH_OK;
}
int32_t tr_lattice_hist_vector(const tr_lattice_hist *h, int32_t projection, uint64_t *out, uint64_t capacity) {
    int32_t rc = vector_check(h, projection, out, capacity);
    return rc ? rc : project(h, projection, out, NULL);
}
int32_t tr_lattice_hist_vector_i128(const tr_lattice_hist *h, int32_t projection, tr_math_i128 *out, uint64_t capacity) {
    int32_t rc = vector_check(h, projection, out, capacity);
    return rc ? rc : project(h, projection, NULL, out);
}

/* ------------------------------------------------------------------ transforms */
int32_t tr_lattice_transform_need(int32_t k, int32_t op, int32_t carrier, tr_math_need *need) {
    if (!need) return TR_MATH_INVALID;
    return tr_lattice_transform_base_need(k, op, carrier, need);
}
static int fits_i64(tr_math_i128 v) { return tr_i128_fits_i64(v); }

int32_t tr_lattice_transform(int32_t k, int32_t op, int32_t carrier, const tr_math_i128 *in, tr_math_i128 *out,
                             const tr_math_budget *budget, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    tr_math_need need;
    if (tr_lattice_transform_base_need(k, op, carrier, &need) || carrier == TR_LATTICE_BIGINT || !in || !out)
        return TR_MATH_INVALID;
    tr_math_budget dflt;
    int32_t rc = tr_budget_check(budget_or_default(budget, &dflt), &need, refusal);
    if (rc) return rc;
    if (k > TR_LATTICE_MAX_AXES) return TR_MATH_UNSUPPORTED;
    const uint64_t n = (uint64_t)1 << k;
    if (carrier == TR_LATTICE_I64)
        for (uint64_t i = 0; i < n; i++) if (!fits_i64(in[i])) return TR_MATH_OVERFLOW;
    tr_math_i128 *v = zalloc(n * 16);
    if (!v) return TR_MATH_MEMORY;
    memcpy(v, in, (size_t)(n * 16));
    for (int32_t j = 0; j < k; j++) {
        const uint64_t half = (uint64_t)1 << j;
        for (uint64_t base = 0; base < n; base += 2 * half)
            for (uint64_t lo = base; lo < base + half; lo++)
                if (!tr_lattice_butterfly(op, v, lo, lo + half)) { free(v); return TR_MATH_OVERFLOW; }
    }
    if (carrier == TR_LATTICE_I64)
        for (uint64_t i = 0; i < n; i++) if (!fits_i64(v[i])) { free(v); return TR_MATH_OVERFLOW; }
    memcpy(out, v, (size_t)(n * 16));
    free(v);
    return TR_MATH_OK;
}

int32_t tr_lattice_transform_bigint(int32_t k, int32_t op, const tr_bigint *const *in, tr_bigint **out,
                                    const tr_math_budget *budget, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    tr_math_need need;
    if (tr_lattice_transform_base_need(k, op, TR_LATTICE_BIGINT, &need) || !in || !out) return TR_MATH_INVALID;
    tr_math_budget dflt;
    const tr_math_budget *b = budget_or_default(budget, &dflt);
    int32_t rc = tr_budget_check(b, &need, refusal);
    if (rc) return rc;
    if (k > TR_LATTICE_MAX_AXES) return TR_MATH_UNSUPPORTED;
    const uint64_t n = (uint64_t)1 << k;
    for (uint64_t i = 0; i < n; i++) if (!tr_bigint_is_handle(in[i])) return TR_MATH_INVALID;
    tr_bigint **v = zalloc(n * sizeof *v);
    if (!v) return TR_MATH_MEMORY;
    int32_t err = TR_MATH_OK;
    for (uint64_t i = 0; i < n && !err; i++) { v[i] = tr_bigint_copy(in[i], b, refusal, &err); if (!v[i] && !err) err = TR_MATH_MEMORY; }
    for (int32_t j = 0; j < k && !err; j++) {
        const uint64_t half = (uint64_t)1 << j;
        for (uint64_t base = 0; base < n && !err; base += 2 * half)
            for (uint64_t lo = base; lo < base + half && !err; lo++) {
                uint64_t hi = lo + half, at = (op == TR_LATTICE_ZETA_UP || op == TR_LATTICE_MOBIUS_UP) ? lo : hi;
                const tr_bigint *x = v[at], *y = v[at == lo ? hi : lo];
                tr_bigint *r = (op == TR_LATTICE_ZETA_UP || op == TR_LATTICE_ZETA_DOWN) ? tr_bigint_add(x, y, b, refusal, &err)
                                                                                      : tr_bigint_sub(x, y, b, refusal, &err);
                if (!r) { if (!err) err = TR_MATH_MEMORY; break; }
                tr_bigint_free(v[at]);
                v[at] = r;
            }
    }
    if (err) {
        for (uint64_t i = 0; i < n; i++) tr_bigint_free(v[i]);
        free(v);
        return err;
    }
    for (uint64_t i = 0; i < n; i++) out[i] = v[i];
    free(v);
    return TR_MATH_OK;
}

/* checks of a list of sets of k axes; the terms of their direct sums */
static int32_t sets_check(int32_t k, int32_t op, const uint64_t *sets, int32_t n, uint64_t *terms) {
    if (k < 0 || n < 0 || (n && !sets) || op < TR_LATTICE_ZETA_UP || op > TR_LATTICE_MOBIUS_DOWN) return TR_MATH_INVALID;
    uint64_t t = 0;
    for (int32_t j = 0; j < n; j++) {
        if (k < 64 && (sets[j] >> k)) return TR_MATH_INVALID;
        int m = popc(sets[j]);
        int up = op == TR_LATTICE_ZETA_UP || op == TR_LATTICE_MOBIUS_UP;
        t = tr_budget_add(t, tr_budget_power2((uint32_t)(up ? k - m : m)));
    }
    *terms = t;
    return TR_MATH_OK;
}
int32_t tr_lattice_coefficients_need(int32_t k, int32_t op, const uint64_t *sets, int32_t n, tr_math_need *need) {
    uint64_t terms = 0;
    if (!need) return TR_MATH_INVALID;
    int32_t rc = sets_check(k, op, sets, n, &terms);
    if (rc) return rc;
    zero_need(need);
    need->need[TR_BUDGET_VARIABLES] = (uint64_t)k;
    need->need[TR_BUDGET_TENSOR] = terms;
    need->need[TR_BUDGET_OUTPUT] = (uint64_t)n;
    need->need[TR_BUDGET_BYTES] = tr_budget_mul((uint64_t)n, 16);
    return TR_MATH_OK;
}

int32_t tr_lattice_coefficients(int32_t k, int32_t op, int32_t carrier, const tr_math_i128 *values, const uint64_t *sets,
                                int32_t n, const tr_math_budget *budget, tr_math_i128 *out, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    tr_math_need need;
    int32_t rc = tr_lattice_coefficients_need(k, op, sets, n, &need);
    if (rc) return rc;
    if ((carrier != TR_LATTICE_I64 && carrier != TR_LATTICE_I128) || !values || (n && !out)) return TR_MATH_INVALID;
    tr_math_budget dflt;
    rc = tr_budget_check(budget_or_default(budget, &dflt), &need, refusal);
    if (rc) return rc;
    if (k > TR_LATTICE_MAX_AXES || n > TR_LATTICE_MAX_REQUESTS) return TR_MATH_UNSUPPORTED;
    const uint64_t full = ((uint64_t)1 << k) - 1;
    if (carrier == TR_LATTICE_I64)
        for (uint64_t i = 0; i <= full; i++) if (!fits_i64(values[i])) return TR_MATH_OVERFLOW;
    tr_math_i128 *tmp = zalloc((uint64_t)n * 16);
    if (!tmp) return TR_MATH_MEMORY;
    const int up = op == TR_LATTICE_ZETA_UP || op == TR_LATTICE_MOBIUS_UP;
    const int alternating = op == TR_LATTICE_MOBIUS_UP || op == TR_LATTICE_MOBIUS_DOWN;
    for (int32_t j = 0; j < n; j++) {
        const uint64_t S = sets[j];
        __int128 sum = 0;
        int ok = 1;
        if (up) {                                             /* the supersets S | u, u a subset of the complement */
            const uint64_t C = full & ~S;
            uint64_t u = 0;
            do {
                __int128 x = tr_i128_get(values[S | u]);
                ok = alternating && (popc(u) & 1) ? tr_lattice_sub128(sum, x, &sum) : tr_lattice_add128(sum, x, &sum);
                u = (u - C) & C;
            } while (u && ok);
        } else {                                              /* the subsets a of S */
            uint64_t a = S;
            const int m = popc(S);
            for (;;) {
                __int128 x = tr_i128_get(values[a]);
                ok = alternating && ((m - popc(a)) & 1) ? tr_lattice_sub128(sum, x, &sum) : tr_lattice_add128(sum, x, &sum);
                if (!ok || !a) break;
                a = (a - 1) & S;
            }
        }
        if (!ok || (carrier == TR_LATTICE_I64 && (sum > INT64_MAX || sum < INT64_MIN))) { free(tmp); return TR_MATH_OVERFLOW; }
        tmp[j] = tr_i128_set(sum);
    }
    if (n) memcpy(out, tmp, (size_t)n * 16);
    free(tmp);
    return TR_MATH_OK;
}

int32_t tr_lattice_order_sets(int32_t k, int32_t min_order, int32_t max_order, const tr_math_budget *budget,
                              uint64_t *sets, uint64_t capacity, uint64_t *n, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (k < 0 || min_order < 0 || max_order < min_order || !n) return TR_MATH_INVALID;
    int32_t top = max_order < k ? max_order : k;
    uint64_t count = 0;
    for (int32_t m = min_order; m <= top; m++) count = tr_budget_add(count, tr_budget_binomial((uint64_t)k, (uint64_t)m));
    tr_math_need need;
    zero_need(&need);
    need.need[TR_BUDGET_VARIABLES] = (uint64_t)k;
    need.need[TR_BUDGET_OUTPUT] = count;
    tr_math_budget dflt;
    int32_t rc = tr_budget_check(budget_or_default(budget, &dflt), &need, refusal);
    if (rc) return rc;
    if (k > TR_LATTICE_MAX_AXES) return TR_MATH_UNSUPPORTED;
    if (!sets) { *n = count; return TR_MATH_OK; }
    if (capacity < count) return TR_MATH_INVALID;
    uint64_t at = 0;
    const uint64_t limit = (uint64_t)1 << k;
    for (int32_t m = min_order; m <= top; m++) {
        if (m == 0) { sets[at++] = 0; continue; }
        uint64_t s = ((uint64_t)1 << m) - 1;                  /* Gosper's hack: the m-subsets in increasing order */
        while (s < limit) {
            sets[at++] = s;
            uint64_t c = s & (~s + 1), r = s + c;
            s = (((r ^ s) >> 2) / c) | r;
        }
    }
    *n = at;
    return TR_MATH_OK;
}

/* ------------------------------------------------------------------ inclusion-exclusion counts */
static int32_t counts_request(const tr_lattice_hist *h, int32_t projection, const uint64_t *sets, int32_t n,
                              tr_math_need *need, int32_t *top) {
    if (!hist_ok(h) || n < 0 || (n && !sets)) return TR_MATH_INVALID;
    if (projection < TR_LATTICE_COMPLETE || projection > TR_LATTICE_UNKNOWN_SETS) return TR_MATH_INVALID;
    if (h->nulls == TR_LATTICE_APART && projection != TR_LATTICE_COMPLETE) return TR_MATH_DOMAIN_ERROR;
    const int32_t k = h->k;
    const uint64_t size = (uint64_t)1 << k;
    uint64_t tensor = h->ncells;
    int32_t m_top = 0;
    for (int32_t j = 0; j < n; j++) {
        if (k < 64 && (sets[j] >> k)) return TR_MATH_INVALID;
        int32_t m = popc(sets[j]);
        if (m > m_top) m_top = m;
        uint64_t sub = (uint64_t)1 << m;
        tensor = tr_budget_add(tensor, tr_budget_add(size, tr_budget_add(tr_budget_mul((uint64_t)m, sub), sub)));
    }
    zero_need(need);
    need->need[TR_BUDGET_VARIABLES] = (uint64_t)k;
    need->need[TR_BUDGET_DEGREE] = (uint64_t)m_top;
    need->need[TR_BUDGET_TENSOR] = tensor;
    need->need[TR_BUDGET_OUTPUT] = (uint64_t)n;
    need->need[TR_BUDGET_BYTES] = tr_budget_add(tr_budget_add(tr_budget_mul(size, 8), tr_budget_mul((uint64_t)1 << m_top, 16)),
                                                tr_budget_mul((uint64_t)n, sizeof(tr_lattice_count)));
    *top = m_top;
    return TR_MATH_OK;
}
int32_t tr_lattice_counts_need(const tr_lattice_hist *h, int32_t projection, const uint64_t *sets, int32_t n,
                               tr_math_need *need) {
    int32_t top = 0;
    if (!need) return TR_MATH_INVALID;
    return counts_request(h, projection, sets, n, need, &top);
}

static __int128 abs128(__int128 x) { return x < 0 ? -x : x; }   /* |x| < 2^127 here: the partial sums are bounded */

int32_t tr_lattice_counts(const tr_lattice_hist *h, int32_t projection, const uint64_t *sets, int32_t n,
                          const tr_math_budget *budget, tr_lattice_count *out, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    tr_math_need need;
    int32_t top = 0;
    int32_t rc = counts_request(h, projection, sets, n, &need, &top);
    if (rc) return rc;
    if (n && !out) return TR_MATH_INVALID;
    tr_math_budget dflt;
    rc = tr_budget_check(budget_or_default(budget, &dflt), &need, refusal);
    if (rc) return rc;
    if (n > TR_LATTICE_MAX_REQUESTS) return TR_MATH_UNSUPPORTED;
    const uint64_t size = (uint64_t)1 << h->k, full = size - 1;
    uint64_t *proj = zalloc(size * 8);
    __int128 *g = zalloc(((uint64_t)1 << top) * 16);
    tr_lattice_count *tmp = zalloc((uint64_t)n * sizeof *tmp);
    if (!proj || !g || !tmp) { free(proj); free(g); free(tmp); return TR_MATH_MEMORY; }
    project(h, projection, proj, NULL);
    uint64_t N = 0;
    for (uint64_t a = 0; a < size; a++) N += proj[a];        /* <= points: never wraps */
    rc = TR_MATH_OK;
    for (int32_t j = 0; j < n && !rc; j++) {
        const uint64_t S = sets[j], C = full & ~S;
        const int32_t m = popc(S);
        const uint64_t sub = (uint64_t)1 << m;
        uint64_t s = 0, t = 0;
        do {                                                  /* the marginal on S: t = the position of s within S */
            uint64_t sum = 0, u = 0;
            do { sum += proj[s | u]; u = (u - C) & C; } while (u);
            g[t++] = (__int128)sum;
            s = (s - S) & S;
        } while (s);
        for (int32_t b = 0; b < m && !rc; b++)                /* superset sums within S: I(T) */
            for (uint64_t x = 0; x < sub; x++)
                if (!((x >> b) & 1) && !tr_lattice_add128(g[x], g[x | ((uint64_t)1 << b)], &g[x])) rc = TR_MATH_OVERFLOW;
        if (rc) break;
        __int128 order_sum[TR_LATTICE_MAX_AXES + 1] = {0};
        for (uint64_t x = 1; x < sub && !rc; x++)
            if (!tr_lattice_add128(order_sum[popc(x)], g[x], &order_sum[popc(x)])) rc = TR_MATH_OVERFLOW;
        __int128 any = 0, peak = 0;
        for (int32_t o = 1; o <= m && !rc; o++) {             /* Bonferroni partial sums, in the order of |T| */
            if (!((o & 1) ? tr_lattice_add128(any, order_sum[o], &any) : tr_lattice_sub128(any, order_sum[o], &any)))
                rc = TR_MATH_OVERFLOW;
            else if (abs128(any) > peak) peak = abs128(any);
        }
        if (rc) break;
        tr_lattice_count *R = &tmp[j];
        R->vars = S; R->order = m; R->reserved = 0; R->terms = sub - 1;
        R->all = tr_i128_set(g[sub - 1]);
        R->any = tr_i128_set(any);
        R->none = tr_i128_set((__int128)N - any);
        R->exactly = tr_i128_set((__int128)proj[S]);
        R->peak = tr_i128_set(peak);
    }
    if (!rc && n) memcpy(out, tmp, (size_t)n * sizeof *tmp);
    free(proj); free(g); free(tmp);
    return rc;
}
