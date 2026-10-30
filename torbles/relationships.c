/* Experimental native algebraic discovery. SPDX-License-Identifier: Apache-2.0
 * The catalog contains finite, full-snapshot proofs, not universal identities.
 * Logical integer values are code + codec offset; every proposal is checked
 * independently on every used, complete row before it may become a rewrite.
 */
#include "relationships.h"
#include <limits.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

extern void at_decode_column(const atable_t *, int, uint32_t *);

typedef __int128 tr_i128;

static uint64_t tail_mask(uint64_t rows, uint64_t word) {
    uint64_t left = rows - word * 64;
    return left >= 64 ? UINT64_MAX : (left ? (UINT64_C(1) << left) - 1 : 0);
}

static int fresh(const tr_catalog *cat) {
    return cat && cat->table && cat->table->revision == cat->revision &&
           cat->table->n == cat->rows;
}

static int descriptor(const tr_catalog *cat, int col) {
    for (int i = 0; i < cat->ncols; ++i)
        if (cat->columns[i].col == col) return i;
    return -1;
}

/* Follow the core's decode-on-use approach: a physical-planning-only catalog
 * needs no row-sized decoded column buffers. Explicit probes decode on demand. */
static int ensure_decoded(tr_catalog *cat) {
    for (int i = 0; i < cat->ncols; ++i) {
        size_t n = cat->rows ? (size_t)cat->rows : 1;
        if (!cat->codes[i]) {
            cat->codes[i] = malloc(n * sizeof *cat->codes[i]);
            if (!cat->codes[i]) return TR_MEMORY;
            at_decode_column(cat->table, cat->columns[i].col, cat->codes[i]);
        }
        if (cat->columns[i].valid >= 0 && !cat->validity[i]) {
            cat->validity[i] = malloc(n * sizeof *cat->validity[i]);
            if (!cat->validity[i]) return TR_MEMORY;
            at_decode_column(cat->table, cat->columns[i].valid, cat->validity[i]);
        }
    }
    return TR_OK;
}

static int used_row(const tr_catalog *cat, uint64_t row) {
    return (int)((cat->used[row >> 6] >> (row & 63)) & 1);
}

static int complete(const tr_catalog *cat, int i, uint64_t row) {
    return !cat->validity[i] || cat->validity[i][row] != 0;
}

static int logical(const tr_catalog *cat, int i, uint64_t row, int64_t *out) {
    tr_i128 value = (tr_i128)cat->codes[i][row] + cat->columns[i].offset;
    if (value < INT64_MIN || value > INT64_MAX) return TR_OVERFLOW;
    *out = (int64_t)value;
    return TR_OK;
}

/* Canonical raw-code RHS. Never silently narrow a coefficient. */
static int term(tr_poly *p, int a, int b, tr_i128 coefficient) {
    if (!coefficient) return TR_OK;
    if (a < 0) a = b = -1;
    else if (b >= 0 && b < a) { int tmp = a; a = b; b = tmp; }
    if (coefficient < INT64_MIN || coefficient > INT64_MAX) return TR_OVERFLOW;
    for (int i = 0; i < p->n; ++i) {
        if (p->terms[i].a == a && p->terms[i].b == b) {
            tr_i128 value = (tr_i128)p->terms[i].coefficient + coefficient;
            if (value < INT64_MIN || value > INT64_MAX) return TR_OVERFLOW;
            p->terms[i].coefficient = (int64_t)value;
            if (!value) p->terms[i] = p->terms[--p->n];
            return TR_OK;
        }
    }
    if (p->n == TR_MAX_TERMS) return TR_INVALID;
    p->terms[p->n++] = (tr_term){a, b, (int64_t)coefficient};
    return TR_OK;
}

/* A module owns its proposal, exact logical-value prediction, and lowering to
 * the shared raw-code equation IR. The verifier and optimizer do not know the
 * algebra of individual modules. This static registry deliberately avoids a
 * dynamic-plugin ABI: a new unary/binary family adds functions and one entry. */
typedef struct {
    int32_t id, arity, symmetric;
    int (*propose)(const tr_catalog *, int, int, int, int64_t *, int64_t *);
    int (*predict)(const tr_rule *, int64_t, int64_t, tr_i128 *);
    int (*lower)(const tr_catalog *, tr_rule *, int, int, int);
    int (*nondegenerate)(const tr_rule *);
} tr_module;

static int affine_proposal(const tr_catalog *, int, int, int, int64_t *, int64_t *);

static int unit_proposal(const tr_catalog *cat, int oi, int li, int ri, int64_t *scale, int64_t *bias) {
    (void)cat; (void)oi; (void)li; (void)ri;
    *scale = 1; *bias = 0;
    return 1;
}

static int affine_predict(const tr_rule *rule, int64_t x, int64_t y, tr_i128 *out) {
    (void)y;
    return (__builtin_mul_overflow((tr_i128)rule->scale, (tr_i128)x, out) ||
            __builtin_add_overflow(*out, (tr_i128)rule->bias, out)) ? TR_OVERFLOW : TR_OK;
}

static int product_predict(const tr_rule *rule, int64_t x, int64_t y, tr_i128 *out) {
    (void)rule;
    return __builtin_mul_overflow((tr_i128)x, (tr_i128)y, out) ? TR_OVERFLOW : TR_OK;
}

static int sum_predict(const tr_rule *rule, int64_t x, int64_t y, tr_i128 *out) {
    (void)rule;
    return __builtin_add_overflow((tr_i128)x, (tr_i128)y, out) ? TR_OVERFLOW : TR_OK;
}

static int affine_lower(const tr_catalog *cat, tr_rule *rule, int oi, int li, int ri) {
    (void)ri;
    int rc = term(&rule->rhs, rule->left, -1, rule->scale);
    if (rc != TR_OK) return rc;
    return term(&rule->rhs, -1, -1, (tr_i128)rule->scale * cat->columns[li].offset +
                rule->bias - cat->columns[oi].offset);
}

static int sum_lower(const tr_catalog *cat, tr_rule *rule, int oi, int li, int ri) {
    int rc = term(&rule->rhs, rule->left, -1, 1);
    if (rc != TR_OK) return rc;
    rc = term(&rule->rhs, rule->right, -1, 1);
    if (rc != TR_OK) return rc;
    return term(&rule->rhs, -1, -1, (tr_i128)cat->columns[li].offset +
                cat->columns[ri].offset - cat->columns[oi].offset);
}

static int product_lower(const tr_catalog *cat, tr_rule *rule, int oi, int li, int ri) {
    tr_i128 left_offset = cat->columns[li].offset, right_offset = cat->columns[ri].offset;
    int rc = term(&rule->rhs, rule->left, rule->right, 1);
    if (rc != TR_OK) return rc;
    rc = term(&rule->rhs, rule->left, -1, right_offset);
    if (rc != TR_OK) return rc;
    rc = term(&rule->rhs, rule->right, -1, left_offset);
    if (rc != TR_OK) return rc;
    return term(&rule->rhs, -1, -1, left_offset * right_offset - cat->columns[oi].offset);
}

static int affine_nondegenerate(const tr_rule *rule) { return rule->scale != 0; }
static int binary_nondegenerate(const tr_rule *rule) { (void)rule; return 1; }

/* Preserve stable module/rule enumeration for existing reports and traces. */
static const tr_module MODULES[] = {
    {TR_AFFINE, 1, 0, affine_proposal, affine_predict, affine_lower, affine_nondegenerate},
    {TR_PRODUCT, 2, 1, unit_proposal, product_predict, product_lower, binary_nondegenerate},
    {TR_SUM, 2, 1, unit_proposal, sum_predict, sum_lower, binary_nondegenerate},
};
#define TR_MODULE_COUNT (sizeof MODULES / sizeof MODULES[0])

static const tr_module *module_by_id(int32_t id) {
    for (size_t i = 0; i < TR_MODULE_COUNT; ++i) if (MODULES[i].id == id) return &MODULES[i];
    return NULL;
}

static int32_t module_mask(void) {
    int32_t mask = 0;
    for (size_t i = 0; i < TR_MODULE_COUNT; ++i) mask |= MODULES[i].id;
    return mask;
}

int32_t tr_scope_valid(const tr_catalog *cat, const atable_t *table, const uint64_t *used) {
    if (!cat || !table || table != cat->table || table->revision != cat->revision ||
        table->n != cat->rows) return 0;
    for (uint64_t w = 0; w < table->nw; ++w) {
        uint64_t mask = tail_mask(cat->rows, w);
        uint64_t current = used ? used[w] & mask : mask;
        if (current != cat->used[w]) return 0;
    }
    return 1;
}

void tr_free(tr_catalog *cat) {
    if (!cat) return;
    tr_system_cache_free(cat->physical_cache);
    free(cat->used);
    for (int i = 0; i < TR_MAX_COLUMNS; ++i) {
        free(cat->codes[i]);
        free(cat->validity[i]);
    }
    free(cat);
}

int32_t tr_rule_count(const tr_catalog *cat) {
    return !fresh(cat) ? TR_STALE : cat->nrules;
}

int32_t tr_stats(const tr_catalog *cat, tr_catalog_stats *out) {
    if (!cat || !out) return TR_INVALID;
    if (!fresh(cat)) return TR_STALE;
    memset(out, 0, sizeof *out);
    out->rows = cat->rows; out->used_rows = cat->used_count;
    out->candidates = cat->candidates; out->rules = cat->nrules;
    out->truncated = cat->truncated; out->refused = cat->refused;
    for (int i = 0; i < cat->ncols; ++i)
        out->decoded_bytes += cat->rows * sizeof(uint32_t) * ((cat->codes[i] ? 1 : 0) + (cat->validity[i] ? 1 : 0));
    return TR_OK;
}

int32_t tr_get_rule(const tr_catalog *cat, int32_t index, tr_rule *out) {
    if (!cat || !out || index < 0 || index >= cat->nrules) return TR_INVALID;
    if (!fresh(cat)) return TR_STALE;
    *out = cat->rules[index];
    return TR_OK;
}

int32_t tr_probe(tr_catalog *cat, int32_t module, int32_t out, int32_t left, int32_t right,
                 int64_t scale, int64_t bias, tr_rule *evidence) {
    if (!cat || !evidence) return TR_INVALID;
    memset(evidence, 0, sizeof *evidence);
    evidence->module = module;
    evidence->out = out; evidence->left = left; evidence->right = right;
    evidence->scale = scale; evidence->bias = bias;
    evidence->witness = UINT64_MAX;
    if (!fresh(cat)) return TR_STALE;
    const tr_module *implementation = module_by_id(module);
    if (!implementation) return TR_INVALID;
    int oi = descriptor(cat, out), li = descriptor(cat, left);
    int ri = implementation->arity == 1 ? -1 : descriptor(cat, right);
    if (oi < 0 || li < 0 || out == left ||
        (implementation->arity == 2 && (ri < 0 || out == right))) return TR_INVALID;
    int decoded = ensure_decoded(cat);
    if (decoded != TR_OK) return decoded;
    int varied_left = 0, varied_right = implementation->arity == 1;
    int seen = 0;
    int64_t first_left = 0, first_right = 0;
    for (uint64_t row = 0; row < cat->rows; ++row) {
        if (!used_row(cat, row)) continue;
        if (!complete(cat, oi, row) || !complete(cat, li, row) ||
            (ri >= 0 && !complete(cat, ri, row))) {
            ++evidence->excluded;
            continue;
        }
        int64_t actual, x, y = 0;
        if (logical(cat, oi, row, &actual) != TR_OK || logical(cat, li, row, &x) != TR_OK ||
            (ri >= 0 && logical(cat, ri, row, &y) != TR_OK)) return TR_OVERFLOW;
        if (!seen) { first_left = x; first_right = y; seen = 1; }
        else { varied_left |= x != first_left; varied_right |= y != first_right; }
        tr_i128 predicted;
        int rc = implementation->predict(evidence, x, y, &predicted);
        if (rc != TR_OK) return rc;
        ++evidence->checked;
        if (predicted != (tr_i128)actual) {
            ++evidence->violations;
            if (evidence->witness == UINT64_MAX) evidence->witness = row;
        }
    }
    /* The current engine's fact scan is physical: subset masks are evidence only,
     * never a license to rewrite unverified rows. */
    evidence->usable = cat->used_count == cat->rows && evidence->checked >= 2 && !evidence->excluded &&
        !evidence->violations && varied_left && varied_right &&
        implementation->nondegenerate(evidence);
    if (!evidence->usable) return TR_OK;
    memset(&evidence->rhs, 0, sizeof evidence->rhs);
    int rc = implementation->lower(cat, evidence, oi, li, ri);
    if (rc != TR_OK) { evidence->usable = 0; return rc; }
    for (int i = 0; i < cat->nrules; ++i) {
        const tr_rule *rule = &cat->rules[i];
        if (rule->module == module && rule->out == out && rule->left == left &&
            rule->right == right && rule->scale == scale && rule->bias == bias) return TR_OK;
    }
    if (cat->nrules == TR_MAX_RULES) { cat->truncated = 1; return TR_OK; }
    cat->rules[cat->nrules++] = *evidence;
    return TR_OK;
}

/* Find a two-point affine proposal in original integer coordinates. It remains
 * only a proposal until tr_probe checks every used row, including NULL policy. */
static int affine_proposal(const tr_catalog *cat, int oi, int li, int ri, int64_t *scale, int64_t *bias) {
    (void)ri;
    int seen = 0;
    int64_t first_x = 0, first_y = 0;
    for (uint64_t row = 0; row < cat->rows; ++row) {
        if (!used_row(cat, row) || !complete(cat, oi, row) || !complete(cat, li, row)) continue;
        int64_t x, y;
        if (logical(cat, li, row, &x) != TR_OK || logical(cat, oi, row, &y) != TR_OK) return 0;
        if (!seen) { first_x = x; first_y = y; seen = 1; continue; }
        if (x == first_x) continue;
        tr_i128 dx = (tr_i128)x - first_x, dy = (tr_i128)y - first_y;
        if (dy % dx) return 0;
        tr_i128 a = dy / dx, b;
        if (!a || a < INT64_MIN || a > INT64_MAX ||
            __builtin_mul_overflow(a, (tr_i128)first_x, &b) ||
            __builtin_sub_overflow((tr_i128)first_y, b, &b) ||
            b < INT64_MIN || b > INT64_MAX) return 0;
        *scale = (int64_t)a; *bias = (int64_t)b;
        return 1;
    }
    return 0;
}

tr_catalog *tr_discover(const atable_t *table, const tr_column *columns, int32_t ncols,
                        const uint64_t *used, int32_t modules, int32_t max_candidates, int32_t *error) {
    if (error) *error = TR_INVALID;
    if (!table || !columns || ncols < 1 || ncols > TR_MAX_COLUMNS || max_candidates < 0 ||
        (modules & ~module_mask()) || modules < 0 ||
        table->n > SIZE_MAX / sizeof(uint32_t) || table->nw > SIZE_MAX / sizeof(uint64_t)) return NULL;
    for (int i = 0; i < ncols; ++i) {
        int col = columns[i].col, valid = columns[i].valid;
        if (col < 0 || col >= table->k || table->width[col] < 1 || table->width[col] > 32 || table->vflag[col] ||
            valid < -1 || valid >= table->k || (valid >= 0 &&
            (table->width[valid] != 1 || table->vflag[valid]))) return NULL;
        for (int j = 0; j < i; ++j) if (columns[j].col == col) return NULL;
    }
    tr_catalog *cat = calloc(1, sizeof *cat);
    if (!cat) { if (error) *error = TR_MEMORY; return NULL; }
    cat->table = table; cat->revision = table->revision; cat->rows = table->n; cat->ncols = ncols;
    memcpy(cat->columns, columns, (size_t)ncols * sizeof *columns);
    cat->used = calloc(table->nw ? (size_t)table->nw : 1, sizeof *cat->used);
    if (!cat->used) goto memory_error;
    for (uint64_t w = 0; w < table->nw; ++w) {
        uint64_t mask = tail_mask(table->n, w);
        cat->used[w] = used ? used[w] & mask : mask;
        cat->used_count += (uint64_t)__builtin_popcountll(cat->used[w]);
    }
    if (modules && max_candidates && ensure_decoded(cat) != TR_OK) goto memory_error;
    for (size_t mi = 0; mi < TR_MODULE_COUNT; ++mi) {
        const tr_module *implementation = &MODULES[mi];
        if (!(modules & implementation->id)) continue;
        for (int oi = 0; oi < ncols; ++oi) for (int li = 0; li < ncols; ++li) {
            if (li == oi) continue;
            int begin = implementation->arity == 1 || !implementation->symmetric ? 0 : li;
            int end = implementation->arity == 1 ? 1 : ncols;
            for (int ri = begin; ri < end; ++ri) {
                if (implementation->arity == 2 && ri == oi) continue;
                if (cat->candidates == max_candidates || cat->nrules == TR_MAX_RULES) {
                    cat->truncated = 1;
                    if (error) *error = TR_OK;
                    return cat;
                }
                ++cat->candidates;
                int64_t scale = 1, bias = 0;
                if (!implementation->propose(cat, oi, li, ri, &scale, &bias)) continue;
                tr_rule report;
                int rc = tr_probe(cat, implementation->id, columns[oi].col, columns[li].col,
                         implementation->arity == 1 ? -1 : columns[ri].col, scale, bias, &report);
                if (rc != TR_OK) cat->refused++;
            }
        }
    }
    if (!fresh(cat)) { tr_free(cat); if (error) *error = TR_STALE; return NULL; }
    if (error) *error = TR_OK;
    return cat;

memory_error:
    tr_free(cat);
    if (error) *error = TR_MEMORY;
    return NULL;
}
