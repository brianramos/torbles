/* FIELD-001: packed GF(2) vectors and matrices and parity dependencies over table rows; see math_field.h.
 * SPDX-License-Identifier: Apache-2.0 */
#include "math_field.h"
#include "mathematics.h"
#include "math_domain.h"
#include "math_handles.h"
#include <stdlib.h>
#include <string.h>

TR_MATH_ABI(tr_field_matrix, TR_FIELD_ABI);        /* the ctypes boundary of FIELD-001 (math_handles.h) */
TR_MATH_ABI(tr_field_bitspec, TR_FIELD_ABI);
TR_MATH_ABI(tr_field_bits_view, TR_FIELD_ABI);
TR_MATH_ABI(tr_field_relation, TR_FIELD_ABI);
TR_MATH_ABI(tr_field_evidence, TR_FIELD_ABI);
TR_MATH_ABI(tr_field_discover_spec, TR_FIELD_ABI);
TR_MATH_ABI(tr_field_discovery, TR_FIELD_ABI);
TR_MATH_ABI(tr_field_bits_info, TR_FIELD_ABI);

_Static_assert(sizeof(tr_field_matrix) == 32, "tr_field_matrix ABI");
_Static_assert(sizeof(tr_field_bitspec) == 24, "tr_field_bitspec ABI");
_Static_assert(sizeof(tr_field_bits_view) == 56, "tr_field_bits_view ABI");
_Static_assert(sizeof(tr_field_relation) == 16, "tr_field_relation ABI");
_Static_assert(sizeof(tr_field_evidence) == 40, "tr_field_evidence ABI");
_Static_assert(sizeof(tr_field_discover_spec) == 16, "tr_field_discover_spec ABI");
_Static_assert(sizeof(tr_field_discovery) == 64, "tr_field_discovery ABI");
_Static_assert(sizeof(tr_field_bits_info) == 56, "tr_field_bits_info ABI");

#define BITS_MAGIC UINT32_C(0x46424954)

struct tr_field_bits_s {
    uint32_t magic;
    int32_t nvars;
    uint64_t rows, nw, selected, bytes;
    uint64_t digest[2];
    uint64_t *selection, *value, *valid;   /* valid NULL: no selected value is NULL */
};

static void set_error(int32_t *error, int32_t code) { if (error) *error = code; }
static void *zalloc(uint64_t bytes) {
    if (bytes > (uint64_t)SIZE_MAX) return NULL;
    return calloc(1, bytes ? (size_t)bytes : 1);
}
static void zero_need(tr_math_need *n) { for (int i = 0; i < TR_BUDGET_RESOURCES; i++) n->need[i] = 0; }
static int32_t check_budget(const tr_math_budget *b, const tr_math_need *need, tr_math_refusal *refusal) {
    tr_math_budget dflt;
    if (!b) { tr_budget_default(&dflt); b = &dflt; }
    return tr_budget_check(b, need, refusal);
}
static uint64_t min_u64(uint64_t a, uint64_t b) { return a < b ? a : b; }
/* clears n words (a row of a zero-stride matrix has no words and may be a NULL pointer: nothing is touched) */
static void zero_words(uint64_t *p, uint64_t n) { if (n) memset(p, 0, (size_t)(n * 8)); }

/* ------------------------------------------------------------------ elimination */
/* a canonical matrix's words into a compact buffer of W words per row (the first `cols` bits; the rest zero) */
static void copy_rows(const tr_field_matrix *a, uint64_t *dst, uint64_t W) {
    uint64_t aw = tr_field_words(a->cols);
    for (uint64_t i = 0; i < a->rows; i++) {
        uint64_t *d = dst + i * W;
        zero_words(d, W);
        if (aw) memcpy(d, a->words + i * a->stride, (size_t)(aw * 8));
    }
}
/* Gauss-Jordan (full = 1: every other row reduced; 0: only the rows below, enough for the rank) on rows x st words,
 * pivots searched in columns 0..ncols-1 in increasing order; pivots[r] = the pivot column of row r; returns the rank.
 * Rows at or below the current row are zero in every column before the current one, so reductions start at the
 * pivot's word. */
static uint64_t eliminate(uint64_t *M, uint64_t rows, uint64_t st, uint64_t ncols, uint64_t *pivots, int full) {
    uint64_t r = 0;
    for (uint64_t c = 0; c < ncols && r < rows; c++) {
        const uint64_t wi = c / 64, bit = (uint64_t)1 << (c % 64);
        uint64_t p = r;
        while (p < rows && !(M[p * st + wi] & bit)) p++;
        if (p == rows) continue;
        uint64_t *R = M + r * st;
        if (p != r) {
            uint64_t *X = M + p * st;
            for (uint64_t w = wi; w < st; w++) { uint64_t t = R[w]; R[w] = X[w]; X[w] = t; }
        }
        for (uint64_t i = full ? 0 : r + 1; i < rows; i++) {
            if (i == r) continue;
            uint64_t *X = M + i * st;
            if (X[wi] & bit) for (uint64_t w = wi; w < st; w++) X[w] ^= R[w];
        }
        if (pivots) pivots[r] = c;
        r++;
    }
    return r;
}
static void elim_need(tr_math_need *need, uint64_t rows, uint64_t cols, uint64_t scratch_bytes) {
    zero_need(need);
    need->need[TR_BUDGET_TENSOR] = tr_budget_mul(rows, cols);
    need->need[TR_BUDGET_BYTES] = scratch_bytes;
}
/* the canonical nullspace vectors of an RREF R (rank rows, W words each, pivots increasing) over `cols` columns into
 * basis rows 0..nullity-1; the other basis rows are zeroed */
static void nullspace_from_rref(const uint64_t *R, uint64_t W, uint64_t rank, const uint64_t *pivots, uint64_t cols,
                                tr_field_matrix *basis) {
    uint64_t n = 0, next = 0;
    for (uint64_t i = 0; i < basis->rows; i++) zero_words(basis->words + i * basis->stride, basis->stride);
    for (uint64_t f = 0; f < cols; f++) {
        if (next < rank && pivots[next] == f) { next++; continue; }       /* a pivot column */
        uint64_t *v = basis->words + n * basis->stride;
        v[f / 64] |= (uint64_t)1 << (f % 64);
        for (uint64_t i = 0; i < next; i++)                               /* pivots before f only: RREF */
            if ((R[i * W + f / 64] >> (f % 64)) & 1) v[pivots[i] / 64] |= (uint64_t)1 << (pivots[i] % 64);
        n++;
    }
}

int32_t tr_field_check(const tr_field_matrix *m) { return tr_field_check_matrix(m); }

int32_t tr_field_rank(const tr_field_matrix *a, const tr_math_budget *budget, uint64_t *rank, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    int32_t rc = tr_field_check_matrix(a);
    if (rc) return rc;
    if (!rank) return TR_MATH_INVALID;
    uint64_t W = tr_field_words(a->cols);
    tr_math_need need;
    elim_need(&need, a->rows, a->cols, tr_budget_mul(tr_budget_mul(a->rows, W), 8));
    rc = check_budget(budget, &need, refusal);
    if (rc) return rc;
    uint64_t *M = zalloc(a->rows * W * 8);
    if (!M) return TR_MATH_MEMORY;
    copy_rows(a, M, W);
    *rank = W ? eliminate(M, a->rows, W, a->cols, NULL, 0) : 0;
    free(M);
    return TR_MATH_OK;
}

int32_t tr_field_rref(const tr_field_matrix *a, tr_field_matrix *out, uint64_t *pivots, uint64_t *rank,
                      const tr_math_budget *budget, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    int32_t rc = tr_field_check_matrix(a);
    if (!rc) rc = tr_field_check_output(out, a->rows, a->cols);
    if (rc) return rc;
    if (!rank || (min_u64(a->rows, a->cols) && !pivots) ||
        tr_field_overlap(a->words, tr_field_matrix_bytes(a), out->words, tr_field_matrix_bytes(out))) return TR_MATH_INVALID;
    tr_math_need need;
    elim_need(&need, a->rows, a->cols, 0);
    need.need[TR_BUDGET_OUTPUT] = tr_budget_mul(a->rows, a->cols);
    rc = check_budget(budget, &need, refusal);
    if (rc) return rc;
    uint64_t W = tr_field_words(a->cols);
    for (uint64_t i = 0; i < a->rows; i++) {
        uint64_t *d = out->words + i * out->stride;
        zero_words(d, out->stride);
        if (W) memcpy(d, a->words + i * a->stride, (size_t)(W * 8));
    }
    *rank = W ? eliminate(out->words, a->rows, out->stride, a->cols, pivots, 1) : 0;
    return TR_MATH_OK;
}

int32_t tr_field_nullspace(const tr_field_matrix *a, tr_field_matrix *basis, uint64_t *nullity,
                           const tr_math_budget *budget, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    int32_t rc = tr_field_check_matrix(a);
    if (rc) return rc;
    if (!basis || !nullity || basis->cols != a->cols || basis->rows < a->cols ||
        basis->stride < tr_field_words(a->cols) || (basis->rows && basis->stride && !basis->words) ||
        tr_field_overlap(a->words, tr_field_matrix_bytes(a), basis->words, tr_field_matrix_bytes(basis)))
        return TR_MATH_INVALID;
    uint64_t W = tr_field_words(a->cols), np = min_u64(a->rows, a->cols);
    tr_math_need need;
    elim_need(&need, a->rows, a->cols, tr_budget_add(tr_budget_mul(tr_budget_mul(a->rows, W), 8), tr_budget_mul(np, 8)));
    rc = check_budget(budget, &need, refusal);
    if (rc) return rc;
    uint64_t *M = zalloc(a->rows * W * 8), *piv = zalloc(np * 8);
    if (!M || !piv) { free(M); free(piv); return TR_MATH_MEMORY; }
    copy_rows(a, M, W);
    uint64_t r = W ? eliminate(M, a->rows, W, a->cols, piv, 1) : 0;
    zero_need(&need);                                         /* the output, now that its size is known */
    need.need[TR_BUDGET_OUTPUT] = tr_budget_mul(a->cols - r, a->cols);
    rc = check_budget(budget, &need, refusal);
    if (!rc) { nullspace_from_rref(M, W, r, piv, a->cols, basis); *nullity = a->cols - r; }
    free(M); free(piv);
    return rc;
}

int32_t tr_field_solve(const tr_field_matrix *a, const uint64_t *b, uint64_t *x, tr_field_matrix *basis,
                       uint64_t *cert, int32_t *status, uint64_t *rank, const tr_math_budget *budget,
                       tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    int32_t rc = tr_field_check_matrix(a);
    if (rc) return rc;
    const uint64_t rows = a->rows, cols = a->cols, RW = tr_field_words(rows), XW = tr_field_words(cols);
    if (!status || !rank || (RW && !b) || (XW && !x)) return TR_MATH_INVALID;
    if (RW && (b[RW - 1] & ~tr_field_tail(rows))) return TR_MATH_INVALID;          /* b is canonical */
    if (basis && (basis->cols != cols || basis->rows < cols || basis->stride < XW ||
                  (basis->rows && basis->stride && !basis->words))) return TR_MATH_INVALID;
    const uint64_t extra = cert ? rows : 0, total = cols + 1 + extra, W = tr_field_words(total);
    const uint64_t np = min_u64(rows, cols + 1);
    tr_math_need need;
    elim_need(&need, rows, cols + 1 + extra, tr_budget_add(tr_budget_mul(tr_budget_mul(rows, W), 8), tr_budget_mul(np, 8)));
    rc = check_budget(budget, &need, refusal);
    if (rc) return rc;
    uint64_t *M = zalloc(rows * W * 8), *piv = zalloc(np * 8);
    if (!M || !piv) { free(M); free(piv); return TR_MATH_MEMORY; }
    copy_rows(a, M, W);                                       /* [A | b | I] */
    for (uint64_t i = 0; i < rows; i++) {
        uint64_t *row = M + i * W;
        if ((b[i / 64] >> (i % 64)) & 1) row[cols / 64] |= (uint64_t)1 << (cols % 64);
        if (cert) { uint64_t c = cols + 1 + i; row[c / 64] |= (uint64_t)1 << (c % 64); }
    }
    uint64_t r = eliminate(M, rows, W, cols + 1, piv, 1);
    int inconsistent = r && piv[r - 1] == cols;
    uint64_t ra = inconsistent ? r - 1 : r, nullity = cols - ra;
    zero_need(&need);
    need.need[TR_BUDGET_OUTPUT] = tr_budget_add(cols, tr_budget_add(basis && !inconsistent ? tr_budget_mul(nullity, cols) : 0,
                                                                   cert && inconsistent ? rows : 0));
    rc = check_budget(budget, &need, refusal);
    if (rc) { free(M); free(piv); return rc; }
    if (XW) memset(x, 0, (size_t)(XW * 8));
    if (inconsistent) {
        *status = TR_FIELD_NONE;
        if (cert) {                                           /* the row operations that produced [0 | 1]: y */
            const uint64_t *row = M + (r - 1) * W;
            if (RW) memset(cert, 0, (size_t)(RW * 8));
            for (uint64_t i = 0; i < rows; i++) {
                uint64_t c = cols + 1 + i;
                if ((row[c / 64] >> (c % 64)) & 1) cert[i / 64] |= (uint64_t)1 << (i % 64);
            }
        }
    } else {
        for (uint64_t i = 0; i < ra; i++)
            if ((M[i * W + cols / 64] >> (cols % 64)) & 1) x[piv[i] / 64] |= (uint64_t)1 << (piv[i] % 64);
        *status = nullity ? TR_FIELD_FAMILY : TR_FIELD_UNIQUE;
        if (basis) nullspace_from_rref(M, W, ra, piv, cols, basis);
    }
    *rank = ra;
    free(M); free(piv);
    return TR_MATH_OK;
}

int32_t tr_field_mul(const tr_field_matrix *a, const tr_field_matrix *b, tr_field_matrix *out,
                     const tr_math_budget *budget, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    int32_t rc = tr_field_check_matrix(a);
    if (!rc) rc = tr_field_check_matrix(b);
    if (rc) return rc;
    if (a->cols != b->rows) return TR_MATH_INVALID;
    rc = tr_field_check_output(out, a->rows, b->cols);
    if (rc) return rc;
    uint64_t ob = tr_field_matrix_bytes(out);
    if (tr_field_overlap(a->words, tr_field_matrix_bytes(a), out->words, ob) ||
        tr_field_overlap(b->words, tr_field_matrix_bytes(b), out->words, ob)) return TR_MATH_INVALID;
    tr_math_need need;
    zero_need(&need);
    need.need[TR_BUDGET_TENSOR] = tr_budget_add(tr_budget_add(tr_budget_mul(a->rows, a->cols), tr_budget_mul(b->rows, b->cols)),
                                                tr_budget_mul(a->rows, b->cols));
    need.need[TR_BUDGET_OUTPUT] = tr_budget_mul(a->rows, b->cols);
    rc = check_budget(budget, &need, refusal);
    if (rc) return rc;
    const uint64_t AW = tr_field_words(a->cols), BW = tr_field_words(b->cols);
    for (uint64_t i = 0; i < a->rows; i++) {
        uint64_t *o = out->words + i * out->stride;
        zero_words(o, out->stride);
        const uint64_t *ar = a->words + i * a->stride;
        for (uint64_t w = 0; w < AW; w++)
            for (uint64_t bits = ar[w]; bits; bits &= bits - 1) {      /* row i of A B: XOR of the rows k of B */
                const uint64_t *br = b->words + (w * 64 + (uint64_t)tr_field_ctz(bits)) * b->stride;
                for (uint64_t v = 0; v < BW; v++) o[v] ^= br[v];
            }
    }
    return TR_MATH_OK;
}

int32_t tr_field_add(const tr_field_matrix *a, const tr_field_matrix *b, tr_field_matrix *out) {
    int32_t rc = tr_field_check_matrix(a);
    if (!rc) rc = tr_field_check_matrix(b);
    if (rc) return rc;
    if (a->rows != b->rows || a->cols != b->cols) return TR_MATH_INVALID;
    rc = tr_field_check_output(out, a->rows, a->cols);
    if (rc) return rc;
    uint64_t ob = tr_field_matrix_bytes(out);
    if (tr_field_overlap(a->words, tr_field_matrix_bytes(a), out->words, ob) ||
        tr_field_overlap(b->words, tr_field_matrix_bytes(b), out->words, ob)) return TR_MATH_INVALID;
    const uint64_t W = tr_field_words(a->cols);
    for (uint64_t i = 0; i < a->rows; i++) {
        uint64_t *o = out->words + i * out->stride;
        zero_words(o, out->stride);
        for (uint64_t w = 0; w < W; w++) o[w] = a->words[i * a->stride + w] ^ b->words[i * b->stride + w];
    }
    return TR_MATH_OK;
}

int32_t tr_field_transpose(const tr_field_matrix *a, tr_field_matrix *out, const tr_math_budget *budget,
                           tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    int32_t rc = tr_field_check_matrix(a);
    if (!rc) rc = tr_field_check_output(out, a->cols, a->rows);
    if (rc) return rc;
    if (tr_field_overlap(a->words, tr_field_matrix_bytes(a), out->words, tr_field_matrix_bytes(out))) return TR_MATH_INVALID;
    tr_math_need need;
    zero_need(&need);
    need.need[TR_BUDGET_TENSOR] = need.need[TR_BUDGET_OUTPUT] = tr_budget_mul(a->rows, a->cols);
    rc = check_budget(budget, &need, refusal);
    if (rc) return rc;
    for (uint64_t j = 0; j < out->rows; j++) zero_words(out->words + j * out->stride, out->stride);
    const uint64_t W = tr_field_words(a->cols);
    for (uint64_t i = 0; i < a->rows; i++)
        for (uint64_t w = 0; w < W; w++)
            for (uint64_t bits = a->words[i * a->stride + w]; bits; bits &= bits - 1) {
                uint64_t j = w * 64 + (uint64_t)tr_field_ctz(bits);
                out->words[j * out->stride + i / 64] |= (uint64_t)1 << (i % 64);
            }
    return TR_MATH_OK;
}

/* a canonical vector of nbits bits: 0 or TR_MATH_INVALID */
static int32_t vector_check(const uint64_t *v, uint64_t nbits) {
    uint64_t W = tr_field_words(nbits);
    if (nbits > TR_FIELD_MAX_DIM) return TR_MATH_UNSUPPORTED;
    if (W && (!v || (v[W - 1] & ~tr_field_tail(nbits)))) return TR_MATH_INVALID;
    return TR_MATH_OK;
}

int32_t tr_field_matvec(const tr_field_matrix *a, const uint64_t *x, uint64_t *y) {
    int32_t rc = tr_field_check_matrix(a);
    if (!rc) rc = vector_check(x, a->cols);
    if (rc) return rc;
    uint64_t YW = tr_field_words(a->rows), W = tr_field_words(a->cols);
    if (YW && !y) return TR_MATH_INVALID;
    if (YW) memset(y, 0, (size_t)(YW * 8));
    for (uint64_t i = 0; i < a->rows; i++)
        if (tr_field_row_dot(a->words + i * a->stride, x, W)) y[i / 64] |= (uint64_t)1 << (i % 64);
    return TR_MATH_OK;
}

int32_t tr_field_syndromes(const tr_field_matrix *h, const tr_field_matrix *x, tr_field_matrix *out,
                           const tr_math_budget *budget, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    int32_t rc = tr_field_check_matrix(h);
    if (!rc) rc = tr_field_check_matrix(x);
    if (rc) return rc;
    if (x->cols != h->cols) return TR_MATH_INVALID;
    rc = tr_field_check_output(out, x->rows, h->rows);
    if (rc) return rc;
    uint64_t ob = tr_field_matrix_bytes(out);
    if (tr_field_overlap(h->words, tr_field_matrix_bytes(h), out->words, ob) ||
        tr_field_overlap(x->words, tr_field_matrix_bytes(x), out->words, ob)) return TR_MATH_INVALID;
    tr_math_need need;
    zero_need(&need);
    need.need[TR_BUDGET_TENSOR] = tr_budget_add(tr_budget_mul(h->rows, h->cols), tr_budget_mul(x->rows, x->cols));
    need.need[TR_BUDGET_OUTPUT] = tr_budget_mul(x->rows, h->rows);
    rc = check_budget(budget, &need, refusal);
    if (rc) return rc;
    const uint64_t O = tr_field_words(h->rows);
    for (uint64_t j = 0; j < x->rows; j++) {
        uint64_t *o = out->words + j * out->stride;
        zero_words(o, out->stride);
        for (uint64_t ow = 0; ow < O; ow++) o[ow] = tr_field_syndrome_word(h, x->words + j * x->stride, ow);
    }
    return TR_MATH_OK;
}

int32_t tr_field_row_sum(const tr_field_matrix *m, uint64_t *out) {
    int32_t rc = tr_field_check_matrix(m);
    if (rc) return rc;
    uint64_t W = tr_field_words(m->cols);
    if (W && !out) return TR_MATH_INVALID;
    if (W) memset(out, 0, (size_t)(W * 8));
    for (uint64_t i = 0; i < m->rows; i++)
        for (uint64_t w = 0; w < W; w++) out[w] ^= m->words[i * m->stride + w];
    return TR_MATH_OK;
}

int32_t tr_field_column_counts(const tr_field_matrix *m, uint64_t *counts) {
    int32_t rc = tr_field_check_matrix(m);
    if (rc) return rc;
    if (m->cols && !counts) return TR_MATH_INVALID;
    if (m->cols) memset(counts, 0, (size_t)(m->cols * 8));
    uint64_t W = tr_field_words(m->cols);
    for (uint64_t i = 0; i < m->rows; i++)
        for (uint64_t w = 0; w < W; w++)
            for (uint64_t bits = m->words[i * m->stride + w]; bits; bits &= bits - 1)
                counts[w * 64 + (uint64_t)tr_field_ctz(bits)]++;
    return TR_MATH_OK;
}

int32_t tr_field_parity(const uint64_t *v, uint64_t nbits, int32_t *bit) {
    int32_t rc = vector_check(v, nbits);
    if (rc) return rc;
    if (!bit) return TR_MATH_INVALID;
    uint64_t x = 0;
    for (uint64_t w = 0; w < tr_field_words(nbits); w++) x ^= v[w];
    *bit = tr_field_popc(x) & 1;
    return TR_MATH_OK;
}
int32_t tr_field_dot(const uint64_t *a, const uint64_t *b, uint64_t nbits, int32_t *bit) {
    int32_t rc = vector_check(a, nbits);
    if (!rc) rc = vector_check(b, nbits);
    if (rc) return rc;
    if (!bit) return TR_MATH_INVALID;
    *bit = tr_field_words(nbits) ? tr_field_row_dot(a, b, tr_field_words(nbits)) : 0;
    return TR_MATH_OK;
}
int32_t tr_field_weight(const uint64_t *v, uint64_t nbits, uint64_t *weight) {
    int32_t rc = vector_check(v, nbits);
    if (rc) return rc;
    if (!weight) return TR_MATH_INVALID;
    uint64_t n = 0;
    for (uint64_t w = 0; w < tr_field_words(nbits); w++) n += (uint64_t)tr_field_popc(v[w]);
    *weight = n;
    return TR_MATH_OK;
}

/* ------------------------------------------------------------------ bit columns over a scope */
static int bits_ok(const tr_field_bits *b) { return b && b->magic == BITS_MAGIC; }
static void bits_release(tr_field_bits *b) {
    if (!b) return;
    free(b->selection); free(b->value); free(b->valid);
    b->magic = 0;
    free(b);
}
void tr_field_bits_free(tr_field_bits *b) { if (bits_ok(b)) bits_release(b); }
static tr_field_bits *bits_fail(tr_field_bits *b, int32_t *error, int32_t code) {
    bits_release(b); set_error(error, code); return NULL;
}
static int32_t specs_check(const tr_scope_view *v, const tr_field_bitspec *s, int32_t n) {
    if (n < 0 || (n && !s)) return TR_MATH_INVALID;
    for (int32_t j = 0; j < n; j++) {
        if (s[j].reserved || s[j].col < 0 || s[j].col >= v->ncols) return TR_MATH_INVALID;
        if (s[j].kind == TR_FIELD_NONZERO) { if (s[j].bit || s[j].base) return TR_MATH_INVALID; }
        else if (s[j].kind != TR_FIELD_PLANE || s[j].bit < 0 || s[j].bit > 63) return TR_MATH_INVALID;
    }
    return TR_MATH_OK;
}
int32_t tr_field_bits_need(const tr_scope *scope, const tr_field_bitspec *specs, int32_t nvars, tr_math_need *need) {
    if (!need) return TR_MATH_INVALID;
    tr_scope_view v;
    int32_t rc = tr_scope_get_view(scope, &v);
    if (!rc) rc = specs_check(&v, specs, nvars);
    if (rc) return rc;
    zero_need(need);
    need->need[TR_BUDGET_VARIABLES] = (uint64_t)nvars;
    need->need[TR_BUDGET_BYTES] = tr_budget_add(sizeof(tr_field_bits),
                                                tr_budget_mul(tr_field_words(v.rows),
                                                              tr_budget_add(8, tr_budget_mul((uint64_t)nvars, 16))));
    return TR_MATH_OK;
}

tr_field_bits *tr_field_bits_build(const tr_scope *scope, const tr_field_bitspec *specs, int32_t nvars,
                                   const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    tr_math_need need;
    int32_t rc = tr_field_bits_need(scope, specs, nvars, &need);
    if (rc) return bits_fail(NULL, error, rc);
    rc = check_budget(budget, &need, refusal);
    if (rc) return bits_fail(NULL, error, rc);
    if (nvars > TR_FIELD_MAX_VARS) return bits_fail(NULL, error, TR_MATH_UNSUPPORTED);
    tr_scope_view v;
    tr_scope_info info;
    tr_scope_get_view(scope, &v);
    tr_scope_get_info(scope, &info);
    tr_field_bits *b = zalloc(sizeof *b);
    if (!b) return bits_fail(NULL, error, TR_MATH_MEMORY);
    b->nvars = nvars; b->rows = v.rows; b->nw = tr_field_words(v.rows); b->selected = v.selected;
    b->digest[0] = info.digest[0]; b->digest[1] = info.digest[1];
    b->selection = zalloc(b->nw * 8);
    b->value = zalloc((uint64_t)nvars * b->nw * 8);
    b->valid = zalloc((uint64_t)nvars * b->nw * 8);
    if (!b->selection || !b->value || !b->valid) return bits_fail(b, error, TR_MATH_MEMORY);
    if (b->nw) memcpy(b->selection, v.selection, (size_t)(b->nw * 8));
    int all = 1;
    for (int32_t j = 0; j < nvars; j++) {
        const tr_field_bitspec *s = &specs[j];
        for (uint64_t w = 0; w < b->nw; w++) {
            uint64_t val = 0, ok = 0;
            for (uint64_t bits = b->selection[w]; bits; bits &= bits - 1) {
                int p = tr_field_ctz(bits);
                uint64_t row = w * 64 + (uint64_t)p;
                if (!tr_scope_present(&v, s->col, row)) continue;
                ok |= (uint64_t)1 << p;
                int64_t x = tr_scope_value(&v, s->col, row);
                int bit = s->kind == TR_FIELD_NONZERO ? x != 0
                        : (int)((((uint64_t)x - (uint64_t)s->base) >> s->bit) & 1);
                if (bit) val |= (uint64_t)1 << p;
            }
            b->value[(uint64_t)j * b->nw + w] = val;
            b->valid[(uint64_t)j * b->nw + w] = ok;
            if (ok != b->selection[w]) all = 0;
        }
    }
    if (all) { free(b->valid); b->valid = NULL; }
    b->bytes = tr_budget_add(sizeof *b, tr_budget_mul(b->nw, 8 + (uint64_t)nvars * (b->valid ? 16 : 8)));
    b->magic = BITS_MAGIC;
    set_error(error, TR_MATH_OK);
    return b;
}
int32_t tr_field_bits_get_view(const tr_field_bits *b, tr_field_bits_view *v) {
    if (!bits_ok(b) || !v) return TR_MATH_INVALID;
    memset(v, 0, sizeof *v);
    v->rows = b->rows; v->nw = b->nw; v->selected = b->selected; v->nvars = b->nvars;
    v->selection = b->selection; v->value = b->value; v->valid = b->valid;
    return TR_MATH_OK;
}
int32_t tr_field_bits_get_info(const tr_field_bits *b, tr_field_bits_info *out) {
    if (!bits_ok(b) || !out) return TR_MATH_INVALID;
    memset(out, 0, sizeof *out);
    out->rows = b->rows; out->nw = b->nw; out->selected = b->selected; out->bytes = b->bytes;
    out->scope_digest[0] = b->digest[0]; out->scope_digest[1] = b->digest[1];
    out->nvars = b->nvars; out->has_valid = b->valid != NULL;
    return TR_MATH_OK;
}

int32_t tr_field_bits_combine(const tr_field_bits_view *v, const tr_field_relation *r, int32_t mode, uint64_t *value,
                              uint64_t *valid) {
    int32_t rc = tr_field_check_bits(v);
    if (!rc) rc = tr_field_check_relations(v, r, 1);
    if (rc) return rc;
    if ((mode != TR_FIELD_SQL && mode != TR_FIELD_LISTWISE) || (v->nw && !value)) return TR_MATH_INVALID;
    for (uint64_t w = 0; w < v->nw; w++) {
        uint64_t checked, violating;                         /* violating = (XOR of the variables) ^ constant */
        tr_field_relation zero = *r;
        zero.constant = 0;
        tr_field_relation_word(v, &zero, mode, w, &checked, &violating);
        value[w] = (violating ^ (r->constant ? ~(uint64_t)0 : 0)) & checked;
        if (valid) valid[w] = checked;
    }
    return TR_MATH_OK;
}

static void verify_cpu(const tr_field_bits_view *v, const tr_field_relation *r, int32_t mode, tr_field_evidence *e) {
    memset(e, 0, sizeof *e);
    e->witness = TR_FIELD_NO_ROW;
    for (uint64_t w = 0; w < v->nw; w++) {
        uint64_t checked, viol;
        tr_field_relation_word(v, r, mode, w, &checked, &viol);
        e->checked += (uint64_t)tr_field_popc(checked);
        e->excluded += (uint64_t)tr_field_popc(v->selection[w] & ~checked);
        e->violations += (uint64_t)tr_field_popc(viol);
        if (viol && e->witness == TR_FIELD_NO_ROW) e->witness = w * 64 + (uint64_t)tr_field_ctz(viol);
    }
    e->holds = !e->violations && !e->excluded;
}

int32_t tr_field_verify(const tr_field_bits_view *v, const tr_field_relation *rels, int32_t n, int32_t mode,
                        const tr_math_budget *budget, tr_field_evidence *out, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    int32_t rc = tr_field_check_bits(v);
    if (!rc) rc = tr_field_check_relations(v, rels, n);
    if (rc) return rc;
    if ((mode != TR_FIELD_SQL && mode != TR_FIELD_LISTWISE) || (n && !out)) return TR_MATH_INVALID;
    tr_math_need need;
    zero_need(&need);
    need.need[TR_BUDGET_OUTPUT] = (uint64_t)n;
    rc = check_budget(budget, &need, refusal);
    if (rc) return rc;
    for (int32_t i = 0; i < n; i++) verify_cpu(v, &rels[i], mode, &out[i]);
    return TR_MATH_OK;
}

int32_t tr_field_discover_needs(const tr_field_bits_view *v, const tr_field_discover_spec *s, int32_t device,
                                tr_math_need *need) {
    if (!need) return TR_MATH_INVALID;
    return tr_field_discover_need(v, s, device, need);
}

int32_t tr_field_propose(const tr_field_bits_view *v, const tr_field_discover_spec *s, const tr_math_budget *budget,
                         tr_field_relation *out, int32_t cap, int32_t *n, tr_field_discovery *stats,
                         tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    tr_math_need need;
    int32_t rc = tr_field_discover_need(v, s, 0, &need);
    if (rc) return rc;
    need.need[TR_BUDGET_ITERATIONS] = 1;
    need.need[TR_BUDGET_OUTPUT] = tr_field_found_capacity(tr_field_discover_columns(v, s));
    rc = check_budget(budget, &need, refusal);
    if (rc) return rc;
    const int32_t columns = tr_field_discover_columns(v, s);
    if (columns > 64) return TR_MATH_UNSUPPORTED;
    if (!n || !stats || (uint64_t)cap < tr_field_found_capacity(columns) || (cap && !out)) return TR_MATH_INVALID;
    uint64_t basis[64];
    tr_field_sample_basis(v, s, basis, stats);
    *n = tr_field_basis_relations(basis, columns, s->affine, out);
    stats->rounds = 0;
    stats->nrelations = *n;
    return TR_MATH_OK;
}

static int32_t verify_cb(void *ctx, const tr_field_relation *rels, int32_t n, tr_field_evidence *out) {
    const tr_field_bits_view *v = (const tr_field_bits_view *)ctx;
    for (int32_t i = 0; i < n; i++) verify_cpu(v, &rels[i], TR_FIELD_LISTWISE, &out[i]);
    return TR_MATH_OK;
}

int32_t tr_field_discover(const tr_field_bits_view *v, const tr_field_discover_spec *s, const tr_math_budget *budget,
                          tr_field_relation *found, tr_field_evidence *found_ev, int32_t found_cap,
                          tr_field_relation *rejected, tr_field_evidence *rejected_ev, int32_t rejected_cap,
                          tr_field_discovery *stats, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    tr_math_need need;
    int32_t rc = tr_field_discover_need(v, s, 0, &need);
    if (rc) return rc;
    rc = check_budget(budget, &need, refusal);
    if (rc) return rc;
    return tr_field_discover_run(v, s, verify_cb, (void *)v, found, found_ev, found_cap, rejected, rejected_ev,
                                 rejected_cap, stats);
}
