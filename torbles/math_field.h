/* FIELD-001: packed GF(2) vectors and matrices (rank, reduced row echelon form, nullspaces, solutions, products,
 * syndromes) and exact parity dependencies over the rows of a data scope (math_scope.h), on the CPU (math_field.c) and
 * on a CUDA device (math_field_gpu.cu: batched syndromes, exhaustive verification), with the same results bit for bit.
 * SPDX-License-Identifier: Apache-2.0
 *
 * ---- Encoding ---------------------------------------------------------------------------------------------------
 * A matrix (tr_field_matrix, a view of the caller's words) has rows x cols bits, row-major: bit j of row i is bit
 * j % 64 of words[i * stride + j / 64], stride >= tr_field_words(cols) = ceil(cols / 64) words per row. Canonical:
 * every bit past cols in a row (the tail of its last word and the words up to stride) is zero; a matrix that is not
 * canonical is refused (TR_MATH_INVALID), so no stray bit can change a result. A vector is one row: ceil(n / 64)
 * words, bit j of word j / 64, the bits past n zero. rows = 0 or cols = 0 is a valid (empty) matrix. Outputs are
 * written canonical; an output must not overlap an input.
 *
 * ---- Field laws, kept apart from integer arithmetic -----------------------------------------------------------
 * GF(2) addition is XOR and multiplication AND: the sum of vectors is their XOR (tr_field_add, tr_field_row_sum), the
 * inner product <a, b> = parity(popcount(a AND b)) is a field element 0 / 1 (tr_field_dot), a product (A B)[i][j] =
 * <row i of A, column j of B>. Integer counts are a different thing and are returned as integers, never as field
 * elements: tr_field_weight(v) = popcount(v) (the Hamming weight), tr_field_column_counts = the number of rows with
 * each bit set. They agree only mod 2: parity(v) = weight(v) mod 2, row_sum bit j = column_counts[j] mod 2, and
 * weight(a XOR b) = weight(a) + weight(b) - 2 weight(a AND b) (not the sum of the weights).
 *
 * ---- Elimination ----------------------------------------------------------------------------------------------
 * Gauss-Jordan over GF(2) on packed rows, columns in increasing order: the pivot of column c is the first row at or
 * below the current row with bit c set (rows are swapped), every other row with bit c set is reduced by it (XOR from
 * the pivot's word on). The reduced row echelon form R (tr_field_rref) is unique; rank = its nonzero rows; pivots[i]
 * = the pivot column of row i < rank, increasing. Nullspace basis (tr_field_nullspace), canonical: one vector per free
 * (non-pivot) column f, increasing: bit f set, bit pivots[i] = R[i][f], every other bit 0; A v = 0 and the vectors are
 * independent (nullity = cols - rank). Solutions of A x = b (tr_field_solve): eliminate [A | b]; NONE when a pivot
 * lies in the b column (with cert: y with y^T A = 0 and y^T b = 1, read from the row operations); else x = the
 * particular solution with every free variable 0 (x[pivots[i]] = R[i][b]), UNIQUE when the nullity is 0, FAMILY
 * otherwise: every solution is x + a combination of the nullspace basis (written when basis is given).
 *
 * ---- Parity dependencies over table rows (tr_field_bits, read as a tr_field_bits_view) --------------------------
 * Bit columns of a scope: variable j is a native bit of a scope column at each selected row (tr_field_bitspec):
 * NONZERO (logical != 0: a Boolean column) or PLANE (bit `bit` 0..63 of (logical - base) mod 2^64: with base = the
 * codec offset, bit plane `bit` of the stored code). A NULL value makes the variable NULL at that row. Stored as a
 * value and a valid bitmap per variable (canonical: value within valid within the selection; valid dropped when no
 * selected value is NULL).
 * A relation (tr_field_relation) XOR_{j in vars} x_j = constant is checked on a selected row where its variables are
 * present (TR_FIELD_SQL: SQL's `a <> b <> ...` is NULL on a row with a NULL operand) or, TR_FIELD_LISTWISE, where every
 * variable of the bit matrix is; the other selected rows are excluded. Evidence (tr_field_verify, exhaustive): checked,
 * excluded, violations (checked rows where the XOR differs from the constant) and the witness, the smallest violating
 * physical row (UINT64_MAX without a violation); holds = no violation and no exclusion (an exact rewrite over the
 * selection; vacuous when nothing is selected). tr_field_bits_combine materializes XOR_{j in vars} x_j XOR constant
 * per row (the substitution of one variable by the others).
 * Discovery (tr_field_discover) finds every linear (affine: with a constant) GF(2) relation among the variables that
 * holds on every complete selected row (every variable present): the nullspace of the complete rows as vectors
 * [1, x_0, .., x_{v-1}] (affine; the constant first, so a dependent variable is expressed by earlier ones and the
 * constant) or [x_0, .., x_{v-1}]. Proposal: the canonical nullspace basis of a deterministic sample (`sample` complete
 * rows at an even stride in physical order; 0 = every complete row). Exact verification: every proposal over every
 * complete row (TR_FIELD_LISTWISE). A proposal that fails is reported (rejected, with its evidence: a one-row near miss
 * has violations 1 and its witness) and its witness row joins the sample; the witness lies outside the sample's row
 * space, so the rank grows and at most columns + 1 rounds run. The final proposals all hold, so they span exactly the
 * relations of the complete rows: `relations` is that space's canonical basis, each with its evidence (excluded > 0
 * when some selected row has a NULL: then no relation holds over the whole selection). With no complete row every
 * vector holds vacuously (affine: including the contradiction 0 = 1, vars 0, constant 1).
 *
 * ---- Budgets (math_budgets.h), checked before anything is allocated (refused: TR_MATH_BUDGET + refusal) -----------
 *   rank / rref / nullspace / solve: tensor = rows x cols of the system (solve: rows x (cols + 1), and + rows x rows
 *     with the certificate), bytes = the scratch copy and the pivots (rref: none, it reduces in its output), output =
 *     the bits written (rref: rows x cols; nullspace and the solve's basis: nullity x cols, checked after the
 *     elimination, before anything is written; solve: + cols for x, + rows for the certificate).
 *   mul: tensor = r n + n c + r c, output = r c. transpose: tensor = output = rows x cols. syndromes: tensor = the
 *     elements of H and X, output = nvec x H.rows; on the device bytes = tr_field_syndromes_device_bytes.
 *   bits: variables = nvars, bytes = the object, the selection and the value and valid words (8 nw (1 + 2 nvars)).
 *   verify: output = nrels; on the device bytes = tr_field_verify_device_bytes.
 *   discover (tr_field_discover_need): variables = columns (nvars + affine), support = the sample rows (0: the selected
 *     rows), iterations = columns + 1 (rounds), output = tr_field_found_capacity + tr_field_rejected_capacity (the
 *     capacities the caller provides), bytes = the proposals and their evidence (plus, on the device, the copies).
 * Representation limits (with a budget that admits them): TR_MATH_UNSUPPORTED for rows or cols > TR_FIELD_MAX_DIM,
 * nvars > TR_FIELD_MAX_VARS (bits) or columns > 64 (discovery).
 *
 * Errors: TR_MATH_INVALID (-1: malformed or non-canonical input, an output too small), _MEMORY (-2), _BUDGET (-5),
 * TR_MATH_DEVICE (-11), TR_MATH_UNSUPPORTED (-12). Bit matrices over scopes are immutable, individually owned
 * snapshots (constructors return NULL and set *error); their freshness is their scope's (tr_scope_fresh; the Python
 * layer checks it before every use). Matrix functions only read their inputs and write their outputs. */
#ifndef TORBLES_MATH_FIELD_H
#define TORBLES_MATH_FIELD_H
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "math_num.h"
#include "math_budgets.h"
#include "math_scope.h"
#ifdef __cplusplus
extern "C" {
#endif

#define TR_FIELD_ABI 1
#define TR_FIELD_MAX_DIM (UINT64_C(1) << 31)   /* rows and columns of a matrix */
#define TR_FIELD_MAX_VARS 64                   /* variables of a bit matrix over rows (relations are uint64 masks) */
#define TR_FIELD_NO_ROW UINT64_MAX
enum { TR_FIELD_NONE = 0, TR_FIELD_UNIQUE = 1, TR_FIELD_FAMILY = 2 };   /* solve status */
enum { TR_FIELD_NONZERO = 1, TR_FIELD_PLANE = 2 };                       /* bit specs */
enum { TR_FIELD_SQL = 0, TR_FIELD_LISTWISE = 1 };                        /* verification domains */
enum { TR_FIELD_NEED_CPU = 0, TR_FIELD_NEED_DEVICE = 1 };

typedef struct { uint64_t rows, cols, stride; uint64_t *words; } tr_field_matrix;          /* 32 bytes */
typedef struct { int32_t col, kind, bit, reserved; int64_t base; } tr_field_bitspec;        /* 24 bytes */
typedef struct {
    uint64_t rows, nw, selected;
    int32_t nvars, reserved;
    const uint64_t *selection;    /* nw words */
    const uint64_t *value;        /* nvars * nw words, variable-major (value[j * nw + w]) */
    const uint64_t *valid;        /* nvars * nw words, or NULL: no selected value is NULL */
} tr_field_bits_view;             /* 56 bytes */
typedef struct { uint64_t vars; int32_t constant, reserved; } tr_field_relation;            /* 16 bytes */
typedef struct {
    uint64_t checked, excluded, violations, witness;
    int32_t holds, reserved;
} tr_field_evidence;              /* 40 bytes */
typedef struct { uint64_t sample; int32_t affine, reserved; } tr_field_discover_spec;       /* 16 bytes */
typedef struct {
    uint64_t complete, excluded;  /* selected rows with every variable present / with a NULL among them */
    uint64_t initial_sample;      /* complete rows in the proposal sample */
    uint64_t sample_rows;         /* rows in the final sample (the initial sample and the witnesses added) */
    uint64_t rank;                /* rank of the complete rows as vectors (with the constant column when affine) */
    uint64_t row_checks;          /* proposals verified x complete rows */
    int32_t rounds, nrelations, nrejected, columns;   /* columns = nvars + affine */
} tr_field_discovery;             /* 64 bytes */
typedef struct {
    uint64_t rows, nw, selected, bytes;
    uint64_t scope_digest[2];
    int32_t nvars, has_valid;
} tr_field_bits_info;             /* 56 bytes */

typedef struct tr_field_bits_s tr_field_bits;

/* ------------------------------------------------------------------ shared helpers (CPU code, CUDA host and kernels) */
TR_NUM_HD uint64_t tr_field_words(uint64_t bits) { return (bits + 63) / 64; }
TR_NUM_HD int tr_field_popc(uint64_t x) {
#if defined(__CUDA_ARCH__)
    return __popcll((unsigned long long)x);
#else
    return __builtin_popcountll(x);
#endif
}
TR_NUM_HD int tr_field_ctz(uint64_t x) {   /* x != 0 */
#if defined(__CUDA_ARCH__)
    return __ffsll((long long)x) - 1;
#else
    return __builtin_ctzll(x);
#endif
}
TR_NUM_HD uint64_t tr_field_tail(uint64_t bits) {   /* the used bits of a row's last word */
    uint64_t r = bits % 64;
    return r ? ((uint64_t)1 << r) - 1 : ~(uint64_t)0;
}
/* <a, b> over words words: the parity of popcount(a AND b) */
TR_NUM_HD int tr_field_row_dot(const uint64_t *a, const uint64_t *b, uint64_t words) {
    uint64_t x = 0;
    for (uint64_t w = 0; w < words; w++) x ^= a[w] & b[w];
    return tr_field_popc(x) & 1;
}
/* the 64 syndrome bits of H rows 64 ow .. 64 ow + 63 for the vector x */
TR_NUM_HD uint64_t tr_field_syndrome_word(const tr_field_matrix *h, const uint64_t *x, uint64_t ow) {
    uint64_t out = 0, first = ow * 64, words = tr_field_words(h->cols);
    for (uint64_t i = first; i < h->rows && i < first + 64; i++)
        out |= (uint64_t)tr_field_row_dot(h->words + i * h->stride, x, words) << (i - first);
    return out;
}
/* relation r at word w: *checked = the rows it is checked on, *violating = those where the XOR differs */
TR_NUM_HD void tr_field_relation_word(const tr_field_bits_view *v, const tr_field_relation *r, int32_t mode,
                                      uint64_t w, uint64_t *checked, uint64_t *violating) {
    uint64_t ok = v->selection[w], x = r->constant ? ~(uint64_t)0 : 0;
    for (int32_t j = 0; j < v->nvars; j++) {
        int in = (int)((r->vars >> j) & 1);
        uint64_t at = (uint64_t)j * v->nw + w;
        if (in) x ^= v->value[at];
        if (v->valid && (in || mode == TR_FIELD_LISTWISE)) ok &= v->valid[at];
    }
    *checked = ok; *violating = x & ok;
}

/* 0 when the view is a canonical matrix within the limits, else TR_MATH_INVALID (-1) / TR_MATH_UNSUPPORTED (-12) */
static inline int32_t tr_field_check_matrix(const tr_field_matrix *m) {
    if (!m) return -1;
    if (m->rows > TR_FIELD_MAX_DIM || m->cols > TR_FIELD_MAX_DIM) return -12;
    uint64_t W = tr_field_words(m->cols);
    if (m->stride < W || (m->rows && m->stride && !m->words) || (m->rows && m->stride > (UINT64_MAX / 8) / m->rows))
        return -1;
    if (!W && m->stride == 0) return 0;
    for (uint64_t i = 0; i < m->rows; i++) {
        const uint64_t *row = m->words + i * m->stride;
        if (W && (row[W - 1] & ~tr_field_tail(m->cols))) return -1;
        for (uint64_t w = W; w < m->stride; w++) if (row[w]) return -1;
    }
    return 0;
}
/* an output view of exactly rows x cols (contents overwritten): 0 or TR_MATH_INVALID */
static inline int32_t tr_field_check_output(const tr_field_matrix *m, uint64_t rows, uint64_t cols) {
    if (!m || m->rows != rows || m->cols != cols || m->stride < tr_field_words(cols) ||
        (rows && m->stride && !m->words) || (rows && m->stride > (UINT64_MAX / 8) / rows)) return -1;
    return 0;
}
static inline int tr_field_overlap(const void *a, uint64_t abytes, const void *b, uint64_t bbytes) {
    uintptr_t x = (uintptr_t)a, y = (uintptr_t)b;
    return abytes && bbytes && x < y + bbytes && y < x + abytes;
}
static inline uint64_t tr_field_matrix_bytes(const tr_field_matrix *m) {
    return tr_budget_mul(tr_budget_mul(m->rows, m->stride), 8);
}
/* 0 when the bits view is consistent, else TR_MATH_INVALID / TR_MATH_UNSUPPORTED */
static inline int32_t tr_field_check_bits(const tr_field_bits_view *v) {
    if (!v || v->nvars < 0 || v->nw != tr_field_words(v->rows) || (v->nw && !v->selection) ||
        (v->nw && v->nvars && !v->value)) return -1;
    if (v->nvars > TR_FIELD_MAX_VARS) return -12;
    return 0;
}
static inline int32_t tr_field_check_relations(const tr_field_bits_view *v, const tr_field_relation *r, int32_t n) {
    if (n < 0 || (n && !r)) return -1;
    for (int32_t i = 0; i < n; i++)
        if ((v->nvars < 64 && (r[i].vars >> v->nvars)) || (r[i].constant != 0 && r[i].constant != 1) || r[i].reserved)
            return -1;
    return 0;
}

/* ------------------------------------------------------------------ device needs (shared with the host bindings) */
static inline uint64_t tr_field_syndromes_device_bytes(const tr_field_matrix *h, const tr_field_matrix *x) {
    uint64_t W = tr_field_words(h->cols), O = tr_field_words(h->rows);
    return tr_budget_add(tr_budget_add(tr_budget_mul(tr_budget_mul(h->rows, W), 8), tr_budget_mul(tr_budget_mul(x->rows, W), 8)),
                         tr_budget_mul(tr_budget_mul(x->rows, O), 8));
}
static inline uint64_t tr_field_verify_device_bytes(const tr_field_bits_view *v, int32_t nrels) {
    uint64_t n = (uint64_t)(nrels > 0 ? nrels : 0), words = tr_budget_mul((uint64_t)v->nvars, v->nw);
    uint64_t bytes = tr_budget_add(tr_budget_mul(v->nw, 8), tr_budget_mul(words, v->valid ? 16 : 8));
    return tr_budget_add(bytes, tr_budget_mul(n, sizeof(tr_field_relation) + 32));   /* relations, 4 accumulators */
}

/* ------------------------------------------------------------------ discovery (one algorithm for both libraries) */
/* the capacities a discovery needs: relations <= columns; rejected proposals <= columns (columns + 1) / 2 (each
 * failing round has a nullity >= 1, the ranks of the failing rounds are distinct) */
TR_NUM_HD uint64_t tr_field_found_capacity(int32_t columns) { return (uint64_t)(columns > 0 ? columns : 0); }
TR_NUM_HD uint64_t tr_field_rejected_capacity(int32_t columns) {
    uint64_t c = (uint64_t)(columns > 0 ? columns : 0);
    return c * (c + 1) / 2;
}
static inline int32_t tr_field_discover_columns(const tr_field_bits_view *v, const tr_field_discover_spec *s) {
    return v->nvars + (s->affine ? 1 : 0);
}
/* the needs of a discovery (see Budgets); device: add the copies of the bits and the proposals' accumulators */
static inline int32_t tr_field_discover_need(const tr_field_bits_view *v, const tr_field_discover_spec *s, int32_t device,
                                             tr_math_need *need) {
    int32_t rc = tr_field_check_bits(v);
    if (rc) return rc;
    if (!s || (s->affine != 0 && s->affine != 1) || s->reserved) return -1;
    int32_t c = tr_field_discover_columns(v, s);
    for (int i = 0; i < TR_BUDGET_RESOURCES; i++) need->need[i] = 0;
    need->need[TR_BUDGET_VARIABLES] = (uint64_t)c;
    need->need[TR_BUDGET_SUPPORT] = s->sample ? s->sample : v->selected;
    need->need[TR_BUDGET_ITERATIONS] = (uint64_t)c + 1;
    need->need[TR_BUDGET_OUTPUT] = tr_field_found_capacity(c) + tr_field_rejected_capacity(c);
    uint64_t bytes = tr_budget_mul((uint64_t)(c > 0 ? c : 1), sizeof(tr_field_relation) + sizeof(tr_field_evidence));
    if (device) bytes = tr_budget_add(bytes, tr_field_verify_device_bytes(v, c > 0 ? c : 1));
    need->need[TR_BUDGET_BYTES] = bytes;
    return 0;
}
/* the vector of physical row r ([1, x_0, ..] affine, else [x_0, ..]) */
static inline uint64_t tr_field_row_vector(const tr_field_bits_view *v, uint64_t r, int32_t affine) {
    uint64_t vec = affine ? 1 : 0;
    for (int32_t j = 0; j < v->nvars; j++)
        if ((v->value[(uint64_t)j * v->nw + r / 64] >> (r % 64)) & 1) vec |= (uint64_t)1 << (j + affine);
    return vec;
}
/* the complete rows of word w: selected with every variable present */
static inline uint64_t tr_field_complete_word(const tr_field_bits_view *v, uint64_t w) {
    uint64_t ok = v->selection[w];
    if (v->valid) for (int32_t j = 0; j < v->nvars; j++) ok &= v->valid[(uint64_t)j * v->nw + w];
    return ok;
}
/* echelon basis of single-word vectors, slot p holding the vector whose lowest set bit is p: 1 when vec raised the
 * rank (it is stored), 0 when it lay in the span */
static inline int tr_field_basis_insert(uint64_t *basis, uint64_t vec) {
    for (int p = 0; p < 64 && vec; p++)
        if (((vec >> p) & 1) && basis[p]) vec ^= basis[p];
    if (!vec) return 0;
    basis[tr_field_ctz(vec)] = vec;
    return 1;
}
/* the canonical nullspace basis of the span of the basis vectors in `columns` columns, as relations (see Discovery);
 * returns their number */
static inline int32_t tr_field_basis_relations(const uint64_t *basis, int32_t columns, int32_t affine,
                                               tr_field_relation *out) {
    uint64_t R[64];
    for (int p = 0; p < 64; p++) R[p] = basis[p];
    for (int p = 0; p < 64; p++) {                  /* reduce to RREF: clear pivot column p from earlier pivots' rows */
        if (!R[p]) continue;
        for (int q = 0; q < p; q++) if (R[q] && ((R[q] >> p) & 1)) R[q] ^= R[p];
    }
    int32_t n = 0;
    for (int f = 0; f < columns; f++) {
        if (R[f]) continue;                         /* a pivot column */
        uint64_t a = (uint64_t)1 << f;
        for (int p = 0; p < f; p++) if (R[p] && ((R[p] >> f) & 1)) a |= (uint64_t)1 << p;
        out[n].vars = affine ? a >> 1 : a;
        out[n].constant = affine ? (int32_t)(a & 1) : 0;
        out[n].reserved = 0;
        n++;
    }
    return n;
}
/* the proposal sample: the complete rows (stats: complete, excluded) and the echelon basis of the sample rows (stats:
 * initial_sample = sample_rows = the rows taken); columns <= 64 */
static inline void tr_field_sample_basis(const tr_field_bits_view *v, const tr_field_discover_spec *s, uint64_t *basis,
                                         tr_field_discovery *stats) {
    const int32_t affine = s->affine;
    memset(stats, 0, sizeof *stats);
    stats->columns = tr_field_discover_columns(v, s);
    uint64_t complete = 0;
    for (uint64_t w = 0; w < v->nw; w++) complete += (uint64_t)tr_field_popc(tr_field_complete_word(v, w));
    stats->complete = complete;
    stats->excluded = v->selected - complete;
    for (int p = 0; p < 64; p++) basis[p] = 0;
    uint64_t want = s->sample && s->sample < complete ? s->sample : complete;
    uint64_t stride = want ? complete / want : 1, seen = 0, taken = 0;
    for (uint64_t w = 0; w < v->nw && taken < want; w++) {      /* complete rows at an even stride, in row order */
        uint64_t bits = tr_field_complete_word(v, w);
        while (bits && taken < want) {
            uint64_t r = w * 64 + (uint64_t)tr_field_ctz(bits);
            bits &= bits - 1;
            if (seen++ % stride) continue;
            tr_field_basis_insert(basis, tr_field_row_vector(v, r, affine));
            taken++;
        }
    }
    stats->initial_sample = stats->sample_rows = taken;
    int32_t rank = 0;
    for (int p = 0; p < 64; p++) rank += basis[p] != 0;
    stats->rank = (uint64_t)rank;
}
typedef int32_t (*tr_field_verify_fn)(void *ctx, const tr_field_relation *rels, int32_t n, tr_field_evidence *out);
/* The discovery loop (see Discovery) with the exhaustive verification `verify` (TR_FIELD_LISTWISE over the view's
 * rows): found[] / found_ev[] (capacity >= tr_field_found_capacity), rejected[] / rejected_ev[] (capacity >=
 * tr_field_rejected_capacity), stats. The caller has checked the budget (tr_field_discover_need). */
static inline int32_t tr_field_discover_run(const tr_field_bits_view *v, const tr_field_discover_spec *s,
                                            tr_field_verify_fn verify, void *ctx, tr_field_relation *found,
                                            tr_field_evidence *found_ev, int32_t found_cap, tr_field_relation *rejected,
                                            tr_field_evidence *rejected_ev, int32_t rejected_cap,
                                            tr_field_discovery *stats) {
    const int32_t affine = s->affine, columns = tr_field_discover_columns(v, s);
    if (columns > 64) return -12;
    if (!stats || (uint64_t)found_cap < tr_field_found_capacity(columns) || (found_cap && (!found || !found_ev)) ||
        (uint64_t)rejected_cap < tr_field_rejected_capacity(columns) || (rejected_cap && (!rejected || !rejected_ev)))
        return -1;
    uint64_t basis[64];
    tr_field_sample_basis(v, s, basis, stats);
    const uint64_t complete = stats->complete;
    tr_field_relation cand[64];
    tr_field_evidence ev[64];
    int32_t nrej = 0;
    for (int32_t round = 0; round <= columns; round++) {
        int32_t n = tr_field_basis_relations(basis, columns, affine, cand);
        stats->rounds = round + 1;
        if (n) {
            int32_t rc = verify(ctx, cand, n, ev);
            if (rc) return rc;
            stats->row_checks += (uint64_t)n * complete;
        }
        int failed = 0;
        for (int32_t i = 0; i < n; i++) {
            if (!ev[i].violations) continue;
            if (ev[i].witness >= v->rows) return -1;              /* never from a correct verifier */
            failed = 1;
            if (nrej < rejected_cap) { rejected[nrej] = cand[i]; rejected_ev[nrej] = ev[i]; }
            nrej++;
            if (tr_field_basis_insert(basis, tr_field_row_vector(v, ev[i].witness, affine))) stats->sample_rows++;
        }
        if (!failed) {
            for (int32_t i = 0; i < n; i++) { found[i] = cand[i]; found_ev[i] = ev[i]; }
            stats->nrelations = n;
            stats->nrejected = nrej;
            int32_t rank = 0;
            for (int p = 0; p < 64; p++) rank += basis[p] != 0;
            stats->rank = (uint64_t)rank;
            return nrej <= rejected_cap ? 0 : -1;               /* never: the capacity bounds every run */
        }
    }
    return -1;                                                  /* never: the rank grows in every failing round */
}

/* ------------------------------------------------------------------ matrices and vectors (libalg) */
int32_t tr_field_check(const tr_field_matrix *);
int32_t tr_field_rank(const tr_field_matrix *, const tr_math_budget *, uint64_t *rank, tr_math_refusal *);
/* out: rows x cols (overwritten with R); pivots: min(rows, cols) entries, the first rank written */
int32_t tr_field_rref(const tr_field_matrix *, tr_field_matrix *out, uint64_t *pivots, uint64_t *rank,
                      const tr_math_budget *, tr_math_refusal *);
/* basis: capacity rows >= cols, cols columns; rows 0..nullity-1 written, the others zeroed */
int32_t tr_field_nullspace(const tr_field_matrix *, tr_field_matrix *basis, uint64_t *nullity, const tr_math_budget *,
                           tr_math_refusal *);
/* b: the right side (rows bits); x: cols bits (zero for NONE); basis: NULL or as for tr_field_nullspace (written for
 * FAMILY and UNIQUE); cert: NULL or rows bits (written for NONE); *status, *rank (of A) */
int32_t tr_field_solve(const tr_field_matrix *, const uint64_t *b, uint64_t *x, tr_field_matrix *basis, uint64_t *cert,
                       int32_t *status, uint64_t *rank, const tr_math_budget *, tr_math_refusal *);
int32_t tr_field_mul(const tr_field_matrix *a, const tr_field_matrix *b, tr_field_matrix *out, const tr_math_budget *,
                     tr_math_refusal *);
int32_t tr_field_add(const tr_field_matrix *a, const tr_field_matrix *b, tr_field_matrix *out);
int32_t tr_field_transpose(const tr_field_matrix *, tr_field_matrix *out, const tr_math_budget *, tr_math_refusal *);
/* y = A x (y: rows bits) */
int32_t tr_field_matvec(const tr_field_matrix *, const uint64_t *x, uint64_t *y);
/* out row j = H x_j for every row x_j of X (X: nvec x H.cols, out: nvec x H.rows) */
int32_t tr_field_syndromes(const tr_field_matrix *h, const tr_field_matrix *x, tr_field_matrix *out,
                           const tr_math_budget *, tr_math_refusal *);
/* the GF(2) sum of the rows (cols bits) / the integer number of rows with each bit set (cols counts) */
int32_t tr_field_row_sum(const tr_field_matrix *, uint64_t *out);
int32_t tr_field_column_counts(const tr_field_matrix *, uint64_t *counts);
/* field elements 0 / 1 (parity, inner product) and the integer Hamming weight of an nbits vector (canonical) */
int32_t tr_field_parity(const uint64_t *v, uint64_t nbits, int32_t *bit);
int32_t tr_field_dot(const uint64_t *a, const uint64_t *b, uint64_t nbits, int32_t *bit);
int32_t tr_field_weight(const uint64_t *v, uint64_t nbits, uint64_t *weight);

/* ------------------------------------------------------------------ bit columns over a scope (libalg) */
tr_field_bits *tr_field_bits_build(const tr_scope *, const tr_field_bitspec *, int32_t nvars, const tr_math_budget *,
                                   tr_math_refusal *, int32_t *error);
int32_t tr_field_bits_need(const tr_scope *, const tr_field_bitspec *, int32_t nvars, tr_math_need *);
int32_t tr_field_bits_get_view(const tr_field_bits *, tr_field_bits_view *);
int32_t tr_field_bits_get_info(const tr_field_bits *, tr_field_bits_info *);
void tr_field_bits_free(tr_field_bits *);
/* value[nw] = XOR_{j in vars} x_j XOR constant on the rows where the relation is checked (mode), valid[nw] = those
 * rows (NULL allowed) */
int32_t tr_field_bits_combine(const tr_field_bits_view *, const tr_field_relation *, int32_t mode, uint64_t *value,
                              uint64_t *valid);
int32_t tr_field_verify(const tr_field_bits_view *, const tr_field_relation *, int32_t nrels, int32_t mode,
                        const tr_math_budget *, tr_field_evidence *out, tr_math_refusal *);
int32_t tr_field_discover(const tr_field_bits_view *, const tr_field_discover_spec *, const tr_math_budget *,
                          tr_field_relation *found, tr_field_evidence *found_ev, int32_t found_cap,
                          tr_field_relation *rejected, tr_field_evidence *rejected_ev, int32_t rejected_cap,
                          tr_field_discovery *stats, tr_math_refusal *);
/* The proposals alone (candidate generation, no verification): the canonical nullspace basis of the sample rows of
 * tr_field_discover's first round; out capacity >= tr_field_found_capacity(columns); *n proposals; stats as after the
 * sample (complete, excluded, initial_sample, rank of the sample). Budget: the needs of tr_field_discover but
 * iterations 1. */
int32_t tr_field_propose(const tr_field_bits_view *, const tr_field_discover_spec *, const tr_math_budget *,
                         tr_field_relation *out, int32_t cap, int32_t *n, tr_field_discovery *stats, tr_math_refusal *);
/* the needs of tr_field_discover (device 0) / tr_mgpu_field_discover (device 1) */
int32_t tr_field_discover_needs(const tr_field_bits_view *, const tr_field_discover_spec *, int32_t device,
                                tr_math_need *);

/* ------------------------------------------------------------------ the CUDA library (math_field_gpu.cu, libmathgpu)
 * Views of host memory, copied to the device and budgeted (bytes) before anything is allocated:
 *   tr_mgpu_field_syndromes  tr_field_syndromes; one work item per (vector, 64-bit syndrome word), one output slot each
 *   tr_mgpu_field_verify     tr_field_verify; work items (relation, word slice) XOR the relation's words, count with
 *                            popcounts and add their checked / excluded / violation counts and the smallest violating
 *                            row (atomic min) once at the end
 *   tr_mgpu_field_discover   tr_field_discover with the device verification (the bits copied once) */
int tr_mgpu_field_syndromes(const tr_field_matrix *h, const tr_field_matrix *x, tr_field_matrix *out,
                            const tr_math_budget *, tr_math_refusal *);
int tr_mgpu_field_verify(const tr_field_bits_view *, const tr_field_relation *, int32_t nrels, int32_t mode,
                         const tr_math_budget *, tr_field_evidence *out, tr_math_refusal *);
int tr_mgpu_field_discover(const tr_field_bits_view *, const tr_field_discover_spec *, const tr_math_budget *,
                           tr_field_relation *found, tr_field_evidence *found_ev, int32_t found_cap,
                           tr_field_relation *rejected, tr_field_evidence *rejected_ev, int32_t rejected_cap,
                           tr_field_discovery *stats, tr_math_refusal *);
#ifdef __cplusplus
}
#endif
#endif
