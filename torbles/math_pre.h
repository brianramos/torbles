/* PRE-001: affine and monotone-piece preimages: a predicate on a derived value as an exact set of stored codes.
 * SPDX-License-Identifier: Apache-2.0
 *
 * A generator column holds stored codes; its codec (tr_pre_codec) gives the logical value L = code + offset and the
 * value x = L / 10^scale, over the domain of codes [lo, hi] (an engine column: 0 .. 2^width - 1). A map y = f(x) and
 * a predicate y op c (op LT LE EQ NE GE GT, c an exact rational) or y IN {values} are transformed into the EXACT set
 * of codes whose value satisfies the predicate (tr_pre_preimage), as a union of spans:
 *   plain spans     codes lo..hi (sorted, disjoint, not adjacent)
 *   residue spans   codes of the domain with code mod M in rlo..rhi (one modulus M; sorted, disjoint)
 * with the flags EMPTY (no code) and ALL (every code of the domain), and the NULL guard: a NULL value never satisfies
 * (the scan requires the row's value to be present). Maps:
 *   AFFINE    y = a x + b, a != 0 (a = 0 is a zero divisor: TR_MATH_DOMAIN_ERROR). With T = 10^scale and t = (c - b)
 *             T / a, y op c is L op' t where op' is op flipped for a < 0 (< and >, <= and >= swap); integer rounding:
 *             L < t: L <= ceil(t) - 1, L <= t: L <= floor(t), L > t: L >= floor(t) + 1, L >= t: L >= ceil(t),
 *             L = t: t integral, L <> t: every L but an integral t. Then code = L - offset, clipped to the domain.
 *   SQUARE    y = a x^2 + b, a != 0: L^2 op' C, C = (c - b) T^2 / a: L^2 <= C: |L| <= isqrt(floor(C)) (C >= 0);
 *             L^2 < C: |L| <= isqrt(ceil(C) - 1) (C > 0); L^2 = C: zero (C not a square), one (C = 0) or two points;
 *             >=, >, <> the complements. Both branches (x <= 0, x >= 0) are always taken.
 *   RESIDUE   y = x mod m (x - m floor(x / m)) for an integer m >= 1 (m = 0: TR_MATH_DOMAIN_ERROR): with M = m T,
 *             y = (L mod M) / T, so y op c is a set of residues R = L mod M in [0, M), and code = R - offset mod M:
 *             residue spans (converted to plain spans when the domain is not larger than M).
 *   PIECES    a piecewise map: pieces with closed domains [lo, hi] of x (possibly unbounded), each AFFINE or SQUARE
 *             with a != 0; their integer code ranges must cover the domain exactly. A gap (a branch missing) is
 *             TR_MATH_DOMAIN_ERROR, an overlap (not a function) TR_MATH_INVALID: a non-invertible map is never
 *             approximated by the branches it was given. Each piece's preimage is computed in its own range and the
 *             union taken. A piece is first classified by UNC-001's exact abstract evaluation of `f(x) op c` over
 *             its code range: ALL (the whole piece) or NONE (nothing) without solving, otherwise solved exactly.
 * Exactness: every threshold is an FND-009 big rational (no overflow, no rounding); rounding to integers is floor /
 * ceil / isqrt of exact values. The set is exact on the codec's domain.
 *
 * Certificates (FND-007): one LAW (ring identity) per branch for the algebraic transformation, a polynomial in the
 * variable code: D f((code + offset) / T) expanded term by term minus its code-space form P code + Q (SQUARE:
 * P2 code^2 + P1 code + P0), D = the positive common denominator; RESIDUE: (code + offset) - (code + R0) - M k0 with
 * offset = M k0 + R0. Its canonical form is zero (decided by FND-007); the law holds for every code, in particular on
 * the codec's domain. A branch whose coefficients leave int64 has no certificate (info.certified < info.branches).
 *
 * Scanning (tr_pre_scan, the CPU reference of the kernel): over a scope column with the codec's scale, bit r is set
 * when selected row r is present and its code (logical - offset) is in the set. With direct = 1 the original
 * predicate is also evaluated on every selected row (UNC-001's exact row evaluator on the branch's program y = f(x),
 * then the comparison) and compared with the set: mismatches (none for a correct transform) and the first one,
 * rows outside the domain, rows the exact evaluation could not decide (TR_MATH_OVERFLOW when nonzero).
 *
 * Budgets (math_budgets.h, before allocation): preimage: support = pieces and IN values, output = spans, bytes = the
 * set (spans, branch programs, values, certificates); FND-009 numbers are budgeted by bytes; scan: bytes = the bitmap
 * (CUDA: the device copies). Representation limits: 256 pieces, 4096 IN values (TR_MATH_UNSUPPORTED).
 * Errors: TR_MATH_INVALID, _MEMORY, _OVERFLOW, _BUDGET, _UNSUPPORTED (a modulus m 10^scale past int64, a wide
 * column), _DOMAIN_ERROR (a zero divisor, a missing branch). Objects are immutable and individually owned.
 *
 * The CUDA library (math_pre_gpu.cu): tr_mgpu_pre_scan, one work item per 64-row word (its own bitmap word, counts by
 * atomics, the first mismatch / overflow by atomic minimum), on the set's plain view (tr_pre_set_get_view) with its
 * pointers swapped for device copies. Equal to tr_pre_scan bit for bit (the same inline functions). */
#ifndef TORBLES_MATH_PRE_H
#define TORBLES_MATH_PRE_H
#include <stdint.h>
#include "math_num.h"
#include "math_budgets.h"
#include "math_scope.h"
#include "math_unc.h"
#ifdef __cplusplus
extern "C" {
#endif

#define TR_PRE_ABI 1
#define TR_PRE_MAX_PIECES 256
#define TR_PRE_MAX_VALUES 4096

enum { TR_PRE_AFFINE = 1, TR_PRE_SQUARE = 2, TR_PRE_RESIDUE = 3, TR_PRE_PIECES = 4 };            /* maps */
enum { TR_PRE_LT = 1, TR_PRE_LE = 2, TR_PRE_EQ = 3, TR_PRE_NE = 4, TR_PRE_GE = 5, TR_PRE_GT = 6, TR_PRE_IN = 7 };
enum { TR_PRE_SET_EMPTY = 1, TR_PRE_SET_ALL = 2, TR_PRE_SET_RESIDUE = 4 };                       /* set flags */

/* ------------------------------------------------------------------ structs crossing the ctypes boundary */
typedef struct {
    int64_t offset;                  /* logical = code + offset */
    int32_t scale, nullable;         /* value = logical / 10^scale (0..18); 1: the column may hold NULL */
    int64_t lo, hi;                  /* the domain of stored codes (inclusive) */
} tr_pre_codec;                      /* 32 bytes */

typedef struct {
    int32_t kind, reserved;          /* TR_PRE_AFFINE or TR_PRE_SQUARE */
    int32_t lo_inf, hi_inf;          /* 1: unbounded below / above */
    tr_num_rational lo, hi;          /* the piece's closed domain of x */
    tr_num_rational a, b;            /* y = a x + b, or a x^2 + b */
} tr_pre_piece;                      /* 80 bytes */

typedef struct {
    int32_t kind, npieces;           /* TR_PRE_*; the pieces of a PIECES map */
    tr_num_rational a, b;            /* AFFINE / SQUARE */
    int64_t modulus, reserved;       /* RESIDUE: m */
    const tr_pre_piece *pieces;
} tr_pre_map;                        /* 64 bytes */

typedef struct {
    int32_t op, reserved;            /* TR_PRE_LT .. TR_PRE_IN */
    uint64_t nvalues;                /* IN: the values */
    tr_num_rational c;               /* the threshold (not IN) */
    const tr_num_rational *values;   /* IN (any order, duplicates allowed) */
} tr_pre_predicate;                  /* 40 bytes */

typedef struct { int64_t lo, hi, modulus, rlo, rhi; } tr_pre_span;   /* 40 bytes; modulus 1: codes lo..hi */

typedef struct {
    uint64_t nspans, codes;          /* spans; codes of the domain in the set (exact; UINT64_MAX when >= 2^64) */
    int32_t flags, nullable;         /* TR_PRE_SET_*; the NULL guard */
    int32_t branches, decided;       /* branches (pieces, or 1); branches decided by UNC-001's abstract check */
    int32_t certified, kind;         /* branches with a law certificate; the map kind */
    int64_t modulus;                 /* the residue spans' modulus (1 for plain spans) */
    tr_pre_codec codec;
} tr_pre_set_info;                   /* 80 bytes */

typedef struct {
    uint64_t rows, selected, nulls;      /* physical rows; selected rows; selected rows with a NULL value */
    uint64_t in_set, direct_true;        /* selected present rows in the set / whose predicate is TRUE (direct = 1) */
    uint64_t mismatches, first_mismatch; /* rows where the two disagree; the smallest (UINT64_MAX: none) */
    uint64_t overflow, first_overflow;   /* rows the exact direct evaluation could not decide; the smallest */
    uint64_t outside;                    /* present rows whose code lies outside the codec's domain */
} tr_pre_scan_stats;                     /* 80 bytes */

/* the plain view of a set for the scan (pointers into the set, valid while it lives; swapped for device copies) */
typedef struct {
    tr_pre_codec codec;
    int32_t op, residue;                 /* the predicate's op; 1: residue spans */
    int32_t nbranch, reserved;
    uint64_t nspans;
    const tr_pre_span *spans;
    tr_num_rational c;
    uint64_t nvalues;
    const tr_num_rational *values;       /* IN: sorted, distinct */
    const int64_t *branch_lo, *branch_hi;   /* the branches' code ranges, sorted and disjoint */
    const int32_t *node_off;             /* nbranch + 1 offsets: branch i's UNC program is nodes[off[i] .. off[i+1]) */
    const tr_unc_node *nodes;            /* each program computes y from column 0 (its root is its last node) */
    uint64_t nnodes;
} tr_pre_view;                           /* 136 bytes */

typedef struct tr_pre_set_s tr_pre_set;
struct atable_s;

/* ------------------------------------------------------------------ shared inline functions (CPU and kernels) */
TR_NUM_HD int64_t tr_pre_fmod(int64_t x, int64_t m) { int64_t r = x % m; return r < 0 ? r + m : r; }   /* m > 0 */
/* 1 when the code is in the set */
TR_NUM_HD int tr_pre_member(const tr_pre_view *v, int64_t code) {
    if (code < v->codec.lo || code > v->codec.hi || !v->nspans) return 0;
    uint64_t lo = 0, hi = v->nspans;
    if (!v->residue) {                                   /* the last span starting at or below the code */
        while (lo < hi) { uint64_t mid = lo + (hi - lo) / 2; if (v->spans[mid].lo <= code) lo = mid + 1; else hi = mid; }
        return lo > 0 && code <= v->spans[lo - 1].hi;
    }
    int64_t r = tr_pre_fmod(code, v->spans[0].modulus);
    while (lo < hi) { uint64_t mid = lo + (hi - lo) / 2; if (v->spans[mid].rlo <= r) lo = mid + 1; else hi = mid; }
    return lo > 0 && r <= v->spans[lo - 1].rhi;
}
/* the original predicate at a selected present row: TR_UNC_ROW_VALUE (*truth), _OVERFLOW, or -2 (no branch: a code
 * outside the domain) */
TR_NUM_HD int tr_pre_direct(const tr_pre_view *v, const tr_scope_view *sv, const int32_t *col, uint64_t row, int64_t code,
                            tr_unc_rowval *vals, int *truth) {
    *truth = 0;
    uint64_t lo = 0, hi = (uint64_t)v->nbranch;
    while (lo < hi) { uint64_t mid = lo + (hi - lo) / 2; if (v->branch_lo[mid] <= code) lo = mid + 1; else hi = mid; }
    if (lo == 0 || code > v->branch_hi[lo - 1]) return -2;
    uint64_t b = lo - 1;
    tr_unc_program p;
    p.nnodes = v->node_off[b + 1] - v->node_off[b]; p.ncols = 1; p.carrier = TR_UNC_EXACT; p.precision = 0;
    p.rank = TR_UNC_MAX_TERMS; p.domains = TR_UNC_DOM_ALL; p.root = p.nnodes - 1; p.reserved = 0;
    p.budget.numerator = 0; p.budget.denominator = 0; p.nodes = v->nodes + v->node_off[b];
    tr_unc_rowval y;
    int st = tr_unc_row_eval(&p, sv, col, row, vals, &y);
    if (st != TR_UNC_ROW_VALUE) return st;
    if (v->op == TR_PRE_IN) {
        uint64_t a = 0, z = v->nvalues;
        while (a < z) { uint64_t mid = a + (z - a) / 2; if (tr_q_cmp(v->values[mid], y.v) < 0) a = mid + 1; else z = mid; }
        *truth = a < v->nvalues && tr_q_cmp(v->values[a], y.v) == 0;
        return st;
    }
    int s = tr_q_cmp(y.v, v->c);
    switch (v->op) {
    case TR_PRE_LT: *truth = s < 0; break;
    case TR_PRE_LE: *truth = s <= 0; break;
    case TR_PRE_EQ: *truth = s == 0; break;
    case TR_PRE_NE: *truth = s != 0; break;
    case TR_PRE_GE: *truth = s >= 0; break;
    default: *truth = s > 0; break;
    }
    return st;
}
/* one 64-row word of the scan of scope column `col`: the bitmap word; acc[0..5] += nulls, in_set, direct_true,
 * mismatches, overflow, outside; *first_mismatch, *first_overflow = the smallest such rows */
TR_NUM_HD uint64_t tr_pre_word(const tr_pre_view *v, const tr_scope_view *sv, int32_t col, int direct, uint64_t w,
                               tr_unc_rowval *vals, uint64_t acc[6], uint64_t *first_mismatch, uint64_t *first_overflow) {
    uint64_t bits = sv->selection[w], out = 0;
    while (bits) {
        int bit = tr_ipoly_ctz(bits);
        bits &= bits - 1;
        uint64_t row = w * 64 + (uint64_t)bit;
        if (!tr_scope_present(sv, col, row)) { acc[0]++; continue; }        /* NULL never satisfies */
        int64_t logical = tr_scope_value(sv, col, row);
        int64_t code = (int64_t)((uint64_t)logical - (uint64_t)v->codec.offset);
        __int128 exact = (__int128)logical - v->codec.offset;
        int inside = exact >= v->codec.lo && exact <= v->codec.hi;
        int m = inside && tr_pre_member(v, code);
        if (!inside) acc[5]++;
        if (m) { out |= (uint64_t)1 << bit; acc[1]++; }
        if (!direct) continue;
        int truth = 0, st = inside ? tr_pre_direct(v, sv, &col, row, code, vals, &truth) : -2;
        if (st == TR_UNC_ROW_OVERFLOW) { acc[4]++; if (row < *first_overflow) *first_overflow = row; continue; }
        if (st == -2) continue;                          /* outside the domain: the set is not stated there */
        if (truth) acc[2]++;
        if (truth != m) { acc[3]++; if (row < *first_mismatch) *first_mismatch = row; }
    }
    return out;
}
/* the device copies of tr_mgpu_pre_scan */
static inline uint64_t tr_pre_scan_device_bytes(const tr_pre_view *v, const tr_scope_view *sv, int32_t col, int direct) {
    uint64_t rows = sv->rows, b = tr_budget_mul(tr_budget_bitmap_bytes(rows), 2);
    b = tr_budget_add(b, tr_budget_mul(rows, sv->col[col].values ? 8 : 4));
    if (sv->col[col].valid) b = tr_budget_add(b, rows);
    b = tr_budget_add(b, tr_budget_mul(v->nspans, sizeof(tr_pre_span)));
    if (direct) {
        b = tr_budget_add(b, tr_budget_mul(v->nvalues, sizeof(tr_num_rational)));
        b = tr_budget_add(b, tr_budget_mul((uint64_t)(v->nbranch > 0 ? v->nbranch : 0), 20));
        b = tr_budget_add(b, tr_budget_mul(v->nnodes, sizeof(tr_unc_node)));
    }
    return tr_budget_add(b, 64);
}

/* ------------------------------------------------------------------ exported (math_pre.c) */
/* the codec of stored column `col` of an engine table: offset, scale and nullable as given (the Python codec's
 * affine (offset, scale)), domain 0 .. 2^width - 1 */
int32_t tr_pre_codec_for_column(const struct atable_s *t, int32_t col, int64_t offset, int32_t scale, int32_t nullable,
                                tr_pre_codec *out);
tr_pre_set *tr_pre_preimage(const tr_pre_codec *, const tr_pre_map *, const tr_pre_predicate *, const tr_math_budget *,
                            tr_math_refusal *, int32_t *error);
int32_t tr_pre_set_get_info(const tr_pre_set *, tr_pre_set_info *);
int32_t tr_pre_set_spans(const tr_pre_set *, tr_pre_span *out, uint64_t capacity);
int32_t tr_pre_set_contains(const tr_pre_set *, int64_t code);        /* 1 / 0 (TR_MATH_INVALID: a bad handle) */
int32_t tr_pre_set_get_view(const tr_pre_set *, tr_pre_view *);
/* the law of branch i (borrowed; NULL when the branch has none) */
const tr_cert *tr_pre_set_certificate(const tr_pre_set *, int32_t i);
void tr_pre_set_free(tr_pre_set *);
/* bitmap: ceil(rows / 64) words; column: a scope column with the codec's scale; TR_MATH_OVERFLOW when a direct
 * evaluation could not be decided (stats count them) */
int32_t tr_pre_scan(const tr_pre_set *, const tr_scope *, int32_t column, int32_t direct, uint64_t *bitmap,
                    tr_pre_scan_stats *, const tr_math_budget *, tr_math_refusal *);

/* ------------------------------------------------------------------ the CUDA library (math_pre_gpu.cu) */
int tr_mgpu_pre_scan(const tr_pre_view *, const tr_scope_view *, int32_t column, int32_t direct, uint64_t *bitmap,
                     tr_pre_scan_stats *, const tr_math_budget *, tr_math_refusal *);
#ifdef __cplusplus
}
#endif
#endif
