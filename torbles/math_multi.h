/* MULTI-001: hierarchies of group keys over a data scope (math_scope.h): certified refinement maps, per-level exact
 * moment banks (math_bank.h) built once at the finest level and rolled up exactly, localized updates, and exact
 * Haar-like contrasts between a child group and its parent or its next sibling.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Levels. A level is a FND-004 scope (its dense groups: the mixed-radix number of its key codes) together with the
 * logical codec descriptor of its keys (an opaque byte string the caller derives from the codecs that turn key codes
 * into logical key values: coltypes kind, offset, scale, dictionary; empty for array scopes). Every level of a
 * hierarchy describes the same rows: the same source (table address, lifetime token, version digest of the revision
 * and sidecars, row count) and exactly the same selection bits (the "rows key"); levels may differ in their value
 * columns. Level 0 is the finest; 2..TR_MULTI_MAX_LEVELS levels.
 *
 * Refinement maps. A map sends each of the nfine groups of a fine level to one of the ncoarse groups of a coarse level
 * (TR_MULTI_UNMAPPED: no parent). It is bound to the grouping of both levels: the grouping descriptor of each scope
 * (tr_scope_key_write part TR_SCOPE_KEY_GROUPS: key specs, radices, NULL codes and lookup contents, or the ids of an
 * array scope) followed by the level's codec descriptor; and, when proven, to the rows key it was proven on. A map is
 * applied only where every binding is equal byte for byte, so a map built for another scope, another grouping, other
 * rows or another codec is refused (TR_MATH_DOMAIN_ERROR), while the same logical keys under the same codecs are
 * accepted (e.g. the same declared map for two tables whose key codecs are equal).
 *   PROVEN    tr_multi_map_certify: exhaustive over the selected rows of two levels of the same rows: the map is
 *             functional (every selected row of a fine group has the same coarse group). map[f] is that coarse group
 *             for every fine group with selected rows, TR_MULTI_UNMAPPED for the others. A violation refuses the map
 *             (TR_MATH_DOMAIN_ERROR) with a witness: the smallest selected row r whose coarse group differs from the
 *             coarse group of the smallest selected row of its fine group (row, first_row, the fine group, both
 *             coarse groups). The CUDA path (tr_mgpu_multi_certify) finds the same witness and the same map.
 *   DECLARED  tr_multi_map_declare: a map given by the caller (by the schema), bound to both groupings but not proven
 *             on any rows; tr_multi_map_verify checks it exhaustively against the data of two levels and either
 *             returns it PROVEN on those rows (entries of fine groups without selected rows keep their declared
 *             value) or refuses with the smallest selected row r where map[fine(r)] != coarse(r) (an unmapped present
 *             group included). tr_multi_map_compose chains two maps (a's coarse binding equal to b's fine binding):
 *             PROVEN when both are proven on the same rows, else DECLARED.
 *
 * Hierarchies. tr_multi_hier_build takes the levels, the bank of level 0 (tr_bank_build of levels[0].scope: scoped,
 * same descriptor digest, rows and groups) and one map per transition (maps[i]: levels[i] -> levels[i+1], bindings
 * equal; a proven map must be proven on the hierarchy's rows key). Level i + 1's states are the exact rollup of level
 * i's along maps[i] (tr_bank_rollup: counts and sums add; a coarse group is present iff one of its children is; an
 * all-NULL group is present with count 0 and sum 0; an absent group has rows 0, at every level). A present group
 * without a parent is refused (TR_MATH_DOMAIN_ERROR, *bad_group set). Status: PROVEN when every map is proven on the
 * hierarchy's rows (then each level's states equal the bank built directly on that level's scope, and a level whose
 * scope has the same rows and value columns as level 0 is scoped to it: cacheable in a FND-004 cache), DECLARED when
 * some map is declared (the states are exact rollups along the declared maps; that they equal the levels' data
 * groupings is not proven; such levels are unscoped).
 *   tr_multi_hier_update: the hierarchy of a new level-0 bank (same basis, mode, columns and groups), recomputing
 *   only the coarse cells it must. The dirty children of a transition are the changed groups of the finer level (any
 *   state word differs) and the children whose parent changed (new maps given); each coarse cell touched is its old
 *   value minus the old states of its dirty children plus their new states, in exact 192-bit arithmetic and then
 *   checked to fit the bank's carriers (counts in [0, 2^64), |sum| <= 2^127 - 1). When the dirty children are more
 *   than half of a level (adaptive), the level is rolled up completely instead. Without new levels and maps the
 *   maps are carried: status CARRIED (the states are exact rollups along maps proven on an earlier version of the
 *   data; certify on the new data to prove them); with them, the result is the hierarchy of the new levels.
 *   Work counters: cells visited (groups x (terms + 1)) per level, the strategy chosen per level.
 *
 * Contrasts (exact rationals in logical units: term j carries the decimal scale sum_c e[j][c] * scale_c). For a child
 * level l < levels - 1, a child group c (parent p = map[c]) and a term j, with n = count (rows where term j is
 * valid) and s = sum of term j over them:
 *   TR_MULTI_MEAN           s_c / n_c - s_p / n_p        (UNDEFINED when n_c = 0: a mean over no valid row)
 *   TR_MULTI_TOTAL          s_c - s_p
 *   TR_MULTI_SHARE          s_c - s_p * n_c / n_p        (= n_c (mean_c - mean_p): the child's excess over its share
 *                                                         of the parent's total; it sums to 0 over the children of
 *                                                         p; UNDEFINED when n_p = 0)
 *   TR_MULTI_SIBLING_MEAN   s_c / n_c - s_b / n_b        (b: the next present child of p after c in id order;
 *   TR_MULTI_SIBLING_TOTAL  s_c - s_b                     NO_SIBLING when there is none; UNDEFINED as above)
 * An absent child (rows 0) has status ABSENT and no value. Values are canonical (gcd(|num|, den) = 1, den >= 1),
 * numerators up to 2^193 (256-bit two's complement), denominators up to 2^128; never a division by zero. The CPU loop
 * and the CUDA kernel (tr_mgpu_multi_contrasts, one work item per child group and term) evaluate the same inline
 * function, so their results are equal bit for bit.
 *
 * Budgets (math_budgets.h), checked before allocation: certify / declare / verify: states = fine groups, bytes = the
 * map, its bindings, the scan scratch and the object (CUDA: the device copies instead, tr_multi_certify_device_need);
 * build / update: support = terms, states = groups x terms of the coarse levels, bytes = every bank, map copy,
 * scratch and object; contrasts: output = child groups x terms requested, bytes = the native scratch (CUDA: the
 * device copies). Refusal: TR_MATH_BUDGET with the tr_math_refusal. More than TR_MULTI_MAX_LEVELS levels:
 * TR_MATH_UNSUPPORTED.
 *
 * Errors: TR_MATH_INVALID, _MEMORY, _OVERFLOW (an update whose cells do not fit), _BUDGET, _UNSUPPORTED,
 * TR_MATH_DOMAIN_ERROR (not a refinement: witness; a binding differs; a present group without a parent). Objects are
 * immutable and individually owned; constructors return NULL and set *error, inputs unchanged. */
#ifndef TORBLES_MATH_MULTI_H
#define TORBLES_MATH_MULTI_H
#include <stdint.h>
#include <string.h>
#include "math_num.h"
#include "math_budgets.h"
#include "math_scope.h"
#include "math_bank.h"
#ifdef __cplusplus
extern "C" {
#endif

#define TR_MULTI_MAX_LEVELS 8
#define TR_MULTI_UNMAPPED UINT32_MAX
#define TR_MULTI_NONE UINT64_MAX
enum { TR_MULTI_PROVEN = 1, TR_MULTI_DECLARED = 2, TR_MULTI_CARRIED = 3 };          /* map and hierarchy status */
enum { TR_MULTI_MEAN = 1, TR_MULTI_TOTAL = 2, TR_MULTI_SHARE = 3, TR_MULTI_SIBLING_MEAN = 4,
       TR_MULTI_SIBLING_TOTAL = 5 };                                                  /* contrast kinds */
enum { TR_MULTI_DEFINED = 0, TR_MULTI_ABSENT = 1, TR_MULTI_UNDEFINED = 2, TR_MULTI_NO_SIBLING = 3 };   /* cells */
enum { TR_MULTI_ROLLUP = 1, TR_MULTI_LOCAL = 2, TR_MULTI_COPY = 3 };   /* per-level strategy of a build / update */

/* ------------------------------------------------------------------ structs crossing the ctypes boundary */
/* a level: a scope and the logical codec descriptor of its keys (codec may be NULL with codec_len 0) */
typedef struct {
    const tr_scope *scope;
    const uint8_t *codec;
    uint64_t codec_len;
} tr_multi_level;                 /* 24 bytes */

/* the refusal of a refinement (found = 1), see the header comment */
typedef struct {
    uint64_t row, first_row;      /* TR_MULTI_NONE when not applicable */
    uint32_t fine, coarse;        /* the fine group of row; its coarse group */
    uint32_t first_coarse;        /* the coarse group of first_row (or the map's value: tr_multi_map_verify) */
    int32_t found;
} tr_multi_witness;               /* 32 bytes */

typedef struct {
    uint64_t nfine, ncoarse;
    uint64_t mapped;              /* fine groups with a parent */
    uint64_t checked;             /* selected rows the proof visited (0 for a declared map) */
    uint64_t bytes;               /* bytes the map owns */
    uint64_t digest[2];           /* of the bindings and the entries (a pre-filter only) */
    int32_t status;               /* TR_MULTI_PROVEN / TR_MULTI_DECLARED */
    int32_t rows_bound;           /* 1: bound to the rows key it was proven on */
} tr_multi_map_info;              /* 64 bytes */

typedef struct {
    int32_t levels, status;       /* TR_MULTI_PROVEN / DECLARED / CARRIED */
    int32_t ncols, nterms, mode, reserved;
    uint64_t bytes;               /* retained: every level's bank, the maps and the object */
    uint64_t rows;                /* selected rows (equal at every level) */
    uint64_t work;                /* cells visited to make the coarse levels (build) or to update them */
    uint64_t changed;             /* level-0 groups that changed (update; 0 for a build) */
    uint64_t groups[TR_MULTI_MAX_LEVELS], present[TR_MULTI_MAX_LEVELS];
    uint64_t level_bytes[TR_MULTI_MAX_LEVELS], level_work[TR_MULTI_MAX_LEVELS];
    int32_t scoped[TR_MULTI_MAX_LEVELS];      /* 1: the level's bank is the bank of its scope */
    int32_t strategy[TR_MULTI_MAX_LEVELS];    /* TR_MULTI_ROLLUP / LOCAL / COPY (level 0: COPY) */
    int32_t map_status[TR_MULTI_MAX_LEVELS];  /* status of maps[i] (i < levels - 1), else 0 */
} tr_multi_hier_info;             /* 408 bytes */

/* one contrast cell: value = num / den (num 256-bit two's complement, least significant word first) */
typedef struct {
    uint64_t num[4];
    uint64_t den[2];              /* den[0] + 2^64 den[1] >= 1 when DEFINED, else 0 */
    uint32_t child, other;        /* the child group; the parent or sibling (UINT32_MAX when none) */
    int32_t status, term;
} tr_multi_q;                     /* 64 bytes */

/* the output of tr_mgpu_multi_certify (host arrays of nfine entries, written by the call) */
typedef struct {
    uint64_t nfine;
    uint64_t *first;              /* smallest selected row of each fine group, TR_MULTI_NONE when none */
    uint32_t *lo, *hi;            /* smallest / largest coarse group over its rows (lo = UINT32_MAX, hi = 0: none) */
    uint64_t witness;             /* the smallest violating row (see the header comment), TR_MULTI_NONE: none */
    uint64_t checked;             /* selected rows */
} tr_multi_cert;                  /* 48 bytes */

typedef struct tr_multi_map_s tr_multi_map;
typedef struct tr_multi_hier_s tr_multi_hier;

/* ------------------------------------------------------------------ exact 256-bit helpers (CPU and CUDA kernels) */
/* |x| of a 128-bit signed value as an unsigned 128-bit magnitude (exact also for -2^127) */
TR_NUM_HD unsigned __int128 tr_multi_mag(__int128 x) {
    return x < 0 ? (unsigned __int128)0 - (unsigned __int128)x : (unsigned __int128)x;
}
/* m[0..3] = a * b (a < 2^128, b < 2^64: the product < 2^192) */
TR_NUM_HD void tr_multi_mul_u128_u64(unsigned __int128 a, uint64_t b, uint64_t m[4]) {
    unsigned __int128 p0 = (unsigned __int128)(uint64_t)a * b, p1 = (unsigned __int128)(uint64_t)(a >> 64) * b;
    unsigned __int128 t = (p0 >> 64) + (unsigned __int128)(uint64_t)p1;
    m[0] = (uint64_t)p0; m[1] = (uint64_t)t; m[2] = (uint64_t)(p1 >> 64) + (uint64_t)(t >> 64); m[3] = 0;
}
TR_NUM_HD int tr_multi_cmp256(const uint64_t a[4], const uint64_t b[4]) {
    for (int i = 3; i >= 0; i--) if (a[i] != b[i]) return a[i] < b[i] ? -1 : 1;
    return 0;
}
TR_NUM_HD void tr_multi_add256(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]) {
    uint64_t carry = 0;
    for (int i = 0; i < 4; i++) {
        uint64_t s = a[i] + b[i], c1 = s < a[i];
        r[i] = s + carry;
        carry = c1 | (r[i] < s);
    }
}
TR_NUM_HD void tr_multi_sub256(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]) {   /* a >= b */
    uint64_t borrow = 0;
    for (int i = 0; i < 4; i++) {
        uint64_t d = a[i] - b[i], b1 = a[i] < b[i];
        r[i] = d - borrow;
        borrow = b1 | (d < borrow);
    }
}
TR_NUM_HD int tr_multi_is_zero256(const uint64_t a[4]) { return !(a[0] | a[1] | a[2] | a[3]); }
/* a signed value as (negative flag, magnitude); x - y of two of them (magnitudes < 2^255) */
TR_NUM_HD void tr_multi_signed_sub(int nx, const uint64_t mx[4], int ny, const uint64_t my[4], int *nr, uint64_t mr[4]) {
    if (nx != ny) { tr_multi_add256(mx, my, mr); *nr = nx; }
    else if (tr_multi_cmp256(mx, my) >= 0) { tr_multi_sub256(mx, my, mr); *nr = nx; }
    else { tr_multi_sub256(my, mx, mr); *nr = !nx; }
    if (tr_multi_is_zero256(mr)) *nr = 0;
}
TR_NUM_HD uint64_t tr_multi_mod_u64(const uint64_t m[4], uint64_t d) {      /* d >= 1 */
    unsigned __int128 r = 0;
    for (int i = 3; i >= 0; i--) r = ((r << 64) | m[i]) % d;
    return (uint64_t)r;
}
TR_NUM_HD void tr_multi_div_u64(uint64_t m[4], uint64_t d) {                /* m /= d, exact (d divides m) */
    unsigned __int128 r = 0;
    for (int i = 3; i >= 0; i--) {
        unsigned __int128 cur = (r << 64) | m[i];
        m[i] = (uint64_t)(cur / d);
        r = cur % d;
    }
}
/* reduce num / d by gcd(|num|, d): returns d / g and divides the magnitude by g */
TR_NUM_HD uint64_t tr_multi_reduce(uint64_t mag[4], uint64_t d) {
    uint64_t g = tr_gcd_u64(tr_multi_mod_u64(mag, d), d);   /* gcd(0, d) = d */
    if (g > 1) { tr_multi_div_u64(mag, g); d /= g; }
    return d;
}
TR_NUM_HD void tr_multi_store(int neg, const uint64_t mag[4], unsigned __int128 den, tr_multi_q *out) {
    uint64_t carry = 1;
    for (int i = 0; i < 4; i++) {
        if (neg) { uint64_t v = ~mag[i] + carry; carry = carry && v == 0; out->num[i] = v; }
        else out->num[i] = mag[i];
    }
    out->den[0] = (uint64_t)den; out->den[1] = (uint64_t)(den >> 64);
}
/* s_a * n_b - s_b * n_a as (sign, magnitude): |.| < 2^192 */
TR_NUM_HD void tr_multi_cross(__int128 sa, uint64_t na, __int128 sb, uint64_t nb, int *neg, uint64_t mag[4]) {
    uint64_t x[4], y[4];
    tr_multi_mul_u128_u64(tr_multi_mag(sa), nb, x);
    tr_multi_mul_u128_u64(tr_multi_mag(sb), na, y);
    tr_multi_signed_sub(sa < 0, x, sb < 0, y, neg, mag);
}
/* One contrast cell (see the header comment). c: the child (rows, count, sum of the term); o: the parent (MEAN, TOTAL,
 * SHARE) or the next present sibling (SIBLING_*), has_other = 0 when there is none. */
TR_NUM_HD void tr_multi_contrast(int32_t kind, uint64_t rows_c, uint64_t n_c, __int128 s_c, int has_other,
                                 uint64_t n_o, __int128 s_o, tr_multi_q *out) {
    uint64_t mag[4] = {0, 0, 0, 0};
    int neg = 0;
    for (int i = 0; i < 4; i++) out->num[i] = 0;
    out->den[0] = 0; out->den[1] = 0;
    if (!rows_c) { out->status = TR_MULTI_ABSENT; return; }
    if (!has_other) { out->status = TR_MULTI_NO_SIBLING; return; }
    if (kind == TR_MULTI_TOTAL || kind == TR_MULTI_SIBLING_TOTAL) {
        uint64_t x[4] = {(uint64_t)tr_multi_mag(s_c), (uint64_t)(tr_multi_mag(s_c) >> 64), 0, 0};
        uint64_t y[4] = {(uint64_t)tr_multi_mag(s_o), (uint64_t)(tr_multi_mag(s_o) >> 64), 0, 0};
        tr_multi_signed_sub(s_c < 0, x, s_o < 0, y, &neg, mag);
        tr_multi_store(neg, mag, 1, out);
        out->status = TR_MULTI_DEFINED;
        return;
    }
    if (kind == TR_MULTI_SHARE) {
        if (!n_o) { out->status = TR_MULTI_UNDEFINED; return; }
        tr_multi_cross(s_c, n_c, s_o, n_o, &neg, mag);      /* (s_c n_o - s_o n_c) / n_o */
        uint64_t d = tr_multi_reduce(mag, n_o);
        tr_multi_store(neg, mag, d, out);
        out->status = TR_MULTI_DEFINED;
        return;
    }
    if (!n_c || !n_o) { out->status = TR_MULTI_UNDEFINED; return; }   /* MEAN, SIBLING_MEAN */
    tr_multi_cross(s_c, n_c, s_o, n_o, &neg, mag);          /* (s_c n_o - s_o n_c) / (n_c n_o) */
    uint64_t dc = tr_multi_reduce(mag, n_c);                /* coprime to n_c / g1, then to n_o / g2: canonical */
    uint64_t dn = tr_multi_reduce(mag, n_o);
    tr_multi_store(neg, mag, (unsigned __int128)dc * dn, out);
    out->status = TR_MULTI_DEFINED;
}
static inline int tr_multi_kind_ok(int32_t kind) { return kind >= TR_MULTI_MEAN && kind <= TR_MULTI_SIBLING_TOTAL; }
static inline int tr_multi_sibling(int32_t kind) { return kind == TR_MULTI_SIBLING_MEAN || kind == TR_MULTI_SIBLING_TOTAL; }
/* the contrast of output cell i (child first + i / nsel, term terms[i % nsel]): shared by the CPU loop and the kernel.
 * child / parent: the states of the two levels; map: the parent of each child (UNMAPPED only for absent children);
 * next: the next present sibling of each child (TR_MULTI_UNMAPPED: none). */
TR_NUM_HD void tr_multi_contrast_at(const tr_bank_states *child, const tr_bank_states *parent, const uint32_t *map,
                                    const uint32_t *next, int32_t kind, const int32_t *terms, int32_t nsel,
                                    uint64_t first, uint64_t i, tr_multi_q *out) {
    uint64_t c = first + i / (uint64_t)nsel, m = (uint64_t)child->nterms;
    int32_t j = terms[i % (uint64_t)nsel];
    uint64_t k = c * m + (uint64_t)j;
    int sib = kind == TR_MULTI_SIBLING_MEAN || kind == TR_MULTI_SIBLING_TOTAL;
    uint32_t o = child->rows[c] ? (sib ? next[c] : map[c]) : TR_MULTI_UNMAPPED;
    const tr_bank_states *os = sib ? child : parent;
    uint64_t ko = (uint64_t)o * m + (uint64_t)j;
    tr_multi_contrast(kind, child->rows[c], child->count[k], tr_i128_get(child->sum[k]), o != TR_MULTI_UNMAPPED,
                      o != TR_MULTI_UNMAPPED ? os->count[ko] : 0,
                      o != TR_MULTI_UNMAPPED ? tr_i128_get(os->sum[ko]) : 0, out);
    out->child = (uint32_t)c; out->other = o; out->term = j;
}

/* the device copies of tr_mgpu_multi_certify: the selection, both levels' group ids, first / lo / hi per fine group
 * (64-bit words on the device) and the host's initialization scratch (8 bytes per fine group) */
static inline void tr_multi_certify_device_need(const tr_scope_view *fine, const tr_scope_view *coarse, tr_math_need *need) {
    for (int i = 0; i < TR_BUDGET_RESOURCES; i++) need->need[i] = 0;
    uint64_t n = fine->rows, bytes = tr_budget_add(tr_budget_bitmap_bytes(n), tr_budget_mul(fine->ngroups, 32));
    if (fine->groups) bytes = tr_budget_add(bytes, tr_budget_mul(n, 4));
    if (coarse->groups) bytes = tr_budget_add(bytes, tr_budget_mul(n, 4));
    need->need[TR_BUDGET_STATES] = fine->ngroups;
    need->need[TR_BUDGET_BYTES] = tr_budget_add(bytes, 16);
}
/* the device copies of tr_mgpu_multi_contrasts: both levels' states, the map and sibling links, the terms, the output */
static inline void tr_multi_contrasts_device_need(uint64_t child_groups, uint64_t parent_groups, int32_t nterms,
                                                  int32_t nsel, uint64_t count, tr_math_need *need) {
    for (int i = 0; i < TR_BUDGET_RESOURCES; i++) need->need[i] = 0;
    uint64_t cells = tr_budget_mul(count, (uint64_t)nsel);
    uint64_t bytes = tr_budget_add(tr_bank_state_bytes(child_groups, nterms), tr_bank_state_bytes(parent_groups, nterms));
    bytes = tr_budget_add(bytes, tr_budget_mul(child_groups, 8));
    bytes = tr_budget_add(bytes, tr_budget_add(tr_budget_mul((uint64_t)nsel, 4), tr_budget_mul(cells, sizeof(tr_multi_q))));
    need->need[TR_BUDGET_SUPPORT] = (uint64_t)nsel;
    need->need[TR_BUDGET_OUTPUT] = cells;
    need->need[TR_BUDGET_BYTES] = bytes;
}

/* ------------------------------------------------------------------ maps (libalg) */
tr_multi_map *tr_multi_map_certify(const tr_multi_level *fine, const tr_multi_level *coarse,
                                   const tr_math_budget *budget, tr_math_refusal *refusal, tr_multi_witness *witness,
                                   int32_t *error);
/* adopts the result of tr_mgpu_multi_certify for the same two levels: each first[f] must be a selected row of fine
 * group f (or NONE), lo/hi consistent with it, and a reported witness a violating row; the map is then PROVEN (or the
 * witness refuses it, exactly as tr_multi_map_certify would) */
tr_multi_map *tr_multi_map_from_cert(const tr_multi_level *fine, const tr_multi_level *coarse, const tr_multi_cert *cert,
                                     const tr_math_budget *budget, tr_math_refusal *refusal, tr_multi_witness *witness,
                                     int32_t *error);
/* map: n = fine groups entries, each < coarse groups or TR_MULTI_UNMAPPED */
tr_multi_map *tr_multi_map_declare(const tr_multi_level *fine, const tr_multi_level *coarse, const uint32_t *map,
                                   uint64_t n, const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error);
tr_multi_map *tr_multi_map_verify(const tr_multi_map *, const tr_multi_level *fine, const tr_multi_level *coarse,
                                  const tr_math_budget *budget, tr_math_refusal *refusal, tr_multi_witness *witness,
                                  int32_t *error);
tr_multi_map *tr_multi_map_compose(const tr_multi_map *a, const tr_multi_map *b, const tr_math_budget *budget,
                                   tr_math_refusal *refusal, int32_t *error);
/* TR_MATH_OK when the map is bound to exactly these levels' groupings and codecs (and, proven, to their rows key),
 * TR_MATH_DOMAIN_ERROR otherwise */
int32_t tr_multi_map_check(const tr_multi_map *, const tr_multi_level *fine, const tr_multi_level *coarse);
/* the needs of certify / declare / verify for these levels (target 0: CPU; 1: tr_mgpu_multi_certify's copies) */
int32_t tr_multi_map_need(const tr_multi_level *fine, const tr_multi_level *coarse, int32_t target, tr_math_need *need);
int32_t tr_multi_map_get_info(const tr_multi_map *, tr_multi_map_info *);
int32_t tr_multi_map_get(const tr_multi_map *, uint32_t *out, uint64_t capacity);
int32_t tr_multi_map_equal(const tr_multi_map *, const tr_multi_map *);   /* same entries, bindings and status: 1/0 */
void tr_multi_map_free(tr_multi_map *);

/* ------------------------------------------------------------------ hierarchies (libalg) */
/* levels[nlevels] (2..TR_MULTI_MAX_LEVELS), fine: the bank of levels[0].scope, maps[nlevels - 1]. bad_group (may be
 * NULL): the present group without a parent of a refusal, else TR_MULTI_NONE. */
tr_multi_hier *tr_multi_hier_build(const tr_multi_level *levels, int32_t nlevels, const tr_bank *fine,
                                   const tr_multi_map *const *maps, const tr_math_budget *budget,
                                   tr_math_refusal *refusal, uint64_t *bad_group, int32_t *error);
/* levels / maps: both NULL (carry the maps) or both given (the hierarchy of the new levels) */
tr_multi_hier *tr_multi_hier_update(const tr_multi_hier *, const tr_bank *fine, const tr_multi_level *levels,
                                    const tr_multi_map *const *maps, const tr_math_budget *budget,
                                    tr_math_refusal *refusal, uint64_t *bad_group, int32_t *error);
/* the needs of a build (fine bank and maps as for tr_multi_hier_build; levels used for their groups only) */
int32_t tr_multi_hier_need(const tr_multi_level *levels, int32_t nlevels, const tr_bank *fine, tr_math_need *need);
int32_t tr_multi_hier_get_info(const tr_multi_hier *, tr_multi_hier_info *);
/* pointers into the level's states (read only, valid while the hierarchy lives) */
int32_t tr_multi_hier_states(const tr_multi_hier *, int32_t level, tr_bank_states *out);
/* an independently owned copy of the level's bank (scoped iff info.scoped[level]) */
tr_bank *tr_multi_hier_bank(const tr_multi_hier *, int32_t level, int32_t *error);
/* an independently owned copy of maps[level] (level < levels - 1) */
tr_multi_map *tr_multi_hier_map(const tr_multi_hier *, int32_t level, int32_t *error);
/* the links of child level `level` (< levels - 1): map[g] (the parent; UNMAPPED for absent children) and next[g]
 * (the next present sibling, UNMAPPED when none); capacity >= the level's groups; scratch budgeted */
int32_t tr_multi_hier_links(const tr_multi_hier *, int32_t level, uint32_t *map, uint32_t *next, uint64_t capacity,
                            const tr_math_budget *budget, tr_math_refusal *refusal);
/* contrasts of child level `level` (see the header comment): out[count * nsel] for child groups first..first+count-1
 * and the terms terms[nsel] (distinct, < nterms), cell i = (child first + i / nsel, term terms[i % nsel]) */
int32_t tr_multi_contrasts(const tr_multi_hier *, int32_t level, int32_t kind, const int32_t *terms, int32_t nsel,
                           uint64_t first, uint64_t count, const tr_math_budget *budget, tr_multi_q *out,
                           tr_math_refusal *refusal);
/* the needs of tr_multi_contrasts (target 0) or of tr_mgpu_multi_contrasts (target 1) */
int32_t tr_multi_contrasts_need(const tr_multi_hier *, int32_t level, int32_t nsel, uint64_t count, int32_t target,
                                tr_math_need *need);
void tr_multi_hier_free(tr_multi_hier *);

/* ------------------------------------------------------------------ the CUDA library (math_multi_gpu.cu)
 * certify: the views of the two levels (tr_scope_get_view; same rows, checked by tr_multi_map_from_cert afterwards);
 * out: host arrays of fine->ngroups entries; then adopt with tr_multi_map_from_cert. contrasts: the states of a child
 * level and its parent level (tr_multi_hier_states), the links (tr_multi_hier_links), and out[count * nsel], equal to
 * tr_multi_contrasts. Both return 0 or a native error code (TR_MATH_DEVICE when the device fails). */
int tr_mgpu_multi_certify(const tr_scope_view *fine, const tr_scope_view *coarse, const tr_math_budget *budget,
                          tr_multi_cert *out, tr_math_refusal *refusal);
int tr_mgpu_multi_contrasts(const tr_bank_states *child, const tr_bank_states *parent, const uint32_t *map,
                            const uint32_t *next, int32_t kind, const int32_t *terms, int32_t nsel, uint64_t first,
                            uint64_t count, const tr_math_budget *budget, tr_multi_q *out, tr_math_refusal *refusal);
#ifdef __cplusplus
}
#endif
#endif
