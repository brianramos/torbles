/* FND-004: the data scope of the mathematical capabilities: the rows the engine selected, value columns as exact
 * logical integers with a validity flag per value, and a dense group id per row.
 * SPDX-License-Identifier: Apache-2.0
 *
 * A scope is an immutable, individually owned snapshot, decoded once when it is built (natively, with
 * at_decode_column for an engine table). Every later data-side capability (moment banks, grouped regression,
 * histograms, distributions, deltas) reads its rows through a scope, on the CPU or in a CUDA kernel (tr_scope_view).
 *
 * Rows. The scope keeps the physical row positions 0..rows-1 of its source and a copy of the selection bitmap (bit r
 * of word r/64; NULL at build = every row; bits past `rows` are cleared). Only selected rows have meaning: a capability
 * visits selected rows only, and the bag multiplicity of a group is its number of selected rows.
 *
 * Values. Column c holds exact logical integers. From an engine table, logical = code + offset (code: the stored
 * code; with a high part, code + (high code << shift): the self-extending columns of algq.Table(extend=...)); the
 * decimal value is logical / 10^scale. From arrays, logical = the given int64. A value is present (not NULL) when its
 * validity flag is nonzero; a column without validity is never NULL. Every selected present value must fit int64
 * (else the build is refused with TR_MATH_OVERFLOW). The scope records per column the number of selected present
 * values and their minimum and maximum (min > max when there is none): later capabilities prove their overflow
 * bounds from these extremes. Values of NULL or unselected rows are unspecified and must not be read.
 *
 * Groups. ngroups >= 1 dense ids; a selected row has one id < ngroups (the id of an unselected row is 0 and
 * meaningless). From an engine table the id is the mixed-radix number of the key codes, the first key most
 * significant (no keys: one group). A key code is the stored code of the key column, or lookup[code] for a dimension
 * attribute reached through a foreign-key code (lookup: one code per dimension row, copied at build). A key is NULL
 * when its validity column says so, when lookup_valid says the dimension attribute is NULL, or when the foreign-key
 * code has no dimension row (code >= lookup_n: an unmatched reference). A nullable key gives NULL the code null_code
 * (a group of its own, like SQL GROUP BY); a non-nullable key refuses a selected row whose key is NULL
 * (TR_MATH_INVALID). Every present key code must be < radix and differ from null_code (else TR_MATH_INVALID).
 * ngroups (the product of the radices) is at most TR_SCOPE_MAX_GROUPS; groups are dense: an id without selected rows
 * is an absent group, never a zero group.
 *
 * Identity and freshness. A table scope records the table's address, the caller's 16-byte identity token (a
 * lifetime token of the table wrapper: the address alone may be reused by another table), the table's revision and
 * row count, the revisions of the sidecars of the self-extending columns it read, and the exact selection.
 * tr_scope_fresh refuses a scope (TR_MATH_STALE) once any of them differs. The scope never dereferences its table
 * after the build (it compares addresses), so it may outlive the table. A lookup is a snapshot copied at build: the
 * freshness of its source (a dimension table) is the caller's to check (the Python wrapper records each dimension's
 * revision and used rows and checks them in Scope.fresh).
 *
 * Descriptor. The exact semantic descriptor of a scope (tr_scope_key_*) is a byte string: for a table scope the
 * source identity and version, the selection bits, every column spec (col, validity col, high col, shift, offset,
 * scale) and every key spec (col, validity col, nullable, null code, radix, the lookup and lookup_valid contents);
 * for an array scope the selection bits and the data itself (values and validity normalized to 0 where a value is
 * unselected or NULL, ids 0 on unselected rows, an all-present validity array dropped). Two scopes with equal
 * descriptors hold the same data; caches key on descriptors, never on hashes alone (the 128-bit digest is a
 * pre-filter). Part TR_SCOPE_KEY_ROWS is everything but the grouping, part TR_SCOPE_KEY_GROUPS the grouping.
 *
 * Budgets (math_budgets.h), checked before anything is allocated: states = ngroups, bytes = every byte the build
 * allocates (the copied selection, the decoded columns, the group ids, the copied lookups, the decoding scratch, the
 * object itself): tr_scope_need_table / tr_scope_need_arrays. Refusal: TR_MATH_BUDGET with the tr_math_refusal.
 * More than TR_SCOPE_MAX_GROUPS groups (with a budget that admits them) is TR_MATH_UNSUPPORTED.
 *
 * Errors: TR_MATH_INVALID (bad argument, a code or id out of range), TR_MATH_MEMORY, TR_MATH_OVERFLOW,
 * TR_MATH_BUDGET, TR_MATH_UNSUPPORTED, TR_MATH_STALE, TR_MATH_DOMAIN_ERROR (tr_scope_rollup_map). Objects are
 * immutable; constructors return NULL and set *error, leaving their inputs unchanged. Read-only calls on one scope may
 * run concurrently. */
#ifndef TORBLES_MATH_SCOPE_H
#define TORBLES_MATH_SCOPE_H
#include <stdint.h>
#include "math_num.h"
#include "math_budgets.h"
#ifdef __cplusplus
extern "C" {
#endif

#define TR_SCOPE_MAX_COLUMNS 32
#define TR_SCOPE_MAX_KEYS 12
#define TR_SCOPE_MAX_SCALE 18
#define TR_SCOPE_MAX_GROUPS UINT64_C(4294967295)   /* ids fit uint32 and UINT32_MAX stays free as a marker */
enum { TR_SCOPE_TABLE = 1, TR_SCOPE_ARRAYS = 2 };                    /* the source of a scope */
enum { TR_SCOPE_KEY_ROWS = 1, TR_SCOPE_KEY_GROUPS = 2, TR_SCOPE_KEY_ALL = 3 };   /* descriptor parts */

/* a value column of an engine table: logical = code + (hi code << shift) + offset; value = logical / 10^scale */
typedef struct {
    int32_t col;      /* stored column of the codes */
    int32_t valid;    /* stored validity column (nonzero = present), or -1: never NULL */
    int32_t hi;       /* stored column of the high part (the "#hi" sidecar of a self-extending column), or -1 */
    int32_t shift;    /* 0..32, used when hi >= 0 (algq.Table(extend={col: B}): B) */
    int64_t offset;   /* the codec offset (coltypes: Codec.affine[0]) */
    int32_t scale;    /* 0..18 (Codec.affine[1]) */
    int32_t reserved; /* 0 */
} tr_scope_column;    /* 32 bytes */

/* a group key of an engine table (see the header comment) */
typedef struct {
    int32_t col;                    /* stored column of the key code (the foreign-key code when lookup is given) */
    int32_t valid;                  /* its validity column, or -1 */
    int32_t nullable;               /* 1: a NULL key takes null_code; 0: a selected NULL key is refused */
    uint32_t null_code;             /* < radix when nullable */
    uint64_t radix;                 /* >= 1: present key codes are < radix */
    const uint32_t *lookup;         /* NULL, or lookup_n key codes indexed by the stored code (copied) */
    const uint32_t *lookup_valid;   /* NULL, or lookup_n flags (0 = the attribute is NULL); needs lookup (copied) */
    uint64_t lookup_n;              /* entries of lookup (codes >= lookup_n are unmatched references) */
} tr_scope_key;                     /* 48 bytes */

/* a value column given as plain arrays (data that is not an engine table) */
typedef struct {
    const int64_t *values;   /* rows logical values */
    const uint8_t *valid;    /* rows flags (nonzero = present), or NULL: never NULL */
    int32_t scale;           /* 0..18: value = logical / 10^scale */
    int32_t reserved;        /* 0 */
} tr_scope_array;            /* 24 bytes */

/* The plain-C view of a scope for CPU loops and CUDA kernels: pointers into the scope's own arrays (valid while the
 * scope lives), copied by value into a kernel with its pointers swapped for device copies (math_bank_gpu.cu). Exactly
 * one of codes / values is non-NULL for a column with rows. */
typedef struct {
    const uint32_t *codes;   /* stored codes (logical = code + offset), or NULL */
    const int64_t *values;   /* logical values, or NULL */
    const uint8_t *valid;    /* 0 = NULL, else present; NULL pointer = no NULL among the selected rows */
    int64_t offset;          /* added to codes (0 with values) */
    int64_t min, max;        /* extremes of the selected present values (min > max when nvalid == 0) */
    uint64_t nvalid;         /* selected rows where the value is present */
    int32_t scale;           /* decimal scale */
    int32_t reserved;
} tr_scope_view_column;      /* 64 bytes */

typedef struct {
    uint64_t rows;                /* physical rows */
    uint64_t selected;            /* selected rows */
    uint64_t ngroups;             /* >= 1 */
    const uint64_t *selection;    /* ceil(rows / 64) words, bits past rows clear (NULL only when rows == 0) */
    const uint32_t *groups;       /* the group id of each row, or NULL when ngroups == 1 (every id is 0) */
    int32_t ncols;                /* 0..TR_SCOPE_MAX_COLUMNS */
    int32_t reserved;
    tr_scope_view_column col[TR_SCOPE_MAX_COLUMNS];
} tr_scope_view;                  /* 2096 bytes: a CUDA kernel argument (< 4 KiB) */

typedef struct {
    uint64_t rows, selected, ngroups;
    uint64_t bytes;                   /* bytes the scope retains */
    uint64_t table;                   /* table address (0 for arrays) */
    uint64_t revision;                /* the table's revision at build (0 for arrays) */
    uint64_t digest[2];               /* 128-bit digest of the whole descriptor (a pre-filter, not an identity) */
    uint64_t version[2];              /* digest of the table version: revision, rows, sidecar revisions */
    uint8_t identity[16];             /* the caller's token (zero for arrays) */
    int32_t source;                   /* TR_SCOPE_TABLE / TR_SCOPE_ARRAYS */
    int32_t ncols, nkeys;
    int32_t reserved;
} tr_scope_info;                      /* 112 bytes */

/* per column: what the build decoded (spec: -1 where it does not apply) */
typedef struct {
    int64_t offset, min, max;   /* min > max when nvalid == 0 */
    uint64_t nvalid;
    int32_t scale;
    int32_t storage;            /* 1 = uint32 codes + offset, 2 = int64 values */
    int32_t col, valid, hi, shift;
} tr_scope_column_info;         /* 56 bytes */

typedef struct tr_scope_s tr_scope;
struct atable_s;

/* ------------------------------------------------------------------ shared accessors (CPU code and CUDA kernels) */
TR_NUM_HD int tr_scope_is_selected(const tr_scope_view *v, uint64_t row) {
    return (int)((v->selection[row >> 6] >> (row & 63)) & 1);
}
TR_NUM_HD int tr_scope_present(const tr_scope_view *v, int32_t c, uint64_t row) {
    return !v->col[c].valid || v->col[c].valid[row] != 0;
}
/* the logical value of column c at a selected row where it is present (computed in unsigned arithmetic: never
 * undefined behaviour, exact for every row the scope proved to fit int64) */
TR_NUM_HD int64_t tr_scope_value(const tr_scope_view *v, int32_t c, uint64_t row) {
    const tr_scope_view_column *k = &v->col[c];
    return k->values ? k->values[row] : (int64_t)((uint64_t)k->codes[row] + (uint64_t)k->offset);
}
TR_NUM_HD uint32_t tr_scope_group(const tr_scope_view *v, uint64_t row) { return v->groups ? v->groups[row] : 0; }

/* ------------------------------------------------------------------ construction */
/* From an engine table (`table`: an atable_t, e.g. algq.Table.at). identity: 16 bytes or NULL (zeros). rows: the
 * selection, ceil(n / 64) words (e.g. Table.used_bm or a filter result), NULL = every row. budget: NULL = defaults. */
tr_scope *tr_scope_from_table(const struct atable_s *table, const uint8_t *identity, const uint64_t *rows,
                              const tr_scope_column *columns, int32_t ncols, const tr_scope_key *keys, int32_t nkeys,
                              const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error);
/* From arrays of n rows: columns[ncols]; groups (n ids < ngroups) or NULL with ngroups == 1; rows as above. */
tr_scope *tr_scope_from_arrays(uint64_t n, const uint64_t *rows, const tr_scope_array *columns, int32_t ncols,
                               const uint32_t *groups, uint64_t ngroups, const tr_math_budget *budget,
                               tr_math_refusal *refusal, int32_t *error);
/* the needs the constructors check (states, bytes); the arguments are validated as by the constructor */
int32_t tr_scope_need_table(const struct atable_s *table, const tr_scope_column *columns, int32_t ncols,
                            const tr_scope_key *keys, int32_t nkeys, tr_math_need *need);
int32_t tr_scope_need_arrays(uint64_t n, const tr_scope_array *columns, int32_t ncols, const uint32_t *groups,
                             uint64_t ngroups, tr_math_need *need);
void tr_scope_free(tr_scope *);

/* ------------------------------------------------------------------ reading */
int32_t tr_scope_get_info(const tr_scope *, tr_scope_info *);
int32_t tr_scope_get_view(const tr_scope *, tr_scope_view *);
int32_t tr_scope_column_get(const tr_scope *, int32_t column, tr_scope_column_info *);
/* TR_MATH_OK when the scope still describes `table` with the selection `rows` (NULL = every row): same address,
 * revision, row count, sidecar revisions and selection bits; TR_MATH_STALE otherwise. An array scope is checked
 * against table == NULL and its selection only. */
int32_t tr_scope_fresh(const tr_scope *, const struct atable_s *table, const uint64_t *rows);
/* exact descriptor equality of two scopes, whole (tr_scope_same) or but for the grouping (tr_scope_same_rows):
 * 1 or 0 (TR_MATH_INVALID on a bad argument) */
int32_t tr_scope_same(const tr_scope *, const tr_scope *);
int32_t tr_scope_same_rows(const tr_scope *, const tr_scope *);
/* The rollup (refinement) map from the groups of `fine` to those of `coarse`: two scopes of the same rows and columns
 * (tr_scope_same_rows) whose coarse grouping is a function of the fine one on the selected rows. map[g] (capacity >=
 * fine's ngroups) is the coarse group of fine group g; fine groups without selected rows map to 0 (they hold nothing).
 * TR_MATH_DOMAIN_ERROR when the scopes differ in rows or columns or when one fine group's rows fall in two coarse
 * groups (not a refinement); the map's contents are then unspecified. Nothing is allocated. */
int32_t tr_scope_rollup_map(const tr_scope *fine, const tr_scope *coarse, uint32_t *map, uint64_t capacity);

/* ------------------------------------------------------------------ the exact descriptor (see the header comment) */
uint64_t tr_scope_key_size(const tr_scope *, int32_t part);
/* writes tr_scope_key_size bytes; TR_MATH_INVALID when capacity is smaller */
int32_t tr_scope_key_write(const tr_scope *, int32_t part, unsigned char *out, uint64_t capacity);
/* 1 when key[0..bytes) is exactly the descriptor part, 0 otherwise */
int32_t tr_scope_key_equal(const tr_scope *, int32_t part, const unsigned char *key, uint64_t bytes);
#ifdef __cplusplus
}
#endif
#endif
