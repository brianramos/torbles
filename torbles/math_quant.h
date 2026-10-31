/* QUANT-001: existential and universal group predicates; QUANT-002: relational division and required-category
 * coverage. Exact per-group states over the child rows of a data scope (math_scope.h), on the CPU (math_quant.c) and
 * on a CUDA device (math_quant_gpu.cu) with the same states bit for bit.
 * SPDX-License-Identifier: Apache-2.0
 *
 * ---- Parents, children and groups (both capabilities) ----------------------------------------------------------
 * The child rows are the selected rows of a FND-004 scope `child` (orders' lines, say); a child row belongs to the
 * group of its scope group id g < G = child ngroups (the parent key, possibly a dimension attribute reached through a
 * foreign key: math_scope.h). Bag multiplicity: every selected child row counts once, duplicated rows count twice.
 * Which groups exist (presence) is declared by the group domain, kept apart from the counts:
 *   TR_QUANT_GROUPED  present iff the group has a selected child row (SQL GROUP BY over the children: no empty group);
 *   TR_QUANT_IDS      every id 0 .. G - 1 is present (array scopes whose ids enumerate the parents);
 *   TR_QUANT_PARENTS  present iff the scope `parents` (the parent rows: one row per order, say) has a selected row in
 *                     the group; parents[g] counts them (parent multiplicity). Parent group g is child group g: the
 *                     parent scope has P <= G groups with the same key radices but possibly the first one (the child's
 *                     NULL code of an unmatched foreign key lies past the parent's ids); child groups >= P have no
 *                     parent. With a primary-key parent this is SQL's correlated subquery `FROM parents p WHERE ...
 *                     (SELECT .. FROM child c WHERE c.key = p.key)`: a NULL or unmatched child key matches no parent
 *                     (NULL keys: below).
 * A present group without child rows is EMPTY: its child set is empty and every quantifier takes its vacuous value.
 * An absent group has no answer (TR_QUANT_ABSENT, never FALSE or UNKNOWN); child rows of absent groups (orphans) are
 * counted in the states and in info.orphans, never in an answer. NULL keys: FND-004 puts a NULL key, an unmatched
 * foreign key and a NULL dimension attribute into the key's NULL group. Under GROUPED it is a group of its own (SQL
 * GROUP BY). Under PARENTS with a non-nullable parent key (a primary key) the child NULL group lies past the parent ids
 * and has no parent (SQL's correlated c.key = p.key); with a nullable parent key the parents whose key is NULL are the
 * parents of the child NULL group (groups match like IS NOT DISTINCT FROM over children LEFT JOIN the dimension).
 *
 * ---- QUANT-001: quantifier states and their answers --------------------------------------------------------------
 * The child predicate p is a BOOL-001 program (math_bool.h) over k native row conditions of the child scope, evaluated
 * on every selected child row with SQL's three-valued logic: TRUE, FALSE or UNKNOWN (a condition over a NULL is
 * UNKNOWN; Kleene AND / OR / NOT). Per group g, exactly (uint64):
 *   rows[g]        selected child rows (n = t + f + u)      qualifying[g]  rows where p is TRUE   (t)
 *   failing[g]     rows where p is FALSE                    (f)          unknown[g]  rows where p is UNKNOWN (u)
 *   parents[g]     selected parent rows (PARENTS; 0 for the other domains)
 * Answers per present group (an absent group: TR_QUANT_ABSENT), with their SQL forms (child c of parent p):
 *   SOME        TRUE if t > 0, else UNKNOWN if u > 0, else FALSE          TRUE = ANY (SELECT p FROM c ...)  (Kleene OR)
 *   ALL         FALSE if f > 0, else UNKNOWN if u > 0, else TRUE          TRUE = ALL (SELECT p FROM c ...)  (Kleene AND)
 *   NONE        NOT SOME: TRUE if t = u = 0, FALSE if t > 0, else UNKNOWN
 *   NOT_ALL     NOT ALL:  TRUE if f > 0, FALSE if f = u = 0, else UNKNOWN
 *   EXISTS      TRUE iff t > 0 (two-valued)                               EXISTS (SELECT 1 FROM c WHERE ... AND p)
 *   NOT_EXISTS  TRUE iff t = 0                                            NOT EXISTS (... AND p)
 *   NO_COUNTEREXAMPLE  TRUE iff f = 0 (an UNKNOWN child does not refute)  NOT EXISTS (... AND NOT p)
 *   ALL_TRUE    TRUE iff t = n (an UNKNOWN child refutes)                 NOT EXISTS (... AND p IS NOT TRUE)
 *   BOOL_OR     UNKNOWN if t + f = 0, else TRUE iff t > 0                 bool_or(p): NULLs skipped, NULL without input
 *   BOOL_AND    UNKNOWN if t + f = 0, else TRUE iff f = 0                 bool_and(p)
 * Conventions: vacuous truth (an EMPTY group: SOME FALSE, ALL TRUE, NONE TRUE, NOT_ALL FALSE, EXISTS FALSE, NOT_EXISTS
 * TRUE, NO_COUNTEREXAMPLE TRUE, ALL_TRUE TRUE, BOOL_OR / BOOL_AND UNKNOWN: SQL's aggregate over no input); all-NULL
 * children (t = f = 0 < u: SOME, ALL, NONE, NOT_ALL, BOOL_OR, BOOL_AND UNKNOWN, EXISTS FALSE, NO_COUNTEREXAMPLE TRUE,
 * ALL_TRUE FALSE: an all-NULL group is not an empty group); UNKNOWN is never read as FALSE.
 * Multiplicity counts rows but never erases coverage: the states are integer row counts and every answer depends only
 * on which of t, f, u, n are zero, so a duplicated child row doubles counts and changes no answer; states combine by
 * integer addition (tr_quant_groups_merge: the bag union), never by a parity, so a covered group stays covered.
 *
 * ---- QUANT-002: coverage states, relational division --------------------------------------------------------------
 * Categories. A second scope `categories` over the same rows (the same source, table version and selection bits; its
 * value columns may differ) gives each child row a category code: its group id c < D = categories' ngroups (e.g. the
 * stored codes of a product column, a dictionary-coded string column or a dimension attribute: FND-004 keys). The
 * category dictionary (tr_quant_dict, the codec) says what the codes mean: CODES (integer codes: key(c) = the logical
 * value c + offset, decimal value key / 10^scale; a key is its 8-byte little-endian int64) or KEYS (key(c) = an exact
 * byte string per code, e.g. the UTF-8 values of a string dictionary; distinct). One code may be the NULL category
 * (null_code: a NULL value or an unmatched reference; it has no key). Rules: a NULL category is never a category: it
 * never satisfies a requirement and is never an extra category (SQL: c.cat = r.cat is UNKNOWN); its rows are counted
 * in nulls[g]. Duplicated categories count once for coverage (sets / OR masks) and every row counts in rows[g].
 * Required set R: a set of keys (duplicates merged), resolved natively against the dictionary: keys of codes form R_in
 * (the required codes); keys outside the dictionary (no row can have them) and a NULL requirement (null_key: SQL's
 * `c.cat = NULL` never matches) can never be satisfied and are missing in every group (constant_missing).
 * Per group g, exactly (uint64):
 *   rows[g], nulls[g], parents[g]   child rows, rows with a NULL category, parent rows (PARENTS)
 *   distinct[g]   |present(g)|: the distinct non-NULL categories of the group's rows
 *   matched[g]    |present(g) & R_in|      missing[g] = |R_in| - matched[g] + constant_missing
 *   extra[g]      |present(g) \ R_in| = distinct[g] - matched[g]
 * Answers per present group (absent: TR_QUANT_ABSENT), two-valued as their SQL forms are:
 *   DIVIDES    missing = 0: the group has every required category (vacuously TRUE for R empty, also for an EMPTY
 *              group): NOT EXISTS (SELECT 1 FROM R r WHERE NOT EXISTS (SELECT 1 FROM c WHERE c.key = p.key AND
 *              c.cat = r.cat)), the classic double NOT EXISTS
 *   EXACTLY    DIVIDES and extra = 0 (present(g) = R_in)
 *   OVERLAPS   matched > 0: EXISTS (SELECT 1 FROM c JOIN R r ON c.cat = r.cat WHERE c.key = p.key)
 *   DISJOINT   matched = 0
 * Representations (the same states, the same answers; tr_quant_cover_list gives the categories in increasing code):
 *   MASKS  per group a bit mask of D bits, W = ceil(D / 64) words, row-major: a canonical FIELD-001 matrix G x D
 *          (tr_quant_cover_masks gives the tr_field_matrix view; bits past D and the NULL code's bit are zero);
 *   SETS   per group the sorted distinct codes (CSR: offsets[G + 1], codes[]), for domains whose masks exceed the budget.
 * The choice (TR_QUANT_REPR_AUTO) is explicit and costed (tr_math_budget_choose: expanded = MASKS, factored = SETS,
 * costs = the bytes each stores: 8 G W and 4 S + 8 (G + 1), S = the selected rows); MASKS or SETS forced: only that one,
 * refused over budget. The CUDA path builds MASKS only.
 * Identity. The state identity is an exact byte string (never a hash alone; info.digest is its 128-bit pre-filter): the
 * presence part (the domain, the FND-004 descriptors of child, categories and parents, the dictionary's descriptor:
 * kind, size, null code, offset and scale or every key) and the requirement part (the sorted required codes, the
 * sorted distinct outside keys, the NULL flag). Changing the required set or the dictionary changes the identity: a
 * state is never reused for another one (tr_quant_cover_same, tr_quant_cover_matches). tr_quant_cover_require derives
 * the state of another required set from the same presence states (no rescan) under the new identity; it equals the
 * state a build with that set gives.
 *
 * ---- Execution ----------------------------------------------------------------------------------------------
 * CPU (one pass over the selection words): QUANT-001 evaluates per 64-row word the conditions the root reads
 * (tr_bool_condition_word) and the program (tr_bool_eval_base on a one-word domain), then attributes the TRUE / FALSE /
 * UNKNOWN words to groups: with popcounts when every selected row of the word has one group (contiguous groups),
 * row by row otherwise. QUANT-002 sets each row's bit in its group's mask (OR: idempotent) or collects (group, code)
 * per group and sorts and deduplicates each group's codes; the coverage test per group is AND-NOT and popcount
 * against the required mask (MASKS) or a binary search of the required codes (SETS).
 * CUDA (tr_mgpu_quant_groups, tr_mgpu_quant_cover, tr_mgpu_quant_cover_test): work items are slices of consecutive
 * words; an item evaluates its words as the CPU does (its own scratch slot: condition and node words), accumulates
 * the counts of a run of rows of one group locally and adds them with 64-bit atomics when the group changes and at its
 * end (a contiguous word: four atomics; scattered groups: about one per row); mask words are combined with atomic OR
 * (pending bits of one mask word kept locally). The coverage test is one work item per group with its own output
 * slots. Parent rows are counted by a third kernel. Integer atomics commute and OR is idempotent, so the states do not
 * depend on thread order. The host adopts the device arrays after consistency checks (tr_quant_groups_adopt,
 * tr_quant_cover_adopt, tr_quant_cover_require).
 *
 * ---- Budgets (math_budgets.h), checked before anything is allocated ----------------------------------------------
 *   groups (tr_quant_groups_need): variables = k, circuit = nodes, states = G, output = G, bytes = the object, the
 *     five arrays (40 G), the identity, the copies of the predicate and parent descriptors and the scan scratch
 *     (16 k + 16 nodes); adoption: the same without the scratch; CUDA: bytes = tr_quant_groups_device_bytes.
 *   dictionary (tr_quant_dict_need): support = D, bytes = the object (+ KEYS: the keys, D + 1 offsets, a 4-byte sorted
 *     index per code and 4 bytes per code of sorting scratch).
 *   cover (tr_quant_cover_need, per representation): support = D, states = G x D (MASKS) or S (SETS: the pairs held),
 *     output = G, bytes = the object, the seven arrays (56 G), the masks and the required mask (8 (G W + W)) or the sets
 *     (8 (G + 1) offsets + 4 S codes, + 8 G cursors of scratch), the required set for n keys (4 n codes, their bytes and
 *     8 (n + 1) offsets for the outside keys, 8 n of resolution scratch), the dictionary copy, the identity (the
 *     requirement part bounded by 40 + 12 n + the key bytes) and the parent descriptor; CUDA: bytes =
 *     tr_quant_cover_device_bytes (the device copies; the adoption is checked as the CPU masks build).
 *   merge / require: bytes = the new object (as above, without scan scratch).
 * Representation limits: D <= TR_SCOPE_MAX_GROUPS (category codes are group ids), k <= 64 and nodes <= 2^24
 * (BOOL-001).
 *
 * Errors: TR_MATH_INVALID (-1: a malformed argument, an inconsistent adopted array), _MEMORY (-2), _OVERFLOW (-3: a
 * merged count past 2^64 - 1), _BUDGET (-5, with the tr_math_refusal), TR_MATH_DOMAIN_ERROR (-6: a program over
 * another k, parents with more groups than the children, scopes over other rows, a dictionary of another size, merging
 * states of other predicates, dictionaries, required sets or domains), TR_MATH_DEVICE (-11), TR_MATH_UNSUPPORTED (-12:
 * the masks of a SETS state, a dictionary past TR_SCOPE_MAX_GROUPS codes, more than 2^32 - 1 required keys). States
 * are immutable, individually owned snapshots (constructors return NULL and set *error, inputs unchanged); they copy
 * what they need (the scopes may be freed after); their freshness is their scopes' (tr_scope_fresh; the Python layer
 * checks it before every use). Read-only calls on one object may run concurrently. */
#ifndef TORBLES_MATH_QUANT_H
#define TORBLES_MATH_QUANT_H
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "math_num.h"
#include "math_budgets.h"
#include "math_scope.h"
#include "math_bool.h"
#include "math_field.h"
#ifdef __cplusplus
extern "C" {
#endif

#define TR_QUANT_ABI 1
#define TR_QUANT_NO_NULL UINT32_MAX            /* a dictionary without a NULL category */
#define TR_QUANT_WORK (UINT64_C(1) << 14)      /* about this many work items per launch */
#define TR_QUANT_MIN_WORDS 4                    /* and at least this many words (or 256 rows) per work item */
#define TR_QUANT_SCRATCH (UINT64_C(1) << 26)   /* at most this many bytes of per-item scratch on the device */

enum { TR_QUANT_FALSE = 0, TR_QUANT_TRUE = 1, TR_QUANT_UNKNOWN = 2, TR_QUANT_ABSENT = 3 };      /* answer codes */
enum { TR_QUANT_GROUPED = 0, TR_QUANT_IDS = 1, TR_QUANT_PARENTS = 2 };                           /* group domains */
enum { TR_QUANT_SOME = 1, TR_QUANT_ALL = 2, TR_QUANT_NONE = 3, TR_QUANT_NOT_ALL = 4, TR_QUANT_EXISTS = 5,
       TR_QUANT_NOT_EXISTS = 6, TR_QUANT_NO_COUNTEREXAMPLE = 7, TR_QUANT_ALL_TRUE = 8, TR_QUANT_BOOL_OR = 9,
       TR_QUANT_BOOL_AND = 10 };                                                                 /* QUANT-001 */
enum { TR_QUANT_DIVIDES = 1, TR_QUANT_EXACTLY = 2, TR_QUANT_OVERLAPS = 3, TR_QUANT_DISJOINT = 4 };   /* QUANT-002 */
enum { TR_QUANT_REPR_AUTO = 0, TR_QUANT_REPR_MASKS = 1, TR_QUANT_REPR_SETS = 2 };
enum { TR_QUANT_DICT_CODES = 1, TR_QUANT_DICT_KEYS = 2 };
enum { TR_QUANT_LIST_PRESENT = 1, TR_QUANT_LIST_MISSING = 2, TR_QUANT_LIST_EXTRA = 3 };
enum { TR_QUANT_NEED_CPU = 0, TR_QUANT_NEED_DEVICE = 1, TR_QUANT_NEED_ADOPT = 2 };

/* the per-group arrays of a quantifier state: pointers into the state (read only), or the caller's arrays for the
 * CUDA path (each [ngroups]) */
typedef struct {
    uint64_t ngroups;
    uint64_t *rows;          /* selected child rows (bag multiplicity) */
    uint64_t *qualifying;    /* rows where the predicate is TRUE */
    uint64_t *failing;       /* rows where it is FALSE */
    uint64_t *unknown;       /* rows where it is UNKNOWN */
    uint64_t *parents;       /* selected parent rows (TR_QUANT_PARENTS, else 0) */
} tr_quant_counts;           /* 48 bytes */

typedef struct {
    uint64_t ngroups, present, empty, absent;     /* empty: present without child rows */
    uint64_t rows, qualifying, failing, unknown;  /* totals over every group */
    uint64_t orphans;                             /* child rows of absent groups (counted, never answered) */
    uint64_t parent_rows;                         /* selected parent rows (PARENTS) */
    uint64_t bytes, identity_bytes;               /* bytes the state retains; bytes of its identity */
    uint64_t digest[2];                           /* 128-bit digest of the identity (a pre-filter, not the identity) */
    int32_t domain, k, nnodes, parts;             /* parts: child bags merged (1 for a build) */
} tr_quant_groups_info;                           /* 128 bytes */

/* a list of keys: key i is bytes[offsets[i] .. offsets[i + 1]) (dictionary entries or a required set) */
typedef struct {
    const uint8_t *bytes;
    const uint64_t *offsets;     /* n + 1 non-decreasing offsets (offsets[0] = 0) */
    uint64_t n;
    int32_t null_key;            /* required sets: 1 = NULL is required (never satisfied); dictionaries: 0 */
    int32_t reserved;            /* 0 */
} tr_quant_keys;                 /* 32 bytes */

typedef struct {
    uint64_t size;               /* D: the codes 0 .. D - 1 */
    uint64_t bytes;              /* bytes the dictionary retains */
    uint64_t key_bytes;          /* KEYS: the bytes of every key; CODES: 0 */
    uint64_t digest[2];          /* digest of its descriptor */
    int64_t offset;              /* CODES: key(c) = c + offset */
    int32_t kind, scale;         /* TR_QUANT_DICT_*; CODES: decimal scale */
    uint32_t null_code, reserved;   /* TR_QUANT_NO_NULL or the NULL category's code */
} tr_quant_dict_info;            /* 64 bytes */

/* the arrays of a coverage state: pointers into the state (read only), or the caller's arrays for the CUDA path */
typedef struct {
    uint64_t ngroups, size, words;                    /* G, D, W = ceil(D / 64) */
    uint64_t *rows, *nulls, *parents;                 /* [G] */
    uint64_t *distinct, *matched, *missing, *extra;   /* [G] (see the header comment) */
    uint64_t *masks;                                  /* MASKS: G * W words, row-major; SETS: NULL */
    uint64_t *offsets;                                /* SETS: G + 1 offsets into codes; MASKS: NULL */
    uint32_t *codes;                                  /* SETS: offsets[G] codes, increasing within a group */
} tr_quant_cover_arrays;                              /* 104 bytes */

typedef struct {
    uint64_t ngroups, size, words, present, empty, absent;
    uint64_t rows, nulls, orphans, parent_rows;
    uint64_t pairs;              /* the sum of distinct[g]: distinct (group, category) pairs */
    uint64_t required, outside;  /* required codes in the dictionary (|R_in|); distinct required keys outside it */
    uint64_t divides;            /* present groups with missing = 0 */
    uint64_t bytes, identity_bytes;
    uint64_t digest[2];          /* digest of the identity (presence and requirement parts) */
    double masks_cost, sets_cost;   /* the costs of the representation choice (bytes each would store) */
    int32_t domain, repr;        /* group domain; TR_QUANT_REPR_MASKS / _SETS (the one built) */
    int32_t reason;              /* TR_CHOICE_CHEAPER / TR_CHOICE_ONLY_FIT (AUTO), -1 when forced or adopted */
    int32_t null_required;       /* 1: NULL is required (missing everywhere) */
    uint32_t null_code;          /* the dictionary's NULL code (TR_QUANT_NO_NULL) */
    int32_t parts;               /* child bags merged (1 for a build) */
} tr_quant_cover_info;           /* 184 bytes */

typedef struct tr_quant_groups_s tr_quant_groups;
typedef struct tr_quant_dict_s tr_quant_dict;
typedef struct tr_quant_cover_s tr_quant_cover;

/* ------------------------------------------------------------------ shared helpers (CPU code, CUDA host and kernels) */
TR_NUM_HD int tr_quant_popc(uint64_t x) {
#if defined(__CUDA_ARCH__)
    return __popcll((unsigned long long)x);
#else
    return __builtin_popcountll(x);
#endif
}
TR_NUM_HD int tr_quant_ctz(uint64_t x) {   /* x != 0 */
#if defined(__CUDA_ARCH__)
    return __ffsll((long long)x) - 1;
#else
    return __builtin_ctzll(x);
#endif
}
/* the answer of quantifier q for a group with t TRUE, f FALSE and u UNKNOWN child rows (see QUANT-001) */
TR_NUM_HD int tr_quant_answer_code(int32_t q, int present, uint64_t t, uint64_t f, uint64_t u) {
    if (!present) return TR_QUANT_ABSENT;
    switch (q) {
    case TR_QUANT_SOME: return t ? TR_QUANT_TRUE : u ? TR_QUANT_UNKNOWN : TR_QUANT_FALSE;
    case TR_QUANT_ALL: return f ? TR_QUANT_FALSE : u ? TR_QUANT_UNKNOWN : TR_QUANT_TRUE;
    case TR_QUANT_NONE: return t ? TR_QUANT_FALSE : u ? TR_QUANT_UNKNOWN : TR_QUANT_TRUE;
    case TR_QUANT_NOT_ALL: return f ? TR_QUANT_TRUE : u ? TR_QUANT_UNKNOWN : TR_QUANT_FALSE;
    case TR_QUANT_EXISTS: return t ? TR_QUANT_TRUE : TR_QUANT_FALSE;
    case TR_QUANT_NOT_EXISTS: return t ? TR_QUANT_FALSE : TR_QUANT_TRUE;
    case TR_QUANT_NO_COUNTEREXAMPLE: return f ? TR_QUANT_FALSE : TR_QUANT_TRUE;
    case TR_QUANT_ALL_TRUE: return (f || u) ? TR_QUANT_FALSE : TR_QUANT_TRUE;
    case TR_QUANT_BOOL_OR: return !(t || f) ? TR_QUANT_UNKNOWN : t ? TR_QUANT_TRUE : TR_QUANT_FALSE;
    default: return !(t || f) ? TR_QUANT_UNKNOWN : f ? TR_QUANT_FALSE : TR_QUANT_TRUE;   /* BOOL_AND */
    }
}
/* the answer of coverage question q for a group (see QUANT-002) */
TR_NUM_HD int tr_quant_cover_code(int32_t q, int present, uint64_t matched, uint64_t missing, uint64_t extra) {
    if (!present) return TR_QUANT_ABSENT;
    switch (q) {
    case TR_QUANT_DIVIDES: return missing ? TR_QUANT_FALSE : TR_QUANT_TRUE;
    case TR_QUANT_EXACTLY: return (missing || extra) ? TR_QUANT_FALSE : TR_QUANT_TRUE;
    case TR_QUANT_OVERLAPS: return matched ? TR_QUANT_TRUE : TR_QUANT_FALSE;
    default: return matched ? TR_QUANT_FALSE : TR_QUANT_TRUE;                              /* DISJOINT */
    }
}
/* present(g) under a domain */
TR_NUM_HD int tr_quant_present(int32_t domain, uint64_t rows, uint64_t parents) {
    return domain == TR_QUANT_IDS ? 1 : domain == TR_QUANT_PARENTS ? parents != 0 : rows != 0;
}
/* the predicate words of word w of the child view: T / F / U = the selected rows where the program is TRUE / FALSE /
 * UNKNOWN. reads: the conditions the root reads (deps[root]; the others are never evaluated); cv, ck: k words of
 * scratch for the conditions, val: 2 * nnodes words for the program (BOOL-001's word evaluator). */
TR_NUM_HD void tr_quant_predicate_word(const tr_scope_view *v, const tr_bool_condition *conds,
                                       const tr_bool_program_view *p, uint64_t reads, uint64_t w, uint64_t *cv,
                                       uint64_t *ck, uint64_t *val, uint64_t *T, uint64_t *F, uint64_t *U) {
    uint64_t sel = v->selection[w];
    for (int32_t i = 0; i < p->k; i++) {
        cv[i] = 0; ck[i] = 0;
        if ((reads >> i) & 1) tr_bool_condition_word(v, &conds[i], w, &cv[i], &ck[i]);
    }
    tr_bool_domain d;
    d.points = 64; d.nw = 1; d.selected = (uint64_t)tr_quant_popc(sel); d.kind = TR_BOOL_ROWS; d.k = p->k;
    d.selection = &sel; d.value = cv; d.valid = ck;
    tr_bool_eval_base(p, &d, 0, val);
    const uint64_t r = (uint64_t)p->root, value = val[2 * r] & sel, known = val[2 * r + 1] & sel;
    *T = value; *F = known & ~value; *U = sel & ~known;
}
/* the group of every selected row of word w when they share one (*uniform = 1), else *uniform = 0 */
TR_NUM_HD uint32_t tr_quant_word_group(const uint32_t *groups, uint64_t w, uint64_t sel, int *uniform) {
    if (!groups || !sel) { *uniform = 1; return 0; }
    const uint32_t *g = groups + w * 64;
    uint32_t first = g[tr_quant_ctz(sel)];
    for (uint64_t bits = sel; bits; bits &= bits - 1)
        if (g[tr_quant_ctz(bits)] != first) { *uniform = 0; return first; }
    *uniform = 1;
    return first;
}

/* ------------------------------------------------------------------ host-side sizes (both libraries) */
/* work items over nw words and their words per item: about TR_QUANT_WORK items, at least TR_QUANT_MIN_WORDS words
 * each, at most TR_QUANT_SCRATCH bytes of scratch (per_item bytes each) */
static inline uint64_t tr_quant_items(uint64_t nw, uint64_t per_item) {
    uint64_t s = TR_QUANT_WORK, cap = (nw + TR_QUANT_MIN_WORDS - 1) / TR_QUANT_MIN_WORDS;
    uint64_t room = per_item ? TR_QUANT_SCRATCH / per_item : s;
    if (s > cap) s = cap;
    if (s > room) s = room;
    return s ? s : 1;
}
/* bytes of the device copies of a scope view: its selection, its group ids and (cols: a bit per column) the codes or
 * values and the validity of those columns */
static inline uint64_t tr_quant_view_bytes(const tr_scope_view *v, uint64_t cols) {
    uint64_t bytes = tr_budget_mul((v->rows + 63) / 64, 8);
    if (v->groups) bytes = tr_budget_add(bytes, tr_budget_mul(v->rows, 4));
    for (int32_t c = 0; c < v->ncols && c < TR_SCOPE_MAX_COLUMNS; c++) {
        if (!((cols >> c) & 1)) continue;
        bytes = tr_budget_add(bytes, tr_budget_mul(v->rows, v->col[c].values ? 8 : 4));
        if (v->col[c].valid) bytes = tr_budget_add(bytes, v->rows);
    }
    return bytes;
}
/* the columns k conditions read (a bit per column) */
static inline uint64_t tr_quant_condition_columns(const tr_bool_condition *c, int32_t k) {
    uint64_t used = 0;
    for (int32_t i = 0; i < k; i++) {
        if (c[i].col >= 0 && c[i].col < 64) used |= (uint64_t)1 << c[i].col;
        if (c[i].op >= TR_BOOL_COL_EQ && c[i].col2 >= 0 && c[i].col2 < 64) used |= (uint64_t)1 << c[i].col2;
    }
    return used;
}
/* scratch words of one QUANT-001 work item: k condition value and valid words, 2 words per program node */
static inline uint64_t tr_quant_item_words(int32_t k, int32_t nnodes) {
    return 2 * (uint64_t)(k > 0 ? k : 0) + 2 * (uint64_t)(nnodes > 0 ? nnodes : 0);
}
/* the device bytes of tr_mgpu_quant_groups */
static inline uint64_t tr_quant_groups_device_bytes(const tr_scope_view *child, const tr_bool_condition *conds,
                                                    int32_t k, const tr_bool_program_view *p,
                                                    const tr_scope_view *parents) {
    uint64_t nw = (child->rows + 63) / 64, per = tr_budget_mul(tr_quant_item_words(k, p->nnodes), 8);
    uint64_t bytes = tr_quant_view_bytes(child, tr_quant_condition_columns(conds, k));
    bytes = tr_budget_add(bytes, tr_budget_mul((uint64_t)(k > 0 ? k : 0), sizeof(tr_bool_condition)));
    bytes = tr_budget_add(bytes, tr_budget_add(tr_budget_mul((uint64_t)p->nnodes, sizeof(tr_bool_node)),
                                               tr_budget_mul((uint64_t)p->nsched, 4)));
    bytes = tr_budget_add(bytes, tr_budget_mul(tr_quant_items(nw, per), per));       /* scratch */
    bytes = tr_budget_add(bytes, tr_budget_mul(child->ngroups, 40));                 /* the five arrays */
    if (parents) bytes = tr_budget_add(bytes, tr_quant_view_bytes(parents, 0));
    return bytes;
}
/* the device bytes of tr_mgpu_quant_cover (categories: only its group ids are copied) */
static inline uint64_t tr_quant_cover_device_bytes(const tr_scope_view *child, const tr_scope_view *categories,
                                                   const tr_scope_view *parents) {
    uint64_t W = (categories->ngroups + 63) / 64, G = child->ngroups;
    uint64_t bytes = tr_quant_view_bytes(child, 0);
    if (categories->groups) bytes = tr_budget_add(bytes, tr_budget_mul(categories->rows, 4));
    bytes = tr_budget_add(bytes, tr_budget_mul(tr_budget_mul(G, W), 8));            /* masks */
    bytes = tr_budget_add(bytes, tr_budget_mul(W, 8));                               /* the required mask */
    bytes = tr_budget_add(bytes, tr_budget_mul(G, 56));                              /* the seven arrays */
    if (parents) bytes = tr_budget_add(bytes, tr_quant_view_bytes(parents, 0));
    return bytes;
}
/* the device bytes of tr_mgpu_quant_cover_test */
static inline uint64_t tr_quant_test_device_bytes(uint64_t ngroups, uint64_t size) {
    uint64_t W = (size + 63) / 64;
    return tr_budget_add(tr_budget_mul(tr_budget_add(tr_budget_mul(ngroups, W), W), 8), tr_budget_mul(ngroups, 32));
}

/* ------------------------------------------------------------------ QUANT-001 (libalg) */
/* the needs of a build (TR_QUANT_NEED_CPU), of tr_mgpu_quant_groups (_DEVICE) or of an adoption (_ADOPT); the
 * arguments are validated as by the build */
int32_t tr_quant_groups_need(const tr_scope *child, const tr_bool_condition *conds, int32_t k,
                             const tr_bool_program *predicate, int32_t domain, const tr_scope *parents, int32_t target,
                             tr_math_need *need);
/* the quantifier states of the selected child rows (parents: the parent scope for TR_QUANT_PARENTS, else NULL) */
tr_quant_groups *tr_quant_groups_build(const tr_scope *child, const tr_bool_condition *conds, int32_t k,
                                       const tr_bool_program *predicate, int32_t domain, const tr_scope *parents,
                                       const tr_math_budget *, tr_math_refusal *, int32_t *error);
/* Adopts counts computed elsewhere (the readback of tr_mgpu_quant_groups) as the states of the same arguments;
 * checked: shape, qualifying + failing + unknown = rows per group, the totals equal the scopes' selected rows,
 * parents 0 past the parent ids and outside TR_QUANT_PARENTS. */
tr_quant_groups *tr_quant_groups_adopt(const tr_scope *child, const tr_bool_condition *conds, int32_t k,
                                       const tr_bool_program *predicate, int32_t domain, const tr_scope *parents,
                                       const tr_quant_counts *counts, const tr_math_budget *, tr_math_refusal *,
                                       int32_t *error);
/* The bag union of two states of the same predicate (conditions and nodes byte for byte), domain, parents and group
 * count (the child scopes may differ: e.g. two partitions of the children; their columns must be in the same order,
 * the caller's contract): counts add exactly (TR_MATH_OVERFLOW past 2^64 - 1), parents are those of a. Its identity is
 * that of the union: a tag, then the identities of a and b in order. */
tr_quant_groups *tr_quant_groups_merge(const tr_quant_groups *a, const tr_quant_groups *b, const tr_math_budget *,
                                       tr_math_refusal *, int32_t *error);
int32_t tr_quant_groups_get_info(const tr_quant_groups *, tr_quant_groups_info *);
int32_t tr_quant_groups_get_counts(const tr_quant_groups *, tr_quant_counts *);
/* one answer code per group (capacity >= ngroups) / the number of groups with each code (out[4], by code) */
int32_t tr_quant_groups_answer(const tr_quant_groups *, int32_t quantifier, uint8_t *out, uint64_t capacity);
int32_t tr_quant_groups_tally(const tr_quant_groups *, int32_t quantifier, uint64_t out[4]);
uint64_t tr_quant_groups_identity_size(const tr_quant_groups *);
int32_t tr_quant_groups_identity(const tr_quant_groups *, uint8_t *out, uint64_t capacity);
/* 1 when the identities are equal byte for byte, 0 otherwise */
int32_t tr_quant_groups_same(const tr_quant_groups *, const tr_quant_groups *);
/* 1 when the state is the build of exactly these arguments (identity equality), 0 otherwise */
int32_t tr_quant_groups_matches(const tr_quant_groups *, const tr_scope *child, const tr_bool_condition *conds,
                                int32_t k, const tr_bool_program *predicate, int32_t domain, const tr_scope *parents);
void tr_quant_groups_free(tr_quant_groups *);

/* ------------------------------------------------------------------ QUANT-002 dictionaries (libalg) */
/* CODES: D codes, key(c) = c + offset (every key fits int64), decimal scale 0..18 */
tr_quant_dict *tr_quant_dict_codes(uint64_t size, uint32_t null_code, int64_t offset, int32_t scale,
                                   const tr_math_budget *, tr_math_refusal *, int32_t *error);
/* KEYS: entries->n = D keys in code order, distinct (TR_MATH_INVALID otherwise); the NULL code's entry is empty and
 * ignored; null_key 0 */
tr_quant_dict *tr_quant_dict_keys(uint64_t size, uint32_t null_code, const tr_quant_keys *entries,
                                  const tr_math_budget *, tr_math_refusal *, int32_t *error);
int32_t tr_quant_dict_need(int32_t kind, uint64_t size, const tr_quant_keys *entries, tr_math_need *need);
int32_t tr_quant_dict_get_info(const tr_quant_dict *, tr_quant_dict_info *);
/* 1 and *code when key[0..len) is the key of a code, 0 when it is not in the dictionary */
int32_t tr_quant_dict_find(const tr_quant_dict *, const uint8_t *key, uint64_t len, uint32_t *code);
/* a required set against the dictionary: mask (words >= ceil(D / 64), or NULL) gets the required codes, *required
 * their number, *constant_missing the distinct outside keys plus the NULL requirement. Budget: bytes = 12 per key (the
 * resolution scratch). */
int32_t tr_quant_dict_mask(const tr_quant_dict *, const tr_quant_keys *required, uint64_t *mask, uint64_t words,
                           uint64_t *required_codes, uint64_t *constant_missing, const tr_math_budget *,
                           tr_math_refusal *);
/* 1 when the descriptors are equal byte for byte */
int32_t tr_quant_dict_same(const tr_quant_dict *, const tr_quant_dict *);
void tr_quant_dict_free(tr_quant_dict *);

/* ------------------------------------------------------------------ QUANT-002 coverage states (libalg) */
/* the needs of a build of representation repr (MASKS or SETS) on the CPU, of tr_mgpu_quant_cover (_DEVICE, MASKS)
 * or of an adoption (_ADOPT, MASKS) */
int32_t tr_quant_cover_need(const tr_scope *child, const tr_scope *categories, const tr_quant_dict *dict,
                            const tr_quant_keys *required, int32_t domain, const tr_scope *parents, int32_t repr,
                            int32_t target, tr_math_need *need);
tr_quant_cover *tr_quant_cover_build(const tr_scope *child, const tr_scope *categories, const tr_quant_dict *dict,
                                     const tr_quant_keys *required, int32_t domain, const tr_scope *parents,
                                     int32_t repr, const tr_math_budget *, tr_math_refusal *, int32_t *error);
/* Adopts the arrays of tr_mgpu_quant_cover (MASKS) after checks: canonical masks (no bit past D, the NULL code's bit
 * clear), the totals, nulls <= rows, distinct = the popcount of the group's mask <= rows - nulls, matched <=
 * min(distinct, |R_in|), extra = distinct - matched, missing = |R_in| - matched + constant_missing, parents. */
tr_quant_cover *tr_quant_cover_adopt(const tr_scope *child, const tr_scope *categories, const tr_quant_dict *dict,
                                     const tr_quant_keys *required, int32_t domain, const tr_scope *parents,
                                     const tr_quant_cover_arrays *device, const tr_math_budget *, tr_math_refusal *,
                                     int32_t *error);
/* The state of another required set over the same presence states (no rescan; the dictionary of the cover). summaries:
 * NULL (the coverage test on the CPU) or the distinct / matched / missing / extra arrays of tr_mgpu_quant_cover_test,
 * adopted after the checks above. */
tr_quant_cover *tr_quant_cover_require(const tr_quant_cover *, const tr_quant_keys *required,
                                       const tr_quant_cover_arrays *summaries, const tr_math_budget *,
                                       tr_math_refusal *, int32_t *error);
/* The bag union of two states with the same dictionary, required set, domain, parents, group count and
 * representation: rows and nulls add exactly, masks OR / sets merge (a category present in either is present),
 * parents are those of a, the coverage is tested again. */
tr_quant_cover *tr_quant_cover_merge(const tr_quant_cover *a, const tr_quant_cover *b, const tr_math_budget *,
                                     tr_math_refusal *, int32_t *error);
int32_t tr_quant_cover_get_info(const tr_quant_cover *, tr_quant_cover_info *);
int32_t tr_quant_cover_get_arrays(const tr_quant_cover *, tr_quant_cover_arrays *);
/* the required mask (MASKS: words >= W; SETS: TR_MATH_UNSUPPORTED) for tr_mgpu_quant_cover_test */
int32_t tr_quant_cover_required_mask(const tr_quant_cover *, uint64_t *mask, uint64_t words);
/* MASKS: the masks as a FIELD-001 matrix view (rows G, cols D, stride W; read only, valid while the state lives) */
int32_t tr_quant_cover_masks(const tr_quant_cover *, tr_field_matrix *view);
/* the categories of group g (TR_QUANT_LIST_PRESENT, _MISSING among the required codes, _EXTRA) in increasing code:
 * min(count, capacity) written, *count = all of them */
int32_t tr_quant_cover_list(const tr_quant_cover *, uint64_t group, int32_t which, uint32_t *out, uint64_t capacity,
                            uint64_t *count);
/* the required codes (increasing) and the distinct outside keys (sorted bytewise) of the state: min(n, capacity)
 * written, *n = all; key i of the outside keys: bytes[offsets[i] .. offsets[i + 1]) */
int32_t tr_quant_cover_required_codes(const tr_quant_cover *, uint32_t *out, uint64_t capacity, uint64_t *n);
int32_t tr_quant_cover_outside(const tr_quant_cover *, uint8_t *bytes, uint64_t bytes_capacity, uint64_t *offsets,
                               uint64_t offsets_capacity, uint64_t *n, uint64_t *total_bytes);
int32_t tr_quant_cover_answer(const tr_quant_cover *, int32_t question, uint8_t *out, uint64_t capacity);
int32_t tr_quant_cover_tally(const tr_quant_cover *, int32_t question, uint64_t out[4]);
uint64_t tr_quant_cover_identity_size(const tr_quant_cover *);
int32_t tr_quant_cover_identity(const tr_quant_cover *, uint8_t *out, uint64_t capacity);
int32_t tr_quant_cover_same(const tr_quant_cover *, const tr_quant_cover *);
int32_t tr_quant_cover_matches(const tr_quant_cover *, const tr_scope *child, const tr_scope *categories,
                               const tr_quant_dict *dict, const tr_quant_keys *required, int32_t domain,
                               const tr_scope *parents);
void tr_quant_cover_free(tr_quant_cover *);

/* ------------------------------------------------------------------ the CUDA library (math_quant_gpu.cu, libmathgpu)
 * Views of host memory (tr_scope_get_view, tr_bool_program_get_view), copied to the device and budgeted (bytes)
 * before anything is allocated; outputs are host arrays, overwritten:
 *   tr_mgpu_quant_groups      the QUANT-001 counts (out->ngroups = child ngroups; parents: NULL or the parent view)
 *   tr_mgpu_quant_cover       the MASKS coverage arrays (out: G, D = categories->ngroups, W, every array) for the
 *                             required mask (W words holding exactly required_codes = |R_in| codes, checked; the
 *                             constant_missing outside keys and NULL requirement) and parents
 *   tr_mgpu_quant_cover_test  the coverage test of masks (in->masks, G, D, W) against a required mask: out->distinct,
 *                             matched, missing, extra */
int tr_mgpu_quant_groups(const tr_scope_view *child, const tr_bool_condition *conds, int32_t k,
                         const tr_bool_program_view *predicate, const tr_scope_view *parents, const tr_math_budget *,
                         tr_quant_counts *out, tr_math_refusal *);
int tr_mgpu_quant_cover(const tr_scope_view *child, const tr_scope_view *categories, uint32_t null_code,
                        const uint64_t *required, uint64_t required_codes, uint64_t constant_missing,
                        const tr_scope_view *parents, const tr_math_budget *, tr_quant_cover_arrays *out,
                        tr_math_refusal *);
int tr_mgpu_quant_cover_test(const tr_quant_cover_arrays *in, const uint64_t *required, uint64_t required_codes,
                             uint64_t constant_missing, const tr_math_budget *, tr_quant_cover_arrays *out,
                             tr_math_refusal *);
#ifdef __cplusplus
}
#endif
#endif
