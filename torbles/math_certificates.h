/* FND-007: proof certificates and their evidence scope.
 * SPDX-License-Identifier: Apache-2.0
 *
 * One native certificate type (tr_cert) for four kinds of claims, never confused (the kind is fixed at creation and
 * decides every later check):
 *   TR_CERT_LAW         a universal law: the claim holds by mathematics, with the law named (tr_cert_law_describe).
 *                       No scope; a data change never touches it. Laws are decided natively at creation:
 *                         RING_IDENTITY   a polynomial whose canonical form in Z[x] is zero (every coefficient of
 *                                         every monomial cancels exactly): holds in every commutative ring (Z, Q,
 *                                         every GF(p)), e.g. x*y - y*x, (x + y)^2 - x^2 - 2xy - y^2.
 *                         GFP_FUNCTION    a polynomial over GF(p) that vanishes at every point of GF(p)^n: its
 *                                         reduction by x^p = x (exponent e >= 1 -> ((e - 1) mod (p - 1)) + 1) and
 *                                         mod p is zero (a complete decision procedure; GF(2): Boolean identities
 *                                         such as x^2 - x). Holds only in GF(p).
 *   TR_CERT_SCHEMA      an enforced schema fact: a declared constraint (UNIQUE key, NOT NULL, a foreign key as a
 *                       REFERENCE claim) that the native code verified on every selected row of its table when the
 *                       certificate was made, together with who enforces it from then on (enforcement):
 *                         ENFORCE_LOAD   verified at load (torbles: database.py refuses a PRIMARY KEY with a NULL
 *                                        or a duplicate; a table has no update path): the fact holds for the
 *                                        revision it was verified on, and for every subset of those rows.
 *                         ENFORCE_WRITE  the storage checks every write: the fact holds for every revision of the
 *                                        same table identity.
 *                       Constraints are hereditary: a fact verified on a selection holds on every sub-selection.
 *   TR_CERT_SNAPSHOT    an exhaustive snapshot certificate: the claim was verified on every selected row of an
 *                       exact scope (tr_cert_record: the table identity and revision, the row count, the exact
 *                       selection bits, the NULL domain of each claim column, the grouping a group claim reads, the
 *                       join multiplicity of every dimension it reaches through) in a stated arithmetic carrier.
 *                       Created only when the evidence is complete and has no violation.
 *   TR_CERT_HYPOTHESIS  a statistical hypothesis: the claim was checked on a sample (tr_cert_sample: method, seed,
 *                       size, population) and held there. Never exact; never authorizes an exact rewrite.
 *
 * Claims (tr_cert_claim, a typed payload; one entry per kind in a constant table, tr_cert_claim_describe; a new kind
 * adds one enum value, one table entry and one case of tr_cert_row). Columns are scope columns (col[]); a law's are
 * variables 0..ncols-1 that any binding may instantiate. col[0] is the output of the kinds that have one.
 *   AFFINE      col0 = scale * col1 + bias                                   (DISC-001)
 *   SUM         col0 = col1 + col2                                           (DISC-001)
 *   PRODUCT     col0 = col1 * col2                                           (DISC-001)
 *   PARITY      XOR over the bit variables j in mask of bit_j = constant     (FIELD-001: bits[j] is a NONZERO or
 *               PLANE bit of column col[bits[j].col], PLANE = bit `bit` of (logical - base) mod 2^64)
 *   REFINEMENT  col0 = map(col1): keys[i] (strictly increasing) -> targets[i]; a value of col1 that is not a key
 *               violates (the map must be total on the data)                  (MULTI-001 refinement maps as a
 *               functional dependency of key values)
 *   COVERAGE    every present group (a group with selected rows) holds every required category keys[0..nvalues) in
 *               col0 (nvalues <= 64; NULL categories never count)             (QUANT-002 coverage / division)
 *   POLYNOMIAL  sum_t coefficients[t] * prod_v col_v^exponents[t * ncols + v] = 0 over named columns, canonical
 *               terms (merged, nonzero, sorted)                               (DISC-002, GEOM-001)
 *   UNIQUE      no two selected present rows share a col0 value (a key); a violation is a row after the first
 *               occurrence of its value (violations = present rows - distinct values)
 *   NOT_NULL    no selected row has col0 NULL
 *   REFERENCE   col0 (a foreign-key code: a dimension row id) names a used row of the dimension: 0 <= v < nvalues
 *               and bit v of used (the dimension's used rows)
 * The claim states where it holds: field 0 = Z (exact integers), field p = GF(p) (PARITY: GF(2)). It also states its
 * check: carrier TR_CARRIER_I128 (exact 128-bit integer evaluation; a polynomial term or sum outside int128 is
 * TR_CERT_ROW_OVERFLOW, never wrapped) or TR_CARRIER_GFP (evaluation modulo the prime given in gf.p, or the claim's
 * field; tr_cert_claim_check fills the rest of gf), and a tolerance (0 exact; > 0: |residual| <= tolerance, an APPROXIMATE claim). The check is
 * exact for the claim (tr_cert_info.exact) iff tolerance = 0 and either carrier I128 and field 0, or carrier GFP
 * with modulus = field. A Z claim checked modulo p is approximate (it holds mod p only).
 *
 * NULL policy (part of the claim), on a selected row where some claim column is NULL (PARITY: a column of a
 * variable of mask):
 *   FORBID      the row violates (the claim asserts no NULL)
 *   SKIP        the row is excluded (SQL: WHERE every column IS NOT NULL); the claim covers the complete rows only
 *   PROPAGATE   (kinds with an output) the row holds iff the output is NULL exactly when an input is NULL
 * Allowed policies per kind: tr_cert_claim_desc.policies (COVERAGE: SKIP; NOT_NULL: FORBID; UNIQUE, REFERENCE:
 * FORBID or SKIP; PARITY, POLYNOMIAL: FORBID or SKIP; AFFINE, SUM, PRODUCT, REFINEMENT: all three).
 *
 * Evidence (tr_cert_evidence), exhaustive over the selected rows, independent of the order rows are visited in:
 * checked (rows, COVERAGE: present groups, decided), excluded (selected rows outside the domain by SKIP), violations
 * and witness = the smallest violating row (COVERAGE: group; UNIQUE: the smallest row that is not the first
 * occurrence of its value), overflow (rows undecided in 128 bits), holds = no overflow and no violation.
 *
 * Records (tr_cert_record): an owned, address-free description of a scope (math_scope.h): the source's lifetime
 * token, revision, row count, selected rows, the FND-004 version digest (revision, rows and sidecar revisions; it
 * covers the sidecars' addresses, so it is process-specific: a record never matches a table of another process), the
 * selection digest and (TR_CERT_RECORD_SELECTION) the exact selection bits, every recorded column's encoding (stored
 * column, validity column, high part, shift, offset, scale) and NULL domain (NULL count and digest of the NULL rows
 * among the selected rows), the grouping descriptor (TR_CERT_RECORD_GROUPS: FND-004's TR_SCOPE_KEY_GROUPS bytes,
 * exact) and the dimensions reached through (tr_cert_dim: lifetime token, revision, rows, used rows and their digest,
 * the key column, whether the key is unique and NULL-free among the used rows). An array scope has no source: its
 * identity is the 128-bit digest of its data (FND-004's descriptor digest). Digests are pre-filters; exact parts
 * (selection bits, grouping bytes) are compared byte for byte when kept.
 *
 * Freshness (tr_cert_fresh, tr_cert_record_compare): TR_MATH_OK, or TR_MATH_STALE with every reason that applies
 * (tr_cert_staleness.reasons) and the first one in this order:
 *   IDENTITY       another source (lifetime token, source kind) or, for an array scope, other data
 *   REVISION       the table was mutated (revision or a sidecar's revision: the version digest)
 *   ROWS           the physical row count changed
 *   SELECTION      the selected rows differ (SCHEMA facts: a current row outside the verified selection)
 *   NULL_DOMAIN    a claim column's NULL rows among the selected rows or its validity column changed
 *   COLUMNS        a claim column's encoding changed (stored column, offset, scale, high part), or it is missing
 *   GROUPING       the grouping a group claim reads changed
 *   JOIN_REVISION  a dimension reached through changed: another revision, rows or used rows, or it is absent
 *   JOIN_KEY       a dimension's key is no longer unique or no longer NULL-free: the join multiplicity changed
 * by kind: LAW always fresh; SNAPSHOT and HYPOTHESIS exact (every part equal); SCHEMA hereditary (ENFORCE_LOAD:
 * identity, revision, rows, columns, dimensions, and the current selection a subset of the verified one;
 * ENFORCE_WRITE: identity, rows need not match, columns and dimension identities and keys). A derived certificate
 * is fresh when every premise is (reasons OR-ed). Only the dimensions recorded are checked: a change of another
 * dimension does not apply.
 *
 * Authorization (tr_cert_authorize_exact): TR_MATH_OK (decision.authorized = 1) only for a universal law that holds
 * in the target (field 0 = Z or p = GF(p): RING_IDENTITY everywhere, GFP_FUNCTION only in its GF(p)), an enforced
 * schema fact fresh for the current scope, or a snapshot certificate that is exact, complete and fresh (a Z claim
 * also authorizes a GF(p) target: Z -> GF(p) is a ring homomorphism; a GF(p) claim only GF(p)). Refusals: a
 * hypothesis (STATISTICAL), an approximate claim or check (APPROXIMATE), a target the claim does not hold in
 * (CARRIER), a missing current scope (NO_SCOPE), a stale scope (STALE, TR_MATH_STALE), a refused premise (PREMISE);
 * all but STALE return TR_MATH_DOMAIN_ERROR. decision.policy is the NULL domain the rewrite holds on (SKIP: the
 * complete rows only).
 *
 * Derivation (tr_cert_derive): a POLYNOMIAL claim P derived from premises G_i (POLYNOMIAL, AFFINE, SUM, PRODUCT
 * claims, as the polynomials col0 - scale col1 - bias, col0 - col1 - col2, col0 - col1 col2) with cofactor
 * polynomials q_i: P - sum_i q_i G_i is verified to be zero in Z[x] (an ideal-membership certificate, e.g. a
 * reduction step of GEOM-001). Variables are matched by scope column. The derived certificate owns copies of its
 * premises; its kind is the weakest premise's (LAW < SCHEMA < SNAPSHOT < HYPOTHESIS), its freshness and
 * authorization those of every premise.
 *
 * Serialization (tr_cert_serialize / tr_cert_deserialize, tr_cert_record_*): canonical little-endian bytes, no
 * address; FND-011 (math_persist.h) wraps them into versioned, checksummed records.
 *
 * Budgets (math_budgets.h), checked before anything is allocated (TR_MATH_BUDGET with the tr_math_refusal):
 *   claims: variables = ncols, degree = the polynomial's total degree, support = payload entries (terms, map
 *     entries, required values, dimension rows / 64), output = claims verified;
 *   verification: states = COVERAGE groups (one mask word each), bytes = the scratch (COVERAGE masks and presence,
 *     UNIQUE: 16 bytes per selected row to sort; on the device: the copies, 8 bytes per value of a UNIQUE column's
 *     [min, max] domain, the masks and accumulators);
 *   records and certificates: bytes = the objects, the payload copy, the selection bits, the grouping bytes;
 *   dimensions: bytes = the decoded key and validity (4 + 4 bytes per row) and the sort scratch (4 per used row);
 *   samples: support = sample size, bytes = the rank directory (8 per selection word) and the drawn rows.
 *
 * Errors: TR_MATH_INVALID (malformed claim, scope, bytes), _MEMORY, _OVERFLOW (an undecided row: evidence.overflow),
 * _BUDGET, _UNSUPPORTED (a representation limit), _CARRIER_ERROR (a GF(p) claim checked in another carrier),
 * TR_MATH_DOMAIN_ERROR (the claim does not hold: evidence.witness; a law that is not an instance; a refusal),
 * TR_MATH_STALE. Objects are immutable and individually owned (constructors return NULL and set *error, inputs
 * unchanged); read-only calls may run concurrently.
 *
 * The CUDA library (math_certificates_gpu.cu): tr_mgpu_cert_verify re-verifies one or many claims over the selected
 * rows of one scope view: work items (claim, contiguous slice of rows) accumulate locally and add their counts with
 * atomics and their smallest witness with an atomic minimum (UNIQUE: first occurrences by atomic minimum over the
 * value domain, then a second pass; COVERAGE: per-group masks by atomic OR, then one work item per group). Equal to
 * tr_cert_verify_view bit for bit; tr_cert_adopt turns its evidence into a snapshot certificate. */
#ifndef TORBLES_MATH_CERTIFICATES_H
#define TORBLES_MATH_CERTIFICATES_H
#include <stdint.h>
#include <string.h>
#include "math_num.h"
#include "math_budgets.h"
#include "math_scope.h"
#include "math_carriers.h"
#include "math_handles.h"
#ifdef __cplusplus
extern "C" {
#endif

#define TR_CERT_ABI 1
#define TR_CERT_MAX_COLS 8                 /* columns one claim reads */
#define TR_CERT_MAX_BITS 64                /* bit variables of a parity claim */
#define TR_CERT_MAX_REQUIRED 64            /* required categories of a coverage claim (one mask word per group) */
#define TR_CERT_MAX_DIMS 8                 /* dimensions a record reaches through */
#define TR_CERT_MAX_PREMISES 16            /* premises of a derived certificate */
#define TR_CERT_MAX_DEPTH 4                /* nesting of derived certificates */
#define TR_CERT_MAX_TERMS 4096             /* terms of a polynomial claim (representation limit) */
#define TR_CERT_NONE UINT64_MAX            /* no witness */

enum { TR_CERT_LAW = 1, TR_CERT_SCHEMA = 2, TR_CERT_SNAPSHOT = 3, TR_CERT_HYPOTHESIS = 4 };   /* kinds */
enum { TR_CLAIM_AFFINE = 1, TR_CLAIM_SUM = 2, TR_CLAIM_PRODUCT = 3, TR_CLAIM_PARITY = 4, TR_CLAIM_REFINEMENT = 5,
       TR_CLAIM_COVERAGE = 6, TR_CLAIM_POLYNOMIAL = 7, TR_CLAIM_UNIQUE = 8, TR_CLAIM_NOT_NULL = 9,
       TR_CLAIM_REFERENCE = 10, TR_CLAIM_KINDS = 11 };
enum { TR_CERT_NULL_FORBID = 1, TR_CERT_NULL_SKIP = 2, TR_CERT_NULL_PROPAGATE = 3 };
enum { TR_CERT_LAW_RING_IDENTITY = 1, TR_CERT_LAW_GFP_FUNCTION = 2, TR_CERT_LAWS = 3 };
enum { TR_CERT_ENFORCE_LOAD = 1, TR_CERT_ENFORCE_WRITE = 2 };
enum { TR_CERT_SAMPLE_UNIFORM = 1,         /* size ranks drawn with replacement: rank = hi64(splitmix64(seed, i) x N) */
       TR_CERT_SAMPLE_STRIDE = 2,          /* ranks floor(i N / size), i < size (deterministic, evenly spread) */
       TR_CERT_SAMPLE_PREFIX = 3 };        /* the first size selected rows (a biased sample, e.g. LIMIT) */
enum { TR_CERT_UNIT_ROWS = 1, TR_CERT_UNIT_GROUPS = 2 };
enum { TR_CERT_BIT_NONZERO = 1, TR_CERT_BIT_PLANE = 2 };                       /* FIELD-001's bit specs */
enum { TR_CERT_ROW_EXCLUDED = 1, TR_CERT_ROW_HOLDS = 2, TR_CERT_ROW_VIOLATES = 3, TR_CERT_ROW_OVERFLOW = 4 };
enum { TR_CERT_BY_CPU = 0, TR_CERT_BY_DEVICE = 1 };   /* who verified a snapshot (DEVICE: the CUDA library, a device
                                                         or its emulation, as the caller of tr_cert_adopt states) */
enum { TR_CERT_RECORD_SELECTION = 1, TR_CERT_RECORD_GROUPS = 2 };              /* record flags */
enum { TR_CERT_MATCH_EXACT = 1, TR_CERT_MATCH_SUBSET = 2, TR_CERT_MATCH_SCHEMA = 3 };   /* comparison modes */
enum { TR_CERT_STALE_IDENTITY = 1 << 0, TR_CERT_STALE_REVISION = 1 << 1, TR_CERT_STALE_ROWS = 1 << 2,
       TR_CERT_STALE_SELECTION = 1 << 3, TR_CERT_STALE_NULL_DOMAIN = 1 << 4, TR_CERT_STALE_COLUMNS = 1 << 5,
       TR_CERT_STALE_GROUPING = 1 << 6, TR_CERT_STALE_JOIN_REVISION = 1 << 7, TR_CERT_STALE_JOIN_KEY = 1 << 8 };
enum { TR_CERT_OK = 0, TR_CERT_REFUSE_STATISTICAL = 1, TR_CERT_REFUSE_APPROXIMATE = 2, TR_CERT_REFUSE_CARRIER = 3,
       TR_CERT_REFUSE_STALE = 4, TR_CERT_REFUSE_INCOMPLETE = 5, TR_CERT_REFUSE_NO_SCOPE = 6,
       TR_CERT_REFUSE_PREMISE = 7 };
enum { TR_CERT_NEED_CPU = 0, TR_CERT_NEED_DEVICE = 1 };

/* ------------------------------------------------------------------ structs crossing the ctypes boundary */
typedef struct { int32_t col, kind, bit, reserved; int64_t base; } tr_cert_bit;   /* col: index into claim col[];
                                                                                     24 bytes */
typedef struct {
    int32_t kind, policy;              /* TR_CLAIM_*, TR_CERT_NULL_* */
    int32_t ncols, nbits;              /* columns read (1..8); PARITY: bit variables */
    int32_t col[TR_CERT_MAX_COLS];     /* scope columns (a law: variables 0..ncols-1) */
    int32_t carrier, constant;         /* the check: TR_CARRIER_I128 or TR_CARRIER_GFP; PARITY: the XOR's value */
    uint64_t field;                    /* where the claim holds: 0 = Z, p = GF(p) */
    uint64_t tolerance;                /* 0 exact; > 0 approximate: |residual| <= tolerance */
    int64_t scale, bias;               /* AFFINE */
    uint64_t mask;                     /* PARITY: the relation's variables (bit j: bits[j]) */
    uint64_t nterms, nvalues;          /* POLYNOMIAL terms; REFINEMENT entries / COVERAGE required / REFERENCE rows */
    tr_gfp gf;                         /* TR_CARRIER_GFP: gf.p = the modulus; tr_cert_claim_check fills the rest */
    const tr_cert_bit *bits;           /* PARITY: nbits */
    const uint8_t *exponents;          /* POLYNOMIAL: nterms x ncols, row-major */
    const int64_t *coefficients;       /* POLYNOMIAL: nterms */
    const int64_t *keys;               /* REFINEMENT: keys; COVERAGE: required values (both strictly increasing) */
    const int64_t *targets;            /* REFINEMENT: the value of each key */
    const uint64_t *used;              /* REFERENCE: ceil(nvalues / 64) words, the dimension's used rows */
} tr_cert_claim;                       /* 192 bytes */

typedef struct {
    uint64_t rows, selected;           /* the scope's physical / selected rows (a hypothesis: rows drawn) */
    uint64_t checked, excluded;        /* rows (COVERAGE: present groups) decided; selected rows excluded by SKIP */
    uint64_t violations, witness;      /* failing rows (groups); the smallest one or TR_CERT_NONE */
    uint64_t overflow;                 /* rows undecided in 128 bits */
    int32_t unit, holds;               /* TR_CERT_UNIT_*; 1: no overflow and no violation */
} tr_cert_evidence;                    /* 64 bytes */

typedef struct {
    int32_t method, reserved;          /* TR_CERT_SAMPLE_* */
    uint64_t size, seed;               /* rows drawn; the sampler's seed */
    uint64_t population;               /* selected rows of the scope sampled */
    uint64_t distinct;                 /* distinct rows drawn (set by the hypothesis) */
} tr_cert_sample;                      /* 40 bytes */

typedef struct {
    uint32_t reasons;                  /* TR_CERT_STALE_* that apply */
    int32_t first;                     /* the first reason (0: fresh) */
    int32_t column, dim;               /* the first claim column / dimension involved (-1) */
    uint64_t recorded, current;        /* the first reason's quantity (revision, rows, selected rows, NULL count, ...) */
} tr_cert_staleness;                   /* 32 bytes */

typedef struct {
    int32_t authorized, reason;        /* 1 / 0; TR_CERT_OK or TR_CERT_REFUSE_* */
    int32_t kind, claim;               /* the certificate's */
    int32_t policy, premise;           /* the NULL domain the rewrite holds on; the refused premise (-1) */
    uint64_t field;                    /* the target: 0 = Z, p = GF(p) */
    tr_cert_staleness stale;
} tr_cert_decision;                    /* 64 bytes */

typedef struct {
    int32_t kind, claim, law, enforcement;
    int32_t policy, carrier, exact, backend;   /* backend: TR_CERT_BY_* */
    uint64_t field, modulus, tolerance;
    int32_t scoped, npremises, depth, reserved;
    uint64_t bytes;                    /* bytes owned (premises included) */
    uint64_t digest[2];                /* of the canonical serialization (a pre-filter) */
    tr_cert_evidence evidence;
    tr_cert_sample sample;
} tr_cert_info;                        /* 200 bytes */

typedef struct {
    int32_t kind, min_cols, max_cols, unit;
    int32_t policies;                  /* bit p: TR_CERT_NULL_p is allowed */
    int32_t output;                    /* 1: col[0] is the output (PROPAGATE) */
    int32_t local;                     /* 1: decided row by row */
    int32_t field;                     /* 0: a Z claim; 2: a GF(2) claim (PARITY) */
    char name[32];
} tr_cert_claim_desc;                  /* 64 bytes */

typedef struct {
    int32_t law, claim;                /* the claim kind it decides */
    int32_t any_ring, reserved;        /* 1: holds in every commutative ring; 0: only in GF(field) */
    char name[48];
    char statement[112];
} tr_cert_law_desc;                    /* 176 bytes */

typedef struct {
    int32_t index;                     /* the scope column */
    int32_t col, valid, hi;            /* stored column, validity column, high part (-1: none / an array scope) */
    int32_t shift, scale, storage, reserved;
    int64_t offset;
    uint64_t nulls;                    /* selected rows where the value is NULL */
    uint64_t null_digest[2];           /* digest of those rows */
} tr_cert_column;                      /* 64 bytes */

typedef struct {
    uint8_t identity[16];              /* the dimension's lifetime token */
    uint64_t revision, rows, used;     /* its revision, physical rows, used rows */
    uint64_t used_digest[2];           /* digest of its used-rows bitmap */
    uint64_t duplicates, nulls;        /* used rows after the first occurrence of their key; used rows with a NULL key */
    int32_t key, key_valid;            /* the key column, its validity column (-1) */
    int32_t unique, reserved;          /* 1: duplicates == 0 and nulls == 0 (join multiplicity <= 1) */
} tr_cert_dim;                         /* 88 bytes */

typedef struct {
    uint8_t identity[16];              /* the source's lifetime token (zero for an array scope) */
    uint64_t revision, rows, selected, ngroups;
    uint64_t version[2];               /* FND-004 version digest (process-specific) */
    uint64_t selection[2];             /* digest of the selection bits */
    uint64_t grouping[2];              /* digest of the grouping descriptor (0 when not bound) */
    uint64_t data[2];                  /* an array scope: its descriptor digest; a table scope: 0 */
    int32_t source, ncols, ndims, flags;   /* TR_SCOPE_TABLE / _ARRAYS; columns and dimensions recorded; TR_CERT_RECORD_* */
    uint64_t grouping_bytes;           /* bytes of the grouping descriptor kept */
    tr_cert_column col[TR_SCOPE_MAX_COLUMNS];
    tr_cert_dim dim[TR_CERT_MAX_DIMS];
} tr_cert_source;                      /* 2888 bytes */

typedef struct tr_cert_record_s tr_cert_record;
typedef struct tr_cert_s tr_cert;

/* ------------------------------------------------------------------ shared row semantics (CPU loops and kernels) */
TR_NUM_HD unsigned __int128 tr_cert_mag(__int128 x) {
    return x < 0 ? (unsigned __int128)0 - (unsigned __int128)x : (unsigned __int128)x;
}
/* x mod p in [0, p) for an exact 128-bit integer */
TR_NUM_HD uint64_t tr_cert_mod(const tr_gfp *f, __int128 x) {
    unsigned __int128 m = tr_cert_mag(x);
    uint64_t r = tr_gfp_reduce_u128_hd(f, (uint64_t)(m >> 64), (uint64_t)m);
    return x < 0 ? tr_gfp_neg_hd(f, r) : r;
}
/* the smallest i with keys[i] >= x (n when none) */
TR_NUM_HD uint64_t tr_cert_lower_bound(const int64_t *keys, uint64_t n, int64_t x) {
    uint64_t lo = 0, hi = n;
    while (lo < hi) { uint64_t mid = lo + (hi - lo) / 2; if (keys[mid] < x) lo = mid + 1; else hi = mid; }
    return lo;
}
/* a residual (an exact 128-bit integer) against the claim's check */
TR_NUM_HD int tr_cert_residual(const tr_cert_claim *c, __int128 r) {
    if (c->carrier == TR_CARRIER_GFP) return tr_cert_mod(&c->gf, r) == 0 ? TR_CERT_ROW_HOLDS : TR_CERT_ROW_VIOLATES;
    return tr_cert_mag(r) <= (unsigned __int128)c->tolerance ? TR_CERT_ROW_HOLDS : TR_CERT_ROW_VIOLATES;
}
/* bit j of the claim's parity variables at a row where its column is present */
TR_NUM_HD int tr_cert_bit_value(const tr_scope_view *v, const tr_cert_claim *c, int32_t j, uint64_t row) {
    const tr_cert_bit *b = &c->bits[j];
    int64_t x = tr_scope_value(v, c->col[b->col], row);
    if (b->kind == TR_CERT_BIT_NONZERO) return x != 0;
    return (int)((((uint64_t)x - (uint64_t)b->base) >> b->bit) & 1);
}
/* the exact value of a polynomial claim at a row (1), or 0 when a canonical term or the sum leaves int128 */
TR_NUM_HD int tr_cert_poly_value(const tr_scope_view *v, const tr_cert_claim *c, uint64_t row, __int128 *value) {
    int64_t x[TR_CERT_MAX_COLS];
    for (int32_t i = 0; i < c->ncols; i++) x[i] = tr_scope_value(v, c->col[i], row);
    if (c->carrier == TR_CARRIER_GFP) {
        uint64_t acc = 0, xm[TR_CERT_MAX_COLS];
        for (int32_t i = 0; i < c->ncols; i++) xm[i] = tr_gfp_reduce_i64_hd(&c->gf, x[i]);
        for (uint64_t t = 0; t < c->nterms; t++) {
            uint64_t term = tr_gfp_reduce_i64_hd(&c->gf, c->coefficients[t]);
            for (int32_t i = 0; i < c->ncols && term; i++) {
                uint8_t e = c->exponents[t * (uint64_t)c->ncols + (uint64_t)i];
                if (e) term = tr_gfp_mul_hd(&c->gf, term, tr_gfp_pow_hd(&c->gf, xm[i], e));
            }
            acc = tr_gfp_add_hd(&c->gf, acc, term);
        }
        *value = (__int128)acc;
        return 1;
    }
    tr_math_ipoly_view p;
    p.nterms = c->nterms; p.nvars = c->ncols; p.degree = 0;
    p.exponents = c->exponents; p.coefficients = c->coefficients;
    return tr_ipoly_point(&p, x, value);
}
/* One selected row (see the header comment): TR_CERT_ROW_EXCLUDED / _HOLDS / _VIOLATES / _OVERFLOW. COVERAGE: *bit =
 * the mask bit of the row's category among the required values (0: not required); UNIQUE: _HOLDS for a present
 * value (its occurrence is decided by the caller). */
TR_NUM_HD int tr_cert_row(const tr_scope_view *v, const tr_cert_claim *c, uint64_t row, uint64_t *bit) {
    *bit = 0;
    int out_null = 0, in_null = 0;
    if (c->kind == TR_CLAIM_PARITY) {
        for (int32_t j = 0; j < c->nbits; j++)
            if (((c->mask >> j) & 1) && !tr_scope_present(v, c->col[c->bits[j].col], row)) in_null = 1;
    } else {
        for (int32_t i = 0; i < c->ncols; i++) {
            if (tr_scope_present(v, c->col[i], row)) continue;
            if (i == 0 && (c->kind == TR_CLAIM_AFFINE || c->kind == TR_CLAIM_SUM || c->kind == TR_CLAIM_PRODUCT ||
                           c->kind == TR_CLAIM_REFINEMENT)) out_null = 1;
            else in_null = 1;
        }
    }
    if (out_null || in_null) {
        if (c->policy == TR_CERT_NULL_SKIP) return TR_CERT_ROW_EXCLUDED;
        if (c->policy == TR_CERT_NULL_PROPAGATE) return out_null == in_null ? TR_CERT_ROW_HOLDS : TR_CERT_ROW_VIOLATES;
        return TR_CERT_ROW_VIOLATES;
    }
    switch (c->kind) {
    case TR_CLAIM_AFFINE: {
        __int128 out = tr_scope_value(v, c->col[0], row), x = tr_scope_value(v, c->col[1], row);
        return tr_cert_residual(c, out - ((__int128)c->scale * x + c->bias));
    }
    case TR_CLAIM_SUM: {
        __int128 out = tr_scope_value(v, c->col[0], row);
        return tr_cert_residual(c, out - ((__int128)tr_scope_value(v, c->col[1], row) + tr_scope_value(v, c->col[2], row)));
    }
    case TR_CLAIM_PRODUCT: {
        __int128 out = tr_scope_value(v, c->col[0], row);
        return tr_cert_residual(c, out - (__int128)tr_scope_value(v, c->col[1], row) * tr_scope_value(v, c->col[2], row));
    }
    case TR_CLAIM_PARITY: {
        int acc = 0;
        for (int32_t j = 0; j < c->nbits; j++) if ((c->mask >> j) & 1) acc ^= tr_cert_bit_value(v, c, j, row);
        return acc == c->constant ? TR_CERT_ROW_HOLDS : TR_CERT_ROW_VIOLATES;
    }
    case TR_CLAIM_REFINEMENT: {
        int64_t fine = tr_scope_value(v, c->col[1], row);
        uint64_t i = tr_cert_lower_bound(c->keys, c->nvalues, fine);
        if (i == c->nvalues || c->keys[i] != fine) return TR_CERT_ROW_VIOLATES;
        return tr_scope_value(v, c->col[0], row) == c->targets[i] ? TR_CERT_ROW_HOLDS : TR_CERT_ROW_VIOLATES;
    }
    case TR_CLAIM_COVERAGE: {
        int64_t cat = tr_scope_value(v, c->col[0], row);
        uint64_t i = tr_cert_lower_bound(c->keys, c->nvalues, cat);
        if (i < c->nvalues && c->keys[i] == cat) *bit = (uint64_t)1 << i;
        return TR_CERT_ROW_HOLDS;
    }
    case TR_CLAIM_POLYNOMIAL: {
        __int128 value = 0;
        if (!tr_cert_poly_value(v, c, row, &value)) return TR_CERT_ROW_OVERFLOW;
        if (c->carrier == TR_CARRIER_GFP) return value == 0 ? TR_CERT_ROW_HOLDS : TR_CERT_ROW_VIOLATES;
        return tr_cert_residual(c, value);
    }
    case TR_CLAIM_UNIQUE:
    case TR_CLAIM_NOT_NULL:
        return TR_CERT_ROW_HOLDS;
    case TR_CLAIM_REFERENCE: {
        int64_t x = tr_scope_value(v, c->col[0], row);
        if (x < 0 || (uint64_t)x >= c->nvalues) return TR_CERT_ROW_VIOLATES;
        return ((c->used[(uint64_t)x >> 6] >> ((uint64_t)x & 63)) & 1) ? TR_CERT_ROW_HOLDS : TR_CERT_ROW_VIOLATES;
    }
    default:
        return TR_CERT_ROW_VIOLATES;
    }
}
/* the full required mask of a coverage claim */
TR_NUM_HD uint64_t tr_cert_full_mask(uint64_t nvalues) {
    return nvalues >= 64 ? UINT64_MAX : (((uint64_t)1 << nvalues) - 1);
}
/* the [min, max] domain size of a UNIQUE claim's column (UINT64_MAX past 2^64 - 1; 0 without a present value) */
TR_NUM_HD uint64_t tr_cert_unique_domain(const tr_scope_view *v, const tr_cert_claim *c) {
    const tr_scope_view_column *k = &v->col[c->col[0]];
    if (!k->nvalid || k->min > k->max) return 0;
    uint64_t span = (uint64_t)k->max - (uint64_t)k->min;
    return span == UINT64_MAX ? UINT64_MAX : span + 1;
}
/* the device copies of tr_mgpu_cert_verify (bytes; UNKNOWN past 64 bits) and its other needs */
static inline void tr_cert_device_need(const tr_scope_view *v, const tr_cert_claim *claims, int32_t n, tr_math_need *need) {
    for (int i = 0; i < TR_BUDGET_RESOURCES; i++) need->need[i] = 0;
    uint64_t rows = v->rows, bytes = tr_budget_add(tr_budget_bitmap_bytes(rows), tr_budget_mul((uint64_t)(n > 0 ? n : 0), sizeof(tr_cert_claim) + 64));
    uint32_t cols = 0;
    int groups = 0;
    uint64_t states = 0, support = 0;
    int32_t vars = 0;
    for (int32_t j = 0; j < n; j++) {
        const tr_cert_claim *c = &claims[j];
        if (c->ncols > vars) vars = c->ncols;
        for (int32_t i = 0; i < c->ncols && i < TR_CERT_MAX_COLS; i++) if (c->col[i] >= 0 && c->col[i] < 32) cols |= (uint32_t)1 << c->col[i];
        uint64_t pay = 0;
        if (c->kind == TR_CLAIM_PARITY) pay = tr_budget_mul((uint64_t)c->nbits, sizeof(tr_cert_bit));
        if (c->kind == TR_CLAIM_POLYNOMIAL) pay = tr_budget_mul(c->nterms, tr_budget_add((uint64_t)c->ncols, 8));
        if (c->kind == TR_CLAIM_REFINEMENT) pay = tr_budget_mul(c->nvalues, 16);
        if (c->kind == TR_CLAIM_COVERAGE) {
            pay = tr_budget_mul(c->nvalues, 8);
            groups = 1;
            states = tr_budget_add(states, v->ngroups);
            pay = tr_budget_add(pay, tr_budget_mul(v->ngroups, 8));
        }
        if (c->kind == TR_CLAIM_REFERENCE) pay = tr_budget_bitmap_bytes(c->nvalues);
        if (c->kind == TR_CLAIM_UNIQUE) pay = tr_budget_mul(tr_cert_unique_domain(v, c), 8);
        bytes = tr_budget_add(bytes, pay);
        uint64_t s = c->kind == TR_CLAIM_POLYNOMIAL ? c->nterms : c->kind == TR_CLAIM_PARITY ? (uint64_t)c->nbits :
                     c->kind == TR_CLAIM_REFERENCE ? (c->nvalues + 63) / 64 : c->nvalues;
        if (s > support) support = s;
    }
    if (groups) {
        bytes = tr_budget_add(bytes, tr_budget_bitmap_bytes(v->ngroups));
        if (v->groups) bytes = tr_budget_add(bytes, tr_budget_mul(rows, 4));
    }
    for (int32_t c = 0; c < v->ncols && c < TR_SCOPE_MAX_COLUMNS; c++) {
        if (!((cols >> c) & 1)) continue;
        bytes = tr_budget_add(bytes, tr_budget_mul(rows, v->col[c].values ? 8 : 4));
        if (v->col[c].valid) bytes = tr_budget_add(bytes, rows);
    }
    need->need[TR_BUDGET_VARIABLES] = (uint64_t)vars;
    need->need[TR_BUDGET_SUPPORT] = support;
    need->need[TR_BUDGET_OUTPUT] = (uint64_t)(n > 0 ? n : 0);
    need->need[TR_BUDGET_STATES] = states;
    need->need[TR_BUDGET_BYTES] = bytes;
}

/* ------------------------------------------------------------------ constant tables */
int32_t tr_cert_claim_describe(int32_t kind, tr_cert_claim_desc *out);
int32_t tr_cert_law_describe(int32_t law, tr_cert_law_desc *out);

/* ------------------------------------------------------------------ claims */
/* Validates a claim against a scope of ncols columns (-1: a law, columns are variables), canonicalizes nothing
 * (a polynomial must already be canonical: tr_cert_poly_canonical), fills claim->gf for a GFP check, and writes its
 * needs (variables, degree, support). TR_MATH_INVALID / _UNSUPPORTED / _CARRIER_ERROR as the header says. */
int32_t tr_cert_claim_check(tr_cert_claim *claim, int32_t ncols, tr_math_need *need);
/* The canonical form of nterms terms (exponents nterms x ncols, coefficients): equal exponent rows merged exactly
 * (a merged coefficient outside int64: TR_MATH_OVERFLOW), zero terms dropped, rows in increasing lexicographic
 * order, written to out_exponents / out_coefficients (capacity nterms each); *out_terms = the canonical terms.
 * field p > 0: coefficients reduced to the symmetric range of GF(p) first (and GFP_FUNCTION's exponent reduction
 * when fermat != 0). */
int32_t tr_cert_poly_canonical(int32_t ncols, uint64_t nterms, const uint8_t *exponents, const int64_t *coefficients,
                               uint64_t field, int32_t fermat, uint8_t *out_exponents, int64_t *out_coefficients,
                               uint64_t *out_terms);

/* ------------------------------------------------------------------ dimensions and records */
/* Describes a dimension reached through: dim (an atable_t), its lifetime token (16 bytes), its used rows (NULL:
 * every row), its key column and the key's validity column (-1). Decodes the key natively and decides uniqueness
 * among the used rows. Budget: bytes = 8 per row + 4 per used row. */
int32_t tr_cert_dim_describe(const struct atable_s *dim, const uint8_t *identity, const uint64_t *used, int32_t key,
                             int32_t key_valid, const tr_math_budget *, tr_math_refusal *, tr_cert_dim *out);
/* The record of a scope over cols[ncols] (scope column indices, distinct; NULL with ncols -1: every column) with
 * flags TR_CERT_RECORD_*, and ndims dimensions. */
tr_cert_record *tr_cert_record_scope(const tr_scope *, const int32_t *cols, int32_t ncols, int32_t flags,
                                     const tr_cert_dim *dims, int32_t ndims, const tr_math_budget *,
                                     tr_math_refusal *, int32_t *error);
int32_t tr_cert_record_get(const tr_cert_record *, tr_cert_source *);
/* the exact selection bits (ceil(rows / 64) words); TR_MATH_UNSUPPORTED when the record keeps only their digest */
int32_t tr_cert_record_selection(const tr_cert_record *, uint64_t *out, uint64_t capacity);
/* recorded vs current under mode (TR_CERT_MATCH_*): TR_MATH_OK or TR_MATH_STALE with the reasons */
int32_t tr_cert_record_compare(const tr_cert_record *recorded, const tr_cert_record *current, int32_t mode,
                               tr_cert_staleness *out);
/* the comparison with a record of the current scope (same columns and flags) and dimensions */
int32_t tr_cert_record_fresh(const tr_cert_record *recorded, const tr_scope *current, const tr_cert_dim *dims,
                             int32_t ndims, int32_t mode, tr_cert_staleness *out);
uint64_t tr_cert_record_bytes(const tr_cert_record *);
uint64_t tr_cert_record_serial_size(const tr_cert_record *);
int32_t tr_cert_record_serialize(const tr_cert_record *, uint8_t *out, uint64_t capacity, uint64_t *written);
tr_cert_record *tr_cert_record_deserialize(const uint8_t *in, uint64_t bytes, uint64_t *consumed,
                                           const tr_math_budget *, tr_math_refusal *, int32_t *error);
tr_cert_record *tr_cert_record_clone(const tr_cert_record *, int32_t *error);
/* a copy keeping only the parts in flags (a subset of the record's TR_CERT_RECORD_* flags): the digests stay, the exact
 * selection bits / grouping bytes not in flags are dropped (comparisons then use their digests) */
tr_cert_record *tr_cert_record_copy(const tr_cert_record *, int32_t flags, int32_t *error);
void tr_cert_record_free(tr_cert_record *);

/* ------------------------------------------------------------------ verification (the CPU reference of the kernel) */
/* Exhaustive evidence of n checked claims (tr_cert_claim_check) over the selected rows of a view; out[n]. */
int32_t tr_cert_verify_view(const tr_scope_view *, const tr_cert_claim *claims, int32_t n, const tr_math_budget *,
                            tr_cert_evidence *out, tr_math_refusal *);
/* tr_cert_claim_check of copies of the claims against the scope, then tr_cert_verify_view */
int32_t tr_cert_verify(const tr_scope *, const tr_cert_claim *claims, int32_t n, const tr_math_budget *,
                       tr_cert_evidence *out, tr_math_refusal *);
/* the needs of the CPU verification (TR_CERT_NEED_CPU) or of tr_mgpu_cert_verify (TR_CERT_NEED_DEVICE) */
int32_t tr_cert_verify_need(const tr_scope_view *, const tr_cert_claim *claims, int32_t n, int32_t target,
                            tr_math_need *need);

/* ------------------------------------------------------------------ certificates */
/* A universal law: TR_CERT_LAW_RING_IDENTITY (claim POLYNOMIAL over Z, field 0) or _GFP_FUNCTION (field p prime);
 * decided natively; TR_MATH_DOMAIN_ERROR when the claim is not an instance of the law. */
tr_cert *tr_cert_law(int32_t law, const tr_cert_claim *, const tr_math_budget *, tr_math_refusal *, int32_t *error);
/* An enforced schema fact (claim UNIQUE, NOT_NULL or REFERENCE) verified on every selected row of the scope (pass
 * the table's used rows); TR_MATH_DOMAIN_ERROR (evidence: the witness) when the data violate it. */
tr_cert *tr_cert_schema(const tr_scope *, const tr_cert_claim *, int32_t enforcement, const tr_cert_dim *dims,
                        int32_t ndims, tr_cert_evidence *evidence, const tr_math_budget *, tr_math_refusal *,
                        int32_t *error);
/* An exhaustive snapshot certificate; TR_MATH_DOMAIN_ERROR (violations) or TR_MATH_OVERFLOW (undecided rows) leave
 * no certificate but write the evidence. */
tr_cert *tr_cert_snapshot(const tr_scope *, const tr_cert_claim *, const tr_cert_dim *dims, int32_t ndims,
                          tr_cert_evidence *evidence, const tr_math_budget *, tr_math_refusal *, int32_t *error);
/* A hypothesis from a sample of the selected rows (sample: method, size, seed; population and distinct are set);
 * the claim must hold on the sample (else TR_MATH_DOMAIN_ERROR). Kinds whose evidence is not row-local (UNIQUE,
 * COVERAGE) are decided on the drawn rows only. */
tr_cert *tr_cert_hypothesis(const tr_scope *, const tr_cert_claim *, const tr_cert_sample *sample,
                            const tr_cert_dim *dims, int32_t ndims, tr_cert_evidence *evidence,
                            const tr_math_budget *, tr_math_refusal *, int32_t *error);
/* A snapshot from evidence computed elsewhere (tr_mgpu_cert_verify's readback) after consistency checks (rows,
 * selected rows, checked + excluded = selected rows, the witness, holds); backend TR_CERT_BY_*. */
tr_cert *tr_cert_adopt(const tr_scope *, const tr_cert_claim *, const tr_cert_evidence *, int32_t backend,
                       const tr_cert_dim *dims, int32_t ndims, const tr_math_budget *, tr_math_refusal *,
                       int32_t *error);
/* The same claim re-verified exhaustively on a current scope (re-certification after an invalidation). */
tr_cert *tr_cert_reverify(const tr_cert *, const tr_scope *current, const tr_cert_dim *dims, int32_t ndims,
                          tr_cert_evidence *evidence, const tr_math_budget *, tr_math_refusal *, int32_t *error);
/* A POLYNOMIAL claim derived from n premises with cofactor polynomials (cofactors[i]: a POLYNOMIAL tr_cert_claim
 * whose col[] names scope columns; its coefficients multiply premise i), see the header comment. */
tr_cert *tr_cert_derive(const tr_cert_claim *claim, const tr_cert *const *premises, const tr_cert_claim *cofactors,
                        int32_t n, const tr_math_budget *, tr_math_refusal *, int32_t *error);

/* ------------------------------------------------------------------ checks and reading */
int32_t tr_cert_fresh(const tr_cert *, const tr_scope *current, const tr_cert_dim *dims, int32_t ndims,
                      tr_cert_staleness *out);
/* field: the target (0 = Z, p = GF(p)); current: the scope the rewrite runs on (NULL: only a law can pass) */
int32_t tr_cert_authorize_exact(const tr_cert *, uint64_t field, const tr_scope *current, const tr_cert_dim *dims,
                                int32_t ndims, tr_cert_decision *out);
int32_t tr_cert_get_info(const tr_cert *, tr_cert_info *);
/* the claim with pointers into the certificate's own arrays (valid while it lives) */
int32_t tr_cert_get_claim(const tr_cert *, tr_cert_claim *);
const tr_cert_record *tr_cert_get_record(const tr_cert *);      /* borrowed; NULL for a law */
const tr_cert *tr_cert_get_premise(const tr_cert *, int32_t i); /* borrowed */
tr_cert *tr_cert_clone(const tr_cert *, int32_t *error);
int32_t tr_cert_equal(const tr_cert *, const tr_cert *);        /* 1: the same canonical bytes */
uint64_t tr_cert_serial_size(const tr_cert *);
int32_t tr_cert_serialize(const tr_cert *, uint8_t *out, uint64_t capacity, uint64_t *written);
tr_cert *tr_cert_deserialize(const uint8_t *in, uint64_t bytes, uint64_t *consumed, const tr_math_budget *,
                             tr_math_refusal *, int32_t *error);
void tr_cert_free(tr_cert *);

/* ------------------------------------------------------------------ the CUDA library (math_certificates_gpu.cu) */
/* claims: checked claims (host pointers); view: tr_scope_get_view of the scope; out[n]. Budget (NULL = defaults):
 * tr_cert_device_need. Returns 0 or a native error code (TR_MATH_DEVICE when the device fails). */
int tr_mgpu_cert_verify(const tr_scope_view *view, const tr_cert_claim *claims, int32_t n, const tr_math_budget *budget,
                        tr_cert_evidence *out, tr_math_refusal *refusal);
#ifdef __cplusplus
}
#endif
#endif
