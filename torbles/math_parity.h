/* FIELD-002: parity relationship discovery and substitution. Bounded exact XOR dependencies among Boolean columns and
 * code bit planes of a scope are found with FIELD-001 (proposals from a sample, exhaustive verification, near misses
 * rejected with their witness row), each verified relation is certified as an FND-007 snapshot certificate, and a
 * predicate circuit that reads one side of a certified relation is rewritten over the other side when the certificate
 * is fresh, the cheaper form chosen by the columns and planes it scans; the rewrite is verified against the original
 * on every selected row, on the CPU (math_parity.c) and on a CUDA device (math_parity_gpu.cu) bit for bit alike.
 * SPDX-License-Identifier: Apache-2.0
 *
 * ---- Discovery and certificates (tr_parity_discover / tr_parity_adopt -> tr_parity_found) ---------------------------
 * Bit variables: FIELD-001 bit specs over the scope (tr_field_bitspec: NONZERO = logical != 0, a Boolean column; PLANE =
 * bit b of (logical - base) mod 2^64, with base = the codec offset a stored code plane). Discovery is FIELD-001's
 * tr_field_discover (or tr_mgpu_field_discover): `relations` the canonical basis of the relations holding on every
 * complete selected row, `rejected` the failed proposals with their evidence (a one-row near miss: violations 1 and its
 * witness row). Each found relation XOR_{j in vars} x_j = constant becomes an FND-007 PARITY claim (tr_parity_claim):
 * col[] = the distinct scope columns of its variables (increasing), bits[] = its variables in increasing index order
 * (kind, plane, base; bits[i].col indexes col[]), mask = every bit, the constant, the NULL policy of the spec. It is
 * certified with tr_cert_snapshot (exhaustive over every selected row of the scope; the certificate records the
 * table's identity and revision, the rows, the exact selection, each column's encoding (stored column, offset, scale,
 * high part) and NULL domain) or, from device evidence (tr_mgpu_cert_verify), with tr_cert_adopt. Status per relation:
 *   CERTIFIED  a snapshot certificate exists
 *   VIOLATED   the FND-007 verification found violations (its evidence is kept): a NULL under the FORBID policy, or a
 *              row where this relation's own columns are present but another variable of the discovery is NULL (FIELD-001
 *              verifies listwise, FND-007 on the relation's own columns)
 *   TOO_WIDE   its variables read more than TR_CERT_MAX_COLS scope columns (no claim)
 *   EMPTY      vars = 0: the contradiction 0 = 1 holding vacuously on no complete row (no claim)
 * Codec changes (another offset, scale or stored column), NULL domain changes and revisions make a certificate stale
 * with FND-007's exact reasons (columns, null_domain, revision, ...); only the certificates whose recorded scope parts
 * changed are affected (a codec or NULL change of another column leaves a certificate fresh unless the table revision
 * moved too).
 *
 * ---- Substitution (tr_parity_substitute -> tr_parity_rewrite) --------------------------------------------------------
 * A predicate: a BOOL-001 program over k conditions (tr_bool_condition) of the current scope. A relation bit is the
 * condition `col <> 0` (NONZERO) or `bit b of (col - base)` (PLANE, TR_BOOL_BIT) of the claim's column: the predicate's
 * conditions equal to one (field for field) are the relation's; the others it needs are appended (k' = k + appended,
 * at most 64). The relation is the GF(2) vector S over the conditions with constant c (XOR_S x = c).
 * Refusals, before anything is rewritten: a claim that is not PARITY (TR_PARITY_REFUSE_CLAIM); FND-007's
 * tr_cert_authorize_exact for GF(2) over the current scope refused (TR_PARITY_REFUSE_CERTIFICATE: a stale certificate
 * returns TR_MATH_STALE with the staleness reasons in info.decision.stale; a hypothesis, an approximate or incomplete
 * claim TR_MATH_DOMAIN_ERROR); a certificate whose NULL policy left rows out (SKIP with excluded rows: the rewrite would
 * change the UNKNOWN rows) (TR_PARITY_REFUSE_NULLS). A fresh FORBID certificate (or SKIP that excluded nothing) proves
 * every relation bit known and XOR_S x = c on every selected row.
 * XOR forms: a scheduled node is a pure form L (a GF(2) vector over the conditions and a constant bit) when it is VAR x
 * (L = e_x), CONST TRUE / FALSE (the constant 1 / 0), NOT of a pure form (L + 1), or XOR of two pure forms whose
 * variables are disjoint (L_a + L_b; disjoint, so no variable cancels and the form is UNKNOWN exactly when one of its
 * variables is). A pure form is maximal when it is the root or an operand of a node that is not pure; it is a
 * candidate when it meets S. Rewriting a candidate replaces it by the form L + (S, c) (the XOR chain of its variables
 * in increasing order, NOT for the constant 1, CONST TRUE / FALSE when no variable is left): exact on every selected row
 * (the variables it adds or removes are relation bits, always known there). Both directions are this one rule: a
 * variable that is the XOR of others (L = e_x, x in S) becomes the XOR of the others, and an XOR of the others becomes
 * that variable.
 * Cost: the columns and planes the predicate's reachable conditions scan: a BIT condition scans its plane (column, bit,
 * base), any other condition its column (and col2 for column comparisons); a scanned column subsumes its planes.
 * TR_PARITY_AUTO rewrites the candidates greedily in increasing node order, keeping a rewrite only when the cost
 * strictly drops: chosen = REWRITTEN when some rewrite was kept, else KEPT (the original is at most as expensive).
 * TR_PARITY_ALL rewrites every candidate (the other direction, whatever it costs). The rewrite object holds the
 * conditions (k'), the original program over them and the rewritten program (equal to the original when nothing was
 * rewritten).
 *
 * ---- Verification (tr_parity_verify, tr_mgpu_parity_verify) ----------------------------------------------------------
 * Two programs over the same domain (BOOL-001 masks of the k' conditions over the current scope) are evaluated on every
 * selected row: their TRUE / FALSE / UNKNOWN counts, the rows where they differ (another value, or UNKNOWN on one side
 * only) and the smallest such row (witness); equivalent = no mismatch. On the device each work item is a 64-row word
 * (or, past TR_PARITY_SCRATCH bytes of node words, a strided slice of words): both programs' value and valid words, their
 * XOR, popcounts, one atomic per counter and an atomic minimum for the witness.
 *
 * ---- Budgets (math_budgets.h), checked before anything is allocated ---------------------------------------------------
 *   discover: FIELD-001's bits and discovery needs (variables, support, iterations, output, bytes), then per relation
 *     FND-007's snapshot needs; bytes += the found object (tr_parity_discover_need: 128 bytes per relation slot).
 *   substitute: variables = k', bytes = the analysis scratch (21 bytes per node) before the analysis; after the choice
 *     circuit = the rewritten program's nodes, bytes += 16 per node and 48 per condition (BOOL-001 checks its programs).
 *   verify: variables = k, circuit = both programs' nodes, bytes = the node words (16 per node of each; on the device
 *     the copies, the node words of every work item and the accumulators).
 * Representation limits: TR_MATH_UNSUPPORTED for k' > 64.
 *
 * Errors: TR_MATH_INVALID (-1), _MEMORY (-2), _BUDGET (-5), TR_MATH_DOMAIN_ERROR (-6), TR_MATH_DEVICE (-11),
 * TR_MATH_UNSUPPORTED (-12), TR_MATH_STALE (-13). Objects are immutable once built and individually owned;
 * constructors return NULL and set *error, leaving their inputs unchanged. */
#ifndef TORBLES_MATH_PARITY_H
#define TORBLES_MATH_PARITY_H
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "math_num.h"
#include "math_budgets.h"
#include "math_scope.h"
#include "math_bool.h"
#include "math_field.h"
#include "math_certificates.h"
#ifdef __cplusplus
extern "C" {
#endif

#define TR_PARITY_ABI 1
#define TR_PARITY_NONE UINT64_MAX
#define TR_PARITY_SCRATCH (UINT64_C(1) << 26)    /* bytes of node words on the device (one word per work item below) */

enum { TR_PARITY_CERTIFIED = 1, TR_PARITY_VIOLATED = 2, TR_PARITY_TOO_WIDE = 3, TR_PARITY_EMPTY = 4 };  /* status */
enum { TR_PARITY_AUTO = 0, TR_PARITY_ALL = 1 };                                                       /* modes */
enum { TR_PARITY_KEPT = 0, TR_PARITY_REWRITTEN = 1 };                                                 /* choice */
enum { TR_PARITY_REFUSE_NONE = 0, TR_PARITY_REFUSE_CERTIFICATE = 1, TR_PARITY_REFUSE_NULLS = 2,
       TR_PARITY_REFUSE_CLAIM = 3 };
enum { TR_PARITY_NEED_CPU = 0, TR_PARITY_NEED_DEVICE = 1 };

/* ------------------------------------------------------------------ structs crossing the ctypes boundary */
typedef struct {
    uint64_t sample;                   /* FIELD-001's proposal sample (0: every complete row) */
    int32_t affine;                    /* 1: relations with a constant */
    int32_t policy;                    /* TR_CERT_NULL_FORBID or TR_CERT_NULL_SKIP: the certificates' NULL policy */
} tr_parity_spec;                      /* 16 bytes */

typedef struct {
    tr_field_relation relation;        /* vars over the bit variables, constant */
    tr_field_evidence evidence;        /* FIELD-001's exhaustive evidence (listwise over the complete rows) */
    tr_cert_evidence certified;        /* FND-007's verification of the claim under the policy (zero without a claim) */
    int32_t status, ncols;             /* TR_PARITY_*; distinct scope columns of its variables */
} tr_parity_relation;                  /* 128 bytes */

typedef struct {
    tr_field_discovery stats;          /* FIELD-001's discovery statistics */
    int32_t nrelations, nrejected, ncertified, policy;
    int32_t nvars, affine, backend, reserved;   /* backend: TR_CERT_BY_* (who verified the claims) */
} tr_parity_found_info;                /* 96 bytes */

typedef struct {
    int32_t chosen, reason;            /* TR_PARITY_KEPT / _REWRITTEN; TR_PARITY_REFUSE_* */
    int32_t k, nnodes;                 /* conditions of the rewrite; nodes of the rewritten program */
    int32_t root, forms;               /* its root; the forms rewritten */
    int32_t cost_original, cost_rewritten;   /* columns and planes scanned */
    int32_t candidates, appended;      /* maximal pure forms meeting the relation; conditions appended */
    int32_t mode, k_original;          /* TR_PARITY_AUTO / _ALL; the predicate's own conditions */
    tr_cert_decision decision;         /* FND-007's authorization for GF(2) over the current scope */
} tr_parity_rewrite_info;              /* 112 bytes */

typedef struct {
    uint64_t points;                   /* selected rows */
    uint64_t mismatches, witness;      /* rows where the programs differ; the smallest (TR_PARITY_NONE) */
    uint64_t a_true, a_false, a_unknown;   /* the first (original) program */
    uint64_t b_true, b_false, b_unknown;   /* the second (substituted) program */
    int32_t equivalent, reserved;
} tr_parity_check;                     /* 80 bytes */

typedef struct tr_parity_found_s tr_parity_found;
typedef struct tr_parity_rewrite_s tr_parity_rewrite;

/* ------------------------------------------------------------------ shared helpers (CPU code, CUDA host and kernels) */
/* the mismatch words of two programs' root words at one word (value and valid of each), among the points that count */
TR_NUM_HD uint64_t tr_parity_mismatch(uint64_t va, uint64_t ka, uint64_t vb, uint64_t kb, uint64_t sel) {
    return ((va ^ vb) | (ka ^ kb)) & sel;
}
/* slices of the verification launch: one word per work item while the node words fit TR_PARITY_SCRATCH */
static inline uint64_t tr_parity_slices(uint64_t nw, int32_t na, int32_t nb) {
    uint64_t per = tr_budget_mul(tr_budget_add((uint64_t)(na > 0 ? na : 0), (uint64_t)(nb > 0 ? nb : 0)), 16);
    uint64_t room = per ? TR_PARITY_SCRATCH / per : nw;
    uint64_t s = nw < room ? nw : room;
    return s ? s : 1;
}
/* the device bytes of tr_mgpu_parity_verify */
static inline uint64_t tr_parity_device_bytes(const tr_bool_domain *d, const tr_bool_program_view *a,
                                              const tr_bool_program_view *b) {
    uint64_t bytes = 0;
    if (d->kind == TR_BOOL_ROWS) {
        bytes = tr_budget_mul(d->nw, 8);
        bytes = tr_budget_add(bytes, tr_budget_mul(tr_budget_mul((uint64_t)d->k, d->nw), d->valid ? 16 : 8));
    }
    bytes = tr_budget_add(bytes, tr_budget_mul(tr_budget_add((uint64_t)a->nnodes, (uint64_t)b->nnodes), sizeof(tr_bool_node)));
    bytes = tr_budget_add(bytes, tr_budget_mul(tr_budget_add((uint64_t)a->nsched, (uint64_t)b->nsched), 4));
    uint64_t items = tr_parity_slices(d->nw, a->nnodes, b->nnodes);
    bytes = tr_budget_add(bytes, tr_budget_mul(items, tr_budget_mul(tr_budget_add((uint64_t)a->nnodes, (uint64_t)b->nnodes), 16)));
    return tr_budget_add(bytes, 64);                                                  /* accumulators */
}

/* ------------------------------------------------------------------ claims and discovery (libalg) */
/* The FND-007 PARITY claim of a FIELD-001 relation over bit variables specs[nvars] (see Discovery); bits: 64 entries
 * of scratch the claim points into. *status = 0 for a claim, else TR_PARITY_TOO_WIDE / TR_PARITY_EMPTY (no claim). */
int32_t tr_parity_claim(const tr_field_bitspec *specs, int32_t nvars, const tr_field_relation *r, int32_t policy,
                        tr_cert_claim *claim, tr_cert_bit *bits, int32_t *status, int32_t *ncols);
/* the needs of a discovery over the scope's bit variables: FIELD-001's (bits and discovery) and the found object */
int32_t tr_parity_discover_need(const tr_scope *, const tr_field_bitspec *, int32_t nvars, const tr_parity_spec *,
                                tr_math_need *);
/* discovery (FIELD-001) and an FND-007 snapshot certificate for each relation found */
tr_parity_found *tr_parity_discover(const tr_scope *, const tr_field_bitspec *, int32_t nvars, const tr_parity_spec *,
                                    const tr_math_budget *, tr_math_refusal *, int32_t *error);
/* The same from device results: FIELD-001's tr_mgpu_field_discover readback (stats, found[nfound] with their
 * evidence, rejected[nrejected] with theirs) and the device evidence of the claims of the found relations
 * (claim_ev[nfound], tr_mgpu_cert_verify over the claims of tr_parity_claim; entries of relations without a claim are
 * ignored). The claims are rebuilt here; each holding claim is adopted with tr_cert_adopt (TR_CERT_BY_DEVICE). */
tr_parity_found *tr_parity_adopt(const tr_scope *, const tr_field_bitspec *, int32_t nvars, const tr_parity_spec *,
                                 const tr_field_discovery *stats, const tr_field_relation *found,
                                 const tr_field_evidence *found_ev, int32_t nfound, const tr_field_relation *rejected,
                                 const tr_field_evidence *rejected_ev, int32_t nrejected,
                                 const tr_cert_evidence *claim_ev, const tr_math_budget *, tr_math_refusal *,
                                 int32_t *error);
int32_t tr_parity_found_get_info(const tr_parity_found *, tr_parity_found_info *);
int32_t tr_parity_found_relation(const tr_parity_found *, int32_t i, tr_parity_relation *);
int32_t tr_parity_found_rejected(const tr_parity_found *, int32_t i, tr_field_relation *, tr_field_evidence *);
const tr_cert *tr_parity_found_certificate(const tr_parity_found *, int32_t i);   /* borrowed; NULL without one */
void tr_parity_found_free(tr_parity_found *);

/* ------------------------------------------------------------------ substitution and verification (libalg) */
/* The rewrite of the predicate (prog over conds[k] of the current scope) with the certificate (see Substitution); on a
 * refusal NULL with *error (TR_MATH_STALE, TR_MATH_DOMAIN_ERROR, ...) and info->reason / info->decision set. */
tr_parity_rewrite *tr_parity_substitute(const tr_scope *current, const tr_bool_condition *conds, int32_t k,
                                        const tr_bool_program *prog, const tr_cert *cert, int32_t mode,
                                        const tr_math_budget *, tr_math_refusal *, tr_parity_rewrite_info *info,
                                        int32_t *error);
int32_t tr_parity_rewrite_get_info(const tr_parity_rewrite *, tr_parity_rewrite_info *);
/* the k' conditions (cap >= k') */
int32_t tr_parity_rewrite_conditions(const tr_parity_rewrite *, tr_bool_condition *out, int32_t cap);
/* borrowed programs over the k' conditions: which 0 = the original, 1 = the rewritten */
const tr_bool_program *tr_parity_rewrite_program(const tr_parity_rewrite *, int32_t which);
void tr_parity_rewrite_free(tr_parity_rewrite *);
/* the needs of tr_parity_verify (TR_PARITY_NEED_CPU) or tr_mgpu_parity_verify (_DEVICE) */
int32_t tr_parity_verify_need(const tr_bool_domain *, const tr_bool_program_view *a, const tr_bool_program_view *b,
                              int32_t target, tr_math_need *);
/* both programs on every selected row of the masks (each program's k must equal the masks') */
int32_t tr_parity_verify(const tr_bool_masks *, const tr_bool_program *a, const tr_bool_program *b,
                         const tr_math_budget *, tr_parity_check *out, tr_math_refusal *);
int32_t tr_parity_verify_view(const tr_bool_domain *, const tr_bool_program_view *a, const tr_bool_program_view *b,
                              const tr_math_budget *, tr_parity_check *out, tr_math_refusal *);

/* ------------------------------------------------------------------ the CUDA library (math_parity_gpu.cu, libmathgpu)
 * tr_mgpu_parity_verify: tr_parity_verify_view on the device (views of host memory, copied and budgeted). */
int tr_mgpu_parity_verify(const tr_bool_domain *, const tr_bool_program_view *a, const tr_bool_program_view *b,
                          const tr_math_budget *, tr_parity_check *out, tr_math_refusal *);
#ifdef __cplusplus
}
#endif
#endif
