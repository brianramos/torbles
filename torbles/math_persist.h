/* FND-011: versioned binary persistence of typed mathematical states.
 * SPDX-License-Identifier: Apache-2.0
 *
 * A record is one immutable state in canonical little-endian bytes (written byte by byte, so the format does not
 * depend on the host), never a live address: every pointer of the state (arrays, the scope, the certificate's
 * premises) is rebuilt when the record is loaded.
 *
 *   header (128 bytes)
 *     0  magic "TORBMATH"                 8  format version (u32, TR_PERSIST_FORMAT)  12  header bytes (u32, 128)
 *    16  endianness marker 0x0102030405060708 as written (a foreign byte order reads it swapped)
 *    24  type tag (TR_PERSIST_*)          28  type version (the state's own layout version, 1)
 *    32  carrier (TR_CARRIER_* of the state's values)   36  exact (1; 0 for an approximate state)
 *    40  flags (TR_PERSIST_SOURCE: a source section; TR_PERSIST_IMMUTABLE, always)   44  reserved 0
 *    48  source bytes   56  basis bytes   64  metadata bytes   72  payload bytes   80  total bytes   88  reserved 0
 *    96  checksum (two u64 lanes)         112 reserved 0 (two u64)
 *   sections, each padded with zeros to a multiple of 8 bytes, in this order:
 *     source    the domain / proof identity: a FND-007 record (tr_cert_record, address-free) of the scope the state
 *               summarizes: the source's lifetime token, revision, row count, selected rows, the version digest, the
 *               selection digest, every scope column's encoding and NULL domain, the grouping descriptor
 *               (absent for carriers and unscoped states)
 *     basis     the state's shape: bank (columns, terms, mode, groups, the exponent matrix), histogram (the spec and
 *               the dense domain), moments (the spec, marginals, slots), carrier values (none)
 *     metadata  owned bytes of the caller (e.g. column names, codecs as JSON), returned as given
 *     payload   the values: bank rows / counts / 128-bit sums; histogram rows, CSR offsets, coordinates, NULL masks,
 *               counts, weighted counts and sums; moment rows / counts / 192-bit sums; big integer sign and limbs;
 *               big rational numerator and denominator; GF(p) modulus and residue; a certificate's FND-007 bytes
 *
 * Checksum. Over the record as 64-bit little-endian words w_i (i = 0 .. total / 8 - 1), the checksum words 12 and 13
 * counted as zero: lane L = sum_i w_i * m_L(i) mod 2^64, m_L(i) = tr_persist_weight(i, L) (an odd multiplier per
 * position). The sum does not depend on the order it is accumulated in (the CUDA kernel adds per-thread partial sums
 * with atomics) and changes whenever any single word changes (m odd: w m != w' m mod 2^64 for w != w').
 *
 * Loading checks, in this order, each with its own reason (tr_persist_status.reason; the return code in brackets):
 *   TRUNCATED  fewer bytes than the header or than its total             [TR_MATH_INVALID]
 *   MAGIC      not a torbles mathematical state                          [TR_MATH_INVALID]
 *   ENDIAN     written in another byte order                             [TR_MATH_UNSUPPORTED]
 *   VERSION    an unknown format version, header size or type version    [TR_MATH_UNSUPPORTED]
 *   MALFORMED  inconsistent sizes, trailing bytes, an unknown type or invalid sections (checksum correct)
 *                                                                        [TR_MATH_INVALID]
 *   CORRUPT    the checksum differs                                      [TR_MATH_INVALID]
 *   TYPE       another state type than the one expected                  [TR_MATH_DOMAIN_ERROR]
 * (MALFORMED for sizes is decided before the checksum, for contents after it.)
 *
 * Immutability and identity. A loaded record (tr_persist_record) keeps the source identity it was saved with; nothing
 * re-stamps it. tr_persist_fresh compares it with a current scope (FND-007 exact comparison: TR_MATH_STALE with the
 * reasons), and every rehydration into a scope (tr_persist_to_bank / _histogram / _moments with an origin) requires
 * that comparison to pass: a reloaded state can never be adopted as the state of a changed source (another revision,
 * selection, NULL domain, grouping or lifetime token; the token and the version digest are process-specific, so a
 * record of another process is never fresh). Without an origin a bank is rehydrated unscoped (it cannot enter a
 * FND-004 cache); histograms and moments need their origin (their own adoption checks the scope).
 *
 * Budgets (math_budgets.h), checked before allocation: saving: bytes = the record; loading: bytes = the record object
 * and its copies of the sections (about the record's size), plus what the rehydrated object's own constructor checks.
 * The CUDA checksum: bytes = the device copy of the words.
 *
 * Errors: TR_MATH_INVALID, _MEMORY, _BUDGET, _UNSUPPORTED, TR_MATH_DOMAIN_ERROR (another type; a state not built from
 * the given scope), TR_MATH_STALE (an origin that is not the recorded source). Objects are immutable and owned. */
#ifndef TORBLES_MATH_PERSIST_H
#define TORBLES_MATH_PERSIST_H
#include <stdint.h>
#include "math_num.h"
#include "math_budgets.h"
#include "math_scope.h"
#include "math_bank.h"
#include "math_dist.h"
#include "math_lin.h"
#include "math_carriers.h"
#include "math_certificates.h"
#ifdef __cplusplus
extern "C" {
#endif

#define TR_PERSIST_FORMAT 1
#define TR_PERSIST_HEADER 128
#define TR_PERSIST_MAGIC UINT64_C(0x4854414d42524f54)     /* "TORBMATH" */
#define TR_PERSIST_ENDIAN UINT64_C(0x0102030405060708)
#define TR_PERSIST_CHECKSUM_WORD 12                         /* words 12 and 13 hold the checksum */
enum { TR_PERSIST_BANK = 1, TR_PERSIST_HISTOGRAM = 2, TR_PERSIST_BIGINT = 3, TR_PERSIST_BIGQ = 4, TR_PERSIST_GFP = 5,
       TR_PERSIST_MOMENTS = 6, TR_PERSIST_CERTIFICATE = 7, TR_PERSIST_TYPES = 8 };
enum { TR_PERSIST_SOURCE = 1, TR_PERSIST_IMMUTABLE = 2 };
enum { TR_PERSIST_OK = 0, TR_PERSIST_TRUNCATED = 1, TR_PERSIST_MAGIC_ERROR = 2, TR_PERSIST_ENDIAN_ERROR = 3,
       TR_PERSIST_VERSION = 4, TR_PERSIST_MALFORMED = 5, TR_PERSIST_CORRUPT = 6, TR_PERSIST_TYPE = 7 };

typedef struct {
    uint32_t format, header_bytes;
    int32_t type, type_version;
    int32_t carrier, exact;
    int32_t flags, reserved;
    uint64_t source_bytes, basis_bytes, metadata_bytes, payload_bytes, total_bytes;
    uint64_t endian;                   /* the marker as read */
    uint64_t checksum[2];              /* as stored */
} tr_persist_header;                   /* 96 bytes */

typedef struct {
    int32_t reason;                    /* TR_PERSIST_* */
    int32_t type;                      /* the type tag read (0 when not read) */
    uint32_t format, type_version;     /* as read */
    uint64_t bytes, needed;            /* bytes given; bytes the header declares (TRUNCATED: the larger) */
    uint64_t stored[2], computed[2];   /* the checksum stored and computed (CORRUPT) */
} tr_persist_status;                   /* 64 bytes */

/* the shape of a persisted bank (pointers into the record) */
typedef struct {
    int32_t ncols, nterms, mode, reserved;
    uint64_t ngroups;
    const uint8_t *exponents;          /* nterms x ncols */
} tr_persist_bank_basis;               /* 32 bytes */

typedef struct tr_persist_record_s tr_persist_record;
typedef struct tr_persist_blob_s tr_persist_blob;

/* ------------------------------------------------------------------ the checksum (CPU and CUDA) */
TR_NUM_HD uint64_t tr_persist_weight(uint64_t i, int lane) {
    uint64_t z = i + (lane ? UINT64_C(0xD1B54A32D192ED03) : UINT64_C(0x9E3779B97F4A7C15));
    z = (z ^ (z >> 30)) * UINT64_C(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94D049BB133111EB);
    return (z ^ (z >> 31)) | 1;
}
/* the device copies of tr_mgpu_persist_checksum for n words (its bytes need) */
static inline uint64_t tr_persist_device_bytes(uint64_t words) { return tr_budget_add(tr_budget_mul(words, 8), 16); }
uint64_t tr_persist_checksum_need(uint64_t words);
/* the position-weighted sums of n words (both lanes), every word counted (the record's checksum excludes words 12
 * and 13: tr_persist_validate_sum subtracts them) */
void tr_persist_checksum(const uint64_t *words, uint64_t n, uint64_t out[2]);

/* ------------------------------------------------------------------ saving: an owned blob of record bytes */
/* bank: built from origin (its scope; NULL: an unscoped bank, no source section). TR_MATH_DOMAIN_ERROR when the
 * bank's scope digest is not origin's. meta: the caller's metadata (may be NULL with 0 bytes). */
tr_persist_blob *tr_persist_bank(const tr_bank *, const tr_scope *origin, const uint8_t *meta, uint64_t meta_bytes,
                                 const tr_math_budget *, tr_math_refusal *, int32_t *error);
tr_persist_blob *tr_persist_histogram(const tr_dist_hist *, const tr_scope *origin, const uint8_t *meta,
                                      uint64_t meta_bytes, const tr_math_budget *, tr_math_refusal *, int32_t *error);
tr_persist_blob *tr_persist_moments(const tr_lin_moments *, const tr_scope *origin, const uint8_t *meta,
                                    uint64_t meta_bytes, const tr_math_budget *, tr_math_refusal *, int32_t *error);
tr_persist_blob *tr_persist_bigint(const tr_bigint *, const uint8_t *meta, uint64_t meta_bytes, const tr_math_budget *,
                                   tr_math_refusal *, int32_t *error);
tr_persist_blob *tr_persist_bigq(const tr_bigq *, const uint8_t *meta, uint64_t meta_bytes, const tr_math_budget *,
                                 tr_math_refusal *, int32_t *error);
tr_persist_blob *tr_persist_gfp(tr_gfp_elem, const uint8_t *meta, uint64_t meta_bytes, const tr_math_budget *,
                                tr_math_refusal *, int32_t *error);
tr_persist_blob *tr_persist_certificate(const tr_cert *, const uint8_t *meta, uint64_t meta_bytes,
                                        const tr_math_budget *, tr_math_refusal *, int32_t *error);
/* the bytes a save of a record of record_bytes checks (the record and the blob object), and a load (the record object
 * and its copies of the sections); UNKNOWN past 64 bits */
uint64_t tr_persist_save_need(uint64_t record_bytes);
uint64_t tr_persist_load_need(uint64_t record_bytes);
uint64_t tr_persist_blob_size(const tr_persist_blob *);
/* copies the record bytes; TR_MATH_INVALID when capacity is smaller */
int32_t tr_persist_blob_copy(const tr_persist_blob *, uint8_t *out, uint64_t capacity);
const uint8_t *tr_persist_blob_data(const tr_persist_blob *);
void tr_persist_blob_free(tr_persist_blob *);

/* ------------------------------------------------------------------ loading */
/* the header as read, after the size, magic, byte order and version checks (status written) */
int32_t tr_persist_header_read(const uint8_t *data, uint64_t bytes, tr_persist_header *out, tr_persist_status *status);
/* every check up to and including the checksum (no allocation) */
int32_t tr_persist_validate(const uint8_t *data, uint64_t bytes, tr_persist_status *status);
/* the checksum check against sums computed elsewhere over every word (tr_persist_checksum, tr_mgpu_persist_checksum) */
int32_t tr_persist_validate_sum(const uint8_t *data, uint64_t bytes, const uint64_t sum[2], tr_persist_status *status);
/* a loaded record (expected: a type tag, or 0 for any) */
tr_persist_record *tr_persist_load(const uint8_t *data, uint64_t bytes, int32_t expected, const tr_math_budget *,
                                   tr_math_refusal *, tr_persist_status *status, int32_t *error);
int32_t tr_persist_get_header(const tr_persist_record *, tr_persist_header *);
const tr_cert_record *tr_persist_get_source(const tr_persist_record *);    /* borrowed; NULL without a source */
/* copies the metadata (*length: its size; TR_MATH_INVALID when capacity is smaller) */
int32_t tr_persist_get_metadata(const tr_persist_record *, uint8_t *out, uint64_t capacity, uint64_t *length);
/* the recorded source against a current scope (and dimensions): TR_MATH_OK or TR_MATH_STALE with the reasons;
 * TR_MATH_UNSUPPORTED for a record without a source */
int32_t tr_persist_fresh(const tr_persist_record *, const tr_scope *current, const tr_cert_dim *dims, int32_t ndims,
                         tr_cert_staleness *out);
/* typed views (pointers into the record, valid while it lives) */
int32_t tr_persist_bank_view(const tr_persist_record *, tr_persist_bank_basis *basis, tr_bank_states *states);
int32_t tr_persist_histogram_view(const tr_persist_record *, tr_dist_spec *spec, tr_dist_states *states);
int32_t tr_persist_moments_view(const tr_persist_record *, tr_lin_spec *spec, int32_t *marginals, tr_lin_states *states);
/* rehydration: origin must be the recorded source (exact freshness, else TR_MATH_STALE with *stale written) */
tr_bank *tr_persist_to_bank(const tr_persist_record *, const tr_scope *origin, const tr_math_budget *,
                            tr_math_refusal *, tr_cert_staleness *stale, int32_t *error);
tr_dist_hist *tr_persist_to_histogram(const tr_persist_record *, const tr_scope *origin, const tr_math_budget *,
                                      tr_math_refusal *, tr_cert_staleness *stale, int32_t *error);
tr_lin_moments *tr_persist_to_moments(const tr_persist_record *, const tr_scope *origin, const tr_math_budget *,
                                      tr_math_refusal *, tr_cert_staleness *stale, int32_t *error);
tr_bigint *tr_persist_to_bigint(const tr_persist_record *, const tr_math_budget *, tr_math_refusal *, int32_t *error);
tr_bigq *tr_persist_to_bigq(const tr_persist_record *, const tr_math_budget *, tr_math_refusal *, int32_t *error);
int32_t tr_persist_to_gfp(const tr_persist_record *, tr_gfp_elem *out);
tr_cert *tr_persist_to_certificate(const tr_persist_record *, const tr_math_budget *, tr_math_refusal *,
                                   int32_t *error);
void tr_persist_record_free(tr_persist_record *);

/* ------------------------------------------------------------------ the CUDA library (math_persist_gpu.cu) */
/* the position-weighted sums of n words on the device (the same as tr_persist_checksum); budget: bytes = 8 n */
int tr_mgpu_persist_checksum(const uint64_t *words, uint64_t n, const tr_math_budget *budget, uint64_t out[2],
                             tr_math_refusal *refusal);
#ifdef __cplusplus
}
#endif
#endif
