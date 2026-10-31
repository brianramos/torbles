/* GEN-002: convolutions of generating states with explicit independence, keyed (conditional) products, joint states,
 * and bounded truncated formal series, on the CPU (math_conv.c, built on GEN-001's generating polynomials,
 * math_gen.h) and in CUDA (math_conv_gpu.cu: batched exact convolutions, per-key conditional products, walk tables).
 * SPDX-License-Identifier: Apache-2.0
 *
 * ---- Factors and their sources -----------------------------------------------------------------------------------
 * A factor is a GEN-001 generating polynomial G(z) = sum c(v) z^v (one or two variables, exact big-integer
 * coefficients; G = z^shift P(z) for negative values) together with the set of data sources its values come from. A
 * source identifies rows: for a table scope (math_scope.h) the engine table (its address and the caller's 16-byte
 * identity token: every scope over that table, whatever its columns, selection or grouping, has the same rows); for
 * an array scope the scope's row descriptor (tr_scope_key part TR_SCOPE_KEY_ROWS: the selection and the data, so
 * equal arrays with the same selection are the same source). A factor of a histogram (tr_conv_factor_from_hist: one
 * key = one group of the histogram's scope, or TR_CONV_ALL: the marginal over every key) has that scope's source; a
 * factor of explicit terms (tr_conv_factor_from_terms, _from_gen) has none (an abstract distribution); a product has
 * the union of its factors' sources. A source is keyed when the scope it was taken from has more than one group.
 *
 * ---- Products: the independence assumption is explicit and checked ------------------------------------------------
 * tr_conv_product multiplies m factors f_i, each with a multiplicity k_i (k_i copies). Its value is the product of the
 * generating polynomials, the exact convolution of the coefficients: with combine TR_CONV_SUM (same variables) the
 * distribution of the sum of one draw per copy, drawn independently (equivalently the full Cartesian product of the
 * bags: the coefficient of z^s counts the tuples whose values sum to s, masses multiplied); with TR_CONV_OUTER (two
 * one-variable factors) the joint distribution of the pair (z_1 for the first, z_2 for the second): their Cartesian
 * product as a two-variable state. Sums of k i.i.d. draws are one factor with multiplicity k. order > 0 (one
 * variable, SUM) keeps only the exponents e < order of P (the sums capped at an order; exact up to it).
 * Two copies whose factors share a source (two copies of one data factor, or factors of one table or array scope)
 * are linked: their rows can be paired (by row, or by a key), so their joint distribution is in general not the
 * product of the marginals (a key-conditioned join, or dependent events such as two columns of the same rows). Such a
 * product is computed only under an independence declaration (tr_conv_declare) whose sources cover every shared
 * source; otherwise it is refused with TR_MATH_DOMAIN_ERROR and tr_conv_check names the pair of factors, the shared
 * source, whether it is keyed and the reason (TR_CONV_NEEDS_DECLARATION: none given; TR_CONV_NOT_COVERED: the given
 * declaration covers other sources). The declaration is a statement of the caller (the model: independent draws, the
 * full Cartesian product), never inferred; the result records that it was declared. Factors of different sources (or
 * without one) combine without a declaration: their product is their Cartesian product.
 * The supported exact alternatives for linked inputs:
 *   - the conditional form (tr_conv_conditional): two keyed families (per key g, the generating polynomial of key g
 *     of a 1-D histogram; tr_conv_family_from_hist) give sum over g of A_g(z) B_g(z) (SUM) or A_g(z_1) B_g(z_2)
 *     (OUTER): the distribution over the pairs of rows with equal keys, i.e. of a.x + b.y over the key-equijoin
 *     A JOIN B ON a.key = b.key (rows with a NULL value are counted apart: info.nulls). Families of one source must
 *     have the same grouping (exact TR_SCOPE_KEY_GROUPS descriptors); families of different sources the same number of
 *     keys, key g of one being key g of the other (TR_CONV_KEYS_DIFFER otherwise).
 *   - the joint state (tr_conv_joint_sum): the two-variable factor of a 2-D histogram of the same rows gives
 *     G(z, z) = sum over the rows of z^(x + y): the per-row sum of dependent coordinates (the conditional form with the
 *     finest key, the row). tr_conv_marginal gives its marginals (factors of the same source: linked).
 *
 * ---- Budgets (math_budgets.h), checked before allocation ----------------------------------------------------------
 *   factor of a histogram: GEN-001's (support = stored terms, bytes) after the source copy (bytes) and, for
 *     TR_CONV_ALL over several keys, DIST-001's rollup
 *   product: variables, then the chain's support growth and pairs before the first product: support = the bound
 *     min(box, prod of terms) of every partial product (box: spans add, capped by order), tensor = the largest pairs
 *     of a step; each GEN-001 step (powers: their whole chain; products: tr_gen_mul_need) checks its own exact needs,
 *     bytes included, before it allocates; a refusal leaves nothing behind. tr_conv_product_need reports the chain
 *     bound, and for one factor (a power) or two factors of multiplicity 1 GEN-001's exact needs (bytes included).
 *   family: variables = 1, states = keys, support = the largest key's dense span, bytes = the arrays and the source
 *   conditional: variables, states = keys, tensor = sum over keys of na nb, support = the per-key output
 *     coefficients (SUM: na + nb - 1 per key; OUTER: na nb), bytes = the 192-bit scratch, the terms and GEN-001's
 *     adoption (TR_CONV_GEN_TERMS_BYTES; tr_gen_from_terms checks its own needs again); the CUDA route: bytes = the
 *     larger of that and the device copies (TR_CONV_NEED_DEVICE)
 *   batch: output = the output coefficients, tensor = sum of na nb; CUDA: bytes = the device copies
 *   series: degree = order - 1, support = order, bytes; products also tensor = order^2, powers iterations = k;
 *     walks (longest walk L): iterations = L, output = (L + 1) x order when the rows are kept
 * Over budget: TR_MATH_BUDGET with the tr_math_refusal. A value outside int64 is TR_MATH_OVERFLOW.
 *
 * ---- Exactness --------------------------------------------------------------------------------------------------
 * Products are GEN-001's (exact big integers, any size). Family coefficients are int64 (counts; weight sums that fit,
 * else TR_MATH_OVERFLOW); every convolution coefficient of int64 arrays is accumulated exactly in 192-bit two's
 * complement (|sum| <= 2^64 2^126 < 2^191: no bound to prove) by the shared tr_conv_coef192, on the CPU and in the
 * kernels. A sum over keys is adopted as a GEN-001 state by tr_gen_from_terms, whose duplicate merge is exact (up to
 * 3 limbs): every per-key product coefficient must fit int128 (always true for counts, at most n_g^2 < 2^64), else
 * TR_MATH_OVERFLOW (never truncated).
 *
 * ---- Bounded truncated series -------------------------------------------------------------------------------------
 * tr_conv_series: f = c_0 + c_1 z + ... + c_(N-1) z^(N-1) + O(z^N) with int64 coefficients and a declared bound B
 * (0 <= B <= 2^63 - 1): every coefficient satisfies |c_e| <= B. Operations are exact up to the order N: each output
 * coefficient is the exact 192-bit sum, then checked against the declared bound of the result (TR_MATH_OVERFLOW with
 * the first violating exponent otherwise; nothing is clipped). Two series combine only with the same order
 * (TR_MATH_CARRIER_ERROR). pow(k) bounds every power a^j, j <= k. Walk tables (tr_conv_series_walks): with a step series s
 * (the generating polynomial of one step's increment), row l = s^l mod z^N counts the walks of exactly l steps by
 * their total, and total = sum over l <= L of the rows (walks of length at most L). L = TR_CONV_UNBOUNDED is allowed
 * only when s has no constant term (then s^l = 0 mod z^N for l >= N and total = 1 / (1 - s) mod z^N); with a nonzero
 * constant term the walk count of every total diverges: TR_MATH_NO_SOLUTION (never a finite answer). Series made from
 * a factor keep its sources: products and walks of linked series need a declaration like factor products.
 *
 * ---- CUDA (math_conv_gpu.cu) -----------------------------------------------------------------------------------
 * tr_mgpu_conv_batch: a batch of convolutions of int64 arrays (pair i = (A_i, B_i), CSR layout, optionally truncated):
 * one work item per output coefficient, the exact 192-bit sum of tr_conv_coef192, written to its own slot (3 words).
 * tr_mgpu_conv_conditional: the same kernel over the keys of two families (pair g = key g): the per-key products in
 * parallel over keys; the host adopts the readback with tr_conv_conditional_adopt, exactly as the CPU path adopts
 * tr_conv_batch_run's output. tr_mgpu_conv_walks: one launch per length, one work item per coefficient (192-bit sum,
 * bound check, the first violation by an atomic minimum), then the totals. Every slot has one writer (or a min
 * atomic), so nothing depends on the order of the threads; the results equal the CPU functions bit for bit.
 *
 * Errors: TR_MATH_INVALID, _MEMORY, _OVERFLOW, _BUDGET, _DOMAIN_ERROR (linked inputs without a covering declaration;
 * key domains or decimal scales that differ), _CARRIER_ERROR (series of different orders), _NO_SOLUTION (an unbounded
 * walk sum), _UNSUPPORTED (more than TR_CONV_MAX_FACTORS factors or TR_CONV_MAX_SOURCES sources, OUTER of other than
 * two one-variable factors, a 2-D family), TR_MATH_DEVICE. Constructors return NULL and set *error; objects are
 * immutable and individually owned; read-only calls may run concurrently. */
#ifndef TORBLES_MATH_CONV_H
#define TORBLES_MATH_CONV_H
#include <stdint.h>
#include "math_num.h"
#include "math_budgets.h"
#include "math_scope.h"
#include "math_dist.h"
#include "math_gen.h"
#include "math_carriers.h"
#ifdef __cplusplus
extern "C" {
#endif

#define TR_CONV_ABI 1
#define TR_CONV_MAX_FACTORS 16
#define TR_CONV_MAX_SOURCES 16
#define TR_CONV_ALL UINT64_MAX            /* a factor of every key: the marginal over the keys */
#define TR_CONV_UNBOUNDED UINT32_MAX      /* walks of any length */
#define TR_CONV_NONE UINT64_MAX
enum { TR_CONV_SUM = 1, TR_CONV_OUTER = 2 };                                    /* how the values combine */
enum { TR_CONV_FREE = 0, TR_CONV_DATA = 1, TR_CONV_DERIVED = 2 };               /* origin kinds */
enum { TR_CONV_MODE_BASE = 0, TR_CONV_MODE_INDEPENDENT = 1, TR_CONV_MODE_CONDITIONAL = 2, TR_CONV_MODE_JOINT = 3,
       TR_CONV_MODE_MARGINAL = 4, TR_CONV_MODE_SERIES = 5 };                    /* how a factor was made */
enum { TR_CONV_OK = 0, TR_CONV_NEEDS_DECLARATION = 1, TR_CONV_NOT_COVERED = 2, TR_CONV_KEYS_DIFFER = 3,
       TR_CONV_GROUPING_DIFFERS = 4, TR_CONV_SCALES_DIFFER = 5 };               /* tr_conv_check reasons */
enum { TR_CONV_NEED_CPU = 0, TR_CONV_NEED_DEVICE = 1 };                       /* need targets */
/* the bytes GEN-001's tr_gen_from_terms checks for n terms of nvars variables (2 x 24-byte terms, 8 + 8 nvars bytes
 * of arrays and a 56-byte coefficient per term; the object bounded by 256 bytes): the adoption of summed states */
#define TR_CONV_GEN_TERMS_BYTES(n, nvars) tr_budget_add(tr_budget_mul((n), 112 + 8 * (uint64_t)(nvars)), 256)

/* a factor's description (128 bytes) */
typedef struct {
    uint64_t sources;          /* data sources of its values (0: none) */
    uint64_t keys, key;        /* data factor: its scope's groups and the key it covers (TR_CONV_ALL: every key) */
    uint64_t tuples, nulls;    /* counted (see flags): the tuples it combines (rows, or pairs ...) and those with a NULL
                                  value (not in G); G(1) = tuples - nulls */
    uint64_t draws;            /* copies multiplied (1 for a base factor) */
    uint64_t digest[2];        /* the first source's digest (display; identity is the exact source) */
    int32_t kind, mode, keyed, declared;   /* TR_CONV_FREE/_DATA/_DERIVED; TR_CONV_MODE_*; a keyed source; declared */
    int32_t combine, counted, nvars, mass; /* TR_CONV_SUM/_OUTER; tuples/nulls are counts; variables; TR_DIST_MASS_* */
    int32_t col[2];            /* data factor: the scope columns of the variables (-1) */
    int32_t scale[2];          /* decimal scales of the values (the factor's; its GEN-001 state carries scale 0) */
    int32_t reserved[4];
} tr_conv_info;

/* why a product was refused (32 bytes) */
typedef struct {
    int32_t reason;            /* TR_CONV_OK or TR_CONV_* */
    int32_t first, second;     /* the factors (indices) concerned (-1: none) */
    int32_t keyed;             /* the shared source is keyed */
    uint64_t digest[2];        /* the shared source's digest */
} tr_conv_check;

/* a declaration's description (24 bytes) */
typedef struct { uint64_t sources, digest[2]; } tr_conv_decl_info;

/* a family as the kernels read it (64 bytes): key g's coefficients coef[offset[g] .. offset[g + 1]) at the values
 * shift[g] + e (e = 0, 1, ...) */
typedef struct {
    uint64_t keys, ncoef, maxspan;
    const uint64_t *offset;    /* [keys + 1] */
    const int64_t *shift;      /* [keys] */
    const int64_t *coef;       /* [ncoef] */
    int32_t mass, scale;
    uint64_t reserved;
} tr_conv_family_view;

/* a family's description (72 bytes) */
typedef struct {
    uint64_t keys, present, ncoef, maxspan, rows, nulls, bytes;
    uint64_t digest[2];
} tr_conv_family_info;

/* a batch of convolutions (72 bytes): pair i = (A_i, B_i), A_i = a[aoff[i] .. aoff[i + 1]) (exponents 0, 1, ...);
 * output i = ooff[i] .. ooff[i + 1] (na + nb - 1 coefficients, 0 when either is empty, at most order when order > 0),
 * 3 words (192-bit two's complement, least significant first) per coefficient */
typedef struct {
    uint64_t npairs, order, total, pairs;   /* total = ooff[npairs]; pairs = sum na nb (the tensor need) */
    const uint64_t *aoff, *boff, *ooff;
    const int64_t *a, *b;
} tr_conv_batch;

/* a bounded series' description (40 bytes) and a walk table's (40 bytes) */
typedef struct { uint64_t order, sources; int64_t bound, maxabs, shift; } tr_conv_series_info;
typedef struct {
    uint64_t lengths;          /* rows computed: L + 1 (L = the longest walk) */
    uint64_t violation;        /* first coefficient past the bound: l * order + e (row l; total: lengths * order + e) */
    int32_t status, reserved;  /* 0, TR_MATH_OVERFLOW (a coefficient past the bound) */
    int64_t bound, maxabs;     /* the declared bound; the largest |coefficient| computed */
} tr_conv_walks;

/* ---------------------------------------------------------------- the shared kernels (CPU and CUDA) */
/* c_k = sum_i a_i b_(k-i) of two dense int64 arrays, exactly, in 192-bit two's complement w[0..2] */
TR_NUM_HD void tr_conv_coef192(const int64_t *a, uint64_t na, const int64_t *b, uint64_t nb, uint64_t k, uint64_t *w) {
    w[0] = w[1] = w[2] = 0;
    if (!na || !nb || k > na + nb - 2) return;
    uint64_t lo = k >= nb ? k - nb + 1 : 0, hi = k < na ? k : na - 1;
    for (uint64_t i = lo; i <= hi; i++) tr_wide_add128(w, (__int128)a[i] * (__int128)b[k - i]);
}
/* 1 when the 192-bit value fits int128 / int64 (and then *v) */
TR_NUM_HD int tr_conv_fits128(const uint64_t *w, __int128 *v) {
    uint64_t ext = (w[1] >> 63) ? ~(uint64_t)0 : 0;
    if (w[2] != ext) return 0;
    *v = (__int128)(((unsigned __int128)w[1] << 64) | w[0]);
    return 1;
}
TR_NUM_HD int tr_conv_fits64(const uint64_t *w, int64_t *v) {
    uint64_t ext = (w[0] >> 63) ? ~(uint64_t)0 : 0;
    if (w[1] != ext || w[2] != ext) return 0;
    *v = (int64_t)w[0];
    return 1;
}
/* |v| <= bound (0 <= bound <= 2^63 - 1) for a 192-bit value: 1 and *v, else 0 */
TR_NUM_HD int tr_conv_bounded(const uint64_t *w, int64_t bound, int64_t *v) {
    int64_t x;
    if (!tr_conv_fits64(w, &x) || x == INT64_MIN) return 0;
    if ((x < 0 ? -x : x) > bound) return 0;
    *v = x;
    return 1;
}
/* the output pair of coefficient t of a batch: the largest i with ooff[i] <= t (ooff nondecreasing, t < total) */
TR_NUM_HD uint64_t tr_conv_batch_pair(const uint64_t *ooff, uint64_t npairs, uint64_t t) {
    uint64_t lo = 0, hi = npairs;          /* ooff[lo] <= t < ooff[hi] */
    while (hi - lo > 1) {
        uint64_t m = lo + (hi - lo) / 2;
        if (ooff[m] <= t) lo = m; else hi = m;
    }
    return lo;
}
/* the output length of one pair */
TR_NUM_HD uint64_t tr_conv_batch_len(uint64_t na, uint64_t nb, uint64_t order) {
    if (!na || !nb) return 0;
    uint64_t n = na + nb - 1;
    return order && n > order ? order : n;
}
/* coefficient t of a batch, 192-bit */
TR_NUM_HD void tr_conv_batch_one(const tr_conv_batch *b, uint64_t t, uint64_t *w) {
    uint64_t i = tr_conv_batch_pair(b->ooff, b->npairs, t);
    tr_conv_coef192(b->a + b->aoff[i], b->aoff[i + 1] - b->aoff[i], b->b + b->boff[i], b->boff[i + 1] - b->boff[i],
                    t - b->ooff[i], w);
}
/* host: the output offsets ooff[npairs + 1] of a batch, *total and *pairs; TR_MATH_INVALID for decreasing offsets */
static inline int32_t tr_conv_batch_plan(uint64_t npairs, const uint64_t *aoff, const uint64_t *boff, uint64_t order,
                                         uint64_t *ooff, uint64_t *total, uint64_t *pairs) {
    uint64_t t = 0, p = 0;
    if (ooff) ooff[0] = 0;
    for (uint64_t i = 0; i < npairs; i++) {
        if (aoff[i + 1] < aoff[i] || boff[i + 1] < boff[i]) return -1;
        uint64_t na = aoff[i + 1] - aoff[i], nb = boff[i + 1] - boff[i];
        t = tr_budget_add(t, tr_conv_batch_len(na, nb, order));
        p = tr_budget_add(p, tr_budget_mul(na, nb));
        if (ooff) ooff[i + 1] = t;
    }
    *total = t; *pairs = p;
    return 0;
}
/* host: 0 when a batch is consistent (offsets, totals, the output layout of tr_conv_batch_plan), else TR_MATH_INVALID */
static inline int32_t tr_conv_batch_check(const tr_conv_batch *b) {
    if (!b || (b->npairs && (!b->aoff || !b->boff || !b->ooff))) return -1;
    if (!b->npairs) return b->total || b->pairs ? -1 : 0;
    uint64_t total = 0, pairs = 0;
    if (tr_conv_batch_plan(b->npairs, b->aoff, b->boff, b->order, NULL, &total, &pairs)) return -1;
    if (total != b->total || pairs != b->pairs || b->ooff[0] != 0 || b->ooff[b->npairs] != total) return -1;
    if ((b->aoff[b->npairs] && !b->a) || (b->boff[b->npairs] && !b->b) || b->aoff[0] || b->boff[0]) return -1;
    for (uint64_t i = 0; i < b->npairs; i++)
        if (b->ooff[i + 1] < b->ooff[i] ||
            b->ooff[i + 1] - b->ooff[i] != tr_conv_batch_len(b->aoff[i + 1] - b->aoff[i], b->boff[i + 1] - b->boff[i], b->order))
            return -1;
    return 0;
}
/* host: the bytes a device batch copies (inputs, offsets, outputs) */
static inline uint64_t tr_conv_batch_bytes(const tr_conv_batch *b) {
    uint64_t na = b->npairs ? b->aoff[b->npairs] : 0, nb = b->npairs ? b->boff[b->npairs] : 0;
    uint64_t bytes = tr_budget_mul(tr_budget_add(na, nb), 8);
    bytes = tr_budget_add(bytes, tr_budget_mul(tr_budget_add(b->npairs, 1), 24));
    return tr_budget_add(bytes, tr_budget_mul(b->total, 24));
}
/* host: the bytes the device walks need (the step, the rows, the totals, the violation word) */
static inline uint64_t tr_conv_walks_bytes(uint64_t order, uint64_t lengths) {
    return tr_budget_add(tr_budget_mul(tr_budget_add(lengths, 2), tr_budget_mul(order, 8)), 16);
}

/* ---------------------------------------------------------------- objects (libalg) */
typedef struct tr_conv_decl_s tr_conv_decl;
typedef struct tr_conv_factor_s tr_conv_factor;
typedef struct tr_conv_family_s tr_conv_family;
typedef struct tr_conv_series_s tr_conv_series;

/* an independence declaration over the sources of n >= 1 scopes (duplicates merge) */
tr_conv_decl *tr_conv_declare(const tr_scope *const *scopes, uint32_t n, const tr_math_budget *, tr_math_refusal *,
                              int32_t *error);
int32_t tr_conv_decl_get_info(const tr_conv_decl *, tr_conv_decl_info *);
/* 1 when the declaration covers the scope's source, else 0 */
int32_t tr_conv_decl_covers(const tr_conv_decl *, const tr_scope *);
void tr_conv_decl_free(tr_conv_decl *);

/* the factor of one key (or TR_CONV_ALL) of a 1-D / 2-D histogram of `scope` (the histogram's own scope: same
 * descriptor digest, groups and rows, else TR_MATH_INVALID); mass, repr, shift as tr_gen_from_hist */
tr_conv_factor *tr_conv_factor_from_hist(const tr_dist_hist *, const tr_scope *scope, uint64_t key, int32_t mass,
                                         int32_t repr, const int64_t *shift, const tr_math_budget *, tr_math_refusal *,
                                         int32_t *error);
/* a factor without a source: explicit terms (tr_gen_from_terms) or a copy of a GEN-001 polynomial */
tr_conv_factor *tr_conv_factor_from_terms(int32_t nvars, uint64_t n, const int64_t *values, const tr_math_i128 *coef,
                                          int32_t repr, const int64_t *shift, const tr_math_budget *,
                                          tr_math_refusal *, int32_t *error);
tr_conv_factor *tr_conv_factor_from_gen(const tr_gen_poly *, const tr_math_budget *, tr_math_refusal *, int32_t *error);
int32_t tr_conv_factor_get_info(const tr_conv_factor *, tr_conv_info *);
/* the factor's polynomial (borrowed: GEN-001's requests read it; valid while the factor lives) and an owned copy */
const tr_gen_poly *tr_conv_factor_poly(const tr_conv_factor *);
tr_gen_poly *tr_conv_factor_poly_copy(const tr_conv_factor *, int32_t repr, const tr_math_budget *, tr_math_refusal *,
                                      int32_t *error);
/* source i of a factor: its digest and whether it is keyed */
int32_t tr_conv_factor_source(const tr_conv_factor *, uint64_t i, uint64_t *digest, int32_t *keyed);
void tr_conv_factor_free(tr_conv_factor *);

/* the independence check of a product (see Products): 0, or TR_MATH_DOMAIN_ERROR with *check filled */
int32_t tr_conv_check_product(const tr_conv_factor *const *f, const uint32_t *mult, uint32_t m,
                              const tr_conv_decl *decl, tr_conv_check *check);
/* prod_i f_i^(mult_i) (mult NULL: every 1); combine SUM or OUTER; order (SUM, one variable): exponents < order */
tr_conv_factor *tr_conv_product(const tr_conv_factor *const *f, const uint32_t *mult, uint32_t m, int32_t combine,
                                uint64_t order, const tr_conv_decl *decl, const tr_math_budget *, tr_math_refusal *,
                                tr_conv_check *check, int32_t *error);
int32_t tr_conv_product_need(const tr_conv_factor *const *f, const uint32_t *mult, uint32_t m, int32_t combine,
                             uint64_t order, const tr_math_budget *, tr_math_need *need, uint64_t *terms);
/* the joint state: G(z, z) of a two-variable factor (per tuple, the sum of its coordinates) / the marginal G(z_keep, 1) */
tr_conv_factor *tr_conv_joint_sum(const tr_conv_factor *, const tr_math_budget *, tr_math_refusal *, int32_t *error);
tr_conv_factor *tr_conv_marginal(const tr_conv_factor *, int32_t keep, const tr_math_budget *, tr_math_refusal *,
                                 int32_t *error);

/* a keyed family: per group of a 1-D histogram of `scope`, its dense int64 coefficients (count, or weight sums that
 * fit int64) over [min, max] of its present values */
tr_conv_family *tr_conv_family_from_hist(const tr_dist_hist *, const tr_scope *scope, int32_t mass,
                                         const tr_math_budget *, tr_math_refusal *, int32_t *error);
int32_t tr_conv_family_get_info(const tr_conv_family *, tr_conv_family_info *);
int32_t tr_conv_family_get_view(const tr_conv_family *, tr_conv_family_view *);
/* key g's factor, or the marginal over every key (TR_CONV_ALL); the family's source */
tr_conv_factor *tr_conv_family_factor(const tr_conv_family *, uint64_t key, int32_t repr, const tr_math_budget *,
                                      tr_math_refusal *, int32_t *error);
void tr_conv_family_free(tr_conv_family *);
/* the conditional form (see Products): the key domains are checked first (TR_MATH_DOMAIN_ERROR, *check) */
tr_conv_factor *tr_conv_conditional(const tr_conv_family *a, const tr_conv_family *b, int32_t combine, int32_t repr,
                                    const tr_math_budget *, tr_math_refusal *, tr_conv_check *check, int32_t *error);
/* target TR_CONV_NEED_CPU: the CPU path (the 192-bit scratch, the terms and GEN-001's adoption); TR_CONV_NEED_DEVICE
 * (SUM): bytes = the larger of the host stage and the device copies */
int32_t tr_conv_conditional_need(const tr_conv_family *a, const tr_conv_family *b, int32_t combine, int32_t target,
                                 tr_math_need *need);
/* the key-domain check alone: 0, or TR_MATH_DOMAIN_ERROR with *check filled */
int32_t tr_conv_conditional_check(const tr_conv_family *a, const tr_conv_family *b, int32_t combine,
                                  tr_conv_check *check);
/* the device route of SUM: the batch over the keys (ooff [keys + 1]) and the adoption of its 192-bit readback */
int32_t tr_conv_conditional_plan(const tr_conv_family *a, const tr_conv_family *b, uint64_t *ooff, uint64_t *total,
                                 uint64_t *pairs);
tr_conv_factor *tr_conv_conditional_adopt(const tr_conv_family *a, const tr_conv_family *b, const uint64_t *ooff,
                                          const int64_t *out, int32_t repr, const tr_math_budget *, tr_math_refusal *,
                                          tr_conv_check *check, int32_t *error);

/* the CPU reference of the batch kernel: out[3 * total]; budget: output = total, tensor = pairs */
int32_t tr_conv_batch_run(const tr_conv_batch *, int64_t *out, const tr_math_budget *, tr_math_refusal *);
/* tr_conv_batch_plan exported for the bindings (ooff [npairs + 1]) */
int32_t tr_conv_batch_layout(uint64_t npairs, const uint64_t *aoff, const uint64_t *boff, uint64_t order,
                             uint64_t *ooff, uint64_t *total, uint64_t *pairs);

/* bounded series (see Bounded truncated series) */
tr_conv_series *tr_conv_series_create(const int64_t *coef, uint32_t count, uint32_t order, int64_t bound,
                                      const tr_math_budget *, tr_math_refusal *, int32_t *error);
/* P mod z^order of a one-variable factor (G = z^shift P), with the factor's sources */
tr_conv_series *tr_conv_series_from_factor(const tr_conv_factor *, uint32_t order, int64_t bound,
                                           const tr_math_budget *, tr_math_refusal *, int32_t *error);
int32_t tr_conv_series_get_info(const tr_conv_series *, tr_conv_series_info *);
int32_t tr_conv_series_coefficients(const tr_conv_series *, int64_t *out, uint64_t capacity);
tr_conv_series *tr_conv_series_add(const tr_conv_series *, const tr_conv_series *, int64_t bound,
                                   const tr_math_budget *, tr_math_refusal *, uint64_t *violation, int32_t *error);
tr_conv_series *tr_conv_series_mul(const tr_conv_series *, const tr_conv_series *, int64_t bound,
                                   const tr_conv_decl *, const tr_math_budget *, tr_math_refusal *,
                                   tr_conv_check *check, uint64_t *violation, int32_t *error);
tr_conv_series *tr_conv_series_pow(const tr_conv_series *, uint32_t k, int64_t bound, const tr_conv_decl *,
                                   const tr_math_budget *, tr_math_refusal *, tr_conv_check *check,
                                   uint64_t *violation, int32_t *error);
/* walk table of a step series up to length L (or TR_CONV_UNBOUNDED): rows (may be NULL) [(L' + 1) x order],
 * L' = L or order - 1 (unbounded); *total (may be NULL) the series of the totals (bound: the declared bound) */
int32_t tr_conv_series_walks(const tr_conv_series *step, uint32_t max_length, int64_t bound, const tr_conv_decl *,
                             int64_t *rows, uint64_t rows_capacity, tr_conv_series **total, tr_conv_walks *info,
                             const tr_math_budget *, tr_math_refusal *, tr_conv_check *check);
/* the walk table's needs and its rows (L' + 1) */
int32_t tr_conv_series_walks_need(const tr_conv_series *step, uint32_t max_length, int32_t keep_rows,
                                  tr_math_need *need, uint64_t *lengths);
/* the CPU reference of the device walks over a raw step (rows [lengths x order], total [order]) */
int32_t tr_conv_walks_run(const int64_t *step, uint64_t order, uint64_t lengths, int64_t bound, int64_t *rows,
                          int64_t *total, tr_conv_walks *info, const tr_math_budget *, tr_math_refusal *);
/* the independence check of a series product a x b (b NULL: a^k) / of walks (k = min(L, 2)): 0 or DOMAIN_ERROR */
int32_t tr_conv_series_check(const tr_conv_series *a, const tr_conv_series *b, uint32_t k, const tr_conv_decl *,
                             tr_conv_check *check);
/* the totals of a walk table computed elsewhere (the device readback) as a series with the step's sources, each
 * coefficient checked against the bound */
tr_conv_series *tr_conv_series_walks_adopt(const tr_conv_series *step, const int64_t *total, int64_t bound,
                                           const tr_math_budget *, tr_math_refusal *, int32_t *error);
/* a one-variable factor of the series (the coefficients at the values shift + e), with its sources */
tr_conv_factor *tr_conv_series_to_factor(const tr_conv_series *, int32_t repr, const tr_math_budget *,
                                         tr_math_refusal *, int32_t *error);
void tr_conv_series_free(tr_conv_series *);

/* ---------------------------------------------------------------- the CUDA library (math_conv_gpu.cu) */
/* the same results on the device (bytes = the device copies; output and tensor as the CPU references) */
int tr_mgpu_conv_batch(const tr_conv_batch *, int64_t *out, const tr_math_budget *, tr_math_refusal *);
int tr_mgpu_conv_conditional(const tr_conv_family_view *a, const tr_conv_family_view *b, const uint64_t *ooff,
                             int64_t *out, const tr_math_budget *, tr_math_refusal *);
int tr_mgpu_conv_walks(const int64_t *step, uint64_t order, uint64_t lengths, int64_t bound, int64_t *rows,
                       int64_t *total, tr_conv_walks *info, const tr_math_budget *, tr_math_refusal *);
#ifdef __cplusplus
}
#endif
#endif
