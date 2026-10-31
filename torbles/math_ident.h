/* DISC-002: Moment-Gram polynomial identity discovery.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Over a FND-004 scope (math_scope.h) and k <= TR_IDENT_MAX_VARS observed integer columns x1..xk, a bounded monomial
 * basis m_1 < ... < m_B (every monomial of total degree <= d, or an explicit list; B <= TR_IDENT_MAX_BASIS, sorted in
 * increasing graded lexicographic order: total degree first, then the exponent of x1, then of x2, ...) gives the Gram
 * matrix
 *     G = sum over the Gram rows r of w_r m(x_r) m(x_r)^T,
 * the Gram rows being the selected rows where every variable (and the weight, when there is one) is present, w_r the
 * row weight (a scope column; 1 without one). A vector c with G c = 0 under nonnegative weights gives
 * sum w_r (c . m(x_r))^2 = c^T G c = 0, hence c . m(x_r) = 0 on every Gram row with w_r > 0: the nullspace holds every
 * polynomial identity of the basis that the data satisfy, found at once.
 *
 * The plan (tr_ident_plan, from a scope and a tr_ident_spec): the canonical basis, the distinct products
 * u = m_i m_j (the moments, sum_r w_r u(x_r)), and the representation, decided before anything is read:
 *   EXACT    every moment proven to fit 128 bits: N * W * prod_v M_v^{e_v} <= 2^127 - 1 (N the rows that can be
 *            Gram rows, W = max |w| or 1, M_v = max(|min_v|, |max_v|, 1) from the scope's extremes); moments are exact
 *            signed 128-bit sums (from a pass of this module, a FND-004 bank, or LIN-001 moments);
 *   MODULAR  otherwise (or when asked): every moment modulo nprimes primes (the largest primes below 2^62, those of
 *            LIN-002, recorded in the plan and the Gram); nothing exact beyond 128 bits is claimed.
 * width TR_IDENT_WIDTH_EXACT refuses an unproven plan (TR_MATH_OVERFLOW), TR_IDENT_WIDTH_MODULAR forces the modular
 * representation, TR_IDENT_WIDTH_AUTO chooses.
 *
 * The Gram (tr_ident_gram) records the domain: rows (Gram rows), positive / zero / negative weights among them, the
 * weight extremes and the sign class: NONNEGATIVE (no negative weight) or SIGNED (a negative weight, or the difference
 * of two Grams: tr_ident_gram_combine with sign -1, e.g. a DELTA-001 deletion).
 *
 * The nullspace (tr_ident_kernel): rank, pivots (the standard monomials) and the canonical basis, one vector per free
 * column f (the vector's leading monomial: coefficient 1 at m_f, other entries at pivot columns p < f only):
 *   EXACT Gram    LIN-002 (math_lin.h): the system [G | 0] as a modular batch (Gauss-Jordan modulo primes in
 *                 [2^61, 2^62) on the CPU or the CUDA kernel tr_mgpu_lin_eliminate), Chinese remaindering within the
 *                 Hadamard bound and the exact verification G c = 0; rank and nullspace exact (exact_rank = 1).
 *   MODULAR Gram  per recorded prime the reduced row echelon form of G mod p (tr_lin_modular_rref), the primes of the
 *                 best key (highest rank, earliest pivots) combined by CRT (FND-009) and each entry rationally
 *                 reconstructed (bounds isqrt((M - 1) / 2)); a vector whose reconstruction fails is unresolved. The
 *                 modular rank is a lower bound of the exact rank (exact_rank = 0): every candidate it proposes is
 *                 only a proposal until the exhaustive verification below; when every one verifies, the rank is
 *                 exact (the verified vectors lie in the rational kernel and are independent).
 *
 * Identities (tr_ident_set): each kernel vector as a primitive integer polynomial over the variables (denominators
 * cleared, content removed, leading coefficient positive; representable when every coefficient fits int64), or given
 * polynomials (tr_ident_candidates), with exact labels:
 *   points S     distinct points of the identity's support variables over its domain (the selected rows where those
 *                variables are present), counted exactly (64-bit fingerprints in a radix sort; rows of equal
 *                fingerprints compared value by value); terms T (nonzero coefficients); forced = max(0, T - S): a combination of T
 *                monomials vanishing on S < T points always exists, so such an identity is UNDERDETERMINED, never a
 *                discovery. The basis level reports S over the Gram rows (points of nonzero total weight) and
 *                forced = max(0, B - S) <= nullity.
 *   known        the identity lies in the span of the declared relations' multiples within the basis (u r_i for
 *                monomials u, every term in the basis): a restated definition.
 *   consequence  not known, but in the span of the declared multiples and of the multiples of the earlier generators
 *                (identities in increasing leading monomial): implied within the basis's degree bound. Otherwise the
 *                identity is a generator and its multiples join the span. Decided exactly by fraction-free
 *                elimination in 128 bits (labels = 1); an overflow leaves every label 0 (labels = 0). An identity
 *                implied only through multiples past the basis is a generator here (GEOM-001 reduces it exactly).
 *                parent (informational): an identity (-2: a declared relation) whose leading monomial divides.
 *   sos          the Gram's own verdict: CERTIFIED (exact Gram, NONNEGATIVE weights, c^T G c = 0 exactly), NONZERO
 *                (c^T G c != 0), refused SIGNED (a zero quadratic form under signed weights is not a sum of squares:
 *                cancellation) or MODULAR (c^T G c = 0 only modulo the recorded primes).
 *   discovery    verified, not underdetermined, not known, not a consequence.
 *
 * Exhaustive verification (tr_ident_verify_view; tr_mgpu_ident_verify on the device, the same evidence bit for bit):
 * every selected row of the scope; a row where a support variable is NULL is excluded (FND-007 policy SKIP), every
 * other row is evaluated exactly in W 64-bit words modulo 2^(64 W), where the host proved the bound
 * sum_t |c_t| prod_v M_v^{e_tv} <= 2^(64 W - 1) - 1 (W = 2: native 128-bit words): the value is then exact, so a zero
 * residue is a zero value and nothing wraps silently. A bound past TR_IDENT_MAX_WORDS words is undecided (every row
 * counted as overflow); a coefficient outside int64 is unrepresentable (not evaluated). The evidence is FND-007's
 * (tr_cert_evidence: checked, excluded, violations, the smallest violating row, overflow, holds).
 *
 * Certificates (tr_ident_certify): a verified identity becomes an FND-007 snapshot certificate of its POLYNOMIAL claim
 * (scope, revision, selection, NULL domains, carrier I128): by adopting the evidence (tr_cert_adopt; TR_CERT_BY_DEVICE
 * for the kernel's) when W = 2 (every term and partial sum then fits FND-007's 128-bit check), else by FND-007's own
 * exhaustive check (tr_cert_snapshot), which may refuse with TR_MATH_OVERFLOW (no certificate). A single violating row
 * rejects an identity (no certificate). tr_ident_hypothesis: the numerical value of c^T G c in binary64 (the Gram's
 * entries rounded to doubles, a fixed evaluation order, relative to sum |c_i| |c_j| |G_ij|) within a tolerance gives at
 * most an FND-007 HYPOTHESIS certificate on a sample of rows, which never authorizes an exact rewrite.
 *
 * Budgets (math_budgets.h), checked before allocation: variables = k, degree = the basis degree, support = B,
 * tensor = B^2, states = moments x primes, iterations = primes, bytes = every allocation (plan, Gram, kernel, the
 * distinct-count scratch: 33 bytes per counted row, 49 with a weight, and 512 KiB of radix counters; candidates, the
 * device copies). B > TR_IDENT_MAX_BASIS, k >
 * TR_IDENT_MAX_VARS, a degree past TR_IDENT_MAX_DEGREE or more than TR_IDENT_MAX_PRIMES primes is TR_MATH_UNSUPPORTED.
 *
 * Errors: TR_MATH_INVALID, _MEMORY, _OVERFLOW, _BUDGET, _UNSUPPORTED, _DOMAIN_ERROR (a bank or LIN state of another
 * scope or shape, combining Grams of different plans, a hypothesis whose numerical value is not near zero),
 * _NOT_CONVERGED (LIN-002 needs more primes), _STALE (checked by the Python layer). Objects are immutable and
 * individually owned; constructors return NULL and set *error. */
#ifndef TORBLES_MATH_IDENT_H
#define TORBLES_MATH_IDENT_H
#include <stdint.h>
#include <string.h>
#include "math_num.h"
#include "math_budgets.h"
#include "math_scope.h"
#include "math_bank.h"
#include "math_carriers.h"
#include "math_lin.h"
#include "math_certificates.h"
#ifdef __cplusplus
extern "C" {
#endif

#define TR_IDENT_ABI 1
#define TR_IDENT_MAX_VARS 8               /* an FND-007 claim reads at most 8 columns */
#define TR_IDENT_MAX_BASIS 63             /* LIN-002 exact systems: at most 63 unknowns */
#define TR_IDENT_MAX_DEGREE 16            /* total degree of a basis monomial (moments: twice that) */
#define TR_IDENT_MAX_MOMENTS 2016         /* 63 * 64 / 2 distinct products at most */
#define TR_IDENT_MAX_WORDS 8              /* exact evaluation width: 8 x 64 bits */
#define TR_IDENT_MAX_PRIMES 32
#define TR_IDENT_DEFAULT_PRIMES 6
#define TR_IDENT_MAX_KNOWN 16             /* declared relations */
#define TR_IDENT_MAX_CANDIDATES 256       /* given candidates of one set */
#define TR_IDENT_CHUNK 32                 /* CUDA Gram: moments per work item */
#define TR_IDENT_CAND_CHUNK 16            /* CUDA verification: candidates per work item */
#define TR_IDENT_SLICE_WORDS 64           /* CUDA: selection words per slice (4096 rows), at least */
#define TR_IDENT_MAX_SLICES 2048

enum { TR_IDENT_EXACT = 1, TR_IDENT_MODULAR = 2 };                                   /* Gram representations */
enum { TR_IDENT_WIDTH_AUTO = 0, TR_IDENT_WIDTH_EXACT = 128, TR_IDENT_WIDTH_MODULAR = -1 };
enum { TR_IDENT_NONNEGATIVE = 1, TR_IDENT_SIGNED = 2 };                              /* weight sign classes */
enum { TR_IDENT_SOURCE_FUSED = 1, TR_IDENT_SOURCE_BANK = 2, TR_IDENT_SOURCE_LIN = 3, TR_IDENT_SOURCE_ADOPTED = 4,
       TR_IDENT_SOURCE_COMBINED = 5 };
enum { TR_IDENT_KERNEL_LIN = 1, TR_IDENT_KERNEL_MODULAR = 2 };
enum { TR_IDENT_SOS_NONE = 0, TR_IDENT_SOS_CERTIFIED = 1, TR_IDENT_SOS_NONZERO = 2, TR_IDENT_SOS_SIGNED = 3,
       TR_IDENT_SOS_MODULAR = 4 };
enum { TR_IDENT_ORIGIN_KERNEL = 1, TR_IDENT_ORIGIN_GIVEN = 2, TR_IDENT_ORIGIN_DECLARED = 3 };
enum { TR_IDENT_PENDING = 0, TR_IDENT_VERIFIED = 1, TR_IDENT_REJECTED = 2, TR_IDENT_UNDECIDED = 3,
       TR_IDENT_UNREPRESENTABLE = 4, TR_IDENT_UNRESOLVED = 5 };                        /* identity statuses */
enum { TR_IDENT_NEED_CPU = 0, TR_IDENT_NEED_DEVICE = 1, TR_IDENT_NEED_ADOPT = 2 };

/* ------------------------------------------------------------------ structs crossing the ctypes boundary */
typedef struct {
    int32_t nvars;                       /* k: 1..TR_IDENT_MAX_VARS */
    int32_t var[TR_IDENT_MAX_VARS];      /* scope columns of x1..xk (distinct) */
    int32_t weight;                      /* scope column of the row weights, or -1 (each selected row weighs 1) */
    int32_t degree;                      /* nterms == 0: every monomial of total degree <= degree */
    int32_t nterms;                      /* > 0: an explicit basis of nterms distinct monomials (exponents) */
    int32_t width;                       /* TR_IDENT_WIDTH_* */
    int32_t nprimes;                     /* modular representation: primes (0: TR_IDENT_DEFAULT_PRIMES) */
    int32_t reserved;                    /* 0 */
    const uint8_t *exponents;            /* nterms x nvars, any order (sorted canonically by the plan) */
} tr_ident_spec;                         /* 72 bytes */

/* the plan as the CPU loop and the CUDA kernel read it (pointers into the plan) */
typedef struct {
    int32_t nvars, nbasis, nmoments, weight;    /* weight: scope column or -1 */
    int32_t var[TR_IDENT_MAX_VARS];
    int32_t repr, nprimes;                      /* TR_IDENT_EXACT / _MODULAR; primes of the modular representation */
    const uint8_t *basis;                       /* nbasis x nvars exponents, increasing graded lexicographic order */
    const int32_t *bparent;                     /* [nbasis] basis index p with m_b = m_p * x_bvar, or -1 (direct) */
    const int32_t *bvar;                        /* [nbasis] the variable of that step */
    const int32_t *mleft, *mright;              /* [nmoments] moment u = m_left * m_right */
    const tr_gfp *primes;                       /* [nprimes] (modular) */
} tr_ident_moment_view;                         /* 104 bytes */

typedef struct {
    int32_t nvars, nbasis, nmoments, degree;    /* degree: the largest basis degree */
    int32_t repr, nprimes, proven, weight;      /* proven: every moment fits 128 bits */
    uint64_t primes[TR_IDENT_MAX_PRIMES];       /* the moduli (0 beyond nprimes) */
    uint64_t bytes;
} tr_ident_plan_info;                           /* 296 bytes */

/* the readback of the CUDA Gram (and the CPU pass's own states): per moment and slot (1 exact, else one per prime)
 * two words lo, hi of a signed 128-bit sum (EXACT: the exact moment; MODULAR: the exact sum of the residues, < 2^94) */
typedef struct {
    int32_t nmoments, nslots;
    uint64_t domain, positive, zero, negative;  /* Gram rows; by the sign of their weight */
    int64_t wmin, wmax;                         /* weight extremes over the Gram rows (unweighted: 1, 1) */
    int64_t *acc;                               /* nmoments x nslots x 2 */
} tr_ident_gram_states;                         /* 64 bytes */

typedef struct {
    uint64_t rows, selected, domain;            /* the scope's rows and selected rows; Gram rows */
    uint64_t positive, zero, negative;
    int64_t wmin, wmax;
    uint64_t bytes;
    uint64_t digest[2];                         /* the scope's descriptor digest (0 when combined) */
    int32_t nvars, nbasis, nmoments, degree;
    int32_t repr, nprimes, sign, source;        /* TR_IDENT_EXACT / _MODULAR; NONNEGATIVE / SIGNED; TR_IDENT_SOURCE_* */
    int32_t weight, reserved;
    uint64_t primes[TR_IDENT_MAX_PRIMES];
} tr_ident_gram_info;                           /* 384 bytes */

typedef struct {
    int32_t nbasis, rank, nullity, method;      /* TR_IDENT_KERNEL_* */
    int32_t exact_rank, primes, agreeing, unresolved;   /* primes used / agreeing (modular); vectors not reconstructed */
    uint64_t pivots;                            /* bit j: m_j is a pivot (standard) monomial */
    uint64_t free;                              /* bit j: m_j leads a kernel vector */
    uint64_t bytes;
} tr_ident_kernel_info;                         /* 56 bytes */

/* a polynomial over the spec's variables (exponents nterms x nvars, int64 coefficients) */
typedef struct {
    uint64_t nterms;
    const uint8_t *exponents;
    const int64_t *coefficients;
} tr_ident_poly;                                /* 24 bytes */

/* the candidates as the CPU loop and the kernel read them */
typedef struct {
    int32_t ncand, nvars;
    int32_t var[TR_IDENT_MAX_VARS];             /* scope columns of the variables */
    uint64_t nterms;                            /* total terms */
    const uint64_t *offset;                     /* [ncand + 1] first term of each candidate */
    const uint32_t *support;                    /* [ncand] bit v: variable v appears (its NULL excludes a row) */
    const int32_t *words;                       /* [ncand] evaluation words (2..8; 0: not evaluated) */
    const uint8_t *exponents;                   /* nterms x nvars */
    const int64_t *coefficients;                /* nterms */
} tr_ident_cand_view;                           /* 88 bytes */

typedef struct {
    int32_t index, origin;                      /* position; TR_IDENT_ORIGIN_* */
    int32_t lead, vector;                       /* basis index of the leading monomial (-1: none); kernel vector (-1) */
    int32_t nterms, ncols;                      /* T; support variables */
    uint32_t support;                           /* bit v: variable v appears */
    int32_t words, representable, resolved;     /* evaluation width (0: undecided); int64 coefficients; reconstructed */
    int32_t known, consequence, parent;         /* labels; parent: identity index, -2 a declared monomial, -1 none */
    int32_t underdetermined, sos, degree;       /* T > S; TR_IDENT_SOS_*; total degree */
    uint64_t points, forced, domain;            /* S; max(0, T - S); rows of its domain */
    uint64_t bound_bits;                        /* bits of the evaluation bound (0 when unrepresentable) */
} tr_ident_item;                                /* 96 bytes */

typedef struct {
    int32_t nvars, nbasis, rank, nullity;
    int32_t exact_rank, repr, sign, ncand;
    int32_t nknown, known_rank, generators, consequences;
    int32_t known, underdetermined, unrepresentable, unresolved;
    int32_t labels, reserved;                   /* labels: 1 decided exactly; 0 undecided (128-bit elimination overflow) */
    uint64_t points, forced, domain;            /* the basis level: S (nonzero total weight), max(0, B - S), Gram rows */
    uint64_t bytes;
} tr_ident_set_info;                            /* 104 bytes */

typedef struct {
    int32_t exact, sos;                         /* 1: the exact value is given (an exact Gram); TR_IDENT_SOS_* */
    int32_t sign, nlimbs;                       /* sign of the exact value, its magnitude limbs written */
    double numeric, scale, relative;            /* binary64: c^T fl(G) c, sum |c_i||c_j||G_ij|, |numeric| / scale */
    uint64_t residues[TR_IDENT_MAX_PRIMES];     /* modular: c^T G c mod each prime */
} tr_ident_quad;                                /* 296 bytes */

typedef struct tr_ident_plan_s tr_ident_plan;
typedef struct tr_ident_gram_s tr_ident_gram;
typedef struct tr_ident_kernel_s tr_ident_kernel;
typedef struct tr_ident_set_s tr_ident_set;

/* ------------------------------------------------------------------ shared helpers (CPU code, CUDA host and kernels) */
/* graded lexicographic comparison of two exponent rows: -1, 0, 1 */
TR_NUM_HD int tr_ident_cmp(const uint8_t *a, const uint8_t *b, int32_t n) {
    int da = 0, db = 0;
    for (int32_t i = 0; i < n; i++) { da += a[i]; db += b[i]; }
    if (da != db) return da < db ? -1 : 1;
    for (int32_t i = 0; i < n; i++) if (a[i] != b[i]) return a[i] < b[i] ? -1 : 1;
    return 0;
}
/* multiword words of two's complement integers modulo 2^(64 W), least significant first */
TR_NUM_HD void tr_ident_w_set(uint64_t *a, int32_t W, uint64_t lo) {
    a[0] = lo;
    for (int32_t i = 1; i < W; i++) a[i] = 0;
}
TR_NUM_HD void tr_ident_w_mul(uint64_t *a, int32_t W, uint64_t m) {          /* a *= m (mod 2^(64 W)) */
    unsigned __int128 carry = 0;
    for (int32_t i = 0; i < W; i++) {
        unsigned __int128 t = (unsigned __int128)a[i] * m + carry;
        a[i] = (uint64_t)t; carry = t >> 64;
    }
}
TR_NUM_HD void tr_ident_w_neg(uint64_t *a, int32_t W) {                      /* a = -a (mod 2^(64 W)) */
    uint64_t c = 1;
    for (int32_t i = 0; i < W; i++) { uint64_t t = ~a[i] + c; c = (c && t == 0) ? 1 : 0; a[i] = t; }
}
TR_NUM_HD void tr_ident_w_add(uint64_t *acc, const uint64_t *a, int32_t W) {
    uint64_t c = 0;
    for (int32_t i = 0; i < W; i++) {
        uint64_t s = acc[i] + a[i], c1 = s < a[i] ? 1 : 0, s2 = s + c, c2 = s2 < c ? 1 : 0;
        acc[i] = s2; c = c1 | c2;
    }
}
TR_NUM_HD int tr_ident_w_zero(const uint64_t *a, int32_t W) {
    for (int32_t i = 0; i < W; i++) if (a[i]) return 0;
    return 1;
}
/* the exact value of candidate j at a row whose support variables are present, in W words (the bound proven):
 * acc[0..W) two's complement */
TR_NUM_HD void tr_ident_cand_value(const tr_scope_view *v, const tr_ident_cand_view *c, int32_t j, uint64_t row,
                                   int32_t W, uint64_t *acc) {
    int64_t x[TR_IDENT_MAX_VARS];
    uint32_t sup = c->support[j];
    for (int32_t i = 0; i < c->nvars; i++) x[i] = ((sup >> i) & 1) ? tr_scope_value(v, c->var[i], row) : 0;
    const uint64_t t0 = c->offset[j], t1 = c->offset[j + 1];
    if (W == 2) {                                         /* native 128-bit words */
        unsigned __int128 s = 0;
        for (uint64_t t = t0; t < t1; t++) {
            const uint8_t *e = c->exponents + t * (uint64_t)c->nvars;
            unsigned __int128 p = (unsigned __int128)(__int128)c->coefficients[t];
            for (int32_t i = 0; i < c->nvars; i++)
                for (uint8_t k = 0; k < e[i]; k++) p *= (unsigned __int128)(__int128)x[i];
            s += p;
        }
        acc[0] = (uint64_t)s; acc[1] = (uint64_t)(s >> 64);
        return;
    }
    uint64_t term[TR_IDENT_MAX_WORDS];
    tr_ident_w_set(acc, W, 0);
    for (uint64_t t = t0; t < t1; t++) {
        const uint8_t *e = c->exponents + t * (uint64_t)c->nvars;
        int64_t co = c->coefficients[t];
        int neg = co < 0;
        tr_ident_w_set(term, W, tr_abs_u64(co));
        for (int32_t i = 0; i < c->nvars; i++) {
            if (!e[i]) continue;
            if (x[i] < 0 && (e[i] & 1)) neg = !neg;
            uint64_t m = tr_abs_u64(x[i]);
            for (uint8_t k = 0; k < e[i]; k++) tr_ident_w_mul(term, W, m);
        }
        if (neg) tr_ident_w_neg(term, W);
        tr_ident_w_add(acc, term, W);
    }
}
/* candidate j at a selected row: TR_CERT_ROW_EXCLUDED (a support variable is NULL), _HOLDS (the value is 0),
 * _VIOLATES, _OVERFLOW (words 0: not evaluated) */
TR_NUM_HD int tr_ident_cand_row(const tr_scope_view *v, const tr_ident_cand_view *c, int32_t j, uint64_t row) {
    uint32_t sup = c->support[j];
    for (int32_t i = 0; i < c->nvars; i++)
        if (((sup >> i) & 1) && !tr_scope_present(v, c->var[i], row)) return TR_CERT_ROW_EXCLUDED;
    int32_t W = c->words[j];
    if (W < 2 || W > TR_IDENT_MAX_WORDS) return TR_CERT_ROW_OVERFLOW;
    uint64_t acc[TR_IDENT_MAX_WORDS];
    tr_ident_cand_value(v, c, j, row, W, acc);
    return tr_ident_w_zero(acc, W) ? TR_CERT_ROW_HOLDS : TR_CERT_ROW_VIOLATES;
}

/* the Gram rows: a selected row where every variable and the weight are present; *w the weight (1 unweighted) */
TR_NUM_HD int tr_ident_gram_row(const tr_scope_view *v, const tr_ident_moment_view *m, uint64_t row, int64_t *w) {
    if (!tr_scope_is_selected(v, row)) return 0;
    for (int32_t i = 0; i < m->nvars; i++) if (!tr_scope_present(v, m->var[i], row)) return 0;
    if (m->weight >= 0) {
        if (!tr_scope_present(v, m->weight, row)) return 0;
        *w = tr_scope_value(v, m->weight, row);
    } else *w = 1;
    return 1;
}
/* the basis values at a Gram row, exact (the plan proved every moment, hence every basis value, fits 128 bits) */
TR_NUM_HD void tr_ident_basis_exact(const tr_scope_view *v, const tr_ident_moment_view *m, uint64_t row, __int128 *val) {
    __int128 x[TR_IDENT_MAX_VARS];
    for (int32_t i = 0; i < m->nvars; i++) x[i] = tr_scope_value(v, m->var[i], row);
    for (int32_t b = 0; b < m->nbasis; b++) {
        if (m->bparent[b] >= 0) { val[b] = val[m->bparent[b]] * x[m->bvar[b]]; continue; }
        __int128 p = 1;
        const uint8_t *e = m->basis + (uint64_t)b * (uint64_t)m->nvars;
        for (int32_t i = 0; i < m->nvars; i++) for (uint8_t k = 0; k < e[i]; k++) p *= x[i];
        val[b] = p;
    }
}
/* the basis values modulo a prime */
TR_NUM_HD void tr_ident_basis_mod(const tr_scope_view *v, const tr_ident_moment_view *m, const tr_gfp *f, uint64_t row,
                                  uint64_t *val) {
    uint64_t x[TR_IDENT_MAX_VARS];
    for (int32_t i = 0; i < m->nvars; i++) x[i] = tr_gfp_reduce_i64_hd(f, tr_scope_value(v, m->var[i], row));
    for (int32_t b = 0; b < m->nbasis; b++) {
        if (m->bparent[b] >= 0) { val[b] = tr_gfp_mul_hd(f, val[m->bparent[b]], x[m->bvar[b]]); continue; }
        uint64_t p = 1 % f->p;
        const uint8_t *e = m->basis + (uint64_t)b * (uint64_t)m->nvars;
        for (int32_t i = 0; i < m->nvars; i++) if (e[i]) p = tr_gfp_mul_hd(f, p, tr_gfp_pow_hd(f, x[i], e[i]));
        val[b] = p;
    }
}

/* the device copies of tr_mgpu_ident_gram: the selection, the columns read (and their validity), the plan arrays,
 * the accumulators and the counters; the states the adopted Gram holds */
static inline void tr_ident_gram_device_need(const tr_scope_view *v, const tr_ident_moment_view *m, tr_math_need *need) {
    for (int i = 0; i < TR_BUDGET_RESOURCES; i++) need->need[i] = 0;
    uint64_t rows = v->rows, bytes = tr_budget_bitmap_bytes(rows);
    for (int32_t i = 0; i <= m->nvars; i++) {
        int32_t c = i < m->nvars ? m->var[i] : m->weight;
        if (c < 0 || c >= v->ncols) continue;
        bytes = tr_budget_add(bytes, tr_budget_mul(rows, v->col[c].values ? 8 : 4));
        if (v->col[c].valid) bytes = tr_budget_add(bytes, rows);
    }
    uint64_t slots = m->repr == TR_IDENT_MODULAR ? (uint64_t)(m->nprimes > 0 ? m->nprimes : 0) : 1;
    bytes = tr_budget_add(bytes, tr_budget_mul((uint64_t)(m->nbasis > 0 ? m->nbasis : 0), (uint64_t)m->nvars + 8));
    bytes = tr_budget_add(bytes, tr_budget_mul((uint64_t)(m->nmoments > 0 ? m->nmoments : 0), 8));
    bytes = tr_budget_add(bytes, tr_budget_mul(slots, sizeof(tr_gfp)));
    bytes = tr_budget_add(bytes, tr_budget_add(tr_budget_mul(tr_budget_mul((uint64_t)m->nmoments, slots), 16), 64));
    need->need[TR_BUDGET_VARIABLES] = (uint64_t)m->nvars;
    need->need[TR_BUDGET_SUPPORT] = (uint64_t)m->nbasis;
    need->need[TR_BUDGET_TENSOR] = tr_budget_mul((uint64_t)m->nbasis, (uint64_t)m->nbasis);
    need->need[TR_BUDGET_STATES] = tr_budget_mul((uint64_t)m->nmoments, slots);
    need->need[TR_BUDGET_ITERATIONS] = m->repr == TR_IDENT_MODULAR ? slots : 0;
    need->need[TR_BUDGET_BYTES] = bytes;
}
/* the device copies of tr_mgpu_ident_verify */
static inline void tr_ident_verify_device_need(const tr_scope_view *v, const tr_ident_cand_view *c, tr_math_need *need) {
    for (int i = 0; i < TR_BUDGET_RESOURCES; i++) need->need[i] = 0;
    uint64_t rows = v->rows, bytes = tr_budget_bitmap_bytes(rows);
    for (int32_t i = 0; i < c->nvars; i++) {
        int32_t k = c->var[i];
        if (k < 0 || k >= v->ncols) continue;
        bytes = tr_budget_add(bytes, tr_budget_mul(rows, v->col[k].values ? 8 : 4));
        if (v->col[k].valid) bytes = tr_budget_add(bytes, rows);
    }
    uint64_t n = (uint64_t)(c->ncand > 0 ? c->ncand : 0);
    bytes = tr_budget_add(bytes, tr_budget_mul(c->nterms, (uint64_t)c->nvars + 8));
    bytes = tr_budget_add(bytes, tr_budget_mul(n, 8 + 4 + 4 + 64));
    need->need[TR_BUDGET_VARIABLES] = (uint64_t)c->nvars;
    need->need[TR_BUDGET_SUPPORT] = c->nterms;
    need->need[TR_BUDGET_OUTPUT] = n;
    need->need[TR_BUDGET_BYTES] = tr_budget_add(bytes, 8);
}

/* ------------------------------------------------------------------ plans */
tr_ident_plan *tr_ident_plan_make(const tr_scope *, const tr_ident_spec *, const tr_math_budget *, tr_math_refusal *,
                                  int32_t *error);
int32_t tr_ident_plan_get_info(const tr_ident_plan *, tr_ident_plan_info *);
int32_t tr_ident_plan_get_view(const tr_ident_plan *, tr_ident_moment_view *);
/* the canonical basis (nbasis x nvars exponents) and the moments (nmoments x nvars) */
int32_t tr_ident_plan_basis(const tr_ident_plan *, uint8_t *out, uint64_t capacity);
int32_t tr_ident_plan_moments(const tr_ident_plan *, uint8_t *out, uint64_t capacity);
/* the needs of tr_ident_gram_build (CPU), tr_mgpu_ident_gram (DEVICE) or adopting its states (ADOPT) */
int32_t tr_ident_plan_need(const tr_ident_plan *, const tr_scope *, int32_t target, tr_math_need *);
void tr_ident_plan_free(tr_ident_plan *);

/* ------------------------------------------------------------------ Grams */
tr_ident_gram *tr_ident_gram_build(const tr_scope *, const tr_ident_plan *, const tr_math_budget *, tr_math_refusal *,
                                   int32_t *error);
/* states computed elsewhere (tr_mgpu_ident_gram's readback), adopted after consistency checks */
tr_ident_gram *tr_ident_gram_from_states(const tr_scope *, const tr_ident_plan *, const tr_ident_gram_states *,
                                         const tr_math_budget *, tr_math_refusal *, int32_t *error);
/* FND-004: a bank of the scope in TR_BANK_JOINT mode over exactly the plan's variables whose basis holds every moment
 * (groups pooled); unweighted plans only (TR_MATH_UNSUPPORTED otherwise), EXACT (proven) plans only */
tr_ident_gram *tr_ident_gram_from_bank(const tr_scope *, const tr_ident_plan *, const tr_bank *, const tr_math_budget *,
                                       tr_math_refusal *, int32_t *error);
/* LIN-001: listwise moments of the scope with the plan's variables as features (no target), for a basis of degree
 * <= 1 (the Gram of [1, x]); unweighted, EXACT plans only */
tr_ident_gram *tr_ident_gram_from_lin(const tr_scope *, const tr_ident_plan *, const tr_lin_moments *,
                                      const tr_math_budget *, tr_math_refusal *, int32_t *error);
/* a + sign * b of Grams of one plan (merges of disjoint selections; sign -1: SIGNED) */
tr_ident_gram *tr_ident_gram_combine(const tr_ident_gram *a, const tr_ident_gram *b, int32_t sign,
                                     const tr_math_budget *, tr_math_refusal *, int32_t *error);
int32_t tr_ident_gram_get_info(const tr_ident_gram *, tr_ident_gram_info *);
/* the states (exact moments or residue sums) as tr_ident_gram_states with acc copied into out_acc */
int32_t tr_ident_gram_get_states(const tr_ident_gram *, tr_ident_gram_states *, int64_t *out_acc, uint64_t capacity);
/* the matrix: slot -1 the exact entries (B x B x 2 words lo, hi; EXACT only), slot p the residues mod prime p */
int32_t tr_ident_gram_matrix(const tr_ident_gram *, int32_t slot, uint64_t *out, uint64_t capacity);
int32_t tr_ident_gram_equal(const tr_ident_gram *, const tr_ident_gram *);
void tr_ident_gram_free(tr_ident_gram *);

/* ------------------------------------------------------------------ the nullspace */
/* EXACT Gram: the LIN-002 batch of the system [G | 0] (run its residues on the CPU or the device, then finish) */
tr_lin_batch *tr_ident_gram_system(const tr_ident_gram *, const tr_math_budget *, tr_math_refusal *, int32_t *error);
/* the kernel from the LIN-002 result of that system (verified exactly by LIN-002) */
tr_ident_kernel *tr_ident_kernel_from_lin(const tr_ident_gram *, const tr_lin_result *, const tr_math_budget *,
                                          tr_math_refusal *, int32_t *error);
/* CPU: EXACT through LIN-002 (rounds of primes up to the iterations budget), MODULAR from the recorded residues */
tr_ident_kernel *tr_ident_kernel_build(const tr_ident_gram *, const tr_math_budget *, tr_math_refusal *, int32_t *error);
int32_t tr_ident_kernel_get_info(const tr_ident_kernel *, tr_ident_kernel_info *);
/* the basis vectors (nullity x nbasis rationals; an unresolved vector exports zeros and is flagged) */
int32_t tr_ident_kernel_export(const tr_ident_kernel *, tr_lin_qbuf *out);
uint64_t tr_ident_kernel_limbs(const tr_ident_kernel *);
/* bit i: vector i was reconstructed (modular); every bit for an exact kernel */
uint64_t tr_ident_kernel_resolved(const tr_ident_kernel *);
void tr_ident_kernel_free(tr_ident_kernel *);

/* ------------------------------------------------------------------ identities */
/* The identities of a kernel (one per vector, in increasing leading monomial), labelled over `scope` (the scope the
 * identities are verified on: columns of the same names as the Gram's; it may hold more rows than the Gram's), and
 * the declared relations (nknown polynomials over the plan's variables) appended as DECLARED items. */
tr_ident_set *tr_ident_propose(const tr_scope *, const tr_ident_gram *, const tr_ident_kernel *,
                               const tr_ident_poly *known, int32_t nknown, const tr_math_budget *, tr_math_refusal *,
                               int32_t *error);
/* Given polynomials over the variables var[nvars] (scope columns) as candidates; gram (optional, same variables in
 * the same order) gives their sos verdicts. */
tr_ident_set *tr_ident_candidates(const tr_scope *, int32_t nvars, const int32_t *var, const tr_ident_poly *polys,
                                  int32_t npolys, const tr_ident_gram *gram, const tr_math_budget *,
                                  tr_math_refusal *, int32_t *error);
int32_t tr_ident_set_get_info(const tr_ident_set *, tr_ident_set_info *);
int32_t tr_ident_set_items(const tr_ident_set *, tr_ident_item *out, int32_t capacity);
int32_t tr_ident_set_get_view(const tr_ident_set *, tr_ident_cand_view *);
/* identity i's terms (exponents nterms x nvars, coefficients; canonical, decreasing graded order) */
int32_t tr_ident_set_poly(const tr_ident_set *, int32_t i, uint8_t *exponents, int64_t *coefficients,
                          uint64_t capacity, uint64_t *nterms);
void tr_ident_set_free(tr_ident_set *);
/* the quadratic form c^T G c of identity i of a set (the same variables as the Gram's plan) */
int32_t tr_ident_quadratic(const tr_ident_gram *, const tr_ident_set *, int32_t i, tr_ident_quad *out,
                           uint64_t *limbs, uint64_t capacity);

/* ------------------------------------------------------------------ verification and certificates */
int32_t tr_ident_verify_need(const tr_scope_view *, const tr_ident_cand_view *, int32_t target, tr_math_need *);
/* exhaustive evidence of every candidate over the selected rows of a view (out[ncand]) */
int32_t tr_ident_verify_view(const tr_scope_view *, const tr_ident_cand_view *, const tr_math_budget *,
                             tr_cert_evidence *out, tr_math_refusal *);
/* per identity: status[i] (TR_IDENT_*), certs[i] (an owned FND-007 snapshot, or NULL), codes[i] (0, or the native code
 * of the refused certificate: TR_MATH_DOMAIN_ERROR rejected, TR_MATH_OVERFLOW FND-007's 128-bit carrier, ...).
 * evidence: the verification of this set on this scope (CPU or device: backend TR_CERT_BY_*); NULL verifies on the
 * CPU. */
int32_t tr_ident_certify(const tr_ident_set *, const tr_scope *, const tr_cert_evidence *evidence, int32_t backend,
                         const tr_math_budget *, int32_t *status, tr_cert **certs, int32_t *codes,
                         tr_math_refusal *);
/* identity i's FND-007 POLYNOMIAL claim (policy SKIP, carrier I128) with pointers into the set */
int32_t tr_ident_claim(const tr_ident_set *, int32_t i, tr_cert_claim *out);
/* A hypothesis: identity i of the set whose numerical quadratic form (tr_ident_quadratic's relative value) is at most
 * tolerance, checked on a sample (FND-007 tr_cert_hypothesis); TR_MATH_DOMAIN_ERROR when not near zero (*quad says)
 * or when the sample refutes it (*evidence says). Never exact. */
tr_cert *tr_ident_hypothesis(const tr_scope *, const tr_ident_gram *, const tr_ident_set *, int32_t i, double tolerance,
                             const tr_cert_sample *sample, tr_ident_quad *quad, tr_cert_evidence *evidence,
                             const tr_math_budget *, tr_math_refusal *, int32_t *error);

/* ------------------------------------------------------------------ exact distinct support (libalg) */
/* Distinct points of the variables in mask (bits over var[0..nvars)) among the selected rows where they are all
 * present (and, with weight >= 0, the weight is present; points then count only when their total weight is nonzero).
 * Budget: bytes = 33 per counted row (49 with a weight) + 524288 radix counters. */
int32_t tr_ident_support(const tr_scope *, int32_t nvars, const int32_t *var, uint32_t mask, int32_t weight,
                         const tr_math_budget *, uint64_t *points, uint64_t *rows, tr_math_refusal *);

/* ------------------------------------------------------------------ the CUDA library (math_ident_gpu.cu) */
/* the Gram states (see tr_ident_gram_states; out->acc of nmoments x slots x 2 words) */
int tr_mgpu_ident_gram(const tr_scope_view *view, const tr_ident_moment_view *plan, const tr_math_budget *budget,
                       tr_ident_gram_states *out, tr_math_refusal *refusal);
/* the evidence of every candidate (equal to tr_ident_verify_view) */
int tr_mgpu_ident_verify(const tr_scope_view *view, const tr_ident_cand_view *cand, const tr_math_budget *budget,
                         tr_cert_evidence *out, tr_math_refusal *refusal);
#ifdef __cplusplus
}
#endif
#endif
