/* GEOM-001: certified sparse ideal reduction.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Relations r_1..r_m are FND-007 certificates (math_certificates.h) of polynomial facts about scope columns:
 * POLYNOMIAL claims (DISC-002 identities), AFFINE / SUM / PRODUCT claims (DISC-001) read as col0 - scale col1 - bias,
 * col0 - col1 - col2, col0 - col1 col2; snapshot, hypothesis or derived certificates (a law or a schema fact holds
 * nothing to reduce by). Their columns together are the ideal's variables (at most TR_GEOM_MAX_VARS); they share one
 * NULL policy and their recorded sources are one table (or array scope) under one column layout: the same identity and
 * every recorded scope column the same stored column (else TR_MATH_DOMAIN_ERROR). The carrier is Z: coefficients are computed exactly as canonical int64
 * rationals (math_num.h; an operation past int64 is TR_MATH_OVERFLOW, never rounded) and certificates are integer.
 *
 * Orders (tr_geom_order): LEX, DEGLEX (graded lexicographic) or DEGREVLEX (graded reverse lexicographic) over a
 * variable sequence, most significant first (scope columns; nvars 0: increasing scope column order); variables a
 * reduction meets that the order does not list are appended, least significant, in increasing scope column order.
 *
 * The ideal (tr_geom_ideal_make): with complete = 1 the reduced Groebner basis of the relations by Buchberger's
 * algorithm (pairs by smallest lcm degree, Buchberger's product criterion, then interreduction), each element g_j
 * carried with its cofactors over the relations (g_j = sum_i A_ji r_i, exact); bounded by the budget: iterations =
 * S-pairs processed plus division steps, support = terms of any polynomial or cofactor, degree = total degree,
 * output = basis elements; past a limit the request
 * is refused (TR_MATH_BUDGET, the refusal naming the resource and the need reached: limit + 1 for iterations and basis
 * elements, the size or degree that would have been built for support and degree), never truncated. complete = 0: the
 * relations themselves (made monic) form the basis: multivariate division by them (the remainder then depends on the
 * relations' order and is canonical only for a Groebner basis).
 *
 * Reduction (tr_geom_reduce): f = sum_j q_j g_j + g with no term of g divisible by a basis leading monomial; the
 * cofactors over the relations h_i = sum_j q_j A_ji, so f - g = sum h_i r_i. Its certificate (tr_geom_certify): with D
 * the least common multiple of every denominator of f, g and the h_i, the POLYNOMIAL claim P = D (f - g) over the
 * claim columns (the ideal's variables and f's, increasing scope column order; the relations' NULL policy) derived
 * from the relation certificates with cofactors D h_i, checked by FND-007 (tr_cert_derive: P - sum D h_i r_i is zero
 * in Z[x]). Its kind is the weakest relation's and its domain the relations' own: a snapshot-certified relation makes
 * g equivalent to f only on that snapshot's rows (authorize_exact refuses any other scope); a hypothesis never
 * authorizes. Coefficients outside int64 or more than TR_CERT_MAX_TERMS terms cannot form an FND-007 claim
 * (TR_MATH_OVERFLOW / TR_MATH_UNSUPPORTED: no certificate).
 *
 * Checks (tr_geom_checks): batches of certificates P = sum_i q_i G_i (FND-007 claims), decided exactly by expanding
 * every product into a dense graded monomial index (tr_geom_rank) with signed 128-bit sums; the host proves
 * sum |P_t| + sum |q_a| |g_b| <= 2^127 - 1 per certificate (else it is undecided: overflow), so no partial sum wraps.
 * A certificate holds iff every slot is zero; the result names the number of nonzero slots and the smallest one (its
 * monomial: tr_geom_unrank). tr_mgpu_geom_check gives the same results bit for bit.
 *
 * Representations (tr_geom_cost, tr_geom_choose): per polynomial over a scope: terms, columns read, total degree,
 * multiplications of a naive evaluation (sum_t max(deg_t - 1, 0) + terms with |c| != 1), the bound bits of its integer
 * form (D p: bits of sum |c_t| prod M_v^e, M_v the column extremes) and, when measured, the actual bits (the largest
 * |value| on the rows: tr_geom_agree). tr_geom_choose picks by a declared priority (columns, then bits, then terms by
 * default; actual bits when measured, else the bound), ties to the earliest candidate.
 *
 * Agreement (tr_geom_pairs, tr_geom_agree_view, tr_mgpu_geom_agree): for pairs (f, g) the integer forms F = D_f f,
 * G = D_g g and P = D (f - g) are evaluated exactly on every selected row (DISC-002's W-word evaluation with proven
 * bounds; a row where a column of f or g is NULL is excluded): checked, excluded, disagreements (P != 0) and the
 * smallest disagreeing row, undecided rows, and the largest bit lengths of |F| and |G|.
 *
 * Derivatives (tr_geom_derivative): through = 0 the formal partial derivative of f as written (its columns
 * independent; no declarations may be passed). An identity that holds on the data (x2 = x1^2 on the rows) does not make x2 a function of x1: a
 * derivative that would use a relation (through = 1) is refused (TR_MATH_DOMAIN_ERROR, reason UNDECLARED) unless the
 * dependency is declared (tr_geom_dep: column v = phi(other columns)); then d/du of f is the total derivative
 * sum over the chain (df/du + sum_v df/dv dv/du, recursively through the declared phi), which f and every
 * representation equivalent to it on the dependency agree on; d/dv of a declared dependent column is refused
 * (DEPENDENT), a cyclic declaration too (CYCLE). tr_geom_dependency_check verifies a declaration on every row
 * (DISC-002's exhaustive evaluation of D v - D phi; the smallest violating row is the witness).
 *
 * Budgets (math_budgets.h), checked before allocation (or, inside Buchberger's iterations, as they are reached):
 * variables, degree, support, iterations, output (basis elements, pairs, certificates), states (dense slots of a
 * check), bytes. Errors: TR_MATH_INVALID, _MEMORY, _OVERFLOW, _BUDGET, _UNSUPPORTED, _DOMAIN_ERROR. Objects are
 * immutable and individually owned. */
#ifndef TORBLES_MATH_GEOM_H
#define TORBLES_MATH_GEOM_H
#include <stdint.h>
#include <string.h>
#include "math_num.h"
#include "math_budgets.h"
#include "math_scope.h"
#include "math_certificates.h"
#include "math_ident.h"
#ifdef __cplusplus
extern "C" {
#endif

#define TR_GEOM_ABI 1
#define TR_GEOM_MAX_VARS 8                /* an FND-007 claim reads at most 8 columns */
#define TR_GEOM_MAX_RELATIONS 16          /* FND-007 derivations: at most 16 premises */
#define TR_GEOM_MAX_BASIS 128             /* Groebner basis elements (representation limit) */
#define TR_GEOM_MAX_DEPS 8                /* declared dependencies */
#define TR_GEOM_MAX_PAIRS 256             /* pairs of one agreement batch */
#define TR_GEOM_MAX_CHECKS 256            /* certificates of one check batch */
#define TR_GEOM_MAX_SLOTS (UINT64_C(1) << 40)
#define TR_GEOM_SLICE_WORDS 64
#define TR_GEOM_MAX_SLICES 2048

enum { TR_GEOM_LEX = 1, TR_GEOM_DEGLEX = 2, TR_GEOM_DEGREVLEX = 3 };
enum { TR_GEOM_PRIORITY_COLUMNS = 1, TR_GEOM_PRIORITY_BITS = 2, TR_GEOM_PRIORITY_TERMS = 3 };
enum { TR_GEOM_DERIV_OK = 0, TR_GEOM_DERIV_UNDECLARED = 1, TR_GEOM_DERIV_DEPENDENT = 2, TR_GEOM_DERIV_CYCLE = 3 };
enum { TR_GEOM_PART_F = 0, TR_GEOM_PART_G = 1, TR_GEOM_PART_COFACTOR = 2 };     /* reduction parts (2 + i: h_i) */

/* ------------------------------------------------------------------ structs crossing the ctypes boundary */
typedef struct {
    int32_t kind, nvars;                 /* TR_GEOM_LEX / _DEGLEX / _DEGREVLEX; 0: increasing scope columns */
    int32_t col[TR_GEOM_MAX_VARS];       /* scope columns, most significant first */
} tr_geom_order;                         /* 40 bytes */

/* a polynomial over scope columns with rational coefficients num / den (den > 0; den NULL: integers) */
typedef struct {
    int32_t ncols, reserved;
    int32_t col[TR_GEOM_MAX_VARS];       /* scope columns of its exponent positions (distinct) */
    uint64_t nterms;
    const uint8_t *exponents;            /* nterms x ncols */
    const int64_t *num;
    const int64_t *den;
} tr_geom_poly;                          /* 72 bytes */

typedef struct {
    int32_t nrelations, nvars, kind, completed;     /* kind: the order; completed: a reduced Groebner basis */
    int32_t col[TR_GEOM_MAX_VARS];                  /* the variables, in the order's significance */
    int32_t nbasis, policy, weakest, reserved;      /* basis elements; the relations' NULL policy; weakest kind */
    uint64_t pairs, reductions, max_terms, max_degree, cofactor_terms, bytes;
} tr_geom_ideal_info;                               /* 112 bytes */

typedef struct {
    int32_t nvars, nrelations;
    int32_t col[TR_GEOM_MAX_VARS];       /* the claim columns (increasing scope column) */
    uint64_t f_terms, g_terms, cofactor_terms;
    uint64_t steps;                      /* division steps */
    int64_t denominator;                 /* D of the certificate (0 when it does not fit int64) */
    int32_t claimable, code;             /* 1: the certificate fits FND-007 (int64, <= 4096 terms); else its code */
} tr_geom_reduction_info;                /* 88 bytes */

typedef struct {
    uint64_t terms, columns, degree, multiplications;
    uint64_t bound_bits;                 /* bits of the integer form's evaluation bound on the scope */
    uint64_t actual_bits;                /* measured (tr_geom_agree): the largest |value| bits; 0 when not measured */
    int64_t denominator;                 /* D of the integer form */
    int32_t measured, reserved;
} tr_geom_cost;                          /* 64 bytes */

typedef struct {
    int32_t npairs, reserved;
    tr_ident_cand_view cand;             /* 3 npairs candidates: F_j, G_j, P_j */
} tr_geom_pair_view;                     /* 96 bytes */

typedef struct {
    uint64_t rows, selected, checked, excluded, disagreements, witness, undecided;
    uint32_t bits_f, bits_g;             /* the largest bit length of |F(x)|, |G(x)| over the checked rows */
} tr_geom_agreement;                     /* 64 bytes */

typedef struct {
    int32_t column, reserved;            /* the dependent scope column v = phi(...) */
    tr_geom_poly phi;
} tr_geom_dep;                           /* 80 bytes */

/* certificates to check, as the CPU loop and the kernel read them */
typedef struct {
    int32_t ncert, reserved;
    uint64_t npolys, nterms, nslots, nwork;
    const int32_t *nvars;                /* [ncert] */
    const int32_t *npremises;            /* [ncert] */
    const int32_t *maxdeg;               /* [ncert] the dense index's degree bound */
    const int32_t *overflow;             /* [ncert] 1: the magnitude bound failed (undecided: no slots, no work) */
    const uint64_t *poly0;               /* [ncert] first polynomial: P, then q_1, G_1, q_2, G_2, ... */
    const uint64_t *slot0;               /* [ncert + 1] first dense slot */
    const uint64_t *work0;               /* [ncert + 1] first work item: P's terms, then the products q_a g_b */
    const uint64_t *term0;               /* [npolys + 1] first term of each polynomial */
    const uint8_t *exponents;            /* nterms x TR_GEOM_MAX_VARS (zero padded) */
    const int64_t *coefficients;         /* nterms */
} tr_geom_check_view;                    /* 120 bytes */

typedef struct {
    uint64_t slots, work, nonzero, first;   /* dense slots; work items; nonzero slots; the smallest (UINT64_MAX) */
    int32_t holds, overflow;             /* holds: decided and zero; overflow: the bound failed (undecided) */
    uint64_t reserved;
} tr_geom_check_result;                  /* 48 bytes */

typedef struct tr_geom_ideal_s tr_geom_ideal;
typedef struct tr_geom_reduction_s tr_geom_reduction;
typedef struct tr_geom_pol_s tr_geom_pol;
typedef struct tr_geom_pairs_s tr_geom_pairs;
typedef struct tr_geom_checks_s tr_geom_checks;

/* ------------------------------------------------------------------ shared helpers (CPU code, CUDA host and kernels) */
/* C(n, k) for k <= TR_GEOM_MAX_VARS and values below the dense index's bound (every partial product is exact) */
TR_NUM_HD uint64_t tr_geom_binom(uint64_t n, uint32_t k) {
    if (k > n) return 0;
    uint64_t r = 1;
    for (uint32_t i = 1; i <= k; i++) r = r * (n - k + i) / i;
    return r;
}
/* the graded rank of an exponent vector over nv variables: the monomials of lower degree first, then the
 * combinatorial number system of the stars-and-bars positions of the composition */
TR_NUM_HD uint64_t tr_geom_rank(const uint8_t *e, int32_t nv) {
    uint64_t t = 0;
    for (int32_t i = 0; i < nv; i++) t += e[i];
    uint64_t r = t ? tr_geom_binom(t + (uint64_t)nv - 1, (uint32_t)nv) : 0, s = 0;
    for (int32_t i = 0; i + 1 < nv; i++) { s += e[i]; r += tr_geom_binom(s + (uint64_t)i, (uint32_t)i + 1); }
    return r;
}
/* the number of dense slots of nv variables and degree <= d: C(nv + d, nv) */
TR_NUM_HD uint64_t tr_geom_slots(int32_t nv, uint64_t d) { return tr_geom_binom(d + (uint64_t)nv, (uint32_t)nv); }
/* the bit length of |value| of a W-word two's complement value */
TR_NUM_HD uint32_t tr_geom_bits(const uint64_t *w, int32_t W) {
    uint64_t m[TR_IDENT_MAX_WORDS];
    for (int32_t i = 0; i < W; i++) m[i] = w[i];
    if (m[W - 1] >> 63) tr_ident_w_neg(m, W);
    for (int32_t i = W; i-- > 0;) {
        if (!m[i]) continue;
        uint32_t b = 0;
        uint64_t x = m[i];
        while (x) { b++; x >>= 1; }
        return (uint32_t)(64 * i) + b;
    }
    return 0;
}
/* work item w of certificate c: its slot (rank) and its signed product (P's terms +, products -) */
TR_NUM_HD void tr_geom_work(const tr_geom_check_view *v, int32_t c, uint64_t w, uint64_t *rank, __int128 *value) {
    const int32_t nv = v->nvars[c];
    const uint64_t p0 = v->poly0[c];
    uint64_t nP = v->term0[p0 + 1] - v->term0[p0];
    uint8_t e[TR_GEOM_MAX_VARS];
    if (w < nP) {
        uint64_t t = v->term0[p0] + w;
        *rank = tr_geom_rank(v->exponents + t * TR_GEOM_MAX_VARS, nv);
        *value = (__int128)v->coefficients[t];
        return;
    }
    w -= nP;
    for (int32_t i = 0; i < v->npremises[c]; i++) {
        uint64_t q = p0 + 1 + 2 * (uint64_t)i, g = q + 1;
        uint64_t nq = v->term0[q + 1] - v->term0[q], ng = v->term0[g + 1] - v->term0[g];
        if (w < nq * ng) {
            uint64_t a = v->term0[q] + w / ng, b = v->term0[g] + w % ng;
            for (int32_t k = 0; k < TR_GEOM_MAX_VARS; k++)
                e[k] = (uint8_t)(v->exponents[a * TR_GEOM_MAX_VARS + k] + v->exponents[b * TR_GEOM_MAX_VARS + k]);
            *rank = tr_geom_rank(e, nv);
            *value = -((__int128)v->coefficients[a] * v->coefficients[b]);
            return;
        }
        w -= nq * ng;
    }
    *rank = 0; *value = 0;
}

/* the device copies of tr_mgpu_geom_agree and tr_mgpu_geom_check */
static inline void tr_geom_agree_device_need(const tr_scope_view *v, const tr_geom_pair_view *p, tr_math_need *need) {
    tr_ident_verify_device_need(v, &p->cand, need);
    need->need[TR_BUDGET_OUTPUT] = (uint64_t)(p->npairs > 0 ? p->npairs : 0);
}
static inline void tr_geom_check_device_need(const tr_geom_check_view *v, tr_math_need *need) {
    for (int i = 0; i < TR_BUDGET_RESOURCES; i++) need->need[i] = 0;
    uint64_t n = (uint64_t)(v->ncert > 0 ? v->ncert : 0);
    uint64_t bytes = tr_budget_mul(n, 4 * 4 + 8 * 3 + 48);
    bytes = tr_budget_add(bytes, tr_budget_mul(tr_budget_add(v->npolys, 1), 8));
    bytes = tr_budget_add(bytes, tr_budget_mul(v->nterms, TR_GEOM_MAX_VARS + 8));
    bytes = tr_budget_add(bytes, tr_budget_mul(v->nslots, 16));
    need->need[TR_BUDGET_SUPPORT] = v->nterms;
    need->need[TR_BUDGET_OUTPUT] = n;
    need->need[TR_BUDGET_STATES] = v->nslots;
    need->need[TR_BUDGET_BYTES] = bytes;
}

/* ------------------------------------------------------------------ polynomials */
int32_t tr_geom_pol_get(const tr_geom_pol *, int32_t *ncols, int32_t *cols, uint64_t *nterms);
/* terms in decreasing order of the polynomial's order: exponents (nterms x ncols), numerators, denominators */
int32_t tr_geom_pol_export(const tr_geom_pol *, uint8_t *exponents, int64_t *num, int64_t *den, uint64_t capacity);
void tr_geom_pol_free(tr_geom_pol *);

/* ------------------------------------------------------------------ ideals and reductions */
tr_geom_ideal *tr_geom_ideal_make(const tr_cert *const *relations, int32_t n, const tr_geom_order *order,
                                  int32_t complete, const tr_math_budget *, tr_math_refusal *, int32_t *error);
int32_t tr_geom_ideal_get_info(const tr_geom_ideal *, tr_geom_ideal_info *);
/* basis element i (monic), or with cofactor = r + 1 its cofactor over relation r */
tr_geom_pol *tr_geom_ideal_element(const tr_geom_ideal *, int32_t i, int32_t cofactor, int32_t *error);
void tr_geom_ideal_free(tr_geom_ideal *);
tr_geom_reduction *tr_geom_reduce(const tr_geom_ideal *, const tr_geom_poly *f, const tr_math_budget *,
                                  tr_math_refusal *, int32_t *error);
int32_t tr_geom_reduction_get_info(const tr_geom_reduction *, tr_geom_reduction_info *);
/* TR_GEOM_PART_F, _G, or _COFACTOR + i (h_i) */
tr_geom_pol *tr_geom_reduction_part(const tr_geom_reduction *, int32_t part, int32_t *error);
/* the FND-007 parts of its certificate: the claim P = D (f - g) and the cofactor claims D h_i (pointers into the
 * reduction; TR_MATH_OVERFLOW / _UNSUPPORTED when it is not claimable) */
int32_t tr_geom_reduction_claims(const tr_geom_reduction *, tr_cert_claim *claim, tr_cert_claim *cofactors,
                                 int32_t capacity);
/* the derived certificate (tr_cert_derive over the ideal's relation certificates) */
tr_cert *tr_geom_certify(const tr_geom_reduction *, const tr_math_budget *, tr_math_refusal *, int32_t *error);
void tr_geom_reduction_free(tr_geom_reduction *);

/* ------------------------------------------------------------------ costs and choices */
int32_t tr_geom_cost_of(const tr_scope *, const tr_geom_poly *, tr_geom_cost *out);
/* the index of the cheapest of n costs by priority (TR_GEOM_PRIORITY_*: that criterion first, then the others in
 * the order columns, bits, terms), ties to the earliest */
int32_t tr_geom_choose(const tr_geom_cost *costs, int32_t n, int32_t priority, int32_t *chosen);

/* ------------------------------------------------------------------ agreement on rows */
tr_geom_pairs *tr_geom_pairs_make(const tr_scope *, const tr_geom_poly *f, const tr_geom_poly *g, int32_t n,
                                  const tr_math_budget *, tr_math_refusal *, int32_t *error);
int32_t tr_geom_pairs_get_view(const tr_geom_pairs *, tr_geom_pair_view *);
void tr_geom_pairs_free(tr_geom_pairs *);
int32_t tr_geom_agree_need(const tr_scope_view *, const tr_geom_pair_view *, int32_t target, tr_math_need *);
int32_t tr_geom_agree_view(const tr_scope_view *, const tr_geom_pair_view *, const tr_math_budget *,
                           tr_geom_agreement *out, tr_math_refusal *);

/* ------------------------------------------------------------------ batched certificate checks */
/* certificate c: the claim claims[c] and npremises[c] premise claims and cofactor claims (flattened in order) */
tr_geom_checks *tr_geom_checks_make(int32_t ncert, const tr_cert_claim *claims, const int32_t *npremises,
                                    const tr_cert_claim *premises, const tr_cert_claim *cofactors,
                                    const tr_math_budget *, tr_math_refusal *, int32_t *error);
int32_t tr_geom_checks_get_view(const tr_geom_checks *, tr_geom_check_view *);
/* the claim columns of certificate c (its variable order for tr_geom_unrank) */
int32_t tr_geom_checks_columns(const tr_geom_checks *, int32_t c, int32_t *cols, int32_t capacity);
/* the needs of running them: TR_IDENT_NEED_CPU (states and 16 bytes per slot of the largest certificate) or
 * TR_IDENT_NEED_DEVICE (tr_geom_check_device_need: every slot at once) */
int32_t tr_geom_checks_need(const tr_geom_checks *, int32_t target, tr_math_need *);
int32_t tr_geom_checks_run(const tr_geom_checks *, const tr_math_budget *, tr_geom_check_result *out,
                           tr_math_refusal *);
void tr_geom_checks_free(tr_geom_checks *);
/* the exponent vector of a graded rank (inverse of tr_geom_rank) */
int32_t tr_geom_unrank(uint64_t rank, int32_t nv, uint8_t *e);

/* ------------------------------------------------------------------ derivatives and dependencies */
tr_geom_pol *tr_geom_derivative(const tr_geom_poly *f, int32_t column, int32_t through, const tr_geom_dep *deps,
                                int32_t ndeps, const tr_math_budget *, int32_t *reason, tr_math_refusal *,
                                int32_t *error);
/* each declaration v = phi verified on every selected row of the scope: out[ndeps] (FND-007 evidence) */
int32_t tr_geom_dependency_check(const tr_scope *, const tr_geom_dep *deps, int32_t ndeps, const tr_math_budget *,
                                 tr_cert_evidence *out, tr_math_refusal *);

/* ------------------------------------------------------------------ the CUDA library (math_geom_gpu.cu) */
int tr_mgpu_geom_agree(const tr_scope_view *view, const tr_geom_pair_view *pairs, const tr_math_budget *budget,
                       tr_geom_agreement *out, tr_math_refusal *refusal);
int tr_mgpu_geom_check(const tr_geom_check_view *view, const tr_math_budget *budget, tr_geom_check_result *out,
                       tr_math_refusal *refusal);
#ifdef __cplusplus
}
#endif
#endif
