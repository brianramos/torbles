/* UNC-002: explicit approximations with certified error records (approximate transforms that carry their error).
 * SPDX-License-Identifier: Apache-2.0
 *
 * An approximation (tr_approx, immutable, individually owned) is an approximant of an exact transform together with
 * everything its error statement needs:
 *   norm         the declared norm of the bound: ABS (sup norm |f(x) - p(x)| over the declared interval), REL (that
 *                bound divided by a certified lower bound of |f|, only when |f| > 0 on the interval), L2 (the L2(dx)
 *                norm on [a, b], <= sqrt(b - a) ABS), FROBENIUS, SPECTRAL, MAX (entrywise) for matrices;
 *   assumptions  TR_APPROX_ASSUME_*: the domain box [a, b] (exact rationals), the derivative bound sup |f^(N+1)| the
 *                Taylor remainder uses, the Lipschitz constant of the exact function on its domain (compositions),
 *                the range inclusion of an inner function in the outer domain (verified), the independence of
 *                randomized probes (a fixed seed, separate streams), the absence of overflow in a floating
 *                evaluation (checked per evaluation: a non-finite value is refused);
 *   budget       an exact rational error budget (or none): a certified bound above it is REFUSED
 *                (TR_MATH_NOT_CONVERGED), never reported as certified;
 *   provenance   DETERMINISTIC, or RANDOMIZED with its seed and its failure probability stated exactly (the bound
 *                holds with probability >= 1 - failure over the probes);
 *   terms        the error budget spent, term by term (TR_APPROX_TERM_*): each a source (Taylor remainder,
 *                coefficient enclosures, Chebyshev economization, quantization, binary64 rounding, truncation of a
 *                product or composition, low-rank residual, probabilistic probe bound) with the id of the
 *                approximation it comes from and an exact rational magnitude; the certified error is their exact sum.
 *                Composition never drops a term: an operand's terms are carried (scaled by the propagation factor
 *                when a product or a Lipschitz map scales them), equal (kind, origin) terms merge additively.
 * Every rational is exact (FND-009 big rationals, budgeted by bytes); every binary64 output is an enclosure from the
 * FND-009 directed operations.
 *
 * Approximations (kinds):
 *   SERIES    p(t) = sum_j c_j B_j(t), t = (x - mid) / rad in [-1, 1] for x in the declared interval [a, b], basis
 *             B_j = T_j (CHEBYSHEV) or t^j (TAYLOR: monomials in t); coefficients exact dyadic rationals.
 *             tr_approx_series(f, a, b, n, basis): f in EXP, LOG, SIN, COS, SQRT, RECIP. The Taylor expansion of
 *             g(t) = f(mid + rad t) at t = 0 to degree N has coefficients g_k = f^(k)(mid) rad^k / k!, enclosed in
 *             exact rationals (e^m, sin m, cos m, log m by series with explicit remainders and argument reduction;
 *             sqrt m by integer square roots; the rest exactly), the representative chosen at the dyadic midpoint
 *             (COEFFICIENT term: sum of the half-widths, |t^k| <= 1); remainder M rad^(N+1) / (N+1)! with M >=
 *             sup |f^(N+1)| on [a, b] (REMAINDER term); CHEBYSHEV converts the Taylor polynomial of degree
 *             N = TR_APPROX_ECONOMIZE(n) exactly to the Chebyshev basis (t^k = 2^(1-k) sum binom(k, (k-j)/2) T_j)
 *             and truncates it to degree n (ECONOMIZATION term: sum |c_j|, j > n, |T_j| <= 1). Domains: LOG, SQRT
 *             need a > 0, RECIP 0 outside [a, b], EXP, SIN, COS |a|, |b| <= 1024 (else TR_MATH_DOMAIN_ERROR or
 *             TR_MATH_UNSUPPORTED). tr_approx_poly(c, n, a, b): an exact polynomial in x (no error).
 *   LOWRANK   a small integer matrix A (rows, cols <= 64) and a rank-k approximant:
 *             CROSS   k steps of fraction-free (Bareiss) elimination with complete pivoting (the largest |entry|,
 *                     ties by row then column): A ~ C U^-1 R with C = A[:, J], U = A[I, J], R = A[I, :]; the
 *                     residual is exactly the Schur complement S = M / d (M the integer minors, d = det U), so the
 *                     errors are exact: ||S||_F^2 = sum M^2 / d^2, ||S||_max, and the certified upper bounds
 *                     ||S||_F <= (isqrt(2^(2P) sum M^2) + 1) / (2^P |d|), ||S||_2 <= min(||S||_F,
 *                     sqrt(||S||_1 ||S||_inf)). Rank r < k with S = 0 is exact (DETERMINISTIC).
 *             SKETCH  Y = A Omega with Omega an n x k Rademacher (+-1) matrix from splitmix64(seed, stream 0); the
 *                     approximant is the exact orthogonal projection onto range(Y) (Y' the independent columns of Y
 *                     by exact elimination, X = (Y'^T Y')^-1 Y'^T A in exact rationals, A ~ Y' X). Its bound is
 *                     PROBABILISTIC: r Rademacher probes w_i (stream 1, independent of Omega); for E = A - Y' X and
 *                     X_i = ||E w_i||^2, E[X_i] = ||E||_F^2 and E[X_i^2] <= 3 ||E||_F^4, so by Paley-Zygmund
 *                     P(X_i > ||E||_F^2 / 4) >= 3/16 and, with probability >= 1 - (13/16)^r, ||E||_2 <= ||E||_F <=
 *                     2 max_i ||E w_i|| (failure = (13/16)^r exactly).
 *   FLOAT     floating evaluation: a SERIES as a monomial polynomial in x (exact substitution), coefficients rounded
 *             to binary64 (ROUNDING term: sum |a_j - fl(a_j)| B^j, B = max(|a|, |b|)) and Horner's rule with one fma
 *             per step in round-to-nearest (ROUNDING term: gamma_2n sum |fl(a_j)| B^j + (n + 1) 2^-1074 max(1, B)^n,
 *             gamma_k = k u / (1 - k u), u = 2^-53: the standard model with gradual underflow). Evaluated at a
 *             binary64 x of [a, b]; an input that is not exactly a double adds L |x - fl(x)| (L the Lipschitz bound).
 * Operations on SERIES over the same variable (equal intervals; compose: the inner range in the outer interval):
 *   add       c = c_F + c_G; terms F + G; L = L_F + L_G
 *   scale     c = k c_F; terms |k| F; L = |k| L_F
 *   mul       Chebyshev product T_i T_j = (T_{i+j} + T_{|i-j|}) / 2; |fg - P_F P_G| <= S_G e_F + (S_F + e_F) e_G
 *             with S = sum |c_j| >= sup |P|: terms S_G F + (S_F + e_F) G; L = (S_F + e_F) L_G + (S_G + e_G) L_F
 *   compose   P_F(P_G): requires [P_G range -+ e_G] within F's interval (else TR_MATH_DOMAIN_ERROR: the assumption
 *             fails); |f(g) - P_F(P_G)| <= e_F + L_F e_G: terms F + L_F G; L = L_F L_G
 *   quantize  coefficients rounded to the nearest multiple of 2^-bits (QUANTIZATION term: sum |c - c'|)
 * A product or composition of degree above TR_APPROX_MAX_DEGREE is truncated with its exact tail (TRUNCATION term).
 * The range of the exact function on its domain is enclosed by [c_0 - S' - e, c_0 + S' + e] (S' = sum_{j>=1} |c_j|).
 *
 * Labels (FND-007): an approximation is never exact. Its label is TR_APPROX_LABEL_CERTIFIED (a deterministic
 * certified bound) or TR_APPROX_LABEL_PROBABILISTIC (a bound that holds with a stated probability): values 5 and 6,
 * after FND-007's certificate kinds (LAW 1 .. HYPOTHESIS 4), so they are never confused with them.
 * tr_approx_authorize_exact answers like tr_cert_authorize_exact and always refuses (TR_MATH_DOMAIN_ERROR,
 * decision.reason TR_CERT_REFUSE_APPROXIMATE, or TR_CERT_REFUSE_STATISTICAL for a probabilistic one).
 * tr_approx_relation turns y ~ p(x) into an FND-007 POLYNOMIAL claim with a tolerance (> 0: an approximate claim)
 * that FND-007 verifies on every selected row; its snapshot certificate is refused by tr_cert_authorize_exact
 * (TR_CERT_REFUSE_APPROXIMATE).
 *
 * Evaluation over rows (tr_approx_eval_rows; the CUDA kernel tr_mgpu_approx_eval runs the same inline function per
 * row): x = logical / 10^scale enclosed in binary64 ([RD(x), RU(x)]); a row is in the domain when that enclosure lies
 * in [RU(a), RD(b)] (a row closer to an endpoint than an ulp is refused, never guessed); SERIES: t enclosed by
 * interval arithmetic and p(t) by interval Clenshaw (or Horner) with enclosures of the coefficients, then widened by
 * the certified error and clamped to the certified range of f; FLOAT: the binary64 value v and its bound. Each row
 * gets [lo, hi] containing f(x) (status OK), or a status NULL / DOMAIN / OVERFLOW. Group sums are exact: floor(lo 2^F)
 * and ceil(hi 2^F) added in 128-bit integers (atomics on the device: the totals do not depend on the order), so the
 * group's SUM(f(x)) lies in [sum_lo / 2^F, sum_hi / 2^F]; the host proves |sums| < 2^126 before running.
 *
 * Approximate predicates in UNC-001 (tr_approx_filter): f(x) op T over the groups of a UNC-001 state: a group whose
 * x-range lies in [a, b] is decided ALL when the binary64 UNC-001 program "p(t) op T -+ e" (the certain side) is ALL,
 * NONE when "p(t) op T +- e" (the possible side) is NONE; the rows of the other groups are decided by their row
 * enclosure, and a row the enclosure cannot decide (|f(x) - T| within the error, or out of the domain) is counted
 * UNDECIDED and set in its own bitmap, never guessed.
 *
 * Budgets (math_budgets.h, checked before allocation; TR_MATH_BUDGET with the tr_math_refusal): degree = the series
 * degree (and the Taylor degree N), support = coefficients, tensor = matrix entries, iterations = probes, output =
 * rows / groups evaluated, bytes = the limbs of every big number (FND-009: bytes / 8 limbs per number), the objects
 * and the scratch (CUDA: the device copies, tr_approx_eval_device_bytes). Representation limits (TR_MATH_UNSUPPORTED):
 * degree 64, 32 error terms, matrices 64 x 64, 64 probes.
 *
 * Errors: TR_MATH_INVALID, _MEMORY, _OVERFLOW, _BUDGET, _UNSUPPORTED, TR_MATH_DOMAIN_ERROR (outside the function's
 * domain, an assumption that does not hold, an approximation asked to authorize an exact rewrite),
 * TR_MATH_NOT_CONVERGED (the certified bound exceeds the error budget: refused), TR_MATH_STALE. */
#ifndef TORBLES_MATH_APPROX_H
#define TORBLES_MATH_APPROX_H
#include <stdint.h>
#include <string.h>
#include <math.h>
#include "math_num.h"
#include "math_budgets.h"
#include "math_scope.h"
#include "math_carriers.h"
#include "math_certificates.h"
#include "math_unc.h"
#ifdef __cplusplus
extern "C" {
#endif

#define TR_APPROX_ABI 1
#define TR_APPROX_MAX_DEGREE 64          /* series degree (representation limit) */
#define TR_APPROX_MAX_TERMS 32           /* error terms of one approximation */
#define TR_APPROX_MAX_DIM 64             /* rows / columns of a low-rank matrix */
#define TR_APPROX_MAX_PROBES 64
#define TR_APPROX_PREC 192               /* significant bits of the dyadic enclosures and coefficients */
#define TR_APPROX_MAX_TAYLOR 160         /* the Taylor degree behind a Chebyshev series */
#define TR_APPROX_ECONOMIZE(n) ((n) + 24 < 2 * (n) + 8 ? (n) + 24 : 2 * (n) + 8)   /* Taylor degree behind a Chebyshev series */
#define TR_APPROX_MAX_FBITS 64           /* fraction bits of the exact group sums */

enum { TR_APPROX_SERIES = 1, TR_APPROX_LOWRANK = 2, TR_APPROX_FLOAT = 3 };                    /* kinds */
enum { TR_APPROX_EXP = 1, TR_APPROX_LOG = 2, TR_APPROX_SIN = 3, TR_APPROX_COS = 4, TR_APPROX_SQRT = 5,
       TR_APPROX_RECIP = 6, TR_APPROX_POLY = 7, TR_APPROX_COMPOSED = 8, TR_APPROX_MATRIX = 9 };   /* the exact transform */
enum { TR_APPROX_TAYLOR = 1, TR_APPROX_CHEBYSHEV = 2, TR_APPROX_MONOMIAL_X = 3 };            /* bases */
enum { TR_APPROX_CROSS = 1, TR_APPROX_SKETCH = 2 };                                          /* low-rank methods */
enum { TR_APPROX_NORM_ABS = 1, TR_APPROX_NORM_REL = 2, TR_APPROX_NORM_L2 = 3, TR_APPROX_NORM_FROBENIUS = 4,
       TR_APPROX_NORM_SPECTRAL = 5, TR_APPROX_NORM_MAX = 6 };
enum { TR_APPROX_DETERMINISTIC = 1, TR_APPROX_RANDOMIZED = 2 };                              /* provenance */
enum { TR_APPROX_LABEL_CERTIFIED = 5, TR_APPROX_LABEL_PROBABILISTIC = 6 };                   /* after FND-007's kinds */
enum { TR_APPROX_TERM_REMAINDER = 1, TR_APPROX_TERM_COEFFICIENT = 2, TR_APPROX_TERM_ECONOMIZATION = 3,
       TR_APPROX_TERM_QUANTIZATION = 4, TR_APPROX_TERM_ROUNDING = 5, TR_APPROX_TERM_TRUNCATION = 6,
       TR_APPROX_TERM_RESIDUAL = 7, TR_APPROX_TERM_PROBABILISTIC = 8 };
enum { TR_APPROX_ASSUME_DOMAIN = 1, TR_APPROX_ASSUME_DERIVATIVE = 2, TR_APPROX_ASSUME_LIPSCHITZ = 4,
       TR_APPROX_ASSUME_RANGE = 8, TR_APPROX_ASSUME_INDEPENDENT = 16, TR_APPROX_ASSUME_NO_OVERFLOW = 32 };
enum { TR_APPROX_OP_ADD = 1, TR_APPROX_OP_MUL = 2, TR_APPROX_OP_SCALE = 3, TR_APPROX_OP_COMPOSE = 4,
       TR_APPROX_OP_QUANTIZE = 5, TR_APPROX_OP_FLOAT = 6 };
/* the exact rationals of an approximation (tr_approx_get_q): which, index */
enum { TR_APPROX_Q_ERROR = 1, TR_APPROX_Q_TERM = 2, TR_APPROX_Q_BUDGET = 3, TR_APPROX_Q_A = 4, TR_APPROX_Q_B = 5,
       TR_APPROX_Q_COEF = 6, TR_APPROX_Q_LIPSCHITZ = 7, TR_APPROX_Q_DERIVATIVE = 8, TR_APPROX_Q_RANGE_LO = 9,
       TR_APPROX_Q_RANGE_HI = 10, TR_APPROX_Q_RELATIVE = 11, TR_APPROX_Q_L2 = 12, TR_APPROX_Q_FROBENIUS2 = 13,
       TR_APPROX_Q_FROBENIUS = 14, TR_APPROX_Q_SPECTRAL = 15, TR_APPROX_Q_MAXENTRY = 16, TR_APPROX_Q_FAILURE = 17,
       TR_APPROX_Q_PROBE = 18, TR_APPROX_Q_ENTRY = 19 };
/* rows of an evaluation */
enum { TR_APPROX_ROW_OK = 0, TR_APPROX_ROW_NULL = 1, TR_APPROX_ROW_DOMAIN = 2, TR_APPROX_ROW_OVERFLOW = 3,
       TR_APPROX_ROW_UNSELECTED = 4 };
enum { TR_APPROX_EVAL_SERIES = 1, TR_APPROX_EVAL_FLOAT = 2 };
enum { TR_APPROX_GT = 1, TR_APPROX_GE = 2, TR_APPROX_LT = 3, TR_APPROX_LE = 4 };            /* filter predicates */
enum { TR_APPROX_NEED_CPU = 0, TR_APPROX_NEED_DEVICE = 1 };

/* ------------------------------------------------------------------ structs crossing the ctypes boundary */
typedef struct {
    int32_t kind, function, basis, method;   /* TR_APPROX_* */
    int32_t norm, label, provenance;         /* the declared norm; TR_APPROX_LABEL_*; TR_APPROX_DETERMINISTIC / RANDOMIZED */
    uint32_t assumptions;                    /* TR_APPROX_ASSUME_* */
    int32_t degree, nterms;                  /* series degree (FLOAT: the monomial degree); error terms */
    int32_t taylor, quantized;               /* the Taylor degree N behind it (0: none); quantization bits (0: none) */
    int32_t rows, cols, rank, probes;        /* LOWRANK: shape, the rank reached, probes */
    int32_t has_budget, has_relative;        /* a budget is set; a relative bound exists (|f| > 0 on the domain) */
    int32_t exact, operation;                /* 1: the error is exactly 0 (an exact representation); TR_APPROX_OP_* that made it */
    uint64_t seed, id;                       /* RANDOMIZED: the seed; the approximation's id (a digest of its construction) */
    uint64_t origins[2];                     /* the ids of the operands of an operation (0: none) */
    uint64_t bytes;                          /* bytes owned */
    double error_hi, budget_hi, failure_hi;  /* RU of the certified error, of the budget, of the failure probability */
    double range_lo, range_hi;               /* RD / RU of the certified range of the exact function (SERIES, FLOAT) */
} tr_approx_info;                            /* 160 bytes */

typedef struct {
    int32_t kind, scaled;                    /* TR_APPROX_TERM_*; 1: propagated with a factor (product, Lipschitz map) */
    uint64_t origin;                         /* the id of the approximation it was introduced by */
    double hi;                               /* RU of its exact magnitude (tr_approx_get_q(TERM, i)) */
} tr_approx_term;                            /* 24 bytes */

/* the plain view the row evaluator reads (pointers into the approximation, valid while it lives) */
typedef struct {
    int32_t mode, basis;                     /* TR_APPROX_EVAL_*; TR_APPROX_CHEBYSHEV / TR_APPROX_TAYLOR (SERIES) */
    int32_t ncoef, reserved;
    tr_math_interval mid, rad;               /* enclosures of the interval's midpoint and radius (rad > 0) */
    tr_math_interval dom;                    /* [RU(a), RD(b)] */
    tr_math_interval range;                  /* the certified range of the exact function (rows are clamped to it) */
    double err;                              /* RU(the certified error); FLOAT: everything but the input rounding */
    double lip;                              /* RU(the Lipschitz bound) for the input rounding (FLOAT) */
    const tr_math_interval *coef;            /* SERIES: coefficient enclosures; FLOAT: binary64 coefficients (points) */
} tr_approx_eval_view;                       /* 104 bytes */

typedef struct {
    int64_t lo[2], hi[2];                    /* sum of floor(lo 2^F) and of ceil(hi 2^F): 128-bit (lo word, hi word) */
    uint64_t rows, ok, nulls, refused;       /* selected rows; OK rows; NULL rows; DOMAIN / OVERFLOW rows */
} tr_approx_group;                           /* 64 bytes */

typedef struct {
    uint64_t rows, selected, ok, nulls, domain, overflow;
    int32_t fbits, groups;                   /* fraction bits of the sums; groups of the scope */
} tr_approx_eval_stats;                      /* 56 bytes */

typedef struct {
    uint64_t rows, selected;
    uint64_t groups_all, groups_none, groups_unknown, groups_empty, groups_outside;   /* groups by proof; outside: the
                                                       x-range leaves the interval, decided row by row */
    uint64_t rows_proven, rows_evaluated, rows_true, rows_false, rows_null, rows_undecided, rows_domain;
    double error_used;                       /* RU of the error the predicates were widened by */
    int32_t degree, reserved;                /* the degree of the UNC-001 program's polynomial */
} tr_approx_filter_stats;                    /* 128 bytes */

typedef struct tr_approx_s tr_approx;

/* ================================================================ inline helpers (host and CUDA device code) */
/* [RD(n / 10^s), RU(n / 10^s)] */
TR_NUM_HD tr_math_interval tr_approx_enclose_logical(int64_t n, int32_t scale) {
    int64_t d = 1;
    for (int32_t i = 0; i < scale; i++) d *= 10;
    tr_math_interval r; r.lo = -INFINITY; r.hi = INFINITY;
    tr_interval_enclose_q64(n, d, &r);
    return r;
}
TR_NUM_HD tr_math_interval tr_approx_iv(double lo, double hi) { tr_math_interval r; r.lo = lo; r.hi = hi; return r; }
TR_NUM_HD tr_math_interval tr_approx_meet(tr_math_interval a, tr_math_interval b) {
    tr_math_interval r; r.lo = a.lo > b.lo ? a.lo : b.lo; r.hi = a.hi < b.hi ? a.hi : b.hi; return r;
}
TR_NUM_HD int tr_approx_finite(tr_math_interval a) { return a.lo > -INFINITY && a.hi < INFINITY && a.lo <= a.hi; }
/* the enclosure of f(x) at one present value (logical, scale): TR_APPROX_ROW_OK with *out, or _DOMAIN / _OVERFLOW;
 * *value (FLOAT) the binary64 value of the floating evaluation, else the midpoint of *out */
TR_NUM_HD int tr_approx_eval_point(const tr_approx_eval_view *v, int64_t logical, int32_t scale, tr_math_interval *out,
                                   double *value) {
    tr_math_interval X = tr_approx_enclose_logical(logical, scale);
    if (!(X.lo >= v->dom.lo && X.hi <= v->dom.hi)) return TR_APPROX_ROW_DOMAIN;
    tr_math_interval F;
    if (v->mode == TR_APPROX_EVAL_FLOAT) {
        double x = X.lo, acc = v->ncoef > 0 ? v->coef[v->ncoef - 1].lo : 0.0;
        for (int32_t k = v->ncoef - 2; k >= 0; k--) acc = fma(acc, x, v->coef[k].lo);   /* one rounding per step */
        if (!(acc > -INFINITY && acc < INFINITY)) return TR_APPROX_ROW_OVERFLOW;
        double e = tr_add_ru(v->err, tr_mul_dir(v->lip, tr_add_ru(X.hi, -X.lo), 1));
        F.lo = tr_add_rd(acc, -e); F.hi = tr_add_ru(acc, e);
        *value = acc;
    } else {
        tr_math_interval T = tr_interval_div(tr_interval_sub(X, v->mid), v->rad);
        T = tr_approx_meet(T, tr_approx_iv(-1.0, 1.0));    /* x in [a, b]: the exact t lies in [-1, 1] */
        tr_math_interval P;
        if (v->basis == TR_APPROX_CHEBYSHEV) {             /* Clenshaw: b_k = c_k + 2 t b_{k+1} - b_{k+2} */
            tr_math_interval b1 = tr_approx_iv(0.0, 0.0), b2 = b1, T2 = tr_interval_mul(T, tr_approx_iv(2.0, 2.0));
            for (int32_t k = v->ncoef - 1; k >= 1; k--) {
                tr_math_interval b0 = tr_interval_sub(tr_interval_add(v->coef[k], tr_interval_mul(T2, b1)), b2);
                b2 = b1; b1 = b0;
            }
            P = v->ncoef > 0 ? tr_interval_sub(tr_interval_add(v->coef[0], tr_interval_mul(T, b1)), b2) : tr_approx_iv(0.0, 0.0);
        } else {                                           /* Horner in t */
            P = v->ncoef > 0 ? v->coef[v->ncoef - 1] : tr_approx_iv(0.0, 0.0);
            for (int32_t k = v->ncoef - 2; k >= 0; k--) P = tr_interval_add(tr_interval_mul(P, T), v->coef[k]);
        }
        if (!tr_approx_finite(P)) return TR_APPROX_ROW_OVERFLOW;
        F.lo = tr_add_rd(P.lo, -v->err); F.hi = tr_add_ru(P.hi, v->err);
        *value = 0.0;
    }
    F = tr_approx_meet(F, v->range);                       /* f(x) lies in the certified range of f */
    if (!tr_approx_finite(F)) return TR_APPROX_ROW_OVERFLOW;
    if (v->mode != TR_APPROX_EVAL_FLOAT) *value = tr_add_rd(tr_mul_dir(F.lo, 0.5, 0), tr_mul_dir(F.hi, 0.5, 0));
    *out = F;
    return TR_APPROX_ROW_OK;
}
/* an integral double |v| < 2^126 as an exact 128-bit integer (no double -> __int128 conversion needed): with
 * m = |v| = M 2^e (M < 2^53), h = floor(m / 2^64) and l = m - h 2^64 = (M mod 2^(64-e)) 2^e are exact doubles */
TR_NUM_HD __int128 tr_approx_i128_of(double v) {
    double m = fabs(v);
    double h = floor(ldexp(m, -64));
    double l = m - ldexp(h, 64);
    unsigned __int128 u = ((unsigned __int128)(uint64_t)h << 64) | (uint64_t)l;
    return v < 0 ? -(__int128)u : (__int128)u;
}
/* floor(lo 2^F), ceil(hi 2^F) of a row enclosure (|values| proven < 2^126 by the host) */
TR_NUM_HD void tr_approx_fixed(tr_math_interval F, int32_t fbits, __int128 *lo, __int128 *hi) {
    *lo = tr_approx_i128_of(floor(ldexp(F.lo, fbits)));
    *hi = tr_approx_i128_of(ceil(ldexp(F.hi, fbits)));
}
/* one selected row of a scope view: status; *F the enclosure (OK) */
TR_NUM_HD int tr_approx_eval_row(const tr_approx_eval_view *v, const tr_scope_view *sv, int32_t col, uint64_t row,
                                 tr_math_interval *F, double *value) {
    if (!tr_scope_present(sv, col, row)) return TR_APPROX_ROW_NULL;
    return tr_approx_eval_point(v, tr_scope_value(sv, col, row), sv->col[col].scale, F, value);
}
/* the bound the host checks before summing: R 2^F rows < 2^125 with R = max |range| (computed rounding up) */
static inline int tr_approx_sums_fit(const tr_approx_eval_view *v, int32_t fbits, uint64_t rows) {
    double R = fabs(v->range.lo) > fabs(v->range.hi) ? fabs(v->range.lo) : fabs(v->range.hi);
    if (!(R < INFINITY) || fbits < 0 || fbits > TR_APPROX_MAX_FBITS) return 0;
    double s = tr_mul_dir(tr_mul_dir(tr_add_ru(R, 1.0), ldexp(1.0, fbits), 1), tr_add_ru((double)rows, 1.0), 1);
    return s < ldexp(1.0, 125);
}
/* the device copies of tr_mgpu_approx_eval (bytes; UNKNOWN past 64 bits): the selection, ids, the column and its
 * validity, the coefficients, the per-row outputs (two doubles and a status byte) and the group accumulators */
static inline uint64_t tr_approx_eval_device_bytes(const tr_scope_view *sv, int32_t col, int32_t ncoef) {
    uint64_t n = sv->rows, b = tr_budget_bitmap_bytes(n);
    if (sv->groups) b = tr_budget_add(b, tr_budget_mul(n, 4));
    if (col >= 0 && col < sv->ncols) {
        b = tr_budget_add(b, tr_budget_mul(n, sv->col[col].values ? 8 : 4));
        if (sv->col[col].valid) b = tr_budget_add(b, n);
    }
    b = tr_budget_add(b, tr_budget_mul((uint64_t)(ncoef > 0 ? ncoef : 1), sizeof(tr_math_interval)));
    b = tr_budget_add(b, tr_budget_mul(n, 17));
    return tr_budget_add(b, tr_budget_mul(sv->ngroups, sizeof(tr_approx_group)));
}

/* ================================================================ exported (math_approx.c) */
/* Construction. budget_q: the error budget (NULL: none) in the declared norm (ABS, REL or L2 for a series); the result
 * is refused with TR_MATH_NOT_CONVERGED when its certified bound exceeds it. a, b: the interval (a < b). taylor: the
 * Taylor degree N behind a CHEBYSHEV series (0: TR_APPROX_ECONOMIZE(degree); degree <= N <= TR_APPROX_MAX_TAYLOR);
 * a TAYLOR series has N = degree (taylor 0 or degree). */
tr_approx *tr_approx_series(int32_t function, const tr_bigq *a, const tr_bigq *b, int32_t degree, int32_t basis,
                            int32_t taylor, int32_t norm, const tr_bigq *budget_q, const tr_math_budget *,
                            tr_math_refusal *, int32_t *error);
/* an exact polynomial sum_j coef[j] x^j (j <= degree) on [a, b] as a Chebyshev series (error 0) */
tr_approx *tr_approx_poly(const tr_bigq *const *coef, int32_t degree, const tr_bigq *a, const tr_bigq *b,
                          const tr_math_budget *, tr_math_refusal *, int32_t *error);
/* A (rows x cols, row-major int64), rank k; method CROSS or SKETCH (seed, probes 1..64); norm FROBENIUS, SPECTRAL
 * or MAX (CROSS only) */
tr_approx *tr_approx_lowrank(const int64_t *A, int32_t rows, int32_t cols, int32_t rank, int32_t method,
                             uint64_t seed, int32_t probes, int32_t norm, const tr_bigq *budget_q,
                             const tr_math_budget *, tr_math_refusal *, int32_t *error);
tr_approx *tr_approx_add(const tr_approx *, const tr_approx *, const tr_bigq *budget_q, const tr_math_budget *,
                         tr_math_refusal *, int32_t *error);
tr_approx *tr_approx_mul(const tr_approx *, const tr_approx *, const tr_bigq *budget_q, const tr_math_budget *,
                         tr_math_refusal *, int32_t *error);
tr_approx *tr_approx_scale(const tr_approx *, const tr_bigq *k, const tr_bigq *budget_q, const tr_math_budget *,
                           tr_math_refusal *, int32_t *error);
/* outer(inner(x)) */
tr_approx *tr_approx_compose(const tr_approx *outer, const tr_approx *inner, const tr_bigq *budget_q,
                             const tr_math_budget *, tr_math_refusal *, int32_t *error);
/* coefficients to multiples of 2^-bits (1..512) */
tr_approx *tr_approx_quantize(const tr_approx *, int32_t bits, const tr_bigq *budget_q, const tr_math_budget *,
                              tr_math_refusal *, int32_t *error);
/* the floating evaluation of a series (binary64 Horner in x with a certified a priori bound) */
tr_approx *tr_approx_float(const tr_approx *, const tr_bigq *budget_q, const tr_math_budget *, tr_math_refusal *,
                           int32_t *error);
void tr_approx_free(tr_approx *);

/* Reading. Rationals are borrowed (valid while the approximation lives); NULL when not applicable. */
int32_t tr_approx_get_info(const tr_approx *, tr_approx_info *);
int32_t tr_approx_get_term(const tr_approx *, int32_t i, tr_approx_term *);
const tr_bigq *tr_approx_get_q(const tr_approx *, int32_t which, int32_t index);
/* LOWRANK: the pivots (CROSS: rows then columns, rank each) or the sketch (SKETCH: cols x rank +-1 entries,
 * row-major; out_cols: the columns of Y kept) */
int32_t tr_approx_get_pivots(const tr_approx *, int32_t *rows_out, int32_t *cols_out, int32_t capacity);
int32_t tr_approx_get_sketch(const tr_approx *, int8_t *omega, int32_t capacity, int32_t *kept, int32_t kept_capacity);
/* the exact approximant at a rational x (SERIES, FLOAT: the exact polynomial; x outside [a, b]: DOMAIN_ERROR) or the
 * approximant's entry (i, j) (LOWRANK: x NULL, index = i * cols + j); an owned big rational */
tr_bigq *tr_approx_value(const tr_approx *, const tr_bigq *x, int32_t index, const tr_math_budget *,
                         tr_math_refusal *, int32_t *error);
/* FLOAT: the binary64 Horner values at binary64 points (out_value[n]; out_status: OK or DOMAIN / OVERFLOW) and the
 * certified bound of each (out_bound: RU) */
int32_t tr_approx_eval_doubles(const tr_approx *, const double *x, uint64_t n, double *out_value, double *out_bound,
                               int8_t *out_status);
/* the evaluation view (pointers into the approximation); TR_MATH_INVALID for LOWRANK */
int32_t tr_approx_get_eval_view(const tr_approx *, tr_approx_eval_view *);

/* Evaluation over the selected rows of a scope column (see the header comment): lo / hi / status per physical row
 * (each may be NULL), groups[scope groups] the exact sums (may be NULL). Budget: output = rows, bytes = the outputs. */
int32_t tr_approx_eval_rows(const tr_approx *, const tr_scope *, int32_t col, int32_t fbits, double *lo, double *hi,
                            int8_t *status, tr_approx_group *groups, tr_approx_eval_stats *, const tr_math_budget *,
                            tr_math_refusal *);
/* the needs of tr_approx_eval_rows (target 0) or tr_mgpu_approx_eval (target 1) */
int32_t tr_approx_eval_need(const tr_approx *, const tr_scope *, int32_t col, int32_t target, tr_math_need *);
/* the same evaluation over a scope view with an evaluation view (the CPU reference of the kernel) */
int32_t tr_approx_eval_view_rows(const tr_approx_eval_view *, const tr_scope_view *, int32_t col, int32_t fbits,
                                 double *lo, double *hi, int8_t *status, tr_approx_group *groups,
                                 tr_approx_eval_stats *);

/* f(x) op T over the groups of a UNC-001 state built from `scope` (x: the state's column `col`); bitmap (true rows)
 * and undecided (rows the approximation cannot decide), ceil(rows / 64) words each */
int32_t tr_approx_filter(const tr_unc_state *, const tr_scope *, int32_t col, const tr_approx *, int32_t op,
                         tr_num_rational threshold, uint64_t *bitmap, uint64_t *undecided, tr_approx_filter_stats *,
                         const tr_math_budget *, tr_math_refusal *);

/* FND-007: the refusal of an exact rewrite (always TR_MATH_DOMAIN_ERROR; decision filled) */
int32_t tr_approx_authorize_exact(const tr_approx *, uint64_t field, tr_cert_decision *out);
/* FND-007: the approximate relationship y ~ p(x) on the scope's columns xcol, ycol as a POLYNOMIAL claim with a
 * tolerance (the certified error, the quantization of p into x and the caller's data tolerance), verified on every
 * selected row: a SNAPSHOT certificate whose claim is approximate (or TR_MATH_DOMAIN_ERROR with the evidence) */
tr_cert *tr_approx_relation(const tr_approx *, const tr_scope *, int32_t xcol, int32_t ycol, const tr_bigq *data_tolerance,
                            int32_t bits, tr_cert_evidence *evidence, const tr_math_budget *, tr_math_refusal *,
                            int32_t *error);

/* ------------------------------------------------------------------ the CUDA library (math_approx_gpu.cu) */
/* the evaluation of tr_approx_eval_view_rows on a device: view (host pointers, copied), ev (the coefficients copied),
 * outputs as there (host arrays; NULL ones are not copied back). Budget: output = rows, bytes =
 * tr_approx_eval_device_bytes. 0 or a native error code (TR_MATH_DEVICE when the device fails). */
int tr_mgpu_approx_eval(const tr_scope_view *view, int32_t col, const tr_approx_eval_view *ev, int32_t fbits,
                        double *lo, double *hi, int8_t *status, tr_approx_group *groups, tr_approx_eval_stats *stats,
                        const tr_math_budget *budget, tr_math_refusal *refusal);
#ifdef __cplusplus
}
#endif
#endif
