/* SPEC-001: fixed spectral basis states. The Fourier features of the selected rows of a scope (math_scope.h), per
 * dense group, for a declared finite set of (possibly multivariate) frequencies, aggregated once as certified complex
 * enclosures accumulated as exact scaled integers; phase actions (translations, phase offsets, derivatives in the
 * translation and the phase, the characteristic function) are answered from that fixed basis without rescanning.
 * SPDX-License-Identifier: Apache-2.0
 *
 * ---- The basis -----------------------------------------------------------------------------------------------------
 * A spec names d <= TR_FOURIER_MAX_DIMS scope columns x_1..x_d, an optional weight column w (every weight 1 without
 * one) and a unit c: TR_FOURIER_RADIAN (c = 1), TR_FOURIER_PI (c = pi) or TR_FOURIER_TURN (c = 2 pi). A frequency is
 * a vector omega of d exact canonical int64 rationals; the basis is a set of m distinct frequencies (their order as
 * given is the coefficient order). For group g the feature at omega is
 *     F_g(omega) = sum_r w_r exp(i c omega . x_r) = C + i S,  C = sum w cos(theta_r), S = sum w sin(theta_r),
 * theta_r = c sum_j omega_j x_rj, over the counted rows: selected rows where every x_j and w are present (SQL: NULLs
 * are skipped). x_rj is the exact logical value / 10^scale of the column (a decimal), w_r the weight's logical value
 * / 10^its scale. Per group the state keeps rows (selected), valid (counted) and the exact weight sum W (logical units,
 * 128 bits; value W / 10^wscale). An empty or all-NULL group has F = 0 exactly; an absent group has no rows.
 *
 * ---- Precision contract (per row, then exact sums) -------------------------------------------------------------------
 * The reduction t = theta (2/pi) = sum_j X_j K_j is formed exactly from the logical values X_j and constants
 * K_j = c omega_j (2/pi) / 10^scale_j in 64.192 fixed point (radians: the floor of the exact rational times a 512-bit
 * 2/pi, error < 2^-191; pi and turn units: K = 2 omega / 10^s or 4 omega / 10^s, exact whenever it is a dyadic rational
 * with at most 192 fractional bits, else floored, error < 2^-192). The terms are summed in 384-bit two's complement;
 * the sign is taken off (sin(-t) = -sin t), the quadrant comes from the exact integer part, the fraction is kept to
 * 128 bits. When every term is exact and the fraction is 0 the values are exact (sin, cos in {-1, 0, 1}, radius 0:
 * the Boolean and quarter-turn cases); otherwise sin and cos of f pi / 2 come from CALC-003's certified 120-bit
 * Taylor sums (math_calc.h: tr_calc_sincos_frac, radius <= 2^-112) widened by 3 ulps of 2^-120 for the reduction
 * (|t error| < (d + 1) 2^-128). For one dimension in radians the per-row enclosures are exactly CALC-003's
 * (tr_calc_sincos_reduce). Each row's enclosure times w_r is turned outward into integers floor(lo 2^s),
 * ceil(hi 2^s) (tr_calc_scaled) and summed exactly in 128 bits, s = min(120, 124 - bits(W)), W = selected rows x
 * max |w| (exact 128-bit), so every sum stays below 2^126 (CALC-003's scale). The exact bounds of C and S are
 * [lo, hi] / (2^s 10^wscale): they contain the exact value; their width is at most rows (|w|max 2^-108 + 2^(1-s)).
 *
 * ---- Phase actions from the fixed basis (no rescan) ---------------------------------------------------------------
 * A request (tau in R^d, phase phi, translation derivative order alpha in N^d, phase derivative order beta) maps the
 * coefficient of omega to
 *     A(omega) = i^(|alpha| + beta) (c omega)^alpha exp(i (c omega . tau + phi)) F(omega)    [/ W when normalized],
 * the features of the translated data x + tau rotated by phi and differentiated: d/dtau_j multiplies by i c omega_j,
 * d/dphi by i. normalize divides by the exact weight sum (the empirical characteristic function phi_X(omega) =
 * F(omega) / W; W = 0 is TR_MATH_DOMAIN_ERROR). tau and phi are doubles (their exact dyadic values, |tau_j|, |phi| <
 * 2^62, else TR_MATH_UNSUPPORTED); the rotation exp(i (c omega . tau + phi)) is reduced exactly as above from the exact
 * frequencies (constants K'_j = c omega_j (2/pi), no decimal scale) and the exact mantissas and exponents of tau and
 * phi (an exact rotation for the Boolean / quarter-turn cases), then everything is outward-rounded interval
 * arithmetic (math_carriers.h): every value returned is a complex rectangle that contains the exact result.
 * tr_fourier_derive returns a new state of net order (alpha, beta) whose exact bounds (tr_fourier_exact) are exact
 * transforms of the sums: the rotation by i^(|alpha| + beta) swaps and negates the bounds and omega^alpha multiplies
 * them exactly (big rationals); the factor c^|alpha| (pi^|alpha| or (2 pi)^|alpha|) is reported apart, never rounded
 * into the exact bounds.
 *
 * ---- Identity and missing support ----------------------------------------------------------------------------------
 * A state records its spec (columns, weight, unit), the exact frequency set and the scope's exact descriptor
 * (tr_scope_key_*). tr_fourier_require answers 0 when every requested frequency is in the basis and the rest of the
 * identity is equal, else TR_MATH_DOMAIN_ERROR: a new frequency requests a new basis state. A frequency outside the
 * basis is never treated as zero: tr_fourier_find, the actions with a selection and the Walsh inverse refuse it
 * (TR_MATH_DOMAIN_ERROR).
 *
 * ---- The Boolean case (SPEC-002) -------------------------------------------------------------------------------------
 * With k columns whose selected present values lie in {0, 1} (scale 0) and the 2^k frequencies S in {0, 1}^k in pi
 * units (or S / 2 in turns), F(S) = sum_r w_r (-1)^{|S and x_r|} = W(S), SPEC-002's Walsh coefficient of the weighted
 * histogram, computed exactly (radius 0, lo = hi). tr_fourier_walsh_inverse returns the histogram v(x) = 2^-k sum_S
 * W(S) (-1)^{|S and x|} exactly through SPEC-002's inverse transform (tr_spec_walsh_inverse) and its full-domain cost
 * (2^k states, k passes, k 2^(k-1) butterflies); it refuses (TR_MATH_DOMAIN_ERROR) a basis that lacks one of the 2^k
 * frequencies (missing support is not zero), data outside {0, 1}, a derivative state, or inexact coefficients.
 *
 * ---- CUDA (math_fourier_gpu.cu: tr_mgpu_fourier_sums) ----------------------------------------------------------------
 * The sums over the rows: a counts kernel (a slice of rows per work item: selected, counted rows and the exact weight
 * sum per group, local while the group stays the same) and a feature kernel (work item = (frequency, slice of rows):
 * the same inline per-row enclosures as the CPU, accumulated locally per group and added with exact two-word 128-bit
 * atomics, tr_atomic_add_wide), so the device sums equal the CPU's bit for bit whatever the thread order.
 *
 * ---- Budgets (math_budgets.h, before anything is allocated) -------------------------------------------------------
 *   build:   variables = d, support = m (frequencies), states = groups x m, tensor = selected rows x m (the per-row
 *            expansion), bytes = the plan, the sums and the state (CUDA: every device copy, tr_fourier_device_bytes)
 *   actions: output = requests x selected frequencies, tensor = output x (d + 1) (reduction terms), bytes = scratch
 *   derive:  degree = |alpha| + beta, states = groups x m, bytes = the new state
 *   walsh inverse: variables = k, tensor = 2^k, iterations = k, bytes = 40 x 2^k
 * Representation limits: d > TR_FOURIER_MAX_DIMS or m > TR_FOURIER_MAX_FREQS is TR_MATH_UNSUPPORTED, as is a
 * constant K >= 2^64 (a frequency too large for the reduction) and |tau|, |phi| >= 2^62.
 *
 * Errors: TR_MATH_INVALID (malformed input, duplicate frequencies, a zero denominator), _MEMORY, _OVERFLOW (a sum
 * outside its proven bound: never with a valid plan), _BUDGET (with the tr_math_refusal), _DOMAIN_ERROR, _UNSUPPORTED,
 * _STALE (a plan applied to another scope). Objects are immutable and individually owned. */
#ifndef TORBLES_MATH_FOURIER_H
#define TORBLES_MATH_FOURIER_H
#include <stdint.h>
#include <stddef.h>
#include "math_num.h"
#include "math_budgets.h"
#include "math_carriers.h"
#include "math_scope.h"
#include "math_calc.h"
#include "math_spec.h"
#ifdef __cplusplus
extern "C" {
#endif

#define TR_FOURIER_ABI 1
#define TR_FOURIER_MAX_DIMS 16
#define TR_FOURIER_MAX_FREQS (UINT64_C(1) << 22)
#define TR_FOURIER_MAX_ORDER 64                    /* |alpha| + beta of one request or state */
#define TR_FOURIER_WALSH_MAX_K 16
enum { TR_FOURIER_RADIAN = 0, TR_FOURIER_PI = 1, TR_FOURIER_TURN = 2 };

/* d columns, a weight column (or -1), the unit */
typedef struct {
    int32_t ndims, weight, unit, reserved;
    int32_t column[TR_FOURIER_MAX_DIMS];
} tr_fourier_spec;                                  /* 80 bytes */

/* the plan of a build as the CPU loop and the kernels read it: per (frequency f, dimension j) entry e = f * ndims + j
 * the limbs k[4 e .. 4 e + 3] of |K| (value (k3 2^192 + k2 2^128 + k1 2^64 + k0) / 2^192), its sign ksign[e] (0: K = 0)
 * and kexact[e] (1: K is exact) */
typedef struct {
    int32_t ndims, weight, scale, unit;            /* scale s: row values enter as floor / ceil(value 2^s) */
    uint64_t nfreq, ngroups;
    int32_t column[TR_FOURIER_MAX_DIMS];
    const uint64_t *k;
    const int32_t *ksign;
    const uint8_t *kexact;
} tr_fourier_plan_view;                             /* 120 bytes */

typedef struct {
    int32_t ndims, unit, scale, wscale;             /* exact bounds: [lo, hi] / (2^scale 10^wscale) */
    int32_t weight, exact, phase_order, unit_power; /* exact: every coefficient is a point; unit_power = |alpha| */
    uint64_t ngroups, nfreq, rows, selected;
    int32_t order[TR_FOURIER_MAX_DIMS];             /* the net translation derivative order alpha */
} tr_fourier_info;                                  /* 128 bytes */

typedef struct tr_fourier_plan_s tr_fourier_plan;
typedef struct tr_fourier_state_s tr_fourier_state;

/* ------------------------------------------------------------------ construction (libalg) */
/* the plan of spec over scope for nfreq frequencies omega[f * ndims + j] (canonical or not; denominators != 0) */
tr_fourier_plan *tr_fourier_plan_create(const tr_scope *, const tr_fourier_spec *, uint64_t nfreq,
                                        const tr_num_rational *omega, const tr_math_budget *, tr_math_refusal *,
                                        int32_t *error);
int32_t tr_fourier_plan_get_view(const tr_fourier_plan *, tr_fourier_plan_view *);
/* the exact sums of a plan over the scope (the CPU reference of tr_mgpu_fourier_sums): sums[4 (g nfreq + f) + c],
 * c = 0, 1: floor / ceil sums of w cos, c = 2, 3: of w sin; rows[g] selected rows, valid[g] counted rows, wsum[g] the
 * exact weight sum of the counted rows (logical units) */
int32_t tr_fourier_sums(const tr_fourier_plan_view *, const tr_scope_view *, tr_math_i128 *sums, uint64_t *rows,
                        uint64_t *valid, tr_math_i128 *wsum);
/* the state from sums (computed by tr_fourier_sums or the CUDA kernels): consistency-checked, budgeted */
tr_fourier_state *tr_fourier_from_sums(const tr_fourier_plan *, const tr_scope *, const tr_math_i128 *sums,
                                       const uint64_t *rows, const uint64_t *valid, const tr_math_i128 *wsum,
                                       const tr_math_budget *, tr_math_refusal *, int32_t *error);
/* plan + sums + from_sums on the CPU */
tr_fourier_state *tr_fourier_build(const tr_scope *, const tr_fourier_spec *, uint64_t nfreq,
                                   const tr_num_rational *omega, const tr_math_budget *, tr_math_refusal *,
                                   int32_t *error);
void tr_fourier_plan_free(tr_fourier_plan *);
/* bytes the CUDA build copies to and allocates on the device (saturating) */
uint64_t tr_fourier_device_need(const tr_fourier_plan_view *, const tr_scope_view *);

/* ------------------------------------------------------------------ reading */
int32_t tr_fourier_get_info(const tr_fourier_state *, tr_fourier_info *);
int32_t tr_fourier_group(const tr_fourier_state *, uint64_t g, uint64_t *rows, uint64_t *valid, tr_math_i128 *wsum);
/* frequency f (ndims exact rationals) */
int32_t tr_fourier_frequency(const tr_fourier_state *, uint64_t f, tr_num_rational *out);
/* the outward enclosure of coefficient f of group g (the state's derivative order applied) */
int32_t tr_fourier_coefficient(const tr_fourier_state *, uint64_t g, uint64_t f, tr_carrier_cinterval *out);
/* the raw exact scaled sums of a state built from data (order 0): lo/hi of C and of S, over 2^scale 10^wscale */
int32_t tr_fourier_bounds(const tr_fourier_state *, uint64_t g, uint64_t f, tr_math_i128 *out4);
/* the exact rational bounds of coefficient f of group g as new big rationals (owned by the caller): out[0..3] = lo
 * and hi of the real part, lo and hi of the imaginary part; the value is c^unit_power times them (c = pi or 2 pi) */
int32_t tr_fourier_exact(const tr_fourier_state *, uint64_t g, uint64_t f, tr_bigq **out, int32_t *unit_power,
                         const tr_math_budget *, tr_math_refusal *);
/* the position of each of n frequencies (omega[i * ndims + j]) in the basis; TR_MATH_DOMAIN_ERROR (missing spectral
 * support) at the first one absent (*missing = its request index; index then unspecified) */
int32_t tr_fourier_find(const tr_fourier_state *, uint64_t n, const tr_num_rational *omega, uint64_t *index,
                        uint64_t *missing);
/* whether the state answers spec over scope for the n frequencies: 0, or TR_MATH_DOMAIN_ERROR (a new basis state) */
int32_t tr_fourier_require(const tr_fourier_state *, const tr_scope *, const tr_fourier_spec *, uint64_t n,
                           const tr_num_rational *omega);

/* ------------------------------------------------------------------ actions from the fixed basis */
/* nreq requests on group g: tau[q * ndims + j] (NULL: 0), phase[q] (NULL: 0), order[q * ndims + j] (NULL: 0),
 * phase_order[q] (NULL: 0); nsel coefficients sel[] (NULL: all, nsel = nfreq); normalize: divide by the weight sum.
 * out[q * nsel + i] encloses the action on coefficient sel[i] */
int32_t tr_fourier_act(const tr_fourier_state *, uint64_t g, uint64_t nreq, const double *tau, const double *phase,
                       const int32_t *order, const int32_t *phase_order, uint64_t nsel, const uint64_t *sel,
                       int32_t normalize, tr_carrier_cinterval *out, const tr_math_budget *, tr_math_refusal *);
/* the derivative state of translation order alpha[ndims] and phase order beta (added to the state's) */
tr_fourier_state *tr_fourier_derive(const tr_fourier_state *, const int32_t *alpha, int32_t beta,
                                    const tr_math_budget *, tr_math_refusal *, int32_t *error);
/* the Boolean special case: k = ndims, out[2^k] = v(x) = 2^-k sum_S W(S) (-1)^{|S and x|} (logical weight units);
 * cost: SPEC-002's full-domain cost of the inverse */
int32_t tr_fourier_walsh_inverse(const tr_fourier_state *, uint64_t g, tr_spec_dyadic *out, tr_spec_cost *cost,
                                 const tr_math_budget *, tr_math_refusal *);
/* how many of the 2^k Walsh frequencies the basis holds (*k = ndims); 0, or TR_MATH_DOMAIN_ERROR when the basis is
 * not of the Boolean kind (unit, k > TR_FOURIER_WALSH_MAX_K) */
int32_t tr_fourier_walsh_support(const tr_fourier_state *, int32_t *k, uint64_t *present);
void tr_fourier_free(tr_fourier_state *);

/* ------------------------------------------------------------------ the CUDA library (math_fourier_gpu.cu)
 * tr_mgpu_fourier_sums: exactly tr_fourier_sums on the device (budget NULL = defaults; device copies budgeted) */
int tr_mgpu_fourier_sums(const tr_fourier_plan_view *, const tr_scope_view *, tr_math_i128 *sums, uint64_t *rows,
                         uint64_t *valid, tr_math_i128 *wsum, const tr_math_budget *, tr_math_refusal *);

/* ================================================================ shared helpers (CPU code, CUDA host and kernels) */
/* bytes the device build copies and allocates: selection, group ids, the used columns and their validity, the K
 * tables (37 bytes per entry), the sums (64 bytes per cell), rows, valid, weight sums, a flag */
static inline uint64_t tr_fourier_device_bytes(const tr_fourier_plan_view *p, const tr_scope_view *v) {
    uint64_t n = v->rows, b = tr_budget_bitmap_bytes(n);
    if (v->groups) b = tr_budget_add(b, tr_budget_mul(n, 4));
    uint32_t used = 0;
    for (int32_t i = 0; i < p->ndims; i++) used |= (uint32_t)1 << p->column[i];
    if (p->weight >= 0) used |= (uint32_t)1 << p->weight;
    for (int32_t c = 0; c < v->ncols; c++) {
        if (!((used >> c) & 1)) continue;
        b = tr_budget_add(b, tr_budget_mul(n, v->col[c].values ? 8 : 4));
        if (v->col[c].valid) b = tr_budget_add(b, n);
    }
    b = tr_budget_add(b, tr_budget_mul(tr_budget_mul(p->nfreq, (uint64_t)p->ndims), 37));
    b = tr_budget_add(b, tr_budget_mul(tr_budget_mul(p->ngroups, p->nfreq), 64));
    return tr_budget_add(b, tr_budget_add(tr_budget_mul(p->ngroups, 32), 8));
}
/* a plan view consistent with a scope view (host code of both libraries) */
static inline int tr_fourier_plan_view_valid(const tr_fourier_plan_view *pl, const tr_scope_view *v) {
    if (!pl || !v || pl->ndims < 1 || pl->ndims > TR_FOURIER_MAX_DIMS || pl->nfreq < 1 ||
        pl->nfreq > TR_FOURIER_MAX_FREQS || pl->ngroups != v->ngroups || pl->weight < -1 || pl->weight >= v->ncols ||
        pl->unit < TR_FOURIER_RADIAN || pl->unit > TR_FOURIER_TURN || pl->scale > TR_CALC_SCALE_MAX ||
        !pl->k || !pl->ksign || !pl->kexact || (v->rows && !v->selection) || (v->rows && v->ngroups > 1 && !v->groups))
        return 0;
    for (int32_t i = 0; i < pl->ndims; i++) if (pl->column[i] < 0 || pl->column[i] >= v->ncols) return 0;
    return 1;
}

/* acc (6 limbs, 384-bit two's complement, 192 fractional bits) += sign |X| K 2^e; e < 64. Bits shifted out below
 * 2^-192 (e < 0) set *lost. */
TR_NUM_HD void tr_fourier_term(uint64_t *acc, const uint64_t *k, int neg, uint64_t ax, int32_t e, int *lost) {
    uint64_t p[6];
    tr_calc_mulk(ax, k, p);
    p[5] = 0;
    if (e > 0) {
        for (int i = 5; i > 0; i--) p[i] = (p[i] << e) | (p[i - 1] >> (64 - e));
        p[0] <<= e;
    } else if (e < 0) {
        int32_t s = -e;
        if (s >= 384) {
            for (int i = 0; i < 6; i++) { if (p[i]) *lost = 1; p[i] = 0; }
        } else {
            int32_t q = s / 64, r = s % 64;
            for (int32_t i = 0; i < q; i++) if (p[i]) *lost = 1;
            if (r && (p[q] & (((uint64_t)1 << r) - 1))) *lost = 1;
            for (int32_t i = 0; i < 6; i++) {
                uint64_t lo = i + q < 6 ? p[i + q] : 0, hi = i + q + 1 < 6 ? p[i + q + 1] : 0;
                p[i] = r ? (lo >> r) | (hi << (64 - r)) : lo;
            }
        }
    }
    if (neg) { uint64_t c = 1; for (int i = 0; i < 6; i++) { uint64_t t = ~p[i] + c; c = c && t == 0; p[i] = t; } }
    unsigned __int128 carry = 0;
    for (int i = 0; i < 6; i++) { carry += (unsigned __int128)acc[i] + p[i]; acc[i] = (uint64_t)carry; carry >>= 64; }
}
/* sin and cos of (pi / 2) t for the accumulated t (inexact: some term or truncation was not exact): the sign taken
 * off, the quadrant from the integer part, the fraction to 128 bits; exact (radius 0) for an exact integer t */
TR_NUM_HD void tr_fourier_finish(uint64_t *acc, int inexact, tr_calc_fx *sn, tr_calc_fx *cs) {
    int neg = (int)(acc[5] >> 63);
    if (neg) { uint64_t c = 1; for (int i = 0; i < 6; i++) { uint64_t t = ~acc[i] + c; c = c && t == 0; acc[i] = t; } }
    uint32_t q = (uint32_t)(acc[3] & 3);
    unsigned __int128 f = ((unsigned __int128)acc[2] << 64) | acc[1];
    if (acc[0]) inexact = 1;
    __int128 S, Cc, sv, cv;
    uint64_t rad;
    if (f == 0 && !inexact) { S = 0; Cc = (__int128)1 << TR_CALC_F; rad = 0; }
    else {
        unsigned __int128 R = tr_calc_mulhi(f, TR_CALC_PIO2, NULL);    /* r = f pi / 2: 2 ulps below at most */
        tr_calc_sincos_frac(R, &S, &Cc, &rad);
        rad += 3;                                                      /* r (2 ulps) and t (< 18 2^-128 pi / 2) */
    }
    switch (q) {
    case 0: sv = S; cv = Cc; break;
    case 1: sv = Cc; cv = -S; break;
    case 2: sv = -S; cv = -Cc; break;
    default: sv = -Cc; cv = S; break;
    }
    if (neg) sv = -sv;                                                 /* sin(-t) = -sin(t), cos(-t) = cos(t) */
    sn->c = sv; sn->r = rad; cs->c = cv; cs->r = rad;
}
/* sin and cos of the angle of frequency f at row r of the scope (the per-row factor of the sums) */
TR_NUM_HD void tr_fourier_row_angle(const tr_fourier_plan_view *pl, const tr_scope_view *v, uint64_t f, uint64_t r,
                                    tr_calc_fx *sn, tr_calc_fx *cs) {
    uint64_t acc[6] = {0, 0, 0, 0, 0, 0};
    int lost = 0, inexact = 0, any = 0;
    const uint64_t e0 = f * (uint64_t)pl->ndims;
    for (int32_t j = 0; j < pl->ndims; j++) {
        int64_t X = tr_scope_value(v, pl->column[j], r);
        int32_t ks = pl->ksign[e0 + (uint64_t)j];
        if (X == 0 || ks == 0) continue;
        if (!pl->kexact[e0 + (uint64_t)j]) inexact = 1;
        tr_fourier_term(acc, pl->k + 4 * (e0 + (uint64_t)j), (X < 0) != (ks < 0), tr_abs_u64(X), 0, &lost);
        any = 1;
    }
    if (!any) { sn->c = 0; sn->r = 0; cs->c = (__int128)1 << TR_CALC_F; cs->r = 0; return; }
    tr_fourier_finish(acc, inexact || lost, sn, cs);
}
/* whether row r counts (selected, every column present) and its weight */
TR_NUM_HD int tr_fourier_row_ok(const tr_fourier_plan_view *pl, const tr_scope_view *v, uint64_t r) {
    if (!tr_scope_is_selected(v, r)) return 0;
    for (int32_t j = 0; j < pl->ndims; j++) if (!tr_scope_present(v, pl->column[j], r)) return 0;
    return pl->weight < 0 || tr_scope_present(v, pl->weight, r);
}
TR_NUM_HD int64_t tr_fourier_row_weight(const tr_fourier_plan_view *pl, const tr_scope_view *v, uint64_t r) {
    return pl->weight < 0 ? 1 : tr_scope_value(v, pl->weight, r);
}
/* the four outward integers of row r and frequency f: [L, H] of w cos(theta) 2^s and of w sin(theta) 2^s; 0 when a
 * bound does not fit int128 (excluded by the plan's proof) */
TR_NUM_HD int tr_fourier_row_terms(const tr_fourier_plan_view *pl, const tr_scope_view *v, uint64_t f, uint64_t r,
                                   __int128 *out4) {
    tr_calc_fx sn, cs;
    tr_fourier_row_angle(pl, v, f, r, &sn, &cs);
    int64_t w = tr_fourier_row_weight(pl, v, r);
    return tr_calc_scaled(w, cs, pl->scale, &out4[0], &out4[1]) && tr_calc_scaled(w, sn, pl->scale, &out4[2], &out4[3]);
}

#ifdef __cplusplus
}
#endif
#endif
