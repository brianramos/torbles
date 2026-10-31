/* CALC-003: non-polynomial finite-separation calculus; every contract is in math_calc.h.
 * SPDX-License-Identifier: Apache-2.0 */
#include "math_calc.h"
#include "math_handles.h"
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

TR_MATH_ABI(tr_calc_spec, TR_CALC_ABI);           /* the ctypes boundary of CALC-003 (math_handles.h) */
TR_MATH_ABI(tr_calc_plan_view, TR_CALC_ABI);
TR_MATH_ABI(tr_calc_info, TR_CALC_ABI);

_Static_assert(sizeof(tr_calc_spec) == 144 && sizeof(tr_calc_plan_view) == 680 && sizeof(tr_calc_info) == 176,
               "CALC-003 ABI");

#define PLAN_MAGIC 0x314e4c50u    /* "PLN1" */
#define STATE_MAGIC 0x31434c43u   /* "CLC1" */

/* floor(2/pi * 2^512) and floor(2^512 / ln 2), little-endian limbs (exact integer arithmetic, Machin's formula and
 * ln 2 = 2 atanh(1/3)); the 192-bit fractions of 2/pi and 1/ln 2 are their top limbs */
static const uint64_t TWO_OVER_PI_512[8] = {0xe88235f52ebb4484ULL, 0xfe1deb1cb129a73eULL, 0x06492eea09d1921cULL,
                                            0xb7246e3a424dd2e0ULL, 0xfe5163abdebbc561ULL, 0xdb6295993c439041ULL,
                                            0xfc2757d1f534ddc0ULL, 0xa2f9836e4e441529ULL};
static const uint64_t LOG2E_512[9] = {0x897f5e06a7be7366ULL, 0xd52173cc1895213fULL, 0x49b25eeb82d7c167ULL,
                                      0xbc3887eeaa2ed9acULL, 0x164a2cd9a342648fULL, 0xd6aef551bad2b4b1ULL,
                                      0x7d0ffda0d23a7d11ULL, 0x71547652b82fe177ULL, 0x0000000000000001ULL};

static void seterr(int32_t *e, int32_t v) { if (e) *e = v; }
static const tr_math_budget *budget_of(const tr_math_budget *b, tr_math_budget *store) {
    if (b) return b;
    tr_budget_default(store);
    return store;
}

/* ================================================================ directed conversions */
/* RD / RU of v * 2^e for a finite double v (ldexp is exact unless the result overflows or is subnormal) */
static double ldexp_dir(double v, int e, int up) {
    if (v == 0) return 0.0;
    double r = ldexp(v, e);
    if (r == INFINITY) return up ? r : DBL_MAX;
    if (r == -INFINITY) return up ? -DBL_MAX : r;
    if (fabs(r) >= DBL_MIN || (r != 0 && ldexp(r, -e) == v)) return r;   /* normal results are exact */
    if (r == 0) return (up == (v > 0)) ? (v > 0 ? 0x1p-1074 : -0x1p-1074) : 0.0;   /* underflow to zero */
    return tr_step(r, up);                                       /* a rounded subnormal: one step outward */
}
/* [RD((c - r) 2^(e - F)), RU((c + r) 2^(e - F))] */
static tr_math_interval fx_interval(tr_calc_fx x, int e) {
    tr_math_interval lo, hi, out;
    tr_carrier_enclose_i128(tr_i128_set(x.c - (__int128)x.r), &lo);
    tr_carrier_enclose_i128(tr_i128_set(x.c + (__int128)x.r), &hi);
    out.lo = ldexp_dir(lo.lo, e - TR_CALC_F, 0);
    out.hi = ldexp_dir(hi.hi, e - TR_CALC_F, 1);
    if (out.lo == 0) out.lo = 0.0;
    if (out.hi == 0) out.hi = 0.0;
    return out;
}

/* ================================================================ reduction constants */
/* |K| = floor(|p| C / (q 10^scale 2^320)) with C = TWO_OVER_PI_512 (PHASE) or LOG2E_512 (EXP), as 4 limbs; the error
 * of K / 2^192 is at most 2^-191 (math_calc.h); TR_MATH_UNSUPPORTED when K >= 2^64 */
static int32_t make_k(const tr_bigq *omega, int32_t scale, int32_t family, uint64_t k[4], int32_t *ksign,
                      const tr_math_budget *b, tr_math_refusal *r) {
    if (!omega || scale < 0 || scale > TR_SCOPE_MAX_SCALE) return TR_MATH_INVALID;
    int32_t s = tr_bigq_sign(omega);
    if (s < -1 || s > 1) return TR_MATH_INVALID;
    memset(k, 0, 4 * sizeof(uint64_t));
    *ksign = s;
    if (!s) return TR_MATH_OK;
    int32_t e = 0;
    tr_bigint *C = tr_bigint_from_limbs(1, family == TR_CALC_PHASE ? TWO_OVER_PI_512 : LOG2E_512,
                                        family == TR_CALC_PHASE ? 8 : 9, b, r, &e);
    tr_bigint *ap = C ? tr_bigint_abs(tr_bigq_numerator(omega), b, r, &e) : NULL;
    tr_bigint *num = ap ? tr_bigint_mul(ap, C, b, r, &e) : NULL;
    tr_bigint *ten = num ? tr_bigint_from_i64(10, b, r, &e) : NULL;
    tr_bigint *tp = ten ? tr_bigint_pow(ten, (uint64_t)scale, b, r, &e) : NULL;
    tr_bigint *den = tp ? tr_bigint_mul(tp, tr_bigq_denominator(omega), b, r, &e) : NULL;
    tr_bigint *q = NULL, *K = NULL;
    if (den) { int32_t rc = tr_bigint_divmod(num, den, TR_DIV_FLOOR, &q, NULL, b, r); if (rc) e = rc; }
    if (q) K = tr_bigint_shr(q, 320, b, r, &e);
    int32_t rc = e;
    if (K && !rc) {
        if (tr_bigint_bit_length(K) > 256) rc = TR_MATH_UNSUPPORTED;
        else { int32_t sg; uint64_t cnt; rc = tr_bigint_sign(K) ? tr_bigint_export(K, &sg, k, 4, &cnt) : TR_MATH_OK; }
    } else if (!rc) rc = TR_MATH_MEMORY;
    tr_bigint_free(C); tr_bigint_free(ap); tr_bigint_free(num); tr_bigint_free(ten); tr_bigint_free(tp);
    tr_bigint_free(den); tr_bigint_free(q); tr_bigint_free(K);
    if (rc == TR_MATH_OK && k[3] == 0 && k[2] == 0 && k[1] == 0 && k[0] == 0) *ksign = 0;   /* below 2^-192 */
    return rc;
}

/* ================================================================ plans */
struct tr_calc_plan_s {
    uint32_t magic, reserved;
    tr_calc_plan_view v;
    tr_calc_spec spec;
    int32_t wscale, ncols;
    uint64_t rows, selected;
    tr_bigq *omega[TR_CALC_FACTORS_MAX];
    unsigned char *key;
    uint64_t keybytes;
};
void tr_calc_plan_free(tr_calc_plan *p) {
    if (!p || p->magic != PLAN_MAGIC) return;
    p->magic = 0;
    for (int32_t i = 0; i < TR_CALC_FACTORS_MAX; i++) tr_bigq_free(p->omega[i]);
    free(p->key);
    free(p);
}
static uint64_t bits_u64(uint64_t x) { return x ? (uint64_t)(64 - __builtin_clzll(x)) : 0; }
static int32_t spec_check(const tr_calc_spec *s, int32_t ncols) {
    if (!s) return TR_MATH_INVALID;
    if (s->family != TR_CALC_PHASE && s->family != TR_CALC_EXP) return TR_MATH_UNSUPPORTED;
    if (s->nfactors > TR_CALC_FACTORS_MAX) return TR_MATH_UNSUPPORTED;
    if (s->nfactors < 1 || s->weight < -1 || s->weight >= ncols) return TR_MATH_INVALID;
    for (int32_t i = 0; i < s->nfactors; i++) {
        if (s->column[i] < 0 || s->column[i] >= ncols) return TR_MATH_INVALID;
        if (s->domega[i] != 0) return TR_MATH_UNSUPPORTED;      /* d/domega needs x-weighted sums: not retained */
    }
    return TR_MATH_OK;
}
static void need_of(const tr_calc_plan *p, int device, const tr_scope_view *view, tr_math_need *n) {
    memset(n, 0, sizeof *n);
    uint64_t G = p->v.ngroups, m = p->v.nsubsets, cells = tr_budget_mul(G, m);
    n->need[TR_BUDGET_VARIABLES] = (uint64_t)p->v.nfactors;
    n->need[TR_BUDGET_SUPPORT] = m;
    n->need[TR_BUDGET_STATES] = cells;
    n->need[TR_BUDGET_TENSOR] = tr_budget_mul(p->selected, m);
    /* the sums and counts, the state's bounds and intervals */
    uint64_t bytes = tr_budget_add(tr_budget_mul(cells, 32 + 32 + 16), tr_budget_mul(G, 32));
    bytes = tr_budget_add(bytes, sizeof(tr_calc_plan) + 512 + p->keybytes);
    if (device) bytes = tr_budget_add(bytes, tr_calc_device_bytes(&p->v, view));
    n->need[TR_BUDGET_BYTES] = bytes;
}

tr_calc_plan *tr_calc_plan_create(const tr_scope *scope, const tr_calc_spec *spec, const tr_bigq *const *omega,
                                  const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    tr_math_budget dflt;
    budget = budget_of(budget, &dflt);
    tr_scope_info info;
    int32_t rc = scope && omega ? tr_scope_get_info(scope, &info) : TR_MATH_INVALID;
    if (!rc) rc = spec_check(spec, info.ncols);
    if (!rc) for (int32_t i = 0; i < spec->nfactors; i++) if (!omega[i]) rc = TR_MATH_INVALID;
    if (rc) { seterr(error, rc); return NULL; }
    tr_calc_plan *p = (tr_calc_plan *)calloc(1, sizeof *p);
    if (!p) { seterr(error, TR_MATH_MEMORY); return NULL; }
    p->magic = PLAN_MAGIC;
    p->spec = *spec;
    p->ncols = info.ncols;
    p->rows = info.rows; p->selected = info.selected;
    tr_calc_plan_view *v = &p->v;
    v->family = spec->family; v->nfactors = spec->nfactors; v->weight = spec->weight;
    v->ngroups = info.ngroups;
    v->nsubsets = spec->family == TR_CALC_PHASE ? tr_budget_power2((uint32_t)spec->nfactors) : 1;
    for (int32_t i = 0; i < spec->nfactors; i++) v->column[i] = spec->column[i];
    /* the scope's descriptor (the state's identity) */
    p->keybytes = tr_scope_key_size(scope, TR_SCOPE_KEY_ALL);
    p->key = (unsigned char *)malloc(p->keybytes ? p->keybytes : 1);
    if (!p->key) rc = TR_MATH_MEMORY;
    else rc = tr_scope_key_write(scope, TR_SCOPE_KEY_ALL, p->key, p->keybytes);
    tr_math_need need;
    tr_scope_view sv;
    if (!rc) rc = tr_scope_get_view(scope, &sv);
    if (!rc) { need_of(p, 0, &sv, &need); rc = tr_budget_check(budget, &need, refusal); }
    /* the reduction constants */
    tr_scope_column_info ci[TR_CALC_FACTORS_MAX + 1];
    for (int32_t i = 0; i < spec->nfactors && !rc; i++) {
        rc = tr_scope_column_get(scope, spec->column[i], &ci[i]);
        if (!rc) rc = make_k(omega[i], ci[i].scale, spec->family, v->k[i], &v->ksign[i], budget, refusal);
        if (!rc) { int32_t e = 0; p->omega[i] = tr_bigq_copy(omega[i], budget, refusal, &e); if (!p->omega[i]) rc = e ? e : TR_MATH_MEMORY; }
    }
    /* the overflow proof: W >= sum |w| over the counted rows */
    uint64_t W = info.selected;
    if (!rc && spec->weight >= 0) {
        tr_scope_column_info wi;
        rc = tr_scope_column_get(scope, spec->weight, &wi);
        if (!rc) {
            p->wscale = wi.scale;
            uint64_t mw = wi.nvalid ? tr_abs_u64(wi.min) : 0, mx = wi.nvalid ? tr_abs_u64(wi.max) : 0;
            W = tr_budget_mul(info.selected, mw > mx ? mw : mx);
        }
    }
    if (!rc) {
        int64_t s;
        if (spec->family == TR_CALC_PHASE) s = 124 - (int64_t)bits_u64(W);   /* rows (|w| 2^(s+1) + 1) < 2^126 */
        else {
            /* nmax = floor(max_r sum_i X_ri K_i): the sum of each factor's maximum over its column's extremes */
            int32_t e = 0;
            tr_bigint *t = tr_bigint_from_i64(0, budget, refusal, &e);
            int any = 1;
            for (int32_t i = 0; i < spec->nfactors && t; i++) {
                if (!ci[i].nvalid) { any = 0; break; }
                if (!v->ksign[i]) continue;
                tr_bigint *K = tr_bigint_from_limbs(v->ksign[i], v->k[i], 4, budget, refusal, &e), *best = NULL;
                int64_t ext[2] = {ci[i].min, ci[i].max};
                for (int j = 0; j < 2 && K; j++) {
                    tr_bigint *x = tr_bigint_from_i64(ext[j], budget, refusal, &e), *y = x ? tr_bigint_mul(x, K, budget, refusal, &e) : NULL;
                    tr_bigint_free(x);
                    if (!y) break;
                    if (!best || tr_bigint_cmp(y, best) > 0) { tr_bigint_free(best); best = y; } else tr_bigint_free(y);
                }
                tr_bigint *t2 = best ? tr_bigint_add(t, best, budget, refusal, &e) : NULL;
                tr_bigint_free(t); tr_bigint_free(best); tr_bigint_free(K);
                t = t2;
            }
            tr_bigint *n = t ? tr_bigint_shr(t, 192, budget, refusal, &e) : NULL;   /* floor */
            int64_t nmax = 0;
            if (!n) rc = e ? e : TR_MATH_MEMORY;
            else if (any && (tr_bigint_to_i64(n, &nmax) || nmax >= TR_CALC_NMAX || nmax <= -TR_CALC_NMAX)) rc = TR_MATH_OVERFLOW;
            tr_bigint_free(t); tr_bigint_free(n);
            v->nmax = any ? nmax : 0;
            s = 124 - (int64_t)bits_u64(W) - v->nmax;          /* rows (|w| 2^(n + s + 1) + 1) < 2^126 */
        }
        if (s > TR_CALC_SCALE_MAX) s = TR_CALC_SCALE_MAX;
        v->scale = (int32_t)s;
    }
    if (rc) { tr_calc_plan_free(p); seterr(error, rc); return NULL; }
    seterr(error, TR_MATH_OK);
    return p;
}
int32_t tr_calc_plan_get_view(const tr_calc_plan *p, tr_calc_plan_view *out) {
    if (!p || p->magic != PLAN_MAGIC || !out) return TR_MATH_INVALID;
    *out = p->v;
    return TR_MATH_OK;
}
uint64_t tr_calc_device_need(const tr_calc_plan_view *p, const tr_scope_view *v) {
    return p && v ? tr_calc_device_bytes(p, v) : TR_BUDGET_UNKNOWN;
}

static int add_i128(tr_math_i128 *acc, __int128 x) {
    __int128 s;
    if (__builtin_add_overflow(tr_i128_get(*acc), x, &s)) return 0;
    *acc = tr_i128_set(s);
    return 1;
}
int32_t tr_calc_sums(const tr_calc_plan_view *pl, const tr_scope_view *v, tr_math_i128 *sums, uint64_t *rows,
                     uint64_t *valid) {
    if (!tr_calc_plan_view_valid(pl, v) || !sums || !rows || !valid) return TR_MATH_INVALID;
    uint64_t m = pl->nsubsets;
    memset(sums, 0, (size_t)(pl->ngroups * m * 2) * sizeof *sums);
    memset(rows, 0, (size_t)pl->ngroups * 8);
    memset(valid, 0, (size_t)pl->ngroups * 8);
    tr_calc_fx fac[2 * TR_CALC_FACTORS_MAX];
    for (uint64_t r = 0; r < v->rows; r++) {
        if (!tr_scope_is_selected(v, r)) continue;
        uint32_t g = tr_scope_group(v, r);
        if ((uint64_t)g >= pl->ngroups) return TR_MATH_INVALID;
        rows[g]++;
        if (!tr_calc_row_ok(pl, v, r)) continue;
        valid[g]++;
        tr_math_i128 *acc = sums + (uint64_t)g * m * 2;
        __int128 L, H;
        if (pl->family == TR_CALC_PHASE) {
            int64_t w = tr_calc_row_weight(pl, v, r);
            for (int32_t i = 0; i < pl->nfactors; i++)
                tr_calc_sincos_reduce(pl->k[i], pl->ksign[i], tr_scope_value(v, pl->column[i], r), 0, &fac[2 * i],
                                      &fac[2 * i + 1]);
            for (uint64_t S = 0; S < m; S++) {
                if (!tr_calc_scaled(w, tr_calc_subset(fac, pl->nfactors, S), pl->scale, &L, &H) ||
                    !add_i128(&acc[2 * S], L) || !add_i128(&acc[2 * S + 1], H)) return TR_MATH_OVERFLOW;
            }
        } else {
            if (!tr_calc_exp_row(pl, v, r, &L, &H) || !add_i128(&acc[0], L) || !add_i128(&acc[1], H))
                return TR_MATH_OVERFLOW;
        }
    }
    return TR_MATH_OK;
}

/* ================================================================ states */
struct tr_calc_state_s {
    uint32_t magic, reserved;
    tr_calc_info info;
    uint64_t *rows, *valid;
    tr_math_i128 *lo, *hi;          /* exact states: ngroups x ncoef */
    tr_math_interval *coef;         /* ngroups x ncoef */
    tr_calc_spec spec;
    tr_bigq *omega[TR_CALC_FACTORS_MAX];
    unsigned char *key;
    uint64_t keybytes;
};
void tr_calc_free(tr_calc_state *s) {
    if (!s || s->magic != STATE_MAGIC) return;
    s->magic = 0;
    free(s->rows); free(s->valid); free(s->lo); free(s->hi); free(s->coef); free(s->key);
    for (int32_t i = 0; i < TR_CALC_FACTORS_MAX; i++) tr_bigq_free(s->omega[i]);
    free(s);
}
static uint64_t state_bytes(uint64_t G, uint64_t m, int exact, uint64_t keybytes) {
    uint64_t b = tr_budget_add(sizeof(tr_calc_state), tr_budget_mul(G, 16));
    b = tr_budget_add(b, tr_budget_mul(tr_budget_mul(G, m), exact ? 48 : 16));
    return tr_budget_add(b, keybytes);
}
/* an empty state of the same identity with ncoef coefficients per group */
static tr_calc_state *state_like(const tr_calc_state *src_or_null, const tr_calc_plan *p, uint64_t G, uint64_t m,
                                 int exact, const tr_math_budget *b, tr_math_refusal *r, int32_t *error) {
    tr_math_need need;
    memset(&need, 0, sizeof need);
    need.need[TR_BUDGET_SUPPORT] = m;
    need.need[TR_BUDGET_STATES] = tr_budget_mul(G, m);
    uint64_t kb = src_or_null ? src_or_null->keybytes : p->keybytes;
    need.need[TR_BUDGET_BYTES] = state_bytes(G, m, exact, kb);
    int32_t rc = tr_budget_check(b, &need, r);
    if (rc) { seterr(error, rc); return NULL; }
    tr_calc_state *s = (tr_calc_state *)calloc(1, sizeof *s);
    if (!s) { seterr(error, TR_MATH_MEMORY); return NULL; }
    s->magic = STATE_MAGIC;
    s->rows = (uint64_t *)calloc((size_t)(G ? G : 1), 8);
    s->valid = (uint64_t *)calloc((size_t)(G ? G : 1), 8);
    s->coef = (tr_math_interval *)calloc((size_t)((G * m) != 0 ? G * m : 1), sizeof *s->coef);
    if (exact) {
        s->lo = (tr_math_i128 *)calloc((size_t)((G * m) != 0 ? G * m : 1), sizeof *s->lo);
        s->hi = (tr_math_i128 *)calloc((size_t)((G * m) != 0 ? G * m : 1), sizeof *s->hi);
    }
    s->keybytes = kb;
    s->key = (unsigned char *)malloc(kb ? kb : 1);
    int ok = s->rows && s->valid && s->coef && s->key && (!exact || (s->lo && s->hi));
    if (ok) memcpy(s->key, src_or_null ? src_or_null->key : p->key, (size_t)kb);
    s->spec = src_or_null ? src_or_null->spec : p->spec;
    for (int32_t i = 0; ok && i < s->spec.nfactors; i++) {
        int32_t e = 0;
        s->omega[i] = tr_bigq_copy(src_or_null ? src_or_null->omega[i] : p->omega[i], b, r, &e);
        if (!s->omega[i]) { ok = 0; rc = e; }
    }
    if (!ok) { tr_calc_free(s); seterr(error, rc ? rc : TR_MATH_MEMORY); return NULL; }
    if (src_or_null) {
        memcpy(s->rows, src_or_null->rows, (size_t)G * 8);
        memcpy(s->valid, src_or_null->valid, (size_t)G * 8);
    }
    s->info.ngroups = G; s->info.ncoef = m; s->info.exact = exact;
    return s;
}
/* the outward double interval of [lo, hi] / (2^scale 10^wscale) */
static int32_t exact_interval(tr_math_i128 lo, tr_math_i128 hi, int32_t scale, int32_t wscale, tr_math_interval *out) {
    tr_math_budget wide;
    tr_budget_default(&wide);
    int32_t e = 0, rc = TR_MATH_OK;
    tr_bigint *one = tr_bigint_from_i64(1, &wide, NULL, &e), *ten = tr_bigint_from_i64(10, &wide, NULL, &e);
    tr_bigint *tp = one && ten ? tr_bigint_pow(ten, (uint64_t)wscale, &wide, NULL, &e) : NULL;
    tr_bigint *p2 = tp ? tr_bigint_shl(one, (uint64_t)(scale >= 0 ? scale : -scale), &wide, NULL, &e) : NULL;
    tr_bigint *den = p2 ? (scale >= 0 ? tr_bigint_mul(tp, p2, &wide, NULL, &e) : tr_bigint_copy(tp, &wide, NULL, &e)) : NULL;
    tr_math_i128 v[2] = {lo, hi};
    double res[2] = {0, 0};
    for (int j = 0; j < 2 && den && !rc; j++) {
        tr_bigint *n = tr_bigint_from_i128(v[j], &wide, NULL, &e), *n2 = NULL;
        if (n && scale < 0) { n2 = tr_bigint_mul(n, p2, &wide, NULL, &e); tr_bigint_free(n); n = n2; }
        tr_bigq *q = n ? tr_bigq_from_bigints(n, den, &wide, NULL, &e) : NULL;
        tr_math_interval iv;
        if (!q) rc = e ? e : TR_MATH_MEMORY;
        else { rc = tr_bigq_enclose(q, &iv); res[j] = j ? iv.hi : iv.lo; }
        tr_bigq_free(q); tr_bigint_free(n);
    }
    if (!den && !rc) rc = e ? e : TR_MATH_MEMORY;
    tr_bigint_free(one); tr_bigint_free(ten); tr_bigint_free(tp); tr_bigint_free(p2); tr_bigint_free(den);
    if (!rc) { out->lo = res[0]; out->hi = res[1]; }
    return rc;
}

tr_calc_state *tr_calc_from_sums(const tr_calc_plan *p, const tr_scope *scope, const tr_math_i128 *sums,
                                 const uint64_t *rows, const uint64_t *valid, const tr_math_budget *budget,
                                 tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    tr_math_budget dflt;
    budget = budget_of(budget, &dflt);
    if (!p || p->magic != PLAN_MAGIC || !scope || !sums || !rows || !valid) { seterr(error, TR_MATH_INVALID); return NULL; }
    if (!tr_scope_key_equal(scope, TR_SCOPE_KEY_ALL, p->key, p->keybytes)) { seterr(error, TR_MATH_STALE); return NULL; }
    uint64_t G = p->v.ngroups, m = p->v.nsubsets, total = 0;
    for (uint64_t g = 0; g < G; g++) {
        if (valid[g] > rows[g]) { seterr(error, TR_MATH_INVALID); return NULL; }
        total += rows[g];
    }
    if (total != p->selected) { seterr(error, TR_MATH_INVALID); return NULL; }
    for (uint64_t i = 0; i < G * m; i++)
        if (tr_i128_get(sums[2 * i]) > tr_i128_get(sums[2 * i + 1])) { seterr(error, TR_MATH_INVALID); return NULL; }
    tr_calc_state *s = state_like(NULL, p, G, m, 1, budget, refusal, error);
    if (!s) return NULL;
    memcpy(s->rows, rows, (size_t)G * 8);
    memcpy(s->valid, valid, (size_t)G * 8);
    s->info.family = p->v.family; s->info.nparams = p->v.nfactors; s->info.nfactors = p->v.nfactors;
    s->info.scale = p->v.scale; s->info.wscale = p->wscale; s->info.weight = p->spec.weight;
    for (int32_t i = 0; i < p->v.nfactors; i++) s->info.param[i] = i;
    for (uint64_t i = 0; i < G * m; i++) {
        s->lo[i] = sums[2 * i]; s->hi[i] = sums[2 * i + 1];
        int32_t rc = exact_interval(s->lo[i], s->hi[i], s->info.scale, s->info.wscale, &s->coef[i]);
        if (rc) { tr_calc_free(s); seterr(error, rc); return NULL; }
    }
    seterr(error, TR_MATH_OK);
    return s;
}

tr_calc_state *tr_calc_build(const tr_scope *scope, const tr_calc_spec *spec, const tr_bigq *const *omega,
                             const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error) {
    tr_calc_plan *p = tr_calc_plan_create(scope, spec, omega, budget, refusal, error);
    if (!p) return NULL;
    uint64_t G = p->v.ngroups, m = p->v.nsubsets;
    tr_math_i128 *sums = (tr_math_i128 *)calloc((size_t)((G * m * 2) != 0 ? G * m * 2 : 1), sizeof *sums);
    uint64_t *rows = (uint64_t *)calloc((size_t)(G ? G : 1), 8), *valid = (uint64_t *)calloc((size_t)(G ? G : 1), 8);
    tr_scope_view v;
    int32_t rc = sums && rows && valid ? tr_scope_get_view(scope, &v) : TR_MATH_MEMORY;
    if (!rc) rc = tr_calc_sums(&p->v, &v, sums, rows, valid);
    tr_calc_state *s = NULL;
    if (!rc) s = tr_calc_from_sums(p, scope, sums, rows, valid, budget, refusal, error);
    else seterr(error, rc);
    free(sums); free(rows); free(valid);
    tr_calc_plan_free(p);
    return s;
}

int32_t tr_calc_get_info(const tr_calc_state *s, tr_calc_info *out) {
    if (!s || s->magic != STATE_MAGIC || !out) return TR_MATH_INVALID;
    *out = s->info;
    return TR_MATH_OK;
}
int32_t tr_calc_group(const tr_calc_state *s, uint64_t g, uint64_t *rows, uint64_t *valid) {
    if (!s || s->magic != STATE_MAGIC || g >= s->info.ngroups || !rows || !valid) return TR_MATH_INVALID;
    *rows = s->rows[g]; *valid = s->valid[g];
    return TR_MATH_OK;
}
int32_t tr_calc_coefficient(const tr_calc_state *s, uint64_t g, uint64_t S, tr_math_interval *out) {
    if (!s || s->magic != STATE_MAGIC || g >= s->info.ngroups || S >= s->info.ncoef || !out) return TR_MATH_INVALID;
    *out = s->coef[g * s->info.ncoef + S];
    return TR_MATH_OK;
}
int32_t tr_calc_coefficient_exact(const tr_calc_state *s, uint64_t g, uint64_t S, tr_math_i128 *lo, tr_math_i128 *hi) {
    if (!s || s->magic != STATE_MAGIC || g >= s->info.ngroups || S >= s->info.ncoef || !lo || !hi) return TR_MATH_INVALID;
    if (!s->info.exact) return TR_MATH_UNSUPPORTED;
    *lo = s->lo[g * s->info.ncoef + S]; *hi = s->hi[g * s->info.ncoef + S];
    return TR_MATH_OK;
}

static tr_math_interval iv_neg(tr_math_interval a) { return tr_interval_neg(a); }
tr_calc_state *tr_calc_derive(const tr_calc_state *s, int32_t param, int32_t order, const tr_math_budget *budget,
                              tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    tr_math_budget dflt;
    budget = budget_of(budget, &dflt);
    if (!s || s->magic != STATE_MAGIC || param < 0 || param >= s->info.nparams || order == INT32_MIN) {
        seterr(error, TR_MATH_INVALID); return NULL;
    }
    int32_t rc = tr_budget_check_one(budget, TR_BUDGET_DEGREE, (uint64_t)(order < 0 ? -(int64_t)order : order), refusal);
    if (rc) { seterr(error, rc); return NULL; }
    uint64_t G = s->info.ngroups, m = s->info.ncoef;
    tr_calc_state *o = state_like(s, NULL, G, m, s->info.exact, budget, refusal, error);
    if (!o) return NULL;
    o->info = s->info;
    int64_t net = (int64_t)s->info.order[param] + order;
    if (net > INT32_MAX || net < INT32_MIN + 1) { tr_calc_free(o); seterr(error, TR_MATH_OVERFLOW); return NULL; }
    o->info.order[param] = (int32_t)net;
    memcpy(o->coef, s->coef, (size_t)(G * m) * sizeof *o->coef);
    if (s->info.exact) { memcpy(o->lo, s->lo, (size_t)(G * m) * sizeof *o->lo); memcpy(o->hi, s->hi, (size_t)(G * m) * sizeof *o->hi); }
    if (s->info.family == TR_CALC_PHASE) {
        int32_t turns = ((order % 4) + 4) % 4;                   /* d^4 / dphi^4 is the identity */
        uint64_t bit = (uint64_t)1 << param;
        for (int32_t t = 0; t < turns; t++)
            for (uint64_t g = 0; g < G; g++)
                for (uint64_t S = 0; S < m; S++) {
                    if (S & bit) continue;
                    uint64_t a = g * m + S, b = g * m + (S | bit);   /* (C_a cos + C_b sin)' = C_b cos - C_a sin */
                    tr_math_interval ca = o->coef[a];
                    o->coef[a] = o->coef[b]; o->coef[b] = iv_neg(ca);
                    if (o->info.exact) {
                        tr_math_i128 la = o->lo[a], ha = o->hi[a];
                        o->lo[a] = o->lo[b]; o->hi[a] = o->hi[b];
                        o->lo[b] = tr_i128_set(-tr_i128_get(ha)); o->hi[b] = tr_i128_set(-tr_i128_get(la));
                    }
                }
    }
    seterr(error, TR_MATH_OK);
    return o;
}

/* per-parameter weights: PHASE (sin-weight, cos-weight) of a parameter, EXP its exp factor */
static int32_t phase_weights(double x, tr_math_interval *ws, tr_math_interval *wc) { return tr_calc_sincos(x, ws, wc); }
static int32_t integral_weights(int32_t family, double a, double b, tr_math_interval *ws, tr_math_interval *wc) {
    if (family == TR_CALC_PHASE) {                               /* int_a^b sin = cos a - cos b, int_a^b cos = sin b - sin a */
        tr_math_interval sa, ca, sb, cb;
        int32_t rc = tr_calc_sincos(a, &sa, &ca);
        if (!rc) rc = tr_calc_sincos(b, &sb, &cb);
        if (rc) return rc;
        *ws = tr_interval_sub(ca, cb); *wc = tr_interval_sub(sb, sa);
        return TR_MATH_OK;
    }
    tr_math_interval ea, eb;                                     /* int_a^b exp = exp b - exp a */
    int32_t rc = tr_calc_exp(a, &ea);
    if (!rc) rc = tr_calc_exp(b, &eb);
    if (rc) return rc;
    *ws = tr_interval_sub(eb, ea); *wc = *ws;
    return TR_MATH_OK;
}
/* contract parameter p of the coefficient vector c (2^k, PHASE) with weights (ws, wc): out has 2^(k-1) entries */
static void contract(const tr_math_interval *c, int32_t k, int32_t p, tr_math_interval ws, tr_math_interval wc,
                     tr_math_interval *out) {
    uint64_t half = (uint64_t)1 << (k - 1), lowmask = ((uint64_t)1 << p) - 1;
    for (uint64_t T = 0; T < half; T++) {
        uint64_t S = ((T & ~lowmask) << 1) | (T & lowmask);      /* insert a 0 bit at position p */
        out[T] = tr_interval_add(tr_interval_mul(c[S | ((uint64_t)1 << p)], ws), tr_interval_mul(c[S], wc));
    }
}
tr_calc_state *tr_calc_integrate(const tr_calc_state *s, int32_t param, double a, double b, const tr_math_budget *budget,
                                 tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    tr_math_budget dflt;
    budget = budget_of(budget, &dflt);
    if (!s || s->magic != STATE_MAGIC || param < 0 || param >= s->info.nparams || !isfinite(a) || !isfinite(b)) {
        seterr(error, TR_MATH_INVALID); return NULL;
    }
    tr_math_interval ws, wc;
    int32_t rc = integral_weights(s->info.family, a, b, &ws, &wc);
    if (rc) { seterr(error, rc); return NULL; }
    uint64_t G = s->info.ngroups, m = s->info.ncoef, nm = s->info.family == TR_CALC_PHASE ? m / 2 : 1;
    tr_calc_state *o = state_like(s, NULL, G, nm, 0, budget, refusal, error);
    if (!o) return NULL;
    o->info = s->info;
    o->info.exact = 0; o->info.ncoef = nm; o->info.nparams = s->info.nparams - 1;
    for (int32_t i = param; i + 1 < s->info.nparams; i++) {
        o->info.order[i] = s->info.order[i + 1]; o->info.param[i] = s->info.param[i + 1];
    }
    o->info.order[s->info.nparams - 1] = 0; o->info.param[s->info.nparams - 1] = 0;
    for (uint64_t g = 0; g < G; g++) {
        if (s->info.family == TR_CALC_PHASE) contract(s->coef + g * m, s->info.nparams, param, ws, wc, o->coef + g * nm);
        else o->coef[g] = tr_interval_mul(s->coef[g], ws);
    }
    seterr(error, TR_MATH_OK);
    return o;
}

/* the value of group g's coefficients with per-parameter weights (ws[i], wc[i]); scratch: 2^k intervals */
static tr_math_interval reduce_all(const tr_calc_state *s, uint64_t g, const tr_math_interval *ws,
                                   const tr_math_interval *wc, tr_math_interval *scratch) {
    uint64_t m = s->info.ncoef;
    int32_t k = s->info.nparams;
    if (s->info.family == TR_CALC_EXP) {
        tr_math_interval v = s->coef[g];
        for (int32_t i = 0; i < k; i++) v = tr_interval_mul(v, ws[i]);
        return v;
    }
    memcpy(scratch, s->coef + g * m, (size_t)m * sizeof *scratch);
    for (int32_t i = k - 1; i >= 0; i--) {                       /* contract the highest parameter first */
        uint64_t half = (uint64_t)1 << i;
        for (uint64_t S = 0; S < half; S++)
            scratch[S] = tr_interval_add(tr_interval_mul(scratch[S + half], ws[i]), tr_interval_mul(scratch[S], wc[i]));
    }
    return scratch[0];
}
static int32_t eval_check(const tr_calc_state *s, uint64_t g, uint64_t n, const tr_math_budget *budget,
                          tr_math_refusal *refusal) {
    tr_math_need need;
    memset(&need, 0, sizeof need);
    need.need[TR_BUDGET_OUTPUT] = n;
    need.need[TR_BUDGET_TENSOR] = tr_budget_mul(n, s->info.ncoef);
    need.need[TR_BUDGET_BYTES] = tr_budget_mul(s->info.ncoef, sizeof(tr_math_interval));
    (void)g;
    return tr_budget_check(budget, &need, refusal);
}
static int32_t weights_at(int32_t family, double x, tr_math_interval *ws, tr_math_interval *wc) {
    if (family == TR_CALC_PHASE) return phase_weights(x, ws, wc);
    int32_t rc = tr_calc_exp(x, ws);
    *wc = *ws;
    return rc;
}
int32_t tr_calc_evaluate(const tr_calc_state *s, uint64_t g, uint64_t n, const double *points, tr_math_interval *out,
                         const tr_math_budget *budget, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    tr_math_budget dflt;
    budget = budget_of(budget, &dflt);
    if (!s || s->magic != STATE_MAGIC || g >= s->info.ngroups || (n && (!out || (s->info.nparams && !points))))
        return TR_MATH_INVALID;
    int32_t rc = eval_check(s, g, n, budget, refusal);
    if (rc) return rc;
    tr_math_interval *scratch = (tr_math_interval *)malloc((size_t)s->info.ncoef * sizeof *scratch);
    if (!scratch) return TR_MATH_MEMORY;
    tr_math_interval ws[TR_CALC_FACTORS_MAX], wc[TR_CALC_FACTORS_MAX];
    int32_t k = s->info.nparams;
    for (uint64_t j = 0; j < n && !rc; j++) {
        for (int32_t i = 0; i < k && !rc; i++) rc = weights_at(s->info.family, points[j * (uint64_t)k + i], &ws[i], &wc[i]);
        if (!rc) out[j] = reduce_all(s, g, ws, wc, scratch);
    }
    free(scratch);
    return rc;
}
int32_t tr_calc_definite(const tr_calc_state *s, uint64_t g, int32_t param, uint64_t n, const double *a, const double *b,
                         const double *rest, tr_math_interval *out, const tr_math_budget *budget,
                         tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    tr_math_budget dflt;
    budget = budget_of(budget, &dflt);
    if (!s || s->magic != STATE_MAGIC || g >= s->info.ngroups || param < 0 || param >= s->info.nparams ||
        (n && (!a || !b || !out || (s->info.nparams > 1 && !rest)))) return TR_MATH_INVALID;
    int32_t rc = eval_check(s, g, n, budget, refusal);
    if (rc) return rc;
    tr_math_interval *scratch = (tr_math_interval *)malloc((size_t)s->info.ncoef * sizeof *scratch);
    if (!scratch) return TR_MATH_MEMORY;
    tr_math_interval ws[TR_CALC_FACTORS_MAX], wc[TR_CALC_FACTORS_MAX];
    int32_t k = s->info.nparams;
    for (uint64_t j = 0; j < n && !rc; j++) {
        if (!isfinite(a[j]) || !isfinite(b[j])) { rc = TR_MATH_INVALID; break; }
        rc = integral_weights(s->info.family, a[j], b[j], &ws[param], &wc[param]);
        for (int32_t i = 0, o = 0; i < k && !rc; i++) {
            if (i == param) continue;
            rc = weights_at(s->info.family, rest[j * (uint64_t)(k - 1) + (uint64_t)o++], &ws[i], &wc[i]);
        }
        if (!rc) out[j] = reduce_all(s, g, ws, wc, scratch);
    }
    free(scratch);
    return rc;
}
int32_t tr_calc_require(const tr_calc_state *s, const tr_scope *scope, const tr_calc_spec *spec,
                        const tr_bigq *const *omega) {
    if (!s || s->magic != STATE_MAGIC || !scope || !spec || !omega) return TR_MATH_INVALID;
    tr_scope_info info;
    int32_t rc = tr_scope_get_info(scope, &info);
    if (!rc) rc = spec_check(spec, info.ncols);
    if (rc) return rc;                                           /* UNSUPPORTED: outside the retained basis */
    if (spec->family != s->spec.family || spec->nfactors != s->spec.nfactors || spec->weight != s->spec.weight)
        return TR_MATH_DOMAIN_ERROR;
    for (int32_t i = 0; i < spec->nfactors; i++) {
        if (!omega[i]) return TR_MATH_INVALID;
        if (spec->column[i] != s->spec.column[i]) return TR_MATH_DOMAIN_ERROR;
        int32_t c = tr_bigq_cmp(omega[i], s->omega[i]);
        if (c < -1 || c > 1) return c;
        if (c) return TR_MATH_DOMAIN_ERROR;                       /* another frequency / rate: a new construction */
    }
    if (!tr_scope_key_equal(scope, TR_SCOPE_KEY_ALL, s->key, s->keybytes)) return TR_MATH_DOMAIN_ERROR;
    return TR_MATH_OK;
}

/* ================================================================ certified elementary functions of doubles */
static void decompose(double x, int64_t *M, int32_t *E) {      /* x = M 2^E, |M| < 2^53 */
    int ex;
    double fr = frexp(x, &ex);                                   /* |fr| in [0.5, 1) */
    *M = (int64_t)ldexp(fr, 53);
    *E = ex - 53;
}
int32_t tr_calc_sincos(double x, tr_math_interval *s, tr_math_interval *c) {
    if (!s || !c || !isfinite(x)) return TR_MATH_INVALID;
    double ax = fabs(x);
    if (ax >= 0x1p62) return TR_MATH_UNSUPPORTED;
    if (x == 0) { s->lo = s->hi = 0.0; c->lo = c->hi = 1.0; return TR_MATH_OK; }
    if (ax < 0x1p-30) {                                          /* |sin x - x| < |x|^3/6 < ulp(x), 1 - cos x < 2^-61 */
        if (x > 0) { s->lo = tr_step(x, 0); s->hi = x; } else { s->lo = x; s->hi = tr_step(x, 1); }
        c->lo = tr_step(1.0, 0); c->hi = 1.0;
        return TR_MATH_OK;
    }
    int64_t M;
    int32_t E;
    decompose(x, &M, &E);
    const uint64_t K[4] = {TWO_OVER_PI_512[5], TWO_OVER_PI_512[6], TWO_OVER_PI_512[7], 0};   /* 2/pi: 192 bits */
    tr_calc_fx sn, cs;
    tr_calc_sincos_reduce(K, 1, M, E, &sn, &cs);
    *s = fx_interval(sn, 0);
    *c = fx_interval(cs, 0);
    if (s->lo < -1) s->lo = -1;                                  /* |sin|, |cos| <= 1 exactly */
    if (s->hi > 1) s->hi = 1;
    if (c->lo < -1) c->lo = -1;
    if (c->hi > 1) c->hi = 1;
    return TR_MATH_OK;
}
int32_t tr_calc_exp(double x, tr_math_interval *out) {
    if (!out || !isfinite(x)) return TR_MATH_INVALID;
    if (x == 0) { out->lo = out->hi = 1.0; return TR_MATH_OK; }
    if (x > 1000) { out->lo = DBL_MAX; out->hi = INFINITY; return TR_MATH_OK; }
    if (x < -1100) { out->lo = 0.0; out->hi = 0x1p-1074; return TR_MATH_OK; }
    if (fabs(x) < 0x1p-60) { out->lo = tr_step(1.0, 0); out->hi = tr_step(1.0, 1); return TR_MATH_OK; }   /* |e^x - 1| < 2^-59 */
    int64_t M;
    int32_t E;
    decompose(x, &M, &E);                                        /* E in [-113, -42]: 1/ln 2 shifted right by -E */
    tr_calc_plan_view pl;
    memset(&pl, 0, sizeof pl);
    pl.family = TR_CALC_EXP; pl.nfactors = 1; pl.weight = -1; pl.nsubsets = 1; pl.ngroups = 1;
    pl.ksign[0] = 1;
    /* K = (1/ln 2) 2^E as a 64.192 fixed point: floor(LOG2E_512 / 2^(320 - E)), truncation below 2^-192 */
    for (int32_t j = 0; j < 4; j++) pl.k[0][j] = tr_calc_bits64(LOG2E_512, 9, 64 * j + 320 - E);
    int64_t n;
    tr_calc_fx m;
    if (!tr_calc_exp2_reduce(&pl, &M, &n, &m)) return TR_MATH_OVERFLOW;
    tr_math_interval iv = fx_interval(m, 0);                     /* 2^f in [1, 2] */
    out->lo = ldexp_dir(iv.lo, (int)n, 0);
    out->hi = ldexp_dir(iv.hi, (int)n, 1);
    if (out->lo < 0) out->lo = 0.0;
    return TR_MATH_OK;
}
int32_t tr_calc_sincos_scaled(const tr_bigq *omega, int64_t X, int32_t scale, tr_math_interval *s, tr_math_interval *c) {
    if (!s || !c) return TR_MATH_INVALID;
    uint64_t k[4];
    int32_t ks;
    int32_t rc = make_k(omega, scale, TR_CALC_PHASE, k, &ks, NULL, NULL);
    if (rc) return rc;
    tr_calc_fx sn, cs;
    tr_calc_sincos_reduce(k, ks, X, 0, &sn, &cs);
    *s = fx_interval(sn, 0);
    *c = fx_interval(cs, 0);
    return TR_MATH_OK;
}
int32_t tr_calc_exp_scaled(const tr_bigq *a, int64_t X, int32_t scale, tr_math_interval *out) {
    if (!out) return TR_MATH_INVALID;
    tr_calc_plan_view pl;
    memset(&pl, 0, sizeof pl);
    pl.family = TR_CALC_EXP; pl.nfactors = 1; pl.weight = -1; pl.nsubsets = 1; pl.ngroups = 1;
    int32_t rc = make_k(a, scale, TR_CALC_EXP, pl.k[0], &pl.ksign[0], NULL, NULL);
    if (rc) return rc;
    int64_t n;
    tr_calc_fx m;
    if (!tr_calc_exp2_reduce(&pl, &X, &n, &m)) return TR_MATH_OVERFLOW;
    tr_math_interval iv = fx_interval(m, 0);
    out->lo = ldexp_dir(iv.lo, (int)n, 0);
    out->hi = ldexp_dir(iv.hi, (int)n, 1);
    if (out->lo < 0) out->lo = 0.0;
    return TR_MATH_OK;
}
