/* SPEC-001: fixed spectral basis states; every contract is in math_fourier.h.
 * SPDX-License-Identifier: Apache-2.0 */
#include "math_fourier.h"
#include "math_handles.h"
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

TR_MATH_ABI(tr_fourier_spec, TR_FOURIER_ABI);       /* the ctypes boundary of SPEC-001 (math_handles.h) */
TR_MATH_ABI(tr_fourier_plan_view, TR_FOURIER_ABI);
TR_MATH_ABI(tr_fourier_info, TR_FOURIER_ABI);

_Static_assert(sizeof(tr_fourier_spec) == 80 && sizeof(tr_fourier_plan_view) == 120 && sizeof(tr_fourier_info) == 128,
               "SPEC-001 ABI");

#define PLAN_MAGIC 0x31504653u    /* "SFP1" */
#define STATE_MAGIC 0x31534653u   /* "SFS1" */

/* floor(2/pi * 2^512), little-endian limbs (the constant of CALC-003, math_calc.c: Machin's formula in exact integer
 * arithmetic); its top three limbs are the 192-bit fraction of 2/pi */
static const uint64_t TWO_OVER_PI_512[8] = {0xe88235f52ebb4484ULL, 0xfe1deb1cb129a73eULL, 0x06492eea09d1921cULL,
                                            0xb7246e3a424dd2e0ULL, 0xfe5163abdebbc561ULL, 0xdb6295993c439041ULL,
                                            0xfc2757d1f534ddc0ULL, 0xa2f9836e4e441529ULL};
static const uint64_t TWO_OVER_PI_K[4] = {0xdb6295993c439041ULL, 0xfc2757d1f534ddc0ULL, 0xa2f9836e4e441529ULL, 0};
/* RD(pi) and RU(pi): pi = 0x1.921fb54442d18469898cc51701b8...p+1 */
#define PI_LO 0x1.921fb54442d18p+1
#define PI_HI 0x1.921fb54442d19p+1
static const double POW10[TR_SCOPE_MAX_SCALE + 1] = {1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11,
                                                     1e12, 1e13, 1e14, 1e15, 1e16, 1e17, 1e18};   /* exact doubles */

static void seterr(int32_t *e, int32_t v) { if (e) *e = v; }
static const tr_math_budget *budget_of(const tr_math_budget *b, tr_math_budget *store) {
    if (b) return b;
    tr_budget_default(store);
    return store;
}
static void *zalloc(uint64_t bytes) {
    if (bytes > (uint64_t)SIZE_MAX) return NULL;
    return calloc(1, bytes ? (size_t)bytes : 1);
}
static uint64_t bits_u128(unsigned __int128 x) {
    uint64_t hi = (uint64_t)(x >> 64), lo = (uint64_t)x;
    return hi ? (uint64_t)(128 - __builtin_clzll(hi)) : lo ? (uint64_t)(64 - __builtin_clzll(lo)) : 0;
}

/* ================================================================ directed conversions */
/* RD / RU of v * 2^e for a finite double v (as CALC-003: ldexp is exact unless the result overflows or is subnormal) */
static double ldexp_dir(double v, int e, int up) {
    if (v == 0) return 0.0;
    double r = ldexp(v, e);
    if (r == INFINITY) return up ? r : DBL_MAX;
    if (r == -INFINITY) return up ? -DBL_MAX : r;
    if (fabs(r) >= DBL_MIN || (r != 0 && ldexp(r, -e) == v)) return r;
    if (r == 0) return (up == (v > 0)) ? (v > 0 ? 0x1p-1074 : -0x1p-1074) : 0.0;
    return tr_step(r, up);
}
static tr_math_interval clean(tr_math_interval a) {
    if (a.lo == 0) a.lo = 0.0;
    if (a.hi == 0) a.hi = 0.0;
    return a;
}
/* the enclosure of [lo, hi] / (2^s 10^ws) */
static tr_math_interval scaled_interval(tr_math_i128 lo, tr_math_i128 hi, int32_t s, int32_t ws) {
    tr_math_interval a, b, r;
    tr_carrier_enclose_i128(lo, &a);
    tr_carrier_enclose_i128(hi, &b);
    double l = a.lo, h = b.hi;
    if (ws) { l = tr_div_dir(l, POW10[ws], 0); h = tr_div_dir(h, POW10[ws], 1); }
    r.lo = ldexp_dir(l, -s, 0);
    r.hi = ldexp_dir(h, -s, 1);
    return clean(r);
}
/* (c +- r) / 2^120 enclosed */
static tr_math_interval fx_interval(tr_calc_fx x) {
    tr_math_interval lo, hi, out;
    tr_carrier_enclose_i128(tr_i128_set(x.c - (__int128)x.r), &lo);
    tr_carrier_enclose_i128(tr_i128_set(x.c + (__int128)x.r), &hi);
    out.lo = ldexp_dir(lo.lo, -TR_CALC_F, 0);
    out.hi = ldexp_dir(hi.hi, -TR_CALC_F, 1);
    if (out.lo < -1) out.lo = -1;                               /* |sin|, |cos| <= 1 exactly */
    if (out.hi > 1) out.hi = 1;
    return clean(out);
}
static tr_carrier_cinterval cmul(tr_carrier_cinterval a, tr_carrier_cinterval b) {
    tr_carrier_cinterval r;
    r.re = tr_interval_sub(tr_interval_mul(a.re, b.re), tr_interval_mul(a.im, b.im));
    r.im = tr_interval_add(tr_interval_mul(a.re, b.im), tr_interval_mul(a.im, b.re));
    r.re = clean(r.re); r.im = clean(r.im);
    return r;
}
static tr_carrier_cinterval cscale(tr_carrier_cinterval a, tr_math_interval f) {
    tr_carrier_cinterval r;
    r.re = clean(tr_interval_mul(a.re, f)); r.im = clean(tr_interval_mul(a.im, f));
    return r;
}
/* times i^m (exact) */
static tr_carrier_cinterval crot(tr_carrier_cinterval a, int32_t m) {
    tr_carrier_cinterval r;
    switch (((m % 4) + 4) % 4) {
    case 0: r = a; break;
    case 1: r.re = tr_interval_neg(a.im); r.im = a.re; break;
    case 2: r.re = tr_interval_neg(a.re); r.im = tr_interval_neg(a.im); break;
    default: r.re = a.im; r.im = tr_interval_neg(a.re); break;
    }
    r.re = clean(r.re); r.im = clean(r.im);
    return r;
}
/* x = M 2^E, |M| < 2^53 */
static void decompose(double x, int64_t *M, int32_t *E) {
    int ex;
    double fr = frexp(x, &ex);
    *M = (int64_t)ldexp(fr, 53);
    *E = ex - 53;
}

/* ================================================================ reduction constants */
/* rem of n[0..len) / d (64-bit divisor), quotient into q */
static uint64_t div_limbs(const uint64_t *n, int32_t len, uint64_t d, uint64_t *q) {
    unsigned __int128 rem = 0;
    for (int32_t i = len - 1; i >= 0; i--) {
        rem = (rem << 64) | n[i];
        q[i] = (uint64_t)(rem / d);
        rem %= d;
    }
    return (uint64_t)rem;
}
/* |K| = floor(|p| C / (q 10^scale)) / 2^192 (C = 2/pi 2^512 / 2^320 for radians, 2 or 4 for pi and turn units) as
 * four limbs; kexact = the floor dropped nothing (never for a nonzero radian frequency); TR_MATH_UNSUPPORTED when
 * K >= 2^64 */
static int32_t make_k(tr_num_rational w, int32_t scale, int32_t unit, uint64_t *k, int32_t *ksign, uint8_t *kexact,
                      const tr_math_budget *b, tr_math_refusal *r) {
    memset(k, 0, 4 * sizeof(uint64_t));
    *kexact = 1;
    if (w.denominator <= 0 || scale < 0 || scale > TR_SCOPE_MAX_SCALE) return TR_MATH_INVALID;
    *ksign = w.numerator > 0 ? 1 : w.numerator < 0 ? -1 : 0;
    if (!*ksign) return TR_MATH_OK;
    uint64_t ap = tr_abs_u64(w.numerator);
    unsigned __int128 D = (uint64_t)w.denominator;
    for (int32_t i = 0; i < scale; i++) D *= 10;                       /* < 2^63 10^18 < 2^123 */
    uint64_t N[9];
    int32_t len;
    memset(N, 0, sizeof N);
    if (unit == TR_FOURIER_RADIAN) {
        unsigned __int128 carry = 0;
        for (int i = 0; i < 8; i++) {
            unsigned __int128 t = (unsigned __int128)ap * TWO_OVER_PI_512[i] + carry;
            N[i] = (uint64_t)t;
            carry = t >> 64;
        }
        N[8] = (uint64_t)carry;
        len = 9;
    } else {
        unsigned __int128 m = (unsigned __int128)ap * (unit == TR_FOURIER_PI ? 2u : 4u);
        N[3] = (uint64_t)m; N[4] = (uint64_t)(m >> 64);
        len = 5;
    }
    int32_t shift = unit == TR_FOURIER_RADIAN ? 5 : 0;                  /* radians: K = floor(N / D) / 2^320 */
    if (D <= (unsigned __int128)UINT64_MAX) {
        uint64_t Q[9];
        uint64_t rem = div_limbs(N, len, (uint64_t)D, Q);
        for (int32_t i = shift + 4; i < len; i++) if (Q[i]) return TR_MATH_UNSUPPORTED;
        for (int32_t i = 0; i < 4; i++) k[i] = Q[shift + i];
        int drop = rem != 0;
        for (int32_t i = 0; i < shift; i++) drop = drop || Q[i] != 0;
        *kexact = unit == TR_FOURIER_RADIAN ? 0 : (uint8_t)!drop;
        return TR_MATH_OK;
    }
    int32_t e = 0;                                                     /* a divisor past 64 bits: big integers */
    uint64_t dl[2] = {(uint64_t)D, (uint64_t)(D >> 64)};
    tr_bigint *bn = tr_bigint_from_limbs(1, N, (uint64_t)len, b, r, &e);
    tr_bigint *bd = bn ? tr_bigint_from_limbs(1, dl, 2, b, r, &e) : NULL;
    tr_bigint *q = NULL, *rm = NULL, *K = NULL;
    int32_t rc = bd ? tr_bigint_divmod(bn, bd, TR_DIV_FLOOR, &q, &rm, b, r) : (e ? e : TR_MATH_MEMORY);
    if (!rc) { K = tr_bigint_shr(q, 64 * (uint64_t)shift, b, r, &e); if (!K) rc = e ? e : TR_MATH_MEMORY; }
    if (!rc) {
        if (tr_bigint_bit_length(K) > 256) rc = TR_MATH_UNSUPPORTED;
        else if (tr_bigint_sign(K)) { int32_t sg; uint64_t cnt; rc = tr_bigint_export(K, &sg, k, 4, &cnt); }
        if (!rc) *kexact = unit == TR_FOURIER_RADIAN ? 0 : (uint8_t)(tr_bigint_sign(rm) == 0);   /* shift 0 */
    }
    tr_bigint_free(bn); tr_bigint_free(bd); tr_bigint_free(q); tr_bigint_free(rm); tr_bigint_free(K);
    return rc;
}

/* ================================================================ frequency sets */
static int freq_cmp(const tr_num_rational *a, const tr_num_rational *b, int32_t d) {
    for (int32_t j = 0; j < d; j++) { int c = tr_q_cmp(a[j], b[j]); if (c) return c; }
    return 0;
}
/* idx = the positions 0..m-1 sorted by frequency (bottom-up merge sort); 1 when two frequencies are equal */
static int32_t sort_frequencies(const tr_num_rational *om, uint64_t m, int32_t d, uint64_t *idx, int *dup) {
    uint64_t *tmp = (uint64_t *)zalloc(m * 8);
    if (!tmp) return TR_MATH_MEMORY;
    for (uint64_t i = 0; i < m; i++) idx[i] = i;
    for (uint64_t w = 1; w < m; w *= 2) {
        for (uint64_t lo = 0; lo < m; lo += 2 * w) {
            uint64_t mid = lo + w < m ? lo + w : m, hi = lo + 2 * w < m ? lo + 2 * w : m, i = lo, j = mid, o = lo;
            while (i < mid && j < hi) tmp[o++] = freq_cmp(om + idx[j] * d, om + idx[i] * d, d) < 0 ? idx[j++] : idx[i++];
            while (i < mid) tmp[o++] = idx[i++];
            while (j < hi) tmp[o++] = idx[j++];
        }
        memcpy(idx, tmp, (size_t)m * 8);
    }
    free(tmp);
    *dup = 0;
    for (uint64_t i = 1; i < m; i++) if (!freq_cmp(om + idx[i - 1] * d, om + idx[i] * d, d)) *dup = 1;
    return TR_MATH_OK;
}
/* the position of frequency w in the sorted basis, or UINT64_MAX */
static uint64_t sorted_find(const tr_num_rational *om, const uint64_t *idx, uint64_t m, int32_t d, const tr_num_rational *w) {
    uint64_t lo = 0, hi = m;
    while (lo < hi) {
        uint64_t mid = lo + (hi - lo) / 2;
        int c = freq_cmp(om + idx[mid] * d, w, d);
        if (!c) return idx[mid];
        if (c < 0) lo = mid + 1; else hi = mid;
    }
    return UINT64_MAX;
}
static int32_t canon(const tr_num_rational *in, uint64_t n, tr_num_rational *out) {
    for (uint64_t i = 0; i < n; i++) {
        if (in[i].denominator == 0) return TR_MATH_INVALID;
        if (!tr_q_make(in[i].numerator, in[i].denominator, &out[i])) return TR_MATH_OVERFLOW;
    }
    return TR_MATH_OK;
}

/* ================================================================ plans */
struct tr_fourier_plan_s {
    uint32_t magic, reserved;
    tr_fourier_plan_view v;
    tr_fourier_spec spec;
    int32_t wscale, ncols;
    uint64_t rows, selected;
    uint64_t *k, *ks;                  /* data and translation constants: m d 4 limbs */
    int32_t *ksign, *kssign;
    uint8_t *kexact, *ksexact;
    tr_num_rational *omega;            /* m d, canonical */
    uint64_t *sorted;
    int64_t colmin[TR_FOURIER_MAX_DIMS], colmax[TR_FOURIER_MAX_DIMS];
    int32_t colscale[TR_FOURIER_MAX_DIMS];
    unsigned char *key;
    uint64_t keybytes;
};
void tr_fourier_plan_free(tr_fourier_plan *p) {
    if (!p || p->magic != PLAN_MAGIC) return;
    p->magic = 0;
    free(p->k); free(p->ks); free(p->ksign); free(p->kssign); free(p->kexact); free(p->ksexact); free(p->omega);
    free(p->sorted); free(p->key);
    free(p);
}
static int32_t spec_check(const tr_fourier_spec *s, int32_t ncols) {
    if (!s) return TR_MATH_INVALID;
    if (s->ndims > TR_FOURIER_MAX_DIMS) return TR_MATH_UNSUPPORTED;
    if (s->ndims < 1 || s->weight < -1 || s->weight >= ncols || s->reserved || s->unit < TR_FOURIER_RADIAN ||
        s->unit > TR_FOURIER_TURN) return TR_MATH_INVALID;
    for (int32_t j = 0; j < s->ndims; j++) if (s->column[j] < 0 || s->column[j] >= ncols) return TR_MATH_INVALID;
    return TR_MATH_OK;
}
/* bytes of a state of G groups and m frequencies of d dimensions */
static uint64_t state_bytes(uint64_t G, uint64_t m, int32_t d, uint64_t keybytes) {
    uint64_t b = tr_budget_add(256, tr_budget_mul(G, 8 + 8 + 16 + 16));
    b = tr_budget_add(b, tr_budget_mul(tr_budget_mul(G, m), 64));                         /* the sums */
    b = tr_budget_add(b, tr_budget_mul(tr_budget_mul(m, (uint64_t)d), 16 + 16 + 32 + 4 + 1));   /* omega, enclosures, K' */
    b = tr_budget_add(b, tr_budget_mul(m, 8));                                            /* the sorted index */
    return tr_budget_add(b, keybytes);
}
static uint64_t plan_bytes(uint64_t m, int32_t d, uint64_t keybytes) {
    uint64_t b = tr_budget_add(sizeof(tr_fourier_plan), tr_budget_mul(tr_budget_mul(m, (uint64_t)d), 2 * (32 + 4 + 1) + 16));
    return tr_budget_add(tr_budget_add(b, tr_budget_mul(m, 16)), keybytes);              /* sorted index and scratch */
}
static void need_of(uint64_t G, uint64_t m, int32_t d, uint64_t selected, uint64_t keybytes, tr_math_need *n) {
    memset(n, 0, sizeof *n);
    n->need[TR_BUDGET_VARIABLES] = (uint64_t)d;
    n->need[TR_BUDGET_SUPPORT] = m;
    n->need[TR_BUDGET_STATES] = tr_budget_mul(G, m);
    n->need[TR_BUDGET_TENSOR] = tr_budget_mul(selected, m);
    uint64_t bytes = tr_budget_add(plan_bytes(m, d, keybytes), state_bytes(G, m, d, keybytes));
    n->need[TR_BUDGET_BYTES] = tr_budget_add(bytes, tr_budget_mul(G, tr_budget_add(tr_budget_mul(m, 64), 32)));   /* sums */
}

tr_fourier_plan *tr_fourier_plan_create(const tr_scope *scope, const tr_fourier_spec *spec, uint64_t nfreq,
                                        const tr_num_rational *omega, const tr_math_budget *budget,
                                        tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    tr_math_budget dflt;
    budget = budget_of(budget, &dflt);
    tr_scope_info info;
    int32_t rc = scope && omega ? tr_scope_get_info(scope, &info) : TR_MATH_INVALID;
    if (!rc) rc = spec_check(spec, info.ncols);
    if (!rc && !nfreq) rc = TR_MATH_INVALID;
    if (!rc && nfreq > TR_FOURIER_MAX_FREQS) rc = TR_MATH_UNSUPPORTED;
    if (rc) { seterr(error, rc); return NULL; }
    const int32_t d = spec->ndims;
    uint64_t keybytes = tr_scope_key_size(scope, TR_SCOPE_KEY_ALL);
    tr_math_need need;
    need_of(info.ngroups, nfreq, d, info.selected, keybytes, &need);
    rc = tr_budget_check(budget, &need, refusal);
    if (rc) { seterr(error, rc); return NULL; }
    tr_fourier_plan *p = (tr_fourier_plan *)zalloc(sizeof *p);
    if (!p) { seterr(error, TR_MATH_MEMORY); return NULL; }
    p->magic = PLAN_MAGIC;
    p->spec = *spec;
    p->ncols = info.ncols;
    p->rows = info.rows; p->selected = info.selected;
    const uint64_t md = nfreq * (uint64_t)d;
    p->k = (uint64_t *)zalloc(md * 32); p->ks = (uint64_t *)zalloc(md * 32);
    p->ksign = (int32_t *)zalloc(md * 4); p->kssign = (int32_t *)zalloc(md * 4);
    p->kexact = (uint8_t *)zalloc(md); p->ksexact = (uint8_t *)zalloc(md);
    p->omega = (tr_num_rational *)zalloc(md * sizeof(tr_num_rational));
    p->sorted = (uint64_t *)zalloc(nfreq * 8);
    p->keybytes = keybytes;
    p->key = (unsigned char *)malloc(keybytes ? (size_t)keybytes : 1);
    if (!p->k || !p->ks || !p->ksign || !p->kssign || !p->kexact || !p->ksexact || !p->omega || !p->sorted || !p->key)
        rc = TR_MATH_MEMORY;
    if (!rc) rc = tr_scope_key_write(scope, TR_SCOPE_KEY_ALL, p->key, keybytes);
    if (!rc) rc = canon(omega, md, p->omega);
    int dup = 0;
    if (!rc) rc = sort_frequencies(p->omega, nfreq, d, p->sorted, &dup);
    if (!rc && dup) rc = TR_MATH_INVALID;                              /* a basis is a set */
    for (int32_t j = 0; j < d && !rc; j++) {
        tr_scope_column_info ci;
        rc = tr_scope_column_get(scope, spec->column[j], &ci);
        if (rc) break;
        p->colmin[j] = ci.min; p->colmax[j] = ci.max; p->colscale[j] = ci.scale;
        if (!ci.nvalid) { p->colmin[j] = 0; p->colmax[j] = -1; }
        for (uint64_t f = 0; f < nfreq && !rc; f++) {
            uint64_t e = f * (uint64_t)d + (uint64_t)j;
            rc = make_k(p->omega[e], ci.scale, spec->unit, p->k + 4 * e, &p->ksign[e], &p->kexact[e], budget, refusal);
            if (!rc) rc = make_k(p->omega[e], 0, spec->unit, p->ks + 4 * e, &p->kssign[e], &p->ksexact[e], budget, refusal);
        }
    }
    /* the scale: W = selected rows x max |w| (exact), every sum below 2^126 */
    unsigned __int128 W = info.selected;
    if (!rc && spec->weight >= 0) {
        tr_scope_column_info wi;
        rc = tr_scope_column_get(scope, spec->weight, &wi);
        if (!rc) {
            p->wscale = wi.scale;
            uint64_t mw = wi.nvalid ? tr_abs_u64(wi.min) : 0, mx = wi.nvalid ? tr_abs_u64(wi.max) : 0;
            W = (unsigned __int128)info.selected * (mw > mx ? mw : mx);
        }
    }
    if (rc) { tr_fourier_plan_free(p); seterr(error, rc); return NULL; }
    int64_t s = 124 - (int64_t)bits_u128(W);
    if (s > TR_CALC_SCALE_MAX) s = TR_CALC_SCALE_MAX;
    tr_fourier_plan_view *v = &p->v;
    v->ndims = d; v->weight = spec->weight; v->scale = (int32_t)s; v->unit = spec->unit;
    v->nfreq = nfreq; v->ngroups = info.ngroups;
    for (int32_t j = 0; j < d; j++) v->column[j] = spec->column[j];
    v->k = p->k; v->ksign = p->ksign; v->kexact = p->kexact;
    seterr(error, TR_MATH_OK);
    return p;
}
int32_t tr_fourier_plan_get_view(const tr_fourier_plan *p, tr_fourier_plan_view *out) {
    if (!p || p->magic != PLAN_MAGIC || !out) return TR_MATH_INVALID;
    *out = p->v;
    return TR_MATH_OK;
}
uint64_t tr_fourier_device_need(const tr_fourier_plan_view *p, const tr_scope_view *v) {
    return p && v ? tr_fourier_device_bytes(p, v) : TR_BUDGET_UNKNOWN;
}

static int add_i128(tr_math_i128 *acc, __int128 x) {
    __int128 s;
    if (__builtin_add_overflow(tr_i128_get(*acc), x, &s)) return 0;
    *acc = tr_i128_set(s);
    return 1;
}
int32_t tr_fourier_sums(const tr_fourier_plan_view *pl, const tr_scope_view *v, tr_math_i128 *sums, uint64_t *rows,
                        uint64_t *valid, tr_math_i128 *wsum) {
    if (!tr_fourier_plan_view_valid(pl, v) || !sums || !rows || !valid || !wsum) return TR_MATH_INVALID;
    const uint64_t m = pl->nfreq, G = pl->ngroups;
    memset(sums, 0, (size_t)(G * m * 4) * sizeof *sums);
    memset(rows, 0, (size_t)G * 8);
    memset(valid, 0, (size_t)G * 8);
    memset(wsum, 0, (size_t)G * sizeof *wsum);
    for (uint64_t r = 0; r < v->rows; r++) {
        if (!tr_scope_is_selected(v, r)) continue;
        uint32_t g = tr_scope_group(v, r);
        if ((uint64_t)g >= G) return TR_MATH_INVALID;
        rows[g]++;
        if (!tr_fourier_row_ok(pl, v, r)) continue;
        valid[g]++;
        if (!add_i128(&wsum[g], tr_fourier_row_weight(pl, v, r))) return TR_MATH_OVERFLOW;
        tr_math_i128 *acc = sums + (uint64_t)g * m * 4;
        for (uint64_t f = 0; f < m; f++) {
            __int128 t[4];
            if (!tr_fourier_row_terms(pl, v, f, r, t)) return TR_MATH_OVERFLOW;
            for (int c = 0; c < 4; c++) if (!add_i128(&acc[4 * f + (uint64_t)c], t[c])) return TR_MATH_OVERFLOW;
        }
    }
    return TR_MATH_OK;
}

/* ================================================================ states */
struct tr_fourier_state_s {
    uint32_t magic, reserved;
    tr_fourier_info info;
    tr_fourier_spec spec;
    uint64_t *rows, *valid;
    tr_math_i128 *wsum;                /* G: exact weight sums (logical) */
    tr_math_interval *wenc;            /* G: the weight sums in value units, enclosed */
    tr_math_i128 *sums;                /* G m 4 */
    tr_num_rational *omega;            /* m d */
    tr_math_interval *oenc;            /* m d: tight enclosures of the frequencies */
    uint64_t *ks;                      /* m d 4: the translation constants c omega (2/pi) */
    int32_t *kssign;
    uint8_t *ksexact;
    uint64_t *sorted;
    int64_t colmin[TR_FOURIER_MAX_DIMS], colmax[TR_FOURIER_MAX_DIMS];
    int32_t colscale[TR_FOURIER_MAX_DIMS];
    unsigned char *key;
    uint64_t keybytes;
};
static int state_ok(const tr_fourier_state *s) { return s && s->magic == STATE_MAGIC; }
void tr_fourier_free(tr_fourier_state *s) {
    if (!state_ok(s)) return;
    s->magic = 0;
    free(s->rows); free(s->valid); free(s->wsum); free(s->wenc); free(s->sums); free(s->omega); free(s->oenc);
    free(s->ks); free(s->kssign); free(s->ksexact); free(s->sorted); free(s->key);
    free(s);
}
static tr_fourier_state *state_alloc(uint64_t G, uint64_t m, int32_t d, uint64_t keybytes) {
    tr_fourier_state *s = (tr_fourier_state *)zalloc(sizeof *s);
    if (!s) return NULL;
    const uint64_t md = m * (uint64_t)d;
    s->rows = (uint64_t *)zalloc(G * 8); s->valid = (uint64_t *)zalloc(G * 8);
    s->wsum = (tr_math_i128 *)zalloc(G * 16); s->wenc = (tr_math_interval *)zalloc(G * 16);
    s->sums = (tr_math_i128 *)zalloc(G * m * 64);
    s->omega = (tr_num_rational *)zalloc(md * 16); s->oenc = (tr_math_interval *)zalloc(md * 16);
    s->ks = (uint64_t *)zalloc(md * 32); s->kssign = (int32_t *)zalloc(md * 4); s->ksexact = (uint8_t *)zalloc(md);
    s->sorted = (uint64_t *)zalloc(m * 8);
    s->keybytes = keybytes;
    s->key = (unsigned char *)malloc(keybytes ? (size_t)keybytes : 1);
    s->magic = STATE_MAGIC;
    if (!s->rows || !s->valid || !s->wsum || !s->wenc || !s->sums || !s->omega || !s->oenc || !s->ks || !s->kssign ||
        !s->ksexact || !s->sorted || !s->key) { tr_fourier_free(s); return NULL; }
    return s;
}
static tr_fourier_state *state_copy(const tr_fourier_state *src) {
    const uint64_t G = src->info.ngroups, m = src->info.nfreq, md = m * (uint64_t)src->info.ndims;
    tr_fourier_state *s = state_alloc(G, m, src->info.ndims, src->keybytes);
    if (!s) return NULL;
    s->info = src->info; s->spec = src->spec;
    memcpy(s->rows, src->rows, (size_t)G * 8); memcpy(s->valid, src->valid, (size_t)G * 8);
    memcpy(s->wsum, src->wsum, (size_t)G * 16); memcpy(s->wenc, src->wenc, (size_t)G * 16);
    memcpy(s->sums, src->sums, (size_t)(G * m) * 64);
    memcpy(s->omega, src->omega, (size_t)md * 16); memcpy(s->oenc, src->oenc, (size_t)md * 16);
    memcpy(s->ks, src->ks, (size_t)md * 32); memcpy(s->kssign, src->kssign, (size_t)md * 4);
    memcpy(s->ksexact, src->ksexact, (size_t)md);
    memcpy(s->sorted, src->sorted, (size_t)m * 8);
    memcpy(s->colmin, src->colmin, sizeof s->colmin); memcpy(s->colmax, src->colmax, sizeof s->colmax);
    memcpy(s->colscale, src->colscale, sizeof s->colscale);
    memcpy(s->key, src->key, (size_t)src->keybytes);
    return s;
}

tr_fourier_state *tr_fourier_from_sums(const tr_fourier_plan *p, const tr_scope *scope, const tr_math_i128 *sums,
                                       const uint64_t *rows, const uint64_t *valid, const tr_math_i128 *wsum,
                                       const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    tr_math_budget dflt;
    budget = budget_of(budget, &dflt);
    if (!p || p->magic != PLAN_MAGIC || !scope || !sums || !rows || !valid || !wsum) { seterr(error, TR_MATH_INVALID); return NULL; }
    if (!tr_scope_key_equal(scope, TR_SCOPE_KEY_ALL, p->key, p->keybytes)) { seterr(error, TR_MATH_STALE); return NULL; }
    const uint64_t G = p->v.ngroups, m = p->v.nfreq;
    const int32_t d = p->v.ndims;
    tr_math_need need;
    memset(&need, 0, sizeof need);
    need.need[TR_BUDGET_SUPPORT] = m;
    need.need[TR_BUDGET_STATES] = tr_budget_mul(G, m);
    need.need[TR_BUDGET_BYTES] = state_bytes(G, m, d, p->keybytes);
    int32_t rc = tr_budget_check(budget, &need, refusal);
    if (rc) { seterr(error, rc); return NULL; }
    uint64_t total = 0;
    for (uint64_t g = 0; g < G; g++) {                                 /* consistency of the readback */
        if (valid[g] > rows[g] || (!valid[g] && tr_i128_get(wsum[g]) != 0)) { seterr(error, TR_MATH_INVALID); return NULL; }
        total += rows[g];
        for (uint64_t i = 0; i < 4 * m; i++)
            if (!valid[g] && tr_i128_get(sums[g * m * 4 + i]) != 0) { seterr(error, TR_MATH_INVALID); return NULL; }
    }
    if (total != p->selected) { seterr(error, TR_MATH_INVALID); return NULL; }
    int exact = 1;
    for (uint64_t i = 0; i < G * m * 2; i++) {
        __int128 lo = tr_i128_get(sums[2 * i]), hi = tr_i128_get(sums[2 * i + 1]);
        if (lo > hi) { seterr(error, TR_MATH_INVALID); return NULL; }
        if (lo != hi) exact = 0;
    }
    tr_fourier_state *s = state_alloc(G, m, d, p->keybytes);
    if (!s) { seterr(error, TR_MATH_MEMORY); return NULL; }
    s->spec = p->spec;
    tr_fourier_info *in = &s->info;
    in->ndims = d; in->unit = p->spec.unit; in->scale = p->v.scale; in->wscale = p->wscale;
    in->weight = p->spec.weight; in->exact = exact; in->phase_order = 0; in->unit_power = 0;
    in->ngroups = G; in->nfreq = m; in->rows = p->rows; in->selected = p->selected;
    memcpy(s->rows, rows, (size_t)G * 8); memcpy(s->valid, valid, (size_t)G * 8);
    memcpy(s->wsum, wsum, (size_t)G * 16);
    memcpy(s->sums, sums, (size_t)(G * m) * 64);
    const uint64_t md = m * (uint64_t)d;
    memcpy(s->omega, p->omega, (size_t)md * 16);
    memcpy(s->ks, p->ks, (size_t)md * 32); memcpy(s->kssign, p->kssign, (size_t)md * 4);
    memcpy(s->ksexact, p->ksexact, (size_t)md);
    memcpy(s->sorted, p->sorted, (size_t)m * 8);
    memcpy(s->colmin, p->colmin, sizeof s->colmin); memcpy(s->colmax, p->colmax, sizeof s->colmax);
    memcpy(s->colscale, p->colscale, sizeof s->colscale);
    memcpy(s->key, p->key, (size_t)p->keybytes);
    for (uint64_t e = 0; e < md && !rc; e++) rc = tr_carrier_enclose_q64(s->omega[e], &s->oenc[e]);
    for (uint64_t g = 0; g < G; g++) {
        tr_math_i128 w = wsum[g];
        s->wenc[g] = scaled_interval(w, w, 0, p->wscale);
    }
    if (rc) { tr_fourier_free(s); seterr(error, rc); return NULL; }
    seterr(error, TR_MATH_OK);
    return s;
}

tr_fourier_state *tr_fourier_build(const tr_scope *scope, const tr_fourier_spec *spec, uint64_t nfreq,
                                   const tr_num_rational *omega, const tr_math_budget *budget,
                                   tr_math_refusal *refusal, int32_t *error) {
    tr_fourier_plan *p = tr_fourier_plan_create(scope, spec, nfreq, omega, budget, refusal, error);
    if (!p) return NULL;
    const uint64_t G = p->v.ngroups, m = p->v.nfreq;
    tr_math_i128 *sums = (tr_math_i128 *)zalloc(G * m * 64), *wsum = (tr_math_i128 *)zalloc(G * 16);
    uint64_t *rows = (uint64_t *)zalloc(G * 8), *valid = (uint64_t *)zalloc(G * 8);
    tr_scope_view v;
    int32_t rc = sums && wsum && rows && valid ? tr_scope_get_view(scope, &v) : TR_MATH_MEMORY;
    if (!rc) rc = tr_fourier_sums(&p->v, &v, sums, rows, valid, wsum);
    tr_fourier_state *s = NULL;
    if (!rc) s = tr_fourier_from_sums(p, scope, sums, rows, valid, wsum, budget, refusal, error);
    else seterr(error, rc);
    free(sums); free(wsum); free(rows); free(valid);
    tr_fourier_plan_free(p);
    return s;
}

/* ================================================================ reading */
int32_t tr_fourier_get_info(const tr_fourier_state *s, tr_fourier_info *out) {
    if (!state_ok(s) || !out) return TR_MATH_INVALID;
    *out = s->info;
    return TR_MATH_OK;
}
int32_t tr_fourier_group(const tr_fourier_state *s, uint64_t g, uint64_t *rows, uint64_t *valid, tr_math_i128 *wsum) {
    if (!state_ok(s) || g >= s->info.ngroups || !rows || !valid || !wsum) return TR_MATH_INVALID;
    *rows = s->rows[g]; *valid = s->valid[g]; *wsum = s->wsum[g];
    return TR_MATH_OK;
}
int32_t tr_fourier_frequency(const tr_fourier_state *s, uint64_t f, tr_num_rational *out) {
    if (!state_ok(s) || f >= s->info.nfreq || !out) return TR_MATH_INVALID;
    memcpy(out, s->omega + f * (uint64_t)s->info.ndims, (size_t)s->info.ndims * sizeof *out);
    return TR_MATH_OK;
}
int32_t tr_fourier_bounds(const tr_fourier_state *s, uint64_t g, uint64_t f, tr_math_i128 *out4) {
    if (!state_ok(s) || g >= s->info.ngroups || f >= s->info.nfreq || !out4) return TR_MATH_INVALID;
    memcpy(out4, s->sums + (g * s->info.nfreq + f) * 4, 4 * sizeof *out4);
    return TR_MATH_OK;
}
static int32_t order_total(const tr_fourier_state *s) {
    int32_t t = s->info.phase_order;
    for (int32_t j = 0; j < s->info.ndims; j++) t += s->info.order[j];
    return t;
}
/* c^p enclosed */
static tr_math_interval unit_power(int32_t unit, int32_t p) {
    tr_math_interval c;
    if (unit == TR_FOURIER_RADIAN || p == 0) { c.lo = c.hi = 1.0; return c; }
    c.lo = PI_LO; c.hi = PI_HI;
    if (unit == TR_FOURIER_TURN) { c.lo *= 2; c.hi *= 2; }                /* exact doublings */
    return tr_interval_pown(c, (uint64_t)p);
}
/* the real derivative factor c^|alpha| prod_j omega_fj^alpha_j of frequency f, enclosed */
static tr_math_interval order_factor(const tr_fourier_state *s, uint64_t f, const int32_t *alpha) {
    int32_t tot = 0;
    tr_math_interval r;
    r.lo = r.hi = 1.0;
    for (int32_t j = 0; j < s->info.ndims; j++) {
        if (!alpha[j]) continue;
        tot += alpha[j];
        r = tr_interval_mul(r, tr_interval_pown(s->oenc[f * (uint64_t)s->info.ndims + (uint64_t)j], (uint64_t)alpha[j]));
    }
    return clean(tr_interval_mul(r, unit_power(s->info.unit, tot)));
}
static tr_carrier_cinterval coef_of(const tr_fourier_state *s, uint64_t g, uint64_t f) {
    const tr_math_i128 *b = s->sums + (g * s->info.nfreq + f) * 4;
    tr_carrier_cinterval z;
    z.re = scaled_interval(b[0], b[1], s->info.scale, s->info.wscale);
    z.im = scaled_interval(b[2], b[3], s->info.scale, s->info.wscale);
    int32_t tot = order_total(s);
    if (!tot) return z;
    return cscale(crot(z, tot), order_factor(s, f, s->info.order));
}
int32_t tr_fourier_coefficient(const tr_fourier_state *s, uint64_t g, uint64_t f, tr_carrier_cinterval *out) {
    if (!state_ok(s) || g >= s->info.ngroups || f >= s->info.nfreq || !out) return TR_MATH_INVALID;
    *out = coef_of(s, g, f);
    return TR_MATH_OK;
}

/* ---------------------------------------------------------------- exact bounds (big rationals) */
typedef struct { const tr_math_budget *b; tr_math_refusal *r; int32_t rc; } qctx;
static tr_bigq *qk(qctx *x, tr_bigq *v, int32_t e) { if (!v && !x->rc) x->rc = e ? e : TR_MATH_MEMORY; return v; }
static tr_bigq *Q_mul(qctx *x, const tr_bigq *a, const tr_bigq *c) { int32_t e = 0; return x->rc ? NULL : qk(x, tr_bigq_mul(a, c, x->b, x->r, &e), e); }
static tr_bigq *Q_neg(qctx *x, const tr_bigq *a) { int32_t e = 0; return x->rc ? NULL : qk(x, tr_bigq_neg(a, x->b, x->r, &e), e); }
static tr_bigq *Q_copy(qctx *x, const tr_bigq *a) { int32_t e = 0; return x->rc ? NULL : qk(x, tr_bigq_copy(a, x->b, x->r, &e), e); }
static tr_bigq *Q_q64(qctx *x, tr_num_rational q) { int32_t e = 0; return x->rc ? NULL : qk(x, tr_bigq_from_q64(q, x->b, x->r, &e), e); }
/* lo / (2^s 10^ws) exactly */
static tr_bigq *Q_scaled(qctx *x, tr_math_i128 v, int32_t s, int32_t ws) {
    if (x->rc) return NULL;
    int32_t e = 0;
    tr_bigint *n = tr_bigint_from_i128(v, x->b, x->r, &e), *one = n ? tr_bigint_from_i64(1, x->b, x->r, &e) : NULL;
    tr_bigint *p2 = one ? tr_bigint_shl(one, (uint64_t)(s >= 0 ? s : -s), x->b, x->r, &e) : NULL;
    tr_bigint *ten = p2 ? tr_bigint_from_i64(10, x->b, x->r, &e) : NULL;
    tr_bigint *t = ten ? tr_bigint_pow(ten, (uint64_t)ws, x->b, x->r, &e) : NULL;
    tr_bigint *num = NULL, *den = NULL;
    if (t) {
        if (s >= 0) { num = tr_bigint_copy(n, x->b, x->r, &e); den = num ? tr_bigint_mul(t, p2, x->b, x->r, &e) : NULL; }
        else { num = tr_bigint_mul(n, p2, x->b, x->r, &e); den = num ? tr_bigint_copy(t, x->b, x->r, &e) : NULL; }
    }
    tr_bigq *q = den ? tr_bigq_from_bigints(num, den, x->b, x->r, &e) : NULL;
    tr_bigint_free(n); tr_bigint_free(one); tr_bigint_free(p2); tr_bigint_free(ten); tr_bigint_free(t);
    tr_bigint_free(num); tr_bigint_free(den);
    return qk(x, q, e);
}
int32_t tr_fourier_exact(const tr_fourier_state *s, uint64_t g, uint64_t f, tr_bigq **out, int32_t *unit_pow,
                         const tr_math_budget *budget, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!state_ok(s) || g >= s->info.ngroups || f >= s->info.nfreq || !out || !unit_pow) return TR_MATH_INVALID;
    tr_math_budget dflt;
    qctx x = {budget_of(budget, &dflt), refusal, 0};
    const tr_math_i128 *b = s->sums + (g * s->info.nfreq + f) * 4;
    tr_bigq *v[4], *w[4] = {NULL, NULL, NULL, NULL};
    for (int i = 0; i < 4; i++) v[i] = Q_scaled(&x, b[i], s->info.scale, s->info.wscale);
    /* times i^m: (re, im) -> (-im, re) per quarter turn; a negated interval swaps its ends */
    int32_t m = ((order_total(s) % 4) + 4) % 4;
    int src[4], neg[4];
    switch (m) {
    case 0: src[0] = 0; src[1] = 1; src[2] = 2; src[3] = 3; neg[0] = neg[1] = neg[2] = neg[3] = 0; break;
    case 1: src[0] = 3; src[1] = 2; src[2] = 0; src[3] = 1; neg[0] = neg[1] = 1; neg[2] = neg[3] = 0; break;
    case 2: src[0] = 1; src[1] = 0; src[2] = 3; src[3] = 2; neg[0] = neg[1] = neg[2] = neg[3] = 1; break;
    default: src[0] = 2; src[1] = 3; src[2] = 1; src[3] = 0; neg[0] = neg[1] = 0; neg[2] = neg[3] = 1; break;
    }
    for (int i = 0; i < 4 && !x.rc; i++) w[i] = neg[i] ? Q_neg(&x, v[src[i]]) : Q_copy(&x, v[src[i]]);
    /* times prod_j omega_j^alpha_j (exact) */
    tr_bigq *r = NULL;
    for (int32_t j = 0; j < s->info.ndims && !x.rc; j++) {
        int32_t a = s->info.order[j];
        if (!a) continue;
        tr_bigq *o = Q_q64(&x, s->omega[f * (uint64_t)s->info.ndims + (uint64_t)j]);
        int32_t e = 0;
        tr_bigq *pw = o ? qk(&x, tr_bigq_pow(o, a, x.b, x.r, &e), e) : NULL;
        tr_bigq_free(o);
        if (!pw) break;
        if (!r) r = pw;
        else { tr_bigq *t = Q_mul(&x, r, pw); tr_bigq_free(r); tr_bigq_free(pw); r = t; }
    }
    if (r && !x.rc) {
        int sg = tr_bigq_sign(r);
        for (int k2 = 0; k2 < 2 && !x.rc; k2++) {
            tr_bigq *lo = Q_mul(&x, w[2 * k2], r), *hi = Q_mul(&x, w[2 * k2 + 1], r);
            tr_bigq_free(w[2 * k2]); tr_bigq_free(w[2 * k2 + 1]);
            w[2 * k2] = sg >= 0 ? lo : hi; w[2 * k2 + 1] = sg >= 0 ? hi : lo;
        }
    }
    tr_bigq_free(r);
    for (int i = 0; i < 4; i++) tr_bigq_free(v[i]);
    if (x.rc) { for (int i = 0; i < 4; i++) tr_bigq_free(w[i]); return x.rc; }
    for (int i = 0; i < 4; i++) out[i] = w[i];
    *unit_pow = s->info.unit_power;
    return TR_MATH_OK;
}

/* ---------------------------------------------------------------- identity */
int32_t tr_fourier_find(const tr_fourier_state *s, uint64_t n, const tr_num_rational *omega, uint64_t *index,
                        uint64_t *missing) {
    if (!state_ok(s) || (n && (!omega || !index))) return TR_MATH_INVALID;
    const int32_t d = s->info.ndims;
    tr_num_rational w[TR_FOURIER_MAX_DIMS];
    for (uint64_t i = 0; i < n; i++) {
        int32_t rc = canon(omega + i * (uint64_t)d, (uint64_t)d, w);
        if (rc) return rc;
        uint64_t at = sorted_find(s->omega, s->sorted, s->info.nfreq, d, w);
        if (at == UINT64_MAX) { if (missing) *missing = i; return TR_MATH_DOMAIN_ERROR; }   /* missing support */
        index[i] = at;
    }
    return TR_MATH_OK;
}
int32_t tr_fourier_require(const tr_fourier_state *s, const tr_scope *scope, const tr_fourier_spec *spec, uint64_t n,
                           const tr_num_rational *omega) {
    if (!state_ok(s) || !scope || !spec || (n && !omega)) return TR_MATH_INVALID;
    tr_scope_info info;
    int32_t rc = tr_scope_get_info(scope, &info);
    if (!rc) rc = spec_check(spec, info.ncols);
    if (rc) return rc;
    if (spec->ndims != s->spec.ndims || spec->weight != s->spec.weight || spec->unit != s->spec.unit ||
        memcmp(spec->column, s->spec.column, (size_t)spec->ndims * sizeof(int32_t)) || order_total(s))
        return TR_MATH_DOMAIN_ERROR;
    if (!tr_scope_key_equal(scope, TR_SCOPE_KEY_ALL, s->key, s->keybytes)) return TR_MATH_DOMAIN_ERROR;
    tr_num_rational w[TR_FOURIER_MAX_DIMS];
    for (uint64_t i = 0; i < n; i++) {
        rc = canon(omega + i * (uint64_t)spec->ndims, (uint64_t)spec->ndims, w);
        if (rc) return rc;
        if (sorted_find(s->omega, s->sorted, s->info.nfreq, spec->ndims, w) == UINT64_MAX) return TR_MATH_DOMAIN_ERROR;
    }
    return TR_MATH_OK;
}

/* ================================================================ actions */
static int32_t check_arg(double x) {
    if (!isfinite(x)) return TR_MATH_INVALID;
    return fabs(x) >= 0x1p62 ? TR_MATH_UNSUPPORTED : TR_MATH_OK;
}
/* exp(i (c omega_f . tau + phi)) enclosed */
static tr_carrier_cinterval rotation(const tr_fourier_state *s, uint64_t f, const double *tau, double phi) {
    uint64_t acc[6] = {0, 0, 0, 0, 0, 0};
    int lost = 0, inexact = 0, any = 0;
    const int32_t d = s->info.ndims;
    for (int32_t j = 0; tau && j < d; j++) {
        uint64_t e = f * (uint64_t)d + (uint64_t)j;
        if (tau[j] == 0 || !s->kssign[e]) continue;
        int64_t M;
        int32_t E;
        decompose(tau[j], &M, &E);
        if (!s->ksexact[e]) inexact = 1;
        tr_fourier_term(acc, s->ks + 4 * e, (M < 0) != (s->kssign[e] < 0), tr_abs_u64(M), E, &lost);
        any = 1;
    }
    if (phi != 0) {
        int64_t M;
        int32_t E;
        decompose(phi, &M, &E);
        tr_fourier_term(acc, TWO_OVER_PI_K, M < 0, tr_abs_u64(M), E, &lost);
        inexact = 1; any = 1;
    }
    tr_carrier_cinterval z;
    if (!any) { z.re.lo = z.re.hi = 1.0; z.im.lo = z.im.hi = 0.0; return z; }
    tr_calc_fx sn, cs;
    tr_fourier_finish(acc, inexact || lost, &sn, &cs);
    z.re = fx_interval(cs); z.im = fx_interval(sn);
    return z;
}
int32_t tr_fourier_act(const tr_fourier_state *s, uint64_t g, uint64_t nreq, const double *tau, const double *phase,
                       const int32_t *order, const int32_t *phase_order, uint64_t nsel, const uint64_t *sel,
                       int32_t normalize, tr_carrier_cinterval *out, const tr_math_budget *budget,
                       tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    tr_math_budget dflt;
    budget = budget_of(budget, &dflt);
    if (!state_ok(s) || g >= s->info.ngroups || (normalize != 0 && normalize != 1)) return TR_MATH_INVALID;
    if (!sel) nsel = s->info.nfreq;
    if (nreq && nsel && !out) return TR_MATH_INVALID;
    const int32_t d = s->info.ndims;
    int32_t base = order_total(s);
    uint64_t degree = 0;
    for (uint64_t q = 0; q < nreq; q++) {                               /* validate every request first */
        int64_t tot = base + (phase_order ? phase_order[q] : 0);
        if (phase_order && phase_order[q] < 0) return TR_MATH_INVALID;
        for (int32_t j = 0; j < d; j++) {
            if (order && order[q * (uint64_t)d + (uint64_t)j] < 0) return TR_MATH_INVALID;
            if (order) tot += order[q * (uint64_t)d + (uint64_t)j];
            if (tau) { int32_t rc = check_arg(tau[q * (uint64_t)d + (uint64_t)j]); if (rc) return rc; }
        }
        if (phase) { int32_t rc = check_arg(phase[q]); if (rc) return rc; }
        if (tot > TR_FOURIER_MAX_ORDER) return TR_MATH_UNSUPPORTED;
        if ((uint64_t)tot > degree) degree = (uint64_t)tot;
    }
    for (uint64_t i = 0; sel && i < nsel; i++) if (sel[i] >= s->info.nfreq) return TR_MATH_INVALID;
    tr_math_need need;
    memset(&need, 0, sizeof need);
    need.need[TR_BUDGET_DEGREE] = degree;
    need.need[TR_BUDGET_OUTPUT] = tr_budget_mul(nreq, nsel);
    need.need[TR_BUDGET_TENSOR] = tr_budget_mul(tr_budget_mul(nreq, nsel), (uint64_t)d + 1);
    need.need[TR_BUDGET_BYTES] = tr_budget_mul(nsel, sizeof(tr_carrier_cinterval));     /* the coefficients once */
    int32_t rc = tr_budget_check(budget, &need, refusal);
    if (rc) return rc;
    if (normalize && tr_i128_get(s->wsum[g]) == 0) return TR_MATH_DOMAIN_ERROR;   /* no characteristic function */
    if (!nreq || !nsel) return TR_MATH_OK;
    tr_carrier_cinterval *coef = (tr_carrier_cinterval *)zalloc(nsel * sizeof *coef);
    if (!coef) return TR_MATH_MEMORY;
    for (uint64_t i = 0; i < nsel; i++) coef[i] = coef_of(s, g, sel ? sel[i] : i);
    int32_t zero[TR_FOURIER_MAX_DIMS];
    memset(zero, 0, sizeof zero);
    for (uint64_t q = 0; q < nreq; q++) {
        const double *t = tau ? tau + q * (uint64_t)d : NULL;
        double phi = phase ? phase[q] : 0.0;
        const int32_t *a = order ? order + q * (uint64_t)d : zero;
        int32_t tot = phase_order ? phase_order[q] : 0, at = 0;
        for (int32_t j = 0; j < d; j++) at += a[j];
        tot += at;
        for (uint64_t i = 0; i < nsel; i++) {
            uint64_t f = sel ? sel[i] : i;
            tr_carrier_cinterval z = cmul(coef[i], rotation(s, f, t, phi));
            if (tot) z = crot(z, tot);
            if (at) z = cscale(z, order_factor(s, f, a));
            if (normalize) {
                z.re = clean(tr_interval_div(z.re, s->wenc[g]));
                z.im = clean(tr_interval_div(z.im, s->wenc[g]));
            }
            out[q * nsel + i] = z;
        }
    }
    free(coef);
    return TR_MATH_OK;
}

tr_fourier_state *tr_fourier_derive(const tr_fourier_state *s, const int32_t *alpha, int32_t beta,
                                    const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    tr_math_budget dflt;
    budget = budget_of(budget, &dflt);
    if (!state_ok(s) || !alpha || beta < 0) { seterr(error, TR_MATH_INVALID); return NULL; }
    int64_t tot = (int64_t)order_total(s) + beta, at = 0;
    for (int32_t j = 0; j < s->info.ndims; j++) {
        if (alpha[j] < 0) { seterr(error, TR_MATH_INVALID); return NULL; }
        tot += alpha[j]; at += (int64_t)s->info.order[j] + alpha[j];
    }
    if (tot > TR_FOURIER_MAX_ORDER) { seterr(error, TR_MATH_UNSUPPORTED); return NULL; }
    const uint64_t G = s->info.ngroups, m = s->info.nfreq;
    tr_math_need need;
    memset(&need, 0, sizeof need);
    need.need[TR_BUDGET_DEGREE] = (uint64_t)tot;
    need.need[TR_BUDGET_STATES] = tr_budget_mul(G, m);
    need.need[TR_BUDGET_BYTES] = state_bytes(G, m, s->info.ndims, s->keybytes);
    int32_t rc = tr_budget_check(budget, &need, refusal);
    if (rc) { seterr(error, rc); return NULL; }
    tr_fourier_state *o = state_copy(s);
    if (!o) { seterr(error, TR_MATH_MEMORY); return NULL; }
    for (int32_t j = 0; j < s->info.ndims; j++) o->info.order[j] += alpha[j];
    o->info.phase_order += beta;
    o->info.unit_power = s->info.unit == TR_FOURIER_RADIAN ? 0 : (int32_t)at;
    seterr(error, TR_MATH_OK);
    return o;
}

/* ================================================================ the Boolean case */
/* bit pattern S of frequency f when every component is 0 or 1 (pi units) / 0 or 1/2 (turns), else UINT64_MAX */
static uint64_t walsh_set(const tr_fourier_state *s, uint64_t f) {
    uint64_t S = 0;
    const int32_t d = s->info.ndims;
    for (int32_t j = 0; j < d; j++) {
        tr_num_rational w = s->omega[f * (uint64_t)d + (uint64_t)j];
        if (w.numerator == 0) continue;
        int one = s->info.unit == TR_FOURIER_PI ? (w.numerator == 1 && w.denominator == 1)
                                                : (w.numerator == 1 && w.denominator == 2);
        if (!one) return UINT64_MAX;
        S |= (uint64_t)1 << j;
    }
    return S;
}
int32_t tr_fourier_walsh_support(const tr_fourier_state *s, int32_t *k, uint64_t *present) {
    if (!state_ok(s) || !k || !present) return TR_MATH_INVALID;
    if ((s->info.unit != TR_FOURIER_PI && s->info.unit != TR_FOURIER_TURN) || s->info.ndims > TR_FOURIER_WALSH_MAX_K)
        return TR_MATH_DOMAIN_ERROR;
    uint64_t n = 0;
    for (uint64_t f = 0; f < s->info.nfreq; f++) n += walsh_set(s, f) != UINT64_MAX;   /* distinct: a set */
    *k = s->info.ndims; *present = n;
    return TR_MATH_OK;
}
int32_t tr_fourier_walsh_inverse(const tr_fourier_state *s, uint64_t g, tr_spec_dyadic *out, tr_spec_cost *cost,
                                 const tr_math_budget *budget, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    tr_math_budget dflt;
    budget = budget_of(budget, &dflt);
    if (!state_ok(s) || g >= s->info.ngroups || !out) return TR_MATH_INVALID;
    int32_t k;
    uint64_t present;
    int32_t rc = tr_fourier_walsh_support(s, &k, &present);
    if (rc) return rc;
    const uint64_t n = (uint64_t)1 << k;
    tr_math_need need;
    memset(&need, 0, sizeof need);
    need.need[TR_BUDGET_VARIABLES] = (uint64_t)k;
    need.need[TR_BUDGET_TENSOR] = n;
    need.need[TR_BUDGET_ITERATIONS] = (uint64_t)k;
    need.need[TR_BUDGET_BYTES] = tr_budget_mul(n, 40);
    rc = tr_budget_check(budget, &need, refusal);
    if (rc) return rc;
    if (present != n || order_total(s)) return TR_MATH_DOMAIN_ERROR;   /* missing spectral support is not zero */
    for (int32_t j = 0; j < k; j++)                                    /* the domain {0, 1}^k */
        if (s->colscale[j] != 0 || (s->colmin[j] <= s->colmax[j] && (s->colmin[j] < 0 || s->colmax[j] > 1)))
            return TR_MATH_DOMAIN_ERROR;
    if (s->info.scale < 0) return TR_MATH_DOMAIN_ERROR;
    tr_math_i128 *vals = (tr_math_i128 *)zalloc(n * 16);
    if (!vals) return TR_MATH_MEMORY;
    for (uint64_t f = 0; f < s->info.nfreq && !rc; f++) {
        uint64_t S = walsh_set(s, f);
        if (S == UINT64_MAX) continue;
        const tr_math_i128 *b = s->sums + (g * s->info.nfreq + f) * 4;
        __int128 lo = tr_i128_get(b[0]), hi = tr_i128_get(b[1]), il = tr_i128_get(b[2]), ih = tr_i128_get(b[3]);
        __int128 unit = (__int128)1 << s->info.scale;
        if (lo != hi || il != 0 || ih != 0 || lo % unit) { rc = TR_MATH_DOMAIN_ERROR; break; }   /* not exact */
        vals[S] = tr_i128_set(lo / unit);
    }
    if (!rc) rc = tr_spec_walsh_inverse(k, vals, out, budget, cost, refusal);
    free(vals);
    return rc;
}
