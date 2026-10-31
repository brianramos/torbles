/* SPEC-002: exact Walsh spectra, Parseval identities, parity correlations, truncation with its exact error and the
 * FIELD-001 cross-check of constant parities; see math_spec.h.
 * SPDX-License-Identifier: Apache-2.0 */
#include "math_spec.h"
#include "mathematics.h"
#include "math_domain.h"
#include "math_field.h"
#include "math_handles.h"
#include <stdlib.h>
#include <string.h>

TR_MATH_ABI(tr_spec_dyadic, TR_SPEC_ABI);      /* the ctypes boundary of SPEC-002 (math_handles.h) */
TR_MATH_ABI(tr_spec_cost, TR_SPEC_ABI);
TR_MATH_ABI(tr_spec_energy, TR_SPEC_ABI);
TR_MATH_ABI(tr_spec_parity, TR_SPEC_ABI);
TR_MATH_ABI(tr_spec_truncation, TR_SPEC_ABI);
TR_MATH_ABI(tr_spec_relation, TR_SPEC_ABI);
TR_MATH_ABI(tr_spec_parities, TR_SPEC_ABI);

_Static_assert(sizeof(tr_spec_dyadic) == 24, "tr_spec_dyadic ABI");
_Static_assert(sizeof(tr_spec_cost) == 40, "tr_spec_cost ABI");
_Static_assert(sizeof(tr_spec_energy) == 72, "tr_spec_energy ABI");
_Static_assert(sizeof(tr_spec_parity) == 96, "tr_spec_parity ABI");
_Static_assert(sizeof(tr_spec_truncation) == 176, "tr_spec_truncation ABI");
_Static_assert(sizeof(tr_spec_relation) == 16, "tr_spec_relation ABI");
_Static_assert(sizeof(tr_spec_parities) == 48, "tr_spec_parities ABI");
_Static_assert(sizeof(tr_spec_relation) == sizeof(tr_field_relation), "the FIELD-001 relation layout");

static void *zalloc(uint64_t bytes) {
    if (bytes > (uint64_t)SIZE_MAX) return NULL;
    return calloc(1, bytes ? (size_t)bytes : 1);
}
static void zero_need(tr_math_need *n) { for (int i = 0; i < TR_BUDGET_RESOURCES; i++) n->need[i] = 0; }
static const tr_math_budget *budget_or_default(const tr_math_budget *b, tr_math_budget *dflt) {
    if (b) return b;
    tr_budget_default(dflt);
    return dflt;
}
static int popc(uint64_t x) { return __builtin_popcountll(x); }
static unsigned __int128 mag(__int128 v) { return v < 0 ? (unsigned __int128)0 - (unsigned __int128)v : (unsigned __int128)v; }

/* ------------------------------------------------------------------ 256-bit unsigned arithmetic (energies) */
/* a += b; returns the carry out of the top word (an overflow past 2^256) */
static int add256(uint64_t *a, const uint64_t *b) {
    uint64_t carry = 0;
    for (int i = 0; i < 4; i++) {
        uint64_t s = a[i] + b[i], c1 = s < a[i];
        uint64_t t = s + carry, c2 = t < s;
        a[i] = t; carry = c1 | c2;
    }
    return (int)carry;
}
/* out = m^2 for m <= 2^127 (fits 256 bits) */
static void square256(unsigned __int128 m, uint64_t *out) {
    uint64_t l = (uint64_t)m, h = (uint64_t)(m >> 64);
    unsigned __int128 ll = (unsigned __int128)l * l, lh = (unsigned __int128)l * h, hh = (unsigned __int128)h * h;
    unsigned __int128 lh2 = lh << 1;                          /* 2 l h < 2^128: h <= 2^63 */
    uint64_t a[4] = {(uint64_t)ll, (uint64_t)(ll >> 64), 0, 0};
    uint64_t b[4] = {0, (uint64_t)lh2, (uint64_t)(lh2 >> 64), 0};
    uint64_t c[4] = {0, 0, (uint64_t)hh, (uint64_t)(hh >> 64)};
    add256(a, b);
    add256(a, c);                                             /* no carry: m^2 <= 2^254 */
    memcpy(out, a, 32);
}
static int add_square(uint64_t *acc, __int128 v) {
    uint64_t sq[4];
    square256(mag(v), sq);
    return add256(acc, sq);
}
/* out = a << s (s < 64); 1 when bits were shifted out */
static int shl256(const uint64_t *a, int s, uint64_t *out) {
    if (!s) { memcpy(out, a, 32); return 0; }
    int lost = (a[3] >> (64 - s)) != 0;
    for (int i = 3; i > 0; i--) out[i] = (a[i] << s) | (a[i - 1] >> (64 - s));
    out[0] = a[0] << s;
    return lost;
}
/* a / 2^k reduced: num = a >> t, den = k - t with t = min(trailing zeros, k) (a = 0: 0 / 2^0) */
static int32_t reduce256(const uint64_t *a, int32_t k, uint64_t *num) {
    int zero = !(a[0] | a[1] | a[2] | a[3]);
    int32_t t = 0;
    uint64_t x[4];
    memcpy(x, a, 32);
    while (!zero && t < k && !(x[0] & 1)) {
        for (int i = 0; i < 3; i++) x[i] = (x[i] >> 1) | (x[i + 1] << 63);
        x[3] >>= 1;
        t++;
    }
    memcpy(num, x, 32);
    return zero ? 0 : k - t;
}

/* ------------------------------------------------------------------ transforms */
int32_t tr_spec_need(int32_t k, tr_math_need *need) { return tr_spec_base_need(k, need); }

/* the fast transform of in into the scratch v: TR_MATH_OK or TR_MATH_OVERFLOW */
static int32_t walsh_run(int32_t k, const tr_math_i128 *in, tr_math_i128 *v) {
    const uint64_t n = (uint64_t)1 << k;
    memcpy(v, in, (size_t)(n * 16));
    for (int32_t j = 0; j < k; j++) {
        const uint64_t half = (uint64_t)1 << j;
        for (uint64_t base = 0; base < n; base += 2 * half)
            for (uint64_t lo = base; lo < base + half; lo++)
                if (!tr_spec_butterfly(v, lo, lo + half)) return TR_MATH_OVERFLOW;
    }
    return TR_MATH_OK;
}
/* the checks and the budget of a transform; a scratch of 2^k values */
static int32_t walsh_start(int32_t k, const void *in, const void *out, const tr_math_budget *budget,
                           tr_math_refusal *refusal, tr_math_i128 **scratch) {
    tr_budget_none(refusal);
    tr_math_need need;
    if (tr_spec_base_need(k, &need) || !in || !out) return TR_MATH_INVALID;
    tr_math_budget dflt;
    int32_t rc = tr_budget_check(budget_or_default(budget, &dflt), &need, refusal);
    if (rc) return rc;
    if (k > TR_SPEC_MAX_K) return TR_MATH_UNSUPPORTED;
    *scratch = zalloc(((uint64_t)1 << k) * 16);
    return *scratch ? TR_MATH_OK : TR_MATH_MEMORY;
}

int32_t tr_spec_walsh(int32_t k, const tr_math_i128 *in, tr_math_i128 *out, const tr_math_budget *budget,
                      tr_spec_cost *cost, tr_math_refusal *refusal) {
    tr_math_i128 *v = NULL;
    int32_t rc = walsh_start(k, in, out, budget, refusal, &v);
    if (rc) return rc;
    rc = walsh_run(k, in, v);
    if (!rc) {
        memcpy(out, v, (size_t)(((uint64_t)1 << k) * 16));
        if (cost) tr_spec_cost_of(k, TR_SPEC_FORWARD, ((uint64_t)1 << k) * 16, cost);
    }
    free(v);
    return rc;
}

int32_t tr_spec_walsh_inverse(int32_t k, const tr_math_i128 *in, tr_spec_dyadic *out, const tr_math_budget *budget,
                              tr_spec_cost *cost, tr_math_refusal *refusal) {
    tr_math_i128 *v = NULL;
    int32_t rc = walsh_start(k, in, out, budget, refusal, &v);
    if (rc) return rc;
    rc = walsh_run(k, in, v);                                 /* H H = 2^k I: the inverse is 2^-k H */
    if (!rc) {
        const uint64_t n = (uint64_t)1 << k;
        for (uint64_t i = 0; i < n; i++) out[i] = tr_spec_dyadic_of(tr_i128_get(v[i]), k);
        if (cost) tr_spec_cost_of(k, TR_SPEC_INVERSE, n * 16, cost);
    }
    free(v);
    return rc;
}

int32_t tr_spec_parseval(int32_t k, const tr_math_i128 *values, const tr_math_i128 *spectrum, tr_spec_energy *out) {
    if (k < 0 || !values || !spectrum || !out) return TR_MATH_INVALID;
    if (k > TR_SPEC_MAX_K) return TR_MATH_UNSUPPORTED;
    const uint64_t n = (uint64_t)1 << k;
    uint64_t cells[4] = {0}, spec[4] = {0}, shifted[4];
    for (uint64_t i = 0; i < n; i++)
        if (add_square(cells, tr_i128_get(values[i])) || add_square(spec, tr_i128_get(spectrum[i])))
            return TR_MATH_OVERFLOW;
    int lost = shl256(cells, k, shifted);
    memset(out, 0, sizeof *out);
    memcpy(out->cells, cells, 32);
    memcpy(out->spectrum, spec, 32);
    out->k = k;
    out->holds = !lost && !memcmp(shifted, spec, 32);
    return TR_MATH_OK;
}

/* ------------------------------------------------------------------ parity correlations */
static unsigned __int128 gcd128(unsigned __int128 a, unsigned __int128 b) {
    while (b) { unsigned __int128 t = a % b; a = b; b = t; }
    return a;
}
/* the parity record of S from W(S) and N: TR_MATH_OK, TR_MATH_OVERFLOW (N +- W past 128 bits) or TR_MATH_INVALID
 * (N and W of different parity: not the spectrum of integer weights) */
static int32_t parity_of(uint64_t S, __int128 W, __int128 N, tr_spec_parity *r) {
    __int128 s, d;
    if (!tr_spec_add128(N, W, &s) || !tr_spec_sub128(N, W, &d)) return TR_MATH_OVERFLOW;
    if (s & 1) return TR_MATH_INVALID;
    memset(r, 0, sizeof *r);
    r->vars = S; r->order = popc(S);
    r->coefficient = tr_i128_set(W);
    r->even = tr_i128_set(s / 2); r->odd = tr_i128_set(d / 2);   /* exact: even sums */
    if (N == 0) { r->corr_num = tr_i128_set(0); r->corr_den = tr_i128_set(1); return TR_MATH_OK; }
    unsigned __int128 g = gcd128(mag(W), mag(N)), lim = (unsigned __int128)1 << 127;
    if (mag(W) / g >= lim || mag(N) / g >= lim) return TR_MATH_OVERFLOW;    /* only for a value -2^127 */
    __int128 num = (__int128)(mag(W) / g), den = (__int128)(mag(N) / g);
    if ((W < 0) != (N < 0)) num = -num;
    r->corr_num = tr_i128_set(num); r->corr_den = tr_i128_set(den);
    return TR_MATH_OK;
}
static int32_t sets_valid(int32_t k, const uint64_t *sets, int32_t n) {
    if (n < 0 || (n && !sets)) return TR_MATH_INVALID;
    for (int32_t j = 0; j < n; j++) if (k < 64 && (sets[j] >> k)) return TR_MATH_INVALID;
    return TR_MATH_OK;
}

int32_t tr_spec_correlations(int32_t k, const tr_math_i128 *spectrum, const uint64_t *sets, int32_t n,
                             tr_spec_parity *out) {
    if (k < 0 || !spectrum || (n && !out)) return TR_MATH_INVALID;
    if (k > TR_SPEC_MAX_K) return TR_MATH_UNSUPPORTED;
    int32_t rc = sets_valid(k, sets, n);
    if (rc) return rc;
    const __int128 N = tr_i128_get(spectrum[0]);
    tr_spec_parity tmp;
    for (int32_t j = 0; j < n; j++) {                         /* every record checked before any is written */
        rc = parity_of(sets[j], tr_i128_get(spectrum[sets[j]]), N, &tmp);
        if (rc) return rc;
    }
    for (int32_t j = 0; j < n; j++) parity_of(sets[j], tr_i128_get(spectrum[sets[j]]), N, &out[j]);
    return TR_MATH_OK;
}

int32_t tr_spec_coefficients_need(int32_t k, const uint64_t *sets, int32_t n, tr_math_need *need) {
    if (k < 0 || !need) return TR_MATH_INVALID;
    int32_t rc = sets_valid(k, sets, n);
    if (rc) return rc;
    zero_need(need);
    need->need[TR_BUDGET_VARIABLES] = (uint64_t)k;
    need->need[TR_BUDGET_TENSOR] = tr_budget_mul((uint64_t)n, tr_budget_power2((uint32_t)k));
    need->need[TR_BUDGET_OUTPUT] = (uint64_t)n;
    need->need[TR_BUDGET_BYTES] = tr_budget_mul((uint64_t)n, sizeof(tr_spec_parity));
    return TR_MATH_OK;
}

int32_t tr_spec_coefficients(int32_t k, const tr_math_i128 *values, const uint64_t *sets, int32_t n,
                             const tr_math_budget *budget, tr_spec_parity *out, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    tr_math_need need;
    int32_t rc = tr_spec_coefficients_need(k, sets, n, &need);
    if (rc) return rc;
    if (!values || (n && !out)) return TR_MATH_INVALID;
    tr_math_budget dflt;
    rc = tr_budget_check(budget_or_default(budget, &dflt), &need, refusal);
    if (rc) return rc;
    if (k > TR_SPEC_MAX_K) return TR_MATH_UNSUPPORTED;
    const uint64_t size = (uint64_t)1 << k;
    __int128 N = 0;
    for (uint64_t x = 0; x < size; x++) if (!tr_spec_add128(N, tr_i128_get(values[x]), &N)) return TR_MATH_OVERFLOW;
    tr_spec_parity *tmp = zalloc((uint64_t)n * sizeof *tmp);
    if (!tmp) return TR_MATH_MEMORY;
    for (int32_t j = 0; j < n && !rc; j++) {
        const uint64_t S = sets[j];
        __int128 W = 0;
        for (uint64_t x = 0; x < size && !rc; x++) {          /* W(S) = sum_x v(x) chi_S(x), checked */
            __int128 v = tr_i128_get(values[x]);
            if (!((popc(S & x) & 1) ? tr_spec_sub128(W, v, &W) : tr_spec_add128(W, v, &W))) rc = TR_MATH_OVERFLOW;
        }
        if (!rc) rc = parity_of(S, W, N, &tmp[j]);
    }
    if (!rc && n) memcpy(out, tmp, (size_t)n * sizeof *tmp);
    free(tmp);
    return rc;
}

/* ------------------------------------------------------------------ truncation */
typedef struct { uint64_t hi, lo, set; } order_entry;        /* |W(S)| as two words, S */
static int order_cmp(const void *a, const void *b) {         /* |W| decreasing, then S increasing: a total order */
    const order_entry *x = (const order_entry *)a, *y = (const order_entry *)b;
    if (x->hi != y->hi) return x->hi < y->hi ? 1 : -1;
    if (x->lo != y->lo) return x->lo < y->lo ? 1 : -1;
    return x->set < y->set ? -1 : x->set > y->set;
}

int32_t tr_spec_truncate(int32_t k, const tr_math_i128 *spectrum, uint64_t m, const tr_math_budget *budget,
                         uint64_t *sets, tr_math_i128 *coef, tr_spec_truncation *out, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (k < 0 || !spectrum || !out || (m && (!sets || !coef))) return TR_MATH_INVALID;
    tr_math_need need;
    zero_need(&need);
    const uint64_t size = tr_budget_power2((uint32_t)k);
    need.need[TR_BUDGET_VARIABLES] = (uint64_t)k;
    need.need[TR_BUDGET_TENSOR] = size;
    need.need[TR_BUDGET_OUTPUT] = m;
    need.need[TR_BUDGET_BYTES] = tr_budget_mul(size, sizeof(order_entry));
    tr_math_budget dflt;
    int32_t rc = tr_budget_check(budget_or_default(budget, &dflt), &need, refusal);
    if (rc) return rc;
    if (k > TR_SPEC_MAX_K) return TR_MATH_UNSUPPORTED;
    if (m > size) return TR_MATH_INVALID;
    order_entry *ord = zalloc(size * sizeof *ord);
    if (!ord) return TR_MATH_MEMORY;
    tr_spec_truncation r;
    memset(&r, 0, sizeof r);
    for (uint64_t S = 0; S < size; S++) {
        __int128 W = tr_i128_get(spectrum[S]);
        unsigned __int128 a = mag(W);
        ord[S].hi = (uint64_t)(a >> 64); ord[S].lo = (uint64_t)a; ord[S].set = S;
        r.nonzero += W != 0;
        if (add_square(r.total, W)) { free(ord); return TR_MATH_OVERFLOW; }
    }
    qsort(ord, (size_t)size, sizeof *ord, order_cmp);
    unsigned __int128 l1 = 0;
    for (uint64_t i = m; i < size; i++) {                     /* the dropped coefficients */
        __int128 W = tr_i128_get(spectrum[ord[i].set]);
        if (!W) continue;
        unsigned __int128 a = mag(W);
        r.dropped++;
        if (add_square(r.residual, W) || l1 + a < l1 || l1 + a > ((unsigned __int128)1 << 127) - 1) {
            free(ord); return TR_MATH_OVERFLOW;
        }
        l1 += a;
    }
    r.states = size; r.kept = m; r.k = k; r.exact = r.dropped == 0;
    r.residual_l1 = tr_i128_set((__int128)l1);
    r.l2_den_log2 = reduce256(r.residual, k, r.l2_num);
    tr_spec_dyadic bound = tr_spec_dyadic_of((__int128)l1, k);
    r.bound_num = bound.num; r.bound_den_log2 = (int32_t)bound.den_log2;
    for (uint64_t i = 0; i < m; i++) { sets[i] = ord[i].set; coef[i] = spectrum[ord[i].set]; }
    *out = r;
    free(ord);
    return TR_MATH_OK;
}

int32_t tr_spec_evaluate(int32_t k, const uint64_t *sets, const tr_math_i128 *coef, uint64_t m, const uint64_t *points,
                         uint64_t npoints, const tr_math_budget *budget, tr_spec_dyadic *out, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (k < 0 || (m && (!sets || !coef)) || (npoints && (!points || !out))) return TR_MATH_INVALID;
    tr_math_need need;
    zero_need(&need);
    need.need[TR_BUDGET_VARIABLES] = (uint64_t)k;
    need.need[TR_BUDGET_TENSOR] = tr_budget_mul(m, npoints);
    need.need[TR_BUDGET_OUTPUT] = npoints;
    need.need[TR_BUDGET_BYTES] = tr_budget_mul(npoints, sizeof(tr_spec_dyadic));
    tr_math_budget dflt;
    int32_t rc = tr_budget_check(budget_or_default(budget, &dflt), &need, refusal);
    if (rc) return rc;
    if (k > TR_SPEC_MAX_K) return TR_MATH_UNSUPPORTED;
    for (uint64_t j = 0; j < m; j++) if (sets[j] >> k) return TR_MATH_INVALID;
    for (uint64_t i = 0; i < npoints; i++) if (points[i] >> k) return TR_MATH_INVALID;
    tr_spec_dyadic *tmp = zalloc(npoints * sizeof *tmp);
    if (!tmp) return TR_MATH_MEMORY;
    for (uint64_t i = 0; i < npoints && !rc; i++) {
        __int128 s = 0;
        for (uint64_t j = 0; j < m && !rc; j++) {
            __int128 c = tr_i128_get(coef[j]);
            if (!((popc(sets[j] & points[i]) & 1) ? tr_spec_sub128(s, c, &s) : tr_spec_add128(s, c, &s)))
                rc = TR_MATH_OVERFLOW;
        }
        tmp[i] = tr_spec_dyadic_of(s, k);
    }
    if (!rc && npoints) memcpy(out, tmp, (size_t)(npoints * sizeof *tmp));
    free(tmp);
    return rc;
}

/* ------------------------------------------------------------------ constant parities and FIELD-001 */
int32_t tr_spec_constant_parities(int32_t k, const tr_math_i128 *values, const tr_math_budget *budget,
                                  tr_spec_relation *relations, uint64_t capacity, tr_spec_relation *basis_out,
                                  int32_t basis_capacity, tr_spec_parities *out, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (k < 0 || !values || !out || (capacity && !relations) || basis_capacity < 0 || (basis_capacity && !basis_out))
        return TR_MATH_INVALID;
    tr_math_need need;
    zero_need(&need);
    const uint64_t size = tr_budget_power2((uint32_t)k);
    need.need[TR_BUDGET_VARIABLES] = (uint64_t)k;
    need.need[TR_BUDGET_TENSOR] = size;
    need.need[TR_BUDGET_ITERATIONS] = (uint64_t)k;
    need.need[TR_BUDGET_OUTPUT] = size;
    need.need[TR_BUDGET_BYTES] = tr_budget_mul(size, 17);
    tr_math_budget dflt;
    int32_t rc = tr_budget_check(budget_or_default(budget, &dflt), &need, refusal);
    if (rc) return rc;
    if (k > TR_SPEC_MAX_K) return TR_MATH_UNSUPPORTED;
    if (basis_capacity < k + 1) return TR_MATH_INVALID;
    __int128 N = 0;
    uint64_t support = 0;
    for (uint64_t x = 0; x < size; x++) {
        __int128 v = tr_i128_get(values[x]);
        if (v < 0) return TR_MATH_DOMAIN_ERROR;               /* parity constancy is a statement about weights */
        if (!tr_spec_add128(N, v, &N)) return TR_MATH_OVERFLOW;
        support += v != 0;
    }
    if (N == 0) return TR_MATH_DOMAIN_ERROR;                  /* no point: every relation would hold vacuously */
    tr_math_i128 *W = zalloc(size * 16);
    int8_t *mark = zalloc(size);
    if (!W || !mark) { free(W); free(mark); return TR_MATH_MEMORY; }
    rc = walsh_run(k, values, W);                             /* |W(S)| <= N: no overflow */
    if (rc) { free(W); free(mark); return rc; }
    uint64_t constant = 0;
    for (uint64_t S = 0; S < size; S++) {
        __int128 w = tr_i128_get(W[S]);
        if (w == N || w == -N) constant++;
    }
    /* FIELD-001: the occupied cells as affine vectors [1, x] (bit 0 the constant, bit i + 1 condition i) */
    uint64_t basis[64] = {0};
    for (uint64_t x = 0; x < size; x++)
        if (tr_i128_get(values[x]) != 0) tr_field_basis_insert(basis, 1 | (x << 1));
    int32_t rank = 0;
    for (int p = 0; p < 64; p++) rank += basis[p] != 0;
    tr_field_relation rel[64];
    int32_t nb = tr_field_basis_relations(basis, k + 1, 1, rel);
    /* the span of the relations: 2^nb distinct (vars, constant), marked; the constant 0 = 1 cannot occur (N > 0) */
    int agree = constant == ((uint64_t)1 << nb);
    memset(mark, 0xff, (size_t)size);                         /* -1: not in the span */
    uint64_t vars = 0;
    int32_t c = 0;
    for (uint64_t g = 0; agree && g < ((uint64_t)1 << nb); g++) {   /* Gray code: one relation added per step */
        if (g) {
            int b = __builtin_ctzll(g);
            vars ^= rel[b].vars; c ^= rel[b].constant;
        }
        if (mark[vars] != -1 && mark[vars] != c) agree = 0;
        mark[vars] = (int8_t)c;
    }
    for (uint64_t S = 0; agree && S < size; S++) {
        __int128 w = tr_i128_get(W[S]);
        int want = w == N ? 0 : w == -N ? 1 : -1;
        if (mark[S] != want) agree = 0;
    }
    if (relations && capacity < constant) { free(W); free(mark); return TR_MATH_INVALID; }
    if (relations) {
        uint64_t at = 0;
        for (uint64_t S = 0; S < size; S++) {
            __int128 w = tr_i128_get(W[S]);
            if (w != N && w != -N) continue;
            relations[at].vars = S; relations[at].constant = w == N ? 0 : 1; relations[at].reserved = 0;
            at++;
        }
    }
    for (int32_t i = 0; i < nb; i++) { basis_out[i].vars = rel[i].vars; basis_out[i].constant = rel[i].constant; basis_out[i].reserved = 0; }
    memset(out, 0, sizeof *out);
    out->total = tr_i128_set(N); out->support = support; out->constant = constant;
    out->k = k; out->rank = rank; out->nbasis = nb; out->agree = agree;
    free(W); free(mark);
    return TR_MATH_OK;
}
