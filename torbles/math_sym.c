/* SYM-001 / SYM-002: closed-basis transformation actions and certified data symmetries (every contract: math_sym.h).
 * SPDX-License-Identifier: Apache-2.0 */
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "math_sym.h"
#include "math_handles.h"
#include "mathematics.h"
#include "math_domain.h"

TR_MATH_ABI(tr_sym_geometry, TR_SYM_ABI);
TR_MATH_ABI(tr_sym_transform, TR_SYM_ABI);
TR_MATH_ABI(tr_sym_closure, TR_SYM_ABI);
TR_MATH_ABI(tr_sym_map_info, TR_SYM_ABI);
TR_MATH_ABI(tr_sym_basis_info, TR_SYM_ABI);
TR_MATH_ABI(tr_sym_action_info, TR_SYM_ABI);
TR_MATH_ABI(tr_sym_action_view, TR_SYM_ABI);
TR_MATH_ABI(tr_sym_state_info, TR_SYM_ABI);
TR_MATH_ABI(tr_sym_query_view, TR_SYM_ABI);
TR_MATH_ABI(tr_sym_request, TR_SYM_ABI);
TR_MATH_ABI(tr_sym_wide, TR_SYM_ABI);
TR_MATH_ABI(tr_sym_result, TR_SYM_ABI);
TR_MATH_ABI(tr_sym_gen, TR_SYM_ABI);
TR_MATH_ABI(tr_sym_group_view, TR_SYM_ABI);
TR_MATH_ABI(tr_sym_group_info, TR_SYM_ABI);
TR_MATH_ABI(tr_sym_support_view, TR_SYM_ABI);
TR_MATH_ABI(tr_sym_support_info, TR_SYM_ABI);
TR_MATH_ABI(tr_sym_evidence, TR_SYM_ABI);
TR_MATH_ABI(tr_sym_cert_info, TR_SYM_ABI);
TR_MATH_ABI(tr_sym_orbits_info, TR_SYM_ABI);
TR_MATH_ABI(tr_sym_orbits_view, TR_SYM_ABI);
TR_MATH_ABI(tr_sym_poly_query, TR_SYM_ABI);
TR_MATH_ABI(tr_sym_poly_answer, TR_SYM_ABI);

_Static_assert(sizeof(tr_sym_geometry) == 536, "tr_sym_geometry");
_Static_assert(sizeof(tr_sym_transform) == 904, "tr_sym_transform");
_Static_assert(sizeof(tr_sym_closure) == 48, "tr_sym_closure");
_Static_assert(sizeof(tr_sym_map_info) == 56, "tr_sym_map_info");
_Static_assert(sizeof(tr_sym_basis_info) == 40, "tr_sym_basis_info");
_Static_assert(sizeof(tr_sym_action_info) == 64, "tr_sym_action_info");
_Static_assert(sizeof(tr_sym_action_view) == 56, "tr_sym_action_view");
_Static_assert(sizeof(tr_sym_state_info) == 80, "tr_sym_state_info");
_Static_assert(sizeof(tr_sym_query_view) == 48, "tr_sym_query_view");
_Static_assert(sizeof(tr_sym_request) == 24, "tr_sym_request");
_Static_assert(sizeof(tr_sym_result) == 160, "tr_sym_result");
_Static_assert(sizeof(tr_sym_gen) == 24, "tr_sym_gen");
_Static_assert(sizeof(tr_sym_group_view) == 40, "tr_sym_group_view");
_Static_assert(sizeof(tr_sym_support_view) == 40, "tr_sym_support_view");
_Static_assert(sizeof(tr_sym_support_info) == 56, "tr_sym_support_info");
_Static_assert(sizeof(tr_sym_evidence) == 80, "tr_sym_evidence");
_Static_assert(sizeof(tr_sym_cert_info) == 72, "tr_sym_cert_info");
_Static_assert(sizeof(tr_sym_orbits_info) == 56, "tr_sym_orbits_info");
_Static_assert(sizeof(tr_sym_orbits_view) == 64, "tr_sym_orbits_view");
_Static_assert(sizeof(tr_sym_poly_query) == 40, "tr_sym_poly_query");
_Static_assert(sizeof(tr_sym_poly_answer) == 96, "tr_sym_poly_answer");

#define BASIS_MAGIC 0x53594d42u
#define MAP_MAGIC 0x53594d4du
#define ACTION_MAGIC 0x53594d41u
#define STATE_MAGIC 0x53594d53u
#define GROUP_MAGIC 0x53594d47u
#define SUPPORT_MAGIC 0x53594d55u
#define CERT_MAGIC 0x53594d43u
#define ORBITS_MAGIC 0x53594d4fu
#define NC TR_SYM_MAX_COORDS

typedef __int128 i128;
typedef unsigned __int128 u128;
#define I128_MAX ((i128)(((u128)1 << 127) - 1))

/* ------------------------------------------------------------------ small helpers */
static void set_error(int32_t *e, int32_t v) { if (e) *e = v; }
static void *zalloc(uint64_t bytes) {
    if (bytes > (uint64_t)SIZE_MAX / 2) return NULL;
    return calloc(1, bytes ? (size_t)bytes : 1);
}
static const tr_math_budget *budget_or(const tr_math_budget *b, tr_math_budget *d) {
    if (b) return b;
    tr_budget_default(d);
    return d;
}
static void need_zero(tr_math_need *n) { memset(n, 0, sizeof *n); }
static int32_t check_need(const tr_math_budget *budget, const tr_math_need *need, tr_math_refusal *refusal) {
    tr_math_budget dflt;
    return tr_budget_check(budget_or(budget, &dflt), need, refusal);
}
static u128 mag128(i128 x) { return x < 0 ? (u128)0 - (u128)x : (u128)x; }
static u128 gcd128(u128 a, u128 b) {
    while (b) { u128 t = a % b; a = b; b = t; }
    return a;
}
static int add128(i128 a, i128 b, i128 *r) { return !__builtin_add_overflow(a, b, r); }
static int mul128(i128 a, i128 b, i128 *r) { return !__builtin_mul_overflow(a, b, r); }
/* the int128 values used as exact numbers stay within |x| <= 2^127 - 1 (negation never overflows) */
static int in_range(i128 x) { return x != -I128_MAX - 1; }
static i128 pow10_128(int k) { i128 r = 1; while (k-- > 0) r *= 10; return r; }
static int geo_equal(const tr_sym_geometry *a, const tr_sym_geometry *b) {
    if (a->n != b->n || a->ncols != b->ncols || a->digest[0] != b->digest[0] || a->digest[1] != b->digest[1]) return 0;
    for (int32_t i = 0; i < a->n; i++)
        if (a->col[i] != b->col[i] || a->kind[i] != b->kind[i] || a->unit[i] != b->unit[i] ||
            a->scale[i] != b->scale[i]) return 0;
    return 1;
}

/* exact rationals with int128 parts (d > 0, reduced) */
typedef struct { i128 n, d; } q128;
static int q_make(i128 n, i128 d, q128 *out) {
    if (d == 0 || !in_range(n) || !in_range(d)) return 0;
    if (d < 0) { n = -n; d = -d; }
    u128 g = gcd128(mag128(n), (u128)d);
    if (g > 1) { n /= (i128)g; d /= (i128)g; }
    out->n = n; out->d = d;
    return 1;
}
static q128 q_int(i128 v) { q128 r; r.n = v; r.d = 1; return r; }
static int q_mul(q128 a, q128 b, q128 *out) {
    i128 n, d;
    u128 g1 = gcd128(mag128(a.n), (u128)b.d), g2 = gcd128(mag128(b.n), (u128)a.d);
    if (!g1) g1 = 1;
    if (!g2) g2 = 1;
    if (!mul128(a.n / (i128)g1, b.n / (i128)g2, &n) || !mul128(a.d / (i128)g2, b.d / (i128)g1, &d)) return 0;
    return q_make(n, d, out);
}
static int q_add(q128 a, q128 b, q128 *out) {
    i128 x, y, n, d;
    if (!mul128(a.n, b.d, &x) || !mul128(b.n, a.d, &y) || !add128(x, y, &n) || !mul128(a.d, b.d, &d)) return 0;
    return q_make(n, d, out);
}
static int q_neg(q128 a, q128 *out) { out->n = -a.n; out->d = a.d; return 1; }
static int q_from(tr_num_rational r, q128 *out) { return q_make(r.numerator, r.denominator, out); }
/* value units -> logical units of a coordinate of scale s: v 10^s */
static int q_scale10(q128 v, int k, q128 *out) {
    if (k >= 0) return q_mul(v, q_int(pow10_128(k)), out);
    q128 r; r.n = 1; r.d = pow10_128(-k);
    return q_mul(v, r, out);
}
static int lcm128(i128 a, i128 b, i128 *out) {          /* a, b > 0 */
    u128 g = gcd128((u128)a, (u128)b);
    return mul128(a / (i128)g, b, out);
}

/* ------------------------------------------------------------------ certified cos / sin of rational degrees */
static tr_math_interval iv_point(double x) { tr_math_interval r; r.lo = x; r.hi = x; return r; }
static tr_math_interval iv_pi(void) {
    tr_math_interval r;
    r.lo = 0x1.921fb54442d18p+1;       /* the double below pi */
    r.hi = 0x1.921fb54442d19p+1;       /* the double above pi */
    return r;
}
/* sin and cos of x in [0, 1] (an interval): the alternating Taylor series to 2N + 1 terms plus the first omitted term
 * as the rigorous truncation bound (terms decrease monotonically for x < 1) */
static void sin_cos_small(tr_math_interval x, tr_math_interval *s, tr_math_interval *c) {
    const int N = 14;
    tr_math_interval x2 = tr_interval_mul(x, x);
    tr_math_interval ts = x, tc = iv_point(1.0), sum_s = x, sum_c = iv_point(1.0);
    for (int k = 1; k <= N; k++) {
        ts = tr_interval_div(tr_interval_mul(ts, x2), iv_point((double)((2 * k) * (2 * k + 1))));
        tc = tr_interval_div(tr_interval_mul(tc, x2), iv_point((double)((2 * k - 1) * (2 * k))));
        if (k & 1) { sum_s = tr_interval_sub(sum_s, ts); sum_c = tr_interval_sub(sum_c, tc); }
        else { sum_s = tr_interval_add(sum_s, ts); sum_c = tr_interval_add(sum_c, tc); }
    }
    int k = N + 1;
    tr_math_interval rs = tr_interval_div(tr_interval_mul(ts, x2), iv_point((double)((2 * k) * (2 * k + 1))));
    tr_math_interval rc = tr_interval_div(tr_interval_mul(tc, x2), iv_point((double)((2 * k - 1) * (2 * k))));
    tr_math_interval es, ec;
    double bs = fabs(rs.lo) > fabs(rs.hi) ? fabs(rs.lo) : fabs(rs.hi);
    double bc = fabs(rc.lo) > fabs(rc.hi) ? fabs(rc.lo) : fabs(rc.hi);
    es.lo = -bs; es.hi = bs; ec.lo = -bc; ec.hi = bc;
    *s = tr_interval_add(sum_s, es);
    *c = tr_interval_add(sum_c, ec);
}
/* (c, s) of a rotation by deg degrees: 1 with exact values in {-1, 0, 1} for a multiple of 90, 0 with certified
 * enclosures otherwise; -1 on a malformed angle */
static int rotation_degrees(tr_num_rational deg, int *ec, int *es, tr_math_interval *c, tr_math_interval *s) {
    if (deg.denominator == 0) return -1;
    i128 p = deg.numerator, q = deg.denominator;
    if (q < 0) { p = -p; q = -q; }
    i128 M = (i128)360 * q, quarter = (i128)90 * q;
    i128 r = p % M;
    if (r < 0) r += M;
    int k = (int)(r / quarter);
    i128 rem = r - (i128)k * quarter;
    static const int C4[4] = {1, 0, -1, 0}, S4[4] = {0, 1, 0, -1};
    if (rem == 0) { *ec = C4[k]; *es = S4[k]; return 1; }
    /* theta = rem / q in (0, 90): take the complement when theta > 45 so that the argument is at most pi / 4 */
    int comp = 2 * rem > quarter;
    i128 a = comp ? quarter - rem : rem;
    tr_math_interval x = tr_interval_div(tr_sym_iv_i128(a), tr_sym_iv_i128(q));
    x = tr_interval_div(tr_interval_mul(x, iv_pi()), iv_point(180.0));
    tr_math_interval sn, cs;
    sin_cos_small(x, &sn, &cs);
    tr_math_interval ct = comp ? sn : cs, st = comp ? cs : sn;   /* cos and sin of theta */
    switch (k) {                                                  /* the quadrant: angle = 90 k + theta */
    case 0: *c = ct; *s = st; break;
    case 1: *c = tr_interval_neg(st); *s = ct; break;
    case 2: *c = tr_interval_neg(ct); *s = tr_interval_neg(st); break;
    default: *c = st; *s = tr_interval_neg(ct); break;
    }
    return 0;
}

/* ------------------------------------------------------------------ geometry */
int32_t tr_sym_geometry_check(const tr_sym_geometry *g) {
    if (!g || g->n < 1 || g->ncols < 1 || g->ncols > TR_SCOPE_MAX_COLUMNS) return TR_MATH_INVALID;
    if (g->n > NC) return TR_MATH_UNSUPPORTED;
    for (int32_t i = 0; i < g->n; i++) {
        if (g->col[i] < 0 || g->col[i] >= g->ncols || (g->kind[i] != TR_SYM_NUMERIC && g->kind[i] != TR_SYM_CATEGORICAL) ||
            g->unit[i] < 0 || g->scale[i] < 0 || g->scale[i] > TR_SCOPE_MAX_SCALE) return TR_MATH_INVALID;
        for (int32_t j = 0; j < i; j++) if (g->col[j] == g->col[i]) return TR_MATH_INVALID;
    }
    return TR_MATH_OK;
}
int32_t tr_sym_geometry_make(const tr_scope *scope, const int32_t *cols, const int32_t *kinds, const int32_t *units,
                             int32_t n, tr_sym_geometry *out) {
    if (!scope || !out || n < 1 || !cols || !kinds || !units) return TR_MATH_INVALID;
    if (n > NC) return TR_MATH_UNSUPPORTED;
    tr_scope_info info;
    if (tr_scope_get_info(scope, &info)) return TR_MATH_INVALID;
    tr_sym_geometry g;
    memset(&g, 0, sizeof g);
    g.n = n; g.ncols = info.ncols;
    g.digest[0] = info.digest[0]; g.digest[1] = info.digest[1];
    for (int32_t i = 0; i < n; i++) {
        if (cols[i] < 0 || cols[i] >= info.ncols) return TR_MATH_INVALID;
        tr_scope_column_info ci;
        if (tr_scope_column_get(scope, cols[i], &ci)) return TR_MATH_INVALID;
        g.col[i] = cols[i]; g.kind[i] = kinds[i]; g.unit[i] = units[i]; g.scale[i] = ci.scale;
    }
    int32_t rc = tr_sym_geometry_check(&g);
    if (rc) return rc;
    *out = g;
    return TR_MATH_OK;
}
int32_t tr_sym_geometry_abstract(int32_t n, const int32_t *kinds, const int32_t *units, const int32_t *scales,
                                 tr_sym_geometry *out) {
    if (!out || n < 1 || !kinds || !units || !scales) return TR_MATH_INVALID;
    if (n > NC) return TR_MATH_UNSUPPORTED;
    tr_sym_geometry g;
    memset(&g, 0, sizeof g);
    g.n = n; g.ncols = n;
    for (int32_t i = 0; i < n; i++) { g.col[i] = i; g.kind[i] = kinds[i]; g.unit[i] = units[i]; g.scale[i] = scales[i]; }
    int32_t rc = tr_sym_geometry_check(&g);
    if (rc) return rc;
    *out = g;
    return TR_MATH_OK;
}
static int32_t coord_of(const tr_sym_geometry *g, int32_t col) {
    for (int32_t i = 0; i < g->n; i++) if (g->col[i] == col) return i;
    return -1;
}

/* ------------------------------------------------------------------ bases */
struct tr_sym_basis_s {
    uint32_t magic;
    int32_t n, nb, degree, constant, homogeneous;
    uint32_t coords;
    uint8_t *exps;            /* nb x n */
    u128 *skey;               /* the keys in increasing order */
    int32_t *sidx;            /* the basis index of each sorted key */
    uint64_t bytes;
};
/* a monomial's key: 4 bits per coordinate (exponents <= 8), coordinate i at bits 4i .. 4i + 3 */
static u128 mono_key(const uint8_t *e, int32_t n) {
    u128 k = 0;
    for (int32_t i = 0; i < n; i++) k |= (u128)e[i] << (4 * i);
    return k;
}
static void key_exps(u128 k, int32_t n, uint8_t *e) {
    for (int32_t i = 0; i < NC; i++) e[i] = i < n ? (uint8_t)((k >> (4 * i)) & 15) : 0;
}
static int basis_ok(const tr_sym_basis *b) { return b && b->magic == BASIS_MAGIC; }
typedef struct { u128 key; int32_t idx; } keyidx;
static int keyidx_cmp(const void *a, const void *b) {
    u128 x = ((const keyidx *)a)->key, y = ((const keyidx *)b)->key;
    return x < y ? -1 : x > y;
}
static uint64_t basis_bytes(int32_t n, int32_t nb) {
    return tr_budget_add(sizeof(tr_sym_basis), tr_budget_mul((uint64_t)nb, (uint64_t)n + 16 + 4 + sizeof(keyidx)));
}
int32_t tr_sym_basis_need(int32_t n, int32_t nb, int32_t degree, tr_math_need *need) {
    if (!need || n < 1 || nb < 1 || degree < 0) return TR_MATH_INVALID;
    need_zero(need);
    need->need[TR_BUDGET_VARIABLES] = (uint64_t)n;
    need->need[TR_BUDGET_DEGREE] = (uint64_t)degree;
    need->need[TR_BUDGET_SUPPORT] = (uint64_t)nb;
    need->need[TR_BUDGET_BYTES] = basis_bytes(n, nb);
    return TR_MATH_OK;
}
/* the lookup of a key: the basis index or -1 */
static int32_t basis_lookup(const tr_sym_basis *b, u128 key) {
    int32_t lo = 0, hi = b->nb;
    while (lo < hi) {
        int32_t mid = lo + (hi - lo) / 2;
        if (b->skey[mid] == key) return b->sidx[mid];
        if (b->skey[mid] < key) lo = mid + 1; else hi = mid;
    }
    return -1;
}
static tr_sym_basis *basis_fail(tr_sym_basis *b, int32_t *error, int32_t code) {
    if (b) { free(b->exps); free(b->skey); free(b->sidx); free(b); }
    set_error(error, code);
    return NULL;
}
static tr_sym_basis *basis_make(int32_t n, const uint8_t *exps, int32_t nb, const tr_math_budget *budget,
                                tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    if (n < 1 || nb < 1 || !exps) return basis_fail(NULL, error, TR_MATH_INVALID);
    if (n > NC || nb > TR_SYM_MAX_BASIS) return basis_fail(NULL, error, TR_MATH_UNSUPPORTED);
    int32_t top = 0, low = TR_SYM_MAX_DEGREE + 1;
    uint32_t coords = 0;
    for (int32_t j = 0; j < nb; j++) {
        int32_t d = 0;
        for (int32_t i = 0; i < n; i++) {
            int e = exps[(uint64_t)j * (uint64_t)n + (uint64_t)i];
            if (e) coords |= UINT32_C(1) << i;
            d += e;
        }
        if (d > TR_SYM_MAX_DEGREE) return basis_fail(NULL, error, TR_MATH_UNSUPPORTED);
        if (d > top) top = d;
        if (d < low) low = d;
    }
    tr_math_need need;
    tr_sym_basis_need(n, nb, top, &need);
    int32_t rc = check_need(budget, &need, refusal);
    if (rc) return basis_fail(NULL, error, rc);
    tr_sym_basis *b = zalloc(sizeof *b);
    if (!b) return basis_fail(NULL, error, TR_MATH_MEMORY);
    b->exps = zalloc((uint64_t)nb * (uint64_t)n);
    b->skey = zalloc((uint64_t)nb * sizeof(u128));
    b->sidx = zalloc((uint64_t)nb * sizeof(int32_t));
    keyidx *ki = zalloc((uint64_t)nb * sizeof(keyidx));
    if (!b->exps || !b->skey || !b->sidx || !ki) { free(ki); return basis_fail(b, error, TR_MATH_MEMORY); }
    memcpy(b->exps, exps, (size_t)nb * (size_t)n);
    b->constant = -1;
    for (int32_t j = 0; j < nb; j++) {
        ki[j].key = mono_key(exps + (uint64_t)j * (uint64_t)n, n);
        ki[j].idx = j;
        if (ki[j].key == 0) b->constant = j;
    }
    qsort(ki, (size_t)nb, sizeof *ki, keyidx_cmp);
    for (int32_t j = 0; j < nb; j++) {
        if (j && ki[j].key == ki[j - 1].key) { free(ki); return basis_fail(b, error, TR_MATH_INVALID); }   /* a repeat */
        b->skey[j] = ki[j].key; b->sidx[j] = ki[j].idx;
    }
    free(ki);
    b->n = n; b->nb = nb; b->degree = top; b->homogeneous = top == low; b->coords = coords;
    b->bytes = basis_bytes(n, nb);
    b->magic = BASIS_MAGIC;
    set_error(error, TR_MATH_OK);
    return b;
}
tr_sym_basis *tr_sym_basis_create(int32_t n, const uint8_t *exponents, int32_t nb, const tr_math_budget *budget,
                                  tr_math_refusal *refusal, int32_t *error) {
    return basis_make(n, exponents, nb, budget, refusal, error);
}
/* the exponent rows of total degree exactly d over the coordinates list[0..k) (FND-004's order: the first
 * coordinate's exponent descending, recursively) */
static void gen_degree(const int32_t *list, int32_t k, int32_t n, int32_t d, int32_t at, uint8_t *row, uint8_t *out,
                       uint64_t *count) {
    if (k == 0) {
        if (d == 0) { memcpy(out + *count * (uint64_t)n, row, (size_t)n); (*count)++; }
        return;
    }
    if (at == k - 1) {
        row[list[at]] = (uint8_t)d;
        memcpy(out + *count * (uint64_t)n, row, (size_t)n);
        (*count)++;
        row[list[at]] = 0;
        return;
    }
    for (int32_t e = d; e >= 0; e--) {
        row[list[at]] = (uint8_t)e;
        gen_degree(list, k, n, d - e, at + 1, row, out, count);
    }
    row[list[at]] = 0;
}
tr_sym_basis *tr_sym_basis_full(int32_t n, uint32_t mask, int32_t degree, int32_t homogeneous,
                                const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    if (n < 1 || degree < 0 || (homogeneous != 0 && homogeneous != 1)) return basis_fail(NULL, error, TR_MATH_INVALID);
    if (n > NC || degree > TR_SYM_MAX_DEGREE) return basis_fail(NULL, error, TR_MATH_UNSUPPORTED);
    if (n < 32 && (mask >> n)) return basis_fail(NULL, error, TR_MATH_INVALID);
    int32_t list[NC], k = 0;
    for (int32_t i = 0; i < n; i++) if ((mask >> i) & 1) list[k++] = i;
    uint64_t count = homogeneous ? tr_budget_homogeneous((uint32_t)k, (uint32_t)degree)
                                 : tr_budget_monomials((uint32_t)k, (uint32_t)degree);
    if (count == 0) return basis_fail(NULL, error, TR_MATH_INVALID);           /* no coordinate, positive degree */
    if (count > TR_SYM_MAX_BASIS) return basis_fail(NULL, error, TR_MATH_UNSUPPORTED);
    uint8_t *out = zalloc(count * (uint64_t)n), row[NC];
    if (!out) return basis_fail(NULL, error, TR_MATH_MEMORY);
    memset(row, 0, sizeof row);
    uint64_t c = 0;
    for (int32_t d = homogeneous ? degree : 0; d <= degree; d++) gen_degree(list, k, n, d, 0, row, out, &c);
    tr_sym_basis *b = basis_make(n, out, (int32_t)c, budget, refusal, error);
    free(out);
    return b;
}
int32_t tr_sym_basis_get_info(const tr_sym_basis *b, tr_sym_basis_info *out) {
    if (!basis_ok(b) || !out) return TR_MATH_INVALID;
    memset(out, 0, sizeof *out);
    out->n = b->n; out->nb = b->nb; out->degree = b->degree; out->constant = b->constant;
    out->homogeneous = b->homogeneous; out->coords = b->coords; out->bytes = b->bytes;
    return TR_MATH_OK;
}
int32_t tr_sym_basis_get_exponents(const tr_sym_basis *b, uint8_t *out, uint64_t capacity) {
    if (!basis_ok(b) || !out || capacity < (uint64_t)b->nb * (uint64_t)b->n) return TR_MATH_INVALID;
    memcpy(out, b->exps, (size_t)b->nb * (size_t)b->n);
    return TR_MATH_OK;
}
int32_t tr_sym_basis_find(const tr_sym_basis *b, const uint8_t *e) {
    if (!basis_ok(b) || !e) return TR_MATH_INVALID;
    for (int32_t i = 0; i < b->n; i++) if (e[i] > TR_SYM_MAX_DEGREE) return -1;
    return basis_lookup(b, mono_key(e, b->n));
}
int32_t tr_sym_basis_check(const tr_sym_basis *b, const tr_sym_geometry *g) {
    if (!basis_ok(b) || tr_sym_geometry_check(g) || b->n != g->n) return TR_MATH_INVALID;
    for (int32_t i = 0; i < b->n; i++)
        if (((b->coords >> i) & 1) && g->kind[i] != TR_SYM_NUMERIC) return TR_MATH_DOMAIN_ERROR;
    return TR_MATH_OK;
}
int32_t tr_sym_basis_equal(const tr_sym_basis *a, const tr_sym_basis *b) {
    if (!basis_ok(a) || !basis_ok(b)) return TR_MATH_INVALID;
    return a == b || (a->n == b->n && a->nb == b->nb && !memcmp(a->exps, b->exps, (size_t)a->nb * (size_t)a->n));
}
tr_sym_basis *tr_sym_basis_clone(const tr_sym_basis *b, int32_t *error) {
    if (!basis_ok(b)) return basis_fail(NULL, error, TR_MATH_INVALID);
    tr_math_budget unlimited;
    for (int i = 0; i < TR_BUDGET_RESOURCES; i++) unlimited.limit[i] = UINT64_MAX;
    return basis_make(b->n, b->exps, b->nb, &unlimited, NULL, error);
}
void tr_sym_basis_free(tr_sym_basis *b) {
    if (!basis_ok(b)) return;
    b->magic = 0;
    basis_fail(b, NULL, 0);
}

/* ------------------------------------------------------------------ maps */
struct tr_sym_map_s {
    uint32_t magic;
    int32_t kind, carrier, n;
    int32_t integral, permutation, invertible, coord;
    uint32_t touched, reserved;
    tr_sym_geometry geo;
    i128 a[NC][NC], b[NC], q[NC];                 /* EXACT: x'_i = (sum_j a_ij x_j + b_i) / q_i */
    tr_math_interval ai[NC][NC], bi[NC];          /* INTERVAL */
    int64_t *keys, *targets;                      /* VALUES: logical, keys increasing */
    uint64_t nvalues, bytes;
};
static int map_ok(const tr_sym_map *m) { return m && m->magic == MAP_MAGIC; }
static tr_sym_map *map_fail(tr_sym_map *m, int32_t *error, int32_t code) {
    if (m) { free(m->keys); free(m->targets); free(m); }
    set_error(error, code);
    return NULL;
}
static tr_sym_map *map_new(const tr_sym_geometry *g, int32_t kind, int32_t carrier) {
    tr_sym_map *m = zalloc(sizeof *m);
    if (!m) return NULL;
    m->kind = kind; m->carrier = carrier; m->n = g->n; m->geo = *g; m->coord = -1;
    for (int32_t i = 0; i < g->n; i++) {
        m->a[i][i] = 1; m->q[i] = 1;
        m->ai[i][i] = iv_point(1.0);
    }
    m->bytes = sizeof *m;
    return m;
}
/* the determinant of an integer matrix by Bareiss's fraction-free elimination: 1 nonzero, 0 zero, -1 undecided
 * (an intermediate past int128), decided first modulo the prime 2^61 - 1 when that is conclusive (full rank) */
static int det_nonzero(i128 a[NC][NC], int32_t n) {
    const uint64_t p = (UINT64_C(1) << 61) - 1;
    tr_gfp f;
    tr_gfp_fill(p, &f);
    uint64_t m[NC][NC];
    for (int32_t i = 0; i < n; i++)
        for (int32_t j = 0; j < n; j++) {
            u128 x = mag128(a[i][j]);
            uint64_t r = tr_gfp_reduce_u128_hd(&f, (uint64_t)(x >> 64), (uint64_t)x);
            m[i][j] = a[i][j] < 0 ? tr_gfp_neg_hd(&f, r) : r;
        }
    int full = 1;
    for (int32_t c = 0; c < n && full; c++) {
        int32_t piv = -1;
        for (int32_t r = c; r < n; r++) if (m[r][c]) { piv = r; break; }
        if (piv < 0) { full = 0; break; }
        if (piv != c) for (int32_t j = 0; j < n; j++) { uint64_t t = m[c][j]; m[c][j] = m[piv][j]; m[piv][j] = t; }
        uint64_t inv = tr_gfp_inv_hd(&f, m[c][c]);
        for (int32_t r = c + 1; r < n; r++) {
            if (!m[r][c]) continue;
            uint64_t factor = tr_gfp_mul_hd(&f, m[r][c], inv);
            for (int32_t j = c; j < n; j++) m[r][j] = tr_gfp_sub_hd(&f, m[r][j], tr_gfp_mul_hd(&f, factor, m[c][j]));
        }
    }
    if (full) return 1;
    i128 w[NC][NC], prev = 1;
    for (int32_t i = 0; i < n; i++) for (int32_t j = 0; j < n; j++) w[i][j] = a[i][j];
    for (int32_t c = 0; c < n; c++) {
        int32_t piv = -1;
        for (int32_t r = c; r < n; r++) if (w[r][c]) { piv = r; break; }
        if (piv < 0) return 0;
        if (piv != c) for (int32_t j = 0; j < n; j++) { i128 t = w[c][j]; w[c][j] = w[piv][j]; w[piv][j] = t; }
        for (int32_t r = c + 1; r < n; r++) {
            for (int32_t j = c + 1; j < n; j++) {
                i128 x, y, z;
                if (!mul128(w[c][c], w[r][j], &x) || !mul128(w[r][c], w[c][j], &y) ||
                    __builtin_sub_overflow(x, y, &z)) return -1;
                w[r][j] = z / prev;
            }
            w[r][c] = 0;
        }
        prev = w[c][c];
    }
    return w[n - 1][n - 1] != 0;
}
/* the flags of an exact affine map: touched, integral, permutation, invertible */
static void map_finish(tr_sym_map *m) {
    const int32_t n = m->n;
    m->touched = 0;
    if (m->kind == TR_SYM_MAP_VALUES) {
        m->integral = 1; m->permutation = 0; m->invertible = 1;
        if (m->nvalues) m->touched = UINT32_C(1) << m->coord;
        return;
    }
    if (m->carrier == TR_SYM_INTERVAL) {
        for (int32_t i = 0; i < n; i++) {
            int id = m->bi[i].lo == 0.0 && m->bi[i].hi == 0.0;
            for (int32_t j = 0; j < n && id; j++)
                id = m->ai[i][j].lo == (i == j ? 1.0 : 0.0) && m->ai[i][j].hi == (i == j ? 1.0 : 0.0);
            if (!id) m->touched |= UINT32_C(1) << i;
        }
        m->integral = 0; m->permutation = 0; m->invertible = -1;
        return;
    }
    int integral = 1, perm = 1;
    int32_t seen[NC];
    for (int32_t j = 0; j < n; j++) seen[j] = 0;
    for (int32_t i = 0; i < n; i++) {
        int id = m->q[i] == 1 && m->b[i] == 0;
        int32_t ones = 0, other = 0, at = -1;
        for (int32_t j = 0; j < n; j++) {
            i128 v = m->a[i][j];
            if (v != (i == j ? 1 : 0)) id = 0;
            if (v < INT64_MIN || v > INT64_MAX) integral = 0;
            if (v == 1) { ones++; at = j; } else if (v != 0) other++;
        }
        if (!id) m->touched |= UINT32_C(1) << i;
        if (m->q[i] != 1 || m->b[i] < INT64_MIN || m->b[i] > INT64_MAX) integral = 0;
        if (m->q[i] != 1 || m->b[i] != 0 || ones != 1 || other) perm = 0;
        else if (seen[at]++) perm = 0;
    }
    m->integral = integral;
    m->permutation = perm;
    m->invertible = det_nonzero(m->a, n);
}
/* rows of a value-space affine map (A: n x n, b: n, rationals) into the exact logical rows of m */
static int32_t map_set_exact(tr_sym_map *m, q128 A[NC][NC], q128 B[NC]) {
    const int32_t n = m->n;
    for (int32_t i = 0; i < n; i++) {
        q128 row[NC + 1];
        i128 q = 1;
        for (int32_t j = 0; j <= n; j++) {
            q128 v = j < n ? A[i][j] : B[i];
            int k = j < n ? m->geo.scale[i] - m->geo.scale[j] : m->geo.scale[i];   /* D A D^-1, D b */
            if (!q_scale10(v, k, &row[j])) return TR_MATH_OVERFLOW;
            if (row[j].n && !lcm128(q, row[j].d, &q)) return TR_MATH_OVERFLOW;
        }
        u128 g = (u128)q;
        i128 nums[NC + 1];
        for (int32_t j = 0; j <= n; j++) {
            if (!mul128(row[j].n, q / row[j].d, &nums[j]) || !in_range(nums[j])) return TR_MATH_OVERFLOW;
            g = gcd128(g, mag128(nums[j]));
        }
        if (!g) g = 1;
        for (int32_t j = 0; j < n; j++) m->a[i][j] = nums[j] / (i128)g;
        m->b[i] = nums[n] / (i128)g;
        m->q[i] = q / (i128)g;
    }
    return TR_MATH_OK;
}
static tr_math_interval iv_pow10(int k) {
    tr_math_interval t = iv_point(1.0);
    for (int i = 0; i < (k < 0 ? -k : k); i++) t = tr_interval_mul(t, iv_point(10.0));   /* exact up to 10^22 */
    return k >= 0 ? t : tr_interval_div(iv_point(1.0), t);
}
static tr_math_interval iv_q(q128 v) { return tr_interval_div(tr_sym_iv_i128(v.n), tr_sym_iv_i128(v.d)); }
static void map_set_interval(tr_sym_map *m, tr_math_interval A[NC][NC], tr_math_interval B[NC]) {
    const int32_t n = m->n;
    for (int32_t i = 0; i < n; i++) {
        for (int32_t j = 0; j < n; j++) {
            int k = m->geo.scale[i] - m->geo.scale[j];
            m->ai[i][j] = (A[i][j].lo == 0.0 && A[i][j].hi == 0.0) || k == 0 ? A[i][j]
                                                                             : tr_interval_mul(A[i][j], iv_pow10(k));
        }
        m->bi[i] = (B[i].lo == 0.0 && B[i].hi == 0.0) || m->geo.scale[i] == 0 ? B[i]
                                                                               : tr_interval_mul(B[i], iv_pow10(m->geo.scale[i]));
    }
}
/* the coordinates a transformation names: distinct declared coordinates */
static int32_t named_coords(const tr_sym_geometry *g, const tr_sym_transform *t, int32_t *c) {
    if (t->k < 0 || t->k > NC) return TR_MATH_INVALID;
    for (int32_t i = 0; i < t->k; i++) {
        c[i] = coord_of(g, t->col[i]);
        if (c[i] < 0) return t->col[i] >= 0 && t->col[i] < g->ncols ? TR_MATH_DOMAIN_ERROR : TR_MATH_INVALID;
        for (int32_t j = 0; j < i; j++) if (c[j] == c[i]) return TR_MATH_INVALID;
    }
    return TR_MATH_OK;
}
typedef struct { int64_t key, target; } valpair;
static int valpair_cmp(const void *a, const void *b) {
    int64_t x = ((const valpair *)a)->key, y = ((const valpair *)b)->key;
    return x < y ? -1 : x > y;
}
static int i64_cmp(const void *a, const void *b) {
    int64_t x = *(const int64_t *)a, y = *(const int64_t *)b;
    return x < y ? -1 : x > y;
}
/* a VALUES map: logical values (exact at the column's scale), a permutation of its keys; the moved pairs by key */
static int32_t map_values(tr_sym_map *m, const tr_sym_transform *t, int32_t c) {
    if (t->nvalues > TR_SYM_MAX_VALUES) return TR_MATH_UNSUPPORTED;
    if (t->nvalues && (!t->keys || !t->targets)) return TR_MATH_INVALID;
    const uint64_t nv = t->nvalues;
    valpair *pr = zalloc(nv * sizeof *pr);
    int64_t *tg = zalloc(nv * 8);
    if (!pr || !tg) { free(pr); free(tg); return TR_MATH_MEMORY; }
    int32_t rc = TR_MATH_OK;
    for (uint64_t i = 0; i < nv && !rc; i++) {
        q128 a, b;
        if (!q_from(t->keys[i], &a) || !q_from(t->targets[i], &b)) { rc = TR_MATH_INVALID; break; }
        if (!q_scale10(a, m->geo.scale[c], &a) || !q_scale10(b, m->geo.scale[c], &b)) { rc = TR_MATH_OVERFLOW; break; }
        if (a.d != 1 || b.d != 1 || a.n < INT64_MIN || a.n > INT64_MAX || b.n < INT64_MIN || b.n > INT64_MAX)
            rc = TR_MATH_DOMAIN_ERROR;                     /* not a value of the column at its scale */
        pr[i].key = (int64_t)a.n; pr[i].target = (int64_t)b.n; tg[i] = (int64_t)b.n;
    }
    if (!rc) {                                             /* keys distinct, and the targets the same set */
        qsort(pr, (size_t)nv, sizeof *pr, valpair_cmp);
        qsort(tg, (size_t)nv, sizeof *tg, i64_cmp);
        for (uint64_t i = 0; i < nv && !rc; i++)
            if ((i && pr[i].key == pr[i - 1].key) || pr[i].key != tg[i]) rc = TR_MATH_DOMAIN_ERROR;
    }
    if (!rc) {                                             /* the moved values, keys increasing */
        uint64_t moved = 0;
        for (uint64_t i = 0; i < nv; i++) if (pr[i].key != pr[i].target) moved++;
        m->keys = zalloc(moved * 8); m->targets = zalloc(moved * 8);
        if (!m->keys || !m->targets) rc = TR_MATH_MEMORY;
        for (uint64_t i = 0, at = 0; i < nv && !rc; i++)
            if (pr[i].key != pr[i].target) { m->keys[at] = pr[i].key; m->targets[at] = pr[i].target; at++; }
        m->nvalues = moved;
        m->bytes += moved * 16;
    }
    free(pr); free(tg);
    return rc;
}
tr_sym_map *tr_sym_map_make(const tr_sym_geometry *g, const tr_sym_transform *t, const tr_math_budget *budget,
                            tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    if (!t || t->reserved) return map_fail(NULL, error, TR_MATH_INVALID);
    int32_t rc = tr_sym_geometry_check(g);
    if (rc) return map_fail(NULL, error, rc);
    if (t->kind < TR_SYM_T_IDENTITY || t->kind > TR_SYM_T_VALUES) return map_fail(NULL, error, TR_MATH_INVALID);
    tr_math_need need;
    need_zero(&need);
    need.need[TR_BUDGET_VARIABLES] = (uint64_t)(t->k > 0 ? t->k : 0);
    need.need[TR_BUDGET_SUPPORT] = t->kind == TR_SYM_T_VALUES ? t->nvalues : 0;
    need.need[TR_BUDGET_BYTES] = tr_budget_add(sizeof(tr_sym_map), t->kind == TR_SYM_T_VALUES ? tr_budget_mul(t->nvalues, 48) : 0);
    rc = check_need(budget, &need, refusal);
    if (rc) return map_fail(NULL, error, rc);
    int32_t c[NC];
    rc = named_coords(g, t, c);
    if (rc) return map_fail(NULL, error, rc);
    const int32_t k = t->k, n = g->n;
    switch (t->kind) {                                            /* the declared geometry the kind needs */
    case TR_SYM_T_IDENTITY:
        if (k) return map_fail(NULL, error, TR_MATH_INVALID);
        break;
    case TR_SYM_T_VALUES:
        if (k != 1) return map_fail(NULL, error, TR_MATH_INVALID);
        break;
    case TR_SYM_T_ROTATE:
        if (k != 2 || (t->mode != TR_SYM_ROT_COSSIN && t->mode != TR_SYM_ROT_DEGREES))
            return map_fail(NULL, error, TR_MATH_INVALID);
        if (g->kind[c[0]] != TR_SYM_NUMERIC || g->kind[c[1]] != TR_SYM_NUMERIC || g->unit[c[0]] != g->unit[c[1]])
            return map_fail(NULL, error, TR_MATH_DOMAIN_ERROR);  /* a rotation needs commensurable metric axes */
        break;
    case TR_SYM_T_PERMUTE: {
        if (k < 1) return map_fail(NULL, error, TR_MATH_INVALID);
        int32_t seen[NC];
        for (int32_t i = 0; i < k; i++) seen[i] = 0;
        for (int32_t i = 0; i < k; i++) {
            if (t->perm[i] < 0 || t->perm[i] >= k || seen[t->perm[i]]++) return map_fail(NULL, error, TR_MATH_INVALID);
        }
        for (int32_t i = 0; i < k; i++) {
            int32_t a = c[i], b = c[t->perm[i]];
            if (g->kind[a] != g->kind[b] || g->unit[a] != g->unit[b]) return map_fail(NULL, error, TR_MATH_DOMAIN_ERROR);
        }
        break;
    }
    default:                                                      /* TRANSLATE, SCALE, REFLECT, AFFINE: metric */
        if (k < 1 || (t->kind == TR_SYM_T_AFFINE && (!t->A || !t->b))) return map_fail(NULL, error, TR_MATH_INVALID);
        for (int32_t i = 0; i < k; i++)
            if (g->kind[c[i]] != TR_SYM_NUMERIC) return map_fail(NULL, error, TR_MATH_DOMAIN_ERROR);
        break;
    }
    if (t->kind == TR_SYM_T_VALUES) {
        tr_sym_map *m = map_new(g, TR_SYM_MAP_VALUES, TR_SYM_EXACT);
        if (!m) return map_fail(NULL, error, TR_MATH_MEMORY);
        m->coord = c[0];
        rc = map_values(m, t, c[0]);
        if (rc) return map_fail(m, error, rc);
        map_finish(m);
        m->magic = MAP_MAGIC;
        set_error(error, TR_MATH_OK);
        return m;
    }
    /* the value-space affine map */
    q128 A[NC][NC], B[NC];
    for (int32_t i = 0; i < n; i++) { for (int32_t j = 0; j < n; j++) A[i][j] = q_int(i == j); B[i] = q_int(0); }
    int interval = 0;
    tr_math_interval IA[NC][NC], IB[NC];
    switch (t->kind) {
    case TR_SYM_T_TRANSLATE:
        for (int32_t i = 0; i < k; i++) if (!q_from(t->value[i], &B[c[i]])) return map_fail(NULL, error, TR_MATH_INVALID);
        break;
    case TR_SYM_T_SCALE:
        for (int32_t i = 0; i < k; i++) if (!q_from(t->value[i], &A[c[i]][c[i]])) return map_fail(NULL, error, TR_MATH_INVALID);
        break;
    case TR_SYM_T_REFLECT:
        for (int32_t i = 0; i < k; i++) {
            q128 a;
            if (!q_from(t->value[i], &a)) return map_fail(NULL, error, TR_MATH_INVALID);
            A[c[i]][c[i]] = q_int(-1);
            if (!q_add(a, a, &B[c[i]])) return map_fail(NULL, error, TR_MATH_OVERFLOW);
        }
        break;
    case TR_SYM_T_PERMUTE:
        for (int32_t i = 0; i < k; i++) {
            A[c[i]][c[i]] = q_int(0);
        }
        for (int32_t i = 0; i < k; i++) A[c[i]][c[t->perm[i]]] = q_int(1);
        break;
    case TR_SYM_T_AFFINE:
        for (int32_t i = 0; i < k; i++) {
            for (int32_t j = 0; j < k; j++)
                if (!q_from(t->A[(uint64_t)i * (uint64_t)k + (uint64_t)j], &A[c[i]][c[j]]))
                    return map_fail(NULL, error, TR_MATH_INVALID);
            if (!q_from(t->b[i], &B[c[i]])) return map_fail(NULL, error, TR_MATH_INVALID);
        }
        break;
    case TR_SYM_T_ROTATE: {
        q128 cx, cy, cc = q_int(1), ss = q_int(0);
        if (!q_from(t->center[0], &cx) || !q_from(t->center[1], &cy)) return map_fail(NULL, error, TR_MATH_INVALID);
        int x = c[0], y = c[1], exact = 1;
        tr_math_interval ic = iv_point(1.0), is = iv_point(0.0);
        if (t->mode == TR_SYM_ROT_COSSIN) {
            if (!q_from(t->cosine, &cc) || !q_from(t->sine, &ss)) return map_fail(NULL, error, TR_MATH_INVALID);
            q128 c2, s2, one;                                      /* c^2 + s^2 = 1 exactly */
            if (!q_mul(cc, cc, &c2) || !q_mul(ss, ss, &s2) || !q_add(c2, s2, &one)) return map_fail(NULL, error, TR_MATH_OVERFLOW);
            if (one.n != 1 || one.d != 1) return map_fail(NULL, error, TR_MATH_DOMAIN_ERROR);
        } else {
            int ec = 0, es = 0;
            int r = rotation_degrees(t->degrees, &ec, &es, &ic, &is);
            if (r < 0) return map_fail(NULL, error, TR_MATH_INVALID);
            if (r == 1) { cc = q_int(ec); ss = q_int(es); } else exact = 0;
        }
        if (exact) {
            q128 ms, t1, t2, t3;
            q_neg(ss, &ms);
            A[x][x] = cc; A[x][y] = ms; A[y][x] = ss; A[y][y] = cc;
            /* bx = cx - c cx + s cy, by = cy - s cx - c cy */
            if (!q_mul(cc, cx, &t1) || !q_mul(ss, cy, &t2) || !q_neg(t1, &t1) || !q_add(cx, t1, &t3) ||
                !q_add(t3, t2, &B[x])) return map_fail(NULL, error, TR_MATH_OVERFLOW);
            if (!q_mul(ss, cx, &t1) || !q_mul(cc, cy, &t2) || !q_neg(t1, &t1) || !q_neg(t2, &t2) ||
                !q_add(cy, t1, &t3) || !q_add(t3, t2, &B[y])) return map_fail(NULL, error, TR_MATH_OVERFLOW);
        } else {
            interval = 1;
            for (int32_t i = 0; i < n; i++) {
                for (int32_t j = 0; j < n; j++) IA[i][j] = iv_point(i == j ? 1.0 : 0.0);
                IB[i] = iv_point(0.0);
            }
            IA[x][x] = ic; IA[x][y] = tr_interval_neg(is); IA[y][x] = is; IA[y][y] = ic;
            tr_math_interval icx = iv_q(cx), icy = iv_q(cy);
            IB[x] = tr_interval_add(tr_interval_sub(icx, tr_interval_mul(ic, icx)), tr_interval_mul(is, icy));
            IB[y] = tr_interval_sub(tr_interval_sub(icy, tr_interval_mul(is, icx)), tr_interval_mul(ic, icy));
        }
        break;
    }
    default:
        break;
    }
    tr_sym_map *m = map_new(g, TR_SYM_MAP_AFFINE, interval ? TR_SYM_INTERVAL : TR_SYM_EXACT);
    if (!m) return map_fail(NULL, error, TR_MATH_MEMORY);
    if (interval) map_set_interval(m, IA, IB);
    else {
        rc = map_set_exact(m, A, B);
        if (rc) return map_fail(m, error, rc);
        for (int32_t i = 0; i < n; i++) {                         /* their interval images too (exact rows enclosed) */
            for (int32_t j = 0; j < n; j++) {
                q128 v; q_make(m->a[i][j], m->q[i], &v);
                m->ai[i][j] = iv_q(v);
            }
            q128 v; q_make(m->b[i], m->q[i], &v);
            m->bi[i] = iv_q(v);
        }
    }
    map_finish(m);
    m->magic = MAP_MAGIC;
    set_error(error, TR_MATH_OK);
    return m;
}
tr_sym_map *tr_sym_map_identity(int32_t n, int32_t *error) {
    if (n < 1 || n > NC) return map_fail(NULL, error, n < 1 ? TR_MATH_INVALID : TR_MATH_UNSUPPORTED);
    int32_t kinds[NC], units[NC], scales[NC];
    for (int32_t i = 0; i < n; i++) { kinds[i] = TR_SYM_NUMERIC; units[i] = 0; scales[i] = 0; }
    tr_sym_geometry g;
    tr_sym_geometry_abstract(n, kinds, units, scales, &g);
    tr_sym_map *m = map_new(&g, TR_SYM_MAP_AFFINE, TR_SYM_EXACT);
    if (!m) return map_fail(NULL, error, TR_MATH_MEMORY);
    map_finish(m);
    m->geo.n = -n;                       /* geometry-free: composes with a map of any geometry with n coordinates */
    m->magic = MAP_MAGIC;
    set_error(error, TR_MATH_OK);
    return m;
}
static int map_geo_free(const tr_sym_map *m) { return m->geo.n < 0; }
tr_sym_map *tr_sym_map_compose(const tr_sym_map *T, const tr_sym_map *S, const tr_math_budget *budget,
                               tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    if (!map_ok(T) || !map_ok(S) || T->n != S->n) return map_fail(NULL, error, TR_MATH_INVALID);
    if (T->kind != TR_SYM_MAP_AFFINE || S->kind != TR_SYM_MAP_AFFINE) return map_fail(NULL, error, TR_MATH_UNSUPPORTED);
    if (!map_geo_free(T) && !map_geo_free(S) && !geo_equal(&T->geo, &S->geo)) return map_fail(NULL, error, TR_MATH_DOMAIN_ERROR);
    tr_math_need need;
    need_zero(&need);
    need.need[TR_BUDGET_VARIABLES] = (uint64_t)T->n;
    need.need[TR_BUDGET_BYTES] = sizeof(tr_sym_map);
    int32_t rc = check_need(budget, &need, refusal);
    if (rc) return map_fail(NULL, error, rc);
    const int32_t n = T->n;
    const tr_sym_geometry *g = map_geo_free(T) ? &S->geo : &T->geo;
    int interval = T->carrier == TR_SYM_INTERVAL || S->carrier == TR_SYM_INTERVAL;
    tr_sym_map *m = map_new(g, TR_SYM_MAP_AFFINE, interval ? TR_SYM_INTERVAL : TR_SYM_EXACT);
    if (!m) return map_fail(NULL, error, TR_MATH_MEMORY);
    if (map_geo_free(T) && map_geo_free(S)) m->geo.n = -n;
    for (int32_t i = 0; i < n; i++) {                             /* intervals: A_T A_S and A_T b_S + b_T */
        for (int32_t k = 0; k < n; k++) {
            tr_math_interval s = iv_point(0.0);
            for (int32_t j = 0; j < n; j++) {
                if ((T->ai[i][j].lo == 0.0 && T->ai[i][j].hi == 0.0) || (S->ai[j][k].lo == 0.0 && S->ai[j][k].hi == 0.0))
                    continue;
                s = tr_interval_add(s, tr_interval_mul(T->ai[i][j], S->ai[j][k]));
            }
            m->ai[i][k] = s;
        }
        tr_math_interval s = T->bi[i];
        for (int32_t j = 0; j < n; j++) {
            if ((T->ai[i][j].lo == 0.0 && T->ai[i][j].hi == 0.0) || (S->bi[j].lo == 0.0 && S->bi[j].hi == 0.0)) continue;
            s = tr_interval_add(s, tr_interval_mul(T->ai[i][j], S->bi[j]));
        }
        m->bi[i] = s;
    }
    if (!interval) {
        /* row i: x''_i = (sum_j aT_ij y_j + bT_i) / qT_i with y_j = (sum_k aS_jk x_k + bS_j) / qS_j */
        for (int32_t i = 0; i < n; i++) {
            i128 L = 1;
            for (int32_t j = 0; j < n; j++) if (T->a[i][j] && !lcm128(L, S->q[j], &L)) return map_fail(m, error, TR_MATH_OVERFLOW);
            i128 nums[NC + 1], q;
            for (int32_t k = 0; k <= n; k++) {
                i128 acc = 0;
                if (k == n && !mul128(T->b[i], L, &acc)) return map_fail(m, error, TR_MATH_OVERFLOW);
                for (int32_t j = 0; j < n; j++) {
                    if (!T->a[i][j]) continue;
                    i128 f, p, s = k < n ? S->a[j][k] : S->b[j];
                    if (!mul128(T->a[i][j], L / S->q[j], &f) || !mul128(f, s, &p) || !add128(acc, p, &acc))
                        return map_fail(m, error, TR_MATH_OVERFLOW);
                }
                if (!in_range(acc)) return map_fail(m, error, TR_MATH_OVERFLOW);
                nums[k] = acc;
            }
            if (!mul128(T->q[i], L, &q)) return map_fail(m, error, TR_MATH_OVERFLOW);
            u128 gg = (u128)q;
            for (int32_t k = 0; k <= n; k++) gg = gcd128(gg, mag128(nums[k]));
            for (int32_t k = 0; k < n; k++) m->a[i][k] = nums[k] / (i128)gg;
            m->b[i] = nums[n] / (i128)gg;
            m->q[i] = q / (i128)gg;
        }
    }
    map_finish(m);
    m->magic = MAP_MAGIC;
    set_error(error, TR_MATH_OK);
    return m;
}
int32_t tr_sym_map_get_info(const tr_sym_map *m, tr_sym_map_info *out) {
    if (!map_ok(m) || !out) return TR_MATH_INVALID;
    memset(out, 0, sizeof *out);
    out->n = m->n; out->kind = m->kind; out->carrier = m->carrier; out->integral = m->integral;
    out->permutation = m->permutation; out->invertible = m->invertible; out->coord = m->coord;
    out->precision = m->carrier == TR_SYM_INTERVAL ? TR_SYM_PRECISION : 0;
    out->touched = m->touched; out->nvalues = m->nvalues; out->bytes = m->bytes;
    return TR_MATH_OK;
}
int32_t tr_sym_map_get_exact(const tr_sym_map *m, tr_math_i128 *a, tr_math_i128 *b, tr_math_i128 *q) {
    if (!map_ok(m) || !a || !b || !q) return TR_MATH_INVALID;
    if (m->kind != TR_SYM_MAP_AFFINE || m->carrier != TR_SYM_EXACT) return TR_MATH_UNSUPPORTED;
    for (int32_t i = 0; i < m->n; i++) {
        for (int32_t j = 0; j < m->n; j++) a[i * m->n + j] = tr_i128_set(m->a[i][j]);
        b[i] = tr_i128_set(m->b[i]); q[i] = tr_i128_set(m->q[i]);
    }
    return TR_MATH_OK;
}
int32_t tr_sym_map_get_interval(const tr_sym_map *m, tr_math_interval *a, tr_math_interval *b) {
    if (!map_ok(m) || !a || !b) return TR_MATH_INVALID;
    if (m->kind != TR_SYM_MAP_AFFINE) return TR_MATH_UNSUPPORTED;
    for (int32_t i = 0; i < m->n; i++) {
        for (int32_t j = 0; j < m->n; j++) a[i * m->n + j] = m->ai[i][j];
        b[i] = m->bi[i];
    }
    return TR_MATH_OK;
}
int32_t tr_sym_map_get_values(const tr_sym_map *m, int64_t *keys, int64_t *targets, uint64_t capacity) {
    if (!map_ok(m) || m->kind != TR_SYM_MAP_VALUES || capacity < m->nvalues || (m->nvalues && (!keys || !targets)))
        return TR_MATH_INVALID;
    if (m->nvalues) { memcpy(keys, m->keys, (size_t)m->nvalues * 8); memcpy(targets, m->targets, (size_t)m->nvalues * 8); }
    return TR_MATH_OK;
}
void tr_sym_map_free(tr_sym_map *m) {
    if (!map_ok(m)) return;
    m->magic = 0;
    map_fail(m, NULL, 0);
}

/* ------------------------------------------------------------------ sparse polynomial expansion (host) */
typedef struct { u128 key; i128 c; } sterm;                   /* an exact term */
typedef struct { u128 key; tr_math_interval c; } siterm;      /* an interval term */
typedef struct {
    int32_t nl, exact;
    int32_t var[NC + 1];                                       /* -1: the constant */
    i128 c[NC + 1];
    tr_math_interval ic[NC + 1];
} linform;
static int sterm_cmp(const void *a, const void *b) {
    u128 x = ((const sterm *)a)->key, y = ((const sterm *)b)->key;
    return x < y ? -1 : x > y;
}
static int siterm_cmp(const void *a, const void *b) {
    u128 x = ((const siterm *)a)->key, y = ((const siterm *)b)->key;
    return x < y ? -1 : x > y;
}
static int iv_zero(tr_math_interval x) { return x.lo == 0.0 && x.hi == 0.0; }
/* the linear form of output coordinate i of an affine map (exact numerators over q_i, or intervals) */
static void map_linform(const tr_sym_map *m, int32_t i, int interval, linform *L) {
    L->nl = 0; L->exact = !interval;
    for (int32_t j = 0; j < m->n; j++) {
        if (interval ? iv_zero(m->ai[i][j]) : m->a[i][j] == 0) continue;
        L->var[L->nl] = j; L->c[L->nl] = interval ? 0 : m->a[i][j]; L->ic[L->nl] = m->ai[i][j]; L->nl++;
    }
    if (interval ? !iv_zero(m->bi[i]) : m->b[i] != 0) {
        L->var[L->nl] = -1; L->c[L->nl] = interval ? 0 : m->b[i]; L->ic[L->nl] = m->bi[i]; L->nl++;
    }
}
/* P (np exact terms, keys increasing, nonzero) times L into out: products, sorted, merged, zeros dropped */
static int32_t mul_linear(const sterm *P, uint64_t np, const linform *L, sterm *out, uint64_t *nout) {
    uint64_t k = 0;
    for (uint64_t p = 0; p < np; p++)
        for (int32_t l = 0; l < L->nl; l++) {
            out[k].key = P[p].key + (L->var[l] >= 0 ? (u128)1 << (4 * L->var[l]) : 0);
            if (!mul128(P[p].c, L->c[l], &out[k].c) || !in_range(out[k].c)) return TR_MATH_OVERFLOW;
            k++;
        }
    if (L->nl > 1) qsort(out, (size_t)k, sizeof *out, sterm_cmp);
    uint64_t m = 0;
    for (uint64_t i = 0; i < k;) {
        i128 s = out[i].c;
        uint64_t j = i + 1;
        for (; j < k && out[j].key == out[i].key; j++) if (!add128(s, out[j].c, &s) || !in_range(s)) return TR_MATH_OVERFLOW;
        if (s) { out[m].key = out[i].key; out[m].c = s; m++; }
        i = j;
    }
    *nout = m;
    return TR_MATH_OK;
}
static void mul_linear_iv(const siterm *P, uint64_t np, const linform *L, siterm *out, uint64_t *nout) {
    uint64_t k = 0;
    for (uint64_t p = 0; p < np; p++)
        for (int32_t l = 0; l < L->nl; l++) {
            out[k].key = P[p].key + (L->var[l] >= 0 ? (u128)1 << (4 * L->var[l]) : 0);
            out[k].c = tr_interval_mul(P[p].c, L->ic[l]);
            k++;
        }
    if (L->nl > 1) qsort(out, (size_t)k, sizeof *out, siterm_cmp);   /* stable enough: equal keys are summed */
    uint64_t m = 0;
    for (uint64_t i = 0; i < k;) {
        tr_math_interval s = out[i].c;
        uint64_t j = i + 1;
        for (; j < k && out[j].key == out[i].key; j++) s = tr_interval_add(s, out[j].c);
        if (!iv_zero(s)) { out[m].key = out[i].key; out[m].c = s; m++; }
        i = j;
    }
    *nout = m;
}
/* an upper bound on the products of one expansion step and on the terms of the expansion of x^e under forms L:
 * with v = the variables the forms of e's coordinates reach and k = |e|, at most C(v + k - 1, k - 1) (v + 1)
 * products and C(v + k, k) terms */
static uint64_t expansion_cap(const linform *L, const uint8_t *e, int32_t n) {
    uint32_t vars = 0;
    int32_t k = 0;
    for (int32_t i = 0; i < n; i++) {
        if (!e[i]) continue;
        k += e[i];
        for (int32_t l = 0; l < L[i].nl; l++) if (L[i].var[l] >= 0) vars |= UINT32_C(1) << L[i].var[l];
    }
    uint64_t v = (uint64_t)__builtin_popcount(vars);
    if (k == 0) return 1;
    uint64_t steps = tr_budget_mul(tr_budget_monomials((uint32_t)v, (uint32_t)(k - 1)), v + 1);
    uint64_t terms = tr_budget_monomials((uint32_t)v, (uint32_t)k);
    uint64_t cap = steps > terms ? steps : terms;
    return cap ? cap : 1;
}
/* the expansion of prod_i L_i^{e_i} into P (capacity cap; tmp the same); *np its terms */
static int32_t expand_exact(const linform *L, const uint8_t *e, int32_t n, sterm *P, sterm *tmp, uint64_t *np) {
    P[0].key = 0; P[0].c = 1;
    uint64_t cnt = 1;
    for (int32_t i = 0; i < n; i++)
        for (int32_t r = 0; r < e[i]; r++) {
            uint64_t m = 0;
            int32_t rc = mul_linear(P, cnt, &L[i], tmp, &m);
            if (rc) return rc;
            memcpy(P, tmp, (size_t)m * sizeof *P);
            cnt = m;
        }
    *np = cnt;
    return TR_MATH_OK;
}
static void expand_interval(const linform *L, const uint8_t *e, int32_t n, siterm *P, siterm *tmp, uint64_t *np) {
    P[0].key = 0; P[0].c = iv_point(1.0);
    uint64_t cnt = 1;
    for (int32_t i = 0; i < n; i++)
        for (int32_t r = 0; r < e[i]; r++) {
            uint64_t m = 0;
            mul_linear_iv(P, cnt, &L[i], tmp, &m);
            memcpy(P, tmp, (size_t)m * sizeof *P);
            cnt = m;
        }
    *np = cnt;
}

/* ------------------------------------------------------------------ actions */
struct tr_sym_action_s {
    uint32_t magic;
    int32_t nb, n, carrier, permutation, monomial, identity, has_geo;
    tr_sym_geometry geo;
    tr_sym_basis *basis;          /* an owned clone */
    i128 den;
    tr_math_i128 *num;            /* EXACT: nb x nb */
    tr_math_interval *iv;         /* INTERVAL: nb x nb */
    int32_t *index;               /* permutation */
    uint64_t nonzeros, bytes;
};
static int action_ok(const tr_sym_action *a) { return a && a->magic == ACTION_MAGIC; }
static tr_sym_action *action_fail(tr_sym_action *a, int32_t *error, int32_t code) {
    if (a) { tr_sym_basis_free(a->basis); free(a->num); free(a->iv); free(a->index); free(a); }
    set_error(error, code);
    return NULL;
}
static uint64_t action_bytes(int32_t nb, int32_t carrier) {
    uint64_t e = tr_budget_mul((uint64_t)nb, (uint64_t)nb);
    return tr_budget_add(tr_budget_add(sizeof(tr_sym_action), tr_budget_mul(e, carrier == TR_SYM_EXACT ? 16 : 16)),
                         tr_budget_add(tr_budget_mul((uint64_t)nb, 4), basis_bytes(TR_SYM_MAX_COORDS, nb)));
}
/* monomial, permutation, identity flags and the nonzeros of a built action; TR_MATH_MEMORY */
static int32_t action_flags(tr_sym_action *a) {
    const int32_t nb = a->nb;
    int mono = 1, perm = 1, id = 1;
    uint64_t nz = 0;
    int32_t *col = zalloc((uint64_t)nb * 4), *seen = zalloc((uint64_t)nb * 4);
    if (!col || !seen) { free(col); free(seen); return TR_MATH_MEMORY; }
    for (int32_t i = 0; i < nb; i++) {
        int32_t cnt = 0;
        for (int32_t j = 0; j < nb; j++) {
            int nzero = a->carrier == TR_SYM_EXACT ? tr_i128_get(a->num[(uint64_t)i * nb + j]) != 0
                                                   : !iv_zero(a->iv[(uint64_t)i * nb + j]);
            if (!nzero) continue;
            nz++; cnt++; col[i] = j;
            int one = a->carrier == TR_SYM_EXACT && a->den == 1 && tr_i128_get(a->num[(uint64_t)i * nb + j]) == 1;
            if (!one) perm = 0;
            if (j != i || !one) id = 0;
        }
        if (cnt != 1) { mono = 0; perm = 0; id = 0; }
    }
    if (perm) for (int32_t i = 0; i < nb; i++) if (seen[col[i]]++) { perm = 0; id = 0; }
    a->monomial = mono; a->permutation = perm; a->identity = id; a->nonzeros = nz;
    if (perm) { a->index = col; col = NULL; }
    free(col); free(seen);
    return TR_MATH_OK;
}
int32_t tr_sym_action_need(const tr_sym_basis *b, const tr_sym_map *m, tr_math_need *need) {
    if (!basis_ok(b) || !map_ok(m) || !need || b->n != m->n) return TR_MATH_INVALID;
    need_zero(need);
    need->need[TR_BUDGET_VARIABLES] = (uint64_t)b->n;
    need->need[TR_BUDGET_DEGREE] = (uint64_t)b->degree;
    need->need[TR_BUDGET_SUPPORT] = (uint64_t)b->nb;
    need->need[TR_BUDGET_TENSOR] = tr_budget_mul((uint64_t)b->nb, (uint64_t)b->nb);
    uint64_t cap = 1;
    if (m->kind == TR_SYM_MAP_AFFINE) {
        linform L[NC];
        for (int32_t i = 0; i < m->n; i++) map_linform(m, i, m->carrier == TR_SYM_INTERVAL, &L[i]);
        for (int32_t r = 0; r < b->nb; r++) {
            uint64_t c = expansion_cap(L, b->exps + (uint64_t)r * (uint64_t)b->n, b->n);
            if (c > cap) cap = c;
        }
    }
    uint64_t bytes = action_bytes(b->nb, m->carrier);
    bytes = tr_budget_add(bytes, tr_budget_mul(cap, 2 * sizeof(sterm)));      /* the expansion and its step */
    bytes = tr_budget_add(bytes, tr_budget_mul((uint64_t)b->nb, 16 + 8));     /* row denominators, flags */
    need->need[TR_BUDGET_BYTES] = bytes;
    return TR_MATH_OK;
}
static tr_sym_action *action_new(const tr_sym_basis *b, int32_t carrier, int32_t *error) {
    tr_sym_action *a = zalloc(sizeof *a);
    if (!a) return NULL;
    a->nb = b->nb; a->n = b->n; a->carrier = carrier; a->den = 1;
    int32_t rc = TR_MATH_OK;
    a->basis = tr_sym_basis_clone(b, &rc);
    uint64_t e = (uint64_t)b->nb * (uint64_t)b->nb;
    if (carrier == TR_SYM_EXACT) a->num = zalloc(e * sizeof(tr_math_i128));
    else a->iv = zalloc(e * sizeof(tr_math_interval));
    if (!a->basis || (carrier == TR_SYM_EXACT ? !a->num : !a->iv)) { action_fail(a, NULL, 0); set_error(error, TR_MATH_MEMORY); return NULL; }
    a->bytes = action_bytes(b->nb, carrier);
    return a;
}
/* gcd-reduces an exact action (D and every numerator) */
static void action_reduce(tr_sym_action *a) {
    u128 g = (u128)a->den;
    uint64_t e = (uint64_t)a->nb * (uint64_t)a->nb;
    for (uint64_t i = 0; i < e && g > 1; i++) g = gcd128(g, mag128(tr_i128_get(a->num[i])));
    if (g <= 1) return;
    for (uint64_t i = 0; i < e; i++) a->num[i] = tr_i128_set(tr_i128_get(a->num[i]) / (i128)g);
    a->den /= (i128)g;
}
tr_sym_action *tr_sym_action_make(const tr_sym_basis *b, const tr_sym_map *m, const tr_math_budget *budget,
                                  tr_math_refusal *refusal, tr_sym_closure *closure, int32_t *error) {
    tr_budget_none(refusal);
    if (closure) { memset(closure, 0, sizeof *closure); closure->term = -1; }
    if (!basis_ok(b) || !map_ok(m) || b->n != m->n) return action_fail(NULL, error, TR_MATH_INVALID);
    if (m->kind != TR_SYM_MAP_AFFINE) return action_fail(NULL, error, TR_MATH_UNSUPPORTED);
    tr_math_need need;
    tr_sym_action_need(b, m, &need);
    int32_t rc = check_need(budget, &need, refusal);
    if (rc) return action_fail(NULL, error, rc);
    const int32_t nb = b->nb, n = b->n;
    const int interval = m->carrier == TR_SYM_INTERVAL;
    linform L[NC];
    uint64_t cap = 1;
    for (int32_t i = 0; i < n; i++) map_linform(m, i, interval, &L[i]);
    for (int32_t r = 0; r < nb; r++) {
        uint64_t c = expansion_cap(L, b->exps + (uint64_t)r * (uint64_t)n, n);
        if (c > cap) cap = c;
    }
    tr_sym_action *a = action_new(b, interval ? TR_SYM_INTERVAL : TR_SYM_EXACT, &rc);
    if (!a) return action_fail(NULL, error, TR_MATH_MEMORY);
    if (!map_geo_free(m)) { a->has_geo = 1; a->geo = m->geo; }
    void *P = zalloc(tr_budget_mul(cap, sizeof(sterm))), *tmp = zalloc(tr_budget_mul(cap, sizeof(sterm)));
    i128 *rden = zalloc((uint64_t)nb * sizeof(i128));
    if (!P || !tmp || !rden) { free(P); free(tmp); free(rden); return action_fail(a, error, TR_MATH_MEMORY); }
    rc = TR_MATH_OK;
    for (int32_t r = 0; r < nb && !rc; r++) {
        const uint8_t *e = b->exps + (uint64_t)r * (uint64_t)n;
        uint64_t np = 0;
        if (interval) expand_interval(L, e, n, (siterm *)P, (siterm *)tmp, &np);
        else if ((rc = expand_exact(L, e, n, (sterm *)P, (sterm *)tmp, &np))) break;
        for (uint64_t t = 0; t < np; t++) {
            u128 key = interval ? ((siterm *)P)[t].key : ((sterm *)P)[t].key;
            int32_t col = basis_lookup(b, key);
            if (col < 0) {                                         /* the basis is not closed under the map */
                if (closure) { closure->term = r; key_exps(key, n, closure->missing); }
                rc = TR_MATH_DOMAIN_ERROR;
                break;
            }
            if (interval) a->iv[(uint64_t)r * nb + col] = ((siterm *)P)[t].c;
            else a->num[(uint64_t)r * nb + col] = tr_i128_set(((sterm *)P)[t].c);
        }
        if (rc || interval) continue;
        i128 d = 1;                                                /* the row's denominator prod q_i^e_i */
        for (int32_t i = 0; i < n && !rc; i++)
            for (int32_t k = 0; k < e[i]; k++) if (!mul128(d, m->q[i], &d)) { rc = TR_MATH_OVERFLOW; break; }
        if (rc) break;
        u128 g = (u128)d;                                          /* reduced by the row's gcd */
        for (int32_t j = 0; j < nb; j++) g = gcd128(g, mag128(tr_i128_get(a->num[(uint64_t)r * nb + j])));
        for (int32_t j = 0; j < nb && g > 1; j++)
            a->num[(uint64_t)r * nb + j] = tr_i128_set(tr_i128_get(a->num[(uint64_t)r * nb + j]) / (i128)g);
        rden[r] = d / (i128)g;
    }
    free(P); free(tmp);
    if (rc) { free(rden); return action_fail(a, error, rc); }
    if (!interval) {                                               /* one common denominator D = lcm of the rows' */
        i128 D = 1;
        for (int32_t r = 0; r < nb && !rc; r++) if (!lcm128(D, rden[r], &D)) rc = TR_MATH_OVERFLOW;
        for (int32_t r = 0; r < nb && !rc; r++) {
            i128 f = D / rden[r];
            for (int32_t j = 0; j < nb && f != 1; j++) {
                i128 v;
                if (!mul128(tr_i128_get(a->num[(uint64_t)r * nb + j]), f, &v) || !in_range(v)) { rc = TR_MATH_OVERFLOW; break; }
                a->num[(uint64_t)r * nb + j] = tr_i128_set(v);
            }
        }
        a->den = D;
        if (!rc) action_reduce(a);
    }
    free(rden);
    if (!rc) rc = action_flags(a);
    if (rc) return action_fail(a, error, rc);
    if (closure) closure->closed = 1;
    a->magic = ACTION_MAGIC;
    set_error(error, TR_MATH_OK);
    return a;
}
tr_sym_action *tr_sym_action_identity(const tr_sym_basis *b, const tr_math_budget *budget, tr_math_refusal *refusal,
                                      int32_t *error) {
    tr_budget_none(refusal);
    if (!basis_ok(b)) return action_fail(NULL, error, TR_MATH_INVALID);
    tr_math_need need;
    need_zero(&need);
    need.need[TR_BUDGET_SUPPORT] = (uint64_t)b->nb;
    need.need[TR_BUDGET_TENSOR] = tr_budget_mul((uint64_t)b->nb, (uint64_t)b->nb);
    need.need[TR_BUDGET_BYTES] = action_bytes(b->nb, TR_SYM_EXACT);
    int32_t rc = check_need(budget, &need, refusal);
    if (rc) return action_fail(NULL, error, rc);
    tr_sym_action *a = action_new(b, TR_SYM_EXACT, &rc);
    if (!a) return action_fail(NULL, error, TR_MATH_MEMORY);
    for (int32_t i = 0; i < b->nb; i++) a->num[(uint64_t)i * b->nb + i] = tr_i128_set(1);
    if ((rc = action_flags(a))) return action_fail(a, error, rc);
    a->magic = ACTION_MAGIC;
    set_error(error, TR_MATH_OK);
    return a;
}
static tr_math_interval action_entry_iv(const tr_sym_action *a, uint64_t at) {
    if (a->carrier == TR_SYM_INTERVAL) return a->iv[at];
    __int128 v = tr_i128_get(a->num[at]);
    if (v == 0) return iv_point(0.0);
    if (a->den == 1) return tr_sym_iv_i128(v);
    return tr_interval_div(tr_sym_iv_i128(v), tr_sym_iv_i128(a->den));
}
tr_sym_action *tr_sym_action_compose(const tr_sym_action *T, const tr_sym_action *S, const tr_math_budget *budget,
                                     tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    if (!action_ok(T) || !action_ok(S)) return action_fail(NULL, error, TR_MATH_INVALID);
    if (tr_sym_basis_equal(T->basis, S->basis) != 1 || (T->has_geo && S->has_geo && !geo_equal(&T->geo, &S->geo)))
        return action_fail(NULL, error, TR_MATH_DOMAIN_ERROR);
    const int32_t nb = T->nb;
    const int interval = T->carrier == TR_SYM_INTERVAL || S->carrier == TR_SYM_INTERVAL;
    tr_math_need need;
    need_zero(&need);
    uint64_t e = tr_budget_mul((uint64_t)nb, (uint64_t)nb);
    need.need[TR_BUDGET_SUPPORT] = (uint64_t)nb;
    need.need[TR_BUDGET_TENSOR] = e;
    need.need[TR_BUDGET_ITERATIONS] = tr_budget_mul(e, (uint64_t)nb);
    need.need[TR_BUDGET_BYTES] = tr_budget_add(action_bytes(nb, interval ? TR_SYM_INTERVAL : TR_SYM_EXACT),
                                               interval ? 0 : tr_budget_mul(e, sizeof(tr_sym_wide)));
    int32_t rc = check_need(budget, &need, refusal);
    if (rc) return action_fail(NULL, error, rc);
    tr_sym_action *a = action_new(T->basis, interval ? TR_SYM_INTERVAL : TR_SYM_EXACT, &rc);
    if (!a) return action_fail(NULL, error, TR_MATH_MEMORY);
    if (T->has_geo || S->has_geo) { a->has_geo = 1; a->geo = T->has_geo ? T->geo : S->geo; }
    if (interval) {
        for (int32_t i = 0; i < nb; i++)
            for (int32_t j = 0; j < nb; j++) {
                tr_math_interval s = iv_point(0.0);
                for (int32_t k = 0; k < nb; k++) {
                    tr_math_interval x = action_entry_iv(T, (uint64_t)i * nb + k);
                    if (iv_zero(x)) continue;
                    tr_math_interval y = action_entry_iv(S, (uint64_t)k * nb + j);
                    if (iv_zero(y)) continue;
                    s = tr_interval_add(s, tr_interval_mul(x, y));
                }
                a->iv[(uint64_t)i * nb + j] = s;
            }
    } else {
        tr_sym_wide *P = zalloc(e * sizeof(tr_sym_wide)), Dw;
        if (!P) return action_fail(a, error, TR_MATH_MEMORY);
        int fits = 1;
        for (int32_t i = 0; i < nb && !rc; i++)
            for (int32_t j = 0; j < nb && !rc; j++) {
                tr_sym_wide *s = &P[(uint64_t)i * nb + j], p;
                tr_sym_w_zero(s);
                for (int32_t k = 0; k < nb; k++) {
                    __int128 x = tr_i128_get(T->num[(uint64_t)i * nb + k]);
                    if (!x) continue;
                    __int128 y = tr_i128_get(S->num[(uint64_t)k * nb + j]);
                    if (!y) continue;
                    tr_sym_w_mul128(x, y, &p);
                    if (!tr_sym_w_add(s, &p, s)) { rc = TR_MATH_OVERFLOW; break; }
                }
                uint64_t fill = tr_sym_w_negative(s) ? ~(uint64_t)0 : 0;   /* fits int128 (with |x| < 2^127)? */
                for (int w = 2; w < TR_SYM_WORDS; w++) if (s->w[w] != fill) fits = 0;
                if ((s->w[1] >> 63) != (fill & 1)) fits = 0;
            }
        tr_sym_w_mul128(T->den, S->den, &Dw);
        for (int w = 2; w < TR_SYM_WORDS; w++) if (Dw.w[w]) fits = 0;
        if (Dw.w[1] >> 63) fits = 0;
        if (!rc && fits) {                                          /* every entry and D in int128: reduce there */
            for (uint64_t x = 0; x < e; x++) {
                i128 v = (i128)(((u128)P[x].w[1] << 64) | P[x].w[0]);
                if (!in_range(v)) { rc = TR_MATH_OVERFLOW; break; }
                a->num[x] = tr_i128_set(v);
            }
            a->den = (i128)(((u128)Dw.w[1] << 64) | Dw.w[0]);
            if (!rc) action_reduce(a);
        } else if (!rc) {                                           /* wide: reduce with FND-009 big integers */
            tr_math_budget unlimited;
            for (int i = 0; i < TR_BUDGET_RESOURCES; i++) unlimited.limit[i] = UINT64_MAX;
            int32_t err = 0;
            tr_bigint *g = tr_bigint_from_twos(Dw.w, TR_SYM_WORDS, &unlimited, NULL, &err);
            for (uint64_t x = 0; x < e && g; x++) {
                if (tr_sym_w_is_zero(&P[x])) continue;
                tr_bigint *v = tr_bigint_from_twos(P[x].w, TR_SYM_WORDS, &unlimited, NULL, &err);
                tr_bigint *h = v ? tr_bigint_gcd(g, v, &unlimited, NULL, &err) : NULL;
                tr_bigint_free(v); tr_bigint_free(g);
                g = h;
            }
            if (!g) rc = TR_MATH_MEMORY;
            for (uint64_t x = 0; x <= e && !rc; x++) {              /* x == e: the denominator */
                const uint64_t *w = x < e ? P[x].w : Dw.w;
                tr_bigint *v = tr_bigint_from_twos(w, TR_SYM_WORDS, &unlimited, NULL, &err);
                tr_bigint *qv = v ? tr_bigint_divexact(v, g, &unlimited, NULL, &err) : NULL;
                tr_math_i128 out;
                if (!qv) rc = TR_MATH_MEMORY;
                else if (tr_bigint_to_i128(qv, &out) || !in_range(tr_i128_get(out))) rc = TR_MATH_OVERFLOW;
                else if (x < e) a->num[x] = out;
                else a->den = tr_i128_get(out);
                tr_bigint_free(v); tr_bigint_free(qv);
            }
            tr_bigint_free(g);
        }
        free(P);
        if (rc) return action_fail(a, error, rc);
    }
    if ((rc = action_flags(a))) return action_fail(a, error, rc);
    a->magic = ACTION_MAGIC;
    set_error(error, TR_MATH_OK);
    return a;
}
int32_t tr_sym_action_equal(const tr_sym_action *a, const tr_sym_action *b) {
    if (!action_ok(a) || !action_ok(b)) return TR_MATH_INVALID;
    if (tr_sym_basis_equal(a->basis, b->basis) != 1 || a->carrier != b->carrier) return 0;
    uint64_t e = (uint64_t)a->nb * (uint64_t)a->nb;
    if (a->carrier == TR_SYM_EXACT) {
        if (a->den != b->den) return 0;
        for (uint64_t i = 0; i < e; i++) if (tr_i128_get(a->num[i]) != tr_i128_get(b->num[i])) return 0;
        return 1;
    }
    for (uint64_t i = 0; i < e; i++) if (a->iv[i].lo != b->iv[i].lo || a->iv[i].hi != b->iv[i].hi) return 0;
    return 1;
}
int32_t tr_sym_action_get_info(const tr_sym_action *a, tr_sym_action_info *out) {
    if (!action_ok(a) || !out) return TR_MATH_INVALID;
    memset(out, 0, sizeof *out);
    out->nb = a->nb; out->n = a->n; out->carrier = a->carrier; out->permutation = a->permutation;
    out->monomial = a->monomial; out->identity = a->identity;
    out->precision = a->carrier == TR_SYM_INTERVAL ? TR_SYM_PRECISION : 0;
    out->approximate = a->carrier == TR_SYM_INTERVAL;
    out->den = tr_i128_set(a->den); out->nonzeros = a->nonzeros; out->bytes = a->bytes;
    return TR_MATH_OK;
}
int32_t tr_sym_action_get_view(const tr_sym_action *a, tr_sym_action_view *v) {
    if (!action_ok(a) || !v) return TR_MATH_INVALID;
    memset(v, 0, sizeof *v);
    v->carrier = a->carrier; v->permutation = a->permutation; v->nb = a->nb;
    v->den = tr_i128_set(a->den); v->num = a->num; v->iv = a->iv; v->index = a->index;
    return TR_MATH_OK;
}
const tr_sym_basis *tr_sym_action_get_basis(const tr_sym_action *a) { return action_ok(a) ? a->basis : NULL; }
void tr_sym_action_free(tr_sym_action *a) {
    if (!action_ok(a)) return;
    a->magic = 0;
    action_fail(a, NULL, 0);
}

/* ------------------------------------------------------------------ states */
struct tr_sym_state_s {
    uint32_t magic;
    int32_t source, weighted, n;
    tr_sym_geometry geo;
    tr_sym_basis *basis;
    uint64_t ngroups, present, rows_total, used_total;
    uint64_t *rows, *used;
    tr_math_i128 *m;              /* ngroups x nb */
    uint64_t bytes;
};
static int state_ok(const tr_sym_state *s) { return s && s->magic == STATE_MAGIC; }
static tr_sym_state *state_fail(tr_sym_state *s, int32_t *error, int32_t code) {
    if (s) { tr_sym_basis_free(s->basis); free(s->rows); free(s->used); free(s->m); free(s); }
    set_error(error, code);
    return NULL;
}
static tr_sym_state *state_alloc(const tr_sym_geometry *g, uint64_t G, int32_t nb, const tr_math_budget *budget,
                                 tr_math_refusal *refusal, int32_t *rc) {
    tr_math_need need;
    need_zero(&need);
    need.need[TR_BUDGET_VARIABLES] = (uint64_t)g->n;
    need.need[TR_BUDGET_SUPPORT] = (uint64_t)nb;
    need.need[TR_BUDGET_STATES] = tr_budget_mul(G, (uint64_t)nb);
    need.need[TR_BUDGET_BYTES] = tr_budget_add(tr_budget_add(sizeof(tr_sym_state), basis_bytes(g->n, nb)),
                                               tr_budget_add(tr_budget_mul(tr_budget_mul(G, (uint64_t)nb), 16),
                                                             tr_budget_mul(G, 16)));
    *rc = check_need(budget, &need, refusal);
    if (*rc) return NULL;
    tr_sym_state *s = zalloc(sizeof *s);
    if (!s) { *rc = TR_MATH_MEMORY; return NULL; }
    s->rows = zalloc(G * 8); s->used = zalloc(G * 8); s->m = zalloc(tr_budget_mul(G, (uint64_t)nb) * 16);
    if (!s->rows || !s->used || !s->m) { state_fail(s, NULL, 0); *rc = TR_MATH_MEMORY; return NULL; }
    s->geo = *g; s->n = g->n; s->ngroups = G;
    s->bytes = need.need[TR_BUDGET_BYTES];
    return s;
}
static void state_totals(tr_sym_state *s) {
    for (uint64_t g = 0; g < s->ngroups; g++) {
        if (s->rows[g]) s->present++;
        s->rows_total += s->rows[g]; s->used_total += s->used[g];
    }
}
tr_sym_state *tr_sym_state_from_bank(const tr_bank *bank, const tr_sym_geometry *g, const tr_math_budget *budget,
                                     tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    int32_t rc = tr_sym_geometry_check(g);
    tr_bank_info bi;
    if (rc || !bank || tr_bank_get_info(bank, &bi)) return state_fail(NULL, error, rc ? rc : TR_MATH_INVALID);
    if (bi.digest[0] != g->digest[0] || bi.digest[1] != g->digest[1] || bi.ncols != g->ncols)
        return state_fail(NULL, error, TR_MATH_DOMAIN_ERROR);      /* a bank of another scope */
    if (bi.nterms < 1) return state_fail(NULL, error, TR_MATH_INVALID);
    const int32_t nb = bi.nterms, nc = bi.ncols, n = g->n;
    uint8_t *be = zalloc((uint64_t)nb * (uint64_t)nc), *ex = zalloc((uint64_t)nb * (uint64_t)n);
    if (!be || !ex) { free(be); free(ex); return state_fail(NULL, error, TR_MATH_MEMORY); }
    rc = tr_bank_get_exponents(bank, be, (uint64_t)nb * (uint64_t)nc);
    for (int32_t j = 0; j < nb && !rc; j++)                       /* each term over the declared coordinates */
        for (int32_t c = 0; c < nc; c++) {
            uint8_t e = be[(uint64_t)j * nc + c];
            if (!e) continue;
            int32_t i = coord_of(g, c);
            if (i < 0 || g->kind[i] != TR_SYM_NUMERIC) { rc = TR_MATH_DOMAIN_ERROR; break; }
            ex[(uint64_t)j * n + i] = e;
        }
    free(be);
    if (rc) { free(ex); return state_fail(NULL, error, rc); }
    tr_sym_basis *basis = basis_make(n, ex, nb, budget, refusal, &rc);
    free(ex);
    if (!basis) return state_fail(NULL, error, rc);
    tr_bank_states st;
    if (tr_bank_get_states(bank, &st)) { tr_sym_basis_free(basis); return state_fail(NULL, error, TR_MATH_INVALID); }
    tr_sym_state *s = state_alloc(g, st.ngroups, nb, budget, refusal, &rc);
    if (!s) { tr_sym_basis_free(basis); return state_fail(NULL, error, rc); }
    s->basis = basis; s->source = TR_SYM_FROM_BANK;
    for (uint64_t gr = 0; gr < st.ngroups && !rc; gr++) {
        s->rows[gr] = st.rows[gr];
        uint64_t used = st.count[gr * (uint64_t)nb];
        for (int32_t j = 0; j < nb; j++) {
            uint64_t c = st.count[gr * (uint64_t)nb + j];
            /* SQL: every term valid on every row; JOINT: one count for all the terms */
            if ((bi.mode == TR_BANK_SQL && c != st.rows[gr]) || (bi.mode == TR_BANK_JOINT && c != used)) {
                rc = bi.mode == TR_BANK_SQL ? TR_MATH_DOMAIN_ERROR : TR_MATH_INVALID;
                break;
            }
            s->m[gr * (uint64_t)nb + j] = st.sum[gr * (uint64_t)nb + j];
        }
        s->used[gr] = used;
    }
    if (rc) return state_fail(s, error, rc);
    state_totals(s);
    s->magic = STATE_MAGIC;
    set_error(error, TR_MATH_OK);
    return s;
}
tr_sym_state *tr_sym_state_from_tensor(const tr_tensor *t, const tr_sym_geometry *g, const tr_math_budget *budget,
                                       tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    int32_t rc = tr_sym_geometry_check(g);
    tr_tensor_info ti;
    if (rc || !t || tr_tensor_get_info(t, &ti)) return state_fail(NULL, error, rc ? rc : TR_MATH_INVALID);
    if (ti.digest[0] != g->digest[0] || ti.digest[1] != g->digest[1] || !ti.scoped)
        return state_fail(NULL, error, TR_MATH_DOMAIN_ERROR);      /* a tensor of another scope (or none) */
    if (ti.kind != TR_TENSOR_MOMENT || ti.policy != TR_TENSOR_LISTWISE || ti.has_counts || ti.order < 1)
        return state_fail(NULL, error, TR_MATH_DOMAIN_ERROR);
    if (ti.order > TR_SYM_MAX_DEGREE) return state_fail(NULL, error, TR_MATH_UNSUPPORTED);
    const int32_t m = ti.order, n = g->n;
    tr_tensor_axis ax, ax0;
    if (tr_tensor_get_axis(t, 0, &ax0) || ax0.kind != TR_TENSOR_AXIS_FEATURE) return state_fail(NULL, error, TR_MATH_DOMAIN_ERROR);
    for (int32_t k = 1; k < m; k++) {                              /* every axis lists the same columns */
        if (tr_tensor_get_axis(t, k, &ax) || ax.kind != TR_TENSOR_AXIS_FEATURE || ax.ncols != ax0.ncols ||
            memcmp(ax.cols, ax0.cols, (size_t)ax0.ncols)) return state_fail(NULL, error, TR_MATH_DOMAIN_ERROR);
    }
    int32_t pos_const = -1, pos[NC];
    uint32_t mask = 0;
    for (int32_t i = 0; i < NC; i++) pos[i] = -1;
    for (int32_t p = 0; p < ax0.ncols; p++) {
        int32_t c = ax0.cols[p];
        if (c < 0) {
            if (pos_const >= 0) return state_fail(NULL, error, TR_MATH_DOMAIN_ERROR);
            pos_const = p;
            continue;
        }
        int32_t i = coord_of(g, c);
        if (i < 0 || g->kind[i] != TR_SYM_NUMERIC || pos[i] >= 0) return state_fail(NULL, error, TR_MATH_DOMAIN_ERROR);
        pos[i] = p; mask |= UINT32_C(1) << i;
    }
    if (!mask) return state_fail(NULL, error, TR_MATH_DOMAIN_ERROR);
    tr_sym_basis *basis = tr_sym_basis_full(n, mask, m, pos_const < 0, budget, refusal, &rc);
    if (!basis) return state_fail(NULL, error, rc);
    tr_tensor_states st;
    if (tr_tensor_get_states(t, &st)) { tr_sym_basis_free(basis); return state_fail(NULL, error, TR_MATH_INVALID); }
    tr_sym_state *s = state_alloc(g, ti.ngroups, basis->nb, budget, refusal, &rc);
    if (!s) { tr_sym_basis_free(basis); return state_fail(NULL, error, rc); }
    s->basis = basis; s->source = TR_SYM_FROM_TENSOR; s->weighted = ti.weighted;
    const int32_t nb = basis->nb;
    for (uint64_t gr = 0; gr < ti.ngroups && !rc; gr++) {
        s->rows[gr] = st.rows[gr]; s->used[gr] = st.used[gr];
        for (int32_t j = 0; j < nb && !rc; j++) {                 /* the moment of x^a: the sorted multi-index */
            const uint8_t *e = basis->exps + (uint64_t)j * (uint64_t)n;
            uint32_t coords[TR_TENSOR_MAX_ORDER];
            int32_t at = 0, deg = 0;
            for (int32_t i = 0; i < n; i++) deg += e[i];
            for (int32_t r = 0; r < m - deg; r++) coords[at++] = (uint32_t)pos_const;
            for (int32_t i = 0; i < n; i++) for (int32_t r = 0; r < e[i]; r++) coords[at++] = (uint32_t)pos[i];
            for (int32_t x = 1; x < at; x++) {
                uint32_t v = coords[x]; int32_t y = x - 1;
                while (y >= 0 && coords[y] > v) { coords[y + 1] = coords[y]; y--; }
                coords[y + 1] = v;
            }
            tr_math_i128 v;
            uint64_t cnt;
            if (tr_tensor_get(t, gr, coords, &v, &cnt)) rc = TR_MATH_INVALID;
            s->m[gr * (uint64_t)nb + j] = v;
        }
    }
    if (rc) return state_fail(s, error, rc);
    state_totals(s);
    s->magic = STATE_MAGIC;
    set_error(error, TR_MATH_OK);
    return s;
}
int32_t tr_sym_state_get_info(const tr_sym_state *s, tr_sym_state_info *out) {
    if (!state_ok(s) || !out) return TR_MATH_INVALID;
    memset(out, 0, sizeof *out);
    out->ngroups = s->ngroups; out->present = s->present; out->rows = s->rows_total; out->used = s->used_total;
    out->bytes = s->bytes; out->nb = s->basis->nb; out->n = s->n; out->source = s->source; out->weighted = s->weighted;
    out->constant = s->basis->constant; out->digest[0] = s->geo.digest[0]; out->digest[1] = s->geo.digest[1];
    return TR_MATH_OK;
}
const tr_sym_basis *tr_sym_state_get_basis(const tr_sym_state *s) { return state_ok(s) ? s->basis : NULL; }
int32_t tr_sym_state_get_view(const tr_sym_state *s, tr_sym_query_view *v) {
    if (!state_ok(s) || !v) return TR_MATH_INVALID;
    memset(v, 0, sizeof *v);
    v->ngroups = s->ngroups; v->nb = s->basis->nb; v->constant = s->basis->constant; v->moments = s->m;
    return TR_MATH_OK;
}
int32_t tr_sym_state_get_rows(const tr_sym_state *s, uint64_t *rows, uint64_t *used, uint64_t capacity) {
    if (!state_ok(s) || !rows || !used || capacity < s->ngroups) return TR_MATH_INVALID;
    memcpy(rows, s->rows, (size_t)s->ngroups * 8); memcpy(used, s->used, (size_t)s->ngroups * 8);
    return TR_MATH_OK;
}
void tr_sym_state_free(tr_sym_state *s) {
    if (!state_ok(s)) return;
    s->magic = 0;
    state_fail(s, NULL, 0);
}

/* ------------------------------------------------------------------ requests */
int32_t tr_sym_query_need(const tr_sym_query_view *v, uint64_t n, int32_t target, tr_math_need *need) {
    if (!need || tr_sym_query_view_check(v) || (target != TR_SYM_NEED_CPU && target != TR_SYM_NEED_DEVICE))
        return TR_MATH_INVALID;
    tr_sym_query_needs(v, n, target, need);
    return TR_MATH_OK;
}
int32_t tr_sym_query_view_eval(const tr_sym_query_view *v, const tr_sym_request *req, uint64_t n, tr_sym_result *out,
                               const tr_math_budget *budget, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (tr_sym_query_view_check(v) || (n && (!req || !out))) return TR_MATH_INVALID;
    tr_math_need need;
    tr_sym_query_need(v, n, TR_SYM_NEED_CPU, &need);
    int32_t rc = check_need(budget, &need, refusal);
    if (rc) return rc;
    for (uint64_t i = 0; i < n; i++) tr_sym_eval_request(v, &req[i], &out[i]);
    return TR_MATH_OK;
}
int32_t tr_sym_query_bind(const tr_sym_state *s, const tr_sym_action *const *actions, int32_t nactions,
                          const int64_t *functionals, int32_t nfunc, tr_sym_action_view *views, tr_sym_query_view *out) {
    if (!state_ok(s) || !out || nactions < 0 || nfunc < 0 || (nactions && (!actions || !views)) ||
        (nfunc && !functionals)) return TR_MATH_INVALID;
    for (int32_t i = 0; i < nactions; i++) {
        if (!action_ok(actions[i])) return TR_MATH_INVALID;
        if (tr_sym_basis_equal(actions[i]->basis, s->basis) != 1 ||
            (actions[i]->has_geo && !geo_equal(&actions[i]->geo, &s->geo))) return TR_MATH_DOMAIN_ERROR;
        tr_sym_action_get_view(actions[i], &views[i]);
    }
    tr_sym_state_get_view(s, out);
    out->actions = nactions ? views : NULL; out->nactions = nactions;
    out->functionals = nfunc ? functionals : NULL; out->nfunc = nfunc;
    return TR_MATH_OK;
}
int32_t tr_sym_query(const tr_sym_state *s, const tr_sym_action *const *actions, int32_t nactions,
                     const int64_t *functionals, int32_t nfunc, const tr_sym_request *req, uint64_t n,
                     tr_sym_result *out, const tr_math_budget *budget, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (nactions < 0 || nactions > (1 << 20)) return TR_MATH_INVALID;
    tr_sym_action_view *views = nactions ? zalloc((uint64_t)nactions * sizeof *views) : NULL;
    if (nactions && !views) return TR_MATH_MEMORY;
    tr_sym_query_view v;
    int32_t rc = tr_sym_query_bind(s, actions, nactions, functionals, nfunc, views, &v);
    if (!rc) rc = tr_sym_query_view_eval(&v, req, n, out, budget, refusal);
    free(views);
    return rc;
}

/* ================================================================== SYM-002 */
/* ------------------------------------------------------------------ groups */
struct tr_sym_group_s {
    uint32_t magic;
    int32_t ngen, n;
    tr_sym_geometry geo;
    tr_sym_gen gens[TR_SYM_MAX_GENERATORS];
    int64_t *coef, *keys, *targets;
    uint64_t ncoef, nvalues, bytes;
};
static int group_ok(const tr_sym_group *g) { return g && g->magic == GROUP_MAGIC; }
static tr_sym_group *group_fail(tr_sym_group *g, int32_t *error, int32_t code) {
    if (g) { free(g->coef); free(g->keys); free(g->targets); free(g); }
    set_error(error, code);
    return NULL;
}
static uint64_t group_bytes(uint64_t ncoef, uint64_t nvalues) {
    return tr_budget_add(sizeof(tr_sym_group), tr_budget_add(tr_budget_mul(ncoef, 8), tr_budget_mul(nvalues, 16)));
}
tr_sym_group *tr_sym_group_create(const tr_sym_map *const *gens, int32_t ngen, const tr_math_budget *budget,
                                  tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    if (ngen < 1 || !gens) return group_fail(NULL, error, TR_MATH_INVALID);
    if (ngen > TR_SYM_MAX_GENERATORS) return group_fail(NULL, error, TR_MATH_UNSUPPORTED);
    uint64_t ncoef = 0, nvalues = 0;
    const tr_sym_geometry *geo = NULL;
    for (int32_t j = 0; j < ngen; j++) {
        const tr_sym_map *m = gens[j];
        if (!map_ok(m)) return group_fail(NULL, error, TR_MATH_INVALID);
        if (map_geo_free(m)) continue;
        if (geo && !geo_equal(geo, &m->geo)) return group_fail(NULL, error, TR_MATH_DOMAIN_ERROR);
        if (!geo) geo = &m->geo;
    }
    if (!geo) return group_fail(NULL, error, TR_MATH_INVALID);      /* the identity alone declares no geometry */
    for (int32_t j = 0; j < ngen; j++) {
        const tr_sym_map *m = gens[j];
        if (m->n != geo->n) return group_fail(NULL, error, TR_MATH_INVALID);
        if (m->kind == TR_SYM_MAP_VALUES) { nvalues += m->nvalues; continue; }
        /* a map of the integer domain to itself: integral and injective */
        if (m->carrier != TR_SYM_EXACT || !m->integral || m->invertible != 1)
            return group_fail(NULL, error, m->invertible == -1 && m->carrier == TR_SYM_EXACT && m->integral ?
                                                TR_MATH_UNSUPPORTED : TR_MATH_DOMAIN_ERROR);
        ncoef += (uint64_t)m->n * (uint64_t)m->n + (uint64_t)m->n;
    }
    tr_math_need need;
    need_zero(&need);
    need.need[TR_BUDGET_VARIABLES] = (uint64_t)geo->n;
    need.need[TR_BUDGET_SUPPORT] = nvalues;
    need.need[TR_BUDGET_BYTES] = group_bytes(ncoef, nvalues);
    int32_t rc = check_need(budget, &need, refusal);
    if (rc) return group_fail(NULL, error, rc);
    tr_sym_group *g = zalloc(sizeof *g);
    if (!g) return group_fail(NULL, error, TR_MATH_MEMORY);
    g->coef = zalloc(ncoef * 8); g->keys = zalloc(nvalues * 8); g->targets = zalloc(nvalues * 8);
    if (!g->coef || !g->keys || !g->targets) return group_fail(g, error, TR_MATH_MEMORY);
    g->ngen = ngen; g->n = geo->n; g->geo = *geo; g->ncoef = ncoef; g->nvalues = nvalues;
    g->bytes = need.need[TR_BUDGET_BYTES];
    uint64_t ca = 0, va = 0;
    for (int32_t j = 0; j < ngen; j++) {
        const tr_sym_map *m = gens[j];
        tr_sym_gen *G = &g->gens[j];
        if (m->kind == TR_SYM_MAP_VALUES) {
            G->kind = TR_SYM_GEN_VALUES; G->coord = m->coord; G->at = va; G->nvalues = m->nvalues;
            if (m->nvalues) { memcpy(g->keys + va, m->keys, (size_t)m->nvalues * 8); memcpy(g->targets + va, m->targets, (size_t)m->nvalues * 8); }
            va += m->nvalues;
            continue;
        }
        G->kind = TR_SYM_GEN_AFFINE; G->coord = -1; G->at = ca; G->nvalues = 0;
        const int32_t n = m->n;
        for (int32_t i = 0; i < n; i++) for (int32_t k = 0; k < n; k++) g->coef[ca + (uint64_t)i * n + k] = (int64_t)m->a[i][k];
        for (int32_t i = 0; i < n; i++) g->coef[ca + (uint64_t)n * n + i] = (int64_t)m->b[i];
        ca += (uint64_t)n * n + n;
    }
    g->magic = GROUP_MAGIC;
    set_error(error, TR_MATH_OK);
    return g;
}
static tr_sym_group *group_clone(const tr_sym_group *s, int32_t *error) {
    tr_sym_group *g = zalloc(sizeof *g);
    if (!g) return group_fail(NULL, error, TR_MATH_MEMORY);
    *g = *s;
    g->coef = zalloc(s->ncoef * 8); g->keys = zalloc(s->nvalues * 8); g->targets = zalloc(s->nvalues * 8);
    if (!g->coef || !g->keys || !g->targets) { g->magic = 0; return group_fail(g, error, TR_MATH_MEMORY); }
    if (s->ncoef) memcpy(g->coef, s->coef, (size_t)s->ncoef * 8);
    if (s->nvalues) { memcpy(g->keys, s->keys, (size_t)s->nvalues * 8); memcpy(g->targets, s->targets, (size_t)s->nvalues * 8); }
    set_error(error, TR_MATH_OK);
    return g;
}
int32_t tr_sym_group_get_info(const tr_sym_group *g, tr_sym_group_info *out) {
    if (!group_ok(g) || !out) return TR_MATH_INVALID;
    memset(out, 0, sizeof *out);
    out->ngen = g->ngen; out->n = g->n; out->ncoef = g->ncoef; out->nvalues = g->nvalues; out->bytes = g->bytes;
    return TR_MATH_OK;
}
int32_t tr_sym_group_get_view(const tr_sym_group *g, tr_sym_group_view *v) {
    if (!group_ok(g) || !v) return TR_MATH_INVALID;
    memset(v, 0, sizeof *v);
    v->ngen = g->ngen; v->n = g->n; v->gens = g->gens; v->coef = g->coef; v->keys = g->keys; v->targets = g->targets;
    return TR_MATH_OK;
}
void tr_sym_group_free(tr_sym_group *g) {
    if (!group_ok(g)) return;
    g->magic = 0;
    group_fail(g, NULL, 0);
}

/* ------------------------------------------------------------------ supports */
struct tr_sym_support_s {
    uint32_t magic;
    int32_t n, policy;
    tr_sym_geometry geo;
    uint64_t rows, selected, checked, excluded, distinct;
    int64_t *tuples;
    uint64_t *mult, *first;
    tr_cert_record *record;
    uint64_t bytes;
};
static int support_ok(const tr_sym_support *s) { return s && s->magic == SUPPORT_MAGIC; }
static tr_sym_support *support_fail(tr_sym_support *s, int32_t *error, int32_t code) {
    if (s) { free(s->tuples); free(s->mult); free(s->first); tr_cert_record_free(s->record); free(s); }
    set_error(error, code);
    return NULL;
}
int32_t tr_sym_support_need(const tr_scope *scope, const tr_sym_geometry *g, tr_math_need *need) {
    tr_scope_view v;
    if (!scope || !need || tr_sym_geometry_check(g) || tr_scope_get_view(scope, &v)) return TR_MATH_INVALID;
    need_zero(need);
    const uint64_t k = v.selected, n = (uint64_t)g->n;
    need->need[TR_BUDGET_VARIABLES] = n;
    need->need[TR_BUDGET_SUPPORT] = k;
    uint64_t bytes = tr_budget_add(sizeof(tr_sym_support), tr_budget_mul(k, tr_budget_mul(n, 8)));   /* the rows' tuples */
    bytes = tr_budget_add(bytes, tr_budget_mul(k, 24));                       /* row ids, sort index and its buffer */
    bytes = tr_budget_add(bytes, tr_budget_mul(k, tr_budget_add(tr_budget_mul(n, 8), 16)));   /* the support */
    need->need[TR_BUDGET_BYTES] = bytes;
    return TR_MATH_OK;
}
/* a stable bottom-up merge sort of row positions by their tuples */
static void sort_tuples(uint64_t *idx, uint64_t *buf, uint64_t k, const int64_t *T, int32_t n) {
    for (uint64_t w = 1; w < k; w *= 2) {
        for (uint64_t lo = 0; lo < k; lo += 2 * w) {
            uint64_t mid = lo + w < k ? lo + w : k, hi = lo + 2 * w < k ? lo + 2 * w : k, a = lo, b = mid, o = lo;
            while (a < mid && b < hi) {
                if (tr_sym_tuple_cmp(T + idx[b] * (uint64_t)n, T + idx[a] * (uint64_t)n, n) < 0) buf[o++] = idx[b++];
                else buf[o++] = idx[a++];
            }
            while (a < mid) buf[o++] = idx[a++];
            while (b < hi) buf[o++] = idx[b++];
        }
        memcpy(idx, buf, (size_t)k * 8);
    }
}
tr_sym_support *tr_sym_support_build(const tr_scope *scope, const tr_sym_geometry *g, int32_t policy,
                                     const tr_math_budget *budget, tr_math_refusal *refusal, uint64_t *null_row,
                                     int32_t *error) {
    tr_budget_none(refusal);
    if (null_row) *null_row = TR_SYM_NONE;
    tr_scope_view v;
    tr_scope_info info;
    int32_t rc = tr_sym_geometry_check(g);
    if (rc || !scope || tr_scope_get_view(scope, &v) || tr_scope_get_info(scope, &info) ||
        (policy != TR_SYM_NULL_FORBID && policy != TR_SYM_NULL_SKIP)) return support_fail(NULL, error, rc ? rc : TR_MATH_INVALID);
    if (info.digest[0] != g->digest[0] || info.digest[1] != g->digest[1] || info.ncols != g->ncols)
        return support_fail(NULL, error, TR_MATH_DOMAIN_ERROR);
    tr_math_need need;
    tr_sym_support_need(scope, g, &need);
    rc = check_need(budget, &need, refusal);
    if (rc) return support_fail(NULL, error, rc);
    const int32_t n = g->n;
    tr_sym_support *s = zalloc(sizeof *s);
    if (!s) return support_fail(NULL, error, TR_MATH_MEMORY);
    s->n = n; s->policy = policy; s->geo = *g; s->rows = v.rows; s->selected = v.selected;
    const uint64_t K = v.selected;
    int64_t *T = zalloc(tr_budget_mul(K, (uint64_t)n * 8));
    uint64_t *rowid = zalloc(K * 8), *idx = zalloc(K * 8), *buf = zalloc(K * 8);
    if (!T || !rowid || !idx || !buf) { free(T); free(rowid); free(idx); free(buf); return support_fail(s, error, TR_MATH_MEMORY); }
    uint64_t k = 0;
    for (uint64_t r = 0; r < v.rows; r++) {
        if (!tr_scope_is_selected(&v, r)) continue;
        int complete = 1;
        for (int32_t i = 0; i < n; i++) if (!tr_scope_present(&v, g->col[i], r)) { complete = 0; break; }
        if (!complete) {
            if (policy == TR_SYM_NULL_FORBID) {
                if (null_row) *null_row = r;
                free(T); free(rowid); free(idx); free(buf);
                return support_fail(s, error, TR_MATH_DOMAIN_ERROR);
            }
            s->excluded++;
            continue;
        }
        for (int32_t i = 0; i < n; i++) T[k * (uint64_t)n + i] = tr_scope_value(&v, g->col[i], r);
        rowid[k] = r; idx[k] = k;
        k++;
    }
    s->checked = k;
    sort_tuples(idx, buf, k, T, n);
    uint64_t d = 0;
    for (uint64_t i = 0; i < k; i++)
        if (!i || tr_sym_tuple_cmp(T + idx[i] * (uint64_t)n, T + idx[i - 1] * (uint64_t)n, n)) d++;
    s->tuples = zalloc(tr_budget_mul(d, (uint64_t)n * 8)); s->mult = zalloc(d * 8); s->first = zalloc(d * 8);
    if (!s->tuples || !s->mult || !s->first) { free(T); free(rowid); free(idx); free(buf); return support_fail(s, error, TR_MATH_MEMORY); }
    uint64_t t = 0;
    for (uint64_t i = 0; i < k; i++) {
        const int64_t *x = T + idx[i] * (uint64_t)n;
        if (i && !tr_sym_tuple_cmp(x, T + idx[i - 1] * (uint64_t)n, n)) { s->mult[t - 1]++; continue; }
        memcpy(s->tuples + t * (uint64_t)n, x, (size_t)n * 8);
        s->mult[t] = 1; s->first[t] = rowid[idx[i]];       /* the stable sort keeps the smallest row first */
        t++;
    }
    s->distinct = d;
    free(T); free(rowid); free(idx); free(buf);
    s->record = tr_cert_record_scope(scope, g->col, n, TR_CERT_RECORD_SELECTION, NULL, 0, budget, refusal, &rc);
    if (!s->record) return support_fail(s, error, rc ? rc : TR_MATH_MEMORY);
    s->bytes = tr_budget_add(tr_budget_add(sizeof *s, tr_budget_mul(d, (uint64_t)n * 8 + 16)), tr_cert_record_bytes(s->record));
    s->magic = SUPPORT_MAGIC;
    set_error(error, TR_MATH_OK);
    return s;
}
int32_t tr_sym_support_get_info(const tr_sym_support *s, tr_sym_support_info *out) {
    if (!support_ok(s) || !out) return TR_MATH_INVALID;
    memset(out, 0, sizeof *out);
    out->rows = s->rows; out->selected = s->selected; out->checked = s->checked; out->excluded = s->excluded;
    out->distinct = s->distinct; out->bytes = s->bytes; out->n = s->n; out->policy = s->policy;
    return TR_MATH_OK;
}
int32_t tr_sym_support_get_view(const tr_sym_support *s, tr_sym_support_view *v) {
    if (!support_ok(s) || !v) return TR_MATH_INVALID;
    memset(v, 0, sizeof *v);
    v->distinct = s->distinct; v->n = s->n; v->tuples = s->tuples; v->mult = s->mult; v->first = s->first;
    return TR_MATH_OK;
}
const tr_cert_record *tr_sym_support_get_record(const tr_sym_support *s) { return support_ok(s) ? s->record : NULL; }
int32_t tr_sym_support_fresh(const tr_sym_support *s, const tr_scope *current, tr_cert_staleness *out) {
    if (!support_ok(s) || !current || !out) return TR_MATH_INVALID;
    return tr_cert_record_fresh(s->record, current, NULL, 0, TR_CERT_MATCH_EXACT, out);
}
void tr_sym_support_free(tr_sym_support *s) {
    if (!support_ok(s)) return;
    s->magic = 0;
    support_fail(s, NULL, 0);
}

/* ------------------------------------------------------------------ verification */
int32_t tr_sym_verify_need(const tr_sym_support_view *s, const tr_sym_group_view *g, uint64_t ncoef, uint64_t nvalues,
                           int32_t target, tr_math_need *need) {
    if (!need || tr_sym_views_check(s, g, ncoef, nvalues) || (target != TR_SYM_NEED_CPU && target != TR_SYM_NEED_DEVICE))
        return TR_MATH_INVALID;
    tr_sym_verify_needs(s, g, ncoef, nvalues, target, need);
    return TR_MATH_OK;
}
int32_t tr_sym_verify_view(const tr_sym_support_view *s, const tr_sym_group_view *g, uint64_t ncoef, uint64_t nvalues,
                           const tr_math_budget *budget, tr_sym_evidence *out, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (tr_sym_views_check(s, g, ncoef, nvalues) || !out) return TR_MATH_INVALID;
    tr_math_need need;
    tr_sym_verify_needs(s, g, ncoef, nvalues, TR_SYM_NEED_CPU, &need);
    int32_t rc = check_need(budget, &need, refusal);
    if (rc) return rc;
    int64_t y[NC];
    for (int32_t j = 0; j < g->ngen; j++) {
        tr_sym_evidence *e = &out[j];
        memset(e, 0, sizeof *e);
        e->witness = TR_SYM_NONE;
        for (uint64_t t = 0; t < s->distinct; t++) {
            uint64_t im = 0;
            int k = tr_sym_check_tuple(s, g, j, t, y, &im);
            e->rows += s->mult[t]; e->tuples++;
            if (k == TR_SYM_HOLDS) continue;
            e->violations += s->mult[t];
            if (k == TR_SYM_MISSING) e->missing++; else e->broken++;
            if (s->first[t] < e->witness) e->witness = s->first[t];
        }
        rc = tr_sym_evidence_finish(s, g, j, e);
        if (rc) return rc;
    }
    return TR_MATH_OK;
}

/* ------------------------------------------------------------------ certificates */
struct tr_sym_cert_s {
    uint32_t magic;
    int32_t backend, policy, n;
    tr_sym_geometry geo;
    tr_sym_group *group;
    tr_sym_evidence evidence[TR_SYM_MAX_GENERATORS];
    tr_cert_record *record;
    uint64_t rows, selected, checked, excluded, distinct, bytes;
};
static int cert_ok(const tr_sym_cert *c) { return c && c->magic == CERT_MAGIC; }
static tr_sym_cert *cert_fail(tr_sym_cert *c, int32_t *error, int32_t code) {
    if (c) { tr_sym_group_free(c->group); tr_cert_record_free(c->record); free(c); }
    set_error(error, code);
    return NULL;
}
static tr_sym_cert *cert_make(const tr_sym_support *s, const tr_sym_group *g, const tr_sym_evidence *ev, int32_t backend,
                              const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error) {
    tr_math_need need;
    need_zero(&need);
    need.need[TR_BUDGET_BYTES] = tr_budget_add(tr_budget_add(sizeof(tr_sym_cert), g->bytes), tr_cert_record_bytes(s->record));
    int32_t rc = check_need(budget, &need, refusal);
    if (rc) return cert_fail(NULL, error, rc);
    tr_sym_cert *c = zalloc(sizeof *c);
    if (!c) return cert_fail(NULL, error, TR_MATH_MEMORY);
    c->group = group_clone(g, &rc);
    if (!c->group) return cert_fail(c, error, rc);
    c->group->magic = GROUP_MAGIC;
    c->record = tr_cert_record_clone(s->record, &rc);
    if (!c->record) return cert_fail(c, error, rc ? rc : TR_MATH_MEMORY);
    memcpy(c->evidence, ev, (size_t)g->ngen * sizeof *ev);
    c->backend = backend; c->policy = s->policy; c->n = s->n; c->geo = s->geo;
    c->rows = s->rows; c->selected = s->selected; c->checked = s->checked; c->excluded = s->excluded;
    c->distinct = s->distinct; c->bytes = need.need[TR_BUDGET_BYTES];
    c->magic = CERT_MAGIC;
    set_error(error, TR_MATH_OK);
    return c;
}
static int32_t pair_check(const tr_sym_support *s, const tr_sym_group *g) {
    if (!support_ok(s) || !group_ok(g)) return TR_MATH_INVALID;
    if (!geo_equal(&s->geo, &g->geo)) return TR_MATH_DOMAIN_ERROR;     /* the group of another geometry */
    return TR_MATH_OK;
}
tr_sym_cert *tr_sym_certify(const tr_sym_support *s, const tr_sym_group *g, const tr_math_budget *budget,
                            tr_math_refusal *refusal, tr_sym_evidence *evidence, int32_t *error) {
    tr_budget_none(refusal);
    int32_t rc = pair_check(s, g);
    if (rc) return cert_fail(NULL, error, rc);
    tr_sym_support_view sv;
    tr_sym_group_view gv;
    tr_sym_support_get_view(s, &sv);
    tr_sym_group_get_view(g, &gv);
    tr_sym_evidence ev[TR_SYM_MAX_GENERATORS];
    rc = tr_sym_verify_view(&sv, &gv, g->ncoef, g->nvalues, budget, ev, refusal);
    if (rc) return cert_fail(NULL, error, rc);
    if (evidence) memcpy(evidence, ev, (size_t)g->ngen * sizeof *ev);
    for (int32_t j = 0; j < g->ngen; j++) if (!ev[j].holds) return cert_fail(NULL, error, TR_MATH_DOMAIN_ERROR);
    return cert_make(s, g, ev, TR_CERT_BY_CPU, budget, refusal, error);
}
tr_sym_cert *tr_sym_adopt(const tr_sym_support *s, const tr_sym_group *g, const tr_sym_evidence *raw, int32_t backend,
                          const tr_math_budget *budget, tr_math_refusal *refusal, tr_sym_evidence *finished,
                          int32_t *error) {
    tr_budget_none(refusal);
    int32_t rc = pair_check(s, g);
    if (rc || !raw || (backend != TR_CERT_BY_CPU && backend != TR_CERT_BY_DEVICE)) return cert_fail(NULL, error, rc ? rc : TR_MATH_INVALID);
    tr_sym_support_view sv;
    tr_sym_group_view gv;
    tr_sym_support_get_view(s, &sv);
    tr_sym_group_get_view(g, &gv);
    tr_sym_evidence ev[TR_SYM_MAX_GENERATORS];
    memcpy(ev, raw, (size_t)g->ngen * sizeof *ev);
    for (int32_t j = 0; j < g->ngen; j++) {
        tr_sym_evidence check = ev[j];                    /* the raw counts must be complete and consistent */
        if (tr_sym_evidence_finish(&sv, &gv, j, &check)) return cert_fail(NULL, error, TR_MATH_INVALID);
        if (raw[j].witness_tuple != TR_SYM_NONE &&
            (raw[j].witness_tuple != check.witness_tuple || raw[j].kind != check.kind || raw[j].holds != check.holds))
            return cert_fail(NULL, error, TR_MATH_INVALID);
        ev[j] = check;
    }
    if (finished) memcpy(finished, ev, (size_t)g->ngen * sizeof *ev);
    for (int32_t j = 0; j < g->ngen; j++) if (!ev[j].holds) return cert_fail(NULL, error, TR_MATH_DOMAIN_ERROR);
    return cert_make(s, g, ev, backend, budget, refusal, error);
}
int32_t tr_sym_cert_get_info(const tr_sym_cert *c, tr_sym_cert_info *out) {
    if (!cert_ok(c) || !out) return TR_MATH_INVALID;
    memset(out, 0, sizeof *out);
    out->kind = TR_CERT_SNAPSHOT; out->backend = c->backend; out->ngen = c->group->ngen; out->n = c->n;
    out->policy = c->policy; out->holds = 1;
    out->rows = c->rows; out->selected = c->selected; out->checked = c->checked; out->excluded = c->excluded;
    out->distinct = c->distinct; out->bytes = c->bytes;
    return TR_MATH_OK;
}
int32_t tr_sym_cert_get_evidence(const tr_sym_cert *c, tr_sym_evidence *out, int32_t capacity) {
    if (!cert_ok(c) || !out || capacity < c->group->ngen) return TR_MATH_INVALID;
    memcpy(out, c->evidence, (size_t)c->group->ngen * sizeof *out);
    return TR_MATH_OK;
}
const tr_cert_record *tr_sym_cert_get_record(const tr_sym_cert *c) { return cert_ok(c) ? c->record : NULL; }
int32_t tr_sym_cert_fresh(const tr_sym_cert *c, const tr_scope *current, tr_cert_staleness *out) {
    if (!cert_ok(c) || !current || !out) return TR_MATH_INVALID;
    return tr_cert_record_fresh(c->record, current, NULL, 0, TR_CERT_MATCH_EXACT, out);
}
void tr_sym_cert_free(tr_sym_cert *c) {
    if (!cert_ok(c)) return;
    c->magic = 0;
    cert_fail(c, NULL, 0);
}

/* ------------------------------------------------------------------ orbits */
struct tr_sym_orbits_s {
    uint32_t magic;
    int32_t n, reserved;
    tr_sym_geometry geo;
    tr_sym_group *group;
    tr_cert_record *record;
    uint64_t norbits, distinct, rows, largest, smallest;
    uint64_t *rep, *size, *mult, *orbit;
    int64_t *tuples;
    uint64_t bytes;
};
static int orbits_ok(const tr_sym_orbits *o) { return o && o->magic == ORBITS_MAGIC; }
static tr_sym_orbits *orbits_fail(tr_sym_orbits *o, int32_t *error, int32_t code) {
    if (o) {
        tr_sym_group_free(o->group); tr_cert_record_free(o->record);
        free(o->rep); free(o->size); free(o->mult); free(o->orbit); free(o->tuples); free(o);
    }
    set_error(error, code);
    return NULL;
}
static uint64_t uf_find(uint64_t *p, uint64_t x) {
    uint64_t r = x;
    while (p[r] != r) r = p[r];
    while (p[x] != r) { uint64_t nx = p[x]; p[x] = r; x = nx; }
    return r;
}
tr_sym_orbits *tr_sym_orbits_build(const tr_sym_support *s, const tr_sym_cert *c, const tr_scope *current,
                                   const tr_math_budget *budget, tr_math_refusal *refusal, tr_cert_staleness *stale,
                                   int32_t *error) {
    tr_budget_none(refusal);
    tr_cert_staleness st;
    if (!support_ok(s) || !cert_ok(c) || !current) return orbits_fail(NULL, error, TR_MATH_INVALID);
    int32_t rc = tr_cert_record_fresh(c->record, current, NULL, 0, TR_CERT_MATCH_EXACT, &st);
    if (stale) *stale = st;
    if (rc) return orbits_fail(NULL, error, rc);                    /* TR_MATH_STALE: nothing is authorized */
    tr_cert_staleness same;                                         /* the certificate was made for this support */
    if (!geo_equal(&s->geo, &c->geo) || s->distinct != c->distinct || s->checked != c->checked ||
        s->policy != c->policy || tr_cert_record_compare(c->record, s->record, TR_CERT_MATCH_EXACT, &same))
        return orbits_fail(NULL, error, TR_MATH_DOMAIN_ERROR);
    const uint64_t d = s->distinct;
    const int32_t n = s->n;
    tr_math_need need;
    need_zero(&need);
    need.need[TR_BUDGET_SUPPORT] = d;
    need.need[TR_BUDGET_BYTES] = tr_budget_add(tr_budget_add(sizeof(tr_sym_orbits), tr_budget_mul(d, (uint64_t)n * 8 + 40)),
                                               tr_budget_add(c->group->bytes, tr_cert_record_bytes(c->record)));
    rc = check_need(budget, &need, refusal);
    if (rc) return orbits_fail(NULL, error, rc);
    tr_sym_orbits *o = zalloc(sizeof *o);
    if (!o) return orbits_fail(NULL, error, TR_MATH_MEMORY);
    o->n = n; o->geo = s->geo; o->distinct = d;
    o->group = group_clone(c->group, &rc);
    if (o->group) o->group->magic = GROUP_MAGIC;
    o->record = o->group ? tr_cert_record_clone(c->record, &rc) : NULL;
    uint64_t *parent = zalloc(d * 8);
    o->orbit = zalloc(d * 8); o->tuples = zalloc(tr_budget_mul(d, (uint64_t)n * 8));
    if (!o->group || !o->record || !parent || !o->orbit || !o->tuples) { free(parent); return orbits_fail(o, error, TR_MATH_MEMORY); }
    if (d) memcpy(o->tuples, s->tuples, (size_t)d * (size_t)n * 8);
    tr_sym_support_view sv;
    tr_sym_group_view gv;
    tr_sym_support_get_view(s, &sv);
    tr_sym_group_get_view(o->group, &gv);
    for (uint64_t t = 0; t < d; t++) parent[t] = t;
    int64_t y[NC];
    for (int32_t j = 0; j < gv.ngen && !rc; j++)
        for (uint64_t t = 0; t < d; t++) {
            uint64_t im = 0;
            if (tr_sym_check_tuple(&sv, &gv, j, t, y, &im) != TR_SYM_HOLDS) { rc = TR_MATH_INVALID; break; }   /* unreachable */
            uint64_t a = uf_find(parent, t), b = uf_find(parent, tr_sym_find(&sv, y));
            if (a != b) { if (a < b) parent[b] = a; else parent[a] = b; }        /* the smaller index is the root */
        }
    uint64_t norb = 0;
    for (uint64_t t = 0; t < d && !rc; t++) if (uf_find(parent, t) == t) norb++;
    o->rep = zalloc(norb * 8); o->size = zalloc(norb * 8); o->mult = zalloc(norb * 8);
    uint64_t *id = zalloc(d * 8);
    if (!rc && (!o->rep || !o->size || !o->mult || !id)) rc = TR_MATH_MEMORY;
    uint64_t at = 0;
    for (uint64_t t = 0; t < d && !rc; t++) {                     /* roots appear first in increasing index order */
        uint64_t r = uf_find(parent, t);
        if (r == t) { id[t] = at; o->rep[at] = t; o->mult[at] = s->mult[t]; at++; }
        uint64_t k = id[r];
        o->orbit[t] = k; o->size[k]++;
        if (s->mult[t] != o->mult[k]) rc = TR_MATH_INVALID;      /* unreachable under the certificate */
    }
    free(parent); free(id);
    if (rc) return orbits_fail(o, error, rc);
    o->norbits = norb; o->smallest = norb ? UINT64_MAX : 0;
    for (uint64_t k = 0; k < norb; k++) {
        o->rows += o->size[k] * o->mult[k];
        if (o->size[k] > o->largest) o->largest = o->size[k];
        if (o->size[k] < o->smallest) o->smallest = o->size[k];
    }
    o->bytes = need.need[TR_BUDGET_BYTES];
    o->magic = ORBITS_MAGIC;
    set_error(error, TR_MATH_OK);
    return o;
}
int32_t tr_sym_orbits_get_info(const tr_sym_orbits *o, tr_sym_orbits_info *out) {
    if (!orbits_ok(o) || !out) return TR_MATH_INVALID;
    memset(out, 0, sizeof *out);
    out->norbits = o->norbits; out->distinct = o->distinct; out->rows = o->rows; out->largest = o->largest;
    out->smallest = o->smallest; out->bytes = o->bytes; out->n = o->n; out->ngen = o->group->ngen;
    return TR_MATH_OK;
}
int32_t tr_sym_orbits_get_view(const tr_sym_orbits *o, tr_sym_orbits_view *v) {
    if (!orbits_ok(o) || !v) return TR_MATH_INVALID;
    memset(v, 0, sizeof *v);
    v->norbits = o->norbits; v->distinct = o->distinct; v->n = o->n;
    v->rep = o->rep; v->size = o->size; v->mult = o->mult; v->tuples = o->tuples; v->orbit = o->orbit;
    return TR_MATH_OK;
}
void tr_sym_orbits_free(tr_sym_orbits *o) {
    if (!orbits_ok(o)) return;
    o->magic = 0;
    orbits_fail(o, NULL, 0);
}

/* ------------------------------------------------------------------ query polynomials */
/* the canonical terms of a query (keys and int128 coefficients: merged, nonzero, keys increasing); the degree */
static int32_t poly_canonical(const tr_sym_poly_query *q, int32_t n, sterm **out, uint64_t *nt, int32_t *degree) {
    if (!q || q->nterms < 0 || (q->nterms && (!q->exponents || !q->coefficients)) || q->reserved ||
        (q->kind != TR_SYM_SUM && q->kind != TR_SYM_COUNT) ||
        (q->kind == TR_SYM_COUNT ? (q->op < TR_SYM_LT || q->op > TR_SYM_GT) : q->op != 0)) return TR_MATH_INVALID;
    if (q->nterms > TR_SYM_MAX_TERMS) return TR_MATH_UNSUPPORTED;
    sterm *t = zalloc((uint64_t)(q->nterms ? q->nterms : 1) * sizeof *t);
    if (!t) return TR_MATH_MEMORY;
    int32_t deg = 0;
    for (int32_t j = 0; j < q->nterms; j++) {
        const uint8_t *e = q->exponents + (uint64_t)j * (uint64_t)n;
        int32_t d = 0;
        for (int32_t i = 0; i < n; i++) d += e[i];
        if (d > TR_SYM_MAX_DEGREE) { free(t); return TR_MATH_UNSUPPORTED; }
        if (d > deg) deg = d;
        t[j].key = mono_key(e, n); t[j].c = q->coefficients[j];
    }
    qsort(t, (size_t)q->nterms, sizeof *t, sterm_cmp);
    uint64_t m = 0;
    for (int32_t i = 0; i < q->nterms;) {
        i128 s = t[i].c;
        int32_t j = i + 1;
        for (; j < q->nterms && t[j].key == t[i].key; j++) s += t[j].c;      /* at most 4096 int64 terms: exact */
        if (s) { t[m].key = t[i].key; t[m].c = s; m++; }
        i = j;
    }
    *out = t; *nt = m; *degree = deg;
    return TR_MATH_OK;
}
static int sterms_equal(const sterm *a, uint64_t na, const sterm *b, uint64_t nb) {
    if (na != nb) return 0;
    for (uint64_t i = 0; i < na; i++) if (a[i].key != b[i].key || a[i].c != b[i].c) return 0;
    return 1;
}
/* merge an unsorted list in place: sorted, equal keys summed (checked), zeros dropped */
static int32_t sterms_merge(sterm *t, uint64_t k, uint64_t *out) {
    qsort(t, (size_t)k, sizeof *t, sterm_cmp);
    uint64_t m = 0;
    for (uint64_t i = 0; i < k;) {
        i128 s = t[i].c;
        uint64_t j = i + 1;
        for (; j < k && t[j].key == t[i].key; j++) if (!add128(s, t[j].c, &s) || !in_range(s)) return TR_MATH_OVERFLOW;
        if (s) { t[m].key = t[i].key; t[m].c = s; m++; }
        i = j;
    }
    *out = m;
    return TR_MATH_OK;
}
/* the linear forms of an integral affine generator */
static void gen_linforms(const tr_sym_group *g, int32_t j, linform *L) {
    const int32_t n = g->n;
    const int64_t *A = g->coef + g->gens[j].at, *b = A + (uint64_t)n * (uint64_t)n;
    for (int32_t i = 0; i < n; i++) {
        L[i].nl = 0; L[i].exact = 1;
        for (int32_t k = 0; k < n; k++) {
            if (!A[(uint64_t)i * n + k]) continue;
            L[i].var[L[i].nl] = k; L[i].c[L[i].nl] = A[(uint64_t)i * n + k]; L[i].nl++;
        }
        if (b[i]) { L[i].var[L[i].nl] = -1; L[i].c[L[i].nl] = b[i]; L[i].nl++; }
    }
}
/* the expansion bound of p under an affine generator: the largest step (cap) and the accumulated terms (total) */
static void invariance_caps(const tr_sym_group *g, int32_t j, const sterm *p, uint64_t nt, uint64_t *cap,
                            uint64_t *total) {
    const int32_t n = g->n;
    linform L[NC];
    gen_linforms(g, j, L);
    *cap = 1; *total = 0;
    for (uint64_t t = 0; t < nt; t++) {
        uint8_t e[NC];
        key_exps(p[t].key, n, e);
        uint64_t c = expansion_cap(L, e, n);
        if (c > *cap) *cap = c;
        *total = tr_budget_add(*total, c);
    }
}
/* the scratch bytes the invariance proof of p under generator j needs */
static uint64_t invariance_bytes(const tr_sym_group *g, int32_t j, const sterm *p, uint64_t nt) {
    if (g->gens[j].kind == TR_SYM_GEN_VALUES) return tr_budget_mul(tr_budget_mul(nt, 2), sizeof(sterm));
    uint64_t cap, total;
    invariance_caps(g, j, p, nt, &cap, &total);
    return tr_budget_add(tr_budget_mul(cap, 2 * sizeof(sterm)), tr_budget_mul(total, sizeof(sterm)));
}
/* 1 invariant, 0 not, or a native error: p o g == p (affine) / p|x_c=a == p|x_c=sigma(a) (values) */
static int32_t invariant_under(const tr_sym_group *g, int32_t j, const sterm *p, uint64_t nt) {
    const int32_t n = g->n;
    const tr_sym_gen *G = &g->gens[j];
    if (G->kind == TR_SYM_GEN_VALUES) {
        int involves = 0;
        for (uint64_t t = 0; t < nt; t++) if ((p[t].key >> (4 * G->coord)) & 15) involves = 1;
        if (!involves || !G->nvalues) return 1;
        sterm *x = zalloc((nt ? nt : 1) * sizeof *x), *y = zalloc((nt ? nt : 1) * sizeof *y);
        if (!x || !y) { free(x); free(y); return TR_MATH_MEMORY; }
        int32_t out = 1;
        for (uint64_t v = 0; v < G->nvalues && out == 1; v++) {
            int64_t vals[2] = {g->keys[G->at + v], g->targets[G->at + v]};
            sterm *dst[2] = {x, y};
            uint64_t cnt[2];
            for (int s = 0; s < 2 && out == 1; s++) {          /* substitute x_c = value */
                for (uint64_t t = 0; t < nt; t++) {
                    unsigned e = (unsigned)((p[t].key >> (4 * G->coord)) & 15);
                    i128 c = p[t].c;
                    for (unsigned r = 0; r < e; r++) if (!mul128(c, vals[s], &c) || !in_range(c)) { out = TR_MATH_OVERFLOW; break; }
                    dst[s][t].key = p[t].key & ~((u128)15 << (4 * G->coord));
                    dst[s][t].c = c;
                }
                if (out == 1) { int32_t rc = sterms_merge(dst[s], nt, &cnt[s]); if (rc) out = rc; }
            }
            if (out == 1 && !sterms_equal(x, cnt[0], y, cnt[1])) out = 0;
        }
        free(x); free(y);
        return out;
    }
    linform L[NC];
    gen_linforms(g, j, L);
    uint64_t cap = 1, total = 0;
    invariance_caps(g, j, p, nt, &cap, &total);
    sterm *P = zalloc(tr_budget_mul(cap, sizeof *P)), *tmp = zalloc(tr_budget_mul(cap, sizeof *tmp));
    sterm *acc = zalloc(tr_budget_mul(total ? total : 1, sizeof *acc));
    if (!P || !tmp || !acc) { free(P); free(tmp); free(acc); return TR_MATH_MEMORY; }
    uint64_t k = 0;
    int32_t rc = TR_MATH_OK;
    for (uint64_t t = 0; t < nt && !rc; t++) {                     /* p(A x + b) = sum_t c_t prod_i L_i^e_ti */
        uint8_t e[NC];
        key_exps(p[t].key, n, e);
        uint64_t np = 0;
        rc = expand_exact(L, e, n, P, tmp, &np);
        for (uint64_t i = 0; i < np && !rc; i++) {
            acc[k].key = P[i].key;
            if (!mul128(P[i].c, p[t].c, &acc[k].c) || !in_range(acc[k].c)) rc = TR_MATH_OVERFLOW;
            k++;
        }
    }
    uint64_t m = 0;
    if (!rc) rc = sterms_merge(acc, k, &m);
    int32_t out = rc ? rc : sterms_equal(acc, m, p, nt);
    free(P); free(tmp); free(acc);
    return out;
}
static int32_t poly_needs(const tr_sym_group *g, const tr_sym_poly_query *qs, int32_t nq, int32_t n, tr_math_need *need,
                          int with_invariance) {
    need_zero(need);
    need->need[TR_BUDGET_VARIABLES] = (uint64_t)n;
    need->need[TR_BUDGET_OUTPUT] = (uint64_t)nq;
    uint64_t scratch = 0, iters = 0;
    for (int32_t i = 0; i < nq; i++) {
        sterm *p = NULL;
        uint64_t nt = 0;
        int32_t deg = 0;
        int32_t rc = poly_canonical(&qs[i], n, &p, &nt, &deg);
        if (rc) return rc;
        if ((uint64_t)qs[i].nterms > need->need[TR_BUDGET_SUPPORT]) need->need[TR_BUDGET_SUPPORT] = (uint64_t)qs[i].nterms;
        if ((uint64_t)deg > need->need[TR_BUDGET_DEGREE]) need->need[TR_BUDGET_DEGREE] = (uint64_t)deg;
        uint64_t b = tr_budget_mul((uint64_t)(qs[i].nterms ? qs[i].nterms : 1), sizeof(sterm));
        for (int32_t j = 0; with_invariance && g && j < g->ngen; j++) {
            uint64_t x = invariance_bytes(g, j, p, nt);
            if (x > b) b = x;
            if (g->gens[j].kind == TR_SYM_GEN_VALUES) iters = tr_budget_add(iters, tr_budget_mul(g->gens[j].nvalues, nt));
        }
        if (b > scratch) scratch = b;
        free(p);
    }
    need->need[TR_BUDGET_ITERATIONS] = iters;
    need->need[TR_BUDGET_BYTES] = tr_budget_add(scratch, tr_budget_mul((uint64_t)nq, sizeof(tr_sym_poly_answer)));
    return TR_MATH_OK;
}
/* the invariance decision itself (the caller checked the budget) */
static int32_t poly_invariant(const tr_sym_group *g, const tr_sym_poly_query *q, int32_t *generator) {
    sterm *p = NULL;
    uint64_t nt = 0;
    int32_t deg = 0;
    int32_t rc = poly_canonical(q, g->n, &p, &nt, &deg);
    if (rc) return rc;
    int32_t out = 1;
    for (int32_t j = 0; j < g->ngen && out == 1; j++) {
        out = invariant_under(g, j, p, nt);
        if (out == 0 && generator) *generator = j;
    }
    free(p);
    return out;
}
int32_t tr_sym_poly_invariant(const tr_sym_group *g, const tr_sym_poly_query *q, int32_t *generator,
                              const tr_math_budget *budget, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (generator) *generator = -1;
    if (!group_ok(g) || !q) return TR_MATH_INVALID;
    tr_math_need need;
    int32_t rc = poly_needs(g, q, 1, g->n, &need, 1);
    if (!rc) rc = check_need(budget, &need, refusal);
    if (rc) return rc;
    return poly_invariant(g, q, generator);
}
/* the canonical query as an evaluable polynomial (exponent rows and int64 coefficients); e and c are NULL unless
 * it returns TR_MATH_OK */
typedef struct { uint8_t *e; int64_t *c; tr_math_ipoly_view v; } evalpoly;
static int32_t evalpoly_make(const tr_sym_poly_query *q, int32_t n, evalpoly *ep) {
    memset(ep, 0, sizeof *ep);
    sterm *p = NULL;
    uint64_t nt = 0;
    int32_t deg = 0;
    int32_t rc = poly_canonical(q, n, &p, &nt, &deg);
    if (rc) return rc;
    uint8_t *ex = zalloc((nt ? nt : 1) * (uint64_t)n);
    int64_t *co = zalloc((nt ? nt : 1) * 8);
    if (!ex || !co) rc = TR_MATH_MEMORY;
    for (uint64_t t = 0; t < nt && !rc; t++) {
        uint8_t e[NC];
        key_exps(p[t].key, n, e);
        memcpy(ex + t * (uint64_t)n, e, (size_t)n);
        if (p[t].c < INT64_MIN || p[t].c > INT64_MAX) rc = TR_MATH_OVERFLOW;
        else co[t] = (int64_t)p[t].c;
    }
    free(p);
    if (rc) { free(ex); free(co); return rc; }
    ep->e = ex; ep->c = co;
    ep->v.nterms = nt; ep->v.nvars = n; ep->v.degree = deg; ep->v.exponents = ex; ep->v.coefficients = co;
    return TR_MATH_OK;
}
static int pred(int32_t op, i128 v, int64_t c) {
    switch (op) {
    case TR_SYM_LT: return v < c;
    case TR_SYM_LE: return v <= c;
    case TR_SYM_EQ: return v == c;
    case TR_SYM_NE: return v != c;
    case TR_SYM_GE: return v >= c;
    default: return v > c;
    }
}
/* adds weight x p(x) (SUM) or weight [p(x) OP c] (COUNT) to the answer; 0 on overflow */
static int accumulate(const tr_sym_poly_query *q, const evalpoly *ep, const int64_t *x, uint64_t weight,
                      tr_sym_poly_answer *a) {
    __int128 v = 0;
    if (!tr_ipoly_point(&ep->v, x, &v)) return 0;
    a->evaluations++;
    tr_sym_wide t;
    if (q->kind == TR_SYM_COUNT) {
        if (!pred(q->op, v, q->constant)) return 1;
        tr_sym_w_from_i128((i128)weight, &t);
    } else tr_sym_w_mul128((i128)weight, v, &t);
    return tr_sym_w_add(&a->value, &t, &a->value);
}
int32_t tr_sym_orbit_query(const tr_sym_orbits *o, const tr_scope *current, const tr_sym_poly_query *qs, int32_t nq,
                           tr_sym_poly_answer *out, const tr_math_budget *budget, tr_math_refusal *refusal,
                           tr_cert_staleness *stale) {
    tr_budget_none(refusal);
    if (!orbits_ok(o) || !current || nq < 0 || (nq && (!qs || !out))) return TR_MATH_INVALID;
    tr_cert_staleness st;
    int32_t rc = tr_cert_record_fresh(o->record, current, NULL, 0, TR_CERT_MATCH_EXACT, &st);
    if (stale) *stale = st;
    if (rc) return rc;
    tr_math_need need;
    rc = poly_needs(o->group, qs, nq, o->n, &need, 1);
    if (!rc) rc = check_need(budget, &need, refusal);
    if (rc) return rc;
    for (int32_t i = 0; i < nq; i++) {
        tr_sym_poly_answer *a = &out[i];
        memset(a, 0, sizeof *a);
        a->generator = -1;
        int32_t inv = poly_invariant(o->group, &qs[i], &a->generator);
        if (inv < 0) { a->status = inv; continue; }
        if (!inv) { a->status = TR_MATH_DOMAIN_ERROR; continue; }   /* not invariant: refused under the reduction */
        a->invariant = 1;
        evalpoly ep;
        rc = evalpoly_make(&qs[i], o->n, &ep);
        if (rc) { a->status = rc; continue; }
        for (uint64_t k = 0; k < o->norbits; k++) {
            uint64_t w = o->size[k] * o->mult[k];
            a->rows += w;
            if (!accumulate(&qs[i], &ep, o->tuples + o->rep[k] * (uint64_t)o->n, w, a)) { a->status = TR_MATH_OVERFLOW; break; }
        }
        free(ep.e); free(ep.c);
        if (a->status) tr_sym_w_zero(&a->value);
    }
    return TR_MATH_OK;
}
int32_t tr_sym_rows_query(const tr_scope *scope, const tr_sym_geometry *g, int32_t policy, const tr_sym_poly_query *qs,
                          int32_t nq, tr_sym_poly_answer *out, const tr_math_budget *budget, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    tr_scope_view v;
    tr_scope_info info;
    int32_t rc = tr_sym_geometry_check(g);
    if (rc || !scope || tr_scope_get_view(scope, &v) || tr_scope_get_info(scope, &info) || nq < 0 ||
        (nq && (!qs || !out)) || (policy != TR_SYM_NULL_FORBID && policy != TR_SYM_NULL_SKIP))
        return rc ? rc : TR_MATH_INVALID;
    if (info.digest[0] != g->digest[0] || info.digest[1] != g->digest[1]) return TR_MATH_DOMAIN_ERROR;
    tr_math_need need;
    rc = poly_needs(NULL, qs, nq, g->n, &need, 0);
    if (!rc) rc = check_need(budget, &need, refusal);
    if (rc) return rc;
    const int32_t n = g->n;
    evalpoly *ep = zalloc((uint64_t)(nq ? nq : 1) * sizeof *ep);
    if (!ep) return TR_MATH_MEMORY;
    for (int32_t i = 0; i < nq; i++) {
        memset(&out[i], 0, sizeof out[i]);
        out[i].generator = -1;
        out[i].status = evalpoly_make(&qs[i], n, &ep[i]);
    }
    int64_t x[NC];
    for (uint64_t r = 0; r < v.rows && !rc; r++) {
        if (!tr_scope_is_selected(&v, r)) continue;
        int complete = 1;
        for (int32_t i = 0; i < n; i++) if (!tr_scope_present(&v, g->col[i], r)) { complete = 0; break; }
        if (!complete) {
            if (policy == TR_SYM_NULL_FORBID) rc = TR_MATH_DOMAIN_ERROR;
            continue;
        }
        for (int32_t i = 0; i < n; i++) x[i] = tr_scope_value(&v, g->col[i], r);
        for (int32_t i = 0; i < nq; i++) {
            if (out[i].status) continue;
            out[i].rows++;
            if (!accumulate(&qs[i], &ep[i], x, 1, &out[i])) { out[i].status = TR_MATH_OVERFLOW; tr_sym_w_zero(&out[i].value); }
        }
    }
    for (int32_t i = 0; i < nq; i++) { free(ep[i].e); free(ep[i].c); }
    free(ep);
    return rc;
}
