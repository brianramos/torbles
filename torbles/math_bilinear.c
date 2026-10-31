/* DELTA-002: bilinear (join) and nonlinear update laws over DELTA-001 states; see math_bilinear.h.
 * SPDX-License-Identifier: Apache-2.0 */
#include "math_bilinear.h"
#include "mathematics.h"
#include "math_domain.h"
#include "math_handles.h"
#include <stdlib.h>
#include <string.h>

TR_MATH_ABI(tr_bilin_wide, 1);           /* the ctypes boundary of DELTA-002 (math_handles.h) */
TR_MATH_ABI(tr_bilin_spec, 1);
TR_MATH_ABI(tr_bilin_info, 1);
TR_MATH_ABI(tr_bilin_cross_view, 1);
TR_MATH_ABI(tr_bilin_requirement, 1);
TR_MATH_ABI(tr_bilin_values_info, 1);
TR_MATH_ABI(tr_bilin_group_values, 1);
TR_MATH_ABI(tr_bilin_violation, 1);

_Static_assert(sizeof(tr_bilin_wide) == 40, "tr_bilin_wide ABI");
_Static_assert(sizeof(tr_bilin_spec) == 40, "tr_bilin_spec ABI");
_Static_assert(sizeof(tr_bilin_info) == 144, "tr_bilin_info ABI");
_Static_assert(sizeof(tr_bilin_cross_view) == 160, "tr_bilin_cross_view ABI");
_Static_assert(sizeof(tr_bilin_requirement) == 64, "tr_bilin_requirement ABI");
_Static_assert(sizeof(tr_bilin_values_info) == 80, "tr_bilin_values_info ABI");
_Static_assert(sizeof(tr_bilin_group_values) == 80, "tr_bilin_group_values ABI");
_Static_assert(sizeof(tr_bilin_violation) == 40, "tr_bilin_violation ABI");

#define JOIN_MAGIC UINT32_C(0x4e494f4a)
#define JOB_MAGIC UINT32_C(0x424f4a42)
#define VALUES_MAGIC UINT32_C(0x534c4156)
#define W TR_BILIN_WORDS
#define MAXC TR_SCOPE_MAX_COLUMNS
enum { JOB_BUILD = 1, JOB_UPDATE = 2 };

struct tr_bilin_join_s {
    uint32_t magic;
    tr_bilin_spec spec;
    int32_t values, terms_a, terms_b;
    uint64_t ngroups, step, bytes, touched;
    uint64_t identity[2], base[2], side_a[2], side_b[2];
    int32_t *pairs;
    tr_bilin_wide *vals;              /* ngroups x values */
    tr_bilin_wide *terms;             /* 3 x ngroups x values */
};

struct tr_bilin_job_s {
    uint32_t magic;
    int32_t kind;
    tr_bilin_spec spec;
    int32_t values, terms_a, terms_b;
    uint64_t ngroups, keys, bytes;
    int32_t *pairs;
    uint64_t *rows[4], *count[4];     /* owned gathered cells of an update: A old, A new, B old, B new */
    tr_math_i128 *sum[4];
    tr_bilin_cross_view view;
    tr_bilin_wide *old_vals;          /* an update: the old join's values */
    uint64_t old_identity[2], base[2], step;
    uint64_t side_a[2], side_b[2];    /* the identities of the new side states */
};

struct tr_bilin_values_s {
    uint32_t magic;
    int32_t kind, column, ncols, scale;
    int32_t scales[MAXC];
    uint64_t ngroups, entries, step, bytes;
    uint64_t identity[2], base[2];
    int64_t *rows, *nulls, *present;  /* per group */
    uint64_t *offset;                 /* MULTISET: ngroups + 1 */
    int64_t *vals, *mult;             /* MULTISET: entries */
    int64_t *mn, *mnc, *mx, *mxc;     /* EXTREMUM: per group */
};

static void set_error(int32_t *error, int32_t code) { if (error) *error = code; }
static void *zalloc(uint64_t bytes) {
    if (bytes > (uint64_t)SIZE_MAX) return NULL;
    return calloc(1, bytes ? (size_t)bytes : 1);
}
static const tr_math_budget *budget_or(const tr_math_budget *b, tr_math_budget *dflt) {
    if (b) return b;
    tr_budget_default(dflt);
    return dflt;
}
static void need_zero(tr_math_need *n) { for (int i = 0; i < TR_BUDGET_RESOURCES; i++) n->need[i] = 0; }
static int join_valid(const tr_bilin_join *j) { return j && j->magic == JOIN_MAGIC; }
static int job_valid(const tr_bilin_job *j) { return j && j->magic == JOB_MAGIC; }
static int values_valid(const tr_bilin_values *v) { return v && v->magic == VALUES_MAGIC; }

/* ------------------------------------------------------------------ digests (identities; never an equality test) */
typedef struct { uint64_t h0, h1; } hasher;
static void h_init(hasher *h, uint64_t tag) {
    h->h0 = UINT64_C(0x452821E638D01377) ^ tag; h->h1 = UINT64_C(0xBE5466CF34E90C6C) + tag;
}
static void h_word(hasher *h, uint64_t w) {
    h->h0 = (h->h0 ^ w) * UINT64_C(0x9E3779B97F4A7C15); h->h0 ^= h->h0 >> 29;
    h->h1 = (h->h1 + w) * UINT64_C(0xBF58476D1CE4E5B9); h->h1 ^= h->h1 >> 31; h->h1 += UINT64_C(0x94D049BB133111EB);
}
static void h_bytes(hasher *h, const void *p, uint64_t len) {
    const unsigned char *b = (const unsigned char *)p;
    h_word(h, len);
    for (; len >= 8; len -= 8, b += 8) { uint64_t w; memcpy(&w, b, 8); h_word(h, w); }
    if (len) { uint64_t w = 0; memcpy(&w, b, (size_t)len); h_word(h, w); }
}
static void h_out(const hasher *h, uint64_t out[2]) { out[0] = h->h0 ^ (h->h1 >> 17); out[1] = h->h1 ^ (h->h0 << 13); }

/* ================================================================== join states */
static uint64_t join_bytes(uint64_t ngroups, int32_t values, int32_t npairs) {
    uint64_t nv = tr_budget_mul(ngroups, (uint64_t)values);
    return tr_budget_add(sizeof(tr_bilin_join), tr_budget_add(tr_budget_mul(tr_budget_mul(nv, 4), sizeof(tr_bilin_wide)),
                                                              tr_budget_mul((uint64_t)npairs, 8)));
}
/* the spec against the side states: shapes, pairs, keys */
static int32_t spec_check(const tr_delta_state *a, const tr_delta_state *b, const tr_bilin_spec *spec,
                          const int32_t *pairs, tr_delta_info *ia, tr_delta_info *ib) {
    if (!a || !b || !spec || tr_delta_state_get_info(a, ia) || tr_delta_state_get_info(b, ib)) return TR_MATH_INVALID;
    if (spec->nkeys < 1 || spec->groups_a < 1 || spec->groups_b < 1 || spec->reserved != 0 || spec->npairs < 1 ||
        !pairs || (spec->null_key != TR_BILIN_NONE && spec->null_key >= spec->nkeys)) return TR_MATH_INVALID;
    if (spec->npairs > TR_BILIN_MAX_PAIRS) return TR_MATH_UNSUPPORTED;
    uint64_t ga = tr_budget_mul(spec->nkeys, spec->groups_a), gb = tr_budget_mul(spec->nkeys, spec->groups_b);
    if (ga != ia->ngroups || gb != ib->ngroups) return TR_MATH_DOMAIN_ERROR;    /* not key-major states of these keys */
    if (tr_budget_mul(spec->groups_a, spec->groups_b) > TR_SCOPE_MAX_GROUPS) return TR_MATH_UNSUPPORTED;
    for (int32_t p = 0; p < spec->npairs; p++)
        if (pairs[2 * p] < 0 || pairs[2 * p] >= ia->nterms || pairs[2 * p + 1] < 0 || pairs[2 * p + 1] >= ib->nterms)
            return TR_MATH_INVALID;
    return TR_MATH_OK;
}
static uint64_t job_bytes(int32_t kind, uint64_t keys, const tr_bilin_spec *spec, int32_t ta, int32_t tb, int a_changes,
                          int b_changes, uint64_t ngroups, int32_t values) {
    uint64_t b = tr_budget_add(sizeof(tr_bilin_job), tr_budget_mul((uint64_t)spec->npairs, 8));
    if (kind == JOB_UPDATE) {
        uint64_t ca = tr_bank_state_bytes(tr_budget_mul(keys, spec->groups_a), ta);
        uint64_t cb = tr_bank_state_bytes(tr_budget_mul(keys, spec->groups_b), tb);
        b = tr_budget_add(b, tr_budget_add(tr_budget_mul(ca, a_changes ? 2 : 1), tr_budget_mul(cb, b_changes ? 2 : 1)));
        b = tr_budget_add(b, tr_budget_mul(tr_budget_mul(ngroups, (uint64_t)values), sizeof(tr_bilin_wide)));   /* old values */
    }
    return b;
}
static void join_need_fill(uint64_t keys, const tr_bilin_spec *spec, int32_t values, uint64_t extra_bytes,
                           tr_math_need *need) {
    need_zero(need);
    uint64_t ngroups = spec->groups_a * spec->groups_b, nv = tr_budget_mul(ngroups, (uint64_t)values);
    need->need[TR_BUDGET_SUPPORT] = (uint64_t)spec->npairs;
    need->need[TR_BUDGET_OUTPUT] = ngroups;
    need->need[TR_BUDGET_STATES] = nv;
    need->need[TR_BUDGET_ITERATIONS] = tr_budget_mul(tr_budget_mul(keys, ngroups), (uint64_t)values);
    /* the new join, the terms scratch of the CPU evaluator and the job */
    uint64_t bytes = tr_budget_add(join_bytes(ngroups, values, spec->npairs), tr_budget_mul(tr_budget_mul(nv, 3), sizeof(tr_bilin_wide)));
    need->need[TR_BUDGET_BYTES] = tr_budget_add(bytes, extra_bytes);
}

/* ------------------------------------------------------------------ jobs */
static void job_release(tr_bilin_job *j) {
    if (!j) return;
    free(j->pairs);
    for (int i = 0; i < 4; i++) { free(j->rows[i]); free(j->count[i]); free(j->sum[i]); }
    free(j->old_vals);
    j->magic = 0;
    free(j);
}
void tr_bilin_job_free(tr_bilin_job *j) { if (job_valid(j)) job_release(j); }

static void view_common(tr_bilin_job *j) {
    tr_bilin_cross_view *v = &j->view;
    v->keys = j->keys; v->groups_a = j->spec.groups_a; v->groups_b = j->spec.groups_b; v->ngroups = j->ngroups;
    v->skip = TR_BILIN_NONE; v->npairs = j->spec.npairs; v->values = j->values; v->terms_a = j->terms_a;
    v->terms_b = j->terms_b; v->pairs = j->pairs;
}

static tr_bilin_job *job_build(const tr_delta_state *a, const tr_delta_state *b, const tr_bilin_spec *spec,
                               const int32_t *pairs, int32_t target, const tr_math_budget *budget,
                               tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    tr_delta_info ia, ib;
    tr_math_need need;
    int32_t rc = target == 0 || target == 1 ? TR_MATH_OK : TR_MATH_INVALID;
    if (!rc) rc = tr_bilin_join_need(a, b, spec, pairs, target, &need);
    tr_math_budget dflt;
    if (!rc) rc = tr_budget_check(budget_or(budget, &dflt), &need, refusal);
    if (rc) { set_error(error, rc); return NULL; }
    spec_check(a, b, spec, pairs, &ia, &ib);
    tr_bilin_job *j = zalloc(sizeof *j);
    if (!j || !(j->pairs = zalloc((uint64_t)spec->npairs * 8))) { free(j); set_error(error, TR_MATH_MEMORY); return NULL; }
    j->kind = JOB_BUILD; j->spec = *spec; j->values = 1 + 2 * spec->npairs; j->terms_a = ia.nterms; j->terms_b = ib.nterms;
    j->ngroups = spec->groups_a * spec->groups_b; j->keys = spec->nkeys;
    memcpy(j->pairs, pairs, (size_t)spec->npairs * 8);
    j->side_a[0] = ia.identity[0]; j->side_a[1] = ia.identity[1]; j->side_b[0] = ib.identity[0]; j->side_b[1] = ib.identity[1];
    view_common(j);
    tr_bank_states sa, sb;
    tr_delta_state_get_states(a, &sa);
    tr_delta_state_get_states(b, &sb);
    j->view.a_rows[1] = sa.rows; j->view.a_count[1] = sa.count; j->view.a_sum[1] = sa.sum;   /* borrowed */
    j->view.b_rows[1] = sb.rows; j->view.b_count[1] = sb.count; j->view.b_sum[1] = sb.sum;
    j->view.skip = spec->null_key;                       /* compact index == key id in a build */
    j->bytes = job_bytes(JOB_BUILD, j->keys, spec, j->terms_a, j->terms_b, 0, 0, j->ngroups, j->values);
    j->magic = JOB_MAGIC;
    set_error(error, TR_MATH_OK);
    return j;
}

int32_t tr_bilin_join_need(const tr_delta_state *a, const tr_delta_state *b, const tr_bilin_spec *spec,
                           const int32_t *pairs, int32_t target, tr_math_need *need) {
    tr_delta_info ia, ib;
    if (!need || (target != 0 && target != 1)) return TR_MATH_INVALID;
    int32_t rc = spec_check(a, b, spec, pairs, &ia, &ib);
    if (rc) return rc;
    int32_t values = 1 + 2 * spec->npairs;
    uint64_t extra = job_bytes(JOB_BUILD, spec->nkeys, spec, ia.nterms, ib.nterms, 0, 0, 0, values);
    if (target == 1) {
        tr_bilin_cross_view v;
        memset(&v, 0, sizeof v);
        tr_bank_states sa, sb;
        tr_delta_state_get_states(a, &sa);
        tr_delta_state_get_states(b, &sb);
        v.keys = spec->nkeys; v.groups_a = spec->groups_a; v.groups_b = spec->groups_b;
        v.ngroups = spec->groups_a * spec->groups_b; v.npairs = spec->npairs; v.values = values;
        v.terms_a = ia.nterms; v.terms_b = ib.nterms; v.a_rows[1] = sa.rows; v.b_rows[1] = sb.rows;
        extra = tr_budget_add(extra, tr_bilin_cross_device_bytes(&v));
    }
    join_need_fill(spec->nkeys, spec, values, extra, need);
    return TR_MATH_OK;
}

tr_bilin_job *tr_bilin_job_build(const tr_delta_state *a, const tr_delta_state *b, const tr_bilin_spec *spec,
                                 const int32_t *pairs, int32_t target, const tr_math_budget *budget,
                                 tr_math_refusal *refusal, int32_t *error) {
    return job_build(a, b, spec, pairs, target, budget, refusal, error);
}

/* the distinct keys of a delta's touched groups (increasing groups give nondecreasing keys; per cells per key) */
static uint64_t keys_of(const tr_delta *d, uint64_t per, uint64_t *out) {
    tr_delta_rows r;
    tr_delta_get_rows(d, &r);
    uint64_t n = 0;
    for (uint64_t u = 0; u < r.touched; u++) {
        uint64_t k = (uint64_t)r.groups[u] / per;
        if (!n || out[n - 1] != k) out[n++] = k;
    }
    return n;
}
/* the union of two increasing key lists, without the null key */
static uint64_t merge_keys(const uint64_t *a, uint64_t na, const uint64_t *b, uint64_t nb, uint64_t skip, uint64_t *out) {
    uint64_t i = 0, j = 0, n = 0;
    while (i < na || j < nb) {
        uint64_t k = j >= nb || (i < na && a[i] < b[j]) ? a[i] : b[j];
        while (i < na && a[i] == k) i++;
        while (j < nb && b[j] == k) j++;
        if (k != skip) out[n++] = k;
    }
    return n;
}

/* one changed side's successor check: a2 is a updated by d (same base, step + 1, same shape, the cells outside d's
 * touched groups unchanged) */
static int32_t successor_check(const tr_delta_state *s, const tr_delta *d, const tr_delta_state *s2) {
    tr_delta_info i1, i2;
    tr_delta_batch_info bi;
    if (tr_delta_state_get_info(s, &i1) || tr_delta_state_get_info(s2, &i2) || tr_delta_get_info(d, &bi))
        return TR_MATH_INVALID;
    if (i2.base[0] != i1.base[0] || i2.base[1] != i1.base[1] || i2.step != i1.step + 1 || i2.ngroups != i1.ngroups ||
        i2.nterms != i1.nterms || i2.ncols != i1.ncols || i2.mode != i1.mode || bi.ngroups != i1.ngroups)
        return TR_MATH_DOMAIN_ERROR;
    tr_bank_states a, b;
    tr_delta_rows r;
    tr_delta_state_get_states(s, &a);
    tr_delta_state_get_states(s2, &b);
    tr_delta_get_rows(d, &r);
    const uint64_t m = (uint64_t)i1.nterms;
    uint64_t lo = 0;
    for (uint64_t u = 0; u <= r.touched; u++) {
        uint64_t hi = u < r.touched ? r.groups[u] : i1.ngroups;
        if (hi > lo) {
            if (memcmp(a.rows + lo, b.rows + lo, (size_t)(hi - lo) * 8) ||
                (m && (memcmp(a.count + lo * m, b.count + lo * m, (size_t)((hi - lo) * m * 8)) ||
                       memcmp(a.sum + lo * m, b.sum + lo * m, (size_t)((hi - lo) * m * 16)))))
                return TR_MATH_DOMAIN_ERROR;              /* a change outside the delta's groups */
        }
        lo = hi + 1;
    }
    return TR_MATH_OK;
}

/* the checks of an update and its touched keys (allocated into *keys_out when keys_out != NULL) */
static int32_t update_check(const tr_bilin_join *j, const tr_delta_state *a, const tr_delta *da,
                            const tr_delta_state *a2, const tr_delta_state *b, const tr_delta *db,
                            const tr_delta_state *b2, uint64_t *nkeys_out, uint64_t **keys_out) {
    if (keys_out) *keys_out = NULL;
    if (!join_valid(j) || !a || !b || (!da) != (!a2) || (!db) != (!b2) || (!da && !db)) return TR_MATH_INVALID;
    tr_delta_info ia, ib;
    if (tr_delta_state_get_info(a, &ia) || tr_delta_state_get_info(b, &ib)) return TR_MATH_INVALID;
    if (ia.identity[0] != j->side_a[0] || ia.identity[1] != j->side_a[1] || ib.identity[0] != j->side_b[0] ||
        ib.identity[1] != j->side_b[1]) return TR_MATH_DOMAIN_ERROR;            /* not the states it summarizes */
    int32_t rc = TR_MATH_OK;
    if (da) rc = successor_check(a, da, a2);
    if (!rc && db) rc = successor_check(b, db, b2);
    if (rc) return rc;
    tr_delta_batch_info ba, bb;
    memset(&ba, 0, sizeof ba); memset(&bb, 0, sizeof bb);
    if (da) tr_delta_get_info(da, &ba);
    if (db) tr_delta_get_info(db, &bb);
    uint64_t *ka = zalloc(tr_budget_mul(ba.touched, 8)), *kb = zalloc(tr_budget_mul(bb.touched, 8));
    uint64_t *keys = zalloc(tr_budget_mul(ba.touched + bb.touched, 8));
    if (!ka || !kb || !keys) { free(ka); free(kb); free(keys); return TR_MATH_MEMORY; }
    uint64_t na = da ? keys_of(da, j->spec.groups_a, ka) : 0, nb = db ? keys_of(db, j->spec.groups_b, kb) : 0;
    uint64_t n = merge_keys(ka, na, kb, nb, j->spec.null_key, keys);   /* the null key never matches: not gathered */
    free(ka); free(kb);
    *nkeys_out = n;
    if (keys_out) *keys_out = keys; else free(keys);
    return TR_MATH_OK;
}

static void update_need_fill(const tr_bilin_join *j, uint64_t keys, int a_changes, int b_changes, int32_t target,
                             tr_math_need *need) {
    uint64_t extra = job_bytes(JOB_UPDATE, keys, &j->spec, j->terms_a, j->terms_b, a_changes, b_changes, j->ngroups,
                               j->values);
    if (target == 1) {
        tr_bilin_cross_view v;
        memset(&v, 0, sizeof v);
        static const uint64_t marker = 0;
        v.keys = keys; v.groups_a = j->spec.groups_a; v.groups_b = j->spec.groups_b; v.ngroups = j->ngroups;
        v.npairs = j->spec.npairs; v.values = j->values; v.terms_a = j->terms_a; v.terms_b = j->terms_b;
        v.a_rows[0] = &marker; v.b_rows[0] = &marker;      /* only which arrays exist matters to the byte count */
        if (a_changes) v.a_rows[1] = &marker;
        if (b_changes) v.b_rows[1] = &marker;
        extra = tr_budget_add(extra, tr_bilin_cross_device_bytes(&v));
    }
    join_need_fill(keys, &j->spec, j->values, extra, need);
}

int32_t tr_bilin_update_need(const tr_bilin_join *j, const tr_delta_state *a, const tr_delta *da,
                             const tr_delta_state *a2, const tr_delta_state *b, const tr_delta *db,
                             const tr_delta_state *b2, int32_t target, tr_math_need *need) {
    if (!need || (target != 0 && target != 1)) return TR_MATH_INVALID;
    uint64_t keys = 0;
    int32_t rc = update_check(j, a, da, a2, b, db, b2, &keys, NULL);
    if (rc) return rc;
    update_need_fill(j, keys, da != NULL, db != NULL, target, need);
    return TR_MATH_OK;
}

/* gathers the cells of the compact keys of one side and version into owned arrays (slot k of the job) */
static int32_t gather(tr_bilin_job *j, int k, const tr_delta_state *s, const uint64_t *keys, uint64_t per, int32_t m) {
    tr_bank_states st;
    tr_delta_state_get_states(s, &st);
    uint64_t cells = j->keys * per, mm = (uint64_t)m;
    j->rows[k] = zalloc(cells * 8); j->count[k] = zalloc(cells * mm * 8); j->sum[k] = zalloc(cells * mm * 16);
    if (!j->rows[k] || !j->count[k] || !j->sum[k]) return TR_MATH_MEMORY;
    for (uint64_t u = 0; u < j->keys; u++) {
        uint64_t g0 = keys[u] * per;
        memcpy(j->rows[k] + u * per, st.rows + g0, (size_t)per * 8);
        if (mm) {
            memcpy(j->count[k] + u * per * mm, st.count + g0 * mm, (size_t)(per * mm * 8));
            memcpy(j->sum[k] + u * per * mm, st.sum + g0 * mm, (size_t)(per * mm * 16));
        }
    }
    return TR_MATH_OK;
}

tr_bilin_job *tr_bilin_job_update(const tr_bilin_join *jn, const tr_delta_state *a, const tr_delta *da,
                                  const tr_delta_state *a2, const tr_delta_state *b, const tr_delta *db,
                                  const tr_delta_state *b2, int32_t target, const tr_math_budget *budget,
                                  tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    if (target != 0 && target != 1) { set_error(error, TR_MATH_INVALID); return NULL; }
    uint64_t nkeys = 0, *keys = NULL;
    int32_t rc = update_check(jn, a, da, a2, b, db, b2, &nkeys, &keys);
    if (rc) { set_error(error, rc); return NULL; }
    tr_math_need need;
    update_need_fill(jn, nkeys, da != NULL, db != NULL, target, &need);
    tr_math_budget dflt;
    rc = tr_budget_check(budget_or(budget, &dflt), &need, refusal);
    if (rc) { free(keys); set_error(error, rc); return NULL; }
    tr_bilin_job *j = zalloc(sizeof *j);
    if (!j) { free(keys); set_error(error, TR_MATH_MEMORY); return NULL; }
    j->kind = JOB_UPDATE; j->spec = jn->spec; j->values = jn->values; j->terms_a = jn->terms_a; j->terms_b = jn->terms_b;
    j->ngroups = jn->ngroups; j->keys = nkeys;
    uint64_t nv = jn->ngroups * (uint64_t)jn->values;
    j->pairs = zalloc((uint64_t)jn->spec.npairs * 8);
    j->old_vals = zalloc(nv * sizeof(tr_bilin_wide));
    rc = (!j->pairs || !j->old_vals) ? TR_MATH_MEMORY : TR_MATH_OK;
    if (!rc) {
        memcpy(j->pairs, jn->pairs, (size_t)jn->spec.npairs * 8);
        memcpy(j->old_vals, jn->vals, (size_t)nv * sizeof(tr_bilin_wide));
        rc = gather(j, 0, a, keys, jn->spec.groups_a, jn->terms_a);
    }
    if (!rc && da) rc = gather(j, 1, a2, keys, jn->spec.groups_a, jn->terms_a);
    if (!rc) rc = gather(j, 2, b, keys, jn->spec.groups_b, jn->terms_b);
    if (!rc && db) rc = gather(j, 3, b2, keys, jn->spec.groups_b, jn->terms_b);
    free(keys);
    if (rc) { job_release(j); set_error(error, rc); return NULL; }
    view_common(j);
    tr_bilin_cross_view *v = &j->view;
    for (int ver = 0; ver < 2; ver++) {
        v->a_rows[ver] = j->rows[ver]; v->a_count[ver] = j->count[ver]; v->a_sum[ver] = j->sum[ver];
        v->b_rows[ver] = j->rows[2 + ver]; v->b_count[ver] = j->count[2 + ver]; v->b_sum[ver] = j->sum[2 + ver];
    }
    tr_delta_info i2a, i2b;
    tr_delta_state_get_info(a2 ? a2 : a, &i2a);
    tr_delta_state_get_info(b2 ? b2 : b, &i2b);
    j->side_a[0] = i2a.identity[0]; j->side_a[1] = i2a.identity[1]; j->side_b[0] = i2b.identity[0]; j->side_b[1] = i2b.identity[1];
    j->old_identity[0] = jn->identity[0]; j->old_identity[1] = jn->identity[1];
    j->base[0] = jn->base[0]; j->base[1] = jn->base[1]; j->step = jn->step;
    j->bytes = job_bytes(JOB_UPDATE, nkeys, &j->spec, j->terms_a, j->terms_b, da != NULL, db != NULL, j->ngroups, j->values);
    j->magic = JOB_MAGIC;
    set_error(error, TR_MATH_OK);
    return j;
}

int32_t tr_bilin_job_get_view(const tr_bilin_job *j, tr_bilin_cross_view *out) {
    if (!job_valid(j) || !out) return TR_MATH_INVALID;
    *out = j->view;
    return TR_MATH_OK;
}

static int view_ok(const tr_bilin_cross_view *v) {
    if (!v || v->groups_a < 1 || v->groups_b < 1 || v->npairs < 1 || v->npairs > TR_BILIN_MAX_PAIRS ||
        v->values != 1 + 2 * v->npairs || v->terms_a < 0 || v->terms_b < 0 || !v->pairs ||
        v->ngroups != v->groups_a * v->groups_b) return 0;
    for (int ver = 0; ver < 2; ver++) {
        if (v->a_rows[ver] && v->terms_a && (!v->a_count[ver] || !v->a_sum[ver])) return 0;
        if (v->b_rows[ver] && v->terms_b && (!v->b_count[ver] || !v->b_sum[ver])) return 0;
    }
    for (int32_t p = 0; p < v->npairs; p++)
        if (v->pairs[2 * p] < 0 || v->pairs[2 * p] >= v->terms_a || v->pairs[2 * p + 1] < 0 ||
            v->pairs[2 * p + 1] >= v->terms_b) return 0;
    return 1;
}

int32_t tr_bilin_cross_eval(const tr_bilin_cross_view *v, tr_bilin_wide *terms, const tr_math_budget *budget,
                            tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!view_ok(v) || !terms) return TR_MATH_INVALID;
    tr_math_budget dflt;
    int32_t rc = tr_budget_check_one(budget_or(budget, &dflt), TR_BUDGET_ITERATIONS, tr_bilin_cross_items(v), refusal);
    if (rc) return rc;
    const uint64_t nv = v->ngroups * (uint64_t)v->values, items = tr_bilin_cross_items(v);
    memset(terms, 0, (size_t)(3 * nv) * sizeof(tr_bilin_wide));
    uint64_t t[3 * W];
    for (uint64_t i = 0; i < items; i++) {
        uint64_t idx = tr_bilin_cross_item(v, i, t);
        if (idx == TR_BILIN_NONE) continue;
        for (int k = 0; k < 3; k++) tr_bilin_w_add(terms[k * nv + idx].w, t + k * W);
    }
    return TR_MATH_OK;
}

static void join_release(tr_bilin_join *j) {
    if (!j) return;
    free(j->pairs); free(j->vals); free(j->terms);
    j->magic = 0;
    free(j);
}
void tr_bilin_join_free(tr_bilin_join *j) { if (join_valid(j)) join_release(j); }

tr_bilin_join *tr_bilin_join_finish(const tr_bilin_job *jb, const tr_bilin_wide *terms, const tr_math_budget *budget,
                                    tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    if (!job_valid(jb) || !terms) { set_error(error, TR_MATH_INVALID); return NULL; }
    const uint64_t nv = jb->ngroups * (uint64_t)jb->values;
    tr_math_need need;
    need_zero(&need);
    need.need[TR_BUDGET_OUTPUT] = jb->ngroups;
    need.need[TR_BUDGET_STATES] = nv;
    need.need[TR_BUDGET_BYTES] = join_bytes(jb->ngroups, jb->values, jb->spec.npairs);
    tr_math_budget dflt;
    int32_t rc = tr_budget_check(budget_or(budget, &dflt), &need, refusal);
    if (rc) { set_error(error, rc); return NULL; }
    tr_bilin_join *j = zalloc(sizeof *j);
    if (!j) { set_error(error, TR_MATH_MEMORY); return NULL; }
    j->pairs = zalloc((uint64_t)jb->spec.npairs * 8);
    j->vals = zalloc(nv * sizeof(tr_bilin_wide));
    j->terms = zalloc(3 * nv * sizeof(tr_bilin_wide));
    if (!j->pairs || !j->vals || !j->terms) { join_release(j); set_error(error, TR_MATH_MEMORY); return NULL; }
    j->spec = jb->spec; j->values = jb->values; j->terms_a = jb->terms_a; j->terms_b = jb->terms_b;
    j->ngroups = jb->ngroups; j->touched = jb->keys;
    memcpy(j->pairs, jb->pairs, (size_t)jb->spec.npairs * 8);
    memcpy(j->terms, terms, (size_t)(3 * nv) * sizeof(tr_bilin_wide));
    if (jb->old_vals) memcpy(j->vals, jb->old_vals, (size_t)nv * sizeof(tr_bilin_wide));
    for (uint64_t i = 0; i < nv; i++)                      /* new = old + T1 + T2 + T3 (exact: |value| < 2^318) */
        for (int k = 0; k < 3; k++) tr_bilin_w_add(j->vals[i].w, terms[k * nv + i].w);
    j->side_a[0] = jb->side_a[0]; j->side_a[1] = jb->side_a[1]; j->side_b[0] = jb->side_b[0]; j->side_b[1] = jb->side_b[1];
    hasher h;
    if (jb->kind == JOB_BUILD) {
        h_init(&h, UINT64_C(0x4e494f4a444c4942));
        h_word(&h, j->side_a[0]); h_word(&h, j->side_a[1]); h_word(&h, j->side_b[0]); h_word(&h, j->side_b[1]);
        h_bytes(&h, &j->spec, sizeof j->spec); h_bytes(&h, j->pairs, (uint64_t)j->spec.npairs * 8);
        h_out(&h, j->identity);
        j->base[0] = j->identity[0]; j->base[1] = j->identity[1]; j->step = 0;
    } else {
        j->step = jb->step + 1;
        h_init(&h, UINT64_C(0x4554414450444c42));
        h_word(&h, jb->old_identity[0]); h_word(&h, jb->old_identity[1]);
        h_word(&h, j->side_a[0]); h_word(&h, j->side_a[1]); h_word(&h, j->side_b[0]); h_word(&h, j->side_b[1]);
        h_word(&h, j->step);
        h_out(&h, j->identity);
        j->base[0] = jb->base[0]; j->base[1] = jb->base[1];
    }
    j->bytes = join_bytes(j->ngroups, j->values, j->spec.npairs);
    j->magic = JOIN_MAGIC;
    set_error(error, TR_MATH_OK);
    return j;
}

/* job + the CPU evaluator + finish */
static tr_bilin_join *run_cpu(tr_bilin_job *jb, const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error) {
    const uint64_t nv = jb->ngroups * (uint64_t)jb->values;
    tr_bilin_wide *terms = zalloc(tr_budget_mul(tr_budget_mul(nv, 3), sizeof(tr_bilin_wide)));
    if (!terms) { job_release(jb); set_error(error, TR_MATH_MEMORY); return NULL; }
    int32_t rc = tr_bilin_cross_eval(&jb->view, terms, budget, refusal);
    tr_bilin_join *out = NULL;
    if (rc) set_error(error, rc);
    else out = tr_bilin_join_finish(jb, terms, budget, refusal, error);
    free(terms);
    job_release(jb);
    return out;
}
tr_bilin_join *tr_bilin_join_build(const tr_delta_state *a, const tr_delta_state *b, const tr_bilin_spec *spec,
                                   const int32_t *pairs, const tr_math_budget *budget, tr_math_refusal *refusal,
                                   int32_t *error) {
    tr_bilin_job *jb = job_build(a, b, spec, pairs, 0, budget, refusal, error);
    return jb ? run_cpu(jb, budget, refusal, error) : NULL;
}
tr_bilin_join *tr_bilin_join_update(const tr_bilin_join *j, const tr_delta_state *a, const tr_delta *da,
                                    const tr_delta_state *a2, const tr_delta_state *b, const tr_delta *db,
                                    const tr_delta_state *b2, const tr_math_budget *budget, tr_math_refusal *refusal,
                                    int32_t *error) {
    tr_bilin_job *jb = tr_bilin_job_update(j, a, da, a2, b, db, b2, 0, budget, refusal, error);
    return jb ? run_cpu(jb, budget, refusal, error) : NULL;
}

int32_t tr_bilin_join_get_info(const tr_bilin_join *j, tr_bilin_info *out) {
    if (!join_valid(j) || !out) return TR_MATH_INVALID;
    memset(out, 0, sizeof *out);
    out->ngroups = j->ngroups; out->nkeys = j->spec.nkeys; out->groups_a = j->spec.groups_a;
    out->groups_b = j->spec.groups_b; out->null_key = j->spec.null_key; out->step = j->step; out->bytes = j->bytes;
    out->touched = j->touched;
    for (int i = 0; i < 2; i++) {
        out->identity[i] = j->identity[i]; out->base[i] = j->base[i]; out->side_a[i] = j->side_a[i]; out->side_b[i] = j->side_b[i];
    }
    out->npairs = j->spec.npairs; out->values = j->values; out->terms_a = j->terms_a; out->terms_b = j->terms_b;
    return TR_MATH_OK;
}
int32_t tr_bilin_join_get_values(const tr_bilin_join *j, tr_bilin_wide *out, uint64_t capacity) {
    if (!join_valid(j) || !out) return TR_MATH_INVALID;
    uint64_t nv = j->ngroups * (uint64_t)j->values;
    if (capacity < nv) return TR_MATH_INVALID;
    memcpy(out, j->vals, (size_t)nv * sizeof(tr_bilin_wide));
    return TR_MATH_OK;
}
int32_t tr_bilin_join_get_terms(const tr_bilin_join *j, tr_bilin_wide *out, uint64_t capacity) {
    if (!join_valid(j) || !out) return TR_MATH_INVALID;
    uint64_t nv = j->ngroups * (uint64_t)j->values;
    if (capacity < 3 * nv) return TR_MATH_INVALID;
    memcpy(out, j->terms, (size_t)(3 * nv) * sizeof(tr_bilin_wide));
    return TR_MATH_OK;
}
int32_t tr_bilin_join_get_pairs(const tr_bilin_join *j, int32_t *out, uint64_t capacity) {
    if (!join_valid(j) || !out || capacity < 2 * (uint64_t)j->spec.npairs) return TR_MATH_INVALID;
    memcpy(out, j->pairs, (size_t)j->spec.npairs * 8);
    return TR_MATH_OK;
}
int32_t tr_bilin_join_equal(const tr_bilin_join *a, const tr_bilin_join *b) {
    if (!join_valid(a) || !join_valid(b)) return TR_MATH_INVALID;
    if (memcmp(&a->spec, &b->spec, sizeof a->spec) || memcmp(a->pairs, b->pairs, (size_t)a->spec.npairs * 8)) return 0;
    return !memcmp(a->vals, b->vals, (size_t)(a->ngroups * (uint64_t)a->values) * sizeof(tr_bilin_wide));
}
int32_t tr_bilin_join_summarizes(const tr_bilin_join *j, const tr_delta_state *a, const tr_delta_state *b) {
    tr_delta_info ia, ib;
    if (!join_valid(j) || tr_delta_state_get_info(a, &ia) || tr_delta_state_get_info(b, &ib)) return TR_MATH_INVALID;
    return ia.identity[0] == j->side_a[0] && ia.identity[1] == j->side_a[1] && ib.identity[0] == j->side_b[0] &&
           ib.identity[1] == j->side_b[1];
}

/* ================================================================== exact finalizers (FND-009 big rationals) */
/* Big numbers made in sequence with a sticky error code: once *rc is set, nothing more is made (NULL) and the first
 * refusal (budget, memory) is the one reported. */
static tr_bigint *big_take(tr_bigint *x, int32_t e, int32_t *rc) {
    if (!x && !*rc) *rc = e ? e : TR_MATH_MEMORY;
    return x;
}
static tr_bigint *big_wide(const uint64_t *w, const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *rc) {
    int32_t e = TR_MATH_OK;
    return *rc ? NULL : big_take(tr_bigint_from_twos(w, W, budget, refusal, &e), e, rc);
}
static tr_bigint *big_i128(__int128 v, const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *rc) {
    int32_t e = TR_MATH_OK;
    return *rc ? NULL : big_take(tr_bigint_from_i128(tr_i128_set(v), budget, refusal, &e), e, rc);
}
static tr_bigint *big_op(int op, const tr_bigint *a, const tr_bigint *b, const tr_math_budget *budget,
                         tr_math_refusal *refusal, int32_t *rc) {
    int32_t e = TR_MATH_OK;
    if (*rc) return NULL;
    tr_bigint *r = op == 0 ? tr_bigint_mul(a, b, budget, refusal, &e) : tr_bigint_sub(a, b, budget, refusal, &e);
    return big_take(r, e, rc);
}
/* num / den as an owned rational (den == 0: the SQL NULL, never an error); takes num and den */
static int32_t make_ratio(tr_bigint *num, tr_bigint *den, int32_t rc, tr_bigq **out, int32_t *is_null,
                          const tr_math_budget *budget, tr_math_refusal *refusal) {
    if (!rc) {
        if (tr_bigint_sign(den) == 0) *is_null = 1;
        else {
            int32_t e = TR_MATH_OK;
            *out = tr_bigq_from_bigints(num, den, budget, refusal, &e);
            if (!*out) rc = e ? e : TR_MATH_MEMORY;
        }
    }
    tr_bigint_free(num); tr_bigint_free(den);
    return rc;
}
static int32_t make_int(tr_bigint *v, int32_t rc, tr_bigq **out, const tr_math_budget *budget, tr_math_refusal *refusal) {
    if (!rc) {
        int32_t e = TR_MATH_OK;
        *out = tr_bigq_from_bigint(v, budget, refusal, &e);
        if (!*out) rc = e ? e : TR_MATH_MEMORY;
    }
    tr_bigint_free(v);
    return rc;
}
/* (n S2 - S1^2) / n^2 or / (n (n - 1)); takes n, s1, s2 */
static int32_t make_variance(tr_bigint *n, tr_bigint *s1, tr_bigint *s2, int32_t rc, int64_t count, int sample,
                             tr_bigq **out, int32_t *is_null, const tr_math_budget *budget, tr_math_refusal *refusal) {
    tr_bigint *a = NULL, *b = NULL, *num = NULL, *one = NULL, *nm1 = NULL, *den = NULL;
    if (!rc && (count == 0 || (sample && count == 1))) *is_null = 1;
    else if (!rc) {
        a = big_op(0, n, s2, budget, refusal, &rc);
        b = big_op(0, s1, s1, budget, refusal, &rc);
        num = big_op(1, a, b, budget, refusal, &rc);
        if (sample) {
            int32_t e = TR_MATH_OK;
            if (!rc) one = big_take(tr_bigint_from_i64(1, budget, refusal, &e), e, &rc);
            nm1 = big_op(1, n, one, budget, refusal, &rc);
            den = big_op(0, n, nm1, budget, refusal, &rc);
        } else den = big_op(0, n, n, budget, refusal, &rc);
        rc = make_ratio(num, den, rc, out, is_null, budget, refusal);
        num = den = NULL;
    }
    tr_bigint_free(n); tr_bigint_free(s1); tr_bigint_free(s2); tr_bigint_free(a); tr_bigint_free(b);
    tr_bigint_free(num); tr_bigint_free(one); tr_bigint_free(nm1); tr_bigint_free(den);
    return rc;
}

static const tr_bilin_requirement REQUIREMENTS[TR_BILIN_FINALIZERS] = {
    {TR_BILIN_SUM, TR_BILIN_RETAIN_LINEAR, 1, TR_BILIN_DELETE_SUBTRACT, 1, 0, 0, 0, "sum"},
    {TR_BILIN_COUNT, TR_BILIN_RETAIN_LINEAR, 1, TR_BILIN_DELETE_SUBTRACT, 1, 0, 0, 0, "count"},
    {TR_BILIN_AVG, TR_BILIN_RETAIN_SUFFICIENT, 0, TR_BILIN_DELETE_SUFFICIENT, 1, 0, 0, 0, "avg"},
    {TR_BILIN_RATIO, TR_BILIN_RETAIN_SUFFICIENT, 0, TR_BILIN_DELETE_SUFFICIENT, 1, 0, 0, 0, "ratio"},
    {TR_BILIN_VAR_POP, TR_BILIN_RETAIN_SUFFICIENT, 0, TR_BILIN_DELETE_SUFFICIENT, 2, 0, 0, 0, "var_pop"},
    {TR_BILIN_VAR_SAMP, TR_BILIN_RETAIN_SUFFICIENT, 0, TR_BILIN_DELETE_SUFFICIENT, 2, 0, 0, 0, "var_samp"},
    {TR_BILIN_COUNT_DISTINCT, TR_BILIN_RETAIN_MULTIPLICITIES, 0, TR_BILIN_DELETE_MULTIPLICITY, 0, 0, 0, 0, "count_distinct"},
    {TR_BILIN_SUM_DISTINCT, TR_BILIN_RETAIN_MULTIPLICITIES, 0, TR_BILIN_DELETE_MULTIPLICITY, 0, 0, 0, 0, "sum_distinct"},
    {TR_BILIN_MIN, TR_BILIN_RETAIN_MULTIPLICITIES, 0, TR_BILIN_DELETE_MULTIPLICITY, 0, 0, TR_BILIN_RETAIN_EXTREMUM,
     TR_BILIN_DELETE_REBUILD, "min"},
    {TR_BILIN_MAX, TR_BILIN_RETAIN_MULTIPLICITIES, 0, TR_BILIN_DELETE_MULTIPLICITY, 0, 0, TR_BILIN_RETAIN_EXTREMUM,
     TR_BILIN_DELETE_REBUILD, "max"},
    {TR_BILIN_JOIN_SUM, TR_BILIN_RETAIN_PER_KEY, 0, TR_BILIN_DELETE_PER_KEY, 1, 1, 0, 0, "join_sum"},
};
int32_t tr_bilin_requirement_get(int32_t finalizer, tr_bilin_requirement *out) {
    if (!out || finalizer < 1 || finalizer > TR_BILIN_FINALIZERS) return TR_MATH_INVALID;
    *out = REQUIREMENTS[finalizer - 1];
    return TR_MATH_OK;
}

int32_t tr_bilin_finalize_state(const tr_delta_state *s, uint64_t group, int32_t fin, int32_t t0, int32_t t1,
                                tr_bigq **out, int32_t *is_null, const tr_math_budget *budget,
                                tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    tr_delta_info info;
    tr_bank_states st;
    if (!out || !is_null || tr_delta_state_get_info(s, &info) || tr_delta_state_get_states(s, &st)) return TR_MATH_INVALID;
    *out = NULL; *is_null = 0;
    if (fin < 1 || fin > TR_BILIN_FINALIZERS) return TR_MATH_INVALID;
    if (fin >= TR_BILIN_COUNT_DISTINCT) return TR_MATH_DOMAIN_ERROR;   /* not in linear cells: its own retained state */
    const int two = fin == TR_BILIN_RATIO || fin == TR_BILIN_VAR_POP || fin == TR_BILIN_VAR_SAMP;
    if (group >= info.ngroups || t0 < 0 || t0 >= info.nterms || (two && (t1 < 0 || t1 >= info.nterms))) return TR_MATH_INVALID;
    tr_math_budget dflt;
    budget = budget_or(budget, &dflt);
    const uint64_t m = (uint64_t)info.nterms;
    if (fin == TR_BILIN_VAR_POP || fin == TR_BILIN_VAR_SAMP) {         /* t1 must be t0 squared, t0 not constant */
        uint64_t ne = m * (uint64_t)info.ncols;
        uint8_t *e = zalloc(ne);
        if (!e) return TR_MATH_MEMORY;
        tr_delta_state_get_exponents(s, e, ne);
        int ok = 0, same = 1;
        for (int32_t c = 0; c < info.ncols; c++) {
            uint32_t x = e[(uint64_t)t0 * info.ncols + c], y = e[(uint64_t)t1 * info.ncols + c];
            ok |= x != 0; same &= y == 2 * x;
        }
        free(e);
        if (!ok || !same) return TR_MATH_DOMAIN_ERROR;                  /* not the sufficient sums n, S1, S2 */
    }
    const uint64_t k0 = group * m + (uint64_t)t0, k1 = group * m + (uint64_t)(two ? t1 : t0);
    int64_t c0 = (int64_t)st.count[k0], c1 = (int64_t)st.count[k1];
    __int128 s0 = tr_i128_get(st.sum[k0]), s1 = tr_i128_get(st.sum[k1]);
    int32_t rc = TR_MATH_OK;
    switch (fin) {
    case TR_BILIN_SUM:
        if (!c0) { *is_null = 1; return TR_MATH_OK; }
        { tr_bigint *v = big_i128(s0, budget, refusal, &rc); return make_int(v, rc, out, budget, refusal); }
    case TR_BILIN_COUNT:
        { tr_bigint *v = big_i128(c0, budget, refusal, &rc); return make_int(v, rc, out, budget, refusal); }
    case TR_BILIN_AVG:
        if (!c0) { *is_null = 1; return TR_MATH_OK; }
        {
            tr_bigint *n = big_i128(s0, budget, refusal, &rc), *d = big_i128(c0, budget, refusal, &rc);
            return make_ratio(n, d, rc, out, is_null, budget, refusal);
        }
    case TR_BILIN_RATIO:
        if (!c0 || !c1) { *is_null = 1; return TR_MATH_OK; }
        {
            tr_bigint *n = big_i128(s0, budget, refusal, &rc), *d = big_i128(s1, budget, refusal, &rc);
            return make_ratio(n, d, rc, out, is_null, budget, refusal);
        }
    default: {                                                         /* VAR_POP, VAR_SAMP: n = count of t */
        tr_bigint *n = big_i128(c0, budget, refusal, &rc), *a = big_i128(s0, budget, refusal, &rc);
        tr_bigint *b = big_i128(s1, budget, refusal, &rc);
        return make_variance(n, a, b, rc, c0, fin == TR_BILIN_VAR_SAMP, out, is_null, budget, refusal);
    }
    }
}

int32_t tr_bilin_join_finalize(const tr_bilin_join *j, uint64_t cell, int32_t fin, int32_t p0, int32_t p1,
                               tr_bigq **out, int32_t *is_null, const tr_math_budget *budget, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!join_valid(j) || !out || !is_null || cell >= j->ngroups || fin < 1 || fin > TR_BILIN_FINALIZERS)
        return TR_MATH_INVALID;
    *out = NULL; *is_null = 0;
    if (fin != TR_BILIN_SUM && fin != TR_BILIN_COUNT && fin != TR_BILIN_AVG && fin != TR_BILIN_RATIO &&
        fin != TR_BILIN_JOIN_SUM) return TR_MATH_DOMAIN_ERROR;           /* a join state holds per-pair sums only */
    if (p0 < 0 || p0 >= j->spec.npairs || (fin == TR_BILIN_RATIO && (p1 < 0 || p1 >= j->spec.npairs))) return TR_MATH_INVALID;
    tr_math_budget dflt;
    budget = budget_or(budget, &dflt);
    const tr_bilin_wide *row = j->vals + cell * (uint64_t)j->values;
    const uint64_t *cnt0 = row[1 + 2 * p0].w, *sum0 = row[2 + 2 * p0].w;
    int32_t rc = TR_MATH_OK;
    if (fin == TR_BILIN_COUNT) { tr_bigint *v = big_wide(cnt0, budget, refusal, &rc); return make_int(v, rc, out, budget, refusal); }
    if (tr_bilin_w_is_zero(cnt0)) { *is_null = 1; return TR_MATH_OK; }
    if (fin == TR_BILIN_SUM || fin == TR_BILIN_JOIN_SUM) {
        tr_bigint *v = big_wide(sum0, budget, refusal, &rc);
        return make_int(v, rc, out, budget, refusal);
    }
    if (fin == TR_BILIN_AVG) {
        tr_bigint *n = big_wide(sum0, budget, refusal, &rc), *d = big_wide(cnt0, budget, refusal, &rc);
        return make_ratio(n, d, rc, out, is_null, budget, refusal);
    }
    const uint64_t *cnt1 = row[1 + 2 * p1].w, *sum1 = row[2 + 2 * p1].w;
    if (tr_bilin_w_is_zero(cnt1)) { *is_null = 1; return TR_MATH_OK; }
    tr_bigint *n = big_wide(sum0, budget, refusal, &rc), *d = big_wide(sum1, budget, refusal, &rc);
    return make_ratio(n, d, rc, out, is_null, budget, refusal);
}

/* ================================================================== dense matrices */
int32_t tr_bilin_matrix_need(uint64_t G, uint64_t K, uint64_t H, int32_t with_da, int32_t with_db, int32_t target,
                             tr_math_need *need) {
    if (!need || (target != 0 && target != 1)) return TR_MATH_INVALID;
    tr_bilin_matrix_need_inline(G, K, H, with_da != 0, with_db != 0, target, need);
    return TR_MATH_OK;
}
int32_t tr_bilin_matrix_delta(uint64_t G, uint64_t K, uint64_t H, const int64_t *A, const int64_t *B,
                              const int64_t *dA, const int64_t *dB, tr_bilin_wide *out, const tr_math_budget *budget,
                              tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!out || (G && K && !A) || (K && H && !B) || G < 1 || H < 1) return TR_MATH_INVALID;
    if (K >= (UINT64_C(1) << 32)) return TR_MATH_UNSUPPORTED;            /* keeps every sum below 2^160 */
    tr_math_need need;
    tr_bilin_matrix_need_inline(G, K, H, dA != NULL, dB != NULL, 0, &need);
    tr_math_budget dflt;
    int32_t rc = tr_budget_check(budget_or(budget, &dflt), &need, refusal);
    if (rc) return rc;
    for (uint64_t g = 0; g < G; g++)
        for (uint64_t h = 0; h < H; h++) tr_bilin_matrix_entry(G, K, H, A, B, dA, dB, g, h, out);
    return TR_MATH_OK;
}

/* ================================================================== value states (MULTISET / EXTREMUM) */
typedef struct { uint64_t g; int64_t v, w; } entry;     /* a (group, value, weight) triple */
static int cmp_entry(const void *x, const void *y) {
    const entry *a = (const entry *)x, *b = (const entry *)y;
    if (a->g != b->g) return a->g < b->g ? -1 : 1;
    return a->v < b->v ? -1 : a->v > b->v;
}
static void values_release(tr_bilin_values *s) {
    if (!s) return;
    free(s->rows); free(s->nulls); free(s->present); free(s->offset); free(s->vals); free(s->mult);
    free(s->mn); free(s->mnc); free(s->mx); free(s->mxc);
    s->magic = 0;
    free(s);
}
void tr_bilin_values_free(tr_bilin_values *s) { if (values_valid(s)) values_release(s); }

static uint64_t values_bytes(int32_t kind, uint64_t ngroups, uint64_t entries) {
    uint64_t b = tr_budget_add(sizeof(tr_bilin_values), tr_budget_mul(ngroups, 24));
    if (kind == TR_BILIN_MULTISET) b = tr_budget_add(b, tr_budget_add(tr_budget_mul(ngroups + 1, 8), tr_budget_mul(entries, 16)));
    else b = tr_budget_add(b, tr_budget_mul(ngroups, 32));
    return b;
}
static tr_bilin_values *values_alloc(int32_t kind, uint64_t ngroups, uint64_t entries) {
    tr_bilin_values *s = zalloc(sizeof *s);
    if (!s) return NULL;
    s->kind = kind; s->ngroups = ngroups; s->entries = entries;
    s->rows = zalloc(ngroups * 8); s->nulls = zalloc(ngroups * 8); s->present = zalloc(ngroups * 8);
    int ok = s->rows && s->nulls && s->present;
    if (kind == TR_BILIN_MULTISET) {
        s->offset = zalloc((ngroups + 1) * 8); s->vals = zalloc(entries * 8); s->mult = zalloc(entries * 8);
        ok = ok && s->offset && s->vals && s->mult;
    } else {
        s->mn = zalloc(ngroups * 8); s->mnc = zalloc(ngroups * 8); s->mx = zalloc(ngroups * 8); s->mxc = zalloc(ngroups * 8);
        ok = ok && s->mn && s->mnc && s->mx && s->mxc;
    }
    if (!ok) { values_release(s); return NULL; }
    s->bytes = values_bytes(kind, ngroups, entries);
    return s;
}

int32_t tr_bilin_values_need_scope(const tr_scope *scope, int32_t column, int32_t kind, tr_math_need *need) {
    tr_scope_view v;
    if (!need || !scope || tr_scope_get_view(scope, &v) || column < 0 || column >= v.ncols ||
        (kind != TR_BILIN_MULTISET && kind != TR_BILIN_EXTREMUM)) return TR_MATH_INVALID;
    need_zero(need);
    uint64_t present = v.col[column].nvalid;
    need->need[TR_BUDGET_SUPPORT] = kind == TR_BILIN_MULTISET ? present : 2 * v.ngroups;
    need->need[TR_BUDGET_STATES] = v.ngroups;
    uint64_t bytes = values_bytes(kind, v.ngroups, kind == TR_BILIN_MULTISET ? present : 0);
    if (kind == TR_BILIN_MULTISET) bytes = tr_budget_add(bytes, tr_budget_mul(present, sizeof(entry)));   /* sort scratch */
    need->need[TR_BUDGET_BYTES] = bytes;
    return TR_MATH_OK;
}

static void values_identity(tr_bilin_values *s, uint64_t tag, const uint64_t *w, int nw) {
    hasher h;
    h_init(&h, tag);
    for (int i = 0; i < nw; i++) h_word(&h, w[i]);
    h_word(&h, (uint64_t)s->kind); h_word(&h, (uint64_t)s->column); h_word(&h, s->ngroups);
    h_bytes(&h, s->rows, s->ngroups * 8); h_bytes(&h, s->nulls, s->ngroups * 8);
    if (s->kind == TR_BILIN_MULTISET) { h_bytes(&h, s->vals, s->entries * 8); h_bytes(&h, s->mult, s->entries * 8); }
    else {
        h_bytes(&h, s->mn, s->ngroups * 8); h_bytes(&h, s->mnc, s->ngroups * 8);
        h_bytes(&h, s->mx, s->ngroups * 8); h_bytes(&h, s->mxc, s->ngroups * 8);
    }
    h_out(&h, s->identity);
}

tr_bilin_values *tr_bilin_values_from_scope(const tr_scope *scope, int32_t column, int32_t kind,
                                            const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    tr_math_need need;
    int32_t rc = tr_bilin_values_need_scope(scope, column, kind, &need);
    tr_math_budget dflt;
    if (!rc) rc = tr_budget_check(budget_or(budget, &dflt), &need, refusal);
    if (rc) { set_error(error, rc); return NULL; }
    tr_scope_view v;
    tr_scope_info si;
    tr_scope_get_view(scope, &v);
    tr_scope_get_info(scope, &si);
    const uint64_t present = v.col[column].nvalid;
    entry *e = NULL;
    if (kind == TR_BILIN_MULTISET && !(e = zalloc(tr_budget_mul(present, sizeof(entry))))) { set_error(error, TR_MATH_MEMORY); return NULL; }
    int64_t *rows = zalloc(v.ngroups * 8), *nulls = zalloc(v.ngroups * 8);
    if (!rows || !nulls) { free(e); free(rows); free(nulls); set_error(error, TR_MATH_MEMORY); return NULL; }
    uint64_t n = 0;
    tr_bilin_values *s = NULL;
    if (kind == TR_BILIN_EXTREMUM) s = values_alloc(kind, v.ngroups, 0);
    for (uint64_t w = 0; w < (v.rows + 63) / 64; w++) {
        uint64_t bits = v.selection[w];
        while (bits) {
            uint64_t r = (w << 6) | (uint64_t)__builtin_ctzll(bits);
            bits &= bits - 1;
            uint64_t g = tr_scope_group(&v, r);
            rows[g]++;
            if (!tr_scope_present(&v, column, r)) { nulls[g]++; continue; }
            int64_t x = tr_scope_value(&v, column, r);
            if (kind == TR_BILIN_MULTISET) { e[n].g = g; e[n].v = x; e[n].w = 1; n++; }
            else if (s) {
                if (!s->present[g] || x < s->mn[g]) { s->mn[g] = x; s->mnc[g] = 0; }
                if (!s->present[g] || x > s->mx[g]) { s->mx[g] = x; s->mxc[g] = 0; }
                if (x == s->mn[g]) s->mnc[g]++;
                if (x == s->mx[g]) s->mxc[g]++;
                s->present[g]++;
            }
        }
    }
    if (kind == TR_BILIN_MULTISET) {
        if (n > 1) qsort(e, (size_t)n, sizeof(entry), cmp_entry);
        uint64_t distinct = 0;
        for (uint64_t i = 0; i < n; i++) if (!i || e[i].g != e[i - 1].g || e[i].v != e[i - 1].v) distinct++;
        s = values_alloc(kind, v.ngroups, distinct);
        if (s) {
            uint64_t q = 0;
            for (uint64_t i = 0; i < n; i++) {
                if (!i || e[i].g != e[i - 1].g || e[i].v != e[i - 1].v) { s->vals[q] = e[i].v; s->mult[q] = 0; q++; s->offset[e[i].g + 1]++; }
                s->mult[q - 1]++;
                s->present[e[i].g]++;
            }
            for (uint64_t g = 0; g < v.ngroups; g++) s->offset[g + 1] += s->offset[g];
        }
    }
    free(e);
    if (!s) { free(rows); free(nulls); set_error(error, TR_MATH_MEMORY); return NULL; }
    memcpy(s->rows, rows, (size_t)v.ngroups * 8);
    memcpy(s->nulls, nulls, (size_t)v.ngroups * 8);
    free(rows); free(nulls);
    s->column = column; s->ncols = v.ncols; s->scale = v.col[column].scale;
    for (int32_t c = 0; c < v.ncols; c++) s->scales[c] = v.col[c].scale;
    values_identity(s, UINT64_C(0x5345554c41564c42), si.digest, 2);
    s->base[0] = s->identity[0]; s->base[1] = s->identity[1];
    s->magic = VALUES_MAGIC;
    set_error(error, TR_MATH_OK);
    return s;
}

/* the delta's (group, value, net weight) of present values sorted, its per-group rows / nulls changes; the delta must
 * have the state's columns, the column's scale and the groups */
static int32_t delta_entries(const tr_bilin_values *s, const tr_delta *d, entry **out, uint64_t *count) {
    tr_delta_batch_info bi;
    tr_scope_view dv;
    tr_delta_rows dr;
    if (tr_delta_get_info(d, &bi) || tr_delta_get_view(d, &dv) || tr_delta_get_rows(d, &dr)) return TR_MATH_INVALID;
    if (bi.ncols != s->ncols || bi.ngroups != s->ngroups) return TR_MATH_DOMAIN_ERROR;
    for (int32_t c = 0; c < s->ncols; c++) if (dv.col[c].scale != s->scales[c]) return TR_MATH_DOMAIN_ERROR;
    entry *e = zalloc(tr_budget_mul(dr.rows, sizeof(entry)));
    if (!e) return TR_MATH_MEMORY;
    uint64_t n = 0;
    for (uint64_t r = 0; r < dr.rows; r++) {
        if (!dr.weights[r]) continue;
        e[n].g = dr.groups[dr.local[r]];
        e[n].w = dr.weights[r];
        /* NULL values sort after every value of their group with a marker: v = INT64_MAX and w kept, flagged below */
        if (tr_scope_present(&dv, s->column, r)) e[n].v = tr_scope_value(&dv, s->column, r);
        else { e[n].v = INT64_MAX; e[n].g |= UINT64_C(1) << 63; }   /* NULLs: their own key (high bit) */
        n++;
    }
    if (n > 1) qsort(e, (size_t)n, sizeof(entry), cmp_entry);
    *out = e; *count = n;
    return TR_MATH_OK;
}

int32_t tr_bilin_values_need_apply(const tr_bilin_values *s, const tr_delta *d, tr_math_need *need) {
    tr_delta_batch_info bi;
    if (!need || !values_valid(s) || tr_delta_get_info(d, &bi)) return TR_MATH_INVALID;
    need_zero(need);
    uint64_t entries = s->kind == TR_BILIN_MULTISET ? tr_budget_add(s->entries, bi.rows) : 0;   /* at most */
    need->need[TR_BUDGET_SUPPORT] = s->kind == TR_BILIN_MULTISET ? entries : 2 * s->ngroups;
    need->need[TR_BUDGET_STATES] = s->ngroups;
    need->need[TR_BUDGET_BYTES] = tr_budget_add(values_bytes(s->kind, s->ngroups, entries), tr_budget_mul(bi.rows, sizeof(entry)));
    return TR_MATH_OK;
}

static void violation_none(tr_bilin_violation *v) {
    if (!v) return;
    memset(v, 0, sizeof *v);
    v->group = TR_BILIN_NONE;
}
static void violation_set(tr_bilin_violation *v, uint64_t g, int32_t reason, int has_value, int64_t value, int64_t have,
                          int64_t change) {
    if (!v || v->group != TR_BILIN_NONE) return;          /* the first violation only */
    v->group = g; v->reason = reason; v->has_value = has_value; v->value = value; v->have = have; v->change = change;
}

/* EXTREMUM: the new (min, count, max, count, present) of group g from its old values and the delta's present-value
 * entries e[0..n) (sorted by value, net weights) */
static int32_t extremum_group(const tr_bilin_values *s, uint64_t g, const entry *e, uint64_t n, tr_bilin_values *o,
                              tr_bilin_violation *viol) {
    int64_t total = 0;
    for (uint64_t i = 0; i < n; i++) if (__builtin_add_overflow(total, e[i].w, &total)) return TR_MATH_OVERFLOW;
    int64_t present = s->present[g], np;
    if (__builtin_add_overflow(present, total, &np)) return TR_MATH_OVERFLOW;
    if (np < 0) { violation_set(viol, g, TR_BILIN_NEGATIVE_VALUE, 0, 0, present, total); return TR_MATH_DOMAIN_ERROR; }
    if (present == 0) {                                   /* nothing retained: every net weight must be an insert */
        for (uint64_t i = 0; i < n; i++)
            if (e[i].w < 0) { violation_set(viol, g, TR_BILIN_NEGATIVE_VALUE, 1, e[i].v, 0, e[i].w); return TR_MATH_DOMAIN_ERROR; }
        uint64_t f = 0, l = n;
        while (f < n && e[f].w == 0) f++;
        while (l > f && e[l - 1].w == 0) l--;
        o->present[g] = np;
        if (np) { o->mn[g] = e[f].v; o->mnc[g] = e[f].w; o->mx[g] = e[l - 1].v; o->mxc[g] = e[l - 1].w; }
        else { o->mn[g] = o->mnc[g] = o->mx[g] = o->mxc[g] = 0; }
        return TR_MATH_OK;
    }
    /* the old multiset: mc copies of m, Mc copies of M (m < M) and `inner` copies of unknown values strictly between.
     * Net weights: below m or above M only inserts (nothing is retained there); at m and M exact; strictly between,
     * a net delete removes old inner copies (R = the old inner copies left), a net insert adds known copies. */
    const int64_t m = s->mn[g], M = s->mx[g];
    const int single = m == M;
    int64_t mc = s->mnc[g], Mc = s->mxc[g], inner = present - mc - (single ? 0 : Mc);
    int64_t wm = 0, wM = 0, delin = 0, insin = 0;
    uint64_t lo_n = 0, hi_f = n, fi = n, li = n;          /* below m: [0, lo_n); above M: [hi_f, n); inner inserts */
    for (uint64_t i = 0; i < n; i++) {
        const int64_t v = e[i].v, w = e[i].w;
        if (!w) continue;
        int bad = 0;
        if (v < m || v > M) {
            if (w < 0) { violation_set(viol, g, TR_BILIN_NEGATIVE_VALUE, 1, v, 0, w); return TR_MATH_DOMAIN_ERROR; }
            if (v < m) lo_n = i + 1; else if (hi_f == n) hi_f = i;
        } else if (v == m) bad = __builtin_add_overflow(wm, w, &wm);
        else if (v == M) bad = __builtin_add_overflow(wM, w, &wM);
        else if (w < 0) bad = __builtin_add_overflow(delin, w, &delin);
        else { bad = __builtin_add_overflow(insin, w, &insin); if (fi == n) fi = i; li = i; }
        if (bad) return TR_MATH_OVERFLOW;
    }
    int64_t mc2, Mc2, R;
    if (__builtin_add_overflow(mc, wm, &mc2) || __builtin_add_overflow(Mc, single ? wm : wM, &Mc2) ||
        __builtin_add_overflow(inner, delin, &R)) return TR_MATH_OVERFLOW;
    if (mc2 < 0) { violation_set(viol, g, TR_BILIN_NEGATIVE_VALUE, 1, m, mc, wm); return TR_MATH_DOMAIN_ERROR; }
    if (!single && Mc2 < 0) { violation_set(viol, g, TR_BILIN_NEGATIVE_VALUE, 1, M, Mc, wM); return TR_MATH_DOMAIN_ERROR; }
    if (R < 0) { violation_set(viol, g, TR_BILIN_NEGATIVE_VALUE, 0, 0, inner, delin); return TR_MATH_DOMAIN_ERROR; }
    o->present[g] = np;
    if (!np) { o->mn[g] = o->mnc[g] = o->mx[g] = o->mxc[g] = 0; return TR_MATH_OK; }
    /* the new minimum: an insert below m; else m while a copy survives; else unknown while old inner copies remain
     * (the next value is not retained: rebuild); else the smallest inner insert, M, or the smallest insert above M */
    uint64_t f = 0;
    while (f < lo_n && e[f].w == 0) f++;
    if (f < lo_n) { o->mn[g] = e[f].v; o->mnc[g] = e[f].w; }
    else if (mc2 > 0) { o->mn[g] = m; o->mnc[g] = mc2; }
    else if (R > 0) { violation_set(viol, g, TR_BILIN_EXTREME_DELETED, 1, m, mc, wm); return TR_MATH_DOMAIN_ERROR; }
    else if (insin > 0) { o->mn[g] = e[fi].v; o->mnc[g] = e[fi].w; }
    else if (!single && Mc2 > 0) { o->mn[g] = M; o->mnc[g] = Mc2; }
    else {
        uint64_t q = hi_f;
        while (q < n && e[q].w == 0) q++;
        o->mn[g] = e[q].v; o->mnc[g] = e[q].w;           /* np > 0: an insert above M remains */
    }
    uint64_t l = n;
    while (l > hi_f && e[l - 1].w == 0) l--;
    if (l > hi_f) { o->mx[g] = e[l - 1].v; o->mxc[g] = e[l - 1].w; }
    else if (Mc2 > 0) { o->mx[g] = M; o->mxc[g] = Mc2; }
    else if (R > 0) { violation_set(viol, g, TR_BILIN_EXTREME_DELETED, 1, M, single ? mc : Mc, single ? wm : wM); return TR_MATH_DOMAIN_ERROR; }
    else if (insin > 0) { o->mx[g] = e[li].v; o->mxc[g] = e[li].w; }
    else if (!single && mc2 > 0) { o->mx[g] = m; o->mxc[g] = mc2; }
    else {
        uint64_t q = lo_n;
        while (q > 0 && e[q - 1].w == 0) q--;
        o->mx[g] = e[q - 1].v; o->mxc[g] = e[q - 1].w;   /* np > 0: an insert below m remains */
    }
    if (o->mn[g] == o->mx[g]) o->mnc[g] = o->mxc[g] = np;  /* one distinct value: every present copy */
    return TR_MATH_OK;
}

tr_bilin_values *tr_bilin_values_apply(const tr_bilin_values *s, const tr_delta *d, const tr_math_budget *budget,
                                       tr_math_refusal *refusal, tr_bilin_violation *viol, int32_t *error) {
    tr_budget_none(refusal);
    violation_none(viol);
    tr_math_need need;
    int32_t rc = tr_bilin_values_need_apply(s, d, &need);
    tr_math_budget dflt;
    if (!rc) rc = tr_budget_check(budget_or(budget, &dflt), &need, refusal);
    if (rc) { set_error(error, rc); return NULL; }
    entry *e = NULL;
    uint64_t n = 0;
    rc = delta_entries(s, d, &e, &n);
    if (rc) { set_error(error, rc); return NULL; }
    /* combine equal (group, value) entries; NULL entries (high bit) combine per group */
    uint64_t q = 0;
    for (uint64_t i = 0; i < n; i++) {
        if (q && e[q - 1].g == e[i].g && e[q - 1].v == e[i].v) {
            if (__builtin_add_overflow(e[q - 1].w, e[i].w, &e[q - 1].w)) { free(e); set_error(error, TR_MATH_OVERFLOW); return NULL; }
        } else e[q++] = e[i];
    }
    n = q;
    const uint64_t G = s->ngroups, high = UINT64_C(1) << 63;
    /* per group: rows and nulls changes; the present entries of group g are contiguous (sorted by group first) */
    uint64_t cap = s->kind == TR_BILIN_MULTISET ? s->entries + n : 0;
    tr_bilin_values *o = values_alloc(s->kind, G, cap);
    if (!o) { free(e); set_error(error, TR_MATH_MEMORY); return NULL; }
    memcpy(o->rows, s->rows, (size_t)G * 8); memcpy(o->nulls, s->nulls, (size_t)G * 8); memcpy(o->present, s->present, (size_t)G * 8);
    if (s->kind == TR_BILIN_EXTREMUM) {
        memcpy(o->mn, s->mn, (size_t)G * 8); memcpy(o->mnc, s->mnc, (size_t)G * 8);
        memcpy(o->mx, s->mx, (size_t)G * 8); memcpy(o->mxc, s->mxc, (size_t)G * 8);
    }
    /* walk the groups that the delta touches in increasing order: present entries [p0, p1) and NULL entries */
    uint64_t pi = 0, ni = 0, w = 0, gprev = 0;
    while (pi < n && !(e[pi].g & high)) pi++;              /* e[0 .. pi) present entries, e[pi .. n) NULL entries */
    const uint64_t npres = pi;
    pi = 0; ni = npres;
    rc = TR_MATH_OK;
    for (;;) {
        uint64_t gp = pi < npres ? e[pi].g : UINT64_MAX, gn = ni < n ? (e[ni].g & ~high) : UINT64_MAX;
        uint64_t g = gp < gn ? gp : gn;
        if (g == UINT64_MAX) break;
        uint64_t p0 = pi;
        while (pi < npres && e[pi].g == g) pi++;
        int64_t dn = 0, dp = 0;
        if (ni < n && (e[ni].g & ~high) == g) dn = e[ni++].w;
        for (uint64_t i = p0; i < pi; i++) if (__builtin_add_overflow(dp, e[i].w, &dp)) { rc = TR_MATH_OVERFLOW; break; }
        int64_t r2, n2;
        if (rc || __builtin_add_overflow(s->rows[g], dp, &r2) || __builtin_add_overflow(r2, dn, &r2) ||
            __builtin_add_overflow(s->nulls[g], dn, &n2)) { rc = rc ? rc : TR_MATH_OVERFLOW; break; }
        if (r2 < 0) { violation_set(viol, g, TR_BILIN_NEGATIVE_ROWS, 0, 0, s->rows[g], dp + dn); rc = TR_MATH_DOMAIN_ERROR; break; }
        if (n2 < 0) { violation_set(viol, g, TR_BILIN_NEGATIVE_NULLS, 0, 0, s->nulls[g], dn); rc = TR_MATH_DOMAIN_ERROR; break; }
        o->rows[g] = r2; o->nulls[g] = n2;
        if (s->kind == TR_BILIN_EXTREMUM) {
            rc = extremum_group(s, g, e + p0, pi - p0, o, viol);
            if (rc) break;
            continue;
        }
        /* MULTISET: copy the untouched groups before g, then merge g's entries */
        for (uint64_t h = gprev; h < g; h++) {
            uint64_t a = s->offset[h], b = s->offset[h + 1];
            memcpy(o->vals + w, s->vals + a, (size_t)(b - a) * 8); memcpy(o->mult + w, s->mult + a, (size_t)(b - a) * 8);
            o->offset[h] = w; w += b - a;
        }
        o->offset[g] = w;
        uint64_t a = s->offset[g], b = s->offset[g + 1], i = p0;
        int64_t pres = 0;
        while (a < b || i < pi) {
            int64_t v, m;
            if (i >= pi || (a < b && s->vals[a] < e[i].v)) { v = s->vals[a]; m = s->mult[a]; a++; }
            else if (a >= b || e[i].v < s->vals[a]) {
                v = e[i].v; m = e[i].w; i++;
                if (m < 0) { violation_set(viol, g, TR_BILIN_NEGATIVE_VALUE, 1, v, 0, m); rc = TR_MATH_DOMAIN_ERROR; break; }
            } else {
                v = s->vals[a];
                if (__builtin_add_overflow(s->mult[a], e[i].w, &m)) { rc = TR_MATH_OVERFLOW; break; }
                if (m < 0) { violation_set(viol, g, TR_BILIN_NEGATIVE_VALUE, 1, v, s->mult[a], e[i].w); rc = TR_MATH_DOMAIN_ERROR; break; }
                a++; i++;
            }
            if (m) { o->vals[w] = v; o->mult[w] = m; w++; pres += m; }
        }
        if (rc) break;
        o->present[g] = pres;
        gprev = g + 1;
    }
    free(e);
    if (rc) { values_release(o); set_error(error, rc); return NULL; }
    if (s->kind == TR_BILIN_MULTISET) {
        for (uint64_t h = gprev; h < G; h++) {
            uint64_t a = s->offset[h], b = s->offset[h + 1];
            memcpy(o->vals + w, s->vals + a, (size_t)(b - a) * 8); memcpy(o->mult + w, s->mult + a, (size_t)(b - a) * 8);
            o->offset[h] = w; w += b - a;
        }
        o->offset[G] = w;
        o->entries = w;
        o->bytes = values_bytes(o->kind, G, w);
    }
    o->column = s->column; o->ncols = s->ncols; o->scale = s->scale;
    memcpy(o->scales, s->scales, sizeof o->scales);
    o->step = s->step + 1; o->base[0] = s->base[0]; o->base[1] = s->base[1];
    tr_delta_batch_info bi;
    tr_delta_get_info(d, &bi);
    uint64_t lineage[5] = {s->identity[0], s->identity[1], bi.identity[0], bi.identity[1], o->step};
    values_identity(o, UINT64_C(0x5050414c41564c42), lineage, 5);
    o->magic = VALUES_MAGIC;
    set_error(error, TR_MATH_OK);
    return o;
}

int32_t tr_bilin_values_get_info(const tr_bilin_values *s, tr_bilin_values_info *out) {
    if (!values_valid(s) || !out) return TR_MATH_INVALID;
    memset(out, 0, sizeof *out);
    out->ngroups = s->ngroups; out->entries = s->entries; out->step = s->step; out->bytes = s->bytes;
    for (int i = 0; i < 2; i++) { out->identity[i] = s->identity[i]; out->base[i] = s->base[i]; }
    out->kind = s->kind; out->column = s->column; out->ncols = s->ncols; out->scale = s->scale;
    return TR_MATH_OK;
}
int32_t tr_bilin_values_group(const tr_bilin_values *s, uint64_t g, tr_bilin_group_values *out) {
    if (!values_valid(s) || !out || g >= s->ngroups) return TR_MATH_INVALID;
    memset(out, 0, sizeof *out);
    out->rows = s->rows[g]; out->nulls = s->nulls[g]; out->present = s->present[g];
    if (s->kind == TR_BILIN_EXTREMUM) {
        if (s->present[g]) { out->min = s->mn[g]; out->min_count = s->mnc[g]; out->max = s->mx[g]; out->max_count = s->mxc[g]; }
        return TR_MATH_OK;
    }
    uint64_t a = s->offset[g], b = s->offset[g + 1];
    out->distinct = b - a;
    __int128 sum = 0;
    for (uint64_t i = a; i < b; i++) sum += s->vals[i];   /* fewer than 2^63 values of |v| < 2^63: fits */
    out->sum_distinct = tr_i128_set(sum);
    if (b > a) { out->min = s->vals[a]; out->min_count = s->mult[a]; out->max = s->vals[b - 1]; out->max_count = s->mult[b - 1]; }
    return TR_MATH_OK;
}
int32_t tr_bilin_values_entries(const tr_bilin_values *s, uint64_t g, int64_t *values, int64_t *mult, uint64_t capacity,
                                uint64_t *count) {
    if (!values_valid(s) || !count || g >= s->ngroups) return TR_MATH_INVALID;
    if (s->kind != TR_BILIN_MULTISET) return TR_MATH_DOMAIN_ERROR;
    uint64_t a = s->offset[g], b = s->offset[g + 1];
    *count = b - a;
    if (!capacity) return TR_MATH_OK;
    if (capacity < b - a || !values || !mult) return TR_MATH_INVALID;
    memcpy(values, s->vals + a, (size_t)(b - a) * 8);
    memcpy(mult, s->mult + a, (size_t)(b - a) * 8);
    return TR_MATH_OK;
}
int32_t tr_bilin_values_finalize(const tr_bilin_values *s, uint64_t g, int32_t fin, tr_bigq **out, int32_t *is_null,
                                 const tr_math_budget *budget, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    tr_bilin_group_values gv;
    if (!out || !is_null || tr_bilin_values_group(s, g, &gv)) return TR_MATH_INVALID;
    *out = NULL; *is_null = 0;
    if (fin < 1 || fin > TR_BILIN_FINALIZERS) return TR_MATH_INVALID;
    if (fin != TR_BILIN_COUNT && fin != TR_BILIN_COUNT_DISTINCT && fin != TR_BILIN_SUM_DISTINCT && fin != TR_BILIN_MIN &&
        fin != TR_BILIN_MAX) return TR_MATH_DOMAIN_ERROR;   /* a value state is not the sufficient sums of SUM / AVG */
    if (s->kind == TR_BILIN_EXTREMUM && (fin == TR_BILIN_COUNT_DISTINCT || fin == TR_BILIN_SUM_DISTINCT))
        return TR_MATH_DOMAIN_ERROR;                        /* needs per-value multiplicities */
    tr_math_budget dflt;
    budget = budget_or(budget, &dflt);
    int32_t rc = TR_MATH_OK;
    __int128 v = 0;
    switch (fin) {
    case TR_BILIN_COUNT: v = gv.present; break;
    case TR_BILIN_COUNT_DISTINCT: v = (__int128)gv.distinct; break;
    case TR_BILIN_SUM_DISTINCT: if (!gv.present) { *is_null = 1; return TR_MATH_OK; } v = tr_i128_get(gv.sum_distinct); break;
    case TR_BILIN_MIN: if (!gv.present) { *is_null = 1; return TR_MATH_OK; } v = gv.min; break;
    default: if (!gv.present) { *is_null = 1; return TR_MATH_OK; } v = gv.max; break;
    }
    tr_bigint *x = big_i128(v, budget, refusal, &rc);
    return make_int(x, rc, out, budget, refusal);
}
int32_t tr_bilin_values_equal(const tr_bilin_values *a, const tr_bilin_values *b) {
    if (!values_valid(a) || !values_valid(b)) return TR_MATH_INVALID;
    if (a->kind != b->kind || a->ngroups != b->ngroups || a->column != b->column) return 0;
    const uint64_t G = a->ngroups;
    if (memcmp(a->rows, b->rows, (size_t)G * 8) || memcmp(a->nulls, b->nulls, (size_t)G * 8) ||
        memcmp(a->present, b->present, (size_t)G * 8)) return 0;
    if (a->kind == TR_BILIN_EXTREMUM) {
        for (uint64_t g = 0; g < G; g++)
            if (a->present[g] && (a->mn[g] != b->mn[g] || a->mnc[g] != b->mnc[g] || a->mx[g] != b->mx[g] || a->mxc[g] != b->mxc[g]))
                return 0;
        return 1;
    }
    return a->entries == b->entries && !memcmp(a->offset, b->offset, (size_t)(G + 1) * 8) &&
           !memcmp(a->vals, b->vals, (size_t)a->entries * 8) && !memcmp(a->mult, b->mult, (size_t)a->entries * 8);
}
