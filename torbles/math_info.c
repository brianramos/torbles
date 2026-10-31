/* DIST-003: entropy, mutual information and contingency statistics of a joint histogram's support; see math_info.h.
 * SPDX-License-Identifier: Apache-2.0 */
#include "math_info.h"
#include "mathematics.h"
#include "math_domain.h"
#include "math_carriers.h"
#include "math_handles.h"
#include <stdlib.h>
#include <string.h>

TR_MATH_ABI(tr_info_view, TR_INFO_ABI);            /* the ctypes boundary of DIST-003 (math_handles.h) */
TR_MATH_ABI(tr_info_support_info, TR_INFO_ABI);
TR_MATH_ABI(tr_info_request, TR_INFO_ABI);
TR_MATH_ABI(tr_info_result, TR_INFO_ABI);

_Static_assert(sizeof(tr_info_view) == 80, "tr_info_view ABI");
_Static_assert(sizeof(tr_info_support_info) == 104, "tr_info_support_info ABI");
_Static_assert(sizeof(tr_info_request) == 32, "tr_info_request ABI");
_Static_assert(sizeof(tr_info_result) == 136, "tr_info_result ABI");

#define SUPPORT_MAGIC UINT32_C(0x46534e49)
#define STATE_MAGIC UINT32_C(0x54534e49)
#define LN2_RD 0x1.62e42fefa39efp-1                 /* the doubles just below and above ln 2 */
#define LN2_RU 0x1.62e42fefa39f0p-1

/* one kept joint cell while the support is built (40 bytes) */
typedef struct {
    uint32_t g;
    uint8_t nulls, pad[3];
    int64_t v[3];
    uint64_t m;
} info_entry;

struct tr_info_support_s {
    uint32_t magic;
    int32_t k, mass, exclude, npairs, wscale;
    int32_t scale[3];
    uint64_t G, nslots, ntab, joint_cells, nlog, nchi, max_cells, present, bytes;
    uint64_t *rows, *total;        /* [G] */
    uint64_t *toff;                /* [ntab + 1] the first entry of each table */
    uint64_t *goff;                /* [ntab (G + 1)] group offsets within each table */
    uint64_t *poff;                /* [npairs + 1] the first chi entry of each pair */
    uint64_t *cgoff;               /* [npairs (G + 1)] group offsets within each pair */
    uint64_t *lkey, *lmass;        /* [nlog] */
    uint64_t *ckey, *co, *cma, *cmb, *cia, *cib;   /* [nchi] */
};

struct tr_info_state_s {
    uint32_t magic;
    int32_t k, mass, exclude, wscale, reserved;
    uint64_t G, nslots, ntab, bytes;
    uint64_t *rows, *total, *cells;   /* [G], [G], [G ntab] */
    uint64_t *sums;                   /* [G nslots 8] */
};

/* ------------------------------------------------------------------ helpers */
static void set_error(int32_t *error, int32_t code) { if (error) *error = code; }
static void *zalloc(uint64_t bytes) {
    if (bytes >= (uint64_t)SIZE_MAX) return NULL;
    return calloc(1, bytes ? (size_t)bytes : 1);
}
static const tr_math_budget *budget_of(const tr_math_budget *b, tr_math_budget *dflt) {
    if (b) return b;
    tr_budget_default(dflt);
    return dflt;
}
static void need_zero(tr_math_need *n) { for (int i = 0; i < TR_BUDGET_RESOURCES; i++) n->need[i] = 0; }
static int32_t check_need(const tr_math_budget *b, const tr_math_need *n, tr_math_refusal *r) {
    int32_t rc = tr_budget_check(b, n, r);
    if (rc) return rc;
    for (int32_t i = 0; i < TR_BUDGET_RESOURCES; i++)
        if (n->need[i] == TR_BUDGET_UNKNOWN) {
            if (r) { r->resource = i; r->refused = 1; r->need = n->need[i]; r->limit = b->limit[i]; }
            return TR_MATH_BUDGET;
        }
    return TR_MATH_OK;
}
static int support_valid(const tr_info_support *s) { return s && s->magic == SUPPORT_MAGIC; }
static int state_valid(const tr_info_state *s) { return s && s->magic == STATE_MAGIC; }
static int is_prefix(uint32_t S) { return (S & (S + 1)) == 0; }       /* 1, 3, 7: the leading coordinates */

/* ------------------------------------------------------------------ the kept cells of a histogram */
typedef struct {
    tr_dist_states st;
    tr_dist_plan plan;          /* dense: the domain */
    int32_t k, mass;
    uint32_t exclude;
} info_src;

static int32_t src_open(const tr_dist_hist *h, int32_t mass, uint32_t exclude, info_src *s, tr_dist_hist_info *hi) {
    memset(s, 0, sizeof *s);
    if (tr_dist_hist_get_states(h, &s->st) || tr_dist_hist_get_info(h, hi)) return TR_MATH_INVALID;
    if (mass != TR_DIST_MASS_COUNT && mass != TR_DIST_MASS_WEIGHT) return TR_MATH_INVALID;
    if (exclude >> hi->k) return TR_MATH_INVALID;
    if (mass == TR_DIST_MASS_WEIGHT && !hi->weighted) return TR_MATH_INVALID;
    if (mass == TR_DIST_MASS_WEIGHT && !hi->wnonneg) return TR_MATH_DOMAIN_ERROR;
    s->k = hi->k; s->mass = mass; s->exclude = exclude;
    if (s->st.repr == TR_DIST_DENSE) tr_dist_states_plan(&s->st, &s->plan);
    return TR_MATH_OK;
}
static void src_group(const info_src *s, uint64_t g, uint64_t *b, uint64_t *e) {
    if (s->st.repr == TR_DIST_DENSE) { *b = g * s->st.cells; *e = *b + s->st.cells; }
    else { *b = s->st.offset[g]; *e = s->st.offset[g + 1]; }
}
/* cell i of group g: 1 when it is kept (nonzero mass, no NULL in an excluded coordinate), with its entry */
static int src_cell(const info_src *s, uint64_t g, uint64_t i, info_entry *x) {
    if (!s->st.count[i]) return 0;
    unsigned __int128 m = s->mass == TR_DIST_MASS_COUNT ? s->st.count[i] : (unsigned __int128)tr_i128_get(s->st.wsum[i]);
    if (!m) return 0;
    memset(x, 0, sizeof *x);
    x->g = (uint32_t)g;
    if (s->st.repr == TR_DIST_DENSE) {
        uint64_t idx = i - g * s->st.cells;
        for (int c = 0; c < s->k; c++) {
            int null;
            x->v[c] = tr_dist_plan_coord(&s->plan, idx, c, &null);
            if (null) x->nulls |= (uint8_t)(1u << c);
        }
    } else {
        x->nulls = s->st.nulls[i];
        for (int c = 0; c < s->k; c++) x->v[c] = s->st.value[c][i];
    }
    if (x->nulls & s->exclude) return 0;
    x->m = (uint64_t)m;                        /* < 2^64: the group total was checked */
    return 1;
}
/* pass 1: kept cells, present groups; TR_MATH_UNSUPPORTED when a group's kept mass reaches 2^64 */
static int32_t src_count(const info_src *s, uint64_t *n0) {
    uint64_t n = 0;
    for (uint64_t g = 0; g < s->st.ngroups; g++) {
        uint64_t b, e;
        src_group(s, g, &b, &e);
        unsigned __int128 t = 0;
        for (uint64_t i = b; i < e; i++) {
            if (!s->st.count[i]) continue;
            unsigned __int128 m = s->mass == TR_DIST_MASS_COUNT ? s->st.count[i]
                                                                : (unsigned __int128)tr_i128_get(s->st.wsum[i]);
            info_entry x;
            if (!m || !src_cell(s, g, i, &x)) continue;              /* a structural zero, or NULL excluded */
            t += m;                                                  /* each m < 2^127: no wrap before the check */
            if ((t >> 64) != 0) return TR_MATH_UNSUPPORTED;
            n++;
        }
    }
    if (n > UINT32_MAX) return TR_MATH_UNSUPPORTED;                  /* cell ids are 32-bit */
    *n0 = n;
    return TR_MATH_OK;
}

/* ------------------------------------------------------------------ sorting by a projection (stable merge sort) */
static int proj_cmp(const info_entry *a, const info_entry *b, uint32_t S) {
    if (a->g != b->g) return a->g < b->g ? -1 : 1;
    for (int c = 0; c < 3; c++) {
        if (!((S >> c) & 1)) continue;
        int na = (a->nulls >> c) & 1, nb = (b->nulls >> c) & 1;
        if (na != nb) return na - nb;                                 /* a value before NULL */
        if (!na && a->v[c] != b->v[c]) return a->v[c] < b->v[c] ? -1 : 1;
    }
    return 0;
}
static void perm_sort(const info_entry *E, uint32_t *p, uint32_t *tmp, uint64_t n, uint32_t S) {
    const uint64_t RUN = 32;
    for (uint64_t s = 0; s < n; s += RUN) {
        uint64_t e = s + RUN < n ? s + RUN : n;
        for (uint64_t i = s + 1; i < e; i++) {
            uint32_t x = p[i];
            uint64_t j = i;
            while (j > s && proj_cmp(&E[p[j - 1]], &E[x], S) > 0) { p[j] = p[j - 1]; j--; }
            p[j] = x;
        }
    }
    uint32_t *src = p, *dst = tmp;
    for (uint64_t w = RUN; w < n; w *= 2) {
        for (uint64_t s = 0; s < n; s += 2 * w) {
            uint64_t m = s + w < n ? s + w : n, e = s + 2 * w < n ? s + 2 * w : n, i = s, j = m, o = s;
            while (i < m && j < e) dst[o++] = proj_cmp(&E[src[j]], &E[src[i]], S) < 0 ? src[j++] : src[i++];
            while (i < m) dst[o++] = src[i++];
            while (j < e) dst[o++] = src[j++];
        }
        uint32_t *t = src; src = dst; dst = t;
    }
    if (src != p) memcpy(p, src, (size_t)n * 4);
}

/* ------------------------------------------------------------------ the support */
static uint64_t support_final_bytes(uint64_t G, uint64_t ntab, uint64_t npairs, uint64_t nlog, uint64_t nchi) {
    uint64_t b = tr_budget_add(sizeof(tr_info_support), tr_budget_mul(G, 16));
    b = tr_budget_add(b, tr_budget_mul(tr_budget_add(tr_budget_mul(ntab + npairs, G + 1), ntab + npairs + 2), 8));
    b = tr_budget_add(b, tr_budget_mul(nlog, 16));
    return tr_budget_add(b, tr_budget_mul(nchi, 48));
}
static uint64_t support_scratch_bytes(uint64_t n0, uint64_t ntab) {
    /* the entries, a permutation and a run id per nonempty subset, the merge buffer, per-group counters */
    uint64_t per = tr_budget_add(sizeof(info_entry), tr_budget_add(tr_budget_mul(ntab - 1, 8), 4));
    return tr_budget_mul(n0, per);
}
static void stage1_need(const info_src *s, uint64_t n0, tr_math_need *need) {
    uint64_t ntab = (uint64_t)1 << s->k, G = s->st.ngroups;
    need_zero(need);
    need->need[TR_BUDGET_VARIABLES] = (uint64_t)s->k;
    need->need[TR_BUDGET_STATES] = n0;
    need->need[TR_BUDGET_BYTES] = tr_budget_add(support_scratch_bytes(n0, ntab), tr_budget_mul(G, 8));
}

static void support_release(tr_info_support *s) {
    if (!s) return;
    free(s->rows); free(s->total); free(s->toff); free(s->goff); free(s->poff); free(s->cgoff);
    free(s->lkey); free(s->lmass); free(s->ckey); free(s->co); free(s->cma); free(s->cmb); free(s->cia); free(s->cib);
    s->magic = 0;
    free(s);
}
void tr_info_support_free(tr_info_support *s) { if (support_valid(s)) support_release(s); }

int32_t tr_info_support_need(const tr_dist_hist *h, int32_t mass, uint32_t exclude, tr_math_need *need) {
    info_src s;
    tr_dist_hist_info hi;
    if (!need) return TR_MATH_INVALID;
    int32_t rc = src_open(h, mass, exclude, &s, &hi);
    uint64_t n0 = 0;
    if (!rc) rc = src_count(&s, &n0);
    if (rc) return rc;
    stage1_need(&s, n0, need);
    return TR_MATH_OK;
}

tr_info_support *tr_info_support_build(const tr_dist_hist *h, int32_t mass, uint32_t exclude,
                                       const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    info_src src;
    tr_dist_hist_info hi;
    int32_t rc = src_open(h, mass, exclude, &src, &hi);
    uint64_t n0 = 0;
    if (!rc) rc = src_count(&src, &n0);
    if (rc) { set_error(error, rc); return NULL; }
    const int k = src.k;
    const uint64_t G = src.st.ngroups, ntab = (uint64_t)1 << k, npairs = tr_info_npairs(k), nslots = tr_info_nslots(k);
    tr_math_budget dflt;
    const tr_math_budget *bud = budget_of(budget, &dflt);
    tr_math_need need;
    stage1_need(&src, n0, &need);
    rc = check_need(bud, &need, refusal);
    if (rc) { set_error(error, rc); return NULL; }
    /* the scratch: entries, per subset S >= 1 a permutation and the run id of each entry, the merge buffer */
    info_entry *E = zalloc(tr_budget_mul(n0, sizeof(info_entry)));
    uint32_t *perm = zalloc(tr_budget_mul(tr_budget_mul(n0, ntab - 1), 4));
    uint32_t *runid = zalloc(tr_budget_mul(tr_budget_mul(n0, ntab - 1), 4));
    uint32_t *tmp = zalloc(tr_budget_mul(n0, 4));
    uint64_t *gtot = zalloc(tr_budget_mul(G, 8));
    uint64_t runs[8] = {0}, maxc = 0;
    if (!E || !perm || !runid || !tmp || !gtot) {
        free(E); free(perm); free(runid); free(tmp); free(gtot);
        set_error(error, TR_MATH_MEMORY); return NULL;
    }
    uint64_t pos = 0;
    for (uint64_t g = 0; g < G; g++) {
        uint64_t b, e;
        src_group(&src, g, &b, &e);
        for (uint64_t i = b; i < e; i++) {
            info_entry x;
            if (src_cell(&src, g, i, &x)) { E[pos++] = x; gtot[g] += x.m; }
        }
    }
    uint64_t present_total = 0;
    for (uint64_t g = 0; g < G; g++) present_total += gtot[g] != 0;
    runs[0] = present_total;                                         /* T(empty): one cell per group with mass */
    for (uint32_t S = 1; S < ntab; S++) {
        uint32_t *p = perm + (uint64_t)(S - 1) * n0, *rid = runid + (uint64_t)(S - 1) * n0;
        for (uint64_t i = 0; i < n0; i++) p[i] = (uint32_t)i;
        if (!is_prefix(S)) perm_sort(E, p, tmp, n0, S);               /* the joint order is sorted by any prefix */
        uint64_t r = 0, gcells = 0;
        uint32_t gcur = UINT32_MAX;
        for (uint64_t t = 0; t < n0; t++) {
            uint32_t i = p[t];
            if (t == 0 || proj_cmp(&E[p[t - 1]], &E[i], S) != 0) {
                r++;
                if (E[i].g != gcur) { gcur = E[i].g; gcells = 0; }
                gcells++;
                if (gcells > maxc) maxc = gcells;
            }
            rid[i] = (uint32_t)(r - 1);
        }
        runs[S] = r;
    }
    uint64_t nlog = 0, nchi = 0, pruns[6] = {0};
    for (uint64_t S = 0; S < ntab; S++) nlog += runs[S];
    for (uint32_t p = 0; p < npairs; p++) {
        uint32_t A = 0, B = 0;
        tr_info_pair_masks(k, p, &A, &B);
        pruns[p] = runs[A | B];
        nchi += pruns[p];
    }
    tr_math_need fin = need;
    fin.need[TR_BUDGET_SUPPORT] = maxc;
    fin.need[TR_BUDGET_STATES] = tr_budget_add(nlog, nchi);
    fin.need[TR_BUDGET_BYTES] = tr_budget_add(need.need[TR_BUDGET_BYTES], support_final_bytes(G, ntab, npairs, nlog, nchi));
    rc = check_need(bud, &fin, refusal);
    tr_info_support *s = rc ? NULL : zalloc(sizeof *s);
    if (s) {
        s->rows = zalloc(G * 8); s->total = gtot; gtot = NULL;
        s->toff = zalloc((ntab + 1) * 8); s->goff = zalloc(ntab * (G + 1) * 8);
        s->poff = zalloc((npairs + 1) * 8); s->cgoff = zalloc((npairs ? npairs : 1) * (G + 1) * 8);
        s->lkey = zalloc(nlog * 8); s->lmass = zalloc(nlog * 8);
        s->ckey = zalloc(nchi * 8); s->co = zalloc(nchi * 8); s->cma = zalloc(nchi * 8); s->cmb = zalloc(nchi * 8);
        s->cia = zalloc(nchi * 8); s->cib = zalloc(nchi * 8);
        if (!s->rows || !s->toff || !s->goff || !s->poff || !s->cgoff || !s->lkey || !s->lmass || !s->ckey || !s->co ||
            !s->cma || !s->cmb || !s->cia || !s->cib) { support_release(s); s = NULL; rc = TR_MATH_MEMORY; }
    } else if (!rc) rc = TR_MATH_MEMORY;
    if (!s) {
        free(E); free(perm); free(runid); free(tmp); free(gtot);
        set_error(error, rc); return NULL;
    }
    s->k = k; s->mass = mass; s->exclude = (int32_t)exclude; s->npairs = (int32_t)npairs;
    s->wscale = mass == TR_DIST_MASS_WEIGHT ? hi.wscale : 0;
    for (int c = 0; c < 3; c++) s->scale[c] = c < k ? hi.scale[c] : 0;
    s->G = G; s->nslots = nslots; s->ntab = ntab; s->joint_cells = n0; s->nlog = nlog; s->nchi = nchi;
    s->max_cells = maxc; s->present = present_total;
    s->bytes = support_final_bytes(G, ntab, npairs, nlog, nchi);
    memcpy(s->rows, src.st.rows, (size_t)G * 8);
    for (uint64_t S = 0; S < ntab; S++) s->toff[S + 1] = s->toff[S] + runs[S];
    /* T(empty): N per group */
    {
        uint64_t *go = s->goff, r = 0;
        for (uint64_t g = 0; g < G; g++) {
            go[g] = r;
            if (s->total[g]) { s->lkey[r] = g * nslots; s->lmass[r] = s->total[g]; r++; }
        }
        go[G] = r;
    }
    for (uint32_t S = 1; S < ntab; S++) {
        uint32_t *p = perm + (uint64_t)(S - 1) * n0, *rid = runid + (uint64_t)(S - 1) * n0;
        uint64_t base = s->toff[S], *go = s->goff + (uint64_t)S * (G + 1), last = UINT64_MAX;
        for (uint64_t g = 0; g <= G; g++) go[g] = 0;
        for (uint64_t t = 0; t < n0; t++) {
            uint32_t i = p[t];
            uint64_t r = rid[i];
            s->lmass[base + r] += E[i].m;
            if (r != last) { s->lkey[base + r] = (uint64_t)E[i].g * nslots + S; go[E[i].g + 1]++; last = r; }
        }
        for (uint64_t g = 0; g < G; g++) go[g + 1] += go[g];
    }
    for (uint32_t p = 0; p < npairs; p++) {
        uint32_t A = 0, B = 0;
        tr_info_pair_masks(k, p, &A, &B);
        const uint32_t U = A | B;
        const uint32_t *pu = perm + (uint64_t)(U - 1) * n0, *ru = runid + (uint64_t)(U - 1) * n0;
        const uint32_t *ra = runid + (uint64_t)(A - 1) * n0, *rb = runid + (uint64_t)(B - 1) * n0;
        uint64_t e = s->poff[p], *cg = s->cgoff + (uint64_t)p * (G + 1), last = UINT64_MAX;
        s->poff[p + 1] = e + pruns[p];
        for (uint64_t g = 0; g <= G; g++) cg[g] = 0;
        for (uint64_t t = 0; t < n0; t++) {
            uint32_t i = pu[t];
            uint64_t r = ru[i];
            if (r == last) continue;
            last = r;
            s->ckey[e] = (uint64_t)E[i].g * nslots + ntab + p;
            s->co[e] = s->lmass[s->toff[U] + r];
            s->cia[e] = ra[i]; s->cib[e] = rb[i];
            s->cma[e] = s->lmass[s->toff[A] + ra[i]];
            s->cmb[e] = s->lmass[s->toff[B] + rb[i]];
            cg[E[i].g + 1]++;
            e++;
        }
        for (uint64_t g = 0; g < G; g++) cg[g + 1] += cg[g];
    }
    free(E); free(perm); free(runid); free(tmp);
    s->magic = SUPPORT_MAGIC;
    set_error(error, TR_MATH_OK);
    return s;
}

int32_t tr_info_support_get_view(const tr_info_support *s, tr_info_view *v) {
    if (!support_valid(s) || !v) return TR_MATH_INVALID;
    memset(v, 0, sizeof *v);
    v->ngroups = s->G; v->nslots = s->nslots; v->nlog = s->nlog; v->nchi = s->nchi;
    v->lkey = s->lkey; v->lmass = s->lmass; v->ckey = s->ckey; v->co = s->co; v->cma = s->cma; v->cmb = s->cmb;
    return TR_MATH_OK;
}
int32_t tr_info_support_get_info(const tr_info_support *s, tr_info_support_info *out) {
    if (!support_valid(s) || !out) return TR_MATH_INVALID;
    memset(out, 0, sizeof *out);
    out->ngroups = s->G; out->present = s->present; out->joint_cells = s->joint_cells; out->nlog = s->nlog;
    out->nchi = s->nchi; out->nslots = s->nslots; out->max_cells = s->max_cells; out->bytes = s->bytes;
    uint64_t t = 0;
    for (uint64_t g = 0; g < s->G; g++) t = tr_budget_add(t, s->total[g]);
    out->total = t;
    out->k = s->k; out->mass = s->mass; out->exclude = s->exclude; out->npairs = s->npairs;
    for (int c = 0; c < 3; c++) out->scale[c] = s->scale[c];
    out->wscale = s->wscale;
    return TR_MATH_OK;
}
int32_t tr_info_support_group(const tr_info_support *s, uint64_t g, uint64_t *rows, uint64_t *total, uint64_t *cells) {
    if (!support_valid(s) || g >= s->G) return TR_MATH_INVALID;
    if (rows) *rows = s->rows[g];
    if (total) *total = s->total[g];
    if (cells)
        for (uint64_t S = 0; S < s->ntab; S++) cells[S] = s->goff[S * (s->G + 1) + g + 1] - s->goff[S * (s->G + 1) + g];
    return TR_MATH_OK;
}
uint64_t tr_info_device_need(const tr_info_view *v) { return v ? tr_info_device_bytes(v) : TR_BUDGET_UNKNOWN; }
int32_t tr_info_term_bounds(int32_t kind, uint64_t a, uint64_t b, uint64_t c, uint64_t out[8]) {
    tr_dist_i256 L, H;
    if (!out || !a) return TR_MATH_INVALID;
    if (kind == 1) tr_info_log_term(a, &L, &H);
    else if (kind == 2 && a <= b && a <= c) tr_info_chi_term(a, b, c, &L, &H);
    else return TR_MATH_INVALID;
    for (int i = 0; i < 4; i++) { out[i] = L.w[i]; out[4 + i] = H.w[i]; }
    return TR_MATH_OK;
}
int32_t tr_info_ln_bounds(uint64_t m, uint64_t out[4]) {
    if (!out || !m) return TR_MATH_INVALID;
    unsigned __int128 lo, hi;
    tr_info_ln(m, &lo, &hi);
    out[0] = (uint64_t)lo; out[1] = (uint64_t)(lo >> 64); out[2] = (uint64_t)hi; out[3] = (uint64_t)(hi >> 64);
    return TR_MATH_OK;
}

/* ------------------------------------------------------------------ the sums (the CPU reference of the kernels) */
int32_t tr_info_sums(const tr_info_view *v, int64_t *sums) {
    if (!tr_info_view_valid(v) || !sums) return TR_MATH_INVALID;
    const uint64_t total = v->ngroups * v->nslots;
    for (uint64_t i = 0; i < v->nlog; i++) if (v->lkey[i] >= total || !v->lmass[i]) return TR_MATH_INVALID;
    for (uint64_t i = 0; i < v->nchi; i++)
        if (v->ckey[i] >= total || !v->co[i] || v->co[i] > v->cma[i] || v->co[i] > v->cmb[i]) return TR_MATH_INVALID;
    memset(sums, 0, (size_t)total * 64);
    uint64_t *w = (uint64_t *)sums;
    /* a local cache of the terms of small masses (the same values as computing them) */
    enum { CACHE = 256 };
    tr_dist_i256 cl[CACHE], ch[CACHE];
    uint8_t have[CACHE];
    memset(have, 0, sizeof have);
    for (uint64_t i = 0; i < v->nlog; i++) {
        uint64_t m = v->lmass[i];
        tr_dist_i256 L, H;
        if (m < CACHE && have[m]) { L = cl[m]; H = ch[m]; }
        else {
            tr_info_log_term(m, &L, &H);
            if (m < CACHE) { cl[m] = L; ch[m] = H; have[m] = 1; }
        }
        tr_info_add256(w + v->lkey[i] * 8, L);
        tr_info_add256(w + v->lkey[i] * 8 + 4, H);
    }
    for (uint64_t i = 0; i < v->nchi; i++) {
        tr_dist_i256 L, H;
        tr_info_chi_term(v->co[i], v->cma[i], v->cmb[i], &L, &H);
        tr_info_add256(w + v->ckey[i] * 8, L);
        tr_info_add256(w + v->ckey[i] * 8 + 4, H);
    }
    return TR_MATH_OK;
}

/* ------------------------------------------------------------------ the state */
static void state_release(tr_info_state *s) {
    if (!s) return;
    free(s->rows); free(s->total); free(s->cells); free(s->sums);
    s->magic = 0;
    free(s);
}
void tr_info_state_free(tr_info_state *s) { if (state_valid(s)) state_release(s); }
uint64_t tr_info_state_bytes(const tr_info_state *s) { return state_valid(s) ? s->bytes : 0; }

static tr_dist_i256 words(const uint64_t *w) { tr_dist_i256 r; for (int i = 0; i < 4; i++) r.w[i] = w[i]; return r; }
static int u256_le(tr_dist_i256 a, tr_dist_i256 b) {
    for (int i = 3; i >= 0; i--) if (a.w[i] != b.w[i]) return a.w[i] < b.w[i];
    return 1;
}

tr_info_state *tr_info_state_build(const tr_info_support *sp, const int64_t *sums, const tr_math_budget *budget,
                                   tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    if (!support_valid(sp)) { set_error(error, TR_MATH_INVALID); return NULL; }
    const uint64_t G = sp->G, slots = tr_budget_mul(G, sp->nslots);
    tr_math_need need;
    need_zero(&need);
    need.need[TR_BUDGET_VARIABLES] = (uint64_t)sp->k;
    need.need[TR_BUDGET_STATES] = slots;
    uint64_t bytes = tr_budget_add(sizeof(tr_info_state), tr_budget_add(tr_budget_mul(slots, 64),
                                   tr_budget_mul(G, tr_budget_add(16, tr_budget_mul(sp->ntab, 8)))));
    need.need[TR_BUDGET_BYTES] = bytes;
    tr_math_budget dflt;
    int32_t rc = check_need(budget_of(budget, &dflt), &need, refusal);
    if (rc) { set_error(error, rc); return NULL; }
    tr_info_state *s = zalloc(sizeof *s);
    if (!s) { set_error(error, TR_MATH_MEMORY); return NULL; }
    s->rows = zalloc(G * 8); s->total = zalloc(G * 8); s->cells = zalloc(G * sp->ntab * 8); s->sums = zalloc(slots * 64);
    if (!s->rows || !s->total || !s->cells || !s->sums) { state_release(s); set_error(error, TR_MATH_MEMORY); return NULL; }
    s->k = sp->k; s->mass = sp->mass; s->exclude = sp->exclude; s->wscale = sp->wscale;
    s->G = G; s->nslots = sp->nslots; s->ntab = sp->ntab; s->bytes = bytes;
    memcpy(s->rows, sp->rows, (size_t)G * 8);
    memcpy(s->total, sp->total, (size_t)G * 8);
    for (uint64_t g = 0; g < G; g++) tr_info_support_group(sp, g, NULL, NULL, s->cells + g * sp->ntab);
    if (!sums) {
        tr_info_view v;
        tr_info_support_get_view(sp, &v);
        rc = tr_info_sums(&v, (int64_t *)s->sums);
    } else {
        memcpy(s->sums, sums, (size_t)slots * 64);
        /* the readback: lower <= upper below 2^255, and a slot without cells sums nothing */
        for (uint64_t i = 0; i < slots && !rc; i++) {
            tr_dist_i256 L = words(s->sums + i * 8), H = words(s->sums + i * 8 + 4);
            uint64_t g = i / s->nslots, slot = i % s->nslots;
            uint64_t ncell = slot < s->ntab ? s->cells[g * s->ntab + slot] : 1;
            int zero = tr_dist_w_is_zero(L) && tr_dist_w_is_zero(H);
            if ((L.w[3] >> 63) || (H.w[3] >> 63) || !u256_le(L, H) || (!ncell && !zero)) rc = TR_MATH_INVALID;
        }
    }
    if (rc) { state_release(s); set_error(error, rc); return NULL; }
    s->magic = STATE_MAGIC;
    set_error(error, TR_MATH_OK);
    return s;
}

int32_t tr_info_state_equal(const tr_info_state *a, const tr_info_state *b) {
    if (!state_valid(a) || !state_valid(b)) return TR_MATH_INVALID;
    if (a->k != b->k || a->mass != b->mass || a->exclude != b->exclude || a->G != b->G) return 0;
    uint64_t G = a->G;
    return !memcmp(a->rows, b->rows, (size_t)G * 8) && !memcmp(a->total, b->total, (size_t)G * 8) &&
           !memcmp(a->cells, b->cells, (size_t)(G * a->ntab) * 8) &&
           !memcmp(a->sums, b->sums, (size_t)(G * a->nslots) * 64);
}

/* ------------------------------------------------------------------ requests */
/* [RD, RU] of x / (d 2^s) for a nonnegative 256-bit x, 1 <= d < 2^64, 0 <= s <= 256 */
static void enclose_ratio(tr_dist_i256 x, uint64_t d, int s, double *lo, double *hi) {
    if (tr_dist_w_is_zero(x)) { *lo = *hi = 0.0; return; }
    /* z = floor(x 2^128 / d) as 6 limbs (4 integer limbs, 2 fraction limbs), sticky = a nonzero remainder */
    uint64_t z[6], r = 0;
    uint64_t num[6] = {0, 0, x.w[0], x.w[1], x.w[2], x.w[3]};
    for (int i = 5; i >= 0; i--) {
        unsigned __int128 cur = ((unsigned __int128)r << 64) | num[i];
        z[i] = (uint64_t)(cur / d); r = (uint64_t)(cur % d);
    }
    int sticky = r != 0, top = 5;
    while (top > 0 && !z[top]) top--;
    int t = 64 * top + 63;
    while (!((z[top] >> (t - 64 * top)) & 1)) t--;                    /* the highest set bit of z */
    uint64_t T;
    int e;
    if (t <= 52) { T = z[0]; e = 0; }
    else {
        int sh = t - 52;                                               /* T = z >> sh: 53 bits */
        int q = sh / 64, b = sh % 64;
        T = z[q] >> b;
        if (b && q + 1 < 6) T |= z[q + 1] << (64 - b);
        T &= (UINT64_C(1) << 53) - 1;
        for (int i = 0; i < q && !sticky; i++) sticky = z[i] != 0;
        if (b && !sticky) sticky = (z[q] & ((UINT64_C(1) << b) - 1)) != 0;
        e = sh;
    }
    *lo = ldexp((double)T, e - 128 - s);
    *hi = sticky ? ldexp((double)(T + 1), e - 128 - s) : *lo;
}
static tr_dist_i256 slot_word(const tr_info_state *s, uint64_t g, uint64_t slot, int upper) {
    return words(s->sums + (g * s->nslots + slot) * 8 + (upper ? 4 : 0));
}
static tr_dist_i256 w_add(tr_dist_i256 a, tr_dist_i256 b) { tr_dist_i256 r; tr_dist_w_add(a, b, &r); return r; }
static tr_dist_i256 w_sub(tr_dist_i256 a, tr_dist_i256 b) { tr_dist_i256 r; tr_dist_w_sub(a, b, &r); return r; }
static tr_dist_i256 w_clamp(tr_dist_i256 a) { return tr_dist_w_negative(a) ? tr_dist_w_zero() : a; }
/* a nonnegative 256-bit a times u < 2^64 (below 2^255 by the bounds of the sums) */
static tr_dist_i256 w_mulu(tr_dist_i256 a, uint64_t u) {
    tr_dist_i256 r;
    unsigned __int128 c = 0;
    for (int i = 0; i < 4; i++) { c += (unsigned __int128)a.w[i] * u; r.w[i] = (uint64_t)c; c >>= 64; }
    return r;
}

static void eval_one(const tr_info_state *s, const tr_info_request *q, tr_info_result *r) {
    memset(r, 0, sizeof *r);
    r->num_lo = r->num_hi = tr_dist_w_zero();
    r->den = 1;
    const int32_t op = q->op;
    const uint32_t K = (1u << s->k) - 1, a = q->a, b = q->b, c = q->c;
    const int two = op != TR_INFO_OP_ENTROPY, three = op == TR_INFO_OP_CONDITIONAL_MI;
    const int units_ok = q->units == TR_INFO_NATS || (q->units == TR_INFO_BITS && op <= TR_INFO_OP_CONDITIONAL_MI);
    if (op < 1 || op > TR_INFO_OPS || !units_ok || q->reserved || q->group >= s->G || !a || (two && !b) || (!two && b) ||
        (three && !c) || (!three && c) || (a & b) || (a & c) || (b & c)) { r->status = TR_MATH_INVALID; return; }
    if ((a | b | c) & ~K) { r->status = TR_MATH_UNSUPPORTED; return; }   /* no joint support over these coordinates */
    const uint64_t g = q->group, N = s->total[g];
    const uint64_t *cells = s->cells + g * s->ntab;
    r->total = N;
    r->rows = cells[a];
    r->cols = b ? cells[b] : 0;
    if (!s->rows[g] || !N) { r->empty = 1; return; }
    tr_dist_i256 lo, hi;
#define LO(S) slot_word(s, g, (S), 0)
#define HI(S) slot_word(s, g, (S), 1)
    int shift = TR_INFO_SCALE;
    uint64_t den = N;
    switch (op) {
    case TR_INFO_OP_ENTROPY:
        lo = w_sub(LO(0), HI(a)); hi = w_sub(HI(0), LO(a)); break;
    case TR_INFO_OP_CONDITIONAL_ENTROPY:
        lo = w_sub(LO(b), HI(a | b)); hi = w_sub(HI(b), LO(a | b)); break;
    case TR_INFO_OP_MUTUAL_INFORMATION:
    case TR_INFO_OP_G_TEST:
        lo = w_sub(w_sub(w_add(LO(0), LO(a | b)), HI(a)), HI(b));
        hi = w_sub(w_sub(w_add(HI(0), HI(a | b)), LO(a)), LO(b));
        if (op == TR_INFO_OP_G_TEST) {
            lo = w_add(w_clamp(lo), w_clamp(lo)); hi = w_add(w_clamp(hi), w_clamp(hi)); den = 1;
            r->df = (r->rows - 1) * (r->cols - 1);
        }
        break;
    case TR_INFO_OP_CONDITIONAL_MI:
        lo = w_sub(w_sub(w_add(LO(a | b | c), LO(c)), HI(a | c)), HI(b | c));
        hi = w_sub(w_sub(w_add(HI(a | b | c), HI(c)), LO(a | c)), LO(b | c));
        break;
    default: {                                                      /* CHI_SQUARE, CRAMERS_V */
        int p = tr_info_pair_index(s->k, a, b);
        if (p < 0) { r->status = TR_MATH_UNSUPPORTED; return; }
        tr_dist_i256 one = tr_dist_w_zero();
        one.w[2] = 1;                                                /* 2^128 */
        lo = w_clamp(w_sub(LO(s->ntab + (uint64_t)p), one));
        hi = w_clamp(w_sub(HI(s->ntab + (uint64_t)p), one));
        shift = TR_INFO_CHI_SCALE;
        r->df = (r->rows - 1) * (r->cols - 1);
        if (op == TR_INFO_OP_CHI_SQUARE) { lo = w_mulu(lo, N); hi = w_mulu(hi, N); den = 1; }
        else {
            uint64_t m = r->rows < r->cols ? r->rows : r->cols;
            r->squared = 1;
            if (m < 2) { r->empty = 1; return; }
            den = m - 1;
        }
    }
    }
#undef LO
#undef HI
    lo = w_clamp(lo); hi = w_clamp(hi);
    r->num_lo = lo; r->num_hi = hi; r->den = den; r->shift = shift;
    double l, h, t;
    enclose_ratio(lo, den, shift, &l, &t);
    enclose_ratio(hi, den, shift, &t, &h);
    if (q->units == TR_INFO_BITS) { l = tr_div_dir(l, LN2_RU, 0); h = tr_div_dir(h, LN2_RD, 1); }
    if (op == TR_INFO_OP_CRAMERS_V) {
        l = tr_sqrt_dir(l, 0); h = tr_sqrt_dir(h, 1);
        if (h > 1.0) h = 1.0;                                        /* V <= 1 */
        if (l > h) l = h;
    }
    r->lo = l; r->hi = h;
}

int32_t tr_info_query(const tr_info_state *s, const tr_info_request *req, uint64_t n, tr_info_result *out,
                      const tr_math_budget *budget, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!state_valid(s) || (n && (!req || !out))) return TR_MATH_INVALID;
    tr_math_budget dflt;
    int32_t rc = tr_budget_check_one(budget_of(budget, &dflt), TR_BUDGET_OUTPUT, n, refusal);
    if (rc) return rc;
    for (uint64_t i = 0; i < n; i++) eval_one(s, &req[i], &out[i]);
    return TR_MATH_OK;
}

/* ------------------------------------------------------------------ exact contingency values (FND-009 carriers) */
static tr_bigint *big_u64(uint64_t v, const tr_math_budget *b, tr_math_refusal *r, int32_t *e) {
    return tr_bigint_from_limbs(v ? 1 : 0, &v, 1, b, r, e);
}
static tr_bigint *big_u128(unsigned __int128 v, const tr_math_budget *b, tr_math_refusal *r, int32_t *e) {
    uint64_t l[2] = {(uint64_t)v, (uint64_t)(v >> 64)};
    return tr_bigint_from_limbs(v ? 1 : 0, l, 2, b, r, e);
}
/* *acc = *acc op x (op: 0 add, 1 mul), freeing the old value; 0 or the error */
static int32_t big_step(tr_bigint **acc, const tr_bigint *x, int op, const tr_math_budget *b, tr_math_refusal *r) {
    int32_t e = TR_MATH_OK;
    tr_bigint *t = op ? tr_bigint_mul(*acc, x, b, r, &e) : tr_bigint_add(*acc, x, b, r, &e);
    if (!t) return e ? e : TR_MATH_MEMORY;
    tr_bigint_free(*acc);
    *acc = t;
    return TR_MATH_OK;
}
/* the lcm of n masses */
static int32_t big_lcm(const uint64_t *m, uint64_t n, tr_bigint **out, const tr_math_budget *b, tr_math_refusal *r) {
    int32_t e = TR_MATH_OK;
    tr_bigint *L = big_u64(1, b, r, &e);
    if (!L) return e;
    for (uint64_t i = 0; i < n && !e; i++) {
        uint64_t rem;
        e = tr_bigint_mod_u64(L, m[i], &rem);
        if (e) break;
        uint64_t f = m[i] / tr_gcd_u64(m[i], rem);
        if (f > 1) {
            tr_bigint *x = big_u64(f, b, r, &e);
            if (x) { e = big_step(&L, x, 1, b, r); tr_bigint_free(x); }
        }
    }
    if (e) { tr_bigint_free(L); return e; }
    *out = L;
    return TR_MATH_OK;
}
static uint64_t bits64(uint64_t v) { uint64_t n = 0; while (v) { n++; v >>= 1; } return n; }

int32_t tr_info_exact(const tr_info_support *s, uint64_t g, int32_t op, uint32_t a, uint32_t b,
                      const tr_math_budget *budget, tr_math_refusal *refusal, tr_bigint **num, tr_bigint **den) {
    tr_budget_none(refusal);
    if (!support_valid(s) || !num || !den || g >= s->G || (op != TR_INFO_OP_CHI_SQUARE && op != TR_INFO_OP_CRAMERS_V) ||
        !a || !b || (a & b)) return TR_MATH_INVALID;
    *num = *den = NULL;
    const uint32_t K = (uint32_t)(s->ntab - 1);
    if ((a | b) & ~K) return TR_MATH_UNSUPPORTED;
    int p = tr_info_pair_index(s->k, a, b);
    if (p < 0) return TR_MATH_UNSUPPORTED;
    uint32_t A = 0, B = 0;
    tr_info_pair_masks(s->k, (uint32_t)p, &A, &B);
    const uint64_t N = s->total[g], G1 = s->G + 1;
    if (!s->rows[g] || !N) return TR_MATH_OK;                         /* empty */
    const uint64_t ga = s->goff[A * G1 + g], na = s->goff[A * G1 + g + 1] - ga;
    const uint64_t gb = s->goff[B * G1 + g], nb = s->goff[B * G1 + g + 1] - gb;
    const uint64_t *ma = s->lmass + s->toff[A] + ga, *mb = s->lmass + s->toff[B] + gb;
    const uint64_t e0 = s->poff[p] + s->cgoff[(uint64_t)p * G1 + g], e1 = s->poff[p] + s->cgoff[(uint64_t)p * G1 + g + 1];
    const uint64_t q = (na < nb ? na : nb);
    if (op == TR_INFO_OP_CRAMERS_V && q < 2) return TR_MATH_OK;      /* empty: min(r, c) - 1 = 0 */
    /* every big number held, bounded from the bit lengths of the masses, before any is made */
    uint64_t bd = 1, be = 1;
    for (uint64_t i = 0; i < nb; i++) bd = tr_budget_add(bd, bits64(mb[i]));
    for (uint64_t i = 0; i < na; i++) be = tr_budget_add(be, bits64(ma[i]));
    uint64_t ld = bd / 64 + 2, le = be / 64 + 2, lt = tr_budget_add(tr_budget_add(ld, le), 6);
    uint64_t limbs = tr_budget_add(tr_budget_mul(nb, ld), tr_budget_mul(na, tr_budget_add(ld, 4)));
    limbs = tr_budget_add(limbs, tr_budget_add(tr_budget_mul(na, le), tr_budget_mul(8, lt)));
    tr_math_need need;
    need_zero(&need);
    need.need[TR_BUDGET_SUPPORT] = e1 - e0;
    need.need[TR_BUDGET_BYTES] = tr_budget_add(tr_budget_mul(limbs, 8), tr_budget_mul(tr_budget_add(na, nb), 16));
    tr_math_budget dflt;
    const tr_math_budget *bud = budget_of(budget, &dflt);
    int32_t rc = check_need(bud, &need, refusal);
    if (rc) return rc;
    tr_bigint *D = NULL, *L = NULL, *T = NULL, *DL = NULL, *diff = NULL, *nu = NULL, *de = NULL, *x = NULL;
    tr_bigint **w = calloc((size_t)(nb ? nb : 1), sizeof *w), **R = calloc((size_t)(na ? na : 1), sizeof *R);
    tr_bigq *Q = NULL;
    rc = (!w || !R) ? TR_MATH_MEMORY : TR_MATH_OK;
    if (!rc) rc = big_lcm(mb, nb, &D, bud, refusal);
    if (!rc) rc = big_lcm(ma, na, &L, bud, refusal);
    for (uint64_t i = 0; i < nb && !rc; i++) {                        /* w_b = D / m_b */
        x = big_u64(mb[i], bud, refusal, &rc);
        if (x) w[i] = tr_bigint_divexact(D, x, bud, refusal, &rc);
        tr_bigint_free(x); x = NULL;
    }
    for (uint64_t i = 0; i < na && !rc; i++) if (!(R[i] = big_u64(0, bud, refusal, &rc)) && !rc) rc = TR_MATH_MEMORY;
    for (uint64_t e = e0; e < e1 && !rc; e++) {                       /* R_a += O^2 w_b */
        uint64_t ia = s->cia[e] - ga, ib = s->cib[e] - gb;
        x = big_u128((unsigned __int128)s->co[e] * s->co[e], bud, refusal, &rc);
        if (x) rc = big_step(&x, w[ib], 1, bud, refusal);
        if (!rc) rc = big_step(&R[ia], x, 0, bud, refusal);
        tr_bigint_free(x); x = NULL;
    }
    if (!rc) T = big_u64(0, bud, refusal, &rc);
    for (uint64_t i = 0; i < na && !rc; i++) {                        /* T += R_a (L / m_a) */
        tr_bigint *ua = NULL;
        x = big_u64(ma[i], bud, refusal, &rc);
        if (x) ua = tr_bigint_divexact(L, x, bud, refusal, &rc);
        tr_bigint_free(x); x = NULL;
        if (ua) rc = big_step(&ua, R[i], 1, bud, refusal);
        if (ua && !rc) rc = big_step(&T, ua, 0, bud, refusal);
        tr_bigint_free(ua);
    }
    if (!rc) DL = tr_bigint_mul(D, L, bud, refusal, &rc);
    if (!rc) diff = tr_bigint_sub(T, DL, bud, refusal, &rc);           /* T - D L >= 0 */
    if (!rc) {
        if (op == TR_INFO_OP_CHI_SQUARE) {
            x = big_u64(N, bud, refusal, &rc);
            if (x) nu = tr_bigint_mul(diff, x, bud, refusal, &rc);
            if (!rc) de = tr_bigint_copy(DL, bud, refusal, &rc);
        } else {
            x = big_u64(q - 1, bud, refusal, &rc);
            if (x) de = tr_bigint_mul(DL, x, bud, refusal, &rc);
            if (!rc) nu = tr_bigint_copy(diff, bud, refusal, &rc);
        }
        tr_bigint_free(x); x = NULL;
    }
    if (!rc) Q = tr_bigq_from_bigints(nu, de, bud, refusal, &rc);
    if (!rc) {
        *num = tr_bigint_copy(tr_bigq_numerator(Q), bud, refusal, &rc);
        if (!rc) *den = tr_bigint_copy(tr_bigq_denominator(Q), bud, refusal, &rc);
        if (rc) { tr_bigint_free(*num); tr_bigint_free(*den); *num = *den = NULL; }
    }
    tr_bigq_free(Q);
    tr_bigint_free(D); tr_bigint_free(L); tr_bigint_free(T); tr_bigint_free(DL); tr_bigint_free(diff);
    tr_bigint_free(nu); tr_bigint_free(de);
    if (w) tr_bigint_free_array(w, nb);
    if (R) tr_bigint_free_array(R, na);
    free(w); free(R);
    return rc;
}
