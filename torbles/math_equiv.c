/* META-001: scoped query equivalence, implication and counterexamples (the contract is math_equiv.h).
 * SPDX-License-Identifier: Apache-2.0 */
#include <stdlib.h>
#include <string.h>
#include "math_equiv.h"
#include "math_handles.h"
#include "relationships.h"
#include "mathematics.h"

#define SET_MAGIC 0x45515331u
#define CHECK_MAGIC 0x45514331u
#define RESULT_MAGIC 0x45515231u
#define MAXV TR_GEOM_MAX_VARS
#define NCOLS TR_SCOPE_MAX_COLUMNS
#define MAXPAT 20                           /* nullable columns of one universal question (2^20 NULL patterns) */

_Static_assert(sizeof(tr_equiv_pred) == 32, "tr_equiv_pred");
_Static_assert(sizeof(tr_equiv_query) == 8, "tr_equiv_query");
_Static_assert(sizeof(tr_equiv_question) == 24, "tr_equiv_question");
_Static_assert(sizeof(tr_equiv_evidence) == 144, "tr_equiv_evidence");
_Static_assert(sizeof(tr_equiv_plan) == 56, "tr_equiv_plan");
_Static_assert(sizeof(tr_equiv_entry) == 16, "tr_equiv_entry");
_Static_assert(sizeof(tr_equiv_check_view) == 2272, "tr_equiv_check_view");
_Static_assert(sizeof(tr_equiv_column) == 48, "tr_equiv_column");
_Static_assert(sizeof(tr_equiv_domain) == 16, "tr_equiv_domain");
_Static_assert(sizeof(tr_equiv_verdict) == 80, "tr_equiv_verdict");
_Static_assert(sizeof(tr_equiv_decision) == 112, "tr_equiv_decision");
_Static_assert(sizeof(tr_equiv_set_info) == 112, "tr_equiv_set_info");
_Static_assert(sizeof(tr_equiv_answer) == 104, "tr_equiv_answer");
_Static_assert(sizeof(tr_equiv_run_stats) == 72, "tr_equiv_run_stats");
_Static_assert(sizeof(tr_equiv_rewrite_info) == 80, "tr_equiv_rewrite_info");
TR_MATH_ABI(tr_equiv_pred, 1);
TR_MATH_ABI(tr_equiv_query, 1);
TR_MATH_ABI(tr_equiv_question, 1);
TR_MATH_ABI(tr_equiv_evidence, 1);
TR_MATH_ABI(tr_equiv_plan, 1);
TR_MATH_ABI(tr_equiv_entry, 1);
TR_MATH_ABI(tr_equiv_check_view, 1);
TR_MATH_ABI(tr_equiv_column, 1);
TR_MATH_ABI(tr_equiv_domain, 1);
TR_MATH_ABI(tr_equiv_verdict, 1);
TR_MATH_ABI(tr_equiv_decision, 1);
TR_MATH_ABI(tr_equiv_set_info, 1);
TR_MATH_ABI(tr_equiv_answer, 1);
TR_MATH_ABI(tr_equiv_run_stats, 1);
TR_MATH_ABI(tr_equiv_rewrite_info, 1);

static void set_error(int32_t *e, int32_t rc) { if (e) *e = rc; }
static void *zalloc(uint64_t bytes) {
    if (bytes > (uint64_t)SIZE_MAX - 16) return NULL;
    return calloc(1, (size_t)(bytes ? bytes : 1));
}
static const tr_math_budget *budget_or(const tr_math_budget *b, tr_math_budget *store) {
    if (b) return b;
    tr_budget_default(store);
    return store;
}
static void need_zero(tr_math_need *n) { for (int i = 0; i < TR_BUDGET_RESOURCES; i++) n->need[i] = 0; }
static uint64_t mix(uint64_t h, uint64_t x) {
    h ^= x + UINT64_C(0x9E3779B97F4A7C15) + (h << 6) + (h >> 2);
    h ^= h >> 31; h *= UINT64_C(0xBF58476D1CE4E5B9); h ^= h >> 29;
    return h;
}
static uint64_t pow2_at_least(uint64_t n) { uint64_t c = 16; while (c < n) c <<= 1; return c; }

/* ================================================================ canonical measures */
typedef struct {
    int32_t ncols, degree;
    int32_t col[MAXV];                 /* increasing scope columns, each read by a term */
    uint64_t nterms;
    uint8_t *e;                        /* nterms x ncols, rows strictly increasing (lexicographic) */
    int64_t *num, *den;                /* reduced, den >= 1, num != 0 */
    int64_t D;                         /* lcm of the denominators */
    uint32_t cols;                     /* scope column mask */
    uint64_t hash;
} emeasure;
typedef struct { uint8_t e[MAXV]; tr_num_rational c; } eterm;
static int eterm_cmp(const void *a, const void *b) { return memcmp(((const eterm *)a)->e, ((const eterm *)b)->e, MAXV); }
static void measure_free(emeasure *m) { free(m->e); free(m->num); free(m->den); memset(m, 0, sizeof *m); }
static int64_t lcm64(int64_t a, int64_t b) {
    if (!a || !b) return 0;
    uint64_t g = tr_gcd_u64((uint64_t)a, (uint64_t)b);
    __int128 r = (__int128)(a / (int64_t)g) * b;
    return r > INT64_MAX ? 0 : (int64_t)r;
}
static int32_t poly_check(const tr_geom_poly *f) {
    if (!f || f->ncols < 0 || f->reserved || (f->nterms && (!f->exponents || !f->num))) return TR_MATH_INVALID;
    if (f->ncols > MAXV) return TR_MATH_UNSUPPORTED;
    for (int32_t i = 0; i < f->ncols; i++) {
        if (f->col[i] < 0 || f->col[i] >= NCOLS) return TR_MATH_INVALID;
        for (int32_t j = 0; j < i; j++) if (f->col[j] == f->col[i]) return TR_MATH_INVALID;
    }
    return TR_MATH_OK;
}
/* the canonical form of f (scratch: nterms eterms) */
static int32_t measure_canon(const tr_geom_poly *f, eterm *t, emeasure *m) {
    memset(m, 0, sizeof *m);
    int32_t rc = poly_check(f);
    if (rc) return rc;
    int32_t nc = f->ncols, perm[MAXV];
    for (int32_t i = 0; i < nc; i++) perm[i] = i;
    for (int32_t i = 1; i < nc; i++)
        for (int32_t j = i; j > 0 && f->col[perm[j - 1]] > f->col[perm[j]]; j--) { int32_t s = perm[j]; perm[j] = perm[j - 1]; perm[j - 1] = s; }
    uint64_t n = f->nterms;
    for (uint64_t k = 0; k < n; k++) {
        memset(t[k].e, 0, MAXV);
        for (int32_t i = 0; i < nc; i++) t[k].e[i] = f->exponents[k * (uint64_t)nc + (uint64_t)perm[i]];
        int64_t den = f->den ? f->den[k] : 1;
        if (den <= 0) return TR_MATH_INVALID;
        if (!tr_q_make(f->num[k], den, &t[k].c)) return TR_MATH_OVERFLOW;
    }
    if (n > 1) qsort(t, (size_t)n, sizeof *t, eterm_cmp);
    uint64_t m2 = 0;
    for (uint64_t k = 0; k < n; k++) {
        if (m2 && !memcmp(t[m2 - 1].e, t[k].e, MAXV)) {
            tr_num_rational s;
            if (!tr_q_add(t[m2 - 1].c, t[k].c, &s)) return TR_MATH_OVERFLOW;
            t[m2 - 1].c = s;
        } else t[m2++] = t[k];
        if (m2 && t[m2 - 1].c.numerator == 0 && (k + 1 == n || memcmp(t[k + 1].e, t[m2 - 1].e, MAXV))) m2--;
    }
    int used[MAXV] = {0};
    for (uint64_t k = 0; k < m2; k++) for (int32_t i = 0; i < nc; i++) used[i] |= t[k].e[i] != 0;
    int32_t map[MAXV], q = 0;
    for (int32_t i = 0; i < nc; i++) if (used[i]) { map[q] = i; m->col[q] = f->col[perm[i]]; m->cols |= UINT32_C(1) << m->col[q]; q++; }
    m->ncols = q; m->nterms = m2;
    m->e = (uint8_t *)zalloc(m2 * (uint64_t)(q ? q : 1));
    m->num = (int64_t *)zalloc(m2 * 8); m->den = (int64_t *)zalloc(m2 * 8);
    if (!m->e || !m->num || !m->den) { measure_free(m); return TR_MATH_MEMORY; }
    int64_t D = 1;
    uint64_t h = mix(0x4D4541, (uint64_t)q);
    for (int32_t i = 0; i < q; i++) h = mix(h, (uint64_t)m->col[i]);
    for (uint64_t k = 0; k < m2; k++) {
        int32_t deg = 0;
        for (int32_t i = 0; i < q; i++) { m->e[k * (uint64_t)q + (uint64_t)i] = t[k].e[map[i]]; deg += t[k].e[map[i]]; h = mix(h, t[k].e[map[i]]); }
        if (deg > m->degree) m->degree = deg;
        m->num[k] = t[k].c.numerator; m->den[k] = t[k].c.denominator;
        h = mix(mix(h, (uint64_t)m->num[k]), (uint64_t)m->den[k]);
        D = lcm64(D, m->den[k]);
        if (!D) { measure_free(m); return TR_MATH_OVERFLOW; }
    }
    m->D = D; m->hash = h;
    return TR_MATH_OK;
}
static int measure_equal(const emeasure *a, const emeasure *b) {
    if (a->hash != b->hash || a->ncols != b->ncols || a->nterms != b->nterms) return 0;
    if (memcmp(a->col, b->col, sizeof a->col)) return 0;
    if (a->nterms && (memcmp(a->e, b->e, (size_t)(a->nterms * (uint64_t)a->ncols)) ||
                      memcmp(a->num, b->num, (size_t)(a->nterms * 8)) || memcmp(a->den, b->den, (size_t)(a->nterms * 8))))
        return 0;
    return 1;
}
static void measure_view(const emeasure *m, tr_geom_poly *p) {
    memset(p, 0, sizeof *p);
    p->ncols = m->ncols;
    for (int32_t i = 0; i < MAXV; i++) p->col[i] = i < m->ncols ? m->col[i] : 0;
    p->nterms = m->nterms; p->exponents = m->e; p->num = m->num; p->den = m->den;
}
/* the exponent of scope column c in term k (0 when the measure does not read it) */
static uint8_t measure_exp(const emeasure *m, uint64_t k, int32_t c) {
    for (int32_t i = 0; i < m->ncols; i++) if (m->col[i] == c) return m->e[k * (uint64_t)m->ncols + (uint64_t)i];
    return 0;
}

/* ================================================================ sets */
struct tr_equiv_set_s {
    uint32_t magic;
    int32_t ncond, nnodes, nmeasure, nquery;   /* distinct */
    int32_t gpred, gmeasure, gquery;           /* given */
    int32_t maxcol;                            /* 1 + the largest scope column any part reads (0: none) */
    tr_bool_condition *conds;
    tr_bool_node *nodes;
    uint64_t *deps;                            /* conditions read at or below each node */
    uint32_t *cols;                            /* scope columns read at or below each node */
    int32_t *proot;                            /* given predicate -> node */
    emeasure *ms;
    int32_t *mid;                              /* given measure -> distinct */
    tr_equiv_query *qs;                        /* distinct (filter node, distinct measure) */
    int32_t *qid;                              /* given query -> distinct */
    tr_equiv_set_info info;
};
static int set_valid(const tr_equiv_set *s) { return s && s->magic == SET_MAGIC; }
void tr_equiv_set_free(tr_equiv_set *s) {
    if (!set_valid(s)) return;
    for (int32_t i = 0; i < s->nmeasure; i++) measure_free(&s->ms[i]);
    free(s->conds); free(s->nodes); free(s->deps); free(s->cols); free(s->proot); free(s->ms); free(s->mid);
    free(s->qs); free(s->qid);
    s->magic = 0;
    free(s);
}
static int32_t cond_static_check(const tr_bool_condition *c) {
    tr_scope_view v;
    v.ncols = NCOLS;
    return tr_bool_condition_check(&v, c) ? TR_MATH_INVALID : TR_MATH_OK;
}
static void cond_canon(const tr_bool_condition *in, tr_bool_condition *out) {
    *out = *in;
    if ((in->op >= TR_BOOL_EQ && in->op <= TR_BOOL_GE) || in->op == TR_BOOL_BETWEEN) {
        for (int k = 0; k < (in->op == TR_BOOL_BETWEEN ? 2 : 1); k++) {
            tr_num_rational q;
            if (tr_q_make(in->num[k], in->den[k], &q)) { out->num[k] = q.numerator; out->den[k] = q.denominator; }
        }
    }
}
static uint32_t cond_cols(const tr_bool_condition *c) {
    uint32_t m = UINT32_C(1) << c->col;
    if (c->op >= TR_BOOL_COL_EQ) m |= UINT32_C(1) << c->col2;
    return m;
}
typedef struct { int32_t *slot; uint64_t mask; } htab;
static int htab_make(htab *h, uint64_t n) {
    uint64_t cap = pow2_at_least(2 * n + 2);
    h->slot = (int32_t *)malloc((size_t)(cap * 4));
    if (!h->slot) return 0;
    memset(h->slot, 0xFF, (size_t)(cap * 4));
    h->mask = cap - 1;
    return 1;
}
static uint64_t node_hash(const tr_bool_node *n) { return mix(mix(mix(0x4E4F44, (uint64_t)n->op), (uint64_t)(uint32_t)n->a), (uint64_t)(uint32_t)n->b); }
/* the node id of n in the set's DAG (added when new) */
static int32_t intern_node(tr_equiv_set *s, htab *h, const tr_bool_node *n) {
    uint64_t i = node_hash(n) & h->mask;
    for (;; i = (i + 1) & h->mask) {
        int32_t id = h->slot[i];
        if (id < 0) break;
        const tr_bool_node *m = &s->nodes[id];
        if (m->op == n->op && m->a == n->a && m->b == n->b) return id;
    }
    int32_t id = s->nnodes++;
    s->nodes[id] = *n;
    h->slot[i] = id;
    uint64_t d = 0;
    uint32_t c = 0;
    if (n->op == TR_BOOL_VAR) { d = (uint64_t)1 << n->a; c = cond_cols(&s->conds[n->a]); }
    else if (n->op == TR_BOOL_NOT) { d = s->deps[n->a]; c = s->cols[n->a]; }
    else if (n->op != TR_BOOL_CONST) { d = s->deps[n->a] | s->deps[n->b]; c = s->cols[n->a] | s->cols[n->b]; }
    s->deps[id] = d; s->cols[id] = c;
    return id;
}
static int32_t pred_check(const tr_equiv_pred *p) {
    if (p->nconds < 0 || p->nnodes < 1 || p->root < 0 || p->root >= p->nnodes || p->reserved || !p->nodes ||
        (p->nconds && !p->conds)) return TR_MATH_INVALID;
    for (int32_t i = 0; i < p->nconds; i++) if (cond_static_check(&p->conds[i])) return TR_MATH_INVALID;
    for (int32_t i = 0; i < p->nnodes; i++) {
        const tr_bool_node *n = &p->nodes[i];
        if (n->reserved) return TR_MATH_INVALID;
        switch (n->op) {
        case TR_BOOL_CONST: if (n->a < 0 || n->a > 2 || n->b != -1) return TR_MATH_INVALID; break;
        case TR_BOOL_VAR: if (n->a < 0 || n->a >= p->nconds || n->b != -1) return TR_MATH_INVALID; break;
        case TR_BOOL_NOT: if (n->a < 0 || n->a >= i || n->b != -1) return TR_MATH_INVALID; break;
        case TR_BOOL_AND: case TR_BOOL_OR: case TR_BOOL_XOR:
            if (n->a < 0 || n->a >= i || n->b < 0 || n->b >= i) return TR_MATH_INVALID;
            break;
        default: return TR_MATH_INVALID;
        }
    }
    return TR_MATH_OK;
}
tr_equiv_set *tr_equiv_set_make(int32_t npred, const tr_equiv_pred *preds, int32_t nmeasure, const tr_geom_poly *measures,
                                int32_t nquery, const tr_equiv_query *queries, const tr_math_budget *budget,
                                tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    if (npred < 0 || nmeasure < 0 || nquery < 0 || (npred && !preds) || (nmeasure && !measures) || (nquery && !queries)) {
        set_error(error, TR_MATH_INVALID); return NULL;
    }
    if (npred > TR_EQUIV_MAX_PARTS || nmeasure > TR_EQUIV_MAX_PARTS || nquery > TR_EQUIV_MAX_PARTS) {
        set_error(error, TR_MATH_UNSUPPORTED); return NULL;
    }
    uint64_t gconds = 0, gnodes = 0, gterms = 0, maxnodes = 0, maxconds = 0, maxterms = 0;
    for (int32_t p = 0; p < npred; p++) {
        int32_t rc = pred_check(&preds[p]);
        if (rc) { set_error(error, rc); return NULL; }
        gconds += (uint64_t)preds[p].nconds; gnodes += (uint64_t)preds[p].nnodes;
        if ((uint64_t)preds[p].nnodes > maxnodes) maxnodes = (uint64_t)preds[p].nnodes;
        if ((uint64_t)preds[p].nconds > maxconds) maxconds = (uint64_t)preds[p].nconds;
    }
    for (int32_t m = 0; m < nmeasure; m++) {
        int32_t rc = poly_check(&measures[m]);
        if (rc) { set_error(error, rc); return NULL; }
        gterms = tr_budget_add(gterms, measures[m].nterms);
        if (measures[m].nterms > maxterms) maxterms = measures[m].nterms;
    }
    for (int32_t q = 0; q < nquery; q++)
        if (queries[q].filter < -1 || queries[q].filter >= npred || queries[q].measure < -1 || queries[q].measure >= nmeasure) {
            set_error(error, TR_MATH_INVALID); return NULL;
        }
    tr_math_need need;
    need_zero(&need);
    need.need[TR_BUDGET_CIRCUIT] = gnodes;
    need.need[TR_BUDGET_SUPPORT] = gterms;
    need.need[TR_BUDGET_OUTPUT] = (uint64_t)npred + (uint64_t)nmeasure + (uint64_t)nquery;
    uint64_t bytes = sizeof(tr_equiv_set);
    bytes = tr_budget_add(bytes, tr_budget_mul(gconds < TR_EQUIV_MAX_CONDITIONS ? gconds : TR_EQUIV_MAX_CONDITIONS, sizeof(tr_bool_condition)));
    bytes = tr_budget_add(bytes, tr_budget_mul(gnodes, sizeof(tr_bool_node) + 8 + 4 + 8 + 8));   /* nodes, deps, cols, hash */
    bytes = tr_budget_add(bytes, tr_budget_mul(maxnodes, 5));                                   /* per-predicate scratch */
    bytes = tr_budget_add(bytes, tr_budget_mul(maxconds, 4));
    bytes = tr_budget_add(bytes, tr_budget_mul((uint64_t)npred, 4));
    bytes = tr_budget_add(bytes, tr_budget_mul(gterms, MAXV + 16));                             /* canonical terms */
    bytes = tr_budget_add(bytes, tr_budget_mul(maxterms, sizeof(eterm)));                       /* canonicalization */
    bytes = tr_budget_add(bytes, tr_budget_mul((uint64_t)nmeasure, sizeof(emeasure) + 4 + 8));
    bytes = tr_budget_add(bytes, tr_budget_mul((uint64_t)nquery, sizeof(tr_equiv_query) + 4 + 8));
    need.need[TR_BUDGET_BYTES] = bytes;
    tr_math_budget store;
    const tr_math_budget *B = budget_or(budget, &store);
    int32_t rc = tr_budget_check(B, &need, refusal);
    if (rc) { set_error(error, rc); return NULL; }
    if (gnodes > TR_BOOL_MAX_NODES) { set_error(error, TR_MATH_UNSUPPORTED); return NULL; }
    tr_equiv_set *s = (tr_equiv_set *)zalloc(sizeof *s);
    if (!s) { set_error(error, TR_MATH_MEMORY); return NULL; }
    s->magic = SET_MAGIC; s->gpred = npred; s->gmeasure = nmeasure; s->gquery = nquery;
    s->conds = (tr_bool_condition *)zalloc(TR_EQUIV_MAX_CONDITIONS * sizeof(tr_bool_condition));
    s->nodes = (tr_bool_node *)zalloc(gnodes * sizeof(tr_bool_node));
    s->deps = (uint64_t *)zalloc(gnodes * 8); s->cols = (uint32_t *)zalloc(gnodes * 4);
    s->proot = (int32_t *)zalloc((uint64_t)npred * 4);
    s->ms = (emeasure *)zalloc((uint64_t)nmeasure * sizeof(emeasure)); s->mid = (int32_t *)zalloc((uint64_t)nmeasure * 4);
    s->qs = (tr_equiv_query *)zalloc((uint64_t)nquery * sizeof(tr_equiv_query)); s->qid = (int32_t *)zalloc((uint64_t)nquery * 4);
    int32_t *g = (int32_t *)zalloc(maxnodes * 4), *cid = (int32_t *)zalloc(maxconds * 4);
    uint8_t *mark = (uint8_t *)zalloc(maxnodes);
    eterm *tt = (eterm *)zalloc(maxterms * sizeof(eterm));
    htab h = {NULL, 0};
    rc = TR_MATH_OK;
    if (!s->conds || !s->nodes || !s->deps || !s->cols || !s->proot || !s->ms || !s->mid || !s->qs || !s->qid || !g ||
        !cid || !mark || !tt || !htab_make(&h, gnodes)) rc = TR_MATH_MEMORY;
    uint64_t shared_scratch = 0;
    for (int32_t p = 0; p < npred && !rc; p++) {
        const tr_equiv_pred *P = &preds[p];
        memset(mark, 0, (size_t)P->nnodes);
        mark[P->root] = 1;
        for (int32_t i = P->root; i >= 0; i--) {
            if (!mark[i]) continue;
            const tr_bool_node *n = &P->nodes[i];
            if (n->op == TR_BOOL_NOT) mark[n->a] = 1;
            else if (n->op >= TR_BOOL_AND) { mark[n->a] = 1; mark[n->b] = 1; }
        }
        for (int32_t c = 0; c < P->nconds; c++) cid[c] = -1;
        for (int32_t i = 0; i <= P->root && !rc; i++) {
            if (!mark[i]) continue;
            tr_bool_node n = P->nodes[i];
            if (n.op == TR_BOOL_VAR) {
                if (cid[n.a] < 0) {
                    tr_bool_condition c;
                    cond_canon(&P->conds[n.a], &c);
                    int32_t at = -1;
                    for (int32_t j = 0; j < s->ncond && at < 0; j++) if (!memcmp(&s->conds[j], &c, sizeof c)) at = j;
                    if (at < 0) {                     /* variables: the distinct conditions, checked as reached */
                        if ((uint64_t)s->ncond + 1 > B->limit[TR_BUDGET_VARIABLES]) {
                            rc = tr_budget_check_one(B, TR_BUDGET_VARIABLES, (uint64_t)s->ncond + 1, refusal);
                            break;
                        }
                        if (s->ncond >= TR_EQUIV_MAX_CONDITIONS) { rc = TR_MATH_UNSUPPORTED; break; }
                        at = s->ncond++;
                        s->conds[at] = c;
                        int32_t top = c.col > (c.op >= TR_BOOL_COL_EQ ? c.col2 : -1) ? c.col : c.col2;
                        if (top + 1 > s->maxcol) s->maxcol = top + 1;
                    }
                    cid[n.a] = at;
                }
                n.a = cid[n.a];
            } else if (n.op == TR_BOOL_NOT) n.a = g[n.a];
            else if (n.op >= TR_BOOL_AND) {
                int32_t x = g[n.a], y = g[n.b];
                n.a = x < y ? x : y; n.b = x < y ? y : x;
            }
            g[i] = intern_node(s, &h, &n);
        }
        if (!rc) s->proot[p] = g[P->root];
    }
    (void)shared_scratch;
    for (int32_t m = 0; m < nmeasure && !rc; m++) {
        emeasure e;
        rc = measure_canon(&measures[m], tt, &e);
        if (rc) break;
        int32_t at = -1;
        for (int32_t j = 0; j < s->nmeasure && at < 0; j++) if (measure_equal(&s->ms[j], &e)) at = j;
        if (at >= 0) measure_free(&e);
        else {
            at = s->nmeasure++;
            s->ms[at] = e;
            if (e.ncols && e.col[e.ncols - 1] + 1 > s->maxcol) s->maxcol = e.col[e.ncols - 1] + 1;
        }
        s->mid[m] = at;
    }
    for (int32_t q = 0; q < nquery && !rc; q++) {
        tr_equiv_query k;
        k.filter = queries[q].filter >= 0 ? s->proot[queries[q].filter] : -1;
        k.measure = queries[q].measure >= 0 ? s->mid[queries[q].measure] : -1;
        int32_t at = -1;
        for (int32_t j = 0; j < s->nquery && at < 0; j++) if (s->qs[j].filter == k.filter && s->qs[j].measure == k.measure) at = j;
        if (at < 0) { at = s->nquery++; s->qs[at] = k; }
        s->qid[q] = at;
    }
    free(g); free(cid); free(mark); free(tt); free(h.slot);
    if (rc) { tr_equiv_set_free(s); set_error(error, rc); return NULL; }
    /* the statistics: distinct parts, the nodes read by more than one distinct predicate */
    tr_equiv_set_info *in = &s->info;
    memset(in, 0, sizeof *in);
    int32_t *stamp = (int32_t *)zalloc((uint64_t)s->nnodes * 4), *count = (int32_t *)zalloc((uint64_t)s->nnodes * 4);
    int32_t *stack = (int32_t *)zalloc((uint64_t)s->nnodes * 4), *roots = (int32_t *)zalloc((uint64_t)npred * 4);
    if (!stamp || !count || !stack || !roots) {
        free(stamp); free(count); free(stack); free(roots); tr_equiv_set_free(s); set_error(error, TR_MATH_MEMORY); return NULL;
    }
    int32_t nroots = 0;
    for (int32_t p = 0; p < npred; p++) {
        int seen = 0;
        for (int32_t j = 0; j < nroots && !seen; j++) seen = roots[j] == s->proot[p];
        if (!seen) roots[nroots++] = s->proot[p];
    }
    for (int32_t r = 0; r < nroots; r++) {
        int32_t sp = 0;
        stack[sp++] = roots[r]; stamp[roots[r]] = r + 1;
        while (sp) {
            int32_t i = stack[--sp];
            count[i]++;
            const tr_bool_node *n = &s->nodes[i];
            int32_t ops[2] = {n->op == TR_BOOL_NOT || n->op >= TR_BOOL_AND ? n->a : -1, n->op >= TR_BOOL_AND ? n->b : -1};
            for (int j = 0; j < 2; j++) if (ops[j] >= 0 && stamp[ops[j]] != r + 1) { stamp[ops[j]] = r + 1; stack[sp++] = ops[j]; }
        }
    }
    for (int32_t i = 0; i < s->nnodes; i++) in->shared_nodes += count[i] >= 2;
    uint32_t cols = 0;
    for (int32_t i = 0; i < s->nnodes; i++) cols |= s->cols[i];
    for (int32_t m = 0; m < s->nmeasure; m++) cols |= s->ms[m].cols;
    free(stamp); free(count); free(stack); free(roots);
    in->npred = nroots; in->nmeasure = s->nmeasure; in->nquery = s->nquery; in->ncond = s->ncond;
    in->nnodes = s->nnodes; in->ncols = tr_bool_popc(cols);
    in->given_preds = (uint64_t)npred; in->given_conds = gconds; in->given_nodes = gnodes;
    in->given_measures = (uint64_t)nmeasure; in->given_queries = (uint64_t)nquery;
    uint64_t fl = 0, ml = 0;
    for (int32_t j = 0; j < s->nquery; j++) {
        int fs = 0, mm = 0;
        for (int32_t i = 0; i < j; i++) { fs |= s->qs[i].filter == s->qs[j].filter; mm |= s->qs[i].measure == s->qs[j].measure; }
        fl += !fs && s->qs[j].filter >= 0; ml += !mm && s->qs[j].measure >= 0;
    }
    in->filters = fl; in->measures = ml; in->pairs = (uint64_t)s->nquery;
    uint64_t own = sizeof *s + TR_EQUIV_MAX_CONDITIONS * sizeof(tr_bool_condition) + gnodes * (sizeof(tr_bool_node) + 12) +
                   (uint64_t)npred * 4 + (uint64_t)nmeasure * (sizeof(emeasure) + 4) + (uint64_t)nquery * (sizeof(tr_equiv_query) + 4);
    for (int32_t m = 0; m < s->nmeasure; m++) own += s->ms[m].nterms * ((uint64_t)s->ms[m].ncols + 16);
    in->bytes = own;
    set_error(error, TR_MATH_OK);
    return s;
}
int32_t tr_equiv_set_get_info(const tr_equiv_set *s, tr_equiv_set_info *out) {
    if (!set_valid(s) || !out) return TR_MATH_INVALID;
    *out = s->info;
    return TR_MATH_OK;
}
int32_t tr_equiv_set_ids(const tr_equiv_set *s, int32_t *pred_roots, int32_t *measure_ids, int32_t *query_ids,
                         int32_t *query_filters, int32_t *query_measures) {
    if (!set_valid(s)) return TR_MATH_INVALID;
    for (int32_t i = 0; pred_roots && i < s->gpred; i++) pred_roots[i] = s->proot[i];
    for (int32_t i = 0; measure_ids && i < s->gmeasure; i++) measure_ids[i] = s->mid[i];
    for (int32_t i = 0; i < s->gquery; i++) {
        if (query_ids) query_ids[i] = s->qid[i];
        if (query_filters) query_filters[i] = s->qs[s->qid[i]].filter;
        if (query_measures) query_measures[i] = s->qs[s->qid[i]].measure;
    }
    return TR_MATH_OK;
}
int32_t tr_equiv_set_conditions(const tr_equiv_set *s, tr_bool_condition *out, int32_t capacity) {
    if (!set_valid(s) || capacity < s->ncond || (s->ncond && !out)) return TR_MATH_INVALID;
    if (s->ncond) memcpy(out, s->conds, (size_t)s->ncond * sizeof *out);
    return TR_MATH_OK;
}
int32_t tr_equiv_set_nodes(const tr_equiv_set *s, tr_bool_node *out, int32_t capacity) {
    if (!set_valid(s) || capacity < s->nnodes || (s->nnodes && !out)) return TR_MATH_INVALID;
    if (s->nnodes) memcpy(out, s->nodes, (size_t)s->nnodes * sizeof *out);
    return TR_MATH_OK;
}
int32_t tr_equiv_poly_from_math(const tr_math_poly *p, const int32_t *cols, int32_t *ncols, int32_t *out_cols,
                                uint8_t *exponents, int64_t *num, int64_t *den, uint64_t capacity, uint64_t *nterms) {
    if (!p || !cols || !ncols || !out_cols || !nterms) return TR_MATH_INVALID;
    int32_t nt = tr_math_poly_term_count(p);
    if (nt < 0) return TR_MATH_INVALID;
    if ((uint64_t)nt > capacity || (nt && (!exponents || !num || !den))) return TR_MATH_INVALID;
    int32_t support = tr_math_poly_support(p);
    if (support < 0) return TR_MATH_INVALID;
    int32_t nv = 0, pos[TR_MATH_VARIABLES];
    for (int32_t v = 0; v < TR_MATH_VARIABLES; v++) {
        pos[v] = -1;
        if (!((support >> v) & 1)) continue;
        if (cols[v] < 0 || cols[v] >= NCOLS) return TR_MATH_INVALID;
        for (int32_t j = 0; j < nv; j++) if (out_cols[j] == cols[v]) return TR_MATH_INVALID;
        pos[v] = nv; out_cols[nv++] = cols[v];
    }
    for (int32_t t = 0; t < nt; t++) {
        tr_math_term term;
        if (tr_math_poly_get_term(p, t, &term)) return TR_MATH_INVALID;
        for (int32_t j = 0; j < nv; j++) exponents[(uint64_t)t * (uint64_t)nv + (uint64_t)j] = 0;
        for (int32_t v = 0; v < TR_MATH_VARIABLES; v++)
            if (pos[v] >= 0) exponents[(uint64_t)t * (uint64_t)nv + (uint64_t)pos[v]] = term.exponents[v];
        if (term.coefficient.denominator <= 0) return TR_MATH_INVALID;
        num[t] = term.coefficient.numerator; den[t] = term.coefficient.denominator;
    }
    *ncols = nv; *nterms = (uint64_t)nt;
    return TR_MATH_OK;
}

/* ================================================================ questions */
typedef struct { int32_t ra, rb, ma, mb; } sides;     /* filter roots (nodes, -1: none), distinct measures (-1) */
static int32_t question_check(const tr_equiv_set *S, const tr_equiv_question *q) {
    switch (q->kind) {
    case TR_EQUIV_PREDICATE:
        if (q->a < 0 || q->a >= S->gpred || q->b < 0 || q->b >= S->gpred || q->nulls) return TR_MATH_INVALID;
        if (q->relation == TR_EQUIV_EQUIVALENT) return q->semantics == TR_EQUIV_FILTER || q->semantics == TR_EQUIV_KLEENE ? 0 : TR_MATH_INVALID;
        if (q->relation == TR_EQUIV_IMPLIES) return q->semantics == TR_EQUIV_FILTER ? 0 : TR_MATH_INVALID;
        return TR_MATH_INVALID;
    case TR_EQUIV_MEASURE:
        if (q->a < 0 || q->a >= S->gmeasure || q->b < 0 || q->b >= S->gmeasure || q->semantics ||
            q->relation != TR_EQUIV_EQUIVALENT || (q->nulls != TR_EQUIV_SQL && q->nulls != TR_EQUIV_SKIP)) return TR_MATH_INVALID;
        return TR_MATH_OK;
    case TR_EQUIV_QUERY:
        if (q->a < 0 || q->a >= S->gquery || q->b < 0 || q->b >= S->gquery || q->semantics ||
            (q->relation != TR_EQUIV_EQUIVALENT && q->relation != TR_EQUIV_IMPLIES) ||
            (q->nulls != TR_EQUIV_SQL && q->nulls != TR_EQUIV_SKIP)) return TR_MATH_INVALID;
        if ((S->qs[S->qid[q->a]].measure < 0) != (S->qs[S->qid[q->b]].measure < 0)) return TR_MATH_INVALID;
        return TR_MATH_OK;
    default:
        return TR_MATH_INVALID;
    }
}
static void question_sides(const tr_equiv_set *S, const tr_equiv_question *q, sides *o) {
    o->ra = o->rb = o->ma = o->mb = -1;
    if (q->kind == TR_EQUIV_PREDICATE) { o->ra = S->proot[q->a]; o->rb = S->proot[q->b]; }
    else if (q->kind == TR_EQUIV_MEASURE) { o->ma = S->mid[q->a]; o->mb = S->mid[q->b]; }
    else {
        const tr_equiv_query *x = &S->qs[S->qid[q->a]], *y = &S->qs[S->qid[q->b]];
        o->ra = x->filter; o->rb = y->filter; o->ma = x->measure; o->mb = y->measure;
    }
}
static uint32_t sides_cols(const tr_equiv_set *S, const sides *d) {
    uint32_t c = 0;
    if (d->ra >= 0) c |= S->cols[d->ra];
    if (d->rb >= 0) c |= S->cols[d->rb];
    if (d->ma >= 0) c |= S->ms[d->ma].cols;
    if (d->mb >= 0) c |= S->ms[d->mb].cols;
    return c;
}
/* the nodes the two roots reach, increasing (at most cap); returns the count, or -1 past cap */
static int32_t cone_of(const tr_equiv_set *S, int32_t ra, int32_t rb, int32_t *stamp, int32_t tag, int32_t *out, int32_t cap) {
    int32_t n = 0, stack[TR_EQUIV_MAX_CONE + 2], sp = 0;
    int32_t roots[2] = {ra, rb};
    for (int r = 0; r < 2; r++) {
        if (roots[r] < 0 || stamp[roots[r]] == tag) continue;
        stamp[roots[r]] = tag;
        if (n >= cap) return -1;
        out[n++] = roots[r]; stack[sp++] = roots[r];
        while (sp) {
            const tr_bool_node *m = &S->nodes[stack[--sp]];
            int32_t ops[2] = {m->op == TR_BOOL_NOT || m->op >= TR_BOOL_AND ? m->a : -1, m->op >= TR_BOOL_AND ? m->b : -1};
            for (int j = 0; j < 2; j++) {
                if (ops[j] < 0 || stamp[ops[j]] == tag) continue;
                stamp[ops[j]] = tag;
                if (n >= cap) return -1;
                out[n++] = ops[j]; stack[sp++] = ops[j];
            }
        }
    }
    for (int32_t i = 1; i < n; i++) for (int32_t j = i; j > 0 && out[j - 1] > out[j]; j--) { int32_t t = out[j]; out[j] = out[j - 1]; out[j - 1] = t; }
    return n;
}

/* ================================================================ checks */
typedef struct {
    int32_t ma, mb;                 /* distinct measures */
    int32_t ncols, constant;        /* claim columns; 1: two different constants (no claim, no candidates) */
    int32_t col[MAXV];
    uint64_t nterms;                /* P = D (f - g), canonical over col */
    uint8_t *e;
    int64_t *c;
    int64_t D;
    int32_t geom, reserved;         /* index in the GEOM-001 batch (-1) */
} epair;
struct tr_equiv_check_s {
    uint32_t magic;
    int32_t nq, npairs, ngeom, ncond, reserved;
    const tr_scope *scope;
    tr_equiv_question *qs;
    tr_equiv_plan *plans;
    tr_equiv_entry *entries;
    uint64_t nentries;
    int32_t *qpair;                 /* question -> epair (-1) */
    uint32_t *qcols;                /* question -> the scope columns it reads */
    epair *pairs;
    tr_bool_masks *masks;
    tr_geom_pairs *geom;
    tr_cert_record *record;
    tr_equiv_check_view view;
    uint64_t bytes;
};
static int check_valid(const tr_equiv_check *k) { return k && k->magic == CHECK_MAGIC; }
void tr_equiv_check_free(tr_equiv_check *k) {
    if (!check_valid(k)) return;
    for (int32_t i = 0; i < k->npairs; i++) { free(k->pairs[i].e); free(k->pairs[i].c); }
    free(k->qs); free(k->plans); free(k->entries); free(k->qpair); free(k->qcols); free(k->pairs);
    tr_bool_masks_free(k->masks);
    tr_geom_pairs_free(k->geom);
    tr_cert_record_free(k->record);
    k->magic = 0;
    free(k);
}
/* P = D (f - g) over the union of the measures' columns, canonical (FND-007) */
static int32_t pair_claim(const emeasure *f, const emeasure *g, epair *p) {
    int32_t n = 0;
    for (int32_t i = 0; i < f->ncols; i++) p->col[n++] = f->col[i];
    for (int32_t i = 0; i < g->ncols; i++) {
        int seen = 0;
        for (int32_t j = 0; j < n; j++) seen |= p->col[j] == g->col[i];
        if (seen) continue;
        if (n >= MAXV) return TR_MATH_UNSUPPORTED;
        p->col[n++] = g->col[i];
    }
    for (int32_t i = 1; i < n; i++) for (int32_t j = i; j > 0 && p->col[j - 1] > p->col[j]; j--) { int32_t t = p->col[j]; p->col[j] = p->col[j - 1]; p->col[j - 1] = t; }
    for (int32_t i = n; i < MAXV; i++) p->col[i] = -1;
    p->ncols = n;
    int64_t D = lcm64(f->D, g->D);
    if (!D) return TR_MATH_OVERFLOW;
    p->D = D;
    uint64_t raw = f->nterms + g->nterms, w = (uint64_t)(n ? n : 1);
    uint8_t *re = (uint8_t *)zalloc(raw * w);
    int64_t *rc = (int64_t *)zalloc(raw * 8);
    p->e = (uint8_t *)zalloc(raw * w); p->c = (int64_t *)zalloc(raw * 8);
    if (!re || !rc || !p->e || !p->c) { free(re); free(rc); return TR_MATH_MEMORY; }
    for (int s = 0; s < 2; s++) {
        const emeasure *m = s ? g : f;
        for (uint64_t k = 0; k < m->nterms; k++) {
            uint64_t at = (s ? f->nterms : 0) + k;
            for (int32_t i = 0; i < n; i++) re[at * w + (uint64_t)i] = measure_exp(m, k, p->col[i]);
            __int128 v = (__int128)m->num[k] * (D / m->den[k]);
            if (s) v = -v;
            if (v < INT64_MIN || v > INT64_MAX) { free(re); free(rc); return TR_MATH_OVERFLOW; }
            rc[at] = (int64_t)v;
        }
    }
    int32_t err = TR_MATH_OK;
    uint64_t m = 0;
    if (n && raw) err = tr_cert_poly_canonical(n, raw, re, rc, 0, 0, p->e, p->c, &m);
    free(re); free(rc);
    p->nterms = m;
    return err;
}
/* the cand view of a check without pairs */
static void empty_cand(tr_ident_cand_view *c) {
    memset(c, 0, sizeof *c);
    for (int32_t i = 0; i < TR_IDENT_MAX_VARS; i++) c->var[i] = -1;
}
static tr_equiv_check *check_build(const tr_equiv_set *S, const tr_equiv_question *qs, int32_t nq, const tr_scope *scope,
                                   const uint64_t *value, const uint64_t *valid, int want_record,
                                   const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    tr_scope_view sv;
    if (!set_valid(S) || !scope || nq < 0 || (nq && !qs) || tr_scope_get_view(scope, &sv)) { set_error(error, TR_MATH_INVALID); return NULL; }
    if (nq > TR_EQUIV_MAX_QUESTIONS) { set_error(error, TR_MATH_UNSUPPORTED); return NULL; }
    if (S->maxcol > sv.ncols) { set_error(error, TR_MATH_INVALID); return NULL; }   /* a part reads a column the scope lacks */
    uint64_t pterms = 0;
    for (int32_t q = 0; q < nq; q++) {
        int32_t rc = question_check(S, &qs[q]);
        if (rc) { set_error(error, rc); return NULL; }
        sides d;
        question_sides(S, &qs[q], &d);
        if (d.ma >= 0 && d.mb >= 0) pterms = tr_budget_add(pterms, S->ms[d.ma].nterms + S->ms[d.mb].nterms);
    }
    tr_math_need need;
    need_zero(&need);
    need.need[TR_BUDGET_OUTPUT] = (uint64_t)nq;
    uint64_t bytes = sizeof(tr_equiv_check);
    bytes = tr_budget_add(bytes, tr_budget_mul((uint64_t)nq, sizeof(tr_equiv_question) + sizeof(tr_equiv_plan) + 8 + sizeof(epair)));
    bytes = tr_budget_add(bytes, tr_budget_mul(tr_budget_mul((uint64_t)nq, TR_EQUIV_MAX_CONE), sizeof(tr_equiv_entry)));
    bytes = tr_budget_add(bytes, tr_budget_mul((uint64_t)S->nnodes, 8));
    bytes = tr_budget_add(bytes, tr_budget_mul(pterms, 2 * (MAXV + 8)));
    bytes = tr_budget_add(bytes, tr_budget_mul((uint64_t)nq, 2 * sizeof(tr_geom_poly)));
    need.need[TR_BUDGET_BYTES] = bytes;
    tr_math_budget store;
    int32_t rc = tr_budget_check(budget_or(budget, &store), &need, refusal);
    if (rc) { set_error(error, rc); return NULL; }
    tr_equiv_check *K = (tr_equiv_check *)zalloc(sizeof *K);
    if (!K) { set_error(error, TR_MATH_MEMORY); return NULL; }
    K->magic = CHECK_MAGIC; K->nq = nq; K->scope = scope; K->ncond = S->ncond; K->bytes = bytes;
    K->qs = (tr_equiv_question *)zalloc((uint64_t)nq * sizeof *K->qs);
    K->plans = (tr_equiv_plan *)zalloc((uint64_t)nq * sizeof *K->plans);
    K->qpair = (int32_t *)zalloc((uint64_t)nq * 4); K->qcols = (uint32_t *)zalloc((uint64_t)nq * 4);
    K->pairs = (epair *)zalloc((uint64_t)nq * sizeof *K->pairs);
    K->entries = (tr_equiv_entry *)zalloc(tr_budget_mul((uint64_t)nq, TR_EQUIV_MAX_CONE) * sizeof *K->entries);
    int32_t *stamp = (int32_t *)zalloc((uint64_t)S->nnodes * 4), *pos = (int32_t *)zalloc((uint64_t)S->nnodes * 4);
    tr_geom_poly *gf = (tr_geom_poly *)zalloc((uint64_t)nq * sizeof *gf), *gg = (tr_geom_poly *)zalloc((uint64_t)nq * sizeof *gg);
    if (!K->qs || !K->plans || !K->qpair || !K->qcols || !K->pairs || !K->entries || !stamp || !pos || !gf || !gg) rc = TR_MATH_MEMORY;
    if (nq && K->qs) memcpy(K->qs, qs, (size_t)nq * sizeof *qs);
    uint64_t at = 0;
    for (int32_t q = 0; q < nq && !rc; q++) {
        sides d;
        question_sides(S, &qs[q], &d);
        tr_equiv_plan *p = &K->plans[q];
        memset(p, 0, sizeof *p);
        p->kind = qs[q].kind; p->relation = qs[q].relation; p->semantics = qs[q].semantics; p->nulls = qs[q].nulls;
        p->ra = p->rb = -1; p->pair = -1;
        K->qpair[q] = -1;
        K->qcols[q] = sides_cols(S, &d);
        int32_t cone[TR_EQUIV_MAX_CONE];
        int32_t n = cone_of(S, d.ra, d.rb, stamp, q + 1, cone, TR_EQUIV_MAX_CONE);
        if (n < 0) { rc = TR_MATH_UNSUPPORTED; break; }
        p->cone = (uint32_t)at; p->ncone = (uint32_t)n;
        for (int32_t i = 0; i < n; i++) pos[cone[i]] = i;
        for (int32_t i = 0; i < n; i++) {
            const tr_bool_node *m = &S->nodes[cone[i]];
            tr_equiv_entry *e = &K->entries[at + (uint64_t)i];
            e->op = m->op; e->reserved = 0; e->b = -1;
            if (m->op == TR_BOOL_CONST || m->op == TR_BOOL_VAR) e->a = m->a;
            else { e->a = pos[m->a]; if (m->op >= TR_BOOL_AND) e->b = pos[m->b]; }
        }
        if (d.ra >= 0) p->ra = pos[d.ra];
        if (d.rb >= 0) p->rb = pos[d.rb];
        at += (uint64_t)n;
        p->hasa = d.ma >= 0; p->hasb = d.mb >= 0;
        if (d.ma >= 0) p->supa = S->ms[d.ma].cols;
        if (d.mb >= 0) p->supb = S->ms[d.mb].cols;
        if (d.ma < 0 || d.mb < 0 || d.ma == d.mb) continue;   /* COUNT queries, or one measure: equal values */
        int32_t pi = -1;
        for (int32_t j = 0; j < K->npairs && pi < 0; j++)
            if ((K->pairs[j].ma == d.ma && K->pairs[j].mb == d.mb) || (K->pairs[j].ma == d.mb && K->pairs[j].mb == d.ma)) pi = j;
        if (pi < 0) {
            pi = K->npairs++;
            epair *E = &K->pairs[pi];
            E->ma = d.ma; E->mb = d.mb; E->geom = -1;
            const emeasure *f = &S->ms[d.ma], *g = &S->ms[d.mb];
            if (!f->ncols && !g->ncols) E->constant = 1;
            else {
                rc = pair_claim(f, g, E);
                if (rc) break;
                E->geom = K->ngeom++;
                measure_view(f, &gf[E->geom]); measure_view(g, &gg[E->geom]);
            }
        }
        K->qpair[q] = pi;
        p->pair = K->pairs[pi].geom;
        p->constant = K->pairs[pi].constant;
    }
    K->nentries = at;
    free(stamp); free(pos);
    if (!rc && S->ncond) {
        int32_t e = 0;
        K->masks = value ? tr_bool_masks_adopt(scope, S->conds, S->ncond, value, valid, budget, refusal, &e)
                         : tr_bool_masks_build(scope, S->conds, S->ncond, budget, refusal, &e);
        if (!K->masks) rc = e ? e : TR_MATH_MEMORY;
    }
    if (!rc && K->ngeom) {
        int32_t e = 0;
        K->geom = tr_geom_pairs_make(scope, gf, gg, K->ngeom, budget, refusal, &e);
        if (!K->geom) rc = e ? e : TR_MATH_MEMORY;
    }
    free(gf); free(gg);
    if (!rc && want_record) {
        int32_t e = 0;
        K->record = tr_cert_record_scope(scope, NULL, -1, TR_CERT_RECORD_SELECTION, NULL, 0, budget, refusal, &e);
        if (!K->record) rc = e ? e : TR_MATH_MEMORY;
    }
    if (rc) { tr_equiv_check_free(K); set_error(error, rc); return NULL; }
    tr_equiv_check_view *v = &K->view;
    memset(v, 0, sizeof *v);
    v->rows = sv;
    if (K->masks) tr_bool_masks_get_domain(K->masks, &v->masks);
    else {
        v->masks.kind = TR_BOOL_ROWS; v->masks.k = 0; v->masks.points = sv.rows; v->masks.nw = (sv.rows + 63) / 64;
        v->masks.selected = sv.selected; v->masks.selection = sv.selection;
    }
    if (K->geom) {
        tr_geom_pair_view pv;
        tr_geom_pairs_get_view(K->geom, &pv);
        v->cand = pv.cand;
    } else empty_cand(&v->cand);
    v->nq = nq; v->nentries = K->nentries; v->plans = K->plans; v->entries = K->entries;
    set_error(error, TR_MATH_OK);
    return K;
}
tr_equiv_check *tr_equiv_check_make(const tr_equiv_set *S, const tr_equiv_question *qs, int32_t nq, const tr_scope *scope,
                                    const uint64_t *value, const uint64_t *valid, const tr_math_budget *budget,
                                    tr_math_refusal *refusal, int32_t *error) {
    return check_build(S, qs, nq, scope, value, valid, 1, budget, refusal, error);
}
int32_t tr_equiv_check_get_view(const tr_equiv_check *k, tr_equiv_check_view *out) {
    if (!check_valid(k) || !out) return TR_MATH_INVALID;
    *out = k->view;
    return TR_MATH_OK;
}
int32_t tr_equiv_check_need(const tr_equiv_check *k, int32_t target, tr_math_need *need) {
    if (!check_valid(k) || !need) return TR_MATH_INVALID;
    if (target == TR_EQUIV_NEED_DEVICE) { tr_equiv_device_need(&k->view, need); return TR_MATH_OK; }
    if (target != TR_EQUIV_NEED_CPU) return TR_MATH_INVALID;
    need_zero(need);
    need->need[TR_BUDGET_VARIABLES] = (uint64_t)k->ncond;
    need->need[TR_BUDGET_SUPPORT] = k->view.cand.nterms;
    need->need[TR_BUDGET_OUTPUT] = (uint64_t)k->nq;
    need->need[TR_BUDGET_BYTES] = 2 * TR_EQUIV_MAX_CONE * 8 + sizeof(tr_equiv_acc);
    return TR_MATH_OK;
}
static void acc_out(const tr_equiv_check_view *v, const tr_equiv_acc *a, tr_equiv_evidence *e) {
    memset(e, 0, sizeof *e);
    e->rows = v->rows.rows; e->selected = v->rows.selected;
    e->checked = a->checked; e->excluded = a->excluded; e->differ = a->differ; e->undecided = a->undecided;
    e->witness = a->witness; e->null_mismatch = a->null_mismatch; e->nulls = a->nulls;
    for (int i = 0; i < 9; i++) e->cells[i] = a->cells[i];
}
static void run_cpu(const tr_equiv_check_view *v, tr_equiv_evidence *out) {
    uint64_t val[2 * TR_EQUIV_MAX_CONE];
    const uint64_t nw = (v->rows.rows + 63) / 64;
    for (int32_t q = 0; q < v->nq; q++) {
        tr_equiv_acc a;
        tr_equiv_acc_init(&a);
        for (uint64_t w = 0; w < nw; w++) tr_equiv_word(v, q, w, val, &a);
        acc_out(v, &a, &out[q]);
    }
}
int32_t tr_equiv_check_run(const tr_equiv_check *k, tr_equiv_evidence *out) {
    if (!check_valid(k) || (k->nq && !out)) return TR_MATH_INVALID;
    run_cpu(&k->view, out);
    return TR_MATH_OK;
}
/* a device result: the counts consistent with the check, and its witness re-evaluated on the CPU */
static int evidence_consistent(const tr_equiv_check *k, int32_t q, const tr_equiv_evidence *e) {
    const tr_equiv_check_view *v = &k->view;
    const tr_equiv_plan *p = &k->plans[q];
    if (e->rows != v->rows.rows || e->selected != v->rows.selected) return 0;
    uint64_t sum = e->checked + e->excluded;
    if (sum < e->checked || sum + e->undecided < sum || sum + e->undecided != e->selected) return 0;
    if (e->differ > e->checked || e->null_mismatch > e->differ || e->excluded > e->nulls || e->nulls > e->selected) return 0;
    uint64_t cells = 0;
    for (int i = 0; i < 9; i++) { if (cells + e->cells[i] < cells) return 0; cells += e->cells[i]; }
    if (p->kind == TR_EQUIV_MEASURE ? cells != 0 : cells != e->selected) return 0;
    if (p->kind == TR_EQUIV_PREDICATE && (e->excluded || e->undecided || e->nulls || e->null_mismatch)) return 0;
    if (!e->differ) return e->witness == TR_EQUIV_NONE;
    if (e->witness >= e->rows || !tr_scope_is_selected(&v->rows, e->witness)) return 0;
    uint64_t val[2 * TR_EQUIV_MAX_CONE];
    tr_equiv_acc a;
    tr_equiv_acc_init(&a);
    tr_equiv_word(v, q, e->witness / 64, val, &a);
    return a.witness == e->witness;
}

/* ================================================================ results */
typedef struct {
    tr_equiv_verdict v;
    tr_equiv_evidence ev;
    tr_cert *cert;
    int32_t haspoint, ncols;
    uint32_t colmask, reserved;
    int64_t values[NCOLS];
    uint8_t present[NCOLS];
} rq;
struct tr_equiv_result_s {
    uint32_t magic;
    int32_t nq, ndom;
    rq *q;
    tr_cert_record *record;             /* SNAPSHOT verdicts: the scope's record */
    tr_equiv_column dom[NCOLS];         /* DOMAIN verdicts: the declared domain (values in dvalues) */
    int64_t *dvalues;
};
static int result_valid(const tr_equiv_result *r) { return r && r->magic == RESULT_MAGIC; }
void tr_equiv_result_free(tr_equiv_result *r) {
    if (!result_valid(r)) return;
    for (int32_t i = 0; i < r->nq; i++) tr_cert_free(r->q[i].cert);
    free(r->q); free(r->dvalues);
    tr_cert_record_free(r->record);
    r->magic = 0;
    free(r);
}
static tr_equiv_result *result_alloc(int32_t nq, const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error) {
    tr_math_need need;
    need_zero(&need);
    need.need[TR_BUDGET_OUTPUT] = (uint64_t)nq;
    need.need[TR_BUDGET_BYTES] = tr_budget_add(sizeof(tr_equiv_result), tr_budget_mul((uint64_t)nq, sizeof(rq)));
    tr_math_budget store;
    int32_t rc = tr_budget_check(budget_or(budget, &store), &need, refusal);
    if (rc) { set_error(error, rc); return NULL; }
    tr_equiv_result *r = (tr_equiv_result *)zalloc(sizeof *r);
    if (r) r->q = (rq *)zalloc((uint64_t)nq * sizeof(rq));
    if (!r || !r->q) { free(r); set_error(error, TR_MATH_MEMORY); return NULL; }
    r->magic = RESULT_MAGIC; r->nq = nq;
    for (int32_t i = 0; i < nq; i++) r->q[i].v.witness = TR_EQUIV_NONE;
    return r;
}
static void point_from_row(const tr_scope_view *v, uint64_t row, rq *o) {
    o->haspoint = 1; o->ncols = v->ncols;
    for (int32_t c = 0; c < v->ncols; c++) {
        o->present[c] = (uint8_t)tr_scope_present(v, c, row);
        o->values[c] = o->present[c] && (v->col[c].codes || v->col[c].values) ? tr_scope_value(v, c, row) : 0;
    }
}
/* the FND-007 POLYNOMIAL claim P = 0 of a pair (policy SKIP over the measures' columns) */
static void pair_claim_view(const epair *E, tr_cert_claim *c) {
    memset(c, 0, sizeof *c);
    c->kind = TR_CLAIM_POLYNOMIAL; c->policy = TR_CERT_NULL_SKIP; c->carrier = TR_CARRIER_I128;
    c->ncols = E->ncols;
    for (int32_t i = 0; i < TR_CERT_MAX_COLS; i++) c->col[i] = i < E->ncols ? E->col[i] : -1;
    c->nterms = E->nterms; c->exponents = E->e; c->coefficients = E->c;
}
tr_equiv_result *tr_equiv_check_decide(const tr_equiv_check *K, const tr_scope *scope, const tr_equiv_evidence *computed,
                                       int32_t backend, const tr_cert *const *relations, int32_t nrelations,
                                       const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    if (!check_valid(K) || !scope || scope != K->scope || (backend != TR_CERT_BY_CPU && backend != TR_CERT_BY_DEVICE) ||
        nrelations < 0 || (nrelations && !relations) || (computed == NULL && backend != TR_CERT_BY_CPU)) {
        set_error(error, TR_MATH_INVALID); return NULL;
    }
    if (K->record) {
        tr_cert_staleness st;
        int32_t f = tr_cert_record_fresh(K->record, scope, NULL, 0, TR_CERT_MATCH_EXACT, &st);
        if (f) { set_error(error, f == TR_MATH_STALE ? TR_MATH_STALE : f); return NULL; }
    }
    const int32_t nq = K->nq;
    tr_equiv_evidence *ev = (tr_equiv_evidence *)zalloc((uint64_t)nq * sizeof *ev);
    if (!ev) { set_error(error, TR_MATH_MEMORY); return NULL; }
    if (computed) {
        for (int32_t q = 0; q < nq; q++)
            if (!evidence_consistent(K, q, &computed[q])) { free(ev); set_error(error, TR_MATH_INVALID); return NULL; }
        memcpy(ev, computed, (size_t)nq * sizeof *ev);
    } else run_cpu(&K->view, ev);
    tr_equiv_result *R = result_alloc(nq, budget, refusal, error);
    if (!R) { free(ev); return NULL; }
    int32_t rc = TR_MATH_OK;
    tr_geom_ideal *ideal = NULL;
    if (nrelations) {
        int32_t e = 0;
        ideal = tr_geom_ideal_make(relations, nrelations, NULL, 1, budget, refusal, &e);
        if (!ideal) rc = e ? e : TR_MATH_MEMORY;
    }
    for (int32_t q = 0; q < nq && !rc; q++) {
        rq *o = &R->q[q];
        const tr_equiv_plan *p = &K->plans[q];
        o->ev = ev[q]; o->colmask = K->qcols[q];
        tr_equiv_verdict *v = &o->v;
        v->label = TR_EQUIV_LABEL_SNAPSHOT; v->method = TR_EQUIV_METHOD_ROWS; v->work = ev[q].checked;
        if (ev[q].differ) {
            v->verdict = TR_EQUIV_REFUTED; v->witness = ev[q].witness; v->point = 1;
            point_from_row(&K->view.rows, ev[q].witness, o);
        } else if (ev[q].undecided) { v->verdict = TR_EQUIV_UNKNOWN; v->why = TR_EQUIV_WHY_UNDECIDED; }
        else v->verdict = TR_EQUIV_PROVEN;
        const epair *E = K->qpair[q] >= 0 ? &K->pairs[K->qpair[q]] : NULL;
        if (ideal && E && !E->constant && p->kind == TR_EQUIV_MEASURE && v->verdict != TR_EQUIV_REFUTED &&
            (p->nulls == TR_EQUIV_SKIP || p->supa == p->supb)) {
            int64_t *dd = (int64_t *)zalloc((E->nterms ? E->nterms : 1) * 8);
            if (!dd) { rc = TR_MATH_MEMORY; break; }
            for (uint64_t t = 0; t < E->nterms; t++) dd[t] = E->D;
            tr_geom_poly h;
            memset(&h, 0, sizeof h);
            h.ncols = E->ncols;
            for (int32_t i = 0; i < MAXV; i++) h.col[i] = i < E->ncols ? E->col[i] : 0;
            h.nterms = E->nterms; h.exponents = E->e; h.num = E->c; h.den = dd;
            tr_math_refusal r2;
            int32_t e2 = 0;
            tr_geom_reduction *red = tr_geom_reduce(ideal, &h, budget, &r2, &e2);
            if (red) {
                tr_geom_pol *g = tr_geom_reduction_part(red, TR_GEOM_PART_G, &e2);
                int32_t gn = 0, gc[MAXV];
                uint64_t gt = 1;
                if (g) { tr_geom_pol_get(g, &gn, gc, &gt); tr_geom_pol_free(g); }
                if (g && gt == 0) {
                    tr_cert *cert = tr_geom_certify(red, budget, &r2, &e2);
                    tr_cert_decision dec;
                    if (cert && !tr_cert_authorize_exact(cert, 0, scope, NULL, 0, &dec) && dec.authorized) {
                        v->verdict = TR_EQUIV_PROVEN; v->label = TR_EQUIV_LABEL_RELATIONS; v->method = TR_EQUIV_METHOD_REDUCTION;
                        v->why = TR_EQUIV_WHY_NONE;
                        o->cert = cert;
                    } else tr_cert_free(cert);
                }
                tr_geom_reduction_free(red);
            }
            free(dd);
        }
        if (v->verdict == TR_EQUIV_PROVEN && v->label == TR_EQUIV_LABEL_SNAPSHOT && p->kind == TR_EQUIV_MEASURE && E &&
            !E->constant && E->ncols) {
            tr_cert_claim c;
            pair_claim_view(E, &c);
            tr_cert_evidence ce;
            memset(&ce, 0, sizeof ce);
            ce.rows = ev[q].rows; ce.selected = ev[q].selected; ce.excluded = ev[q].nulls;
            ce.checked = ev[q].selected - ev[q].nulls; ce.witness = TR_CERT_NONE; ce.unit = TR_CERT_UNIT_ROWS; ce.holds = 1;
            int32_t e = 0;
            o->cert = tr_cert_adopt(scope, &c, &ce, backend, NULL, 0, budget, refusal, &e);
            if (!o->cert) rc = e ? e : TR_MATH_MEMORY;
        }
    }
    tr_geom_ideal_free(ideal);
    free(ev);
    if (!rc && K->record) {
        int32_t e = 0;
        R->record = tr_cert_record_clone(K->record, &e);
        if (!R->record) rc = e ? e : TR_MATH_MEMORY;
    }
    if (rc) { tr_equiv_result_free(R); set_error(error, rc); return NULL; }
    set_error(error, TR_MATH_OK);
    return R;
}

/* ================================================================ universal decisions */
typedef struct {
    const tr_equiv_set *S;
    const tr_equiv_domain *D;
    const tr_math_budget *budget;
    int32_t dcol[NCOLS];            /* scope column -> domain column (-1) */
    int32_t scale[NCOLS];
    int32_t ncols;                  /* columns of the synthetic scopes */
    uint64_t iters;                 /* searches of the current question */
} uctx;
typedef struct { uint64_t n; int64_t *v[NCOLS]; uint8_t *p[NCOLS]; } pts;
static void pts_free(pts *P, int32_t ncols) {
    for (int32_t c = 0; c < ncols; c++) { free(P->v[c]); free(P->p[c]); }
    memset(P, 0, sizeof *P);
}
static int pts_alloc(pts *P, int32_t ncols, uint64_t n) {
    memset(P, 0, sizeof *P);
    P->n = n;
    for (int32_t c = 0; c < ncols; c++) {
        P->v[c] = (int64_t *)zalloc(n * 8); P->p[c] = (uint8_t *)zalloc(n);
        if (!P->v[c] || !P->p[c]) { pts_free(P, ncols); return 0; }
        memset(P->p[c], 1, (size_t)n);
    }
    return 1;
}
static void unknown(rq *o, int32_t why, const tr_math_refusal *r) {
    o->v.verdict = TR_EQUIV_UNKNOWN; o->v.why = why;
    if (why == TR_EQUIV_WHY_BUDGET && r) { o->v.resource = r->resource; o->v.need = r->need; o->v.limit = r->limit; }
}
static void unknown_need(rq *o, int32_t resource, uint64_t need, uint64_t limit) {
    o->v.verdict = TR_EQUIV_UNKNOWN; o->v.why = TR_EQUIV_WHY_BUDGET;
    o->v.resource = resource; o->v.need = need; o->v.limit = limit;
}
/* question q on the synthetic rows P (the snapshot check of an array scope); TR_MATH_BUDGET / _UNSUPPORTED are
 * reported to the caller with the refusal (an UNKNOWN), other codes are errors */
static int32_t eval_points(const uctx *u, const tr_equiv_question *q, const pts *P, tr_equiv_evidence *ev,
                           tr_math_refusal *refusal) {
    tr_scope_array arr[NCOLS];
    for (int32_t c = 0; c < u->ncols; c++) {
        arr[c].values = P->v[c]; arr[c].valid = P->p[c]; arr[c].scale = u->scale[c]; arr[c].reserved = 0;
    }
    int32_t e = 0;
    tr_scope *sc = tr_scope_from_arrays(P->n, NULL, arr, u->ncols, NULL, 1, u->budget, refusal, &e);
    if (!sc) return e ? e : TR_MATH_MEMORY;
    tr_equiv_check *k = check_build(u->S, q, 1, sc, NULL, NULL, 0, u->budget, refusal, &e);
    if (!k) { tr_scope_free(sc); return e ? e : TR_MATH_MEMORY; }
    run_cpu(&k->view, ev);
    tr_equiv_check_free(k);
    tr_scope_free(sc);
    return TR_MATH_OK;
}
static void point_from_pts(const uctx *u, const pts *P, uint64_t row, rq *o) {
    o->haspoint = 1; o->ncols = u->ncols; o->v.point = 1;
    for (int32_t c = 0; c < u->ncols; c++) { o->present[c] = P->p[c][row] ? 1 : 0; o->values[c] = o->present[c] ? P->v[c][row] : 0; }
}
static const tr_equiv_column *dcol_of(const uctx *u, int32_t c) { return u->dcol[c] >= 0 ? &u->D->cols[u->dcol[c]] : NULL; }
static uint64_t dom_count(const tr_equiv_column *d) {
    if (d->kind == TR_EQUIV_VALUES) return d->nvalues;
    __int128 n = (__int128)d->hi - d->lo + 1;
    return n > (__int128)UINT64_MAX ? UINT64_MAX : (uint64_t)n;
}
static int64_t dom_value(const tr_equiv_column *d, uint64_t i) {
    return d->kind == TR_EQUIV_VALUES ? d->values[i] : (int64_t)((__int128)d->lo + (__int128)i);
}
/* k consecutive integers of [L, U] (k <= U - L + 1) nearest 0 */
static void near0_range(__int128 L, __int128 U, uint64_t k, int64_t *out) {
    __int128 s = -(__int128)((k - 1) / 2);
    if (s < L) s = L;
    if (s > U - (__int128)k + 1) s = U - (__int128)k + 1;
    for (uint64_t i = 0; i < k; i++) out[i] = (int64_t)(s + (__int128)i);
}
/* the k values of a domain column nearest 0 */
static void near0_dom(const tr_equiv_column *d, uint64_t k, int64_t *out) {
    if (d->kind == TR_EQUIV_RANGE) { near0_range(d->lo, d->hi, k, out); return; }
    uint64_t n = d->nvalues, r = 0;
    while (r < n && d->values[r] < 0) r++;
    uint64_t l = r;                                   /* window [l, r) grows toward the smaller magnitude */
    for (uint64_t i = 0; i < k; i++) {
        if (r < n && (l == 0 || (unsigned __int128)d->values[r] <= (unsigned __int128)(-(__int128)d->values[l - 1]))) out[i] = d->values[r++];
        else out[i] = d->values[--l];
    }
}
/* the raw terms D f - D g over variables 0..n-1: the FND-007 ring-identity law when they cancel */
static tr_cert *law_of(const emeasure *f, const emeasure *g, const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error) {
    epair E;
    memset(&E, 0, sizeof E);
    int32_t n = 0, col[MAXV];
    for (int32_t s = 0; s < 2; s++) {
        const emeasure *m = s ? g : f;
        for (int32_t i = 0; i < m->ncols; i++) {
            int seen = 0;
            for (int32_t j = 0; j < n; j++) seen |= col[j] == m->col[i];
            if (seen) continue;
            if (n >= MAXV) { set_error(error, TR_MATH_UNSUPPORTED); return NULL; }
            col[n++] = m->col[i];
        }
    }
    int64_t D = lcm64(f->D, g->D);
    uint64_t raw = f->nterms + g->nterms;
    int32_t w = n ? n : 1;
    uint8_t *e = (uint8_t *)zalloc(raw * (uint64_t)w);
    int64_t *c = (int64_t *)zalloc(raw * 8);
    if (!D || !e || !c) { free(e); free(c); set_error(error, D ? TR_MATH_MEMORY : TR_MATH_OVERFLOW); return NULL; }
    for (int32_t s = 0; s < 2; s++) {
        const emeasure *m = s ? g : f;
        for (uint64_t k = 0; k < m->nterms; k++) {
            uint64_t at = (s ? f->nterms : 0) + k;
            for (int32_t i = 0; i < n; i++) e[at * (uint64_t)w + (uint64_t)i] = measure_exp(m, k, col[i]);
            __int128 v = (__int128)m->num[k] * (D / m->den[k]);
            if (s) v = -v;
            if (v < INT64_MIN || v > INT64_MAX) { free(e); free(c); set_error(error, TR_MATH_OVERFLOW); return NULL; }
            c[at] = (int64_t)v;
        }
    }
    tr_cert_claim k;
    memset(&k, 0, sizeof k);
    k.kind = TR_CLAIM_POLYNOMIAL; k.policy = TR_CERT_NULL_FORBID; k.carrier = TR_CARRIER_I128;
    k.ncols = w;
    for (int32_t i = 0; i < TR_CERT_MAX_COLS; i++) k.col[i] = i < w ? i : -1;
    k.nterms = raw; k.exponents = e; k.coefficients = c;
    tr_cert *cert = tr_cert_law(TR_CERT_LAW_RING_IDENTITY, &k, budget, refusal, error);
    free(e); free(c);
    return cert;
}

/* ---------------------------------------------------------------- exhaustive enumeration */
static int32_t decide_enumerate(uctx *u, const tr_equiv_question *q, uint32_t C, rq *o, tr_math_refusal *refusal) {
    int32_t list[NCOLS], m = 0;
    for (int32_t c = 0; c < NCOLS; c++) if ((C >> c) & 1) list[m++] = c;
    uint64_t radix[NCOLS], total = 1;
    for (int32_t i = 0; i < m; i++) {
        const tr_equiv_column *d = dcol_of(u, list[i]);
        radix[i] = tr_budget_add(dom_count(d), (uint64_t)(d->nullable ? 1 : 0));
        total = tr_budget_mul(total, radix[i]);
    }
    o->v.label = TR_EQUIV_LABEL_DOMAIN; o->v.method = TR_EQUIV_METHOD_ENUMERATE;
    if (total > u->budget->limit[TR_BUDGET_SUPPORT]) { unknown_need(o, TR_BUDGET_SUPPORT, total, u->budget->limit[TR_BUDGET_SUPPORT]); return TR_MATH_OK; }
    uint64_t bytes = tr_budget_mul(total, (uint64_t)u->ncols * 9);
    if (bytes > u->budget->limit[TR_BUDGET_BYTES]) { unknown_need(o, TR_BUDGET_BYTES, bytes, u->budget->limit[TR_BUDGET_BYTES]); return TR_MATH_OK; }
    pts P;
    if (!pts_alloc(&P, u->ncols, total)) return TR_MATH_MEMORY;
    for (uint64_t r = 0; r < total; r++) {
        uint64_t x = r;
        for (int32_t i = m - 1; i >= 0; i--) {
            const tr_equiv_column *d = dcol_of(u, list[i]);
            uint64_t digit = x % radix[i];
            x /= radix[i];
            if (digit == dom_count(d)) { P.p[list[i]][r] = 0; P.v[list[i]][r] = 0; }
            else P.v[list[i]][r] = dom_value(d, digit);
        }
    }
    tr_equiv_evidence ev;
    int32_t rc = eval_points(u, q, &P, &ev, refusal);
    o->v.work = total;
    if (rc == TR_MATH_BUDGET) unknown(o, TR_EQUIV_WHY_BUDGET, refusal);
    else if (rc == TR_MATH_UNSUPPORTED) unknown(o, TR_EQUIV_WHY_UNSUPPORTED, NULL);
    else if (!rc) {
        o->ev = ev;
        if (ev.differ) { o->v.verdict = TR_EQUIV_REFUTED; point_from_pts(u, &P, ev.witness, o); }
        else if (ev.undecided) unknown(o, TR_EQUIV_WHY_UNDECIDED, NULL);
        else o->v.verdict = TR_EQUIV_PROVEN;
    }
    pts_free(&P, u->ncols);
    return rc == TR_MATH_BUDGET || rc == TR_MATH_UNSUPPORTED ? TR_MATH_OK : rc;
}

/* ---------------------------------------------------------------- measures: normal forms and Nullstellensatz grids */
/* grid points of P's variables (sets near 0 inside L..U per column), the other present columns near 0, NULL ones
 * absent; evaluated: 1 a witness (o's point), 0 none, 2 UNKNOWN (o says why), or a negative error code */
enum { S_NONE = 0, S_FOUND = 1, S_UNKNOWN = 2 };
static int32_t grid_search(uctx *u, const tr_equiv_question *q, const epair *E, uint32_t C, uint32_t nullmask,
                           const __int128 *L, const __int128 *U, const tr_equiv_column *const *dc, rq *o,
                           tr_math_refusal *refusal) {
    int32_t vars[MAXV], nv = 0;
    uint64_t k[MAXV], total = 1;
    for (int32_t i = 0; i < E->ncols; i++) {
        int32_t c = E->col[i];
        uint8_t deg = 0;
        for (uint64_t t = 0; t < E->nterms; t++) if (E->e[t * (uint64_t)E->ncols + (uint64_t)i] > deg) deg = E->e[t * (uint64_t)E->ncols + (uint64_t)i];
        if (!deg) continue;
        uint64_t avail = dc && dc[c] ? dom_count(dc[c]) : (uint64_t)(U[c] - L[c] + 1 > (__int128)UINT64_MAX ? UINT64_MAX : (uint64_t)(U[c] - L[c] + 1));
        k[nv] = avail < (uint64_t)deg + 1 ? avail : (uint64_t)deg + 1;
        vars[nv++] = c;
        total = tr_budget_mul(total, k[nv - 1]);
    }
    o->v.method = TR_EQUIV_METHOD_GRID;
    if (total > u->budget->limit[TR_BUDGET_SUPPORT]) { unknown_need(o, TR_BUDGET_SUPPORT, total, u->budget->limit[TR_BUDGET_SUPPORT]); return S_UNKNOWN; }
    uint64_t bytes = tr_budget_mul(total, (uint64_t)u->ncols * 9);
    if (bytes > u->budget->limit[TR_BUDGET_BYTES]) { unknown_need(o, TR_BUDGET_BYTES, bytes, u->budget->limit[TR_BUDGET_BYTES]); return S_UNKNOWN; }
    int64_t sets[MAXV][257];                              /* deg <= 255: at most 256 values per variable */
    for (int32_t i = 0; i < nv; i++) {
        if (dc && dc[vars[i]]) near0_dom(dc[vars[i]], k[i], sets[i]);
        else near0_range(L[vars[i]], U[vars[i]], k[i], sets[i]);
    }
    pts P;
    if (!pts_alloc(&P, u->ncols, total)) return TR_MATH_MEMORY;
    for (int32_t c = 0; c < u->ncols; c++) {
        if (!((C >> c) & 1)) continue;
        int64_t v0;
        if ((nullmask >> c) & 1) { memset(P.p[c], 0, (size_t)total); continue; }
        if (dc && dc[c]) near0_dom(dc[c], 1, &v0);
        else near0_range(L[c], U[c], 1, &v0);
        for (uint64_t r = 0; r < total; r++) P.v[c][r] = v0;
    }
    for (uint64_t r = 0; r < total; r++) {
        uint64_t x = r;
        for (int32_t i = nv - 1; i >= 0; i--) { P.v[vars[i]][r] = sets[i][x % k[i]]; x /= k[i]; }
    }
    tr_equiv_evidence ev;
    int32_t rc = eval_points(u, q, &P, &ev, refusal);
    o->v.work += total;
    int32_t out = S_NONE;
    if (rc == TR_MATH_BUDGET) { unknown(o, TR_EQUIV_WHY_BUDGET, refusal); out = S_UNKNOWN; }
    else if (rc == TR_MATH_UNSUPPORTED) { unknown(o, TR_EQUIV_WHY_UNSUPPORTED, NULL); out = S_UNKNOWN; }
    else if (rc) out = rc;
    else if (ev.differ) { point_from_pts(u, &P, ev.witness, o); out = S_FOUND; }
    else if (ev.undecided) { unknown(o, TR_EQUIV_WHY_UNDECIDED, NULL); out = S_UNKNOWN; }
    pts_free(&P, u->ncols);
    return out;
}
static int32_t decide_measure(uctx *u, const tr_equiv_question *q, const sides *d, uint32_t C, rq *o, tr_math_refusal *refusal) {
    const emeasure *f = &u->S->ms[d->ma], *g = &u->S->ms[d->mb];
    if (d->ma == d->mb) {                                 /* one canonical measure: a ring identity */
        o->v.verdict = TR_EQUIV_PROVEN; o->v.label = TR_EQUIV_LABEL_UNIVERSAL; o->v.method = TR_EQUIV_METHOD_NORMAL_FORM;
        int32_t e = 0;
        o->cert = law_of(f, g, u->budget, refusal, &e);
        if (!o->cert && e != TR_MATH_UNSUPPORTED && e != TR_MATH_BUDGET) return e ? e : TR_MATH_MEMORY;
        return TR_MATH_OK;
    }
    o->v.label = TR_EQUIV_LABEL_DOMAIN;
    for (int32_t c = 0; c < NCOLS; c++)                   /* a column only one side reads, NULL: one side NULL */
        if (q->nulls == TR_EQUIV_SQL && (((f->cols ^ g->cols) >> c) & 1) && dcol_of(u, c)->nullable) {
            pts P;
            if (!pts_alloc(&P, u->ncols, 1)) return TR_MATH_MEMORY;
            for (int32_t x = 0; x < u->ncols; x++)
                if ((C >> x) & 1) near0_dom(dcol_of(u, x), 1, &P.v[x][0]);
            P.p[c][0] = 0; P.v[c][0] = 0;
            tr_equiv_evidence ev;
            int32_t rc = eval_points(u, q, &P, &ev, refusal);
            o->v.method = TR_EQUIV_METHOD_ENUMERATE; o->v.work = 1;
            if (!rc && ev.differ) { o->v.verdict = TR_EQUIV_REFUTED; point_from_pts(u, &P, 0, o); }
            pts_free(&P, u->ncols);
            if (rc == TR_MATH_BUDGET) { unknown(o, TR_EQUIV_WHY_BUDGET, refusal); return TR_MATH_OK; }
            if (rc == TR_MATH_UNSUPPORTED) { unknown(o, TR_EQUIV_WHY_UNSUPPORTED, NULL); return TR_MATH_OK; }
            if (rc) return rc;
            if (o->v.verdict == TR_EQUIV_REFUTED) return TR_MATH_OK;
            break;
        }
    epair E;
    memset(&E, 0, sizeof E);
    int32_t rc = TR_MATH_OK;
    if (!f->ncols && !g->ncols) E.nterms = 1;             /* two different constants: any point */
    else rc = pair_claim(f, g, &E);
    if (rc == TR_MATH_UNSUPPORTED || rc == TR_MATH_OVERFLOW) { free(E.e); free(E.c); unknown(o, TR_EQUIV_WHY_UNSUPPORTED, NULL); return TR_MATH_OK; }
    if (rc) { free(E.e); free(E.c); return rc; }
    const tr_equiv_column *dc[NCOLS];
    __int128 L[NCOLS], U[NCOLS];
    for (int32_t c = 0; c < NCOLS; c++) { dc[c] = (C >> c) & 1 ? dcol_of(u, c) : NULL; L[c] = U[c] = 0; }
    int32_t s = grid_search(u, q, &E, C, 0, L, U, dc, o, refusal);
    free(E.e); free(E.c);
    if (s < 0) return s;
    if (s == S_FOUND) o->v.verdict = TR_EQUIV_REFUTED;
    else if (s == S_NONE) o->v.verdict = TR_EQUIV_PROVEN;   /* P vanishes on the box (the grid covers each small range) */
    if (o->v.verdict != TR_EQUIV_UNKNOWN) o->v.why = TR_EQUIV_WHY_NONE;
    return TR_MATH_OK;
}

/* ---------------------------------------------------------------- predicates: reduced ordered BDDs */
typedef struct { int32_t var, lo, hi; } bnode;
typedef struct { int32_t op, a, b, r; } bmemo;
typedef struct {
    bnode *n;
    uint32_t count, cap;
    int32_t *uniq;
    uint64_t umask;
    bmemo *memo;
    uint64_t mmask;
    uint64_t limit;
    int32_t over, nomem;
} bdd;
enum { B_AND = 1, B_OR = 2, B_XOR = 3 };
#define BTERM INT32_MAX
static void bdd_free(bdd *B) { free(B->n); free(B->uniq); free(B->memo); memset(B, 0, sizeof *B); }
static int bdd_init(bdd *B, uint64_t limit) {
    memset(B, 0, sizeof *B);
    B->cap = 1024; B->limit = limit;
    B->n = (bnode *)malloc(B->cap * sizeof *B->n);
    B->umask = 2047; B->uniq = (int32_t *)malloc((size_t)(B->umask + 1) * 4);
    B->mmask = (1u << 14) - 1; B->memo = (bmemo *)calloc((size_t)(B->mmask + 1), sizeof *B->memo);
    if (!B->n || !B->uniq || !B->memo) { bdd_free(B); return 0; }
    memset(B->uniq, 0xFF, (size_t)(B->umask + 1) * 4);
    B->n[0] = (bnode){BTERM, 0, 0}; B->n[1] = (bnode){BTERM, 1, 1}; B->count = 2;
    return 1;
}
static uint64_t bhash(int32_t v, int32_t lo, int32_t hi) { return mix(mix(mix(0x424444, (uint64_t)v), (uint64_t)lo), (uint64_t)hi); }
static int32_t bdd_mk(bdd *B, int32_t v, int32_t lo, int32_t hi) {
    if (lo < 0 || hi < 0) return -1;
    if (lo == hi) return lo;
    uint64_t i = bhash(v, lo, hi) & B->umask;
    for (; B->uniq[i] >= 0; i = (i + 1) & B->umask) {
        const bnode *m = &B->n[B->uniq[i]];
        if (m->var == v && m->lo == lo && m->hi == hi) return B->uniq[i];
    }
    if (B->count >= B->limit) { B->over = 1; return -1; }
    if (B->count == B->cap) {
        bnode *g = (bnode *)realloc(B->n, (size_t)B->cap * 2 * sizeof *g);
        if (!g) { B->nomem = 1; return -1; }
        B->n = g; B->cap *= 2;
    }
    if (2 * ((uint64_t)B->count + 1) > B->umask + 1) {
        uint64_t cap = 2 * (B->umask + 1);
        int32_t *t = (int32_t *)malloc((size_t)cap * 4);
        if (!t) { B->nomem = 1; return -1; }
        memset(t, 0xFF, (size_t)cap * 4);
        for (uint32_t id = 2; id < B->count; id++) {
            uint64_t j = bhash(B->n[id].var, B->n[id].lo, B->n[id].hi) & (cap - 1);
            while (t[j] >= 0) j = (j + 1) & (cap - 1);
            t[j] = (int32_t)id;
        }
        free(B->uniq); B->uniq = t; B->umask = cap - 1;
        i = bhash(v, lo, hi) & B->umask;
        while (B->uniq[i] >= 0) i = (i + 1) & B->umask;
    }
    int32_t id = (int32_t)B->count++;
    B->n[id] = (bnode){v, lo, hi};
    B->uniq[i] = id;
    return id;
}
static int32_t bdd_apply(bdd *B, int32_t op, int32_t a, int32_t b) {
    if (a < 0 || b < 0) return -1;
    if (op == B_AND) { if (!a || !b) return 0; if (a == 1) return b; if (b == 1) return a; if (a == b) return a; }
    else if (op == B_OR) { if (a == 1 || b == 1) return 1; if (!a) return b; if (!b) return a; if (a == b) return a; }
    else { if (!a) return b; if (!b) return a; if (a == b) return 0; }
    if (a > b) { int32_t t = a; a = b; b = t; }
    uint64_t slot = mix(mix(mix(0x4150, (uint64_t)op), (uint64_t)a), (uint64_t)b) & B->mmask;
    bmemo *m = &B->memo[slot];
    if (m->op == op && m->a == a && m->b == b) return m->r;
    int32_t va = B->n[a].var, vb = B->n[b].var, v = va < vb ? va : vb;
    int32_t a0 = va == v ? B->n[a].lo : a, a1 = va == v ? B->n[a].hi : a;
    int32_t b0 = vb == v ? B->n[b].lo : b, b1 = vb == v ? B->n[b].hi : b;
    int32_t lo = bdd_apply(B, op, a0, b0);
    int32_t hi = lo < 0 ? -1 : bdd_apply(B, op, a1, b1);
    int32_t r = bdd_mk(B, v, lo, hi);
    if (r >= 0) { m = &B->memo[slot]; m->op = op; m->a = a; m->b = b; m->r = r; }
    return r;
}
typedef struct { int32_t t, f; } tf;
enum { CS_ATOM = 0, CS_TRUE = 1, CS_FALSE = 2, CS_UNKNOWN = 3 };
static int8_t cond_status(const tr_bool_condition *c, uint32_t nullmask) {
    int xn = (int)((nullmask >> c->col) & 1);
    if (c->op == TR_BOOL_IS_NULL) return xn ? CS_TRUE : CS_FALSE;
    if (c->op == TR_BOOL_IS_NOT_NULL) return xn ? CS_FALSE : CS_TRUE;
    if (xn || (c->op >= TR_BOOL_COL_EQ && ((nullmask >> c->col2) & 1))) return CS_UNKNOWN;
    return CS_ATOM;
}
/* the TRUE and FALSE sets of every cone node under a NULL pattern */
static int cone_tf(bdd *B, const tr_equiv_set *S, const int32_t *cone, int32_t n, const int8_t *cs, int32_t *pos, tf *val) {
    for (int32_t i = 0; i < n; i++) pos[cone[i]] = i;
    for (int32_t i = 0; i < n; i++) {
        const tr_bool_node *m = &S->nodes[cone[i]];
        tf r = {0, 0};
        switch (m->op) {
        case TR_BOOL_CONST: r.t = m->a == TR_BOOL_TRUE; r.f = m->a == TR_BOOL_FALSE; break;
        case TR_BOOL_VAR:
            if (cs[m->a] == CS_ATOM) { r.t = bdd_mk(B, m->a, 0, 1); r.f = bdd_mk(B, m->a, 1, 0); }
            else { r.t = cs[m->a] == CS_TRUE; r.f = cs[m->a] == CS_FALSE; }
            break;
        case TR_BOOL_NOT: r.t = val[pos[m->a]].f; r.f = val[pos[m->a]].t; break;
        default: {
            tf x = val[pos[m->a]], y = val[pos[m->b]];
            if (m->op == TR_BOOL_AND) { r.t = bdd_apply(B, B_AND, x.t, y.t); r.f = bdd_apply(B, B_OR, x.f, y.f); }
            else if (m->op == TR_BOOL_OR) { r.t = bdd_apply(B, B_OR, x.t, y.t); r.f = bdd_apply(B, B_AND, x.f, y.f); }
            else {
                r.t = bdd_apply(B, B_OR, bdd_apply(B, B_AND, x.t, y.f), bdd_apply(B, B_AND, x.f, y.t));
                r.f = bdd_apply(B, B_OR, bdd_apply(B, B_AND, x.t, y.t), bdd_apply(B, B_AND, x.f, y.f));
            }
        }
        }
        if (r.t < 0 || r.f < 0) return 0;
        val[i] = r;
    }
    return 1;
}

/* ---------------------------------------------------------------- the integer theory of a cube */
typedef struct { int32_t x, y; int64_t c; } dcons;          /* x - y <= c */
typedef struct { int32_t kind, x, y, reserved; __int128 a, b; } branch;   /* 1: x != a; 2: x outside [a, b]; 3: x != y */
typedef struct {
    __int128 L[NCOLS], U[NCOLS];
    dcons dc[2 * TR_EQUIV_MAX_CONDITIONS + 2];
    int32_t ndc, nbr;
    branch br[TR_EQUIV_MAX_CONDITIONS];
    int32_t outside, infeasible;
} theory;
static __int128 fdiv(__int128 a, __int128 d) { __int128 q = a / d; if (a % d != 0 && ((a < 0) != (d < 0))) q--; return q; }
static __int128 cdiv(__int128 a, __int128 d) { return -fdiv(-a, d); }
static __int128 p10(int32_t s) { __int128 p = 1; for (int32_t i = 0; i < s; i++) p *= 10; return p; }
static void lower(theory *T, int32_t x, __int128 v) { if (v > T->L[x]) T->L[x] = v; }
static void upper(theory *T, int32_t x, __int128 v) { if (v < T->U[x]) T->U[x] = v; }
static void lit_add(const uctx *u, theory *T, const tr_bool_condition *c, int pol) {
    int32_t x = c->col;
    int32_t op = c->op;
    if (op >= TR_BOOL_COL_EQ) {
        int32_t y = c->col2;
        if (u->scale[x] != u->scale[y]) { T->outside = 1; return; }
        op = op - TR_BOOL_COL_EQ + TR_BOOL_EQ;
        if (!pol) op = op == TR_BOOL_EQ ? TR_BOOL_NE : op == TR_BOOL_NE ? TR_BOOL_EQ : op == TR_BOOL_LT ? TR_BOOL_GE :
                       op == TR_BOOL_LE ? TR_BOOL_GT : op == TR_BOOL_GT ? TR_BOOL_LE : TR_BOOL_LT;
        switch (op) {
        case TR_BOOL_EQ: T->dc[T->ndc++] = (dcons){x, y, 0}; T->dc[T->ndc++] = (dcons){y, x, 0}; break;
        case TR_BOOL_NE: T->br[T->nbr++] = (branch){3, x, y, 0, 0, 0}; break;
        case TR_BOOL_LT: T->dc[T->ndc++] = (dcons){x, y, -1}; break;
        case TR_BOOL_LE: T->dc[T->ndc++] = (dcons){x, y, 0}; break;
        case TR_BOOL_GT: T->dc[T->ndc++] = (dcons){y, x, -1}; break;
        default: T->dc[T->ndc++] = (dcons){y, x, 0}; break;
        }
        return;
    }
    const __int128 P = p10(u->scale[x]);
    const __int128 K0 = (__int128)c->num[0] * P, d0 = c->den[0];
    switch (op) {
    case TR_BOOL_EQ: case TR_BOOL_NE: {
        int eq = (op == TR_BOOL_EQ) == (pol != 0);
        if (K0 % d0) { if (eq) T->infeasible = 1; return; }  /* never equal to a value of this scale */
        __int128 v = K0 / d0;
        if (eq) { lower(T, x, v); upper(T, x, v); }
        else T->br[T->nbr++] = (branch){1, x, -1, 0, v, 0};
        return;
    }
    case TR_BOOL_LT: case TR_BOOL_LE: case TR_BOOL_GT: case TR_BOOL_GE:
        if (!pol) op = op == TR_BOOL_LT ? TR_BOOL_GE : op == TR_BOOL_LE ? TR_BOOL_GT : op == TR_BOOL_GT ? TR_BOOL_LE : TR_BOOL_LT;
        if (op == TR_BOOL_LT) upper(T, x, cdiv(K0, d0) - 1);
        else if (op == TR_BOOL_LE) upper(T, x, fdiv(K0, d0));
        else if (op == TR_BOOL_GT) lower(T, x, fdiv(K0, d0) + 1);
        else lower(T, x, cdiv(K0, d0));
        return;
    case TR_BOOL_BETWEEN: {
        __int128 lo = cdiv(K0, d0), hi = fdiv((__int128)c->num[1] * P, c->den[1]);
        if (pol) { lower(T, x, lo); upper(T, x, hi); }
        else if (lo <= hi) T->br[T->nbr++] = (branch){2, x, -1, 0, lo, hi};
        return;
    }
    default:
        T->outside = 1;                                   /* BIT: not a linear integer constraint */
    }
}
/* bounds and difference constraints over the present columns: Bellman-Ford shortest paths (exact over Z) */
static int solve(const theory *T, uint32_t present, int64_t *x) {
    int32_t id[NCOLS], nv = 1, cols[NCOLS + 1];
    for (int32_t c = 0; c < NCOLS; c++) {
        if (!((present >> c) & 1)) continue;
        if (T->L[c] > T->U[c]) return 0;
        id[c] = nv; cols[nv++] = c;
    }
    __int128 dist[NCOLS + 1];
    for (int32_t i = 0; i < nv; i++) dist[i] = 0;
    for (int32_t round = 0; round <= nv; round++) {
        int changed = 0;
        for (int32_t i = 1; i < nv; i++) {
            int32_t c = cols[i];
            if (dist[0] + T->U[c] < dist[i]) { dist[i] = dist[0] + T->U[c]; changed = 1; }
            if (dist[i] - T->L[c] < dist[0]) { dist[0] = dist[i] - T->L[c]; changed = 1; }
        }
        for (int32_t k = 0; k < T->ndc; k++) {
            int32_t a = id[T->dc[k].x], b = id[T->dc[k].y];
            if (dist[b] + T->dc[k].c < dist[a]) { dist[a] = dist[b] + T->dc[k].c; changed = 1; }
        }
        if (!changed) {
            for (int32_t i = 1; i < nv; i++) x[cols[i]] = (int64_t)(dist[i] - dist[0]);
            return 1;
        }
    }
    return 0;                                              /* still relaxing after |V| rounds: a negative cycle */
}
typedef struct {
    uctx *u;
    const tr_equiv_question *q;
    uint32_t C, nullmask;
    int grid;                       /* 0: a feasible point is a witness; 1: P's grids on the cube's boxes */
    const epair *E;
    rq *o;
    tr_math_refusal *refusal;
    int32_t undecided, stop;        /* stop: S_FOUND, S_UNKNOWN (o says why), or a negative error code */
    uint64_t cubes;
} sctx;
static int32_t search_point(sctx *s, const theory *T) {
    uctx *u = s->u;
    uint32_t present = s->C & ~s->nullmask;
    if (s->grid) {
        __int128 L[NCOLS], U[NCOLS];
        for (int32_t c = 0; c < NCOLS; c++) { L[c] = T->L[c]; U[c] = T->U[c]; if (((present >> c) & 1) && L[c] > U[c]) return 0; }
        int32_t r = grid_search(u, s->q, s->E, s->C, s->nullmask, L, U, NULL, s->o, s->refusal);
        if (r != S_NONE) { s->stop = r; return 1; }
        return 0;
    }
    int64_t x[NCOLS];
    if (!solve(T, present, x)) return 0;
    pts P;
    if (!pts_alloc(&P, u->ncols, 1)) { s->stop = TR_MATH_MEMORY; return 1; }
    for (int32_t c = 0; c < u->ncols; c++) {
        if (!((s->C >> c) & 1)) continue;
        if ((s->nullmask >> c) & 1) P.p[c][0] = 0;
        else P.v[c][0] = x[c];
    }
    tr_equiv_evidence ev;
    int32_t rc = eval_points(u, s->q, &P, &ev, s->refusal);
    s->o->v.work++;
    if (rc == TR_MATH_BUDGET) { unknown(s->o, TR_EQUIV_WHY_BUDGET, s->refusal); s->stop = S_UNKNOWN; }
    else if (rc == TR_MATH_UNSUPPORTED) { unknown(s->o, TR_EQUIV_WHY_UNSUPPORTED, NULL); s->stop = S_UNKNOWN; }
    else if (rc) s->stop = rc;
    else if (ev.differ) { point_from_pts(u, &P, 0, s->o); s->stop = S_FOUND; }
    else s->undecided = 1;                                 /* the theory and the evaluation disagree: never claimed */
    pts_free(&P, u->ncols);
    return s->stop != 0;
}
static int32_t branches(sctx *s, theory *T, int32_t i) {
    if (++s->u->iters > s->u->budget->limit[TR_BUDGET_ITERATIONS]) {
        unknown_need(s->o, TR_BUDGET_ITERATIONS, s->u->iters, s->u->budget->limit[TR_BUDGET_ITERATIONS]);
        s->stop = S_UNKNOWN; return 1;
    }
    if (i == T->nbr) return search_point(s, T);
    const branch *b = &T->br[i];
    for (int opt = 0; opt < 2; opt++) {
        __int128 L = b->x >= 0 ? T->L[b->x] : 0, U = b->x >= 0 ? T->U[b->x] : 0;
        int32_t ndc = T->ndc;
        if (b->kind == 1) { if (opt) lower(T, b->x, b->a + 1); else upper(T, b->x, b->a - 1); }
        else if (b->kind == 2) { if (opt) lower(T, b->x, b->b + 1); else upper(T, b->x, b->a - 1); }
        else T->dc[T->ndc++] = opt ? (dcons){b->y, b->x, -1} : (dcons){b->x, b->y, -1};
        int32_t r = branches(s, T, i + 1);
        T->L[b->x] = L; T->U[b->x] = U; T->ndc = ndc;
        if (r) return r;
    }
    return 0;
}
typedef struct { int32_t var; int32_t pol; } lit;
static int32_t cube_found(sctx *s, const lit *path, int32_t n) {
    uctx *u = s->u;
    s->cubes++;
    if (++u->iters > u->budget->limit[TR_BUDGET_ITERATIONS]) {
        unknown_need(s->o, TR_BUDGET_ITERATIONS, u->iters, u->budget->limit[TR_BUDGET_ITERATIONS]);
        s->stop = S_UNKNOWN; return 1;
    }
    theory *T = (theory *)malloc(sizeof *T);
    if (!T) { s->stop = TR_MATH_MEMORY; return 1; }
    T->ndc = T->nbr = T->outside = T->infeasible = 0;
    for (int32_t c = 0; c < NCOLS; c++) {
        const tr_equiv_column *d = ((s->C >> c) & 1) ? dcol_of(u, c) : NULL;
        T->L[c] = d ? d->lo : 0; T->U[c] = d ? d->hi : 0;
    }
    for (int32_t i = 0; i < n; i++) lit_add(u, T, &u->S->conds[path[i].var], path[i].pol);
    int32_t r = 0;
    if (T->infeasible) r = 0;
    else if (T->outside || (s->grid && T->ndc)) s->undecided = 1;
    else {
        int col = 0;
        for (int32_t i = 0; i < T->nbr; i++) col |= T->br[i].kind == 3;
        if (s->grid && col) s->undecided = 1;
        else r = branches(s, T, 0);
    }
    free(T);
    return r;
}
static int32_t cubes_of(sctx *s, bdd *B, int32_t root, lit *path, int32_t depth) {
    if (root == 0) return 0;
    if (root == 1) return cube_found(s, path, depth);
    const bnode *m = &B->n[root];
    path[depth].var = m->var; path[depth].pol = 0;
    int32_t lo = m->lo, hi = m->hi;
    if (cubes_of(s, B, lo, path, depth + 1)) return 1;
    path[depth].var = m->var; path[depth].pol = 1;
    return cubes_of(s, B, hi, path, depth + 1);
}
static int32_t decide_logic(uctx *u, const tr_equiv_question *q, const sides *d, uint32_t C, rq *o, tr_math_refusal *refusal) {
    const tr_equiv_set *S = u->S;
    o->v.label = TR_EQUIV_LABEL_DOMAIN; o->v.method = TR_EQUIV_METHOD_BDD;
    if (d->ra == d->rb && d->ma == d->mb) {               /* one shared DAG node and one measure: identical parts */
        o->v.verdict = TR_EQUIV_PROVEN; o->v.label = TR_EQUIV_LABEL_UNIVERSAL; o->v.method = TR_EQUIV_METHOD_NORMAL_FORM;
        return TR_MATH_OK;
    }
    int32_t nl[NCOLS], m = 0, allnull = 1;
    for (int32_t c = 0; c < NCOLS; c++) {
        if (!((C >> c) & 1)) continue;
        const tr_equiv_column *dc = dcol_of(u, c);
        if (dc->kind != TR_EQUIV_RANGE) { unknown(o, TR_EQUIV_WHY_UNSUPPORTED, NULL); return TR_MATH_OK; }
        if (dc->nullable) nl[m++] = c; else allnull = 0;
    }
    if (m > MAXPAT) { unknown(o, TR_EQUIV_WHY_UNSUPPORTED, NULL); return TR_MATH_OK; }
    int32_t *stamp = (int32_t *)zalloc((uint64_t)S->nnodes * 4), *pos = (int32_t *)zalloc((uint64_t)S->nnodes * 4);
    int32_t cone[TR_EQUIV_MAX_CONE];
    tf *val = (tf *)zalloc(TR_EQUIV_MAX_CONE * sizeof(tf));
    if (!stamp || !pos || !val) { free(stamp); free(pos); free(val); return TR_MATH_MEMORY; }
    int32_t n = cone_of(S, d->ra, d->rb, stamp, 1, cone, TR_EQUIV_MAX_CONE);
    int32_t pa = -1, pb = -1;
    free(stamp);
    if (n < 0) { free(pos); free(val); unknown(o, TR_EQUIV_WHY_UNSUPPORTED, NULL); return TR_MATH_OK; }
    for (int32_t i = 0; i < n; i++) { if (cone[i] == d->ra) pa = i; if (cone[i] == d->rb) pb = i; }
    epair E;
    memset(&E, 0, sizeof E);
    int pzero = d->ma == d->mb;
    if (!pzero) {
        const emeasure *f = &S->ms[d->ma], *g = &S->ms[d->mb];
        int32_t rc = (!f->ncols && !g->ncols) ? (E.nterms = 1, TR_MATH_OK) : pair_claim(f, g, &E);
        if (rc) { free(E.e); free(E.c); free(pos); free(val); if (rc == TR_MATH_UNSUPPORTED || rc == TR_MATH_OVERFLOW) { unknown(o, TR_EQUIV_WHY_UNSUPPORTED, NULL); return TR_MATH_OK; } return rc; }
    }
    bdd B;
    if (!bdd_init(&B, u->budget->limit[TR_BUDGET_CIRCUIT])) { free(E.e); free(E.c); free(pos); free(val); return TR_MATH_MEMORY; }
    int8_t cs[TR_EQUIV_MAX_CONDITIONS];
    lit path[TR_EQUIV_MAX_CONDITIONS + 1];
    sctx s;
    memset(&s, 0, sizeof s);
    s.u = u; s.q = q; s.C = C; s.o = o; s.refusal = refusal; s.E = &E;
    int theory = 0;
    int32_t rc = TR_MATH_OK;
    uint64_t npat = (uint64_t)1 << m;
    for (uint64_t pat = 0; pat < npat && !s.stop && !rc; pat++) {
        if (++u->iters > u->budget->limit[TR_BUDGET_ITERATIONS]) {
            unknown_need(o, TR_BUDGET_ITERATIONS, u->iters, u->budget->limit[TR_BUDGET_ITERATIONS]); s.stop = S_UNKNOWN; break;
        }
        uint32_t nullmask = 0;
        for (int32_t i = 0; i < m; i++) if ((pat >> i) & 1) nullmask |= UINT32_C(1) << nl[i];
        s.nullmask = nullmask;
        for (int32_t c = 0; c < S->ncond; c++) cs[c] = cond_status(&S->conds[c], nullmask);
        o->v.patterns++;
        if (!cone_tf(&B, S, cone, n, cs, pos, val)) {
            if (B.nomem) rc = TR_MATH_MEMORY;
            else { unknown_need(o, TR_BUDGET_CIRCUIT, (uint64_t)B.count + 1, B.limit); s.stop = S_UNKNOWN; }
            break;
        }
        tf A = pa >= 0 ? val[pa] : (tf){1, 0}, Bv = pb >= 0 ? val[pb] : (tf){1, 0};
        int32_t diff;
        if (q->kind == TR_EQUIV_PREDICATE && q->semantics == TR_EQUIV_KLEENE)
            diff = bdd_apply(&B, B_OR, bdd_apply(&B, B_XOR, A.t, Bv.t), bdd_apply(&B, B_XOR, A.f, Bv.f));
        else if (q->relation == TR_EQUIV_IMPLIES) diff = bdd_apply(&B, B_AND, A.t, bdd_apply(&B, B_XOR, Bv.t, 1));
        else diff = bdd_apply(&B, B_XOR, A.t, Bv.t);
        int32_t M = 0;
        if (diff >= 0 && q->kind == TR_EQUIV_QUERY && d->ma >= 0) M = bdd_apply(&B, B_AND, A.t, Bv.t);
        if (diff < 0 || M < 0) {
            if (B.nomem) rc = TR_MATH_MEMORY;
            else { unknown_need(o, TR_BUDGET_CIRCUIT, (uint64_t)B.count + 1, B.limit); s.stop = S_UNKNOWN; }
            break;
        }
        if (diff) { theory = 1; s.grid = 0; cubes_of(&s, &B, diff, path, 0); if (s.stop) break; }
        if (q->kind == TR_EQUIV_QUERY && d->ma >= 0 && M && !pzero) {
            int na = (S->ms[d->ma].cols & nullmask) != 0, nb = (S->ms[d->mb].cols & nullmask) != 0;
            if (na != nb && q->nulls == TR_EQUIV_SQL) { theory = 1; s.grid = 0; cubes_of(&s, &B, M, path, 0); }
            else if (!na && !nb) { theory = 1; s.grid = 1; cubes_of(&s, &B, M, path, 0); }
        }
    }
    o->v.nodes = B.count;
    o->v.work += s.cubes;
    bdd_free(&B);
    free(E.e); free(E.c); free(pos); free(val);
    if (rc) return rc;
    if (s.stop < 0) return s.stop;
    if (s.stop == S_FOUND) { o->v.verdict = TR_EQUIV_REFUTED; o->v.why = TR_EQUIV_WHY_NONE; return TR_MATH_OK; }
    if (s.stop == S_UNKNOWN) return TR_MATH_OK;            /* o is UNKNOWN with the reason */
    if (s.undecided) { unknown(o, TR_EQUIV_WHY_UNSUPPORTED, NULL); return TR_MATH_OK; }
    o->v.verdict = TR_EQUIV_PROVEN;
    o->v.label = !theory && allnull ? TR_EQUIV_LABEL_UNIVERSAL : TR_EQUIV_LABEL_DOMAIN;
    return TR_MATH_OK;
}
static int32_t domain_check(const tr_equiv_domain *D) {
    if (!D || D->ncols < 0 || D->ncols > NCOLS || (D->ncols && !D->cols) ||
        (D->mode != TR_EQUIV_ENUMERATE && D->mode != TR_EQUIV_DECIDE)) return TR_MATH_INVALID;
    for (int32_t i = 0; i < D->ncols; i++) {
        const tr_equiv_column *c = &D->cols[i];
        if (c->col < 0 || c->col >= NCOLS || c->scale < 0 || c->scale > TR_SCOPE_MAX_SCALE || (c->nullable != 0 && c->nullable != 1))
            return TR_MATH_INVALID;
        for (int32_t j = 0; j < i; j++) if (D->cols[j].col == c->col) return TR_MATH_INVALID;
        if (c->kind == TR_EQUIV_RANGE) { if (c->lo > c->hi || c->nvalues || c->values) return TR_MATH_INVALID; }
        else if (c->kind == TR_EQUIV_VALUES) {
            if (c->nvalues < 1 || !c->values || c->lo || c->hi) return TR_MATH_INVALID;
            for (uint64_t k = 1; k < c->nvalues; k++) if (c->values[k - 1] >= c->values[k]) return TR_MATH_INVALID;
        } else return TR_MATH_INVALID;
    }
    return TR_MATH_OK;
}
tr_equiv_result *tr_equiv_decide(const tr_equiv_set *S, const tr_equiv_question *qs, int32_t nq, const tr_equiv_domain *D,
                                 const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    if (!set_valid(S) || nq < 0 || (nq && !qs) || domain_check(D)) { set_error(error, TR_MATH_INVALID); return NULL; }
    if (nq > TR_EQUIV_MAX_QUESTIONS) { set_error(error, TR_MATH_UNSUPPORTED); return NULL; }
    uctx u;
    memset(&u, 0, sizeof u);
    tr_math_budget store;
    u.S = S; u.D = D; u.budget = budget_or(budget, &store);
    for (int32_t c = 0; c < NCOLS; c++) { u.dcol[c] = -1; u.scale[c] = 0; }
    for (int32_t i = 0; i < D->ncols; i++) { u.dcol[D->cols[i].col] = i; u.scale[D->cols[i].col] = D->cols[i].scale; }
    u.ncols = S->maxcol > 0 ? S->maxcol : 1;
    uint64_t nvals = 0;
    for (int32_t i = 0; i < D->ncols; i++) nvals = tr_budget_add(nvals, D->cols[i].nvalues);
    for (int32_t q = 0; q < nq; q++) {
        int32_t rc = question_check(S, &qs[q]);
        if (rc) { set_error(error, rc); return NULL; }
        sides d;
        question_sides(S, &qs[q], &d);
        uint32_t C = sides_cols(S, &d);
        for (int32_t c = 0; c < NCOLS; c++) if (((C >> c) & 1) && u.dcol[c] < 0) { set_error(error, TR_MATH_INVALID); return NULL; }
    }
    if (tr_budget_check_one(u.budget, TR_BUDGET_BYTES, tr_budget_mul(nvals, 8), refusal)) { set_error(error, TR_MATH_BUDGET); return NULL; }
    tr_equiv_result *R = result_alloc(nq, u.budget, refusal, error);
    if (!R) return NULL;
    R->ndom = D->ncols;
    R->dvalues = (int64_t *)zalloc(nvals * 8);
    if (!R->dvalues) { tr_equiv_result_free(R); set_error(error, TR_MATH_MEMORY); return NULL; }
    uint64_t at = 0;
    for (int32_t i = 0; i < D->ncols; i++) {
        R->dom[i] = D->cols[i];
        if (D->cols[i].kind == TR_EQUIV_VALUES) {
            memcpy(R->dvalues + at, D->cols[i].values, (size_t)(D->cols[i].nvalues * 8));
            R->dom[i].values = R->dvalues + at;
            at += D->cols[i].nvalues;
        }
    }
    int32_t rc = TR_MATH_OK;
    for (int32_t q = 0; q < nq && !rc; q++) {
        rq *o = &R->q[q];
        sides d;
        question_sides(S, &qs[q], &d);
        uint32_t C = sides_cols(S, &d);
        o->colmask = C;
        u.iters = 0;
        tr_math_refusal r2;
        tr_budget_none(&r2);
        if (D->mode == TR_EQUIV_ENUMERATE) rc = decide_enumerate(&u, &qs[q], C, o, &r2);
        else if (qs[q].kind == TR_EQUIV_MEASURE) rc = decide_measure(&u, &qs[q], &d, C, o, &r2);
        else rc = decide_logic(&u, &qs[q], &d, C, o, &r2);
        if (o->v.verdict != TR_EQUIV_UNKNOWN) o->v.why = TR_EQUIV_WHY_NONE;
        o->v.witness = TR_EQUIV_NONE;
    }
    if (rc) { tr_equiv_result_free(R); set_error(error, rc); return NULL; }
    set_error(error, TR_MATH_OK);
    return R;
}

/* ================================================================ reading results and authorizing rewrites */
int32_t tr_equiv_result_count(const tr_equiv_result *r) { return result_valid(r) ? r->nq : TR_MATH_INVALID; }
int32_t tr_equiv_result_get(const tr_equiv_result *r, int32_t q, tr_equiv_verdict *out, tr_equiv_evidence *evidence) {
    if (!result_valid(r) || q < 0 || q >= r->nq) return TR_MATH_INVALID;
    if (out) *out = r->q[q].v;
    if (evidence) *evidence = r->q[q].ev;
    return TR_MATH_OK;
}
int32_t tr_equiv_result_point(const tr_equiv_result *r, int32_t q, int64_t *values, uint8_t *present, int32_t capacity) {
    if (!result_valid(r) || q < 0 || q >= r->nq || capacity < 0 || (capacity && (!values || !present))) return TR_MATH_INVALID;
    const rq *o = &r->q[q];
    if (!o->haspoint) return TR_MATH_NO_SOLUTION;
    for (int32_t c = 0; c < capacity; c++) {
        values[c] = c < o->ncols ? o->values[c] : 0;
        present[c] = c < o->ncols ? o->present[c] : 1;
    }
    return TR_MATH_OK;
}
const tr_cert *tr_equiv_result_certificate(const tr_equiv_result *r, int32_t q) {
    return result_valid(r) && q >= 0 && q < r->nq ? r->q[q].cert : NULL;
}
static int in_domain(const tr_equiv_column *d, int64_t x) {
    if (d->kind == TR_EQUIV_RANGE) return x >= d->lo && x <= d->hi;
    uint64_t lo = 0, hi = d->nvalues;
    while (lo < hi) { uint64_t mid = lo + (hi - lo) / 2; if (d->values[mid] < x) lo = mid + 1; else hi = mid; }
    return lo < d->nvalues && d->values[lo] == x;
}
int32_t tr_equiv_authorize(const tr_equiv_result *r, int32_t q, const tr_scope *current, tr_equiv_decision *out) {
    if (!result_valid(r) || q < 0 || q >= r->nq || !out) return TR_MATH_INVALID;
    const rq *o = &r->q[q];
    memset(out, 0, sizeof *out);
    out->label = o->v.label; out->verdict = o->v.verdict;
    out->stale.column = -1; out->stale.dim = -1; out->cert.premise = -1;
    if (o->v.verdict != TR_EQUIV_PROVEN) { out->reason = TR_EQUIV_AUTH_NOT_PROVEN; return TR_MATH_OK; }
    if (o->v.label == TR_EQUIV_LABEL_UNIVERSAL) { out->authorized = 1; return TR_MATH_OK; }
    if (!current) { out->reason = TR_EQUIV_AUTH_NO_SCOPE; return TR_MATH_OK; }
    if (o->v.label == TR_EQUIV_LABEL_SNAPSHOT) {
        int32_t f = r->record ? tr_cert_record_fresh(r->record, current, NULL, 0, TR_CERT_MATCH_EXACT, &out->stale) : TR_MATH_STALE;
        if (f) { out->reason = TR_EQUIV_AUTH_STALE; return TR_MATH_OK; }
        if (o->cert) {
            int32_t e = tr_cert_authorize_exact(o->cert, 0, current, NULL, 0, &out->cert);
            if (e && e != TR_MATH_STALE) return e;
            if (e || !out->cert.authorized) { out->reason = TR_EQUIV_AUTH_STALE; return TR_MATH_OK; }
        }
        out->authorized = 1;
        return TR_MATH_OK;
    }
    if (o->v.label == TR_EQUIV_LABEL_RELATIONS) {
        int32_t e = tr_cert_authorize_exact(o->cert, 0, current, NULL, 0, &out->cert);
        if (e && e != TR_MATH_STALE) return e;
        out->authorized = !e && out->cert.authorized;
        if (!out->authorized) out->reason = TR_EQUIV_AUTH_RELATIONS;
        return TR_MATH_OK;
    }
    tr_scope_view v;                                      /* DOMAIN: every selected value of the columns read inside */
    if (tr_scope_get_view(current, &v)) return TR_MATH_INVALID;
    for (int32_t c = 0; c < NCOLS; c++) {
        if (!((o->colmask >> c) & 1)) continue;
        const tr_equiv_column *d = NULL;
        for (int32_t i = 0; i < r->ndom; i++) if (r->dom[i].col == c) d = &r->dom[i];
        if (!d || c >= v.ncols || v.col[c].scale != d->scale) { out->reason = TR_EQUIV_AUTH_OUTSIDE; return TR_MATH_OK; }
        const uint64_t nw = (v.rows + 63) / 64;
        for (uint64_t w = 0; w < nw; w++) {
            uint64_t bits = v.selection[w];
            while (bits) {
                uint64_t row = w * 64 + (uint64_t)__builtin_ctzll(bits);
                bits &= bits - 1;
                if (!tr_scope_present(&v, c, row)) { if (!d->nullable) { out->reason = TR_EQUIV_AUTH_OUTSIDE; return TR_MATH_OK; } continue; }
                if (!in_domain(d, tr_scope_value(&v, c, row))) { out->reason = TR_EQUIV_AUTH_OUTSIDE; return TR_MATH_OK; }
            }
        }
    }
    out->authorized = 1;
    return TR_MATH_OK;
}

/* ================================================================ families */
/* the integer forms D f of distinct measures over a scope: GEOM-001 candidates (pair (f, f), F = 3 j), constants
 * directly */
typedef struct {
    int32_t n, ngeom;
    int32_t *geom;                  /* measure slot -> GEOM pair (-1: a constant) */
    int64_t *constant;              /* D f of a constant measure */
    tr_geom_pairs *pairs;
    tr_ident_cand_view cand;
} mforms;
static void mforms_free(mforms *F) { free(F->geom); free(F->constant); tr_geom_pairs_free(F->pairs); memset(F, 0, sizeof *F); }
static int32_t mforms_make(const tr_equiv_set *S, const int32_t *ids, int32_t n, const tr_scope *scope,
                           const tr_math_budget *budget, tr_math_refusal *refusal, mforms *F) {
    memset(F, 0, sizeof *F);
    F->n = n;
    F->geom = (int32_t *)zalloc((uint64_t)n * 4); F->constant = (int64_t *)zalloc((uint64_t)n * 8);
    tr_geom_poly *p = (tr_geom_poly *)zalloc((uint64_t)n * sizeof *p);
    if (!F->geom || !F->constant || !p) { free(p); mforms_free(F); return TR_MATH_MEMORY; }
    for (int32_t i = 0; i < n; i++) {
        const emeasure *m = &S->ms[ids[i]];
        F->geom[i] = -1;
        if (m->ncols) { measure_view(m, &p[F->ngeom]); F->geom[i] = F->ngeom++; }
        else F->constant[i] = m->nterms ? m->num[0] * (m->D / m->den[0]) : 0;
    }
    empty_cand(&F->cand);
    int32_t rc = TR_MATH_OK;
    if (F->ngeom) {
        if (F->ngeom > TR_GEOM_MAX_PAIRS) rc = TR_MATH_UNSUPPORTED;
        else {
            int32_t e = 0;
            F->pairs = tr_geom_pairs_make(scope, p, p, F->ngeom, budget, refusal, &e);
            if (!F->pairs) rc = e ? e : TR_MATH_MEMORY;
            else { tr_geom_pair_view pv; tr_geom_pairs_get_view(F->pairs, &pv); F->cand = pv.cand; }
        }
    }
    for (int32_t i = 0; !rc && i < n; i++) {
        int32_t W = F->geom[i] >= 0 ? F->cand.words[3 * F->geom[i]] : 2;
        if (W + 1 > TR_IDENT_MAX_WORDS) rc = TR_MATH_UNSUPPORTED;    /* the sum needs one more word */
    }
    free(p);
    if (rc) mforms_free(F);
    return rc;
}
static int32_t mforms_words(const mforms *F, int32_t i) { return F->geom[i] >= 0 ? F->cand.words[3 * F->geom[i]] : 2; }
/* D f at a row where its columns are present, sign-extended into W + 1 words and added to sum */
static void mforms_add(const mforms *F, const tr_scope_view *v, int32_t i, uint64_t row, uint64_t *sum) {
    int32_t W = mforms_words(F, i);
    uint64_t acc[TR_IDENT_MAX_WORDS];
    if (F->geom[i] >= 0) tr_ident_cand_value(v, &F->cand, 3 * F->geom[i], row, W, acc);
    else { acc[0] = (uint64_t)F->constant[i]; acc[1] = F->constant[i] < 0 ? ~(uint64_t)0 : 0; }
    acc[W] = (acc[W - 1] >> 63) ? ~(uint64_t)0 : 0;
    tr_ident_w_add(sum, acc, W + 1);
}
/* the nodes a root reaches, increasing (dynamic) */
static int32_t cone_all(const tr_equiv_set *S, int32_t root, int32_t *stamp, int32_t tag, int32_t *stack, int32_t *out) {
    if (root < 0) return 0;
    int32_t n = 0, sp = 0;
    stamp[root] = tag; stack[sp++] = root;
    while (sp) {
        int32_t i = stack[--sp];
        out[n++] = i;
        const tr_bool_node *m = &S->nodes[i];
        int32_t ops[2] = {m->op == TR_BOOL_NOT || m->op >= TR_BOOL_AND ? m->a : -1, m->op >= TR_BOOL_AND ? m->b : -1};
        for (int j = 0; j < 2; j++) if (ops[j] >= 0 && stamp[ops[j]] != tag) { stamp[ops[j]] = tag; stack[sp++] = ops[j]; }
    }
    for (int32_t i = 1; i < n; i++) for (int32_t j = i; j > 0 && out[j - 1] > out[j]; j--) { int32_t t = out[j]; out[j] = out[j - 1]; out[j - 1] = t; }
    return n;
}
/* node words at w: val[2 i], val[2 i + 1] for the listed nodes (increasing); var: condition -> mask index */
static void eval_nodes(const tr_equiv_set *S, const int32_t *list, int32_t n, const tr_bool_domain *d, const int32_t *var,
                       uint64_t w, uint64_t *val) {
    for (int32_t k = 0; k < n; k++) {
        int32_t i = list[k];
        const tr_bool_node *m = &S->nodes[i];
        uint64_t x, y;
        if (m->op == TR_BOOL_CONST) tr_bool_const_word(m->a, &x, &y);
        else if (m->op == TR_BOOL_VAR) tr_bool_var_word(d, var ? var[m->a] : m->a, w, &x, &y);
        else tr_bool_apply(m->op, val[2 * (uint64_t)m->a], val[2 * (uint64_t)m->a + 1], m->b >= 0 ? val[2 * (uint64_t)m->b] : 0,
                           m->b >= 0 ? val[2 * (uint64_t)m->b + 1] : 0, &x, &y);
        val[2 * (uint64_t)i] = x; val[2 * (uint64_t)i + 1] = y;
    }
}
static void answer_init(const tr_equiv_set *S, int32_t j, const mforms *F, int32_t slot, tr_equiv_answer *a) {
    memset(a, 0, sizeof *a);
    a->filter = S->qs[j].filter; a->measure = S->qs[j].measure; a->pair = j; a->denominator = 1;
    if (a->measure >= 0) { a->denominator = S->ms[a->measure].D; a->words = mforms_words(F, slot) + 1; }
}
static int measure_present(const tr_scope_view *v, uint32_t cols, uint64_t row) { return !tr_equiv_null(v, cols, row); }
static int32_t run_shared(const tr_equiv_set *S, const tr_scope *scope, const tr_scope_view *v, const tr_math_budget *budget,
                          tr_equiv_answer *out, tr_equiv_run_stats *st, tr_math_refusal *refusal) {
    const int32_t nq = S->nquery;
    tr_math_need need;
    need_zero(&need);
    need.need[TR_BUDGET_STATES] = (uint64_t)nq;
    need.need[TR_BUDGET_OUTPUT] = (uint64_t)S->gquery;
    uint64_t bytes = tr_budget_mul((uint64_t)S->nnodes, 16 + 12);
    bytes = tr_budget_add(bytes, tr_budget_mul((uint64_t)nq, sizeof(tr_equiv_answer) + 24));
    need.need[TR_BUDGET_BYTES] = tr_budget_add(bytes, tr_budget_mul((uint64_t)S->nmeasure, 16));
    tr_math_budget store;
    int32_t rc = tr_budget_check(budget_or(budget, &store), &need, refusal);
    if (rc) return rc;
    uint64_t *val = (uint64_t *)zalloc((uint64_t)S->nnodes * 16);
    int32_t *stamp = (int32_t *)zalloc((uint64_t)S->nnodes * 4), *stack = (int32_t *)zalloc((uint64_t)S->nnodes * 4);
    int32_t *list = (int32_t *)zalloc((uint64_t)S->nnodes * 4), *tmp = (int32_t *)zalloc((uint64_t)S->nnodes * 4);
    int32_t *slot = (int32_t *)zalloc((uint64_t)S->nmeasure * 4), *ids = (int32_t *)zalloc((uint64_t)S->nmeasure * 4);
    int32_t *moff = (int32_t *)zalloc(((uint64_t)S->nmeasure + 1) * 4), *mlist = (int32_t *)zalloc((uint64_t)nq * 4);
    int32_t *fill = (int32_t *)zalloc((uint64_t)S->nmeasure * 4);
    tr_equiv_answer *acc = (tr_equiv_answer *)zalloc((uint64_t)nq * sizeof *acc);
    uint64_t *T = (uint64_t *)zalloc((uint64_t)nq * 8);
    uint8_t *mark = (uint8_t *)zalloc((uint64_t)S->nnodes);
    tr_bool_masks *masks = NULL;
    mforms F;
    memset(&F, 0, sizeof F);
    if (!val || !stamp || !stack || !list || !tmp || !slot || !ids || !moff || !mlist || !fill || !acc || !T || !mark) rc = TR_MATH_MEMORY;
    int32_t nlist = 0, nm = 0;
    for (int32_t j = 0; j < nq && !rc; j++) {                 /* the nodes the filters reach, once */
        int32_t n = cone_all(S, S->qs[j].filter, stamp, j + 1, stack, tmp);
        for (int32_t i = 0; i < n; i++) if (!mark[tmp[i]]) { mark[tmp[i]] = 1; nlist++; }
    }
    for (int32_t i = 0, k = 0; !rc && i < S->nnodes; i++) if (mark[i]) list[k++] = i;
    for (int32_t i = 0; !rc && i < S->nmeasure; i++) slot[i] = -1;
    for (int32_t j = 0; j < nq && !rc; j++) {
        int32_t m = S->qs[j].measure;
        if (m >= 0 && slot[m] < 0) { slot[m] = nm; ids[nm++] = m; }
    }
    if (!rc) {                                                /* per measure slot its queries */
        for (int32_t j = 0; j < nq; j++) if (S->qs[j].measure >= 0) moff[slot[S->qs[j].measure] + 1]++;
        for (int32_t i = 0; i < nm; i++) moff[i + 1] += moff[i];
        for (int32_t i = 0; i < nm; i++) fill[i] = moff[i];
        for (int32_t j = 0; j < nq; j++) if (S->qs[j].measure >= 0) mlist[fill[slot[S->qs[j].measure]]++] = j;
    }
    if (!rc && S->ncond) {
        int32_t e = 0;
        masks = tr_bool_masks_build(scope, S->conds, S->ncond, budget, refusal, &e);
        if (!masks) rc = e ? e : TR_MATH_MEMORY;
    }
    if (!rc && nm) rc = mforms_make(S, ids, nm, scope, budget, refusal, &F);
    if (!rc) {
        tr_bool_domain d;
        memset(&d, 0, sizeof d);
        if (masks) tr_bool_masks_get_domain(masks, &d);
        for (int32_t j = 0; j < nq; j++) answer_init(S, j, &F, S->qs[j].measure >= 0 ? slot[S->qs[j].measure] : 0, &acc[j]);
        const uint64_t nw = (v->rows + 63) / 64;
        for (uint64_t w = 0; w < nw; w++) {
            const uint64_t sel = v->selection[w];
            if (!sel) continue;
            eval_nodes(S, list, nlist, &d, NULL, w, val);
            st->node_words += (uint64_t)nlist;
            for (int32_t j = 0; j < nq; j++) {
                int32_t f = S->qs[j].filter;
                T[j] = (f >= 0 ? val[2 * (uint64_t)f] & val[2 * (uint64_t)f + 1] : ~(uint64_t)0) & sel;
                acc[j].rows += (uint64_t)tr_bool_popc(T[j]);
            }
            for (int32_t i = 0; i < nm; i++) {
                uint64_t U = 0;
                for (int32_t k = moff[i]; k < moff[i + 1]; k++) U |= T[mlist[k]];
                const uint32_t cols = S->ms[ids[i]].cols;
                while (U) {
                    int b = tr_bool_ctz(U);
                    U &= U - 1;
                    uint64_t row = w * 64 + (uint64_t)b, bit = (uint64_t)1 << b;
                    if (!measure_present(v, cols, row)) continue;
                    uint64_t one[TR_IDENT_MAX_WORDS] = {0};
                    mforms_add(&F, v, i, row, one);               /* the value once */
                    st->measure_rows++;
                    int32_t W1 = mforms_words(&F, i) + 1;
                    for (int32_t k = moff[i]; k < moff[i + 1]; k++) {
                        int32_t j = mlist[k];
                        if (!(T[j] & bit)) continue;
                        acc[j].nonnull++;
                        tr_ident_w_add(acc[j].sum, one, W1);
                    }
                }
            }
        }
        st->scans = 1;
        st->condition_rows = tr_budget_mul((uint64_t)S->ncond, v->selected);
        for (int32_t i = 0; i < S->gquery; i++) out[i] = acc[S->qid[i]];
        st->filters = S->info.filters; st->measures = (uint64_t)nm; st->pairs = (uint64_t)nq;
        st->bytes = need.need[TR_BUDGET_BYTES];
    }
    tr_bool_masks_free(masks);
    mforms_free(&F);
    free(val); free(stamp); free(stack); free(list); free(tmp); free(slot); free(ids); free(moff); free(mlist); free(fill); free(acc);
    free(T); free(mark);
    return rc;
}
static int32_t run_separate(const tr_equiv_set *S, const tr_scope *scope, const tr_scope_view *v, const tr_math_budget *budget,
                            tr_equiv_answer *out, tr_equiv_run_stats *st, tr_math_refusal *refusal) {
    tr_math_need need;
    need_zero(&need);
    need.need[TR_BUDGET_STATES] = 1;
    need.need[TR_BUDGET_OUTPUT] = (uint64_t)S->gquery;
    need.need[TR_BUDGET_BYTES] = tr_budget_add(tr_budget_mul((uint64_t)S->nnodes, 16 + 16), TR_EQUIV_MAX_CONDITIONS * (sizeof(tr_bool_condition) + 4));
    tr_math_budget store;
    int32_t rc = tr_budget_check(budget_or(budget, &store), &need, refusal);
    if (rc) return rc;
    uint64_t *val = (uint64_t *)zalloc((uint64_t)S->nnodes * 16);
    int32_t *stamp = (int32_t *)zalloc((uint64_t)S->nnodes * 4), *stack = (int32_t *)zalloc((uint64_t)S->nnodes * 4);
    int32_t *list = (int32_t *)zalloc((uint64_t)S->nnodes * 4);
    if (!val || !stamp || !stack || !list) rc = TR_MATH_MEMORY;
    const uint64_t nw = (v->rows + 63) / 64;
    for (int32_t i = 0; i < S->gquery && !rc; i++) {          /* every query alone: its own conditions, nodes, measure */
        const int32_t j = S->qid[i], f = S->qs[j].filter, m = S->qs[j].measure;
        int32_t n = cone_all(S, f, stamp, i + 1, stack, list);
        tr_bool_condition conds[TR_EQUIV_MAX_CONDITIONS];
        int32_t var[TR_EQUIV_MAX_CONDITIONS], k = 0;
        for (int32_t c = 0; c < S->ncond; c++) var[c] = -1;
        for (int32_t x = 0; x < n; x++) {
            const tr_bool_node *nd = &S->nodes[list[x]];
            if (nd->op == TR_BOOL_VAR && var[nd->a] < 0) { var[nd->a] = k; conds[k++] = S->conds[nd->a]; }
        }
        tr_bool_masks *masks = NULL;
        mforms F;
        memset(&F, 0, sizeof F);
        if (k) {
            int32_t e = 0;
            masks = tr_bool_masks_build(scope, conds, k, budget, refusal, &e);
            if (!masks) { rc = e ? e : TR_MATH_MEMORY; break; }
        }
        if (m >= 0) { rc = mforms_make(S, &m, 1, scope, budget, refusal, &F); if (rc) { tr_bool_masks_free(masks); break; } }
        tr_bool_domain d;
        memset(&d, 0, sizeof d);
        if (masks) tr_bool_masks_get_domain(masks, &d);
        tr_equiv_answer a;
        answer_init(S, j, &F, 0, &a);
        const uint32_t cols = m >= 0 ? S->ms[m].cols : 0;
        const int32_t W1 = m >= 0 ? mforms_words(&F, 0) + 1 : 0;
        for (uint64_t w = 0; w < nw; w++) {
            const uint64_t sel = v->selection[w];
            if (!sel) continue;
            eval_nodes(S, list, n, &d, var, w, val);
            st->node_words += (uint64_t)n;
            uint64_t Tw = (f >= 0 ? val[2 * (uint64_t)f] & val[2 * (uint64_t)f + 1] : ~(uint64_t)0) & sel;
            a.rows += (uint64_t)tr_bool_popc(Tw);
            if (m < 0) continue;
            while (Tw) {
                int b = tr_bool_ctz(Tw);
                Tw &= Tw - 1;
                uint64_t row = w * 64 + (uint64_t)b;
                if (!measure_present(v, cols, row)) continue;
                uint64_t one[TR_IDENT_MAX_WORDS] = {0};
                mforms_add(&F, v, 0, row, one);
                tr_ident_w_add(a.sum, one, W1);
                a.nonnull++;
                st->measure_rows++;
            }
        }
        out[i] = a;
        st->scans++;
        st->condition_rows = tr_budget_add(st->condition_rows, tr_budget_mul((uint64_t)k, v->selected));
        tr_bool_masks_free(masks);
        mforms_free(&F);
    }
    st->filters = S->info.filters; st->measures = S->info.measures; st->pairs = (uint64_t)S->gquery;
    st->bytes = need.need[TR_BUDGET_BYTES];
    free(val); free(stamp); free(stack); free(list);
    return rc;
}
int32_t tr_equiv_family_run(const tr_equiv_set *S, const tr_scope *scope, int32_t mode, const tr_math_budget *budget,
                            tr_equiv_answer *out, tr_equiv_run_stats *stats, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    tr_scope_view v;
    if (!set_valid(S) || !scope || !stats || (S->gquery && !out) || tr_scope_get_view(scope, &v) || S->maxcol > v.ncols ||
        (mode != TR_EQUIV_SHARED && mode != TR_EQUIV_SEPARATE)) return TR_MATH_INVALID;
    memset(stats, 0, sizeof *stats);
    stats->queries = (uint64_t)S->gquery;
    return mode == TR_EQUIV_SHARED ? run_shared(S, scope, &v, budget, out, stats, refusal)
                                   : run_separate(S, scope, &v, budget, out, stats, refusal);
}

/* ================================================================ the relationship optimizer */
typedef struct { int32_t a, b; __int128 c; } lterm;
static int32_t lt_add(lterm *t, int32_t *n, int32_t cap, int32_t a, int32_t b, __int128 c) {
    if (a > b) { int32_t x = a; a = b; b = x; }
    if (a < 0 && b >= 0) { a = b; b = -1; }               /* linear: (a, -1); constant: (-1, -1) */
    for (int32_t i = 0; i < *n; i++)
        if (t[i].a == a && t[i].b == b) { if (__builtin_add_overflow(t[i].c, c, &t[i].c)) return TR_MATH_OVERFLOW; return TR_MATH_OK; }
    if (*n >= cap) return TR_MATH_UNSUPPORTED;
    t[*n].a = a; t[*n].b = b; t[*n].c = c; (*n)++;
    return TR_MATH_OK;
}
static int catalog_offset(const tr_catalog *cat, int32_t stored, int64_t *offset) {
    for (int32_t k = 0; k < cat->ncols && k < TR_MAX_COLUMNS; k++) if (cat->columns[k].col == stored) { *offset = cat->columns[k].offset; return 1; }
    return 0;
}
static int32_t mul3(__int128 a, __int128 b, __int128 c, __int128 *r) {
    __int128 x;
    if (__builtin_mul_overflow(a, b, &x) || __builtin_mul_overflow(x, c, r)) return TR_MATH_OVERFLOW;
    return TR_MATH_OK;
}
int32_t tr_equiv_rewrite(struct tr_catalog_s *cat, const struct atable_s *table, const uint64_t *used, const tr_geom_poly *measure,
                         const int32_t *stored, int32_t nscope, tr_equiv_rewrite_info *info, uint8_t *exponents,
                         int64_t *coefficients, uint64_t capacity) {
    if (!cat || !table || !measure || !stored || nscope < 1 || nscope > NCOLS || !info) return TR_MATH_INVALID;
    memset(info, 0, sizeof *info);
    emeasure m;
    eterm *tt = (eterm *)zalloc((measure->nterms ? measure->nterms : 1) * sizeof *tt);
    if (!tt) return TR_MATH_MEMORY;
    int32_t rc = measure_canon(measure, tt, &m);
    free(tt);
    if (rc) return rc;
    lterm L[4 * TR_MAX_TERMS + 4];
    int32_t nl = 0;
    for (uint64_t k = 0; k < m.nterms && !rc; k++) {      /* logical x = code u + offset o: lowered to codes */
        if (m.den[k] != 1 || m.degree > 2) { rc = TR_MATH_UNSUPPORTED; break; }
        int32_t s[2] = {-1, -1}, ns = 0;
        int64_t o[2] = {0, 0};
        for (int32_t i = 0; i < m.ncols && !rc; i++) {
            if (m.col[i] >= nscope) { rc = TR_MATH_INVALID; break; }
            for (uint8_t r = 0; r < m.e[k * (uint64_t)m.ncols + (uint64_t)i]; r++) {
                s[ns] = stored[m.col[i]];
                if (!catalog_offset(cat, s[ns], &o[ns])) { rc = TR_MATH_UNSUPPORTED; break; }
                ns++;
            }
        }
        if (rc) break;
        __int128 c = m.num[k], t;
        if (ns == 0) rc = lt_add(L, &nl, 4 * TR_MAX_TERMS + 4, -1, -1, c);
        else if (ns == 1) {
            rc = lt_add(L, &nl, 4 * TR_MAX_TERMS + 4, s[0], -1, c);
            if (!rc) rc = mul3(c, o[0], 1, &t);
            if (!rc) rc = lt_add(L, &nl, 4 * TR_MAX_TERMS + 4, -1, -1, t);
        } else {
            rc = lt_add(L, &nl, 4 * TR_MAX_TERMS + 4, s[0], s[1], c);
            if (!rc) rc = mul3(c, o[1], 1, &t);
            if (!rc) rc = lt_add(L, &nl, 4 * TR_MAX_TERMS + 4, s[0], -1, t);
            if (!rc) rc = mul3(c, o[0], 1, &t);
            if (!rc) rc = lt_add(L, &nl, 4 * TR_MAX_TERMS + 4, s[1], -1, t);
            if (!rc) rc = mul3(c, o[0], o[1], &t);
            if (!rc) rc = lt_add(L, &nl, 4 * TR_MAX_TERMS + 4, -1, -1, t);
        }
    }
    measure_free(&m);
    if (rc) return rc;
    tr_poly in, out;
    memset(&in, 0, sizeof in);
    for (int32_t i = 0; i < nl; i++) {
        if (!L[i].c) continue;
        if (L[i].c < INT64_MIN || L[i].c > INT64_MAX) return TR_MATH_OVERFLOW;
        if (in.n >= TR_MAX_TERMS) return TR_MATH_UNSUPPORTED;
        in.terms[in.n].a = L[i].a; in.terms[in.n].b = L[i].b; in.terms[in.n].coefficient = (int64_t)L[i].c; in.n++;
    }
    tr_trace trace;
    int32_t o = tr_optimize(cat, table, used, &in, &out, &trace);
    if (o == TR_STALE) return TR_MATH_STALE;
    if (o == TR_OVERFLOW) return TR_MATH_OVERFLOW;
    if (o == TR_MEMORY) return TR_MATH_MEMORY;
    if (o != TR_OK) return TR_MATH_INVALID;
    info->changed = trace.changed; info->steps = trace.steps; info->explored = trace.explored; info->truncated = trace.truncated;
    info->before_cost = trace.before_cost; info->after_cost = trace.after_cost;
    nl = 0;
    for (int32_t i = 0; i < out.n && !rc; i++) {          /* lifted back: u = x - o */
        const tr_term *q = &out.terms[i];
        __int128 c = q->coefficient, t;
        int64_t oa = 0, ob = 0;
        if (q->a >= 0 && !catalog_offset(cat, q->a, &oa)) rc = TR_MATH_UNSUPPORTED;
        if (!rc && q->b >= 0 && !catalog_offset(cat, q->b, &ob)) rc = TR_MATH_UNSUPPORTED;
        if (rc) break;
        if (q->a < 0) rc = lt_add(L, &nl, 4 * TR_MAX_TERMS + 4, -1, -1, c);
        else if (q->b < 0) {
            rc = lt_add(L, &nl, 4 * TR_MAX_TERMS + 4, q->a, -1, c);
            if (!rc) rc = mul3(-c, oa, 1, &t);
            if (!rc) rc = lt_add(L, &nl, 4 * TR_MAX_TERMS + 4, -1, -1, t);
        } else {
            rc = lt_add(L, &nl, 4 * TR_MAX_TERMS + 4, q->a, q->b, c);
            if (!rc) rc = mul3(-c, ob, 1, &t);
            if (!rc) rc = lt_add(L, &nl, 4 * TR_MAX_TERMS + 4, q->a, -1, t);
            if (!rc) rc = mul3(-c, oa, 1, &t);
            if (!rc) rc = lt_add(L, &nl, 4 * TR_MAX_TERMS + 4, q->b, -1, t);
            if (!rc) rc = mul3(c, oa, ob, &t);
            if (!rc) rc = lt_add(L, &nl, 4 * TR_MAX_TERMS + 4, -1, -1, t);
        }
    }
    if (rc) return rc;
    int32_t col[MAXV], nc = 0;                             /* stored columns back to scope columns */
    int32_t sc[4 * TR_MAX_TERMS + 4][2];
    uint64_t nt = 0;
    for (int32_t i = 0; i < nl; i++) {
        if (!L[i].c) continue;
        if (L[i].c < INT64_MIN || L[i].c > INT64_MAX) return TR_MATH_OVERFLOW;
        int32_t st[2] = {L[i].a, L[i].b};
        for (int j = 0; j < 2; j++) {
            sc[nt][j] = -1;
            if (st[j] < 0) continue;
            int32_t x = -1;
            for (int32_t c = 0; c < nscope && x < 0; c++) if (stored[c] == st[j]) x = c;
            if (x < 0) return TR_MATH_UNSUPPORTED;               /* a column the scope does not carry */
            sc[nt][j] = x;
            int seen = 0;
            for (int32_t k = 0; k < nc; k++) seen |= col[k] == x;
            if (!seen) { if (nc >= MAXV) return TR_MATH_UNSUPPORTED; col[nc++] = x; }
        }
        L[nt].c = L[i].c;
        nt++;
    }
    for (int32_t i = 1; i < nc; i++) for (int32_t j = i; j > 0 && col[j - 1] > col[j]; j--) { int32_t t = col[j]; col[j] = col[j - 1]; col[j - 1] = t; }
    info->ncols = nc;
    for (int32_t i = 0; i < MAXV; i++) info->col[i] = i < nc ? col[i] : -1;
    info->nterms = nt;
    if (nt > capacity || (nt && (!exponents || !coefficients))) return TR_MATH_INVALID;
    for (uint64_t t = 0; t < nt; t++) {
        for (int32_t k = 0; k < nc; k++) {
            uint8_t e = 0;
            for (int j = 0; j < 2; j++) e = (uint8_t)(e + (sc[t][j] == col[k]));
            exponents[t * (uint64_t)nc + (uint64_t)k] = e;
        }
        coefficients[t] = (int64_t)L[t].c;
    }
    return TR_MATH_OK;
}
