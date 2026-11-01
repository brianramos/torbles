/* ATTR-002: bounded explanation and repair search; see math_repair.h.
 * SPDX-License-Identifier: Apache-2.0 */
#include "math_repair.h"
#include "math_carriers.h"
#include "math_domain.h"
#include "math_handles.h"
#include <stdlib.h>
#include <string.h>

TR_MATH_ABI(tr_repair_view, 1);         /* the ctypes boundary of ATTR-002 (math_handles.h) */
TR_MATH_ABI(tr_repair_out, 1);
TR_MATH_ABI(tr_repair_query, 1);
TR_MATH_ABI(tr_repair_plan, 1);
TR_MATH_ABI(tr_repair_info, 1);
TR_MATH_ABI(tr_repair_problem_info, 1);
TR_MATH_ABI(tr_repair_removal, 1);

_Static_assert(sizeof(tr_repair_view) == 472, "tr_repair_view ABI");
_Static_assert(sizeof(tr_repair_out) == 32, "tr_repair_out ABI");
_Static_assert(sizeof(tr_repair_query) == 40, "tr_repair_query ABI");
_Static_assert(sizeof(tr_repair_plan) == 72, "tr_repair_plan ABI");
_Static_assert(sizeof(tr_repair_info) == 120, "tr_repair_info ABI");
_Static_assert(sizeof(tr_repair_problem_info) == 104, "tr_repair_problem_info ABI");
_Static_assert(sizeof(tr_repair_removal) == 56, "tr_repair_removal ABI");

#define PROBLEM_MAGIC UINT32_C(0x52504552)
#define NONE TR_REPAIR_NONE

static void set_error(int32_t *error, int32_t code) { if (error) *error = code; }
static void *zalloc(uint64_t bytes) {
    if (bytes > (uint64_t)SIZE_MAX / 2) return NULL;
    return calloc(1, bytes ? (size_t)bytes : 1);
}
static const tr_math_budget *budget_or(const tr_math_budget *b, tr_math_budget *dflt) {
    if (b) return b;
    tr_budget_default(dflt);
    return dflt;
}
static void need_zero(tr_math_need *n) { for (int i = 0; i < TR_BUDGET_RESOURCES; i++) n->need[i] = 0; }
static int bitw(const uint64_t *w, uint64_t i) { return (int)((w[i >> 6] >> (i & 63)) & 1); }

struct tr_repair_problem_s {
    uint32_t magic;
    tr_repair_view view;
    tr_repair_problem_info info;
    uint64_t *masks;                                   /* HIT */
    uint64_t *sel, *val, *vld, *target;                /* CIRCUIT: the domain's words, f's TRUE words */
    tr_bool_node *nodes;
    uint64_t *deps;
    uint32_t *sched;
    tr_math_i128 *table;                               /* GAME copies */
    tr_prov_node *pnode;
    uint32_t *pchild, *pout, *var_player;
    int64_t *weights;
    uint8_t *qual, *vvalid;                            /* DELETE */
    int64_t *vval;
    uint64_t *rows;
};
static int problem_ok(const tr_repair_problem *p) { return p && p->magic == PROBLEM_MAGIC; }
void tr_repair_problem_free(tr_repair_problem *p) {
    if (!p || p->magic != PROBLEM_MAGIC) return;
    p->magic = 0;
    free(p->masks); free(p->sel); free(p->val); free(p->vld); free(p->target); free(p->nodes); free(p->deps);
    free(p->sched); free(p->table); free(p->pnode); free(p->pchild); free(p->pout); free(p->var_player);
    free(p->weights); free(p->qual); free(p->vvalid); free(p->vval); free(p->rows);
    free(p);
}
static tr_repair_problem *problem_new(int32_t kind, int32_t source, int32_t n) {
    tr_repair_problem *p = zalloc(sizeof *p);
    if (!p) return NULL;
    p->magic = PROBLEM_MAGIC;
    p->view.kind = kind; p->view.n = n;
    p->info.kind = kind; p->info.source = source; p->info.n = n;
    p->info.witness_a = NONE; p->info.witness_b = NONE;
    return p;
}
static void finish_info(tr_repair_problem *p) {
    p->info.allowed = p->view.allowed;
    p->info.nmasks = p->view.nmasks;
    p->info.monotone = p->view.monotone;
}

/* ================================================================== hitting sets */
static int u64_cmp(const void *a, const void *b) {
    uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
    return x < y ? -1 : x > y;
}
static int pop_cmp(const void *a, const void *b) {
    uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
    int px = __builtin_popcountll(x), py = __builtin_popcountll(y);
    if (px != py) return px < py ? -1 : 1;
    return x < y ? -1 : x > y;
}
/* the inclusion-minimal masks of a list (sorted by size, then value; duplicates removed): *nout; in place */
static uint64_t minimal_masks(uint64_t *m, uint64_t n) {
    if (!n) return 0;
    qsort(m, (size_t)n, 8, pop_cmp);
    uint64_t k = 0;
    for (uint64_t i = 0; i < n; i++) {
        if (k && m[k - 1] == m[i]) continue;
        int dominated = 0;
        for (uint64_t j = 0; j < k && !dominated; j++) if ((m[j] & m[i]) == m[j]) dominated = 1;
        if (!dominated) m[k++] = m[i];
    }
    return k;
}
static tr_repair_problem *hit_problem(int32_t n, uint64_t *masks, uint64_t nmasks, uint64_t allowed, int32_t source,
                                      const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error) {
    tr_math_need need;
    need_zero(&need);
    need.need[TR_BUDGET_VARIABLES] = (uint64_t)n;
    need.need[TR_BUDGET_SUPPORT] = nmasks;
    need.need[TR_BUDGET_BYTES] = tr_budget_add(sizeof(tr_repair_problem), tr_budget_mul(nmasks, 8));
    int32_t rc = tr_budget_check(budget, &need, refusal);
    if (rc) { free(masks); set_error(error, rc); return NULL; }
    tr_repair_problem *p = problem_new(TR_REPAIR_HIT, source, n);
    if (!p) { free(masks); set_error(error, TR_MATH_MEMORY); return NULL; }
    p->masks = masks;
    p->view.masks = masks; p->view.nmasks = nmasks; p->view.allowed = allowed; p->view.monotone = 1;
    for (uint64_t i = 0; i < nmasks; i++) if (!(masks[i] & allowed)) p->info.infeasible = 1;
    p->info.bytes = need.need[TR_BUDGET_BYTES];
    finish_info(p);
    set_error(error, TR_MATH_OK);
    return p;
}
tr_repair_problem *tr_repair_masks(int32_t n, uint64_t nmasks, const uint64_t *masks, uint64_t allowed,
                                   const tr_math_budget *budget, tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    tr_math_budget dflt;
    budget = budget_or(budget, &dflt);
    if (n < 1 || (nmasks && !masks)) { set_error(error, TR_MATH_INVALID); return NULL; }
    if (n > TR_REPAIR_MAX_ELEMENTS) { set_error(error, TR_MATH_UNSUPPORTED); return NULL; }
    const uint64_t all = tr_repair_mask_all(n);
    if (!allowed) allowed = all;
    if (allowed & ~all) { set_error(error, TR_MATH_INVALID); return NULL; }
    for (uint64_t i = 0; i < nmasks; i++) if (masks[i] & ~all) { set_error(error, TR_MATH_INVALID); return NULL; }
    tr_math_need need;
    need_zero(&need);
    need.need[TR_BUDGET_VARIABLES] = (uint64_t)n;
    need.need[TR_BUDGET_SUPPORT] = nmasks;
    int32_t rc = tr_budget_check(budget, &need, refusal);
    if (rc) { set_error(error, rc); return NULL; }
    uint64_t *m = zalloc((nmasks ? nmasks : 1) * 8);
    if (!m) { set_error(error, TR_MATH_MEMORY); return NULL; }
    memcpy(m, masks, (size_t)nmasks * 8);
    uint64_t k = minimal_masks(m, nmasks);
    return hit_problem(n, m, k, allowed, TR_REPAIR_SRC_MASKS, budget, refusal, error);
}

/* the patterns of the points of a word set: (value bits, valid bits) over the k conditions, with a representative row */
typedef struct { uint64_t v, k, row; } pat_t;
static int pat_cmp(const void *a, const void *b) {
    const pat_t *x = (const pat_t *)a, *y = (const pat_t *)b;
    if (x->v != y->v) return x->v < y->v ? -1 : 1;
    if (x->k != y->k) return x->k < y->k ? -1 : 1;
    return x->row < y->row ? -1 : x->row > y->row;
}
static uint64_t words_popcount(const uint64_t *w, uint64_t nw) {
    uint64_t n = 0;
    for (uint64_t i = 0; i < nw; i++) n += (uint64_t)__builtin_popcountll(w[i]);
    return n;
}
/* distinct patterns of the points in `words` (selected by the domain); SELECTED convention: TRUE bits only */
static pat_t *patterns(const tr_bool_domain *d, const uint64_t *words, int32_t convention, uint64_t *count,
                       int32_t *rc) {
    const uint64_t npts = words_popcount(words, d->nw);
    pat_t *p = zalloc((npts ? npts : 1) * sizeof *p);
    if (!p) { *rc = TR_MATH_MEMORY; return NULL; }
    uint64_t n = 0;
    for (uint64_t w = 0; w < d->nw; w++) {
        uint64_t bits = words[w] & tr_bool_selection_word(d, w);
        if (!bits) continue;
        uint64_t vw[TR_BOOL_MAX_CONDITIONS], kw[TR_BOOL_MAX_CONDITIONS];
        for (int32_t i = 0; i < d->k; i++) tr_bool_var_word(d, i, w, &vw[i], &kw[i]);
        while (bits) {
            const int b = __builtin_ctzll(bits);
            bits &= bits - 1;
            uint64_t v = 0, k = 0;
            for (int32_t i = 0; i < d->k; i++) {
                if ((vw[i] >> b) & 1) v |= (uint64_t)1 << i;
                if ((kw[i] >> b) & 1) k |= (uint64_t)1 << i;
            }
            v &= k;                                                  /* value implies valid */
            if (convention == TR_REPAIR_SELECTED) k = 0;             /* only TRUE / not TRUE counts */
            p[n].v = v; p[n].k = k; p[n].row = w * 64 + (uint64_t)b; n++;
        }
    }
    qsort(p, (size_t)n, sizeof *p, pat_cmp);
    uint64_t u = 0;
    for (uint64_t i = 0; i < n; i++) if (!u || p[u - 1].v != p[i].v || p[u - 1].k != p[i].k) p[u++] = p[i];
    *count = u;
    return p;
}
static tr_repair_problem *separate(const tr_bool_domain *d, const uint64_t *a_words, const uint64_t *b_words,
                                   int32_t convention, uint64_t allowed, int32_t source, const tr_math_budget *budget,
                                   tr_math_refusal *refusal, int32_t *error) {
    int32_t rc = tr_bool_domain_check(d);
    if (rc) { set_error(error, rc); return NULL; }
    if (!a_words || !b_words || (convention != TR_REPAIR_VALUES && convention != TR_REPAIR_SELECTED) || d->k < 1) {
        set_error(error, TR_MATH_INVALID); return NULL;
    }
    if (d->k > TR_REPAIR_MAX_ELEMENTS) { set_error(error, TR_MATH_UNSUPPORTED); return NULL; }
    const int32_t n = d->k;
    const uint64_t all = tr_repair_mask_all(n);
    if (!allowed) allowed = all;
    if (allowed & ~all) { set_error(error, TR_MATH_INVALID); return NULL; }
    tr_math_need need;
    need_zero(&need);
    need.need[TR_BUDGET_VARIABLES] = (uint64_t)n;
    const uint64_t na = words_popcount(a_words, d->nw), nb = words_popcount(b_words, d->nw);
    need.need[TR_BUDGET_BYTES] = tr_budget_mul(tr_budget_add(na, nb), sizeof(pat_t));
    if ((rc = tr_budget_check(budget, &need, refusal))) { set_error(error, rc); return NULL; }
    uint64_t ua = 0, ub = 0;
    pat_t *pa = patterns(d, a_words, convention, &ua, &rc), *pb = rc ? NULL : patterns(d, b_words, convention, &ub, &rc);
    if (rc) { free(pa); free(pb); set_error(error, rc); return NULL; }
    need.need[TR_BUDGET_TENSOR] = tr_budget_mul(ua, ub);                  /* the pairs compared */
    need.need[TR_BUDGET_BYTES] = tr_budget_add(need.need[TR_BUDGET_BYTES], tr_budget_mul(tr_budget_mul(ua, ub), 8));
    if ((rc = tr_budget_check(budget, &need, refusal))) { free(pa); free(pb); set_error(error, rc); return NULL; }
    uint64_t *m = zalloc((ua && ub ? ua * ub : 1) * 8);
    if (!m) { free(pa); free(pb); set_error(error, TR_MATH_MEMORY); return NULL; }
    uint64_t nm = 0, wa = NONE, wb = NONE;
    for (uint64_t i = 0; i < ua; i++)
        for (uint64_t j = 0; j < ub; j++) {
            uint64_t diff = convention == TR_REPAIR_SELECTED ? (pa[i].v ^ pb[j].v) : ((pa[i].v ^ pb[j].v) | (pa[i].k ^ pb[j].k));
            diff &= allowed;
            if (!diff && wa == NONE) { wa = pa[i].row; wb = pb[j].row; }
            m[nm++] = diff;
        }
    uint64_t k = minimal_masks(m, nm);
    tr_repair_problem *p = hit_problem(n, m, k, allowed, source, budget, refusal, error);
    if (p) {
        p->info.convention = convention; p->info.patterns_a = ua; p->info.patterns_b = ub; p->info.pairs = ua * ub;
        p->info.witness_a = wa; p->info.witness_b = wb; p->info.points = d->selected;
        p->info.selected = na;
        if (wa != NONE) p->info.infeasible = 1;
    }
    free(pa); free(pb);
    return p;
}
tr_repair_problem *tr_repair_separate(const tr_bool_domain *d, const uint64_t *a_words, const uint64_t *b_words,
                                      int32_t convention, uint64_t allowed, const tr_math_budget *budget,
                                      tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    tr_math_budget dflt;
    budget = budget_or(budget, &dflt);
    return separate(d, a_words, b_words, convention, allowed, TR_REPAIR_SRC_SEPARATE, budget, refusal, error);
}
/* the TRUE words of a program over a domain (owned) */
static uint64_t *program_true(const tr_bool_domain *d, const tr_bool_program_view *pv, int32_t *rc) {
    uint64_t *out = zalloc(d->nw * 8 + 8), *val = zalloc((uint64_t)(pv->nnodes > 0 ? pv->nnodes : 1) * 16);
    if (!out || !val) { free(out); free(val); *rc = TR_MATH_MEMORY; return NULL; }
    for (uint64_t w = 0; w < d->nw; w++) {
        const uint64_t sel = tr_bool_selection_word(d, w);
        if (!sel) continue;
        tr_bool_eval_base(pv, d, w, val);
        out[w] = val[2 * (uint64_t)pv->root] & val[2 * (uint64_t)pv->root + 1] & sel;
    }
    free(val);
    return out;
}
static int32_t program_against(const tr_bool_domain *d, const tr_bool_program_view *pv) {
    int32_t rc = tr_bool_domain_check(d);
    if (!rc) rc = tr_bool_program_check(pv);
    if (rc) return rc;
    if (pv->k != d->k) return TR_MATH_DOMAIN_ERROR;
    return 0;
}
tr_repair_problem *tr_repair_explain(const tr_bool_domain *d, const tr_bool_program_view *pv, int32_t convention,
                                     uint64_t allowed, const tr_math_budget *budget, tr_math_refusal *refusal,
                                     int32_t *error) {
    tr_budget_none(refusal);
    tr_math_budget dflt;
    budget = budget_or(budget, &dflt);
    int32_t rc = program_against(d, pv);
    if (rc) { set_error(error, rc); return NULL; }
    uint64_t *a = program_true(d, pv, &rc), *b = rc ? NULL : zalloc(d->nw * 8 + 8);
    if (rc || !b) { free(a); free(b); set_error(error, rc ? rc : TR_MATH_MEMORY); return NULL; }
    for (uint64_t w = 0; w < d->nw; w++) b[w] = tr_bool_selection_word(d, w) & ~a[w];
    tr_repair_problem *p = separate(d, a, b, convention, allowed, TR_REPAIR_SRC_EXPLAIN, budget, refusal, error);
    free(a); free(b);
    return p;
}
tr_repair_problem *tr_repair_keep(const tr_bool_domain *d, uint64_t conjunction, const tr_math_budget *budget,
                                  tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    tr_math_budget dflt;
    budget = budget_or(budget, &dflt);
    int32_t rc = tr_bool_domain_check(d);
    if (rc) { set_error(error, rc); return NULL; }
    if (d->k < 1) { set_error(error, TR_MATH_INVALID); return NULL; }
    if (d->k > TR_REPAIR_MAX_ELEMENTS) { set_error(error, TR_MATH_UNSUPPORTED); return NULL; }
    const int32_t n = d->k;
    const uint64_t C = conjunction;
    if (C & ~tr_repair_mask_all(n)) { set_error(error, TR_MATH_INVALID); return NULL; }
    tr_math_need need;
    need_zero(&need);
    need.need[TR_BUDGET_VARIABLES] = (uint64_t)n;
    need.need[TR_BUDGET_BYTES] = tr_budget_mul(d->selected, 8);
    if ((rc = tr_budget_check(budget, &need, refusal))) { set_error(error, rc); return NULL; }
    /* the failure sets {i in C : c_i not TRUE} of every point the conjunction rejects */
    uint64_t *m = zalloc((d->selected ? d->selected : 1) * 8), nm = 0, sel_true = 0;
    if (!m) { set_error(error, TR_MATH_MEMORY); return NULL; }
    for (uint64_t w = 0; w < d->nw; w++) {
        const uint64_t sel = tr_bool_selection_word(d, w);
        if (!sel) continue;
        uint64_t tw[TR_BOOL_MAX_CONDITIONS], all_true = sel;
        for (int32_t i = 0; i < n; i++) {
            uint64_t v, k;
            tr_bool_var_word(d, i, w, &v, &k);
            tw[i] = v & k & sel;
            if ((C >> i) & 1) all_true &= tw[i];
        }
        sel_true += (uint64_t)__builtin_popcountll(all_true);
        uint64_t rej = sel & ~all_true;
        while (rej) {
            const int b = __builtin_ctzll(rej);
            rej &= rej - 1;
            uint64_t f = 0;
            for (uint64_t c = C; c; c &= c - 1) { int i = __builtin_ctzll(c); if (!((tw[i] >> b) & 1)) f |= (uint64_t)1 << i; }
            m[nm++] = f;
        }
    }
    qsort(m, (size_t)nm, 8, u64_cmp);
    uint64_t u = 0;
    for (uint64_t i = 0; i < nm; i++) if (!u || m[u - 1] != m[i]) m[u++] = m[i];
    const uint64_t distinct = u;
    uint64_t k = minimal_masks(m, u);
    tr_repair_problem *p = hit_problem(n, m, k, C, TR_REPAIR_SRC_KEEP, budget, refusal, error);
    if (p) { p->info.points = d->selected; p->info.selected = sel_true; p->info.patterns_b = distinct; }
    return p;
}
tr_repair_problem *tr_repair_program(const tr_bool_domain *d, const tr_bool_program_view *pv, uint64_t baseline,
                                     uint64_t allowed, const tr_math_budget *budget, tr_math_refusal *refusal,
                                     int32_t *error) {
    tr_budget_none(refusal);
    tr_math_budget dflt;
    budget = budget_or(budget, &dflt);
    int32_t rc = program_against(d, pv);
    if (rc) { set_error(error, rc); return NULL; }
    if (d->k > TR_REPAIR_MAX_ELEMENTS) { set_error(error, TR_MATH_UNSUPPORTED); return NULL; }
    const int32_t n = d->k;
    const uint64_t all = tr_repair_mask_all(n), support = pv->deps[pv->root] & all;
    if (!allowed) allowed = support;
    if (allowed & ~all) { set_error(error, TR_MATH_INVALID); return NULL; }
    tr_math_need need;
    need_zero(&need);
    need.need[TR_BUDGET_VARIABLES] = (uint64_t)n;
    need.need[TR_BUDGET_CIRCUIT] = (uint64_t)pv->nnodes;
    const uint64_t words = d->kind == TR_BOOL_ROWS ? tr_budget_mul(d->nw, tr_budget_add(2, tr_budget_mul((uint64_t)n, d->valid ? 2 : 1))) : d->nw;
    need.need[TR_BUDGET_BYTES] = tr_budget_add(tr_budget_add(sizeof(tr_repair_problem), tr_budget_mul(words, 8)),
                                               tr_budget_mul((uint64_t)pv->nnodes, 16 + 8 + 4 + 16));
    if ((rc = tr_budget_check(budget, &need, refusal))) { set_error(error, rc); return NULL; }
    tr_repair_problem *p = problem_new(TR_REPAIR_CIRCUIT, TR_REPAIR_SRC_PROGRAM, n);
    if (!p) { set_error(error, TR_MATH_MEMORY); return NULL; }
    tr_repair_view *v = &p->view;
    v->dom = *d;
    if (d->kind == TR_BOOL_ROWS) {
        p->sel = zalloc(d->nw * 8 + 8); p->val = zalloc((uint64_t)n * d->nw * 8 + 8);
        p->vld = d->valid ? zalloc((uint64_t)n * d->nw * 8 + 8) : NULL;
        if (!p->sel || !p->val || (d->valid && !p->vld)) rc = TR_MATH_MEMORY;
        else {
            memcpy(p->sel, d->selection, (size_t)d->nw * 8);
            memcpy(p->val, d->value, (size_t)(n * d->nw) * 8);
            if (d->valid) memcpy(p->vld, d->valid, (size_t)(n * d->nw) * 8);
            v->dom.selection = p->sel; v->dom.value = p->val; v->dom.valid = p->vld;
        }
    }
    p->nodes = zalloc((uint64_t)pv->nnodes * sizeof(tr_bool_node) + 16); p->deps = zalloc((uint64_t)pv->nnodes * 8 + 8);
    p->sched = zalloc((uint64_t)pv->nsched * 4 + 4);
    if (!p->nodes || !p->deps || !p->sched) rc = TR_MATH_MEMORY;
    if (!rc) {
        memcpy(p->nodes, pv->nodes, (size_t)pv->nnodes * sizeof(tr_bool_node));
        memcpy(p->deps, pv->deps, (size_t)pv->nnodes * 8);
        memcpy(p->sched, pv->schedule, (size_t)pv->nsched * 4);
        v->prog = *pv; v->prog.nodes = p->nodes; v->prog.deps = p->deps; v->prog.schedule = p->sched;
        p->target = program_true(&v->dom, &v->prog, &rc);
        v->target = p->target;
    }
    if (rc) { tr_repair_problem_free(p); set_error(error, rc); return NULL; }
    v->baseline = baseline & all; v->allowed = allowed; v->monotone = 0;
    p->info.points = d->selected; p->info.selected = words_popcount(p->target, d->nw);
    p->info.bytes = need.need[TR_BUDGET_BYTES];
    finish_info(p);
    set_error(error, TR_MATH_OK);
    return p;
}
static void game_monotone(tr_repair_view *v) {
    const int up = v->game.monotone > 0, down = v->game.monotone < 0;
    const int ge = v->op == TR_REPAIR_GE || v->op == TR_REPAIR_GT, le = v->op == TR_REPAIR_LE || v->op == TR_REPAIR_LT;
    v->monotone = v->explain ? ((up && ge) || (down && le)) : ((up && le) || (down && ge));
}
tr_repair_problem *tr_repair_game(const tr_attr_game *game, int32_t kind, int32_t op, tr_math_i128 threshold,
                                  uint64_t allowed, const tr_math_budget *budget, tr_math_refusal *refusal,
                                  int32_t *error) {
    tr_budget_none(refusal);
    tr_math_budget dflt;
    budget = budget_or(budget, &dflt);
    tr_attr_game_view g;
    if (!game || tr_attr_game_get_view(game, &g) || (kind != TR_REPAIR_EXPLAIN && kind != TR_REPAIR_REPAIR) ||
        op < TR_REPAIR_EQ || op > TR_REPAIR_GE) { set_error(error, TR_MATH_INVALID); return NULL; }
    int32_t rc = tr_attr_view_check(&g, 1);
    if (rc) { set_error(error, rc); return NULL; }
    if (g.n > TR_REPAIR_MAX_ELEMENTS) { set_error(error, TR_MATH_UNSUPPORTED); return NULL; }
    const uint64_t all = tr_repair_mask_all(g.n);
    if (!allowed) allowed = all;
    if (allowed & ~all) { set_error(error, TR_MATH_INVALID); return NULL; }
    tr_math_need need;
    need_zero(&need);
    need.need[TR_BUDGET_VARIABLES] = (uint64_t)g.n;
    need.need[TR_BUDGET_BYTES] = tr_budget_add(sizeof(tr_repair_problem), tr_attr_game_device_bytes(&g));
    if ((rc = tr_budget_check(budget, &need, refusal))) { set_error(error, rc); return NULL; }
    tr_repair_problem *p = problem_new(TR_REPAIR_GAME, TR_REPAIR_SRC_GAME, g.n);
    if (!p) { set_error(error, TR_MATH_MEMORY); return NULL; }
    tr_repair_view *v = &p->view;
    v->game = g;
    if (g.kind == TR_ATTR_GAME_TABLE) {
        const uint64_t cnt = (uint64_t)1 << g.n;
        p->table = zalloc(cnt * 16);
        if (!p->table) rc = TR_MATH_MEMORY;
        else { memcpy(p->table, g.table, (size_t)cnt * 16); v->game.table = p->table; }
    } else {
        const tr_prov_plan_view *pl = &g.plan;
        p->pnode = zalloc(pl->nodes * sizeof(tr_prov_node) + 16); p->pchild = zalloc(pl->nchild * 4 + 4);
        p->pout = zalloc(pl->noutputs * 4 + 4); p->weights = zalloc(pl->noutputs * 8 + 8); p->var_player = zalloc(g.nvars * 4 + 4);
        if (!p->pnode || !p->pchild || !p->pout || !p->weights || !p->var_player) rc = TR_MATH_MEMORY;
        else {
            memcpy(p->pnode, pl->node, (size_t)pl->nodes * sizeof(tr_prov_node));
            memcpy(p->pchild, pl->child, (size_t)pl->nchild * 4);
            memcpy(p->pout, pl->output, (size_t)pl->noutputs * 4);
            memcpy(p->weights, g.weights, (size_t)pl->noutputs * 8);
            memcpy(p->var_player, g.var_player, (size_t)g.nvars * 4);
            v->game.plan.node = p->pnode; v->game.plan.child = p->pchild; v->game.plan.output = p->pout;
            v->game.weights = p->weights; v->game.var_player = p->var_player;
        }
    }
    if (rc) { tr_repair_problem_free(p); set_error(error, rc); return NULL; }
    v->explain = kind == TR_REPAIR_EXPLAIN; v->op = op; v->threshold = threshold; v->allowed = allowed;
    game_monotone(v);
    p->info.bytes = need.need[TR_BUDGET_BYTES];
    finish_info(p);
    set_error(error, TR_MATH_OK);
    return p;
}
tr_repair_problem *tr_repair_delete(const tr_scope *scope, const tr_bool_domain *d, const tr_bool_program_view *pv,
                                    int32_t value, int32_t agg, int32_t op, const tr_bigq *t, int32_t n,
                                    const uint64_t *rows, const tr_math_budget *budget, tr_math_refusal *refusal,
                                    int32_t *error) {
    tr_budget_none(refusal);
    tr_math_budget dflt;
    budget = budget_or(budget, &dflt);
    tr_scope_view sv;
    if (!scope || tr_scope_get_view(scope, &sv) || !d || d->kind != TR_BOOL_ROWS || d->points != sv.rows || !t ||
        !tr_bigq_is_handle(t) || n < 1 || !rows || agg < TR_REPAIR_COUNT_STAR || agg > TR_REPAIR_AVG ||
        op < TR_REPAIR_EQ || op > TR_REPAIR_GE || (agg != TR_REPAIR_COUNT_STAR && (value < 0 || value >= sv.ncols)) ||
        (agg == TR_REPAIR_COUNT_STAR && value >= sv.ncols)) { set_error(error, TR_MATH_INVALID); return NULL; }
    if (n > TR_REPAIR_MAX_ELEMENTS) { set_error(error, TR_MATH_UNSUPPORTED); return NULL; }
    int32_t rc = tr_bool_domain_check(d);
    if (!rc && pv) rc = program_against(d, pv);
    if (rc) { set_error(error, rc); return NULL; }
    for (uint64_t w = 0; w < d->nw; w++) if (d->selection[w] != sv.selection[w]) { set_error(error, TR_MATH_DOMAIN_ERROR); return NULL; }
    for (int32_t i = 0; i < n; i++) {
        if (rows[i] >= sv.rows || !tr_scope_is_selected(&sv, rows[i])) { set_error(error, TR_MATH_INVALID); return NULL; }
        for (int32_t j = 0; j < i; j++) if (rows[j] == rows[i]) { set_error(error, TR_MATH_INVALID); return NULL; }
    }
    tr_num_rational tq;
    if ((rc = tr_bigq_to_q64(t, &tq))) { set_error(error, TR_MATH_OVERFLOW); return NULL; }
    tr_math_need need;
    need_zero(&need);
    need.need[TR_BUDGET_VARIABLES] = (uint64_t)n;
    need.need[TR_BUDGET_BYTES] = tr_budget_add(tr_budget_add(sizeof(tr_repair_problem), (uint64_t)n * 26),
                                               tr_budget_add(tr_budget_mul(d->nw, 8), tr_budget_mul((uint64_t)(pv ? pv->nnodes : 0), 16)));
    if ((rc = tr_budget_check(budget, &need, refusal))) { set_error(error, rc); return NULL; }
    uint64_t *qw = NULL;
    if (pv) { qw = program_true(d, pv, &rc); if (rc) { set_error(error, rc); return NULL; } }
    tr_repair_problem *p = problem_new(TR_REPAIR_DELETE, TR_REPAIR_SRC_DELETE, n);
    if (!p) { free(qw); set_error(error, TR_MATH_MEMORY); return NULL; }
    p->qual = zalloc((uint64_t)n); p->vvalid = zalloc((uint64_t)n); p->vval = zalloc((uint64_t)n * 8); p->rows = zalloc((uint64_t)n * 8);
    if (!p->qual || !p->vvalid || !p->vval || !p->rows) rc = TR_MATH_MEMORY;
    uint8_t *cand = zalloc(sv.rows + 1);
    if (!cand) rc = TR_MATH_MEMORY;
    unsigned __int128 mag = 0;
    const unsigned __int128 lim = (unsigned __int128)1 << 124;
    uint64_t bc = 0, bv = 0;
    __int128 bs = 0;
    int nonneg = 1, nonpos = 1;
    for (int32_t i = 0; !rc && i < n; i++) cand[rows[i]] = 1;
    for (uint64_t r = 0; !rc && r < sv.rows; r++) {                  /* every selected qualifying row */
        if (!tr_scope_is_selected(&sv, r)) continue;
        if (qw && !bitw(qw, r)) continue;
        const int has = value >= 0 && tr_scope_present(&sv, value, r);
        const int64_t x = has ? tr_scope_value(&sv, value, r) : 0;
        if (has) { mag += tr_repair_uabs((__int128)x); if (mag >= lim) rc = TR_MATH_OVERFLOW; }
        if (cand[r]) continue;
        bc++;
        if (has) { bv++; bs += x; }
    }
    for (int32_t i = 0; !rc && i < n; i++) {
        const uint64_t r = rows[i];
        p->rows[i] = r;
        p->qual[i] = (uint8_t)(!qw || bitw(qw, r));
        const int has = value >= 0 && tr_scope_present(&sv, value, r);
        p->vvalid[i] = (uint8_t)has;
        p->vval[i] = has ? tr_scope_value(&sv, value, r) : 0;
        if (p->qual[i] && has) { if (p->vval[i] < 0) nonneg = 0; if (p->vval[i] > 0) nonpos = 0; }
    }
    /* the exact comparisons and the bounds stay within 128-bit products: |sum| td and |tnum| (rows + 1) below 2^125 */
    const int32_t sc = value >= 0 ? sv.col[value].scale : 0;
    __int128 tn = tq.numerator;
    if (!rc && (agg == TR_REPAIR_SUM || agg == TR_REPAIR_AVG)) for (int32_t q = 0; q < sc; q++) tn *= 10;
    if (!rc) {
        uint64_t prod[4];
        tr_repair_umul(mag + 1, (unsigned __int128)tq.denominator, prod);
        if (prod[2] || prod[3] || (prod[1] >> 60)) rc = TR_MATH_OVERFLOW;
        tr_repair_umul(tr_repair_uabs(tn) + 1, (unsigned __int128)(sv.selected + 1), prod);
        if (prod[2] || prod[3] || (prod[1] >> 60)) rc = TR_MATH_OVERFLOW;
    }
    free(cand); free(qw);
    if (rc) { tr_repair_problem_free(p); set_error(error, rc); return NULL; }
    tr_repair_view *v = &p->view;
    v->agg = agg; v->op = op; v->vscale = sc;
    v->qual = p->qual; v->vvalid = p->vvalid; v->vval = p->vval;
    v->base_count = bc; v->base_vcount = bv; v->base_sum = tr_i128_set(bs);
    v->tnum = tr_i128_set(tn); v->tden = tq.denominator;
    v->allowed = tr_repair_mask_all(n);
    const int le = op == TR_REPAIR_LE || op == TR_REPAIR_LT, ge = op == TR_REPAIR_GE || op == TR_REPAIR_GT;
    v->monotone = ((agg == TR_REPAIR_COUNT_STAR || agg == TR_REPAIR_COUNT) && le) ||
                  (agg == TR_REPAIR_SUM && ((le && nonneg) || (ge && nonpos)));
    p->info.points = sv.rows; p->info.selected = sv.selected;
    p->info.bytes = need.need[TR_BUDGET_BYTES];
    finish_info(p);
    set_error(error, TR_MATH_OK);
    return p;
}
int32_t tr_repair_problem_get_info(const tr_repair_problem *p, tr_repair_problem_info *out) {
    if (!problem_ok(p) || !out) return TR_MATH_INVALID;
    *out = p->info;
    return 0;
}
int32_t tr_repair_problem_get_view(const tr_repair_problem *p, tr_repair_view *out) {
    if (!problem_ok(p) || !out) return TR_MATH_INVALID;
    *out = p->view;
    return 0;
}
int32_t tr_repair_problem_masks(const tr_repair_problem *p, uint64_t *out, uint64_t capacity) {
    if (!problem_ok(p) || p->view.kind != TR_REPAIR_HIT || capacity < p->view.nmasks || (!out && p->view.nmasks))
        return TR_MATH_INVALID;
    memcpy(out, p->view.masks, (size_t)p->view.nmasks * 8);
    return 0;
}

/* ================================================================== the CPU evaluators */
typedef struct { const tr_repair_view *v; uint64_t *scratch; } cpu_eval;
static int32_t cpu_batch(void *ctx, const uint64_t *masks, uint64_t count, tr_repair_out *out) {
    cpu_eval *e = (cpu_eval *)ctx;
    for (uint64_t i = 0; i < count; i++) tr_repair_eval_one(e->v, masks[i], e->scratch, &out[i]);
    return 0;
}
static int32_t cpu_scan(void *ctx, const uint64_t *masks, const int64_t *costs, uint64_t count, int64_t match,
                        uint64_t *feasible, uint64_t *ties, int64_t *mincost) {
    cpu_eval *e = (cpu_eval *)ctx;
    const uint64_t words = (count + 63) / 64;
    memset(feasible, 0, (size_t)words * 8); memset(ties, 0, (size_t)words * 8);
    int64_t mn = INT64_MAX;
    tr_repair_out o;
    for (uint64_t i = 0; i < count; i++) {
        tr_repair_eval_one(e->v, masks[i], e->scratch, &o);
        if (!o.feasible) continue;
        feasible[i >> 6] |= (uint64_t)1 << (i & 63);
        if (costs[i] < mn) mn = costs[i];
    }
    const int64_t m = match == INT64_MIN ? mn : match;
    for (uint64_t i = 0; i < count; i++)
        if (((feasible[i >> 6] >> (i & 63)) & 1) && costs[i] == m) ties[i >> 6] |= (uint64_t)1 << (i & 63);
    *mincost = mn;
    return 0;
}
static int32_t cpu_open(const tr_repair_view *v, cpu_eval *e) {
    e->v = v;
    e->scratch = zalloc(tr_repair_scratch_words(v) * 8 + 16);
    return e->scratch ? 0 : TR_MATH_MEMORY;
}
int32_t tr_repair_evaluate(const tr_repair_view *v, const uint64_t *masks, uint64_t count, tr_repair_out *out,
                           const tr_math_budget *budget, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    int32_t rc = tr_repair_view_check(v);
    if (rc) return rc;
    if (count && (!masks || !out)) return TR_MATH_INVALID;
    tr_math_budget dflt;
    budget = budget_or(budget, &dflt);
    tr_math_need need;
    need_zero(&need);
    need.need[TR_BUDGET_OUTPUT] = count;
    need.need[TR_BUDGET_BYTES] = tr_budget_mul(tr_repair_scratch_words(v), 8);
    if ((rc = tr_budget_check(budget, &need, refusal))) return rc;
    cpu_eval e;
    if ((rc = cpu_open(v, &e))) return rc;
    cpu_batch(&e, masks, count, out);
    free(e.scratch);
    return 0;
}
int32_t tr_repair_scan(const tr_repair_view *v, const uint64_t *masks, const int64_t *costs, uint64_t count,
                       int64_t match, uint64_t *feasible, uint64_t *ties, int64_t *mincost, const tr_math_budget *budget,
                       tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    int32_t rc = tr_repair_view_check(v);
    if (rc) return rc;
    if (count && (!masks || !costs || !feasible || !ties)) return TR_MATH_INVALID;
    if (!mincost) return TR_MATH_INVALID;
    tr_math_budget dflt;
    budget = budget_or(budget, &dflt);
    tr_math_need need;
    need_zero(&need);
    need.need[TR_BUDGET_OUTPUT] = count;
    need.need[TR_BUDGET_BYTES] = tr_budget_mul(tr_repair_scratch_words(v), 8);
    if ((rc = tr_budget_check(budget, &need, refusal))) return rc;
    cpu_eval e;
    if ((rc = cpu_open(v, &e))) return rc;
    cpu_scan(&e, masks, costs, count, match, feasible, ties, mincost);
    free(e.scratch);
    return 0;
}

/* ================================================================== prepared searches */
typedef struct { const tr_attr_q *val; const uint32_t *el; } ord_ctx;
/* the order of the allowed elements: INDEX, COST (cheapest first), SHAPLEY / BANZHAF (decreasing index; ties by index) */
static int32_t order_elements(const tr_repair_view *v, int32_t order, const int64_t *cost, const tr_attr_game *game,
                              uint32_t *perm, int32_t m, const tr_math_budget *budget, tr_math_refusal *refusal) {
    int32_t k = 0;
    for (uint64_t a = v->allowed; a; a &= a - 1) perm[k++] = (uint32_t)__builtin_ctzll(a);
    if (order == TR_REPAIR_ORDER_INDEX) return 0;
    if (order == TR_REPAIR_ORDER_COST) {
        for (int32_t i = 1; i < m; i++) {
            uint32_t x = perm[i];
            int32_t j = i;
            while (j > 0 && cost[perm[j - 1]] > cost[x]) { perm[j] = perm[j - 1]; j--; }
            perm[j] = x;
        }
        return 0;
    }
    if (!game) return TR_MATH_INVALID;
    tr_attr_game_info gi;
    if (tr_attr_game_get_info(game, &gi)) return TR_MATH_INVALID;
    if (gi.n != v->n) return TR_MATH_DOMAIN_ERROR;                   /* the game's players are the elements */
    const int32_t n = gi.n;
    tr_attr_q *vals = zalloc((uint64_t)n * 2 * sizeof(tr_attr_q));
    if (!vals) return TR_MATH_MEMORY;
    int32_t rc;
    if (n <= 20) {
        tr_math_i128 *sn = zalloc((uint64_t)n * 16), *bn = zalloc((uint64_t)n * 16);
        tr_attr_exhaustive_info ei;
        rc = sn && bn ? tr_attr_exhaustive(game, sn, bn, vals, vals + n, &ei, budget, refusal) : TR_MATH_MEMORY;
        free(sn); free(bn);
    } else {
        int32_t e = 0;
        tr_attr_support *s = tr_attr_support_build(game, TR_ATTR_SUPPORT_AUTO, budget, refusal, &e);
        rc = s ? tr_attr_support_values(s, TR_ATTR_SHAPLEY, vals, NULL, budget, refusal) : e;
        if (!rc) rc = tr_attr_support_values(s, TR_ATTR_BANZHAF, vals + n, NULL, budget, refusal);
        tr_attr_support_free(s);
    }
    if (rc) { free(vals); return rc; }
    const tr_attr_q *x = order == TR_REPAIR_ORDER_SHAPLEY ? vals : vals + n;
    for (int32_t i = 1; i < m; i++) {                                 /* decreasing value, then increasing index */
        uint32_t e = perm[i];
        int32_t j = i;
        while (j > 0) {
            const tr_attr_q *a = &x[perm[j - 1]], *b = &x[e];
            int c = tr_repair_cmp_prod(tr_i128_get(a->num), tr_i128_get(b->den), tr_i128_get(b->num), tr_i128_get(a->den));
            if (c > 0 || (c == 0 && perm[j - 1] < e)) break;
            perm[j] = perm[j - 1]; j--;
        }
        perm[j] = e;
    }
    free(vals);
    return 0;
}
/* positive rational costs as integers over their common denominator (unit costs: NULL) */
static int32_t scale_costs(const tr_repair_view *v, const tr_bigq *const *cost, int64_t *out, int64_t *den,
                           int64_t *unit) {
    const int32_t n = v->n;
    int64_t L = 1;
    *unit = 1;
    if (!cost) { for (int32_t i = 0; i < n; i++) out[i] = 1; *den = 1; return 0; }
    for (int32_t i = 0; i < n; i++) {
        if (!((v->allowed >> i) & 1)) continue;
        tr_num_rational q;
        if (!cost[i] || !tr_bigq_is_handle(cost[i])) return TR_MATH_INVALID;
        if (tr_bigq_sign(cost[i]) <= 0) return TR_MATH_INVALID;
        if (tr_bigq_to_q64(cost[i], &q)) return TR_MATH_OVERFLOW;
        uint64_t g = tr_gcd_u64((uint64_t)L, (uint64_t)q.denominator);
        int64_t t;
        if (!tr_mul_i64(L / (int64_t)g, q.denominator, &t)) return TR_MATH_OVERFLOW;
        L = t;
    }
    int64_t total = 0;
    for (int32_t i = 0; i < n; i++) {
        if (!((v->allowed >> i) & 1)) { out[i] = 1; continue; }
        tr_num_rational q;
        tr_bigq_to_q64(cost[i], &q);
        int64_t c;
        if (!tr_mul_i64(q.numerator, L / q.denominator, &c) || !tr_add_i64(total, c, &total) || total > (INT64_C(1) << 62))
            return TR_MATH_OVERFLOW;
        out[i] = c;
        if (c != 1) *unit = 0;
    }
    *den = L;
    return 0;
}
static int32_t plan_fill(const tr_repair_problem *p, const tr_repair_query *q, tr_repair_plan *pl) {
    if (!problem_ok(p) || !q || q->reserved0 || q->reserved1 || q->method < TR_REPAIR_AUTO || q->method > TR_REPAIR_BRANCH_BOUND ||
        q->order < TR_REPAIR_ORDER_INDEX || q->order > TR_REPAIR_ORDER_BANZHAF || q->max_states < 1) return TR_MATH_INVALID;
    memset(pl, 0, sizeof *pl);
    pl->n = p->view.n; pl->allowed = p->view.allowed; pl->m = __builtin_popcountll(p->view.allowed);
    pl->order = q->order; pl->max_states = q->max_states; pl->max_solutions = q->max_solutions;
    int method = q->method;
    const uint64_t space = pl->m >= 63 ? UINT64_MAX : (uint64_t)1 << pl->m;
    if (method == TR_REPAIR_AUTO) method = (space <= q->max_states && pl->m <= TR_REPAIR_MAX_SORTED) ? TR_REPAIR_EXHAUSTIVE : TR_REPAIR_BRANCH_BOUND;
    pl->method = method;
    return 0;
}
int32_t tr_repair_plan_need(const tr_repair_problem *p, const tr_repair_query *q, tr_math_need *need) {
    tr_repair_plan pl;
    if (!need) return TR_MATH_INVALID;
    int32_t rc = plan_fill(p, q, &pl);
    if (rc) return rc;
    pl.unit = q->cost == NULL;
    tr_repair_search_need(&pl, need);
    return 0;
}
int32_t tr_repair_prepare(const tr_repair_problem *p, const tr_repair_query *q, const tr_attr_game *order_game,
                          int64_t *cost_out, uint32_t *perm_out, tr_repair_plan *plan, const tr_math_budget *budget,
                          tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    tr_repair_plan pl;
    int32_t rc = plan_fill(p, q, &pl);
    if (rc) return rc;
    if (!cost_out || !perm_out || !plan) return TR_MATH_INVALID;
    tr_math_budget dflt;
    budget = budget_or(budget, &dflt);
    if ((rc = scale_costs(&p->view, q->cost, cost_out, &pl.den, &pl.unit))) return rc;
    if (pl.method == TR_REPAIR_EXHAUSTIVE && !pl.unit && pl.m > TR_REPAIR_MAX_SORTED) return TR_MATH_UNSUPPORTED;
    pl.cost = cost_out;
    tr_math_need need;
    tr_repair_search_need(&pl, &need);
    if ((rc = tr_budget_check(budget, &need, refusal))) return rc;
    if ((rc = order_elements(&p->view, q->order, cost_out, order_game, perm_out, pl.m, budget, refusal))) return rc;
    pl.perm = perm_out;
    *plan = pl;
    return 0;
}
static int32_t plan_check(const tr_repair_view *v, const tr_repair_plan *pl) { return tr_repair_plan_check(v, pl); }
int32_t tr_repair_search(const tr_repair_problem *p, const tr_repair_plan *pl, uint64_t *solutions, int64_t *sol_cost,
                         tr_repair_info *info, const tr_math_budget *budget, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!problem_ok(p) || !info || (pl && pl->max_solutions && !solutions)) return TR_MATH_INVALID;
    int32_t rc = plan_check(&p->view, pl);
    if (rc) return rc;
    tr_math_budget dflt;
    budget = budget_or(budget, &dflt);
    tr_math_need need;
    tr_repair_search_need(pl, &need);
    need.need[TR_BUDGET_BYTES] = tr_budget_add(need.need[TR_BUDGET_BYTES], tr_budget_mul(tr_repair_scratch_words(&p->view), 8));
    if ((rc = tr_budget_check(budget, &need, refusal))) return rc;
    cpu_eval e;
    if ((rc = cpu_open(&p->view, &e))) return rc;
    rc = tr_repair_run(&p->view, pl, cpu_batch, cpu_scan, &e, TR_REPAIR_BATCH, solutions, sol_cost, info);
    free(e.scratch);
    return rc;
}

/* ================================================================== redundancy */
int32_t tr_repair_redundancy(const tr_repair_problem *p, tr_repair_removal *out, uint64_t capacity, uint64_t *rows,
                             uint64_t max_witnesses, const tr_math_budget *budget, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (!problem_ok(p)) return TR_MATH_INVALID;
    const tr_repair_view *v = &p->view;
    const uint64_t m = (uint64_t)__builtin_popcountll(v->allowed);
    if (capacity < m || (m && !out) || (max_witnesses && m && !rows)) return TR_MATH_INVALID;
    tr_math_budget dflt;
    budget = budget_or(budget, &dflt);
    tr_math_need need;
    need_zero(&need);
    need.need[TR_BUDGET_OUTPUT] = tr_budget_add(m, tr_budget_mul(m, max_witnesses));
    need.need[TR_BUDGET_BYTES] = tr_budget_mul(tr_repair_scratch_words(v), 8);
    int32_t rc = tr_budget_check(budget, &need, refusal);
    if (rc) return rc;
    cpu_eval e;
    if ((rc = cpu_open(v, &e))) return rc;
    uint64_t q = 0;
    for (uint64_t a = v->allowed; a; a &= a - 1, q++) {
        const int32_t i = __builtin_ctzll(a);
        const uint64_t S = v->allowed & ~((uint64_t)1 << i);
        tr_repair_out o;
        tr_repair_eval_one(v, S, e.scratch, &o);
        tr_repair_removal *r = &out[q];
        memset(r, 0, sizeof *r);
        r->element = i; r->redundant = o.feasible; r->count = o.count; r->value = o.value;
        if (v->kind == TR_REPAIR_CIRCUIT) {                          /* the rows where the selections differ */
            uint64_t *wr = max_witnesses ? rows + q * max_witnesses : NULL, nw = 0;
            for (int pass = 0; pass < 2; pass++)
                for (uint64_t w = 0; w < v->dom.nw; w++) {
                    if (!tr_bool_selection_word(&v->dom, w)) continue;
                    const uint64_t x = tr_repair_circuit_word(v, S, w, e.scratch), t = v->target[w];
                    uint64_t bits = pass == 0 ? (x & ~t) : (t & ~x);
                    if (pass == 0) r->gained += (uint64_t)__builtin_popcountll(bits);
                    else r->lost += (uint64_t)__builtin_popcountll(bits);
                    while (bits && nw < max_witnesses) {
                        int b = __builtin_ctzll(bits);
                        bits &= bits - 1;
                        wr[nw++] = w * 64 + (uint64_t)b;
                    }
                }
            r->witnesses = nw;
        }
    }
    free(e.scratch);
    return 0;
}

/* ================================================================== continuous repairs (INV-002) */
static int32_t repair_op_to_interv(int32_t op) { return op - 1; }   /* TR_REPAIR_EQ.. = TR_INTERV_EQ.. + 1 */
static tr_bigq *bigq_value(int64_t logical, int32_t scale, const tr_math_budget *b, tr_math_refusal *r, int32_t *e) {
    int64_t p = 1;
    for (int32_t i = 0; i < scale; i++) p *= 10;
    tr_num_rational q = {logical, p};
    tr_num_rational c;
    if (!tr_q_make(q.numerator, q.denominator, &c)) { *e = TR_MATH_OVERFLOW; return NULL; }
    return tr_bigq_from_q64(c, b, r, e);
}
tr_interv_result *tr_repair_edit(const tr_scope *scope, const tr_bool_domain *d, const tr_bool_program_view *pv,
                                 int32_t value, int32_t agg, int32_t op, const tr_bigq *t, int32_t n,
                                 const uint64_t *rows, int32_t cost, const tr_bigq *const *weights,
                                 const tr_bigq *const *dlo, const tr_bigq *const *dhi, const tr_math_budget *budget,
                                 tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    tr_math_budget dflt;
    budget = budget_or(budget, &dflt);
    tr_scope_view sv;
    if (!scope || tr_scope_get_view(scope, &sv) || !d || d->kind != TR_BOOL_ROWS || d->points != sv.rows || !t ||
        !tr_bigq_is_handle(t) || n < 1 || !rows || (agg != TR_REPAIR_SUM && agg != TR_REPAIR_AVG) || value < 0 ||
        value >= sv.ncols || op < TR_REPAIR_EQ || op > TR_REPAIR_GE) { set_error(error, TR_MATH_INVALID); return NULL; }
    if (n > TR_INTERV_MAX_DIMS) { set_error(error, TR_MATH_UNSUPPORTED); return NULL; }
    int32_t rc = tr_bool_domain_check(d);
    if (!rc && pv) rc = program_against(d, pv);
    if (rc) { set_error(error, rc); return NULL; }
    for (uint64_t w = 0; w < d->nw; w++) if (d->selection[w] != sv.selection[w]) { set_error(error, TR_MATH_DOMAIN_ERROR); return NULL; }
    for (int32_t i = 0; i < n; i++) {
        if (rows[i] >= sv.rows || !tr_scope_is_selected(&sv, rows[i]) || !tr_scope_present(&sv, value, rows[i])) {
            set_error(error, TR_MATH_INVALID); return NULL;            /* an edited value must be present */
        }
        for (int32_t j = 0; j < i; j++) if (rows[j] == rows[i]) { set_error(error, TR_MATH_INVALID); return NULL; }
    }
    uint64_t *qw = NULL;
    if (pv) { qw = program_true(d, pv, &rc); if (rc) { set_error(error, rc); return NULL; } }
    /* the other qualifying values (their exact sum) and the count of qualifying values */
    uint8_t *ed = zalloc(sv.rows + 1);
    if (!ed) { free(qw); set_error(error, TR_MATH_MEMORY); return NULL; }
    for (int32_t i = 0; i < n; i++) ed[rows[i]] = 1;
    __int128 base = 0;
    uint64_t vc = 0;
    const unsigned __int128 lim = (unsigned __int128)1 << 120;
    for (uint64_t r = 0; r < sv.rows && !rc; r++) {
        if (!tr_scope_is_selected(&sv, r) || (qw && !bitw(qw, r)) || !tr_scope_present(&sv, value, r)) continue;
        vc++;
        if (!ed[r]) { base += tr_scope_value(&sv, value, r); if (tr_repair_uabs(base) >= lim) rc = TR_MATH_OVERFLOW; }
    }
    if (!rc && !vc) rc = TR_MATH_INFEASIBLE;                         /* SUM / AVG of no value is NULL: no edit helps */
    const int32_t sc = sv.col[value].scale;
    tr_bigq **x0 = zalloc((uint64_t)n * sizeof *x0), **a = zalloc((uint64_t)n * sizeof *a);
    tr_bigq *a0 = NULL, *one = NULL;
    if (!x0 || !a) rc = rc ? rc : TR_MATH_MEMORY;
    int32_t e = 0;
    if (!rc) {
        tr_math_i128 bw = tr_i128_set(base);
        tr_bigint *bi = tr_bigint_from_i128(bw, budget, refusal, &e), *p10 = NULL, *ten = NULL, *den = NULL;
        ten = tr_bigint_from_i64(10, budget, refusal, &e);
        p10 = ten ? tr_bigint_pow(ten, (uint64_t)sc, budget, refusal, &e) : NULL;
        den = p10;
        if (p10 && agg == TR_REPAIR_AVG) {
            tr_bigint *cnt = tr_bigint_from_i64((int64_t)vc, budget, refusal, &e);
            den = cnt ? tr_bigint_mul(p10, cnt, budget, refusal, &e) : NULL;
            tr_bigint_free(cnt);
        }
        a0 = bi && den ? tr_bigq_from_bigints(bi, den, budget, refusal, &e) : NULL;
        if (agg == TR_REPAIR_AVG) {
            tr_bigint *o = tr_bigint_from_i64(1, budget, refusal, &e), *cnt = tr_bigint_from_i64((int64_t)vc, budget, refusal, &e);
            one = o && cnt ? tr_bigq_from_bigints(o, cnt, budget, refusal, &e) : NULL;
            tr_bigint_free(o); tr_bigint_free(cnt);
        } else {
            tr_num_rational q1 = {1, 1};
            one = tr_bigq_from_q64(q1, budget, refusal, &e);
        }
        if (den != p10) tr_bigint_free(den);
        tr_bigint_free(p10); tr_bigint_free(ten); tr_bigint_free(bi);
        if (!a0 || !one) rc = e ? e : TR_MATH_MEMORY;
    }
    for (int32_t i = 0; !rc && i < n; i++) {
        x0[i] = bigq_value(tr_scope_value(&sv, value, rows[i]), sc, budget, refusal, &e);
        if (!x0[i]) { rc = e ? e : TR_MATH_MEMORY; break; }
        const int q = !qw || bitw(qw, rows[i]);
        if (q) { a[i] = tr_bigq_copy(one, budget, refusal, &e); if (!a[i]) { rc = e ? e : TR_MATH_MEMORY; break; } }
    }
    tr_interv_result *res = NULL;
    if (!rc) {
        tr_interv_linear_problem lp;
        memset(&lp, 0, sizeof lp);
        lp.n = n; lp.cost = cost; lp.op = repair_op_to_interv(op); lp.flip = 0; lp.semantics = TR_INTERV_ALGEBRAIC;
        lp.x0 = (const tr_bigq *const *)x0; lp.weight = weights; lp.a = (const tr_bigq *const *)a; lp.a0 = a0; lp.t = t;
        lp.dlo = dlo; lp.dhi = dhi;
        res = tr_interv_linear(&lp, budget, refusal, &e);
        if (!res) rc = e ? e : TR_MATH_MEMORY;
    }
    for (int32_t i = 0; x0 && i < n; i++) tr_bigq_free(x0[i]);
    for (int32_t i = 0; a && i < n; i++) tr_bigq_free(a[i]);
    free(x0); free(a); tr_bigq_free(a0); tr_bigq_free(one); free(ed); free(qw);
    set_error(error, rc);
    return rc ? NULL : res;
}
int32_t tr_repair_threshold(const tr_scope *scope, const tr_bool_condition *conds, int32_t k, uint64_t which,
                            int32_t value, int32_t agg, int32_t op, const tr_bigq *t, const tr_bigq *const *weights,
                            tr_interv_result **results, uint64_t *attaining, const tr_math_budget *budget,
                            tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    tr_math_budget dflt;
    budget = budget_or(budget, &dflt);
    tr_scope_view sv;
    if (!scope || tr_scope_get_view(scope, &sv) || !conds || k < 1 || k > TR_BOOL_MAX_CONDITIONS || !which ||
        (k < 64 && (which >> k)) || !t || !tr_bigq_is_handle(t) || !results || !attaining ||
        agg < TR_REPAIR_COUNT_STAR || agg > TR_REPAIR_AVG || op < TR_REPAIR_EQ || op > TR_REPAIR_GE ||
        (agg != TR_REPAIR_COUNT_STAR && (value < 0 || value >= sv.ncols))) return TR_MATH_INVALID;
    for (int32_t i = 0; i < k; i++) {
        if (tr_bool_condition_check(&sv, &conds[i])) return TR_MATH_INVALID;
        if (((which >> i) & 1) && (conds[i].op < TR_BOOL_EQ || conds[i].op > TR_BOOL_GE)) return TR_MATH_UNSUPPORTED;
    }
    tr_math_need need;
    need_zero(&need);
    need.need[TR_BUDGET_VARIABLES] = (uint64_t)k;
    need.need[TR_BUDGET_OUTPUT] = (uint64_t)__builtin_popcountll(which);
    need.need[TR_BUDGET_BYTES] = tr_budget_mul(sv.rows, 8 + 8 + 2 + 1);
    int32_t rc = tr_budget_check(budget, &need, refusal);
    if (rc) return rc;
    const uint64_t nw = (sv.rows + 63) / 64;
    uint8_t *ok = zalloc(sv.rows * (uint64_t)k + 1);                 /* the TRUE flag of every condition per row */
    uint64_t *selw = zalloc(nw * 8 + 8);
    int64_t *xs = zalloc(sv.rows * 8 + 8), *vs = zalloc(sv.rows * 8 + 8);
    uint8_t *xv = zalloc(sv.rows + 1), *vv = zalloc(sv.rows + 1);
    if (!ok || !selw || !xs || !vs || !xv || !vv) rc = TR_MATH_MEMORY;
    for (uint64_t r = 0; !rc && r < sv.rows; r++) {
        if (!tr_scope_is_selected(&sv, r)) continue;
        for (int32_t i = 0; i < k; i++) ok[r * k + i] = (uint8_t)(tr_bool_condition_row(&sv, &conds[i], r) == TR_BOOL_TRUE);
    }
    uint64_t q = 0;
    tr_interv_result *made[TR_BOOL_MAX_CONDITIONS];
    memset(made, 0, sizeof made);
    for (int32_t j = 0; !rc && j < k; j++) {
        if (!((which >> j) & 1)) continue;
        const tr_bool_condition *c = &conds[j];
        /* the rows the other conditions keep TRUE; the condition's column and the value column */
        memset(selw, 0, (size_t)nw * 8);
        for (uint64_t r = 0; r < sv.rows; r++) {
            int keep = tr_scope_is_selected(&sv, r);
            for (int32_t i = 0; keep && i < k; i++) if (i != j && !ok[r * k + i]) keep = 0;
            if (keep) selw[r >> 6] |= (uint64_t)1 << (r & 63);
            const int hx = keep && tr_scope_present(&sv, c->col, r), hv = keep && value >= 0 && tr_scope_present(&sv, value, r);
            xv[r] = (uint8_t)hx; xs[r] = hx ? tr_scope_value(&sv, c->col, r) : 0;
            vv[r] = (uint8_t)hv; vs[r] = hv ? tr_scope_value(&sv, value, r) : 0;
        }
        tr_scope_array cols[2];
        memset(cols, 0, sizeof cols);
        cols[0].values = xs; cols[0].valid = xv; cols[0].scale = sv.col[c->col].scale;
        cols[1].values = vs; cols[1].valid = vv; cols[1].scale = value >= 0 ? sv.col[value].scale : 0;
        int32_t e = 0;
        tr_scope *sub = tr_scope_from_arrays(sv.rows, selw, cols, value >= 0 ? 2 : 1, NULL, 1, budget, refusal, &e);
        if (!sub) { rc = e ? e : TR_MATH_MEMORY; break; }
        tr_interv_agg *st = tr_interv_agg_build(sub, 0, value >= 0 ? 1 : -1, budget, refusal, &e);
        tr_bigq *cq = NULL;
        if (st) {
            tr_num_rational cr;
            if (!tr_q_make(c->num[0], c->den[0], &cr)) e = TR_MATH_OVERFLOW;
            else cq = tr_bigq_from_q64(cr, budget, refusal, &e);
        }
        if (st && cq) {
            tr_interv_agg_query aq;
            memset(&aq, 0, sizeof aq);
            aq.aggregate = agg; aq.pred_op = c->op - 1; aq.op = repair_op_to_interv(op); aq.flip = 0;
            aq.cost = TR_INTERV_L1; aq.semantics = TR_INTERV_ALGEBRAIC;
            aq.c = cq; aq.t = t; aq.weight = weights ? weights[j] : NULL;
            made[j] = tr_interv_agg_solve(st, &aq, budget, refusal, &e);
        }
        if (!made[j]) rc = e ? e : TR_MATH_MEMORY;
        tr_bigq_free(cq);
        tr_interv_agg_free(st);
        tr_scope_free(sub);
        q++;
    }
    /* the minimum cost over the conditions with an answer (optimal or an infimum), and every condition attaining it */
    uint64_t att = 0;
    const tr_bigq *best = NULL;
    for (int32_t j = 0; !rc && j < k; j++) {
        if (!made[j]) continue;
        tr_interv_info ii;
        tr_interv_result_get_info(made[j], &ii);
        if (ii.status != TR_INTERV_OPTIMAL && ii.status != TR_INTERV_UNATTAINED) continue;
        uint64_t cnt = 0;
        const tr_bigq *const *lo = tr_interv_result_part(made[j], TR_INTERV_PART_LOWER, &cnt);
        if (!lo || !cnt) continue;
        int c = best ? tr_bigq_cmp(lo[0], best) : -1;
        if (c < 0) { best = lo[0]; att = (uint64_t)1 << j; }
        else if (c == 0) att |= (uint64_t)1 << j;
    }
    free(ok); free(selw); free(xs); free(vs); free(xv); free(vv);
    if (rc) { for (int32_t j = 0; j < k; j++) tr_interv_result_free(made[j]); return rc; }
    q = 0;
    for (int32_t j = 0; j < k; j++) if ((which >> j) & 1) results[q++] = made[j];
    *attaining = att;
    return 0;
}

tr_bigq *tr_repair_threshold_after(const tr_bool_condition *c, const tr_bigq *shift, const tr_math_budget *budget,
                                   tr_math_refusal *refusal, int32_t *error) {
    tr_budget_none(refusal);
    tr_math_budget dflt;
    budget = budget_or(budget, &dflt);
    if (!c || !shift || !tr_bigq_is_handle(shift) || c->den[0] < 1) { set_error(error, TR_MATH_INVALID); return NULL; }
    tr_num_rational cr;
    if (!tr_q_make(c->num[0], c->den[0], &cr)) { set_error(error, TR_MATH_OVERFLOW); return NULL; }
    int32_t e = 0;
    tr_bigq *cq = tr_bigq_from_q64(cr, budget, refusal, &e), *out = cq ? tr_bigq_sub(cq, shift, budget, refusal, &e) : NULL;
    tr_bigq_free(cq);
    set_error(error, out ? TR_MATH_OK : (e ? e : TR_MATH_MEMORY));
    return out;
}
int32_t tr_repair_device_need(const tr_repair_view *v, uint64_t count, const tr_repair_plan *pl, tr_math_need *need) {
    if (!need) return TR_MATH_INVALID;
    int32_t rc = tr_repair_view_check(v);
    if (rc) return rc;
    need_zero(need);
    if (pl) {
        if ((rc = plan_check(v, pl))) return rc;
        tr_repair_search_need(pl, need);
        need->need[TR_BUDGET_BYTES] = tr_budget_add(need->need[TR_BUDGET_BYTES],
                                                    tr_repair_view_device_bytes(v, 2 * TR_REPAIR_BATCH));
        return 0;
    }
    need->need[TR_BUDGET_OUTPUT] = count;
    need->need[TR_BUDGET_BYTES] = tr_repair_view_device_bytes(v, count);
    return 0;
}
int32_t tr_repair_group_words(const tr_scope *scope, uint64_t group, uint64_t *words, uint64_t nwords) {
    tr_scope_view sv;
    if (!scope || tr_scope_get_view(scope, &sv) || !words || nwords != (sv.rows + 63) / 64 || group >= sv.ngroups)
        return TR_MATH_INVALID;
    memset(words, 0, (size_t)nwords * 8);
    for (uint64_t r = 0; r < sv.rows; r++)
        if (tr_scope_is_selected(&sv, r) && tr_scope_group(&sv, r) == group) words[r >> 6] |= (uint64_t)1 << (r & 63);
    return 0;
}
