/* PROV-002: explanation specialization and deletion propagation over retained PROV-001 provenance: the exact effect
 * of deleting (or re-valuing) source tuples on every answer's existence and multiplicity, propagated incrementally
 * through the factorized DAG with the update laws of the semirings; explanations (lineage, relevant and necessary
 * sources, minimal sufficient sets, a smallest one, why-not restorations); and answer probabilities under independent
 * tuple probabilities, exact on the DAG with its shared variables (knowledge compilation to a decision diagram,
 * COUNT-002) or certified bounds, never products of per-derivation probabilities. CPU: math_explain.c; the batched
 * deletion propagation (one work item per change set) in CUDA: math_explain_gpu.cu, equal bit for bit.
 * SPDX-License-Identifier: Apache-2.0
 *
 * ---- The state (tr_expl_state, immutable) --------------------------------------------------------------------------
 * Built from a PROV-001 evaluation plan (tr_prov_plan_view: the cone of some answers' annotations, retained as it was
 * derived; nothing is recomputed from the sources) and a base valuation (count 1: a default value for every variable
 * and sorted overrides; e.g. 1 = every source tuple present, 0 = absent). It copies the cone and adds the reverse
 * edges (one parent entry per child entry, parents increasing), the outputs of every node, the cone node of every
 * variable, and the baseline per node: N = the multiplicity (unsigned 128 bits, PROV-001's tr_prov_eval_count) and
 * B = the existence (tr_prov_eval_bool), and the support counts the B updates need: for PLUS the child entries with
 * B = 1, for TIMES the child entries with B = 0. count_ok = 0 when the baseline N leaves 128 bits somewhere (then only
 * B is propagated and every N result reports TR_MATH_OVERFLOW).
 *
 * ---- Deletion propagation (tr_expl_propagate / tr_mgpu_explain_propagate) -----------------------------------------
 * A change set overrides some variables' values (a deletion is 0; any natural is allowed); the other variables keep
 * their base values (tr_prov_valuations with fallback NULL). Only the nodes above a changed variable are visited: a
 * min-heap of node ids processes them in increasing order (children before parents), each once, and a node whose N
 * and B both equal the baseline stops the propagation (early cutoff). Update laws (exact; DELTA-002's finite
 * differences, not derivatives):
 *   PLUS  N' = N - sum over the changed child entries of N(c) + sum of N'(c)        (the linear law: the SUM rule)
 *   TIMES N' = prod over the child entries of their new values: N + dA B + A dB + dA dB for two children (bilinear)
 *   B     counting maintenance: PLUS exists iff (entries with B = 1) > 0, TIMES iff (entries with B = 0) = 0, the
 *         counts adjusted by the changed entries only (OR / AND without re-reading the unchanged children)
 * Per set: the N (two words per answer) and B rows of every answer (the baseline where nothing changed), the status
 * (0, or TR_MATH_OVERFLOW: a value past 128 bits; the set is then redone in B alone and its N row is zero), the
 * nodes recomputed (affected: the incremental cost, against the cone's size) and the answers changed. The CPU and
 * the kernel run the same tr_expl_set; a work item uses its own scratch row and restores it (the touched entries
 * only) when done, so the results do not depend on the thread order.
 *
 * ---- Explanations (tr_expl_explain -> tr_expl_sets) ---------------------------------------------------------------
 * For one answer (output j) relative to the base: present sources P (base value != 0) and absent ones. Its existence
 * is the monotone Boolean function f of the lineage (PLUS = OR, TIMES = AND). Compiled (COUNT-002's decision diagram)
 * with substitutions: WHY g = f with the absent sources FALSE; WHY_NOT h = f with the present sources TRUE.
 *   LINEAGE     the source tuples of the answer's cone (every one occurs in some derivation: N depends on it)
 *   RELEVANT    the support of g: the present sources the existence depends on (x + x y: x only, though y is in the
 *               lineage)
 *   NECESSARY   the counterfactual sources: x in P with g|x=0 = FALSE (deleting x alone removes the answer)
 *   SUFFICIENT  the minimal sufficient sets: the minimal S subset of P with f(S) = 1 (the minimal models of g:
 *               MM(n) = MM(lo) + {x} x {S in MM(hi) : lo(S) = 0}, on the diagram; for the all-present base these are
 *               PROV-001's minimal witnesses)
 *   SMALLEST    one minimum-cardinality sufficient set (the fewest hi edges on a path to TRUE; ties: lo first)
 *   WHY_NOT     for an absent answer: the minimal sets A of absent sources with f(P + A) = 1 (the minimal models of h)
 * Status: FOUND; ABSENT (g = FALSE: the answer does not exist under the base, so nothing explains it); PRESENT
 * (WHY_NOT of an existing answer: h = TRUE, the empty set); UNDERIVABLE (h = FALSE: no recorded derivation can make
 * it exist). Sets are sorted variable lists; SUFFICIENT and WHY_NOT lists are refused past max_sets (never truncated).
 *
 * ---- Probabilities ------------------------------------------------------------------------------------------------
 * Independent tuples with p_v = num/den (0 <= num <= den): P(answer exists) = the weighted count of f's diagram with
 * the literal weights (p, 1 - p) (tr_expl_lineage + COUNT-002's tr_symc_dd_count or tr_mgpu_symcount_levels): exact
 * rationals, shared variables handled exactly (a tuple used by several derivations is one variable of the diagram).
 * Past the diagram's node / step budget the caller gets the refusal, or tr_expl_probability_bounds: certified
 * enclosures on the DAG without compilation (outward-rounded doubles): VAR [p, p]; a PLUS / TIMES whose children
 * have pairwise disjoint supports is exact independence (OR 1 - prod (1 - P_c), AND prod P_c); otherwise
 * Harris-FKG for increasing events: AND in [prod lo_c, min hi_c], OR in [max lo_c, 1 - prod (1 - hi_c)].
 * read_once = 1 when every node of the cone had disjoint children (the enclosure is then of the exact value).
 *
 * ---- Budgets (math_budgets.h), checked before allocation -----------------------------------------------------------
 *   state: circuit = cone nodes, support = child entries, output = answers, variables = the store's variables, bytes
 *   propagate: output = sets x answers, bytes = the scratch (65 per cone node) (CUDA: + the device copies, the outputs
 *     and the scratch chunk: tr_expl_device_bytes)
 *   lineage: the manager's (circuit = diagram nodes, iterations = apply steps, bytes; variables = the levels)
 *   explain: NECESSARY iterations = levels x diagram nodes; SUFFICIENT / WHY_NOT support = the set entries held by the
 *     memo, output = max_sets; bytes
 *   bounds: bytes = the support bitsets (cone nodes x words) and the intervals
 * Errors: TR_MATH_INVALID (-1), _MEMORY (-2), _OVERFLOW (-3), _BUDGET (-5), TR_MATH_DEVICE (-11), _UNSUPPORTED (-12). */
#ifndef TORBLES_MATH_EXPLAIN_H
#define TORBLES_MATH_EXPLAIN_H
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "math_num.h"
#include "math_budgets.h"
#include "math_prov.h"
#include "math_symcount.h"
#ifdef __cplusplus
extern "C" {
#endif

#define TR_EXPL_ABI 1
#define TR_EXPL_NONE UINT32_MAX
#define TR_EXPL_SCRATCH_PER_NODE 65                   /* bytes of propagation scratch per cone node */
#define TR_EXPL_DEVICE_SCRATCH (UINT64_C(1) << 26)    /* bytes of device scratch per launch chunk (at most) */
enum { TR_EXPL_QUEUED = 1, TR_EXPL_CHANGED = 2, TR_EXPL_BNEW = 4, TR_EXPL_ADD_OVER = 8 };   /* scratch flags */
enum { TR_EXPL_LINEAGE = 1, TR_EXPL_RELEVANT = 2, TR_EXPL_NECESSARY = 3, TR_EXPL_SUFFICIENT = 4,
       TR_EXPL_SMALLEST = 5, TR_EXPL_WHY_NOT = 6 };                                         /* explanation kinds */
enum { TR_EXPL_FOUND = 1, TR_EXPL_ABSENT = 2, TR_EXPL_PRESENT = 3, TR_EXPL_UNDERIVABLE = 4 };   /* their status */
enum { TR_EXPL_FREE = 0, TR_EXPL_WHY = 1, TR_EXPL_WHYNOT = 2 };                             /* lineage substitutions */
enum { TR_EXPL_NEED_CPU = 0, TR_EXPL_NEED_DEVICE = 1 };

/* ------------------------------------------------------------------ structs crossing the ctypes boundary */
/* the state as the CPU and the kernel read it (pointers into the state; valid while it lives) */
typedef struct {
    uint64_t nodes, nchild, noutputs, nvars;
    const tr_prov_node *node;          /* [nodes] the cone (PROV-001 plan nodes, cone-local children) */
    const uint32_t *child;             /* [nchild] */
    const uint32_t *output;            /* [noutputs] the cone node of each answer */
    const uint64_t *rev_at;            /* [nodes + 1] parents of node i: rev[rev_at[i] .. rev_at[i + 1]) */
    const uint32_t *rev;               /* [nchild] one parent per child entry, increasing */
    const uint64_t *out_at;            /* [nodes + 1] answers of node i: outs[out_at[i] .. out_at[i + 1]) */
    const uint32_t *outs;              /* [noutputs] */
    const uint32_t *var_node;          /* [nvars] the cone node of variable v, or TR_EXPL_NONE */
    const uint64_t *base_n;            /* [2 nodes] the baseline N (low, high words) */
    const uint8_t *base_b;             /* [nodes] the baseline B */
    const uint32_t *base_c;            /* [nodes] PLUS: entries with B = 1; TIMES: entries with B = 0 */
    int32_t count_ok, reserved;
} tr_expl_view;                        /* 128 bytes */

typedef struct { int32_t status, reserved; uint64_t affected, changed; } tr_expl_set_result;   /* 24 bytes */

typedef struct {
    uint64_t nodes, children, outputs, variables;   /* cone, child entries, answers, the store's variables */
    uint64_t cone_variables, present, bytes;        /* variables of the cone; present under the base; bytes owned */
    int32_t count_ok, reserved;
} tr_expl_info;                        /* 64 bytes */

typedef struct {
    int32_t kind, status;
    uint64_t nsets, entries;           /* sets; variables listed in all */
    uint64_t diagram_nodes, levels;    /* the compiled diagram (0 for LINEAGE) */
} tr_expl_sets_info;                   /* 40 bytes */

/* the propagation scratch of one change set (pointers into a row: 65 bytes per node) */
typedef struct {
    uint64_t *newn, *sub, *add;        /* [2 nodes] each */
    int64_t *dcnt;                     /* [nodes] */
    uint32_t *heap, *touched;          /* [nodes] each */
    uint8_t *flag;                     /* [nodes] */
} tr_expl_scratch;

typedef struct tr_expl_state_s tr_expl_state;
typedef struct tr_expl_sets_s tr_expl_sets;

/* ------------------------------------------------------------------ the shared propagation (CPU and CUDA) */
TR_NUM_HD unsigned __int128 tr_expl_get(const uint64_t *a, uint64_t i) {
    return ((unsigned __int128)a[2 * i + 1] << 64) | a[2 * i];
}
TR_NUM_HD void tr_expl_put(uint64_t *a, uint64_t i, unsigned __int128 x) { a[2 * i] = (uint64_t)x; a[2 * i + 1] = (uint64_t)(x >> 64); }
/* the scratch of row i of a chunk whose arrays hold `rows` rows of n nodes (structure of arrays) */
TR_NUM_HD tr_expl_scratch tr_expl_scratch_row(const tr_expl_scratch *all, uint64_t n, uint64_t i) {
    tr_expl_scratch s;
    s.newn = all->newn + 2 * n * i; s.sub = all->sub + 2 * n * i; s.add = all->add + 2 * n * i;
    s.dcnt = all->dcnt + n * i; s.heap = all->heap + n * i; s.touched = all->touched + n * i; s.flag = all->flag + n * i;
    return s;
}
TR_NUM_HD void tr_expl_heap_push(uint32_t *h, uint64_t *n, uint32_t x) {
    uint64_t i = (*n)++;
    while (i) {
        uint64_t p = (i - 1) / 2;
        if (h[p] <= x) break;
        h[i] = h[p];
        i = p;
    }
    h[i] = x;
}
TR_NUM_HD uint32_t tr_expl_heap_pop(uint32_t *h, uint64_t *n) {
    uint32_t top = h[0], x = h[--(*n)];
    uint64_t i = 0, m = *n;
    for (;;) {
        uint64_t c = 2 * i + 1;
        if (c >= m) break;
        if (c + 1 < m && h[c + 1] < h[c]) c++;
        if (h[c] >= x) break;
        h[i] = h[c];
        i = c;
    }
    if (m) h[i] = x;
    return top;
}
/* node c changed (its new N in newn, its new B in the BNEW flag): every parent entry is updated and queued */
TR_NUM_HD int tr_expl_notify(const tr_expl_view *s, const tr_expl_scratch *x, uint32_t c, int count, uint64_t *nh,
                             uint64_t *nt) {
    const unsigned __int128 top = ~(unsigned __int128)0;
    const int ob = s->base_b[c], nb = (x->flag[c] & TR_EXPL_BNEW) ? 1 : 0;
    int over = 0;
    for (uint64_t r = s->rev_at[c]; r < s->rev_at[c + 1]; r++) {
        const uint32_t q = s->rev[r];
        if (s->node[q].op == TR_PROV_PLUS) {
            if (count) {
                tr_expl_put(x->sub, q, tr_expl_get(x->sub, q) + tr_expl_get(s->base_n, c));   /* <= N(q): no wrap */
                unsigned __int128 a = tr_expl_get(x->add, q), y = tr_expl_get(x->newn, c);
                if (y > top - a) { x->flag[q] |= TR_EXPL_ADD_OVER; over = 1; }
                else tr_expl_put(x->add, q, a + y);
            }
            x->dcnt[q] += nb - ob;
        } else x->dcnt[q] += ob - nb;                      /* TIMES: the entries with B = 0 */
        if (!(x->flag[q] & TR_EXPL_QUEUED)) {
            x->flag[q] |= TR_EXPL_QUEUED;
            x->touched[(*nt)++] = q;
            tr_expl_heap_push(x->heap, nh, q);
        }
    }
    return over;
}
/* one pass over change set w (count: with N); returns 0 or -3 (an N overflow, count only); *nt touched entries */
TR_NUM_HD int32_t tr_expl_pass(const tr_expl_view *s, const tr_prov_valuations *v, uint64_t w, int count,
                               const tr_expl_scratch *x, uint64_t *nt, uint64_t *affected) {
    const unsigned __int128 top = ~(unsigned __int128)0;
    uint64_t nh = 0, aff = 0;
    *nt = 0;
    for (uint64_t k = v->offset[w]; k < v->offset[w + 1]; k++) {
        const uint32_t c = s->var_node[v->var[k]];
        if (c == TR_EXPL_NONE) continue;                  /* a variable outside the cone changes nothing */
        const unsigned __int128 nn = v->value[k];
        const uint8_t nb = v->value[k] != 0;
        if ((!count || nn == tr_expl_get(s->base_n, c)) && nb == s->base_b[c]) continue;
        x->flag[c] = (uint8_t)(TR_EXPL_CHANGED | (nb ? TR_EXPL_BNEW : 0));
        tr_expl_put(x->newn, c, nn);
        x->touched[(*nt)++] = c;
        aff++;
        if (tr_expl_notify(s, x, c, count, &nh, nt)) return -3;
    }
    while (nh) {
        const uint32_t p = tr_expl_heap_pop(x->heap, &nh);
        const tr_prov_node nd = s->node[p];
        aff++;
        const int64_t cnt = (int64_t)s->base_c[p] + x->dcnt[p];
        const uint8_t nb = nd.op == TR_PROV_PLUS ? cnt > 0 : cnt == 0;
        unsigned __int128 nn = 0;
        if (count) {
            if (nd.op == TR_PROV_PLUS) {
                if (x->flag[p] & TR_EXPL_ADD_OVER) return -3;
                nn = tr_expl_get(s->base_n, p) - tr_expl_get(x->sub, p);
                const unsigned __int128 a = tr_expl_get(x->add, p);
                if (a > top - nn) return -3;
                nn += a;
            } else {
                nn = 1;
                for (uint32_t e = 0; e < nd.n && nn; e++) {
                    const uint32_t c = s->child[nd.arg + e];
                    const unsigned __int128 y = (x->flag[c] & TR_EXPL_CHANGED) ? tr_expl_get(x->newn, c) : tr_expl_get(s->base_n, c);
                    if (y && nn > top / y) return -3;
                    nn *= y;
                }
            }
        }
        if ((!count || nn == tr_expl_get(s->base_n, p)) && nb == s->base_b[p]) continue;   /* unchanged: cut off */
        x->flag[p] |= (uint8_t)(TR_EXPL_CHANGED | (nb ? TR_EXPL_BNEW : 0));
        tr_expl_put(x->newn, p, nn);
        if (tr_expl_notify(s, x, p, count, &nh, nt)) return -3;
    }
    *affected = aff;
    return 0;
}
/* restore the touched scratch entries */
TR_NUM_HD void tr_expl_reset(const tr_expl_scratch *x, uint64_t nt) {
    for (uint64_t i = 0; i < nt; i++) {
        const uint32_t c = x->touched[i];
        x->flag[c] = 0; x->dcnt[c] = 0;
        tr_expl_put(x->sub, c, 0); tr_expl_put(x->add, c, 0);
    }
}
/* change set w: the answers' rows (out_n: 2 words per answer, out_b), its status (0 or -3), affected, changed */
TR_NUM_HD int32_t tr_expl_set(const tr_expl_view *s, const tr_prov_valuations *v, uint64_t w, const tr_expl_scratch *x,
                              uint64_t *out_n, uint8_t *out_b, uint64_t *affected, uint64_t *changed) {
    uint64_t nt = 0, aff = 0, ch = 0;
    int count = s->count_ok != 0;
    int32_t status = count ? 0 : -3;
    if (count && tr_expl_pass(s, v, w, 1, x, &nt, &aff)) {   /* an N overflow: redo the set in B alone */
        tr_expl_reset(x, nt);
        count = 0; status = -3;
    }
    if (!count) tr_expl_pass(s, v, w, 0, x, &nt, &aff);
    for (uint64_t j = 0; j < s->noutputs; j++) {
        const uint32_t c = s->output[j];
        out_n[2 * j] = count ? s->base_n[2 * (uint64_t)c] : 0;
        out_n[2 * j + 1] = count ? s->base_n[2 * (uint64_t)c + 1] : 0;
        out_b[j] = s->base_b[c];
    }
    for (uint64_t i = 0; i < nt; i++) {
        const uint32_t c = x->touched[i];
        if (!(x->flag[c] & TR_EXPL_CHANGED)) continue;
        for (uint64_t o = s->out_at[c]; o < s->out_at[c + 1]; o++) {
            const uint32_t j = s->outs[o];
            if (count) { out_n[2 * (uint64_t)j] = x->newn[2 * (uint64_t)c]; out_n[2 * (uint64_t)j + 1] = x->newn[2 * (uint64_t)c + 1]; }
            out_b[j] = (x->flag[c] & TR_EXPL_BNEW) ? 1 : 0;
            ch++;
        }
    }
    tr_expl_reset(x, nt);
    *affected = aff; *changed = ch;
    return status;
}

/* ------------------------------------------------------------------ host-side checks and sizes (both libraries) */
/* 0, or TR_MATH_INVALID for a malformed state view (the cone as PROV-001 checks plans, the reverse edges, the answer
 * lists, the variable map); O(nodes + children) without allocating */
static inline int32_t tr_expl_view_check(const tr_expl_view *s) {
    if (!s || !s->rev_at || !s->out_at || (s->nvars && !s->var_node) || (s->nodes && (!s->base_n || !s->base_b ||
        !s->base_c)) || (s->nchild && !s->rev) || (s->noutputs && !s->outs)) return -1;
    tr_prov_plan_view p;
    p.nodes = s->nodes; p.nchild = s->nchild; p.noutputs = s->noutputs; p.nvars = s->nvars;
    p.node = s->node; p.child = s->child; p.output = s->output;
    if (tr_prov_plan_view_check(&p)) return -1;
    if (s->rev_at[0] != 0 || s->rev_at[s->nodes] != s->nchild || s->out_at[0] != 0 || s->out_at[s->nodes] != s->noutputs)
        return -1;
    for (uint64_t i = 0; i < s->nodes; i++) {
        if (s->rev_at[i + 1] < s->rev_at[i] || s->out_at[i + 1] < s->out_at[i]) return -1;
        for (uint64_t r = s->rev_at[i]; r < s->rev_at[i + 1]; r++) {
            uint32_t q = s->rev[r];
            if ((uint64_t)q <= i || q >= s->nodes || (s->node[q].op != TR_PROV_PLUS && s->node[q].op != TR_PROV_TIMES))
                return -1;
        }
        for (uint64_t o = s->out_at[i]; o < s->out_at[i + 1]; o++)
            if (s->outs[o] >= s->noutputs || s->output[s->outs[o]] != (uint32_t)i) return -1;
        if (s->base_b[i] > 1) return -1;
    }
    for (uint64_t v = 0; v < s->nvars; v++) {
        uint32_t c = s->var_node[v];
        if (c != TR_EXPL_NONE && (c >= s->nodes || s->node[c].op != TR_PROV_VAR || s->node[c].arg != v)) return -1;
    }
    return 0;
}
/* 0, or TR_MATH_INVALID for change sets that are not well formed (PROV-001's check, and no fallback) */
static inline int32_t tr_expl_sets_check(const tr_prov_valuations *v, uint64_t nvars) {
    if (!v || v->fallback || tr_prov_valuations_check(v, nvars)) return -1;
    return 0;
}
/* change sets per device launch so that the scratch stays within TR_EXPL_DEVICE_SCRATCH (at least 1) */
static inline uint64_t tr_expl_device_chunk(const tr_expl_view *s, uint64_t count) {
    uint64_t per = tr_budget_mul(s->nodes ? s->nodes : 1, TR_EXPL_SCRATCH_PER_NODE);
    uint64_t c = per >= TR_EXPL_DEVICE_SCRATCH ? 1 : TR_EXPL_DEVICE_SCRATCH / per;
    return count < c ? (count ? count : 1) : c;
}
/* the bytes tr_mgpu_explain_propagate copies and allocates: the view's arrays, the change sets, the outputs and the
 * scratch chunk */
static inline uint64_t tr_expl_device_bytes(const tr_expl_view *s, const tr_prov_valuations *v, uint64_t chunk) {
    uint64_t overrides = v->count ? v->offset[v->count] : 0, n = s->nodes;
    uint64_t b = tr_budget_add(tr_budget_mul(n, 16 + 8 + 8 + 16 + 1 + 4), tr_budget_mul(s->nchild, 8));
    b = tr_budget_add(b, tr_budget_add(tr_budget_mul(s->noutputs, 8), tr_budget_mul(s->nvars, 4)));
    b = tr_budget_add(b, tr_budget_add(tr_budget_mul(tr_budget_add(v->count, 1), 8), tr_budget_mul(overrides, 12)));
    b = tr_budget_add(b, tr_budget_mul(tr_budget_mul(v->count, s->noutputs), 17));
    b = tr_budget_add(b, tr_budget_mul(v->count, sizeof(tr_expl_set_result)));
    return tr_budget_add(b, tr_budget_mul(tr_budget_mul(chunk, n), TR_EXPL_SCRATCH_PER_NODE));
}

/* ------------------------------------------------------------------ the state (libalg) */
/* the state of a plan's cone under a base valuation (count 1; NULL: every variable 1) */
tr_expl_state *tr_expl_create(const tr_prov_plan_view *, const tr_prov_valuations *base, const tr_math_budget *,
                              tr_math_refusal *, int32_t *error);
int32_t tr_expl_create_need(const tr_prov_plan_view *, tr_math_need *);
int32_t tr_expl_get_view(const tr_expl_state *, tr_expl_view *);
int32_t tr_expl_get_info(const tr_expl_state *, tr_expl_info *);
/* the base value of every variable (out[nvars]) */
int32_t tr_expl_base(const tr_expl_state *, uint64_t *out, uint64_t capacity);
void tr_expl_free(tr_expl_state *);

/* the needs of a propagation of v on the CPU (TR_EXPL_NEED_CPU) or the device (_DEVICE) */
int32_t tr_expl_propagate_need(const tr_expl_state *, const tr_prov_valuations *, int32_t target, tr_math_need *);
/* every change set of v (fallback NULL): out_n [count x noutputs x 2], out_b [count x noutputs], res [count] */
int32_t tr_expl_propagate(const tr_expl_state *, const tr_prov_valuations *, uint64_t *out_n, uint8_t *out_b,
                          tr_expl_set_result *res, const tr_math_budget *, tr_math_refusal *);

/* ------------------------------------------------------------------ explanations and probabilities (libalg) */
/* the decision diagram of answer j's existence over its cone's variables (levels: TR_SYMC_ORDER_DFS = first
 * appearance from the answer, _NATURAL = increasing variable, _GIVEN = order[nvars] restricted to the levels), with
 * the substitution TR_EXPL_FREE / _WHY (absent sources FALSE) / _WHYNOT (present sources TRUE) */
tr_symc_dd *tr_expl_lineage(const tr_expl_state *, uint32_t output, int32_t substitution, int32_t order_kind,
                            const uint32_t *order, const tr_math_budget *, tr_math_refusal *, int32_t *error);
tr_expl_sets *tr_expl_explain(const tr_expl_state *, uint32_t output, int32_t kind, uint64_t max_sets,
                              const tr_math_budget *, tr_math_refusal *, int32_t *error);
int32_t tr_expl_sets_get_info(const tr_expl_sets *, tr_expl_sets_info *);
/* offsets [nsets + 1] into vars [entries] (variable ids, increasing within a set) */
int32_t tr_expl_sets_get(const tr_expl_sets *, uint64_t *offsets, uint32_t *vars, uint64_t sets_capacity,
                         uint64_t vars_capacity);
void tr_expl_sets_free(tr_expl_sets *);
/* the literal weights (p, 1 - p) of the store's variables from p = num[v] / den[v] (0 <= num <= den, den >= 1) */
int32_t tr_expl_probability_weights(const tr_expl_state *, const int64_t *num, const int64_t *den,
                                    tr_symc_weight *out);
/* certified enclosure of P(answer j exists) on the DAG (no compilation); read_once: 1 when exact independence held at
 * every node */
int32_t tr_expl_probability_bounds(const tr_expl_state *, uint32_t output, const int64_t *num, const int64_t *den,
                                   tr_math_interval *out, int32_t *read_once, const tr_math_budget *,
                                   tr_math_refusal *);

/* ------------------------------------------------------------------ the CUDA library (math_explain_gpu.cu)
 * The same outputs as tr_expl_propagate from the view (host memory, copied): one work item per change set with its
 * own scratch row (tr_expl_set), in chunks of tr_expl_device_chunk sets; budget: output = sets x answers, bytes =
 * tr_expl_device_bytes. */
int tr_mgpu_explain_propagate(const tr_expl_view *, const tr_prov_valuations *, uint64_t *out_n, uint8_t *out_b,
                              tr_expl_set_result *res, const tr_math_budget *, tr_math_refusal *);
#ifdef __cplusplus
}
#endif
#endif
