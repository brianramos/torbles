/* Experimental native algebraic relationship optimizer. SPDX-License-Identifier: Apache-2.0
 *
 * The module equations are exact, table-scoped equalities out = rhs. Search uses both
 * orientations: substitute an output column by its rhs, or cancel an rhs monomial
 * with an integer multiple of the equation. A bounded beam keeps intermediate
 * states even when they cost more than the input, so independently proved modules
 * can compose. Only a final strictly cheaper polynomial replaces the input.
 *
 * No query-plan ABI changes or data writes occur here. The caller lowers typed
 * expressions to stored codes first and rebuilds its ordinary query from the result.
 */
#include "relationships.h"
#include <limits.h>
#include <stdlib.h>
#include <string.h>

#define TR_SEARCH_STATES 256
#define TR_BEAM 16
#define TR_POOL 128
#define TR_DEPTH 8

typedef struct {
    tr_poly p;
    double cost;
    int parent, rule, direction, depth;
} tr_state;

static int fits64(__int128 v) { return v >= INT64_MIN && v <= INT64_MAX; }

static int term_cmp(const void *pa, const void *pb) {
    const tr_term *a = pa, *b = pb;
    if (a->a != b->a) return a->a < b->a ? -1 : 1;
    return a->b == b->b ? 0 : a->b < b->b ? -1 : 1;
}

/* Combine first, never silently wrap an IR coefficient. Conservative failures
 * discard that search edge, retaining the original valid query. */
static int add_term(tr_poly *p, int a, int b, __int128 coefficient) {
    if (!coefficient) return 1;
    for (int i = 0; i < p->n; ++i) {
        if (p->terms[i].a != a || p->terms[i].b != b) continue;
        __int128 sum = (__int128)p->terms[i].coefficient + coefficient;
        if (!fits64(sum)) return 0;
        if (!sum) p->terms[i] = p->terms[--p->n];
        else p->terms[i].coefficient = (int64_t)sum;
        return 1;
    }
    if (!fits64(coefficient) || p->n == TR_MAX_TERMS) return 0;
    p->terms[p->n++] = (tr_term){a, b, (int64_t)coefficient};
    return 1;
}

static int normalize(const atable_t *t, const tr_poly *src, tr_poly *dst) {
    if (!src || src->n < 0 || src->n > TR_MAX_TERMS) return TR_INVALID;
    memset(dst, 0, sizeof *dst);
    for (int i = 0; i < src->n; ++i) {
        int a = src->terms[i].a, b = src->terms[i].b;
        if (a < -1 || b < -1 || a >= t->k || b >= t->k || (a == -1 && b != -1)) return TR_INVALID;
        if (b >= 0 && a > b) { int tmp = a; a = b; b = tmp; }
        if (!add_term(dst, a, b, src->terms[i].coefficient)) return TR_OVERFLOW;
    }
    qsort(dst->terms, (size_t)dst->n, sizeof dst->terms[0], term_cmp);
    return TR_OK;
}

static int same_poly(const tr_poly *a, const tr_poly *b) {
    if (a->n != b->n) return 0;
    for (int i = 0; i < a->n; ++i)
        if (a->terms[i].a != b->terms[i].a || a->terms[i].b != b->terms[i].b ||
            a->terms[i].coefficient != b->terms[i].coefficient) return 0;
    return 1;
}

double tr_poly_cost(const atable_t *t, const tr_poly *p) {
    if (!t || !p || p->n < 0 || p->n > TR_MAX_TERMS) return -1.0;
    double cost = 0.0;
    for (int i = 0; i < p->n; ++i) {
        int a = p->terms[i].a, b = p->terms[i].b;
        if (!p->terms[i].coefficient) continue;
        if (a == -1 && b == -1) cost += 1.0;
        else if (a < 0 || a >= t->k || b < -1 || b >= t->k) return -1.0;
        else if (b == -1) cost += t->width[a];
        else cost += (double)t->width[a] * t->width[b];
    }
    return cost;
}

static tr_poly variable(int col) {
    tr_poly p = {0}; p.n = 1; p.terms[0] = (tr_term){col, -1, 1}; return p;
}

static int multiply(const tr_poly *a, const tr_poly *b, tr_poly *out) {
    memset(out, 0, sizeof *out);
    for (int i = 0; i < a->n; ++i) for (int j = 0; j < b->n; ++j) {
        const tr_term *x = &a->terms[i], *y = &b->terms[j];
        int vars[4], n = 0;
        if (x->a >= 0) vars[n++] = x->a;
        if (x->b >= 0) vars[n++] = x->b;
        if (y->a >= 0) vars[n++] = y->a;
        if (y->b >= 0) vars[n++] = y->b;
        if (n > 2) return 0;
        if (n == 2 && vars[0] > vars[1]) { int v = vars[0]; vars[0] = vars[1]; vars[1] = v; }
        __int128 coef = (__int128)x->coefficient * y->coefficient;
        if (!add_term(out, n ? vars[0] : -1, n == 2 ? vars[1] : -1, coef)) return 0;
    }
    return 1;
}

/* Substitute the output in every monomial, including either factor of a square.
 * Degree growth and coefficient/term-budget overflow reject the entire edge. */
static int expand_rule(const tr_poly *p, int output, const tr_poly *rhs, tr_poly *out) {
    int touched = 0;
    memset(out, 0, sizeof *out);
    for (int i = 0; i < p->n; ++i) {
        const tr_term *term = &p->terms[i];
        tr_poly left, right, product;
        if (term->a < 0) {
            if (!add_term(out, -1, -1, term->coefficient)) return 0;
            continue;
        }
        if (term->a == output) { left = *rhs; touched = 1; }
        else left = variable(term->a);
        if (term->b >= 0) {
            if (term->b == output) { right = *rhs; touched = 1; }
            else right = variable(term->b);
            if (!multiply(&left, &right, &product)) return 0;
        } else product = left;
        for (int j = 0; j < product.n; ++j)
            if (!add_term(out, product.terms[j].a, product.terms[j].b,
                          (__int128)term->coefficient * product.terms[j].coefficient)) return 0;
    }
    if (!touched) return 0;
    qsort(out->terms, (size_t)out->n, sizeof out->terms[0], term_cmp);
    return 1;
}

/* Q - k*rhs + k*output. The exact ratio at an anchor monomial removes that
 * monomial; other terms may introduce work that a later module can eliminate. */
static int contract_rule(const tr_poly *p, int output, const tr_poly *rhs, int anchor,
                         int64_t k, tr_poly *out) {
    *out = *p;
    const tr_term *a = &rhs->terms[anchor];
    if (!add_term(out, a->a, a->b, -(__int128)k * a->coefficient)) return 0;
    for (int i = 0; i < rhs->n; ++i) if (i != anchor)
        if (!add_term(out, rhs->terms[i].a, rhs->terms[i].b,
                      -(__int128)k * rhs->terms[i].coefficient)) return 0;
    if (!add_term(out, output, -1, k)) return 0;
    qsort(out->terms, (size_t)out->n, sizeof out->terms[0], term_cmp);
    return 1;
}

static int already_seen(const tr_state *states, int n, const tr_poly *p) {
    for (int i = 0; i < n; ++i) if (same_poly(&states[i].p, p)) return 1;
    return 0;
}

static int state_better(const tr_state *a, const tr_state *b) {
    if (a->cost != b->cost) return a->cost < b->cost;
    if (a->p.n != b->p.n) return a->p.n < b->p.n;
    if (a->rule != b->rule) return a->rule < b->rule;
    return a->direction < b->direction;
}

static int state_cmp(const void *a, const void *b) {
    const tr_state *x = a, *y = b;
    return state_better(x, y) ? -1 : state_better(y, x) ? 1 : 0;
}

static void consider(const atable_t *t, const tr_poly *p, int parent, int rule, int direction,
                     int depth, const tr_state *states, int nstates,
                     tr_state *pool, int *npool, tr_trace *trace) {
    if (already_seen(states, nstates, p)) return;
    tr_state candidate = {*p, tr_poly_cost(t, p), parent, rule, direction, depth};
    for (int i = 0; i < *npool; ++i) if (same_poly(&pool[i].p, p)) return;
    if (*npool < TR_POOL) { pool[(*npool)++] = candidate; return; }
    trace->truncated = 1;
    int worst = 0;
    for (int i = 1; i < *npool; ++i) if (state_better(&pool[worst], &pool[i])) worst = i;
    if (state_better(&candidate, &pool[worst])) pool[worst] = candidate;
}

int32_t tr_optimize(const tr_catalog *catalog, const atable_t *table, const uint64_t *used,
                    const tr_poly *input, tr_poly *output, tr_trace *trace) {
    if (!catalog || !table || !input || !output || !trace) return TR_INVALID;
    memset(trace, 0, sizeof *trace);
    *output = *input;
    if (!tr_scope_valid(catalog, table, used)) return TR_STALE;
    tr_poly normalized;
    int rc = normalize(table, input, &normalized);
    if (rc != TR_OK) return rc;
    trace->before_cost = trace->after_cost = tr_poly_cost(table, &normalized);

    tr_state *states = calloc(TR_SEARCH_STATES, sizeof *states);
    tr_state *pool = calloc(TR_POOL, sizeof *pool);
    tr_poly *rhs = calloc(TR_MAX_RULES, sizeof *rhs);
    unsigned char valid[TR_MAX_RULES] = {0};
    if (!states || !pool || !rhs) { free(states); free(pool); free(rhs); return TR_MEMORY; }
    if (catalog->nrules < 0 || catalog->nrules > TR_MAX_RULES) {
        free(states); free(pool); free(rhs); return TR_INVALID;
    }
    for (int r = 0; r < catalog->nrules; ++r) {
        const tr_rule *rule = &catalog->rules[r];
        valid[r] = rule->usable && rule->out >= 0 && rule->out < table->k &&
                   normalize(table, &rule->rhs, &rhs[r]) == TR_OK;
    }
    states[0] = (tr_state){normalized, trace->before_cost, -1, -1, 0, 0};
    int frontier[TR_BEAM] = {0}, nfrontier = 1, nstates = 1, best = 0;
    for (int depth = 1; depth <= TR_DEPTH && nfrontier; ++depth) {
        int npool = 0;
        for (int fi = 0; fi < nfrontier; ++fi) {
            int parent = frontier[fi];
            const tr_poly *p = &states[parent].p;
            for (int r = 0; r < catalog->nrules; ++r) if (valid[r]) {
                tr_poly next;
                const tr_rule *rule = &catalog->rules[r];
                if (expand_rule(p, rule->out, &rhs[r], &next))
                    consider(table, &next, parent, r, 1, depth, states, nstates, pool, &npool, trace);
                for (int j = 0; j < rhs[r].n; ++j) {
                    const tr_term *term = &rhs[r].terms[j];
                    if (term->a < 0) continue; /* constant-only contractions add work */
                    for (int i = 0; i < p->n; ++i) {
                        const tr_term *anchor = &p->terms[i];
                        if (anchor->a != term->a || anchor->b != term->b) continue;
                        __int128 ratio = (__int128)anchor->coefficient / term->coefficient;
                        if (!ratio || !fits64(ratio) || (__int128)anchor->coefficient % term->coefficient) continue;
                        if (contract_rule(p, rule->out, &rhs[r], j, (int64_t)ratio, &next))
                            consider(table, &next, parent, r, -1, depth, states, nstates, pool, &npool, trace);
                    }
                }
            }
        }
        qsort(pool, (size_t)npool, sizeof *pool, state_cmp);
        nfrontier = npool < TR_BEAM ? npool : TR_BEAM;
        if (npool > nfrontier) trace->truncated = 1;
        for (int i = 0; i < nfrontier; ++i) {
            if (nstates == TR_SEARCH_STATES) { trace->truncated = 1; nfrontier = i; break; }
            states[nstates] = pool[i]; frontier[i] = nstates;
            if (states[nstates].cost < states[best].cost) best = nstates;
            ++nstates;
        }
        if (depth == TR_DEPTH && nfrontier) trace->truncated = 1;
    }
    trace->explored = nstates;
    if (best != 0 && states[best].cost < trace->before_cost) {
        *output = states[best].p;
        trace->changed = 1; trace->after_cost = states[best].cost;
        int rule_path[TR_MAX_STEPS], direction_path[TR_MAX_STEPS], steps = 0;
        for (int cur = best; states[cur].parent >= 0 && steps < TR_MAX_STEPS; cur = states[cur].parent) {
            rule_path[steps] = states[cur].rule; direction_path[steps++] = states[cur].direction;
        }
        trace->steps = steps;
        for (int i = 0; i < steps; ++i) {
            trace->rule[i] = rule_path[steps - 1 - i];
            trace->direction[i] = direction_path[steps - 1 - i];
        }
    }
    free(states); free(pool); free(rhs);
    return TR_OK;
}
