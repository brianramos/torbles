/* Bounded native algebraic regions and shared scalar intermediates.
 * SPDX-License-Identifier: Apache-2.0
 *
 * A region is combined only through checked integer polynomial identities.
 * Higher-degree surroundings stay as operations. The selected DAG is lowered to
 * the same row executor's bytecode, with SAVE/LOAD for common arithmetic nodes;
 * this does not materialize bit-plane intermediates or temporary relations.
 */
#include "relationships.h"
#include <limits.h>
#include <stdlib.h>
#include <string.h>

#define DAG_MAX 1024
#define DAG_PASSES 8

int32_t tr_measure_valid(const qplan_t *q, int32_t columns) {
    if (!q || columns < 1 || columns > MAXCOL || q->nmeas < 0 || q->nmeas > 48 ||
        q->nattr < 0 || q->nattr > 8) return 0;
    int sp = 0; uint64_t initialized = 0;
    for (int i = 0; i < q->nmeas; ++i) {
        const minstr_t *m = &q->meas[i];
        switch (m->op) {
        case M_FCOL:
            if (m->arg < 0 || m->arg >= columns || sp == 24) return 0;
            ++sp; break;
        case M_ATTR:
            if (m->arg < 0 || m->arg >= q->nattr || sp == 24) return 0;
            ++sp; break;
        case M_CONST:
            if (sp == 24) return 0;
            ++sp; break;
        case M_ADD: case M_SUB: case M_MUL:
            if (sp < 2) return 0;
            --sp; break;
        case M_SAVE:
            if (!sp || m->arg < 0 || m->arg >= 48) return 0;
            initialized |= UINT64_C(1) << m->arg; break;
        case M_LOAD:
            if (sp == 24 || m->arg < 0 || m->arg >= 48 ||
                !(initialized >> m->arg & 1)) return 0;
            ++sp; break;
        default: return 0;
        }
    }
    return !q->nmeas || sp == 1;
}

typedef struct {
    int op, arg, left, right, exact;
    int64_t constant;
    tr_poly poly;
} dag_node;
typedef struct {
    int n, root, replacements[DAG_MAX];
    dag_node nodes[DAG_MAX];
} dag;
typedef struct {
    const dag *d;
    qplan_t *out;
    int refs[DAG_MAX], registers[DAG_MAX], nregisters;
    unsigned char visited[DAG_MAX], active[DAG_MAX], stored[DAG_MAX], regions[DAG_MAX];
    int sp, peak, operations, nodes;
    uint64_t inputs;
} lowering;

static int fits(__int128 x) { return x >= INT64_MIN && x <= INT64_MAX; }

static int term_add(tr_poly *p, int a, int b, __int128 c) {
    if (!c) return 1;
    if (b >= 0 && a > b) { int swap = a; a = b; b = swap; }
    for (int i = 0; i < p->n; ++i) if (p->terms[i].a == a && p->terms[i].b == b) {
        c += p->terms[i].coefficient;
        if (!fits(c)) return 0;
        if (!c) p->terms[i] = p->terms[--p->n];
        else p->terms[i].coefficient = (int64_t)c;
        return 1;
    }
    if (!fits(c) || p->n == TR_MAX_TERMS) return 0;
    p->terms[p->n++] = (tr_term){a, b, (int64_t)c}; return 1;
}

static int polynomial(const tr_poly *a, const tr_poly *b, int op, tr_poly *out) {
    memset(out, 0, sizeof *out);
    if (op != M_MUL) {
        for (int i = 0; i < a->n; ++i)
            if (!term_add(out, a->terms[i].a, a->terms[i].b, a->terms[i].coefficient)) return 0;
        for (int i = 0; i < b->n; ++i)
            if (!term_add(out, b->terms[i].a, b->terms[i].b,
                          (__int128)b->terms[i].coefficient * (op == M_SUB ? -1 : 1))) return 0;
        return 1;
    }
    for (int i = 0; i < a->n; ++i) for (int j = 0; j < b->n; ++j) {
        const tr_term *x = &a->terms[i], *y = &b->terms[j]; int vars[4], degree = 0;
        if (x->a >= 0) vars[degree++] = x->a;
        if (x->b >= 0) vars[degree++] = x->b;
        if (y->a >= 0) vars[degree++] = y->a;
        if (y->b >= 0) vars[degree++] = y->b;
        if (degree > 2 || !term_add(out, degree ? vars[0] : -1, degree == 2 ? vars[1] : -1,
                                  (__int128)x->coefficient * y->coefficient)) return 0;
    }
    return 1;
}

/* Hash-cons by exact structure. Commutative operands have a stable order. The
 * source budget is tiny, so a bounded linear table avoids another allocation. */
static int intern(dag *d, int op, int arg, int64_t c, int left, int right) {
    if ((op == M_ADD || op == M_MUL) && left > right) { int s = left; left = right; right = s; }
    for (int i = 0; i < d->n; ++i) {
        const dag_node *n = &d->nodes[i];
        if (n->op == op && n->arg == arg && n->constant == c && n->left == left && n->right == right) return i;
    }
    if (d->n == DAG_MAX || ((op == M_ADD || op == M_SUB || op == M_MUL) && (left < 0 || right < 0))) return -1;
    int id = d->n++; dag_node *n = &d->nodes[id];
    memset(n, 0, sizeof *n);
    n->op = op; n->arg = arg; n->constant = c; n->left = left; n->right = right;
    d->replacements[id] = id;
    if (op == M_FCOL || op == M_CONST) {
        n->exact = 1;
        term_add(&n->poly, op == M_FCOL ? arg : -1, -1, op == M_FCOL ? 1 : c);
    } else if (d->nodes[left].exact && d->nodes[right].exact)
        n->exact = polynomial(&d->nodes[left].poly, &d->nodes[right].poly, op, &n->poly);
    return id;
}

static int parse(dag *d, const qplan_t *q) {
    int stack[24], registers[48], sp = 0;
    for (int i = 0; i < q->nmeas; ++i) {
        const minstr_t *m = &q->meas[i]; int id;
        if (m->op == M_SAVE) { registers[m->arg] = stack[sp-1]; continue; }
        if (m->op == M_LOAD) { stack[sp++] = registers[m->arg]; continue; }
        if (m->op == M_FCOL || m->op == M_CONST) id = intern(d, m->op, m->op == M_FCOL ? m->arg : 0, m->op == M_CONST ? m->c : 0, -1, -1);
        else if (m->op == M_ADD || m->op == M_SUB || m->op == M_MUL) {
            int right = stack[--sp], left = stack[--sp];
            id = intern(d, m->op, 0, 0, left, right);
        } else return 0;
        if (id < 0) return 0;
        stack[sp++] = id;
    }
    if (sp != 1) return 0;
    d->root = stack[0]; return 1;
}

static int term_order(const void *a, const void *b) {
    const tr_term *x = a, *y = b;
    if (x->a != y->a) return (x->a > y->a) - (x->a < y->a);
    return (x->b > y->b) - (x->b < y->b);
}

static int normalized(dag *d, const tr_poly *poly) {
    tr_poly p = *poly; qsort(p.terms, (size_t)p.n, sizeof p.terms[0], term_order);
    int result = -1;
    for (int i = 0; i < p.n; ++i) {
        const tr_term *t = &p.terms[i]; int term;
        int subtract = result >= 0 && t->coefficient < 0 && t->coefficient != INT64_MIN;
        int64_t c = subtract ? -t->coefficient : t->coefficient;
        if (t->a < 0) term = intern(d, M_CONST, 0, c, -1, -1);
        else {
            term = intern(d, M_FCOL, t->a, 0, -1, -1);
            if (t->b >= 0) term = intern(d, M_MUL, 0, 0, term, intern(d, M_FCOL, t->b, 0, -1, -1));
            if (c != 1) term = intern(d, M_MUL, 0, 0, term, intern(d, M_CONST, 0, c, -1, -1));
        }
        if (term < 0) return -1;
        result = result < 0 ? term : intern(d, subtract ? M_SUB : M_ADD, 0, 0, result, term);
        if (result < 0) return -1;
    }
    return result < 0 ? intern(d, M_CONST, 0, 0, -1, -1) : result;
}

static int resolve(lowering *l, int id) {
    for (int n = 0; n < DAG_MAX; ++n) {
        int next = l->d->replacements[id];
        if (next == id) return id;
        l->regions[id] = 1; id = next;
    }
    return -1;
}

static int references(lowering *l, int id) {
    id = resolve(l, id);
    if (id < 0 || l->active[id]) return 0;
    ++l->refs[id];
    if (l->visited[id]) return 1;
    l->visited[id] = l->active[id] = 1; ++l->nodes;
    const dag_node *n = &l->d->nodes[id];
    if (n->op == M_FCOL) l->inputs |= UINT64_C(1) << n->arg;
    else if (n->op != M_CONST && (!references(l, n->left) || !references(l, n->right))) return 0;
    l->active[id] = 0; return 1;
}

static int output(lowering *l, int op, int arg, int64_t c) {
    if (l->out->nmeas == 48) return 0;
    l->out->meas[l->out->nmeas++] = (minstr_t){op, arg, c};
    if (op == M_FCOL || op == M_CONST || op == M_LOAD) {
        if (++l->sp > 24) return 0;
        if (l->sp > l->peak) l->peak = l->sp;
    } else if (op != M_SAVE) { --l->sp; ++l->operations; }
    return 1;
}

static int lower_node(lowering *l, int id) {
    id = resolve(l, id);
    if (id < 0) return 0;
    const dag_node *n = &l->d->nodes[id];
    if (l->stored[id]) return output(l, M_LOAD, l->registers[id], 0);
    if (n->op == M_FCOL || n->op == M_CONST) return output(l, n->op, n->arg, n->constant);
    if (!lower_node(l, n->left) || !lower_node(l, n->right) || !output(l, n->op, 0, 0)) return 0;
    if (l->refs[id] > 1) {
        if (l->nregisters == 48) return 0;
        int reg = l->nregisters++;
        if (!output(l, M_SAVE, reg, 0)) return 0;
        l->registers[id] = reg; l->stored[id] = 1;
    }
    return 1;
}

static int compile(const dag *d, qplan_t *out, tr_dag_stats *stats) {
    lowering l; memset(&l, 0, sizeof l); l.d = d; l.out = out;
    out->nmeas = 0;
    if (!references(&l, d->root) || !lower_node(&l, d->root) || l.sp != 1) return 0;
    memset(stats, 0, sizeof *stats);
    stats->nodes = l.nodes; stats->operations = l.operations; stats->registers = l.nregisters;
    stats->stack_peak = l.peak; stats->inputs = l.inputs;
    for (int i = 0; i < d->n; ++i) stats->regions += l.regions[i] != 0;
    out->agg = A_GENERAL; return 1;
}

/* Local search uses the same scalar/stack weights as the physical chooser, with
 * a per-row extraction term for surviving input bits. Final sampled costs still
 * compare this candidate against the original chain and whole-system kernels. */
static double work(const atable_t *t, const qplan_t *q, const tr_dag_stats *s) {
    double bits = 0;
    for (int c = 0; c < t->k; ++c) if (s->inputs >> c & 1) bits += t->width[c];
    return s->operations*1.5 + s->stack_peak*0.1 + q->nmeas*0.15 + bits*0.5;
}

int32_t tr_dag_optimize(const atable_t *t, const qplan_t *source, qplan_t *out, tr_dag_stats *stats) {
    if (!t || !source || !out || !stats || !source->nmeas || !tr_measure_valid(source, t->k)) return TR_INVALID;
    dag *d = calloc(1, sizeof *d);
    if (!d) return TR_MEMORY;
    if (!parse(d, source)) { free(d); return TR_INVALID; }
    int originals = d->n, alternatives[48];
    for (int i = 0; i < originals; ++i) {
        dag_node *n = &d->nodes[i];
        alternatives[i] = n->exact && n->op >= M_ADD ? normalized(d, &n->poly) : -1;
    }
    *out = *source;
    if (!compile(d, out, stats)) { free(d); return TR_OVERFLOW; }
    double best = work(t, out, stats);
    for (int pass = 0; pass < DAG_PASSES; ++pass) {
        int changed = 0;
        for (int i = 0; i < originals; ++i) {
            int alt = alternatives[i], old = d->replacements[i];
            if (alt < 0 || alt == i || alt == old) continue;
            d->replacements[i] = alt;
            qplan_t trial = *source; tr_dag_stats next;
            if (compile(d, &trial, &next)) {
                double cost = work(t, &trial, &next);
                if (cost < best) { *out = trial; *stats = next; best = cost; changed = 1; continue; }
            }
            d->replacements[i] = old;
        }
        if (!changed) break;
    }
    free(d);
    return tr_measure_valid(out, t->k) ? TR_OK : TR_INVALID;
}
