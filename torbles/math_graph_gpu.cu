/* GRAPH-001 / GRAPH-002 on a CUDA device: degrees, frontier expansion and the semiring products (one work item per
 * vertex or frontier word), the fixed points (reach, closure, bounded walk counts, Bellman-Ford: one launch sequence
 * per round over the round's candidate list), the exact Dirichlet energy and the PageRank / power iterations, with
 * exactly the CPU's results (math_graph.h).
 * SPDX-License-Identifier: Apache-2.0
 *
 * Thread-order independence: every per-vertex result is computed by one work item from the shared inline pulls of
 * math_graph.h (the same source as the CPU loops, entries in their stored order) and written to its own slot. Shared
 * words are combined with commutative integer atomics only: OR into bitsets, integer adds into counters and into the
 * multiword exact accumulators (walk length totals, the 320-bit energy), minimum for first violations and witnesses.
 * A round's candidates are the distinct push neighbors of the vertices that changed: a vertex is appended to the list
 * by the work item whose atomic add on its touch counter returns 0, so the list's order depends on the threads but
 * the set does not, and nothing reads the order. Floating sums are fixed blocked sums (one work item per block, in
 * index order; the block sums in order by one work item), exactly the CPU's operations. */
#include "math_cuda.h"
#include "math_graph.h"
#include "math_handles.h"

TR_MGPU_ABI(tr_graph_view, TR_GRAPH_ABI);          /* structs passed to the CUDA library (math_handles.h) */
TR_MGPU_ABI(tr_graph_rounds, TR_GRAPH_ABI);
TR_MGPU_ABI(tr_graph_walks_info, TR_GRAPH_ABI);
TR_MGPU_ABI(tr_graph_paths_info, TR_GRAPH_ABI);
TR_MGPU_ABI(tr_graph_iter_info, TR_GRAPH_ABI);

/* ------------------------------------------------------------------ device helpers */
/* acc[0..na) += x[0..nx) with per-limb atomic adds and their carries (exact modulo 2^(64 na): every carry is
 * propagated once, whatever the interleaving) */
TR_DEVICE void graph_atomic_add_limbs(uint64_t *acc, uint32_t na, const uint64_t *x, uint32_t nx) {
    uint64_t carry = 0;
    for (uint32_t i = 0; i < na; i++) {
        uint64_t xi = i < nx ? x[i] : 0;
        if (!xi && !carry) { if (i >= nx) break; continue; }
        uint64_t a = xi + carry, c1 = a < xi, c2 = 0;
        if (a) { uint64_t old = tr_atomic_add_u64(&acc[i], a); c2 = (old + a) < old; }
        carry = c1 + c2;
    }
}
TR_DEVICE void graph_set_bit(uint64_t *b, uint64_t i) { tr_atomic_or_u64(&b[i >> 6], (uint64_t)1 << (i & 63)); }

/* the device copies of a call (freed together) */
typedef struct { void *p[48]; int n; int rc; } graph_mem;
static void *graph_dev(graph_mem *m, uint64_t bytes) {
    void *p = NULL;
    if (m->rc) return NULL;
    if (m->n >= 48) { m->rc = TR_GRAPH_E_MEMORY; return NULL; }
    m->rc = tr_cuda_alloc(&p, (size_t)(bytes ? bytes : 8));
    if (m->rc) return NULL;
    m->p[m->n++] = p;
    return p;
}
static void *graph_dev_copy(graph_mem *m, const void *src, uint64_t bytes) {
    void *p = graph_dev(m, bytes);
    if (p && bytes && src) m->rc = tr_cuda_copy_in(p, src, (size_t)bytes);
    return p;
}
static void graph_free(graph_mem *m) { for (int i = 0; i < m->n; i++) tr_cuda_free(m->p[i]); m->n = 0; }
static int graph_sync(graph_mem *m) { if (!m->rc) m->rc = tr_cuda_sync(); return m->rc; }
static int graph_out(graph_mem *m, void *dst, const void *src, uint64_t bytes) {
    if (!m->rc && bytes) m->rc = tr_cuda_copy_out(dst, src, (size_t)bytes);
    return m->rc;
}
static int graph_in(graph_mem *m, void *dst, const void *src, uint64_t bytes) {
    if (!m->rc && bytes) m->rc = tr_cuda_copy_in(dst, src, (size_t)bytes);
    return m->rc;
}
static int32_t graph_budget(const tr_math_budget *budget, const tr_math_need *need, tr_math_refusal *refusal) {
    tr_math_budget dflt;
    if (!budget) { tr_budget_default(&dflt); budget = &dflt; }
    return tr_budget_check(budget, need, refusal);
}
static int graph_canonical(const uint64_t *b, uint64_t n) {
    uint64_t nw = tr_graph_words(n);
    return b && (!(n % 64) || !nw || !(b[nw - 1] >> (n % 64)));
}
/* a device copy of the view: the in-form and / or the out-form (once when they are shared) and the edges, as asked;
 * the forms not copied are NULL */
static void graph_upload(graph_mem *m, const tr_graph_view *g, tr_graph_view *d, int in, int out, int edges) {
    *d = *g;
    const uint64_t n = g->n, E = g->entries, M = g->edges;
    const int shared = g->out_off == g->in_off;
    d->in_off = d->out_off = NULL; d->in_nbr = d->out_nbr = NULL; d->in_w = d->out_w = NULL; d->in_m = d->out_m = NULL;
    if (in || (out && shared)) {
        d->in_off = (const uint64_t *)graph_dev_copy(m, g->in_off, (n + 1) * 8);
        d->in_nbr = (const uint32_t *)graph_dev_copy(m, g->in_nbr, E * 4);
        d->in_w = (const int64_t *)graph_dev_copy(m, g->in_w, E * 8);
        d->in_m = (const uint64_t *)graph_dev_copy(m, g->in_m, E * 8);
    }
    if (out && shared) { d->out_off = d->in_off; d->out_nbr = d->in_nbr; d->out_w = d->in_w; d->out_m = d->in_m; }
    else if (out) {
        d->out_off = (const uint64_t *)graph_dev_copy(m, g->out_off, (n + 1) * 8);
        d->out_nbr = (const uint32_t *)graph_dev_copy(m, g->out_nbr, E * 4);
        d->out_w = (const int64_t *)graph_dev_copy(m, g->out_w, E * 8);
        d->out_m = (const uint64_t *)graph_dev_copy(m, g->out_m, E * 8);
    }
    d->e_tail = d->e_head = NULL; d->e_w = NULL; d->e_m = NULL;
    if (edges) {
        d->e_tail = (const uint32_t *)graph_dev_copy(m, g->e_tail, M * 4);
        d->e_head = (const uint32_t *)graph_dev_copy(m, g->e_head, M * 4);
        d->e_w = (const int64_t *)graph_dev_copy(m, g->e_w, M * 8);
        d->e_m = (const uint64_t *)graph_dev_copy(m, g->e_m, M * 8);
    }
}
/* the forms a direction's pull and push read: forward pulls the in-form and pushes the out-form */
#define GRAPH_PULL_IN(dir) ((dir) == TR_GRAPH_FORWARD)
#define GRAPH_PUSH_IN(dir) ((dir) == TR_GRAPH_BACKWARD)

/* ------------------------------------------------------------------ kernels: GRAPH-001 */
TR_KERNEL void graph_degrees_kernel(tr_graph_view g, int32_t kind, int32_t weights, tr_math_i128 *out, uint64_t *bad) {
    TR_GRID_LOOP(x, g.n) {
        tr_graph_adj in = tr_graph_pull(&g, TR_GRAPH_FORWARD), outf = tr_graph_pull(&g, TR_GRAPH_BACKWARD);
        __int128 a = 0, b = 0;
        int ok;
        if (kind == TR_GRAPH_DEG_OUT) ok = tr_graph_row_degree(outf, x, weights, 0, &a);
        else if (kind == TR_GRAPH_DEG_IN) ok = tr_graph_row_degree(in, x, weights, 0, &a);
        else if (kind == TR_GRAPH_DEG_LOOPS) ok = tr_graph_row_degree(outf, x, weights, 1, &a);
        else {
            ok = tr_graph_row_degree(outf, x, weights, 0, &a);
            if (ok && g.directed) ok = tr_graph_row_degree(in, x, weights, 0, &b) && tr_graph_add128(a, b, &a);
        }
        if (!ok) tr_atomic_min_u64(bad, x);
        else out[x] = tr_i128_set(a);
    }
}
TR_KERNEL void graph_push_kernel(tr_graph_adj push, const uint64_t *front, uint64_t nw, uint64_t *next) {
    TR_GRID_LOOP(w, nw) {
        uint64_t bits = front[w];
        while (bits) {
            uint64_t u = w * 64 + (uint64_t)tr_graph_ctz(bits);
            bits &= bits - 1;
            for (uint64_t e = push.off[u]; e < push.off[u + 1]; e++) graph_set_bit(next, push.nbr[e]);
        }
    }
}
TR_KERNEL void graph_mask_kernel(uint64_t *next, const uint64_t *exclude, uint64_t nw, uint64_t *count) {
    TR_GRID_LOOP(w, nw) {
        uint64_t x = next[w];
        if (exclude) { x &= ~exclude[w]; next[w] = x; }
        int c = tr_graph_popc(x);
        if (c) tr_atomic_add_u64(count, (uint64_t)c);
    }
}
TR_KERNEL void graph_spmv_kernel(tr_graph_adj a, uint64_t n, int32_t semiring, int32_t weights, const void *x,
                                 const uint64_t *xv, void *y, uint64_t *yv, double *err, const double *coef,
                                 uint64_t *where) {
    TR_GRID_LOOP(u, n) {
        int has = 0, ok = 1;
        if (semiring == TR_GRAPH_COUNT) {
            __int128 r;
            ok = tr_graph_row_count(a, u, weights, (const int64_t *)x, xv, &r, &has);
            if (ok) ((tr_math_i128 *)y)[u] = tr_i128_set(r);
        } else if (semiring == TR_GRAPH_BOOL) {
            if (tr_graph_row_bool(a, u, (const uint64_t *)x, xv, &has)) graph_set_bit((uint64_t *)y, u);
        } else if (semiring == TR_GRAPH_GF2) {
            if (tr_graph_row_gf2(a, u, weights, (const uint64_t *)x, xv, &has)) graph_set_bit((uint64_t *)y, u);
        } else if (semiring == TR_GRAPH_MIN_PLUS || semiring == TR_GRAPH_MAX_PLUS) {
            int64_t r;
            ok = tr_graph_row_tropical(a, u, weights, semiring == TR_GRAPH_MAX_PLUS, (const int64_t *)x, xv, &r, &has);
            if (ok) ((int64_t *)y)[u] = has ? r : 0;
        } else {
            double e;
            ((double *)y)[u] = tr_graph_row_float(a, u, weights, coef, (const double *)x, xv, &e, &has);
            err[u] = e;
        }
        if (!ok) tr_atomic_min_u64(where, u);
        else if (has && yv) graph_set_bit(yv, u);
    }
}

/* ------------------------------------------------------------------ kernels: frontier rounds (GRAPH-002) */
/* counters: [0] list size, [1] work, [2] second list size, [3] witness / first violation, [4] flags, [5] maxbits,
 * [6] second violation */
TR_KERNEL void graph_reach_kernel(tr_graph_adj push, const uint32_t *F, uint64_t fn, uint64_t *vis, uint32_t *level,
                                  uint32_t round, uint32_t *N, uint64_t *counters) {
    TR_GRID_LOOP(i, fn) {
        uint32_t u = F[i];
        tr_atomic_add_u64(&counters[1], push.off[u + 1] - push.off[u]);
        for (uint64_t e = push.off[u]; e < push.off[u + 1]; e++) {
            uint32_t x = push.nbr[e];
            uint64_t bit = (uint64_t)1 << (x & 63);
            if (!(tr_atomic_or_u64(&vis[x >> 6], bit) & bit)) {
                level[x] = round;
                N[tr_atomic_add_u64(&counters[0], 1)] = x;
            }
        }
    }
}
TR_KERNEL void graph_cand_kernel(tr_graph_adj push, const uint32_t *A, uint64_t na, uint64_t *touch, uint32_t *C,
                                 uint64_t *counters) {
    TR_GRID_LOOP(i, na) {
        uint32_t u = A[i];
        tr_atomic_add_u64(&counters[1], push.off[u + 1] - push.off[u]);
        for (uint64_t e = push.off[u]; e < push.off[u + 1]; e++) {
            uint32_t x = push.nbr[e];
            if (tr_atomic_add_u64(&touch[x], 1) == 0) C[tr_atomic_add_u64(&counters[0], 1)] = x;
        }
    }
}
TR_KERNEL void graph_lanes_kernel(tr_graph_adj pull, const uint32_t *C, uint64_t nc, const uint64_t *F,
                                  const uint64_t *R, uint64_t *slot, uint64_t *counters) {
    TR_GRID_LOOP(i, nc) {
        uint32_t c = C[i];
        tr_atomic_add_u64(&counters[1], pull.off[c + 1] - pull.off[c]);
        slot[i] = tr_graph_row_lanes(pull, c, F) & ~R[c];
    }
}
TR_KERNEL void graph_clear_lanes_kernel(const uint32_t *A, uint64_t na, uint64_t *F) {
    TR_GRID_LOOP(i, na) F[A[i]] = 0;
}
TR_KERNEL void graph_commit_lanes_kernel(const uint32_t *C, uint64_t nc, const uint64_t *slot, uint64_t *R,
                                         uint64_t *F, uint64_t *touch, uint32_t *NA, uint64_t *counters) {
    TR_GRID_LOOP(i, nc) {
        uint32_t c = C[i];
        touch[c] = 0;
        if (slot[i]) {
            R[c] |= slot[i];
            F[c] = slot[i];
            NA[tr_atomic_add_u64(&counters[2], 1)] = c;
        }
    }
}
TR_KERNEL void graph_walk_kernel(tr_graph_adj pull, const uint32_t *C, uint64_t nc, uint32_t W, const uint64_t *prev,
                                 uint64_t *cur, const uint64_t *bound, uint64_t *counters) {
    TR_GRID_LOOP(i, nc) {
        uint32_t c = C[i];
        uint64_t acc[TR_GRAPH_MAX_LIMBS + 2];
        tr_atomic_add_u64(&counters[1], pull.off[c + 1] - pull.off[c]);
        tr_graph_row_walk(pull, c, W, prev, acc);
        tr_atomic_max_u64(&counters[5], tr_graph_limbs_bits(acc, W + 2));
        if (!tr_graph_limbs_le(acc, W + 2, bound, W)) { tr_atomic_min_u64(&counters[3], c); continue; }
        for (uint32_t j = 0; j < W; j++) cur[(uint64_t)c * W + j] = acc[j];
    }
}
TR_KERNEL void graph_walk_commit_kernel(const uint32_t *C, uint64_t nc, uint32_t W, const uint64_t *cur,
                                        uint64_t *totals, uint64_t *row, uint64_t *lsum, const uint64_t *bound,
                                        uint64_t *touch, uint64_t *counters) {
    TR_GRID_LOOP(i, nc) {
        uint32_t c = C[i];
        uint64_t tmp[TR_GRAPH_MAX_LIMBS + 1];
        const uint64_t *x = cur + (uint64_t)c * W;
        touch[c] = 0;
        for (uint32_t j = 0; j < W; j++) tmp[j] = totals[(uint64_t)c * W + j];
        tmp[W] = 0;
        tr_graph_limbs_add(tmp, W + 1, x, W);
        if (!tr_graph_limbs_le(tmp, W + 1, bound, W)) { tr_atomic_min_u64(&counters[6], c); continue; }
        for (uint32_t j = 0; j < W; j++) totals[(uint64_t)c * W + j] = tmp[j];
        if (row) for (uint32_t j = 0; j < W; j++) row[(uint64_t)c * W + j] = x[j];
        graph_atomic_add_limbs(lsum, W + 2, x, W);
    }
}
TR_KERNEL void graph_zero_rows_kernel(const uint32_t *A, uint64_t na, uint32_t W, uint64_t *v) {
    TR_GRID_LOOP(i, na) for (uint32_t j = 0; j < W; j++) v[(uint64_t)A[i] * W + j] = 0;
}
TR_KERNEL void graph_relax_kernel(tr_graph_adj pull, const uint32_t *C, uint64_t nc, int32_t maxplus, const int64_t *d,
                                  const uint64_t *fin, int64_t *nd, uint32_t *np, uint8_t *imp, uint64_t *counters) {
    TR_GRID_LOOP(i, nc) {
        uint32_t c = C[i];
        int im = 0;
        tr_atomic_add_u64(&counters[1], pull.off[c + 1] - pull.off[c]);
        if (!tr_graph_row_relax(pull, c, maxplus, d, fin, &nd[i], &np[i], &im)) { tr_atomic_or_u64(&counters[4], 1); im = 0; }
        imp[i] = (uint8_t)im;
    }
}
TR_KERNEL void graph_relax_commit_kernel(const uint32_t *C, uint64_t nc, const int64_t *nd, const uint32_t *np,
                                         const uint8_t *imp, int64_t *d, uint64_t *fin, uint32_t *hops, uint32_t *pred,
                                         uint32_t round, uint64_t *touch, uint32_t *NA, uint64_t *counters) {
    TR_GRID_LOOP(i, nc) {
        uint32_t c = C[i];
        touch[c] = 0;
        if (!imp[i]) continue;
        d[c] = nd[i]; hops[c] = round; pred[c] = np[i];
        graph_set_bit(fin, c);
        NA[tr_atomic_add_u64(&counters[2], 1)] = c;
        tr_atomic_min_u64(&counters[3], c);
    }
}
TR_KERNEL void graph_pred_kernel(tr_graph_adj pull, uint64_t n, const int64_t *d, const uint64_t *fin,
                                 const uint32_t *hops, uint32_t *pred) {
    TR_GRID_LOOP(x, n) pred[x] = tr_graph_row_pred(pull, x, d, fin, hops);
}
TR_KERNEL void graph_energy_kernel(tr_graph_view g, const int64_t *x, uint64_t items, uint64_t *acc) {
    TR_GRID_LOOP(item, items) {
        uint64_t s[TR_GRAPH_ENERGY_LIMBS] = {0, 0, 0, 0, 0}, t[TR_GRAPH_ENERGY_LIMBS];
        for (uint64_t e = item; e < g.edges; e += items) {
            tr_graph_energy_term(g.e_m[e], g.e_w[e], x[g.e_tail[e]], x[g.e_head[e]], t);
            tr_graph_energy_add(s, t);
        }
        graph_atomic_add_limbs(acc, TR_GRAPH_ENERGY_LIMBS, s, TR_GRAPH_ENERGY_LIMBS);
    }
}
TR_KERNEL void graph_block_kernel(const double *x, const double *y, const uint64_t *mask, uint64_t n, uint64_t nb,
                                  double *part) {
    TR_GRID_LOOP(b, nb) part[b] = tr_graph_block(x, y, mask, n, b);
}
TR_KERNEL void graph_total_kernel(const double *part, uint64_t nb, double *out) {
    TR_GRID_LOOP(i, 1) *out = tr_graph_blocks_total(part, nb);
}
TR_KERNEL void graph_tele_kernel(double *scal, double d, double t) {
    TR_GRID_LOOP(i, 1) scal[2] = tr_graph_pr_tele(d, scal[0], scal[1], t);
}
TR_KERNEL void graph_pr_kernel(tr_graph_adj pull, const double *coef, const double *r, double *rn, uint64_t n, double d,
                               const uint64_t *teleport, const double *scal) {
    TR_GRID_LOOP(v, n)
        rn[v] = tr_graph_pr_update(d, tr_graph_row_dot(pull, v, coef, r), scal[2], !teleport || tr_graph_bit(teleport, v));
}
TR_KERNEL void graph_dot_kernel(tr_graph_adj pull, const double *coef, const double *x, double *y, uint64_t n) {
    TR_GRID_LOOP(v, n) y[v] = tr_graph_row_dot(pull, v, coef, x);
}
TR_KERNEL void graph_scale_kernel(double *y, uint64_t n, const double *scal) {
    TR_GRID_LOOP(v, n) y[v] = y[v] / scal[1];
}

/* ------------------------------------------------------------------ host entries */
#ifdef __cplusplus
extern "C" {
#endif
int tr_mgpu_graph_degrees(const tr_graph_view *g, int32_t kind, int32_t weights, tr_math_i128 *out,
                          const tr_math_budget *budget, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (tr_graph_view_check(g) || (g->n && !out) || kind < TR_GRAPH_DEG_OUT || kind > TR_GRAPH_DEG_LOOPS ||
        weights < TR_GRAPH_W_EDGE || weights > TR_GRAPH_W_DISTINCT) return TR_GRAPH_E_INVALID;
    tr_math_need need;
    tr_graph_degrees_base_need(g, TR_GRAPH_NEED_DEVICE, &need);
    int rc = graph_budget(budget, &need, refusal);
    if (rc) return rc;
    if (!g->n) return 0;
    graph_mem m = {{0}, 0, 0};
    tr_graph_view d;
    graph_upload(&m, g, &d, 1, 1, 0);
    tr_math_i128 *dout = (tr_math_i128 *)graph_dev(&m, g->n * 16);
    uint64_t bad = UINT64_MAX, *dbad = (uint64_t *)graph_dev_copy(&m, &bad, 8);
    if (!m.rc) TR_LAUNCH(graph_degrees_kernel, g->n, d, kind, weights, dout, dbad);
    graph_sync(&m);
    graph_out(&m, &bad, dbad, 8);
    if (!m.rc && bad != UINT64_MAX) m.rc = TR_GRAPH_E_OVERFLOW;
    graph_out(&m, out, dout, g->n * 16);
    rc = m.rc;
    graph_free(&m);
    return rc;
}

int tr_mgpu_graph_expand(const tr_graph_view *g, int32_t dir, const uint64_t *frontier, const uint64_t *exclude,
                         uint64_t *next, uint64_t *count, const tr_math_budget *budget, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    const uint64_t nw = g ? tr_graph_words(g->n) : 0;
    if (tr_graph_view_check(g) || !tr_graph_dir_ok(dir) ||
        (nw && (!next || !graph_canonical(frontier, g->n) || (exclude && !graph_canonical(exclude, g->n)))))
        return TR_GRAPH_E_INVALID;
    tr_math_need need;
    tr_graph_expand_base_need(g, TR_GRAPH_NEED_DEVICE, &need);
    int rc = graph_budget(budget, &need, refusal);
    if (rc) return rc;
    if (!nw) { if (count) *count = 0; return 0; }
    graph_mem m = {{0}, 0, 0};
    tr_graph_view d;
    graph_upload(&m, g, &d, GRAPH_PUSH_IN(dir), !GRAPH_PUSH_IN(dir), 0);
    uint64_t *df = (uint64_t *)graph_dev_copy(&m, frontier, nw * 8), *dn = (uint64_t *)graph_dev(&m, nw * 8);
    uint64_t *dx = exclude ? (uint64_t *)graph_dev_copy(&m, exclude, nw * 8) : NULL;
    uint64_t c = 0, *dc = (uint64_t *)graph_dev(&m, 8);
    if (!m.rc) TR_LAUNCH(graph_push_kernel, nw, tr_graph_push(&d, dir), df, nw, dn);
    graph_sync(&m);
    if (!m.rc) TR_LAUNCH(graph_mask_kernel, nw, dn, dx, nw, dc);
    graph_sync(&m);
    graph_out(&m, next, dn, nw * 8);
    graph_out(&m, &c, dc, 8);
    if (!m.rc && count) *count = c;
    rc = m.rc;
    graph_free(&m);
    return rc;
}

int tr_mgpu_graph_spmv(const tr_graph_view *g, int32_t semiring, int32_t dir, int32_t weights, const void *x,
                       const uint64_t *x_valid, void *y, uint64_t *y_valid, double *err, uint64_t *where,
                       const tr_math_budget *budget, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (tr_graph_view_check(g) || semiring < TR_GRAPH_COUNT || semiring > TR_GRAPH_FLOAT || !tr_graph_dir_ok(dir) ||
        weights < TR_GRAPH_W_EDGE || weights > TR_GRAPH_W_DISTINCT ||
        (g->n && (!x || !y || (semiring == TR_GRAPH_FLOAT && !err)))) return TR_GRAPH_E_INVALID;
    if (semiring == TR_GRAPH_GF2 && weights == TR_GRAPH_W_EDGE && g->denominator != 1) return TR_GRAPH_E_DOMAIN;
    const int bits = semiring == TR_GRAPH_BOOL || semiring == TR_GRAPH_GF2;
    if (bits && g->n && !graph_canonical((const uint64_t *)x, g->n)) return TR_GRAPH_E_INVALID;
    tr_math_need need;
    tr_graph_spmv_base_need(g, semiring, TR_GRAPH_NEED_DEVICE, &need);
    int rc = graph_budget(budget, &need, refusal);
    if (rc) return rc;
    const uint64_t n = g->n, nw = tr_graph_words(n);
    if (!n) return 0;
    graph_mem m = {{0}, 0, 0};
    tr_graph_view d;
    graph_upload(&m, g, &d, GRAPH_PULL_IN(dir), !GRAPH_PULL_IN(dir), 0);
    double *coef = NULL, *dcoef = NULL, *derr = NULL;
    if (semiring == TR_GRAPH_FLOAT) {
        coef = (double *)tr_graph_alloc(g->entries * 8);
        if (!coef) m.rc = TR_GRAPH_E_MEMORY;
        else tr_graph_float_coef(g, dir, weights, coef);
        dcoef = (double *)graph_dev_copy(&m, coef, g->entries * 8);
        derr = (double *)graph_dev(&m, n * 8);
    }
    uint64_t xb = bits ? nw * 8 : n * 8, yb = bits ? nw * 8 : (semiring == TR_GRAPH_COUNT ? n * 16 : n * 8);
    void *dx = graph_dev_copy(&m, x, xb), *dy = graph_dev(&m, yb);
    uint64_t *dxv = x_valid ? (uint64_t *)graph_dev_copy(&m, x_valid, nw * 8) : NULL;
    uint64_t *dyv = (uint64_t *)graph_dev(&m, nw * 8);
    uint64_t w = UINT64_MAX, *dw = (uint64_t *)graph_dev_copy(&m, &w, 8);
    if (!m.rc) TR_LAUNCH(graph_spmv_kernel, n, tr_graph_pull(&d, dir), n, semiring, weights, dx, dxv, dy, dyv, derr,
                         dcoef, dw);
    graph_sync(&m);
    graph_out(&m, &w, dw, 8);
    if (!m.rc && w != UINT64_MAX) { m.rc = TR_GRAPH_E_OVERFLOW; if (where) *where = w; }
    graph_out(&m, y, dy, yb);
    if (y_valid) graph_out(&m, y_valid, dyv, nw * 8);
    if (derr) graph_out(&m, err, derr, n * 8);
    rc = m.rc;
    graph_free(&m);
    free(coef);
    return rc;
}

/* ------------------------------------------------------------------ GRAPH-002 */
static void graph_counters_zero(graph_mem *m, uint64_t *dc, uint64_t first_min) {
    uint64_t c[8] = {0, 0, 0, first_min, 0, 0, first_min, 0};
    graph_in(m, dc, c, sizeof c);
}

int tr_mgpu_graph_reach(const tr_graph_view *g, int32_t dir, const uint64_t *sources, int32_t reflexive,
                        uint64_t max_rounds, uint64_t *reached, uint32_t *level, tr_graph_rounds *info,
                        const tr_math_budget *budget, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (tr_graph_view_check(g) || !tr_graph_dir_ok(dir) || !info ||
        (g->n && (!reached || !level || !graph_canonical(sources, g->n)))) return TR_GRAPH_E_INVALID;
    tr_math_need need;
    tr_graph_reach_base_need(g, max_rounds, TR_GRAPH_NEED_DEVICE, &need);
    int rc = graph_budget(budget, &need, refusal);
    if (rc) return rc;
    const uint64_t n = g->n, nw = tr_graph_words(n);
    if (!n) { info->rounds = info->reached = info->frontier_max = info->work = 0; return 0; }
    graph_mem m = {{0}, 0, 0};
    uint32_t *F = (uint32_t *)tr_graph_alloc(n * 4);
    if (!F) return TR_GRAPH_E_MEMORY;
    for (uint64_t w = 0; w < nw; w++) reached[w] = 0;
    for (uint64_t x = 0; x < n; x++) level[x] = TR_GRAPH_NONE;
    uint64_t fn = 0;
    for (uint64_t x = 0; x < n; x++)
        if (tr_graph_bit(sources, x)) {
            F[fn++] = (uint32_t)x;
            if (reflexive) { reached[x >> 6] |= (uint64_t)1 << (x & 63); level[x] = 0; }
        }
    tr_graph_view d;
    graph_upload(&m, g, &d, GRAPH_PUSH_IN(dir), !GRAPH_PUSH_IN(dir), 0);
    uint32_t *dl[2] = {(uint32_t *)graph_dev_copy(&m, F, n * 4), (uint32_t *)graph_dev(&m, n * 4)};
    uint64_t *dvis = (uint64_t *)graph_dev_copy(&m, reached, nw * 8);
    uint32_t *dlev = (uint32_t *)graph_dev_copy(&m, level, n * 4);
    uint64_t *dc = (uint64_t *)graph_dev(&m, 64), c[8];
    free(F);
    uint64_t rounds = 0, work = 0, fmax = fn;
    int cur = 0;
    tr_graph_adj push = tr_graph_push(&d, dir);
    while (!m.rc && fn) {
        if (rounds == max_rounds) { m.rc = TR_GRAPH_E_NOT_CONVERGED; break; }
        rounds++;
        graph_counters_zero(&m, dc, 0);
        if (!m.rc) TR_LAUNCH(graph_reach_kernel, fn, push, dl[cur], fn, dvis, dlev, (uint32_t)rounds, dl[cur ^ 1], dc);
        graph_sync(&m);
        graph_out(&m, c, dc, sizeof c);
        if (m.rc) break;
        work += c[1];
        fn = c[0];
        cur ^= 1;
        if (fn > fmax) fmax = fn;
    }
    graph_out(&m, reached, dvis, nw * 8);
    graph_out(&m, level, dlev, n * 4);
    rc = m.rc;
    graph_free(&m);
    if (rc) return rc;
    info->rounds = rounds; info->reached = tr_graph_count_bits(reached, n); info->frontier_max = fmax; info->work = work;
    return 0;
}

int tr_mgpu_graph_closure(const tr_graph_view *g, int32_t dir, const uint32_t *sources, uint64_t k, int32_t reflexive,
                          uint64_t max_rounds, uint64_t *rows, uint64_t stride, tr_graph_rounds *info,
                          const tr_math_budget *budget, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (tr_graph_view_check(g) || !tr_graph_dir_ok(dir) || !info || (k && (!sources || !rows)) ||
        stride < tr_graph_words(g->n)) return TR_GRAPH_E_INVALID;
    const uint64_t n = g->n;
    for (uint64_t i = 0; i < k; i++) if (sources[i] >= n) return TR_GRAPH_E_INVALID;
    tr_math_need need;
    tr_graph_closure_base_need(g, k, max_rounds, TR_GRAPH_NEED_DEVICE, &need);
    int rc = graph_budget(budget, &need, refusal);
    if (rc) return rc;
    if (k) memset(rows, 0, (size_t)(k * stride * 8));
    info->rounds = info->reached = info->frontier_max = info->work = 0;
    if (!k || !n) return 0;
    graph_mem m = {{0}, 0, 0};
    tr_graph_view d;
    graph_upload(&m, g, &d, 1, 1, 0);
    uint64_t *R = (uint64_t *)tr_graph_alloc(n * 8), *F = (uint64_t *)tr_graph_alloc(n * 8);
    uint32_t *A = (uint32_t *)tr_graph_alloc(n * 4);
    if (!R || !F || !A) m.rc = TR_GRAPH_E_MEMORY;
    uint64_t *dR = (uint64_t *)graph_dev(&m, n * 8), *dF = (uint64_t *)graph_dev(&m, n * 8),
             *dslot = (uint64_t *)graph_dev(&m, n * 8), *dtouch = (uint64_t *)graph_dev(&m, n * 8),
             *dc = (uint64_t *)graph_dev(&m, 64), c[8];
    uint32_t *dA = (uint32_t *)graph_dev(&m, n * 4), *dC = (uint32_t *)graph_dev(&m, n * 4),
             *dNA = (uint32_t *)graph_dev(&m, n * 4);
    tr_graph_adj push = tr_graph_push(&d, dir), pull = tr_graph_pull(&d, dir);
    uint64_t rounds = 0, work = 0, fmax = 0;
    for (uint64_t base = 0; base < k && !m.rc; base += TR_GRAPH_LANES) {
        uint64_t lanes = k - base < TR_GRAPH_LANES ? k - base : TR_GRAPH_LANES, na = 0, r = 0;
        memset(R, 0, (size_t)(n * 8)); memset(F, 0, (size_t)(n * 8));
        for (uint64_t j = 0; j < lanes; j++) {
            uint32_t s = sources[base + j];
            if (!F[s]) A[na++] = s;
            F[s] |= (uint64_t)1 << j;
            if (reflexive) R[s] |= (uint64_t)1 << j;
        }
        if (na > fmax) fmax = na;
        graph_in(&m, dR, R, n * 8);
        graph_in(&m, dF, F, n * 8);
        graph_in(&m, dA, A, na * 4);
        while (!m.rc && na) {
            if (r == max_rounds) { m.rc = TR_GRAPH_E_NOT_CONVERGED; break; }
            r++;
            graph_counters_zero(&m, dc, 0);
            if (!m.rc) TR_LAUNCH(graph_cand_kernel, na, push, dA, na, dtouch, dC, dc);
            graph_sync(&m);
            graph_out(&m, c, dc, sizeof c);
            uint64_t nc = c[0];
            if (!m.rc && nc) TR_LAUNCH(graph_lanes_kernel, nc, pull, dC, nc, dF, dR, dslot, dc);
            graph_sync(&m);
            if (!m.rc) TR_LAUNCH(graph_clear_lanes_kernel, na, dA, na, dF);
            graph_sync(&m);
            if (!m.rc && nc) TR_LAUNCH(graph_commit_lanes_kernel, nc, dC, nc, dslot, dR, dF, dtouch, dNA, dc);
            graph_sync(&m);
            graph_out(&m, c, dc, sizeof c);
            if (m.rc) break;
            work += c[1];
            na = c[2];
            uint32_t *t = dA; dA = dNA; dNA = t;
            if (na > fmax) fmax = na;
        }
        if (r > rounds) rounds = r;
        graph_out(&m, R, dR, n * 8);
        if (!m.rc)
            for (uint64_t x = 0; x < n; x++) {
                uint64_t b = R[x];
                while (b) {
                    uint64_t j = (uint64_t)tr_graph_ctz(b);
                    b &= b - 1;
                    if (j < lanes) rows[(base + j) * stride + (x >> 6)] |= (uint64_t)1 << (x & 63);
                }
            }
    }
    rc = m.rc;
    graph_free(&m);
    free(R); free(F); free(A);
    if (rc) return rc;
    uint64_t reached = 0;
    for (uint64_t i = 0; i < k; i++) reached += tr_graph_count_bits(rows + i * stride, n);
    info->rounds = rounds; info->reached = reached; info->frontier_max = fmax; info->work = work;
    return 0;
}

static void graph_walk_violation(tr_graph_walks_info *info, int32_t kind, uint64_t l, uint64_t v) {
    info->status = TR_GRAPH_E_OVERFLOW; info->kind = kind; info->violation_length = l; info->violation_vertex = v;
    info->maxbits = 0; info->work = 0;
}
int tr_mgpu_graph_walks(const tr_graph_view *g, int32_t dir, const uint64_t *sources, uint32_t max_length,
                        const uint64_t *bound, uint32_t limbs, uint64_t *rows, uint64_t rows_capacity, uint64_t *totals,
                        uint64_t *lengths, uint64_t lengths_capacity, uint64_t *grand, tr_graph_walks_info *info,
                        const tr_math_budget *budget, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (tr_graph_view_check(g) || !tr_graph_dir_ok(dir) || !info || !bound || !grand || limbs < 1 ||
        (g->n && !graph_canonical(sources, g->n))) return TR_GRAPH_E_INVALID;
    const uint64_t n = g->n;
    const uint32_t W = tr_graph_bound_limbs(bound, limbs);
    tr_math_need need;
    uint64_t L = max_length;
    int rc;
    if (max_length == TR_GRAPH_UNBOUNDED) {
        tr_graph_zero_need(&need);
        need.need[TR_BUDGET_BYTES] = tr_graph_horizon_bytes(n);
        rc = graph_budget(budget, &need, refusal);
        if (rc) return rc;
        rc = tr_graph_horizon(g, dir, sources, &L, NULL);
        if (rc) return rc;
    }
    tr_graph_walks_base_need(g, L, W, rows != NULL, TR_GRAPH_NEED_DEVICE, &need);
    rc = graph_budget(budget, &need, refusal);
    if (rc) return rc;
    if (W > TR_GRAPH_MAX_LIMBS) return TR_GRAPH_E_UNSUPPORTED;
    const uint64_t lens = L + 1;
    if ((n && !totals) || !lengths || lengths_capacity < lens * W || (rows && rows_capacity < lens * n * W))
        return TR_GRAPH_E_INVALID;
    memset(info, 0, sizeof *info);
    info->lengths = lens; info->limbs = W; info->unbounded = max_length == TR_GRAPH_UNBOUNDED;
    info->violation_length = TR_GRAPH_NONE; info->violation_vertex = TR_GRAPH_NONE;
    memset(totals, 0, (size_t)(n * W * 8));
    memset(lengths, 0, (size_t)(lens * W * 8));
    if (rows) memset(rows, 0, (size_t)(lens * n * W * 8));
    uint64_t gsum[TR_GRAPH_MAX_LIMBS + 2] = {0}, lsum[TR_GRAPH_MAX_LIMBS + 2] = {0};
    const uint64_t one[1] = {1};
    uint64_t *prev0 = (uint64_t *)tr_graph_alloc(n * W * 8);
    uint32_t *A = (uint32_t *)tr_graph_alloc(n * 4);
    if (!prev0 || !A) { free(prev0); free(A); return TR_GRAPH_E_MEMORY; }
    uint64_t na = 0;
    for (uint64_t x = 0; x < n; x++)
        if (tr_graph_bit(sources, x)) {
            if (!tr_graph_limbs_le(one, 1, bound, W)) { free(prev0); free(A); graph_walk_violation(info, TR_GRAPH_V_ROW, 0, x); return TR_GRAPH_E_OVERFLOW; }
            prev0[x * W] = 1; totals[x * W] = 1;
            if (rows) rows[x * W] = 1;
            A[na++] = (uint32_t)x;
        }
    lsum[0] = na;
    if (!tr_graph_limbs_le(lsum, W + 2, bound, W)) { free(prev0); free(A); graph_walk_violation(info, TR_GRAPH_V_LENGTH, 0, TR_GRAPH_NONE); return TR_GRAPH_E_OVERFLOW; }
    memcpy(lengths, lsum, (size_t)W * 8);
    tr_graph_limbs_add(gsum, W + 2, lsum, W + 2);
    uint64_t maxbits = na ? 1 : 0, work = 0;
    graph_mem m = {{0}, 0, 0};
    tr_graph_view d;
    graph_upload(&m, g, &d, 1, 1, 0);
    uint64_t *dprev = (uint64_t *)graph_dev_copy(&m, prev0, n * W * 8), *dcur = (uint64_t *)graph_dev(&m, n * W * 8),
             *dtot = (uint64_t *)graph_dev_copy(&m, totals, n * W * 8), *dtouch = (uint64_t *)graph_dev(&m, n * 8),
             *dbound = (uint64_t *)graph_dev_copy(&m, bound, (uint64_t)W * 8),
             *dls = (uint64_t *)graph_dev(&m, lens * ((uint64_t)W + 2) * 8), *dc = (uint64_t *)graph_dev(&m, 64), c[8];
    uint64_t *drows = rows ? (uint64_t *)graph_dev_copy(&m, rows, lens * n * W * 8) : NULL;
    uint32_t *dA = (uint32_t *)graph_dev(&m, n * 4), *dC = (uint32_t *)graph_dev(&m, n * 4);   /* lists swap: n each */
    graph_in(&m, dA, A, na * 4);
    free(prev0); free(A);
    tr_graph_adj push = tr_graph_push(&d, dir), pull = tr_graph_pull(&d, dir);
    int violated = 0;
    for (uint64_t l = 1; l <= L && na && !m.rc; l++) {
        graph_counters_zero(&m, dc, TR_GRAPH_NONE);
        if (!m.rc) TR_LAUNCH(graph_cand_kernel, na, push, dA, na, dtouch, dC, dc);
        graph_sync(&m);
        graph_out(&m, c, dc, sizeof c);
        uint64_t nc = c[0];
        if (!m.rc && nc) TR_LAUNCH(graph_walk_kernel, nc, pull, dC, nc, W, dprev, dcur, dbound, dc);
        graph_sync(&m);
        graph_out(&m, c, dc, sizeof c);
        if (m.rc) break;
        if (c[5] > maxbits) maxbits = c[5];
        if (c[3] != TR_GRAPH_NONE) { graph_walk_violation(info, TR_GRAPH_V_ROW, l, c[3]); violated = 1; break; }
        uint64_t *drow = drows ? drows + l * n * W : NULL, *dl = dls + l * ((uint64_t)W + 2);
        if (nc) TR_LAUNCH(graph_walk_commit_kernel, nc, dC, nc, W, dcur, dtot, drow, dl, dbound, dtouch, dc);
        graph_sync(&m);
        graph_out(&m, c, dc, sizeof c);
        if (m.rc) break;
        if (c[6] != TR_GRAPH_NONE) { graph_walk_violation(info, TR_GRAPH_V_TOTAL, l, c[6]); violated = 1; break; }
        work += c[1];
        graph_out(&m, lsum, dl, ((uint64_t)W + 2) * 8);
        if (m.rc) break;
        if (!tr_graph_limbs_le(lsum, W + 2, bound, W)) { graph_walk_violation(info, TR_GRAPH_V_LENGTH, l, TR_GRAPH_NONE); violated = 1; break; }
        memcpy(lengths + l * W, lsum, (size_t)W * 8);
        tr_graph_limbs_add(gsum, W + 2, lsum, W + 2);
        if (!m.rc) TR_LAUNCH(graph_zero_rows_kernel, na, dA, na, W, dprev);
        graph_sync(&m);
        uint64_t *t = dprev; dprev = dcur; dcur = t;
        uint32_t *tl = dA; dA = dC; dC = tl;
        na = nc;
    }
    if (!violated && !m.rc) {
        graph_out(&m, totals, dtot, n * W * 8);
        if (rows) graph_out(&m, rows, drows, lens * n * W * 8);
    }
    rc = m.rc;
    graph_free(&m);
    if (rc) return rc;
    if (violated) return TR_GRAPH_E_OVERFLOW;
    if (!tr_graph_limbs_le(gsum, W + 2, bound, W)) { graph_walk_violation(info, TR_GRAPH_V_GRAND, lens, TR_GRAPH_NONE); return TR_GRAPH_E_OVERFLOW; }
    memcpy(grand, gsum, (size_t)W * 8);
    info->maxbits = maxbits; info->work = work;
    return 0;
}

int tr_mgpu_graph_paths(const tr_graph_view *g, int32_t semiring, int32_t dir, const uint64_t *sources, int64_t *dist,
                        uint64_t *reached, uint32_t *hops, uint32_t *pred, uint32_t *cycle, uint64_t cycle_capacity,
                        tr_graph_paths_info *info, const tr_math_budget *budget, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (tr_graph_view_check(g) || !tr_graph_dir_ok(dir) || !info ||
        (semiring != TR_GRAPH_MIN_PLUS && semiring != TR_GRAPH_MAX_PLUS) ||
        (g->n && (!dist || !reached || !hops || !pred || !graph_canonical(sources, g->n)))) return TR_GRAPH_E_INVALID;
    tr_math_need need;
    tr_graph_paths_base_need(g, TR_GRAPH_NEED_DEVICE, &need);
    int rc = graph_budget(budget, &need, refusal);
    if (rc) return rc;
    const uint64_t n = g->n, nw = tr_graph_words(n);
    const int maxplus = semiring == TR_GRAPH_MAX_PLUS;
    memset(info, 0, sizeof *info);
    info->witness = TR_GRAPH_NONE;
    info->algorithm = TR_GRAPH_BELLMAN_FORD;
    for (uint64_t w = 0; w < nw; w++) reached[w] = 0;
    uint32_t *A = (uint32_t *)tr_graph_alloc(n * 4);
    if (!A) return TR_GRAPH_E_MEMORY;
    uint64_t na = 0;
    for (uint64_t x = 0; x < n; x++) {
        dist[x] = 0; hops[x] = TR_GRAPH_NONE; pred[x] = TR_GRAPH_NONE;
        if (tr_graph_bit(sources, x)) { reached[x >> 6] |= (uint64_t)1 << (x & 63); hops[x] = 0; A[na++] = (uint32_t)x; }
    }
    if (!n) { free(A); return 0; }
    graph_mem m = {{0}, 0, 0};
    tr_graph_view d;
    graph_upload(&m, g, &d, 1, 1, 0);
    int64_t *dd = (int64_t *)graph_dev_copy(&m, dist, n * 8), *dnd = (int64_t *)graph_dev(&m, n * 8);
    uint64_t *dfin = (uint64_t *)graph_dev_copy(&m, reached, nw * 8), *dtouch = (uint64_t *)graph_dev(&m, n * 8),
             *dc = (uint64_t *)graph_dev(&m, 64), c[8];
    uint32_t *dh = (uint32_t *)graph_dev_copy(&m, hops, n * 4), *dp = (uint32_t *)graph_dev_copy(&m, pred, n * 4),
             *dnp = (uint32_t *)graph_dev(&m, n * 4), *dA = (uint32_t *)graph_dev(&m, n * 4),
             *dC = (uint32_t *)graph_dev(&m, n * 4), *dNA = (uint32_t *)graph_dev(&m, n * 4);   /* lists swap: n each */
    uint8_t *dimp = (uint8_t *)graph_dev(&m, n);
    graph_in(&m, dA, A, na * 4);
    free(A);
    tr_graph_adj push = tr_graph_push(&d, dir), pull = tr_graph_pull(&d, dir);
    uint64_t round = 0, work = 0;
    int unbounded = 0;
    while (!m.rc && na) {
        round++;
        graph_counters_zero(&m, dc, TR_GRAPH_NONE);
        if (!m.rc) TR_LAUNCH(graph_cand_kernel, na, push, dA, na, dtouch, dC, dc);
        graph_sync(&m);
        graph_out(&m, c, dc, sizeof c);
        uint64_t nc = c[0];
        if (!m.rc && nc) TR_LAUNCH(graph_relax_kernel, nc, pull, dC, nc, maxplus, dd, dfin, dnd, dnp, dimp, dc);
        graph_sync(&m);
        graph_out(&m, c, dc, sizeof c);
        if (m.rc) break;
        if (c[4]) { m.rc = TR_GRAPH_E_OVERFLOW; break; }
        if (nc) TR_LAUNCH(graph_relax_commit_kernel, nc, dC, nc, dnd, dnp, dimp, dd, dfin, dh, dp, (uint32_t)round,
                          dtouch, dNA, dc);
        graph_sync(&m);
        graph_out(&m, c, dc, sizeof c);
        if (m.rc) break;
        work += c[1];
        na = c[2];
        uint32_t *t = dA; dA = dNA; dNA = t;
        if (na && round >= n) { unbounded = 1; info->witness = c[3]; break; }
    }
    if (!m.rc && !unbounded && n) TR_LAUNCH(graph_pred_kernel, n, pull, n, dd, dfin, dh, dp);
    graph_sync(&m);
    graph_out(&m, dist, dd, n * 8);
    graph_out(&m, reached, dfin, nw * 8);
    graph_out(&m, hops, dh, n * 4);
    graph_out(&m, pred, dp, n * 4);
    rc = m.rc;
    graph_free(&m);
    if (rc) return rc;
    info->rounds = round; info->work = work;
    if (unbounded) {
        info->status = TR_GRAPH_E_UNBOUNDED;
        uint32_t *cyc = cycle && cycle_capacity >= n ? cycle : (uint32_t *)tr_graph_alloc(n * 4);
        uint64_t len = cyc ? tr_graph_cycle(pred, n, info->witness, cyc, n) : 0;
        __int128 cw = 0;
        if (len) tr_graph_cycle_weight(g, dir, maxplus, cyc, len, &cw);
        if (cyc != cycle) free(cyc);
        info->cycle_length = len;
        info->cycle_weight = tr_i128_set(cw);
        return TR_GRAPH_E_UNBOUNDED;
    }
    info->reached = tr_graph_count_bits(reached, n);
    return 0;
}

int tr_mgpu_graph_energy(const tr_graph_view *g, const int64_t *x, uint64_t *out, const tr_math_budget *budget,
                         tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    if (tr_graph_view_check(g) || !out || (g->n && !x)) return TR_GRAPH_E_INVALID;
    tr_math_need need;
    tr_graph_energy_base_need(g, TR_GRAPH_NEED_DEVICE, &need);
    int rc = graph_budget(budget, &need, refusal);
    if (rc) return rc;
    memset(out, 0, TR_GRAPH_ENERGY_LIMBS * 8);
    if (!g->edges) return 0;
    graph_mem m = {{0}, 0, 0};
    tr_graph_view d;
    graph_upload(&m, g, &d, 0, 0, 1);
    int64_t *dx = (int64_t *)graph_dev_copy(&m, x, g->n * 8);
    uint64_t *dacc = (uint64_t *)graph_dev(&m, TR_GRAPH_ENERGY_LIMBS * 8);
    uint64_t items = g->edges < (UINT64_C(1) << 16) ? g->edges : (UINT64_C(1) << 16);
    if (!m.rc) TR_LAUNCH(graph_energy_kernel, items, d, dx, items, dacc);
    graph_sync(&m);
    graph_out(&m, out, dacc, TR_GRAPH_ENERGY_LIMBS * 8);
    rc = m.rc;
    graph_free(&m);
    return rc;
}

/* the blocked sum of n values on the device into *out (device) */
static void graph_blocked(graph_mem *m, const double *x, const double *y, const uint64_t *mask, uint64_t n, double *part,
                          double *out) {
    uint64_t nb = tr_graph_nblocks(n);
    if (!m->rc) TR_LAUNCH(graph_block_kernel, nb, x, y, mask, n, nb, part);
    graph_sync(m);
    if (!m->rc) TR_LAUNCH(graph_total_kernel, 1, part, nb, out);
    graph_sync(m);
}
static int graph_iter_args(const tr_graph_view *g, int32_t dir, double tol, uint64_t max_iter, const uint64_t *start) {
    if (tr_graph_view_check(g) || !tr_graph_dir_ok(dir) || !(tol >= 0) || max_iter < 1 || !g->n) return TR_GRAPH_E_INVALID;
    if (start && (!graph_canonical(start, g->n) || !tr_graph_count_bits(start, g->n))) return TR_GRAPH_E_INVALID;
    return 0;
}
static void graph_uniform(double *x, uint64_t n, const uint64_t *set) {
    uint64_t c = set ? tr_graph_count_bits(set, n) : n;
    double q = 1.0 / (double)c;
    for (uint64_t i = 0; i < n; i++) x[i] = (!set || tr_graph_bit(set, i)) ? q : 0.0;
}
static double graph_host_blocked(const double *x, uint64_t n) {
    double s = 0.0;
    for (uint64_t b = 0, nb = tr_graph_nblocks(n); b < nb; b++) s += tr_graph_block(x, NULL, NULL, n, b);
    return s;
}

int tr_mgpu_graph_pagerank(const tr_graph_view *g, int32_t dir, double damping, const uint64_t *teleport,
                           const uint64_t *start, double tol, uint64_t max_iter, double *rank, tr_graph_iter_info *info,
                           const tr_math_budget *budget, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    int rc = graph_iter_args(g, dir, tol, max_iter, start);
    if (rc || !info || !rank || !(damping > 0 && damping <= 1)) return TR_GRAPH_E_INVALID;
    const uint64_t n = g->n, nw = tr_graph_words(n);
    if (teleport && (!graph_canonical(teleport, n) || !tr_graph_count_bits(teleport, n))) return TR_GRAPH_E_INVALID;
    tr_math_need need;
    tr_graph_iter_base_need(g, max_iter, TR_GRAPH_NEED_DEVICE, &need);
    need.need[TR_BUDGET_BYTES] = tr_budget_add(need.need[TR_BUDGET_BYTES], tr_budget_mul(n, 16));
    rc = graph_budget(budget, &need, refusal);
    if (rc) return rc;
    memset(info, 0, sizeof *info);
    double *coef = (double *)tr_graph_alloc(g->entries * 8), *r = (double *)tr_graph_alloc(n * 8);
    uint64_t *dang = (uint64_t *)tr_graph_alloc(nw * 8), nd = 0;
    __int128 *W = (__int128 *)tr_graph_alloc(n * 16);
    if (!coef || !r || !dang || !W) { free(coef); free(r); free(dang); free(W); return TR_GRAPH_E_MEMORY; }
    rc = tr_graph_pagerank_coef(g, dir, W, coef, dang, &nd);
    if (rc) { free(coef); free(r); free(dang); free(W); return rc; }
    info->dangling = nd;
    graph_uniform(r, n, start ? start : teleport);
    const double t = (double)(teleport ? tr_graph_count_bits(teleport, n) : n);
    graph_mem m = {{0}, 0, 0};
    tr_graph_view d;
    graph_upload(&m, g, &d, GRAPH_PULL_IN(dir), !GRAPH_PULL_IN(dir), 0);
    double *dcoef = (double *)graph_dev_copy(&m, coef, g->entries * 8), *dr = (double *)graph_dev_copy(&m, r, n * 8),
           *drn = (double *)graph_dev(&m, n * 8), *dpart = (double *)graph_dev(&m, tr_graph_nblocks(n) * 8),
           *dscal = (double *)graph_dev(&m, 64);
    uint64_t *ddang = (uint64_t *)graph_dev_copy(&m, dang, nw * 8);
    uint64_t *dtele = teleport ? (uint64_t *)graph_dev_copy(&m, teleport, nw * 8) : NULL;
    tr_graph_adj pull = tr_graph_pull(&d, dir);
    int converged = 0;
    for (uint64_t k = 1; k <= max_iter && !m.rc; k++) {
        graph_blocked(&m, dr, NULL, ddang, n, dpart, dscal + 0);
        graph_blocked(&m, dr, NULL, NULL, n, dpart, dscal + 1);
        if (!m.rc) TR_LAUNCH(graph_tele_kernel, 1, dscal, damping, t);
        graph_sync(&m);
        if (!m.rc) TR_LAUNCH(graph_pr_kernel, n, pull, dcoef, dr, drn, n, damping, dtele, dscal);
        graph_sync(&m);
        graph_blocked(&m, drn, dr, NULL, n, dpart, dscal + 3);
        double delta = 0;
        graph_out(&m, &delta, dscal + 3, 8);
        if (m.rc) break;
        double *tmp = dr; dr = drn; drn = tmp;
        info->iterations = k; info->delta = delta;
        if (delta <= tol) { converged = 1; break; }
    }
    if (converged) graph_out(&m, r, dr, n * 8);
    rc = m.rc;
    graph_free(&m);
    free(coef); free(dang);
    if (!rc && !converged) rc = TR_GRAPH_E_NOT_CONVERGED;
    if (rc) { info->status = rc; free(r); free(W); return rc; }
    memcpy(rank, r, (size_t)(n * 8));
    info->value = graph_host_blocked(r, n);
    info->converged = 1;
    rc = tr_graph_pagerank_certify(g, dir, damping, teleport, r, W, info);
    free(r); free(W);
    return rc;
}

int tr_mgpu_graph_power(const tr_graph_view *g, int32_t dir, const uint64_t *start, double tol, uint64_t max_iter,
                        double *vec, tr_graph_iter_info *info, const tr_math_budget *budget, tr_math_refusal *refusal) {
    tr_budget_none(refusal);
    int rc = graph_iter_args(g, dir, tol, max_iter, start);
    if (rc || !info || !vec) return TR_GRAPH_E_INVALID;
    const uint64_t n = g->n;
    tr_math_need need;
    tr_graph_iter_base_need(g, max_iter, TR_GRAPH_NEED_DEVICE, &need);
    need.need[TR_BUDGET_BYTES] = tr_budget_add(need.need[TR_BUDGET_BYTES], tr_budget_mul(n, 16));
    rc = graph_budget(budget, &need, refusal);
    if (rc) return rc;
    memset(info, 0, sizeof *info);
    double *coef = (double *)tr_graph_alloc(g->entries * 8), *x = (double *)tr_graph_alloc(n * 8);
    if (!coef || !x) { free(coef); free(x); return TR_GRAPH_E_MEMORY; }
    rc = tr_graph_power_coef(g, dir, coef);
    if (rc) { free(coef); free(x); return rc; }
    graph_uniform(x, n, start);
    graph_mem m = {{0}, 0, 0};
    tr_graph_view d;
    graph_upload(&m, g, &d, GRAPH_PULL_IN(dir), !GRAPH_PULL_IN(dir), 0);
    double *dcoef = (double *)graph_dev_copy(&m, coef, g->entries * 8), *dx = (double *)graph_dev_copy(&m, x, n * 8),
           *dy = (double *)graph_dev(&m, n * 8), *dpart = (double *)graph_dev(&m, tr_graph_nblocks(n) * 8),
           *dscal = (double *)graph_dev(&m, 64);
    tr_graph_adj pull = tr_graph_pull(&d, dir);
    int status = TR_GRAPH_E_NOT_CONVERGED;
    for (uint64_t k = 1; k <= max_iter && !m.rc; k++) {
        if (!m.rc) TR_LAUNCH(graph_dot_kernel, n, pull, dcoef, dx, dy, n);
        graph_sync(&m);
        graph_blocked(&m, dy, NULL, NULL, n, dpart, dscal + 1);
        double S = 0;
        graph_out(&m, &S, dscal + 1, 8);
        if (m.rc) break;
        info->iterations = k;
        if (S == 0) { status = TR_GRAPH_E_NO_SOLUTION; break; }
        if (!(S < INFINITY)) { status = TR_GRAPH_E_OVERFLOW; break; }
        if (!m.rc) TR_LAUNCH(graph_scale_kernel, n, dy, n, dscal);
        graph_sync(&m);
        graph_blocked(&m, dy, dx, NULL, n, dpart, dscal + 3);
        double delta = 0;
        graph_out(&m, &delta, dscal + 3, 8);
        if (m.rc) break;
        double *tmp = dx; dx = dy; dy = tmp;
        info->delta = delta; info->value = S;
        if (delta <= tol) { status = 0; break; }
    }
    if (!status) graph_out(&m, x, dx, n * 8);
    rc = m.rc;
    graph_free(&m);
    free(coef);
    if (!rc) rc = status;
    if (rc) { info->status = rc; free(x); return rc; }
    memcpy(vec, x, (size_t)(n * 8));
    info->converged = 1;
    rc = tr_graph_power_certify(g, dir, x, info->value, info);
    free(x);
    return rc;
}
#ifdef __cplusplus
}
#endif
