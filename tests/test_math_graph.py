# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""GRAPH-001 (declared graph layouts and reductions) and GRAPH-002 (bounded fixed points, paths and graph energies):
torbles/math_graph.c, math_graph_gpu.cu, math_graph.py.

Oracles, independent of the implementation and written here: the tiny-graph reference of the declared semantics
(dictionary, canonical edges, adjacency entries, dense adjacency matrices of Python ints / Fractions), dense products
in every semiring, Python BFS, adjacency-matrix powers for walk counts, Bellman-Ford with a lexicographic hop count
and Floyd-Warshall for paths, x^T L x with a dense Laplacian, exact Fraction PageRank by Gaussian elimination, the
stated binary64 contracts replayed with math.fma / math.nextafter, NumPy (PageRank and eigenvectors, at stated
tolerances), edge-list degree counts, DuckDB GROUP BY and recursive CTEs (reachability, bounded walk counts). Every native
result is also computed on the CUDA path (emulated here with TORBLES_EMU_SCRAMBLE=1, a scrambled thread order) and
compared for equality. Seeds are fixed.
"""
import itertools
import math
import random
import unittest
from fractions import Fraction

import numpy as np
import pandas as pd

import torbles
from torbles import math_bank as mbk
from torbles import math_field as mf
from torbles import math_graph as mg
from torbles import math_handles as mh
from torbles.math_budgets import Budget
from torbles.math_domains import DomainError
from torbles.math_errors import (NoSolutionError, NotConvergedError, StaleStateError, UnboundedError,
                                 UnsupportedError)
from torbles.mathematics import AlgebraBudgetError

INF = float("inf")


# ---------------------------------------------------------------------- the declared semantics, by hand

def cuda_label():
    """the backend label of a CUDA result: "cuda" on a device, "cuda-emulated" when the kernels run in the emulation"""
    from torbles import math_gpu
    return "cuda-emulated" if math_gpu.device()["emulated"] else "cuda"


def reference(src, dst, w=None, directed=True, dedup=False, loops="keep", combine="refuse", extra=()):
    """dictionary, canonical edges (tail, head, w, m) sorted, entries (u, v, w, m) sorted by (u, v, w)"""
    w = [1] * len(src) if w is None else [Fraction(x) for x in w]
    rows = [(s, t, x) for s, t, x in zip(src, dst, w) if not (loops == "drop" and s == t)]
    keys = sorted(set([k for s, t, _ in rows for k in (s, t)] + list(extra)))
    idx = {k: i for i, k in enumerate(keys)}
    canon = {}
    for s, t, x in rows:
        u, v = idx[s], idx[t]
        if not directed and v < u:
            u, v = v, u
        if dedup:
            canon.setdefault((u, v), []).append(x)
        else:
            canon[(u, v, x)] = canon.get((u, v, x), 0) + 1
    edges = []
    if dedup:
        for (u, v), ws in sorted(canon.items()):
            if combine == "refuse":
                if len(set(ws)) > 1:
                    raise DomainError("different weights")
                c = ws[0]
            else:
                c = {"min": min, "max": max, "sum": sum}[combine](ws)
            edges.append((u, v, c, 1))
    else:
        edges = [(u, v, x, m) for (u, v, x), m in sorted(canon.items())]
    entries = []
    for u, v, x, m in edges:
        entries.append((u, v, x, m))
        if not directed and u != v:
            entries.append((v, u, x, m))
    entries.sort(key=lambda e: (e[0], e[1], e[2]))
    return keys, edges, entries


def dense(n, entries, mode="edge"):
    """A[u][v] over the entries: edge sum m w, one sum m, distinct 1 per pair"""
    A = [[Fraction(0)] * n for _ in range(n)]
    seen = set()
    for u, v, x, m in entries:
        if mode == "edge":
            A[u][v] += m * x
        elif mode == "one":
            A[u][v] += m
        elif (u, v) not in seen:
            seen.add((u, v))
            A[u][v] += 1
    return A


def pulled(entries, n, direction):
    """per vertex: the entries it aggregates, in stored order (forward: in-entries by (u, w); backward: out-entries)"""
    out = [[] for _ in range(n)]
    if direction == "forward":
        for u, v, x, m in sorted(entries, key=lambda e: (e[1], e[0], e[2])):
            out[v].append((u, x, m))
    else:
        for u, v, x, m in entries:
            out[u].append((v, x, m))
    return out


def random_multigraph(rng, nkeys, nedges, wlo=-5, whi=5, loops=True, keyspan=None):
    keys = rng.sample(range(-50, 50) if keyspan is None else keyspan, nkeys)
    src, dst, w = [], [], []
    for _ in range(nedges):
        s = rng.choice(keys)
        t = s if loops and rng.random() < 0.12 else rng.choice(keys)
        src.append(s); dst.append(t); w.append(rng.randint(wlo, whi))
        if rng.random() < 0.2:                                  # a parallel copy (same or other weight)
            src.append(s); dst.append(t); w.append(w[-1] if rng.random() < 0.5 else rng.randint(wlo, whi))
    return src, dst, w


def bfs_levels(n, entries, sources, reflexive=True, direction="forward"):
    adj = [[] for _ in range(n)]
    for u, v, _, _ in entries:
        if direction == "forward":
            adj[u].append(v)
        else:
            adj[v].append(u)
    level = [None] * n
    front = list(sources)
    if reflexive:
        for s in sources:
            level[s] = 0
    r = 0
    while front:
        r += 1
        nxt = []
        for u in front:
            for v in adj[u]:
                if level[v] is None:
                    level[v] = r
                    nxt.append(v)
        front = nxt
    return level, (r if sources else 0)


def mat_mul(A, B):
    n = len(A)
    return [[sum(A[i][k] * B[k][j] for k in range(n) if A[i][k]) for j in range(n)] for i in range(n)]


def walk_rows(n, entries, sources, L, direction="forward"):
    """rows[l][v] = ((A^T)^l 1_S)[v] with A = sum of multiplicities"""
    A = dense(n, entries, "one")
    row = [1 if v in sources else 0 for v in range(n)]
    rows = [row]
    for _ in range(L):
        if direction == "forward":
            row = [int(sum(A[u][v] * row[u] for u in range(n))) for v in range(n)]
        else:
            row = [int(sum(A[u][v] * row[v] for v in range(n))) for u in range(n)]
        rows.append(row)
    return rows


def bellman_ford(n, entries, sources, maxplus=False):
    """(dist, hops) by lexicographic relaxation, or "cycle" when a reachable improving cycle exists"""
    better = (lambda a, b: a > b) if maxplus else (lambda a, b: a < b)
    d, h = [None] * n, [None] * n
    for s in sources:
        d[s], h[s] = Fraction(0), 0
    for it in range(n + 1):
        changed = False
        for u, v, x, m in entries:
            if d[u] is None:
                continue
            c, ch = d[u] + x, h[u] + 1
            if d[v] is None or better(c, d[v]) or (c == d[v] and ch < h[v]):
                if it == n and (d[v] is None or better(c, d[v])):
                    return "cycle"
                if d[v] is None or better(c, d[v]) or ch < h[v]:
                    d[v], h[v] = c, ch
                    changed = True
        if not changed:
            return d, h
    return "cycle"


def floyd_warshall(n, entries):
    d = [[None] * n for _ in range(n)]
    for i in range(n):
        d[i][i] = Fraction(0)
    for u, v, x, m in entries:
        if d[u][v] is None or x < d[u][v]:
            d[u][v] = x
    for k in range(n):
        for i in range(n):
            if d[i][k] is None:
                continue
            for j in range(n):
                if d[k][j] is not None and (d[i][j] is None or d[i][k] + d[k][j] < d[i][j]):
                    d[i][j] = d[i][k] + d[k][j]
    return d


def frac_solve(A, b):
    n = len(A)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for c in range(n):
        p = next(r for r in range(c, n) if M[r][c] != 0)
        M[c], M[p] = M[p], M[c]
        for r in range(n):
            if r != c and M[r][c] != 0:
                f = M[r][c] / M[c][c]
                M[r] = [a - f * bb for a, bb in zip(M[r], M[c])]
    return [M[i][n] / M[i][i] for i in range(n)]


def up(x):
    return math.nextafter(x, INF)


def blocked(values, block=mg.BLOCK):
    parts = []
    for b in range(0, len(values), block):
        s = 0.0
        for v in values[b:b + block]:
            s += v
        parts.append(s)
    t = 0.0
    for p in parts:
        t += p
    return t


def table_db(values, ddl):
    frame = pd.DataFrame({name: pd.Series(v, dtype=object) for name, v in values.items()})
    return torbles.from_ddl(ddl, {"t": frame})


class Graph(unittest.TestCase):
    def assertCudaSame(self, cpu, cuda, keys=None):
        for k in (keys or [k for k in cpu if k not in ("backend", "device")]):
            a, b = cpu[k], cuda[k]
            if isinstance(a, np.ndarray):
                self.assertTrue(np.array_equal(a, b), k)
            elif isinstance(a, mg.VertexSet):
                self.assertEqual(a, b, k)
            elif isinstance(a, mf.Matrix):
                self.assertTrue(np.array_equal(a.words, b.words), k)
            else:
                self.assertEqual(a, b, k)
        self.assertIn(cuda["backend"], ("cuda", "cuda-emulated"))

    def vec_same(self, a, b):
        self.assertEqual(a.valid, b.valid)
        if isinstance(a.values, np.ndarray):
            self.assertTrue(np.array_equal(a.values.view(np.uint64), b.values.view(np.uint64)))
        else:
            self.assertEqual(a.values, b.values)
        if a.error is not None:
            self.assertTrue(np.array_equal(a.error.view(np.uint64), b.error.view(np.uint64)))
        self.assertIn(b.backend, ("cuda", "cuda-emulated"))

    # ================================================================== GRAPH-001
    def test_graph001_directed_undirected_selfloop_multiedge_semantics_match_tiny_references(self):
        tiny = [
            ([1, 2, 2, 3, 3, 1, 1], [2, 3, 3, 3, 1, 2, 2], [1, 2, 2, 5, -1, 1, 4]),   # loop, parallel, mixed weights
            ([7, 7, 7], [7, 7, 9], [3, 3, 1]),                                        # two loops at one vertex
            ([1, 2, 4, 5], [2, 1, 5, 4], [1, 1, 2, 2]),                               # two components, 2-cycles
            ([], [], []),                                                             # empty
        ]
        rng = random.Random(1001)
        tiny += [random_multigraph(rng, 6, 9) for _ in range(4)]
        for src, dst, w in tiny:
            for directed, dedup, loops in itertools.product((True, False), (False, True), ("keep", "drop")):
                for weighted in (True, False):
                    ws = w if weighted else None
                    combine = "max"
                    try:
                        keys, edges, entries = reference(src, dst, ws, directed, dedup, loops, combine)
                    except DomainError:
                        continue
                    G = mg.Layout.from_edges(src, dst, weight=ws, directed=directed, dedup=dedup, loops=loops,
                                             combine=combine)
                    tag = (src, directed, dedup, loops, weighted)
                    self.assertEqual(G.keys(), keys, tag)
                    n = len(keys)
                    self.assertEqual((G.n, G.edges, G.entries), (n, len(edges), len(entries)), tag)
                    self.assertEqual(G.edges_list(), [(u, v, x, m) for u, v, x, m in edges], tag)
                    out = G.adjacency("out")
                    got = [(u, out["neighbors"][e], out["weights"][e], out["multiplicity"][e])
                           for u in range(n) for e in range(out["offsets"][u], out["offsets"][u + 1])]
                    self.assertEqual(got, entries, tag)
                    inn = G.adjacency("in")
                    got_in = [(inn["neighbors"][e], v, inn["weights"][e], inn["multiplicity"][e])
                              for v in range(n) for e in range(inn["offsets"][v], inn["offsets"][v + 1])]
                    self.assertEqual(got_in, sorted(entries, key=lambda e: (e[1], e[0], e[2])), tag)
                    # degrees: row / column sums of the dense matrices; a loop counts once
                    for mode in ("edge", "one", "distinct"):
                        A = dense(n, entries, mode)
                        outd = [sum(A[u]) for u in range(n)]
                        ind = [sum(A[u][v] for u in range(n)) for v in range(n)]
                        tot = [o + i for o, i in zip(outd, ind)] if directed else outd
                        loopd = [A[u][u] for u in range(n)]
                        for kind, want in (("out", outd), ("in", ind), ("total", tot), ("loops", loopd)):
                            cpu = G.degrees(kind, weights=mode)
                            self.assertEqual(cpu.values, want, (tag, kind, mode))
                            self.vec_same(cpu, G.degrees(kind, weights=mode, backend="cuda"))
                    inf = G.info
                    self.assertEqual(inf["loops"], sum(m for u, v, x, m in edges if u == v), tag)
                    self.assertEqual(inf["multiplicity"], sum(m for *_, m in edges), tag)
                    self.assertEqual(inf["dropped_loops"], sum(1 for s, t in zip(src, dst) if s == t)
                                     if loops == "drop" else 0, tag)
                    self.assertEqual(inf["isolated"], 0, tag)
                    # the incidence lists: per vertex, its edges in order with their role
                    want_inc = [[] for _ in range(n)]
                    for e, (u, v, x, m) in enumerate(edges):
                        if u == v:
                            want_inc[u].append((e, "loop"))
                        else:
                            want_inc[u].append((e, "tail"))
                            want_inc[v].append((e, "head"))
                    self.assertEqual(G.incidence(), want_inc, tag)

    def test_graph001_degree_conventions_against_edge_counts(self):
        rng = random.Random(1002)
        for _ in range(5):
            src, dst, _w = random_multigraph(rng, 7, 12)
            G = mg.Layout.from_edges(src, dst)
            keys = G.keys()
            out_deg = [sum(1 for s in src if s == k) for k in keys]      # edge ends counted on the edge list
            in_deg = [sum(1 for t in dst if t == k) for k in keys]
            loops = [sum(1 for s, t in zip(src, dst) if s == t == k) for k in keys]
            self.assertEqual(G.degrees("out").values, out_deg)
            self.assertEqual(G.degrees("in").values, in_deg)
            self.assertEqual(G.degrees("total").values, [a + b for a, b in zip(out_deg, in_deg)])
            U = mg.Layout.from_edges(src, dst, directed=False)
            # the handshake degree counts every edge end, a self-loop twice; the layout's row sum counts it once
            tot, lp = U.degrees("total").values, U.degrees("loops").values
            self.assertEqual([t + l for t, l in zip(tot, lp)], [a + b for a, b in zip(out_deg, in_deg)])
            self.assertEqual(lp, loops)

    def test_graph001_dedup_weight_combination_is_declared(self):
        src, dst, w = [1, 1, 1, 2, 2], [2, 2, 2, 1, 3], [4, -1, 4, 7, 2]
        with self.assertRaises(DomainError):
            mg.Layout.from_edges(src, dst, weight=w, dedup=True)            # REFUSE: parallel weights differ
        for combine, want in (("min", -1), ("max", 4), ("sum", 7)):
            G = mg.Layout.from_edges(src, dst, weight=w, dedup=True, combine=combine)
            self.assertEqual(G.edges_list()[0], (0, 1, want, 1), combine)
            self.assertEqual(G.info["merged"], 2)
        U = mg.Layout.from_edges(src, dst, weight=w, directed=False, dedup=True, combine="sum")
        self.assertEqual(U.edges_list()[0], (0, 1, 14, 1))                  # {1, 2} and {2, 1} are one edge
        with self.assertRaises(OverflowError):
            mg.Layout.from_edges([1, 1], [2, 2], weight=[(1 << 62), (1 << 62)], dedup=True, combine="sum")
        same = mg.Layout.from_edges([1, 1], [2, 2], weight=[3, 3], dedup=True)
        self.assertEqual(same.edges_list(), [(0, 1, 3, 1)])                 # equal weights merge under REFUSE

    def test_graph001_semiring_products_match_dense_references(self):
        rng = random.Random(1003)
        graphs = [random_multigraph(rng, 7, 14) for _ in range(4)] + [([1, 2, 3], [2, 3, 1], [2, 3, 5])]
        for gi, (src, dst, w) in enumerate(graphs):
            for directed in (True, False):
                ws = [Fraction(x, 2) for x in w] if gi % 2 else w
                G = mg.Layout.from_edges(src, dst, weight=ws, directed=directed)
                keys, edges, entries = reference(src, dst, ws, directed)
                n, D = len(keys), G.denominator
                xs = [rng.randint(-9, 9) for _ in range(n)]
                valid = [rng.random() > 0.2 for _ in range(n)]
                xin = [x if ok else None for x, ok in zip(xs, valid)]
                for direction, mode in itertools.product(("forward", "backward"), ("edge", "one", "distinct")):
                    tag = (gi, directed, direction, mode)
                    P = pulled(entries, n, direction)
                    seen = [set() for _ in range(n)]
                    terms = [[] for _ in range(n)]          # (neighbor, coefficient value, w') per counted term
                    for v in range(n):
                        for u, x, m in P[v]:
                            if mode == "distinct":
                                if u in seen[v]:
                                    continue
                                seen[v].add(u)
                            if not valid[u]:
                                continue
                            c = m * x if mode == "edge" else (m if mode == "one" else 1)
                            terms[v].append((u, c, x if mode == "edge" else 0))
                    # count: exact sums of c x
                    got = G.spmv(xin, "count", direction=direction, weights=mode)
                    self.assertEqual(got.values, [sum(c * xs[u] for u, c, _ in terms[v]) for v in range(n)], tag)
                    self.assertEqual(got.valid, [bool(terms[v]) for v in range(n)], tag)
                    self.vec_same(got, G.spmv(xin, "count", direction=direction, weights=mode, backend="cuda"))
                    # tropical: min / max of w' + x over the terms (exact values; None without a term)
                    xu = [Fraction(x, D) if ok else None for x, ok in zip(xs, valid)]   # multiples of 1/D
                    for sr, f in (("min_plus", min), ("max_plus", max)):
                        got = G.spmv(xu, sr, direction=direction, weights=mode)
                        want = [f(wp + xu[u] for u, _, wp in terms[v]) if terms[v] else None for v in range(n)]
                        self.assertEqual(got.values, want, (tag, sr))
                        self.vec_same(got, G.spmv(xu, sr, direction=direction, weights=mode, backend="cuda"))
                    # Boolean and GF(2)
                    bx = [bool(rng.random() < 0.5) for _ in range(n)]
                    bxin = [b if ok else None for b, ok in zip(bx, valid)]
                    got = G.spmv(bxin, "bool", direction=direction, weights=mode)
                    self.assertEqual(got.values, [any(bx[u] for u, _, _ in terms[v]) for v in range(n)], tag)
                    self.vec_same(got, G.spmv(bxin, "bool", direction=direction, weights=mode, backend="cuda"))
                    if not (mode == "edge" and D != 1):
                        got = G.spmv(bxin, "gf2", direction=direction, weights=mode)
                        want = [sum(int(c) % 2 * int(bx[u]) for u, c, _ in terms[v]) % 2 == 1 for v in range(n)]
                        self.assertEqual(got.values, want, tag)
                        self.vec_same(got, G.spmv(bxin, "gf2", direction=direction, weights=mode, backend="cuda"))
                    else:
                        with self.assertRaises(DomainError):
                            G.spmv(bxin, "gf2", weights="edge")
                    # float: the stated fma accumulation replayed, and the rigorous error bound
                    fx = [rng.uniform(-2, 2) for _ in range(n)]
                    fxin = [x if ok else None for x, ok in zip(fx, valid)]
                    got = G.spmv(fxin, "float", direction=direction, weights=mode)
                    for v in range(n):
                        acc, mag = 0.0, 0.0
                        for u, c, _ in terms[v]:
                            # RN(RN(m w) / RN(D)) of the logical weight (EDGE), RN(m) (ONE), 1 (DISTINCT)
                            cf = float(int(Fraction(c) * D)) / float(D) if mode == "edge" else float(c)
                            acc = math.fma(cf, fx[u], acc)
                            mag = up(math.fma(abs(cf), abs(fx[u]), mag))
                        k = len(terms[v])
                        gamma = up(float(k + 4) / float((1 << 53) - (k + 4)))
                        err = up(gamma * mag) if k else 0.0
                        self.assertEqual(float(got.values[v]), acc, (tag, v))
                        self.assertEqual(float(got.error[v]), err, (tag, v))
                        exact = sum(Fraction(c) * Fraction(fx[u]) for u, c, _ in terms[v])
                        self.assertLessEqual(abs(Fraction(float(got.values[v])) - exact), Fraction(float(got.error[v])))
                    self.vec_same(got, G.spmv(fxin, "float", direction=direction, weights=mode, backend="cuda"))

    def test_graph001_neighborhood_aggregates(self):
        rng = random.Random(1004)
        src, dst, w = random_multigraph(rng, 8, 16)
        G = mg.Layout.from_edges(src, dst, weight=w)
        keys, edges, entries = reference(src, dst, w)
        n = len(keys)
        x = {k: rng.randint(-20, 20) for k in keys if rng.random() < 0.85}
        for direction in ("forward", "backward"):
            P = pulled(entries, n, direction)
            vals = [[(u, m) for u, _, m in P[v] if keys[u] in x] for v in range(n)]
            want = {
                "sum": [sum(m * x[keys[u]] for u, m in vals[v]) for v in range(n)],
                "distinct_sum": [sum(x[keys[u]] for u in sorted(set(u for u, _ in vals[v]))) for v in range(n)],
                "min": [min(x[keys[u]] for u, _ in vals[v]) if vals[v] else None for v in range(n)],
                "max": [max(x[keys[u]] for u, _ in vals[v]) if vals[v] else None for v in range(n)],
                "count": [sum(m for u, m in vals[v]) for v in range(n)],
            }
            for how, ref in want.items():
                got = G.aggregate(x, how, direction=direction)
                self.assertEqual(got.values, ref, (direction, how))
                self.vec_same(got, G.aggregate(x, how, direction=direction, backend="cuda"))
            got = G.aggregate({k: v > 0 for k, v in x.items()}, "any", direction=direction)
            self.assertEqual(got.values, [any(x[keys[u]] > 0 for u, _ in vals[v]) for v in range(n)])
            self.assertEqual(got.as_dict()[keys[0]], got.values[0] if got.valid[0] else None)

    def test_graph001_frontier_expansion_at_every_density(self):
        rng = random.Random(1005)
        for directed in (True, False):
            src, dst, _ = random_multigraph(rng, 150, 400, keyspan=range(10000))
            G = mg.Layout.from_edges(src, dst, directed=directed)
            keys, edges, entries = reference(src, dst, None, directed)
            n = len(keys)
            for density in (0.0, 0.01, 0.1, 0.5, 1.0):
                front = [i for i in range(n) if rng.random() < density]
                excl = [i for i in range(n) if rng.random() < 0.3]
                F = G.vertex_set(ids=front)
                X = G.vertex_set(ids=excl)
                for direction in ("forward", "backward"):
                    fs = set(front)
                    want = sorted(set(v for u, v, _, _ in entries if u in fs) if direction == "forward" else
                                  set(u for u, v, _, _ in entries if v in fs))
                    got = G.expand(F, direction=direction)
                    self.assertEqual(got.ids(), want, (directed, density, direction))
                    self.assertEqual(got.count, len(want))
                    cu = G.expand(F, direction=direction, backend="cuda")
                    self.assertEqual(cu, got)
                    self.assertEqual(cu.backend, cuda_label())
                    got = G.expand(F, direction=direction, exclude=X)
                    self.assertEqual(got.ids(), sorted(set(want) - set(excl)))
                    self.assertEqual(G.expand(F, direction=direction, exclude=X, backend="cuda"), got)

    def test_graph001_gf2_product_is_the_field001_matvec(self):
        rng = random.Random(1006)
        for directed in (True, False):
            src, dst, w = random_multigraph(rng, 70, 160, keyspan=range(1000))
            G = mg.Layout.from_edges(src, dst, weight=w, directed=directed)
            keys, edges, entries = reference(src, dst, w, directed)
            n = G.n
            for mode in ("edge", "one", "distinct"):
                A = G.gf2_matrix(weights=mode)
                ref = dense(n, entries, mode)
                bits = [[int(ref[u][v]) % 2 for v in range(n)] for u in range(n)]
                self.assertTrue(np.array_equal(A.words, mf.Matrix.from_bits(np.array(bits)).words), mode)
                x = [bool(rng.random() < 0.5) for _ in range(n)]
                xi = sum(1 << i for i, b in enumerate(x) if b)
                back = G.spmv(x, "gf2", direction="backward", weights=mode).values
                self.assertEqual(sum(1 << i for i, b in enumerate(back) if b), A.matvec(xi), mode)
                fwd = G.spmv(x, "gf2", direction="forward", weights=mode).values
                self.assertEqual(sum(1 << i for i, b in enumerate(fwd) if b), A.T.matvec(xi), mode)
        H = mg.Layout.from_edges([1, 2], [2, 1], weight=[Fraction(1, 2), 1])
        with self.assertRaises(DomainError):
            H.gf2_matrix()                                     # w mod 2 needs integral weights

    def test_graph001_overflow_is_refused_never_wrapped(self):
        big, top = (1 << 62), (1 << 63) - 1
        C = mg.Layout.from_edges([1, 1, 2], [3, 3, 3], weight=[top, top, top])   # into 3: 2 top (m = 2) + top
        G = mg.Layout.from_edges([1, 1, 2], [3, 3, 3], weight=[big, big, big])
        for backend in ("cpu", "cuda"):
            ok = C.spmv([top, 0, 0], "count", backend=backend)
            self.assertEqual(ok.values[2], 2 * top * top)                      # past int64, exact in int128
            with self.assertRaises(OverflowError) as cm:
                C.spmv([top, top, 0], "count", backend=backend)                # 3 top^2 > 2^127 - 1
            self.assertEqual(cm.exception.vertex, 3)
            with self.assertRaises(OverflowError):
                G.spmv([(1 << 63) - 1 - big + 1, 0, 0], "min_plus", backend=backend)
            self.assertEqual(G.spmv([(1 << 63) - 1 - big, None, None], "min_plus", backend=backend).values[2],
                             (1 << 63) - 1)
        # weighted degrees past int64 stay exact (int128 sums)
        huge = mg.Layout.from_edges([1] * 3, [2] * 3, weight=[(1 << 63) - 1] * 3)
        self.assertEqual(huge.degrees("out", weights="edge").values[0], 3 * ((1 << 63) - 1))

    def test_graph001_table_layouts_match_duckdb(self):
        rng = np.random.default_rng(1007)
        n = 400
        src = [int(v) for v in rng.integers(0, 60, n)]
        dst = [int(v) for v in rng.integers(0, 60, n)]
        w = [f"{int(v) / 100:.2f}" for v in rng.integers(-500, 500, n)]
        for i in rng.choice(n, 20, replace=False):
            src[int(i)] = None
        for i in rng.choice(n, 15, replace=False):
            w[int(i)] = None
        db = table_db({"src": src, "dst": dst, "w": w}, "CREATE TABLE t (src BIGINT, dst BIGINT, w DECIMAL(10,2))")
        t = db["t"]
        G = mg.Layout.from_table(t, "src", "dst", "w")
        self.assertEqual(G.denominator, 100)
        inf = G.info
        self.assertEqual(inf["null_endpoints"], int(db.con.execute(
            "SELECT COUNT(*) FROM t WHERE src IS NULL OR dst IS NULL").fetchone()[0]))
        self.assertEqual(inf["null_weights"], int(db.con.execute(
            "SELECT COUNT(*) FROM t WHERE src IS NOT NULL AND dst IS NOT NULL AND w IS NULL").fetchone()[0]))
        keys = [int(r[0]) for r in db.con.execute(
            "SELECT DISTINCT k FROM (SELECT src k FROM t WHERE src IS NOT NULL AND dst IS NOT NULL AND w IS NOT NULL "
            "UNION SELECT dst FROM t WHERE src IS NOT NULL AND dst IS NOT NULL AND w IS NOT NULL) ORDER BY k").fetchall()]
        self.assertEqual(G.keys(), keys)
        out = dict(db.con.execute("SELECT src, COUNT(*) FROM t WHERE src IS NOT NULL AND dst IS NOT NULL AND "
                                  "w IS NOT NULL GROUP BY src").fetchall())
        wout = dict(db.con.execute("SELECT src, SUM(w) FROM t WHERE src IS NOT NULL AND dst IS NOT NULL AND "
                                   "w IS NOT NULL GROUP BY src").fetchall())
        self.assertEqual(G.degrees("out").values, [int(out.get(k, 0)) for k in keys])
        self.assertEqual(G.degrees("out", weights="edge").values, [Fraction(wout[k]) if k in wout else 0 for k in keys])
        win = dict(db.con.execute("SELECT dst, SUM(w) FROM t WHERE src IS NOT NULL AND dst IS NOT NULL AND "
                                  "w IS NOT NULL GROUP BY dst").fetchall())
        self.assertEqual(G.degrees("in", weights="edge", backend="cuda").values,
                         [Fraction(win[k]) if k in win else 0 for k in keys])
        with self.assertRaises(DomainError):
            mg.Layout.from_table(t, "src", "dst", "w", nulls="forbid")
        # a vertex table adds isolated vertices; the dictionary is the union
        vdb = table_db({"id": [5, 1000, 1001, None]}, "CREATE TABLE t (id BIGINT)")
        V = mg.Layout.from_table(t, "src", "dst", "w", vertices=(vdb["t"], "id"))
        self.assertEqual(V.keys(), sorted(set(keys) | {5, 1000, 1001}))
        self.assertEqual(V.info["isolated"], len(set(V.keys()) - set(keys)))
        # a decimal key column: keys are Fractions, a different scale on the other key column is refused
        ddb = table_db({"a": ["1.50", "2.25"], "b": ["2.25", "3.00"], "c": [1, 2]},
                       "CREATE TABLE t (a DECIMAL(6,2), b DECIMAL(6,2), c BIGINT)")
        K = mg.Layout.from_table(ddb["t"], "a", "b")
        self.assertEqual(K.keys(), [Fraction(3, 2), Fraction(9, 4), Fraction(3)])
        self.assertEqual(K.ids([Fraction(9, 4)]), [1])
        with self.assertRaises(DomainError):
            mg.Layout.from_table(ddb["t"], "a", "c")

    def test_graph001_vertex_dictionaries_and_snapshot_revisions_remain_layout_identity(self):
        db = table_db({"s": [1, 2, 3, 3], "d": [2, 3, 1, 4], "x": [0, 0, 0, 0]},
                      "CREATE TABLE t (s BIGINT, d BIGINT, x BIGINT)")
        t = db["t"]
        A = mg.Layout.from_table(t, "s", "d")
        A2 = mg.Layout.from_table(t, "s", "d")
        self.assertTrue(A.same(A2))                                   # same snapshot, declaration, dictionary
        self.assertEqual(A.info["digest"], A2.info["digest"])
        F = A.vertex_set([1])
        self.assertEqual(A2.expand(F).keys(), [2])                    # interchangeable states
        B = mg.Layout.from_edges([10, 20, 30, 30], [20, 30, 10, 40])  # same shape, another dictionary
        self.assertEqual(B.n, A.n)
        self.assertFalse(B.same(A))
        for op in (lambda: B.expand(F), lambda: B.reach(F), lambda: B.walks(F, 2), lambda: B.paths(F)):
            with self.assertRaises(DomainError):
                op()
        U = mg.Layout.from_table(t, "s", "d", directed=False)         # another declaration
        self.assertFalse(U.same(A))
        with self.assertRaises(DomainError):
            U.expand(F)
        sub = mg.Layout.from_table(t, "s", "d", rows=np.array([True, True, True, False]))   # another selection
        self.assertFalse(sub.same(A))
        # a mutation: a new revision; every use of the old layout (and of its states) is refused
        t.flip(0, 0)
        self.assertFalse(A.fresh())
        for op in (lambda: A.keys() and A.adjacency(), lambda: A.degrees(), lambda: A.expand(F),
                   lambda: A.spmv([1, 1, 1, 1], "count"), lambda: A.reach(F), lambda: A.walks(F, 2),
                   lambda: A.paths(F), lambda: A.energy([0, 0, 0, 0]), lambda: A.pagerank(),
                   lambda: A.degrees(backend="cuda"), lambda: A.spmv([1, 1, 1, 1], "count", backend="cuda")):
            with self.assertRaises(StaleStateError):
                op()
        R = mg.Layout.from_table(t, "s", "d")                         # rebuilt on the new revision
        self.assertFalse(R.same(A))                                   # the revision is part of the identity
        with self.assertRaises(DomainError):
            R.expand(F)                                               # a state of the old snapshot
        self.assertTrue(R.fresh())
        self.assertEqual(R.expand(R.vertex_set([R.keys()[0]])).layout, R)
        with self.assertRaises(NoSolutionError):
            R.ids([999])

    def test_graph001_budgets_at_the_limit_and_one_past(self):
        rng = random.Random(1008)
        src, dst, w = random_multigraph(rng, 30, 80)
        scope = mbk.Scope.from_arrays({"s": src, "d": dst, "w": w})
        need = mg.Layout.build_need(scope, "s", "d", "w", directed=False)
        self.assertEqual(need["support"], len(src))
        for res in ("bytes", "support", "states", "output"):
            mg.Layout.build(scope, "s", "d", "w", directed=False, budget=Budget(**{res: need[res]}))
            with self.assertRaises(AlgebraBudgetError) as cm:
                mg.Layout.build(scope, "s", "d", "w", directed=False, budget=Budget(**{res: need[res] - 1}))
            self.assertEqual((cm.exception.resource, cm.exception.need), (res, need[res]))
        G = mg.Layout.build(scope, "s", "d", "w")
        x = [1] * G.n
        for backend in ("cpu", "cuda"):
            for sr in ("count", "float"):
                nd = G.need("spmv", sr, backend=backend)
                xs = [1.0] * G.n if sr == "float" else x
                for res in [r for r in ("tensor", "output", "bytes") if r in nd]:
                    G.spmv(xs, sr, backend=backend, budget=Budget(**{res: nd[res]}))
                    with self.assertRaises(AlgebraBudgetError) as cm:
                        G.spmv(xs, sr, backend=backend, budget=Budget(**{res: nd[res] - 1}))
                    self.assertEqual(cm.exception.resource, res)
            nd = G.need("degrees", backend=backend)
            G.degrees(backend=backend, budget=Budget(tensor=nd["tensor"]))
            with self.assertRaises(AlgebraBudgetError):
                G.degrees(backend=backend, budget=Budget(tensor=nd["tensor"] - 1))
        nd = G.need("gf2")
        G.gf2_matrix(budget=Budget(tensor=nd["tensor"]))
        with self.assertRaises(AlgebraBudgetError):
            G.gf2_matrix(budget=Budget(tensor=nd["tensor"] - 1))

    def test_graph001_rational_weights_and_units(self):
        G = mg.Layout.from_edges([1, 2, 3], [2, 3, 1], weight=[Fraction(1, 3), Fraction(1, 2), Fraction(-5, 6)])
        self.assertEqual(G.denominator, 6)
        self.assertEqual([e[2] for e in G.edges_list()], [Fraction(1, 3), Fraction(1, 2), Fraction(-5, 6)])
        y = G.spmv([Fraction(1, 6), 0, 1], "min_plus")
        self.assertEqual(y.values, [Fraction(1, 6), Fraction(1, 2), Fraction(1, 2)])
        self.vec_same(y, G.spmv([Fraction(1, 6), 0, 1], "min_plus", backend="cuda"))
        with self.assertRaises(ValueError):
            G.spmv([Fraction(1, 7), 0, 0], "min_plus")               # not a multiple of the weight unit 1/6
        with self.assertRaises(TypeError):
            G.spmv([0.5, 0, 0], "count")
        self.assertEqual(G.spmv([6, 6, 6], "count").values, [-5, 2, 3])
        self.vec_same(G.spmv([6, 6, 6], "count"), G.spmv([6, 6, 6], "count", backend="cuda"))
        self.assertEqual(G.spmv(np.array([6, 6, 6]), "count").values, [-5, 2, 3])          # an int64 array
        self.assertEqual(G.spmv(np.array([1, 0, 0]), "max_plus").values,          # an array has no NULL
                         [Fraction(-5, 6), Fraction(4, 3), Fraction(1, 2)])
        self.assertEqual(G.spmv([1, None, None], "max_plus").values, [None, Fraction(4, 3), None])

    def test_graph001_ctypes_structures_are_registered(self):
        report = mh.abi_check()
        mine = {r["native"] for r in report["checked"] if r["structure"].startswith("torbles.math_graph.")}
        self.assertEqual(mine, {"tr_graph_spec", "tr_graph_info", "tr_graph_view", "tr_graph_rounds",
                                "tr_graph_walks_info", "tr_graph_paths_info", "tr_graph_iter_info"})
        cuda = {r["native"] for r in report["checked"] if r["library"] == "cuda"
                and r["structure"].startswith("torbles.math_graph.")}
        self.assertEqual(cuda, {"tr_graph_view", "tr_graph_rounds", "tr_graph_walks_info", "tr_graph_paths_info",
                                "tr_graph_iter_info"})
        self.assertTrue(all(r["match"] for r in report["checked"]))

    # ================================================================== GRAPH-002
    def duck_reach(self, src, dst, sources, depth):
        import duckdb
        con = duckdb.connect()
        con.register("e", pd.DataFrame({"src": src, "dst": dst}))
        con.register("s", pd.DataFrame({"v": list(sources)}))
        rows = con.execute(f"WITH RECURSIVE r(v, d) AS (SELECT v, 0 FROM s UNION SELECT e.dst, r.d + 1 FROM r JOIN e "
                           f"ON e.src = r.v WHERE r.d < {depth}) SELECT v, MIN(d) FROM r GROUP BY v").fetchall()
        return {int(v): int(d) for v, d in rows}

    def test_graph002_reachability_on_tiny_cycles_and_disconnected_graphs(self):
        rng = random.Random(2001)
        cases = [([1, 2, 3], [2, 3, 1]),                               # a 3-cycle
                 ([1, 2, 4, 5, 6], [2, 1, 5, 6, 4]),                   # two disconnected cycles
                 ([1, 2, 3, 3], [2, 3, 3, 4]),                         # a chain with a loop
                 ([1], [1])]                                           # one loop
        cases += [random_multigraph(rng, 9, 9)[:2] for _ in range(4)]
        for src, dst in cases:
            for directed in (True, False):
                G = mg.Layout.from_edges(src, dst, directed=directed)
                keys, edges, entries = reference(src, dst, None, directed)
                n = len(keys)
                for sset in ([0], [0, n - 1], []):
                    sk = [keys[i] for i in sset]
                    for reflexive, direction in itertools.product((True, False), ("forward", "backward")):
                        tag = (src, directed, sset, reflexive, direction)
                        lev, rounds = bfs_levels(n, entries, sset, reflexive, direction)
                        got = G.reach(sk, reflexive=reflexive, direction=direction)
                        self.assertEqual(got["level"], lev, tag)
                        self.assertEqual(got["reached"].ids(), [i for i in range(n) if lev[i] is not None], tag)
                        self.assertEqual(got["rounds"], rounds, tag)
                        self.assertCudaSame(got, G.reach(sk, reflexive=reflexive, direction=direction, backend="cuda"))
                        if got["rounds"]:
                            G.reach(sk, reflexive=reflexive, direction=direction, max_rounds=got["rounds"])
                            for backend in ("cpu", "cuda"):
                                with self.assertRaises(NotConvergedError):
                                    G.reach(sk, reflexive=reflexive, direction=direction,
                                            max_rounds=got["rounds"] - 1, backend=backend)
                    if directed and sset:
                        duck = self.duck_reach(src, dst, sk, n)
                        lev, _ = bfs_levels(n, entries, sset)
                        self.assertEqual({keys[i]: l for i, l in enumerate(lev) if l is not None}, duck)
                # the closure: row s = the reach set of s (R+ and R*)
                for reflexive in (False, True):
                    cl = G.closure(reflexive=reflexive)
                    for s in range(n):
                        lev, _ = bfs_levels(n, entries, [s], reflexive)
                        want = sum(1 << v for v in range(n) if lev[v] is not None)
                        self.assertEqual(sum(int(cl["matrix"].words[s][j]) << (64 * j)
                                             for j in range(cl["matrix"].words.shape[1])), want)
                    self.assertCudaSame(cl, G.closure(reflexive=reflexive, backend="cuda"))

    def test_graph002_closure_over_more_than_one_lane_batch(self):
        n = 150
        src = list(range(n)) + [7, 90]
        dst = [(i + 1) % n if i % 50 else i for i in range(n)] + [120, 3]
        G = mg.Layout.from_edges(src, dst)
        keys, edges, entries = reference(src, dst)
        cl = G.closure(direction="backward", reflexive=True)
        self.assertEqual(cl["matrix"].rows, n)
        for s in (0, 63, 64, 127, 128, 149):
            lev, _ = bfs_levels(n, entries, [s], True, "backward")
            row = sum(int(cl["matrix"].words[s][j]) << (64 * j) for j in range(cl["matrix"].words.shape[1]))
            self.assertEqual(row, sum(1 << v for v in range(n) if lev[v] is not None), s)
        self.assertCudaSame(cl, G.closure(direction="backward", reflexive=True, backend="cuda"))
        part = G.closure([keys[5], keys[140]])
        self.assertEqual(part["sources"], [keys[5], keys[140]])
        self.assertTrue(np.array_equal(part["matrix"].words[1], G.closure()["matrix"].words[140]))

    def duck_walks(self, src, dst, sources, L):
        import duckdb
        con = duckdb.connect()
        con.register("e", pd.DataFrame({"src": src, "dst": dst}))
        con.register("s", pd.DataFrame({"v": list(sources)}))
        rows = con.execute(f"WITH RECURSIVE w(v, l) AS (SELECT v, 0 FROM s UNION ALL SELECT e.dst, w.l + 1 FROM w "
                           f"JOIN e ON e.src = w.v WHERE w.l < {L}) SELECT l, v, COUNT(*) FROM w GROUP BY l, v").fetchall()
        return {(int(l), int(v)): int(c) for l, v, c in rows}

    def test_graph002_walk_counts_match_adjacency_matrix_powers(self):
        rng = random.Random(2002)
        cases = [([1, 2, 3, 3, 1, 1], [2, 3, 1, 3, 2, 2])] + [random_multigraph(rng, 6, 10)[:2] for _ in range(4)]
        for src, dst in cases:
            for directed in (True, False):
                G = mg.Layout.from_edges(src, dst, directed=directed)
                keys, edges, entries = reference(src, dst, None, directed)
                n = len(keys)
                for sset, L, direction in (([0], 6, "forward"), ([0, n - 1], 5, "backward"), ([], 3, "forward")):
                    sk = [keys[i] for i in sset]
                    rows = walk_rows(n, entries, set(sset), L, direction)
                    got = G.walks(sk, L, direction=direction)
                    self.assertEqual(got["rows"], rows, (src, directed, sset, direction))
                    self.assertEqual(got["totals"], [sum(r[v] for r in rows) for v in range(n)])
                    self.assertEqual(got["lengths"], [sum(r) for r in rows])
                    self.assertEqual(got["grand"], sum(map(sum, rows)))
                    self.assertCudaSame(got, G.walks(sk, L, direction=direction, backend="cuda"))
                    nr = G.walks(sk, L, direction=direction, rows=False)
                    self.assertNotIn("rows", nr)
                    self.assertEqual(nr["totals"], got["totals"])
                if directed:
                    duck = self.duck_walks(src, dst, [keys[0]], 5)
                    rows = walk_rows(n, entries, {0}, 5)
                    self.assertEqual({(l, keys[v]): c for l, r in enumerate(rows) for v, c in enumerate(r) if c}, duck)
        # counts far past 64 bits under a declared big bound (complete digraph with loops: n^l walks)
        n = 9
        src = [a for a in range(n) for b in range(n)]
        dst = [b for a in range(n) for b in range(n)]
        G = mg.Layout.from_edges(src, dst)
        got = G.walks([0], 60, bound=1 << 200, rows=False, budget=Budget(degree=60))
        self.assertEqual(got["lengths"], [1] + [9 ** l for l in range(1, 61)])
        self.assertEqual(got["totals"], [1 + sum(9 ** (l - 1) for l in range(1, 61))] + [sum(9 ** (l - 1) for l in range(1, 61))] * 8)
        self.assertEqual(got["limbs"], 4)
        self.assertCudaSame(got, G.walks([0], 60, bound=1 << 200, rows=False, backend="cuda", budget=Budget(degree=60)))

    def test_graph002_unbounded_walk_counts_and_bound_violations_never_give_a_finite_answer(self):
        # a DAG: every walk has length <= the longest path, the unbounded count is finite
        rng = random.Random(2003)
        for _ in range(4):
            n = 9
            pairs = [(a, b) for a in range(n) for b in range(a + 1, n) if rng.random() < 0.35]
            pairs += [pairs[0]] if pairs else []
            src, dst = [a for a, _ in pairs], [b for _, b in pairs]
            if not src:
                continue
            G = mg.Layout.from_edges(src, dst)
            keys, edges, entries = reference(src, dst)
            rows = walk_rows(len(keys), entries, {0}, len(keys))
            while len(rows) > 1 and not any(rows[-1]):
                rows.pop()
            got = G.walks([keys[0]], None)
            self.assertEqual(got["length"], len(rows) - 1)
            self.assertEqual(got["rows"], rows)
            self.assertEqual(got["totals"], [sum(r[v] for r in rows) for v in range(len(keys))])
            self.assertCudaSame(got, G.walks([keys[0]], None, backend="cuda"))
        # a reachable cycle (a 2-cycle, a self-loop): the unbounded counts diverge
        for src, dst, s in (([1, 2, 3, 3], [2, 3, 2, 4], 1), ([1, 2], [2, 2], 1), ([5, 1, 2], [1, 2, 3], 5)):
            G = mg.Layout.from_edges(src, dst)
            if src == [5, 1, 2]:
                self.assertEqual(G.walks([s], None)["grand"], 4)        # a chain: finite
                continue
            for backend in ("cpu", "cuda"):
                with self.assertRaises(NoSolutionError):
                    G.walks([s], None, backend=backend)
            with self.assertRaises(NoSolutionError):
                G.walks_need([s], None)
        G = mg.Layout.from_edges([1, 2, 3, 4], [2, 1, 4, 4])             # cycle 1 <-> 2; loop at 4
        self.assertEqual(G.walks([3], None, direction="backward")["grand"], 1)   # nothing reaches 3 backward
        with self.assertRaises(NoSolutionError):
            G.walks([3], None)                                           # the loop at 4 is reachable from 3
        # the first coefficient past the declared bound, in the stated order
        G = mg.Layout.from_edges([1, 1, 2, 2, 1], [2, 2, 1, 1, 3])
        keys, edges, entries = reference([1, 1, 2, 2, 1], [2, 2, 1, 1, 3])
        rows = walk_rows(3, entries, {0}, 12)
        for bound in (0, 1, 2, 3, 7, 40, 100, 4095, 4096):
            want = None
            tot, gsum = [0, 0, 0], 0
            for l, r in enumerate(rows):
                if any(c > bound for c in r):
                    want = ("row", l, keys[min(v for v in range(3) if r[v] > bound)])
                    break
                tot = [a + b for a, b in zip(tot, r)]
                if any(c > bound for c in tot):
                    want = ("total", l, keys[min(v for v in range(3) if tot[v] > bound)])
                    break
                if sum(r) > bound:
                    want = ("length", l, None)
                    break
                gsum += sum(r)
            if want is None and gsum > bound:
                want = ("grand", 13, None)
            for backend in ("cpu", "cuda"):
                if want is None:
                    got = G.walks([1], 12, bound=bound, backend=backend)
                    self.assertEqual(got["rows"], rows)
                    continue
                with self.assertRaises(OverflowError) as cm:
                    G.walks([1], 12, bound=bound, backend=backend)
                e = cm.exception
                self.assertEqual((e.kind, e.length, e.vertex), want, (bound, backend))
        exact = max(max(r) for r in rows)
        tot = [sum(r[v] for r in rows) for v in range(3)]
        top = max(max(tot), max(sum(r) for r in rows), sum(map(sum, rows)))
        self.assertEqual(G.walks([1], 12, bound=top)["grand"], sum(map(sum, rows)))
        with self.assertRaises(OverflowError):
            G.walks([1], 12, bound=top - 1)
        self.assertLessEqual(exact, top)
        with self.assertRaises(UnsupportedError):
            G.walks([1], 2, bound=1 << 2048)

    def test_graph002_shortest_paths_with_negative_and_zero_weights(self):
        rng = random.Random(2004)
        for trial in range(10):
            n = 8
            phi = [rng.randint(-6, 6) for _ in range(n)]
            src, dst, w = [], [], []
            for _ in range(18):
                a, b = rng.randrange(n), rng.randrange(n)
                c = rng.choice([0, 0, 1, 2, 5])
                src.append(a); dst.append(b); w.append(c + phi[a] - phi[b])  # no negative cycle; zero cycles
            src.append(n + 1); dst.append(n + 2); w.append(-3)              # a disconnected component
            G = mg.Layout.from_edges(src, dst, weight=w)
            keys, edges, entries = reference(src, dst, w)
            m = len(keys)
            fw = floyd_warshall(m, entries)
            for sset in ([0], [0, 3], [m - 1]):
                sk = [keys[i] for i in sset]
                d, h = bellman_ford(m, entries, sset)
                want = [min((fw[s][v] for s in sset if fw[s][v] is not None), default=None) for v in range(m)]
                self.assertEqual(d, want)
                got = G.paths(sk)
                self.assertEqual(got["algorithm"], "bellman_ford" if G.info["wmin"] < 0 else "dijkstra")
                self.assertEqual(got["distance"], [x if x is None else (int(x) if x.denominator == 1 else x)
                                                   for x in d], (trial, sset))
                self.assertEqual(got["hops"], h)
                for v in range(m):                                   # pred: canonical, consistent
                    p = got["pred"][v]
                    if h[v] in (None, 0):
                        self.assertIsNone(p)
                        continue
                    cands = [u for u, vv, x, _ in entries if vv == v and d[u] is not None and h[u] + 1 == h[v]
                             and d[u] + x == d[v]]
                    self.assertEqual(p, min(cands))
                self.assertCudaSame(got, G.paths(sk, backend="cuda"), keys=["distance", "hops", "pred", "reached"])
                bf = G.paths(sk, algorithm="bellman_ford")
                self.assertCudaSame(bf, G.paths(sk, backend="cuda"))
            if G.info["wmin"] < 0:
                with self.assertRaises(DomainError):
                    G.paths([keys[0]], algorithm="dijkstra")
            # nonnegative weights: Dijkstra and Bellman-Ford agree exactly
            nn = mg.Layout.from_edges(src, dst, weight=[abs(x) for x in w])
            a, b = nn.paths([keys[0]], algorithm="dijkstra"), nn.paths([keys[0]], algorithm="bellman_ford")
            self.assertEqual((a["distance"], a["hops"], a["pred"]), (b["distance"], b["hops"], b["pred"]))
        # longest paths on a DAG (max_plus), against the reference
        pairs = [(a, b) for a in range(7) for b in range(a + 1, 7) if (a * 7 + b) % 3]
        w = [((a + 2 * b) % 5) - 2 for a, b in pairs]
        G = mg.Layout.from_edges([a for a, _ in pairs], [b for _, b in pairs], weight=w)
        keys, edges, entries = reference([a for a, _ in pairs], [b for _, b in pairs], w)
        d, h = bellman_ford(len(keys), entries, [0], maxplus=True)
        got = G.paths([keys[0]], "max_plus")
        self.assertEqual(got["distance"], [None if x is None else int(x) for x in d])
        self.assertEqual(got["hops"], h)
        self.assertCudaSame(got, G.paths([keys[0]], "max_plus", backend="cuda"))
        with self.assertRaises(DomainError):
            G.paths([keys[0]], "max_plus", algorithm="dijkstra")
        # zero-weight cycles converge; hop counts and predecessors do not loop
        Z = mg.Layout.from_edges([1, 2, 3, 2, 4], [2, 3, 2, 4, 1], weight=[2, 0, 0, -1, 0])
        got = Z.paths([1])
        self.assertEqual(got["distance"], [0, 2, 2, 1])
        self.assertEqual((got["hops"], got["pred"]), ([0, 1, 2, 2], [None, 0, 1, 1]))

    def test_graph002_reachable_negative_cycles_are_reported_never_a_finite_answer(self):
        rng = random.Random(2005)
        for trial in range(8):
            n = 7
            src, dst, w = [], [], []
            for _ in range(14):
                src.append(rng.randrange(n)); dst.append(rng.randrange(n)); w.append(rng.randint(-4, 6))
            G = mg.Layout.from_edges(src, dst, weight=w)
            keys, edges, entries = reference(src, dst, w)
            m = len(keys)
            for maxplus in (False, True):
                sr = "max_plus" if maxplus else "min_plus"
                ref = bellman_ford(m, entries, [0], maxplus)
                if ref != "cycle":
                    got = G.paths([keys[0]], sr)
                    self.assertEqual(got["distance"], [None if x is None else int(x) for x in ref[0]])
                    continue
                cycles = []
                for backend in ("cpu", "cuda"):
                    with self.assertRaises(UnboundedError) as cm:
                        G.paths([keys[0]], sr, backend=backend)
                    e = cm.exception
                    cyc = e.cycle_ids
                    self.assertTrue(cyc)
                    total = 0
                    for i, u in enumerate(cyc):                        # a real cycle along entries
                        v = cyc[(i + 1) % len(cyc)]
                        ws = [x for a, b, x, _ in entries if a == u and b == v]
                        self.assertTrue(ws, (trial, cyc))
                        total += max(ws) if maxplus else min(ws)
                    self.assertEqual(total, e.weight)
                    self.assertTrue(total > 0 if maxplus else total < 0)
                    lev, _ = bfs_levels(m, entries, [0])
                    self.assertIsNotNone(lev[cyc[0]])                  # reachable from the source
                    cycles.append((cyc, e.weight, e.witness))
                self.assertEqual(cycles[0], cycles[1])
        # a negative cycle that the sources cannot reach is no obstacle
        G = mg.Layout.from_edges([1, 3, 4, 4], [2, 4, 3, 5], weight=[1, -2, 1, 1])
        self.assertEqual(G.paths([1])["distance"], [0, 1, None, None, None])
        for backend in ("cpu", "cuda"):
            with self.assertRaises(UnboundedError):
                G.paths([3], backend=backend)
        loop = mg.Layout.from_edges([1], [1], weight=[-1])             # a negative self-loop at the source
        with self.assertRaises(UnboundedError) as cm:
            loop.paths([1])
        self.assertEqual((cm.exception.cycle, cm.exception.weight), ([1], -1))

    def test_graph002_exact_dirichlet_energy(self):
        rng = random.Random(2006)
        for trial in range(6):
            src, dst, w = random_multigraph(rng, 8, 14)
            ws = [Fraction(x, 4) for x in w] if trial % 2 else w
            for directed in (True, False):
                G = mg.Layout.from_edges(src, dst, weight=ws, directed=directed)
                keys = G.keys()
                xs = {k: Fraction(rng.randint(-30, 30), rng.choice([1, 3])) for k in keys}
                want = sum(Fraction(x) * (xs[t] - xs[s]) ** 2 for s, t, x in zip(src, dst, ws))
                n = len(keys)
                idx = {k: i for i, k in enumerate(keys)}
                L = [[Fraction(0)] * n for _ in range(n)]
                for s, t, x in zip(src, dst, ws):
                    a, b = idx[s], idx[t]
                    if a != b:
                        L[a][a] += x; L[b][b] += x; L[a][b] -= x; L[b][a] -= x
                vec = [xs[k] for k in keys]
                self.assertEqual(want, sum(vec[i] * L[i][j] * vec[j] for i in range(n) for j in range(n)))
                got = G.energy(xs)
                self.assertEqual(got, want)
                self.assertEqual(G.energy(xs, backend="cuda"), want)
        # values at the int64 extremes: 320-bit exact
        big = (1 << 63) - 1
        G = mg.Layout.from_edges([1, 2, 1], [2, 3, 3], weight=[big, -big, big])
        x = {1: -(1 << 63), 2: big, 3: 0}
        want = big * (big + (1 << 63)) ** 2 - big * big ** 2 + big * (1 << 63) ** 2
        self.assertEqual(G.energy(x), want)
        self.assertEqual(G.energy(x, backend="cuda"), want)
        with self.assertRaises(ValueError):
            G.energy({1: 1})

    def replica_pagerank(self, G, d, tol, max_iter, teleport=None, start=None):
        """the stated binary64 iteration, replayed with math.fma (forward)"""
        n = G.n
        inn, out = G.adjacency("in"), G.adjacency("out")
        Wu = [sum(out["multiplicity"][e] * out["weights"][e] * G.denominator for e in
                  range(out["offsets"][u], out["offsets"][u + 1])) for u in range(n)]
        coef = []
        for v in range(n):
            for e in range(inn["offsets"][v], inn["offsets"][v + 1]):
                u = inn["neighbors"][e]
                mw = inn["multiplicity"][e] * inn["weights"][e] * G.denominator
                coef.append(0.0 if Wu[u] == 0 else float(int(mw)) / float(int(Wu[u])))
        T = set(range(n)) if teleport is None else set(teleport)
        S0 = T if start is None else set(start)
        r = [1.0 / len(S0) if i in S0 else 0.0 for i in range(n)]
        t = float(len(T))
        for k in range(1, max_iter + 1):
            D = blocked([r[u] if Wu[u] == 0 else 0.0 for u in range(n)])
            S = blocked(r)
            tele = math.fma(d, D, (1.0 - d) * S) / t
            rn = []
            for v in range(n):
                acc = 0.0
                for e in range(inn["offsets"][v], inn["offsets"][v + 1]):
                    acc = math.fma(coef[e], r[inn["neighbors"][e]], acc)
                rn.append(math.fma(d, acc, tele if v in T else 0.0))
            delta = blocked([abs(a - b) for a, b in zip(rn, r)])
            r = rn
            if delta <= tol:
                return r, k, delta
        return None, max_iter, delta

    def exact_pagerank(self, G, d, teleport=None):
        n = G.n
        A = dense(n, [(u, v, x, m) for u, v, x, m in self.entries_of(G)], "edge")
        Wu = [sum(A[u]) for u in range(n)]
        T = set(range(n)) if teleport is None else set(teleport)
        p = [Fraction(1, len(T)) if i in T else Fraction(0) for i in range(n)]
        dd = Fraction(d)
        M = [[(A[u][v] / Wu[u] if Wu[u] else p[v]) for u in range(n)] for v in range(n)]   # column-stochastic
        lhs = [[(1 if i == j else 0) - dd * M[i][j] for j in range(n)] for i in range(n)]
        return frac_solve(lhs, [(1 - dd) * p[i] for i in range(n)])

    def entries_of(self, G):
        out = G.adjacency("out")
        return [(u, out["neighbors"][e], out["weights"][e], out["multiplicity"][e]) for u in range(G.n)
                for e in range(out["offsets"][u], out["offsets"][u + 1])]

    def test_graph002_pagerank_and_power_iteration_with_declared_tolerances(self):
        rng = random.Random(2007)
        cases = [([1, 2, 3, 3, 1, 4], [2, 3, 1, 3, 3, 1], None),        # a dangling-free graph with a loop
                 ([1, 2, 2, 3], [2, 3, 4, 1], [1, 2, 3, 1]),            # a dangling vertex (4), weights
                 random_multigraph(rng, 6, 12, wlo=0, whi=4)]
        for src, dst, w in cases:
            for directed in (True, False):
                G = mg.Layout.from_edges(src, dst, weight=w, directed=directed)
                for d, teleport in ((0.85, None), (0.5, [G.keys()[0], G.keys()[-1]])):
                    got = G.pagerank(d, tol=1e-13, max_iter=400, teleport=teleport)
                    tid = None if teleport is None else G.ids(teleport)
                    rep, k, delta = self.replica_pagerank(G, d, 1e-13, 400, tid)
                    self.assertEqual(got["rank"].tolist(), rep)               # the stated operations, bit for bit
                    self.assertEqual((got["iterations"], got["delta"]), (k, delta))
                    exact = self.exact_pagerank(G, d, tid)
                    err = sum(abs(Fraction(float(a)) - b) for a, b in zip(got["rank"], exact))
                    self.assertLessEqual(err, Fraction(got["bound"]))         # the certified bound holds
                    self.assertLess(got["bound"], 1e-9)
                    # NumPy: the dense Google matrix's fixed point at a stated tolerance (1e-9 in L1)
                    n = G.n
                    A = np.array([[float(x) for x in row] for row in dense(n, self.entries_of(G), "edge")])
                    Wu = A.sum(axis=1)
                    p = np.zeros(n)
                    p[list(tid) if tid else list(range(n))] = 1.0
                    p /= p.sum()
                    Mx = np.where(Wu[None, :] > 0, (A / np.where(Wu > 0, Wu, 1)[:, None]).T, p[:, None])
                    ref = np.linalg.solve(np.eye(n) - d * Mx, (1 - d) * p)
                    self.assertLess(np.abs(got["rank"] - ref).sum(), 1e-9)
                    cu = G.pagerank(d, tol=1e-13, max_iter=400, teleport=teleport, backend="cuda")
                    self.assertCudaSame(got, cu)
            # power iteration: the dominant eigenvector of A^T (an aperiodic strongly connected part)
            P = mg.Layout.from_edges([1, 2, 3, 3, 1, 2], [2, 3, 1, 3, 3, 2], directed=False)
            got = P.power(tol=1e-14, max_iter=2000)
            A = np.array([[float(x) for x in row] for row in dense(P.n, self.entries_of(P), "edge")])
            vals, vecs = np.linalg.eigh(A)
            top = vecs[:, -1] / vecs[:, -1].sum()
            self.assertLess(np.abs(got["vector"] - top).sum(), 1e-9)
            self.assertLessEqual(abs(got["eigenvalue"] - vals[-1]), got["radius"] + 1e-12)
            self.assertLess(got["radius"], 1e-9)
            self.assertCudaSame(got, P.power(tol=1e-14, max_iter=2000, backend="cuda"))
            Dg = mg.Layout.from_edges([1, 2, 3, 3, 1], [2, 3, 1, 3, 3])
            gd = Dg.power(tol=1e-14, max_iter=2000)
            A = np.array([[float(x) for x in row] for row in dense(Dg.n, self.entries_of(Dg), "edge")])
            r = A.T @ gd["vector"] - gd["eigenvalue"] * gd["vector"]
            self.assertLessEqual(np.abs(r).sum(), gd["residual_l1"] * (1 + 1e-12) + 1e-300)
            self.assertEqual(gd["radius"], INF)
            self.assertCudaSame(gd, Dg.power(tol=1e-14, max_iter=2000, backend="cuda"))
        with self.assertRaises(DomainError):
            mg.Layout.from_edges([1, 2], [2, 1], weight=[1, -1]).pagerank()

    def test_graph002_nonconvergent_recurrences_never_return_a_finite_answer(self):
        C2 = mg.Layout.from_edges([1, 2], [2, 1])                      # periodic: no damping, a point start
        for backend in ("cpu", "cuda"):
            with self.assertRaises(NotConvergedError) as cm:
                C2.pagerank(1.0, start=[1], max_iter=40, backend=backend)
            self.assertEqual((cm.exception.iterations, cm.exception.delta), (40, 2.0))
            ok = C2.pagerank(1.0, max_iter=5, backend=backend)          # the uniform start is stationary
            self.assertEqual((ok["iterations"], ok["bound"]), (1, INF))  # d = 1: no contraction, no bound claimed
        G = mg.Layout.from_edges([1, 2, 3, 3], [2, 3, 1, 2])
        tight = G.pagerank(0.85, tol=1e-14, max_iter=500)
        for backend in ("cpu", "cuda"):
            with self.assertRaises(NotConvergedError):
                G.pagerank(0.85, tol=1e-14, max_iter=tight["iterations"] - 1, backend=backend)
            self.assertEqual(G.pagerank(0.85, tol=1e-14, max_iter=tight["iterations"], backend=backend)["rank"].tolist(),
                             tight["rank"].tolist())
            with self.assertRaises(NotConvergedError):
                G.pagerank(0.85, tol=0.0, max_iter=60, backend=backend)  # below the reachable rounding level
        B = mg.Layout.from_edges([1, 2, 3], [2, 3, 4], directed=False)   # bipartite: eigenvalues +-lambda
        for backend in ("cpu", "cuda"):
            with self.assertRaises(NotConvergedError):
                B.power(start=[1], max_iter=80, backend=backend)
            with self.assertRaises(NoSolutionError):
                mg.Layout.from_edges([1, 2], [2, 3]).power(backend=backend)   # nilpotent: the iterate vanishes
        with self.assertRaises(ValueError):
            G.pagerank(0.0)
        with self.assertRaises(ValueError):
            G.pagerank(0.85, max_iter=0)

    def test_graph002_iteration_and_round_budgets(self):
        G = mg.Layout.from_edges(list(range(30)), list(range(1, 31)))     # a chain of 31 vertices
        for backend in ("cpu", "cuda"):
            G.walks([0], 20, budget=Budget(iterations=20, degree=20), backend=backend)
            with self.assertRaises(AlgebraBudgetError) as cm:
                G.walks([0], 21, budget=Budget(iterations=20, degree=64), backend=backend)
            self.assertEqual(cm.exception.resource, "iterations")
            nd = G.walks_need([0], 20, backend=backend)
            G.walks([0], 20, backend=backend, budget=Budget(bytes=nd["bytes"], output=nd["output"]))
            with self.assertRaises(AlgebraBudgetError):
                G.walks([0], 20, backend=backend, budget=Budget(bytes=nd["bytes"] - 1))
            with self.assertRaises(AlgebraBudgetError):
                G.walks([0], 20, backend=backend, budget=Budget(output=nd["output"] - 1))
            G.reach([0], max_rounds=31, budget=Budget(iterations=31), backend=backend)
            with self.assertRaises(AlgebraBudgetError):
                G.reach([0], max_rounds=31, budget=Budget(iterations=30), backend=backend)
            G.pagerank(max_iter=200, budget=Budget(iterations=200), backend=backend)
            with self.assertRaises(AlgebraBudgetError):
                G.pagerank(max_iter=200, budget=Budget(iterations=199), backend=backend)
            nd = G.need("paths", backend=backend)
            G.paths([0], algorithm="bellman_ford" if backend == "cpu" else "auto", backend=backend,
                    budget=Budget(iterations=nd["iterations"], bytes=nd["bytes"]))
            with self.assertRaises(AlgebraBudgetError):
                G.paths([0], backend=backend, budget=Budget(iterations=nd["iterations"] - 1))
            nd = G.need("closure", a=31, b=32, backend=backend)
            G.closure(budget=Budget(tensor=nd["tensor"]), backend=backend)
            with self.assertRaises(AlgebraBudgetError):
                G.closure(budget=Budget(tensor=nd["tensor"] - 1), backend=backend)
            nd = G.need("energy", backend=backend)
            G.energy(list(range(31)), budget=Budget(tensor=nd["tensor"]), backend=backend)
            with self.assertRaises(AlgebraBudgetError):
                G.energy(list(range(31)), budget=Budget(tensor=nd["tensor"] - 1), backend=backend)
        self.assertEqual(G.walks([0], None)["length"], 30)
        self.assertEqual(G.reach([0])["rounds"], 31)                     # the fixed point after 31 expansions


if __name__ == "__main__":
    unittest.main()
