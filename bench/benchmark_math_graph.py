# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""GRAPH-001 / GRAPH-002 benchmark: declared graph layouts and reductions; bounded fixed points, paths and energies.

    python bench/benchmark_math_graph.py --capability GRAPH-001 | GRAPH-002 --out FILE [--device] [--quick]

Case 1 (shapes x sizes): chain, star, disconnected (64 chains) and cyclic (a cycle plus 10% random chords) graphs
with 1k / 100k / 1M edges (1k / 100k with --quick). Reported separately: the edge scope (the FND-004 data path), the
adjacency / incidence build (dictionary, CSR, CSC, canonical edges, incidence lists), then the frontier and reduction
iterations on the built layout: one-step frontier expansion at densities 0.1% / 10% / 50%, the semiring products
(count, bool, min_plus, float), degrees, reachability from one vertex (its rounds), shortest paths and PageRank.
Baselines: SciPy sparse (COO -> CSR build, A^T x, csgraph breadth-first order and Dijkstra), NumPy (bincount degrees,
a vectorized PageRank power iteration to the same tolerance). SciPy is not a declared dependency: without it the SciPy
baselines, their cross-checks and the PageRank baseline (which uses SciPy's sparse product) are skipped and noted.
Case 2 (bounded path lengths 1 / 4 / 16 / 32): walk counts from one vertex on the cyclic, star and disconnected graphs
(100k edges; 1k with --quick) under a declared bound; the semirings, reachability and bounded walk counts of a small
graph compared with exact oracles written here (Python ints / Fractions over the edge list, DuckDB recursive CTEs).
Refusals retained: an overflow (walk counts past the declared bound), nontermination (unbounded walk counts with a
reachable cycle, PageRank without damping from a point start, Bellman-Ford with a planted negative cycle) and the
budget thresholds (layout bytes, walk iterations) at the last admitted and the first refused point. Timings of the
Python API include the exact formatting of the returned values. CUDA is timed only with --device. Both capabilities
run both cases (the inventory gives them the same benchmark cases).
"""
import random
from fractions import Fraction

import numpy as np

try:
    import scipy.sparse as sp
    import scipy.sparse.csgraph as cg
except ImportError:                                               # not a declared dependency: baselines skipped
    sp = cg = None

from math_bench import Bench
from torbles import math_bank as mbk
from torbles import math_graph as mg
from torbles.math_budgets import Budget
from torbles.math_errors import NoSolutionError, NotConvergedError, UnboundedError


def shape(kind, m, rng):
    """(src, dst) int64 arrays of m edges"""
    if kind == "chain":
        s = np.arange(m, dtype=np.int64)
        return s, s + 1
    if kind == "star":
        return np.zeros(m, dtype=np.int64), np.arange(1, m + 1, dtype=np.int64)
    if kind == "disconnected":
        per = max(1, m // 64)
        s = np.arange(m, dtype=np.int64)
        comp = s // per
        return s + comp, s + comp + 1                        # 64 chains, one id skipped between components
    n = max(2, (m * 9) // 10)                                 # cyclic: a cycle and random chords
    s = np.arange(n, dtype=np.int64)
    extra = m - n
    cs = rng.integers(0, n, extra, dtype=np.int64)
    cd = rng.integers(0, n, extra, dtype=np.int64)
    return np.concatenate([s, cs]), np.concatenate([(s + 1) % n, cd])


def scipy_csr(src, dst, n, data=None):
    return sp.coo_matrix((np.ones(len(src)) if data is None else data, (src, dst)), shape=(n, n)).tocsr()


def numpy_pagerank(A, d, tol, max_iter):
    """the dense-vector power iteration of the same Google matrix (SciPy sparse products)"""
    n = A.shape[0]
    out = np.asarray(A.sum(axis=1)).ravel()
    dang = out == 0
    inv = np.where(dang, 0.0, 1.0 / np.where(dang, 1.0, out))
    r = np.full(n, 1.0 / n)
    AT = A.T.tocsr()
    for k in range(1, max_iter + 1):
        y = AT @ (r * inv)
        rn = d * y + (d * r[dang].sum() + (1 - d) * r.sum()) / n
        delta = np.abs(rn - r).sum()
        r = rn
        if delta <= tol:
            return r, k
    return r, max_iter


def case_shapes(bench):
    rng = np.random.default_rng(bench.seed)
    sizes = (1000, 100_000) if bench.quick else (1000, 100_000, 1_000_000)
    for kind in ("chain", "star", "disconnected", "cyclic"):
        for m in sizes:
            src, dst = shape(kind, m, rng)
            w = rng.integers(0, 10, len(src), dtype=np.int64)
            p = dict(shape=kind, edges=int(len(src)))
            rep = 2 if m >= 1_000_000 else 3
            scope = bench.measure("edge scope (FND-004 arrays)", "cpu",
                                  lambda: mbk.Scope.from_arrays({"s": src, "d": dst, "w": w}), repeats=rep,
                                  phase="build", **p)
            G = bench.measure("adjacency / incidence build (dictionary, CSR, CSC, incidence)", "cpu",
                              lambda: mg.Layout.build(scope, "s", "d", "w"), repeats=rep, phase="build", **p)
            info = G.info
            bench.records[-1]["result"] = dict(vertices=info["vertices"], entries=info["entries"],
                                               edges=info["edges"], layout_bytes=info["bytes"])
            n = G.n
            ids = np.searchsorted(np.array(G.keys(), dtype=np.int64), src), np.searchsorted(np.array(G.keys()), dst)
            if sp is not None:
                bench.measure("adjacency build (COO -> CSR)", "scipy", lambda: scipy_csr(ids[0], ids[1], n),
                              repeats=rep, phase="build", **p)
                A = scipy_csr(ids[0], ids[1], n)
                AT = A.T.tocsr()
            # frontier expansion at three densities
            for dens in (0.001, 0.1, 0.5):
                mask = rng.random(n) < dens
                F = G.vertex_set(mask=mask)
                for backend in bench.backends():
                    got = bench.measure("frontier expansion", backend, lambda: G.expand(F, backend=backend).count,
                                        repeats=5, density=dens, **p)
                if sp is not None:
                    ref = bench.measure("frontier expansion (A^T x > 0)", "scipy",
                                        lambda: int(np.count_nonzero(AT @ mask.astype(np.float64))), repeats=5,
                                        density=dens, **p)
                    if got != ref:
                        raise SystemExit(f"expansion disagrees with SciPy: {kind} {m} {dens}")
            x = rng.integers(-5, 6, n, dtype=np.int64)
            xf = rng.random(n)
            for sr, xv in (("count", x), ("bool", x > 0), ("min_plus", x), ("float", xf)):
                for backend in bench.backends():
                    bench.measure(f"semiring product ({sr})", backend,
                                  lambda: G.spmv(xv, sr, weights="one", backend=backend).array.shape[0], repeats=5,
                                  semiring=sr, **p)
            cnt = G.spmv(x, "count", weights="one").array[:, 0]
            if sp is not None:
                ref = bench.measure("semiring product (count): A^T x", "scipy", lambda: AT @ x, repeats=5,
                                    semiring="count", **p)
                if not np.array_equal(cnt, ref.astype(np.int64)):
                    raise SystemExit(f"count product disagrees with SciPy: {kind} {m}")
            for backend in bench.backends():
                deg = bench.measure("degrees (in)", backend, lambda: G.degrees("in", backend=backend).array[:, 0],
                                    repeats=5, **p)
            nb = bench.measure("degrees (in): bincount", "numpy", lambda: np.bincount(ids[1], minlength=n), repeats=5,
                               **p)
            if not np.array_equal(deg[:n], nb):
                raise SystemExit("degrees disagree with NumPy")
            start = int(ids[0][0])
            S = G.vertex_set(ids=[start])
            for backend in bench.backends():
                rr = bench.measure("reachability from one vertex (fixed point)", backend,
                                   lambda: G.reach(S, backend=backend), repeats=2, **p)
                bench.records[-1]["result"] = dict(rounds=rr["rounds"], reached=rr["reached_count"])
            if sp is not None:
                order = bench.measure("reachability (breadth-first order)", "scipy",
                                      lambda: cg.breadth_first_order(A, start, directed=True,
                                                                     return_predecessors=False), repeats=2, **p)
                if len(order) != rr["reached_count"]:
                    raise SystemExit("reachability disagrees with SciPy")
            for alg in ("dijkstra", "bellman_ford"):
                for backend in bench.backends():
                    if backend == "cuda" and alg == "dijkstra":
                        continue
                    pr = bench.measure(f"shortest paths ({alg})", backend,
                                       lambda: G.paths(S, algorithm=alg if backend == "cpu" else "auto",
                                                       backend=backend), repeats=2, algorithm=alg, **p)
                    bench.records[-1]["result"] = dict(rounds=pr["rounds"], reached=pr["reached_count"])
            if sp is not None:
                # SciPy sums duplicate COO entries: keep the lightest parallel edge (the shortest-path semantics), and
                # a tiny positive offset keeps zero weights as stored edges
                o = np.lexsort((w, ids[1], ids[0]))
                first = np.ones(len(o), dtype=bool)
                first[1:] = (ids[0][o][1:] != ids[0][o][:-1]) | (ids[1][o][1:] != ids[1][o][:-1])
                sel = o[first]
                Aw = scipy_csr(ids[0][sel], ids[1][sel], n, data=w[sel].astype(np.float64) + 1e-300)
                dj = bench.measure("shortest paths (Dijkstra)", "scipy",
                                   lambda: cg.dijkstra(Aw, directed=True, indices=start), repeats=2, **p)
                mine = G.paths(S)["distance"]
                if any((v is None) != (not np.isfinite(r)) or (v is not None and abs(v - r) > 1e-6)
                       for v, r in zip(mine, dj)):
                    raise SystemExit("shortest paths disagree with SciPy")
            for backend in bench.backends():
                prk = bench.measure("PageRank (d = 0.85, tol 1e-10)", backend,
                                    lambda: G.pagerank(0.85, tol=1e-10, max_iter=1000, backend=backend),
                                    repeats=2, **p)
                bench.records[-1]["result"] = dict(iterations=prk["iterations"], bound=prk["bound"])
            if sp is not None:
                Apr = scipy_csr(ids[0], ids[1], n, data=w.astype(np.float64))   # transition weights m w (summed)
                nr, nk = bench.measure("PageRank (sparse power iteration)", "numpy",
                                       lambda: numpy_pagerank(Apr, 0.85, 1e-10, 1000), repeats=2, **p)
                if np.abs(nr - prk["rank"]).sum() > 1e-8:
                    raise SystemExit("PageRank disagrees with NumPy")
            for backend in bench.backends():
                bench.measure("Dirichlet energy (exact)", backend, lambda: G.energy(x, backend=backend), repeats=3, **p)


def oracle_walks(n, src, dst, start, L):
    rows = [[0] * n]
    rows[0][start] = 1
    for _ in range(L):
        nxt = [0] * n
        prev = rows[-1]
        for a, b in zip(src, dst):
            if prev[a]:
                nxt[b] += prev[a]
        rows.append(nxt)
    return rows


def case_lengths(bench):
    rng = np.random.default_rng(bench.seed + 1)
    m = 1000 if bench.quick else 100_000
    for kind in ("cyclic", "star", "disconnected"):
        src, dst = shape(kind, m, rng)
        G = mg.Layout.from_edges(src.tolist(), dst.tolist())
        S = G.vertex_set([int(src[0])])
        for L in (1, 4, 16, 32):
            for backend in bench.backends():
                got = bench.measure("bounded walk counts (totals)", backend,
                                    lambda: G.walks(S, L, bound=(1 << 127) - 1, rows=False, backend=backend),
                                    repeats=3, shape=kind, edges=m, length=L)
                bench.records[-1]["result"] = dict(grand=str(got["grand"]), maxbits=got["maxbits"], work=got["work"])
            if m <= 1000:
                bench.measure("bounded walk counts (rows kept)", "cpu",
                              lambda: G.walks(S, L, bound=(1 << 127) - 1, rows=True)["grand"], repeats=3, shape=kind,
                              edges=m, length=L)
    # small exact oracles: every semiring, reachability and bounded walks on a 300-edge weighted multigraph
    r = random.Random(bench.seed)
    keys = list(range(60))
    src = [r.choice(keys) for _ in range(300)]
    dst = [r.choice(keys) if r.random() > 0.1 else s for s in src]
    w = [r.randint(-4, 9) for _ in src]
    G = mg.Layout.from_edges(src, dst, weight=w)
    n = G.n
    idx = {k: i for i, k in enumerate(G.keys())}
    x = [r.randint(-9, 9) for _ in range(n)]
    agree = {}
    ref_count = [0] * n
    ref_min, ref_max = [None] * n, [None] * n
    ref_bool, ref_gf2 = [False] * n, [0] * n
    for a, b, ww in zip(src, dst, w):
        u, v = idx[a], idx[b]
        ref_count[v] += ww * x[u]
        ref_min[v] = ww + x[u] if ref_min[v] is None else min(ref_min[v], ww + x[u])
        ref_max[v] = ww + x[u] if ref_max[v] is None else max(ref_max[v], ww + x[u])
        ref_bool[v] = ref_bool[v] or x[u] > 0
        ref_gf2[v] ^= (ww % 2) & int(x[u] > 0)
    agree["count"] = G.spmv(x, "count").values == ref_count
    agree["min_plus"] = G.spmv(x, "min_plus").values == ref_min
    agree["max_plus"] = G.spmv(x, "max_plus").values == ref_max
    agree["bool"] = G.spmv([v > 0 for v in x], "bool").values == ref_bool
    agree["gf2"] = G.spmv([v > 0 for v in x], "gf2").values == [bool(b) for b in ref_gf2]
    exact = [sum(Fraction(ww) * Fraction(float(x[idx[a]])) for a, b, ww in zip(src, dst, w) if idx[b] == v)
             for v in range(n)]
    fl = G.spmv([float(v) for v in x], "float")
    agree["float_within_bound"] = all(abs(Fraction(float(y)) - e) <= Fraction(float(b))
                                      for y, e, b in zip(fl.values, exact, fl.error))
    s0 = idx[src[0]]
    rows = oracle_walks(n, [idx[a] for a in src], [idx[b] for b in dst], s0, 6)
    agree["walks"] = G.walks([src[0]], 6, bound=1 << 200)["rows"] == rows
    import duckdb
    import pandas as pd
    con = duckdb.connect()
    con.register("e", pd.DataFrame({"src": src, "dst": dst}))
    reach = dict(con.execute(f"WITH RECURSIVE r(v, d) AS (SELECT {src[0]}, 0 UNION SELECT e.dst, r.d + 1 FROM r JOIN e "
                             f"ON e.src = r.v WHERE r.d < {n}) SELECT v, MIN(d) FROM r GROUP BY v").fetchall())
    lev = G.reach([src[0]])["level"]
    agree["reach_duckdb"] = {k: lev[i] for k, i in idx.items() if lev[i] is not None} == reach
    walks = con.execute(f"WITH RECURSIVE w(v, l) AS (SELECT {src[0]}, 0 UNION ALL SELECT e.dst, w.l + 1 FROM w JOIN e "
                        f"ON e.src = w.v WHERE w.l < 4) SELECT l, COUNT(*) FROM w GROUP BY l ORDER BY l").fetchall()
    agree["walks_duckdb"] = [c for _, c in walks] == G.walks([src[0]], 4)["lengths"]
    bench.note(f"small exact oracles (300 edges, 60 keys): {agree}")
    if not all(agree.values()):
        raise SystemExit(f"an oracle disagrees: {agree}")


def keep(bench, case, backend, fn, expected, **params):
    """a nontermination / overflow refusal, retained as a record (the exception type and message)"""
    try:
        fn()
    except expected as e:
        bench.records.append(dict(case=case, backend=backend, phase="refusal", params=params, refused=True,
                                  resource=None, error=type(e).__name__, message=str(e)))
        return
    raise SystemExit(f"{case}: no refusal")


def case_refusals(bench):
    cyc = mg.Layout.from_edges([0, 1, 1], [1, 0, 2])                       # a reachable 2-cycle
    two = mg.Layout.from_edges([0, 1], [1, 0])                             # periodic: no dangling vertex
    tri = mg.Layout.from_edges([0, 0, 1, 1, 2, 2], [1, 1, 2, 2, 0, 0])      # 2^l walks: past any int64 bound
    for backend in bench.backends():
        label = backend
        keep(bench, "walk counts past the declared bound (overflow)", label,
             lambda: tri.walks([0], 64, bound=(1 << 63) - 1, rows=False, backend=backend), OverflowError, length=64)
        keep(bench, "unbounded walk counts with a reachable cycle (divergence)", label,
             lambda: cyc.walks([0], None, backend=backend), NoSolutionError)
        keep(bench, "PageRank without damping from a point start (nonconvergence)", label,
             lambda: two.pagerank(1.0, start=[0], max_iter=100, backend=backend), NotConvergedError)
        neg = mg.Layout.from_edges([0, 1, 2], [1, 2, 1], weight=[1, -3, 1])
        keep(bench, "Bellman-Ford with a planted negative cycle (no finite answer)", label,
             lambda: neg.paths([0], backend=backend), UnboundedError)
    # budget thresholds, measured at the last admitted and the first refused point
    rng = np.random.default_rng(bench.seed + 2)
    src, dst = shape("cyclic", 20_000, rng)
    scope = mbk.Scope.from_arrays({"s": src, "d": dst})
    need = mg.Layout.build_need(scope, "s", "d")
    bench.measure("layout build at its byte need", "cpu",
                  lambda: mg.Layout.build(scope, "s", "d", budget=Budget(bytes=need["bytes"])).n, repeats=1,
                  bytes=need["bytes"])
    bench.refusal("layout build one byte past", "cpu",
                  lambda: mg.Layout.build(scope, "s", "d", budget=Budget(bytes=need["bytes"] - 1)),
                  bytes=need["bytes"] - 1)
    bench.threshold("layout bytes (20k-edge cyclic graph)", admitted=dict(bytes=need["bytes"]),
                    refused=dict(bytes=need["bytes"] - 1))
    G = mg.Layout.build(scope, "s", "d")
    S = G.vertex_set(ids=[0])
    b = Budget(iterations=32, degree=64)
    bench.measure("walk counts at a 32-iteration budget", "cpu", lambda: G.walks(S, 32, rows=False, budget=b)["grand"],
                  repeats=1, length=32)
    bench.refusal("walk counts past a 32-iteration budget", "cpu", lambda: G.walks(S, 33, rows=False, budget=b),
                  length=33)
    bench.threshold("walk length (iterations 32)", admitted=dict(length=32), refused=dict(length=33))


def main():
    bench = Bench.from_args("GRAPH-001", __doc__)
    if sp is None:
        bench.note("SciPy is not installed: the SciPy baselines (CSR build, frontier expansion, count product, "
                   "breadth-first order, Dijkstra) and the sparse PageRank baseline are skipped with their "
                   "cross-checks; the exact oracles of case 2 and of tests/test_math_graph.py still apply")
    case_shapes(bench)
    case_lengths(bench)
    case_refusals(bench)
    bench.write()


if __name__ == "__main__":
    main()
