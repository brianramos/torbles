# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""DIST-001 / DIST-002 benchmark: histogram construction (sparse and dense, joint dimensions 1..3) and prefix requests
(CDF, tails, quantiles, trimmed sums) reused from one shared support.

    python bench/benchmark_math_dist.py --capability DIST-001 --out FILE [--device] [--quick]
    python bench/benchmark_math_dist.py --capability DIST-002 --out FILE [--device] [--quick]

DIST-001: 1k / 100k / 1M rows (quick: 1k / 100k), support sizes 8 / 64 / 1024 per column, joint dimensions 1 / 2 / 3,
5% NULLs per column (a NULL category of its own). The scope is built once per grid point (measured apart); then the
sparse state (native CPU; the strategy AUTO picks dense counting + compaction or sorting, recorded) and the dense
state (refused over the default budget: the refusal is recorded, never extrapolated), with their bytes, cells,
occupied cells, scans of the rows (1 per build) and passes over the scratch. Baselines: DuckDB GROUP BY over the same
columns (COUNT(*), NULL groups), checked equal in number of cells and rows, and NumPy np.unique(axis=0). Extra cases:
64 groups, a near-row-sized support (1M rows over 2M values) and a structural-zero domain (even values only, dense).
CUDA (--device only): the same builds on the device.

DIST-002: 100k rows (quick: 20k), supports 8 / 64 / 1024 and near-row-sized, with duplicates and structural zeros
(a sparse support of even values). Construction (histogram, then the prefix state) is measured apart from execution:
1 / 100 / 10,000 requests of each kind (CDF, tail, PERCENTILE_CONT quantile, trimmed sum), first batch and warm
batches. Baselines: DuckDB SQL (one query per request up to 100 requests; for 10,000 a single quantile_cont over a list,
or a cumulative histogram with an ASOF join of the thresholds), and NumPy complete reconstruction (sort the rows once, then np.searchsorted per batch;
the one-shot cost includes the sort). Answers are checked against NumPy on a sample. CUDA (--device only): the upload
of the prefix state, then the batches.
"""
import time
from fractions import Fraction

import duckdb
import numpy as np
import pandas as pd

from math_bench import Bench
from torbles import math_bank as mbk, math_dist as md
from torbles.math_budgets import Budget


def columns(rng, rows, support, k, nulls=0.05):
    vals = {f"c{i}": rng.integers(0, support, rows).astype(np.int64) for i in range(k)}
    valid = {n: rng.random(rows) >= nulls for n in vals}
    return vals, valid


def nullable(values, valid):
    return pd.Series(np.where(valid, values, 0), dtype="Int64").where(valid)


def frame(vals, valid):
    return pd.DataFrame({n: nullable(v, valid[n]) for n, v in vals.items()})


# ---------------------------------------------------------------------- DIST-001
def dist001(bench, rng):
    budget = Budget()
    for rows in ((1_000, 100_000) if bench.quick else (1_000, 100_000, 1_000_000)):
        for support in (8, 64, 1024):
            for k in (1, 2, 3):
                vals, valid = columns(rng, rows, support, k)
                names = list(vals)
                p = dict(rows=rows, support=support, dims=k, groups=1)
                scope = bench.measure("scope", "cpu", lambda: mbk.Scope.from_arrays(vals, valid=valid), repeats=2,
                                      phase="build", **p)
                con = duckdb.connect()
                con.register("data", frame(vals, valid))
                q = f"SELECT {', '.join(names)}, COUNT(*) FROM data GROUP BY ALL"
                expect = bench.measure("histogram", "duckdb", lambda: con.execute(q).fetchall(), repeats=3,
                                       phase="run", **p)
                stacked = np.stack([np.where(valid[n], vals[n], -1) for n in names], axis=1)
                bench.measure("histogram", "numpy", lambda: len(np.unique(stacked, axis=0, return_counts=True)[1]),
                              repeats=3, phase="run", **p)
                for repr_ in ("sparse", "dense"):
                    need = md.need(scope, names, repr=repr_)
                    over = [r for r, v in need.items() if v > budget.limits[r]]
                    if over:
                        bench.refusal("histogram", "cpu", lambda: md.histogram(scope, names, repr=repr_, budget=budget),
                                      repr=repr_, **p)
                        continue
                    for backend in bench.backends():
                        h = bench.measure("histogram", backend,
                                          lambda: md.histogram(scope, names, repr=repr_, backend=backend),
                                          repeats=3, phase="build", repr=repr_, **p)
                        i = h.info
                        bench.records[-1].update(state_bytes=i["bytes"], cells=i["cells"], occupied=i["occupied"],
                                                 scans=i["scans"], passes=i["passes"], strategy=i["strategy"])
                        if i["occupied"] != len(expect) or i["rows"] != rows:
                            raise SystemExit(f"DIST-001: {repr_} histogram differs from DuckDB at {p}")
                con.close()
    # 64 groups; a near-row-sized support; structural zeros in a dense domain
    rows = 100_000 if bench.quick else 1_000_000
    vals, valid = columns(rng, rows, 64, 2)
    groups = rng.integers(0, 64, rows)
    scope = mbk.Scope.from_arrays(vals, valid=valid, groups=groups, ngroups=64)
    for repr_ in ("sparse", "dense"):
        for backend in bench.backends():
            h = bench.measure("histogram, 64 groups", backend, lambda: md.histogram(scope, ["c0", "c1"], repr=repr_,
                                                                                    backend=backend),
                              repeats=3, phase="build", rows=rows, support=64, dims=2, groups=64, repr=repr_)
            bench.records[-1].update(state_bytes=h.bytes, cells=h.info["cells"], strategy=h.info["strategy"])
    near = {"x": rng.integers(0, 2 * rows, rows).astype(np.int64)}
    scope = mbk.Scope.from_arrays(near)
    for strategy in ("count", "sort"):
        h = bench.measure("near-row-sized support", "cpu", lambda: md.histogram(scope, ["x"], strategy=strategy),
                          repeats=2, phase="build", rows=rows, strategy=strategy)
        bench.records[-1].update(state_bytes=h.bytes, occupied=h.info["occupied"], passes=h.info["passes"])
    con = duckdb.connect()
    con.register("near", pd.DataFrame(near))
    bench.measure("near-row-sized support", "duckdb", lambda: con.execute("SELECT x, COUNT(*) FROM near GROUP BY x")
                  .fetchall(), repeats=2, phase="run", rows=rows)
    con.close()
    even = {"x": (2 * rng.integers(0, 512, rows)).astype(np.int64)}
    scope = mbk.Scope.from_arrays(even)
    for repr_ in ("sparse", "dense"):
        h = bench.measure("structural zeros (even values)", "cpu", lambda: md.histogram(scope, ["x"], repr=repr_),
                          repeats=3, phase="build", rows=rows, repr=repr_)
        bench.records[-1].update(state_bytes=h.bytes, cells=h.info["cells"], occupied=h.info["occupied"])
    # conversions and the dense joint-domain refusal threshold (support budget 2^20 cells per group)
    vals, valid = columns(rng, 100_000, 1024, 2, nulls=0)
    scope = mbk.Scope.from_arrays(vals)
    sparse = md.histogram(scope, ["c0", "c1"])
    dense = bench.measure("sparse -> dense", "cpu", lambda: sparse.to_dense(), repeats=3, phase="run",
                          cells=1024 * 1024)
    bench.measure("dense -> sparse", "cpu", lambda: dense.to_sparse(), repeats=3, phase="run", cells=1024 * 1024)
    wider = [(0, 1024, False), (0, 1023, False)]
    bench.refusal("sparse -> dense", "cpu", lambda: sparse.to_dense(wider), cells=1025 * 1024)
    bench.threshold("dense joint cells per group (support budget)", dict(cells=1024 * 1024),
                    dict(cells=1025 * 1024), budget_support=Budget().limits["support"])
    bench.note("DIST-001 backends: cpu = math_dist.c (single-threaded), cuda = math_dist_gpu.cu (device runs only), "
               "duckdb = GROUP BY with COUNT(*) over a registered DataFrame (DuckDB's own threads), numpy = "
               "np.unique(axis=0, return_counts=True) with NULL encoded as -1.")
    bench.note("Every build scans the scope's rows once (scans = 1); a sparse state is counted in a dense scratch and "
               "compacted (strategy count, 2 passes over the scratch) or built by a stable merge sort of one entry per "
               "row (strategy sort). The scope (FND-004 snapshot) is built once per grid point and measured apart.")


# ---------------------------------------------------------------------- DIST-002
def request_batch(rng, kind, n, M, lo, hi):
    if kind == "cdf":
        return md.requests([dict(op="range", hi=Fraction(int(t), 2), closed=(True, True))
                            for t in rng.integers(2 * lo - 2, 2 * hi + 3, n)])
    if kind == "tail":
        return md.requests([dict(op="range", lo=Fraction(int(t), 2), closed=(False, True))
                            for t in rng.integers(2 * lo - 2, 2 * hi + 3, n)])
    if kind == "quantile":
        return md.requests([dict(op="quantile", p=Fraction(int(a), 1000), rule="cont") for a in rng.integers(0, 1001, n)])
    pairs = np.sort(rng.integers(1, M + 1, (n, 2)), axis=1)
    return md.requests([dict(op="trimmed", r1=int(a), r2=int(b), k=1) for a, b in pairs])


def dist002(bench, rng):
    rows = 20_000 if bench.quick else 100_000
    cases = [("support 8", rng.integers(0, 8, rows)), ("support 64", rng.integers(0, 64, rows)),
             ("support 1024", rng.integers(0, 1024, rows)),
             ("structural zeros (even values of 0..2046)", 2 * rng.integers(0, 1024, rows)),
             ("near-row-sized support", rng.integers(0, 2 * rows, rows))]
    for name, x in cases:
        x = x.astype(np.int64)
        valid = rng.random(rows) >= 0.05
        scope = mbk.Scope.from_arrays({"x": x}, valid={"x": valid})
        p0 = dict(rows=rows, support=name)
        h = bench.measure("histogram (construction)", "cpu", lambda: md.histogram(scope, ["x"]), repeats=3,
                          phase="build", **p0)
        pre = bench.measure("prefix state (construction)", "cpu", lambda: h.prefix("count", 1), repeats=3,
                            phase="build", **p0)
        bench.records[-1].update(state_bytes=pre.info["bytes"], support_entries=pre.info["support"])
        xs = np.sort(x[valid])
        M, lo, hi = len(xs), int(xs[0]), int(xs[-1])
        bench.measure("numpy sort (reconstruction)", "numpy", lambda: np.sort(x[valid]), repeats=3, phase="build", **p0)
        con = duckdb.connect()
        con.register("data", pd.DataFrame({"x": nullable(x, valid)}))
        dev_uploaded = False
        for kind in ("cdf", "tail", "quantile", "trimmed"):
            for n in (1, 100, 10_000):
                reqs = request_batch(rng, kind, n, M, lo, hi)
                p = dict(p0, kind=kind, requests=n)
                for backend in bench.backends():
                    if backend == "cuda" and not dev_uploaded:
                        bench.measure("prefix upload (construction)", "cuda", lambda: pre.device(), repeats=1,
                                      phase="build", **p0)
                        dev_uploaded = True
                    out = bench.measure("requests", backend, lambda: pre.query_arrays(reqs, backend=backend),
                                        repeats=5, phase="run", **p)
                    if (out["status"] != 0).any():
                        raise SystemExit(f"DIST-002: refused requests at {p}")
                # baselines and a check on a sample of the answers
                if kind in ("cdf", "tail"):
                    num, den = ((reqs["b_num"], reqs["b_den"]) if kind == "cdf" else (reqs["a_num"], reqs["a_den"]))
                    th = num / den                               # halves of small integers: exact in float64

                    def numpy_batch():
                        return (np.searchsorted(xs, th, side="right") if kind == "cdf"
                                else M - np.searchsorted(xs, th, side="right"))
                    got = bench.measure("requests (retained sort)", "numpy", numpy_batch, repeats=5, phase="run", **p)
                    mass = out["mass"][:, 0]
                    if not np.array_equal(mass, got):
                        raise SystemExit(f"DIST-002: {kind} differs from NumPy at {p}")
                    op = "<=" if kind == "cdf" else ">"
                    if n <= 100:
                        sqls = [f"SELECT COUNT(*) FROM data WHERE x {op} {float(t)}" for t in th]
                        bench.measure("requests (one query each)", "duckdb",
                                      lambda: [con.execute(s).fetchall()[0][0] for s in sqls], repeats=2, phase="run", **p)
                    else:                                    # one scan: a cumulative histogram, then an ASOF join
                        con.register("thr", pd.DataFrame({"t": th}))
                        cum = ("WITH cum AS (SELECT x, SUM(COUNT(*)) OVER (ORDER BY x) AS c FROM data WHERE x IS NOT "
                               "NULL GROUP BY x) SELECT thr.t, " +
                               ("COALESCE(cum.c, 0)" if kind == "cdf" else
                                "(SELECT COUNT(x) FROM data) - COALESCE(cum.c, 0)") +
                               " FROM thr ASOF LEFT JOIN cum ON thr.t >= cum.x")
                        bench.measure("requests (cumulative histogram + ASOF join)", "duckdb",
                                      lambda: con.execute(cum).fetchall(), repeats=2, phase="run", **p)
                elif kind == "quantile":
                    levels = (reqs["a_num"] / reqs["a_den"]).tolist()     # p = a / 1000, stored reduced
                    if n <= 100:
                        sqls = [f"SELECT quantile_cont(x, {q}) FROM data" for q in levels]
                        bench.measure("requests (one query each)", "duckdb",
                                      lambda: [con.execute(s).fetchall()[0][0] for s in sqls], repeats=2, phase="run", **p)
                    else:
                        bench.measure("requests (one quantile_cont over a list)", "duckdb",
                                      lambda: con.execute("SELECT quantile_cont(x, ?) FROM data", [levels]).fetchall(),
                                      repeats=2, phase="run", **p)
                    bench.measure("requests (retained sort)", "numpy",
                                  lambda: np.quantile(xs, np.asarray(levels)), repeats=5, phase="run", **p)
                    num, den = md._i256(out["num"][0]), md._i128(out["den"][0])
                    ref = float(np.quantile(xs, levels[0]))
                    if abs(num / den - ref) > 1e-6 * max(1.0, abs(ref)):
                        raise SystemExit(f"DIST-002: quantile differs from NumPy at {p}")
                else:
                    csum = np.concatenate([[0], np.cumsum(xs)])
                    bench.measure("requests (retained sort + cumsum)", "numpy",
                                  lambda: csum[reqs["b_num"]] - csum[reqs["a_num"] - 1], repeats=5, phase="run", **p)
                    v = md._i256(out["num"][0])
                    if v != int(xs[int(reqs["a_num"][0]) - 1:int(reqs["b_num"][0])].sum()):
                        raise SystemExit(f"DIST-002: trimmed sum differs at {p}")
                # one-shot: construction + one batch versus the reconstruction
                if n == 1:
                    bench.measure("one-shot (histogram + prefix + 1 request)", "cpu",
                                  lambda: md.histogram(scope, ["x"]).prefix("count", 1).query_arrays(reqs),
                                  repeats=3, phase="first", **p)
        con.close()
    bench.note("DIST-002: requests reuse the retained prefix state: no scan of the rows per request (a DuckDB query "
               "scans its table each time). cpu = tr_dist_query (single-threaded binary searches), cuda = one work "
               "item per request (device runs only), numpy = complete reconstruction (np.sort once, then "
               "np.searchsorted / np.quantile / cumulative sums: floating quantiles, exact counts).")


def main():
    bench = Bench.from_args("DIST-001", __doc__)
    rng = np.random.default_rng(bench.seed)
    t0 = time.perf_counter()
    if bench.capability == "DIST-001":
        dist001(bench, rng)
    elif bench.capability == "DIST-002":
        dist002(bench, rng)
    else:
        raise SystemExit("--capability is DIST-001 or DIST-002")
    bench.note(f"grid wall time {time.perf_counter() - t0:.1f} s")
    bench.write()


if __name__ == "__main__":
    main()
