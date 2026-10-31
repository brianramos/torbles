# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""RESP-002 benchmark: squared, absolute and hinge loss curves from one retained prefix state.

    python bench/benchmark_math_resp.py --capability RESP-002 --out FILE [--device] [--quick]

Grid 1: 1k / 100k / 1M rows (quick: 1k / 100k) over 1024 distinct values, the squared, absolute and hinge losses,
1 / 10 / 100 / 10,000 theta requests (exact rationals). Construction (the sparse histogram, then the prefix state of
degree 2) is measured apart from evaluation (first batch and warm batches). Baselines: NumPy direct evaluation (one
pass over the rows per theta, float64) and DuckDB (SUM over a cross join of the rows with a theta table); a baseline
whose work rows x thetas exceeds 2 x 10^8 is not run (recorded as such). Native answers are checked against NumPy on
every grid point (float64 tolerance) and against exact Python sums on a sample.
Grid 2: distinct breakpoints from 2 to the supported limit (the default support budget, 2^20 cells per group; quick:
a budget of 4096) at fixed rows, with the refusal one past the limit recorded; one-shot requests (construction plus a
single theta) against one NumPy pass, where retained-state construction loses; the complete piecewise function
(pieces) and the minimizer set per loss.
CUDA (--device only): the upload of the prefix state, then the batches.
"""
import time
from fractions import Fraction

import duckdb
import numpy as np
import pandas as pd

from math_bench import Bench
from torbles import math_bank as mbk, math_dist as md, math_resp as mr
from torbles.math_budgets import Budget

LOSSES = ("squared", "abs", "hinge")


def numpy_loss(loss, x, t):
    u = x - t
    if loss == "squared":
        return float(np.dot(u, u))
    if loss == "abs":
        return float(np.abs(u).sum())
    return float(np.maximum(u, 0).sum())


def duck_expr(loss):
    return {"squared": "(x - t) * (x - t)", "abs": "ABS(x - t)", "hinge": "GREATEST(x - t, 0)"}[loss]


def thetas(rng, n, lo, hi):
    return [Fraction(int(a), 4) for a in rng.integers(4 * lo - 8, 4 * hi + 9, n)]


def grid1(bench, rng):
    for rows in ((1_000, 100_000) if bench.quick else (1_000, 100_000, 1_000_000)):
        x = rng.integers(0, 1024, rows).astype(np.int64)
        xf = x.astype(np.float64)
        scope = mbk.Scope.from_arrays({"x": x})
        p0 = dict(rows=rows, breakpoints=int(len(np.unique(x))))
        h = bench.measure("histogram (construction)", "cpu", lambda: md.histogram(scope, ["x"]), repeats=3,
                          phase="build", **p0)
        pre = bench.measure("prefix state (construction)", "cpu", lambda: h.prefix("count", 2), repeats=3,
                            phase="build", **p0)
        bench.records[-1].update(state_bytes=pre.info["bytes"])
        if bench.device:
            bench.measure("prefix upload (construction)", "cuda", lambda: pre.device(), repeats=1, phase="build", **p0)
        con = duckdb.connect()
        con.register("data", pd.DataFrame({"x": x}))
        for loss in LOSSES:
            for m in (1, 10, 100, 10_000):
                ts = thetas(rng, m, 0, 1023)
                reqs = mr.requests(loss, ts)
                p = dict(p0, loss=loss, requests=m)
                for backend in bench.backends():
                    out = bench.measure("evaluate", backend, lambda: mr.evaluate_arrays(pre, reqs, backend=backend),
                                        repeats=5, phase="run", **p)
                    if (out["status"] != 0).any():
                        raise SystemExit(f"RESP-002: refused requests at {p}")
                tf = np.array([float(t) for t in ts])
                if rows * m <= 200_000_000:
                    ref = bench.measure("evaluate (direct pass per theta)", "numpy",
                                        lambda: [numpy_loss(loss, xf, t) for t in tf], repeats=2 if m > 100 else 3,
                                        phase="run", **p)
                    for r, e in zip(out, ref):
                        v = md._i256(r["value"]) / md._i128(r["value_den"])
                        if abs(v - e) > 1e-9 * max(1.0, abs(e)):
                            raise SystemExit(f"RESP-002: {loss} differs from NumPy at {p}")
                    con.register("thetas", pd.DataFrame({"t": tf}))
                    bench.measure("evaluate (cross join)", "duckdb",
                                  lambda: con.execute(f"SELECT t, SUM({duck_expr(loss)}) FROM data, thetas GROUP BY t")
                                  .fetchall(), repeats=2, phase="run", **p)
                else:
                    bench.records.append(dict(case="evaluate (direct pass per theta)", backend="numpy", phase="run",
                                              params=p, skipped=f"rows x thetas = {rows * m} > 2e8"))
                # exact check on a sample against Python integers
                if m <= 10 and rows <= 100_000:
                    xs = x.tolist()
                    for r, t in list(zip(out, ts))[:2]:
                        v = Fraction(md._i256(r["value"]), md._i128(r["value_den"]))
                        u = [xi - t for xi in xs]
                        e = (sum(a * a for a in u) if loss == "squared" else sum(abs(a) for a in u) if loss == "abs"
                             else sum(max(a, 0) for a in u))
                        if v != e:
                            raise SystemExit(f"RESP-002: inexact {loss} at {p}")
        con.close()


def grid2(bench, rng):
    rows = 1 << 16 if bench.quick else 1 << 20
    budget = Budget(support=4096) if bench.quick else Budget()
    limit = budget.limits["support"]
    last = None
    for d in (2, 16, 256, 4096, 65536, 1 << 20):
        if d > rows:
            break
        if d > limit:
            break
        x = (rng.permutation(rows) % d).astype(np.int64)            # exactly d distinct breakpoints
        scope = mbk.Scope.from_arrays({"x": x})
        p = dict(rows=rows, breakpoints=d)
        h = bench.measure("histogram (construction)", "cpu", lambda: md.histogram(scope, ["x"], budget=budget),
                          repeats=2, phase="build", **p)
        pre = bench.measure("prefix state (construction)", "cpu", lambda: h.prefix("count", 2, budget=budget),
                            repeats=2, phase="build", **p)
        bench.records[-1].update(state_bytes=pre.info["bytes"])
        last = dict(p)
        for loss in LOSSES:
            reqs = mr.requests(loss, thetas(rng, 10_000, 0, d))
            for backend in bench.backends():
                bench.measure("evaluate 10,000", backend, lambda: mr.evaluate_arrays(pre, reqs, backend=backend),
                              repeats=3, phase="run", loss=loss, **p)
            L = mr.Loss(pre, loss)
            bench.measure("minimizers", "cpu", lambda: L.minimizers()["kind"], repeats=3, phase="run", loss=loss, **p)
            if d <= 4096:
                bench.measure("complete function (pieces)", "cpu", lambda: len(L.pieces()), repeats=2, phase="run",
                              loss=loss, **p)
            one = mr.requests(loss, [Fraction(d, 3)])
            bench.measure("one-shot (histogram + prefix + 1 theta)", "cpu",
                          lambda: mr.evaluate_arrays(md.histogram(scope, ["x"], budget=budget).prefix("count", 2,
                                                                                                     budget=budget), one),
                          repeats=2, phase="first", loss=loss, **p)
            xf = x.astype(np.float64)
            bench.measure("one-shot (one direct pass)", "numpy", lambda: numpy_loss(loss, xf, d / 3), repeats=3,
                          phase="first", loss=loss, **p)
    # one past the supported limit: the support budget refuses the state before anything is built
    over = limit + 1
    x = np.arange(over, dtype=np.int64)
    scope = mbk.Scope.from_arrays({"x": x})
    rec = bench.refusal("histogram (construction)", "cpu", lambda: md.histogram(scope, ["x"], budget=budget),
                        rows=over, breakpoints=over)
    bench.threshold("distinct breakpoints per group (support budget)", last, dict(rows=over, breakpoints=over),
                    budget_support=limit, refused_resource=rec.get("resource"))


def main():
    bench = Bench.from_args("RESP-002", __doc__)
    if bench.capability != "RESP-002":
        raise SystemExit("--capability is RESP-002")
    rng = np.random.default_rng(bench.seed)
    t0 = time.perf_counter()
    grid1(bench, rng)
    grid2(bench, rng)
    bench.note("Backends: cpu = math_resp.c over math_dist.c prefix states (single-threaded: two binary searches and "
               "exact 256-bit arithmetic per theta), cuda = one work item per theta (device runs only), numpy = one "
               "float64 pass over the rows per theta, duckdb = SUM over a cross join (DuckDB's own threads).")
    bench.note("Construction (histogram, prefix state) is paid once; each later theta costs O(log breakpoints) with "
               "no scan of the rows. A single theta is cheaper as one direct pass (the one-shot records).")
    bench.note(f"grid wall time {time.perf_counter() - t0:.1f} s")
    bench.write()


if __name__ == "__main__":
    main()
