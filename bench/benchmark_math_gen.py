# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""GEN-001 benchmark: generating polynomials of histogram supports (exact big-integer coefficients): construction,
coefficient / moment / evaluation requests, convolution powers and truncated series.

    python bench/benchmark_math_gen.py --capability GEN-001 --out FILE [--device] [--quick]

States: 100k rows (quick: 20k), supports 8 / 64 / 1024 values centred on 0 (negative values: the declared shift) and a
two-dimensional support (x by 16 values), sparse and dense; the histogram (DIST-001) is measured apart from the state.
Requests, measured separately: coefficient lookups (1 / 100 / 1,000), raw and factorial moments of orders 1..4, exact
evaluation at rational points (1 / 10 / 100, big rationals) and the batched 128-bit evaluation (1 / 100 / 10,000
points; CUDA with --device). Baselines: DuckDB GROUP BY counts and SUM of HUGEINT powers (checked equal), Python dict
polynomials with Fractions for the evaluations (small supports).
Convolutions: powers G^k for k = 4 / 16 / 64 of supports 8 / 64 / 1024 (support growth k (span - 1) + 1, the largest
coefficient's bits, the intermediate bytes the chain checks before its first product; refusals past the benchmark
budget of 2^24 coefficient pairs per product are recorded, never extrapolated), against explicit products of dict polynomials (support 8) and the FND-009
truncated series to orders 4 / 16 / 64. CUDA (--device): the 128-bit convolution G x G.
"""
from fractions import Fraction

import duckdb
import numpy as np
import pandas as pd

from math_bench import Bench
from torbles import math_bank as mbk, math_dist as md, math_gen as mg
from torbles.math_budgets import Budget
from torbles.mathematics import AlgebraBudgetError


def pmul(a, b):
    out = {}
    for u, x in a.items():
        for v, y in b.items():
            out[u + v] = out.get(u + v, 0) + x * y
    return out


def states(bench, rng):
    rows = 20_000 if bench.quick else 100_000
    for support in (8, 64, 1024):
        x = rng.integers(-(support // 2), support - support // 2, rows).astype(np.int64)
        y = rng.integers(0, 16, rows).astype(np.int64)
        valid = rng.random(rows) > 0.02
        scope = mbk.Scope.from_arrays({"x": x, "y": y}, valid={"x": valid})
        p = dict(rows=rows, support=support)
        con = duckdb.connect()
        con.register("t", pd.DataFrame({"x": pd.Series(np.where(valid, x, 0), dtype="Int64").where(valid), "y": y}))
        h = bench.measure("histogram (DIST-001)", "cpu", lambda: md.histogram(scope, ["x"]), repeats=2, phase="build", **p)
        h2 = md.histogram(scope, ["x", "y"])
        for repr_ in ("sparse", "dense"):
            g = bench.measure("generating state", "cpu", lambda: mg.generating(h, repr=repr_), repeats=3, phase="build",
                              repr=repr_, **p)
            bench.records[-1].update(terms=g.info["terms"], bytes=g.info["bytes"], shift=g.info["shift"][0])
            g2 = bench.measure("generating state (2-D support, x by 16)", "cpu",
                               lambda: mg.generating(h2, repr=repr_), repeats=2, phase="build", repr=repr_, **p)
            bench.records[-1].update(terms=g2.info["terms"], bytes=g2.info["bytes"])
        g = mg.generating(h)
        counts = dict(bench.measure("coefficients (GROUP BY)", "duckdb", lambda: con.execute(
            "SELECT x, COUNT(*) FROM t WHERE x IS NOT NULL GROUP BY x").fetchall(), repeats=2, **p))
        values = list(counts)
        for n in (1, 100, 1000):
            vs = [int(values[int(i)]) for i in rng.integers(0, len(values), n)]
            got = bench.measure("coefficient requests", "cpu", lambda: [g.coefficient(v) for v in vs], repeats=3,
                                phase="first+warm", requests=n, **p)
            if got != [counts[v] for v in vs]:
                raise SystemExit(f"GEN-001: coefficients differ from DuckDB at {p}")
        for k in (1, 2, 3, 4):
            m = bench.measure("raw moment (Euler operator at 1)", "cpu", lambda: g.moment(k), repeats=3, order=k, **p)
            sql = " * ".join(["CAST(x AS HUGEINT)"] * k)
            want = bench.measure("raw moment", "duckdb", lambda: con.execute(
                f"SELECT SUM({sql}) FROM t WHERE x IS NOT NULL").fetchone()[0], repeats=2, order=k, **p)
            if m != int(want):
                raise SystemExit(f"GEN-001: moment {k} differs from DuckDB at {p}")
            bench.measure("factorial moment (k-th derivative at 1)", "cpu", lambda: g.moment(k, falling=True),
                          repeats=3, order=k, **p)
        terms = g.terms()
        for n in (1, 10, 100):
            pts = [Fraction(int(a) * int(sg), int(b)) for a, sg, b in   # nonzero: z = 0 is a pole (negative values)
                   zip(rng.integers(1, 10, n), rng.choice([-1, 1], n), rng.integers(1, 6, n))]
            got = bench.measure("exact evaluation (big rationals)", "cpu", lambda: [g(z) for z in pts], repeats=2,
                                phase="first+warm", points=n, **p)
            if support <= 64 and n <= 10:
                def oracle(pts=pts):
                    return [sum(c * z ** v for v, c in terms.items()) for z in pts]
                want = bench.measure("exact evaluation (dict polynomial, Fractions)", "python", oracle, repeats=1,
                                     points=n, **p)
                if [Fraction(v) for v in got] != want:
                    raise SystemExit(f"GEN-001: evaluation differs from the dict polynomial at {p}")
        for n in (1, 100, 10_000):
            pts = [Fraction(int(a) * int(sg), int(b)) for a, sg, b in
                   zip(rng.integers(1, 4, n), rng.choice([-1, 1], n), rng.integers(1, 4, n))]
            for backend in bench.backends():
                bench.measure("batched 128-bit evaluation", backend,
                              lambda: g.evaluate_many(pts, backend=backend), repeats=3, phase="first+warm", points=n,
                              **p)
            exact = sum(v is not None for v in g.evaluate_many(pts))
            bench.records[-1].update(exact_in_128_bits=exact)
        con.close()


def convolutions(bench, rng):
    budget = Budget(tensor=1 << 24)                    # the benchmark's limit: 2^24 coefficient pairs per product
    for support in (8, 64, 1024):
        x = rng.integers(-(support // 2), support - support // 2, 5000).astype(np.int64)
        g = mg.generating(md.histogram(mbk.Scope.from_arrays({"x": x}), ["x"]))
        for k in (4, 16, 64):
            need = g.need_pow(k, budget=budget)
            p = dict(support=support, order=k, rows=5000)
            over = [r for r in ("tensor", "support", "bytes") if need[r] > budget.limits[r]]
            if over:
                bench.refusal("convolution power G^k", "cpu", lambda: g.pow(k, budget=budget), predicted=need, **p)
                continue
            r = bench.measure("convolution power G^k", "cpu", lambda: g.pow(k, budget=budget),
                              repeats=1 if k == 64 or (support == 1024 and k >= 16) else 2,
                              phase="build", **p)
            bench.records[-1].update(terms=r.info["terms"], max_bits=r.info["max_bits"], state_bytes=r.info["bytes"],
                                     intermediate_bytes_checked=need["bytes"], pairs_checked=need["tensor"],
                                     support_growth=need["terms"])
            if r.total() != 5000 ** k:
                raise SystemExit("GEN-001: G^k(1) != rows^k")
            if support == 8:
                A = g.terms()

                def explicit(A=A, k=k):
                    out = {0: 1}
                    for _ in range(k):
                        out = pmul(out, A)
                    return out
                want = bench.measure("convolution power (explicit dict products)", "python", explicit, repeats=1, **p)
                if r.terms() != {v: c for v, c in want.items() if c}:
                    raise SystemExit("GEN-001: G^k differs from the explicit products")
        # truncated series to orders 4 / 16 / 64: G^8 mod z^N natively, and the FND-009 series products
        for order in (4, 16, 64):
            p = dict(support=support, series_order=order, power=8)
            t = bench.measure("truncated power G^8 mod z^N", "cpu", lambda: g.pow(8, order=order), repeats=2, **p)
            s = g.series(order)

            def series_pow(s=s):
                acc = s
                for _ in range(7):
                    acc = acc * s
                return acc
            acc = bench.measure("truncated power (FND-009 series products)", "cpu", series_pow, repeats=2, **p)
            want = {e + 8 * g.shift[0]: int(c) for e, c in enumerate(acc.coefficients) if c}
            if t.terms() != want:
                raise SystemExit("GEN-001: the truncated power differs from the series products")
        for backend in bench.backends():
            if backend == "cuda":
                bench.measure("product G x G (128-bit convolution)", "cuda", lambda: g.mul(g, backend="cuda"),
                              repeats=3, support=support)
            else:
                bench.measure("product G x G", "cpu", lambda: g.mul(g), repeats=3, support=support)
    # the refusal threshold of the chain (support 1024): the largest admitted and the smallest refused power
    x = rng.integers(-512, 512, 5000).astype(np.int64)
    g = mg.generating(md.histogram(mbk.Scope.from_arrays({"x": x}), ["x"]))
    admitted = refused = None
    for k in range(2, 65):
        need = g.need_pow(k, budget=budget)
        if any(need[r] > budget.limits[r] for r in ("tensor", "support", "bytes")):
            refused = dict(power=k, **{r: need[r] for r in ("tensor", "support", "bytes")})
            bench.refusal("power past the benchmark budget (support 1024)", "cpu", lambda: g.pow(k, budget=budget),
                          power=k)
            break
        admitted = dict(power=k, **{r: need[r] for r in ("tensor", "support", "bytes")})
    bench.threshold("convolution power of a 1024-value support (2^24 pairs per product)", admitted,
                    refused or "none up to G^64", limits={r: budget.limits[r] for r in ("tensor", "support", "bytes")})


def main():
    bench = Bench.from_args("GEN-001", __doc__)
    rng = np.random.default_rng(bench.seed)
    states(bench, rng)
    convolutions(bench, rng)
    bench.note("cpu = math_gen.c (single-threaded; FND-009 big-integer coefficients, 128-bit accumulation when the "
               "inputs fit int64 and the bound holds); cuda = math_gen_gpu.cu (one work item per output coefficient or "
               "per point; device runs only); duckdb = GROUP BY counts and HUGEINT power sums; python = dict "
               "polynomials with Fractions (the explicit products).")
    bench.note("A power's need is checked for its largest step before the first product (support growth, coefficient "
               "pairs, intermediate bytes); a refused power records that need. The histogram is the only pass over "
               "the rows.")
    bench.write()


if __name__ == "__main__":
    main()
