# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""FND-010 benchmark: expansion versus factored fallback at polynomial, histogram and Boolean domain boundaries,
and the measured admission/refusal thresholds (never extrapolated).

    python bench/benchmark_math_budgets.py --capability FND-010 --out FILE [--device] [--quick]

Polynomial: (x0 + ... + x(n-1))^d expanded by the native kernel (C(n+d-1, d) terms, refused past 128) versus its
factored form (the linear factor evaluated natively, then raised natively); the budget module's choice is recorded.
Histogram and Boolean: the exact distinct-cell pre-pass (native CPU; CUDA with --device) against a NumPy np.unique
baseline, up to and past the states budget.
"""
import numpy as np

from math_bench import Bench
from torbles import math_budgets as mb
from torbles.mathematics import Context, AlgebraBudgetError


def polynomial(bench):
    kernel = mb.Budget.polynomial_kernel()
    ctx = Context()
    xs = ctx.symbols(*(f"x{i}" for i in range(8)))
    point = {f"x{i}": i + 1 for i in range(8)}
    admitted, refused = {}, {}
    for n in (2, 3, 4, 6, 8):
        lin = xs[0]
        for x in xs[1:n]:
            lin = lin + x
        for d in (1, 2, 3, 4, 6, 8):
            terms = mb.monomials(n, d, homogeneous=True)
            decision = mb.choose(kernel, dict(degree=d, support=terms), float(terms * n),
                                 dict(degree=1, support=n), float(n + d))
            bench.records.append(dict(case="polynomial choice", backend="cpu", phase="decision",
                                      params=dict(n=n, d=d, expanded_terms=terms), result=decision["choice"],
                                      reason=decision["reason"], expanded_refusal=decision["expanded_refusal"]))
            try:
                expanded = bench.measure("polynomial expanded", "cpu", lambda: lin ** d, repeats=5, phase="build",
                                         n=n, d=d, terms=terms)
                value = bench.measure("polynomial expanded", "cpu", lambda: expanded(**point), repeats=5,
                                      phase="evaluate", n=n, d=d, terms=terms)
                admitted[n] = max(admitted.get(n, 0), d)
            except AlgebraBudgetError:
                bench.refusal("polynomial expanded", "cpu", lambda: lin ** d, n=n, d=d, terms=terms)
                refused[n] = min(refused.get(n, 99), d)
                value = None

            def factored():
                v = lin(**point)
                return (ctx.constant(v) ** d)(**point)
            got = bench.measure("polynomial factored", "cpu", factored, repeats=5, phase="evaluate", n=n, d=d)
            if value is not None and got != value:
                raise SystemExit(f"factored and expanded values differ at n={n}, d={d}: {got} != {value}")
    for n in admitted:
        bench.threshold("polynomial expanded degree", dict(n=n, d=admitted[n]),
                        dict(n=n, d=refused[n]) if n in refused else None, budget="128 terms, degree 8")


def cells(bench, rows):
    rng = np.random.default_rng(bench.seed)
    budget = mb.Budget(states=1 << 24, output=1 << 24, bytes=1 << 30)
    # dense joint histograms of three columns: domain = product of the dimensions
    last_ok = None
    for dims in ((16, 16, 16), (64, 64, 64), (128, 128, 256), (256, 256, 256), (256, 256, 512)):
        ncells = mb.product(dims)
        cols = [rng.integers(0, m, size=rows, dtype=np.int64) for m in dims]
        cell = ((cols[0] * dims[1] + cols[1]) * dims[2] + cols[2]).astype(np.uint32)
        p = dict(dims=list(dims), cells=ncells, rows=rows)
        if ncells > budget.limits["states"]:
            bench.refusal("histogram pre-pass", "cpu", lambda: mb.count_cells(cell, ncells, budget=budget), **p)
            bench.threshold("histogram dense cells", last_ok, p, budget_states=budget.limits["states"])
            continue
        for backend in bench.backends():
            bench.measure("histogram pre-pass", backend,
                          lambda: mb.count_cells(cell, ncells, budget=budget, backend=backend), repeats=5, **p)
        bench.measure("histogram pre-pass", "numpy", lambda: int(len(np.unique(cell))), repeats=3, **p)
        last_ok = p
    # Boolean domains: k predicates over the rows, an assignment = the k predicate bits
    last_ok = None
    for k in (4, 8, 12, 16, 20, 24, 25):
        bits = rng.random((rows, k)) < 0.5
        cell = (bits.astype(np.uint32) << np.arange(k, dtype=np.uint32)).sum(axis=1).astype(np.uint32)
        ncells = mb.power2(k)
        p = dict(predicates=k, cells=ncells, rows=rows)
        if ncells > budget.limits["states"]:
            bench.refusal("boolean pre-pass", "cpu", lambda: mb.count_cells(cell, ncells, budget=budget), **p)
            bench.threshold("boolean assignments", last_ok, p, budget_states=budget.limits["states"])
            continue
        for backend in bench.backends():
            bench.measure("boolean pre-pass", backend,
                          lambda: mb.count_cells(cell, ncells, budget=budget, backend=backend), repeats=5, **p)
        bench.measure("boolean pre-pass", "numpy", lambda: int(len(np.unique(cell))), repeats=3, **p)
        last_ok = p


def main():
    bench = Bench.from_args("FND-010", __doc__)
    polynomial(bench)
    cells(bench, 100_000 if bench.quick else 1_000_000)
    bench.note("Costs passed to the representation choice are relative work units (terms x variables for the "
               "expanded evaluation, n + d for the factored one), not times.")
    bench.write()


if __name__ == "__main__":
    main()
