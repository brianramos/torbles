# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""CALC-003 benchmark: non-polynomial finite-separation calculus (separable phase and exponential families).

    python bench/benchmark_math_calc.py --capability CALC-003 --out FILE [--device] [--quick]

Case 1 (one to eight parameters, 8 / 32 / 128 terms, derivative order 1 / 2 / 4; construction and coefficient
transforms reported separately): the retained state of sum_r w_r prod_i sin(omega_i x_ri + phi_i) (2^k coefficients)
and of sum_r w_r exp(sum_i theta_i + a_i x_ri) (one coefficient) is built from a scope of 8 / 32 / 128 rows ("build":
certified per-row enclosures and exact 128-bit sums; CPU, and CUDA with --device), against the NumPy float64
computation of the same coefficients (uncertified baseline); then derivatives of order 1 / 2 / 4 in the first and the
last parameter ("run": exact permutations and negations of the coefficients).
Case 2 (1 / 100 / 10,000 evaluations or bound pairs after one retained-state build; zero families; degree and
coefficient refusal boundaries): one build over 10,000 rows (2,000 with --quick), then 1 / 100 / 10,000 certified
evaluations F(phi) and definite integrals over (a, b) pairs, against direct NumPy evaluation that rescans the rows for
every point (sum w sin(omega x + phi), sum w (cos(theta + a) - cos(theta + b))); a two-parameter family; the zero
family (an empty selection: exactly [0, 0]); the retained-state cache (a hit against a new construction for another
frequency); and the refusal boundaries: coefficients (support 2^k), derivative order (degree), evaluation points
(output), each measured at the largest admitted and the smallest refused point.
"""
import time
from fractions import Fraction

import numpy as np

from math_bench import Bench
from torbles import math_bank as mbk, math_calc as mc
from torbles.math_budgets import Budget


def scope_of(rng, rows, k):
    cols = {f"x{i}": rng.integers(-10_000, 10_001, rows) for i in range(k)}
    cols["w"] = rng.integers(-50, 51, rows)
    return mbk.Scope.from_arrays(cols, scales={f"x{i}": 2 for i in range(k)}), cols


def numpy_phase_coefficients(cols, oms, k):
    w = cols["w"].astype(float)
    th = [cols[f"x{i}"].astype(float) / 100 * float(oms[i]) for i in range(k)]
    s, c = [np.sin(t) for t in th], [np.cos(t) for t in th]
    out = []
    for S in range(2 ** k):
        prod = w.copy()
        for i in range(k):
            prod *= c[i] if (S >> i) & 1 else s[i]
        out.append(float(prod.sum()))
    return out


def case_parameters(bench):
    rng = np.random.default_rng(bench.seed)
    ks = (1, 2, 4, 8)
    for k in ks:
        oms = [Fraction(i + 1, 7) for i in range(k)]
        for rows in (8, 32, 128):
            scope, cols = scope_of(rng, rows, k)
            factors = [(f"x{i}", oms[i]) for i in range(k)]
            p = dict(parameters=k, terms=rows, coefficients=2 ** k)
            for backend in bench.backends():
                F = bench.measure("phase family: construction", backend,
                                  lambda: mc.phase(scope, factors, weight="w", backend=backend), repeats=3,
                                  phase="build", **p)
            F = mc.phase(scope, factors, weight="w")
            lo, hi = F.coefficients()[0]
            bench.records[-1]["result"] = dict(width_c0=hi - lo, scale=F.info["scale"])
            bench.measure("phase coefficients (numpy float64, uncertified)", "numpy",
                          lambda: len(numpy_phase_coefficients(cols, oms, k)), repeats=3, phase="build", **p)
            for order in (1, 2, 4):
                for param in sorted({0, k - 1}):
                    bench.measure("phase family: derivative (coefficient transform)", "cpu",
                                  lambda: F.derivative(param, order), repeats=5, order=order, parameter=param, **p)
            pe = dict(parameters=k, terms=rows, coefficients=1)
            rates = [(f"x{i}", Fraction(1, 1000 * (i + 1))) for i in range(k)]
            for backend in bench.backends():
                bench.measure("exponential family: construction", backend,
                              lambda: mc.exponential(scope, rates, weight="w", backend=backend), repeats=3,
                              phase="build", **pe)
            G = mc.exponential(scope, rates, weight="w")
            for order in (1, 2, 4):
                bench.measure("exponential family: derivative (coefficient transform)", "cpu",
                              lambda: G.derivative(0, order), repeats=5, order=order, **pe)


def case_evaluations(bench):
    rng = np.random.default_rng(bench.seed + 1)
    rows = 2_000 if bench.quick else 10_000
    scope, cols = scope_of(rng, rows, 2)
    omega = Fraction(3, 7)
    F = bench.measure("retained state (one build)", "cpu", lambda: mc.phase(scope, [("x0", omega)], weight="w"),
                      repeats=1, phase="build", rows=rows, parameters=1)
    x = cols["x0"].astype(float) / 100 * float(omega)
    w = cols["w"].astype(float)
    for n in (1, 100, 10_000):
        phis = np.linspace(-20, 20, n)
        p = dict(rows=rows, parameters=1, evaluations=n)
        out = bench.measure("certified evaluations F(phi)", "cpu", lambda: F.evaluate(phis), repeats=3, **p)
        bench.records[-1]["result"] = dict(max_width=max(hi - lo for lo, hi in out))
        bench.measure("direct evaluation (numpy, rescans the rows)", "numpy",
                      lambda: [float(np.sum(w * np.sin(x + phi))) for phi in phis], repeats=3 if n < 10_000 else 1,
                      **p)
        a, b = phis - 1.0, phis + 0.5
        bench.measure("certified definite integrals over (a, b) pairs", "cpu", lambda: F.definite(0, a, b), repeats=3,
                      **p)
        bench.measure("direct integrals (numpy, rescans the rows)", "numpy",
                      lambda: [float(np.sum(w * (np.cos(x + ai) - np.cos(x + bi)))) for ai, bi in zip(a, b)],
                      repeats=3 if n < 10_000 else 1, **p)
    F2 = mc.phase(scope, [("x0", omega), ("x1", Fraction(-5, 11))], weight="w")
    pts = rng.uniform(-5, 5, (10_000, 2))
    bench.measure("certified evaluations, two parameters", "cpu", lambda: F2.evaluate(pts), repeats=3, rows=rows,
                  parameters=2, evaluations=10_000)
    # the zero family: an empty selection gives exactly [0, 0]
    empty = mbk.Scope.from_arrays({"x0": cols["x0"], "w": cols["w"]}, rows=np.zeros(rows, dtype=bool))
    Z = bench.measure("zero family (empty selection): construction", "cpu",
                      lambda: mc.phase(empty, [("x0", omega)], weight="w"), repeats=3, phase="build", rows=rows)
    zv = bench.measure("zero family: 10,000 evaluations", "cpu", lambda: Z.evaluate(np.linspace(-1, 1, 10_000)),
                       repeats=3, rows=rows)
    bench.records[-1]["result"] = dict(all_exact_zero=all(v == (0.0, 0.0) for v in zv))
    # retained states: a hit against a new construction for another frequency
    cache = mc.Retained()
    cache.phase(scope, [("x0", omega)], weight="w")
    bench.measure("retained cache: hit (same identity)", "cpu", lambda: cache.phase(scope, [("x0", omega)], weight="w"),
                  repeats=5, rows=rows)
    k = iter(range(10 ** 6))
    bench.measure("retained cache: new construction (another frequency)", "cpu",
                  lambda: cache.phase(scope, [("x0", Fraction(1, 2 + next(k)))], weight="w"), repeats=5, rows=rows)
    bench.records[-1]["result"] = dict(**cache.stats)
    # refusal boundaries: coefficients (support), derivative order (degree), evaluation points (output)
    small, _ = scope_of(rng, 32, 9)
    for kk in (8, 9):
        factors = [(f"x{i}", Fraction(1, i + 2)) for i in range(kk)]
        bench.measure("coefficient boundary: admitted", "cpu",
                      lambda: len(mc.phase(small, factors, budget=Budget(support=2 ** kk)).coefficients()), repeats=1,
                      parameters=kk, support=2 ** kk)
        bench.refusal("coefficient boundary: refused", "cpu",
                      lambda: mc.phase(small, factors, budget=Budget(support=2 ** kk - 1)), parameters=kk,
                      support=2 ** kk - 1)
    bench.threshold("coefficients (support)", admitted=dict(parameters=9, support=512),
                    refused=dict(parameters=9, support=511))
    bench.measure("derivative order boundary: admitted", "cpu", lambda: F.derivative(0, 4, budget=Budget(degree=4)),
                  repeats=1, order=4, degree=4)
    bench.refusal("derivative order boundary: refused", "cpu", lambda: F.derivative(0, 5, budget=Budget(degree=4)),
                  order=5, degree=4)
    bench.threshold("derivative order (degree)", admitted=dict(order=4, degree=4), refused=dict(order=5, degree=4))
    bench.measure("evaluation points boundary: admitted", "cpu",
                  lambda: len(F.evaluate(np.zeros(10_000), budget=Budget(output=10_000))), repeats=1, points=10_000)
    bench.refusal("evaluation points boundary: refused", "cpu",
                  lambda: F.evaluate(np.zeros(10_001), budget=Budget(output=10_000)), points=10_001)
    bench.threshold("evaluation points (output)", admitted=dict(points=10_000), refused=dict(points=10_001))


def main():
    bench = Bench.from_args("CALC-003", __doc__)
    t0 = time.perf_counter()
    case_parameters(bench)
    case_evaluations(bench)
    bench.note("Native values are rigorous enclosures (math_calc.h precision contract); the numpy rows are float64 "
               "baselines without an error bound. CUDA is timed only with --device (the emulation is not a timing).")
    bench.note(f"wall time {time.perf_counter() - t0:.1f} s")
    bench.write()


if __name__ == "__main__":
    main()
