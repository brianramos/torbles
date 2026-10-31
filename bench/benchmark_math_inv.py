# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""INV-001 benchmark: linear and univariate polynomial inverse answers, and finite-domain searches.

    python bench/benchmark_math_inv.py --capability INV-001 --out FILE [--device] [--quick]

Case 1 (linear and bounded-degree univariate targets with zero / one / multiple / no solutions and domains with exact
boundary roots; solving reported separately from state construction). State construction ("build"): the RESP-001
response function of a table (MomentBasis + reduce of a squared loss, rows 1k / 100k), and first-slice expressions.
Solving ("run", first and warm samples): linear targets (unique, inconsistent, outside the box, whole domain, a box
family decided by Fourier-Motzkin, the least-squares zero of the RESP-001 gradient) and univariate targets of degree
2 .. 32 built from known factors (no root, one root, several simple roots, multiplicities, roots exactly at both
bounds), isolated and then refined to 2^-50 (the CPU sign evaluator; the CUDA kernel with --device). Baselines:
numpy.roots and numpy.linalg.lstsq (floating, uncertified: the receipt records whether their real-root count in the
bounds and their classification agree with the exact answer).
Case 2 (parameter dimension 1 / 2 / 3, finite domains of 8 / 16 / 32 values per coordinate, incomplete searches and
proof bounds): the exhaustive exact search (CPU; CUDA with --device) against a Python Fraction brute force (exact, the
reference) and a NumPy float64 brute force with a 1e-9 tolerance (uncertified: false matches are counted), plus
searches truncated by the iteration and output budgets (the examined prefix, the resume index) and the proof (value
bound bits, number of primes).
"""
import time
from fractions import Fraction

import numpy as np
import pandas as pd

from math_bench import Bench
import torbles
from torbles import math_inv as mi
from torbles.math_budgets import Budget
from torbles.mathematics import Context, MomentBasis


def pmul(a, b):
    out = [Fraction(0)] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            out[i + j] += x * y
    return out


def from_factors(factors):
    p = [Fraction(1)]
    for f, m in factors:
        for _ in range(m):
            p = pmul(p, f)
    return p


def univariate_cases(quick):
    """(name, coefficients, lo, hi, expected distinct real roots in [lo, hi])"""
    lin = lambda r: [-Fraction(r).numerator, Fraction(r).denominator]
    cases = [
        ("no root (x^2 + 1)", [1, 0, 1], -10, 10, 0),
        ("one root (2x - 3)", [-3, 2], -10, 10, 1),
        ("no root in the bounds (x^2 - 2 on [2, 3])", [-2, 0, 1], 2, 3, 0),
        ("boundary roots ((x - 1)(x - 3)^2 (x^2 - 2) on [1, 3])",
         from_factors([(lin(1), 1), (lin(3), 2), ([-2, 0, 1], 1)]), 1, 3, 3),
        ("multiplicities ((3x - 1)^3 (x + 2)^2 (x^2 - 5))",
         from_factors([(lin(Fraction(1, 3)), 3), (lin(-2), 2), ([-5, 0, 1], 1)]), None, None, 4),
        ("rational roots in a quartic factor ((97x - 89)(101x - 7)(x^2 - 3))",
         from_factors([([-89, 97], 1), ([-7, 101], 1), ([-3, 0, 1], 1)]), None, None, 4),
    ]
    degrees = (8, 16) if quick else (8, 16, 32)
    for d in degrees:                                           # degree d: d / 4 rational, 3 d / 4 irrational roots
        fs = [(lin(Fraction(k, 3)), 1) for k in range(1, d // 4 + 1)]
        fs += [([-Fraction(c), 0, 1], 1) for c in (2, 3, 5, 6, 7, 10, 11, 13, 14, 15, 17, 19)[:3 * d // 8]]
        p = from_factors(fs)
        cases.append((f"degree {len(p) - 1}: {d // 4} rational, {3 * d // 4} irrational roots", p, None, None, d))
    return cases


def numpy_real_roots(coef, lo, hi):
    r = np.roots([float(c) for c in reversed(coef)])
    real = r[np.abs(r.imag) <= 1e-9].real
    lo = -np.inf if lo is None else float(lo)
    hi = np.inf if hi is None else float(hi)
    return int(np.sum((real >= lo - 1e-12) & (real <= hi + 1e-12)))


def case_targets(bench):
    # ---------------------------------------------------------------- state construction: RESP-001 functions
    rng = np.random.default_rng(bench.seed)
    for rows in ((1000, 20_000) if bench.quick else (1000, 100_000)):
        x = rng.integers(-50, 51, rows)
        y = 3 * x + rng.integers(-20, 21, rows)
        db = torbles.from_frames({"r": pd.DataFrame({"x": x, "y": y})})
        t = db["r"]

        def construct():
            basis = MomentBasis(db.query(t), ["x", "y"])
            xv, yv = basis.variables["x"], basis.variables["y"]
            theta = basis.parameter("theta")
            return basis, basis.reduce((yv - theta * xv) ** 2)
        basis, loss = bench.measure("RESP-001 response function (moments + reduce)", "cpu", construct, repeats=1,
                                    phase="build", rows=rows)
        grad = loss.diff("theta")
        res = bench.measure("least-squares zero of the RESP-001 gradient", "cpu",
                            lambda: mi.solve([(grad, 0)]).kind, repeats=5, rows=rows)
        best = mi.solve([(grad, 0)]).point["theta"]
        bench.measure("RESP-001 loss = minimum + 1 (two irrational roots)", "cpu",
                      lambda: len(mi.roots(loss, loss.evaluate(theta=best) + 1, width=Fraction(1, 2 ** 50)).roots),
                      repeats=5, rows=rows)
        X = np.c_[x.astype(float)]
        bench.measure("least squares (numpy.linalg.lstsq, floating)", "numpy",
                      lambda: float(np.linalg.lstsq(X, y.astype(float), rcond=None)[0][0]), repeats=5, rows=rows)
        bench.note(f"RESP-001 at {rows} rows: exact theta = {best} ({res}); the response function is built once "
                   "(phase build), every later solve reads no rows.")
        basis.close()
    # ---------------------------------------------------------------- linear targets
    ctx = bench.measure("first-slice expressions", "cpu", lambda: Context(), repeats=1, phase="build")
    t, u, w = ctx.symbols("t", "u", "w")
    linear = [
        ("unique (2 x 2)", lambda: mi.solve([(2 * t + u, 3), (t - u, 0)])),
        ("empty: inconsistent", lambda: mi.solve_affine([[1, 1], [2, 2]], [1, 3])),
        ("empty: outside the box", lambda: mi.solve_affine([[1, 1]], [5], lo=[0, 0], hi=[1, 1])),
        ("whole domain (0 = 0)", lambda: mi.solve_affine([[0, 0]], [0], lo=[0, 0], hi=[1, 1])),
        ("unique by the box (x + y = 2, x, y <= 1)", lambda: mi.solve_affine([[1, 1]], [2], hi=[1, 1])),
        ("family: simplex in 3 unknowns", lambda: mi.solve([(t + u + w, 1)], domain={n: (0, None) for n in "tuw"})),
        ("family: simplex in 6 unknowns", lambda: mi.solve_affine([[1] * 6], [1], lo=[0] * 6)),
    ]
    for name, fn in linear:
        out = bench.measure(f"linear: {name}", "cpu", lambda: fn().kind, repeats=7)
        ans = fn()
        bench.records[-1]["result"] = dict(kind=ans.kind, reason=ans.reason, dims=ans.dims,
                                           fm_inequalities=ans.iterations)
        del out
    for m, n in ((4, 4), (8, 8), (16, 8)):                      # a random dense rank-deficient system
        rngl = np.random.default_rng(bench.seed + m)
        basis = rngl.integers(-9, 10, (min(m, n) - 1, n))
        A = (rngl.integers(-3, 4, (m, basis.shape[0])) @ basis).tolist()
        b = (np.array(A) @ rngl.integers(-5, 6, n)).tolist()
        bench.measure(f"linear: rank-deficient {m} x {n} (family, no box)", "cpu",
                      lambda: mi.solve_affine(A, b).kind, repeats=5, equations=m, unknowns=n)
        bench.measure(f"linear: rank-deficient {m} x {n} (lstsq, floating)", "numpy",
                      lambda: int(np.linalg.matrix_rank(np.array(A, dtype=float))), repeats=5, equations=m, unknowns=n)
    # ---------------------------------------------------------------- univariate targets
    for name, coef, lo, hi, want in univariate_cases(bench.quick):
        p = dict(target=name, degree=len(coef) - 1, lo=None if lo is None else str(lo), hi=None if hi is None else str(hi))
        r = bench.measure("isolation (exact)", "cpu", lambda: mi.roots(coef, 0, lo, hi), repeats=3, **p)
        bench.records[-1]["result"] = dict(roots=len(r.roots), exact=sum(x.exact for x in r.roots),
                                           expected=want, iterations=r.iterations)
        for backend in bench.backends():
            fine = bench.measure("refinement to 2^-50", backend, lambda: r.refine(Fraction(1, 2 ** 50), backend=backend),
                                 repeats=3, **p)
            bench.records[-1]["result"] = dict(rounds=fine.rounds, roots=len(fine.roots), backend=fine.backend)
        got = numpy_real_roots(coef, lo, hi)
        bench.measure("real roots (numpy.roots, floating)", "numpy", lambda: numpy_real_roots(coef, lo, hi),
                      repeats=3, **p)
        bench.records[-1]["result"] = dict(real_roots_in_bounds=got, exact_count=want, agrees=got == want,
                                           certified=False)


def oracle_sweep(f_terms, names, grid, target):
    """the exact brute force (Python Fractions): matching grid indices"""
    sizes = [len(grid[n]) for n in names]
    total = int(np.prod(sizes))
    hits = []
    for idx in range(total):
        rem, coords = idx, []
        for s in reversed(sizes):
            coords.append(rem % s)
            rem //= s
        coords.reverse()
        vals = [grid[n][c] for n, c in zip(names, coords)]
        acc = Fraction(0)
        for exps, coef in f_terms:
            term = Fraction(coef)
            for v, e in zip(vals, exps):
                if e:
                    term *= v ** e
            acc += term
        if acc == target:
            hits.append(idx)
    return hits


def case_sweeps(bench):
    rng = np.random.default_rng(bench.seed + 7)
    ctx = Context()
    t, u, w = ctx.symbols("t", "u", "w")
    polys = {1: t ** 3 - 2 * t * t - Fraction(5, 4) * t, 2: t * t * u - 3 * u + Fraction(1, 2) * t,
             3: t * u - w * w + Fraction(1, 3) * t * w}
    for dims in (1, 2, 3):
        f = polys[dims]
        names = ["t", "u", "w"][:dims]
        for size in (8, 16, 32):
            grid = {n: [Fraction(int(v), int(d)) for v, d in zip(rng.integers(-12, 13, size), rng.choice([1, 2, 4], size))]
                    for n in names}
            # plant a solution so that every grid has at least one match
            target = Fraction(0)
            point = {n: grid[n][size // 2] for n in names}
            terms = [(e[:dims], c) for e, c in f.terms]
            for exps, coef in terms:
                term = Fraction(coef)
                for n, e in zip(names, exps):
                    term *= point[n] ** e
                target += term
            p = dict(dimension=dims, values=size, grid_points=size ** dims)
            for backend in bench.backends():
                s = bench.measure("exhaustive exact search", backend, lambda: mi.sweep(f, grid, target, backend=backend),
                                  repeats=3, **p)
                bench.records[-1]["result"] = dict(matches=len(s.matches), complete=s.complete,
                                                   bound_bits=s.bound_bits, primes=s.primes)
            want = bench.measure("exact brute force (Python Fractions)", "python",
                                 lambda: len(oracle_sweep(terms, names, grid, target)), repeats=1, **p)
            # floating brute force: vectorized float64 with a tolerance (uncertified)
            mesh = np.meshgrid(*[np.array([float(v) for v in grid[n]]) for n in names], indexing="ij")

            def floating():
                acc = np.zeros_like(mesh[0])
                for exps, coef in terms:
                    term = np.full_like(mesh[0], float(coef))
                    for m, e in zip(mesh, exps):
                        if e:
                            term = term * m ** e
                    acc = acc + term
                return int(np.sum(np.abs(acc - float(target)) <= 1e-9))
            fl = bench.measure("float64 brute force (tolerance 1e-9)", "numpy", floating, repeats=3, **p)
            bench.records[-1]["result"] = dict(matches=fl, exact_matches=want, agrees=fl == want, certified=False)
            # incomplete searches: an iteration budget of a third of the grid, and an output budget of one match
            total = size ** dims
            part = bench.measure("search truncated by iterations (total / 3)", "cpu",
                                 lambda: mi.sweep(f, grid, target, budget=Budget(iterations=max(1, total // 3))),
                                 repeats=3, **p)
            bench.records[-1]["result"] = dict(complete=part.complete, reason=part.reason, searched=part.searched,
                                               next=part.next, matches=len(part.matches), bound_bits=part.bound_bits)
            out = mi.sweep(f, grid, target, budget=Budget(output=1))
            bench.records.append(dict(case="search truncated by the output budget (1 match)", backend="cpu",
                                      phase="run", params=p, result=dict(complete=out.complete, reason=out.reason,
                                                                         searched=out.searched, next=out.next,
                                                                         matches=len(out.matches))))
    # proof bounds that need several primes: large rational grid values and coefficients
    big = 10 ** 18 * t * u - Fraction(3, 10 ** 18) * u * u + 7 * t   # int64 rationals (the first slice)
    for size in (8, 16) if bench.quick else (8, 16, 32):
        grid = {"t": [Fraction(k, 10 ** 25 + k) for k in range(1, size + 1)],
                "u": [Fraction(10 ** 20 * k + 1, 3 ** k) for k in range(size)]}
        p = dict(dimension=2, values=size, grid_points=size * size, coefficients="up to 10^18, denominators 10^18")
        for backend in bench.backends():
            s = bench.measure("exhaustive exact search (large rationals)", backend,
                              lambda: mi.sweep(big, grid, 0, backend=backend), repeats=3, **p)
            bench.records[-1]["result"] = dict(matches=len(s.matches), complete=s.complete, bound_bits=s.bound_bits,
                                               primes=s.primes)
    bench.threshold("sweep coordinates", admitted=dict(coordinates=3), refused=dict(coordinates=4),
                    detail="more than 3 searched coordinates is UnsupportedError (TR_INV_SWEEP_DIMS)")
    bench.threshold("multimodular bound (signs)", admitted=dict(bound_bits=7806, primes=128),
                    refused=dict(bound_bits=7807, primes=129), detail="OverflowError past 128 primes (tests)")


def main():
    bench = Bench.from_args("INV-001", __doc__)
    t0 = time.perf_counter()
    case_targets(bench)
    case_sweeps(bench)
    bench.note("Exactness: every native answer is exact (rationals, isolating intervals with certified sign changes); "
               "the numpy rows are uncertified floating baselines, their 'agrees' flags record whether they "
               "happened to match the exact answer.")
    bench.note("CUDA is timed only with --device; the emulation is not a CUDA timing. The refinement on CUDA uses the "
               "device sign kernel once per bisection round (one kernel launch per round).")
    bench.note(f"wall time {time.perf_counter() - t0:.1f} s")
    bench.write()


if __name__ == "__main__":
    main()
