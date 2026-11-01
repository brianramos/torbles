# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""ENV-002 benchmark: bounded multivariate parameter regions (the exact regions of a box where each group function,
affine in a parameter vector, attains the minimum or maximum; vertices, ties, point location), against repeated
all-group evaluation.

    python bench/benchmark_math_region.py --capability ENV-002 --out FILE [--device] [--quick]

Lookups (case 1): 8 / 64 / 1,024 group functions (quick: 8 / 64) over the box [-1, 1]^2: random affine functions with
integer coefficients in [-100, 100] (a few nonempty regions, the minimum) and the tangent planes of a paraboloid under
the maximum (every group a region: n regions); 1 / 100 / 10,000 parameter values on the 1/100 grid. Recorded: the
build (regions, vertices, boundaries, the index; state construction), then the lookups by the kd-tree index, by every
region's inequalities and by every class (the native all-group path) with the values as int64 arrays and the answers
as arrays (and the per-value path: Fraction tuples in, one dict out), with --device also on CUDA; the baselines: NumPy
evaluating every group at every value in exact int64 arithmetic (the values share the denominator 100) with argmin /
argmax (the smallest group on ties, as the tie policy), and Python Fractions (100 values). Every winner is checked
equal. The comparison of interest: one build plus the lookups against repeated all-group evaluation.
Structure (case 2): coincident functions (64 functions of which 0 / 8 / 32 copy others: classes and tied boundaries),
the breakpoint count (tangent families of 8 / 64 / 256 planes; quick: 8 / 64: vertices and regions; d = 1 against
ENV-001's breakpoints), the parameter dimension d = 1 .. 4 (64 random and 64 tangent functions; quick: 16); every size
recorded exactly (classes, regions, vertices, incidences, boundaries, index nodes, LPs, pivots, bytes). Over-budget
cases: the build refused at output = regions - 1 (the need reported is the region count; nothing is truncated) and
admitted at output = regions, likewise support = region-vertex incidences; d = 5 and quadratic functions without a
contract refused as unsupported; the piecewise-linear contract labelled approximate with its certified error.
"""
from fractions import Fraction

import numpy as np

from math_bench import Bench
from torbles import math_env as me
from torbles import math_region as mr
from torbles.math_budgets import Budget
from torbles.math_errors import UnsupportedError

DEN = 100


def random_affine(rng, n, d):
    return [tuple(int(a) for a in rng.integers(-100, 101, d + 1)) for _ in range(n)]


def tangents(rng, n, d):
    """the tangent planes of |theta|^2 at n distinct points p / 1000 of the box, times 10^6: -|p|^2 + 2000 p.theta;
    under the maximum every plane is the region around its point"""
    pts = set()
    while len(pts) < n:
        pts.add(tuple(int(a) for a in rng.integers(-1000, 1001, d)))
    return [tuple([-sum(q * q for q in p)] + [2000 * q for q in p]) for p in sorted(pts)]


def grid_values(rng, m, d):
    """m parameter values with coordinates k / 100, k in -100 .. 100 (integer numerators)"""
    return rng.integers(-DEN, DEN + 1, size=(m, d)).astype(np.int64)


def numpy_winners(fs, K, upper):
    """the baseline: every group at every value, exact int64 (100 f(k / 100) = 100 a0 + sum a_j k_j), argmin / argmax
    (the first index: the smallest group on ties)"""
    A = np.array(fs, dtype=np.int64)
    out = np.empty(len(K), dtype=np.int64)
    for s in range(0, len(K), 2000):
        Kc = K[s:s + 2000]
        V = A[:, :1] * DEN + A[:, 1:] @ Kc.T
        out[s:s + 2000] = (V.argmax(axis=0) if upper else V.argmin(axis=0))
    return out.tolist()


def fraction_winners(fs, pts, upper):
    out = []
    for p in pts:
        vals = [Fraction(f[0]) + sum(Fraction(a) * x for a, x in zip(f[1:], p)) for f in fs]
        best = max(vals) if upper else min(vals)
        out.append(vals.index(best))
    return out


def sizes(R):
    i = R.info
    kinds = {}
    for b in R.boundaries:
        kinds[b["kind"]] = kinds.get(b["kind"], 0) + 1
    return dict(classes=len(R.classes), regions=len(R.regions), vertices=len(R.vertices),
                boundaries=len(R.boundaries), boundary_kinds=str(sorted(kinds.items())),
                incidences=i["incidences"], index_nodes=i["nodes"], lps=i["lps"], pivots=i["pivots"],
                bytes=i["bytes"], claim=R.claim)


# ---------------------------------------------------------------------- case 1: build + lookups vs all-group evaluation
def lookups(bench, rng):
    ns = (8, 64) if bench.quick else (8, 64, 1024)
    ms = (1, 100, 10000)
    d = 2
    box = [(-1, 1)] * d
    for family in ("random", "tangent"):
        upper = family == "tangent"
        for n in ns:
            fs = random_affine(rng, n, d) if family == "random" else tangents(rng, n, d)
            p = dict(family=family, groups=n, d=d, upper=upper)
            R = bench.measure("regions build (classes, regions, vertices, boundaries, kd-tree index)", "cpu",
                              lambda: mr.regions(fs, box, upper=upper), phase="build",
                              repeats=1 if n >= 1024 else 3, **p)
            bench.records[-1].update(sizes(R))
            for m in ms:
                K = grid_values(rng, m, d)
                V = mr.Values(K, DEN)
                pts = [tuple(Fraction(int(k), DEN) for k in row) for row in K]
                q = dict(p, values=m)
                ref = None
                for mode in ("index", "regions", "all"):
                    for backend in bench.backends():
                        ans = bench.measure(f"lookup ({mode}; int64 value arrays, array answers)", backend,
                                            lambda b=backend, mo=mode: R.locate_arrays(V, mode=mo, backend=b),
                                            repeats=3, **q)
                        w = ans["winner"].tolist()
                        if ref is None:
                            ref = w
                        elif w != ref:
                            raise AssertionError(f"lookup {mode} on {backend} differs ({family}, {n}, {m})")
                ans = bench.measure("lookup (index; Fraction tuples, one dict per value)", "cpu",
                                    lambda: R.locate(pts), repeats=2, **q)
                if [a["winner"] for a in ans] != ref:
                    raise AssertionError(f"the per-value path differs ({family}, {n}, {m})")
                wn = bench.measure("all-group evaluation (NumPy exact int64, argmin / argmax)", "numpy",
                                   lambda: numpy_winners(fs, K, upper), repeats=3, **q)
                if wn != ref:
                    raise AssertionError(f"NumPy winners differ ({family}, {n}, {m})")
                if m <= 100:
                    wf = bench.measure("all-group evaluation (Python Fractions)", "python",
                                       lambda: fraction_winners(fs, pts, upper), repeats=1, **q)
                    if wf != ref:
                        raise AssertionError(f"Fraction winners differ ({family}, {n}, {m})")
            R.close()
    bench.note("lookups: the index, region and all-class lookups (and CUDA with --device), the NumPy exact int64 and "
               "the Fraction baselines give the same winner at every value; build and lookups are timed apart so "
               "build + m lookups can be set against m all-group evaluations")


# ---------------------------------------------------------------------- case 2: structure, sizes, refusals
def structure(bench, rng):
    small = bench.quick
    box2 = [(-1, 1)] * 2
    # coincident functions: copies of other groups merge into classes and tie on whole regions
    for copies in (0, 8, 32):
        fs = random_affine(rng, 64 - copies, 2)
        fs += [fs[int(i)] for i in rng.integers(0, len(fs), copies)]
        R = bench.measure("coincident functions", "cpu", lambda: mr.regions(fs, box2), phase="build", repeats=2,
                          groups=64, coincident=copies, d=2)
        tied = sum(1 for r in R.regions if len(r["groups"]) > 1)
        bench.records[-1].update(sizes(R), tied_regions=tied)
        R.close()
    # breakpoints: tangent families, every plane a region
    for n in ((8, 64) if small else (8, 64, 256)):
        fs = tangents(rng, n, 2)
        R = bench.measure("breakpoints: tangent planes (d = 2)", "cpu", lambda: mr.regions(fs, box2, upper=True),
                          phase="build", repeats=1, groups=n, d=2)
        bench.records[-1].update(sizes(R))
        R.close()
        f1 = tangents(rng, n, 1)
        R = bench.measure("breakpoints: tangent lines (d = 1)", "cpu",
                          lambda: mr.regions(f1, [(-1, 1)], upper=True), phase="build", repeats=2, groups=n, d=1)
        bench.records[-1].update(sizes(R))
        env = bench.measure("breakpoints: ENV-001 envelope (d = 1 baseline)", "env001",
                            lambda: me.envelope(f1, lo=-1, hi=1, upper=True), phase="build", repeats=2, groups=n, d=1)
        inner = sorted(v["point"][0] for v in R.vertices if v["point"][0] not in (-1, 1))
        if inner != sorted(env.breakpoints):
            raise AssertionError("d = 1 vertices differ from ENV-001's breakpoints")
        bench.records[-1].update(breakpoints=len(env.breakpoints))
        R.close()
    # the parameter dimension
    n = 16 if small else 64
    for d in (1, 2, 3, 4):
        for family in ("random", "tangent"):
            fs = random_affine(rng, n, d) if family == "random" else tangents(rng, n, d)
            R = bench.measure("dimension", "cpu", lambda: mr.regions(fs, [(-1, 1)] * d, upper=family == "tangent"),
                              phase="build", repeats=1, groups=n, d=d, family=family)
            bench.records[-1].update(sizes(R))
            R.close()
    # over-budget region cases: refused at limit + 1 with the exact need, never truncated
    fs = tangents(rng, 16 if small else 64, 2)
    R = mr.regions(fs, box2, upper=True)
    reg, inc = len(R.regions), R.info["incidences"]
    R.close()
    for resource, need in (("output", reg), ("support", inc)):
        bench.measure(f"build at the {resource} limit", "cpu",
                      lambda r=resource, v=need: mr.regions(fs, box2, upper=True, budget=Budget(**{r: v})),
                      phase="build", repeats=1, groups=len(fs), **{resource: need})
        rec = bench.refusal(f"build one past the {resource} limit", "cpu",
                            lambda r=resource, v=need: mr.regions(fs, box2, upper=True, budget=Budget(**{r: v - 1})),
                            groups=len(fs), **{resource: need - 1})
        if not rec["refused"] or rec["need"] != need:
            raise AssertionError(f"{resource}: the refusal does not report the exact need")
        bench.threshold(f"regions build, {resource} ({'nonempty regions' if resource == 'output' else 'incidences'})",
                        admitted={resource: need}, refused={resource: need - 1}, need_reported=need)
    for what, call in (("d = 5", lambda: mr.regions([(0, 1, 0, 0, 0, 0)], [(-1, 1)] * 5)),
                       ("quadratic without a contract", lambda: mr.regions([(0, 0, 0, 1, 0, 1), (1, 0, 0, 0, 0, 0)],
                                                                          box2))):
        try:
            call()
            raise AssertionError(f"{what} was not refused")
        except UnsupportedError as e:
            bench.note(f"refused as unsupported: {what} ({e})")
    quad = [(0, 0, 0, 1, 0, 1), (Fraction(1, 2), 0, 0, 0, 0, 0), (0, 1, -1, 0, 0, 0)]
    R = bench.measure("quadratics under the piecewise-linear contract (approximate)", "cpu",
                      lambda: mr.regions(quad, box2, contract=mr.PiecewiseLinear(4, Fraction(1, 10))),
                      phase="build", repeats=1, groups=3, d=2, pieces=4, error_budget="1/10")
    bench.records[-1].update(sizes(R), error=str(R.error))
    if R.claim != "approximate":
        raise AssertionError("the contract's result is not labelled approximate")
    R.close()


def main():
    bench = Bench.from_args("ENV-002", __doc__)
    rng = np.random.default_rng(bench.seed)
    lookups(bench, rng)
    structure(bench, rng)
    bench.write()


if __name__ == "__main__":
    main()
