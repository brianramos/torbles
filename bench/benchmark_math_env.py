# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""ENV-001 benchmark: envelope builds and lookups against repeated all-group evaluation.

    python bench/benchmark_math_env.py --capability ENV-001 --out FILE [--device] [--quick]

Case 1: 8 / 64 / 1,024 group functions (quick: 8 / 64 / 256; affine and quadratic, exact rational coefficients) and
1 / 100 / 10,000 parameter values. Phases: "build" = the envelope (the exact sweep, INV-001's isolation of the
algebraic breakpoints), "run" = the lookups (binary search over the exact breakpoints: native CPU, CUDA with
--device) against the baseline the case names, repeated all-group evaluation: every group function evaluated exactly
at every value (the native all-group evaluator, CPU and CUDA), plus NumPy float64 evaluation as an inexact reference
(it can misplace ties; recorded with its disagreements). Every lookup's winner is checked equal to the exact
all-group evaluation's. Case 2: coincident curves (0 / 50 / 90% duplicated functions of 256), breakpoint count
(tangent lines of t^2: 16 .. 1,024 pieces (quick: 16 .. 256); parabolas of distinct curvatures whose neighbours cross
at irrational points: 8 / 32 / 128), parameter dimension (1, and 2 refused: UnsupportedError), the exact region /
output size of every build, and over-budget region cases (an output budget one below the exact region count:
refused with that exact count; the threshold recorded, never extrapolated).
"""
from fractions import Fraction as F

import numpy as np

from math_bench import Bench
from torbles import math_env as me
from torbles.math_budgets import Budget
from torbles.math_errors import UnsupportedError
from torbles.mathematics import AlgebraBudgetError


def functions(rng, n, kind="mixed"):
    out = []
    for _ in range(n):
        a, b = F(int(rng.integers(-10**4, 10**4)), int(rng.integers(1, 50))), F(int(rng.integers(-500, 500)), int(rng.integers(1, 20)))
        if kind == "affine" or (kind == "mixed" and rng.random() < 0.5):
            out.append((a, b))
        else:
            out.append((a, b, F(int(rng.integers(-20, 21)), int(rng.integers(1, 9)))))
    return out


def queries(rng, q, lim=1000):
    """q rationals in [-lim, lim] (denominators 1..100)"""
    d = rng.integers(1, 101, q)
    return [F(int(v), int(e)) for v, e in zip(rng.integers(-lim * d, lim * d + 1), d)]


def numpy_winners(fs, ts):
    c = np.array([[float(f[k]) if k < len(f) else 0.0 for k in range(3)] for f in fs])
    t = np.array([float(x) for x in ts])
    vals = c[:, 0][:, None] + c[:, 1][:, None] * t[None, :] + c[:, 2][:, None] * t[None, :] ** 2
    return vals.argmin(axis=0)


def case_sizes(bench, rng):
    for n in ((8, 64, 256) if bench.quick else (8, 64, 1024)):
        fs = functions(rng, n)
        p0 = dict(groups=n, kind="mixed")
        env = bench.measure("envelope build (exact sweep + INV-001 isolation)", "cpu",
                            lambda: me.envelope(fs, lo=-1000, hi=1000), repeats=2 if n >= 1024 else 3, phase="build", **p0)
        bench.records[-1].update(regions=env.info["regions"], points=env.info["points"],
                                 algebraic=env.info["algebraic"], iterations=env.info["iterations"])
        for q in (1, 100, 10_000):
            ts = queries(rng, q)
            p = dict(p0, values=q)
            for backend in bench.backends():
                got = bench.measure("lookups (binary search over the breakpoints)", backend,
                                    lambda: env.lookup(ts, backend=backend), repeats=3, phase="run", **p)
                bench.records[-1]["primes"] = env.last_lookup["primes"]
                ref = bench.measure("repeated all-group evaluation (exact, native)", backend,
                                    lambda: env.evaluate_all(ts, backend=backend), repeats=2 if q >= 10_000 else 3,
                                    phase="run", **p)
                if [a["winner"] for a in got] != [a["winner"] for a in ref]:
                    raise SystemExit(f"ENV-001: a lookup differs from the all-group evaluation ({p}, {backend})")
            fl = bench.measure("repeated all-group evaluation (NumPy float64, inexact)", "numpy",
                               lambda: numpy_winners(fs, ts), repeats=3, phase="run", **p)
            exact = [a["winner"] for a in got]
            bench.records[-1]["disagreements_with_exact"] = int(sum(1 for a, b, r in zip(exact, fl.tolist(), got)
                                                                    if r["status"] == "ok" and a != b))
            bench.measure("envelope build + lookups (one-shot)", "cpu",
                          lambda: me.envelope(fs, lo=-1000, hi=1000).lookup(ts), repeats=2, phase="first", **p)


def case_shapes(bench, rng):
    n = 256
    for dup in (0.0, 0.5, 0.9):                                    # coincident curves
        base = functions(rng, n)
        fs = [base[int(rng.integers(0, max(1, int(n * (1 - dup)))))] if rng.random() < dup else f for f in base]
        p = dict(groups=n, duplicated=dup)
        env = bench.measure("envelope build (coincident curves)", "cpu", lambda: me.envelope(fs, lo=-500, hi=500),
                            repeats=2, phase="build", **p)
        bench.records[-1].update(classes=env.info["classes"], regions=env.info["regions"], points=env.info["points"],
                                 coincident_max=max(len(r["groups"]) for r in env.regions if r["kind"] == "piece"))
    for m in ((16, 64, 256) if bench.quick else (16, 64, 256, 1024)):   # breakpoint count: every line wins a piece
        lines = [(-k * k, 2 * k) for k in range(-(m // 2), m - m // 2)]
        p = dict(groups=m, shape="tangent lines of t^2 (upper)")
        budget = None
        try:                                                       # the default iteration budget (2^20) first
            me.envelope(lines, upper=True)
        except AlgebraBudgetError:
            rec = bench.refusal("over-budget sweep (default iterations budget)", "cpu",
                                lambda: me.envelope(lines, upper=True), **p)
            budget = Budget(iterations=4 * rec["limit"])           # then measured with an explicit larger budget
            p = dict(p, iterations_budget=4 * rec["limit"])
        env = bench.measure("envelope build (breakpoint count)", "cpu",
                            lambda: me.envelope(lines, upper=True, budget=budget), repeats=2, phase="build", **p)
        bench.records[-1].update(regions=env.info["regions"], pieces=env.info["pieces"], points=env.info["points"],
                                 iterations=env.info["iterations"])
        ts = queries(rng, 1000)
        for backend in bench.backends():
            bench.measure("lookups (1,000 values)", backend, lambda: env.lookup(ts, backend=backend), repeats=3,
                          phase="run", **p)
        regions = env.info["regions"]
        if regions > 1:                                            # over budget: refused with the exact count
            over = Budget(budget or Budget(), output=regions - 1)
            rec = bench.refusal("over-budget regions (output = regions - 1)", "cpu",
                                lambda: me.envelope(lines, upper=True, budget=over), **p)
            if rec.get("need") != regions:
                raise SystemExit("ENV-001: an over-budget envelope did not report its exact region count")
            bench.threshold("regions (output budget)", admitted=dict(output=regions), refused=dict(output=regions - 1),
                            groups=m, exact_regions=regions)
    for m in (8, 32, 128):                                         # irrational breakpoints: parabolas of distinct
        fs = [(c * 9 * k * k - 5 * k, -6 * c * k, c)               # curvatures c (t - 3k)^2 - 5k, each winning near
              for k, c in ((k, 2 + F(k % 5, 3)) for k in range(m))]  # its minimum; neighbours cross irrationally
        p = dict(groups=m, shape="parabolas of distinct curvatures (irrational crossings)")
        env = bench.measure("envelope build (algebraic breakpoints)", "cpu", lambda: me.envelope(fs),
                            repeats=2, phase="build", **p)
        bench.records[-1].update(regions=env.info["regions"], points=env.info["points"], algebraic=env.info["algebraic"])
    from torbles.mathematics import Context                        # parameter dimension: 1 supported, 2 refused
    ctx = Context()
    th, u = ctx.symbols("theta", "u")
    p = dict(dimension=1)
    bench.measure("envelope of expressions (dimension 1)", "cpu",
                  lambda: me.from_expressions([(th - k) ** 2 for k in range(8)], "theta"), repeats=3, phase="build", **p)
    try:
        me.from_expressions([th * th + u, th], "theta")
        raise SystemExit("ENV-001: a two-dimensional parameter was accepted")
    except UnsupportedError as e:
        bench.records.append(dict(case="parameter dimension 2", backend="cpu", phase="refusal", params=dict(dimension=2),
                                  refused=True, error="UnsupportedError", message=str(e)))


def main():
    bench = Bench.from_args("ENV-001", __doc__)
    rng = np.random.default_rng(bench.seed)
    case_sizes(bench, rng)
    case_shapes(bench, rng)
    bench.note("Backends: cpu = math_env.c (single-threaded; the sweep and the lookups), cuda = math_env_gpu.cu "
               "(device runs only), numpy = float64 evaluation of every function at every value (inexact: its "
               "disagreements with the exact winners are recorded, not hidden).")
    bench.note("A build costs one exact difference and root computation per class per sweep step (steps = regions); a "
               "lookup costs a binary search over the points, each comparison the sign of an integer polynomial at t "
               "from residues modulo the job's primes; the all-group evaluation costs one exact comparison per class.")
    bench.note("ENV-001 is univariate: a parameter dimension of 2 is refused (UnsupportedError), not measured.")
    bench.note("The sweep forms one difference per class per step, so an envelope with as many pieces as groups costs "
               "about groups^2 differences: 1,024 tangent lines exceed the default iteration budget (2^20) and are "
               "recorded as that refusal, then measured with an explicit budget of 4 x 2^20.")
    bench.write()


if __name__ == "__main__":
    main()
