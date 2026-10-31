# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""OPT-001 benchmark: exact minimizers of quadratic objectives.

    python bench/benchmark_math_opt.py --capability OPT-001 --out FILE [--device] [--quick]

Quadratics: positive definite, singular (PSD, consistent and inconsistent) and indefinite objectives in 1 / 2 / 4 /
8 parameters, without and with a box [-1, 1]^n (and one free parameter): objective build ("build") and exact solve
("run") separately, with the status, the faces enumerated, those solved exactly and the filter's discards; the face
filter alone on the CPU and (with --device) on CUDA; baselines NumPy (float64 solve, unconstrained PD) and SciPy
L-BFGS-B (float64, a local method: its objective is recorded against the exact optimum, so a nonconvex miss shows).
Shared states: over 1k / 100k / 1M rows, one retained LIN-001 state serves repeated requests (ridge penalties and
boxes) against full objective recomputation per request (the state rebuilt from the rows; DuckDB exact HUGEINT
sums; NumPy float64 normal equations); an infeasible box and an unbounded objective are kept as recorded cases.
"""
import numpy as np
from fractions import Fraction as F
import duckdb
import pandas as pd

from math_bench import Bench
from torbles import math_bank as mbk
from torbles import math_lin as lin
from torbles import math_opt as opt
from torbles.math_budgets import Budget
from torbles.math_errors import InfeasibleError


def matrix(rng, n, kind):
    B = rng.integers(-6, 7, (n, n))
    if kind == "pd":
        A = B.T @ B + np.eye(n, dtype=np.int64)
    elif kind == "singular":
        B[-1] = B[0]                                         # rank n - 1 (n >= 2), 0 for n = 1
        A = B.T @ B if n > 1 else np.zeros((1, 1), np.int64)
    else:
        A = B.T @ B - 40 * np.eye(n, dtype=np.int64)        # some negative eigenvalues
    return [[F(int(v)) for v in row] for row in A]


def quadratics(bench, rng):
    for n in (1, 2, 4, 8):
        for kind in ("pd", "singular", "singular-inconsistent", "indefinite"):
            A = matrix(rng, n, "singular" if kind.startswith("singular") else kind)
            b = [F(int(v)) for v in rng.integers(-20, 21, n)]
            if kind == "singular":                           # b in range(A)
                z = rng.integers(-3, 4, n)
                b = [sum(A[i][j] * int(z[j]) for j in range(n)) for i in range(n)]
            p = dict(parameters=n, kind=kind)
            q = bench.measure("objective build", "cpu", lambda: opt.Quadratic.from_matrices(A, b, 0), repeats=3,
                              phase="build", **p)
            r = bench.measure("minimize, unconstrained", "cpu", lambda: opt.minimize(q), repeats=3, phase="run", **p)
            bench.records.append(dict(case="unconstrained result", backend="cpu", phase="measure", params=p,
                                      status=r.status, certificate=r.certificate))
            if kind == "pd":
                An, bn = np.array(A, float), np.array(b, float)
                bench.measure("numpy float64 solve", "numpy", lambda: np.linalg.solve(An, bn), repeats=3, phase="run", **p)
            for label, bounds in (("box", [(-1, 1)] * n), ("box + free", [None] + [(-1, 1)] * (n - 1))):
                if label == "box + free" and n == 1:
                    continue
                pb = dict(p, bounds=label)
                for backend in bench.backends():
                    bench.measure("face filter", backend, lambda: opt.faces(q, bounds, backend), repeats=3,
                                  phase="run", **pb)
                    r = bench.measure("minimize, box", backend, lambda: opt.minimize(q, bounds, backend=backend),
                                      repeats=3, phase="run", **pb)
                bench.records.append(dict(case="box result", backend="cpu", phase="measure", params=pb, status=r.status,
                                          faces=r.faces, exact_checks=r.exact_checks, filtered=r.filtered,
                                          value=None if r.value is None else str(r.value)))
                if label == "box":
                    try:
                        from scipy.optimize import minimize as sp_min
                        An, bn = np.array(A, float), np.array(b, float)
                        fun = lambda t: 0.5 * t @ An @ t - bn @ t
                        jac = lambda t: An @ t - bn
                        res = bench.measure("scipy L-BFGS-B (local, float64)", "scipy",
                                            lambda: sp_min(fun, np.zeros(n), jac=jac, method="L-BFGS-B",
                                                           bounds=[(-1, 1)] * n), repeats=3, phase="run", **pb)
                        bench.records.append(dict(case="scipy against the exact optimum", backend="scipy", phase="measure",
                                                  params=pb, scipy_value=float(res.fun), exact_value=float(r.value),
                                                  missed=bool(res.fun > float(r.value) + 1e-9 * (1 + abs(float(r.value))))))
                    except ImportError:
                        bench.note("SciPy not installed: no L-BFGS-B baseline")


def shared_states(bench, rng):
    d = 4
    con = duckdb.connect()
    for rows in ((1000, 100_000) if bench.quick else (1000, 100_000, 1_000_000)):
        cols = {f"x{i}": rng.integers(-1000, 1000, rows) for i in range(d)}
        cols["y"] = 3 * cols["x0"] - 2 * cols["x1"] + cols["x2"] // 3 + rng.integers(-50, 50, rows)
        feats = [f"x{i}" for i in range(d)]
        requests = [(F(k, 2), None) for k in range(4)] + [(0, [None] + [(-1, 1)] * d), (1, [None] + [(0, 2)] * d),
                                                          (0, [(-5, 5)] + [(-3, F(5, 2))] * d), (F(1, 3), [None] * (d + 1))]
        p = dict(rows=rows, features=d, requests=len(requests))
        scope = mbk.Scope.from_arrays(cols)
        st = bench.measure("shared state build", "cpu", lambda: lin.moments(scope, feats, "y"), repeats=2, phase="build", **p)

        def from_state():
            out = []
            for ridge, bounds in requests:
                q = opt.Quadratic.from_moments(st, 0, ridge=ridge)
                out.append(opt.minimize(q, bounds))
            return out
        res = bench.measure("requests from the shared state", "cpu", from_state, repeats=3, phase="warm", **p)

        def rebuild():
            out = []
            for ridge, bounds in requests:
                s2 = lin.moments(mbk.Scope.from_arrays(cols), feats, "y")
                out.append(opt.minimize(opt.Quadratic.from_moments(s2, 0, ridge=ridge), bounds))
            return out
        res2 = bench.measure("full recomputation: state rebuilt per request", "reconstruction", rebuild,
                             repeats=1 if rows == 1_000_000 else 2, phase="run", **p)
        con.register("t", pd.DataFrame(cols))
        z = ["CAST(1 AS HUGEINT)"] + [f"CAST({f} AS HUGEINT)" for f in feats]
        sums = [f"SUM({a} * {b})" for a in z for b in z] + [f"SUM({a} * CAST(y AS HUGEINT))" for a in z] + \
            ["SUM(CAST(y AS HUGEINT) * CAST(y AS HUGEINT))"]
        sql = f"SELECT {', '.join(sums)} FROM t"
        k = d + 1

        def duck():
            out = []
            for ridge, bounds in requests:
                v = [int(x) for x in con.execute(sql).fetchall()[0]]
                A = [[2 * (v[a * k + b] + (ridge if a == b and a else 0)) for b in range(k)] for a in range(k)]
                b = [2 * v[k * k + a] for a in range(k)]
                out.append(opt.minimize(opt.Quadratic.from_matrices(A, b, v[-1]), bounds))
            return out
        res3 = bench.measure("full recomputation: DuckDB HUGEINT sums per request", "duckdb", duck,
                             repeats=1 if rows == 1_000_000 else 2, phase="run", **p)
        for a, b, c in zip(res, res2, res3):
            if not (a.same(b) and a.same(c)):
                raise SystemExit("OPT-001: shared-state optimum differs from the recomputation")
        X = np.stack([np.ones(rows)] + [cols[f].astype(np.float64) for f in feats], 1)
        yv = cols["y"].astype(np.float64)

        def numpy_unconstrained():
            return [np.linalg.solve(X.T @ X + float(r) * np.diag([0.0] + [1.0] * d), X.T @ yv)
                    for r, bounds in requests if bounds is None]
        bench.measure("numpy float64 normal equations per request (unconstrained only)", "numpy", numpy_unconstrained,
                      repeats=2, phase="run", **p)
        con.unregister("t")
        bench.records.append(dict(case="shared-state results", backend="cpu", phase="measure", params=p,
                                  statuses=[r.status for r in res]))
        del scope, st, cols, X
    # the face budget's measured threshold: 3^8 faces of an 8-parameter box
    q = opt.Quadratic.from_matrices([[2 if i == j else 0 for j in range(8)] for i in range(8)], [1] * 8, 0)
    opt.minimize(q, [(-1, 1)] * 8, budget=Budget(iterations=6561))
    bench.refusal("box faces budget", "cpu", lambda: opt.minimize(q, [(-1, 1)] * 8, budget=Budget(iterations=6560)),
                  parameters=8)
    bench.threshold("box faces (8 parameters)", admitted=dict(iterations=6561), refused=dict(iterations=6560))
    # kept cases: an infeasible box (refused) and an unbounded objective (with its certificate)
    q = opt.Quadratic.from_matrices([[1, 0], [0, 1]], [0, 0], 0)
    try:
        opt.minimize(q, [(1, 0), (0, 1)])
        bench.records.append(dict(case="infeasible box", backend="cpu", phase="refusal", refused=False))
    except InfeasibleError as e:
        bench.records.append(dict(case="infeasible box", backend="cpu", phase="refusal", refused=True, error=type(e).__name__,
                                  message=str(e)))
    q = opt.Quadratic.from_matrices([[1, 2], [2, -1]], [1, 0], 0)          # negative curvature along the free t2
    r = bench.measure("unbounded objective (indefinite)", "cpu", lambda: opt.minimize(q, [(0, 1), None]), repeats=3,
                      phase="run", parameters=2)
    bench.records.append(dict(case="unbounded result", backend="cpu", phase="measure", status=r.status,
                              certificate=r.certificate, rate=str(r.rate)))
    q = opt.Quadratic.from_matrices([[1, 1], [1, 1]], [1, 0], 0)
    r = opt.minimize(q)
    bench.records.append(dict(case="unbounded result (inconsistent PSD)", backend="cpu", phase="measure", status=r.status,
                              certificate=r.certificate, rate=str(r.rate)))


def main():
    bench = Bench.from_args("OPT-001", __doc__)
    if bench.capability != "OPT-001":
        raise SystemExit("--capability OPT-001")
    rng = np.random.default_rng(bench.seed)
    quadratics(bench, rng)
    shared_states(bench, rng)
    if not bench.device:
        bench.note("CUDA face filter not timed (no --device): the emulation is never timed as CUDA")
    bench.write()


if __name__ == "__main__":
    main()
