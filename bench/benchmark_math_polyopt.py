# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""OPT-002 benchmark: bounded convex and polynomial optimization with feasibility and optimality evidence.

    python bench/benchmark_math_polyopt.py --capability OPT-002 --out FILE [--device] [--quick]

Case 1 (positive definite, singular and indefinite quadratics in 1 / 2 / 4 / 8 parameters, with and without box
constraints; matrix build and solve reported separately): the objective build (OPT-001 Quadratic.from_matrices,
"build") and the exact solve under no constraints, a box [-1, 1]^n, and the box with general rows A x <= b and an
equality ("run"), with the status, the method (simplex, active set, the certified branch and bound of an indefinite
QP), its certificate (KKT, a ray with its curvature, Farkas) and the iterations; baselines OPT-001's exact box
minimizer (where it applies: no rows) and SciPy (float64 local methods: L-BFGS-B on the box, SLSQP with rows; their
objective is recorded against the exact optimum, so a nonconvex miss shows).
Case 2 (repeated optimization over 1k / 100k / 1M source rows using shared states against full objective
recomputation; infeasible and unbounded cases retained): one LIN-001 moment state serves many constrained
least-squares requests (ridge penalties, a simplex constraint, nonnegativity, an order row) against the state rebuilt
per request and against DuckDB exact HUGEINT sums per request; an infeasible request (its Farkas certificate) and an
unbounded one (a collinear feature with a linear tilt: its ray) are kept as recorded cases.
Polynomial problems (the certified branch and bound of method 2): small objectives in 1 .. 4 variables at
tolerances 2^-10 and 2^-20 (complete enclosures), node budgets that stop early (incomplete, with the certified
bounds reached), an infeasible problem (leaf certificate), the batched box enclosure on the CPU (CUDA with --device),
against a Fraction grid of the box (the best grid value: an upper bound only) and SciPy local minimization.
"""
import time
from fractions import Fraction as F

import duckdb
import numpy as np
import pandas as pd

from math_bench import Bench
from torbles import math_bank as mbk
from torbles import math_lin as lin
from torbles import math_opt as opt
from torbles import math_polyopt as po
from torbles.math_budgets import Budget
from torbles.math_errors import UnsupportedError
from torbles.mathematics import AlgebraBudgetError, Context

try:
    from scipy.optimize import minimize as sp_min
except ImportError:                                               # pragma: no cover
    sp_min = None


def fq(v):
    return None if v is None else str(v)


def summary(r):
    return dict(status=r.status, method=r.method, unique=r.unique, handed=r.handed, convex=r.convex, kkt=r.kkt,
                certificate=r.certificate, iterations=r.iterations, lps=r.lps, value=fq(r.value), lower=fq(r.lower),
                upper=fq(r.upper), feasibility=r.feasibility, reason=r.reason,
                ray_curvature=fq(r.ray["curvature"]) if r.ray else None)


def matrix(rng, n, kind):
    B = rng.integers(-6, 7, (n, n))
    if kind == "pd":
        A = B.T @ B + np.eye(n, dtype=np.int64)
    elif kind.startswith("singular"):
        B[-1] = B[0]                                             # rank n - 1 (n >= 2), 0 for n = 1
        A = B.T @ B if n > 1 else np.zeros((1, 1), np.int64)
    else:
        A = B.T @ B - 40 * np.eye(n, dtype=np.int64)            # some negative eigenvalues
    return [[F(int(v)) for v in row] for row in A]


# ---------------------------------------------------------------------------------------------------- case 1
def quadratics(bench, rng):
    for n in (1, 2, 4, 8):
        for kind in ("pd", "singular", "singular-inconsistent", "indefinite"):
            A = matrix(rng, n, kind)
            b = [F(int(v)) for v in rng.integers(-20, 21, n)]
            if kind == "singular":                               # b in range(A): bounded below
                z = rng.integers(-3, 4, n)
                b = [sum(A[i][j] * int(z[j]) for j in range(n)) for i in range(n)]
            p = dict(parameters=n, kind=kind)
            q = bench.measure("objective build (matrices)", "cpu", lambda: opt.Quadratic.from_matrices(A, b, 0),
                              repeats=3, phase="build", **p)
            rows = [[int(v) for v in rng.integers(-3, 4, n)] for _ in range(n)]
            rhs = [int(rng.integers(0, 4)) for _ in range(n)]     # 0 is feasible
            settings = [("none", {}), ("box", dict(bounds=[(-1, 1)] * n)),
                        ("box + rows", dict(bounds=[(-1, 1)] * n, A_ub=rows, b_ub=rhs,
                                            A_eq=[[1] * n] if n > 1 else None, b_eq=[0] if n > 1 else None))]
            for label, kw in settings:
                pc = dict(p, constraints=label)
                try:
                    r = bench.measure("solve", "cpu", lambda: po.qp_from(q, **kw), repeats=3, phase="run", **pc)
                except UnsupportedError as e:                    # an indefinite problem the methods do not decide
                    bench.records.append(dict(case="solve refused", backend="cpu", phase="refusal", params=pc,
                                              refused=True, error=type(e).__name__, message=str(e)))
                    continue
                bench.records.append(dict(case="solve result", backend="cpu", phase="measure", params=pc, **summary(r)))
                if label == "none" or label == "box":            # OPT-001 applies: the same exact answer
                    bounds = kw.get("bounds")
                    o = bench.measure("OPT-001 exact minimizer", "opt001", lambda: opt.minimize(q, bounds),
                                      repeats=3, phase="run", **pc)
                    agree = (o.status in ("unique", "nonunique") and r.status in ("optimal", "complete") and
                             (o.value == r.value if r.status == "optimal" else r.lower <= o.value <= r.upper)) or \
                            (o.status == "unbounded" and r.status == "unbounded")
                    bench.records.append(dict(case="OPT-001 against OPT-002", backend="opt001", phase="measure",
                                              params=pc, opt001_status=o.status, opt001_value=fq(o.value),
                                              agree=bool(agree)))
                if sp_min is not None and label != "none" and r.status in ("optimal", "complete"):
                    An, bn = np.array(A, float), np.array(b, float)
                    fun = lambda t: 0.5 * t @ An @ t - bn @ t
                    jac = lambda t: An @ t - bn
                    if label == "box":
                        call = lambda: sp_min(fun, np.zeros(n), jac=jac, method="L-BFGS-B", bounds=[(-1, 1)] * n)
                        name = "scipy L-BFGS-B (local, float64)"
                    else:
                        cons = [dict(type="ineq", fun=lambda t, R=np.array(rows, float), h=np.array(rhs, float): h - R @ t)]
                        if n > 1:
                            cons.append(dict(type="eq", fun=lambda t: np.array([t.sum()])))
                        call = lambda: sp_min(fun, np.zeros(n), jac=jac, method="SLSQP", bounds=[(-1, 1)] * n,
                                              constraints=cons)
                        name = "scipy SLSQP (local, float64)"
                    res = bench.measure(name, "scipy", call, repeats=3, phase="run", **pc)
                    exact = float(r.value if r.status == "optimal" else r.upper)
                    bench.records.append(dict(case="scipy against the exact optimum", backend="scipy", phase="measure",
                                              params=pc, scipy_value=float(res.fun), exact_value=exact,
                                              missed=bool(res.fun > exact + 1e-7 * (1 + abs(exact)))))


# ---------------------------------------------------------------------------------------------------- case 2
def shared_states(bench, rng):
    d = 3
    con = duckdb.connect()
    feats = [f"x{i}" for i in range(d)]
    for rows in ((1000, 100_000) if bench.quick else (1000, 100_000, 1_000_000)):
        cols = {f: rng.integers(-1000, 1000, rows) for f in feats}
        cols["y"] = 3 * cols["x0"] - 2 * cols["x1"] + cols["x2"] // 3 + rng.integers(-50, 50, rows)
        cols["xs"] = cols["x0"] + cols["x1"]                     # collinear: the unbounded request
        k = d + 1                                                # (intercept, x0, x1, x2)
        requests = [  # ridge, constraints
            (0, {}),
            (F(1, 2), dict(A_eq=[[0, 1, 1, 1]], b_eq=[1])),                         # coefficients sum to 1
            (0, dict(bounds=[None, (0, None), (0, None), (0, None)])),             # nonnegative coefficients
            (1, dict(A_ub=[[0, -1, 1, 0]], b_ub=[0])),                              # beta_1 <= beta_0... order
            (0, dict(A_ub=[[0, 1, 0, 0], [0, -1, 0, 0]], b_ub=[4, -5])),           # beta_0 <= 4 and >= 5
        ]
        p = dict(rows=rows, features=d, requests=len(requests))
        scope = mbk.Scope.from_arrays(cols)
        st = bench.measure("shared state build (LIN-001 moments)", "cpu", lambda: lin.moments(scope, feats, "y"),
                           repeats=2, phase="build", **p)

        def from_state():
            return [po.qp_from(opt.Quadratic.from_moments(st, 0, ridge=ridge), **kw) for ridge, kw in requests]
        res = bench.measure("requests from the shared state", "cpu", from_state, repeats=3, phase="warm", **p)

        def rebuild():
            out = []
            for ridge, kw in requests:
                s2 = lin.moments(mbk.Scope.from_arrays(cols), feats, "y")
                out.append(po.qp_from(opt.Quadratic.from_moments(s2, 0, ridge=ridge), **kw))
            return out
        res2 = bench.measure("full recomputation: state rebuilt per request", "reconstruction", rebuild,
                             repeats=1 if rows == 1_000_000 else 2, phase="run", **p)
        frame = pd.DataFrame(cols)
        con.register("t", frame)
        z = ["CAST(1 AS HUGEINT)"] + [f"CAST({f} AS HUGEINT)" for f in feats]
        sums = [f"SUM({a} * {b2})" for a in z for b2 in z] + [f"SUM({a} * CAST(y AS HUGEINT))" for a in z] + \
            ["SUM(CAST(y AS HUGEINT) * CAST(y AS HUGEINT))"]
        sql = f"SELECT {', '.join(sums)} FROM t"

        def duck():
            out = []
            for ridge, kw in requests:
                v = [int(x) for x in con.execute(sql).fetchall()[0]]
                A = [[2 * (v[a * k + b2] + (ridge if a == b2 and a else 0)) for b2 in range(k)] for a in range(k)]
                bb = [2 * v[k * k + a] for a in range(k)]
                out.append(po.qp_from(opt.Quadratic.from_matrices(A, bb, v[-1]), **kw))
            return out
        res3 = bench.measure("full recomputation: DuckDB HUGEINT sums per request", "duckdb", duck,
                             repeats=1 if rows == 1_000_000 else 2, phase="run", **p)
        for a, b2, c in zip(res, res2, res3):
            if not (a.same(b2) and a.same(c)):
                raise SystemExit("OPT-002: shared-state optimum differs from the recomputation")
        con.unregister("t")
        bench.records.append(dict(case="shared-state results", backend="cpu", phase="measure", params=p,
                                  results=[summary(r) for r in res]))
        # the unbounded request: a collinear feature (xs = x0 + x1) and a linear tilt along the kernel direction
        s3 = bench.measure("collinear state build", "cpu", lambda: lin.moments(scope, ["x0", "x1", "xs"], "y"),
                           repeats=1, phase="build", **p)
        q3 = opt.Quadratic.from_moments(s3, 0)
        r3 = bench.measure("unbounded request (collinear features, tilted)", "cpu",
                           lambda: po.qp_from(q3, linear=[0, 0, 0, -1]), repeats=2, phase="run", **p)
        bench.records.append(dict(case="unbounded result", backend="cpu", phase="measure", params=p, **summary(r3),
                                  direction=[fq(v) for v in r3.ray["direction"]] if r3.ray else None,
                                  rate=fq(r3.ray["rate"]) if r3.ray else None))
        del scope, st, s3, cols, frame


# ---------------------------------------------------------------------------------------------------- polynomials
def polynomials(bench):
    ctx = Context()
    x, y, z, w = ctx.symbols("x", "y", "z", "w")
    cases = [  # label, objective, box, constraints, scipy objective
        ("1 variable, degree 4", x ** 4 - 3 * x ** 3 + x, {"x": (-2, 3)}, [], lambda v: v[0] ** 4 - 3 * v[0] ** 3 + v[0]),
        ("2 variables, double well", x ** 4 - 2 * x ** 2 + y ** 2 + F(1, 4) * x, {"x": (-2, 2), "y": (-1, 1)}, [],
         lambda v: v[0] ** 4 - 2 * v[0] ** 2 + v[1] ** 2 + 0.25 * v[0]),
        ("2 variables, constrained", x * y, {"x": (-1, 2), "y": (-3, 1)}, [(x + y, "<=", 0)],
         lambda v: v[0] * v[1]),
        ("3 variables, degree 3", x * y * z + x ** 2 - y, {"x": (-1, 1), "y": (-1, 1), "z": (-1, 1)},
         [(x + y + z, "<=", 1)], lambda v: v[0] * v[1] * v[2] + v[0] ** 2 - v[1]),
        ("4 variables, indefinite quadratic", x * y - z * w + F(1, 2) * (x ** 2 + w ** 2) - y * z,
         {"x": (-1, 1), "y": (-1, 1), "z": (-1, 1), "w": (-1, 1)}, [],
         lambda v: v[0] * v[1] - v[2] * v[3] + 0.5 * (v[0] ** 2 + v[3] ** 2) - v[1] * v[2]),
    ]
    tols = (F(1, 2 ** 10),) if bench.quick else (F(1, 2 ** 10), F(1, 2 ** 20))
    for label, f, box, cons, fun in cases:
        pr = bench.measure("polynomial problem build", "cpu", lambda: po.Problem(f, box, cons), repeats=3,
                           phase="build", kind=label)
        for tol in tols:
            for backend in bench.backends():
                p = dict(kind=label, tolerance=str(tol))
                r = bench.measure("certified branch and bound", backend, lambda: pr.minimize(tol, backend=backend),
                                  repeats=2, phase="run", **p)
            bench.records.append(dict(case="branch and bound result", backend="cpu", phase="measure", params=p,
                                      **summary(r), gap=fq(r.gap), pruned=r.pruned, open=r.open,
                                      incumbents=r.incumbents, point=[fq(v) for v in r.point or []]))
        # a node budget that stops the search: incomplete with the certified bounds reached
        for nodes in (1, 9, 33):
            try:
                ri = pr.minimize(F(1, 2 ** 20), budget=Budget(iterations=nodes))
            except AlgebraBudgetError as e:                      # one variable: INV-001's budget refuses
                bench.records.append(dict(case="node budget", backend="cpu", phase="refusal",
                                          params=dict(kind=label, iterations=nodes), refused=True,
                                          resource=e.resource, need=e.need, limit=e.limit, message=str(e)))
                continue
            bench.records.append(dict(case="node budget", backend="cpu", phase="measure",
                                      params=dict(kind=label, iterations=nodes), **summary(ri), gap=fq(ri.gap)))
            if ri.status == "complete":
                continue
            if ri.status != "incomplete" or ri.reason != "nodes":
                raise SystemExit(f"OPT-002: a node budget gave {ri.status} / {ri.reason}")
        # baselines: the best value on a Fraction grid of the box (feasible points only; an upper bound, no proof)
        names = list(box)
        steps = 16 if len(box) <= 2 else (4 if bench.quick else 8)
        axes = [[F(lo) + F(hi - lo) * i / steps for i in range(steps + 1)] for lo, hi in box.values()]

        def grid_best():
            import itertools
            best = None
            for pt in itertools.product(*axes):
                if any(not ((pr.value(pt, 1 + k) <= 0) if op == "<=" else (pr.value(pt, 1 + k) >= 0))
                       for k, (_, op, _) in enumerate(cons)):
                    continue
                v = pr.value(pt)
                best = v if best is None or v < best else best
            return best
        gb = bench.measure("Fraction grid of the box (no proof)", "python", grid_best, repeats=1, phase="run",
                           kind=label, points_per_variable=steps + 1)
        r = pr.minimize(F(1, 2 ** 10))
        rec = dict(case="grid and local baselines against the certified enclosure", backend="python",
                   phase="measure", params=dict(kind=label), grid_best=fq(gb), lower=fq(r.lower), upper=fq(r.upper),
                   grid_respects_lower=bool(gb is None or gb >= r.lower))
        if sp_min is not None and not cons:
            starts = [np.array([float(lo) for lo, _ in box.values()]), np.array([float(hi) for _, hi in box.values()]),
                      np.zeros(len(box))]
            res = bench.measure("scipy L-BFGS-B from 3 starts (local, float64)", "scipy",
                                lambda: min((sp_min(fun, s0, method="L-BFGS-B",
                                                    bounds=[(float(lo), float(hi)) for lo, hi in box.values()])
                                             for s0 in starts), key=lambda o: o.fun), repeats=2, phase="run",
                                kind=label)
            rec.update(scipy_value=float(res.fun), scipy_missed=bool(res.fun > float(r.upper) + 1e-6))
        bench.records.append(rec)
    # an infeasible problem: the leaves partition the box, each proven to violate a constraint
    r = bench.measure("infeasible problem (leaf certificate)", "cpu",
                      lambda: po.polynomial(x + y, {"x": (-1, 1), "y": (-1, 1)},
                                            [(x ** 2 + y ** 2, "<=", F(1, 4)), (x + y, ">=", 1)],
                                            tolerance=F(1, 100)), repeats=2, phase="run")
    bench.records.append(dict(case="infeasible result", backend="cpu", phase="measure", **summary(r),
                              leaves=len(r.leaves)))
    # the batched box enclosure: the data-parallel part (one work item per box on CUDA)
    pr = po.Problem(x ** 4 - 2 * x ** 2 + y ** 2 + x * y, {"x": (-2, 2), "y": (-2, 2)}, [(x + y, "<=", 1)])
    rng = np.random.default_rng(bench.seed)
    for count in ((1000,) if bench.quick else (1000, 10_000)):
        lo = rng.integers(-16, 15, (count, 2))
        boxes = [[(F(int(a), 8), F(int(a) + 1, 8)), (F(int(b), 8), F(int(b) + 1, 8))] for a, b in lo]
        for backend in bench.backends():
            bench.measure("box enclosures", backend, lambda: po.boxes(pr, boxes, ubound=0, backend=backend), repeats=3,
                          phase="run", boxes=count)
    # the polynomial budget's measured threshold: the support (terms) of a problem
    f = x ** 4 - 2 * x ** 2 + y ** 2 + F(1, 4) * x
    po.Problem(f, {"x": (-2, 2), "y": (-1, 1)}, budget=Budget(support=4))
    rec = bench.refusal("polynomial support budget", "cpu",
                        lambda: po.Problem(f, {"x": (-2, 2), "y": (-1, 1)}, budget=Budget(support=3)), terms=4)
    bench.threshold("polynomial support (terms)", admitted=dict(support=4),
                    refused=dict(support=3) if rec.get("refused") else None, need=rec.get("need"))


def main():
    bench = Bench.from_args("OPT-002", __doc__)
    if bench.capability != "OPT-002":
        raise SystemExit("--capability OPT-002")
    rng = np.random.default_rng(bench.seed)
    t0 = time.perf_counter()
    quadratics(bench, rng)
    shared_states(bench, rng)
    polynomials(bench)
    if not bench.device:
        bench.note("CUDA box enclosures not timed (no --device): the emulation is never timed as CUDA")
    bench.note("the simplex, the active set, the uniqueness LPs and the branch-and-bound control run on the CPU")
    bench.note(f"total {time.perf_counter() - t0:.1f} s")
    bench.write()


if __name__ == "__main__":
    main()
