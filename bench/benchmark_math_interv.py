# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""INV-002 benchmark: constrained minimal interventions (algebraic, never causal).

    python bench/benchmark_math_interv.py --capability INV-002 --out FILE [--device] [--quick]

Case 1 (linear and bounded-degree univariate targets with zero / one / multiple / no solutions and domains containing
exact boundary roots; solving reported separately from state construction). Build ("build"): first-slice target
expressions, a RESP-001 response function of a table (MomentBasis + reduce of a squared loss over 1k / 100k rows) and
the aggregate state of a query whose rows cross its predicate when a feature is shifted (1k / 100k / 1M rows).
Solving ("run", first and warm samples): linear targets with L1 / Linf / L2 / L0 costs (a unique change, several
optimal changes, a target already attained, an unattained strict infimum, an infeasible target) and in 2 .. 16 inputs
with general rows; univariate targets of degree 2 .. 8 built from known factors (no root in the domain, one root,
ties on both sides of x0, rational and irrational, multiplicities, roots exactly at both domain bounds); the
sublevel target of the RESP-001 loss; aggregate targets with rows entering and leaving. Baselines: SciPy linprog
(HiGHS, float64) for the L1 / Linf LPs, numpy.roots (float64, uncertified: whether its nearest root agrees), DuckDB
(the aggregate at the answer; the frozen row set of d = 0) and the aggregate state rebuilt per request.
Case 2 (parameter dimension 1 / 2 / 3, finite domains of 8 / 16 / 32 values per coordinate, incomplete searches and
proof bounds): the plan build (cleared integer forms, primes, cost order, enclosure pruning) and the exhaustive exact
search (CPU; CUDA with --device) against a Python Fraction brute force (exact, the reference) and a NumPy float64
brute force with a 1e-9 tolerance (uncertified: mismatches counted); searches truncated by the iteration and output
budgets keep their proven lower / upper bounds and resume index; the proof (bound bits, primes, pruned points).
"""
import itertools
import time
from fractions import Fraction as F

import duckdb
import numpy as np
import pandas as pd

from math_bench import Bench
import torbles
from torbles import math_bank as mbk
from torbles import math_interv as iv
from torbles.math_budgets import Budget
from torbles.mathematics import AlgebraBudgetError, Context, MomentBasis

try:
    from scipy.optimize import linprog
except ImportError:                                               # pragma: no cover
    linprog = None


def fq(v):
    return None if v is None else (str(v) if not isinstance(v, tuple) else [str(v[0]), str(v[1])])


def summary(r):
    return dict(status=r.status, method=r.method, cost=fq(r.cost), unique=r.unique, solutions=len(r.solutions),
                claim=r.claim, causal=r.causal, lower=fq(r.lower), upper=fq(r.upper))


# ---------------------------------------------------------------------------------------------------- case 1
def linear_targets(bench, rng):
    ctx = Context()
    t, u, w = ctx.symbols("t", "u", "w")
    f = bench.measure("first-slice target build", "cpu", lambda: 2 * t + 3 * u - w, repeats=3, phase="build", inputs=3)
    x0 = {"t": 0, "u": 0, "w": 0}
    cases = [
        ("one change", dict(target=f, op=">=", value=6)),
        ("several optimal changes", dict(target=t + u + 0 * w, op=">=", value=2)),
        ("already attained (zero change)", dict(target=f, op="<=", value=0)),
        ("strict infimum (unattained)", dict(target=f, op=">", value=6, d_bounds={"u": (0, 1), "t": (0, 1)})),
        ("no change attains it", dict(target=f, op=">=", value=100, d_bounds={k: (-2, 2) for k in x0})),
    ]
    for cost in ("l1", "linf", "l2", "l0"):
        for label, kw in cases:
            kw = dict(kw)
            target = kw.pop("target")
            p = dict(cost=cost, kind=label, inputs=3)
            r = bench.measure("linear target", "cpu", lambda: iv.linear(target, x0, cost=cost, **kw), repeats=3,
                              phase="run", **p)
            bench.records.append(dict(case="linear target result", backend="cpu", phase="measure", params=p, **summary(r)))
    # n inputs with general rows: the exact LP (L1, Linf) and projection (L2) against SciPy HiGHS (float64)
    for n in ((2, 4, 8) if bench.quick else (2, 4, 8, 16)):
        a = [int(v) for v in rng.integers(1, 6, n)]
        xs = [int(v) for v in rng.integers(-3, 4, n)]
        A = [[int(v) for v in rng.integers(-2, 3, n)] for _ in range(n)]
        b = [int(sum(ai * xi for ai, xi in zip(row, xs))) + int(rng.integers(1, 6)) for row in A]
        tv = sum(ai * xi for ai, xi in zip(a, xs)) + 3 * n
        ws = [F(int(rng.integers(1, 4)), int(rng.integers(1, 3))) for _ in range(n)]
        for cost in ("l1", "linf", "l2"):
            p = dict(cost=cost, inputs=n, rows=n)
            r = bench.measure("linear target, general rows", "cpu",
                              lambda: iv.linear((a, 0), xs, ">=", tv, cost=cost, weights=ws, A_ub=A, b_ub=b,
                                                d_bounds=[(-10, 10)] * n), repeats=1 if n >= 16 else 3, phase="run",
                              **p)
            bench.records.append(dict(case="linear target result", backend="cpu", phase="measure", params=p, **summary(r)))
            if linprog is None or cost == "l2":
                continue
            # variables (d, u): L1 min w^T u, u >= +-d; Linf min s, +-w_i d_i <= s
            wf = [float(v) for v in ws]
            if cost == "l1":
                c = [0.0] * n + wf
                G = [[(1.0 if j == i else 0.0) for j in range(n)] + [(-1.0 if j == i else 0.0) for j in range(n)]
                     for i in range(n)] + [[(-1.0 if j == i else 0.0) for j in range(n)] +
                                           [(-1.0 if j == i else 0.0) for j in range(n)] for i in range(n)]
                h = [0.0] * (2 * n)
                pad = [0.0] * n
            else:
                c = [0.0] * n + [1.0]
                G = [[(wf[i] if j == i else 0.0) for j in range(n)] + [-1.0] for i in range(n)] + \
                    [[(-wf[i] if j == i else 0.0) for j in range(n)] + [-1.0] for i in range(n)]
                h = [0.0] * (2 * n)
                pad = [0.0]
            G += [[-float(v) for v in a] + pad]                    # a^T (x0 + d) >= t
            h += [float(sum(ai * xi for ai, xi in zip(a, xs)) - tv)]
            for row, bb in zip(A, b):                               # A (x0 + d) <= b
                G.append([float(v) for v in row] + pad)
                h.append(float(bb - sum(ri * xi for ri, xi in zip(row, xs))))
            bounds = [(-10, 10)] * n + [(0, None)] * len(pad)
            res = bench.measure("scipy linprog HiGHS (float64)", "scipy",
                                lambda: linprog(c, A_ub=G, b_ub=h, bounds=bounds, method="highs"), repeats=3,
                                phase="run", **p)
            exact = None if r.cost is None else float(r.cost)
            bench.records.append(dict(case="scipy against the exact cost", backend="scipy", phase="measure", params=p,
                                      scipy_status=int(res.status), scipy_cost=None if res.status else float(res.fun),
                                      exact_cost=exact, agree=bool((res.status == 0) == (r.status == "optimal") and (
                                          exact is None or abs(float(res.fun) - exact) <= 1e-7 * (1 + abs(exact))))))


def pmul(a, b):
    out = [F(0)] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            out[i + j] += F(x) * F(y)
    return out


def univariate_targets(bench):
    """targets from known factors (coefficients low degree first): the exact product is the numpy input, the
    first-slice Expression built from the same factors is the native input"""
    ctx = Context()
    t, = ctx.symbols("t")
    cases = [  # label, factors, x0, x_bounds
        ("no root in the domain", [[1, 0, 1], [-5, 1]], 0, (-2, 2)),
        ("one root", [[-1, 3], [1, 0, 1]], 0, None),
        ("rational tie on both sides", [[-1, 1], [1, 1], [-3, 1]], 0, None),
        ("irrational tie (sqrt 2)", [[-2, 0, 1]], 0, None),
        ("multiplicity 3", [[-1, 1]] * 3 + [[4, 1]], 0, None),
        ("roots exactly at both bounds", [[1, 1], [-2, 1], [3, 0, 1]], F(1, 2), (-1, 2)),
        ("degree 8, irrational roots", [[-2, 0, 1], [-3, 0, 1], [-5, 0, 1], [-7, 0, 1]], F(1, 3), None),
    ]

    def build():
        out = []
        for label, factors, x0, xb in cases:
            f = 1
            for fac in factors:
                f = f * sum(F(c) * t ** k for k, c in enumerate(fac))
            out.append(f)
        return out
    exprs = bench.measure("univariate targets build (first-slice expressions)", "cpu", build, repeats=3, phase="build",
                          cases=len(cases))
    for (label, factors, x0, xb), f in zip(cases, exprs):
        dense = [F(1)]
        for fac in factors:
            dense = pmul(dense, fac)
        kw = {} if xb is None else dict(x_bounds=xb)
        for cost in ("l1", "l2"):
            p = dict(kind=label, cost=cost, degree=len(dense) - 1)
            r = bench.measure("univariate target", "cpu", lambda: iv.univariate(f, {"t": x0}, "=", 0, cost=cost, **kw),
                              repeats=3, phase="run", **p)
            bench.records.append(dict(case="univariate result", backend="cpu", phase="measure", params=p, **summary(r),
                                      changes=[fq(sol.change["t"]) for sol in r.solutions],
                                      exact=[sol.exact for sol in r.solutions]))
        lo, hi = (None, None) if xb is None else xb

        def nearest():                                            # float64, no certificate
            rts = np.roots([float(c) for c in dense[::-1]])
            real = [float(z.real) for z in rts if abs(z.imag) < 1e-7 and (lo is None or z.real >= lo - 1e-12) and
                    (hi is None or z.real <= hi + 1e-12)]
            return min((abs(z - float(x0)) for z in real), default=None)
        dist = bench.measure("numpy.roots nearest root (float64)", "numpy", nearest, repeats=3, phase="run", kind=label)
        rr = bench.measure("univariate target, isolating intervals refined to 2^-40", "cpu",
                           lambda: iv.univariate(f, {"t": x0}, "=", 0, width=F(1, 2 ** 40), **kw), repeats=3,
                           phase="run", kind=label)
        exact = None                                              # the exact distance (an enclosure when irrational)
        if rr.solutions:
            c = rr.solutions[0].change["t"]
            exact = (abs(c), abs(c)) if not isinstance(c, tuple) else tuple(sorted((abs(c[0]), abs(c[1]))))
        agree = (dist is None) == (exact is None) and (
            dist is None or float(exact[0]) - 1e-9 * (1 + dist) <= dist <= float(exact[1]) + 1e-9 * (1 + dist))
        bench.records.append(dict(case="numpy.roots against the exact change", backend="numpy", phase="measure",
                                  params=dict(kind=label), numpy_distance=dist,
                                  exact_distance=None if exact is None else [str(exact[0]), str(exact[1])],
                                  agree=bool(agree), tolerance=1e-9))


def resp_function(bench, rng):
    for rows in ((1000,) if bench.quick else (1000, 100_000)):
        xs = rng.integers(-9, 10, rows)
        ys = 3 * xs + rng.integers(-20, 21, rows)
        frame = pd.DataFrame({"x": xs, "y": ys})
        db = torbles.from_frames({"r": frame})
        tab = db["r"]
        p = dict(rows=rows)

        def build():
            basis = MomentBasis(db.query(tab), ["x", "y"])
            x, y = basis.variables["x"], basis.variables["y"]
            theta = basis.parameter("theta")
            return basis, basis.reduce((y - theta * x) ** 2)
        basis, loss = bench.measure("RESP-001 response function build (moments + reduce)", "cpu", build, repeats=2,
                                    phase="build", **p)
        sxx = int((xs.astype(object) ** 2).sum())
        sxy = int((xs.astype(object) * ys.astype(object)).sum())
        syy = int((ys.astype(object) ** 2).sum())
        level = F(syy) - F(sxy) ** 2 / sxx + rows
        r = bench.measure("sublevel target of the RESP-001 loss", "cpu",
                          lambda: iv.univariate(loss, {"theta": 2 * F(sxy, sxx)}, "<=", level, cost="l2",
                                                width=F(1, 2 ** 40)), repeats=3, phase="run", **p)
        bench.records.append(dict(case="RESP-001 sublevel result", backend="cpu", phase="measure", params=p, **summary(r)))
        basis.close()


def aggregates(bench, rng):
    con = duckdb.connect()
    for rows in ((1000, 100_000) if bench.quick else (1000, 100_000, 1_000_000)):
        x = rng.integers(-500, 501, rows).astype(np.int64)
        v = rng.integers(-20, 41, rows).astype(np.int64)
        cols = dict(x=x, v=v)
        p = dict(rows=rows)
        scope = mbk.Scope.from_arrays(cols)
        st = bench.measure("aggregate state build", "cpu", lambda: iv.Aggregate(scope, "x", "v"), repeats=2,
                           phase="build", **p)
        requests = [("count", (">=", 100), (">=", rows // 2)), ("sum", (">=", 0), (">=", 2 * int(v.sum()) // 3)),
                    ("avg", ("<", 0), (">=", 12)), ("count_value", ("<=", -100), ("<", rows // 10)),
                    ("sum", (">", 250), ("<=", -1))]

        def from_state():
            return [st.minimal(agg, pred, tgt) for agg, pred, tgt in requests]
        res = bench.measure("requests from the shared aggregate state", "cpu", from_state, repeats=3, phase="warm",
                            requests=len(requests), **p)

        def rebuild():
            return [iv.Aggregate(mbk.Scope.from_arrays(cols), "x", "v").minimal(agg, pred, tgt)
                    for agg, pred, tgt in requests]
        res2 = bench.measure("full recomputation: state rebuilt per request", "reconstruction", rebuild,
                             repeats=1 if rows == 1_000_000 else 2, phase="run", requests=len(requests), **p)
        for a, b in zip(res, res2):
            if not a.same(b):
                raise SystemExit("INV-002: aggregate answer differs from the recomputation")
        con.register("t", pd.DataFrame(cols))
        sql = {"count": "COUNT(*)", "sum": "SUM(v)", "avg": "AVG(v)", "count_value": "COUNT(v)"}

        def duck_at_answers():
            out = []
            for (agg, (pop, c), _), r in zip(requests, res):
                if not r.solutions:
                    out.append(None)
                    continue
                d = r.solutions[0].change["d"]
                d = d if not isinstance(d, tuple) else d[0]
                q = f"SELECT {sql[agg]} FROM t WHERE x + {float(d)!r} {pop} {c}"
                out.append(con.execute(q).fetchall()[0][0])
            return out
        vals = bench.measure("DuckDB: the aggregate at each answer (verification)", "duckdb", duck_at_answers,
                             repeats=2, phase="run", **p)
        frozen = []
        for (agg, (pop, c), (top, tv)), r, dv in zip(requests, res, vals):
            base = con.execute(f"SELECT {sql[agg]} FROM t WHERE x {pop} {c}").fetchall()[0][0]
            frozen.append(dict(aggregate=agg, status=r.status, change=fq(r.solutions[0].change["d"]) if r.solutions else None,
                               value=fq(r.solutions[0].value) if r.solutions else None, duckdb_value=dv,
                               frozen_row_set_value=base, entered=r.solutions[0].entered if r.solutions else None,
                               left=r.solutions[0].left if r.solutions else None))
        bench.records.append(dict(case="aggregate answers (rows entering and leaving, the frozen row set)",
                                  backend="cpu", phase="measure", params=p, answers=frozen,
                                  note="frozen_row_set_value: the aggregate over the rows selected at d = 0, which a "
                                       "frozen-row-set method would keep for every d"))
        con.unregister("t")
        del scope, st, cols


# ---------------------------------------------------------------------------------------------------- case 2
TARGETS = {1: ({(3,): 1, (1,): -3}, "=", 2),                      # t^3 - 3t = 2: t = -1 (double), 2
           2: ({(2, 0): 1, (0, 2): 1, (1, 1): -1}, "=", 7),        # t^2 - t u + u^2 = 7
           3: ({(2, 0, 0): 1, (0, 2, 0): 1, (0, 0, 2): 1, (1, 0, 1): -1}, "<=", 1)}


def peval(terms, x):
    total = F(0)
    for e, c in terms.items():
        t = F(c)
        for xi, k in zip(x, e):
            t *= xi ** k
        total += t
    return total


def grids(bench, rng):
    sizes = (8, 16) if bench.quick else (8, 16, 32)
    for nd in (1, 2, 3):
        terms, op, tv = TARGETS[nd]
        for size in sizes:
            vals = [F(k - size // 2, 2) for k in range(size)]      # -size/4 .. size/4 - 1/2 in steps of 1/2
            grid = {f"z{d}": vals for d in range(nd)}
            x0 = {f"z{d}": F(1, 3) for d in range(nd)}
            p = dict(dimension=nd, values=size, points=size ** nd, op=op)
            g = bench.measure("grid plan build", "cpu", lambda: iv.Grid(terms, x0, grid, op, tv, cost="l1"), repeats=2,
                              phase="build", **p)
            for backend in bench.backends():
                r = bench.measure("grid search", backend, lambda: g.search(backend=backend), repeats=3, phase="run",
                                  **p)
            bench.records.append(dict(case="grid result and proof", backend="cpu", phase="measure", params=p,
                                      **summary(r), pruned=g.info["pruned"], blocks_pruned=g.info["blocks_pruned"],
                                      bound_bits=g.info["bound_bits"], primes=g.info["primes"]))
            pts = [[vals[i] for i in dg] for dg in itertools.product(range(size), repeat=nd)]
            xv = [F(1, 3)] * nd
            cmp = {"=": lambda a: a == 0, "<=": lambda a: a <= 0}[op]

            def brute():
                best, ties = None, []
                for idx, pt in enumerate(pts):
                    if not cmp(peval(terms, pt) - tv):
                        continue
                    c = sum(abs(a - b) for a, b in zip(pt, xv))
                    if best is None or c < best:
                        best, ties = c, [idx]
                    elif c == best:
                        ties.append(idx)
                return best, ties
            best, ties = bench.measure("Python Fraction brute force (exact reference)", "python", brute,
                                       repeats=1, phase="run", **p)
            if (best is None) != (r.cost is None) or (best is not None and (best != r.cost or
                                                                              sorted(ties) != sorted(s.index for s in r.solutions))):
                raise SystemExit(f"INV-002: grid search differs from the brute force at {p}")
            P = np.array([[float(v) for v in pt] for pt in pts])

            def numpy_brute():
                val = np.zeros(len(P))
                for e, c in terms.items():
                    val += float(c) * np.prod(P ** np.array(e, float), axis=1)
                ok = np.abs(val - float(tv)) <= 1e-9 if op == "=" else val - float(tv) <= 1e-9
                cost = np.abs(P - 1 / 3).sum(axis=1)
                if not ok.any():
                    return None, []
                m = cost[ok].min()
                return float(m), [int(i) for i in np.nonzero(ok & (np.abs(cost - m) <= 1e-9))[0]]
            nb, nt = bench.measure("NumPy float64 brute force (tolerance 1e-9, uncertified)", "numpy", numpy_brute,
                                   repeats=3, phase="run", **p)
            bench.records.append(dict(case="numpy brute force against the exact search", backend="numpy",
                                      phase="measure", params=p, same_minimum=(nb is None) == (best is None) and (
                                          best is None or abs(nb - float(best)) <= 1e-9),
                                      tie_mismatches=len(set(nt) ^ set(ties))))
            # incomplete: the iteration budget stops the scan with proven bounds; the output budget truncates ties
            for lim in sorted({1, max(1, g.info["total"] // 8)}):
                ri = bench.measure("grid search, iteration budget", "cpu", lambda: g.search(budget=Budget(iterations=lim)),
                                   repeats=2, phase="run", iterations=lim, **p)
                rec = dict(case="search under an iteration budget", backend="cpu", phase="measure", params=dict(p, iterations=lim),
                           **summary(ri), reason=ri.reason, examined=ri.examined, next=ri.next)
                if ri.status == "incomplete":
                    rest = g.search(start=ri.next)
                    rec["resumed"] = summary(rest)
                    if best is not None and ri.lower > best:
                        raise SystemExit("INV-002: an incomplete lower bound exceeds the minimum")
                bench.records.append(rec)
            if len(ties) > 1:
                ro = g.search(budget=Budget(output=1))
                bench.records.append(dict(case="output budget: ties truncated", backend="cpu", phase="measure",
                                          params=p, **summary(ro), reason=ro.reason, next=ro.next, ties=len(ties)))
    # the plan's state budget: measured at the limit and one past it (never extrapolated)
    terms, op, tv = TARGETS[3]
    vals = [F(k - 8, 2) for k in range(16)]
    grid = {f"z{d}": vals for d in range(3)}
    x0 = {f"z{d}": 0 for d in range(3)}
    iv.Grid(terms, x0, grid, op, tv, budget=Budget(states=16 ** 3))
    rec = bench.refusal("grid plan states budget", "cpu", lambda: iv.Grid(terms, x0, grid, op, tv,
                                                                           budget=Budget(states=16 ** 3 - 1)),
                        points=16 ** 3)
    bench.threshold("grid points (3 coordinates x 16 values)", admitted=dict(states=16 ** 3),
                    refused=dict(states=16 ** 3 - 1) if rec.get("refused") else None, need=rec.get("need"))


def main():
    bench = Bench.from_args("INV-002", __doc__)
    if bench.capability != "INV-002":
        raise SystemExit("--capability INV-002")
    rng = np.random.default_rng(bench.seed)
    t0 = time.perf_counter()
    linear_targets(bench, rng)
    univariate_targets(bench)
    resp_function(bench, rng)
    aggregates(bench, rng)
    grids(bench, rng)
    bench.note("every answer is an algebraic intervention on the function as given (claim 'algebraic', causal False)")
    if not bench.device:
        bench.note("CUDA grid search not timed (no --device): the emulation is never timed as CUDA")
    bench.note(f"total {time.perf_counter() - t0:.1f} s")
    bench.write()


if __name__ == "__main__":
    main()
