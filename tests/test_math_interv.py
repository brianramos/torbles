# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""INV-002: constrained minimal interventions (torbles/math_interv.c, math_interv_gpu.cu, math_interv.py).

Oracles, independent of the implementation: brute force over small discrete domains with Python Fractions written
here (every point, its exact cost, the minimum and every tie), the candidate points of 2-D polygons (vertices, edge
crossings of the cost's breakpoint lines and the projections of the base point: the exact minima of L1 / Linf / L2
costs over a polygon), exact rational roots and symmetric constructions for univariate targets (numpy roots only as
float checks of isolating intervals, with a stated tolerance), and DuckDB SQL evaluating the query at shifted data
for aggregates whose rows cross their predicate. Every grid search runs on the CPU and on the CUDA path (emulated here
with TORBLES_EMU_SCRAMBLE=1) and the results are compared. Seeds are fixed.
"""
import itertools
import unittest
from fractions import Fraction as F

import duckdb
import numpy as np
import pandas as pd

from torbles import math_bank as mbk
from torbles import math_interv as iv
from torbles import math_opt as opt
from torbles.math_budgets import Budget
from torbles.math_errors import UnsupportedError
from torbles.mathematics import AlgebraBudgetError, Context, MomentBasis

CMP = {"=": lambda v: v == 0, "!=": lambda v: v != 0, "<": lambda v: v < 0, "<=": lambda v: v <= 0,
       ">": lambda v: v > 0, ">=": lambda v: v >= 0}
COMPLEMENT = {"=": "!=", "!=": "=", "<": ">=", "<=": ">", ">": "<=", ">=": "<"}


# ---------------------------------------------------------------------------------------------------- oracles
def peval(terms, x):
    """an exact polynomial {exponent tuple: coefficient} at the point x"""
    total = F(0)
    for e, c in terms.items():
        t = F(c)
        for xi, k in zip(x, e):
            t *= F(xi) ** k
        total += t
    return total


def coord_cost(kind, w, d):
    d, w = F(d), F(w)
    if kind == "l0":
        return w if d != 0 else F(0)
    return w * (d * d if kind == "l2" else abs(d))


def total_cost(kind, ws, ds):
    parts = [coord_cost(kind, w, d) for w, d in zip(ws, ds)]
    return max(parts) if kind == "linf" else sum(parts)


def brute_grid(terms, x0, grid, op, t, kind, ws, cons=(), A=(), b=(), flip=False):
    """every point of the grid: the minimal cost and every tie in (cost, grid index) order"""
    base = peval(terms, x0) - F(t)
    member = CMP[op](base)
    if flip:
        op = COMPLEMENT[op] if member else op
    best, ties = None, []
    for idx, digits in enumerate(itertools.product(*[range(len(g)) for g in grid])):
        p = [F(grid[d][i]) for d, i in enumerate(digits)]
        if not CMP[op](peval(terms, p) - F(t)):
            continue
        if any(not CMP[cop](peval(cterms, p) - F(rhs)) for cterms, cop, rhs in cons):
            continue
        if any(sum(F(a) * x for a, x in zip(row, p)) > F(bb) for row, bb in zip(A, b)):
            continue
        c = total_cost(kind, ws, [x - F(x0i) for x, x0i in zip(p, x0)])
        if best is None or c < best:
            best, ties = c, [idx]
        elif c == best:
            ties.append(idx)
    return best, ties, member


def poly_terms(rng, nd, deg=3, nterms=4):
    terms = {}
    for _ in range(nterms):
        e = tuple(int(v) for v in rng.integers(0, deg + 1, nd))
        if sum(e) > deg:
            continue
        terms[e] = F(int(rng.integers(-4, 5)), int(rng.integers(1, 3)))
    return {e: c for e, c in terms.items() if c} or {(1,) + (0,) * (nd - 1): F(1)}


def lines_meet(l1, l2):
    """the intersection of two lines a . x = c in the plane (None when parallel)"""
    (a1, b1, c1), (a2, b2, c2) = l1, l2
    det = a1 * b2 - a2 * b1
    if det == 0:
        return None
    return ((c1 * b2 - c2 * b1) / det, (a1 * c2 - a2 * c1) / det)


def polygon_min(kind, ws, halfplanes, equality=None):
    """the exact minimum of an L1 / Linf / L2 cost of d over a 2-D polygon {h . d <= c} (and a line) by candidates:
    the vertices, the crossings of the boundary lines with the cost's breakpoint lines, the projections of 0 onto the
    boundary lines (L2), the origin: the minimum of these convex piecewise costs lies at one of them"""
    w1, w2 = F(ws[0]), F(ws[1])
    lines = [tuple(F(v) for v in h) for h in halfplanes]
    if equality is not None:
        lines.append(tuple(F(v) for v in equality))
    breaks = [(F(1), F(0), F(0)), (F(0), F(1), F(0)), (w1, -w2, F(0)), (w1, w2, F(0))]
    cands = [(F(0), F(0))]
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            p = lines_meet(lines[i], lines[j])
            if p:
                cands.append(p)
        for br in breaks:
            p = lines_meet(lines[i], br)
            if p:
                cands.append(p)
        a, b2, c = lines[i]
        if kind == "l2" and (a or b2):                                 # the weighted projection of 0
            s = c / (a * a / w1 + b2 * b2 / w2)
            cands.append((s * a / w1, s * b2 / w2))
    def ok(p):
        if any(h[0] * p[0] + h[1] * p[1] > h[2] for h in lines[:len(halfplanes)]):
            return False
        return equality is None or equality[0] * p[0] + equality[1] * p[1] == equality[2]
    feas = [p for p in cands if ok(p)]
    if not feas:
        return None, []
    vals = [total_cost(kind, ws, p) for p in feas]
    best = min(vals)
    return best, [p for p, v in zip(feas, vals) if v == best]


def duck_value(con, agg, pop, c, d, shifted):
    """the aggregate of the query at the shifted data, computed by DuckDB exactly (integers scaled by d's and c's
    denominators); None for SQL NULL"""
    d, c = F(d), F(c)
    p, q = d.numerator, d.denominator
    cond = f"(CAST({q * c.denominator} AS HUGEINT) * x + {p * c.denominator}) {pop} {c.numerator * q}"
    if agg == "count":
        return F(con.execute(f"SELECT COUNT(*) FROM t WHERE {cond}").fetchone()[0])
    if agg == "count_value":
        return F(con.execute(f"SELECT COUNT(v) FROM t WHERE {cond}").fetchone()[0])
    col = f"(CAST({q} AS HUGEINT) * x + {p})" if shifted else "CAST(v AS HUGEINT)"
    s, n = con.execute(f"SELECT SUM({col}), COUNT({'x' if shifted else 'v'}) FROM t WHERE {cond}").fetchone()
    if s is None:
        return None
    s = F(int(s), q if shifted else 1)
    return s if agg == "sum" else s / int(n)


class Interv(unittest.TestCase):
    def grid_both(self, *args, **kwargs):
        g = iv.Grid(*args, **{k: v for k, v in kwargs.items() if k not in ("start", "budget2")})
        budget = kwargs.get("budget2")
        cpu = g.search(kwargs.get("start", 0), budget=budget)
        gpu = g.search(kwargs.get("start", 0), backend="cuda", budget=budget)
        self.assertTrue(cpu.same(gpu))
        self.assertIn(gpu.backend, ("cuda", "cuda-emulated"))
        return g, cpu

    # ------------------------------------------------------------------------------------------------ grids
    def test_inv002_brute_force_small_discrete_domains_certify_the_minimum_and_every_tie(self):
        rng = np.random.default_rng(20261101)
        kinds = ("l1", "linf", "l2", "l0")
        ops = ("=", "!=", "<", "<=", ">", ">=")
        outcomes = {"none": 0, "one": 0, "ties": 0}
        for trial in range(60):
            nd = 1 + trial % 3
            terms = poly_terms(rng, nd)
            size = 4 + int(rng.integers(0, 4))
            grid = [sorted({F(int(v), int(rng.integers(1, 3))) for v in rng.integers(-6, 7, size)}) for _ in range(nd)]
            x0 = [F(int(rng.integers(-3, 4)), int(rng.integers(1, 3))) for _ in range(nd)]
            kind = kinds[trial % 4]
            ws = [F(int(rng.integers(1, 4)), int(rng.integers(1, 3))) for _ in range(nd)]
            op = ops[trial % 6]
            pts = [tuple(F(grid[d][i]) for d, i in enumerate(dg)) for dg in itertools.product(*[range(len(g)) for g in grid])]
            t = peval(terms, pts[int(rng.integers(0, len(pts)))]) if trial % 5 else F(int(rng.integers(-5, 6)))
            cons, A, b = [], [], []
            if trial % 4 == 1:
                cons = [({(1,) + (0,) * (nd - 1): F(1)}, "<=", F(2))]       # x_0 <= 2
            if trial % 6 == 2:
                A, b = [[1] * nd], [F(3)]
            flip = trial % 7 == 3
            best, ties, member = brute_grid(terms, x0, grid, op, t, kind, ws, cons, A, b, flip)
            g, r = self.grid_both(terms, x0, {f"c{d}": grid[d] for d in range(nd)}, op, t, cost=kind, weights=ws,
                                  constraints=[(ct, o, rh) for ct, o, rh in cons], A_ub=A or None, b_ub=b or None,
                                  flip=flip, names=None)
            self.assertEqual(r.base_member, member)
            if best is None:
                self.assertEqual(r.status, "infeasible", trial)
                self.assertEqual(r.solutions, [])
                outcomes["none"] += 1
                continue
            self.assertEqual(r.status, "optimal", trial)
            self.assertEqual(r.cost, best)
            self.assertEqual(sorted(s.index for s in r.solutions), sorted(ties))       # every tie, nothing else
            self.assertEqual(r.unique, len(ties) == 1)
            for s in r.solutions:
                self.assertEqual(s.cost, best)
                p = [s.point[f"c{d}"] for d in range(nd)]
                self.assertEqual(s.value, peval(terms, p))
                self.assertEqual([s.change[f"c{d}"] for d in range(nd)], [pi - x0i for pi, x0i in zip(p, x0)])
            outcomes["one" if len(ties) == 1 else "ties"] += 1
            self.assertEqual(r.claim, "algebraic")
        self.assertTrue(all(v >= 3 for v in outcomes.values()), outcomes)

    def test_inv002_incomplete_searches_report_proven_bounds_and_resume(self):
        rng = np.random.default_rng(31)
        ctx = Context()
        a, b2, c = ctx.symbols("a", "b", "c")
        f = a * a + b2 * c - 3 * a
        fterms = {(2, 0, 0): 1, (0, 1, 1): 1, (1, 0, 0): -3}
        grid = [list(range(-5, 6))] * 3
        x0 = [0, 0, 0]
        for target in (7, 23, 1000):
            best, ties, _ = brute_grid(fterms, x0, grid, "=", target, "l1", [1, 1, 1])
            gfull = iv.Grid(f, {"a": 0, "b": 0, "c": 0}, {"a": grid[0], "b": grid[1], "c": grid[2]}, "=", target)
            full = gfull.search()
            if best is None:
                self.assertEqual(full.status, "infeasible")
            else:
                self.assertEqual((full.status, full.cost), ("optimal", best))
            for budget in (1, 7, 40):
                g, r = self.grid_both(f, {"a": 0, "b": 0, "c": 0}, {"a": grid[0], "b": grid[1], "c": grid[2]}, "=",
                                      target, budget2=Budget(iterations=budget))
                if r.status == "incomplete":
                    self.assertEqual(r.reason, "iterations")
                    self.assertLessEqual(r.examined, budget)
                    if best is not None:
                        self.assertLessEqual(r.lower, best)          # proven: nothing cheaper than lower is feasible
                    if r.upper is not None:
                        self.assertGreaterEqual(r.upper, best)
                    # resuming from next finishes the search with the same minimum and ties
                    rest = g.search(start=r.next)
                    if best is not None and r.upper is None:
                        self.assertEqual(rest.cost, best)
                else:
                    self.assertEqual(r.status, full.status)
        # the output budget truncates the ties explicitly: the listed prefix is complete, next resumes at the rest
        terms = {(1, 0): 1, (0, 1): 1}
        grid = [list(range(-3, 4))] * 2
        best, ties, _ = brute_grid(terms, [0, 0], grid, "=", 0, "l0", [1, 1])
        g = iv.Grid(terms, [0, 0], {"p": grid[0], "q": grid[1]}, "=", 0, cost="l0")
        r = g.search(budget=Budget(output=1))
        self.assertEqual((r.status, r.cost, len(r.solutions)), ("optimal", 0, 1))   # (0, 0): one tie at cost 0
        self.assertEqual(r.solutions[0].index, ties[0])
        r = g.search(budget=Budget(output=0))                          # the minimum is proven, its tie is not listed
        self.assertEqual((r.status, r.reason, r.lower, r.upper, len(r.solutions)), ("incomplete", "output", 0, 0, 0))
        terms = {(1, 0): 1, (0, 1): 1}
        best, ties, _ = brute_grid(terms, [1, 1], grid, "=", 0, "l1", [1, 1])
        g = iv.Grid(terms, [1, 1], {"p": grid[0], "q": grid[1]}, "=", 0, cost="l1")
        r = g.search(budget=Budget(output=1))
        self.assertEqual((r.status, r.reason, len(r.solutions)), ("incomplete", "output", 1))
        self.assertEqual(r.lower, best)
        rest = g.search(start=r.next)
        self.assertEqual(sorted([s.index for s in r.solutions] + [s.index for s in rest.solutions]), sorted(ties))

    def test_inv002_enclosure_pruning_is_sound(self):
        rng = np.random.default_rng(5)
        for trial in range(20):
            nd = 2 + trial % 2
            terms = poly_terms(rng, nd, deg=2)
            grid = [[F(v) for v in range(-12, 13)]] * nd
            g = iv.Grid(terms, [0] * nd, {f"z{d}": grid[d] for d in range(nd)}, ">=", 30, cost="l2")
            info = g.info
            # every pruned point is infeasible: the surviving order holds every feasible point
            feasible = [idx for idx, dg in enumerate(itertools.product(*[range(len(x)) for x in grid]))
                        if peval(terms, [grid[d][i] for d, i in enumerate(dg)]) >= 30]
            order = set(g.order().tolist())
            self.assertTrue(set(feasible) <= order, trial)
            self.assertEqual(info["grid"], info["total"] + info["pruned"])

    def test_inv002_grid_kernel_equals_the_cpu_bit_for_bit(self):
        rng = np.random.default_rng(77)
        for trial in range(12):
            nd = 1 + trial % 3
            terms = poly_terms(rng, nd)
            grid = {f"g{d}": [F(int(v), 3) for v in range(-8, 9)] for d in range(nd)}
            g = iv.Grid(terms, [0] * nd, grid, "<=", F(1, 2), cost=("l1", "l2", "linf")[trial % 3])
            total = g.info["total"]
            for start, count in ((0, total), (total // 3, total - total // 3), (0, min(5, total))):
                for match in (None, 0):
                    a = g.evaluate(start, count, match)
                    bb = g.evaluate(start, count, match, backend="cuda")
                    self.assertTrue(np.array_equal(a[0], bb[0]) and np.array_equal(a[1], bb[1]))
                    self.assertEqual(a[2], bb[2])

    # ------------------------------------------------------------------------------------------------ linear
    def test_inv002_linear_targets_are_exact_lps_and_projections(self):
        rng = np.random.default_rng(13)
        checked = 0
        for trial in range(48):
            kind = ("l1", "linf", "l2")[trial % 3]
            ws = [F(int(rng.integers(1, 4)), int(rng.integers(1, 3))) for _ in range(2)]
            a = [int(v) for v in rng.integers(-3, 4, 2)]
            if a == [0, 0]:
                a = [1, 2]
            x0 = [F(int(v)) for v in rng.integers(-3, 4, 2)]
            t = F(int(rng.integers(-8, 9)))
            op = ("=", "<=", ">=")[trial % 3 if trial % 2 else 2]
            # constraints on d: a box and one random row (on the point x: A x <= b)
            row = [int(v) for v in rng.integers(-2, 3, 2)]
            bx = F(int(rng.integers(-2, 6)))
            dbox = [(-4, 4), (-4, 4)]
            r = iv.linear((a, 0), x0, op, t, cost=kind, weights=ws, A_ub=[row], b_ub=[bx], d_bounds=dbox)
            # the polygon in d: the box, the row (row . (x0 + d) <= bx), the target a . (x0 + d) op t
            hp = [(1, 0, 4), (-1, 0, 4), (0, 1, 4), (0, -1, 4),
                  (row[0], row[1], bx - row[0] * x0[0] - row[1] * x0[1])]
            rr = t - a[0] * x0[0] - a[1] * x0[1]
            eq = None
            if op == "=":
                eq = (a[0], a[1], rr)
            elif op == "<=":
                hp.append((a[0], a[1], rr))
            else:
                hp.append((-a[0], -a[1], -rr))
            best, opts = polygon_min(kind, ws, hp, eq)
            if best is None:
                self.assertEqual(r.status, "infeasible", trial)
                continue
            self.assertEqual(r.status, "optimal", trial)
            self.assertEqual(r.method, "qp" if kind == "l2" else "lp")
            self.assertEqual(r.cost, best, trial)
            d = [r.solutions[0].change["x0"], r.solutions[0].change["x1"]]
            self.assertIn(tuple(d), [tuple(p) for p in opts] + [tuple(d)] if r.unique is False else [tuple(p) for p in opts])
            self.assertEqual(total_cost(kind, ws, d), best)
            if r.unique is False:
                fd = [r.face["x0"], r.face["x1"]]
                self.assertNotEqual(fd, d)
                self.assertEqual(total_cost(kind, ws, fd), best)
            else:
                self.assertTrue(len({tuple(p) for p in opts}) == 1)   # one optimal candidate: a unique optimum
            checked += 1
        self.assertGreater(checked, 25)
        # the squared-L2 projection against OPT-001's box minimizer: eliminating the free coordinate d2 of the target
        # hyperplane leaves a positive definite quadratic of (d0, d1) over their box
        for trial in range(12):
            a = [int(v) for v in rng.integers(-3, 4, 2)] + [int(rng.integers(1, 4)) * (1 if trial % 2 else -1)]
            ws = [F(int(rng.integers(1, 4)), int(rng.integers(1, 3))) for _ in range(3)]
            x0 = [F(int(v)) for v in rng.integers(-3, 4, 3)]
            t = F(int(rng.integers(-9, 10)))
            box = [(F(int(rng.integers(-3, 1))), F(int(rng.integers(0, 4)))) for _ in range(2)]
            r = iv.linear((a, 0), x0, "=", t, cost="l2", weights=ws, d_bounds=box + [None])
            rr = t - sum(ai * xi for ai, xi in zip(a, x0))
            al, be = rr / a[2], [F(-a[0], a[2]), F(-a[1], a[2])]
            A = [[2 * ((ws[i] if i == j else 0) + ws[2] * be[i] * be[j]) for j in range(2)] for i in range(2)]
            q = opt.Quadratic.from_matrices(A, [-2 * ws[2] * al * be[i] for i in range(2)], ws[2] * al * al)
            o = opt.minimize(q, bounds=box)
            self.assertEqual((r.status, r.method, r.unique), ("optimal", "qp", True), trial)
            self.assertEqual(r.cost, o.value, trial)
            ch = r.solutions[0].change
            self.assertEqual([ch["x0"], ch["x1"]], list(o.point), trial)
            self.assertEqual(ch["x2"], al + be[0] * o.point[0] + be[1] * o.point[1], trial)

    def test_inv002_strict_targets_infima_and_supports_of_the_l0_cost(self):
        # strict targets: closure optimum, attainment decided exactly
        r = iv.linear(([1, 1], 0), [0, 0], ">", 2, cost="l2")
        self.assertEqual((r.status, r.cost, r.solutions[0].attained), ("unattained", 2, False))
        self.assertEqual(r.solutions[0].change, {"x0": 1, "x1": 1})
        r = iv.linear(([1, 1], 0), [0, 0], ">", 2, cost="l1", d_bounds=[(0, 1), (0, 1)])
        self.assertEqual(r.status, "infeasible")                       # max d1 + d2 = 2 on the box: not > 2
        r = iv.linear(([1, 0], 0), [0, 0], ">", 2, cost="linf", d_bounds=[(None, None), (0, 0)])
        self.assertEqual((r.status, r.cost), ("unattained", 2))
        r = iv.linear(([1, 0], 0), [5, 0], "!=", 5, cost="l1")
        self.assertEqual((r.status, r.cost, len(r.solutions)), ("unattained", 0, 1))
        r = iv.linear(([1, 0], 0), [5, 0], "!=", 4, cost="l1")           # already satisfied: no change
        self.assertEqual((r.status, r.cost, r.unique), ("optimal", 0, True))
        # L1 over a segment of optima: an attained strict optimum exists off the boundary
        r = iv.linear(([1, 1], 0), [0, 0], ">", 2, cost="l1", A_ub=[[1, 0]], b_ub=[5])
        self.assertEqual((r.status, r.cost), ("unattained", 2))
        r = iv.linear(([1, 1], 0), [0, 0], ">=", 2, cost="l1")
        self.assertEqual((r.status, r.cost, r.unique), ("optimal", 2, False))
        self.assertEqual(sum(r.face.values()), 2)
        # L0: every minimal support with a witness; weights decide
        r = iv.linear(([2, 1, 0], 0), [1, 0, 0], ">=", 7, cost="l0")
        self.assertEqual((r.status, r.cost), ("optimal", 1))
        self.assertEqual(sorted(s.support for s in r.solutions), [("x0",), ("x1",)])
        for s in r.solutions:
            dv = [s.change[k] for k in ("x0", "x1", "x2")]
            self.assertGreaterEqual(2 * (1 + dv[0]) + dv[1], 7)
            self.assertEqual(sum(1 for v in dv if v) , 1)
        r = iv.linear(([2, 1, 0], 0), [1, 0, 0], ">=", 7, cost="l0", weights=[3, 2, 1])
        self.assertEqual([s.support for s in r.solutions], [("x1",)])
        r = iv.linear(([1, 1], 0), [0, 0], "=", 3, cost="l0", d_bounds=[(0, 1), (0, 1)])
        self.assertEqual(r.status, "infeasible")
        r = iv.linear(([1, 1], 0), [0, 0], "=", 2, cost="l0", d_bounds=[(0, 1), (0, 1)])
        self.assertEqual((r.status, r.cost, r.solutions[0].support), ("optimal", 2, ("x0", "x1")))
        # a budget of supports: incomplete with the proven lower bound
        r = iv.linear(([1, 1, 1, 1], 0), [0, 0, 0, 0], "=", 4, cost="l0", d_bounds=[(0, 1)] * 4,
                      budget=Budget(iterations=6))
        self.assertEqual((r.status, r.reason), ("incomplete", "iterations"))
        self.assertLessEqual(r.lower, 4)

    def test_inv002_linear_targets_of_first_slice_and_resp001_functions(self):
        ctx = Context()
        t, u = ctx.symbols("t", "u")
        r = iv.linear(2 * t + u, {"t": 1, "u": 0}, ">=", 7, cost="l1", d_bounds={"u": (0, None)})
        self.assertEqual((r.cost, r.solutions[0].change), (F(5, 2), {"t": F(5, 2), "u": 0}))
        with self.assertRaises(UnsupportedError):
            iv.linear(t * u, {"t": 1, "u": 0}, ">=", 7)
        # a RESP-001 response: the squared loss of y ~ theta x over the selected rows, reduced against the moments
        import torbles
        rng = np.random.default_rng(17)
        xs = [int(v) for v in rng.integers(-9, 10, 40)]
        ys = [int(v) for v in rng.integers(-20, 21, 40)]
        keep = [int(v) for v in rng.integers(0, 2, 40)]
        db = torbles.from_frames({"r": pd.DataFrame({"x": xs, "y": ys, "keep": keep})})
        tab = db["r"]
        con = duckdb.connect()
        con.register("r", pd.DataFrame({"x": xs, "y": ys, "keep": keep}))
        sxx, sxy, syy = (int(v) for v in con.execute(
            "SELECT SUM(x*x)::HUGEINT, SUM(x*y)::HUGEINT, SUM(y*y)::HUGEINT FROM r WHERE keep = 1").fetchone())
        with MomentBasis(db.query(tab).where(tab.keep == 1), ["x", "y"]) as basis:
            x, y = basis.variables["x"], basis.variables["y"]
            theta = basis.parameter("theta")
            loss = basis.reduce((y - theta * x) ** 2)
            star = F(sxy, sxx)
            best = F(syy) - F(sxy) ** 2 / sxx
            level = best + 5
            # the cheapest change of theta from 2 star (outside the sublevel set) to a loss <= level: a root of
            # sxx theta^2 - 2 sxy theta + syy - level, the one nearer to 2 star; exact signs at its interval's ends
            r = iv.univariate(loss, {"theta": 2 * star}, "<=", level, cost="l2", width=F(1, 2 ** 40))
            self.assertEqual(r.status, "optimal")
            q = lambda th: sxx * th * th - 2 * sxy * th + syy - level
            lo, hi = r.solutions[0].point["theta"] if isinstance(r.solutions[0].point["theta"], tuple) else (
                r.solutions[0].point["theta"],) * 2
            self.assertTrue(lo == hi and q(lo) == 0 or (q(lo) < 0) != (q(hi) < 0))
            disc = (2 * sxy) ** 2 - 4 * sxx * (syy - level)
            far = (2 * sxy + float(disc) ** 0.5) / (2 * sxx) if star > 0 else (2 * sxy - float(disc) ** 0.5) / (2 * sxx)
            self.assertLess(abs(float(lo) - far), 1e-6)                # the root on 2 star's side (float check)
            # the same target in OPT-001's terms: the least-squares optimum lies inside the sublevel set
            qd = opt.Quadratic.from_polynomial(loss, ["theta"])
            self.assertEqual(opt.minimize(qd).point, [star])

    # ------------------------------------------------------------------------------------------------ univariate
    def test_inv002_univariate_targets_exact_roots_isolated_roots_ties_and_infima(self):
        ctx = Context()
        t, = ctx.symbols("t")
        # exact rational roots: (t - 1)(t + 2)(3t - 1), from x0 = 0: the nearest root 1/3
        f = (t - 1) * (t + 2) * (3 * t - 1)
        r = iv.univariate(f, {"t": 0}, "=", 0, cost="l2", weight=3)
        self.assertEqual((r.status, r.cost, r.solutions[0].change["t"], r.unique), ("optimal", F(1, 3), F(1, 3), True))
        self.assertTrue(r.solutions[0].exact)
        # an exact tie: roots -1 and 1 from 0
        r = iv.univariate(t ** 2 - 1, {"t": 0}, "=", 0)
        self.assertEqual(sorted(s.change["t"] for s in r.solutions), [-1, 1])
        self.assertFalse(r.unique)
        # an irrational tie decided exactly (+-sqrt 2), and an irrational nearest root (sqrt 2 - 1 from 1)
        r = iv.univariate(t ** 2, {"t": 0}, "=", 2, width=F(1, 2 ** 30))
        self.assertEqual(len(r.solutions), 2)
        for s in r.solutions:
            lo, hi = s.change["t"]
            self.assertLess(hi - lo, F(1, 2 ** 29))
            self.assertTrue(lo * lo < 2 < hi * hi if lo > 0 else hi * hi < 2 < lo * lo)
            self.assertFalse(s.exact)
            self.assertEqual(s.value, 2)
        r = iv.univariate(t ** 2, {"t": 1}, "=", 2, width=F(1, 2 ** 20))
        self.assertEqual(len(r.solutions), 1)
        lo, hi = r.solutions[0].change["t"]
        self.assertTrue(lo < F(2 ** 0.5) - 1 < hi or abs(float(lo) - (2 ** 0.5 - 1)) < 1e-6)
        # a near tie that is not one: sqrt 2 vs 1.4142 on the other side (distinct, decided by refinement)
        g = (t - F(14142, 10000)) * (t * t - 2)
        r = iv.univariate(g, {"t": 0}, "=", 0)
        self.assertEqual(len(r.solutions), 1)
        self.assertEqual(r.solutions[0].change["t"], F(14142, 10000))
        # strict targets: infima, not attained
        r = iv.univariate(t ** 2 - 4, {"t": 0}, ">", 0)
        self.assertEqual((r.status, sorted(s.change["t"] for s in r.solutions)), ("unattained", [-2, 2]))
        self.assertFalse(any(s.attained for s in r.solutions))
        r = iv.univariate(t ** 2 - 4, {"t": 0}, ">=", 0, d_bounds=(0, None))
        self.assertEqual((r.status, [s.change["t"] for s in r.solutions]), ("optimal", [2]))
        # a boundary root of the domain, a domain end as the answer, an empty domain, no root
        r = iv.univariate(t - 3, {"t": 0}, ">=", 0, x_bounds=(-5, 3))
        self.assertEqual((r.status, r.solutions[0].change["t"]), ("optimal", 3))
        r = iv.univariate(t * 0 + 5, {"t": 0}, ">=", 1, d_bounds=(2, 7))
        self.assertEqual((r.status, r.solutions[0].change["t"]), ("optimal", 2))
        r = iv.univariate(t ** 2 + 1, {"t": 0}, "=", 0)
        self.assertEqual(r.status, "infeasible")
        r = iv.univariate(t, {"t": 0}, "=", 1, d_bounds=(2, 1))
        self.assertEqual(r.status, "infeasible")
        # a membership change: t^3 - t > 0 at 2: leave it with the least change (t = 1, exact)
        r = iv.univariate(t ** 3 - t, {"t": 2}, ">", 0, flip=True)
        self.assertEqual((r.base_member, r.op, r.solutions[0].change["t"]), (True, "<=", -1))

    # ------------------------------------------------------------------------------------------------ aggregates
    def test_inv002_predicate_membership_changes_are_kept_in_the_objective(self):
        rng = np.random.default_rng(2026)
        n = 300
        x = rng.integers(-50, 51, n).astype(np.int64)
        v = rng.integers(-20, 41, n).astype(np.int64)
        valid_v = rng.random(n) > 0.1
        valid_x = rng.random(n) > 0.05
        df = pd.DataFrame({"x": pd.array(np.where(valid_x, x, 0), dtype="Int64"),
                           "v": pd.array(np.where(valid_v, v, 0), dtype="Int64")})
        df.loc[~valid_x, "x"] = pd.NA
        df.loc[~valid_v, "v"] = pd.NA
        con = duckdb.connect()
        con.register("t", df)
        scope = mbk.Scope.from_arrays({"x": x, "v": v}, valid={"x": valid_x.astype(np.uint8), "v": valid_v.astype(np.uint8)})
        st = iv.Aggregate(scope, "x", "v")
        sx = iv.Aggregate(scope, "x", "x")
        xs = sorted(set(int(a) for a, ok in zip(x, valid_x) if ok))
        cases = [("count", ">=", 10, ">=", 120), ("count", "<", 0, "=", 50), ("sum", ">", 5, ">=", 3000),
                 ("avg", ">=", 0, "<=", 9), ("count_value", "<=", -10, ">", 140), ("sum", "!=", 7, "<", 2000),
                 ("avg", "<", 20, ">", 12)]
        frozen_differs = 0
        for agg, pop, c, top, target in cases:
            for state, shifted in ((st, False), (sx, True)):
                if shifted and agg == "count_value":
                    continue
                r = state.minimal(agg, (pop, c), (top, target))
                base = duck_value(con, agg, pop, c, 0, shifted)
                self.assertEqual(r.base_value, base)
                self.assertEqual(r.claim, "algebraic")
                # candidate shifts: every breakpoint c - x_k and the midpoints between neighbours (and beyond)
                bps = sorted({F(c) - xk for xk in xs})
                cands = set(bps) | {F(0)}
                for p, q in zip(bps, bps[1:]):
                    cands.add((p + q) / 2)
                cands |= {bps[0] - 1, bps[-1] + 1}
                holds = lambda d: (lambda val: val is not None and CMP[top](val - target))(
                    duck_value(con, agg, pop, c, d, shifted))
                if r.status == "infeasible":
                    self.assertFalse(any(holds(d) for d in cands))
                    continue
                self.assertIn(r.status, ("optimal", "unattained"))
                dstar = r.solutions[0].change["d"]
                if r.status == "optimal":
                    for s in r.solutions:
                        self.assertTrue(holds(s.change["d"]))
                        self.assertEqual(s.value, duck_value(con, agg, pop, c, s.change["d"], shifted))
                else:                                               # the infimum: not attained at it, attained beyond
                    self.assertFalse(holds(dstar))
                    step = F(1, 10 ** 6) * (1 if r.solutions[0].side >= 0 else -1)
                    self.assertTrue(holds(dstar + step))
                for d in cands:                                     # nothing smaller attains it
                    if abs(d) < abs(dstar):
                        self.assertFalse(holds(d), (agg, pop, c, top, target, d, dstar))
                # the rows that entered and left the predicate, by DuckDB
                d = dstar
                cnt = lambda dd: duck_value(con, "count", pop, c, dd, False)
                self.assertEqual(r.solutions[0].rows, cnt(d))
                # on the frozen row set of d = 0 the answer would differ
                if not shifted and agg in ("count", "count_value") and r.status == "optimal" and dstar != 0:
                    frozen_differs += 1                             # frozen counts never change: no finite answer
        self.assertGreater(frozen_differs, 1)
        # SQL NULL: an empty selection has no SUM; it attains no target
        r = st.minimal("sum", (">=", 10 ** 6), ("<=", 0))
        self.assertEqual(r.base_value, None)
        val, rows = st.value("sum", (">=", 10 ** 6), 0)
        self.assertEqual((val, rows), (None, 0))

    def test_inv002_results_are_algebraic_interventions_never_causal_claims(self):
        ctx = Context()
        t, u = ctx.symbols("t", "u")
        scope = mbk.Scope.from_arrays({"x": np.arange(10, dtype=np.int64)})
        st = iv.Aggregate(scope, "x")
        results = [iv.linear((t + u), {"t": 0, "u": 0}, ">=", 1), iv.univariate(t * t, {"t": 0}, ">=", 1),
                   iv.grid(t + u, {"t": 0, "u": 0}, {"t": range(3), "u": range(3)}, "=", 2),
                   st.minimal("count", (">=", 5), (">=", 7))]
        for r in results:
            self.assertEqual(r.claim, "algebraic")
            self.assertIs(r.causal, False)
            self.assertIn("no causal model", r.statement)
        with self.assertRaises(UnsupportedError):
            iv.linear((t + u), {"t": 0, "u": 0}, ">=", 1, semantics="causal")
        with self.assertRaises(UnsupportedError):
            iv.univariate(t * t, {"t": 0}, ">=", 1, semantics="causal")
        with self.assertRaises(UnsupportedError):
            iv.grid(t + u, {"t": 0, "u": 0}, {"t": range(3), "u": range(3)}, "=", 2, semantics="causal")
        with self.assertRaises(UnsupportedError):
            st.minimal("count", (">=", 5), (">=", 7), semantics="causal")
        doc = open(__file__.replace("tests/test_math_interv.py", "docs/math/interv.md")).read()
        self.assertIn("no causal", doc)

    def test_inv002_budgets_at_the_limit_and_one_past(self):
        terms = {(1, 0): 1, (0, 1): 1}
        grid = {"p": range(4), "q": range(4)}
        iv.Grid(terms, [0, 0], grid, "=", 3, budget=Budget(states=16))
        with self.assertRaises(AlgebraBudgetError) as cm:
            iv.Grid(terms, [0, 0], grid, "=", 3, budget=Budget(states=15))
        self.assertEqual((cm.exception.resource, cm.exception.need), ("states", 16))
        g = iv.Grid(terms, [0, 0], grid, "=", 3)
        total = g.info["total"]
        need = g.device_bytes(total)
        g.evaluate(0, total, backend="cuda", budget=Budget(bytes=need))
        with self.assertRaises(AlgebraBudgetError) as cm:
            g.evaluate(0, total, backend="cuda", budget=Budget(bytes=need - 1))
        self.assertEqual(cm.exception.resource, "bytes")
        g.evaluate(0, total, budget=Budget(output=total))
        with self.assertRaises(AlgebraBudgetError):
            g.evaluate(0, total, budget=Budget(output=total - 1))
        with self.assertRaises(AlgebraBudgetError):
            g.evaluate(0, total, backend="cuda", budget=Budget(output=total - 1))
        # linear problems: the tensor of the largest LP tableau, found from the refusals' own needs
        limit = 0
        for _ in range(20):
            try:
                iv.linear(([1, 1], 0), [0, 0], ">=", 1, A_ub=[[1, 0]], b_ub=[3], budget=Budget(tensor=limit))
                break
            except AlgebraBudgetError as e:
                self.assertEqual(e.resource, "tensor")
                self.assertGreater(e.need, limit)
                limit = e.need
        r = iv.linear(([1, 1], 0), [0, 0], ">=", 1, A_ub=[[1, 0]], b_ub=[3], budget=Budget(tensor=limit))
        self.assertEqual((r.status, r.cost), ("optimal", 1))
        with self.assertRaises(AlgebraBudgetError) as cm:
            iv.linear(([1, 1], 0), [0, 0], ">=", 1, A_ub=[[1, 0]], b_ub=[3], budget=Budget(tensor=limit - 1))
        self.assertEqual(cm.exception.need, limit)
        # aggregates: states = rows with the feature present
        scope = mbk.Scope.from_arrays({"x": np.arange(10, dtype=np.int64)})
        iv.Aggregate(scope, "x", budget=Budget(states=10))
        with self.assertRaises(AlgebraBudgetError):
            iv.Aggregate(scope, "x", budget=Budget(states=9))
        # representation limits refused explicitly
        with self.assertRaises(UnsupportedError):
            iv.linear(([1] * 17, 0), [0] * 17, ">=", 1)
        with self.assertRaises(UnsupportedError):
            iv.Grid({(1,) * 9: 1}, [0] * 9, {f"g{i}": [0, 1] for i in range(9)}, "=", 1)
        with self.assertRaises(TypeError):
            iv.linear(([1.5, 1], 0), [0, 0], ">=", 1)

    def test_inv002_abi_registry_covers_the_interv_structs(self):
        from torbles import math_handles as mh
        cpu = {e["name"] for e in mh.abi_entries("cpu")}
        cuda = {e["name"] for e in mh.abi_entries("cuda")}
        self.assertLessEqual({"tr_interv_linear_problem", "tr_interv_univariate_problem", "tr_interv_grid_problem",
                              "tr_interv_grid_view", "tr_interv_grid_info", "tr_interv_info", "tr_interv_solution",
                              "tr_interv_agg_info", "tr_interv_agg_query"}, cpu)
        self.assertIn("tr_interv_grid_view", cuda)
        report = mh.abi_check()
        self.assertTrue(all(r["match"] for r in report["checked"]))


if __name__ == "__main__":
    unittest.main()
