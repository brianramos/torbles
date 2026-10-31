# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""OPT-001: exact minimizers of quadratic objectives (torbles/math_opt.c, math_opt_gpu.cu, math_opt.py).

Oracles, independent of the implementation: Python Fraction arithmetic written here (Gauss-Jordan solves of
A t = b, exact objective values and gradients, KKT signs, and for compact boxes a brute-force minimum over every face
with a nonsingular restricted Hessian and every vertex), least squares rebuilt from the raw rows, and the first
slice's own polynomial evaluation. The face filter is computed on the CPU and on the CUDA path (emulated here with
TORBLES_EMU_SCRAMBLE=1) and compared bit for bit, and every result of the CUDA path is compared with the CPU's.
Seeds are fixed.
"""
import itertools
import unittest
from fractions import Fraction as F

import numpy as np
import pandas as pd

import torbles
from torbles import math_bank as mbk
from torbles import math_lin as lin
from torbles import math_opt as opt
from torbles.math_budgets import Budget
from torbles.math_errors import InfeasibleError, UnsupportedError
from torbles.mathematics import AlgebraBudgetError, Context, MomentBasis


# ---------------------------------------------------------------------- independent exact oracles
def solve(A, b):
    """the unique solution of a square nonsingular system (Fraction Gauss-Jordan), or None"""
    n = len(A)
    M = [[F(v) for v in row] + [F(bb)] for row, bb in zip(A, b)]
    for c in range(n):
        p = next((i for i in range(c, n) if M[i][c] != 0), None)
        if p is None:
            return None
        M[c], M[p] = M[p], M[c]
        inv = 1 / M[c][c]
        M[c] = [v * inv for v in M[c]]
        for i in range(n):
            if i != c and M[i][c]:
                f = M[i][c]
                M[i] = [a - f * bb for a, bb in zip(M[i], M[c])]
    return [M[i][n] for i in range(n)]


def value(A, b, c, t):
    n = len(t)
    return sum(F(t[i]) * F(A[i][j]) * F(t[j]) for i in range(n) for j in range(n)) / 2 - sum(
        F(b[i]) * F(t[i]) for i in range(n)) + F(c)


def gradient(A, b, t):
    return [sum(F(A[i][j]) * F(t[j]) for j in range(len(t))) - F(b[i]) for i in range(len(t))]


def box_oracle(A, b, c, lo, hi):
    """min over a compact box: every face with a nonsingular restricted Hessian (its stationary point, if feasible)
    and every vertex; the minimum lies at such a point"""
    n = len(A)
    best = None
    for states in itertools.product((0, 1, 2), repeat=n):
        free = [i for i in range(n) if states[i] == 2]
        t = [F(lo[i]) if states[i] == 0 else F(hi[i]) if states[i] == 1 else None for i in range(n)]
        if free:
            AF = [[A[i][j] for j in free] for i in free]
            rF = [F(b[i]) - sum(F(A[i][j]) * t[j] for j in range(n) if states[j] != 2) for i in free]
            s = solve(AF, rF)
            if s is None:
                continue
            for i, v in zip(free, s):
                t[i] = v
            if any(not (F(lo[i]) <= t[i] <= F(hi[i])) for i in free):
                continue
        v = value(A, b, c, t)
        if best is None or v < best:
            best = v
    return best


def kkt(A, b, t, lo, hi):
    g = gradient(A, b, t)
    for i in range(len(t)):
        if lo[i] is None:
            if g[i] != 0:
                return False
        elif F(lo[i]) == F(hi[i]):
            continue
        elif t[i] == F(lo[i]):
            if g[i] < 0:
                return False
        elif t[i] == F(hi[i]):
            if g[i] > 0:
                return False
        elif g[i] != 0:
            return False
    return True


def rnd_sym(rng, n, kind):
    """a rational symmetric matrix: "pd", "psd" (rank deficient) or "indefinite" """
    r = n if kind in ("pd", "indefinite") else max(1, n - 1)
    B = [[F(int(rng.integers(-5, 6)), int(rng.integers(1, 3))) for _ in range(n)] for _ in range(r)]
    sg = [1] * r if kind != "indefinite" else [1 if k % 2 else -1 for k in range(r)]
    A = [[sum(sg[k] * B[k][i] * B[k][j] for k in range(r)) for j in range(n)] for i in range(n)]
    if kind == "pd":
        A = [[A[i][j] + (F(1, 2) if i == j else 0) for j in range(n)] for i in range(n)]
    return A


class Opt(unittest.TestCase):
    def both(self, q, bounds=None):
        """the CPU minimum, checked equal to the result with the face filter on the (emulated, scrambled) CUDA path"""
        cpu = opt.minimize(q, bounds)
        if bounds is not None:
            gpu = opt.minimize(q, bounds, backend="cuda")
            self.assertTrue(cpu.same(gpu), bounds)
            self.assertIn(gpu.backend, ("cuda", "cuda-emulated"))
            self.assertTrue(np.array_equal(opt.faces(q, bounds), opt.faces(q, bounds, "cuda")))
        return cpu

    def test_opt001_positive_definite_solutions_match_independent_exact_linear_solves(self):
        rng = np.random.default_rng(3001)
        for n in (1, 2, 3, 5, 8, 12):
            A = rnd_sym(rng, n, "pd")
            b = [F(int(rng.integers(-30, 31)), int(rng.integers(1, 4))) for _ in range(n)]
            c = F(int(rng.integers(-9, 10)), 7)
            q = opt.Quadratic.from_matrices(A, b, c)
            self.assertEqual((q.A, q.b, q.c), (A, b, c))
            r = self.both(q)
            t = solve(A, b)
            self.assertEqual((r.status, r.point, r.value, r.family), ("unique", t, c - sum(bi * ti for bi, ti in zip(b, t)) / 2, []))
            self.assertEqual(r.gradient, [0] * n)
            self.assertTrue(r.kkt and r.convex)
            self.assertEqual(q.value(t), value(A, b, c, t))
            if n <= 8:                                          # a box with the minimizer strictly inside: same point
                bounds = [(ti - 1, ti + F(1, 3)) for ti in t]
                rb = self.both(q, bounds)
                self.assertEqual((rb.status, rb.point, rb.value, rb.active), ("unique", t, r.value, [0] * n))
        # the least-squares loss of a LIN-001 state: the minimizer is the exact regression, the value its RSS
        n = 400
        frame = pd.DataFrame({"g": rng.integers(0, 3, n), "x": rng.integers(-100, 100, n),
                              "d": [f"{v / 100:.2f}" for v in rng.integers(-5000, 5000, n)],
                              "y": rng.integers(-1000, 1000, n)})
        db = torbles.from_ddl("CREATE TABLE f (g BIGINT, x BIGINT, d DECIMAL(10,2), y BIGINT)", {"f": frame})
        scope = mbk.Scope.from_table(db["f"], ["x", "d", "y"], group_by=["g"])
        st = lin.moments(scope, ["x", "d"], "y")
        for g in range(st.groups):
            rows = frame[frame["g"] == g]
            Z = [[F(1), F(int(x)), F(round(float(d) * 100), 100)] for x, d in zip(rows["x"], rows["d"])]
            yv = [F(int(v)) for v in rows["y"]]
            for ridge in (0, F(3, 2)):
                AtA = [[sum(z[i] * z[j] for z in Z) + (ridge if i == j and i else 0) for j in range(3)] for i in range(3)]
                Aty = [sum(z[i] * yy for z, yy in zip(Z, yv)) for i in range(3)]
                beta = solve(AtA, Aty)
                q = opt.Quadratic.from_moments(st, g, ridge=ridge)
                self.assertEqual(q.names, ("1", "x", "d"))
                r = self.both(q)
                loss = sum((yy - sum(bi * zi for bi, zi in zip(beta, z))) ** 2 for z, yy in zip(Z, yv)) + \
                    ridge * (beta[1] ** 2 + beta[2] ** 2)
                self.assertEqual((r.point, r.value), (beta, loss))
                self.assertEqual(r.point, st.fit(g, ridge=ridge).x)
        # a first-slice polynomial response function: loss(theta) = 13 theta^2 - 16 theta + 5
        dbm = torbles.from_frames({"t": pd.DataFrame({"x": [-2, 1, 3], "y": [-1, 4, 2], "keep": [1, 0, 1]})})
        t = dbm["t"]
        with MomentBasis(dbm.query(t).where(t.keep == 1), ["x", "y"]) as basis:
            x, y = basis.variables["x"], basis.variables["y"]
            theta = basis.parameter("theta")
            loss = basis.reduce((y - theta * x) ** 2)
            q = opt.Quadratic.from_polynomial(loss, ["theta"])
            self.assertEqual((q.A, q.b, q.c), ([[26]], [16], 5))
            r = self.both(q)
            self.assertEqual((r.point, r.value), ([F(8, 13)], F(1, 13)))
            self.assertEqual(loss.evaluate(theta=F(8, 13)), F(1, 13))
            self.assertEqual(self.both(q, [(1, 2)]).point, [1])
        ctx = Context()
        a, bb = ctx.symbols("a", "b")
        f = (a - 3) ** 2 + (a - bb) ** 2 + 2 * bb + F(1, 2) * a * bb
        q = opt.Quadratic.from_polynomial(f, ["a", "b"])
        r = self.both(q)
        self.assertEqual(r.point, solve(q.A, q.b))
        self.assertEqual(f.evaluate(a=r.point[0], b=r.point[1]), r.value)

    def test_opt001_unbounded_nonunique_and_constrained_boundary_optima_without_division_by_zero(self):
        rng = np.random.default_rng(3002)
        seen = set()
        for n in (1, 2, 3, 4, 6):
            for kind in ("psd", "indefinite"):
                A = rnd_sym(rng, n, kind) if n > 1 or kind == "psd" else [[F(-3)]]
                if kind == "psd" and n == 1:
                    A = [[F(0)]]
                inv = lin.inertia(A)
                for consistent in (True, False):
                    if inv.zero:
                        z = [F(int(rng.integers(-3, 4))) for _ in range(n)]
                        b = [sum(A[i][j] * z[j] for j in range(n)) for i in range(n)]
                        if not consistent:
                            k = inv.kernel[0]
                            b = [bi + ki for bi, ki in zip(b, k)]        # b leaves range(A): b . k != 0
                    else:
                        b = [F(int(rng.integers(-5, 6))) for _ in range(n)]
                    q = opt.Quadratic.from_matrices(A, b, 1)
                    r = self.both(q)
                    seen.add((r.status, r.certificate))
                    if inv.negative:
                        self.assertEqual((r.status, r.certificate), ("unbounded", "curvature"))
                        d = r.direction
                        self.assertLess(sum(d[i] * A[i][j] * d[j] for i in range(n) for j in range(n)), 0)
                        self.assertEqual(r.rate, sum(d[i] * A[i][j] * d[j] for i in range(n) for j in range(n)))
                    elif inv.zero and not consistent:
                        self.assertEqual((r.status, r.certificate), ("unbounded", "slope"))
                        d = r.direction
                        self.assertEqual(gradient(A, [0] * n, d), [0] * n)          # A d = 0
                        self.assertGreater(sum(bi * di for bi, di in zip(b, d)), 0)
                        self.assertEqual(r.rate, -sum(bi * di for bi, di in zip(b, d)))
                    elif inv.zero:
                        self.assertEqual(r.status, "nonunique")
                        self.assertEqual(len(r.family), inv.zero)
                        self.assertEqual(gradient(A, b, r.point), [0] * n)
                        for v in r.family:
                            self.assertEqual(gradient(A, [0] * n, v), [0] * n)
                            moved = [p + 7 * vi for p, vi in zip(r.point, v)]
                            self.assertEqual(value(A, b, 1, moved), r.value)
                    else:
                        self.assertEqual(r.status, "unique")
                    if r.status == "unbounded":                 # the ray really decreases without bound
                        vals = [value(A, b, 1, [a + s * di for a, di in zip(r.anchor, r.direction)]) for s in (1, 10, 100)]
                        self.assertTrue(vals[0] > vals[1] > vals[2])
        self.assertTrue({("unbounded", "curvature"), ("unbounded", "slope"), ("nonunique", None)} <= seen)
        # the zero objective: every point is optimal; a linear one: unbounded along b
        r = self.both(opt.Quadratic.from_matrices([[0, 0], [0, 0]], [0, 0], 3))
        self.assertEqual((r.status, r.value, len(r.family)), ("nonunique", 3, 2))
        r = self.both(opt.Quadratic.from_matrices([[0, 0], [0, 0]], [1, -2], 3))
        self.assertEqual((r.status, r.certificate), ("unbounded", "slope"))
        # compact boxes: the exact optimum against the brute-force oracle, boundary optima, KKT exactly
        boundary = 0
        for trial in range(36):
            n = 1 + trial % 4
            A = rnd_sym(rng, n, ("pd", "psd", "indefinite")[trial % 3])
            b = [F(int(rng.integers(-12, 13)), int(rng.integers(1, 3))) for _ in range(n)]
            lo = [F(int(rng.integers(-4, 1)), int(rng.integers(1, 3))) for _ in range(n)]
            hi = [l + F(int(rng.integers(0, 5)), int(rng.integers(1, 3))) for l in lo]
            q = opt.Quadratic.from_matrices(A, b, 0)
            bounds = list(zip(lo, hi))
            r = self.both(q, bounds)
            self.assertEqual(r.value, box_oracle(A, b, 0, lo, hi), (A, b, lo, hi))
            self.assertEqual(r.value, value(A, b, 0, r.point))
            self.assertTrue(all(l <= p <= h for p, l, h in zip(r.point, lo, hi)))
            self.assertTrue(r.kkt and kkt(A, b, r.point, lo, hi))
            for p in r.optima:
                self.assertEqual(value(A, b, 0, p), r.value)
            boundary += any(r.active)
            for _ in range(10):                                  # no feasible point does better
                t = [l + (h - l) * F(int(rng.integers(0, 9)), 8) for l, h in zip(lo, hi)]
                self.assertGreaterEqual(value(A, b, 0, t), r.value)
        self.assertGreater(boundary, 10)
        # several optima: (t1 - t2)^2 on [0, 1]^2 attains 0 on the diagonal
        q = opt.Quadratic.from_matrices([[2, -2], [-2, 2]], [0, 0], 0)
        r = self.both(q, [(0, 1), (0, 1)])
        self.assertEqual((r.status, r.value), ("nonunique", 0))
        self.assertTrue([0, 0] in r.optima and [1, 1] in r.optima)
        # a degenerate bound (lo == hi) fixes the parameter
        q = opt.Quadratic.from_matrices([[2, 1], [1, 2]], [5, 5], 0)
        r = self.both(q, [(F(1, 2), F(1, 2)), (-10, 10)])
        self.assertEqual((r.point, r.active), ([F(1, 2), F(9, 4)], [-1, 0]))
        # free and bounded parameters together
        q = opt.Quadratic.from_matrices([[2, 1], [1, -1]], [0, 0], 0)          # negative curvature along t2
        r = self.both(q, [(0, 1), None])
        self.assertEqual((r.status, r.certificate), ("unbounded", "curvature"))
        self.assertEqual(r.direction[0], 0)
        q = opt.Quadratic.from_matrices([[2, 1], [1, 0]], [0, 0], 0)           # flat t2, slope t1 * t2 at t1 != 0
        r = self.both(q, [(1, 2), None])
        self.assertEqual((r.status, r.certificate), ("unbounded", "slope"))
        self.assertEqual(r.direction[0], 0)
        self.assertLess(sum(g * d for g, d in zip(gradient(q.A, q.b, r.anchor), r.direction)), 0)
        q = opt.Quadratic.from_matrices([[2, 1], [1, 4]], [1, 1], 0)           # bounded: t2 free, t1 in [1, 2]
        r = self.both(q, [(1, 2), None])
        self.assertEqual((r.status, r.point, r.active), ("unique", [1, 0], [-1, 0]))
        self.assertTrue(kkt(q.A, q.b, r.point, [1, None], [2, None]))
        q = opt.Quadratic.from_matrices([[2, 0, 0], [0, 0, 0], [0, 0, 0]], [1, 0, 0], 0)   # flat free directions
        r = self.both(q, [(1, 2), None, None])
        self.assertEqual((r.status, r.point, len(r.family)), ("nonunique", [1, 0, 0], 2))
        # refusals
        q = opt.Quadratic.from_matrices([[1, 0], [0, 1]], [0, 0], 0)
        with self.assertRaises(InfeasibleError):
            opt.minimize(q, [(1, 0), (0, 1)])
        with self.assertRaises(InfeasibleError):
            opt.minimize(q, [(1, 0), (0, 1)], backend="cuda")
        with self.assertRaises(UnsupportedError):
            opt.minimize(q, [(0, None), (0, 1)])                 # one-sided
        with self.assertRaises(UnsupportedError):
            opt.minimize(opt.Quadratic.from_matrices([[1 if i == j else 0 for j in range(9)] for i in range(9)],
                                                     [0] * 9, 0), [(0, 1)] * 9)
        with self.assertRaises(ValueError):
            opt.Quadratic.from_matrices([[1, 2], [3, 1]], [0, 0])   # not symmetric
        with self.assertRaises(TypeError):
            opt.Quadratic.from_matrices([[1.5]], [0])
        ctx = Context()
        a, bb, cc = ctx.symbols("a", "b", "c")
        with self.assertRaises(UnsupportedError):
            opt.Quadratic.from_polynomial(a ** 3 + a, ["a"])
        with self.assertRaises(ValueError):
            opt.Quadratic.from_polynomial(a * bb + cc, ["a", "b"])  # c is not a parameter

    def test_opt001_face_filter_is_rigorous_and_identical_on_cpu_and_cuda(self):
        rng = np.random.default_rng(3003)
        reasons = set()
        for trial in range(30):
            n = 1 + trial % 4
            A = rnd_sym(rng, n, ("pd", "psd", "indefinite")[trial % 3])
            b = [F(int(rng.integers(-12, 13)), int(rng.integers(1, 3))) for _ in range(n)]
            lo = [F(int(rng.integers(-4, 1))) for _ in range(n)]
            hi = [l + int(rng.integers(0, 4)) for l in lo]
            q = opt.Quadratic.from_matrices(A, b, 0)
            bounds = list(zip(lo, hi))
            cpu, gpu = opt.faces(q, bounds), opt.faces(q, bounds, "cuda")
            self.assertTrue(np.array_equal(cpu, gpu))
            self.assertEqual(len(cpu), 3 ** n)
            for face, status in enumerate(cpu):
                name = opt.FACE_STATES[status]
                reasons.add(name)
                states, f = [], face
                for _ in range(n):
                    states.append(f % 3)
                    f //= 3
                if name == "duplicate":
                    self.assertTrue(any(s and lo[i] == hi[i] for i, s in enumerate(states)))
                    continue
                if name == "not psd":
                    self.assertTrue(any(A[i][i] < 0 for i in range(n) if states[i] == 2))
                    continue
                if name not in ("infeasible", "kkt"):
                    continue
                free = [i for i in range(n) if states[i] == 2]   # a discarded face: its unique stationary point
                t = [lo[i] if s == 0 else hi[i] if s == 1 else None for i, s in enumerate(states)]
                if free:
                    s = solve([[A[i][j] for j in free] for i in free],
                              [b[i] - sum(A[i][j] * t[j] for j in range(n) if states[j] != 2) for i in free])
                    self.assertIsNotNone(s)
                    for i, v in zip(free, s):
                        t[i] = v
                if name == "infeasible":
                    self.assertTrue(any(not lo[i] <= t[i] <= hi[i] for i in free))
                else:
                    g = gradient(A, b, t)
                    self.assertTrue(any((states[i] == 0 and g[i] < 0) or (states[i] == 1 and g[i] > 0)
                                        for i in range(n) if states[i] != 2 and lo[i] != hi[i]))
            r = opt.minimize(q, bounds)
            self.assertEqual(sum(r.filtered.values()), r.faces)
            self.assertLessEqual(r.exact_checks, r.faces)
        self.assertTrue({"candidate", "infeasible", "kkt"} <= reasons)

    def test_opt001_budgets_at_the_limit_and_one_past(self):
        q = opt.Quadratic.from_matrices([[2, 0, 0], [0, 2, 0], [0, 0, 2]], [1, 1, 1], 0)
        bounds = [(0, 1), (0, 1), None]
        opt.minimize(q, bounds, budget=Budget(iterations=9))        # 3^2 faces
        with self.assertRaises(AlgebraBudgetError) as ctx:
            opt.minimize(q, bounds, budget=Budget(iterations=8))
        self.assertEqual((ctx.exception.resource, ctx.exception.need, ctx.exception.limit), ("iterations", 9, 8))
        with self.assertRaises(AlgebraBudgetError):
            opt.minimize(q, bounds, backend="cuda", budget=Budget(iterations=8))
        opt.minimize(q, budget=Budget(variables=3, tensor=12))      # the augmented system [A | b]
        with self.assertRaises(AlgebraBudgetError) as ctx:
            opt.minimize(q, budget=Budget(tensor=11))
        self.assertEqual((ctx.exception.resource, ctx.exception.need), ("tensor", 12))
        with self.assertRaises(AlgebraBudgetError):
            opt.minimize(q, budget=Budget(variables=2))
        with self.assertRaises(AlgebraBudgetError):
            opt.Quadratic.from_matrices([[1]], [0], 0, budget=Budget(variables=0))
        opt.minimize(q, bounds, budget=Budget(output=3 * 10))       # n x (faces + 1)
        with self.assertRaises(AlgebraBudgetError):
            opt.minimize(q, bounds, budget=Budget(output=3 * 10 - 1))
        with self.assertRaises(UnsupportedError):
            opt.Quadratic.from_matrices([[1 if i == j else 0 for j in range(64)] for i in range(64)], [0] * 64, 0)

    def test_opt001_shared_states_against_full_objective_recomputation(self):
        rng = np.random.default_rng(3004)
        n = 300
        x1, x2 = rng.integers(-50, 50, n), rng.integers(-50, 50, n)
        y = 3 * x1 - x2 + rng.integers(-20, 20, n)
        groups = rng.integers(0, 4, n)
        scope = mbk.Scope.from_arrays({"x1": x1, "x2": x2, "y": y}, groups=groups, ngroups=4)
        st = lin.moments(scope, ["x1", "x2"], "y")                   # one scan, many requests
        for g in range(4):
            sel = groups == g
            Z = [[F(1), F(int(a)), F(int(b))] for a, b in zip(x1[sel], x2[sel])]
            yv = [F(int(v)) for v in y[sel]]
            for ridge, bounds in ((0, None), (F(1, 3), None), (0, [None, (0, 2), (-2, 0)]), (5, [(-1, 1), (0, 1), None])):
                A = [[2 * (sum(z[i] * z[j] for z in Z) + (ridge if i == j and i else 0)) for j in range(3)] for i in range(3)]
                b = [2 * sum(z[i] * yy for z, yy in zip(Z, yv)) for i in range(3)]
                c = sum(yy * yy for yy in yv)
                fresh = opt.Quadratic.from_matrices(A, b, c)          # the objective recomputed from the rows
                shared = opt.Quadratic.from_moments(st, g, ridge=ridge)
                self.assertEqual((shared.A, shared.b, shared.c), (A, b, c))
                r1, r2 = self.both(shared, bounds), self.both(fresh, bounds)
                self.assertTrue(r1.same(r2))
                loss = lambda t: sum((yy - sum(ti * zi for ti, zi in zip(t, z))) ** 2 for z, yy in zip(Z, yv)) + \
                    ridge * (t[1] ** 2 + t[2] ** 2)
                self.assertEqual(r1.value, loss(r1.point))


if __name__ == "__main__":
    unittest.main()
