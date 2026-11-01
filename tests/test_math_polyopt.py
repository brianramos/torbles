# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""OPT-002: bounded convex and polynomial optimization (torbles/math_polyopt.c, math_polyopt_gpu.cu, math_polyopt.py).

Oracles, independent of the implementation and written here with Python Fractions: the vertices of small polytopes
enumerated directly (every n-subset of the constraints solved by Gauss-Jordan elimination, kept when feasible), the
stationary points of every face of a QP (the KKT system of every independent active set: the exact global minimum of
a convex or indefinite quadratic over a polytope), exhaustive feasible grids for polynomial problems, exact interval
arithmetic over the leaves of infeasibility certificates, and OPT-001 / LIN-001 for least-squares objectives. The box
enclosures run on the CPU and on the CUDA path (emulated here with TORBLES_EMU_SCRAMBLE=1) and are compared bit for
bit; every branch-and-bound result is computed on both paths and compared. Seeds are fixed.
"""
import itertools
import unittest
from fractions import Fraction as F

import numpy as np

from torbles import math_bank as mbk
from torbles import math_lin as lin
from torbles import math_opt as opt
from torbles import math_polyopt as po
from torbles.math_budgets import Budget
from torbles.math_errors import InfeasibleError, NotConvergedError, UnsupportedError
from torbles.mathematics import AlgebraBudgetError, Context


# ---------------------------------------------------------------------------------------------------- exact oracles
def gauss(M, rhs):
    """the unique solution of a square system (Fraction Gauss-Jordan), or None when singular"""
    n = len(M)
    A = [[F(v) for v in row] + [F(b)] for row, b in zip(M, rhs)]
    for c in range(n):
        p = next((i for i in range(c, n) if A[i][c] != 0), None)
        if p is None:
            return None
        A[c], A[p] = A[p], A[c]
        inv = 1 / A[c][c]
        A[c] = [v * inv for v in A[c]]
        for i in range(n):
            if i != c and A[i][c]:
                f = A[i][c]
                A[i] = [a - f * b for a, b in zip(A[i], A[c])]
    return [A[i][n] for i in range(n)]


def rank(rows):
    A = [[F(v) for v in r] for r in rows]
    rk, cols = 0, len(A[0]) if A else 0
    for c in range(cols):
        p = next((i for i in range(rk, len(A)) if A[i][c] != 0), None)
        if p is None:
            continue
        A[rk], A[p] = A[p], A[rk]
        for i in range(len(A)):
            if i != rk and A[i][c]:
                f = A[i][c] / A[rk][c]
                A[i] = [a - f * b for a, b in zip(A[i], A[rk])]
        rk += 1
    return rk


def dot(a, b):
    return sum(F(x) * F(y) for x, y in zip(a, b))


def rows_of(n, A, b, E, e, bounds):
    """all constraints as (row, rhs, kind) with kind 'le' / 'eq', bounds included"""
    out = [(list(r), F(v), "le") for r, v in zip(A or [], b or [])]
    out += [(list(r), F(v), "eq") for r, v in zip(E or [], e or [])]
    for i, bd in enumerate(bounds or [None] * n):
        if bd is None:
            continue
        lo, hi = bd
        unit = [0] * n
        if lo is not None:
            r = list(unit); r[i] = -1
            out.append((r, -F(lo), "le"))
        if hi is not None:
            r = list(unit); r[i] = 1
            out.append((r, F(hi), "le"))
    return out


def feasible(rows, x):
    return all((dot(r, x) <= v) if k == "le" else (dot(r, x) == v) for r, v, k in rows)


def qvalue(H, g, x, c0=0):
    n = len(x)
    quad = sum(F(x[i]) * F(H[i][j]) * F(x[j]) for i in range(n) for j in range(n)) if H is not None else 0
    return quad / 2 + dot(g, x) + F(c0)


def stationary_points(H, g, rows, n):
    """the solutions of the KKT systems of every independent active set (equalities always active) that are
    feasible: the global minimum of a quadratic over a polytope is attained at one of them"""
    Hm = H if H is not None else [[0] * n for _ in range(n)]
    eqs = [r for r in rows if r[2] == "eq"]
    ineqs = [r for r in rows if r[2] == "le"]
    out = []
    for k in range(0, n + 1):
        for S in itertools.combinations(range(len(ineqs)), k):
            act = eqs + [ineqs[i] for i in S]
            if len(act) > n or (act and rank([r for r, _, _ in act]) < len(act)):
                continue
            m = len(act)
            K = [[F(Hm[i][j]) for j in range(n)] + [F(act[a][0][i]) for a in range(m)] for i in range(n)]
            K += [[F(act[a][0][j]) for j in range(n)] + [F(0)] * m for a in range(m)]
            rhs = [-F(v) for v in g] + [act[a][1] for a in range(m)]
            sol = gauss(K, rhs)
            if sol is None:
                continue
            x = sol[:n]
            if feasible(rows, x):
                out.append(x)
    return out


def vertices(rows, n):
    out = []
    for S in itertools.combinations(range(len(rows)), n):
        M = [rows[i][0] for i in S]
        x = gauss(M, [rows[i][1] for i in S])
        if x is not None and feasible(rows, x) and x not in out:
            out.append(x)
    return out


def check_kkt(tc, r, H, g, A, b, E, e, bounds):
    """the KKT conditions recomputed here from the returned multipliers"""
    n = len(g)
    x = r.point
    rows = rows_of(n, A, b, E, e, bounds)
    tc.assertTrue(feasible(rows, x))
    lam, mu = r.multipliers["ineq"], r.multipliers["eq"]
    llo, lhi = r.multipliers["lower"], r.multipliers["upper"]
    tc.assertTrue(all(v >= 0 for v in lam + llo + lhi))
    grad = [sum(F(H[i][j]) * x[j] for j in range(n)) + F(g[i]) if H is not None else F(g[i]) for i in range(n)]
    for i, (row, v) in enumerate(zip(A or [], b or [])):
        tc.assertEqual(lam[i] * (F(v) - dot(row, x)), 0)
        for j in range(n):
            grad[j] += lam[i] * F(row[j])
    for i, row in enumerate(E or []):
        for j in range(n):
            grad[j] += mu[i] * F(row[j])
    for i, bd in enumerate(bounds or [None] * n):
        lo, hi = (None, None) if bd is None else bd
        if lo is None:
            tc.assertEqual(llo[i], 0)
        else:
            tc.assertEqual(llo[i] * (x[i] - F(lo)), 0)
        if hi is None:
            tc.assertEqual(lhi[i], 0)
        else:
            tc.assertEqual(lhi[i] * (F(hi) - x[i]), 0)
        grad[i] += lhi[i] - llo[i]
    tc.assertEqual(grad, [0] * n)


def check_farkas(tc, r, A, b, E, e, bounds, n):
    y, z, ylo, yhi = r.farkas["y"], r.farkas["z"], r.farkas["lower"], r.farkas["upper"]
    tc.assertTrue(all(v >= 0 for v in y + ylo + yhi))
    s = [sum(y[i] * F(A[i][j]) for i in range(len(y))) + sum(z[i] * F(E[i][j]) for i in range(len(z))) + yhi[j] - ylo[j]
         for j in range(n)]
    tc.assertEqual(s, [0] * n)
    val = dot(y, b or []) + dot(z, e or []) + sum(yhi[i] * F(bd[1]) for i, bd in enumerate(bounds or [None] * n)
                                                 if bd is not None and bd[1] is not None) \
        - sum(ylo[i] * F(bd[0]) for i, bd in enumerate(bounds or [None] * n) if bd is not None and bd[0] is not None)
    tc.assertLess(val, 0)


def check_ray(tc, r, H, g, A, b, E, e, bounds, convex=True):
    """q(x0 + s d) = q(x0) + s g(x0)^T d + s^2 d^T H d / 2 -> -inf: d a recession direction with d^T H d < 0, or
    d^T H d = 0 and g(x0)^T d < 0 (convex: H d = 0)"""
    n = len(g)
    x0, d = r.ray["anchor"], r.ray["direction"]
    rows = rows_of(n, A, b, E, e, bounds)
    tc.assertTrue(feasible(rows, x0))
    tc.assertNotEqual(d, [0] * n)
    for row, v, k in rows:
        if k == "le":
            tc.assertLessEqual(dot(row, d), 0)
        else:
            tc.assertEqual(dot(row, d), 0)
    hd = [dot(H[i], d) if H is not None else F(0) for i in range(n)]
    if convex:
        tc.assertEqual(hd, [0] * n)
    curv = dot(d, hd)
    tc.assertEqual(curv, r.ray["curvature"])
    grad = [(dot(H[i], x0) if H is not None else 0) + F(g[i]) for i in range(n)]
    tc.assertEqual(dot(grad, d), r.ray["rate"])
    tc.assertLessEqual(curv, 0)
    if curv == 0:
        tc.assertLess(r.ray["rate"], 0)
    for s in (1, 10, 1000):                                          # the exact expansion along the ray
        qs = qvalue(H, g, [x0[i] + s * d[i] for i in range(n)])
        tc.assertEqual(qs, qvalue(H, g, x0) + s * r.ray["rate"] + F(s * s, 2) * curv)


def rnd_rows(rng, n, m, lo=-4, hi=5):
    A = [[int(rng.integers(lo, hi)) for _ in range(n)] for _ in range(m)]
    b = [int(rng.integers(0, 9)) for _ in range(m)]
    return A, b


class _Poly:
    """a polynomial as {exponent tuple: Fraction} with exact evaluation and exact natural interval extension"""

    def __init__(self, terms):
        self.terms = {tuple(e): F(c) for e, c in terms.items() if c}

    def __call__(self, x):
        return sum(c * np.prod([F(xi) ** k for xi, k in zip(x, e)], dtype=object) for e, c in self.terms.items())

    def interval(self, box):
        lo = hi = F(0)
        for e, c in self.terms.items():
            tl, th = c, c
            for (a, b), k in zip(box, e):
                if not k:
                    continue
                pl, ph = F(a) ** k, F(b) ** k
                cand = [pl, ph]
                if k % 2 == 0 and a < 0 < b:
                    cand.append(F(0))
                plo, phi = min(cand), max(cand)
                prods = [tl * plo, tl * phi, th * plo, th * phi]
                tl, th = min(prods), max(prods)
            lo, hi = lo + tl, hi + th
        return lo, hi


def proves_positive(p, box, depth=10):
    """exact proof that p > 0 on the box: the natural interval extension, bisected when needed"""
    lo, _ = p.interval(box)
    if lo > 0:
        return True
    if depth == 0:
        return False
    k = max(range(len(box)), key=lambda i: box[i][1] - box[i][0])
    a, b = box[k]
    m = (a + b) / 2
    left, right = list(box), list(box)
    left[k], right[k] = (a, m), (m, b)
    return proves_positive(p, left, depth - 1) and proves_positive(p, right, depth - 1)


class PolyOpt(unittest.TestCase):
    def both(self, *args, **kwargs):
        """the branch and bound on the CPU and with the CUDA enclosures (emulated, scrambled): the same answer"""
        cpu = po.polynomial(*args, **kwargs)
        gpu = po.polynomial(*args, backend="cuda", **kwargs)
        self.assertTrue(cpu.same(gpu))
        self.assertIn(gpu.backend, ("cuda", "cuda-emulated"))
        return cpu

    # ------------------------------------------------------------------------------------------------ LP
    def test_opt002_lp_optima_match_enumerated_polytope_vertices_with_kkt_and_uniqueness(self):
        rng = np.random.default_rng(20261101)
        checked = 0
        for trial in range(36):
            n = 2 + trial % 2
            m = int(rng.integers(1, 5))
            A, b = rnd_rows(rng, n, m)
            bounds = [(int(rng.integers(-4, 0)), int(rng.integers(1, 5))) for _ in range(n)]
            E, e = ([[int(v) for v in rng.integers(-2, 3, n)]], [int(rng.integers(-2, 3))]) if trial % 5 == 0 else (None, None)
            c = [int(rng.integers(-3, 4)) for _ in range(n)]
            if trial % 7 == 0:
                c = [A[0][j] * -1 for j in range(n)]               # parallel to a facet: often nonunique
            rows = rows_of(n, A, b, E, e, bounds)
            verts = vertices(rows, n)
            r = po.lp(c, A_ub=A, b_ub=b, A_eq=E, b_eq=e, bounds=bounds)
            if not verts:
                self.assertEqual(r.status, "infeasible")
                check_farkas(self, r, A, b, E or [], e or [], bounds, n)
                continue
            best = min(dot(c, v) for v in verts)
            self.assertEqual(r.status, "optimal")
            self.assertEqual(r.method, "simplex")
            self.assertEqual(r.value, best)
            self.assertEqual(dot(c, r.point), best)
            self.assertTrue(r.kkt)
            check_kkt(self, r, None, c, A, b, E, e, bounds)
            optimal_vertices = [v for v in verts if dot(c, v) == best]
            self.assertEqual(r.unique, len(optimal_vertices) == 1)          # a polytope: the hull of its optimal vertices
            if not r.unique:
                self.assertTrue(feasible(rows, r.face))
                self.assertEqual(dot(c, r.face), best)
                self.assertNotEqual(r.face, r.point)
            checked += 1
        self.assertGreater(checked, 20)

    def test_opt002_infeasible_farkas_certificates_and_unbounded_rays_verified_independently(self):
        rng = np.random.default_rng(7)
        for trial in range(24):
            n = 2 + trial % 3
            A, b = rnd_rows(rng, n, 3)
            # x_0 >= 1 + |b| and a row forcing x_0 <= -1: infeasible
            A2 = A + [[-1] + [0] * (n - 1), [1] + [0] * (n - 1)]
            b2 = b + [-1, -2]
            r = po.lp([1] * n, A_ub=A2, b_ub=b2)
            self.assertEqual(r.status, "infeasible")
            self.assertEqual(r.feasibility, "infeasible")
            self.assertTrue(r.certificate)
            check_farkas(self, r, A2, b2, [], [], None, n)
            # inconsistent equalities
            E = [[1] * n, [1] * n]
            r = po.lp([0] * n, A_eq=E, b_eq=[1, 2])
            self.assertEqual(r.status, "infeasible")
            check_farkas(self, r, [], [], E, [1, 2], None, n)
            # unbounded: minimize -x_0 over a cone
            Au = [[1] + [-1] * (n - 1)]
            r = po.lp([-1] + [0] * (n - 1), A_ub=Au, b_ub=[1], bounds=[(0, None)] * n)
            self.assertEqual(r.status, "unbounded")
            check_ray(self, r, None, [-1] + [0] * (n - 1), Au, [1], [], [], [(0, None)] * n)
        # convex QP unbounded along a flat direction of H, and infeasible bounds
        H = [[2, 0], [0, 0]]
        r = po.qp(H, [0, -1], A_ub=[[1, 0]], b_ub=[3])
        self.assertEqual(r.status, "unbounded")
        check_ray(self, r, H, [0, -1], [[1, 0]], [3], [], [], None)
        r = po.qp(H, [0, 1], bounds=[(2, 1), None])
        self.assertEqual(r.status, "infeasible")
        check_farkas(self, r, [], [], [], [], [(2, 1), None], 2)

    # ------------------------------------------------------------------------------------------------ convex QP
    def test_opt002_convex_qps_match_face_enumeration_with_kkt_certificates(self):
        rng = np.random.default_rng(11)
        for trial in range(30):
            n = 2 + trial % 2
            B = rng.integers(-3, 4, (n, n))
            H = (B.T @ B).tolist()
            if trial % 3 == 0:                                      # PSD singular: rank n - 1
                B[0] = 0
                H = (B.T @ B).tolist()
            else:
                H = [[H[i][j] + (1 if i == j else 0) for j in range(n)] for i in range(n)]
            g = [int(v) for v in rng.integers(-6, 7, n)]
            A, b = rnd_rows(rng, n, int(rng.integers(1, 4)))
            bounds = [(int(rng.integers(-3, 0)), int(rng.integers(1, 4))) for _ in range(n)]
            E, e = ([[1] * n], [int(rng.integers(-1, 2))]) if trial % 4 == 0 else (None, None)
            rows = rows_of(n, A, b, E, e, bounds)
            pts = stationary_points(H, g, rows, n)
            r = po.qp(H, g, A_ub=A, b_ub=b, A_eq=E, b_eq=e, bounds=bounds)
            if not pts:
                self.assertEqual(r.status, "infeasible")
                check_farkas(self, r, A, b, E or [], e or [], bounds, n)
                continue
            best = min(qvalue(H, g, x) for x in pts)
            self.assertEqual(r.status, "optimal", trial)
            self.assertEqual(r.method, "active set")
            self.assertTrue(r.convex)
            self.assertEqual(r.value, best)
            self.assertEqual(qvalue(H, g, r.point), best)
            check_kkt(self, r, H, g, A, b, E, e, bounds)
            if r.unique is False:
                self.assertTrue(feasible(rows, r.face))
                self.assertEqual(qvalue(H, g, r.face), best)
            if trial % 3:                                            # positive definite: a unique minimizer
                self.assertTrue(r.unique)

    def test_opt002_least_squares_objectives_of_shared_lin001_states_under_linear_constraints(self):
        rng = np.random.default_rng(5)
        rows = 400
        x1 = rng.integers(-20, 21, rows)
        x2 = rng.integers(-20, 21, rows)
        y = 3 * x1 - 2 * x2 + rng.integers(-5, 6, rows)
        scope = mbk.Scope.from_arrays({"x1": x1, "x2": x2, "y": y})
        st = lin.moments(scope, ["x1", "x2"], "y")
        q = opt.Quadratic.from_moments(st, 0)                        # unknowns: intercept, x1, x2
        # nonnegative slopes summing to 1 (a simplex constraint): the exact constrained least squares
        r = po.qp_from(q, A_eq=[[0, 1, 1]], b_eq=[1], bounds=[None, (0, None), (0, None)])
        X = [[1, int(a), int(c)] for a, c in zip(x1, x2)]
        ys = [int(v) for v in y]
        H = [[2 * sum(F(r1[i]) * F(r1[j]) for r1 in X) for j in range(3)] for i in range(3)]
        g = [-2 * sum(F(r1[i]) * F(yy) for r1, yy in zip(X, ys)) for i in range(3)]
        cst = sum(F(v) ** 2 for v in ys)
        frows = rows_of(3, None, None, [[0, 1, 1]], [1], [None, (0, None), (0, None)])
        pts = stationary_points(H, g, frows, 3)
        best = min(qvalue(H, g, p) for p in pts) + cst
        self.assertEqual(r.status, "optimal")
        self.assertEqual(r.value, best)
        self.assertEqual(q.value(r.point), best)
        check_kkt(self, r, [[F(v) for v in row] for row in q.A], [-v for v in q.b], None, None, [[0, 1, 1]], [1],
                  [None, (0, None), (0, None)])
        # a box only: the same exact optimum as OPT-001's box minimizer
        bx = [(-5, 5), (0, 1), (-1, 0)]
        r = po.qp_from(q, bounds=bx)
        o = opt.minimize(q, bx)
        self.assertEqual((r.value, r.point), (o.value, o.point))

    # ------------------------------------------------------------------------------------------------ nonconvex
    def test_opt002_indefinite_qps_are_handed_to_certified_branch_and_bound(self):
        rng = np.random.default_rng(3)
        for trial in range(10):
            n = 2
            B = rng.integers(-3, 4, (n, n))
            H = (B.T @ B - 6 * np.eye(n, dtype=np.int64)).tolist()     # some negative curvature
            g = [int(v) for v in rng.integers(-4, 5, n)]
            A, b = rnd_rows(rng, n, 2)
            bounds = [(-2, 2), (-2, 2)]
            rows = rows_of(n, A, b, None, None, bounds)
            pts = stationary_points(H, g, rows, n)
            tol = F(1, 10 ** 4)
            r = po.qp(H, g, A_ub=A, b_ub=b, bounds=bounds, tolerance=tol)
            r2 = po.qp(H, g, A_ub=A, b_ub=b, bounds=bounds, tolerance=tol, backend="cuda")
            self.assertTrue(r.same(r2))
            if not pts:
                self.assertEqual(r.status, "infeasible")
                continue
            if min(np.linalg.eigvalsh(np.array(H, float))) >= 0:
                continue
            fstar = min(qvalue(H, g, x) for x in pts)                # the exact global minimum
            self.assertTrue(r.handed)
            self.assertFalse(r.convex)
            self.assertEqual(r.method, "branch and bound")
            self.assertEqual(r.status, "complete")
            self.assertLessEqual(r.lower, fstar)
            self.assertLessEqual(fstar, r.upper)
            self.assertLessEqual(r.upper - r.lower, tol)
            self.assertTrue(feasible(rows, r.point))
            self.assertEqual(qvalue(H, g, r.point), r.upper)
        # an unbounded polyhedron with an indefinite objective and no recession direction of negative curvature
        # (H copositive on the cone x >= 0): not decided by these methods, refused, never guessed
        with self.assertRaises(UnsupportedError):
            po.qp([[1, 3], [3, 1]], [1, 1], bounds=[(0, None), (0, None)])

    def test_opt002_indefinite_qps_on_affine_sets_and_recession_rays_are_decided_exactly(self):
        I2 = [[1, 0], [0, -1]]
        cases = [   # H, g, A_ub, b_ub, A_eq, b_eq, bounds, expected status
            ([[1, 2], [2, 1]], [1, 0], None, None, None, None, None, "unbounded"),        # unconstrained
            (I2, [0, 0], None, None, [[0, 1]], [1], None, "optimal"),                    # x2 = 1: convex in x1
            (I2, [0, 0], None, None, [[1, 0]], [2], None, "unbounded"),                  # x1 = 2: -x2^2 / 2
            ([[0, 1], [1, 0]], [0, 1], None, None, [[1, 0]], [0], None, "unbounded"),    # x1 = 0: q = x2 (rate)
            ([[0, 1], [1, 0]], [0, 0], None, None, [[1, 0]], [0], None, "optimal"),      # q = 0: nonunique
            ([[1, 0, 0], [0, -1, 0], [0, 0, 2]], [1, 0, -1], [[0, 1, 0]], [3], [[0, 1, 0]], [1], None, "optimal"),
            ([[-1, 0], [0, 1]], [0, 0], None, None, None, None, [(0, None), (-1, 1)], "unbounded"),   # along x1
            ([[-1, 0], [0, -1]], [0, 0], [[1, -1]], [0], None, None, [(0, None), (0, None)], "unbounded"),
        ]
        for k, (H, g, A, b, E, e, bounds, status) in enumerate(cases):
            r = po.qp(H, g, A_ub=A, b_ub=b, A_eq=E, b_eq=e, bounds=bounds)
            r2 = po.qp(H, g, A_ub=A, b_ub=b, A_eq=E, b_eq=e, bounds=bounds, backend="cuda")
            self.assertTrue(r.same(r2), k)
            self.assertEqual(r.status, status, k)
            self.assertFalse(r.handed, k)
            if status == "unbounded":
                check_ray(self, r, H, g, A, b, E, e, bounds, convex=False)
                continue
            check_kkt(self, r, H, g, A, b, E, e, bounds)
            self.assertTrue(r.convex, k)                              # convex on the (affine) feasible set
            n = len(g)
            rows = rows_of(n, A, b, E, e, bounds)
            # on the affine set every feasible point is x(t) = x* + t v: the minimum checked along its directions
            if r.unique:
                self.assertEqual(k in (1, 5), True)
            else:
                self.assertTrue(feasible(rows, r.face))
                self.assertEqual(qvalue(H, g, r.face), r.value)
                self.assertNotEqual(r.face, r.point)
        r = po.qp(I2, [0, 0], A_eq=[[0, 1]], b_eq=[1])
        self.assertEqual((r.point, r.value, r.unique, r.multipliers["eq"]), ([0, 1], F(-1, 2), True, [1]))
        r = po.qp([[1, 0, 0], [0, -1, 0], [0, 0, 2]], [1, 0, -1], A_ub=[[0, 1, 0]], b_ub=[3], A_eq=[[0, 1, 0]],
                  b_eq=[1])
        self.assertEqual((r.point, r.value, r.unique), ([-1, 1, F(1, 2)], F(-1, 2) - F(1, 2) - F(1, 4), True))

    def test_opt002_polynomial_minima_against_exhaustive_feasible_sets_and_known_optima(self):
        ctx = Context()
        x, y, z = ctx.symbols("x", "y", "z")
        cases = [
            # (objective, box, constraints, oracle polynomial, exact minimum or None)
            ((x - F(1, 3)) ** 2 * 4 + (y + F(2, 5)) ** 2, {"x": (-1, 1), "y": (-1, 1)}, [],
             _Poly({(2, 0): 4, (1, 0): F(-8, 3), (0, 0): F(4, 9) + F(4, 25), (0, 2): 1, (0, 1): F(4, 5)}), F(0)),
            (x ** 4 - 2 * x ** 2 + y ** 2, {"x": (-2, 2), "y": (-1, 1)}, [],
             _Poly({(4, 0): 1, (2, 0): -2, (0, 2): 1}), F(-1)),
            (x * y, {"x": (-1, 2), "y": (-3, 1)}, [(x + y, "<=", 0)], _Poly({(1, 1): 1}), F(-6)),
            (x ** 2 + y ** 2, {"x": (-2, 2), "y": (-2, 2)}, [(x + 2 * y, ">=", 2)], _Poly({(2, 0): 1, (0, 2): 1}),
             F(4, 5)),
            (x ** 3 - 3 * x * y + y ** 3, {"x": (-1, 2), "y": (-1, 2)}, [], _Poly({(3, 0): 1, (1, 1): -3, (0, 3): 1}),
             F(-5)),
            (x * y * z + x ** 2, {"x": (-1, 1), "y": (-1, 1), "z": (-1, 1)}, [(x + y + z, "<=", 1)],
             _Poly({(1, 1, 1): 1, (2, 0, 0): 1}), None),
        ]
        cons_polys = {2: [(_Poly({(1, 0): 1, (0, 1): 1}), "<=", 0)], 3: [(_Poly({(1, 0): 1, (0, 1): 2}), ">=", 2)],
                      5: [(_Poly({(1, 0, 0): 1, (0, 1, 0): 1, (0, 0, 1): 1}), "<=", 1)]}
        tol = F(1, 10 ** 5)
        for k, (f, box, cons, oracle, exact) in enumerate(cases):
            r = self.both(f, box, cons, tolerance=tol)
            names = list(box)
            # the exhaustive feasible set of a rational grid: every feasible point is >= the minimum >= lower
            axes = [[F(lo) + F(hi - lo) * i / 12 for i in range(13)] for lo, hi in box.values()]
            feas = []
            for p in itertools.product(*axes):
                ok = all((g(p) <= rhs) if op == "<=" else (g(p) >= rhs) for g, op, rhs in cons_polys.get(k, []))
                if ok:
                    feas.append(oracle(p))
            self.assertTrue(feas)
            self.assertEqual(r.status, "complete", k)
            self.assertEqual(r.feasibility, "feasible")
            self.assertLessEqual(r.lower, min(feas))
            self.assertLessEqual(r.upper - r.lower, tol)
            self.assertEqual(oracle(r.point), r.upper)               # the witness attains upper exactly
            self.assertTrue(all((g(r.point) <= rhs) if op == "<=" else (g(r.point) >= rhs)
                                for g, op, rhs in cons_polys.get(k, [])))
            for name, v in zip(names, r.point):
                lo, hi = box[name]
                self.assertTrue(lo <= v <= hi)
            if exact is not None:
                self.assertLessEqual(r.lower, exact)
                self.assertLessEqual(exact, r.upper)

    def test_opt002_univariate_problems_use_inv001_roots_exact_or_certified(self):
        ctx = Context()
        x, = ctx.symbols("x")
        # rational critical point 2/3: f(2/3) exact; the global minimum over [-1, 2] is exact (gap 0)
        f = 3 * x ** 3 - 3 * x ** 2 + 4                              # f' = 9x^2 - 6x: critical points 0, 2/3
        r = self.both(f, {"x": (-1, 2)}, tolerance=0)
        cand = {F(-1), F(2), F(0), F(2, 3)}
        fv = lambda t: 3 * t ** 3 - 3 * t ** 2 + 4
        self.assertEqual(r.method, "roots")
        self.assertEqual(r.status, "complete")
        self.assertEqual(r.lower, r.upper)
        self.assertEqual(r.upper, min(fv(t) for t in cand))
        # irrational critical points: certified within the tolerance
        f = x ** 4 - 3 * x ** 2 + x
        tol = F(1, 10 ** 12)
        r = self.both(f, {"x": (-2, 2)}, tolerance=tol)
        crit = [c.real for c in np.roots([4, 0, -6, 1]) if abs(c.imag) < 1e-12]
        fmin = min(c ** 4 - 3 * c ** 2 + c for c in crit + [-2.0, 2.0])
        self.assertEqual(r.status, "complete")
        self.assertLessEqual(r.upper - r.lower, tol)
        self.assertLessEqual(float(r.lower), fmin + 1e-9)
        self.assertGreaterEqual(float(r.upper), fmin - 1e-9)
        self.assertEqual(r.point[0] ** 4 - 3 * r.point[0] ** 2 + r.point[0], r.upper)

    # ------------------------------------------------------------------------------------------------ limits
    def test_opt002_resource_limited_search_returns_explicit_incomplete_bounds(self):
        ctx = Context()
        x, y = ctx.symbols("x", "y")
        f = x ** 4 - 3 * x ** 2 + y ** 2 + x * y
        fstar = -F(13, 4) ** 2 / 4                                   # at x^2 = 13/8, y = -x/2
        seen = set()
        for nodes in (1, 2, 5, 20, 100):
            r = self.both(f, {"x": (-2, 2), "y": (-2, 2)}, tolerance=0, budget=Budget(iterations=nodes))
            self.assertEqual(r.status, "incomplete")
            self.assertEqual(r.reason, "nodes")
            self.assertLessEqual(r.iterations, nodes)
            self.assertFalse(r.exact)
            self.assertLessEqual(r.lower, fstar)
            if r.upper is not None:
                self.assertGreaterEqual(r.upper, fstar)
                self.assertEqual(_Poly({(4, 0): 1, (2, 0): -3, (0, 2): 1, (1, 1): 1})(r.point), r.upper)
            seen.add((r.lower, r.upper))
        self.assertGreater(len(seen), 2)                             # the bounds tighten with the budget
        r = po.polynomial(f, {"x": (-2, 2), "y": (-2, 2)}, tolerance=0, budget=Budget(iterations=0))
        self.assertEqual((r.status, r.lower, r.feasibility), ("incomplete", None, "feasible"))   # corners only
        r = po.polynomial(f, {"x": (-2, 2), "y": (-2, 2)}, tolerance=0, time_limit=1e-6)
        self.assertEqual(r.status, "incomplete")
        self.assertIn(r.reason, ("time", "nodes"))
        # an equality whose solutions are irrational: no exact witness, so feasibility stays unknown, with a bound
        r = po.polynomial(x + y, {"x": (0, 2), "y": (0, 2)}, [(x * x, "=", 2)], tolerance=F(1, 100),
                          budget=Budget(iterations=400))
        self.assertEqual(r.status, "incomplete")
        self.assertEqual(r.feasibility, "unknown")
        self.assertIsNone(r.upper)
        self.assertLessEqual(r.lower, 2 ** 0.5 + 1e-12)

    def test_opt002_infeasibility_certificates_are_leaf_partitions_proven_exactly(self):
        ctx = Context()
        x, y = ctx.symbols("x", "y")
        g1 = _Poly({(2, 0): 1, (0, 2): 1, (0, 0): F(-1, 4)})           # x^2 + y^2 <= 1/4
        g2 = _Poly({(1, 0): -1, (0, 1): -1, (0, 0): 1})                # x + y >= 1
        r = self.both(x + y, {"x": (-1, 1), "y": (-1, 1)}, [(x ** 2 + y ** 2, "<=", F(1, 4)), (x + y, ">=", 1)],
                      tolerance=F(1, 100))
        self.assertEqual(r.status, "infeasible")
        self.assertEqual(r.feasibility, "infeasible")
        self.assertTrue(r.certificate)
        self.assertTrue(r.leaves)
        area = F(0)
        for lo, hi, con in r.leaves:
            box = list(zip(lo, hi))
            area += (hi[0] - lo[0]) * (hi[1] - lo[1])
            g = g1 if con == 0 else g2                               # the constraint the leaf violates
            self.assertTrue(proves_positive(g, box), (lo, hi, con))
        self.assertEqual(area, 4)                                    # the leaves partition the box

    # ------------------------------------------------------------------------------------------------ kernels
    def test_opt002_box_enclosures_are_sound_and_identical_on_cpu_and_cuda(self):
        rng = np.random.default_rng(99)
        for trial in range(12):
            n = 1 + trial % 3
            terms = {}
            for _ in range(int(rng.integers(2, 7))):
                e = tuple(int(v) for v in rng.integers(0, 4, n))
                terms[e] = F(int(rng.integers(-9, 10)), int(rng.integers(1, 5)))
            cons = {tuple(int(v) for v in rng.integers(0, 3, n)): F(int(rng.integers(-5, 6)))}
            pr = po.Problem(terms, [(-3, 3)] * n, [(cons, "<=", 1)])
            boxes = []
            for _ in range(20):
                b = []
                for _ in range(n):
                    a = F(int(rng.integers(-30, 30)), 10)
                    b.append((a, a + F(int(rng.integers(0, 15)), 7)))
                boxes.append(b)
            ub = F(int(rng.integers(-20, 20)))
            cpu = po.boxes(pr, boxes, ub)
            gpu = po.boxes(pr, boxes, ub, backend="cuda")
            self.assertEqual(cpu, gpu)
            P = _Poly(terms)
            G = _Poly({k: v for k, v in cons.items()})
            for b, out in zip(boxes, cpu):
                pts = list(itertools.product(*[(lo, (lo + hi) / 2, hi) for lo, hi in b]))
                for p in pts:
                    v = P(p)
                    self.assertLessEqual(F(out["objective"][0]), v)    # every point is enclosed
                    self.assertLessEqual(v, F(out["objective"][1]))
                if out["status"] == "infeasible":
                    self.assertTrue(all(G(p) - 1 > 0 for p in pts))
                if out["status"] == "bound":
                    self.assertTrue(all(P(p) > ub for p in pts))
                if out["inner"]:
                    self.assertTrue(all(G(p) <= 1 for p in pts))

    def test_opt002_budgets_at_the_limit_and_one_past(self):
        # the tableau of an LP: tensor = (rows + 1) (columns + 1); rows = 3 (2 bounds + 1), columns 2n + m + rows
        n, A, b = 2, [[1, 1]], [4]
        bounds = [(0, None), (0, None)]
        R, cols = 3, 2 * n + 3 + 3
        need = (R + 1) * (cols + 1)
        po.lp([1, 1], A_ub=A, b_ub=b, bounds=bounds, uniqueness=False, budget=Budget(tensor=need))
        with self.assertRaises(AlgebraBudgetError) as cm:
            po.lp([1, 1], A_ub=A, b_ub=b, bounds=bounds, uniqueness=False, budget=Budget(tensor=need - 1))
        self.assertEqual((cm.exception.resource, cm.exception.need), ("tensor", need))
        # the boxes of one enclosure call: output = boxes; the device bytes
        ctx = Context()
        x, = ctx.symbols("x")
        pr = po.Problem(x ** 2, {"x": (-1, 1)})
        po.boxes(pr, [[(0, 1)]] * 3, budget=Budget(output=3))
        with self.assertRaises(AlgebraBudgetError):
            po.boxes(pr, [[(0, 1)]] * 3, budget=Budget(output=2))
        with self.assertRaises(AlgebraBudgetError):
            po.boxes(pr, [[(0, 1)]] * 3, backend="cuda", budget=Budget(output=2))
        dev = 3 * 16 + 3 * 16 + 3 * 32 + 3 * 16 * 256
        po.boxes(pr, [[(0, 1)]] * 2, backend="cuda", budget=Budget(bytes=dev))
        with self.assertRaises(AlgebraBudgetError) as cm:
            po.boxes(pr, [[(0, 1)]] * 3, backend="cuda", budget=Budget(bytes=dev - 1))
        self.assertEqual(cm.exception.resource, "bytes")
        # an infeasibility certificate: output = leaves
        r = po.polynomial(x, {"x": (-1, 1)}, [(x * x, "<=", -1)])
        self.assertEqual(len(r.leaves), 1)
        with self.assertRaises(AlgebraBudgetError):
            po.polynomial(x, {"x": (-1, 1)}, [(x * x, "<=", -1)], budget=Budget(output=0))
        # representation limits: refused explicitly
        with self.assertRaises(UnsupportedError):
            po.Problem({(1, 0, 0, 0, 0): 1}, [(0, 1)] * 5)          # 5 variables > 4
        with self.assertRaises(UnsupportedError):
            po.Problem({(9,): 1}, [(0, 1)])                          # degree 9 > 8
        with self.assertRaises(InfeasibleError):
            po.Problem(x, {"x": (1, 0)})
        with self.assertRaises(TypeError):
            po.lp([1.5, 1], A_ub=A, b_ub=b)

    def test_opt002_active_set_iteration_cap_is_not_converged_not_wrong(self):
        # a tiny iteration budget stops the exact methods explicitly, never with a wrong optimum
        H = [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
        A = [[1, 1, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1]]
        with self.assertRaises((NotConvergedError, AlgebraBudgetError)):
            po.qp(H, [-4, -4, -4], A_ub=A, b_ub=[1, 0, 0, 0], budget=Budget(iterations=1))
        r = po.qp(H, [-4, -4, -4], A_ub=A, b_ub=[1, 0, 0, 0])
        self.assertEqual((r.status, r.point, r.value), ("optimal", [F(1, 3)] * 3, F(-11, 3)))

    def test_opt002_abi_registry_covers_the_polyopt_structs(self):
        from torbles import math_handles as mh
        cpu = {e["name"] for e in mh.abi_entries("cpu")}
        cuda = {e["name"] for e in mh.abi_entries("cuda")}
        self.assertLessEqual({"tr_polyopt_info", "tr_polyopt_spec", "tr_polyopt_encl_view", "tr_polyopt_box_out"}, cpu)
        self.assertLessEqual({"tr_polyopt_encl_view", "tr_polyopt_box_out"}, cuda)
        report = mh.abi_check()
        self.assertTrue(all(r["match"] for r in report["checked"]))


if __name__ == "__main__":
    unittest.main()
