# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""INV-001: linear and univariate polynomial inverse answers (torbles/math_inv.c, math_inv_gpu.cu, math_inv.py).

Oracles, all independent of the implementation and written here with Python integers and fractions.Fraction: exact
Gaussian elimination (ranks of A and [A | b], the unique solution), exact substitution of every returned point and
direction into the equations and the box, polynomials built from known factors (rational roots with multiplicities,
irreducible quadratics x^2 - c with irrational roots +-sqrt(c), compared through c exactly), exact evaluation of
polynomials and of their square-free parts at interval endpoints (sign changes), and brute-force exact evaluation of
the polynomial's terms on every grid point of a finite domain. DuckDB gives the moments of the RESP-001 response
function that is inverted. Every native result is also computed on the CUDA path (emulated here with scrambled
thread order, TORBLES_EMU_SCRAMBLE=1) and compared for equality. Budgets are tested at the limit and one past it.
Seeds are fixed.
"""
import ctypes as C
import os
import random
import unittest
from fractions import Fraction

import numpy as np

os.environ.setdefault("TORBLES_EMU_SCRAMBLE", "1")

from torbles import math_inv as mi, math_gpu
from torbles.math_budgets import Budget
from torbles.math_domains import DomainError
from torbles.math_errors import UnsupportedError
from torbles.mathematics import AlgebraBudgetError, Context

SEED = 20261031


# ------------------------------------------------------------------ the oracles
def rank(rows):
    """exact rank of a list of Fraction rows (Gaussian elimination written here)"""
    m = [list(r) for r in rows]
    r = 0
    ncol = len(m[0]) if m else 0
    for c in range(ncol):
        piv = next((i for i in range(r, len(m)) if m[i][c] != 0), None)
        if piv is None:
            continue
        m[r], m[piv] = m[piv], m[r]
        for i in range(len(m)):
            if i != r and m[i][c] != 0:
                f = m[i][c] / m[r][c]
                m[i] = [a - f * b for a, b in zip(m[i], m[r])]
        r += 1
    return r


def solve_unique(A, b):
    n = len(A[0])
    m = [list(map(Fraction, r)) + [Fraction(v)] for r, v in zip(A, b)]
    for c in range(n):
        piv = next(i for i in range(c, len(m)) if m[i][c] != 0)
        m[c], m[piv] = m[piv], m[c]
        m[c] = [v / m[c][c] for v in m[c]]
        for i in range(len(m)):
            if i != c and m[i][c] != 0:
                f = m[i][c]
                m[i] = [a - f * x for a, x in zip(m[i], m[c])]
    return [m[j][n] for j in range(n)]


def pmul(a, b):
    out = [Fraction(0)] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            out[i + j] += x * y
    return out


def peval(p, x):
    s = Fraction(0)
    for c in reversed(p):
        s = s * x + c
    return s


def sign(x):
    return (x > 0) - (x < 0)


def known_poly(rng, nlin=3, nquad=2, maxmult=3):
    """(coefficients, rational roots {r: mult}, irrational roots [(c, mult)] for x^2 - c (c > 0 non-square),
    square-free part): a product of known factors times a rational constant"""
    rats, quads = {}, []
    p = [Fraction(rng.choice([1, 2, 3, 5, -7]), rng.choice([1, 2, 3]))]
    sqfree = [Fraction(1)]
    while len(rats) < nlin:
        r = Fraction(rng.randint(-12, 12), rng.choice([1, 1, 2, 3, 5, 7]))
        if r in rats:
            continue
        rats[r] = rng.randint(1, maxmult)
        lin = [-r.numerator, r.denominator]          # (den x - num), integer
        for _ in range(rats[r]):
            p = pmul(p, lin)
        sqfree = pmul(sqfree, lin)
    used = set()
    while len(quads) < nquad:
        c = Fraction(rng.choice([2, 3, 5, 6, 7, 10, 11, 13]), rng.choice([1, 4, 9, 2, 3]))
        if c in used:
            continue
        num, den = c.numerator, c.denominator
        if round(num ** 0.5) ** 2 == num and round(den ** 0.5) ** 2 == den:
            continue
        used.add(c)
        mult = rng.randint(1, 2)
        quads.append((c, mult))
        q = [-c, Fraction(0), Fraction(1)]
        for _ in range(mult):
            p = pmul(p, q)
        sqfree = pmul(sqfree, q)
    if rng.random() < 0.5:                            # an irreducible quadratic without real roots
        p = pmul(p, [Fraction(rng.randint(1, 9)), Fraction(0), Fraction(1)])
    return p, rats, quads, sqfree


def irrational_in(c, positive, lo, hi):
    """whether +-sqrt(c) (c > 0 not a square) lies in [lo, hi] (None = unbounded), exactly"""
    if positive:
        above = lo is None or lo < 0 or lo * lo < c
        below = hi is not None and hi > 0 and hi * hi > c if hi is not None else True
    else:
        above = lo is None or (lo < 0 and lo * lo > c)
        below = hi is None or hi > 0 or hi * hi < c
    return above and below


def contains_irrational(root, c, positive):
    """lo < +-sqrt(c) < hi for an isolating interval, exactly"""
    return irrational_in(c, positive, root.lo, root.hi)


def terms_eval(expr, values):
    """the first-slice polynomial's own terms evaluated exactly here (independent of INV-001)"""
    names = expr.context.names
    s = Fraction(0)
    for exps, coef in expr.terms:
        t = Fraction(coef)
        for name, e in zip(names, exps):
            if e:
                t *= Fraction(values[name]) ** e
        s += t
    return s


class TestInv001Linear(unittest.TestCase):
    def check_point(self, A, b, lo, hi, x):
        for row, v in zip(A, b):
            self.assertEqual(sum(Fraction(a) * x[j] for j, a in enumerate(row)), Fraction(v))
        for j, v in enumerate(x):
            if lo is not None and lo[j] is not None:
                self.assertGreaterEqual(v, lo[j])
            if hi is not None and hi[j] is not None:
                self.assertLessEqual(v, hi[j])

    def test_inv001_linear_detects_unique_empty_and_whole_domain(self):
        # unique
        ans = mi.solve_affine([[2, 1], [1, -1]], [3, 0])
        self.assertEqual((ans.kind, ans.point), ("unique", {"x0": 1, "x1": 1}))
        # empty: inconsistent equations (rank A < rank [A | b], by the oracle)
        A, b = [[1, 1], [2, 2]], [1, 3]
        self.assertLess(rank([list(map(Fraction, r)) for r in A]),
                        rank([list(map(Fraction, r)) + [Fraction(v)] for r, v in zip(A, b)]))
        ans = mi.solve_affine(A, b)
        self.assertEqual((ans.kind, ans.reason, ans.point), ("empty", "inconsistent", None))
        # empty: the unique solution is outside the box
        ans = mi.solve_affine([[1, 0], [0, 1]], [3, Fraction(1, 2)], lo=[0, 0], hi=[2, 1])
        self.assertEqual((ans.kind, ans.reason), ("empty", "outside"))
        # empty: a family that misses the box (x + y = 5 cannot hold with 0 <= x, y <= 1)
        ans = mi.solve_affine([[1, 1]], [5], lo=[0, 0], hi=[1, 1])
        self.assertEqual((ans.kind, ans.reason), ("empty", "outside"))
        # whole domain: 0 = 0 (no division by zero), and an equation made trivial by a fixed coordinate
        ans = mi.solve_affine([[0, 0]], [0], lo=[0, 0], hi=[1, 1])
        self.assertEqual((ans.kind, ans.point), ("whole", None))
        ans = mi.solve_affine([[0, 0]], [1], lo=[0, 0], hi=[1, 1])
        self.assertEqual((ans.kind, ans.reason), ("empty", "inconsistent"))
        ans = mi.solve_affine([[2, 0]], [6], lo=[3, -1], hi=[3, 1])
        self.assertEqual(ans.kind, "whole")
        # a box that fixes the unknowns: no solution in it, but the equations alone have one (outside) or none
        ans = mi.solve_affine([[1]], [2], lo=[1], hi=[1])
        self.assertEqual((ans.kind, ans.reason), ("empty", "outside"))
        ans = mi.solve_affine([[1, 1], [1, 1]], [1, 2], lo=[0, 0], hi=[0, 0])
        self.assertEqual((ans.kind, ans.reason), ("empty", "inconsistent"))
        ans = mi.solve_affine([[1, 1], [1, -1]], [1, 0], lo=[0, None], hi=[0, None])   # x0 = 0 forces x1 = 1 and 0
        self.assertEqual((ans.kind, ans.reason), ("empty", "outside"))
        # one unknown a t = c: a = 0 decides between whole and empty, never dividing by zero
        for a, c, kind in ((0, 0, "whole"), (0, 5, "empty"), (4, 2, "unique")):
            self.assertEqual(mi.solve_affine([[a]], [c]).kind, kind)
        self.assertEqual(mi.solve_affine([[4]], [2]).point, {"x0": Fraction(1, 2)})
        # the box admits exactly one point of a family: unique (x + y = 2, x <= 1, y <= 1)
        ans = mi.solve_affine([[1, 1]], [2], lo=[None, None], hi=[1, 1])
        self.assertEqual((ans.kind, ans.point), ("unique", {"x0": 1, "x1": 1}))
        # a family: the simplex x + y + z = 1, x, y, z >= 0
        A, b, lo = [[1, 1, 1]], [1], [0, 0, 0]
        ans = mi.solve_affine(A, b, lo=lo)
        self.assertEqual((ans.kind, ans.dims, ans.rank), ("family", 2, 1))
        x = [ans.point[n] for n in ("x0", "x1", "x2")]
        self.check_point(A, b, lo, None, x)
        self.assertEqual(ans.ranges, [(0, 1), (0, 1)])
        for d in range(ans.dims):
            v = [ans.directions[d][n] for n in ("x0", "x1", "x2")]
            self.assertEqual(sum(v), 0)                        # A v = 0
            self.assertEqual(ans.directions[d][ans.free[d]], 1)

    def test_inv001_linear_random_systems_against_exact_elimination(self):
        rng = random.Random(SEED)
        kinds = set()
        for trial in range(120):
            n, m = rng.randint(1, 5), rng.randint(0, 5)
            r = rng.randint(0, min(m, n))
            basis = [[Fraction(rng.randint(-5, 5), rng.choice([1, 2, 3])) for _ in range(n)] for _ in range(r)]
            A = []
            for _ in range(m):                                  # rows in the span of `basis`: rank <= r
                coef = [rng.randint(-2, 2) for _ in range(r)]
                A.append([sum(c * row[j] for c, row in zip(coef, basis)) for j in range(n)])
            x_true = [Fraction(rng.randint(-6, 6), rng.choice([1, 2])) for _ in range(n)]
            b = [sum(a * xv for a, xv in zip(row, x_true)) for row in A]
            if m and rng.random() < 0.3:
                b[rng.randrange(m)] += 1                        # perhaps inconsistent
            ans = mi.solve_affine(A, b, names=[f"x{j}" for j in range(n)]) if m else \
                mi.solve_affine([], [], names=[f"x{j}" for j in range(n)])
            kinds.add(ans.kind)
            rA = rank(A) if m else 0
            rAb = rank([row + [v] for row, v in zip(A, b)]) if m else 0
            if rA < rAb:
                self.assertEqual((ans.kind, ans.reason), ("empty", "inconsistent"), trial)
                continue
            self.assertEqual(ans.rank, rA, trial)
            if rA == 0:
                self.assertEqual(ans.kind, "whole", trial)
            elif rA == n:
                self.assertEqual(ans.kind, "unique", trial)
                sel = []
                for row, v in zip(A, b):                        # the unique solution from independent rows
                    if rank([s[0] for s in sel] + [row]) > len(sel):
                        sel.append((row, v))
                self.assertEqual([ans.point[f"x{j}"] for j in range(n)], solve_unique([s[0] for s in sel],
                                                                                      [s[1] for s in sel]))
            else:
                self.assertEqual((ans.kind, ans.dims), ("family", n - rA), trial)
                x0 = [ans.point[f"x{j}"] for j in range(n)]
                self.check_point(A, b, None, None, x0)
                for d in range(ans.dims):
                    v = [ans.directions[d][f"x{j}"] for j in range(n)]
                    self.assertTrue(all(sum(a * vj for a, vj in zip(row, v)) == 0 for row in A))
                    self.assertEqual(ans.ranges[d], (None, None))
                dirs = [[ans.directions[d][f"x{j}"] for j in range(n)] for d in range(ans.dims)]
                self.assertEqual(rank(dirs), ans.dims)              # independent directions
        self.assertEqual(kinds, {"unique", "empty", "whole", "family"})

    def test_inv001_linear_box_families_exact_ranges(self):
        rng = random.Random(SEED + 1)
        seen = set()
        for trial in range(120):
            # one equation a0 x0 + a1 x1 = c in a box: one free parameter, its exact range by intervals here
            a = [Fraction(rng.randint(-4, 4)) for _ in range(2)]
            if a == [0, 0]:
                a[rng.randrange(2)] = Fraction(rng.choice([-3, 2, 5]))
            c = Fraction(rng.randint(-6, 6), rng.choice([1, 2]))
            lo = [Fraction(rng.randint(-3, 0)), Fraction(rng.randint(-3, 0))]
            hi = [lo[0] + rng.randint(0, 4), lo[1] + rng.randint(0, 4)]
            ans = mi.solve_affine([a], [c], lo=lo, hi=hi)
            seen.add(ans.kind)
            # Gauss-Jordan pivots on the first nonzero coefficient; the other unknown is the free one
            p, f = (0, 1) if a[0] != 0 else (1, 0)
            if lo[p] == hi[p]:                                  # the pivot is fixed by the box
                xp = lo[p]
                if a[f] == 0:
                    want = "whole" if a[p] * xp == c else "empty"
                    self.assertEqual(ans.kind, want, trial)
                    continue
                xf = (c - a[p] * xp) / a[f]
                ok = lo[f] <= xf <= hi[f]
                self.assertEqual(ans.kind, "unique" if ok else "empty", trial)
                if ok:
                    self.assertEqual(ans.point, {"x0": xp if p == 0 else xf, "x1": xp if p == 1 else xf})
                continue
            k, r = -a[f] / a[p], c / a[p]                      # x_p = k x_f + r
            L, H = lo[f], hi[f]
            for bound, upper in ((lo[p], False), (hi[p], True)):
                if k == 0:
                    if (r > bound) if upper else (r < bound):
                        L, H = Fraction(1), Fraction(0)
                    continue
                t = (bound - r) / k
                if (k > 0) == upper:
                    H = min(H, t)
                else:
                    L = max(L, t)
            if lo[f] == hi[f]:                                  # the free unknown is fixed by the box
                want = "unique" if L <= H else "empty"
                self.assertEqual(ans.kind, want, trial)
                continue
            if L > H:
                self.assertEqual((ans.kind, ans.reason), ("empty", "outside"), trial)
            elif L == H:
                self.assertEqual(ans.kind, "unique", trial)
                self.assertEqual(ans.point[f"x{f}"], L)
                self.check_point([a], [c], lo, hi, [ans.point["x0"], ans.point["x1"]])
            else:
                self.assertEqual(ans.kind, "family", trial)
                x0 = [ans.point["x0"], ans.point["x1"]]
                self.check_point([a], [c], lo, hi, x0)
                self.assertEqual(ans.free, [f"x{f}"])
                rlo, rhi = ans.ranges[0]
                self.assertEqual((x0[f] + rlo, x0[f] + rhi), (L, H), trial)
        self.assertLessEqual({"unique", "empty", "family"}, seen)

    def test_inv001_linear_targets_of_first_slice_and_resp001_functions(self):
        ctx = Context()
        t, u, w = ctx.symbols("t", "u", "w")
        ans = mi.solve([(2 * t + u, 3), (t - u, 0)])
        self.assertEqual((ans.kind, ans.point), ("unique", {"t": 1, "u": 1}))
        ans = mi.solve([(t + u + w, 1)], domain={"t": (0, None), "u": (0, None), "w": (0, None)})
        self.assertEqual((ans.kind, ans.dims), ("family", 2))
        self.assertEqual(ans.free, ["u", "w"])
        ans = mi.solve([(t - t, 0)], unknowns=["t"], domain={"t": (0, 1)})
        self.assertEqual(ans.kind, "whole")
        with self.assertRaises(UnsupportedError):
            mi.solve([(t * t, 1)])                              # not a linear target
        with self.assertRaises(DomainError):
            mi.solve([(t + u, 1)], unknowns=["t"])              # u is neither an unknown nor bound
        # RESP-001: the squared loss of y ~ theta x over selected rows; its gradient is linear in theta and its zero
        # is the least-squares theta = sum(xy) / sum(x^2), from DuckDB's exact sums
        import pandas as pd
        import torbles
        from torbles.mathematics import MomentBasis
        rng = np.random.default_rng(SEED)
        xs = [int(v) for v in rng.integers(-9, 10, 40)]
        ys = [int(v) for v in rng.integers(-20, 21, 40)]
        keep = [int(v) for v in rng.integers(0, 2, 40)]
        db = torbles.from_frames({"r": pd.DataFrame({"x": xs, "y": ys, "keep": keep})})
        tab = db["r"]
        import duckdb
        con = duckdb.connect()
        con.register("r", pd.DataFrame({"x": xs, "y": ys, "keep": keep}))
        sxy, sxx = con.execute("SELECT SUM(x*y)::HUGEINT, SUM(x*x)::HUGEINT FROM r WHERE keep = 1").fetchone()
        with MomentBasis(db.query(tab).where(tab.keep == 1), ["x", "y"]) as basis:
            x, y = basis.variables["x"], basis.variables["y"]
            theta = basis.parameter("theta")
            loss = basis.reduce((y - theta * x) ** 2)
            grad = loss.diff("theta")
            ans = mi.solve([(grad, 0)])
            self.assertEqual(ans.kind, "unique")
            self.assertEqual(ans.point["theta"], Fraction(int(sxy), int(sxx)))
            # the loss as a univariate target: loss(theta) = loss(theta*) + 1 has exactly two real roots
            best = loss.evaluate(theta=ans.point["theta"])
            r = mi.roots(loss, best + 1)
            self.assertEqual(len(r.roots), 2)
            for root in r.roots:
                self.assertTrue(root.exact or sign(terms_eval(loss, {"theta": root.lo}) - best - 1)
                                != sign(terms_eval(loss, {"theta": root.hi}) - best - 1))
            r = mi.roots(loss, best)                            # the minimum: one double root, exact
            self.assertEqual([(x.exact, x.lo, x.multiplicity) for x in r.roots], [(True, ans.point["theta"], 2)])


class TestInv001Roots(unittest.TestCase):
    def check_roots(self, coef, rats, quads, sqfree, lo, hi, res, width=None):
        """every known root in [lo, hi] is in exactly one returned item, nothing else is returned, exact items are
        exact roots, interval items contain an irrational root with a certified sign change"""
        expected = [("q", r, m) for r, m in rats.items() if (lo is None or r >= lo) and (hi is None or r <= hi)]
        for c, m in quads:
            for positive in (True, False):
                if irrational_in(c, positive, lo, hi):
                    expected.append(("i", (c, positive), m))
        self.assertEqual(len(res.roots), len(expected), (coef, lo, hi, res.roots))
        used = set()
        for root in res.roots:
            if root.exact:
                self.assertEqual(peval(coef, root.value), 0)
                hit = [i for i, e in enumerate(expected) if e[0] == "q" and e[1] == root.value]
            else:
                self.assertLess(root.lo, root.hi)
                self.assertNotEqual(peval(sqfree, root.lo), 0)
                self.assertEqual(sign(peval(sqfree, root.lo)), -sign(peval(sqfree, root.hi)))   # sign change
                fac = res.factors[root.multiplicity - 1]
                self.assertEqual(sign(peval(fac, root.lo)), root.sign_lo)
                self.assertEqual(sign(peval(fac, root.hi)), -root.sign_lo)
                hit = [i for i, e in enumerate(expected) if e[0] == "i" and contains_irrational(root, *e[1])]
                # no known rational root inside an isolating interval: the item is not a disguised rational root
                self.assertFalse(any(root.lo < r < root.hi for r in rats))
                if width is not None:
                    self.assertLessEqual(root.width, width)
            self.assertEqual(len(hit), 1, (root, expected))
            self.assertNotIn(hit[0], used)
            used.add(hit[0])
            self.assertEqual(root.multiplicity, expected[hit[0]][2])
            if root.boundary:
                self.assertIn(root.value, [lo, hi])
        # the square-free factorization: prod factor_i^(i+1) is a constant multiple of f - target
        prod = [Fraction(1)]
        for i, fac in enumerate(res.factors):
            for _ in range(i + 1):
                prod = pmul(prod, [Fraction(c) for c in fac])
        ratio = coef[-1] / prod[-1]
        self.assertEqual([ratio * c for c in prod], coef)

    def test_inv001_root_isolation_returns_all_roots_in_declared_bounds(self):
        rng = random.Random(SEED + 2)
        for trial in range(40):
            coef, rats, quads, sqfree = known_poly(rng, nlin=rng.randint(0, 3), nquad=rng.randint(0, 2),
                                                   maxmult=rng.randint(1, 3))
            if len(coef) < 2:
                continue
            target = Fraction(rng.randint(-5, 5), rng.choice([1, 3]))
            f = list(coef)
            f[0] += target                                       # f - target = the known polynomial
            choice = rng.random()
            if choice < 0.25:
                lo, hi = None, None
            elif choice < 0.6 and rats:                          # bounds exactly at known rational roots
                rs = sorted(rats)
                lo, hi = rs[0], rs[-1]
            else:
                lo = Fraction(rng.randint(-8, 2), rng.choice([1, 2]))
                hi = lo + Fraction(rng.randint(0, 12), rng.choice([1, 3]))
            res = mi.roots(f, target, lo, hi)
            self.assertEqual(res.kind, "finite" if res.roots else "empty")
            self.check_roots(coef, rats, quads, sqfree, lo, hi, res)
            if rats and lo is not None and lo == min(rats):
                self.assertEqual(res.roots[0].boundary in ("lo", "lo and hi"), True)
            width = Fraction(1, 2 ** rng.randint(10, 60))
            refined = res.refine(width)
            self.check_roots(coef, rats, quads, sqfree, lo, hi, refined, width)
            gpu = res.refine(width, backend="cuda")             # the CUDA sign evaluations: the same intervals
            self.assertEqual(gpu.backend, "cuda-emulated" if math_gpu.device()["emulated"] else "cuda")
            self.assertEqual([(r.exact, r.lo, r.hi, r.multiplicity) for r in gpu.roots],
                             [(r.exact, r.lo, r.hi, r.multiplicity) for r in refined.roots])
            self.assertEqual(gpu.rounds, refined.rounds)

    def test_inv001_never_labels_a_numerical_candidate_exact(self):
        # irrational roots within 1e-40 of simple rationals: floating evaluation cannot tell them apart
        eps = Fraction(1, 10 ** 40)
        for c in (Fraction(1, 4) + eps, Fraction(1, 9) - eps, Fraction(4) + eps):
            res = mi.roots([-c, 0, 1], width=Fraction(1, 10 ** 45))
            self.assertEqual(len(res.roots), 2)
            for root in res.roots:
                self.assertFalse(root.exact)
                self.assertIsNone(root.value)
                self.assertTrue(contains_irrational(root, c, root.lo > 0))
                self.assertLess(root.width, Fraction(1, 10 ** 45))
                self.assertEqual(root.how, "irrational")
                near = Fraction(1, 2) if c < 1 and c > Fraction(1, 5) else (Fraction(1, 3) if c < 1 else Fraction(2))
                self.assertFalse(root.lo <= near * (1 if root.lo > 0 else -1) <= root.hi)
        # rational roots inside one square-free factor of degree 4 are found exactly by the candidate test
        f = pmul(pmul([-89, 97], [-7, 101]), [-3, 0, 1])           # (97x - 89)(101x - 7)(x^2 - 3)
        res = mi.roots(f)
        self.assertEqual(sorted(r.value for r in res.roots if r.exact), [Fraction(7, 101), Fraction(89, 97)])
        self.assertEqual({r.how for r in res.roots if r.exact}, {"rational candidate"})
        self.assertEqual(sum(1 for r in res.roots if not r.exact), 2)
        # a cubic with one real irrational root: x^3 - 2 (the candidate 1 fails, 5/4 has a too large denominator)
        res = mi.roots([-2, 0, 0, 1])
        self.assertEqual(len(res.roots), 1)
        self.assertFalse(res.roots[0].exact)
        self.assertLess(res.roots[0].lo ** 3, 2)
        self.assertGreater(res.roots[0].hi ** 3, 2)
        # multiplicities and boundary roots: (x - 1)^2 (x - 3)^3 (x - 1/3) on [1, 3]
        f = pmul(pmul(pmul([-1, 1], [-1, 1]), pmul(pmul([-3, 1], [-3, 1]), [-3, 1])), [-1, 3])
        res = mi.roots(f, 0, 1, 3)
        self.assertEqual([(r.value, r.multiplicity, r.boundary) for r in res.roots],
                         [(1, 2, "lo"), (3, 3, "hi")])
        res = mi.roots(f, 0, 0, 3)
        self.assertEqual([(r.value, r.multiplicity) for r in res.roots], [(Fraction(1, 3), 1), (1, 2), (3, 3)])
        # degenerate targets: the zero polynomial (every point), a nonzero constant (none), a point domain
        self.assertEqual(mi.roots([5, 0, 0], 5, 0, 1).kind, "whole")
        res = mi.roots([5], 4)
        self.assertEqual((res.kind, res.reason), ("empty", "constant"))
        self.assertEqual([r.value for r in mi.roots([-4, 0, 1], 0, 2, 2).roots], [2])
        self.assertEqual(mi.roots([-4, 0, 1], 0, 3, 3).kind, "empty")
        with self.assertRaises(ValueError):
            mi.roots([-4, 0, 1], 0, 3, 2)                          # lo > hi

    def test_inv001_roots_of_first_slice_polynomials(self):
        ctx = Context()
        t, u = ctx.symbols("t", "u")
        f = t ** 3 - 2 * t
        res = mi.roots(f, 0, -2, 2, width=Fraction(1, 2 ** 30))
        self.assertEqual([r.exact for r in res.roots], [False, True, False])
        self.assertEqual(res.roots[1].value, 0)
        for r in (res.roots[0], res.roots[2]):
            self.assertTrue(contains_irrational(r, Fraction(2), r.lo > 0))
        with self.assertRaises(ValueError):
            mi.roots(t * u, 1)                                     # two variables: name the one to solve for
        with self.assertRaises(DomainError):
            mi.roots(t * u, 1, variable="t")
        g = Fraction(1, 3) * u ** 2 - Fraction(4, 3)
        self.assertEqual([r.value for r in mi.roots(g, 0).roots], [-2, 2])


class TestInv001Signs(unittest.TestCase):
    def test_inv001_batched_exact_signs_cpu_equals_cuda_and_oracle(self):
        rng = random.Random(SEED + 3)
        polys = []
        for _ in range(6):
            d = rng.randint(0, 9)
            polys.append([Fraction(rng.randint(-10 ** 6, 10 ** 6), rng.randint(1, 1000)) for _ in range(d + 1)])
        polys.append([Fraction(0)])                                # the zero polynomial: sign 0 everywhere
        polys.append([Fraction(-2), 0, 1])
        pts, which = [], []
        for i in range(600):
            k = rng.randrange(len(polys))
            bits = rng.choice([4, 30, 64, 200, 700])
            pts.append(Fraction(rng.randint(-2 ** bits, 2 ** bits), rng.randint(1, 2 ** bits)))
            which.append(k)
        pts += [Fraction(0), Fraction(1)]
        which += [len(polys) - 1, len(polys) - 2]
        cpu = mi.signs(polys, pts, which)
        gpu = mi.signs(polys, pts, which, backend="cuda")
        self.assertEqual(cpu.values, gpu.values)
        self.assertEqual(gpu.backend, "cuda-emulated" if math_gpu.device()["emulated"] else "cuda")
        self.assertEqual(cpu.values, [sign(peval(polys[k], x)) for k, x in zip(which, pts)])
        self.assertGreater(cpu.primes, 1)
        # exact zeros at rational roots, among huge points
        f = pmul([-7, 3], [-5, 0, 1])                              # roots 7/3 (rational), +-sqrt(5)
        xs = [Fraction(7, 3), Fraction(7, 3) + Fraction(1, 10 ** 60), Fraction(7, 3) - Fraction(1, 10 ** 60)]
        self.assertEqual(mi.signs([f], xs).values, [0, 1, -1])
        self.assertEqual(mi.signs([f], xs, backend="cuda").values, [0, 1, -1])

    def test_inv001_sign_moduli_and_budget_limits(self):
        # the multimodular bound: B = bits(sum |a|) + d bits(point); 61 k >= B + 2 with k <= 128 primes
        x1 = [1, 1]                                                # x + 1: bits(sum |a|) = 2, degree 1
        ok = mi.signs([x1], [Fraction(2 ** 7803)])                 # B = 2 + 7804 = 7806 -> k = 128
        self.assertEqual((ok.values, ok.primes), ([1], 128))
        self.assertEqual(mi.signs([x1], [Fraction(2 ** 7803)], backend="cuda").values, [1])
        with self.assertRaises(OverflowError):
            mi.signs([x1], [Fraction(2 ** 7804)])                  # B = 7807 -> k = 129 > 128
        # output budget (points) at the limit and one past it
        pts = [Fraction(i) for i in range(10)]
        self.assertEqual(len(mi.signs([x1], pts, budget=Budget(output=10)).values), 10)
        with self.assertRaises(AlgebraBudgetError) as e:
            mi.signs([x1], pts, budget=Budget(output=9))
        self.assertEqual((e.exception.resource, e.exception.need, e.exception.limit), ("output", 10, 9))
        # the CUDA entry point budgets its device copies: exactly the need passes, one byte less is refused
        pts = [Fraction(i, 7) for i in range(1000)]
        need = mi.signs_device_bytes([x1], pts)
        self.assertEqual(len(mi.signs([x1], pts, backend="cuda", budget=Budget(bytes=need)).values), 1000)
        with self.assertRaises(AlgebraBudgetError) as e:
            mi.signs([x1], pts, backend="cuda", budget=Budget(bytes=need - 1))
        self.assertEqual((e.exception.resource, e.exception.need), ("bytes", need))


class TestInv001Sweep(unittest.TestCase):
    def oracle(self, f, target, grid):
        names = list(grid)
        sizes = [len(grid[n]) for n in names]
        hits = []
        total = int(np.prod(sizes))
        for idx in range(total):
            rem, coords = idx, []
            for s in reversed(sizes):
                coords.append(rem % s)
                rem //= s
            coords.reverse()
            vals = {n: Fraction(grid[n][c]) for n, c in zip(names, coords)}
            if terms_eval(f, vals) == target:
                hits.append(idx)
        return hits

    def test_inv001_finite_domain_sweep_finds_every_exact_match(self):
        rng = random.Random(SEED + 4)
        ctx = Context()
        t, u, w = ctx.symbols("t", "u", "w")
        cases = [
            (t * t - 2 * t, {"t": [Fraction(k, 2) for k in range(-16, 16)]}, 0),
            (t * u - 6, {"t": list(range(-8, 8)), "u": [Fraction(k, 3) for k in range(-8, 8)]}, 0),
            (t + u + w, {"t": list(range(8)), "u": list(range(8)), "w": list(range(8))}, 7),
            (Fraction(1, 3) * t ** 3 + u ** 2 - w, {"t": [Fraction(k, 4) for k in range(-4, 12)],
                                                     "u": list(range(-4, 4)), "w": list(range(-8, 8))}, 1),
            (t ** 2 + u ** 2 - 25, {"t": list(range(-16, 16)), "u": list(range(-16, 16))}, 0),
            (t - t, {"t": list(range(8))}, 0),                    # the zero polynomial: every point
            (t ** 8 - 2, {"t": [Fraction(k, 7) for k in range(-16, 16)]}, 0),   # no rational solution
        ]
        for size in (8, 16, 32):                                   # 8/16/32 values per coordinate
            vals = [Fraction(rng.randint(-20, 20), rng.choice([1, 2, 4])) for _ in range(size)]
            cases.append((t * t * u - 2 * u + Fraction(1, 2), {"t": vals, "u": vals}, 0))
        for f, grid, target in cases:
            want = self.oracle(f, Fraction(target), grid)
            cpu = mi.sweep(f, grid, target)
            gpu = mi.sweep(f, grid, target, backend="cuda")
            self.assertTrue(cpu.complete)
            self.assertEqual(cpu.indices, want, (f, grid))
            self.assertEqual((gpu.indices, gpu.matches, gpu.complete), (cpu.indices, cpu.matches, True))
            self.assertTrue(np.array_equal(gpu.bitmap_words, cpu.bitmap_words))
            self.assertEqual(gpu.backend, "cuda-emulated" if math_gpu.device()["emulated"] else "cuda")
            for m in cpu.matches:
                self.assertEqual(terms_eval(f, m), target)
            self.assertGreaterEqual(cpu.primes, 1)
            self.assertLessEqual(cpu.bound_bits, 61 * cpu.primes)

    def test_inv001_incomplete_search_is_explicit_and_resumable(self):
        ctx = Context()
        t, u = ctx.symbols("t", "u")
        f, grid = t * t + u - 5, {"t": list(range(-16, 16)), "u": list(range(32))}
        full = mi.sweep(f, grid)
        self.assertTrue(full.complete)
        # an iteration budget truncates the search: incomplete, the examined prefix and where to resume
        for limit in (1, 100, 513, 1023, 1024):
            part = mi.sweep(f, grid, budget=Budget(iterations=limit))
            self.assertEqual(part.searched, min(limit, 1024))
            self.assertEqual(part.complete, limit >= 1024)
            self.assertEqual(part.reason, "complete" if limit >= 1024 else "iterations")
            self.assertEqual(part.indices, [i for i in full.indices if i < part.next])
            gpu = mi.sweep(f, grid, budget=Budget(iterations=limit), backend="cuda")
            self.assertEqual((gpu.indices, gpu.next, gpu.complete), (part.indices, part.next, part.complete))
        found, start, steps = [], 0, 0
        while True:                                                 # resuming from next covers the grid exactly
            part = mi.sweep(f, grid, start=start, budget=Budget(iterations=200))
            found += part.indices
            steps += 1
            if part.complete:
                break
            start = part.next
        self.assertEqual((found, steps), (full.indices, 6))
        # the output budget: the search stops before the first match it cannot list
        n = len(full.indices)
        part = mi.sweep(f, grid, budget=Budget(output=n))
        self.assertTrue(part.complete)
        part = mi.sweep(f, grid, budget=Budget(output=n - 1))
        self.assertEqual((part.complete, part.reason, part.next), (False, "output", full.indices[n - 1]))
        self.assertEqual(part.indices, full.indices[:n - 1])
        gpu = mi.sweep(f, grid, budget=Budget(output=n - 1), backend="cuda")
        self.assertEqual((gpu.indices, gpu.next, gpu.reason), (part.indices, part.next, "output"))
        part = mi.sweep(f, grid, budget=Budget(output=0))
        self.assertEqual((part.indices, part.next, part.searched), ([], full.indices[0], full.indices[0]))

    def test_inv001_sweep_scope_and_budget_limits(self):
        ctx = Context()
        t, u, v, w = ctx.symbols("t", "u", "v", "w")
        grid3 = {"t": range(4), "u": range(4), "v": range(4)}
        self.assertTrue(mi.sweep(t + u + v, grid3, budget=Budget(variables=3)).complete)
        with self.assertRaises(AlgebraBudgetError) as e:
            mi.sweep(t + u + v, grid3, budget=Budget(variables=2))
        self.assertEqual((e.exception.resource, e.exception.need, e.exception.limit), ("variables", 3, 2))
        with self.assertRaises(UnsupportedError):
            mi.sweep(t + u + v + w, {**grid3, "w": range(2)})      # at most three searched coordinates
        with self.assertRaises(DomainError):
            mi.sweep(t + u, {"t": range(4)})                        # u is not searched
        f = t * t + u                                               # 2 terms + the target's constant: support 3
        self.assertTrue(mi.sweep(f, {"t": range(4), "u": range(4)}, 3, budget=Budget(support=3)).complete)
        with self.assertRaises(AlgebraBudgetError) as e:
            mi.sweep(f, {"t": range(4), "u": range(4)}, 3, budget=Budget(support=2))
        self.assertEqual(e.exception.resource, "support")
        # the device copies are budgeted by the CUDA entry point itself: at the need it runs, one byte less refuses
        plan_args = (f, {"t": range(4), "u": range(4)}, 3)
        need = mi.sweep_device_bytes(*plan_args)
        api, gpu = mi._api(), mi._gpu()
        error, refusal = C.c_int32(), mi._Refusal()
        flat = mi._Values([Fraction(i) for i in range(4)] * 2)
        sizes = (C.c_uint64 * 2)(4, 4)
        idx = (C.c_int32 * 2)(0, 1)
        tq, keep = mi._one(3, "target")
        plan = api["sweep_prepare"](f._open(), tq, 2, idx, sizes, flat.array, 0, None, C.byref(refusal),
                                    C.byref(error))
        self.assertTrue(plan)
        try:
            view = mi._SweepView()
            api["sweep_get_view"](plan, C.byref(view))
            bits = np.zeros(1, dtype=np.uint64)
            run = gpu.lib().tr_mgpu_inv_sweep
            self.assertEqual(run(C.byref(view), bits.ctypes.data, C.byref(Budget(bytes=need).raw),
                                 C.byref(refusal)), 0)
            self.assertEqual(run(C.byref(view), bits.ctypes.data, C.byref(Budget(bytes=need - 1).raw),
                                 C.byref(refusal)), -5)
            self.assertEqual((refusal.resource, refusal.need, refusal.limit), (8, need, need - 1))
        finally:
            api["sweep_plan_free"](plan)


class TestInv001Budgets(unittest.TestCase):
    def test_inv001_budgets_at_the_limit_and_one_past(self):
        # univariate degree
        f = [Fraction(-1), 0, 0, 0, 1]                              # x^4 - 1
        self.assertEqual(len(mi.roots(f, budget=Budget(degree=4)).roots), 2)
        with self.assertRaises(AlgebraBudgetError) as e:
            mi.roots(f, budget=Budget(degree=3))
        self.assertEqual((e.exception.resource, e.exception.need, e.exception.limit), ("degree", 4, 3))
        # iterations of the isolation: the count of a run is exactly the limit it needs
        f = pmul(pmul([-89, 97], [-7, 101]), [-3, 0, 1])
        used = mi.roots(f, 0, -10, 10).iterations
        self.assertEqual(len(mi.roots(f, 0, -10, 10, budget=Budget(iterations=used)).roots), 4)
        with self.assertRaises(AlgebraBudgetError) as e:
            mi.roots(f, 0, -10, 10, budget=Budget(iterations=used - 1))
        self.assertEqual((e.exception.resource, e.exception.need), ("iterations", used))
        # output: the roots returned
        self.assertEqual(len(mi.roots(f, budget=Budget(output=4)).roots), 4)
        with self.assertRaises(AlgebraBudgetError) as e:
            mi.roots(f, budget=Budget(output=3))
        self.assertEqual(e.exception.resource, "output")
        # refinement iterations: rounds x roots refined
        r = mi.roots([-2, 0, 1])
        fine = r.refine(Fraction(1, 2 ** 20))
        need = fine.iterations - r.iterations
        self.assertEqual(r.refine(Fraction(1, 2 ** 20), budget=Budget(iterations=need)).roots[0].hi,
                         fine.roots[0].hi)
        with self.assertRaises(AlgebraBudgetError):
            r.refine(Fraction(1, 2 ** 20), budget=Budget(iterations=need - 1))
        # linear: unknowns (variables) and the Fourier-Motzkin work (iterations, support)
        A, b, lo = [[1, 1, 1, 1]], [1], [0, 0, 0, 0]
        self.assertEqual(mi.solve_affine(A, b, lo=lo, budget=Budget(variables=4)).kind, "family")
        with self.assertRaises(AlgebraBudgetError) as e:
            mi.solve_affine(A, b, lo=lo, budget=Budget(variables=3))
        self.assertEqual((e.exception.resource, e.exception.need), ("variables", 4))
        ans = mi.solve_affine(A, b, lo=lo)
        self.assertEqual(ans.ranges, [(0, 1)] * 3)
        used = ans.iterations
        self.assertEqual(mi.solve_affine(A, b, lo=lo, budget=Budget(iterations=used)).ranges, ans.ranges)
        with self.assertRaises(AlgebraBudgetError) as e:
            mi.solve_affine(A, b, lo=lo, budget=Budget(iterations=used - 1))
        self.assertEqual(e.exception.resource, "iterations")
        with self.assertRaises(AlgebraBudgetError) as e:
            mi.solve_affine([[1, 1]], [1], budget=Budget(tensor=2))   # m (n + 1) = 3
        self.assertEqual((e.exception.resource, e.exception.need), ("tensor", 3))
        self.assertEqual(mi.solve_affine([[1, 1]], [1], budget=Budget(tensor=3)).kind, "family")

    def test_inv001_moduli_table_is_verified_natively(self):
        table, ok = mi.primes()
        self.assertTrue(ok)
        self.assertEqual(len(set(table)), 128)
        self.assertEqual(table, sorted(table, reverse=True))
        for q in table:                                             # trial division and Fermat bases written here
            self.assertTrue((1 << 61) < q < (1 << 62))
            self.assertTrue(all(q % d for d in range(2, 2000)))
            self.assertTrue(all(pow(b, q - 1, q) == 1 for b in (2, 3, 5, 7, 11, 13)))

    def test_inv001_abi_registry_covers_the_inv_structs(self):
        from torbles import math_handles as mh
        report = mh.abi_check()
        checked = {(r["native"], r["library"]) for r in report["checked"]}
        for name in ("tr_inv_linear_info", "tr_inv_roots_info", "tr_inv_root_info", "tr_inv_signs_view",
                     "tr_inv_sweep_view", "tr_inv_sweep_info"):
            self.assertIn((name, "libalg"), checked)
        for name in ("tr_inv_signs_view", "tr_inv_sweep_view"):
            self.assertIn((name, "cuda"), checked)


if __name__ == "__main__":
    unittest.main()
