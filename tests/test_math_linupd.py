# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""LIN-003: leave-group-out and low-rank model updates (torbles/math_linupd.c, math_linupd_gpu.cu, math_linupd.py).

Oracles, independent of the implementation and written here: the normal equations of the remaining rows built with
Python Fractions directly from the data (value units: logical / 10^scale), solved by a Fraction Gauss-Jordan
elimination into the canonical solution set (free unknowns 0, one nullspace vector per free column); rss, tss, r2,
sigma2 summed over the remaining rows themselves; update denominators det(I + V^T A^-1 U) from a Fraction inverse and
a Fraction determinant. Complete reconstruction: LIN-001 moments built again from a scope whose selection drops the
rows, then LIN-002's fit (a different code path that never subtracts). The CUDA path (emulated here with
TORBLES_EMU_SCRAMBLE=1) solves every (group, prime) on the device and must give the same canonical models. Budgets are
tested at the limit and one past it. Seeds are fixed.
"""
import os
import unittest
from fractions import Fraction

import numpy as np

os.environ.setdefault("TORBLES_EMU_SCRAMBLE", "1")

from torbles import math_bank as mbk, math_lin as lin, math_linupd as lu
from torbles.math_budgets import Budget
from torbles.math_domains import DomainError
from torbles.math_errors import NotConvergedError, SingularError, UnsupportedError
from torbles.mathematics import AlgebraBudgetError

SEED = 20261102


# ------------------------------------------------------------------ the oracle
def gauss_jordan(A, b):
    """the canonical solution set of A x = b over Fractions: (kind, x, nullspace)"""
    n = len(A)
    m = [[Fraction(v) for v in row] + [Fraction(bi)] for row, bi in zip(A, b)]
    piv, r = [], 0
    for c in range(n):
        p = next((i for i in range(r, n) if m[i][c] != 0), None)
        if p is None:
            continue
        m[r], m[p] = m[p], m[r]
        pv = m[r][c]
        m[r] = [v / pv for v in m[r]]
        for i in range(n):
            if i != r and m[i][c] != 0:
                f = m[i][c]
                m[i] = [a - f * c2 for a, c2 in zip(m[i], m[r])]
        piv.append(c)
        r += 1
    if any(m[i][n] != 0 for i in range(r, n)):
        return "none", None, []
    x = [Fraction(0)] * n
    for t, c in enumerate(piv):
        x[c] = m[t][n]
    null = []
    for j in range(n):
        if j in piv:
            continue
        v = [Fraction(0)] * n
        v[j] = Fraction(1)
        for t, c in enumerate(piv):
            v[c] = -m[t][j]
        null.append(v)
    return ("unique" if r == n else "family"), x, null


def inverse(A):
    n = len(A)
    cols = [gauss_jordan(A, [Fraction(int(i == j)) for i in range(n)])[1] for j in range(n)]
    return [[cols[j][i] for j in range(n)] for i in range(n)]


def det(A):
    m = [[Fraction(v) for v in row] for row in A]
    n, d = len(m), Fraction(1)
    for c in range(n):
        p = next((i for i in range(c, n) if m[i][c] != 0), None)
        if p is None:
            return Fraction(0)
        if p != c:
            m[c], m[p] = m[p], m[c]
            d = -d
        d *= m[c][c]
        for i in range(c + 1, n):
            f = m[i][c] / m[c][c]
            m[i] = [a - f * b for a, b in zip(m[i], m[c])]
    return d


def normal_equations(rows, intercept, ridge):
    """A, A0, b of rows [(features (Fractions), y)] with the design [1?] + features"""
    Z = [([Fraction(1)] if intercept else []) + list(x) for x, _ in rows]
    n = len(Z[0]) if Z else (int(intercept) + 0)
    A0 = [[sum((z[u] * z[w] for z in Z), Fraction(0)) for w in range(n)] for u in range(n)]
    A = [[A0[u][w] + (ridge if u == w and (u > 0 or not intercept) else 0) for w in range(n)] for u in range(n)]
    b = [sum((z[u] * y for z, (_, y) in zip(Z, rows)), Fraction(0)) for u in range(n)]
    return A, A0, b, Z


def statistics(rows, Z, x, rank, intercept, ridge):
    ys = [y for _, y in rows]
    rss = sum(((y - sum(zu * xu for zu, xu in zip(z, x))) ** 2 for z, y in zip(Z, ys)), Fraction(0))
    n = len(ys)
    if intercept:
        tss = sum(((y - sum(ys, Fraction(0)) / n) ** 2 for y in ys), Fraction(0)) if n else None
    else:
        tss = sum((y * y for y in ys), Fraction(0))
    r2 = 1 - rss / tss if tss else None
    sigma2 = rss / (n - rank) if not ridge and n > rank else None
    return rss, tss, r2, sigma2


# ------------------------------------------------------------------ data
def dataset(rng, n, d, groups, nulls=0.08, collinear=False):
    cols, valid, scales = {}, {}, {}
    for j in range(d):
        cols[f"x{j}"] = rng.integers(-60, 61, n)
        scales[f"x{j}"] = 1 if j == 0 else 0
        valid[f"x{j}"] = rng.random(n) >= nulls
    if collinear and d >= 3:
        cols["x2"] = cols["x0"] + 10 * cols["x1"]             # x2 = 10 (x0 / 10) + 10 x1: rank deficient (scales)
        scales["x2"] = 1
        valid["x2"] = valid["x0"] & valid["x1"]
    cols["y"] = 3 * cols["x0"] - 2 * (cols["x1"] if d > 1 else 0) + rng.integers(-40, 41, n)
    scales["y"] = 2
    valid["y"] = rng.random(n) >= nulls
    g = rng.integers(0, groups, n)
    return cols, valid, scales, g


def scope_of(cols, valid, scales, g=None, groups=1, rows=None):
    return mbk.Scope.from_arrays(cols, valid=valid, scales=scales, groups=g if groups > 1 else None,
                                 ngroups=groups if groups > 1 else None, rows=rows)


def complete_rows(cols, valid, scales, features, keep, state=None):
    """the common-listwise rows of a state over `state` columns (all of them present), projected on `features`"""
    out = []
    for r in range(len(cols["y"])):
        if not keep[r] or not valid["y"][r] or not all(valid[f][r] for f in (state or features)):
            continue
        x = tuple(Fraction(int(cols[f][r]), 10 ** scales[f]) for f in features)
        out.append((x, Fraction(int(cols["y"][r]), 10 ** scales["y"])))
    return out


class Lin003(unittest.TestCase):
    def check_against_oracle(self, M, rows, intercept, ridge):
        A, A0, b, Z = normal_equations(rows, intercept, ridge)
        kind, x, null = gauss_jordan(A, b)
        self.assertEqual((M.kind, M.x, M.nullspace), (kind, x, null))
        self.assertEqual((M.A, M.b), (A, b))
        self.assertEqual(M.rows, len(rows))
        if kind != "none":
            rank = len(A) - len(null)
            rss, tss, r2, sigma2 = statistics(rows, Z, x, rank, intercept, ridge)
            self.assertEqual((M.rss, M.tss, M.r2, M.sigma2), (rss, tss, r2, sigma2))

    # ---------------------------------------------------------- acceptance 1: LOGO vs complete reconstruction
    def test_lin003_leave_group_out_equals_complete_reconstruction_and_the_fraction_oracle(self):
        rng = np.random.default_rng(SEED)
        for d, G, ridge, features, intercept, collinear in ((2, 5, 0, None, None, False),
                                                            (3, 4, Fraction(1, 3), None, None, False),
                                                            (3, 3, 0, ["x0", "x2"], False, False),
                                                            (3, 4, 0, None, None, True)):
            cols, valid, scales, g = dataset(rng, 160, d, G, collinear=collinear)
            names = [f"x{j}" for j in range(d)]
            scope = scope_of(cols, valid, scales, g, G)
            st = lin.moments(scope, names, "y")
            L = lu.leave_one_group_out(st, ridge=ridge, features=features, intercept=intercept)
            Lc = lu.leave_one_group_out(st, ridge=ridge, features=features, intercept=intercept, backend="cuda")
            self.assertEqual(len(L), G)
            feats = features or names
            icpt = True if intercept is None else intercept
            for grp in range(G):
                self.assertTrue(L[grp].same(Lc[grp]), (d, grp))                # CPU Bareiss == device modular
                self.assertEqual((L[grp].group, L[grp].method, Lc[grp].method), (grp, "subtracted", "modular"))
                self.assertIn(Lc[grp].backend, ("cuda", "cuda-emulated"))
                keep = g != grp
                st2 = lin.moments(scope_of(cols, valid, scales, rows=keep), names, "y")
                F = st2.fit(0, ridge=ridge, features=features, intercept=intercept)        # complete reconstruction
                self.assertEqual((L[grp].kind, L[grp].x, L[grp].nullspace), (F.kind, F.x, F.nullspace))
                self.assertEqual((L[grp].rss, L[grp].tss, L[grp].r2, L[grp].sigma2, L[grp].yty, L[grp].projection),
                                 (F.rss, F.tss, F.r2, F.sigma2, F.yty, F.projection))
                self.assertEqual((L[grp].A, L[grp].b, L[grp].design), (F.A, F.b, F.design))
                self.check_against_oracle(L[grp], complete_rows(cols, valid, scales, feats, keep, names), icpt, ridge)
            if collinear:
                self.assertTrue(all(m.kind == "family" for m in L))
        # a subset of groups, pairwise states (statistics: y^T y only) and a single group (its complement is empty)
        cols, valid, scales, g = dataset(rng, 120, 2, 6)
        scope = scope_of(cols, valid, scales, g, 6)
        st = lin.moments(scope, ["x0", "x1"], "y", policy="pairwise")
        L = lu.leave_one_group_out(st, groups=[4, 1])
        Lc = lu.leave_one_group_out(st, groups=[4, 1], backend="cuda")
        self.assertEqual([m.group for m in L], [4, 1])
        for m, mc, grp in zip(L, Lc, (4, 1)):
            self.assertTrue(m.same(mc))
            F = lin.moments(scope_of(cols, valid, scales, rows=g != grp), ["x0", "x1"], "y", policy="pairwise").fit(0)
            self.assertEqual((m.x, m.yty, m.rss), (F.x, F.yty, None))
        one = lin.moments(scope_of(cols, valid, scales), ["x0", "x1"], "y")
        (E,) = lu.leave_one_group_out(one)
        self.assertEqual((E.rows, E.kind, E.x), (0, "family", [0, 0, 0]))   # no rows left: x = 0 + the nullspace

    def test_lin003_subtracting_a_compatible_group_state_is_leave_group_out(self):
        rng = np.random.default_rng(SEED + 1)
        cols, valid, scales, g = dataset(rng, 150, 2, 4)
        whole = lin.moments(scope_of(cols, valid, scales), ["x0", "x1"], "y")
        grouped = lin.moments(scope_of(cols, valid, scales, g, 4), ["x0", "x1"], "y")
        T = lu.fit(whole, 0, ridge=Fraction(2, 5))
        L = lu.leave_one_group_out(grouped, ridge=Fraction(2, 5), inverse=True)
        for grp in range(4):
            S = T.subtract(grouped, grp)
            self.assertTrue(S.same(L[grp]), grp)
            self.assertEqual(S.method, "subtracted")
            self.assertTrue(S.subtract(grouped, grp, sign=1).same(T))         # adding it back: the total
        other = lin.moments(scope_of(cols, valid, scales, g, 4), ["x1", "x0"], "y")    # another column order
        with self.assertRaises(DomainError):
            T.subtract(other, 0)
        with self.assertRaises(DomainError):
            lu.system([[1, 0], [0, 1]], [1, 1]).subtract(grouped, 0)            # not a regression model

    # ---------------------------------------------------------- acceptance 1: deletions, insertions, ridge, updates
    def test_lin003_row_deletion_insertion_and_ridge_updates_equal_reconstruction(self):
        rng = np.random.default_rng(SEED + 2)
        cols, valid, scales, g = dataset(rng, 140, 3, 1, nulls=0.0)
        names = ["x0", "x1", "x2"]
        st = lin.moments(scope_of(cols, valid, scales), names, "y")
        for ridge in (0, Fraction(3, 4)):
            M = lu.fit(st, 0, ridge=ridge)
            self.assertTrue(M.has_inverse)
            A, A0, b, Z = normal_equations(complete_rows(cols, valid, scales, names, np.ones(140, bool)), True, ridge)
            self.assertEqual(M.inverse, inverse(A))
            self.assertEqual(M.det, det(A))
            for drop in ([5], [0, 17, 33], list(range(40, 52))):
                keep = np.ones(140, bool)
                keep[drop] = False
                rows = [dict(x0=Fraction(int(cols["x0"][i]), 10), x1=int(cols["x1"][i]), x2=int(cols["x2"][i]),
                             y=Fraction(int(cols["y"][i]), 100)) for i in drop]
                D = M.delete_rows(rows)
                R = lu.fit(lin.moments(scope_of(cols, valid, scales, rows=keep), names, "y"), 0, ridge=ridge)
                self.assertEqual((D.method, D.fallback, D.update_rank), ("woodbury", False, len(drop)))
                self.assertTrue(D.same(R))                                   # == complete reconstruction
                self.check_against_oracle(D, complete_rows(cols, valid, scales, names, keep), True, ridge)
                # the exact update denominator det(I - Z A^-1 Z^T) (one row: 1 - h, the leverage complement)
                Ainv = inverse(A)
                Zd = [[Fraction(1), r["x0"], Fraction(r["x1"]), Fraction(r["x2"])] for r in rows]
                K = [[Fraction(int(i == j)) - sum(Zd[i][u] * Ainv[u][w] * Zd[j][w] for u in range(4) for w in range(4))
                      for j in range(len(rows))] for i in range(len(rows))]
                self.assertEqual(D.denominator, det(K))
                self.assertEqual(D.det, det(A) * det(K))                     # det(A') = det(A) det(K)
                I = D.insert_rows(rows)                                      # inserting them back: the model again
                self.assertTrue(I.same(M))
                tuples = [((r["x0"], r["x1"], r["x2"]), r["y"]) for r in rows]
                self.assertTrue(M.delete_rows(tuples).same(D))
            # ridge changes: Woodbury (rank 3) == rebuild == LIN-002's fit with that ridge == the oracle
            for lam in (Fraction(1, 7), 5, 0):
                W, Rb = M.ridge(lam), M.ridge(lam, method="rebuild")
                same = lam == ridge                                           # no change: both return a copy
                self.assertEqual((W.method, Rb.method), (M.method, M.method) if same else ("woodbury", "rebuilt"))
                self.assertTrue(W.same(Rb))
                F = st.fit(0, ridge=lam)
                self.assertEqual((W.x, W.rss, W.sigma2), (F.x, F.rss, F.sigma2))
        # a generic low-rank update of a square system: Woodbury == rebuilt system; det(K) exact
        A = [[4, 1, 0], [1, 3, 1], [0, 1, 2]]
        S = lu.system(A, [1, 2, 3])
        U = [[1, 0], [0, Fraction(1, 2)], [2, 1]]
        V = [[0, 1], [1, 1], [Fraction(-1, 3), 0]]
        db = [0, 1, -1]
        Su = S.update(U, V, db)
        A2 = [[Fraction(A[i][j]) + sum(Fraction(U[i][t]) * V[j][t] for t in range(2)) for j in range(3)] for i in range(3)]
        Sr = lu.system(A2, [1, 3, 2])
        self.assertTrue(Su.same(Sr))
        Ainv = inverse(A)
        K = [[Fraction(int(s == t)) + sum(Fraction(V[i][s]) * Ainv[i][j] * U[j][t] for i in range(3) for j in range(3))
              for t in range(2)] for s in range(2)]
        self.assertEqual(Su.denominator, det(K))
        self.assertEqual(Su.inverse, inverse(A2))

    # ---------------------------------------------------------- acceptance 2: denominators and the contract
    def test_lin003_singular_update_denominators_rebuild_or_fail_by_contract(self):
        # the only row with z != 0 has leverage 1: deleting it makes A' singular, det(K) = 1 - h = 0 exactly
        xs, zs, ys = np.array([1, 2, 3, 4, 5]), np.array([0, 0, 0, 0, 7]), np.array([2, 4, 7, 8, 11])
        cols = {"x": xs, "z": zs, "y": ys}
        st = lin.moments(mbk.Scope.from_arrays(cols), ["x", "z"], "y")
        M = lu.fit(st, 0)
        row = dict(x=5, z=7, y=11)
        D = M.delete_rows([row])
        self.assertEqual((D.denominator, D.denominator_sign, D.method, D.fallback), (0, 0, "rebuilt", True))
        F = lin.moments(mbk.Scope.from_arrays(cols, rows=np.array([1, 1, 1, 1, 0], bool)), ["x", "z"], "y").fit(0)
        self.assertEqual((D.kind, D.x, D.nullspace, D.rss), (F.kind, F.x, F.nullspace, F.rss))
        self.assertEqual(D.kind, "family")
        self.assertFalse(D.has_inverse)
        with self.assertRaises(SingularError):
            M.delete_rows([row], fallback="fail")
        # a singular base (exactly rank deficient: z = 2 x) has no inverse: rebuilt, or refused under "fail"
        cols2 = {"x": xs, "z": 2 * xs, "y": ys}
        st2 = lin.moments(mbk.Scope.from_arrays(cols2), ["x", "z"], "y")
        Ms = lu.fit(st2, 0)
        self.assertEqual((Ms.kind, Ms.has_inverse, Ms.det), ("family", False, None))
        Ri = Ms.insert_rows([dict(x=1, z=5, y=0)])                           # the new row restores full rank
        self.assertEqual((Ri.method, Ri.kind, Ri.has_inverse), ("rebuilt", "unique", True))
        cols3 = {"x": np.append(xs, 1), "z": np.append(2 * xs, 5), "y": np.append(ys, 0)}
        self.assertTrue(Ri.same(lu.fit(lin.moments(mbk.Scope.from_arrays(cols3), ["x", "z"], "y"), 0)))
        with self.assertRaises(SingularError):
            Ms.insert_rows([dict(x=1, z=5, y=0)], fallback="fail")
        # a generic rank-one update to a singular matrix: 1 + v^T A^-1 u = 0
        S = lu.system([[2, 0], [0, 3]], [1, 1])
        Z = S.update([[-2], [0]], [[1], [0]], db=[-1, 0])                   # A' = diag(0, 3), b' = (0, 1)
        self.assertEqual((Z.denominator, Z.kind, Z.method), (0, "family", "rebuilt"))
        self.assertEqual((Z.x, Z.nullspace), ([0, Fraction(1, 3)], [[1, 0]]))
        Zn = S.update([[-2], [0]], [[1], [0]])                               # b' = (1, 1): inconsistent
        self.assertEqual((Zn.kind, Zn.certificate), ("none", [Fraction(1), 0]))
        with self.assertRaises(SingularError):
            S.update([[-2], [0]], [[1], [0]], fallback="fail")
        N = lu.system([[2, 0], [0, 3]], [1, 1], inverse=False)               # no inverse kept: Woodbury n/a
        self.assertEqual(N.update([[1], [0]], [[1], [0]]).method, "rebuilt")
        with self.assertRaises(DomainError):
            N.update([[1], [0]], [[1], [0]], fallback="fail")
        with self.assertRaises(ValueError):
            M.delete_rows([row], fallback="ignore")

    def test_lin003_floating_variant_carries_condition_bounds_and_refuses_when_they_fail(self):
        rng = np.random.default_rng(SEED + 3)
        cols, valid, scales, g = dataset(rng, 120, 3, 1, nulls=0.0)
        st = lin.moments(scope_of(cols, valid, scales), ["x0", "x1", "x2"], "y")
        M = lu.fit(st, 0)
        f = M.float_solve(tol=1e-9)
        self.assertTrue(f["certified"])
        self.assertLess(f["alpha"], 1)
        self.assertLessEqual(f["bound"], 1e-9)
        err = max(abs(Fraction(a) - b) for a, b in zip(f["x"], M.x))
        self.assertLessEqual(err, Fraction(f["bound"]))                     # the certified bound holds exactly
        self.assertGreaterEqual(f["kappa"], 1)
        # nearly singular: kappa ~ 1e12, the bound fails tol = 1e-9: refused (or the exact model by declaration)
        eps = Fraction(1, 10 ** 12)
        A = [[1, 1, 0], [1, 1 + eps, 0], [0, 0, 1]]
        Nm = lu.system(A, [2, 2 + eps, 1])
        with self.assertRaises(NotConvergedError) as e:
            Nm.float_solve(tol=1e-9)
        self.assertGreater(e.exception.info["kappa"], 1e11)
        loose = Nm.float_solve(tol=1e-2)                                     # a looser contract certifies it
        self.assertTrue(loose["certified"])
        self.assertLessEqual(max(abs(Fraction(a) - b) for a, b in zip(loose["x"], Nm.x)), Fraction(loose["bound"]))
        ex = Nm.float_solve(tol=1e-9, fallback="exact")
        self.assertEqual((ex["certified"], ex["x"]), (False, [1, 1, 1]))
        # the Hilbert matrix of order 13: the condition bound fails (alpha >= 1 or a zero pivot)
        H = [[Fraction(1, i + j + 1) for j in range(13)] for i in range(13)]
        Hm = lu.system(H, [1] * 13)
        with self.assertRaises(NotConvergedError) as e:
            Hm.float_solve(tol=1.0)
        self.assertNotEqual(e.exception.info["reason"], "certified")
        # exactly rank deficient: refused as singular (the exact model knows it)
        with self.assertRaises(SingularError):
            lu.system([[1, 2], [2, 4]], [1, 2]).float_solve()

    # ---------------------------------------------------------- budgets and refusals
    def test_lin003_budgets_at_the_limit_and_one_past(self):
        rng = np.random.default_rng(SEED + 4)
        cols, valid, scales, g = dataset(rng, 80, 2, 4)
        st = lin.moments(scope_of(cols, valid, scales, g, 4), ["x0", "x1"], "y")
        n, k = 3, 4                                  # unknowns; augmented columns (LIN-002's normal equations k (k + 1))
        lu.fit(st, 0, budget=Budget(tensor=n * (2 * n + 1), output=k * (k + 1)))
        with self.assertRaises(AlgebraBudgetError) as e:
            lu.fit(st, 0, budget=Budget(tensor=n * (2 * n + 1) - 1))          # the elimination of [M | L | c]
        self.assertEqual((e.exception.resource, e.exception.need), ("tensor", n * (2 * n + 1)))
        with self.assertRaises(AlgebraBudgetError) as e:
            lu.fit(st, 0, budget=Budget(output=k * (k + 1) - 1))              # LIN-002's system of the fit
        self.assertEqual((e.exception.resource, e.exception.need), ("output", k * (k + 1)))
        M = lu.fit(st, 0)
        rows = [dict(x0=1, x1=2, y=3), dict(x0=0, x1=1, y=1)]
        r = 2
        M.delete_rows(rows, budget=Budget(tensor=n * r + r * r + n * n))
        with self.assertRaises(AlgebraBudgetError) as e:
            M.delete_rows(rows, budget=Budget(tensor=n * r + r * r + n * n - 1))
        self.assertEqual((e.exception.resource, e.exception.need), ("tensor", n * r + r * r + n * n))
        S = st.nslots
        # the plan: states = groups x slots, tensor = systems x n (n + 1), output = systems; each system's solve:
        # LIN-002's need (output n (n + 1)) and the elimination's (tensor n (2n + 1))
        lu.leave_one_group_out(st, budget=Budget(states=4 * S, tensor=4 * n * (n + 1), output=n * (n + 1)))
        with self.assertRaises(AlgebraBudgetError) as e:
            lu.leave_one_group_out(st, budget=Budget(states=4 * S - 1))
        self.assertEqual((e.exception.resource, e.exception.need), ("states", 4 * S))
        with self.assertRaises(AlgebraBudgetError) as e:
            lu.leave_one_group_out(st, budget=Budget(tensor=4 * n * (n + 1) - 1))
        self.assertEqual((e.exception.resource, e.exception.need), ("tensor", 4 * n * (n + 1)))
        with self.assertRaises(AlgebraBudgetError) as e:
            lu.leave_one_group_out(st, budget=Budget(output=3))
        self.assertEqual((e.exception.resource, e.exception.need), ("output", 4))
        with self.assertRaises(AlgebraBudgetError) as e:
            lu.leave_one_group_out(st, budget=Budget(output=n * (n + 1) - 1))
        self.assertEqual((e.exception.resource, e.exception.need), ("output", n * (n + 1)))
        plan = lu.logo_plan(st)
        k = plan.primes_needed
        need = plan.device_bytes(k)
        plan.residues(k, "cuda", budget=Budget(bytes=need))
        with self.assertRaises(AlgebraBudgetError) as e:
            plan.residues(k, "cuda", budget=Budget(bytes=need - 1))
        self.assertEqual((e.exception.resource, e.exception.need), ("bytes", need))
        M.float_solve(budget=Budget(tensor=n * n))
        with self.assertRaises(AlgebraBudgetError):
            M.float_solve(budget=Budget(tensor=n * n - 1))

    def test_lin003_refusals_and_the_modular_rerun(self):
        rng = np.random.default_rng(SEED + 5)
        cols, valid, scales, g = dataset(rng, 90, 2, 3)
        st = lin.moments(scope_of(cols, valid, scales, g, 3), ["x0", "x1"], "y")
        M = lu.fit(st, 0)
        with self.assertRaises(UnsupportedError):
            lu.system([[int(i == j) for j in range(64)] for i in range(64)], [0] * 64)
        with self.assertRaises(UnsupportedError):
            M.update([[0] * 65] * 3, [[0] * 65] * 3)
        with self.assertRaises(ValueError):
            M.update([[1]] * 2, [[1]] * 2)                                    # not n x r
        with self.assertRaises(ValueError):
            M.delete_rows([dict(x0=1, y=2)])                                  # a row without x1
        with self.assertRaises(ValueError):
            M.delete_rows([dict(x0=1, x1=1, y=1)] * (M.rows + 1))           # more rows than the model holds
        with self.assertRaises(TypeError):
            M.delete_rows([dict(x0=1.5, x1=1, y=1)])                          # floats are not exact input
        with self.assertRaises(ValueError):
            M.ridge(-1)
        pw = lu.fit(lin.moments(scope_of(cols, valid, scales), ["x0", "x1"], "y", policy="pairwise"), 0)
        with self.assertRaises(DomainError):
            pw.delete_rows([dict(x0=1, x1=1, y=1)])                           # not common-listwise
        with self.assertRaises(DomainError):
            lu.system([[1]], [1]).ridge(1)
        with self.assertRaises(ValueError):
            lu.leave_one_group_out(st, groups=[3])                            # no such group
        # too few primes: the native finish asks for more (NOT_CONVERGED with the count), the rerun converges
        plan = lu.logo_plan(st)
        res, fields = plan.residues(1, "cpu")
        models, more = plan.finish(1, res, fields)
        self.assertGreater(plan.primes_needed, 3)                             # these sums need several primes
        self.assertIsNone(models)
        self.assertGreater(more, 1)
        res, fields = plan.residues(more, "cpu")
        self.assertIsNotNone(plan.finish(more, res, fields)[0])
        dev, _ = plan.residues(plan.primes_needed, "cuda")
        cpu, _ = plan.residues(plan.primes_needed, "cpu")
        self.assertTrue(np.array_equal(dev, cpu))                              # the kernel's residues, bit for bit


if __name__ == "__main__":
    unittest.main()
