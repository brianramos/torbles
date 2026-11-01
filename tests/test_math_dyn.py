# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""DYN-001 / DYN-002: finite closed-observable dynamics, operator powers, semigroups and recurrences
(torbles/math_dyn.c, math_dyn_gpu.cu, math_dyn.py).

Oracles, independent of the implementation: every map written here as an explicit point map with Python Fractions,
applied step by step to every selected complete row (explicit row evolution) and the moments sum_r w_r prod x^a formed
directly; the closure witness and the dropped-term bounds from a small polynomial algebra written here (dicts of
exponent tuples); powers as repeated Fraction matrix products; the characteristic polynomial by Faddeev-LeVerrier in
Fractions; the linear complexity of a sequence as the rank of its Hankel matrix (Fraction elimination); mixtures as
the exact expectation over every sequence of random map choices; NumPy only for the floating contract (stated
tolerances); DuckDB SQL for the moment sums of rows evolved in exact HUGEINT arithmetic. Every native result is also
computed on the CUDA path (emulated here with TORBLES_EMU_SCRAMBLE=1) and compared for equality. Seeds are fixed.
"""
import ctypes as C
import itertools
import unittest
from fractions import Fraction

import duckdb
import numpy as np
import pandas as pd

import torbles
from torbles import math_bank as mbk, math_dyn as dyn, math_sym as ms, math_tensor as mt
from torbles.math_budgets import Budget
from torbles.math_domains import DomainError
from torbles.math_errors import UnsupportedError
from torbles.mathematics import AlgebraBudgetError

F = Fraction


# ---------------------------------------------------------------------- oracles: point maps (value units)
def o_affine(idx, A, b):
    def f(p):
        q = list(p)
        for r, i in enumerate(idx):
            q[i] = sum(F(A[r][k]) * p[idx[k]] for k in range(len(idx))) + F(b[r])
        return tuple(q)
    return f


def o_rotate(i, j, c, s, center=(0, 0)):
    cx, cy = F(center[0]), F(center[1])

    def f(p):
        q = list(p)
        x, y = p[i] - cx, p[j] - cy
        q[i], q[j] = cx + c * x - s * y, cy + s * x + c * y
        return tuple(q)
    return f


def o_compose(*fs):
    def h(p):
        for f in reversed(fs):
            p = f(p)
        return p
    return h


def o_poly(components, n):
    """components {i: {exponent tuple: coefficient}}: x'_i = sum c x^e (others fixed)"""
    def f(p):
        q = list(p)
        for i, terms in components.items():
            q[i] = sum((F(c) * mono(p, e) for e, c in terms.items()), F(0))
        return tuple(q)
    return f


def mono(p, t):
    v = F(1)
    for x, e in zip(p, t):
        v *= F(x) ** e
    return v


def moments(points, terms, weights=None):
    w = weights or [1] * len(points)
    return [sum((F(wi) * mono(p, t) for p, wi in zip(points, w)), F(0)) for t in terms]


def evolve_points(points, f, t):
    for _ in range(t):
        points = [f(p) for p in points]
    return points


def norm(v):
    v = F(v)
    return v.numerator if v.denominator == 1 else v


# ---------------------------------------------------------------------- oracles: polynomials and matrices
def p_mul(a, b):
    out = {}
    for ea, ca in a.items():
        for eb, cb in b.items():
            e = tuple(x + y for x, y in zip(ea, eb))
            out[e] = out.get(e, F(0)) + ca * cb
    return {e: c for e, c in out.items() if c}


def p_image(term, comps, n):
    """phi_term(F(x)) as a polynomial: prod_i F_i^term_i (F_i = x_i when not declared)"""
    out = {tuple([0] * n): F(1)}
    for i, k in enumerate(term):
        Fi = comps.get(i, {tuple(1 if j == i else 0 for j in range(n)): F(1)})
        for _ in range(k):
            out = p_mul(out, {e: F(c) for e, c in Fi.items()})
    return out


def m_mul(A, B):
    return [[sum((A[i][k] * B[k][j] for k in range(len(B))), F(0)) for j in range(len(B[0]))] for i in range(len(A))]


def m_pow(A, n):
    R = [[F(int(i == j)) for j in range(len(A))] for i in range(len(A))]
    for _ in range(n):                                   # direct repeated products (not squaring)
        R = m_mul(R, A)
    return R


def m_vec(A, v):
    return [sum((A[i][k] * v[k] for k in range(len(v))), F(0)) for i in range(len(A))]


def faddeev_leverrier(A):
    """det(x I - A) = x^n + c_(n-1) x^(n-1) + ... + c_0 (constant first, leading 1 last)"""
    n = len(A)
    M = [[F(0)] * n for _ in range(n)]
    c = [F(0)] * (n + 1)
    c[n] = F(1)
    I = [[F(int(i == j)) for j in range(n)] for i in range(n)]
    for k in range(1, n + 1):
        AM = m_mul(A, M)
        M = [[AM[i][j] + c[n - k + 1] * I[i][j] for j in range(n)] for i in range(n)]
        AM = m_mul(A, M)
        c[n - k] = -sum(AM[i][i] for i in range(n)) / k
    return c


def rank(rows):
    M = [list(map(F, r)) for r in rows]
    r = 0
    for c in range(len(M[0]) if M else 0):
        piv = next((i for i in range(r, len(M)) if M[i][c] != 0), None)
        if piv is None:
            continue
        M[r], M[piv] = M[piv], M[r]
        for i in range(len(M)):
            if i != r and M[i][c] != 0:
                f = M[i][c] / M[r][c]
                M[i] = [x - f * y for x, y in zip(M[i], M[r])]
        r += 1
    return r


# ---------------------------------------------------------------------- data
def make_xyz(rng, n=48, G=3, nulls=0.15, lo=-4, hi=5):
    vals = {"x": rng.integers(lo, hi, n), "y": rng.integers(lo, hi, n), "z": rng.integers(lo, hi, n),
            "w": rng.integers(-2, 4, n)}
    valid = {"z": rng.random(n) >= nulls}
    groups = rng.integers(0, G, n)
    rows = rng.random(n) < 0.9
    scope = mbk.Scope.from_arrays(vals, valid=valid, groups=groups, ngroups=G, rows=rows)
    pts, wts = [[] for _ in range(G)], [[] for _ in range(G)]
    for r in range(n):
        if rows[r] and valid["z"][r]:
            pts[groups[r]].append((F(int(vals["x"][r])), F(int(vals["y"][r])), F(int(vals["z"][r]))))
            wts[groups[r]].append(int(vals["w"][r]))
    return scope, pts, wts


def make_xy(rng, n=30, G=2, lo=-3, hi=4):
    xs, ys, g = rng.integers(lo, hi, n), rng.integers(lo, hi, n), rng.integers(0, G, n)
    scope = mbk.Scope.from_arrays({"x": xs, "y": ys}, groups=g, ngroups=G)
    pts = [[(F(int(xs[r])), F(int(ys[r]))) for r in range(n) if g[r] == k] for k in range(G)]
    return scope, pts


class Base(unittest.TestCase):
    def check_evolution(self, st, K, f, pts, steps, wts=None, methods=("stepping", "squaring", "modular", "auto"),
                        cuda=True):
        """every method, CPU and CUDA, against the explicit row evolution"""
        terms = st.basis.terms
        for t in steps:
            evs = [dyn.evolve(st, K, t, method=m) for m in methods]
            if cuda:
                evs.append(dyn.evolve(st, K, t, backend="cuda"))
            for g in range(st.info["groups"]):
                want = [norm(v) for v in moments(evolve_points(pts[g], f, t), terms, None if wts is None else wts[g])]
                for ev in evs:
                    self.assertEqual(ev.moments(g), want, (t, g, ev.info["method"], ev.backend))
            if cuda:
                self.assertEqual(evs[-1].info["method"], "modular")
                self.assertTrue(evs[-1].info["checked"])
                self.assertIn(evs[-1].backend, ("cuda", "cuda-emulated"))
        return evs


# ====================================================================== DYN-001
class ClosedDynamics(Base):
    # ------------------------------------------------------------------ acceptance 1: repeated affine transforms
    def test_dyn001_repeated_affine_transforms_agree_with_explicit_row_evolution(self):
        rng = np.random.default_rng(6101)
        scope, pts, wts = make_xyz(rng)
        geo = ms.geometry(scope, numeric={"x": "m", "y": "m", "z": "m"})
        B = ms.basis(geo, degree=2)
        states = [ms.state(mbk.build(scope, B.terms_for(scope), "joint"), geo),
                  ms.state(mt.moments(scope, ["1", "x", "y", "z"], order=2), geo),
                  ms.state(mt.moments(scope, ["1", "x", "y", "z"], order=2, repr="sparse"), geo)]
        for st in states:
            self.assertTrue(st.basis.equal(B))
        decl = {
            "rotation and translation": (ms.compose(ms.rotate("x", "y", cos=F(3, 5), sin=F(4, 5), center=(1, -1)),
                                                    ms.translate(z=F(1, 2))),
                                         o_compose(o_rotate(0, 1, F(3, 5), F(4, 5), (1, -1)),
                                                   o_affine([2], [[1]], [F(1, 2)]))),
            "affine": (ms.affine(["x", "y", "z"], [[F(1, 2), 1, 0], [0, F(-2, 3), F(1, 4)], [1, 0, F(1, 3)]], [1, 0, -2]),
                       o_affine([0, 1, 2], [[F(1, 2), 1, 0], [0, F(-2, 3), F(1, 4)], [1, 0, F(1, 3)]], [1, 0, -2])),
            "permutation": (ms.permute({"x": "z", "y": "x", "z": "y"}),
                            o_affine([0, 1, 2], [[0, 0, 1], [1, 0, 0], [0, 1, 0]], [0, 0, 0])),
            "scaling and reflection": (ms.compose(ms.scale(x=F(3, 2)), ms.reflect(y=F(1, 2))),
                                       o_affine([0, 1], [[F(3, 2), 0], [0, -1]], [0, 1])),
        }
        for k, (tr, f) in decl.items():
            K = dyn.operator(B, dyn.affine(geo, tr))
            info = K.info
            self.assertTrue(info["sym_action"] and info["closed"] and not info["approximate"], k)
            self.assertTrue(K.matrix() == ms.action(B, tr, geo).matrix(), k)    # SYM-001's action matrix
            self.check_evolution(states[len(k) % 3], K, f, pts, (0, 1, 2, 3, 7))
            # one step on the device (512-bit) and on the CPU, equal to the evolution
            self.assertEqual(K.apply(states[0]), K.apply(states[0], backend="cuda"))
            self.assertEqual(K.apply(states[0]), [dyn.evolve(states[0], K, 1).moments(g) for g in range(3)])
            # the explicit native row evolution agrees with the operator (and the oracle)
            rows = dyn.evolve_rows(scope, dyn.affine(geo, tr), B, 4)
            ev = dyn.evolve(states[0], K, 4)
            for g in range(3):
                self.assertEqual(rows.moments(g), ev.moments(g), k)
                self.assertEqual(rows.group(g), ev.group(g))
        # the same affine map declared as a degree-1 polynomial: an independent expansion gives the same K
        A = [[F(1, 2), 1, 0], [0, F(-2, 3), F(1, 4)], [1, 0, F(1, 3)]]
        comps = {c: {**{n: A[r][j] for j, n in enumerate(("x", "y", "z")) if A[r][j]}, "1": [1, 0, -2][r]}
                 for r, c in enumerate(("x", "y", "z"))}
        Kp = dyn.operator(B, dyn.polynomial(geo, comps))
        self.assertFalse(Kp.info["sym_action"])
        self.assertEqual(Kp.matrix(), dyn.operator(B, dyn.affine(geo, decl["affine"][0])).matrix())
        # weighted moments (a TENSOR-001 weight) and a degree-3 basis
        B3 = ms.basis(geo, degree=3)
        wst = ms.state(mt.moments(scope, ["1", "x", "y", "z"], order=3, weight="w"), geo)
        self.assertTrue(wst.basis.equal(B3) and wst.info["weighted"])
        tr, f = decl["rotation and translation"]
        K3 = dyn.operator(B3, dyn.affine(geo, tr))
        self.check_evolution(wst, K3, f, pts, (0, 2, 5), wts=wts)
        rows = dyn.evolve_rows(scope, dyn.affine(geo, tr), B3, 5, weight="w")
        ev = dyn.evolve(wst, K3, 5)
        for g in range(3):
            self.assertEqual(rows.moments(g), ev.moments(g))

    def test_dyn001_engine_table_decimals_and_duckdb_moment_sums_of_evolved_rows(self):
        frame = pd.DataFrame({
            "k": pd.Series([0, 1, 0, 1, 2, 2, 0, 1, 2, 0], dtype=object),
            "x": pd.Series(["1.25", "-2.50", "3.00", "0.75", "-1.00", "2.25", "0.00", "-3.50", "1.10", "0.05"],
                           dtype=object),
            "y": pd.Series([4, -1, 2, 0, 3, -2, 1, 5, -4, 2], dtype=object)})
        db = torbles.from_ddl("CREATE TABLE t (k BIGINT, x DECIMAL(10,2), y BIGINT)", {"t": frame})
        scope = mbk.Scope.from_table(db["t"], ["x", "y"], group_by=["k"])
        geo = ms.geometry(scope, numeric={"x": "m", "y": "m"})
        self.assertEqual(geo.scales, (2, 0))
        B = ms.basis(geo, degree=2)
        st = ms.state(mbk.build(scope, B.terms_for(scope), "joint"), geo)
        groups = {scope.key_values(g)[0]: g for g in range(scope.groups)}
        # x' = 2 x - 3 y + 0.25, y' = y + 1 (value units): integral in logical units, x_l' = 2 x_l - 300 y + 25
        D = dyn.affine(geo, ms.affine(["x", "y"], [[2, -3], [0, 1]], ["0.25", 1]))
        K = dyn.operator(B, D)
        self.assertEqual(K.info["den"], 1)
        steps = 3
        evs = [dyn.evolve(st, K, steps, method=m) for m in ("stepping", "squaring", "modular")]
        evs.append(dyn.evolve(st, K, steps, backend="cuda"))
        sql = ["WITH s0 AS (SELECT k, CAST(x * 100 AS HUGEINT) AS xl, CAST(y AS HUGEINT) AS y FROM t)"]
        for s in range(steps):
            sql.append(f", s{s + 1} AS (SELECT k, 2 * xl - 300 * y + 25 AS xl, y + 1 AS y FROM s{s})")
        sql.append(f" SELECT k, COUNT(*), SUM(xl), SUM(y), SUM(xl * xl), SUM(xl * y), SUM(y * y) FROM s{steps} GROUP BY k")
        duck = {int(r[0]): [int(v) for v in r[1:]] for r in db.con.execute("".join(sql)).fetchall()}
        self.assertEqual(B.terms, [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2)])
        for key, g in groups.items():
            for ev in evs:
                self.assertEqual(ev.moments(g), duck[key], (key, ev.backend))
        # a rotation in value units acts on the logical integers (x: cents, y: units): the Fraction oracle
        R = ms.rotate("x", "y", cos=F(3, 5), sin=F(4, 5), center=(1, 0))
        KR = dyn.operator(B, dyn.affine(geo, R))
        pts = [[] for _ in range(scope.groups)]
        for k, x, y in zip(frame["k"], frame["x"], frame["y"]):
            pts[groups[int(k)]].append((F(x), F(int(y))))
        f = o_rotate(0, 1, F(3, 5), F(4, 5), (1, 0))
        for t in (1, 4):
            ev, evc = dyn.evolve(st, KR, t), dyn.evolve(st, KR, t, backend="cuda")
            for g in range(scope.groups):
                logical = [(p[0] * 100, p[1]) for p in evolve_points(pts[g], f, t)]
                want = [norm(v) for v in moments(logical, B.terms)]
                self.assertEqual(ev.moments(g), want)
                self.assertEqual(evc.moments(g), want)
        rows = dyn.evolve_rows(scope, dyn.affine(geo, R), B, 4)
        self.assertEqual([rows.moments(g) for g in range(scope.groups)],
                         [dyn.evolve(st, KR, 4).moments(g) for g in range(scope.groups)])

    def test_dyn001_nonlinear_closed_observables_are_exact(self):
        rng = np.random.default_rng(6102)
        scope, pts = make_xy(rng)
        geo = ms.geometry(scope, numeric={"x": "m", "y": "m"})
        # x' = x / 2 + 1, y' = y / 3 + x^2: {1, x, x^2, y} is closed (Koopman-invariant observables)
        comps = {"x": {"x": F(1, 2), "1": 1}, "y": {"y": F(1, 3), "x^2": 1}}
        P = dyn.polynomial(geo, comps)
        self.assertEqual(P.info["kind"], "polynomial")
        Bc = ms.basis(geo, terms=["1", "x", "x^2", "y"])
        K = dyn.operator(Bc, P)
        self.assertTrue(K.info["closed"] and K.info["kind"] == "exact")
        oc = {0: {(1, 0): F(1, 2), (0, 0): 1}, 1: {(0, 1): F(1, 3), (2, 0): 1}}
        # the matrix: the images of the basis terms (polynomial oracle)
        terms = Bc.terms
        for a, term in enumerate(terms):
            img = p_image(term, oc, 2)
            self.assertEqual(K.matrix()[a], [norm(img.get(b, 0)) for b in terms])
        st = ms.state(mbk.build(scope, Bc.terms_for(scope), "joint"), geo)
        f = o_poly(oc, 2)
        self.check_evolution(st, K, f, pts, (0, 1, 3, 8))
        for t in (2, 5):
            rows = dyn.evolve_rows(scope, P, Bc, t)
            for g in range(2):
                self.assertEqual(rows.moments(g), [norm(v) for v in moments(evolve_points(pts[g], f, t), terms)])
        # x' = -x, y' = 2 y - x^2 + 1 on {1, x, x^2, y, x^4, x^2 y, y^2} (degree 4: an FND-004 bank)
        quad = {"x": {"x": -1}, "y": {"y": 2, "x^2": -1, "1": 1}}
        Bq = ms.basis(geo, terms=["1", "x", "x^2", "y", "x^4", "x^2*y", "y^2"])
        Kq = dyn.operator(Bq, dyn.polynomial(geo, quad))
        stq = ms.state(mbk.build(scope, Bq.terms_for(scope), "joint"), geo)
        fq = o_poly({0: {(1, 0): -1}, 1: {(0, 1): 2, (2, 0): -1, (0, 0): 1}}, 2)
        self.check_evolution(stq, Kq, fq, pts, (0, 1, 4))
        dyn.operator(ms.basis(geo, terms=["1", "x", "x^2", "y", "x^4", "x^2*y"]), dyn.polynomial(geo, quad))  # closed
        with self.assertRaises(dyn.NotClosedError) as cm:   # the image of y^2 has x^4, outside {1, x, y, y^2}
            dyn.operator(ms.basis(geo, terms=["1", "x", "y", "y^2"]), dyn.polynomial(geo, quad))
        self.assertEqual(cm.exception.term, (0, 1))           # already y -> 2 y - x^2 + 1 leaves it
        self.assertEqual(cm.exception.missing, {"x": 2})

    # ------------------------------------------------------------------ acceptance 2: non-closure and truncation
    def test_dyn001_nonclosure_detected_and_truncation_requires_a_contract(self):
        rng = np.random.default_rng(6103)
        scope, pts = make_xy(rng, n=28)
        geo = ms.geometry(scope, numeric={"x": "m", "y": "m"})
        B = ms.basis(geo, degree=2)
        comps = {"x": {"x": F(1, 2)}, "y": {"y": F(1, 3), "x^2": F(1, 8)}}
        oc = {0: {(1, 0): F(1, 2)}, 1: {(0, 1): F(1, 3), (2, 0): F(1, 8)}}
        P = dyn.polynomial(geo, comps)
        # the witness: the first basis term whose image leaves the span, with a monomial outside it
        first = next(a for a, t in enumerate(B.terms) if any(e not in B.terms for e in p_image(t, oc, 2)))
        with self.assertRaises(dyn.NotClosedError) as cm:
            dyn.operator(B, P)
        self.assertEqual(cm.exception.term, B.terms[first])
        miss = tuple(cm.exception.missing.get(n, 0) for n in ("x", "y"))
        self.assertIn(miss, p_image(B.terms[first], oc, 2))
        self.assertNotIn(miss, B.terms)
        # the contract: a box, its dropped-term bounds, and the approximate label
        bx = dyn.box(geo, {"x": (-3, 3), "y": (-3, 3)})
        Kt = dyn.operator(B, P, truncate=bx)
        info = Kt.info
        self.assertTrue(info["approximate"] and info["kind"] == "truncated" and not info["closed"])
        self.assertEqual(info["label"], "certified approximate")
        self.assertTrue(info["invariant"])           # |x|/2 <= 3/2, |y|/3 + x^2/8 <= 1 + 9/8 <= 3
        eps = []
        for t in B.terms:
            img = p_image(t, oc, 2)
            eps.append(norm(sum((abs(c) * F(3) ** (e[0] + e[1]) for e, c in img.items() if e not in B.terms), F(0))))
        self.assertEqual(Kt.epsilon(), eps)
        for a, t in enumerate(B.terms):              # the kept terms
            img = p_image(t, oc, 2)
            self.assertEqual(Kt.matrix()[a], [norm(img.get(b, 0)) for b in B.terms])
        with self.assertRaises(dyn.NotAuthorizedError) as cm:
            Kt.authorize_exact()
        self.assertEqual(cm.exception.reason, "approximate")
        self.assertTrue(dyn.operator(ms.basis(geo, terms=["1", "x", "x^2", "y"]), P).authorize_exact()["authorized"])
        st = ms.state(mbk.build(scope, B.terms_for(scope), "joint"), geo)
        f = o_poly(oc, 2)
        with self.assertRaises(ValueError):          # the scope the rows' box membership is checked on
            dyn.evolve(st, Kt, 1)
        for t in (0, 1, 2, 4):
            for method in ("stepping", "squaring"):
                ev = dyn.evolve(st, Kt, t, method=method, scope=scope)
                self.assertTrue(ev.info["approximate"] and ev.info["invariant"])
                for g in range(2):
                    true = moments(evolve_points(pts[g], f, t), B.terms)
                    got, err = ev.moments(g), ev.errors(g)
                    for a in range(len(B)):
                        self.assertLessEqual(abs(true[a] - F(got[a])), F(err[a]), (t, g, a))
                    if t == 1:                       # one step: E_1 = W eps exactly
                        self.assertEqual(err, [norm(ev.group(g)["used"] * F(e)) for e in eps])
                with self.assertRaises(dyn.NotAuthorizedError):
                    ev.authorize_exact()
        with self.assertRaises(UnsupportedError):    # truncated evolutions: stepping or squaring
            dyn.evolve(st, Kt, 2, method="modular", scope=scope)
        # rows outside the box: refused
        with self.assertRaises(DomainError):
            dyn.evolve(st, dyn.operator(B, P, truncate=dyn.box(geo, {"x": (-2, 2), "y": (-3, 3)})), 1, scope=scope)
        # a box that is not F-invariant certifies one step only
        Pw = dyn.polynomial(geo, {"x": {"x": F(1, 2)}, "y": {"y": F(1, 3), "x^2": 1}})
        Kw = dyn.operator(B, Pw, truncate=bx)
        self.assertFalse(Kw.info["invariant"])
        dyn.evolve(st, Kw, 1, scope=scope)
        with self.assertRaises(DomainError):
            dyn.evolve(st, Kw, 2, scope=scope)
        # every coordinate the evolution reads must be bounded
        with self.assertRaises(DomainError):
            dyn.operator(B, P, truncate=dyn.box(geo, {"x": (-3, 3)}))
        # no exact use of an approximate operator drops its uncertainty
        for fn in (lambda: Kt.power(3), lambda: Kt.charpoly(), lambda: Kt.recurrence({"x": 1}, state=st),
                   lambda: Kt.float_power(3, 1e-6), lambda: dyn.mixture([(1, Kt)]), lambda: Kt.at(2),
                   lambda: Kt.apply(st), lambda: dyn.semigroup(Kt, 1, 1), lambda: Kt.power(2, backend="cuda")):
            with self.assertRaises(DomainError):
                fn()
        # a truncation of a closed basis is exact; weighted states and parametric maps have no contract here
        Kc = dyn.operator(ms.basis(geo, terms=["1", "x", "x^2", "y"]), P, truncate=bx)
        self.assertTrue(Kc.info["closed"] and not Kc.info["approximate"])
        wscope = mbk.Scope.from_arrays({"x": np.arange(5), "y": np.arange(5), "w": np.ones(5, dtype=np.int64)})
        wgeo = ms.geometry(wscope, numeric={"x": "m", "y": "m"})
        wst = ms.state(mt.moments(wscope, ["1", "x", "y"], order=2, weight="w"), wgeo)
        Kwt = dyn.operator(ms.basis(wgeo, degree=2), dyn.polynomial(wgeo, comps),
                           truncate=dyn.box(wgeo, {"x": (0, 9), "y": (0, 9)}))
        with self.assertRaises(UnsupportedError):        # sum |w| is not retained: no certified bound
            dyn.evolve(wst, Kwt, 1, scope=wscope)
        with self.assertRaises(UnsupportedError):
            dyn.operator(B, dyn.polynomial(geo, {"x": {"a*x": 1}}, parameter="a"), truncate=bx)

    def test_dyn001_only_declared_geometry_and_exact_maps(self):
        rng = np.random.default_rng(6104)
        n = 20
        scope = mbk.Scope.from_arrays({"x": rng.integers(-3, 4, n), "y": rng.integers(-3, 4, n),
                                       "c": rng.integers(0, 3, n)})
        geo = ms.geometry(scope, numeric={"x": "m", "y": "m"}, categorical={"c": "colour"})
        B = ms.basis(geo, degree=2)
        for comps in ({"c": {"c": 1, "1": 1}}, {"x": {"c": 1}}, {"x": {"x*c": 1}}):
            with self.assertRaises(DomainError):          # no metric on a code: neither written nor read
                dyn.polynomial(geo, comps)
        with self.assertRaises(DomainError):              # a basis over a categorical code
            ms.basis(geo, terms=["1", "c"])
        with self.assertRaises(UnsupportedError):         # an interval (irrational) rotation is not exact dynamics
            dyn.affine(geo, ms.rotate("x", "y", degrees=30))
        with self.assertRaises(UnsupportedError):         # a value permutation is not affine dynamics
            dyn.affine(geo, ms.values("c", {0: 1, 1: 0}))
        with self.assertRaises(TypeError):                # floats are refused
            dyn.polynomial(geo, {"x": {"x": 0.5}})
        with self.assertRaises(ValueError):
            dyn.polynomial(geo, {"q": {"x": 1}})
        K = dyn.operator(B, dyn.affine(geo, ms.translate(x=1)))
        # a state of another scope, and an abstract geometry, have no evolution under this operator
        other = mbk.Scope.from_arrays({"x": rng.integers(-3, 4, n), "y": rng.integers(-3, 4, n)})
        ogeo = ms.geometry(other, numeric={"x": "m", "y": "m"})
        ost = ms.state(mbk.build(other, ms.basis(ogeo, degree=2).terms_for(other), "joint"), ogeo)
        with self.assertRaises(DomainError):
            dyn.evolve(ost, K, 1)
        ageo = ms.abstract(["x", "y", "c"], categorical=["c"])
        Ka = dyn.operator(ms.basis(ageo, degree=2), dyn.affine(ageo, ms.translate(x=1)))
        self.assertEqual(len(Ka.power(3).matrix()), 6)    # an abstract operator has powers
        st = ms.state(mbk.build(scope, B.terms_for(scope), "joint"), geo)
        with self.assertRaises(DomainError):
            dyn.evolve(st, Ka, 1)
        # the same scope with its coordinates declared in another order: equal digests and even equal full-basis
        # exponent rows, but other columns: refused
        swapped = ms.geometry(scope, numeric={"y": "m", "x": "m"}, categorical={"c": "colour"})
        sst = ms.state(mbk.build(scope, ms.basis(swapped, degree=2).terms_for(scope), "joint"), swapped)
        self.assertTrue(sst.basis.equal(B))
        for fn in (lambda: dyn.evolve(sst, K, 1), lambda: K.apply(sst), lambda: dyn.evolve(sst, K, 1, backend="cuda"),
                   lambda: K.recurrence({"x": 1}, state=sst), lambda: K.float_power(1, 1e-9, state=sst)):
            with self.assertRaises(DomainError):
                fn()
        # a state over another basis
        with self.assertRaises(DomainError):
            dyn.evolve(ms.state(mbk.build(scope, ms.basis(geo, degree=1).terms_for(scope), "joint"), geo), K, 1)

    def test_dyn001_budgets_at_the_limit_and_one_past(self):
        rng = np.random.default_rng(6105)
        scope, pts, _ = make_xyz(rng, n=30)
        geo = ms.geometry(scope, numeric={"x": "m", "y": "m", "z": "m"})
        B = ms.basis(geo, degree=2)
        st = ms.state(mbk.build(scope, B.terms_for(scope), "joint"), geo)
        P = dyn.polynomial(geo, {"x": {"x": F(1, 2), "y": 1}, "z": {"z": 1, "x": 2}})
        need = dyn.operator_need(B, P)
        self.assertEqual(need["tensor"], 100)
        dyn.operator(B, P, budget=Budget(tensor=100))
        with self.assertRaises(AlgebraBudgetError) as cm:
            dyn.operator(B, P, budget=Budget(tensor=99))
        self.assertEqual((cm.exception.resource, cm.exception.need, cm.exception.limit), ("tensor", 100, 99))
        K = dyn.operator(B, P)
        for method in ("stepping", "squaring", "modular"):
            n = dyn.need("evolve", K, steps=6, state=st, method=method)
            self.assertEqual((n["iterations"], n["output"], n["method"]), (6, 3 * 10, method))
            dyn.evolve(st, K, 6, method=method, budget=Budget(iterations=6, output=30, bytes=n["bytes"]))
            for kw, res in ((dict(iterations=5), "iterations"), (dict(output=29), "output"),
                            (dict(bytes=n["bytes"] - 1), "bytes")):
                with self.assertRaises(AlgebraBudgetError) as cm:
                    dyn.evolve(st, K, 6, method=method, budget=Budget(**kw))
                self.assertEqual(cm.exception.resource, res, (method, kw))
        rn = dyn.rows_need(scope, P, B, 4)
        dyn.evolve_rows(scope, P, B, 4, budget=Budget(bytes=rn["bytes"], iterations=4))
        with self.assertRaises(AlgebraBudgetError):
            dyn.evolve_rows(scope, P, B, 4, budget=Budget(bytes=rn["bytes"] - 1))
        with self.assertRaises(AlgebraBudgetError):
            dyn.evolve_rows(scope, P, B, 4, budget=Budget(iterations=3))


# ====================================================================== DYN-002
class Powers(Base):
    def operators(self):
        rng = np.random.default_rng(6201)
        scope, pts = make_xy(rng, n=24)
        geo = ms.geometry(scope, numeric={"x": "m", "y": "m"})
        B = ms.basis(geo, degree=2)
        ops = {
            "rotation": dyn.operator(B, dyn.affine(geo, ms.compose(ms.rotate("x", "y", cos=F(3, 5), sin=F(4, 5)),
                                                                   ms.translate(x=1)))),
            "affine": dyn.operator(B, dyn.affine(geo, ms.affine(["x", "y"], [[F(1, 2), 1], [-1, F(1, 3)]], [2, -1]))),
            "nonlinear": dyn.operator(ms.basis(geo, terms=["1", "x", "x^2", "y"]),
                                      dyn.polynomial(geo, {"x": {"x": F(-1, 2), "1": 1}, "y": {"y": 2, "x^2": -1}})),
        }
        return scope, pts, geo, B, ops

    # ------------------------------------------------------------------ acceptance 1: powers and semigroups
    def test_dyn002_zero_one_many_step_powers_and_semigroup_identities(self):
        scope, pts, geo, B, ops = self.operators()
        for name, K in ops.items():
            M = K.matrix()
            nb = len(M)
            for n in (0, 1, 2, 3, 5, 8, 13, 40):
                want = m_pow([[F(x) for x in row] for row in M], n)
                ps = [K.power(n, method="squaring"), K.power(n, method="modular"), K.power(n, backend="cuda"),
                      K.power(n)]
                for p in ps:
                    self.assertEqual(p.matrix(), [[norm(x) for x in row] for row in want], (name, n, p.info["method"]))
                    self.assertEqual(p.info["exponent"], n)
                    self.assertTrue(ps[0].equal(p))
                self.assertTrue(ps[1].info["checked"] and ps[2].info["checked"])
                self.assertEqual(ps[2].backend, "cuda-emulated" if dyn._gpu().device()["emulated"] else "cuda")
                self.assertEqual(ps[0].info["identity"], n == 0)
                if n == 0:
                    self.assertEqual(ps[0].matrix(), [[int(i == j) for j in range(nb)] for i in range(nb)])
                if n == 1:
                    self.assertEqual(ps[0].matrix(), M)
                self.assertLessEqual(ps[0].info["max_bits"], ps[0].info["bound_bits"])
            for m, n in ((0, 0), (0, 3), (1, 1), (2, 5), (7, 4)):
                self.assertTrue(dyn.semigroup(K, m, n)["holds"])
                self.assertTrue(dyn.semigroup(K, m, n, method="modular")["holds"])
                self.assertTrue(K.power(m).compose(K.power(n)).equal(K.power(m + n)))
                self.assertEqual(K.power(m).compose(K.power(n, backend="cuda")).matrix(), K.power(m + n).matrix())
        # composition of different operators: (K1^2 K2^3) as Fraction products
        K1, K2 = ops["rotation"], ops["affine"]
        prod = K1.power(2).compose(K2.power(3))
        want = m_mul(m_pow([[F(x) for x in r] for r in K1.matrix()], 2), m_pow([[F(x) for x in r] for r in K2.matrix()], 3))
        self.assertEqual(prod.matrix(), [[norm(x) for x in row] for row in want])
        self.assertFalse(prod.equal(K2.power(3).compose(K1.power(2))))       # these maps do not commute
        with self.assertRaises(DomainError):                                 # powers of another basis
            K1.power(1).compose(ops["nonlinear"].power(1))
        # the powers act on retained moments: K^n m_0 = the n-step evolution
        st = ms.state(mbk.build(scope, B.terms_for(scope), "joint"), geo)
        for n in (0, 1, 6):
            Pn = K1.power(n).matrix()
            for g in range(2):
                self.assertEqual([norm(v) for v in m_vec(Pn, [F(v) for v in st.moments(g)])],
                                 dyn.evolve(st, K1, n).moments(g))

    def test_dyn002_recurrences_charpoly_minimal_and_fast_terms(self):
        scope, pts, geo, B, ops = self.operators()
        st = ms.state(mbk.build(scope, B.terms_for(scope), "joint"), geo)
        for name, K in ops.items():
            M = [[F(x) for x in row] for row in K.matrix()]
            nb = len(M)
            cp = K.charpoly()
            self.assertEqual(cp.coefficients, [norm(c) for c in faddeev_leverrier(M)], name)
            self.assertTrue(cp.info["checked"] and cp.info["primes"] >= 2)
            c = [1] + [0] * (nb - 2) + [-2]
            v = list(range(1, nb + 1))
            seq = [sum((F(ci) * x for ci, x in zip(c, m_vec(m_pow(M, t), [F(x) for x in v]))), F(0)) for t in range(3 * nb)]
            hank = rank([[seq[i + j] for j in range(nb)] for i in range(nb)])
            for kind in ("charpoly", "minimal"):
                rec = K.recurrence(c, vector=v, kind=kind)
                d, r = rec.info["degree"], rec.coefficients
                self.assertEqual(d, nb if kind == "charpoly" else hank, (name, kind))
                self.assertEqual(rec.initial, [norm(x) for x in seq[:d]])
                for t in range(3 * nb - d):                       # s_(t+d) = sum_k r_k s_(t+k)
                    self.assertEqual(seq[t + d], sum((F(r[k]) * seq[t + k] for k in range(d)), F(0)), (name, kind, t))
                for T in (0, 1, d, 2 * nb + 3, 60):
                    want = sum((F(ci) * x for ci, x in zip(c, m_vec(m_pow(M, T), [F(x) for x in v]))), F(0))
                    self.assertEqual(rec.term(T), norm(want), (name, kind, T))
            if name == "rotation":
                self.assertEqual(cp.coefficients[0], 1)               # det K = 1 for a rigid motion
        # a state's group moments as the vector, a functional over the basis terms
        K = ops["rotation"]
        rec = K.recurrence({"x^2": 1, "y^2": 1}, state=st, group=1)
        for T in (0, 7, 25):
            m = dyn.evolve(st, K, T).moments(1)
            self.assertEqual(rec.term(T), norm(F(m[B.index("x^2")]) + F(m[B.index("y^2")])))
        zero = K.recurrence({"x": 1}, vector=[0] * len(B))
        self.assertEqual((zero.info["degree"], zero.term(1000)), (0, 0))

    def test_dyn002_parameterized_evolution_mixtures_and_polynomial_parameters(self):
        rng = np.random.default_rng(6202)
        scope, pts = make_xy(rng, n=20)
        geo = ms.geometry(scope, numeric={"x": "m", "y": "m"})
        B = ms.basis(geo, degree=2)
        st = ms.state(mbk.build(scope, B.terms_for(scope), "joint"), geo)
        maps = [(ms.translate(x=1, y=-1), o_affine([0, 1], [[1, 0], [0, 1]], [1, -1])),
                (ms.rotate("x", "y", degrees=90), o_affine([0, 1], [[0, -1], [1, 0]], [0, 0])),
                (ms.scale(x=F(1, 2)), o_affine([0, 1], [[F(1, 2), 0], [0, 1]], [0, 0]))]
        weights = [F(1, 2), F(1, 3), F(1, 6)]
        Ks = [dyn.operator(B, dyn.affine(geo, t)) for t, _ in maps]
        Mx = dyn.mixture(list(zip(weights, Ks)))
        self.assertEqual(Mx.info["source"], "mixture")
        want = [[sum((w * F(K.matrix()[i][j]) for w, K in zip(weights, Ks)), F(0)) for j in range(len(B))] for i in range(len(B))]
        self.assertEqual(Mx.matrix(), [[norm(x) for x in r] for r in want])
        for t in (0, 1, 3):                                       # the expectation over every choice sequence
            ev, evc = dyn.evolve(st, Mx, t), dyn.evolve(st, Mx, t, backend="cuda")
            for g in range(2):
                exp = [F(0)] * len(B)
                for seq in itertools.product(range(3), repeat=t):
                    prob = F(1)
                    p = pts[g]
                    for j in seq:
                        prob *= weights[j]
                        p = [maps[j][1](q) for q in p]
                    exp = [e + prob * m for e, m in zip(exp, moments(p, B.terms))]
                self.assertEqual(ev.moments(g), [norm(x) for x in exp])
                self.assertEqual(evc.moments(g), ev.moments(g))
        for bad in ([(F(1, 2), Ks[0]), (F(1, 3), Ks[1])], [(F(3, 2), Ks[0]), (F(-1, 2), Ks[1])]):
            with self.assertRaises(DomainError):                  # convex weights only
                dyn.mixture(bad)
        with self.assertRaises(DomainError):                      # one basis
            dyn.mixture([(F(1, 2), Ks[0]), (F(1, 2), dyn.operator(ms.basis(geo, degree=1), dyn.affine(geo, maps[0][0])))])
        # a polynomial parameter: x' = a x + 1, y' = y + a x (K(a) has degree 2 in a on this basis)
        Pa = dyn.polynomial(geo, {"x": {"a*x": 1, "1": 1}, "y": {"y": 1, "a*x": 1}}, parameter="a")
        Ka = dyn.operator(B, Pa)
        self.assertEqual(Ka.info["pdegree"], 2)
        for t in (0, 1, 3):
            ev = dyn.evolve(st, Ka, t)
            self.assertEqual(ev.info["pdegree"], 2 * t)
            for theta in (F(0), F(1, 2), F(-3), F(5, 7)):
                f = o_affine([0, 1], [[theta, 0], [theta, 1]], [1, 0])
                at = ev.at(theta)
                for g in range(2):
                    want = [norm(v) for v in moments(evolve_points(pts[g], f, t), B.terms)]
                    self.assertEqual(at.moments(g), want, (t, theta))
                    self.assertEqual(dyn.evolve(st, Ka.at(theta), t).moments(g), want)
                    if t:                                 # the polynomial in theta, evaluated here
                        poly = ev.moments(g)
                        self.assertEqual([norm(sum((F(c) * theta ** d for d, c in enumerate(cs)), F(0))) for cs in poly],
                                         want)
        with self.assertRaises(UnsupportedError):                 # parametric moments are stepped
            dyn.evolve(st, Ka, 2, method="squaring")
        with self.assertRaises(UnsupportedError):
            Ka.power(2)

    # ------------------------------------------------------------------ acceptance 2: budgets and stability
    def test_dyn002_budgets_bound_coefficient_growth_iterations_and_output(self):
        scope, pts, geo, B, ops = self.operators()
        K = ops["rotation"]
        nb = len(B)
        st = ms.state(mbk.build(scope, B.terms_for(scope), "joint"), geo)
        prev = 0
        for n in (10, 20, 40):                                    # the proven coefficient bound grows with n
            need = dyn.need("power", K, steps=n, method="squaring")
            p = K.power(n, method="squaring", budget=Budget(bytes=need["bytes"]))
            self.assertEqual(p.info["bound_bits"], n * K.info["norm_bits"])
            self.assertLessEqual(p.info["max_bits"], p.info["bound_bits"])
            self.assertGreater(need["bytes"], prev)
            prev = need["bytes"]
            with self.assertRaises(AlgebraBudgetError) as cm:
                K.power(n, method="squaring", budget=Budget(bytes=need["bytes"] - 1))
            self.assertEqual((cm.exception.resource, cm.exception.need), ("bytes", need["bytes"]))
        for method in ("squaring", "modular"):
            K.power(30, method=method, budget=Budget(iterations=30, output=nb * nb))
            with self.assertRaises(AlgebraBudgetError) as cm:
                K.power(31, method=method, budget=Budget(iterations=30))
            self.assertEqual((cm.exception.resource, cm.exception.need, cm.exception.limit), ("iterations", 31, 30))
            with self.assertRaises(AlgebraBudgetError) as cm:
                K.power(30, method=method, budget=Budget(output=nb * nb - 1))
            self.assertEqual(cm.exception.resource, "output")
        dn = dyn.need("power", K, steps=30, backend="cuda")
        self.assertEqual(dn["method"], "modular")
        K.power(30, backend="cuda", budget=Budget(bytes=dn["bytes"]))
        with self.assertRaises(AlgebraBudgetError):
            K.power(30, backend="cuda", budget=Budget(bytes=dn["bytes"] - 1))
        en = dyn.need("evolve", K, steps=12, state=st, backend="cuda")
        dyn.evolve(st, K, 12, backend="cuda", budget=Budget(bytes=en["bytes"], iterations=12, output=2 * nb))
        for kw in (dict(bytes=en["bytes"] - 1), dict(iterations=11), dict(output=2 * nb - 1)):
            with self.assertRaises(AlgebraBudgetError):
                dyn.evolve(st, K, 12, backend="cuda", budget=Budget(**kw))
        rec = K.recurrence({"x": 1}, state=st, group=0)
        tn = rec.term_need(500)
        rec.term(500, budget=Budget(iterations=500, bytes=tn["bytes"]))
        with self.assertRaises(AlgebraBudgetError):
            rec.term(501, budget=Budget(iterations=500))
        with self.assertRaises(AlgebraBudgetError):
            rec.term(500, budget=Budget(bytes=tn["bytes"] - 1))
        cn = dyn.need("charpoly", K)
        K.charpoly(budget=Budget(bytes=cn["bytes"]))
        with self.assertRaises(AlgebraBudgetError):
            K.charpoly(budget=Budget(bytes=cn["bytes"] - 1))
        # AUTO reports its estimates and picks the cheaper one
        p = K.power(200)
        i = p.info
        self.assertEqual(i["method"], "squaring" if i["cost_squaring"] <= i["cost_modular"] else "modular")
        # the estimates count the work each method does: on a sparse (degree-triangular) operator the binary method's
        # big-integer products skip N's zeros while the residue products loop over every (i, j, k) per prime and
        # Garner's reconstruction is O(k^2) per value, so at 3,000 steps squaring stays the cheaper (measured about
        # 30x), where an estimate from dense limb counts alone picked modular
        rng = np.random.default_rng(6203)
        names = ["a", "b", "c", "d"]
        sscope = mbk.Scope.from_arrays({v: rng.integers(-50, 51, 40) for v in names})
        sgeo = ms.geometry(sscope, numeric={v: "m" for v in names})
        sparse = dyn.operator(ms.basis(sgeo, degree=2), dyn.affine(sgeo, ms.compose(
            ms.rotate("a", "b", cos=F(3, 5), sin=F(4, 5), center=(1, 0)),
            ms.affine(["c", "d"], [[F(1, 2), 0], [0, F(1, 2)]], [1, 1]))))
        sst = ms.state(mbk.build(sscope, ms.basis(sgeo, degree=2).terms_for(sscope), "joint"), sgeo)
        for request, kw in (("power", {}), ("evolve", dict(state=sst))):
            self.assertEqual(dyn.need(request, sparse, steps=3000, **kw)["method"], "squaring", request)

    def test_dyn002_floating_powers_certified_or_diagnosed_unstable(self):
        scope, pts, geo, B, ops = self.operators()
        st = ms.state(mbk.build(scope, B.terms_for(scope), "joint"), geo)
        contracting = dyn.operator(B, dyn.affine(geo, ms.affine(["x", "y"], [[F(1, 2), F(1, 4)], [F(-1, 4), F(1, 3)]],
                                                                [1, F(-1, 2)])))
        for K, n in ((contracting, 64), (ops["rotation"], 40)):
            fp = K.float_power(n, tolerance=1e-8)
            self.assertTrue(fp.info["stable"] and fp.info["approximate"] and fp.info["certified"])
            self.assertEqual(fp.info["reached"], n)
            exact = K.power(n).matrix()
            for i in range(len(B)):
                for j in range(len(B)):
                    v, r = F(fp.values[i, j]), F(fp.bounds[i, j])
                    self.assertLessEqual(abs(F(exact[i][j]) - v), r)       # the rigorous bound holds exactly
            self.assertLessEqual(fp.info["error"], 1e-8)
            ref = np.linalg.matrix_power(np.array([[float(x) for x in r] for r in K.matrix()]), n)
            self.assertTrue(np.allclose(fp.values, ref, rtol=1e-9, atol=1e-9))   # NumPy at a stated tolerance
            fm = K.float_power(n, tolerance=1e-6, norm="rel", state=st)
            for g in range(2):
                ex = dyn.evolve(st, K, n).moments(g)
                for a in range(len(B)):
                    self.assertLessEqual(abs(F(ex[a]) - F(fm.values[g, a])), F(fm.bounds[g, a]))
        expanding = dyn.operator(B, dyn.affine(geo, ms.affine(["x", "y"], [[F(3, 2), F(1, 3)], [0, F(5, 4)]], [1, 1])))
        with self.assertRaises(dyn.UnstableError) as cm:
            expanding.float_power(200, tolerance=1e-9)
        info = cm.exception.info
        self.assertFalse(info["stable"])
        self.assertGreater(info["error"], 1e-9)
        self.assertLess(info["reached"], 200)
        self.assertGreater(info["magnitude"], 1e20)
        with self.assertRaises(dyn.UnstableError) as cm:            # past the double range: diagnosed, not returned
            expanding.float_power(2000, tolerance=1e-3, norm="rel")
        self.assertEqual(cm.exception.info["error"], float("inf"))
        self.assertTrue(expanding.float_power(10, tolerance=1e-3, norm="rel").info["stable"])
        with self.assertRaises(AlgebraBudgetError):
            contracting.float_power(65, tolerance=1e-8, budget=Budget(iterations=64))

    # ------------------------------------------------------------------ CUDA: batched kernels equal the CPU
    def test_dyn002_cuda_batched_application_and_modular_powers_equal_the_cpu(self):
        scope, pts, geo, B, ops = self.operators()
        st = ms.state(mbk.build(scope, B.terms_for(scope), "joint"), geo)
        gpu = dyn._gpu()
        nb = len(B)
        Ks = [ops["rotation"], ops["affine"]]
        # two operators in one view (the batch dimension), every group
        num = np.zeros((2, nb * nb, 2), dtype=np.int64)
        for k, K in enumerate(Ks):
            N, _ = K.numerators()
            for i in range(nb):
                for j in range(nb):
                    x = N[i][j]
                    num[k, i * nb + j] = (x & ((1 << 64) - 1)) - (1 << 64) * ((x >> 63) & 1), x >> 64
        qv = ms._QueryView()
        ms._call("tr_sym_state_get_view", st._open(), C.byref(qv))
        view = dyn._ApplyView(nb, 2, int(qv.ngroups), num.ctypes.data, qv.moments)
        total = 2 * int(qv.ngroups) * nb
        cpu, dev = (ms._Wide * total)(), (ms._Wide * total)()
        r = dyn._Refusal()
        self.assertEqual(dyn._call("tr_dyn_apply_view_eval", C.byref(view), cpu, C.byref(Budget().raw), C.byref(r)), 0)
        gpu.run("tr_mgpu_dyn_apply", C.byref(view), dev, C.byref(Budget().raw), C.byref(r), refusal=r)
        self.assertEqual(bytes(cpu), bytes(dev))
        for k, K in enumerate(Ks):
            D = K.info["den"]
            for g in range(int(qv.ngroups)):
                self.assertEqual([norm(F(ms._wide(cpu[(k * int(qv.ngroups) + g) * nb + i]), D)) for i in range(nb)],
                                 dyn.evolve(st, K, 1).moments(g))
        pr = (dyn._GFP * 5)()
        dyn._call("tr_dyn_primes", 5, pr)
        for e in (0, 1, 2, 7, 64, 1000):
            mv = dyn._ModView(nb, 2, 5, e, num.ctypes.data, C.cast(pr, C.c_void_p))
            a, b = np.zeros(2 * 5 * nb * nb, dtype=np.uint64), np.zeros(2 * 5 * nb * nb, dtype=np.uint64)
            self.assertEqual(dyn._call("tr_dyn_powmod_view_eval", C.byref(mv), a.ctypes.data, C.byref(Budget().raw),
                                       C.byref(r)), 0)
            gpu.run("tr_mgpu_dyn_powmod", C.byref(mv), b.ctypes.data, C.byref(Budget().raw), C.byref(r), refusal=r)
            self.assertTrue(np.array_equal(a, b), e)
            G = int(qv.ngroups)
            a2, b2 = np.zeros(2 * 5 * G * nb, dtype=np.uint64), np.zeros(2 * 5 * G * nb, dtype=np.uint64)
            self.assertEqual(dyn._call("tr_dyn_evolvemod_view_eval", C.byref(mv), qv.moments, G, a2.ctypes.data,
                                       C.byref(Budget().raw), C.byref(r)), 0)
            gpu.run("tr_mgpu_dyn_evolvemod", C.byref(mv), qv.moments, G, b2.ctypes.data, C.byref(Budget().raw),
                    C.byref(r), refusal=r)
            self.assertTrue(np.array_equal(a2, b2), e)
            # the residues are those of the exact power (oracle: Python ints mod p)
            if e <= 7:
                for k, K in enumerate(Ks):
                    Pn = m_pow([[F(x) for x in row] for row in K.numerators()[0]], e)
                    for q in range(5):
                        p = int(pr[q].p)
                        blk = a[(k * 5 + q) * nb * nb:(k * 5 + q + 1) * nb * nb]
                        self.assertEqual([int(x) for x in blk], [int(Pn[i][j]) % p for i in range(nb) for j in range(nb)])
        # the device's residues are adopted only when the check primes agree
        K = ops["affine"]
        n = 9
        pr2, mv2 = K._mod_setup(n, None)
        res = np.zeros(len(pr2) * nb * nb, dtype=np.uint64)
        gpu.run("tr_mgpu_dyn_powmod", C.byref(mv2), res.ctypes.data, C.byref(Budget().raw), C.byref(r), refusal=r)
        bad = res.copy()
        bad[(len(pr2) - 1) * nb * nb + 3] ^= np.uint64(1)         # one residue of the check prime corrupted
        err, ref = C.c_int32(), dyn._Refusal()
        h = dyn._call("tr_dyn_power_adopt", K._open(), n, pr2, len(pr2), bad.ctypes.data, 1, C.byref(Budget().raw),
                      C.byref(ref), C.byref(err))
        self.assertFalse(h)
        self.assertEqual(err.value, -1)
        big = res.copy()
        big[5] = int(pr2[0].p)                                    # not a residue: p itself, for the first prime
        h = dyn._call("tr_dyn_power_adopt", K._open(), n, pr2, len(pr2), big.ctypes.data, 1, C.byref(Budget().raw),
                      C.byref(ref), C.byref(err))
        self.assertFalse(h)
        self.assertEqual(err.value, -1)
        h = dyn._call("tr_dyn_power_adopt", K._open(), n, pr2, len(pr2) - 1, res.ctypes.data, 1, C.byref(Budget().raw),
                      C.byref(ref), C.byref(err))
        self.assertFalse(h)                                       # too few primes for the proven bound
        self.assertEqual(dyn.primes(3), [int(pr[q].p) for q in range(3)])


if __name__ == "__main__":
    unittest.main()
