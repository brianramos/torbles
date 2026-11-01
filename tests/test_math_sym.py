# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""SYM-001 / SYM-002: closed-basis transformation actions and certified data symmetries (torbles/math_sym.c,
math_sym_gpu.cu, math_sym.py).

Oracles, independent of the implementation: every transformation written here as an explicit map of points with
Python Fractions (rotations x' = cx + c (x - cx) - s (y - cy), ...), applied to every selected complete row, and the
moments sum_r w_r prod_i x'_i^a_i formed directly; a 60-digit Decimal Taylor series for the cosine and sine of
irrational rotations (interval containment); DuckDB SQL for the moment sums of transformed rows of an engine table;
collections.Counter over the row tuples for multiset preservation and the witness rows; breadth-first orbits written
here; exhaustive evaluation of the queries over the rows. Every native request and verification is also run on the
CUDA path (emulated here with TORBLES_EMU_SCRAMBLE=1) and compared for equality. Seeds are fixed.
"""
import itertools
import subprocess
import sys
import unittest
from collections import Counter
from decimal import Decimal, getcontext
from fractions import Fraction

import numpy as np
import pandas as pd

import torbles
from torbles import algq, math_bank as mbk, math_sym as ms, math_tensor as mt
from torbles.math_budgets import Budget
from torbles.math_certificates import StaleCertificateError
from torbles.math_domains import DomainError
from torbles.math_errors import UnsupportedError
from torbles.mathematics import AlgebraBudgetError

F = Fraction


# ---------------------------------------------------------------------- oracles: explicit point maps (value units)
def o_rotate(i, j, c, s, center=(0, 0)):
    cx, cy = F(center[0]), F(center[1])

    def f(p):
        q = list(p)
        x, y = p[i] - cx, p[j] - cy
        q[i], q[j] = cx + c * x - s * y, cy + s * x + c * y
        return tuple(q)
    return f


def o_translate(t):
    return lambda p: tuple(p[i] + F(t.get(i, 0)) for i in range(len(p)))


def o_scale(f_):
    return lambda p: tuple(p[i] * F(f_.get(i, 1)) for i in range(len(p)))


def o_reflect(a):
    return lambda p: tuple(2 * F(a[i]) - p[i] if i in a else p[i] for i in range(len(p)))


def o_permute(src):                 # src: {target index: source index}
    return lambda p: tuple(p[src[i]] if i in src else p[i] for i in range(len(p)))


def o_affine(idx, A, b):
    def f(p):
        q = list(p)
        for r, i in enumerate(idx):
            q[i] = sum(F(A[r][k]) * p[idx[k]] for k in range(len(idx))) + F(b[r])
        return tuple(q)
    return f


def o_compose(*fs):
    def h(p):
        for f in reversed(fs):
            p = f(p)
        return p
    return h


def mono(p, t):
    v = F(1)
    for x, e in zip(p, t):
        v *= F(x) ** e
    return v


def moments(points, terms, weights=None):
    w = weights or [1] * len(points)
    return [sum((F(wi) * mono(p, t) for p, wi in zip(points, w)), F(0)) for t in terms]


def dpow(x, e):
    """x ** e for Decimals with x ** 0 = 1 (Decimal refuses 0 ** 0)"""
    r = Decimal(1)
    for _ in range(e):
        r *= x
    return r


def norm(v):
    """an exact value as the module returns it: int when integral"""
    v = F(v)
    return v.numerator if v.denominator == 1 else v


# ---------------------------------------------------------------------- data
def make_xyz(rng, n=60, G=3, nulls=0.1, lo=-6, hi=7):
    vals = {"x": rng.integers(lo, hi, n), "y": rng.integers(lo, hi, n), "z": rng.integers(lo, hi, n),
            "w": rng.integers(-3, 5, n), "u": rng.integers(0, 3, n)}
    valid = {"z": rng.random(n) >= nulls}
    groups = rng.integers(0, G, n)
    rows = rng.random(n) < 0.9
    scope = mbk.Scope.from_arrays(vals, valid=valid, groups=groups, ngroups=G, rows=rows)
    pts = [[] for _ in range(G)]
    wts = [[] for _ in range(G)]
    for r in range(n):
        if rows[r] and valid["z"][r]:
            pts[groups[r]].append((F(int(vals["x"][r])), F(int(vals["y"][r])), F(int(vals["z"][r]))))
            wts[groups[r]].append(int(vals["w"][r]))
    return scope, pts, wts


class Base(unittest.TestCase):
    def check_state(self, st, A, f, pts, wts=None, mean=False):
        """the transformed moments of every group, CPU and CUDA, against the direct transformed rows"""
        terms = st.basis.terms
        for g in range(st.info["groups"]):
            cpu = st.transformed(A, g, mean=mean)
            self.assertEqual(cpu, st.transformed(A, g, backend="cuda", mean=mean))
            img = [f(p) for p in pts[g]]
            want = moments(img, terms, None if wts is None else wts[g])
            if mean:
                den = sum(wts[g]) if wts is not None else len(pts[g])
                want = [None if den == 0 else norm(F(v) / den) for v in want]
            else:
                want = [norm(v) for v in want]
            self.assertEqual(cpu, want, (g, A))


# ====================================================================== SYM-001
class Actions(Base):
    # ------------------------------------------------------------------ acceptance 1: composition and identity
    def test_sym001_composition_and_identity_actions_match_transformed_rows(self):
        rng = np.random.default_rng(5101)
        scope, pts, wts = make_xyz(rng)
        geo = ms.geometry(scope, numeric={"x": "m", "y": "m", "z": "m"})
        B = ms.basis(geo, degree=3)
        self.assertEqual(len(B), 20)
        bank = mbk.build(scope, B.terms_for(scope), "joint")
        tensor = mt.moments(scope, ["1", "x", "y", "z"], order=3)
        states = [ms.state(bank, geo), ms.state(tensor, geo), ms.state(mt.moments(scope, ["1", "x", "y", "z"], order=3,
                                                                                   repr="sparse"), geo)]
        for st in states:
            self.assertTrue(st.basis.equal(B))
        decl = {
            "R1": (ms.rotate("x", "y", cos=F(3, 5), sin=F(4, 5), center=(1, -2)), o_rotate(0, 1, F(3, 5), F(4, 5), (1, -2))),
            "R2": (ms.rotate("y", "z", cos=F(5, 13), sin=F(-12, 13)), o_rotate(1, 2, F(5, 13), F(-12, 13))),
            "T": (ms.translate(x=F(1, 2), z=-3), o_translate({0: F(1, 2), 2: -3})),
            "S": (ms.scale(y=-2, z=F(1, 3)), o_scale({1: -2, 2: F(1, 3)})),
            "P": (ms.permute({"x": "z", "y": "x", "z": "y"}), o_permute({0: 2, 1: 0, 2: 1})),
            "AF": (ms.affine(["x", "y"], [[1, 2], [0, F(1, 2)]], [F(1, 3), 0]), o_affine([0, 1], [[1, 2], [0, F(1, 2)]], [F(1, 3), 0])),
            "F": (ms.reflect(z=F(1, 2)), o_reflect({2: F(1, 2)})),
            "R90": (ms.rotate("z", "x", degrees=-270), o_rotate(2, 0, 0, 1)),
        }
        acts = {k: ms.action(B, t, geo) for k, (t, _) in decl.items()}
        self.assertTrue(acts["P"].info["permutation"])
        self.assertTrue(acts["R90"].info["monomial"])
        for k, (t, f) in decl.items():                             # each action alone
            for st in states:
                self.check_state(st, acts[k], f, pts)
        for a, b in itertools.permutations(["R1", "R2", "T", "S", "P", "AF", "F"], 2):
            prod = acts[a].compose(acts[b])                       # M_a M_b
            direct = ms.action(B, ms.compose(decl[a][0], decl[b][0]), geo)   # the action of the composed map
            self.assertTrue(prod.equal(direct), (a, b))
            self.check_state(states[(len(a) + len(b)) % 3], prod, o_compose(decl[a][1], decl[b][1]), pts)
        # a chain of five, as one composed map and as a product of five actions
        chain = ["R1", "T", "P", "R2", "S"]
        prod = acts[chain[0]]
        for k in chain[1:]:
            prod = prod.compose(acts[k])
        self.assertTrue(prod.equal(ms.action(B, ms.compose(*[decl[k][0] for k in chain]), geo)))
        self.check_state(states[0], prod, o_compose(*[decl[k][1] for k in chain]), pts)
        # identity: I itself, the identity transformation, and M_T M_T^-1 = M_T^-1 M_T = I
        I = ms.identity_action(B)
        self.assertTrue(I.info["identity"] and I.info["permutation"])
        self.assertTrue(ms.action(B, ms.identity(), geo).equal(I))
        self.check_state(states[1], I, lambda p: p, pts)
        inverses = {"R1": ms.rotate("x", "y", cos=F(3, 5), sin=F(-4, 5), center=(1, -2)),
                    "R2": ms.rotate("y", "z", cos=F(5, 13), sin=F(12, 13)), "T": ms.translate(x=F(-1, 2), z=3),
                    "S": ms.scale(y=F(-1, 2), z=3), "P": ms.permute({"z": "x", "x": "y", "y": "z"}),
                    "AF": ms.affine(["x", "y"], [[1, -4], [0, 2]], [F(-1, 3), 0]), "F": ms.reflect(z=F(1, 2)),
                    "R90": ms.rotate("z", "x", degrees=270)}
        for k, inv in inverses.items():
            Ai = ms.action(B, inv, geo)
            self.assertTrue(acts[k].compose(Ai).equal(I), k)
            self.assertTrue(Ai.compose(acts[k]).equal(I), k)
            self.assertTrue(ms.action(B, ms.compose(decl[k][0], inv), geo).equal(I), k)
        self.assertFalse(acts["R1"].equal(acts["R2"]))
        # weighted moments (a TENSOR-001 weight) and means over the weight sum
        wt = ms.state(mt.moments(scope, ["1", "x", "y", "z"], order=3, weight="w"), geo)
        self.assertTrue(wt.info["weighted"])
        for k in ("R1", "AF"):
            self.check_state(wt, acts[k], decl[k][1], pts, wts)
            self.check_state(wt, acts[k], decl[k][1], pts, wts, mean=True)
        self.check_state(states[0], acts["T"], decl["T"][1], pts, mean=True)

    def test_sym001_batched_requests_functionals_and_means_cpu_equals_cuda(self):
        rng = np.random.default_rng(5102)
        scope, pts, _ = make_xyz(rng, n=80, G=4)
        geo = ms.geometry(scope, numeric={"x": "m", "y": "m", "z": "m"})
        B = ms.basis(geo, degree=2)
        st = ms.state(mbk.build(scope, B.terms_for(scope), "joint"), geo)
        decl = [(ms.rotate("x", "y", cos=F(8, 17), sin=F(15, 17)), o_rotate(0, 1, F(8, 17), F(15, 17))),
                (ms.translate(y=7), o_translate({1: 7})),
                (ms.permute({"x": "y", "y": "x"}), o_permute({0: 1, 1: 0})),
                (ms.rotate("x", "z", degrees=33), None)]
        acts = [ms.action(B, t, geo) for t, _ in decl]
        funcs = [{"x^2": 1, "y^2": 1}, {"1": 5, "x*y": -3, "z": 2}, {"z^2": 7}]
        reqs = []
        for a in (None, 0, 1, 2, 3):
            for g in range(4):
                for term in ("1", "x", "y*z", "z^2"):
                    reqs.append(dict(action=a, group=g, term=term))
                for f in range(3):
                    reqs.append(dict(action=a, group=g, functional=f, mean=bool(g % 2)))
        cpu = st.query(reqs, actions=acts, functionals=funcs)
        gpu = st.query(reqs, actions=acts, functionals=funcs, backend="cuda")
        self.assertEqual(cpu, gpu)
        names = ("x", "y", "z")
        c33, s33 = np.cos(np.radians(33.0)), np.sin(np.radians(33.0))
        rot33 = lambda p: (c33 * float(p[0]) - s33 * float(p[2]), float(p[1]), s33 * float(p[0]) + c33 * float(p[2]))
        for q, r in zip(reqs, cpu):
            if q["action"] == 3:                                  # an interval: it encloses the float64 value
                self.assertEqual(r["carrier"], "interval")
                lo, hi = r["value"]
                img = [rot33(p) for p in pts[q["group"]]]
                terms = ([ms._exponents(names, q["term"])], [1]) if "term" in q else (
                    [ms._exponents(names, t) for t in funcs[q["functional"]]], list(funcs[q["functional"]].values()))
                v = sum(c * sum(float(np.prod([p[i] ** e for i, e in enumerate(t)])) for p in img)
                        for t, c in zip(*terms))
                if q.get("mean"):
                    v /= len(img)
                tol = 1e-9 * max(1.0, abs(v))
                self.assertTrue(lo - tol <= v <= hi + tol and hi - lo <= tol, (q, lo, v, hi))
                continue
            f = (lambda p: p) if q["action"] is None else decl[q["action"]][1]
            img = [f(p) for p in pts[q["group"]]]
            if "term" in q:
                poly = {ms._exponents(names, q["term"]): 1}
            else:
                poly = {ms._exponents(names, t): c for t, c in funcs[q["functional"]].items()}
            val = sum((c * sum((mono(p, t) for p in img), F(0)) for t, c in poly.items()), F(0))
            if q.get("mean"):
                val = None if not img else val / len(img)
            self.assertEqual(r["value"], None if val is None else norm(val), q)
        # MEAN without a constant term: refused per request
        hom = ms.state(mt.moments(scope, ["x", "y"], order=2), geo)
        raw = hom.query([dict(group=0, term="x^2", mean=True), dict(group=0, term="x*y")], raw=True)
        self.assertEqual([r["status"] for r in raw], [-12, 0])
        with self.assertRaises(UnsupportedError):
            hom.query([dict(group=0, term="x^2", mean=True)], backend="cuda")

    # ------------------------------------------------------------------ acceptance 2: declared geometry only
    def test_sym001_only_declared_geometry_codes_have_no_metric(self):
        frame = pd.DataFrame({
            "k": pd.Series([int(v) for v in [0, 1, 0, 1, 2, 2, 0, 1]], dtype=object),
            "x": pd.Series(["1.25", "-2.50", "3.00", "0.75", "-1.00", "2.25", "0.00", "-3.50"], dtype=object),
            "y": pd.Series([4, -1, 2, 0, 3, -2, 1, 5], dtype=object),
            "h": pd.Series([10, 20, 30, 40, 50, 60, 70, 80], dtype=object),
            "c": pd.Series(["red", "green", "blue", "red", "green", "blue", "red", "blue"], dtype=object)})
        db = torbles.from_ddl("CREATE TABLE t (k BIGINT, x DECIMAL(10,2), y BIGINT, h BIGINT, c VARCHAR)", {"t": frame})
        t = db["t"]
        scope = mbk.Scope.from_table(t, ["x", "y", "h"], group_by=["k"])
        geo = ms.geometry(scope, numeric={"x": "m", "y": "m"})
        self.assertEqual(geo.scales, (2, 0))
        B = ms.basis(geo, degree=2)
        st = ms.state(mbk.build(scope, B.terms_for(scope), "joint"), geo)
        # a rotation, translation and scaling in value units act on the logical integers (x: cents, y: units)
        pts = [[] for _ in range(scope.groups)]
        for k, x, y in zip(frame["k"], frame["x"], frame["y"]):
            g = [i for i in range(scope.groups) if scope.key_values(i) == (int(k),)][0]
            pts[g].append((F(x), F(int(y))))
        to_logical = lambda p: (p[0] * 100, p[1])
        cases = [(ms.rotate("x", "y", cos=F(3, 5), sin=F(4, 5)), o_rotate(0, 1, F(3, 5), F(4, 5))),
                 (ms.translate(x="1.25", y=-2), o_translate({0: F(5, 4), 1: -2})),
                 (ms.scale(x=Decimal("0.5")), o_scale({0: F(1, 2)})),
                 (ms.permute({"x": "y", "y": "x"}), o_permute({0: 1, 1: 0}))]
        for tr, f in cases:
            A = ms.action(B, tr, geo)
            logical = [[to_logical(f(p)) for p in pts[g]] for g in range(scope.groups)]
            for g in range(scope.groups):
                want = [norm(v) for v in moments(logical[g], B.terms)]
                self.assertEqual(st.transformed(A, g), want)
                self.assertEqual(st.transformed(A, g, backend="cuda"), want)
        # the permutation of coordinates of different scales is not an index permutation of the basis
        self.assertFalse(ms.action(B, ms.permute({"x": "y", "y": "x"}), geo).info["permutation"])
        # DuckDB: 25 x the moments of x' = (3/5 x - 4/5 y) (logical: 100 x' = 3/5 xl - 80 y), grouped by k
        sql = ("SELECT k, COUNT(*), SUM(CAST(3 * CAST(x * 100 AS BIGINT) - 400 * y AS HUGEINT)), "
               "SUM(CAST(3 * CAST(x * 100 AS BIGINT) - 400 * y AS HUGEINT) * (3 * CAST(x * 100 AS BIGINT) - 400 * y)), "
               "SUM(CAST(4 * CAST(x * 100 AS BIGINT) + 300 * y AS HUGEINT) * (3 * CAST(x * 100 AS BIGINT) - 400 * y)) "
               "FROM t GROUP BY k")
        duck = {int(r[0]): [int(v) for v in r[1:]] for r in db.con.execute(sql).fetchall()}
        A = ms.action(B, cases[0][0], geo)
        for g in range(scope.groups):
            key = scope.key_values(g)[0]
            m = dict(zip(B.terms, st.transformed(A, g)))
            n_, s1, s2, sxy = duck[key]
            self.assertEqual(m[(0, 0)], n_)
            self.assertEqual(m[(1, 0)] * 5, s1)
            self.assertEqual(m[(2, 0)] * 25, s2)
            self.assertEqual(m[(1, 1)] * 2500, sxy)          # y'l = 4/500 xl + 3/5 y: 500 y'l = 4 xl + 300 y
        # undeclared columns: h is a scope column but no coordinate: refused natively, also through a bank basis
        for tr in (ms.rotate("x", "h", cos=F(3, 5), sin=F(4, 5)), ms.translate(h=1), ms.permute({"x": "h", "h": "x"}),
                   ms.affine(["x", "h"], [[1, 0], [0, 1]], [0, 0])):
            with self.assertRaises(DomainError):
                ms.action(B, tr, geo)
        with self.assertRaises(DomainError):
            ms.state(mbk.build(scope, ["1", "x", "h", "x*h"], "joint"), geo)
        with self.assertRaises(ValueError):                   # not even a scope column
            ms.action(B, ms.translate(q=1), geo)
        # incommensurable units: a rotation needs one unit
        geo2 = ms.geometry(scope, numeric={"x": "m", "y": "s"})
        with self.assertRaises(DomainError):
            ms.action(ms.basis(geo2, degree=1), ms.rotate("x", "y", cos=0, sin=1), geo2)
        with self.assertRaises(DomainError):
            ms.action(ms.basis(geo2, degree=1), ms.permute({"x": "y", "y": "x"}), geo2)
        # dictionary codes (c: VARCHAR, alphabetical codes blue=0 < green=1 < red=2): adjacency is no metric
        from torbles import math_certificates as mc
        cscope = mc.code_scope(t, ["c", "y"])
        with self.assertRaises(DomainError):                  # codes cannot be declared numeric
            ms.geometry(cscope, numeric={"c": "m", "y": "m"})
        cgeo = ms.geometry(cscope, numeric={"y": "m"}, categorical={"c": "colour"})
        for tr in (ms.translate(c=1), ms.scale(c=2), ms.reflect(c=1), ms.rotate("c", "y", cos=0, sin=1),
                   ms.affine(["c", "y"], [[1, 1], [0, 1]], [0, 0]), ms.permute({"c": "y", "y": "c"})):
            with self.assertRaises(DomainError):
                ms.action(ms.basis(cgeo, degree=1), tr, cgeo)
        with self.assertRaises(DomainError):                  # no polynomial basis over a code
            ms.basis(cgeo, terms=["1", "c", "c*y"])
        with self.assertRaises(DomainError):
            ms.state(mbk.build(cscope, ["1", "c", "y"], "joint"), cgeo)
        with self.assertRaises(DomainError):                  # code permutations need the geometry's own kind
            ms.group(cgeo, [ms.translate(c=1)])
        # a declared value permutation of the codes is a SYM-002 generator (no metric involved); its verdict is the
        # multiset's: red->blue->green->red on (c) alone does not preserve these rows, the identity of codes does
        codes = {lab: i for i, lab in enumerate(sorted(set(frame["c"])))}
        cyc = ms.values("c", {codes["red"]: codes["blue"], codes["blue"]: codes["green"], codes["green"]: codes["red"]})
        with self.assertRaises(ms.SymmetryViolated):
            ms.certify(cscope, cgeo, [cyc])
        # the same data with its labels renamed (another dictionary order) gives the same verdicts and witnesses
        ren = {"red": "a", "green": "z", "blue": "m"}
        db2 = torbles.from_ddl("CREATE TABLE t (y BIGINT, c VARCHAR)",
                               {"t": pd.DataFrame({"y": frame["y"], "c": frame["c"].map(ren)})})
        cs2 = mc.code_scope(db2["t"], ["c", "y"])
        cg2 = ms.geometry(cs2, numeric={"y": "m"}, categorical={"c": "colour"})
        c2 = {lab: i for i, lab in enumerate(sorted(set(frame["c"].map(ren))))}
        cyc2 = ms.values("c", {c2[ren[a]]: c2[ren[b]] for a, b in (("red", "blue"), ("blue", "green"), ("green", "red"))})
        with self.assertRaises(ms.SymmetryViolated) as e1:
            ms.certify(cscope, cgeo, [cyc])
        with self.assertRaises(ms.SymmetryViolated) as e2:
            ms.certify(cs2, cg2, [cyc2])
        self.assertEqual(e1.exception.witness, e2.exception.witness)
        # an action of one geometry is refused on a state of another
        other = ms.geometry(scope, numeric={"y": "m", "x": "m"})
        with self.assertRaises(DomainError):
            st.query([dict(action=0, group=0, term="x")], actions=[ms.action(ms.basis(other, degree=2),
                                                                             ms.translate(x=1), other)])

    def test_sym001_closure_is_decided_and_reported(self):
        rng = np.random.default_rng(5103)
        scope, pts, _ = make_xyz(rng, n=30)
        geo = ms.geometry(scope, numeric={"x": "m", "y": "m", "z": "m"})
        multi = ms.basis(geo, terms=["1", "x", "y", "x*y"])          # closed under translations and scalings
        ms.action(multi, ms.translate(x=3, y=-1), geo)
        ms.action(multi, ms.scale(x=2), geo)
        ms.action(multi, ms.permute({"x": "y", "y": "x"}), geo)
        with self.assertRaises(ms.NotClosedError) as cm:             # a rotation needs x^2, y^2
            ms.action(multi, ms.rotate("x", "y", cos=F(3, 5), sin=F(4, 5)), geo)
        self.assertEqual(cm.exception.term, (1, 1, 0))
        self.assertIn(cm.exception.missing, ({"x": 2}, {"y": 2}))
        hom = ms.basis(geo, degree=2, homogeneous=True)              # closed under linear maps only
        ms.action(hom, ms.rotate("x", "z", cos=F(-7, 25), sin=F(24, 25)), geo)
        with self.assertRaises(ms.NotClosedError):
            ms.action(hom, ms.translate(y=1), geo)
        with self.assertRaises(ms.NotClosedError):
            ms.action(ms.basis(geo, degree=2, columns=["x", "y"]), ms.rotate("y", "z", degrees=90), geo)
        with self.assertRaises(ms.NotClosedError):                   # structurally, also for an interval rotation
            ms.action(multi, ms.rotate("x", "y", degrees=10), geo)
        # the homogeneous TENSOR-001 tensor is such a basis: rotations act, translations are refused
        ht = ms.state(mt.moments(scope, ["x", "y", "z"], order=2), geo)
        self.assertTrue(ht.basis.equal(hom))
        A = ms.action(ht.basis, ms.rotate("x", "z", cos=F(-7, 25), sin=F(24, 25)), geo)
        self.check_state(ht, A, o_rotate(0, 2, F(-7, 25), F(24, 25)), pts)

    def test_sym001_irrational_rotations_are_certified_interval_enclosures(self):
        getcontext().prec = 60
        pi = Decimal("3.14159265358979323846264338327950288419716939937510582097494459")

        def cos_sin(deg):                                            # 60-digit Taylor series (the oracle)
            x = Decimal(deg.numerator) / Decimal(deg.denominator) * pi / 180
            s = c = Decimal(0)
            term = Decimal(1)
            for k in range(80):
                if k % 2 == 0:
                    c += term * (1 if k % 4 == 0 else -1)
                else:
                    s += term * (1 if k % 4 == 1 else -1)
                term = term * x / (k + 1)
            return c, s
        rng = np.random.default_rng(5104)
        scope, pts, _ = make_xyz(rng, n=40, G=2)
        geo = ms.geometry(scope, numeric={"x": "m", "y": "m", "z": "m"})
        B = ms.basis(geo, degree=3)
        st = ms.state(mbk.build(scope, B.terms_for(scope), "joint"), geo)
        for deg in (F(30), F(75, 2), F(200), F(-45), F(1, 3), F(1000)):
            A = ms.action(B, ms.rotate("x", "y", degrees=deg, center=(1, 0)), geo)
            info = A.info
            self.assertEqual((info["carrier"], info["approximate"], info["precision"]), ("interval", True, 53))
            c, s = cos_sin(deg)
            for g in range(2):
                got = st.transformed(A, g)
                self.assertEqual(got, st.transformed(A, g, backend="cuda"))
                for t, (lo, hi) in zip(B.terms, got):
                    true = sum((dpow(c * (Decimal(p[0].numerator) - 1) - s * Decimal(p[1].numerator) + 1, t[0]) *
                                dpow(s * (Decimal(p[0].numerator) - 1) + c * Decimal(p[1].numerator), t[1]) *
                                dpow(Decimal(p[2].numerator), t[2]) for p in pts[g]), Decimal(0))
                    self.assertTrue(Decimal(lo) <= true <= Decimal(hi), (deg, t, lo, true, hi))
                    self.assertLessEqual(hi - lo, 1e-9 * max(1.0, abs(lo)), (deg, t))   # precision-declared
        # multiples of 90 degrees are exact; composing intervals stays an enclosure (carried through)
        self.assertEqual(ms.action(B, ms.rotate("x", "y", degrees=450), geo).info["carrier"], "exact")
        A = ms.action(B, ms.rotate("x", "y", degrees=20), geo).compose(ms.action(B, ms.rotate("x", "y", degrees=25), geo))
        self.assertEqual(A.info["carrier"], "interval")
        c, s = cos_sin(F(45))
        got = st.transformed(A, 0)
        for t, (lo, hi) in zip(B.terms, got):
            true = sum((dpow(c * Decimal(p[0].numerator) - s * Decimal(p[1].numerator), t[0]) *
                        dpow(s * Decimal(p[0].numerator) + c * Decimal(p[1].numerator), t[1]) *
                        dpow(Decimal(p[2].numerator), t[2]) for p in pts[0]), Decimal(0))
            self.assertTrue(Decimal(lo) <= true <= Decimal(hi))
        exact = ms.action(B, ms.translate(x=1), geo).compose(ms.action(B, ms.rotate("x", "y", degrees=10), geo))
        self.assertEqual(exact.info["carrier"], "interval")
        # not a point of the unit circle: refused
        with self.assertRaises(DomainError):
            ms.action(B, ms.rotate("x", "y", cos=F(1, 2), sin=F(1, 2)), geo)
        with self.assertRaises(TypeError):                           # floats are never accepted
            ms.rotate("x", "y", degrees=30.0).map(geo)

    def test_sym001_bases_of_3_6_15_32_states_and_retained_states(self):
        rng = np.random.default_rng(5105)
        n = 50
        cols = {f"x{i}": rng.integers(-4, 5, n) for i in range(31)}
        gid = [int(g) for g in rng.integers(0, 2, n)]
        scope = mbk.Scope.from_arrays(cols, groups=gid, ngroups=2)
        geo = ms.geometry(scope, numeric={k: "m" for k in cols})
        names = list(cols)
        pts = [[tuple(F(int(cols[k][r])) for k in names) for r in range(n) if gid[r] == g] for g in range(2)]
        for size, (cs, d) in {3: (names[:2], 1), 6: (names[:2], 2), 15: (names[:4], 2), 32: (names, 1)}.items():
            B = ms.basis(geo, degree=d, columns=cs)
            self.assertEqual(len(B), size)
            st = ms.state(mbk.build(scope, B.terms_for(scope), "joint"), geo)
            i0, i1 = names.index(cs[0]), names.index(cs[1])
            tests = [(ms.translate({cs[0]: 3, cs[-1]: F(-1, 2)}), o_translate({i0: 3, names.index(cs[-1]): F(-1, 2)})),
                     (ms.rotate(cs[0], cs[1], cos=F(20, 29), sin=F(21, 29)), o_rotate(i0, i1, F(20, 29), F(21, 29))),
                     (ms.permute({cs[0]: cs[-1], cs[-1]: cs[0]}), o_permute({i0: names.index(cs[-1]), names.index(cs[-1]): i0}))]
            for tr, f in tests:
                A = ms.action(B, tr, geo)
                self.check_state(st, A, f, pts)

    def test_sym001_retained_state_sources_and_their_refusals(self):
        rng = np.random.default_rng(5106)
        scope, pts, _ = make_xyz(rng, n=40, nulls=0.0)
        geo = ms.geometry(scope, numeric={"x": "m", "y": "m", "z": "m"})
        B = ms.basis(geo, degree=2)
        # SQL banks: transformable only where every term is valid on every row (no NULL here)
        st = ms.state(mbk.build(scope, B.terms_for(scope), "sql"), geo)
        self.check_state(st, ms.action(B, ms.translate(x=2), geo), o_translate({0: 2}), pts)
        nscope, _, _ = make_xyz(np.random.default_rng(5107), n=40, nulls=0.3)
        ngeo = ms.geometry(nscope, numeric={"x": "m", "y": "m", "z": "m"})
        with self.assertRaises(DomainError):
            ms.state(mbk.build(nscope, ms.basis(ngeo, degree=2).terms_for(nscope), "sql"), ngeo)
        ms.state(mbk.build(nscope, ms.basis(ngeo, degree=2).terms_for(nscope), "joint"), ngeo)
        with self.assertRaises(DomainError):                  # SQL tensors carry per-entry counts
            ms.state(mt.moments(scope, ["1", "x", "y"], order=2, policy="sql"), geo)
        with self.assertRaises(DomainError):                  # axes listing different columns
            ms.state(mt.moments(scope, [["1", "x"], ["1", "y"]]), geo)
        with self.assertRaises(DomainError):                  # a state of another scope
            ms.state(mbk.build(nscope, ["1", "x"], "joint"), geo)
        with self.assertRaises(DomainError):                  # a count tensor
            ms.state(mt.counts(scope, ["u"]), geo)
        with self.assertRaises(TypeError):
            ms.state(object(), geo)
        st2 = ms.state(mt.moments(scope, ["1", "y", "x"], order=2), geo)       # any axis order of the columns
        Bxy = ms.basis(geo, degree=2, columns=["x", "y"])
        self.assertTrue(st2.basis.equal(Bxy))
        st3 = ms.state(mbk.build(scope, Bxy.terms_for(scope), "joint"), geo)
        self.assertEqual([st2.moments(g) for g in range(3)], [st3.moments(g) for g in range(3)])
        # groups: rows and contributing rows kept apart from the moments
        self.assertEqual(st.group(0)["rows"], len(pts[0]))

    def test_sym001_exactness_overflow_is_refused_never_wrapped(self):
        rng = np.random.default_rng(5108)
        scope, pts, _ = make_xyz(rng, n=20)
        geo = ms.geometry(scope, numeric={"x": "m", "y": "m", "z": "m"})
        B = ms.basis(geo, degree=3)
        st = ms.state(mbk.build(scope, B.terms_for(scope), "joint"), geo)
        big = ms.translate(x=2 ** 50)                                 # (x + 2^50)^3 has 2^150 entries
        with self.assertRaises(OverflowError):
            ms.action(B, big, geo)
        A = ms.action(B, ms.translate(x=2 ** 40), geo)                # fits int128; its products need 512 bits
        self.check_state(st, A, o_translate({0: 2 ** 40}), pts)
        A4 = A.compose(A).compose(A).compose(A)                       # (x + 2^42)^3: 2^126 still fits
        self.check_state(st, A4, o_translate({0: 2 ** 42}), pts)
        with self.assertRaises(OverflowError):                        # (x + 2^43)^3: 2^129 is refused
            A4.compose(A4)
        R = ms.action(B, ms.rotate("x", "y", cos=F(119, 169), sin=F(120, 169)), geo)
        R2 = R
        for _ in range(3):
            R2 = R2.compose(R)                                        # denominators 169^12: still int128
        self.check_state(st, R2, o_compose(*[o_rotate(0, 1, F(119, 169), F(120, 169))] * 4), pts)
        with self.assertRaises(OverflowError):
            for _ in range(8):
                R2 = R2.compose(R)
        # degree 8: D = 481^8; R R^-1 sums products past int128 and reduces exactly (FND-009 big integers) to I,
        # while R R needs the denominator 481^16 and is refused
        B8 = ms.basis(geo, degree=8, columns=["x", "y"])
        R8 = ms.action(B8, ms.rotate("x", "y", cos=F(319, 481), sin=F(360, 481)), geo)
        R8i = ms.action(B8, ms.rotate("x", "y", cos=F(319, 481), sin=F(-360, 481)), geo)
        self.assertEqual(R8.info["den"], 481 ** 8)
        self.assertTrue(R8.compose(R8i).equal(ms.identity_action(B8)))
        with self.assertRaises(OverflowError):
            R8.compose(R8)
        s8, p8, _ = make_xyz(np.random.default_rng(5110), n=20, nulls=0.0)
        g8 = ms.geometry(s8, numeric={"x": "m", "y": "m", "z": "m"})
        t8 = ms.state(mt.moments(s8, ["1", "x", "y"], order=8), g8)
        R8g = ms.action(t8.basis, ms.rotate("x", "y", cos=F(319, 481), sin=F(360, 481)), g8)
        self.check_state(t8, R8g, o_rotate(0, 1, F(319, 481), F(360, 481)), p8)

    def test_sym001_budgets_at_the_limit_and_one_past(self):
        rng = np.random.default_rng(5109)
        scope, pts, _ = make_xyz(rng, n=30)
        geo = ms.geometry(scope, numeric={"x": "m", "y": "m", "z": "m"})
        B = ms.basis(geo, degree=2)
        st = ms.state(mbk.build(scope, B.terms_for(scope), "joint"), geo)
        A = ms.action(B, ms.rotate("x", "y", cos=F(3, 5), sin=F(4, 5)), geo)
        reqs = [dict(action=0, group=g, term="x^2") for g in range(3)] * 3

        def limit_of(fn, resource):
            try:
                fn(Budget(**{resource: 0}))
            except AlgebraBudgetError as e:
                self.assertEqual(e.resource, resource)
                return e.need
            self.fail(f"no {resource} need")
        for resource, fn in (
                ("tensor", lambda b: ms.action(B, ms.translate(x=1), geo, budget=b)),
                ("bytes", lambda b: ms.action(B, ms.translate(x=1), geo, budget=b)),
                ("support", lambda b: ms.basis(geo, degree=3, budget=b)),
                ("iterations", lambda b: A.compose(A, budget=b)),
                ("tensor", lambda b: A.compose(A, budget=b)),
                ("states", lambda b: ms.state(mbk.build(scope, B.terms_for(scope), "joint"), geo, budget=b)),
                ("output", lambda b: st.query(reqs, actions=[A], budget=b)),
                ("output", lambda b: st.query(reqs, actions=[A], budget=b, backend="cuda")),
                ("bytes", lambda b: st.query(reqs, actions=[A], budget=b, backend="cuda"))):
            need = limit_of(fn, resource)
            fn(Budget(**{resource: need}))
            with self.assertRaises(AlgebraBudgetError) as cm:
                fn(Budget(**{resource: need - 1}))
            self.assertEqual((cm.exception.resource, cm.exception.need, cm.exception.limit), (resource, need, need - 1))
        self.assertEqual(ms.need("action", B, ms.translate(x=1).map(geo))["tensor"], 100)
        self.assertEqual(limit_of(lambda b: A.compose(A, budget=b), "iterations"), 1000)
        self.assertEqual(limit_of(lambda b: st.query(reqs, actions=[A], budget=b), "output"), 9)
        self.assertEqual(ms.need("query", st, 9, [A], [], backend="cuda")["bytes"],
                         limit_of(lambda b: st.query(reqs, actions=[A], budget=b, backend="cuda"), "bytes"))
        with self.assertRaises(UnsupportedError):
            ms.basis(geo, degree=9)
        with self.assertRaises(UnsupportedError):
            ms.basis(geo, terms=["x^9"])

    def test_sym001_opt_in_import(self):
        out = subprocess.run([sys.executable, "-c", "import sys, torbles; print('torbles.math_sym' in sys.modules)"],
                             capture_output=True, text=True)
        self.assertEqual(out.stdout.strip(), "False")


# ====================================================================== SYM-002
def symmetric_rows(rng, reps, images, mult=(1, 3)):
    """full orbits: every image of every representative with one multiplicity per orbit (shuffled)"""
    rows = []
    for p in reps:
        m = int(rng.integers(mult[0], mult[1] + 1))
        for q in sorted(set(images(p))):
            rows += [q] * m
    rng.shuffle(rows)
    return rows


def d4(p):                                                            # the dihedral group of the square on (x, y)
    x, y = p[0], p[1]
    rest = tuple(p[2:])
    return [(u, v) + rest for u, v in ((x, y), (-y, x), (-x, -y), (y, -x), (-x, y), (y, x), (x, -y), (-y, -x))]


def gens_d4(geo):
    return [ms.rotate("x", "y", degrees=90), ms.reflect(x=0)]


def o_gens_d4():
    return [lambda p: (-p[1], p[0]) + tuple(p[2:]), lambda p: (-p[0],) + tuple(p[1:])]


def counter_evidence(rows, sel, gens):
    """per generator: violating rows (Counter of the selected rows' tuples) and the smallest violating row"""
    cnt = Counter(tuple(rows[r]) for r in range(len(rows)) if sel[r])
    out = []
    for g in gens:
        bad = [r for r in range(len(rows)) if sel[r] and cnt[tuple(g(tuple(rows[r])))] != cnt[tuple(rows[r])]]
        out.append((len(bad), min(bad) if bad else None))
    return out


def bfs_orbits(tuples, gens):
    """orbits by breadth-first search over the generators (the oracle): {frozenset of tuples}"""
    left, out = set(tuples), []
    while left:
        start = min(left)
        seen, todo = {start}, [start]
        while todo:
            p = todo.pop()
            for g in gens:
                q = tuple(g(p))
                if q not in seen:
                    seen.add(q)
                    todo.append(q)
        out.append((start, frozenset(seen)))
        left -= seen
    return out


def decode(table, col):
    out = np.empty(max(table.n, 1), dtype=np.uint32)
    algq.lib.at_decode_column(table.at, table.idx(col), out.ctypes.data)
    return out[:table.n].astype(np.int64)


def flip_column(table, col, row):
    """flips one stored bit so that exactly column `col` changes at `row`"""
    before = {c: decode(table, c) for c in table.cols}
    for p in range(table.nplanes()):
        table.flip(p, row)
        after = {c: decode(table, c) for c in table.cols}
        if [c for c in table.cols if not np.array_equal(before[c], after[c])] == [col]:
            return
        table.flip(p, row)
    raise AssertionError(f"no single plane changes only {col} at row {row}")


def array_scope(rows, names, sel=None, valid=None):
    X = np.array(rows, dtype=np.int64).reshape(len(rows), len(names))
    return mbk.Scope.from_arrays({n: X[:, i] for i, n in enumerate(names)}, rows=sel, valid=valid)


class Symmetries(unittest.TestCase):
    # ------------------------------------------------------------------ acceptance 1: invalidation with a witness
    def test_sym002_broken_multiplicity_or_missing_orbit_members_invalidate_with_a_witness(self):
        rng = np.random.default_rng(5201)
        reps = [(int(a), int(b), int(c)) for a, b, c in zip(rng.integers(-5, 6, 9), rng.integers(-5, 6, 9),
                                                            rng.integers(0, 3, 9))]
        sw = {0: 1, 1: 0}
        images = lambda p: [q[:2] + (c,) for q in d4(p) for c in ([sw[p[2]], p[2]] if p[2] in sw else [p[2]])]
        rows = symmetric_rows(rng, reps, images)
        names = ["x", "y", "c"]
        scope = array_scope(rows, names)
        geo = ms.geometry(scope, numeric={"x": "m", "y": "m"}, categorical={"c": "colour"})
        decl = gens_d4(geo) + [ms.values("c", sw)]
        og = o_gens_d4() + [lambda p: p[:2] + (sw.get(p[2], p[2]),)]
        cert = ms.certify(scope, geo, decl)
        dev = ms.certify(scope, geo, decl, backend="cuda")
        self.assertEqual(cert.evidence, dev.evidence)
        self.assertEqual(dev.info["verified_by"], "cuda library")
        self.assertEqual(cert.info["kind"], "snapshot")
        self.assertTrue(all(e["holds"] and e["rows"] == len(rows) for e in cert.evidence))
        self.assertTrue(cert.fresh(scope).fresh)
        sel_all = [True] * len(rows)
        self.assertEqual(counter_evidence(rows, sel_all, og), [(0, None)] * 3)
        cnt = Counter(rows)
        # 1. a missing orbit member: every row of one tuple deselected; 2. a broken multiplicity: one row of a tuple
        #    with multiplicity >= 2 deselected; 3. a duplicated row (an extra row); 4. a row moved off its orbit
        single = next(r for r, p in enumerate(rows) if cnt[p] >= 2)
        cases = {"missing": [rows[r] != rows[0] for r in range(len(rows))],
                 "broken": [r != single for r in range(len(rows))]}
        for label, sel in cases.items():
            s2 = array_scope(rows, names, sel=np.array(sel))
            g2 = ms.geometry(s2, numeric={"x": "m", "y": "m"}, categorical={"c": "colour"})
            oracle = counter_evidence(rows, sel, og)
            for backend in ("cpu", "cuda"):
                with self.assertRaises(ms.SymmetryViolated) as cm:
                    ms.certify(s2, g2, gens_d4(g2) + [ms.values("c", sw)], backend=backend)
                e = cm.exception
                self.assertEqual([(x["violations"], x["witness"]) for x in e.evidence], oracle, (label, backend))
                self.assertEqual(e.witness, min(w for _, w in oracle if w is not None))
                kinds = {x["kind"] for x in e.evidence if not x["holds"]}
                self.assertIn("broken" if label == "broken" else "missing", kinds)
        extra = rows + [rows[3]]
        for moved in (extra, rows[:5] + [(rows[5][0] + 1,) + rows[5][1:]] + rows[6:]):
            s3 = array_scope(moved, names)
            g3 = ms.geometry(s3, numeric={"x": "m", "y": "m"}, categorical={"c": "colour"})
            oracle = counter_evidence(moved, [True] * len(moved), og)
            with self.assertRaises(ms.SymmetryViolated) as cm:
                ms.certify(s3, g3, gens_d4(g3) + [ms.values("c", sw)])
            self.assertEqual([(x["violations"], x["witness"]) for x in cm.exception.evidence], oracle)
            with self.assertRaises(ms.SymmetryViolated) as cd:
                ms.certify(s3, g3, gens_d4(g3) + [ms.values("c", sw)], backend="cuda")
            self.assertEqual(cd.exception.evidence, cm.exception.evidence)
        # a certified engine table that is then mutated: the certificate is stale and authorizes no orbit reuse;
        # re-verifying the current rows finds the witness
        frame = pd.DataFrame({"x": pd.Series([p[0] for p in rows], dtype=object),
                              "y": pd.Series([p[1] for p in rows], dtype=object),
                              "c": pd.Series([p[2] for p in rows], dtype=object)})
        db = torbles.from_ddl("CREATE TABLE p (x BIGINT, y BIGINT, c BIGINT)", {"p": frame})
        t = db["p"]
        ts = mbk.Scope.from_table(t, ["x", "y", "c"])
        tg = ms.geometry(ts, numeric={"x": "m", "y": "m"}, categorical={"c": "colour"})
        tc = ms.certify(ts, tg, gens_d4(tg) + [ms.values("c", sw)])
        orb = tc.orbits(ts)
        self.assertEqual(orb.query([ms.total({"x^2": 1, "y^2": 1})])[0]["value"],
                         sum(p[0] ** 2 + p[1] ** 2 for p in rows))
        flip_column(t, "x", 0)                                        # one stored bit: only x changes, in row 0
        now = mbk.Scope.from_table(t, ["x", "y", "c"])
        f = tc.fresh(now)
        self.assertFalse(f.fresh)
        self.assertEqual(f.first, "revision")
        with self.assertRaises(StaleCertificateError):
            tc.orbits(now)
        with self.assertRaises(StaleCertificateError):
            orb.query([ms.total({"x^2": 1})], scope=now)
        cur = list(zip(*[decode(t, c) + t.codecs[c].affine[0] for c in ("x", "y", "c")]))
        cur = [tuple(int(v) for v in r) for r in cur]
        self.assertNotEqual(cur, [tuple(p) for p in rows])
        self.assertEqual(cur[1:], [tuple(p) for p in rows[1:]])
        oracle = counter_evidence(cur, [True] * len(cur), og)
        ng = ms.geometry(now, numeric={"x": "m", "y": "m"}, categorical={"c": "colour"})
        with self.assertRaises(ms.SymmetryViolated) as cm:
            ms.certify(now, ng, gens_d4(ng) + [ms.values("c", sw)])
        self.assertEqual([(x["violations"], x["witness"]) for x in cm.exception.evidence], oracle)
        self.assertEqual(cm.exception.witness, 0)

    # ------------------------------------------------------------------ acceptance 2: orbit-reduced answers
    def test_sym002_orbit_reduced_answers_match_exhaustive_evaluation(self):
        rng = np.random.default_rng(5202)
        for label, n_extra in (("d4", 0), ("d4 x z", 1)):
            reps = [tuple(int(v) for v in rng.integers(-6, 7, 2 + n_extra)) for _ in range(14)]
            if n_extra:
                images = lambda p: [q[:2] + (z,) for q in d4(p) for z in {p[2], 3 - p[2]}]
                names = ["x", "y", "z"]
            else:
                images, names = d4, ["x", "y"]
            rows = symmetric_rows(rng, reps, images, mult=(1, 4))
            scope = array_scope(rows, names)
            geo = ms.geometry(scope, numeric={k: "m" for k in names})
            decl = gens_d4(geo) + ([ms.reflect(z=F(3, 2))] if n_extra else [])
            og = o_gens_d4() + ([lambda p: p[:2] + (3 - p[2],)] if n_extra else [])
            cert = ms.certify(scope, geo, decl)
            orb = cert.orbits(scope)
            info = orb.info
            # the summary against breadth-first orbits of the support (Counter of the rows)
            cnt = Counter(tuple(p) for p in rows)
            oracle = bfs_orbits(cnt.keys(), og)
            summary = orb.summary()
            self.assertEqual([(rep, len(members), cnt[rep]) for rep, members in sorted(oracle)], summary)
            self.assertTrue(all(len({cnt[q] for q in members}) == 1 for _, members in oracle))
            self.assertEqual((info["orbits"], info["distinct"], info["rows"]), (len(oracle), len(cnt), len(rows)))
            self.assertLessEqual(info["largest"], 16 if n_extra else 8)
            r2 = {"x^2": 1, "y^2": 1}
            invariant = [ms.total(r2), ms.count(r2, "<=", 25), ms.count(r2, ">", 9), ms.total({"x^2*y^2": 3, "1": -1}),
                         ms.total({"x^4": 1, "y^4": 1, "x^2*y^2": -6}), ms.count({"x^2*y^2": 1}, "==", 0)]
            if n_extra:
                invariant += [ms.total({"z^2": 1, "z": -3}), ms.count({"z^2": 1, "z": -3, "x^2": 1, "y^2": 1}, "<", 4)]
            got = orb.query(invariant)
            full = ms.evaluate(scope, geo, invariant)
            for q, a, b in zip(invariant, got, full):
                self.assertTrue(a["invariant"])
                self.assertEqual(a["value"], b["value"], (label, q))
                self.assertEqual((a["rows"], b["rows"]), (len(rows), len(rows)))
                self.assertEqual(a["evaluations"], info["orbits"])           # one evaluation per orbit
                self.assertEqual(b["evaluations"], len(rows))                # versus every row
                # the exhaustive Python evaluation over every row
                ev = [sum(c * np.prod([p[i] ** e for i, e in enumerate(ms._exponents(names, t))])
                          for t, c in q.terms.items()) for p in rows]
                ops = {"<": lambda v: v < q.constant, "<=": lambda v: v <= q.constant, ">": lambda v: v > q.constant,
                       "==": lambda v: v == q.constant}
                want = sum(int(v) for v in ev) if q.kind == "sum" else sum(1 for v in ev if ops[q.op](v))
                self.assertEqual(a["value"], want, (label, q))
            # non-invariant queries are refused under the reduction (the exhaustive evaluation answers them)
            # (x^2 - y^2 == 0 is an invariant predicate of a non-invariant polynomial: refused conservatively)
            for q, gen in ((ms.total({"x": 1}), 0), (ms.count({"x": 1, "y": 2}, ">", 0), 0), (ms.total({"x^2": 1}), 0),
                           (ms.total({"x*y": 1}), 0), (ms.count({"y": 1}, ">=", 0), 0),
                           (ms.count({"x^2": 1, "y^2": -1}, "==", 0), 0)):
                with self.assertRaises(ms.NotInvariantError) as cm:
                    orb.query([q])
                self.assertEqual(cm.exception.generator, gen)
                self.assertEqual(orb.query([q], raw=True)[0]["status"], -6)
                ms.evaluate(scope, geo, [q])
            if n_extra:
                with self.assertRaises(ms.NotInvariantError) as cm:
                    orb.query([ms.total({"z": 1})])
                self.assertEqual(cm.exception.generator, 2)
            self.assertEqual(cert.group.invariant(ms.total(r2)), (True, None))
            self.assertEqual(cert.group.invariant(ms.total({"x^3": 1}))[0], False)

    def test_sym002_value_permutations_and_coordinate_permutations(self):
        rng = np.random.default_rng(5203)
        # categories 0..3 permuted by the cycle (0 1 2) with 3 fixed, and the swap of two numeric columns
        cyc = {0: 1, 1: 2, 2: 0}
        reps = [(int(a), int(b), int(c)) for a, b, c in zip(rng.integers(0, 9, 10), rng.integers(0, 9, 10),
                                                            rng.integers(0, 4, 10))]
        images = lambda p: [(u, v, c) for (u, v) in {(p[0], p[1]), (p[1], p[0])}
                            for c in ([0, 1, 2] if p[2] in cyc else [p[2]])]
        rows = symmetric_rows(rng, reps, images)
        scope = array_scope(rows, ["a", "b", "c"])
        geo = ms.geometry(scope, numeric={"a": "kg", "b": "kg"}, categorical={"c": "kind"})
        decl = [ms.values("c", cyc), ms.permute({"a": "b", "b": "a"})]
        cert = ms.certify(scope, geo, decl)
        self.assertEqual(cert.evidence, ms.certify(scope, geo, decl, backend="cuda").evidence)
        orb = cert.orbits()
        self.assertLessEqual(orb.info["largest"], 6)
        moved = {"c^3": 1, "c^2": -3, "c": 2}                         # c (c - 1) (c - 2): 0 on every moved value
        qs = [ms.total({"a": 1, "b": 1}), ms.count({"a*b": 1}, ">", 20), ms.total({"a^2*b": 1, "a*b^2": 1}),
              ms.count(moved, "==", 0), ms.total(dict(moved, a=5, b=5)), ms.count({"a": 1, "b": 1}, ">", 9)]
        with self.assertRaises(ms.NotInvariantError):                 # c == 3: an invariant predicate, refused
            orb.query([ms.count({"c": 1}, "==", 3)])
        got, full = orb.query(qs), ms.evaluate(scope, geo, qs)
        self.assertEqual([a["value"] for a in got], [b["value"] for b in full])
        for bad in (ms.count({"c": 1}, "==", 0), ms.total({"a": 1}), ms.total({"c": 1}),
                    ms.count({"c": 1, "a": 1, "b": 1}, ">", 9)):
            with self.assertRaises(ms.NotInvariantError):
                orb.query([bad])
        # a value permutation must be a permutation of its values; a broken cycle is refused at the map
        with self.assertRaises(DomainError):
            ms.group(geo, [ms.values("c", {0: 1, 1: 1})])
        with self.assertRaises(DomainError):
            ms.group(geo, [ms.values("c", {0: 1, 1: 2})])
        with self.assertRaises(DomainError):                          # not a value at the column's scale
            ms.group(geo, [ms.values("a", {F(1, 2): 1, 1: F(1, 2)})])

    def test_sym002_generators_must_map_the_integer_domain_to_itself(self):
        rows = [(1, 2), (-2, 1), (-1, -2), (2, -1)]
        scope = array_scope(rows, ["x", "y"])
        geo = ms.geometry(scope, numeric={"x": "m", "y": "m"})
        ms.group(geo, [ms.rotate("x", "y", degrees=90)])
        for bad in (ms.rotate("x", "y", cos=F(3, 5), sin=F(4, 5)),   # not integral
                    ms.rotate("x", "y", degrees=45),                  # an interval map
                    ms.scale(x=0),                                    # not invertible
                    ms.affine(["x", "y"], [[1, 2], [2, 4]], [0, 0]),  # singular
                    ms.translate(x=F(1, 2)),                          # leaves the integers
                    ms.rotate("x", "y", degrees=90, center=(F(1, 2), 0))):
            with self.assertRaises(DomainError):
                ms.group(geo, [bad])
        ms.certify(scope, geo, [ms.rotate("x", "y", degrees=90)])
        # an injective non-surjective integral map is decided exactly (here it breaks the multiset)
        with self.assertRaises(ms.SymmetryViolated):
            ms.certify(scope, geo, [ms.scale(x=2)])
        # decimal coordinates: logical integers at the scale; a rotation by 90 needs equal scales to be integral
        frame = pd.DataFrame({"x": pd.Series(["1.00", "-2.00", "-1.00", "2.00"], dtype=object),
                              "y": pd.Series([2, 1, -2, -1], dtype=object),
                              "v": pd.Series(["2.00", "1.00", "-2.00", "-1.00"], dtype=object)})
        db = torbles.from_ddl("CREATE TABLE q (x DECIMAL(6,2), y BIGINT, v DECIMAL(6,2))", {"q": frame})
        qs = mbk.Scope.from_table(db["q"], ["x", "y", "v"])
        mixed = ms.geometry(qs, numeric={"x": "m", "y": "m"})
        with self.assertRaises(DomainError):
            ms.group(mixed, [ms.rotate("x", "y", degrees=90)])
        same = ms.geometry(qs, numeric={"x": "m", "v": "m"})
        ms.certify(qs, same, [ms.rotate("x", "v", degrees=90)])

    def test_sym002_null_policies_and_empty_supports(self):
        rows = [(1, 2), (-2, 1), (-1, -2), (2, -1), (5, 5)]
        valid = {"y": np.array([True, True, True, True, False])}
        scope = array_scope(rows, ["x", "y"], valid=valid)
        geo = ms.geometry(scope, numeric={"x": "m", "y": "m"})
        cert = ms.certify(scope, geo, [ms.rotate("x", "y", degrees=90)], nulls="skip")
        self.assertEqual((cert.info["checked"], cert.info["excluded"]), (4, 1))
        with self.assertRaises(DomainError) as cm:
            ms.certify(scope, geo, [ms.rotate("x", "y", degrees=90)], nulls="forbid")
        self.assertEqual(cm.exception.row, 4)
        q = [ms.total({"x^2": 1, "y^2": 1})]
        self.assertEqual(ms.evaluate(scope, geo, q)[0]["value"], 20)
        self.assertEqual(cert.orbits().query(q)[0]["value"], 20)
        with self.assertRaises(DomainError):
            ms.evaluate(scope, geo, q, nulls="forbid")
        empty = array_scope(rows, ["x", "y"], sel=np.zeros(5, bool))
        eg = ms.geometry(empty, numeric={"x": "m", "y": "m"})
        ec = ms.certify(empty, eg, [ms.rotate("x", "y", degrees=90)], backend="cuda")
        self.assertEqual(ec.orbits().info["orbits"], 0)
        self.assertEqual(ec.orbits().query(q)[0]["value"], 0)

    def test_sym002_budgets_at_the_limit_and_one_past(self):
        rng = np.random.default_rng(5205)
        rows = symmetric_rows(rng, [tuple(int(v) for v in rng.integers(-4, 5, 2)) for _ in range(6)], d4)
        scope = array_scope(rows, ["x", "y"])
        geo = ms.geometry(scope, numeric={"x": "m", "y": "m"})
        G = ms.group(geo, gens_d4(geo))
        cert = ms.certify(scope, geo, G)
        orb = cert.orbits()
        qs = [ms.total({"x^2": 1, "y^2": 1}), ms.count({"x^2": 1, "y^2": 1}, "<", 10)] * 2

        def limit_of(fn, resource):
            try:
                fn(Budget(**{resource: 0}))
            except AlgebraBudgetError as e:
                self.assertEqual(e.resource, resource)
                return e.need
            self.fail(f"no {resource} need")
        for resource, fn in (
                ("support", lambda b: ms.support(scope, geo, budget=b)),
                ("bytes", lambda b: ms.support(scope, geo, budget=b)),
                ("output", lambda b: ms.certify(scope, geo, G, budget=b)),
                ("output", lambda b: ms.certify(scope, geo, G, budget=b, backend="cuda")),
                ("bytes", lambda b: ms.certify(scope, geo, G, budget=b, backend="cuda")),
                ("support", lambda b: cert.orbits(budget=b)),
                ("output", lambda b: orb.query(qs, budget=b)),
                ("output", lambda b: ms.evaluate(scope, geo, qs, budget=b)),
                ("support", lambda b: ms.group(geo, [ms.values("x", {i: (i + 1) % 5 for i in range(5)})], budget=b))):
            need = limit_of(fn, resource)
            fn(Budget(**{resource: need}))
            with self.assertRaises(AlgebraBudgetError) as cm:
                fn(Budget(**{resource: need - 1}))
            self.assertEqual((cm.exception.resource, cm.exception.need, cm.exception.limit), (resource, need, need - 1))
        self.assertEqual(limit_of(lambda b: ms.support(scope, geo, budget=b), "support"), len(rows))
        self.assertEqual(limit_of(lambda b: cert.orbits(budget=b), "support"), len(set(rows)))
        self.assertEqual(limit_of(lambda b: orb.query(qs, budget=b), "output"), 4)
        with self.assertRaises(UnsupportedError):
            ms.group(geo, gens_d4(geo) * 9)

    def test_sym002_opt_in_import(self):
        out = subprocess.run([sys.executable, "-c", "import sys, torbles; print('torbles.math_sym' in sys.modules)"],
                             capture_output=True, text=True)
        self.assertEqual(out.stdout.strip(), "False")


if __name__ == "__main__":
    unittest.main()
