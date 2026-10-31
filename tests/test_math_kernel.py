# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""KERNEL-001: exact finite-feature kernel embeddings (torbles/math_kernel.c, math_kernel_gpu.cu, math_kernel.py).

Oracles, independent of the implementation: explicit kernel matrices written here with Python integers (every pair
(i, l) of the selected rows of two groups, k evaluated directly: (x.y + c)^d, sum_t w_t x^a y^a, joint category
equality with NULL matching NULL), their weighted sums with and without the diagonal, unordered halves, and the means,
MMDs (biased and unbiased) and witness values formed with Fractions; DuckDB for the grouped weight sums of an engine
table. Every native result is also computed on the CUDA path (the device build and the device requests, emulated
here with TORBLES_EMU_SCRAMBLE=1) and compared for equality. Seeds are fixed.
"""
import itertools
import subprocess
import sys
import unittest
from fractions import Fraction

import numpy as np
import pandas as pd

import torbles
from torbles import math_bank as mbk, math_kernel as mk, math_tensor as mt
from torbles.math_budgets import Budget
from torbles.math_domains import DomainError
from torbles.math_errors import StaleStateError, UnsupportedError
from torbles.mathematics import AlgebraBudgetError


def make(rng, n=36, ngroups=4, absent=2, nulls=0.12):
    vals = {"x": rng.integers(-3, 4, n), "y": rng.integers(-2, 3, n), "z": rng.integers(0, 3, n),
            "c": rng.integers(0, 3, n), "w": rng.integers(-2, 5, n)}
    valid = {k: rng.random(n) >= nulls for k in ("x", "y", "c", "w")}
    groups = rng.integers(0, ngroups, n)
    groups[groups == absent] = 0
    rows = rng.random(n) < 0.9
    scope = mbk.Scope.from_arrays(vals, valid=valid, groups=groups, ngroups=ngroups, rows=rows)
    py = {k: [None if (k in valid and not valid[k][r]) else int(vals[k][r]) for r in range(n)] for k in vals}
    return scope, py, [int(g) for g in groups], [bool(r) for r in rows]


def poly(d, c):
    return lambda a, b: (sum(x * y for x, y in zip(a, b)) + c) ** d


def mono(terms):
    def k(a, b):
        out = 0
        for expo, w in terms:
            p = w
            for x, y, e in zip(a, b, expo):
                p *= (x * y) ** e
            out += p
        return out
    return k


def match(a, b):
    return int(all(x == y for x, y in zip(a, b)))


class Oracle:
    """the contributing rows of each group (weights, points) and explicit kernel matrices over them"""

    def __init__(self, py, groups, rows, ngroups, cols, weight, categorical=False):
        self.members = []
        for g in range(ngroups):
            m = []
            for r in range(len(groups)):
                if not rows[r] or groups[r] != g:
                    continue
                w = 1 if weight is None else py[weight][r]
                if w is None:
                    continue
                point = tuple(py[c][r] for c in cols)
                if not categorical and any(v is None for v in point):
                    continue
                m.append((w, point))
            self.members.append(m)

    def W(self, g):
        return sum(w for w, _ in self.members[g])

    def Q(self, g):
        return sum(w * w for w, _ in self.members[g])

    def ksum(self, k, a, b, self_pairs=True):
        A, B = self.members[a], self.members[b]
        return sum(wi * wl * k(xi, xl) for i, (wi, xi) in enumerate(A) for l, (wl, xl) in enumerate(B)
                   if self_pairs or a != b or i != l)

    def mmd2(self, k, a, b, unbiased):
        Wa, Wb = self.W(a), self.W(b)
        Pa, Pb = (Wa * Wa - self.Q(a), Wb * Wb - self.Q(b)) if unbiased else (Wa * Wa, Wb * Wb)
        if not Pa or not Pb or not Wa * Wb:
            return None
        return (Fraction(self.ksum(k, a, a, not unbiased), Pa) + Fraction(self.ksum(k, b, b, not unbiased), Pb)
                - 2 * Fraction(self.ksum(k, a, b), Wa * Wb))

    def witness(self, k, a, z):
        Wa = self.W(a)
        return None if not Wa else Fraction(sum(w * k(x, z) for w, x in self.members[a]), Wa)


def requests_for(G, npoints):
    reqs = []
    for a in range(G):
        for b in range(G):
            reqs.append(dict(op="sum", a=a, b=b))
            reqs.append(dict(op="mean", a=a, b=b))
            reqs.append(dict(op="mmd2", a=a, b=b, unbiased=True))
            reqs.append(dict(op="mmd2", a=a, b=b, unbiased=False))
        for self_pairs in (False, True):
            for ordered in (True, False):
                reqs.append(dict(op="sum", a=a, self_pairs=self_pairs, ordered=ordered))
                reqs.append(dict(op="mean", a=a, self_pairs=self_pairs, ordered=ordered))
        for p in range(npoints):
            reqs.append(dict(op="witness", a=a, point=p))
    return reqs


class Base(unittest.TestCase):
    def check_answers(self, st, oracle, k, points, kernel=None):
        G = st.info["groups"]
        reqs = requests_for(G, len(points))
        cpu = st.query(reqs, kernel=kernel, points=points)
        gpu = st.query(reqs, kernel=kernel, points=points, backend="cuda")
        self.assertEqual(cpu, gpu)                                          # the device answers bit for bit
        for q, r in zip(reqs, cpu):
            a, b = q["a"], q.get("b", q["a"])
            if q["op"] == "witness":
                self.assertEqual(r["value"], oracle.witness(k, a, points[q["point"]]), q)
                continue
            if q["op"] == "mmd2":
                self.assertEqual(r["value"], oracle.mmd2(k, a, b, q["unbiased"]), q)
                continue
            self_pairs, ordered = q.get("self_pairs", False), q.get("ordered", True)
            total = oracle.ksum(k, a, b, self_pairs or a != b)
            pairs = oracle.W(a) * oracle.W(b) - (0 if (self_pairs or a != b) else oracle.Q(a))
            if a == b and not ordered:                     # unordered pairs: the off-diagonal half (+ the diagonal)
                diag = oracle.ksum(k, a, a) - oracle.ksum(k, a, a, False)
                off = oracle.ksum(k, a, a, False)
                total = off // 2 + (diag if self_pairs else 0)
                self.assertEqual(off % 2, 0)
                pairs = (oracle.W(a) ** 2 - oracle.Q(a)) // 2 + (oracle.Q(a) if self_pairs else 0)
            if q["op"] == "sum":
                self.assertEqual((r["value"], r["pairs"]), (total, pairs), q)
            else:
                self.assertEqual(r["value"], None if pairs == 0 else Fraction(total, pairs), q)


# ====================================================================== KERNEL-001
class Kernels(Base):
    # ------------------------------------------------------------------ acceptance 1: explicit kernel matrices
    def test_kernel001_polynomial_kernels_match_explicit_matrices_with_weights_and_diagonals(self):
        rng = np.random.default_rng(3001)
        scope, py, groups, rows = make(rng)
        G = scope.groups
        points = [(1, 2), (0, 0), (-3, 1)]
        for d, c in ((0, 0), (1, 0), (2, 1), (3, -2), (4, 3), (2, 0)):
            k = poly(d, c)
            for weight in (None, "w"):
                oracle = Oracle(py, groups, rows, G, ["x", "y"], weight)
                for repr_ in ("dense", "sparse"):
                    st = mk.features(scope, ["x", "y"], mk.polynomial(d, c), weight=weight, repr=repr_)
                    dev = mk.features(scope, ["x", "y"], mk.polynomial(d, c), weight=weight, repr=repr_, backend="cuda")
                    self.assertIn(dev.backend, ("cuda", "cuda-emulated"))
                    self.assertTrue(st.tensor.same(dev.tensor))
                    for g in range(G):
                        grp = st.group(g)
                        self.assertEqual(grp, dev.group(g))
                        self.assertEqual((grp["weight"], grp["squares"]), (oracle.W(g), oracle.Q(g)))
                        self.assertEqual(grp["self"], sum(w * w * k(x, x) for w, x in oracle.members[g]))
                        self.assertEqual(grp["used"], len(oracle.members[g]))
                    self.check_answers(st, oracle, k, points)
                    self.check_answers(dev, oracle, k, points)
        # the convenience calls
        st = mk.features(scope, ["x", "y"], mk.polynomial(2, 1), weight="w")
        oracle = Oracle(py, groups, rows, G, ["x", "y"], "w")
        k = poly(2, 1)
        self.assertEqual(st.sum(0, 1), dict(value=oracle.ksum(k, 0, 1), pairs=oracle.W(0) * oracle.W(1)))
        self.assertEqual(st.mmd2(0, 3, unbiased=False), oracle.mmd2(k, 0, 3, False))
        self.assertEqual(st.witness(1, points), [oracle.witness(k, 1, z) for z in points])
        self.assertEqual(st.mean(0, 1, backend="cuda"), Fraction(oracle.ksum(k, 0, 1), oracle.W(0) * oracle.W(1)))

    def test_kernel001_monomial_and_delta_kernels_match_explicit_matrices(self):
        rng = np.random.default_rng(3002)
        scope, py, groups, rows = make(rng, n=40)
        G = scope.groups
        terms = [({"x": 1, "y": 1}, 3), ({"x": 2}, -1), ({}, 5), ({"y": 3, "x": 1}, 2), ({"x": 1, "y": 1}, 1)]
        expo = [((1, 1), 4), ((2, 0), -1), ((0, 0), 5), ((1, 3), 2)]     # merged: equal monomials add
        k = mono(expo)
        for weight in (None, "w"):
            oracle = Oracle(py, groups, rows, G, ["x", "y"], weight)
            for repr_ in ("dense", "sparse"):
                st = mk.features(scope, ["x", "y"], mk.monomial(terms, ["x", "y"]), weight=weight, repr=repr_)
                self.assertEqual(st.info["degree"], 4)
                dev = mk.features(scope, ["x", "y"], mk.monomial(terms, ["x", "y"]), weight=weight, repr=repr_,
                                  backend="cuda")
                self.assertTrue(st.tensor.same(dev.tensor))
                for g in range(G):
                    self.assertEqual(st.group(g), dev.group(g))
                    self.assertEqual(st.group(g)["self"], sum(w * w * k(x, x) for w, x in oracle.members[g]))
                self.check_answers(st, oracle, k, [(2, -1), (0, 3)])
        # delta: joint category equality, NULL matching NULL, rows with NULL categories included
        for cols in (["c"], ["c", "x"], ["x", "c", "z"]):
            for weight in (None, "w"):
                oracle = Oracle(py, groups, rows, G, cols, weight, categorical=True)
                for repr_ in ("sparse", "dense"):
                    st = mk.features(scope, cols, mk.delta(), weight=weight, repr=repr_)
                    dev = mk.features(scope, cols, mk.delta(), weight=weight, repr=repr_, backend="cuda")
                    self.assertTrue(st.tensor.same(dev.tensor))
                    self.assertEqual(st.tensor.kind, "count")
                    pts = [tuple([None] * len(cols)), tuple([1] * len(cols)), tuple([99] * len(cols))]
                    self.check_answers(st, oracle, match, pts)
                    self.check_answers(dev, oracle, match, pts)

    def test_kernel001_engine_table_decimals_and_duckdb_weights(self):
        rng = np.random.default_rng(3003)
        n = 120
        frame = pd.DataFrame({
            "k": pd.Series([int(v) for v in rng.integers(0, 3, n)], dtype=object),
            "a": pd.Series([f"{int(v) / 100:.2f}" for v in rng.integers(-300, 300, n)], dtype=object),
            "b": pd.Series([f"{int(v) / 100:.2f}" if rng.random() > 0.1 else None for v in rng.integers(-300, 300, n)],
                           dtype=object),
            "w": pd.Series([int(v) if rng.random() > 0.1 else None for v in rng.integers(1, 9, n)], dtype=object)})
        db = torbles.from_ddl("CREATE TABLE f (k BIGINT, a DECIMAL(10,2), b DECIMAL(10,2), w BIGINT)", {"f": frame})
        t = db["f"]
        scope = mbk.Scope.from_table(t, ["a", "b", "w"], group_by=["k"])
        st = mk.features(scope, ["a", "b"], mk.polynomial(2, 10000), weight="w")
        dev = mk.features(scope, ["a", "b"], mk.polynomial(2, 10000), weight="w", backend="cuda")
        sql = ('SELECT k, SUM(w), SUM(w * w), COUNT(*) FROM f WHERE a IS NOT NULL AND b IS NOT NULL AND w IS NOT NULL '
               'GROUP BY k')
        duck = {r[0]: r[1:] for r in db.con.execute(sql).fetchall()}
        rowsel = [True] * n
        py = {"a": [None if v is None else round(float(v) * 100) for v in frame["a"]],
              "b": [None if v is None else round(float(v) * 100) for v in frame["b"]],
              "w": list(frame["w"])}
        groups = [int(v) for v in frame["k"]]
        oracle = Oracle(py, groups, rowsel, 3, ["a", "b"], "w")
        k = poly(2, 10000)
        for g in range(scope.groups):
            key = scope.key_values(g)[0]
            got = st.group(g)
            if key in duck:
                self.assertEqual((got["weight"], got["squares"], got["used"]), tuple(int(x) for x in duck[key]))
            self.assertEqual(got, dev.group(g))
        for a, b in itertools.product(range(scope.groups), repeat=2):
            ga, gb = scope.key_values(a)[0], scope.key_values(b)[0]
            if ga is None or gb is None:
                continue
            self.assertEqual(st.sum(a, b, self_pairs=True)["value"], oracle.ksum(k, ga, gb))
        self.assertEqual(st.query([dict(op="mmd2", a=0, b=1, unbiased=True)]),
                         dev.query([dict(op="mmd2", a=0, b=1, unbiased=True)], backend="cuda"))
        mixed = mbk.Scope.from_table(t, ["a", "w"])
        with self.assertRaises(DomainError):                          # x.y mixes decimal scales
            mk.features(mixed, ["a", "w"], mk.linear())
        t.flip(0, 0)
        with self.assertRaises(StaleStateError):
            mk.features(scope, ["a", "b"], mk.linear())

    # ------------------------------------------------------------------ acceptance 2: degree, dimensions, refusals
    def test_kernel001_required_degree_and_dimensions_are_verified_never_approximated(self):
        rng = np.random.default_rng(3004)
        scope, py, groups, rows = make(rng, n=40)
        G = scope.groups
        st = mk.features(scope, ["x", "y"], mk.polynomial(3, 1), weight="w", degree=4)
        self.assertEqual((st.info["degree"], st.info["order"]), (3, 4))
        # every retained kernel is exact on the state: a lower degree, another c, a subset of the columns
        for kern, cols, k in ((mk.polynomial(2, -1), ["x", "y"], poly(2, -1)), (mk.linear(["x"]), ["x"], poly(1, 0)),
                              (mk.polynomial(4, 2, ["y", "x"]), ["y", "x"], poly(4, 2)),
                              (mk.monomial([({"x": 2, "y": 2}, 1), ({"y": 1}, 3)]), ["x", "y"],
                               mono([((2, 2), 1), ((0, 1), 3)]))):
            oracle = Oracle(py, groups, rows, G, cols, "w")
            # restricted to complete rows of the state's columns: x and y
            both = Oracle(py, groups, rows, G, ["x", "y"], "w")
            idx = [["x", "y"].index(cc) for cc in cols]
            oracle.members = [[(w, tuple(p[i] for i in idx)) for w, p in m] for m in both.members]
            reqs = [dict(op="sum", a=a, b=b, self_pairs=True) for a in range(G) for b in range(G)]
            reqs += [dict(op="mmd2", a=0, b=1, unbiased=False), dict(op="witness", a=0, point=0)]
            pts = [tuple(range(1, len(cols) + 1))]
            cpu = st.query(reqs, kernel=kern, points=pts)
            self.assertEqual(cpu, st.query(reqs, kernel=kern, points=pts, backend="cuda"))
            for q, r in zip(reqs, cpu):
                if q["op"] == "sum":
                    self.assertEqual(r["value"], oracle.ksum(k, q["a"], q["b"]))
                elif q["op"] == "mmd2":
                    self.assertEqual(r["value"], oracle.mmd2(k, 0, 1, False))
                else:
                    self.assertEqual(r["value"], oracle.witness(k, 0, pts[0]))
            # the self-pair term belongs to the declared kernel: excluding it with another kernel is refused
            with self.assertRaises(UnsupportedError):
                st.sum(0, kernel=kern)
            raw = st.query([dict(op="sum", a=0), dict(op="mmd2", a=0, b=1, unbiased=True)], kernel=kern, raw=True)
            self.assertEqual([r["status"] for r in raw], [-12, -12])
        # absent degree, absent dimension, another family: refused natively, never approximated
        for bad in (mk.polynomial(5, 1), mk.polynomial(2, 0, ["x", "z"]), mk.delta(["x", "y"]),
                    mk.monomial([({"x": 5}, 1)])):
            with self.assertRaises(UnsupportedError):
                st.sum(0, 1, kernel=bad)
        counts = mk.features(scope, ["c", "x"], mk.delta())
        for bad in (mk.linear(["c", "x"]), mk.delta(["c"]), mk.delta(["x", "c"])):
            with self.assertRaises(UnsupportedError):
                counts.sum(0, 1, kernel=bad)
        # kernels without an exact finite feature map are refused at once (KERNEL-002 approximates explicitly)
        for name in ("rbf", "gaussian", "laplace", "sigmoid"):
            with self.assertRaises(UnsupportedError):
                mk.features(scope, ["x", "y"], mk.kernel(name, gamma=1))
            with self.assertRaises(UnsupportedError):
                mk.features(scope, ["x", "y"], mk.kernel(name, gamma=1), backend="cuda")
        for exc, fn in ((UnsupportedError, lambda: mk.features(scope, ["x"], mk.polynomial(9))),
                        (UnsupportedError, lambda: mk.features(scope, ["x"], mk.monomial([({"x": 9}, 1)]))),
                        (ValueError, lambda: mk.features(scope, ["x"], mk.polynomial(2), degree=1)),
                        (ValueError, lambda: mk.features(scope, ["x", "x"], mk.linear())),
                        (ValueError, lambda: mk.features(scope, ["x"], mk.monomial([({"x": 1}, 1), ({"x": 1}, -1)]))),
                        (ValueError, lambda: mk.features(scope, ["q"], mk.linear())),
                        (OverflowError, lambda: mk.features(scope, ["x"], mk.polynomial(8, 2 ** 20))),
                        (ValueError, lambda: st.witness(0, [(1,)]))):
            with self.assertRaises(exc):
                fn()

    def test_kernel001_exactness_proofs_and_large_values(self):
        big = 2 ** 30
        x = np.array([big, -big, big - 7, 3], dtype=np.int64)
        y = np.array([-big, big, 5, big - 1], dtype=np.int64)
        scope = mbk.Scope.from_arrays({"x": x, "y": y}, groups=[0, 0, 1, 1], ngroups=2)
        k = poly(3, 2 ** 40)
        oracle = Oracle({"x": x.tolist(), "y": y.tolist()}, [0, 0, 1, 1], [True] * 4, 2, ["x", "y"], None)
        for repr_ in ("dense", "sparse"):
            st = mk.features(scope, ["x", "y"], mk.polynomial(3, 2 ** 40), repr=repr_)
            dev = mk.features(scope, ["x", "y"], mk.polynomial(3, 2 ** 40), repr=repr_, backend="cuda")
            self.check_answers(st, oracle, k, [(big, big)])
            self.check_answers(dev, oracle, k, [(big, big)])
        wide = mbk.Scope.from_arrays({"x": np.array([2 ** 40, 1], dtype=np.int64)})
        with self.assertRaises(OverflowError):                        # the moment tensor's proof (TENSOR-001)
            mk.features(wide, ["x"], mk.polynomial(4))
        with self.assertRaises(OverflowError):
            mk.features(wide, ["x"], mk.polynomial(4), backend="cuda")

    def test_kernel001_budgets_at_the_limit_and_one_past(self):
        rng = np.random.default_rng(3005)
        scope, py, groups, rows = make(rng, n=30)
        st = mk.features(scope, ["x", "y"], mk.polynomial(2, 1), weight="w")

        def limit_of(fn, resource):
            try:
                fn(Budget(**{resource: 0}))
            except AlgebraBudgetError as e:
                self.assertEqual(e.resource, resource)
                return e.need
            self.fail(f"no {resource} need")

        reqs = [dict(op="sum", a=0, b=1)] * 7

        def fresh():                                  # an evaluation is budgeted when a kernel is first used
            return mk.features(scope, ["x", "y"], mk.polynomial(2, 1), weight="w")
        for resource, fn in (
                ("tensor", lambda b: mk.features(scope, ["x", "y"], mk.polynomial(3, 1), budget=b)),
                ("bytes", lambda b: mk.features(scope, ["x", "y"], mk.polynomial(3, 1), budget=b)),
                ("bytes", lambda b: mk.features(scope, ["x", "y"], mk.polynomial(2), repr="sparse", backend="cuda",
                                                budget=b)),
                ("variables", lambda b: mk.features(scope, ["x", "y"], mk.linear(), weight="w", budget=b)),
                ("output", lambda b: st.query(reqs, budget=b)),
                ("output", lambda b: st.query(reqs, budget=b, backend="cuda")),
                ("support", lambda b: fresh().query(reqs, budget=b)),
                ("support", lambda b: fresh().query(reqs, budget=b, backend="cuda")),
                ("degree", lambda b: fresh().query(reqs, kernel=mk.linear(["x"]), budget=b))):
            need = limit_of(fn, resource)
            fn(Budget(**{resource: need}))
            with self.assertRaises(AlgebraBudgetError) as cm:
                fn(Budget(**{resource: need - 1}))
            self.assertEqual((cm.exception.resource, cm.exception.need, cm.exception.limit), (resource, need, need - 1))
        self.assertEqual(mk.need(scope, ["x", "y"], mk.polynomial(2, 1))["tensor"], scope.groups * 9)
        # the evaluation reads the C(n + d, d) sorted multi-indices of the symmetric tensor, not its (n + 1)^d cells
        self.assertEqual(limit_of(lambda b: fresh().query(reqs, budget=b), "support"), 6)
        cubic = mk.polynomial(3, 1)
        self.assertEqual(limit_of(lambda b: mk.features(scope, ["x", "y", "z"], cubic).query(reqs, budget=b), "support"),
                         20)

    def test_kernel001_opt_in_import(self):
        out = subprocess.run([sys.executable, "-c", "import sys, torbles; print('torbles.math_kernel' in sys.modules)"],
                             capture_output=True, text=True)
        self.assertEqual(out.stdout.strip(), "False")


if __name__ == "__main__":
    unittest.main()
