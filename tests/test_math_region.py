# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""ENV-002: bounded multivariate parameter regions (torbles/math_region.c, math_region_gpu.cu, math_region.py).

Oracles, independent of the implementation: every group function evaluated with Python Fractions at every point of
exhaustive rational grids of the box, at every reported vertex and at the centroids of every reported facet and
region (the tied groups: those attaining the exact minimum; the tie policy's winner; the rows of P checked exactly);
membership in each reported region by its own inequalities and equalities against the oracle's tie set; the
certifying LP's point checked against the oracle; d = 1 slices against ENV-001 (math_env, an independent univariate
implementation); the contract's enclosures against the true quadratics on grids. Every lookup also runs in the three
native modes (index, regions, all) and on the CUDA kernel (emulated here, TORBLES_EMU_SCRAMBLE=1), all compared for
equality. Seeds are fixed.
"""
import itertools
import random
import unittest
from fractions import Fraction as F

from torbles import math_env as me
from torbles import math_region as mr
from torbles.math_budgets import Budget
from torbles.math_domains import DomainError
from torbles.math_errors import InfeasibleError, NotConvergedError, UnsupportedError
from torbles.mathematics import AlgebraBudgetError, Context

MODES = ("index", "regions", "all")
BACKENDS = ("cpu", "cuda")


def value(f, p):
    """an affine (or quadratic, with the monomials theta_k theta_l, k <= l, appended) function at p"""
    d = len(p)
    v = F(f[0]) + sum(F(f[1 + k]) * p[k] for k in range(d))
    if len(f) > 1 + d:
        i = 1 + d
        for k in range(d):
            for l in range(k, d):
                v += F(f[i]) * p[k] * p[l]
                i += 1
    return v


def oracle(fs, p, upper=False, rows=None):
    """the groups attaining the minimum (maximum) at p, or None outside the rows"""
    d = len(p)
    for a, b in rows or []:
        if sum(F(a[k]) * p[k] for k in range(d)) > F(b):
            return None
    vals = {g: (-1 if upper else 1) * value(f, p) for g, f in enumerate(fs) if f is not None}
    if not vals:
        return []
    m = min(vals.values())
    return sorted(g for g, v in vals.items() if v == m)


def grid(box, n):
    axes = [[F(lo) + (F(hi) - F(lo)) * F(i, n - 1) for i in range(n)] for lo, hi in box]
    return [tuple(p) for p in itertools.product(*axes)]


def centroid(R, vids):
    d = R.d
    return tuple(sum(R.vertices[v]["point"][k] for v in vids) / len(vids) for k in range(d))


def inside(region, p):
    d = len(p)
    return all(sum(F(q["normal"][k]) * p[k] for k in range(d)) <= q["rhs"] for q in region["inequalities"]) and \
        all(sum(F(q["normal"][k]) * p[k] for k in range(d)) == q["rhs"] for q in region["equalities"])


class RegionCase(unittest.TestCase):
    def locate_all(self, R, pts, ties=64):
        """the answers of every mode and backend, checked equal; returns one of them"""
        base = None
        for mode in MODES:
            for be in BACKENDS:
                ans = R.locate(pts, mode=mode, backend=be, ties=ties)
                if base is None:
                    base = ans
                else:
                    self.assertEqual(ans, base, (mode, be))
        return base

    def check(self, fs, box, rows=None, upper=False, tie="smallest", n=5, extra=()):
        R = mr.regions(fs, box, rows, upper=upper, tie=tie)
        pts = grid(box, n) + [v["point"] for v in R.vertices] + list(extra)
        pts += [centroid(R, b["vertices"]) for b in R.boundaries] + [centroid(R, r["vertices"]) for r in R.regions]
        ans = self.locate_all(R, pts)
        for p, a in zip(pts, ans):
            o = oracle(fs, p, upper, rows)
            if o is None:
                self.assertEqual(a["status"], "outside", p)
                continue
            self.assertEqual(a["status"], "ok", p)
            groups = sorted(g for c in a["classes"] for g in R.classes[c]["groups"])
            self.assertEqual(groups, o, p)                                     # every tied group, exactly
            self.assertEqual(a["winner"], min(o) if tie == "smallest" else max(o), p)
            self.assertEqual(a["kind"], "boundary" if len(a["classes"]) > 1 else "interior", p)
            self.assertEqual(a["tied"], len(a["classes"]))
            self.assertIn(a["winner"], R.classes[a["cls"]]["groups"])
            self.assertEqual(a["region"], R.classes[a["cls"]]["region"])
            for r in R.regions:                                                # region membership, exactly
                self.assertEqual(inside(r, p), set(r["groups"]) <= set(o), (p, r["cls"]))
        for v in R.vertices:                                                   # the tie structure at the vertices
            self.assertEqual(v["groups"], oracle(fs, v["point"], upper, rows))
            w = v["groups"]
            self.assertEqual(v["winner"], min(w) if tie == "smallest" else max(w))
        for b in R.boundaries:                                                 # and on the whole facets
            c = centroid(R, b["vertices"])
            self.assertEqual(b["groups"], oracle(fs, c, upper, rows), b)
            self.assertGreaterEqual(len(b["vertices"]), R.d)
            for v in b["vertices"]:                                            # the facet's hyperplane
                p = R.vertices[v]["point"]
                self.assertEqual(sum(F(b["normal"][k]) * p[k] for k in range(R.d)), b["rhs"])
        return R


class TestRegions(RegionCase):
    def test_env002_exhaustive_grids_agree_with_membership_and_ties(self):
        rnd = random.Random(2026)
        for d in (1, 2, 3, 4):
            for trial in range(3):
                n = rnd.choice([3, 5, 8])
                fs = [tuple(rnd.randint(-3, 3) for _ in range(d + 1)) for _ in range(n)]
                fs[1] = fs[0]                                                  # coincident functions: one class
                if n > 3:
                    fs[2] = None                                               # an empty group
                box = [(rnd.randint(-2, -1), rnd.randint(1, 2)) for _ in range(d)]
                rows = [(tuple(rnd.choice([-1, 1, 2]) for _ in range(d)), 1)] if trial == 2 else None
                with self.subTest(d=d, trial=trial):
                    R = self.check(fs, box, rows, upper=trial == 1, tie="largest" if trial == 1 else "smallest",
                                   n=5 if d < 4 else 3)
                    self.assertEqual(R.claim, "exact")
                    self.assertEqual(R.classes[0]["groups"][:2], [0, 1])
                    if n > 3:
                        self.assertTrue(all(2 not in c["groups"] for c in R.classes))

    def test_env002_regions_are_exact_polytopes_with_lp_evidence(self):
        rnd = random.Random(7)
        for d in (2, 3):
            fs = [tuple(rnd.randint(-5, 5) for _ in range(d + 1)) for _ in range(7)]
            R = mr.regions(fs, [(-2, 3)] * d, rows=[((1,) * d, 2)])
            self.assertEqual(R.info["regions"], R.info["full"] + R.info["lower"])
            for r in R.regions:
                pts = [R.vertices[v]["point"] for v in r["vertices"]]
                for p in pts:                                                  # the vertices satisfy the description
                    self.assertTrue(inside(r, p))
                    self.assertIn(r["cls"], R.vertices[[v for v in r["vertices"] if R.vertices[v]["point"] == p][0]]["classes"])
                for q in r["inequalities"]:                                    # each inequality is a (relative) facet
                    on = [p for p in pts if sum(F(q["normal"][k]) * p[k] for k in range(d)) == q["rhs"]]
                    self.assertGreaterEqual(len(on), max(r["dimension"], 1))
                lp = r["lp"]
                self.assertGreater(lp["rows"], 0)
                self.assertEqual(lp["sign"] > 0, r["full"])                    # s* > 0 iff an interior point
                self.assertEqual(lp["margin"] > 0, r["full"])
                if r["full"]:                                                  # the LP's point: the class alone wins
                    self.assertEqual(oracle(fs, lp["point"], rows=[((1,) * d, 2)]), r["groups"])
                    self.assertEqual(len(r["inequalities"]) + 0, len({(q["normal"], q["rhs"]) for q in r["inequalities"]}))
                    self.assertEqual(r["equalities"], [])
                else:
                    self.assertEqual(len(r["equalities"]), d - r["dimension"])

    def test_env002_tie_structure_coincident_and_degenerate(self):
        # a fan of planes through the line x = 0 and a common point: lower-dimensional regions and many-way ties
        fs = [(0, i, -1) for i in range(-3, 4)] + [(0, 0, 0), (0, 0, 0), None]
        R = self.check(fs, [(-1, 1), (-1, 1)], n=7)
        self.assertEqual(R.classes[R.vertices[0]["classes"][0]]["groups"][0] in range(9), True)
        self.assertGreater(R.info["lower"], 0)
        for r in R.regions:
            if not r["full"]:
                self.assertLess(r["dimension"], 2)
        fs3 = [(0, i, j, 1) for i in range(-1, 2) for j in range(-1, 2)]
        self.check(fs3, [(-1, 1)] * 3, upper=True, tie="largest", n=5)
        # every class identical: one class, one region, the tie policy over its members
        same = [(1, 2, 3)] * 4
        for tie, w in (("smallest", 0), ("largest", 3)):
            R = mr.regions(same, [(-1, 1), (0, 2)], tie=tie)
            self.assertEqual(len(R.classes), 1)
            self.assertEqual(R.classes[0]["groups"], [0, 1, 2, 3])
            self.assertEqual(len(R.regions), 1)
            a = R.locate([(0, 1)])[0]
            self.assertEqual((a["winner"], a["kind"], a["tied"]), (w, "interior", 1))

    def test_env002_d1_slices_match_env001(self):
        rnd = random.Random(11)
        for trial in range(4):
            fs = [tuple(F(rnd.randint(-6, 6), rnd.randint(1, 3)) for _ in range(3)) for _ in range(6)]
            R = mr.regions(fs, [(-3, 3), (-3, 3)])
            for y0 in (F(-3), F(-1, 2), F(0), F(7, 3), F(3)):
                env = me.envelope([(f[0] + f[2] * y0, f[1]) for f in fs], lo=-3, hi=3)
                for r in env.regions:                                          # the ENV-001 pieces and points
                    if r["kind"] == "point":
                        if not isinstance(r["at"], F):
                            continue
                        t = r["at"]
                    else:
                        lo = r["lo"] if r["lo"] is not None else F(-3)
                        hi = r["hi"] if r["hi"] is not None else F(3)
                        t = (lo + hi) / 2
                    a = R.locate([(t, y0)], ties=16)[0]
                    self.assertEqual(sorted(g for c in a["classes"] for g in R.classes[c]["groups"]), r["groups"],
                                     (trial, y0, t))
        # d = 1 itself against ENV-001's breakpoints
        fs = [(0, 1), (2, -1), (1, 0), (F(1, 2), F(1, 2))]
        R = mr.regions(fs, [(-4, 4)])
        env = me.envelope(fs, lo=-4, hi=4)
        self.assertEqual(sorted(v["point"][0] for v in R.vertices), sorted(r["at"] for r in env.regions if r["kind"] == "point"))

    def test_env002_point_location_modes_and_cuda_agree(self):
        rnd = random.Random(5)
        for d in (2, 3, 4):
            fs = []
            for _ in range(40 if d < 4 else 16):                               # tangent planes: many regions
                p = [F(rnd.randint(-9, 9), 10) for _ in range(d)]
                fs.append(tuple([-sum(q * q for q in p)] + [2 * q for q in p]))
            R = mr.regions(fs, [(-1, 1)] * d, upper=True)
            pts = [tuple(F(rnd.randint(-20, 20), 20) for _ in range(d)) for _ in range(150)]
            pts += [v["point"] for v in R.vertices[:50]]
            ans = self.locate_all(R, pts)
            for p, a in zip(pts, ans):
                self.assertEqual(sorted(g for c in a["classes"] for g in R.classes[c]["groups"]), oracle(fs, p, True))
            self.assertIn(R.last_lookup["backend"], ("cuda-emulated", "cuda"))
        # values past int64: the big-rational path, the same answers
        big = 10 ** 30
        fs = [(0, 1, -1), (1, -1, 0), (F(1, big), 0, 1)]
        R = mr.regions(fs, [(-2, 2), (-2, 2)])
        pts = [(F(big + 1, big), F(-1, 3)), (F(1, big), F(-big, big + 7)), (0, 0)]
        ans = self.locate_all(R, pts)
        for p, a in zip(pts, ans):
            self.assertEqual(sorted(g for c in a["classes"] for g in R.classes[c]["groups"]), oracle(fs, p))
        self.assertGreater(R.last_lookup["primes"], 1)

    def test_env002_int64_value_arrays_and_array_answers_equal_the_exact_path(self):
        rnd = random.Random(8)
        fs = [tuple(rnd.randint(-9, 9) for _ in range(3)) for _ in range(30)] + [(1, 2, 3), (1, 2, 3)]
        box = [(-2, 2), (-1, 3)]
        R = mr.regions(fs, box)
        den = [[rnd.randint(1, 30), 30] for _ in range(300)] + [[30, 30], [30, 30]]
        num = [[rnd.randint(-2 * p, 2 * p), rnd.randint(-30, 90)] for p, _ in den[:300]] + [[-61, 0], [0, 91]]
        pts = [(F(a, p), F(b, q)) for (a, b), (p, q) in zip(num, den)]
        V = mr.Values(num, den)
        none = lambda x: -1 if x is None else x
        for mode in MODES:
            for be in BACKENDS:
                ref = R.locate(pts, mode=mode, backend=be, ties=4)
                self.assertEqual(R.locate(V, mode=mode, backend=be, ties=4), ref)
                arr = R.locate_arrays(V, mode=mode, backend=be, ties=4)
                self.assertEqual(arr["status"].tolist(), [{"ok": 0, "outside": 1, "empty": 2}[a["status"]] for a in ref])
                self.assertEqual(arr["kind"].tolist(), [{None: 0, "interior": 1, "boundary": 2}[a["kind"]] for a in ref])
                for key in ("cell", "winner", "cls", "region", "tied"):
                    self.assertEqual(arr[key].tolist(), [none(a[key]) for a in ref], key)
                self.assertEqual(arr["classes"].tolist(), [a["classes"] + [-1] * (4 - len(a["classes"])) for a in ref])
        self.assertEqual([a["status"] for a in ref[-2:]], ["outside", "outside"])
        self.assertEqual({a["status"] for a in ref[:300]}, {"ok"})
        for p, a in zip(pts[:300], ref):                                       # the oracle at every value inside
            self.assertEqual(sorted(g for c in a["classes"] for g in R.classes[c]["groups"]), oracle(fs, p))
        self.assertEqual(R.locate(mr.Values([[1, 2], [3, -4]], 7)), R.locate([(F(1, 7), F(2, 7)), (F(3, 7), F(-4, 7))]))
        self.assertEqual(R.locate_need(V), R.locate_need(pts))
        with self.assertRaises(TypeError):                                     # floats are refused
            mr.Values([[0.5, 1.0]])
        with self.assertRaises(ValueError):                                    # a denominator below 1 (native)
            R.locate(mr.Values([[1, 2]], 0))
        with self.assertRaises(ValueError):                                    # the wrong dimension
            R.locate(mr.Values([[1, 2, 3]]))
        R.locate_arrays(V, budget=Budget(output=len(V)))
        with self.assertRaises(AlgebraBudgetError):                            # output = values, one past
            R.locate_arrays(V, budget=Budget(output=len(V) - 1))

    def test_env002_unsupported_nonlinear_not_approximated_without_contract(self):
        q = [(0, 0, 0, 1, 0, 1), (F(1, 2), -1, 0, 0, 0, 0), (0, 1, 1, 0, F(1, 2), 0)]   # quadratics in (x, y)
        box = [(-1, 1), (-1, 1)]
        with self.assertRaises(UnsupportedError):                              # never approximated silently
            mr.regions(q, box)
        ctx = Context()
        a, b, z = ctx.symbols("a", "b", "z")
        with self.assertRaises(UnsupportedError):
            mr.from_expressions([a * a + b, 1 - a], ["a", "b"], box)
        with self.assertRaises(UnsupportedError):                              # degree 3: never
            mr.from_expressions([a * a * b, 1 - a], ["a", "b"], box, contract=mr.PiecewiseLinear(2, 10))
        with self.assertRaises(DomainError):                                   # a variable outside the parameters
            mr.from_expressions([a + z, 1 - a], ["a", "b"], box)
        with self.assertRaises(NotConvergedError):                             # a bound not achieved is refused
            mr.regions(q, box, contract=mr.PiecewiseLinear(2, F(1, 100)))
        for K in (1, 2, 4):
            R = mr.regions(q, box, contract=mr.PiecewiseLinear(K, 4))
            self.assertEqual(R.claim, "approximate")
            self.assertEqual(R.contract["cells"], K * K)
            self.assertEqual(R.envelope_error, 2 * R.error)
            self.assertEqual(R.error, max(c["eps"] for c in R.classes))
            self.assertFalse(R.decide()["authorized"])
            with self.assertRaises(Exception) as cm:
                R.authorize_exact()
            self.assertEqual(cm.exception.reason, "approximate")
            pts = grid(box, 9)
            for p, ans in zip(pts, self.locate_all(R, pts, ties=8)):
                tv = [value(f, p) for f in q]
                self.assertLessEqual(tv[ans["winner"]] - min(tv), R.envelope_error, p)   # the certified error
                cls = R.classes[ans["cls"]]
                L = cls["coefficients"][0] + sum(cls["coefficients"][1 + k] * p[k] for k in range(2))
                self.assertLessEqual(abs(tv[ans["winner"]] - L), cls["eps"], p)        # the enclosure
                for c in range(len(R.classes)):                                        # every enclosure of the cell
                    k = R.classes[c]
                    if k["cell"] != ans["cell"]:
                        continue
                    L = k["coefficients"][0] + sum(k["coefficients"][1 + t] * p[t] for t in range(2))
                    for g in k["groups"]:
                        self.assertLessEqual(abs(value(q[g], p) - L), k["eps"])
        # the contract over affine functions changes nothing: eps = 0, exact
        R = mr.regions([(0, 1, 1), (1, -1, 0)], box, contract=mr.PiecewiseLinear(2, 0))
        self.assertEqual((R.claim, R.error), ("exact", 0))
        self.assertTrue(R.authorize_exact()["authorized"])
        self.assertTrue(mr.regions([(0, 1, 1), (1, -1, 0)], box).decide()["authorized"])

    def test_env002_budgets_at_the_limit_and_past_it(self):
        fs = []
        rnd = random.Random(3)
        for _ in range(12):
            p = [F(rnd.randint(-9, 9), 10) for _ in range(2)]
            fs.append(tuple([-sum(x * x for x in p)] + [2 * x for x in p]))
        box = [(-1, 1), (-1, 1)]
        R = mr.regions(fs, box, upper=True)
        info = R.info
        counted = dict(output=info["regions"], support=info["incidences"], circuit=info["nodes"],
                       iterations=info["iterations"], variables=2, states=12)
        for resource, value_ in counted.items():
            with self.subTest(resource=resource):
                self.assertEqual(mr.regions(fs, box, upper=True, budget=Budget(**{resource: value_})).info,
                                 info)                                         # at the limit
                with self.assertRaises(AlgebraBudgetError) as cm:
                    mr.regions(fs, box, upper=True, budget=Budget(**{resource: value_ - 1}))
                self.assertEqual(cm.exception.resource, resource)
                self.assertEqual(cm.exception.need, value_)                    # refused at limit + 1
        with self.assertRaises(AlgebraBudgetError) as cm:                     # bytes: the largest amount checked
            mr.regions(fs, box, upper=True, budget=Budget(bytes=info["bytes"] - 1))
        self.assertEqual(cm.exception.resource, "bytes")
        self.assertEqual(mr.regions(fs, box, upper=True, budget=Budget(bytes=info["bytes"])).info, info)
        self.assertEqual(mr.need(fs, box, upper=True)["variables"], 2)
        with self.assertRaises(AlgebraBudgetError) as cm:                     # the degree of a contract build
            mr.regions([(0, 0, 0, 1, 0, 1)], box, contract=mr.PiecewiseLinear(1, 9), budget=Budget(degree=1))
        self.assertEqual((cm.exception.resource, cm.exception.need), ("degree", 2))
        pts = [(F(i, 5), F(j, 5)) for i in range(-5, 6) for j in range(-5, 6)]
        for be in BACKENDS:                                                    # lookups: output, bytes
            R.locate(pts, backend=be, budget=Budget(output=len(pts)))
            with self.assertRaises(AlgebraBudgetError) as cm:
                R.locate(pts, backend=be, budget=Budget(output=len(pts) - 1))
            self.assertEqual((cm.exception.resource, cm.exception.need), ("output", len(pts)))
            need = R.locate_need(pts, backend=be, ties=4)
            R.locate(pts, backend=be, ties=4, budget=Budget(bytes=need["bytes"]))
            with self.assertRaises(AlgebraBudgetError) as cm:
                R.locate(pts, backend=be, ties=4, budget=Budget(bytes=need["bytes"] - 1))
            self.assertEqual(cm.exception.resource, "bytes")
        self.assertGreater(R.locate_need(pts, backend="cuda")["bytes"], R.locate_need(pts, backend="cpu")["bytes"])

    def test_env002_refusals_and_declared_domain(self):
        fs = [(0, 1, 1), (1, -1, 0)]
        with self.assertRaises(InfeasibleError):                               # P empty
            mr.regions(fs, [(-1, 1), (-1, 1)], rows=[((1, 0), -2)])
        with self.assertRaises(UnsupportedError):                              # P without interior
            mr.regions(fs, [(-1, 1), (-1, 1)], rows=[((1, 0), 0), ((-1, 0), 0)])
        with self.assertRaises(UnsupportedError):                              # d above 4
            mr.regions([(0,) * 6], [(-1, 1)] * 5)
        with self.assertRaises(TypeError):                                     # floats are not exact
            mr.regions([(0.5, 1, 1)], [(-1, 1), (-1, 1)])
        with self.assertRaises(ValueError):                                    # lo >= hi
            mr.regions(fs, [(1, 1), (-1, 1)])
        with self.assertRaises(ValueError):                                    # a row without coefficients
            mr.regions(fs, [(-1, 1), (-1, 1)], rows=[((0, 0), 1)])
        with self.assertRaises(UnsupportedError):                              # too many rows
            mr.regions(fs, [(-1, 1), (-1, 1)], rows=[((1, 1), 5)] * 65)
        R = mr.regions(fs, [(-1, 1), (-1, 1)], rows=[((1, 1), 1)])
        out = self.locate_all(R, [(1, 1), (2, 0), (0, 0)])
        self.assertEqual([a["status"] for a in out], ["outside", "outside", "ok"])
        E = mr.regions([None, None], [(-1, 1), (-1, 1)])
        self.assertEqual((E.kind, len(E.regions)), ("empty", 0))
        self.assertEqual([a["status"] for a in self.locate_all(E, [(0, 0)])], ["empty"])

    def test_env002_maximizer_and_expressions(self):
        rnd = random.Random(9)
        fs = [tuple(rnd.randint(-4, 4) for _ in range(3)) for _ in range(6)]
        up = mr.regions(fs, [(-2, 2), (-1, 2)], upper=True)
        neg = mr.regions([tuple(-x for x in f) for f in fs], [(-2, 2), (-1, 2)])
        self.assertEqual([r["groups"] for r in up.regions], [r["groups"] for r in neg.regions])
        self.assertEqual([v["point"] for v in up.vertices], [v["point"] for v in neg.vertices])
        ctx = Context()
        a, b = ctx.symbols("a", "b")
        ex = mr.from_expressions([a + b, 1 - a, None, 2 - b * F(1, 2)], ["a", "b"], [(-2, 2), (-2, 2)])
        co = mr.regions([(0, 1, 1), (1, -1, 0), None, (2, 0, F(-1, 2))], [(-2, 2), (-2, 2)])
        self.assertEqual([c["coefficients"] for c in ex.classes], [c["coefficients"] for c in co.classes])
        self.assertEqual([r["vertices"] for r in ex.regions], [r["vertices"] for r in co.regions])


if __name__ == "__main__":
    unittest.main()
