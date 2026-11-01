# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""PAIR-002: structured pairwise query planning (torbles/math_pairplan.c, math_pairplan_gpu.cu, math_pairplan.py).

Oracles, independent of the implementation: explicit enumeration of the declared pair set over the elements of each
group, written here with Python integers and Fractions (every pair (i, j) of rows, k evaluated directly from its
definition, the weighted pair count summed pair by pair), the closed forms m^2, m (m - 1), m (m - 1) / 2,
m (m + 1) / 2 and W^2, W^2 - Q, (W^2 -+ Q) / 2 for multiplicities, DuckDB self-joins over the rows (pair counts and
sums in HUGEINT), and the PAIR-001 (math_pair) and KERNEL-001 (math_kernel) results where they apply. Every native
result is computed by every admissible strategy (support aggregation, finite features dense and sparse, sorted
prefix, direct pairs) and every CUDA candidate (emulated here, TORBLES_EMU_SCRAMBLE=1 scrambles the thread order) is
compared bit for bit with its CPU twin. Seeds are fixed.
"""
import itertools
import subprocess
import sys
import unittest
from fractions import Fraction

import duckdb
import numpy as np
import pandas as pd

import torbles
from torbles import math_bank as mbk, math_dist as md, math_kernel as mk, math_pair as mpr, math_pairplan as pp
from torbles.math_budgets import Budget
from torbles.math_domains import DomainError
from torbles.math_errors import StaleStateError, UnsupportedError
from torbles.mathematics import AlgebraBudgetError

FLAGS = list(itertools.product((False, True), repeat=3))       # ordered, self_pairs, mean


# ---------------------------------------------------------------------- oracles (explicit pairs)
def kfun(name, **p):
    if name == "abs":
        return lambda x, y: sum(abs(a - b) for a, b in zip(x, y))
    if name == "mann_whitney":
        return lambda x, y: Fraction(1) if x[0] > y[0] else Fraction(1, 2) if x[0] == y[0] else Fraction(0)
    if name == "poly":
        return lambda x, y: (sum(a * b for a, b in zip(x, y)) + p["c"]) ** p["d"]
    if name == "monomial":
        def mono(x, y):
            out = 0
            for exps, w in p["terms"]:
                t = w
                for a, b, e in zip(x, y, exps):
                    t *= (a * b) ** e
                out += t
            return out
        return mono
    if name == "delta":
        return lambda x, y: int(all(a == b for a, b in zip(x, y)))       # NULL (None) matches NULL
    if name == "band":
        return lambda x, y: int(max(abs(a - b) for a, b in zip(x, y)) <= p["h"])
    if name == "sqdist":
        return lambda x, y: sum((a - b) ** 2 for a, b in zip(x, y))
    if name == "table":
        def tup(u):
            return u if isinstance(u, tuple) else (u,)
        tab = {(tup(u), tup(v)): k for (u, v), k in p["tab"].items()}
        dflt, sym = p["default"], p["symmetric"]
        return lambda x, y: tab[(x, y)] if (x, y) in tab else (tab[(y, x)] if sym and (y, x) in tab else dflt)
    raise ValueError(name)


def elements(py, cols, weight, groups, rows, g, category=False):
    """the elements of group g: [(w, x)] (selected, weight present; listwise: every feature present)"""
    out = []
    for r in range(len(groups)):
        if not rows[r] or groups[r] != g:
            continue
        w = 1 if weight is None else py[weight][r]
        if w is None:
            continue
        x = tuple(py[c][r] for c in cols)
        if not category and any(v is None for v in x):
            continue
        out.append((w, x))
    return out


def oracle(k, A, B, within, ordered, self_pairs, mean):
    """(S or the mean, P) over the declared pair set, pair by pair"""
    S, P = Fraction(0), 0
    if within:
        n = len(A)
        for i in range(n):
            for j in range(n):
                if (not ordered and j < i) or (i == j and not self_pairs):
                    continue
                (wi, xi), (wj, xj) = A[i], A[j]
                S += wi * wj * k(xi, xj)
                P += wi * wj
    else:
        for wi, xi in A:
            for wj, xj in B:
                S += wi * wj * k(xi, xj)
                P += wi * wj
                if ordered:
                    S += wi * wj * k(xj, xi)
                    P += wi * wj
    if mean:
        return (None if P == 0 else S / P), P
    return S, P


def every_request(G):
    items = []
    for a, b in itertools.product(range(G), repeat=2):
        for ordered, self_pairs, mean in FLAGS:
            if a != b and self_pairs:
                continue                                    # a no-op between groups (checked apart)
            items.append(dict(a=a, b=b, ordered=ordered, self_pairs=self_pairs, mean=mean))
    return items


def make(rng, n=40, G=4, ncols=1, span=3, weights="pos", nulls=0.15, absent=2):
    names = [f"x{j}" for j in range(ncols)]
    vals = {nm: rng.integers(-span, span + 1, n) for nm in names}
    vals["w"] = rng.integers(0, 4, n) if weights == "pos" else rng.integers(-3, 4, n)
    if weights == "huge":                                             # requests past 126 bits: the 1024-bit path
        vals["w"] = vals["w"] * (1 << 59)
    valid = {nm: rng.random(n) >= nulls for nm in names}
    valid["w"] = rng.random(n) >= nulls / 2
    groups = rng.integers(0, G, n)
    groups[groups == absent] = 0                                      # an absent group
    rows = rng.random(n) < 0.9
    scope = mbk.Scope.from_arrays(vals, valid=valid, groups=groups, ngroups=G, rows=rows)
    py = {nm: [int(vals[nm][r]) if valid[nm][r] else None for r in range(n)] for nm in vals}
    return scope, py, [int(x) for x in groups], [bool(x) for x in rows], names


def value(v):
    return v.numerator if isinstance(v, Fraction) and v.denominator == 1 else v


class Base(unittest.TestCase):
    def run_all(self, p):
        """the raw answers of every admissible candidate by name; a CUDA candidate equals its CPU twin bit for bit and
        every strategy gives the same statuses, num, den, P and emptiness (only the evaluations differ)"""
        outs = {}
        for c in p.candidates:
            if c["status"] != "rejected":
                outs[c["name"]] = p.run(c["index"], raw=True)["results"]
        self.assertTrue(outs)
        for name, o in outs.items():
            if name.endswith("@cuda"):
                self.assertTrue(np.array_equal(o, outs[name.replace("@cuda", "@cpu")]), name)
        ref = next(iter(outs.values()))
        for name, o in outs.items():
            for f in ("status", "empty", "num", "den", "pairs"):
                self.assertTrue(np.array_equal(o[f], ref[f]), (name, f))
        return outs

    def check_oracle(self, p, raw, k, py, cols, weight, groups, rows, category=False, symmetric=True):
        G = p.info["groups"]
        el = [elements(py, cols, weight, groups, rows, g, category) for g in range(G)]
        for q, r in zip(p.requests, raw):
            a, b, f = int(q["a"]), int(q["b"]), int(q["flags"])
            within, ordered, self_pairs, mean = a == b, bool(f & 1), bool(f & 2), bool(f & 4)
            if within and not ordered and not symmetric:
                self.assertEqual(int(r["status"]), -1)                 # an unordered pair has no orientation
                continue
            got = pp._answer(r)
            want, P = oracle(k, el[a], el[b], within, ordered, self_pairs, mean)
            self.assertEqual((got["value"], got["pairs"]), (value(want), P), (a, b, f))
            self.assertEqual(got["empty"], mean and P == 0)


class PairPlan(Base):
    # ------------------------------------------------------------------ every strategy against the enumeration
    def test_pair002_every_strategy_matches_enumerated_pairs_on_tied_weighted_inputs(self):
        rng = np.random.default_rng(4201)
        cases = [  # kernel, its oracle, columns, category, symmetric
            (pp.absdiff(), kfun("abs"), 1, False, True),
            (pp.mann_whitney(), kfun("mann_whitney"), 1, False, False),
            (pp.features(mk.polynomial(2, c=1)), kfun("poly", d=2, c=1), 2, False, True),
            (pp.features(mk.monomial([({"x0": 1, "x1": 1}, 2), ({"x0": 2}, -1)])),
             kfun("monomial", terms=[((1, 1), 2), ((2, 0), -1)]), 2, False, True),
            (pp.features(mk.delta()), kfun("delta"), 2, True, True),
            (pp.table({(1, 2): 5, (0, 0): -2, (-1, 3): 7}, default=1), kfun("table", tab={(1, 2): 5, (0, 0): -2,
                                                                                          (-1, 3): 7}, default=1,
                                                                             symmetric=True), 1, False, True),
            (pp.table([(1, 2, 5), (2, 1, -4), (3, 3, 2)], default=0, symmetric=False),
             kfun("table", tab={(1, 2): 5, (2, 1): -4, (3, 3): 2}, default=0, symmetric=False), 1, False, False),
            (pp.band(1), kfun("band", h=1), 2, False, True),
            (pp.sqdist(), kfun("sqdist"), 2, False, True),
        ]
        for trial, weights in enumerate(("pos", "mixed", None, "huge")):
            for kern, k, ncols, category, symmetric in cases:
                scope, py, groups, rows, names = make(rng, n=int(rng.integers(18, 36)), ncols=ncols,
                                                      weights="pos" if weights is None else weights)
                weight = None if weights is None else "w"
                p = pp.plan(scope, names, kern, every_request(4), weight=weight, devices=("cpu", "cuda"))
                outs = self.run_all(p)
                self.assertIn("support@cpu", outs)
                self.assertIn("direct@cuda", outs)
                for name, raw in outs.items():
                    self.check_oracle(p, raw, k, py, names, weight, groups, rows, category, symmetric)
                strategies = {c["strategy"] for c in p.candidates if c["status"] != "rejected"}
                if kern.kind == "feature":
                    self.assertIn("features", strategies)
                if kern.kind in ("abs", "mann_whitney") and weights in ("pos", None):
                    self.assertIn("prefix", strategies)                  # PAIR-001's sorted prefix applies
                if kern.kind in ("abs", "mann_whitney") and weights in ("mixed", "huge"):
                    c = next(c for c in p.candidates if c["strategy"] == "prefix")
                    self.assertEqual((c["status"], c["code"]), ("rejected", -6))   # weight masses are >= 0
                bits = max(c["bound_bits"] for c in p.candidates if c["strategy"] == "direct")
                self.assertEqual(bits > 126, weights == "huge")            # 1024-bit sums, else 128-bit ones

    # ------------------------------------------------------------------ acceptance 2: multiplicities and the diagonal
    def test_pair002_pair_multiplicities_and_diagonal_treatment_across_support_compression(self):
        # group 0: the value 5 with multiplicity m = 4 (unit weights); group 1: 5 x 3 and 7 x 2 with weights;
        # group 2: one element; group 3: absent
        x = [5, 5, 5, 5, 5, 5, 5, 7, 7, 9]
        w = [1, 1, 1, 1, 2, -1, 3, 0, 4, 6]
        g = [0, 0, 0, 0, 1, 1, 1, 1, 1, 2]
        scope = mbk.Scope.from_arrays({"x": x, "w": w}, groups=g, ngroups=4)
        one = pp.features(mk.polynomial(0))                                # k = 1: S is the weighted pair count
        reqs = [dict(a=a, ordered=o, self_pairs=s) for a in range(4) for o, s in itertools.product((False, True),
                                                                                                 repeat=2)]
        reqs += [dict(a=0, b=1, ordered=o) for o in (False, True)]
        m = 4
        closed = {(False, False): m * (m - 1) // 2, (False, True): m * (m + 1) // 2, (True, False): m * (m - 1),
                  (True, True): m * m}
        for weight in (None, "w"):
            st = pp.retain(scope, ["x"], "support", weight=weight)
            rows_st = pp.retain(scope, ["x"], "rows", weight=weight)
            # the support compression: one entry per distinct value with its multiplicity, W = sum w, Q = sum w^2
            self.assertEqual(st.entries(0), [((5,), 4, 4, 4)])
            if weight:
                self.assertEqual(st.entries(1), [((5,), 3, 2 - 1 + 3, 4 + 1 + 9), ((7,), 2, 4, 16)])
                self.assertEqual(st.totals(1), dict(rows=5, entries=2, W=8, A=10, Q=30))
            else:
                self.assertEqual(st.entries(1), [((5,), 3, 3, 3), ((7,), 2, 2, 2)])
            self.assertEqual(len(rows_st.entries(1)), 5)                   # no compression: one entry per row
            self.assertEqual(st.entries(3), [])
            Wg, Qg = (8, 30) if weight else (5, 5)                          # group 1
            per_group = {(True, True): Wg * Wg, (True, False): Wg * Wg - Qg, (False, True): (Wg * Wg + Qg) // 2,
                         (False, False): (Wg * Wg - Qg) // 2}
            for kern in (one, pp.absdiff(), pp.mann_whitney()):
                items = [q for q in reqs if kern.symmetric or q.get("ordered")]
                p = pp.plan(scope, ["x"], kern, items, weight=weight, devices=("cpu", "cuda"), retained=[st, rows_st])
                outs = self.run_all(p)                                      # compressed (support) = rows (direct)
                self.assertIn("direct@cpu", outs)
                self.assertEqual({c["strategy"]: c["reuse"] for c in p.candidates if c["device"] == "cpu" and
                                  c["strategy"] in ("support", "direct")}, {"support": 0, "direct": 1})
                support = p.run("support@cpu")["results"]
                for q, r in zip(items, support):
                    within = q.get("b") is None
                    key = (q.get("ordered", False), q.get("self_pairs", False))
                    if within and q["a"] == 0:
                        self.assertEqual(r["pairs"], closed[key])           # m^2, m (m - 1), m (m -+ 1) / 2
                    if within and q["a"] == 1:
                        self.assertEqual(r["pairs"], per_group[key])        # W^2, W^2 - Q, (W^2 -+ Q) / 2
                    if within and q["a"] == 2:                              # one element: only its self-pair
                        self.assertEqual(r["pairs"], (36 if weight else 1) if key[1] else 0)
                    if kern is one:
                        self.assertEqual(r["value"], r["pairs"])            # the multiplicities sum to P
                    if within and q["a"] == 0 and kern.kind == "abs":
                        self.assertEqual(r["value"], 0)                     # one value: every |x - y| is 0
                    if within and q["a"] == 0 and kern.kind == "mann_whitney":
                        self.assertEqual(r["value"], Fraction(closed[key], 2))   # every pair a tie, self-pairs too
                k = kfun("poly", d=0, c=0) if kern is one else kfun(kern.kind)
                self.check_oracle(p, outs["support@cpu"], k, {"x": x, "w": w}, ["x"], weight, g, [True] * 10,
                                  symmetric=kern.symmetric)
            # DuckDB self-joins: the pair counts and sums of every pair set, over the rows
            con = duckdb.connect()
            con.register("t", pd.DataFrame({"rid": np.arange(10), "g": g, "x": x, "w": w if weight else [1] * 10}))
            conds = {(False, False): "a.rid < b.rid", (False, True): "a.rid <= b.rid", (True, False): "a.rid <> b.rid",
                     (True, True): "TRUE"}
            p = pp.plan(scope, ["x"], pp.absdiff(), [dict(a=gg, ordered=o, self_pairs=s) for gg in range(3)
                                                     for (o, s) in conds], weight=weight, retained=[st])
            res = p.run()["results"]
            i = 0
            for gg in range(3):
                for (o, s), cond in conds.items():
                    cnt, tot = con.execute(
                        f"SELECT COALESCE(SUM(CAST(a.w AS HUGEINT) * b.w), 0), "
                        f"COALESCE(SUM(CAST(a.w AS HUGEINT) * b.w * ABS(a.x - b.x)), 0) FROM t a JOIN t b "
                        f"ON a.g = {gg} AND b.g = {gg} AND {cond}").fetchone()
                    self.assertEqual((res[i]["pairs"], res[i]["value"]), (int(cnt), int(tot)), (gg, o, s))
                    i += 1
            con.close()

    # ------------------------------------------------------------------ acceptance 1: arbitrary similarities
    def test_pair002_unsupported_arbitrary_similarities_keep_a_costed_direct_plan_or_fail_explicitly(self):
        rng = np.random.default_rng(4203)
        scope, py, groups, rows, names = make(rng, n=60, ncols=2, span=4, weights="mixed")
        tab = {((1, 2), (0, 0)): 9, ((3, -1), (3, -1)): -5}
        arbitrary = [(pp.table(tab, default=2), kfun("table", tab=tab, default=2, symmetric=True)),
                     (pp.band(2), kfun("band", h=2)), (pp.sqdist(), kfun("sqdist"))]
        reqs = [dict(a=0, b=1), dict(a=0, ordered=True), dict(a=1, mean=True), dict(a=3, b=0, ordered=True)]
        rates = pp.default_rates()
        for kern, k in arbitrary:
            p = pp.plan(scope, names, kern, reqs, weight="w", devices=("cpu", "cuda"))
            cands = {c["name"]: c for c in p.candidates}
            for name in ("features@cpu:dense", "features@cpu:sparse", "prefix@cpu"):
                self.assertEqual(cands[name]["status"], "rejected")
                self.assertEqual(cands[name]["refusal"]["kind"], "unsupported")   # no structure for this kernel
            direct = cands["direct@cpu"]
            self.assertNotEqual(direct["status"], "rejected")
            # the direct plan is costed exactly: its evaluations are the pairs of rows it visits
            el = [len(elements(py, names, "w", groups, rows, gg)) for gg in range(4)]
            visits = el[0] * el[1] + el[0] ** 2 + el[1] * (el[1] + 1) // 2 + el[3] * el[0]
            self.assertEqual((direct["evaluations"], direct["need"]["iterations"]), (visits, visits))
            self.assertTrue(direct["exact"])
            for comp, amount in direct["amounts"].items():
                self.assertEqual(direct["costs"][comp], amount * rates["cpu"][comp])
            self.assertEqual(direct["total"], sum(direct["costs"].values()))
            out = p.run("direct@cpu")
            self.assertEqual(out["evaluations"], visits)
            self.check_oracle(p, p.run("direct@cpu", raw=True)["results"], k, py, names, "w", groups, rows)
            self.assertTrue(np.array_equal(p.run("direct@cuda", raw=True)["results"],
                                           p.run("direct@cpu", raw=True)["results"]))
            # the declared pair budget (iterations): direct at the limit runs, one past it is refused
            forced = pp.plan(scope, names, kern, reqs, weight="w", force="direct", budget=Budget(iterations=visits))
            self.assertEqual(forced.candidate(forced.chosen)["name"], "direct@cpu")
            with self.assertRaises(AlgebraBudgetError) as ctx:
                pp.plan(scope, names, kern, reqs, weight="w", force="direct", budget=Budget(iterations=visits - 1))
            self.assertEqual((ctx.exception.resource, ctx.exception.need, ctx.exception.limit),
                             ("iterations", visits, visits - 1))
            # over every pair budget (support included): an explicit refusal with the evidence, never an approximation
            with self.assertRaises(AlgebraBudgetError) as ctx:
                pp.plan(scope, names, kern, reqs, weight="w", budget=Budget(iterations=10))
            ev = ctx.exception.plan
            self.assertIsNone(ev.chosen)
            self.assertEqual(ev.reason, "no admissible candidate")
            self.assertTrue(all(c["status"] == "rejected" for c in ev.candidates))
            self.assertEqual({c["reason"] for c in ev.candidates if c["strategy"] in ("support", "direct")},
                             {"over budget"})
        # similarities without an exact value are refused natively, whatever the budget
        for name in ("rbf", "gaussian", "laplace", "laplacian", "cosine"):
            with self.assertRaises(UnsupportedError):
                pp.plan(scope, names, pp.kernel(name, gamma=1), reqs)
            with self.assertRaises(UnsupportedError):
                pp.retain(scope, names, "support", kernel=pp.kernel(name))
        with self.assertRaises(UnsupportedError):
            pp.plan(scope, names, pp.features(mk.kernel("rbf", gamma=1)), reqs)

    # ------------------------------------------------------------------ FND-008 costs, the choice, forces, reuse
    def test_pair002_candidates_are_costed_through_the_planner_and_the_cheapest_runs(self):
        rng = np.random.default_rng(4204)
        n = 600
        x = rng.integers(0, 8, n)
        w = rng.integers(0, 5, n)
        g = rng.integers(0, 3, n)
        scope = mbk.Scope.from_arrays({"x": x, "w": w}, groups=g, ngroups=3)
        reqs = [dict(a=0, b=1), dict(a=2, ordered=True), dict(a=1, mean=True)]
        rates = pp.default_rates()
        p = pp.plan(scope, ["x"], pp.absdiff(), reqs, weight="w", devices=("cpu", "cuda"))
        cands = p.candidates
        for c in cands:
            dev = rates[c["device"]]
            for comp, amount in c["amounts"].items():
                self.assertEqual(c["costs"][comp], 0.0 if comp == "fusion" else amount * dev[comp], (c["name"], comp))
        admissible = [c for c in cands if c["status"] != "rejected"]
        best = min(admissible, key=lambda c: (c["total"], c["index"]))
        self.assertEqual(p.chosen, best["index"])
        self.assertEqual(p.candidate(p.chosen)["reason"], "cheapest")
        self.assertEqual(p.candidate(p.chosen)["name"], "prefix@cpu")    # 8 support values: the sorted prefix wins
        # a force runs exactly that strategy (and device / representation), or refuses; it is never replaced
        for force, name in (("support", "support@cpu"), ("direct@cuda", "direct@cuda"), ("prefix", "prefix@cpu")):
            f = pp.plan(scope, ["x"], pp.absdiff(), reqs, weight="w", force=force)
            self.assertEqual((f.candidate(f.chosen)["name"], f.reason), (name, "forced"))
            self.assertTrue(all(c["reason"] in ("not forced", "refused") for c in f.candidates if c["index"] != f.chosen))
        with self.assertRaises(UnsupportedError) as ctx:
            pp.plan(scope, ["x"], pp.absdiff(), reqs, weight="w", force="features")       # no finite feature map
        self.assertEqual(ctx.exception.plan.reason, "forced candidate refused")
        with self.assertRaises(UnsupportedError) as ctx:
            pp.plan(scope, ["x"], pp.absdiff(), reqs, weight="w", strategies=("support",), force="features")
        self.assertEqual(ctx.exception.plan.reason, "forced candidate refused")
        for bad in ("prefix@tpu", "support:packed", "nearest"):
            with self.assertRaises(ValueError):
                pp.plan(scope, ["x"], pp.absdiff(), reqs, force=bad)
        # the rates decide: expensive kernel evaluations favour the linear prefix pass, expensive builds direct pairs
        small = mbk.Scope.from_arrays({"x": [i % 3 for i in range(40)], "w": [1 + i % 3 for i in range(40)]},
                                      groups=[i % 2 for i in range(40)], ngroups=2)
        for r, want in (({"cpu": {"chain": 1000.0}}, "prefix@cpu"), ({"cpu": {"construct": 1e6, "fixed": 0.0}},
                                                                     "direct@cpu")):
            q = pp.plan(small, ["x"], pp.absdiff(), [dict(a=0, b=1)], weight="w", rates=r)
            self.assertEqual(q.candidate(q.chosen)["name"], want, r)
        # retained states: reuse instead of construction, and exact sizes
        st = pp.retain(scope, ["x"], "support", weight="w")
        pre = pp.retain(scope, ["x"], "prefix", weight="w")
        r = pp.plan(scope, ["x"], pp.absdiff(), reqs, weight="w", retained=[st, pre])
        self.assertTrue(r.info["profile_reused"])
        for c in r.candidates:
            if c["strategy"] in ("support", "prefix"):
                self.assertEqual((c["amounts"]["construct"], c["amounts"]["reuse"], c["exact"]), (0, 1, True))
                self.assertEqual(c["reuse"], 0 if c["strategy"] == "support" else 1)
                out = r.run(c["index"])
                self.assertFalse(out["built"])
                self.assertEqual(out["evaluations"], c["evaluations"])        # planned = actual
        self.assertEqual(r.info["profile_reused"], True)
        # a state of another scope or other columns is not reused (the plan builds its own)
        other = mbk.Scope.from_arrays({"x": x, "w": w}, groups=g, ngroups=3)
        q = pp.plan(other, ["x"], pp.absdiff(), reqs, weight="w", retained=[st])
        self.assertTrue(all(c["reuse"] is None for c in q.candidates))
        q = pp.plan(scope, ["x"], pp.absdiff(), reqs, retained=[st])          # no weight: another source
        self.assertTrue(all(c["reuse"] is None for c in q.candidates))
        self.assertEqual([a["value"] for a in q.run()["results"]],
                         [a["value"] for a in pp.query(scope, ["x"], pp.absdiff(), reqs, force="direct")["results"]])

    # ------------------------------------------------------------------ PAIR-001, KERNEL-001 and DuckDB agree
    def test_pair002_strategies_agree_with_pair001_kernel001_and_duckdb(self):
        rng = np.random.default_rng(4205)
        n = 500
        x = rng.integers(-20, 21, n)
        y = rng.integers(-5, 6, n)
        w = rng.integers(0, 6, n)
        g = rng.integers(0, 3, n)
        vx = rng.random(n) > 0.08
        scope = mbk.Scope.from_arrays({"x": x, "y": y, "w": w}, valid={"x": vx}, groups=g, ngroups=3)
        # PREFIX = PAIR-001 (abs_within, abs_cross, mann_whitney over the same prefix state)
        prefix = md.histogram(scope, ["x"], weight="w").prefix("weight", degree=0)
        reqs = [dict(a=0), dict(a=1, ordered=True), dict(a=0, b=2), dict(a=2, b=1, ordered=True)]
        out = pp.query(scope, ["x"], pp.absdiff(), reqs, weight="w", force="prefix")["results"]
        self.assertEqual(out[0]["value"], mpr.abs_within(prefix, 0)["value"])
        self.assertEqual(out[1]["value"], mpr.abs_within(prefix, 1, ordered=True)["value"])
        self.assertEqual(out[2]["value"], mpr.abs_cross(prefix, 0, 2)["value"])
        self.assertEqual(out[3]["value"], 2 * mpr.abs_cross(prefix, 2, 1)["value"])
        mw = pp.query(scope, ["x"], pp.mann_whitney(), [dict(a=0, b=1), dict(a=1, b=0), dict(a=0, b=1, ordered=True),
                                                        dict(a=2, ordered=True, self_pairs=True)],
                      weight="w", force="prefix")["results"]
        self.assertEqual(mw[0]["value"], mpr.mann_whitney(prefix, 0, 1)["value"])
        self.assertEqual(mw[1]["value"], mpr.mann_whitney(prefix, 1, 0)["value"])
        self.assertEqual(mw[2]["value"], mw[0]["pairs"])                    # U_ab + U_ba = W_a W_b
        self.assertEqual(mw[3]["value"], mpr.mann_whitney(prefix, 2, 2, self_pairs=True)["value"])
        # FEATURES = KERNEL-001 (the feature state's sums with the same pair conventions)
        for kern in (mk.polynomial(2, c=3), mk.delta()):
            cols = ["x", "y"] if kern.kind != "delta" else ["y"]
            st = mk.features(scope, cols, kern, weight="w")
            res = pp.query(scope, cols, pp.features(kern), [dict(a=0, b=1), dict(a=1, ordered=True),
                                                             dict(a=2, self_pairs=True), dict(a=0)],
                           weight="w", force="features")["results"]
            self.assertEqual((res[0]["value"], res[0]["pairs"]), (st.sum(0, 1)["value"], st.sum(0, 1)["pairs"]))
            self.assertEqual(res[1]["value"], st.sum(1)["value"])
            self.assertEqual(res[2]["value"], st.sum(2, self_pairs=True, ordered=False)["value"])
            self.assertEqual(res[3]["value"], st.sum(0, ordered=False)["value"])
            kept = pp.retain(scope, cols, "features", kernel=pp.features(kern), weight="w", repr="sparse")
            again = pp.plan(scope, cols, pp.features(kern), [dict(a=0, b=1), dict(a=1, ordered=True)], weight="w",
                            retained=[kept], force="features:sparse")
            c = again.candidate(again.chosen)
            self.assertEqual((c["reuse"], c["amounts"]["construct"], c["amounts"]["reuse"]), (0, 0, 1))
            out = again.run()
            self.assertFalse(out["built"])
            self.assertEqual([r["value"] for r in out["results"]], [res[0]["value"], res[1]["value"]])
        # DuckDB self-joins over the rows: ABS within / across groups and the weighted pair counts
        con = duckdb.connect()
        con.register("t", pd.DataFrame({"rid": np.arange(n), "g": g, "w": w,
                                        "x": pd.Series(np.where(vx, x, 0), dtype="Int64").where(vx)}))
        sql = ("SELECT a.g, b.g, SUM(CAST(a.w AS HUGEINT) * b.w * ABS(a.x - b.x)), SUM(CAST(a.w AS HUGEINT) * b.w) "
               "FROM t a JOIN t b ON a.rid < b.rid OR a.g <> b.g WHERE a.x IS NOT NULL AND b.x IS NOT NULL "
               "GROUP BY a.g, b.g")
        duck = {(int(r[0]), int(r[1])): (int(r[2]), int(r[3])) for r in con.execute(sql).fetchall()}
        con.close()
        items = [dict(a=a, b=b) for a in range(3) for b in range(3)]
        for force in ("support", "prefix", "direct"):
            res = pp.query(scope, ["x"], pp.absdiff(), items, weight="w", force=force, budget=Budget(iterations=10**6))
            for q, r in zip(items, res["results"]):
                self.assertEqual((r["value"], r["pairs"]), duck[(q["a"], q["b"])], (force, q))

    # ------------------------------------------------------------------ CUDA equals the CPU
    def test_pair002_cuda_support_and_direct_equal_cpu_bit_for_bit_scrambled(self):
        rng = np.random.default_rng(4206)
        n, G = 900, 5
        x = rng.integers(-60, 60, (n, 2))
        w = rng.integers(-4, 9, n)
        g = rng.integers(0, G, n)
        valid = {"a": rng.random(n) > 0.05, "b": rng.random(n) > 0.05}
        scope = mbk.Scope.from_arrays({"a": x[:, 0], "b": x[:, 1], "w": w}, valid=valid, groups=g, ngroups=G)
        items = [dict(a=int(rng.integers(0, G)), b=int(rng.integers(0, G)), ordered=bool(rng.integers(0, 2)),
                      self_pairs=bool(rng.integers(0, 2)), mean=bool(rng.integers(0, 2))) for _ in range(40)]
        items += [dict(a=1, ordered=False), dict(a=1, ordered=True, self_pairs=True)]   # many tiles, the triangle
        for kern, cols in ((pp.absdiff(), ["a", "b"]), (pp.features(mk.polynomial(3, c=-2)), ["a", "b"]),
                           (pp.mann_whitney(), ["a"]), (pp.features(mk.delta()), ["a", "b"])):
            its = [q for q in items if kern.symmetric or q["ordered"] or q["a"] != q.get("b", q["a"])]
            for strategy in ("support", "rows"):
                st = pp.retain(scope, cols, strategy, kernel=kern, weight="w")
                p = pp.plan(scope, cols, kern, its, weight="w", retained=[st], budget=Budget(iterations=10**7))
                cpu = pp.eval_table(st, p, budget=Budget(iterations=10**7))
                gpu = pp.eval_table(st, p, backend="cuda", budget=Budget(iterations=10**7))
                self.assertTrue(np.array_equal(cpu, gpu), (kern.kind, strategy))
                self.assertTrue((cpu["status"] == 0).all())
                self.assertGreater(st.info["largest"], pp.TILE)             # several tiles per request
                again = pp.eval_table(st, p, backend="cuda", budget=Budget(iterations=10**7))   # the upload reused
                self.assertTrue(np.array_equal(cpu, again))
            p = pp.plan(scope, cols, kern, its, weight="w", devices=("cuda",), budget=Budget(iterations=10**7))
            self.assertTrue(all(c["device"] == "cuda" for c in p.candidates))
            out = p.run()
            self.assertEqual(out["device"], "cuda")
            self.assertIn(out["backend"], ("cuda", "cuda-emulated"))

    # ------------------------------------------------------------------ budgets at the limit and one past
    def test_pair002_budgets_at_the_limit_and_one_past(self):
        rng = np.random.default_rng(4207)
        scope, py, groups, rows, names = make(rng, n=80, ncols=1, span=50)
        reqs = [dict(a=0, b=1), dict(a=1, ordered=True)]
        st = pp.retain(scope, names, "support", weight="w")
        p = pp.plan(scope, names, pp.absdiff(), reqs, weight="w", retained=[st])
        c = p.candidate(p.index("support@cpu"))
        visits = c["evaluations"]
        for backend in ("cpu", "cuda"):
            self.assertEqual(len(pp.eval_table(st, p, backend=backend, budget=Budget(iterations=visits))), 2)
            with self.assertRaises(AlgebraBudgetError) as ctx:
                pp.eval_table(st, p, backend=backend, budget=Budget(iterations=visits - 1))
            self.assertEqual((ctx.exception.resource, ctx.exception.need), ("iterations", visits))
            self.assertEqual(len(pp.eval_table(st, p, backend=backend, budget=Budget(output=2))), 2)
            with self.assertRaises(AlgebraBudgetError) as ctx:
                pp.eval_table(st, p, backend=backend, budget=Budget(output=1))
            self.assertEqual((ctx.exception.resource, ctx.exception.need), ("output", 2))
        # the batch's device copies and the upload's
        batch = 2 * (24 + 4 + 8 + 128 + 16 + 400) + 8 + 1096
        self.assertEqual(len(pp.eval_table(st, p, backend="cuda", budget=Budget(bytes=batch))), 2)
        with self.assertRaises(AlgebraBudgetError) as ctx:
            pp.eval_table(st, p, backend="cuda", budget=Budget(bytes=batch - 1))
        self.assertEqual((ctx.exception.resource, ctx.exception.need), ("bytes", batch))
        fresh = pp.retain(scope, names, "support", weight="w")
        info = fresh.info
        up = (info["groups"] + 1) * 8 + info["groups"] * 80 + info["entries"] * (8 + 16 + 40)
        up += (info["groups"] + 1) * 8 + info["groups"] * 80                  # host copies of offsets and totals
        with self.assertRaises(AlgebraBudgetError) as ctx:
            fresh.device(Budget(bytes=up - 1))
        self.assertEqual((ctx.exception.resource, ctx.exception.need), ("bytes", up))
        self.assertEqual(fresh.device(Budget(bytes=up)), up)
        # the plan's output (its requests) and a support build's states (the elements it sorts)
        with self.assertRaises(AlgebraBudgetError) as ctx:
            pp.plan(scope, names, pp.absdiff(), reqs, budget=Budget(output=1))
        self.assertEqual((ctx.exception.resource, ctx.exception.need), ("output", 2))
        N = st.info["elements"]
        self.assertEqual(pp.retain(scope, names, "support", weight="w", budget=Budget(states=N)).info["elements"], N)
        with self.assertRaises(AlgebraBudgetError) as ctx:
            pp.retain(scope, names, "support", weight="w", budget=Budget(states=N - 1))
        self.assertEqual((ctx.exception.resource, ctx.exception.need), ("states", N))
        # a plan's candidate over the byte budget is rejected with its need; the build agrees at that need
        scope, py, groups, rows, names = make(rng, n=600, ncols=1, span=500)
        q = pp.plan(scope, names, pp.absdiff(), reqs, weight="w", strategies=("support",))
        need = q.candidate(0)["need"]["bytes"]
        ok = pp.plan(scope, names, pp.absdiff(), reqs, weight="w", strategies=("support",), budget=Budget(bytes=need))
        self.assertEqual(len(ok.run()["results"]), 2)
        with self.assertRaises(AlgebraBudgetError) as ctx:
            pp.plan(scope, names, pp.absdiff(), reqs, weight="w", strategies=("support",), budget=Budget(bytes=need - 1))
        self.assertEqual((ctx.exception.resource, ctx.exception.need), ("bytes", need))

    # ------------------------------------------------------------------ exactness at the extremes
    def test_pair002_extreme_values_are_exact_and_unproven_magnitudes_are_refused(self):
        big = (1 << 62) - 1
        x = [-big, big, 3, -big, big, 0]
        g = [0, 0, 0, 1, 1, 1]
        w = [1 << 40, -(1 << 41), 3, 5, 1 << 50, 7]
        scope = mbk.Scope.from_arrays({"x": x, "w": w}, groups=g, ngroups=2)
        reqs = [dict(a=0, b=1, ordered=True), dict(a=0, ordered=True), dict(a=1, self_pairs=True, mean=True)]
        el = [elements({"x": x, "w": w}, ["x"], "w", g, [True] * 6, gg) for gg in range(2)]
        for kern, k in ((pp.absdiff(), kfun("abs")), (pp.sqdist(), kfun("sqdist")),
                        (pp.features(mk.polynomial(2, c=7)), kfun("poly", d=2, c=7))):
            p = pp.plan(scope, ["x"], kern, reqs, weight="w", devices=("cpu", "cuda"))
            outs = self.run_all(p)
            self.assertIn("support@cuda", outs)
            for q, r in zip(reqs, outs["support@cpu"]):
                within = q.get("b") is None
                want, P = oracle(k, el[q["a"]], el[q.get("b", q["a"])], within, q.get("ordered", False),
                                 q.get("self_pairs", False), q.get("mean", False))
                self.assertEqual(pp._answer(r)["value"], value(want))
        # a magnitude that cannot be proven within 1024 bits: support and direct refuse (bound), and the KERNEL-001
        # strategy refuses its own proof, so the plan is refused (OverflowError), never wrapped
        p8 = pp.features(mk.polynomial(8, c=1))
        with self.assertRaises(OverflowError) as ctx:
            pp.plan(scope, ["x"], p8, reqs, weight="w")
        for c in ctx.exception.plan.candidates:
            if c["strategy"] in ("support", "direct"):
                self.assertEqual((c["refusal"]["kind"], c["code"]), ("bound", -3))
                self.assertGreater(c["bound_bits"], pp.BITS_LIMIT)
        # PAIR-001 needs every class mass below 2^64: weight masses past it leave the prefix strategy (bound) while
        # support aggregation stays exact
        W = (1 << 63) - 1
        heavy = mbk.Scope.from_arrays({"x": [1, 2, 3, 4, 5, 6], "w": [W] * 6}, groups=[0, 0, 0, 1, 1, 1], ngroups=2)
        p = pp.plan(heavy, ["x"], pp.absdiff(), [dict(a=0, b=1)], weight="w")   # 3 W >= 2^64
        pre = next(c for c in p.candidates if c["strategy"] == "prefix")
        self.assertEqual((pre["status"], pre["refusal"]["kind"]), ("rejected", "bound"))
        self.assertEqual(p.run()["results"][0]["value"], W * W * (3 + 4 + 5 + 2 + 3 + 4 + 1 + 2 + 3))
        light = mbk.Scope.from_arrays({"x": [1, 2, 4, 5], "w": [W] * 4}, groups=[0, 0, 1, 1], ngroups=2)
        r = pp.query(light, ["x"], pp.absdiff(), [dict(a=0, b=1)], weight="w", force="prefix")   # 2 W < 2^64
        self.assertEqual(r["results"][0]["value"], W * W * (3 + 4 + 2 + 3))

    # ------------------------------------------------------------------ declarations, invalid input, opt-in
    def test_pair002_pair_set_declarations_invalid_requests_and_opt_in_import(self):
        rng = np.random.default_rng(4208)
        scope, py, groups, rows, names = make(rng, n=30, ncols=1)
        # self-pairs between two groups add nothing (no element is in both groups)
        a = pp.query(scope, names, pp.absdiff(), [dict(a=0, b=1), dict(a=0, b=1, self_pairs=True)], weight="w")
        self.assertEqual(a["results"][0], a["results"][1])
        # an unordered pair set with an asymmetric kernel is refused per request (status -1), the batch runs
        p = pp.plan(scope, names, pp.mann_whitney(), [dict(a=0), dict(a=0, ordered=True), dict(a=9),
                                                      dict(a=0, b=1)], weight="w", devices=("cpu", "cuda"))
        outs = self.run_all(p)
        self.assertEqual(list(outs["direct@cpu"]["status"]), [-1, 0, -1, 0])
        with self.assertRaises(ValueError):
            p.run()
        bad = pp.requests([dict(a=0, b=1)])
        bad["flags"][0] = 8
        q = pp.plan(scope, names, pp.absdiff(), bad)
        self.assertEqual(int(q.run(raw=True)["results"]["status"][0]), -1)
        bad["flags"][0], bad["reserved"][0] = 0, 3
        q = pp.plan(scope, names, pp.absdiff(), bad)
        self.assertEqual(int(q.run(raw=True)["results"]["status"][0]), -1)
        # malformed specs
        two = mbk.Scope.from_arrays({"x": [1, 2], "d": [3, 4]}, scales={"d": 2})
        with self.assertRaises(DomainError):
            pp.plan(two, ["x", "d"], pp.absdiff(), [dict(a=0)])           # a distance mixes decimal scales
        self.assertEqual(pp.query(two, ["x", "d"], pp.table({}, default=1), [dict(a=0, ordered=True)])["results"][0]
                         ["value"], 2)                                      # a table compares no scales
        with self.assertRaises(ValueError):
            pp.plan(two, ["x", "d"], pp.mann_whitney(), [dict(a=0, ordered=True)])   # one column only
        with self.assertRaises(ValueError):
            pp.plan(two, ["x"], pp.table([(1, 2, 3), (2, 1, 4)], symmetric=True), [dict(a=0)])   # declared twice
        with self.assertRaises(ValueError):
            pp.plan(two, ["x"], pp.band(-1), [dict(a=0)])
        with self.assertRaises(ValueError):
            pp.plan(two, ["x", "x"], pp.absdiff(), [dict(a=0)])
        with self.assertRaises(UnsupportedError):
            pp.plan(two, ["x"] * 9, pp.absdiff(), [dict(a=0)])
        with self.assertRaises(ValueError):
            pp.retain(two, ["x", "d"], "prefix")                            # PAIR-001 reads one column
        with self.assertRaises(ValueError):
            pp.retain(two, ["x"], "features")                               # not a feature kernel
        with self.assertRaises(ValueError):
            pp.retain(two, ["x"], "bitplane")
        empty = pp.query(two, ["x"], pp.absdiff(), [])
        self.assertEqual(empty["results"], [])
        code = "import sys, torbles; print('torbles.math_pairplan' in sys.modules)"
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.split()
        self.assertEqual(out, ["False"])

    # ------------------------------------------------------------------ an engine table and its freshness
    def test_pair002_engine_table_scope_duckdb_and_staleness(self):
        rng = np.random.default_rng(4209)
        n = 150
        frame = pd.DataFrame({
            "k": pd.Series([int(v) for v in rng.integers(0, 3, n)], dtype=object),
            "x": pd.Series([f"{int(v) / 100:.2f}" if rng.random() > 0.1 else None for v in rng.integers(-500, 500, n)],
                           dtype=object),
            "w": pd.Series([int(v) for v in rng.integers(1, 5, n)], dtype=object)})
        db = torbles.from_ddl("CREATE TABLE f (k BIGINT, x DECIMAL(10,2), w BIGINT)", {"f": frame})
        t = db["f"]
        scope = mbk.Scope.from_table(t, ["x", "w"], group_by=["k"])
        keys = {scope.key_values(g)[0]: g for g in range(scope.groups)}
        items = [dict(a=keys[0], b=keys[1]), dict(a=keys[2], ordered=True)]
        res = pp.query(scope, ["x"], pp.absdiff(), items, weight="w")["results"]
        sql = ("SELECT SUM(CAST(a.w AS HUGEINT) * b.w * ABS(CAST(a.x * 100 AS BIGINT) - CAST(b.x * 100 AS BIGINT))), "
               "SUM(CAST(a.w AS HUGEINT) * b.w) FROM f a, f b WHERE a.x IS NOT NULL AND b.x IS NOT NULL AND "
               "a.k = {} AND b.k = {}")
        self.assertEqual((res[0]["value"], res[0]["pairs"]), tuple(int(v) for v in db.con.execute(
            sql.format(0, 1)).fetchone()))
        same = db.con.execute(sql.format(2, 2)).fetchone()                 # ordered within: the self-pairs add 0
        self.assertEqual(res[1]["value"], int(same[0]))
        st = pp.retain(scope, ["x"], "support", weight="w")
        t.flip(0, 0)
        with self.assertRaises(StaleStateError):
            pp.plan(scope, ["x"], pp.absdiff(), items, weight="w", retained=[st])


if __name__ == "__main__":
    unittest.main()
