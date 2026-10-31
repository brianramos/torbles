# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""PAIR-001 (sorted-distribution pairwise summaries) (torbles/math_pair.c, math_pair_gpu.cu, math_pair.py).

Oracles, independent of the implementation: explicit enumeration of the pairs of the expanded multisets in Python
integers / Fractions (each value repeated once per row, or once per unit of its integer weight), written here from the
definitions (KS from the empirical CDFs at every pooled point, the 1-Wasserstein distance from the quantile functions
integrated over [0, 1], Cramer-von Mises from the pooled elements), and DuckDB SQL self-joins over the rows (pair sums
in HUGEINT). Every native result is also computed on the CUDA path (emulated here, TORBLES_EMU_SCRAMBLE=1 scrambles
the thread order) and compared for equality with the CPU result. Seeds are fixed.
"""
import itertools
import subprocess
import sys
import unittest
from fractions import Fraction

import duckdb
import numpy as np
import pandas as pd

from torbles import math_bank as mbk, math_dist as md, math_pair as mp
from torbles.math_budgets import Budget
from torbles.math_domains import DomainError
from torbles.mathematics import AlgebraBudgetError

OPS = ("abs_within", "abs_cross", "mann_whitney", "ks", "wasserstein", "cramer_von_mises")


# ---------------------------------------------------------------------- oracles (explicit pairs)
def expand(values, weights=None, rows=None, groups=None, g=0):
    out = []
    for i, x in enumerate(values):
        if (rows is not None and not rows[i]) or (groups is not None and groups[i] != g) or x is None:
            continue
        rep = 1 if weights is None else weights[i]
        if rep is None or rep <= 0:
            continue
        out += [x] * rep
    return sorted(out)


def value(v):
    return v.numerator if isinstance(v, Fraction) and v.denominator == 1 else v


def o_abs_within(ys, ordered, self_pairs, mean):
    pairs = [(i, j) for i in range(len(ys)) for j in range(len(ys))
             if (ordered or i <= j) and (self_pairs or i != j)]
    s = sum(abs(ys[i] - ys[j]) for i, j in pairs)
    if mean:
        return (None if not pairs else value(Fraction(s, len(pairs)))), len(pairs)
    return s, len(pairs)


def o_cross(xs, ys, same, self_pairs, kind, mean):
    pairs = [(i, j) for i in range(len(xs)) for j in range(len(ys)) if not same or self_pairs or i != j]
    if kind == "abs_cross":
        s = Fraction(sum(abs(xs[i] - ys[j]) for i, j in pairs))
    else:
        s = Fraction(sum(2 if xs[i] > ys[j] else 1 if xs[i] == ys[j] else 0 for i, j in pairs), 2)
    if mean:
        return (None if not pairs else value(s / len(pairs))), len(pairs)
    return value(s), len(pairs)


def cdf(xs, t):
    return Fraction(sum(1 for x in xs if x <= t), len(xs))


def o_ks(xs, ys):
    if not xs or not ys:
        return None, None
    best, at = Fraction(0), None
    for t in sorted(set(xs) | set(ys)):
        d = abs(cdf(xs, t) - cdf(ys, t))
        if d > best:
            best, at = d, t
    return value(best), at


def o_wasserstein(xs, ys):
    """integral over u in [0, 1] of |F_A^-1(u) - F_B^-1(u)| (the quantile functions; independent of the CDF form)"""
    if not xs or not ys:
        return None
    cuts = sorted({Fraction(i, len(xs)) for i in range(len(xs) + 1)} | {Fraction(j, len(ys)) for j in range(len(ys) + 1)})
    total = Fraction(0)
    for u0, u1 in zip(cuts, cuts[1:]):
        mid = (u0 + u1) / 2
        qa, qb = xs[int(mid * len(xs))], ys[int(mid * len(ys))]
        total += abs(qa - qb) * (u1 - u0)
    return value(total)


def o_cvm(xs, ys):
    if not xs or not ys:
        return None
    na, nb = len(xs), len(ys)
    s = sum((cdf(xs, u) - cdf(ys, u)) ** 2 for u in xs + ys)
    return value(Fraction(na * nb, (na + nb) ** 2) * s)


def oracle(item, A, B):
    """(value, pairs) of one request dict over the expanded classes A (and B)"""
    op = item["op"]
    if op == "abs_within":
        return o_abs_within(A, item.get("ordered", False), item.get("self_pairs", False), item.get("mean", False))
    same = item.get("same", False)
    if op in ("abs_cross", "mann_whitney"):
        return o_cross(A, B, same, item.get("self_pairs", False), op, item.get("mean", False) or item.get("auc", False))
    if op == "ks":
        return o_ks(A, B)[0], len(A) * len(B)
    if op == "wasserstein":
        return o_wasserstein(A, B), len(A) * len(B)
    return o_cvm(A, B), len(A) * len(B)


def every_request(a, b):
    """every op with every flag combination it accepts, between groups a and b"""
    items = []
    for ordered, self_pairs, mean in itertools.product((False, True), repeat=3):
        items.append(dict(op="abs_within", a=a, ordered=ordered, self_pairs=self_pairs, mean=mean))
    for op in ("abs_cross", "mann_whitney"):
        for self_pairs, mean in itertools.product((False, True), repeat=2):
            items.append(dict(op=op, a=a, b=b, self_pairs=self_pairs, mean=mean))
    for op in ("ks", "wasserstein", "cramer_von_mises"):
        items.append(dict(op=op, a=a, b=b))
    return items


def tiny(rng, n, G, span, weighted, nulls=0.2):
    x = [int(v) for v in rng.integers(-span, span + 1, n)]
    w = [int(v) for v in rng.integers(0, 4, n)]
    vx = rng.random(n) >= nulls
    groups = rng.integers(0, G, n)
    groups[groups == G - 1] = 0                                          # the last group is absent
    rows = rng.random(n) < 0.9
    scope = mbk.Scope.from_arrays({"x": x, "w": w}, valid={"x": vx}, groups=groups, ngroups=G, rows=rows)
    h = md.histogram(scope, ["x"], weight="w" if weighted else None)
    p = h.prefix("weight" if weighted else "count", degree=0)
    xs = [x[i] if vx[i] else None for i in range(n)]
    classes = {g: expand(xs, w if weighted else None, rows, groups, g) for g in range(G)}
    return p, classes


class Pairs(unittest.TestCase):
    def same_on_cuda(self, p, items, other=None):
        reqs = mp.requests(items)
        cpu = mp.query_arrays(p, reqs, other=other)
        gpu = mp.query_arrays(p, reqs, other=other, backend="cuda")
        self.assertTrue(np.array_equal(cpu, gpu))
        return cpu

    def check(self, p, classes, items, other=None, other_classes=None):
        res = mp.query(p, items, other=other)
        self.same_on_cuda(p, items, other)
        for q, r in zip(items, res):
            A = classes[q["a"]]
            B = (other_classes or classes)[q.get("b", 0)]
            q = dict(q, same=other is None and q["op"] not in ("abs_within",) and q["a"] == q.get("b"))
            want, pairs = oracle(q, A, B)
            self.assertEqual(r["value"], want, q)
            self.assertEqual(r["empty"], want is None, q)
            self.assertEqual(r["pairs"], pairs, q)
            if q["op"] == "ks" and want is not None:
                self.assertEqual(r["at"], o_ks(A, B)[1], q)

    # ------------------------------------------------------------------ acceptance 1: enumerated pairs
    def test_pair001_matches_enumerated_pairs_for_tiny_tied_weighted_inputs(self):
        rng = np.random.default_rng(2001)
        for trial in range(12):
            weighted = trial % 2 == 1
            p, classes = tiny(rng, int(rng.integers(5, 30)), 4, int(rng.integers(1, 6)), weighted)
            items = []
            for a, b in itertools.product(range(4), repeat=2):
                items += every_request(a, b)
            self.check(p, classes, items)

    def test_pair001_two_states_scales_and_duckdb_self_joins(self):
        rng = np.random.default_rng(2002)
        n = 900
        x = rng.integers(-40, 41, n)
        y = rng.integers(-30, 60, n)
        g = rng.integers(0, 3, n)
        vy = rng.random(n) > 0.1
        scope = mbk.Scope.from_arrays({"x": x, "y": y, "d": y}, valid={"y": vy, "d": vy}, scales={"d": 2}, groups=g,
                                      ngroups=3)
        px = md.histogram(scope, ["x"]).prefix(degree=0)
        py = md.histogram(scope, ["y"]).prefix(degree=1)
        cx = {k: expand(list(x), None, None, g, k) for k in range(3)}
        cy = {k: expand([int(v) if ok else None for v, ok in zip(y, vy)], None, None, g, k) for k in range(3)}
        items = [dict(op=op, a=a, b=b, mean=mean) for op in ("abs_cross", "mann_whitney") for a in range(3)
                 for b in range(3) for mean in (False, True)]
        items += [dict(op=op, a=a, b=b) for op in ("ks", "wasserstein", "cramer_von_mises") for a in range(3)
                  for b in range(3)]
        sub = items[::3]                                          # the enumeration oracle is quadratic: a sample
        res = mp.query(px, sub, other=py)
        self.same_on_cuda(px, sub, other=py)
        for q, r in zip(sub, res):
            want, pairs = oracle(dict(q, same=False), cx[q["a"]], cy[q["b"]])
            self.assertEqual((r["value"], r["pairs"]), (want, pairs), q)
        # DuckDB self-joins over the rows: the exact pair sums in HUGEINT
        con = duckdb.connect()
        con.register("t", pd.DataFrame({"rid": np.arange(n), "g": g, "x": x,
                                        "y": pd.Series(np.where(vy, y, 0), dtype="Int64").where(vy)}))
        within = dict(con.execute("SELECT a.g, SUM(CAST(ABS(a.x - b.x) AS HUGEINT)) FROM t a JOIN t b "
                                  "ON a.g = b.g AND a.rid < b.rid GROUP BY a.g").fetchall())
        cross = {(int(r[0]), int(r[1])): (int(r[2]), int(r[3])) for r in con.execute(
            "SELECT a.g, b.g, SUM(CAST(ABS(a.x - b.y) AS HUGEINT)), "
            "SUM(CASE WHEN a.x > b.y THEN 2 WHEN a.x = b.y THEN 1 ELSE 0 END) FROM t a, t b "
            "WHERE b.y IS NOT NULL GROUP BY a.g, b.g").fetchall()}
        con.close()
        for k in range(3):
            self.assertEqual(mp.abs_within(px, k)["value"], int(within[k]))
            for j in range(3):
                self.assertEqual(mp.abs_cross(px, k, j, other=py)["value"], cross[(k, j)][0])
                self.assertEqual(mp.mann_whitney(px, k, j, other=py)["value"], Fraction(cross[(k, j)][1], 2))
        # two states of different decimal scales do not compare
        pd_ = md.histogram(scope, ["d"]).prefix(degree=0)
        with self.assertRaises(DomainError):
            mp.ks(px, 0, 1, other=pd_)
        with self.assertRaises(DomainError):
            mp.ks(px, 0, 1, other=pd_, backend="cuda")
        self.assertEqual(mp.abs_within(pd_, 0)["value"], mp.abs_within(py, 0)["value"])   # within: no comparison

    # ------------------------------------------------------------------ acceptance 2: conventions
    def test_pair001_specified_self_pair_ordering_normalization_and_empty_class_conventions(self):
        # class 0: 1, 2, 2, 3 (a tie); class 1: 4, 4, 5; class 2: one element; class 3: only NULL; class 4: absent;
        # class 5: zero weights only (weighted state)
        x = [1, 2, 2, 3, 5, 4, 4, 7, None, None]
        g = [0, 0, 0, 0, 1, 1, 1, 2, 3, 3]
        w = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        vals = [0 if v is None else v for v in x]
        scope = mbk.Scope.from_arrays({"x": vals, "w": w}, valid={"x": [v is not None for v in x]}, groups=g,
                                      ngroups=6)
        p = md.histogram(scope, ["x"]).prefix(degree=0)
        A = [1, 2, 2, 3]
        # within one class: unordered pairs {i, j}: 6 pairs, sum 6; ordered: 12 pairs, sum 12; self-pairs add M
        # pairs of difference 0
        for ordered, self_pairs, pairs, total in ((False, False, 6, 6), (True, False, 12, 12), (False, True, 10, 6),
                                                  (True, True, 16, 12)):
            r = mp.abs_within(p, 0, ordered=ordered, self_pairs=self_pairs)
            self.assertEqual((r["value"], r["pairs"]), (total, pairs))
            m = mp.abs_within(p, 0, ordered=ordered, self_pairs=self_pairs, mean=True)
            self.assertEqual(m["value"], value(Fraction(total, pairs)))
        # the Gini mean difference: the mean over pairs without self-pairs, ordered or not (equal)
        self.assertEqual(mp.abs_within(p, 0, mean=True)["value"], mp.abs_within(p, 0, mean=True, ordered=True)["value"])
        self.assertEqual(mp.abs_within(p, 0, mean=True)["value"], o_abs_within(A, False, False, True)[0])
        # one element: no pair without self-pairs (sum 0, mean empty); with self-pairs one pair of difference 0
        self.assertEqual((mp.abs_within(p, 2)["value"], mp.abs_within(p, 2)["pairs"]), (0, 0))
        self.assertTrue(mp.abs_within(p, 2, mean=True)["empty"])
        self.assertEqual(mp.abs_within(p, 2, self_pairs=True, mean=True)["value"], 0)
        # Mann-Whitney: ties count half; class order matters (U_AB + U_BA = M_A M_B)
        self.assertEqual(mp.mann_whitney(p, 1, 0)["value"], 12)
        self.assertEqual(mp.mann_whitney(p, 0, 1)["value"], 0)
        u = mp.mann_whitney(p, 0, 0)                            # the same class: 12 ordered pairs, 2 tied
        self.assertEqual((u["value"], u["pairs"]), (6, 12))
        u = mp.mann_whitney(p, 0, 0, self_pairs=True)            # + 4 self-pairs, each a tie (1/2)
        self.assertEqual((u["value"], u["pairs"]), (8, 16))
        self.assertEqual(mp.mann_whitney(p, 0, 0, auc=True)["value"], Fraction(1, 2))
        # empty classes (only NULL values, absent): sums over no pair are 0, ratios are empty
        for empty in (3, 4):
            self.assertEqual(mp.abs_within(p, empty)["value"], 0)
            self.assertTrue(mp.abs_within(p, empty, mean=True)["empty"])
            self.assertEqual(mp.abs_cross(p, 0, empty)["value"], 0)
            self.assertEqual(mp.mann_whitney(p, 0, empty)["value"], 0)
            self.assertTrue(mp.mann_whitney(p, empty, 0, auc=True)["empty"])
            for fn in (mp.ks, mp.wasserstein, mp.cramer_von_mises):
                r = fn(p, 0, empty)
                self.assertTrue(r["empty"])
                self.assertIsNone(r["value"])
        # KS: where the supremum is attained; identical distributions give 0 and no point
        r = mp.ks(p, 0, 1)
        self.assertEqual((r["value"], r["at"]), (1, 3))
        self.assertEqual((mp.ks(p, 0, 0)["value"], mp.ks(p, 0, 0)["at"]), (0, None))
        self.assertEqual(mp.wasserstein(p, 0, 1)["value"], Fraction(7, 3))
        self.assertEqual(mp.cramer_von_mises(p, 0, 1)["value"], Fraction(347, 588))
        # weights: zero-weight rows leave the class; weights replicate (units)
        wscope = mbk.Scope.from_arrays({"x": [1, 2, 3, 9], "w": [2, 0, 1, 0]}, groups=[0, 0, 0, 1], ngroups=2)
        pw = md.histogram(wscope, ["x"], weight="w").prefix("weight", degree=0)
        self.assertEqual(mp.abs_within(pw, 0)["value"], o_abs_within([1, 1, 3], False, False, False)[0])
        self.assertEqual(mp.abs_within(pw, 0, mean=True)["pairs"], 3)
        self.assertTrue(mp.ks(pw, 0, 1)["empty"])                 # class 1: one row of weight 0
        # the same request on the device
        items = every_request(0, 1) + every_request(0, 0) + every_request(3, 4)
        self.same_on_cuda(p, items)

    def test_pair001_cuda_batches_equal_cpu_on_many_groups_scrambled(self):
        rng = np.random.default_rng(2004)
        n, G = 6000, 40
        x, w, vx, groups = (rng.integers(-500, 500, n), rng.integers(0, 1000, n), rng.random(n) > 0.05,
                            rng.integers(0, G, n))
        scope = mbk.Scope.from_arrays({"x": x, "w": w}, valid={"x": vx}, groups=groups, ngroups=G)
        xs = [int(v) if ok else None for v, ok in zip(x, vx)]
        for mass, weight in (("count", None), ("weight", "w")):
            p = md.histogram(scope, ["x"], weight=weight).prefix(mass, degree=0)
            items = [dict(op=OPS[int(rng.integers(0, 6))], a=int(rng.integers(0, G)), b=int(rng.integers(0, G)),
                          ordered=bool(rng.integers(0, 2)), self_pairs=bool(rng.integers(0, 2)),
                          mean=bool(rng.integers(0, 2))) for _ in range(800)]
            for q in items:                               # only the flags each op accepts
                if q["op"] != "abs_within":
                    q["ordered"] = False
                if q["op"] in ("ks", "wasserstein", "cramer_von_mises"):
                    q["self_pairs"] = q["mean"] = False
            cpu = self.same_on_cuda(p, items)
            self.assertTrue((cpu["status"] == 0).all())
            if weight is None:                              # spot checks against the enumeration (about 150 rows)
                classes = {k: expand(xs, None, None, groups, k) for k in range(G)}
                for q in items[:12]:
                    r = mp.query(p, [q])[0]
                    q = dict(q, same=q["op"] != "abs_within" and q["a"] == q["b"])
                    self.assertEqual(r["value"], oracle(q, classes[q["a"]], classes[q["b"]])[0], q)

    def test_pair001_extreme_values_are_exact_and_overflow_is_refused(self):
        big = (1 << 62) - 1
        x = [-big, -big, big, 3, big, -5]
        g = [0, 0, 0, 0, 1, 1]
        scope = mbk.Scope.from_arrays({"x": x, "w": [1 << 61] * 6}, groups=g, ngroups=2)
        p = md.histogram(scope, ["x"]).prefix(degree=0)
        A, B = sorted(x[:4]), sorted(x[4:])
        for q in every_request(0, 1):
            r = mp.query(p, [q])[0]
            want, pairs = oracle(dict(q, same=False), A, B if q["op"] != "abs_within" else None)
            self.assertEqual((r["value"], r["pairs"]), (want, pairs), q)
        self.same_on_cuda(p, every_request(0, 1))
        # weight masses: a class of 4 x 2^61 units is below 2^64 (exact); 8 x 2^61 is not (refused per request)
        pw = md.histogram(scope, ["x"], weight="w").prefix("weight", degree=0)
        r = mp.abs_within(pw, 1)
        self.assertEqual(r["value"], (1 << 61) ** 2 * (big + 5))
        self.assertEqual(r["pairs"], (2 * (1 << 61)) * (2 * (1 << 61) - 1) // 2)
        scope2 = mbk.Scope.from_arrays({"x": list(range(8)), "w": [1 << 61] * 8})
        pw2 = md.histogram(scope2, ["x"], weight="w").prefix("weight", degree=0)
        for backend in ("cpu", "cuda"):
            with self.assertRaises(OverflowError):
                mp.abs_within(pw2, 0, backend=backend)
            with self.assertRaises(OverflowError):
                mp.ks(pw2, 0, 0, backend=backend)
        # Cramer-von Mises needs (M_A M_B)^2 (M_A + M_B) < 2^255: a 2^61-unit class against a 2^62-unit class fits
        # the masses but not that bound
        scope3 = mbk.Scope.from_arrays({"x": [1, 2, 3], "w": [1 << 61, 1 << 61, 1 << 62]}, groups=[0, 1, 1],
                                       ngroups=2)
        pw3 = md.histogram(scope3, ["x"], weight="w").prefix("weight", degree=0)
        self.assertEqual(mp.ks(pw3, 0, 1)["value"], 1)
        self.assertEqual(mp.wasserstein(pw3, 0, 1)["value"], Fraction(1, 3) * 1 + Fraction(2, 3) * 2)
        for backend in ("cpu", "cuda"):
            with self.assertRaises(OverflowError):
                mp.cramer_von_mises(pw3, 0, 1, backend=backend)

    def test_pair001_budgets_invalid_requests_and_opt_in_import(self):
        rng = np.random.default_rng(2006)
        scope = mbk.Scope.from_arrays({"x": rng.integers(0, 50, 500)}, groups=rng.integers(0, 4, 500), ngroups=4)
        p = md.histogram(scope, ["x"]).prefix(degree=0)
        items = every_request(0, 1)
        reqs = mp.requests(items)
        n = len(items)
        for backend in ("cpu", "cuda"):
            self.assertEqual(len(mp.query_arrays(p, reqs, backend=backend, budget=Budget(output=n))), n)
            with self.assertRaises(AlgebraBudgetError) as ctx:
                mp.query_arrays(p, reqs, backend=backend, budget=Budget(output=n - 1))
            self.assertEqual((ctx.exception.resource, ctx.exception.need), ("output", n))
        device_bytes = n * (32 + 152)
        p.device()
        self.assertEqual(len(mp.query_arrays(p, reqs, backend="cuda", budget=Budget(bytes=device_bytes))), n)
        with self.assertRaises(AlgebraBudgetError) as ctx:
            mp.query_arrays(p, reqs, backend="cuda", budget=Budget(bytes=device_bytes - 1))
        self.assertEqual((ctx.exception.resource, ctx.exception.need), ("bytes", device_bytes))
        # each request has its own status: an invalid one does not fail the batch natively
        bad = mp.requests([dict(op="ks", a=0, b=1), dict(op="ks", a=0, b=9), dict(op="abs_within", a=7)])
        bad = np.concatenate([bad, bad[:1]])
        bad["flags"][3] = 1                                     # ordered does not apply to ks
        for backend in ("cpu", "cuda"):
            out = mp.query_arrays(p, bad, backend=backend)
            self.assertEqual(list(out["status"]), [0, -1, -1, -1])
        raw = mp.requests([dict(op="ks", a=0, b=1)])
        raw["op"][0] = 99
        self.assertEqual(int(mp.query_arrays(p, raw)["status"][0]), -1)
        raw["op"][0], raw["reserved"][0] = 4, 5
        self.assertEqual(int(mp.query_arrays(p, raw)["status"][0]), -1)
        with self.assertRaises(ValueError):
            mp.ks(p, 0, 9)
        with self.assertRaises(ValueError):
            mp.requests([dict(op="energy", a=0, b=1)])
        with self.assertRaises(TypeError):
            mp.ks(md.histogram(scope, ["x"]), 0, 1)              # a histogram, not a prefix state
        with self.assertRaises(ValueError):
            mp.query(p, items, backend="tpu")
        code = "import sys, torbles; print('torbles.math_pair' in sys.modules)"
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.split()
        self.assertEqual(out, ["False"])


if __name__ == "__main__":
    unittest.main()
