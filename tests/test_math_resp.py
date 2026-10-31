# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""RESP-002: complete piecewise absolute, hinge and squared-hinge loss curves (torbles/math_resp.c,
math_resp_gpu.cu, math_resp.py) over DIST-002 prefix states.

Oracles, independent of the implementation: direct row sums in Python Fractions (each row's loss and its textbook
one-sided derivatives at theta, summed over the rows, never through prefix masses), and DuckDB SUM over the typed
source table for integer theta. Every batch is also evaluated by the CUDA kernel (emulated here,
TORBLES_EMU_SCRAMBLE=1 scrambles the thread order) and compared bit for bit with the CPU answers. Seeds are fixed.
"""
import math
import unittest
from fractions import Fraction

import numpy as np
import pandas as pd

import torbles
from torbles import math_bank as mbk, math_dist as md, math_resp as mr
from torbles.math_budgets import Budget
from torbles.math_errors import UnsupportedError
from torbles.mathematics import AlgebraBudgetError

LOSSES = [("abs", None), ("hinge", None), ("hinge_down", None), ("sqhinge", None), ("sqhinge_down", None),
          ("squared", None), ("pinball", Fraction(1, 4)), ("pinball", Fraction(1, 2)), ("pinball", Fraction(5, 7)),
          ("pinball", Fraction(0)), ("pinball", Fraction(1))]


def row_terms(loss, tau, x, t):
    """(value, left derivative, right derivative) of one row's loss at theta = t (textbook one-sided derivatives)"""
    u = x - t
    if loss == "abs":
        return abs(u), (1 if x < t else -1), (1 if x <= t else -1)
    if loss == "hinge":
        return max(0, u), (-1 if x >= t else 0), (-1 if x > t else 0)
    if loss == "hinge_down":
        return max(0, -u), (1 if x < t else 0), (1 if x <= t else 0)
    if loss == "sqhinge":
        return max(0, u) ** 2, -2 * max(0, u), -2 * max(0, u)
    if loss == "sqhinge_down":
        return max(0, -u) ** 2, 2 * max(0, -u), 2 * max(0, -u)
    if loss == "squared":
        return u * u, -2 * u, -2 * u
    up, down = row_terms("hinge", None, x, t), row_terms("hinge_down", None, x, t)
    return tuple(tau * a + (1 - tau) * b for a, b in zip(up, down))


def direct(loss, tau, rows, t):
    """sum over the rows (x, mass) of the row loss and its one-sided derivatives, exactly"""
    v = l = r = Fraction(0)
    for x, m in rows:
        a, b, c = row_terms(loss, tau, x, t)
        v += m * a; l += m * b; r += m * c
    return v, l, r


def thetas(values):
    xs = sorted(set(values))
    if not xs:
        return [Fraction(0), Fraction(5, 3)]
    out = [Fraction(xs[0] - 2), Fraction(xs[-1] + 3), Fraction(2 * xs[0] - 1, 2), Fraction(2 * xs[-1] + 1, 2)]
    for a, b in zip(xs, xs[1:]):
        out += [Fraction(a + b, 2), Fraction(2 * a + b, 3), Fraction(a * 7 + b, 8)]
    return out + [Fraction(x) for x in xs]


def mask(rng, n, p):
    return rng.random(n) < p


class Losses(unittest.TestCase):
    def data(self, rng, n=500, G=5, span=10, weighted=False):
        x = [int(a) for a in rng.integers(-span, span + 1, n)]
        w = [int(a) for a in rng.integers(0, 7, n)]
        vx, vw = mask(rng, n, 0.85), mask(rng, n, 0.9)
        groups = rng.integers(0, G, n)
        groups[groups == 1] = 0                                       # group 1 absent
        groups[:4] = G - 1                                            # group G - 1: NULL values only
        vx[:4] = False
        groups[4:] = np.where(groups[4:] == G - 1, 2, groups[4:])
        scope = mbk.Scope.from_arrays({"x": x, "w": w}, valid={"x": vx, "w": vw}, groups=groups, ngroups=G)
        rows = {}
        for i in range(n):
            if not vx[i]:
                continue
            m = (w[i] if vw[i] else 0) if weighted else 1
            rows.setdefault(int(groups[i]), []).append((x[i], m))
        p = md.histogram(scope, ["x"], weight="w" if weighted else None).prefix("weight" if weighted else "count", 2)
        return scope, p, rows

    def batch(self, loss, p, items, tau=None):
        """CPU answers, checked equal bit for bit to the CUDA kernel's"""
        reqs = np.concatenate([mr.requests(loss, ts, g, tau) for g, ts in items]) if items else mr.requests(loss, [])
        cpu = mr.evaluate_arrays(p, reqs)
        gpu = mr.evaluate_arrays(p, reqs, backend="cuda")
        self.assertTrue(np.array_equal(cpu, gpu), loss)
        return cpu

    # ------------------------------------------------------------------ acceptance 1: values and one-sided slopes
    def test_resp002_values_and_slopes_at_and_between_every_breakpoint_match_row_sums(self):
        rng = np.random.default_rng(3001)
        for weighted in (False, True):
            scope, p, rows = self.data(rng, weighted=weighted)
            for loss, tau in LOSSES:
                L = mr.Loss(p, loss, tau=tau)
                items = [(g, thetas([x for x, _ in rows.get(g, [])])) for g in range(scope.groups)]
                self.batch(loss, p, items, tau)
                for g, ts in items:
                    got = L.evaluate(ts, group=g)
                    self.assertEqual(got, L.evaluate(ts, group=g, backend="cuda"))
                    support = {x for x, m in rows.get(g, []) if m}
                    for t, r in zip(ts, got):
                        v, left, right = direct(loss, tau, rows.get(g, []), t)
                        self.assertEqual((r["value"], r["left"], r["right"]), (v, left, right), (weighted, loss, tau, g, t))
                        self.assertEqual(r["at_breakpoint"], t in support)
                        self.assertEqual(r["index"], sum(1 for x in sorted(support) if x < t))

    def test_resp002_values_match_duckdb_over_a_decimal_column(self):
        rng = np.random.default_rng(3002)
        n = 1500
        d = [None if rng.random() < 0.1 else f"{int(a) / 100:.2f}" for a in rng.integers(-500, 500, n)]
        w = [None if rng.random() < 0.1 else int(a) for a in rng.integers(0, 9, n)]
        k = [int(a) for a in rng.integers(0, 4, n)]
        db = torbles.from_ddl("CREATE TABLE f (k BIGINT, d DECIMAL(10,2), w BIGINT)",
                              {"f": pd.DataFrame({"k": pd.Series(k, dtype=object), "d": pd.Series(d, dtype=object),
                                                  "w": pd.Series(w, dtype=object)})})
        scope = mbk.Scope.from_table(db["f"], ["d", "w"], group_by=["k"])
        gmap = {scope.key_values(g): g for g in range(scope.groups)}
        pc = md.histogram(scope, ["d"]).prefix("count", 2)
        pw = md.histogram(scope, ["d"], weight="w").prefix("weight", 2)
        lg = 'CAST(d * 100 AS HUGEINT)'
        for t in (-500, -37, 0, 1, 250, 499, 700):
            sql = (f"SELECT k, SUM(ABS({lg} - {t})), SUM(GREATEST({lg} - {t}, 0)), SUM(GREATEST({t} - {lg}, 0)), "
                   f"SUM(({lg} - {t}) * ({lg} - {t})), SUM(w * ABS({lg} - {t})), "
                   f"SUM(w * GREATEST({lg} - {t}, 0) * GREATEST({lg} - {t}, 0)) FROM f WHERE d IS NOT NULL GROUP BY k")
            res = db.con.execute(sql).fetchall()
            self.assertEqual(len(res), 4)
            for key, a, h, hd, sq, wa, wsq in res:
                g = gmap[(key,)]
                got = [mr.Loss(pc, loss).value(t, group=g) for loss in ("abs", "hinge", "hinge_down", "squared")]
                self.assertEqual(got, [int(a), int(h), int(hd), int(sq)], (t, key))
                self.assertEqual(got, [mr.Loss(pc, loss).value(t, group=g, backend="cuda")
                                       for loss in ("abs", "hinge", "hinge_down", "squared")])
                self.assertEqual(mr.Loss(pw, "abs").value(t, group=g, backend="cuda"), int(wa or 0))
                self.assertEqual(mr.Loss(pw, "sqhinge").value(t, group=g), int(wsq or 0))
        self.assertEqual(pc.info["scale"], 2)

    def test_resp002_complete_pieces_reproduce_the_function(self):
        rng = np.random.default_rng(3003)
        for weighted in (False, True):
            scope, p, rows = self.data(rng, n=300, weighted=weighted)
            for loss, tau in LOSSES:
                L = mr.Loss(p, loss, tau=tau)
                for g in range(scope.groups):
                    support = sorted({x for x, m in rows.get(g, []) if m})
                    pieces = L.pieces(group=g)
                    self.assertEqual(len(pieces), 1 if loss == "squared" or not support else len(support) + 1)
                    self.assertIsNone(pieces[0]["lo"]); self.assertIsNone(pieces[-1]["hi"])
                    if loss != "squared":
                        self.assertEqual([pc["hi"] for pc in pieces[:-1]], support)
                        self.assertEqual([pc["lo"] for pc in pieces[1:]], support)
                    for i, pc in enumerate(pieces):
                        lo, hi = pc["lo"], pc["hi"]
                        pts = [Fraction(lo if lo is not None else (hi if hi is not None else 0) - 3),
                               Fraction(hi if hi is not None else (lo if lo is not None else 0) + 3)]
                        pts.append((pts[0] + pts[1]) / 2)
                        c0, c1, c2 = pc["coefficients"]
                        for t in pts:                               # the polynomial on the closed piece
                            self.assertEqual(c0 + c1 * t + c2 * t * t, direct(loss, tau, rows.get(g, []), t)[0],
                                             (loss, g, i, t))
                        if i and loss != "squared":                 # continuity at the shared breakpoint
                            a0, a1, a2 = pieces[i - 1]["coefficients"]
                            self.assertEqual(a0 + a1 * lo + a2 * lo * lo, c0 + c1 * lo + c2 * lo * lo)

    # ------------------------------------------------------------------ acceptance 2: ties, empty input, degenerate minima
    def minimize(self, loss, tau, values, weights=None):
        s = mbk.Scope.from_arrays({"x": values or [0], "w": weights or [1] * max(len(values), 1)},
                                  rows=np.array([True] * len(values) if values else [False]))
        p = md.histogram(s, ["x"], weight="w").prefix("weight" if weights else "count", 2)
        return mr.Loss(p, loss, tau=tau).minimizers(), p

    def test_resp002_ties_empty_input_and_degenerate_minima_follow_the_convention(self):
        F = Fraction
        cases = [
            ("abs", None, [1, 2, 2, 3], None, ("point", 2, 2, 2)),         # duplicates: the median ties
            ("abs", None, [1, 3], None, ("interval", 1, 3, 1)),            # a flat bottom: the whole interval
            ("abs", None, [1, 3, 3, 8, 9, 9], None, ("interval", 3, 8, 3)),
            ("abs", None, [5, 5, 5], None, ("point", 5, 5, 5)),            # all ties
            ("abs", None, [4], None, ("point", 4, 4, 4)),
            ("abs", None, [1, 2, 10], [0, 3, 0], ("point", 2, 2, 2)),      # zero weights leave the support
            ("abs", None, [1, 2, 10], [1, 1, 2], ("interval", 2, 10, 2)),  # replicated weights
            ("hinge", None, [3, -1, 7, 7], None, ("up", 7, None, 7)),
            ("sqhinge", None, [3, -1, 7, 7], None, ("up", 7, None, 7)),
            ("hinge_down", None, [3, -1, 7, 7], None, ("down", None, -1, -1)),
            ("sqhinge_down", None, [3, -1, 7, 7], None, ("down", None, -1, -1)),
            ("pinball", F(0), [3, -1, 7], None, ("down", None, -1, -1)),
            ("pinball", F(1), [3, -1, 7], None, ("up", 7, None, 7)),
            ("pinball", F(1, 4), [1, 2, 3, 4], None, ("interval", 1, 2, 1)),   # tau M = 1: an integer rank
            ("pinball", F(1, 3), [1, 2, 3, 4], None, ("point", 2, 2, 2)),
            ("pinball", F(1, 2), [1, 3], None, ("interval", 1, 3, 1)),         # the ABS set
            ("squared", None, [1, 2, 2, 4], None, ("point", F(9, 4), F(9, 4), F(9, 4))),
            ("squared", None, [6, 6], None, ("point", 6, 6, 6)),
        ]
        for loss, tau, values, weights, (kind, lo, hi, point) in cases:
            m, p = self.minimize(loss, tau, values, weights)
            self.assertEqual((m["kind"], m["lo"], m["hi"], m["point"]), (kind, lo, hi, point), (loss, tau, values))
            rows = list(zip(values, weights or [1] * len(values)))
            grid = sorted({F(v) for v in values} | {F(2 * v + 1, 2) for v in values} | {F(min(values) - 3), F(max(values) + 3)})
            best = min(direct(loss, tau, rows, t)[0] for t in grid + ([m["point"]] if m["point"] is not None else []))
            self.assertEqual(m["value"], direct(loss, tau, rows, m["point"])[0])
            self.assertEqual(m["value"], best, (loss, values))
            for t in grid:                                         # inside the set: minimal; outside: strictly larger
                inside = (m["lo"] is None or t >= m["lo"]) and (m["hi"] is None or t <= m["hi"])
                v = direct(loss, tau, rows, t)[0]
                self.assertEqual(v == m["value"], inside, (loss, tau, values, t))
            L = mr.Loss(p, loss, tau=tau)                          # the native values on the grid, CPU and CUDA
            self.assertEqual([r["value"] for r in L.evaluate(grid)], [direct(loss, tau, rows, t)[0] for t in grid])
            self.assertEqual(L.evaluate(grid), L.evaluate(grid, backend="cuda"))
        # empty input: an absent group, a group of NULLs only, an empty selection; the loss is 0 everywhere
        s = mbk.Scope.from_arrays({"x": [1, 2, 3]}, valid={"x": [False, True, True]}, groups=[0, 2, 2], ngroups=3)
        p = md.histogram(s, ["x"]).prefix()
        for loss, tau in LOSSES:
            L = mr.Loss(p, loss, tau=tau)
            for g in (0, 1):
                self.assertEqual(L.minimizers(group=g), dict(kind="all", lo=None, hi=None, point=None, value=0,
                                                              lo_index=None, hi_index=None))
                ev = L.evaluate([F(-7, 3), 0, 2], group=g)
                self.assertTrue(all(r["value"] == r["left"] == r["right"] == 0 for r in ev))
                self.assertEqual(ev, L.evaluate([F(-7, 3), 0, 2], group=g, backend="cuda"))
                self.assertEqual(len(L.pieces(group=g)), 1)
        m, _ = self.minimize("abs", None, [])
        self.assertEqual(m["kind"], "all")
        # random data: the minimum is the smallest value over every breakpoint (and the mean for squared)
        rng = np.random.default_rng(3004)
        for weighted in (False, True):
            scope, p, rows = self.data(rng, weighted=weighted)
            for loss, tau in LOSSES:
                L = mr.Loss(p, loss, tau=tau)
                for g in range(scope.groups):
                    r = rows.get(g, [])
                    m = L.minimizers(group=g)
                    support = sorted({x for x, w in r if w})
                    if not support:
                        self.assertEqual(m["kind"], "all")
                        continue
                    cand = [F(x) for x in support] + [m["point"]]
                    self.assertEqual(m["value"], min(direct(loss, tau, r, t)[0] for t in cand))
                    self.assertEqual(L.value(m["point"], group=g), m["value"])

    # ------------------------------------------------------------------ CUDA, exactness, budgets, refusals
    def test_resp002_cuda_batches_equal_cpu(self):
        rng = np.random.default_rng(3005)
        n, G = 40000, 40
        scope = mbk.Scope.from_arrays({"x": rng.integers(-10**6, 10**6, n), "w": rng.integers(0, 100, n)},
                                      valid={"x": mask(rng, n, 0.95)}, groups=rng.integers(0, G, n), ngroups=G)
        for mass, weight in (("count", None), ("weight", "w")):
            p = md.histogram(scope, ["x"], weight=weight).prefix(mass, 2)
            N = 10000
            r = np.zeros(N, dtype=mr.REQUEST_DTYPE)
            r["loss"] = rng.integers(1, 8, N)
            r["group"] = rng.integers(0, G + 1, N)                    # some past the groups
            r["num"] = rng.integers(-2 * 10**6, 2 * 10**6, N)
            r["den"] = rng.integers(0, 50, N)                         # den 0: an invalid request
            r["tau_den"] = rng.integers(1, 10, N)
            r["tau_num"] = (rng.random(N) * (r["tau_den"] + 1)).astype(np.int64)   # sometimes tau > 1: invalid
            cpu = mr.evaluate_arrays(p, r)
            gpu = mr.evaluate_arrays(p, r, backend="cuda")
            self.assertTrue(np.array_equal(cpu, gpu))
            ok = cpu["status"] == 0
            self.assertGreater(int(ok.sum()), N // 2)
            self.assertTrue((cpu["status"][(r["den"] == 0) | (r["group"] >= G)] == -1).all())

    def test_resp002_large_magnitudes_are_exact(self):
        big = [(1 << 62) - 5, -(1 << 62) + 3, (1 << 61) + 7]
        s = mbk.Scope.from_arrays({"x": big})
        p = md.histogram(s, ["x"]).prefix("count", 2)                 # T x X^2 = 3 x 2^124 < 2^127
        rows = [(x, 1) for x in big]
        ts = [Fraction((1 << 62) - 1, (1 << 61) + 1), Fraction(-(1 << 62) + 1, 3), Fraction(big[2]),
              Fraction(5, (1 << 62) - 1), Fraction(-(1 << 63), (1 << 62) + 13)]
        for loss, tau in LOSSES + [("pinball", Fraction((1 << 61) + 1, (1 << 62) - 1))]:
            L = mr.Loss(p, loss, tau=tau)
            got = L.evaluate(ts)
            self.assertEqual(got, L.evaluate(ts, backend="cuda"))
            for t, r in zip(ts, got):
                self.assertEqual((r["value"], r["left"], r["right"]), direct(loss, tau, rows, t), (loss, t))
        m = mr.Loss(p, "squared").minimizers()
        self.assertEqual(m["point"], Fraction(sum(big), 3))
        self.assertEqual(m["value"], sum((x - Fraction(sum(big), 3)) ** 2 for x in big))

    def test_resp002_budgets_and_refusals(self):
        rng = np.random.default_rng(3006)
        scope, p, rows = self.data(rng)
        L = mr.Loss(p, "abs")
        ts = [Fraction(k, 3) for k in range(-30, 31)]
        for backend in ("cpu", "cuda"):
            self.assertEqual(len(L.evaluate(ts, backend=backend, budget=Budget(output=len(ts)))), len(ts))
            with self.assertRaises(AlgebraBudgetError) as ctx:
                L.evaluate(ts, backend=backend, budget=Budget(output=len(ts) - 1))
            self.assertEqual((ctx.exception.resource, ctx.exception.need), ("output", len(ts)))
        n = len(L.pieces(group=0))
        self.assertEqual(len(L.pieces(group=0, budget=Budget(output=n))), n)
        with self.assertRaises(AlgebraBudgetError) as ctx:
            L.pieces(group=0, budget=Budget(output=n - 1))
        self.assertEqual((ctx.exception.resource, ctx.exception.need), ("output", n))
        # the CUDA batch's device copies are budgeted (bytes)
        need = len(ts) * (48 + 144)
        p.device()
        with self.assertRaises(AlgebraBudgetError) as ctx:
            L.evaluate(ts, backend="cuda", budget=Budget(bytes=need - 1))
        self.assertEqual((ctx.exception.resource, ctx.exception.need), ("bytes", need))
        # a prefix of degree 1 keeps no second moments: the squared forms are unsupported, the others answer
        p1 = md.histogram(scope, ["x"]).prefix("count", 1)
        for loss in ("sqhinge", "sqhinge_down", "squared"):
            for call in (lambda: mr.Loss(p1, loss).value(0), lambda: mr.Loss(p1, loss).value(0, backend="cuda"),
                         lambda: mr.Loss(p1, loss).pieces(), lambda: mr.Loss(p1, loss).minimizers()):
                with self.assertRaises(UnsupportedError):
                    call()
        self.assertEqual(mr.Loss(p1, "abs").value(Fraction(1, 2), group=2), L.value(Fraction(1, 2), group=2))
        for bad in (lambda: mr.Loss(p, "pinball"), lambda: mr.Loss(p, "pinball", tau=Fraction(3, 2)),
                    lambda: mr.Loss(p, "abs", tau=Fraction(1, 2)), lambda: mr.Loss(p, "huber"),
                    lambda: L.evaluate([0], group=99), lambda: L.pieces(group=99), lambda: L.minimizers(group=99)):
            with self.assertRaises(ValueError):
                bad()
        with self.assertRaises(TypeError):
            L.value(0.5)                                              # exact theta only
        with self.assertRaises(OverflowError):
            L.value(Fraction(1, 1 << 64))
        with self.assertRaises(TypeError):
            mr.Loss(md.histogram(scope, ["x"]), "abs")               # a prefix state is required


if __name__ == "__main__":
    unittest.main()
