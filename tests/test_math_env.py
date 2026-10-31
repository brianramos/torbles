# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""ENV-001: univariate affine / quadratic group envelopes (torbles/math_env.c, math_env_gpu.cu, math_env.py).

Oracles, independent of the implementation: every group function evaluated with Python Fractions at a rational
point strictly inside every piece and at every rational boundary and breakpoint; at an algebraic breakpoint, exact
arithmetic in Q(sqrt(D)) (x + y sqrt(D) with Fractions, signs by squaring) of every group function at the root
(-q1 + s sqrt(D)) / (2 q2), whose isolating interval is checked (a sign change of q, the right root) and narrowed by
Python bisection to order the points; the response functions of a bank against the losses summed over the raw rows.
Every lookup is also computed by the CUDA kernel (emulated here, TORBLES_EMU_SCRAMBLE=1) and compared for equality,
and the binary-search lookup against the all-group exact evaluation. Seeds are fixed.
"""
import subprocess
import sys
import unittest
from fractions import Fraction as F

import numpy as np

from torbles import math_bank as mbk
from torbles import math_env as me
from torbles.math_budgets import Budget
from torbles.math_errors import UnsupportedError
from torbles.mathematics import AlgebraBudgetError, Context


def sgn(x):
    return (x > 0) - (x < 0)


def poly_at(c, t):
    return sum(F(ci) * t ** i for i, ci in enumerate(c))


class QD:
    """x + y sqrt(d) with Fractions x, y and an integer d > 0"""
    def __init__(self, x, y, d):
        self.x, self.y, self.d = F(x), F(y), d

    def __add__(self, o):
        o = o if isinstance(o, QD) else QD(o, 0, self.d)
        return QD(self.x + o.x, self.y + o.y, self.d)

    def __sub__(self, o):
        o = o if isinstance(o, QD) else QD(o, 0, self.d)
        return QD(self.x - o.x, self.y - o.y, self.d)

    def __mul__(self, o):
        o = o if isinstance(o, QD) else QD(o, 0, self.d)
        return QD(self.x * o.x + self.y * o.y * self.d, self.x * o.y + self.y * o.x, self.d)

    def sign(self):
        sx, sy = sgn(self.x), sgn(self.y)
        if not sy:
            return sx
        if not sx or sx == sy:
            return sy
        return sx if self.x * self.x > self.y * self.y * self.d else sy


class Alg:
    """the oracle's copy of an algebraic breakpoint: checked, evaluated exactly, narrowed by bisection"""
    def __init__(self, test, p):
        q0, q1, q2 = p.polynomial
        self.q, (self.lo, self.hi) = (q0, q1, q2), p.interval
        test.assertGreater(q2, 0)
        self.d = q1 * q1 - 4 * q2 * q0
        test.assertGreater(self.d, 0)
        test.assertNotEqual(int(self.d ** 0.5 + 0.5) ** 2, self.d)                # irrational: not a square
        sl, sh = sgn(poly_at(self.q, self.lo)), sgn(poly_at(self.q, self.hi))
        test.assertEqual(sl, -sh)                                                 # a certified sign change
        test.assertNotEqual(sl, 0)
        test.assertEqual(p.sign_lo, sl)
        test.assertEqual(p.root, 0 if sl > 0 else 1)                              # q2 > 0: the smaller root falls
        s = -1 if p.root == 0 else 1
        self.value = QD(F(-q1, 2 * q2), F(s, 2 * q2), self.d)
        test.assertEqual((self.value - self.lo).sign(), 1)                        # the root is inside (lo, hi)
        test.assertEqual((self.value - self.hi).sign(), -1)
        self.sl = sl

    def narrow(self):
        mid = (self.lo + self.hi) / 2
        s = sgn(poly_at(self.q, mid))
        assert s != 0
        if s == self.sl:
            self.lo = mid
        else:
            self.hi = mid


def lower_of(p):
    return p if isinstance(p, F) else p.lo


def upper_of(p):
    return p if isinstance(p, F) else p.hi


def separate(a, b):
    """narrow two points (a < b expected) until upper(a) < lower(b)"""
    for _ in range(400):
        if upper_of(a) < lower_of(b):
            return True
        if isinstance(a, Alg):
            a.narrow()
        if isinstance(b, Alg):
            b.narrow()
    return False


def value_at(f, t):
    """f(t) for coefficients f at a Fraction or an algebraic point (a QD)"""
    if isinstance(t, F):
        return poly_at(f, t)
    out = QD(0, 0, t.d)
    p = QD(1, 0, t.d)
    for c in f:
        out = out + p * F(c)
        p = p * t.value
    return out


def argmin_set(functions, t, upper=False):
    """the groups attaining the min (max) of the present functions at t, exactly"""
    best, groups = None, []
    for g, f in enumerate(functions):
        if f is None:
            continue
        v = value_at(f, t)
        if best is None:
            best, groups = v, [g]
            continue
        diff = v - best
        s = diff.sign() if isinstance(diff, QD) else sgn(diff)
        if upper:
            s = -s
        if s < 0:
            best, groups = v, [g]
        elif s == 0:
            groups.append(g)
    return groups


class Env(unittest.TestCase):
    def check(self, env, functions, lo=None, hi=None, upper=False, tie="smallest", extra=()):
        """every region against every group function (the acceptance oracle); lookups CPU = CUDA = all-group"""
        present = [g for g, f in enumerate(functions) if f is not None]
        if not present:
            self.assertEqual((env.kind, env.regions), ("empty", []))
            out = env.lookup([0, 1])
            self.assertEqual([a["status"] for a in out], ["empty", "empty"])
            self.assertEqual(env.lookup([0, 1], backend="cuda"), out)
            return
        pick = min if tie == "smallest" else max
        regs = env.regions
        self.assertEqual(env.info["regions"], len(regs))
        # the regions in order: boundaries, alternation, increasing points
        self.assertEqual(regs[0]["kind"], "point" if lo is not None else "piece")
        self.assertEqual(regs[-1]["kind"], "point" if hi is not None else "piece")
        for a, b in zip(regs, regs[1:]):
            self.assertNotEqual(a["kind"], b["kind"])
        pts = [Alg(self, p) if not isinstance(p, F) else p for p in env.points]
        for a, b in zip(pts, pts[1:]):
            self.assertTrue(separate(a, b), "points are strictly increasing")
        if lo is not None:
            self.assertEqual((env.points[0], regs[0]["boundary"] in ("lo", "lo and hi")), (F(lo), True))
        if hi is not None:
            self.assertEqual((env.points[-1], regs[-1]["boundary"] in ("hi", "lo and hi")), (F(hi), True))
        samples, expect = [], []
        for i, r in enumerate(regs):
            if r["kind"] == "point":
                p = pts[r["point"]]
                groups = argmin_set(functions, p, upper)                         # exact at the point
                self.assertEqual(groups, r["groups"], f"point {i}")
                self.assertEqual(r["winner"], pick(groups))
                self.assertGreaterEqual(len(r["classes"]), 1 if r["boundary"] else 2)
                if isinstance(p, F):
                    samples.append(p); expect.append((r, i))
                continue
            left = pts[env.points.index(r["lo"])] if r["lo"] is not None else None
            right = pts[env.points.index(r["hi"])] if r["hi"] is not None else None
            if left is not None and right is not None:
                self.assertTrue(separate(left, right))
                ts = [(upper_of(left) + lower_of(right)) / 2]
            elif left is not None:
                ts = [upper_of(left) + 1, upper_of(left) + 1000]
            elif right is not None:
                ts = [lower_of(right) - 1, lower_of(right) - 1000]
            else:
                ts = [F(0), F(-10**6), F(10**6)]
            if left is not None:                                                 # just inside the left end too
                ts.append(upper_of(left) + (ts[0] - upper_of(left)) / 7)
            for t in ts:
                groups = argmin_set(functions, t, upper)                         # exact inside the piece
                self.assertEqual(groups, r["groups"], f"piece {i} at {t}")
                self.assertEqual(len(r["classes"]), 1)
                self.assertEqual(r["winner"], pick(groups))
                samples.append(t); expect.append((r, i))
        # lookups: the same answers on the CPU, in CUDA and by the all-group evaluation
        outside = []
        if lo is not None:
            outside.append(F(lo) - F(1, 3))
        if hi is not None:
            outside.append(F(hi) + F(1, 3))
        ts = samples + list(extra) + outside
        cpu = env.lookup(ts)
        self.assertEqual(env.lookup(ts, backend="cuda"), cpu)
        allc = env.evaluate_all(ts)
        self.assertEqual(env.evaluate_all(ts, backend="cuda"), allc)
        for k, (t, a, b) in enumerate(zip(ts, cpu, allc)):
            self.assertEqual((a["status"], a["winner"], a["tied"]), (b["status"], b["winner"], b["tied"]), t)
            if k < len(samples):
                r, i = expect[k]
                self.assertEqual((a["status"], a["region"], a["winner"]), ("ok", i, r["winner"]), t)
            elif k >= len(samples) + len(extra):
                self.assertEqual(a["status"], "outside")
            if a["status"] == "ok":                                              # the oracle at every query
                groups = argmin_set(functions, F(t), upper)
                self.assertEqual(a["winner"], pick(groups))
                val, reg = env.value(F(t))
                self.assertEqual(val, poly_at(functions[groups[0]], F(t)))
                self.assertEqual(reg, a["region"])
        return env

    # ------------------------------------------------------------------ acceptance 1: winners vs every function
    def test_env001_acceptance_winners_against_all_functions_at_every_interval_and_boundary(self):
        rng = np.random.default_rng(3101)

        def rq(lim=20, den=6):
            return F(int(rng.integers(-lim, lim + 1)), int(rng.integers(1, den + 1)))
        domains = [(None, None), (-3, None), (None, F(5, 2)), (F(-7, 3), 4), (0, 0), (-50, 50)]
        for trial in range(18):
            kind = ("affine", "quadratic", "mixed")[trial % 3]
            n = int(rng.integers(2, 11))
            fs = []
            for g in range(n):
                if rng.random() < 0.12:
                    fs.append(None)                                              # an empty group
                elif fs and rng.random() < 0.15:
                    fs.append(next((f for f in fs if f is not None), (1, 1)))    # an identical function
                elif kind == "affine" or (kind == "mixed" and rng.random() < 0.5):
                    fs.append((rq(), rq(5, 4)))
                else:
                    fs.append((rq(), rq(8, 3), rq(4, 3)))
            lo, hi = domains[trial % len(domains)]
            for upper in (False, True):
                env = me.envelope(fs, lo=lo, hi=hi, upper=upper)
                self.check(env, fs, lo, hi, upper, extra=[rq(), rq(100, 7)])
        # a larger mixed set (concave and convex parabolas, lines, duplicates), every region checked
        fs = []
        for g in range(60):
            r = rng.random()
            fs.append(None if r < 0.05 else (rq(200, 9), rq(30, 7)) if r < 0.45 else (rq(200, 9), rq(30, 7), rq(3, 5)))
        fs[17], fs[41] = fs[3], fs[3]
        for lo, hi in ((None, None), (-40, 25)):
            for upper in (False, True):
                env = self.check(me.envelope(fs, lo=lo, hi=hi, upper=upper), fs, lo, hi, upper)
                self.assertGreater(env.info["regions"], 3)
        # many breakpoints: tangent lines of t^2 at the integers (every line wins one piece), plus the parabola
        lines = [(-k * k, 2 * k) for k in range(-6, 7)]
        env = self.check(me.envelope(lines, upper=True), lines, upper=True)
        self.assertEqual((env.info["pieces"], env.info["points"]), (13, 12))
        env = self.check(me.envelope(lines + [(0, 0, 1)], upper=True), lines + [(0, 0, 1)], upper=True)
        self.assertEqual({r["winner"] for r in env.regions if r["kind"] == "piece"}, {13})   # the parabola wins
        self.assertEqual(len(env.breakpoints), 13)                                # touched by every line
        # irrational breakpoints: certified, refinable, ordered against rational ones
        fs = [(0, 0, 1), (2,), (F(1, 2), 1), (-1, 0, 2)]
        env = self.check(me.envelope(fs, lo=-3, hi=3), fs, -3, 3)
        alg = [p for p in env.points if isinstance(p, me.Algebraic)]
        self.assertGreater(len(alg), 0)
        for p in alg:
            lo_, hi_ = p.refine(F(1, 10**30))
            self.assertLessEqual(hi_ - lo_, F(1, 10**30))
            self.assertEqual(sgn(poly_at(p.polynomial, lo_)), -sgn(poly_at(p.polynomial, hi_)))
            self.assertTrue(p.interval[0] <= lo_ < hi_ <= p.interval[1])

    # ------------------------------------------------------------------ acceptance 2: identical, empty, unbounded
    def test_env001_acceptance_identical_empty_and_unbounded_exactly(self):
        same = [(1, 2, 3)] * 5
        env = self.check(me.envelope(same), same)                                 # one class, coincident
        self.assertEqual((len(env.regions), env.regions[0]["groups"], env.regions[0]["winner"]), (1, [0, 1, 2, 3, 4], 0))
        self.assertEqual(me.envelope(same, tie="largest").regions[0]["winner"], 4)
        self.check(me.envelope(same, tie="largest"), same, tie="largest")
        self.assertEqual(len(env.classes), 1)
        empty = [None, None, None]
        env = me.envelope(empty, lo=0, hi=1)
        self.check(env, empty)
        self.assertEqual(env.value(F(1, 2)), (None, None))
        mixed = [None, (5,), None, (5,), (F(9, 2),)]
        env = self.check(me.envelope(mixed), mixed)                               # empty groups never win
        self.assertEqual(env.regions[0]["groups"], [4])
        self.assertTrue(all(g not in r["groups"] for r in env.regions for g in (0, 2)))
        # unbounded on both sides: asymptotic order of the coefficients decides the outer pieces
        fs = [(0, 1), (0, -1), (100,), (0, 0, F(1, 1000))]
        env = self.check(me.envelope(fs), fs)
        self.assertEqual((env.regions[0]["winner"], env.regions[-1]["winner"]), (0, 1))
        env = self.check(me.envelope(fs, upper=True), fs, upper=True)
        self.assertEqual((env.regions[0]["winner"], env.regions[-1]["winner"]), (3, 3))
        # one point domain, three lines through one point, a tangency, a group winning only at one point
        self.check(me.envelope(fs, lo=F(7, 3), hi=F(7, 3)), fs, F(7, 3), F(7, 3))
        three = [(1, 1), (1, -1), (1, 0), (3,)]
        env = self.check(me.envelope(three), three)
        self.assertEqual(env.breakpoints, [F(0)])
        self.assertEqual(env.regions[1]["groups"], [0, 1, 2])
        touch = [(1, -2, 1), (0,)]                                                # (t - 1)^2 touches 0 at t = 1
        env = self.check(me.envelope(touch), touch)
        self.assertEqual([r["kind"] for r in env.regions], ["piece", "point", "piece"])
        self.assertEqual((env.regions[0]["winner"], env.regions[1]["winner"], env.regions[2]["winner"]), (1, 0, 1))
        self.assertEqual(env.regions[1]["at"], F(1))                              # group 0 wins only at t = 1
        # constant ties with boundaries exactly at a crossing, a degenerate domain at a breakpoint
        fs = [(0, 1), (2, -1)]
        self.check(me.envelope(fs, lo=1, hi=3), fs, 1, 3)
        self.check(me.envelope(fs, lo=1, hi=1), fs, 1, 1)
        self.check(me.envelope(fs, lo=-2, hi=1), fs, -2, 1)
        # huge exact coefficients (beyond 128 bits) and breakpoints of large height
        big = [(F(3 ** 90, 7 ** 40), F(-2 ** 100, 3)), (F(-(5 ** 70)), F(2 ** 99, 11), F(1, 3 ** 60)), (F(1, 2 ** 80),)]
        env = self.check(me.envelope(big), big, extra=[F(10**40), F(-10**40, 7)])
        self.assertTrue(env.last_lookup["primes"] >= 2)

    # ------------------------------------------------------------------ the tie policy and upper envelopes
    def test_env001_tie_policy_and_coincident_pieces(self):
        fs = [(0, 1), (3,), (0, 1), (3,), (6, -1), (0, 1)]
        for tie in ("smallest", "largest"):
            for upper in (False, True):
                env = self.check(me.envelope(fs, lo=-1, hi=8, upper=upper, tie=tie), fs, -1, 8, upper, tie)
                for r in env.regions:
                    want = min(r["groups"]) if tie == "smallest" else max(r["groups"])
                    self.assertEqual(r["winner"], want)
        env = me.envelope(fs)
        self.assertEqual([c["groups"] for c in env.classes], [[0, 2, 5], [1, 3], [4]])
        self.assertEqual(env.regions[0]["groups"], [0, 2, 5])                    # coincident pieces reported

    # ------------------------------------------------------------------ response functions from retained moments
    def test_env001_response_functions_from_bank_and_expressions(self):
        rng = np.random.default_rng(3104)
        n, G = 400, 9
        x, y = rng.integers(-20, 21, n), rng.integers(-50, 51, n)
        vx = rng.random(n) < 0.9
        g = rng.integers(0, G - 1, n)                                             # group G - 1 stays empty
        scope = mbk.Scope.from_arrays({"x": x, "y": y}, valid={"x": vx}, groups=g, ngroups=G)
        bank = mbk.build(scope, mbk.monomials(["x", "y"], 2), "joint")
        rows = {k: [(int(a), int(b)) for a, b, ok, gg in zip(x, y, vx, g) if ok and gg == k] for k in range(G)}

        def loss(k, t, mean=False):
            s = sum((F(b) - t * a) ** 2 for a, b in rows[k])
            return s / len(rows[k]) if mean else s
        for mean in (False, True):
            env = me.from_bank(bank, constant="y^2", linear=(-2, "x*y"), quadratic="x^2",
                               per="x^2" if mean else None, lo=-10, hi=10)   # the count of x^2: valid rows
            fs = [None if not rows[k] else (loss(k, F(0), mean), (loss(k, F(1), mean) - loss(k, F(-1), mean)) / 2,
                                             (loss(k, F(1), mean) + loss(k, F(-1), mean)) / 2 - loss(k, F(0), mean))
                  for k in range(G)]
            self.assertEqual([c["coefficients"] for c in env.classes],
                             [tuple(fs[c["groups"][0]]) for c in env.classes])
            self.check(env, fs, -10, 10)
            for t in (F(-3), F(1, 3), F(7, 2)):                                  # against the raw rows
                val, _ = env.value(t)
                self.assertEqual(val, min(loss(k, t, mean) for k in range(G) if rows[k]))
        ctx = Context()
        th, u = ctx.symbols("theta", "u")
        exprs = [(th - 1) ** 2, 3 * th + 1, None, (th - 1) ** 2, 2 * th * th - F(1, 2)]
        env = me.from_expressions(exprs, "theta", lo=-4, hi=4)
        fs = [None, None, None, None, None]
        for i, e in enumerate(exprs):
            if e is not None:
                c0 = F(e.evaluate(theta=0))
                c1 = (F(e.evaluate(theta=1)) - F(e.evaluate(theta=-1))) / 2
                fs[i] = (c0, c1, (F(e.evaluate(theta=1)) + F(e.evaluate(theta=-1))) / 2 - c0)
        self.check(env, fs, -4, 4)
        with self.assertRaises(UnsupportedError):                                 # parameter dimension 2
            me.from_expressions([th + u, th], "theta")
        with self.assertRaises(UnsupportedError):                                 # degree 3
            me.from_expressions([th ** 3, th], "theta")
        with self.assertRaises(UnsupportedError):
            me.envelope([(1, 2, 3, 4), (0,)])

    # ------------------------------------------------------------------ CUDA lookups equal the CPU at scale
    def test_env001_cuda_lookups_equal_cpu_scrambled(self):
        rng = np.random.default_rng(3105)
        n = 64
        fs = [(F(int(rng.integers(-500, 500)), int(rng.integers(1, 9))), F(int(rng.integers(-40, 40)), int(rng.integers(1, 5))),
               F(int(rng.integers(0, 6)), int(rng.integers(1, 4)))) for _ in range(n)]
        env = me.envelope(fs, lo=-30, hi=40)
        ts = [F(int(v), int(d)) for v, d in zip(rng.integers(-3000, 4000, 1500), rng.integers(1, 97, 1500))]
        ts += [p for p in env.points if isinstance(p, F)]
        ts += [p.interval[0] for p in env.points if isinstance(p, me.Algebraic)]
        cpu = env.lookup(ts)
        self.assertEqual(env.lookup(ts, backend="cuda"), cpu)
        small = env.last_lookup["primes"]
        huge = ts + [F(10 ** 40 + 1, 3)]                                         # one value past int64: big rationals
        for backend in ("cpu", "cuda"):
            self.assertEqual(env.lookup(huge, backend=backend)[:-1], cpu)
            self.assertGreater(env.last_lookup["primes"], small)               # a larger proven bound, same answers
        allc = env.evaluate_all(ts, backend="cuda")
        self.assertEqual(allc, env.evaluate_all(ts))
        for t, a, b in zip(ts, cpu, allc):
            self.assertEqual((a["status"], a["winner"], a["tied"]), (b["status"], b["winner"], b["tied"]), t)
        self.assertGreater(env.info["algebraic"], 0)

    # ------------------------------------------------------------------ budgets
    def test_env001_budgets_at_the_limit_and_one_past(self):
        rng = np.random.default_rng(3106)
        fs = [(F(int(rng.integers(-99, 99)), 3), F(int(rng.integers(-9, 9)), 2), F(int(rng.integers(0, 4)), 5))
              for _ in range(12)] + [None, (1, 1)]
        env = me.envelope(fs, lo=-20, hi=20)
        needs = me.need(len(fs), present=13, degree=2)
        self.assertEqual((needs["variables"], needs["degree"], needs["support"], needs["states"]), (1, 2, 13, 14))
        for resource, value in needs.items():
            if resource == "bytes":                                             # the a-priori part of the bytes
                self.assertLessEqual(value, env.info["bytes"])
                continue
            me.envelope(fs, lo=-20, hi=20, budget=Budget(**{resource: value}))
            with self.assertRaises(AlgebraBudgetError) as ctx:
                me.envelope(fs, lo=-20, hi=20, budget=Budget(**{resource: value - 1}))
            self.assertEqual((ctx.exception.resource, ctx.exception.need), (resource, value), resource)
        regions, iters, nbytes = env.info["regions"], env.info["iterations"], env.info["bytes"]
        for resource, value in (("output", regions), ("iterations", iters), ("bytes", nbytes)):   # counted exactly
            self.assertEqual(me.envelope(fs, lo=-20, hi=20, budget=Budget(**{resource: value})).info["regions"], regions)
            with self.assertRaises(AlgebraBudgetError) as ctx:
                me.envelope(fs, lo=-20, hi=20, budget=Budget(**{resource: value - 1}))
            self.assertEqual((ctx.exception.resource, ctx.exception.need), (resource, value), resource)
        with self.assertRaises(AlgebraBudgetError) as ctx:                       # an over-budget region count
            me.envelope(fs, lo=-20, hi=20, budget=Budget(output=2))
        self.assertEqual(ctx.exception.need, regions)                           # the exact count, not a bound
        ts = [F(k, 3) for k in range(-30, 30)]
        for backend in ("cpu", "cuda"):
            for mode in ("search", "all"):
                needs = env.lookup_need(ts, mode=mode, backend=backend)
                self.assertEqual(needs["output"], len(ts))
                ref = env.lookup(ts, mode=mode)
                for resource, value in needs.items():
                    self.assertEqual(env.lookup(ts, backend=backend, mode=mode, budget=Budget(**{resource: value})), ref)
                    with self.assertRaises(AlgebraBudgetError) as ctx:
                        env.lookup(ts, backend=backend, mode=mode, budget=Budget(**{resource: value - 1}))
                    self.assertEqual((ctx.exception.resource, ctx.exception.need), (resource, value), (backend, mode))
        with self.assertRaises(AlgebraBudgetError):
            me.envelope([(0, 0, 1)], budget=Budget(degree=1))                    # degree 2 over a budget of 1
        with self.assertRaises(TypeError):
            me.envelope([(0.5, 1)])                                               # floats are not exact input
        with self.assertRaises(ValueError):
            me.envelope([(1,)], lo=2, hi=1)

    def test_env001_opt_in_import(self):
        code = "import sys, torbles; print('torbles.math_env' in sys.modules)"
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.split()
        self.assertEqual(out, ["False"])


if __name__ == "__main__":
    unittest.main()
