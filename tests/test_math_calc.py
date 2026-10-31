# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""CALC-003: non-polynomial finite-separation calculus (torbles/math_calc.c, math_calc_gpu.cu, math_calc.py).

Oracles, independent of the implementation and written here: sin, cos and exp of exact rationals in 400-bit integer
fixed point (pi by Machin's formula and ln 2 by atanh(1/3) recomputed here, Taylor series in Python integers; error
below 2^-300), applied to the exact data (logical integers / 10^scale, exact frequencies) and to the exact values of
the double evaluation points, so every enclosure is checked to contain the exact value within 2^-150 (Fraction
comparisons). NumPy / math only for the declared numerical claims: the direct float64 evaluation sum w sin(omega x +
phi) must agree with the enclosure within the stated float error margin, math.sin / math.exp of exactly representable
arguments must lie within one ulp of the per-row enclosure, and a per-row enclosure is at most a few ulps wide.
Every native build is also run on the CUDA path (emulated here, scrambled thread order: TORBLES_EMU_SCRAMBLE=1) and
its exact bounds compared for equality. Budgets are tested at the limit and one past it. Seeds are fixed.
"""
import math
import os
import random
import unittest
from fractions import Fraction

import numpy as np

os.environ.setdefault("TORBLES_EMU_SCRAMBLE", "1")

from torbles import math_bank as mbk, math_calc as mc, math_gpu
from torbles.math_budgets import Budget
from torbles.math_domains import DomainError
from torbles.math_errors import StaleStateError, UnsupportedError
from torbles.mathematics import AlgebraBudgetError

SEED = 20261031
PREC = 400
EPS = Fraction(1, 2 ** 150)           # the oracle's error is below 2^-300; enclosures are checked to 2^-150


# ------------------------------------------------------------------ the oracle: exact-argument sin, cos, exp
def _atan_inv(x, prec):
    one = 1 << (prec + 32)
    s, t, k, x2 = 0, one // x, 0, x * x
    while t:
        s += t // (2 * k + 1) if k % 2 == 0 else -(t // (2 * k + 1))
        t //= x2
        k += 1
    return s >> 32


def _atanh_inv(x, prec):
    one = 1 << (prec + 32)
    s, t, k, x2 = 0, one // x, 0, x * x
    while t:
        s += t // (2 * k + 1)
        t //= x2
        k += 1
    return s >> 32


PI = 4 * (4 * _atan_inv(5, PREC) - _atan_inv(239, PREC))          # pi * 2^PREC
LN2 = 2 * _atanh_inv(3, PREC)                                       # ln 2 * 2^PREC
HALF_PI = Fraction(PI, 2 ** (PREC + 1))                             # pi / 2 within 2^-398


def _fixed(q):
    return (q.numerator << PREC) // q.denominator


def sincos_hp(theta):
    """(sin, cos) of an exact rational, as Fractions within 2^-300"""
    theta = Fraction(theta)
    t, half = _fixed(theta), PI >> 1
    k = (2 * t + half) // (2 * half)                                 # the nearest multiple of pi/2
    r = t - k * half
    s, c, term, i = 0, 0, 1 << PREC, 0
    while term:
        if i % 2 == 0:
            c += term if (i // 2) % 2 == 0 else -term
        else:
            s += term if (i // 2) % 2 == 0 else -term
        i += 1
        term = (term * r >> PREC) // i if term * r >= 0 else -((-term * r >> PREC) // i)
    S, Cc = Fraction(s, 1 << PREC), Fraction(c, 1 << PREC)
    q = k % 4
    return [(S, Cc), (Cc, -S), (-S, -Cc), (-Cc, S)][q]


def exp_hp(y):
    """exp of an exact rational, as a Fraction within a relative 2^-300"""
    y = Fraction(y)
    t = _fixed(y)
    n = t // LN2
    r = t - n * LN2                                                   # 0 <= r < ln 2
    e, term, i = 0, 1 << PREC, 0
    while term:
        e += term
        i += 1
        term = (term * r >> PREC) // i
    return Fraction(e, 1 << PREC) * Fraction(2) ** n


def within(iv, value, eps=EPS):
    lo, hi = iv
    return (lo == -math.inf or Fraction(lo) <= value + eps) and (hi == math.inf or value - eps <= Fraction(hi))


def within_rel(iv, value, rel=EPS):
    """lo <= value <= hi up to a relative rel (an infinite endpoint is unbounded)"""
    lo, hi = iv
    return (lo == -math.inf or Fraction(lo) <= value + abs(value) * rel) and \
        (hi == math.inf or value - abs(value) * rel <= Fraction(hi))


def ulp(x):
    return math.ulp(x)


# ------------------------------------------------------------------ data
def dataset(rng, n, nulls=0.1, groups=1):
    xs = rng.integers(-5000, 5001, n)
    ws = rng.integers(-9, 10, n)
    vx = rng.random(n) >= nulls
    vw = rng.random(n) >= nulls
    sel = rng.random(n) < 0.85
    g = rng.integers(0, groups, n) if groups > 1 else None
    scope = mbk.Scope.from_arrays({"x": xs, "w": ws}, valid={"x": vx, "w": vw}, scales={"x": 2},
                                  groups=g, ngroups=groups if groups > 1 else None, rows=sel)
    return scope, xs, ws, vx, vw, sel, g


def counted(xs, ws, vx, vw, sel, g, group=0):
    out = []
    for i in range(len(xs)):
        if not sel[i] or not vx[i] or not vw[i]:
            continue
        if g is not None and g[i] != group:
            continue
        out.append((Fraction(int(xs[i]), 100), int(ws[i])))
    return out


def same_exact(a, b, groups=1):
    return all(a.exact_bounds(g) == b.exact_bounds(g) for g in range(groups))


class TestCalc003Phase(unittest.TestCase):
    def test_calc003_fixed_frequency_sine_phase_sweep_matches_direct_evaluation(self):
        rng = np.random.default_rng(SEED)
        scope, xs, ws, vx, vw, sel, g = dataset(rng, 120)
        rows = counted(xs, ws, vx, vw, sel, g)
        for omega in (Fraction(3, 7), 0.37, Fraction(-11, 2)):
            F = mc.phase(scope, [("x", omega)], weight="w")
            Fg = mc.phase(scope, [("x", omega)], weight="w", backend="cuda")
            self.assertEqual(Fg.backend, "cuda-emulated" if math_gpu.device()["emulated"] else "cuda")
            self.assertTrue(same_exact(F, Fg))                         # the device sums equal the CPU's exactly
            self.assertEqual(F.counts(), (int(sel.sum()), len(rows)))
            om = Fraction(omega)
            A = sum(w * sincos_hp(om * x)[1] for x, w in rows)
            B = sum(w * sincos_hp(om * x)[0] for x, w in rows)
            (blo, bhi), (alo, ahi) = F.exact_bounds()
            self.assertTrue(blo - EPS <= B <= bhi + EPS and alo - EPS <= A <= ahi + EPS)
            sw = sum(abs(w) for _, w in rows)
            self.assertLess(bhi - blo, Fraction(sw + 1, 2 ** 90))      # far inside the declared precision
            # the sweep: 201 phases; the declared precision 2^-40 (sum |w| + 1); numpy within its float margin
            phis = np.linspace(-10, 10, 201)
            vals = F.evaluate(phis)
            xf = np.array([float(x) for x, _ in rows]); wf = np.array([float(w) for _, w in rows])
            for phi, (lo, hi) in zip(phis, vals):
                p = Fraction(float(phi))
                s, c = sincos_hp(p)
                exact = A * s + B * c                                 # F(phi) = A sin(phi) + B cos(phi)
                self.assertTrue(within((lo, hi), exact), (omega, phi))
                self.assertLessEqual(hi - lo, (sw + 1) * 2.0 ** -40)
                direct = float(np.sum(wf * np.sin(float(om) * xf + phi)))
                margin = 2.0 ** -50 * float(np.sum(np.abs(wf) * (2 + np.abs(float(om) * xf) + abs(phi)))) + \
                    len(rows) * 2.0 ** -52 * sw
                self.assertTrue(lo - margin <= direct <= hi + margin, (omega, phi, lo, hi, direct))
            # the coefficients as a function of the frequency are exact: derivatives rotate the pair exactly
            for order in (1, 2, 3, 4, 5, -1, -2):
                D = F.derivative(0, order)
                Dg = Fg.derivative(0, order)
                self.assertEqual(D.exact_bounds(), Dg.exact_bounds())
                for phi in (-2.5, 0.0, 0.75, 3.0):
                    p = Fraction(phi)
                    exact = sum(w * sincos_hp(om * x + p + order * HALF_PI)[0]
                                for x, w in rows)
                    self.assertTrue(within(D(phi), exact, Fraction(1, 2 ** 100)), (order, phi))
            self.assertEqual(F.derivative(0, 4).exact_bounds(), F.exact_bounds())       # d^4 = identity, exactly
            self.assertEqual(F.antiderivative(0).derivative(0).exact_bounds(), F.exact_bounds())
            self.assertEqual(F.derivative(0, 2).exact_bounds(), [(-hi, -lo) for lo, hi in F.exact_bounds()])
            # definite integrals over bound pairs: sum w (cos(theta + a) - cos(theta + b))
            pairs = [(-1.0, 1.0), (0.25, 3.5), (2.0, -4.0)]
            got = F.definite(0, [a for a, _ in pairs], [b for _, b in pairs])
            for (a, b), iv in zip(pairs, got):
                exact = sum(w * (sincos_hp(om * x + Fraction(a))[1] - sincos_hp(om * x + Fraction(b))[1])
                            for x, w in rows)
                self.assertTrue(within(iv, exact), (a, b))
                self.assertTrue(within(F.integral(0, a, b)(), exact))

    def test_calc003_per_row_enclosures_are_certified_and_few_ulps_wide(self):
        rng = random.Random(SEED + 1)
        exact_doubles = 0
        for _ in range(300):
            X = rng.randint(-2 ** 52, 2 ** 52) if rng.random() < 0.3 else rng.randint(-10 ** 6, 10 ** 6)
            omega = Fraction(1, 2 ** rng.randint(0, 6)) * rng.choice([1, -1, 3])
            theta = omega * X                                         # exactly a double
            s, c = mc.sincos_scaled(omega, X)
            es, ec = sincos_hp(theta)
            self.assertTrue(within(s, es) and within(c, ec))
            if Fraction(float(theta)) == theta:                       # math.sin of the same argument, within an ulp
                exact_doubles += 1
                ts, tc = math.sin(float(theta)), math.cos(float(theta))
                self.assertTrue(s[0] <= ts + ulp(ts) and ts - ulp(ts) <= s[1], (X, omega))
                self.assertTrue(c[0] <= tc + ulp(tc) and tc - ulp(tc) <= c[1], (X, omega))
            if abs(es) > 2 ** -40:                                     # a few ulps wide
                self.assertLessEqual(s[1] - s[0], 4 * ulp(float(es)))
            if abs(ec) > 2 ** -40:
                self.assertLessEqual(c[1] - c[0], 4 * ulp(float(ec)))
            # decimals: x = X / 10^scale exactly (not a double): the exact oracle decides
            scale = rng.randint(1, 6)
            s, c = mc.sincos_scaled(omega, X, scale)
            es, ec = sincos_hp(omega * Fraction(X, 10 ** scale))
            self.assertTrue(within(s, es) and within(c, ec))
            # exponentials of exact rationals
            a = Fraction(rng.randint(-500, 500), rng.choice([1, 7, 64, 1000]))
            Y = rng.randint(-1000, 1000)
            e = mc.exp_scaled(a, Y, 2)
            ee = exp_hp(a * Fraction(Y, 100))
            self.assertTrue(within_rel(e, ee), (a, Y))
            if ee > 1e300:                                             # past the double range: [DBL_MAX, inf]
                self.assertEqual(e, (1.7976931348623157e308, math.inf))
            elif 1e-300 < ee:
                self.assertLessEqual(e[1] - e[0], 4 * ulp(float(ee)))
        self.assertGreater(exact_doubles, 200)
        # certified functions of doubles, including large arguments and the tiny path
        for x in (1e-40, -3e-12, 0.5, -1.25, 3.0, 1e6, -7.5e10, 2.0 ** 61, 123456.789):
            s, c = mc.sincos(x)
            es, ec = sincos_hp(Fraction(x))
            self.assertTrue(within(s, es) and within(c, ec), x)
        with self.assertRaises(UnsupportedError):
            mc.sincos(2.0 ** 62)
        for x in (-745.0, -20.25, -1e-20, 1e-9, 0.75, 42.0, 709.5):
            e = mc.exp(x)
            ee = exp_hp(Fraction(x))
            self.assertTrue(within_rel(e, ee), x)
            if 1e-300 < ee < 1e300:
                self.assertLessEqual(e[1] - e[0], 4 * ulp(float(ee)))
        self.assertEqual(mc.exp(800.0)[1], float("inf"))               # overflow: [DBL_MAX, inf]
        self.assertEqual(mc.exp(-1200.0)[0], 0.0)

    def test_calc003_products_over_several_phase_parameters(self):
        rng = np.random.default_rng(SEED + 2)
        n = 40
        cols = {f"x{i}": rng.integers(-300, 301, n) for i in range(3)}
        cols["w"] = rng.integers(1, 5, n)
        scope = mbk.Scope.from_arrays(cols)
        oms = [Fraction(1, 3), Fraction(-2, 5), Fraction(7, 4)]
        for k in (1, 2, 3):
            factors = [(f"x{i}", oms[i]) for i in range(k)]
            F = mc.phase(scope, factors, weight="w")
            Fg = mc.phase(scope, factors, weight="w", backend="cuda")
            self.assertTrue(same_exact(F, Fg))
            self.assertEqual(len(F.coefficients()), 2 ** k)
            data = [([Fraction(int(cols[f"x{i}"][r])) for i in range(k)], int(cols["w"][r])) for r in range(n)]
            trig = [[sincos_hp(oms[i] * x[i]) for i in range(k)] for x, _ in data]
            for S, (lo, hi) in enumerate(F.exact_bounds()):
                exact = sum(w * math.prod(t[i][1] if (S >> i) & 1 else t[i][0] for i in range(k))
                            for (x, w), t in zip(data, trig))
                self.assertTrue(lo - EPS <= exact <= hi + EPS, (k, S))
            pts = [[float(v) for v in rng.uniform(-3, 3, k)] for _ in range(5)]
            for p, iv in zip(pts, F.evaluate(pts)):
                exact = sum(w * math.prod(sincos_hp(oms[i] * x[i] + Fraction(p[i]))[0] for i in range(k))
                            for x, w in data)
                self.assertTrue(within(iv, exact), (k, p))
            if k >= 2:                                                 # a derivative in the second parameter
                D = F.derivative(1, 1)
                p = pts[0]
                exact = sum(w * math.prod(sincos_hp(oms[i] * x[i] + Fraction(p[i]) +
                                                    (HALF_PI if i == 1 else 0))[0]
                                          for i in range(k)) for x, w in data)
                self.assertTrue(within(D(*p), exact, Fraction(1, 2 ** 100)))
                I = F.integral(0, -1.0, 2.0)                           # integrate the first parameter out
                self.assertEqual(I.nparams, k - 1)
                exact = sum(w * (sincos_hp(oms[0] * x[0] - 1)[1] - sincos_hp(oms[0] * x[0] + 2)[1]) *
                            math.prod(sincos_hp(oms[i] * x[i] + Fraction(p[i]))[0] for i in range(1, k))
                            for x, w in data)
                self.assertTrue(within(I(*p[1:]), exact, Fraction(1, 2 ** 100)))
                direct = F.definite(0, [-1.0], [2.0], [p[1:]])[0]       # another contraction order: valid too
                self.assertTrue(within(direct, exact, Fraction(1, 2 ** 100)))
                self.assertTrue(direct[0] <= I(*p[1:])[1] and I(*p[1:])[0] <= direct[1])

    def test_calc003_changing_frequency_or_family_outside_basis(self):
        rng = np.random.default_rng(SEED + 3)
        scope, xs, ws, vx, vw, sel, g = dataset(rng, 60)
        rows = counted(xs, ws, vx, vw, sel, g)
        cache = mc.Retained(capacity=4)
        F = cache.phase(scope, [("x", Fraction(1, 2))], weight="w")
        self.assertEqual(cache.stats["constructions"], 1)
        again = cache.phase(scope, [("x", Fraction(1, 2))], weight="w")   # the retained state answers
        self.assertIs(again, F)
        self.assertEqual((cache.stats["hits"], cache.stats["constructions"]), (1, 1))
        self.assertTrue(F.require(scope, [("x", 0.5)], weight="w"))     # 0.5 is exactly 1/2: the same identity
        # another frequency: the retained state refuses natively; the cache builds anew, equal to a direct build
        with self.assertRaises(DomainError):
            F.require(scope, [("x", Fraction(1, 3))], weight="w")
        G = cache.phase(scope, [("x", Fraction(1, 3))], weight="w")
        self.assertIsNot(G, F)
        self.assertEqual(cache.stats["constructions"], 2)
        self.assertEqual(G.exact_bounds(), mc.phase(scope, [("x", Fraction(1, 3))], weight="w").exact_bounds())
        B = sum(w * sincos_hp(Fraction(1, 3) * x)[0] for x, w in rows)
        self.assertTrue(within(G.coefficients()[0], B))
        # another weight, column or family: new constructions too
        cache.phase(scope, [("x", Fraction(1, 2))])
        cache.exponential(scope, [("x", Fraction(1, 2))], weight="w")
        self.assertEqual(cache.stats["constructions"], 4)
        with self.assertRaises(DomainError):
            F.require(scope, [("x", Fraction(1, 2))], weight="w", family="exp")
        # outside the retained basis: a derivative in the frequency needs sums weighted by x -> explicit refusals
        with self.assertRaises(UnsupportedError):
            F.derivative("omega")
        with self.assertRaises(UnsupportedError):
            cache.phase(scope, [("x", Fraction(1, 2))], weight="w", domega=[1])
        with self.assertRaises(UnsupportedError):
            mc.phase(scope, [("x", Fraction(1, 2))], weight="w", domega=[2])
        with self.assertRaises(UnsupportedError):
            mc.exponential(scope, [("x", Fraction(1, 2))], domega=[1])
        self.assertEqual(cache.stats["refusals"], 1)                     # the cache's own refusal
        self.assertEqual(cache.stats["constructions"], 4)
        # another scope (other rows selected): a new construction
        scope2 = mbk.Scope.from_arrays({"x": xs, "w": ws}, valid={"x": vx, "w": vw}, scales={"x": 2})
        with self.assertRaises(DomainError):
            F.require(scope2, [("x", Fraction(1, 2))], weight="w")
        cache.phase(scope2, [("x", Fraction(1, 2))], weight="w")
        self.assertEqual(cache.stats["constructions"], 5)
        self.assertEqual(cache.stats["evictions"], 1)                     # capacity 4

    def test_calc003_engine_table_scope_and_staleness(self):
        import pandas as pd
        import torbles
        rng = np.random.default_rng(SEED + 4)
        n = 80
        d = [None if rng.random() < 0.1 else f"{int(v) / 100:.2f}" for v in rng.integers(-99999, 100000, n)]
        w = [int(v) for v in rng.integers(-4, 5, n)]
        frame = pd.DataFrame({"d": pd.Series(d, dtype=object), "w": w})
        db = torbles.from_ddl("CREATE TABLE t (d DECIMAL(10,2), w BIGINT)", {"t": frame})
        t = db["t"]
        scope = mbk.Scope.from_table(t, ["d", "w"])
        F = mc.phase(scope, [("d", Fraction(2, 3))], weight="w")
        Fg = mc.phase(scope, [("d", Fraction(2, 3))], weight="w", backend="cuda")
        self.assertTrue(same_exact(F, Fg))
        rows = [(Fraction(v), wv) for v, wv in zip(d, w) if v is not None]      # the exact decimals
        A = sum(wv * sincos_hp(Fraction(2, 3) * x)[1] for x, wv in rows)
        self.assertTrue(within(F.coefficients()[1], A))
        E = mc.exponential(scope, [("d", Fraction(1, 300))], weight="w")
        Ee = sum(wv * exp_hp(Fraction(1, 300) * x) for x, wv in rows)
        lo, hi = E.exact_bounds()[0]
        self.assertTrue(lo - EPS <= Ee <= hi + EPS)
        # a mutated table: the scope is stale, a build and the retained cache refuse it
        t.flip(0, 0)                                                   # one stored bit: a new revision
        with self.assertRaises(StaleStateError):
            mc.phase(scope, [("d", Fraction(2, 3))], weight="w")
        with self.assertRaises(StaleStateError):
            mc.Retained().phase(scope, [("d", Fraction(2, 3))], weight="w")


class TestCalc003Exponential(unittest.TestCase):
    def test_calc003_separable_exponentials(self):
        rng = np.random.default_rng(SEED + 5)
        scope, xs, ws, vx, vw, sel, g = dataset(rng, 100)
        rows = counted(xs, ws, vx, vw, sel, g)
        for a in (Fraction(-1, 40), Fraction(3, 1000), 0.0625, Fraction(1, 5)):  # e^(a x) up to e^(10) .. e^(1000)
            G = mc.exponential(scope, [("x", a)], weight="w")
            Gg = mc.exponential(scope, [("x", a)], weight="w", backend="cuda")
            self.assertTrue(same_exact(G, Gg))
            E = sum(w * exp_hp(Fraction(a) * x) for x, w in rows)
            lo, hi = G.exact_bounds()[0]
            self.assertTrue(lo - EPS * abs(E) <= E <= hi + EPS * abs(E), a)
            sw = sum(abs(w) * exp_hp(Fraction(a) * x) for x, w in rows)
            self.assertLess(hi - lo, sw * Fraction(1, 2 ** 90) + Fraction(len(rows), 2 ** G.info["scale"]))
            for theta in (-2.0, 0.0, 1.5):
                iv = G(theta)
                exact = E * exp_hp(Fraction(theta))
                tol = abs(exact) * Fraction(1, 2 ** 60) + EPS
                self.assertTrue(Fraction(iv[0]) <= exact + tol and exact - tol <= Fraction(iv[1]), (a, theta))
            D = G.derivative(0, 3)                                     # d/dtheta exp = exp: unchanged
            self.assertEqual(D.exact_bounds(), G.exact_bounds())
            self.assertEqual(D.info["order"], [3])
            I = G.definite(0, [0.0], [1.0])[0]
            exact = E * (exp_hp(Fraction(1)) - 1)
            tol = abs(exact) * Fraction(1, 2 ** 60) + EPS
            self.assertTrue(Fraction(I[0]) <= exact + tol and exact - tol <= Fraction(I[1]))
        # two rates: G(t1, t2) = E exp(t1 + t2), E = sum w exp(a1 x + a2 w)
        G2 = mc.exponential(scope, [("x", Fraction(-1, 1000)), ("w", Fraction(1, 2))], weight="w")
        G2g = mc.exponential(scope, [("x", Fraction(-1, 1000)), ("w", Fraction(1, 2))], weight="w", backend="cuda")
        self.assertTrue(same_exact(G2, G2g))
        E2 = sum(w * exp_hp(Fraction(-1, 1000) * x + Fraction(w, 2)) for x, w in rows)
        lo, hi = G2.exact_bounds()[0]
        self.assertTrue(lo - EPS <= E2 <= hi + EPS)
        # huge exponents (e^(20 x) up to e^1000, past the double range): the scale goes negative, the exact bounds
        # stay rigorous, the double enclosure is [DBL_MAX, inf] or a finite one
        Gh = mc.exponential(scope, [("x", 20)], weight="w")
        Eh = sum(w * exp_hp(20 * x) for x, w in rows)
        lo, hi = Gh.exact_bounds()[0]
        self.assertLess(Gh.info["scale"], 0)
        self.assertTrue(lo <= Eh + abs(Eh) * EPS and Eh - abs(Eh) * EPS <= hi)
        self.assertLess(hi - lo, abs(Eh) * Fraction(1, 2 ** 90) + Fraction(len(rows), 1) * Fraction(2) ** -Gh.info["scale"])
        self.assertTrue(within_rel(Gh.coefficients()[0], Eh))
        self.assertTrue(same_exact(Gh, mc.exponential(scope, [("x", 20)], weight="w", backend="cuda")))


class TestCalc003Groups(unittest.TestCase):
    def test_calc003_groups_nulls_and_zero_families(self):
        rng = np.random.default_rng(SEED + 6)
        n, G = 150, 5
        scope, xs, ws, vx, vw, sel, g = dataset(rng, n, nulls=0.2, groups=G)
        F = mc.phase(scope, [("x", Fraction(5, 9))], weight="w")
        Fg = mc.phase(scope, [("x", Fraction(5, 9))], weight="w", backend="cuda")
        self.assertTrue(same_exact(F, Fg, G))
        for grp in range(G):
            rows = counted(xs, ws, vx, vw, sel, g, grp)
            self.assertEqual(F.counts(grp), (int(sum(1 for i in range(n) if sel[i] and g[i] == grp)), len(rows)))
            self.assertEqual(Fg.counts(grp), F.counts(grp))
            A = sum(w * sincos_hp(Fraction(5, 9) * x)[1] for x, w in rows)
            self.assertTrue(within(F.coefficients(grp)[1], A))
        # an all-NULL group and an empty selection: the zero family, exactly [0, 0]
        xs2 = np.arange(10)
        valid = np.array([True] * 5 + [False] * 5)
        gid = np.array([0] * 5 + [1] * 5)
        s2 = mbk.Scope.from_arrays({"x": xs2}, valid={"x": valid}, groups=gid, ngroups=3)
        Z = mc.phase(s2, [("x", Fraction(1, 2))])
        self.assertEqual(Z.counts(1), (5, 0))
        self.assertEqual(Z.counts(2), (0, 0))                          # an absent group
        for grp in (1, 2):
            self.assertEqual(Z.exact_bounds(grp), [(0, 0), (0, 0)])
            self.assertEqual(Z.evaluate([0.3, 1.7], grp), [(0.0, 0.0), (0.0, 0.0)])
        self.assertTrue(same_exact(Z, mc.phase(s2, [("x", Fraction(1, 2))], backend="cuda"), 3))
        empty = mbk.Scope.from_arrays({"x": xs2}, rows=np.zeros(10, dtype=bool))
        Ze = mc.exponential(empty, [("x", Fraction(1, 2))])
        self.assertEqual((Ze.counts(), Ze.exact_bounds(), Ze(1.0)), ((0, 0), [(0, 0)], (0.0, 0.0)))
        self.assertTrue(same_exact(Ze, mc.exponential(empty, [("x", Fraction(1, 2))], backend="cuda")))
        # a zero frequency: sin(0 + phi) exactly, the coefficients exact integers
        F0 = mc.phase(s2, [("x", 0)])
        self.assertEqual(F0.exact_bounds(0), [(0, 0), (5, 5)])            # B = 0, A = count

    def test_calc003_cuda_equals_cpu_on_random_families(self):
        rng = np.random.default_rng(SEED + 7)
        for trial in range(12):
            n = int(rng.integers(1, 300))
            k = int(rng.integers(1, 4))
            groups = int(rng.integers(1, 4))
            cols = {f"x{i}": rng.integers(-10 ** 6, 10 ** 6, n) for i in range(k)}
            cols["w"] = rng.integers(-1000, 1000, n)
            valid = {f"x{i}": rng.random(n) > 0.05 for i in range(k)}
            gid = rng.integers(0, groups, n) if groups > 1 else None
            scope = mbk.Scope.from_arrays(cols, valid=valid, groups=gid, ngroups=groups if groups > 1 else None,
                                          scales={f"x{i}": int(rng.integers(0, 4)) for i in range(k)},
                                          rows=rng.random(n) < 0.9)
            factors = [(f"x{i}", Fraction(int(rng.integers(-50, 50)), int(rng.integers(1, 30)))) for i in range(k)]
            for fam in ("phase", "exp"):
                if fam == "exp":
                    factors = [(c, w / 10 ** 6) for c, w in factors]
                    build = mc.exponential
                else:
                    build = mc.phase
                a = build(scope, factors, weight="w")
                b = build(scope, factors, weight="w", backend="cuda")
                self.assertTrue(same_exact(a, b, groups), (trial, fam))
                self.assertEqual([a.counts(i) for i in range(groups)], [b.counts(i) for i in range(groups)])


class TestCalc003Budgets(unittest.TestCase):
    def test_calc003_budgets_at_the_limit_and_one_past(self):
        rng = np.random.default_rng(SEED + 8)
        n = 200
        cols = {f"x{i}": rng.integers(-100, 100, n) for i in range(4)}
        scope = mbk.Scope.from_arrays(cols)
        f3 = [(f"x{i}", Fraction(1, i + 2)) for i in range(3)]
        # support: 2^k coefficients
        self.assertEqual(len(mc.phase(scope, f3, budget=Budget(support=8)).coefficients()), 8)
        with self.assertRaises(AlgebraBudgetError) as e:
            mc.phase(scope, f3, budget=Budget(support=7))
        self.assertEqual((e.exception.resource, e.exception.need, e.exception.limit), ("support", 8, 7))
        # variables: k parameters
        mc.phase(scope, f3, budget=Budget(variables=3))
        with self.assertRaises(AlgebraBudgetError) as e:
            mc.phase(scope, f3, budget=Budget(variables=2))
        self.assertEqual((e.exception.resource, e.exception.need), ("variables", 3))
        # tensor: selected rows x coefficients (the per-row expansion)
        mc.phase(scope, f3, budget=Budget(tensor=n * 8))
        with self.assertRaises(AlgebraBudgetError) as e:
            mc.phase(scope, f3, budget=Budget(tensor=n * 8 - 1))
        self.assertEqual((e.exception.resource, e.exception.need), ("tensor", n * 8))
        # degree: the order of a derivative request
        F = mc.phase(scope, f3[:1])
        F.derivative(0, 4, budget=Budget(degree=4))
        F.derivative(0, -4, budget=Budget(degree=4))
        with self.assertRaises(AlgebraBudgetError) as e:
            F.derivative(0, 5, budget=Budget(degree=4))
        self.assertEqual((e.exception.resource, e.exception.need), ("degree", 5))
        # output: evaluation points
        self.assertEqual(len(F.evaluate(np.zeros(100), budget=Budget(output=100))), 100)
        with self.assertRaises(AlgebraBudgetError) as e:
            F.evaluate(np.zeros(101), budget=Budget(output=100))
        self.assertEqual((e.exception.resource, e.exception.need), ("output", 101))
        # the CUDA build budgets its device copies: exactly the need passes, one byte less is refused
        need = mc.device_bytes(scope, "phase", f3)
        self.assertTrue(same_exact(mc.phase(scope, f3, backend="cuda", budget=Budget(bytes=need)), mc.phase(scope, f3)))
        with self.assertRaises(AlgebraBudgetError) as e:
            mc.phase(scope, f3, backend="cuda", budget=Budget(bytes=need - 1))
        self.assertEqual((e.exception.resource, e.exception.need), ("bytes", need))
        # representation limits: 16 factors at most; a frequency with |K| >= 2^64; an exponent past 2^24
        many = {f"y{i}": np.arange(4) for i in range(17)}
        s17 = mbk.Scope.from_arrays(many)
        with self.assertRaises(AlgebraBudgetError) as e:                # 16 factors: 2^16 coefficients
            mc.phase(s17, [(f"y{i}", 1) for i in range(16)], budget=Budget(support=2 ** 16 - 1))
        self.assertEqual((e.exception.resource, e.exception.need), ("support", 2 ** 16))
        with self.assertRaises(UnsupportedError):                       # 17 factors: outside the declared scope
            mc.phase(s17, [(f"y{i}", 1) for i in range(17)])
        self.assertEqual(len(mc.exponential(s17, [(f"y{i}", 1) for i in range(16)]).coefficients()), 1)
        with self.assertRaises(UnsupportedError):
            mc.phase(scope, [("x0", 2 ** 70)])
        big = mbk.Scope.from_arrays({"x": np.array([10 ** 6])})
        with self.assertRaises(OverflowError):
            mc.exponential(big, [("x", 2 ** 30)])
        mc.exponential(big, [("x", Fraction(1, 1000))])

    def test_calc003_abi_registry_covers_the_calc_structs(self):
        from torbles import math_handles as mh
        report = mh.abi_check()
        checked = {(r["native"], r["library"]) for r in report["checked"]}
        for name in ("tr_calc_spec", "tr_calc_plan_view", "tr_calc_info"):
            self.assertIn((name, "libalg"), checked)
        self.assertIn(("tr_calc_plan_view", "cuda"), checked)


if __name__ == "__main__":
    unittest.main()
