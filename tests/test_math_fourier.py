# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""SPEC-001: fixed spectral basis states (torbles/math_fourier.c, math_fourier_gpu.cu, math_fourier.py).

Oracles, independent of the implementation and written here: sin and cos of exact angles in 400-bit integer fixed
point (pi by Machin's formula recomputed here, Taylor series in Python integers; error below 2^-300), applied to the
exact data (logical integers / 10^scale, exact frequencies) and the exact values of the double shifts and phases, so
every exact bound and every enclosure is checked to contain the exact value within 2^-150 (Fraction comparisons);
NumPy / cmath float64 direct evaluation (rescanning the rows) for the declared numerical claim: it agrees with the
enclosures within the stated float tolerance TOL (relative to the sum of the magnitudes of the terms). For the
Boolean case: SPEC-002's exact Walsh transform (math_spec.walsh) of the LATTICE-001 histogram of the bits and of a
NumPy bincount of the weighted bits, and SPEC-002's inverse; CALC-003's phase family for the univariate radian case.
Every native build is also run on the CUDA path (emulated here, scrambled thread order: TORBLES_EMU_SCRAMBLE=1) and
its exact sums compared for equality. Budgets are tested at the limit and one past it. Seeds are fixed.
"""
import cmath
import math
import os
import unittest
from fractions import Fraction

import numpy as np

os.environ.setdefault("TORBLES_EMU_SCRAMBLE", "1")

from torbles import math_bank as mbk, math_fourier as mf
from torbles.math_budgets import Budget
from torbles.math_domains import DomainError
from torbles.math_errors import StaleStateError, UnsupportedError
from torbles.mathematics import AlgebraBudgetError

SEED = 20261101
PREC = 400
EPS = Fraction(1, 2 ** 150)
TOL = 1e-13                       # float64 direct evaluation: |numpy - exact| <= TOL * sum_r |w_r| (1 + |theta_r|)


# ------------------------------------------------------------------ the oracle: sin, cos of exact angles
def _atan_inv(x, prec):
    one = 1 << (prec + 32)
    s, t, k, x2 = 0, one // x, 0, x * x
    while t:
        s += t // (2 * k + 1) if k % 2 == 0 else -(t // (2 * k + 1))
        t //= x2
        k += 1
    return s >> 32


PI = 4 * (4 * _atan_inv(5, PREC) - _atan_inv(239, PREC))          # pi * 2^PREC (error a few units)
UNIT = {"radian": 1 << PREC, "pi": PI, "turn": 2 * PI}              # c * 2^PREC


def theta_fixed(cq, rq, unit):
    """(c cq + rq) 2^PREC for exact rationals cq, rq"""
    cq, rq = Fraction(cq), Fraction(rq)
    return (UNIT[unit] * cq.numerator) // cq.denominator + (rq.numerator << PREC) // rq.denominator


def sincos_fx(t):
    """(sin, cos) of t / 2^PREC as Fractions within 2^-300"""
    half = PI >> 1
    k = (2 * t + half) // (2 * half)                               # the nearest multiple of pi/2
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
    return [(S, Cc), (Cc, -S), (-S, -Cc), (-Cc, S)][k % 4]


def unit_power_q(unit, p):
    """c^p as a Fraction within 2^-300 (exact for radians)"""
    return Fraction(UNIT[unit] ** p, 1 << (PREC * p))


def feature_oracle(rows, omega, unit, tau=None, phi=0, order=None, phase_order=0):
    """the exact action i^(|a| + b) (c omega)^a exp(i (c omega.(x + tau) + phi)) sum w: (re, im) Fractions"""
    d = len(omega)
    tau = [Fraction(t) for t in (tau or [0] * d)]
    order = order or [0] * d
    re = im = Fraction(0)
    for x, w in rows:
        cq = sum(Fraction(o) * (Fraction(xi) + ti) for o, xi, ti in zip(omega, x, tau))
        s, c = sincos_fx(theta_fixed(cq, Fraction(phi), unit))
        re += w * c
        im += w * s
    m = (sum(order) + phase_order) % 4
    re, im = [(re, im), (-im, re), (-re, -im), (im, -re)][m]
    factor = Fraction(1)
    for o, a in zip(omega, order):
        factor *= Fraction(o) ** a
    factor *= unit_power_q(unit, sum(order))
    return re * factor, im * factor


def numpy_action(xs, ws, omega, unit, tau=None, phi=0.0, order=None, phase_order=0):
    """float64 direct evaluation (rescans the rows) and the sum of the magnitudes of its terms"""
    c = {"radian": 1.0, "pi": math.pi, "turn": 2 * math.pi}[unit]
    d = len(omega)
    tau = tau or [0.0] * d
    order = order or [0] * d
    th = np.zeros(len(ws))
    for j in range(d):
        th += c * float(omega[j]) * (xs[j] + tau[j])
    th += phi
    factor = (1j) ** (sum(order) + phase_order)
    for j in range(d):
        factor *= (1j * 0 + c * float(omega[j])) ** order[j]
    terms = ws * np.exp(1j * th)
    return complex(np.sum(terms) * factor), float(np.sum(np.abs(ws) * (1 + np.abs(th)))) * abs(factor)


def contains(iv, value, eps=EPS):
    lo, hi = iv
    return Fraction(lo) <= value + eps and value - eps <= Fraction(hi)


def cinterval_contains(z, re, im, eps=EPS):
    return contains(z[0], re, eps) and contains(z[1], im, eps)


def near(z, value, scale):
    """the enclosure's midpoint agrees with a float64 value within the stated tolerance (float64 loses about
    |theta| ulps per term in the argument, so the scale is sum_r |w_r| (1 + |theta_r|))"""
    (a, b), (c, d) = z
    tol = TOL * max(scale, 1.0) + (b - a) + (d - c)
    return abs((a + b) / 2 - value.real) <= tol and abs((c + d) / 2 - value.imag) <= tol


# ------------------------------------------------------------------ data
def dataset(rng, n, d, groups=1, nulls=0.1, scale=2, weights=True, wscale=0):
    cols, valid, scales = {}, {}, {}
    for j in range(d):
        cols[f"x{j}"] = rng.integers(-5000, 5001, n)
        valid[f"x{j}"] = rng.random(n) >= nulls
        scales[f"x{j}"] = scale if j % 2 == 0 else 0
    if weights:
        cols["w"] = rng.integers(-90, 91, n)
        scales["w"] = wscale
    g = rng.integers(0, groups, n) if groups > 1 else None
    scope = mbk.Scope.from_arrays(cols, valid=valid, scales=scales, groups=g, ngroups=groups if groups > 1 else None)
    return scope, cols, valid, scales, g


def rows_of(cols, valid, scales, g, d, group=0, weight=True, wscale=0):
    """the counted rows of a group: (exact x vector, exact weight)"""
    n = len(cols["x0"])
    out = []
    for r in range(n):
        if g is not None and g[r] != group:
            continue
        if not all(valid[f"x{j}"][r] for j in range(d)):
            continue
        x = tuple(Fraction(int(cols[f"x{j}"][r]), 10 ** scales[f"x{j}"]) for j in range(d))
        w = Fraction(int(cols["w"][r]), 10 ** wscale) if weight else Fraction(1)
        out.append((x, w))
    return out


def float_columns(rows, d):
    xs = [np.array([float(x[j]) for x, _ in rows]) for j in range(d)]
    ws = np.array([float(w) for _, w in rows])
    return xs, ws


def frequency_set(rng, m, d, unit):
    seen, out = set(), []
    while len(out) < m:
        if unit == "radian":
            f = tuple(Fraction(int(rng.integers(-40, 41)), int(rng.integers(1, 30))) for _ in range(d))
        else:
            f = tuple(Fraction(int(rng.integers(-12, 13)), int(rng.choice([1, 2, 3, 4, 8, 7]))) for _ in range(d))
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


class Spec001(unittest.TestCase):
    def build(self, scope, columns, freqs, **kw):
        B = mf.basis(scope, columns, freqs, **kw)
        G = mf.basis(scope, columns, freqs, backend="cuda", **kw)
        self.assertIn(G.backend, ("cuda", "cuda-emulated"))
        for g in range(scope.groups):
            self.assertEqual(B.bounds(g), G.bounds(g))                # bit for bit (exact sums)
            self.assertEqual(B.counts(g), G.counts(g))
            self.assertEqual(B.weight(g), G.weight(g))
        return B

    # ---------------------------------------------------------- acceptance 1: features match direct evaluation
    def test_spec001_features_contain_the_exact_values_and_match_direct_evaluation(self):
        rng = np.random.default_rng(SEED)
        for d, m, unit, groups, wscale in ((1, 1, "radian", 1, 0), (1, 8, "pi", 3, 2), (2, 8, "radian", 2, 0),
                                           (3, 32, "turn", 1, 1), (2, 32, "radian", 1, 0)):
            n = 60 if m == 32 else 120
            scope, cols, valid, scales, g = dataset(rng, n, d, groups=groups, wscale=wscale)
            columns = [f"x{j}" for j in range(d)]
            freqs = frequency_set(rng, m, d, unit)
            B = self.build(scope, columns, freqs, weight="w", unit=unit)
            self.assertEqual(B.frequencies, tuple(freqs))
            info = B.info
            self.assertEqual((info["frequencies"], info["dimensions"], info["unit"]), (m, d, unit))
            for grp in range(groups):
                rows = rows_of(cols, valid, scales, g, d, grp, wscale=wscale)
                sel = int(np.sum(g == grp)) if g is not None else n
                self.assertEqual(B.counts(grp), (sel, len(rows)))
                self.assertEqual(B.weight(grp), sum((w for _, w in rows), Fraction(0)))
                xs, ws = float_columns(rows, d)
                bounds, coef = B.bounds(grp), B.coefficients(grp)
                for f, omega in enumerate(freqs):
                    re, im = feature_oracle(rows, omega, unit)
                    (rl, rh), (il, ih) = bounds[f]
                    self.assertTrue(rl - EPS <= re <= rh + EPS and il - EPS <= im <= ih + EPS, (d, unit, omega))
                    # the double enclosure contains the exact bounds; the width is the contract's
                    self.assertTrue(Fraction(coef[f][0][0]) <= rl and rh <= Fraction(coef[f][0][1]))
                    self.assertTrue(Fraction(coef[f][1][0]) <= il and ih <= Fraction(coef[f][1][1]))
                    wmax = max((abs(w) for _, w in rows), default=Fraction(0))
                    bound = len(rows) * (wmax * Fraction(1, 2 ** 108) + Fraction(2, 2 ** info["scale"]))
                    self.assertLessEqual(rh - rl, bound)
                    z, mag = numpy_action(xs, ws, omega, unit)
                    self.assertTrue(near(coef[f], z, mag), (omega, coef[f], z))

    def test_spec001_univariate_radians_equal_calc003_bit_for_bit(self):
        from torbles import math_calc as mc
        rng = np.random.default_rng(SEED + 1)
        scope, cols, valid, scales, g = dataset(rng, 150, 1)
        for omega in (Fraction(3, 7), Fraction(-5, 2), 2.5, 1, Fraction(-1, 1000)):
            F = mc.phase(scope, [("x0", omega)], weight="w")
            B = mf.basis(scope, ["x0"], [omega], weight="w")
            (re, im), = B.bounds()
            eb = F.exact_bounds()
            self.assertEqual((re, im), (eb[1], eb[0]))                # CALC-003: C_1 = sum w cos, C_0 = sum w sin
            self.assertEqual(B.info["scale"], F.info["scale"])

    # ---------------------------------------------------------- acceptance 1: phase transforms
    def test_spec001_phase_actions_match_direct_evaluation_without_rescanning(self):
        rng = np.random.default_rng(SEED + 2)
        for d, unit in ((1, "radian"), (2, "pi"), (2, "radian"), (3, "turn")):
            scope, cols, valid, scales, g = dataset(rng, 50, d)
            columns = [f"x{j}" for j in range(d)]
            freqs = frequency_set(rng, 6, d, unit)
            B = self.build(scope, columns, freqs, weight="w", unit=unit)
            rows = rows_of(cols, valid, scales, g, d)
            xs, ws = float_columns(rows, d)
            requests = [dict(),
                        dict(shift=[0.5] * d),
                        dict(shift=[float(v) for v in rng.normal(0, 3, d)], phase=0.75),
                        dict(order=[1] + [0] * (d - 1)),
                        dict(shift=[-1.25] * d, order=[2] + [1] * (d - 1), phase_order=1, phase=-0.3),
                        dict(phase_order=3, phase=2.0)]
            scope.close()                                          # the actions read only the basis
            pick = [freqs[1], freqs[4], freqs[0]]
            for normalize in (False, True):
                got = B.act(requests, frequencies=pick, normalize=normalize)
                W = B.weight()
                for q, r in enumerate(requests):
                    for i, omega in enumerate(pick):
                        re, im = feature_oracle(rows, omega, unit, r.get("shift"), Fraction(r.get("phase", 0.0)),
                                                r.get("order"), r.get("phase_order", 0))
                        if normalize:
                            re, im = re / W, im / W
                        self.assertTrue(cinterval_contains(got[q][i], re, im), (unit, r, omega))
                        z, mag = numpy_action(xs, ws, omega, unit, r.get("shift"), r.get("phase", 0.0),
                                              r.get("order"), r.get("phase_order", 0))
                        if normalize:
                            z, mag = z / float(W), mag / abs(float(W))
                        self.assertTrue(near(got[q][i], z, mag), (r, omega, got[q][i], z))
            # shift and characteristic are actions; one action per request, the order of the selection kept
            self.assertEqual(B.shift([0.5] * d, frequencies=pick), B.act(dict(shift=[0.5] * d), frequencies=pick))
            self.assertEqual(B.characteristic(frequencies=pick), B.act(dict(), frequencies=pick, normalize=True))
            arr = B.act(requests, arrays=True)
            self.assertEqual(arr.shape, (len(requests), len(freqs), 4))

    def test_spec001_derivatives_are_exact_transforms(self):
        rng = np.random.default_rng(SEED + 3)
        for d, unit in ((1, "radian"), (2, "radian"), (2, "pi")):
            scope, cols, valid, scales, g = dataset(rng, 40, d)
            columns = [f"x{j}" for j in range(d)]
            freqs = frequency_set(rng, 4, d, unit)
            B = self.build(scope, columns, freqs, weight="w", unit=unit)
            rows = rows_of(cols, valid, scales, g, d)
            for alpha, beta in (([1] + [0] * (d - 1), 0), ([2] * d, 1), ([0] * d, 3), ([3] + [1] * (d - 1), 2)):
                D = B.derivative(alpha, beta)
                DD = D.derivative([1] * d)                           # derivatives compose
                self.assertEqual(D.info["order"], alpha)
                self.assertEqual(DD.info["order"], [a + 1 for a in alpha])
                for f, omega in enumerate(freqs):
                    (rl, rh), (il, ih) = B.bounds()[f]
                    m = (sum(alpha) + beta) % 4
                    # the oracle of the transform: rotate the exact intervals by i^m, scale by omega^alpha exactly
                    rot = [((rl, rh), (il, ih)), ((-ih, -il), (rl, rh)), ((-rh, -rl), (-ih, -il)),
                           ((il, ih), (-rh, -rl))][m]
                    fac = Fraction(1)
                    for o, a in zip(omega, alpha):
                        fac *= o ** a
                    want = [tuple(sorted((fac * a, fac * b))) for a, b in rot]
                    ex = D.exact(0, f)
                    self.assertEqual((ex["re"], ex["im"]), tuple(want), (unit, alpha, beta, omega))
                    self.assertEqual(ex["unit_power"], 0 if unit == "radian" else sum(alpha))
                    re, im = feature_oracle(rows, omega, unit, order=alpha, phase_order=beta)
                    self.assertTrue(cinterval_contains(D.coefficients()[f], re, im))
                    up = unit_power_q(unit, sum(alpha))
                    self.assertTrue(ex["re"][0] * up - EPS <= re <= ex["re"][1] * up + EPS or
                                    ex["re"][1] * up - EPS <= re <= ex["re"][0] * up + EPS)
                    # an action on the derivative state equals the derivative of the action
                    a1 = D.act(dict(shift=[0.25] * d), frequencies=[omega])[0]
                    re2, im2 = feature_oracle(rows, omega, unit, [0.25] * d, 0, alpha, beta)
                    self.assertTrue(cinterval_contains(a1, re2, im2))
            with self.assertRaises(ValueError):
                B.derivative([-1] + [0] * (d - 1))
            with self.assertRaises(UnsupportedError):
                B.derivative([65] + [0] * (d - 1))

    # ---------------------------------------------------------- acceptance 2: identity and missing support
    def test_spec001_new_frequencies_request_a_new_basis_and_missing_support_is_not_zero(self):
        rng = np.random.default_rng(SEED + 4)
        scope, cols, valid, scales, g = dataset(rng, 80, 2)
        cache = mf.Retained()
        f1 = [(1, 0), (Fraction(1, 2), 3), (0, Fraction(-1, 4))]
        A = cache.basis(scope, ["x0", "x1"], f1, weight="w")
        self.assertEqual(cache.stats["constructions"], 1)
        self.assertIs(cache.basis(scope, ["x0", "x1"], f1, weight="w"), A)               # the same set: a hit
        self.assertIs(cache.basis(scope, ["x0", "x1"], [f1[2], f1[0]], weight="w"), A)   # a subset: a hit
        self.assertEqual(cache.stats["hits"], 2)
        B = cache.basis(scope, ["x0", "x1"], [(1, 0), (2, 0)], weight="w")              # a new frequency
        self.assertIsNot(B, A)
        self.assertEqual(cache.stats["constructions"], 2)
        self.assertEqual(B.frequencies, ((1, 0), (2, 0)))                 # the new state's identity: its set
        for other in (dict(weight=None), dict(weight="w", unit="pi")):
            cache.basis(scope, ["x0", "x1"], [(1, 0)], **other)            # another weight or unit: new states
        cache.basis(scope, ["x1", "x0"], [(0, 1)], weight="w")             # other columns: a new state
        self.assertEqual(cache.stats["constructions"], 5)
        self.assertTrue(A.require(scope, ["x0", "x1"], [f1[1]], weight="w"))
        for args in (([(2, 0)], "w", None), ([f1[0]], None, None), ([f1[0]], "w", "turn")):
            with self.assertRaises(DomainError):
                A.require(scope, ["x0", "x1"], args[0], weight=args[1], unit=args[2])
        scope2, *_ = dataset(np.random.default_rng(SEED + 99), 80, 2)
        with self.assertRaises(DomainError):
            A.require(scope2, ["x0", "x1"], [f1[0]], weight="w")            # another scope
        # missing spectral support is refused, never treated as zero
        for bad in ((2, 0), (1, 1), (0, 0)):
            with self.assertRaises(DomainError):
                A.feature(bad)
            with self.assertRaises(DomainError):
                A.index([f1[0], bad])
            with self.assertRaises(DomainError):
                A.act(dict(shift=(1.0, 0.0)), frequencies=[bad])
            with self.assertRaises(DomainError):
                A.characteristic(frequencies=[bad])
        self.assertEqual(A.index([f1[2], f1[0]]), [2, 0])
        self.assertEqual(A.feature(f1[1]), A.coefficients()[1])
        D = A.derivative((1, 0))
        with self.assertRaises(DomainError):
            D.require(scope, ["x0", "x1"], [f1[0]], weight="w")             # a derivative state is no basis
        # the zero frequency is exact: the weight sum
        Z = mf.basis(scope, ["x0", "x1"], [(0, 0), (1, 1)], weight="w")
        W = Z.weight()
        self.assertEqual(Z.bounds()[0], ((W, W), (0, 0)))

    # ---------------------------------------------------------- the Boolean case: exact, SPEC-002
    def test_spec001_walsh_characters_are_the_exact_special_case_of_spec002(self):
        from torbles import math_bool as mb, math_lattice as ml, math_spec as ms
        rng = np.random.default_rng(SEED + 5)
        for k in (1, 2, 3, 5, 7):
            n = 400
            bits = rng.integers(0, 2, (n, k))
            cols = {f"b{j}": bits[:, j] for j in range(k)}
            cols["w"] = rng.integers(-3, 9, n)
            valid = {"b0": rng.random(n) > 0.05}
            scope = mbk.Scope.from_arrays(cols, valid=valid)
            names = [f"b{j}" for j in range(k)]
            # unweighted: the LATTICE-001 histogram of the bit conditions and SPEC-002's transform
            Wb = mf.walsh_basis(scope, names)
            self.assertTrue(Wb.info["exact"])
            masks = mb.Masks.from_scope(scope, [mb.condition(c, "bit", 0) for c in names])
            h = ml.Histogram.build(masks)
            f = ms.walsh(h)
            got = [b[0][0] for b in Wb.bounds()]
            self.assertEqual(got, [Fraction(v) for v in f["spectrum"]])
            self.assertTrue(all(b[0][0] == b[0][1] and b[1] == (0, 0) for b in Wb.bounds()))
            self.assertEqual(Wb.walsh_inverse()["values"], [Fraction(int(v)) for v in h.counts()])
            # weighted (CPU and CUDA): a NumPy bincount of the complete rows and SPEC-002 forward and inverse
            Ww = self.build(scope, names, mf.walsh_frequencies(k), weight="w", unit="pi")
            ok = valid["b0"]
            idx = (bits[ok] * (1 << np.arange(k))).sum(1)
            hist = [int(v) for v in np.bincount(idx, weights=cols["w"][ok], minlength=1 << k).round().astype(np.int64)]
            spec = ms.walsh(hist)["spectrum"]
            self.assertEqual([b[0][0] for b in Ww.bounds()], [Fraction(v) for v in spec])
            inv = Ww.walsh_inverse()
            self.assertEqual(inv["values"], [Fraction(v) for v in hist])
            self.assertEqual(inv["values"], ms.inverse(spec)["values"])
            self.assertEqual(inv["cost"]["states"], 1 << k)                  # the full domain, whatever the rows
            self.assertEqual((inv["cost"]["passes"], inv["cost"]["butterflies"]), (k, k * (1 << (k - 1))))
            self.assertEqual(inv["normalization"], ms.INVERSE)
            # the same characters in turns (S / 2); an integer translation is an exact sign (-1)^{S.tau}
            Wt = mf.basis(scope, names, mf.walsh_frequencies(k, "turn"), weight="w", unit="turn")
            self.assertEqual(Wt.bounds(), Ww.bounds())
            tau = [int(t) for t in rng.integers(-3, 4, k)]
            shifted = Ww.shift([float(t) for t in tau])
            for S, z in enumerate(shifted):
                sign = -1 if sum(t for j, t in enumerate(tau) if (S >> j) & 1) % 2 else 1
                self.assertEqual(z, ((sign * spec[S], sign * spec[S]), (0.0, 0.0)))
            N = Ww.weight()
            if N:
                chi = Ww.characteristic()
                for S, z in enumerate(chi):
                    self.assertTrue(contains(z[0], Fraction(spec[S]) / N, 0) and z[1] == (0.0, 0.0))
            # missing spectral support: the inverse is refused, never computed with a zero
            P = mf.basis(scope, names, mf.walsh_frequencies(k)[1:], weight="w", unit="pi")
            self.assertEqual(P.walsh_support(), dict(k=k, present=(1 << k) - 1, states=1 << k))
            with self.assertRaises(DomainError):
                P.walsh_inverse()
            with self.assertRaises(DomainError):
                Ww.derivative([1] + [0] * (k - 1)).walsh_inverse()
        # data outside {0, 1}: not the Boolean domain
        scope = mbk.Scope.from_arrays({"a": np.array([0, 1, 2, 1]), "b": np.array([1, 0, 1, 1])})
        with self.assertRaises(DomainError):
            mf.walsh_basis(scope, ["a", "b"]).walsh_inverse()
        with self.assertRaises(DomainError):                                # radians: not the Boolean kind
            mf.basis(scope, ["a"], [0, 1]).walsh_inverse()

    # ---------------------------------------------------------- semantics: NULLs, groups, tables
    def test_spec001_null_validity_bag_multiplicity_and_group_presence_are_separate(self):
        xs = np.array([100, -250, 300, 7, 0, 55])
        ws = np.array([2, 3, -1, 4, 5, 6])
        vx = np.array([True, True, False, True, True, True])
        g = np.array([0, 0, 0, 2, 2, 2])
        rows = np.array([True, True, True, True, False, True])
        scope = mbk.Scope.from_arrays({"x": xs, "w": ws}, valid={"x": vx}, scales={"x": 2}, groups=g, ngroups=4,
                                      rows=rows)
        B = self.build(scope, ["x"], [Fraction(1, 3), 0], weight="w")
        self.assertEqual([B.counts(i) for i in range(4)], [(3, 2), (0, 0), (2, 2), (0, 0)])
        self.assertEqual(B.weight(0), 5)
        self.assertEqual(B.bounds(1), [((0, 0), (0, 0))] * 2)               # an absent group: exactly zero
        self.assertEqual(B.bounds(3), [((0, 0), (0, 0))] * 2)
        with self.assertRaises(DomainError):
            B.characteristic(group=1)                                      # no rows: undefined, not zero
        allnull = mbk.Scope.from_arrays({"x": xs, "w": ws}, valid={"x": np.zeros(6, bool)})
        Z = mf.basis(allnull, ["x"], [1], weight="w")
        self.assertEqual((Z.counts(), Z.bounds()), ((6, 0), [((0, 0), (0, 0))]))
        re, im = feature_oracle([((Fraction(100, 100),), 2), ((Fraction(-250, 100),), 3)], (Fraction(1, 3),), "radian")
        (rl, rh), (il, ih) = B.bounds(0)[0]
        self.assertTrue(rl - EPS <= re <= rh + EPS and il - EPS <= im <= ih + EPS)

    def test_spec001_engine_table_scope_and_staleness(self):
        import pandas as pd
        import torbles
        rng = np.random.default_rng(SEED + 6)
        n = 90
        d = [None if rng.random() < 0.1 else f"{int(v) / 100:.2f}" for v in rng.integers(-99999, 100000, n)]
        y = [int(v) for v in rng.integers(-40, 41, n)]
        w = [int(v) for v in rng.integers(-4, 5, n)]
        frame = pd.DataFrame({"d": pd.Series(d, dtype=object), "y": y, "w": w})
        db = torbles.from_ddl("CREATE TABLE t (d DECIMAL(10,2), y BIGINT, w BIGINT)", {"t": frame})
        t = db["t"]
        scope = mbk.Scope.from_table(t, ["d", "y", "w"])
        freqs = [(Fraction(2, 3), 1), (1, Fraction(-1, 5))]
        B = self.build(scope, ["d", "y"], freqs, weight="w")
        rows = [((Fraction(v), Fraction(yv)), wv) for v, yv, wv in zip(d, y, w) if v is not None]
        for f, omega in enumerate(freqs):
            re, im = feature_oracle(rows, omega, "radian")
            (rl, rh), (il, ih) = B.bounds()[f]
            self.assertTrue(rl - EPS <= re <= rh + EPS and il - EPS <= im <= ih + EPS)
        cache = mf.Retained()
        cache.basis(scope, ["d", "y"], freqs, weight="w")
        t.flip(0, 0)                                                       # one stored bit: a new revision
        with self.assertRaises(StaleStateError):
            mf.basis(scope, ["d", "y"], freqs, weight="w")
        with self.assertRaises(StaleStateError):
            cache.basis(scope, ["d", "y"], freqs, weight="w")

    # ---------------------------------------------------------- budgets and refusals
    def test_spec001_budgets_at_the_limit_and_one_past(self):
        rng = np.random.default_rng(SEED + 7)
        scope, cols, valid, scales, g = dataset(rng, 50, 2, groups=3)
        freqs = frequency_set(rng, 8, 2, "radian")
        cols2 = ["x0", "x1"]
        for resource, limit in (("support", 8), ("states", 24), ("tensor", 50 * 8), ("variables", 2)):
            ok = Budget(**{resource: limit})
            mf.basis(scope, cols2, freqs, weight="w", budget=ok)
            mf.basis(scope, cols2, freqs, weight="w", backend="cuda", budget=ok)
            with self.assertRaises(AlgebraBudgetError) as e:
                mf.basis(scope, cols2, freqs, weight="w", budget=Budget(**{resource: limit - 1}))
            self.assertEqual((e.exception.resource, e.exception.need, e.exception.limit),
                             (resource, limit, limit - 1))
        # bytes: find the need by refusal, then admit exactly it (CPU), and the device copies (CUDA)
        with self.assertRaises(AlgebraBudgetError) as e:
            mf.basis(scope, cols2, freqs, weight="w", budget=Budget(bytes=1))
        need = e.exception.need
        mf.basis(scope, cols2, freqs, weight="w", budget=Budget(bytes=need))
        with self.assertRaises(AlgebraBudgetError):
            mf.basis(scope, cols2, freqs, weight="w", budget=Budget(bytes=need - 1))
        dev = mf.device_bytes(scope, cols2, freqs, weight="w")
        G = mf.basis(scope, cols2, freqs, weight="w")
        mf.basis(scope, cols2, freqs, weight="w", backend="cuda", budget=Budget(bytes=max(dev, need)))
        with self.assertRaises(AlgebraBudgetError) as e:
            mf.basis(scope, cols2, freqs, weight="w", backend="cuda", budget=Budget(bytes=dev - 1))
        # actions: output = requests x frequencies, tensor = output x (d + 1), degree; derivative degree
        reqs = [dict(shift=(0.5, 0.5))] * 5
        G.act(reqs, budget=Budget(output=40, tensor=120))
        with self.assertRaises(AlgebraBudgetError) as e:
            G.act(reqs, budget=Budget(output=39))
        self.assertEqual((e.exception.resource, e.exception.need), ("output", 40))
        with self.assertRaises(AlgebraBudgetError) as e:
            G.act(reqs, budget=Budget(tensor=119))
        self.assertEqual((e.exception.resource, e.exception.need), ("tensor", 120))
        G.act(dict(order=(2, 1), phase_order=1), budget=Budget(degree=4))
        with self.assertRaises(AlgebraBudgetError):
            G.act(dict(order=(2, 1), phase_order=1), budget=Budget(degree=3))
        G.derivative((2, 2), budget=Budget(degree=4))
        with self.assertRaises(AlgebraBudgetError) as e:
            G.derivative((2, 2), budget=Budget(degree=3))
        self.assertEqual((e.exception.resource, e.exception.need), ("degree", 4))
        # the Boolean inverse: tensor = 2^k
        bits = mbk.Scope.from_arrays({f"b{j}": rng.integers(0, 2, 30) for j in range(4)})
        Wb = mf.walsh_basis(bits, [f"b{j}" for j in range(4)])
        Wb.walsh_inverse(budget=Budget(tensor=16))
        with self.assertRaises(AlgebraBudgetError):
            Wb.walsh_inverse(budget=Budget(tensor=15))

    def test_spec001_refusals(self):
        rng = np.random.default_rng(SEED + 8)
        scope, cols, valid, scales, g = dataset(rng, 30, 2)
        with self.assertRaises(ValueError):
            mf.basis(scope, ["x0", "x1"], [(1, 2), (1, 2)])               # a basis is a set
        with self.assertRaises(ValueError):
            mf.basis(scope, ["x0", "x1"], [(Fraction(2, 4), 1), (Fraction(1, 2), 1)])   # equal after reduction
        with self.assertRaises(ValueError):
            mf.basis(scope, ["x0", "x1"], [(1,)])                          # one component per column
        with self.assertRaises(ValueError):
            mf.basis(scope, ["x0", "nope"], [(1, 2)])
        with self.assertRaises(ValueError):
            mf.basis(scope, ["x0"], [1], unit="degree")
        with self.assertRaises(ValueError):
            mf.basis(scope, ["x0"], [])
        with self.assertRaises(OverflowError):
            mf.basis(scope, ["x0"], [Fraction(1, 2 ** 70)])               # not an int64 rational
        mf.basis(scope, ["x1"], [2 ** 62], unit="pi")                     # K = 2 omega = 2^63 < 2^64
        with self.assertRaises(UnsupportedError):
            mf.basis(scope, ["x1"], [2 ** 62], unit="turn")               # K = 4 omega = 2^64: too large to reduce
        wide = mbk.Scope.from_arrays({f"c{j}": rng.integers(0, 9, 10) for j in range(17)})
        with self.assertRaises(UnsupportedError):
            mf.basis(wide, [f"c{j}" for j in range(17)], [tuple(range(17))])
        B = mf.basis(scope, ["x0", "x1"], [(1, 2)], weight="w")
        with self.assertRaises(UnsupportedError):
            B.shift((2.0 ** 62, 0.0))
        with self.assertRaises(UnsupportedError):
            B.act(dict(phase=-2.0 ** 63))
        with self.assertRaises(ValueError):
            B.shift((float("nan"), 0.0))
        with self.assertRaises(ValueError):
            B.act(dict(order=(-1, 0)))
        with self.assertRaises(ValueError):
            B.bounds() and B.derivative((1, 0)).bounds()                   # a derivative: use exact()
        with self.assertRaises(TypeError):
            mf.basis(scope, ["x0"], ["1"])
        # a huge exact shift is reduced exactly: exp(i omega tau) for tau = 2^61 (radians)
        x = mbk.Scope.from_arrays({"x": np.array([0])})
        one = mf.basis(x, ["x"], [1])
        (re, im), = one.shift(2.0 ** 61)
        s, c = sincos_fx(theta_fixed(Fraction(2 ** 61), 0, "radian"))
        self.assertTrue(contains(re, c) and contains(im, s))


if __name__ == "__main__":
    unittest.main()
