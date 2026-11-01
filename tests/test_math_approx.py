# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""UNC-002: explicit approximations with certified error records: truncated Taylor and Chebyshev series, coefficient
quantization, low-rank approximations of small matrices (exact cross elimination; seeded sketches with a
probabilistic bound), binary64 evaluation with a rounding bound, compositions that carry every error term, and the
refusal of exact rewrites (torbles/math_approx.c, math_approx_gpu.cu, math_approx.py).

Oracles: Python decimal at 90 digits (exp, ln, sqrt; sin and cos by their Taylor series written here) at Chebyshev
nodes, endpoints and seeded rational points, compared with the exact approximant values (Fractions); exact Fraction
arithmetic for low-rank residuals (C U^-1 R rebuilt here from the pivots, the residual's Frobenius norm, its largest
entry and a Rayleigh-quotient lower bound of its spectral norm); the exact real value of each binary64 point for the
floating evaluation; FND-007 for the authorization of an exact claim (the control). Every row evaluation is also run on
the CUDA path (emulated here, TORBLES_EMU_SCRAMBLE=1 for scrambled thread order) and compared bit for bit with the
CPU result. Seeds are fixed.
"""
import math
import subprocess
import sys
import unittest
from decimal import Decimal, localcontext
from fractions import Fraction

import numpy as np
import pandas as pd

import torbles
from torbles import math_approx as ma
from torbles import math_bank as mbk
from torbles import math_certificates as mc
from torbles import math_unc as mu
from torbles.math_budgets import Budget
from torbles.math_domains import DomainError
from torbles.math_errors import NotConvergedError, UnsupportedError
from torbles.mathematics import AlgebraBudgetError

PREC = 90
ORACLE_SLACK = Fraction(1, 10 ** 80)          # the decimal oracle's own error (90 digits, |f| <= e^8)


# ------------------------------------------------------------------ the decimal oracle
def dec(q):
    return Decimal(q.numerator) / Decimal(q.denominator)


def _series(x, start, sign):
    """sum (-1)^k x^(2k + start) / (2k + start)! at PREC digits"""
    total, term, k = Decimal(0), Decimal(1), 0
    for i in range(1, start + 1):
        term = term * x / i
    while True:
        total += term
        k += 1
        n = 2 * k + start
        term = term * x * x / ((n - 1) * n) * sign
        if abs(term) < Decimal(10) ** -(PREC + 5):
            return total + term


def oracle(function, q):
    """the function at the exact rational q, as an exact Fraction of a PREC-digit decimal"""
    with localcontext() as ctx:
        ctx.prec = PREC
        x = dec(Fraction(q))
        if function == "exp":
            v = x.exp()
        elif function == "log":
            v = x.ln()
        elif function == "sqrt":
            v = x.sqrt()
        elif function == "recip":
            v = 1 / x
        elif function == "sin":
            v = _series(x, 1, -1)
        elif function == "cos":
            v = _series(x, 0, -1)
        else:
            raise ValueError(function)
        return Fraction(v)


def points(a, b, rng, n=33):
    """Chebyshev nodes of [a, b] (rounded to rationals inside it), the endpoints, and seeded rational points"""
    a, b = Fraction(a), Fraction(b)
    out = {a, b, (a + b) / 2}
    for k in range(n):
        t = Fraction(math.cos(math.pi * (2 * k + 1) / (2 * n))).limit_denominator(10 ** 12)
        out.add(min(max((a + b) / 2 + (b - a) / 2 * t, a), b))
    for _ in range(n // 2):
        out.add(a + (b - a) * Fraction(int(rng.integers(0, 10 ** 9)), 10 ** 9))
    return sorted(out)


def measured(f, function, xs):
    return max(abs(f.value(x) - oracle(function, x)) for x in xs)


CASES = [("exp", 0, 1), ("exp", -2, 3), ("log", 1, 2), ("log", Fraction(1, 2), 4), ("sin", 0, 1), ("sin", -3, 3),
         ("cos", 0, 2), ("sqrt", 1, 4), ("sqrt", Fraction(1, 10), 1), ("recip", 1, 3), ("recip", -3, -1)]


def frac_matrix(A):
    return [[Fraction(int(v)) for v in row] for row in A]


def solve(M, B):
    """M^-1 B by exact Gauss-Jordan (M square, nonsingular)"""
    n = len(M)
    aug = [list(M[i]) + list(B[i]) for i in range(n)]
    for c in range(n):
        p = next(r for r in range(c, n) if aug[r][c] != 0)
        aug[c], aug[p] = aug[p], aug[c]
        piv = aug[c][c]
        aug[c] = [v / piv for v in aug[c]]
        for r in range(n):
            if r != c and aug[r][c] != 0:
                f = aug[r][c]
                aug[r] = [v - f * w for v, w in zip(aug[r], aug[c])]
    return [row[n:] for row in aug]


class Approx(unittest.TestCase):
    # ------------------------------------------------------------------ acceptance 1: measured <= advertised
    def test_unc002_acceptance_measured_error_within_the_advertised_bound(self):
        rng = np.random.default_rng(2002)
        for function, a, b in CASES:
            xs = points(a, b, rng)
            for basis in ("chebyshev", "taylor"):
                for degree in (4, 9, 16):
                    f = ma.series(function, a, b, degree, basis=basis)
                    err = f.error
                    self.assertEqual(f.label, "certified")
                    self.assertEqual(f.provenance["kind"], "deterministic")
                    self.assertEqual(sum(t["magnitude"] for t in f.terms), err, (function, basis, degree))
                    m = measured(f, function, xs)
                    self.assertLessEqual(m, err + ORACLE_SLACK, (function, a, b, basis, degree, float(m), float(err)))
                    lo, hi = f.range
                    for x in xs[::4]:
                        v = oracle(function, x)
                        self.assertTrue(lo - ORACLE_SLACK <= v <= hi + ORACLE_SLACK, (function, x))
                    rel = f.relative
                    if rel is not None:                 # |f| > 0 on the interval: the relative bound holds too
                        for x in xs[::3]:
                            v = oracle(function, x)
                            self.assertLessEqual(abs(f.value(x) - v), rel * abs(v) + ORACLE_SLACK, (function, x))
                    l2 = f.l2                           # sqrt(b - a) rounded up (64 bits) times the sup bound
                    sup2 = (Fraction(b) - Fraction(a)) * err * err
                    self.assertTrue(sup2 <= l2 * l2 <= sup2 * (1 + Fraction(1, 1 << 60)), (function, basis, degree))
        # higher degree certifies smaller errors (Chebyshev economization of the Taylor series)
        errs = [ma.series("exp", 0, 1, n).error for n in (4, 8, 12, 16)]
        self.assertEqual(errs, sorted(errs, reverse=True))
        self.assertLess(errs[-1], Fraction(1, 10 ** 20))

    def test_unc002_error_budget_achieved_or_refused_never_reported(self):
        f = ma.series("exp", 0, 1, 8)
        err = f.error
        ok = ma.series("exp", 0, 1, 8, error_budget=err)              # at the limit: certified <= budget
        self.assertEqual(ok.error, err)
        self.assertEqual(ok.error_budget, err)
        with self.assertRaises(NotConvergedError):
            ma.series("exp", 0, 1, 8, error_budget=err - Fraction(1, 10 ** 40))
        with self.assertRaises(NotConvergedError):                    # cos on a long interval at degree 8
            ma.series("cos", 0, 100, 8, error_budget=Fraction(1, 1000))
        with self.assertRaises(NotConvergedError):                    # relative norm budget
            ma.series("exp", 0, 1, 4, norm="rel", error_budget=Fraction(1, 10 ** 12))
        r = ma.series("exp", 0, 1, 12, norm="rel", error_budget=Fraction(1, 10 ** 12))
        self.assertEqual(r.norm, "rel")
        self.assertLessEqual(r.relative, Fraction(1, 10 ** 12))
        with self.assertRaises(DomainError):                          # sin vanishes on [-1, 1]: no relative bound
            ma.series("sin", -1, 1, 8, norm="rel", error_budget=Fraction(1, 10))
        # operations keep the budget contract: the composed error above the budget is refused
        g = ma.series("sin", 0, 1, 8)
        s = f.add(g)
        self.assertEqual(s.error, f.error + g.error)
        with self.assertRaises(NotConvergedError):
            f.add(g, error_budget=s.error - Fraction(1, 10 ** 50))
        self.assertEqual(f.add(g, error_budget=s.error).error, s.error)

    def test_unc002_resource_budgets_at_the_limit_and_one_past(self):
        n = 12
        N = min(n + 24, 2 * n + 8)                                    # the economized Taylor degree
        ma.series("exp", 0, 1, n, budget=Budget(degree=n))
        with self.assertRaises(AlgebraBudgetError) as e:
            ma.series("exp", 0, 1, n, budget=Budget(degree=n - 1))
        self.assertEqual(e.exception.resource, "degree")
        ma.series("exp", 0, 1, n, budget=Budget(support=N + 1))
        with self.assertRaises(AlgebraBudgetError) as e:
            ma.series("exp", 0, 1, n, budget=Budget(support=N))
        self.assertEqual(e.exception.resource, "support")
        A = np.arange(20, dtype=np.int64).reshape(5, 4) ** 2
        ma.lowrank(A, 2, budget=Budget(tensor=20))
        with self.assertRaises(AlgebraBudgetError) as e:
            ma.lowrank(A, 2, budget=Budget(tensor=19))
        self.assertEqual(e.exception.resource, "tensor")
        ma.lowrank(A, 2, method="sketch", seed=3, probes=12, budget=Budget(iterations=12))
        with self.assertRaises(AlgebraBudgetError) as e:
            ma.lowrank(A, 2, method="sketch", seed=3, probes=12, budget=Budget(iterations=11))
        self.assertEqual(e.exception.resource, "iterations")
        x = np.arange(0, 100, dtype=np.int64)
        scope = mbk.Scope.from_arrays({"x": x}, scales={"x": 2})
        f = ma.series("exp", 0, 1, 8)
        for backend in ("cpu", "cuda"):
            need = f.need(scope, "x", backend)
            f.evaluate(scope, "x", backend=backend, budget=Budget(**need))
            for resource, v in need.items():
                with self.assertRaises(AlgebraBudgetError) as e:
                    f.evaluate(scope, "x", backend=backend, budget=Budget(**{resource: v - 1}))
                self.assertEqual(e.exception.resource, resource)

    def test_unc002_quantization_adds_the_induced_error(self):
        rng = np.random.default_rng(17)
        f = ma.series("exp", 0, 1, 10)
        xs = points(0, 1, rng)
        for bits in (8, 20, 40):
            q = f.quantize(bits)
            self.assertEqual(q.info["quantized"], bits)
            kinds = [t["kind"] for t in q.terms]
            self.assertIn("quantization", kinds)
            qterm = sum(t["magnitude"] for t in q.terms if t["kind"] == "quantization")
            self.assertEqual(q.error, f.error + qterm)                # nothing dropped, nothing double counted
            for c in q.coefficients:                                  # multiples of 2^-bits (Chebyshev basis)
                self.assertEqual((c * (1 << bits)).denominator, 1)
            self.assertLessEqual(max(abs(q.value(x) - f.value(x)) for x in xs), qterm)
            self.assertLessEqual(measured(q, "exp", xs), q.error + ORACLE_SLACK)
        coarse, fine = f.quantize(8), f.quantize(40)
        self.assertGreater(coarse.error, fine.error)

    # ------------------------------------------------------------------ low rank
    def test_unc002_low_rank_cross_exact_residual_matches_fraction_oracle(self):
        rng = np.random.default_rng(31)
        mats = [rng.integers(-9, 10, (5, 4)), rng.integers(-50, 50, (6, 6)),
                rng.integers(-5, 6, (6, 2)) @ rng.integers(-5, 6, (2, 7)),       # rank 2
                np.diag([1000, 100, 10, 1]).astype(np.int64)]
        for A in mats:
            A = np.ascontiguousarray(A, dtype=np.int64)
            r = int(np.linalg.matrix_rank(A))
            for k in range(1, min(A.shape) + 1):
                f = ma.lowrank(A, k, method="cross")
                self.assertEqual((f.label, f.provenance["kind"]), ("certified", "deterministic"))
                rows, cols = f.pivots()
                Af = frac_matrix(A)
                Cm = [[Af[i][j] for j in cols] for i in range(A.shape[0])]
                U = [[Af[i][j] for j in cols] for i in rows]
                R = [[Af[i][j] for j in range(A.shape[1])] for i in rows]
                UR = solve(U, R)
                approx = [[sum(Cm[i][t] * UR[t][j] for t in range(len(rows))) for j in range(A.shape[1])]
                          for i in range(A.shape[0])]
                self.assertEqual(f.entries(), approx, (A.tolist(), k))
                E = [[Af[i][j] - approx[i][j] for j in range(A.shape[1])] for i in range(A.shape[0])]
                F2 = sum(v * v for row in E for v in row)
                b = f.bounds()
                self.assertEqual(b["frobenius_squared_exact"], F2)
                self.assertGreaterEqual(b["frobenius"] ** 2, F2)
                self.assertGreaterEqual(b["max"], max(abs(v) for row in E for v in row))
                # the spectral bound is above a Rayleigh quotient of E^T E (a lower bound of ||E||_2^2)
                if F2:
                    _, _, vt = np.linalg.svd(np.array([[float(v) for v in row] for row in E]))
                    v = [Fraction(float(t)).limit_denominator(10 ** 9) for t in vt[0]]
                    Ev = [sum(row[j] * v[j] for j in range(len(v))) for row in E]
                    rq = sum(t * t for t in Ev) / sum(t * t for t in v)
                    self.assertGreaterEqual(b["spectral"] ** 2, rq)
                    self.assertLessEqual(b["spectral"] ** 2, F2 * (1 + Fraction(1, 10 ** 6)) + Fraction(1, 10 ** 30))
                if k >= r:
                    self.assertEqual(F2, 0)                           # an exact factorization
                    self.assertEqual(f.error, 0)
        with self.assertRaises(UnsupportedError):
            ma.lowrank(np.ones((65, 2), dtype=np.int64), 1)

    def test_unc002_sketch_is_labelled_probabilistic_with_its_seed_and_failure_probability(self):
        rng = np.random.default_rng(41)
        A = np.ascontiguousarray(rng.integers(-5, 6, (8, 3)) @ rng.integers(-5, 6, (3, 6)) + np.eye(8, 6, dtype=np.int64),
                                 dtype=np.int64)
        for seed in (1, 2, 3):
            f = ma.lowrank(A, 3, method="sketch", seed=seed, probes=32)
            self.assertEqual(f.label, "probabilistic")
            p = f.provenance
            self.assertEqual((p["kind"], p["seed"], p["probes"]), ("randomized", seed, 32))
            self.assertEqual(p["failure"], Fraction(13, 16) ** 32)
            self.assertIn("probability", p["statement"])
            self.assertIn("independent", f.assumptions)
            Af = frac_matrix(A)
            E = [[Af[i][j] - e for j, e in enumerate(row)] for i, row in enumerate(f.entries())]
            F2 = sum(v * v for row in E for v in row)
            self.assertGreaterEqual(f.bounds()["frobenius"] ** 2, F2)  # this seed's outcome (labelled probabilistic)
            same = ma.lowrank(A, 3, method="sketch", seed=seed, probes=32)
            self.assertEqual(same.entries(), f.entries())               # reproducible from the seed
            self.assertEqual(same.probes(), f.probes())
            om, kept = f.sketch()
            self.assertTrue(set(np.unique(om)) <= {-1, 1})
            with self.assertRaises(mc.NotAuthorizedError) as e:
                f.authorize_exact()
            self.assertEqual(e.exception.reason, "statistical")

    def test_unc002_float_evaluation_carries_a_rounding_bound(self):
        for function, a, b, n in (("exp", 0, 1, 10), ("sin", -3, 3, 20), ("log", 1, 2, 12), ("recip", 1, 3, 14)):
            f = ma.series(function, a, b, n)
            fl = f.to_float()
            self.assertIn("rounding", [t["kind"] for t in fl.terms])
            self.assertGreaterEqual(fl.error, f.error)
            xs = np.concatenate([np.linspace(float(a), float(b), 41), [float(a), float(b)]])
            vals, bnds, status = fl.doubles(xs)
            for x, v, e, s in zip(xs, vals, bnds, status):
                self.assertEqual(s, "ok")
                self.assertLessEqual(abs(Fraction(float(v)) - oracle(function, Fraction(float(x)))),
                                     Fraction(float(e)) + ORACLE_SLACK, (function, x))
            _, _, st = fl.doubles([float(a) - 0.5, float(b) + 0.5, float("nan")])
            self.assertEqual(st[:2], ["domain", "domain"])

    # ------------------------------------------------------------------ acceptance 2: composition, authorization
    def test_unc002_acceptance_composition_never_drops_uncertainty(self):
        rng = np.random.default_rng(53)
        xs = points(0, 1, rng, 17)
        f, g = ma.series("exp", 0, 1, 8), ma.series("sin", 0, 1, 8)
        fid, gid = f.info["id"], g.info["id"]
        h = f * g + 3 * f
        origins = {t["origin"] for t in h.terms}
        self.assertTrue({fid, gid} <= origins, (fid, gid, h.terms))
        self.assertTrue(any(t["scaled"] for t in h.terms))
        self.assertEqual(sum(t["magnitude"] for t in h.terms), h.error)
        self.assertGreaterEqual(h.error, 4 * f.error)                 # 3 f + (f g) carries f's error at least 4 times
        for x in xs:
            exact = oracle("exp", x) * oracle("sin", x) + 3 * oracle("exp", x)
            self.assertLessEqual(abs(h.value(x) - exact), h.error + ORACLE_SLACK)
        d = (f - f)                                                   # cancellation never cancels the error
        self.assertEqual(d.error, 2 * f.error)
        for x in xs[::4]:
            self.assertEqual(d.value(x), 0)
        # log(exp(x)) = x on [1/2, 3]: the inner range must lie in the outer interval
        inner = ma.series("exp", Fraction(1, 2), 2, 12)
        outer = ma.series("log", 1, 8, 16)
        comp = outer(inner)
        self.assertEqual(comp.info["operation"], "compose")
        corigins = {t["origin"] for t in comp.terms}
        self.assertTrue({inner.info["id"], outer.info["id"]} <= corigins)
        for x in points(Fraction(1, 2), 2, rng, 9):
            self.assertLessEqual(abs(comp.value(x) - x), comp.error + ORACLE_SLACK)
        self.assertGreaterEqual(comp.error, outer.error)
        with self.assertRaises(DomainError):                          # exp([0, 1]) is not inside [3, 4]
            ma.series("log", 3, 4, 8)(f)
        # a product whose degree passes the representation limit is truncated with its own term
        big = ma.series("exp", 0, 1, 40).mul(ma.series("cos", 0, 1, 40))
        self.assertIn("truncation", [t["kind"] for t in big.terms])
        self.assertLessEqual(big.info["degree"], ma.MAX_DEGREE)
        for x in xs[::4]:
            self.assertLessEqual(abs(big.value(x) - oracle("exp", x) * oracle("cos", x)), big.error + ORACLE_SLACK)

    def test_unc002_acceptance_approximations_never_authorize_exact_rewrites(self):
        f = ma.series("exp", 0, 1, 8)
        cands = [f, f.quantize(20), f.to_float(), f * f, ma.series("log", Fraction(1, 2), 3, 8)(ma.series("exp", 0, 1, 8)),
                 ma.lowrank(np.arange(12, dtype=np.int64).reshape(3, 4), 1)]
        for c in cands:
            d = c.decide()
            self.assertFalse(d["authorized"])
            self.assertEqual(d["reason"], "approximate")
            with self.assertRaises(mc.NotAuthorizedError) as e:
                c.authorize_exact()
            self.assertEqual(e.exception.reason, "approximate")
        exact_poly = ma.poly([1, 2, 3], 0, 1)                          # error 0, still an approximation object
        self.assertEqual(exact_poly.error, 0)
        with self.assertRaises(mc.NotAuthorizedError):
            exact_poly.authorize_exact()
        # FND-007: the approximate relationship y ~ p(x) is a snapshot with a tolerance, refused as "approximate";
        # an exact claim on the same scope (the control) is authorized
        rng = np.random.default_rng(61)
        n = 400
        x = rng.integers(0, 101, n)
        df = pd.DataFrame({"x": pd.Series([f"{v / 100:.2f}" for v in x], dtype=object),
                           "y": pd.Series([f"{float(oracle('exp', Fraction(int(v), 100))):.4f}" for v in x], dtype=object),
                           "z": pd.Series([f"{2 * v / 100:.2f}" for v in x], dtype=object)})
        db = torbles.from_ddl("CREATE TABLE f (x DECIMAL(4,2), y DECIMAL(8,4), z DECIMAL(5,2));", {"f": df})
        scope = mbk.Scope.from_table(db["f"], ["x", "y", "z"])
        g = ma.series("exp", 0, 1, 4)
        cert = ma.relation(g, scope, "x", "y", tolerance=Fraction(1, 10 ** 4), bits=16)
        self.assertEqual(cert.kind, "snapshot")
        dd = cert.decide(scope)
        self.assertEqual((dd["authorized"], dd["reason"]), (False, "approximate"))
        with self.assertRaises(mc.NotAuthorizedError) as e:
            cert.authorize_exact(scope)
        self.assertEqual(e.exception.reason, "approximate")
        control = mc.snapshot(scope, mc.affine("z", "x", scale=2))
        self.assertTrue(control.decide(scope)["authorized"])
        control.authorize_exact(scope)

    # ------------------------------------------------------------------ rows: CPU == CUDA, enclosures, exact sums
    def test_unc002_row_evaluation_cpu_equals_cuda_and_encloses_the_function(self):
        rng = np.random.default_rng(71)
        n = 1500
        x = rng.integers(-20, 121, n)                                 # x / 100 in [-0.2, 1.2]: some rows outside
        valid = rng.random(n) > 0.1
        groups = rng.integers(0, 7, n)
        sel = rng.random(n) < 0.9
        scope = mbk.Scope.from_arrays({"x": x}, valid={"x": valid}, scales={"x": 2}, groups=groups, ngroups=7,
                                      rows=sel)
        inside = mbk.Scope.from_arrays({"x": np.clip(x, 0, 100)}, valid={"x": valid}, scales={"x": 2},
                                       groups=groups, ngroups=7, rows=sel)
        for f, fn in ((ma.series("exp", 0, 1, 10), "exp"), (ma.series("exp", 0, 1, 10, basis="taylor"), "exp"),
                      (ma.series("exp", 0, 1, 10).to_float(), "exp"), (ma.series("sqrt", Fraction(1, 100), 1, 12), "sqrt")):
            for sc, xs in ((scope, x), (inside, np.clip(x, 0, 100))):
                cpu = f.evaluate(sc, "x", fbits=40)
                gpu = f.evaluate(sc, "x", fbits=40, backend="cuda")
                self.assertIn(gpu.backend, ("cuda", "cuda-emulated"))
                self.assertTrue(cpu.same(gpu))
                a, b = f.interval
                total = {g: Fraction(0) for g in range(7)}
                refused = set()
                for i in range(n):
                    st = cpu.statuses()[i]
                    if not sel[i]:
                        self.assertEqual(st, "unselected")
                        continue
                    if not valid[i]:
                        self.assertEqual(st, "null")
                        continue
                    q = Fraction(int(xs[i]), 100)
                    edge = q in (a, b) and q.denominator & (q.denominator - 1)   # a non-dyadic endpoint: the
                    if not a <= q <= b or (edge and st == "domain"):            # binary64 enclosure of x leaves
                        self.assertEqual(st, "domain")                          # the domain: refused, never guessed
                        refused.add(int(groups[i]))
                        continue
                    self.assertEqual(st, "ok")
                    v = oracle(fn, q)
                    self.assertTrue(Fraction(cpu.lo[i]) - ORACLE_SLACK <= v <= Fraction(cpu.hi[i]) + ORACLE_SLACK,
                                    (fn, q, cpu.lo[i], cpu.hi[i]))
                    total[int(groups[i])] += v
                for g in range(7):
                    grp = cpu.group(g)
                    if g in refused:
                        self.assertIsNone(grp["sum"])
                        self.assertGreater(grp["refused"], 0)
                    else:
                        lo, hi = grp["sum"]
                        self.assertTrue(lo - ORACLE_SLACK <= total[g] <= hi + ORACLE_SLACK, (g, float(lo), float(hi)))

    def test_unc002_approximate_predicates_change_the_decided_fraction_soundly(self):
        rng = np.random.default_rng(83)
        n = 3000
        G = 30
        groups = rng.integers(0, G, n)
        centre = rng.integers(0, 101, G)                              # each group's values near its centre
        x = np.clip(centre[groups] + rng.integers(-3, 4, n), 0, 100)
        valid = rng.random(n) > 0.05
        scope = mbk.Scope.from_arrays({"x": x}, valid={"x": valid}, scales={"x": 2}, groups=groups, ngroups=G)
        state = mu.State.from_scope(scope)
        T = Fraction(2)                                               # exp(x) > 2 <=> x > ln 2 ~ 0.693
        fractions = []
        for degree in (2, 4, 8):
            f = ma.series("exp", 0, 1, degree)
            bits, und, st = ma.filter(state, scope, "x", f, ">", T)
            ev_cpu = f.evaluate(scope, "x")
            ev_gpu = f.evaluate(scope, "x", backend="cuda")
            self.assertTrue(ev_cpu.same(ev_gpu))
            b = np.unpackbits(bits.view(np.uint8), bitorder="little")[:n]
            u = np.unpackbits(und.view(np.uint8), bitorder="little")[:n]
            for i in range(n):
                if not valid[i]:
                    self.assertEqual((b[i], u[i]), (0, 0))
                    continue
                v = oracle("exp", Fraction(int(x[i]), 100))
                if b[i]:
                    self.assertGreater(v, T, (degree, i))
                elif not u[i]:
                    self.assertLessEqual(v, T, (degree, i))
                else:                                                  # undecided: the enclosure straddles T
                    self.assertTrue(Fraction(ev_cpu.lo[i]) <= T <= Fraction(ev_cpu.hi[i]), (degree, i))
            self.assertEqual(st["rows_true"], int(b.sum()))
            self.assertEqual(st["rows_undecided"], int(u.sum()))
            self.assertEqual(st["groups_all"] + st["groups_none"] + st["groups_unknown"] + st["groups_empty"] +
                             st["groups_outside"], G)
            fractions.append(st["rows_undecided"])
        self.assertGreaterEqual(fractions[0], fractions[-1])           # a sharper approximation decides more rows

    def test_unc002_domain_and_representation_refusals(self):
        with self.assertRaises(DomainError):
            ma.series("log", -1, 1, 8)
        with self.assertRaises(DomainError):
            ma.series("sqrt", -1, 2, 8)
        with self.assertRaises(DomainError):
            ma.series("recip", -1, 1, 8)
        with self.assertRaises(UnsupportedError):
            ma.series("exp", 0, 2000, 8)
        with self.assertRaises(UnsupportedError):
            ma.series("exp", 0, 1, ma.MAX_DEGREE + 1)
        with self.assertRaises(ValueError):
            ma.series("exp", 1, 0, 8)
        with self.assertRaises(TypeError):
            ma.series("exp", 0.0, 1, 8)                               # exact numbers only
        with self.assertRaises(UnsupportedError):
            ma.lowrank(np.ones((4, 4), dtype=np.int64), 2, method="sketch", probes=ma.MAX_PROBES + 1)
        with self.assertRaises(ValueError):
            ma.series("tan", 0, 1, 8)

    def test_unc002_module_is_opt_in(self):
        code = "import sys, torbles; print('torbles.math_approx' in sys.modules)"
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip()
        self.assertEqual(out, "False")


if __name__ == "__main__":
    unittest.main()
