# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""GEN-001 (finite-support distribution generating polynomials) (torbles/math_gen.c, math_gen_gpu.cu, math_gen.py).

Oracles, independent of the implementation: DuckDB SQL over the same rows (GROUP BY counts and weight sums for the
coefficients, COUNT for G(1), SUM of products of HUGEINT casts for the exact row moments), Python polynomial
arithmetic on dicts {value: coefficient} with Python ints and Fractions (products, powers, evaluation at rational
points, Euler operators, derivatives, falling-factorial moments), brute-force sums of independent draws over the rows,
and the FND-009 truncated series products (math_carriers.Series) for the truncated powers. Every CUDA result (the
128-bit convolution, the batched evaluation; emulated here, TORBLES_EMU_SCRAMBLE=1 scrambles the thread order) is
compared bit for bit with its CPU reference and with the exact big-integer result. Seeds are fixed.
"""
import itertools
import subprocess
import sys
import unittest
from fractions import Fraction

import duckdb
import numpy as np
import pandas as pd

from torbles import math_bank as mbk, math_carriers as mc, math_dist as md, math_gen as mg
from torbles.math_budgets import Budget
from torbles.math_domains import DomainError
from torbles.math_errors import NoSolutionError, UnsupportedError
from torbles.mathematics import AlgebraBudgetError


# ---------------------------------------------------------------------- dict polynomials (the oracle)
def pmul(a, b):
    out = {}
    for u, x in a.items():
        for v, y in b.items():
            k = tuple(p + q for p, q in zip(u, v)) if isinstance(u, tuple) else u + v
            out[k] = out.get(k, 0) + x * y
    return {k: c for k, c in out.items() if c}


def ppow(a, k):
    out = {(0,) * len(next(iter(a))) if a and isinstance(next(iter(a)), tuple) else 0: 1}
    for _ in range(k):
        out = pmul(out, a)
    return out


def peval(a, z):
    zs = z if isinstance(z, tuple) else (z,)
    total = Fraction(0)
    for v, c in a.items():
        vs = v if isinstance(v, tuple) else (v,)
        t = Fraction(c)
        for x, e in zip(zs, vs):
            t *= Fraction(x) ** e
        total += t
    return total.numerator if total.denominator == 1 else total


def falling(v, k):
    out = 1
    for i in range(k):
        out *= v - i
    return out


def frame(values, valid, groups):
    return pd.DataFrame({"g": groups, **{n: pd.Series(np.where(valid[n], v, 0), dtype="Int64").where(valid[n])
                                         for n, v in values.items()}})


def powsql(col, k):
    return " * ".join([f"CAST({col} AS HUGEINT)"] * k) if k else "1"


class Generating(unittest.TestCase):
    def data(self, rng, n=3000, G=3):
        x = rng.integers(-40, 60, n)
        y = rng.integers(-3, 5, n)
        w = rng.integers(-5, 30, n)
        valid = {"x": rng.random(n) > 0.1, "y": rng.random(n) > 0.05, "w": np.ones(n, bool)}
        g = rng.integers(0, G, n)
        scope = mbk.Scope.from_arrays({"x": x, "y": y, "w": w}, valid={"x": valid["x"], "y": valid["y"]}, groups=g,
                                      ngroups=G + 1)                      # the last group is absent
        con = duckdb.connect()
        con.register("t", frame({"x": x, "y": y, "w": w}, valid, g))
        return scope, con

    # ------------------------------------------------------------------ acceptance 1: G(1), coefficients, moments
    def test_gen001_g1_coefficients_and_euler_moments_match_counts_and_exact_row_moments(self):
        rng = np.random.default_rng(4001)
        scope, con = self.data(rng)
        for repr_ in ("sparse", "dense"):
            h = md.histogram(scope, ["x"], weight="w", repr=repr_)
            for gid in range(3):
                for rep in ("sparse", "dense"):
                    gp = mg.generating(h, gid, repr=rep)
                    counts = dict(con.execute(f"SELECT x, COUNT(*) FROM t WHERE g = {gid} AND x IS NOT NULL GROUP BY x")
                                  .fetchall())
                    self.assertEqual(gp.terms(), {int(k): int(v) for k, v in counts.items()})
                    total, nulls = con.execute(f"SELECT COUNT(x), COUNT(*) - COUNT(x) FROM t WHERE g = {gid}").fetchone()
                    self.assertEqual(gp.total(), total)                       # G(1) = the non-NULL rows
                    self.assertEqual((gp.info["nulls"], gp.info["rows"]), (nulls, total + nulls))
                    for v in list(counts)[:5] + [-41, 59, 1000]:
                        self.assertEqual(gp.coefficient(int(v)), int(counts.get(v, 0)))
                    for k in range(7):
                        want = con.execute(f"SELECT SUM({powsql('x', k)}) FROM t WHERE g = {gid} AND x IS NOT NULL"
                                           ).fetchone()[0]
                        self.assertEqual(gp.moment(k), int(want), (gid, k))           # the exact row moment
                        self.assertEqual(gp.euler(k).total(), int(want), (gid, k))    # (z d/dz)^k G at z = 1
                        rows = [int(v) for v in con.execute(f"SELECT x FROM t WHERE g = {gid} AND x IS NOT NULL")
                                .fetchnumpy()["x"]]
                        fm = sum(falling(v, k) for v in rows)
                        self.assertEqual(gp.moment(k, falling=True), fm, (gid, k))    # factorial moments
                        self.assertEqual(gp.derivative(k).total(), fm, (gid, k))      # G^(k)(1)
                    for z in (2, -1, Fraction(1, 3), Fraction(-7, 5)):
                        self.assertEqual(gp(z), peval(gp.terms(), z))
                wp = mg.generating(h, gid, mass="weight")
                sums = dict(con.execute(f"SELECT x, SUM(w) FROM t WHERE g = {gid} AND x IS NOT NULL GROUP BY x "
                                        "HAVING SUM(w) <> 0").fetchall())
                self.assertEqual(wp.terms(), {int(k): int(v) for k, v in sums.items()})
                self.assertEqual(wp.total(), int(con.execute(f"SELECT SUM(w) FROM t WHERE g = {gid} AND x IS NOT NULL")
                                                 .fetchone()[0]))
                self.assertEqual(wp.moment(2), int(con.execute(f"SELECT SUM({powsql('x', 2)} * w) FROM t WHERE g = "
                                                               f"{gid} AND x IS NOT NULL").fetchone()[0]))
        # the absent group: the zero polynomial
        empty = mg.generating(md.histogram(scope, ["x"]), 3)
        self.assertEqual((empty.terms(), empty.total(), empty.moment(3), empty.info["rows"]), ({}, 0, 0, 0))
        # two-dimensional support: coefficients, G(1, 1), mixed moments, marginals
        h2 = md.histogram(scope, ["x", "y"])
        for gid in range(3):
            g2 = mg.generating(h2, gid)
            cells = {(int(a), int(b)): int(c) for a, b, c in con.execute(
                f"SELECT x, y, COUNT(*) FROM t WHERE g = {gid} AND x IS NOT NULL AND y IS NOT NULL GROUP BY x, y")
                .fetchall()}
            self.assertEqual(g2.terms(), cells)
            self.assertEqual(g2.total(), sum(cells.values()))
            for k1, k2 in ((1, 0), (0, 2), (2, 1), (3, 3)):
                want = con.execute(f"SELECT SUM({powsql('x', k1)} * {powsql('y', k2)}) FROM t WHERE g = {gid} AND "
                                   "x IS NOT NULL AND y IS NOT NULL").fetchone()[0]
                self.assertEqual(g2.moment((k1, k2)), int(want))
                self.assertEqual(g2.euler(k1, var=0).euler(k2, var=1).total(), int(want))
            self.assertEqual(g2((Fraction(1, 2), -2)), peval(cells, (Fraction(1, 2), -2)))
            marg = {}
            for (a, b), c in cells.items():
                marg[a] = marg.get(a, 0) + c
            self.assertEqual(g2.marginal(0).terms(), marg)
            self.assertTrue(g2.to_dense().equal(g2))
            self.assertTrue(mg.generating(md.histogram(scope, ["x", "y"], repr="dense"), gid, repr="dense").equal(g2))
        con.close()

    # ------------------------------------------------------------------ acceptance 2: shifts, Laurent, multiplicity
    def test_gen001_negative_values_use_declared_shifts_and_coefficients_keep_bag_multiplicity(self):
        x = [-3, -3, -3, 0, 2, 2, None, -1]
        scope = mbk.Scope.from_arrays({"x": [0 if v is None else v for v in x]}, valid={"x": [v is not None for v in x]})
        h = md.histogram(scope, ["x"])
        g = mg.generating(h)                                        # shift = the smallest value
        self.assertEqual(g.shift, (-3,))
        self.assertEqual(g.terms(), {-3: 3, -1: 1, 0: 1, 2: 2})    # duplicates are multiplicities, not one term each
        s = g.series(6)                                              # P(z) = z^3 G(z) = 3 + z^2 + z^3 + 2 z^5
        self.assertEqual(s.coefficients, [3, 0, 1, 1, 0, 2])
        low = mg.generating(h, shift=-10)                            # a declared shift below the smallest value
        self.assertEqual((low.shift, low.terms(), low.info["min"]), ((-10,), g.terms(), (-3,)))
        self.assertTrue(low.equal(g))
        s = low.series(10)                                           # P(z) = z^10 G(z): exponents from 7
        self.assertEqual(s.coefficients, [0] * 7 + [3, 0, 1])
        with self.assertRaises(DomainError):
            mg.generating(h, shift=-2)                               # a value below the declared shift
        # Laurent operators: G'(z) = sum c v z^(v - 1): the shift moves, the values stay exact
        d = g.derivative()
        self.assertEqual((d.terms(), d.shift), ({-4: -9, -2: -1, 1: 4}, (-4,)))
        self.assertEqual(g.derivative(2).terms(), {-5: 36, -3: 2, 0: 4})
        self.assertEqual(g(Fraction(1, 2)), 3 * 8 + 2 + 1 + 2 * Fraction(1, 4))
        self.assertEqual(g(-1), -3 - 1 + 1 + 2)
        with self.assertRaises(NoSolutionError):                     # z = 0 is a pole of a Laurent polynomial
            g(0)
        pos = mg.from_terms({0: 4, 2: 1, 5: 7}, shift=-3)
        self.assertEqual(pos(0), 4)                                  # nonnegative values: G(0) = c(0)
        self.assertEqual(pos.evaluate_many([0, 1]), [4, 12])
        with self.assertRaises(NoSolutionError):
            g.evaluate_many([0])
        # weights of any sign: zero sums leave the support
        ws = mbk.Scope.from_arrays({"x": [-2, -2, 1, 4, 4], "w": [3, -3, 5, -2, 7]})
        wg = mg.generating(md.histogram(ws, ["x"], weight="w"), mass="weight")
        self.assertEqual((wg.terms(), wg.shift, wg.info["nonnegative"]), ({1: 5, 4: 5}, (1,), True))
        with self.assertRaises(ValueError):
            mg.generating(h, mass="weight")                          # no weight column
        # explicit terms: duplicates add, zeros vanish; decimal scales are kept as logical integers
        t = mg.from_terms([(-1, 2), (3, 1), (-1, 5), (7, 0)])
        self.assertEqual(t.terms(), {-1: 7, 3: 1})
        dec = mbk.Scope.from_arrays({"d": [-125, 250, 250]}, scales={"d": 2})
        gd = mg.generating(md.histogram(dec, ["d"]))
        self.assertEqual((gd.terms(), gd.info["scales"]), ({-125: 1, 250: 2}, (2,)))
        with self.assertRaises(DomainError):
            gd * g                                                   # values of different scales do not add

    # ------------------------------------------------------------------ products, powers, truncated series
    def test_gen001_products_powers_and_truncated_series_match_explicit_products(self):
        rng = np.random.default_rng(4003)
        a = mg.from_terms({int(v): int(c) for v, c in zip(rng.integers(-6, 9, 7), rng.integers(1, 50, 7))})
        b = mg.from_terms({int(v): int(c) for v, c in zip(rng.integers(-4, 12, 9), rng.integers(-20, 50, 9))})
        A, B = a.terms(), b.terms()
        self.assertEqual((a * b).terms(), pmul(A, B))
        for k in range(0, 9):
            self.assertEqual(a.pow(k).terms(), ppow(A, k), k)
        big = a.pow(40)                                             # coefficients far past 2^128: exact big integers
        self.assertGreater(big.info["max_bits"], 200)
        self.assertEqual(big.terms(), ppow(A, 40))
        self.assertEqual(big.total(), sum(A.values()) ** 40)
        # truncated powers equal the FND-009 series products (P = z^-shift G)
        for order in (4, 16, 64):
            s = a.series(order)
            acc = s
            for k in range(2, 6):
                acc = acc * s                                         # FND-009: the truncated series product
                tp = a.pow(k, order=order)
                want = {e + k * a.shift[0]: int(c) for e, c in enumerate(acc.coefficients) if c != 0}
                self.assertEqual(tp.terms(), want, (order, k))
            self.assertEqual(a.mul(b, order=order).terms(),
                             {e: c for e, c in pmul(A, B).items() if e - a.shift[0] - b.shift[0] < order})
        # the distribution of the sum of two independent draws: brute force over the rows of two groups
        x = rng.integers(-5, 6, 400)
        g = rng.integers(0, 2, 400)
        h = md.histogram(mbk.Scope.from_arrays({"x": x}, groups=g, ngroups=2), ["x"])
        g0, g1 = mg.generating(h, 0), mg.generating(h, 1)
        brute = {}
        for u in x[g == 0]:
            for v in x[g == 1]:
                brute[int(u + v)] = brute.get(int(u + v), 0) + 1
        self.assertEqual((g0 * g1).terms(), brute)
        self.assertEqual((g0 * g1).total(), int((g == 0).sum() * (g == 1).sum()))
        # the box and the sorted-pairs strategies: wide sparse values sort, narrow ones use a box; dense stays dense
        wide = mg.from_terms({int(v): 1 for v in rng.integers(-10 ** 12, 10 ** 12, 50)})
        W = wide.terms()
        self.assertEqual((wide * wide).terms(), pmul(W, W))
        self.assertEqual((wide * a).terms(), pmul(W, A))
        da, db = a.to_dense(), b.to_dense()
        self.assertEqual((da * db).info["repr"], "dense")
        self.assertTrue((da * db).equal(a * b))
        self.assertEqual((da * b).info["repr"], "sparse")
        # two variables
        p2 = mg.from_terms({(0, 1): 2, (1, 0): -1, (2, 3): 5, (-1, -2): 3})
        q2 = mg.from_terms({(1, 1): 1, (0, -1): 4, (3, 0): -2})
        P2, Q2 = p2.terms(), q2.terms()
        self.assertEqual((p2 * q2).terms(), pmul(P2, Q2))
        self.assertEqual(p2.pow(3).terms(), ppow(P2, 3))
        self.assertEqual(p2.to_dense().pow(2).terms(), ppow(P2, 2))
        with self.assertRaises(UnsupportedError):
            p2.series(4)
        with self.assertRaises(UnsupportedError):
            p2.mul(q2, order=3)

    def test_gen001_cuda_convolution_and_batched_evaluation_equal_cpu(self):
        rng = np.random.default_rng(4004)
        for trial in range(6):
            x = rng.integers(-30, 200, 5000)
            g = rng.integers(0, 2, 5000)
            h = md.histogram(mbk.Scope.from_arrays({"x": x}, groups=g, ngroups=2), ["x"])
            a, b = mg.generating(h, 0), mg.generating(h, 1, repr="dense" if trial % 2 else "sparse")
            cpu, gpu = a.mul(b), a.mul(b, backend="cuda")
            self.assertTrue(cpu.equal(gpu))
            self.assertEqual(gpu.terms(), pmul(a.terms(), b.terms()))
            # the raw 128-bit outputs of the CPU reference and of the kernel, bit for bit
            da, db = a.dense_i64(), b.dense_i64()
            ref, dev = np.zeros((len(da) + len(db) - 1, 2), np.int64), np.zeros((len(da) + len(db) - 1, 2), np.int64)
            refusal = mg._Refusal()
            self.assertEqual(mg._call("tr_gen_conv_i128", da.ctypes.data, len(da), db.ctypes.data, len(db),
                                      ref.ctypes.data, None, None), 0)
            mg._gpu().run("tr_mgpu_gen_conv", da.ctypes.data, len(da), db.ctypes.data, len(db), dev.ctypes.data, None,
                          mg.C.byref(refusal), refusal=refusal)
            self.assertTrue(np.array_equal(ref, dev))
            # batched evaluation: CPU reference and kernel bit for bit; the exact value wherever no overflow
            pts = [1, -1, 2, Fraction(1, 2), Fraction(-3, 7), Fraction(5, 3), 10 ** 6, 0]
            cv = a.evaluate_many(pts[:-1])
            rawc = a._last_values.copy()
            gv = a.evaluate_many(pts[:-1], backend="cuda")
            self.assertTrue(np.array_equal(rawc, a._last_values))
            self.assertEqual(cv, gv)
            for z, v in zip(pts, cv):
                if v is None:
                    self.assertGreater(abs(a(z)), (1 << 120) // 2 ** 20 if z == 10 ** 6 else 0)
                else:
                    self.assertEqual(v, a(z))
            self.assertIsNone(cv[6])                                 # 10^6 ^ 230: past int128
        # the 128-bit proof: a convolution that could leave int128 is refused on both paths (the CPU state is exact)
        huge = mg.from_terms({0: (1 << 63) - 1, 1: (1 << 63) - 1, 2: (1 << 63) - 1})   # c_2 = 3 (2^63 - 1)^2 > 2^127
        exact = huge * huge
        self.assertEqual(exact.terms(), pmul(huge.terms(), huge.terms()))
        with self.assertRaises(OverflowError):
            huge.mul(huge, backend="cuda")
        d = huge.dense_i64()
        out = np.zeros((5, 2), np.int64)
        self.assertEqual(mg._call("tr_gen_conv_i128", d.ctypes.data, 3, d.ctypes.data, 3, out.ctypes.data, None, None),
                         -3)
        with self.assertRaises(OverflowError):                       # coefficients past int64: no dense int64 copy
            exact.mul(exact, backend="cuda")

    def test_gen001_budgets_support_growth_and_bytes_are_refused_before_allocation(self):
        rng = np.random.default_rng(4005)
        x = rng.integers(0, 64, 2000)
        h = md.histogram(mbk.Scope.from_arrays({"x": x}), ["x"])
        g = mg.generating(h)
        n = g.info["nonzero"]
        with self.assertRaises(AlgebraBudgetError) as ctx:
            mg.generating(h, budget=Budget(support=n - 1))
        self.assertEqual((ctx.exception.resource, ctx.exception.need), ("support", n))
        self.assertTrue(mg.generating(h, budget=Budget(support=n)).equal(g))
        # products: the coefficient pairs (tensor), the result's terms (support growth) and bytes
        need = g.need_mul(g)
        self.assertEqual((need["tensor"], need["terms"]), (n * n, 2 * g.info["span"][0] - 1))
        for resource in ("tensor", "support", "bytes"):
            value = need[resource]
            self.assertTrue(g.mul(g, budget=Budget(**{resource: value})).equal(g * g))
            with self.assertRaises(AlgebraBudgetError) as ctx:
                g.mul(g, budget=Budget(**{resource: value - 1}))
            self.assertEqual((ctx.exception.resource, ctx.exception.need), (resource, value))
        # powers: the largest step of the chain is checked before the first product
        for k in (4, 16, 64):
            pn = g.need_pow(k)
            self.assertEqual(pn["terms"], k * (g.info["span"][0] - 1) + 1)
            for resource in ("tensor", "support", "bytes"):
                ok = g.pow(k, budget=Budget(**{resource: pn[resource]}))
                self.assertEqual(ok.info["span"], (pn["terms"],))
                with self.assertRaises(AlgebraBudgetError) as ctx:
                    g.pow(k, budget=Budget(**{resource: pn[resource] - 1}))
                self.assertEqual((ctx.exception.resource, ctx.exception.need), (resource, pn[resource]))
        self.assertEqual(g.pow(64).total(), 2000 ** 64)
        tn = g.need_pow(64, order=16)
        self.assertEqual(tn["terms"], 16)
        self.assertLess(tn["bytes"], g.need_pow(64)["bytes"])
        # the dense box of a wide support, a huge evaluation, moment orders, series orders
        wide = mbk.Scope.from_arrays({"x": [0, 10 ** 9]})
        hw = md.histogram(wide, ["x"])
        with self.assertRaises(AlgebraBudgetError) as ctx:
            mg.generating(hw, repr="dense")
        self.assertEqual(ctx.exception.resource, "support")
        sw = mg.generating(hw)
        self.assertEqual(sw(1), 2)
        self.assertEqual(sw(-1), 2)                                  # two terms: the gap costs one power
        with self.assertRaises(AlgebraBudgetError) as ctx:
            sw(Fraction(1, 3))                                       # 3^(10^9): bytes refused before computing
        self.assertEqual(ctx.exception.resource, "bytes")
        with self.assertRaises(AlgebraBudgetError) as ctx:
            g.moment(65)
        self.assertEqual((ctx.exception.resource, ctx.exception.need), ("degree", 65))
        self.assertEqual(g.moment(64), sum(int(v) ** 64 for v in x))
        with self.assertRaises(AlgebraBudgetError):
            g.series(66)                                             # FND-009: degree = order - 1 <= 64
        self.assertEqual(g.series(65).coefficient(0), g.coefficient(0))
        # the device kernel's own budget (output and the bytes of its copies) at the limit and one past
        d = g.dense_i64()
        nout = 2 * len(d) - 1
        out = np.zeros((nout, 2), np.int64)
        gpu, refusal = mg._gpu(), mg._Refusal()
        for resource, value in (("output", nout), ("bytes", 16 * len(d) + 16 * nout)):
            args = (d.ctypes.data, len(d), d.ctypes.data, len(d), out.ctypes.data)
            gpu.run("tr_mgpu_gen_conv", *args, mg.C.byref(Budget(**{resource: value}).raw), mg.C.byref(refusal),
                    refusal=refusal)
            with self.assertRaises(AlgebraBudgetError) as ctx:
                gpu.run("tr_mgpu_gen_conv", *args, mg.C.byref(Budget(**{resource: value - 1}).raw),
                        mg.C.byref(refusal), refusal=refusal)
            self.assertEqual((ctx.exception.resource, ctx.exception.need), (resource, value))
        self.assertTrue(g.mul(g, backend="cuda").equal(g * g))
        pts = list(range(10))
        self.assertEqual(len(g.evaluate_many(pts, backend="cuda", budget=Budget(output=10))), 10)
        for backend in ("cpu", "cuda"):
            with self.assertRaises(AlgebraBudgetError):
                g.evaluate_many(pts, backend=backend, budget=Budget(output=9))

    def test_gen001_invalid_inputs_and_opt_in_import(self):
        s3 = mbk.Scope.from_arrays({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
        with self.assertRaises(UnsupportedError):
            mg.generating(md.histogram(s3, ["a", "b", "c"]))
        one = mg.from_terms({1: 1})
        two = mg.from_terms({(1, 1): 1})
        with self.assertRaises(ValueError):
            one * two                                                # different numbers of variables
        with self.assertRaises(ValueError):
            one.truncate(0)
        with self.assertRaises(ValueError):
            one((1, 2))                                              # one coordinate per variable
        with self.assertRaises(TypeError):
            one(0.5)                                                 # a float point is refused (inexact)
        with self.assertRaises(TypeError):
            mg.from_terms({1: 0.5})
        with self.assertRaises(ValueError):
            two.marginal(2)
        with self.assertRaises(TypeError):
            mg.generating(md.histogram(s3, ["a"]).prefix(degree=1))
        code = "import sys, torbles; print('torbles.math_gen' in sys.modules)"
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.split()
        self.assertEqual(out, ["False"])


if __name__ == "__main__":
    unittest.main()
