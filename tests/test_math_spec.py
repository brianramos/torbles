# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""SPEC-002: Walsh spectra and parity correlations of bounded Boolean histograms (torbles/math_spec.c,
math_spec_gpu.cu, math_spec.py).

Oracles, independent of the implementation: the Walsh-Hadamard sums by their definition with Python integers
(sum_x v(x) (-1)^|S and x| over every x, every S), the inverse and the truncation errors with Python Fractions, the
energies with Python integers, DuckDB SQL over the typed source table for the parity counts of condition subsets
(SUM of the condition values mod 2), and FIELD-001's own parity discovery (math_field.Bits.discover over the scope's bit
columns) for the constant parities. Every transform is also computed on the CUDA path (emulated here with
TORBLES_EMU_SCRAMBLE=1, a scrambled thread order) and compared for equality. Seeds are fixed.
"""
import itertools
import random
import unittest
from fractions import Fraction

import numpy as np
import pandas as pd

import torbles
from torbles import math_bank as mbk
from torbles import math_bool as mb
from torbles import math_field as mf
from torbles import math_handles as mh
from torbles import math_lattice as ml
from torbles import math_spec as ms
from torbles.math_budgets import Budget
from torbles.math_domains import DomainError
from torbles.math_errors import UnsupportedError
from torbles.mathematics import AlgebraBudgetError


def chi(S, x):
    return -1 if bin(S & x).count("1") % 2 else 1


def walsh_oracle(v, k):
    return [sum(v[x] * chi(S, x) for x in range(1 << k)) for S in range(1 << k)]


def inverse_oracle(W, k):
    return [Fraction(sum(W[S] * chi(S, x) for S in range(1 << k)), 1 << k) for x in range(1 << k)]


def obj(values):
    return pd.Series(values, dtype=object)


class Spec(unittest.TestCase):
    def forward(self, values, **kw):
        cpu = ms.walsh(values, **kw)
        gpu = ms.walsh(values, backend="cuda", **kw)
        self.assertEqual(cpu["spectrum"], gpu["spectrum"])
        self.assertEqual(cpu["cost"], gpu["cost"])
        self.assertIn(gpu["backend"], ("cuda", "cuda-emulated"))
        return cpu

    def backward(self, spectrum, **kw):
        cpu = ms.inverse(spectrum, **kw)
        gpu = ms.inverse(spectrum, backend="cuda", **kw)
        self.assertEqual(cpu["values"], gpu["values"])
        self.assertEqual(cpu["cost"], gpu["cost"])
        return cpu

    def test_spec002_forward_inverse_and_parseval_under_the_declared_normalization(self):
        rng = np.random.default_rng(2101)
        for k in (0, 1, 2, 3, 5, 8, 10):
            for kind in ("counts", "signed"):
                v = ([int(x) for x in rng.integers(0, 50, 1 << k)] if kind == "counts"
                     else [int(x) for x in rng.integers(-10 ** 15, 10 ** 15, 1 << k)])
                f = self.forward(v)
                W = f["spectrum"]
                self.assertEqual(f["normalization"], ms.FORWARD)
                if k <= 8:
                    self.assertEqual(W, walsh_oracle(v, k), (k, kind))
                inv = self.backward(W)
                self.assertEqual(inv["normalization"], ms.INVERSE)
                self.assertEqual(inv["values"], [Fraction(x) for x in v])     # the round trip, exactly
                p = ms.parseval(v, W)
                self.assertEqual((p["cells"], p["spectrum"]), (sum(x * x for x in v), sum(w * w for w in W)))
                self.assertTrue(p["holds"])
                self.assertEqual(p["spectrum"], (1 << k) * p["cells"])
                if kind == "counts" and sum(v):                # the probability normalization: divided by N^2
                    N = sum(v)
                    self.assertEqual(sum(Fraction(w, N) ** 2 for w in W),
                                     (1 << k) * sum(Fraction(x, N) ** 2 for x in v))
        # an arbitrary spectrum: the inverse is a dyadic rational vector, exactly 2^-k H W
        for k in (1, 3, 6):
            W = [2 * int(x) for x in rng.integers(-1000, 1000, 1 << k)]
            W[0] += 1                                              # an odd sum at every x: not an integer vector
            got = self.backward(W)["values"]
            self.assertEqual(got, inverse_oracle(W, k))
            self.assertTrue(all(x.denominator == 1 << k for x in got))
            self.assertTrue(all(x.denominator & (x.denominator - 1) == 0 for x in got))   # powers of two
            self.assertEqual(self.forward([int(x * (1 << k)) for x in got])["spectrum"], [w << k for w in W])
        tampered = self.forward([1, 2, 3, 4])["spectrum"]
        tampered[1] += 2
        self.assertFalse(ms.parseval([1, 2, 3, 4], tampered)["holds"])
        # 128-bit values: exact where they fit, refused (never wrapped) where they do not
        big = [(1 << 124) + i for i in range(4)]                  # |W| <= 2^126 + 6: fits
        self.assertEqual(self.forward(big)["spectrum"], walsh_oracle(big, 2))
        for backend in ("cpu", "cuda"):
            with self.assertRaises(OverflowError):
                ms.walsh([(1 << 126)] * 4, backend=backend)
            with self.assertRaises(OverflowError):
                ms.inverse([(1 << 126)] * 4, backend=backend)

    def test_spec002_histogram_spectra_and_parity_correlations_match_duckdb(self):
        nrng = np.random.default_rng(2102)
        n = 1500
        frame = pd.DataFrame({"x": obj([int(v) if r > 0.1 else None for v, r in zip(nrng.integers(-50, 51, n),
                                                                                       nrng.random(n))]),
                              "y": obj([int(v) for v in nrng.integers(-50, 51, n)]),
                              "z": obj([int(v) for v in nrng.integers(0, 10, n)])})
        db = torbles.from_ddl("CREATE TABLE t (x BIGINT, y BIGINT, z BIGINT)", {"t": frame})
        scope = mbk.Scope.from_table(db["t"], ["x", "y", "z"])
        conds = [mb.condition("x", ">", 0), mb.condition("y", "<=", 10), mb.condition("z", "bit", 0),
                 mb.condition("x", "<", 30), mb.condition("z", ">=", 7), mb.condition("y", "<>", 3)]
        csql = ['("x" > 0)', '("y" <= 10)', '(("z" % 2) = 1)', '("x" < 30)', '("z" >= 7)', '("y" <> 3)']
        k = len(conds)
        masks = mb.Masks.from_scope(scope, conds)
        h = ml.Histogram.build(masks)                              # the complete rows (x is NULL on some)
        f = self.forward(h)
        cells = h.counts().tolist()
        self.assertEqual(f["spectrum"], walsh_oracle(cells, k))
        complete = " AND ".join(f"({c}) IS NOT NULL" for c in csql)
        N = int(db.con.execute(f"SELECT COUNT(*) FROM t WHERE {complete}").fetchall()[0][0])
        sets = [list(S) for m in range(k + 1) for S in itertools.combinations(range(k), m)]
        full = ms.correlations(f["spectrum"], sets)
        direct = ms.coefficients(h, sets)                          # new frequencies, without the full transform
        self.assertEqual(full, direct)
        for S, r in zip(sets, full):
            parity = "(" + " + ".join(f"CAST({csql[i]} AS INTEGER)" for i in S) + ") % 2" if S else "0"
            even, odd = db.con.execute(f"SELECT COUNT(*) FILTER (WHERE {parity} = 0), COUNT(*) FILTER (WHERE "
                                       f"{parity} = 1) FROM t WHERE {complete}").fetchall()[0]
            self.assertEqual((r["even"], r["odd"]), (int(even), int(odd)), S)
            self.assertEqual(r["coefficient"], int(even) - int(odd))
            self.assertEqual(r["correlation"], Fraction(int(even) - int(odd), N))
            self.assertEqual(r["order"], len(S))
        # the spectrum of the TRUE sets of a three-valued histogram (SQL: UNKNOWN is not TRUE), every selected row
        h3 = ml.Histogram.build(masks, nulls="three")
        f3 = self.forward(h3, projection="true_sets")
        self.assertEqual(f3["spectrum"], walsh_oracle(h3.vector("true_sets").tolist(), k))
        self.assertEqual(f3["spectrum"][0], n)

    def test_spec002_full_domain_cost_and_truncation_is_an_explicit_approximation(self):
        for k in (3, 6, 10, 12):                                  # the cost is 2^k states whatever the occupancy
            cells = [0] * (1 << k)
            cells[5] = 7                                           # one occupied cell
            for values in (cells, ml.Histogram.from_counts(k, cells)):
                f = self.forward(values)
                self.assertEqual(f["cost"]["states"], 1 << k)
                self.assertEqual((f["cost"]["passes"], f["cost"]["butterflies"]), (k, k << (k - 1)))
                self.assertEqual(f["cost"]["bytes"], 16 << k)
                self.assertEqual(ms.inverse(f["spectrum"])["cost"]["states"], 1 << k)
            self.assertEqual(ms.walsh_need(k)["tensor"], 1 << k)
        rng = np.random.default_rng(2103)
        k = 6
        v = [int(x) for x in rng.integers(0, 20, 1 << k)]
        W = ms.walsh(v)["spectrum"]
        order = sorted(range(1 << k), key=lambda S: (-abs(W[S]), S))
        for m in (0, 1, 5, 17, 40, 1 << k):
            t = ms.truncate(W, m)
            kept = order[:m]
            self.assertEqual(t["kept_sets"], kept)
            self.assertEqual(t["kept_coefficients"], [W[S] for S in kept])
            dropped = [S for S in range(1 << k) if S not in set(kept)]
            residual = sum(W[S] ** 2 for S in dropped)
            self.assertEqual((t["residual_energy"], t["total_energy"]), (residual, sum(w * w for w in W)))
            self.assertEqual(t["states"], 1 << k)
            self.assertEqual(t["approximation"], any(W[S] for S in dropped))
            self.assertEqual(t["exact"], not t["approximation"])
            approx = ms.evaluate(t, range(1 << k))                 # the fixed-basis action at every point
            want = [Fraction(sum(W[S] * chi(S, x) for S in kept), 1 << k) for x in range(1 << k)]
            self.assertEqual(approx, want)
            err = [Fraction(x) - a for x, a in zip(v, approx)]
            self.assertEqual(t["l2_error_squared"], sum(e * e for e in err))          # Parseval, exactly
            self.assertEqual(t["l2_error_squared"], Fraction(residual, 1 << k))
            self.assertEqual(t["max_error_bound"], Fraction(sum(abs(W[S]) for S in dropped), 1 << k))
            self.assertLessEqual(max(abs(e) for e in err), t["max_error_bound"])
        self.assertTrue(ms.truncate(W, 1 << k)["exact"])
        sparse = [0] * 16
        sparse[3] = 4
        spec = ms.walsh(sparse)["spectrum"]
        t = ms.truncate(spec, 16)
        self.assertEqual((t["exact"], t["dropped"]), (True, 0))
        with self.assertRaises(ValueError):
            ms.truncate(W, (1 << k) + 1)

    def test_spec002_constant_parities_are_field001_parity_dependencies(self):
        nrng = np.random.default_rng(2104)
        n = 2000
        a, b, d = (nrng.integers(0, 2, n) for _ in range(3))
        g = nrng.integers(0, 2, n)
        cols = {"a": a, "b": b, "c": a ^ b, "d": d, "e": 1 - d, "f": np.ones(n, dtype=np.int64), "g": g}
        frame = pd.DataFrame({name: obj([int(x) for x in v]) for name, v in cols.items()})
        names = list(cols)
        db = torbles.from_ddl("CREATE TABLE t (" + ", ".join(f"{c} BIGINT" for c in names) + ")", {"t": frame})
        for sel in (np.ones(n, bool), nrng.random(n) < 0.05):
            scope = mbk.Scope.from_table(db["t"], names, rows=sel)
            masks = mb.Masks.from_scope(scope, [mb.condition(c, "bit", 0) for c in names])
            h = ml.Histogram.build(masks)
            got = ms.constant_parities(h)
            self.assertTrue(got["agree"])
            W = ms.walsh(h)["spectrum"]
            N = int(sel.sum())
            self.assertEqual(got["total"], N)
            want = [(tuple(i for i in range(7) if (S >> i) & 1), 0 if W[S] == N else 1)
                    for S in range(1 << 7) if abs(W[S]) == N]
            self.assertEqual(got["relations"], want)
            # FIELD-001's discovery over the same bits: the span of its relations is exactly the constant set
            bits = mf.Bits.from_scope(scope, [(c, "bit", 0) for c in names])
            disc = bits.discover(sample=0, affine=True)
            span = set()
            basis = [(sum(1 << j for j in r["vars"]), r["constant"]) for r in disc["relations"]]
            for choice in itertools.product((0, 1), repeat=len(basis)):
                vars_, c = 0, 0
                for take, (bv, bc) in zip(choice, basis):
                    if take:
                        vars_ ^= bv
                        c ^= bc
                span.add((tuple(i for i in range(7) if (vars_ >> i) & 1), c))
            self.assertEqual(span, set(want))
            self.assertEqual(len(got["basis"]), len(disc["relations"]))
            self.assertEqual(got["rank"], disc["rank"])
            # the planted relations are there
            for planted in (((0, 1, 2), 0), ((3, 4), 1), ((5,), 1)):
                self.assertIn(planted, set(want))
        with self.assertRaises(DomainError):                       # weights only (a signed vector is refused)
            ms.constant_parities([1, -1, 0, 0])
        with self.assertRaises(DomainError):                       # no points: vacuous
            ms.constant_parities([0, 0])

    def test_spec002_budgets_at_the_limit_and_one_past(self):
        k = 10
        v = list(range(1 << k))
        need = ms.walsh_need(k)
        self.assertEqual((need["variables"], need["tensor"], need["iterations"], need["bytes"]), (10, 1024, 10, 16384))
        for backend in ("cpu", "cuda"):
            for fn in (ms.walsh, ms.inverse):
                for resource in ("tensor", "iterations", "bytes"):
                    fn(v, backend=backend, budget=Budget(**{resource: need[resource]}))
                    with self.assertRaises(AlgebraBudgetError) as ctx:
                        fn(v, backend=backend, budget=Budget(**{resource: need[resource] - 1}))
                    self.assertEqual((ctx.exception.resource, ctx.exception.need), (resource, need[resource]))
        sets = [[0], [1, 2]]
        cn = ms.coefficients_need(v, sets)
        self.assertEqual((cn["tensor"], cn["output"]), (2 << k, 2))
        ms.coefficients(v, sets, budget=Budget(tensor=cn["tensor"]))
        with self.assertRaises(AlgebraBudgetError):
            ms.coefficients(v, sets, budget=Budget(tensor=cn["tensor"] - 1))
        W = ms.walsh(v)["spectrum"]
        ms.truncate(W, 5, budget=Budget(output=5))
        with self.assertRaises(AlgebraBudgetError):
            ms.truncate(W, 5, budget=Budget(output=4))
        t = ms.truncate(W, 4)
        ms.evaluate(t, range(10), budget=Budget(tensor=40))
        with self.assertRaises(AlgebraBudgetError):
            ms.evaluate(t, range(10), budget=Budget(tensor=39))
        ms.constant_parities(v, budget=Budget(output=1024))
        with self.assertRaises(AlgebraBudgetError):
            ms.constant_parities(v, budget=Budget(output=1023))
        with self.assertRaises(ValueError):
            ms.walsh([1, 2, 3])
        with self.assertRaises(ValueError):
            ms.correlations(W, [[10]])
        with self.assertRaises(UnsupportedError):
            ml._k_of(1 << 31)

    def test_spec002_ctypes_structures_are_registered(self):
        report = mh.abi_check()
        mine = {r["native"] for r in report["checked"] if r["structure"].startswith("torbles.math_spec.")}
        self.assertEqual(mine, {"tr_spec_dyadic", "tr_spec_cost", "tr_spec_energy", "tr_spec_parity",
                                "tr_spec_truncation", "tr_spec_relation", "tr_spec_parities"})
        cuda = {r["native"] for r in report["checked"] if r["library"] == "cuda"
                and r["structure"].startswith("torbles.math_spec.")}
        self.assertEqual(cuda, {"tr_spec_dyadic", "tr_spec_cost"})
        self.assertTrue(all(r["match"] for r in report["checked"]))


if __name__ == "__main__":
    unittest.main()
