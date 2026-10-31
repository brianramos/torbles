# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""DIST-003 (entropy, mutual information and contingency transforms) (torbles/math_info.c, math_info_gpu.cu,
math_info.py).

Oracles, independent of the implementation: explicit normalized joint tables written in the tests (Python Fractions
p = count / N, structural zeros as absent cells, NULL as a category or dropped), DuckDB GROUP BY counts of the same rows;
the textbook forms of every statistic over them: -sum p ln p, sum p ln(p / (p_a p_b)), Pearson's sum (O - E)^2 / E over
the full product of the marginal supports (structural zeros included) as an exact Fraction, 2 sum O ln(O / E). The
transcendental values are checked twice: against 60-digit decimal logarithms (Decimal.ln) the certified enclosures must
contain exactly, and against math.log doubles they must contain up to the double result's own rounding; enclosure
widths are checked against the stated bounds. Every state is also built with the CUDA kernels (emulated here,
TORBLES_EMU_SCRAMBLE=1 scrambles the thread order) and compared for equality with the CPU state. Seeds are fixed.
"""
import itertools
import math
import subprocess
import sys
import unittest
from decimal import Decimal, getcontext
from fractions import Fraction

import duckdb
import numpy as np
import pandas as pd

from torbles import math_bank as mbk, math_dist as md, math_info as mi
from torbles.math_budgets import Budget
from torbles.math_domains import DomainError
from torbles.math_errors import UnsupportedError
from torbles.mathematics import AlgebraBudgetError

getcontext().prec = 60
NAMES = ("x", "y", "z")


# ---------------------------------------------------------------------- explicit tables and the oracles
def rows_of(tables, rng):
    """rows (group, x, y, z) of explicit joint tables {group: {(x, y, z): count}} (None = NULL), shuffled"""
    out = []
    for g, t in tables.items():
        for key, c in t.items():
            out += [(g,) + key] * c
    idx = rng.permutation(len(out))
    return [out[i] for i in idx]


def scope_of(rows, G, weights=None):
    cols = {n: [0 if r[1 + i] is None else r[1 + i] for r in rows] for i, n in enumerate(NAMES)}
    valid = {n: [r[1 + i] is not None for r in rows] for i, n in enumerate(NAMES)}
    if weights is not None:
        cols["w"] = weights
    return mbk.Scope.from_arrays(cols, valid=valid, groups=[r[0] for r in rows], ngroups=G)


def table(rows, g, coords, exclude=(), weights=None):
    """the joint table of group g over coordinate positions coords; rows with NULL in an excluded position are dropped"""
    t = {}
    for i, r in enumerate(rows):
        if r[0] != g or any(r[1 + e] is None for e in exclude):
            continue
        m = 1 if weights is None else weights[i]
        if not m:
            continue
        key = tuple(r[1 + c] for c in coords)
        t[key] = t.get(key, 0) + m
    return t


def marginal(t, positions):
    out = {}
    for key, m in t.items():
        k = tuple(key[p] for p in positions)
        out[k] = out.get(k, 0) + m
    return out


def h_dec(t):
    N = sum(t.values())
    return Decimal(N).ln() - sum(Decimal(m) * Decimal(m).ln() for m in t.values()) / Decimal(N)


def h_float(t):
    N = sum(t.values())
    return -sum((m / N) * math.log(m / N) for m in t.values())


def mi_float(t, pa, pb):
    """sum p(a, b) ln(p(a, b) / (p(a) p(b))) over the joint table t (positions pa, pb of its keys)"""
    N = sum(t.values())
    A, B = marginal(t, pa), marginal(t, pb)
    s = 0.0
    for key, m in t.items():
        a, b = tuple(key[p] for p in pa), tuple(key[p] for p in pb)
        s += (m / N) * math.log(m * N / (A[a] * B[b]))
    return s


def chi2_exact(t, pa, pb):
    """Pearson's X^2 over the full product of the marginal supports, structural zeros included (Fractions)"""
    N = sum(t.values())
    A, B = marginal(t, pa), marginal(t, pb)
    full = {(tuple(k[p] for p in pa), tuple(k[p] for p in pb)): m for k, m in t.items()}
    x2 = Fraction(0)
    for a, b in itertools.product(A, B):
        E = Fraction(A[a] * B[b], N)
        x2 += (full.get((a, b), 0) - E) ** 2 / E
    return x2, len(A), len(B)


def g_float(t, pa, pb):
    N = sum(t.values())
    A, B = marginal(t, pa), marginal(t, pb)
    return 2 * sum(m * math.log(m * N / (A[tuple(k[p] for p in pa)] * B[tuple(k[p] for p in pb)]))
                   for k, m in t.items())


def g_dec(t, pa, pb):
    N = sum(t.values())
    A, B = marginal(t, pa), marginal(t, pb)
    return 2 * sum(Decimal(m) * (Decimal(m * N) / Decimal(A[tuple(k[p] for p in pa)] * B[tuple(k[p] for p in pb)])).ln()
                   for k, m in t.items())


def contains(test, r, exact_dec, approx, what, bits=False, width=Fraction(1, 2 ** 104)):
    """the enclosure contains the 60-digit value exactly and the math.log double up to its own rounding; tight (the
    exact rational bounds at most `width` apart: 2^-104 for H and I, 2^-101 N for G)"""
    test.assertFalse(r["empty"], what)
    lo, hi = r["lo"], r["hi"]
    test.assertLessEqual(Decimal(lo), exact_dec, what)
    test.assertGreaterEqual(Decimal(hi), exact_dec, what)
    tol = 1e-12 * max(1.0, abs(approx))
    test.assertTrue(lo - tol <= approx <= hi + tol, (what, lo, approx, hi))
    v = max(abs(lo), abs(hi))
    test.assertLessEqual(hi - lo, 2.0 ** -100 + (2.0 ** -50 if bits else 2.0 ** -51) * v * 2, (what, lo, hi))
    if not bits:                                       # the exact rational bounds contain the value too
        b0, b1 = r["bounds"]
        test.assertLessEqual(Decimal(b0.numerator) / Decimal(b0.denominator), exact_dec + Decimal(10) ** -50, what)
        test.assertGreaterEqual(Decimal(b1.numerator) / Decimal(b1.denominator), exact_dec - Decimal(10) ** -50, what)
        test.assertLessEqual(b1 - b0, width, what)


def random_tables(rng, G, zero=0.4, null=True):
    xs, ys, zs = [-1, 0, 3, None] if null else [-1, 0, 3], [0, 1, 2, 5, None] if null else [0, 1, 2, 5], [0, 1]
    out = {}
    for g in range(G):
        t = {}
        for key in itertools.product(xs, ys, zs):
            if rng.random() < zero:
                continue                                   # a structural zero: the cell is absent
            t[key] = int(rng.integers(1, 9))
        out[g] = t
    return out


class Base(unittest.TestCase):
    def both(self, h, **kw):
        cpu = mi.information(h, **kw)
        gpu = mi.information(h, backend="cuda", **kw)
        self.assertTrue(cpu.equal(gpu))
        self.assertIn(gpu.backend, ("cuda", "cuda-emulated"))
        return cpu


class Information(Base):
    # ------------------------------------------------------------------ acceptance 1: explicit normalized tables
    def test_dist003_entropy_and_mi_match_explicit_normalized_joint_tables_with_structural_zeros(self):
        rng = np.random.default_rng(3001)
        G = 4
        tables = random_tables(rng, G - 1)                  # group G - 1 is absent
        rows = rows_of(tables, rng)
        scope = scope_of(rows, G)
        for repr_ in ("sparse", "dense"):
            h = md.histogram(scope, list(NAMES), repr=repr_)
            inf = self.both(h)
            if repr_ == "dense":
                self.assertTrue(inf.equal(mi.information(md.histogram(scope, list(NAMES)))))   # zeros are skipped
            for g in range(G - 1):
                t = table(rows, g, (0, 1, 2))
                N = sum(t.values())
                self.assertEqual(inf.group(g)["total"], N)
                # entropies of every marginal (NULL a category): H = -sum p ln p, p = count / N
                for S in range(1, 8):
                    pos = [i for i in range(3) if (S >> i) & 1]
                    names = [NAMES[i] for i in pos]
                    m = marginal(t, pos)
                    self.assertEqual(inf.group(g)["cells"][tuple(names)], len(m))
                    for units in ("nats", "bits"):
                        r = inf.entropy(names, group=g, units=units)
                        scale = Decimal(2).ln() if units == "bits" else Decimal(1)
                        contains(self, r, h_dec(m) / scale, h_float(m) / (math.log(2) if units == "bits" else 1),
                                 (g, names, units), bits=units == "bits")
                # conditional entropy, mutual information (as sum p ln(p / (p_a p_b))), conditional MI
                for a, b in (((0,), (1,)), ((1,), (2,)), ((0,), (1, 2)), ((0, 2), (1,))):
                    an, bn = [NAMES[i] for i in a], [NAMES[i] for i in b]
                    tab = marginal(t, sorted(a + b))
                    pa = [sorted(a + b).index(i) for i in a]
                    pb = [sorted(a + b).index(i) for i in b]
                    hab, hb = h_dec(tab), h_dec(marginal(tab, pb))
                    contains(self, inf.conditional_entropy(an, bn, group=g), hab - hb,
                             h_float(tab) - h_float(marginal(tab, pb)), (g, "H|", a, b))
                    mid = h_dec(marginal(tab, pa)) + hb - hab
                    contains(self, inf.mutual_information(an, bn, group=g), mid, mi_float(tab, pa, pb), (g, "I", a, b))
                    contains(self, inf.g_test(an, bn, group=g), g_dec(tab, pa, pb), g_float(tab, pa, pb),
                             (g, "G", a, b), width=Fraction(N, 2 ** 101))
                    # Pearson X^2 exactly (textbook form over every cell of T(a) x T(b), structural zeros included)
                    x2, rcount, ccount = chi2_exact(tab, pa, pb)
                    r = inf.chi_square(an, bn, group=g, exact=True)
                    self.assertEqual(r["exact"], x2)
                    self.assertEqual((r["rows"], r["cols"], r["df"]), (rcount, ccount, (rcount - 1) * (ccount - 1)))
                    self.assertTrue(r["bounds"][0] <= x2 <= r["bounds"][1])
                    self.assertTrue(Fraction(r["lo"]) <= x2 <= Fraction(r["hi"]))
                    self.assertLessEqual(r["bounds"][1] - r["bounds"][0], Fraction(N * len(tab), 2 ** 127))
                    v = inf.cramers_v(an, bn, group=g, exact=True)
                    v2 = x2 / (N * (min(rcount, ccount) - 1))
                    self.assertEqual(v["exact_v2"], v2)
                    vd = (Decimal(v2.numerator) / Decimal(v2.denominator)).sqrt()
                    self.assertTrue(Decimal(v["lo"]) <= vd <= Decimal(v["hi"]))
                    self.assertLessEqual(v["hi"] - v["lo"], 2.0 ** -48)
                # conditional MI: I(x; y | z) = H(x, z) + H(y, z) - H(x, y, z) - H(z)
                cmi = h_dec(marginal(t, [0, 2])) + h_dec(marginal(t, [1, 2])) - h_dec(t) - h_dec(marginal(t, [2]))
                cmf = h_float(marginal(t, [0, 2])) + h_float(marginal(t, [1, 2])) - h_float(t) - h_float(marginal(t, [2]))
                contains(self, inf.mutual_information("x", "y", given="z", group=g), cmi, cmf, (g, "I|"))
            for op in ("entropy", "mutual_information", "chi_square"):    # the absent group: no answer
                q = dict(op=op, a=["x"], b=None if op == "entropy" else ["y"], group=G - 1)
                self.assertTrue(inf.query([q])[0]["empty"])

    def test_dist003_counts_come_from_duckdb_group_by(self):
        rng = np.random.default_rng(3002)
        n = 3000
        x = rng.integers(0, 6, n); y = rng.integers(0, 4, n); g = rng.integers(0, 3, n)
        y[(x == 2) & (y == 1)] = 3                                # a structural zero
        vx = rng.random(n) > 0.1
        scope = mbk.Scope.from_arrays({"x": x, "y": y}, valid={"x": vx}, groups=g, ngroups=3)
        inf = self.both(md.histogram(scope, ["x", "y"]))
        con = duckdb.connect()
        con.register("t", pd.DataFrame({"g": g, "x": pd.Series(np.where(vx, x, 0), dtype="Int64").where(vx), "y": y}))
        for k in range(3):
            t = {(a, b): int(c) for a, b, c in con.execute(f"SELECT x, y, COUNT(*) FROM t WHERE g = {k} GROUP BY x, y")
                 .fetchall()}
            self.assertNotIn((2, 1), t)
            contains(self, inf.entropy(["x", "y"], group=k), h_dec(t), h_float(t), (k, "H(x, y)"))
            contains(self, inf.mutual_information("x", "y", group=k), h_dec(marginal(t, [0])) + h_dec(marginal(t, [1]))
                     - h_dec(t), mi_float(t, [0], [1]), (k, "I"))
            self.assertEqual(inf.chi_square("x", "y", group=k, exact=True)["exact"], chi2_exact(t, [0], [1])[0])
        con.close()

    def test_dist003_null_categories_are_included_or_excluded_as_declared(self):
        rng = np.random.default_rng(3003)
        tables = random_tables(rng, 2, zero=0.3)
        rows = rows_of(tables, rng)
        h = md.histogram(scope_of(rows, 2), list(NAMES))
        for exclude, positions in (((), ()), (["x"], (0,)), (["y"], (1,)), ("all", (0, 1, 2))):
            inf = self.both(h, exclude=exclude)
            self.assertEqual(inf.info["excluded"], [NAMES[p] for p in positions])
            for g in range(2):
                t = table(rows, g, (0, 1, 2), exclude=positions)
                self.assertEqual(inf.group(g)["total"], sum(t.values()))
                for names, pos in ((["x"], [0]), (["y"], [1]), (["x", "y"], [0, 1])):
                    m = marginal(t, pos)
                    if set(pos) <= set(positions):            # every coordinate excluded: no NULL category left
                        self.assertFalse(any(None in k for k in m))
                    self.assertEqual(inf.group(g)["cells"][tuple(names)], len(m))
                    contains(self, inf.entropy(names, group=g), h_dec(m), h_float(m), (exclude, g, names))
                x2 = chi2_exact(marginal(t, [0, 1]), [0], [1])[0]
                self.assertEqual(inf.chi_square("x", "y", group=g, exact=True)["exact"], x2)
        # NULL as a category versus dropped: the category changes H(x) and N
        a, b = mi.information(h), mi.information(h, exclude=["x"])
        self.assertGreater(a.group(0)["total"], b.group(0)["total"])
        self.assertNotEqual(a.entropy("x")["lo"], b.entropy("x")["lo"])
        with self.assertRaises(ValueError):
            mi.information(h, exclude=["w"])

    def test_dist003_structural_zeros_degenerate_tables_and_presence(self):
        # an exactly independent table (counts a_i b_j): I = 0 inside the enclosure, X^2 = 0 exactly
        t = {(x, y, 0): (x + 1) * (y + 2) for x in range(3) for y in range(4)}
        # one cell only: H = 0, no contingency (V empty); a group whose every cell has a NULL x, excluded: N = 0
        tables = {0: t, 1: {(5, 7, 0): 9}, 2: {(None, 1, 0): 3, (None, 2, 1): 4}}
        rng = np.random.default_rng(3004)
        rows = rows_of(tables, rng)
        h = md.histogram(scope_of(rows, 4), list(NAMES))
        inf = self.both(h, exclude=["x"])
        r = inf.mutual_information("x", "y")
        self.assertEqual(r["lo"], 0.0)
        self.assertLess(r["hi"], 1e-25)
        self.assertEqual(inf.chi_square("x", "y", exact=True)["exact"], 0)
        self.assertEqual(inf.chi_square("x", "y")["lo"], 0.0)
        one = inf.entropy(["x", "y", "z"], group=1)
        self.assertEqual(one["lo"], 0.0)
        self.assertLess(one["hi"], 1e-25)
        c = inf.chi_square("x", "y", group=1, exact=True)
        self.assertEqual((c["exact"], c["df"], c["rows"], c["cols"]), (0, 0, 1, 1))
        v = inf.cramers_v("x", "y", group=1, exact=True)
        self.assertTrue(v["empty"])
        self.assertIsNone(v["exact_v2"])
        for g in (2, 3):                                     # N = 0 (excluded), absent
            for q in (dict(op="entropy", a="y"), dict(op="chi_square", a="x", b="y"), dict(op="g_test", a="x", b="y")):
                self.assertTrue(inf.query([dict(q, group=g)])[0]["empty"])
            self.assertIsNone(inf.chi_square("x", "y", group=g, exact=True)["exact"])
        self.assertEqual(inf.group(2), dict(rows=7, total=0, cells={k: 0 for k in inf.group(2)["cells"]}))
        self.assertEqual(inf.group(3)["rows"], 0)

    # ------------------------------------------------------------------ acceptance 2: no inferred joint distribution
    def test_dist003_never_infers_a_joint_distribution_from_marginals_or_moments(self):
        # equal marginals, different dependence: the joint support decides
        dep = {(0, 0, 0): 5, (1, 1, 0): 5}
        ind = {(0, 0, 0): 2, (0, 1, 0): 3, (1, 0, 0): 3, (1, 1, 0): 2}
        rng = np.random.default_rng(3005)
        rows = rows_of({0: dep, 1: ind}, rng)
        h = md.histogram(scope_of(rows, 2), list(NAMES))
        inf = self.both(h)
        for col in ("x", "y"):
            self.assertEqual(inf.entropy(col, group=0)["bounds"], inf.entropy(col, group=1)["bounds"])
        self.assertGreater(inf.mutual_information("x", "y", group=0)["lo"], 0.69)          # ln 2
        self.assertLess(inf.mutual_information("x", "y", group=1)["hi"], 0.03)
        self.assertEqual(inf.chi_square("x", "y", group=0, exact=True)["exact"], 10)
        self.assertEqual(inf.chi_square("x", "y", group=1, exact=True)["exact"], Fraction(2, 5))
        # a state that holds x alone: statistics that need (x, y) jointly are refused, natively and in Python
        hx = md.histogram(scope_of(rows, 2), ["x"])
        hy = md.histogram(scope_of(rows, 2), ["y"])
        ix = mi.information(hx)
        for fn in (lambda: ix.mutual_information("x", "y"), lambda: ix.chi_square("x", "y"),
                   lambda: ix.conditional_entropy("x", "y"), lambda: ix.entropy(["x", "y"]),
                   lambda: ix.chi_square("x", "y", exact=True)):
            with self.assertRaises(UnsupportedError):
                fn()
        raw = ix.requests([dict(op="mutual_information", a="x", b="x")])
        raw["b"][0] = 2                                       # a coordinate the state does not hold
        self.assertEqual(int(ix.query_arrays(raw)["status"][0]), -12)
        raw["op"][0], raw["b"][0] = 5, 2
        self.assertEqual(int(ix.query_arrays(raw)["status"][0]), -12)
        e = mi._api()["tr_info_exact"]
        num, den = mi.vp(), mi.vp()
        self.assertEqual(e(ix._support, 0, 5, 1, 2, None, None, C_ref(num), C_ref(den)), -12)
        # marginal histograms, banks of moments and prefix states are not joint support states
        for other in (hy.prefix(degree=1), mbk.build(mbk.Scope.from_arrays({"x": [1, 2]}), ["x", "x^2"]), [hx, hy]):
            with self.assertRaises(TypeError):
                mi.information(other)
        self.assertEqual(ix.entropy("x", group=0)["bounds"], inf.entropy("x", group=0)["bounds"])

    # ------------------------------------------------------------------ the certified logarithm, weights, CUDA
    def test_dist003_certified_logarithm_contains_high_precision_values(self):
        rng = np.random.default_rng(3006)
        ms = list(range(1, 300)) + [2 ** k + d for k in range(1, 64) for d in (-1, 0, 1) if 1 <= 2 ** k + d < 2 ** 64]
        ms += [int(v) for v in rng.integers(1, 2 ** 63, 300, dtype=np.int64)] + [2 ** 64 - 1, 3 ** 40, 10 ** 19]
        two120 = Decimal(2) ** 120
        for m in ms:
            lo, hi = mi.ln_bounds(m)
            exact = Decimal(m).ln() * two120
            self.assertTrue(Decimal(lo) <= exact <= Decimal(hi), m)
            self.assertLessEqual(hi - lo, 2 * 40 + 65, m)
        self.assertEqual(mi.ln_bounds(1), (0, 0))
        with self.assertRaises(ValueError):
            mi.ln_bounds(0)

    def test_dist003_per_cell_term_bounds_are_exact_floors_and_ceilings(self):
        rng = np.random.default_rng(3011)
        term = mi.algq.lib.tr_info_term_bounds                   # the test hook of the inline per-cell terms
        term.restype, term.argtypes = mi.i32, [mi.i32, mi.u64, mi.u64, mi.u64, mi.P(mi.u64)]

        def bounds(kind, a, b=0, c=0):
            out = (mi.u64 * 8)()
            self.assertEqual(term(kind, a, b, c, out), 0)
            w = [int(v) for v in out]
            return sum(w[i] << (64 * i) for i in range(4)), sum(w[4 + i] << (64 * i) for i in range(4))
        # chi-square cells: exactly floor and ceil of O^2 2^128 / (m_a m_b), both division paths (m_a m_b below and
        # past 2^64) and their edges
        triples = [(1, 1, 1), (1, 1, 2), (2, 3, 2), (3, 7, 5), (2 ** 32 - 1, 2 ** 32 - 1, 2 ** 32 - 1),
                   (5, 2 ** 32, 2 ** 32 - 1), (1, 2 ** 63, 2), (2 ** 63, 2 ** 63, 2 ** 64 - 1), (7, 2 ** 64 - 1, 2 ** 64 - 1),
                   (2 ** 64 - 1, 2 ** 64 - 1, 2 ** 64 - 1), (12345, 2 ** 40 + 3, 2 ** 24 + 1)]
        for _ in range(3000):
            ma, mb = (int(v) for v in rng.integers(1, 2 ** 62, 2, dtype=np.int64) >> rng.integers(0, 60, 2))
            ma, mb = max(ma, 1), max(mb, 1)
            triples.append((int(rng.integers(1, min(ma, mb) + 1)), ma, mb))
        for o, ma, mb in triples:
            lo, hi = bounds(2, o, ma, mb)
            num, den = o * o << 128, ma * mb
            self.assertEqual((lo, hi), (num // den, -(-num // den)), (o, ma, mb))
        # log cells: lo <= m ln(m) 2^112 <= hi, at most m 144 / 2^8 + 2 apart
        two112 = Decimal(2) ** 112
        for m in [1, 2, 3, 255, 256, 257, 2 ** 32 + 1, 2 ** 63, 2 ** 64 - 1] + \
                 [int(v) for v in rng.integers(1, 2 ** 62, 400, dtype=np.int64)]:
            lo, hi = bounds(1, m)
            exact = Decimal(m) * Decimal(m).ln() * two112
            self.assertTrue(Decimal(lo) <= exact <= Decimal(hi), m)
            self.assertLessEqual(hi - lo, m * 144 // 256 + 2, m)
        with self.assertRaises(AssertionError):
            bounds(2, 5, 3, 9)                                   # O > m_a: not a cell

    def test_dist003_weight_masses_and_their_scope(self):
        rng = np.random.default_rng(3007)
        tables = random_tables(rng, 2, zero=0.3, null=False)
        rows = rows_of(tables, rng)
        weights = [int(v) for v in rng.integers(0, 50, len(rows))]                 # zero weights leave the support
        h = md.histogram(scope_of(rows, 2, weights), list(NAMES), weight="w")
        inf = self.both(h, mass="weight")
        for g in range(2):
            t = table(rows, g, (0, 1, 2), weights=weights)
            contains(self, inf.entropy(["x", "y"], group=g), h_dec(marginal(t, [0, 1])), h_float(marginal(t, [0, 1])),
                     (g, "weighted H"))
            self.assertEqual(inf.chi_square("x", "z", group=g, exact=True)["exact"],
                             chi2_exact(marginal(t, [0, 2]), [0], [1])[0])
        # huge masses (cells near 2^42: m_a m_b past 2^64, logarithms of large arguments), checked the same way
        hw = [int(v) for v in rng.integers(1, 1 << 40, len(rows))]
        hh = md.histogram(scope_of(rows, 2, hw), list(NAMES), weight="w")
        big_inf = self.both(hh, mass="weight")
        for g in range(2):
            t = table(rows, g, (0, 1, 2), weights=hw)
            for pos in ([0], [0, 1], [0, 1, 2]):
                m = marginal(t, pos)
                contains(self, big_inf.entropy([NAMES[p] for p in pos], group=g), h_dec(m), h_float(m), (g, pos))
            xy = marginal(t, [0, 1])
            contains(self, big_inf.mutual_information("x", "y", group=g), h_dec(marginal(xy, [0])) +
                     h_dec(marginal(xy, [1])) - h_dec(xy), mi_float(xy, [0], [1]), (g, "I big"))
            x2 = chi2_exact(xy, [0], [1])[0]
            r = big_inf.chi_square("x", "y", group=g, exact=True)
            self.assertEqual(r["exact"], x2)
            self.assertTrue(r["bounds"][0] <= x2 <= r["bounds"][1])
        # weights replicate in units: a weight-2 row equals two rows
        doubled = [r for r, w in zip(rows, weights) for _ in range(w)]
        hc = md.histogram(scope_of(doubled, 2), list(NAMES))
        self.assertEqual(mi.information(hc).entropy("y")["bounds"], inf.entropy("y")["bounds"])
        # a weight total at or past 2^64 is outside the scope; negative weights are no mass; unweighted: no weights
        big = mbk.Scope.from_arrays({"x": [1, 2, 3, 4], "w": [1 << 62] * 4})
        with self.assertRaises(UnsupportedError):
            mi.information(md.histogram(big, ["x"], weight="w"), mass="weight")
        ok = mbk.Scope.from_arrays({"x": [1, 2, 3], "w": [1 << 62] * 3})
        r = mi.information(md.histogram(ok, ["x"], weight="w"), mass="weight").entropy("x")
        self.assertTrue(Decimal(r["lo"]) <= Decimal(3).ln() <= Decimal(r["hi"]))
        neg = mbk.Scope.from_arrays({"x": [1, 2], "w": [3, -1]})
        with self.assertRaises(DomainError):
            mi.information(md.histogram(neg, ["x"], weight="w"), mass="weight")
        with self.assertRaises(ValueError):
            mi.information(md.histogram(ok, ["x"]), mass="weight")

    def test_dist003_cuda_sums_equal_cpu_on_many_groups_and_near_row_sized_support(self):
        rng = np.random.default_rng(3008)
        n, G = 20000, 37
        cols = {"x": rng.integers(0, 40, n), "y": rng.integers(-5, 5, n), "z": rng.integers(0, 3, n)}
        valid = {"y": rng.random(n) > 0.05}
        scope = mbk.Scope.from_arrays(cols, valid=valid, groups=rng.integers(0, G, n), ngroups=G)
        for exclude in ((), ["y"]):
            inf = self.both(md.histogram(scope, ["x", "y", "z"]), exclude=exclude)
            self.assertEqual(inf.info["groups"], G)
        near = mbk.Scope.from_arrays({"x": rng.integers(0, 10 ** 12, n), "y": rng.integers(0, 10 ** 12, n)})
        h = md.histogram(near, ["x", "y"])                                         # about one cell per row
        inf = self.both(h)
        self.assertGreater(inf.info["joint_cells"], 0.99 * n)
        r = inf.mutual_information("x", "y")
        mid = Decimal(n).ln()                                                      # every value distinct: I = ln n
        if inf.group(0)["cells"][("x",)] == n and inf.group(0)["cells"][("y",)] == n:
            self.assertTrue(Decimal(r["lo"]) <= mid <= Decimal(r["hi"]))
        items = [dict(op=op, a="x", b=None if op == "entropy" else "y") for op in ("entropy", "mutual_information",
                                                                                  "chi_square", "g_test", "cramers_v")]
        self.assertTrue(all(not q["empty"] for q in inf.query(items)))

    def test_dist003_budgets_and_refusals_at_the_limit(self):
        rng = np.random.default_rng(3009)
        tables = random_tables(rng, 3, zero=0.2)
        rows = rows_of(tables, rng)
        h = md.histogram(scope_of(rows, 3), list(NAMES))
        ref = mi.information(h)
        first = mi.need(h)
        info = ref.info
        self.assertEqual((first["variables"], first["states"]), (3, info["joint_cells"]))
        # stage 1 (before the scratch): one past each need is refused with that need
        for resource, value in first.items():
            with self.assertRaises(AlgebraBudgetError) as ctx:
                mi.information(h, budget=Budget(**{resource: value - 1}))
            self.assertEqual((ctx.exception.resource, ctx.exception.need), (resource, value))
        self.assertTrue(mi.information(h, budget=Budget(variables=3)).equal(ref))
        # stage 2 (once the tables are counted): states = table + contingency cells, support = the largest table of a
        # group, bytes = scratch + support; at the limit it runs, one past it is refused
        stage2 = dict(states=info["table_cells"] + info["contingency_cells"], support=info["max_cells"])
        with self.assertRaises(AlgebraBudgetError) as ctx:
            mi.information(h, budget=Budget(bytes=first["bytes"]))
        stage2["bytes"] = ctx.exception.need
        self.assertGreater(stage2["bytes"], first["bytes"])
        self.assertEqual(stage2["bytes"] - first["bytes"], info["support_bytes"])
        for resource, value in stage2.items():
            self.assertTrue(mi.information(h, budget=Budget(**{resource: value})).equal(ref))
            with self.assertRaises(AlgebraBudgetError) as ctx:
                mi.information(h, budget=Budget(**{resource: value - 1}))
            self.assertEqual((ctx.exception.resource, ctx.exception.need), (resource, value))
        # the device copies of the sums (the support built first, then the kernels under their own budget)
        view = ref.view()
        dev = int(mi._api()["tr_info_device_need"](mi.C.byref(view)))
        self.assertEqual(dev, 16 * view.nlog + 32 * view.nchi + 64 * view.ngroups * view.nslots)
        gpu = mi._gpu()
        sums = np.zeros(int(view.ngroups) * int(view.nslots) * 8, dtype=np.int64)
        refusal = mi._Refusal()
        gpu.run("tr_mgpu_info_sums", mi.C.byref(view), mi.C.byref(Budget(bytes=dev).raw), sums.ctypes.data,
                mi.C.byref(refusal))
        cpu_sums = np.zeros_like(sums)
        self.assertEqual(mi._api()["tr_info_sums"](mi.C.byref(view), cpu_sums.ctypes.data), 0)
        self.assertTrue(np.array_equal(sums, cpu_sums))
        with self.assertRaises(AlgebraBudgetError) as ctx:
            gpu.run("tr_mgpu_info_sums", mi.C.byref(view), mi.C.byref(Budget(bytes=dev - 1).raw), sums.ctypes.data,
                    mi.C.byref(refusal), refusal=refusal)
        self.assertEqual((ctx.exception.resource, ctx.exception.need), ("bytes", dev))
        self.assertTrue(mi.information(h, backend="cuda", budget=Budget(bytes=max(stage2["bytes"], dev))).equal(ref))
        # requests: the output budget
        items = [dict(op="entropy", a="x", group=g) for g in range(3)]
        self.assertEqual(len(ref.query(items, budget=Budget(output=3))), 3)
        with self.assertRaises(AlgebraBudgetError) as ctx:
            ref.query(items, budget=Budget(output=2))
        self.assertEqual((ctx.exception.resource, ctx.exception.need), ("output", 3))
        # the exact X^2: every big number bounded before any is made
        with self.assertRaises(AlgebraBudgetError) as ctx:
            ref.chi_square("x", "y", exact=True, budget=Budget(bytes=64))
        need_bytes = ctx.exception.need
        self.assertIsNotNone(ref.chi_square("x", "y", exact=True, budget=Budget(bytes=need_bytes))["exact"])
        with self.assertRaises(AlgebraBudgetError):
            ref.chi_square("x", "y", exact=True, budget=Budget(bytes=need_bytes - 1))

    def test_dist003_invalid_requests_each_status_and_opt_in_import(self):
        rng = np.random.default_rng(3010)
        rows = rows_of(random_tables(rng, 1), rng)
        inf = mi.information(md.histogram(scope_of(rows, 1), list(NAMES)))
        good = inf.requests([dict(op="mutual_information", a="x", b="y")])
        cases = []
        for field, value in (("op", 0), ("op", 8), ("units", 2), ("group", 5), ("b", 0), ("b", 1), ("c", 4),
                             ("reserved", 1)):
            q = good.copy()
            q[field][0] = value
            cases.append(q)
        q = good.copy(); q["op"][0] = 5; q["units"][0] = 1                # bits do not apply to X^2
        cases.append(q)
        q = good.copy(); q["op"][0] = 1                                   # entropy takes one mask
        cases.append(q)
        out = inf.query_arrays(np.concatenate(cases + [good]))
        self.assertEqual(list(out["status"]), [-1] * 10 + [0])
        with self.assertRaises(ValueError):
            inf.requests([dict(op="kl", a="x")])
        with self.assertRaises(ValueError):
            inf.entropy("x", units="hartleys")
        with self.assertRaises(ValueError):
            inf.query([dict(op="mutual_information", a="x", b="x")])       # overlapping masks: native INVALID
        code = "import sys, torbles; print('torbles.math_info' in sys.modules)"
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.split()
        self.assertEqual(out, ["False"])


def C_ref(x):
    import ctypes
    return ctypes.byref(x)


if __name__ == "__main__":
    unittest.main()
