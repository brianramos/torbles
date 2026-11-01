# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""PROB-001 (small-support partition and likelihood functions) (torbles/math_prob.c, math_prob_gpu.cu, math_prob.py).

Oracles, independent of the implementation: enumeration of the declared support written in the tests, with Python
Fractions for categorical models (Z = sum w, p = w / Z, the observed T totals minus N E_p[T], -N Cov_p[T], exactly) and
80-digit Decimal exp / ln for exponential families and for every logarithm (Z, log Z, the log-likelihood, the gradient
and the Hessian must lie inside the certified enclosures and the exact rational bounds, compared exactly as
Fractions); central finite differences of the log-likelihood as a sanity check of the gradient and Hessian paths;
DuckDB GROUP BY and CROSS JOIN counts behind the histograms. Every evaluation is also run on the CUDA path (emulated
here; TORBLES_EMU_SCRAMBLE=1 scrambles the thread order) and compared bit for bit with the CPU. Seeds are fixed.
"""
import itertools
import math
import unittest
from decimal import Decimal, getcontext
from fractions import Fraction

import duckdb
import numpy as np
import pandas as pd

import torbles
from torbles import math_bank as mbk, math_conv as mc, math_dist as md, math_prob as mp
from torbles.math_budgets import Budget
from torbles.math_domains import DomainError
from torbles.math_errors import StaleStateError, UnsupportedError
from torbles.mathematics import AlgebraBudgetError

getcontext().prec = 80


# ---------------------------------------------------------------------- oracles

def cuda_label():
    """the backend label of a CUDA result: "cuda" on a device, "cuda-emulated" when the kernels run in the emulation"""
    from torbles import math_gpu
    return "cuda-emulated" if math_gpu.device()["emulated"] else "cuda"


def dec(q):
    q = Fraction(q)
    return Decimal(q.numerator) / Decimal(q.denominator)


def inside(iv, v):
    """the exact value v (Fraction or Decimal) lies in the double or Fraction interval iv, compared exactly"""
    return Fraction(iv[0]) <= Fraction(v) <= Fraction(iv[1])


def cat_oracle(w, n, T=None):
    """enumeration of a categorical model with weights w over the support, masses n, statistic T (rows; None: one-hot)"""
    d = len(w)
    T = T if T is not None else [[Fraction(int(i == j)) for j in range(d)] for i in range(d)]
    m = len(T[0])
    Z = sum(Fraction(x) for x in w)
    p = [Fraction(x) / Z for x in w]
    N = sum(n)
    E = [sum(p[i] * T[i][j] for i in range(d)) for j in range(m)]
    tobs = [sum(n[i] * Fraction(T[i][j]) for i in range(d)) for j in range(m)]
    grad = [tobs[j] - N * E[j] for j in range(m)]
    hess = [[-N * (sum(p[i] * T[i][j] * T[i][l] for i in range(d)) - E[j] * E[l]) for l in range(m)] for j in range(m)]
    if any(n[i] and not w[i] for i in range(d)):
        ll = None
    else:
        ll = sum(Decimal(n[i]) * (dec(w[i]) / dec(Z)).ln() for i in range(d) if n[i])
    return dict(Z=Z, logZ=dec(Z).ln(), loglik=ll, grad=grad, hess=hess)


def exp_oracle(h, T, theta, n):
    """80-digit enumeration of p_theta(x) = h(x) exp(theta . T(x)) / Z"""
    d, m = len(h), len(theta)
    s = [sum(Fraction(theta[j]) * Fraction(T[i][j]) for j in range(m)) for i in range(d)]
    e = [dec(h[i]) * dec(s[i]).exp() for i in range(d)]
    Z = sum(e)
    p = [x / Z for x in e]
    N = sum(n)
    E = [sum(p[i] * dec(T[i][j]) for i in range(d)) for j in range(m)]
    tobs = [sum(n[i] * dec(T[i][j]) for i in range(d)) for j in range(m)]
    grad = [tobs[j] - N * E[j] for j in range(m)]
    hess = [[-N * (sum(p[i] * dec(T[i][j]) * dec(T[i][l]) for i in range(d)) - E[j] * E[l]) for l in range(m)]
            for j in range(m)]
    if any(n[i] and not h[i] for i in range(d)):
        ll = None
    else:
        ll = sum(Decimal(n[i]) * p[i].ln() for i in range(d) if n[i])
    return dict(Z=Z, logZ=Z.ln(), loglik=ll, grad=grad, hess=hess)


def arrays_scope(cols, valid=None, groups=None, ngroups=None):
    return mbk.Scope.from_arrays({k: np.asarray(v, dtype=np.int64) for k, v in cols.items()}, valid=valid,
                                 groups=None if groups is None else np.asarray(groups, dtype=np.uint32), ngroups=ngroups)


def nullable(values):
    """values with None -> (zeros, validity mask)"""
    return [0 if v is None else v for v in values], np.array([v is not None for v in values])


def table(db_frame):
    cols = list(db_frame.columns)
    ddl = "CREATE TABLE t (" + ", ".join(f'"{c}" BIGINT' for c in cols) + ")"
    frame = pd.DataFrame({c: pd.Series(list(db_frame[c]), dtype=object) for c in cols})
    return torbles.from_ddl(ddl, {"t": frame}, keep_plain=True)


def both(model, st, params, **kw):
    """the CPU and the (emulated) CUDA evaluations, checked equal bit for bit"""
    cpu = model.evaluate(st, params, backend="cpu", **kw)
    gpu = model.evaluate(st, params, backend="cuda", **kw)
    assert gpu.backend in ("cuda", "cuda-emulated")
    assert cpu.equal(gpu), "CUDA sums / answers differ from the CPU"
    return cpu


class Prob001(unittest.TestCase):
    # ------------------------------------------------------------------ the declared support
    def test_prob001_declared_support_order_identity_and_refusals(self):
        S = mp.support([3, None, -2, 7, 0])
        self.assertEqual(S.points, (-2, 0, 3, 7, None))                 # values increasing, NULL last
        self.assertEqual(S.index(None), 4)
        self.assertEqual(S.index(3), 2)
        self.assertNotIn(5, S)
        J = mp.support([(1, None), (0, 2), (1, 0), (0, None)])
        self.assertEqual(J.points, ((0, 2), (0, None), (1, 0), (1, None)))
        self.assertEqual(J.k, 2)
        P = mp.support([0, 1]).product(mp.support([None, 5]))
        self.assertEqual(P.points, ((0, 5), (0, None), (1, 5), (1, None)))
        self.assertTrue(mp.support([2, 1]).equal(mp.support([1, 2])))
        self.assertFalse(mp.support([1, 2]).equal(mp.support([1, 2], scales=(1,))))
        with self.assertRaises(ValueError):
            mp.support([1, 2, 1])                                       # duplicates are refused
        with self.assertRaises(ValueError):
            mp.support([])
        with self.assertRaises(ValueError):
            mp.support([(1, 2, 3, 4)])
        with self.assertRaises(UnsupportedError):
            mp.support([0, 1]).product(mp.support([(1, 2, 3)]))         # k = 4
        D = mp.support([Fraction(1, 2), Decimal("1.25")], scales=(2,))  # decimal values in logical units
        self.assertEqual(D.points, (50, 125))
        with self.assertRaises(ValueError):
            mp.support([Fraction(1, 3)], scales=(2,))

    # ------------------------------------------------------------------ retained statistics vs DuckDB
    def test_prob001_retained_statistics_match_duckdb_counts(self):
        rng = np.random.default_rng(101)
        n = 400
        x = [int(v) if rng.random() > 0.1 else None for v in rng.integers(0, 6, n)]
        y = [int(v) for v in rng.integers(-1, 2, n)]
        k = [int(v) for v in rng.integers(0, 3, n)]
        w = [int(v) for v in rng.integers(0, 9, n)]
        frame = pd.DataFrame({"x": pd.Series(x, dtype=object), "y": y, "k": k, "w": w})
        con = duckdb.connect()
        con.register("t", frame)
        db = table(frame)
        scope = mbk.Scope.from_table(db["t"], ["x", "y", "w"], group_by=["k"])
        S = mp.support(list(range(0, 8)) + [None])                      # 6, 7 are declared but never observed
        J = mp.support([(a, b) for a in list(range(0, 6)) + [None] for b in (-1, 0, 1)])
        for repr_ in ("sparse", "dense"):
            st = mp.stats(md.histogram(scope, ["x"], repr=repr_), S)
            for g in range(3):
                want = {p: 0 for p in S.points}
                for (v, c) in con.execute(f"SELECT x, COUNT(*) FROM t WHERE k = {g} GROUP BY x").fetchall():
                    want[None if v is None or (isinstance(v, float) and math.isnan(v)) else int(v)] = int(c)
                self.assertEqual(st.counts(g), want, (repr_, g))
                self.assertEqual(st.group(g)["total"], sum(want.values()))
            self.assertEqual(st.counts(0)[6], 0)
            sj = mp.stats(md.histogram(scope, ["x", "y"], repr=repr_), J)
            for g in range(3):
                got = {p: c for p, c in sj.counts(g).items() if c}
                rows = con.execute(f"SELECT x, y, COUNT(*) FROM t WHERE k = {g} GROUP BY x, y").fetchall()
                want = {((None if a is None or (isinstance(a, float) and math.isnan(a)) else int(a)), int(b)): int(c)
                        for a, b, c in rows}
                self.assertEqual(got, want)
        # the NULL category dropped by declaration (listwise): the dropped mass is DuckDB's NULL count
        S0 = mp.support(range(0, 6))
        with self.assertRaises(DomainError) as ctx:
            mp.stats(md.histogram(scope, ["x"]), S0)                     # NULL is observed and not declared
        self.assertIsNone(ctx.exception.point)
        se = mp.stats(md.histogram(scope, ["x"]), S0, exclude=["x"])
        for g in range(3):
            nulls = con.execute(f"SELECT COUNT(*) FROM t WHERE k = {g} AND x IS NULL").fetchone()[0]
            self.assertEqual(se.group(g)["dropped"], nulls)
        self.assertEqual(se.info["excluded"], ["x"])
        # weight masses: DuckDB SUM(w)
        sw = mp.stats(md.histogram(scope, ["x"], weight="w"), S, mass="weight")
        for g in range(3):
            for v, s_ in con.execute(f"SELECT x, SUM(w) FROM t WHERE k = {g} GROUP BY x").fetchall():
                p = None if v is None or (isinstance(v, float) and math.isnan(v)) else int(v)
                self.assertEqual(sw.counts(g)[p], int(s_))
        neg = arrays_scope({"x": [0, 1], "w": [2, -1]})
        with self.assertRaises(DomainError):
            mp.stats(md.histogram(neg, ["x"], weight="w"), mp.support([0, 1]), mass="weight")
        con.close()

    # ------------------------------------------------------------------ acceptance 1: exact categorical models
    def test_prob001_categorical_normalization_gradient_hessian_match_enumeration(self):
        rng = np.random.default_rng(202)
        for trial in range(6):
            d = int(rng.integers(2, 9))
            S = mp.support(range(d))
            data = [int(v) for v in rng.integers(0, d, int(rng.integers(5, 60)))]
            st = mp.stats(md.histogram(arrays_scope({"x": data}), ["x"]), S)
            nvec = [st.counts()[i] for i in range(d)]
            T = None
            if trial % 2:
                T = [[Fraction(int(rng.integers(-9, 10)), int(rng.integers(1, 5))) for _ in range(2)] for _ in range(d)]
            M = mp.categorical(S, features=T)
            reqs = []
            for _ in range(4):
                reqs.append([Fraction(int(rng.integers(1, 20)), int(rng.integers(1, 7))) for _ in range(d)])
            ev = both(M, st, reqs, grad=True, hess=True)
            for i, w in enumerate(reqs):
                o = cat_oracle(w, nvec, T)
                a, ex = ev[i], ev.exact(i)
                self.assertEqual(a["Z_value"], o["Z"])
                self.assertEqual(ex["Z"], (o["Z"], o["Z"]))
                self.assertTrue(ex["exact"]["Z"] and ex["exact"]["grad"] and ex["exact"]["hess"])
                self.assertEqual(ex["grad"], [(g, g) for g in o["grad"]])
                self.assertEqual([[h[0] for h in row] for row in ex["hess"]], o["hess"])
                self.assertTrue(inside(a["Z"], o["Z"]) and inside(a["logZ"], o["logZ"]))
                self.assertTrue(inside(a["loglik"], o["loglik"]) and inside(ex["loglik"], o["loglik"]))
                self.assertTrue(inside(ex["logZ"], o["logZ"]))
                for j, g in enumerate(o["grad"]):
                    self.assertTrue(inside(a["grad"][j], g))
                for j, row in enumerate(o["hess"]):
                    for l, hv in enumerate(row):
                        self.assertTrue(inside(a["hess"][j][l], hv))
                self.assertLess(float(ex["logZ"][1] - ex["logZ"][0]), 2.0 ** -100)
                self.assertLess(float(ex["loglik"][1] - ex["loglik"][0]), 2.0 ** -90 * max(1, sum(nvec)))
        # exact probabilities: Z = 1 exactly; probabilities that do not sum to 1 are refused per request
        S = mp.support([0, 1, 2])
        st = mp.stats(md.histogram(arrays_scope({"x": [0, 1, 1, 2]}), ["x"]), S)
        M = mp.categorical(S)
        ev = both(M, st, [[Fraction(1, 6), Fraction(1, 3), Fraction(1, 2)], [Fraction(1, 2), Fraction(1, 2), 1]],
                  mode="probabilities", grad=True)
        self.assertEqual(ev[0]["Z_value"], 1)
        self.assertEqual(ev.exact(0)["grad"], [(g, g) for g in cat_oracle([Fraction(1, 6), Fraction(1, 3), Fraction(1, 2)],
                                                                           [1, 2, 1])["grad"]])
        with self.assertRaises(DomainError):
            ev[1]
        # weights given as a dict over the support (unlisted points: weight 0)
        a = M.loglik(st, {0: 1, 1: 2, 2: 3}, grad=True)
        self.assertEqual(a["Z_value"], 6)

    # ------------------------------------------------------------------ acceptance 1: exponential families
    def test_prob001_exponential_family_enclosures_contain_80_digit_enumeration(self):
        rng = np.random.default_rng(303)
        cases = 0
        for trial in range(8):
            k = 1 if trial < 5 else 2
            if k == 1:
                vals = sorted(set(int(v) for v in rng.integers(-6, 9, int(rng.integers(2, 9)))))
                S = mp.support(vals)
                data = [int(rng.choice(vals)) for _ in range(int(rng.integers(0, 50)))]
                sc = arrays_scope({"x": data or [vals[0]]}, groups=[0] * max(len(data), 1) if data else [1], ngroups=2)
                hist = md.histogram(sc, ["x"])
                feats = [(Fraction(x), Fraction(x * x, 3)) for x in vals]
            else:
                pts = [(a, b) for a in range(3) for b in (0, 1, None)]
                S = mp.support(pts)
                xs = [int(v) for v in rng.integers(0, 3, 40)]
                ys = [int(v) if rng.random() > 0.2 else None for v in rng.integers(0, 2, 40)]
                yv, yok = nullable(ys)
                sc = mbk.Scope.from_arrays({"x": np.array(xs), "y": np.array(yv)}, valid={"y": yok},
                                           groups=np.array([i % 2 for i in range(40)], dtype=np.uint32), ngroups=2)
                hist = md.histogram(sc, ["x", "y"])
                feats = [(Fraction(a), Fraction(-1 if b is None else b), Fraction(a * (b or 0), 2)) for a, b in S.points]
            base = None if trial % 3 == 0 else [Fraction(int(rng.integers(1, 5)), int(rng.integers(1, 4)))
                                                for _ in range(len(S))]
            st = mp.stats(hist, S)
            E = mp.exponential(S, features=feats, base=base)
            m = len(feats[0])
            h = base if base is not None else [1] * len(S)
            scale = [1, 5, 40][trial % 3]                                # large parameters: exp far past doubles
            thetas = [tuple(Fraction(int(rng.integers(-scale * 8, scale * 8 + 1)), 8) for _ in range(m)) for _ in range(5)]
            thetas.append(tuple([0] * m))
            for g in range(2):
                ev = both(E, st, thetas, groups=g, grad=True, hess=True)
                nvec = [st.counts(g)[p] for p in S.points]
                for i, th in enumerate(thetas):
                    o = exp_oracle(h, feats, th, nvec)
                    a, ex = ev[i], ev.exact(i)
                    self.assertTrue(inside(ex["Z"], o["Z"]), (trial, g, i))
                    self.assertTrue(inside(a["logZ"], o["logZ"]) and inside(ex["logZ"], o["logZ"]))
                    if a["Z"][1] != math.inf and a["Z"][0] > 0:
                        self.assertTrue(inside(a["Z"], o["Z"]))
                    self.assertTrue(inside(a["loglik"], o["loglik"]) and inside(ex["loglik"], o["loglik"]))
                    for j in range(m):
                        self.assertTrue(inside(a["grad"][j], o["grad"][j]) and inside(ex["grad"][j], o["grad"][j]))
                        for l in range(m):
                            self.assertTrue(inside(a["hess"][j][l], o["hess"][j][l]), (trial, g, i, j, l))
                            self.assertTrue(inside(ex["hess"][j][l], o["hess"][j][l]))
                    # the reported widths and the stated precision of the exact sums
                    self.assertLess(a["width"]["Z"], 2.0 ** -100)
                    zl, zh, ze, zd = a["Z_exact"]
                    self.assertEqual((Fraction(zl) * Fraction(2) ** ze / zd, Fraction(zh) * Fraction(2) ** ze / zd), ex["Z"])
                    self.assertLess(float(ex["logZ"][1] - ex["logZ"][0]), 2.0 ** -100)
                    self.assertLessEqual(float(a["logZ"][1] - a["logZ"][0]), a["width"]["logZ"])
                    scale_t = max(1, sum(nvec)) * max(1, max(abs(Fraction(t)) for t in th) * 50)
                    for j in range(m):
                        self.assertLess(float(ex["grad"][j][1] - ex["grad"][j][0]), 2.0 ** -80 * scale_t)
                    cases += 1
        self.assertEqual(cases, 8 * 2 * 6)
        # Z past the double range: [DBL_MAX, inf] with the exact form and a finite log Z
        S = mp.support(range(8))
        st = mp.stats(md.histogram(arrays_scope({"x": [1, 7, 7]}), ["x"]), S)
        E = mp.exponential(S, features=lambda x: (x,))
        a = E.loglik(st, (300,))
        self.assertEqual(a["Z"], (np.finfo(float).max, math.inf))
        o = exp_oracle([1] * 8, [(x,) for x in range(8)], (300,), [0, 1, 0, 0, 0, 0, 0, 2])
        self.assertTrue(inside(a["logZ"], o["logZ"]) and inside(a["loglik"], o["loglik"]))
        b = E.loglik(st, (-300,))
        self.assertTrue(inside(b["logZ"], exp_oracle([1] * 8, [(x,) for x in range(8)], (-300,), [0] * 8)["logZ"]))

    # ------------------------------------------------------------------ sanity: finite differences
    def test_prob001_gradient_and_hessian_agree_with_finite_differences(self):
        S = mp.support(range(-3, 5))
        rng = np.random.default_rng(404)
        data = [int(v) for v in rng.integers(-3, 5, 200)]
        st = mp.stats(md.histogram(arrays_scope({"x": data}), ["x"]), S)
        E = mp.exponential(S, features=lambda x: (x, Fraction(x * x, 4)))
        theta = (Fraction(1, 5), Fraction(-1, 3))
        eps = 1e-5

        def ll(t):
            r = E.loglik(st, t)["loglik"]
            return (r[0] + r[1]) / 2

        def gr(t):
            g = E.loglik(st, t, grad=True)["grad"]
            return [(lo + hi) / 2 for lo, hi in g]
        a = E.loglik(st, theta, hess=True)
        for j in range(2):
            up = [float(x) for x in theta]; dn = list(up)
            up[j] += eps; dn[j] -= eps
            fd = (ll(tuple(up)) - ll(tuple(dn))) / (2 * eps)
            self.assertAlmostEqual(fd, sum(a["grad"][j]) / 2, delta=1e-4 * max(1, abs(fd)))
            g_up, g_dn = gr(tuple(up)), gr(tuple(dn))
            for l in range(2):
                fh = (g_up[l] - g_dn[l]) / (2 * eps)
                self.assertAlmostEqual(fh, sum(a["hess"][l][j]) / 2, delta=1e-4 * max(1, abs(fh)))

    # ------------------------------------------------------------------ acceptance 2: zero mass
    def test_prob001_zero_mass_is_minus_infinity_never_a_large_finite_number(self):
        S = mp.support(range(4))
        st = mp.stats(md.histogram(arrays_scope({"x": [0, 1, 1, 3]}), ["x"]), S)
        M = mp.categorical(S, features=lambda x: (x,))
        ev = both(M, st, [[1, 2, 5, 0], [1, 2, 0, 3], [0, 0, 0, 0]], grad=True, hess=True)
        a = ev[0]                                                       # w = 0 at x = 3, which has mass 1
        self.assertTrue(a["neg_inf"])
        self.assertEqual(a["loglik"], (-math.inf, -math.inf))
        self.assertIsNone(a["grad"]); self.assertIsNone(a["hess"])
        ex = ev.exact(0)
        self.assertIsNone(ex["loglik"]); self.assertIsNone(ex["grad"])
        self.assertEqual(ex["Z"], (Fraction(8), Fraction(8)))           # Z is still defined
        b = ev[1]                                                       # w = 0 at x = 2, which has no mass: finite
        o = cat_oracle([1, 2, 0, 3], [1, 2, 0, 1], [[Fraction(x)] for x in range(4)])
        self.assertFalse(b["neg_inf"])
        self.assertTrue(inside(b["loglik"], o["loglik"]))
        self.assertEqual(ev.exact(1)["grad"], [(o["grad"][0], o["grad"][0])])
        with self.assertRaises(DomainError):
            ev[2]                                                       # Z = 0 is refused
        # exponential families: h = 0 at an observed point is -inf for every theta; at an unobserved point it is fine
        E = mp.exponential(S, features=lambda x: (x,), base=[1, 1, 1, 0])
        ev = both(E, st, [(Fraction(1, 2),), (-3,)], grad=True)
        for i in range(2):
            self.assertTrue(ev[i]["neg_inf"])
            self.assertEqual(ev[i]["loglik"], (-math.inf, -math.inf))
            self.assertIsNone(ev[i]["grad"])
        st2 = mp.stats(md.histogram(arrays_scope({"x": [0, 1, 1]}), ["x"]), S)
        a = E.loglik(st2, (Fraction(1, 2),), grad=True)
        o = exp_oracle([1, 1, 1, 0], [(x,) for x in range(4)], (Fraction(1, 2),), [1, 2, 0, 0])
        self.assertTrue(inside(a["loglik"], o["loglik"]) and inside(a["grad"][0], o["grad"][0]))
        with self.assertRaises(DomainError):
            mp.exponential(S, features=lambda x: (x,), base=[0, 0, 0, 0])   # Z = 0 for every theta
        with self.assertRaises(DomainError):
            mp.exponential(S, features=lambda x: (x,), base=[1, -1, 0, 0])
        # zero counts are fine: an absent group is the empty sample (loglik 0, gradient 0)
        sg = mp.stats(md.histogram(arrays_scope({"x": [0, 2]}, groups=[0, 0], ngroups=2), ["x"]), S)
        a = mp.exponential(S, features=lambda x: (x,)).loglik(sg, (1,), group=1, grad=True, hess=True)
        self.assertTrue(a["empty"])
        self.assertEqual((a["loglik"], a["grad"], a["mass"]), ((0.0, 0.0), [(0.0, 0.0)], 0))

    # ------------------------------------------------------------------ acceptance 2: support changes
    def test_prob001_support_changes_are_different_states(self):
        data = [0, 1, 1, 2, 2, 2]
        hist = md.histogram(arrays_scope({"x": data}), ["x"])
        S3, S5 = mp.support(range(3)), mp.support(range(5))
        st3, st5 = mp.stats(hist, S3), mp.stats(hist, S5)
        self.assertNotEqual(st3.info["support_digest"], st5.info["support_digest"])
        self.assertEqual(st5.counts(), {0: 1, 1: 2, 2: 3, 3: 0, 4: 0})   # declared, unobserved: mass 0
        E3, E5 = mp.exponential(S3, features=lambda x: (x,)), mp.exponential(S5, features=lambda x: (x,))
        with self.assertRaises(DomainError):
            E5.evaluate(st3, [(1,)])                                    # a model of another support: refused
        with self.assertRaises(DomainError):
            E3.evaluate(st5, [(1,)])
        with self.assertRaises(DomainError):
            mp.need(st3, E5)
        a3, a5 = E3.loglik(st3, (Fraction(1, 2),), grad=True), E5.loglik(st5, (Fraction(1, 2),), grad=True)
        o3 = exp_oracle([1] * 3, [(x,) for x in range(3)], (Fraction(1, 2),), [1, 2, 3])
        o5 = exp_oracle([1] * 5, [(x,) for x in range(5)], (Fraction(1, 2),), [1, 2, 3, 0, 0])
        self.assertTrue(inside(a3["logZ"], o3["logZ"]) and inside(a5["logZ"], o5["logZ"]))
        self.assertTrue(inside(a3["loglik"], o3["loglik"]) and inside(a5["loglik"], o5["loglik"]))
        self.assertGreater(a3["loglik"][0], a5["loglik"][1])            # the zero-mass points change Z
        # an observed value outside the declared support is refused, naming it
        with self.assertRaises(DomainError) as ctx:
            mp.stats(hist, mp.support([0, 1]))
        self.assertEqual((ctx.exception.point, ctx.exception.rows, ctx.exception.group), (2, 3, 0))
        # the histogram's scales and coordinates must be the support's
        with self.assertRaises(DomainError):
            mp.stats(hist, mp.support(range(3), scales=(1,)))
        with self.assertRaises(DomainError):
            mp.stats(hist, mp.support([(0, 0)]))

    # ------------------------------------------------------------------ acceptance 2: dependent events
    def test_prob001_dependent_events_joint_state_and_declared_products(self):
        rng = np.random.default_rng(505)
        n = 300
        x = [int(v) for v in rng.integers(0, 3, n)]
        y = [(v + int(rng.random() < 0.2)) % 2 for v in x]              # y depends on x
        k = [int(v) for v in rng.integers(0, 2, n)]
        frame = pd.DataFrame({"x": x, "y": y, "k": k})
        con = duckdb.connect()
        con.register("t", frame)
        db = table(frame)
        scope = mbk.Scope.from_table(db["t"], ["x", "y"], group_by=["k"])
        Sx, Sy = mp.support(range(3)), mp.support(range(2))
        Sxy = Sx.product(Sy)
        joint = mp.stats(md.histogram(scope, ["x", "y"]), Sxy)
        sx, sy = mp.stats(md.histogram(scope, ["x"]), Sx), mp.stats(md.histogram(scope, ["y"]), Sy)
        for g in range(2):
            want = {(a, b): 0 for a, b in Sxy.points}
            for a, b, c in con.execute(f"SELECT x, y, COUNT(*) FROM t WHERE k = {g} GROUP BY x, y").fetchall():
                want[(int(a), int(b))] = int(c)
            self.assertEqual(joint.counts(g), want)
        # the product of the marginals is the joint only under an independence assumption: refused undeclared
        with self.assertRaises(DomainError) as ctx:
            mp.product(sx, sy)
        self.assertEqual(ctx.exception.reason, "needs_declaration")
        other = arrays_scope({"z": [1, 2]})
        with self.assertRaises(DomainError) as ctx:
            mp.product(sx, sy, declaration=mc.independent(other))
        self.assertEqual(ctx.exception.reason, "not_covered")
        prod = mp.product(sx, sy, declaration=mc.independent(scope))
        self.assertTrue(prod.info["declared"] and prod.info["product"])
        for g in range(2):
            cross = {(a, b): 0 for a, b in Sxy.points}
            for a, b, c in con.execute(f"SELECT a.x, b.y, COUNT(*) FROM t a CROSS JOIN t b WHERE a.k = {g} AND "
                                       f"b.k = {g} GROUP BY 1, 2").fetchall():
                cross[(int(a), int(b))] = int(c)
            self.assertEqual(prod.counts(g), cross)                     # the CROSS JOIN within the group
        # a joint model on the joint state equals the enumeration over the joint counts (dependence kept), and
        # differs from the same model on the declared product
        feats = lambda p: (p[0], p[1], p[0] * p[1])
        E = mp.exponential(Sxy, features=feats)
        th = (Fraction(1, 4), Fraction(-1, 2), Fraction(2, 3))
        aj = both(E, joint, [th], groups=1, grad=True, hess=True)[0]
        nj = [joint.counts(1)[p] for p in Sxy.points]
        oj = exp_oracle([1] * 6, [feats(p) for p in Sxy.points], th, nj)
        self.assertTrue(inside(aj["loglik"], oj["loglik"]))
        for j in range(3):
            self.assertTrue(inside(aj["grad"][j], oj["grad"][j]))
        ap = both(E, prod, [th], groups=1, grad=True)[0]
        npd = [prod.counts(1)[p] for p in Sxy.points]
        op = exp_oracle([1] * 6, [feats(p) for p in Sxy.points], th, npd)
        self.assertTrue(inside(ap["loglik"], op["loglik"]))
        Nj = sum(nj)
        self.assertNotAlmostEqual(aj["loglik"][0] / Nj, ap["loglik"][0] / sum(npd), places=3)
        # a marginal state does not answer a joint model; states of different sources multiply undeclared
        with self.assertRaises(DomainError):
            E.evaluate(sx, [th])
        s1, s2 = arrays_scope({"x": [0, 1, 1]}), arrays_scope({"x": [1, 0], "pad": [5, 6]})
        p12 = mp.product(mp.stats(md.histogram(s1, ["x"]), mp.support(range(2))),
                         mp.stats(md.histogram(s2, ["x"]), mp.support(range(2))))
        self.assertFalse(p12.info["declared"])
        self.assertEqual(p12.counts(), {(0, 0): 1, (0, 1): 1, (1, 0): 2, (1, 1): 2})
        # one source with two groupings: the groups do not correspond
        regroup = mbk.Scope.from_table(db["t"], ["x", "y"], group_by=["y"])
        with self.assertRaises(DomainError) as ctx:
            mp.product(sx, mp.stats(md.histogram(regroup, ["y"]), Sy), declaration=mc.independent(scope))
        self.assertEqual(ctx.exception.reason, "grouping_differs")
        con.close()

    # ------------------------------------------------------------------ decimal scales, weight masses, 3 coordinates
    def test_prob001_decimal_scales_weight_masses_and_three_coordinates(self):
        # a DECIMAL(., 2) coordinate: the support is declared in the same logical units and scale
        sc = mbk.Scope.from_arrays({"x": np.array([125, 250, 250, 375]), "w": np.array([2, 3, 4, 1])},
                                   scales={"x": 2})
        h = md.histogram(sc, ["x"], weight="w")
        S = mp.support([Decimal("1.25"), Decimal("2.50"), Decimal("3.75"), Decimal("5.00")], scales=(2,))
        st = mp.stats(h, S, mass="weight")
        self.assertEqual(st.counts(), {125: 2, 250: 7, 375: 1, 500: 0})
        with self.assertRaises(DomainError):
            mp.stats(h, mp.support([125, 250, 375, 500]))               # scale 0 is another support
        feats = [(Fraction(x, 100),) for x in S.points]                 # the decimal value as the feature
        E = mp.exponential(S, features=feats, base=[1, 2, 1, Fraction(1, 2)])
        a = both(E, st, [(Fraction(1, 2),), (-3,)], grad=True, hess=True)
        for i, th in enumerate([(Fraction(1, 2),), (-3,)]):
            o = exp_oracle([1, 2, 1, Fraction(1, 2)], feats, th, [2, 7, 1, 0])   # weight sums as masses
            r = a[i]
            self.assertTrue(inside(r["loglik"], o["loglik"]) and inside(r["grad"][0], o["grad"][0]))
            self.assertTrue(inside(r["hess"][0][0], o["hess"][0][0]) and inside(a.exact(i)["loglik"], o["loglik"]))
        # dense histograms over a wider explicit domain: structural zeros outside the support are not observations
        hd = md.histogram(arrays_scope({"x": [1, 2, 2]}), ["x"], repr="dense", domain=[(-5, 9, False)])
        self.assertEqual(mp.stats(hd, mp.support([1, 2, 7])).counts(), {1: 1, 2: 2, 7: 0})
        # three coordinates with NULLs: the joint state equals DuckDB's counts and is evaluated as enumerated
        rng = np.random.default_rng(808)
        n = 120
        cols = {c: [int(v) if rng.random() > 0.15 else None for v in rng.integers(0, 2, n)] for c in ("x", "y", "z")}
        frame = pd.DataFrame({c: pd.Series(v, dtype=object) for c, v in cols.items()})
        con = duckdb.connect()
        con.register("t", frame)
        db = table(frame)
        scope = mbk.Scope.from_table(db["t"], ["x", "y", "z"])
        pts = list(itertools.product((0, 1, None), repeat=3))
        S3 = mp.support(pts)
        st3 = mp.stats(md.histogram(scope, ["x", "y", "z"]), S3)
        want = {p: 0 for p in S3.points}
        for a_, b_, c_, cnt in con.execute("SELECT x, y, z, COUNT(*) FROM t GROUP BY x, y, z").fetchall():
            key = tuple(None if v is None or (isinstance(v, float) and math.isnan(v)) else int(v) for v in (a_, b_, c_))
            want[key] = int(cnt)
        self.assertEqual(st3.counts(), want)
        f3 = lambda p: (0 if p[0] is None else p[0], 0 if p[1] is None else p[1], int(p[2] is None))
        E3 = mp.exponential(S3, features=f3)
        th = (Fraction(1, 3), Fraction(-2, 5), Fraction(3, 4))
        r = both(E3, st3, [th], grad=True, hess=True)[0]
        o = exp_oracle([1] * len(S3), [f3(p) for p in S3.points], th, [st3.counts()[p] for p in S3.points])
        self.assertTrue(inside(r["loglik"], o["loglik"]))
        for j in range(3):
            self.assertTrue(inside(r["grad"][j], o["grad"][j]) and all(inside(r["hess"][j][l], o["hess"][j][l])
                                                                         for l in range(3)))
        # a 2-coordinate joint state times a 1-coordinate state of the same table: three coordinates, declared
        sxy = mp.stats(md.histogram(scope, ["x", "y"]), mp.support(list(itertools.product((0, 1, None), repeat=2))))
        sz = mp.stats(md.histogram(scope, ["z"]), mp.support([0, 1, None]))
        with self.assertRaises(DomainError):
            mp.product(sxy, sz)
        p3 = mp.product(sxy, sz, declaration=mc.independent(scope))
        cross = {p: 0 for p in S3.points}
        for a_, b_, c_, cnt in con.execute("SELECT a.x, a.y, b.z, COUNT(*) FROM t a CROSS JOIN t b GROUP BY 1, 2, 3").fetchall():
            key = tuple(None if v is None or (isinstance(v, float) and math.isnan(v)) else int(v) for v in (a_, b_, c_))
            cross[key] = int(cnt)
        self.assertEqual(p3.counts(), cross)
        self.assertTrue(p3.support.equal(S3))
        con.close()

    # ------------------------------------------------------------------ CUDA equals the CPU
    def test_prob001_cuda_batches_equal_cpu_bit_for_bit(self):
        rng = np.random.default_rng(606)
        d = 150                                                          # three chunks per request
        S = mp.support(range(d))
        data = [int(v) for v in rng.integers(0, d, 2000)]
        grp = [int(v) for v in rng.integers(0, 4, 2000)]
        st = mp.stats(md.histogram(arrays_scope({"x": data}, groups=grp, ngroups=4), ["x"]), S)
        E = mp.exponential(S, features=lambda x: (x % 7, Fraction(x, 10), x * x % 11),
                           base=lambda x: Fraction(1 + x % 3, 2))
        n = 120
        tn = rng.integers(-40, 41, (n, 3)).astype(np.int64)
        td = rng.integers(1, 9, (n, 3)).astype(np.int64)
        groups = rng.integers(0, 4, n)
        cpu = E.evaluate(st, (tn, td), groups=groups, grad=True, hess=True)
        gpu = E.evaluate(st, (tn, td), groups=groups, grad=True, hess=True, backend="cuda")
        self.assertTrue(cpu.equal(gpu))
        self.assertEqual(gpu.backend, cuda_label())
        for i in (0, 57, 119):
            self.assertEqual(cpu.exact(i), gpu.exact(i))
        M = mp.categorical(S, features=lambda x: (x % 5, Fraction(x, 3)))
        W = rng.integers(0, 50, (60, d)).astype(np.int64)
        W[:, ::7] = 0
        cpu = M.evaluate(st, W, groups=rng.integers(0, 4, 60), grad=True, hess=True)
        gpu = M.evaluate(st, W, groups=cpu.arrays()["answers"]["group"], grad=True, hess=True, backend="cuda")
        self.assertTrue(cpu.equal(gpu))
        flags = cpu.arrays()["answers"]["flags"]
        self.assertTrue(any(f & 1 for f in flags))                      # some -inf answers, the same on both
        one_hot = mp.categorical(mp.support(range(6)))
        st6 = mp.stats(md.histogram(arrays_scope({"x": [0, 5, 5, 2]}), ["x"]), mp.support(range(6)))
        W6 = rng.integers(1, 9, (10, 6)).astype(np.int64)
        ev = both(one_hot, st6, W6, grad=True, hess=True)
        o = cat_oracle([int(v) for v in W6[3]], [1, 0, 1, 0, 0, 2])
        ex = ev.exact(3)
        self.assertEqual(ex["grad"], [(g, g) for g in o["grad"]])
        self.assertEqual([[h[0] for h in row] for row in ex["hess"]], o["hess"])

    # ------------------------------------------------------------------ requests have their own statuses
    def test_prob001_request_statuses_and_refusals(self):
        S = mp.support(range(3))
        st = mp.stats(md.histogram(arrays_scope({"x": [0, 1, 2]}), ["x"]), S)
        E = mp.exponential(S, features=lambda x: (x,))
        ev = both(E, st, [(1,), (10 ** 8,), (Fraction(-1, 3),)], groups=[0, 0, 0], grad=True)
        self.assertEqual([ev.status(i) for i in range(3)], [0, -3, 0])  # an exponent past 2^(2^24): overflow
        with self.assertRaises(OverflowError):
            ev[1]
        self.assertTrue(inside(ev[2]["logZ"], exp_oracle([1] * 3, [(0,), (1,), (2,)], (Fraction(-1, 3),), [1, 1, 1])["logZ"]))
        ev = E.evaluate(st, [(1,), (1,)], groups=[0, 5])                # a group out of range
        self.assertEqual(ev.status(1), -1)
        with self.assertRaises(ValueError):
            ev[1]
        M = mp.categorical(S)
        ev = both(M, st, [[1, -1, 1], [Fraction(1, 3), Fraction(1, 5), Fraction(1, 7)],
                          [Fraction(1, 2 ** 40), Fraction(1, 3 ** 25), Fraction(1, 5 ** 17)]])
        with self.assertRaises(DomainError):
            ev[0]                                                       # a negative weight
        self.assertEqual(ev[1]["Z_value"], Fraction(1, 3) + Fraction(1, 5) + Fraction(1, 7))
        with self.assertRaises(OverflowError):
            ev[2]                                                       # the common denominator overflows int64
        with self.assertRaises(OverflowError):
            E.evaluate(st, [(Fraction(1, 2 ** 70),)])                   # a parameter outside int64
        with self.assertRaises(ValueError):
            E.evaluate(st, [(1, 2)])                                    # m = 1 parameter
        with self.assertRaises(UnsupportedError):
            mp.exponential(S, features=lambda x: tuple(range(17)))
        with self.assertRaises(OverflowError):
            mp.exponential(S, features=lambda x: (Fraction(1, 2 ** 40 + x),))   # the features' lcm overflows int64

    # ------------------------------------------------------------------ budgets: at the limit and one past
    def test_prob001_budgets_at_the_limit_and_one_past(self):
        S = mp.support(range(10), budget=Budget(support=10))
        with self.assertRaises(AlgebraBudgetError) as ctx:
            mp.support(range(10), budget=Budget(support=9))
        self.assertEqual(ctx.exception.resource, "support")
        hist = md.histogram(arrays_scope({"x": [1, 2, 3]}, groups=[0, 1, 2], ngroups=3), ["x"])
        mp.stats(hist, S, budget=Budget(states=30))
        with self.assertRaises(AlgebraBudgetError) as ctx:
            mp.stats(hist, S, budget=Budget(states=29))
        self.assertEqual((ctx.exception.resource, ctx.exception.need), ("states", 30))
        st = mp.stats(hist, S)
        E = mp.exponential(S, features=lambda x: (x, -x))
        params = [(1, 2)] * 4
        need = mp.need(st, E, 4, hess=True)
        self.assertEqual(need["output"], 4 * (3 + 2 + 3))
        self.assertEqual(need["tensor"], 4 * 10)
        E.evaluate(st, params, hess=True, budget=Budget(output=32))
        with self.assertRaises(AlgebraBudgetError) as ctx:
            E.evaluate(st, params, hess=True, budget=Budget(output=31))
        self.assertEqual((ctx.exception.resource, ctx.exception.need, ctx.exception.limit), ("output", 32, 31))
        E.evaluate(st, params, budget=Budget(tensor=40))
        with self.assertRaises(AlgebraBudgetError):
            E.evaluate(st, params, budget=Budget(tensor=39))
        dev = mp.need(st, E, 4, hess=True, backend="cuda")["bytes"]
        E.evaluate(st, params, hess=True, backend="cuda", budget=Budget(bytes=dev))
        with self.assertRaises(AlgebraBudgetError) as ctx:
            E.evaluate(st, params, hess=True, backend="cuda", budget=Budget(bytes=dev - 1))
        self.assertEqual(ctx.exception.resource, "bytes")
        M = mp.categorical(S)
        nh = mp.need(st, M, 2, hess=True)["output"]
        self.assertEqual(nh, 2 * (3 + 10 + 55))                         # one-hot: d and d (d + 1) / 2 entries
        M.evaluate(st, [[1] * 10] * 2, hess=True, budget=Budget(output=nh))
        with self.assertRaises(AlgebraBudgetError):
            M.evaluate(st, [[1] * 10] * 2, hess=True, budget=Budget(output=nh - 1))
        with self.assertRaises(AlgebraBudgetError):
            mp.exponential(S, features=lambda x: (x, -x), budget=Budget(tensor=19))
        mp.exponential(S, features=lambda x: (x, -x), budget=Budget(tensor=20))

    # ------------------------------------------------------------------ the certified logarithm, freshness
    def test_prob001_certified_logarithm_of_128_bit_integers(self):
        rng = np.random.default_rng(707)
        vals = [1, 2, 3, (1 << 64) - 1, 1 << 64, (1 << 64) + 1, (1 << 127) + 12345, (1 << 128) - 1]
        vals += [int(rng.integers(1, 1 << 62)) << int(rng.integers(0, 66)) | int(rng.integers(0, 1 << 30)) for _ in range(40)]
        for v in vals:
            lo, hi = mp.ln_bounds(v)
            exact = Decimal(v).ln() * (Decimal(2) ** 120)
            self.assertTrue(Decimal(lo) <= exact <= Decimal(hi), v)
            self.assertLessEqual(hi - lo, v.bit_length() + 149)
        with self.assertRaises(ValueError):
            mp.ln_bounds(0)

    def test_prob001_stale_table_scope_is_refused(self):
        db = table(pd.DataFrame({"x": [0, 1, 1, 2]}))
        t = db["t"]
        scope = mbk.Scope.from_table(t, ["x"])
        S = mp.support(range(4))
        st = mp.stats(md.histogram(scope, ["x"]), S)
        E = mp.exponential(S, features=lambda x: (x,))
        E.loglik(st, (1,))
        t.flip(0, 3)
        with self.assertRaises(StaleStateError):
            E.loglik(st, (1,))


if __name__ == "__main__":
    unittest.main()
