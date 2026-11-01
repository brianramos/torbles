# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""KERNEL-002: approximate spectral feature kernels (torbles/math_kapprox.c, math_kapprox_gpu.cu, math_kapprox.py).

Oracles, independent of the implementation and written here: direct small kernel matrices with Python's decimal
module at 60 digits (exp; cos and sin by Taylor series after reduction with pi from Machin's formula; ln and sqrt from
decimal) and exact Fractions (the Cauchy kernel, the Taylor polynomials, the weights). The declared approximants are
recomputed from their definitions (the truncated Gaussian expansion E(x) E(y) sum_n (2 gamma x.y)^n / n!, and the
random-feature average (1 / D) sum_f m_f cos(omega_f.(x - y)) over the state's exact frequencies) and must lie in
every native enclosure; the exact kernel quantities must lie in the stated bounds (certified for TAYLOR; for random
features with failure probability 1e-6 per answer: the empirical errors are recorded against the bound); the frequency
generator is reimplemented here (splitmix64 and the Box-Muller / tangent / exponential transforms in decimal) and must
agree with the native draws within the declared per-sample bound; KERNEL-001 and exact Python integers for polynomial
kernels; FND-011's own loader and the raw header bytes for the persisted records. Every native build is also run on the
CUDA path (emulated here, scrambled thread order: TORBLES_EMU_SCRAMBLE=1) and compared for equality. Budgets are tested
at the limit and one past it. Seeds are fixed.
"""
import math
import os
import struct
import unittest
from decimal import Decimal, getcontext, localcontext
from fractions import Fraction

import numpy as np

os.environ.setdefault("TORBLES_EMU_SCRAMBLE", "1")

import torbles
from torbles import math_bank as mbk, math_kapprox as ka, math_kernel as mk, math_persist as mp
from torbles.math_budgets import Budget
from torbles.math_certificates import NotAuthorizedError
from torbles.math_domains import DomainError
from torbles.math_errors import NotConvergedError, StaleStateError, UnsupportedError
from torbles.mathematics import AlgebraBudgetError

SEED = 20261102
getcontext().prec = 60
TOL = Decimal(10) ** -45                     # decimal oracle vs exact enclosures (widths ~1e-32, oracle error 1e-55)
EMPIRICAL = []                               # (kernel, D, seed, op, empirical error, stated bound): RFF receipts


# ------------------------------------------------------------------ the decimal oracle
def D(q):
    q = Fraction(q)
    return Decimal(q.numerator) / Decimal(q.denominator)


def _atan_inv(x):
    s, t, k, x2 = Decimal(0), Decimal(1) / x, 0, Decimal(x * x)
    while abs(t) > Decimal(10) ** -70:
        s += t / (2 * k + 1) if k % 2 == 0 else -t / (2 * k + 1)
        t /= x2
        k += 1
    return s


PI = 4 * (4 * _atan_inv(5) - _atan_inv(239))


def dcos(x):
    r = x - (x / (2 * PI)).to_integral_value() * 2 * PI
    s, t, i = Decimal(0), Decimal(1), 0
    while abs(t) > Decimal(10) ** -70:
        s += t
        i += 2
        t = -t * r * r / (i * (i - 1))
    return s


def dsin(x):
    r = x - (x / (2 * PI)).to_integral_value() * 2 * PI
    s, t, i = Decimal(0), r, 1
    while abs(t) > Decimal(10) ** -70:
        s += t
        i += 2
        t = -t * r * r / (i * (i - 1))
    return s


def k_exact(kind, gamma, x, y):
    d = [a - b for a, b in zip(x, y)]
    if kind == "gaussian":
        return (-D(gamma * sum(t * t for t in d))).exp()
    if kind == "laplace":
        return (-D(gamma * sum(abs(t) for t in d))).exp()
    return D(math.prod(Fraction(1) / (1 + gamma * gamma * t * t) for t in d))


def k_taylor(gamma, N, x, y):
    s = 2 * gamma * sum(a * b for a, b in zip(x, y))
    poly = sum(s ** k / math.factorial(k) for k in range(N + 1))
    return (-D(gamma * sum(a * a for a in x))).exp() * (-D(gamma * sum(b * b for b in y))).exp() * D(poly)


def k_rff(freqs, count, x, y):
    return sum(m * dcos(sum(D(o) * D(a - b) for o, a, b in zip(om, x, y))) for om, m in freqs) / count


def quantities(rows, kf, points=()):
    """the quantities of two groups' rows [(x, w)] under kernel kf: Decimal. The diagonal is the exact kernel's,
    k(x, x) = 1 for the three kernels (the estimators approximate the double sums and subtract sum w^2 exactly)"""
    A, B = rows
    K = lambda P, Q: sum(D(w * v) * kf(x, y) for x, w in P for y, v in Q)
    diag = lambda P: sum(D(w * w) for x, w in P)
    Wa, Wb = sum(w for _, w in A), sum(w for _, w in B)
    Qa, Qb = sum(w * w for _, w in A), sum(w * w for _, w in B)
    kaa, kbb, kab = K(A, A), K(B, B), K(A, B)
    da, db = diag(A), diag(B)
    out = dict(sum_ab=kab, sum_aa=kaa - da, sum_aa_self=kaa, sum_aa_unordered=(kaa - da) / 2,
               sum_aa_unordered_self=(kaa + da) / 2, mean_ab=kab / D(Wa * Wb),
               mean_aa=(kaa - da) / D(Wa * Wa - Qa),
               mmd2_biased=kaa / D(Wa * Wa) + kbb / D(Wb * Wb) - 2 * kab / D(Wa * Wb),
               mmd2_unbiased=(kaa - da) / D(Wa * Wa - Qa) + (kbb - db) / D(Wb * Wb - Qb) - 2 * kab / D(Wa * Wb))
    for i, z in enumerate(points):
        out[f"witness{i}"] = sum(D(w) * kf(x, z) for x, w in A) / D(Wa)
    return out


REQUESTS = [("sum_ab", dict(op="sum", a=0, b=1)), ("sum_aa", dict(op="sum", a=0)),
            ("sum_aa_self", dict(op="sum", a=0, self_pairs=True)),
            ("sum_aa_unordered", dict(op="sum", a=0, ordered=False)),
            ("sum_aa_unordered_self", dict(op="sum", a=0, self_pairs=True, ordered=False)),
            ("mean_ab", dict(op="mean", a=0, b=1)), ("mean_aa", dict(op="mean", a=0)),
            ("mmd2_biased", dict(op="mmd2", a=0, b=1, unbiased=False)),
            ("mmd2_unbiased", dict(op="mmd2", a=0, b=1, unbiased=True))]


def make_scope(rng, n, d, scale=1, lo=-9, hi=10, weights=True, groups=2, negative=False):
    X = rng.integers(lo, hi, (groups * n, d)).astype(np.int64)
    W = rng.integers(1, 4, groups * n).astype(np.int64)
    if negative:
        W[::3] *= -1
    g = np.repeat(np.arange(groups), n)
    names = [f"x{j}" for j in range(d)]
    cols = {names[j]: X[:, j] for j in range(d)}
    if weights:
        cols["w"] = W
    scope = mbk.Scope.from_arrays(cols, scales={c: scale for c in names}, groups=g, ngroups=groups)
    u = Fraction(10) ** scale
    rows = [[(tuple(Fraction(int(v)) / u for v in X[i]), Fraction(int(W[i])) if weights else Fraction(1))
             for i in range(groups * n) if g[i] == k] for k in range(groups)]
    return scope, names, rows, X


def in_iv(value, iv, tol=TOL):
    return D(iv[0]) - tol <= value <= D(iv[1]) + tol


class Kapprox(unittest.TestCase):
    def check(self, ans, approx, exact, certified=True, tag=None):
        """approx in the enclosure, exact in the bound, the error the sum of its terms; the empirical error"""
        self.assertEqual(ans.status, 0)
        self.assertFalse(ans.empty)
        self.assertFalse(ans.exact)
        self.assertEqual(ans.label, "certified" if certified else "probabilistic")
        self.assertTrue(in_iv(approx, ans.value), (tag, approx, ans.value))
        self.assertTrue(in_iv(exact, ans.bound), (tag, exact, ans.bound))
        self.assertEqual(ans.error, sum(t for _, t in ans.terms))
        self.assertEqual(ans.bound, (ans.value[0] - ans.error, ans.value[1] + ans.error))
        emp = abs(exact - approx)
        self.assertLessEqual(emp, D(ans.error) + TOL, tag)
        return emp

    def equal_builds(self, a, b):
        """two builds (CPU and CUDA) hold the same sums, statistics and answers"""
        for g in range(a.info["groups"]):
            self.assertEqual(a.sums(g), b.sums(g))
            self.assertEqual(a.group(g), b.group(g))
        self.assertEqual(a.info["maxsq"], b.info["maxsq"])
        self.assertEqual(a.info["id"], b.info["id"])
        self.assertEqual(a.record, b.record)
        self.assertEqual(b.backend, "cuda-emulated")
        x, y = a.mmd2(0, 1), b.mmd2(0, 1)
        self.assertEqual((x.value, x.error, x.terms), (y.value, y.error, y.terms))

    # ------------------------------------------------------------------ deterministic features
    def test_kernel002_taylor_gaussian_matches_direct_kernel_matrices(self):
        rng = np.random.default_rng(SEED)
        for d, scale, n, gamma, degrees in ((2, 1, 5, Fraction(1, 3), (4, 8, 12)), (3, 2, 4, Fraction(2, 5), (6, 10))):
            scope, names, rows, X = make_scope(rng, n, d, scale, lo=-90 if scale == 2 else -9, hi=91 if scale == 2 else 10)
            pts = [tuple(int(v) for v in X[1]), tuple(int(v) // 2 for v in X[-1])]
            zs = [tuple(Fraction(v) / 10 ** scale for v in p) for p in pts]
            exact = quantities(rows, lambda x, y: k_exact("gaussian", gamma, x, y), zs)
            for N in degrees:
                st = ka.features(scope, names, ka.gaussian(gamma), method="taylor", degree=N, weight="w")
                self.equal_builds(st, ka.features(scope, names, ka.gaussian(gamma), method="taylor", degree=N,
                                                  weight="w", backend="cuda"))
                approx = quantities(rows, lambda x, y: k_taylor(gamma, N, x, y), zs)
                answers = st.query([q for _, q in REQUESTS] + [dict(op="witness", a=0, point=i) for i in range(2)],
                                   points=pts)
                tags = [t for t, _ in REQUESTS] + ["witness0", "witness1"]
                for tag, ans in zip(tags, answers):
                    self.check(ans, approx[tag], exact[tag], tag=(d, N, tag))
                    self.assertEqual(ans.terms[0][0], "remainder")
                eps = st.q("per_pair")
                rho2 = max(sum(t * t for t in x) for grp in rows for x, _ in grp)
                self.assertEqual(st.q("radius2"), rho2)                # the data's largest |x|^2, exact
                self.assertEqual(st.q("U"), 2 * gamma * rho2)
                # the per-pair bound holds on every pair of the direct kernel matrix
                worst = max(abs(k_exact("gaussian", gamma, x, y) - k_taylor(gamma, N, x, y))
                            for grp in rows for x, _ in grp for grp2 in rows for y, _ in grp2)
                self.assertLessEqual(worst, D(eps))
                rec = st.record
                self.assertEqual((rec["label"], rec["provenance"], rec["norm"], rec["degree"]),
                                 ("certified", "deterministic", "abs", N))
                self.assertIn("domain", rec["assumptions"])
                self.assertFalse(rec["exact"])
                pairs = st.sum(0, 1).pairs
                self.assertEqual(pairs, sum(w for _, w in rows[0]) * sum(w for _, w in rows[1]))

    def test_kernel002_taylor_bound_is_tight_monotone_and_needs_its_domain(self):
        # one row at |x| = rho and the witness at x itself: k = 1, k_N = e^-U P_N(U), the worst case of the bound
        gamma = Fraction(1, 2)
        scope = mbk.Scope.from_arrays({"x": np.array([12, 3]), "y": np.array([-5, 4])}, scales={"x": 1, "y": 1},
                                      groups=np.array([0, 1]), ngroups=2)
        x = (Fraction(12, 10), Fraction(-5, 10))
        prev = None
        for N in (2, 4, 6, 8, 10, 12):
            st = ka.features(scope, ["x", "y"], ka.gaussian(gamma), method="taylor", degree=N)
            eps = st.q("per_pair")
            U = st.q("U")
            self.assertEqual(U, 2 * gamma * Fraction(169, 100))
            w = st.witness(0, [(12, -5)])[0]
            observed = 1 - D(sum(U ** k / math.factorial(k) for k in range(N + 1))) * (-D(U)).exp()
            self.check(w, 1 - observed, Decimal(1), tag=N)
            self.assertLessEqual(observed, D(eps))
            self.assertGreaterEqual(observed, D(eps) / 2)              # tight: within a factor 2 of the worst case
            if prev is not None:
                self.assertLess(eps, prev)
            prev = eps
        # a declared radius: larger than the data's, a larger bound; smaller: the data leave it (refused)
        st = ka.features(scope, ["x", "y"], ka.gaussian(gamma), method="taylor", degree=8, radius2=4)
        self.assertEqual(st.q("radius2"), 4)
        self.assertTrue(st.record["radius_declared"])
        self.assertGreater(st.q("per_pair"), ka.features(scope, ["x", "y"], ka.gaussian(gamma), method="taylor",
                                                         degree=8).q("per_pair"))
        st.witness(0, [(19, 5)])                                       # |z|^2 = 3.86 <= 4: inside the domain
        with self.assertRaises(DomainError):
            st.witness(0, [(20, 1)])                                   # |z|^2 = 4.01 > 4: outside, refused
        with self.assertRaises(DomainError):
            ka.features(scope, ["x", "y"], ka.gaussian(gamma), method="taylor", degree=8, radius2=Fraction(168, 100))
        ka.features(scope, ["x", "y"], ka.gaussian(gamma), method="taylor", degree=8, radius2=Fraction(169, 100))
        on_edge = ka.features(scope, ["x", "y"], ka.gaussian(gamma), method="taylor", degree=8)
        on_edge.witness(0, [(13, 0)])                                  # |z|^2 = 1.69: the data's radius, inside
        with self.assertRaises(DomainError):                          # beyond the data's radius
            on_edge.witness(0, [(13, 1)])
        with self.assertRaises(UnsupportedError):                     # the Gaussian factor underflows past 2^-(2^24)
            ka.features(scope, ["x", "y"], ka.gaussian(10 ** 8), method="taylor", degree=4)

    # ------------------------------------------------------------------ random Fourier features
    def test_kernel002_rff_matches_direct_within_probabilistic_bound(self):
        rng = np.random.default_rng(SEED + 1)
        scope, names, rows, X = make_scope(rng, 4, 2, 1)
        pts = [tuple(int(v) for v in X[2])]
        zs = [tuple(Fraction(v, 10) for v in pts[0])]
        allrows = [x for grp in rows for x, _ in grp] + zs
        for kind, gamma in (("gaussian", Fraction(1, 2)), ("laplace", Fraction(1, 2)), ("cauchy", Fraction(2, 3))):
            exact = quantities(rows, lambda x, y: k_exact(kind, gamma, x, y), zs)
            for count in (16, 64, 256):
                st = ka.features(scope, names, getattr(ka, kind)(gamma), method="rff", features=count, seed=7 + count,
                                 weight="w", failure=Fraction(1, 10 ** 6))
                if count == 64:
                    self.equal_builds(st, ka.features(scope, names, getattr(ka, kind)(gamma), method="rff",
                                                      features=count, seed=7 + count, weight="w",
                                                      failure=Fraction(1, 10 ** 6), backend="cuda"))
                freqs = st.frequencies
                self.assertEqual(sum(m for _, m in freqs), count)
                cache = {}
                for i, x in enumerate(allrows):                      # the approximant's matrix, once per pair
                    for j, y in enumerate(allrows):
                        if (j, i) in cache:
                            cache[(i, j)] = cache[(j, i)]
                        else:
                            cache[(i, j)] = k_rff(freqs, count, x, y)
                index = {x: i for i, x in enumerate(allrows)}
                approx = quantities(rows, lambda x, y: cache[(index[x], index[y])], zs)
                answers = st.query([q for _, q in REQUESTS] + [dict(op="witness", a=0, point=0)], points=pts)
                for tag, ans in zip([t for t, _ in REQUESTS] + ["witness0"], answers):
                    emp = self.check(ans, approx[tag], exact[tag], certified=False, tag=(kind, count, tag))
                    self.assertEqual([t for t, _ in ans.terms], ["probabilistic", "quantization"])
                    self.assertEqual(ans.failure, Fraction(1, 10 ** 6))
                    EMPIRICAL.append((kind, count, 7 + count, tag, float(emp), float(ans.error)))
                rec = st.record
                self.assertEqual((rec["label"], rec["provenance"], rec["seed"], rec["count"]),
                                 ("probabilistic", "randomized", 7 + count, count))
                self.assertIn("independent", rec["assumptions"])
                # the per-pair bound 2 h + beta, h = sqrt(ln(2 / delta) / (2 D)) bounded above exactly
                h = st.q("hoeffding")
                self.assertGreaterEqual(D(h), (D(2 * 10 ** 6).ln() / (2 * count)).sqrt())
                self.assertLess(D(h), (D(2 * 10 ** 6).ln() / (2 * count)).sqrt() * (1 + Decimal(10) ** -15))
                self.assertEqual(st.q("per_pair"), 2 * h + st.q("bias"))
                self.assertLess(st.q("bias"), Fraction(1, 10 ** 8))   # the frequency bias: negligible, but stated

    def test_kernel002_generator_reproduces_and_matches_independent_transforms(self):
        M64 = (1 << 64) - 1

        def sm64(z):
            z = (z + 0x9E3779B97F4A7C15) & M64
            z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M64
            z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M64
            return z ^ (z >> 31)

        def u128(key, i):
            return sm64((key + i) & M64) | sm64((key + i + 1) & M64) << 64

        def uniform(N):
            return D(Fraction(2 * N + 1, 1 << 129))

        for kind, gamma, d in (("gaussian", Fraction(3, 2), 3), ("laplace", Fraction(1, 3), 2),
                               ("cauchy", Fraction(5, 4), 2)):
            kern = getattr(ka, kind)(gamma)
            draws, fbits = ka.generate(kern, d, 40, 99)
            again, fb2 = ka.generate(kern, d, 40, 99)
            self.assertEqual((draws, fbits), (again, fb2))            # deterministic from the seed
            self.assertNotEqual(draws, ka.generate(kern, d, 40, 100)[0])
            st = ka.features(mbk.Scope.from_arrays({"a": np.array([1, -2]), "b": np.array([3, 0]),
                                                    "c": np.array([0, 1])}), ["a", "b", "c"][:d], kern, method="rff",
                             features=40, seed=99)
            eta_num, step = st.q("eta_num"), Fraction(1, 2 ** (fbits + 1))
            self.assertEqual(st.info["fbits"], fbits)
            key = sm64(99 ^ 0xD1B54A32D192ED03)
            for f, om in enumerate(draws):
                for j, w in enumerate(om):
                    self.assertEqual((w * 2 ** fbits).denominator, 1)   # dyadic: M / 2^fbits
                    if kind == "gaussian":
                        q, k = divmod(j, 2)
                        base = 4 * (f * ((d + 1) // 2) + q)
                        U1, U2 = uniform(u128(key, base)), uniform(u128(key, base + 2))
                        r = (-2 * U1.ln()).sqrt()
                        z = r * (dcos(2 * PI * U2) if k == 0 else dsin(2 * PI * U2))
                        ideal = D(2 * gamma).sqrt() * z
                    elif kind == "laplace":
                        N = u128(key, 2 * (f * d + j))
                        if N < 1 << 96 or N >= (1 << 128) - (1 << 96):
                            self.assertEqual(w, 0)                     # a bad cell (probability 2^-31): omega~ = 0
                            continue
                        U = uniform(N)
                        ideal = D(gamma) * dcos(PI * U) / dsin(PI * U)
                    else:
                        base = 3 * (f * d + j)
                        U = uniform(u128(key, base))
                        sign = -1 if sm64((key + base + 2) & M64) >> 63 else 1
                        ideal = sign * D(gamma) * -U.ln()
                    self.assertLessEqual(abs(D(w) - ideal), D(eta_num + step), (kind, f, j))
            # the state holds exactly the sorted distinct draws with their multiplicities
            want = {}
            for om in draws:
                want[om] = want.get(om, 0) + 1
            self.assertEqual(st.frequencies, sorted(want.items()))
            self.assertEqual(st.record["generator"], ka.GENERATOR)

    # ------------------------------------------------------------------ persistence and labels
    def test_kernel002_persisted_parameters_and_seeds_give_identical_answers(self):
        rng = np.random.default_rng(SEED + 2)
        scope, names, rows, X = make_scope(rng, 6, 2, 1)
        other, _, _, _ = make_scope(rng, 6, 2, 1)
        pts = [tuple(int(v) for v in X[0])]
        reqs = [q for _, q in REQUESTS] + [dict(op="witness", a=1, point=0)]
        states = [ka.features(scope, names, ka.gaussian(Fraction(1, 4)), method="taylor", degree=9, weight="w"),
                  ka.features(scope, names, ka.laplace(Fraction(1, 2)), method="rff", features=48, seed=5, weight="w"),
                  ka.features(scope, names, ka.gaussian(Fraction(1, 4)), method="rff", features=32, seed=6,
                              weight="w", backend="cuda", error_budget=Fraction(1, 1))]
        for st in states:
            data = st.dumps(metadata={"run": "persist"})
            self.assertEqual(data, st.dumps(metadata={"run": "persist"}))   # the same state: the same bytes
            self.assertEqual(struct.unpack_from("<8sIIqII", data, 0)[:2], (b"TORBMATH", 1))
            self.assertEqual(struct.unpack_from("<IIII", data, 24), (0x4b02, 1, 8, 0))   # type, version, carrier, exact 0
            want = [(a.value, a.error, a.terms, a.bound, a.failure, a.pairs) for a in st.query(reqs, points=pts)]
            for origin in (scope, None):
                back = ka.loads(data, origin)
                got = [(a.value, a.error, a.terms, a.bound, a.failure, a.pairs) for a in back.query(reqs, points=pts)]
                self.assertEqual(got, want)
                self.assertEqual(back.record, st.record)
                self.assertEqual(back.frequencies, st.frequencies)
                self.assertEqual(back.info["id"], st.info["id"])
                self.assertEqual(back.user_metadata, {"run": "persist"})
                self.assertEqual(back.dumps(metadata={"run": "persist"}), data)   # reloaded, written again: identical
                self.assertTrue(back.fresh(scope)["fresh"])
            rec = st.record
            if rec["method"] == "rff":                                # the seed persisted regenerates the frequencies
                draws, _ = ka.generate(ka.Kernel(rec["kernel"], rec["gamma"]), 2, rec["count"], rec["seed"])
                want_f = {}
                for om in draws:
                    want_f[om] = want_f.get(om, 0) + 1
                self.assertEqual(ka.loads(data).frequencies, sorted(want_f.items()))
            self.assertFalse(st.fresh(other)["fresh"])
            with self.assertRaises(mp.StaleSourceError):
                ka.loads(data, other)
            bad = bytearray(data)
            bad[len(bad) // 2] ^= 1
            with self.assertRaises(mp.CorruptStateError):
                ka.loads(bytes(bad))
            with self.assertRaises(mp.TruncatedStateError):
                ka.loads(data[:-8])
            with self.assertRaises(mp.MalformedStateError):            # FND-011's typed loader refuses the type
                mp.loads(data)
            with self.assertRaises(mp.MalformedStateError):
                mp.validate(data)
        # an FND-011 record of an exact state (a big integer) is not a KERNEL-002 state
        from torbles import math_carriers as mc
        with self.assertRaises(mp.WrongTypeError):
            ka.loads(mp.dumps(mc.BigInt(12345)))

    def test_kernel002_approximate_states_never_authorize_exact_rewrites(self):
        rng = np.random.default_rng(SEED + 3)
        scope, names, rows, X = make_scope(rng, 4, 2, 1)
        taylor = ka.features(scope, names, ka.gaussian(1), method="taylor", degree=6)
        rff = ka.features(scope, names, ka.cauchy(1), method="rff", features=16, seed=1)
        for st, reason, label in ((taylor, "approximate", "certified"), (rff, "statistical", "probabilistic")):
            with self.assertRaises(NotAuthorizedError) as cm:
                st.authorize_exact()
            self.assertEqual(cm.exception.reason, reason)
            d = st.decide(field=(1 << 61) - 1)
            self.assertEqual((d["authorized"], d["reason"], d["label"], d["code"]), (False, reason, label, -6))
            for a in st.query([q for _, q in REQUESTS]):
                self.assertFalse(a.exact)
                self.assertEqual(a.label, label)
                self.assertNotEqual(a.bound[0], a.bound[1])            # an interval with its error, never a point
            self.assertFalse(st.record["exact"])
        # KERNEL-001's exact machinery never approximates an RBF kernel: refused, and so is an approximation method
        with self.assertRaises(UnsupportedError):
            ka.features(scope, names, mk.kernel("rbf", gamma=1))
        with self.assertRaises(ValueError):
            ka.features(scope, names, mk.polynomial(2), method="rff")
        with self.assertRaises(mp.MalformedStateError):               # nor does FND-011 load it as an exact state
            mp.loads(taylor.dumps())
        import subprocess
        import sys
        out = subprocess.run([sys.executable, "-c", "import sys, torbles; print('torbles.math_kapprox' in sys.modules)"],
                             capture_output=True, text=True, env=dict(os.environ))
        self.assertEqual(out.stdout.strip(), "False")                 # opt-in: import torbles does not load it

    def test_kernel002_exact_polynomial_kernels_answered_by_kernel001(self):
        rng = np.random.default_rng(SEED + 4)
        X = rng.integers(-5, 6, (12, 3)).astype(np.int64)
        W = rng.integers(1, 4, 12).astype(np.int64)
        g = np.repeat(np.arange(2), 6)
        scope = mbk.Scope.from_arrays({"a": X[:, 0], "b": X[:, 1], "c": X[:, 2], "w": W}, groups=g, ngroups=2)
        for kern, kf in ((mk.polynomial(2, c=1), lambda x, y: (int(x @ y) + 1) ** 2),
                         (mk.homogeneous(3), lambda x, y: int(x @ y) ** 3)):
            for backend in ("cpu", "cuda"):
                st = ka.features(scope, ["a", "b", "c"], kern, weight="w", backend=backend)
                self.assertIsInstance(st, mk.FeatureState)             # KERNEL-001's exact state
                A, B = [(X[i], int(W[i])) for i in range(6)], [(X[i], int(W[i])) for i in range(6, 12)]
                kab = sum(w * v * kf(x, y) for x, w in A for y, v in B)
                kaa = sum(w * v * kf(x, y) for x, w in A for y, v in A)
                kbb = sum(w * v * kf(x, y) for x, w in B for y, v in B)
                da, db = sum(w * w * kf(x, x) for x, w in A), sum(w * w * kf(x, x) for x, w in B)
                Wa, Wb, Qa, Qb = sum(W[:6]), sum(W[6:]), sum(W[:6] ** 2), sum(W[6:] ** 2)
                self.assertEqual(st.sum(0, 1)["value"], kab)
                self.assertEqual(st.sum(0)["value"], kaa - da)
                self.assertEqual(st.mmd2(0, 1), Fraction(kaa - da, Wa * Wa - Qa) + Fraction(kbb - db, Wb * Wb - Qb)
                                 - 2 * Fraction(kab, Wa * Wb))
                direct = mk.features(scope, ["a", "b", "c"], kern, weight="w")
                self.assertEqual(st.mmd2(0, 1, unbiased=False), direct.mmd2(0, 1, unbiased=False))

    # ------------------------------------------------------------------ CUDA, NULLs, signed weights
    def test_kernel002_cuda_build_equals_cpu_with_nulls_and_signed_weights(self):
        rng = np.random.default_rng(SEED + 5)
        n = 30
        X = rng.integers(-40, 41, (n, 3)).astype(np.int64)
        W = rng.integers(-3, 4, n).astype(np.int64)
        valid = rng.random(n) > 0.15
        g = rng.integers(0, 3, n)
        chosen = rng.random(n) > 0.1
        scope = mbk.Scope.from_arrays({"a": X[:, 0], "b": X[:, 1], "c": X[:, 2], "w": W}, valid={"b": valid},
                                      scales={"a": 2, "b": 2, "c": 2}, groups=g, ngroups=3, rows=chosen)
        for kern, method, opts in ((ka.gaussian(Fraction(3, 1)), "taylor", dict(degree=7)),
                                   (ka.gaussian(Fraction(3, 1)), "rff", dict(features=40, seed=3)),
                                   (ka.laplace(2), "rff", dict(features=24, seed=4)),
                                   (ka.cauchy(Fraction(1, 7)), "rff", dict(features=24, seed=5))):
            cpu = ka.features(scope, ["a", "b", "c"], kern, method=method, weight="w", **opts)
            gpu = ka.features(scope, ["a", "b", "c"], kern, method=method, weight="w", backend="cuda", **opts)
            self.equal_builds(cpu, gpu)
            for k in range(3):                                         # listwise: a NULL feature drops the row
                gr = cpu.group(k)
                rows_k = [i for i in range(n) if g[i] == k and chosen[i]]
                ok = [i for i in rows_k if valid[i]]
                self.assertEqual((gr["rows"], gr["valid"]), (len(rows_k), len(ok)))
                self.assertEqual(gr["weight"], sum(int(W[i]) for i in ok))
                self.assertEqual(gr["abs_weight"], sum(abs(int(W[i])) for i in ok))
                self.assertEqual(gr["squares"], sum(int(W[i]) ** 2 for i in ok))
            # the exact quantities of signed weights stay inside the bounds
            u = Fraction(100)
            rows = [[(tuple(Fraction(int(v)) / u for v in X[i]), Fraction(int(W[i])))
                     for i in range(n) if g[i] == k and valid[i] and chosen[i]]
                    for k in range(2)]
            kf = (lambda x, y: k_exact("gaussian", Fraction(3), x, y)) if kern.kind == "gaussian" else (
                (lambda x, y: k_exact("laplace", Fraction(2), x, y)) if kern.kind == "laplace" else
                (lambda x, y: k_exact("cauchy", Fraction(1, 7), x, y)))
            exact = quantities(rows, kf)
            for tag in ("sum_ab", "mean_ab", "mmd2_biased"):
                ans = cpu.query([dict(REQUESTS)[tag]])[0]
                self.assertTrue(in_iv(exact[tag], ans.bound), (method, tag))

    # ------------------------------------------------------------------ budgets
    def test_kernel002_budgets_at_limit_and_one_past(self):
        rng = np.random.default_rng(SEED + 6)
        scope, names, rows, X = make_scope(rng, 1000, 3, 1)
        kern = ka.gaussian(Fraction(1, 2))
        items = math.comb(3 + 6, 6)
        need = ka.need(scope, names, kern, method="taylor", degree=6, weight="w")
        self.assertEqual(need["support"], items)
        self.assertEqual(need["tensor"], 2000 * items)
        self.assertEqual(need["states"], 2 * items)
        for res, val in (("support", items), ("tensor", 2000 * items), ("states", 2 * items), ("degree", 6)):
            ka.features(scope, names, kern, method="taylor", degree=6, weight="w", budget=Budget(**{res: val}))
            with self.assertRaises(AlgebraBudgetError) as cm:
                ka.features(scope, names, kern, method="taylor", degree=6, weight="w", budget=Budget(**{res: val - 1}))
            self.assertEqual((cm.exception.resource, cm.exception.need, cm.exception.limit), (res, val, val - 1))
        # RFF: support = D, iterations = D d (the draws)
        for res, val in (("support", 50), ("iterations", 150)):
            ka.features(scope, names, kern, method="rff", features=50, seed=1, budget=Budget(**{res: val}))
            with self.assertRaises(AlgebraBudgetError) as cm:
                ka.features(scope, names, kern, method="rff", features=50, seed=1, budget=Budget(**{res: val - 1}))
            self.assertEqual(cm.exception.resource, res)
        # requests: output = requests, tensor = points x items
        st = ka.features(scope, names, kern, method="taylor", degree=6, weight="w")
        reqs = [dict(op="witness", a=0, point=i % 3) for i in range(5)]
        pts = [tuple(int(v) for v in X[i]) for i in range(3)]
        st.query(reqs, points=pts, budget=Budget(output=5, tensor=3 * items))
        with self.assertRaises(AlgebraBudgetError):
            st.query(reqs, points=pts, budget=Budget(output=4))
        with self.assertRaises(AlgebraBudgetError):
            st.query(reqs, points=pts, budget=Budget(tensor=3 * items - 1))
        # CUDA: the device copies, checked by the device entry point itself (the plan made with the defaults)
        import ctypes as C
        from torbles.math_budgets import _Refusal
        for method, opts in (("taylor", dict(degree=6)), ("rff", dict(features=50, seed=1))):
            dev = ka.need(scope, names, kern, method=method, weight="w", backend="cuda", **opts)["bytes"]
            ka.features(scope, names, kern, method=method, weight="w", backend="cuda",
                        budget=Budget(bytes=max(dev, ka.need(scope, names, kern, method=method, weight="w",
                                                             **opts)["bytes"])), **opts)
            s = ka._spec(scope, names, kern, method, opts.get("degree", 0), opts.get("features", 0), opts.get("seed", 0),
                         Fraction(1, 100), None, None, "w")
            error, refusal = C.c_int32(), _Refusal()
            plan = ka._api()["tr_kapprox_plan_create"](scope._handle, C.byref(s), None, C.byref(refusal), C.byref(error))
            try:
                pv = ka._PlanView()
                ka._api()["tr_kapprox_plan_get_view"](plan, C.byref(pv))
                G, n = int(pv.ngroups), int(pv.nitems)
                out = [(ka._Group * G)(), (C.c_uint64 * 3)(), np.zeros((G * n * 2, 2), np.int64), np.zeros(G, np.uint64),
                       np.zeros(G, np.uint64), np.zeros((G, 2), np.int64)]
                gpu, view = ka._gpu(), scope.view()
                for limit, ok in ((dev, True), (dev - 1, False)):
                    refusal = _Refusal()
                    rc = gpu.lib().tr_mgpu_kapprox_sums(C.byref(pv), C.byref(view), out[0], out[1], out[2].ctypes.data,
                                                        out[3].ctypes.data, out[4].ctypes.data, out[5].ctypes.data,
                                                        C.byref(Budget(bytes=limit).raw), C.byref(refusal))
                    self.assertEqual(rc, 0 if ok else -5, (method, limit))
                    if not ok:
                        self.assertEqual((refusal.resource, refusal.need, refusal.limit), (8, dev, dev - 1))
            finally:
                ka._api()["tr_kapprox_plan_free"](plan)

    # ------------------------------------------------------------------ refusals and the error budget
    def test_kernel002_refusals_and_error_budget(self):
        rng = np.random.default_rng(SEED + 7)
        scope, names, rows, X = make_scope(rng, 5, 2, 1)
        g = ka.gaussian(Fraction(1, 4))
        st = ka.features(scope, names, g, method="taylor", degree=8)
        eps = st.q("per_pair")                                         # budgets are int64 rationals: the nearest ones
        above, below = Fraction(math.ceil(eps * 10 ** 15), 10 ** 15), Fraction(math.floor(eps * 10 ** 15), 10 ** 15)
        self.assertTrue(below < eps <= above)
        ka.features(scope, names, g, method="taylor", degree=8, error_budget=above)
        with self.assertRaises(NotConvergedError):                    # the certified bound exceeds the budget
            ka.features(scope, names, g, method="taylor", degree=8, error_budget=below)
        r = ka.features(scope, names, g, method="rff", features=64, seed=2)
        half = Fraction(math.floor(r.q("per_pair") * 10 ** 6), 2 * 10 ** 6)
        with self.assertRaises(NotConvergedError):
            ka.features(scope, names, g, method="rff", features=64, seed=2, error_budget=half)
        with self.assertRaises(UnsupportedError):                     # no deterministic features for Laplace / Cauchy
            ka.features(scope, names, ka.laplace(1), method="taylor", degree=4)
        with self.assertRaises(UnsupportedError):
            ka.features(scope, names, g, method="taylor", degree=13)
        with self.assertRaises(ValueError):
            ka.features(scope, names, g, method="taylor", degree=-1)
        with self.assertRaises(UnsupportedError):
            ka.features(scope, names, g, method="rff", features=(1 << 20) + 1)
        for bad in (0, 1, Fraction(3, 2)):
            with self.assertRaises(ValueError):
                ka.features(scope, names, g, method="rff", features=8, failure=bad)
        for bad in (0, -1):
            with self.assertRaises(ValueError):
                ka.features(scope, names, ka.gaussian(bad), method="taylor", degree=2)
        with self.assertRaises(ValueError):
            ka.features(scope, names, g, method="other")
        with self.assertRaises(ValueError):
            ka.features(scope, names, g, method="rff", features=0)
        with self.assertRaises(UnsupportedError):                     # frequencies past 2^60
            ka.features(scope, names, ka.laplace(1 << 31), method="rff", features=4)
        mixed = mbk.Scope.from_arrays({"a": np.array([1, 2]), "b": np.array([3, 4])}, scales={"a": 1, "b": 2})
        with self.assertRaises(DomainError):                          # |x|^2 needs one decimal scale
            ka.features(mixed, ["a", "b"], g, method="rff", features=4)
        wide = mbk.Scope.from_arrays({f"c{j}": np.array([1, 2]) for j in range(17)})
        with self.assertRaises(UnsupportedError):
            ka.features(wide, [f"c{j}" for j in range(17)], g, method="rff", features=4)
        with self.assertRaises(DomainError):
            r.witness(0, [(10 ** 6, 0)])
        with self.assertRaises(ValueError):
            st.query([dict(op="mmd2", a=0, b=5)])
        with self.assertRaises(TypeError):
            ka.gaussian(lengthscale=0.5)
        self.assertEqual(ka.gaussian(lengthscale=Fraction(1, 2)).gamma, 2)
        # a table scope that changed since: refused
        import pandas as pd
        db = torbles.from_ddl("CREATE TABLE t (x BIGINT, y BIGINT)", {"t": pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})})
        t = db["t"]
        ts = mbk.Scope.from_table(t, ["x", "y"])
        ka.features(ts, ["x", "y"], g, method="rff", features=4)
        t.flip(0, 0)
        with self.assertRaises(StaleStateError):
            ka.features(ts, ["x", "y"], g, method="rff", features=4)

    # ------------------------------------------------------------------ fixed-point constants
    def test_kernel002_fixed_point_constants_match_decimal(self):
        api = ka._api()
        scope = mbk.Scope.from_arrays({"x": np.array([123, -45]), "y": np.array([7, 300])}, scales={"x": 2, "y": 2})
        for gamma in (Fraction(1, 2), Fraction(7, 3), Fraction(10 ** 6, 7)):
            s = ka._spec(scope, ["x", "y"], ka.gaussian(gamma), "taylor", 5, 0, 0, None, None, None, None)
            import ctypes as C
            from torbles.math_budgets import _Refusal
            error, refusal = C.c_int32(), _Refusal()
            plan = api["tr_kapprox_plan_create"](scope._handle, C.byref(s), None, C.byref(refusal), C.byref(error))
            self.assertTrue(plan, error.value)
            try:
                pv = ka._PlanView()
                api["tr_kapprox_plan_get_view"](plan, C.byref(pv))
                kq = sum(int(pv.kq[i]) << (64 * i) for i in range(6))
                with localcontext() as ctx:
                    ctx.prec = 250
                    exact = D(gamma) * (Decimal(2) ** 320) / (Decimal(10) ** 4 * Decimal(2).ln())
                    self.assertLessEqual(kq, exact)                   # rounded down, within 2 units of 2^-320
                    self.assertLess(exact - kq, 2)
                    # tmax bounds floor(gamma |x|^2 / ln 2) on the domain E = sum_j max |X_j|^2 / 10^4
                    E = 123 ** 2 + 300 ** 2
                    self.assertGreaterEqual(int(pv.tmax), int(D(gamma) * Decimal(E) / Decimal(10 ** 4) / Decimal(2).ln()))
                # 2^b: the smallest power of two whose square covers E (every |X_j| and the domain)
                self.assertEqual(int(pv.shift), min(b for b in range(64) if 4 ** b >= E))
            finally:
                api["tr_kapprox_plan_free"](plan)


def tearDownModule():
    if EMPIRICAL and os.environ.get("TORBLES_KAPPROX_REPORT"):
        worst = max(EMPIRICAL, key=lambda r: r[4] / r[5])
        print(f"\nKERNEL-002 random features: {len(EMPIRICAL)} answers, largest empirical / stated = "
              f"{worst[4] / worst[5]:.3f} ({worst})")


if __name__ == "__main__":
    unittest.main()
