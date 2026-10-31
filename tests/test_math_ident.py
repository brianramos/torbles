# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""DISC-002: Moment-Gram polynomial identity discovery.

Oracles (independent of the implementation): the monomial basis enumerated with itertools, the Gram matrix summed row
by row with Python integers (and reduced modulo the primes in Python), its nullspace by Gauss-Jordan elimination over
Fractions written here, every identity evaluated row by row with Python integers, distinct points counted with Python
sets and DuckDB (COUNT(DISTINCT ...)), violating rows counted with DuckDB, span membership by Fraction elimination,
and FND-007's own verifier for the evidence of the same claims. Every native result is also computed on the CUDA path
(emulated here, threads in a scrambled order) and compared for equality. Seeds are fixed.
"""
import itertools
import math
import unittest
from fractions import Fraction

import duckdb
import numpy as np
import pandas as pd

import torbles
from torbles import math_bank as mbk
from torbles import math_certificates as mc
from torbles import math_ident as mi
from torbles import math_lin as lin
from torbles.math_budgets import Budget
from torbles.math_domains import DomainError
from torbles.math_errors import StaleStateError, UnsupportedError
from torbles.mathematics import AlgebraBudgetError

SEED = 2002


def obj(values):
    return pd.Series(values, dtype=object)


def sql(query, **frames):
    """one row of a DuckDB query over pandas frames"""
    con = duckdb.connect()
    for name, frame in frames.items():
        con.register(name, frame)
    return con.execute(query).fetchone()


def make(values, name="t"):
    frame = pd.DataFrame({c: obj(v) for c, v in values.items()})
    ddl = f"CREATE TABLE {name} (" + ", ".join(f'"{c}" BIGINT' for c in values) + ")"
    return torbles.from_ddl(ddl, {name: frame})


# ------------------------------------------------------------------ oracles
def basis_oracle(k, d):
    return sorted((e for e in itertools.product(range(d + 1), repeat=k) if sum(e) <= d), key=lambda e: (sum(e), e))


def mono(e, point):
    v = 1
    for x, p in zip(point, e):
        v *= x ** p
    return v


def gram_oracle(values, variables, basis, weight=None, mask=None):
    """the exact Gram over the selected rows where every variable (and the weight) is present"""
    n = len(values[variables[0]])
    B = len(basis)
    G = [[0] * B for _ in range(B)]
    domain = 0
    for r in range(n):
        if mask is not None and not mask[r]:
            continue
        pt = [values[v][r] for v in variables]
        w = 1 if weight is None else values[weight][r]
        if any(x is None for x in pt) or w is None:
            continue
        domain += 1
        m = [mono(e, pt) for e in basis]
        for i in range(B):
            for j in range(B):
                G[i][j] += w * m[i] * m[j]
    return G, domain


def rref_nullspace(G):
    """rank, pivots, free columns and the canonical nullspace basis by Gauss-Jordan over Fractions"""
    B = len(G)
    M = [[Fraction(x) for x in row] for row in G]
    piv, r = [], 0
    for c in range(B):
        p = next((i for i in range(r, len(M)) if M[i][c] != 0), None)
        if p is None:
            continue
        M[r], M[p] = M[p], M[r]
        inv = 1 / M[r][c]
        M[r] = [x * inv for x in M[r]]
        for i in range(len(M)):
            if i != r and M[i][c] != 0:
                f = M[i][c]
                M[i] = [a - f * b for a, b in zip(M[i], M[r])]
        piv.append(c)
        r += 1
    free = [c for c in range(B) if c not in piv]
    vecs = []
    for f in free:
        v = [Fraction(0)] * B
        v[f] = Fraction(1)
        for t, pc in enumerate(piv):
            v[pc] = -M[t][f]
        vecs.append(v)
    return len(piv), piv, free, vecs


def primitive(vec):
    den = 1
    for x in vec:
        den = den * x.denominator // math.gcd(den, x.denominator)
    ints = [int(x * den) for x in vec]
    g = 0
    for x in ints:
        g = math.gcd(g, x)
    return [x // g for x in ints] if g else ints


def evaluate(terms, point):
    return sum(c * mono(e, point) for e, c in terms)


def evidence_oracle(values, variables, terms, mask=None):
    """the exhaustive evidence of one identity (support variables NULL: excluded) with Python integers"""
    n = len(values[variables[0]])
    support = [q for q in range(len(variables)) if any(e[q] for e, _ in terms)] or list(range(len(variables)))
    checked = excluded = violations = 0
    witness = None
    for r in range(n):
        if mask is not None and not mask[r]:
            continue
        pt = [values[v][r] for v in variables]
        if any(pt[q] is None for q in support):
            excluded += 1
            continue
        pt = [0 if x is None else x for x in pt]
        checked += 1
        if evaluate(terms, pt) != 0:
            violations += 1
            witness = r if witness is None else min(witness, r)
    return dict(checked=checked, excluded=excluded, violations=violations, witness=witness)


def in_span(rows, vec):
    """vec in the row span of rows (Fraction elimination)"""
    def row_rank(vs):
        M = [list(map(Fraction, v)) for v in vs]
        r = 0
        cols = len(M[0]) if M else 0
        for c in range(cols):
            p = next((i for i in range(r, len(M)) if M[i][c] != 0), None)
            if p is None:
                continue
            M[r], M[p] = M[p], M[r]
            for i in range(len(M)):
                if i != r and M[i][c] != 0:
                    f = M[i][c] / M[r][c]
                    M[i] = [a - f * b for a, b in zip(M[i], M[r])]
            r += 1
        return r
    return row_rank(rows) == row_rank(rows + [vec])


def strip(ev):
    return {k: v for k, v in ev.items() if k not in ("backend", "device")}


def sieve(N):
    """omega, Omega, d, mu^2 for n = 1..N (a smallest-prime-factor sieve)"""
    spf = list(range(N + 1))
    for p in range(2, int(N ** 0.5) + 1):
        if spf[p] == p:
            for q in range(p * p, N + 1, p):
                if spf[q] == q:
                    spf[q] = p
    om, Om, d, sq = [0] * (N + 1), [0] * (N + 1), [1] * (N + 1), [1] * (N + 1)
    for n in range(2, N + 1):
        p, m, e = spf[n], n // spf[n], 1
        while m % p == 0:
            m //= p
            e += 1
        om[n], Om[n], d[n], sq[n] = om[m] + 1, Om[m] + e, d[m] * (e + 1), sq[m] * (1 if e == 1 else 0)
    return om, Om, d, sq


def planted(rng, n):
    x = [int(v) for v in rng.integers(-30, 31, n)]
    y = [int(v) for v in rng.integers(-25, 40, n)]
    u = [int(v) for v in rng.integers(-9, 10, n)]
    return {"a": [3 * v - 7 for v in x], "s": [p + q for p, q in zip(x, y)], "z": [p * q for p, q in zip(x, y)],
            "x": x, "y": y, "u": u}


PLANTED = ["a", "s", "z", "x", "y", "u"]
EXPECTED = [{(1, 0, 0, 0, 0, 0): 1, (0, 0, 0, 1, 0, 0): -3, (0, 0, 0, 0, 0, 0): 7},      # a - 3x + 7
            {(0, 1, 0, 0, 0, 0): 1, (0, 0, 0, 1, 0, 0): -1, (0, 0, 0, 0, 1, 0): -1},     # s - x - y
            {(0, 0, 0, 1, 1, 0): 1, (0, 0, 1, 0, 0, 0): -1}]                             # x*y - z


class Disc002(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(SEED)

    # ---------------------------------------------------------- plan, Gram, nullspace
    def test_disc002_basis_and_moments_match_the_enumeration(self):
        db = make({"x": [1, 2, 3], "y": [4, 5, 6], "z": [7, 8, 9]})
        scope = mbk.Scope.from_table(db["t"], ["x", "y", "z"])
        for d in range(4):
            p = mi.plan(scope, ["x", "y", "z"], degree=d)
            want = basis_oracle(3, d)
            self.assertEqual(p.basis, want)
            pairs = sorted({tuple(a + b for a, b in zip(want[i], want[j])) for i in range(len(want))
                            for j in range(i, len(want))}, key=lambda e: (sum(e), e))
            self.assertEqual(p.moments, pairs)
            self.assertEqual(p.repr, "exact")
        explicit = [(2, 0, 0), (0, 0, 0), (1, 1, 0)]
        p = mi.plan(scope, ["x", "y", "z"], basis=explicit)
        self.assertEqual(p.basis, sorted(explicit, key=lambda e: (sum(e), e)))
        with self.assertRaises(ValueError):
            mi.plan(scope, ["x", "y", "z"], basis=[(1, 0, 0), (1, 0, 0)])            # a duplicate monomial
        with self.assertRaises(UnsupportedError):
            mi.plan(scope, ["x", "y", "z"], degree=6)                               # C(9, 3) = 84 > 63

    def test_disc002_gram_matches_python_cpu_cuda_bank_and_lin(self):
        n = 260
        v = planted(self.rng, n)
        w = [int(t) for t in self.rng.integers(0, 4, n)]
        v["w"] = w
        nulls = set(int(i) for i in self.rng.choice(n, 12, replace=False))
        v["u"] = [None if i in nulls else t for i, t in enumerate(v["u"])]
        db = make(v)
        names = PLANTED + ["w"]
        scope = mbk.Scope.from_table(db["t"], names)
        for variables, d, weight in ((["x", "y", "u"], 2, None), (["x", "y", "z"], 2, "w"), (["a", "x"], 3, None)):
            p = mi.plan(scope, variables, degree=d, weight=weight)
            basis = basis_oracle(len(variables), d)
            G, domain = gram_oracle(v, variables, basis, weight)
            g = mi.gram(scope, p)
            self.assertEqual(g.matrix(), G, (variables, d, weight))
            info = g.info
            self.assertEqual(info["domain"], domain)
            self.assertEqual(info["sign"], "nonnegative")
            if weight:
                rows = [r for r in range(n) if all(v[c][r] is not None for c in variables)]
                self.assertEqual((info["positive"], info["zero"]), (sum(w[r] > 0 for r in rows), sum(w[r] == 0 for r in rows)))
                self.assertEqual((info["wmin"], info["wmax"]), (min(w[r] for r in rows), max(w[r] for r in rows)))
            gc = mi.gram(scope, p, backend="cuda")
            self.assertTrue(g == gc, (variables, d, weight))
            self.assertIn(gc.backend, ("cuda", "cuda-emulated"))
        # FND-004 bank assembly (joint mode over exactly the variables) and LIN-001 moments give the same Gram
        p = mi.plan(scope, ["x", "y", "u"], degree=2)
        bank = mbk.build(scope, [tuple(m[0] if c == "x" else m[1] if c == "y" else m[2] if c == "u" else 0
                                       for c in names) for m in p.moments], mode="joint")
        self.assertTrue(mi.gram(scope, p, source="bank", bank=bank) == mi.gram(scope, p))
        p1 = mi.plan(scope, ["x", "y", "u"], degree=1)
        mom = lin.moments(scope, ["x", "y", "u"], policy="listwise")
        self.assertTrue(mi.gram(scope, p1, source="lin", lin=mom) == mi.gram(scope, p1))
        sql = mbk.build(scope, ["1", "x", "y"], mode="sql")                          # another mode: refused
        with self.assertRaises(DomainError):
            mi.gram(scope, p, source="bank", bank=sql)
        # merges of disjoint selections add; a difference is signed
        half = np.arange(n) < n // 2
        sa = mbk.Scope.from_table(db["t"], names, rows=half)
        sb = mbk.Scope.from_table(db["t"], names, rows=~half)
        ga, gb = mi.gram(sa, mi.plan(sa, ["x", "y"], degree=2)), mi.gram(sb, mi.plan(sb, ["x", "y"], degree=2))
        full = mi.gram(scope, mi.plan(scope, ["x", "y"], degree=2))
        self.assertEqual((ga + gb).matrix(), full.matrix())
        self.assertEqual((full - gb).matrix(), ga.matrix())
        self.assertEqual((full - gb).sign, "signed")
        self.assertEqual((ga + gb).sign, "nonnegative")

    def test_disc002_modular_gram_matches_python_residues_and_the_exact_kernel(self):
        n = 200
        v = planted(self.rng, n)
        db = make(v)
        scope = mbk.Scope.from_table(db["t"], PLANTED)
        variables = ["s", "x", "y"]
        p = mi.plan(scope, variables, degree=2, width="modular", primes=4)
        self.assertEqual(p.repr, "modular")
        primes, _ = lin.primes(4)
        self.assertEqual(p.primes, primes)
        G, _ = gram_oracle(v, variables, basis_oracle(3, 2))
        g = mi.gram(scope, p)
        for q, prime in enumerate(primes):
            self.assertEqual(g.matrix(q), [[x % prime for x in row] for row in G])
        self.assertTrue(g == mi.gram(scope, p, backend="cuda"))
        exact = mi.gram(scope, mi.plan(scope, variables, degree=2)).kernel()
        modular = g.kernel()
        self.assertEqual((modular.rank, modular.free, modular.vectors), (exact.rank, exact.free, exact.vectors))
        self.assertFalse(modular.exact_rank)
        self.assertTrue(exact.exact_rank)
        self.assertEqual(modular.agreeing, 4)
        with self.assertRaises(OverflowError):                                      # 2^62-sized values: unproven
            big = mbk.Scope.from_arrays({"x": np.array([2 ** 61, 3, -2 ** 62]), "y": np.array([1, 2, 3])})
            mi.plan(big, ["x", "y"], degree=2, width="exact")
        self.assertEqual(mi.plan(big, ["x", "y"], degree=2).repr, "modular")

    def test_disc002_nullspace_matches_fraction_elimination_on_cpu_and_cuda(self):
        n = 150
        v = planted(self.rng, n)
        db = make(v)
        scope = mbk.Scope.from_table(db["t"], PLANTED)
        for variables, d in ((["s", "x", "y"], 2), (["a", "z", "x", "y"], 2), (["x", "y"], 3)):
            p = mi.plan(scope, variables, degree=d)
            G, _ = gram_oracle(v, variables, basis_oracle(len(variables), d))
            rank, piv, free, vecs = rref_nullspace(G)
            g = mi.gram(scope, p)
            k = g.kernel()
            self.assertEqual((k.rank, k.pivots, k.free, k.vectors), (rank, piv, free, vecs), variables)
            self.assertTrue(k.exact_rank)
            kc = g.kernel(backend="cuda")
            self.assertEqual((kc.rank, kc.free, kc.vectors), (k.rank, k.free, k.vectors))
            self.assertIn(kc.backend, ("cuda", "cuda-emulated"))

    # ---------------------------------------------------------- acceptance: recovery and one violating row
    def test_disc002_recovers_exact_small_polynomial_dependencies(self):
        n = 400
        v = planted(self.rng, n)
        db = make(v)
        t = db["t"]
        scope = mbk.Scope.from_table(t, PLANTED)
        found = mi.discover(scope, PLANTED, degree=2)
        G, _ = gram_oracle(v, PLANTED, basis_oracle(6, 2))
        rank, piv, free, vecs = rref_nullspace(G)
        self.assertEqual((found.kernel.rank, found.kernel.nullity), (rank, len(free)))
        disc = found.discoveries()
        self.assertEqual(sorted((dict(i["terms"]) for i in disc), key=str), sorted(EXPECTED, key=str))
        cuda = mi.identities(scope, found.gram, found.kernel).certify(backend="cuda")
        for i, (it, ic) in enumerate(zip(found.identities, cuda)):
            self.assertEqual(it["status"], "verified")
            self.assertEqual(it["terms"], ic["terms"])
            self.assertEqual(strip(it["evidence"]), strip(ic["evidence"]))
            self.assertEqual(primitive(vecs[i]), [dict(it["terms"]).get(e, 0) for e in basis_oracle(6, 2)])
            want = evidence_oracle(v, PLANTED, it["terms"])
            self.assertEqual({k: it["evidence"][k] for k in want}, want)
            self.assertEqual(it["sos"], "certified")
            cert = it["certificate"]
            self.assertEqual((cert.kind, cert.info["exact"], cert.info["nulls"]), ("snapshot", True, "skip"))
            self.assertTrue(cert.fresh(scope).fresh)
            self.assertEqual(cert.authorize_exact(scope).kind, "snapshot")
            self.assertEqual(ic["certificate"].info["verified_by"], "cuda library")
            # FND-007's own verifier agrees with the evidence
            fnd = mc.verify(scope, [found.identities.claim(i)])[0]
            self.assertEqual({k: fnd[k] for k in ("checked", "excluded", "violations", "witness", "holds")},
                             {k: it["evidence"][k] for k in ("checked", "excluded", "violations", "witness", "holds")})
        labels = [(i["consequence"], i["known"]) for i in found.identities if not i["discovery"]]
        self.assertTrue(labels and all(c for c, _ in labels))          # every other identity is a consequence
        self.assertEqual(found.report["generators"], 3)
        self.assertTrue(found.report["labels_exact"])
        # DuckDB on the planted data: the three relations hold on every row
        frame = pd.DataFrame(v)
        self.assertEqual(sql("SELECT count(*) FROM frame WHERE a <> 3 * x - 7 OR s <> x + y OR z <> x * y", frame=frame)[0], 0)
        t.flip(0, 3)                                                    # a mutation: the certificate is stale
        now = mbk.Scope.from_table(t, PLANTED)
        self.assertIn("revision", disc[0]["certificate"].fresh(now).reasons)
        with self.assertRaises(StaleStateError):
            mi.gram(scope, found.plan)

    def test_disc002_rejects_a_single_violating_row(self):
        n = 400
        v = planted(self.rng, n)
        bad = 271
        v["s"][bad] += 1                                                # a one-row near miss of a conservation law
        db = make(v)
        t = db["t"]
        full = mbk.Scope.from_table(t, PLANTED)
        clean = np.ones(n, bool)
        clean[bad] = False
        sub = mbk.Scope.from_table(t, PLANTED, rows=clean)              # the Gram of the rows without it
        g = mi.gram(sub, mi.plan(sub, ["s", "x", "y"], degree=1))
        k = g.kernel()
        self.assertEqual(k.nullity, 1)
        for backend in ("cpu", "cuda"):
            ids = mi.identities(full, g, k).certify(backend=backend)
            it = ids[0]
            self.assertEqual(dict(it["terms"]), {(1, 0, 0): 1, (0, 1, 0): -1, (0, 0, 1): -1})
            self.assertEqual(it["status"], "rejected")
            self.assertEqual((it["evidence"]["violations"], it["evidence"]["witness"]), (1, bad))
            self.assertIsNone(it["certificate"])
            self.assertEqual(it["certificate_code"], -6)
        frame = pd.DataFrame(v).reset_index()
        self.assertEqual(sql('SELECT count(*), min("index") FROM frame WHERE s <> x + y', frame=frame), (1, bad))
        # on the full scope the Gram does not propose it: the violating row contributes (c . m)^2 = 1
        gf = mi.gram(full, mi.plan(full, ["s", "x", "y"], degree=1))
        self.assertEqual(gf.kernel().nullity, 0)
        cand = mi.candidates(full, ["s", "x", "y"], [{(1, 0, 0): 1, (0, 1, 0): -1, (0, 0, 1): -1}], gram=gf)
        self.assertEqual(mi.quadratic(gf, cand, 0)["exact"], 1)
        self.assertEqual(cand[0]["sos"], "nonzero")
        self.assertEqual(cand.certify(backend="cuda")[0]["status"], "rejected")
        self.assertEqual(mi.candidates(sub, ["s", "x", "y"], [{(1, 0, 0): 1, (0, 1, 0): -1, (0, 0, 1): -1}])
                         .certify()[0]["status"], "verified")

    # ---------------------------------------------------------- acceptance: signed weights and numerical values
    def test_disc002_signed_weights_cannot_certify_by_a_zero_sum_of_squares(self):
        xs = [-1, 0, 1] * 6
        ws = [1, -2, 1] * 6                                             # sum w x^0 = 0, sum w x = 0: "1" is in the kernel
        db = make({"x": xs, "w": ws})
        scope = mbk.Scope.from_table(db["t"], ["x", "w"])
        p = mi.plan(scope, ["x"], degree=1, weight="w")
        g = mi.gram(scope, p)
        self.assertEqual((g.sign, g.info["negative"]), ("signed", 6))
        self.assertEqual(g.matrix(), [[0, 0], [0, 12]])
        k = g.kernel()
        self.assertEqual(k.vectors, [[Fraction(1), Fraction(0)]])       # c^T G c = 0 for the constant 1
        for backend in ("cpu", "cuda"):
            ids = mi.identities(scope, g, k).certify(backend=backend)
            it = ids[0]
            self.assertEqual((it["text"], it["sos"]), ("1", "signed"))  # refused: not a sum of squares
            self.assertEqual(mi.quadratic(g, mi.candidates(scope, ["x"], [{(0,): 1}], gram=g), 0)["sos"], "signed")
            self.assertEqual(it["status"], "rejected")                  # the exhaustive check: 1 != 0 on every row
            self.assertEqual((it["evidence"]["violations"], it["evidence"]["witness"]), (18, 0))
            self.assertIsNone(it["certificate"])
        # with nonnegative weights the same zero quadratic form is a certificate of the Gram rows
        pos = mbk.Scope.from_table(make({"x": [1, 2, 3, 4], "y": [2, 4, 6, 8], "w": [1, 2, 0, 1]})["t"], ["x", "y", "w"])
        gp = mi.gram(pos, mi.plan(pos, ["x", "y"], degree=1, weight="w"))
        ip = mi.identities(pos, gp, gp.kernel())
        self.assertEqual([(i["text"], i["sos"]) for i in ip], [("2*x - y", "certified")])
        # a difference of Grams (a DELTA-style deletion) is signed: its kernel never certifies by a zero form
        n = 120
        v = planted(self.rng, n)
        db2 = make(v)
        allrows = mbk.Scope.from_table(db2["t"], PLANTED)
        some = mbk.Scope.from_table(db2["t"], PLANTED, rows=np.arange(n) < 40)
        delta = mi.gram(allrows, mi.plan(allrows, ["s", "x", "y"], degree=1)) - mi.gram(some, mi.plan(some, ["s", "x", "y"], degree=1))
        self.assertEqual(delta.sign, "signed")
        di = mi.identities(allrows, delta, delta.kernel())
        self.assertEqual([i["sos"] for i in di], ["signed"])
        self.assertEqual(di.certify()[0]["status"], "verified")         # it still verifies exhaustively on the rows

    def test_disc002_numerical_near_zero_is_only_a_hypothesis(self):
        n = 2000
        a = [int(t) for t in self.rng.integers(10 ** 9, 2 * 10 ** 9, n)]
        b = [int(t) for t in self.rng.integers(10 ** 9, 2 * 10 ** 9, n)]
        s = [p + q for p, q in zip(a, b)]
        bad = 1234
        s[bad] += 1
        db = make({"a": a, "b": b, "s": s})
        scope = mbk.Scope.from_table(db["t"], ["a", "b", "s"])
        g = mi.gram(scope, mi.plan(scope, ["a", "b", "s"], degree=1))
        self.assertEqual(g.repr, "exact")
        poly = {(0, 0, 1): 1, (1, 0, 0): -1, (0, 1, 0): -1}
        cand = mi.candidates(scope, ["a", "b", "s"], [poly], gram=g)
        q = mi.quadratic(g, cand, 0)
        self.assertEqual(q["exact"], sum((z - p - r) ** 2 for p, r, z in zip(a, b, s)))   # exactly 1
        self.assertEqual(q["sos"], "nonzero")
        self.assertLess(q["relative"], 1e-12)                           # numerically "zero"
        h = mi.hypothesis(scope, g, cand, 0, tolerance=1e-12, size=500, method="stride")
        self.assertEqual(h.kind, "hypothesis")
        self.assertEqual(h.sample["size"], 500)
        self.assertEqual(h.decide(scope)["reason"], "statistical")
        with self.assertRaises(mc.NotAuthorizedError):
            h.authorize_exact(scope)
        for backend in ("cpu", "cuda"):
            it = mi.candidates(scope, ["a", "b", "s"], [poly], gram=g).certify(backend=backend)[0]
            self.assertEqual((it["status"], it["evidence"]["witness"]), ("rejected", bad))
        with self.assertRaises(DomainError):                            # not within a tighter tolerance
            mi.hypothesis(scope, g, cand, 0, tolerance=1e-20, size=500, method="stride")
        with self.assertRaises(mc.ClaimViolated):                       # a sample that holds the bad row refutes it
            mi.hypothesis(scope, g, cand, 0, tolerance=1e-12, size=n, method="prefix")
        gm = mi.gram(scope, mi.plan(scope, ["a", "b", "s"], degree=1, width="modular"))
        qm = mi.quadratic(gm, mi.candidates(scope, ["a", "b", "s"], [poly], gram=gm), 0)
        self.assertEqual((qm["exact"], qm["sos"]), (None, "modular"))
        self.assertTrue(math.isnan(qm["relative"]))
        primes = gm.info["primes"]
        self.assertEqual(qm["residues"], [1 % p for p in primes])
        with self.assertRaises(DomainError):                            # no numerical value for residues
            mi.hypothesis(scope, gm, mi.candidates(scope, ["a", "b", "s"], [poly], gram=gm), 0, tolerance=1, size=10)

    # ---------------------------------------------------------- underdetermination and restated definitions
    def test_disc002_identities_on_few_distinct_points_are_underdetermined(self):
        w = [0, 1, 2] * 15 + [1] * 5
        db = make({"w": w, "i": [2 ** t for t in w]})
        scope = mbk.Scope.from_table(db["t"], ["w", "i"])
        for d in (2, 3):
            found = mi.discover(scope, ["w", "i"], degree=d)
            B = len(basis_oracle(2, d))
            pts = len(set(zip(w, [2 ** t for t in w])))
            self.assertEqual(pts, 3)
            self.assertEqual((found.report["points"], found.report["forced"]), (3, B - 3))
            self.assertEqual(found.kernel.nullity, B - 3)
            for it in found.identities:
                sup = [q for q, name in enumerate(["w", "i"]) if name in it["support"]]
                want = len({tuple((w[r], 2 ** w[r])[q] for q in sup) for r in range(len(w))})
                self.assertEqual(it["points"], want)
                self.assertEqual(it["forced"], max(0, it["nterms"] - want))
                self.assertEqual(it["underdetermined"], it["nterms"] > want)
                self.assertEqual(it["status"], "verified")
                self.assertFalse(it["discovery"] and it["underdetermined"])
            self.assertTrue(all(i["underdetermined"] or i["consequence"] for i in found.identities))
        frame = pd.DataFrame({"w": w, "i": [2 ** t for t in w]})
        self.assertEqual(sql("SELECT count(DISTINCT (w, i)) FROM frame", frame=frame)[0], 3)
        self.assertEqual(mi.support(scope, ["w", "i"]), (3, len(w)))
        # a 0/1 variable: x^2 - x has 2 terms on 2 points, not forced by counting
        db2 = make({"b": [0, 1, 1, 0, 1], "y": [3, 4, 5, 6, 7]})
        s2 = mbk.Scope.from_table(db2["t"], ["b", "y"])
        found = mi.discover(s2, ["b"], degree=2)
        self.assertEqual([(i["text"], i["underdetermined"], i["points"]) for i in found.identities], [("b^2 - b", False, 2)])

    def test_disc002_declared_definitions_and_consequences(self):
        n = 200
        v = planted(self.rng, n)
        v["q"] = [p * p for p in v["x"]]
        db = make(v)
        scope = mbk.Scope.from_table(db["t"], PLANTED + ["q"])
        variables = ["q", "x", "y"]
        declared = {(1, 0, 0): 1, (0, 2, 0): -1}                         # the definition q = x^2
        found = mi.discover(scope, variables, degree=2, known=[declared])
        basis = basis_oracle(3, 2)
        multiples = []
        for b in basis:                                                 # u = b / lead(q - x^2), every term in the basis
            u = tuple(x - y for x, y in zip(b, (0, 2, 0)))
            if all(t >= 0 for t in u) and sum(u) + 2 <= 2:
                row = [0] * len(basis)
                for e, c in declared.items():
                    row[basis.index(tuple(p + q for p, q in zip(u, e)))] = c
                multiples.append(row)
        kernel_items = [i for i in found.identities if i["origin"] == "kernel"]
        for it in kernel_items:
            vec = [dict(it["terms"]).get(e, 0) for e in basis]
            self.assertEqual(it["known"], in_span(multiples, vec), it["text"])
        self.assertTrue(any(i["known"] and i["text"] == "x^2 - q" for i in kernel_items))
        declared_items = [i for i in found.identities if i["origin"] == "declared"]
        self.assertEqual([(i["text"], i["status"]) for i in declared_items], [("-x^2 + q", "verified")])  # as declared
        self.assertFalse(any(i["discovery"] and i["known"] for i in found.identities))
        self.assertEqual(found.report["declared_rank"], len(multiples))
        # without the declaration the same identity is a discovery
        plain = mi.discover(scope, variables, degree=2)
        self.assertTrue(any(i["text"] == "x^2 - q" and i["discovery"] for i in plain.identities))
        # consequences: multiples of a generator within the basis are labelled, with exact span membership
        found = mi.discover(scope, ["s", "x", "y", "u"], degree=2)
        gens = [i for i in found.identities if not i["consequence"] and not i["known"]]
        self.assertEqual([i["text"] for i in gens], ["s - x - y"])
        basis = basis_oracle(4, 2)
        rows = []
        for u in basis:
            if sum(u) <= 1:
                row = [0] * len(basis)
                for e, c in {(1, 0, 0, 0): 1, (0, 1, 0, 0): -1, (0, 0, 1, 0): -1}.items():
                    row[basis.index(tuple(p + q for p, q in zip(u, e)))] = c
                rows.append(row)
        for it in found.identities:
            self.assertTrue(in_span(rows, [dict(it["terms"]).get(e, 0) for e in basis]))

    def test_disc002_number_theory_gated_identities_and_three_point_artifacts(self):
        N = 3000
        om, Om, d, sq = sieve(N)
        v = {"n": list(range(1, N + 1)), "om": om[1:], "big": Om[1:], "i": [2 ** t for t in om[1:]], "mu2": sq[1:],
             "d": d[1:]}
        db = make(v)
        scope = mbk.Scope.from_table(db["t"], list(v))
        found = mi.discover(scope, ["om", "big", "i", "mu2", "d"], degree=2)
        texts = {i["text"] for i in found.discoveries()}
        self.assertEqual(texts, {"mu2^2 - mu2", "i*mu2 - mu2*d", "om*mu2 - big*mu2"})
        for it in found.identities:
            self.assertEqual(it["status"], "verified")
        frame = pd.DataFrame(v)
        self.assertEqual(sql("SELECT count(*) FROM frame WHERE mu2 = 1 AND (i <> d OR om <> big)", frame=frame)[0], 0)
        # on n <= 29 omega takes 3 values: the interpolations of I = 2^omega are underdetermined, never discoveries
        small = mbk.Scope.from_table(db["t"], list(v), rows=np.arange(1, N + 1) <= 29)
        art = mi.discover(small, ["om", "i"], degree=2)
        self.assertEqual(art.report["points"], 3)
        self.assertTrue(art.identities and all(i["underdetermined"] for i in art.identities))
        self.assertEqual(art.discoveries(), [])
        self.assertIn("om^2 + om - 2*i + 2", {i["text"] for i in art.identities})   # I = 1 + w/2 + w^2/2 on 3 points
        # restricted to squarefree n the gated identities become plain ones
        sqf = mbk.Scope.from_table(db["t"], list(v), rows=np.array(sq[1:]) == 1)
        plain = mi.discover(sqf, ["om", "big", "i", "d"], degree=1)
        self.assertEqual({i["text"] for i in plain.discoveries()}, {"om - big", "i - d"})

    # ---------------------------------------------------------- NULLs, widths, budgets, freshness
    def test_disc002_null_domains(self):
        n = 120
        v = planted(self.rng, n)
        holes = set(int(i) for i in self.rng.choice(n, 10, replace=False))
        v["s"] = [None if i in holes else t for i, t in enumerate(v["s"])]
        db = make(v)
        scope = mbk.Scope.from_table(db["t"], PLANTED)
        g = mi.gram(scope, mi.plan(scope, ["s", "x", "y"], degree=1))
        self.assertEqual(g.info["domain"], n - len(holes))
        ids = mi.identities(scope, g, g.kernel()).certify(backend="cuda")
        it = ids[0]
        self.assertEqual(it["text"], "s - x - y")
        self.assertEqual((it["evidence"]["checked"], it["evidence"]["excluded"]), (n - len(holes), len(holes)))
        self.assertEqual(it["certificate"].authorize_exact(scope).nulls, "skip")
        self.assertEqual(mi.support(scope, ["s", "x"])[1], n - len(holes))

    def test_disc002_wide_rows_are_decided_exactly(self):
        # x*y*z = 2^128 at one row: zero modulo 2^128, not zero; the bound makes the evaluation 3 words wide
        x = np.array([2 ** 62, 5, 7], dtype=np.int64)
        y = np.array([2 ** 62, 0, 1], dtype=np.int64)
        z = np.array([16, 3, 0], dtype=np.int64)
        scope = mbk.Scope.from_arrays({"x": x, "y": y, "z": z})
        cand = mi.candidates(scope, ["x", "y", "z"], [{(1, 1, 1): 1}])
        self.assertEqual(cand[0]["words"], 3)
        for backend in ("cpu", "cuda"):
            it = mi.candidates(scope, ["x", "y", "z"], [{(1, 1, 1): 1}]).certify(backend=backend)[0]
            self.assertEqual((it["status"], it["evidence"]["violations"], it["evidence"]["witness"]), ("rejected", 1, 0))
        # an identity with terms near 2^186 that holds: verified in 3 words; FND-007's 128-bit check cannot hold it
        a = np.array([2 ** 61, 3 * 2 ** 59, 12345, -2 ** 60], dtype=np.int64)
        b = np.array([2 ** 61 - 1, 2 ** 59, -45, 2 ** 61], dtype=np.int64)
        s = a + b
        sc = mbk.Scope.from_arrays({"s": s, "a": a, "b": b})
        cube = {(3, 0, 0): 1, (0, 3, 0): -1, (0, 2, 1): -3, (0, 1, 2): -3, (0, 0, 3): -1}
        for backend in ("cpu", "cuda"):
            it = mi.candidates(sc, ["s", "a", "b"], [cube]).certify(backend=backend)[0]
            self.assertEqual(it["words"], 3)
            self.assertEqual(it["status"], "verified")
            self.assertEqual(it["evidence"]["checked"], 4)
            self.assertIsNone(it["certificate"])
            self.assertEqual(it["certificate_code"], -3)                # TR_MATH_OVERFLOW: FND-007's carrier
        self.assertEqual(evidence_oracle({"s": s.tolist(), "a": a.tolist(), "b": b.tolist()}, ["s", "a", "b"],
                                         list(cube.items()))["violations"], 0)

    def test_disc002_budgets_at_the_limit_and_one_past(self):
        v = planted(self.rng, 80)
        db = make(v)
        scope = mbk.Scope.from_table(db["t"], PLANTED)
        variables = ["s", "x", "y", "u"]
        B = len(basis_oracle(4, 2))
        mi.plan(scope, variables, degree=2, budget=Budget(support=B))
        with self.assertRaises(AlgebraBudgetError) as e:
            mi.plan(scope, variables, degree=2, budget=Budget(support=B - 1))
        self.assertEqual((e.exception.resource, e.exception.need, e.exception.limit), ("support", B, B - 1))
        two = basis_oracle(2, 10)[:63]                                  # the representation limit: 63 monomials
        self.assertEqual(len(mi.plan(scope, ["x", "y"], basis=two).basis), 63)
        with self.assertRaises(UnsupportedError):
            mi.plan(scope, ["x", "y"], basis=basis_oracle(2, 10)[:64])
        p = mi.plan(scope, variables, degree=2, width="modular", primes=4)
        with self.assertRaises(AlgebraBudgetError) as e:
            mi.plan(scope, variables, degree=2, width="modular", primes=4, budget=Budget(iterations=3))
        self.assertEqual(e.exception.resource, "iterations")
        mi.plan(scope, variables, degree=2, width="modular", primes=4, budget=Budget(iterations=4))
        need = p.need("cpu")["bytes"]
        mi.gram(scope, p, budget=Budget(bytes=need))
        with self.assertRaises(AlgebraBudgetError):
            mi.gram(scope, p, budget=Budget(bytes=need - 1))
        dev = p.need("cuda")["bytes"]
        mi.gram(scope, p, backend="cuda", budget=Budget(bytes=dev))
        with self.assertRaises(AlgebraBudgetError):
            mi.gram(scope, p, backend="cuda", budget=Budget(bytes=dev - 1))
        rows = mi.support(scope, ["x", "y"])[1]
        sneed = 33 * rows + 65536 * 8
        mi.support(scope, ["x", "y"], budget=Budget(bytes=sneed))
        with self.assertRaises(AlgebraBudgetError):
            mi.support(scope, ["x", "y"], budget=Budget(bytes=sneed - 1))
        g = mi.gram(scope, mi.plan(scope, variables, degree=2))
        k = g.kernel()
        mi.identities(scope, g, k, budget=Budget(output=k.nullity))
        with self.assertRaises(AlgebraBudgetError) as e:
            mi.identities(scope, g, k, budget=Budget(output=k.nullity - 1))
        self.assertEqual(e.exception.resource, "output")
        nine = mbk.Scope.from_arrays({f"c{i}": np.arange(4) for i in range(9)})
        with self.assertRaises(UnsupportedError):                       # an FND-007 claim reads at most 8 columns
            mi.plan(nine, [f"c{i}" for i in range(9)], degree=1)

    def test_disc002_support_counts_match_python_and_duckdb(self):
        n = 500
        a = [int(t) for t in self.rng.integers(0, 6, n)]
        b = [int(t) for t in self.rng.integers(0, 4, n)]
        w = [int(t) for t in self.rng.integers(-1, 2, n)]
        db = make({"a": a, "b": b, "w": w})
        scope = mbk.Scope.from_table(db["t"], ["a", "b", "w"])
        frame = pd.DataFrame({"a": a, "b": b, "w": w})
        self.assertEqual(mi.support(scope, ["a", "b"]), (len(set(zip(a, b))), n))
        self.assertEqual(mi.support(scope, ["a", "b"])[0], sql("SELECT count(DISTINCT (a, b)) FROM frame", frame=frame)[0])
        totals = {}
        for p, q, r in zip(a, b, w):
            totals[(p, q)] = totals.get((p, q), 0) + r
        self.assertEqual(mi.support(scope, ["a", "b"], weight="w")[0], sum(1 for t in totals.values() if t != 0))
        self.assertEqual(mi.support(scope, ["a", "b"], weight="w")[0],
                         sql("SELECT count(*) FROM (SELECT a, b, sum(w) s FROM frame GROUP BY a, b) WHERE s <> 0", frame=frame)[0])


if __name__ == "__main__":
    unittest.main()
