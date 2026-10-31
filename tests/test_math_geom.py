# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""GEOM-001: certified sparse ideal reduction.

Oracles (independent of the implementation): polynomials as dictionaries of Fractions, the multivariate division
algorithm and Buchberger's algorithm written here (the reduced Groebner basis and the normal form are unique, so any
correct implementation must agree), hand-built ideals with known reductions (the twisted cubic, the circle, a rational
relation), the certificate identity P = sum q_i r_i expanded with Python integers, every polynomial evaluated row by
row with Python integers (agreement, bit lengths, dependency declarations), DuckDB counts on the planted data, and the
chain rule checked by substituting the declared dependencies. FND-007's own derivation (tr_cert_derive) is the
certificate checker; the batched checks are compared with it and with the expansion here. Every result with a CUDA path
is computed on it too (emulated here, threads in a scrambled order) and compared for equality. Seeds are fixed.
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
from torbles import math_geom as mg
from torbles import math_ident as mi
from torbles.math_budgets import Budget
from torbles.math_domains import DomainError
from torbles.math_errors import StaleStateError, UnsupportedError
from torbles.mathematics import AlgebraBudgetError

SEED = 1001


def make(values, name="t"):
    frame = pd.DataFrame({c: pd.Series(v, dtype=object) for c, v in values.items()})
    ddl = f"CREATE TABLE {name} (" + ", ".join(f'"{c}" BIGINT' for c in values) + ")"
    return torbles.from_ddl(ddl, {name: frame})


def sql(query, **frames):
    con = duckdb.connect()
    for name, frame in frames.items():
        con.register(name, frame)
    return con.execute(query).fetchone()


# ------------------------------------------------------------------ the polynomial oracle (Fractions)
class Order:
    """a monomial order over positions 0..n-1: sig lists the positions by significance (most significant first)"""

    def __init__(self, kind, sig):
        self.kind, self.sig = kind, list(sig)

    def key(self, e):
        if self.kind == "lex":
            return tuple(e[p] for p in self.sig)
        if self.kind == "deglex":
            return (sum(e),) + tuple(e[p] for p in self.sig)
        return (sum(e),) + tuple(-e[p] for p in reversed(self.sig))


def clean(p):
    return {e: Fraction(c) for e, c in p.items() if c}


def add(a, b, c=1, m=None):
    """a + c * x^m * b"""
    out = dict(a)
    for e, v in b.items():
        f = tuple(x + y for x, y in zip(e, m)) if m is not None else e
        out[f] = out.get(f, 0) + c * v
    return clean(out)


def mul(a, b):
    out = {}
    for e, u in a.items():
        for f, v in b.items():
            g = tuple(x + y for x, y in zip(e, f))
            out[g] = out.get(g, 0) + u * v
    return clean(out)


def lead(p, order):
    e = max(p, key=order.key)
    return e, p[e]


def divides(a, b):
    return all(x <= y for x, y in zip(a, b))


def divide(f, G, order):
    """the division algorithm: the leading term reduced by the first divisor in list order"""
    p, r, q = dict(f), {}, [{} for _ in G]
    while p:
        e, c = lead(p, order)
        for k, g in enumerate(G):
            eg, cg = lead(g, order)
            if divides(eg, e):
                m = tuple(x - y for x, y in zip(e, eg))
                q[k] = add(q[k], {m: c / cg})
                p = add(p, g, -c / cg, m)
                break
        else:
            r[e] = c
            del p[e]
    return q, clean(r)


def monic(p, order):
    _, c = lead(p, order)
    return {e: v / c for e, v in p.items()}


def spoly(f, g, order):
    (ef, cf), (eg, cg) = lead(f, order), lead(g, order)
    L = tuple(max(a, b) for a, b in zip(ef, eg))
    return add(add({}, f, 1 / cf, tuple(l - a for l, a in zip(L, ef))), g, -1 / cg, tuple(l - b for l, b in zip(L, eg)))


def groebner(F, order):
    """the reduced Groebner basis (Buchberger, every pair; then minimal and reduced), increasing leading monomial"""
    G = [monic(f, order) for f in F if f]
    pairs = [(i, j) for i in range(len(G)) for j in range(i)]
    while pairs:
        i, j = pairs.pop()
        _, r = divide(spoly(G[i], G[j], order), G, order)
        if r:
            G.append(monic(r, order))
            pairs += [(len(G) - 1, k) for k in range(len(G) - 1)]
    lm = [lead(g, order)[0] for g in G]
    keep = [g for i, g in enumerate(G) if not any(k != i and divides(lm[k], lm[i]) and (lm[k] != lm[i] or k < i)
                                                  for k in range(len(G)))]
    out = []
    for i, g in enumerate(keep):
        e, _ = lead(g, order)
        _, r = divide({m: v for m, v in g.items() if m != e}, keep[:i] + keep[i + 1:], order)
        out.append(clean({e: 1, **r}))
    return sorted(out, key=lambda p: order.key(lead(p, order)[0]))


def evaluate(p, point):
    total = Fraction(0)
    for e, c in p.items():
        v = c
        for x, k in zip(point, e):
            v *= x ** k
        total += v
    return total


def denominator(*polys):
    d = 1
    for p in polys:
        for c in p.values():
            d = d * Fraction(c).denominator // math.gcd(d, Fraction(c).denominator)
    return d


def partial(p, q):
    out = {}
    for e, c in p.items():
        if e[q]:
            f = list(e)
            f[q] -= 1
            out[tuple(f)] = out.get(tuple(f), 0) + c * e[q]
    return clean(out)


def substitute(p, q, phi):
    """p with position q replaced by phi (same positions)"""
    out = {}
    for e, c in p.items():
        k = e[q]
        rest = list(e)
        rest[q] = 0
        term = {tuple(rest): Fraction(c)}
        for _ in range(k):
            term = mul(term, phi)
        out = add(out, term)
    return out


def bits(v):
    return abs(int(v)).bit_length()


def cert(scope, terms, cols, nulls="skip"):
    """an FND-007 snapshot certificate of the polynomial relation terms over cols"""
    return mc.snapshot(scope, mc.polynomial(terms, cols, nulls=nulls))


def over(p, names):
    """a native Polynomial as an oracle dictionary over names"""
    return clean(p.aligned(names))


def native_order(I):
    """the ideal's order as an oracle Order over its variables in increasing scope column order"""
    info = I.info
    var = sorted(info["variables"], key=lambda v: I.names.index(v))
    return var, Order(info["order"], [var.index(v) for v in info["variables"]])


# ------------------------------------------------------------------ data
def cubic_table(n=41):
    x = list(range(-(n // 2), n - n // 2))
    return make({"x": x, "y": [v * v for v in x], "z": [v ** 3 for v in x]})


class Geom001(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(SEED)

    # ---------------------------------------------------------- reductions against the oracle
    def test_geom001_known_reductions_match_the_fraction_oracle(self):
        db = cubic_table()
        scope = mbk.Scope.from_table(db["t"], ["x", "y", "z"])
        r1 = cert(scope, {(1, 0): 1, (0, 2): -1}, ["y", "x"])                       # y - x^2
        r2 = cert(scope, {(1, 0): 1, (0, 3): -1}, ["z", "x"])                       # z - x^3
        # the twisted cubic, lex z > y > x: {y - x^2, z - x^3} is already a Groebner basis; y^2 + z -> x^4 + x^3
        I = mg.ideal([r1, r2], order="lex", variables=["z", "y", "x"])
        self.assertEqual([b.text for b in I.basis], ["y - x^2", "z - x^3"])
        r = I.reduce(mg.poly({(0, 2, 0): 1, (0, 0, 1): 1}, ["x", "y", "z"]))
        self.assertEqual(r.g, mg.poly({(4,): 1, (3,): 1}, ["x"]))
        self.assertEqual(r.g.text, "x^4 + x^3")
        # deglex x > y > z: the classical four-element basis
        J = mg.ideal([r1, r2], order="deglex", variables=["x", "y", "z"])
        self.assertEqual(sorted(b.text for b in J.basis), sorted(["x*z - y^2", "x*y - z", "x^2 - y", "y^3 - z^2"]))
        self.assertEqual(J.reduce(mg.poly({(5, 0, 0): 1}, ["x", "y", "z"])).g, mg.poly({(1, 1): 1}, ["y", "z"]))  # x^5 = y z
        # the circle: x^3 + x y^2 -> x modulo x^2 + y^2 - 1
        pts = [(1, 0), (-1, 0), (0, 1), (0, -1)] * 3
        dc = make({"x": [p[0] for p in pts], "y": [p[1] for p in pts]}, "c")
        sc = mbk.Scope.from_table(dc["c"], ["x", "y"])
        circle = cert(sc, {(2, 0): 1, (0, 2): 1, (0, 0): -1}, ["x", "y"])
        r = mg.ideal([circle], order="deglex", variables=["x", "y"]).reduce(mg.poly({(3, 0): 1, (1, 2): 1}, ["x", "y"]))
        self.assertEqual(r.g, mg.poly({(1,): 1}, ["x"]))
        self.assertEqual([h.text for h in r.cofactors], ["x"])
        # a rational relation 2y = x: y^2 -> x^2 / 4, the certificate's denominator D = 4 and P = 4 y^2 - x^2
        xs = [2 * k for k in range(-6, 7)]
        dq = make({"x": xs, "y": [k // 2 for k in xs]}, "q")
        sq = mbk.Scope.from_table(dq["q"], ["x", "y"])
        half = cert(sq, {(0, 1): 2, (1, 0): -1}, ["x", "y"])                       # 2y - x
        r = mg.ideal([half], order="lex", variables=["y", "x"]).reduce(mg.poly({(0, 2): 1}, ["x", "y"]))
        self.assertEqual(r.g, mg.poly({(2,): Fraction(1, 4)}, ["x"]))
        self.assertEqual(r.info["denominator"], 4)
        P, cof = r.claims()
        self.assertEqual(dict(zip(map(tuple, P.params["exponents"].tolist()), P.params["coefficients"].tolist())),
                         {(0, 2): 4, (2, 0): -1})
        self.assertEqual(dict(zip(map(tuple, cof[0].params["exponents"].tolist()), cof[0].params["coefficients"].tolist())),
                         {(0, 1): 2, (1, 0): 1})                                    # 4h = 2y + x
        self.assertEqual(r.certify().authorize_exact(sq).kind, "snapshot")

    def test_geom001_groebner_bases_and_normal_forms_match_buchberger_in_fractions(self):
        n = 60
        x = [int(v) for v in self.rng.integers(-9, 10, n)]
        y = [int(v) for v in self.rng.integers(-9, 10, n)]
        u = [a * b + 1 for a, b in zip(x, y)]                                        # u = x y + 1
        w = [a * a - b for a, b in zip(x, y)]                                        # w = x^2 - y
        s = [a + 2 * b for a, b in zip(x, y)]                                        # s = x + 2y
        db = make({"x": x, "y": y, "u": u, "w": w, "s": s})
        names = ["x", "y", "u", "w", "s"]
        scope = mbk.Scope.from_table(db["t"], names)
        rels = [((1, 0, 0, 0, 0), {(0, 0, 1, 0, 0): 1, (1, 1, 0, 0, 0): -1, (0, 0, 0, 0, 0): -1}),
                ((0, 0, 0, 1, 0), {(0, 0, 0, 1, 0): 1, (2, 0, 0, 0, 0): -1, (0, 1, 0, 0, 0): 1}),
                ((0, 0, 0, 0, 1), {(0, 0, 0, 0, 1): 1, (1, 0, 0, 0, 0): -1, (0, 1, 0, 0, 0): -2})]
        certs = [cert(scope, p, names) for _, p in rels]
        fs = [{(2, 2, 0, 0, 0): 1, (0, 0, 2, 0, 0): -3, (1, 0, 0, 1, 0): Fraction(1, 2)},
              {(0, 0, 1, 1, 0): 1, (0, 0, 0, 0, 2): 2, (3, 0, 0, 0, 0): -1},
              {(1, 1, 1, 0, 0): 1, (0, 0, 0, 0, 0): 5}]
        cases = [("deglex", ["u", "w", "s", "x", "y"]), ("degrevlex", ["x", "y", "u", "w", "s"]),
                 ("lex", ["u", "w", "s", "x", "y"]), ("deglex", ["x", "y", "s", "w", "u"]), ("lex", ["y", "s", "u", "w", "x"])]
        for kind, seq in cases:
            for complete in (True, False):
                I = mg.ideal(certs, order=kind, variables=seq, complete=complete)
                var, order = native_order(I)
                self.assertEqual(var, names)
                rel = [clean(p) for _, p in rels]
                want = groebner(rel, order) if complete else [monic(p, order) for p in rel]
                got = [over(b, names) for b in I.basis]
                self.assertEqual(got, want, (kind, seq, complete))
                # every basis element is the combination of the relations its cofactors say
                for j, b in enumerate(I.basis):
                    combo = {}
                    for h, rr in zip(I.cofactors(j), rel):
                        combo = add(combo, mul(over(h, names), rr))
                    self.assertEqual(combo, over(b, names))
                for f in fs:
                    red = I.reduce(mg.poly(f, names))
                    _, g = divide(clean(f), want, order)
                    self.assertEqual(over(red.g, names), g, (kind, seq, complete, f))
                    # f - g = sum h_i r_i exactly, and the certificate's integer parts agree
                    combo = {}
                    hs = [over(h, names) for h in red.cofactors]
                    for h, rr in zip(hs, rel):
                        combo = add(combo, mul(h, rr))
                    self.assertEqual(add(clean(f), g, -1), combo)
                    D = denominator(clean(f), g, *hs)
                    self.assertEqual(red.info["denominator"], D)
                    if complete:
                        cert_ = red.certify()
                        self.assertEqual(cert_.kind, "snapshot")
                        self.assertEqual(len(cert_.premises), 3)
                        self.assertTrue(cert_.authorize_exact(scope))

    # ---------------------------------------------------------- acceptance: wrong coefficients, domains, dependencies
    def test_geom001_certificate_verification_catches_wrong_coefficients(self):
        db = cubic_table()
        scope = mbk.Scope.from_table(db["t"], ["x", "y", "z"])
        r1 = cert(scope, {(1, 0): 1, (0, 2): -1}, ["y", "x"])
        r2 = cert(scope, {(1, 0): 1, (0, 3): -1}, ["z", "x"])
        I = mg.ideal([r1, r2], order="deglex", variables=["x", "y", "z"])
        red = I.reduce(mg.poly({(4, 1, 0): 3, (0, 0, 2): -1, (2, 0, 1): 2}, ["x", "y", "z"]))
        P, cof = red.claims()
        self.assertEqual(mc.derive(P, [r1, r2], cof, scope).kind, "snapshot")       # FND-007 accepts the truth
        good = mg.check([red])[0]
        self.assertTrue(good["holds"] and good["decided"])
        self.assertEqual(mg.check([red], backend="cuda")[0]["holds"], True)
        names = ["x", "y", "z"]
        rel = [{(0, 1, 0): 1, (2, 0, 0): -1}, {(0, 0, 1): 1, (3, 0, 0): -1}]

        def poly_of(claim):
            cols = list(claim.columns)
            return clean({tuple(dict(zip(cols, e)).get(v, 0) for v in names): int(c)
                          for e, c in zip(claim.params["exponents"].tolist(), claim.params["coefficients"].tolist())})

        tampered = []
        for which, t in ((0, 0), (1, 0), (2, -1)):                                  # P's first term, h_1's, h_2's last
            Pt, ct = red.claims()
            target = Pt if which == 0 else ct[which - 1]
            target.params["coefficients"][t] += 1
            with self.assertRaises(DomainError):
                mc.derive(Pt, [r1, r2], ct, scope)
            tampered.append((Pt, [r1, r2], ct))
            diff = add(poly_of(Pt), add(mul(poly_of(ct[0]), rel[0]), mul(poly_of(ct[1]), rel[1])), -1)
            self.assertTrue(diff)
        res_cpu = mg.check(tampered)
        res_gpu = mg.check(tampered, backend="cuda")
        strip = lambda rs: [{k: v for k, v in r.items() if k not in ("backend", "device")} for r in rs]
        self.assertEqual(strip(res_cpu), strip(res_gpu))
        self.assertEqual(res_gpu[0]["backend"] in ("cuda", "cuda-emulated"), True)
        for (Pt, _, ct), res in zip(tampered, res_cpu):
            diff = add(poly_of(Pt), add(mul(poly_of(ct[0]), rel[0]), mul(poly_of(ct[1]), rel[1])), -1)
            self.assertFalse(res["holds"])
            self.assertEqual(res["nonzero"], len(diff))
            ranks = {mg.unrank(k, 3): k for k in range(res["slots"])}
            first = min(diff, key=lambda e: ranks[e])
            self.assertEqual(res["first_rank"], ranks[first])
            self.assertEqual(res["first"], {v: k for v, k in zip(names, first) if k})
        # a wrong remainder is caught the same way: P' = D (f - g') with g' off by one in its leading coefficient
        g = red.g
        e0, c0 = g.ordered[0]
        wrong = mg.poly({**g.terms, e0: c0 + 1}, g.columns)
        f = red.f
        Pw = mc.polynomial({e: int(c) for e, c in add(over(f, names), over(wrong, names), -1).items()}, names,
                           nulls="skip")
        with self.assertRaises(DomainError):
            mc.derive(Pw, [r1, r2], cof, scope)
        self.assertFalse(mg.check([(Pw, [r1, r2], cof)], backend="cuda")[0]["holds"])

    def test_geom001_certificate_verification_catches_wrong_domains(self):
        n = 120
        x = [int(v) for v in self.rng.integers(-50, 50, n)]
        y = [a * a for a in x]
        broken = list(y)
        for r in (7, 64, 99):
            broken[r] += 1                                                          # y = x^2 fails on 3 rows
        db = make({"x": x, "y": broken})
        t = db["t"]
        good = np.ones(n, bool)
        good[[7, 64, 99]] = False
        sub = mbk.Scope.from_table(t, ["x", "y"], rows=good)                        # the relation's own domain
        full = mbk.Scope.from_table(t, ["x", "y"])
        with self.assertRaises(mc.ClaimViolated):
            cert(full, {(0, 1): 1, (2, 0): -1}, ["x", "y"])                         # not certifiable on every row
        rel = cert(sub, {(0, 1): 1, (2, 0): -1}, ["x", "y"])
        I = mg.ideal([rel], order="lex", variables=["y", "x"])
        f = mg.poly({(1, 2): 1, (1, 0): -3}, ["x", "y"])                            # x y^2 - 3x
        red = I.reduce(f)
        self.assertEqual(red.g, mg.poly({(5,): 1, (1,): -3}, ["x"]))                # x^5 - 3x
        c = red.certify()
        self.assertEqual(c.authorize_exact(sub).kind, "snapshot")
        with self.assertRaises((mc.NotAuthorizedError, StaleStateError)):          # the full table: not its domain
            c.authorize_exact(full)
        self.assertFalse(c.decide(full)["authorized"])
        # the two representations agree on the certified rows and disagree exactly where the relation fails
        for backend in ("cpu", "cuda"):
            on = mg.agree(sub, [(f, red.g)], backend=backend)[0]
            self.assertEqual((on["checked"], on["disagreements"], on["witness"]), (n - 3, 0, None))
            off = mg.agree(full, [(f, red.g)], backend=backend)[0]
            fx = [a * b * b - 3 * a for a, b in zip(x, broken)]
            gx = [a ** 5 - 3 * a for a in x]
            wrong = [r for r in range(n) if fx[r] != gx[r]]
            self.assertEqual((off["disagreements"], off["witness"]), (len(wrong), min(wrong)))
            self.assertEqual(off["bits_f"], max(bits(v) for v in fx))
            self.assertEqual(off["bits_g"], max(bits(v) for v in gx))
        frame = pd.DataFrame({"x": x, "y": broken})
        self.assertEqual(sql("SELECT count(*) FROM frame WHERE y <> x * x", frame=frame)[0], 3)
        # a hypothesis relation never authorizes; the derived certificate is a hypothesis too
        hyp = mc.hypothesis(sub, mc.polynomial({(0, 1): 1, (2, 0): -1}, ["x", "y"], nulls="skip"), 30, seed=3)
        hc = mg.ideal([hyp], order="lex", variables=["y", "x"]).reduce(f).certify()
        self.assertEqual(hc.kind, "hypothesis")
        self.assertEqual(hc.decide(sub)["reason"], "statistical")
        # a mutation of the table: the derived certificate is stale through its premise
        t.flip(0, 5)
        now = mbk.Scope.from_table(t, ["x", "y"], rows=good)
        with self.assertRaises((StaleStateError, mc.NotAuthorizedError)):
            c.authorize_exact(now)
        # one NULL policy per ideal; laws and schemas hold nothing to reduce by
        dz = make({"x": x, "y": y}, "u")
        su = mbk.Scope.from_table(dz["u"], ["x", "y"])
        forbid = cert(su, {(0, 1): 1, (2, 0): -1}, ["x", "y"], nulls="forbid")
        skip = cert(su, {(0, 2): 1, (4, 0): -1}, ["x", "y"])
        with self.assertRaises(DomainError):
            mg.ideal([forbid, skip])
        law = mc.law(mc.polynomial([(1, {"x": 2}), (-1, {"x": 2})], ["x", "y"]))
        with self.assertRaises(DomainError):
            mg.ideal([law, skip], names=["x", "y"])
        # relations of two tables, or of one table numbered through two column layouts, form no ideal (natively
        # checked on the FND-007 records: a certificate read back without its names is caught too)
        do = make({"x": x, "y": y}, "o")
        ro = cert(mbk.Scope.from_table(do["o"], ["x", "y"]), {(0, 1): 1, (2, 0): -1}, ["x", "y"])
        with self.assertRaises(DomainError):
            mg.ideal([skip, ro])
        swapped = cert(mbk.Scope.from_table(dz["u"], ["y", "x"]), {(1, 0): 1, (0, 2): -1}, ["y", "x"])
        with self.assertRaises(DomainError):
            mg.ideal([skip, swapped], names=["x", "y"])                          # the names differ
        with self.assertRaises(DomainError):
            mg.ideal([skip, mc.Certificate.from_bytes(swapped.to_bytes())], names=["x", "y"])   # the record differs
        same = cert(mbk.Scope.from_table(dz["u"], ["x", "y"]), {(0, 1): 1, (2, 0): -1}, ["x", "y"])
        self.assertEqual(mg.ideal([skip, mc.Certificate.from_bytes(same.to_bytes())], names=["x", "y"]).info["relations"], 2)

    def test_geom001_certificate_verification_catches_wrong_dependencies(self):
        db = cubic_table()
        scope = mbk.Scope.from_table(db["t"], ["x", "y", "z"])
        r1 = cert(scope, {(1, 0): 1, (0, 2): -1}, ["y", "x"])
        r2 = cert(scope, {(1, 0): 1, (0, 3): -1}, ["z", "x"])
        r3 = cert(scope, {(1, 0, 0): 1, (0, 1, 1): -1}, ["z", "x", "y"])          # z - x y (true: x^3 = x x^2)
        I = mg.ideal([r1, r2], order="deglex", variables=["x", "y", "z"])
        red = I.reduce(mg.poly({(1, 1, 0): 1, (0, 0, 1): 2, (3, 0, 0): -3}, ["x", "y", "z"]))   # x r1 + 2 r2
        P, cof = red.claims()
        self.assertTrue(all(len(q.params["coefficients"]) for q in cof))           # both relations are used
        mc.derive(P, [r1, r2], cof, scope)                                          # the right premises
        wrong = [("swapped premises", [r2, r1], cof), ("a premise replaced", [r1, r3], cof),
                 ("a premise dropped", [r1], cof[:1])]
        items = []
        for what, prem, cs in wrong:
            with self.assertRaises(DomainError, msg=what):
                mc.derive(P, prem, cs, scope)
            items.append((P, prem, cs))
        cpu, gpu = mg.check(items), mg.check(items, backend="cuda")
        self.assertEqual([r["holds"] for r in cpu], [False, False, False])
        self.assertEqual([(r["nonzero"], r["first_rank"]) for r in cpu], [(r["nonzero"], r["first_rank"]) for r in gpu])
        # the derived certificate records its premises: they are the relations
        c = red.certify()
        self.assertEqual(len(c.premises), 2)
        self.assertTrue(c.premises[0] == r1 and c.premises[1] == r2)
        # a cofactor reading a column outside the claim's columns is refused by FND-007
        Pn = mc.polynomial({(1, 0): 1, (0, 1): -1}, ["y", "x"], nulls="skip")
        with self.assertRaises((DomainError, ValueError)):
            mc.derive(Pn, [r1], [mc.polynomial({(1,): 1}, ["z"], nulls="skip")], scope)

    def test_geom001_equivalent_candidates_agree_on_certified_rows_cpu_and_cuda(self):
        n = 3000
        x = [int(v) for v in self.rng.integers(-1000, 1000, n)]
        y = [int(v) for v in self.rng.integers(-1000, 1000, n)]
        z = [a * b for a, b in zip(x, y)]
        s = [a + b for a, b in zip(x, y)]
        nulls = set(int(i) for i in self.rng.choice(n, 40, replace=False))
        yv = [None if i in nulls else v for i, v in enumerate(y)]
        sv = [None if i in nulls else v for i, v in enumerate(s)]
        zv = [None if i in nulls else v for i, v in enumerate(z)]
        db = make({"x": x, "y": yv, "z": zv, "s": sv})
        names = ["x", "y", "z", "s"]
        scope = mbk.Scope.from_table(db["t"], names)
        rz = cert(scope, {(1, 0, 0): 1, (0, 1, 1): -1}, ["z", "x", "y"])           # z = x y
        rs = cert(scope, {(1, 0, 0): 1, (0, 1, 0): -1, (0, 0, 1): -1}, ["s", "x", "y"])   # s = x + y
        I = mg.ideal([rz, rs], order="lex", variables=["z", "s", "x", "y"])
        fs = [mg.poly({(0, 0, 2, 0): 1, (0, 0, 0, 2): Fraction(-1, 3), (1, 0, 0, 0): 7}, names),
              mg.poly({(0, 0, 1, 1): 2, (2, 0, 0, 0): -1}, names),
              mg.poly({(0, 1, 0, 1): 1, (0, 0, 1, 0): 1}, names)]
        pairs = [(f, I.reduce(f).g) for f in fs]
        cpu = mg.agree(scope, pairs)
        gpu = mg.agree(scope, pairs, backend="cuda")
        keys = ("rows", "selected", "checked", "excluded", "disagreements", "witness", "undecided", "bits_f", "bits_g")
        self.assertEqual([[a[k] for k in keys] for a in cpu], [[a[k] for k in keys] for a in gpu])
        for (f, g), a in zip(pairs, cpu):
            used = set(f.used) | set(g.used)
            Df, Dg = denominator(f.terms), denominator(g.terms)
            checked = [r for r in range(n) if not any({"x": x, "y": yv, "z": zv, "s": sv}[c][r] is None for c in used)]
            self.assertEqual((a["checked"], a["excluded"], a["disagreements"]), (len(checked), n - len(checked), 0))
            pt = lambda r: [x[r], y[r], z[r], s[r]]
            self.assertEqual(a["bits_f"], max(bits(Df * evaluate(f.aligned(names), pt(r))) for r in checked))
            self.assertEqual(a["bits_g"], max(bits(Dg * evaluate(g.aligned(names), pt(r))) for r in checked))
        # a representation that is not equivalent disagrees, with the smallest disagreeing row
        off = mg.poly({(0, 1, 0, 1): 1, (0, 0, 1, 0): 1}, names)                   # y s + z = 2xy + y^2
        alt = mg.poly({(0, 1, 1, 0): 1, (0, 2, 0, 0): 1, (1, 1, 0, 0): 2}, names)    # y z + y^2 + 2xy: off by x y^2
        for backend in ("cpu", "cuda"):
            a = mg.agree(scope, [(off, alt)], backend=backend)[0]
            fx = [None if r in nulls else y[r] * s[r] + z[r] for r in range(n)]
            gx = [None if r in nulls else y[r] * z[r] + y[r] ** 2 + 2 * x[r] * y[r] for r in range(n)]
            bad = [r for r in range(n) if fx[r] is not None and fx[r] != gx[r]]
            self.assertEqual((a["disagreements"], a["witness"]), (len(bad), min(bad)))
        frame = pd.DataFrame({"x": x, "y": y, "z": z, "s": s})
        self.assertEqual(sql("SELECT count(*) FROM frame WHERE z <> x * y OR s <> x + y", frame=frame)[0], 0)

    # ---------------------------------------------------------- derivatives and declared dependencies
    def test_geom001_derivatives_respect_declared_dependencies(self):
        names = ["x", "y", "z"]
        f = mg.poly({(0, 1, 0): 1}, names)                                          # y
        g = mg.poly({(2, 0, 0): 1}, names)                                          # x^2 (equal to y on the rows)
        self.assertEqual(mg.derivative(f, "x"), mg.poly({}, names))                 # formal: 0
        self.assertEqual(mg.derivative(g, "x"), mg.poly({(1,): 2}, ["x"]))          # formal: 2x: they differ
        with self.assertRaises(DomainError) as e:
            mg.derivative(f, "x", through=True)                                     # a relation is no dependency
        self.assertEqual(e.exception.reason, "undeclared")
        dep = {"y": mg.poly({(2,): 1}, ["x"])}
        self.assertEqual(mg.derivative(f, "x", through=True, dependencies=dep), mg.poly({(1,): 2}, ["x"]))
        self.assertEqual(mg.derivative(g, "x", through=True, dependencies=dep), mg.poly({(1,): 2}, ["x"]))
        with self.assertRaises(DomainError) as e:
            mg.derivative(f, "y", through=True, dependencies=dep)
        self.assertEqual(e.exception.reason, "dependent")
        with self.assertRaises(DomainError) as e:
            mg.derivative(f, "x", through=True, dependencies={"y": mg.poly({(1,): 1}, ["z"]),
                                                              "z": mg.poly({(1, 1): 1}, ["y", "x"])})
        self.assertEqual(e.exception.reason, "cycle")
        with self.assertRaises(ValueError):
            mg.derivative(f, "x", dependencies=dep)                                 # formal: no declarations
        # the chain rule through a declared chain z = y + x^3, y = x^2 - 3x, checked by substitution
        deps = {"z": mg.poly({(0, 1): 1, (3, 0): 1}, ["x", "y"]), "y": mg.poly({(2,): 1, (1,): -3}, ["x"])}
        h = mg.poly({(1, 1, 1): 2, (0, 0, 2): Fraction(1, 2), (3, 0, 0): -1, (0, 2, 0): 5}, names)
        got = mg.derivative(h, "x", through=True, dependencies=deps)
        phi_y, phi_z = over(deps["y"], names), over(deps["z"], names)

        def full(p):
            p = substitute(p, 2, phi_z)
            return substitute(p, 1, phi_y)

        self.assertEqual(full(over(got, names)), partial(full(over(h, names)), 0))
        # two representations equal on the dependency have equal total derivatives (and different formal ones)
        h2 = substitute(over(h, names), 2, phi_z)
        d1 = mg.derivative(h, "x", through=True, dependencies=deps)
        d2 = mg.derivative(mg.poly(h2, names), "x", through=True, dependencies=deps)
        self.assertEqual(full(over(d1, names)), full(over(d2, names)))
        self.assertNotEqual(over(mg.derivative(h, "x"), names), over(mg.derivative(mg.poly(h2, names), "x"), names))
        # with only z = y + x^3 declared, y is independent: d/dy = dh/dy + dh/dz dz/dy
        self.assertEqual(mg.derivative(h, "y", through=True, dependencies={"z": deps["z"]}),
                         mg.poly(add(partial(over(h, names), 1), mul(partial(over(h, names), 2), partial(phi_z, 1))), names))

    def test_geom001_dependency_declarations_are_verified_on_every_row(self):
        n = 500
        x = [int(v) for v in self.rng.integers(-40, 40, n)]
        y = [a * a - 3 * a for a in x]
        z = [b + a ** 3 for a, b in zip(x, y)]
        z[311] += 1                                                                 # the declaration fails once
        x[17] = None
        db = make({"x": x, "y": y, "z": z})
        scope = mbk.Scope.from_table(db["t"], ["x", "y", "z"])
        deps = {"y": mg.poly({(2,): 1, (1,): -3}, ["x"]), "z": mg.poly({(0, 1): 1, (3, 0): 1}, ["x", "y"]),
                "x": mg.poly({(1,): Fraction(1, 2)}, ["y"])}
        ev = mg.dependency_check(scope, deps)
        self.assertEqual([(e["checked"], e["excluded"], e["violations"], e["witness"]) for e in ev[:2]],
                         [(n - 1, 1, 0, None), (n - 1, 1, 1, 311)])
        bad = [r for r in range(n) if x[r] is not None and Fraction(y[r], 2) != x[r]]
        self.assertEqual((ev[2]["violations"], ev[2]["witness"]), (len(bad), min(bad)))
        self.assertTrue(ev[0]["holds"])
        self.assertFalse(ev[1]["holds"])

    # ---------------------------------------------------------- budgets, costs, batches
    def test_geom001_completion_is_refused_beyond_its_budget(self):
        db = cubic_table()
        scope = mbk.Scope.from_table(db["t"], ["x", "y", "z"])
        r1 = cert(scope, {(1, 0): 1, (0, 2): -1}, ["y", "x"])
        r2 = cert(scope, {(1, 0): 1, (0, 3): -1}, ["z", "x"])
        I = mg.ideal([r1, r2], order="deglex", variables=["x", "y", "z"])
        info = I.info
        need = info["pairs"] + info["reductions"]
        self.assertEqual(mg.ideal([r1, r2], order="deglex", variables=["x", "y", "z"],
                                  budget=Budget(iterations=need)).info["basis"], 4)
        with self.assertRaises(AlgebraBudgetError) as e:
            mg.ideal([r1, r2], order="deglex", variables=["x", "y", "z"], budget=Budget(iterations=need - 1))
        self.assertEqual((e.exception.resource, e.exception.need, e.exception.limit), ("iterations", need, need - 1))
        # the smallest passing limit of each resource is the peak the completion reports; below it the request is
        # refused naming that resource (never a truncated basis)
        smallest = {}
        for resource in ("output", "support", "degree"):
            limit = 1
            while True:
                try:
                    mg.ideal([r1, r2], order="deglex", variables=["x", "y", "z"], budget=Budget(**{resource: limit}))
                    break
                except AlgebraBudgetError as err:
                    self.assertEqual(err.resource, resource)
                    self.assertGreater(err.need, err.limit)
                    limit += 1
            smallest[resource] = limit
        self.assertEqual((smallest["support"], smallest["degree"]), (info["max_terms"], info["max_degree"]))
        self.assertGreaterEqual(smallest["output"], info["basis"])
        # complete=False never iterates Buchberger; reducing has its own steps
        J = mg.ideal([r1, r2], order="deglex", variables=["x", "y", "z"], complete=False)
        self.assertEqual((J.info["pairs"], J.info["basis"]), (0, 2))
        f = mg.poly({(9, 0, 0): 1}, ["x", "y", "z"])
        steps = I.reduce(f).info["steps"]
        I.reduce(f, budget=Budget(iterations=steps))
        with self.assertRaises(AlgebraBudgetError):
            I.reduce(f, budget=Budget(iterations=steps - 1))

    def test_geom001_choose_the_cheaper_representation_and_term_explosion(self):
        n = 400
        x = [int(v) for v in self.rng.integers(-5000, 5000, n)]
        db = make({"x": x, "y": [a + 1 for a in x], "z": [a * (a + 1) for a in x], "w": [3 * a for a in x]})
        names = ["x", "y", "z", "w"]
        scope = mbk.Scope.from_table(db["t"], names)
        ry = cert(scope, {(1, 0): 1, (0, 1): -1, (0, 0): -1}, ["y", "x"])          # y = x + 1
        rz = cert(scope, {(1, 0, 0): 1, (0, 1, 1): -1}, ["z", "x", "y"])           # z = x y
        I = mg.ideal([ry, rz], order="lex", variables=["z", "w", "y", "x"])
        f = mg.poly({(0, 6): 1}, ["x", "y"])                                        # y^6: one term
        g = I.reduce(f).g                                                           # (x + 1)^6: seven terms
        self.assertEqual(g, mg.poly({(k,): math.comb(6, k) for k in range(7)}, ["x"]))
        M = {c: max(abs(min(v)), abs(max(v)), 1) for c, v in (("x", x), ("y", [a + 1 for a in x]))}

        def oracle(p):
            D = denominator(p.terms)
            ints = {e: c * D for e, c in p.terms.items()}
            bound = sum(abs(int(c)) * math.prod(M[v] ** k for v, k in zip(p.columns, e) if k) for e, c in ints.items())
            used = p.used
            return dict(terms=len(p.terms), columns=len(used), degree=max(sum(e) for e in p.terms),
                        multiplications=sum(max(sum(e) - 1, 0) + (abs(c) != 1 and sum(e) > 0) for e, c in ints.items()),
                        bound_bits=bound.bit_length(), denominator=D)

        for p in (f, g):
            c = mg.cost(scope, p)
            self.assertEqual({k: c[k] for k in oracle(p)}, oracle(p))
        idx, costs = mg.choose(scope, [f, g], priority="terms")
        self.assertEqual(idx, 0)                                                    # the reduction is no cheaper here
        idx, costs = mg.choose(scope, [g, f], priority="bits", measure=True)
        self.assertEqual(idx, 1)
        self.assertEqual(costs[1]["actual_bits"], max(bits((a + 1) ** 6) for a in x))
        self.assertTrue(all(c["agrees"] for c in costs))
        self.assertEqual(mg.choose(scope, [f, g], priority="columns")[0], 0)        # ties on columns, then bits
        # here the reduction is the cheaper one: x y - z + 3x -> 3x (one term, one column)
        h = mg.poly({(1, 1, 0): 1, (0, 0, 1): -1, (1, 0, 0): 3}, ["x", "y", "z"])
        hg = I.reduce(h).g
        self.assertEqual(hg, mg.poly({(1,): 3}, ["x"]))
        for priority in ("columns", "bits", "terms"):
            self.assertEqual(mg.choose(scope, [h, hg], priority=priority, measure=True, backend="cuda")[0], 1)
        # a term explosion past the support budget is refused, never truncated
        with self.assertRaises(AlgebraBudgetError) as e:
            I.reduce(mg.poly({(0, 40): 1}, ["x", "y"]), budget=Budget(support=30))
        self.assertEqual(e.exception.resource, "support")
        self.assertEqual(len(I.reduce(mg.poly({(0, 40): 1}, ["x", "y"])).g), 41)

    def test_geom001_batched_checks_cpu_equal_cuda_and_overflow_is_undecided(self):
        names = ["x", "y", "z"]
        rng = self.rng
        items, truth = [], []
        rels = [mc.polynomial({(1, 0, 0): 1, (0, 2, 1): -2, (0, 0, 0): 3}, names, nulls="skip"),
                mc.polynomial({(0, 1, 0): 1, (1, 0, 1): 1}, names, nulls="skip")]
        relp = [{(1, 0, 0): 1, (0, 2, 1): -2, (0, 0, 0): 3}, {(0, 1, 0): 1, (1, 0, 1): 1}]
        for k in range(40):
            qs = []
            for _ in rels:
                terms = {tuple(int(v) for v in rng.integers(0, 3, 3)): int(rng.integers(-9, 10)) for _ in range(3)}
                qs.append(clean(terms))
            Pp = {}
            for q, r in zip(qs, relp):
                Pp = add(Pp, mul(q, r))
            if k % 3 == 1 and Pp:
                e0 = sorted(Pp)[0]
                Pp = add(Pp, {e0: 1})                                               # a wrong coefficient
            if k % 3 == 2:
                Pp = add(Pp, {(4, 4, 4): 2})                                        # a stray term
            P = mc.polynomial({e: int(c) for e, c in Pp.items()} or {(0, 0, 0): 0}, names, nulls="skip")
            Q = [mc.polynomial({e: int(c) for e, c in q.items()} or {(0, 0, 0): 0}, names, nulls="skip") for q in qs]
            items.append((P, rels, Q))
            diff = add(Pp, add(mul(qs[0], relp[0]), mul(qs[1], relp[1])), -1)
            truth.append(diff)
        cpu = mg.check(items, names=names)
        gpu = mg.check(items, names=names, backend="cuda")
        key = lambda r: (r["holds"], r["decided"], r["nonzero"], r["first_rank"], r["first"], r["slots"], r["work"])
        self.assertEqual([key(r) for r in cpu], [key(r) for r in gpu])
        for r, diff in zip(cpu, truth):
            self.assertEqual(r["holds"], not diff)
            self.assertEqual(r["nonzero"], len(diff))
            if diff:
                ranks = {mg.unrank(k, 3): k for k in range(r["slots"])}
                self.assertEqual(r["first_rank"], min(ranks[e] for e in diff))
        self.assertTrue(any(r["holds"] for r in cpu) and not all(r["holds"] for r in cpu))
        # magnitudes past the proven 2^127 - 1 bound: undecided (never a wrapped verdict) on both backends
        big = (1 << 63) - 1
        P = mc.polynomial({(1, 0, 0): 1}, names, nulls="skip")
        G = mc.polynomial({(1, 0, 0): big, (0, 1, 0): big}, names, nulls="skip")
        Q = mc.polynomial({(0, 0, 0): big, (0, 0, 1): big}, names, nulls="skip")
        for backend in ("cpu", "cuda"):
            r = mg.check([(P, [G], [Q])], names=names, backend=backend)[0]
            self.assertEqual((r["holds"], r["decided"], r["slots"], r["work"]), (False, False, 0, 0))
        need = mg.check_need(items, names=names, backend="cuda")
        self.assertEqual(need["states"], sum(r["slots"] for r in cpu))
        for backend in ("cpu", "cuda"):                                     # at the limit, then one past it
            self.assertEqual(len(mg.check(items, names=names, budget=Budget(states=need["states"]), backend=backend)), 40)
            with self.assertRaises(AlgebraBudgetError) as e:
                mg.check(items, names=names, budget=Budget(states=need["states"] - 1), backend=backend)
            self.assertEqual(e.exception.resource, "states")

    def test_geom001_dense_rank_is_a_graded_bijection(self):
        for nv in range(1, 5):
            for d in range(0, 6):
                count = math.comb(nv + d, nv)
                seen = [mg.unrank(r, nv) for r in range(count)]
                want = {e for e in itertools.product(range(d + 1), repeat=nv) if sum(e) <= d}
                self.assertEqual(set(seen), want)
                self.assertEqual(len(seen), len(want))
                self.assertEqual([sum(e) for e in seen], sorted(sum(e) for e in seen))
        # one monomial left over: the check names it
        names = ["x", "y", "z"]
        for e in ((0, 0, 0), (2, 1, 0), (0, 3, 1), (1, 1, 1)):
            P = mc.polynomial({e: 5}, names, nulls="skip")
            G = mc.polynomial({(1, 0, 0): 1}, names, nulls="skip")
            Z = mc.polynomial({(0, 0, 0): 0}, names, nulls="skip")
            for backend in ("cpu", "cuda"):
                r = mg.check([(P, [G], [Z])], names=names, backend=backend)[0]
                self.assertEqual((r["nonzero"], r["first"]), (1, {v: k for v, k in zip(names, e) if k}))

    # ---------------------------------------------------------- DISC-002 identities end to end
    def test_geom001_reduces_with_disc002_identities_end_to_end(self):
        n = 400
        x = [int(v) for v in self.rng.integers(-30, 31, n)]
        y = [int(v) for v in self.rng.integers(-25, 40, n)]
        v = {"a": [3 * p - 7 for p in x], "s": [p + q for p, q in zip(x, y)], "z": [p * q for p, q in zip(x, y)],
             "x": x, "y": y}
        db = make(v)
        t = db["t"]
        names = ["a", "s", "z", "x", "y"]
        scope = mbk.Scope.from_table(t, names)
        found = mi.discover(scope, names, degree=2)
        rels = [i["certificate"] for i in found.discoveries()]
        self.assertEqual(len(rels), 3)
        I = mg.ideal(rels, order="lex", variables=["a", "s", "z"])
        f = mg.poly({(1, 1, 0, 0, 0): 1, (0, 0, 1, 0, 0): -3}, names)               # a s - 3 z
        red = I.reduce(f)
        self.assertEqual(red.g, mg.poly({(2, 0): 3, (1, 0): -7, (0, 1): -7}, ["x", "y"]))   # 3x^2 - 7x - 7y
        c = red.certify()
        self.assertEqual((c.kind, c.info["exact"], c.info["premises"]), ("snapshot", True, 3))
        self.assertEqual(c.authorize_exact(scope).nulls, "skip")
        for backend in ("cpu", "cuda"):
            a = mg.agree(scope, [(f, red.g)], backend=backend)[0]
            self.assertEqual((a["checked"], a["disagreements"]), (n, 0))
            self.assertTrue(mg.check([red], backend=backend)[0]["holds"])
        idx, costs = mg.choose(scope, [f, red.g], priority="columns", measure=True)
        self.assertEqual(idx, 1)                                                    # two columns instead of three
        frame = pd.DataFrame(v)
        self.assertEqual(sql("SELECT count(*) FROM frame WHERE a * s - 3 * z <> 3 * x * x - 7 * x - 7 * y",
                             frame=frame)[0], 0)
        t.flip(0, 3)
        now = mbk.Scope.from_table(t, names)
        with self.assertRaises((StaleStateError, mc.NotAuthorizedError)):
            c.authorize_exact(now)
        with self.assertRaises(StaleStateError):
            mg.agree(scope, [(f, red.g)])

    def test_geom001_validation_and_representation_limits(self):
        db = cubic_table()
        scope = mbk.Scope.from_table(db["t"], ["x", "y", "z"])
        r1 = cert(scope, {(1, 0): 1, (0, 2): -1}, ["y", "x"])
        with self.assertRaises(UnsupportedError):
            mg.ideal([r1] * 17)
        with self.assertRaises(ValueError):
            mg.ideal([r1], order="revlex")
        with self.assertRaises(TypeError):
            mg.ideal([mc.polynomial({(1, 0): 1}, ["x", "y"])])
        with self.assertRaises(ValueError):
            mg.ideal([], order="lex")
        I = mg.ideal([r1], order="lex", variables=["y", "x"])
        with self.assertRaises(ValueError):
            I.reduce(mg.poly({(1,): 1}, ["q"]))                                     # not a column of the scope
        with self.assertRaises(UnsupportedError):
            mg.poly({tuple([1] * 9): 1}, [f"c{i}" for i in range(9)])
        # coefficients past int64 are an overflow, never rounded: y^30 with 3y = x gives x^30 / 3^30... over int64
        xs = [3 * k for k in range(-5, 6)]
        dq = make({"x": xs, "y": [k // 3 for k in xs]}, "q")
        sq = mbk.Scope.from_table(dq["q"], ["x", "y"])
        third = cert(sq, {(0, 1): 3, (1, 0): -1}, ["x", "y"])                      # 3y - x
        J = mg.ideal([third], order="lex", variables=["y", "x"])
        self.assertEqual(J.reduce(mg.poly({(0, 2): 1}, ["x", "y"])).g, mg.poly({(2,): Fraction(1, 9)}, ["x"]))
        with self.assertRaises(OverflowError):
            J.reduce(mg.poly({(0, 41): 1}, ["x", "y"]))                             # 3^41 > 2^63
        r = J.reduce(mg.poly({(0, 39): 1}, ["x", "y"]))                             # 3^39 < 2^63: claimable
        self.assertEqual((r.info["claimable"], r.info["denominator"]), (True, 3 ** 39))
        self.assertEqual(r.certify().authorize_exact(sq).kind, "snapshot")
        # a certificate whose denominator lcm(5^20, 3^39) passes int64: no FND-007 claim, no certificate
        r = J.reduce(mg.poly({(0, 39): 1, (1, 0): Fraction(1, 5 ** 20)}, ["x", "y"]))
        self.assertEqual((r.info["claimable"], r.info["code"], r.info["denominator"]), (False, -3, None))
        with self.assertRaises(OverflowError):
            r.certify()
        with self.assertRaises(OverflowError):
            r.claims()
        with self.assertRaises(UnsupportedError):
            mg.agree(scope, [(mg.poly({(1,): 1}, ["x"]), mg.poly({(1,): 1}, ["x"]))] * 257)


if __name__ == "__main__":
    unittest.main()
