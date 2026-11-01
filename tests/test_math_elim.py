# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""GEOM-002: bounded elimination and generator selection.

Oracles (independent of the implementation): polynomials as dictionaries of Fractions with Buchberger's algorithm and
the division algorithm written here (the reduced Groebner basis under an elimination order is unique, so the generator
sets an exact search may admit are decided here independently), every reconstruction and every form evaluated row by
row with Python integers and Fractions (values, SQL NULL propagation, ranges), the costs recomputed from the scope's
extremes and the measured values, DuckDB counts on the planted data. FND-007 (tr_cert_derive) checks the certificates.
Every check has a CUDA path (emulated here, threads in a scrambled order) compared for equality. Seeds are fixed.
"""
import itertools
import math
import os
import subprocess
import sys
import unittest
from fractions import Fraction

import duckdb
import numpy as np
import pandas as pd

import torbles
from torbles import math_bank as mbk
from torbles import math_certificates as mc
from torbles import math_elim as me
from torbles import math_geom as mg
from torbles import math_ident as mi
from torbles.math_budgets import Budget
from torbles.math_domains import DomainError
from torbles.math_errors import StaleStateError, UnsupportedError
from torbles.mathematics import AlgebraBudgetError

SEED = 2002


def make(values, name="t"):
    frame = pd.DataFrame({c: pd.Series(v, dtype=object) for c, v in values.items()})
    ddl = f"CREATE TABLE {name} (" + ", ".join(f'"{c}" BIGINT' for c in values) + ")"
    return torbles.from_ddl(ddl, {name: frame})


def cert(scope, terms, cols, nulls="skip"):
    return mc.snapshot(scope, mc.polynomial(terms, cols, nulls=nulls))


# ------------------------------------------------------------------ the polynomial oracle (Fractions)
class Order:
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
    p, r = dict(f), {}
    while p:
        e, c = lead(p, order)
        for g in G:
            eg, cg = lead(g, order)
            if divides(eg, e):
                p = add(p, g, -c / cg, tuple(x - y for x, y in zip(e, eg)))
                break
        else:
            r[e] = c
            del p[e]
    return clean(r)


def monic(p, order):
    _, c = lead(p, order)
    return {e: v / c for e, v in p.items()}


def spoly(f, g, order):
    (ef, cf), (eg, cg) = lead(f, order), lead(g, order)
    L = tuple(max(a, b) for a, b in zip(ef, eg))
    return add(add({}, f, 1 / cf, tuple(l - a for l, a in zip(L, ef))), g, -1 / cg, tuple(l - b for l, b in zip(L, eg)))


def groebner(F, order):
    G = [monic(f, order) for f in F if f]
    pairs = [(i, j) for i in range(len(G)) for j in range(i)]
    while pairs:
        i, j = pairs.pop()
        li, lj = lead(G[i], order)[0], lead(G[j], order)[0]
        if not any(a and b for a, b in zip(li, lj)):                  # coprime leading monomials: reduces to 0
            continue
        r = divide(spoly(G[i], G[j], order), G, order)
        if r:
            G.append(monic(r, order))
            pairs += [(len(G) - 1, k) for k in range(len(G) - 1)]
    lm = [lead(g, order)[0] for g in G]
    keep = [g for i, g in enumerate(G) if not any(k != i and divides(lm[k], lm[i]) and (lm[k] != lm[i] or k < i)
                                                  for k in range(len(G)))]
    out = []
    for i, g in enumerate(keep):
        e, _ = lead(g, order)
        r = divide({m: v for m, v in g.items() if m != e}, keep[:i] + keep[i + 1:], order)
        out.append(clean({e: 1, **r}))
    return out


def evaluate(p, point):
    total = Fraction(0)
    for e, c in p.items():
        v = c
        for x, k in zip(point, e):
            v *= x ** k
        total += v
    return total


def over(p, names):
    return clean(p.aligned(names))


def unit(names, v):
    return {tuple(int(n == v) for n in names): Fraction(1)}


def oracle_reconstructions(rels, names, variables, gen):
    """{x: NF(x)} under lex (the eliminated variables first, in variable order, then the generators), or None when a
    normal form leaves the generators"""
    elim = [v for v in variables if v not in gen]
    seq = [names.index(v) for v in elim + [v for v in variables if v in gen]]
    order = Order("lex", seq + [i for i in range(len(names)) if i not in seq])
    G = groebner([clean(r) for r in rels], order)
    out = {}
    for x in elim:
        g = divide(unit(names, x), G, order)
        if any(e[names.index(c)] for e in g for c in names if c not in gen):
            return None
        out[x] = g
    return out


def row_check(rows, names, x, f):
    """the reconstruction f of column x on every row: (compared, nulls, null_lost, null_extra, values, first, range)"""
    used = [c for q, c in enumerate(names) if any(e[q] for e in f)]
    comp = nulls = lost = extra = vals = 0
    first = None
    recon = []
    for r, row in enumerate(rows):
        px = row[x] is not None
        pf = all(row[c] is not None for c in used)
        if not px and not pf:
            nulls += 1
            continue
        if px != pf:
            lost += px
            extra += pf
            first = r if first is None else first
            continue
        comp += 1
        y = evaluate(f, [row[c] if row[c] is not None else 0 for c in names])
        recon.append(y)
        if y != row[x]:
            vals += 1
            first = r if first is None else first
    present = [row[x] for row in rows if row[x] is not None]
    rng = (min(recon), max(recon)) if recon else None
    return dict(compared=comp, nulls=nulls, null_lost=lost, null_extra=extra, values=vals, first=first,
                range=rng, stored=(min(present), max(present)) if present else None)


def width_of(values):
    present = [v for v in values if v is not None]
    bits = max((max(present) - min(present)).bit_length(), 1) if present else 1
    return bits + (1 if any(v is None for v in values) else 0)


class Elim002(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(SEED)

    def planted(self, n=300, nulls=True):
        x = [int(v) for v in self.rng.integers(-40, 41, n)]
        y = [int(v) for v in self.rng.integers(-40, 41, n)]
        data = {"x": x, "y": y, "z": [a * b for a, b in zip(x, y)], "s": [a + b for a, b in zip(x, y)],
                "w": [a * a - b for a, b in zip(x, y)], "u": [int(v) for v in self.rng.integers(-5, 6, n)]}
        if nulls:
            xn = set(int(i) for i in self.rng.choice(n, 12, replace=False))
            for c in ("x", "z", "s", "w"):                               # NULL x: every column defined from it NULL
                data[c] = [None if i in xn else v for i, v in enumerate(data[c])]
        db = make(data)
        names = list(data)
        scope = mbk.Scope.from_table(db["t"], names)
        rels = [{(1, 0, 0): 1, (0, 1, 1): -1}, {(1, 0, 0): 1, (0, 1, 0): -1, (0, 0, 1): -1},
                {(1, 0, 0): 1, (0, 2, 0): -1, (0, 0, 1): 1}]
        cols = [["z", "x", "y"], ["s", "x", "y"], ["w", "x", "y"]]
        certs = [cert(scope, t, c) for t, c in zip(rels, cols)]
        full = []
        for t, c in zip(rels, cols):
            full.append({tuple(dict(zip(c, e)).get(v, 0) for v in names): k for e, k in t.items()})
        rows = [dict(zip(names, vals)) for vals in zip(*[data[c] for c in names])]
        return db, data, names, scope, certs, full, rows

    # ---------------------------------------------------------- acceptance 1: coverage, ranges, NULL domains
    def test_geom002_reconstruction_covers_every_supported_variable_and_preserves_ranges_and_null_domains(self):
        db, data, names, scope, certs, full, rows = self.planted()
        cols = ["x", "y", "z", "s", "w", "u"]                          # u is in no relation: always a generator
        sel = me.select(scope, certs, columns=cols)
        info = sel.info
        self.assertEqual(info["variables"], cols)
        self.assertEqual(sorted(info["generators"] + info["eliminated"]), sorted(cols))     # every variable covered
        self.assertFalse(set(info["generators"]) & set(info["eliminated"]))
        self.assertIn("u", info["generators"])
        self.assertEqual(info["minimum"], 3)                            # two free columns and u
        self.assertTrue(info["minimal"])
        checks = sel.check()
        gpu = sel.check(backend="cuda")
        self.assertIn(gpu[info["eliminated"][0]]["backend"], ("cuda", "cuda-emulated"))
        strip = lambda d: {k: v for k, v in d.items() if k not in ("backend", "device")}
        self.assertEqual({k: strip(v) for k, v in checks.items()}, {k: strip(v) for k, v in gpu.items()})
        for x, f in sel.eliminated.items():
            fx = over(f, names)
            self.assertTrue(set(f.used) <= set(info["generators"]))      # reconstructed from the generators alone
            o = row_check(rows, names, x, fx)
            c = checks[x]
            for k in ("compared", "nulls", "null_lost", "null_extra", "values", "first"):
                self.assertEqual(c[k], o[k], (x, k))
            self.assertTrue(c["holds"])
            self.assertEqual(o["range"], o["stored"])                    # the range is preserved, checked here too
            self.assertEqual((c["below"], c["above"]), (0, 0))
            self.assertGreater(c["at_lo"], 0)
            self.assertGreater(c["at_hi"], 0)
            # the NULL mask: NULL exactly where the stored column is
            mask = [any(row[g] is None for g in f.used) for row in rows]
            self.assertEqual(mask, [row[x] is None for row in rows])
        # a column NULL where its generators are present is not reconstructed by them (checked, not assumed)
        data2 = dict(data)
        data2["z"] = [None if i in (3, 150, 222) else v for i, v in enumerate(data["z"])]
        db2 = make(data2, "t2")
        scope2 = mbk.Scope.from_table(db2["t2"], names)
        certs2 = [cert(scope2, t, list(dict.fromkeys(c))) for t, c in
                  zip([{(1, 0, 0): 1, (0, 1, 1): -1}, {(1, 0, 0): 1, (0, 1, 0): -1, (0, 0, 1): -1}],
                      [["z", "x", "y"], ["s", "x", "y"]])]
        sel2 = me.select(scope2, certs2, columns=["x", "y", "z", "s"])
        cand = {tuple(c["generators"]): c for c in sel2.candidates}
        self.assertEqual((cand[("x", "y")]["status"], cand[("x", "y")]["reason"], cand[("x", "y")]["failed"]),
                         ("rejected", "null mismatch", "z"))
        self.assertIn("z", sel2.generators)                             # kept as a generator
        rows2 = [dict(zip(names, vals)) for vals in zip(*[data2[c] for c in names])]
        bad = me.check(scope2, [("z", mg.poly({(1, 1): 1}, ["x", "y"]))], backend="cuda")[0]
        want = row_check(rows2, names, "z", over(mg.poly({(1, 1): 1}, ["x", "y"]), names))
        self.assertEqual((bad["null_extra"], bad["null_lost"], bad["first"], bad["holds"]),
                         (want["null_extra"], want["null_lost"], want["first"], False))
        self.assertEqual(bad["null_extra"], 3)
        # the reconstruction checked on another scope: the relation fails on two rows there; values and range report it
        data3 = {c: [v for v in data[c]] for c in names}
        data3["z"][10] = (data3["z"][10] or 0) + 10 ** 6                 # above the range of x y
        data3["z"][40] = -(10 ** 6)                                     # below it
        db3 = make(data3, "t3")
        scope3 = mbk.Scope.from_table(db3["t3"], names)
        f = sel.form("z")
        c3 = me.check(scope3, [("z", f)])[0]
        rows3 = [dict(zip(names, vals)) for vals in zip(*[data3[c] for c in names])]
        o3 = row_check(rows3, names, "z", over(f, names))
        self.assertEqual((c3["values"], c3["first_value"], c3["holds"]), (o3["values"], 10, False))
        stored = o3["stored"]
        recon = [evaluate(over(f, names), [r[c] if r[c] is not None else 0 for c in names]) for r in rows3
                 if r["z"] is not None and r["x"] is not None]
        self.assertEqual(c3["below"], sum(1 for v in recon if v < stored[0]))
        self.assertEqual(c3["above"], sum(1 for v in recon if v > stored[1]))
        self.assertEqual((c3["at_lo"], c3["at_hi"]), (sum(v == stored[0] for v in recon), sum(v == stored[1] for v in recon)))
        self.assertEqual(c3, {**me.check(scope3, [("z", f)], backend="cuda")[0], "backend": "cpu", "device": "native CPU"})

    # ---------------------------------------------------------- the search and elimination orders against the oracle
    def test_geom002_bounded_search_and_elimination_orders_match_the_groebner_oracle(self):
        n = 200
        x = [int(v) for v in self.rng.integers(-40, 41, n)]
        y = [int(v) for v in self.rng.integers(-40, 41, n)]
        data = {"x": x, "y": y, "z": [a * b for a, b in zip(x, y)], "s": [a + b for a, b in zip(x, y)],
                "w": [a + 1 for a in x]}
        db = make(data)
        names = variables = list(data)
        scope = mbk.Scope.from_table(db["t"], names)
        defs = [({(1, 0, 0): 1, (0, 1, 1): -1}, ["z", "x", "y"]), ({(1, 0, 0): 1, (0, 1, 0): -1, (0, 0, 1): -1}, ["s", "x", "y"]),
                ({(1, 0): 1, (0, 1): -1, (0, 0): -1}, ["w", "x"])]
        certs = [cert(scope, t, c) for t, c in defs]
        full = [{tuple(dict(zip(c, e)).get(v, 0) for v in names): k for e, k in t.items()} for t, c in defs]
        rows = [dict(zip(names, vals)) for vals in zip(*[data[c] for c in names])]
        sel = me.select(scope, certs, columns=variables, rank_all=True)
        # the oracle: every subset by increasing size, lexicographic positions; feasible when every normal form lies
        # in the generators and every reconstruction checks on the rows
        feasible, minimum = [], None
        for k in range(len(variables)):
            for gen in itertools.combinations(variables, k):
                rec = oracle_reconstructions(full, names, variables, set(gen))
                ok = rec is not None and all(row_check(rows, names, x_, f)["values"] == 0 for x_, f in rec.items())
                if ok:
                    feasible.append(gen)
            if feasible:
                minimum = k
                break
        self.assertEqual(sel.info["minimum"], minimum)
        cands = sel.candidates
        self.assertEqual(sum(math.comb(len(variables), k) for k in range(minimum + 1)), len(cands))
        self.assertEqual(sel.info["solves"], len(cands))
        admitted = [tuple(c["generators"]) for c in cands if c["status"] in ("chosen", "admissible")]
        self.assertEqual(admitted, feasible)
        rejected = [tuple(c["generators"]) for c in cands if c["status"] == "rejected"]
        for gen in rejected:
            self.assertIsNone(oracle_reconstructions(full, names, variables, set(gen)), gen)
        # every admitted set's reconstructions equal the oracle's normal forms
        for gen in feasible:
            one = me.select(scope, certs, columns=variables, mode="order",
                            order=[v for v in variables if v not in gen] + list(gen))
            self.assertEqual(tuple(one.generators), gen)
            want = oracle_reconstructions(full, names, variables, set(gen))
            self.assertEqual({x: over(f, names) for x, f in one.eliminated.items()}, want)
        # the chosen set is the cheapest admitted one by the total cost (ties to the earliest)
        tot = [(c["total"], i) for i, c in enumerate(cands) if c["status"] in ("chosen", "admissible")]
        self.assertEqual(cands[min(tot)[1]]["status"], "chosen")
        # one elimination order: the variables whose normal form is not themselves, in the oracle's reduced basis
        for seq in (["z", "s", "w", "x", "y"], ["x", "z", "s", "w", "y"], ["w", "s", "z", "y", "x"], ["y", "s", "x", "z", "w"]):
            o = me.select(scope, certs, columns=variables, mode="order", order=seq)
            order = Order("lex", [names.index(v) for v in seq] + [i for i in range(len(names)) if names[i] not in seq])
            G = groebner([clean(r) for r in full], order)
            elim = [v for v in seq if divide(unit(names, v), G, order) != unit(names, v)]
            self.assertEqual(sorted(o.info["eliminated"]), sorted(elim), seq)
            self.assertEqual(o.info["solves"], 1)
            self.assertFalse(o.info["minimal"])
        # stop at the first feasible set
        first = me.select(scope, certs, columns=variables, rank_all=False)
        self.assertEqual(tuple(first.generators), feasible[0])
        self.assertLess(first.info["solves"], sel.info["solves"])
        # max_size below the minimum: nothing eliminated, minimality not claimed
        none = me.select(scope, certs, columns=variables, max_size=minimum - 1)
        self.assertEqual((none.generators, none.info["eliminated"], none.info["minimal"]), (variables, [], False))
        self.assertEqual(none.candidates[-1]["reason"], "direct")

    # ---------------------------------------------------------- acceptance 2: budgets at the limit and one past
    def test_geom002_degree_term_and_solve_budgets_are_enforced_at_the_limit_and_one_past(self):
        db, data, names, scope, certs, full, rows = self.planted(n=150, nulls=False)
        variables = ["x", "y", "z", "s", "w"]
        sel = me.select(scope, certs, columns=variables)
        solves = sel.info["solves"]
        self.assertEqual(me.select(scope, certs, columns=variables, budget=Budget(iterations=solves)).generators,
                         sel.generators)
        with self.assertRaises(AlgebraBudgetError) as e:
            me.select(scope, certs, columns=variables, budget=Budget(iterations=solves - 1))
        self.assertEqual((e.exception.resource, e.exception.need, e.exception.limit), ("iterations", solves, solves - 1))
        # degree and support bound every completion: at the smallest limit deciding every set the search is the
        # unbudgeted one; one below, the sets past it are undecided (rejected "budget" with need = limit, never
        # guessed), the search goes on and minimality is only claimed when every smaller set was decided
        reasons = [(c["generators"], c["reason"]) for c in sel.candidates]
        self.assertEqual((sel.info["undecided"], sel.info["undecided_size"], sel.info["minimal"]), (0, None, True))
        for resource in ("degree", "support"):
            limit = 1
            while me.select(scope, certs, columns=variables, budget=Budget(**{resource: limit})).info["undecided"]:
                limit += 1
            at = me.select(scope, certs, columns=variables, budget=Budget(**{resource: limit}))
            self.assertEqual((at.generators, at.info["minimal"], at.info["solves"]), (sel.generators, True, solves))
            self.assertEqual([(c["generators"], c["reason"]) for c in at.candidates], reasons)
            self.assertTrue(all(c["refusal"] is None for c in at.candidates))
            self.assertGreaterEqual(limit, sel.info["max_degree" if resource == "degree" else "max_terms"])
            past = me.select(scope, certs, columns=variables, budget=Budget(**{resource: limit - 1}))
            und = [c for c in past.candidates if c["reason"] == "budget"]
            self.assertTrue(und)
            self.assertEqual(past.info["undecided"], len(und))
            self.assertEqual(past.info["undecided_size"], min(c["size"] for c in und))
            for c in und:
                self.assertEqual((c["status"], c["refusal"]), ("rejected", dict(resource=resource, need=limit, limit=limit - 1)))
            decided = {tuple(c["generators"]): c["reason"] for c in past.candidates if c["reason"] != "budget"}
            for g, r in reasons:                                       # every decided set decided as without a budget
                if tuple(g) in decided and r not in ("cheapest", "costlier"):
                    self.assertEqual(decided[tuple(g)], r, g)
            chosen = past.candidates[past.info["chosen"]]
            self.assertEqual(past.info["minimal"], not any(c["size"] < chosen["size"] for c in und))
            for x_, f in past.eliminated.items():                       # whatever is chosen is still checked
                self.assertEqual(row_check(rows, names, x_, over(f, names))["values"], 0)
            # one elimination order has no other set to go on with: it is refused, need = its limit
            spec = dict(columns=variables, mode="order", order=["z", "s", "w"])
            lo = 1
            while True:
                try:
                    me.select(scope, certs, budget=Budget(**{resource: lo}), **spec)
                    break
                except AlgebraBudgetError as err:
                    self.assertEqual(err.resource, resource)
                    lo += 1
            with self.assertRaises(AlgebraBudgetError) as e:
                me.select(scope, certs, budget=Budget(**{resource: lo - 1}), **spec)
            self.assertEqual((e.exception.resource, e.exception.need, e.exception.limit), (resource, lo, lo - 1))
        # forms: the solves are the alternatives, counted before any solve
        q = mg.poly({(0, 0, 1, 1, 0, 0): 1, (0, 0, 0, 0, 1, 0): 2}, names)          # z s + 2 w
        f = me.forms(scope, sel, q)
        alt = f.info["candidates"] - 1
        self.assertEqual(f.info["solves"], alt)
        me.forms(scope, sel, q, budget=Budget(iterations=alt))
        with self.assertRaises(AlgebraBudgetError) as e:
            me.forms(scope, sel, q, budget=Budget(iterations=alt - 1))
        self.assertEqual((e.exception.resource, e.exception.need, e.exception.limit), ("iterations", alt, alt - 1))
        # an expansion past the degree or support budget is rejected with its refusal at the limit + 1; the direct
        # form stays chosen. Each candidate's peak: the smallest limit admitting it (the division's intermediates, not
        # only the 13 terms of (x^2 - y)^12)
        big = mg.poly({(0, 0, 0, 0, 12, 0): 1}, names)                              # w^12
        free = me.forms(scope, sel, big)
        self.assertTrue(all(c["status"] != "rejected" for c in free.candidates))
        self.assertEqual(len(free.candidates[1]["polynomial"]), 13)
        for resource in ("support", "degree"):
            peaks = {}
            for limit in range(1, 80):
                fb = me.forms(scope, sel, big, budget=Budget(**{resource: limit}))
                for c in fb.candidates[1:]:
                    if c["status"] != "rejected" and c["index"] not in peaks:
                        peaks[c["index"]] = limit
                if len(peaks) == len(free.candidates) - 1:
                    break
            self.assertEqual(len(peaks), len(free.candidates) - 1)
            for idx, peak in peaks.items():
                fb = me.forms(scope, sel, big, budget=Budget(**{resource: peak - 1}))
                c = fb.candidates[idx]
                self.assertEqual((c["status"], c["reason"]), ("rejected", "budget"))
                self.assertEqual(c["refusal"], dict(resource=resource, need=peak, limit=peak - 1))
                self.assertEqual(fb.candidates[0]["status"], "chosen" if all(
                    k["status"] == "rejected" for k in fb.candidates[1:]) else fb.candidates[0]["status"])
            low = me.forms(scope, sel, big, budget=Budget(**{resource: min(peaks.values()) - 1}))
            self.assertEqual((low.chosen, low.best["kind"]), (0, "direct"))      # every expansion refused: direct kept
        self.assertGreater(max(peaks.values()), 12)                                 # degree 24: x^24 at the peak

    # ---------------------------------------------------------- a set the exact arithmetic cannot decide
    def test_geom002_sets_past_int64_are_undecided_and_the_search_goes_on(self):
        n = 300
        x = [int(v) for v in self.rng.integers(-20, 21, n)]
        y = [int(v) for v in self.rng.integers(-20, 21, n)]
        U = {(3, 0): 3, (2, 1): -2, (1, 2): 3, (0, 3): -1, (2, 0): 2, (1, 1): -3, (0, 2): 1, (1, 0): 1, (0, 1): -2, (0, 0): 3}
        V = {(3, 0): -1, (2, 1): 2, (0, 3): -1, (2, 0): 3, (1, 1): 1, (0, 2): -2, (1, 0): -3, (0, 1): 1, (0, 0): -1}
        ev = lambda P, a, b: sum(c * a ** i * b ** j for (i, j), c in P.items())
        data = {"x": x, "y": y, "u": [ev(U, a, b) for a, b in zip(x, y)], "v": [ev(V, a, b) for a, b in zip(x, y)]}
        db = make(data)
        names = list(data)
        scope = mbk.Scope.from_table(db["t"], names)
        rows = [dict(zip(names, vals)) for vals in zip(*[data[c] for c in names])]

        def rel(name, P):
            out = {(1, 0, 0): 1}
            for (i, j), c in P.items():
                out[(0, i, j)] = -c
            return cert(scope, out, [name, "x", "y"])
        certs = [rel("u", U), rel("v", V)]
        sel = me.select(scope, certs)
        cand = {tuple(c["generators"]): c for c in sel.candidates}
        # eliminating x and y from two dense cubics (the empty set, {u}, {v}, {u, v}) passes int64 in the completion:
        # those sets are undecided, never guessed feasible or not, and the search goes on to {x, y}
        und = [g for g, c in cand.items() if c["reason"] == "unclaimable"]
        self.assertEqual(sorted(und), sorted([(), ("u",), ("v",), ("u", "v")]))
        self.assertTrue(all(cand[g]["status"] == "rejected" and cand[g]["refusal"] is None for g in und))
        self.assertEqual(sel.generators, ["x", "y"])
        self.assertEqual((sel.info["undecided"], sel.info["undecided_size"], sel.info["minimal"]), (4, 0, False))
        self.assertEqual({g: cand[g]["reason"] for g in [("x",), ("y",)]}, {("x",): "not reconstructible",
                                                                         ("y",): "not reconstructible"})
        want = {k: {(i, j, 0, 0): Fraction(c) for (i, j), c in P.items()} for k, P in (("u", U), ("v", V))}
        got = {k: over(f, names) for k, f in sel.eliminated.items()}
        self.assertEqual(got, want)                                     # u = U(x, y), v = V(x, y) exactly
        checks, gpu = sel.check(), sel.check(backend="cuda")
        for k in ("u", "v"):
            o = row_check(rows, names, k, got[k])
            self.assertEqual((checks[k]["values"], checks[k]["compared"], checks[k]["holds"]), (0, o["compared"], True))
            self.assertEqual({a: b for a, b in checks[k].items() if a not in ("backend", "device")},
                             {a: b for a, b in gpu[k].items() if a not in ("backend", "device")})
        # bounded below the decidable size: nothing feasible, G = V, and no claim of minimality
        low = me.select(scope, certs, max_size=1)
        self.assertEqual((low.generators, low.info["minimal"], low.candidates[-1]["reason"]),
                         (names, False, "direct"))
        # one elimination order with the defined columns first decides it in one solve
        order = me.select(scope, certs, mode="order", order=["u", "v", "x", "y"])
        self.assertEqual((order.generators, order.info["solves"]), (["x", "y"], 1))
        self.assertEqual({k: over(f, names) for k, f in order.eliminated.items()}, got)

    # ---------------------------------------------------------- acceptance 2: the direct candidate when expansion loses
    def test_geom002_retains_the_direct_candidate_when_expansion_loses(self):
        n = 500
        x = [int(v) for v in self.rng.integers(-2000, 2001, n)]
        y = [int(v) for v in self.rng.integers(-2000, 2001, n)]
        data = {"x": x, "y": y, "w": [a + 1 for a in x], "z": [a * b for a, b in zip(x, y)], "s": [a + b for a, b in zip(x, y)]}
        db = make(data)
        names = list(data)
        scope = mbk.Scope.from_table(db["t"], names)
        certs = [cert(scope, {(1, 0): 1, (0, 1): -1, (0, 0): -1}, ["w", "x"]),
                 cert(scope, {(1, 0, 0): 1, (0, 1, 1): -1}, ["z", "x", "y"]),
                 cert(scope, {(1, 0, 0): 1, (0, 1, 0): -1, (0, 0, 1): -1}, ["s", "x", "y"])]
        sel = me.select(scope, certs)
        self.assertEqual(sel.generators, ["x", "y"])
        N = scope.selected
        widths = {c: width_of(data[c]) for c in names}

        def oracle_cost(p):
            D = 1
            for c in p.terms.values():
                D = D * c.denominator // math.gcd(D, c.denominator)
            ints = {e: int(c * D) for e, c in p.terms.items()}
            used = p.used
            vals = [abs(D * evaluate(p.aligned(names), [data[c][r] for c in names])) for r in range(N)]
            actual = max(int(v).bit_length() for v in vals)
            mults = sum(max(sum(e) - 1, 0) + (abs(k) != 1 and sum(e) > 0) for e, k in ints.items())
            words = max(1, (actual + 1 + 63) // 64)
            scan = (N + 63) // 64 * sum(widths[c] for c in used)
            work = N * (len(ints) + mults) * words
            return dict(terms=len(ints), multiplications=mults, columns=len(used), actual_bits=actual, words=words,
                        scan=scan, work=work, total=scan + work)

        # w^6: its local form (x + 1)^6 is a term explosion; the direct form is kept
        q = mg.poly({(0, 0, 6, 0, 0): 1}, names)
        f = me.forms(scope, sel, q)
        cands = f.candidates
        self.assertEqual([c["kind"] for c in cands], ["direct", "local", "normal"])
        self.assertEqual(cands[1]["polynomial"], mg.poly({(k,): math.comb(6, k) for k in range(7)}, ["x"]))
        for c in cands:
            o = oracle_cost(c["polynomial"])
            self.assertEqual({k: c["cost"][k] for k in o}, o, c["kind"])
            self.assertTrue(c["check"]["holds"])
        self.assertEqual((f.chosen, f.best["kind"], f.best["reason"]), (0, "direct", "cheapest"))
        self.assertGreater(cands[1]["cost"]["total"], cands[0]["cost"]["total"])
        self.assertEqual(cands[1]["reason"], "costlier")
        # never the canonical normal form by assumption: x y - z + 3 w is 3w after the local elimination of z alone
        # (one column, two operations), cheaper than the direct form and than the normal form 3x + 3
        h = mg.poly({(1, 1, 0, 0, 0): 1, (0, 0, 0, 1, 0): -1, (0, 0, 1, 0, 0): 3}, names)
        fh = me.forms(scope, sel, h)
        best = fh.best
        self.assertEqual((best["kind"], best["eliminated"]), ("local", ["z"]))
        self.assertEqual(best["polynomial"], mg.poly({(1,): 3}, ["w"]))
        normal = [c for c in fh.candidates if c["kind"] == "normal"][0]
        self.assertEqual(normal["polynomial"], mg.poly({(1,): 3, (0,): 3}, ["x"]))
        self.assertLess(best["cost"]["total"], normal["cost"]["total"])
        self.assertLess(best["cost"]["total"], fh.candidates[0]["cost"]["total"])
        for c in fh.candidates:
            self.assertEqual({k: c["cost"][k] for k in oracle_cost(c["polynomial"])}, oracle_cost(c["polynomial"]))
        # z itself against its reconstruction x y: the stored column is cheaper (a cost regression of the expansion)
        g = mg.poly({(0, 0, 0, 1, 0): 1}, names)                                    # z
        fg = me.forms(scope, sel, g)
        self.assertEqual(fg.chosen, 0)
        self.assertTrue(all(c["cost"]["total"] >= fg.candidates[0]["cost"]["total"] for c in fg.candidates))
        # every priority keeps the direct form for the term explosion (ties on scan and bits go to the total)
        for priority in ("total", "scan", "work", "bits"):
            self.assertEqual(me.forms(scope, sel, q, priority=priority).best["kind"], "direct", priority)
        # choose() over given candidates: the first is kept unless strictly cheaper
        idx, costs = me.choose(scope, [q, cands[1]["polynomial"]])
        self.assertEqual(idx, 0)
        self.assertEqual(me.choose(scope, [cands[1]["polynomial"], q])[0], 1)
        self.assertEqual(me.cost(scope, q), {k: v for k, v in costs[0].items()})
        frame = pd.DataFrame(data)
        con = duckdb.connect()
        con.register("frame", frame)
        self.assertEqual(con.execute("SELECT count(*) FROM frame WHERE w <> x + 1 OR z <> x * y").fetchone()[0], 0)

    # ---------------------------------------------------------- checks: CPU equal to CUDA, the first mismatch
    def test_geom002_batched_checks_cpu_equal_cuda_with_the_first_mismatch_independent_of_thread_order(self):
        n = 9000                                                         # several CUDA row slices
        x = [int(v) for v in self.rng.integers(-10 ** 6, 10 ** 6, n)]
        y = [int(v) for v in self.rng.integers(-10 ** 6, 10 ** 6, n)]
        z = [a * b for a, b in zip(x, y)]
        planted = [8901, 4097, 7777]
        for r in planted:
            z[r] += 1
        nul = set(int(i) for i in self.rng.choice(n, 30, replace=False))
        Z = [None if i in nul else v for i, v in enumerate(z)]
        db = make({"x": x, "y": y, "z": Z})
        names = ["x", "y", "z"]
        scope = mbk.Scope.from_table(db["t"], names)
        xy = mg.poly({(1, 1): 1}, ["x", "y"])
        pairs = [("z", xy), (mg.poly({(0, 0, 2): 1}, names), mg.poly({(2, 2, 0): 1}, names)),
                 ("x", mg.poly({(1,): 1}, ["x"])), (mg.poly({(1, 0, 0): Fraction(1, 2)}, names), mg.poly({(1,): Fraction(1, 2)}, ["x"])),
                 ("y", mg.poly({(1,): 1, (0,): 1}, ["y"]))]
        cpu = me.check(scope, pairs)
        gpu = me.check(scope, pairs, backend="cuda")
        strip = lambda d: {k: v for k, v in d.items() if k not in ("backend", "device")}
        self.assertEqual([strip(c) for c in cpu], [strip(c) for c in gpu])
        rows = [dict(zip(names, v)) for v in zip(x, y, Z)]
        o = row_check(rows, names, "z", over(xy, names))
        self.assertEqual((cpu[0]["values"], cpu[0]["first_value"], cpu[0]["first"], cpu[0]["null_extra"]),
                         (o["values"], min(r for r in planted if r not in nul), o["first"], o["null_extra"]))
        self.assertEqual(cpu[2]["holds"], True)
        self.assertEqual((cpu[3]["holds"], cpu[3]["values"]), (True, 0))
        self.assertEqual((cpu[4]["values"], cpu[4]["above"], cpu[4]["first"]), (n, 1, 0))
        need = me.check_need(scope, pairs, backend="cuda")
        me.check(scope, pairs, backend="cuda", budget=Budget(bytes=need["bytes"]))
        with self.assertRaises(AlgebraBudgetError) as e:
            me.check(scope, pairs, backend="cuda", budget=Budget(bytes=need["bytes"] - 1))
        self.assertEqual(e.exception.resource, "bytes")
        with self.assertRaises(UnsupportedError):
            me.check(scope, [("z", xy)] * 257)
        # pairs proven for one scope refuse a view whose extremes they do not cover
        bigger = make({"x": [v * 10 ** 6 for v in x], "y": y, "z": z}, "b")
        sb = mbk.Scope.from_table(bigger["b"], names)
        self.assertTrue(me.check(sb, [("z", xy)])[0]["values"] > 0)       # a new batch proves its own bound

    # ---------------------------------------------------------- certificates and local forms
    def test_geom002_reconstruction_and_form_certificates_are_fnd007_derived(self):
        db, data, names, scope, certs, full, rows = self.planted(n=250)
        sel = me.select(scope, certs, columns=["x", "y", "z", "s", "w"])
        for x in sel.info["eliminated"]:
            c = sel.certificate(x)
            self.assertEqual((c.kind, c.info["premises"], c.info["depth"]), ("snapshot", 3, 1))
            self.assertEqual(c.authorize_exact(scope).nulls, "skip")
            claim = c.claim
            f = over(sel.form(x), names)
            terms = {tuple(dict(zip([names[k] for k in claim["columns"]], e)).get(v, 0) for v in names): k
                     for e, k in claim["terms"].items()}
            ratio = None
            for e, k in clean(add(unit(names, x), f, -1)).items():         # P = D (x - f)
                r = Fraction(terms.get(e, 0)) / k
                ratio = r if ratio is None else ratio
                self.assertEqual(r, ratio)
            self.assertEqual(len(terms), len(clean(add(unit(names, x), f, -1))))
        q = mg.poly({(0, 0, 1, 1, 0, 0): 1, (0, 0, 0, 0, 2, 0): -1, (1, 0, 0, 0, 0, 0): 3}, names)   # z s - w^2 + 3x
        f = me.forms(scope, sel, q)
        for c in f.candidates[1:]:
            if c["status"] == "rejected":
                continue
            cert_ = f.certificate(c["index"])
            self.assertEqual(cert_.kind, "snapshot")
            self.assertEqual(cert_.info["depth"], c["depth"])
            self.assertTrue(cert_.authorize_exact(scope))
            # D (q - p) is the claim, and p agrees with q on every row (values and NULL mask)
            p = over(c["polynomial"], names)
            for r in rows:
                pq = [r[n_] for n_ in names]
                usedq = [n_ for n_ in q.used]
                usedp = c["polynomial"].used
                nq = any(r[u] is None for u in usedq)
                np_ = any(r[u] is None for u in usedp)
                self.assertEqual(nq, np_)
                if not nq:
                    point = [v if v is not None else 0 for v in pq]
                    self.assertEqual(evaluate(over(q, names), point), evaluate(p, point))
        with self.assertRaises(ValueError):
            f.certificate(0)                                                # the direct form has no derivation
        # a mutation makes the certificates stale and the scope refuses new checks
        db["t"].flip(0, 2)
        now = mbk.Scope.from_table(db["t"], names)
        with self.assertRaises((StaleStateError, mc.NotAuthorizedError)):
            sel.certificate("z").authorize_exact(now)
        with self.assertRaises(StaleStateError):
            sel.check()

    # ---------------------------------------------------------- DISC-002 relations end to end
    def test_geom002_selects_generators_from_disc002_identities(self):
        n = 400
        x = [int(v) for v in self.rng.integers(-30, 31, n)]
        y = [int(v) for v in self.rng.integers(-25, 40, n)]
        v = {"a": [3 * p - 7 for p in x], "s": [p + q for p, q in zip(x, y)], "z": [p * q for p, q in zip(x, y)],
             "x": x, "y": y}
        db = make(v)
        names = list(v)
        scope = mbk.Scope.from_table(db["t"], names)
        found = mi.discover(scope, names, degree=2)
        rels = [i["certificate"] for i in found.discoveries()]
        sel = me.select(scope, rels)
        self.assertEqual(sel.info["minimum"], 2)
        rows = [dict(zip(names, t)) for t in zip(*[v[c] for c in names])]
        for col, f in sel.eliminated.items():
            self.assertTrue(set(f.used) <= set(sel.generators))
            o = row_check(rows, names, col, over(f, names))
            self.assertEqual((o["values"], o["null_lost"], o["null_extra"]), (0, 0, 0))
        self.assertTrue(all(c["holds"] for c in sel.check(backend="cuda").values()))
        # the canonical normal form is one candidate: a s - 3 z in the generators
        q = mg.poly({(1, 1, 0, 0, 0): 1, (0, 0, 1, 0, 0): -3}, names)
        f = me.forms(scope, sel, q)
        self.assertTrue(any(c["kind"] == "normal" for c in f.candidates))
        for c in f.candidates:
            if c["status"] != "rejected":
                self.assertTrue(c["check"]["holds"])
        con = duckdb.connect()
        con.register("frame", pd.DataFrame(v))
        self.assertEqual(con.execute("SELECT count(*) FROM frame WHERE a <> 3 * x - 7 OR z <> x * y").fetchone()[0], 0)

    # ---------------------------------------------------------- validation and limits
    def test_geom002_validation_and_representation_limits(self):
        db, data, names, scope, certs, full, rows = self.planted(n=80)
        with self.assertRaises(ValueError):
            me.select(scope, [])
        with self.assertRaises(UnsupportedError):
            me.select(scope, certs * 6)
        with self.assertRaises(TypeError):
            me.select(scope, [mc.polynomial({(1, 0): 1}, ["x", "y"])])
        with self.assertRaises(ValueError):
            me.select(scope, certs, mode="greedy")
        with self.assertRaises(ValueError):
            me.select(scope, certs, columns=["nope"])
        with self.assertRaises(ValueError):
            me.select(scope, certs, mode="order", order=["u"], columns=["x", "y"])   # not a variable
        other = mbk.Scope.from_table(db["t"], ["y", "x", "z", "s", "w", "u"])
        with self.assertRaises(DomainError):
            me.select(other, certs)
        sel = me.select(scope, certs)
        with self.assertRaises(ValueError):
            sel.form("x")                                                   # a generator
        with self.assertRaises(TypeError):
            me.forms(scope, object(), mg.poly({(1,): 1}, ["x"]))
        with self.assertRaises(ValueError):
            me.forms(scope, sel, mg.poly({(1,): 1}, ["x"]), priority="cheapest")
        db2, data2, names2, scope2, certs2, full2, rows2 = self.planted(n=80, nulls=False)
        mixed = [certs2[0], cert(scope2, {(1, 0, 0): 1, (0, 1, 0): -1, (0, 0, 1): -1}, ["s", "x", "y"], nulls="forbid")]
        with self.assertRaises(DomainError):
            me.select(scope2, mixed)                                         # two NULL policies: GEOM-001 refuses

    def test_geom002_import_torbles_does_not_load_the_module(self):
        code = "import sys, torbles; print('torbles.math_elim' in sys.modules)"
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=dict(os.environ),
                             check=True).stdout.strip()
        self.assertEqual(out, "False")


if __name__ == "__main__":
    unittest.main()
