# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""BOOL-002: influences and predicate interactions under the uniform measure of the 2^k assignments and the
empirical measure of stored rows (torbles/math_influence.c, math_influence_gpu.cu, math_influence.py).

Oracles, independent of the implementation: exhaustive truth tables written here with Python integers as bit sets
(dual-rail TRUE / FALSE sets, Kleene's connectives; a cofactor by substituting a constant set for a condition's
leaf, a difference by combining the cofactor sets point by point), the Walsh-Fourier identity Inf_i = sum over S
containing i of fhat(S)^2 for two-valued predicates (Python Fractions over every assignment), and DuckDB SQL over the
typed source table (the cofactors as SQL with TRUE / FALSE substituted, `f1 <> f0` counted with SQL's three-valued
WHERE). Every native result is also computed on the CUDA path (emulated here with TORBLES_EMU_SCRAMBLE=1, a scrambled
thread order) and compared for equality. Seeds are fixed.
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
from torbles import math_handles as mh
from torbles import math_influence as mi
from torbles.math_budgets import Budget
from torbles.math_domains import DomainError
from torbles.math_errors import StaleStateError, UnsupportedError
from torbles.mathematics import AlgebraBudgetError

CONST, VAR, NOT, AND, OR, XOR = mb.CONST, mb.VAR, mb.NOT, mb.AND, mb.OR, mb.XOR


# ---------------------------------------------------------------------- oracles
def tables(expr, leaves, full):
    """(TRUE set, FALSE set) of expr over the points (Python ints as bit sets); leaves[i] = condition i's sets"""
    memo = {}

    def go(e):
        if id(e) in memo:
            return memo[id(e)]
        if e.op == CONST:
            r = ((full, 0), (0, full), (0, 0))[[1, 0, 2].index(e.a)]
        elif e.op == VAR:
            r = leaves[e.a]
        elif e.op == NOT:
            t, f = go(e.a)
            r = (f, t)
        else:
            (t1, f1), (t2, f2) = go(e.a), go(e.b)
            if e.op == AND:
                r = (t1 & t2, f1 | f2)
            elif e.op == OR:
                r = (t1 | t2, f1 & f2)
            else:
                r = ((t1 & f2) | (f1 & t2), (t1 & t2) | (f1 & f2))
        memo[id(e)] = r
        return r
    return go(expr)


def assignment_leaves(k):
    n = 1 << k
    return [(sum(1 << a for a in range(n) if (a >> i) & 1), sum(1 << a for a in range(n) if not (a >> i) & 1))
            for i in range(k)], n


def oracle(expr, leaves, n, sel, S):
    """the counts of the differences over S at the points of the bit set sel"""
    full = (1 << n) - 1
    m = len(S)
    known, parity, cof = sel, 0, []
    for t in range(1 << m):
        sub = list(leaves)
        for j, c in enumerate(S):
            sub[c] = (full, 0) if (t >> j) & 1 else (0, full)
        T, F = tables(expr, sub, full)
        cof.append(T)
        known &= T | F
        parity ^= T
    sens = parity & known
    D = [0] * n
    for t, T in enumerate(cof):
        sign = -1 if (m - bin(t).count("1")) % 2 else 1
        for p in range(n):
            if (T >> p) & 1:
                D[p] += sign
    kp = [p for p in range(n) if (known >> p) & 1]
    return dict(sensitive=sens.bit_count(), known=known.bit_count(), unknown=(sel & ~known).bit_count(),
                increase=sum(1 for p in kp if D[p] > 0), decrease=sum(1 for p in kp if D[p] < 0),
                signed_sum=sum(D[p] for p in kp), sens_set=sens, known_set=known)


def random_expr(rng, k, size, share=0.5, unknown=True):
    pool = [mb.var(i) for i in range(k)] + [mb.TRUE, mb.FALSE] + ([mb.UNKNOWN] if unknown else [])
    built = []
    for _ in range(size):
        cand = built[-6:] if built and rng.random() < share else pool + built[-3:]
        op = rng.choice([AND, OR, XOR, NOT, AND, OR])
        a = rng.choice(cand)
        if op == NOT:
            e = ~a
        else:
            b = rng.choice(pool + built)
            e = a & b if op == AND else a | b if op == OR else a ^ b
        built.append(e)
    top = built[-1]
    for i in range(k):                                     # every condition is read
        top = top ^ mb.var(i) if rng.random() < 0.3 else top | (mb.var(i) & built[len(built) // 2])
    return top


def obj(values):
    return pd.Series(values, dtype=object)


def expr_sql(expr, conds_sql):
    memo = {}

    def go(e):
        if id(e) in memo:
            return memo[id(e)]
        if e.op == CONST:
            s = ("FALSE", "TRUE", "CAST(NULL AS BOOLEAN)")[e.a]
        elif e.op == VAR:
            s = conds_sql[e.a]
        elif e.op == NOT:
            s = f"(NOT {go(e.a)})"
        elif e.op == AND:
            s = f"({go(e.a)} AND {go(e.b)})"
        elif e.op == OR:
            s = f"({go(e.a)} OR {go(e.b)})"
        else:
            s = f"({go(e.a)} <> {go(e.b)})"
        memo[id(e)] = s
        return s
    return go(expr)


DDL = "CREATE TABLE t (x BIGINT, y BIGINT, z BIGINT, w BIGINT)"
SPECS = [("x", ">", 5), ("y", "<=", -7), ("z", "<>", 3), ("w", ">=", 0), ("x", "<", -20), ("y", ">", 25),
         ("z", "<", 0), ("w", "<=", 40)]


def dataset(rng, n, nulls=0.15):
    cols = {}
    for c in ("x", "y", "z", "w"):
        v = [int(a) for a in rng.integers(-50, 51, n)]
        for i in np.flatnonzero(rng.random(n) < nulls):
            v[int(i)] = None
        cols[c] = obj(v)
    return torbles.from_ddl(DDL, {"t": pd.DataFrame(cols)})


def sql(spec):
    return f'("{spec[0]}" {spec[1]} {spec[2]})'


class Influence(unittest.TestCase):
    def both(self, fn, *args, **kw):
        """the CPU result, checked equal to the (emulated, scrambled) CUDA result"""
        cpu = fn(*args, **kw)
        gpu = fn(*args, backend="cuda", **kw)
        strip = lambda r: {k: v for k, v in r.items() if k not in ("backend", "device")}
        self.assertEqual(strip(cpu), strip(gpu))
        self.assertIn(gpu["backend"], ("cuda", "cuda-emulated"))
        return cpu

    def check_first(self, got, expr, leaves, n, sel, k, what):
        full_sets = []
        for i in range(k):
            want = oracle(expr, leaves, n, sel, [i])
            g = got["influences"][i]
            self.assertEqual((g["condition"], g["sensitive"], g["known"], g["unknown"], g["increase"], g["decrease"],
                              g["signed_sum"]),
                             (i, want["sensitive"], want["known"], want["unknown"], want["increase"], want["decrease"],
                              want["signed_sum"]), (what, i))
            self.assertEqual(g["insensitive"] + g["sensitive"] + g["unknown"], sel.bit_count())
            self.assertEqual(g["influence"], Fraction(want["sensitive"], sel.bit_count() or 1) if sel else 0)
            full_sets.append(want)
        total = sum(w["sensitive"] for w in full_sets)
        self.assertEqual(got["total_sensitive"], total)
        self.assertEqual(got["total_influence"], Fraction(total, sel.bit_count()) if sel else 0)
        allk = sel
        for w in full_sets:
            allk &= w["known_set"]
        bins = {b: 0 for b in range(k + 1)}
        for p in range(n):
            if (allk >> p) & 1:
                bins[sum((w["sens_set"] >> p) & 1 for w in full_sets)] += 1
        self.assertEqual(got["sensitivity"], bins, what)
        self.assertEqual(got["sensitivity_known"], allk.bit_count())
        self.assertEqual(got["max_sensitivity"], max([b for b, c in bins.items() if c] or [0]))

    # ------------------------------------------------------------------ uniform weights
    def test_bool002_influences_match_exhaustive_uniform_calculations(self):
        rng = random.Random(3101)
        for k in (1, 2, 3, 5, 8, 10):
            leaves, n = assignment_leaves(k)
            for trial in range(3):
                expr = random_expr(rng, k, size=2 + 3 * k, share=(0.0, 0.5, 0.8)[trial], unknown=trial != 1)
                prog = mb.Program(expr, k)
                got = self.both(mi.uniform, prog, order=min(3, k))
                self.assertEqual((got["measure"], got["points"]), ("uniform", n))
                self.check_first(got, expr, leaves, n, (1 << n) - 1, k, (k, trial))
                sets = [tuple(S) for m in range(2, min(3, k) + 1) for S in
                        sorted(itertools.combinations(range(k), m), key=lambda T: sum(1 << i for i in T))]
                self.assertEqual([t["conditions"] for t in got["interactions"]], sets)
                for t in got["interactions"]:
                    want = oracle(expr, leaves, n, (1 << n) - 1, list(t["conditions"]))
                    self.assertEqual((t["sensitive"], t["unknown"], t["increase"], t["decrease"], t["signed_sum"]),
                                     (want["sensitive"], want["unknown"], want["increase"], want["decrease"],
                                      want["signed_sum"]), (k, trial, t["conditions"]))
                    self.assertEqual(t["measure"], Fraction(want["sensitive"], n))
        # sixteen conditions: every first-order influence and a few pairs over the 65536 assignments
        k = 16
        leaves, n = assignment_leaves(k)
        expr = random_expr(rng, k, size=40, share=0.7, unknown=False)
        prog = mb.Program(expr, k)
        got = self.both(mi.uniform, prog, terms=[[0, 15], [3, 7]])
        self.check_first(got, expr, leaves, n, (1 << n) - 1, k, 16)
        for t, S in zip(got["interactions"], ([0, 15], [3, 7])):
            self.assertEqual(t["sensitive"], oracle(expr, leaves, n, (1 << n) - 1, S)["sensitive"])

    def test_bool002_uniform_influence_is_the_fourier_weight_of_two_valued_predicates(self):
        rng = random.Random(3102)
        for k in (2, 4, 6):
            leaves, n = assignment_leaves(k)
            expr = random_expr(rng, k, size=3 * k, share=0.5, unknown=False)
            T, F = tables(expr, leaves, (1 << n) - 1)
            self.assertEqual(T | F, (1 << n) - 1)                # two-valued: no UNKNOWN constant
            g = [(-1 if (T >> a) & 1 else 1) for a in range(n)]  # (-1)^f
            fhat = [Fraction(sum(g[a] * (-1 if bin(S & a).count("1") % 2 else 1) for a in range(n)), n)
                    for S in range(n)]
            got = mi.uniform(mb.Program(expr, k))
            for i in range(k):
                self.assertEqual(got["influences"][i]["influence"],
                                 sum(fhat[S] ** 2 for S in range(n) if (S >> i) & 1), (k, i))
            self.assertEqual(got["total_influence"], sum(bin(S).count("1") * fhat[S] ** 2 for S in range(n)))

    # ------------------------------------------------------------------ empirical weights
    def test_bool002_empirical_influences_match_rows_and_duckdb(self):
        nrng = np.random.default_rng(3103)
        prng = random.Random(3103)
        n = 900
        db = dataset(nrng, n, nulls=0.2)
        sel = nrng.random(n) < 0.75
        scope = mbk.Scope.from_table(db["t"], ["x", "y", "z", "w"], rows=sel)
        k = 6
        specs = SPECS[:k]
        csql = [sql(s) for s in specs]
        rows = mb.Masks.from_scope(scope, [mb.condition(*s) for s in specs])
        db.con.register("sel_rows", pd.DataFrame({"rid": np.flatnonzero(sel).astype(np.int64)}))
        where = "rowid IN (SELECT rid FROM sel_rows)"
        leaves = []                                              # each condition's TRUE / FALSE rows by DuckDB
        for s in csql:
            tt = [r for r, in db.con.execute(f"SELECT rowid FROM t WHERE {where} AND {s}").fetchall()]
            ff = [r for r, in db.con.execute(f"SELECT rowid FROM t WHERE {where} AND NOT {s}").fetchall()]
            leaves.append((sum(1 << r for r in tt), sum(1 << r for r in ff)))
        selbits = sum(1 << int(r) for r in np.flatnonzero(sel))
        for trial in range(3):
            expr = random_expr(prng, k, size=12, share=0.5)
            prog = mb.Program(expr, k)
            got = self.both(mi.empirical, prog, rows, order=2, terms=[[0, 2, 4]])
            self.assertEqual((got["measure"], got["points"]), ("empirical", int(sel.sum())))
            self.check_first(got, expr, leaves, n, selbits, k, trial)
            for t in got["interactions"]:
                want = oracle(expr, leaves, n, selbits, list(t["conditions"]))
                self.assertEqual((t["sensitive"], t["unknown"], t["signed_sum"], t["increase"], t["decrease"]),
                                 (want["sensitive"], want["unknown"], want["signed_sum"], want["increase"],
                                  want["decrease"]), (trial, t["conditions"]))
            # the first-order influences once more, wholly in SQL: f|x_i=TRUE <> f|x_i=FALSE (NULL when one is)
            for i in range(k):
                f1 = expr_sql(expr, [("TRUE" if j == i else c) for j, c in enumerate(csql)])
                f0 = expr_sql(expr, [("FALSE" if j == i else c) for j, c in enumerate(csql)])
                s, u = db.con.execute(f"SELECT COUNT(*) FILTER (WHERE {f1} <> {f0}), COUNT(*) FILTER (WHERE "
                                      f"({f1} <> {f0}) IS NULL) FROM t WHERE {where}").fetchall()[0]
                g = got["influences"][i]
                self.assertEqual((g["sensitive"], g["unknown"]), (int(s), int(u)), (trial, i))
            fs = expr_sql(expr, csql)
            ft = db.con.execute(f"SELECT COUNT(*) FILTER (WHERE {fs}), COUNT(*) FILTER (WHERE NOT {fs}), COUNT(*) "
                                f"FILTER (WHERE ({fs}) IS NULL) FROM t WHERE {where}").fetchall()[0]
            self.assertEqual(tuple(got["f"].values()), tuple(int(v) for v in ft))

    def test_bool002_measures_are_separate_and_never_confused_or_called_causal(self):
        x = mb.variables(3)
        expr = (x[0] & x[1]) | (x[2] ^ x[0])
        prog = mb.Program(expr, 3)
        # rows holding every assignment of the three conditions exactly once: the two measures coincide
        combos = list(itertools.product((0, 1), repeat=3))

        def rows_of(table):
            frame = pd.DataFrame({c: obj([r[j] for r in table]) for j, c in enumerate("abc")})
            db = torbles.from_ddl("CREATE TABLE t (a BIGINT, b BIGINT, c BIGINT)", {"t": frame})
            scope = mbk.Scope.from_table(db["t"], ["a", "b", "c"])
            return mb.Masks.from_scope(scope, [mb.condition(c, "=", 1) for c in "abc"])
        once = rows_of([(a, b, c) for c, b, a in combos])
        r = mi.influence(prog, rows=once, order=2)
        self.assertEqual(set(r), {"uniform", "empirical", "interpretation", "causal"})
        u, e = r["uniform"], r["empirical"]
        self.assertEqual((u["measure"], e["measure"]), ("uniform", "empirical"))
        self.assertEqual([i["sensitive"] for i in u["influences"]], [i["sensitive"] for i in e["influences"]])
        self.assertEqual(u["interactions"], e["interactions"])
        # stored rows that repeat some assignments and miss others: the empirical measure is theirs alone
        skew = rows_of([(1, 1, 0)] * 5 + [(0, 0, 1)] * 2 + [(1, 0, 0)])
        r = mi.influence(prog, rows=skew)
        u, e = r["uniform"], r["empirical"]
        self.assertEqual((u["points"], e["points"]), (8, 8))
        self.assertNotEqual([i["influence"] for i in u["influences"]], [i["influence"] for i in e["influences"]])
        leaves = [(sum(1 << p for p, row in enumerate([(1, 1, 0)] * 5 + [(0, 0, 1)] * 2 + [(1, 0, 0)]) if row[i]),
                   sum(1 << p for p, row in enumerate([(1, 1, 0)] * 5 + [(0, 0, 1)] * 2 + [(1, 0, 0)]) if not row[i]))
                  for i in range(3)]
        for i in range(3):
            self.assertEqual(e["influences"][i]["sensitive"], oracle(expr, leaves, 8, 255, [i])["sensitive"])
        for res in (u, e, r):                                    # labelled: coordinate sensitivity, never causal
            self.assertIs(res["causal"], False)
            self.assertIn("not a causal effect", res["interpretation"])
            self.assertIn("coordinate sensitivity", res["interpretation"])
        self.assertIn("2^k assignments", u["description"])
        self.assertIn("stored rows", e["description"])
        with self.assertRaises(DomainError):                    # the assignments are not stored rows
            mi.empirical(prog, mb.Masks.assignments(3))
        with self.assertRaises(DomainError):
            mi.influence(prog, rows=mb.Masks.assignments(3))
        with self.assertRaises(ValueError):                     # an empirical measure needs rows
            mi.influence(prog, measures=("empirical",))
        with self.assertRaises(DomainError):
            mi.empirical(mb.Program(x[0], 4), skew)
        only = mi.influence(prog, rows=skew, measures=("empirical",))
        self.assertNotIn("uniform", only)

    def test_bool002_sixty_four_conditions_empty_selection_and_k0(self):
        vals = list(range(-30, 30)) + [None] * 4
        frame = pd.DataFrame({"x": obj(vals)})
        db = torbles.from_ddl("CREATE TABLE t (x BIGINT)", {"t": frame})
        scope = mbk.Scope.from_table(db["t"], ["x"])
        rows = mb.Masks.from_scope(scope, [mb.condition("x", ">", i - 32) for i in range(64)])
        x = mb.variables(64)
        f = x[0]
        for i in range(1, 64):
            f = (f ^ x[i]) if i % 3 else (f & x[i]) | x[i - 1]
        prog = mb.Program(f, 64)
        n = len(vals)
        leaves = [(sum(1 << r for r, v in enumerate(vals) if v is not None and v > i - 32),
                   sum(1 << r for r, v in enumerate(vals) if v is not None and v <= i - 32)) for i in range(64)]
        got = self.both(mi.empirical, prog, rows, terms=[[0, 63], [31, 32, 33]])
        self.check_first(got, f, leaves, n, (1 << n) - 1, 64, 64)
        for t in got["interactions"]:
            want = oracle(f, leaves, n, (1 << n) - 1, list(t["conditions"]))
            self.assertEqual((t["sensitive"], t["unknown"], t["signed_sum"]),
                             (want["sensitive"], want["unknown"], want["signed_sum"]))
        empty = mb.Masks.from_scope(mbk.Scope.from_table(db["t"], ["x"], rows=np.zeros(n, bool)),
                                    [mb.condition("x", ">", 0), mb.condition("x", "<", 5)])
        e = self.both(mi.empirical, mb.Program(mb.var(0) | mb.var(1), 2), empty, order=2)
        self.assertEqual((e["points"], e["total_influence"], e["sensitivity"]), (0, 0, {0: 0, 1: 0, 2: 0}))
        self.assertEqual({(i["sensitive"], i["unknown"], i["influence"]) for i in e["influences"]}, {(0, 0, 0)})
        u = self.both(mi.uniform, mb.Program(mb.TRUE, 0))                  # the one empty assignment
        self.assertEqual((u["points"], u["influences"], u["sensitivity"], u["f"]["true"]), (1, [], {0: 1}, 1))

    def test_bool002_budgets_at_the_limit_and_one_past(self):
        k = 12
        prog = mb.Program(random_expr(random.Random(3104), k, size=30, share=0.6), k)
        mi.uniform(prog, budget=Budget(support=4096))
        with self.assertRaises(AlgebraBudgetError) as ctx:        # the uniform measure enumerates 2^k points
            mi.uniform(prog, budget=Budget(support=4095))
        self.assertEqual((ctx.exception.resource, ctx.exception.need), ("support", 4096))
        nrng = np.random.default_rng(3105)
        db = dataset(nrng, 600)
        scope = mbk.Scope.from_table(db["t"], ["x", "y", "z", "w"])
        k = 8
        rows = mb.Masks.from_scope(scope, [mb.condition(*s) for s in SPECS[:k]])
        p8 = mb.Program(random_expr(random.Random(3106), k, size=20, share=0.6), k)
        for backend in ("cpu", "cuda"):
            need = mi.influence_need(p8, rows=rows, order=2, terms=[[0, 1, 2]], backend=backend)
            self.assertEqual((need["degree"], need["tensor"], need["output"]),
                             (3, 2 * k + 4 * 28 + 8, 28 + 1 + 2 * k + 1))
            for resource in ("degree", "tensor", "output", "bytes"):
                mi.empirical(p8, rows, order=2, terms=[[0, 1, 2]], backend=backend,
                             budget=Budget(**{resource: need[resource]}))
                with self.assertRaises(AlgebraBudgetError) as ctx:
                    mi.empirical(p8, rows, order=2, terms=[[0, 1, 2]], backend=backend,
                                 budget=Budget(**{resource: need[resource] - 1}))
                self.assertEqual((ctx.exception.resource, ctx.exception.need), (resource, need[resource]))
        self.assertEqual(len(mi.interaction_sets(8, 2, budget=Budget(output=28))), 28)
        with self.assertRaises(AlgebraBudgetError):
            mi.interaction_sets(8, 2, budget=Budget(output=27))
        p21 = mb.Program(mb.var(20) & mb.var(0), 21)
        rows21 = mb.Masks.from_scope(scope, [mb.condition("x", ">", i) for i in range(21)])
        with self.assertRaises(UnsupportedError):                 # order > 20 (representation limit)
            mi.empirical(p21, rows21, terms=[list(range(21))], first=False,
                         budget=Budget(tensor=1 << 22, degree=64, output=1 << 22))

    def test_bool002_stale_scope_is_refused(self):
        nrng = np.random.default_rng(3107)
        db = dataset(nrng, 300)
        t = db["t"]
        scope = mbk.Scope.from_table(t, ["x", "y"])
        rows = mb.Masks.from_scope(scope, [mb.condition("x", ">", 0), mb.condition("y", "<", 0)])
        prog = mb.Program(mb.var(0) & mb.var(1), 2)
        mi.empirical(prog, rows)
        t.flip(0, 0)
        for backend in ("cpu", "cuda"):
            with self.assertRaises(StaleStateError):
                mi.empirical(prog, rows, backend=backend)
        mi.uniform(prog)                                          # the uniform measure reads no table

    def test_bool002_ctypes_structures_are_registered(self):
        report = mh.abi_check()
        mine = {r["native"] for r in report["checked"] if r["structure"].startswith("torbles.math_influence.")}
        self.assertEqual(mine, {"tr_influence_term", "tr_influence_summary"})
        cuda = {r["native"] for r in report["checked"] if r["library"] == "cuda"
                and r["structure"].startswith("torbles.math_influence.")}
        self.assertEqual(cuda, {"tr_influence_term", "tr_influence_summary"})
        self.assertTrue(all(r["match"] for r in report["checked"]))


if __name__ == "__main__":
    unittest.main()
