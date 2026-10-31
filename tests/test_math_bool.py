# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""BOOL-001: predicate cofactors, XOR derivatives and signed first and mixed differences over the assignment domain
and over table rows (torbles/math_bool.c, math_bool_gpu.cu, math_bool.py).

Oracles, independent of the implementation: complete truth tables written here with Python integers as bit sets
(dual-rail TRUE / FALSE sets, Kleene's connectives, cofactors by substituting constant sets), scalar brute force
(Python True / False / None per point), Python Fractions on the source values for exact constants and bit tests, and
DuckDB SQL over the typed source table (three-valued WHERE / FILTER semantics, NULL arithmetic for the signed
differences). Every native result is also computed on the CUDA path (emulated here with TORBLES_EMU_SCRAMBLE=1, a
scrambled thread order) and compared for equality. Seeds are fixed.
"""
import random
import unittest
from fractions import Fraction

import numpy as np
import pandas as pd

import torbles
from torbles import math_bank as mbk
from torbles import math_bool as mb
from torbles import math_handles as mh
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
        key = id(e)
        if key in memo:
            return memo[key]
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
        memo[key] = r
        return r
    return go(expr)


def scalar(expr, cond):
    """brute force at one point: cond[i] in (True, False, None); Kleene logic, None = UNKNOWN"""
    if expr.op == CONST:
        return (False, True, None)[expr.a]
    if expr.op == VAR:
        return cond[expr.a]
    if expr.op == NOT:
        a = scalar(expr.a, cond)
        return None if a is None else not a
    a, b = scalar(expr.a, cond), scalar(expr.b, cond)
    if expr.op == AND:
        return False if a is False or b is False else (None if a is None or b is None else True)
    if expr.op == OR:
        return True if a is True or b is True else (None if a is None or b is None else False)
    return None if a is None or b is None else a != b


def bits(x, n):
    """a bit set as a numpy bool array of n points"""
    raw = x.to_bytes((n + 7) // 8, "little")
    return np.unpackbits(np.frombuffer(raw, dtype=np.uint8), bitorder="little")[:n].astype(bool)


def oracle(expr, leaves, n, sel, S, cofactors=True):
    """the counts of the differences over S at the points of the bit set sel (leaves: the conditions' sets)"""
    full = (1 << n) - 1
    S = sorted(S)
    m = len(S)
    T, F = tables(expr, leaves, full)
    out = dict(f=dict(true=(T & sel).bit_count(), false=(F & sel).bit_count(),
                      unknown=(sel & ~(T | F)).bit_count()))
    cof, known, parity = [], sel, 0
    D = np.zeros(n, dtype=np.int64)
    for t in range(1 << m):
        sub = list(leaves)
        for j, c in enumerate(S):
            sub[c] = (full, 0) if (t >> j) & 1 else (0, full)
        Tt, Ft = tables(expr, sub, full)
        cof.append(dict(assignment={c: (t >> j) & 1 for j, c in enumerate(S)}, true=(Tt & sel).bit_count(),
                        false=(Ft & sel).bit_count(), unknown=(sel & ~(Tt | Ft)).bit_count()))
        known &= Tt | Ft
        parity ^= Tt
        sign = -1 if (m - bin(t).count("1")) % 2 else 1
        D += sign * bits(Tt, n).astype(np.int64)
    kb = bits(known, n)
    lo, nb = (-(1 << (m - 1)), (1 << m) + 1) if m else (0, 2)
    hist = {lo + b: 0 for b in range(nb)}
    vals, counts = np.unique(D[kb], return_counts=True)
    for v, c in zip(vals.tolist(), counts.tolist()):
        hist[v] = c
    out.update(conditions=tuple(S), order=m, points=sel.bit_count(), known=known.bit_count(),
               unknown=(sel & ~known).bit_count(), xor=dict(true=(parity & known).bit_count(),
                                                             false=(~parity & known & full).bit_count()),
               signed=hist, sum=int(D[kb].sum()), nonzero=int((D[kb] != 0).sum()),
               cofactors=cof if cofactors else None)
    return out


def assignment_leaves(k):
    n = 1 << k
    return [(sum(1 << a for a in range(n) if (a >> i) & 1), sum(1 << a for a in range(n) if not (a >> i) & 1))
            for i in range(k)], n


def random_expr(rng, k, size, share=0.5, constants=True, unknown=True):
    """a random DAG: leaves are conditions (and constants); each new node draws operands from the recent nodes, so
    a high `share` reuses subexpressions"""
    pool = [mb.var(i) for i in range(k)]
    if constants:
        pool += [mb.TRUE, mb.FALSE] + ([mb.UNKNOWN] if unknown else [])
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
    for e in built[-4:-1]:                       # tie in more of the DAG
        top = top ^ e if rng.random() < 0.5 else top | e
    return top


def strip(results):
    return [{k: v for k, v in r.items() if k not in ("backend", "device")} for r in results]


def obj(values):
    return pd.Series(values, dtype=object)


DDL = "CREATE TABLE t (x BIGINT, y BIGINT, d DECIMAL(10,2), b BOOLEAN, k BIGINT)"


def dataset(rng, n, nulls=0.2):
    x = [int(v) for v in rng.integers(-50, 51, n)]
    y = [int(v) for v in rng.integers(-50, 51, n)]
    d = [f"{int(v) / 100:.2f}" for v in rng.integers(-5000, 5001, n)]
    b = [bool(v) for v in rng.integers(0, 2, n)]
    k = [int(v) for v in rng.integers(0, 4, n)]
    for col in (x, y, d, b):
        for i in np.flatnonzero(rng.random(n) < nulls):
            col[int(i)] = None
    frame = pd.DataFrame({"x": obj(x), "y": obj(y), "d": obj(d), "b": obj(b), "k": obj(k)})
    return torbles.from_ddl(DDL, {"t": frame}), dict(x=x, y=y, d=d, b=b, k=k)


SCALES = {"x": 0, "y": 0, "d": 2, "b": 0, "k": 0}
SQLOP = {"=": "=", "<>": "<>", "<": "<", "<=": "<=", ">": ">", ">=": ">="}


def cond_sql(spec):
    """the SQL of a condition spec (column, op, value, high) over the typed source table"""
    col, op, value, high = spec

    def scaled(c, f):
        cast = f'CAST("{c}" AS DECIMAL(38,{SCALES[c]}))' if SCALES[c] else f'CAST("{c}" AS HUGEINT)'
        return f"({cast} * {f.denominator})", f.numerator
    if op in ("is null", "is not null"):
        return f'("{col}" {op.upper()})'
    if isinstance(value, mb.column):
        s = max(SCALES[col], SCALES[value.name])
        return f'(CAST("{col}" AS DECIMAL(38,{s})) {SQLOP[op]} CAST("{value.name}" AS DECIMAL(38,{s})))'
    if op == "between":
        (l, n0), (h, n1) = scaled(col, Fraction(value)), scaled(col, Fraction(high))
        return f"({l} BETWEEN {n0} AND {n1})" if l == h else f"({l} >= {n0} AND {h} <= {n1})"
    lhs, num = scaled(col, Fraction(value))
    return f"({lhs} {SQLOP[op]} {num})"


def expr_sql(expr, conds_sql):
    """nested SQL of a predicate DAG (shared nodes are repeated: the test DAGs are small)"""
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


class Bool(unittest.TestCase):
    def both(self, masks, prog, queries, cofactors=True, **kw):
        """the CPU result, checked equal to the (emulated, scrambled) CUDA result"""
        cpu = mb.differences(masks, prog, queries, cofactors=cofactors, **kw)
        gpu = mb.differences(masks, prog, queries, cofactors=cofactors, backend="cuda", **kw)
        self.assertEqual(strip(cpu), strip(gpu))
        self.assertIn(gpu[0]["backend"] if gpu else "cuda-emulated", ("cuda", "cuda-emulated"))
        return strip(cpu)

    def check(self, got, want, what=""):
        for g, w in zip(got, want):
            for key in w:
                self.assertEqual(g[key], w[key], (what, g["conditions"], key))

    # ------------------------------------------------------------------ complete truth tables
    def test_bool001_cofactors_and_first_differences_match_complete_truth_tables(self):
        rng = random.Random(1001)
        for k in (1, 2, 3, 5, 8):
            leaves, n = assignment_leaves(k)
            masks = mb.Masks.assignments(k)
            self.assertEqual(masks.points, n)
            for trial in range(4):
                expr = random_expr(rng, k, size=3 + 3 * k, share=(0.0, 0.4, 0.8, 0.6)[trial])
                prog = mb.Program(expr, k)
                got = self.both(masks, prog, [[i] for i in range(k)])
                want = [oracle(expr, leaves, n, (1 << n) - 1, [i]) for i in range(k)]
                self.check(got, want, (k, trial))
                for i in range(k):                      # the convenience readers agree
                    c0, c1 = mb.cofactors(masks, prog, i)
                    self.assertEqual((c0, c1), tuple({x: c[x] for x in ("true", "false", "unknown")}
                                                     for c in want[i]["cofactors"]))

    def test_bool001_mixed_differences_match_complete_truth_tables(self):
        rng = random.Random(1002)
        for k in (3, 4, 6):
            leaves, n = assignment_leaves(k)
            masks = mb.Masks.assignments(k)
            expr = random_expr(rng, k, size=4 * k, share=0.6)
            prog = mb.Program(expr, k)
            subsets = [[i for i in range(k) if (s >> i) & 1] for s in range(1 << k)]   # every S, the empty one too
            got = self.both(masks, prog, subsets)
            self.check(got, [oracle(expr, leaves, n, (1 << n) - 1, S) for S in subsets], k)
        for k in (10, 12):
            leaves, n = assignment_leaves(k)
            masks = mb.Masks.assignments(k)
            expr = random_expr(rng, k, size=3 * k, share=0.7)
            prog = mb.Program(expr, k)
            queries = [sorted(rng.sample(range(k), m)) for m in (1, 2, 3, 4, 5) for _ in range(2)]
            got = self.both(masks, prog, queries)
            self.check(got, [oracle(expr, leaves, n, (1 << n) - 1, S) for S in queries], k)

    def test_bool001_sixteen_conditions_first_and_second_differences(self):
        rng = random.Random(1016)
        k = 16
        leaves, n = assignment_leaves(k)
        masks = mb.Masks.assignments(k)
        expr = random_expr(rng, k, size=48, share=0.7, unknown=False)
        prog = mb.Program(expr, k)
        queries = [[i] for i in range(k)] + [[0, 15], [3, 7], [5, 6]]
        got = self.both(masks, prog, queries, cofactors=True)
        self.check(got, [oracle(expr, leaves, n, (1 << n) - 1, S) for S in queries], k)

    def test_bool001_xor_derivative_is_not_the_signed_difference(self):
        x = mb.variables(3)
        masks = mb.Masks.assignments(3)
        prog = mb.Program(x[0] ^ x[1], 3)
        r01, r0 = self.both(masks, prog, [[0, 1], [0]])
        self.assertEqual((r01["signed"][-2], r01["xor"]["true"], r01["sum"]), (8, 0, -16))   # D = -2, dS = 0
        self.assertEqual((r0["signed"], r0["xor"]["true"], r0["sum"]), ({-1: 4, 0: 0, 1: 4}, 8, 0))   # cancels
        mono = mb.Program(x[0] | (x[1] & x[2]), 3)
        for r in self.both(masks, mono, [[0], [1], [2]]):
            self.assertEqual(r["signed"][-1], 0)                # monotone: no -1
            self.assertEqual(r["xor"]["true"], r["signed"][1])  # first order: dS = |DS|
        anti = mb.Program(x[0] & ~x[1], 3)
        r = self.both(masks, anti, [[1]])[0]
        self.assertEqual(r["signed"], {-1: 4, 0: 4, 1: 0})          # D1 f = -x0
        self.assertEqual(r["xor"]["true"], 4)
        for expr in (x[0] & x[1] & x[2], x[0] ^ x[1] ^ x[2], (x[0] | x[1]) & ~x[2]):   # mod 2 law, any order
            p = mb.Program(expr, 3)
            for res in self.both(masks, p, [[0], [0, 1], [0, 1, 2], [1, 2]]):
                odd = sum(c for v, c in res["signed"].items() if v % 2)
                self.assertEqual(res["xor"]["true"], odd)
                self.assertEqual(res["nonzero"], sum(c for v, c in res["signed"].items() if v))

    # ------------------------------------------------------------------ rows: SQL semantics
    SPECS = [("x", ">", 5, None), ("y", "<=", Fraction(-7, 3), None), ("d", "between", "-12.50", "20.25"),
             ("b", "=", 1, None), ("x", "is null", None, None), ("d", "<>", "3.07", None),
             ("x", "<", mb.column("d"), None), ("y", "is not null", None, None), ("d", ">=", Fraction(1, 3), None),
             ("x", ">=", mb.column("y"), None)]

    def conditions(self, specs):
        return [mb.condition(c, op, v, h) for c, op, v, h in specs]

    def duck_rows(self, db, sel_mask):
        con = db.con
        con.register("sel_rows", pd.DataFrame({"rid": np.flatnonzero(sel_mask).astype(np.int64)}))
        return "rowid IN (SELECT rid FROM sel_rows)"

    def duck_leaves(self, db, specs, sel_mask, n):
        """each condition's TRUE and FALSE row sets by DuckDB (three-valued: NULL in neither)"""
        where = self.duck_rows(db, sel_mask)
        leaves = []
        for spec in specs:
            s = cond_sql(spec)
            t = [r for r, in db.con.execute(f"SELECT rowid FROM t WHERE {where} AND {s}").fetchall()]
            f = [r for r, in db.con.execute(f"SELECT rowid FROM t WHERE {where} AND NOT {s}").fetchall()]
            leaves.append((sum(1 << r for r in t), sum(1 << r for r in f)))
        return leaves

    def test_bool001_row_conditions_match_duckdb_with_sql_nulls(self):
        rng = np.random.default_rng(2001)
        n = 1500
        db, src = dataset(rng, n)
        t = db["t"]
        for sel in (np.ones(n, bool), rng.random(n) < 0.4):
            scope = mbk.Scope.from_table(t, ["x", "y", "d", "b"], rows=sel)
            conds = self.conditions(self.SPECS)
            masks = mb.Masks.from_scope(scope, conds)
            gpu = mb.Masks.from_scope(scope, conds, backend="cuda")
            self.assertTrue(masks.equal(gpu))
            self.assertIn(gpu.backend, ("cuda", "cuda-emulated"))
            for i in range(len(conds)):
                a, b = masks.words(i), gpu.words(i)
                self.assertTrue(np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]))
            where = self.duck_rows(db, sel)
            self.assertEqual(masks.points, int(sel.sum()))
            for i, spec in enumerate(self.SPECS):
                s = cond_sql(spec)
                want = db.con.execute(f"SELECT COUNT(*) FILTER (WHERE {s}), COUNT(*) FILTER (WHERE NOT {s}), "
                                      f"COUNT(*) FILTER (WHERE ({s}) IS NULL) FROM t WHERE {where}").fetchall()[0]
                self.assertEqual(tuple(masks.counts(i).values()), tuple(int(v) for v in want), spec)

    def test_bool001_row_cofactors_and_differences_match_duckdb(self):
        rng = np.random.default_rng(2002)
        prng = random.Random(2002)
        n = 900
        db, src = dataset(rng, n, nulls=0.25)
        t = db["t"]
        sel = rng.random(n) < 0.7
        scope = mbk.Scope.from_table(t, ["x", "y", "d", "b"], rows=sel)
        specs = self.SPECS[:6]
        k = len(specs)
        masks = mb.Masks.from_scope(scope, self.conditions(specs))
        leaves = self.duck_leaves(db, specs, sel, n)
        selbits = sum(1 << int(r) for r in np.flatnonzero(sel))
        for trial in range(3):
            expr = random_expr(prng, k, size=14, share=0.5)
            prog = mb.Program(expr, k)
            queries = [[i] for i in range(k)] + [[0, 3], [1, 2, 5], [], [0, 1, 2, 3]]
            got = self.both(masks, prog, queries)
            self.check(got, [oracle(expr, leaves, n, selbits, S) for S in queries], trial)
            # the whole evaluation in SQL: f, a signed mixed difference and its XOR derivative
            csql = [cond_sql(s) for s in specs]
            where = self.duck_rows(db, sel)
            fs = expr_sql(expr, csql)
            want_f = db.con.execute(f"SELECT COUNT(*) FILTER (WHERE {fs}), COUNT(*) FILTER (WHERE NOT {fs}), "
                                    f"COUNT(*) FILTER (WHERE ({fs}) IS NULL) FROM t WHERE {where}").fetchall()[0]
            self.assertEqual(tuple(got[0]["f"].values()), tuple(int(v) for v in want_f))
            S = [1, 2, 5]
            terms = []
            for tt in range(8):
                sub = list(csql)
                for j, c in enumerate(S):
                    sub[c] = "TRUE" if (tt >> j) & 1 else "FALSE"
                sign = "-" if (3 - bin(tt).count("1")) % 2 else "+"
                terms.append(f"{sign} CAST({expr_sql(expr, sub)} AS INTEGER)")
            D = "(" + " ".join(terms) + ")"
            rows = db.con.execute(f"SELECT {D} AS v, COUNT(*) FROM t WHERE {where} GROUP BY v").fetchall()
            hist = {v: int(c) for v, c in rows if v is not None}
            unknown = sum(int(c) for v, c in rows if v is None)
            r = got[queries.index(S)]
            self.assertEqual({v: c for v, c in r["signed"].items() if c}, hist)
            self.assertEqual(r["unknown"], unknown)
            self.assertEqual(r["xor"]["true"], sum(c for v, c in hist.items() if v % 2))

    def test_bool001_unknown_rows_are_counted_separately_never_false(self):
        frame = pd.DataFrame({"x": obj([1, 5, None, 9, None, 2]), "y": obj([None, 3, 4, None, None, 8])})
        db = torbles.from_ddl("CREATE TABLE t (x BIGINT, y BIGINT)", {"t": frame})
        scope = mbk.Scope.from_table(db["t"], ["x", "y"])
        conds = [mb.condition("x", ">", 3), mb.condition("y", ">", 3)]
        masks = mb.Masks.from_scope(scope, conds)
        self.assertEqual(masks.counts(0), dict(true=2, false=2, unknown=2))
        self.assertEqual(masks.counts(1), dict(true=2, false=1, unknown=3))
        c = mb.variables(2)
        f = c[0] | c[1]
        prog = mb.Program(f, 2)
        r0, r1, r01 = self.both(masks, prog, [[0], [1], [0, 1]])
        # scalar brute force with None for UNKNOWN (rows: x > 3, y > 3)
        cond = [(False, None), (True, False), (None, True), (True, None), (None, None), (False, True)]
        self.assertEqual(r0["f"], dict(true=4, false=0, unknown=2))
        for i, r in ((0, r0), (1, r1)):
            t1 = [scalar(f, [True if j == i else cv[j] for j in range(2)]) for cv in cond]
            t0 = [scalar(f, [False if j == i else cv[j] for j in range(2)]) for cv in cond]
            known = [a is not None and b is not None for a, b in zip(t1, t0)]
            self.assertEqual((r["known"], r["unknown"]), (sum(known), 6 - sum(known)))
            self.assertEqual(r["signed"][1], sum(1 for a, b, k in zip(t1, t0, known) if k and a and not b))
            self.assertEqual(r["xor"]["false"], sum(1 for a, b, k in zip(t1, t0, known) if k and a == b))
            self.assertEqual(r["cofactors"][1]["unknown"], sum(1 for a in t1 if a is None))
        # the cofactor over a NULL condition is definite: f|y=1 is TRUE on every row (OR), also where y is NULL
        self.assertEqual(r1["cofactors"][1], dict(assignment={1: 1}, true=6, false=0, unknown=0))
        # f|y=0 = x > 3: UNKNOWN where x is NULL (rows 2 and 4), never FALSE there
        self.assertEqual(r1["cofactors"][0], dict(assignment={1: 0}, true=2, false=2, unknown=2))
        self.assertEqual(r01["unknown"], 0)                   # every cofactor over {x, y} is a constant
        self.assertEqual(r01["signed"], {-2: 0, -1: 6, 0: 0, 1: 0, 2: 0})   # 1 - 1 - 1 + 0
        for r in (r0, r1, r01):                              # the partitions of the points
            self.assertEqual(sum(r["f"].values()), r["points"])
            self.assertEqual(r["known"] + r["unknown"], r["points"])
            self.assertEqual(sum(r["xor"].values()), r["known"])
            self.assertEqual(sum(r["signed"].values()), r["known"])
        u = mb.Program(c[0] & mb.UNKNOWN, 2)                  # SQL's NULL literal: FALSE AND NULL = FALSE
        r = self.both(masks, u, [[0]])[0]
        self.assertEqual(r["f"], dict(true=0, false=2, unknown=4))
        self.assertEqual(r["cofactors"][0]["false"], 6)
        self.assertEqual((r["known"], r["unknown"]), (0, 6))

    def test_bool001_bit_conditions_and_exact_constants(self):
        rng = np.random.default_rng(2003)
        n = 700
        db, src = dataset(rng, n)
        scope = mbk.Scope.from_table(db["t"], ["x", "d"])
        specs = [("x", "bit", 0, 0), ("x", "bit", 63, 0), ("x", "bit", 2, -9), ("d", "bit", 5, 0),
                 ("d", "bit", 63, 100), ("d", "<", Fraction(10, 3), None), ("x", "=", Fraction(-14, 2), None),
                 ("d", "between", Fraction(-1, 7), "0.15", None)]
        conds = []
        for s in specs:
            if s[1] == "bit":
                conds.append(mb.condition(s[0], "bit", s[2], base=s[3]))
            elif s[1] == "between":
                conds.append(mb.condition(s[0], "between", s[2], s[3]))
            else:
                conds.append(mb.condition(s[0], s[1], s[2]))
        masks = mb.Masks.from_scope(scope, conds)
        gpu = mb.Masks.from_scope(scope, conds, backend="cuda")
        self.assertTrue(masks.equal(gpu))
        for i, s in enumerate(specs):
            want = dict(true=0, false=0, unknown=0)
            for r in range(n):
                v = src[s[0]][r]
                if v is None:
                    want["unknown"] += 1
                    continue
                if s[0] == "d":
                    value = Fraction(v)
                    logical = int(value * 100)
                else:
                    value = Fraction(v)
                    logical = int(v)
                if s[1] == "bit":
                    hit = (((logical - s[3]) % (1 << 64)) >> s[2]) & 1 == 1
                elif s[1] == "<":
                    hit = value < Fraction(s[2])
                elif s[1] == "=":
                    hit = value == Fraction(s[2])
                else:
                    hit = Fraction(s[2]) <= value <= Fraction(s[3])
                want["true" if hit else "false"] += 1
            self.assertEqual(masks.counts(i), want, s)
        with self.assertRaises(TypeError):
            mb.condition("x", ">", 0.5)                           # a float is not exact
        with self.assertRaises(OverflowError):
            mb.condition("x", ">", 1 << 63)
        with self.assertRaises(ValueError):
            mb.condition("x", "bit", 64)

    # ------------------------------------------------------------------ sharing, the baseline, programs
    def test_bool001_shared_subexpressions_direct_baseline_and_unreachable_nodes(self):
        prng = random.Random(3001)
        rng = np.random.default_rng(3001)
        n = 600
        db, src = dataset(rng, n)
        sel = rng.random(n) < 0.8
        scope = mbk.Scope.from_table(db["t"], ["x", "y", "d", "b"], rows=sel)
        specs = self.SPECS[:5]
        conds = self.conditions(specs)
        masks = mb.Masks.from_scope(scope, conds)
        x = mb.variables(5)
        s1 = (x[0] & x[1]) | x[2]                                  # shared three ways
        s2 = s1 ^ x[3]
        f = (s1 & s2) | (~s1 & x[4]) | (s2 ^ s1)
        prog = mb.Program(f, 5)
        self.assertLess(prog.info["nodes"], 20)                   # a DAG, not the expanded tree
        for S in ([0], [2, 4], [0, 1, 3]):
            r = self.both(masks, prog, [S])[0]
            for c in r["cofactors"]:
                d = mb.direct(prog, c["assignment"], scope=scope, conditions=conds)
                self.assertEqual(d, {k: c[k] for k in ("true", "false", "unknown")}, (S, c["assignment"]))
        self.assertEqual(mb.direct(prog, {}, scope=scope, conditions=conds), r["f"])
        uniform = mb.Masks.assignments(5)
        r = self.both(uniform, prog, [[1, 4]])[0]
        for c in r["cofactors"]:
            self.assertEqual(mb.direct(prog, c["assignment"]), {k: c[k] for k in ("true", "false", "unknown")})
        # explicit nodes, with unreachable nodes and a root that is not the last node
        nodes = [("var", 0), ("var", 1), ("and", 0, 1), ("var", 2), ("xor", 2, 3), ("not", 4), ("or", 5, 0),
                 ("const", 2), ("and", 7, 6)]
        p = mb.Program.from_nodes(3, nodes, root=6)
        self.assertEqual((p.info["nodes"], p.info["evaluated"], p.info["support"]), (9, 7, (0, 1, 2)))
        q = mb.Program(~(((x[0] & x[1]) ^ x[2])) | x[0], 3)
        u3 = mb.Masks.assignments(3)
        self.assertEqual(strip(mb.differences(u3, p, [[0], [1, 2]], cofactors=True)),
                         strip(mb.differences(u3, q, [[0], [1, 2]], cofactors=True)))

    def test_bool001_sixty_four_conditions_empty_domains_and_k0(self):
        vals = list(range(-30, 30)) + [None] * 4
        frame = pd.DataFrame({"x": obj(vals)})
        db = torbles.from_ddl("CREATE TABLE t (x BIGINT)", {"t": frame})
        scope = mbk.Scope.from_table(db["t"], ["x"])
        conds = [mb.condition("x", ">", i - 32) for i in range(64)]
        masks = mb.Masks.from_scope(scope, conds)
        x = mb.variables(64)
        f = x[0]
        for i in range(1, 64):
            f = (f ^ x[i]) if i % 3 else (f & x[i]) | x[i - 1]
        prog = mb.Program(f, 64)
        n = len(vals)
        leaves = [(sum(1 << r for r, v in enumerate(vals) if v is not None and v > i - 32),
                   sum(1 << r for r, v in enumerate(vals) if v is not None and v <= i - 32)) for i in range(64)]
        queries = [[0], [63], [0, 63], [31, 32, 33]]
        got = self.both(masks, prog, queries)
        self.check(got, [oracle(f, leaves, n, (1 << n) - 1, S) for S in queries], 64)
        cond = [[None if v is None else v > i - 32 for i in range(64)] for v in vals]
        for S, r in zip(queries, got):                          # scalar brute force of the cofactors too
            for c in r["cofactors"]:
                outs = [scalar(f, [bool(c["assignment"][i]) if i in c["assignment"] else cv[i] for i in range(64)])
                        for cv in cond]
                self.assertEqual((c["true"], c["false"], c["unknown"]),
                                 (outs.count(True), outs.count(False), outs.count(None)), (S, c["assignment"]))
        empty = mb.Masks.from_scope(mbk.Scope.from_table(db["t"], ["x"], rows=np.zeros(n, bool)), conds[:3])
        p3 = mb.Program(mb.var(0) | mb.var(2), 3)
        for r in self.both(empty, p3, [[0], [1, 2], []]):        # nothing selected: every count is zero
            self.assertEqual((r["points"], r["known"], r["unknown"], r["sum"], set(r["signed"].values())),
                             (0, 0, 0, 0, {0}))
            self.assertEqual({c["true"] + c["false"] + c["unknown"] for c in r["cofactors"]}, {0})
        r = self.both(mb.Masks.assignments(0), mb.Program(mb.TRUE, 0), [[]])[0]   # the one empty assignment
        self.assertEqual((r["points"], r["f"]["true"], r["signed"], r["xor"]["true"]), (1, 1, {0: 0, 1: 1}, 1))

    def test_bool001_malformed_input_is_refused(self):
        with self.assertRaises(ValueError):
            mb.Program.from_nodes(2, [("var", 0), ("and", 0, 2), ("var", 1)])     # a forward reference
        with self.assertRaises(ValueError):
            mb.Program.from_nodes(2, [("var", 2)])                                 # condition >= k
        with self.assertRaises(ValueError):
            mb.Program.from_nodes(2, [("var", 0), (9, 0, -1)])                     # unknown op
        with self.assertRaises(ValueError):
            mb.Program.from_nodes(2, [("var", 0)], root=1)
        with self.assertRaises(ValueError):
            mb.Program.from_nodes(1, [("const", 3)])
        with self.assertRaises(TypeError):
            bool(mb.var(0) & mb.var(1))                                            # not a Python bool
        x = mb.variables(3)
        masks = mb.Masks.assignments(3)
        prog = mb.Program(x[0] & x[2], 3)
        with self.assertRaises(ValueError):
            mb.differences(masks, prog, [[3]])
        with self.assertRaises(ValueError):
            mb.differences(masks, prog, [[1, 1]])
        with self.assertRaises(DomainError):
            mb.differences(mb.Masks.assignments(4), prog, [[0]])
        big = mb.Masks.assignments(21, budget=Budget(support=1 << 21))
        p21 = mb.Program(mb.var(20), 21)
        with self.assertRaises(UnsupportedError):                                  # order > 20
            mb.differences(big, p21, [list(range(21))], budget=Budget(support=1 << 21, tensor=1 << 22,
                                                                      output=1 << 22, degree=64))
        frame = pd.DataFrame({"x": obj([1, 2, 3])})
        db = torbles.from_ddl("CREATE TABLE t (x BIGINT)", {"t": frame})
        scope = mbk.Scope.from_table(db["t"], ["x"])
        with self.assertRaises(ValueError):
            mb.Masks.from_scope(scope, [mb.condition("nope", ">", 1)])
        with self.assertRaises(UnsupportedError):                                  # more than 63 variables
            mb.Masks.assignments(64, budget=Budget(variables=64, support=(1 << 64) - 1))

    def test_bool001_budgets_at_the_limit_and_one_past(self):
        need = mb.masks_need(None, 10)
        self.assertEqual((need["variables"], need["support"]), (10, 1024))
        mb.Masks.assignments(10, budget=Budget(support=1024))
        with self.assertRaises(AlgebraBudgetError) as ctx:
            mb.Masks.assignments(11, budget=Budget(support=1024))
        self.assertEqual((ctx.exception.resource, ctx.exception.need, ctx.exception.limit), ("support", 2048, 1024))
        rng = np.random.default_rng(4001)
        db, src = dataset(rng, 1000)
        scope = mbk.Scope.from_table(db["t"], ["x", "y", "d", "b"])
        conds = self.conditions(self.SPECS[:4])
        for backend in ("cpu", "cuda"):
            nb = mb.masks_need(scope, conds, backend=backend)["bytes"]
            mb.Masks.from_scope(scope, conds, backend=backend, budget=Budget(bytes=nb))
            with self.assertRaises(AlgebraBudgetError) as ctx:
                mb.Masks.from_scope(scope, conds, backend=backend, budget=Budget(bytes=nb - 1))
            self.assertEqual((ctx.exception.resource, ctx.exception.need), ("bytes", nb))
        x = mb.variables(4)
        expr = (x[0] & x[1]) ^ (x[2] | x[3])
        prog = mb.Program(expr, 4)
        nn = prog.info["nodes"]
        mb.Program(expr, 4, budget=Budget(circuit=nn))
        with self.assertRaises(AlgebraBudgetError) as ctx:
            mb.Program(expr, 4, budget=Budget(circuit=nn - 1))
        self.assertEqual(ctx.exception.resource, "circuit")
        masks = mb.Masks.from_scope(scope, conds)
        queries = [[0, 1], [2], [0, 1, 2, 3]]
        for backend in ("cpu", "cuda"):
            need = mb.differences_need(masks, prog, queries, cofactors=True, backend=backend)
            self.assertEqual((need["degree"], need["tensor"], need["output"]), (4, 4 + 2 + 16, 3 + 5 + 3 + 17 + 3 * 22))
            for resource in ("degree", "tensor", "output", "bytes"):
                mb.differences(masks, prog, queries, cofactors=True, backend=backend,
                               budget=Budget(**{resource: need[resource]}))
                with self.assertRaises(AlgebraBudgetError) as ctx:
                    mb.differences(masks, prog, queries, cofactors=True, backend=backend,
                                   budget=Budget(**{resource: need[resource] - 1}))
                self.assertEqual((ctx.exception.resource, ctx.exception.need), (resource, need[resource]))
        uniform = mb.Masks.assignments(12, budget=Budget(support=4096))
        p12 = mb.Program(mb.var(0) ^ mb.var(11), 12)
        mb.differences(uniform, p12, [[0]], budget=Budget(support=4096))
        with self.assertRaises(AlgebraBudgetError) as ctx:
            mb.differences(uniform, p12, [[0]], budget=Budget(support=4095))
        self.assertEqual(ctx.exception.resource, "support")
        for budget in (Budget(support=4096), Budget(support=4095)):          # the independent baseline too
            if budget.limits["support"] < 4096:
                with self.assertRaises(AlgebraBudgetError):
                    mb.direct(p12, {0: 1}, budget=budget)
            else:
                self.assertEqual(sum(mb.direct(p12, {0: 1}, budget=budget).values()), 4096)

    def test_bool001_stale_scope_is_refused(self):
        rng = np.random.default_rng(5001)
        db, src = dataset(rng, 300)
        t = db["t"]
        scope = mbk.Scope.from_table(t, ["x", "y"])
        conds = [mb.condition("x", ">", 0), mb.condition("y", "<", 0)]
        masks = mb.Masks.from_scope(scope, conds)
        prog = mb.Program(mb.var(0) & mb.var(1), 2)
        mb.differences(masks, prog, [[0]])
        t.flip(0, 0)                                                     # a mutation: new revision
        for backend in ("cpu", "cuda"):
            with self.assertRaises(StaleStateError):
                mb.differences(masks, prog, [[0]], backend=backend)
            with self.assertRaises(StaleStateError):
                mb.Masks.from_scope(scope, conds, backend=backend)
        with self.assertRaises(StaleStateError):
            mb.direct(prog, {0: 1}, scope=scope, conditions=conds)

    def test_bool001_ctypes_structures_are_registered(self):
        report = mh.abi_check()
        mine = {r["native"] for r in report["checked"] if r["structure"].startswith("torbles.math_bool.")}
        self.assertEqual(mine, {"tr_bool_condition", "tr_bool_node", "tr_bool_program_view", "tr_bool_domain",
                                "tr_bool_query", "tr_bool_result", "tr_bool_masks_info", "tr_bool_program_info"})
        cuda = {r["native"] for r in report["checked"] if r["library"] == "cuda"
                and r["structure"].startswith("torbles.math_bool.")}
        self.assertEqual(cuda, {"tr_bool_condition", "tr_bool_node", "tr_bool_program_view", "tr_bool_domain",
                                "tr_bool_query", "tr_bool_result"})


if __name__ == "__main__":
    unittest.main()
