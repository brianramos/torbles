# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""LATTICE-001 (bounded Boolean occupancy histograms) and LATTICE-002 (zeta, Moebius and inclusion-exclusion
transforms) over the assignment domain and over table rows (torbles/math_lattice.c, math_lattice_gpu.cu,
math_lattice.py).

Oracles, independent of the implementation: complete enumeration of the 2^k assignments with scalar Kleene logic
written here (Python True / False / None per point), DuckDB SQL over the typed source table (three-valued GROUP BY of
the condition values, COUNT(*) FILTER (WHERE c1 AND c2 ...), WHERE c1 OR c2 ...), and Python integer sums over every
subset for the transforms (zeta, Moebius, inclusion-exclusion, Harsanyi dividends). Every native result is also
computed on the CUDA path (emulated here with TORBLES_EMU_SCRAMBLE=1, a scrambled thread order) and compared for
equality. Seeds are fixed.
"""
import itertools
import random
import unittest
from fractions import Fraction
from math import comb

import numpy as np
import pandas as pd

import torbles
from torbles import math_bank as mbk
from torbles import math_bool as mb
from torbles import math_handles as mh
from torbles import math_lattice as ml
from torbles.math_budgets import Budget
from torbles.math_domains import DomainError
from torbles.math_errors import StaleStateError, UnsupportedError
from torbles.mathematics import AlgebraBudgetError

CONST, VAR, NOT, AND, OR, XOR = mb.CONST, mb.VAR, mb.NOT, mb.AND, mb.OR, mb.XOR


# ---------------------------------------------------------------------- oracles
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
    for e in built[-4:-1]:
        top = top ^ e if rng.random() < 0.5 else top | e
    return top


def cell_of(values, nulls):
    """the cell id of a tuple of axis values (True / False / None)"""
    if nulls == "apart":
        return sum(1 << i for i, v in enumerate(values) if v)
    return sum((2 if v is None else int(v)) * 3 ** i for i, v in enumerate(values))


def expected_cells(groups, k, nulls):
    """cells and unknown points of a histogram from {axis values tuple: count}"""
    cells = [0] * (3 ** k if nulls == "three" else 1 << k)
    unknown = 0
    for vals, n in groups.items():
        if nulls == "apart" and any(v is None for v in vals):
            unknown += n
            continue
        cells[cell_of(vals, nulls)] += n
    return cells, unknown


def zeta_oracle(v, k, op):
    """op of a 2^k vector by its definition (Python ints, every subset)"""
    out = []
    for S in range(1 << k):
        total = 0
        for A in range(1 << k):
            if op in ("zeta_up", "mobius_up") and (A & S) == S:
                sign = -1 if op == "mobius_up" and bin(A & ~S).count("1") % 2 else 1
                total += sign * v[A]
            elif op in ("zeta_down", "mobius_down") and (A & ~S) == 0:
                sign = -1 if op == "mobius_down" and bin(S & ~A).count("1") % 2 else 1
                total += sign * v[A]
        out.append(total)
    return out


def by_mask(k, m):
    """the m-subsets of k axes in increasing mask order (the declared order of the coalitions)"""
    return sorted(itertools.combinations(range(k), m), key=lambda T: sum(1 << i for i in T))


def obj(values):
    return pd.Series(values, dtype=object)


DDL = "CREATE TABLE t (x BIGINT, y BIGINT, d DECIMAL(10,2), b BOOLEAN)"
SCALES = {"x": 0, "y": 0, "d": 2, "b": 0}
SQLOP = {"=": "=", "<>": "<>", "<": "<", "<=": "<=", ">": ">", ">=": ">="}


def dataset(rng, n, nulls=0.15):
    x = [int(v) for v in rng.integers(-50, 51, n)]
    y = [int(v) for v in rng.integers(-50, 51, n)]
    d = [f"{int(v) / 100:.2f}" for v in rng.integers(-5000, 5001, n)]
    b = [bool(v) for v in rng.integers(0, 2, n)]
    for col in (x, y, d, b):
        for i in np.flatnonzero(rng.random(n) < nulls):
            col[int(i)] = None
    frame = pd.DataFrame({"x": obj(x), "y": obj(y), "d": obj(d), "b": obj(b)})
    return torbles.from_ddl(DDL, {"t": frame})


def cond_sql(spec):
    col, op, value = spec[:3]

    def scaled(c, f):
        cast = f'CAST("{c}" AS DECIMAL(38,{SCALES[c]}))' if SCALES[c] else f'CAST("{c}" AS HUGEINT)'
        return f"({cast} * {f.denominator})", f.numerator
    if op in ("is null", "is not null"):
        return f'("{col}" {op.upper()})'
    if op == "between":
        (l, n0), (h, n1) = scaled(col, Fraction(value)), scaled(col, Fraction(spec[3]))
        return f"({l} >= {n0} AND {h} <= {n1})"
    lhs, num = scaled(col, Fraction(value))
    return f"({lhs} {SQLOP[op]} {num})"


def condition(spec):
    return mb.condition(spec[0], spec[1], spec[2], spec[3] if len(spec) > 3 else None)


def specs_of(k):
    """k conditions over the columns, thresholds spread (some overlap, some independent)"""
    pool = [("x", ">", 5), ("y", "<=", Fraction(-7, 3)), ("d", "between", "-12.50", "20.25"), ("b", "=", 1),
            ("x", "<", -20), ("d", ">=", Fraction(1, 3)), ("y", ">", 30), ("x", "<>", 0), ("d", "<", "-30.5"),
            ("y", ">=", 0), ("x", ">", 25), ("d", "<=", "40.00"), ("y", "is null", None), ("x", "<=", -40)]
    return pool[:k]


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


def duck_where(db, sel):
    db.con.register("sel_rows", pd.DataFrame({"rid": np.flatnonzero(sel).astype(np.int64)}))
    return "rowid IN (SELECT rid FROM sel_rows)"


def duck_groups(db, axes_sql, where):
    """{axis values tuple: count} by a three-valued DuckDB GROUP BY of the axis predicates"""
    if not axes_sql:
        return {(): int(db.con.execute(f"SELECT COUNT(*) FROM t WHERE {where}").fetchall()[0][0])}
    cols = ", ".join(f"{s} AS a{i}" for i, s in enumerate(axes_sql))
    rows = db.con.execute(f"SELECT {cols}, COUNT(*) FROM t WHERE {where} GROUP BY ALL").fetchall()
    return {tuple(r[:-1]): int(r[-1]) for r in rows}


def duck_count(db, predicate, where):
    return int(db.con.execute(f"SELECT COUNT(*) FROM t WHERE {where} AND ({predicate})").fetchall()[0][0])


class Lattice(unittest.TestCase):
    def build(self, masks, axes=None, nulls="apart", **kw):
        """the CPU histogram, checked equal to the (emulated, scrambled) CUDA one"""
        cpu = ml.Histogram.build(masks, axes, nulls=nulls, **kw)
        gpu = ml.Histogram.build(masks, axes, nulls=nulls, backend="cuda", **kw)
        self.assertTrue(cpu.equal(gpu))
        self.assertEqual(cpu.counts().tolist(), gpu.counts().tolist())
        self.assertEqual(cpu.axis_counts(), gpu.axis_counts())
        self.assertIn(gpu.backend, ("cuda", "cuda-emulated"))
        return cpu

    def transform(self, values, op, **kw):
        """the CPU transform, checked equal to the CUDA one"""
        cpu = ml.transform(values, op, **kw)
        gpu = ml.transform(values, op, backend="cuda", **kw)
        self.assertEqual(list(cpu["values"]), list(gpu["values"]), op)
        self.assertIn(gpu["backend"], ("cuda", "cuda-emulated"))
        return cpu["values"]

    def check_hist(self, h, groups, k, nulls, what=""):
        cells, _ = expected_cells(groups, k, nulls)
        self.assertEqual(h.counts().tolist(), cells, what)
        total = sum(groups.values())
        unknown = sum(n for vals, n in groups.items() if any(v is None for v in vals))   # either convention
        self.assertEqual((h.points, h.unknown, h.complete), (total, unknown, total - unknown), what)
        for i, a in enumerate(h.axis_counts()):
            want = dict(true=0, false=0, unknown=0)
            for vals, n in groups.items():
                want["unknown" if vals[i] is None else "true" if vals[i] else "false"] += n
            self.assertEqual(a, want, (what, i))

    # ================================================================== LATTICE-001
    def test_lattice001_cells_match_enumerated_predicate_outcomes(self):
        rng = random.Random(1101)
        for k0 in (1, 2, 3, 5, 8):
            masks = mb.Masks.assignments(k0)
            for trial in range(3):
                naxes = rng.randint(1, 6)
                axes, exprs = [], []
                for _ in range(naxes):
                    if rng.random() < 0.4:
                        i = rng.randrange(k0)
                        axes.append(i)
                        exprs.append(mb.var(i))
                    else:
                        e = random_expr(rng, k0, size=2 + 2 * k0, share=0.5)
                        axes.append(mb.Program(e, k0))
                        exprs.append(e)
                groups = {}
                for a in range(1 << k0):                     # every assignment, every axis by scalar Kleene logic
                    cond = [bool((a >> i) & 1) for i in range(k0)]
                    vals = tuple(scalar(e, cond) for e in exprs)
                    groups[vals] = groups.get(vals, 0) + 1
                for nulls in ("apart", "three"):
                    h = self.build(masks, axes, nulls)
                    self.check_hist(h, groups, naxes, nulls, (k0, trial, nulls))
                    self.assertEqual(h.kind, "assignments")
                    self.assertEqual(sum(h.counts().tolist()), h.complete if nulls == "apart" else 1 << k0)
                    self.assertEqual(h.info["occupied"], sum(1 for c in h.counts().tolist() if c))
        # the conditions themselves over the 2^k assignments: every cell exactly once
        for k in (0, 1, 4, 7):
            h = self.build(mb.Masks.assignments(k))
            self.assertEqual(h.counts().tolist(), [1] * (1 << k))

    def test_lattice001_row_cells_match_duckdb_group_by_for_k_3_6_10_12(self):
        nrng = np.random.default_rng(1102)
        n = 1500
        db = dataset(nrng, n)
        t = db["t"]
        for sel in (np.ones(n, bool), nrng.random(n) < 0.35):
            scope = mbk.Scope.from_table(t, ["x", "y", "d", "b"], rows=sel)
            where = duck_where(db, sel)
            for k in (3, 6, 10, 12):
                specs = specs_of(k)
                masks = mb.Masks.from_scope(scope, [condition(s) for s in specs])
                groups = duck_groups(db, [cond_sql(s) for s in specs], where)
                for nulls in ("apart", "three"):
                    h = self.build(masks, nulls=nulls)
                    self.check_hist(h, groups, k, nulls, (k, nulls))
                    self.assertEqual(h.kind, "rows")
                    self.assertEqual(h.points, int(sel.sum()))
                h3, h2 = ml.Histogram.build(masks, nulls="three"), ml.Histogram.build(masks)
                self.assertEqual(h3.vector("complete").tolist(), h2.counts().tolist())

    def test_lattice001_sixteen_and_twenty_axes_within_budget(self):
        nrng = np.random.default_rng(1107)
        n = 4000
        cols = {f"u{j}": nrng.integers(0, 1 << 10, n) for j in range(2)}
        db = torbles.from_ddl("CREATE TABLE t (u0 BIGINT, u1 BIGINT)", {"t": pd.DataFrame(cols)})
        sel = nrng.random(n) < 0.6
        scope = mbk.Scope.from_table(db["t"], ["u0", "u1"], rows=sel)
        for k in (16, 20):
            pairs = [(f"u{i % 2}", i // 2) for i in range(k)]   # bit b of a column: independent conditions
            masks = mb.Masks.from_scope(scope, [mb.condition(c, "bit", b) for c, b in pairs])
            h = self.build(masks)
            ids = np.zeros(n, dtype=np.int64)                  # the cell of every row, by NumPy
            for i, (c, b) in enumerate(pairs):
                ids |= ((cols[c] >> b) & 1).astype(np.int64) << i
            want = np.bincount(ids[sel], minlength=1 << k)
            self.assertEqual(h.counts().tolist(), want.tolist(), k)
            self.assertEqual((h.info["cells"], h.info["occupied"]), (1 << k, int((want > 0).sum())))

    def test_lattice001_condition_order_program_axes_and_shared_programs(self):
        nrng = np.random.default_rng(1103)
        prng = random.Random(1103)
        n = 900
        db = dataset(nrng, n)
        sel = nrng.random(n) < 0.8
        scope = mbk.Scope.from_table(db["t"], ["x", "y", "d", "b"], rows=sel)
        specs = specs_of(5)
        csql = [cond_sql(s) for s in specs]
        masks = mb.Masks.from_scope(scope, [condition(s) for s in specs])
        where = duck_where(db, sel)
        base = ml.Histogram.build(masks, [0, 1, 2], nulls="three")
        perm = self.build(masks, [2, 0, 1], nulls="three")        # axis i is digit i, in the given order
        for vals, cnt in base.table():
            self.assertEqual(perm.cell((vals[2], vals[0], vals[1])), cnt)
        self.assertEqual(perm.axes, (("condition", 2), ("condition", 0), ("condition", 1)))
        rep = self.build(masks, [3, 3])                            # a repeated condition: the mixed cells are empty
        self.assertEqual((rep.cell((True, False)), rep.cell((False, True))), (0, 0))
        exprs = [random_expr(prng, 5, size=10, share=0.6) for _ in range(3)]
        shared = mb.Program(exprs[0], 5)
        axes = [shared, 4, mb.Program(exprs[1], 5), exprs[2], shared]   # a Program object read twice: one program
        for nulls in ("apart", "three"):
            h = self.build(masks, axes, nulls)
            asql = [expr_sql(exprs[0], csql), csql[4], expr_sql(exprs[1], csql), expr_sql(exprs[2], csql),
                    expr_sql(exprs[0], csql)]
            self.check_hist(h, duck_groups(db, asql, where), 5, nulls, nulls)
            self.assertEqual(len(h._programs), 3)
            self.assertEqual([a[0] for a in h.axes], ["program", "condition", "program", "program", "program"])
        with self.assertRaises(DomainError):
            ml.Histogram.build(masks, [mb.Program(mb.var(0), 4)])
        with self.assertRaises(ValueError):
            ml.Histogram.build(masks, [5])

    def test_lattice001_null_conventions_never_count_unknown_as_false(self):
        frame = pd.DataFrame({"x": obj([1, 5, None, 9, None, 2, 7, 7]), "y": obj([None, 3, 4, None, None, 8, 1, 9])})
        db = torbles.from_ddl("CREATE TABLE t (x BIGINT, y BIGINT)", {"t": frame})
        scope = mbk.Scope.from_table(db["t"], ["x", "y"])
        masks = mb.Masks.from_scope(scope, [mb.condition("x", ">", 3), mb.condition("y", ">", 3)])
        apart, three = self.build(masks), self.build(masks, nulls="three")
        # x > 3: F T U T U F T T ; y > 3: U F T U U T F T  (scalar values per row)
        rows = [(False, None), (True, False), (None, True), (True, None), (None, None), (False, True), (True, False),
                (True, True)]
        self.assertEqual(apart.counts().tolist(), [0, 2, 1, 1])  # only the 4 complete rows; FF is empty
        self.assertEqual((apart.complete, apart.unknown), (4, 4))
        self.assertEqual(apart.axis_counts(), [dict(true=4, false=2, unknown=2), dict(true=3, false=2, unknown=3)])
        self.assertEqual(dict(three.table()), {r: rows.count(r) for r in set(rows)})
        self.assertEqual(three.cell((False, None)), 1)            # UNKNOWN is its own digit, never FALSE
        self.assertEqual(three.cell((False, False)), 0)
        self.assertEqual(three.vector("complete").tolist(), apart.counts().tolist())
        self.assertEqual(three.vector("true_sets").tolist(), [2, 3, 2, 1])
        self.assertEqual(three.vector("false_sets").tolist(), [4, 2, 2, 0])
        self.assertEqual(three.vector("unknown_sets").tolist(), [4, 1, 2, 1])
        for projection in ("true_sets", "false_sets", "unknown_sets"):
            with self.assertRaises(DomainError):                   # an 'apart' histogram did not record them
                apart.vector(projection)
            with self.assertRaises(DomainError):
                ml.counts(apart, [[0]], projection=projection)
        with self.assertRaises(ValueError):
            apart.cell((None, True))
        # SQL WHERE semantics of the TRUE sets: AND / OR over all rows, NULLs included
        where = "TRUE"
        c = ['("x" > 3)', '("y" > 3)']
        r = ml.counts(three, [[0, 1]], projection="true_sets")[0]
        self.assertEqual((r["all"], r["any"]), (duck_count(db, f"{c[0]} AND {c[1]}", where),
                                                duck_count(db, f"{c[0]} OR {c[1]}", where)))
        self.assertEqual(r["none"], 8 - r["any"])
        rc = ml.counts(apart, [[0, 1]])[0]                         # the complete rows only
        self.assertEqual((rc["all"], rc["any"], rc["none"]), (1, 4, 0))

    def test_lattice001_from_counts_merge_and_consistency(self):
        nrng = np.random.default_rng(1104)
        n = 1200
        db = dataset(nrng, n)
        t = db["t"]
        part = nrng.random(n) < 0.5
        specs = specs_of(6)
        conds = [condition(s) for s in specs]

        def hist(sel, nulls):
            return ml.Histogram.build(mb.Masks.from_scope(mbk.Scope.from_table(t, ["x", "y", "d", "b"], rows=sel),
                                                          conds), nulls=nulls)
        for nulls in ("apart", "three"):
            a, b, whole = hist(part, nulls), hist(~part, nulls), hist(np.ones(n, bool), nulls)
            merged = a.merge(b)
            self.assertTrue(merged.equal(whole), nulls)
            self.assertEqual(merged.kind, "rows")
            again = ml.Histogram.from_counts(6, whole.counts().tolist(), nulls=nulls, points=whole.points,
                                             axis_counts=[(c["true"], c["false"], c["unknown"])
                                                          for c in whole.axis_counts()])
            self.assertTrue(again.equal(whole))
            self.assertEqual(again.kind, "counts")
        three = hist(np.ones(n, bool), "three")
        bad = three.counts().tolist()
        bad[0] += 1
        with self.assertRaises(ValueError):                        # cells that do not sum to the points
            ml.Histogram.from_counts(6, bad, nulls="three", points=three.points)
        with self.assertRaises(ValueError):                        # axis counts the cells do not give
            ml.Histogram.from_counts(6, three.counts().tolist(), nulls="three",
                                     axis_counts=[(1, 0, three.points - 1)] * 6)
        apart = hist(np.ones(n, bool), "apart")
        with self.assertRaises(ValueError):                        # unknown points without axis counts
            ml.Histogram.from_counts(6, apart.counts().tolist(), points=apart.points)
        ax = [(c["true"], c["false"], c["unknown"]) for c in apart.axis_counts()]
        with self.assertRaises(ValueError):                        # fewer unknown points than one axis has
            ml.Histogram.from_counts(6, apart.counts().tolist(), points=apart.complete + 1, axis_counts=ax)
        with self.assertRaises(DomainError):
            apart.merge(hist(np.ones(n, bool), "three"))
        plain = ml.Histogram.from_counts(2, [3, 0, 1, 2])          # points default: the native sum
        self.assertEqual((plain.points, plain.unknown), (6, 0))
        self.assertEqual(plain.axis_counts(), [dict(true=2, false=4, unknown=0), dict(true=3, false=3, unknown=0)])

    def test_lattice001_budgets_at_the_limit_and_one_past(self):
        nrng = np.random.default_rng(1105)
        db = dataset(nrng, 800)
        scope = mbk.Scope.from_table(db["t"], ["x", "y", "d", "b"])
        masks = mb.Masks.from_scope(scope, [condition(s) for s in specs_of(10)])
        need = ml.histogram_need(masks)
        self.assertEqual((need["variables"], need["states"], need["output"]), (10, 1024, 1024))
        ml.Histogram.build(masks, budget=Budget(states=1024))
        with self.assertRaises(AlgebraBudgetError) as ctx:
            ml.Histogram.build(masks, budget=Budget(states=1023))
        self.assertEqual((ctx.exception.resource, ctx.exception.need, ctx.exception.limit), ("states", 1024, 1023))
        need3 = ml.histogram_need(masks, [0, 1, 2, 3, 4, 5], nulls="three")
        self.assertEqual(need3["states"], 729)
        for backend in ("cpu", "cuda"):
            ml.Histogram.build(masks, [0, 1, 2, 3, 4, 5], nulls="three", backend=backend, budget=Budget(states=729))
            with self.assertRaises(AlgebraBudgetError) as ctx:
                ml.Histogram.build(masks, [0, 1, 2, 3, 4, 5], nulls="three", backend=backend,
                                   budget=Budget(states=728))
            self.assertEqual((ctx.exception.resource, ctx.exception.need), ("states", 729))
            ml.Histogram.build(masks, backend=backend, budget=Budget(variables=10))
            with self.assertRaises(AlgebraBudgetError) as ctx:     # the condition budget
                ml.Histogram.build(masks, backend=backend, budget=Budget(variables=9))
            self.assertEqual(ctx.exception.resource, "variables")
            nb = ml.histogram_need(masks, backend=backend)["bytes"]
            ml.Histogram.build(masks, backend=backend, budget=Budget(bytes=nb))
            with self.assertRaises(AlgebraBudgetError) as ctx:
                ml.Histogram.build(masks, backend=backend, budget=Budget(bytes=nb - 1))
            self.assertEqual((ctx.exception.resource, ctx.exception.need), ("bytes", nb))
        prog = mb.Program((mb.var(0) & mb.var(1)) | mb.var(2), 10)
        nodes = prog.info["nodes"]
        self.assertEqual(ml.histogram_need(masks, [prog, 3])["circuit"], nodes)
        ml.Histogram.build(masks, [prog, 3], budget=Budget(circuit=nodes))
        with self.assertRaises(AlgebraBudgetError):
            ml.Histogram.build(masks, [prog, 3], budget=Budget(circuit=nodes - 1))
        uniform = mb.Masks.assignments(12, budget=Budget(support=4096))
        ml.Histogram.build(uniform, [0, 5], budget=Budget(support=4096))
        with self.assertRaises(AlgebraBudgetError) as ctx:         # the enumerated points of the assignments
            ml.Histogram.build(uniform, [0, 5], budget=Budget(support=4095))
        self.assertEqual(ctx.exception.resource, "support")
        for backend in ("cpu", "cuda"):                            # the device need covers the adopted histogram
            nb = ml.histogram_need(uniform, [0, 5], backend=backend)["bytes"]
            ml.Histogram.build(uniform, [0, 5], backend=backend, budget=Budget(support=4096, bytes=nb))
            with self.assertRaises(AlgebraBudgetError) as ctx:
                ml.Histogram.build(uniform, [0, 5], backend=backend, budget=Budget(support=4096, bytes=nb - 1))
            self.assertEqual((ctx.exception.resource, ctx.exception.need), ("bytes", nb))
        wide = mb.Masks.from_scope(scope, [condition(specs_of(1)[0])] * 31)
        huge = Budget(variables=64, states=(1 << 64) - 1, output=(1 << 64) - 1, bytes=(1 << 64) - 1)
        with self.assertRaises(AlgebraBudgetError):                # 2^31 cells: refused by the default budget
            ml.Histogram.build(wide)
        for backend in ("cpu", "cuda"):
            with self.assertRaises(UnsupportedError):              # past the representation limit of 30 axes
                ml.Histogram.build(wide, backend=backend, budget=huge)
        ml.Histogram.from_counts(10, [0] * 1024, budget=Budget(states=1024))
        with self.assertRaises(AlgebraBudgetError):
            ml.Histogram.from_counts(10, [0] * 1024, budget=Budget(states=1023))

    def test_lattice001_stale_scope_is_refused(self):
        nrng = np.random.default_rng(1106)
        db = dataset(nrng, 300)
        t = db["t"]
        scope = mbk.Scope.from_table(t, ["x", "y"])
        masks = mb.Masks.from_scope(scope, [mb.condition("x", ">", 0), mb.condition("y", "<", 0)])
        h = ml.Histogram.build(masks)
        h.counts()
        t.flip(0, 0)                                               # a mutation: a new revision
        for backend in ("cpu", "cuda"):
            with self.assertRaises(StaleStateError):
                ml.Histogram.build(masks, backend=backend)
        with self.assertRaises(StaleStateError):
            h.counts()
        with self.assertRaises(StaleStateError):
            ml.zeta(h)
        with self.assertRaises(StaleStateError):
            ml.counts(h, [[0]])

    def test_lattice001_ctypes_structures_are_registered(self):
        report = mh.abi_check()
        mine = {r["native"] for r in report["checked"] if r["structure"].startswith("torbles.math_lattice.")}
        self.assertEqual(mine, {"tr_lattice_axis", "tr_lattice_hist_info", "tr_lattice_count"})
        cuda = {r["native"] for r in report["checked"] if r["library"] == "cuda"
                and r["structure"].startswith("torbles.math_lattice.")}
        self.assertEqual(cuda, {"tr_lattice_axis"})
        self.assertTrue(all(r["match"] for r in report["checked"]))

    # ================================================================== LATTICE-002
    def test_lattice002_zeta_mobius_round_trips_match_direct_set_counts(self):
        nrng = np.random.default_rng(1201)
        n = 1200
        db = dataset(nrng, n)
        sel = nrng.random(n) < 0.7
        scope = mbk.Scope.from_table(db["t"], ["x", "y", "d", "b"], rows=sel)
        where = duck_where(db, sel)
        for k in (3, 5, 8):
            specs = specs_of(k)
            csql = [cond_sql(s) for s in specs]
            masks = mb.Masks.from_scope(scope, [condition(s) for s in specs])
            three = ml.Histogram.build(masks, nulls="three")
            complete = " AND ".join(f"({c}) IS NOT NULL" for c in csql)
            subsets = range(1 << k) if k <= 5 else random.Random(k).sample(range(1 << k), 24)
            for projection in ("true_sets", "complete"):
                h = three.vector(projection).tolist()
                up = self.transform(three, "zeta_up", projection=projection)
                down = self.transform(three, "zeta_down", projection=projection)
                self.assertEqual(ml.transform(three, "zeta_up", carrier="bigint", projection=projection)["values"], up)
                base = where if projection == "true_sets" else f"{where} AND {complete}"
                for S in subsets:
                    inside = [csql[i] for i in range(k) if (S >> i) & 1]
                    outside = [csql[i] for i in range(k) if not (S >> i) & 1]
                    # Z+(S): TRUE on at least S (SQL: WHERE c_i AND ...); Z-(S): TRUE on nothing outside S
                    self.assertEqual(up[S], duck_count(db, " AND ".join(inside) or "TRUE", base), (k, projection, S))
                    self.assertEqual(down[S], duck_count(db, " AND ".join(f"NOT COALESCE({c}, FALSE)"
                                                                          for c in outside) or "TRUE", base))
                # Moebius inversion gives the exclusive regions (the cells) back, both directions
                self.assertEqual(list(self.transform(up, "mobius_up")), h)
                self.assertEqual(list(self.transform(down, "mobius_down")), h)
                self.assertEqual(up, zeta_oracle(h, k, "zeta_up"))
                self.assertEqual(down, zeta_oracle(h, k, "zeta_down"))
            groups = duck_groups(db, csql, where)                  # the exclusive cells by GROUP BY
            self.assertEqual(three.vector("complete").tolist(), expected_cells(groups, k, "apart")[0])
        # signed set functions (int64 values): every transform against its definition, both round trips
        rng = np.random.default_rng(1202)
        for k in (0, 1, 2, 4, 7):
            v = [int(x) for x in rng.integers(-10 ** 12, 10 ** 12, 1 << k)]
            for op in ("zeta_up", "zeta_down", "mobius_up", "mobius_down"):
                got = self.transform(v, op)
                self.assertEqual(got, zeta_oracle(v, k, op), (k, op))
                inverse = {"zeta_up": "mobius_up", "zeta_down": "mobius_down", "mobius_up": "zeta_up",
                           "mobius_down": "zeta_down"}[op]
                self.assertEqual(self.transform(got, inverse), v)
            arr = np.array(v, dtype=np.int64)
            self.assertEqual(list(self.transform(arr, "zeta_up", carrier="i64")), zeta_oracle(v, k, "zeta_up"))

    def test_lattice002_inclusion_exclusion_counts_match_direct_counts(self):
        nrng = np.random.default_rng(1203)
        n = 1000
        db = dataset(nrng, n, nulls=0.2)
        sel = nrng.random(n) < 0.8
        scope = mbk.Scope.from_table(db["t"], ["x", "y", "d", "b"], rows=sel)
        where = duck_where(db, sel)
        k = 6
        specs = specs_of(k)
        csql = [cond_sql(s) for s in specs]
        masks = mb.Masks.from_scope(scope, [condition(s) for s in specs])
        three = ml.Histogram.build(masks, nulls="three")
        complete = " AND ".join(f"({c}) IS NOT NULL" for c in csql)
        sets = [[0], [1, 2], [0, 3, 5], [0, 1, 2, 3, 4, 5], [], [2, 4]]
        for projection, base in (("true_sets", where), ("complete", f"{where} AND {complete}")):
            N = duck_count(db, "TRUE", base)
            h = three.vector(projection).tolist()
            got = ml.counts(three, sets, projection=projection)
            for S, r in zip(sets, got):
                self.assertEqual(r["conditions"], tuple(S))
                self.assertEqual(r["all"], duck_count(db, " AND ".join(csql[i] for i in S) or "TRUE", base))
                self.assertEqual(r["any"], duck_count(db, " OR ".join(csql[i] for i in S) or "FALSE", base))
                self.assertEqual(r["none"], N - r["any"])
                self.assertEqual(r["exactly"], h[sum(1 << i for i in S)])
                self.assertEqual(r["terms"], (1 << len(S)) - 1)
                # the explicit inclusion-exclusion over the SQL intersections, and its Bonferroni peak
                partial, peak = 0, 0
                for o in range(1, len(S) + 1):
                    partial += (-1) ** (o + 1) * sum(
                        duck_count(db, " AND ".join(csql[i] for i in T), base) for T in itertools.combinations(S, o))
                    peak = max(peak, abs(partial))
                self.assertEqual((r["any"], r["peak"]), (partial, peak))

    def test_lattice002_coefficients_and_bounded_interactions(self):
        rng = np.random.default_rng(1204)
        for k in (3, 6, 9):
            v = [int(x) for x in rng.integers(-1000, 1000, 1 << k)]
            sets = [sorted(random.Random(k * 7 + j).sample(range(k), j % (k + 1))) for j in range(12)]
            masks = [sum(1 << i for i in S) for S in sets]
            for op in ("zeta_up", "zeta_down", "mobius_up", "mobius_down"):
                full = self.transform(v, op)
                direct = ml.coefficients(v, sets, op)
                self.assertEqual(direct, [full[m] for m in masks], (k, op))
            for order in (0, 1, 2, 3):
                got = ml.interactions(v, order)
                want = [(T, sum((-1) ** (len(T) - len(A)) * v[sum(1 << i for i in A)]
                                for r in range(len(T) + 1) for A in itertools.combinations(T, r)))
                        for m in range(order + 1) for T in by_mask(k, m)]
                self.assertEqual(got, want, (k, order))
        self.assertEqual(ml.order_sets(5, 3, 2), [T for m in (2, 3) for T in by_mask(5, m)])
        # the coverage game: the Harsanyi dividends of the union counts are (-1)^(|T|+1) times the intersections
        nrng = np.random.default_rng(1205)
        db = dataset(nrng, 700, nulls=0.0)
        scope = mbk.Scope.from_table(db["t"], ["x", "y", "d", "b"])
        specs = specs_of(5)
        h = ml.Histogram.build(mb.Masks.from_scope(scope, [condition(s) for s in specs]))
        union = [r["any"] for r in ml.counts(h, [[i for i in range(5) if (S >> i) & 1] for S in range(32)])]
        inter = ml.zeta(h)["values"]
        for T, m in ml.interactions(union, 5):
            mask = sum(1 << i for i in T)
            self.assertEqual(m, 0 if not T else (-1) ** (len(T) + 1) * inter[mask], T)

    def test_lattice002_signed_intermediates_use_a_compatible_carrier_and_never_wrap(self):
        k = 8
        top = (1 << 63) - 1
        v = [top] * (1 << k)
        exact = zeta_oracle(v, k, "zeta_up")                      # up to 2^8 (2^63 - 1): past int64
        self.assertGreater(max(exact), top)
        self.assertEqual(self.transform(v, "zeta_up"), exact)      # 128-bit cells: exact
        for backend in ("cpu", "cuda"):
            with self.assertRaises(OverflowError):                 # the int64 carrier refuses, never wraps
                ml.transform(v, "zeta_up", carrier="i64", backend=backend)
            with self.assertRaises(OverflowError):                 # an input past int64 with the int64 carrier
                ml.transform([1 << 63] + [0] * 3, "zeta_up", carrier="i64", backend=backend)
        with self.assertRaises(OverflowError):
            ml.coefficients(v, [[]], "zeta_up", carrier="i64")
        self.assertEqual(ml.coefficients(v, [[]], "zeta_up"), [exact[0]])
        # 128-bit inputs whose transform leaves the 128-bit range: refused on both backends, exact as big integers
        w = [(1 << 126) + i for i in range(4)]
        for op in ("zeta_up", "zeta_down", "mobius_up"):
            for backend in ("cpu", "cuda"):
                with self.assertRaises(OverflowError):
                    ml.transform(w if op != "mobius_up" else [-(1 << 126)] + w[1:], op, backend=backend)
        big = ml.transform(w, "zeta_up", carrier="bigint")
        self.assertEqual(big["values"], zeta_oracle(w, 2, "zeta_up"))
        self.assertEqual(big["carrier"], "bigint")
        with self.assertRaises(OverflowError):
            ml.transform([1 << 127], "zeta_up")                    # does not fit the 128-bit carrier at all
        huge = [(1 << 200) + 3 * i - (1 << 199) * (i % 2) for i in range(16)]
        for op in ("zeta_up", "mobius_down"):
            got = ml.transform(huge, op, carrier="bigint")["values"]
            self.assertEqual(got, zeta_oracle(huge, 4, op))
            inverse = {"zeta_up": "mobius_up", "mobius_down": "zeta_down"}[op]
            self.assertEqual(ml.transform(got, inverse, carrier="bigint")["values"], huge)
        with self.assertRaises(UnsupportedError):
            ml.transform(huge, "zeta_up", carrier="bigint", backend="cuda")
        # inclusion-exclusion: Bonferroni partial sums past int64 while the count fits, exact in 128 bits
        k = 16
        cells = [0] * (1 << k)
        cells[-1] = 1 << 62                                        # 2^62 points TRUE on every axis
        h = ml.Histogram.from_counts(k, cells)
        r = ml.counts(h, [list(range(k))])[0]
        partial, peak = 0, 0
        for o in range(1, k + 1):
            partial += (-1) ** (o + 1) * comb(k, o) * (1 << 62)
            peak = max(peak, abs(partial))
        self.assertEqual((r["all"], r["any"], r["none"], r["peak"]), (1 << 62, 1 << 62, 0, peak))
        self.assertGreater(r["peak"], top)

    def test_lattice002_budgets_at_the_limit_and_one_past(self):
        k = 10
        v = list(range(1 << k))
        need = ml.transform_need(k, "zeta_up")
        self.assertEqual((need["variables"], need["tensor"], need["iterations"], need["bytes"]), (10, 1024, 10, 16384))
        for backend in ("cpu", "cuda"):
            for resource in ("tensor", "iterations", "bytes"):
                ml.transform(v, "zeta_up", backend=backend, budget=Budget(**{resource: need[resource]}))
                with self.assertRaises(AlgebraBudgetError) as ctx:
                    ml.transform(v, "zeta_up", backend=backend, budget=Budget(**{resource: need[resource] - 1}))
                self.assertEqual((ctx.exception.resource, ctx.exception.need), (resource, need[resource]))
        nb = ml.transform_need(k, "zeta_up", "bigint")["bytes"]
        self.assertEqual(nb, 8 * 1024)
        with self.assertRaises(AlgebraBudgetError):
            ml.transform(v, "zeta_up", carrier="bigint", budget=Budget(bytes=nb - 1))
        sets = [[0], [1, 2, 3], []]
        cn = ml.coefficients_need(v, sets, "zeta_up")
        self.assertEqual(cn["tensor"], (1 << 9) + (1 << 7) + (1 << 10))
        ml.coefficients(v, sets, "zeta_up", budget=Budget(tensor=cn["tensor"]))
        with self.assertRaises(AlgebraBudgetError):
            ml.coefficients(v, sets, "zeta_up", budget=Budget(tensor=cn["tensor"] - 1))
        self.assertEqual(ml.coefficients_need(v, sets, "zeta_down")["tensor"], 2 + 8 + 1)
        self.assertEqual(len(ml.order_sets(10, 2, budget=Budget(output=56))), 56)
        with self.assertRaises(AlgebraBudgetError):
            ml.order_sets(10, 2, budget=Budget(output=55))
        with self.assertRaises(UnsupportedError):
            ml.order_sets(31, 1, budget=Budget(variables=64))
        h = ml.Histogram.from_counts(6, list(range(64)))
        qs = [[0, 1, 2], [5]]
        need = ml.counts_need(h, qs)
        self.assertEqual((need["degree"], need["tensor"], need["output"]),
                         (3, 64 + (64 + 3 * 8 + 8) + (64 + 2 + 2), 2))
        for resource in ("degree", "tensor", "output", "bytes"):
            ml.counts(h, qs, budget=Budget(**{resource: need[resource]}))
            with self.assertRaises(AlgebraBudgetError) as ctx:
                ml.counts(h, qs, budget=Budget(**{resource: need[resource] - 1}))
            self.assertEqual(ctx.exception.resource, resource)
        with self.assertRaises(ValueError):
            ml.transform([1, 2, 3], "zeta_up")                     # not 2^k values
        with self.assertRaises(ValueError):
            ml.counts(h, [[6]])


if __name__ == "__main__":
    unittest.main()
