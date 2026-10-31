# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""UNC-001: intervals, affine forms and congruence reasoning (certified abstract states), CPU and CUDA.

Oracles (never the native code): every concrete value of small abstract inputs enumerated with Python Fractions and
SQL three-valued logic written here; the affine semantics re-checked node by node with exact noise values; DuckDB
evaluating the original predicates over the same engine tables. Every native result is also computed by the CUDA
kernels (emulated here, threads scrambled) and compared bit for bit. Seeds are fixed.
"""
import itertools
import os
import random
import subprocess
import sys
import unittest
from fractions import Fraction

import numpy as np
import pandas as pd

import torbles
from torbles import math_bank as mbk
from torbles import math_certificates as mc
from torbles import math_unc as mu
from torbles.math_budgets import Budget
from torbles.math_domains import DomainError
from torbles.math_errors import StaleStateError, UnsupportedError
from torbles.mathematics import AlgebraBudgetError

NULL = None


# ------------------------------------------------------------------ the oracle: concrete SQL semantics in Python
def concretize(spec, limit=64):
    """the values one column of a group can take (None: NULL), from an input dict; a class with more than `limit`
    members gives its extremes and members around 0 and the middle (a necessary check only: see sampled())"""
    rows, present = spec["rows"], spec.get("present", spec["rows"])
    vals = []
    if present:
        lo, hi, m, s = spec["min"], spec["max"], spec.get("modulus", 0), spec.get("scale", 0)
        if m == 0:
            logical = [lo]
        elif (hi - lo) // m + 1 <= limit:
            logical = list(range(lo, hi + 1, m))
        else:
            mid = lo + ((hi - lo) // (2 * m)) * m
            zero = lo + ((0 - lo) // m) * m if lo <= 0 <= hi else lo
            logical = sorted({lo + i * m for i in range(4)} | {hi - i * m for i in range(4)} |
                             {mid + i * m for i in range(-2, 3)} | {zero + i * m for i in range(-2, 3) if lo <= zero + i * m <= hi})
        vals = [Fraction(v, 10 ** s) for v in logical]
    if present < rows:
        vals.append(NULL)
    return vals


def concrete(e, env):
    """the SQL value of an expression (Fraction / True / False / None)"""
    op = e.op
    if op == "col":
        return env[e.name]
    if op == "const":
        return e.k
    a = [concrete(x, env) for x in e.args]
    if op == "and":
        if a[0] is False or a[1] is False:
            return False
        return None if a[0] is None or a[1] is None else True
    if op == "or":
        if a[0] is True or a[1] is True:
            return True
        return None if a[0] is None or a[1] is None else False
    if op == "not":
        return None if a[0] is None else (not a[0])
    if op == "is_null":
        return a[0] is None
    if op == "not_null":
        return a[0] is not None
    if any(x is None for x in a):
        return None
    if op == "add": return a[0] + a[1]
    if op == "sub": return a[0] - a[1]
    if op == "mul": return a[0] * a[1]
    if op == "neg": return -a[0]
    if op == "div": return a[0] / e.k
    if op == "min": return min(a[0], a[1])
    if op == "max": return max(a[0], a[1])
    if op == "abs": return abs(a[0])
    if op == "sqr": return a[0] * a[0]
    if op == "mod": return a[0] % e.k                 # x - k floor(x / k)
    if op == "lt": return a[0] < a[1]
    if op == "le": return a[0] <= a[1]
    if op == "eq": return a[0] == a[1]
    if op == "ne": return a[0] != a[1]
    if op == "ge": return a[0] >= a[1]
    if op == "gt": return a[0] > a[1]
    raise ValueError(op)


def envs(names, specs):
    for combo in itertools.product(*[concretize(s) for s in specs]):
        yield dict(zip(names, combo))


def frac(x):
    return Fraction(x) if not isinstance(x, Fraction) else x


def in_class(v, cong):
    r, m = cong
    return v == r if m == 0 else ((v - r) / m).denominator == 1


class Oracle(unittest.TestCase):
    """checks of one abstract value against one concrete value"""

    def encloses(self, res, v, what):
        if res["kind"] == "boolean":
            key = {True: "true", False: "false", None: "null"}[v]
            self.assertIn(key, res["outcomes"], what)
            return
        if v is None:
            self.assertTrue(res["null"]["maybe_null"], what)
            return
        self.assertTrue(res["null"]["maybe_value"], what)
        lo, hi = res["interval"]
        if lo is not None:
            self.assertLessEqual(frac(lo), v, what)
        if hi is not None:
            self.assertGreaterEqual(frac(hi), v, what)
        if res["congruence"] is not None:
            self.assertTrue(in_class(v, res["congruence"]), (what, res["congruence"], v))
        if res["zero"] == "zero":
            self.assertEqual(v, 0, what)
        if res["zero"] == "nonzero":
            self.assertNotEqual(v, 0, what)
        if res.get("form"):
            f = res["form"]
            radius = frac(f["radius"]) if "radius" in f else sum(abs(frac(c)) for _, _, c in f["terms"])
            self.assertLessEqual(abs(v - frac(f["center"])), radius, what)

    def affine_residuals(self, trace, prog, env, what):
        """exact noise values node by node: column symbols from the column nodes, each node's own symbol from its
        residual (|residual| <= its coefficient); every form must hold with the values already defined"""
        eps = {}
        for k, (expr, v) in enumerate(zip(prog.exprs, trace)):
            if v["kind"] != "numeric" or v["form"] is None:
                continue
            value = concrete(expr, env)
            if value is None:
                continue
            terms = v["form"]["terms"]
            if expr.op == "col":
                j = prog.columns.index(expr.name)
                if terms:
                    (sym, _, coef), = terms
                    e = (value - frac(v["form"]["center"])) / frac(coef)
                    self.assertLessEqual(abs(e), 1, (what, "column symbol out of [-1, 1]"))
                    eps[sym] = e
                else:
                    self.assertEqual(value, frac(v["form"]["center"]), what)
                    eps[j] = Fraction(0)
                continue
            own = mu.FRESH + k
            lin = frac(v["form"]["center"])
            own_coef = Fraction(0)
            for sym, _, coef in terms:
                if sym == own:
                    own_coef = frac(coef)
                    continue
                self.assertIn(sym, eps, (what, k, "a symbol before its definition", sym))
                lin += frac(coef) * eps[sym]
            residual = value - lin
            self.assertLessEqual(abs(residual), abs(own_coef), (what, k, expr, value, lin, own_coef))
            eps[own] = residual / own_coef if own_coef else Fraction(0)


SPECS = [   # one column's inputs: integer and decimal ranges, classes, constants, NULLs, all-NULL
    dict(rows=4, present=4, min=-3, max=4, modulus=1),
    dict(rows=4, present=4, min=-6, max=6, modulus=3),
    dict(rows=4, present=4, min=2, max=11, modulus=3),
    dict(rows=4, present=4, min=-9, max=-1, modulus=2),
    dict(rows=4, present=4, min=5, max=5, modulus=0),
    dict(rows=4, present=4, min=0, max=0, modulus=0),
    dict(rows=4, present=3, min=-2, max=3, modulus=1),
    dict(rows=4, present=4, min=-15, max=25, modulus=10, scale=1),
    dict(rows=4, present=4, min=-150, max=75, modulus=75, scale=2),
    dict(rows=4, present=0),
]
CARRIERS = [dict(carrier="exact"), dict(carrier="binary64"), dict(carrier="binary64", precision=6),
            dict(carrier="exact", rank=1), dict(carrier="exact", budget=Fraction(1, 2))]


def unary_exprs(x):
    return [-x, abs(x), x.sqr(), x / 3, x / Fraction(-2, 5), x % 4, x % Fraction(3, 2), x.is_null(), x.not_null(),
            x * 0, x + Fraction(1, 3)]


def binary_exprs(x, y):
    return [x + y, x - y, x * y, mu.min_(x, y), mu.max_(x, y), x < y, x <= y, x.eq(y), x.ne(y), x >= y, x > y,
            (x < 1) & (y > 0), (x < 1) | (y > 0), ~(x.eq(y)), (x * y) - (y * x), x * x - x * y]


class Unc(Oracle):
    def run_both(self, state, prog):
        cpu = state.evaluate(prog)
        gpu = state.evaluate(prog, backend="cuda")
        self.assertIn(gpu.backend, ("cuda", "cuda-emulated"))
        self.assertTrue(cpu.same(gpu), prog)
        return cpu

    # ------------------------------------------------------------------ acceptance 1: exhaustive enclosure
    def test_unc001_every_operation_encloses_exhaustive_small_values(self):
        x, y = mu.column("x"), mu.column("y")
        checked = 0
        for opts in CARRIERS:
            for spec in SPECS:
                state = mu.State.from_inputs([[spec]], ["x"])
                for e in unary_exprs(x):
                    prog = mu.Program(e, ["x"], **opts)
                    res = self.run_both(state, prog)[0]
                    for env in envs(["x"], [spec]):
                        self.encloses(res, concrete(e, env), (opts, spec, e, env))
                        checked += 1
            pairs = [(SPECS[i], SPECS[j]) for i in range(len(SPECS)) for j in range(len(SPECS)) if (i + 2 * j) % 3 == 0]
            for sx, sy in pairs:
                state = mu.State.from_inputs([[sx, sy]], ["x", "y"])
                for e in binary_exprs(x, y):
                    prog = mu.Program(e, ["x", "y"], **opts)
                    res = self.run_both(state, prog)[0]
                    for env in envs(["x", "y"], [sx, sy]):
                        self.encloses(res, concrete(e, env), (opts, sx, sy, e, env))
                        checked += 1
        self.assertGreater(checked, 20000)

    def test_unc001_affine_forms_hold_node_by_node_through_shared_nodes(self):
        rng = random.Random(1001)
        x, y, z = mu.column("x"), mu.column("y"), mu.column("z")
        for trial in range(120):
            pool = [x, y, z, mu.const(Fraction(rng.randint(-5, 5), rng.choice([1, 2, 3])))]
            for _ in range(rng.randint(2, 9)):
                a, b = rng.choice(pool), rng.choice(pool)
                if a.boolean or b.boolean:
                    continue
                kind = rng.choice(["add", "sub", "mul", "neg", "abs", "sqr", "min", "max", "div", "mod"])
                e = {"add": lambda: a + b, "sub": lambda: a - b, "mul": lambda: a * b, "neg": lambda: -a,
                     "abs": lambda: abs(a), "sqr": lambda: a.sqr(), "min": lambda: mu.min_(a, b),
                     "max": lambda: mu.max_(a, b), "div": lambda: a / rng.choice([2, -3, Fraction(5, 2)]),
                     "mod": lambda: a % rng.choice([2, 5, Fraction(7, 2)])}[kind]()
                pool.append(e)
            root = pool[-1] - pool[-1] if trial % 5 == 0 else pool[-1] + pool[len(pool) // 2]
            specs = [rng.choice(SPECS[:9]) for _ in range(3)]
            state = mu.State.from_inputs([specs], ["x", "y", "z"])
            opts = [dict(carrier="exact"), dict(carrier="binary64"), dict(carrier="binary64", precision=12),
                    dict(carrier="exact", rank=rng.randint(1, 6)), dict(carrier="binary64", rank=3),
                    dict(carrier="exact", budget=Fraction(rng.randint(0, 8), 2))][trial % 6]
            prog = mu.Program(root, ["x", "y", "z"], **opts)
            res = self.run_both(state, prog)[0]
            trace = state.trace(prog, 0)
            for env in envs(["x", "y", "z"], specs):
                self.encloses(res, concrete(root, env), (trial, root, env))
                for k, (expr, v) in enumerate(zip(prog.exprs, trace)):
                    self.encloses(v, concrete(expr, env), (trial, k, expr, env))
                self.affine_residuals(trace, prog, env, (trial, opts))

    def test_unc001_groups_from_engine_tables_enclose_every_row(self):
        rng = np.random.default_rng(2002)
        n = 600
        k = rng.integers(0, 12, n)
        x = [int(v) for v in rng.integers(-40, 41, n)]
        y = [int(v) * 3 + 1 for v in rng.integers(-20, 21, n)]
        d = [f"{int(v) / 100:.2f}" for v in rng.integers(-900, 901, n)]
        for i in np.flatnonzero(rng.random(n) < 0.15):
            y[int(i)] = None
        frame = pd.DataFrame({"k": k, "x": pd.Series(x, dtype=object), "y": pd.Series(y, dtype=object),
                              "d": pd.Series(d, dtype=object)})
        db = torbles.from_ddl("CREATE TABLE t (k BIGINT, x BIGINT, y BIGINT, d DECIMAL(10,2))", {"t": frame})
        t = db["t"]
        scope = mbk.Scope.from_table(t, ["x", "y", "d"], group_by=["k"])
        state = mu.State.from_scope(scope, ["x", "y", "d"])
        X, Y, D = mu.column("x"), mu.column("y"), mu.column("d")
        exprs = [X * Y - D, (Y - 1) / 3, abs(X - D).sqr(), mu.max_(X, D) - mu.min_(X, D), Y % 6, X * 2 + Y]
        rows_by_group = {}
        for r in range(n):
            rows_by_group.setdefault(int(k[r]), []).append(r)
        for e in exprs:
            for opts in (dict(carrier="exact"), dict(carrier="binary64")):
                prog = mu.Program(e, ["x", "y", "d"], **opts)
                res = self.run_both(state, prog)
                for g in range(scope.groups):
                    for r in rows_by_group.get(g, []):
                        env = {"x": Fraction(x[r]), "y": None if y[r] is None else Fraction(y[r]), "d": Fraction(d[r])}
                        self.encloses(res[g], concrete(e, env), (e, g, r))
        cong = state.evaluate(mu.Program(Y, ["x", "y", "d"]))
        for g in range(scope.groups):
            if cong[g]["congruence"] is not None:
                r0, m = cong[g]["congruence"]
                self.assertEqual(m % 3, 0)                          # y = 1 (mod 3) in every group
                self.assertEqual(r0 % 3, 1)

    # ------------------------------------------------------------------ acceptance 2: correlation, no unsound pruning
    def test_unc001_affine_correlation_keeps_x_minus_x_zero_where_intervals_lose_it(self):
        x, y = mu.column("x"), mu.column("y")
        spec = [dict(rows=10, present=10, min=-7, max=13, modulus=1), dict(rows=10, present=10, min=2, max=9, modulus=1)]
        state = mu.State.from_inputs([spec], ["x", "y"])
        for carrier in ("exact", "binary64"):
            xy = x * y
            cases = [(x - x, (0, 0), (-20, 20)), ((x + y) - y, (-7, 13), (-14, 20)), (xy - xy, (0, 0), (-180, 180)),
                     ((x * 2 + 1) - x * 2, (1, 1), (-39, 41)), (-(-x) - x, (0, 0), (-20, 20))]
            for e, affine, interval in cases:
                a = self.run_both(state, mu.Program(e, ["x", "y"], carrier=carrier))[0]
                i = self.run_both(state, mu.Program(e, ["x", "y"], carrier=carrier, domains=("congruence",)))[0]
                self.assertEqual(tuple(frac(v) for v in a["interval"]), affine, (carrier, e))
                self.assertEqual(tuple(frac(v) for v in i["interval"]), interval, (carrier, e))
                if affine == (0, 0):
                    self.assertEqual(a["zero"], "zero")
                    self.assertNotEqual(i["zero"], "zero")
            # a correlated comparison is decided only with forms
            for pred, decision in ((x + 1 > x, "all"), (x - 1 >= x, "none"), ((x + y).ne(y + x + 1), "all"),
                                   (mu.min_(x, x + 1).eq(x), "all")):
                a = self.run_both(state, mu.Program(pred, ["x", "y"], carrier=carrier))[0]
                i = self.run_both(state, mu.Program(pred, ["x", "y"], carrier=carrier, domains=()))[0]
                self.assertEqual(a["decision"], decision, (carrier, pred))
                self.assertEqual(i["decision"], "unknown", (carrier, pred))

    def test_unc001_exact_identities_are_fnd007_laws(self):
        x, y = mu.column("x"), mu.column("y")
        xy = x * y
        sq = x.sqr()
        for e, coefs in ((x - x, {}), ((x + y) - y, {"x": 1}), (xy - xy, {}), ((x * 3 + 1) / 2 - x, {"x": Fraction(1, 2)}),
                         (sq - sq + y * 2, {"y": 2}), (x * 0 + y, {"y": 1})):
            prog = mu.Program(e, ["x", "y"])
            s = mu.symbolic(prog)
            self.assertTrue(s["exact"], e)
            self.assertEqual({k: v for k, v in s["coefficients"].items()}, coefs, e)
            cert = mu.identity(prog)
            self.assertEqual(cert.kind, "law")
            self.assertEqual(cert.info["law"], "ring_identity")
            self.assertTrue(cert.fresh().fresh)
            self.assertEqual(cert.authorize_exact(field=0).kind, "law")
            # the stored claim is the expansion minus the form: its canonical form is zero (independently)
            c = cert._native_claim()
            n, m = int(c.nterms), int(c.ncols)
            terms = {}
            import ctypes
            exps = (ctypes.c_uint8 * (n * m)).from_address(c.exponents)
            co = (ctypes.c_int64 * n).from_address(c.coefficients)
            for t in range(n):
                key = tuple(exps[t * m:(t + 1) * m])
                terms[key] = terms.get(key, 0) + co[t]
            self.assertTrue(all(v == 0 for v in terms.values()), (e, terms))
            self.assertGreater(n, 0)
        for e in (x * y, x.sqr(), abs(x) - abs(x) + x * y):
            with self.assertRaises(DomainError):
                mu.identity(mu.Program(e, ["x", "y"]))
        with self.assertRaises(UnsupportedError):          # exact (min picks x) but not a polynomial identity
            mu.identity(mu.Program(mu.min_(x, x + 1) - x, ["x", "y"]))

    def test_unc001_unsound_pruning_is_prohibited(self):
        """adversarial groups: every decision is checked against every concrete row combination"""
        x, y = mu.column("x"), mu.column("y")
        big = (1 << 62) + 3
        groups = [
            [dict(rows=3, present=3, min=0, max=5, modulus=1), dict(rows=3, present=3, min=5, max=5, modulus=0)],
            [dict(rows=3, present=2, min=0, max=4, modulus=1), dict(rows=3, present=3, min=5, max=9, modulus=1)],
            [dict(rows=0), dict(rows=0)],
            [dict(rows=2, present=0), dict(rows=2, present=2, min=1, max=1, modulus=0)],
            [dict(rows=5, present=5, min=-big, max=big, modulus=2), dict(rows=5, present=5, min=big - 2, max=big, modulus=2)],
            [dict(rows=4, present=4, min=(1 << 53) - 2, max=(1 << 53) + 2, modulus=1),
             dict(rows=4, present=4, min=(1 << 53) + 1, max=(1 << 53) + 1, modulus=0)],
            [dict(rows=3, present=3, min=1, max=10, modulus=3), dict(rows=3, present=3, min=2, max=2, modulus=0)],
            [dict(rows=3, present=3, min=-5, max=5, modulus=5), dict(rows=3, present=3, min=0, max=10, modulus=10)],
        ]
        groups = [[dict(g, rows=g.get("rows", 0)) for g in grp] for grp in groups]
        state = mu.State.from_inputs(groups, ["x", "y"])
        preds = [x <= y, x < y, x.eq(y), x.ne(y), x >= y, x > y, (x * y) > y * x, x * 2 + 1 > y * 2, (x % 3).eq(y % 3),
                 (x - y).is_null(), (x < y) | (x >= y), (x < y) & (x >= y), ~(x.eq(x)), x * x >= 0, (x * y) * (x * y) >= 0]
        seen = set()
        for opts in (dict(carrier="exact"), dict(carrier="binary64"), dict(carrier="binary64", precision=4),
                     dict(carrier="exact", rank=1)):
            for pred in preds:
                ev = self.run_both(state, mu.Program(pred, ["x", "y"], **opts))
                for g, spec in enumerate(groups):
                    d = ev[g]["decision"]
                    seen.add(d)
                    vals = [concrete(pred, env) for env in envs(["x", "y"], spec)] if spec[0]["rows"] else []
                    if d == "all":
                        self.assertTrue(vals and all(v is True for v in vals), (opts, pred, g))
                    elif d == "none":
                        self.assertTrue(all(v is not True for v in vals), (opts, pred, g))
                    elif d == "empty":
                        self.assertEqual(spec[0]["rows"], 0)
                    for v in vals:
                        self.encloses(ev[g], v, (opts, pred, g))
        self.assertEqual(seen, {"all", "none", "unknown", "empty"})
        # widened, never wrong: the exact carrier past int64 rationals
        ev = state.evaluate(mu.Program(x * y, ["x", "y"]))
        self.assertIn("widened", ev[4]["flags"])

    def test_unc001_filter_matches_duckdb_on_engine_tables(self):
        rng = np.random.default_rng(3003)
        n = 3000
        k = rng.integers(0, 40, n)
        x = [int(v) for v in rng.integers(-50, 51, n) + 30 * (k % 5)]
        y = [int(v) for v in rng.integers(-20, 21, n) * 5 + 2]
        d = [f"{int(v) / 100:.2f}" for v in rng.integers(-2000, 2001, n)]
        for col, frac_null in ((x, 0.05), (y, 0.1)):
            for i in np.flatnonzero(rng.random(n) < frac_null):
                col[int(i)] = None
        frame = pd.DataFrame({"k": k, "x": pd.Series(x, dtype=object), "y": pd.Series(y, dtype=object),
                              "d": pd.Series(d, dtype=object)})
        db = torbles.from_ddl("CREATE TABLE t (k BIGINT, x BIGINT, y BIGINT, d DECIMAL(10,2))", {"t": frame})
        t = db["t"]
        X, Y, D = mu.column("x"), mu.column("y"), mu.column("d")
        dec = 'CAST(d AS DECIMAL(38,2))'
        cases = [
            (X > 40, "x > 40"),
            ((X + Y) - Y >= 60, "(x + y) - y >= 60"),
            (Y.eq(7), "y = 7"),
            ((Y - 2) % 5 > 0, "((y - 2) % 5 + 5) % 5 > 0"),
            ((X * 2 + D <= 10) & Y.not_null(), f"(x * 2 + {dec} <= 10) AND y IS NOT NULL"),
            ((X - D > 0) | Y.is_null(), f"(x - {dec} > 0) OR y IS NULL"),
            (abs(X - 60) < 15, "abs(x - 60) < 15"),
            (mu.max_(X, Y) >= 100, "CASE WHEN x IS NULL OR y IS NULL THEN NULL ELSE greatest(x, y) END >= 100"),
            (~(X.sqr() > 900), "NOT (x * x > 900)"),
        ]
        for sel in (None, rng.random(n) < 0.6):
            scope = mbk.Scope.from_table(t, ["x", "y", "d"], group_by=["k"], rows=sel)
            state = mu.State.from_scope(scope, ["x", "y", "d"])
            where = "TRUE"
            if sel is not None:
                db.con.register("unc_sel", pd.DataFrame({"rid": np.flatnonzero(sel).astype(np.int64)}))
                where = "rowid IN (SELECT rid FROM unc_sel)"
            for pred, sql in cases:
                for opts in (dict(carrier="exact"), dict(carrier="binary64", precision=20)):
                    prog = mu.Program(pred, ["x", "y", "d"], **opts)
                    bits, st = state.filter(scope, prog)
                    gbits, gst = state.filter(scope, prog, backend="cuda")
                    self.assertTrue(np.array_equal(bits, gbits), sql)
                    self.assertEqual({a: b for a, b in st.items() if a not in ("backend", "device")},
                                     {a: b for a, b in gst.items() if a not in ("backend", "device")})
                    got = np.flatnonzero(np.unpackbits(bits.view(np.uint8), bitorder="little")[:n]).tolist()
                    want = [r for r, in db.con.execute(f"SELECT rowid FROM t WHERE {where} AND ({sql}) ORDER BY rowid").fetchall()]
                    self.assertEqual(got, want, (sql, opts))
                    self.assertEqual(st["rows_proven"] + st["rows_evaluated"], scope.selected)
                    self.assertEqual(st["rows_true"], len(want))
                    plain, pst = mu.rows(scope, prog)          # every row concretely: the same answer
                    self.assertTrue(np.array_equal(plain, bits))
                    self.assertEqual(pst["rows_evaluated"], scope.selected)
        # the pruning is real: groups decided without evaluation (d has no NULL: ALL is possible; x has NULLs)
        scope = mbk.Scope.from_table(t, ["x", "y", "d"], group_by=["k"])
        state = mu.State.from_scope(scope, ["x", "y", "d"])
        bits, st = state.filter(scope, mu.Program(X < -45, ["x", "y", "d"]))
        self.assertGreater(st["groups_none"], 0)
        self.assertGreater(st["rows_proven"], 0)
        bits, st = state.filter(scope, mu.Program(D >= -20, ["x", "y", "d"]))
        self.assertEqual(st["groups_all"], scope.groups)
        self.assertEqual((st["rows_proven"], st["rows_evaluated"]), (scope.selected, 0))

    def test_unc001_stale_state_is_refused(self):
        frame = pd.DataFrame({"k": [0, 0, 1, 1], "x": [1, 2, 3, 4]})
        db = torbles.from_ddl("CREATE TABLE t (k BIGINT, x BIGINT)", {"t": frame})
        t = db["t"]
        scope = mbk.Scope.from_table(t, ["x"], group_by=["k"])
        state = mu.State.from_scope(scope, ["x"])
        prog = mu.Program(mu.column("x") > 2, ["x"])
        self.assertEqual(state.fresh(scope), (True, []))
        t.flip(0, 0); t.repair()
        now = mbk.Scope.from_table(t, ["x"], group_by=["k"])
        ok, reasons = state.fresh(now)
        self.assertFalse(ok)
        self.assertIn("revision", reasons)
        for backend in ("cpu", "cuda"):
            with self.assertRaises(StaleStateError):
                state.filter(now, prog, backend=backend)
        with self.assertRaises(StaleStateError):
            state.certify(now, mu.Program(mu.column("x"), ["x"]), "not_null")

    # ------------------------------------------------------------------ congruences, GF(p), certificates
    def test_unc001_crt_meet_matches_brute_force(self):
        rng = random.Random(4004)
        for _ in range(400):
            m1, m2 = rng.randint(0, 12), rng.randint(0, 12)
            r1, r2 = rng.randint(-20, 20), rng.randint(-20, 20)
            d = rng.choice([1, 1, 2, 3])
            a, b = (Fraction(r1, d), Fraction(m1, d)), (Fraction(r2, d), Fraction(m2, d))
            got = mu.meet(a, b)
            span = [Fraction(v, d) for v in range(-400, 401)]
            want = [v for v in span if in_class(v, a) and in_class(v, b)]
            if got is None:
                self.assertEqual(want, [], (a, b))
                continue
            r, m = got
            self.assertEqual([v for v in span if in_class(v, (r, m))], want, (a, b, got))
            self.assertTrue(0 <= r < m or m == 0)
        self.assertEqual(mu.meet((1, 3), (2, 5)), (Fraction(7), Fraction(15)))
        self.assertIsNone(mu.meet((1, 4), (2, 6)))
        self.assertEqual(mu.meet((Fraction(1, 2), 1), (Fraction(3, 2), 2)), (Fraction(3, 2), Fraction(2)))

    def test_unc001_gfp_congruence_constant_and_not_null_certificates(self):
        rng = np.random.default_rng(5005)
        n = 500
        k = rng.integers(0, 6, n)
        x = 7 * rng.integers(-30, 30, n) + 3                    # x = 3 (mod 7)
        z = np.full(n, 42)
        y = [None if v else int(w) for v, w in zip(rng.random(n) < 0.1, rng.integers(0, 9, n))]
        frame = pd.DataFrame({"k": k, "x": x, "z": z, "y": pd.Series(y, dtype=object)})
        db = torbles.from_ddl("CREATE TABLE t (k BIGINT, x BIGINT, z BIGINT, y BIGINT)", {"t": frame})
        t = db["t"]
        scope = mbk.Scope.from_table(t, ["x", "z", "y"], group_by=["k"])
        state = mu.State.from_scope(scope, ["x", "z", "y"])
        X, Z, Y = mu.column("x"), mu.column("z"), mu.column("y")
        cols = ["x", "z", "y"]
        cert = state.certify(scope, mu.Program(X, cols), "congruence", modulus=7)
        self.assertEqual(cert.kind, "snapshot")
        self.assertEqual(cert.info["field"], 7)
        self.assertTrue(cert.evidence["holds"])
        self.assertEqual(cert.evidence["checked"], n)
        self.assertTrue(cert.fresh(scope).fresh)
        self.assertEqual(cert.authorize_exact(scope, field=7).kind, "snapshot")
        cert2 = state.certify(scope, mu.Program(X * 2 + 1, cols), "congruence", modulus=7)   # 2x + 1 = 0 (mod 7)
        self.assertTrue(cert2.evidence["holds"])
        c3 = state.certify(scope, mu.Program(Z * 2 - 84, cols), "constant")
        self.assertTrue(c3.evidence["holds"])
        c4 = state.certify(scope, mu.Program(X, cols), "not_null")
        self.assertEqual(c4.claim["kind"], "not_null")
        with self.assertRaises(DomainError):                    # NULLs: not proven, no certificate
            state.certify(scope, mu.Program(Y, cols), "not_null")
        with self.assertRaises(DomainError):                    # not a class mod 5
            state.certify(scope, mu.Program(X, cols), "congruence", modulus=5)
        with self.assertRaises(ValueError):                     # not a prime
            state.certify(scope, mu.Program(X, cols), "congruence", modulus=21)
        # GF(p) is the special case m = p: arithmetic of classes mod 7 in the evaluation
        ev = self.run_both(state, mu.Program(X * X + X * 3, cols))
        for g in range(scope.groups):
            r, m = ev[g]["congruence"]
            self.assertEqual(m % 7, 0)
            self.assertEqual(r % 7, (3 * 3 + 3 * 3) % 7)

    def test_unc001_sum_overflow_decisions(self):
        x = mu.column("x")
        for lo, hi, mod, rows, bits in ((-5, 9, 1, 7, 8), (0, 100, 1, 3, 9), (-(1 << 40), 1 << 40, 1, 1 << 22, 64),
                                        (-(1 << 40), 1 << 40, 1, 1 << 23, 64), (3, 3, 0, 40, 8), (-9, 9, 3, 14, 8)):
            state = mu.State.from_inputs([[dict(rows=rows, present=rows, min=lo, max=hi, modulus=mod)]], ["x"])
            for e, unit in ((x, 1), (x * 3, 1), (x / 2, Fraction(1, 2)), (x / 2, 1)):
                res = state.evaluate(mu.Program(e, ["x"]))[0]
                s = mu.sum_fits(res, bits=bits, unit=unit)
                vlo, vhi = (frac(v) for v in res["interval"])
                integral = all((frac(v) / unit).denominator == 1 for v in (res["congruence"][0], res["congruence"][1]))
                self.assertEqual(s["integral"], integral, (e, unit))
                if s["proven"]:
                    worst_hi = rows * max(vhi / unit, 0)
                    worst_lo = rows * min(vlo / unit, 0)
                    self.assertLessEqual(worst_hi, (1 << (bits - 1)) - 1)
                    self.assertGreaterEqual(worst_lo, -(1 << (bits - 1)))
                    self.assertLessEqual(s["bounds"][1], (1 << (bits - 1)) - 1)
                elif integral and s["bounds"] is not None:
                    self.assertTrue(s["bounds"][1] > (1 << (bits - 1)) - 1 or s["bounds"][0] < -(1 << (bits - 1)))
        r = mu.State.from_inputs([[dict(rows=1 << 22, present=1 << 22, min=-(1 << 40), max=1 << 40, modulus=1)]],
                                 ["x"]).evaluate(mu.Program(x, ["x"]))[0]
        self.assertTrue(mu.sum_fits(r, 64)["proven"])
        self.assertFalse(mu.sum_fits(r, 63)["proven"])

    # ------------------------------------------------------------------ budgets, refusals, malformed programs
    def test_unc001_budgets_at_the_limit_and_one_past(self):
        x = mu.column("x")
        state = mu.State.from_inputs([[dict(rows=3, present=3, min=1, max=4, modulus=1)]] * 5, ["x"])
        e = x
        for _ in range(9):
            e = e + 1
        prog = mu.Program(e, ["x"])
        n = len(prog)
        state.evaluate(prog, budget=Budget(circuit=n))
        with self.assertRaises(AlgebraBudgetError) as cm:
            state.evaluate(prog, budget=Budget(circuit=n - 1))
        self.assertEqual((cm.exception.resource, cm.exception.need, cm.exception.limit), ("circuit", n, n - 1))
        state.evaluate(prog, budget=Budget(output=5))
        with self.assertRaises(AlgebraBudgetError):
            state.evaluate(prog, budget=Budget(output=4))
        with self.assertRaises(AlgebraBudgetError):
            state.evaluate(prog, backend="cuda", budget=Budget(output=4))
        dev = mu.need(prog, 5, backend="cuda")["bytes"]
        state.evaluate(prog, backend="cuda", budget=Budget(bytes=dev))
        with self.assertRaises(AlgebraBudgetError) as cm:
            state.evaluate(prog, backend="cuda", budget=Budget(bytes=dev - 1))
        self.assertEqual(cm.exception.resource, "bytes")
        cpu = mu.need(prog, 5)["bytes"]
        state.evaluate(prog, budget=Budget(bytes=cpu))
        with self.assertRaises(AlgebraBudgetError):
            state.evaluate(prog, budget=Budget(bytes=cpu - 1))
        p8 = mu.Program(e, ["x"], rank=8)
        state.evaluate(p8, budget=Budget(support=8))
        with self.assertRaises(AlgebraBudgetError):
            state.evaluate(p8, budget=Budget(support=7))
        mu.State.from_inputs([[dict(rows=1, present=1, min=0, max=0)]] * 6, ["x"], budget=Budget(states=6))
        with self.assertRaises(AlgebraBudgetError):
            mu.State.from_inputs([[dict(rows=1, present=1, min=0, max=0)]] * 6, ["x"], budget=Budget(states=5))
        frame = pd.DataFrame({"k": [0, 1, 2, 3], "x": [1, 2, 3, 4]})
        t = torbles.from_ddl("CREATE TABLE t (k BIGINT, x BIGINT)", {"t": frame})["t"]
        scope = mbk.Scope.from_table(t, ["x"], group_by=["k"])
        mu.State.from_scope(scope, ["x"], budget=Budget(states=scope.groups))
        with self.assertRaises(AlgebraBudgetError):
            mu.State.from_scope(scope, ["x"], budget=Budget(states=scope.groups - 1))
        # the identity's expansion: support = its terms (here 2^5 + the form)
        y = mu.column("y")
        p = (x + y)
        for _ in range(4):
            p = p * (x + y)
        q = mu.Program(p * 0 + x, ["x", "y"])
        with self.assertRaises(DomainError):
            mu.identity(mu.Program(p, ["x", "y"]))
        terms = 2 ** 5 + 1 + 1
        mu.identity(q, budget=Budget(support=terms))
        with self.assertRaises(AlgebraBudgetError) as cm:
            mu.identity(q, budget=Budget(support=terms - 1))
        self.assertEqual(cm.exception.resource, "support")

    def test_unc001_malformed_programs_are_refused(self):
        x = mu.column("x")
        with self.assertRaises(ValueError):
            mu.Program(x / 0, ["x"])                                    # a zero divisor
        with self.assertRaises(ValueError):
            mu.Program(x % 0, ["x"])
        with self.assertRaises(ValueError):
            mu.Program(x % -3, ["x"])
        with self.assertRaises(TypeError):
            x / mu.column("x")
        with self.assertRaises(TypeError):
            x + (x > 1)
        with self.assertRaises(TypeError):
            x + 0.5                                                     # floats are not exact input
        with self.assertRaises(ValueError):
            mu.Program(mu.column("q"), ["x"])
        with self.assertRaises(UnsupportedError):
            mu.Program(x, ["x"], rank=17)
        with self.assertRaises(ValueError):
            mu.Program(x, ["x"], carrier="binary64", precision=1)
        e = x
        for _ in range(mu.MAX_NODES):
            e = e + 1
        with self.assertRaises(UnsupportedError):
            mu.Program(e, ["x"])
        with self.assertRaises(ValueError):                             # inconsistent inputs
            mu.State.from_inputs([[dict(rows=2, present=3, min=0, max=1, modulus=1)]], ["x"])
        with self.assertRaises(ValueError):
            mu.State.from_inputs([[dict(rows=2, present=2, min=0, max=5, modulus=2)]], ["x"])
        with self.assertRaises(ValueError):
            mu.State.from_inputs([[dict(rows=2, present=2, min=0, max=0, scale=19)]], ["x"])

    def test_unc001_import_torbles_does_not_load_the_module(self):
        code = "import sys, torbles; print('torbles.math_unc' in sys.modules)"
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                             env=dict(os.environ), check=True).stdout.strip()
        self.assertEqual(out, "False")


if __name__ == "__main__":
    unittest.main()
