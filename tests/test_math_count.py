# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""COUNT-001: observed-domain constraint counts and witnesses (torbles/math_count.c, math_count_gpu.cu, math_count.py).

Oracles, independent of the implementation: exhaustive mask evaluation with Python integers as bit sets (each
condition's TRUE and FALSE row sets computed here from the source values with Fractions and SQL NULL semantics,
Kleene's connectives on the sets, the constraints as NOT a OR b, NOT (a XOR b), NOT (a AND b)), DuckDB SQL over the
typed source table (COUNT(*) FILTER with three-valued WHERE semantics, the satisfying rowids, the smallest violating
rowids), scalar brute force over every assignment of the universe, and Python sets of the observed assignment patterns.
Every native result is also computed on the CUDA path (emulated here with TORBLES_EMU_SCRAMBLE=1, a scrambled thread
order) and compared for equality. Seeds are fixed.
"""
import random
import unittest
from fractions import Fraction

import numpy as np
import pandas as pd

import algq
import torbles
from torbles import coltypes
from torbles import math_bank as mbk
from torbles import math_bool as mb
from torbles import math_count as mc
from torbles import math_handles as mh
from torbles.math_budgets import Budget
from torbles.math_certificates import NotAuthorizedError, StaleCertificateError
from torbles.math_domains import DomainError
from torbles.math_errors import StaleStateError, UnsupportedError
from torbles.mathematics import AlgebraBudgetError

CONST, VAR, NOT, AND, OR, XOR = mb.CONST, mb.VAR, mb.NOT, mb.AND, mb.OR, mb.XOR
M64 = (1 << 64) - 1


def obj(values):
    return pd.Series(values, dtype=object)


def decode(table, col):
    """the stored codes of a column decoded now, natively"""
    out = np.empty(max(table.n, 1), dtype=np.uint32)
    algq.lib.at_decode_column(table.at, table.idx(col), out.ctypes.data)
    return out[:table.n]


def flip_column(table, col, row):
    """flips one stored bit so that exactly column `col` changes at `row`"""
    before = {c: decode(table, c) for c in table.cols}
    for p in range(table.nplanes()):
        table.flip(p, row)
        after = {c: decode(table, c) for c in table.cols}
        if [c for c in table.cols if not np.array_equal(before[c], after[c])] == [col]:
            return after[col]
        table.flip(p, row)
    raise AssertionError(f"no single plane changes only {col} at row {row}")


DDL = "CREATE TABLE t (x BIGINT, y BIGINT, d DECIMAL(10,2), b BOOLEAN, k BIGINT)"
SCALES = {"x": 0, "y": 0, "d": 2, "b": 0, "k": 0}


def dataset(rng, n, nulls=0.15):
    x = [int(v) for v in rng.integers(-40, 41, n)]
    y = [int(v) for v in rng.integers(-40, 41, n)]
    d = [f"{int(v) / 100:.2f}" for v in rng.integers(-4000, 4001, n)]
    b = [bool(v) for v in rng.integers(0, 2, n)]
    k = [int(v) for v in rng.integers(0, 6, n)]
    for col in (x, y, d, b):
        for i in np.flatnonzero(rng.random(n) < nulls):
            col[int(i)] = None
    frame = pd.DataFrame({"x": obj(x), "y": obj(y), "d": obj(d), "b": obj(b), "k": obj(k)})
    return torbles.from_ddl(DDL, {"t": frame}), dict(x=x, y=y, d=d, b=b, k=k)


# ---------------------------------------------------------------------- the oracle: bit sets
def logical(col, v):
    return int(Fraction(v) * 10 ** SCALES[col]) if col == "d" else int(v)


def condition_sets(values, spec, n):
    """(TRUE rows, FALSE rows) of a condition spec as Python int bit sets; a NULL operand: in neither"""
    col, op = spec[0], spec[1]
    T = F = 0
    for r in range(n):
        v = values[col][r]
        if op in ("is null", "is not null"):
            hit = (v is None) == (op == "is null")
        elif v is None:
            continue
        elif op == "bit":
            bit, base = spec[2], spec[3]
            hit = (((logical(col, v) - base) & M64) >> bit) & 1 == 1
        else:
            x = Fraction(v) if col == "d" else Fraction(int(v))
            if op == "between":
                hit = Fraction(spec[2]) <= x <= Fraction(spec[3])
            else:
                c = Fraction(spec[2])
                hit = {">": x > c, ">=": x >= c, "<": x < c, "<=": x <= c, "=": x == c, "<>": x != c}[op]
        if hit:
            T |= 1 << r
        else:
            F |= 1 << r
    return T, F


def sets(expr, leaves, full):
    """(TRUE set, FALSE set) of an expression DAG over the leaves' sets (Kleene's connectives)"""
    memo = {}

    def go(e):
        if id(e) in memo:
            return memo[id(e)]
        if e.op == CONST:
            r = ((0, full), (full, 0), (0, 0))[e.a]
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


def constraint_sets(c, leaves, full):
    """(satisfied, violated) sets of a constraint"""
    ta, fa = sets(c.a, leaves, full)
    if c.kind == "holds":
        return ta, fa
    tb, fb = sets(c.b, leaves, full)
    if c.kind == "implies":                       # NOT a OR b
        return fa | tb, ta & fb
    if c.kind == "equiv":                         # NOT (a XOR b)
        return (ta & tb) | (fa & fb), (ta & fb) | (fa & tb)
    return fa | fb, ta & tb                       # excludes: NOT (a AND b)


def members(x):
    out = []
    while x:
        low = x & -x
        out.append(low.bit_length() - 1)
        x ^= low
    return out


def oracle(cs, leaves, sel, full, witnesses, rows=True):
    out = []
    for c in cs.constraints:
        s, v = constraint_sets(c, leaves, full)
        s &= sel
        v &= sel
        out.append(dict(constraint=c.kind, points=sel.bit_count(), satisfied=s.bit_count(), violated=v.bit_count(),
                        unknown=(sel & ~(s | v)).bit_count(), witness=members(v)[0] if v else None,
                        rows=members(s) if rows else None, witnesses=members(v)[:witnesses]))
    return out


def strip(results):
    out = []
    for r in results:
        r = {k: v for k, v in r.items() if k not in ("backend", "device", "a", "b")}
        if r["rows"] is not None:
            r["rows"] = [int(x) for x in r["rows"]]
        out.append(r)
    return out


def random_expr(rng, k, size, unknown=True):
    pool = [mb.var(i) for i in range(k)] + [mb.TRUE, mb.FALSE] + ([mb.UNKNOWN] if unknown else [])
    built = []
    for _ in range(size):
        op = rng.choice([AND, OR, XOR, NOT, AND, OR])
        a = rng.choice(pool + built[-4:])
        if op == NOT:
            built.append(~a)
        else:
            b = rng.choice(pool + built)
            built.append(a & b if op == AND else a | b if op == OR else a ^ b)
    return built[-1]


def random_constraints(rng, k, n, size=5, unknown=True):
    shared = random_expr(rng, k, size, unknown)                 # one subexpression shared by several constraints
    out = []
    for j in range(n):
        kind = ("implies", "equiv", "excludes", "holds")[j % 4]
        a = random_expr(rng, k, size, unknown) if rng.random() < 0.7 else shared
        if kind == "holds":
            out.append(mc.holds(a))
        else:
            b = random_expr(rng, k, size, unknown) if rng.random() < 0.7 else shared
            out.append(mc.Constraint(kind, a, b))
    return mc.Constraints(k, out)


SPECS = [("x", ">", 5), ("y", "<=", Fraction(-7, 3)), ("d", "between", "-12.50", "20.25"), ("b", "=", 1),
         ("x", "is null"), ("d", "<>", "3.07"), ("x", "bit", 2, -9), ("y", "is not null"), ("x", "<", 0),
         ("d", ">=", Fraction(1, 3))]


def native_condition(spec):
    col, op = spec[0], spec[1]
    if op in ("is null", "is not null"):
        return mb.condition(col, op)
    if op == "bit":
        return mb.condition(col, "bit", spec[2], base=spec[3])
    if op == "between":
        return mb.condition(col, "between", spec[2], spec[3])
    return mb.condition(col, op, spec[2])


def cond_sql(spec):
    col, op = spec[0], spec[1]
    if op in ("is null", "is not null"):
        return f'("{col}" {op.upper()})'
    if op == "bit":
        return f'((((CAST("{col}" AS HUGEINT) - ({spec[3]})) >> {spec[2]}) & 1) = 1)'
    cast = f'CAST("{col}" AS DECIMAL(38,{SCALES[col]}))' if SCALES[col] else f'CAST("{col}" AS HUGEINT)'
    if op == "between":
        lo, hi = Fraction(spec[2]), Fraction(spec[3])
        return f"({cast} * {lo.denominator * hi.denominator} BETWEEN {lo.numerator * hi.denominator} AND " \
               f"{hi.numerator * lo.denominator})"
    c = Fraction(spec[2])
    return f"({cast} * {c.denominator} {op} {c.numerator})"


def expr_sql(e, conds):
    if e.op == CONST:
        return ("FALSE", "TRUE", "CAST(NULL AS BOOLEAN)")[e.a]
    if e.op == VAR:
        return conds[e.a]
    if e.op == NOT:
        return f"(NOT {expr_sql(e.a, conds)})"
    sym = {AND: "AND", OR: "OR", XOR: "<>"}[e.op]
    return f"({expr_sql(e.a, conds)} {sym} {expr_sql(e.b, conds)})"


def constraint_sql(c, conds):
    a = expr_sql(c.a, conds)
    if c.kind == "holds":
        return a
    b = expr_sql(c.b, conds)
    return {"implies": f"((NOT {a}) OR {b})", "equiv": f"(NOT ({a} <> {b}))", "excludes": f"(NOT ({a} AND {b}))"}[c.kind]


class Count(unittest.TestCase):
    def both(self, domain, cs, rows=True, witnesses=4, **kw):
        """the CPU answer, checked equal to the CUDA path's (emulated, scrambled), on CPU-built and device-built domains"""
        cpu = mc.count(domain, cs, rows=rows, witnesses=witnesses, **kw)
        gpu = mc.count(domain, cs, rows=rows, witnesses=witnesses, backend="cuda", **kw)
        self.assertEqual(strip(cpu.results), strip(gpu.results))
        cc, gc = cpu.certificate, gpu.certificate
        self.assertEqual({k: v for k, v in cc.items() if k not in ("backend", "device", "verified_by")},
                         {k: v for k, v in gc.items() if k not in ("backend", "device", "verified_by")})
        self.assertEqual((cc["verified_by"], gc["verified_by"]), ("cpu", "cuda library"))
        self.assertIn(gpu.backend, ("cuda", "cuda-emulated"))
        return cpu

    # ------------------------------------------------------------------ acceptance: exhaustive mask evaluation
    def test_count001_implication_equivalence_violations_match_exhaustive_mask_evaluation(self):
        rng = np.random.default_rng(1101)
        prng = random.Random(1101)
        n = 2600
        db, values = dataset(rng, n)
        full = (1 << n) - 1
        leaves = [condition_sets(values, s, n) for s in SPECS]
        for masked in (False, True):
            sel_mask = rng.random(n) < 0.6 if masked else np.ones(n, bool)
            scope = mbk.Scope.from_table(db["t"], ["x", "y", "d", "b"], rows=sel_mask if masked else None)
            conds = [native_condition(s) for s in SPECS]
            dom = mc.Domain.observe(scope, conds)
            gdom = mc.Domain.observe(scope, conds, backend="cuda")
            self.assertIn(gdom.backend, ("cuda", "cuda-emulated"))
            sel = sum(1 << int(r) for r in np.flatnonzero(sel_mask))
            for trial in range(3):
                cs = random_constraints(prng, len(SPECS), 9)
                want = oracle(cs, leaves, sel, full, witnesses=4)
                got = self.both(dom, cs)
                self.assertEqual(strip(got.results), want, (masked, trial))
                self.assertEqual(strip(mc.count(gdom, cs, rows=True, witnesses=4).results), want)
                for r in got.results:                         # the partitions of the points
                    self.assertEqual(r["satisfied"] + r["violated"] + r["unknown"], r["points"])
            # the planted laws: x > 5 implies NOT (x < 0); x IS NULL excludes x > 5; y <= -7/3 implies y IS NOT NULL
            x = mb.variables(len(SPECS))
            laws = mc.Constraints(len(SPECS), [mc.implies(x[0], ~x[8]), mc.excludes(x[4], x[0]),
                                               mc.implies(x[1], x[7]), mc.equiv(x[0], ~x[8])])
            res = self.both(dom, laws).results
            self.assertEqual([r["violated"] for r in res[:3]], [0, 0, 0])
            self.assertEqual(strip(res), oracle(laws, leaves, sel, full, witnesses=4))
            self.assertGreater(res[3]["violated"], 0)        # 0 <= x <= 5 breaks the equivalence
            # SQL: a NULL premise leaves the implication UNKNOWN, never satisfied and never violated
            self.assertEqual(res[0]["unknown"], sum(1 for r in np.flatnonzero(sel_mask) if values["x"][r] is None))

    def test_count001_counts_and_witnesses_match_duckdb_sql_semantics(self):
        rng = np.random.default_rng(1102)
        prng = random.Random(1102)
        n = 1800
        db, values = dataset(rng, n, nulls=0.25)
        sel_mask = rng.random(n) < 0.7
        scope = mbk.Scope.from_table(db["t"], ["x", "y", "d", "b"], rows=sel_mask)
        specs = SPECS[:7]
        dom = mc.Domain.observe(scope, [native_condition(s) for s in specs])
        con = db.con
        con.register("sel_rows", pd.DataFrame({"rid": np.flatnonzero(sel_mask).astype(np.int64)}))
        where = "rowid IN (SELECT rid FROM sel_rows)"
        csql = [cond_sql(s) for s in specs]
        for trial in range(2):
            cs = random_constraints(prng, len(specs), 8, size=4)
            got = self.both(dom, cs, witnesses=5)
            for c, r in zip(cs.constraints, got.results):
                q = constraint_sql(c, csql)
                want = con.execute(f"SELECT COUNT(*) FILTER (WHERE {q}), COUNT(*) FILTER (WHERE NOT {q}), "
                                   f"COUNT(*) FILTER (WHERE ({q}) IS NULL), COUNT(*) FROM t WHERE {where}").fetchall()[0]
                self.assertEqual((r["satisfied"], r["violated"], r["unknown"], r["points"]),
                                 tuple(int(v) for v in want), c)
                rows = [int(v) for v, in con.execute(f"SELECT rowid FROM t WHERE {where} AND {q} ORDER BY rowid")
                        .fetchall()]
                self.assertEqual([int(v) for v in r["rows"]], rows)
                bad = [int(v) for v, in con.execute(f"SELECT rowid FROM t WHERE {where} AND NOT {q} ORDER BY rowid "
                                                    f"LIMIT 5").fetchall()]
                self.assertEqual(r["witnesses"], bad)
                self.assertEqual(r["witness"], bad[0] if bad else None)

    def test_count001_predicates_are_holds_constraints_equal_to_bool001_counts(self):
        rng = np.random.default_rng(1103)
        prng = random.Random(1103)
        n = 900
        db, values = dataset(rng, n)
        scope = mbk.Scope.from_table(db["t"], ["x", "y", "d", "b"])
        conds = [native_condition(s) for s in SPECS[:6]]
        dom = mc.Domain.observe(scope, conds)
        masks = mb.Masks.from_scope(scope, conds)
        for _ in range(4):
            e = random_expr(prng, 6, 8)
            got = mc.predicate(dom, e, rows=True, witnesses=3)
            gpu = mc.predicate(dom, e, rows=True, witnesses=3, backend="cuda")
            self.assertEqual(strip(got.results), strip(gpu.results))
            r = got.results[0]
            f = mb.differences(masks, mb.Program(e, 6), [[]])[0]["f"]
            self.assertEqual((r["satisfied"], r["violated"], r["unknown"]), (f["true"], f["false"], f["unknown"]))

    # ------------------------------------------------------------------ the assignment universe
    def test_count001_universe_counts_match_brute_force_truth_tables(self):
        prng = random.Random(1104)

        def scalar(e, a):
            if e.op == CONST:
                return (False, True, None)[e.a]
            if e.op == VAR:
                return bool((a >> e.a) & 1)
            if e.op == NOT:
                v = scalar(e.a, a)
                return None if v is None else not v
            u, v = scalar(e.a, a), scalar(e.b, a)
            if e.op == AND:
                return False if u is False or v is False else (None if u is None or v is None else True)
            if e.op == OR:
                return True if u is True or v is True else (None if u is None or v is None else False)
            return None if u is None or v is None else u != v

        def value(c, a):
            u = scalar(c.a, a)
            if c.kind == "holds":
                return u
            v = scalar(c.b, a)
            if c.kind == "implies":
                return True if u is False or v is True else (None if u is None or v is None else False)
            if c.kind == "equiv":
                return None if u is None or v is None else u == v
            return True if u is False or v is False else (None if u is None or v is None else False)
        for k in (0, 1, 2, 3, 5, 7, 10):
            uni = mc.Domain.universe(k)
            self.assertEqual((uni.kind, uni.points), ("universe", 1 << k))
            cs = random_constraints(prng, k, 6, size=3 + k) if k else mc.Constraints(0, [
                mc.holds(mb.TRUE), mc.implies(mb.TRUE, mb.UNKNOWN), mc.excludes(mb.TRUE, mb.TRUE),
                mc.equiv(mb.FALSE, mb.FALSE)])
            ans = self.both(uni, cs, witnesses=3)
            for c, r in zip(cs.constraints, ans.results):
                vals = [value(c, a) for a in range(1 << k)]
                self.assertEqual((r["satisfied"], r["violated"], r["unknown"]),
                                 (vals.count(True), vals.count(False), vals.count(None)), (k, c))
                self.assertEqual([int(v) for v in r["rows"]], [a for a, v in enumerate(vals) if v is True])
                self.assertEqual(r["witnesses"], [a for a, v in enumerate(vals) if v is False][:3])
            cert = ans.certificate
            s = len(cert["support"])
            self.assertEqual((cert["kind"], cert["domain"], cert["points"], cert["complete"], cert["observed"],
                              cert["universe"], cert["covers_universe"]),
                             ("law", "universe", 1 << k, 1 << k, 1 << s, 1 << s, True))
            self.assertIsNone(ans.source())
            self.assertTrue(ans.fresh().fresh)
            self.assertTrue(ans.authorize("universe")["authorized"])
            with self.assertRaises(NotAuthorizedError) as ctx:
                ans.authorize("snapshot")
            self.assertEqual(ctx.exception.reason, "domain")
        # 16 conditions: a structured formula (a parity chain AND a threshold) over 65536 assignments
        x = mb.variables(16)
        par = x[0]
        for i in range(1, 8):
            par = par ^ x[i]
        cs = mc.Constraints(16, [mc.holds(par), mc.implies(x[8] & x[9], par | x[10]), mc.excludes(par, ~par)])
        r = self.both(mc.Domain.universe(16, budget=Budget(support=1 << 16)), cs, rows=False, witnesses=2).results
        self.assertEqual((r[0]["satisfied"], r[0]["violated"]), (1 << 15, 1 << 15))
        bad = [a for a in range(1 << 16) if (a >> 8) & 1 and (a >> 9) & 1 and not (bin(a & 0xFF).count("1") & 1)
               and not (a >> 10) & 1]
        self.assertEqual((r[1]["violated"], r[1]["witnesses"]), (len(bad), bad[:2]))
        self.assertEqual(r[2]["violated"], 0)

    # ------------------------------------------------------------------ acceptance: the certificate
    def test_count001_certificates_report_rows_covered_and_never_unobserved_assignments(self):
        rng = np.random.default_rng(1105)
        n = 1500
        x = [int(v) for v in rng.integers(-30, 31, n)]
        y = [v + 100 if v > 10 else int(rng.integers(-30, 31)) for v in x]     # x > 10 implies y > 50, always
        z = [None if rng.random() < 0.1 else int(v) for v in rng.integers(0, 4, n)]
        db = torbles.from_ddl("CREATE TABLE t (x BIGINT, y BIGINT, z BIGINT)",
                              {"t": pd.DataFrame({"x": obj(x), "y": obj(y), "z": obj(z)})})
        sel_mask = rng.random(n) < 0.75
        scope = mbk.Scope.from_table(db["t"], ["x", "y", "z"], rows=sel_mask)
        conds = [mb.condition("x", ">", 10), mb.condition("y", ">", 50), mb.condition("z", "=", 0)]
        v = mb.variables(3)
        cs = mc.Constraints(3, [mc.implies(v[0], v[1]), mc.implies(v[2], v[0] | ~v[0])])
        dom = mc.Domain.observe(scope, conds)
        ans = self.both(dom, cs)
        r0 = ans.results[0]
        self.assertEqual(r0["violated"], 0)                  # the implication holds on every covered row
        cert = ans.certificate
        selected = int(sel_mask.sum())
        self.assertEqual((cert["kind"], cert["domain"], cert["points"], cert["rows"], cert["exact_selection"]),
                         ("snapshot", "observed", selected, n, True))
        src = ans.source()                                   # the FND-007 record of exactly these rows
        self.assertEqual((src["source"], src["rows"], src["selected"], src["exact_selection"]),
                         ("table", n, selected, True))
        self.assertEqual(src["revision"], scope.revision)
        self.assertEqual([c["index"] for c in src["columns"]], [0, 1, 2])
        self.assertEqual(src["columns"][2]["nulls"], sum(1 for r in np.flatnonzero(sel_mask) if z[r] is None))
        # coverage: the complete rows and the distinct assignments of the support among them, in Python
        pats = set()
        complete = 0
        for r in np.flatnonzero(sel_mask):
            if z[r] is None:
                continue
            complete += 1
            pats.add((x[r] > 10) | ((y[r] > 50) << 1) | ((z[r] == 0) << 2))
        self.assertEqual((cert["support"], cert["complete"], cert["observed"], cert["universe"]),
                         ((0, 1, 2), complete, len(pats), 8))
        self.assertNotIn(1, pats)                            # x > 10 and y <= 50 never occurs: unobserved
        self.assertEqual(cert["unobserved"], 8 - len(pats))
        self.assertFalse(cert["covers_universe"])
        # the rows never imply the unobserved assignments were checked: a universe statement is refused
        with self.assertRaises(NotAuthorizedError) as ctx:
            ans.authorize("universe", scope)
        self.assertEqual((ctx.exception.reason, ctx.exception.unobserved), ("unobserved", 8 - len(pats)))
        self.assertTrue(ans.authorize("snapshot", scope)["authorized"])
        with self.assertRaises(NotAuthorizedError) as ctx:
            ans.authorize("snapshot")
        self.assertEqual(ctx.exception.reason, "no scope")
        # ... and the universe (a different, labelled result) does violate the implication there
        uni = self.both(mc.Domain.universe(3), cs, witnesses=8)
        self.assertEqual(uni.certificate["domain"], "universe")
        self.assertEqual(uni.results[0]["witnesses"], [1, 5])          # x > 10, y <= 50 (z either)
        self.assertGreater(uni.results[0]["violated"], 0)
        # unselected rows are never covered
        self.assertTrue(all(sel_mask[int(r)] for r in ans.results[1]["rows"]))
        # every assignment observed (and the record fresh): the universe statement is supported
        full = mbk.Scope.from_table(db["t"], ["x", "y", "z"])
        cs2 = mc.Constraints(3, [mc.implies(v[0] & v[1], v[2] | ~v[2])])
        all_seen = self.both(mc.Domain.observe(full, [mb.condition("x", ">", 0), mb.condition("y", ">", 0),
                                                      mb.condition("z", "=", 0)], backend="cuda"), cs2)
        c2 = all_seen.certificate
        seen = {(x[r] > 0) | ((y[r] > 0) << 1) | ((z[r] == 0) << 2) for r in range(n) if z[r] is not None}
        self.assertEqual((c2["observed"], c2["universe"]), (len(seen), 8))
        self.assertEqual(len(seen), 8)
        self.assertTrue(c2["covers_universe"])
        self.assertTrue(all_seen.authorize("universe", full)["authorized"])
        self.assertEqual(c2["points"], n)

    def test_count001_stale_certificates_and_domains_are_refused(self):
        rng = np.random.default_rng(1106)
        n = 600
        db, values = dataset(rng, n)
        t = db["t"]
        names = ["x", "y", "d", "b", "k"]
        mask = rng.random(n) < 0.8
        scope = mbk.Scope.from_table(t, names, rows=mask)
        conds = [mb.condition("x", ">", 0), mb.condition("d", "<", "1.5"), mb.condition("b", "is null")]
        v = mb.variables(3)
        cs = mc.Constraints(3, [mc.implies(v[0], v[1]), mc.equiv(v[2], v[0])])
        dom = mc.Domain.observe(scope, conds)
        ans = mc.count(dom, cs, rows=True)
        dev = mc.count(mc.Domain.observe(scope, conds, backend="cuda"), cs, rows=True, backend="cuda")
        self.assertEqual(strip(ans.results), strip(dev.results))
        self.assertTrue(ans.fresh(scope).fresh and dev.fresh(scope).fresh)
        self.assertEqual(ans.source(), dev.source())
        # another selection of the same table: stale with the exact reason
        other = mask.copy(); other[int(np.flatnonzero(mask)[0])] = False
        f = ans.fresh(mbk.Scope.from_table(t, names, rows=other))
        self.assertEqual(f.reasons, ("selection",))
        self.assertEqual((f.recorded, f.current), (int(mask.sum()), int(mask.sum()) - 1))
        # a codec change of a column a condition reads: the encoding was recorded
        old = t.codecs["k"]
        t.codecs["k"] = coltypes.IntCodec(old.affine[0] + 3)
        try:
            cs_k = mc.count(mc.Domain.observe(mbk.Scope.from_table(t, names, rows=mask),
                                              [mb.condition("k", ">", 2)]), mc.Constraints(1, [mc.holds(mb.var(0))]))
            self.assertTrue(cs_k.fresh().fresh)
            self.assertTrue(ans.fresh(mbk.Scope.from_table(t, names, rows=mask)).fresh)   # k is not read by ans
        finally:
            t.codecs["k"] = old
        moved = coltypes.IntCodec(old.affine[0] + 3)
        before = mc.count(mc.Domain.observe(mbk.Scope.from_table(t, names, rows=mask), [mb.condition("k", ">", 2)]),
                          mc.Constraints(1, [mc.holds(mb.var(0))]))
        t.codecs["k"] = moved
        try:
            self.assertEqual(before.fresh(mbk.Scope.from_table(t, names, rows=mask)).reasons, ("columns",))
        finally:
            t.codecs["k"] = old
        # a mutation: a new revision; the domain is refused, the certificate reports why
        t.flip(0, int(np.flatnonzero(mask)[3]))
        now = mbk.Scope.from_table(t, names, rows=mask)
        f = ans.fresh(now)
        self.assertFalse(f.fresh)
        self.assertEqual(f.first, "revision")
        self.assertEqual(dev.fresh(now), f)                    # the device answer's certificate: the same record
        with self.assertRaises(StaleCertificateError) as ctx:
            ans.authorize("snapshot", now)
        self.assertIn("revision", ctx.exception.reasons)
        for backend in ("cpu", "cuda"):
            with self.assertRaises(StaleStateError):
                mc.count(dom, cs, backend=backend)
            with self.assertRaises(StaleStateError):
                mc.Domain.observe(scope, conds, backend=backend)
        self.assertFalse(dom.fresh(now).fresh)
        # a NULL-domain change of a column a condition reads (b becomes NULL at one row)
        db2, values2 = dataset(np.random.default_rng(1107), n)
        t2 = db2["t"]
        a2 = mc.count(mc.Domain.observe(mbk.Scope.from_table(t2, names), conds), cs)
        b2 = mc.count(mc.Domain.observe(mbk.Scope.from_table(t2, names), [mb.condition("x", ">", 0)]),
                      mc.Constraints(1, [mc.holds(mb.var(0))]))
        row = next(r for r in range(n) if values2["b"][r] is not None)
        flip_column(t2, t2.nullable["b"], row)
        now2 = mbk.Scope.from_table(t2, names)
        self.assertEqual(now2.column("b")["valid"], sum(v is not None for v in values2["b"]) - 1)
        f2 = a2.fresh(now2)
        self.assertEqual(set(f2.reasons), {"revision", "null_domain"})
        self.assertEqual(f2.first, "revision")
        self.assertEqual(b2.fresh(now2).reasons, ("revision",))      # x's NULL domain did not change

    # ------------------------------------------------------------------ extraction over many words, budgets, limits
    def test_count001_extraction_over_many_words_and_sparse_selections(self):
        rng = np.random.default_rng(1108)
        n = 40_000
        a = rng.integers(0, 2, n).astype(bool)
        b = rng.integers(0, 2, n).astype(bool)
        c = a ^ b
        c[[17, 20_001, 39_998]] = ~c[[17, 20_001, 39_998]]        # three rows break c = a XOR b
        frame = pd.DataFrame({"a": a, "b": b, "c": c})
        db = torbles.from_ddl("CREATE TABLE t (a BOOLEAN, b BOOLEAN, c BOOLEAN)", {"t": frame})
        for sel_mask in (np.ones(n, bool), rng.random(n) < 0.02):
            scope = mbk.Scope.from_table(db["t"], ["a", "b", "c"], rows=sel_mask)
            conds = [mb.condition(col, "<>", 0) for col in ("a", "b", "c")]
            v = mb.variables(3)
            cs = mc.Constraints(3, [mc.equiv(v[2], v[0] ^ v[1]), mc.implies(v[0] & v[1], ~v[2]), mc.holds(v[2])])
            for dom in (mc.Domain.observe(scope, conds), mc.Domain.observe(scope, conds, backend="cuda")):
                ans = self.both(dom, cs, witnesses=10)
                sel = np.flatnonzero(sel_mask)
                eq = c == (a ^ b)
                r = ans.results
                self.assertEqual(r[0]["witnesses"], [int(i) for i in sel if not eq[i]][:10])
                self.assertEqual(r[0]["violated"], int((~eq[sel]).sum()))
                self.assertTrue(np.array_equal(r[2]["rows"], sel[c[sel]].astype(np.uint64)))
                imp_bad = sel[a[sel] & b[sel] & c[sel]]
                self.assertEqual(r[1]["witnesses"], [int(i) for i in imp_bad[:10]])

    def test_count001_budgets_at_the_limit_and_one_past(self):
        rng = np.random.default_rng(1109)
        n = 1200
        db, values = dataset(rng, n)
        scope = mbk.Scope.from_table(db["t"], ["x", "y", "d", "b"])
        conds = [native_condition(s) for s in SPECS[:5]]
        dom = mc.Domain.observe(scope, conds)
        v = mb.variables(5)
        cs = mc.Constraints(5, [mc.implies(v[0], v[1]), mc.holds(v[2] | v[3]), mc.excludes(v[4], v[0])])
        ref = mc.count(dom, cs, rows=True, witnesses=3).results
        out = 3 + sum(r["satisfied"] for r in ref) + sum(min(3, r["violated"]) for r in ref)
        for backend in ("cpu", "cuda"):
            mc.count(dom, cs, rows=True, witnesses=3, backend=backend, budget=Budget(output=out))
            with self.assertRaises(AlgebraBudgetError) as ctx:              # refused past the output budget
                mc.count(dom, cs, rows=True, witnesses=3, backend=backend, budget=Budget(output=out - 1))
            self.assertEqual((ctx.exception.resource, ctx.exception.need, ctx.exception.limit),
                             ("output", out, out - 1))
            need = mc.count_need(dom, cs, rows=True, witnesses=3, backend=backend)
            nn = need["circuit"]
            mc.count(dom, cs, rows=True, witnesses=3, backend=backend, budget=Budget(circuit=nn))
            with self.assertRaises(AlgebraBudgetError) as ctx:
                mc.count(dom, cs, rows=True, witnesses=3, backend=backend, budget=Budget(circuit=nn - 1))
            self.assertEqual((ctx.exception.resource, ctx.exception.need), ("circuit", nn))
            # bytes before the counts (the CPU scratch, or the device copies of the counting launch), checked with
            # nothing requested so that no output enters the need
            nb = mc.count_need(dom, cs, rows=False, witnesses=0, backend=backend)["bytes"]
            with self.assertRaises(AlgebraBudgetError) as ctx:
                mc.count(dom, cs, rows=False, witnesses=0, backend=backend, budget=Budget(bytes=nb - 1))
            self.assertEqual((ctx.exception.resource, ctx.exception.need), ("bytes", nb))
            cpu_nb = mc.count_need(dom, cs, rows=False, witnesses=0)["bytes"]
            self.assertLessEqual(cpu_nb, nb)                  # the adoption's own need fits as well
            mc.count(dom, cs, rows=False, witnesses=0, backend=backend, budget=Budget(bytes=nb))
        # the universe: support 2^k at the limit and one past (k <= 20 with the default budget)
        self.assertEqual(mc.domain_need(None, 12)["support"], 4096)
        uni = mc.Domain.universe(12, budget=Budget(support=4096))
        with self.assertRaises(AlgebraBudgetError) as ctx:
            mc.Domain.universe(13, budget=Budget(support=4096))
        self.assertEqual((ctx.exception.resource, ctx.exception.need), ("support", 8192))
        with self.assertRaises(AlgebraBudgetError) as ctx:
            mc.Domain.universe(21)
        self.assertEqual((ctx.exception.resource, ctx.exception.need, ctx.exception.limit), ("support", 1 << 21, 1 << 20))
        p12 = mc.Constraints(12, [mc.holds(mb.var(0) ^ mb.var(11))])
        mc.count(uni, p12, budget=Budget(support=4096))
        with self.assertRaises(AlgebraBudgetError):
            mc.count(uni, p12, budget=Budget(support=4095))
        # the domain's bytes and the set's circuit
        for backend in ("cpu", "cuda"):
            nb = mc.domain_need(scope, conds, backend=backend)["bytes"]
            mc.Domain.observe(scope, conds, backend=backend, budget=Budget(bytes=nb))
            with self.assertRaises(AlgebraBudgetError) as ctx:
                mc.Domain.observe(scope, conds, backend=backend, budget=Budget(bytes=nb - 1))
            self.assertEqual((ctx.exception.resource, ctx.exception.need), ("bytes", nb))
        nn = cs.info["nodes"]
        mc.Constraints(5, cs.constraints, budget=Budget(circuit=nn))
        with self.assertRaises(AlgebraBudgetError) as ctx:
            mc.Constraints(5, cs.constraints, budget=Budget(circuit=nn - 1))
        self.assertEqual((ctx.exception.resource, ctx.exception.need), ("circuit", nn))
        # the coverage bitmap of the observed assignments: states 2^s
        self.assertEqual(mc.count_need(dom, cs)["states"], 1 << 5)
        with self.assertRaises(AlgebraBudgetError) as ctx:
            mc.count(dom, cs, budget=Budget(states=31))
        self.assertEqual((ctx.exception.resource, ctx.exception.need), ("states", 32))

    def test_count001_malformed_input_and_representation_limits(self):
        v = mb.variables(3)
        with self.assertRaises(ValueError):
            mc.Constraint("implies", v[0])
        with self.assertRaises(ValueError):
            mc.Constraint("nope", v[0], v[1])
        with self.assertRaises(ValueError):
            mc.Constraints.from_nodes(2, [("var", 0), ("var", 1)], [("implies", 0, 2)])     # no node 2
        with self.assertRaises(ValueError):
            mc.Constraints.from_nodes(2, [("var", 0), ("var", 1)], [("holds", 0, 1)])       # holds takes one
        with self.assertRaises(ValueError):
            mc.Constraints.from_nodes(2, [("var", 2)], [("holds", 0)])                      # condition >= k
        with self.assertRaises(ValueError):
            mc.Constraints(2, [])                                                           # no constraint
        cs = mc.Constraints(3, [mc.implies(v[0], v[2])])
        with self.assertRaises(DomainError):
            mc.count(mc.Domain.universe(4), cs)
        with self.assertRaises(UnsupportedError):
            mc.Domain.universe(64, budget=Budget(variables=64, support=(1 << 64) - 1))
        with self.assertRaises(ValueError):
            mc.count(mc.Domain.universe(3), cs, witnesses=-1)
        frame = pd.DataFrame({"x": obj([1, 2, 3])})
        db = torbles.from_ddl("CREATE TABLE t (x BIGINT)", {"t": frame})
        scope = mbk.Scope.from_table(db["t"], ["x"])
        with self.assertRaises(ValueError):
            mc.Domain.observe(scope, [mb.condition("nope", ">", 1)])
        # an empty selection: every count zero, nothing covered
        empty = mc.Domain.observe(mbk.Scope.from_table(db["t"], ["x"], rows=np.zeros(3, bool)),
                                  [mb.condition("x", ">", 1)])
        ans = self.both(empty, mc.Constraints(1, [mc.holds(mb.var(0)), mc.implies(mb.var(0), mb.FALSE)]))
        self.assertEqual([(r["points"], r["satisfied"], r["violated"], r["witness"]) for r in ans.results],
                         [(0, 0, 0, None)] * 2)
        self.assertEqual((ans.certificate["points"], ans.certificate["complete"], ans.certificate["observed"]), (0, 0, 0))
        # a readback that is not this domain's is refused by the adoption
        dom = mc.Domain.observe(scope, [mb.condition("x", ">", 1)])
        one = mc.Constraints(1, [mc.holds(mb.var(0))])
        res = (mc._Result * 1)()
        q = mc._Request(1, 0, 1)
        err, ref = mc.C.c_int32(), mc._Refusal()
        rc = mc._call("tr_count_counts_view", mc.C.byref(dom.domain()), mc.C.byref(one.view()), None, res,
                      mc.C.byref(ref))
        self.assertEqual(rc, 0)
        pt, wt = mc.C.c_uint64(), mc.C.c_uint64()
        mc._call("tr_count_plan", res, 1, mc.C.byref(q), mc.C.byref(pt), mc.C.byref(wt))
        pts = np.array([2, 1], dtype=np.uint64)                   # out of order
        wit = np.array([0], dtype=np.uint64)
        h = mc._call("tr_count_adopt", dom._handle, one._handle, mc.C.byref(q), res, pts.ctypes.data,
                     wit.ctypes.data, 1, None, mc.C.byref(ref), mc.C.byref(err))
        self.assertEqual((h, err.value), (None, -1))
        good = np.array([1, 2], dtype=np.uint64)
        h = mc._call("tr_count_adopt", dom._handle, one._handle, mc.C.byref(q), res, good.ctypes.data,
                     wit.ctypes.data, 1, None, mc.C.byref(ref), mc.C.byref(err))
        self.assertTrue(h)
        mc._call("tr_count_answer_free", h)

    def test_count001_ctypes_structures_are_registered(self):
        report = mh.abi_check()
        mine = {r["native"] for r in report["checked"] if r["structure"].startswith("torbles.math_count.")}
        self.assertEqual(mine, {"tr_count_constraint", "tr_count_request", "tr_count_result", "tr_count_set_view",
                                "tr_count_set_info", "tr_count_domain_info", "tr_count_certificate",
                                "tr_count_answer_view", "tr_count_decision"})
        cuda = {r["native"] for r in report["checked"] if r["library"] == "cuda"
                and r["structure"].startswith("torbles.math_count.")}
        self.assertEqual(cuda, {"tr_count_constraint", "tr_count_request", "tr_count_result", "tr_count_set_view"})


if __name__ == "__main__":
    unittest.main()
