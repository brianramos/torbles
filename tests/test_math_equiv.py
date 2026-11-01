# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
"""META-001: scoped query equivalence, implication and counterexamples.

Oracles are independent of the implementation: SQL three-valued logic and exact Fraction arithmetic written here,
exhaustive enumeration of declared domains in Python, and DuckDB SQL on the same rows (NULLs included). Every snapshot
check also runs on the CUDA path (emulated here) and must give identical evidence and verdicts.
Run: python -m unittest tests/test_math_equiv.py -k meta001
"""
import itertools
import random
import unittest
from fractions import Fraction

import duckdb
import numpy as np
import pandas as pd

import torbles
from torbles import math_bank as mbk
from torbles import math_certificates as mc
from torbles import math_equiv as meq
from torbles import math_geom as mg
from torbles.math_budgets import Budget
from torbles.math_errors import UnsupportedError
from torbles.mathematics import AlgebraBudgetError

T_, F_, U_ = 1, 0, None


# ------------------------------------------------------------------ oracles

def cuda_label():
    """the backend label of a CUDA result: "cuda" on a device, "cuda-emulated" when the kernels run in the emulation"""
    from torbles import math_gpu
    return "cuda-emulated" if math_gpu.device()["emulated"] else "cuda"


def cmp(op, x, y):
    return {"=": x == y, "<>": x != y, "<": x < y, "<=": x <= y, ">": x > y, ">=": x >= y}[op]


def ev_cond(c, row, scales):
    def val(name):
        v = row[name]
        return None if v is None else Fraction(v, 10 ** scales.get(name, 0))
    x = val(c.column)
    if c.op == "is null":
        return T_ if x is None else F_
    if c.op == "is not null":
        return F_ if x is None else T_
    if x is None:
        return U_
    if c.other is not None:
        y = val(c.other)
        return U_ if y is None else int(cmp(c.op, x, y))
    if c.op == "between":
        return int(Fraction(*c.args[0]) <= x <= Fraction(*c.args[1]))
    if c.op == "bit":
        base, bit = c.args
        return ((row[c.column] - base) % (1 << 64)) >> bit & 1
    return int(cmp(c.op, x, Fraction(*c.args[0])))


def ev(p, row, scales=None):
    """SQL three-valued logic: 1 TRUE, 0 FALSE, None UNKNOWN"""
    scales = scales or {}
    if p.op == 1:
        return {1: T_, 0: F_, 2: U_}[p.a]
    if p.op == 2:
        return ev_cond(p.a, row, scales)
    if p.op == 3:
        v = ev(p.a, row, scales)
        return None if v is None else 1 - v
    a, b = ev(p.a, row, scales), ev(p.b, row, scales)
    if p.op == 4:
        return F_ if F_ in (a, b) else (U_ if U_ in (a, b) else T_)
    if p.op == 5:
        return T_ if T_ in (a, b) else (U_ if U_ in (a, b) else F_)
    return None if a is None or b is None else a ^ b


def ev_poly(f, row):
    """a math_geom.Polynomial at a row of logical integers; None when a column it reads is NULL (SQL)"""
    used = [c for i, c in enumerate(f.columns) if any(e[i] for e in f.terms)]
    if any(row[c] is None for c in used):
        return None
    total = Fraction(0)
    for e, c in f.terms.items():
        t = Fraction(c)
        for i, k in enumerate(e):
            if k:
                t *= Fraction(row[f.columns[i]]) ** k
        total += t
    return total


def pred_differs(kind_semantics, a, b):
    rel, sem = kind_semantics
    if rel == "implies":
        return a == T_ and b != T_
    if sem == "kleene":
        return a != b
    return (a == T_) != (b == T_)


def measure_differs(nulls, fa, fb):
    if fa is None or fb is None:
        return nulls == "sql" and (fa is None) != (fb is None)
    return fa != fb


def to_sql(p):
    if p.op == 1:
        return {1: "TRUE", 0: "FALSE", 2: "CAST(NULL AS BOOLEAN)"}[p.a]
    if p.op == 2:
        c = p.a
        if c.op in ("is null", "is not null"):
            return f"({c.column} {c.op.upper()})"
        if c.other is not None:
            return f"({c.column} {c.op} {c.other})"
        if c.op == "between":
            return f"({c.column} BETWEEN {Fraction(*c.args[0])} AND {Fraction(*c.args[1])})".replace("/", ".0/")
        k = Fraction(*c.args[0])
        return f"({c.column} {c.op} ({k.numerator}.0 / {k.denominator}))"
    if p.op == 3:
        return f"(NOT {to_sql(p.a)})"
    op = {4: "AND", 5: "OR", 6: "<>"}[p.op]
    return f"({to_sql(p.a)} {op} {to_sql(p.b)})"


def poly_sql(f):
    parts = []
    for e, c in f.terms.items():
        fac = [f"CAST({f.columns[i]} AS HUGEINT)" for i, k in enumerate(e) for _ in range(k)]
        parts.append("*".join([f"({Fraction(c).numerator})"] + fac) + f"/{Fraction(c).denominator}.0")
    return "(" + (" + ".join(parts) or "0") + ")"


def duck(cols, sql_where, mask=None):
    """(count, smallest row) of the selected rows of cols where sql_where holds, by DuckDB"""
    n = len(next(iter(cols.values())))
    frame = pd.DataFrame({k: pd.array(v, dtype="Int64") for k, v in cols.items()})
    frame["i"] = range(n)
    frame["sel"] = [True] * n if mask is None else list(mask)
    con = duckdb.connect()
    con.register("t", frame)
    cnt, first = con.execute(f"SELECT count(*), min(i) FROM t WHERE sel AND ({sql_where})").fetchone()
    con.close()
    return int(cnt), (None if first is None else int(first))


def scope_of(cols, mask=None, scales=None):
    names = list(cols)
    vals = {k: np.array([0 if v is None else v for v in cols[k]], dtype=np.int64) for k in names}
    valid = {k: np.array([v is not None for v in cols[k]]) for k in names if any(v is None for v in cols[k])}
    rows = None if mask is None else np.array(mask, dtype=bool)
    return mbk.Scope.from_arrays(vals, valid=valid, scales=scales or {}, rows=rows)


def rows_of(cols):
    names = list(cols)
    n = len(cols[names[0]])
    return [{k: cols[k][i] for k in names} for i in range(n)]


def strip(item):
    return {k: v for k, v in item.items() if k not in ("backend", "device")}


def both(test, s, scope, qs, **kw):
    """the CPU result, after asserting that the CUDA path gives identical evidence and verdicts"""
    cpu = s.check(scope, qs, backend="cpu", **kw)
    gpu = s.check(scope, qs, backend="cuda", **kw)
    test.assertEqual(gpu.backend, cuda_label())
    test.assertEqual([strip(x) for x in cpu], [strip(x) for x in gpu])
    return cpu


def random_pred(rng, names, depth, pool=None):
    if depth == 0 or rng.random() < 0.3:
        if pool is not None:
            return rng.choice(pool)
        c = rng.choice(names)
        r = rng.random()
        if r < 0.1:
            return meq.cond(c, rng.choice(["is null", "is not null"]))
        if r < 0.25:
            return meq.cond(c, "between", rng.randint(-3, 1), rng.randint(0, 3))
        if r < 0.4:
            return meq.cond(c, rng.choice(["<", "<=", "=", "<>"]), meq.column(rng.choice(names)))
        return meq.cond(c, rng.choice(["<", "<=", ">", ">=", "=", "<>"]), rng.randint(-3, 3))
    r = rng.random()
    if r < 0.15:
        return ~random_pred(rng, names, depth - 1, pool)
    a, b = random_pred(rng, names, depth - 1, pool), random_pred(rng, names, depth - 1, pool)
    return (a & b) if r < 0.5 else ((a | b) if r < 0.85 else (a ^ b))


def points(spec, nullable):
    names = list(spec)
    axes = []
    for n in names:
        s = spec[n]
        vals = list(range(s[0], s[1] + 1)) if isinstance(s, tuple) else list(s)
        axes.append(vals + ([None] if n in nullable else []))
    for combo in itertools.product(*axes):
        yield dict(zip(names, combo))


class Meta001(unittest.TestCase):
    def check_witness_row(self, item, cols, differs):
        """a refuted snapshot verdict: its witness is the smallest selected differing row, its point that row"""
        self.assertEqual(item["verdict"], "refuted")
        row = item["witness"]
        self.assertEqual(item["point"], {k: cols[k][row] for k in cols})
        self.assertTrue(differs(row))

    # ---------------------------------------------------------------- sharing
    def test_meta001_set_shares_conditions_nodes_and_measures(self):
        a, b = meq.cond("x", "<", 5), meq.cond("y", ">=", Fraction(1, 2))
        p0 = a & b
        p1 = meq.cond("y", ">=", "0.5") & meq.cond("x", "<", 5)           # the same conditions, operands swapped
        p2 = (meq.cond("x", "<", 5) & meq.cond("y", ">=", 1, None)) | meq.cond("z", "is null")
        f = mg.poly({(1, 1): 2}, ["x", "y"])
        g = mg.poly([(1, {"y": 1, "x": 1}), (1, {"x": 1, "y": 1})], ["y", "x"])   # 2xy written differently
        h = mg.poly({(1, 0, 0): 1, (0, 0, 1): Fraction(-1, 3)}, ["x", "y", "z"])
        s = meq.Set(["x", "y", "z"], predicates=[p0, p1, p2], measures=[f, g, h],
                    queries=[(0, 0), (1, 1), (2, 2), (0, None), (1, 0)])
        info = s.info()
        self.assertEqual(info["conditions"], 4)          # x<5, y>=1/2, y>=1, z is null
        self.assertEqual(info["predicates"], 2)          # p0 and p1 are one root
        self.assertEqual(info["measures"], 2)            # f and g are one canonical measure
        self.assertEqual(info["queries"], 3)             # (p0,f) = (p1,g) = (p1,f); (p2,h); (p0,COUNT)
        self.assertEqual((info["given_predicates"], info["given_conditions"], info["given_measures"]), (3, 7, 3))
        self.assertEqual(info["distinct_filters"], 2)
        self.assertEqual(info["distinct_measures"], 2)
        self.assertEqual(info["shared_nodes"], 1)        # the VAR node of x < 5, read by both distinct roots
        ids = s.ids()
        self.assertEqual(ids["predicate_roots"][0], ids["predicate_roots"][1])
        self.assertEqual(ids["measure_ids"], [0, 0, 1])
        self.assertEqual(ids["query_ids"], [0, 0, 1, 2, 0])
        nodes = s.nodes()
        self.assertEqual(len(nodes), info["nodes"])
        for i, (op, x, y) in enumerate(nodes):           # a DAG: operands before nodes; AND/OR/XOR ordered
            if op >= 4:
                self.assertTrue(x <= y < i)
        conds, k = s.conditions()
        self.assertEqual(k, 4)
        self.assertEqual((conds[1].num[0], conds[1].den[0]), (1, 2))   # "0.5" and Fraction(1, 2): one condition

    # ---------------------------------------------------------------- snapshot checks
    def test_meta001_snapshot_checks_match_python_and_duckdb(self):
        rng = random.Random(11)
        n = 300
        cols = {c: [None if rng.random() < 0.12 else rng.randint(-4, 4) for _ in range(n)] for c in ("x", "y", "z")}
        mask = [rng.random() < 0.9 for _ in range(n)]
        scope = scope_of(cols, mask)
        pool = [random_pred(rng, ["x", "y", "z"], 0) for _ in range(40)]
        preds = [random_pred(rng, ["x", "y", "z"], 3, pool) for _ in range(14)]
        preds += [~(~preds[0] | ~preds[1]), preds[0] & preds[1]]          # De Morgan: equal everywhere
        polys = [mg.poly({(1, 1): 1}, ["x", "y"]), mg.poly({(1, 0): 1, (0, 1): 1}, ["x", "y"]),
                 mg.poly({(2, 0): 1, (0, 0): -4}, ["x", "y"]), mg.poly({(1, 0): 1, (0, 0): 2}, ["x", "z"]),
                 mg.poly({(1, 0): 1, (0, 0): -2}, ["x", "y"]), mg.poly({(): 3}, [])]
        queries = [(i % len(preds), i % len(polys)) for i in range(8)] + [(0, None), (14, None), (15, None)]
        s = meq.Set(["x", "y", "z"], predicates=preds, measures=polys, queries=queries)
        qs, want = [], []
        for i in range(len(preds)):
            for j in range(i + 1, len(preds)):
                if rng.random() < 0.25 or (i, j) == (14, 15):
                    for rel, sem in (("equivalent", "filter"), ("equivalent", "kleene"), ("implies", "filter")):
                        qs.append(meq.predicates(i, j, relation=rel, semantics=sem))
                        want.append(("p", (rel, sem), i, j))
        for i in range(len(polys)):
            for j in range(len(polys)):
                for nulls in ("sql", "skip"):
                    qs.append(meq.measures(i, j, nulls=nulls))
                    want.append(("m", nulls, i, j))
        for i in range(len(queries)):
            for j in range(len(queries)):
                if (queries[i][1] is None) == (queries[j][1] is None) and rng.random() < 0.4:
                    for rel in ("equivalent", "implies"):
                        qs.append(meq.queries(i, j, relation=rel, nulls="sql"))
                        want.append(("q", rel, i, j))
        r = both(self, s, scope, qs)
        rows = rows_of(cols)
        for item, (kind, how, i, j) in zip(r, want):
            self.assertEqual(item["label"], "snapshot")
            if kind == "p":
                diff = lambda k: pred_differs(how, ev(preds[i], rows[k]), ev(preds[j], rows[k]))
                rel, sem = how
                if rel == "implies":
                    sql = f"coalesce({to_sql(preds[i])}, false) AND NOT coalesce({to_sql(preds[j])}, false)"
                elif sem == "kleene":
                    sql = f"{to_sql(preds[i])} IS DISTINCT FROM {to_sql(preds[j])}"
                else:
                    sql = f"coalesce({to_sql(preds[i])}, false) <> coalesce({to_sql(preds[j])}, false)"
            elif kind == "m":
                diff = lambda k: measure_differs(how, ev_poly(polys[i], rows[k]), ev_poly(polys[j], rows[k]))
                a, b = poly_sql(polys[i]), poly_sql(polys[j])
                sql = f"({a}) <> ({b})" if how == "skip" else f"({a}) IS DISTINCT FROM ({b})"
            else:
                (fa, ma), (fb, mb_) = queries[i], queries[j]
                def diff(k, fa=fa, fb=fb, ma=ma, mb_=mb_, rel=how):
                    ta = ev(preds[fa], rows[k]) == T_ if fa is not None else True
                    tb = ev(preds[fb], rows[k]) == T_ if fb is not None else True
                    if (ta and not tb) or (rel == "equivalent" and tb and not ta):
                        return True
                    if ta and tb and ma is not None:
                        return measure_differs("sql", ev_poly(polys[ma], rows[k]), ev_poly(polys[mb_], rows[k]))
                    return False
                sql = None
            expect = [k for k in range(n) if mask[k] and diff(k)]
            self.assertEqual(item["evidence"]["differ"], len(expect), (kind, how, i, j))
            if expect:
                self.check_witness_row(item, cols, diff)
                self.assertEqual(item["witness"], expect[0])
            else:
                self.assertEqual(item["verdict"], "proven")
            if sql is not None:
                self.assertEqual(duck(cols, sql, mask), (len(expect), expect[0] if expect else None))
        de_morgan = want.index(("p", ("equivalent", "kleene"), 14, 15))
        self.assertEqual(r[de_morgan]["verdict"], "proven")
        ev0 = r[de_morgan]["evidence"]
        self.assertEqual(sum(ev0["cells"].values()), sum(mask))          # the truth table over the selected rows

    def test_meta001_one_row_different_parts_give_that_row(self):
        rng = random.Random(5)
        n = 500
        x = [rng.randint(-50, 50) for _ in range(n)]
        x = [v if v != 7 else 8 for v in x]
        x[123] = 7                                        # x = 7 on exactly one row
        y = [rng.randint(-20, 20) for _ in range(n)]
        z = [a * b for a, b in zip(x, y)]
        bad = 337
        x[bad] = 1
        z = [a * b for a, b in zip(x, y)]
        z[bad] += 1                                       # z = x y on every row but one
        cols = {"x": x, "y": y, "z": z}
        scope = scope_of(cols)
        s = meq.Set(["x", "y", "z"],
                    predicates=[meq.cond("x", "<", 7), meq.cond("x", "<=", 7), meq.cond("x", "<", 8)],
                    measures=[mg.poly({(1, 1, 0): 1}, ["x", "y", "z"]), mg.poly({(0, 0, 1): 1}, ["x", "y", "z"])],
                    queries=[(0, 0), (2, 0), (2, 1), (1, 0)])
        qs = [meq.predicates(0, 1), meq.predicates(1, 2), meq.measures(0, 1), meq.queries(0, 1), meq.queries(1, 2),
              meq.queries(0, 3), meq.predicates(0, 1, relation="implies"), meq.predicates(1, 0, relation="implies")]
        r = both(self, s, scope, qs)
        self.assertEqual((r[0]["verdict"], r[0]["witness"], r[0]["evidence"]["differ"]), ("refuted", 123, 1))
        self.assertEqual(r[0]["point"], {"x": 7, "y": y[123], "z": z[123]})
        self.assertEqual(r[1]["verdict"], "proven")       # x <= 7 and x < 8 agree on integers
        self.assertEqual((r[2]["verdict"], r[2]["witness"], r[2]["evidence"]["differ"]), ("refuted", bad, 1))
        self.assertEqual(r[3]["witness"], 123)            # the filters differ on the row x = 7 only
        expect = [k for k in range(n) if x[k] < 8 and z[k] != x[k] * y[k]]
        self.assertEqual((r[4]["verdict"], r[4]["witness"]), ("refuted", expect[0]))
        self.assertEqual(r[5]["witness"], 123)
        self.assertEqual(r[6]["verdict"], "proven")       # x < 7 implies x <= 7
        self.assertEqual((r[7]["verdict"], r[7]["witness"]), ("refuted", 123))
        self.assertEqual(duck(cols, "z <> x * y"), (1, bad))

    def test_meta001_measure_null_semantics_and_certificates(self):
        rng = random.Random(3)
        n = 200
        x = [rng.randint(-9, 9) for _ in range(n)]
        y = [None if rng.random() < 0.2 else 0 for _ in range(n)]   # y = 0 where present
        cols = {"x": x, "y": y}
        scope = scope_of(cols)
        f, g = mg.poly({(1, 0): 1}, ["x", "y"]), mg.poly({(1, 0): 1, (0, 1): 1}, ["x", "y"])
        s = meq.Set(["x", "y"], measures=[f, g])
        r = both(self, s, scope, [meq.measures(0, 1, nulls="sql"), meq.measures(0, 1, nulls="skip")])
        first_null = y.index(None)
        self.assertEqual((r[0]["verdict"], r[0]["witness"]), ("refuted", first_null))
        self.assertEqual(r[0]["evidence"]["null_mismatch"], y.count(None))
        self.assertEqual(r[1]["verdict"], "proven")
        self.assertEqual((r[1]["evidence"]["excluded"], r[1]["evidence"]["nulls"]), (y.count(None), y.count(None)))
        cert = r.certificate(1)                           # FND-007: the snapshot of P = D (f - g) = 0, SKIP
        self.assertEqual(cert.kind, "snapshot")
        self.assertEqual(cert.info["nulls"], "skip")
        self.assertEqual(cert.evidence["checked"], n - y.count(None))
        self.assertTrue(cert.decide(scope)["authorized"])
        self.assertIsNone(r.certificate(0))

    # ---------------------------------------------------------------- the acceptance tests
    def test_meta001_snapshot_labels_never_become_universal(self):
        rng = random.Random(9)
        n = 120
        x = [rng.randint(0, 9) for _ in range(n)]
        y = [rng.randint(0, 9) for _ in range(n)]
        cols = {"x": x, "y": y, "z": [a * b for a, b in zip(x, y)]}
        scope = scope_of(cols)
        names = ["x", "y", "z"]
        s = meq.Set(names, predicates=[meq.cond("x", "<", 10), meq.cond("x", "<=", 9) & meq.cond("y", ">=", 0)],
                    measures=[mg.poly({(1, 1, 0): 1}, names), mg.poly({(0, 0, 1): 1}, names),
                              mg.poly({(1, 1, 0): 1, (0, 0, 0): 0}, names)])
        qs = [meq.measures(0, 1, nulls="skip"), meq.predicates(0, 1), meq.measures(0, 2)]
        snap = both(self, s, scope, qs)
        self.assertEqual([(i["verdict"], i["label"]) for i in snap], [("proven", "snapshot")] * 3)
        d = meq.domain({"x": (-20, 20), "y": (-20, 20), "z": (-400, 400)})
        uni = s.decide(qs, d)
        self.assertEqual((uni[0]["verdict"], uni[0]["label"], uni[0]["method"]), ("refuted", "domain", "grid"))
        pt = uni[0]["point"]
        self.assertNotEqual(pt["x"] * pt["y"], pt["z"])    # a concrete counterexample, in the domain
        self.assertEqual(uni[1]["verdict"], "refuted")    # x = -1 < 10 but y >= 0 fails somewhere in the box
        p = uni[1]["point"]
        self.assertNotEqual(ev(s.predicates[0], p) == T_, ev(s.predicates[1], p) == T_)
        self.assertEqual((uni[2]["verdict"], uni[2]["label"]), ("proven", "universal"))   # one canonical measure
        # authorization: the snapshot proof only on its exact scope, the universal one anywhere
        self.assertTrue(snap.authorize(0, scope)["authorized"])
        self.assertEqual(snap.authorize(0)["reason"], "no scope")
        changed = dict(cols)
        changed["z"] = list(changed["z"])
        changed["z"][5] += 1
        other = scope_of(changed)
        self.assertEqual(snap.authorize(0, other)["reason"], "stale")
        self.assertEqual(snap.authorize(1, other)["reason"], "stale")
        self.assertTrue(uni.authorize(2)["authorized"])
        self.assertEqual(uni.authorize(0, scope)["reason"], "not proven")
        # unsupported or over-budget universal questions answer unknown, never a guess
        s2 = meq.Set(["x", "y", "z"], predicates=[meq.cond("x", "bit", 0), meq.cond("x", "bit", 0) | meq.cond("x", "=", 4),
                                                   meq.cond("x", "<", 3), meq.cond("x", "<=", 2)])
        u2 = s2.decide([meq.predicates(0, 1), meq.predicates(2, 3)], meq.domain({"x": (-5, 5)}))
        self.assertEqual((u2[0]["verdict"], u2[0]["why"]), ("unknown", "unsupported"))   # BIT: outside the theory
        self.assertEqual((u2[1]["verdict"], u2[1]["label"]), ("proven", "domain"))
        u3 = s2.decide([meq.predicates(0, 1)], meq.domain({"x": (-5, 5)}, mode="enumerate"))
        self.assertEqual(u3[0]["verdict"], "refuted")     # enumeration decides it: x = 4 is even
        self.assertEqual(u3[0]["point"]["x"], 4)
        u4 = s2.decide([meq.predicates(0, 1)], meq.domain({"x": (-1000, 1000)}, mode="enumerate"),
                       budget=Budget(support=1000))
        self.assertEqual((u4[0]["verdict"], u4[0]["why"]), ("unknown", "budget"))
        self.assertEqual(u4[0]["budget"], {"resource": "support", "need": 2001, "limit": 1000})
        u5 = s2.decide([meq.predicates(2, 3)], meq.domain({"x": [1, 2, 3]}))
        self.assertEqual((u5[0]["verdict"], u5[0]["why"]), ("unknown", "unsupported"))   # finite sets: enumerate

    def test_meta001_universal_predicates_agree_with_exhaustive_enumeration(self):
        rng = random.Random(21)
        spec = {"x": (-3, 3), "y": (-2, 2), "z": (0, 2)}
        nullable = {"y", "z"}
        names = ["x", "y", "z"]
        pool = [random_pred(rng, names, 0) for _ in range(40)]
        preds = [random_pred(rng, names, 3, pool) for _ in range(24)]
        for i in range(0, 24, 3):                         # equivalent rewrites by construction
            preds.append(~~preds[i])
            preds.append((preds[i] | preds[i + 1]) & (preds[i] | ~preds[i + 1]) | (preds[i] & meq.UNKNOWN))
        s = meq.Set(names, predicates=preds)
        pts = list(points(spec, nullable))
        qs, want = [], []
        for _ in range(70):
            i, j = rng.randrange(len(preds)), rng.randrange(len(preds))
            rel, sem = rng.choice([("equivalent", "filter"), ("equivalent", "kleene"), ("implies", "filter")])
            qs.append(meq.predicates(i, j, relation=rel, semantics=sem))
            want.append(((rel, sem), i, j))
        for k, i in enumerate(range(0, 24, 3)):
            qs.append(meq.predicates(i, 24 + 2 * k, semantics="kleene"))
            want.append((("equivalent", "kleene"), i, 24 + 2 * k))
        dec = s.decide(qs, meq.domain(spec, nullable=nullable))
        enu = s.decide(qs, meq.domain(spec, nullable=nullable, mode="enumerate"))
        decided = 0
        for a, b, (how, i, j) in zip(dec, enu, want):
            diffs = [p for p in pts if pred_differs(how, ev(preds[i], p), ev(preds[j], p))]
            self.assertEqual(b["verdict"], "refuted" if diffs else "proven")
            self.assertEqual(b["label"], "domain")
            if b["point"] is not None:
                self.assertTrue(pred_differs(how, ev(preds[i], b["point"]), ev(preds[j], b["point"])))
            if a["verdict"] == "unknown":
                continue
            decided += 1
            self.assertEqual(a["verdict"], b["verdict"], (how, i, j))
            if a["verdict"] == "refuted":
                p = a["point"]
                self.assertTrue(all(p[c] is None or spec[c][0] <= p[c] <= spec[c][1] for c in names))
                self.assertTrue(pred_differs(how, ev(preds[i], p), ev(preds[j], p)))
        self.assertGreater(decided, 60)
        # identical Boolean structure over nullable columns: universal; with a non-nullable column: domain only
        a, b = meq.cond("x", "<", 1), meq.cond("y", "=", meq.column("x"))
        s2 = meq.Set(["x", "y"], predicates=[~(~a | ~b), a & b, ~(a & b) | ~b, ~a | ~b])
        q2 = [meq.predicates(0, 1, semantics="kleene"), meq.predicates(2, 3, semantics="kleene")]
        u = s2.decide(q2, meq.domain({"x": (-5, 5), "y": (-5, 5)}, nullable=["x", "y"]))
        self.assertEqual([(i["verdict"], i["label"]) for i in u], [("proven", "universal")] * 2)
        u = s2.decide(q2, meq.domain({"x": (-5, 5), "y": (-5, 5)}, nullable=["y"]))
        self.assertEqual([(i["verdict"], i["label"]) for i in u], [("proven", "domain")] * 2)

    def test_meta001_integer_theory_columns_scales_and_implication(self):
        names = ["x", "y", "z", "d"]
        P = [meq.cond("x", "<", meq.column("y")) & meq.cond("y", "<", meq.column("z")),   # 0
             meq.cond("x", "<", meq.column("z")),                                          # 1
             meq.cond("x", "<=", meq.column("z")) & meq.cond("x", "<>", meq.column("z")),   # 2
             meq.cond("d", "<", "1.5"), meq.cond("d", "<=", "1.49"), meq.cond("d", "<=", "1.4"),   # 3 4 5
             meq.cond("x", "between", 2, 4), meq.cond("x", ">=", 2) & ~meq.cond("x", ">", 4),     # 6 7
             meq.cond("x", "<>", 3) | meq.cond("y", "=", 3), meq.cond("x", "=", Fraction(7, 2))]  # 8 9
        s = meq.Set(names, predicates=P)
        qs = [meq.predicates(0, 1, relation="implies"), meq.predicates(1, 0, relation="implies"),
              meq.predicates(1, 2), meq.predicates(3, 4), meq.predicates(3, 5), meq.predicates(6, 7),
              meq.predicates(8, 6, relation="implies"), meq.predicates(9, 9), meq.predicates(9, 5, relation="implies")]
        spec = {"x": (-6, 6), "y": (-6, 6), "z": (-6, 6), "d": (100, 200)}
        u = s.decide(qs, meq.domain(spec, scales={"d": 2}))
        e = s.decide(qs, meq.domain(spec, scales={"d": 2}, mode="enumerate"), budget=Budget(support=1 << 22))
        verdicts = [i["verdict"] for i in u]
        self.assertEqual(verdicts, ["proven", "refuted", "proven", "proven", "refuted", "proven", "refuted", "proven",
                                    "proven"])
        self.assertEqual(verdicts, [i["verdict"] for i in e])
        sc = {"d": 2}
        for item, q in zip(u, qs):
            if item["verdict"] == "refuted":
                p = item["point"]
                a, b = ev(P[q[4]], p, sc), ev(P[q[5]], p, sc)
                self.assertTrue(pred_differs(("implies" if q[1] == 2 else "equivalent", "filter"), a, b))
        self.assertEqual(u[4]["point"]["d"] in range(141, 150), True)   # 1.41 .. 1.49: < 1.5 but > 1.4
        self.assertEqual(u[7]["label"], "universal")
        self.assertEqual(u[8]["verdict"], "proven")       # x = 7/2 never holds for an integer: implies anything

    def test_meta001_universal_measures_identities_and_grids(self):
        names = ["x", "y", "z"]
        sq = mg.poly({(2, 0, 0): 1, (1, 1, 0): 2, (0, 2, 0): 1}, names)
        P = [mg.poly([(1, {"x": 1}), (1, {"y": 1})], names), sq,                     # x + y, (x + y)^2 expanded
             mg.poly({(2, 0, 0): 1}, names), mg.poly({(1, 0, 0): 1}, names),          # x^2, x
             mg.poly({(1, 0, 0): Fraction(1, 2), (0, 0, 1): 1}, names),               # x/2 + z
             mg.poly({(3, 0, 0): 1, (1, 0, 0): -1}, names), mg.poly({}, names),       # x^3 - x, 0
             mg.poly({(1, 0, 1): 1}, names), mg.poly({(1, 0, 0): 1, (0, 0, 1): 1}, names)]   # x z, x + z
        s = meq.Set(names, measures=P)
        qs = [meq.measures(1, 1), meq.measures(2, 3), meq.measures(5, 6), meq.measures(3, 4), meq.measures(7, 8),
              meq.measures(0, 8, nulls="sql"), meq.measures(0, 8, nulls="skip")]
        dom = meq.domain({"x": (0, 1), "y": (-3, 3), "z": (-2, 2)}, nullable=["y"])
        u = s.decide(qs, dom)
        e = s.decide(qs, meq.domain({"x": (0, 1), "y": (-3, 3), "z": (-2, 2)}, nullable=["y"], mode="enumerate"))
        self.assertEqual((u[0]["verdict"], u[0]["label"], u[0]["method"]), ("proven", "universal", "normal form"))
        law = u.certificate(0)
        self.assertEqual(law.kind, "law")
        self.assertEqual((u[1]["verdict"], u[1]["label"]), ("proven", "domain"))   # x^2 = x on {0, 1}
        self.assertEqual((u[2]["verdict"], u[2]["label"]), ("proven", "domain"))   # x^3 - x = 0 on {0, 1}
        pts = list(points({"x": (0, 1), "y": (-3, 3), "z": (-2, 2)}, {"y"}))
        for k, q in enumerate(qs):
            nulls = "sql" if q[3] == 1 else "skip"
            diffs = [p for p in pts if measure_differs(nulls, ev_poly(P[q[4]], p), ev_poly(P[q[5]], p))]
            self.assertEqual(e[k]["verdict"], "refuted" if diffs else "proven", k)
            self.assertEqual(u[k]["verdict"], e[k]["verdict"], k)
            if u[k]["verdict"] == "refuted":
                p = u[k]["point"]
                self.assertTrue(measure_differs(nulls, ev_poly(P[q[4]], p), ev_poly(P[q[5]], p)))
        self.assertEqual(u[5]["point"]["y"], None)        # SQL: the column only one side reads is NULL
        self.assertEqual(u[6]["method"], "grid")
        big = s.decide([meq.measures(7, 8)], meq.domain({"x": (-10 ** 9, 10 ** 9), "y": (0, 1), "z": (-10 ** 9, 10 ** 9)}))
        p = big[0]["point"]
        self.assertNotEqual(p["x"] * p["z"], p["x"] + p["z"])   # the grid of the degrees, not the box
        self.assertLessEqual(big[0]["work"], 4)

    def test_meta001_measures_modulo_certified_relations(self):
        rng = random.Random(4)
        n = 150
        x = [rng.randint(-30, 30) for _ in range(n)]
        y = [rng.randint(-30, 30) for _ in range(n)]
        names = ["x", "y", "p", "w"]
        cols = {"x": x, "y": y, "p": [a * b for a, b in zip(x, y)], "w": [rng.randint(0, 5) for _ in range(n)]}
        scope = scope_of(cols)
        rel = mc.snapshot(scope, mc.polynomial({(0, 0, 1): 1, (1, 1, 0): -1}, ["x", "y", "p"], nulls="skip"))
        f = mg.poly({(1, 1, 0, 0): 3, (0, 0, 0, 1): 1}, names)          # 3 x y + w
        g = mg.poly({(0, 0, 1, 0): 3, (0, 0, 0, 1): 1}, names)          # 3 p + w
        h = mg.poly({(0, 0, 1, 0): 2, (0, 0, 0, 1): 1}, names)          # 2 p + w: different
        s = meq.Set(names, measures=[f, g, h])
        qs = [meq.measures(0, 1, nulls="skip"), meq.measures(0, 2, nulls="skip"), meq.measures(0, 1, nulls="sql")]
        plain = both(self, s, scope, qs)
        self.assertEqual([i["label"] for i in plain], ["snapshot"] * 3)
        r = both(self, s, scope, qs, relations=[rel])
        self.assertEqual((r[0]["verdict"], r[0]["label"], r[0]["method"]), ("proven", "relations", "reduction"))
        self.assertEqual(r[1]["verdict"], "refuted")
        self.assertEqual(r[2]["label"], "snapshot")       # SQL semantics with other columns: the rows decide
        cert = r.certificate(0)
        self.assertEqual(cert.kind, "snapshot")           # the weakest premise: a snapshot relation
        self.assertTrue(r.authorize(0, scope)["authorized"])
        moved = dict(cols)
        moved["p"] = list(moved["p"])
        moved["p"][0] += 1
        a = r.authorize(0, scope_of(moved))
        self.assertFalse(a["authorized"])
        self.assertEqual(a["reason"], "relations")
        self.assertEqual(s.decide(qs[:1], meq.domain({c: (-5, 5) for c in names}))[0]["verdict"], "refuted")

    # ---------------------------------------------------------------- families
    def test_meta001_query_families_share_scans(self):
        rng = random.Random(8)
        n = 400
        cols = {c: [None if rng.random() < 0.1 else rng.randint(-6, 6) for _ in range(n)] for c in ("x", "y", "z")}
        mask = [rng.random() < 0.85 for _ in range(n)]
        scope = scope_of(cols, mask)
        base = [meq.cond("x", ">", 0), meq.cond("y", "<=", 2), meq.cond("z", "is not null")]
        preds = [base[0] & base[1], base[0] & base[1] & base[2], base[1] | ~base[0], base[2], base[0] ^ base[1]]
        polys = [mg.poly({(1, 1): 1}, ["x", "y"]), mg.poly({(1, 0): Fraction(1, 3), (0, 1): 2}, ["y", "z"]),
                 mg.poly({(2,): 1}, ["x"])]
        queries = [(f, m) for f in range(len(preds)) for m in (None, 0, 1, 2)] + [(None, 1), (0, 0), (None, None)]
        s = meq.Set(["x", "y", "z"], predicates=preds, measures=polys, queries=queries)
        shared, st1 = s.run(scope, mode="shared")
        separate, st2 = s.run(scope, mode="separate")
        self.assertEqual(shared, separate)
        rows = rows_of(cols)
        for (f, m), a in zip(queries, shared):
            sel = [k for k in range(n) if mask[k] and (f is None or ev(preds[f], rows[k]) == T_)]
            self.assertEqual(a["count"], len(sel))
            if m is None:
                self.assertIsNone(a["sum"])
                continue
            vals = [ev_poly(polys[m], rows[k]) for k in sel]
            vals = [v for v in vals if v is not None]
            self.assertEqual(a["nonnull"], len(vals))
            self.assertEqual(a["sum"], sum(vals) if vals else None)
            where = "TRUE" if f is None else to_sql(preds[f])
            frame = pd.DataFrame({k: pd.array(v, dtype="Int64") for k, v in cols.items()})
            frame["sel"] = mask
            con = duckdb.connect()
            con.register("t", frame)
            got = con.execute(f"SELECT count(*), sum({poly_sql(polys[m])}) FROM t WHERE sel AND coalesce({where}, false)").fetchone()
            con.close()
            self.assertEqual(got[0], len(sel))
            if vals:
                self.assertEqual(Fraction(got[1]).limit_denominator(1000), a["sum"])
        self.assertEqual(st1["scans"], 1)
        self.assertEqual(st2["scans"], len(queries))
        info = s.info()
        self.assertEqual(st1["condition_rows"], info["conditions"] * sum(mask))
        self.assertLess(st1["condition_rows"], st2["condition_rows"])
        distinct = {}
        for f, m in queries:                              # rows where a distinct measure is evaluated once
            if m is None:
                continue
            for k in range(n):
                if mask[k] and (f is None or ev(preds[f], rows[k]) == T_) and ev_poly(polys[m], rows[k]) is not None:
                    distinct.setdefault(m, set()).add(k)
        self.assertEqual(st1["measure_rows"], sum(len(v) for v in distinct.values()))
        self.assertEqual(st2["measure_rows"], sum(a["nonnull"] for a in separate))
        self.assertLess(st1["node_words"], st2["node_words"])

    # ---------------------------------------------------------------- devices, budgets, bridges
    def test_meta001_device_evidence_is_checked_before_adoption(self):
        import ctypes as C
        rng = random.Random(2)
        n = 1000
        cols = {"x": [rng.randint(-3, 3) for _ in range(n)], "y": [None if rng.random() < 0.3 else rng.randint(0, 3) for _ in range(n)]}
        scope = scope_of(cols)
        s = meq.Set(["x", "y"], predicates=[meq.cond("x", "<", 2), meq.cond("x", "<=", 1), meq.cond("y", ">", 0)],
                    measures=[mg.poly({(1, 1): 1}, ["x", "y"]), mg.poly({(1, 0): 1}, ["x", "y"])],
                    queries=[(0, 0), (2, 1)])
        qs = [meq.predicates(0, 1), meq.predicates(0, 2), meq.measures(0, 1), meq.queries(0, 1)]
        good = both(self, s, scope, qs)
        self.assertEqual([i["verdict"] for i in good], ["proven", "refuted", "refuted", "refuted"])
        arr, nq = meq._questions(qs)
        error, refusal = C.c_int32(), meq._Refusal()
        k = meq._call("tr_equiv_check_make", s._handle, arr, nq, scope._handle, None, None, None, C.byref(refusal), C.byref(error))
        self.assertTrue(k)
        ev = (meq._Evidence * nq)()
        self.assertEqual(meq._call("tr_equiv_check_run", k, ev), 0)
        rels = (C.c_void_p * 1)()
        tampered = []
        for change in ("witness", "differ", "cells", "rows"):
            bad = (meq._Evidence * nq)()
            C.memmove(bad, ev, C.sizeof(ev))
            if change == "witness":
                bad[1].witness = bad[1].witness + 1 if bad[1].witness + 1 < n else bad[1].witness - 1
            elif change == "differ":
                bad[0].differ, bad[0].witness = 1, 3
            elif change == "cells":
                bad[0].cells[0] += 1
            else:
                bad[2].rows += 1
            h = meq._call("tr_equiv_check_decide", k, scope._handle, bad, 1, rels, 0, None, C.byref(refusal), C.byref(error))
            tampered.append((bool(h), error.value))
            if h:
                meq._call("tr_equiv_result_free", h)
        self.assertEqual(tampered, [(False, -1)] * 4)
        h = meq._call("tr_equiv_check_decide", k, scope._handle, ev, 1, rels, 0, None, C.byref(refusal), C.byref(error))
        self.assertTrue(h)
        meq._call("tr_equiv_result_free", h)
        meq._call("tr_equiv_check_free", k)

    def test_meta001_budgets_at_the_limit_and_one_past(self):
        preds = [meq.cond("x", "<", i) & meq.cond("y", ">", i) for i in range(5)]
        info = meq.Set(["x", "y"], predicates=preds).info()
        nodes = info["given_nodes"]
        meq.Set(["x", "y"], predicates=preds, budget=Budget(circuit=nodes))
        with self.assertRaises(AlgebraBudgetError) as cm:
            meq.Set(["x", "y"], predicates=preds, budget=Budget(circuit=nodes - 1))
        self.assertEqual((cm.exception.resource, cm.exception.need, cm.exception.limit), ("circuit", nodes, nodes - 1))
        s = meq.Set(["x", "y"], predicates=preds)
        cols = {"x": list(range(-50, 50)), "y": list(range(50, -50, -1))}
        scope = scope_of(cols)
        qs = [meq.predicates(i, j) for i in range(5) for j in range(5)]
        s.check(scope, qs, budget=Budget(output=len(qs)))
        with self.assertRaises(AlgebraBudgetError) as cm:
            s.check(scope, qs, budget=Budget(output=len(qs) - 1))
        self.assertEqual(cm.exception.resource, "output")
        need = s.check_need(scope, qs, backend="cuda")
        s.check(scope, qs, backend="cuda", budget=Budget(bytes=max(need["bytes"], 1 << 20)))
        with self.assertRaises(AlgebraBudgetError) as cm:
            s.check(scope, qs[:1], backend="cuda", budget=Budget(bytes=s.check_need(scope, qs[:1], "cuda")["bytes"] - 1))
        self.assertEqual(cm.exception.resource, "bytes")
        dom = meq.domain({"x": (0, 9), "y": (0, 9)}, mode="enumerate")
        at = s.decide([meq.predicates(0, 1)], dom, budget=Budget(support=100))
        self.assertEqual(at[0]["verdict"], "refuted")
        past = s.decide([meq.predicates(0, 1)], dom, budget=Budget(support=99))
        self.assertEqual(past[0]["budget"], {"resource": "support", "need": 100, "limit": 99})
        dec = s.decide([meq.predicates(0, 1)], meq.domain({"x": (0, 9), "y": (0, 9)}), budget=Budget(iterations=1))
        self.assertEqual((dec[0]["verdict"], dec[0]["why"], dec[0]["budget"]["resource"]), ("unknown", "budget", "iterations"))
        many = [meq.cond("x", "=", i) for i in range(65)]
        meq.Set(["x"], predicates=many[:64])
        with self.assertRaises(AlgebraBudgetError) as cm:                  # variables: the distinct conditions
            meq.Set(["x"], predicates=many[:64] + many[:64], budget=Budget(variables=63))
        self.assertEqual((cm.exception.resource, cm.exception.need, cm.exception.limit), ("variables", 64, 63))
        with self.assertRaises(UnsupportedError):                          # past BOOL-001's 64 masks
            meq.Set(["x"], predicates=many, budget=Budget(variables=100))

    def test_meta001_relationship_optimizer_rewrites_are_checked_not_trusted(self):
        from torbles.relationships import discover
        xs = [-3, -3, -1, -1, 4, 4, 2, 7]
        ys = [2, 5, 2, 5, 2, 5, -6, 1]
        frame = pd.DataFrame({"x": pd.Series(xs, dtype=object), "y": pd.Series(ys, dtype=object),
                              "p": pd.Series([a * b for a, b in zip(xs, ys)], dtype=object),
                              "z": pd.Series(list(range(8)), dtype=object)})
        db = torbles.from_ddl('CREATE TABLE t ("x" BIGINT, "y" BIGINT, "p" BIGINT, "z" BIGINT)', {"t": frame},
                              keep_plain=True)
        cat = discover(db, "t", ["x", "y", "p", "z"], modules=("product",), max_candidates=1000)
        self.addCleanup(cat.close)
        scope = mbk.Scope.from_table(db["t"], ["x", "y", "p", "z"])
        names = ["x", "y", "p", "z"]
        f = mg.poly({(1, 1, 0, 0): 5, (0, 0, 0, 1): 1}, names)                 # 5 x y + z
        g, trace = meq.rewrite(cat, scope, f)
        self.assertTrue(trace["changed"])
        self.assertLess(trace["after_cost"], trace["before_cost"])
        self.assertEqual(g.aligned(names), {(0, 0, 1, 0): 5, (0, 0, 0, 1): 1})
        s = meq.Set(names, measures=[f, g])
        snap = both(self, s, scope, [meq.measures(0, 1)])
        self.assertEqual((snap[0]["verdict"], snap[0]["label"]), ("proven", "snapshot"))
        uni = s.decide([meq.measures(0, 1)], meq.domain({c: (-9, 9) for c in names}))
        self.assertEqual(uni[0]["verdict"], "refuted")    # the rule holds on the table, not as a polynomial law

    def test_meta001_first_slice_polynomials_are_measures(self):
        from torbles import mathematics
        ctx = mathematics.Context()
        a, b = ctx.symbols("a", "b")
        e = (a + b) * (a - b) + ctx.constant(Fraction(1, 2))
        f = meq.measure_from_math(e, ["x", "y"])
        self.assertEqual(f.terms, mg.poly({(2, 0): 1, (0, 2): -1, (0, 0): Fraction(1, 2)}, ["x", "y"]).terms)
        s = meq.Set(["x", "y"], measures=[f, mg.poly({(2, 0): 1, (0, 2): -1, (0, 0): Fraction(1, 2)}, ["x", "y"])])
        u = s.decide([meq.measures(0, 1)], meq.domain({"x": (-3, 3), "y": (-3, 3)}))
        self.assertEqual((u[0]["verdict"], u[0]["label"]), ("proven", "universal"))


if __name__ == "__main__":
    unittest.main()
