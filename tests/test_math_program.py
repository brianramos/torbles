# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""META-002: query families as one multi-output program (torbles/math_program.c, math_program_gpu.cu,
math_program.py), executed through FND-006 (math_reduce).

Oracles, independent of the code under test: DuckDB executing every query of a family separately (WHERE, GROUP BY
over the scope's keys, a foreign-key dimension attribute through LEFT JOIN, a CASE map, no GROUP BY for scalar
queries; COUNT(*), COUNT(p), SUM(p) in HUGEINT on logical values, AVG as the exact Fraction SUM / COUNT; SQL NULLs),
Python three-valued evaluation of the predicates (snapshot comparisons, witnesses), Python integers. Every program is
run on every variant (fused / repeated pass x CPU / CUDA, finalization CPU / GPU; CUDA emulated here, scrambled
thread order) and the values are compared with the oracle and with each other. Seeds are fixed.
"""
import ctypes as C
import time
import unittest
from decimal import Decimal
from fractions import Fraction

import numpy as np
import pandas as pd

import torbles
from torbles import math_bank as mbk, math_bool as mb, math_program as mp
from torbles.math_budgets import Budget
from torbles.math_errors import StaleStateError
from torbles.mathematics import AlgebraBudgetError

DDL = """CREATE TABLE dim (id BIGINT PRIMARY KEY, region VARCHAR);
CREATE TABLE f (k BIGINT, x BIGINT, d DECIMAL(10,2), y BIGINT, c BIGINT REFERENCES dim(id), w BIGINT);"""
COLS = ["x", "d", "y", "w"]
SCALES = {"x": 0, "d": 2, "y": 0, "w": 0}
RUNS = [(p, b, f) for p in ("fused", "repeated") for b in ("cpu", "cuda") for f in ("cpu", "gpu")]


def obj(values):
    return pd.Series(values, dtype=object)


def dataset(seed, n, dup=True):
    rng = np.random.default_rng(seed)
    k = [int(v) for v in rng.integers(0, 6, n)]
    x = [int(v) for v in rng.integers(-500, 501, n)]
    d = [f"{int(v) / 100:.2f}" for v in rng.integers(-99999, 100000, n)]
    y = [int(v) for v in rng.integers(-30, 31, n)]
    c = [int(v) for v in rng.choice([10, 20, 30, 40, 99], n)]
    w = [int(v) for v in rng.integers(0, 1001, n)]
    for col, p in ((k, 0.08), (x, 0.15), (d, 0.15), (y, 0.2), (c, 0.1), (w, 0.1)):
        for i in np.flatnonzero(rng.random(n) < p):
            col[int(i)] = None
    for i in rng.choice(n, size=min(6, n), replace=False):
        k[int(i)], x[int(i)], d[int(i)] = 7, None, None
    if dup:
        m = n // 3
        k, x, d, y, c, w = (v + v[:m] for v in (k, x, d, y, c, w))
    fact = pd.DataFrame({"k": obj(k), "x": obj(x), "d": obj(d), "y": obj(y), "c": obj(c), "w": obj(w)})
    dim = pd.DataFrame({"id": [10, 20, 30, 40], "region": obj(["EU", "US", None, "EU"])})
    return torbles.from_ddl(DDL, {"f": fact, "dim": dim}), fact


CONDS = [mb.condition("x", ">", 0), mb.condition("d", "is null"), mb.condition("y", "<", 0),
         mb.condition("x", "between", -100, 100), mb.condition("x", ">=", 1), mb.condition("w", "<=", 499),
         mb.condition("x", "<", mb.column("y")), mb.condition("d", ">=", "1.5"), mb.condition("x", ">", 7),
         mb.condition("x", ">=", 7), mb.condition("x", ">", 1000), mb.condition("x", ">", 2000),
         mb.condition("x", "bit", 0), mb.condition("y", "is not null")]
SQL = ["x > 0", "d IS NULL", "y < 0", "x BETWEEN -100 AND 100", "x >= 1", "w <= 499", "x < y", "d >= 1.5", "x > 7",
       "x >= 7", "x > 1000", "x > 2000", None, "y IS NOT NULL"]


def term_sql(monomial):
    """a monomial string in logical units (decimal factors scaled), as HUGEINT"""
    if monomial.strip() == "1":
        return "CAST(1 AS HUGEINT)"
    out = []
    for f in monomial.split("*"):
        name, _, e = f.partition("^")
        col = f'CAST(f."{name}" * {10 ** SCALES[name]} AS HUGEINT)' if SCALES[name] else f'CAST(f."{name}" AS HUGEINT)'
        out += [col] * int(e or 1)
    return "(" + " * ".join(out) + ")"


def poly_sql(poly):
    items = [(poly, 1)] if isinstance(poly, str) else (list(poly.items()) if isinstance(poly, dict) else
                                                       [(m, c) for c, m in poly])
    return "(" + " + ".join(f"({int(c)}) * {term_sql(m)}" for m, c in items) + ")" if items else "CAST(0 AS HUGEINT)"


class Fixture:
    def __init__(self, seed=6, n=900):
        self.db, self.fact = dataset(seed, n)
        t, dim = self.db["f"], self.db["dim"]
        self.t = t
        self.scope = mbk.Scope.from_table(t, COLS, group_by=["k"])
        self.by_c = mbk.Scope.from_table(t, [], group_by=[("c", dim, "id")])
        self.by_region = mbk.Scope.from_table(t, [], group_by=[("c", dim, "region")])
        self.coarse = mbk.Scope.from_table(t, COLS)
        keys = [self.scope.key_values(g)[0] for g in range(self.scope.groups)]
        self.parity = np.array([0 if k is None else 1 + k % 2 for k in keys], dtype=np.uint32)

    def family(self):
        fam = mp.Family(self.scope, conditions=CONDS,
                        groupings={"c": self.by_c, "region": self.by_region, "all": mp.rollup(self.coarse),
                                   "parity": mp.rollup(map=self.parity, groups=3)})
        self.queries = {}
        v = mb.variables(len(CONDS))

        def q(name, measures, where, wsql, group=None, scalar=None):
            fam.query(name, [m for _, m, _ in measures], where=where, group=group, scalar=scalar)
            self.queries[name] = (measures, wsql, group, scalar)

        q("Q1", [("n", mp.count(name="n"), ("count",)), ("sx", mp.sum("x", name="sx"), ("sum", "x")),
                 ("a", mp.avg({"x": 2, "x*y": -1, "1": 3}, name="a"), ("avg", {"x": 2, "x*y": -1, "1": 3}))],
          v[0], "x > 0")
        q("Q2", [("cy", mp.count("y", name="cy"), ("count", "y")), ("sxy", mp.sum("x*y", name="sxy"), ("sum", "x*y")),
                 ("h", mp.histogram(["y"], name="h"), ("hist", "y", None))], v[4], "x >= 1", group="c")
        q("Q3", [("sxd", mp.sum("x*d", name="sxd"), ("sum", "x*d")), ("n", mp.count(name="n"), ("count",))],
          ~~v[0], "x > 0", group="all", scalar=True)
        q("Q4", [("sx", mp.sum("x", name="sx"), ("sum", "x")), ("ay", mp.avg("y", name="ay"), ("avg", "y"))],
          v[3] & ~v[1], "x BETWEEN -100 AND 100 AND d IS NOT NULL")
        q("Q5", [("n", mp.count(name="n"), ("count",)), ("sd2", mp.sum({"d^2": 1}, name="sd2"), ("sum", "d^2")),
                 ("aw", mp.avg("w", name="aw"), ("avg", "w"))], v[2] | v[1], "y < 0 OR d IS NULL", group="region")
        q("Q6", [("sxy", mp.sum({"x": 1, "y": 1}, name="sxy"), ("sum", {"x": 1, "y": 1})),
                 ("cxy", mp.count({"x": 1, "y": 1}, name="cxy"), ("count", {"x": 1, "y": 1}))],
          ~v[1] & v[3], "x BETWEEN -100 AND 100 AND d IS NOT NULL", group="parity")
        q("Q7", [("n", mp.count(name="n"), ("count",)), ("sw", mp.sum("w", name="sw"), ("sum", "w"))], None, None,
          group="c")
        q("Q8", [("a", mp.avg({"x*d": 1, "w": -2}, name="a"), ("avg", {"x*d": 1, "w": -2})),
                 ("s1", mp.sum("1", name="s1"), ("sum", "1")), ("cd", mp.count("d", name="cd"), ("count", "d"))],
          v[6] & v[7], "x < y AND d >= 1.5")
        q("Q9", [("n", mp.count(name="n"), ("count",)), ("sx", mp.sum("x", name="sx"), ("sum", "x"))], mb.FALSE,
          "FALSE")
        q("Q10", [("sx", mp.sum("x", name="sx"), ("sum", "x")), ("n", mp.count(name="n"), ("count",))],
          ~(~v[0] | v[1]), "x > 0 AND d IS NOT NULL", group="all", scalar=True)
        q("Q11", [("hw", mp.histogram(["x"], weight="w", name="hw"), ("hist", "x", "w"))], v[5], "w <= 499")
        q("Q12", [("s5", mp.sum([(2, "x"), (3, "x")], name="s5"), ("sum", {"x": 5}))], v[0], "x > 0")
        return fam

    KEYS = {None: ('f."k"', ""), "c": ("dim.id", "LEFT JOIN dim ON f.c = dim.id"),
            "region": ("dim.region", "LEFT JOIN dim ON f.c = dim.id"), "all": (None, ""),
            "parity": ('CASE WHEN f."k" IS NULL THEN 0 WHEN f."k" % 2 = 0 THEN 1 ELSE 2 END', "")}

    def duck(self, name):
        """{key: {label: value}} of one query executed by DuckDB on its own"""
        measures, wsql, group, scalar = self.queries[name]
        key, join = self.KEYS[group]
        where = wsql or "TRUE"
        con = self.db.con
        out = {}
        hists = [m for m in measures if m[2][0] == "hist"]
        cols, labels = [], []
        for label, _, spec in measures:
            if spec[0] == "count" and len(spec) == 1:
                cols.append("COUNT(*)"); labels.append((label, "n"))
            elif spec[0] == "count":
                cols.append(f"COUNT({poly_sql(spec[1])})"); labels.append((label, "n"))
            elif spec[0] in ("sum", "avg"):
                cols += [f"SUM({poly_sql(spec[1])})", f"COUNT({poly_sql(spec[1])})"]; labels.append((label, spec[0]))
        cols.append("COUNT(*)")
        sel = (f"{key} AS g, " if key else "") + ", ".join(cols)
        sql = f"SELECT {sel} FROM f {join} WHERE {where}" + (" GROUP BY g" if key else "")
        for r in con.execute(sql).fetchall():
            r = list(r)
            k = (r.pop(0),) if key else ()
            if key and r[-1] == 0:
                continue
            row, i = {}, 0
            for label, kind in labels:
                if kind == "n":
                    row[label] = int(r[i]); i += 1
                else:
                    s, n = r[i], r[i + 1]
                    i += 2
                    row[label] = None if n == 0 or s is None else (int(s) if kind == "sum" else Fraction(int(s), int(n)))
            out[k] = row
        for label, _, (_, col, weight) in hists:
            extra = f', COUNT(f."{weight}"), SUM(CAST(f."{weight}" AS HUGEINT))' if weight else ""
            gsel = f"{key} AS g, " if key else ""
            rows = con.execute(f'SELECT {gsel}f."{col}" AS v, COUNT(*){extra} FROM f {join} WHERE {where} GROUP BY '
                               + ("g, v" if key else "v")).fetchall()
            cells = {}
            for r in rows:
                r = list(r)
                k = (r.pop(0),) if key else ()
                cells.setdefault(k, []).append(((r[0],), int(r[1])) + ((int(r[2]), 0 if r[3] is None else int(r[3]))
                                                                      if weight else ()))
            for k, cs in cells.items():
                out.setdefault(k, {})[label] = sorted(cs, key=lambda c: (c[0][0] is None, c[0][0] or 0))
        return out


def frac(v):
    return None if v is None else Fraction(Decimal(str(v)))


def kand(a, b):
    return False if a is False or b is False else (None if a is None or b is None else True)


def kor(a, b):
    return True if a is True or b is True else (None if a is None or b is None else False)


def knot(a):
    return None if a is None else (not a)


def mine(qres):
    return {row["key"]: {k: v for k, v in row.items() if k != "key"} for row in qres.rows()}


class Program(unittest.TestCase):
    maxDiff = None

    def test_meta002_joint_outputs_match_queries_executed_separately(self):
        fx = Fixture()
        prog = fx.family().compile(prove=True)
        first = None
        for plan, backend, fin in RUNS:
            res = prog.run(plan=plan, backend=backend, finalize=fin)
            self.assertEqual(res.how["plan"], plan)
            got = {name: mine(res[name]) for name in fx.queries}
            if first is None:
                first = got
                for name in fx.queries:
                    self.assertEqual(got[name], fx.duck(name), name)
            else:
                self.assertEqual(got, first, (plan, backend, fin))

    def test_meta002_incompatible_and_overlapping_groupings_share_one_pass(self):
        fx = Fixture(seed=9, n=600)
        prog = fx.family().compile()
        res = prog.run(plan="fused")
        self.assertEqual((res.stats["scans"], res.stats["native_calls"]), (1, 1))
        info = prog.info
        self.assertLess(info["outputs"], info["nmeasures"])          # states shared by several measures
        groupings = {q["grouping"] for q in (prog.query_info(n) for n in prog.names)}
        self.assertEqual(len(groupings), 5)                        # k, c, region (incompatible), all, parity
        # Q1, Q12 (same predicate class and grouping) read the same SQL-mode state
        e1, e12 = prog.explain("Q1"), prog.explain("Q12")
        self.assertTrue(set(s["state"] for s in e1["states"]) & set(s["state"] for s in e12["states"]))
        self.assertIn("Q12", next(s for s in e1["states"] if s["mode"] == "sql")["shared_with"])
        # Q4 and Q6: the same predicate, overlapping groupings (k and its parity map): one class, two states
        self.assertEqual(prog.query_info("Q4")["klass"], prog.query_info("Q6")["klass"])
        for name in fx.queries:
            self.assertEqual(mine(res[name]), fx.duck(name), name)

    def test_meta002_cancelled_terms_keep_the_sql_validity_of_the_written_measure(self):
        fx = Fixture(seed=17, n=500)
        fam = mp.Family(fx.scope, conditions=CONDS, groupings={"all": mp.rollup(fx.coarse)})
        v = mb.variables(len(CONDS))
        polys = {"xmx": [(1, "x"), (-1, "x")],             # x - x: 0 where x is present, NULL where it is not
                 "xyy": [(1, "x"), (1, "y"), (-1, "y")],   # x + y - y: NULL where y is
                 "zd": [(0, "d"), (2, "x")],               # 0 * d + 2 x: NULL where d is
                 "one": [(1, "1"), (-1, "1")]}             # 1 - 1: no column, valid on every row
        fx.queries = {}
        specs = []
        for name, poly in polys.items():
            specs += [(f"c_{name}", mp.count(poly, name=f"c_{name}"), ("count", poly)),
                      (f"s_{name}", mp.sum(poly, name=f"s_{name}"), ("sum", poly)),
                      (f"a_{name}", mp.avg(poly, name=f"a_{name}"), ("avg", poly))]
        for qname, where, wsql, group, scalar in (("grouped", v[3], "x BETWEEN -100 AND 100", None, None),
                                                  ("scalar", v[0] & v[2], "x > 0 AND y < 0", "all", True)):
            fam.query(qname, [m for _, m, _ in specs], where=where, group=group, scalar=scalar)
            fx.queries[qname] = (specs, wsql, group, scalar)
        pairs = {"cx": ("count", "x"), "c5x": ("count", {"x": 5}), "c1": ("count", "1"), "n": ("count",),
                 "sx": ("sum", "x"), "sxyy": ("sum", polys["xyy"]), "cxmx": ("count", polys["xmx"]),
                 "sxmx": ("sum", polys["xmx"]), "sone": ("sum", polys["one"])}
        for name, spec in pairs.items():
            m = mp.count(name=name) if spec == ("count",) else getattr(mp, spec[0])(spec[1], name=name)
            fam.query(name, [m], where=v[3])
            fx.queries[name] = ([(name, m, spec)], "x BETWEEN -100 AND 100", None, None)
        prog = fam.compile()
        first = None
        for plan, backend, fin in RUNS:
            res = prog.run(plan=plan, backend=backend, finalize=fin)
            got = {name: mine(res[name]) for name in fx.queries}
            if first is None:
                first = got
                for name in fx.queries:
                    self.assertEqual(got[name], fx.duck(name), name)
            self.assertEqual(got, first, (plan, backend, fin))
        # the written columns are a listwise state's: x + y - y reads y although no canonical monomial does
        states = prog.explain("grouped")["states"]
        self.assertTrue(any(s["mode"] == "joint" for s in states), states)
        # measures compared by their values on every database
        same = lambda a, b: prog.compare(a, b)["measures_equal"]
        self.assertTrue(same("cx", "c5x"))           # COUNT ignores the coefficients
        self.assertTrue(same("cx", "cxmx"))          # COUNT(x - x) counts the rows where x is present
        self.assertTrue(same("c1", "n"))             # COUNT(1) is COUNT(*)
        self.assertFalse(same("sx", "sxyy"))         # SUM(x + y - y) skips the rows where y is NULL
        self.assertFalse(same("sxmx", "sone"))       # 0 where x is present vs 0 on every row
        self.assertTrue(prog.compare("cx", "c5x")["equivalent_queries"])

    def test_meta002_shared_negations_compile_in_linear_time(self):
        fx = Fixture(seed=19, n=400)
        fam = mp.Family(fx.scope, conditions=CONDS)
        v = mb.variables(len(CONDS))
        pool = [0, 2, 3, 5, 6, 7, 8, 13]
        level = v[1]
        depth = 26
        for i in range(depth):                      # every level reads the previous one twice
            a, b = pool[i % len(pool)], pool[(3 * i + 1) % len(pool)]
            level = (level & v[a]) | (level & ~v[b])
        fam.query("deep", [mp.count()], where=~level)
        fam.query("deep3", [mp.count()], where=~~~level)
        t0 = time.perf_counter()
        prog = fam.compile()
        self.assertLess(time.perf_counter() - t0, 20)          # each node is negated once (2^26 times otherwise)
        self.assertLess(prog.info["nodes"], 2000)
        self.assertEqual(prog.query_info("deep3")["klass"], "deep")
        # the oracle: the predicate evaluated row by row with SQL's three-valued logic
        f = fx.fact
        x, d, y, w = f["x"].tolist(), f["d"].tolist(), f["y"].tolist(), f["w"].tolist()
        def cond(i, r):
            xv, dv, yv, wv = x[r], frac(d[r]), y[r], w[r]
            return {0: None if xv is None else xv > 0, 1: dv is None, 2: None if yv is None else yv < 0,
                    3: None if xv is None else -100 <= xv <= 100, 5: None if wv is None else wv <= 499,
                    6: None if xv is None or yv is None else xv < yv, 7: None if dv is None else dv >= Fraction(3, 2),
                    8: None if xv is None else xv > 7, 13: yv is not None}[i]
        kept = 0
        for r in range(len(f)):
            lv = cond(1, r)
            for i in range(depth):
                a, b = pool[i % len(pool)], pool[(3 * i + 1) % len(pool)]
                lv = kor(kand(lv, cond(a, r)), kand(lv, knot(cond(b, r))))
            kept += knot(lv) is True
        for plan, backend, fin in RUNS[::3]:
            res = prog.run(plan=plan, backend=backend, finalize=fin)
            self.assertEqual(sum(row["count(*)"] for row in res["deep"].rows()), kept, (plan, backend, fin))

    def test_meta002_canonical_conditions_and_predicates_merge(self):
        fx = Fixture(seed=3, n=400)
        fam = mp.Family(fx.scope, conditions=CONDS)
        v = mb.variables(len(CONDS))
        forms = [v[0], v[4], ~~v[0], ~(~v[0]), v[0] & v[0], v[0] | mb.FALSE, (v[0] & mb.TRUE) | (v[0] & v[0]),
                 ~(~v[4])]
        for i, f in enumerate(forms):
            fam.query(f"f{i}", [mp.count(), mp.sum("x")], where=f)
        fam.query("n0", [mp.count()], where=~v[0])                  # NOT x > 0 is x <= 0: another class
        fam.query("assoc1", [mp.count()], where=(v[0] & v[1]) & v[2])
        fam.query("assoc2", [mp.count()], where=v[2] & (v[1] & v[0]))
        fam.query("dm1", [mp.count()], where=~(v[1] | v[2]))
        fam.query("dm2", [mp.count()], where=~v[1] & ~v[2])
        fam.query("t1", [mp.count()])                               # no WHERE
        fam.query("t2", [mp.count()], where=mb.TRUE)                # WHERE TRUE: the same, canonically
        fam.query("t3", [mp.count()], where=~mb.FALSE | v[2])
        fam.query("lem", [mp.count()], where=v[0] | ~v[0])          # UNKNOWN where x is NULL: not TRUE
        prog = fam.compile()
        classes = {n: prog.query_info(n)["klass"] for n in prog.names}
        self.assertEqual({classes[f"f{i}"] for i in range(len(forms))}, {"f0"})
        self.assertEqual(prog.query_info("f1")["proof"], "canonical")
        self.assertNotEqual(classes["n0"], "f0")
        self.assertEqual(classes["assoc2"], classes["assoc1"])
        self.assertEqual(classes["dm2"], classes["dm1"])
        self.assertEqual({classes["t1"], classes["t2"], classes["t3"]}, {"t1"})
        self.assertEqual(prog.query_info("t2")["filter"], -1)
        self.assertNotEqual(classes["lem"], "t1")
        c = prog.compare("t1", "lem")
        self.assertEqual((c["outcome"], c["witness"]["row"], c["witness"]["accepted_by"]), ("different", {"x": None}, "t1"))
        info = prog.info
        self.assertLess(info["conds"], len(CONDS) + 4)               # x >= 1 is x > 0; complements interned
        res = prog.run()
        mask = np.array([x is not None and x > 0 for x in fx.fact["x"]])
        total = int(mask.sum())
        for i in range(len(forms)):
            self.assertEqual(sum(r["count(*)"] for r in res[f"f{i}"].rows()), total)

    def test_meta002_equivalent_one_row_different_and_universally_different_pairs(self):
        fx = Fixture(seed=4, n=500)
        x = fx.fact["x"].tolist()
        # make exactly one row hold x = 7 (one-row-different pair x > 7 / x >= 7)
        fact = fx.fact.copy()
        fact["x"] = [7 if i == 5 else (8 if v == 7 else v) for i, v in enumerate(x)]
        db = torbles.from_ddl(DDL, {"f": fact, "dim": pd.DataFrame({"id": [10, 20, 30, 40],
                                                                    "region": obj(["EU", "US", None, "EU"])})})
        scope = mbk.Scope.from_table(db["f"], COLS, group_by=["k"])
        fam = mp.Family(scope, conditions=CONDS)
        v = mb.variables(len(CONDS))
        pairs = {"canonical": (v[0], v[4]), "search": ((v[0] & v[1]) | (v[0] & v[13] & ~v[1]),   # distributivity
                                                       v[0] & (v[1] | (v[13] & ~v[1]))),
                 "one_row": (v[8], v[9]), "universal_only": (v[10], v[11]), "unknown": (v[12], v[12] & v[0]),
                 "complement": (v[0], ~v[0])}
        for name, (a, b) in pairs.items():
            fam.query(name + "_a", [mp.count()], where=a)
            fam.query(name + "_b", [mp.count()], where=b)
        prog = fam.compile()
        u = {name: prog.compare(name + "_a", name + "_b") for name in pairs}
        s = {name: prog.snapshot_compare(name + "_a", name + "_b") for name in pairs}
        self.assertEqual((u["canonical"]["outcome"], u["canonical"]["how"]), ("equivalent", "canonical"))
        self.assertEqual((u["search"]["outcome"], u["search"]["how"]), ("equivalent", "search"))
        self.assertGreater(u["search"]["iterations"], 0)
        self.assertEqual(u["one_row"]["outcome"], "different")
        self.assertEqual(u["one_row"]["witness"]["row"], {"x": 7})
        self.assertEqual(u["one_row"]["witness"]["accepted_by"], "one_row_b")
        self.assertEqual((s["one_row"]["differ"], s["one_row"]["first"]), (1, 5))
        self.assertEqual(u["universal_only"]["outcome"], "different")              # a witness no row of the data has
        self.assertTrue(s["universal_only"]["equal"])                              # equal on this snapshot only
        self.assertEqual(s["universal_only"]["rows"], (0, 0))
        self.assertEqual(u["unknown"]["outcome"], "unknown")                       # BIT: feasibility not decided
        self.assertEqual(u["complement"]["outcome"], "different")
        # the witness rows are checked by the oracle: one predicate TRUE, the other not
        for name in ("one_row", "universal_only", "complement"):
            row = u[name]["witness"]["row"]
            xv = row.get("x")
            truth = lambda f: f(xv)
            evals = {"one_row": (lambda q: q is not None and q > 7, lambda q: q is not None and q >= 7),
                     "universal_only": (lambda q: q is not None and q > 1000, lambda q: q is not None and q > 2000),
                     "complement": (lambda q: q is not None and q > 0, lambda q: q is not None and not q > 0)}[name]
            ta, tb = truth(evals[0]), truth(evals[1])
            self.assertNotEqual(ta, tb, name)
            self.assertEqual(u[name]["witness"]["accepted_by"], name + ("_a" if ta else "_b"))
        # snapshot counts against the Python oracle
        xs = fact["x"].tolist()
        rows_a = sum(1 for q in xs if q is not None and q > 7)
        self.assertEqual(s["one_row"]["rows"], (rows_a, rows_a + 1))
        # a budget-limited search keeps the outcome "budget"; the same search with its need admitted decides
        need = u["search"]["iterations"]
        self.assertEqual(need, 3 ** 4)                   # x > 0, d IS NULL, d IS NOT NULL, y IS NOT NULL
        self.assertEqual(prog.compare("search_a", "search_b", budget=Budget(iterations=need))["outcome"], "equivalent")
        self.assertEqual(prog.compare("search_a", "search_b", budget=Budget(iterations=need - 1))["outcome"], "budget")

    def test_meta002_proof_search_merges_at_compile_time(self):
        fx = Fixture(seed=7, n=400)
        fam = mp.Family(fx.scope, conditions=CONDS)
        v = mb.variables(len(CONDS))
        fam.query("a", [mp.count(), mp.sum("x")], where=(v[0] & v[1]) | (v[0] & v[13] & ~v[1]))
        fam.query("b", [mp.count(), mp.sum("x")], where=v[0] & (v[1] | (v[13] & ~v[1])))
        fam.query("c", [mp.count()], where=v[0] & v[1])
        plain, proved = fam.compile(), fam.compile(prove=True)
        self.assertNotEqual(plain.query_info("b")["klass"], "a")
        self.assertEqual((proved.query_info("b")["klass"], proved.query_info("b")["proof"]), ("a", "search"))
        self.assertEqual(proved.info["merged_search"], 1)
        self.assertLess(proved.info["outputs"], plain.info["outputs"])
        r1, r2 = plain.run(), proved.run(backend="cuda")
        for n in ("a", "b", "c"):
            self.assertEqual(r1[n].rows(), r2[n].rows())
        # a search budget too small leaves the pair apart, recorded as budget-limited
        limited = fam.compile(prove=True, budget=Budget(iterations=10))
        self.assertEqual(limited.info["merged_search"], 0)
        self.assertGreaterEqual(limited.info["search_budget"], 1)

    # ------------------------------------------------------------------ costs, devices, lifetimes
    def test_meta002_memory_dispatch_costs_and_paths_are_recorded(self):
        fx = Fixture(seed=2, n=400)
        prog = fx.family().compile()
        from torbles import math_gpu
        dev = "cuda-emulated" if math_gpu.device()["emulated"] else "cuda"
        for plan, backend, fin in RUNS:
            res = prog.run(plan=plan, backend=backend, finalize=fin)
            want = dict(plan=plan, pass_device=backend if backend == "cpu" else dev,
                        finalize_device=fin if fin == "cpu" else dev)
            self.assertEqual(res.how, want)
            for name in prog.names:
                e = res[name].explanation
                self.assertEqual(e["devices"], want)
                self.assertEqual(e["dispatch"]["shared_pass"]["plan"], plan)
                self.assertEqual(e["dispatch"]["finalize"]["device"], want["finalize_device"])
                self.assertEqual(e["value_bytes"], e["value_slots"] * 72)
                for state, nbytes, readers in e["memory"]["states"]:
                    self.assertGreater(nbytes, 0)
                    self.assertGreaterEqual(readers, 1)
                for m in e["measures"]:                     # per measure: where its state and its values came from
                    hist = m["kind"] == "histogram"
                    self.assertEqual(m["path"], dict(state=want["pass_device"],
                                                     values=None if hist else want["finalize_device"]), (name, m))
                    self.assertEqual(m["slots"] == 0, hist)
            st = res.stats
            if plan == "fused" and backend == "cuda" and fin == "gpu":
                self.assertEqual(st["kernels"], 2)           # the fused pass and the finalization, states kept on the device
            if plan == "fused":
                self.assertEqual(st["scans"], 1)
            else:
                self.assertGreater(st["scans"], 1)

    def test_meta002_no_state_reuse_outside_declared_lifetimes(self):
        fx = Fixture(seed=13, n=400)
        run_prog = fx.family().compile(lifetime="run")
        r1, r2 = run_prog.run(), run_prog.run()
        self.assertFalse(r1.reused or r2.reused)
        self.assertEqual((r1.stats["scans"], r2.stats["scans"]), (1, 1))     # each run traverses again
        self.assertIsNot(r1.result, r2.result)
        snap = fx.family().compile(lifetime="snapshot")
        s1, s2 = snap.run(), snap.run(backend="cuda")
        self.assertFalse(s1.reused)
        self.assertTrue(s2.reused)                                            # declared: the same snapshot
        self.assertEqual(s2.stats["scans"], 0)
        self.assertIs(s1.result, s2.result)
        for n in snap.names:
            self.assertEqual(s1[n].rows(), s2[n].rows())
            self.assertEqual(s1[n].rows(), r1[n].rows())
        other = fx.family().compile(lifetime="snapshot")                      # another program: nothing shared
        o1 = other.run()
        self.assertFalse(o1.reused)
        self.assertIsNot(o1.result, s1.result)
        snap.drop()
        self.assertFalse(snap.run().reused)                                   # the lifetime ended explicitly
        fx.t.flip(0, 0)                                                       # the snapshot changes: stale
        for p in (snap, run_prog, other):
            with self.assertRaises(StaleStateError):
                p.run()
        self.assertIsNone(snap._kept)

    def test_meta002_budgets_at_the_limit_and_one_past(self):
        fx = Fixture(seed=5, n=300)
        fam = fx.family()
        prog = fam.compile()
        oracle = {n: mine(prog.run()[n]) for n in prog.names}
        # a run checks its whole need (the FND-006 pass and the finalization) before anything is allocated
        for plan, backend, fin in RUNS:
            need = prog.need(plan, backend, fin)
            self.assertTrue({"circuit", "output", "states", "bytes"} <= set(need), need)
            for resource, value in need.items():
                res = prog.run(plan=plan, backend=backend, finalize=fin, budget=Budget(**{resource: value}))
                self.assertEqual({n: mine(res[n]) for n in prog.names}, oracle, (plan, backend, fin, resource))
                with self.assertRaises(AlgebraBudgetError) as cm:
                    prog.run(plan=plan, backend=backend, finalize=fin, budget=Budget(**{resource: value - 1}))
                self.assertEqual((cm.exception.resource, cm.exception.need, cm.exception.limit),
                                 (resource, value, value - 1), (plan, backend, fin))
        # the finalization entry points check their own needs (CPU: the views and the values; GPU: the device copies)
        res = prog.run(plan="fused")
        nslots = prog.info["slots"]
        for finalize, states in ((0, 0), (1, 1)):
            raw = mp._Need()
            mp._call("tr_prog_final_need", prog._handle, finalize, states, C.byref(raw))
            fin_bytes = int(raw.need[8])
            for limit, code in ((fin_bytes, 0), (fin_bytes - 1, -5)):
                values, refusal = (mp._Value * max(nslots, 1))(), mp._Refusal()
                if finalize == 0:
                    rc = mp._call("tr_prog_finalize", prog._handle, res.result._open(), C.cast(values, C.c_void_p),
                                  nslots, C.byref(Budget(bytes=limit).raw), C.byref(refusal))
                else:
                    gpu = mp._gpu()
                    outs, fv = (mp._FOut * max(prog.info["outputs"], 1))(), mp._FinalView()
                    mp._call("tr_prog_final_view_get", prog._handle, res.result._open(), outs, len(outs), C.byref(fv))
                    rc = gpu.lib().tr_mgpu_program_finalize(C.byref(fv), C.byref(Budget(bytes=limit).raw),
                                                            C.cast(values, C.c_void_p), C.byref(refusal))
                self.assertEqual(rc, code, (finalize, limit))
                if code:
                    self.assertEqual((refusal.need, refusal.limit), (fin_bytes, fin_bytes - 1))
        # tr_mgpu_program_run: the fused pass's device need and the finalization's device tables (states left there)
        raw = mp._Need()
        mp._call("tr_prog_final_need", prog._handle, 1, 0, C.byref(raw))
        run_bytes = prog._rplan.need("fused", "kernel")["bytes"] + int(raw.need[8])
        gpu = mp._gpu()
        view, base = mp.mr._View(), mp._Need()
        mp.mr._call("tr_reduce_get_view", prog._rplan._open(), C.byref(view))
        mp.mr._call("tr_reduce_need", prog._rplan._open(), 0, C.byref(base))
        outs, fv = (mp._FOut * max(prog.info["outputs"], 1))(), mp._FinalView()
        mp._call("tr_prog_final_view_get", prog._handle, None, outs, len(outs), C.byref(fv))
        for limit, code in ((run_bytes, 0), (run_bytes - 1, -5)):
            stage = mp.mr._call("tr_reduce_stage_create", prog._rplan._open(), 1, C.byref(Budget().raw),
                                C.byref(mp._Refusal()), C.byref(C.c_int32()))
            arena, refusal, values = mp.mr._Arena(), mp._Refusal(), (mp._Value * max(nslots, 1))()
            mp.mr._call("tr_reduce_stage_arena", stage, C.byref(arena))
            rc = gpu.lib().tr_mgpu_program_run(C.byref(view), C.byref(base), C.byref(fv), C.byref(Budget(bytes=limit).raw),
                                               C.byref(arena), C.cast(values, C.c_void_p), C.byref(mp.mr._Stats()),
                                               C.byref(refusal))
            mp.mr._call("tr_reduce_stage_free", stage)
            self.assertEqual(rc, code, limit)
            if code:
                self.assertEqual((refusal.need, refusal.limit), (run_bytes, run_bytes - 1))
        # compile: the finalization tables (output = queries + slots, bytes = the program and the slots), the
        # canonical circuit, the request's size; each at its need and one less
        nq = len(prog.names)
        conds, nodes = len(CONDS), len(mp.mr._compile_nodes([q["where"] for q in fam.queries])[0])
        for resource, value in (("output", nq + nslots), ("bytes", prog.info["bytes"] + 72 * nslots),
                                ("circuit", max(conds + nodes, prog.info["conds"] + prog.info["nodes"]))):
            fam.compile(budget=Budget(**{resource: value}))
            with self.assertRaises(AlgebraBudgetError) as cm:
                fam.compile(budget=Budget(**{resource: value - 1}))
            self.assertEqual((cm.exception.resource, cm.exception.need), (resource, value))
        # the proof search's iterations: a pair needing N assignments merges at N and stays apart at N - 1
        pf = mp.Family(fx.scope, conditions=CONDS)
        v = mb.variables(len(CONDS))
        pf.query("a", [mp.count()], where=(v[0] & v[1]) | (v[0] & v[13] & ~v[1]))
        pf.query("b", [mp.count()], where=v[0] & (v[1] | (v[13] & ~v[1])))
        full = pf.compile(prove=True)
        n_it = full.info["search_iterations"]
        self.assertEqual((full.info["merged_search"], n_it), (1, 3 ** 4))
        self.assertEqual(pf.compile(prove=True, budget=Budget(iterations=n_it)).info["merged_search"], 1)
        short = pf.compile(prove=True, budget=Budget(iterations=n_it - 1)).info
        self.assertEqual((short["merged_search"], short["search_budget"], short["search_iterations"]), (0, 1, 0))
        # refusals
        bad = mp.Family(fx.scope)
        bad.query("s", [mp.count()], scalar=True)                             # a scalar query over k's groups
        with self.assertRaises(ValueError) as cm:
            bad.compile()
        self.assertEqual(cm.exception.failed, "s")
        with self.assertRaises(KeyError):
            prog.compare("Q1", "nope")
        wide = mp.Family(mbk.Scope.from_arrays({"u": [1 << 62, 3], "v": [1 << 62, 1]}))
        wide.query("w", [mp.sum("u^3")])
        with self.assertRaises(OverflowError) as cm:                         # FND-006's proof refuses it
            wide.compile()
        self.assertEqual(cm.exception.failed, "w")
        # the coefficients of equal monomials add exactly: the total must fit int64, whatever the order of the terms
        big = (1 << 63) - 1
        fam2 = mp.Family(fx.scope)
        fam2.query("big", [mp.sum([(big, "x"), (big, "x"), (-big, "x")], name="s")])
        fam2.query("sx", [mp.sum("x", name="s")])
        res = fam2.compile().run(backend="cuda")
        self.assertEqual({r["key"]: r["s"] for r in res["big"].rows()},
                         {r["key"]: None if r["s"] is None else big * r["s"] for r in res["sx"].rows()})
        over = mp.Family(fx.scope)
        over.query("ok", [mp.count()])
        over.query("over", [mp.sum([(big, "x"), (1, "x")])])
        with self.assertRaises(OverflowError) as cm:
            over.compile()
        self.assertEqual(cm.exception.failed, "over")

    def test_meta002_families_of_related_queries_scale_with_shared_scans(self):
        fx = Fixture(seed=21, n=500)
        for size in (1, 8, 32):
            fam = mp.Family(fx.scope, conditions=[mb.condition("w", "<", t) for t in range(0, 1000, 100)])
            v = mb.variables(10)
            for i in range(size):
                fam.query(f"q{i}", [mp.count(), mp.sum("x" if i % 2 else "y"), mp.avg({"x": 1, "w": i})],
                          where=v[i % 10] & ~v[(i + 3) % 10] if i % 3 else v[i % 10])
            prog = fam.compile()
            res = prog.run(plan="fused", backend="cuda", finalize="gpu")
            self.assertEqual(res.stats["scans"], 1)
            self.assertLessEqual(prog.info["outputs"], 2 * size)
            if size > 1:                                     # the planner prefers the fused pass for a family
                self.assertEqual(prog.run().how["plan"], "fused")
            ref = prog.run(plan="repeated")
            for n in prog.names:
                self.assertEqual(res[n].rows(), ref[n].rows())


if __name__ == "__main__":
    unittest.main()
