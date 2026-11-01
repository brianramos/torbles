# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""FND-006: fused multi-output basis reduction (torbles/math_reduce.c, math_reduce_gpu.cu, math_reduce.py).

Oracles, independent of the code under test: (1) the independent native reductions of each output's derived scope:
FND-004 banks and DIST-001 histograms built over a selection computed here, in Python, with SQL three-valued logic on
the source frame; (2) DuckDB executing each output as its own query (WHERE, GROUP BY, COUNT(*), COUNT(term),
SUM(term) in HUGEINT, FILTER for the listwise mode, LEFT JOIN for a dimension attribute); (3) Python integers for
128-bit sums of an array scope. Every variant (fused / repeated x CPU / CUDA: the CUDA library is emulated here, with
TORBLES_EMU_SCRAMBLE=1 for a scrambled thread order) is compared with the oracles and with the others. Seeds fixed.
"""
import ctypes as C
import os
import unittest
from decimal import Decimal
from fractions import Fraction

import numpy as np
import pandas as pd

import torbles
from torbles import math_bank as mbk, math_bool as mb, math_dist as md, math_reduce as mr
from torbles.math_budgets import Budget
from torbles.math_domains import DomainError
from torbles.math_errors import StaleStateError
from torbles.mathematics import AlgebraBudgetError

DDL = """CREATE TABLE dim (id BIGINT PRIMARY KEY, region VARCHAR);
CREATE TABLE f (k BIGINT, x BIGINT, d DECIMAL(10,2), y BIGINT, c BIGINT REFERENCES dim(id), w BIGINT);"""
COLS = ["x", "d", "y", "w"]
SCALES = {"x": 0, "d": 2, "y": 0, "w": 0}
VARIANTS = (("fused", "cpu"), ("fused", "cuda"), ("repeated", "cpu"), ("repeated", "cuda"))


def obj(values):
    return pd.Series(values, dtype=object)


def dataset(seed, n, dup=True):
    """signed and decimal values with NULLs, an all-NULL group (k = 7), an absent key value (k = 6), foreign keys
    (some unmatched: 99) to a dimension with a NULL attribute, and (dup) a third of the rows duplicated exactly"""
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


# ---------------------------------------------------------------------- the oracle's three-valued predicates
def kand(a, b):
    return False if a is False or b is False else (None if a is None or b is None else True)


def kor(a, b):
    return True if a is True or b is True else (None if a is None or b is None else False)


def knot(a):
    return None if a is None else (not a)


def frac(v):
    return None if v is None else Fraction(Decimal(str(v)))


class Oracle:
    """the conditions of the request evaluated row by row in Python (NULL -> UNKNOWN), Kleene connectives"""

    def __init__(self, fact):
        self.f = fact
        x, d, y = fact["x"].tolist(), fact["d"].tolist(), fact["y"].tolist()
        self.cond = [
            [None if v is None else v > 0 for v in x],                                  # 0: x > 0
            [v is None for v in d],                                                     # 1: d is null
            [None if v is None else v < 0 for v in y],                                  # 2: y < 0
            [None if v is None else -100 <= v <= 100 for v in x],                       # 3: x between -100 and 100
            [None if a is None or b is None else a < b for a, b in zip(x, y)],          # 4: x < y
            [None if v is None else frac(v) >= Fraction(3, 2) for v in d],              # 5: d >= 1.5
        ]

    def mask(self, fn):
        """the rows where the predicate fn(conds) is TRUE (SQL WHERE)"""
        n = len(self.f)
        return np.array([fn([c[i] for c in self.cond]) is True for i in range(n)], dtype=bool)


CONDITIONS = [mb.condition("x", ">", 0), mb.condition("d", "is null"), mb.condition("y", "<", 0),
              mb.condition("x", "between", -100, 100), mb.condition("x", "<", mb.column("y")),
              mb.condition("d", ">=", "1.5")]


def logical(name):
    return f'CAST(f."{name}" * {10 ** SCALES[name]} AS HUGEINT)' if SCALES[name] else f'CAST(f."{name}" AS HUGEINT)'


def term_sql(names, exps):
    factors = [logical(n) for n, e in zip(names, exps) for _ in range(e)]
    return "(" + " * ".join(factors) + ")" if factors else "1"


def duck_bank(db, terms, where_rows, key_sql, join="", mode="sql"):
    """{key: (rows, [count per term], [sum per term])} by DuckDB for the rows of a boolean mask"""
    con = db.con
    con.register("sel_rows", pd.DataFrame({"rid": np.flatnonzero(where_rows).astype(np.int64)}))
    used = sorted({n for t in terms for n, e in zip(COLS, t) if e})
    joint = " AND ".join(f'f."{n}" IS NOT NULL' for n in used) or "TRUE"
    parts = []
    for t in terms:
        expr = term_sql(COLS, t)
        if mode == "sql":
            parts += [f"COUNT({expr})", f"SUM({expr})"]
        else:
            parts += [f"COUNT(*) FILTER (WHERE {joint})", f"SUM({expr}) FILTER (WHERE {joint})"]
    keys = [f"{key_sql} AS g0"] if key_sql else []
    sql = (f"SELECT {', '.join(keys + ['COUNT(*)'] + parts)} FROM f {join} WHERE f.rowid IN (SELECT rid FROM sel_rows)"
           + (" GROUP BY g0" if key_sql else ""))
    out = {}
    for r in con.execute(sql).fetchall():
        nk = 1 if key_sql else 0
        if not key_sql and r[0] == 0:
            continue
        vals = r[nk + 1:]
        key = (r[0],) if key_sql else ()
        out[key] = (int(r[nk]), [int(v) for v in vals[0::2]], [0 if v is None else int(v) for v in vals[1::2]])
    return out


def bank_table(out):
    return {row["key"]: (row["rows"], row["count"], row["sum"]) for row in out.table()}


class Fixture:
    """the table, the scopes and a request with banks, counts and histograms of different bases, modes, predicates and
    groupings (the base grouping k, another scope's c, a dimension attribute, a rollup to one group, an explicit map)"""

    def __init__(self, seed=6, n=1200, dup=True, rows=None):
        self.db, self.fact = dataset(seed, n, dup)
        t, dim = self.db["f"], self.db["dim"]
        self.t, self.rows = t, rows
        self.scope = mbk.Scope.from_table(t, COLS, group_by=["k"], rows=rows)
        self.by_c = mbk.Scope.from_table(t, [], group_by=[("c", dim, "id")], rows=rows)   # the referenced id
        self.by_region = mbk.Scope.from_table(t, [], group_by=[("c", dim, "region")], rows=rows)
        self.coarse = mbk.Scope.from_table(t, COLS, rows=rows)
        self.parity = np.array([g % 2 for g in range(self.scope.groups)], dtype=np.uint32)
        self.oracle = Oracle(self.fact)

    def request(self):
        req = mr.Request(self.scope, conditions=CONDITIONS,
                         groupings={"c": self.by_c, "region": self.by_region, "all": mr.rollup(self.coarse),
                                    "parity": mr.rollup(map=self.parity, groups=2)})
        big, dn, yneg, mid, xy, dd = mb.variables(6)
        req.bank("A", ["1", "x", "d", "x*d", "x^2", "d^2", "x*y"], where=big)
        req.bank("B", ["1", "x", "y", "x*y"], where=dn | yneg, mode="joint", group="c")
        req.count("C", where=~big, group="region")
        req.histogram("D", ["x"], weight="w", where=big)
        req.histogram("E", ["y"], repr="dense", where=mid, group="c")
        req.bank("F", ["x", "x^2", "y"], where=mid & ~dn, group="all")
        req.count("G", group="parity")
        req.bank("H", ["1", "x"], where=mb.FALSE)
        req.histogram("I", ["x", "y"], where=xy & dd, group="c")
        req.bank("J", ["w", "w*x", "w^2"], where=xy)
        return req

    PREDICATES = {"A": lambda c: c[0], "B": lambda c: kor(c[1], c[2]), "C": lambda c: knot(c[0]),
                  "D": lambda c: c[0], "E": lambda c: c[3], "F": lambda c: kand(c[3], knot(c[1])),
                  "G": lambda c: True, "H": lambda c: False, "I": lambda c: kand(c[4], c[5]), "J": lambda c: c[4]}
    GROUPS = {"A": ["k"], "B": "id", "C": "region", "D": ["k"], "E": "id", "F": [], "G": ["k"], "H": ["k"],
              "I": "id", "J": ["k"]}

    def derived(self, name, cols):
        """the independently derived scope of an output: the rows where its predicate is TRUE (Python oracle)"""
        mask = self.oracle.mask(self.PREDICATES[name])
        if self.rows is not None:
            mask &= np.asarray(self.rows, dtype=bool)
        keys = self.GROUPS[name]
        if keys in ("region", "id"):
            keys = [("c", self.db["dim"], keys)]
        return mbk.Scope.from_table(self.t, cols, group_by=keys, rows=mask), mask


SPECS = {"A": ("bank", ["1", "x", "d", "x*d", "x^2", "d^2", "x*y"], "sql"), "B": ("bank", ["1", "x", "y", "x*y"], "joint"),
         "C": ("bank", [], "sql"), "F": ("bank", ["x", "x^2", "y"], "sql"), "G": ("bank", [], "sql"),
         "H": ("bank", ["1", "x"], "sql"), "J": ("bank", ["w", "w*x", "w^2"], "sql"),
         "D": ("hist", ["x"], "w", "sparse"), "E": ("hist", ["y"], None, "dense"), "I": ("hist", ["x", "y"], None, "sparse")}


class FusedReduce(unittest.TestCase):
    maxDiff = None

    def check_against_independent(self, fx, res):
        """every output equals the independent FND-004 / DIST-001 reduction of its derived scope, group by group"""
        for name, spec in SPECS.items():
            out = res[name]
            ds, _ = fx.derived(name, COLS)
            if spec[0] == "bank":
                if name == "G":                               # an explicit map: the independent bank rolled up
                    ref = mbk.build(ds, spec[1], spec[2]).rollup(map=fx.parity, groups=2)
                else:
                    ref = mbk.build(ds, spec[1], spec[2])
                self.assertEqual(out.info["groups"], ref.groups, name)
                for g in range(ref.groups):
                    self.assertEqual(out.state(g), ref.state(g), (name, g))
            else:
                _, cols, weight, repr_ = spec
                dom = out.info.get("domain") if repr_ == "dense" else None
                ref = md.histogram(ds, cols, weight=weight, repr=repr_, domain=dom)
                self.assertEqual(out.info["groups"], ref.groups, name)
                for g in range(ref.groups):
                    self.assertEqual(out.group(g), ref.group(g), (name, g))

    def test_fnd006_every_output_matches_independent_reductions_on_every_variant(self):
        fx = Fixture()
        plan = fx.request().compile()
        results = {v: plan.execute(*v) for v in VARIANTS}
        self.check_against_independent(fx, results[("fused", "cpu")])
        for v, res in results.items():
            self.assertTrue(res.equal(results[("fused", "cpu")]), v)
            for name in SPECS:                                  # value by value, not only the native comparison
                self.assertEqual(res[name].states(), results[("fused", "cpu")][name].states(), (v, name))
        self.assertEqual(results[("fused", "cuda")].backend, "cuda-emulated")
        self.assertEqual(results[("repeated", "cuda")].backend, "cuda-emulated")
        self.assertEqual(results[("fused", "cpu")].backend, "cpu")

    def test_fnd006_outputs_match_duckdb_executing_each_output_separately(self):
        fx = Fixture(seed=11)
        res = fx.request().compile().execute("fused", "cpu")
        o = fx.oracle
        terms = lambda basis: [tuple(r) for r in mbk._basis(COLS, basis)]
        cases = [("A", terms(SPECS["A"][1]), 'f."k"', "", "sql"),
                 ("B", terms(SPECS["B"][1]), "dim.id", "LEFT JOIN dim ON f.c = dim.id", "joint"),
                 ("C", [], "dim.region", "LEFT JOIN dim ON f.c = dim.id", "sql"),
                 ("F", terms(SPECS["F"][1]), None, "", "sql"), ("J", terms(SPECS["J"][1]), 'f."k"', "", "sql")]
        for name, tm, key, join, mode in cases:
            want = duck_bank(fx.db, tm, o.mask(fx.PREDICATES[name]), key, join, mode)
            got = bank_table(res[name])
            if key is None:
                got = {(): v for v in got.values()}
            self.assertEqual(got, want, name)
        # a histogram as GROUP BY x (count, COUNT(w), SUM(w)) in DuckDB
        con = fx.db.con
        con.register("sel_rows", pd.DataFrame({"rid": np.flatnonzero(o.mask(fx.PREDICATES["D"])).astype(np.int64)}))
        rows = con.execute('SELECT f."k", f."x", COUNT(*), COUNT(f."w"), SUM(CAST(f."w" AS HUGEINT)) FROM f WHERE '
                           'f.rowid IN (SELECT rid FROM sel_rows) GROUP BY f."k", f."x"').fetchall()
        want = {}
        for k, x, cnt, wc, ws in rows:
            want.setdefault(k, []).append(((x,), int(cnt), int(wc), 0 if ws is None else int(ws)))
        got = {row["key"][0]: row["cells"] for row in res["D"].table()}
        self.assertEqual({k: sorted(v, key=lambda t: (t[0][0] is None, t[0][0] or 0)) for k, v in want.items()}, got)

    def test_fnd006_duplicated_rows_count_twice_exactly(self):
        fx1, fx2 = Fixture(seed=3, n=600, dup=False), None
        db, fact = dataset(3, 600, dup=False)
        both = pd.concat([fact, fact], ignore_index=True)
        db2 = torbles.from_ddl(DDL, {"f": both, "dim": pd.DataFrame({"id": [10, 20, 30, 40],
                                                                     "region": obj(["EU", "US", None, "EU"])})})
        s1 = mbk.Scope.from_table(fx1.t, COLS, group_by=["k"])
        s2 = mbk.Scope.from_table(db2["f"], COLS, group_by=["k"])
        outs = []
        for s in (s1, s2):
            req = mr.Request(s, conditions=CONDITIONS[:2])
            big, dn = mb.variables(2)
            req.bank("A", ["1", "x", "x*d", "d^2"], where=big | dn)
            req.histogram("D", ["x"], weight="w", where=big)
            outs.append({v: req.compile().execute(*v) for v in VARIANTS})
        for v in VARIANTS:
            one, two = outs[0][v], outs[1][v]
            for g in range(s1.groups):
                a, b = one["A"].state(g), two["A"].state(g)
                self.assertEqual(b["rows"], 2 * a["rows"])
                self.assertEqual(b["count"], [2 * c for c in a["count"]])
                self.assertEqual(b["sum"], [2 * c for c in a["sum"]])
                self.assertEqual(two["D"].cells(g), [(key, 2 * c, 2 * wc, 2 * ws) for key, c, wc, ws in one["D"].cells(g)])

    def test_fnd006_empty_selection_never_true_filters_and_selective_groups(self):
        fx = Fixture(seed=5, n=400, rows=np.zeros(400 + 133, dtype=bool))
        plan = fx.request().compile()
        for v in VARIANTS:
            res = plan.execute(*v)
            for name in SPECS:
                info = res[name].info
                self.assertEqual((info["present"], info["rows"]), (0, 0), (v, name))
                for s in res[name].states():
                    self.assertFalse(s["present"])
                    if res[name].kind == "bank":
                        self.assertEqual(set(s["count"]) | set(s["sum"]) | {0}, {0})
                    else:
                        self.assertEqual(s["cells"], [])
        # a full selection: the never-true filter is empty, the absent key 6 is absent, the all-NULL group k = 7 is
        # present in an unfiltered output with zero valid counts
        fx = Fixture(seed=5, n=400)
        res = fx.request().compile().execute("fused", "cpu")
        self.assertEqual(res["H"].info["rows"], 0)
        keys = {row["key"]: row for row in mr.Request(fx.scope).count("n").compile().execute().__getitem__("n").table()}
        self.assertNotIn((6,), keys)
        req = mr.Request(fx.scope)
        req.bank("all", ["1", "x", "d"])
        out = req.compile().execute("fused", "cuda")["all"]
        g7 = next(r for r in out.table() if r["key"] == (7,))
        self.assertGreater(g7["rows"], 0)
        self.assertEqual(g7["count"][1:], [0, 0])
        self.assertEqual(g7["sum"][1:], [0, 0])
        self.check_against_independent(fx, res)

    def test_fnd006_shared_masks_columns_and_products(self):
        fx = Fixture(seed=8, n=500)
        req = mr.Request(fx.scope, conditions=CONDITIONS + [mb.condition("x", ">", 0)])    # a repeated condition
        a = mb.var(0) & mb.var(1)
        b = mb.var(0) & mb.var(1)                    # an equal but distinct expression: merged natively
        req.bank("P", ["x", "x^2", "x*d"], where=a)
        req.bank("Q", ["x^2", "x*d", "d"], where=b)
        req.bank("R", ["x", "d"], where=mb.var(6))    # the repeated condition: one mask
        plan = req.compile()
        info = plan.info
        self.assertEqual(info["merged_conds"], 1)
        self.assertGreaterEqual(info["merged_nodes"], 2)
        self.assertEqual(info["nproducts"], 4)          # x, x^2, x*d, d: shared by 8 terms
        self.assertEqual(info["term_rows"], 8)
        fused, rep = plan.execute("fused", "cpu"), plan.execute("repeated", "cpu")
        self.assertTrue(fused.equal(rep))
        sf, sr = fused.stats, rep.stats
        self.assertEqual((sf["scans"], sf["native_calls"]), (1, 1))
        self.assertEqual(sr["scans"], 3 * 3)           # masks, derived scope, reduction per output
        self.assertLessEqual(sf["products"], sf["rows_visited"] * info["nproducts"])
        self.assertLess(sf["products"], sf["accumulations"])   # a product feeds several terms
        dev = plan.execute("fused", "cuda").stats
        self.assertEqual((dev["kernels"], dev["scans"]), (1, 1))
        self.assertGreater(dev["transfer_bytes"], 0)

    def test_fnd006_wide_values_against_python_integers(self):
        rng = np.random.default_rng(17)
        n = 24
        u = [int(v) for v in rng.integers(-(1 << 60), 1 << 60, n)]
        v = [int(v) for v in rng.integers(-(1 << 60), 1 << 60, n)]
        valid_v = rng.random(n) < 0.8
        g = rng.integers(0, 3, n)
        scope = mbk.Scope.from_arrays({"u": u, "v": v}, valid={"v": valid_v}, groups=g, ngroups=3)
        req = mr.Request(scope, conditions=[mb.condition("u", ">", 0)])
        req.bank("S", ["u", "u*v", "u^2"], where=mb.var(0))
        req.bank("T", ["1", "v", "u*v"], mode="joint")
        for variant in VARIANTS:
            res = req.compile().execute(*variant)
            for grp in range(3):
                rows = [i for i in range(n) if g[i] == grp]
                pos = [i for i in rows if u[i] > 0]
                s = res["S"].state(grp)
                self.assertEqual(s["rows"], len(pos))
                self.assertEqual(s["sum"], [sum(u[i] for i in pos), sum(u[i] * v[i] for i in pos if valid_v[i]),
                                            sum(u[i] ** 2 for i in pos)])
                self.assertEqual(s["count"], [len(pos), sum(1 for i in pos if valid_v[i]), len(pos)])
                t = res["T"].state(grp)
                both = [i for i in rows if valid_v[i]]
                self.assertEqual(t["count"], [len(both)] * 3)
                self.assertEqual(t["sum"], [len(both), sum(v[i] for i in both), sum(u[i] * v[i] for i in both)])
                self.assertGreater(max(abs(x) for x in s["sum"] + [1]), 1 << 63 if pos else 0)

    # ------------------------------------------------------------------ devices, overflow, partial failures
    def test_fnd006_devices_are_recorded(self):
        fx = Fixture(seed=2, n=300)
        plan = fx.request().compile()
        for which, backend in VARIANTS:
            res = plan.execute(which, backend)
            st = res.stats
            self.assertEqual((st["plan"], st["device"]), (which, res.backend))
            if backend == "cuda":
                from torbles import math_gpu
                self.assertEqual(res.backend, "cuda-emulated" if math_gpu.device()["emulated"] else "cuda")
                self.assertEqual(res.device_name, math_gpu.device()["name"])
                self.assertGreater(st["kernels"], 0)
            else:
                self.assertEqual((res.backend, st["kernels"]), ("cpu", 0))
        res = plan.run(devices=("cpu", "cuda"), force="fused@cuda")
        self.assertEqual((res.plan["choice"]["plan"], res.plan["choice"]["device"]), ("fused", "cuda"))

    def test_fnd006_overflow_refuses_the_whole_request_atomically(self):
        big = [(1 << 45) - 3, -(1 << 45), 12, 7]
        scope = mbk.Scope.from_arrays({"u": big, "v": [1, 2, 3, 4]})
        req = mr.Request(scope)
        req.bank("fine", ["u", "v", "u*v"])
        req.bank("wide", ["u^3"])                         # 2^135 per row: unprovable
        req.count("rows")
        with self.assertRaises(OverflowError) as cm:
            req.compile()
        self.assertEqual(cm.exception.failed, "wide")
        # a weighted histogram whose weight sum cannot be proven
        scope = mbk.Scope.from_arrays({"x": [1, 2, 3], "w": [(1 << 62), (1 << 62), -(1 << 62)]})
        req = mr.Request(scope)
        req.bank("ok", ["x"])
        req.histogram("h", ["x"], weight="w")
        plan = req.compile()                               # 3 x 2^62 fits 127 bits
        self.assertTrue(plan.execute("fused", "cuda").equal(plan.execute("repeated", "cpu")))

    def test_fnd006_budget_at_the_limit_and_one_past(self):
        fx = Fixture(seed=4, n=300)
        req = mr.Request(fx.scope, conditions=CONDITIONS, groupings={"id": fx.by_c})
        req.bank("A", ["1", "x", "d", "x*d"], where=mb.var(0))
        req.bank("B", ["x", "y"], where=mb.var(2) | mb.var(4), mode="joint", group="id")
        req.histogram("E", ["y"], weight="w", repr="dense", where=mb.var(3))
        req.count("C")
        plan = req.compile()                          # no sparse output: every need is exact, not data-dependent
        oracle = plan.execute("fused", "cpu")
        for which, backend in VARIANTS:               # every resource a variant needs: at its need, and one less
            need = plan.need(which, backend)
            self.assertEqual(set(need), {"variables", "degree", "support", "circuit", "output", "states", "bytes"})
            for resource, value in need.items():
                ok = plan.execute(which, backend, budget=Budget(**{resource: value}))
                self.assertTrue(ok.equal(oracle), (which, backend, resource))
                with self.assertRaises(AlgebraBudgetError) as cm:
                    plan.execute(which, backend, budget=Budget(**{resource: value - 1}))
                self.assertEqual((cm.exception.resource, cm.exception.need, cm.exception.limit),
                                 (resource, value, value - 1), (which, backend))
        # the CUDA entry point checks its own device copies and scratch (part of the fused CUDA need)
        kernel = plan.need("fused", "kernel")["bytes"]
        total = plan.need("fused", "cuda")["bytes"]
        self.assertGreater(total, kernel)
        view, base, arena = mr._View(), mr._Need(), mr._Arena()
        mr._call("tr_reduce_get_view", plan._handle, C.byref(view))
        mr._call("tr_reduce_need", plan._handle, 0, C.byref(base))
        stage = mr._call("tr_reduce_stage_create", plan._handle, 1, C.byref(Budget().raw), C.byref(mr._Refusal()),
                         C.byref(C.c_int32()))
        mr._call("tr_reduce_stage_arena", stage, C.byref(arena))
        gpu = mr._gpu()
        for limit, code in ((kernel, 0), (kernel - 1, -5)):
            refusal = mr._Refusal()
            rc = gpu.lib().tr_mgpu_reduce_fused(C.byref(view), C.byref(base), C.byref(Budget(bytes=limit).raw),
                                                C.byref(arena), C.byref(mr._Stats()), C.byref(refusal))
            self.assertEqual(rc, code)
            if code:
                self.assertEqual((refusal.need, refusal.limit), (kernel, kernel - 1))
        mr._call("tr_reduce_stage_free", stage)
        # the planner: a candidate at its need is admissible, one byte less rejected
        for which, backend in VARIANTS:
            nb = plan.need(which, backend)["bytes"]
            cand = {(c["plan"], c["device"]): c for c in plan.choose(devices=("cpu", "cuda"), budget=Budget(bytes=nb))["candidates"]}
            self.assertNotEqual(cand[(which, backend)]["status"], "rejected")
            cand = {(c["plan"], c["device"]): c for c in plan.choose(devices=("cpu", "cuda"), budget=Budget(bytes=nb - 1))["candidates"]}
            self.assertEqual((cand[(which, backend)]["status"], cand[(which, backend)]["reason"],
                              cand[(which, backend)]["budget"]), ("rejected", "over budget",
                                                                  dict(resource="bytes", need=nb, limit=nb - 1)))
        # the request-sized resources, checked by compile before anything is allocated, and the plan's own bytes
        rq, _ = req._native()
        for resource, value in (("output", 4), ("circuit", rq.nconds + rq.nnodes), ("bytes", plan.info["bytes"])):
            req.compile(budget=Budget(**{resource: value}))
            with self.assertRaises(AlgebraBudgetError) as cm:
                req.compile(budget=Budget(**{resource: value - 1}))
            self.assertEqual((cm.exception.resource, cm.exception.need), (resource, value))

    def test_fnd006_sparse_final_state_refused_after_counting_atomically(self):
        rng = np.random.default_rng(9)
        n = 3000
        scope = mbk.Scope.from_arrays({"x": rng.integers(0, 2000, n), "y": rng.integers(0, 3, n)})
        req = mr.Request(scope)
        req.bank("m", ["x", "y", "x*y"])
        req.histogram("h", ["x"], repr="sparse")
        plan = req.compile()
        need = plan.need("fused", "cpu")["bytes"]
        res = plan.execute("fused", "cpu")                        # the final state is known only after counting
        final = res.stats["state_bytes"]
        with self.assertRaises(AlgebraBudgetError) as cm:
            plan.execute("fused", "cpu", budget=Budget(bytes=need))   # the pass fits; its sparse final state does not
        self.assertEqual(cm.exception.resource, "bytes")
        self.assertGreater(cm.exception.need, need)
        self.assertGreater(final, 0)
        self.assertTrue(res.equal(plan.execute("repeated", "cpu")))

    def test_fnd006_partial_failures_of_the_repeated_plan_and_of_adoption(self):
        # outputs put one by one: the copies of the sparse states (known only once counted) outgrow the budget after
        # several outputs were computed; the run is refused with that output and nothing of it is exposed
        rng = np.random.default_rng(23)
        n = 2000
        scope = mbk.Scope.from_arrays({"x": rng.permutation(n), "y": rng.integers(0, 5, n)})
        req = mr.Request(scope, conditions=[mb.condition("y", ">=", 0)])
        req.bank("first", ["x", "y", "x*y"])
        for i in range(16):                                   # every row its own cell: n occupied cells each
            req.histogram(f"h{i}", ["x"], where=mb.var(0) if i % 2 else None)
        plan = req.compile()
        reference = plan.execute("fused", "cpu")
        for backend in ("cpu", "cuda"):
            need = plan.need("repeated", backend)["bytes"]
            with self.assertRaises(AlgebraBudgetError) as cm:
                plan.execute("repeated", backend, budget=Budget(bytes=need))
            e = cm.exception
            self.assertEqual((e.resource, e.limit), ("bytes", need))
            self.assertGreater(e.need, need)
            self.assertIn(e.failed, [f"h{i}" for i in range(1, 16)])          # earlier outputs were computed
            self.assertTrue(plan.execute("repeated", backend).equal(reference))   # the plan is intact
        # a repeated stage finished before every output was put: nothing is exposed
        stage = mr._call("tr_reduce_stage_create", plan._handle, 2, C.byref(Budget().raw), C.byref(mr._Refusal()),
                         C.byref(C.c_int32()))
        self.assertTrue(stage)
        err, failed = C.c_int32(), C.c_int32(-1)
        h = mr._call("tr_reduce_stage_finish", stage, 1, None, C.byref(Budget().raw), C.byref(mr._Refusal()),
                     C.byref(failed), C.byref(err))
        self.assertFalse(h)
        self.assertEqual((err.value, failed.value), (-1, 0))
        # device states that are not consistent (a count above its group's rows) are refused at adoption
        view, base = mr._View(), mr._Need()
        mr._call("tr_reduce_get_view", plan._handle, C.byref(view))
        mr._call("tr_reduce_need", plan._handle, 0, C.byref(base))
        stage = mr._call("tr_reduce_stage_create", plan._handle, 1, C.byref(Budget().raw), C.byref(mr._Refusal()),
                         C.byref(C.c_int32()))
        arena, stats = mr._Arena(), mr._Stats()
        mr._call("tr_reduce_stage_arena", stage, C.byref(arena))
        gpu = mr._gpu()
        gpu.run("tr_mgpu_reduce_fused", C.byref(view), C.byref(base), C.byref(Budget().raw), C.byref(arena),
                C.byref(stats), C.byref(mr._Refusal()))
        a64 = np.ctypeslib.as_array((C.c_uint64 * arena.n64).from_address(arena.a64))
        a64[len(a64) - 1] += 1 << 40                     # the last cell count of the last output: rows != its cells
        h = mr._call("tr_reduce_stage_finish", stage, 2, C.byref(stats), C.byref(Budget().raw), C.byref(mr._Refusal()),
                     C.byref(failed), C.byref(err))
        self.assertFalse(h)
        self.assertEqual((err.value, failed.value), (-1, plan.info["noutputs"] - 1))

    # ------------------------------------------------------------------ plan choice
    def test_fnd006_planner_sees_both_plans_with_costs_and_forces_either(self):
        fx = Fixture(seed=21, n=800)
        plan = fx.request().compile()
        ev = plan.choose(devices=("cpu", "cuda"))
        cands = {(c["plan"], c["device"]): c for c in ev["candidates"]}
        self.assertEqual(set(cands), {(p, d) for p in ("fused", "repeated") for d in ("cpu", "cuda")})
        table = mr.coefficients()
        for (p, d), c in cands.items():
            rates = table[mr.PLANS[p] - 1, mr.DEVICES[d] - 1]
            for i, comp in enumerate(mr.reg.COST_COMPONENTS):
                if comp != "fusion":
                    self.assertEqual(c["costs"][comp], c["amounts"][comp] * rates[i], (p, d, comp))
            self.assertAlmostEqual(c["total"], sum(c["costs"].values()))
        fused = cands[("fused", "cpu")]
        self.assertEqual(fused["batch"], len(SPECS))
        self.assertLess(fused["costs"]["fusion"], 0)
        self.assertEqual(cands[("repeated", "cpu")]["costs"]["fusion"], 0)
        admissible = [c for c in ev["candidates"] if c["status"] != "rejected"]
        self.assertEqual(ev["choice"]["total"], min(c["total"] for c in admissible))
        self.assertEqual(ev["choice"]["reason"], "cheapest")
        # forced controls run the named plan and keep the evidence of the others
        res = plan.run(force="repeated@cpu")
        self.assertEqual((res.plan_kind, res.backend), ("repeated", "cpu"))
        self.assertEqual(res.plan["choice"]["reason"], "forced")
        self.assertTrue(all(c["reason"] in ("not forced", "forced", "over budget") for c in res.plan["candidates"]))
        res2 = plan.run(force="repeated@cuda")             # a force naming the device adds its candidates
        self.assertEqual((res2.plan_kind, res2.backend), ("repeated", "cuda-emulated"))
        self.assertTrue(res.equal(res2))
        os.environ["TORBLES_MATH_REDUCE_FORCE"] = "repeated"
        try:
            self.assertEqual(plan.run().plan_kind, "repeated")
            self.assertTrue(plan.choose()["forced"])
        finally:
            del os.environ["TORBLES_MATH_REDUCE_FORCE"]
        with self.assertRaises(ValueError):
            plan.run(force="histogram")
        with self.assertRaises(AlgebraBudgetError) as cm:               # a forced plan over budget is refused
            plan.run(force="fused@cpu", budget=Budget(states=10))
        self.assertEqual(cm.exception.plan["reason"], "forced plan refused")
        # an override table making the fused CPU plan expensive: the planner then prefers the repeated one
        expensive = mr.coefficients({("fused", "cpu"): {"chain": 1e9}})
        self.assertEqual(plan.choose(coefficients=expensive)["choice"]["plan"], "repeated")

    def test_fnd006_fused_over_budget_leaves_the_repeated_plan(self):
        rng = np.random.default_rng(31)
        n = 400
        x = rng.integers(-(1 << 40), 1 << 40, n)          # a domain far too wide for the fused counting scratch
        y, g = rng.integers(0, 4, n), rng.integers(0, 2, n)
        scope = mbk.Scope.from_arrays({"x": x, "y": y}, groups=g, ngroups=2)
        req = mr.Request(scope, conditions=[mb.condition("y", ">", 0)])
        req.histogram("h", ["x"], where=mb.var(0))
        req.bank("b", ["x", "y"], where=mb.var(0))
        plan = req.compile()
        ev = plan.choose()
        cand = {c["plan"]: c for c in ev["candidates"]}
        self.assertEqual((cand["fused"]["status"], cand["fused"]["reason"]), ("rejected", "over budget"))
        self.assertEqual(ev["choice"]["plan"], "repeated")
        res = plan.run()
        self.assertEqual(res.plan_kind, "repeated")
        with self.assertRaises(AlgebraBudgetError):
            plan.execute("fused", "cpu")
        ds = mbk.Scope.from_arrays({"x": x, "y": y}, groups=g, ngroups=2, rows=y > 0)   # the oracle's derived scope
        ref_h, ref_b = md.histogram(ds, ["x"]), mbk.build(ds, ["x", "y"])
        for grp in range(2):
            self.assertEqual(res["h"].group(grp), ref_h.group(grp))
            self.assertEqual(res["b"].state(grp), ref_b.state(grp))
        cand = {(c["plan"], c["device"]): c for c in plan.choose(devices=("cpu", "cuda"))["candidates"]}
        dev = cand[("repeated", "cuda")]
        self.assertEqual((dev["status"], dev["reason"]), ("rejected", "over budget"))   # DIST-001's device scratch
        with self.assertRaises(AlgebraBudgetError) as cm:
            plan.execute("repeated", "cuda")
        self.assertEqual((cm.exception.resource, cm.exception.need), (dev["budget"]["resource"], dev["budget"]["need"]))
        with self.assertRaises(AlgebraBudgetError):          # as DIST-001's own CUDA build of the derived scope
            md.histogram(ds, ["x"], backend="cuda")

    # ------------------------------------------------------------------ refusals
    def test_fnd006_stale_misaligned_and_malformed_requests_are_refused(self):
        fx = Fixture(seed=14, n=300)
        plan = fx.request().compile()
        other = mbk.Scope.from_table(fx.t, [], group_by=["c"], rows=np.arange(fx.t.n) % 2 == 0)
        req = mr.Request(fx.scope, groupings={"half": other})
        req.count("n", group="half")
        with self.assertRaises(ValueError):                   # another selection: no exact alignment
            req.compile()
        req = mr.Request(fx.scope, groupings={"bad": mr.rollup(map=np.full(fx.scope.groups, 5, np.uint32), groups=2)})
        req.count("n", group="bad")
        with self.assertRaises(ValueError):                   # a map entry outside the coarse groups
            req.compile()
        req = mr.Request(fx.scope)
        req.histogram("h", ["y"], repr="dense", domain=[(-5, 5, True)])
        with self.assertRaises(DomainError) as cm:            # an explicit domain that does not hold the values
            req.compile()
        self.assertEqual(cm.exception.failed, "h")
        req = mr.Request(fx.scope)
        req.bank("p", ["x", "x"])
        with self.assertRaises(ValueError):                   # a duplicate term
            req.compile()
        fx.t.flip(0, 0)                                       # one stored bit: a new revision
        with self.assertRaises(StaleStateError):
            plan.run()
        with self.assertRaises(StaleStateError):
            fx.request().compile()


if __name__ == "__main__":
    unittest.main()
