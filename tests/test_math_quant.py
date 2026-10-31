# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""QUANT-001 (existential and universal group predicates) and QUANT-002 (relational division and required-category
coverage) (torbles/math_quant.c, math_quant_gpu.cu, math_quant.py).

Oracles, independent of the implementation: DuckDB SQL over the source tables of the engine tables (correlated
EXISTS / NOT EXISTS subqueries, the quantified comparisons TRUE = ANY / TRUE = ALL, bool_or / bool_and with NULLs,
COUNT(*) FILTER, COUNT(DISTINCT), the classic double NOT EXISTS division and the GROUP BY ... HAVING COUNT(DISTINCT)
division), Python sets and counters over child rows written explicitly in the tests, and tables of expected answers
written by hand. Every native result is also computed on the CUDA path (emulated here with TORBLES_EMU_SCRAMBLE=1, a
scrambled thread order) and compared for equality. Seeds are fixed.
"""
import random
import unittest

import numpy as np
import pandas as pd

import torbles
from torbles import math_bank as mbk
from torbles import math_bool as mb
from torbles import math_handles as mh
from torbles import math_quant as mq
from torbles.math_budgets import Budget
from torbles.math_domains import DomainError
from torbles.math_errors import StaleStateError, UnsupportedError
from torbles.mathematics import AlgebraBudgetError

QS = list(mq.QUANTIFIERS)
T, F, U, A = True, False, None, mq.ABSENT
DDL = """CREATE TABLE orders (id BIGINT PRIMARY KEY, cust BIGINT);
CREATE TABLE products (id BIGINT PRIMARY KEY, name VARCHAR, family VARCHAR);
CREATE TABLE lines (oid BIGINT REFERENCES orders(id), pid BIGINT REFERENCES products(id), qty BIGINT,
                    price DECIMAL(10,2), item VARCHAR);"""
SCALES = {"qty": 0, "price": 2}


def obj(values):
    return pd.Series(values, dtype=object)


def make_db(orders, lines, products=None):
    """orders: [(id, cust)], lines: [(oid, pid, qty, price, item)], products: [(id, name, family)] (cust and family
    have NULLs or many values, so the dimensions keep plain foreign keys: FND-004 reads them through lookups)"""
    products = products or [(1, "p1", None), (2, "p2", "f")]
    o = pd.DataFrame({"id": [r[0] for r in orders], "cust": obj([r[1] for r in orders])})
    p = pd.DataFrame({"id": [r[0] for r in products], "name": obj([r[1] for r in products]),
                      "family": obj([r[2] for r in products])})
    cols = list(zip(*lines)) if lines else [[]] * 5
    l = pd.DataFrame({n: obj(list(c)) for n, c in zip(("oid", "pid", "qty", "price", "item"), cols)})
    return torbles.from_ddl(DDL, {"orders": o, "products": p, "lines": l})


def random_db(rng, n_orders=40, n_lines=700, nulls=0.2):
    ids = sorted(int(v) for v in rng.choice(np.arange(100, 100 + 3 * n_orders), n_orders, replace=False))
    cust = [None if rng.random() < 0.15 else int(v) for v in rng.integers(0, 6, n_orders)]
    cust[0] = None                                                       # nullable: a plain key design
    orders = list(zip(ids, cust))
    products = [(i, f"p{i:03d}", None if i == 1 or rng.random() < 0.2 else str(rng.choice(["fruit", "veg", "nut"])))
                for i in range(1, 25)]
    with_lines = [i for i in ids if rng.random() < 0.8]                 # the others are empty orders
    lines = []
    for _ in range(n_lines):
        u = rng.random()
        oid = None if u < 0.04 else int(rng.integers(1000, 1010)) if u < 0.08 else int(rng.choice(with_lines))
        pid = None if rng.random() < 0.05 else int(rng.integers(1, 30))  # 25..29 match no product
        qty = None if rng.random() < nulls else int(rng.integers(-5, 30))
        price = None if rng.random() < nulls else f"{int(rng.integers(0, 4000)) / 100:.2f}"
        item = None if rng.random() < 0.1 else str(rng.choice(["apple", "pear", "fig", "kiwi", "plum", "lime"]))
        lines.append((oid, pid, qty, price, item))
    for i in rng.choice(len(lines), size=len(lines) // 10, replace=False):   # exact duplicate child rows
        lines.append(lines[int(i)])
    return make_db(orders, lines, products), orders, lines


def mask_rows(db, mask):
    """the SQL condition selecting the rows of lines in the boolean mask (fact rows: torbles row r = rowid r)"""
    db.con.register("sel_rows", pd.DataFrame({"rid": np.flatnonzero(mask).astype(np.int64)}))
    return "l.rowid IN (SELECT rid FROM sel_rows)"


def order_mask(db, keep_ids):
    """the boolean row mask of orders whose id is in keep_ids (orders is a dimension: rows sorted by id)"""
    t = db["orders"]
    ids = t.codecs["id"].decode(mbk._decode(t, "id"))
    return np.array([int(i) in keep_ids for i in ids], dtype=bool)


# ---------------------------------------------------------------------- SQL rendering of predicates
def cond_sql(spec):
    col, op, *v = spec
    c = f'CAST(l."{col}" AS DECIMAL(38,{SCALES[col]}))' if SCALES[col] else f'l."{col}"'
    if op in ("is null", "is not null"):
        return f'(l."{col}" {op.upper()})'
    if op == "between":
        return f"({c} BETWEEN {v[0]} AND {v[1]})"
    return f"({c} {op} {v[0]})"


def expr_sql(e, conds):
    if e.op == mb.CONST:
        return ("FALSE", "TRUE", "CAST(NULL AS BOOLEAN)")[e.a]
    if e.op == mb.VAR:
        return conds[e.a]
    if e.op == mb.NOT:
        return f"(NOT {expr_sql(e.a, conds)})"
    op = {mb.AND: "AND", mb.OR: "OR", mb.XOR: "<>"}[e.op]
    return f"({expr_sql(e.a, conds)} {op} {expr_sql(e.b, conds)})"


def conditions(specs):
    return [mb.condition(s[0], s[1], *s[2:]) for s in specs]


# ---------------------------------------------------------------------- the SQL oracles
QSQL = {"some": "TRUE = ANY (SELECT ch.v FROM ch WHERE {m})",
        "all": "TRUE = ALL (SELECT ch.v FROM ch WHERE {m})",
        "none": "NOT (TRUE = ANY (SELECT ch.v FROM ch WHERE {m}))",
        "not_all": "NOT (TRUE = ALL (SELECT ch.v FROM ch WHERE {m}))",
        "exists": "EXISTS (SELECT 1 FROM ch WHERE {m} AND ch.v)",
        "not_exists": "NOT EXISTS (SELECT 1 FROM ch WHERE {m} AND ch.v)",
        "no_counterexample": "NOT EXISTS (SELECT 1 FROM ch WHERE {m} AND NOT ch.v)",
        "all_true": "NOT EXISTS (SELECT 1 FROM ch WHERE {m} AND ch.v IS NOT TRUE)",
        "bool_or": "(SELECT bool_or(ch.v) FROM ch WHERE {m})",
        "bool_and": "(SELECT bool_and(ch.v) FROM ch WHERE {m})"}


def sql_quantifiers(con, ch_sql, par_sql=None, match="="):
    """{parent key: counts and every quantifier} by DuckDB: ch_sql yields (k, v) child rows (v the predicate), par_sql
    the parent rows (k); without par_sql the parents are the distinct child keys (GROUP BY, the NULL key a group)"""
    m = f"ch.k {match} pk.k"
    parents = f"(SELECT count(*) FROM par WHERE par.k IS NOT DISTINCT FROM pk.k)" if par_sql else "0"
    cols = [f"(SELECT count(*) FROM ch WHERE {m})", f"(SELECT count(*) FROM ch WHERE {m} AND ch.v)",
            f"(SELECT count(*) FROM ch WHERE {m} AND NOT ch.v)", f"(SELECT count(*) FROM ch WHERE {m} AND ch.v IS NULL)",
            parents] + [QSQL[q].format(m=m) for q in QS]
    src = f"(SELECT DISTINCT k FROM par)" if par_sql else "(SELECT DISTINCT k FROM ch)"
    sql = (f"WITH ch AS ({ch_sql}), par AS ({par_sql or 'SELECT NULL AS k WHERE FALSE'}) "
           f"SELECT pk.k, {', '.join(cols)} FROM {src} pk")
    out = {}
    for row in con.execute(sql).fetchall():
        rec = dict(zip(["rows", "true", "false", "unknown", "parents"], (int(x) for x in row[1:6])))
        rec.update(zip(QS, row[6:]))
        out[None if row[0] is None else int(row[0])] = rec
    return out


def native_quantifiers(q):
    """{group key: counts and every quantifier} of the present groups of a native state"""
    out = {}
    for rec in q.table(QS):
        key = rec["key"][0] if rec["key"] else rec["group"]
        out[key] = {k: v for k, v in rec.items() if k not in ("group", "key")}
    return out


def sql_coverage(con, ch_sql, required, par_sql=None, match="="):
    """{parent key: rows, nulls, distinct, matched, missing and the division answers} by DuckDB SQL: ch_sql yields
    (k, c) child rows, required the required values (None: a NULL requirement)"""
    vals = ", ".join("(NULL)" if r is None else f"({r!r})" if isinstance(r, str) else f"({int(r)})" for r in required)
    req = f"SELECT * FROM (VALUES {vals}) r(c)" if required else "SELECT CAST(NULL AS VARCHAR) AS c WHERE FALSE"
    if required and not any(isinstance(r, str) for r in required):
        req = f"SELECT CAST(c AS BIGINT) AS c FROM (VALUES {vals}) r(c)"
    m = f"ch.k {match} pk.k"
    cols = [f"(SELECT count(*) FROM ch WHERE {m})",
            f"(SELECT count(*) FROM ch WHERE {m} AND ch.c IS NULL)",
            f"(SELECT count(DISTINCT ch.c) FROM ch WHERE {m})",
            f"(SELECT count(DISTINCT ch.c) FROM ch WHERE {m} AND ch.c IN (SELECT c FROM req))",
            f"(SELECT count(*) FROM (SELECT DISTINCT c FROM req) r WHERE NOT EXISTS (SELECT 1 FROM ch WHERE {m} AND "
            f"ch.c = r.c))",
            f"NOT EXISTS (SELECT 1 FROM req r WHERE NOT EXISTS (SELECT 1 FROM ch WHERE {m} AND ch.c = r.c))",
            f"(NOT EXISTS (SELECT 1 FROM req r WHERE NOT EXISTS (SELECT 1 FROM ch WHERE {m} AND ch.c = r.c)) AND "
            f"NOT EXISTS (SELECT 1 FROM ch WHERE {m} AND ch.c IS NOT NULL AND NOT EXISTS "
            f"(SELECT 1 FROM req r WHERE r.c = ch.c)))",
            f"EXISTS (SELECT 1 FROM ch JOIN req r ON ch.c = r.c WHERE {m})"]
    src = f"(SELECT DISTINCT k FROM par)" if par_sql else "(SELECT DISTINCT k FROM ch)"
    sql = (f"WITH ch AS ({ch_sql}), req AS ({req}), par AS ({par_sql or 'SELECT NULL AS k WHERE FALSE'}) "
           f"SELECT pk.k, {', '.join(cols)} FROM {src} pk")
    out = {}
    for row in con.execute(sql).fetchall():
        rows, nulls, distinct, matched, missing, divides, exactly, overlaps = row[1:]
        out[None if row[0] is None else int(row[0])] = dict(
            rows=int(rows), nulls=int(nulls), distinct=int(distinct), matched=int(matched), missing=int(missing),
            extra=int(distinct) - int(matched), divides=bool(divides), exactly=bool(exactly),
            overlaps=bool(overlaps), disjoint=not overlaps)
    return out


def native_coverage(cov):
    out = {}
    for rec in cov.table():
        key = rec["key"][0] if rec["key"] else rec["group"]
        out[key] = {k: v for k, v in rec.items() if k not in ("group", "key", "parents")}
    return out


def strip_q(q):
    s = q.states()
    return {k: v.tolist() for k, v in s.items()}


class Quant(unittest.TestCase):
    def both(self, *args, **kw):
        """the CPU state, checked equal to the (emulated, scrambled) CUDA state"""
        cpu = mq.Quantifier.build(*args, backend="cpu", **kw)
        gpu = mq.Quantifier.build(*args, backend="cuda", **kw)
        self.assertEqual(strip_q(cpu), strip_q(gpu))
        self.assertTrue(cpu.same(gpu))                           # the same identity: the same state
        self.assertIn(gpu.backend, ("cuda", "cuda-emulated"))
        for qn in QS:
            self.assertTrue(np.array_equal(cpu.answers(qn), gpu.answers(qn)))
        return cpu

    def cover_both(self, *args, representation="auto", **kw):
        """the CPU state of every representation and the CUDA state (masks), checked equal"""
        cpu = mq.Coverage.build(*args, representation=representation, backend="cpu", **kw)
        other = "sets" if cpu.info["representation"] == "masks" else "masks"
        alt = mq.Coverage.build(*args, representation=other, backend="cpu", **kw)
        gpu = mq.Coverage.build(*args, representation="masks", backend="cuda", **kw)
        want = {k: v.tolist() for k, v in cpu.states().items()}
        for st in (alt, gpu):
            self.assertEqual({k: v.tolist() for k, v in st.states().items()}, want)
            self.assertTrue(st.same(cpu))
            G = cpu.info["groups"]
            for g in range(G):
                for which in ("present", "missing", "extra"):
                    self.assertEqual(st.codes(g, which), cpu.codes(g, which), (g, which))
        self.assertIn(gpu.backend, ("cuda", "cuda-emulated"))
        return cpu

    # ================================================================== QUANT-001
    EXPLICIT = {   # order id: (qty of its lines, the expected answer of every quantifier for qty > 5)
        1: ([], dict(some=F, all=T, none=T, not_all=F, exists=F, not_exists=T, no_counterexample=T, all_true=T,
                     bool_or=U, bool_and=U)),                                                     # empty
        2: ([None, None], dict(some=U, all=U, none=U, not_all=U, exists=F, not_exists=T, no_counterexample=T,
                               all_true=F, bool_or=U, bool_and=U)),                               # all NULL
        3: ([7, 2, None], dict(some=T, all=F, none=F, not_all=T, exists=T, not_exists=F, no_counterexample=F,
                               all_true=F, bool_or=T, bool_and=F)),                               # mixed
        4: ([8, 9], dict(some=T, all=T, none=F, not_all=F, exists=T, not_exists=F, no_counterexample=T, all_true=T,
                         bool_or=T, bool_and=T)),                                                 # all TRUE
        5: ([1, 2], dict(some=F, all=F, none=T, not_all=T, exists=F, not_exists=T, no_counterexample=F, all_true=F,
                         bool_or=F, bool_and=F)),                                                 # all FALSE
        6: ([9, None], dict(some=T, all=U, none=F, not_all=U, exists=T, not_exists=F, no_counterexample=T, all_true=F,
                            bool_or=T, bool_and=T)),                                              # TRUE and NULL
        7: ([3, None], dict(some=U, all=F, none=U, not_all=T, exists=F, not_exists=T, no_counterexample=F,
                            all_true=F, bool_or=F, bool_and=F)),                                  # FALSE and NULL
        8: ([10, 10, 10], dict(some=T, all=T, none=F, not_all=F, exists=T, not_exists=F, no_counterexample=T,
                               all_true=T, bool_or=T, bool_and=T)),                               # duplicates
    }

    def explicit_db(self, extra_lines=()):
        orders = [(i, None if i % 3 else i) for i in self.EXPLICIT]          # cust nullable: a plain key design
        lines = [(oid, 1, q, None, None) for oid, (qs, _) in self.EXPLICIT.items() for q in qs]
        lines += [(999, 1, 10, None, None), (None, 1, 1, None, None)] + list(extra_lines)   # orphans: no parent
        return make_db(orders, lines)

    def test_quant001_some_all_none_match_explicit_child_rows_empty_all_null_mixed(self):
        db = self.explicit_db()
        child = mbk.Scope.from_table(db["lines"], ["qty"], group_by=[("oid", db["orders"], "id")])
        parents = mbk.Scope.from_table(db["orders"], [], group_by=["id"])
        conds = [mb.condition("qty", ">", 5)]
        q = self.both(child, conds, mb.var(0), parents=parents)
        got = native_quantifiers(q)
        self.assertEqual(sorted(got), sorted(self.EXPLICIT))       # the orphans (999, NULL) have no parent: absent
        for oid, (qs, want) in self.EXPLICIT.items():
            t = sum(1 for v in qs if v is not None and v > 5)
            u = sum(1 for v in qs if v is None)
            self.assertEqual((got[oid]["rows"], got[oid]["true"], got[oid]["unknown"], got[oid]["parents"]),
                             (len(qs), t, u, 1), oid)
            for qn, w in want.items():
                self.assertIs(got[oid][qn], w, (oid, qn))
        self.assertEqual((q.info["empty"], q.info["orphans"]), (1, 2))
        # the same answers by DuckDB: correlated subqueries over the source tables
        sql = sql_quantifiers(db.con, 'SELECT l."oid" AS k, l."qty" > 5 AS v FROM lines l',
                              'SELECT o."id" AS k FROM orders o')
        self.assertEqual(got, sql)
        # every group of the code space that is not an order is absent: ABSENT, never FALSE or UNKNOWN
        for qn in QS:
            codes = q.answers(qn)
            self.assertEqual(int((codes == 3).sum()), q.info["absent"])
            self.assertEqual(int((codes != 3).sum()), len(self.EXPLICIT))

    def test_quant001_vacuous_truth_unknown_and_absent_conventions(self):
        db = self.explicit_db()
        lines, orders = db["lines"], db["orders"]
        child = mbk.Scope.from_table(lines, ["qty"], group_by=[("oid", orders, "id")])
        parents = mbk.Scope.from_table(orders, [], group_by=["id"])
        # constant predicates (k = 0): TRUE, FALSE and SQL's NULL literal on every child row
        for const, label in ((mb.TRUE, "TRUE"), (mb.FALSE, "FALSE"), (mb.UNKNOWN, "CAST(NULL AS BOOLEAN)")):
            q = self.both(child, [], const, parents=parents)
            sql = sql_quantifiers(db.con, f'SELECT l."oid" AS k, {label} AS v FROM lines l',
                                  'SELECT o."id" AS k FROM orders o')
            self.assertEqual(native_quantifiers(q), sql, label)
            empty = native_quantifiers(q)[1]                        # vacuous truth does not depend on the predicate
            self.assertEqual({k: empty[k] for k in QS}, self.EXPLICIT[1][1])
        q = self.both(child, [], mb.UNKNOWN, parents=parents)
        nonempty = native_quantifiers(q)[4]                         # an all-UNKNOWN group is not an empty group
        self.assertEqual({k: nonempty[k] for k in QS}, self.EXPLICIT[2][1])
        # x OR NOT x is UNKNOWN where x is NULL (SQL evaluation, not a supervaluation)
        x = mb.var(0)
        q = self.both(child, [mb.condition("qty", ">", 5)], x | ~x, parents=parents)
        self.assertEqual(native_quantifiers(q)[2]["all"], None)
        self.assertEqual(native_quantifiers(q)[3]["all"], None)
        self.assertEqual(native_quantifiers(q)[4]["all"], True)
        # GROUPED (SQL GROUP BY over the children): no empty group; the NULL key (NULL and unmatched references) is a
        # group of its own; IDS: every id is present
        conds = [mb.condition("qty", ">", 5)]
        g = self.both(child, conds, mb.var(0))
        sql = sql_quantifiers(db.con, 'SELECT o."id" AS k, l."qty" > 5 AS v FROM lines l LEFT JOIN orders o ON '
                                      'l."oid" = o."id"', match="IS NOT DISTINCT FROM")
        self.assertEqual(native_quantifiers(g), sql)
        self.assertIn(None, native_quantifiers(g))
        self.assertNotIn(1, native_quantifiers(g))                  # the empty order is absent without parents
        self.assertEqual((g.info["empty"], g.info["orphans"]), (0, 0))
        ids = self.both(child, conds, mb.var(0), parents="ids")
        self.assertEqual((ids.info["present"], ids.info["absent"]), (child.groups, 0))
        self.assertEqual(ids.answer("all", 0), True)                 # order 1 (code 0) has no rows: vacuously TRUE
        self.assertEqual(ids.answer("some", child.groups - 1), True)   # IDS: the NULL group is present (orphans)
        p = self.both(child, conds, mb.var(0), parents=parents)
        self.assertIs(p.answer("some", child.groups - 1), mq.ABSENT)   # PARENTS: the NULL group has no parent
        self.assertEqual(p.info["absent"], 1)
        with self.assertRaises(TypeError):
            bool(mq.ABSENT)

    PSPECS = [("qty", ">", 5), ("price", "<=", "12.50"), ("qty", "is null"), ("price", "between", "3.00", "30.00"),
              ("qty", "<", 0)]

    def random_exprs(self, rng, k):
        x = mb.variables(k)
        out = [x[0], x[0] & x[1], x[0] | (x[1] & ~x[2]), (x[0] ^ x[3]) | x[2], ~(x[1] | x[4]) & x[3],
               x[0] & mb.UNKNOWN, x[1] | ~x[1]]
        for _ in range(3):
            e = x[rng.randrange(k)]
            for _ in range(4):
                o = x[rng.randrange(k)]
                e = rng.choice([e & o, e | o, e ^ o, ~e & o])
            out.append(e)
        return out

    def test_quant001_random_tables_match_duckdb_quantified_subqueries(self):
        rng = np.random.default_rng(1101)
        prng = random.Random(1101)
        db, orders, lines = random_db(rng)
        t, o = db["lines"], db["orders"]
        csql = [cond_sql(s) for s in self.PSPECS]
        conds = conditions(self.PSPECS)
        for trial, (sel, keep) in enumerate(((None, None), (rng.random(t.n) < 0.6, None),
                                             (rng.random(t.n) < 0.8, {i for i, _ in orders if rng.random() < 0.7}))):
            child = mbk.Scope.from_table(t, ["qty", "price"], group_by=[("oid", o, "id")], rows=sel)
            parents = mbk.Scope.from_table(o, [], group_by=["id"], rows=None if keep is None else order_mask(db, keep))
            where = mask_rows(db, sel) if sel is not None else "TRUE"
            psel = f'o."id" IN ({", ".join(map(str, sorted(keep)))})' if keep else "TRUE"
            for expr in self.random_exprs(prng, len(conds)):
                p = mb.Program(expr, len(conds))
                v = expr_sql(expr, csql)
                q = self.both(child, conds, p, parents=parents)
                sql = sql_quantifiers(db.con, f'SELECT l."oid" AS k, {v} AS v FROM lines l WHERE {where}',
                                      f'SELECT o."id" AS k FROM orders o WHERE {psel}')
                self.assertEqual(native_quantifiers(q), sql, (trial, expr))
                g = self.both(child, conds, p)
                sql = sql_quantifiers(db.con, f'SELECT o."id" AS k, {v} AS v FROM lines l LEFT JOIN orders o ON '
                                              f'l."oid" = o."id" WHERE {where}', match="IS NOT DISTINCT FROM")
                self.assertEqual(native_quantifiers(g), sql, (trial, expr, "grouped"))

    def test_quant001_groups_through_a_dimension_lookup_with_parent_multiplicity(self):
        rng = np.random.default_rng(1102)
        db, orders, lines = random_db(rng, n_orders=60, n_lines=900)
        t, o = db["lines"], db["orders"]
        conds = conditions(self.PSPECS[:3])
        x = mb.variables(3)
        expr = x[0] | (x[1] & ~x[2])
        v = expr_sql(expr, [cond_sql(s) for s in self.PSPECS[:3]])
        sel = rng.random(t.n) < 0.75
        where = mask_rows(db, sel)
        # each line's group is the customer of its order (a dimension attribute through the foreign key); the parent
        # rows are the orders of a customer (parent multiplicity); a NULL customer is the NULL group of both scopes
        child = mbk.Scope.from_table(t, ["qty", "price"], group_by=[("oid", o, "cust")], rows=sel)
        parents = mbk.Scope.from_table(o, [], group_by=["cust"])
        q = self.both(child, conds, mb.Program(expr, 3), parents=parents)
        sql = sql_quantifiers(db.con, f'SELECT o."cust" AS k, {v} AS v FROM lines l LEFT JOIN orders o ON '
                                      f'l."oid" = o."id" WHERE {where}', 'SELECT o."cust" AS k FROM orders o',
                              match="IS NOT DISTINCT FROM")
        got = native_quantifiers(q)
        self.assertEqual(got, sql)
        self.assertGreater(max(r["parents"] for r in got.values()), 1)     # several orders per customer
        # with the customer's NULL group excluded from the parents, it is absent (SQL's correlated c.key = p.key)
        nn = mbk.Scope.from_table(o, [], group_by=["cust"], rows=self.not_null_rows(o, "cust"))
        q2 = self.both(child, conds, mb.Program(expr, 3), parents=nn)
        sql2 = sql_quantifiers(db.con, f'SELECT o."cust" AS k, {v} AS v FROM lines l JOIN orders o ON '
                                       f'l."oid" = o."id" WHERE {where}',
                               'SELECT o."cust" AS k FROM orders o WHERE o."cust" IS NOT NULL')
        self.assertEqual(native_quantifiers(q2), sql2)
        self.assertNotIn(None, native_quantifiers(q2))

    @staticmethod
    def not_null_rows(t, name):
        return mbk._decode(t, t.nullable[name]) != 0

    def test_quant001_multiplicity_counts_rows_never_erases_coverage(self):
        rng = np.random.default_rng(1103)
        db, orders, lines = random_db(rng, n_orders=30, n_lines=400)
        twice = make_db(orders, lines + lines, [(i, f"p{i:03d}", None) for i in range(1, 25)])
        conds = conditions(self.PSPECS[:3])
        x = mb.variables(3)
        expr = (x[0] & ~x[2]) | x[1]
        prog = mb.Program(expr, 3)
        states = []
        for d in (db, twice):
            child = mbk.Scope.from_table(d["lines"], ["qty", "price"], group_by=[("oid", d["orders"], "id")])
            parents = mbk.Scope.from_table(d["orders"], [], group_by=["id"])
            states.append(self.both(child, conds, prog, parents=parents))
        one, two = (s.states() for s in states)
        for key in ("rows", "true", "false", "unknown"):
            self.assertTrue(np.array_equal(two[key], 2 * one[key]), key)    # every duplicated row counts again
        self.assertTrue(np.array_equal(two["parents"], one["parents"]))
        for qn in QS:                                                       # and changes no answer
            self.assertTrue(np.array_equal(states[0].answers(qn), states[1].answers(qn)), qn)
        v = expr_sql(expr, [cond_sql(s) for s in self.PSPECS[:3]])
        sql = sql_quantifiers(twice.con, f'SELECT l."oid" AS k, {v} AS v FROM lines l', 'SELECT o."id" AS k FROM orders o')
        self.assertEqual(native_quantifiers(states[1]), sql)
        # the bag union of states (merge) adds counts exactly and never erases a covered group
        t, o = db["lines"], db["orders"]
        part = rng.random(t.n) < 0.5
        parents = mbk.Scope.from_table(o, [], group_by=["id"])
        a, b, whole = (mq.Quantifier.build(mbk.Scope.from_table(t, ["qty", "price"], group_by=[("oid", o, "id")], rows=r),
                                           conds, prog, parents=parents) for r in (part, ~part, None))
        ab = a.merge(b)
        self.assertEqual(strip_q(ab), strip_q(whole))
        self.assertFalse(ab.same(whole))                    # the union of two bags is another identity
        self.assertEqual(ab.info["parts"], 2)
        aa = a.merge(a)
        for key in ("rows", "true", "false", "unknown"):
            self.assertTrue(np.array_equal(aa.states()[key], 2 * a.states()[key]))
        for qn in QS:
            self.assertTrue(np.array_equal(aa.answers(qn), a.answers(qn)), qn)
        covered = a.answers("exists") == 1
        self.assertTrue(np.all(ab.answers("exists")[covered] == 1))   # a covered group stays covered in the union
        other = mq.Quantifier.build(mbk.Scope.from_table(t, ["qty", "price"], group_by=[("oid", o, "id")]), conds,
                                    mb.Program(x[0], 3), parents=parents)
        with self.assertRaises(DomainError):                # another predicate is another state: never merged
            a.merge(other)

    def test_quant001_cuda_scan_equals_cpu_contiguous_and_scattered_groups(self):
        rng = np.random.default_rng(1104)
        n, G = 30000, 700
        x = rng.integers(-50, 50, n)
        y = rng.integers(0, 1000, n)
        vx = rng.random(n) > 0.1
        sizes = rng.integers(0, 90, G)
        groups = np.repeat(np.arange(G, dtype=np.uint32), sizes)[:n]
        groups = np.concatenate([groups, rng.integers(0, G, n - len(groups)).astype(np.uint32)])
        groups[:n - 4000].sort()                            # contiguous runs (whole words of one group), then scattered
        conds = [mb.condition("x", ">", 0), mb.condition("y", "<", 300), mb.condition("x", "is null")]
        e = mb.variables(3)
        expr = (e[0] & e[1]) | e[2]
        for gids, rows in ((groups, None), (rng.permutation(groups), None), (groups, rng.random(n) < 0.3)):
            child = mbk.Scope.from_arrays({"x": x, "y": y}, valid={"x": vx}, groups=gids, ngroups=G, rows=rows)
            q = self.both(child, conds, mb.Program(expr, 3), parents="ids")
            s = q.states()
            sel = np.ones(n, bool) if rows is None else rows
            px = np.where(vx, x > 0, False)
            # the Kleene value per row: TRUE, FALSE or UNKNOWN (x NULL makes x > 0 UNKNOWN, x IS NULL TRUE)
            truth = np.where(~vx, 1, np.where(px & (y < 300), 1, 0))
            for g in rng.choice(G, 40, replace=False):
                m = sel & (gids == g)
                self.assertEqual((int(s["rows"][g]), int(s["true"][g]), int(s["false"][g]), int(s["unknown"][g])),
                                 (int(m.sum()), int((truth[m] == 1).sum()), int((truth[m] == 0).sum()), 0))
        # nothing selected: every group empty (IDS) or absent (GROUPED)
        empty = mbk.Scope.from_arrays({"x": x, "y": y}, groups=groups, ngroups=G, rows=np.zeros(n, bool))
        z = self.both(empty, conds, mb.Program(expr, 3), parents="ids")
        self.assertEqual((z.info["rows"], z.info["empty"]), (0, G))
        self.assertEqual(self.both(empty, conds, mb.Program(expr, 3)).info["absent"], G)

    def test_quant001_identity_and_matches(self):
        db = self.explicit_db()
        t, o = db["lines"], db["orders"]
        child = mbk.Scope.from_table(t, ["qty"], group_by=[("oid", o, "id")])
        parents = mbk.Scope.from_table(o, [], group_by=["id"])
        conds = [mb.condition("qty", ">", 5)]
        q = mq.Quantifier.build(child, conds, mb.var(0), parents=parents)
        again = mq.Quantifier.build(child, conds, mb.var(0), parents=parents)
        self.assertTrue(q.same(again) and q.identity == again.identity)
        self.assertTrue(q.matches(child, conds, mb.var(0), parents=parents))
        for variant in (dict(conditions=[mb.condition("qty", ">", 6)], predicate=mb.var(0), parents=parents),
                        dict(conditions=conds, predicate=~mb.var(0), parents=parents),
                        dict(conditions=conds, predicate=mb.var(0), parents=None),
                        dict(conditions=conds, predicate=mb.var(0), parents="ids")):
            self.assertFalse(q.matches(child, variant["conditions"], variant["predicate"], parents=variant["parents"]))
            other = mq.Quantifier.build(child, variant["conditions"], variant["predicate"], parents=variant["parents"])
            self.assertFalse(q.same(other))
            self.assertNotEqual(q.info["digest"], other.info["digest"])
        sub = mbk.Scope.from_table(t, ["qty"], group_by=[("oid", o, "id")], rows=np.arange(t.n) % 2 == 0)
        self.assertFalse(q.matches(sub, conds, mb.var(0), parents=parents))   # another selection

    def test_quant001_budgets_at_the_limit_and_one_past(self):
        rng = np.random.default_rng(1105)
        db, orders, lines = random_db(rng, n_orders=20, n_lines=300)
        t, o = db["lines"], db["orders"]
        child = mbk.Scope.from_table(t, ["qty", "price"], group_by=[("oid", o, "id")])
        parents = mbk.Scope.from_table(o, [], group_by=["id"])
        conds = conditions(self.PSPECS[:3])
        x = mb.variables(3)
        prog = mb.Program((x[0] & x[1]) | x[2], 3)
        for backend in ("cpu", "cuda"):
            need = mq.groups_need(child, conds, prog, parents=parents, backend=backend)
            self.assertEqual((need["variables"], need["circuit"], need["states"], need["output"]),
                             (3, prog.info["nodes"], child.groups, child.groups))
            for resource in ("variables", "circuit", "states", "output", "bytes"):
                mq.Quantifier.build(child, conds, prog, parents=parents, backend=backend,
                                    budget=Budget(**{resource: need[resource]}))
                with self.assertRaises(AlgebraBudgetError) as ctx:
                    mq.Quantifier.build(child, conds, prog, parents=parents, backend=backend,
                                        budget=Budget(**{resource: need[resource] - 1}))
                self.assertEqual((ctx.exception.resource, ctx.exception.need, ctx.exception.limit),
                                 (resource, need[resource], need[resource] - 1))
        q = mq.Quantifier.build(child, conds, prog, parents=parents)
        nb = q.info["bytes"] + 24 + q.info["identity_bytes"]     # the merged object: two identities and a tag
        q.merge(q, budget=Budget(bytes=nb))
        with self.assertRaises(AlgebraBudgetError):
            q.merge(q, budget=Budget(bytes=nb - 1))

    def test_quant001_stale_and_malformed_requests_are_refused(self):
        db = self.explicit_db()
        t, o = db["lines"], db["orders"]
        child = mbk.Scope.from_table(t, ["qty"], group_by=[("oid", o, "id")])
        parents = mbk.Scope.from_table(o, [], group_by=["id"])
        conds = [mb.condition("qty", ">", 5)]
        q = mq.Quantifier.build(child, conds, mb.var(0), parents=parents)
        with self.assertRaises(ValueError):
            q.answers("most")
        with self.assertRaises(DomainError):                            # the predicate reads 2 conditions, 1 given
            mq.Quantifier.build(child, conds, mb.Program(mb.var(0) & mb.var(1), 2))
        with self.assertRaises(ValueError):
            mq.Quantifier.build(child, [mb.condition("nope", ">", 1)], mb.var(0))
        small = mbk.Scope.from_arrays({"qty": np.array([1, 7])}, groups=np.array([0, 1]), ngroups=2)
        with self.assertRaises(DomainError):                            # more parent groups than child groups
            mq.Quantifier.build(small, conds, mb.var(0),
                                parents=mbk.Scope.from_arrays({}, groups=np.array([0, 4]), ngroups=5))
        with self.assertRaises(ValueError):                             # parent codes name other values
            mq.Quantifier.build(child, conds, mb.var(0), parents=mbk.Scope.from_table(o, [], group_by=["cust"]))
        with self.assertRaises(TypeError):
            mq.Quantifier.build(child, conds, "qty > 5")
        t.flip(0, 0)                                                     # a mutation: a new revision
        for backend in ("cpu", "cuda"):
            with self.assertRaises(StaleStateError):
                mq.Quantifier.build(child, conds, mb.var(0), parents=parents, backend=backend)
        with self.assertRaises(StaleStateError):
            q.answers("some")
        self.assertFalse(q.fresh())

    def test_quant001_ctypes_structures_are_registered(self):
        report = mh.abi_check()
        mine = {r["native"] for r in report["checked"] if r["structure"].startswith("torbles.math_quant.")}
        self.assertEqual(mine, {"tr_quant_counts", "tr_quant_groups_info", "tr_quant_keys", "tr_quant_dict_info",
                                "tr_quant_cover_arrays", "tr_quant_cover_info"})
        cuda = {r["native"] for r in report["checked"] if r["library"] == "cuda"
                and r["structure"].startswith("torbles.math_quant.")}
        self.assertEqual(cuda, {"tr_quant_counts", "tr_quant_cover_arrays"})

    # ================================================================== QUANT-002
    def explicit_cover(self):
        """child rows (group, category) written out: duplicates, NULL categories, an empty group, an absent id"""
        rows = [(0, "apple"), (0, "pear"), (0, "pear"), (0, None),            # duplicate pear, a NULL category
                (1, "apple"), (1, "fig"), (1, "kiwi"),                         # an extra category (kiwi)
                (2, None), (2, None),                                          # only NULL categories
                (3, "fig"), (3, "fig"), (3, "fig"),                            # one category three times
                (5, "pear"), (5, "apple"), (5, "fig"), (5, "plum")]           # group 4: empty; 6: an absent id
        values = ["apple", "fig", "kiwi", "pear", "plum", None]                # code 5: the NULL category
        g = np.array([r[0] for r in rows], dtype=np.uint32)
        c = np.array([values.index(r[1]) for r in rows], dtype=np.uint32)
        child = mbk.Scope.from_arrays({}, groups=g, ngroups=7)
        cats = mbk.Scope.from_arrays({}, groups=c, ngroups=len(values))
        return rows, values, child, cats, mq.Dictionary.keys(values, null=5)

    def test_quant002_missing_extra_duplicated_null_categories_match_sets(self):
        rows, values, child, cats, d = self.explicit_cover()
        self.assertEqual([d.find(v) for v in ("apple", "pear", "plum", "durian")], [0, 3, 4, None])   # native lookup
        self.assertEqual([d.value(c) for c in range(6)], values)
        present_groups = set(range(6))                                       # declared parents: ids 0..5
        parents = mbk.Scope.from_arrays({}, groups=np.arange(6, dtype=np.uint32), ngroups=6)
        for required in (["apple", "pear"], ["fig"], [], ["apple", "apple", "pear"], ["apple", "durian"],
                         ["fig", None], ["kiwi", "plum", "pear", "apple", "fig"]):
            cov = self.cover_both(child, cats, required, dictionary=d, parents=parents)
            R = set(required)
            for gid in range(7):
                mine = [r for r in rows if r[0] == gid]
                present = {cat for _, cat in mine if cat is not None}
                st = cov.state(gid)
                if gid not in present_groups:
                    self.assertFalse(st["present"])
                    self.assertIs(cov.answer("divides", gid), mq.ABSENT)
                    continue
                missing = R - present                     # a NULL requirement (None) is never present
                extra = present - {r for r in R if r is not None}
                self.assertEqual((st["rows"], st["nulls"], st["distinct"]),
                                 (len(mine), sum(1 for _, cat in mine if cat is None), len(present)), (gid, required))
                self.assertEqual(set(cov.categories(gid, "present")), present)
                self.assertEqual(sorted(cov.categories(gid, "extra")), sorted(extra))
                self.assertEqual(sorted(cov.categories(gid, "missing"), key=str), sorted(missing, key=str))
                self.assertEqual((st["missing"], st["extra"], st["matched"]),
                                 (len(missing), len(extra), len(present & R)))
                self.assertIs(cov.answer("divides", gid), not missing)
                self.assertIs(cov.answer("exactly", gid), not missing and not extra)
                self.assertIs(cov.answer("overlaps", gid), bool(present & R))
                self.assertIs(cov.answer("disjoint", gid), not (present & R))
        # duplicates count every row and the category once; NULL rows are rows, never categories
        cov = mq.Coverage.build(child, cats, ["fig"], dictionary=d, parents=parents)
        self.assertEqual((cov.state(3)["rows"], cov.state(3)["distinct"]), (3, 1))
        self.assertEqual((cov.state(2)["rows"], cov.state(2)["nulls"], cov.state(2)["distinct"]), (2, 2, 0))
        self.assertIs(cov.answer("divides", 4), False)            # the empty group lacks fig
        self.assertIs(mq.Coverage.build(child, cats, [], dictionary=d, parents=parents).answer("divides", 4), True)

    def item_scopes(self, db, sel=None, parents_keep=None):
        t, o = db["lines"], db["orders"]
        child = mbk.Scope.from_table(t, [], group_by=[("oid", o, "id")], rows=sel)
        cats = mbk.Scope.from_table(t, [], group_by=["item"], rows=sel)
        parents = mbk.Scope.from_table(o, [], group_by=["id"],
                                       rows=None if parents_keep is None else order_mask(db, parents_keep))
        return child, cats, parents

    def test_quant002_division_matches_duckdb_double_not_exists(self):
        rng = np.random.default_rng(1201)
        dense = random_db(rng, n_orders=50, n_lines=800)           # about 16 lines per order
        sparse = random_db(rng, n_orders=60, n_lines=150)          # about 3: every answer varies
        items = ["apple", "pear", "fig", "kiwi", "plum", "lime"]
        seen = set()
        for trial, (db, orders, sel, keep) in enumerate(
                ((dense[0], dense[1], None, None),
                 (dense[0], dense[1], rng.random(dense[0]["lines"].n) < 0.5, {i for i, _ in dense[1] if rng.random() < 0.6}),
                 (sparse[0], sparse[1], None, None))):
            child, cats, parents = self.item_scopes(db, sel, keep)
            where = mask_rows(db, sel) if sel is not None else "TRUE"
            psel = f'o."id" IN ({", ".join(map(str, sorted(keep)))})' if keep else "TRUE"
            for required in ([], ["apple"], ["apple", "pear"], ["apple", "pear", "fig", "kiwi"], items,
                             ["fig", "durian"], ["kiwi", None]):
                cov = self.cover_both(child, cats, required, parents=parents)
                sql = sql_coverage(db.con, f'SELECT l."oid" AS k, l."item" AS c FROM lines l WHERE {where}', required,
                                   f'SELECT o."id" AS k FROM orders o WHERE {psel}')
                self.assertEqual(native_coverage(cov), sql, (trial, required))
                seen |= {(q, r[q]) for r in sql.values() for q in ("divides", "exactly", "overlaps")}
                g = self.cover_both(child, cats, required)
                sql = sql_coverage(db.con, f'SELECT o."id" AS k, l."item" AS c FROM lines l LEFT JOIN orders o ON '
                                           f'l."oid" = o."id" WHERE {where}', required, match="IS NOT DISTINCT FROM")
                self.assertEqual(native_coverage(g), sql, (trial, required, "grouped"))
                if required and None not in required:          # the GROUP BY / HAVING COUNT(DISTINCT) division
                    req = ", ".join(repr(r) for r in required)
                    having = {None if k is None else int(k) for k, in db.con.execute(
                        f'SELECT o."id" FROM lines l LEFT JOIN orders o ON l."oid" = o."id" WHERE {where} AND '
                        f'l."item" IN ({req}) GROUP BY o."id" HAVING count(DISTINCT l."item") = '
                        f'(SELECT count(DISTINCT c) FROM (VALUES {", ".join(f"({r!r})" for r in required)}) r(c))'
                    ).fetchall()}
                    divides = {k for k, r in native_coverage(g).items() if r["divides"]}
                    self.assertEqual(divides, having, (trial, required))
        self.assertEqual(seen, {(q, v) for q in ("divides", "exactly", "overlaps") for v in (True, False)})

    def test_quant002_categories_through_a_dimension_and_integer_codes(self):
        rng = np.random.default_rng(1202)
        db, orders, lines = random_db(rng, n_orders=40, n_lines=700)
        t, o, p = db["lines"], db["orders"], db["products"]
        child = mbk.Scope.from_table(t, [], group_by=[("oid", o, "id")])
        parents = mbk.Scope.from_table(o, [], group_by=["id"])
        # the product name through the foreign key (unmatched and NULL references: the NULL category)
        names = mbk.Scope.from_table(t, [], group_by=[("pid", p, "name")])
        for required in (["p001", "p002"], ["p003", "p007", "p011", "p019"], ["p030"]):
            cov = self.cover_both(child, names, required, parents=parents)
            sql = sql_coverage(db.con, 'SELECT l."oid" AS k, p."name" AS c FROM lines l LEFT JOIN products p ON '
                                       'l."pid" = p."id"', required, 'SELECT o."id" AS k FROM orders o')
            self.assertEqual(native_coverage(cov), sql, required)
        self.assertEqual(cov.outside(), (["p030"], False))           # p030 is no product: missing everywhere
        # a NULL family attribute is a NULL category
        fam = mbk.Scope.from_table(t, [], group_by=[("pid", p, "family")])
        cov = self.cover_both(child, fam, ["fruit", "nut"], parents=parents)
        sql = sql_coverage(db.con, 'SELECT l."oid" AS k, p."family" AS c FROM lines l LEFT JOIN products p ON '
                                   'l."pid" = p."id"', ["fruit", "nut"], 'SELECT o."id" AS k FROM orders o')
        self.assertEqual(native_coverage(cov), sql)
        # integer codes: the quantity itself as the category (CODES: value = code + offset)
        qty = mbk.Scope.from_table(t, [], group_by=["qty"])
        d = mq.Dictionary.from_scope(qty)
        self.assertEqual(d.info["kind"], "codes")
        cov = self.cover_both(child, qty, [3, 7, 29, 500], parents=parents)
        sql = sql_coverage(db.con, 'SELECT l."oid" AS k, l."qty" AS c FROM lines l', [3, 7, 29, 500],
                           'SELECT o."id" AS k FROM orders o')
        self.assertEqual(native_coverage(cov), sql)
        self.assertEqual(cov.outside(), ([500], False))

    def test_quant002_identity_changes_with_required_set_and_dictionary(self):
        rng = np.random.default_rng(1203)
        db, orders, lines = random_db(rng, n_orders=30, n_lines=500)
        child, cats, parents = self.item_scopes(db)
        d = mq.Dictionary.from_scope(cats)
        a = mq.Coverage.build(child, cats, ["apple", "pear"], parents=parents)
        a2 = mq.Coverage.build(child, cats, ["pear", "apple", "pear"], parents=parents)   # the same set
        self.assertTrue(a.same(a2) and a.identity == a2.identity)
        b = mq.Coverage.build(child, cats, ["apple", "fig"], parents=parents)
        self.assertFalse(a.same(b))
        self.assertNotEqual(a.identity, b.identity)
        self.assertNotEqual(a.info["digest"], b.info["digest"])
        self.assertTrue(a.matches(child, cats, ["apple", "pear"], parents=parents))
        self.assertFalse(a.matches(child, cats, ["apple", "fig"], parents=parents))    # never reused for another set
        self.assertFalse(a.matches(child, cats, ["apple", "pear", None], parents=parents))
        self.assertFalse(a.matches(child, cats, ["apple", "pear", "durian"], parents=parents))
        # the same presence states under another required set: exactly the state a build gives, with its identity
        r = a.require(["apple", "fig"])
        self.assertTrue(r.same(b))
        self.assertEqual({k: v.tolist() for k, v in r.states().items()}, {k: v.tolist() for k, v in b.states().items()})
        self.assertFalse(r.same(a))
        # another dictionary (another codec: the codes name other values) is another state, never a silent reuse
        vals = list(d.values)
        null = d.null
        names = [v for v in vals if v is not None]
        remapped = list(reversed(names)) + ([None] if null is not None else [])
        d2 = mq.Dictionary.keys(remapped, null=null)
        self.assertFalse(d.same(d2))
        c = mq.Coverage.build(child, cats, ["apple", "pear"], dictionary=d2, parents=parents)
        self.assertFalse(c.same(a))
        self.assertFalse(a.matches(child, cats, ["apple", "pear"], dictionary=d2, parents=parents))
        g = int(np.flatnonzero(a.states()["distinct"] > 0)[0])
        self.assertEqual(sorted(c.categories(g)), sorted(remapped[k] for k in a.codes(g)))   # the codes, read anew
        # an integer dictionary with another offset is another dictionary too
        e1, e2 = mq.Dictionary.codes(cats.groups, null=null), mq.Dictionary.codes(cats.groups, null=null, offset=1)
        self.assertFalse(e1.same(e2))
        x1 = mq.Coverage.build(child, cats, [0, 1], dictionary=e1, parents=parents)
        x2 = mq.Coverage.build(child, cats, [1, 2], dictionary=e2, parents=parents)     # the same codes 0, 1
        self.assertEqual(x1.states()["matched"].tolist(), x2.states()["matched"].tolist())
        self.assertFalse(x1.same(x2))
        # merging states of other dictionaries or required sets is refused
        with self.assertRaises(DomainError):
            a.merge(b)
        with self.assertRaises(DomainError):
            a.merge(c)

    def test_quant002_masks_and_sets_agree_and_the_choice_is_costed(self):
        rng = np.random.default_rng(1204)
        G, D, n = 300, 5000, 4000
        groups = rng.integers(0, G, n).astype(np.uint32)
        codes = rng.integers(0, D, n).astype(np.uint32)
        codes[rng.random(n) < 0.05] = D - 1                       # the NULL category
        child = mbk.Scope.from_arrays({}, groups=groups, ngroups=G)
        cats = mbk.Scope.from_arrays({}, groups=codes, ngroups=D)
        d = mq.Dictionary.codes(D, null=D - 1)
        required = [int(v) for v in rng.choice(D - 1, 16, replace=False)]
        nm = mq.cover_need(child, cats, required, dictionary=d, representation="masks")
        ns = mq.cover_need(child, cats, required, dictionary=d, representation="sets")
        self.assertEqual((nm["states"], ns["states"]), (G * D, n))
        auto = mq.Coverage.build(child, cats, required, dictionary=d)
        self.assertEqual((auto.info["representation"], auto.info["choice"]), ("sets", "cheaper"))   # sparse presence
        masks = mq.Coverage.build(child, cats, required, dictionary=d, representation="masks")
        self.assertEqual({k: v.tolist() for k, v in auto.states().items()},
                         {k: v.tolist() for k, v in masks.states().items()})
        # masks over budget: AUTO falls back to sets (only fit), MASKS forced is refused
        budget = Budget(bytes=nm["bytes"] - 1)
        fit = mq.Coverage.build(child, cats, required, dictionary=d, budget=Budget(states=G * D - 1))
        self.assertEqual((fit.info["representation"], fit.info["choice"]), ("sets", "only fit"))
        with self.assertRaises(AlgebraBudgetError) as ctx:
            mq.Coverage.build(child, cats, required, dictionary=d, representation="masks", budget=budget)
        self.assertEqual((ctx.exception.resource, ctx.exception.need), ("bytes", nm["bytes"]))
        with self.assertRaises(AlgebraBudgetError):                # neither fits
            mq.Coverage.build(child, cats, required, dictionary=d, budget=Budget(bytes=min(nm["bytes"], ns["bytes"]) - 1))
        # dense presence (few categories): masks are cheaper
        small = mbk.Scope.from_arrays({}, groups=(codes % 8).astype(np.uint32), ngroups=8)
        s8 = mq.Coverage.build(child, small, [1, 2, 3], dictionary=mq.Dictionary.codes(8))
        self.assertEqual((s8.info["representation"], s8.info["choice"]), ("masks", "cheaper"))
        # the per-group sets against Python sets
        for g in rng.choice(G, 25, replace=False):
            m = groups == g
            present = {int(c) for c in codes[m] if c != D - 1}
            self.assertEqual(auto.codes(int(g)), sorted(present))
            self.assertEqual(auto.codes(int(g), "missing"), sorted(set(required) - present))
            self.assertEqual(auto.codes(int(g), "extra"), sorted(present - set(required)))

    def test_quant002_cuda_masks_and_coverage_test_equal_cpu(self):
        rng = np.random.default_rng(1205)
        G, D, n = 2000, 70, 50000                                  # two mask words per group
        sizes = rng.integers(0, 40, G)
        groups = np.repeat(np.arange(G, dtype=np.uint32), sizes)[:n]
        groups = np.concatenate([groups, rng.integers(0, G, n - len(groups)).astype(np.uint32)])
        groups[:n // 2].sort()                                     # contiguous runs, then scattered rows
        codes = rng.integers(0, D, n).astype(np.uint32)
        sel = rng.random(n) < 0.9
        child = mbk.Scope.from_arrays({}, groups=groups, ngroups=G, rows=sel)
        cats = mbk.Scope.from_arrays({}, groups=codes, ngroups=D, rows=sel)
        d = mq.Dictionary.codes(D, null=0)
        parents = mbk.Scope.from_arrays({}, groups=rng.integers(0, G, 3000).astype(np.uint32), ngroups=G)
        for required in ([1, 2, 3, 4], list(range(1, 17)), list(range(1, 65)), [69, 68, 2000]):
            cov = self.cover_both(child, cats, required, dictionary=d, parents=parents)
            cpu = cov.require([5, 6, 66])
            gpu = cov.require([5, 6, 66], backend="cuda")
            self.assertEqual({k: v.tolist() for k, v in cpu.states().items()},
                             {k: v.tolist() for k, v in gpu.states().items()})
            self.assertTrue(cpu.same(gpu))
        s = cov.states()
        for g in rng.choice(G, 30, replace=False):
            m = sel & (groups == g)
            present = {int(c) for c in codes[m] if c != 0}
            self.assertEqual((int(s["rows"][g]), int(s["nulls"][g]), int(s["distinct"][g])),
                             (int(m.sum()), int((codes[m] == 0).sum()), len(present)))
        with self.assertRaises(UnsupportedError):
            mq.Coverage.build(child, cats, [1], dictionary=d, representation="sets", backend="cuda")

    def test_quant002_masks_are_field001_packed_bits(self):
        from torbles import math_field as mf
        rng = np.random.default_rng(1206)
        db, orders, lines = random_db(rng, n_orders=40, n_lines=600)
        child, cats, parents = self.item_scopes(db)
        cov = mq.Coverage.build(child, cats, ["apple", "fig"], parents=parents, representation="masks")
        M = cov.masks()
        self.assertIsInstance(M, mf.Matrix)
        M.check()                                                  # canonical: no bit past the categories
        self.assertEqual((M.rows, M.cols), (child.groups, cats.groups))
        # FIELD-001's integer column counts: the groups that contain each category (SQL COUNT(DISTINCT) per item)
        counts = M.column_counts()
        d = cov.dictionary
        sql = dict(db.con.execute('SELECT l."item", count(DISTINCT CASE WHEN o."id" IS NULL THEN -1 ELSE o."id" END) '
                                  'FROM lines l LEFT JOIN orders o ON l."oid" = o."id" WHERE l."item" IS NOT NULL '
                                  'GROUP BY l."item"').fetchall())
        self.assertEqual({d.value(c): int(n) for c, n in enumerate(counts) if n}, {k: int(v) for k, v in sql.items()})
        self.assertEqual(counts[d.null], 0)                        # the NULL category has no bit
        self.assertEqual([mf.weight(row, cats.groups) for row in M.to_ints()], cov.states()["distinct"].tolist())
        with self.assertRaises(UnsupportedError):
            mq.Coverage.build(child, cats, ["apple"], parents=parents, representation="sets").masks()

    def test_quant002_merge_bag_union_and_multiplicity(self):
        rng = np.random.default_rng(1207)
        db, orders, lines = random_db(rng, n_orders=30, n_lines=500)
        t, o = db["lines"], db["orders"]
        parents = mbk.Scope.from_table(o, [], group_by=["id"])
        part = rng.random(t.n) < 0.5
        req = ["apple", "pear", "kiwi"]
        for representation in ("masks", "sets"):
            built = []
            for rows in (part, ~part, None):
                child = mbk.Scope.from_table(t, [], group_by=[("oid", o, "id")], rows=rows)
                cats = mbk.Scope.from_table(t, [], group_by=["item"], rows=rows)
                built.append(mq.Coverage.build(child, cats, req, parents=parents, representation=representation))
            a, b, whole = built
            ab = a.merge(b)
            self.assertEqual({k: v.tolist() for k, v in ab.states().items()},
                             {k: v.tolist() for k, v in whole.states().items()})
            self.assertFalse(ab.same(whole))
            aa = a.merge(a)                                        # every row twice: rows double, coverage unchanged
            sa, saa = a.states(), aa.states()
            self.assertTrue(np.array_equal(saa["rows"], 2 * sa["rows"]) and np.array_equal(saa["nulls"], 2 * sa["nulls"]))
            for key in ("distinct", "matched", "missing", "extra"):
                self.assertTrue(np.array_equal(saa[key], sa[key]), key)
            for qn in mq.QUESTIONS:
                self.assertTrue(np.array_equal(aa.answers(qn), a.answers(qn)))
        # duplicated child rows in the table: the same coverage as SQL over the duplicated rows
        twice = make_db(orders, lines + lines, [(i, f"p{i:03d}", None) for i in range(1, 25)])
        child, cats, parents2 = self.item_scopes(twice)
        cov = self.cover_both(child, cats, req, parents=parents2)
        sql = sql_coverage(twice.con, 'SELECT l."oid" AS k, l."item" AS c FROM lines l', req,
                           'SELECT o."id" AS k FROM orders o')
        self.assertEqual(native_coverage(cov), sql)

    def test_quant002_budgets_at_the_limit_and_one_past(self):
        rng = np.random.default_rng(1208)
        db, orders, lines = random_db(rng, n_orders=20, n_lines=300)
        child, cats, parents = self.item_scopes(db)
        req = ["apple", "pear", "durian"]
        for representation, backend in (("masks", "cpu"), ("sets", "cpu"), ("masks", "cuda")):
            need = mq.cover_need(child, cats, req, parents=parents, representation=representation, backend=backend)
            self.assertEqual((need["support"], need["output"]), (cats.groups, child.groups))
            for resource in ("support", "states", "output", "bytes"):
                mq.Coverage.build(child, cats, req, parents=parents, representation=representation, backend=backend,
                                  budget=Budget(**{resource: need[resource]}))
                with self.assertRaises(AlgebraBudgetError) as ctx:
                    mq.Coverage.build(child, cats, req, parents=parents, representation=representation,
                                      backend=backend, budget=Budget(**{resource: need[resource] - 1}))
                self.assertEqual((ctx.exception.resource, ctx.exception.need), (resource, need[resource]))
        mq.Dictionary.codes(1000, budget=Budget(support=1000))
        with self.assertRaises(AlgebraBudgetError):
            mq.Dictionary.codes(1001, budget=Budget(support=1000))
        keys = [f"k{i}" for i in range(50)] + [None]
        need = mq.dictionary_need(keys)
        self.assertEqual(need["support"], 51)
        self.assertGreaterEqual(need["bytes"], 51 * 4 * 2 + 52 * 8 + sum(len(k) for k in keys if k))
        mq.Dictionary.keys(keys, null=50, budget=Budget(bytes=need["bytes"]))
        with self.assertRaises(AlgebraBudgetError) as ctx:
            mq.Dictionary.keys(keys, null=50, budget=Budget(bytes=need["bytes"] - 1))
        self.assertEqual((ctx.exception.resource, ctx.exception.need), ("bytes", need["bytes"]))
        self.assertEqual(mq.dictionary_need(size=7), dict(support=7, bytes=mq.dictionary_need(size=8)["bytes"]))

    def test_quant002_stale_and_malformed_requests_are_refused(self):
        rng = np.random.default_rng(1209)
        db, orders, lines = random_db(rng, n_orders=20, n_lines=200)
        t, o = db["lines"], db["orders"]
        child, cats, parents = self.item_scopes(db)
        cov = mq.Coverage.build(child, cats, ["apple"], parents=parents)
        other_rows = mbk.Scope.from_table(t, [], group_by=["item"], rows=np.arange(t.n) % 2 == 0)
        with self.assertRaises(DomainError):                         # the categories of other rows
            mq.Coverage.build(child, other_rows, ["apple"], parents=parents)
        with self.assertRaises(DomainError):                         # a dictionary of another size
            mq.Coverage.build(child, cats, [0], dictionary=mq.Dictionary.codes(cats.groups + 1), parents=parents)
        with self.assertRaises(ValueError):                          # a dictionary maps codes to distinct keys
            mq.Dictionary.keys(["a", "b", "a"])
        with self.assertRaises(ValueError):                          # the NULL code has no key
            mq.Dictionary.keys(["a", "b"], null=1)
        with self.assertRaises(TypeError):
            mq.Coverage.build(child, cats, [1.5], dictionary=mq.Dictionary.codes(cats.groups), parents=parents)
        with self.assertRaises(ValueError):
            cov.answers("contains")
        with self.assertRaises(ValueError):
            mq.Coverage.build(child, cats, ["apple"], parents=parents, representation="bitmap")
        with self.assertRaises(UnsupportedError):                    # a category scope has one key
            mq.Dictionary.from_scope(mbk.Scope.from_table(t, [], group_by=["item", "qty"]))
        t.flip(0, 0)
        with self.assertRaises(StaleStateError):
            cov.answers("divides")
        with self.assertRaises(StaleStateError):
            cov.require(["pear"])
        for backend in ("cpu", "cuda"):
            with self.assertRaises(StaleStateError):
                mq.Coverage.build(child, cats, ["apple"], parents=parents, backend=backend)

    def test_quant002_ctypes_structures_are_registered(self):
        report = mh.abi_check()
        rows = [r for r in report["checked"] if r["structure"].startswith("torbles.math_quant.")]
        self.assertTrue(rows and all(r["match"] for r in rows))
        self.assertIn("tr_quant_cover_arrays", {r["native"] for r in rows if r["library"] == "cuda"})


if __name__ == "__main__":
    unittest.main()
