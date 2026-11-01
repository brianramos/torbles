# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""PROV-001 (factorized provenance payloads) (torbles/math_prov.c, math_prov_gpu.cu, math_prov.py).

Oracles, independent of the implementation: derivation polynomials written out in the tests (Python dicts of
monomials -> integer coefficients, a monomial being the sorted tuple of the source tuples it uses), DuckDB SQL
re-executing the query (bag multiplicities with GROUP BY ALL / COUNT(*), after deleting source rows, with duplicated
rows for multiplicities), and Python set enumeration of the minimal witnesses. Every evaluation is also run on the CUDA
path (emulated here, TORBLES_EMU_SCRAMBLE=1 scrambles the thread order) and compared value for value with the CPU.
Seeds are fixed.
"""
import itertools
import subprocess
import sys
import unittest

import duckdb
import numpy as np
import pandas as pd

import torbles
from torbles import math_bank as mbk, math_prov as mp
from torbles.math_budgets import Budget
from torbles.math_domains import DomainError
from torbles.mathematics import AlgebraBudgetError

R_ROWS = [(1, 10), (1, 20), (2, 30), (None, 40), (1, 10)]          # a duplicate row and a NULL key
S_ROWS = [(1, 5), (2, 6), (2, 7), (3, 8)]
T_ROWS = [(5, 100), (6, 100), (6, 200), (7, 300)]

def cuda_label():
    """the backend label of a CUDA result: "cuda" on a device, "cuda-emulated" when the kernels run in the emulation"""
    from torbles import math_gpu
    return "cuda-emulated" if math_gpu.device()["emulated"] else "cuda"


def m(*labels):
    return tuple(sorted(labels))


def r(i):
    return ("R", i)


def s(i):
    return ("S", i)


def t(i):
    return ("T", i)


def cols(rows, names):
    return {n: [row[c] for row in rows] for c, n in enumerate(names)}


def con_for(tables):
    con = duckdb.connect()
    for name, (rows, names) in tables.items():
        df = pd.DataFrame({"rid": list(range(len(rows))),
                           **{n: pd.Series([row[c] for row in rows], dtype="Int64") for c, n in enumerate(names)}})
        con.register(name + "_all", df)
    return con


def sqlbag(con, sql):
    """{tuple: multiplicity} from rows (..., count)"""
    out = {}
    for row in con.execute(sql).fetchall():
        key = tuple(None if v is None else int(v) for v in row[:-1])
        out[key] = out.get(key, 0) + int(row[-1])
    return out


class Provenance(unittest.TestCase):
    def schema(self, st):
        R = st.source("R", cols(R_ROWS, ["k", "a"]))
        S = st.source("S", cols(S_ROWS, ["k", "b"]))
        T = st.source("T", cols(T_ROWS, ["b", "c"]))
        return R, S, T

    # ------------------------------------------------------------------ acceptance 1: explicit polynomials, multiplicities
    def test_prov001_union_join_group_match_explicit_derivation_polynomials_and_multiplicities(self):
        st = mp.Store()
        R, S, T = self.schema(st)
        # source: equal rows are one tuple whose annotation sums their variables
        self.assertEqual(R.tuples(), [(1, 10), (1, 20), (2, 30), (None, 40)])
        self.assertEqual([R.polynomial(i).terms for i in range(4)],
                         [{m(r(0)): 1, m(r(4)): 1}, {m(r(1)): 1}, {m(r(2)): 1}, {m(r(3)): 1}])
        # join R.k = S.k: products of the joined tuples (the NULL key joins nothing)
        J = R.join(S, on=[("k", "k")])
        self.assertEqual(J.tuples(), [(1, 10, 1, 5), (1, 20, 1, 5), (2, 30, 2, 6), (2, 30, 2, 7)])
        self.assertEqual([J.polynomial(i) for i in range(4)],
                         [{m(r(0), s(0)): 1, m(r(4), s(0)): 1}, {m(r(1), s(0)): 1}, {m(r(2), s(1)): 1},
                          {m(r(2), s(2)): 1}])
        # group: the sum of the members (a three-way join, grouped)
        J3 = J.join(T, on=[("b", "b")])
        G = J3.group(["c"])
        self.assertEqual(G.tuples(), [(100,), (200,), (300,)])
        self.assertEqual(G.polynomial(0), {m(r(0), s(0), t(0)): 1, m(r(4), s(0), t(0)): 1, m(r(1), s(0), t(0)): 1,
                                           m(r(2), s(1), t(1)): 1})
        self.assertEqual(G.polynomial(1), {m(r(2), s(1), t(2)): 1})
        self.assertEqual(G.polynomial(2), {m(r(2), s(2), t(3)): 1})
        p = G.polynomial(0)
        self.assertEqual((p.semiring, p.derivations, p.degree), ("N[X]", 4, 3))
        # union ALL: equal tuples add
        R2 = st.source("R2", cols([(1, 20), (5, 50)], ["k", "a"]))
        U = R.union(R2)
        self.assertEqual(U.tuples(), [(1, 10), (1, 20), (2, 30), (5, 50), (None, 40)])
        self.assertEqual(U.polynomial(1), {m(r(1)): 1, m(("R2", 0)): 1})
        UU = U.union(R)                                           # repeated sources: coefficients
        self.assertEqual(UU.polynomial(0), {m(r(0)): 2, m(r(4)): 2})
        # multiplicities (N at 1) equal DuckDB's bag multiplicities of the same queries
        con = con_for({"R": (R_ROWS, ["k", "a"]), "S": (S_ROWS, ["k", "b"]), "T": (T_ROWS, ["b", "c"]),
                       "R2": ([(1, 20), (5, 50)], ["k", "a"])})
        queries = {
            "join": (J, "SELECT r.k, r.a, s.k, s.b, COUNT(*) FROM R r JOIN S s ON r.k = s.k GROUP BY ALL"),
            "group": (G, "SELECT t.c, COUNT(*) FROM R r JOIN S s ON r.k = s.k JOIN T t ON s.b = t.b GROUP BY t.c"),
            "union": (U, "SELECT k, a, COUNT(*) FROM (SELECT k, a FROM R UNION ALL SELECT k, a FROM R2) GROUP BY ALL"),
        }

        def views(deleted):
            for name in ("R", "S", "T", "R2"):
                ids = [i for (src, i) in deleted if src == name] or [-1]
                con.execute(f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM {name}_all WHERE rid NOT IN "
                            f"({', '.join(map(str, ids))})")
        views([])
        for name, (rel, sql) in queries.items():
            want = sqlbag(con, sql)
            self.assertEqual(dict(zip(rel.tuples(), rel.counts())), want, name)
        # every single-source deletion: N and B against DuckDB re-executing the query, on the CPU and CUDA paths
        labels = [r(i) for i in range(5)] + [s(i) for i in range(4)] + [t(i) for i in range(4)] + [("R2", 0), ("R2", 1)]
        for name, (rel, sql) in queries.items():
            dels = [[lab] for lab in labels] + [[r(0), r(4)], [s(0), t(1)]]
            cpu = st.evaluate(rel, "count", deletions=dels)
            gpu = st.evaluate(rel, "count", deletions=dels, backend="cuda")
            bcpu = st.evaluate(rel, "bool", deletions=dels)
            bgpu = st.evaluate(rel, "bool", deletions=dels, backend="cuda")
            self.assertEqual((cpu.semiring, bcpu.semiring, gpu.backend), ("N", "B", cuda_label()))
            self.assertEqual((cpu.values, bcpu.values), (gpu.values, bgpu.values))
            for w, d in enumerate(dels):
                views(d)
                want = sqlbag(con, sql)
                got = {tp: c for tp, c in zip(rel.tuples(), cpu.values[w]) if c}
                self.assertEqual(got, want, (name, d))
                self.assertEqual({tp for tp, b in zip(rel.tuples(), bcpu.values[w]) if b}, set(want), (name, d))
        views([])
        # B: which sources suffice (the minimal witnesses), against an enumeration of the source subsets
        W = G.witnesses(0)
        self.assertEqual(W.semiring, "B")
        self.assertEqual(set(W.sets), {frozenset(x) for x in [(r(0), s(0), t(0)), (r(4), s(0), t(0)),
                                                              (r(1), s(0), t(0)), (r(2), s(1), t(1))]})
        uni = sorted({x for mono in G.polynomial(0).terms for x in mono})
        suff = []
        for k in range(len(uni) + 1):
            for sub in itertools.combinations(uni, k):
                if any(set(mono) <= set(sub) for mono in G.polynomial(0).terms):
                    if not any(set(x) < set(sub) for x in suff):
                        suff.append(sub)
        self.assertEqual(set(W.sets), {frozenset(x) for x in suff})
        # correlated derivations: a self-join uses one tuple twice (x^2); multiplicities as bag semantics
        SJ = R.join(R, on=[("k", "k")])
        self.assertEqual(SJ.polynomial(0), {m(r(0), r(0)): 1, m(r(0), r(4)): 2, m(r(4), r(4)): 1})
        rows3 = R_ROWS + [R_ROWS[2]]                              # row 2 duplicated: multiplicity 2
        con.register("R3", pd.DataFrame({"k": pd.Series([x[0] for x in rows3], dtype="Int64"),
                                         "a": [x[1] for x in rows3]}))
        want = sqlbag(con, "SELECT a.k, a.a, b.k, b.a, COUNT(*) FROM R3 a JOIN R3 b ON a.k = b.k GROUP BY ALL")
        got = st.evaluate(SJ, "count", valuations=[{r(2): 2}])
        self.assertEqual(dict(zip(SJ.tuples(), got.values[0])), want)
        self.assertEqual(st.evaluate(SJ, "count", valuations=[{r(2): 2}], backend="cuda").values, got.values)
        # tables reach the store through FND-004 scopes (the physical rows of a selection)
        db = torbles.from_ddl("CREATE TABLE q (k BIGINT, v BIGINT)",
                              {"q": pd.DataFrame({"k": [1, 2, 1, 3], "v": [7, 8, 7, 9]})})
        sc = mbk.Scope.from_table(db["q"], ["k", "v"], rows=np.array([True, True, True, False]))
        Q = st.source_scope("Q", sc, ["k", "v"])
        self.assertEqual(Q.tuples(), [(1, 7), (2, 8)])
        self.assertEqual(Q.polynomial(0), {m(("Q", 0)): 1, m(("Q", 2)): 1})
        con.close()

    # ------------------------------------------------------------------ acceptance 2: factored, limits, NULL / presence
    def test_prov001_shared_derivations_stay_factored_limits_and_null_group_presence_are_explicit(self):
        st = mp.Store()
        n_r, n_t = 20, 30
        R = st.source("R", {"k": [i % 2 for i in range(n_r)], "a": list(range(n_r))})
        T = st.source("T", {"k": [i % 2 for i in range(n_t)], "c": list(range(n_t))})
        Gk = R.group(["k"])                                       # two shared sums of 10 variables each
        before = st.info["nodes"]
        J = Gk.join(T, on=[("k", "k")])
        info = st.info
        self.assertEqual(info["nodes"] - before, n_t)            # one product node per output: the sums are shared
        self.assertEqual(J.info["new_nodes"], n_t)
        sums = Gk.annotations()
        self.assertEqual([st.parents(x) for x in sums], [n_t // 2, n_t // 2])   # each sum used by 15 outputs
        expanded = sum(len(J.polynomial(i).terms) for i in range(len(J)))
        self.assertEqual(expanded, n_t * n_r // 2)               # the expansion is quadratic, the DAG linear
        self.assertGreaterEqual(info["shared"], 2)
        again = Gk.join(T, on=[("k", "k")])                      # the same derivations: no new node
        self.assertEqual((st.info["nodes"], again.info["new_nodes"]), (info["nodes"], 0))
        self.assertEqual(st.info["hits"] - info["hits"], n_t)
        # annotation limits: nodes (circuit) and child entries (support) at the limit and one past
        def build(limits):
            s2 = mp.Store(budget=Budget(**limits))
            R2 = s2.source("R", {"k": [1, 1, 2], "a": [1, 2, 3]})
            S2 = s2.source("S", {"k": [1, 2, 2], "b": [4, 5, 6]})
            return s2, R2.join(S2, on=[("k", "k")]).group(["k"])
        s_ok, g_ok = build({})
        need_nodes, need_kids = s_ok.info["nodes"], s_ok.info["children"]
        for resource, value in (("circuit", need_nodes), ("support", need_kids)):
            s2, g2 = build({resource: value})
            self.assertEqual(g2.tuples(), g_ok.tuples())
            with self.assertRaises(AlgebraBudgetError) as ctx:
                build({resource: value - 1})
            self.assertEqual((ctx.exception.resource, ctx.exception.need), (resource, value))
        # the operators report what they check: the join's store need is its bound
        s3 = mp.Store()
        A = s3.source("A", {"k": [1, 1, 2], "i": [0, 1, 2]})          # distinct rows: 3 and 4 tuples
        B = s3.source("B", {"k": [1, 2, 2, 2], "j": [0, 1, 2, 3]})
        nd, before = A.need("join", B, on=[("k", "k")]), s3.info["nodes"]
        self.assertEqual((nd["call"]["output"], nd["store"]["circuit"], nd["store"]["support"]),
                         (5, before + 5, s3.info["children"] + 10))
        self.assertEqual(len(A.join(B, on=[("k", "k")])), 5)
        self.assertLessEqual(s3.info["nodes"], before + 5)
        # expansions: support = the number of derivations, degree; refused at one past, never truncated
        D = len(J.polynomial(0).terms)
        en = J.expand_need(0)
        self.assertEqual((en["support"], en["degree"]), (D, 2))
        J.polynomial(0, budget=Budget(support=D))
        with self.assertRaises(AlgebraBudgetError) as ctx:
            J.polynomial(0, budget=Budget(support=D - 1))
        self.assertEqual((ctx.exception.resource, ctx.exception.need), ("support", D))
        with self.assertRaises(AlgebraBudgetError) as ctx:
            J.polynomial(0, budget=Budget(degree=1))
        self.assertEqual(ctx.exception.resource, "degree")
        # evaluations: output = valuations x outputs, the device copies' bytes
        dels = [[("T", i)] for i in range(5)]
        st.evaluate(J, "count", deletions=dels, budget=Budget(output=5 * n_t))
        for backend in ("cpu", "cuda"):
            with self.assertRaises(AlgebraBudgetError) as ctx:
                st.evaluate(J, "count", deletions=dels, budget=Budget(output=5 * n_t - 1), backend=backend)
            self.assertEqual((ctx.exception.resource, ctx.exception.need), ("output", 5 * n_t))
        with self.assertRaises(AlgebraBudgetError) as ctx:
            st.evaluate(J, "bool", deletions=dels, budget=Budget(bytes=64), backend="cuda", plan=st.plan(J))
        self.assertEqual(ctx.exception.resource, "bytes")
        # NULL is explicit: NULL keys never join, form their own group, fail comparisons, pass IS NULL
        sn = mp.Store()
        N = sn.source("N", {"k": [None, 1, None, 2], "v": [1, 2, 3, None]})
        M = sn.source("M", {"k": [None, 1], "w": [9, 8]})
        self.assertEqual(N.join(M, on=[("k", "k")]).tuples(), [(1, 2, 1, 8)])
        g = N.group(["k"])
        self.assertEqual(g.tuples(), [(1,), (2,), (None,)])
        self.assertEqual(g.polynomial(2), {m(("N", 0)): 1, m(("N", 2)): 1})
        self.assertEqual(N.where(("v", ">", 1)).tuples(), [(1, 2), (None, 3)])   # NULL > 1 is UNKNOWN: dropped
        self.assertEqual(N.where(("v", "is null")).tuples(), [(2, None)])
        self.assertEqual(N.where(("k", "=", ("col", "v"))).tuples(), [])        # every row has a NULL or k != v
        self.assertEqual(N.union(N).tuples(), N.tuples())          # NULL equals NULL when tuples merge
        # group presence: a group derived from no tuple does not exist; the global group always does
        con = duckdb.connect()
        con.register("N_all", pd.DataFrame({"rid": [0, 1, 2, 3], "k": pd.Series([None, 1, None, 2], dtype="Int64")}))
        dels = [[("N", 1)], [("N", 0), ("N", 2)], [("N", i) for i in range(4)]]
        cnt = sn.evaluate(g, "count", deletions=dels)
        ex = sn.evaluate(g, "bool", deletions=dels, backend="cuda")
        for w, d in enumerate(dels):
            ids = ", ".join(str(i) for _, i in d)
            want = sqlbag(con, f"SELECT k, COUNT(*) FROM N_all WHERE rid NOT IN ({ids}) GROUP BY k")
            self.assertEqual({tp: c for tp, c, e in zip(g.tuples(), cnt.values[w], ex.values[w]) if e}, want)
            self.assertEqual([c > 0 for c in cnt.values[w]], ex.values[w])     # present iff derived
        glob = N.group([])
        self.assertTrue(glob.info["global_group"])
        everything = [[("N", i) for i in range(4)]]
        self.assertEqual(sn.evaluate(glob, "count", deletions=everything).values, [[0]])   # present, COUNT(*) = 0
        self.assertEqual(con.execute("SELECT COUNT(*) FROM N_all WHERE rid NOT IN (0, 1, 2, 3)").fetchone()[0], 0)
        self.assertEqual(len(glob), 1)
        # the semirings are never confused: one union, three interpretations
        U = N.union(N)
        self.assertEqual(U.polynomial(0), {m(("N", 1)): 2})                       # N[X]: 2 x
        self.assertEqual((U.counts()[0], U.exists()[0]), (2, True))               # N: 2; B: exists
        self.assertEqual((sn.evaluate(U, "count", deletions=[[]]).semiring,
                          sn.evaluate(U, "bool", deletions=[[]]).semiring), ("N", "B"))
        with self.assertRaises(ValueError):
            sn.evaluate(U, "N[X]", deletions=[[]])                   # N[X] is expanded, not evaluated
        # N past 128 bits: that valuation's status, on both paths; the others unaffected
        big = (1 << 64) - 1
        X3 = N.join(N, on=[("k", "k")]).join(N, on=[("k", "k")])
        vals = [{("N", 1): big}, {("N", 1): 3}]
        c1 = sn.evaluate(X3, "count", valuations=vals)
        c2 = sn.evaluate(X3, "count", valuations=vals, backend="cuda")
        self.assertEqual((c1.status, c1.values), (c2.status, c2.values))
        self.assertEqual(c1.status, ["overflow", None])
        self.assertEqual(c1.values[1][X3.tuples().index((1, 2, 1, 2, 1, 2))], 27)
        con.close()

    def test_prov001_cuda_evaluation_of_many_valuations_equals_cpu(self):
        rng = np.random.default_rng(1001)
        st = mp.Store()
        R = st.source("R", {"k": [int(x) for x in rng.integers(0, 6, 60)], "a": [int(x) for x in rng.integers(0, 4, 60)]})
        S = st.source("S", {"k": [int(x) for x in rng.integers(0, 6, 40)], "b": [int(x) for x in rng.integers(0, 3, 40)]})
        Q = R.join(S, on=[("k", "k")]).group(["a", "b"]).union(R.group(["a"]).join(S.group(["b"]))).group([])
        J = R.join(S, on=[("k", "k")]).group(["a", "b"])
        labels = [("R", i) for i in range(60)] + [("S", i) for i in range(40)]
        dels = [[labels[int(i)] for i in rng.choice(100, int(rng.integers(0, 6)), replace=False)] for _ in range(300)]
        vals = [{labels[int(i)]: int(rng.integers(0, 4)) for i in rng.choice(100, 8, replace=False)} for _ in range(100)]
        pl = st.plan([J, Q])
        self.assertGreater(pl.info["nodes"], 100)
        for semiring in ("count", "bool"):
            a = st.evaluate([J, Q], semiring, deletions=dels, plan=pl)
            raw = [x.copy() for x in pl._last]
            b = st.evaluate([J, Q], semiring, deletions=dels, plan=pl, backend="cuda")
            self.assertTrue(all(np.array_equal(x, y) for x, y in zip(raw, pl._last)))   # the same bytes
            self.assertEqual((a.values, a.status), (b.values, b.status))
        a = st.evaluate([J, Q], "count", valuations=vals, default=2)
        b = st.evaluate([J, Q], "count", valuations=vals, default=2, backend="cuda")
        self.assertEqual(a.values, b.values)
        # the valuation semantics against the expanded polynomials (exact Python evaluation)
        polys = [J.polynomial(i) for i in range(len(J))]
        for w in range(5):
            nu = dict.fromkeys(labels, 2)
            nu.update(vals[w])
            for j, p in enumerate(polys):
                want = 0
                for mono, c in p.terms.items():
                    term = c
                    for lab in mono:
                        term *= nu[lab]
                    want += term
                self.assertEqual(a.of(w, 0)[j], want)

    def test_prov001_invalid_inputs_and_opt_in_import(self):
        st = mp.Store()
        R = st.source("R", {"k": [1, 2], "a": [1, 2]}, scales={"a": 2})
        S = st.source("S", {"k": [1], "b": [1]})
        with self.assertRaises(DomainError):
            R.union(st.source("X", {"k": [1]}))                   # different arities
        with self.assertRaises(DomainError):
            R.join(S, on=[("a", "b")])                            # decimal scales differ
        with self.assertRaises(ValueError):
            R.where(("a", "~", 1))
        other = mp.Store()
        O = other.source("O", {"k": [1]})
        with self.assertRaises(ValueError):
            R.join(O, on=[("k", "k")])                            # a relation of another store
        with self.assertRaises(ValueError):
            st.evaluate(R, "count", deletions=[[("Z", 0)]])
        with self.assertRaises(TypeError):
            st.source("F", {"x": [0.5]})                         # values are exact logical integers
        with self.assertRaises(AlgebraBudgetError):
            mp.Store(budget=Budget(circuit=1))
        code = "import sys, torbles; print('torbles.math_prov' in sys.modules)"
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.split()
        self.assertEqual(out, ["False"])


if __name__ == "__main__":
    unittest.main()
