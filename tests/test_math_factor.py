# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""PROB-002 (factorized probabilistic inference) (torbles/math_factor.c, math_factor_gpu.cu, math_factor.py).

Oracles, independent of the implementation: exhaustive enumeration of the joint support written in the tests with
Python Fractions (the weight of every assignment as the product of its factors' entries; Z, P(e), marginals,
conditionals, the maximum and its ties, the k best in canonical order); DuckDB GROUP BY counts (and the join of two
GROUP BYs) behind the count-tensor factors; PROB-001's exact categorical normalizer and COUNT-002's elimination /
brute force; a brute force over every elimination order (an independent graph simulation) for the planner's exact
minima and its proofs that no order fits. Every elimination is also run on the CUDA
path (emulated here; TORBLES_EMU_SCRAMBLE=1 scrambles the thread order) and compared bit for bit (the raw arena:
sums, maxima, counts, argmax) and answer for answer with the CPU. Seeds are fixed.
"""
import itertools
import unittest
from fractions import Fraction

import duckdb
import numpy as np
import pandas as pd

import torbles
from torbles import math_bank as mbk, math_conv as mc, math_dist as md, math_factor as mf, math_prob as mp
from torbles import math_symcount as ms, math_tensor as mt
from torbles.math_budgets import Budget
from torbles.math_domains import DomainError
from torbles.math_errors import StaleStateError, UnsupportedError
from torbles.mathematics import AlgebraBudgetError


# ---------------------------------------------------------------------- the oracle: exhaustive enumeration
class Oracle:
    """the joint of a factor graph given as {name: labels} and [(scope, {labels tuple: Fraction})], enumerated"""

    def __init__(self, variables, factors):
        self.names = list(variables)
        self.labels = [list(variables[n]) for n in self.names]
        self.factors = factors
        self.joint = {}
        for asg in itertools.product(*self.labels):
            a = dict(zip(self.names, asg))
            w = Fraction(1)
            for scope, tab in factors:
                w *= Fraction(tab[tuple(a[v] for v in scope)])
            self.joint[asg] = w

    def consistent(self, ev):
        return {k: w for k, w in self.joint.items() if all(k[self.names.index(v)] == x for v, x in (ev or {}).items())}

    def Z(self, ev=None):
        return sum(self.consistent(ev).values(), Fraction(0))

    def marginal(self, vs, ev=None):
        Ze = self.Z(ev)
        out = {key: Fraction(0) for key in itertools.product(*[self.labels[self.names.index(v)] for v in vs])}
        for k, w in self.consistent(ev).items():
            out[tuple(k[self.names.index(v)] for v in vs)] += w
        return {k: v / Ze for k, v in out.items()}

    def conditional(self, vs, given, ev=None):
        joint = self.marginal(list(vs) + list(given), ev)
        out = {}
        for key, p in joint.items():
            q, r = key[:len(vs)], key[len(vs):]
            denom = sum(pp for kk, pp in joint.items() if kk[len(vs):] == r)
            out[key] = None if denom == 0 else p / denom
        return out

    def ranked(self, ev=None):
        """(weight, assignment) of positive weight, weight decreasing then lexicographic by variable (index order)"""
        items = [(w, k) for k, w in self.consistent(ev).items() if w > 0]
        idx = [{lab: i for i, lab in enumerate(ls)} for ls in self.labels]
        items.sort(key=lambda t: (-t[0], tuple(idx[i][t[1][i]] for i in range(len(self.names)))))
        return items


def table_dict(scope_labels, nested):
    """{labels tuple: Fraction} of a nested-list table over the scopes' labels"""
    arr = np.array(nested, dtype=object)
    return {key: Fraction(arr[tuple(i for i in idx)])
            for idx, key in zip(itertools.product(*[range(len(l)) for l in scope_labels]),
                                itertools.product(*scope_labels))}


def random_graph(rng, names, domains, scopes, zero=0.2, dens=(1, 2, 3, 5)):
    """random rational tables (some zeros) over the scopes: (variables dict, mf factors, oracle factors)"""
    V = {n: list(range(d)) for n, d in zip(names, domains)}
    F, O = [], []
    for sc in scopes:
        dims = [domains[names.index(v)] for v in sc]
        flat = []
        for _ in range(int(np.prod(dims)) if dims else 1):
            if rng.random() < zero:
                flat.append(Fraction(0))
            else:
                flat.append(Fraction(int(rng.integers(1, 9)), int(rng.choice(dens))))
        nested = np.array(flat, dtype=object).reshape(dims) if dims else np.array(flat[0], dtype=object)
        F.append(mf.table(list(sc), nested.tolist() if dims else [flat[0]]))
        O.append((tuple(sc), {k: flat[i] for i, k in enumerate(itertools.product(*[V[v] for v in sc]))}))
    return V, F, O


def both(G, keep, evidence, mode, kind, given=0, **kw):
    """one retained query (its plan and program) run on the CPU and with the (emulated) CUDA kernels, checked equal
    bit for bit (the raw arena: sums, maxima, counts, argmax) and answer for answer"""
    fixed, ev = G._evidence(evidence)
    q = G.query(tuple(keep), evidence=fixed, mode=mode, kind=kind, given=given, requests=ev.shape[0], **kw)
    cpu = q.run(ev, backend="cpu")
    gpu = q.run(ev, backend="cuda")
    assert cpu.backend == "cpu" and gpu.backend in ("cuda", "cuda-emulated"), (cpu.backend, gpu.backend)
    assert cpu.equal(gpu), "the CUDA arena or answers differ from the CPU"
    return cpu


def arrays_scope(cols, valid=None, groups=None, ngroups=None):
    return mbk.Scope.from_arrays({k: np.asarray(v, dtype=np.int64) for k, v in cols.items()}, valid=valid,
                                 groups=None if groups is None else np.asarray(groups, dtype=np.uint32), ngroups=ngroups)


def order_metrics(edges, dom, eliminate, keep, order):
    """width, largest joint table, largest intermediate table of an elimination order (graph simulation)"""
    nb = {v: set() for v in dom}
    for a, b in edges:
        if a != b:
            nb[a].add(b); nb[b].add(a)
    width = joint = support = 0
    for x in order:
        N = set(nb[x])
        msg = 1
        for u in N:
            msg *= dom[u]
        width, joint, support = max(width, len(N)), max(joint, msg * dom[x]), max(support, msg)
        for u in N:
            nb[u] |= N - {u}
            nb[u].discard(x)
        del nb[x]
    fin = 1
    for q in keep:
        fin *= dom[q]
    return max(width, len(keep) - 1 if keep else 0), max(joint, fin), max(support, fin)


class Prob002(unittest.TestCase):
    # ------------------------------------------------------------------ tiny correlated graphs vs enumeration
    def check_graph(self, V, F, O, queries, evidences):
        G = mf.graph(V, F)
        orc = Oracle(V, O)
        Z = orc.Z()
        self.assertEqual(G.partition().Z_e, Z)
        for meth in ("symcount", "brute", "tensor"):
            self.assertEqual(G.count(method=meth), Z, meth)
        for ev in evidences:
            Ze = orc.Z(ev)
            b = both(G, (), ev, "sum", "partition")
            self.assertEqual((b[0].Z_e, b[0].probability), (Ze, Ze / Z if Z else None))
            self.assertEqual(G.count(ev, method="symcount"), Ze)
            for vs in queries:
                vs = [v for v in vs if v not in (ev or {})]
                if not vs:
                    continue
                m = both(G, vs, ev, "sum", "marginal")[0]
                if Ze == 0:
                    self.assertEqual(m.status, -6)                      # the evidence has probability zero
                    continue
                self.assertEqual(m.table, orc.marginal(vs, ev), (vs, ev))
                self.assertEqual(m.Z_e, Ze)
                if len(vs) >= 2:
                    c = both(G, vs, ev, "sum", "conditional", given=1)[0]
                    self.assertEqual(c.table, orc.conditional(vs[:-1], vs[-1:], ev), (vs, ev))
            mx = both(G, (), ev, "max", "max")[0]
            ranked = orc.ranked(ev)
            if not ranked:
                self.assertEqual(mx.status, -6)
                continue
            best = ranked[0][0]
            ties = [dict(zip(orc.names, k)) for w, k in ranked if w == best]
            self.assertEqual((mx.weight, mx.map_probability, mx.count), (best, best / Ze, len(ties)))
            self.assertEqual(mx.assignments, ties)
            for kk in (1, 2, 5):
                kb = G.kbest(kk, ev)
                want = ranked[:kk]
                self.assertEqual([(a, w) for a, w, _ in kb.best], [(dict(zip(orc.names, k)), w) for w, k in want])
                self.assertEqual([p for _, _, p in kb.best], [w / Ze for w, _ in want])
                if len(ranked) > kk:
                    wk = ranked[kk - 1][0]
                    self.assertEqual(kb.tied, sum(1 for w, _ in ranked[kk:] if w == wk))
        return G

    def test_prob002_tiny_correlated_graphs_match_exhaustive_enumeration(self):
        rng = np.random.default_rng(2026)
        shapes = [
            # (names, domains, scopes): a chain, a cycle, a triangle with a pendant, a 2x3 grid, a 4-clique
            (["a", "b", "c", "d"], [2, 3, 2, 2], [("a", "b"), ("b", "c"), ("c", "d"), ("a",)]),
            (["a", "b", "c", "d"], [2, 2, 3, 2], [("a", "b"), ("b", "c"), ("c", "d"), ("d", "a")]),
            (["a", "b", "c", "d"], [3, 2, 2, 2], [("a", "b", "c"), ("c", "d"), ("b",)]),
            (["g0", "g1", "g2", "g3", "g4", "g5"], [2] * 6,
             [("g0", "g1"), ("g1", "g2"), ("g3", "g4"), ("g4", "g5"), ("g0", "g3"), ("g1", "g4"), ("g2", "g5")]),
            (["a", "b", "c", "d"], [2, 2, 2, 2],
             [("a", "b"), ("a", "c"), ("a", "d"), ("b", "c"), ("b", "d"), ("c", "d")]),
        ]
        for seed in range(3):
            for names, doms, scopes in shapes:
                V, F, O = random_graph(rng, names, doms, scopes, zero=0.1 + 0.15 * seed)
                queries = [[names[0]], [names[-1]], [names[0], names[-1]], [names[1], names[0]], names[:3]]
                evidences = [None, {names[1]: 1}, {names[0]: 0, names[-1]: 1}]
                self.check_graph(V, F, O, queries, evidences)
        # conditions of several variables
        V, F, O = random_graph(rng, ["a", "b", "c", "d"], [2, 3, 2, 2], [("a", "b"), ("b", "c"), ("c", "d"), ("a", "d")])
        G, orc = mf.graph(V, F), Oracle(V, O)
        c = both(G, ("a", "b", "d"), {"c": 1}, "sum", "conditional", given=2)[0]
        self.assertEqual((c.variables, c.given), (("a",), ("b", "d")))
        self.assertEqual(c.table, orc.conditional(["a"], ["b", "d"], {"c": 1}))

    def test_prob002_marginals_of_a_joint_are_not_products_of_single_marginals(self):
        # x and y are strongly correlated through z: P(x, y) != P(x) P(y); the joint marginal is computed from the
        # factor graph (elimination of z), equal to the enumeration, and differs from the product of the marginals
        V = {"x": [0, 1], "y": [0, 1], "z": [0, 1]}
        tx = [[9, 1], [1, 9]]                                  # x copies z
        ty = [[8, 2], [Fraction(1, 2), 7]]                     # y copies z
        F = [mf.table(["z", "x"], tx), mf.table(["z", "y"], ty), mf.table(["z"], [1, 3])]
        O = [(("z", "x"), table_dict([V["z"], V["x"]], tx)), (("z", "y"), table_dict([V["z"], V["y"]], ty)),
             (("z",), table_dict([V["z"]], [1, 3]))]
        G = mf.graph(V, F)
        orc = Oracle(V, O)
        joint = both(G, ("x", "y"), None, "sum", "marginal")[0].table
        px, py = G.marginal(["x"]).table, G.marginal(["y"]).table
        self.assertEqual(joint, orc.marginal(["x", "y"]))
        product = {(a, b): px[(a,)] * py[(b,)] for a in (0, 1) for b in (0, 1)}
        self.assertNotEqual(joint, product)                    # no undeclared independence: the joint is kept
        self.assertEqual(sum(joint.values()), 1)
        # the joint's own factors (z kept) and the reversed order of the kept variables
        self.assertEqual(G.marginal(["y", "x"]).table, orc.marginal(["y", "x"]))
        self.assertEqual(G.marginal(["x", "z", "y"]).table, orc.marginal(["x", "z", "y"]))

    def test_prob002_map_every_tied_maximizer_and_the_k_best(self):
        # constructed ties: a symmetric chain where the maximum is reached 4 times
        V = {"a": [0, 1], "b": [0, 1, 2], "c": [0, 1]}
        tab = [[2, 2, 1], [2, 2, 1]]
        tbc = [[3, 3], [3, 3], [Fraction(1, 3), 0]]
        F = [mf.table(["a", "b"], tab), mf.table(["b", "c"], tbc)]
        O = [(("a", "b"), table_dict([V["a"], V["b"]], tab)), (("b", "c"), table_dict([V["b"], V["c"]], tbc))]
        G = self.check_graph(V, F, O, [["a"], ["b", "c"]], [None, {"c": 1}, {"b": 2, "c": 1}])
        m = G.map()
        self.assertEqual((m.weight, m.count), (6, 8))
        self.assertEqual(m.assignments, [dict(a=a, b=b, c=c) for a in (0, 1) for b in (0, 1) for c in (0, 1)])
        # the k best: ties at the boundary are counted beyond k, the canonical order breaks them
        kb = G.kbest(3)
        self.assertEqual([w for _, w, _ in kb.best], [6, 6, 6])
        self.assertEqual(kb.tied, 5)
        self.assertEqual([a for a, _, _ in kb.best], [dict(a=0, b=0, c=0), dict(a=0, b=0, c=1), dict(a=0, b=1, c=0)])
        # zero-probability evidence: the conditionals are undefined (the request's own status)
        z = G.map({"b": 2, "c": 1})
        self.assertEqual(z.status, -6)
        with self.assertRaises(DomainError):
            z.require()
        self.assertEqual(G.partition({"b": 2, "c": 1}).Z_e, 0)
        # the output budget counts the ties exactly before enumerating them (limit and one past)
        need = G.query(mode="max").plan.chosen["need"]["output"]       # the plan's values (5 per request)
        ok = G.map(budget=Budget(output=need + 8))
        self.assertEqual(ok.count, 8)
        with self.assertRaises(AlgebraBudgetError) as e:
            G.map(budget=Budget(output=need + 7))
        self.assertEqual((e.exception.resource, e.exception.need), ("output", need + 8))

    def test_prob002_evidence_batches_and_the_retained_normalizer(self):
        rng = np.random.default_rng(7)
        names = ["a", "b", "c", "d", "e"]
        V, F, O = random_graph(rng, names, [2, 3, 2, 2, 3],
                               [("a", "b"), ("b", "c"), ("c", "d"), ("d", "e"), ("a", "e"), ("b", "d")], zero=0.1)
        G = mf.graph(V, F)
        orc = Oracle(V, O)
        batch = [{"a": a, "e": e} for a in (0, 1) for e in (0, 1, 2)]
        b = both(G, (), batch, "sum", "partition")
        Z = orc.Z()
        self.assertEqual([(r.Z_e, r.probability) for r in b], [(orc.Z(e), orc.Z(e) / Z) for e in batch])
        self.assertEqual(sum(r.probability for r in b), 1)            # the evidence partitions the universe
        self.assertEqual(b.column("probability"), [r.probability for r in b])     # one native call, the same values
        self.assertEqual(b.column("Z_e"), [orc.Z(e) for e in batch])
        # a retained query runs many batches with one plan and program
        q = G.query(["c"], evidence=["a", "e"], requests=len(batch))
        out = q.run(batch)
        self.assertEqual([r.table for r in out], [orc.marginal(["c"], e) if orc.Z(e) else None for e in batch])
        self.assertEqual([r.status for r in out], [0 if orc.Z(e) else -6 for e in batch])
        # the planner's own CUDA candidate (devices=("cuda",)) gives the same exact answers
        dev = G.likelihood(batch, backend="cuda")
        self.assertEqual(dev.backend, "cuda-emulated")
        self.assertEqual([(r.Z_e, r.probability) for r in dev], [(r.Z_e, r.probability) for r in b])
        self.assertEqual(q.run(np.array([[1, 2]], np.uint32))[0].table, orc.marginal(["c"], {"a": 1, "e": 2}))
        self.assertIs(G.normalizer(), G.normalizer())                 # computed once, natively
        self.assertEqual(G.normalizer()[0].Z_e, Z)
        with self.assertRaises(ValueError):
            G.likelihood([{"a": 0}, {"e": 1}])                         # one shape per batch

    # ------------------------------------------------------------------ factors from TENSOR-001 and PROB-001
    def data(self, n=400, seed=11):
        rng = np.random.default_rng(seed)
        x = rng.integers(0, 3, n)
        y = np.where(rng.random(n) < 0.7, x, rng.integers(0, 3, n))           # y depends on x
        z = np.where(rng.random(n) < 0.6, (y >= 1).astype(np.int64), rng.integers(0, 2, n))
        zv = rng.random(n) > 0.1                                               # z is NULL on some rows
        return x, y, z, zv

    def test_prob002_count_tensor_factors_match_duckdb_group_by(self):
        x, y, z, zv = self.data()
        scope = arrays_scope({"x": x, "y": y, "z": z}, valid={"z": zv})
        con = duckdb.connect()
        con.register("u", pd.DataFrame({"x": x, "y": y,
                                        "z": pd.array([int(v) if ok else None for v, ok in zip(z, zv)], dtype="Int64")}))
        T = mt.counts(scope, ["x", "y", "z"], repr="sparse")
        V = {"x": [0, 1, 2], "y": [0, 1, 2], "z": mf.values([0, 1, None])}
        G = mf.graph(V, [mf.tensor(T, ["x", "y", "z"])])
        counts = {(a, b, c): n for a, b, c, n in con.execute(
            "SELECT x, y, z, COUNT(*) FROM u GROUP BY ALL").fetchall()}
        tab = G.table(0)
        self.assertEqual({k: v for k, v in tab.items() if v}, {k: Fraction(n) for k, n in counts.items()})
        N = len(x)
        self.assertEqual(G.partition().Z_e, N)
        for vs in (["x"], ["x", "z"], ["z", "y"]):
            q = ", ".join(vs)
            sql = {tuple(r[:-1]): r[-1] for r in con.execute(f"SELECT {q}, COUNT(*) FROM u GROUP BY ALL").fetchall()}
            m = both(G, vs, None, "sum", "marginal")[0].table
            self.assertEqual({k: v for k, v in m.items() if v}, {k: Fraction(n, N) for k, n in sql.items()})
        # conditionals with the NULL category as evidence: P(x | y, z IS NULL)
        sql = con.execute("SELECT x, y, COUNT(*) FROM u WHERE z IS NULL GROUP BY ALL").fetchall()
        tot = {}
        for a, b, n in sql:
            tot[b] = tot.get(b, 0) + n
        c = G.conditional(["x"], ["y"], evidence={"z": None})
        for (a, b), p in c.table.items():
            n = next((n for aa, bb, n in sql if (aa, bb) == (a, b)), 0)
            self.assertEqual(p, None if not tot.get(b) else Fraction(n, tot[b]))
        # a declared value the data never holds keeps weight 0; an observed value outside the support is refused
        G2 = mf.graph({"x": [0, 1, 2, 3], "y": 3, "z": mf.values([0, 1, None])}, [mf.tensor(T, ["x", "y", "z"])])
        self.assertEqual(G2.marginal(["x"]).table[(3,)], 0)
        with self.assertRaises(DomainError) as e:
            mf.graph({"x": [0, 1], "y": 3, "z": mf.values([0, 1, None])}, [mf.tensor(T, ["x", "y", "z"])])
        self.assertEqual((e.exception.reason, e.exception.variable, e.exception.value), ("value", "x", 2))
        with self.assertRaises(DomainError) as e:
            mf.graph({"x": 3, "y": 3, "z": [0, 1]}, [mf.tensor(T, ["x", "y", "z"])])      # NULL not declared
        self.assertEqual((e.exception.reason, e.exception.variable, e.exception.value), ("value", "z", None))
        # the dense and the sparse forms of the same counts give the same graph; a grouped tensor's group g
        Td = mt.counts(scope, ["x", "y", "z"], repr="dense")
        self.assertEqual(mf.graph(V, [mf.tensor(Td, ["x", "y", "z"])]).info["digest"], G.info["digest"])
        g = (x % 2).astype(np.uint32)
        sg = arrays_scope({"x": x, "y": y}, groups=g, ngroups=2)
        Tg = mt.counts(sg, ["x", "y"])
        for grp in (0, 1):
            Gg = mf.graph({"x": 3, "y": 3}, [mf.tensor(Tg, ["x", "y"], group=grp)])
            want = con.execute(f"SELECT x, y, COUNT(*) FROM u WHERE x % 2 = {grp} GROUP BY ALL").fetchall()
            self.assertEqual({k: v for k, v in Gg.table(0).items() if v}, {(a, b): Fraction(n) for a, b, n in want})
        with self.assertRaises(ValueError):
            mf.graph({"x": 3, "y": 3}, [mf.tensor(Tg, ["x", "y"], group=2)])
        # a GENERIC tensor (arrays) maps by index, its dimensions the variables' domains
        Ta = mt.dense([[1, 2, 3], [4, 5, 6]])
        Ga = mf.graph({"a": [10, 20], "b": 3}, [mf.tensor(Ta, ["a", "b"])])
        self.assertEqual(Ga.table(0)[(20, 1)], 5)
        self.assertEqual(Ga.marginal(["a"]).table, {(10,): Fraction(6, 21), (20,): Fraction(15, 21)})
        with self.assertRaises(DomainError) as e:
            mf.graph({"a": 2, "b": 4}, [mf.tensor(Ta, ["a", "b"])])
        self.assertEqual(e.exception.reason, "dims")
        con.close()

    def test_prob002_linked_data_factors_need_a_gen002_declaration(self):
        x, y, z, zv = self.data(300, 5)
        scope = arrays_scope({"x": x, "y": y, "z": z})
        Txy, Tyz, Ty = mt.counts(scope, ["x", "y"]), mt.counts(scope, ["y", "z"]), mt.counts(scope, ["y"])
        V = {"x": 3, "y": 3, "z": 2}
        with self.assertRaises(DomainError) as e:
            mf.graph(V, [mf.tensor(Txy, ["x", "y"]), mf.tensor(Tyz, ["y", "z"])])
        self.assertEqual((e.exception.reason, e.exception.factors), ("needs_declaration", (0, 1)))
        other = arrays_scope({"x": x[:10]})
        with self.assertRaises(DomainError) as e:
            mf.graph(V, [mf.tensor(Txy, ["x", "y"]), mf.tensor(Tyz, ["y", "z"])], declaration=mc.independent(other))
        self.assertEqual(e.exception.reason, "not_covered")
        # declared: the decomposable model n(x, y) n(y, z) / n(y) (x and z independent given y)
        ny = {(b,): Fraction(1, int((y == b).sum())) for b in range(3)}
        G = mf.graph(V, [mf.tensor(Txy, ["x", "y"]), mf.tensor(Tyz, ["y", "z"]), mf.table(["y"], ny)],
                     declaration=mc.independent(scope))
        self.assertTrue(G.info["declared"])
        self.assertEqual(G.info["linked"], 2)
        con = duckdb.connect()
        con.register("t", pd.DataFrame({"x": x, "y": y, "z": z}))
        rows = con.execute("SELECT a.x, b.z, a.y, a.n, b.n, c.n FROM (SELECT x, y, COUNT(*) n FROM t GROUP BY ALL) a "
                           "JOIN (SELECT y, z, COUNT(*) n FROM t GROUP BY ALL) b ON a.y = b.y "
                           "JOIN (SELECT y, COUNT(*) n FROM t GROUP BY ALL) c ON a.y = c.y").fetchall()
        want = {}
        for a, c, b, nxy, nyz, nyy in rows:
            want[(a, c)] = want.get((a, c), Fraction(0)) + Fraction(nxy * nyz, nyy)
        N = len(x)
        self.assertEqual(G.partition().Z_e, N)
        m = both(G, ("x", "z"), None, "sum", "marginal")[0].table
        self.assertEqual({k: v for k, v in m.items() if v}, {k: v / N for k, v in want.items()})
        # the joint of the same rows differs from the declared factorization (the declaration is the caller's)
        Txyz = mt.counts(scope, ["x", "y", "z"])
        J = mf.graph(V, [mf.tensor(Txyz, ["x", "y", "z"])]).marginal(["x", "z"]).table
        self.assertNotEqual(J, m)
        # factors of different sources multiply without a declaration (their Cartesian product)
        s2 = arrays_scope({"z": z[::-1].copy()})
        G3 = mf.graph(V, [mf.tensor(Txy, ["x", "y"]), mf.tensor(mt.counts(s2, ["z"]), ["z"])])
        self.assertFalse(G3.info["declared"])
        self.assertEqual(G3.partition().Z_e, N * N)
        con.close()

    def test_prob002_prob001_models_and_statistics_as_factors(self):
        x, y, z, zv = self.data(200, 3)
        scope = arrays_scope({"x": x, "y": y})
        S = mp.support([(a, b) for a in range(3) for b in range(3)])
        st = mp.stats(md.histogram(scope, ["x", "y"]), S)
        M = mp.categorical(S)
        w = [Fraction(i + 1, 3) for i in range(9)]
        V = {"x": 3, "y": 3}
        G = mf.graph(V, [mf.categorical(M, ["x", "y"], w)])
        ev = M.evaluate(st, [w])
        self.assertEqual(G.partition().Z_e, ev[0]["Z_value"])               # PROB-001's exact normalizer
        self.assertEqual(G.marginal(["x", "y"]).table, {p: w[i] / sum(w) for i, p in enumerate(S.points)})
        # exact probabilities (summing to 1) and their refusal
        pr = [Fraction(1, 9)] * 9
        Gp = mf.graph(V, [mf.categorical(M, ["x", "y"], pr, mode="probabilities")])
        self.assertEqual(Gp.partition().Z_e, 1)
        with self.assertRaises(DomainError):
            mf.graph(V, [mf.categorical(M, ["x", "y"], [Fraction(1, 10)] * 9, mode="probabilities")])
        # the statistics' masses (the empirical joint) as a factor; a point of the model outside the variables
        Gs = mf.graph(V, [mf.stats(st, ["x", "y"])])
        cnt = st.counts(0)
        self.assertEqual({k: v for k, v in Gs.table(0).items() if v}, {k: Fraction(v) for k, v in cnt.items() if v})
        self.assertEqual(Gs.marginal(["y"]).table, {(b,): Fraction(int((y == b).sum()), len(y)) for b in range(3)})
        with self.assertRaises(DomainError) as e:
            mf.graph({"x": [0, 1], "y": 3}, [mf.categorical(M, ["x", "y"], w)])
        self.assertEqual(e.exception.reason, "value")
        # a support point of zero weight outside the variables' values is no mass: admitted
        w0 = [0 if p[0] == 2 else 1 for p in S.points]
        self.assertEqual(mf.graph({"x": [0, 1], "y": 3}, [mf.categorical(M, ["x", "y"], w0)]).partition().Z_e, 6)
        # exponential families are enclosures, not exact tables
        E = mp.exponential(S, features=lambda p: (p[0],))
        with self.assertRaises(UnsupportedError):
            mf.graph(V, [mf.categorical(E, ["x", "y"], w)])
        # statistics and data tensors of one source are linked like any data factors
        with self.assertRaises(DomainError) as e:
            mf.graph(V, [mf.stats(st, ["x", "y"]), mf.tensor(mt.counts(scope, ["x"]), ["x"])])
        self.assertEqual(e.exception.reason, "needs_declaration")

    def test_prob002_count002_and_tensor002_backends_agree(self):
        rng = np.random.default_rng(99)
        names = [f"v{i}" for i in range(5)]
        V, F, O = random_graph(rng, names, [2] * 5,
                               [("v0", "v1"), ("v1", "v2"), ("v2", "v3"), ("v3", "v4"), ("v4", "v0"), ("v2",)],
                               zero=0.15)
        G = mf.graph(V, F)
        orc = Oracle(V, O)
        for ev in (None, {"v2": 1}, {"v0": 0, "v3": 1}):
            want = orc.Z(ev)
            got = {m: G.count(ev, method=m) for m in ("symcount", "brute", "diagram", "tensor")}
            self.assertEqual(got, {m: want for m in got}, ev)
            self.assertEqual(G.partition(ev).Z_e, want)
        # COUNT-002 directly on the same tables (its own elimination and brute force)
        fs = ms.Factors([2] * 5, [(tuple(names.index(v) for v in sc),
                                   np.array([tab[k] for k in itertools.product(*[V[v] for v in sc])],
                                            dtype=object).reshape([2] * len(sc)).tolist()) for sc, tab in O])
        self.assertEqual(fs.count().value, orc.Z())
        self.assertEqual(fs.brute().value, orc.Z())
        # the diagram backend is for binary variables; TENSOR-002 for networks of at most 8 tables
        G3 = mf.graph({"a": 3}, [mf.table(["a"], [1, 2, 3])])
        with self.assertRaises(UnsupportedError):
            G3.count(method="diagram")
        self.assertEqual(G3.count(method="tensor"), 6)

    # ------------------------------------------------------------------ budgets govern execution
    def grid(self, rows, cols, rng, d=2):
        names = [f"g{r}{c}" for r in range(rows) for c in range(cols)]
        scopes = [(f"g{r}{c}", f"g{r}{c + 1}") for r in range(rows) for c in range(cols - 1)]
        scopes += [(f"g{r}{c}", f"g{r + 1}{c}") for r in range(rows - 1) for c in range(cols)]
        return random_graph(rng, names, [d] * len(names), scopes, zero=0.05)

    def test_prob002_treewidth_and_intermediate_support_budgets_govern_execution(self):
        rng = np.random.default_rng(3)
        V, F, O = self.grid(3, 3, rng)                     # a 3 x 3 grid: treewidth 3
        G = mf.graph(V, F)
        orc = Oracle(V, O)
        p = G.plan(budget=Budget(degree=3))
        self.assertEqual(p.chosen["width"], 3)
        self.assertTrue(p.info["complete"])                # the exact DP considered every order
        # at the limit the cheapest admissible order runs; one past (width 2) no order exists: refused, proven
        r = G.partition(budget=Budget(degree=3))
        self.assertEqual(r.Z_e, orc.Z())
        with self.assertRaises(AlgebraBudgetError) as e:
            G.partition(budget=Budget(degree=2))
        self.assertEqual((e.exception.resource, e.exception.need, e.exception.limit), ("degree", 3, 2))
        self.assertTrue(e.exception.every_order)
        self.assertEqual(e.exception.minima, dict(width=3, joint=16, support=8, exact=True))
        # the largest joint table (tensor) and the largest intermediate table (support): at the limit, one past
        self.assertEqual(G.partition(budget=Budget(tensor=16)).Z_e, orc.Z())
        with self.assertRaises(AlgebraBudgetError) as e:
            G.partition(budget=Budget(tensor=15))
        self.assertEqual((e.exception.resource, e.exception.every_order), ("tensor", True))
        self.assertEqual(G.partition(budget=Budget(support=8)).Z_e, orc.Z())
        with self.assertRaises(AlgebraBudgetError) as e:
            G.partition(budget=Budget(support=7))
        self.assertTrue(e.exception.every_order)
        # every candidate carries its exact needs and costs; the chosen one is the cheapest admissible
        cands = G.plan(devices=("cpu", "cuda")).candidates
        adm = [c for c in cands if c["status"] != "rejected"]
        ch = [c for c in cands if c["status"] == "chosen"][0]
        self.assertEqual(ch["total"], min(c["total"] for c in adm))
        for c in cands:
            self.assertEqual(c["need"]["degree"], c["width"])
            self.assertEqual(c["need"]["tensor"], c["joint"])
            self.assertEqual(c["need"]["support"], c["support"])

    def test_prob002_exact_minima_and_refusals_match_every_order(self):
        # tiny random graphs: the plan's exact minima (its subset DP) against a brute force over every order, and
        # "no order fits" exactly when the brute force finds none (limit and one past for each per-step budget)
        rng = np.random.default_rng(31)
        for trial in range(6):
            n = 6 if trial < 3 else 7
            names = [f"v{i}" for i in range(n)]
            doms = [int(rng.integers(2, 4)) for _ in names]
            edges = [(names[i], names[j]) for i in range(n) for j in range(i + 1, n) if rng.random() < 0.45]
            edges = edges or [(names[0], names[1])]
            V, F, O = random_graph(rng, names, doms, edges, zero=0.0)
            G = mf.graph(V, F)
            keep = [names[-1]] if trial % 2 else []
            elim = [v for v in names if v not in keep]
            dom = dict(zip(names, doms))
            metrics = [order_metrics(edges, dom, elim, keep, perm) for perm in itertools.permutations(elim)]
            mins = tuple(min(m[i] for m in metrics) for i in range(3))
            for i, res in enumerate(("degree", "tensor", "support")):
                ok = G.plan(keep, budget=Budget(**{res: mins[i]}))
                self.assertEqual(ok.info["complete"], True)
                self.assertLessEqual([ok.chosen["width"], ok.chosen["joint"], ok.chosen["support"]][i], mins[i])
                with self.assertRaises(AlgebraBudgetError) as e:
                    G.plan(keep, budget=Budget(**{res: mins[i] - 1}))
                self.assertTrue(e.exception.every_order)
                self.assertEqual(e.exception.minima, dict(width=mins[0], joint=mins[1], support=mins[2], exact=True))
            # a joint budget: admitted iff some order satisfies every limit at once
            W, T, S = mins[0] + 1, mins[1], mins[2]
            fits = any(m[0] <= W and m[1] <= T and m[2] <= S for m in metrics)
            if fits:
                p = G.plan(keep, budget=Budget(degree=W, tensor=T, support=S))
                c = p.chosen
                self.assertTrue(c["width"] <= W and c["joint"] <= T and c["support"] <= S)
            else:
                with self.assertRaises(AlgebraBudgetError) as e:
                    G.plan(keep, budget=Budget(degree=W, tensor=T, support=S))
                self.assertTrue(e.exception.every_order)

    def test_prob002_the_cheapest_admissible_order_runs_and_a_force_is_never_replaced(self):
        # a star: eliminating the centre first joins every leaf (a joint of 2^(n+1) cells); leaves first stays small
        n = 6
        names = ["c"] + [f"l{i}" for i in range(n)]
        rng = np.random.default_rng(4)
        V, F, O = random_graph(rng, names, [2] * (n + 1), [("c", f"l{i}") for i in range(n)], zero=0.0)
        G = mf.graph(V, F)
        orc = Oracle(V, O)
        bad = names                                       # the natural order eliminates the centre first
        p = G.plan(order=bad, budget=Budget(tensor=4))
        cands = p.candidates
        given = [c for c in cands if c["order"] == tuple(bad)][0]
        self.assertEqual(given["status"], "rejected")     # over the tensor budget
        self.assertEqual(given["budget"]["resource"], "tensor")
        self.assertEqual(given["joint"], 2 ** (n + 1))
        self.assertLessEqual(p.chosen["joint"], 4)
        self.assertEqual(G.partition(order=bad, budget=Budget(tensor=4)).Z_e, orc.Z())
        with self.assertRaises(AlgebraBudgetError) as e:
            G.partition(order=bad, force=given["index"], budget=Budget(tensor=4))
        self.assertEqual(e.exception.resource, "tensor")
        self.assertFalse(e.exception.every_order)         # another order fits: the forced one is refused alone
        self.assertEqual(G.partition(order=bad, force=given["index"]).Z_e, orc.Z())   # without the budget, it runs
        # without the DP (a large star) the heuristics still find the leaves-first order; a refusal is not proven
        n = 20
        names = ["c"] + [f"l{i}" for i in range(n)]
        V, F, O = random_graph(rng, names, [2] * (n + 1), [("c", f"l{i}") for i in range(n)], zero=0.0)
        G = mf.graph(V, F)
        p = G.plan(budget=Budget(tensor=4))
        self.assertFalse(p.info["complete"])
        self.assertEqual(p.chosen["joint"], 4)

    def test_prob002_budget_refusals_at_the_limit_and_one_past(self):
        V = {"a": 2, "b": 3}
        F = [mf.table(["a", "b"], [[1, 2, 3], [4, 5, 6]])]
        # the graph: support = table entries, tensor = the largest table
        mf.graph(V, F, budget=Budget(support=6, tensor=6))
        with self.assertRaises(AlgebraBudgetError) as e:
            mf.graph(V, F, budget=Budget(support=5))
        self.assertEqual((e.exception.resource, e.exception.need), ("support", 6))
        G = mf.graph(V, F)
        nbytes = G.info["bytes"]                          # the object's exact bytes: limit and one past
        mf.graph(V, F, budget=Budget(bytes=nbytes))
        with self.assertRaises(AlgebraBudgetError) as e:
            mf.graph(V, F, budget=Budget(bytes=nbytes - 1))
        self.assertEqual((e.exception.resource, e.exception.need), ("bytes", nbytes))
        # the elimination's states (the arena of messages for R requests): limit and one past
        need = G.plan(["a"], requests=3).chosen["need"]
        G.likelihood([{"a": 0}, {"a": 1}, {"a": 0}], budget=Budget(states=need["states"]))
        with self.assertRaises(AlgebraBudgetError) as e:
            G.query(["a"], requests=3, budget=Budget(states=need["states"] - 1))
        self.assertEqual(e.exception.resource, "states")
        # the k best: the lists' support (assignments held per cell)
        kb = G.kbest(2, budget=Budget(support=6))
        self.assertEqual([w for _, w, _ in kb.best], [6, 5])
        with self.assertRaises(AlgebraBudgetError):
            G.kbest(2, budget=Budget(output=5))           # 2 + 2 k values per request
        # the plan's search: iterations (orders evaluated); the DP is skipped, not refused, past it
        V2, F2, O2 = self.grid(2, 3, np.random.default_rng(8))
        G2 = mf.graph(V2, F2)
        p = G2.plan(budget=Budget(iterations=8))
        self.assertFalse(p.info["complete"])
        self.assertEqual(G2.partition(budget=Budget(iterations=8)).Z_e, Oracle(V2, O2).Z())
        with self.assertRaises(AlgebraBudgetError) as e:
            G2.plan(budget=Budget(iterations=0))
        self.assertEqual(e.exception.resource, "iterations")

    # ------------------------------------------------------------------ larger instances on both paths
    def test_prob002_bounded_larger_models_cpu_equals_cuda(self):
        rng = np.random.default_rng(12)
        V, F, O = self.grid(3, 4, rng, d=2)
        G = mf.graph(V, F)
        orc = Oracle(V, O)
        b = both(G, ("g00", "g23"), {"g11": 1}, "sum", "marginal")
        self.assertEqual(b[0].table, orc.marginal(["g00", "g23"], {"g11": 1}))
        m = both(G, (), {"g11": 1}, "max", "max")[0]
        ranked = orc.ranked({"g11": 1})
        self.assertEqual(m.weight, ranked[0][0])
        # a chain of 40 three-valued variables: exact rationals far past 64 bits (L limbs), equal on both paths
        names = [f"c{i}" for i in range(40)]
        V, F, O = random_graph(rng, names, [3] * 40, [(names[i], names[i + 1]) for i in range(39)], zero=0.0,
                               dens=(7, 11, 13))
        G = mf.graph(V, F)
        self.assertGreater(G.info["limbs"], 2)
        self.assertLessEqual(G.info["limbs"], 16)
        r = both(G, ("c0", "c39"), {"c20": 2}, "sum", "marginal")[0]
        self.assertEqual(sum(r.table.values()), 1)
        self.assertEqual(G.partition().Z_e, G.count(method="symcount"))
        self.assertEqual(G.partition({"c20": 2}).Z_e, G.count({"c20": 2}, method="symcount"))
        # past the device's 16 limbs (1024-bit values) a CUDA candidate is rejected; the CPU still runs
        big = (1 << 62) + 1
        V = {f"w{i}": 2 for i in range(24)}
        F = [mf.table([f"w{i}", f"w{i + 1}"], [[big, 1], [1, big]]) for i in range(23)]
        G = mf.graph(V, F, budget=Budget())
        self.assertGreater(G.info["limbs"], 16)
        p = G.plan(devices=("cpu", "cuda"))
        cuda = [c for c in p.candidates if c["device"] == "cuda"]
        self.assertTrue(cuda and all(c["status"] == "rejected" and not c["device_ok"] for c in cuda))
        self.assertEqual(p.chosen["device"], "cpu")
        with self.assertRaises(UnsupportedError):
            G.partition(backend="cuda")
        self.assertEqual(G.partition().Z_e, G.count(method="symcount"))

    # ------------------------------------------------------------------ freshness and refusals of the domain
    def test_prob002_stale_tables_and_input_refusals(self):
        frame = pd.DataFrame({"x": pd.Series([0, 1, 1, 2, 0, 1], dtype=object),
                              "y": pd.Series([1, 1, 0, 0, 1, 0], dtype=object)})
        db = torbles.from_ddl("CREATE TABLE t (x BIGINT, y BIGINT)", {"t": frame}, keep_plain=True)
        t = db["t"]
        scope = mbk.Scope.from_table(t, ["x", "y"])
        T = mt.counts(scope, ["x", "y"])
        G = mf.graph({"x": 3, "y": 2}, [mf.tensor(T, ["x", "y"])])
        self.assertEqual(G.partition().Z_e, 6)
        self.assertEqual(G.marginal(["x"]).table, {(0,): Fraction(1, 3), (1,): Fraction(1, 2), (2,): Fraction(1, 6)})
        t.flip(0, 0)                                       # the table changes: every later query is refused
        with self.assertRaises(StaleStateError):
            G.partition()
        # inputs
        with self.assertRaises(TypeError):
            mf.graph({"a": 2}, [mf.table(["a"], [0.5, 1.0])])          # exact rationals only
        with self.assertRaises(DomainError) as e:
            mf.graph({"a": 2}, [mf.table(["a"], [1, -1])])
        self.assertEqual(e.exception.reason, "negative")
        with self.assertRaises(ValueError):
            mf.graph({"a": 2}, [mf.table(["a", "a"], [[1, 1], [1, 1]])])
        with self.assertRaises(ValueError):
            mf.graph({"a": [1, 1]}, [])
        G = mf.graph({"a": 2, "b": 2}, [mf.table(["a", "b"], [[1, 2], [3, 4]])])
        with self.assertRaises(ValueError):
            G.marginal(["a"], evidence={"a": 0})          # a kept variable cannot be observed
        with self.assertRaises(ValueError):
            G.partition({"a": 5})                         # not a declared value
        with self.assertRaises(ValueError):
            G.plan(order=["a"])                           # not a permutation of the eliminated variables


if __name__ == "__main__":
    unittest.main()
