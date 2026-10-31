# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""PROV-002: explanation specialization and deletion propagation (torbles/math_explain.c, math_explain_gpu.cu,
math_explain.py; probabilities through COUNT-002's decision diagrams).

Oracles, independent of the implementation: DuckDB re-executing each query after the deletions (complete reruns: bag
multiplicities by GROUP BY ALL / COUNT(*), duplicated rows for multiplicities), a Python evaluation of the same queries
over any set of present source tuples (bag joins with SQL NULL semantics) used to enumerate every possible world:
exact answer probabilities as sums of Fraction world probabilities, minimal sufficient sets, counterfactual and
relevant sources and why-not restorations by subset enumeration; and the naive product of per-derivation
probabilities, which the exact answers must differ from on correlated lineages. Every propagation is also run on the
CUDA path (emulated here, TORBLES_EMU_SCRAMBLE=1 scrambles the thread order) and compared byte for byte with the CPU.
Seeds are fixed.
"""
import itertools
import random
import subprocess
import sys
import unittest
from fractions import Fraction as Fr

import duckdb
import numpy as np
import pandas as pd

from torbles import math_explain as mx, math_prov as mp
from torbles.math_budgets import Budget
from torbles.mathematics import AlgebraBudgetError

R_ROWS = [(1, 10), (1, 20), (2, 30), (None, 40), (1, 10)]          # a duplicate row and a NULL key
S_ROWS = [(1, 5), (2, 6), (2, 7), (3, 8)]
T_ROWS = [(5, 100), (6, 100), (6, 200), (7, 300)]
R2_ROWS = [(1, 20), (5, 50)]
TABLES = {"R": (R_ROWS, ["k", "a"]), "S": (S_ROWS, ["k", "b"]), "T": (T_ROWS, ["b", "c"]), "R2": (R2_ROWS, ["k", "a"])}
LABELS = [(n, i) for n in ("R", "S", "T", "R2") for i in range(len(TABLES[n][0]))]


def cols(rows, names):
    return {n: [row[c] for row in rows] for c, n in enumerate(names)}


# ---------------------------------------------------------------------- the Python oracle: queries over present tuples
def present_rows(name, present):
    return [row for i, row in enumerate(TABLES[name][0]) if (name, i) in present]


def bag_add(bag, key, n=1):
    bag[key] = bag.get(key, 0) + n


def q_join(present):                                    # R JOIN S ON k
    out = {}
    for r in present_rows("R", present):
        for s in present_rows("S", present):
            if r[0] is not None and r[0] == s[0]:
                bag_add(out, r + s)
    return out


def q_group_c(present):                                 # R JOIN S ON k JOIN T ON b GROUP BY c
    out = {}
    for (rk, ra, sk, sb), n in q_join(present).items():
        for t in present_rows("T", present):
            if sb == t[0]:
                bag_add(out, (t[1],), n)
    return out


def q_union(present):                                   # R UNION ALL R2
    out = {}
    for name in ("R", "R2"):
        for row in present_rows(name, present):
            bag_add(out, row)
    return out


def q_filtered(present):                                # (R WHERE a > 15) JOIN S ON k GROUP BY b
    out = {}
    for (rk, ra, sk, sb), n in q_join(present).items():
        if ra > 15:
            bag_add(out, (sb,), n)
    return out


def q_global(present):                                  # SELECT COUNT(*) FROM R JOIN S: the global group
    return {(): sum(q_join(present).values())}


QUERIES = [q_join, q_group_c, q_union, q_filtered, q_global]


def build(st):
    R = st.source("R", cols(R_ROWS, ["k", "a"]))
    S = st.source("S", cols(S_ROWS, ["k", "b"]))
    T = st.source("T", cols(T_ROWS, ["b", "c"]))
    R2 = st.source("R2", cols(R2_ROWS, ["k", "a"]))
    J = R.join(S, on=[("k", "k")])
    G = J.join(T, on=[("b", "b")]).group(["c"])
    U = R.union(R2)
    F = R.where(("a", ">", 15)).join(S, on=[("k", "k")]).group(["b"])
    GL = J.group([])
    return [J, G, U, F, GL]


def con_for():
    con = duckdb.connect()
    for name, (rows, names) in TABLES.items():
        df = pd.DataFrame({"rid": list(range(len(rows))),
                           **{n: pd.Series([row[c] for row in rows], dtype="Int64") for c, n in enumerate(names)}})
        con.register(name + "_all", df)
    return con


SQL = ["SELECT r.k, r.a, s.k, s.b, COUNT(*) FROM R r JOIN S s ON r.k = s.k GROUP BY ALL",
       "SELECT t.c, COUNT(*) FROM R r JOIN S s ON r.k = s.k JOIN T t ON s.b = t.b GROUP BY t.c",
       "SELECT k, a, COUNT(*) FROM (SELECT k, a FROM R UNION ALL SELECT k, a FROM R2) GROUP BY ALL",
       "SELECT s.b, COUNT(*) FROM R r JOIN S s ON r.k = s.k WHERE r.a > 15 GROUP BY s.b",
       "SELECT COUNT(*) FROM R r JOIN S s ON r.k = s.k"]


def views(con, deleted=(), copies=None):
    """the tables without the deleted rows (and with row copies {label: n} for multiplicities)"""
    copies = copies or {}
    for name in TABLES:
        ids = [i for (src, i) in deleted if src == name] or [-1]
        extra = " ".join(f"UNION ALL SELECT * FROM {name}_all WHERE rid = {i}" for (src, i), n in copies.items()
                         if src == name for _ in range(n - 1))
        con.execute(f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM {name}_all WHERE rid NOT IN "
                    f"({', '.join(map(str, ids))}) {extra}")


def sqlbag(con, sql):
    out = {}
    for row in con.execute(sql).fetchall():
        key = tuple(None if v is None else int(v) for v in row[:-1])
        out[key] = out.get(key, 0) + int(row[-1])
    return out


def world_probability(probs, labels, exists):
    """sum over every world (subset of the labels) of its probability where exists(world)"""
    total = Fr(0)
    for bits in range(1 << len(labels)):
        world = {labels[i] for i in range(len(labels)) if (bits >> i) & 1}
        if exists(world):
            p = Fr(1)
            for i, lab in enumerate(labels):
                p *= probs[lab] if (bits >> i) & 1 else 1 - probs[lab]
            total += p
    return total


class Explanations(unittest.TestCase):
    # ------------------------------------------------------------------ acceptance 1: deletions against complete reruns
    def test_prov002_deleting_each_tiny_source_matches_complete_reruns(self):
        st = mp.Store()
        rels = build(st)
        ex = mx.Explainer(st, rels)
        con = con_for()
        views(con)
        self.assertEqual([sqlbag(con, q) for q in SQL], [q(set(LABELS)) for q in QUERIES])   # the oracles agree
        answers = [(k, tp) for k, r in enumerate(rels) for tp in r.tuples()]
        for k, r in enumerate(rels):                          # the baseline is the query's bag
            want = sqlbag(con, SQL[k])
            self.assertEqual(dict(zip(r.tuples(), ex.baseline[0][ex.splits[k]:ex.splits[k + 1]])), want)
        deletions = [[lab] for lab in LABELS] + [[("R", 0), ("R", 4)], [("S", 1), ("T", 1)], [("R", i) for i in range(5)],
                                                 [("S", 1), ("S", 2)], [("T", 0), ("R2", 0), ("R", 1)], []]
        cpu = ex.delete(deletions)
        cpu_raw = ex._last
        gpu = ex.delete(deletions, backend="cuda")
        self.assertEqual(gpu.backend, "cuda-emulated")
        self.assertTrue(all(np.array_equal(a, b) if isinstance(a, np.ndarray) else a == b
                            for a, b in zip(cpu_raw, ex._last)))         # the same bytes
        full = st.evaluate(rels, "count", deletions=deletions)          # PROV-001: the whole DAG re-evaluated
        fullb = st.evaluate(rels, "bool", deletions=deletions)
        self.assertEqual((cpu.counts, cpu.exists), (full.values, fullb.values))
        cone = ex.info["nodes"]
        base_bags = [q(set(LABELS)) for q in QUERIES]
        for w, d in enumerate(deletions):
            views(con, d)
            for k, r in enumerate(rels):
                want = sqlbag(con, SQL[k])
                counts, exists = cpu.of(w, k)
                if k == 4:                                    # the global group: present, COUNT(*) may be 0
                    self.assertEqual(counts, [want[()]], d)
                    continue
                self.assertEqual({tp: c for tp, c in zip(r.tuples(), counts) if c}, want, (k, d))
                self.assertEqual({tp for tp, e in zip(r.tuples(), exists) if e}, set(want), (k, d))
            # the effects are exactly the answers whose rerun multiplicity changed
            rerun = [sqlbag(con, SQL[k]) for k in range(5)]
            changed = {(k, tp) for k, tp in answers if rerun[k].get(tp, 0) != base_bags[k].get(tp, 0)}
            self.assertEqual({(e["relation"], e["values"]) for e in cpu.effects(w)}, changed, d)
            self.assertEqual(cpu.changed[w], len(changed))
            self.assertLessEqual(cpu.affected[w], cone)
        # incremental: deleting one tuple recomputes only the nodes above it
        self.assertLess(max(cpu.affected[:len(LABELS)]), cone)
        self.assertEqual(cpu.affected[-1], 0)
        # a base valuation (some tuples absent), restorations and multiplicities, against reruns
        base = {("R", 2): 0, ("T", 3): 0}
        eb = mx.Explainer(st, rels, base=base)
        views(con, list(base))
        want = sqlbag(con, SQL[1])
        self.assertEqual(dict(zip(rels[1].tuples(), eb.baseline[0][eb.splits[1]:eb.splits[2]])),
                         {tp: want.get(tp, 0) for tp in rels[1].tuples()})
        res = eb.revalue([{("R", 2): 1}, {("R", 2): 1, ("T", 3): 1}, {("S", 1): 3}, {("R", 0): 0, ("R", 2): 2}])
        cases = [([("T", 3)], {}), ([], {}), ([("R", 2), ("T", 3)], {("S", 1): 3}), ([("R", 0), ("T", 3)], {("R", 2): 2})]
        for w, (gone, copies) in enumerate(cases):
            views(con, gone, copies)
            for k, r in enumerate(rels[:4]):
                counts, _ = res.of(w, k)
                self.assertEqual({tp: c for tp, c in zip(r.tuples(), counts) if c}, sqlbag(con, SQL[k]), (w, k))
        self.assertEqual(res.counts, eb.revalue([{("R", 2): 1}, {("R", 2): 1, ("T", 3): 1}, {("S", 1): 3},
                                                 {("R", 0): 0, ("R", 2): 2}], backend="cuda").counts)
        con.close()

    # ------------------------------------------------------------------ acceptance 2: correlated derivations
    def test_prov002_correlated_derivations_are_not_independent_events(self):
        # one tuple shared by two derivations: P = p(r0) P(s0 OR s1), not 1 - (1 - p r0 s0)(1 - p r0 s1)
        st = mp.Store()
        R = st.source("R", {"k": [1], "a": [10]})
        S = st.source("S", {"k": [1, 1], "b": [5, 6]})
        G = R.join(S, on=[("k", "k")]).group(["k"])
        ex = mx.Explainer(st, [G])
        half = {lab: Fr(1, 2) for lab in [("R", 0), ("S", 0), ("S", 1)]}
        exact = ex.probability(0, half)
        worlds = world_probability(half, list(half), lambda w: ("R", 0) in w and (("S", 0) in w or ("S", 1) in w))
        naive = 1 - (1 - Fr(1, 4)) * (1 - Fr(1, 4))           # the derivations treated as independent events
        self.assertEqual((exact.status, exact.value, worlds), ("exact", Fr(3, 8), Fr(3, 8)))
        self.assertNotEqual(exact.value, naive)
        b = ex.probability(0, half, method="bounds")
        self.assertEqual((b.status, b.read_once), ("bounds", False))
        self.assertTrue(b.bounds[0] <= Fr(3, 8) <= b.bounds[1])
        self.assertEqual(ex.probability(0, half, backend="cuda").value, Fr(3, 8))
        # random small queries: every answer's exact probability equals the enumeration of all 2^n worlds
        rng = random.Random(2026)
        for trial in range(12):
            rrows = [(rng.randint(0, 2), rng.randint(0, 3)) for _ in range(5)]
            srows = [(rng.randint(0, 2), rng.randint(0, 2)) for _ in range(4)]
            st = mp.Store()
            R = st.source("R", cols(rrows, ["k", "a"]))
            S = st.source("S", cols(srows, ["k", "b"]))
            J = R.join(S, on=[("k", "k")])
            G, H = J.group(["b"]), J.group(["a", "b"])
            SJ = R.join(R, on=[("k", "k")]).group(["a"])      # a self-join: squared variables
            rels = [G, H, SJ]
            ex = mx.Explainer(st, rels)
            labels = [("R", i) for i in range(5)] + [("S", i) for i in range(4)]
            probs = {lab: Fr(rng.randint(1, 6), 7) for lab in labels}

            def answers(world):
                rr = [r for i, r in enumerate(rrows) if ("R", i) in world]
                ss = [s for i, s in enumerate(srows) if ("S", i) in world]
                pairs = [(r, s) for r in rr for s in ss if r[0] == s[0]]
                return [{(s[1],) for r, s in pairs}, {(r[1], s[1]) for r, s in pairs},
                        {(r[1],) for r in rr for q in rr if r[0] == q[0]}]
            for k, rel in enumerate(rels):
                for i, tp in enumerate(rel.tuples()):
                    p = ex.probability((k, i), probs)
                    want = world_probability(probs, labels, lambda w, k=k, tp=tp: tp in answers(w)[k])
                    self.assertEqual(p.value, want, (trial, k, tp))
                    bd = ex.probability((k, i), probs, method="bounds")
                    self.assertTrue(bd.bounds[0] <= want <= bd.bounds[1])
                    if trial < 3:
                        self.assertEqual(ex.probability((k, i), probs, backend="cuda").value, want)
        # a read-once lineage (sums joined): the certified bounds enclose exact independence tightly
        st = mp.Store()
        R = st.source("R", {"k": [0, 0, 0], "a": [1, 2, 3]})
        S = st.source("S", {"k": [0, 0], "b": [1, 2]})
        RO = R.group(["k"]).join(S.group(["k"]), on=[("k", "k")])
        ex = mx.Explainer(st, [RO])
        probs = {("R", 0): Fr(1, 3), ("R", 1): Fr(1, 5), ("R", 2): Fr(1, 2), ("S", 0): Fr(2, 3), ("S", 1): Fr(1, 4)}
        want = (1 - Fr(2, 3) * Fr(4, 5) * Fr(1, 2)) * (1 - Fr(1, 3) * Fr(3, 4))
        self.assertEqual(ex.probability(0, probs).value, want)
        bd = ex.probability(0, probs, method="bounds")
        self.assertTrue(bd.read_once and bd.bounds[0] <= want <= bd.bounds[1] and bd.bounds[1] - bd.bounds[0] < 1e-12)
        # past the diagram's budget: refused, or certified bounds with the refusal recorded (never an estimate)
        n = 7
        st = mp.Store()
        R = st.source("R", {"k": list(range(n)), "g": [0] * n})
        S = st.source("S", {"k": list(range(n))})
        P = R.join(S, on=[("k", "k")]).group(["g"])          # sum_i r_i s_i: linear under dfs, exponential natural
        ex = mx.Explainer(st, [P])
        probs = {lab: Fr(1, 2) for lab in [("R", i) for i in range(n)] + [("S", i) for i in range(n)]}
        good, bad = ex.probability(0, probs), ex.probability(0, probs, order="natural")
        self.assertEqual(good.value, bad.value)
        self.assertEqual(good.value, 1 - Fr(3, 4) ** n)
        self.assertEqual(good.diagram["nodes"], 2 + 2 * n)
        self.assertEqual(bad.diagram["nodes"], 2 ** (n + 1))
        peak = bad.diagram["peak_nodes"]
        ex.probability(0, probs, order="natural", budget=Budget(circuit=peak))
        with self.assertRaises(AlgebraBudgetError) as ctx:
            ex.probability(0, probs, order="natural", budget=Budget(circuit=peak - 1))
        self.assertEqual((ctx.exception.resource, ctx.exception.need), ("circuit", peak))
        auto = ex.probability(0, probs, order="natural", method="auto", budget=Budget(circuit=peak - 1))
        self.assertEqual((auto.status, auto.value, auto.refusal["resource"]), ("bounds", None, "circuit"))
        self.assertTrue(auto.bounds[0] <= good.value <= auto.bounds[1])
        self.assertEqual(ex.probability(0, probs, method="auto").status, "exact")

    # ------------------------------------------------------------------ explanations against subset enumeration
    def test_prov002_explanations_match_subset_enumeration(self):
        st = mp.Store()
        rels = build(st)
        ex = mx.Explainer(st, rels)
        everything = set(LABELS)
        for k, rel in enumerate(rels[:4]):
            q = QUERIES[k]
            for i, tp in enumerate(rel.tuples()):
                exists = lambda world, q=q, tp=tp: q(world).get(tp, 0) > 0      # noqa: E731
                lin = ex.lineage((k, i)).set
                # lineage: every source tuple of some derivation (its multiplicity changes without it)
                self.assertEqual(lin, {lab for lab in LABELS if q(everything - {lab}).get(tp, 0) < q(everything)[tp]})
                universe = sorted(lin)
                worlds = [set(c) for n in range(len(universe) + 1) for c in itertools.combinations(universe, n)]
                minimal = [w for w in worlds if exists(w) and not any(exists(w - {x}) for x in w)]
                self.assertEqual(set(ex.sufficient((k, i)).sets), {frozenset(w) for w in minimal})
                self.assertEqual(set(ex.sufficient((k, i)).sets), set(rel.witnesses(i).sets))   # PROV-001's witnesses
                self.assertEqual(ex.necessary((k, i)).set, {x for x in lin if not exists(everything - {x})})
                small = ex.smallest((k, i)).set
                self.assertTrue(exists(small) and len(small) == min(len(w) for w in minimal))
                self.assertEqual(ex.relevant((k, i)).set,
                                 {x for x in lin if any(exists(w | {x}) != exists(w - {x}) for w in worlds)})
                self.assertEqual(ex.why_not((k, i)).status, "present")
        # relevant is not lineage: x + x y depends on x alone (y changes the multiplicity, not the existence)
        R1 = rels[0].store.source("Q", {"k": [1, 1], "v": [0, 0]})
        X = R1.group(["k"]).union(R1.join(st.source("Y", {"k": [1]}), on=[("k", "k")]).group(["k"]))
        exy = mx.Explainer(st, [X])
        self.assertEqual(exy.lineage(0).set, {("Q", 0), ("Q", 1), ("Y", 0)})
        self.assertEqual(exy.relevant(0).set, {("Q", 0), ("Q", 1)})
        self.assertEqual(set(exy.sufficient(0).sets), {frozenset({("Q", 0)}), frozenset({("Q", 1)})})
        self.assertEqual(exy.necessary(0).set, set())
        d = exy.delete([[("Y", 0)], [("Q", 0)], [("Q", 0), ("Q", 1)]])
        self.assertEqual((d.counts, d.exists), ([[2], [2], [0]], [[True], [True], [False]]))
        # why-not: a base with absent tuples; the minimal restorations against enumeration
        base = {("S", 0): 0, ("T", 0): 0, ("R", 2): 0, ("R", 1): 0}
        eb = mx.Explainer(st, rels, base=base)
        present = everything - set(base)
        for k, rel in enumerate(rels[:4]):
            q = QUERIES[k]
            for i, tp in enumerate(rel.tuples()):
                exists = lambda world, q=q, tp=tp: q(world).get(tp, 0) > 0      # noqa: E731
                wn = eb.why_not((k, i))
                if exists(present):
                    self.assertEqual((wn.status, wn.sets), ("present", []))
                    self.assertEqual(set(eb.sufficient((k, i)).sets),
                                     {frozenset(w) for w in (set(c) for n in range(len(present) + 1) for c in
                                                             itertools.combinations(sorted(eb.lineage((k, i)).set &
                                                                                           present), n))
                                      if exists(w) and not any(exists(w - {x}) for x in w)})
                    continue
                self.assertEqual(eb.sufficient((k, i)).status, "absent")
                cands = sorted(set(base))
                fixes = [set(c) for n in range(len(cands) + 1) for c in itertools.combinations(cands, n)
                         if exists(present | set(c))]
                minimal = {frozenset(w) for w in fixes if not any(exists(present | (w - {x})) for x in w)}
                self.assertEqual((wn.status, set(wn.sets)), ("found", minimal), (k, tp))
        # repeated child entries: a self-join's (r_i, r_j) and (r_j, r_i) share one product node, so the global group
        # sums ~k^2 entries over fewer nodes; its existence is OR r_i
        sk = mp.Store()
        Rk = sk.source("R", {"k": [1] * 6, "a": list(range(6))})
        SJ = Rk.join(Rk, on=[("k", "k")]).group([])
        ek = mx.Explainer(sk, [SJ])
        labs = [("R", i) for i in range(6)]
        self.assertEqual(ek.lineage(0).set, set(labs))
        self.assertEqual(set(ek.sufficient(0).sets), {frozenset({lab}) for lab in labs})
        self.assertEqual((ek.necessary(0).set, ek.relevant(0).set, len(ek.smallest(0).set)), (set(), set(labs), 1))
        pk = {lab: Fr(i + 1, 9) for i, lab in enumerate(labs)}
        none = Fr(1)
        for lab in labs:
            none *= 1 - pk[lab]
        self.assertEqual(ek.probability(0, pk).value, 1 - none)
        self.assertEqual(world_probability(pk, labs, lambda w: bool(w)), 1 - none)
        self.assertEqual(ek.delete([labs[:5], labs]).counts, [[1], [0]])
        # an answer no recorded derivation can make exist
        st2 = mp.Store()
        E = st2.source("E", {"a": [1, 2]}).where(("a", ">", 5)).group([])
        self.assertEqual(mx.Explainer(st2, [E]).why_not(0).status, "underivable")
        self.assertEqual(mx.Explainer(st2, [E]).sufficient(0).status, "absent")

    # ------------------------------------------------------------------ CUDA: many change sets at once
    def test_prov002_cuda_batched_deletions_equal_cpu(self):
        rng = np.random.default_rng(1001)
        st = mp.Store()
        R = st.source("R", {"k": [int(x) for x in rng.integers(0, 6, 60)], "a": [int(x) for x in rng.integers(0, 4, 60)]})
        S = st.source("S", {"k": [int(x) for x in rng.integers(0, 6, 40)], "b": [int(x) for x in rng.integers(0, 3, 40)]})
        J = R.join(S, on=[("k", "k")]).group(["a", "b"])
        Q = R.join(S, on=[("k", "k")]).group(["a", "b"]).union(R.group(["a"]).join(S.group(["b"]))).group([])
        ex = mx.Explainer(st, [J, Q])
        self.assertGreater(ex.info["nodes"], 100)
        labels = [("R", i) for i in range(60)] + [("S", i) for i in range(40)]
        dels = [[labels[int(i)] for i in rng.choice(100, int(rng.integers(0, 6)), replace=False)] for _ in range(300)]
        a = ex.delete(dels)
        raw = ex._last
        b = ex.delete(dels, backend="cuda")
        self.assertTrue(np.array_equal(raw[0], ex._last[0]) and np.array_equal(raw[1], ex._last[1]) and raw[2] == ex._last[2])
        self.assertEqual((a.counts, a.exists, a.affected), (b.counts, b.exists, b.affected))
        self.assertEqual(a.counts, st.evaluate([J, Q], "count", deletions=dels).values)
        self.assertEqual(a.exists, st.evaluate([J, Q], "bool", deletions=dels).values)
        self.assertLess(sum(a.affected) / len(dels), ex.info["nodes"])
        vals = [{labels[int(i)]: int(rng.integers(0, 4)) for i in rng.choice(100, 8, replace=False)} for _ in range(100)]
        va, vb = ex.revalue(vals), ex.revalue(vals, backend="cuda")
        self.assertEqual((va.counts, va.exists), (vb.counts, vb.exists))
        self.assertEqual(va.counts, st.evaluate([J, Q], "count", valuations=vals, default=1).values)
        # N past 128 bits: that set's status, B still exact; a baseline past 128 bits: B only, every N reported
        sn = mp.Store()
        N = sn.source("N", {"k": [1, 1, 2]})
        X3 = N.join(N, on=[("k", "k")]).join(N, on=[("k", "k")])
        e3 = mx.Explainer(sn, [X3])
        big = (1 << 64) - 1
        c, g = e3.revalue([{("N", 0): big}, {("N", 0): 3}]), e3.revalue([{("N", 0): big}, {("N", 0): 3}], backend="cuda")
        self.assertEqual((c.status, c.counts, c.exists), (g.status, g.counts, g.exists))
        self.assertEqual(c.status, ["overflow", None])
        self.assertIsNone(c.counts[0])
        self.assertEqual(c.exists[0], sn.evaluate(X3, "bool", valuations=[{("N", 0): big}]).values[0])
        self.assertEqual(c.counts[1], sn.evaluate(X3, "count", valuations=[{("N", 0): 3}]).values[0])
        eo = mx.Explainer(sn, [X3], base={("N", 0): big})
        self.assertFalse(eo.info["count_ok"])
        d = eo.delete([[("N", 0)], [("N", 2)]])
        self.assertEqual(d.status, ["overflow", "overflow"])
        self.assertEqual(d.exists, sn.evaluate(X3, "bool", valuations=[{("N", 0): 0}, {("N", 2): 0, ("N", 0): big}]).values)
        self.assertEqual(d.exists, eo.delete([[("N", 0)], [("N", 2)]], backend="cuda").exists)

    def test_prov002_budgets_and_invalid_inputs(self):
        st = mp.Store()
        rels = build(st)
        cone = mx.Explainer(st, rels).info["nodes"]
        mx.Explainer(st, rels, budget=Budget(circuit=cone))
        with self.assertRaises(AlgebraBudgetError) as ctx:
            mx.Explainer(st, rels, budget=Budget(circuit=cone - 1))
        self.assertEqual((ctx.exception.resource, ctx.exception.need), ("circuit", cone))
        ex = mx.Explainer(st, rels)
        n = ex.info["outputs"]
        sets = [[lab] for lab in LABELS[:5]]
        for backend in ("cpu", "cuda"):
            ex.delete(sets, backend=backend, budget=Budget(output=5 * n))
            with self.assertRaises(AlgebraBudgetError) as ctx:
                ex.delete(sets, backend=backend, budget=Budget(output=5 * n - 1))
            self.assertEqual((ctx.exception.resource, ctx.exception.need), ("output", 5 * n))
        need = ex.need(sets, backend="cuda")["bytes"]
        ex.delete(sets, backend="cuda", budget=Budget(bytes=need))
        with self.assertRaises(AlgebraBudgetError) as ctx:
            ex.delete(sets, backend="cuda", budget=Budget(bytes=need - 1))
        self.assertEqual((ctx.exception.resource, ctx.exception.need), ("bytes", need))
        cpu_need = ex.need(sets)["bytes"]
        ex.delete(sets, budget=Budget(bytes=cpu_need))
        with self.assertRaises(AlgebraBudgetError):
            ex.delete(sets, budget=Budget(bytes=cpu_need - 1))
        # the minimal sufficient sets: refused past max_sets (never truncated)
        j = ex.splits[1]                                      # G's first group: four minimal witnesses
        m = len(ex.sufficient(j).sets)
        self.assertGreater(m, 1)
        ex.sufficient(j, max_sets=m)
        with self.assertRaises(AlgebraBudgetError) as ctx:
            ex.sufficient(j, max_sets=m - 1)
        self.assertEqual((ctx.exception.resource, ctx.exception.need, ctx.exception.limit), ("output", m, m - 1))
        with self.assertRaises(ValueError):
            ex.delete([[("Z", 0)]])
        with self.assertRaises(ValueError):
            ex.probability(0, {("R", 0): Fr(3, 2)})
        with self.assertRaises(TypeError):
            ex.probability(0, {("R", 0): 0.5})
        with self.assertRaises(IndexError):
            ex.lineage(10 ** 6)
        with self.assertRaises(ValueError):
            ex.probability(0, method="guess")
        with self.assertRaises(ValueError):
            mx.Explainer(mp.Store(), rels)                    # relations of another store
        code = "import sys, torbles; print('torbles.math_explain' in sys.modules)"
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.split()
        self.assertEqual(out, ["False"])


if __name__ == "__main__":
    unittest.main()
