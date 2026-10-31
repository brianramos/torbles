# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""COUNT-002: symbolic full-domain algebraic model counting (torbles/math_symcount.c, math_symcount_gpu.cu,
math_symcount.py).

Oracles, independent of the implementation: brute-force enumeration in Python of every assignment of the universe
(SQL / Kleene three-valued evaluation of the predicate's node list written here, products of the literal weights and
of the factor entries as Fractions, models sorted in the stated lexicographic orders), COUNT-001's observed counts
over table rows and its enumerated universe for the distinction between the observed snapshot and the assignment
universe. Every native result that has a CUDA path (the level-by-level diagram count, the brute-force enumeration)
is also computed there (emulated with TORBLES_EMU_SCRAMBLE=1, a scrambled thread order) and compared word for word
with the CPU's raw values. Seeds are fixed.
"""
import itertools
import random
import subprocess
import sys
import unittest
from fractions import Fraction as Fr

import numpy as np
import pandas as pd

import torbles
from torbles import math_bank as mbk, math_bool as mb, math_count as mc, math_symcount as ms
from torbles.math_budgets import Budget
from torbles.math_errors import UnsupportedError
from torbles.mathematics import AlgebraBudgetError



def kleene(op, a, b=None):
    if op == "not":
        return None if a is None else 1 - a
    if op == "and":
        if a == 0 or b == 0:
            return 0
        return None if a is None or b is None else 1
    if op == "or":
        if a == 1 or b == 1:
            return 1
        return None if a is None or b is None else 0
    return None if a is None or b is None else int(a != b)            # xor


def evaluate(nodes, assignment):
    """the program's value at an assignment (bit i of the int = variable i): 1, 0 or None"""
    val = []
    for op, a, b in nodes:
        if op == "const":
            val.append({0: 0, 1: 1, 2: None}[a])
        elif op == "var":
            val.append((assignment >> a) & 1)
        elif op == "not":
            val.append(kleene("not", val[a]))
        else:
            val.append(kleene(op, val[a], val[b]))
    return val[-1]


def random_nodes(rng, k, n, unknown=0.0):
    """a random predicate DAG over k variables: n nodes, shared subexpressions, the last node the root; operands are
    mostly among the latest nodes (deep circuits that read many variables), sometimes anywhere (sharing)"""
    nodes = [("var", i, -1) for i in range(k)]

    def pick():
        if rng.random() < 0.3:
            return rng.randrange(len(nodes))
        return len(nodes) - 1 - min(int(rng.expovariate(0.4)), len(nodes) - 1)
    for j in range(n):
        r = rng.random()
        if r < unknown:
            nodes.append(("const", rng.choice([0, 1, 2]), -1))
        elif r < 0.2:
            nodes.append(("not", pick(), -1))
        else:
            a = j if j < k else pick()                          # the first gates read every variable
            nodes.append((rng.choice(["and", "or", "xor"]), a, pick()))
    return nodes


def random_tree_nodes(rng, k, leaves, unknown=0.0):
    """a random formula as a node list: literals covering every variable combined pairwise into one root (a combined
    operand sometimes kept: shared subformulas; UNKNOWN constants sometimes)"""
    nodes = [("var", i, -1) for i in range(k)]
    items = []
    for v in list(range(k)) + [rng.randrange(k) for _ in range(leaves - k)]:
        if rng.random() < 0.4:
            nodes.append(("not", v, -1))
            items.append(len(nodes) - 1)
        else:
            items.append(v)
    if rng.random() < unknown:
        nodes.append(("const", 2, -1))
        items.append(len(nodes) - 1)
    rng.shuffle(items)
    while len(items) > 1:
        a = items.pop(rng.randrange(len(items)))
        b = items.pop(rng.randrange(len(items)))
        nodes.append((rng.choice(["and", "or", "xor"]), a, b))
        items.append(len(nodes) - 1)
        if rng.random() < 0.1 and len(items) > 1:
            items.append(a)
    nodes.append(("or", items[0], items[0]))                  # the root last
    return nodes


def random_weights(rng, k, zero=0.1):
    w = {}
    for v in range(k):
        pair = []
        for _ in range(2):
            pair.append(Fr(0) if rng.random() < zero else Fr(rng.randint(1, 9), rng.randint(1, 7)))
        w[v] = tuple(pair)
    return w


def oracle(nodes, k, weights=None):
    """(counts {1, 0, None: assignments}, weighted {1, 0, None: weight}, total weight, models sorted by assignment)"""
    counts, wsum, total, models = {1: 0, 0: 0, None: 0}, {1: Fr(0), 0: Fr(0), None: Fr(0)}, Fr(0), {1: [], 0: []}
    for a in range(1 << k):
        v = evaluate(nodes, a)
        counts[v] += 1
        p = Fr(1)
        for i in range(k):
            w1, w0 = weights[i] if weights else (1, 1)
            p *= w1 if (a >> i) & 1 else w0
        wsum[v] += p
        total += p
        if v is not None:
            models[v].append(a)
    return counts, wsum, total, models


def lex_first(models, order, m):
    """the first m assignments in lexicographic order of the levels (level 0 most significant, 0 before 1)"""
    key = lambda a: tuple((a >> v) & 1 for v in order)             # noqa: E731
    return sorted(models, key=key)[:m]


def as_mask(w):
    return sum(bit << v for v, bit in w.items())


class SymbolicCounts(unittest.TestCase):
    # ------------------------------------------------------------------ acceptance 1: formulas against brute force
    def test_count002_all_models_of_tiny_formulas_match_brute_force(self):
        rng = random.Random(2026)
        checked = 0
        for trial in range(60):
            k = rng.randint(1, 7)
            if trial % 3 == 2:
                nodes = random_tree_nodes(rng, k, rng.randint(k, 2 * k + 2), unknown=0.3 if trial % 2 else 0.0)
            else:
                nodes = random_nodes(rng, k, rng.randint(1, 14), unknown=0.12 if trial % 2 else 0.0)
            prog = mb.Program.from_nodes(k, nodes)
            weights = random_weights(rng, k)
            counts, wsum, total, models = oracle(nodes, k, weights)
            ucounts, _, _, _ = oracle(nodes, k, None)
            orders = ["natural", "dfs", rng.sample(range(k), k)]
            sizes = set()
            for order in orders:
                d = ms.compile(prog, order=order)
                lv = d.order
                self.assertEqual(sorted(lv), list(range(k)))
                c = d.count()
                self.assertEqual((c.domain, c.universe, c.assignments, c.method, c.status, c.weighted),
                                 ("universe", k, 1 << k, "decision diagram", "exact", False))
                self.assertEqual((c.true, c.false, c.unknown, c.total), (ucounts[1], ucounts[0], ucounts[None], 1 << k))
                cw = d.count(weights=weights)
                self.assertEqual((cw.true, cw.false, cw.unknown, cw.total), (wsum[1], wsum[0], wsum[None], total))
                self.assertTrue(cw.weighted)
                # the level kernels: the same raw root words as the CPU loop, the same answer
                for w in (None, weights):
                    cpu_raw, L, _, _ = d.raw(w)
                    gpu_raw, L2, _, dev = d.raw(w, backend="cuda")
                    self.assertTrue(np.array_equal(cpu_raw, gpu_raw) and L == L2)
                g = d.count(weights=weights, backend="cuda")
                self.assertEqual((g.true, g.false, g.unknown, g.backend), (cw.true, cw.false, cw.unknown, "cuda-emulated"))
                # witnesses: the first models in lexicographic order of the levels
                for m in (1, 3, 1 << k):
                    got = [as_mask(w) for w in d.witnesses(m)]
                    self.assertEqual(got, lex_first(models[1], lv, m))
                    if d.info["three_valued"]:
                        self.assertEqual([as_mask(w) for w in d.witnesses(m, value="false")], lex_first(models[0], lv, m))
                # a canonical diagram: the same function under the same order has the same size
                if order == "natural":
                    sizes.add(d.info["nodes"])
                    again = ms.compile(mb.Program.from_nodes(k, nodes + [("not", len(nodes) - 1, -1),
                                                                          ("not", len(nodes), -1)]))
                    self.assertEqual(again.info["nodes"], d.info["nodes"])
            # the native brute force, CPU and CUDA, unweighted and weighted
            for w in (None, weights):
                b1 = ms.brute(prog, weights=w)
                b2 = ms.brute(prog, weights=w, backend="cuda")
                self.assertTrue(np.array_equal(b1.raw[0], b2.raw[0]))
                if w is not None:
                    self.assertTrue(np.array_equal(b1.raw[1], b2.raw[1]))
                want = (wsum[1], wsum[0], wsum[None], total) if w else (ucounts[1], ucounts[0], ucounts[None], 1 << k)
                self.assertEqual((b1.true, b1.false, b1.unknown, b1.total), want)
                self.assertEqual((b2.true, b2.false, b2.unknown, b2.total), want)
                self.assertEqual((b1.method, b1.domain), ("brute force", "universe"))
            checked += 1
        self.assertEqual(checked, 60)
        # probabilities as literal weights (p, 1 - p): the total weight is 1 and TRUE is P(f)
        x = mb.variables(3)
        f = (x[0] & x[1]) | x[2]
        p = {0: (Fr(1, 3), Fr(2, 3)), 1: (Fr(1, 4), Fr(3, 4)), 2: (Fr(1, 2), Fr(1, 2))}
        c = ms.compile(f, k=3).count(weights=p)
        self.assertEqual((c.true, c.total), (1 - (1 - Fr(1, 12)) * Fr(1, 2), 1))
        # wider universes: a 40-variable formula is counted symbolically (2^40 assignments never enumerated)
        y = mb.variables(40)
        g = mb.FALSE
        for i in range(20):
            g = g | (y[i] & y[i + 20])
        d = ms.compile(g, k=40, order=[v for i in range(20) for v in (i, i + 20)])
        c = d.count()
        self.assertEqual(c.false, 3 ** 20)                            # each pair not both 1: 3 of 4 assignments
        self.assertEqual(c.true, (1 << 40) - 3 ** 20)
        self.assertEqual(d.count(backend="cuda").true, c.true)
        self.assertEqual(d.info["nodes"], 2 + 2 * 20)
        with self.assertRaises(AlgebraBudgetError):
            ms.brute(g, k=40)                                          # 2^40 > the default support budget

    # ------------------------------------------------------------------ acceptance 1: weighted and dependent factors
    def test_count002_weighted_and_dependent_factors_match_brute_force(self):
        rng = random.Random(7)
        for trial in range(40):
            n = rng.randint(1, 5)
            doms = [rng.randint(1, 4) for _ in range(n)]
            factors = []
            for _ in range(rng.randint(0, 5)):
                scope = tuple(rng.sample(range(n), rng.randint(0, min(3, n))))
                dims = [doms[v] for v in scope]
                size = int(np.prod(dims)) if dims else 1
                entries = [Fr(0) if rng.random() < 0.25 else Fr(rng.randint(1, 6), rng.randint(1, 5)) for _ in range(size)]
                table = np.array(entries, dtype=object).reshape(dims) if dims else entries[0]
                factors.append((scope, table, entries))
            fs = ms.Factors(doms, [(sc, tb) for sc, tb, _ in factors])
            # the oracle: every assignment, the product of its factors' entries (C order: the last scope variable fastest)
            Z, models = Fr(0), []
            for asg in itertools.product(*[range(d) for d in doms]):
                p = Fr(1)
                for sc, _, entries in factors:
                    idx = 0
                    for v in sc:
                        idx = idx * doms[v] + asg[v]
                    p *= entries[idx]
                Z += p
                if p:
                    models.append(asg)
            assignments = int(np.prod(doms))
            for order in ("natural", "min-fill", "min-degree", rng.sample(range(n), n)):
                elim, width = fs.order(order)
                r = fs.count(order=order, witnesses=4)
                self.assertEqual((r.value, r.assignments, r.domain, r.method), (Z, assignments, "universe",
                                                                                 "variable elimination"))
                self.assertEqual(r.order, elim)
                self.assertEqual(r.info["width"], width)
                m = fs.count(order=order, weighted=False)
                self.assertEqual(m.value, len(models))
                # witnesses: the first models in lexicographic order of the reverse elimination order
                key = lambda a, e=elim: tuple(a[v] for v in reversed(e))       # noqa: E731
                self.assertEqual(r.witnesses, sorted(models, key=key)[:4])
            b = fs.brute(witnesses=4)
            self.assertEqual((b.value, b.assignments, b.witnesses), (Z, assignments, sorted(models)[:4]))
            self.assertEqual(fs.brute(weighted=False).value, len(models))
        # literal weights and a formula as factors (unary weight tables, CNF clauses as 0/1 tables): variable
        # elimination equals the decision diagram's weighted count and the enumeration
        x = mb.variables(4)
        cnf = [(0, 1), (1, 2, 3), (0, 3)]                            # (x0 | x1) & (x1 | x2 | x3) & (x0 | x3)
        f = mb.TRUE
        for clause in cnf:
            c = mb.FALSE
            for v in clause:
                c = c | x[v]
            f = f & c
        w = {0: (Fr(2, 3), Fr(1, 3)), 1: (Fr(1, 5), Fr(4, 5)), 2: (3, 1), 3: (Fr(1, 2), Fr(7, 2))}
        bdd = ms.compile(f, k=4).count(weights=w)
        tables = [((v,), [w[v][1], w[v][0]]) for v in range(4)]     # value 0 first
        for clause in cnf:
            t = np.ones([2] * len(clause), dtype=object)
            t[(0,) * len(clause)] = 0
            tables.append((clause, t))
        ve = ms.Factors([2] * 4, tables).count()
        self.assertEqual(ve.value, bdd.true)
        _, wsum, _, _ = oracle([("var", i, -1) for i in range(4)] + [("or", 0, 1), ("or", 1, 2), ("or", 5, 3),
                                                                      ("or", 0, 3), ("and", 4, 6), ("and", 8, 7)], 4, w)
        self.assertEqual(bdd.true, wsum[1])
        # a dependent (non-product) weight table over two variables: the joint weights are not literal weights
        dep = ms.Factors([2, 2, 3], [((0, 1), [[Fr(1, 10), Fr(2, 10)], [Fr(3, 10), Fr(4, 10)]]),
                                     ((1, 2), [[1, 0, 2], [Fr(1, 3), 1, 0]])])
        want = sum(Fr([[1, 2], [3, 4]][a][b], 10) * [[1, 0, 2], [Fr(1, 3), 1, 0]][b][c]
                   for a in range(2) for b in range(2) for c in range(3))
        self.assertEqual(dep.count().value, want)
        self.assertEqual(dep.count(order=[2, 1, 0]).value, want)

    # ------------------------------------------------------------------ acceptance 2: budgets, incompleteness
    def test_count002_budgets_refuse_or_return_explicit_incompleteness(self):
        # the order controls the diagram's size: (x0 & xn) | (x1 & xn+1) | ... is linear interleaved, exponential split
        n = 8
        x = mb.variables(2 * n)
        f = mb.FALSE
        for i in range(n):
            f = f | (x[i] & x[i + n])
        prog = mb.Program(f, 2 * n)
        inter = ms.compile(prog, order=[v for i in range(n) for v in (i, i + n)])
        split = ms.compile(prog)
        self.assertEqual(inter.info["nodes"], 2 + 2 * n)
        self.assertEqual(split.info["nodes"], 2 + 2 * (2 ** n) - 2)   # 2^(n+1) - 2 internal nodes
        self.assertEqual(inter.count().true, split.count().true)
        # the node budget (circuit: every node the manager holds) at the limit and one past: a refusal, never a
        # partial diagram
        peak = split.info["peak_nodes"]
        ok = ms.compile(prog, budget=Budget(circuit=peak))
        self.assertEqual(ok.info["nodes"], split.info["nodes"])
        with self.assertRaises(AlgebraBudgetError) as ctx:
            ms.compile(prog, budget=Budget(circuit=peak - 1))
        self.assertEqual((ctx.exception.resource, ctx.exception.need, ctx.exception.limit), ("circuit", peak, peak - 1))
        ms.compile(prog, order=[v for i in range(n) for v in (i, i + n)], budget=Budget(circuit=peak - 1))
        steps = split.info["steps"]
        ms.compile(prog, budget=Budget(iterations=steps))
        with self.assertRaises(AlgebraBudgetError) as ctx:
            ms.compile(prog, budget=Budget(iterations=steps - 1))
        self.assertEqual((ctx.exception.resource, ctx.exception.need), ("iterations", steps))
        # bounded Shannon expansion: incomplete with certified bounds that contain the exact counts, then exact
        rng = random.Random(11)
        for trial in range(25):
            k = rng.randint(2, 7)
            nodes = random_nodes(rng, k, rng.randint(3, 12), unknown=0.1)
            prog = mb.Program.from_nodes(k, nodes)
            weights = random_weights(rng, k, zero=0.0)
            counts, wsum, total, _ = oracle(nodes, k, weights)
            for evals in (0, 1, 3, 7, 1 << (k + 1)):
                for w, want, tot in ((None, {c: Fr(v) for c, v in counts.items()}, 1 << k), (weights, wsum, total)):
                    s = ms.search(prog, weights=w, max_evaluations=evals, order="dfs")
                    self.assertEqual((s.method, s.domain, s.total), ("bounded search", "universe", tot))
                    for (lo, hi), cls in ((s.true, 1), (s.false, 0), (s.unknown, None)):
                        self.assertLessEqual(lo, want[cls])
                        self.assertLessEqual(want[cls], hi)
                    self.assertLessEqual(s.true[0] + s.false[0] + s.unknown[0], tot)
                    self.assertGreaterEqual(s.true[1] + s.false[1] + s.unknown[1], tot)
                    if s.status == "exact":
                        self.assertEqual((s.true, s.false, s.unknown),
                                         tuple((want[c], want[c]) for c in (1, 0, None)))
                        self.assertEqual(s.info["open"], 0)
                    else:
                        self.assertGreater(s.info["open"], 0)
                        self.assertLessEqual(s.info["evaluations"], evals)
                if evals == 0:
                    self.assertEqual(s.status, "incomplete")
            self.assertEqual(ms.search(prog, max_evaluations=1 << (k + 1)).status, "exact")   # 2^(k+1) - 1 suffice
        ms.search(prog, max_evaluations=100, budget=Budget(iterations=100))
        with self.assertRaises(AlgebraBudgetError) as ctx:
            ms.search(prog, max_evaluations=100, budget=Budget(iterations=99))
        self.assertEqual((ctx.exception.resource, ctx.exception.need), ("iterations", 100))
        # variable elimination: the width (the largest joint table) is checked first, with the exact need
        chain = ms.Factors([3] * 6, [((i, i + 1), np.ones((3, 3), dtype=object)) for i in range(5)] +
                           [((0, 5), np.ones((3, 3), dtype=object))])
        order, width = chain.order("min-fill")
        self.assertEqual(width, 27)                                   # a cycle: one triangle of 3 x 3 x 3
        self.assertEqual(chain.count(budget=Budget(tensor=27)).value, int(sum(1 for _ in itertools.product(
            range(3), repeat=6))))
        with self.assertRaises(AlgebraBudgetError) as ctx:
            chain.count(budget=Budget(tensor=26))
        self.assertEqual((ctx.exception.resource, ctx.exception.need, ctx.exception.limit), ("tensor", 27, 26))
        star = ms.Factors([3] * 6, [((0, i), np.ones((3, 3), dtype=object)) for i in range(1, 6)])
        self.assertEqual(star.width("min-fill"), 9)                  # the leaves first
        self.assertEqual(star.width([0, 1, 2, 3, 4, 5]), 3 ** 6)     # the centre first: a clique of the leaves
        self.assertEqual(star.need([0, 1, 2, 3, 4, 5])["tensor"], 3 ** 6)
        self.assertEqual(star.count().value, star.count(order=list(range(6))).value)
        # brute force: support = 2^k (or prod d_i) at the limit and one past
        y = mb.variables(10)
        ms.brute(y[0] | y[9], k=10, budget=Budget(support=1 << 10))
        with self.assertRaises(AlgebraBudgetError) as ctx:
            ms.brute(y[0] | y[9], k=10, budget=Budget(support=(1 << 10) - 1))
        self.assertEqual((ctx.exception.resource, ctx.exception.need), ("support", 1 << 10))
        with self.assertRaises(AlgebraBudgetError):
            ms.brute(y[0] | y[9], k=10, backend="cuda", budget=Budget(support=(1 << 10) - 1))
        self.assertEqual(chain.brute(budget=Budget(support=3 ** 6)).value, 3 ** 6)
        with self.assertRaises(AlgebraBudgetError):
            chain.brute(budget=Budget(support=3 ** 6 - 1))
        # witnesses: output = m at the limit and one past; count bytes on the device
        d = ms.compile(y[0] | y[9], k=10)
        self.assertEqual(len(d.witnesses(8, budget=Budget(output=8))), 8)
        with self.assertRaises(AlgebraBudgetError) as ctx:
            d.witnesses(8, budget=Budget(output=7))
        self.assertEqual((ctx.exception.resource, ctx.exception.need), ("output", 8))
        need = d.need(backend="cuda")["bytes"]
        d.count(backend="cuda", budget=Budget(bytes=need))
        with self.assertRaises(AlgebraBudgetError) as ctx:
            d.count(backend="cuda", budget=Budget(bytes=need - 1))
        self.assertEqual(ctx.exception.resource, "bytes")

    # ------------------------------------------------------------------ the universe is not the observed snapshot
    def test_count002_universe_counts_are_not_observed_row_counts(self):
        frame = pd.DataFrame({"a": [5, 7, 9, 2, 8], "b": [1, 3, 6, 4, 2]})
        db = torbles.from_ddl("CREATE TABLE t (a BIGINT, b BIGINT)", {"t": frame})
        scope = mbk.Scope.from_table(db["t"], ["a", "b"])
        conds = [mb.condition("a", ">", 4), mb.condition("b", ">", 2), mb.condition("a", "<", 6)]
        c = mb.variables(3)
        f = (c[0] & c[1]) | c[2]
        obs = mc.predicate(mc.Domain.observe(scope, conds), f)
        observed = obs.results[0]
        rows_true = sum(1 for a, b in zip(frame.a, frame.b) if (a > 4 and b > 2) or a < 6)
        self.assertEqual((observed["points"], observed["satisfied"]), (5, rows_true))
        sym = ms.compile(f, k=3).count()
        self.assertEqual((sym.domain, sym.assignments, sym.true, sym.false), ("universe", 8, 5, 3))
        self.assertNotEqual((sym.true, sym.assignments), (observed["satisfied"], observed["points"]))
        uni = mc.predicate(mc.Domain.universe(3), f).results[0]           # COUNT-001's enumerated universe
        self.assertEqual((uni["satisfied"], uni["violated"], uni["unknown"]), (sym.true, sym.false, sym.unknown))
        # three-valued: UNKNOWN assignments are counted apart, as COUNT-001 counts UNKNOWN points
        g = (c[0] & mb.UNKNOWN) | c[1]
        s3 = ms.compile(g, k=3).count()
        u3 = mc.predicate(mc.Domain.universe(3), g).results[0]
        self.assertEqual((s3.true, s3.false, s3.unknown), (u3["satisfied"], u3["violated"], u3["unknown"]))
        self.assertEqual((s3.true, s3.false, s3.unknown), (4, 2, 2))

    def test_count002_invalid_inputs_and_opt_in_import(self):
        x = mb.variables(3)
        with self.assertRaises(ValueError):
            ms.compile(x[0] | x[1])                                   # k is required with an expression
        with self.assertRaises(ValueError):
            ms.compile(x[0], k=3, order=[0, 0, 1])                   # not a permutation
        with self.assertRaises(ValueError):
            ms.compile(x[0], k=3, order=[0, 1])
        with self.assertRaises(TypeError):
            ms.compile(x[0], k=3).count(weights={0: (0.5, 0.5)})     # floats are refused
        with self.assertRaises(ValueError):
            ms.compile(x[0], k=3).count(weights={0: (Fr(-1), 1)})
        with self.assertRaises(ValueError):
            ms.compile(x[0], k=3).count(weights={5: (1, 1)})
        with self.assertRaises(ValueError):
            ms.compile(x[0], k=3).witnesses(1, value="false")        # two-valued: one root
        with self.assertRaises(ValueError):
            ms.Factors([2, 2], [((0, 2), [[1, 1], [1, 1]])])
        with self.assertRaises(ValueError):
            ms.Factors([2, 2], [((0, 1), [1, 1, 1])])                # wrong shape
        with self.assertRaises(ValueError):
            ms.Factors([2, 0], [])                                   # an empty domain
        with self.assertRaises(UnsupportedError):
            ms.Factors([3], [((0,), [Fr(1, (1 << 62) + 1), Fr(1, (1 << 62) - 1), Fr(1, (1 << 62) + 3)])])   # lcm > 2^126
        with self.assertRaises(ValueError):
            ms.Factors([2, 2], [((0, 1), [[1, 1], [1, 1]])]).count(order="dfs")
        code = "import sys, torbles; print('torbles.math_symcount' in sys.modules)"
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.split()
        self.assertEqual(out, ["False"])


if __name__ == "__main__":
    unittest.main()
