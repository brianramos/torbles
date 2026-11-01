# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""ATTR-002: bounded explanation and repair search (torbles/math_repair.c, math_repair_gpu.cu, math_repair.py).

Oracles, independent of the implementation: every subset of the elements enumerated in Python (ints and Fractions)
for the exhaustive minima and all their ties; the selections of conjunctions and of BOOL-001 programs with the
conditions outside S at their baselines evaluated by DuckDB SQL with its NULL semantics (keep sets, redundant
predicates and their witness rows), the conditions' three-valued values from DuckDB for separations and membership
explanations (and the minimal difference sets of every pair); ATTR-001 games' coalition values with exact Shapley /
Banzhaf values by the subset formula (the branching orders) and ATTR-001's own level search (unit costs); the
aggregates after deletions recomputed in Python and re-run by DuckDB; value edits against closed forms and threshold
changes against every candidate threshold, re-run by DuckDB. Every search runs exhaustively and by branch and bound,
on the CPU and on the CUDA path (emulated here with TORBLES_EMU_SCRAMBLE=1, a scrambled thread order), compared for
equality. Seeds are fixed.
"""
import math
import subprocess
import sys
import unittest
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd

import torbles
from torbles import math_attr as ma
from torbles import math_bank as mbk
from torbles import math_bool as mb
from torbles import math_lattice as ml
from torbles import math_repair as mr
from torbles.math_budgets import Budget
from torbles.math_domains import DomainError
from torbles.math_errors import StaleStateError, UnsupportedError
from torbles.mathematics import AlgebraBudgetError

import duckdb

F = Fraction
METHODS = ("exhaustive", "branch_bound")


# ---------------------------------------------------------------------- oracles (Python ints and Fractions)
def popc(S):
    return bin(S).count("1")


def members(S, n):
    return [i for i in range(n) if (S >> i) & 1]


def submasks(U):
    S = U
    while True:
        yield S
        if not S:
            return
        S = (S - 1) & U


def minima(universe, feasible, cost):
    """the minimum cost over the feasible subsets of `universe` (None: none is feasible) and every subset attaining it"""
    best, ties = None, []
    for S in submasks(universe):
        if not feasible(S):
            continue
        c = cost(S)
        if best is None or c < best:
            best, ties = c, [S]
        elif c == best:
            ties.append(S)
    return best, sorted(ties)


def unit(S):
    return F(popc(S))


def weighted(w):
    return lambda S: sum((F(w[i]) for i in members(S, len(w))), F(0))


def holds(op, v, t):
    if v is None:
        return False                                    # SQL: a NULL aggregate attains no target
    return {"=": v == t, "!=": v != t, "<>": v != t, "<": v < t, "<=": v <= t, ">": v > t, ">=": v >= t}[op]


def shapley_subsets(v, n):
    out = []
    for i in range(n):
        tot = F(0)
        for S in range(1 << n):
            if not (S >> i) & 1:
                s = popc(S)
                tot += F(math.factorial(s) * math.factorial(n - s - 1), math.factorial(n)) * (v[S | (1 << i)] - v[S])
        out.append(tot)
    return out


def banzhaf_subsets(v, n):
    return [sum(F(v[S | (1 << i)] - v[S]) for S in range(1 << n) if not (S >> i) & 1) / 2 ** (n - 1)
            for i in range(n)]


def minimal_sets(sets):
    sets = set(sets)
    return sorted(s for s in sets if not any(t != s and t & s == t for t in sets))


# ---------------------------------------------------------------------- data: an engine table and DuckDB
CONDS = [("x", ">", 10), ("x", ">", 0), ("y", "<=", 40), ("z", ">=", 3), ("y", "<", 60), ("x", "<", 100),
         ("z", ">", 2)]
SQLC = ["(x > 10)", "(x > 0)", "(y <= 40)", "(z >= 3)", "(y < 60)", "(x < 100)", "(z > 2)"]
K = len(CONDS)
PROGRAM_SQL = "(({0} AND {2}) OR ({3} AND NOT {4}) OR ({5} <> {6}))"


def program_expr():
    v = mb.variables(K)
    return (v[0] & v[2]) | (v[3] & ~v[4]) | (v[5] ^ v[6])


def engine(rng, n=240):
    x = [int(a) if rng.random() > 0.1 else None for a in rng.integers(-20, 60, n)]
    y = [int(a) if rng.random() > 0.15 else None for a in rng.integers(0, 80, n)]
    z = [int(a) for a in rng.integers(0, 8, n)]
    v = [int(a) if rng.random() > 0.12 else None for a in rng.integers(-30, 90, n)]
    frame = pd.DataFrame({c: pd.Series(vals, dtype=object) for c, vals in (("x", x), ("y", y), ("z", z), ("v", v))})
    db = torbles.from_ddl("CREATE TABLE t (x BIGINT, y BIGINT, z BIGINT, v BIGINT)", {"t": frame})
    sel = rng.random(n) < 0.85
    db.con.register("sel_rows", pd.DataFrame({"rid": np.flatnonzero(sel).astype(np.int64)}))
    scope = mbk.Scope.from_table(db["t"], ["x", "y", "z", "v"], rows=sel)
    masks = mb.Masks.from_scope(scope, [mb.condition(*c) for c in CONDS])
    return db, dict(x=x, y=y, z=z, v=v), sel, scope, masks


def rows_where(db, pred):
    """the selected rows (row ids) where the SQL predicate is TRUE, by DuckDB"""
    return frozenset(r for (r,) in db.con.execute(
        f"SELECT rowid FROM t WHERE rowid IN (SELECT rid FROM sel_rows) AND ({pred})").fetchall())


def conjunction_sql(S):
    return " AND ".join(["TRUE"] + [SQLC[i] for i in members(S, K)])


def program_sql(S, base):
    return PROGRAM_SQL.format(*[SQLC[i] if (S >> i) & 1 else ("TRUE" if base[i] else "FALSE") for i in range(K)])


class Repair(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(20261101)

    # ------------------------------------------------------------------ shared checks
    def search(self, P, **kw):
        """one search on the CPU and on the CUDA path: equal in everything but the backend label"""
        cpu = mr.search(P, backend="cpu", **kw)
        cu = mr.search(P, backend="cuda", **kw)
        self.assertEqual(cpu["backend"], "cpu")
        self.assertIn(cu["backend"], ("cuda", "cuda-emulated"))
        self.assertEqual({k: v for k, v in cu.items() if k != "backend"}, {k: v for k, v in cpu.items() if k != "backend"})
        return cpu

    def assert_minima(self, P, best, ties, cost=None, orders=("index", "cost"), game=None):
        """every method, order and backend gives the exhaustive minimum with all its ties"""
        for method in METHODS:
            for order in orders:
                r = self.search(P, cost=cost, method=method, order=order, game=game, max_solutions=len(ties) + 3)
                self.assertEqual(r["method"], method)
                self.assertFalse(r["causal"])
                if best is None:
                    self.assertEqual((r["status"], r["exact"], r["minimum"], r["listed"], r["multiplicity"]),
                                     ("none", True, None, 0, None))
                    continue
                self.assertEqual((r["status"], r["exact"], r["reason"], r["minimum"], r["lower"], r["upper"]),
                                 ("found", True, "complete", best, best, best), (method, order))
                self.assertEqual((r["multiplicity"], r["multiplicity_bounds"]), (len(ties), (len(ties), len(ties))))
                self.assertEqual(sorted(r["masks"]), ties, (method, order))
                self.assertEqual(r["costs"], [best] * len(ties))
                self.assertLessEqual(r["explored"], r["max_states"])
                if method == "exhaustive":
                    self.assertEqual(r["masks"], ties)               # listed in increasing (cost, mask) order
        if best is not None and len(ties) > 1:                       # more ties than listed: still counted exactly
            for method in METHODS:
                r = self.search(P, cost=cost, method=method, max_solutions=1)
                self.assertEqual((r["listed"], r["multiplicity"]), (1, len(ties)))
                self.assertIn(r["masks"][0], ties)

    def limit(self, call, resource):
        """the smallest limit of `resource` that call(budget) passes; one below it is refused with that need"""
        lim = 0
        for _ in range(16):
            try:
                out = call(Budget(**{resource: lim}))
                break
            except AlgebraBudgetError as e:
                self.assertEqual(e.resource, resource)
                self.assertGreater(e.need, lim)
                lim = e.need
        else:
            self.fail(f"{resource}: no limit passes")
        if lim:
            with self.assertRaises(AlgebraBudgetError) as cm:
                call(Budget(**{resource: lim - 1}))
            self.assertEqual((cm.exception.resource, cm.exception.need), (resource, lim))
        return lim, out

    # ------------------------------------------------------------------ acceptance 1: exhaustive minima, all ties
    def test_attr002_keep_sets_and_redundant_predicates_match_duckdb_with_sql_nulls(self):
        db, cols, sel, scope, masks = engine(self.rng)
        C = (1 << K) - 1
        sel_of = {S: rows_where(db, conjunction_sql(S)) for S in range(1 << K)}
        full = sel_of[C]
        keeps = lambda S: sel_of[S] == full
        P = mr.Problem.keep(masks, range(K))
        self.assertEqual((P.kind, P.source, P.n, P.allowed), ("hit", "keep", K, C))
        best, ties = minima(C, keeps, unit)
        self.assertEqual((best, len(ties)), (3, 2))                  # x > 10, y <= 40 and either of z >= 3, z > 2
        self.assert_minima(P, best, ties)
        w = [F(3), F(1, 2), F(2), F(5, 3), F(1), F(1, 4), F(7, 4)]
        self.assert_minima(P, *minima(C, keeps, weighted(w)), cost=w)
        tie_w = [F(1)] * K                                           # equal weights: the unit minima again
        self.assert_minima(P, best, ties, cost=tie_w)
        # every minimal keep set's complement is jointly redundant: DuckDB selects the same rows with S alone
        for S in ties:
            self.assertEqual(rows_where(db, conjunction_sql(S)), full)
        # the minimal masks are the failure sets of the rejected rows, minimized (every condition not TRUE there)
        fails = []
        truth = dict((r, vals) for r, *vals in db.con.execute(f"SELECT rowid, {', '.join(SQLC)} FROM t").fetchall())
        for r in np.flatnonzero(sel):
            f = sum(1 << i for i in range(K) if truth[int(r)][i] is not True)
            if f:
                fails.append(f)
        self.assertEqual(sorted(sum(1 << i for i in m) for m in P.minimal_masks()), minimal_sets(fails))
        # redundant predicates: removing one condition keeps the selection (exact counts), as DuckDB says
        red = mr.redundant(P, backend="cuda")
        self.assertEqual([q["index"] for q in red], list(range(K)))
        for q in red:
            i = q["index"]
            self.assertEqual(q["redundant"], keeps(C & ~(1 << i)), i)
            self.assertEqual(q["device"]["redundant"], q["redundant"])
            self.assertEqual(q["differ"] == 0, q["redundant"])        # masks missed
        self.assertEqual([q["redundant"] for q in red], [False, True, False, True, True, True, True])
        # a conjunction over part of the conditions, and the same keep sets from the LATTICE-002 conjunction game
        sub = 0b1010111
        Ps = mr.Problem.keep(masks, members(sub, K))
        self.assertEqual(Ps.allowed, sub)
        self.assert_minima(Ps, *minima(sub, lambda S: sel_of[S] == sel_of[sub], unit))
        hist = ml.Histogram.build(masks, nulls="three")
        Pg = mr.Problem.conjunction(hist, range(K))
        self.assertEqual((Pg.kind, Pg.source), ("game", "game"))
        self.assert_minima(Pg, best, ties)
        self.assertEqual([q["redundant"] for q in mr.redundant(Pg, witnesses=0)], [q["redundant"] for q in red])
        # the search order from ATTR-001's exact indices of that game (the DuckDB counts, the subset formula)
        counts = [len(sel_of[S]) for S in range(1 << K)]
        game = ma.Game.conjunction(hist)
        self.assertEqual(game.values(range(1 << K)), counts)
        for order, idx in (("shapley", shapley_subsets(counts, K)), ("banzhaf", banzhaf_subsets(counts, K))):
            r = self.search(P, method="branch_bound", order=order, game=game, max_solutions=8)
            self.assertEqual(r["order_elements"], tuple(P.names[i] for i in sorted(range(K), key=lambda i: (-idx[i], i))))
            self.assertEqual(sorted(r["masks"]), ties)               # the order changes the work, never the answer
        r = self.search(P, cost=w, method="branch_bound", order="cost")
        self.assertEqual(r["order_elements"], tuple(P.names[i] for i in sorted(range(K), key=lambda i: (w[i], i))))

    def test_attr002_program_keep_sets_and_witness_rows_match_duckdb(self):
        db, cols, sel, scope, masks = engine(self.rng)
        prog = mb.Program(program_expr(), K)
        C = (1 << K) - 1
        target = rows_where(db, program_sql(C, [True] * K))
        for base in ([True] * K, [False] * K, [True, False, False, True, True, False, True]):
            sel_of = {S: rows_where(db, program_sql(S, base)) for S in range(1 << K)}
            self.assertEqual(sel_of[C], target)
            P = mr.Problem.program(masks, prog, baseline=base if len(set(base)) > 1 else base[0])
            self.assertEqual((P.kind, P.source, P.info["selected"]), ("circuit", "program", len(target)))
            self.assertEqual(P.allowed, C & ~0b10)                     # by default the conditions the program reads
            best, ties = minima(C, lambda S: sel_of[S] == target, unit)
            self.assert_minima(P, best, ties)
            w = [F(int(a), int(b)) for a, b in zip(self.rng.integers(1, 9, K), self.rng.integers(1, 4, K))]
            self.assert_minima(P, *minima(C, lambda S: sel_of[S] == target, weighted(w)), cost=w)
            # redundancy with witness rows (the first rows gained without the condition, then the first lost)
            W = 5
            red = mr.redundant(P, witnesses=W, backend="cuda")
            for q in red:
                i = q["index"]
                S = C & ~(1 << i)
                gained, lost = sorted(sel_of[S] - target), sorted(target - sel_of[S])
                self.assertEqual((q["redundant"], q["differ"], q["gained"], q["lost"], q["value"]),
                                 (not gained and not lost, len(gained) + len(lost), len(gained), len(lost),
                                  len(sel_of[S])), (base, i))
                self.assertEqual(q["witnesses"]["gained"], gained[:W])
                self.assertEqual(q["witnesses"]["lost"], lost[:max(0, W - len(gained))])
                self.assertEqual(q["device"], dict(redundant=q["redundant"], differ=q["differ"], value=q["value"]))
            # every subset's exact count of differing rows, on both backends
            subsets = list(range(1 << K))
            ev = P.evaluate(subsets)
            self.assertEqual(P.evaluate(subsets, backend="cuda"), ev)
            self.assertEqual([e["count"] for e in ev], [len(sel_of[S] ^ target) for S in subsets])
            self.assertEqual([e["value"] for e in ev], [len(sel_of[S]) for S in subsets])
        # allowed elements: the conditions outside are always at their baselines
        allowed = 0b0111101
        P = mr.Problem.program(masks, prog, baseline=True, allowed=members(allowed, K))
        sel_of = {S: rows_where(db, program_sql(S, [True] * K)) for S in submasks(allowed)}
        self.assert_minima(P, *minima(allowed, lambda S: sel_of[S] == target, unit))

    def test_attr002_separations_and_membership_explanations_match_exhaustive_minima(self):
        rng = self.rng
        N = 180
        x, y, z = rng.integers(-20, 60, N), rng.integers(0, 80, N), rng.integers(0, 8, N)
        xv, yv = rng.random(N) > 0.1, rng.random(N) > 0.15
        sel = rng.random(N) < 0.9
        con = duckdb.connect()
        con.register("t", pd.DataFrame({"r": np.arange(N), "x": pd.Series([int(a) if ok else None for a, ok in zip(x, xv)],
                                                                             dtype="Int64"),
                                        "y": pd.Series([int(a) if ok else None for a, ok in zip(y, yv)], dtype="Int64"),
                                        "z": pd.Series([int(a) for a in z], dtype="Int64")}))
        truth = {r: tuple(vals) for r, *vals in con.execute(f"SELECT r, {', '.join(SQLC)} FROM t ORDER BY r").fetchall()}
        group = np.array([0 if truth[r][0] is True and truth[r][3] is True else 1 if truth[r][2] is True else 2
                          for r in range(N)])
        scope = mbk.Scope.from_arrays({"x": x, "y": y, "z": z}, valid={"x": xv, "y": yv}, groups=group, ngroups=3,
                                      rows=sel)
        masks = mb.Masks.from_scope(scope, [mb.condition(*c) for c in CONDS])
        rows = [r for r in range(N) if sel[r]]

        def pattern(r, S, conv):
            return tuple(truth[r][i] if conv == "values" else truth[r][i] is True for i in members(S, K))

        def separates(A, B, conv):
            return lambda S: not ({pattern(a, S, conv) for a in A} & {pattern(b, S, conv) for b in B})

        def diff(a, b, conv):
            return sum(1 << i for i in range(K) if (truth[a][i] != truth[b][i] if conv == "values" else
                                                     (truth[a][i] is True) != (truth[b][i] is True)))
        C = (1 << K) - 1
        A = [r for r in rows if group[r] == 0]
        B = [r for r in rows if group[r] == 1]
        for conv in ("values", "selected"):
            best, ties = minima(C, separates(A, B, conv), unit)
            self.assertIsNotNone(best)
            Pg = mr.Problem.separate(masks, ("group", 0), ("group", 1), convention=conv)
            Pr = mr.Problem.separate(masks, A, np.isin(np.arange(N), B), convention=conv)   # rows, a boolean array
            for P in (Pg, Pr):
                self.assertFalse(P.info["infeasible"])
                self.assertEqual(sorted(sum(1 << i for i in m) for m in P.minimal_masks()),
                                 minimal_sets(diff(a, b, conv) for a in A for b in B))
                self.assert_minima(P, best, ties)
            w = [F(int(a)) for a in rng.integers(1, 6, K)]
            self.assert_minima(Pg, *minima(C, separates(A, B, conv), weighted(w)), cost=w)
            allowed = C & ~1                                         # without x > 10
            Pa = mr.Problem.separate(masks, ("group", 0), ("group", 1), convention=conv, allowed=members(allowed, K))
            self.assert_minima(Pa, *minima(allowed, separates(A, B, conv), unit))
        # rows that differ only by FALSE against UNKNOWN: separable on the values, not on SQL selection
        pair = next((a, b) for a in rows for b in rows if a < b and truth[a] != truth[b] and
                    all((p is True) == (q is True) for p, q in zip(truth[a], truth[b])))
        Pv = mr.Problem.separate(masks, [pair[0]], [pair[1]], convention="values")
        self.assert_minima(Pv, *minima(C, separates([pair[0]], [pair[1]], "values"), unit))
        Ps = mr.Problem.separate(masks, [pair[0]], [pair[1]], convention="selected")
        self.assertEqual((Ps.info["infeasible"], Ps.info["witness"]), (True, pair))
        self.assert_minima(Ps, None, [])
        same = next((a, b) for a in rows for b in rows if a < b and truth[a] == truth[b])
        Pi = mr.Problem.separate(masks, [same[0]], [same[1]])
        self.assertEqual((Pi.info["infeasible"], Pi.info["witness"]), (True, same))
        r = self.search(Pi)
        self.assertEqual((r["status"], r["lower"], r["upper"]), ("none", None, None))
        self.assertIn("proven", r["statement"])
        # membership explanations of a program: the selected rows against the other selected rows
        prog = mb.Program(program_expr(), K)
        chosen = {r for (r,) in con.execute(f"SELECT r FROM t WHERE {program_sql(C, [True] * K)}").fetchall()}
        A = [r for r in rows if r in chosen]
        B = [r for r in rows if r not in chosen]
        for conv in ("selected", "values"):
            P = mr.Problem.explain(masks, prog, convention=conv)
            self.assertEqual((P.source, P.info["selected"]), ("explain", len(A)))
            best, ties = minima(C, separates(A, B, conv), unit)
            self.assert_minima(P, best, ties)
        # a membership explanation must contain a separating set of the program's own variables' rows
        self.assertTrue(all(separates(A, B, "values")(S) for S in ties))

    def test_attr002_game_searches_match_attr001_and_exhaustive_coalitions(self):
        rng = self.rng
        n = 6
        full = (1 << n) - 1
        for trial in range(3):
            vals = [int(a) for a in rng.integers(0, 20, 1 << n)]
            vals[0] = 0
            g = ma.Game.table(vals)
            sh, bz = shapley_subsets(vals, n), banzhaf_subsets(vals, n)
            w = [F(int(a), int(b)) for a, b in zip(rng.integers(1, 7, n), rng.integers(1, 3, n))]
            for kind, op, t in (("explanation", ">=", 15), ("repair", "<=", 5), ("explanation", "=", 7),
                                ("repair", "<>", vals[-1]), ("explanation", ">", 100)):
                feasible = (lambda S, op=op, t=t: holds(op, vals[S], t)) if kind == "explanation" else \
                    (lambda S, op=op, t=t: holds(op, vals[full & ~S], t))
                best, ties = minima(full, feasible, unit)
                P = mr.Problem.game(g, kind, op, t)
                self.assertIn(op, P.convention["problem"])
                self.assert_minima(P, best, ties, orders=("index", "cost", "shapley", "banzhaf"))
                a1 = ma.search(g, kind, op, t, max_solutions=64)                   # ATTR-001's level search
                if best is None:
                    self.assertEqual(a1["status"], "none")
                else:
                    self.assertEqual((a1["minimum"], a1["count"]), (best, len(ties)))
                    self.assertEqual(sorted(sum(1 << p for p in s) for s in a1["solutions"]), ties)
                self.assert_minima(P, *minima(full, feasible, weighted(w)), cost=w)
                for order, idx in (("shapley", sh), ("banzhaf", bz)):
                    r = self.search(P, method="branch_bound", order=order)
                    self.assertEqual(r["order_elements"], tuple(sorted(range(n), key=lambda i: (-idx[i], i))))
            # the indices themselves: ATTR-001 against the subset formula (the orders rest on them)
            self.assertEqual(ma.shapley(g)["values"], sh)
            self.assertEqual(ma.banzhaf(g)["values"], bz)
        # evaluation of every coalition on both backends: v and feasibility
        P = mr.Problem.game(g, "repair", "<=", 9)
        ev = P.evaluate(range(1 << n))
        self.assertEqual(P.evaluate(range(1 << n), backend="cuda"), ev)
        self.assertEqual([(e["value"], e["feasible"]) for e in ev], [(vals[full & ~S], vals[full & ~S] <= 9)
                                                                       for S in range(1 << n)])

    def test_attr002_deletion_repairs_match_exhaustive_minima_and_duckdb_aggregates(self):
        db, cols, sel, scope, masks = engine(self.rng, n=160)
        x, v = cols["x"], cols["v"]
        pred = mb.Program(mb.var(0), k=K)                              # qualifying: x > 10 (TRUE)
        qual = [bool(sel[r]) and x[r] is not None and x[r] > 10 for r in range(len(x))]
        q_rows = [r for r in range(len(x)) if qual[r]]
        q_null = [r for r in q_rows if v[r] is None]
        q_val = [r for r in q_rows if v[r] is not None]
        other = [r for r in range(len(x)) if sel[r] and not qual[r]]
        cand = sorted(q_val[:8] + q_null[:2] + other[:1])                # 11 declared deletable rows
        n = len(cand)
        base_sum = sum(v[r] for r in q_val)
        base_avg = F(base_sum, len(q_val))

        def aggregate(agg, S):
            keep = [r for r in q_rows if not (r in cand and (S >> cand.index(r)) & 1)]
            vs = [v[r] for r in keep if v[r] is not None]
            if agg == "count(*)":
                return F(len(keep))
            if agg == "count":
                return F(len(vs))
            if not vs:
                return None
            return F(sum(vs)) if agg == "sum" else F(sum(vs), len(vs))

        def duck(agg, S):
            deleted = [cand[i] for i in members(S, n)] or [-1]
            s, c, k = db.con.execute(
                f"SELECT SUM(v), COUNT(v), COUNT(*) FROM t WHERE rowid IN (SELECT rid FROM sel_rows) AND x > 10 AND "
                f"rowid NOT IN ({', '.join(map(str, deleted))})").fetchone()
            return {"count(*)": F(k), "count": F(c), "sum": None if s is None else F(int(s)),
                    "avg": None if not c else F(int(s), c)}[agg]
        w = [F(int(a), int(b)) for a, b in zip(self.rng.integers(1, 6, n), self.rng.integers(1, 3, n))]
        cases = [("sum", "<=", base_sum - 60), ("sum", ">=", base_sum + 25), ("count(*)", "<=", len(q_rows) - 4),
                 ("count", "=", len(q_val) - 3), ("avg", ">=", base_avg + 4), ("avg", "<", base_avg - 3),
                 ("sum", "!=", base_sum), ("count(*)", "<=", 0)]
        for agg, op, t in cases:
            P = mr.Problem.delete(masks, cand, aggregate=agg, value=None if agg == "count(*)" else "v", op=op,
                                  target=t, predicate=pred)
            self.assertEqual((P.kind, P.n, P.names), ("delete", n, tuple(cand)))
            feasible = lambda S, agg=agg, op=op, t=t: holds(op, aggregate(agg, S), t)
            for cost in (None, w):
                best, ties = minima((1 << n) - 1, feasible, unit if cost is None else weighted(cost))
                self.assert_minima(P, best, ties, cost=cost)
                for S in ties:                                          # DuckDB re-runs the aggregate
                    self.assertTrue(holds(op, duck(agg, S), t), (agg, op, S))
            if agg == "count(*)" and t == 0:
                self.assertIsNone(best)                                 # 11 deletions cannot empty the rows
            ev = P.evaluate(range(1 << n))
            self.assertEqual(P.evaluate(range(1 << n), backend="cuda"), ev)
            for S in (0, 5, 77, (1 << n) - 1):
                a = aggregate(agg, S)
                self.assertEqual(ev[S]["null"], a is None)
                self.assertEqual(ev[S]["feasible"], holds(op, a, t))
                self.assertEqual(ev[S]["count"], len([r for r in q_rows if not (r in cand and (S >> cand.index(r)) & 1)]))
        # deleting a row that does not qualify, or whose value is NULL, never changes SUM: never in a minimum
        P = mr.Problem.delete(masks, cand, aggregate="sum", value="v", op="<=", target=base_sum - 60, predicate=pred)
        idle = {cand.index(r) for r in cand if not qual[r] or v[r] is None}
        r = self.search(P, max_solutions=64)
        self.assertTrue(all(not (S >> i) & 1 for S in r["masks"] for i in idle))

    # ------------------------------------------------------------------ acceptance 2: incomplete with proven bounds
    def test_attr002_budget_exhaustion_is_incomplete_with_proven_bounds_and_explored_states(self):
        rng = self.rng
        n = 14
        msk = []
        for _ in range(26):
            m = 0
            for e in rng.choice(n, size=int(rng.integers(2, 5)), replace=False):
                m |= 1 << int(e)
            msk.append(m)
        P = mr.Problem.masks(n, msk)
        hits = lambda S: all(S & m for m in msk)
        w = [F(int(a), int(b)) for a, b in zip(rng.integers(1, 9, n), rng.integers(1, 4, n))]
        full = (1 << n) - 1
        for cost in (None, w):
            fn = unit if cost is None else weighted(cost)
            best, ties = minima(full, hits, fn)
            for method in METHODS:
                seen = set()
                for states in (1, 2, 3, 7, 20, 64, 150, 400, 1200, 4000, 1 << n):
                    r = self.search(P, cost=cost, method=method, max_states=states, max_solutions=8)
                    seen.add(r["status"])
                    self.assertLessEqual(r["explored"], states)
                    if states > 1:                                      # the root may need two states (I and I + U)
                        self.assertGreater(r["explored"], 0)
                    self.assertEqual(r["max_states"], states)
                    for S, c in zip(r["masks"], r["costs"]):            # every listed subset is feasible at its cost
                        self.assertTrue(hits(S))
                        self.assertEqual(c, fn(S))
                    if r["status"] == "found":
                        self.assertEqual((r["minimum"], r["multiplicity"]), (best, len(ties)))
                        continue
                    self.assertEqual((r["status"], r["exact"], r["reason"], r["minimum"], r["multiplicity"]),
                                     ("incomplete", False, "states", None, None))
                    self.assertLessEqual(r["lower"], best, (method, states))    # proven bounds contain the minimum
                    if r["upper"] is not None:
                        self.assertGreaterEqual(r["upper"], best)
                        self.assertEqual(r["costs"], [r["upper"]] * r["listed"])
                        self.assertGreater(r["listed"], 0)
                    else:
                        self.assertEqual(r["listed"], 0)
                    lo, hi = r["multiplicity_bounds"]
                    if r["lower"] == r["upper"]:
                        self.assertTrue(lo <= len(ties) and (hi is None or len(ties) <= hi), (lo, hi, len(ties)))
                    else:
                        self.assertEqual((lo, hi), (0, None))
                    self.assertIn("not a claimed minimum", r["statement"])
                    if method == "branch_bound":
                        self.assertGreater(r["open"], 0)
                self.assertEqual(seen, {"incomplete", "found"}, (method, cost))
            # every budget's proven lower bound holds, and the gap closes once the budget suffices
            lows = [self.search(P, cost=cost, method="branch_bound", max_states=s)["lower"]
                    for s in (2, 20, 200, 2000, 1 << n)]
            self.assertTrue(all(lo <= best for lo in lows), lows)
            self.assertEqual(lows[-1], best)
        # unit costs: an exhaustive search stopped inside the minimum's level knows the minimum, not every tie
        tie_masks = [0b11, 0b1100, 0b110000, 0b11000000]                # 2^4 = 16 minimum hitting sets of size 4
        Pt = mr.Problem.masks(8, tie_masks)
        ok = lambda S: all(S & m for m in tie_masks)
        before = sum(math.comb(8, s) for s in range(4))                 # levels 0..3: none meets every mask
        level = sorted(S for S in range(256) if popc(S) == 4)           # examined in increasing mask order
        k = 30
        found = [S for S in level[:k] if ok(S)]
        self.assertTrue(0 < len(found) < 16)
        r = self.search(Pt, method="exhaustive", max_states=before + k)
        self.assertEqual((r["status"], r["lower"], r["upper"], r["masks"], r["explored"]),
                         ("incomplete", 4, 4, found, before + k))
        # at least the ties found, at most those and the level's subsets not examined yet
        self.assertEqual(r["multiplicity_bounds"], (len(found), len(found) + len(level) - k))
        r = self.search(Pt, method="branch_bound", max_states=40)
        lo, hi = r["multiplicity_bounds"]
        if r["lower"] == r["upper"]:
            self.assertTrue(lo <= 16 and (hi is None or 16 <= hi))
        r = self.search(Pt, method="exhaustive", max_states=before + len(level), max_solutions=32)
        self.assertEqual((r["status"], r["minimum"], r["multiplicity"], r["listed"]), ("found", 4, 16, 16))
        # deletions stopped by the budget: the proven bounds contain the exhaustive minimum
        db, cols, sel, scope, masks = engine(self.rng, n=120)
        x, v = cols["x"], cols["v"]
        cand = [r for r in range(120) if sel[r] and x[r] is not None and x[r] > 10 and v[r] is not None][:12]
        base = sum(v[r] for r in range(120) if sel[r] and x[r] is not None and x[r] > 10 and v[r] is not None)
        Pd = mr.Problem.delete(masks, cand, aggregate="sum", value="v", op="<=", target=base - 150,
                               predicate=mb.Program(mb.var(0), k=K))
        feasible = lambda S: base - sum(v[cand[i]] for i in members(S, len(cand))) <= base - 150
        best, ties = minima((1 << len(cand)) - 1, feasible, unit)
        self.assertIsNotNone(best)
        for method in METHODS:
            for states in (5, 40, 300):
                r = self.search(Pd, method=method, max_states=states)
                if r["status"] == "incomplete":
                    self.assertLessEqual(r["lower"], best)
                    self.assertTrue(r["upper"] is None or r["upper"] >= best)
                else:
                    self.assertEqual(r["minimum"], best)

    # ------------------------------------------------------------------ CUDA: batched candidates equal the CPU
    def test_attr002_cuda_batched_candidate_evaluation_and_scans_equal_the_cpu(self):
        rng = self.rng
        db, cols, sel, scope, masks = engine(rng)
        hist = ml.Histogram.build(masks, nulls="three")
        problems = [mr.Problem.keep(masks, range(K)), mr.Problem.program(masks, mb.Program(program_expr(), K)),
                    mr.Problem.conjunction(hist, range(K)),
                    mr.Problem.delete(masks, [int(r) for r in np.flatnonzero(sel)[:14]], aggregate="avg", value="v",
                                      op=">=", target=30),
                    mr.Problem.masks(20, [int(m) for m in rng.integers(1, 1 << 20, 30)])]
        for P in problems:
            subs = [0, P.allowed] + [int(s) & P.allowed for s in rng.integers(0, 1 << 62, 3000)]   # 3 device batches
            cpu = P.evaluate(subs)
            self.assertEqual(P.evaluate(subs, backend="cuda"), cpu, P)
            need = P.device_need(len(subs))
            self.assertEqual(need["output"], len(subs))
            P.evaluate(subs, backend="cuda", budget=Budget(bytes=need["bytes"]))
            with self.assertRaises(AlgebraBudgetError):
                P.evaluate(subs, backend="cuda", budget=Budget(bytes=need["bytes"] - 1))
        # many ties of one cost through the device scan (atomic minimum and OR): the same list as the CPU
        Pt = mr.Problem.masks(12, [0b11 << (2 * i) for i in range(6)])     # 2^6 = 64 minimum hitting sets
        for cost in (None, [F(1)] * 12):
            r = self.search(Pt, cost=cost, method="exhaustive", max_solutions=100)
            self.assertEqual((r["minimum"], r["multiplicity"]), (6, 64))
            self.assertEqual(r["masks"], sorted(S for S in range(1 << 12) if popc(S) == 6 and
                                                all(S & (0b11 << (2 * i)) for i in range(6))))

    # ------------------------------------------------------------------ continuous repairs (INV-002)
    def test_attr002_value_edits_and_threshold_repairs_match_closed_forms_and_duckdb(self):
        rng = self.rng
        N = 40
        x = rng.integers(0, 10, N)
        v = rng.integers(-50, 200, N)                                  # value units: v / 100 (scale 2)
        vv = rng.random(N) > 0.1
        scope = mbk.Scope.from_arrays({"x": x, "v": v}, valid={"v": vv}, scales={"v": 2})
        conds = [mb.condition("x", ">", 4), mb.condition("v", "<", "1.5")]
        masks = mb.Masks.from_scope(scope, conds)
        pred = mb.Program(mb.var(0), k=2)
        q_val = [r for r in range(N) if x[r] > 4 and vv[r]]
        s = sum(F(int(v[r]), 100) for r in q_val)
        rows = q_val[:3]
        # l1, weights 2 1 3: the cheapest row takes the whole change
        r = mr.edit(masks, rows, "v", "sum", ">=", s + 5, cost="l1", weights=[2, 1, 3], predicate=pred)
        self.assertEqual((r.status, r.cost, r.claim, r.causal), ("optimal", F(5), "algebraic", False))
        self.assertEqual([sol.change for sol in r.solutions], [{f"row {rows[0]}": 0, f"row {rows[1]}": 5,
                                                                 f"row {rows[2]}": 0}])
        # l2 on AVG: d_i proportional to 1 / w_i, cost D^2 / sum(1 / w)
        avg = s / len(q_val)
        D = (avg - 1) * len(q_val) - s
        r = mr.edit(masks, rows, "v", "avg", "<=", avg - 1, cost="l2", weights=[1, 2, 3], predicate=pred)
        inv = [F(1), F(1, 2), F(1, 3)]
        self.assertEqual(r.cost, D * D / sum(inv))
        self.assertEqual(r.solutions[0].change, {f"row {rows[i]}": D * inv[i] / sum(inv) for i in range(3)})
        # linf: every w_i d_i equal
        r = mr.edit(masks, rows, "v", "sum", ">=", s + 6, cost="linf", weights=[1, 2, 3], predicate=pred)
        lam = F(6) / sum(inv)
        self.assertEqual((r.cost, r.solutions[0].change), (lam, {f"row {rows[i]}": lam * inv[i] for i in range(3)}))
        # bounded edits: the cheapest rows saturate first; past the bounds no edit attains the target
        r = mr.edit(masks, rows, "v", "sum", ">=", s + 5, cost="l1", weights=[2, 1, 3], d_bounds=(None, 2),
                    predicate=pred)
        self.assertEqual((r.cost, r.solutions[0].change), (F(9), {f"row {rows[0]}": 2, f"row {rows[1]}": 2,
                                                                    f"row {rows[2]}": 1}))
        r = mr.edit(masks, rows, "v", "sum", ">=", s + 5, cost="l1", d_bounds=(None, 1), predicate=pred)
        self.assertEqual(r.status, "infeasible")
        # thresholds: COUNT(*) WHERE x > c0 AND v < c1 <= 5, changing c0 or c1: every candidate threshold checked
        con = duckdb.connect()
        con.register("t", pd.DataFrame({"x": pd.Series([int(a) for a in x], dtype="Int64"),
                                        "v": pd.Series([int(a) if ok else None for a, ok in zip(v, vv)], dtype="Int64")}))

        def count(c0, c1):
            """DuckDB: the rows selected with the constants c0, c1 (v in hundredths)"""
            return con.execute(f"SELECT COUNT(*) FROM t WHERE x * {c0.denominator} > {c0.numerator} AND "
                               f"v * {c1.denominator} < {c1.numerator * 100}").fetchone()[0]
        for target, wts in ((5, [1, 1]), (3, [F(1, 2), 4]), (0, [1, 1])):
            out = mr.threshold(scope, conds, [0, 1], aggregate="count(*)", op="<=", target=target, weights=wts)
            self.assertFalse(out["causal"])
            best = {}
            for j, cands in ((0, {F(int(a)) for a in x} | {F(4)}), (1, {F(int(a), 100) for a in v} | {F(3, 2)})):
                c = F(4) if j == 0 else F(3, 2)
                feas = [cp for cp in cands if count(cp if j == 0 else F(4), cp if j == 1 else F(3, 2)) <= target]
                best[j] = min((F(wts[j]) * abs(cp - c) for cp in feas), default=None)
                res = out["results"][j]
                if best[j] is None:
                    self.assertEqual(res.status, "infeasible")
                    continue
                self.assertEqual((res.status, res.cost), ("optimal", best[j]), (target, j))
                for c_new in out["thresholds"][j]:                      # the new constant attains the target
                    self.assertEqual(F(wts[j]) * abs(c_new - c), best[j])
                    self.assertLessEqual(count(c_new if j == 0 else F(4), c_new if j == 1 else F(3, 2)), target)
            m = min(b for b in best.values() if b is not None)
            self.assertEqual(out["minimum"], m)
            self.assertEqual(out["attaining"], [j for j in (0, 1) if best[j] == m])

    # ------------------------------------------------------------------ budgets, refusals, staleness
    def test_attr002_budgets_at_the_limit_and_one_past(self):
        db, cols, sel, scope, masks = engine(self.rng, n=120)
        prog = mb.Program(program_expr(), K)
        msk = [0b11, 0b110, 0b1100, 0b11000, 0b10001, 0b111]
        # problems: variables = elements, support = the masks kept, tensor = pattern pairs, circuit = program nodes
        self.assertEqual(self.limit(lambda b: mr.Problem.masks(5, msk, budget=b), "variables")[0], 5)
        lim, P = self.limit(lambda b: mr.Problem.masks(5, msk, budget=b), "support")
        self.assertEqual((lim, len(P.minimal_masks())), (len(msk), 5))  # the declared copy; 0b111 contains 0b11
        self.limit(lambda b: mr.Problem.masks(5, msk, budget=b), "bytes")
        rows = [int(r) for r in np.flatnonzero(sel)]
        A, B = rows[:len(rows) // 2], rows[len(rows) // 2:]
        lim, P = self.limit(lambda b: mr.Problem.separate(masks, A, B, convention="selected", budget=b), "tensor")
        self.assertEqual(lim, P.info["pairs"])
        self.assertEqual(P.info["pairs"], P.info["patterns_a"] * P.info["patterns_b"])
        lim, P = self.limit(lambda b: mr.Problem.separate(masks, A, B, convention="selected", budget=b), "support")
        self.assertEqual(lim, P.info["masks"])
        self.limit(lambda b: mr.Problem.separate(masks, A, B, budget=b), "bytes")
        self.assertEqual(self.limit(lambda b: mr.Problem.program(masks, prog, budget=b), "circuit")[0],
                         prog.info["nodes"])
        self.limit(lambda b: mr.Problem.program(masks, prog, budget=b), "bytes")
        self.limit(lambda b: mr.Problem.keep(masks, range(K), budget=b), "bytes")
        self.assertEqual(self.limit(lambda b: mr.Problem.keep(masks, range(K), budget=b), "variables")[0], K)
        g = ma.Game.table([int(a) for a in self.rng.integers(0, 9, 32)])
        self.assertEqual(self.limit(lambda b: mr.Problem.game(g, "explanation", ">=", 5, budget=b), "variables")[0], 5)
        rows10 = [int(r) for r in np.flatnonzero(sel)[:10]]
        self.assertEqual(self.limit(lambda b: mr.Problem.delete(masks, rows10, aggregate="sum", value="v", op="<=",
                                                                target=0, budget=b), "variables")[0], 10)
        self.limit(lambda b: mr.Problem.delete(masks, rows10, aggregate="sum", value="v", op="<=", target=0, budget=b),
                   "bytes")
        # evaluation: output = candidates; searches: iterations = max_states, output = max_solutions, bytes
        P = mr.Problem.program(masks, prog)
        for be in ("cpu", "cuda"):
            lim, _ = self.limit(lambda b: P.evaluate(range(100), backend=be, budget=b), "output")
            self.assertEqual(lim, 100)
            self.limit(lambda b: P.evaluate(range(100), backend=be, budget=b), "bytes")
            for method in METHODS:
                lim, _ = self.limit(lambda b: mr.search(P, method=method, max_states=50, backend=be, budget=b),
                                    "iterations")
                self.assertEqual(lim, 50)
                lim, _ = self.limit(lambda b: mr.search(P, method=method, max_solutions=10, backend=be, budget=b),
                                    "output")
                self.assertEqual(lim, 10)
                self.limit(lambda b: mr.search(P, method=method, max_states=300, backend=be, budget=b), "bytes")
        # the declared state budget is the iterations limit by default: past it the answer is incomplete
        r = mr.search(P, method="branch_bound", budget=Budget(iterations=3))
        self.assertEqual((r["status"], r["max_states"]), ("incomplete", 3))
        # redundancy: output = elements x (1 + witnesses); Shapley orders: the game's exhaustive needs (ATTR-001)
        lim, _ = self.limit(lambda b: mr.redundant(P, witnesses=4, budget=b), "output")
        m = popc(P.allowed)                                             # the conditions the program reads
        self.assertEqual((m, lim), (K - 1, m + 4 * m))
        game = ma.Game.conjunction(ml.Histogram.build(masks, nulls="three"))
        self.limit(lambda b: mr.search(mr.Problem.keep(masks, range(K)), order="shapley", game=game, budget=b),
                   "bytes")
        # thresholds: variables = conditions, output = changeable conditions
        conds = [mb.condition("x", ">", 4), mb.condition("v", "<", 30)]
        self.assertEqual(self.limit(lambda b: mr.threshold(scope, conds, [0, 1], target=5, budget=b), "output")[0], 2)
        self.assertEqual(self.limit(lambda b: mr.threshold(scope, conds, [1], target=5, budget=b), "variables")[0], 2)

    def test_attr002_refusals_invalid_inputs_staleness_and_opt_in_import(self):
        db, cols, sel, scope, masks = engine(self.rng, n=100)
        with self.assertRaises(UnsupportedError):                       # subsets are 64-bit masks
            mr.Problem.masks(63, [[0, 62]])
        big = mr.Problem.masks(21, [[i, i + 1] for i in range(20)])
        with self.assertRaises(UnsupportedError):                       # a weighted exhaustive order of 2^21 subsets
            mr.search(big, cost=[1] * 20 + [2], method="exhaustive")
        r = mr.search(big, cost=[1] * 20 + [2], max_states=200_000)
        self.assertEqual((r["method"], r["status"], r["minimum"]), ("branch_bound", "found", 10))
        r = mr.search(big, method="exhaustive")                         # unit costs: levels, no sorted order
        self.assertEqual((r["status"], r["minimum"], r["multiplicity"], r["explored"]), ("found", 10, 1, 1 << 20))
        P = mr.Problem.masks(4, [[0, 1], [2]], allowed=[0, 1, 3])
        self.assertTrue(P.info["infeasible"])
        self.assertEqual(self.search(P)["status"], "none")              # mask {2} cannot be met: proven at once
        Q = mr.Problem.masks(4, [[0, 1], [2, 3]])
        for bad in ([1, 0, 1, 1], [1, -2, 1, 1], [1, 1, 1]):
            with self.assertRaises(ValueError):
                mr.search(Q, cost=bad)
        with self.assertRaises(OverflowError):                          # the common denominator leaves int64
            mr.search(Q, cost=[F(1, 2 ** 31 - 1), F(1, 2 ** 31 - 19), F(1, 2 ** 31 + 11), F(1, 3)])
        with self.assertRaises(ValueError):
            mr.search(Q, max_states=0)
        with self.assertRaises(ValueError):
            mr.search(Q, method="greedy")
        with self.assertRaises(ValueError):
            mr.search(Q, order="shapley")                               # no game to order by
        with self.assertRaises(DomainError):                            # a game over another number of players
            mr.search(Q, order="shapley", game=ma.Game.table(list(range(8))))
        with self.assertRaises(DomainError):                            # a program over another number of conditions
            mr.Problem.program(masks, mb.Program(mb.var(0), k=1))
        with self.assertRaises(TypeError):
            mr.Problem.game(ma.Game.table([0, 1, 2, 3]), "explanation", ">=", 1.5)
        with self.assertRaises(ValueError):
            mr.Problem.masks(4, [[0, 4]])
        with self.assertRaises(TypeError):
            mr.Problem(4)
        # the problem states its convention; staleness: a mutated table refuses every later use
        P = mr.Problem.keep(masks, range(K))
        self.assertIn("AND over S", P.convention["problem"])
        r = mr.search(P)
        self.assertEqual((r["causal"], r["convention"]), (False, P.convention))
        db["t"].flip(0, int(np.flatnonzero(sel)[0]))
        for call in (lambda: mr.search(P), lambda: P.evaluate([0]), lambda: mr.redundant(P),
                     lambda: mr.Problem.keep(masks, range(K))):
            with self.assertRaises(StaleStateError):
                call()
        P.close()
        with self.assertRaises(ValueError):
            P.evaluate([0])
        # opt-in: import torbles does not load the module
        root = Path(__file__).resolve().parents[1]
        out = subprocess.run([sys.executable, "-c", "import sys, torbles; print('torbles.math_repair' in sys.modules)"],
                             capture_output=True, text=True, cwd=root, check=True)
        self.assertEqual(out.stdout.strip(), "False")


if __name__ == "__main__":
    unittest.main()
