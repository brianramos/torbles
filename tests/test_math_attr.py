# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""ATTR-001 (sparse interaction and contribution attribution) (torbles/math_attr.c, math_attr_gpu.cu, math_attr.py).

Oracles, independent of the implementation: brute-force enumeration of every coalition with Python Fractions written
here (Shapley values by permutations for tiny games and by the subset formula for larger ones, Banzhaf values, the
Moebius transform by its definition, Shapley / Banzhaf interaction indices and Shapley-Taylor indices by their
discrete-derivative definitions), coalition values by DuckDB re-running the query with only a coalition's source
tuples present (COUNT(*), SUM, COUNT(DISTINCT), existence; COUNT(*) WHERE conditions with SQL NULL semantics), and
DuckDB's join pairs for the large sparse games (each pair a unanimity game). Every native result is also computed on
the CUDA path (emulated here with TORBLES_EMU_SCRAMBLE=1, a scrambled thread order) and compared for equality.
Seeds are fixed.
"""
import itertools
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
from torbles import math_prov as mp
from torbles.math_budgets import Budget
from torbles.math_domains import DomainError
from torbles.math_errors import StaleStateError, UnsupportedError
from torbles.mathematics import AlgebraBudgetError

import duckdb


# ---------------------------------------------------------------------- oracles (Python Fractions, every coalition)

def cuda_label():
    """the backend label of a CUDA result: "cuda" on a device, "cuda-emulated" when the kernels run in the emulation"""
    from torbles import math_gpu
    return "cuda-emulated" if math_gpu.device()["emulated"] else "cuda"


def members(S, n):
    return [i for i in range(n) if (S >> i) & 1]


def popc(S):
    return bin(S).count("1")


def moebius(v, n):
    return [sum((-1) ** (popc(T) - popc(S)) * v[S] for S in range(1 << n) if S & ~T == 0) for T in range(1 << n)]


def shapley_perm(v, n):
    out = [Fraction(0)] * n
    for perm in itertools.permutations(range(n)):
        S = 0
        for p in perm:
            out[p] += v[S | (1 << p)] - v[S]
            S |= 1 << p
    return [x / math.factorial(n) for x in out]


def shapley_subsets(v, n):
    out = []
    for i in range(n):
        tot = Fraction(0)
        for S in range(1 << n):
            if (S >> i) & 1:
                continue
            s = popc(S)
            tot += Fraction(math.factorial(s) * math.factorial(n - s - 1), math.factorial(n)) * (v[S | (1 << i)] - v[S])
        out.append(tot)
    return out


def banzhaf_subsets(v, n):
    return [sum(Fraction(v[S | (1 << i)] - v[S]) for S in range(1 << n) if not (S >> i) & 1) / 2 ** (n - 1)
            for i in range(n)]


def subsets_of(S):
    L = S
    while True:
        yield L
        if not L:
            return
        L = (L - 1) & S


def deriv(v, S, T):
    return sum((-1) ** (popc(S) - popc(L)) * v[L | T] for L in subsets_of(S))


def shapley_interaction(v, n, S):
    s, rest = popc(S), ((1 << n) - 1) & ~S
    return sum(Fraction(math.factorial(n - popc(T) - s) * math.factorial(popc(T)), math.factorial(n - s + 1)) *
               deriv(v, S, T) for T in subsets_of(rest))


def banzhaf_interaction(v, n, S):
    rest = ((1 << n) - 1) & ~S
    return sum(Fraction(deriv(v, S, T)) for T in subsets_of(rest)) / 2 ** (n - popc(S))


def shapley_taylor(v, n, S, k):
    if popc(S) < k:
        return Fraction(deriv(v, S, 0))
    rest = ((1 << n) - 1) & ~S
    return Fraction(k, n) * sum(Fraction(deriv(v, S, T), math.comb(n - 1, popc(T))) for T in subsets_of(rest))


def holds(op, v, t):
    return {"=": v == t, "<>": v != t, "<": v < t, "<=": v <= t, ">": v > t, ">=": v >= t}[op]


def as_sets(d, n):
    """{mask: value} of a {tuple of player indices: value} result"""
    return {sum(1 << i for i in s): x for s, x in d.items()}


# ---------------------------------------------------------------------- provenance data and DuckDB re-execution
R_ROWS = [(1, 10), (1, 20), (2, 30), (None, 40), (1, 10), (3, 5)]
S_ROWS = [(1, 5), (2, 6), (2, 7), (4, 8)]


def cols(rows, names):
    return {n: [row[c] for row in rows] for c, n in enumerate(names)}


def duck(tables):
    con = duckdb.connect()
    for name, (rows, names) in tables.items():
        con.register(name + "_all", pd.DataFrame({"rid": list(range(len(rows))), **{
            nm: pd.Series([row[c] for row in rows], dtype="Int64") for c, nm in enumerate(names)}}))
    return con


def present_views(con, tables, present):
    """views R, S, ... holding only the present rows ({name: set of row ids})"""
    for name in tables:
        ids = sorted(present.get(name, ())) or [-1]
        con.execute(f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM {name}_all WHERE rid IN ({', '.join(map(str, ids))})")


def scalar(con, sql):
    x = con.execute(sql).fetchone()[0]
    return 0 if x is None else int(x)


class Attribution(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(20261031)

    def same_cuda(self, game, coalitions):
        cpu = game.values(coalitions)
        self.assertEqual(game.values(coalitions, backend="cuda"), cpu)
        return cpu

    # ------------------------------------------------------------------ acceptance 1: coalition values, brute force
    def test_attr001_provenance_coalition_values_match_duckdb_rerunning_the_query(self):
        st = mp.Store()
        R = st.source("R", cols(R_ROWS, ["k", "a"]))
        S = st.source("S", cols(S_ROWS, ["k", "b"]))
        J = R.join(S, on=[("k", "k")])                            # columns k, a, k_r, b
        tables = {"R": (R_ROWS, ["k", "a"]), "S": (S_ROWS, ["k", "b"])}
        con = duck(tables)
        join = "FROM R JOIN S ON R.k = S.k"
        games = {
            "COUNT(*)": (ma.Game.provenance(J.group([])), f"SELECT COUNT(*) {join}"),
            "SUM(a)": (ma.Game.provenance(J, weights="a"), f"SELECT SUM(R.a) {join}"),
            "COUNT(DISTINCT b)": (ma.Game.provenance(J.group(["b"]), semiring="bool"),
                                  f"SELECT COUNT(DISTINCT S.b) {join}"),
            "EXISTS k = 1": (ma.Game.provenance(J.group(["k"]), tuples=[0], semiring="bool"),
                             f"SELECT CASE WHEN COUNT(*) > 0 THEN 1 ELSE 0 END {join} WHERE R.k = 1"),
            "2 SUM(b) - COUNT(*), by source": (ma.Game.provenance(
                J, weights=[2 * b - 1 for (_, _, _, b) in J.tuples()], players="sources"),
                f"SELECT COALESCE(SUM(2 * S.b - 1), 0) {join}"),
        }
        for name, (g, sql) in games.items():
            n = g.n
            self.assertEqual(g.info["kind"], "provenance")
            vals = self.same_cuda(g, list(range(1 << n)))
            for mask in range(1 << n):
                if all(isinstance(p, str) for p in g.players):           # players are whole sources
                    present = {src: set(range(len(tables[src][0]))) for i, src in enumerate(g.players) if (mask >> i) & 1}
                    for src in tables:                                   # sources that are no player stay present
                        if src not in g.players:
                            present[src] = set(range(len(tables[src][0])))
                else:                                                   # players are source tuples; the others stay
                    present = {src: {i for i in range(len(rows)) if (src, i) not in g.players}
                               for src, (rows, _) in tables.items()}
                    for i, lab in enumerate(g.players):
                        if (mask >> i) & 1:
                            present[lab[0]].add(lab[1])
                present_views(con, tables, present)
                self.assertEqual(vals[mask], scalar(con, sql), (name, mask))
            # the values by coalition iterables and the game's own v(empty), v(P)
            self.assertEqual(g.values([(), tuple(g.players)]), [g.info["empty"], g.info["full"]])
        # the tuples of no player fixed absent instead of present: S row 1 alone is the only player
        g = ma.Game.provenance(J.group([]), players=[("S", 1)], others="absent")
        self.assertEqual(g.values([0, 1]), [0, 0])                 # without R nothing joins
        g = ma.Game.provenance(J.group([]), players=[("S", 1), "R"], others="absent")
        self.assertEqual(g.values([0, 1, 2, 3]), [0, 0, 0, 1])     # R(2, 30) joins S(2, 6)
        con.close()

    def test_attr001_shapley_and_banzhaf_match_permutation_and_subset_brute_force(self):
        rng = self.rng
        cases = []
        for n in (1, 2, 3, 4, 5):                                    # tiny: Shapley by permutations
            cases.append(("perm", n, [int(x) for x in rng.integers(-20, 21, 1 << n)]))
        for n in (7, 9):                                             # larger: the subset formula
            cases.append(("subsets", n, [int(x) for x in rng.integers(-(1 << 40), 1 << 40, 1 << n)]))
        for kind, n, vals in cases:
            g = ma.Game.table(vals)
            want_sh = shapley_perm(vals, n) if kind == "perm" else shapley_subsets(vals, n)
            if kind == "perm":
                self.assertEqual(want_sh, shapley_subsets(vals, n))
            want_bz = banzhaf_subsets(vals, n)
            sup = g.support()
            for method in ("support", "exhaustive"):
                for backend in (("cpu", "cuda") if method == "exhaustive" else ("cpu",)):
                    r = ma.shapley(sup if method == "support" else g, method=method, backend=backend)
                    self.assertEqual(r["values"], want_sh, (n, method, backend))
                    eff = r["identities"]["efficiency"]
                    self.assertTrue(eff["holds"] and eff["exact"])
                    self.assertEqual((eff["sum"], eff["expected"]), (sum(want_sh), vals[-1] - vals[0]))
                    self.assertEqual(sum(want_sh), vals[-1] - vals[0])   # the oracle's own efficiency
                    b = ma.banzhaf(sup if method == "support" else g, method=method, backend=backend)
                    self.assertEqual(b["values"], want_bz, (n, method, backend))
            cpu, gpu = ma.exhaustive(g), ma.exhaustive(g, backend="cuda")
            self.assertEqual((cpu["numerators"], cpu["shapley"], cpu["banzhaf"]),
                             (gpu["numerators"], gpu["shapley"], gpu["banzhaf"]))   # bit for bit
            self.assertEqual(gpu["backend"], cuda_label())
            self.assertEqual(cpu["denominators"], dict(shapley=math.factorial(n), banzhaf=2 ** (n - 1)))
            self.assertEqual([Fraction(x, math.factorial(n)) for x in cpu["numerators"]["shapley"]], want_sh)
        # provenance games against values recomputed in DuckDB (independent of the native evaluation)
        st = mp.Store()
        R = st.source("R", cols(R_ROWS[:3], ["k", "a"]))
        S = st.source("S", cols(S_ROWS, ["k", "b"]))
        T = st.source("T", cols([(5, 1), (6, 2), (6, 3), (7, 1)], ["b", "c"]))
        J3 = R.join(S, on=[("k", "k")]).join(T, on=[("b", "b")])
        tables = {"R": (R_ROWS[:3], ["k", "a"]), "S": (S_ROWS, ["k", "b"]), "T": ([(5, 1), (6, 2), (6, 3), (7, 1)], ["b", "c"])}
        con = duck(tables)
        for semiring, sql in (("count", "SELECT COUNT(*) FROM R JOIN S ON R.k = S.k JOIN T ON S.b = T.b"),
                              ("bool", "SELECT CASE WHEN COUNT(*) > 0 THEN 1 ELSE 0 END FROM R JOIN S ON R.k = S.k "
                                       "JOIN T ON S.b = T.b")):
            g = ma.Game.provenance(J3.group([]), semiring=semiring)
            n = g.n
            vals = []
            for mask in range(1 << n):
                present = {src: {i for i in range(len(rows)) if (src, i) not in g.players}
                           for src, (rows, _) in tables.items()}
                for i, lab in enumerate(g.players):
                    if (mask >> i) & 1:
                        present[lab[0]].add(lab[1])
                present_views(con, tables, present)
                vals.append(scalar(con, sql))
            want = shapley_subsets(vals, n)
            self.assertEqual(ma.shapley(g)["values"], want, semiring)           # the sparse support
            self.assertEqual(ma.shapley(g, method="exhaustive", backend="cuda")["values"], want)
            self.assertEqual(ma.banzhaf(g)["values"], banzhaf_subsets(vals, n))
            sup = g.support()
            self.assertEqual(sup.info["method"], "expansion" if semiring == "count" else "witnesses")
            self.assertTrue(sup.info["reconstructs"])
            # the support is the brute-force Moebius transform's nonzero part, exactly
            m = moebius(vals, n)
            self.assertEqual(as_sets({tuple(g.players.index(p) for p in k): c for k, c in sup.terms.items()}, n),
                             {T: c for T, c in enumerate(m) if c})
            if semiring == "count":
                self.assertTrue(sup.info["nonnegative"] and sup.info["degree"] == 3)
        con.close()

    def test_attr001_interaction_indices_match_their_definitions_and_sum_identities(self):
        rng = self.rng
        for n in (3, 5, 6):
            vals = [int(x) for x in rng.integers(-1000, 1001, 1 << n)]
            g = ma.Game.table(vals)
            sup = g.support()
            m = moebius(vals, n)
            self.assertEqual(as_sets({tuple(g.players.index(p) for p in k): c for k, c in sup.terms.items()}, n),
                             {T: c for T, c in enumerate(m) if c})
            full_order = n
            for order in range(1, n + 1):
                si = ma.interactions(sup, order, index="shapley")
                bi = ma.interactions(g, order, index="banzhaf")
                mo = ma.interactions(sup, order, index="moebius", min_order=0)
                stt = ma.interactions(sup, order, index="shapley_taylor")
                for r in (si, bi, mo, stt):
                    self.assertIs(r["causal"], False)
                    self.assertIn("weighting", r["convention"])
                want_si = {S: shapley_interaction(vals, n, S) for S in range(1, 1 << n) if popc(S) <= order}
                want_bi = {S: banzhaf_interaction(vals, n, S) for S in range(1, 1 << n) if popc(S) <= order}
                want_st = {S: shapley_taylor(vals, n, S, order) for S in range(1, 1 << n) if popc(S) <= order}
                for got, want in ((si, want_si), (bi, want_bi), (stt, want_st)):
                    self.assertEqual(as_sets(got["by_set"], n), {S: x for S, x in want.items() if x})
                self.assertEqual(as_sets(mo["by_set"], n), {T: c for T, c in enumerate(m) if c and popc(T) <= order})
                # sum identities: Shapley-Taylor is efficient; singletons are the Shapley / Banzhaf values
                self.assertEqual(sum(want_st.values()), vals[-1] - vals[0])
                ide = stt["identities"]["shapley_taylor_efficiency"]
                self.assertTrue(ide["holds"])
                self.assertEqual((ide["sum"], ide["expected"]), (vals[-1] - vals[0], vals[-1] - vals[0]))
                self.assertEqual([want_si[1 << i] for i in range(n)], shapley_subsets(vals, n))
                self.assertEqual([want_bi[1 << i] for i in range(n)], banzhaf_subsets(vals, n))
                if order == full_order:
                    self.assertTrue(mo["identities"]["moebius_sum"]["holds"])
                    self.assertEqual(sum(m), vals[-1])
            # order 1 Shapley-Taylor = Shapley values
            st1 = ma.interactions(sup, 1, index="shapley_taylor")
            self.assertEqual([st1["by_set"].get((i,), 0) for i in range(n)], shapley_subsets(vals, n))
        # the sparse path: a declared support over 100,000 players (no coalition is ever enumerated)
        wide = Budget(variables=100_000)                              # players are the support's variables
        with self.assertRaises(AlgebraBudgetError) as ctx:
            ma.Support.from_terms(100_000, {(3,): 1})
        self.assertEqual((ctx.exception.resource, ctx.exception.need), ("variables", 100_000))
        big = ma.Support.from_terms(100_000, {(3, 99_999): 6, (5,): -2, (3, 5, 70_000): 9, (): 4}, budget=wide)
        r = ma.shapley(big, budget=wide)
        self.assertEqual((r["values"][3], r["values"][5], r["values"][99_999], r["values"][70_000], r["values"][0]),
                         (Fraction(6, 2) + 3, -2 + Fraction(9, 3), 3, 3, 0))
        self.assertTrue(r["identities"]["efficiency"]["holds"])
        self.assertEqual(r["identities"]["efficiency"]["expected"], 13)
        b = ma.banzhaf(big, budget=wide)
        self.assertEqual((b["values"][3], b["values"][5]), (Fraction(6, 2) + Fraction(9, 4), -2 + Fraction(9, 4)))
        it = ma.interactions(big, 2, index="shapley", budget=wide)
        self.assertEqual(it["by_set"][(3, 5)], Fraction(9, 2))
        self.assertEqual(it["by_set"][(3, 99_999)], 6)
        self.assertEqual(len(it["sets"]), 4 + 4)                     # 4 singletons, 4 pairs
        self.assertEqual(it["note"], "every set not listed has index exactly 0")

    def test_attr001_lattice_and_predicate_games_match_duckdb_counts_with_sql_nulls(self):
        rng = self.rng
        n = 400
        x = [int(v) if rng.random() > 0.15 else None for v in rng.integers(-50, 51, n)]
        y = [int(v) if rng.random() > 0.1 else None for v in rng.integers(-50, 51, n)]
        z = [int(v) for v in rng.integers(0, 10, n)]
        frame = pd.DataFrame({"x": pd.Series(x, dtype=object), "y": pd.Series(y, dtype=object),
                              "z": pd.Series(z, dtype=object)})
        db = torbles.from_ddl("CREATE TABLE t (x BIGINT, y BIGINT, z BIGINT)", {"t": frame})
        sel = rng.random(n) < 0.8
        scope = mbk.Scope.from_table(db["t"], ["x", "y", "z"], rows=sel)
        conds = [mb.condition("x", ">", 5), mb.condition("y", "<=", 10), mb.condition("z", ">=", 4),
                 mb.condition("x", "<", 30)]
        sqlc = ["(x > 5)", "(y <= 10)", "(z >= 4)", "(x < 30)"]
        masks = mb.Masks.from_scope(scope, conds)
        db.con.register("sel_rows", pd.DataFrame({"rid": np.flatnonzero(sel).astype(np.int64)}))

        def count(pred):
            return int(db.con.execute(f"SELECT COUNT(*) FROM t WHERE rowid IN (SELECT rid FROM sel_rows) AND "
                                      f"({pred})").fetchone()[0])
        k = 4
        h3 = ml.Histogram.build(masks, nulls="three")
        conj, disj = ma.Game.conjunction(h3), ma.Game.disjunction(h3)
        want_c = [count(" AND ".join(["TRUE"] + [sqlc[i] for i in members(S, k)])) for S in range(1 << k)]
        want_d = [count(" OR ".join(["FALSE"] + [sqlc[i] for i in members(S, k)])) for S in range(1 << k)]
        self.assertEqual(self.same_cuda(conj, range(1 << k)), want_c)
        self.assertEqual(self.same_cuda(disj, range(1 << k)), want_d)
        self.assertEqual(conj.info["monotone"], "decreasing")
        self.assertEqual(disj.info["monotone"], "increasing")
        # the same games as predicates (AND with absent = TRUE, OR with absent = FALSE), and a general predicate
        v = mb.variables(k)
        self.assertEqual(ma.Game.predicate(masks, mb.Program(v[0] & v[1] & v[2] & v[3], k)).values(range(16)), want_c)
        self.assertEqual(ma.Game.predicate(masks, mb.Program(v[0] | v[1] | v[2] | v[3], k), baseline=False)
                         .values(range(16)), want_d)
        expr = (v[0] & ~v[1]) | (v[2] ^ v[3]) | ~(v[0] | v[3])
        base = [True, False, True, False]
        g = ma.Game.predicate(masks, mb.Program(expr, k), baseline=base)

        def sub(S):
            c = [sqlc[i] if (S >> i) & 1 else ("TRUE" if base[i] else "FALSE") for i in range(k)]
            return f"(({c[0]} AND NOT {c[1]}) OR ({c[2]} <> {c[3]}) OR NOT ({c[0]} OR {c[3]}))"
        want_p = [count(sub(S)) for S in range(1 << k)]
        self.assertEqual(self.same_cuda(g, range(1 << k)), want_p)
        self.assertEqual(ma.shapley(g)["values"], shapley_perm(want_p, k))
        self.assertEqual(ma.shapley(g, method="exhaustive", backend="cuda")["values"], shapley_perm(want_p, k))
        self.assertEqual(ma.banzhaf(conj)["values"], banzhaf_subsets(want_c, k))
        # an "apart" histogram: the complete points only (an UNKNOWN condition excludes the point everywhere)
        ha = ml.Histogram.build(masks, nulls="apart")
        ga = ma.Game.conjunction(ha)
        known = " AND ".join(f"({c}) IS NOT NULL" for c in sqlc)
        want_a = [count(" AND ".join([known] + [sqlc[i] for i in members(S, k)])) for S in range(1 << k)]
        self.assertEqual(ga.values(range(16)), want_a)
        self.assertIn("complete points", ga.convention["game"])
        with self.assertRaises(DomainError):
            ma.Game.predicate(masks, mb.Program(v[0], 1))            # a program over another k
        # staleness: a mutated table refuses every later use of the game
        t = db["t"]
        t.flip(0, int(np.flatnonzero(sel)[0]))
        for call in (lambda: conj.values([0]), lambda: ma.shapley(conj), lambda: ma.explore(g)):
            with self.assertRaises(StaleStateError):
                call()

    # ------------------------------------------------------------------ acceptance 2: the convention, no causal reading
    def test_attr001_every_result_states_its_game_baseline_weighting_and_no_causal_reading(self):
        st = mp.Store()
        R = st.source("R", cols(R_ROWS, ["k", "a"]))
        S = st.source("S", cols(S_ROWS, ["k", "b"]))
        g = ma.Game.provenance(R.join(S, on=[("k", "k")]).group([]), players="sources")
        results = [ma.shapley(g), ma.banzhaf(g), ma.shapley(g, method="exhaustive", backend="cuda"),
                   ma.interactions(g, 2), ma.interactions(g, 2, index="shapley_taylor"), ma.explore(g, 1),
                   ma.search(g, "repair", "<=", 0), ma.exhaustive(g)]
        for r in results:
            conv = r["convention"]
            for key in ("game", "players", "baseline", "value"):
                self.assertIn(key, conv)
            self.assertIn("weighting", conv)
            self.assertIs(r["causal"], False)
            self.assertIn("no automatic causal interpretation", r["interpretation"])
        self.assertIn("v(empty): every player's source tuples deleted", g.convention["baseline"])
        self.assertIn("|S|! (n-|S|-1)! / n!", results[0]["convention"]["weighting"])
        self.assertIn("2^-(n-1)", results[1]["convention"]["weighting"])
        self.assertEqual(results[0]["index"], "shapley")
        self.assertEqual(results[0]["values"], [Fraction(5, 2), Fraction(5, 2)])   # COUNT 5 = one 2-player term
        # the attribution depends on the declared convention, not on the data alone: the same rows and predicate,
        # two baselines for the absent conditions, two different (each efficient) attributions
        frame = pd.DataFrame({"x": pd.Series([1, 2, 3, 4, 5, 6, None, 8], dtype=object)})
        db = torbles.from_ddl("CREATE TABLE t (x BIGINT)", {"t": frame})
        scope = mbk.Scope.from_table(db["t"], ["x"])
        masks = mb.Masks.from_scope(scope, [mb.condition("x", ">", 2), mb.condition("x", "<", 7)])
        v = mb.variables(2)
        prog = mb.Program(v[0] & v[1], 2)
        a = ma.shapley(ma.Game.predicate(masks, prog, baseline=True))
        b = ma.shapley(ma.Game.predicate(masks, prog, baseline=False))
        self.assertNotEqual(a["values"], b["values"])
        self.assertEqual(a["convention"]["baseline_constants"], (True, True))
        self.assertEqual(b["convention"]["baseline_constants"], (False, False))
        self.assertEqual(sum(a["values"]), 4 - 8)                   # v(P) = 4 rows; v(empty) = 8: no condition
        self.assertEqual(sum(b["values"]), 4 - 0)                   # applied (the NULL row counts too), or 0
        doc = (Path(ma.__file__).resolve().parent.parent / "docs" / "math" / "attr.md").read_text()
        self.assertIn("no automatic causal interpretation", doc)

    # ------------------------------------------------------------------ sparse provenance, bounded exploration
    def test_attr001_sparse_provenance_attribution_against_duckdb_join_pairs(self):
        rng = self.rng
        nr, ns, nt = 24, 20, 6
        rrows = [(int(rng.integers(0, 6)), int(rng.integers(0, 9))) for _ in range(nr)]
        srows = [(int(rng.integers(0, 6)), int(rng.integers(0, 3))) for _ in range(ns)]
        trows = [(int(rng.integers(0, 3)), int(rng.integers(0, 5))) for _ in range(nt)]
        st = mp.Store()
        R = st.source("R", cols(rrows, ["k", "a"]))
        S = st.source("S", cols(srows, ["k", "b"]))
        T = st.source("T", cols(trows, ["b", "c"]))
        con = duck({"R": (rrows, ["k", "a"]), "S": (srows, ["k", "b"]), "T": (trows, ["b", "c"])})
        for name in "RST":
            con.execute(f"CREATE VIEW {name} AS SELECT * FROM {name}_all")
        # COUNT(*) of R JOIN S: a sum of unanimity games, one per join pair: phi = beta = (pairs of the tuple) / 2
        g = ma.Game.provenance(R.join(S, on=[("k", "k")]).group([]))
        pairs = con.execute("SELECT R.rid, S.rid FROM R JOIN S ON R.k = S.k").fetchall()
        want = {lab: Fraction(0) for lab in g.players}
        for r_, s_ in pairs:
            want[("R", r_)] += Fraction(1, 2)
            want[("S", s_)] += Fraction(1, 2)
        sh, bz = ma.shapley(g), ma.banzhaf(g)
        self.assertGreater(g.n, 40)                                  # 2^n coalitions are out of reach
        self.assertEqual(sh["by_player"], want)
        self.assertEqual(bz["by_player"], want)                      # every term has 2 players
        self.assertEqual(sh["support"]["terms"], len(pairs))        # one term per join pair: sparse, never 2^n
        self.assertEqual(sh["identities"]["efficiency"]["expected"], len(pairs))
        # bounded exploration: with the degree bound 2, levels n-1 and 1 already pin every index (the unknown
        # coefficients all belong to sets of exactly 2 players): 2n evaluations, not 2^n
        n = g.n
        e = ma.explore(g, 2 * n)
        self.assertEqual((e["status"], e["regime"], e["degree"], e["levels"]), ("exact", "moebius", 2, [0, 1, n - 1, n]))
        self.assertEqual(e["explored"], 2 * n)
        self.assertEqual(e["values"]["shapley"], sh["values"])
        self.assertEqual(e["values"]["banzhaf"], bz["values"])
        self.assertEqual(ma.explore(g, 2 * n, backend="cuda")["values"], e["values"])
        # one coalition fewer: level 1 does not fit, certified bounds (never presented as exact)
        e1 = ma.explore(g, 2 * n - 1)
        self.assertEqual((e1["status"], e1["exact"], e1["explored"], e1["levels"]), ("bounds", False, n, [0, n - 1, n]))
        self.assertNotIn("values", e1)
        for (lo, hi), x in zip(e1["bounds"]["shapley"], sh["values"]):
            self.assertTrue(lo <= x <= hi)
        self.assertTrue(any(lo < hi for lo, hi in e1["bounds"]["shapley"]))
        self.assertTrue(e1["identities"]["efficiency"]["holds"])
        # SUM(c) over a three-way join, one player per source: degree 3, exact sparse values against DuckDB coalitions
        g3 = ma.Game.provenance(R.join(S, on=[("k", "k")]).join(T, on=[("b", "b")]), weights="c", players="sources")
        vals = []
        for mask in range(8):
            for i, name in enumerate("RST"):
                con.execute(f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM {name}_all WHERE {(mask >> i) & 1} = 1")
            vals.append(scalar(con, "SELECT COALESCE(SUM(T.c), 0) FROM R JOIN S ON R.k = S.k JOIN T ON S.b = T.b"))
        self.assertEqual(g3.values(range(8)), vals)
        self.assertEqual(vals[:7], [0] * 7)                          # one unanimity game of the three sources
        self.assertEqual(ma.shapley(g3)["values"], [Fraction(vals[7], 3)] * 3)
        con.close()

    def test_attr001_bounded_exploration_bounds_are_certified_and_count_explored_states(self):
        rng = self.rng
        st = mp.Store()
        R = st.source("R", {"k": [int(x) for x in rng.integers(0, 3, 6)], "a": list(range(6))})
        S = st.source("S", {"k": [int(x) for x in rng.integers(0, 3, 5)], "b": list(range(5))})
        J = R.join(S, on=[("k", "k")])
        gc = ma.Game.provenance(J.group([]))                          # N: the Moebius regime
        gb = ma.Game.provenance(J.group([]), semiring="bool")         # B: the range regime (monotone)
        gt = ma.Game.table([int(x) for x in rng.integers(-50, 51, 1 << 7)])   # the range regime (not monotone)
        for g in (gc, gb, gt):
            n = g.n
            vals = g.values(range(1 << n))
            ex_sh, ex_bz = shapley_subsets(vals, n), banzhaf_subsets(vals, n)
            regime = "moebius" if g is gc else "range"
            seen_exact = False
            for budget in sorted({0, 1, n - 1, n, 2 * n, 3 * n, n + math.comb(n, 2), 2 * n + math.comb(n, 2),
                                  (1 << n) // 2, (1 << n) - 2, 1 << n}):
                e = ma.explore(g, budget)
                self.assertEqual(e["regime"], regime)
                self.assertLessEqual(e["explored"], budget)
                self.assertEqual(e["explored"], sum(math.comb(n, s) for s in e["levels"] if 0 < s < n))
                self.assertEqual(e["known"], e["explored"] + 2)
                for (lo, hi), x in zip(e["bounds"]["shapley"], ex_sh):
                    self.assertTrue(lo <= x <= hi, (regime, budget))
                for (lo, hi), x in zip(e["bounds"]["banzhaf"], ex_bz):
                    self.assertTrue(lo <= x <= hi, (regime, budget))
                self.assertTrue(e["identities"]["efficiency"]["holds"])
                if e["exact"]:
                    seen_exact = True
                    self.assertEqual((e["values"]["shapley"], e["values"]["banzhaf"]), (ex_sh, ex_bz))
                    self.assertTrue(all(lo == hi for lo, hi in e["bounds"]["shapley"]))
                else:
                    self.assertNotIn("values", e)
                    self.assertTrue(any(lo < hi for lo, hi in e["bounds"]["shapley"] + e["bounds"]["banzhaf"]))
                d = ma.explore(g, budget, backend="cuda")
                self.assertEqual((d["bounds"], d["explored"], d["exact"]), (e["bounds"], e["explored"], e["exact"]))
            self.assertTrue(seen_exact)
            self.assertTrue(ma.explore(g, (1 << n) - 2)["exact"])     # every level known
        # the degree bound: a join of two sources reaches exactness after levels n-1, 1, 2 only
        self.assertTrue(ma.explore(gc, gc.n * 2 + math.comb(gc.n, 2))["exact"])

    def test_attr001_minimum_explanations_and_repairs_exact_or_certified(self):
        rng = self.rng
        # minimum repairs (hitting sets of the witnesses) with planted multiplicities: w disjoint witness pairs
        # {R_i, S_i} need w deletions, one from each pair (2^w minimum repairs); a minimum explanation is one pair (w)
        for w in (1, 2, 3, 4, 5):
            st = mp.Store()
            R = st.source("R", {"k": list(range(w)), "a": [0] * w})
            S = st.source("S", {"k": list(range(w)), "b": list(range(w))})
            g = ma.Game.provenance(R.join(S, on=[("k", "k")]).group([]), semiring="bool")
            n = g.n
            pos = {lab: i for i, lab in enumerate(g.players)}
            exists = [int(any((C >> pos[("R", i)]) & 1 and (C >> pos[("S", i)]) & 1 for i in range(w)))
                      for C in range(1 << n)]                          # the planted structure, in Python
            self.assertEqual(g.values(range(1 << n)), exists)
            rep = [C for C in range(1 << n) if exists[((1 << n) - 1) & ~C] <= 0]
            best = min(popc(C) for C in rep)
            mult = sum(1 for C in rep if popc(C) == best)
            self.assertEqual((best, mult), (w, 2 ** w))
            ex = ma.search(g, "explanation", ">=", 1)
            self.assertEqual((ex["minimum"], ex["count"]), (2, w))
            r = ma.search(g, "repair", "<=", 0, max_solutions=100)
            self.assertEqual((r["status"], r["minimum"], r["count"], r["exact"]), ("found", best, mult, True))
            self.assertEqual({sum(1 << g.players.index(p) for p in s) for s in r["solutions"]},
                             {C for C in rep if popc(C) == best})
            self.assertEqual(r["explored"], sum(math.comb(n, d) for d in range(1, best + 1)))
            cu = ma.search(g, "repair", "<=", 0, max_solutions=100, backend="cuda")
            self.assertEqual({k: cu[k] for k in ("solutions", "explored", "min_size", "multiplicity")},
                             {k: r[k] for k in ("solutions", "explored", "min_size", "multiplicity")})
        # explanations and repairs of random games, every budget: the certified bounds contain the brute-force answer
        for trial in range(6):
            n = 6
            vals = [int(x) for x in rng.integers(0, 12, 1 << n)]
            g = ma.Game.table(vals)
            for kind, op, th in (("explanation", ">=", 9), ("repair", "<=", 3), ("explanation", "=", 5),
                                 ("repair", "<>", vals[-1]), ("explanation", ">", 100)):
                full = (1 << n) - 1
                sols = [C for C in range(1 << n) if holds(op, vals[C if kind == "explanation" else full & ~C], th)]
                best = min((popc(C) for C in sols), default=None)
                mult = sum(1 for C in sols if popc(C) == best) if sols else 0
                for budget in (0, 3, 6, 10, 21, 30, 41, 63):
                    r = ma.search(g, kind, op, th, max_coalitions=budget, max_solutions=64)
                    self.assertLessEqual(r["explored"], budget)
                    if r["status"] == "found":
                        self.assertEqual((r["minimum"], r["count"]), (best, mult), (kind, op, th, budget))
                    elif r["status"] == "none":
                        self.assertIsNone(best)
                    else:
                        lo, hi = r["min_size"]
                        if hi is not None:
                            self.assertIsNotNone(best)
                        if best is not None:
                            self.assertLessEqual(lo, best)
                            if hi is not None:
                                self.assertGreaterEqual(hi, best)
                        else:
                            self.assertIsNone(hi)
                        if r["multiplicity"] is not None:
                            mlo, mhi = r["multiplicity"]
                            self.assertTrue(mlo <= mult and (mhi is None or mult <= mhi))
                    for s in r["solutions"]:
                        C = sum(1 << p for p in s)
                        self.assertIn(C, sols)
                    cu = ma.search(g, kind, op, th, max_coalitions=budget, max_solutions=64, backend="cuda")
                    self.assertEqual({k: cu[k] for k in ("status", "solutions", "explored", "min_size", "multiplicity")},
                                     {k: r[k] for k in ("status", "solutions", "explored", "min_size", "multiplicity")})
                r = ma.search(g, kind, op, th, max_coalitions=1 << n)
                self.assertNotEqual(r["status"], "incomplete")
        # a monotone game proves "none" without exploring: COUNT can never reach more than v(P)
        st = mp.Store()
        R = st.source("R", {"k": [1, 1, 2], "a": [1, 2, 3]})
        g = ma.Game.provenance(R.group([]))
        r = ma.search(g, "explanation", ">=", 4, max_coalitions=0)
        self.assertEqual((r["status"], r["explored"], r["reason"]), ("none", 0, "monotone game: the extreme value fails"))
        r = ma.search(g, "explanation", ">=", 2, max_coalitions=2)
        self.assertEqual((r["status"], r["min_size"], r["multiplicity"]), ("incomplete", (1, 3), None))

    # ------------------------------------------------------------------ budgets, overflow, invalid inputs
    def test_attr001_budgets_at_the_limit_and_one_past_and_overflow_reported(self):
        vals = list(range(16))
        ma.Game.table(vals, budget=Budget(tensor=16))
        with self.assertRaises(AlgebraBudgetError) as ctx:
            ma.Game.table(vals, budget=Budget(tensor=15))
        self.assertEqual((ctx.exception.resource, ctx.exception.need), ("tensor", 16))
        g = ma.Game.table(vals)
        for backend in ("cpu", "cuda"):
            g.values(range(10), budget=Budget(output=10), backend=backend)
            with self.assertRaises(AlgebraBudgetError) as ctx:
                g.values(range(10), budget=Budget(output=9), backend=backend)
            self.assertEqual((ctx.exception.resource, ctx.exception.need), ("output", 10))
            ma.exhaustive(g, budget=Budget(iterations=16), backend=backend)
            with self.assertRaises(AlgebraBudgetError) as ctx:
                ma.exhaustive(g, budget=Budget(iterations=15), backend=backend)
            self.assertEqual((ctx.exception.resource, ctx.exception.need), ("iterations", 16))
            need = g.need("exhaustive", backend=backend)["bytes"]
            ma.exhaustive(g, budget=Budget(bytes=need), backend=backend)
            with self.assertRaises(AlgebraBudgetError) as ctx:
                ma.exhaustive(g, budget=Budget(bytes=need - 1), backend=backend)
            self.assertEqual((ctx.exception.resource, ctx.exception.need), ("bytes", need))
            ma.explore(g, 10, budget=Budget(iterations=10), backend=backend)
            with self.assertRaises(AlgebraBudgetError) as ctx:
                ma.explore(g, 11, budget=Budget(iterations=10), backend=backend)
            self.assertEqual((ctx.exception.resource, ctx.exception.need), ("iterations", 11))
            ma.search(g, max_coalitions=7, budget=Budget(iterations=7), backend=backend)
            with self.assertRaises(AlgebraBudgetError) as ctx:
                ma.search(g, max_coalitions=8, budget=Budget(iterations=7), backend=backend)
            self.assertEqual((ctx.exception.resource, ctx.exception.need), ("iterations", 8))
        need = g.need("values", 10, backend="cuda")["bytes"]
        g.values(range(10), budget=Budget(bytes=need), backend="cuda")
        with self.assertRaises(AlgebraBudgetError) as ctx:
            g.values(range(10), budget=Budget(bytes=need - 1), backend="cuda")
        self.assertEqual(ctx.exception.resource, "bytes")
        # supports and interactions
        st = mp.Store()
        R = st.source("R", {"k": [1, 1, 2, 2], "a": [1, 2, 3, 4]})
        S = st.source("S", {"k": [1, 2, 2], "b": [1, 2, 3]})
        gp = ma.Game.provenance(R.join(S, on=[("k", "k")]).group(["k"]))
        sn = gp.support_need()
        self.assertEqual((sn["support"], sn["degree"]), (6, 2))
        gp.support(budget=Budget(support=sn["support"]))
        with self.assertRaises(AlgebraBudgetError) as ctx:
            gp.support(budget=Budget(support=sn["support"] - 1))
        self.assertEqual((ctx.exception.resource, ctx.exception.need), ("support", 6))
        sup = gp.support()
        cand = sum(math.comb(len(T), j) for T in sup.terms for j in (1, 2))
        self.assertEqual(ma.interactions_need(sup, 2)["tensor"], cand)
        ma.interactions(sup, 2, budget=Budget(tensor=cand))
        with self.assertRaises(AlgebraBudgetError) as ctx:
            ma.interactions(sup, 2, budget=Budget(tensor=cand - 1))
        self.assertEqual((ctx.exception.resource, ctx.exception.need), ("tensor", cand))
        gt = ma.Game.provenance(R.join(S, on=[("k", "k")]).group(["k"]), semiring="bool")
        tn = gt.support_need("transform")
        gt.support("transform", budget=Budget(iterations=tn["iterations"]))
        with self.assertRaises(AlgebraBudgetError):
            gt.support("transform", budget=Budget(iterations=tn["iterations"] - 1))
        # the provenance plan's circuit: the cone's nodes
        with self.assertRaises(AlgebraBudgetError) as ctx:
            ma.Game.provenance(R.join(S, on=[("k", "k")]).group([]), budget=Budget(circuit=3))
        self.assertEqual(ctx.exception.resource, "circuit")
        # exact values that leave 128 bits are reported, never wrapped
        big = ma.Game.table([0, 1 << 125, 1 << 125, -(1 << 125)])
        for backend in ("cpu", "cuda"):
            with self.assertRaises(OverflowError):
                ma.exhaustive(big, backend=backend)
        with self.assertRaises(OverflowError):
            ma.Game.table([0, 1 << 127])
        with self.assertRaises(OverflowError):                       # v(P), the coefficients' sum, leaves 128 bits
            ma.Support.from_terms(2, {(0,): (1 << 126) + 1, (1,): (1 << 126) + 1})
        wide = ma.Support.from_terms(2, {(0, 1): (1 << 126) + 1, (0,): -(1 << 126)})
        self.assertEqual(ma.shapley(wide)["values"], [Fraction((1 << 126) + 1, 2) - (1 << 126), Fraction((1 << 126) + 1, 2)])
        # representation limits
        stb = mp.Store()
        Rb = stb.source("R", {"k": list(range(40)), "a": [0] * 40})
        g40 = ma.Game.provenance(Rb.group([]))
        with self.assertRaises(UnsupportedError):
            ma.exhaustive(g40)
        self.assertEqual(ma.shapley(g40)["values"], [1] * 40)       # the sparse path has no such limit
        self.assertTrue(ma.explore(g40, 100)["exact"])               # degree 1: level n-1 and level 1 suffice

    def test_attr001_invalid_inputs_and_opt_in_import(self):
        with self.assertRaises(ValueError):
            ma.Game.table([1, 2, 3])                                # not 2^n values
        with self.assertRaises(TypeError):
            ma.Game.table([0, 0.5])
        g = ma.Game.table([0, 1, 1, 3])
        with self.assertRaises(ValueError):
            g.values([4])                                           # a mask outside the players
        with self.assertRaises(ValueError):
            g.values([("nobody",)])
        with self.assertRaises(ValueError):
            ma.search(g, "repair", "~", 0)
        with self.assertRaises(ValueError):
            ma.interactions(g, 2, index="owen")
        with self.assertRaises(ValueError):
            ma.Support.from_terms(3, {(0, 0): 1})
        with self.assertRaises(ValueError):
            ma.Support.from_terms(3, {(0, 5): 1})                   # a player outside n
        with self.assertRaises(UnsupportedError):
            ma.shapley(g.support(), backend="cuda")                 # the sparse path runs on the CPU
        st = mp.Store()
        R = st.source("R", {"k": [1, 2], "a": [1, 2]})
        with self.assertRaises(ValueError):
            ma.Game.provenance(R, players=[("R", 0), ("R", 0)])    # a tuple of two players
        with self.assertRaises(ValueError):
            ma.Game.provenance(R, semiring="N[X]")
        with self.assertRaises(ValueError):
            ma.Game.provenance(R, players="tuples", tuples=[5])
        code = "import sys, torbles; print('torbles.math_attr' in sys.modules)"
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.split()
        self.assertEqual(out, ["False"])


if __name__ == "__main__":
    unittest.main()
