# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""DELTA-002: bilinear (join) and nonlinear update laws (torbles/math_bilinear.c, math_bilinear_gpu.cu,
math_bilinear.py).

Oracles, independent of the implementation: explicit matrix products with Python integers; complete reconstruction
of the join from the two bags by enumerating every matched row pair (Python ints); the per-key cells of the bags and
the three cross terms computed from them in Python; DuckDB re-executing the join aggregate (FROM A JOIN B ON key
GROUP BY a.g, b.h with HUGEINT sums and FILTER for the SQL validity of each pair) after the updates; DuckDB and
Python for COUNT(DISTINCT), SUM(DISTINCT), MIN, MAX over the materialized bags; exact Fractions for ratios and
variances; JET-001 for the derivative of the bilinear map. Every native result is also computed on the CUDA path
(emulated here, TORBLES_EMU_SCRAMBLE=1 scrambles the thread order) and compared for equality. Seeds are fixed.
"""
import subprocess
import sys
import unittest
from fractions import Fraction

import numpy as np
import pandas as pd

import torbles
from torbles import math_bank as mbk
from torbles import math_delta as md
from torbles import math_bilinear as mbl
from torbles.math_budgets import Budget
from torbles.math_domains import DomainError
from torbles.mathematics import AlgebraBudgetError

I64 = (1 << 63) - 1


# ---------------------------------------------------------------------- bags, scopes and oracles
def term_value(term, vals, valid):
    """(valid, value) of a monomial (exponent tuple) on a row (SQL validity: every factor present)"""
    out = 1
    for e, v, ok in zip(term, vals, valid):
        if e:
            if not ok:
                return False, 0
            out *= v ** e
    return True, out


def materialize(bag, names, nkeys, groups):
    """the FND-004 array scope of a bag {id: (key, group, values, valid, weight)}, key-major ids key * groups + g,
    each row repeated by its multiplicity"""
    cols, flags, ids = {n: [] for n in names}, {n: [] for n in names}, []
    for k, g, vals, valid, w in bag.values():
        for _ in range(w):
            for n, v, ok in zip(names, vals, valid):
                cols[n].append(int(v)); flags[n].append(bool(ok))
            ids.append(k * groups + g)
    return mbk.Scope.from_arrays({n: np.array(cols[n], dtype=np.int64) for n in names},
                                 valid={n: np.array(flags[n], dtype=bool) for n in names},
                                 groups=np.array(ids, dtype=np.int64), ngroups=nkeys * groups)


def cells_of(bag, terms, nkeys, groups):
    """per-key cells {(key, g): [rows, [counts], [sums]]} of a weighted bag (Python ints)"""
    out = {(k, g): [0, [0] * len(terms), [0] * len(terms)] for k in range(nkeys) for g in range(groups)}
    for k, g, vals, valid, w in bag.values():
        c = out[(k, g)]
        c[0] += w
        for t, term in enumerate(terms):
            ok, v = term_value(term, vals, valid)
            if ok:
                c[1][t] += w; c[2][t] += w * v
    return out


def oracle_join(bag_a, bag_b, pairs, GA, GB, null_key=None):
    """complete reconstruction: every matched row pair (same key, not the null key) of the two bags"""
    out = [dict(rows=0, count=[0] * len(pairs), sum=[0] * len(pairs)) for _ in range(GA * GB)]
    for ka, g, va, oa, wa in bag_a.values():
        for kb, h, vb, ob, wb in bag_b.values():
            if ka != kb or ka == null_key:
                continue
            c, w = out[g * GB + h], wa * wb
            c["rows"] += w
            for p, (ta, tb) in enumerate(pairs):
                ok1, x = term_value(ta, va, oa)
                ok2, y = term_value(tb, vb, ob)
                if ok1 and ok2:
                    c["count"][p] += w; c["sum"][p] += w * x * y
    return out


def cross_terms(old_a, new_a, old_b, new_b, pairs, ia, ib, nkeys, GA, GB, null_key=None):
    """T1 = dA B, T2 = A dB, T3 = dA dB from per-key cells (Python), as join cells"""
    def slot_values(cell, p, t_index):
        return [cell[0]] + [v for q in range(len(pairs)) for v in (cell[1][t_index[q]], cell[2][t_index[q]])]
    res = {}
    for name in ("dA_B", "A_dB", "dA_dB"):
        res[name] = [dict(rows=0, count=[0] * len(pairs), sum=[0] * len(pairs)) for _ in range(GA * GB)]
    for k in range(nkeys):
        if k == null_key:
            continue
        for g in range(GA):
            ao, an = slot_values(old_a[(k, g)], 0, ia), slot_values(new_a[(k, g)], 0, ia)
            for h in range(GB):
                bo, bn = slot_values(old_b[(k, h)], 0, ib), slot_values(new_b[(k, h)], 0, ib)
                for name, f in (("dA_B", lambda s: (an[s] - ao[s]) * bo[s]), ("A_dB", lambda s: ao[s] * (bn[s] - bo[s])),
                                ("dA_dB", lambda s: (an[s] - ao[s]) * (bn[s] - bo[s]))):
                    c = res[name][g * GB + h]
                    c["rows"] += f(0)
                    for p in range(len(pairs)):
                        c["count"][p] += f(1 + 2 * p); c["sum"][p] += f(2 + 2 * p)
    return res


def duck_join(bag_a, names_a, bag_b, names_b, pairs, GA, GB, null_key=None):
    """DuckDB re-executes the join aggregate of the weighted bags (SQL NULL validity per pair)"""
    import duckdb
    con = duckdb.connect()

    def frame(bag, names, gname):
        recs = [dict(k=k, **{gname: g}, w=w, **{n: (int(v) if ok else None) for n, v, ok in zip(names, vals, valid)})
                for k, g, vals, valid, w in bag.values() if w]
        return pd.DataFrame(recs, columns=["k", gname, "w"] + list(names)).astype(object)
    con.register("A", frame(bag_a, names_a, "g"))
    con.register("B", frame(bag_b, names_b, "h"))

    def expr(term, names, alias):
        f = [f"CAST({alias}.{n} AS HUGEINT)" for n, e in zip(names, term) for _ in range(e)]
        ok = " AND ".join(f"{alias}.{n} IS NOT NULL" for n, e in zip(names, term) if e) or "TRUE"
        return " * ".join(f) or "1", ok
    parts = []
    for ta, tb in pairs:
        ea, oa = expr(ta, names_a, "a")
        eb, ob = expr(tb, names_b, "b")
        w = "CAST(a.w AS HUGEINT) * CAST(b.w AS HUGEINT)"
        parts += [f"SUM({w}) FILTER (WHERE {oa} AND {ob})", f"SUM({w} * {ea} * {eb}) FILTER (WHERE {oa} AND {ob})"]
    nk = "" if null_key is None else f" AND a.k <> {int(null_key)}"
    sql = (f"SELECT a.g, b.h, SUM(CAST(a.w AS HUGEINT) * CAST(b.w AS HUGEINT)), {', '.join(parts)} "
           f"FROM A a JOIN B b ON a.k = b.k{nk} GROUP BY a.g, b.h")
    out = [dict(rows=0, count=[0] * len(pairs), sum=[0] * len(pairs)) for _ in range(GA * GB)]
    if len(bag_a) and len(bag_b):
        for r in con.execute(sql).fetchall():
            vals = [0 if v is None else int(v) for v in r[3:]]
            out[int(r[0]) * GB + int(r[1])] = dict(rows=int(r[2]), count=vals[0::2], sum=vals[1::2])
    return out


def delta_of(rows, names, nkeys, groups):
    """a DELTA-001 delta from [(key, group, values, valid, weight)]"""
    return md.Delta.from_arrays({n: np.array([r[2][c] for r in rows], dtype=np.int64) for c, n in enumerate(names)},
                                valid={n: np.array([r[3][c] for r in rows], dtype=bool) for c, n in enumerate(names)},
                                groups=np.array([r[0] * groups + r[1] for r in rows], dtype=np.int64),
                                ngroups=nkeys * groups, weights=np.array([r[4] for r in rows], dtype=np.int64))


def jet_available():
    try:
        from torbles import math_jet  # noqa: F401
        return True
    except Exception:
        return False


class Bilinear(unittest.TestCase):
    def both(self, join, **kw):
        """the CPU update, checked equal to the (emulated, scrambled) CUDA update and its cross terms"""
        cpu = join.apply(backend="cpu", **kw)
        gpu = join.apply(backend="cuda", **kw)
        self.assertTrue(cpu.equal(gpu))
        self.assertEqual(cpu.terms, gpu.terms)
        self.assertEqual((cpu.info["side_a"], cpu.info["side_b"]), (gpu.info["side_a"], gpu.info["side_b"]))
        self.assertEqual(cpu.identity, gpu.identity)
        self.assertIn(gpu.backend, ("cuda", "cuda-emulated"))
        return cpu

    def build(self, a, b, **kw):
        cpu = mbl.Join.build(a, b, backend="cpu", **kw)
        gpu = mbl.Join.build(a, b, backend="cuda", **kw)
        self.assertTrue(cpu.equal(gpu))
        self.assertEqual(cpu.terms, gpu.terms)
        self.assertEqual(cpu.identity, gpu.identity)
        return cpu

    # ------------------------------------------------------------------ acceptance 1: the product rule, matrices
    def test_delta002_acceptance_product_rule_on_small_matrices(self):
        rng = np.random.default_rng(2101)
        shapes = [(1, 1, 1), (2, 3, 4), (5, 1, 3), (4, 7, 2), (3, 64, 3)]
        for G, K, H in shapes:
            for kind in ("small", "extreme", "zero dA", "zero dB"):
                lim = 1000 if kind != "extreme" else I64
                A = [[int(v) for v in rng.integers(-lim, lim, K, endpoint=True)] for _ in range(G)]
                B = [[int(v) for v in rng.integers(-lim, lim, H, endpoint=True)] for _ in range(K)]
                dA = None if kind == "zero dA" else [[int(v) for v in rng.integers(-lim, lim, K, endpoint=True)] for _ in range(G)]
                dB = None if kind == "zero dB" else [[int(v) for v in rng.integers(-lim, lim, H, endpoint=True)] for _ in range(K)]
                if kind == "extreme":                                 # the int64 corners
                    A[0][0], B[0][0] = -(1 << 63), -(1 << 63)
                cpu = mbl.matrix_delta(A, B, dA, dB, backend="cpu")
                gpu = mbl.matrix_delta(A, B, dA, dB, backend="cuda")
                self.assertEqual({k: v for k, v in cpu.items() if k not in ("backend", "device")},
                                 {k: v for k, v in gpu.items() if k not in ("backend", "device")})

                def mul(X, Y):                                        # explicit products, Python ints
                    return [[sum(X[g][q] * Y[q][h] for q in range(K)) for h in range(H)] for g in range(G)]
                zA = [[0] * K for _ in range(G)] if dA is None else dA
                zB = [[0] * H for _ in range(K)] if dB is None else dB
                AB, T1, T2, T3 = mul(A, B), mul(zA, B), mul(A, zB), mul(zA, zB)
                new = mul([[A[g][q] + zA[g][q] for q in range(K)] for g in range(G)],
                          [[B[q][h] + zB[q][h] for h in range(H)] for q in range(K)])
                self.assertEqual((cpu["AB"], cpu["dA_B"], cpu["A_dB"], cpu["dA_dB"]), (AB, T1, T2, T3))
                self.assertEqual(cpu["updated"], new)                 # (A + dA)(B + dB) computed independently
                for g in range(G):                                    # d(AB) = dA B + A dB + dA dB, entry by entry
                    for h in range(H):
                        self.assertEqual(new[g][h] - AB[g][h], cpu["dA_B"][g][h] + cpu["A_dB"][g][h] + cpu["dA_dB"][g][h])
                if kind == "extreme":
                    self.assertTrue(any(abs(v) >= 1 << 127 for row in cpu["updated"] for v in row) or K < 4)
        # the derivative's linearization (dA B + A dB) misses dA dB whenever both sides change
        r = mbl.matrix_delta([[3]], [[5]], [[2]], [[7]])
        self.assertEqual((r["dA_B"], r["A_dB"], r["dA_dB"], r["updated"]), ([[10]], [[21]], [[14]], [[60]]))
        self.assertNotEqual(r["AB"][0][0] + r["dA_B"][0][0] + r["A_dB"][0][0], r["updated"][0][0])
        with self.assertRaises(ValueError):
            mbl.matrix_delta([[1, 2]], [[1, 2]])                      # K mismatch
        with self.assertRaises(OverflowError):
            mbl.matrix_delta([[1 << 63]], [[1]])

    # ------------------------------------------------------------------ acceptance 1: the product rule, joins
    def test_delta002_acceptance_product_rule_on_small_joins(self):
        rng = np.random.default_rng(2102)
        names_a, names_b = ["x", "u"], ["y"]
        K, GA, GB, NULLK = 6, 2, 3, 5
        terms_a = [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1)]
        terms_b = [(0,), (1,), (2,)]
        pairs = [((0, 0), (0,)), ((1, 0), (0,)), ((0, 0), (1,)), ((1, 0), (1,)), ((2, 0), (1,)), ((1, 1), (2,)),
                 ((0, 1), (1,))]
        ia = [terms_a.index(p[0]) for p in pairs]
        ib = [terms_b.index(p[1]) for p in pairs]
        nid = [0]

        def row_a():
            return (int(rng.integers(0, K)), int(rng.integers(0, GA)),
                    [int(rng.integers(-300, 300)), int(rng.integers(-20, 20))],
                    [bool(rng.random() < 0.85), bool(rng.random() < 0.9)], 1)

        def row_b():
            return (int(rng.integers(0, K)), int(rng.integers(0, GB)), [int(rng.integers(-300, 300))],
                    [bool(rng.random() < 0.85)], 1)
        bag_a, bag_b = {}, {}
        for _ in range(40):
            bag_a[nid[0]] = row_a(); nid[0] += 1
        for _ in range(30):
            bag_b[nid[0]] = row_b(); nid[0] += 1
        sa = md.State.from_bank(mbk.build(materialize(bag_a, names_a, K, GA), terms_a))
        sb = md.State.from_bank(mbk.build(materialize(bag_b, names_b, K, GB), terms_b))
        j = self.build(sa, sb, keys=K, groups_a=GA, groups_b=GB, pairs=pairs, null_key=NULLK)
        self.assertEqual(j.cells(), oracle_join(bag_a, bag_b, pairs, GA, GB, NULLK))
        self.assertEqual(j.cells(), duck_join(bag_a, names_a, bag_b, names_b, pairs, GA, GB, NULLK))
        self.assertEqual(j.terms["dA_dB"], j.cells())                 # a build: T3 = AB from zero states
        sides = ["a", "b", "both", "a", "b", "both", "both", "a", "b", "both"]
        for step, side in enumerate(sides):
            rows_a, rows_b = [], []
            new_a, new_b = dict(bag_a), dict(bag_b)

            def change(bag, rows, fresh, ngroups):
                for _ in range(int(rng.integers(1, 5))):              # inserts
                    r = fresh()
                    r = (r[0], r[1], r[2], r[3], int(rng.integers(1, 3)))
                    bag[nid[0]] = r; nid[0] += 1; rows.append(r)
                live = [i for i, r in bag.items() if r[4] > 0]
                for i in rng.choice(live, min(2, len(live)), replace=False):        # deletes
                    k, g, v, o, w = bag.pop(int(i)); rows.append((k, g, v, o, -w))
                live = [i for i, r in bag.items() if r[4] > 0]
                for i in rng.choice(live, min(2, len(live)), replace=False):        # weight changes
                    k, g, v, o, w = bag[int(i)]
                    nw = int(rng.integers(0, 4))
                    if nw != w:
                        bag[int(i)] = (k, g, v, o, nw); rows.append((k, g, v, o, nw - w))
                live = [i for i, r in bag.items() if r[4] > 0]
                for i in rng.choice(live, min(2, len(live)), replace=False):        # key moves (gain / lose matches)
                    k, g, v, o, w = bag[int(i)]
                    nk = int((k + 1 + rng.integers(0, K - 1)) % K)
                    bag[int(i)] = (nk, g, v, o, w); rows += [(k, g, v, o, -w), (nk, g, v, o, w)]
            if side in ("a", "both"):
                change(new_a, rows_a, row_a, GA)
            if side in ("b", "both"):
                change(new_b, rows_b, row_b, GB)
            da = delta_of(rows_a, names_a, K, GA) if rows_a else None
            db = delta_of(rows_b, names_b, K, GB) if rows_b else None
            nxt = self.both(j, a=da, b=db)
            self.assertEqual((nxt.step, nxt.info["base"]), (j.step + 1, j.info["base"]))
            self.assertNotEqual(nxt.identity, j.identity)
            self.assertTrue(nxt.summarizes(nxt.a, nxt.b) and not nxt.summarizes(j.a, j.b))
            expect = oracle_join(new_a, new_b, pairs, GA, GB, NULLK)                 # complete reconstruction
            self.assertEqual(nxt.cells(), expect, step)
            self.assertEqual(nxt.cells(), duck_join(new_a, names_a, new_b, names_b, pairs, GA, GB, NULLK), step)
            terms = cross_terms(cells_of(bag_a, terms_a, K, GA), cells_of(new_a, terms_a, K, GA),
                                cells_of(bag_b, terms_b, K, GB), cells_of(new_b, terms_b, K, GB), pairs, ia, ib, K, GA,
                                GB, NULLK)
            self.assertEqual(nxt.terms, terms, step)                  # T1, T2, T3 from the per-key cells
            old = j.cells()
            for c, cell in enumerate(nxt.cells()):                    # d(AB) = dA B + A dB + dA dB
                for key in ("rows", "count", "sum"):
                    if key == "rows":
                        self.assertEqual(cell["rows"] - old[c]["rows"], sum(nxt.terms[t][c]["rows"] for t in terms))
                    else:
                        for p in range(len(pairs)):
                            self.assertEqual(cell[key][p] - old[c][key][p],
                                             sum(nxt.terms[t][c][key][p] for t in terms))
            if side != "both":                                        # one changing side: T3 = 0
                self.assertTrue(all(v == 0 for c in nxt.terms["dA_dB"] for v in [c["rows"]] + c["count"] + c["sum"]))
            rebuilt = mbl.Join.build(nxt.a, nxt.b, keys=K, groups_a=GA, groups_b=GB, pairs=pairs, null_key=NULLK)
            self.assertTrue(rebuilt.equal(nxt))                       # the join of the new side states, exactly
            j, bag_a, bag_b = nxt, new_a, new_b

    # ------------------------------------------------------------------ acceptance 2: changing join multiplicities
    def test_delta002_acceptance_changing_join_multiplicities_use_per_key_states(self):
        names_a, names_b, K = ["x"], ["y"], 4
        terms_a, terms_b = [(0,), (1,)], [(0,), (1,)]
        pairs = [((0,), (0,)), ((1,), (0,)), ((0,), (1,)), ((1,), (1,))]   # COUNT(*), SUM(a.x), SUM(b.y), SUM(a.x b.y)
        bag_a = {0: (0, 0, [5], [True], 1), 1: (0, 0, [7], [True], 1), 2: (1, 0, [100], [True], 1),
                 3: (1, 0, [-30], [True], 2)}
        bag_b = {10: (0, 0, [2], [True], 1)}
        sa = md.State.from_bank(mbk.build(materialize(bag_a, names_a, K, 1), terms_a))
        sb = md.State.from_bank(mbk.build(materialize(bag_b, names_b, K, 1), terms_b))
        j = self.build(sa, sb, keys=K, pairs=pairs)
        self.assertEqual(j.cell()["sum"][1], 12)                      # only key 0 matches: 5 + 7
        req = mbl.requirement("join_sum")
        self.assertEqual((req["retained"], req["needs_both_sides"], req["linear"], req["deletes"]),
                         ("per-key cells", True, False, "per-key cross terms"))
        steps = [
            ("key 1 gains a match", [], [(1, 0, [3], [True], 1)], 12 + 100 - 60),
            ("key 0 gains a second match", [], [(0, 0, [-1], [True], 1)], 12 + 12 + 40),
            ("key 1 loses its only match", [], [(1, 0, [3], [True], -1)], 24),
            ("key 1 matches again", [], [(1, 0, [8], [True], 2)], 24 + 2 * 40),
            ("both sides insert at key 3", [(3, 0, [9], [True], 1)], [(3, 0, [4], [True], 1)], 104 + 9),
            ("an A row moves to an unmatched key", [(1, 0, [100], [True], -1), (2, 0, [100], [True], 1)], [], 104 + 9 - 200),
        ]
        bag_b_id = 20
        for what, ra, rb, expect_sum_x in steps:
            for r in ra:
                if r[4] > 0:
                    bag_a[bag_b_id] = r
                else:
                    victim = next(i for i, v in bag_a.items() if v[:4] == r[:4] and v[4] > 0)
                    k, g, v, o, w = bag_a[victim]
                    bag_a[victim] = (k, g, v, o, w + r[4])
                bag_b_id += 1
            for r in rb:
                if r[4] > 0:
                    bag_b[bag_b_id] = r
                else:
                    victim = next(i for i, v in bag_b.items() if v[:4] == r[:4] and v[4] > 0)
                    k, g, v, o, w = bag_b[victim]
                    bag_b[victim] = (k, g, v, o, w + r[4])
                bag_b_id += 1
            da = delta_of(ra, names_a, K, 1) if ra else None
            db = delta_of(rb, names_b, K, 1) if rb else None
            nxt = self.both(j, a=da, b=db)
            self.assertEqual(nxt.cell()["sum"][1], expect_sum_x, what)
            self.assertEqual(nxt.cells(), oracle_join(bag_a, bag_b, pairs, 1, 1), what)
            self.assertEqual(nxt.cells(), duck_join(bag_a, names_a, bag_b, names_b, pairs, 1, 1), what)
            t = nxt.terms
            if not ra:                                                # A unchanged, yet SUM(a.x) moved: through B's
                self.assertNotEqual(nxt.cell()["sum"][1], j.cell()["sum"][1])   # per-key multiplicity (T2)
                self.assertEqual(t["dA_B"][0]["sum"][1], 0)
                self.assertEqual(t["A_dB"][0]["sum"][1], nxt.cell()["sum"][1] - j.cell()["sum"][1])
                # a SUM deletion rule on the joined value would add or subtract the B rows' own values instead
                naive = j.cell()["sum"][1] + sum(r[4] * r[2][0] for r in rb)
                self.assertNotEqual(naive, nxt.cell()["sum"][1], what)
            if ra and rb:                                             # a new pair matched on both sides: dA dB only
                self.assertEqual(t["dA_dB"][0]["sum"][3], 9 * 4)
                self.assertEqual(t["dA_B"][0]["sum"][3] + t["A_dB"][0]["sum"][3], 0)
            j = nxt
        # the join refuses side states it does not summarize, and a "successor" that changed other groups
        stray = md.Delta.from_arrays({"y": [1]}, groups=[2], ngroups=K)
        with self.assertRaises(DomainError):
            j.apply(b=stray, new_b=sb.apply(stray))                    # sb is not j's side B any more
        d1 = md.Delta.from_arrays({"y": [1]}, groups=[0], ngroups=K)
        d2 = md.Delta.from_arrays({"y": [1]}, groups=[3], ngroups=K)
        with self.assertRaises(DomainError):
            j.apply(b=d1, new_b=j.b.apply(d2))                         # cells outside d1's groups changed

    # ------------------------------------------------------------------ acceptance 2: DISTINCT, MIN / MAX, ratios
    def test_delta002_acceptance_distinct_minmax_request_own_state(self):
        expect = {"sum": ("linear", True), "count": ("linear", True), "avg": ("sufficient sums", False),
                  "ratio": ("sufficient sums", False), "var_pop": ("sufficient sums", False),
                  "var_samp": ("sufficient sums", False), "count_distinct": ("multiplicities", False),
                  "sum_distinct": ("multiplicities", False), "min": ("multiplicities", False),
                  "max": ("multiplicities", False), "join_sum": ("per-key cells", False)}
        for name, (retained, linear) in expect.items():
            r = mbl.requirement(name)
            self.assertEqual((r["retained"], r["linear"]), (retained, linear), name)
            self.assertEqual(r["deletes"] == "subtract", linear, name)   # only SUM / COUNT use the SUM deletion rule
        for name in ("min", "max"):
            self.assertEqual((mbl.requirement(name)["alternative"], mbl.requirement(name)["alternative_deletes"]),
                             ("extremum", "rebuild when the extreme goes"))
        rng = np.random.default_rng(2104)
        G = 5
        bag, nid = {}, 0
        for _ in range(60):
            bag[nid] = (0, int(rng.integers(0, G)), [int(rng.integers(-6, 7))], [bool(rng.random() < 0.85)], 1); nid += 1
        scope = materialize(bag, ["x"], 1, G)
        lin = md.State.from_bank(mbk.build(scope, ["1", "x", "x^2"]))
        for name in ("count_distinct", "sum_distinct", "min", "max", "join_sum"):
            with self.assertRaises(DomainError):                      # not in linear cells
                mbl.finalize(lin, 0, name, "x")
        ms = mbl.Values.from_scope(scope, "x", "multiset")
        ex = mbl.Values.from_scope(scope, "x", "extremum")
        for name in ("count_distinct", "sum_distinct"):
            with self.assertRaises(DomainError):                      # an extremum keeps no multiplicities
                ex.finalize(0, name)
        for name in ("sum", "avg", "var_pop"):
            with self.assertRaises(DomainError):
                ms.finalize(0, name)

        def check_values(st, bag, stage):
            vals = {g: [] for g in range(G)}
            for k, g, v, o, w in bag.values():
                if o[0]:
                    vals[g] += [v[0]] * w
            rebuilt = mbl.Values.from_scope(materialize(bag, ["x"], 1, G), "x", st.kind)
            self.assertTrue(st.equal(rebuilt), stage)
            import duckdb
            recs = [dict(g=g, x=(v[0] if o[0] else None)) for k, g, v, o, w in bag.values() for _ in range(w)]
            frame = pd.DataFrame(recs, columns=["g", "x"]).astype(object)
            con = duckdb.connect(); con.register("bag", frame)
            duck = {int(r[0]): r[1:] for r in con.execute(
                "SELECT g, COUNT(DISTINCT x), SUM(DISTINCT x), MIN(x), MAX(x), COUNT(x) FROM bag GROUP BY g").fetchall()}
            for g in range(G):
                xs = vals[g]
                py = (len(set(xs)), sum(set(xs)) if xs else None, min(xs) if xs else None, max(xs) if xs else None, len(xs))
                d = duck.get(g, (0, None, None, None, 0))
                self.assertEqual(py, tuple(None if v is None else int(v) for v in d), (stage, g))
                names = ("count_distinct", "sum_distinct", "min", "max", "count") if st.kind == "multiset" else \
                    ("min", "max", "count")
                got = tuple(st.finalize(g, n) for n in names)
                want = py if st.kind == "multiset" else (py[2], py[3], py[4])
                self.assertEqual(got, want, (stage, g))
        check_values(ms, bag, "build")
        check_values(ex, bag, "build")
        rebuilds = 0
        for step in range(25):
            rows, new = [], dict(bag)
            for _ in range(int(rng.integers(0, 4))):
                r = (0, int(rng.integers(0, G)), [int(rng.integers(-8, 9))], [bool(rng.random() < 0.85)], 1)
                new[nid] = r; nid += 1; rows.append(r)
            live = [i for i, r in new.items() if r[4] > 0]
            for i in rng.choice(live, min(int(rng.integers(0, 5)), len(live)), replace=False):
                k, g, v, o, w = new.pop(int(i)); rows.append((k, g, v, o, -w))
            live = [i for i, r in new.items() if r[4] > 0]
            for i in rng.choice(live, min(2, len(live)), replace=False):
                k, g, v, o, w = new[int(i)]
                nw = int(rng.integers(0, 4))
                new[int(i)] = (k, g, v, o, nw); rows.append((k, g, v, o, nw - w))
            if not rows:
                continue
            d = delta_of(rows, ["x"], 1, G)
            ms = ms.apply(d)                                          # exact under every update
            check_values(ms, new, f"step {step}")
            try:
                ex = ex.apply(d)
                check_values(ex, new, f"step {step}")
            except mbl.RebuildRequired as e:                         # the last copy of an extreme went
                rebuilds += 1
                vals = [r[2][0] for r in bag.values() if r[1] == e.group and r[3][0] for _ in range(r[4])]
                self.assertIn(e.value, (min(vals), max(vals)))
                ex = mbl.Values.from_scope(materialize(new, ["x"], 1, G), "x", "extremum")   # the rebuild
                check_values(ex, new, f"rebuild {step}")
            bag = new
        self.assertGreater(rebuilds, 0)
        # a deleted copy of a duplicated value leaves COUNT DISTINCT unchanged (no SUM deletion rule), the last copy not
        s1 = mbl.Values.from_scope(mbk.Scope.from_arrays({"x": [4, 4, 9]}), "x")
        one = md.Delta.from_arrays({"x": [4]}, weights=[-1])
        s2 = s1.apply(one)
        self.assertEqual((s1.finalize(0, "count_distinct"), s2.finalize(0, "count_distinct")), (2, 2))
        self.assertEqual(s2.apply(one).finalize(0, "count_distinct"), 1)
        # deleting a value that is not there: refused exactly by the multiset, invisible to linear cells
        absent = md.Delta.from_arrays({"x": [6, 7]}, weights=[-1, 1])                 # 6 is not there; 7 inserted
        with self.assertRaises(md.NegativeMultiplicityError) as ctx:
            s1.apply(absent)
        self.assertEqual((ctx.exception.group, ctx.exception.value, ctx.exception.have, ctx.exception.change),
                         (0, 6, 0, -1))
        lin1 = md.State.from_bank(mbk.build(mbk.Scope.from_arrays({"x": [4, 4, 9]}), ["1", "x"]))
        lin1.apply(absent)                                            # the aggregates cannot see it
        # the extremum: interior deletes and surviving copies are exact; deleting the last copy of the minimum is not
        e1 = mbl.Values.from_scope(mbk.Scope.from_arrays({"x": [1, 1, 5, 8, 9]}), "x", "extremum")
        e2 = e1.apply(md.Delta.from_arrays({"x": [5, 1]}, weights=[-1, -1]))
        self.assertEqual((e2.finalize(0, "min"), e2.group(0)["min_count"], e2.finalize(0, "max")), (1, 1, 9))
        with self.assertRaises(mbl.RebuildRequired) as ctx:
            e2.apply(md.Delta.from_arrays({"x": [1]}, weights=[-1]))  # 8 may or may not be the next minimum
        self.assertEqual((ctx.exception.group, ctx.exception.value, ctx.exception.have), (0, 1, 1))
        e3 = e2.apply(md.Delta.from_arrays({"x": [1, 0]}, weights=[-1, 2]))           # a smaller value arrives with it
        self.assertEqual((e3.finalize(0, "min"), e3.group(0)["min_count"]), (0, 2))
        e4 = e2.apply(md.Delta.from_arrays({"x": [8, 1]}, weights=[-1, -1]))          # no unknown value is left
        self.assertEqual((e4.finalize(0, "min"), e4.finalize(0, "max"), e4.finalize(0, "count")), (9, 9, 1))
        with self.assertRaises(md.NegativeMultiplicityError):
            e1.apply(md.Delta.from_arrays({"x": [0]}, weights=[-1]))  # below the minimum: not there
        # ratios and variances keep their sufficient sums: finalized again after the update, never updated themselves
        rng = np.random.default_rng(2105)
        bag2 = {i: (0, int(rng.integers(0, 3)), [int(rng.integers(-50, 50)), int(rng.integers(1, 9))],
                    [bool(rng.random() < 0.9), True], 1) for i in range(40)}
        st = md.State.from_bank(mbk.build(materialize(bag2, ["x", "y"], 1, 3), ["1", "x", "x^2", "y"]))
        rows = [(0, r[1], r[2], r[3], -1) for i, r in list(bag2.items())[:12]] + [(0, 2, [77, 3], [True, True], 2)]
        for i in list(bag2)[:12]:
            del bag2[i]
        bag2[999] = (0, 2, [77, 3], [True, True], 2)
        st2 = st.apply(delta_of(rows, ["x", "y"], 1, 3))
        for g in range(3):
            xs = [r[2][0] for r in bag2.values() if r[1] == g and r[3][0] for _ in range(r[4])]
            ys = [r[2][1] for r in bag2.values() if r[1] == g for _ in range(r[4])]
            n, s1_, s2_ = len(xs), sum(xs), sum(v * v for v in xs)
            self.assertEqual(mbl.finalize(st2, g, "avg", "x"), Fraction(s1_, n) if n else None)
            self.assertEqual(mbl.finalize(st2, g, "ratio", "x", "y"), Fraction(s1_, sum(ys)) if n and sum(ys) else None)
            self.assertEqual(mbl.finalize(st2, g, "var_pop", "x", "x^2"), Fraction(n * s2_ - s1_ * s1_, n * n) if n else None)
            self.assertEqual(mbl.finalize(st2, g, "var_samp", "x", "x^2"),
                             Fraction(n * s2_ - s1_ * s1_, n * (n - 1)) if n > 1 else None)
            self.assertEqual(mbl.finalize(st2, g, "count", "x"), n)
            old_avg = mbl.finalize(st, g, "avg", "x")
            removed = [r for r in rows if r[1] == g and r[3][0]]
            if old_avg is not None and removed and n:
                naive = old_avg - Fraction(sum(r[4] * r[2][0] for r in removed), len(xs) or 1)
                self.assertNotEqual(naive, mbl.finalize(st2, g, "avg", "x"))   # no SUM rule on the ratio itself
        with self.assertRaises(DomainError):
            mbl.finalize(st2, 0, "var_pop", "x", "y")                 # not the sufficient sums n, S1, S2

    # ------------------------------------------------------------------ the finite difference is not a derivative
    def test_delta002_finite_difference_is_not_the_derivative(self):
        K = 1
        sa = md.State.from_bank(mbk.build(mbk.Scope.from_arrays({"x": [3, 4]}), ["1", "x"]))
        sb = md.State.from_bank(mbk.build(mbk.Scope.from_arrays({"y": [5]}), ["1", "y"]))
        j = mbl.Join.build(sa, sb, keys=K, pairs=[("x", "y")])
        A, B = 7, 5                                                   # the key's cells: sum x = 7, sum y = 5
        self.assertEqual(j.cell()["sum"], [A * B])
        da = md.Delta.from_arrays({"x": [2]})
        db = md.Delta.from_arrays({"y": [-3]})
        nxt = self.both(j, a=da, b=db)
        dA, dB = 2, -3
        t = nxt.terms
        self.assertEqual((t["dA_B"][0]["sum"][0], t["A_dB"][0]["sum"][0], t["dA_dB"][0]["sum"][0]),
                         (dA * B, A * dB, dA * dB))
        delta = nxt.cell()["sum"][0] - j.cell()["sum"][0]
        self.assertEqual(delta, (A + dA) * (B + dB) - A * B)
        first_order = dA * B + A * dB                                 # D(AB)[dA, dB]: the derivative
        self.assertNotEqual(first_order, delta)
        self.assertEqual(delta - first_order, dA * dB)
        if jet_available():                                           # JET-001 computes the derivative
            from torbles import math_jet as mj
            bld = mj.Builder(params=["a", "b"])
            a, b = bld.params
            circuit = bld.build(a * b)
            jet1 = circuit.plan(1, {"a": A, "b": B}, directions=[{"a": dA, "b": dB}]).at()
            self.assertEqual(jet1.derivative((1,)), first_order)      # the directional derivative = T1 + T2 only
            jet2 = circuit.plan(2, {"a": A, "b": B}, directions=[{"a": dA, "b": dB}]).at()
            self.assertEqual(jet2.coefficient((1,)) + jet2.coefficient((2,)), delta)   # + the second-order term

    # ------------------------------------------------------------------ exactness past 128 and 256 bits
    def test_delta002_wide_values_exact_past_256_bits(self):
        big, K = 1 << 62, 64
        xs, kx = [], []
        for k in range(K):
            xs += [big, big, -big if k % 3 == 0 else big]; kx += [k] * 3
        # FND-004 proves a bank from all its rows (192 x 2^124 > 2^127); DELTA-001 proves each touched cell: the big
        # rows arrive as a delta, giving cells of 3 * 2^124 (< 2^127) in every key
        zeros = mbk.Scope.from_arrays({"x": np.zeros(K, np.int64)}, groups=np.arange(K), ngroups=K)
        sa = md.State.from_bank(mbk.build(zeros, ["x^2"])).apply(md.Delta.from_arrays({"x": xs}, groups=kx, ngroups=K))
        zeros_b = mbk.Scope.from_arrays({"y": np.zeros(K, np.int64)}, groups=np.arange(K), ngroups=K)
        sb = md.State.from_bank(mbk.build(zeros_b, ["y^2"])).apply(md.Delta.from_arrays({"y": xs}, groups=kx, ngroups=K))
        j = self.build(sa, sb, keys=K, pairs=[((2,), (2,))])
        cell = 3 * big * big
        self.assertEqual(j.cell()["sum"], [K * cell * cell])          # 64 * 9 * 2^248 > 2^255
        self.assertGreater(j.cell()["sum"][0], 1 << 255)
        neg = md.Delta.from_arrays({"x": [big]}, groups=[5], ngroups=K, weights=[-1])
        nxt = self.both(j, a=neg)
        self.assertEqual(nxt.cell()["sum"], [(K - 1) * cell * cell + 2 * big * big * cell])
        with self.assertRaises(OverflowError):                        # DELTA-001's proof refuses a side past 2^127
            j.apply(a=md.Delta.from_arrays({"x": [big]}, groups=[0], ngroups=K, weights=[5]))

    # ------------------------------------------------------------------ budgets
    def test_delta002_budgets_at_the_limit_and_one_past(self):
        rng = np.random.default_rng(2107)
        K, GA, GB = 30, 3, 2
        sa = md.State.from_bank(mbk.build(mbk.Scope.from_arrays({"x": rng.integers(-9, 9, 400)},
                                                                groups=rng.integers(0, K * GA, 400), ngroups=K * GA), ["1", "x"]))
        sb = md.State.from_bank(mbk.build(mbk.Scope.from_arrays({"y": rng.integers(-9, 9, 300)},
                                                                groups=rng.integers(0, K * GB, 300), ngroups=K * GB), ["1", "y"]))
        kw = dict(keys=K, groups_a=GA, groups_b=GB, pairs=[("1", "1"), ("x", "y")])
        ref = mbl.Join.build(sa, sb, **kw)
        for backend in ("cpu", "cuda"):
            needs = mbl.Join.need(sa, sb, backend=backend, **kw)
            self.assertEqual((needs["support"], needs["output"], needs["states"], needs["iterations"]),
                             (2, GA * GB, GA * GB * 5, K * GA * GB * 5))
            for resource, value in needs.items():
                self.assertTrue(mbl.Join.build(sa, sb, backend=backend, budget=Budget(**{resource: value}), **kw).equal(ref))
                with self.assertRaises(AlgebraBudgetError) as ctx:
                    mbl.Join.build(sa, sb, backend=backend, budget=Budget(**{resource: value - 1}), **kw)
                self.assertEqual((ctx.exception.resource, ctx.exception.need), (resource, value), (backend, resource))
        da = md.Delta.from_arrays({"x": [1, 2, 3]}, groups=[0, 7, 50], ngroups=K * GA)
        sa2 = sa.apply(da)
        cpu = ref.apply(a=da, new_a=sa2)
        for backend in ("cpu", "cuda"):
            needs = ref.update_need(a=da, new_a=sa2, backend=backend)
            self.assertEqual(needs["iterations"], 3 * GA * GB * 5)   # 3 touched keys
            for resource, value in needs.items():
                self.assertTrue(ref.apply(a=da, new_a=sa2, backend=backend, budget=Budget(**{resource: value})).equal(cpu))
                with self.assertRaises(AlgebraBudgetError) as ctx:
                    ref.apply(a=da, new_a=sa2, backend=backend, budget=Budget(**{resource: value - 1}))
                self.assertEqual((ctx.exception.resource, ctx.exception.need), (resource, value), (backend, resource))
        A, B = [[1, 2, 3]] * 4, [[4, 5]] * 3
        for backend in ("cpu", "cuda"):
            needs = mbl.matrix_need(4, 3, 2, backend=backend)
            self.assertEqual((needs["tensor"], needs["output"], needs["iterations"]), (2 * (12 + 6), 40, 24))
            for resource, value in needs.items():
                mbl.matrix_delta(A, B, A, B, backend=backend, budget=Budget(**{resource: value}))
                with self.assertRaises(AlgebraBudgetError) as ctx:
                    mbl.matrix_delta(A, B, A, B, backend=backend, budget=Budget(**{resource: value - 1}))
                self.assertEqual((ctx.exception.resource, ctx.exception.need), (resource, value), (backend, resource))
        scope = mbk.Scope.from_arrays({"x": rng.integers(-9, 9, 500)}, groups=rng.integers(0, 20, 500), ngroups=20)
        for kind in ("multiset", "extremum"):
            needs = mbl.Values.need(scope, "x", kind)
            for resource, value in needs.items():
                mbl.Values.from_scope(scope, "x", kind, budget=Budget(**{resource: value}))
                with self.assertRaises(AlgebraBudgetError) as ctx:
                    mbl.Values.from_scope(scope, "x", kind, budget=Budget(**{resource: value - 1}))
                self.assertEqual((ctx.exception.resource, ctx.exception.need), (resource, value), (kind, resource))
            st = mbl.Values.from_scope(scope, "x", kind)
            d = md.Delta.from_arrays({"x": [5, 6]}, groups=[1, 2], ngroups=20)
            needs = st.apply_need(d)
            for resource, value in needs.items():
                st.apply(d, budget=Budget(**{resource: value}))
                with self.assertRaises(AlgebraBudgetError):
                    st.apply(d, budget=Budget(**{resource: value - 1}))

    # ------------------------------------------------------------------ identity and compatibility
    def test_delta002_identity_lineage_and_incompatible_inputs(self):
        K = 4
        sa = md.State.from_bank(mbk.build(mbk.Scope.from_arrays({"x": [1, 2, 3]}, groups=[0, 1, 2], ngroups=K), ["1", "x"]))
        sb = md.State.from_bank(mbk.build(mbk.Scope.from_arrays({"y": [4, 5]}, groups=[1, 3], ngroups=K), ["1", "y"]))
        j1 = mbl.Join.build(sa, sb, keys=K, pairs=[("x", "y")])
        j2 = mbl.Join.build(sa, sb, keys=K, pairs=[("x", "y")])
        self.assertEqual(j1.identity, j2.identity)                    # deterministic
        self.assertNotEqual(j1.identity, mbl.Join.build(sa, sb, keys=K, pairs=[("1", "y")]).identity)
        d = md.Delta.from_arrays({"x": [7]}, groups=[1], ngroups=K)
        u1, u2 = j1.apply(a=d), j1.apply(a=d)
        self.assertEqual((u1.identity, u1.step, u1.info["base"]), (u2.identity, 1, j1.identity))
        self.assertNotEqual(u1.identity, j1.apply(a=md.Delta.from_arrays({"x": [8]}, groups=[1], ngroups=K)).identity)
        self.assertEqual(u1.info["touched"], 1)
        for bad in (dict(keys=K + 1), dict(keys=2, groups_a=2)):     # side groups != keys x groups
            with self.assertRaises(DomainError):
                mbl.Join.build(sa, sb, pairs=[("x", "y")], **bad)
        with self.assertRaises(ValueError):
            mbl.Join.build(sa, sb, keys=K, pairs=[("x^2", "y")])      # not a term of side A
        with self.assertRaises(ValueError):
            mbl.Join.build(sa, sb, keys=K, pairs=[("x", "y")], null_key=K)
        nk = mbl.Join.build(sa, sb, keys=K, pairs=[("1", "1")], null_key=1)
        self.assertEqual(nk.cell()["rows"], 0)                        # the only shared key is the null key
        with self.assertRaises(DomainError):
            u1.apply(a=d, new_a=sa.apply(d))                          # sa is not u1's side A (it is j1's)
        vs = mbl.Values.from_scope(mbk.Scope.from_arrays({"x": [1, 2]}, groups=[0, 1], ngroups=K), "x")
        for bad in (md.Delta.from_arrays({"x": [1]}, groups=[0], ngroups=K + 1),
                    md.Delta.from_arrays({"x": [1]}, scales={"x": 2}, groups=[0], ngroups=K),
                    md.Delta.from_arrays({"x": [1], "y": [1]}, groups=[0], ngroups=K)):
            with self.assertRaises(DomainError):
                vs.apply(bad)
        with self.assertRaises(DomainError):
            j1.finalize(0, 0, "min")                                  # a join keeps per-pair sums only
        self.assertEqual(u1.finalize(0, 0, "sum"), (2 + 7) * 4)            # key 1 only: (2 + 7) * 4

    # ------------------------------------------------------------------ CUDA equals CPU on larger batches
    def test_delta002_cuda_equals_cpu_large_batches_scrambled(self):
        rng = np.random.default_rng(2109)
        K, GA, GB = 2000, 4, 2
        na, nb = 20000, 12000
        basis_a, basis_b = mbk.monomials(["x", "u"], 2), ["1", "y", "y^2"]
        sa = md.State.from_bank(mbk.build(mbk.Scope.from_arrays(
            {"x": rng.integers(-10**6, 10**6, na), "u": rng.integers(-99, 99, na)}, valid={"x": rng.random(na) < 0.9},
            groups=rng.integers(0, K * GA, na), ngroups=K * GA), basis_a), promise="signed")
        sb = md.State.from_bank(mbk.build(mbk.Scope.from_arrays(
            {"y": rng.integers(-10**5, 10**5, nb)}, valid={"y": rng.random(nb) < 0.95},
            groups=rng.integers(0, K * GB, nb), ngroups=K * GB), basis_b), promise="signed")
        pairs = [("1", "1"), ("x", "1"), ("x", "y"), ("x^2", "y^2"), ("x*u", "y"), ("u^2", "1")]
        j = self.build(sa, sb, keys=K, groups_a=GA, groups_b=GB, pairs=pairs, null_key=17)
        k = 3000
        da = md.Delta.from_arrays({"x": rng.integers(-10**6, 10**6, k), "u": rng.integers(-99, 99, k)},
                                  valid={"x": rng.random(k) < 0.9}, groups=rng.integers(0, K * GA, k), ngroups=K * GA,
                                  weights=rng.integers(-3, 4, k))
        db = md.Delta.from_arrays({"y": rng.integers(-10**5, 10**5, k)}, valid={"y": rng.random(k) < 0.95},
                                  groups=rng.integers(0, K * GB, k), ngroups=K * GB, weights=rng.integers(-3, 4, k))
        for kw in (dict(a=da), dict(b=db), dict(a=da, b=db)):
            nxt = self.both(j, **kw)
            rebuilt = mbl.Join.build(nxt.a, nxt.b, keys=K, groups_a=GA, groups_b=GB, pairs=pairs, null_key=17)
            self.assertTrue(rebuilt.equal(nxt))

    # ------------------------------------------------------------------ engine tables
    def test_delta002_engine_tables_join_matches_duckdb(self):
        rng = np.random.default_rng(2110)
        K, n, m = 12, 600, 300
        fa = pd.DataFrame({"k": np.r_[np.arange(K), rng.integers(0, K, n - K)], "g": rng.integers(0, 3, n),
                           "x": pd.Series([int(v) if rng.random() > 0.1 else None for v in rng.integers(-500, 500, n)], dtype=object)})
        fb = pd.DataFrame({"k": np.r_[np.arange(K), rng.integers(0, K, m - K)],
                           "y": pd.Series([f"{int(v) / 100:.2f}" for v in rng.integers(-9999, 9999, m)], dtype=object)})
        db = torbles.from_ddl("CREATE TABLE a (k BIGINT, g BIGINT, x BIGINT); CREATE TABLE b (k BIGINT, y DECIMAL(8,2))",
                              {"a": fa, "b": fb})
        ta, tb = db["a"], db["b"]
        sel = rng.random(n) < 0.7
        scope_a = mbk.Scope.from_table(ta, ["x"], group_by=["k", "g"], rows=sel)
        scope_b = mbk.Scope.from_table(tb, ["y"], group_by=["k"])
        self.assertEqual((scope_a.groups, scope_b.groups), (K * 3, K))
        sa = md.State.from_bank(mbk.build(scope_a, ["1", "x"]))
        sb = md.State.from_bank(mbk.build(scope_b, ["1", "y"]))
        pairs = [("1", "1"), ("x", "1"), ("x", "y")]
        j = self.build(sa, sb, keys=K, groups_a=3, pairs=pairs)

        def duck(selection):
            cur = fa.iloc[np.flatnonzero(selection)].copy()
            db.con.register("aa", cur)
            db.con.register("bb", fb)
            got = {}
            for r in db.con.execute(
                    "SELECT aa.g, COUNT(*), COUNT(aa.x), SUM(CAST(aa.x AS HUGEINT)), "
                    "SUM(CAST(aa.x AS HUGEINT) * CAST(ROUND(CAST(bb.y AS DECIMAL(8,2)) * 100) AS HUGEINT)) "
                    "FROM aa JOIN bb ON aa.k = bb.k GROUP BY aa.g").fetchall():
                got[int(r[0])] = (int(r[1]), int(r[2]), int(r[3] or 0), int(r[4] or 0))
            return got

        def ours(join):
            out = {}
            for g in range(3):
                c = join.cell(g, 0)
                if c["rows"]:
                    out[g] = (c["rows"], c["count"][1], c["sum"][1], c["sum"][2])
            return out
        self.assertEqual(ours(j), duck(sel))
        for step in range(3):                                         # selection changes on side A: table diffs
            new_sel = sel.copy()
            flip = rng.choice(n, 60, replace=False)
            new_sel[flip] = ~new_sel[flip]
            new_scope = mbk.Scope.from_table(ta, ["x"], group_by=["k", "g"], rows=new_sel)
            nxt = self.both(j, a=md.Delta.between(scope_a, new_scope))
            self.assertEqual(ours(nxt), duck(new_sel), step)
            self.assertTrue(nxt.a.bound_to(new_scope))
            j, sel, scope_a = nxt, new_sel, new_scope

    def test_delta002_opt_in_import(self):
        code = "import sys, torbles; print('torbles.math_bilinear' in sys.modules)"
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.split()
        self.assertEqual(out, ["False"])


if __name__ == "__main__":
    unittest.main()
